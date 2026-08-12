"""LLM provider adapters — เรียก REST API ตรงผ่าน httpx (ไม่พึ่ง SDK หนัก)

adapter เดียว (OpenAI-compatible) ครอบทั้ง OpenAI จริงและ endpoint local ทุกตัว
Gemini ใช้ REST ของ Google โดยตรง — Anthropic เตรียม interface ไว้ (เฟส 2)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

import httpx

from lmds.config.settings import ProviderConfig, ProviderName

OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
MINIMAX_BASE = "https://api.minimax.io/v1"

# error ชั่วคราว: rate limit / ฝั่ง provider ล่มชั่วคราว — retry คุ้ม
# ไม่รวม 400/401/403/404 (ผิดที่ config ของเรา retry ไปก็เหมือนเดิม)
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_HTTP_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 30.0


class ProviderError(Exception):
    pass


def _backoff_sleep(seconds: float) -> None:
    """แยกออกมาเป็นฟังก์ชันเพื่อให้เทส monkeypatch ได้โดยไม่ต้องรอจริง"""
    time.sleep(seconds)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """เคารพ Retry-After ถ้า provider บอกมา ไม่งั้น exponential backoff"""
    if response is not None:
        retry_after = (response.headers.get("Retry-After") or "").strip()
        if retry_after.isdigit():
            return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


def _post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    provider_name: str,
    max_attempts: int = MAX_HTTP_ATTEMPTS,
) -> httpx.Response:
    """POST พร้อม backoff สำหรับ error ชั่วคราว

    เน็ตกระตุกหรือโดน 429 ครั้งเดียวไม่ควรทำให้ทั้ง flow ตกไป rule-based —
    คืน response ตัวสุดท้ายให้ผู้เรียกตัดสิน (ผู้เรียกเป็นคนแปลง status เป็น ProviderError)
    """
    last_transport_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        response: httpx.Response | None = None
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            last_transport_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code not in RETRYABLE_STATUSES:
                return response
            last_transport_error = None

        if attempt == max_attempts:
            break
        _backoff_sleep(_retry_delay(response, attempt))

    if response is not None:
        return response
    raise ProviderError(
        f"{provider_name} เชื่อมต่อไม่ได้หลังลอง {max_attempts} ครั้ง — {last_transport_error}"
    )


class MissingKey(ProviderError):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"ไม่พบ API key ของ {provider} — ตั้งด้วย: lmds config set-key {provider} "
            f"(หรือใช้ --no-llm สำหรับ rule-based mode)"
        )


def _mentions_response_format(response: httpx.Response) -> bool:
    """400 ที่บ่นถึง response_format/json_object เท่านั้นถึงจะ retry แบบตัด field ออก

    ไม่ใช่ 400 ทุกตัว — 400 จากสาเหตุอื่น (model ไม่มี, payload ผิด) ยิงซ้ำก็เหมือนเดิม
    """
    try:
        body = response.text[:2000].lower()
    except Exception:
        return False
    return "response_format" in body or "json_object" in body or "json mode" in body


class LlmProvider(ABC):
    name: str = ""
    model: str = ""

    @abstractmethod
    def complete_json(self, system: str, user: str) -> str:
        """เรียก LLM ขอคำตอบเป็น JSON string — ผู้เรียกเป็นคน parse/validate เอง"""

    def complete_chat(self, system: str, messages: list[dict]) -> str:
        """คุยแบบข้อความธรรมดาหลาย turn — ไม่บังคับ JSON

        ค่าตั้งต้นใช้ complete_json() ไม่ได้ เพราะ JSON mode จะได้ object กลับมา
        ไม่ใช่ประโยคที่คนอ่าน · provider ตัวไหนยังไม่รองรับก็บอกไปตรง ๆ ดีกว่า
        ให้กล่องแชทได้ก้อน JSON แปลก ๆ ไปแสดง
        """
        raise ProviderError(f"{self.name} ยังไม่รองรับโหมดแชท")

    def stream_chat(self, system: str, messages: list[dict]):
        """สตรีมทีละชิ้น — ตัวที่สตรีมไม่ได้ก็ส่งก้อนเดียวจบ

        ผู้เรียกจึงเขียนทางเดียวได้ ไม่ต้องแยกว่า provider ไหนสตรีมได้
        """
        yield self.complete_chat(system, messages)


class OpenAiCompatProvider(LlmProvider):
    """ใช้ได้ทั้ง api.openai.com และทุก endpoint ที่พูด /chat/completions"""

    def __init__(self, name: str, model: str, api_key: str | None, base_url: str | None = None,
                 client: httpx.Client | None = None):
        self.name = name
        self.model = model
        self._key = api_key or ""  # endpoint local (Ollama/vLLM) มักไม่ต้องใช้ key
        self._base = (base_url or OPENAI_BASE).rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        url = f"{self._base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        resp = _post_with_retry(
            self._client, url, headers=headers, payload=payload, provider_name=self.name
        )

        # engine local รุ่นเก่า (vLLM/llama.cpp server/LM Studio บางเวอร์ชัน) ไม่รู้จัก
        # response_format แล้วตอบ 400 ทั้งคำขอ — ลองใหม่โดยตัด field นี้ออก
        # prompt บังคับ JSON อยู่แล้ว และ orchestrator validate ด้วย schema + retry อีกชั้น
        if resp.status_code == 400 and _mentions_response_format(resp):
            payload.pop("response_format", None)
            resp = _post_with_retry(
                self._client, url, headers=headers, payload=payload, provider_name=self.name
            )

        if resp.status_code != 200:
            raise ProviderError(f"{self.name} ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ {self.name} ผิดปกติ: {exc}")


    def complete_chat(self, system: str, messages: list[dict]) -> str:
        resp = _post_with_retry(
            self._client,
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"} if self._key else {},
            payload=self._chat_payload(system, messages, stream=False),
            provider_name=self.name,
        )
        if resp.status_code != 200:
            raise ProviderError(f"{self.name} ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ {self.name} ผิดปกติ: {exc}")

    def stream_chat(self, system: str, messages: list[dict]):
        """สตรีมจริงผ่าน SSE — คำตอบยาว ๆ จะได้ทยอยขึ้นแทนที่จะเงียบไป 30 วินาที

        ไม่ใช้ _post_with_retry เพราะ retry กลาง stream แปลว่าผู้ใช้เห็นคำตอบซ้ำ
        ต่อไม่ติดตั้งแต่แรกค่อยตกไปเป็นแบบไม่สตรีม (ผู้เรียกจับ ProviderError)
        """
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        payload = self._chat_payload(system, messages, stream=True)
        try:
            with self._client.stream(
                "POST", f"{self._base}/chat/completions", headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise ProviderError(
                        f"{self.name} ตอบ HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"]
                    except (KeyError, IndexError, ValueError):
                        continue  # keepalive หรือ chunk ที่ไม่มี delta — ข้ามไป
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} สตรีมไม่สำเร็จ: {exc}") from exc

    def _chat_payload(self, system: str, messages: list[dict], *, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.3,
            "max_tokens": 1500,
            "stream": stream,
        }


class GeminiProvider(LlmProvider):
    def __init__(self, model: str, api_key: str, client: httpx.Client | None = None):
        self.name = "gemini"
        self.model = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        resp = _post_with_retry(
            self._client,
            f"{GEMINI_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._key},
            payload={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            },
            provider_name=self.name,
        )
        if resp.status_code != 200:
            raise ProviderError(f"gemini ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ gemini ผิดปกติ: {exc}")


    def complete_chat(self, system: str, messages: list[dict]) -> str:
        resp = _post_with_retry(
            self._client,
            f"{GEMINI_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._key},
            payload={
                "systemInstruction": {"parts": [{"text": system}]},
                # Gemini เรียก assistant ว่า "model" — role อื่นจะโดนปฏิเสธทั้งคำขอ
                "contents": [
                    {"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]}
                    for m in messages
                ],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500},
            },
            provider_name=self.name,
        )
        if resp.status_code != 200:
            raise ProviderError(f"gemini ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ gemini ผิดปกติ: {exc}")


class MiniMaxProvider(LlmProvider):
    """MiniMax cloud API (chatcompletion_v2) — โครง request/response ใกล้ OpenAI แต่ path ต่างกัน"""

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 client: httpx.Client | None = None):
        self.name = "minimax"
        self.model = model
        self._key = api_key
        self._base = (base_url or MINIMAX_BASE).rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        resp = _post_with_retry(
            self._client,
            f"{self._base}/text/chatcompletion_v2",
            headers={"Authorization": f"Bearer {self._key}"},
            payload={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            provider_name=self.name,
        )
        if resp.status_code != 200:
            raise ProviderError(f"minimax ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code") not in (None, 0):
            raise ProviderError(f"minimax error: {base_resp.get('status_msg', 'unknown')}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ minimax ผิดปกติ: {exc}")


    def complete_chat(self, system: str, messages: list[dict]) -> str:
        resp = _post_with_retry(
            self._client,
            f"{self._base}/text/chatcompletion_v2",
            headers={"Authorization": f"Bearer {self._key}"},
            payload={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "temperature": 0.3,
                "max_tokens": 1500,
            },
            provider_name=self.name,
        )
        if resp.status_code != 200:
            raise ProviderError(f"minimax ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code") not in (None, 0):
            raise ProviderError(f"minimax error: {base_resp.get('status_msg', 'unknown')}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ minimax ผิดปกติ: {exc}")


def make_provider(config: ProviderConfig, api_key: str | None,
                  client: httpx.Client | None = None) -> LlmProvider:
    """สร้าง provider จาก config + key

    openai-compat (Ollama/vLLM/endpoint local) ใช้ได้โดยไม่มี key — provider อื่นต้องมี
    """
    if config.name is ProviderName.OPENAI_COMPAT:
        return OpenAiCompatProvider(
            config.name.value, config.model, api_key, base_url=config.base_url, client=client
        )
    if not api_key:
        raise MissingKey(config.name.value)
    if config.name is ProviderName.GEMINI:
        return GeminiProvider(config.model, api_key, client=client)
    if config.name is ProviderName.MINIMAX:
        return MiniMaxProvider(config.model, api_key, base_url=config.base_url, client=client)
    if config.name is ProviderName.ANTHROPIC:
        raise ProviderError("Anthropic adapter อยู่ใน roadmap เฟส 2 — ใช้ openai/gemini/openai-compat ก่อน")
    return OpenAiCompatProvider(config.name.value, config.model, api_key, client=client)


# ── รายชื่อโมเดลที่ key นี้ใช้ได้จริง ────────────────────────────────────────────
# ผู้ใช้ที่ไม่ได้อยู่กับ provider นั้นทุกวันไม่มีทางรู้ชื่อโมเดล — พิมพ์ผิดตัวเดียวแล้วรู้ตอน
# deploy ล้มกลางทาง · ถามจาก provider ตรง ๆ ได้ ก็ควรถาม
ANTHROPIC_BASE = "https://api.anthropic.com/v1"

_LIST_TIMEOUT = 15.0


def list_models(name: ProviderName, api_key: str, base_url: str | None = None) -> list[str]:
    """ถาม provider ว่า key นี้ใช้โมเดลอะไรได้บ้าง — คืนรายชื่อเรียงแล้ว

    รองรับทุกตัวที่มี endpoint รายชื่อ · ตัวไหนไม่มีก็คืนลิสต์ว่าง แล้วให้ผู้ใช้พิมพ์เอง
    (ว่างเปล่าดีกว่ารายชื่อที่เดาขึ้นมาเอง — ผู้ใช้จะเลือกตัวที่ไม่มีอยู่จริง)
    """
    name = ProviderName(name)
    base = (base_url or "").rstrip("/")
    try:
        with httpx.Client(timeout=_LIST_TIMEOUT) as client:
            if name is ProviderName.GEMINI:
                resp = client.get(f"{GEMINI_BASE}/models", params={"key": api_key})
                resp.raise_for_status()
                return sorted(
                    m["name"].split("/", 1)[-1]
                    for m in resp.json().get("models", [])
                    if "generateContent" in (m.get("supportedGenerationMethods") or [])
                )
            if name is ProviderName.ANTHROPIC:
                resp = client.get(f"{ANTHROPIC_BASE}/models", headers={
                    "x-api-key": api_key, "anthropic-version": "2023-06-01"})
                resp.raise_for_status()
                return sorted(m["id"] for m in resp.json().get("data", []))

            # ที่เหลือพูด /v1/models แบบ OpenAI — รวม vLLM, Ollama, LocalAI, Bifrost
            root = base or {ProviderName.OPENAI: OPENAI_BASE,
                            ProviderName.MINIMAX: MINIMAX_BASE}.get(name, "")
            if not root:
                return []
            resp = client.get(f"{root}/models",
                              headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") if isinstance(data, dict) else data
            return sorted(str(m.get("id") or m.get("name") or "") for m in (rows or []) if m)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            raise ProviderError("key ใช้ไม่ได้ (ถูกปฏิเสธ) — ตรวจว่า copy มาครบไหม") from exc
        if code == 404:
            raise ProviderError("ปลายทางไม่มี /v1/models — พิมพ์ชื่อโมเดลเองได้เลย") from exc
        raise ProviderError(f"ขอรายชื่อโมเดลไม่สำเร็จ (HTTP {code})") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"ต่อไปที่ provider ไม่ได้: {exc}") from exc
