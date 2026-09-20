"""LLM provider adapters — เรียก REST API ตรงผ่าน httpx (ไม่พึ่ง SDK หนัก)

adapter เดียว (OpenAI-compatible) ครอบทั้ง OpenAI จริงและ endpoint local ทุกตัว
Gemini ใช้ REST ของ Google โดยตรง · Anthropic ใช้ Messages API ซึ่งคนละโครงกับ
/chat/completions พอสมควร (system แยกออกจาก messages, max_tokens เป็นฟิลด์บังคับ)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

import httpx

from lmds.config.settings import DEFAULT_MODELS, ProviderConfig, ProviderName

OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
MINIMAX_BASE = "https://api.minimax.io/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"

# error ชั่วคราว: rate limit / ฝั่ง provider ล่มชั่วคราว — retry คุ้ม
# ไม่รวม 400/401/403/404 (ผิดที่ config ของเรา retry ไปก็เหมือนเดิม)
# 529 เป็นของ Anthropic ตัวเดียว (overloaded_error = ฝั่งเขารับงานไม่ไหว ไม่ใช่เราส่งผิด)
# ตัวอื่นไม่เคยคืน status นี้ อยู่ในลิสต์รวมจึงไม่กระทบใคร
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
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
            raise ProviderError(f"รูปแบบคำตอบของ {self.name} ผิดปกติ: {exc}") from exc


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
            raise ProviderError(f"รูปแบบคำตอบของ {self.name} ผิดปกติ: {exc}") from exc

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
            raise ProviderError(f"รูปแบบคำตอบของ gemini ผิดปกติ: {exc}") from exc


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
            raise ProviderError(f"รูปแบบคำตอบของ gemini ผิดปกติ: {exc}") from exc


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
            raise ProviderError(f"รูปแบบคำตอบของ minimax ผิดปกติ: {exc}") from exc


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
            raise ProviderError(f"รูปแบบคำตอบของ minimax ผิดปกติ: {exc}") from exc


# ── Anthropic (Messages API) ──────────────────────────────────────────────────
# ทุกคำขอต้องมี header เวอร์ชัน ไม่งั้นโดน 400 ตั้งแต่ก่อนดู payload
ANTHROPIC_VERSION = "2023-06-01"

# งานของสมองตัวนี้คือวางแผน deploy: อ่าน fit report แล้วตอบ JSON ตาม schema —
# ต้องการ reasoning มากกว่าความเร็ว และยิงครั้งเดียวต่อ deploy หนึ่งครั้ง ไม่ใช่งานปริมาณมาก
# ผู้ใช้ทับได้ด้วย `--model` ตอน set-provider (ดูชื่อที่ key ใช้ได้จริงจาก list_models())
# ค่านี้ได้ใช้เมื่อ config ไม่ได้ระบุรุ่นมาเท่านั้น — ปกติ set-provider เติมจาก DEFAULT_MODELS
# ให้ก่อนแล้ว · **ดึงจากที่นั่นตรง ๆ ไม่เขียนชื่อรุ่นซ้ำ**: ค่าคงที่สองตัวที่ "ต้องเป็นรุ่นเดียวกัน"
# จะแยกจากกันวันหนึ่งเสมอ แล้วผู้ใช้ที่แก้ config.yaml เองจะได้คนละรุ่นกับที่ set-provider เขียนให้
# โดยไม่มีอะไรบอก (เกิดขึ้นแล้วตอนเพิ่ม adapter ตัวนี้: opus-5 กับ sonnet-5)
ANTHROPIC_DEFAULT_MODEL = DEFAULT_MODELS[ProviderName.ANTHROPIC]

# Anthropic บังคับ max_tokens เสมอ (ไม่มี default ฝั่ง server แบบ OpenAI) และรุ่น 4.6 ขึ้นไป
# เปิด adaptive thinking มาให้เอง โดย token ที่ใช้คิดถูกนับรวมในเพดานนี้ด้วย — ตั้งแคบแบบ
# 1500 ที่ provider อื่นใช้กับแชท จะหมดไปกับการคิดจนได้ content ว่างกลับมา
# พร้อม stop_reason="max_tokens" · แผน deploy เป็น JSON ยาว จึงให้ที่มากกว่าฝั่งแชท
ANTHROPIC_MAX_TOKENS = 8192
ANTHROPIC_CHAT_MAX_TOKENS = 4096


class AnthropicProvider(LlmProvider):
    """Claude ผ่าน Messages API

    ลอกโครง OpenAI มาตรง ๆ ไม่ได้อยู่ 4 จุด: system เป็นฟิลด์บนสุดไม่ใช่ message,
    max_tokens บังคับ, คำตอบเป็น list ของ block หลายชนิดไม่ใช่ message.content ก้อนเดียว,
    และรุ่น 4.6 ขึ้นไปไม่รับ temperature/top_p อีกแล้ว (ส่งไป = 400 ทั้งคำขอ)
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 client: httpx.Client | None = None):
        self.name = "anthropic"
        self.model = model or ANTHROPIC_DEFAULT_MODEL
        self._key = api_key
        self._base = (base_url or ANTHROPIC_BASE).rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        # ไม่มี response_format/JSON mode แบบ OpenAI ให้เปิด — ความเป็น JSON มาจาก prompt
        # และ orchestrator ที่ validate ด้วย schema + retry อยู่แล้ว (เหมือนทาง minimax)
        resp = _post_with_retry(
            self._client,
            f"{self._base}/messages",
            headers=self._headers(),
            payload=self._payload(system, [{"role": "user", "content": user}],
                                  max_tokens=ANTHROPIC_MAX_TOKENS),
            provider_name=self.name,
        )
        return self._text_or_raise(resp)

    def complete_chat(self, system: str, messages: list[dict]) -> str:
        resp = _post_with_retry(
            self._client,
            f"{self._base}/messages",
            headers=self._headers(),
            payload=self._payload(system, self._chat_messages(messages),
                                  max_tokens=ANTHROPIC_CHAT_MAX_TOKENS),
            provider_name=self.name,
        )
        # แชทที่ถูกตัดกลางคันยังอ่านรู้เรื่อง — โชว์เท่าที่ได้ดีกว่าขึ้น error แล้วไม่ได้อะไรเลย
        return self._text_or_raise(resp, allow_truncated=True)

    def stream_chat(self, system: str, messages: list[dict]):
        """สตรีมผ่าน SSE — เหตุผลเดียวกับฝั่ง OpenAI-compat: คำตอบยาวต้องทยอยขึ้น

        ไม่ผ่าน _post_with_retry เพราะ retry กลาง stream แปลว่าผู้ใช้เห็นคำตอบซ้ำ
        ต่อไม่ติดตั้งแต่แรกค่อยตกไปเป็นแบบไม่สตรีม (ผู้เรียกจับ ProviderError)
        """
        payload = self._payload(system, self._chat_messages(messages),
                                max_tokens=ANTHROPIC_CHAT_MAX_TOKENS)
        payload["stream"] = True
        try:
            with self._client.stream(
                "POST", f"{self._base}/messages", headers=self._headers(), json=payload
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise self._http_error(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue  # บรรทัด event:/ว่างของ SSE — ข้อมูลจริงอยู่บรรทัด data
                    try:
                        event = json.loads(line[5:].strip())
                    except ValueError:
                        continue  # ping หรือบรรทัดที่ไม่ใช่ JSON — ข้ามไป
                    # ล้มกลางสตรีมยังเป็น HTTP 200 (ต่อสายไปแล้ว) — error มาเป็น event
                    # ไม่เช็คตรงนี้คำตอบจะหยุดกลางประโยคเฉย ๆ โดยไม่มีใครบอกว่าทำไม
                    if event.get("type") == "error":
                        detail = (event.get("error") or {}).get("message") or "ไม่ทราบสาเหตุ"
                        raise ProviderError(f"anthropic หยุดกลางคำตอบ: {detail}")
                    # event มีหลายชนิด (message_start / content_block_stop / message_delta)
                    # เอาเฉพาะ text_delta · thinking_delta เป็นความคิด ไม่ใช่คำตอบที่จะโชว์
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic สตรีมไม่สำเร็จ: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        # ไม่ใช่ Bearer — Anthropic ใช้ x-api-key ของมันเอง
        return {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION}

    def _payload(self, system: str, messages: list[dict], *, max_tokens: int) -> dict:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        # system เป็นฟิลด์บนสุด ไม่ใช่ role ใน messages — ใส่เป็น role จะโดนปฏิเสธทั้งคำขอ
        # ส่งตอนว่างก็ไม่ผ่านเหมือนกัน (ห้าม text block ว่าง) จึงใส่เฉพาะตอนมีค่า
        if system:
            payload["system"] = system
        # ไม่ส่ง temperature: รุ่น 4.6 ขึ้นไปถอด sampling parameter ออกแล้ว ส่งไปได้ 400
        # (ค่าตั้งต้นเราเป็นรุ่นใหม่ และผู้ใช้ทับเป็นรุ่นเก่าได้ — ไม่ส่งเลยใช้ได้ทั้งสองทาง)
        return payload

    def _chat_messages(self, messages: list[dict]) -> list[dict]:
        """ปรับประวัติแชทให้ตรงกติกาของ Anthropic ก่อนส่ง

        สามข้อที่ทำให้ทั้งคำขอถูกปฏิเสธถ้าปล่อยผ่าน: content ว่าง (ห้าม text block ว่าง),
        role อื่นนอกจาก user/assistant, และข้อความแรกต้องเป็น user เสมอ
        ข้อสุดท้ายเกิดจริงกับกล่องแชท — build_messages() ตัดท้ายด้วย history[-MAX_TURNS:]
        ซึ่งตัดมาลงตรงคำตอบของผู้ช่วยได้ตามปกติ (กติกาคนละแบบกับ Gemini ที่ต้อง map
        assistant → model แต่ทำที่จุดเดียวกัน)
        """
        cleaned = [
            {"role": "assistant" if m.get("role") == "assistant" else "user",
             "content": str(m.get("content") or "")}
            for m in messages
            if str(m.get("content") or "").strip()
        ]
        while cleaned and cleaned[0]["role"] != "user":
            cleaned.pop(0)
        return cleaned

    def _text_or_raise(self, resp: httpx.Response, *, allow_truncated: bool = False) -> str:
        if resp.status_code != 200:
            raise self._http_error(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"รูปแบบคำตอบของ anthropic ผิดปกติ: {exc}") from exc

        stop = data.get("stop_reason")
        # การปฏิเสธด้วยเหตุผลความปลอดภัยยังเป็น HTTP 200 — ไม่เช็คตรงนี้ผู้ใช้จะเห็นแค่
        # "คำตอบว่าง" แล้วไปไล่หาที่ key กับเน็ตแทน ทั้งที่ต้องแก้ที่คำถาม
        if stop == "refusal":
            raise ProviderError(
                "anthropic ปฏิเสธคำขอนี้ (stop_reason=refusal) — ลองเรียบเรียงคำถามใหม่ "
                "หรือใช้ provider อื่น/--no-llm สำหรับ rule-based mode"
            )
        # คำตอบเป็น list ของ block และรุ่นใหม่เปิด adaptive thinking มาให้เอง — block แรก
        # จึงเป็น thinking (ข้อความว่างเมื่อไม่ได้ขอให้แสดง) ไม่ใช่คำตอบ · หยิบ content[0]
        # ตรง ๆ แบบ OpenAI จะได้ string ว่างทั้งที่คำตอบอยู่ block ถัดไป
        text = "".join(
            block.get("text") or ""
            for block in (data.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if stop == "max_tokens" and not (allow_truncated and text):
            raise ProviderError(
                "anthropic ตอบไม่จบ — ชนเพดาน max_tokens (stop_reason=max_tokens) "
                "คำตอบถูกตัดกลางคันจึงใช้ไม่ได้ · ลองใหม่อีกครั้ง หรือลดขนาดข้อมูลที่ส่งไป"
            )
        if not text:
            raise ProviderError(
                f"รูปแบบคำตอบของ anthropic ผิดปกติ: ไม่มี block ชนิด text (stop_reason={stop})"
            )
        return text

    def _http_error(self, resp: httpx.Response) -> ProviderError:
        """แปลง status เป็นข้อความที่บอกว่าต้องไปแก้อะไร

        Anthropic ส่งเหตุผลจริงมาใน error.message (เช่นชื่อโมเดลไหนไม่มี) แต่ตัว status
        อย่างเดียวไม่พอให้ผู้ใช้รู้ว่าต้องแก้ที่ key, ที่ชื่อโมเดล หรือแค่รอ — เคสที่เจอบ่อย
        คือ 401 กับ 400 แล้วไปนั่งไล่ผิดที่
        """
        hints = {
            400: "ชื่อโมเดลไม่มีจริงหรือพารามิเตอร์ผิด — ตรวจชื่อรุ่นที่ตั้งไว้ "
                 f"(ต้องเป็น id เต็ม เช่น {ANTHROPIC_DEFAULT_MODEL})",
            401: "key ไม่ถูกต้องหรือหมดอายุ — ตั้งใหม่ด้วย: lmds config set-key anthropic",
            403: "key นี้ไม่มีสิทธิ์เรียกโมเดลนี้ — ตรวจสิทธิ์/workspace ของ key",
            404: "ไม่พบปลายทางหรือโมเดลนี้ — ตรวจชื่อโมเดลและ base URL",
            413: "prompt ใหญ่เกินที่ API รับ — ลดข้อมูลที่ส่งไป",
            429: "ยิงถี่เกินโควตา (rate limit) — รอแล้วลองใหม่ หรือลดความถี่/เพิ่มโควตาในบัญชี",
            529: "ฝั่ง Anthropic รับงานไม่ไหวชั่วคราว — ลองใหม่อีกครั้ง",
        }
        hint = hints.get(resp.status_code, "")
        detail = self._error_detail(resp)
        message = f"anthropic ตอบ HTTP {resp.status_code}: {detail}"
        return ProviderError(f"{message} — {hint}" if hint else message)

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        """เอา error.message มาแทน body ดิบ — body ดิบ 300 ตัวแรกมักหมดไปกับ wrapper JSON"""
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:300]
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        return resp.text[:300]


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
        return AnthropicProvider(config.model, api_key, base_url=config.base_url, client=client)
    return OpenAiCompatProvider(config.name.value, config.model, api_key, client=client)


# ── รายชื่อโมเดลที่ key นี้ใช้ได้จริง ────────────────────────────────────────────
# ผู้ใช้ที่ไม่ได้อยู่กับ provider นั้นทุกวันไม่มีทางรู้ชื่อโมเดล — พิมพ์ผิดตัวเดียวแล้วรู้ตอน
# deploy ล้มกลางทาง · ถามจาก provider ตรง ๆ ได้ ก็ควรถาม
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
                    "x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION})
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
