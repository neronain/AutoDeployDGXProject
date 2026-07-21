"""LLM provider adapters — เรียก REST API ตรงผ่าน httpx (ไม่พึ่ง SDK หนัก)

adapter เดียว (OpenAI-compatible) ครอบทั้ง OpenAI จริงและ endpoint local ทุกตัว
Gemini ใช้ REST ของ Google โดยตรง — Anthropic เตรียม interface ไว้ (เฟส 2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from lmds.config.settings import ProviderConfig, ProviderName

OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class ProviderError(Exception):
    pass


class MissingKey(ProviderError):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"ไม่พบ API key ของ {provider} — ตั้งด้วย: lmds config set-key {provider} "
            f"(หรือใช้ --no-llm สำหรับ rule-based mode)"
        )


class LlmProvider(ABC):
    name: str = ""
    model: str = ""

    @abstractmethod
    def complete_json(self, system: str, user: str) -> str:
        """เรียก LLM ขอคำตอบเป็น JSON string — ผู้เรียกเป็นคน parse/validate เอง"""


class OpenAiCompatProvider(LlmProvider):
    """ใช้ได้ทั้ง api.openai.com และทุก endpoint ที่พูด /chat/completions"""

    def __init__(self, name: str, model: str, api_key: str, base_url: str | None = None,
                 client: httpx.Client | None = None):
        self.name = name
        self.model = model
        self._key = api_key
        self._base = (base_url or OPENAI_BASE).rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        resp = self._client.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        )
        if resp.status_code != 200:
            raise ProviderError(f"{self.name} ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ {self.name} ผิดปกติ: {exc}")


class GeminiProvider(LlmProvider):
    def __init__(self, model: str, api_key: str, client: httpx.Client | None = None):
        self.name = "gemini"
        self.model = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=120.0)

    def complete_json(self, system: str, user: str) -> str:
        resp = self._client.post(
            f"{GEMINI_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            },
        )
        if resp.status_code != 200:
            raise ProviderError(f"gemini ตอบ HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"รูปแบบคำตอบของ gemini ผิดปกติ: {exc}")


def make_provider(config: ProviderConfig, api_key: str | None,
                  client: httpx.Client | None = None) -> LlmProvider:
    """สร้าง provider จาก config + key — MissingKey ถ้าไม่มี key"""
    if not api_key:
        raise MissingKey(config.name.value)
    if config.name is ProviderName.GEMINI:
        return GeminiProvider(config.model, api_key, client=client)
    if config.name is ProviderName.ANTHROPIC:
        raise ProviderError("Anthropic adapter อยู่ใน roadmap เฟส 2 — ใช้ openai/gemini/openai-compat ก่อน")
    base_url = config.base_url if config.name is ProviderName.OPENAI_COMPAT else None
    return OpenAiCompatProvider(config.name.value, config.model, api_key, base_url=base_url, client=client)
