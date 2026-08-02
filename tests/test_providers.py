import json

import httpx
import pytest

from lmds.brain import providers
from lmds.brain import (
    GeminiProvider,
    MiniMaxProvider,
    MissingKey,
    OpenAiCompatProvider,
    ProviderError,
    make_provider,
)
from lmds.config.settings import ProviderConfig, ProviderName


def test_openai_compat_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-test123456789012345",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = provider.complete_json("SYSTEM", "USER")
    assert out == '{"ok": true}'
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test123456789012345"
    assert seen["body"]["model"] == "gpt-4.1"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_openai_compat_custom_base_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    provider = OpenAiCompatProvider(
        "openai-compat", "qwen3-coder", "k-123456789012",
        base_url="http://10.100.152.1:8000/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.complete_json("s", "u")
    assert seen["url"] == "http://10.100.152.1:8000/v1/chat/completions"


@pytest.fixture
def no_sleep(monkeypatch):
    """เก็บเวลาที่ควรจะ sleep ไว้ตรวจ แต่ไม่รอจริงตอนเทส"""
    slept: list[float] = []
    monkeypatch.setattr(providers, "_backoff_sleep", slept.append)
    return slept


def test_openai_compat_http_error_raises_after_retries(no_sleep):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="rate limited")

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-x123456789012",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="429"):
        provider.complete_json("s", "u")
    assert len(calls) == providers.MAX_HTTP_ATTEMPTS  # 429 = ชั่วคราว → ต้อง retry ก่อนยอมแพ้
    assert no_sleep == [2.0, 4.0]                     # exponential backoff


def test_retry_recovers_from_transient_503(no_sleep):
    """503 ครั้งเดียวไม่ควรทำให้ทั้ง flow ตกไป rule-based"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-x123456789012",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.complete_json("s", "u") == '{"ok": 1}'
    assert len(calls) == 2


def test_retry_honours_retry_after_header(no_sleep):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "7"})

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-x123456789012",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError):
        provider.complete_json("s", "u")
    assert no_sleep == [7.0, 7.0]


def test_no_retry_on_permanent_error(no_sleep):
    """401 = key ผิด — retry ไปก็เหมือนเดิม ต้องเด้งทันที"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="invalid api key")

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-bad12345678901",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="401"):
        provider.complete_json("s", "u")
    assert len(calls) == 1
    assert no_sleep == []


def test_retry_on_transport_error_then_raises(no_sleep):
    """เน็ตหลุดทุกครั้ง → ProviderError ที่บอกว่าลองไปกี่ครั้ง"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection refused")

    provider = OpenAiCompatProvider(
        "openai-compat", "qwen3", None, base_url="http://10.0.0.9:8000/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="เชื่อมต่อไม่ได้"):
        provider.complete_json("s", "u")
    assert len(calls) == providers.MAX_HTTP_ATTEMPTS


def test_gemini_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": '{"plan": 1}'}]}}]
        })

    provider = GeminiProvider(
        "gemini-2.5-pro", "AIzaTestKey1234567890123456789012345",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = provider.complete_json("SYS", "USR")
    assert out == '{"plan": 1}'
    assert "gemini-2.5-pro:generateContent" in seen["url"]
    assert seen["key"].startswith("AIza")
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"


def test_minimax_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "choices": [{"message": {"role": "assistant", "content": '{"plan": true}'}}],
        })

    provider = MiniMaxProvider(
        "MiniMax-M2", "mmkey-1234567890abcdef",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = provider.complete_json("SYS", "USR")
    assert out == '{"plan": true}'
    assert seen["url"] == "https://api.minimax.io/v1/text/chatcompletion_v2"
    assert seen["auth"] == "Bearer mmkey-1234567890abcdef"
    assert seen["body"]["model"] == "MiniMax-M2"
    assert seen["body"]["messages"][0]["role"] == "system"


def test_minimax_base_resp_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
        })

    provider = MiniMaxProvider(
        "MiniMax-M2", "badkey-123456789012",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="invalid api key"):
        provider.complete_json("s", "u")


def test_make_provider_minimax_dispatch():
    provider = make_provider(ProviderConfig(name=ProviderName.MINIMAX, model="MiniMax-M2"), "k-1234")
    assert isinstance(provider, MiniMaxProvider)
    with pytest.raises(MissingKey, match="set-key minimax"):
        make_provider(ProviderConfig(name=ProviderName.MINIMAX, model="MiniMax-M2"), None)


def test_make_provider_missing_key():
    config = ProviderConfig(name=ProviderName.OPENAI, model="gpt-4.1")
    with pytest.raises(MissingKey, match="set-key openai"):
        make_provider(config, api_key=None)


def test_make_provider_dispatch():
    openai = make_provider(ProviderConfig(name=ProviderName.OPENAI, model="gpt-4.1"), "k-123")
    assert isinstance(openai, OpenAiCompatProvider)

    gemini = make_provider(ProviderConfig(name=ProviderName.GEMINI, model="gemini-2.5-pro"), "k-123")
    assert isinstance(gemini, GeminiProvider)

    compat = make_provider(
        ProviderConfig(name=ProviderName.OPENAI_COMPAT, model="qwen", base_url="http://x:8000/v1"), "k-123"
    )
    assert isinstance(compat, OpenAiCompatProvider)


def test_openai_compat_keyless_local_endpoint():
    """Ollama/vLLM local ไม่มี key — ต้องใช้ได้และไม่ส่ง Authorization header"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    config = ProviderConfig(
        name=ProviderName.OPENAI_COMPAT, model="gpt-oss:20b", base_url="http://10.10.10.1:11434/v1"
    )
    provider = make_provider(config, api_key=None, client=httpx.Client(transport=httpx.MockTransport(handler)))
    provider.complete_json("s", "u")
    assert seen["auth"] is None  # ไม่มี key → ไม่ส่ง header


def test_openai_real_still_requires_key():
    with pytest.raises(MissingKey):
        make_provider(ProviderConfig(name=ProviderName.OPENAI, model="gpt-4.1"), api_key=None)


def test_make_provider_anthropic_phase2():
    config = ProviderConfig(name=ProviderName.ANTHROPIC, model="claude-sonnet-5")
    with pytest.raises(ProviderError, match="เฟส 2"):
        make_provider(config, "k-123")
