import json

import httpx
import pytest

from lmds.brain import GeminiProvider, MissingKey, OpenAiCompatProvider, ProviderError, make_provider
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


def test_openai_compat_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    provider = OpenAiCompatProvider(
        "openai", "gpt-4.1", "sk-x123456789012",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="429"):
        provider.complete_json("s", "u")


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


def test_make_provider_anthropic_phase2():
    config = ProviderConfig(name=ProviderName.ANTHROPIC, model="claude-sonnet-5")
    with pytest.raises(ProviderError, match="เฟส 2"):
        make_provider(config, "k-123")
