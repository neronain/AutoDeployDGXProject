"""Anthropic adapter (Messages API)

โครงคนละแบบกับ /chat/completions ทุกจุดที่ต่างจึงมีเทสคุมไว้: system เป็นฟิลด์บนสุด,
max_tokens บังคับ, ไม่ส่ง temperature (รุ่น 4.6+ ถอดออกแล้ว), คำตอบเป็น list ของ block
หลายชนิด และ error ต้องบอกว่าไปแก้ที่ไหน ไม่ใช่โยน status เปล่า ๆ

ไม่ยิง API จริงสักตัว — httpx.MockTransport ทั้งไฟล์
"""

import json

import httpx
import pytest

from lmds.brain import providers
from lmds.brain.providers import (
    ANTHROPIC_DEFAULT_MODEL,
    AnthropicProvider,
    MissingKey,
    ProviderError,
    make_provider,
)
from lmds.config.settings import ProviderConfig, ProviderName

KEY = "sk-ant-test1234567890"


def _provider(handler, **kwargs) -> AnthropicProvider:
    return AnthropicProvider(
        kwargs.pop("model", "claude-opus-5"),
        kwargs.pop("api_key", KEY),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _text_reply(text: str, **extra) -> dict:
    body = {
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }
    body.update(extra)
    return body


@pytest.fixture
def no_sleep(monkeypatch):
    """เก็บเวลาที่ควรจะ sleep ไว้ตรวจ แต่ไม่รอจริงตอนเทส"""
    slept: list[float] = []
    monkeypatch.setattr(providers, "_backoff_sleep", slept.append)
    return slept


# ── รูปร่างคำขอ ────────────────────────────────────────────────────────────────

def test_request_shape_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply('{"plan": 1}'))

    assert _provider(handler).complete_json("SYS", "USR") == '{"plan": 1}'
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["key"] == KEY
    assert seen["auth"] is None  # ไม่ใช่ Bearer — Anthropic ใช้ x-api-key ของมันเอง
    assert seen["version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-opus-5"
    assert seen["body"]["max_tokens"] == providers.ANTHROPIC_MAX_TOKENS
    assert seen["body"]["messages"] == [{"role": "user", "content": "USR"}]


def test_system_is_a_top_level_field_not_a_message():
    """ใส่ system เป็น role ใน messages = โดนปฏิเสธทั้งคำขอ"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply("{}"))

    _provider(handler).complete_json("SYS", "USR")
    assert seen["body"]["system"] == "SYS"
    assert all(m["role"] != "system" for m in seen["body"]["messages"])


def test_temperature_is_never_sent():
    """รุ่น 4.6 ขึ้นไปถอด sampling parameter ออกแล้ว — ส่งไปได้ 400 ทั้งคำขอ"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply("{}"))

    _provider(handler).complete_json("s", "u")
    assert "temperature" not in seen["body"]
    assert "top_p" not in seen["body"]


def test_empty_system_is_omitted_entirely():
    """text block ว่างไม่ผ่านฝั่ง Anthropic — ต้องไม่ส่งฟิลด์เลย ไม่ใช่ส่งค่าว่าง"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply("hi"))

    _provider(handler).complete_chat("", [{"role": "user", "content": "ว่าไง"}])
    assert "system" not in seen["body"]


def test_custom_base_url_is_honoured():
    """gateway ที่พูด Messages API ได้ (เช่น proxy ในองค์กร) ต้องชี้ไปได้"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_text_reply("{}"))

    _provider(handler, base_url="http://10.100.152.9:8080/v1/").complete_json("s", "u")
    assert seen["url"] == "http://10.100.152.9:8080/v1/messages"


def test_default_model_when_config_leaves_it_blank():
    """ผู้ใช้ไม่ระบุรุ่น = ได้รุ่นที่เหมาะกับงานวางแผน ไม่ใช่คำขอที่ model ว่างแล้ว 400"""
    provider = AnthropicProvider("", KEY)
    assert provider.model == ANTHROPIC_DEFAULT_MODEL


# ── อ่านคำตอบ ──────────────────────────────────────────────────────────────────

def test_thinking_block_before_the_answer_is_skipped():
    """รุ่นใหม่เปิด adaptive thinking เอง block[0] จึงเป็น thinking ไม่ใช่คำตอบ

    หยิบ content[0] ตรง ๆ แบบ OpenAI จะได้ string ว่างทั้งที่คำตอบอยู่ block ถัดไป
    """
    payload = {
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": '{"engine": "vllm"}'},
        ],
        "stop_reason": "end_turn",
    }
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    assert provider.complete_json("s", "u") == '{"engine": "vllm"}'


def test_multiple_text_blocks_are_joined():
    payload = {
        "content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}],
        "stop_reason": "end_turn",
    }
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    assert provider.complete_json("s", "u") == '{"a": 1}'


def test_refusal_says_so_instead_of_an_empty_answer():
    """การปฏิเสธยังเป็น HTTP 200 — ไม่เช็คก็จะไปไล่หาที่ key กับเน็ตแทน"""
    payload = {"content": [], "stop_reason": "refusal"}
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="ปฏิเสธ"):
        provider.complete_json("s", "u")


def test_truncated_json_is_an_error_not_a_broken_plan():
    """JSON ที่ถูกตัดกลางคันเอาไป parse ต่อไม่ได้ — บอกสาเหตุจริงดีกว่าให้ไปเดาที่ schema"""
    payload = {"content": [{"type": "text", "text": '{"engine": "vl'}], "stop_reason": "max_tokens"}
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="max_tokens"):
        provider.complete_json("s", "u")


def test_truncated_chat_still_shows_what_came_back():
    """แชทที่ถูกตัดยังอ่านรู้เรื่อง — โชว์เท่าที่ได้ ดีกว่าไม่ได้อะไรเลย"""
    payload = {"content": [{"type": "text", "text": "เครื่อง spark-01 กำลัง"}],
               "stop_reason": "max_tokens"}
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    out = provider.complete_chat("s", [{"role": "user", "content": "ถาม"}])
    assert out == "เครื่อง spark-01 กำลัง"


def test_body_without_text_block_is_reported_as_malformed():
    payload = {"content": [{"type": "tool_use", "name": "x"}], "stop_reason": "tool_use"}
    provider = _provider(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="ผิดปกติ"):
        provider.complete_json("s", "u")


# ── error → ข้อความที่บอกว่าต้องแก้อะไร ────────────────────────────────────────

def test_bad_key_points_at_set_key_and_does_not_retry(no_sleep):
    """401 = key ผิด — ยิงซ้ำก็เหมือนเดิม ต้องเด้งทันทีพร้อมคำสั่งที่ใช้แก้"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
        })

    with pytest.raises(ProviderError, match="set-key anthropic") as exc:
        _provider(handler).complete_json("s", "u")
    assert "401" in str(exc.value)
    assert "invalid x-api-key" in str(exc.value)  # เหตุผลจริงจากฝั่งเขา ไม่ใช่แค่ status
    assert len(calls) == 1
    assert no_sleep == []


def test_unknown_model_400_tells_you_to_check_the_model_name(no_sleep):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "model: claude-typo not found"},
        })

    with pytest.raises(ProviderError, match="ชื่อโมเดล") as exc:
        _provider(handler, model="claude-typo").complete_json("s", "u")
    assert "claude-typo not found" in str(exc.value)
    assert len(calls) == 1  # 400 ไม่ใช่ error ชั่วคราว


def test_rate_limit_retries_then_explains_the_quota(no_sleep):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={
            "type": "error", "error": {"type": "rate_limit_error", "message": "slow down"},
        })

    with pytest.raises(ProviderError, match="rate limit"):
        _provider(handler).complete_json("s", "u")
    assert len(calls) == providers.MAX_HTTP_ATTEMPTS  # ชั่วคราว → retry ก่อนยอมแพ้
    assert no_sleep == [2.0, 4.0]


def test_overloaded_529_is_treated_as_temporary(no_sleep):
    """529 overloaded_error เป็นของ Anthropic ตัวเดียว — ฝั่งเขาแน่น ไม่ใช่เราส่งผิด"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(529, json={
                "type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"},
            })
        return httpx.Response(200, json=_text_reply('{"ok": 1}'))

    assert _provider(handler).complete_json("s", "u") == '{"ok": 1}'
    assert len(calls) == 2


def test_non_json_error_body_falls_back_to_raw_text():
    """proxy/load balancer ขวางหน้าอาจตอบ HTML — ต้องไม่ระเบิดตอนแกะ error"""
    provider = _provider(lambda r: httpx.Response(502, text="<html>bad gateway</html>"))
    with pytest.raises(ProviderError, match="502"):
        provider.complete_json("s", "u")


# ── ประวัติแชท ────────────────────────────────────────────────────────────────

def test_history_is_trimmed_to_start_at_a_user_turn():
    """build_messages() ตัดท้ายด้วย history[-MAX_TURNS:] ซึ่งลงตรงคำตอบของผู้ช่วยได้

    Anthropic บังคับให้ข้อความแรกเป็น user — ปล่อยผ่านคือกล่องแชทพังทั้งกล่อง
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply("ได้เลย"))

    history = [
        {"role": "assistant", "content": "ก่อนหน้านี้ตอบไว้"},
        {"role": "user", "content": "แล้ว spark-02 ล่ะ"},
    ]
    assert _provider(handler).complete_chat("SYS", history) == "ได้เลย"
    assert seen["body"]["messages"] == [{"role": "user", "content": "แล้ว spark-02 ล่ะ"}]


def test_blank_and_foreign_roles_are_cleaned_out():
    """content ว่างและ role แปลก ๆ ทำให้ทั้งคำขอถูกปฏิเสธ — กรองทิ้งที่จุดเดียว"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_text_reply("ok"))

    history = [
        {"role": "user", "content": "ถามแรก"},
        {"role": "assistant", "content": "   "},
        {"role": "system", "content": "เศษจากประวัติเก่า"},
        {"role": "user", "content": "ถามต่อ"},
    ]
    _provider(handler).complete_chat("SYS", history)
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "ถามแรก"},
        {"role": "user", "content": "เศษจากประวัติเก่า"},
        {"role": "user", "content": "ถามต่อ"},
    ]
    assert seen["body"]["max_tokens"] == providers.ANTHROPIC_CHAT_MAX_TOKENS


# ── สตรีม ─────────────────────────────────────────────────────────────────────

def _sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_stream_yields_only_text_deltas():
    """event มีหลายชนิด — thinking_delta เป็นความคิด ไม่ใช่คำตอบที่จะเอาไปโชว์"""
    body = _sse(
        {"type": "message_start", "message": {"id": "msg_1"}},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "ขอคิด"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "spark-01 "}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ปกติดี"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    )
    provider = _provider(lambda r: httpx.Response(200, text=body))
    pieces = list(provider.stream_chat("SYS", [{"role": "user", "content": "ถาม"}]))
    assert pieces == ["spark-01 ", "ปกติดี"]
    assert "".join(pieces) == "spark-01 ปกติดี"


def test_stream_sets_the_stream_flag():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=_sse({"type": "message_stop"}))

    list(_provider(handler).stream_chat("SYS", [{"role": "user", "content": "ถาม"}]))
    assert seen["body"]["stream"] is True


def test_stream_that_dies_halfway_says_why():
    """ต่อสายไปแล้วจึงยังเป็น HTTP 200 — error มาเป็น event กลางทาง

    ไม่เช็คก็จะหยุดกลางประโยคเฉย ๆ โดยไม่มีใครบอกว่าทำไม
    """
    body = _sse(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "spark-01 "}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    provider = _provider(lambda r: httpx.Response(200, text=body))
    stream = provider.stream_chat("SYS", [{"role": "user", "content": "ถาม"}])
    assert next(stream) == "spark-01 "
    with pytest.raises(ProviderError, match="Overloaded"):
        next(stream)


def test_stream_error_status_becomes_a_readable_provider_error():
    """สตรีมต่อไม่ติดต้องได้ข้อความเดียวกับตอนไม่สตรีม — หน้าเว็บเอาไปโชว์ในกล่องแชท"""
    provider = _provider(lambda r: httpx.Response(401, json={
        "type": "error", "error": {"type": "authentication_error", "message": "bad key"},
    }))
    with pytest.raises(ProviderError, match="set-key anthropic"):
        list(provider.stream_chat("SYS", [{"role": "user", "content": "ถาม"}]))


# ── ต่อเข้ากับ config ─────────────────────────────────────────────────────────

def test_make_provider_dispatches_to_the_adapter():
    config = ProviderConfig(name=ProviderName.ANTHROPIC, model="claude-opus-5")
    provider = make_provider(config, "k-1234567890")
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic" and provider.model == "claude-opus-5"


def test_make_provider_requires_a_key():
    config = ProviderConfig(name=ProviderName.ANTHROPIC, model="claude-opus-5")
    with pytest.raises(MissingKey, match="set-key anthropic"):
        make_provider(config, None)
