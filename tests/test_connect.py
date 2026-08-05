"""ค่าตั้ง Claude Code — ทุกเคสที่เคยพลาดตอนต่อด้วยมือมีเทสคุมไว้"""

import json

import httpx
import pytest

from lmds.connect import (
    ClaudeCodeConfig,
    ConnectError,
    build_config,
    env_lines,
    probe_endpoint,
    write_settings,
)
from lmds.connect.claude_code import AUTO_COMPACT_MIN, MODEL_ENV_KEYS

LLAMACPP_CONFIG = {
    "base_url": "http://10.0.0.5:8000/v1",
    "anthropic_base_url": "http://10.0.0.5:8000",
    "model": "demo-gguf",
    "api_key": "not-required",
    "max_input_tokens": 4096,
    "max_output_tokens": 2048,
    "server_context_total": 32768,
}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _reply(text="OK", blocks=None, status=200):
    body = {"content": blocks if blocks is not None else [{"type": "text", "text": text}]}
    return httpx.Response(status, json=body)


# ── base URL ────────────────────────────────────────────────────────
def test_uses_anthropic_base_url_when_bundle_provides_it():
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert config.base_url == "http://10.0.0.5:8000"


def test_strips_v1_from_openai_base_url_for_older_bundles():
    """bundle ที่ generate ก่อนมี anthropic_base_url ก็ยังต่อได้ ไม่ต้อง deploy ใหม่"""
    old = {k: v for k, v in LLAMACPP_CONFIG.items() if k != "anthropic_base_url"}
    assert build_config(old, port=8000).base_url == "http://10.0.0.5:8000"


def test_base_url_never_keeps_v1_suffix():
    """client สาย Anthropic เติม /v1/messages เอง — เหลือ /v1 ไว้จะได้ /v1/v1/messages แล้ว 404"""
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert not config.base_url.endswith("/v1")


def test_falls_back_to_localhost_when_client_config_has_no_url():
    config = build_config({"model": "demo"}, port=9001)
    assert config.base_url == "http://127.0.0.1:9001"


# ── ช่องโมเดล ───────────────────────────────────────────────────────
def test_maps_every_model_slot():
    """ตั้งช่องเดียวไม่พอ — งานเบื้องหลังกับ subagent ยิงชื่อโมเดลของ Anthropic มาที่เครื่องเรา"""
    config = build_config(LLAMACPP_CONFIG, port=8000)
    for key in MODEL_ENV_KEYS:
        assert config.env[key] == "demo-gguf", f"ขาดการ map ช่อง {key}"


def test_missing_model_is_an_error():
    with pytest.raises(ConnectError, match="ชื่อโมเดล"):
        build_config({"base_url": "http://x/v1"}, port=8000)


# ── เพดาน context ───────────────────────────────────────────────────
def test_skips_auto_compact_when_context_below_claude_code_minimum():
    """Claude Code clamp ค่านี้ขึ้นเป็น 100,000 — ใส่ค่าต่ำกว่านั้นคือบรรทัดที่ไม่มีผล"""
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert config.context < AUTO_COMPACT_MIN
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in config.env
    assert "compact" in config.compact_hint


def test_sets_auto_compact_when_context_is_large_enough():
    big = {**LLAMACPP_CONFIG, "server_context_total": 262144}
    config = build_config(big, port=8000)
    assert config.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "262144"
    assert config.compact_hint == ""


def test_vllm_context_key_is_read_too():
    vllm = {"base_url": "http://h:8000/v1", "model": "m", "server_context": 131072}
    assert build_config(vllm, port=8000).context == 131072


# ── token ───────────────────────────────────────────────────────────
def test_no_token_env_when_endpoint_needs_none():
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert "ANTHROPIC_AUTH_TOKEN" not in config.env
    assert config.needs_token is False


def test_shell_block_references_env_var_not_the_secret():
    """ไม่พ่น token ขึ้นจอ/ลงประวัติเชลล์ — กฎข้อ 4 ของโปรเจกต์"""
    config = build_config({**LLAMACPP_CONFIG, "api_key": "s3cr3t-token-value"}, port=8000)
    block = "\n".join(env_lines(config))
    assert 's3cr3t-token-value' not in block
    assert 'export ANTHROPIC_AUTH_TOKEN="$API_KEY"' in block


def test_literal_token_only_when_asked():
    """settings.json อ่านโดยไม่ผ่านเชลล์ จึงต้องเป็นค่าจริง — แต่ต้องขอเท่านั้น"""
    config = build_config({**LLAMACPP_CONFIG, "api_key": "s3cr3t-token-value"}, port=8000)
    block = "\n".join(env_lines(config, literal_token=True))
    assert "export ANTHROPIC_AUTH_TOKEN=s3cr3t-token-value" in block


# ── probe ───────────────────────────────────────────────────────────
def test_probe_reports_ok_and_skips_thinking_blocks():
    """โมเดลสาย reasoning ส่ง block ชนิด thinking มาก่อน ซึ่งไม่มี key text"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            return _reply(blocks=[{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}])
        return _reply(blocks=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": "OK"}])

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert result.messages_ok and result.tools_ok
    assert result.sample == "OK"


def test_probe_missing_messages_endpoint_says_what_to_do():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert not result.messages_ok
    assert "/v1/messages" in result.detail


def test_probe_rejected_token_points_at_the_right_variable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    config = build_config({**LLAMACPP_CONFIG, "api_key": "k"}, port=8000)
    result = probe_endpoint(config, client=_client(handler))
    assert not result.messages_ok
    assert "API_KEY" in result.detail


def test_probe_tools_unsupported_is_a_warning_not_a_failure():
    """ตอบข้อความได้แต่ไม่ออก tool_use = ต่อติดแต่ใช้งานจริงไม่ได้ — คนละเรื่องกับต่อไม่ติด"""

    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("OK")

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert result.messages_ok
    assert not result.tools_ok


def test_probe_sends_x_api_key_when_endpoint_has_one():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        return _reply("OK")

    probe_endpoint(build_config({**LLAMACPP_CONFIG, "api_key": "k"}, port=8000), client=_client(handler))
    assert seen["key"] == "k"
    assert seen["version"] == "2023-06-01"


# ── settings.json ───────────────────────────────────────────────────
def test_write_settings_merges_without_touching_other_keys(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"model": "opus", "env": {"FOO": "bar"}}), encoding="utf-8")

    config = build_config(LLAMACPP_CONFIG, port=8000)
    written, backup = write_settings(config, path=target)

    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["model"] == "opus"          # คีย์อื่นของผู้ใช้ต้องไม่หาย
    assert data["env"]["FOO"] == "bar"      # env เดิมก็ต้องไม่หาย
    assert data["env"]["ANTHROPIC_BASE_URL"] == "http://10.0.0.5:8000"
    assert backup is not None and json.loads(backup.read_text(encoding="utf-8"))["env"] == {"FOO": "bar"}


def test_write_settings_creates_file_when_missing(tmp_path):
    target = tmp_path / "nested" / "settings.json"
    written, backup = write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)
    assert backup is None
    assert json.loads(written.read_text(encoding="utf-8"))["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "demo-gguf"


def test_write_settings_refuses_broken_json(tmp_path):
    """ไฟล์ของผู้ใช้พังอยู่แล้ว — เขียนทับจะทำให้ของเดิมหายโดยที่เขาไม่รู้"""
    target = tmp_path / "settings.json"
    target.write_text("{ไม่ใช่ json", encoding="utf-8")
    with pytest.raises(ConnectError, match="JSON"):
        write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)


def test_write_settings_refuses_non_object_env(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"env": "ไม่ใช่ object"}), encoding="utf-8")
    with pytest.raises(ConnectError, match="env"):
        write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)


def test_write_settings_keeps_token_out_of_world_readable_file(tmp_path):
    target = tmp_path / "settings.json"
    config = build_config({**LLAMACPP_CONFIG, "api_key": "s3cr3t-token-value"}, port=8000)
    written, _ = write_settings(config, path=target)
    assert oct(written.stat().st_mode)[-3:] == "600"
    assert json.loads(written.read_text(encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"] == "s3cr3t-token-value"


def test_config_dataclass_defaults_are_safe():
    config = ClaudeCodeConfig(base_url="http://x", model="m")
    assert config.needs_token is False
    assert config.compact_hint == ""
