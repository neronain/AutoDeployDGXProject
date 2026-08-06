"""ค่าตั้ง Claude Code — ทุกเคสที่เคยพลาดตอนต่อด้วยมือมีเทสคุมไว้"""

import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.connect import (
    ClaudeCodeConfig,
    ConnectError,
    ProbeResult,
    build_config,
    env_lines,
    probe_endpoint,
    settings_path,
    write_settings,
)
from lmds.connect.claude_code import (
    AUTO_COMPACT_MIN,
    MODEL_ENV_KEYS,
    NO_AUTH_TOKEN,
    SHELL_UNSET_KEYS,
)

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
    if status != 200:
        return httpx.Response(status, text=text)
    payloads = []
    for block in blocks if blocks is not None else [{"type": "text", "text": text}]:
        payloads.append({"type": "content_block_start", "content_block": block})
    body = "".join(f"event: {p['type']}\ndata: {json.dumps(p)}\n\n" for p in payloads)
    body += 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def _stream_reply(*events):
    body = "".join(f"event: e\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


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
    big = {**LLAMACPP_CONFIG, "max_input_tokens": 200000, "server_context_total": 262144}
    config = build_config(big, port=8000)
    assert config.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
    assert config.compact_hint == ""


def test_vllm_context_key_is_read_too():
    vllm = {"base_url": "http://h:8000/v1", "model": "m", "server_context": 131072}
    assert build_config(vllm, port=8000).context == 131072


def test_llamacpp_uses_safe_input_budget_not_total_context():
    config = build_config(
        {
            "base_url": "http://h:8000/v1",
            "model": "m",
            "server_context_total": 524288,
            "context_per_slot": 131072,
            "max_input_tokens": 98304,
        },
        port=8000,
    )
    assert config.context == 98304
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in config.env


@pytest.mark.parametrize("value", [0, 1.5, True, -1, "1.5", "-1"])
def test_rejects_non_positive_or_fractional_token_budgets(value):
    with pytest.raises(ConnectError, match="จำนวนเต็มบวก"):
        build_config({**LLAMACPP_CONFIG, "max_input_tokens": value}, port=8000)


# ── token ───────────────────────────────────────────────────────────
def test_no_token_endpoint_gets_non_secret_gateway_credential():
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert config.env["ANTHROPIC_AUTH_TOKEN"] == NO_AUTH_TOKEN
    assert config.needs_token is False


def test_shell_block_references_env_var_not_the_secret():
    """ไม่พ่น token ขึ้นจอ/ลงประวัติเชลล์ — กฎข้อ 4 ของโปรเจกต์"""
    config = build_config({**LLAMACPP_CONFIG, "api_key": "s3cr3t-token-value"}, port=8000)
    block = "\n".join(env_lines(config))
    assert 's3cr3t-token-value' not in block
    assert 'export ANTHROPIC_AUTH_TOKEN="$API_KEY"' in block


def test_shell_block_quotes_values_and_unsets_cloud_provider_modes():
    config = ClaudeCodeConfig(
        base_url="http://x", model="m", env={"ANTHROPIC_BASE_URL": "http://host/a b;echo nope"}
    )
    lines = env_lines(config)
    assert lines[0] == "unset " + " ".join(SHELL_UNSET_KEYS)
    assert lines[1] == "export ANTHROPIC_BASE_URL='http://host/a b;echo nope'"


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
            assert body["stream"] is True
            assert body["tool_choice"] == {"type": "tool", "name": "read_file"}
            return _reply(blocks=[{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}])
        assert body["stream"] is True
        assert request.url.query == b"beta=true"
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
    assert "tool_use" in result.detail


def test_probe_sends_same_bearer_auth_path_as_claude_code():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["version"] = request.headers.get("anthropic-version")
        return _reply("OK")

    probe_endpoint(build_config({**LLAMACPP_CONFIG, "api_key": "k"}, port=8000), client=_client(handler))
    assert seen["authorization"] == "Bearer k"
    assert seen["version"] == "2023-06-01"


def test_probe_rejects_http_200_without_text_block():
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply(blocks=[{"type": "thinking", "thinking": "still thinking"}])

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert not result.messages_ok
    assert "text block" in result.detail


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
    assert oct(backup.stat().st_mode)[-3:] == "600"


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


def test_write_settings_removes_stale_lmds_and_cloud_provider_values(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "FOO": "keep",
                    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "ANTHROPIC_AUTH_TOKEN": "old-secret",
                }
            }
        ),
        encoding="utf-8",
    )
    written, _ = write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)
    env = json.loads(written.read_text(encoding="utf-8"))["env"]
    assert env["FOO"] == "keep"
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert env["ANTHROPIC_AUTH_TOKEN"] == NO_AUTH_TOKEN


def test_write_settings_failure_keeps_original_intact(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    original = '{"env":{"KEEP":"yes"}}\n'
    target.write_text(original, encoding="utf-8")
    real_replace = os.replace

    def fail_target_replace(src, dst):
        if Path(dst) == target:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_target_replace)
    with pytest.raises(OSError, match="simulated"):
        write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)
    assert target.read_text(encoding="utf-8") == original


def test_write_settings_preserves_symlink(tmp_path):
    actual = tmp_path / "actual.json"
    actual.write_text('{"env":{"KEEP":"yes"}}', encoding="utf-8")
    link = tmp_path / "settings.json"
    link.symlink_to(actual)
    written, _ = write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=link)
    assert written == link and link.is_symlink()
    assert json.loads(actual.read_text(encoding="utf-8"))["env"]["KEEP"] == "yes"


def test_settings_path_honors_claude_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-alt"))
    assert settings_path() == tmp_path / "claude-alt" / "settings.json"


def test_current_main_and_alias_model_slots_are_all_pinned():
    config = build_config(LLAMACPP_CONFIG, port=8000)
    assert config.env["ANTHROPIC_MODEL"] == "demo-gguf"
    assert config.env["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "demo-gguf"
    assert all(config.env[name] == "demo-gguf" for name in MODEL_ENV_KEYS)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://user:pass@host:8000",
        "http://host:8000/?token=x",
        "http://host:99999",
        "http://host:not-a-port",
        "http://[::1",
        "http://[::gg]/",
    ],
)
def test_build_config_rejects_unsafe_base_url(url):
    with pytest.raises(ConnectError, match="base URL"):
        build_config({"model": "demo", "anthropic_base_url": url}, port=8000)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("model", "model\nother", "ชื่อโมเดล"), ("api_key", "secret\rheader", "HTTP header")],
)
def test_build_config_rejects_control_characters(field, value, message):
    with pytest.raises(ConnectError, match=message):
        build_config({**LLAMACPP_CONFIG, field: value}, port=8000)


def test_probe_reads_realistic_text_delta_stream():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stream_reply(
                {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "O"}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "K"}},
            )
        return _stream_reply(
            {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": "t",
                    "name": "read_file",
                    "input": {},
                },
            }
        )

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert result.messages_ok and result.tools_ok
    assert result.sample == "OK"


def test_probe_requires_sse_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "OK"}]})

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert not result.messages_ok
    assert "Content-Type" in result.detail


def test_probe_never_echoes_api_key_from_server_error():
    key = "super-secret-token-value"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"bad bearer {key}")

    config = build_config({**LLAMACPP_CONFIG, "api_key": key}, port=8000)
    result = probe_endpoint(config, client=_client(handler))
    assert key not in result.detail
    assert "***" in result.detail


def test_probe_redacts_api_key_before_truncating_server_error():
    key = "SECRET_BOUNDARY_TOKEN"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="x" * 198 + key)

    config = build_config({**LLAMACPP_CONFIG, "api_key": key}, port=8000)
    result = probe_endpoint(config, client=_client(handler))
    assert key not in result.detail
    assert "SE" not in result.detail


def test_probe_strips_terminal_control_characters_from_server_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="bad\x1b]8;;https://evil\x07link")

    result = probe_endpoint(build_config(LLAMACPP_CONFIG, port=8000), client=_client(handler))
    assert "\x1b" not in result.detail
    assert "\x07" not in result.detail


def test_write_settings_refuses_non_regular_target(tmp_path):
    target = tmp_path / "settings.json"
    target.mkdir()
    with pytest.raises(ConnectError, match="regular file"):
        write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)


def test_write_settings_detects_concurrent_change(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    original = '{"env":{"KEEP":"original"}}\n'
    changed = '{"env":{"KEEP":"newer"}}\n'
    target.write_text(original, encoding="utf-8")
    real_read_text = Path.read_text
    reads = 0

    def race_on_final_read(path, *args, **kwargs):
        nonlocal reads
        value = real_read_text(path, *args, **kwargs)
        if path == target:
            reads += 1
            if reads == 2:
                target.write_text(changed, encoding="utf-8")
                return changed
        return value

    monkeypatch.setattr(Path, "read_text", race_on_final_read)
    with pytest.raises(ConnectError, match="เปลี่ยนระหว่างคำสั่ง"):
        write_settings(build_config(LLAMACPP_CONFIG, port=8000), path=target)
    assert real_read_text(target, encoding="utf-8") == changed


def _patch_connect_cli(tmp_path, monkeypatch, probe):
    from lmds.fleet import ServerInfo

    controller = tmp_path / "controller.sh"
    controller.write_text("#!/bin/sh\n", encoding="utf-8")
    server = ServerInfo(
        slug="demo", controller=str(controller), port=8000, running=True, healthy=True
    )
    monkeypatch.setattr("lmds.fleet.find", lambda slug: server)
    payload = json.dumps(LLAMACPP_CONFIG)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, payload, ""),
    )
    monkeypatch.setattr("lmds.connect.probe_endpoint", lambda config: probe)


def test_cli_tool_probe_failure_blocks_write(tmp_path, monkeypatch):
    _patch_connect_cli(
        tmp_path,
        monkeypatch,
        ProbeResult(messages_ok=True, tools_ok=False, sample="OK", detail="no tool_use"),
    )
    writes = []
    monkeypatch.setattr("lmds.cli.main._connect_write", lambda config, yes: writes.append(config))

    result = CliRunner().invoke(app, ["connect", "demo", "--write", "--yes"])

    assert result.exit_code == 2, result.output
    assert not writes
    assert "ยังไม่พร้อมใช้กับ Claude Code" in result.output


def test_cli_write_refuses_inherited_cloud_provider_route(tmp_path, monkeypatch):
    _patch_connect_cli(
        tmp_path,
        monkeypatch,
        ProbeResult(messages_ok=True, tools_ok=True, sample="OK"),
    )
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    writes = []
    monkeypatch.setattr("lmds.cli.main._connect_write", lambda config, yes: writes.append(config))

    result = CliRunner().invoke(app, ["connect", "demo", "--write", "--yes"])

    assert result.exit_code == 2, result.output
    assert not writes
    assert "unset CLAUDE_CODE_USE_BEDROCK" in result.output


def test_cli_client_config_timeout_is_clean_error(tmp_path, monkeypatch):
    _patch_connect_cli(
        tmp_path,
        monkeypatch,
        ProbeResult(messages_ok=True, tools_ok=True, sample="OK"),
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = CliRunner().invoke(app, ["connect", "demo"])
    assert result.exit_code == 2, result.output
    assert "เรียก client-config ไม่สำเร็จ" in result.output
    assert "Traceback" not in result.output


def test_cli_renders_untrusted_probe_sample_as_plain_text(tmp_path, monkeypatch):
    _patch_connect_cli(
        tmp_path,
        monkeypatch,
        ProbeResult(messages_ok=True, tools_ok=True, sample="[link=https://evil]click[/link]"),
    )
    result = CliRunner().invoke(app, ["connect", "demo"])
    assert result.exit_code == 0, result.output
    assert "[link=https://evil]click[/link]" in result.output


def test_config_dataclass_defaults_are_safe():
    config = ClaudeCodeConfig(base_url="http://x", model="m")
    assert config.needs_token is False
    assert config.compact_hint == ""
