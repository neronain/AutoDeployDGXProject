"""`lmds mcp` พูด MCP ทาง stdio ได้จริง · stdout มีแต่ JSON-RPC · ปฏิเสธของที่ไม่ควรรับก่อนมีคำสั่งไหนถูกรัน

ที่มา: เจ้าของคุม LMDS ผ่านผู้ช่วยเขียนโค้ดที่รัน `lmds …` แล้วแกะตารางของ rich (ตารางถูกตัดตามความกว้างจอ) —
MCP server ให้มันถามผ่านเครื่องมือที่คืน JSON แทน (2026-10-09) · ทุกเทสในไฟล์นี้เปิด server ตัวจริงเป็น process ลูก
แล้วคุยด้วยข้อความ JSON-RPC จริงทาง stdin/stdout บนฟลีตจำลองของ tests/mcp_fleet.py — ไม่มีเทสไหนเรียกฟังก์ชันตรง ๆ
"""

from __future__ import annotations

import json
import os
import pty
import sys
import threading
from pathlib import Path

import pytest

from tests import mcp_fleet
from tests.mcp_fleet import DOCKER_SLUG, NATIVE_SLUG, NODE_OK

EXPECTED_TOOLS = {
    "lmds_version", "lmds_nodes", "lmds_models", "lmds_inspect", "lmds_plan", "lmds_fit", "lmds_fleet_check",
    "lmds_watchdog_status", "lmds_logs", "lmds_doctor",
}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    built = mcp_fleet.build(tmp_path, monkeypatch)
    yield built
    built.close()


# ── handshake · รายการเครื่องมือ ──────────────────────────────────────────────────────────────────
def test_handshake_then_tool_listing_over_stdio(fleet):
    client = fleet.mcp()
    hello = client.initialize("2025-06-18")
    assert hello["jsonrpc"] == "2.0" and hello["result"]["protocolVersion"] == "2025-06-18"
    assert hello["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert hello["result"]["serverInfo"]["name"] == "lmds"
    assert "READ-ONLY" in hello["result"]["instructions"]

    assert client.request("ping")["result"] == {}

    tools = client.request("tools/list")["result"]["tools"]
    assert {tool["name"] for tool in tools} == EXPECTED_TOOLS
    for tool in tools:
        said = tool["description"]
        # คำอธิบายคือสิ่งเดียวที่ผู้ช่วยอ่านตอนเลือกเครื่องมือ — ต้องบอกว่าอ่านอย่างเดียวและราคาของการเรียก
        assert "Read-only" in said and "Cost:" in said, tool["name"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        assert tool["annotations"]["readOnlyHint"] is True and tool["annotations"]["destructiveHint"] is False
    by_name = {tool["name"]: tool for tool in tools}
    assert "EVERY registered node" in by_name["lmds_fleet_check"]["description"]
    assert "Hugging Face" in by_name["lmds_inspect"]["description"]
    assert by_name["lmds_logs"]["inputSchema"]["properties"]["lines"]["maximum"] == 500
    assert client.close() == 0


def test_an_older_protocol_version_is_honoured_and_an_unknown_one_gets_the_latest(fleet):
    old = fleet.mcp()
    assert old.initialize("2024-11-05")["result"]["protocolVersion"] == "2024-11-05"
    future = fleet.mcp()
    assert future.initialize("2099-01-01")["result"]["protocolVersion"] == "2025-11-25"


def test_notifications_get_no_reply_and_the_server_ends_when_stdin_closes(fleet):
    client = fleet.mcp()
    client.initialize()
    client.notify("notifications/cancelled", {"requestId": 99, "reason": "user"})
    client.notify("notifications/roots/list_changed")
    assert client.request("ping", request_id="after-notifications")["id"] == "after-notifications"
    assert client.close() == 0
    replies = [json.loads(line) for line in client.stdout_lines]
    assert [reply.get("id") for reply in replies] == [1, "after-notifications"], "notification ต้องไม่มีคำตอบ"


# ── JSON-RPC error ────────────────────────────────────────────────────────────────────────────────
def test_protocol_errors_use_the_json_rpc_codes(fleet):
    client = fleet.mcp()
    client.initialize()
    assert client.request("resources/list")["error"]["code"] == -32601
    assert client.request("tools/run", {"name": "lmds_version"})["error"]["code"] == -32601

    unknown = client.request("tools/call", {"name": "lmds_restart", "arguments": {"slug": NATIVE_SLUG}})
    assert unknown["error"]["code"] == -32602 and "unknown tool" in unknown["error"]["message"]
    assert client.request("tools/call", {"arguments": {}})["error"]["code"] == -32602
    assert client.request("tools/call", {"name": "lmds_version", "arguments": ["x"]})["error"]["code"] == -32602
    assert client.request("tools/call", ["lmds_version"])["error"]["code"] == -32602

    client.send_raw("this is not json")
    assert client.next_message() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
    client.send_raw(json.dumps({"id": 7, "method": "ping"}))            # ไม่มี "jsonrpc": "2.0"
    assert client.next_message()["error"]["code"] == -32600
    client.send_raw(json.dumps([]))
    assert client.next_message()["error"]["code"] == -32600
    # server ยังอยู่หลังข้อความพังทั้งชุด
    assert client.request("ping")["result"] == {}


def test_a_batch_is_answered_item_by_item(fleet):
    client = fleet.mcp()
    client.initialize()
    client.send_raw(json.dumps([{"jsonrpc": "2.0", "id": "a", "method": "ping"},
                                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                                {"jsonrpc": "2.0", "id": "b", "method": "nope"}]))
    assert client.wait_for("a")["result"] == {}
    assert client.wait_for("b")["error"]["code"] == -32601


# ── argument ที่ไม่ไว้ใจ: ปฏิเสธก่อนมีคำสั่งไหนถูกประกอบ ─────────────────────────────────────────────
HOSTILE = [
    "x; rm -rf ~", "$(touch PWNED)", "`touch PWNED`", "../../etc/passwd", "a b", "x' ; id ; '", "-oProxyCommand=id",
    "x\nlmds stop all", "x|id", "x&&id", "${IFS}", "é" * 80,
]


@pytest.mark.parametrize("value", HOSTILE)
def test_a_hostile_node_name_or_slug_is_refused_before_anything_runs(fleet, value):
    """ผู้ช่วยส่งอะไรก็ตามที่หน้าเว็บ/README ที่มันเพิ่งอ่านบอกให้ส่ง — ชื่อเครื่องต้องอยู่ในทะเบียน slug ต้องเป็นชื่อ bundle"""
    client = fleet.mcp()
    client.initialize()
    attempts = [
        ("lmds_models", {"node": value}),
        ("lmds_fit", {"slug": value}),
        ("lmds_fit", {"slug": value, "node": NODE_OK}),
        ("lmds_fit", {"slug": DOCKER_SLUG, "node": value}),
        ("lmds_logs", {"slug": value}),
        ("lmds_logs", {"slug": value, "node": NODE_OK}),
        ("lmds_doctor", {"slug": value, "node": NODE_OK}),
        ("lmds_doctor", {"slug": "gemma-3-27b", "node": value}),
        ("lmds_watchdog_status", {"slug": value, "node": NODE_OK}),
        ("lmds_watchdog_status", {"node": value}),
    ]
    for name, arguments in attempts:
        answer = client.call(name, arguments)
        assert answer["error"] is True and answer["payload"].get("refused") is True, (name, arguments, answer["payload"])
    assert client.close() == 0
    # ไม่มีอะไรถูกรันเลย: ไม่มี ssh ไม่มี docker (ซึ่ง controller ของ bundle เรียก) และไม่มีไฟล์ที่ payload พยายามสร้าง
    assert fleet.calls("ssh") == [] and fleet.calls("remote") == [] and fleet.calls("docker") == []
    assert not list(fleet.home.rglob("PWNED"))


@pytest.mark.parametrize("name,arguments,complaint", [
    ("lmds_version", {"verbose": True}, "unknown argument"),
    ("lmds_fit", {}, "missing required"),
    ("lmds_fit", {"slug": DOCKER_SLUG, "slots": "4"}, "must be an integer"),
    ("lmds_fit", {"slug": DOCKER_SLUG, "slots": True}, "must be an integer"),
    ("lmds_fit", {"slug": DOCKER_SLUG, "slots": 0}, "between 1 and"),
    ("lmds_fit", {"slug": DOCKER_SLUG, "context": 10**12}, "between 1 and"),
    ("lmds_fit", {"slug": DOCKER_SLUG, "apply": True}, "unknown argument"),
    ("lmds_logs", {"slug": DOCKER_SLUG, "lines": 100000}, "between 1 and 500"),
    ("lmds_logs", {"slug": ["a"]}, "must be a string"),
    ("lmds_fleet_check", {"check": "yes"}, "true or false"),
    ("lmds_inspect", {"model": "Qwen/Qwen3-32B", "targets": ["nope-9000"]}, "unknown target"),
    ("lmds_inspect", {"model": "Qwen/Qwen3-32B", "targets": ["dgx-spark-single"] * 9}, "too many items"),
    ("lmds_inspect", {"model": "x" * 301}, "too long"),
    ("lmds_inspect", {"model": "ftp://example.com/a/b"}, "not a Hugging Face repo"),
    ("lmds_inspect", {"model": "Qwen/Qwen3-32B", "revision": "main; id"}, "revision"),
    ("lmds_inspect", {"model": "Qwen/Qwen3-32B", "revision": "../../x"}, "revision"),
    ("lmds_inspect", {"model": "Qwen/Qwen3-32B", "kv_dtype": "fp4"}, "unknown kv_dtype"),
    ("lmds_plan", {"model": "Qwen/Qwen3-32B", "engine": "ollama"}, "unknown engine"),
    ("lmds_plan", {"model": "Qwen/Qwen3-32B", "no_llm": False}, "unknown argument"),
    ("lmds_plan", {"model": "Qwen/Qwen3-32B", "target": "$(id)"}, "unknown target"),
])
def test_bad_arguments_are_refused_with_the_reason(fleet, name, arguments, complaint):
    client = fleet.mcp(fake_hub=True)
    client.initialize()
    answer = client.call(name, arguments)
    assert answer["error"] is True and answer["payload"]["refused"] is True
    assert complaint in answer["payload"]["error"], answer["payload"]["error"]
    assert fleet.calls("ssh") == [] and fleet.calls("docker") == []


# ── stdout มีแต่ JSON-RPC ─────────────────────────────────────────────────────────────────────────
def _console_script() -> list[str] | None:
    script = Path(sys.executable).parent / "lmds"
    return [str(script), "mcp"] if script.is_file() else None


def test_stdout_carries_only_json_rpc_even_with_everything_that_normally_prints(fleet):
    """บรรทัดเดียวที่ไม่ใช่ JSON-RPC บน stdout = client ตัดสายทั้ง session

    เปิดผ่าน entry point ตัวจริง (`lmds mcp`) โดยให้ stderr เป็น tty — banner ของ CLI จึงพิมพ์เหมือนตอนคนรันเอง —
    แล้วเรียกทุกเครื่องมือ รวมตัวที่ CLI ปกติพิมพ์ตาราง/คำเตือน/ข้อความแดง และตัวที่รัน controller ของ bundle
    """
    master, slave = pty.openpty()
    env = {key: value for key, value in fleet.env.items() if key != "NO_COLOR"}
    client = fleet.mcp(argv=_console_script() or fleet.argv("mcp"), env=env, stderr=slave)
    os.close(slave)
    seen_on_tty = bytearray()

    def drain() -> None:
        # ต้องมีคนอ่าน tty ตลอด — buffer ของ pty เต็มแล้ว server ค้างที่การเขียน stderr (banner แบบเคลื่อนไหวพิมพ์เยอะ)
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                return
            if not chunk:
                return
            seen_on_tty.extend(chunk)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    client.initialize()
    client.request("tools/list")
    calls = [
        ("lmds_version", {}), ("lmds_nodes", {}), ("lmds_models", {}), ("lmds_models", {"node": NODE_OK}),
        ("lmds_fleet_check", {"check": True}), ("lmds_watchdog_status", {}), ("lmds_logs", {"slug": DOCKER_SLUG}),
        ("lmds_logs", {"slug": NATIVE_SLUG, "lines": 3}), ("lmds_doctor", {"slug": DOCKER_SLUG}),
        ("lmds_fit", {"slug": DOCKER_SLUG}), ("lmds_fit", {"slug": "no-such-bundle"}),
        ("lmds_plan", {"model": "not a repo at all"}), ("lmds_models", {"node": "nobody"}),
    ]
    for name, arguments in calls:
        client.call(name, arguments)
    client.send_raw("garbage")
    client.next_message()
    assert client.close() == 0

    reader.join(timeout=10)
    os.close(master)
    said = seen_on_tty.decode("utf-8", "replace")
    assert "Local Model Deploy Studio" in said, "banner ต้องพิมพ์จริง (บน stderr) — ไม่งั้นเทสนี้ไม่ได้พิสูจน์อะไร"
    assert "lmds-mcp: ready" in said

    assert len(client.stdout_lines) == 2 + len(calls) + 1
    for raw in client.stdout_lines:
        assert raw.endswith(b"\n")
        message = json.loads(raw)                      # ทุกบรรทัดต้อง parse ได้
        assert isinstance(message, dict) and message["jsonrpc"] == "2.0" and "id" in message
        assert ("result" in message) != ("error" in message)


def test_tool_calls_do_not_block_ping(fleet):
    """เครื่องมือ SSH ไปเครื่องอื่นได้เป็นสิบวินาที — ระหว่างนั้น client ที่ ping ต้องยังได้คำตอบ"""
    fleet.set_node(NODE_OK, "hung")
    client = fleet.mcp()
    client.initialize()
    client.send_raw(json.dumps({"jsonrpc": "2.0", "id": "slow", "method": "tools/call",
                                "params": {"name": "lmds_models", "arguments": {"node": NODE_OK}}}))
    assert client.request("ping", timeout=10)["result"] == {}
    assert client.request("tools/list", timeout=10)["result"]["tools"]
    client.proc.kill()
