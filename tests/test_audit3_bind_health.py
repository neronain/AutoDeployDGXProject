"""audit 2026-10-06 — health / endpoint / watchdog ยิง `127.0.0.1` เสมอ ไม่สนว่า bundle ผูกที่อยู่ไหน

เคสที่ auditor รันได้จริง: `lmds set qwen --bind <ip ของสาย management>` แล้ว start · โมเดลฟังเฉพาะ IP นั้น
(controller ส่ง `--host <ip>` และรันแบบ `--network host`) — ของจริงตอบ `GET http://<ip>:<port>/health → 200`
แต่ LMDS บอก `running=True healthy=False endpoint=http://127.0.0.1:<port>/v1` และ watchdog ที่เปิดไว้
สั่ง restart 3 ครั้งใส่โมเดลที่ตอบทุกคำขอ แล้วเลิก (gave up) — โมเดลของลูกค้าถูก restart สามรอบเพราะ
*เรา* ถามผิดที่

เทสรันของจริง: เซิร์ฟเวอร์ HTTP จริงที่ฟัง **เฉพาะ** ที่อยู่ที่ไม่ใช่ 127.0.0.1 · health ของ LMDS กับ
probe ของ watchdog เป็น httpx จริง ไม่มี stub ของตัวยิง
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.fleet import bundle_settings, manager  # noqa: E402
from lmds.fleet import watchdog as wd  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "lmds" / "generator" / "templates"


class _Model(BaseHTTPRequestHandler):
    """llama-server ย่อส่วน: /health · /v1/models · /v1/completions — จดว่าใครถามอะไร"""

    hits: list[str] = []

    def log_message(self, *args):  # noqa: D401 — เงียบ
        pass

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        type(self).hits.append(f"GET {self.path}")
        self._send({"status": "ok"} if self.path == "/health" else {"data": [{"id": "qwen"}]})

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).hits.append(f"POST {self.path}")
        self._send({"choices": [{"text": " pong"}]})


def _other_local_address() -> str:
    """ที่อยู่ของเครื่องนี้ที่ **ไม่ใช่ 127.0.0.1** — ตัวแทนของ IP สาย management ที่ส่งให้ `--bind`

    Linux: ทั้ง 127.0.0.0/8 เป็น loopback ใช้ 127.0.0.2 ได้เลย · macOS มีแค่ 127.0.0.1 จึงใช้ IP ของ
    การ์ดเครือข่าย (ถามด้วย UDP connect ซึ่งไม่ส่งอะไรออกไปจริง)
    """
    candidates = ["127.0.0.2"]
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))
        candidates.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    for address in candidates:
        if address.startswith("127.0.0.1"):
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test:
                test.bind((address, 0))
            return address
        except OSError:
            continue
    return ""


@pytest.fixture
def bound_only_elsewhere():
    """เซิร์ฟเวอร์ที่ฟังเฉพาะที่อยู่อื่น — 127.0.0.1 พอร์ตเดียวกันต้องต่อไม่ติด"""
    address = _other_local_address()
    if not address:
        pytest.skip("เครื่องนี้ไม่มีที่อยู่อื่นนอกจาก 127.0.0.1 ให้ผูก (ไม่มีเครือข่าย)")
    _Model.hits = []
    server = ThreadingHTTPServer((address, 0), _Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield address, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LMDS_WATCHDOG_ROOT", str(tmp_path / "watchdog"))
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _bundle(root: Path, port: int, *, bind: str | None) -> Path:
    """bundle native ที่ "รันอยู่" (pid ของ pytest เอง) · bind=None คือไม่เคยตั้ง"""
    bundle = root / "work" / "qwen"
    bundle.mkdir(parents=True)
    controller = bundle / "qwen-single.sh"
    controller.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{bundle}/calls.log"\n', encoding="utf-8")
    controller.chmod(0o755)
    (bundle / "MODEL_PROFILE.yaml").write_text(
        "runtime: {engine: llamacpp, native_build: true}\nmodel: {id: org/qwen, served_name: qwen}\n",
        encoding="utf-8")
    settings = {"port": str(port)}
    if bind is not None:
        settings["bind"] = bind
    bundle_settings.write(bundle, settings)          # สิ่งที่ `lmds set qwen --bind <ip>` เขียน
    run_dir = root / "run" / "qwen"
    run_dir.mkdir(parents=True)
    (run_dir / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
    (run_dir / "server.meta").write_text(
        f"slug=qwen\nmodel=qwen\nmodel_id=org/qwen\nengine=llamacpp\nmode=native\nport={port}\n"
        f"container=\npid_file={run_dir / 'server.pid'}\ncontroller={controller}\n"
        "started_at=2026-10-06T10:00:00\n", encoding="utf-8")
    return bundle


def test_health_and_endpoint_follow_the_address_the_bundle_binds(sandbox, bound_only_elsewhere):
    address, port = bound_only_elsewhere
    _bundle(sandbox, port, bind=address)

    info = manager.find("qwen")

    assert info.running is True
    assert info.healthy is True, "ของจริงตอบ /health 200 ที่ที่อยู่ที่มันผูก — LMDS ต้องเห็นเหมือนกัน"
    assert info.endpoint == f"http://{address}:{port}/v1"
    assert "GET /health" in _Model.hits


def test_the_watchdog_probes_where_the_model_listens_and_restarts_nothing(sandbox, bound_only_elsewhere):
    """12 รอบ · probe เป็น httpx จริง · เดิม: waiting, waiting, restart ×3 แล้ว gave-up"""
    address, port = bound_only_elsewhere
    bundle = _bundle(sandbox, port, bind=address)
    wd.arm(manager.find("qwen"), policy=wd.Policy(interval=1, failures_before_restart=3,
                                                 settle_seconds=5, backoff_seconds=0, probe_timeout=5))
    clock = [1000.0]

    def tick() -> float:
        clock[0] += 120
        return clock[0]

    actions: list[str] = []
    state = wd.loop("qwen", rounds=12, clock=tick, sleeper=lambda s: None,
                    on_tick=lambda report: actions.append(report["action"]))

    assert actions == ["ok"] * 12, actions
    assert state.restarts == []
    assert state.last_ok_at == clock[0]
    assert _Model.hits.count("POST /v1/completions") == 12
    assert not (bundle / "calls.log").exists(), "ห้ามมีคำสั่งใดถูกส่งไปที่ controller"


@pytest.mark.parametrize("bind", [None, "0.0.0.0"])
def test_a_wildcard_or_unset_bind_still_means_loopback(sandbox, bind):
    """0.0.0.0 / ไม่ได้ตั้ง → 127.0.0.1 ตามเดิม — เซิร์ฟเวอร์ที่ฟังเฉพาะ loopback ต้อง healthy"""
    _Model.hits = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        _bundle(sandbox, port, bind=bind)

        info = manager.find("qwen")

        assert info.healthy is True
        assert info.endpoint == f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_a_bind_given_only_on_the_command_line_is_found_from_the_running_process(
        sandbox, bound_only_elsewhere, monkeypatch):
    """`lmds start qwen --bind <ip>` ไม่ได้บันทึกลง bundle.env — ความจริงอยู่ที่ argv ของ process ที่รันอยู่

    argv อ่านจาก /proc หรือ `docker inspect` (ไม่มีบน macOS ที่รันเทสนี้) จึงป้อนผ่านจุดอ่านเดียวกับที่
    `running_context` ใช้ · ที่ยืนยันคือ health จริงที่ที่อยู่นั้น และ endpoint ที่รายงานออกมา
    """
    address, port = bound_only_elsewhere
    _bundle(sandbox, port, bind=None)
    monkeypatch.setattr(manager, "_running_words",
                        lambda info: ["llama-server", "--host", address, "--port", str(port)])

    info = manager.find("qwen")

    assert info.healthy is True
    assert info.endpoint == f"http://{address}:{port}/v1"


def test_an_address_from_the_process_is_not_believed_until_it_answers(sandbox, monkeypatch):
    """argv ของ container ที่ไม่ได้ใช้ host network คือที่อยู่ *ข้างใน* คอนเทนเนอร์ — ใช้ได้ก็ต่อเมื่อตอบจริง"""
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    _bundle(sandbox, port, bind=None)
    monkeypatch.setattr(manager, "_running_words", lambda info: ["vllm", "serve", "--host", "172.17.0.2"])
    monkeypatch.setattr(manager, "_health_ok", lambda port, engine="", host="127.0.0.1": False)

    info = manager.find("qwen")

    assert info.healthy is False
    assert info.endpoint == f"http://127.0.0.1:{port}/v1", "ไม่ตอบ = ไม่เปลี่ยนที่อยู่ที่รายงาน"


def test_python_resolves_the_local_address_exactly_like_the_controller_does():
    """แหล่งความจริงเดียว: `_local_api_host` ของ controller (stacked) — ดึงฟังก์ชัน bash จริงออกมารัน
    แล้วเทียบกับ `manager.local_api_host` ทีละค่า · ถ้าวันหนึ่งสองฝั่งตีความ `--bind` ต่างกัน
    start จะรอ health ที่หนึ่ง แต่ `lmds ps` กับ watchdog ไปถามอีกที่"""
    template = (TEMPLATES / "stacked-vllm-controller.sh.j2").read_text(encoding="utf-8")
    start = template.index("_local_api_host() {")
    function = template[start: template.index("\n}", start) + 2]

    for bind in ["", "0.0.0.0", "127.0.0.1", "localhost", "::", "[::]", "10.2.1.195", "192.168.1.7", "spark-head"]:
        said = subprocess.run(["bash", "-c", function + "\n_local_api_host"], capture_output=True,
                              text=True, env={**os.environ, "API_HOST": bind}).stdout
        assert manager.local_api_host(bind) == said, f"bind={bind!r}: controller ใช้ {said!r}"
