"""ตาม log แบบเกือบ realtime — controller `logs -f` · SSE ของ hub · CLI

เจ้าของ 2026-09-07: "ส่วนของ log ทำให้เป็นการแสดง detail แบบเกือบ realtime ได้ไหม user จะได้ดูว่า error
อะไรด้วย แทนการกด 1 ครั้งแสดง 1 รอบ"

สิ่งที่ต้องเป็นจริงและเคยไม่จริง:
- controller ทั้ง 4 แบบรับ `logs [N] [-f|--follow|follow]` · ไม่มี container/ไฟล์ = บอกบรรทัดเดียวแล้วจบ 0
  · SIGTERM ถึง child (docker logs/tail) จริง ไม่ค้างจนมีบรรทัดใหม่
- hub: SSE framing · keepalive · **ปิดสาย = child ตาย** · เพดานจำนวนสตรีม · token ทาง ?token= · node ผ่าน
  stream() ที่ถือ stdin (ไม่ใช่ run() ที่หมดเวลา 60 วิ)

TestClient ของ Starlette รัน app จนจบแล้วค่อยคืน body ทั้งก้อน — ใช้กับสายที่ไม่จบเองไม่ได้ (ค้างตลอดกาล)
และไม่มีทาง "ตัดสาย" จากฝั่ง client · เทส SSE จึงยก uvicorn จริงขึ้นบน 127.0.0.1 ใน thread แล้วคุยด้วย
http.client ซึ่งปิด socket ได้จริง — นั่นคือสิ่งที่เบราว์เซอร์ทำตอนผู้ใช้ปิดแผง
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import textwrap
import threading
import time
from pathlib import Path

import pytest

from lmds.brain.plan_schema import Engine
from tests.test_review_templates import (
    _bundle, _fake_bin, _gguf_report, _safetensors_report, _stacked_bundle, extract_fn, run_bash,
)

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from lmds.web import create_app, logstream  # noqa: E402
from tests.test_web import fleet, fresh_jobs, no_host_scan, registered  # noqa: E402,F401 — fixture ใช้ร่วมกัน

FAKES = textwrap.dedent('''
    docker() { echo "docker $*"; }
    tail() { echo "tail $*"; }
    ssh_at() { echo "ssh_at $*"; }
    die() { echo "die: $*" >&2; exit 1; }
    worker_count() { set -- $WORKER_IPS; echo "$#"; }
    _cluster_config_is_placeholder() { return 1; }
''')
NO_CONTAINER = 'docker() { if [[ "$1 $2" == "container inspect" ]]; then return 1; fi; echo "docker $*"; }\n'


def _logs_fn(controller: Path) -> str:
    return "set -Eeuo pipefail\n" + FAKES + extract_fn(controller.read_text(encoding="utf-8"), "logs") + "\n"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 4.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


# ───────────────────────── controller: parsing + missing-source path ─────────────────────────
@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG])
def test_docker_controllers_parse_follow_and_keep_the_one_shot(tmp_path, engine):
    bundle = _bundle(tmp_path, _safetensors_report(), engine=engine)
    fn = _logs_fn(bundle.controller)

    out = run_bash(fn + 'CONTAINER_NAME=c1 logs 50 -f; echo "rc=$?"')
    assert out.stdout.strip().splitlines() == ["docker logs --tail 50 -f c1", "rc=0"], out.stderr
    # ลำดับ/คำสะกดอื่นของ follow — หน้าเว็บส่ง `logs N -f` · คนพิมพ์เอง `logs follow`, `logs --follow 20`
    for argv in ("follow", "--follow 20", "20 follow", "-f"):
        out = run_bash(fn + f"CONTAINER_NAME=c1 logs {argv}")
        n = "20" if "20" in argv else "300"
        assert out.stdout.strip() == f"docker logs --tail {n} -f c1", (argv, out.stdout, out.stderr)
    # one-shot เหมือนเดิมเป๊ะ — hub และ `lmds logs` ยังเรียกแบบนี้
    out = run_bash(fn + "CONTAINER_NAME=c1 logs 12")
    assert out.stdout.strip() == "docker logs --tail 12 c1"
    out = run_bash(fn + "CONTAINER_NAME=c1 logs")
    assert out.stdout.strip() == "docker logs --tail 300 c1"


@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG])
def test_docker_controllers_do_not_spin_without_a_container(tmp_path, engine):
    """ยังไม่ start = ไม่มี container · ต้องบอกหนึ่งบรรทัดแล้วจบ 0 — หน้าเว็บต่อใหม่เองระหว่างรอ start"""
    bundle = _bundle(tmp_path, _safetensors_report(), engine=engine)
    out = run_bash(_logs_fn(bundle.controller) + NO_CONTAINER + 'CONTAINER_NAME=c1 logs 5 -f; echo "rc=$?"')
    assert out.returncode == 0, out.stderr
    assert "ยังไม่มี container c1" in out.stdout and "docker logs" not in out.stdout
    assert out.stdout.strip().endswith("rc=0")


def test_llamacpp_native_follows_the_file_name_and_docker_mode_the_container(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report(), engine=Engine.LLAMACPP)
    fn = _logs_fn(bundle.controller)
    log = tmp_path / "server.log"
    log.write_text("x\n", encoding="utf-8")

    out = run_bash(fn + f'RUNTIME_MODE=native LOG_FILE={log} logs 40 -f; echo "rc=$?"')
    # -F ไม่ใช่ -f: ตามชื่อไฟล์ — restart เขียนไฟล์ใหม่แล้วยังตามต่อได้
    assert out.stdout.strip().splitlines() == [f"tail -n 40 -F {log}", "rc=0"], out.stderr
    out = run_bash(fn + f"RUNTIME_MODE=native LOG_FILE={log} logs 40")
    assert out.stdout.strip() == f"tail -n 40 {log}"

    missing = tmp_path / "none.log"
    out = run_bash(fn + f'RUNTIME_MODE=native LOG_FILE={missing} logs -f; echo "rc=$?"')
    assert out.returncode == 0 and "ยังไม่มีไฟล์ log" in out.stdout and "tail" not in out.stdout

    out = run_bash(fn + 'RUNTIME_MODE=docker CONTAINER_NAME=c9 logs 7 -f')
    assert out.stdout.strip() == "docker logs --tail 7 -f c9"
    out = run_bash(fn + NO_CONTAINER + 'RUNTIME_MODE=docker CONTAINER_NAME=c9 logs -f; echo "rc=$?"')
    assert "ยังไม่มี container c9" in out.stdout and out.stdout.strip().endswith("rc=0")


def test_stacked_follows_head_or_every_worker_over_ssh(tmp_path):
    bundle = _stacked_bundle(tmp_path)
    fn = _logs_fn(bundle.controller)
    env = "MASTER_CONTAINER=h WORKER_CONTAINER=w SLUG=s"

    out = run_bash(fn + f'{env} logs 30 -f; echo "rc=$?"')
    assert out.stdout.strip().splitlines() == ["docker logs --tail 30 -f h", "rc=0"], out.stderr
    out = run_bash(fn + f"{env} logs head -f")
    assert out.stdout.strip() == "docker logs --tail 200 -f h"
    # รูปเดิมยังใช้ได้: logs N / logs head N / logs worker N
    assert run_bash(fn + f"{env} logs 15").stdout.strip() == "docker logs --tail 15 h"
    assert run_bash(fn + f"{env} logs head 15").stdout.strip() == "docker logs --tail 15 h"
    out = run_bash(fn + f'{env} WORKER_IPS="10.0.0.2" logs worker 15')
    assert out.stdout.strip().splitlines() == ["===== worker 10.0.0.2 =====",
                                                "ssh_at 10.0.0.2 docker logs --tail '15' 'w' 2>&1"]

    # worker เดียว: ssh docker logs -f ตรง ๆ · หลาย worker: ใส่ [ip] นำหน้าฝั่งโน้น (sed -u ไม่ buffer)
    out = run_bash(fn + f'{env} WORKER_IPS="10.0.0.2" logs worker 25 -f; echo "rc=$?"')
    lines = out.stdout.strip().splitlines()
    assert "ssh_at 10.0.0.2 docker container inspect 'w' >/dev/null 2>&1" in lines
    assert "ssh_at 10.0.0.2 docker logs --tail '25' -f 'w' 2>&1" in lines and lines[-1] == "rc=0"
    out = run_bash(fn + f'{env} WORKER_IPS="10.0.0.2 10.0.0.3" logs worker -f')
    assert "ssh_at 10.0.0.3 docker logs --tail '200' -f 'w' 2>&1 | sed -u 's/^/[10.0.0.3] /'" in out.stdout

    # worker ยังไม่มี container (head ยังไม่ได้สั่ง start) — บอกแล้วจบ 0 ไม่ค้าง
    no_worker = 'ssh_at() { if [[ "$*" == *"container inspect"* ]]; then return 1; fi; echo "ssh_at $*"; }\n'
    out = run_bash(fn + no_worker + f'{env} WORKER_IPS="10.0.0.2" logs worker -f; echo "rc=$?"')
    assert out.returncode == 0 and "ยังไม่มี container w บน worker 10.0.0.2" in out.stdout
    assert "docker logs" not in out.stdout

    out = run_bash(fn + f"{env} logs sideways")
    assert out.returncode == 1 and "head | worker" in out.stderr


# ───────────────────────── controller: SIGTERM really reaches the child ─────────────────────────
def _sleeping_docker(tmp_path):
    """docker ปลอมที่ `logs` ค้างเหมือน docker logs -f จริง — พิมพ์ pid ของตัวเองให้เทสตามไปดูว่าตายจริง"""
    return _fake_bin(tmp_path, docker='''
        case "$1" in
          logs) echo "docker $*"; echo "pid $$"; exec sleep 60 ;;
          *)    exit 0 ;;
        esac
    ''')


@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG])
def test_sigterm_to_the_controller_kills_docker_logs(tmp_path, engine):
    """ทางเดินจริงผ่าน dispatch: `controller logs 5 -f` → TERM (สิ่งที่ hub ส่งตอนผู้ใช้ปิดแผง) →
    docker logs ข้างในต้องตายด้วย ไม่ใช่ถูกทิ้งเป็นกำพร้าจนกว่าจะมีบรรทัดใหม่"""
    bundle = _bundle(tmp_path, _safetensors_report(), engine=engine)
    fake = _sleeping_docker(tmp_path)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "RUN_DIR": str(tmp_path / "run")}
    proc = subprocess.Popen(["bash", str(bundle.controller), "logs", "5", "-f"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    first = proc.stdout.readline().strip()
    assert first.startswith("docker logs --tail 5 -f "), first
    child = int(proc.stdout.readline().split()[1])
    assert _pid_alive(child)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=5) == 0, "trap ต้องจบด้วย 0 — ผู้ใช้ปิดแผง ไม่ใช่ความผิดพลาด"
    assert _wait_gone(child), "docker logs -f ยังอยู่หลัง controller ตาย"


def test_sigterm_to_llamacpp_native_kills_tail(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report(), engine=Engine.LLAMACPP)
    fake = _fake_bin(tmp_path, tail='''
        echo "tail $*"; echo "pid $$"; exec sleep 60
    ''')
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "server.log").write_text("boot\n", encoding="utf-8")
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "RUN_DIR": str(run_dir),
           "RUNTIME_MODE": "native"}
    proc = subprocess.Popen(["bash", str(bundle.controller), "logs", "9", "-f"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    first = proc.stdout.readline().strip()
    assert first == f"tail -n 9 -F {run_dir / 'server.log'}", first
    child = int(proc.stdout.readline().split()[1])
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=5) == 0
    assert _wait_gone(child)


# ───────────────────────── fleet.follow_argv ─────────────────────────
def test_follow_argv_prefers_a_controller_that_can_follow_and_falls_back_otherwise(tmp_path):
    from lmds.fleet import FleetError, follow_argv
    from lmds.fleet.manager import ServerInfo

    new = tmp_path / "new.sh"
    new.write_text("#!/bin/bash\nlogs() { case x in -f|--follow|follow) ;; esac; }\n", encoding="utf-8")
    new.chmod(0o755)
    info = ServerInfo(slug="m", controller=str(new), mode="docker", container="lmds-m", run_dir=tmp_path)
    assert follow_argv(info, 50) == [str(new), "logs", "50", "-f"]
    assert follow_argv(info, 50, worker=True) == [str(new), "logs", "worker", "50", "-f"]

    old = tmp_path / "old.sh"
    old.write_text('#!/bin/bash\nlogs() { docker logs --tail "${1:-300}" "$CONTAINER_NAME"; }\n', encoding="utf-8")
    old.chmod(0o755)
    info = ServerInfo(slug="m", controller=str(old), mode="docker", container="lmds-m", run_dir=tmp_path)
    assert follow_argv(info, 50) == ["docker", "logs", "-f", "--tail", "50", "lmds-m"]
    with pytest.raises(FleetError, match="worker"):
        follow_argv(info, 50, worker=True)

    (tmp_path / "server.log").write_text("", encoding="utf-8")
    info = ServerInfo(slug="m", controller=str(old), mode="native", run_dir=tmp_path)
    assert follow_argv(info, 50) == ["tail", "-n", "50", "-F", str(tmp_path / "server.log")]

    info = ServerInfo(slug="m", controller="", mode="native", run_dir=tmp_path / "nowhere")
    with pytest.raises(FleetError, match="realtime"):
        follow_argv(info)


# ───────────────────────── hub SSE: real server, real sockets ─────────────────────────
class LiveServer:
    """uvicorn ใน thread บนพอร์ตว่าง — ปิด socket ได้จริง (สิ่งที่ TestClient ทำไม่ได้)"""

    def __init__(self, app):
        import uvicorn

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self.server.started, "uvicorn ไม่ขึ้น"
        return self

    def __exit__(self, *_exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def open(self, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path)
        return conn, conn.getresponse()


def _read_until_end(resp, limit: float = 10.0) -> list[str]:
    lines, deadline = [], time.time() + limit
    while time.time() < deadline:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8").rstrip("\n")
        lines.append(line)
        if line.startswith("event: end"):
            lines.append(resp.readline().decode("utf-8").rstrip("\n"))
            break
    return lines


def _next_data(resp) -> str:
    """บรรทัด data: ถัดไป — ข้ามบรรทัดว่างที่คั่น frame และ comment"""
    while True:
        raw = resp.readline()
        assert raw, "สายจบก่อนจะมี data:"
        line = raw.decode("utf-8").rstrip("\n")
        if line.startswith("data: "):
            return line


def _controller(fleet_slug: str, tmp_path: Path, body: str) -> Path:
    """เขียนทับ controller ของ fixture ให้เล่นบท `logs N -f` — มี marker ของ follow ให้ follow_argv เลือกมัน"""
    controller = tmp_path / "bundles" / fleet_slug / f"{fleet_slug}-single.sh"
    controller.write_text("#!/usr/bin/env bash\n# -f|--follow|follow)\n" + textwrap.dedent(body), encoding="utf-8")
    controller.chmod(0o755)
    return controller


def test_local_stream_frames_lines_keepalive_and_end(fleet, tmp_path, monkeypatch):
    monkeypatch.setattr(logstream, "KEEPALIVE", 0.3)
    monkeypatch.setattr(logstream, "POLL", 0.1)
    _controller(fleet, tmp_path, '''
        [[ "$1 $2 $3" == "logs 25 -f" ]] || { echo "bad argv: $*"; exit 9; }
        echo "INFO loading weights"
        sleep 0.9
        printf 'ERROR CUDA out of memory\\r\\n'
        echo "$(head -c 5000 /dev/zero | tr '\\0' x)"
        exit 3
    ''')
    with LiveServer(create_app()) as srv:
        conn, resp = srv.open(f"/api/models/{fleet}/logs/stream?tail=25")
        assert resp.status == 200 and resp.getheader("content-type", "").startswith("text/event-stream")
        lines = _read_until_end(resp)
        conn.close()
    assert lines[0] == "retry: 2000", lines
    data = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and not l.startswith("data: {\"exit\"")]
    assert [d["line"] for d in data][:2] == ["INFO loading weights", "ERROR CUDA out of memory"], lines
    assert all(isinstance(d["ts"], float) for d in data)
    assert len(data[2]["line"]) == logstream.LINE_CAP, "บรรทัดยาวเกินต้องถูกตัด"
    assert ": keepalive" in lines, "เงียบ 0.9 วิกับ KEEPALIVE 0.3 ต้องมี comment กัน proxy ตัดสาย"
    assert lines[-2] == "event: end" and json.loads(lines[-1][6:]) == {"exit": 3}
    assert logstream.active() == 0


def test_closing_the_socket_kills_the_controller(fleet, tmp_path, monkeypatch):
    """ผู้ใช้ปิดแผง/แท็บ = child ต้องตาย — ไม่งั้น docker logs -f สะสมบน hub ทั้งวัน"""
    monkeypatch.setattr(logstream, "POLL", 0.1)
    _controller(fleet, tmp_path, '''
        echo "pid $$"
        exec sleep 60
    ''')
    with LiveServer(create_app()) as srv:
        conn, resp = srv.open(f"/api/models/{fleet}/logs/stream")
        assert resp.readline().decode().startswith("retry:")
        pid = json.loads(_next_data(resp)[6:])["line"].split()[1]
        assert logstream.active(fleet) == 1
        conn.close()
        assert _wait_gone(int(pid)), "controller ยังอยู่หลัง client ปิดสาย"
        deadline = time.time() + 5
        while logstream.active() and time.time() < deadline:
            time.sleep(0.05)
        assert logstream.active() == 0, "โควตาต้องคืนเมื่อสายปิด"


def test_stream_limits_answer_429_with_a_reason(fleet, tmp_path, monkeypatch):
    monkeypatch.setattr(logstream, "POLL", 0.1)
    monkeypatch.setattr(logstream, "MAX_PER_BUNDLE", 1)
    _controller(fleet, tmp_path, "echo hi; exec sleep 60\n")
    with LiveServer(create_app()) as srv:
        first, resp = srv.open(f"/api/models/{fleet}/logs/stream")
        assert resp.status == 200
        resp.readline()
        second, denied = srv.open(f"/api/models/{fleet}/logs/stream")
        assert denied.status == 429
        assert "เพดาน 1 ต่อโมเดล" in json.loads(denied.read())["detail"]
        second.close()
        first.close()
        deadline = time.time() + 5
        while logstream.active() and time.time() < deadline:
            time.sleep(0.05)
        # โควตาคืนแล้ว = เปิดใหม่ได้
        third, again = srv.open(f"/api/models/{fleet}/logs/stream")
        assert again.status == 200
        third.close()

    monkeypatch.setattr(logstream, "MAX_PER_HUB", 0)
    with LiveServer(create_app()) as srv:
        conn, denied = srv.open(f"/api/models/{fleet}/logs/stream")
        assert denied.status == 429 and "hub ตาม log อยู่" in json.loads(denied.read())["detail"]
        conn.close()


def test_stream_takes_the_token_as_a_query_parameter_like_events(fleet, tmp_path):
    """EventSource ใส่ header ไม่ได้ — ต้องรับ ?token= เหมือน /api/events"""
    _controller(fleet, tmp_path, "echo one\n")
    with LiveServer(create_app(token="s3cret-token")) as srv:
        conn, resp = srv.open(f"/api/models/{fleet}/logs/stream")
        assert resp.status == 401
        conn.close()
        conn, resp = srv.open(f"/api/models/{fleet}/logs/stream?token=s3cret-token")
        assert resp.status == 200
        assert '"line": "one"' in "\n".join(_read_until_end(resp))
        conn.close()


def test_unknown_model_and_bad_slug_are_rejected_before_spawning(fleet):
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    assert client.get("/api/models/nope/logs/stream").status_code == 404
    assert client.get("/api/models/x%27%3Bid/logs/stream").status_code == 400
    assert client.get("/api/nodes/ghost/models/m/logs/stream").status_code == 404
    assert logstream.active() == 0


def test_node_stream_uses_the_held_stdin_ssh_path_not_run(registered, monkeypatch):
    """ทาง node ต้องเป็น stream(hold_stdin=True) + follow_wrap — nodes.run หมดเวลา 60 วิ ใช้ตามสายไม่ได้"""
    seen = {}

    def fake_stream(node, command, secret_env=None, stdin_text="", hold_stdin=False):
        seen.update(node=node.name, command=command, hold_stdin=hold_stdin)
        return logstream.local_process(["bash", "-c", "echo 'INFO worker up'; echo 'ERROR boom'"])

    def never_run(*_a, **_k):
        raise AssertionError("ห้ามใช้ nodes.run กับสายที่ไม่จบเอง")

    monkeypatch.setattr("lmds.nodes.stream", fake_stream)
    monkeypatch.setattr("lmds.nodes.run", never_run)
    with LiveServer(create_app()) as srv:
        conn, resp = srv.open("/api/nodes/spark2/models/big-model/logs/stream?tail=120&worker=1")
        assert resp.status == 200
        lines = _read_until_end(resp)
        conn.close()
    assert seen["node"] == "spark2" and seen["hold_stdin"] is True
    assert "bundles/big-model" in seen["command"]
    assert "logs worker 120 -f & p=$!; ( cat >/dev/null; kill $p 2>/dev/null ) >/dev/null 2>&1 & wait $p" in seen["command"]
    got = [json.loads(l[6:])["line"] for l in lines if l.startswith("data: {\"line\"")]
    assert got == ["INFO worker up", "ERROR boom"]
    assert lines[-2:] == ["event: end", 'data: {"exit": 0}']
    assert logstream.active() == 0


def test_node_stream_reports_a_spawn_failure_as_a_line_not_a_hang(registered, monkeypatch):
    from lmds.nodes import NodeError

    def broken(*_a, **_k):
        raise NodeError("ไม่พบคำสั่ง ssh — ติดตั้ง openssh-client ก่อน")

    monkeypatch.setattr("lmds.nodes.stream", broken)
    with LiveServer(create_app()) as srv:
        conn, resp = srv.open("/api/nodes/spark2/models/big-model/logs/stream")
        lines = _read_until_end(resp)
        conn.close()
    assert any("เปิดสตรีม log ไม่ได้" in l and "openssh-client" in l for l in lines), lines
    assert lines[-1] == 'data: {"exit": -1}' and logstream.active() == 0


def test_stop_never_kills_a_group_that_is_not_ours():
    """ssh ของ node ไม่ได้ตั้ง session เอง — killpg จะโดนกลุ่มของ hub · ต้องเป็น terminate ธรรมดา"""
    proc = subprocess.Popen(["sleep", "60"])      # pgid = ของ pytest ไม่ใช่ตัวมันเอง
    assert not logstream._own_group(proc)
    logstream.stop(proc, grace=2.0)
    assert proc.poll() is not None and os.getpid()  # ยังอยู่ = ไม่โดน killpg ตัวเอง
    own = logstream.local_process(["sleep", "60"])
    assert logstream._own_group(own)
    logstream.stop(own)
    assert own.poll() is not None


# ───────────────────────── CLI ─────────────────────────
def test_node_ctl_logs_follow_holds_stdin_wraps_the_verb_and_exits_cleanly_on_ctrl_c(monkeypatch):
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes import Node, add

    add(Node(name="n9", host="10.0.0.9", user="u"))
    seen = {}

    class Stdout:
        def __init__(self):
            self.n = 0

        def readline(self):
            self.n += 1
            if self.n == 1:
                return b"INFO first line\n"
            raise KeyboardInterrupt

    class Proc:
        stdout = Stdout()

        def terminate(self):
            seen["terminated"] = True

        def kill(self):
            seen["killed"] = True

        def wait(self):
            raise AssertionError("Ctrl-C แล้วต้องไม่รอ ssh จนจบ")

    def fake_stream(node, command, secret_env=None, hold_stdin=False):
        seen.update(command=command, hold_stdin=hold_stdin)
        return Proc()

    monkeypatch.setattr("lmds.nodes.stream", fake_stream)
    result = CliRunner().invoke(app, ["node", "ctl", "n9", "my-model", "logs", "-f"])
    assert result.exit_code == 0, result.output
    assert "INFO first line" in result.output
    assert seen["hold_stdin"] is True and seen["terminated"] is True
    assert "logs -f & p=$!; ( cat >/dev/null; kill $p 2>/dev/null ) >/dev/null 2>&1 & wait $p" in seen["command"]


def test_node_ctl_without_follow_keeps_the_plain_path(monkeypatch):
    """`node ctl … logs 200` (one-shot) ต้องไม่ถูกห่อ — เดิมของ hub/ปุ่ม logs-worker เรียกแบบนี้"""
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes import Node, add
    from tests.test_nodes import _Lines

    add(Node(name="n8", host="10.0.0.8", user="u"))
    seen = {}

    class Proc:
        stdout = _Lines([b"ok\n"])

        def wait(self):
            return 0

    def fake_stream(node, command, secret_env=None):
        seen["command"] = command
        return Proc()

    monkeypatch.setattr("lmds.nodes.stream", fake_stream)
    result = CliRunner().invoke(app, ["node", "ctl", "n8", "my-model", "logs", "200"])
    assert result.exit_code == 0, result.output
    assert seen["command"].endswith('"$ctl" logs 200') and "cat >/dev/null" not in seen["command"]


def test_lmds_logs_follow_runs_the_controller_follow_argv(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.fleet.manager import ServerInfo

    ctl = tmp_path / "m-single.sh"
    ctl.write_text("#!/bin/bash\n# -f|--follow|follow)\n", encoding="utf-8")
    ctl.chmod(0o755)
    info = ServerInfo(slug="m", controller=str(ctl), mode="docker", container="lmds-m", run_dir=tmp_path,
                      running=True)
    seen = {}

    def fake_run(argv, *a, **k):
        seen["argv"] = argv

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr("lmds.fleet.find", lambda slug: info if slug == "m" else None)
    monkeypatch.setattr("lmds.fleet.manager.subprocess.run", fake_run)
    result = CliRunner().invoke(app, ["logs", "m", "-f", "-n", "77"])
    assert result.exit_code == 0, result.output
    assert seen["argv"] == [str(ctl), "logs", "77", "-f"]
