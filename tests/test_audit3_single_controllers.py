"""Audit รอบ 3 (2026-10-06) — single controller ทั้งสาม: สิ่งที่ controller *รายงาน* ต้องตรงกับของจริง

ทุกข้อรัน controller ที่ render แล้วทั้งไฟล์ใต้ bash กับ docker ปลอมที่ **รัน engine ปลอมจริง** (HTTP server ที่ฟังตาม
--host/--port ที่ controller สั่ง — ดู tests/single_controller_harness.py) · ไม่มีข้อไหนยืนยันด้วยข้อความในซอร์ส
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from tests.single_controller_harness import KINDS, Box, free_port, pid_alive, render, specific_ip, write_exe


@pytest.fixture
def box(tmp_path, request):
    """Box ของ engine ที่เทสขอผ่าน parametrize("kind") — เก็บกวาด process ปลอมทุกตัวตอนจบ"""
    kind = request.getfixturevalue("kind")
    made = Box(tmp_path, kind, render(tmp_path, kind))
    try:
        yield made
    finally:
        made.close()


# ═════════════════════ 2. --bind <IP เฉพาะ>: ทุกคำสั่งต้องคุยกับที่อยู่ที่เซิร์ฟเวอร์ฟังจริง ═════════════════════
@pytest.mark.parametrize("kind", KINDS)
def test_start_with_a_specific_bind_address_sees_the_server_it_just_started(box, kind):
    """`--bind <ip>` ส่ง `--host <ip>` ให้ engine → ไม่มีใครฟัง 127.0.0.1 · เดิม start ยิง /health ที่ 127.0.0.1 จนครบ
    HEALTH_TIMEOUT (600–7200 วิ) แล้วจบ exit 1 ทั้งที่เซิร์ฟเวอร์ขึ้นแล้ว · status/info/test-text ก็บอกว่า down"""
    ip, port = specific_ip(), free_port()
    flags = ["--bind", ip, "--port", str(port)]
    started = box.run("start", *flags)
    assert started.returncode == 0, started.stdout + started.stderr
    assert "started:" in started.stdout
    assert f"listening {ip}:{port}" in box.calls(), "engine ปลอมต้องถูกสั่งให้ผูก IP นั้นจริง"

    status = box.run("status", *flags)
    assert "API: healthy" in status.stdout, status.stdout + status.stderr
    info = box.run("info", *flags)
    assert "RUNNING" in info.stdout, info.stdout
    assert box.run("wait-health", *flags).returncode == 0
    text = box.run("test-text", *flags)
    assert text.returncode == 0 and "test-text: OK" in text.stdout, text.stdout + text.stderr

    stopped = box.run("stop", *flags)
    assert stopped.returncode == 0 and "stopped" in stopped.stdout, stopped.stdout + stopped.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_a_saved_bind_address_is_used_by_every_later_command(box, kind):
    """ค่าที่ `lmds set --bind` บันทึกไว้ (API_HOST ใน bundle.env) — คำสั่งถัดไปเรียก controller เปล่า ๆ ไม่มี flag"""
    ip, port = specific_ip(), free_port()
    (box.bundle.directory / "bundle.env").write_text(f"API_HOST={ip}\nAPI_PORT={port}\n", encoding="utf-8")
    box.serve(port, host=ip)
    status = box.run("status")
    assert "API: healthy" in status.stdout, status.stdout + status.stderr
    assert "RUNNING" in box.run("info").stdout


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_llamacpp_status_finds_a_server_started_with_a_one_off_bind(box, kind):
    """`start --bind <ip>` ครั้งเดียว (ไม่ได้บันทึก) แล้ว `status` เปล่า ๆ — เป็นคนละ process · port/ชื่อถูกอ่านกลับจาก
    server.meta อยู่แล้ว ที่อยู่ที่ผูกก็ต้องตามมาด้วย ไม่งั้นไปถาม 127.0.0.1 ที่ไม่มีใครฟัง"""
    ip, port = specific_ip(), free_port()
    started = box.run("start", "--bind", ip, "--port", str(port))
    assert started.returncode == 0, started.stdout + started.stderr
    status = box.run("status")
    assert "process: running" in status.stdout and "API: healthy" in status.stdout, status.stdout + status.stderr
    assert box.run("stop").returncode == 0


_LOG_CURL = 'for a in "$@"; do case "$a" in http://*) echo "curl $a" >> "$FAKE_LOG" ;; esac; done; exit 7\n'


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bind, host", [
    ("10.1.1.1", "10.1.1.1"),
    ("::", "127.0.0.1"),            # ฟังทุกการ์ดแบบ dual-stack
    ("::1", "[::1]"),               # IPv6 เฉพาะ — URL ต้องมีวงเล็บ ไม่งั้น curl อ่าน ::1:8000 ไม่ออก
    ("fd00::5", "[fd00::5]"),
    ("localhost", "localhost"),     # ไม่เดาว่า localhost คือ v4 — ให้ client resolve ชื่อเดียวกับ engine
])
def test_the_probe_url_is_built_from_the_bind_address(box, kind, bind, host):
    write_exe(box.bin / "curl", _LOG_CURL)
    box.run("status", "--bind", bind, "--port", "18123")
    assert f"curl http://{host}:18123/health" in box.calls(), box.calls()


_TIMEOUT_EXPIRES = 'echo "timeout $*" >> "$FAKE_LOG"; exit 124\n'


@pytest.mark.parametrize("kind", KINDS)
def test_the_port_check_of_a_bind_address_has_a_deadline(box, kind):
    """`/dev/tcp/<ip>/<port>` ของ bash ไม่มีเพดานเวลา: ที่อยู่ที่ทิ้ง SYN เงียบ ๆ ค้างจน kernel เลิกลอง (~2 นาที)

    เคสจริง 2026-10-06 → 2026-10-09: CI บน main แดง 5 commit ติด เพราะเทสข้างบนข้อ `10.1.1.1` หมดเวลา
    (subprocess.TimeoutExpired) บน runner ที่บังเอิญอยู่ในวง 10.1.x.x ซึ่งไม่ตอบ SYN ไป 10.1.1.1 — runner อื่นตอบ
    "ไปไม่ถึง" ทันทีจึงผ่าน · เครื่องจริงเจอแบบเดียวกันเมื่อ `--bind` ชี้ IP ของการ์ดที่สายหลุด/ย้ายวง:
    `status` เงียบไป 2 นาทีต่อการถามหนึ่งครั้ง · ที่นี่ `timeout` เป็นตัวปลอมที่ "หมดเวลา" ทันที —
    ดูว่า controller ถามพอร์ตผ่านมันด้วยเพดานไม่กี่วินาที และ status ยังจบพร้อมคำตอบ
    """
    write_exe(box.bin / "curl", _LOG_CURL)
    write_exe(box.bin / "timeout", _TIMEOUT_EXPIRES)
    done = box.run("status", "--bind", "10.1.1.1", "--port", "18123")
    asked = [line.split() for line in box.calls().splitlines()
             if line.startswith("timeout ") and "/dev/tcp" in line and line.endswith("10.1.1.1 18123")]
    assert asked, f"ถามพอร์ตโดยไม่มีเพดานเวลา:\n{box.calls()}\n{done.stdout}{done.stderr}"
    assert all(0 < float(words[1]) <= 5 for words in asked), asked
    assert "healthy" not in done.stdout, done.stdout


@pytest.mark.parametrize("kind", KINDS)
def test_the_advertised_endpoint_follows_a_specific_bind(box, kind):
    """ผูก IP เฉพาะแล้ว Endpoint/base_url ที่พิมพ์ให้ client ต้องเป็น IP นั้น — IP ที่เดาจาก default route อาจเป็นการ์ดอื่น
    ที่ไม่มีใครฟัง (`--bind 127.0.0.1` แล้วโฆษณา IP ของ LAN = ชี้ไปประตูที่ล็อก) · --advertise-ip ที่ผู้ใช้สั่งเองยังชนะ"""
    auto = {"ADVERTISE_IP": ""}
    info = box.run("network-info", "--bind", "10.2.1.195", env=auto)
    assert "http://10.2.1.195:8000/v1" in info.stdout, info.stdout + info.stderr
    config = json.loads(box.run("client-config", "--bind", "127.0.0.1", env=auto).stdout)
    assert config["base_url"] == "http://127.0.0.1:8000/v1"
    explicit = box.run("network-info", "--bind", "10.2.1.195", "--advertise-ip", "203.0.113.7", env=auto)
    assert "http://203.0.113.7:8000/v1" in explicit.stdout


@pytest.mark.parametrize("kind", KINDS)
def test_the_default_bind_still_probes_loopback(box, kind):
    """0.0.0.0 (ค่าตั้งต้น) ฟังทุกการ์ด — controller ต้องยังถามที่ 127.0.0.1 ไม่ใช่ยิงไปที่ 0.0.0.0"""
    port = free_port()
    box.serve(port)
    for bind in ("0.0.0.0", "127.0.0.1", ""):
        status = box.run("status", "--port", str(port), *(["--bind", bind] if bind else []))
        assert "API: healthy" in status.stdout, bind + status.stdout + status.stderr


# ═════════════════════ 1. llama.cpp native: server.pid ค้าง ต้องไม่กลายเป็นใบสั่งฆ่า process อื่น ═════════════════════
def _native_run_dir(box):
    box.run_dir.mkdir(parents=True, exist_ok=True)
    return box.run_dir


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_a_stale_pid_file_never_gets_an_unrelated_process_killed(box, kind):
    """server.pid อยู่ใต้ ~/.lmds/run/<slug>/ จึงรอดข้าม reboot/ไฟดับ/OOM-kill · หลัง reboot เลข PID เริ่มนับใหม่ และ unit
    autostart รัน `stop` ก่อน `start` ทุกครั้ง (ExecStartPre) — PID เก่าตกเป็นของ process อื่นได้จริง · เดิม status บอก
    "process: running" แล้ว stop ส่ง SIGTERM/SIGKILL ใส่มันพร้อมพิมพ์ "stopped" rc 0"""
    victim = subprocess.Popen(["sleep", "300"])          # process ของ user เดียวกันที่ได้เลข PID นั้นไปใช้
    try:
        run_dir = _native_run_dir(box)
        (run_dir / "server.pid").write_text(f"{victim.pid}\n", encoding="utf-8")
        (run_dir / "server.meta").write_text(f"model={box.model}\nport=8000\n", encoding="utf-8")

        status = box.run("status")
        assert "process: stopped" in status.stdout and "process: running" not in status.stdout, status.stdout
        assert str(victim.pid) in status.stderr and "server.pid" in status.stderr, "ต้องบอกว่า pid file ค้างและเป็นของใคร"
        assert not (run_dir / "server.pid").exists(), "pid file ค้างต้องถูกลบ — hub อ่านไฟล์นี้ตัดสินว่ารันอยู่"

        (run_dir / "server.pid").write_text(f"{victim.pid}\n", encoding="utf-8")
        stopped = box.run("stop")
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert "ไม่มีอะไรให้หยุด" in stopped.stdout and "stopped " not in stopped.stdout, stopped.stdout
        time.sleep(0.5)
        assert victim.poll() is None, "stop ฆ่า process ที่ไม่ใช่ llama-server ของ bundle นี้"
        assert not (run_dir / "server.pid").exists()
    finally:
        victim.kill()
        victim.wait(timeout=10)


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_a_stale_pid_that_now_belongs_to_another_bundles_llama_server_is_left_alone(box, kind):
    """ตัวที่น่าจะชนที่สุดหลัง reboot: llama-server ของ bundle ข้าง ๆ ที่ autostart ขึ้นก่อน — ไบนารีเดียวกันเป๊ะ
    ต่างกันแค่ port/ชื่อที่ถูกสั่ง · "เป็น llama-server" อย่างเดียวจึงยังไม่พอ"""
    other = box.serve(free_port(), model="some-other-model", ours=False)
    run_dir = _native_run_dir(box)
    (run_dir / "server.pid").write_text(f"{other.pid}\n", encoding="utf-8")
    (run_dir / "server.meta").write_text(f"model={box.model}\nport=8000\n", encoding="utf-8")
    assert "process: stopped" in box.run("status").stdout
    (run_dir / "server.pid").write_text(f"{other.pid}\n", encoding="utf-8")
    stopped = box.run("stop")
    assert stopped.returncode == 0 and "ไม่มีอะไรให้หยุด" in stopped.stdout, stopped.stdout + stopped.stderr
    time.sleep(0.5)
    assert other.poll() is None, "stop ฆ่า llama-server ของ bundle อื่น"


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_start_is_not_refused_by_a_stale_pid_file(box, kind):
    """เดิม: PID ค้างที่ยังมี process อื่นใช้อยู่ → "เซิร์ฟเวอร์รันอยู่แล้ว — stop ก่อน" แล้ว stop ก็ไปฆ่าตัวนั้น"""
    victim = subprocess.Popen(["sleep", "300"])
    try:
        run_dir = _native_run_dir(box)
        (run_dir / "server.pid").write_text(f"{victim.pid}\n", encoding="utf-8")
        port = free_port()
        started = box.run("start", "--port", str(port))
        assert started.returncode == 0, started.stdout + started.stderr
        assert "รันอยู่แล้ว" not in started.stderr
        new_pid = int((run_dir / "server.pid").read_text().strip())
        assert new_pid != victim.pid and pid_alive(new_pid)
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait(timeout=10)


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_the_real_server_is_still_recognised_and_stopped(box, kind):
    """ด่านใหม่ต้องไม่ทำให้ของจริงหยุดไม่ได้: start (port/ชื่อไม่ใช่ค่าตั้งต้น) → status เห็น → stop เปล่า ๆ ฆ่าตัวนั้นจริง"""
    port = free_port()
    started = box.run("start", "--port", str(port), "--name", "renamed-model")
    assert started.returncode == 0, started.stdout + started.stderr
    pid = int((box.run_dir / "server.pid").read_text().strip())
    status = box.run("status")
    assert f"process: running (PID {pid})" in status.stdout, status.stdout + status.stderr
    stopped = box.run("stop")
    assert stopped.returncode == 0 and "stopped renamed-model" in stopped.stdout, stopped.stdout + stopped.stderr
    deadline = time.monotonic() + 10
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not pid_alive(pid), "llama-server ของ bundle นี้ยังอยู่หลัง stop"
    assert not (box.run_dir / "server.pid").exists()


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_a_server_started_by_an_older_controller_can_still_be_stopped(box, kind):
    """rollout: controller ถูกเขียนทับขณะเซิร์ฟเวอร์ยังรัน — server.pid/server.meta เป็นของรุ่นก่อน (ไม่มีอะไรใหม่ให้เทียบ)
    ต้องยังจำตัวเองได้จาก argv ที่รุ่นเก่าก็ส่งเหมือนกัน (--alias/--port) ไม่งั้นทั้งฟลีต stop ไม่ได้หลังอัปเดต"""
    port = free_port()
    server = box.serve(port)              # เขียน server.pid + server.meta แบบที่ start รุ่นก่อนเขียน
    assert f"process: running (PID {server.pid})" in box.run("status").stdout
    stopped = box.run("stop")
    assert stopped.returncode == 0 and "stopped " in stopped.stdout, stopped.stdout + stopped.stderr
    assert server.wait(timeout=10) is not None


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_a_pid_file_of_a_dead_process_is_cleaned_up_quietly(box, kind):
    dead = subprocess.Popen(["true"])
    dead.wait()
    run_dir = _native_run_dir(box)
    (run_dir / "server.pid").write_text(f"{dead.pid}\n", encoding="utf-8")
    status = box.run("status")
    assert "process: stopped" in status.stdout and "server.pid" not in status.stderr, status.stdout + status.stderr
    assert not (run_dir / "server.pid").exists()
