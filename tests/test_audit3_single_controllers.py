"""Audit รอบ 3 (2026-10-06) — single controller ทั้งสาม: สิ่งที่ controller *รายงาน* ต้องตรงกับของจริง

ทุกข้อรัน controller ที่ render แล้วทั้งไฟล์ใต้ bash กับ docker ปลอมที่ **รัน engine ปลอมจริง** (HTTP server ที่ฟังตาม
--host/--port ที่ controller สั่ง — ดู tests/single_controller_harness.py) · ไม่มีข้อไหนยืนยันด้วยข้อความในซอร์ส
"""

from __future__ import annotations

import json

import pytest

from tests.single_controller_harness import KINDS, Box, free_port, render, specific_ip, write_exe


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
    started = box.run("start", *flags, env={"HEALTH_TIMEOUT": "12"})
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
