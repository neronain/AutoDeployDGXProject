"""คอนโซลเว็บ: ต้องยืนยันตัวตนเสมอ + เก็บร่องรอยว่าใครสั่งอะไร

เดิม `lmds web` สุ่ม token ให้เฉพาะตอน bind ออก network · bind 127.0.0.1 ซึ่งเป็น
ค่าปริยายถูกปล่อยโล่ง (`require_token`: `if not token: return`) โดยถือว่า "เครื่องนี้
เท่านั้น" = ปลอดภัย ซึ่งไม่จริงสองทาง:

  * ผู้ใช้อื่นบนเครื่องเดียวกันเปิด localhost:8600 แล้วได้สิทธิ์ของ user ที่รัน hub
    ซึ่งคุม start/stop/ลบโมเดลได้ทุกเครื่องในทะเบียนผ่าน SSH
  * เพจใดก็ได้ที่ผู้ใช้เปิดในเบราว์เซอร์ยิง POST มาที่ 127.0.0.1 ได้ (CSRF/DNS rebinding)
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.cli.main import app  # noqa: E402
from lmds.web import audit, create_app  # noqa: E402

runner = CliRunner()
TOKEN = "s3cret-token-value"


@pytest.fixture
def audit_log(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "audit.log"
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(path))
    monkeypatch.delenv("LMDS_AUDIT", raising=False)
    return path


# ── auth ──────────────────────────────────────────────────────────────────────

def test_the_page_is_told_that_a_token_is_required(audit_log):
    """หน้าเว็บถาม /api/auth ก่อนวาด — ตอบผิดแปลว่ามันไม่ขึ้นหน้า login ให้"""
    assert TestClient(create_app(TOKEN)).get("/api/auth").json() == {"required": True}


def test_a_state_changing_call_without_a_token_is_refused(audit_log):
    client = TestClient(create_app(TOKEN))
    assert client.post("/api/auth").status_code == 401
    assert client.post("/api/auth", headers={"x-lmds-token": TOKEN}).status_code == 200


def test_the_shell_and_fonts_stay_open_so_the_login_page_can_render(audit_log):
    """หน้า HTML กับฟอนต์ไม่มีข้อมูล — ต้องโหลดได้ ไม่งั้นไม่มีที่ให้กรอก token"""
    client = TestClient(create_app(TOKEN))
    assert client.get("/").status_code == 200
    assert client.get("/api/auth").status_code == 200


# ── `lmds web` ต้องตั้ง token ให้เสมอ ไม่ใช่เฉพาะตอนเปิดออก network ──────────────────

def _web_sandbox(monkeypatch, tmp_path):
    """ตัดทุกอย่างที่แตะเครื่องจริงออก เหลือแค่การตัดสินใจเรื่อง token กับการ launch"""
    from lmds.web import daemon

    monkeypatch.setattr(daemon, "token_file", lambda: tmp_path / "web-token")
    monkeypatch.setattr(daemon, "state_file", lambda: tmp_path / "web.json")
    monkeypatch.setattr(daemon, "log_file", lambda: tmp_path / "web.log")
    monkeypatch.setattr(daemon, "running", lambda: None)
    monkeypatch.setattr(daemon, "service_active", lambda: False)
    monkeypatch.setattr(daemon, "port_busy", lambda host, port, timeout=0.4: False)
    monkeypatch.setattr(daemon, "wait_until_serving", lambda bind, port, pid: True)
    monkeypatch.setattr(daemon, "write_state", lambda *a, **k: None)
    # เทสที่อยากจำลอง env ของ unit เดิมจะตั้งเองทีหลัง — ที่นี่แค่ล้างของเครื่องที่รันเทส
    monkeypatch.delenv("LMDS_WEB_TOKEN", raising=False)
    return daemon


def _launch(monkeypatch, tmp_path, args: list[str]):
    """เรียก `lmds web -b` โดยดัก Popen — คืน (argv, env) ที่ถูกส่งให้ลูก"""
    import subprocess

    daemon = _web_sandbox(monkeypatch, tmp_path)
    seen: dict = {}

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = runner.invoke(app, ["web", "-b", *args])
    return result, seen, daemon


def test_binding_only_loopback_still_gets_a_token(monkeypatch, tmp_path, audit_log):
    """นี่คือหัวใจของข้อนี้ — ค่าปริยายเคยเปิดโล่ง"""
    result, seen, daemon = _launch(monkeypatch, tmp_path, [])
    assert result.exit_code == 0, result.output
    assert daemon.remembered_token(), "ต้องมี token ถูกสร้างและจำไว้"
    assert seen["env"]["LMDS_WEB_TOKEN"] == daemon.remembered_token()


def test_the_token_is_handed_over_in_the_environment_never_on_the_command_line(monkeypatch, tmp_path, audit_log):
    """`ps` อ่าน argv ของทุก process บนเครื่องได้ · /proc/<pid>/environ อ่านได้เฉพาะเจ้าของกับ root
    — เหตุผลเดียวกับที่ controller ส่ง key ของโมเดลทาง --api-key-file"""
    result, seen, daemon = _launch(monkeypatch, tmp_path, [])
    assert result.exit_code == 0, result.output
    token = daemon.remembered_token()
    assert token and token not in " ".join(seen["argv"])
    assert "--token" not in seen["argv"]


def test_no_auth_is_possible_but_has_to_be_asked_for_and_says_so_loudly(monkeypatch, tmp_path, audit_log):
    result, seen, daemon = _launch(monkeypatch, tmp_path, ["--no-auth"])
    assert result.exit_code == 0, result.output
    assert "--no-auth" in seen["argv"], "ลูกต้องรู้ว่าตั้งใจเปิดโล่ง ไม่ใช่แค่ไม่มี token มาให้"
    assert "LMDS_WEB_TOKEN" not in seen["env"]
    assert not daemon.remembered_token(), "--no-auth ต้องไม่ไปสร้าง token ค้างไว้"
    assert "ไม่ต้องยืนยันตัวตน" in (result.stdout + (result.stderr or ""))


def test_no_auth_together_with_a_token_is_a_contradiction_and_is_refused(monkeypatch, tmp_path, audit_log):
    _web_sandbox(monkeypatch, tmp_path)
    result = runner.invoke(app, ["web", "--no-auth", "--token", "some-token-value"])
    assert result.exit_code == 1
    assert "เลือกอย่างใดอย่างหนึ่ง" in (result.stdout + (result.stderr or ""))


def test_a_remembered_token_is_reused_so_bookmarks_keep_working(monkeypatch, tmp_path, audit_log):
    daemon = _web_sandbox(monkeypatch, tmp_path)
    daemon.remember_token("kept-across-restarts")
    _, seen, _ = _launch(monkeypatch, tmp_path, [])
    assert seen["env"]["LMDS_WEB_TOKEN"] == "kept-across-restarts"


def test_files_holding_the_token_are_never_world_readable_even_for_an_instant(tmp_path, monkeypatch):
    """เขียนก่อนแล้ว chmod ทีหลังเปิดช่องระหว่างสองขั้น — ไฟล์เกิดมาด้วย umask ปกติ"""
    from lmds.web import daemon

    target = tmp_path / "web-token"
    monkeypatch.setattr(daemon, "token_file", lambda: target)
    daemon.remember_token("a-token-value")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert daemon.remembered_token() == "a-token-value"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "ไฟล์ชั่วคราวต้องไม่ค้าง"


# ── audit ─────────────────────────────────────────────────────────────────────

def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_state_changing_call_leaves_a_trace(audit_log):
    client = TestClient(create_app(TOKEN))
    client.post("/api/auth", headers={"x-lmds-token": TOKEN})
    entries = _entries(audit_log)
    assert entries and entries[-1]["method"] == "POST" and entries[-1]["path"] == "/api/auth"
    assert entries[-1]["status"] == 200 and "at" in entries[-1] and "ip" in entries[-1]


def test_refused_calls_are_recorded_whatever_the_method(audit_log):
    """การไล่เดา token คือสิ่งที่ audit มีไว้ให้เห็น — GET ที่ถูกปฏิเสธก็ต้องถูกเก็บ"""
    client = TestClient(create_app(TOKEN))
    client.get("/api/version")
    client.get("/api/version", headers={"x-lmds-token": "wrong"})
    statuses = [e["status"] for e in _entries(audit_log)]
    assert statuses == [401, 401]


def test_a_successful_read_is_not_recorded(audit_log):
    """หน้าเว็บ poll สถานะตลอดเวลา — เก็บทุก GET ที่ผ่าน = ไฟล์เต็มไปด้วยเสียงรบกวน"""
    client = TestClient(create_app(TOKEN))
    assert client.get("/api/version", headers={"x-lmds-token": TOKEN}).status_code == 200
    assert not audit_log.exists() or not _entries(audit_log)


def test_the_token_never_reaches_the_audit_file(audit_log):
    """`require_token` รับ token ทาง ?token= ได้ — เก็บ query string = เขียน token ลงไฟล์"""
    client = TestClient(create_app(TOKEN))
    client.post(f"/api/auth?token={TOKEN}")
    text = audit_log.read_text(encoding="utf-8")
    assert TOKEN not in text
    assert "/api/auth" in text


def test_the_audit_file_is_readable_only_by_its_owner(audit_log):
    TestClient(create_app(TOKEN)).post("/api/auth")
    assert stat.S_IMODE(audit_log.stat().st_mode) == 0o600


def test_the_file_is_rotated_instead_of_growing_without_end(audit_log, monkeypatch):
    monkeypatch.setattr(audit, "MAX_BYTES", 200)
    for _ in range(12):
        audit.record("POST", "/api/models/demo/start", ip="10.0.0.5", status=200, ms=3)
    assert audit_log.with_name(audit_log.name + ".1").is_file()
    assert audit_log.stat().st_size < 4096


def test_audit_can_be_turned_off(audit_log, monkeypatch):
    monkeypatch.setenv("LMDS_AUDIT", "0")
    TestClient(create_app(TOKEN)).post("/api/auth")
    assert not audit_log.exists()


def test_a_disk_that_cannot_be_written_does_not_break_the_command(monkeypatch, tmp_path):
    """audit ที่หายไปหนึ่งบรรทัดแย่กว่าไม่มี audit นิดเดียว · คำสั่ง start ที่ล้มเพราะ
    เขียน log ไม่ได้แย่กว่ามาก"""
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "nope" / "a.log"))
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    audit.record("POST", "/api/models/demo/start", ip="10.0.0.5", status=200)   # ต้องไม่โยน


# ── `lmds audit` ──────────────────────────────────────────────────────────────

def test_the_cli_shows_the_trail_and_points_at_refusals(audit_log):
    audit.record("POST", "/api/models/demo/start", ip="10.0.0.5", status=200, ms=12)
    audit.record("POST", "/api/models/demo/stop", ip="10.9.9.9", status=401, ms=1)

    result = runner.invoke(app, ["audit"], env={"COLUMNS": "300"})
    assert result.exit_code == 0
    assert "10.0.0.5" in result.stdout and "/api/models/demo/start" in result.stdout
    assert "ถูกปฏิเสธ 1 รายการ" in result.stdout


def test_the_cli_can_show_only_what_was_refused(audit_log):
    for _ in range(5):
        audit.record("POST", "/api/models/demo/start", ip="10.0.0.5", status=200)
    audit.record("POST", "/api/models/demo/stop", ip="10.9.9.9", status=401)

    result = runner.invoke(app, ["audit", "--failed", "--json"], env={"COLUMNS": "300"})
    rows = [json.loads(line) for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert [r["status"] for r in rows] == [401]


def test_the_cli_says_where_the_file_is_when_there_is_nothing_yet(audit_log):
    result = runner.invoke(app, ["audit"], env={"COLUMNS": "300"})
    assert result.exit_code == 0 and "ยังไม่มีรายการ" in result.stdout
    assert audit_log.name in result.stdout


def test_an_old_service_unit_with_an_empty_token_closes_itself_on_the_next_restart(monkeypatch, tmp_path, audit_log):
    """ทางอัปเกรดจริงของเครื่องที่เปิดคอนโซลไว้แล้ว

    unit ที่สร้างไว้ก่อนหน้านี้บน 127.0.0.1 มีบรรทัด `Environment=LMDS_WEB_TOKEN=` ว่าง
    ถ้าค่าว่างถูกนับว่า "ตั้งมาแล้ว" service จะกลับมาเปิดโล่งต่อหลัง restart โดยไม่มีใครรู้
    """
    import subprocess

    daemon = _web_sandbox(monkeypatch, tmp_path)
    seen: dict = {}

    class FakeProc:
        pid = 99

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen",
                        lambda argv, **kw: (seen.update(argv=list(argv), env=dict(kw.get("env") or {})), FakeProc())[1])
    monkeypatch.setenv("LMDS_WEB_TOKEN", "")          # เหมือน unit เดิมเป๊ะ
    result = runner.invoke(app, ["web", "-b"])
    assert result.exit_code == 0, result.output

    minted = daemon.remembered_token()
    assert minted, "ค่าว่างต้องไม่ถูกนับว่าเป็น token ที่ตั้งมาแล้ว"
    assert seen["env"]["LMDS_WEB_TOKEN"] == minted
    assert "--no-auth" not in seen["argv"]
