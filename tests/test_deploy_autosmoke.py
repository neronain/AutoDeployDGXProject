"""`lmds deploy --smoke` — พิสูจน์ว่า bundle *รัน* ได้ ไม่ใช่แค่ผ่าน gate แบบ static

ROADMAP เฟส 2 ข้อ 5: "บั๊กที่เจ็บที่สุดทุกตัวของรอบ 0.2.0 ผ่าน gate แบบ static ทั้งหมด
แล้วไปตายตอนรันจริง" — งานต่อยอดคือให้รันอัตโนมัติหลัง deploy

เลือกเป็น **opt-in** ไม่ใช่ opt-out:
  · smoke โหลด weight จริงหลายสิบ GB — คนที่แค่อยากได้ bundle ไปวางบนเครื่องอื่นจะรอเป็นชั่วโมง
  · `deploy` เป็นทางเดินที่ hub/หน้าเว็บ/สคริปต์เรียกด้วย `--yes` อยู่แล้ว การเปลี่ยนให้มันยาว
    และพึ่ง GPU+เน็ตโดยไม่มีใครขอ คือเปลี่ยนสัญญาของคำสั่งที่มีคนใช้อยู่
  · deploy บน hub มักสร้าง bundle ให้ *เครื่องอื่น* — smoke บน hub จึงผิดตั้งแต่ต้น
    (hub อาจไม่มี GPU ด้วยซ้ำ) การเปิดเป็นค่าเริ่มต้นทำให้ทางเดิน fleet พังทั้งเส้น
"""

import os
import stat
import types

from typer.testing import CliRunner

from lmds.cli import main as cli_main
from lmds.cli.main import app
from tests.test_generator import safetensors_report

runner = CliRunner()

DEPLOY = ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-single"]

# controller ปลอมที่รันได้จริง: จดคำสั่งที่ถูกเรียกลง calls.log แล้วล้มเฉพาะขั้นที่สั่งให้ล้ม
FAKE_CONTROLLER = """#!/bin/bash
echo "$1" >> "$(dirname "$0")/calls.log"
echo "ran $1 ok"
[ "$1" = "{fail_at}" ] && {{ echo "boom in $1" >&2; exit 7; }}
exit 0
"""


def _fake_bundle(tmp_path, monkeypatch, slug="demo", fail_at=""):
    """วาง controller ปลอมแล้วให้ `lmds.fleet.find` ชี้มาที่มัน — ไม่โหลดโมเดล ไม่แตะเน็ต"""
    directory = tmp_path / slug
    directory.mkdir(parents=True, exist_ok=True)
    controller = directory / f"{slug}-single.sh"
    controller.write_text(FAKE_CONTROLLER.format(fail_at=fail_at), encoding="utf-8")
    controller.chmod(controller.stat().st_mode | stat.S_IXUSR)
    server = types.SimpleNamespace(controller=str(controller))
    monkeypatch.setattr("lmds.fleet.find", lambda s: server if s == slug else None)
    return directory


def _calls(directory):
    log = directory / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)


# ── opt-in: ไม่ขอ = ไม่รัน ────────────────────────────────────────────────────

def test_deploy_without_the_flag_never_runs_smoke(isolated_config, tmp_path, monkeypatch):
    """สัญญาหลักของการเลือกแบบ opt-in — deploy เปล่า ๆ ต้องเร็วเท่าเดิมและไม่แตะเน็ต"""
    ran = []
    monkeypatch.setattr(cli_main, "_run_smoke", lambda *a, **k: ran.append(a) or "")
    _patch_inspect(monkeypatch, safetensors_report())

    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert ran == []
    assert "smoke" not in result.output


def test_deploy_with_the_flag_smokes_the_bundle_it_just_made(isolated_config, tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(cli_main, "_run_smoke", lambda slug, **k: ran.append(slug) or "")
    _patch_inspect(monkeypatch, safetensors_report())

    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--smoke"])
    assert result.exit_code == 0, result.output
    assert ran == ["qwen3-32b"]


def test_a_failed_smoke_gets_its_own_exit_code_not_the_gate_one(isolated_config, tmp_path, monkeypatch):
    """exit 2 ของ deploy แปลว่า "ไม่ผ่าน quality gates" = สคริปต์ผิดตั้งแต่ยังไม่รัน

    bundle ที่ผ่าน gate แต่รันจริงไม่ขึ้นเป็นคนละอาการคนละทางแก้ — ใช้รหัสเดียวกัน
    แปลว่าสคริปต์ที่เรียก deploy แยกสองเรื่องนี้ไม่ออก
    """
    monkeypatch.setattr(cli_main, "_run_smoke", lambda slug, **k: "start")
    _patch_inspect(monkeypatch, safetensors_report())

    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--smoke"])
    assert result.exit_code == 6, result.output
    # ZIP ที่ผ่าน gate ยังอยู่ — ของที่ทำเสร็จแล้วไม่ถูกลบเพราะขั้นพิสูจน์ล้ม
    assert (tmp_path / "qwen3-32b.zip").is_file()


def test_smoke_runs_on_the_single_bundle_only_when_both_were_built(
        isolated_config, tmp_path, monkeypatch):
    """stacked ต้องกรอก MASTER_IP/WORKER_IP + กุญแจ head→worker ก่อนถึงจะรันได้

    ยิง smoke ใส่มันทันทีหลัง generate = โหลด weight หลายสิบ GB ทิ้งแล้วไปล้มที่
    sync-worker ทุกครั้ง · ต้องข้ามพร้อมบอกเหตุผล ไม่ใช่ข้ามเงียบ ๆ
    """
    ran = []
    monkeypatch.setattr(cli_main, "_run_smoke", lambda slug, **k: ran.append(slug) or "")
    _patch_inspect(monkeypatch, safetensors_report())

    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes",
                                 "--also-stacked", "--smoke"])
    assert result.exit_code == 0, result.output
    assert ran == ["qwen3-32b"], "stacked ต้องไม่ถูก smoke"
    assert "qwen3-32b-stacked" in result.output and "cluster write" in result.output


# ── ตัวเดินขั้น (_run_smoke) — ใช้ร่วมกับคำสั่ง `lmds smoke` ─────────────────────

def test_run_smoke_walks_every_step_and_stops_the_server(isolated_config, tmp_path, monkeypatch):
    directory = _fake_bundle(tmp_path, monkeypatch)
    assert cli_main._run_smoke("demo") == ""
    assert _calls(directory) == ["download", "verify-files", "start", "test-text", "stop"]


def test_run_smoke_stops_the_server_even_when_a_step_blows_up(isolated_config, tmp_path, monkeypatch):
    """ของเดิมสัญญาไว้ว่า "หยุด server เสมอแม้ล้มกลางทาง" — การแยกฟังก์ชันต้องไม่ทำหาย"""
    directory = _fake_bundle(tmp_path, monkeypatch, fail_at="test-text")
    assert cli_main._run_smoke("demo") == "test-text"
    assert _calls(directory) == ["download", "verify-files", "start", "test-text", "stop"]


def test_run_smoke_keeps_the_server_when_asked(isolated_config, tmp_path, monkeypatch):
    directory = _fake_bundle(tmp_path, monkeypatch)
    assert cli_main._run_smoke("demo", keep=True) == ""
    assert "stop" not in _calls(directory)


def test_run_smoke_skips_the_download_pair_on_request(isolated_config, tmp_path, monkeypatch):
    directory = _fake_bundle(tmp_path, monkeypatch)
    assert cli_main._run_smoke("demo", skip_download=True) == ""
    assert _calls(directory) == ["start", "test-text", "stop"]


def test_smoke_command_still_exits_2_on_failure_and_0_on_success(
        isolated_config, tmp_path, monkeypatch):
    """สัญญาเดิมของ `lmds smoke` (exit 0 ผ่าน · 2 ล้ม) ต้องไม่เปลี่ยนเพราะการ refactor"""
    _fake_bundle(tmp_path, monkeypatch, slug="ok")
    assert runner.invoke(app, ["smoke", "ok"]).exit_code == 0

    _fake_bundle(tmp_path, monkeypatch, slug="bad", fail_at="start")
    result = runner.invoke(app, ["smoke", "bad"])
    assert result.exit_code == 2
    assert "ติดที่ 'start'" in result.output


def test_smoke_command_reports_an_unknown_bundle_as_bad_input(isolated_config, monkeypatch):
    monkeypatch.setattr("lmds.fleet.find", lambda s: None)
    result = runner.invoke(app, ["smoke", "nope"])
    assert result.exit_code == 1
    assert "ไม่พบ bundle" in result.output


def test_fake_controller_is_really_executed(isolated_config, tmp_path, monkeypatch):
    """กันเทสข้างบนหลอกตัวเอง: ถ้า subprocess ไม่ได้รันจริง calls.log จะไม่มีวันเกิด"""
    directory = _fake_bundle(tmp_path, monkeypatch)
    cli_main._run_smoke("demo", keep=True)
    assert (directory / "calls.log").is_file()
    assert os.access(directory / "demo-single.sh", os.X_OK)
