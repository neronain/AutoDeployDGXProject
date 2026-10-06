"""Audit รอบ 3 (2026-10-06) ข้อ 3 — คำสั่งที่ไม่รู้จัก/ไม่รองรับของ single controller ต้องล้ม ไม่ใช่พิมพ์ usage แล้วคืน 0

รัน controller ที่ render แล้วทั้งไฟล์ใต้ bash (tests/single_controller_harness.py) · ฝั่งผู้ช่วย: รหัสออกเฉพาะของ
"bundle นี้ไม่มีคำสั่งนี้" ต้องถูกอ่านเป็น "ไม่รองรับ" ไม่ใช่ "เทสล้ม" และไม่ใช่ "ผ่าน"
"""

from __future__ import annotations

import pytest

from lmds.assistant import catalog, policy, runner
from lmds.inventory import controller_commands
from tests.single_controller_harness import KINDS, box  # noqa: F401 — fixture

UNSUPPORTED_EXIT = 64
ENGINE_LABEL = {"vllm": "vLLM", "sglang": "SGLang", "llamacpp": "llama.cpp"}


def _supported(stderr: str) -> list[str]:
    return stderr.split("รองรับ:", 1)[1].splitlines()[0].split()


@pytest.mark.parametrize("kind", KINDS)
def test_a_mistyped_verb_fails_and_names_what_this_bundle_can_do(box, kind):
    """`<ctl> stpo` เดิมพิมพ์ usage ยาว ๆ แล้วคืน 0 — hub/ผู้ช่วยอ่าน exit code แล้วรายงานว่า "สำเร็จ" ทั้งที่ไม่ได้ทำอะไร"""
    done = box.run("stpo")
    assert done.returncode == UNSUPPORTED_EXIT, done.stdout + done.stderr
    assert "'stpo'" in done.stderr and "ไม่รู้จัก" in done.stderr
    listed = _supported(done.stderr)
    assert {"start", "stop", "status", "logs", "download", "info"} <= set(listed), done.stderr
    assert "stpo" not in listed
    assert "COMMANDS" not in done.stdout, "ไม่พ่น usage ทั้งหน้า — hub โชว์แค่ท้าย ข้อความสาเหตุต้องเป็นสิ่งที่เห็น"


@pytest.mark.parametrize("kind, verb", [
    ("vllm", "test-vision"),        # template เดียวกันมี แต่ bundle นี้ไม่ใช่ multimodal จึงไม่มี
    ("vllm", "test-embed"), ("vllm", "serve-args"), ("vllm", "sync-worker"),
    ("sglang", "bench"), ("sglang", "stress"), ("sglang", "test-vision"), ("sglang", "test-rerank"),
    ("llamacpp", "test-reasoning"), ("llamacpp", "bench"), ("llamacpp", "parsers"), ("llamacpp", "test-embed"),
])
def test_a_verb_of_another_engine_says_not_supported_instead_of_succeeding(box, kind, verb):
    """ผู้ช่วย (run_test) ส่ง test-vision/test-reasoning/test-embed/test-rerank ให้ bundle ไหนก็ได้โดยไม่ดูว่า bundle นั้น
    มีคำสั่งไหม — เดิมได้ usage + exit 0 = "เทสผ่าน" ทั้งที่ไม่มีอะไรถูกทดสอบเลย"""
    done = box.run(verb)
    assert done.returncode == UNSUPPORTED_EXIT, done.stdout + done.stderr
    assert f"'{verb}'" in done.stderr and "ไม่รองรับ" in done.stderr and ENGINE_LABEL[kind] in done.stderr, done.stderr
    assert verb not in _supported(done.stderr)
    assert "COMMANDS" not in done.stdout


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("verb", ["repair", "remove", "doctor"])
def test_a_verb_that_belongs_to_the_lmds_cli_points_there(box, kind, verb):
    """`repair`/`remove`/`doctor` เป็นคำสั่งของ lmds ไม่ใช่ของ controller — คนที่พิมพ์ใส่ controller ต้องได้ทางไปต่อ"""
    done = box.run(verb)
    assert done.returncode == UNSUPPORTED_EXIT
    assert f"lmds {verb} {box.bundle.directory.name}" in done.stderr, done.stderr


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("args", [[], ["help"], ["--help"], ["-h"]])
def test_asking_for_help_still_exits_zero(box, kind, args):
    done = box.run(*args)
    assert done.returncode == 0 and "COMMANDS" in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_the_supported_list_matches_the_buttons_the_web_console_shows(box, kind):
    """ปุ่มบนหน้าเว็บอ่านจาก dispatch table ของสคริปต์ (inventory.controller_commands) — รายการที่บอกผู้ใช้ต้องครอบทุกปุ่ม
    และคำสั่งที่เพิ่งถูกปฏิเสธ (bench ของ SGLang ฯลฯ) ต้องไม่โผล่เป็นปุ่มเพราะมีบรรทัดปฏิเสธอยู่ในสคริปต์"""
    listed = set(_supported(box.run("no-such-verb").stderr))
    buttons = set(controller_commands(str(box.bundle.controller)))
    assert buttons and buttons <= listed, buttons - listed
    for verb in buttons:                 # ทุกปุ่มต้องไม่ใช่คำสั่งที่ controller ปฏิเสธเอง
        assert verb in listed
    if kind == "sglang":
        assert not {"bench", "stress", "test-vision"} & buttons


# ───────────────────────── ผู้ช่วย: run_test ─────────────────────────
def _fake_run(monkeypatch, code: int, out: str = "", err: str = ""):
    seen: list[str] = []

    def run_local(command: str, timeout: int):
        seen.append(command)
        return code, out, err

    monkeypatch.setattr(runner, "_run_local", run_local)
    return seen


def test_run_test_reports_not_supported_instead_of_a_failed_test(monkeypatch):
    """controller ตอบ 64 = bundle นี้ไม่มีคำสั่งนั้น — ไม่ใช่ "โมเดลสอบตก" และไม่ใช่ "ผ่าน" · ผู้ช่วยต้องเห็นคำว่าไม่รองรับ"""
    _fake_run(monkeypatch, UNSUPPORTED_EXIT, err="ERROR: bundle นี้ (SGLang) ไม่รองรับคำสั่ง 'test-vision'")
    outcome = runner.run_action("run_test", "this", {"slug": "nemo", "test": "test-vision"})
    assert outcome.unsupported is True and outcome.ok is False
    payload = outcome.payload()
    assert payload["unsupported"] is True and payload["ok"] is False
    assert payload["output"].startswith("ไม่รองรับ"), payload["output"]
    assert "ไม่ใช่ผลทดสอบ" in payload["output"] and "test-vision" in payload["output"]


def test_a_real_test_failure_is_still_a_failure(monkeypatch):
    for code in (1, 2):
        _fake_run(monkeypatch, code, out="FAIL: ไม่มี tool_calls")
        outcome = runner.run_action("run_test", "this", {"slug": "nemo", "test": "test-tools"})
        assert outcome.unsupported is False and outcome.ok is False
        assert not outcome.output.startswith("ไม่รองรับ")
    _fake_run(monkeypatch, 0, out="PASS")
    passed = runner.run_action("run_test", "this", {"slug": "nemo", "test": "test-tools"})
    assert passed.ok and not passed.unsupported


def test_exit_64_from_any_other_action_is_just_a_failure(monkeypatch):
    """เฉพาะ action ที่ประกาศว่ารู้จักรหัสนี้ — `lmds node install` ที่จบ 64 ด้วยเหตุอื่นต้องไม่ถูกเรียกว่า "ไม่รองรับ" """
    _fake_run(monkeypatch, UNSUPPORTED_EXIT, err="boom")
    outcome = runner.run_action("model_restart", "this", {"slug": "nemo"})
    assert outcome.unsupported is False and outcome.ok is False


def test_the_exit_code_is_the_one_the_controllers_use():
    assert catalog.CONTROLLER_UNSUPPORTED_EXIT == UNSUPPORTED_EXIT
    assert catalog.ACTIONS["run_test"].unsupported_exit == UNSUPPORTED_EXIT


def test_an_unsupported_test_does_not_stop_the_rest_of_the_ticket(monkeypatch):
    """ตั๋ว deploy จบด้วย test-text → test-tools: เทสที่ bundle ไม่มีไม่ใช่เหตุให้ทิ้งขั้นที่เหลือ (ขั้นที่ล้มจริงยังหยุดเหมือนเดิม)"""
    policy.reset()
    codes = iter([UNSUPPORTED_EXIT, 0])
    monkeypatch.setattr(runner, "_run_local", lambda command, timeout: (next(codes), "ok", ""))
    ticket = policy.propose([
        {"action": "run_test", "target": "this", "params": {"slug": "nemo", "test": "test-vision"}},
        {"action": "run_test", "target": "this", "params": {"slug": "nemo", "test": "test-text"}},
    ])
    policy.choose(ticket.id, policy.APPLY)
    _, outcomes = policy.advance(ticket.id)
    assert [o.unsupported for o in outcomes] == [True, False], "ขั้นที่สองต้องได้รัน"
    assert outcomes[1].ok

    policy.reset()
    codes = iter([1, 0])
    ticket = policy.propose([
        {"action": "run_test", "target": "this", "params": {"slug": "nemo", "test": "test-text"}},
        {"action": "run_test", "target": "this", "params": {"slug": "nemo", "test": "test-tools"}},
    ])
    policy.choose(ticket.id, policy.APPLY)
    _, outcomes = policy.advance(ticket.id)
    assert len(outcomes) == 1, "เทสที่ล้มจริงยังต้องหยุดตั๋ว"
