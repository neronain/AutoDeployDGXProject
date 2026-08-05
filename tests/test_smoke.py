"""smoke test runner — พิสูจน์ว่า bundle รันได้จริง แล้วยกสถานะเป็น hardware-validated

เทสในไฟล์นี้ไม่รันโมเดลจริง (ต้องมี GPU) แต่ทดสอบสิ่งที่พังได้จริงในตัว runner:
ลำดับขั้น, การ stop เสมอเมื่อ start ไปแล้ว, และเงื่อนไขที่ทำให้ยกสถานะได้/ไม่ได้
"""

from __future__ import annotations

import pytest

from lmds.smoke import (
    STATUS_HARDWARE,
    STATUS_STATIC,
    available_tests,
    plan_steps,
    read_record,
    record_path,
    run_smoke,
    validation_status,
)

VLLM_DISPATCH = """#!/usr/bin/env bash
RUNTIME_MODE="${RUNTIME_MODE:-docker}"
case "${1:-help}" in
  download)     download ;;
  verify-files) verify_files ;;
  start)        start ;;
  stop)         stop ;;
  test-reasoning)  test_reasoning ;;
  test-tools)      test_tools "${2:-required}" ;;
  test-text)    test_text ;;
  *)            usage ;;
esac
"""

LLAMACPP_NATIVE_DISPATCH = """#!/usr/bin/env bash
RUNTIME_MODE="${RUNTIME_MODE:-native}"
case "${1:-help}" in
  download)     download ;;
  verify-files) verify_files ;;
  prepare-runtime) prepare_runtime ;;
  start)        start ;;
  stop)         stop ;;
  test-text)    test_text ;;
  *)            usage ;;
esac
"""


def _controller(tmp_path, text: str, name: str = "demo-single.sh"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class Recorder:
    """แทน run_step — จำลำดับคำสั่งที่ถูกเรียก และให้กำหนดได้ว่าคำสั่งไหน exit ไม่เป็นศูนย์"""

    def __init__(self, failures: dict[str, int] | None = None):
        self.calls: list[str] = []
        self.failures = failures or {}

    def __call__(self, command: str) -> int:
        self.calls.append(command)
        return self.failures.get(command, 0)


# ── การอ่านคำสั่ง test-* จากสคริปต์จริง ───────────────────────────────────────

def test_tests_are_read_from_the_script_not_a_hardcoded_list():
    """แต่ละ bundle มี test ไม่เท่ากัน — llama.cpp มีแค่ test-text, vLLM มีตาม plan ที่เปิดไว้

    ถ้า hardcode รายการไว้ smoke จะสั่ง test-tools กับ bundle ที่ไม่มีคำสั่งนั้น แล้ว
    controller ตอบ usage + exit 0 เงียบ ๆ = รายงานว่าผ่านทั้งที่ไม่เคยทดสอบอะไรเลย
    """
    assert available_tests(VLLM_DISPATCH) == ["test-text", "test-reasoning", "test-tools"]
    assert available_tests(LLAMACPP_NATIVE_DISPATCH) == ["test-text"]


def test_test_text_runs_before_the_other_tests():
    """test-text คือด่านที่บอกว่าโมเดลตอบได้จริงไหม — ตกตรงนี้แล้วตัวอื่นไม่มีความหมาย"""
    assert available_tests(VLLM_DISPATCH)[0] == "test-text"


def test_unknown_future_tests_are_included_automatically():
    """test-* ที่เพิ่มเข้า template ทีหลังต้องถูกรันเอง ไม่ต้องกลับมาแก้ smoke"""
    script = VLLM_DISPATCH.replace(
        "  test-text)    test_text ;;",
        "  test-text)    test_text ;;\n  test-anthropic) test_anthropic ;;",
    )
    assert "test-anthropic" in available_tests(script)


def test_prepare_runtime_only_for_native_builds():
    """dispatch case prepare-runtime มีอยู่ทุก mode — ต้องดูที่ RUNTIME_MODE ไม่ใช่ที่ case"""
    native = [c for c, _ in plan_steps(LLAMACPP_NATIVE_DISPATCH)]
    docker = [c for c, _ in plan_steps(VLLM_DISPATCH)]
    assert "prepare-runtime" in native
    assert native.index("prepare-runtime") < native.index("start")
    assert "prepare-runtime" not in docker


# ── ลำดับการรันจริง ───────────────────────────────────────────────────────────

def test_full_sequence_then_stop(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder()
    record = run_smoke(controller, "demo", run_step=runner)

    assert runner.calls == [
        "download", "verify-files", "start",
        "test-text", "test-reasoning", "test-tools",
        "stop",
    ]
    assert record.passed
    assert record.stopped


def test_stop_still_runs_when_a_test_fails(tmp_path):
    """เคสที่เจ็บที่สุด: smoke ตกหลัง start แล้วทิ้งเซิร์ฟเวอร์กินหน่วยความจำค้างไว้

    รอบถัดไปจะชน port ตัวเอง แล้วดูเหมือน "โมเดลนี้รันไม่ได้" ทั้งที่สาเหตุคือตัวเก่ายังอยู่
    """
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder(failures={"test-tools": 3})
    record = run_smoke(controller, "demo", run_step=runner)

    assert runner.calls[-1] == "stop"
    assert not record.passed
    assert record.failed_step.command == "test-tools"
    assert record.stopped


def test_no_stop_when_start_never_succeeded(tmp_path):
    """ยังไม่เคย start = ไม่มีอะไรให้หยุด · สั่ง stop เปล่า ๆ ทำให้ log อ่านสับสน"""
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder(failures={"download": 1})
    record = run_smoke(controller, "demo", run_step=runner)

    assert runner.calls == ["download"]
    assert "stop" not in runner.calls
    assert not record.stopped


def test_steps_after_a_failure_are_not_run(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder(failures={"start": 7})
    record = run_smoke(controller, "demo", run_step=runner)

    assert "test-text" not in runner.calls
    assert record.failed_step.code == 7


# ── สถานะ validation ─────────────────────────────────────────────────────────

def test_status_is_static_until_a_smoke_run_exists(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "ยังไม่เคยรัน" in reason


def test_passing_smoke_makes_it_hardware_validated(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    status, _ = validation_status("demo", controller)
    assert status == STATUS_HARDWARE


def test_failing_smoke_does_not_claim_hardware_validated(tmp_path):
    """กฎข้อ 3 — ห้ามอ้าง hardware-validated โดยไม่ได้รันจริง (และรันแล้วตกก็ไม่นับ)"""
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder(failures={"test-text": 1}))
    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "test-text" in reason


def test_editing_the_controller_drops_the_claim(tmp_path):
    """ผลผูกกับ sha256 ของสคริปต์ที่รันจริง

    ไม่งั้น: รัน smoke ผ่าน → แก้ context/flag ในสคริปต์ → ยังขึ้น hardware-validated
    ทั้งที่สคริปต์ตัวที่ผ่านไม่มีอยู่แล้ว
    """
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    assert validation_status("demo", controller)[0] == STATUS_HARDWARE

    controller.write_text(VLLM_DISPATCH + "\n# แก้อะไรสักอย่าง\n", encoding="utf-8")
    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "ถูกแก้" in reason


def test_missing_controller_drops_the_claim(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    controller.unlink()
    assert validation_status("demo", controller)[0] == STATUS_STATIC


def test_record_lives_outside_the_bundle(tmp_path):
    """เพิ่มไฟล์ในโฟลเดอร์ bundle = gate checksums ตกทันที · และสถานะเป็นของ (bundle × เครื่อง)
    ไม่ใช่ของ bundle เดี่ยว ๆ — ส่ง ZIP ไปเครื่องอื่นแล้วต้องไม่ติดสถานะไปด้วย
    """
    bundle_dir = tmp_path / "bundles" / "demo"
    bundle_dir.mkdir(parents=True)
    controller = _controller(bundle_dir, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    written = record_path("demo")
    assert written.is_file()
    assert bundle_dir not in written.parents
    assert not list(bundle_dir.glob("smoke*"))


def test_record_survives_a_round_trip(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    original = run_smoke(controller, "demo", run_step=Recorder(failures={"test-tools": 3}))
    loaded = read_record("demo")
    assert loaded is not None
    assert [s.command for s in loaded.steps] == [s.command for s in original.steps]
    assert loaded.failed_step.code == 3
    assert loaded.fingerprint == original.fingerprint


def test_corrupt_record_is_ignored_not_fatal(tmp_path):
    """ไฟล์ผลเสียแล้วต้องกลับไปเป็น static-validated ไม่ใช่ traceback ตอน lmds doctor"""
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    record_path("demo").write_text("{ไม่ใช่ json", encoding="utf-8")
    assert read_record("demo") is None
    assert validation_status("demo", controller)[0] == STATUS_STATIC


@pytest.mark.parametrize("payload", ['{"slug": ""}', "[]", "null"])
def test_record_without_a_slug_is_rejected(tmp_path, payload):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    record_path("demo").write_text(payload, encoding="utf-8")
    assert read_record("demo") is None
