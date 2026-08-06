"""smoke test runner — พิสูจน์ว่า bundle รันได้จริง แล้วยกสถานะเป็น hardware-validated

เทสในไฟล์นี้ไม่รันโมเดลจริง (ต้องมี GPU) แต่ทดสอบสิ่งที่พังได้จริงในตัว runner:
ลำดับขั้น, การ stop เสมอเมื่อ start ไปแล้ว, และเงื่อนไขที่ทำให้ยกสถานะได้/ไม่ได้
"""

from __future__ import annotations

import json

import pytest

from lmds.smoke import (
    STATUS_HARDWARE,
    STATUS_STATIC,
    SmokeBusy,
    SmokeError,
    available_tests,
    plan_steps,
    read_record,
    record_path,
    run_smoke,
    smoke_lock,
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

VLLM_ASSET_DISPATCH = VLLM_DISPATCH.replace(
    "  verify-files) verify_files ;;",
    "  prepare-runtime) prepare_runtime ;;\n  verify-files) verify_files ;;",
).replace('RUNTIME_MODE="${RUNTIME_MODE:-docker}"\n', "")

STACKED_DISPATCH = """#!/usr/bin/env bash
case "${1:-help}" in
  prepare-runtime) prepare_runtime ;;
  download) download ;;
  verify-files) verify_files ;;
  sync-worker) sync_worker ;;
  verify-worker) verify_worker ;;
  start) start ;;
  stop) stop ;;
  test-text) test_text ;;
  *) usage ;;
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


def test_runtime_assets_are_prepared_before_verify_files():
    commands = [command for command, _ in plan_steps(VLLM_ASSET_DISPATCH)]
    assert commands[:4] == ["download", "prepare-runtime", "verify-files", "start"]


def test_stacked_smoke_includes_cross_node_prerequisites():
    commands = [command for command, _ in plan_steps(STACKED_DISPATCH)]
    assert commands == [
        "prepare-runtime", "download", "verify-files", "sync-worker",
        "verify-worker", "start", "test-text",
    ]


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


def test_optional_capability_exit_two_is_skipped_and_later_tests_run(tmp_path):
    script = VLLM_DISPATCH.replace(
        "  test-text)    test_text ;;",
        "  test-text)    test_text ;;\n  test-anthropic) test_anthropic ;;",
    )
    controller = _controller(tmp_path, script)
    runner = Recorder(failures={"test-anthropic": 2})
    record = run_smoke(controller, "demo", run_step=runner)

    assert runner.calls == [
        "download", "verify-files", "start", "test-text", "test-anthropic",
        "test-reasoning", "test-tools", "stop",
    ]
    assert next(step for step in record.steps if step.command == "test-anthropic").skipped
    assert record.passed
    assert record.failure_code == 0


def test_required_text_exit_two_is_a_failure_not_a_skip(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder(failures={"test-text": 2})
    record = run_smoke(controller, "demo", run_step=runner)

    assert not record.passed
    assert record.failed_step.command == "test-text"
    assert record.failure_code == 2
    assert "test-reasoning" not in runner.calls


def test_start_failure_still_attempts_compensating_stop(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder(failures={"start": 7})
    record = run_smoke(controller, "demo", run_step=runner)

    assert runner.calls == ["download", "verify-files", "start", "stop"]
    assert record.failure_code == 7
    assert record.stopped


def test_stop_failure_fails_the_smoke_and_the_validation_claim(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    record = run_smoke(controller, "demo", run_step=Recorder(failures={"stop": 9}))

    assert not record.passed
    assert record.failure_code == 9
    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "หยุดเซิร์ฟเวอร์ไม่สำเร็จ" in reason


def test_interrupted_rerun_invalidates_old_success_and_stops(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    calls: list[str] = []

    def interrupt_on_text(command: str) -> int:
        calls.append(command)
        if command == "test-text":
            raise KeyboardInterrupt
        return 0

    with pytest.raises(KeyboardInterrupt):
        run_smoke(controller, "demo", run_step=interrupt_on_text)

    assert calls[-1] == "stop"
    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "ไม่ครบทุกขั้น" in reason or "ไม่สมบูรณ์" in reason


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


def test_record_has_only_source_evidence_not_stale_derived_status(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    payload = json.loads(record_path("demo").read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert "status" not in payload
    assert "passed" not in payload
    assert payload["planned_steps"] == [
        "download", "verify-files", "start", "test-text", "test-reasoning", "test-tools",
    ]


def test_truncated_successful_steps_do_not_validate(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    path = record_path("demo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"] = payload["steps"][:4]
    path.write_text(json.dumps(payload), encoding="utf-8")

    status, reason = validation_status("demo", controller)
    assert status == STATUS_STATIC
    assert "ไม่ครบทุกขั้น" in reason


def test_corrupt_record_is_ignored_not_fatal(tmp_path):
    """ไฟล์ผลเสียแล้วต้องกลับไปเป็น static-validated ไม่ใช่ traceback ตอน lmds doctor"""
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    record_path("demo").write_text("{ไม่ใช่ json", encoding="utf-8")
    assert read_record("demo") is None
    assert validation_status("demo", controller)[0] == STATUS_STATIC


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["steps"][0].update(code="oops"),
        lambda payload: payload["steps"][0].update(seconds=float("nan")),
        lambda payload: payload.update(stop_code=True),
        lambda payload: payload.update(planned_steps="download"),
    ],
)
def test_semantically_corrupt_record_is_ignored_not_fatal(tmp_path, mutate):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    path = record_path("demo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_record("demo") is None
    assert validation_status("demo", controller)[0] == STATUS_STATIC


def test_record_write_failure_aborts_before_controller_actions(tmp_path, monkeypatch):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    runner = Recorder()
    monkeypatch.setattr("lmds.smoke.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(SmokeError, match="บันทึกผล"):
        run_smoke(controller, "demo", run_step=runner)
    assert runner.calls == []


def test_same_bundle_smoke_runs_are_serialized(tmp_path):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    with smoke_lock("demo"):
        with pytest.raises(SmokeBusy, match="รันอยู่แล้ว"):
            run_smoke(controller, "demo", run_step=Recorder())


@pytest.mark.parametrize("payload", ['{"slug": ""}', "[]", "null"])
def test_record_without_a_slug_is_rejected(tmp_path, payload):
    controller = _controller(tmp_path, VLLM_DISPATCH)
    run_smoke(controller, "demo", run_step=Recorder())
    record_path("demo").write_text(payload, encoding="utf-8")
    assert read_record("demo") is None
