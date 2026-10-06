"""Audit รอบ 3 (2026-10-06) ข้อ 7 — ค่า serving ที่บันทึกไว้ผิดช่วง ต้องไม่ทำให้ stop/status/logs ใช้ไม่ได้

validate_numbers เดิมรันจาก parse_options ก่อน dispatch ทุกคำสั่ง: `GPU_MEMORY_UTILIZATION=0.99` ใน bundle.env (`lmds set`
รับค่านี้) ทำให้ stop · status · logs จบ exit 1 — โมเดลที่รันอยู่หยุดผ่าน controller ของตัวเองไม่ได้ (repro_gpu_util.sh)
"""

from __future__ import annotations

import pytest

from tests.single_controller_harness import KINDS, box, free_port, pid_alive  # noqa: F401 — fixture

# ค่าที่ validate_numbers ของแต่ละ engine ปฏิเสธ
BAD_KNOB = {"vllm": "GPU_MEMORY_UTILIZATION=0.99", "sglang": "GPU_MEMORY_UTILIZATION=0.99", "llamacpp": "CTX_SIZE=0"}
FIX_FLAG = {"vllm": "--gpu-util", "sglang": "--gpu-util", "llamacpp": "--context"}


def _save_bad_knob(box, port: int) -> None:
    (box.bundle.directory / "bundle.env").write_text(f"{BAD_KNOB[box.kind]}\nAPI_PORT={port}\n", encoding="utf-8")


@pytest.mark.parametrize("kind", KINDS)
def test_a_running_model_can_still_be_inspected_and_stopped_with_a_bad_saved_knob(box, kind):
    port = free_port()
    server = box.serve(port, ours=True)
    (box.run_dir / "server.log").parent.mkdir(parents=True, exist_ok=True)
    (box.run_dir / "server.log").write_text("llama-server: listening\n", encoding="utf-8")
    _save_bad_knob(box, port)

    verbs = ["status", "info", "logs", "network-info", "verify-files", "help"]
    if kind != "llamacpp":            # llama.cpp: context 0 ทำให้คำนวณงบ token ไม่ได้จริง ๆ — client-config ล้มได้โดยชอบ
        verbs.append("client-config")
    for verb in verbs:
        done = box.run(verb)
        assert done.returncode == 0, f"{verb}: {done.stdout}{done.stderr}"
        assert "invalid" not in done.stderr, f"{verb}: {done.stderr}"
    assert "API: healthy" in box.run("status").stdout

    stopped = box.run("stop")
    assert stopped.returncode == 0 and "stopped" in stopped.stdout, stopped.stdout + stopped.stderr
    assert server.wait(timeout=10) is not None, "เซิร์ฟเวอร์ยังรันอยู่หลัง stop"


@pytest.mark.parametrize("kind", KINDS)
def test_start_still_refuses_the_bad_knob_and_says_how_to_fix_the_saved_value(box, kind):
    port = free_port()
    _save_bad_knob(box, port)
    started = box.run("start")
    assert started.returncode != 0, started.stdout + started.stderr
    assert "invalid" in started.stderr, started.stderr
    assert f"lmds set {box.bundle.directory.name} {FIX_FLAG[kind]}" in started.stderr, started.stderr
    calls = box.calls()
    assert "run -d" not in calls and "engine[" not in calls, "ต้องไม่สั่ง engine รันด้วยค่าที่ผิด"


@pytest.mark.parametrize("kind", KINDS)
def test_restart_validates_before_it_stops_the_running_model(box, kind):
    """restart = stop แล้ว start — ถ้าตรวจค่าหลัง stop โมเดลที่ใช้งานอยู่จะถูกหยุดแล้ว start ไม่ขึ้น"""
    port = free_port()
    server = box.serve(port, ours=True)
    _save_bad_knob(box, port)
    restarted = box.run("restart")
    assert restarted.returncode != 0 and "invalid" in restarted.stderr, restarted.stdout + restarted.stderr
    assert pid_alive(server.pid) and server.poll() is None, "restart หยุดโมเดลไปก่อนจะรู้ว่าค่าใช้ไม่ได้"


@pytest.mark.parametrize("kind", KINDS)
def test_a_one_off_bad_flag_is_still_rejected_on_start(box, kind):
    done = box.run("start", "--port", "99999")
    assert done.returncode != 0 and "invalid API_PORT" in done.stderr, done.stdout + done.stderr


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_client_config_explains_a_context_it_cannot_do_arithmetic_on(box, kind):
    """ไม่ผ่าน validate_numbers แล้ว client-config ต้องยังไม่พังด้วย error ของ bash arithmetic"""
    done = box.run("client-config", env={"MAX_MODEL_LEN": "abc"})
    assert done.returncode != 0 and "client-config" in done.stderr and "abc" in done.stderr, done.stdout + done.stderr
    assert "syntax error" not in done.stderr and "unbound variable" not in done.stderr
