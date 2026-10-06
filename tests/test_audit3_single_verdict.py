"""Audit รอบ 3 (2026-10-06) ข้อ 4 — "รันอยู่ / healthy / ผ่าน" ต้องผูกกับเซิร์ฟเวอร์ของ bundle นี้ และกับคำตอบจริง

(a) info · status · wait-health เดิมเชื่อ 200 อะไรก็ได้บนพอร์ต — บนเครื่องจริงเครื่องหนึ่งพอร์ต 8000 คือ portainer
(b) test-text เดิมจบ 0 แม้คำตอบว่าง / ไม่ใช่ chat completion / ไม่มี python3 ให้อ่านคำตอบ — `lmds smoke` ดูแค่ exit code

รัน controller ที่ render แล้วทั้งไฟล์ กับ engine ปลอมที่เป็น HTTP server จริง (tests/single_controller_harness.py)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.single_controller_harness import KINDS, box, free_port  # noqa: F401 — fixture

DOCKER_KINDS = ["vllm", "sglang"]


def _state_line(stdout: str) -> str:
    return next(line for line in stdout.splitlines() if line.strip().startswith("State"))


# ═════════════════════ (a) พอร์ตตอบ ≠ เซิร์ฟเวอร์ของเรา ═════════════════════
@pytest.mark.parametrize("kind", KINDS)
def test_another_server_on_the_port_is_not_reported_as_this_bundle_running(box, kind):
    """ไม่มี container/PID ของเราเลย แต่มีโมเดลอื่นตอบอยู่บนพอร์ตเดียวกัน — เดิม info: RUNNING · status: API healthy ·
    wait-health rc=0 (llama.cpp พิมพ์ "process: stopped / API: healthy" คู่กัน)"""
    port = free_port()
    box.serve(port, model="some-other-model", ours=False)
    flags = ["--port", str(port)]

    info = box.run("info", *flags)
    assert "RUNNING" not in info.stdout, info.stdout
    assert f"port {port} is answered by something else" in _state_line(info.stdout), info.stdout

    status = box.run("status", *flags)
    assert status.returncode == 0
    assert "API: healthy" not in status.stdout, status.stdout
    assert f"port {port} is answered by something else" in status.stdout, status.stdout

    waited = box.run("wait-health", *flags, env={"HEALTH_TIMEOUT": "3"})
    assert waited.returncode != 0, waited.stdout + waited.stderr
    assert f"port {port} is answered by something else" in waited.stderr, waited.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_a_server_with_our_model_name_but_not_our_container_is_still_not_ours(box, kind):
    """ชื่อโมเดลตรงยังไม่พอ — ต้องเป็น container/process ที่ bundle นี้ start เอง (stop ของเราหยุดมันไม่ได้ จึงห้ามเรียกว่า "ของเรารันอยู่")"""
    port = free_port()
    box.serve(port, ours=False)
    status = box.run("status", "--port", str(port))
    assert "API: healthy" not in status.stdout, status.stdout
    assert "is answered by something else" in status.stdout
    assert "RUNNING" not in box.run("info", "--port", str(port)).stdout


@pytest.mark.parametrize("kind", KINDS)
def test_our_container_alive_but_the_port_serving_another_model_is_a_mismatch(box, kind):
    """container ของเรายังอยู่ (เช่นกำลังจะตายเพราะ bind ไม่ได้) แต่ที่ตอบบนพอร์ตคือโมเดลอื่น"""
    port = free_port()
    box.serve(port, ours=True, models="some-other-model")
    status = box.run("status", "--port", str(port))
    assert "API: healthy" not in status.stdout, status.stdout
    assert "some-other-model" in status.stdout and box.model in status.stdout, status.stdout
    assert "RUNNING" not in box.run("info", "--port", str(port)).stdout
    waited = box.run("wait-health", "--port", str(port), env={"HEALTH_TIMEOUT": "3"})
    assert waited.returncode != 0 and "some-other-model" in waited.stderr, waited.stdout + waited.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_our_own_healthy_server_is_running(box, kind):
    port = free_port()
    box.serve(port, ours=True)
    flags = ["--port", str(port)]
    assert "API: healthy" in box.run("status", *flags).stdout
    assert "RUNNING" in _state_line(box.run("info", *flags).stdout)
    assert box.run("wait-health", *flags).returncode == 0


@pytest.mark.parametrize("kind", DOCKER_KINDS)
def test_a_container_that_is_up_but_not_answering_yet_is_not_running(box, kind):
    """container รันอยู่ (ยังโหลดโมเดล) แต่ API ยังไม่ตอบ — ไม่ใช่ RUNNING และไม่ใช่ "มีคนอื่นตอบ" """
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        (box.state / f"{box.container}.pid").write_text(str(sleeper.pid), encoding="utf-8")
        port = free_port()
        status = box.run("status", "--port", str(port))
        assert "API: not responding" in status.stdout, status.stdout
        state = _state_line(box.run("info", "--port", str(port)).stdout)
        assert "RUNNING" not in state and "something else" not in state, state
        waited = box.run("wait-health", "--port", str(port), env={"HEALTH_TIMEOUT": "1"})
        assert waited.returncode == 1 and "ยังไม่ health" in waited.stderr, waited.stdout + waited.stderr
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)


@pytest.mark.parametrize("kind", KINDS)
def test_nothing_running_and_nothing_listening_is_plainly_down(box, kind):
    port = free_port()
    status = box.run("status", "--port", str(port))
    assert "API: not responding" in status.stdout and "something else" not in status.stdout, status.stdout
    assert "stopped" in _state_line(box.run("info", "--port", str(port)).stdout)
    waited = box.run("wait-health", "--port", str(port), env={"HEALTH_TIMEOUT": "3"})
    assert waited.returncode != 0 and "หยุดก่อน health ผ่าน" in waited.stderr, waited.stdout + waited.stderr


@pytest.mark.parametrize("kind", DOCKER_KINDS)
def test_status_without_the_servers_key_does_not_call_a_healthy_server_down(box, kind):
    """เซิร์ฟเวอร์ถูก start ด้วย API key แต่ status ถูกเรียกโดยไม่มี key (key มาจาก env ครั้งเดียว) → /v1/models ตอบ 401
    ยืนยันชื่อไม่ได้ · container ของเรารันอยู่และ /health ตอบ = ยัง healthy พร้อมบอกว่ายืนยันชื่อไม่ได้ —
    รายงานว่า down แล้วมีคนไป restart โมเดลที่ใช้งานอยู่ แย่กว่ามาก"""
    port = free_port()
    started = box.run("start", "--port", str(port), env={"API_KEY": "sekrit-key-1"})
    assert started.returncode == 0, started.stdout + started.stderr
    with_key = box.run("status", "--port", str(port), env={"API_KEY": "sekrit-key-1"})
    assert "API: healthy" in with_key.stdout and "ยืนยัน" not in with_key.stdout, with_key.stdout
    without = box.run("status", "--port", str(port))
    assert "API: healthy" in without.stdout and "ยืนยันชื่อโมเดลไม่ได้" in without.stdout, without.stdout
    assert "RUNNING" in box.run("info", "--port", str(port)).stdout


@pytest.mark.parametrize("kind", DOCKER_KINDS)
def test_a_server_started_under_a_one_off_name_is_still_recognised(box, kind):
    """`SERVED_MODEL_NAME=foo <ctl> start` ครั้งเดียว แล้ว status เปล่า ๆ — ชื่อที่ start จดไว้ใน server.meta นับเป็นชื่อของเรา"""
    port = free_port()
    started = box.run("start", "--port", str(port), env={"SERVED_MODEL_NAME": "one-off-name"})
    assert started.returncode == 0, started.stdout + started.stderr
    status = box.run("status", "--port", str(port))
    assert "API: healthy" in status.stdout, status.stdout


@pytest.mark.parametrize("kind", DOCKER_KINDS)
def test_when_docker_cannot_be_asked_the_model_name_decides(box, kind):
    """user ต่อ docker socket ไม่ได้ (ยังไม่อยู่ในกลุ่ม docker) — ถาม container ไม่ได้ ไม่ใช่ "container ไม่ได้รัน" ·
    ตัดสินจากชื่อโมเดลที่พอร์ตตอบแทน ไม่กล่าวหาว่ามีคนอื่นยึดพอร์ต"""
    port = free_port()
    box.serve(port, ours=True)
    status = box.run("status", "--port", str(port), env={"FAKE_DOCKER_DENIED": "1"})
    assert "API: healthy" in status.stdout and "something else" not in status.stdout, status.stdout + status.stderr


# ═════════════════════ (b) test-text ตัดสินจากคำตอบ ไม่ใช่จาก HTTP 200 ═════════════════════
def _test_text(box, **fake_env):
    port = free_port()
    box.serve(port, ours=True, **fake_env)
    return box.run("test-text", "--port", str(port))


@pytest.mark.parametrize("kind", KINDS)
def test_test_text_passes_on_a_real_answer(box, kind):
    done = _test_text(box, content="2+2 = 4")
    assert done.returncode == 0 and "test-text: OK — 2+2 = 4" in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_test_text_fails_on_an_empty_answer(box, kind):
    """HTTP 200 + message ว่าง: เดิมพิมพ์ "ไม่มีข้อความ" แล้วจบ 0 — `lmds smoke` / `deploy --smoke` ขึ้น "ผ่าน" """
    done = _test_text(box, content="")
    assert done.returncode != 0, done.stdout + done.stderr
    assert "ไม่มีข้อความ" in done.stdout + done.stderr


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("body", [
    '{"error": {"message": "model is still loading", "code": 503}}',     # error object มากับ HTTP 200
    '{"object": "list", "data": []}',                                     # JSON ที่ไม่ใช่ chat completion
    '{"choices": []}',
    "<html>portainer</html>",                                             # ไม่ใช่ JSON เลย
])
def test_test_text_fails_when_the_reply_is_not_a_chat_completion(box, kind, body):
    done = _test_text(box, chat_body=body)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "test-text: FAIL" in done.stdout + done.stderr, done.stdout + done.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_test_text_still_accepts_a_reasoning_model_that_ran_out_of_budget(box, kind):
    """content ว่างแต่มี reasoning = เซิร์ฟเวอร์ทำงานจริง แค่คิดไม่จบใน 512 tokens — ไม่ใช่ความล้มเหลวของ smoke"""
    done = _test_text(box, content="", reasoning="Let me think: 2+2 …")
    assert done.returncode == 0 and "ยังคิดไม่จบ" in done.stdout, done.stdout + done.stderr


def _path_without_python(box) -> str:
    """PATH ที่มีทุกอย่างของเครื่องยกเว้น python3 — /usr/bin มี python3 ทั้งบน macOS และ Ubuntu จึงต้อง symlink รายตัว"""
    real = box.tmp / "nopython"
    real.mkdir()
    for src_dir in (Path("/usr/bin"), Path("/bin")):
        for tool in src_dir.iterdir():
            if tool.name.startswith("python") or (real / tool.name).exists():
                continue
            try:
                (real / tool.name).symlink_to(tool)
            except OSError:
                pass
    return f"{box.bin}:{real}"


@pytest.mark.parametrize("kind", KINDS)
def test_test_text_fails_when_it_cannot_read_the_answer(box, kind):
    """ไม่มี python3 = ไม่มีใครอ่านคำตอบ — เดิมข้ามการตัดสินไปเงียบ ๆ แล้วจบ 0 (ยิงได้ HTTP 200 ≠ โมเดลตอบ)"""
    port = free_port()
    box.serve(port, ours=True, content="")
    done = box.run("test-text", "--port", str(port), path=_path_without_python(box))
    assert done.returncode != 0, done.stdout + done.stderr
    assert "python3" in done.stderr, done.stderr


# ═════════════════════ test-vision: คิดไม่จบ ≠ เห็นภาพ ═════════════════════
def _multimodal(plan):
    plan.multimodal.modalities = ["image"]


@pytest.mark.parametrize("content, reasoning, code", [("Red.", "", 0), ("", "The image seems …", 2), ("Blue", "", 1)])
def test_test_vision_only_passes_when_the_model_names_the_colour(tmp_path, content, reasoning, code):
    """เดิม content ว่าง + มี reasoning → "WARN คิดไม่จบ" แล้วจบ 0 = ปุ่ม test-vision ขึ้นผ่านทั้งที่ไม่ได้พิสูจน์ว่าเห็นภาพ"""
    from tests.single_controller_harness import Box, render

    made = Box(tmp_path, "vllm", render(tmp_path, "vllm", tweak=_multimodal))
    try:
        port = free_port()
        made.serve(port, ours=True, content=content, reasoning=reasoning)
        done = made.run("test-vision", "--port", str(port))
        assert done.returncode == code, done.stdout + done.stderr
    finally:
        made.close()
