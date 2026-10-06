"""Audit รอบ 3 (2026-10-06) ข้อ 5 — API key ต้องถึง engine โดยไม่อยู่บน argv ที่เครื่องมองเห็น

SGLang เดิมได้ `--api-key "$API_KEY"` บน argv ของ `docker run` (เห็นใน `ps` · ถูกพิมพ์ตอน DRY_RUN) · vLLM ใช้ env ·
llama.cpp ใช้ไฟล์ · เทสนี้ไม่เชื่อว่า "ส่ง key ไปแล้ว": engine ปลอมรับ key **เฉพาะช่องที่ engine จริงรับ** (ดู
tests/single_controller_harness.py) แล้วเทสยิง HTTP จริงโดยไม่ใส่ key — ต้องได้ 401 · key ที่ไปผิดช่อง = 200 = เทสแดง
(บทเรียน LLAMA_ARG_API_KEY: เทสที่เชื่อ env ผ่านทั้งที่เซิร์ฟเวอร์รันแบบไม่มี auth)
"""

from __future__ import annotations

import json
import stat
import urllib.error
import urllib.request

import pytest

from tests.single_controller_harness import box, free_port  # noqa: F401 — fixture

KEY = "SECRET-KEY-123"
MODES = [("vllm", "docker"), ("sglang", "docker"), ("llamacpp", "native"), ("llamacpp", "docker")]


def _models_status(port: int, key: str = "") -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}",
                                     headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.mark.parametrize("kind, mode", MODES)
def test_the_key_reaches_the_engine_but_never_an_argv(box, kind, mode):
    port = free_port()
    env = {"API_KEY": KEY, "RUNTIME_MODE": mode, "LMDS_SKIP_ARCH_CHECK": "1"}
    started = box.run("start", "--port", str(port), env=env)
    assert started.returncode == 0, started.stdout + started.stderr
    assert KEY not in started.stdout + started.stderr

    calls = box.calls()
    argv_lines = [line for line in calls.splitlines() if line.startswith("docker ") or " argv: " in line]
    assert argv_lines and not [line for line in argv_lines if KEY in line], \
        "key อยู่บน argv:\n" + "\n".join(line for line in argv_lines if KEY in line)
    # engine ได้ key จริง (ผ่านช่องของมันเอง) และบังคับใช้: ไม่ใส่ key = 401 · ใส่ถูก = 200
    assert f"auth: key={KEY}" in calls, calls
    assert _models_status(port) == 401, "เซิร์ฟเวอร์ตอบคำขอที่ไม่มี key — key ไม่ถึง engine"
    assert _models_status(port, KEY) == 200

    # controller ของตัวเองยังคุยกับเซิร์ฟเวอร์ได้ และ stop เก็บไฟล์ key
    assert "API: healthy" in box.run("status", "--port", str(port), env=env).stdout
    assert box.run("stop", env=env).returncode == 0
    leftovers = [p for p in box.run_dir.glob("api-key*")] if box.run_dir.exists() else []
    assert not leftovers, f"ไฟล์ key ค้างหลัง stop: {leftovers}"


@pytest.mark.parametrize("kind", ["sglang"])
def test_sglang_key_file_is_private_and_holds_exactly_the_key(box, kind):
    """key ที่มีอักขระที่ YAML ตีความเองได้ (# : " \\ และตัวเลขล้วน) ต้องไปถึง engine ตรงตัว"""
    for key in ('we"ird: #key\\x', "1234567890"):
        port = free_port()
        started = box.run("start", "--port", str(port), env={"API_KEY": key})
        assert started.returncode == 0, started.stdout + started.stderr
        key_file = box.run_dir / "api-key.yaml"
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
        assert json.loads(key_file.read_text(encoding="utf-8").split(":", 1)[1]) == key
        assert _models_status(port) == 401 and _models_status(port, key) == 200
        assert box.run("stop").returncode == 0


@pytest.mark.parametrize("kind", ["sglang", "vllm"])
def test_dry_run_never_prints_the_key(box, kind):
    """DRY_RUN พิมพ์ argv ที่จะรัน — ผลนี้ถูกแปะลง ticket/แชต · SGLang เดิมพิมพ์ key ออกมาทั้งตัว"""
    done = box.run("start", env={"DRY_RUN": "1", "API_KEY": KEY})
    assert done.returncode == 0, done.stderr
    assert KEY not in done.stdout + done.stderr
    assert "--served-model-name" in done.stdout
    if kind == "sglang":
        forced = box.run("start", env={"DRY_RUN": "1", "API_KEY": KEY, "SGLANG_KEY_ON_ARGV": "1"})
        assert KEY not in forced.stdout + forced.stderr and "********" in forced.stdout


@pytest.mark.parametrize("kind", ["sglang"])
def test_sglang_that_does_not_take_the_key_from_the_file_is_stopped_not_left_open(box, kind):
    """ด่านหลัง start: ตั้ง key ไว้แต่เซิร์ฟเวอร์ตอบโดยไม่ต้องใช้ key = key ไม่ถึง engine — ต้องไม่รายงานว่า start สำเร็จ
    และต้องไม่ทิ้งโมเดลเปิดโล่งไว้ (จำลอง SGLang ที่เมินคีย์ในไฟล์ config)"""
    port = free_port()
    started = box.run("start", "--port", str(port), env={"API_KEY": KEY, "FAKE_IGNORE_CONFIG_KEY": "1"})
    assert started.returncode != 0, started.stdout + started.stderr
    assert "ไม่บังคับ API key" in started.stderr and "SGLANG_KEY_ON_ARGV=1" in started.stderr, started.stderr
    assert "started:" not in started.stdout
    assert "auth: OPEN" in box.calls(), "เทสนี้ต้องเจอเซิร์ฟเวอร์ที่เปิดโล่งจริง"
    assert not (box.state / f"{box.container}.pid").exists(), "container ที่เปิดโล่งต้องถูกหยุด"
    assert not (box.run_dir / "api-key.yaml").exists()


@pytest.mark.parametrize("kind", ["sglang"])
def test_an_sglang_too_old_for_config_files_fails_loudly_and_offers_the_explicit_fallback(box, kind):
    port = free_port()
    env = {"API_KEY": KEY, "FAKE_NO_CONFIG": "1"}
    started = box.run("start", "--port", str(port), env=env)
    assert started.returncode != 0
    assert "--config" in started.stderr and "SGLANG_KEY_ON_ARGV=1" in started.stderr, started.stderr

    forced = box.run("start", "--port", str(port), env={**env, "SGLANG_KEY_ON_ARGV": "1"})
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "อยู่บน argv" in forced.stderr, "ทางถอยต้องบอกราคาที่จ่าย"
    assert _models_status(port) == 401 and _models_status(port, KEY) == 200


@pytest.mark.parametrize("kind", ["sglang"])
def test_sglang_without_a_key_starts_without_a_key_file(box, kind):
    port = free_port()
    started = box.run("start", "--port", str(port))
    assert started.returncode == 0, started.stdout + started.stderr
    assert "--config" not in box.calls() and not (box.run_dir / "api-key.yaml").exists()
    assert "ไม่มี API key" in started.stderr                 # คำเตือน endpoint เปิดโล่งยังอยู่
