"""Audit รอบ 3 (2026-10-06) ข้อ 6 — llama.cpp โหมด docker ทิ้ง ENGINE_ENV

`lmds set <slug> --engine-env "KEY=VALUE …"` ถูก export เข้าเชลล์ของ controller เท่านั้น — โหมด native ได้ไปเพราะ
llama-server เป็นลูกของเชลล์นั้น แต่ `docker run` ของโหมด docker (ที่เครื่อง RTX/x86 ได้) ไม่มี `-e` สักตัว ค่าจึงไม่ถึง
คอนเทนเนอร์โดยไม่มีอะไรบอก · docker ปลอมของ harness ส่ง env ให้ engine เฉพาะที่มากับ `-e` เหมือน docker จริง
"""

from __future__ import annotations

import pytest

from tests.single_controller_harness import box, free_port  # noqa: F401 — fixture

ENGINE_ENV = "LLAMA_ARG_NO_KV_OFFLOAD=1 LLAMA_LOG_PREFIX=value-that-must-not-be-on-argv"


def _engine_env_line(calls: str) -> str:
    return next(line for line in calls.splitlines() if line.startswith("engine[llama-server] env:"))


@pytest.mark.parametrize("kind", ["llamacpp"])
@pytest.mark.parametrize("mode", ["docker", "native"])
def test_engine_env_reaches_llama_server(box, kind, mode):
    port = free_port()
    started = box.run("start", "--port", str(port),
                      env={"RUNTIME_MODE": mode, "ENGINE_ENV": ENGINE_ENV, "LMDS_SKIP_ARCH_CHECK": "1"})
    assert started.returncode == 0, started.stdout + started.stderr
    seen = _engine_env_line(box.calls())
    assert "LLAMA_ARG_NO_KV_OFFLOAD=1" in seen and "LLAMA_LOG_PREFIX=value-that-must-not-be-on-argv" in seen, seen


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_docker_mode_passes_names_only_so_values_stay_off_argv(box, kind):
    port = free_port()
    started = box.run("start", "--port", str(port),
                      env={"RUNTIME_MODE": "docker", "ENGINE_ENV": ENGINE_ENV, "LMDS_SKIP_ARCH_CHECK": "1"})
    assert started.returncode == 0, started.stdout + started.stderr
    run_line = next(line for line in box.calls().splitlines() if line.startswith("docker run -d"))
    assert " -e LLAMA_ARG_NO_KV_OFFLOAD " in run_line and " -e LLAMA_LOG_PREFIX " in run_line, run_line
    assert "value-that-must-not-be-on-argv" not in run_line and "LLAMA_ARG_NO_KV_OFFLOAD=1" not in run_line, run_line


@pytest.mark.parametrize("kind", ["llamacpp"])
def test_docker_mode_without_engine_env_adds_nothing(box, kind):
    port = free_port()
    started = box.run("start", "--port", str(port), env={"RUNTIME_MODE": "docker", "LMDS_SKIP_ARCH_CHECK": "1"})
    assert started.returncode == 0, started.stdout + started.stderr
    run_line = next(line for line in box.calls().splitlines() if line.startswith("docker run -d"))
    assert " -e " not in run_line, run_line
