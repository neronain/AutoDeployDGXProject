"""เครื่องที่รันโมเดลไม่ได้ ต้องไม่ยอมดูด weight ลงมาเปล่า ๆ

เคสจริง 2026-08-19: `lmds repair` บน hub VM (ไม่มี GPU/docker/llama.cpp, RAM 12 GB)
เริ่มโหลด weight 15.6 GB อย่างว่าง่าย — ไฟล์ที่ต่อให้โหลดจบก็ไม่มีอะไรรันมันได้
"""

import pytest

from lmds.hardware import serving


@pytest.fixture(autouse=True)
def _fresh_cache():
    serving.reset_cache()
    yield
    serving.reset_cache()


def _no_runtime(monkeypatch):
    monkeypatch.setattr(serving.shutil, "which", lambda name: None)
    monkeypatch.setattr(serving, "llamacpp_server", lambda pinned="": None)


def _with_llamacpp(monkeypatch, tmp_path):
    monkeypatch.setattr(serving.shutil, "which", lambda name: None)
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(serving, "llamacpp_server", lambda pinned="": binary)


def test_no_engine_means_control_plane(monkeypatch):
    _no_runtime(monkeypatch)
    monkeypatch.delenv(serving.ROLE_ENV, raising=False)
    capability = serving.detect()
    assert capability.is_control_plane
    assert capability.engines == []


def test_llamacpp_binary_means_serving(monkeypatch, tmp_path):
    _with_llamacpp(monkeypatch, tmp_path)
    monkeypatch.delenv(serving.ROLE_ENV, raising=False)
    capability = serving.detect()
    assert capability.can_serve
    assert "llamacpp" in capability.engines


def test_guard_blocks_weight_hungry_commands(monkeypatch):
    _no_runtime(monkeypatch)
    monkeypatch.delenv(serving.ROLE_ENV, raising=False)
    for action in ("download", "repair", "start", "restart", "prepare-runtime"):
        assert serving.guard("some-model", action), f"{action} ควรถูกปฏิเสธ"


def test_guard_lets_read_only_commands_through(monkeypatch):
    """doctor/status/logs ไม่ได้ดูดอะไรเพิ่ม — คนบน hub ต้องใช้ได้ตามปกติ"""
    _no_runtime(monkeypatch)
    monkeypatch.delenv(serving.ROLE_ENV, raising=False)
    for action in ("verify-files", "status", "doctor", "logs", "stop"):
        assert serving.guard("some-model", action) == ""


def test_force_and_env_override(monkeypatch, tmp_path):
    _no_runtime(monkeypatch)
    monkeypatch.delenv(serving.ROLE_ENV, raising=False)
    assert serving.guard("some-model", "repair", force=True) == ""

    serving.reset_cache()
    monkeypatch.setenv(serving.ROLE_ENV, "serving")
    assert serving.guard("some-model", "repair") == ""


def test_hub_override_blocks_even_with_gpu(monkeypatch, tmp_path):
    """เครื่องที่มี GPU แต่ตั้งใจให้เป็น hub ต้องถูกกันเหมือนกัน"""
    _with_llamacpp(monkeypatch, tmp_path)
    monkeypatch.setenv(serving.ROLE_ENV, "hub")
    assert serving.detect().is_control_plane
    assert serving.guard("some-model", "repair")


def test_refusal_names_the_way_out():
    capability = serving.ServingCapability(gpus=0, docker=False, llamacpp=None, engines=[])
    message = capability.refusal("my-model", "repair", "spark-head")
    assert "lmds node push spark-head my-model --download" in message
    assert "control plane" in message
