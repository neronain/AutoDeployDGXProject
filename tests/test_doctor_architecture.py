"""รันไทม์รู้จักสถาปัตยกรรมของ checkpoint ไหม — คำถามที่ doctor ไม่เคยถาม

muse-glimmer-30b บน spark-head ผ่าน doctor ครบ 7 ข้อ แล้วตายใน 29 วินาที
ตอน start ด้วย "Transformers does not recognize this architecture" · ทุกข้อที่
ตรวจถูกหมด สิ่งที่ขาดคือข้อที่ไม่มีใครเขียนไว้
"""

from __future__ import annotations

import json

import pytest

from lmds.doctor import checks
from lmds.doctor.checks import Status


class _Server:
    mode = "docker"
    slug = "demo"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """ให้ _weight_paths ชี้มาที่ tmp แทน cache จริง"""
    directory = tmp_path / "snapshot"
    directory.mkdir()
    monkeypatch.setattr(checks, "_weight_paths", lambda profile, slug: (directory, []))
    monkeypatch.setattr(checks.shutil, "which", lambda name: "/usr/bin/docker")
    return directory


PROFILE = {"runtime": {"image": "vllm:test"}}


def _write_config(directory, model_type):
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))


def test_an_architecture_the_image_does_not_know_is_a_failure(model_dir, monkeypatch):
    _write_config(model_dir, "muse_glimmer")
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (0, "UNKNOWN 5.6.0"))

    finding = checks._check_architecture(PROFILE, _Server(), "demo")[0]
    assert finding.status is Status.FAIL
    assert "muse_glimmer" in finding.detail
    # ต้องบอกด้วยว่าจะเกิดอะไรขึ้นถ้าไม่แก้ ไม่ใช่แค่ว่าอะไรผิด
    assert "container จะตาย" in finding.detail
    assert "VLLM_IMAGE" in finding.fix


def test_a_known_architecture_passes(model_dir, monkeypatch):
    _write_config(model_dir, "qwen3")
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (0, "KNOWN 5.14.1"))

    finding = checks._check_architecture(PROFILE, _Server(), "demo")[0]
    assert finding.status is Status.OK
    assert "qwen3" in finding.detail


def test_an_image_that_is_not_pulled_yet_is_skipped(model_dir, monkeypatch):
    """ไม่ดึง image 20 GB มาเพื่อถามคำถามนี้ — _check_image บอกเรื่องนั้นไปแล้ว"""
    _write_config(model_dir, "muse_glimmer")
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (1, ""))

    assert checks._check_architecture(PROFILE, _Server(), "demo") == []


def test_a_checkpoint_with_no_config_is_skipped(model_dir, monkeypatch):
    """ยังไม่ได้โหลด weight — _check_weights บอกไปแล้ว อย่าเตือนซ้ำคนละเรื่อง"""
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (0, "UNKNOWN 5.6.0"))

    assert checks._check_architecture(PROFILE, _Server(), "demo") == []


def test_a_probe_that_cannot_run_says_nothing(model_dir, monkeypatch):
    """ถามไม่ได้ ไม่ได้แปลว่าใช้ไม่ได้ — เงียบดีกว่าเตือนผิด"""
    _write_config(model_dir, "muse_glimmer")
    calls = []

    def fake_run(args, timeout=10):
        calls.append(args)
        return (0, "") if len(calls) > 1 else (0, "ok")

    monkeypatch.setattr(checks, "_run", fake_run)
    assert checks._check_architecture(PROFILE, _Server(), "demo") == []


def test_a_native_deployment_has_no_image_to_ask(model_dir, monkeypatch):
    _write_config(model_dir, "muse_glimmer")
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (0, "UNKNOWN 5.6.0"))

    class _Native(_Server):
        mode = "native"

    assert checks._check_architecture(PROFILE, _Native(), "demo") == []


def test_unreadable_config_does_not_raise(model_dir, monkeypatch):
    (model_dir / "config.json").write_text("{ not json")
    monkeypatch.setattr(checks, "_run", lambda args, timeout=10: (0, "UNKNOWN 5.6.0"))

    assert checks._check_architecture(PROFILE, _Server(), "demo") == []
