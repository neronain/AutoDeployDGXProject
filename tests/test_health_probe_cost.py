"""เช็คสถานะต้องไม่ไปปลุก GPU

/health ของ SGLang รันโมเดลจริงหนึ่งรอบทุกครั้งที่ถูกเรียก (prefill 1 token) ไม่ใช่แค่
ตอบว่ายังไม่ตาย · LMDS เช็คสถานะเป็นระยะทั้งจาก CLI และหน้าเว็บ โมเดลจึงถูกสั่งคิด
ตลอดเวลาโดยไม่มีใครถามอะไร

เคสจริง 2026-09-01 บน spark-head: ผู้ใช้เห็น GPU 78% ทั้งที่ยังไม่เคยยิงคำสั่งสักครั้ง
log ฝั่งเซิร์ฟเวอร์มีแต่ `GET /health` 11 ครั้งใน 3 นาที กับ `Prefill batch` 11 ครั้ง
ตรงกันหนึ่งต่อหนึ่ง · วัดเทียบแล้ว /v1/models กับ /get_model_info เสีย prefill +0
ส่วน /health กับ /health_generate เสีย +1
"""

import httpx
import pytest

from lmds.fleet import manager


@pytest.fixture
def seen(monkeypatch):
    calls: list[str] = []

    class R:
        status_code = 200

    def fake_get(url, **_):
        calls.append(url)
        return R()

    monkeypatch.setattr(manager.httpx, "get", fake_get)
    return calls


def test_sglang_is_probed_without_running_the_model(seen):
    assert manager._health_ok(8000, "sglang") is True
    assert seen == ["http://127.0.0.1:8000/v1/models"], seen


@pytest.mark.parametrize("engine", ["vllm", "llamacpp", ""])
def test_other_engines_keep_using_health(seen, engine):
    """/health ของสองตัวนั้นถูกอยู่แล้ว และแยก 'กำลังโหลด' (503) ออกจาก 'พร้อม' ได้"""
    assert manager._health_ok(8000, engine) is True
    assert seen == ["http://127.0.0.1:8000/health"], seen


def test_no_port_means_no_request_at_all(seen):
    assert manager._health_ok(0, "sglang") is False
    assert seen == []


def test_a_server_that_refuses_the_connection_is_not_healthy(monkeypatch):
    def boom(url, **_):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(manager.httpx, "get", boom)
    assert manager._health_ok(8000, "sglang") is False


def test_sglang_images_are_recognised_so_orphans_get_the_cheap_probe():
    """container ที่ LMDS ไม่ได้สร้างเองก็ต้องได้ engine ที่ถูก ไม่งั้นโดน /health เหมือนเดิม"""
    from lmds.fleet.manager import _engine_from_image
    for image in ("lmsysorg/sglang:latest", "scitrera/dgx-spark-sglang-mm:v0",
                  "nvcr.io/nvidia/sglang:26.02-py3"):
        assert _engine_from_image(image) == "sglang", image
    assert _engine_from_image("nvcr.io/nvidia/vllm:26.08-py3") == "vllm"
    assert _engine_from_image("acme/mystery:1") == ""
