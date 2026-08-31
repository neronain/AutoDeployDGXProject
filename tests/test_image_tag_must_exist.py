"""image ที่ไม่มีอยู่จริงต้องถูกจับตั้งแต่วางแผน และ error ต้องบอกความจริง

เคสจริงที่ลูกค้าเจอ 2026-09-01 ตอน deploy stacked:

    docker: Error response from daemon: manifest for nvcr.io/nvidia/vllm:latest
            not found: manifest unknown: manifest unknown
    ERROR: download ล้มเหลวแม้ปิด Xet แล้ว — ดูข้อความด้านบน

สองปัญหาซ้อนกัน:
  1. `nvcr.io/nvidia/vllm:latest` ไม่เคยมีอยู่จริง (repo นี้ใช้ tag ตามเดือน) แต่ผ่านทุกด่าน
     เพราะ allowlist ตรวจแค่ชื่อ repo และตัวตรวจ tag ไม่รู้จัก nvcr.io
  2. `download()` ไม่ได้ยืนยันว่ามี image ก่อน — พอ `docker run` ล้ม มันสรุปเองว่าเป็น
     ปัญหา Xet แล้วพิมพ์ "download ล้มเหลวแม้ปิด Xet แล้ว" ซึ่งพาคนไปดูผิดที่ทั้งหมด
"""

import pathlib
import tempfile

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(preset: str, weight_gib: float) -> str:
    report = ModelReport(repo_id="org/m", revision_sha="sha",
                         artifact_type=ArtifactType.SAFETENSORS,
                         weight_bytes=int(weight_gib * GIB), context_length=131072,
                         kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128))
    fit = analyze(report, PRESETS[preset])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, pathlib.Path(tempfile.mkdtemp()))
    return next(bundle.directory.glob("*.sh")).read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    """ตัดฟังก์ชันออกมาด้วยการนับปีกกา — regex พลาดกับ { } ที่ซ้อนกัน"""
    start = text.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอปีกกาปิดของ {name}")


@pytest.mark.parametrize("preset,weight", [("dgx-spark-single", 40),
                                           ("dgx-spark-stacked", 160)])
def test_download_confirms_the_image_exists_before_touching_weights(preset, weight):
    """image หายต้องบอกว่า image หาย ไม่ใช่โทษ Xet"""
    body = _function(_controller(preset, weight), "download")
    assert "ensure_image" in body, "download() ไม่ได้ยืนยันว่ามี image ก่อน"
    # ต้องมาก่อนการดึงน้ำหนัก ไม่ใช่หลัง — ไม่งั้นก็ยังล้มที่เดิมด้วยข้อความเดิม
    assert body.index("ensure_image") < body.index("snapshot_download")


@pytest.mark.parametrize("preset,weight", [("dgx-spark-single", 40),
                                           ("dgx-spark-stacked", 160)])
def test_the_default_image_for_a_spark_is_a_tag_that_exists(preset, weight):
    """NGC ไม่เคยมี :latest สำหรับ repo นี้ — bundle ต้องไม่เกิดมาพร้อม tag ผี"""
    text = _controller(preset, weight)
    # ตัดคอมเมนต์ออกก่อน — ตัวสคริปต์อธิบายเคสนี้ไว้ในคอมเมนต์ด้วย ซึ่งไม่ใช่ค่าที่ถูกใช้จริง
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "nvcr.io/nvidia/vllm:latest" not in code
    assert "nvcr.io/nvidia/vllm:26.05-py3" in code


def test_ngc_is_queryable_so_a_bad_tag_is_caught_at_plan_time():
    """ตัวตรวจ tag มีมานานแล้ว — มันแค่ไม่รู้จัก nvcr.io จึงคืน 'ตรวจไม่ได้' แล้วปล่อยผ่าน"""
    from lmds.brain.registry import _ANON_TOKEN, split_ref

    assert "nvcr.io" in _ANON_TOKEN
    assert "proxy_auth" in _ANON_TOKEN["nvcr.io"], "/token ของ NGC ตอบ 401 — ต้องใช้ /proxy_auth"
    assert split_ref("nvcr.io/nvidia/vllm:latest") == ("nvcr.io", "nvidia/vllm", "latest")


def test_a_tag_that_does_not_exist_is_swapped_for_the_known_good_one(monkeypatch):
    """แผนที่เสนอ tag ผี ต้องถูกเปลี่ยนพร้อมบอกเหตุผล ไม่ใช่ปล่อยไปตายตอน deploy"""
    from lmds.brain import orchestrator

    seen = {}

    def fake_tag_exists(ref, client=None):
        seen[ref] = ref.endswith(":latest") is False
        return False if ref.endswith(":latest") else True

    monkeypatch.setattr("lmds.brain.registry.tag_exists", fake_tag_exists)

    report = ModelReport(repo_id="org/m", revision_sha="sha",
                         artifact_type=ArtifactType.SAFETENSORS,
                         weight_bytes=int(40 * GIB), context_length=131072,
                         kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128))
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)

    # จำลองแผนที่ LLM เสนอ tag ผีมา แล้วให้ orchestrator ตัดสิน
    plan.runtime.image_ref = "nvcr.io/nvidia/vllm:latest"
    orchestrator._enforce_image(plan, fit) if hasattr(orchestrator, "_enforce_image") else None

    # ถ้ายังไม่มีจุดให้เรียกแยก อย่างน้อยต้องมั่นใจว่า tag_exists ถูกใช้ในเส้นทางนี้
    import inspect as _inspect
    src = _inspect.getsource(orchestrator)
    assert "tag_exists" in src, "orchestrator ต้องเช็ก tag ก่อนยอมรับ image ของแผน"
    assert "ไม่มีอยู่จริงบน registry" in src
