"""stacked + engine ที่ไม่มี template ต้องบอกตรง ๆ ไม่ใช่เงียบแล้วส่ง vLLM ให้

เคสจริง 2026-09-01: sparkarena/Minimax-M3-v0-NVFP4-REAP50 ขนาด 129 GB ใหญ่เกิน
เครื่องเดียว (GB10 มี 128 GB) จึงต้อง stacked · สถาปัตยกรรม
MiniMaxM3SparseForConditionalGeneration รันได้เฉพาะบน SGLang — vLLM 26.08 มีชื่อ
สถาปัตยกรรมในทะเบียนก็จริง แต่ตัวโหลดน้ำหนักพังที่ `assert shard_id in qkv_idxs`

สั่ง `lmds deploy --target dgx-spark-stacked --engine sglang` แล้วได้ controller ของ
vLLM กลับมาเงียบ ๆ เพราะ renderer เลือก template จาก is_stacked อย่างเดียว ไม่ดู engine
เลย — เสียเวลาไป sync น้ำหนัก 129 GB ข้ามเครื่องก่อนจะรู้ว่าเลือก engine ผิดตั้งแต่แรก
"""

import pathlib
import tempfile

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _bundle(engine: Engine | None):
    report = ModelReport(
        repo_id="org/big-moe", revision_sha="sha",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(160 * GIB),
        context_length=131072, kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    if engine is not None:
        plan.runtime.engine = engine
    return render_bundle(plan, report, fit, pathlib.Path(tempfile.mkdtemp()))


def test_stacked_with_sglang_says_so_instead_of_handing_back_a_vllm_controller():
    with pytest.raises(ValueError) as e:
        _bundle(Engine.SGLANG)
    msg = str(e.value)
    assert "sglang" in msg.lower(), msg
    assert "stacked" in msg.lower(), msg


def test_stacked_with_vllm_still_works():
    bundle = _bundle(Engine.VLLM)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")
    assert "stacked" in pathlib.Path(bundle.controller).name
    assert "vllm serve" in text or "vllm" in text
