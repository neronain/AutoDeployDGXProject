"""llama.cpp: context ที่แผนสัญญา ต้องเป็นค่าที่แต่ละ request ได้จริง

เคสจริง 2026-08-13 — Muse-Glimmer แผนบอก context 131,072 แต่ /props รายงาน 32,768
เพราะ max_num_seqs default เป็น 4 แล้ว llama.cpp หาร --ctx-size ให้ทุก slot เท่า ๆ กัน
ส่วน fit คำนวณที่ concurrency=1 ค่าที่ได้จึงเป็นของ slot เดียวอยู่แล้ว — ถูกหารซ้ำ
"""

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _gguf_plan():
    report = ModelReport(
        repo_id="unsloth/Muse-Glimmer-30B-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=int(30.1 * GIB),
        selected_gguf="Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        context_length=131072,
        kv_dims=KvDims(layers=52, kv_heads=2, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    return report, fit, build_plan(report, fit, provider=None)


def test_llamacpp_serves_one_slot_so_the_context_is_not_divided():
    _, _, plan = _gguf_plan()
    assert plan.serving.max_num_seqs == 1
    assert plan.serving.context == 131072


def test_the_controller_gets_that_same_pair(tmp_path):
    report, fit, plan = _gguf_plan()
    bundle = render_bundle(plan, report, fit, tmp_path)
    script = next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")
    assert 'CTX_SIZE="${CTX_SIZE:-131072}"' in script
    assert 'PARALLEL_SEQS="${PARALLEL_SEQS:-1}"' in script


def test_asking_for_more_slots_is_allowed_but_the_cost_is_stated():
    """ไม่แก้ค่าที่คนตั้งเอง แต่ต้องบอกว่าแต่ละ request จะเหลือเท่าไร"""
    from lmds.brain.orchestrator import harden_plan

    report, fit, plan = _gguf_plan()
    plan.serving.max_num_seqs = 4
    hardened = harden_plan(plan, report, fit)
    assert hardened.serving.max_num_seqs == 4, "ห้ามแอบเปลี่ยนค่าที่ผู้ใช้เลือก"
    assert any("32,768" in w for w in hardened.warnings), hardened.warnings


def test_vllm_is_untouched():
    """vLLM แชร์ KV แบบ dynamic — กติกานี้ใช้กับมันไม่ได้"""
    report = ModelReport(
        repo_id="Qwen/Qwen3-8B",
        revision_sha="sha",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=int(16 * GIB),
        context_length=131072,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    assert plan.serving.max_num_seqs == 4
