"""ที่นั่งของ KV cache — ตอบคำถามที่ fit analyzer ตอบไม่ได้

analyzer เสนอ context สูงสุดที่ "ใส่ได้" ซึ่งตามนิยามคือค่าที่คนเดียวกิน pool หมดพอดี
ตัวเลขนั้นถูก แต่ถูกอ่านผิดตลอด — ตั้งตามแล้วคนที่สองต่อคิวโดยไม่มีอะไรบอก

ตัวเลขในไฟล์นี้อ้างอิงเคสจริง: MiniMax-M3 REAP25 บน 2x DGX Spark
(60 layers, GQA 4 หัว, head_dim 128, weight 174.2 GB, budget 227 GB)
"""

from __future__ import annotations

import pytest

from lmds.fit import memory as mem
from lmds.fit.analyzer import FitReport
from lmds.hardware import MemoryModel
from lmds.inspector.report import KvDims

M3 = KvDims(layers=60, kv_heads=4, head_dim=128)


def stacked(weights=174.2, budget=227.0) -> FitReport:
    return FitReport(target_name="dgx-spark-stacked", memory_model=MemoryModel.UNIFIED,
                     engine_assumed="vllm", weights_gb=weights, budget_gb=budget)


def kinds(advice) -> set[str]:
    return {a.kind for a in advice}


# ── ตัวเลขพื้นฐาน ────────────────────────────────────────────────────────────

def test_bytes_per_token_counts_both_k_and_v_every_layer():
    # 60 layers x (K+V) x 4 heads x 128 dim x 2 ไบต์
    assert mem.bytes_per_token(M3) == 122_880


def test_fp8_is_exactly_half():
    assert mem.bytes_per_token(M3, "fp8") == mem.bytes_per_token(M3, "bf16") // 2


def test_an_unknown_dtype_is_refused_not_guessed():
    with pytest.raises(ValueError):
        mem.bytes_per_token(M3, "int3")


# ── คำถามที่กลับด้านจาก analyzer ─────────────────────────────────────────────

def test_it_says_how_many_people_fit_at_a_given_context():
    """analyzer ตอบ context จาก concurrency · ตัวนี้ตอบ concurrency จาก context"""
    assert mem.plan(stacked(), M3, 32768).concurrency > 10
    assert mem.plan(stacked(), M3, 262144).concurrency < 2


def test_halving_the_context_doubles_the_people():
    wide = mem.plan(stacked(), M3, 262144).concurrency
    half = mem.plan(stacked(), M3, 131072).concurrency
    assert half == pytest.approx(wide * 2, rel=0.02)


def test_fp8_buys_the_same_thing_as_halving_the_context():
    """ลด context ครึ่งหนึ่ง กับ ลด KV ครึ่งหนึ่ง ให้ผลเท่ากัน — เลือกได้ว่าจะเสียอะไร"""
    shorter = mem.plan(stacked(), M3, 131072, "bf16").concurrency
    cheaper = mem.plan(stacked(), M3, 262144, "fp8").concurrency
    assert shorter == pytest.approx(cheaper, rel=0.02)


def test_a_context_that_does_not_fit_says_so():
    assert mem.plan(stacked(), M3, 524288).fits is False


def test_the_ladder_stops_at_the_models_own_limit():
    steps = [p.context for p in mem.ladder(stacked(), M3, native_context=65536)]
    assert max(steps) == 65536


def test_max_context_falls_as_concurrency_rises():
    alone = mem.max_context(stacked(), M3, concurrency=1)
    shared = mem.max_context(stacked(), M3, concurrency=4)
    assert shared == pytest.approx(alone / 4, rel=0.01)


# ── คำแนะนำ ──────────────────────────────────────────────────────────────────

def test_a_context_only_one_person_can_use_is_flagged():
    """นี่คือข้อที่ทำให้ทั้งไฟล์นี้มีอยู่ — ค่าที่ analyzer เสนอเองก็ติดข้อนี้"""
    assert "single-user" in kinds(mem.advise(stacked(), M3, 262144))


def test_a_comfortable_context_is_not_nagged():
    assert "single-user" not in kinds(mem.advise(stacked(), M3, 32768))


def test_fp8_is_offered_only_when_it_changes_the_answer():
    assert "fp8-would-help" in kinds(mem.advise(stacked(), M3, 262144))
    assert "fp8-would-help" not in kinds(mem.advise(stacked(), M3, 32768))


def test_asking_for_more_than_the_model_supports_is_a_model_problem_not_a_memory_one():
    said = kinds(mem.advise(stacked(), M3, 262144, native_context=131072))
    assert "over-native" in said


def test_running_out_of_memory_is_reported_as_such():
    assert "over-memory" in kinds(mem.advise(stacked(), M3, 524288))


def test_it_offers_a_bigger_context_when_there_is_room():
    advice = mem.advise(stacked(), M3, 8192)
    grow = next(a for a in advice if a.kind == "room-to-grow")
    assert grow.facts["suggest"] > 8192
    assert grow.facts["concurrency"] >= mem.MIN_USEFUL_CONCURRENCY


def test_a_thin_margin_is_called_out_even_when_it_fits():
    """พอดีเป๊ะไม่ใช่พอ — CUDA graph, activation และ NCCL ไม่ได้อยู่ในงบนี้"""
    tight = stacked(weights=210.0)          # เหลือ 17 GB
    assert "thin-margin" in kinds(mem.advise(tight, M3, 131072))


def test_multi_node_always_mentions_the_uncounted_comms_buffer():
    assert "stacked-comms-unbudgeted" in kinds(
        mem.advise(stacked(), M3, 32768, gpu_count=2))
    assert "stacked-comms-unbudgeted" not in kinds(
        mem.advise(stacked(), M3, 32768, gpu_count=1))


def test_a_context_off_the_power_of_two_ladder_is_noted():
    assert "odd-step" in kinds(mem.advise(stacked(), M3, 200000))


def test_it_says_it_cannot_answer_rather_than_guessing():
    assert kinds(mem.advise(stacked(), None, 32768)) == {"kv-dims-unknown"}
    unknown = FitReport(target_name="t", memory_model=MemoryModel.UNIFIED,
                        engine_assumed="vllm", budget_gb=100.0)
    assert kinds(mem.advise(unknown, M3, 32768)) == {"weights-unknown"}


def test_every_code_the_advisor_emits_has_an_explanation():
    """ผู้ช่วย LLM ได้รหัสไป ไม่ได้ประโยค — รหัสที่ไม่มีคำอธิบายคือรหัสที่มันต้องเดาเอง"""
    emitted = set()
    for weights in (174.2, 210.0, 226.0):
        for ctx in (4096, 8192, 32768, 200000, 262144, 524288):
            for nodes in (1, 2):
                emitted |= kinds(mem.advise(stacked(weights=weights), M3, ctx,
                                            native_context=524288, gpu_count=nodes))
    emitted |= kinds(mem.advise(stacked(), None, 4096))
    missing = emitted - set(mem.ADVICE_LEGEND)
    assert not missing, f"รหัสที่ยังไม่มีคำอธิบาย: {missing}"


# ── MLA: K กับ V ถูกบีบเป็น latent ก้อนเดียว ────────────────────────────────
# DeepSeek-V2/V3 และ Kimi K2/K3 ใช้ท่านี้ · สูตร GQA ปกติจะนับ 2 x heads x head_dim
# ซึ่งเกินจริงหลายสิบเท่า แล้วไปตัด context ทิ้งโดยไม่มีใครรู้ว่าเสียอะไรไป

def test_mla_stores_one_latent_not_a_key_and_a_value():
    """เคสจริง Kimi-K3: 93 layers, kv_lora_rank 512, qk_rope_head_dim 64"""
    mla = KvDims(layers=93, kv_heads=1, head_dim=576, latent_dim=576)
    assert mem.bytes_per_token(mla) == 93 * 576 * 2       # ไม่มี x2 ของ K/V
    assert mem.bytes_per_token(mla) / 1024 < 110          # ~105 KiB/token


def test_the_gqa_formula_would_have_been_twentyfold_wrong():
    """ตัวเลขที่ทำให้เจอบั๊ก — กันไม่ให้ใครเผลอเอาสูตรเดิมกลับมา"""
    mla = KvDims(layers=93, kv_heads=1, head_dim=576, latent_dim=576)
    as_if_gqa = KvDims(layers=93, kv_heads=96, head_dim=74)
    assert mem.bytes_per_token(as_if_gqa) > mem.bytes_per_token(mla) * 20


def test_a_plain_gqa_model_is_untouched_by_the_latent_path():
    assert M3.latent_dim is None
    assert mem.bytes_per_token(M3) == 2 * 60 * 4 * 128 * 2


def test_mla_context_beats_what_the_old_formula_allowed():
    """93 layers ที่คิดแบบ GQA เหลือ context หลักหมื่น · คิดแบบ MLA ได้หลักแสน"""
    fit = stacked(weights=153.6, budget=227.0)
    mla = KvDims(layers=93, kv_heads=1, head_dim=576, latent_dim=576)
    as_if_gqa = KvDims(layers=93, kv_heads=96, head_dim=74)
    assert mem.plan(fit, mla, 262144).fits
    assert not mem.plan(fit, as_if_gqa, 262144).fits


# ── หน้า wizard: ตอบจาก session ที่วิเคราะห์ไว้แล้ว ─────────────────────────
# ที่ต้องมีข้อนี้: คำแนะนำต้องขึ้นกับ **โมเดลที่กำลังจะ deploy** ได้ทันที ไม่ใช่รอให้
# bundle ถูกสร้างแล้วไปบันทึกมิติ KV ไว้ก่อน — ไม่งั้นโมเดลที่เพิ่งเลือกจะไม่มีคำแนะนำ
# ซึ่งเป็นจังหวะเดียวที่คนต้องการมันจริง ๆ

def _wizard_session(kv_dims=M3, weights=174.2):
    from lmds.inspector.report import ArtifactType, ModelReport
    from lmds.web import deploy

    report = ModelReport(repo_id="acme/m", revision_sha="deadbeef",
                         artifact_type=ArtifactType.SAFETENSORS,
                         kv_dims=kv_dims, context_length=524288)
    session = deploy.Session(source=None, report=report,
                             fit=stacked(weights=weights), plan=None)
    deploy._SESSIONS["sid"] = session
    return deploy


def test_the_wizard_answers_for_the_model_being_deployed():
    deploy = _wizard_session()
    out = deploy.context_advice("sid", 262144)
    assert out["available"] is True
    assert out["kv_bytes_per_token"] == 122_880
    assert {a["kind"] for a in out["advice"]} >= {"single-user", "fp8-would-help"}


def test_the_wizard_ladder_stops_at_the_models_own_limit():
    deploy = _wizard_session()
    steps = [row["context"] for row in deploy.context_advice("sid", 32768)["ladder"]]
    assert max(steps) <= 524288


def test_asking_in_fp8_changes_the_whole_answer():
    deploy = _wizard_session()
    out = deploy.context_advice("sid", 262144, "fp8")
    assert out["kv_bytes_per_token"] == 61_440
    assert "single-user" not in {a["kind"] for a in out["advice"]}


def test_a_model_whose_kv_cannot_be_read_says_so_instead_of_going_quiet():
    deploy = _wizard_session(kv_dims=None)
    out = deploy.context_advice("sid", 32768)
    assert out["available"] is False
    assert out["reason"] in mem.ADVICE_LEGEND


def test_an_expired_session_is_an_error_not_an_empty_answer():
    from lmds.web import deploy

    deploy._SESSIONS.pop("gone", None)
    with pytest.raises(deploy.DeployError):
        deploy.context_advice("gone", 32768)
