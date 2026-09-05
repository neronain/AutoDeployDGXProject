"""Audit stacked รอบ 2 (ชุดวางแผน: fit / brain / recipes / inspector / renderer)

โจทย์จากเจ้าของ 2026-09-05: "สร้างทีมตรวจสอบและวิเคราะห์การทำงานของ stack พยายามเก็บบั๊กให้หมด" —
มองจากลูกค้าที่เอาโมเดลเครื่องเดียวไป deploy แบบ stacked เพื่อให้ได้ KV/context/concurrency มากขึ้น
ทุกเทสในนี้ล้มก่อนแก้ · แต่ละเทสคือข้อบกพร่องหนึ่งข้อพร้อมเหตุผลว่าทำไมลูกค้าถึงเจอ
"""

from __future__ import annotations

import json

import pytest
import yaml

from lmds.brain import build_plan
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import DeploymentPlan, Engine, PlanError, Topology
from lmds.brain.rulebased import (
    SPARK_NVFP4_ENV,
    SPARK_NVFP4_VLLM_IMAGE,
    SPARK_VLLM_IMAGE,
    SPARK_VLLM_NIGHTLY_IMAGE,
    apply_recipe,
    is_nvfp4,
    rule_based_plan,
)
from lmds.fit import PRESETS, Verdict, analyze
from lmds.fit.analyzer import CONTEXT_STEPS, GIB, VLLM_MIN_KV_GB
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from lmds.recipes import find_recipe, load_catalog, synced_path

DEEPSEEK = "nvidia/DeepSeek-V4-Flash-NVFP4"
QWEN_CODER = "ucbye/Qwen3-Coder-Next-NVFP4-GB10"
STACKED = PRESETS["dgx-spark-stacked"]


def deepseek_report(**overrides) -> ModelReport:
    """nvidia/DeepSeek-V4-Flash-NVFP4: MLA (kv_heads=1) · 43 layers · 156.7 GiB · ctx 1M"""
    base = dict(
        repo_id=DEEPSEEK, revision_sha="e3cd60e7de98e9867116860d522499a728de1cf9",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=168_281_985_176, shard_count=46,
        context_length=1_048_576, quantization="fp8", architecture="DeepseekV4ForCausalLM",
        model_type="deepseek_v4", kv_dims=KvDims(layers=43, kv_heads=1, head_dim=512),
        moe_experts=256, moe_experts_active=6, has_chat_template=False,
    )
    base.update(overrides)
    return ModelReport(**base)


def qwen_coder_report(**overrides) -> ModelReport:
    base = dict(
        repo_id=QWEN_CODER, revision_sha="sha-coder", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=int(42.7 * GIB), shard_count=10, context_length=262_144, quantization="nvfp4",
        architecture="Qwen3NextForCausalLM", kv_dims=KvDims(layers=48, kv_heads=2, head_dim=256),
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def llm_plan_dict(report: ModelReport, **overrides) -> dict:
    """แผนที่ LLM ตอบกลับมา (ผ่าน schema) — ค่าตั้งต้นถูกทั้งหมด เทสแต่ละข้อบิดทีละจุด"""
    base = {
        "plan_version": 1,
        "model_id": report.repo_id,
        "revision": report.revision_sha,
        "served_model_name": report.repo_id.split("/")[-1].lower(),
        "artifact_type": report.artifact_type.value,
        "runtime": {"engine": "vllm", "image_ref": SPARK_NVFP4_VLLM_IMAGE, "rationale": "x"},
        "topology": "stacked",
        "serving": {"context": 32768, "max_output_tokens": 8192},
    }
    base.update(overrides)
    return base


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete_json(self, system, user):
        return self.responses.pop(0)


def _write_synced(entries: list[dict]) -> None:
    synced_path().parent.mkdir(parents=True, exist_ok=True)
    synced_path().write_text(yaml.safe_dump({"version": 1, "recipes": entries}, allow_unicode=True),
                             encoding="utf-8")
    load_catalog.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_catalog():
    load_catalog.cache_clear()
    yield
    load_catalog.cache_clear()


# ── 1. fit: KV ของโมเดล MLA/GQA เล็กไม่ได้ถูกแบ่งด้วย TP — ทุก rank ถือสำเนาเต็ม ──────────────
def test_kv_cache_of_an_mla_model_is_replicated_on_every_tp_rank_not_split():
    """ลูกค้า deploy DeepSeek-V4 (MLA · kv_heads=1) แบบ stacked เพื่อได้ context ยาวขึ้น — analyzer คิดว่า
    KV ทั้งคลัสเตอร์ (64 GB) หารสองเครื่องแล้วเสนอ 524,288 · แต่ vLLM แบ่ง KV ตาม kv_heads: head เดียว
    แบ่ง 2 rank ไม่ได้ จึง **ทำสำเนา** ไว้ทุกเครื่อง = แต่ละเครื่องต้องมี KV เต็ม context ในงบต่อเครื่อง
    (32 GB) · ที่ 524,288 ต้องใช้ 44 GB/เครื่อง → vLLM ตาย "No available memory for the cache blocks"
    ตอน profiling · Qwen3-Coder-Next (kv_heads=2) บน 4 เครื่องก็โดนแบบเดียวกัน (สำเนา 2 ชุด)"""
    report = deepseek_report()
    fit = analyze(report, STACKED)
    per_token = report.kv_dims.bytes_per_token_fp16
    per_node_kv = fit.kv_budget_gb / 2
    expected = max(s for s in CONTEXT_STEPS if s <= per_node_kv * GIB / per_token)
    assert fit.recommended_context == expected, (fit.recommended_context, expected, fit.notes)
    # ตัวเลขต่อ token ที่รายงาน (หน้าเว็บ/profile ใช้คำนวณแรม) ต้องเป็นค่าของทั้งคลัสเตอร์ = สำเนา × 2
    assert fit.kv_bytes_per_token == per_token * 2
    assert any("สำเนา" in n or "replicat" in n.lower() for n in fit.notes), fit.notes

    # GQA ที่ kv_heads หาร TP ลงตัว (4 หัว / 2 เครื่อง) — แบ่งได้จริง ไม่มีสำเนา ค่าเดิมต้องไม่เปลี่ยน
    gqa = deepseek_report(kv_dims=KvDims(layers=43, kv_heads=4, head_dim=128))
    gqa_fit = analyze(gqa, STACKED)
    assert gqa_fit.kv_bytes_per_token == gqa.kv_dims.bytes_per_token_fp16
    assert gqa_fit.recommended_context == max(
        s for s in CONTEXT_STEPS if s <= gqa_fit.kv_budget_gb * GIB / gqa.kv_dims.bytes_per_token_fp16)

    # kv_heads=2 บน 4 เครื่อง = แต่ละหัวถูกทำสำเนา 2 ชุด
    four = analyze(qwen_coder_report(), PRESETS["dgx-spark-stacked-4"])
    assert four.kv_bytes_per_token == qwen_coder_report().kv_dims.bytes_per_token_fp16 * 2

    # บันได/คำแนะนำของหน้า settings ต้องคิดสำเนาด้วย — ไม่งั้นบอกว่า "รับได้ 2 คน" ทั้งที่ได้คนเดียว
    from lmds.fit import plan as context_plan

    single_ctx = context_plan(analyze(report, PRESETS["dgx-spark-single"]), report.kv_dims, 131_072)
    stacked_ctx = context_plan(fit, report.kv_dims, 131_072)
    assert stacked_ctx.kv_gb == pytest.approx(single_ctx.kv_gb * 2, abs=0.2)


def test_stacked_min_kv_headroom_for_vllm_is_checked_per_node():
    """งบ KV รวม 3 GB บน 2 เครื่อง = เครื่องละ 1.5 GB ซึ่งต่ำกว่าที่ vLLM ต้องใช้ทำ profiling run (2 GB)
    → start ไม่ขึ้นแน่นอน · เดิมเทียบยอดรวม (3 ≥ 2) แล้วตอบ fits-reduced-context"""
    report = deepseek_report(kv_dims=KvDims(layers=43, kv_heads=4, head_dim=128))
    fit = analyze(report, STACKED)
    cluster_kv = VLLM_MIN_KV_GB * 1.5                      # 3 GB รวม → 1.5 GB ต่อเครื่อง
    tight = deepseek_report(kv_dims=report.kv_dims, weight_bytes=int((fit.budget_gb - cluster_kv) * GIB))
    verdict = analyze(tight, STACKED)
    assert verdict.verdict is Verdict.NEEDS_SMALLER_QUANT, (verdict.verdict, verdict.notes)
    assert any("ต่อเครื่อง" in n for n in verdict.notes), verdict.notes


def test_a_model_whose_native_context_is_below_the_ladder_still_fits():
    """โมเดล embedding เล็ก (MiniLM native 512) — ขั้นต่ำสุดของบันไดคือ 4,096 จึงหา step ≤ 512 ไม่เจอ
    แล้วตอบ needs-smaller-quant ให้โมเดล 90 MB บนเครื่อง 128 GB · ต้อง fits ที่ native ของมันเอง"""
    tiny = ModelReport(repo_id="sentence-transformers/all-MiniLM-L6-v2", revision_sha="sha",
                       artifact_type=ArtifactType.SAFETENSORS, weight_bytes=90_000_000, task="embed",
                       context_length=512, kv_dims=KvDims(layers=6, kv_heads=12, head_dim=32))
    fit = analyze(tiny, PRESETS["dgx-spark-single"])
    assert fit.verdict is Verdict.FITS, (fit.verdict, fit.notes)
    assert fit.recommended_context == 512
    assert fit.max_safe_context == 512


# ── 2. brain: env ของ NVFP4 หายเมื่อสูตรมี image แต่ image นั้นไม่ได้ถูกใช้ ────────────────────
def test_nvfp4_env_is_applied_when_the_recipe_image_is_not_the_one_in_the_plan():
    """ลูกค้า publish controller ที่ `lmds set --image registry.local:5000/vllm:custom` ไว้ แล้ว sync มาใช้กับ
    เครื่องอื่น → image นั้นไม่อยู่ใน registry ที่ยอมรับ harden จึงถอยไป image NVFP4 ตัวที่พิสูจน์แล้ว…
    แต่ apply_nvfp4_defaults เห็นว่า "สูตรมี image" ก็ข้าม env marlin ไป → image ตัวที่ถอยมาต้องมี env นี้
    ไม่งั้น JIT cutlass FP4 ตายตอน start (msi-6 2026-08-20) · เช่นเดียวกับสูตรที่ image ผูกกับ RTX"""
    report = qwen_coder_report(repo_id="someone/Qwen3-Coder-Next-NVFP4-Custom")
    fit = analyze(report, STACKED)

    _write_synced([{"match": report.repo_id, "engine": "vllm", "image": "registry.local:5000/vllm:custom",
                    "source": "customer", "validated_on": "their spark", "topology": "single"}])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    for key, value in SPARK_NVFP4_ENV.items():
        assert plan.serving.extra_env.get(key) == value, (key, plan.serving.extra_env)

    _write_synced([{"match": report.repo_id, "engine": "vllm", "image": "vllm/vllm-openai:v0.9.2",
                    "image_for": ["discrete"], "source": "customer", "validated_on": "rtx-5090"}])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    for key, value in SPARK_NVFP4_ENV.items():
        assert plan.serving.extra_env.get(key) == value, (key, plan.serving.extra_env)

    # สูตรที่ image ถูกใช้จริง (มี kernel ของตัวเอง) — ยังไม่ยุ่งเหมือนเดิม
    _write_synced([{"match": report.repo_id, "engine": "vllm", "image": "avarok/dgx-vllm-nvfp4-kernel:latest",
                    "image_for": ["unified"], "source": "team", "validated_on": "spark-head"}])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == "avarok/dgx-vllm-nvfp4-kernel:latest"
    assert "VLLM_NVFP4_GEMM_BACKEND" not in plan.serving.extra_env


# ── 3. brain: image ที่ "อยู่ใน allowlist" แต่รู้อยู่แล้วว่าไม่มี kernel ให้โมเดลนี้ ────────────
def test_harden_replaces_a_known_but_kernel_less_image_for_nvfp4_on_a_spark():
    """LLM เสนอ nvcr 26.05 (repo ถูก tag มีจริง) ให้โมเดล NVFP4 บน Spark → ผ่านด่านทุกด่าน แล้วตายตอน start
    `cvt .e2m1x2 not supported on sm_121` (env marlin อย่างเดียวไม่พอกับ image นั้น) · และ glm5_next
    ที่ LLM เสนอ image 61fc… (vLLM 0.28.0 ไม่รู้จัก) → check_architecture หยุดก่อน start · ทั้งคู่
    เรารู้คำตอบอยู่แล้ว ต้องเปลี่ยนให้ตั้งแต่วางแผนพร้อมบอกเหตุผล — ยกเว้นเมื่อสูตรที่รันผ่านจริงบอกมา"""
    report = qwen_coder_report()
    fit = analyze(report, STACKED)
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": SPARK_VLLM_IMAGE, "rationale": "x"}))
    hardened = harden_plan(plan, report, fit)
    assert hardened.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    assert any("FP4" in w and "kernel" in w for w in hardened.warnings), hardened.warnings

    glm = qwen_coder_report(repo_id="orcarouter/GLM-5.3-Flash-Uncensored-NVFP4", model_type="glm5_next",
                            architecture="Glm5NextForConditionalGeneration")
    plan = DeploymentPlan.model_validate(llm_plan_dict(glm))          # 61fc… = image NVFP4 ตัวเดิม
    assert harden_plan(plan, glm, analyze(glm, STACKED)).runtime.image_ref == SPARK_VLLM_NIGHTLY_IMAGE

    # bf16 ธรรมดาบน Spark: nvcr คือคำตอบที่ถูก ต้องคงไว้
    llama = qwen_coder_report(repo_id="meta-llama/Llama-3.3-70B-Instruct", quantization=None,
                              architecture="LlamaForCausalLM", weight_bytes=131 * GIB)
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        llama, runtime={"engine": "vllm", "image_ref": SPARK_VLLM_IMAGE, "rationale": "x"}))
    assert harden_plan(plan, llama, analyze(llama, STACKED)).runtime.image_ref == SPARK_VLLM_IMAGE

    # สูตรที่รันผ่านจริงชนะเสมอ — ถ้าลูกค้าพิสูจน์แล้วว่า nvcr ตัวนั้นรัน NVFP4 ของเขาได้ ก็คงไว้
    _write_synced([{"match": QWEN_CODER, "engine": "vllm", "image": SPARK_VLLM_IMAGE,
                    "source": "customer", "validated_on": "their spark"}])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == SPARK_VLLM_IMAGE


def test_an_image_pin_from_the_llm_or_a_stale_plan_is_not_trusted():
    """LLM ใส่ `image_pin: sha256:dddd…` มาเอง (มโน) → harden เห็นว่ามี pin แล้วจึงไม่ resolve ซ้ำ → template
    เขียน `vllm/vllm-openai@sha256:dddd…` → docker pull "manifest unknown" ทุกครั้ง · และสูตรที่เปลี่ยน
    image_ref ทับของ LLM ต้องล้าง pin ของ image เก่าด้วย ไม่งั้น repo ใหม่ + digest เก่า"""
    report = qwen_coder_report()
    fit = analyze(report, STACKED)
    junk = "sha256:" + "d" * 64
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": "vllm/vllm-openai:v0.9.2", "image_pin": junk,
                         "rationale": "x"}))
    hardened = harden_plan(plan, report, fit)
    assert hardened.runtime.image_pin != junk, hardened.runtime.image_pin
    assert any("pin" in w.lower() or "digest" in w for w in hardened.warnings), hardened.warnings

    # image ที่ตรึง digest ในชื่ออยู่แล้ว: pin = digest ในชื่อ (ไม่ใช่ของ LLM)
    good = "sha256:" + "a" * 64
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": f"vllm/vllm-openai@{good}", "image_pin": junk,
                         "rationale": "x"}))
    assert harden_plan(plan, report, fit).runtime.image_pin == good

    # สูตรเปลี่ยน image → pin ของ image เก่าต้องไม่ติดมา
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": "vllm/vllm-openai:v0.9.2", "image_pin": junk,
                         "rationale": "x"}))
    apply_recipe(plan, find_recipe(QWEN_CODER), "unified")
    assert plan.runtime.image_ref == "avarok/dgx-vllm-nvfp4-kernel:latest"
    assert plan.runtime.image_pin is None


def test_context_never_exceeds_native_even_when_fit_has_no_ceiling():
    """Hub ไม่รายงานขนาดไฟล์ → fit ไม่มี recommended_context → harden ไม่มีเพดานให้ clamp · LLM ตั้ง 262,144
    ให้ Llama-3.3-70B (native 131,072) → controller ปฏิเสธตอน validate (หรือ vLLM ตายที่ ModelConfig)
    ทั้งที่ native อยู่ในรายงานตั้งแต่แรก (เคสจริง 2026-09-05 msi-4/msi-5 แต่ทางนั้นมาจาก lmds set)"""
    report = qwen_coder_report(repo_id="meta-llama/Llama-3.3-70B-Instruct", quantization=None,
                               architecture="LlamaForCausalLM", weight_bytes=None, context_length=131_072)
    fit = analyze(report, STACKED)
    assert fit.recommended_context is None
    plan = DeploymentPlan.model_validate(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": SPARK_VLLM_IMAGE, "rationale": "x"},
        serving={"context": 262_144, "max_output_tokens": 8192}))
    hardened = harden_plan(plan, report, fit)
    assert hardened.serving.context <= 131_072, hardened.serving.context
    assert any("native" in w for w in hardened.warnings), hardened.warnings


# ── 4. brain: embedding + stacked หลุดมาทาง LLM ────────────────────────────────────────────
def test_embedding_plus_stacked_is_refused_on_the_llm_path_too():
    """rule-based ปฏิเสธ embedding+stacked แล้ว (0.6.0) แต่ทาง LLM harden แค่ตั้ง task=embed + topology=stacked
    → render ด้วย template stacked ซึ่งไม่รู้จักโหมด pooling → ได้ chat server ให้โมเดล embedding เงียบ ๆ"""
    report = qwen_coder_report(repo_id="Qwen/Qwen3-Embedding-8B", task="embed", quantization=None,
                               weight_bytes=16 * GIB)
    fit = analyze(report, STACKED)
    provider = FakeProvider([json.dumps(llm_plan_dict(
        report, runtime={"engine": "vllm", "image_ref": SPARK_VLLM_IMAGE, "rationale": "x"}))])
    with pytest.raises(PlanError, match="embedding"):
        build_plan(report, fit, provider)
    # single ยังผ่านตามเดิม
    single = analyze(report, PRESETS["dgx-spark-single"])
    provider = FakeProvider([json.dumps(llm_plan_dict(
        report, topology="single",
        runtime={"engine": "vllm", "image_ref": SPARK_VLLM_IMAGE, "rationale": "x"}))])
    assert build_plan(report, single, provider).task == "embed"


# ── 5. recipes: prefix ของ match ต้องหยุดที่ขอบชื่อ ─────────────────────────────────────────
def test_recipe_prefix_matches_only_at_a_name_boundary():
    """`match: zai-org/GLM-4.7-Flash` ตั้งใจครอบ variant (`…-Flash-NVFP4`) แต่ startswith ล้วนครอบ
    `…-Flashlight`/`…-FlashX` ที่เป็นคนละโมเดลด้วย → ได้ image/env/parser ของโมเดลอื่นเงียบ ๆ"""
    _write_synced([{"match": "someorg/GLM-4.7-Flash", "engine": "vllm", "source": "t", "validated_on": "t",
                    "tool_calling": {"enabled": True, "parser": "glm47"}}])
    assert find_recipe("someorg/GLM-4.7-Flash") is not None
    assert find_recipe("someorg/glm-4.7-flash-nvfp4") is not None
    assert find_recipe("someorg/GLM-4.7-Flash_v2") is not None
    assert find_recipe("someorg/GLM-4.7-Flashlight") is None
    assert find_recipe("someorg/GLM-4.7-FlashX-NVFP4") is None


# ── 6. inspector + brain: NVFP4 ที่ตั้งชื่อแบบ NVIDIA (-FP4) / config ของ ModelOpp ────────────
def test_nvidia_fp4_naming_and_modelopt_config_are_recognised_as_nvfp4():
    """repo ทางการของ NVIDIA ตั้งชื่อ `…-FP4` ไม่ใช่ `…-NVFP4` และ config.json บอกแค่
    `quant_method: modelopt` (quant_algo อยู่ใน hf_quant_config.json) → inspector รายงาน quantization="modelopt"
    → is_nvfp4 ไม่จับ → บน Spark ได้ nvcr 26.05 ไม่มี FP4 kernel + ไม่มี env marlin → ptxas ตายตอน start"""
    import httpx

    from lmds.inspector import HfClient, inspect_model
    from lmds.resolver import parse_source

    def client_for(config: dict, hf_quant: dict | None):
        files = {"config.json": json.dumps(config)}
        if hf_quant is not None:
            files["hf_quant_config.json"] = json.dumps(hf_quant)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith("/api/models/"):
                return httpx.Response(200, json={
                    "sha": "abc", "gated": False, "private": False, "tags": [],
                    "siblings": [{"rfilename": "model.safetensors", "lfs": {"size": 40_000_000_000}},
                                 {"rfilename": "config.json", "size": 700}]})
            name = request.url.path.split("/abc/")[-1]
            return httpx.Response(200, content=files[name].encode()) if name in files else httpx.Response(404)

        return HfClient(client=httpx.Client(transport=httpx.MockTransport(handler)))

    base = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "max_position_embeddings": 131072,
            "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "hidden_size": 8192}
    # ModelOpt: quant_algo อยู่ใน config.json
    report = inspect_model(parse_source("nvidia/Llama-3.3-70B-Instruct-FP4"), client_for(
        {**base, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}, None))
    assert "nvfp4" in (report.quantization or "").lower(), report.quantization
    # ModelOpt: config.json บอกแค่ modelopt · quant_algo อยู่ใน hf_quant_config.json
    report = inspect_model(parse_source("nvidia/Llama-3.3-70B-Instruct-FP4"), client_for(
        {**base, "quantization_config": {"quant_method": "modelopt"}},
        {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}))
    assert "nvfp4" in (report.quantization or "").lower(), report.quantization
    # llm-compressor: quant_method compressed-tensors · format บอกว่าเป็น nvfp4
    report = inspect_model(parse_source("RedHatAI/Llama-3.3-70B-Instruct-NVFP4"), client_for(
        {**base, "quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}},
        None))
    assert "nvfp4" in (report.quantization or "").lower(), report.quantization
    # fp8 ธรรมดายังเป็น fp8
    report = inspect_model(parse_source("nvidia/Llama-3.3-70B-Instruct-FP8"), client_for(
        {**base, "quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}}, None))
    assert report.quantization == "fp8"

    # ชื่อแบบ NVIDIA (-FP4) แม้ quantization อ่านได้แค่ "modelopt" ก็ต้องนับเป็น NVFP4
    fp4 = qwen_coder_report(repo_id="nvidia/Llama-3.3-70B-Instruct-FP4", quantization="modelopt",
                            architecture="LlamaForCausalLM")
    assert is_nvfp4(fp4)
    plan = harden_plan(rule_based_plan(fp4, analyze(fp4, STACKED)), fp4, analyze(fp4, STACKED))
    assert plan.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    assert plan.serving.extra_env.get("VLLM_NVFP4_GEMM_BACKEND") == "marlin"
    # MXFP4 (gpt-oss) ไม่ใช่ NVFP4 — kernel คนละชุด อย่าเหมารวม
    mx = qwen_coder_report(repo_id="openai/gpt-oss-120b", quantization="mxfp4", architecture="GptOssForCausalLM")
    assert not is_nvfp4(mx)
    # families.nvfp4_on_sm121 (fleet.suggest ใช้เติม bundle เก่า) ต้องเห็นตรงกัน
    from lmds.brain.families import nvfp4_on_sm121

    assert nvfp4_on_sm121("nvidia/Llama-3.3-70B-Instruct-FP4", "modelopt", "vllm", "unified").image
    assert nvfp4_on_sm121("openai/gpt-oss-120b", "mxfp4", "vllm", "unified").image is None


# ── 7. renderer: served_model_name ชนกันข้ามเจ้าของ ────────────────────────────────────────
def test_a_second_owner_of_the_same_model_name_gets_its_own_served_name(tmp_path):
    """0.6.1 แยกโฟลเดอร์ให้ `nvidia/X` กับ `ucbye/X` แล้ว (slug x / x-nvidia) แต่ทั้งคู่ยังเสิร์ฟชื่อ `x`
    → gateway (LiteGate/bifrost) ที่รวมโมเดลจากทั้งฟลีตด้วยชื่อ เห็นสองตัวชื่อเดียวกันแล้ว route มั่ว ·
    ชื่อที่เสิร์ฟของ bundle ที่ต้องเลี่ยงชื่อโฟลเดอร์ ต้องเลี่ยงแบบเดียวกัน (ตั้งเองได้ด้วย lmds set --model-id)"""
    from lmds.generator import render_bundle
    from tests.test_review_templates import _plan, _safetensors_report

    def render(repo_id, target):
        report = _safetensors_report(repo_id=repo_id, revision_sha="rev-" + repo_id.split("/")[0])
        plan, fit = _plan(report, target)
        return plan, render_bundle(plan, report, fit, tmp_path)

    render("ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "dgx-spark-single")
    plan, second = render("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "dgx-spark-stacked")
    assert second.directory.name == "nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia"
    assert plan.served_model_name == "nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia"
    text = second.controller.read_text(encoding="utf-8")
    assert 'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia}"' in text
    assert any("served" in w.lower() or "ชื่อที่เสิร์ฟ" in w for w in plan.warnings), plan.warnings
