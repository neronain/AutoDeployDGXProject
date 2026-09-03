"""Rule-based planner — degraded mode เมื่อไม่มี LLM key (FR-4.4) และเป็น fallback มาตรฐาน

ให้ plan ที่ปลอดภัยแบบอนุรักษ์นิยม: engine ตาม decision matrix, ค่าจาก Fit Analyzer ล้วน,
ไม่เปิด feature ที่ต้องการหลักฐานเพิ่ม (tool calling ฯลฯ)
"""

from __future__ import annotations

import re

from lmds.fit import FitReport
from lmds.recipes import find_recipe
from lmds.inspector.report import ArtifactType, ModelReport

from .plan_schema import (
    Confidence,
    DeploymentPlan,
    Engine,
    Fact,
    RuntimeChoice,
    Serving,
    Topology,
)

# image ตั้งต้นต่อ engine — template registry (M5) เป็นผู้ pin digest จริง
DEFAULT_IMAGES = {
    Engine.VLLM: "vllm/vllm-openai:latest",
    Engine.LLAMACPP: "ghcr.io/ggml-org/llama.cpp:server-cuda",
    Engine.SGLANG: "lmsysorg/sglang:latest",
}

# DGX Spark (GB10 / SM121 / ARM64) ใช้ image ของ NGC ที่ NVIDIA build มาให้เครื่องนี้โดยเฉพาะ
# image upstream มี manifest arm64 ก็จริง แต่ไม่ได้ build kernel สำหรับ SM121 —
# controller ที่ทีมรันจริงบน Spark ทุกตัวใช้ NGC ทั้งหมด (26.05-py3 / 26.06-py3)
SPARK_VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.05-py3"
# ฝั่ง SGLang ของ NGC (26.02-py3) ยังมากับ transformers 4.57.1 ซึ่งไม่รู้จักสถาปัตยกรรม
# ที่ออกหลังจากนั้นเลย — วัดจริงบน spark-head 2026-08-14: qwen3_5_moe ไม่อยู่ใน
# CONFIG_MAPPING_NAMES ของ image นั้น แต่รู้จักใน scitrera (5.6.0) และ lmsysorg (5.12.1)
#
# เลือก build ของ DGX Spark ที่ transformers ใหม่พอ แทนที่จะเอา NGC มาเพราะมันคู่กับ
# ของ vLLM · ตัวนี้ทำมาสำหรับเครื่องนี้เหมือนกัน (ชื่อมันบอก) และ pin ที่ v0 ไม่ใช่ latest
#
# ถ้า image ไหนไม่รู้จักสถาปัตยกรรมของโมเดล controller จะหยุดตั้งแต่ก่อน start
# พร้อมบอกให้เปลี่ยน image — ไม่ปล่อยให้ไปตายตอนอ่าน config
SPARK_SGLANG_IMAGE = "scitrera/dgx-spark-sglang-mm:v0"


def default_image(engine: Engine, memory_model) -> str:
    """image ตั้งต้นตามเครื่องเป้าหมาย — unified memory = DGX Spark"""
    unified = getattr(memory_model, "value", memory_model) == "unified"
    if unified and engine is Engine.VLLM:
        return SPARK_VLLM_IMAGE
    if unified and engine is Engine.SGLANG:
        return SPARK_SGLANG_IMAGE
    return DEFAULT_IMAGES[engine]



# ข้อบังคับของสถาปัตยกรรมโมเดล ไม่ใช่การปรับจูน — ไม่ตั้งแล้ว vLLM ตายตอน load_model
# เช่น DeepSeek V4 ใช้ attention layout `fp8_ds_mla`/`nvfp4_ds_mla` ซึ่งบังคับ kv-cache เป็น fp8:
#   AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto
# rule-based ไม่มี LLM ไปค้นให้ ความรู้แบบนี้จึงต้องเขียนไว้ตรง ๆ
ARCH_REQUIREMENTS: dict[str, dict] = {
    "deepseek-v4": {"kv_cache_dtype": "fp8"},
}


def arch_requirements(repo_id: str) -> dict:
    key = repo_id.lower().replace("_", "-")
    for marker, requirements in ARCH_REQUIREMENTS.items():
        if marker in key:
            return requirements
    return {}


# กับดักที่ไม่ใช่ "ค่าที่ต้องตั้ง" แต่ควรเตือนก่อน deploy — สกัดจากการรันจริงบน DGX Spark
# (งานวิจัย Qwen3.5-122B บน SM121 + เคสของทีม) · rule-based ไม่มี LLM ไปค้นให้ จึงเขียนไว้ตรง ๆ
def arch_notes(repo_id: str, quantization: str = "",
               hybrid_attention: bool = False) -> list[str]:
    key = repo_id.lower().replace("_", "-")
    quant = (quantization or "").lower()
    notes: list[str] = []
    if "nvfp4" in key or "nvfp4" in quant or "fp4" in quant:
        notes.append(
            "NVFP4/FP4 บน SM121 (GB10) รันได้เมื่อ image มี FP4 kernel ของ sm_121 และบังคับ Marlin — "
            "พิสูจน์แล้ว 2026-09-03 บน spark-head: Qwen3-Coder-Next-NVFP4-GB10 (MoE 512 expert) บน "
            "vllm/vllm-openai:cu130-nightly@3dbe092e + env VLLM_NVFP4_GEMM_BACKEND=marlin "
            "VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_MARLIN_USE_ATOMIC_ADD=1 "
            "ได้ 61 tok/s เดี่ยว / 103 tok/s รวม 3 สาย · ตั้ง env ผ่าน --engine-env"
        )
        notes.append(
            "ถ้า image ไม่มี FP4 kernel ของ sm_121 จะ**ล้มตั้งแต่ start ไม่ใช่แค่ช้า** — เคสจริง "
            "2026-08-20 บน msi-6: ptxas ปฏิเสธ `cvt with .e2m1x2 not supported on .target sm_121` "
            "ตอน JIT cutlass_fused_moe แล้ว engine core ตายก่อน health · env marlin อย่างเดียว"
            "ไม่พอกับ image นั้น เพราะ vLLM import flashinfer cutlass fused-MoE (ซึ่ง JIT ทันที) "
            "ก่อนจะดู env — ต้องปิดด้วย VLLM_USE_FLASHINFER_MOE_FP4=0 ร่วมด้วย หรือใช้ image "
            "ที่ build มาให้ (avarok/dgx-vllm-nvfp4-kernel, dspark-vllm-gx10)"
        )
    # จับชื่อ 3.6 ด้วย ไม่ใช่รอ hybrid_attention จากการอ่านไฟล์อย่างเดียว — repo ที่ inspect
    # ไม่ผ่าน (เน็ตสะดุด/ไฟล์ไม่ครบ) จะได้คำเตือนเหมือนกัน
    if (hybrid_attention or "qwen3.5" in key or "qwen3-5" in key
            or "qwen3.6" in key or "qwen3-6" in key or "deltanet" in key):
        notes.append(
            "Qwen3.5 (DeltaNet hybrid attention): อย่าเปิด --enable-prefix-caching (output ผิด) · "
            "kv-cache fp8 ได้ผลน้อยบน SM121"
        )
        notes.append(
            "Qwen3.5/3.6 มี MTP head ในตัว → เปิด speculative decoding ได้ฟรี · "
            "vLLM: --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' "
            "(ชื่อเจาะจงรุ่นอย่าง qwen3_next_mtp ยัง deprecated อยู่ vLLM แปลงกลับเป็น mtp ให้เอง) · "
            "llama.cpp: --spec-type draft-mtp --spec-draft-n-max 2 — คนละแฟล็กกันสิ้นเชิง "
            "ห้ามลอกของ vLLM ไปใส่ · bit-exact ที่ temp=0 (rejection sampling) · single Spark ได้ ~2-3x throughput"
        )
    if ("qwen3" in key or "qwen-3" in key) and "coder" not in key:
        notes.append(
            "ถ้าเปิด tool calling: Qwen3/3.5/3.6 ใช้ --tool-call-parser qwen3_xml หรือ qwen3_coder "
            "— สองชื่อนี้ map ไป Qwen3EngineToolParser ตัวเดียวกันใน vLLM รุ่นใหม่ ใส่ชื่อไหนก็ได้ "
            "(ที่พังคือ hermes ซึ่งอ่าน syntax ของตระกูลนี้ไม่ออก) · และต้องมี --reasoning-parser qwen3 "
            "ด้วย ไม่งั้นส่วน thinking จะไม่ถูกแยกออกจากคำตอบ"
        )
    # Gemma 4 พ่น `<|tool_call>call:name{...}` ซึ่งมีแต่ parser ชื่อ gemma4 ที่อ่านออก ·
    # เคสจริง (msi-2, 2026-09-02): ตั้ง hermes ไว้ vLLM ขึ้นปกติและตอบ 200 ทุกครั้ง แต่คืน
    # finish_reason=stop + tool_calls=null แล้วยัด call ดิบไว้ใน content — จากข้างนอกดูเหมือน
    # โมเดลใช้ tool ไม่เป็น ทั้งที่ผิดแค่ชื่อ parser · ตอนนั้น LMDS ไม่เคยแนะ Gemma ไว้เลย
    # คนจึงเดาไปที่ hermes ซึ่งเป็นค่าที่คนมักใช้เป็นค่าเริ่มต้น
    if "gemma-4" in key or "gemma4" in key:
        notes.append(
            "ถ้าเปิด tool calling: Gemma 4 ใช้ --tool-parser gemma4 เท่านั้น (hermes อ่าน "
            "syntax ของมันไม่ออก) · ตั้งผิดจะไม่ error ให้เห็น — vLLM ตอบ 200 แต่ tool_calls "
            "เป็น null และ call ดิบ <|tool_call> โผล่ใน content · พิสูจน์ด้วยการยิง request "
            "ที่มี tools จริงแล้วดู finish_reason ต้องได้ tool_calls ไม่ใช่ stop"
        )
    # Nemotron-3.x เป็น hybrid Mamba/SSM (ไม่ใช่ full attention) เหมือน DeltaNet — มี flag เฉพาะ
    if "nemotron-3" in key or "nemotron3" in key or "lightning" in key:
        notes.append(
            "Nemotron-3.x = hybrid Mamba/SSM → ต้องตั้ง --mamba-backend flashinfer + "
            "--mamba-ssm-cache-dtype (float16 Lightning / float32 Super) + --mamba-cache-mode align · "
            "เป็น hybrid attention เหมือน DeltaNet ให้ระวัง prefix-caching และการประเมิน KV"
        )
    # DeepSeek-V4-Flash: MLA + sparse indexer + DSpark draft — คนละโลกกับโมเดล dense ทั่วไป
    if "deepseek-v4" in key or "deepseek-v4-flash" in key:
        notes.append(
            "DeepSeek-V4-Flash (SM121): kv-cache = fp8_ds_mla (หรือ nvfp4_ds_mla ถ้า image มี flashmla "
            "fp8-kernel fix ไม่งั้น decode ตกเหลือ ~1 tok/s ที่ ctx ยาว) · --block-size 256 · "
            "speculative method=dspark (k≥5, มี draft head แยก) ไม่ใช่ mtp ธรรมดา · "
            "ข้าม 2 Spark = TP=2 executor mp (ไม่ใช่ Ray), worker headless start ก่อน head"
        )
    return notes

def slugify(repo_id: str) -> str:
    name = repo_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "model"


def topology_for_target(target_name: str) -> Topology:
    """topology เป็นสมบัติของ *target* (กี่ node/GPU) ไม่ใช่การตัดสินใจของ LLM

    - ชื่อ target มี 'stacked' → STACKED (multi-node เช่น dgx-spark-stacked)
    - มี 'dual'/'multi'       → MULTI_GPU (หลาย GPU ในเครื่องเดียว เช่น rtx-*-dual)
    - นอกนั้น                  → SINGLE
    """
    name = target_name.lower()
    if "stacked" in name:
        return Topology.STACKED
    if "dual" in name or "multi" in name:
        return Topology.MULTI_GPU
    return Topology.SINGLE


def build_facts(report: ModelReport) -> list[Fact]:
    facts = [
        Fact(claim=f"artifact เป็น {report.artifact_type.value}", source="hub-api", confidence=Confidence.VERIFIED),
        Fact(claim=f"revision pinned: {report.revision_sha}", source="hub-api", confidence=Confidence.VERIFIED),
    ]
    if report.weight_bytes:
        facts.append(Fact(
            claim=f"ขนาด weight รวม {report.weight_bytes / 1e9:.1f} GB",
            source="hub-api file sizes", confidence=Confidence.VERIFIED,
        ))
    if report.context_length:
        facts.append(Fact(
            claim=f"native context {report.context_length:,} tokens",
            source="config.json/gguf-header", confidence=Confidence.VERIFIED,
        ))
    if report.license:
        facts.append(Fact(claim=f"license: {report.license}", source="hub-api", confidence=Confidence.VERIFIED))
    if report.quantization:
        facts.append(Fact(
            claim=f"quantization: {report.quantization}",
            source="config/ชื่อไฟล์", confidence=Confidence.VERIFIED,
        ))
    return facts



def apply_recipe(plan: DeploymentPlan, recipe, memory_model: str = "") -> DeploymentPlan:
    """เติมค่าจากสูตรที่รันผ่านจริงลง plan — ทับเฉพาะสิ่งที่สูตรระบุไว้เท่านั้น

    context/max_output ไม่แตะ เพราะต้องมาจากการวิเคราะห์หน่วยความจำของ *เครื่องเป้าหมาย*
    ไม่ใช่ค่าคงที่จากเครื่องที่เคยรัน
    """
    # สูตรของ engine อื่นใช้ไม่ได้กับ controller ที่เรากำลังสร้าง — ใส่ image ของ SGLang
    # ลง controller ของ vLLM แล้ว bundle จะผ่าน gate ทุกด่านแต่ start ไม่ขึ้นเลย
    if recipe.engine and recipe.engine != plan.runtime.engine.value:
        plan.warnings.append(
            f"โมเดลนี้ทดสอบมาด้วย {recipe.engine} ซึ่ง LMDS ยังไม่ได้ generate — "
            f"bundle นี้เป็น {plan.runtime.engine.value} · ดูสูตรเต็ม: {recipe.source}"
        )
        return plan

    if recipe.image and recipe.image_applies_to(memory_model):
        plan.runtime.image_ref = recipe.image
    elif recipe.image:
        # image ที่ทดสอบมาเป็นของอีกสถาปัตยกรรม — บอกไว้ ดีกว่าใช้เงียบ ๆ แล้วพังตอน start
        plan.warnings.append(
            f"สูตรนี้ทดสอบด้วย image {recipe.image} บนเครื่องคนละแบบ — ใช้ค่าตั้งต้นแทน"
        )

    serving_fields = set(Serving.model_fields)
    extra: list[str] = []
    for key, value in (recipe.serving or {}).items():
        if key in serving_fields:
            setattr(plan.serving, key, value)
        elif value is True:
            extra.append(f"--{key.replace('_', '-')}")
        else:
            extra.extend([f"--{key.replace('_', '-')}", str(value)])
    if extra:
        plan.serving.extra_flags = list(dict.fromkeys(plan.serving.extra_flags + extra))

    if recipe.env:
        plan.serving.extra_env = {**plan.serving.extra_env, **{k: str(v) for k, v in recipe.env.items()}}

    tools = recipe.tool_calling or {}
    if tools.get("enabled"):
        plan.tool_calling.enabled = True
        plan.tool_calling.parser = tools.get("parser") or plan.tool_calling.parser
        if tools.get("chat_template"):
            plan.tool_calling.chat_template_override = tools["chat_template"]

    thinking = recipe.reasoning or {}
    if thinking.get("enabled"):
        plan.reasoning.enabled = True
        plan.reasoning.parser = thinking.get("parser") or plan.reasoning.parser

    plan.runtime.rationale += f" + สูตรที่รันผ่านจริง ({recipe.validated_on or recipe.source})"
    # rule-based ไม่มีการวิจัยก็จริง แต่สูตรนี้มาจากการรันบนฮาร์ดแวร์ — คำเตือนเดิมจึงไม่ตรงแล้ว
    plan.warnings = [w for w in plan.warnings if "ไม่มีการวิจัย parser" not in w]
    plan.warnings.insert(0, f"ใช้สูตรที่รันผ่านจริง: {recipe.label or recipe.match} — {recipe.source}")
    for note in recipe.notes or []:
        if note not in plan.warnings:
            plan.warnings.append(note)
    return plan


def rule_based_plan(report: ModelReport, fit: FitReport,
                    engine: Engine | None = None) -> DeploymentPlan:
    # engine ที่ผู้ใช้เลือกมาชนะการเดา แต่ GGUF ยังบังคับ llama.cpp เสมอ —
    # vLLM กับ SGLang อ่านไฟล์ GGUF ไม่ได้ ยอมตามคำขอคือส่ง bundle ที่ start ไม่ขึ้นให้
    if report.artifact_type is ArtifactType.GGUF:
        engine = Engine.LLAMACPP
    elif engine is None:
        engine = Engine.VLLM
    topology = topology_for_target(fit.target_name)

    # llama.cpp: fit หาร context ด้วย concurrency ที่ขอมาแล้ว (ค่าต่อ 1 sequence) แต่ --ctx-size
    # ของ llama-server คือ pool ที่แบ่งให้ทุก slot เท่า ๆ กัน → คูณกลับ และตั้ง slot = concurrency
    # เดิมขอ --concurrency 4 แล้วได้ slot เดียวกับ context หารสี่ = แย่กว่าไม่ใส่ (รีวิว 2026-09-04)
    per_sequence = fit.recommended_context or 8192
    slots = max(1, int(getattr(fit, "concurrency", 1) or 1)) if engine is Engine.LLAMACPP else 4
    context = per_sequence * slots if engine is Engine.LLAMACPP else per_sequence
    plan = DeploymentPlan(
        model_id=report.repo_id,
        revision=report.revision_sha,
        served_model_name=slugify(report.repo_id),
        artifact_type=report.artifact_type,
        selected_gguf=report.selected_gguf,
        facts=build_facts(report),
        runtime=RuntimeChoice(
            engine=engine,
            image_ref=default_image(engine, fit.memory_model),
            rationale="rule-based: เลือกตาม decision matrix (GGUF→llama.cpp, safetensors→vLLM)"
            " + image ตามเครื่องเป้าหมาย (unified memory → NGC build ของ DGX Spark)",
        ),
        topology=topology,
        serving=Serving(
            context=context,
            max_output_tokens=fit.client_output_default,
            # llama.cpp แบ่ง --ctx-size แบบตายตัวให้ทุก slot ต่างจาก vLLM ที่แชร์ KV
            # แบบ dynamic — 4 slot จึงแปลว่าแต่ละ request ได้ context เหลือหนึ่งในสี่
            # ส่วน fit คำนวณมาที่ concurrency=1 ค่าที่แผนสัญญาไว้จึงเป็นค่าของ slot เดียว
            #
            # เคสจริง 2026-08-13: Muse-Glimmer แผนบอก 131,072 แต่ /props รายงาน 32,768
            # เพราะ 131,072 ถูกหารด้วย 4 slot ที่ไม่มีใครขอ
            #
            # ต้องการ concurrency จริง: `--concurrency N` → slot = N และ context = N × ค่าต่อ slot
            max_num_seqs=slots,
            **arch_requirements(report.repo_id),
        ),
        special_files=list(report.trust_remote_code_files),
        warnings=[
            "plan นี้สร้างแบบ rule-based (ไม่มี LLM) — ไม่มีการวิจัย parser/feature เชิงลึก",
            "runtime image ยังไม่ pin digest — ต้อง pin ก่อน deploy จริง",
        ],
        generator="rule-based",
    )
    if report.has_chat_template is False:
        plan.warnings.append("ไม่พบ chat template — ต้องระบุ template เองตอนใช้งาน chat")
    if report.trust_remote_code_files:
        plan.warnings.append("repo มีไฟล์ remote code — review ก่อนเปิด --trust-remote-code (ต้องอนุมัติเอง)")
    for note in arch_notes(report.repo_id, report.quantization or "",
                           hybrid_attention=report.hybrid_attention):
        if note not in plan.warnings:
            plan.warnings.append(note)

    # ตระกูลที่รู้แน่ → ใส่ parser ให้เป็น *ค่า* ไม่ใช่แค่คำเตือน · bundle จะได้เกิดมาถูก
    # ไม่ต้องให้คนที่ไม่รู้ไปพิมพ์ชื่อเอง (ผู้ใช้ 2026-09-04: "กลัวใส่ผิด แล้วไม่มีให้ใช้งาน")
    # สูตรที่รันผ่านจริง (ด้านล่าง) ยังชนะค่านี้เสมอ · ชื่อทั้งหมดผ่าน _harden_parsers แน่นอน (มีเทส)
    from lmds.brain.families import parsers_for

    choice = parsers_for(report.repo_id, report.architecture or "", engine.value)
    if choice.tool:
        plan.tool_calling.enabled = True
        plan.tool_calling.parser = choice.tool
        plan.warnings.append(
            f"เปิด tool calling ให้แล้ว: --tool-call-parser {choice.tool}"
            + (f" · --reasoning-parser {choice.reasoning}" if choice.reasoning else "")
            + f" — {choice.why} · เปลี่ยน/ปิดได้ด้วย lmds set --tool-parser / --reasoning-parser"
        )
    if choice.reasoning:
        plan.reasoning.enabled = True
        plan.reasoning.parser = choice.reasoning
    # สูตรที่รันผ่านจริงมาก่อนค่าตั้งต้นเสมอ — นี่คือสิ่งที่ทดแทน LLM ให้เครื่องที่ไม่มี provider
    recipe = find_recipe(report.repo_id)
    if recipe is not None:
        plan = apply_recipe(plan, recipe, fit.memory_model.value)
    return plan
