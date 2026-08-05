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
}

# DGX Spark (GB10 / SM121 / ARM64) ใช้ image ของ NGC ที่ NVIDIA build มาให้เครื่องนี้โดยเฉพาะ
# image upstream มี manifest arm64 ก็จริง แต่ไม่ได้ build kernel สำหรับ SM121 —
# controller ที่ทีมรันจริงบน Spark ทุกตัวใช้ NGC ทั้งหมด (26.05-py3 / 26.06-py3)
SPARK_VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.05-py3"


def default_image(engine: Engine, memory_model) -> str:
    """image ตั้งต้นตามเครื่องเป้าหมาย — unified memory = DGX Spark"""
    if engine is Engine.VLLM and getattr(memory_model, "value", memory_model) == "unified":
        return SPARK_VLLM_IMAGE
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


def rule_based_plan(report: ModelReport, fit: FitReport) -> DeploymentPlan:
    engine = Engine.LLAMACPP if report.artifact_type is ArtifactType.GGUF else Engine.VLLM
    topology = topology_for_target(fit.target_name)

    context = fit.recommended_context or 8192
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

    # สูตรที่รันผ่านจริงมาก่อนค่าตั้งต้นเสมอ — นี่คือสิ่งที่ทดแทน LLM ให้เครื่องที่ไม่มี provider
    recipe = find_recipe(report.repo_id)
    if recipe is not None:
        plan = apply_recipe(plan, recipe, fit.memory_model.value)
    return plan
