"""Rule-based planner — degraded mode เมื่อไม่มี LLM key (FR-4.4) และเป็น fallback มาตรฐาน

ให้ plan ที่ปลอดภัยแบบอนุรักษ์นิยม: engine ตาม decision matrix, ค่าจาก Fit Analyzer ล้วน,
ไม่เปิด feature ที่ต้องการหลักฐานเพิ่ม (tool calling ฯลฯ)
"""

from __future__ import annotations

import re

from lmds.fit import FitReport
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
            image_ref=DEFAULT_IMAGES[engine],
            rationale="rule-based: เลือกตาม decision matrix (GGUF→llama.cpp, safetensors→vLLM)",
        ),
        topology=topology,
        serving=Serving(
            context=context,
            max_output_tokens=fit.client_output_default,
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
    return plan
