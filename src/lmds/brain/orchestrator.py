"""Orchestrator ของขั้นวางแผน: evidence → LLM (หรือ rule-based) → validate → harden → session log

harden_plan คือด่านสุดท้าย: ต่อให้ LLM ตอบอะไรมา ค่าที่ขัดกับข้อเท็จจริงเชิงคำนวณ
จะถูกบังคับกลับเสมอ (revision, context, engine, flag allowlist)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from lmds.config.paths import sessions_dir
from lmds.fit import FitReport
from lmds.inspector.report import ArtifactType, ModelReport
from lmds.recipes import find_recipe
from lmds.secrets import redact

from .allowlists import is_known_image, split_flags
from .plan_schema import DeploymentPlan, Engine, PlanError
from .prompts import build_system_prompt, build_user_prompt
from .providers import LlmProvider
from .rulebased import apply_recipe, rule_based_plan

MAX_ATTEMPTS = 3


def _evidence(report: ModelReport, fit: FitReport) -> dict[str, Any]:
    return {
        "model": report.model_dump(mode="json"),
        "fit": fit.model_dump(mode="json"),
    }


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# llama.cpp แบ่ง --ctx-size เท่า ๆ กันให้ทุก slot (--parallel) — output ที่ใหญ่กว่า
# context ต่อ slot คือค่าที่เป็นไปไม่ได้ · ค่าเดียวกับ TEMPLATE_OVERHEAD_TOKENS ในสคริปต์
_TEMPLATE_OVERHEAD_TOKENS = 2048
_MIN_OUTPUT_TOKENS = 512
# ต้องเหลือที่ให้ *คำถาม* ด้วย — ตั้ง output จนพอดีเป๊ะแปลว่า input ได้ 0 token
# ซึ่ง controller ปฏิเสธเหมือนกัน (`input_budget > 0`)
_MIN_INPUT_TOKENS = 1024


def _fit_output_into_slots(plan: DeploymentPlan) -> None:
    """ทำให้ context / slots / max_output_tokens สอดคล้องกันจริง

    เคสจริง: context 16,384 · slots 4 · output 8,192 — bundle ผ่าน gate ทุกด่าน แต่
    `client-config` ของตัวมันเองปฏิเสธทันทีว่า "context ต่อ slot เล็กเกิน (4096 = 16384/4)"
    เพราะ 4,096 - 8,192 ติดลบ · bundle ที่ขัดแย้งกับตัวเองไม่ควรออกจากโรงงานตั้งแต่แรก
    """
    if plan.runtime.engine is not Engine.LLAMACPP:
        return          # vLLM แชร์ KV cache แบบ dynamic ไม่ได้หารตาม slot
    slots = max(1, plan.serving.max_num_seqs)
    if slots > 1:
        # ไม่ไปแก้ค่าที่คนตั้งมาเอง แต่ต้องพูดราคาออกมาให้ชัด — README/profile/banner
        # ทุกที่แสดง context ของ plan ซึ่งเป็น *pool* ไม่ใช่ค่าที่แต่ละ request ได้
        plan.warnings.append(
            f"context ต่อ request จะเป็น {plan.serving.context // slots:,} ไม่ใช่ "
            f"{plan.serving.context:,} — llama.cpp แบ่ง --ctx-size ให้ {slots} slot เท่า ๆ กัน"
        )
    per_slot = plan.serving.context // slots
    usable = per_slot - _TEMPLATE_OVERHEAD_TOKENS - _MIN_INPUT_TOKENS
    if usable > plan.serving.max_output_tokens:
        return

    # ลด output ก่อน — ผู้ใช้ปรับ slots เองได้ตอน start แต่ output ที่เป็นไปไม่ได้แก้ไม่ได้
    if usable >= _MIN_OUTPUT_TOKENS:
        if usable == plan.serving.max_output_tokens:
            return          # เท่าเดิมอยู่แล้ว — "ลด 1,024 เหลือ 1,024" ไม่ได้บอกอะไรใคร
        plan.warnings.append(
            f"ลด max_output_tokens จาก {plan.serving.max_output_tokens:,} เหลือ {usable:,} — "
            f"context ต่อ slot มีแค่ {per_slot:,} ({plan.serving.context:,}/{slots})"
        )
        plan.serving.max_output_tokens = usable
        return

    # เหลือน้อยจนแม้แต่ output ขั้นต่ำก็ไม่พอ = จำนวน slot มากเกินไปสำหรับ context เท่านี้
    fits = max(1, plan.serving.context // (
        _MIN_OUTPUT_TOKENS + _TEMPLATE_OVERHEAD_TOKENS + _MIN_INPUT_TOKENS))
    plan.warnings.append(
        f"ลด slots จาก {slots} เหลือ {fits} — context {plan.serving.context:,} "
        f"แบ่ง {slots} slot แล้วเหลือไม่พอสำหรับคำตอบ"
    )
    plan.serving.max_num_seqs = fits
    plan.serving.max_output_tokens = max(
        _MIN_OUTPUT_TOKENS,
        plan.serving.context // fits - _TEMPLATE_OVERHEAD_TOKENS - _MIN_INPUT_TOKENS)


def harden_plan(plan: DeploymentPlan, report: ModelReport, fit: FitReport) -> DeploymentPlan:
    """บังคับข้อเท็จจริงกลับเข้า plan — LLM output เป็น untrusted input (PRD §9.3)"""
    if plan.revision != report.revision_sha:
        plan.warnings.append(f"แก้ revision จาก {plan.revision!r} เป็น SHA ที่ pin จริง")
        plan.revision = report.revision_sha
    if plan.model_id != report.repo_id:
        plan.warnings.append(f"แก้ model_id จาก {plan.model_id!r} เป็น {report.repo_id}")
        plan.model_id = report.repo_id

    # GGUF อ่านได้เฉพาะ llama.cpp · ส่วน safetensors เสิร์ฟได้ทั้ง vLLM และ SGLang
    # จึงบังคับเฉพาะฝั่ง GGUF · เดิมบังคับเป็น vLLM เสมอ ทำให้ผู้ใช้เลือก SGLang ไม่ได้เลย
    if report.artifact_type is ArtifactType.GGUF:
        allowed = {Engine.LLAMACPP}
        expected_engine = Engine.LLAMACPP
    else:
        allowed = {Engine.VLLM, Engine.SGLANG}
        expected_engine = Engine.VLLM
    if plan.runtime.engine not in allowed:
        plan.warnings.append(
            f"แก้ engine จาก {plan.runtime.engine.value} เป็น {expected_engine.value} ตาม artifact จริง"
        )
        plan.runtime.engine = expected_engine

    # image ตั้งต้นต้องตรงกับเครื่องเป้าหมาย — DGX Spark ใช้ NGC ไม่ใช่ upstream
    # (upstream มี manifest arm64 แต่ไม่ได้ build kernel ให้ SM121)
    from .rulebased import default_image

    fallback = default_image(plan.runtime.engine, fit.memory_model)

    if not is_known_image(plan.runtime.engine, plan.runtime.image_ref):
        plan.warnings.append(
            f"image ที่แผนเสนอ ({plan.runtime.image_ref}) ไม่อยู่ใน registry ที่ยอมรับ — "
            f"เปลี่ยนเป็น {fallback}"
        )
        plan.runtime.image_ref = fallback
        plan.runtime.image_pin = None
    else:
        # repo ถูกไม่ได้แปลว่า tag มีอยู่จริง — LLM เคยเสนอ `vllm/vllm-openai:v0.6.3.ss`
        # ซึ่งผ่าน gate ทุกด่านแล้วไปตายตอนรันด้วย "manifest unknown"
        from .registry import tag_exists

        if tag_exists(plan.runtime.image_ref) is False:
            plan.warnings.append(
                f"tag ของ image ที่แผนเสนอ ({plan.runtime.image_ref}) ไม่มีอยู่จริงบน registry — "
                f"เปลี่ยนเป็น {fallback}"
            )
            plan.runtime.image_ref = fallback
            plan.runtime.image_pin = None

    # ตรึง image ที่ digest — tag เคลื่อนที่ได้ digest ไม่เคลื่อน · bundle ที่ทดสอบ
    # ผ่านเมื่อวานจึงไม่กลายเป็นคนละ runtime วันนี้โดยไม่มีอะไรในไฟล์เปลี่ยน
    #
    # ถามไม่ได้ (registry ต้องล็อกอิน / ไม่มีเน็ต) ก็ปล่อยว่าง แล้วใช้ tag ตามเดิม —
    # การห้าม deploy เพราะถาม registry ไม่ได้ แพงกว่าประโยชน์ที่ได้
    if plan.runtime.image_pin is None:
        from .registry import resolve_digest

        plan.runtime.image_pin = resolve_digest(plan.runtime.image_ref)

    _fit_output_into_slots(plan)

    if fit.recommended_context and plan.serving.context > fit.recommended_context:
        plan.warnings.append(
            f"ลด context จาก {plan.serving.context:,} เหลือ {fit.recommended_context:,} ตาม fit analysis"
        )
        plan.serving.context = fit.recommended_context

    # topology เป็นสมบัติของ target (กี่ node/GPU) — บังคับจาก target เสมอ ไม่ให้ LLM เลือกเอง
    from .rulebased import topology_for_target

    expected_topology = topology_for_target(fit.target_name)
    if plan.topology is not expected_topology:
        plan.warnings.append(
            f"แก้ topology จาก {plan.topology.value} เป็น {expected_topology.value} ตาม target {fit.target_name!r}"
        )
        plan.topology = expected_topology

    if plan.tool_calling.parallel:
        plan.tool_calling.parallel = False
        plan.warnings.append("บังคับ parallel_tool_calls=false ตามมาตรฐาน v3.0.0 (ต้องผ่านเทสก่อนเปิด)")

    # รวม flag+ค่าที่ LLM แยก item มา ('--threads','4' → '--threads 4') ก่อนตรวจ allowlist
    from .allowlists import coalesce_flag_tokens

    coalesced = coalesce_flag_tokens(plan.serving.extra_flags)
    allowed, needs_approval = split_flags(plan.runtime.engine, coalesced)
    if plan.runtime.engine is Engine.LLAMACPP:
        # llama.cpp ใหม่: --flash-attn ต้องมีค่า — เติม 'on' ให้ flag ที่ LLM ใส่มาแบบ bare
        from .allowlists import normalize_llamacpp_flags

        allowed = normalize_llamacpp_flags(allowed)
    plan.serving.extra_flags = allowed
    if needs_approval:
        plan.flags_needing_approval = sorted(set(plan.flags_needing_approval + needs_approval))
        plan.warnings.append(
            "flag นอก allowlist ต้องได้รับอนุมัติจากผู้ใช้ก่อนใช้จริง: " + ", ".join(needs_approval)
        )
    _harden_runtime_assets(plan)
    _harden_projector(plan, report)
    _harden_draft(plan, report)
    _harden_moe(plan, report)

    plan.artifact_type = report.artifact_type
    plan.selected_gguf = plan.selected_gguf or report.selected_gguf
    return plan


def _harden_moe(plan: DeploymentPlan, report: ModelReport) -> None:
    """MoE อ่านจากไฟล์ได้ตรง ๆ — ไม่เปิดให้ LLM เดา"""
    plan.moe.experts = report.moe_experts
    plan.moe.experts_active = report.moe_experts_active


def _harden_draft(plan: DeploymentPlan, report: ModelReport) -> None:
    """ไฟล์ MTP เป็นข้อเท็จจริงจาก repo เช่นเดียวกับ mmproj — บังคับให้ตรงของจริง

    เจอจริงกับ HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-*-MTP: LLM เสนอ
    `--mtp mtp-gemma-4-26B-it.gguf` ซึ่งผิดสองชั้น — llama.cpp ไม่มี flag ชื่อ `--mtp`
    (ของจริงคือ `--spec-draft-model` + `--spec-type draft-mtp`) และชื่อไฟล์ตก `-A4B` ไป
    ถ้าปล่อยผ่านคือ start ไม่ขึ้น ถ้าไม่ปล่อยก็เสียความเร็วที่ repo ตั้งใจให้ไปเปล่า ๆ
    """
    if plan.runtime.engine is not Engine.LLAMACPP:
        if plan.speculative.draft_files:
            plan.warnings.append("ตัด draft_files ออก — ใช้ได้เฉพาะ engine llama.cpp")
            plan.speculative.draft_files = []
        return

    available = [v for v in report.gguf_variants if v.is_mtp]
    if not available:
        if plan.speculative.draft_files:
            plan.warnings.append("ตัด draft_files ออก — ไม่พบไฟล์ MTP ใน repo จริง")
            plan.speculative.draft_files = []
        return

    chosen = available[0].filename
    if plan.speculative.draft_files != [chosen]:
        plan.warnings.append(
            f"repo มีไฟล์ MTP — เปิด speculative decoding ให้อัตโนมัติด้วย "
            f"{chosen.rsplit(chr(47), 1)[-1]} (output เหมือนเดิม ได้มาแต่ความเร็ว)"
        )
    plan.speculative.draft_files = [chosen]  # llama-server รับ draft ได้ไฟล์เดียว


def _harden_projector(plan: DeploymentPlan, report: ModelReport) -> None:
    """ไฟล์ mmproj เป็นข้อเท็จจริงจาก repo ไม่ใช่การตัดสินใจ — บังคับให้ตรงของจริงเสมอ

    ครอบคลุมสองทางที่พังได้:
    - LLM ไม่ประกาศ (หรือใช้ --no-llm) ทั้งที่ repo มี mmproj → โมเดล multimodal กลายเป็น
      text-only เงียบ ๆ เพราะ controller ไม่รู้ว่าต้องโหลดอะไร
    - LLM เดาชื่อไฟล์ที่ไม่มีจริง → URL ดาวน์โหลด 404 ตอนผู้ใช้รัน download
    """
    available = [v for v in report.gguf_variants if v.is_mmproj]
    declared = list(plan.multimodal.projector_files)

    if plan.runtime.engine is not Engine.LLAMACPP:
        # vLLM โหลด vision tower จาก safetensors ของ repo อยู่แล้ว ไม่มีไฟล์ projector แยก
        if declared:
            plan.warnings.append("ตัด projector_files ออก — ใช้ได้เฉพาะ engine llama.cpp")
            plan.multimodal.projector_files = []
        return

    if not available:
        if declared:
            plan.warnings.append(
                "ตัด projector_files ออก — ไม่พบไฟล์ mmproj ใน repo จริง: " + ", ".join(declared)
            )
            plan.multimodal.projector_files = []
        return

    by_basename = {v.filename.rsplit("/", 1)[-1]: v for v in available}
    kept = [n for n in declared if n.rsplit("/", 1)[-1] in by_basename]

    if declared and not kept:
        plan.warnings.append(
            f"projector_files ที่แผนเสนอไม่มีอยู่จริง ({', '.join(declared)}) — ใช้ไฟล์จาก repo แทน"
        )
    if not kept:
        chosen = _pick_projector(available, report)
        kept = [chosen.filename]
        if not declared:
            plan.warnings.append(
                f"repo มีไฟล์ mmproj — เปิดโหมด multimodal ให้อัตโนมัติด้วย {kept[0]}"
            )

    plan.multimodal.projector_files = kept[:1]  # llama-server รับ --mmproj ได้ไฟล์เดียว
    if not plan.multimodal.modalities:
        plan.multimodal.modalities = ["image", "text"]


def _pick_projector(available: list, report: ModelReport):
    """เลือก projector ที่คู่กับ weight ที่เราจะรันจริง

    เดิมเลือกไฟล์เล็กสุดเสมอ ด้วยเหตุผลว่า projector ใหญ่กว่าไม่คุ้มหน่วยความจำ
    ซึ่งจริงเมื่อ repo มี projector ตัวเดียวในหลายระดับ quant แต่ repo จำนวนหนึ่ง
    ใส่ projector ของ "โมเดลคนละตัว" ไว้ด้วยกัน แล้วกติกาเล็กสุดจะหยิบผิดตัวเงียบ ๆ

    เคสจริง 2026-08-13 — unsloth/Muse-Glimmer-30B-GGUF มีสามไฟล์:
        mmproj-Muse-Glimmer-30B-BF16.gguf   คู่กับ weight ปกติ
        mmproj-Muse-Glimmer-30B-Q8_0.gguf   คู่กับ weight ปกติ
        mmproj-kquant.gguf                  คู่กับ dflash-kquant.gguf (คนละโมเดล)
    เล็กสุดคือ mmproj-kquant ซึ่งไม่ได้คู่กับ UD-Q8_K_XL ที่เราเลือกไว้เลย

    จึงเลือกตามลำดับนี้: ชื่อที่ใช้ stem เดียวกับ weight ก่อน แล้วค่อยเล็กสุดในกลุ่มนั้น
    ไม่มีตัวไหนเข้าเกณฑ์ค่อยกลับไปใช้เล็กสุดทั้งหมดตามเดิม
    """
    def size(v):
        return (v.size_bytes is None, v.size_bytes or 0)

    selected = (report.selected_gguf or "").rsplit("/", 1)[-1]
    # "Muse-Glimmer-30B-UD-Q8_K_XL.gguf" -> "muse-glimmer-30b"
    stem = selected.lower().removesuffix(".gguf")
    parts = stem.split("-")
    related = []
    while len(parts) >= 2 and not related:
        prefix = "-".join(parts)
        related = [
            v for v in available
            if prefix in v.filename.rsplit("/", 1)[-1].lower()
        ]
        parts.pop()  # ตัดท้ายทีละส่วนจนเจอกลุ่มที่ชื่อร่วมกัน

    return min(related or available, key=size)


def _harden_runtime_assets(plan: DeploymentPlan) -> None:
    """ไฟล์ runtime ภายนอกเป็นโค้ดที่รันจริงใน container — ไม่มีทางเข้า bundle เองได้

    ทุกตัวถูกย้ายไปรออนุมัติเสมอ (แม้ LLM จะใส่มาใน runtime_assets ตรง ๆ) และตัวที่
    URL/ชื่อไฟล์ไม่ผ่าน allowlist ถูกทิ้งพร้อมเหตุผล
    """
    from .allowlists import is_allowed_asset_url, is_safe_asset_filename

    candidates = list(plan.runtime_assets) + list(plan.assets_needing_approval)
    plan.runtime_assets = []

    kept: list = []
    seen: set[str] = set()
    for asset in candidates:
        if asset.filename in seen:
            continue
        seen.add(asset.filename)
        if not is_safe_asset_filename(asset.filename):
            plan.warnings.append(f"ตัดไฟล์ runtime ที่ชื่อไม่ปลอดภัย: {asset.filename!r}")
            continue
        if not is_allowed_asset_url(asset.url):
            plan.warnings.append(
                f"ตัดไฟล์ runtime {asset.filename} — URL ไม่อยู่ใน allowlist ({asset.url})"
            )
            continue
        kept.append(asset)

    plan.assets_needing_approval = kept
    if kept:
        plan.warnings.append(
            "ไฟล์ runtime ภายนอกต้องได้รับอนุมัติจากผู้ใช้ก่อนใช้จริง: "
            + ", ".join(a.filename for a in kept)
        )


def apply_flag_approvals(plan: DeploymentPlan, approved: list[str]) -> DeploymentPlan:
    """ย้าย flag ที่ผู้ใช้อนุมัติแบบ explicit จาก flags_needing_approval → extra_flags

    เรียกได้จากขั้นยืนยันของ deploy เท่านั้น — การอนุมัติเป็นการตัดสินใจของผู้ใช้ ไม่ใช่ LLM
    """
    for flag in approved:
        if flag in plan.flags_needing_approval:
            plan.flags_needing_approval.remove(flag)
            if flag not in plan.serving.extra_flags:
                plan.serving.extra_flags.append(flag)
            plan.warnings.append(f"ผู้ใช้อนุมัติ flag: {flag}")
    return plan


def apply_asset_approvals(plan: DeploymentPlan, approved_filenames: list[str]) -> DeploymentPlan:
    """ย้ายไฟล์ runtime ที่ผู้ใช้อนุมัติ → runtime_assets (ตัวที่ไม่อนุมัติถูกทิ้งไปเลย)"""
    approved = set(approved_filenames)
    still_pending = []
    for asset in plan.assets_needing_approval:
        if asset.filename in approved:
            plan.runtime_assets.append(asset)
            plan.warnings.append(f"ผู้ใช้อนุมัติไฟล์ runtime: {asset.filename} ({asset.url})")
        else:
            still_pending.append(asset)
    plan.assets_needing_approval = still_pending
    return plan


def build_plan(
    report: ModelReport,
    fit: FitReport,
    provider: LlmProvider | None,
    max_attempts: int = MAX_ATTEMPTS,
    engine: Engine | None = None,
) -> DeploymentPlan:
    """provider=None → rule-based (degraded/--no-llm); มี provider → LLM + validate + retry

    `engine` คือสิ่งที่ผู้ใช้เลือกมาเอง (`--engine sglang`) · safetensors เสิร์ฟได้ทั้ง
    vLLM และ SGLang การเดาจึงเป็นแค่ค่าตั้งต้น ไม่ใช่คำตัดสิน
    """
    if provider is None:
        plan = harden_plan(rule_based_plan(report, fit, engine), report, fit)
        _log_session(report, fit, [], plan)
        return plan

    # ผู้ใช้ระบุ engine มา = ไม่ต้องให้ LLM เลือกให้ · เดินทาง rule-based ที่แน่นอนกว่า
    if engine is not None:
        plan = harden_plan(rule_based_plan(report, fit, engine), report, fit)
        _log_session(report, fit, [], plan)
        return plan

    evidence = _evidence(report, fit)
    system = build_system_prompt(DeploymentPlan.model_json_schema())
    attempts: list[dict[str, str]] = []
    feedback = ""

    for _ in range(max_attempts):
        raw = provider.complete_json(system, build_user_prompt(evidence, fit.target_name, feedback))
        try:
            data = json.loads(_strip_fences(raw))
            plan = DeploymentPlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = str(exc)[:2000]
            attempts.append({"raw": redact(raw)[:8000], "error": feedback})
            continue
        plan.generator = f"llm:{provider.name}/{provider.model}"
        # สูตรที่รันผ่านบนฮาร์ดแวร์แล้ว ชนะสิ่งที่ LLM ค้นมาเสมอในส่วนที่ทับกัน —
        # อย่างหนึ่งคือหลักฐาน อีกอย่างคือการอนุมาน · ส่วนที่สูตรไม่ครอบคลุม LLM ยังคุมเหมือนเดิม
        # (ถ้าไม่ทำตรงนี้ ลูกค้าที่ "มี" API key จะได้ผลแย่กว่าคนที่ไม่มี ซึ่งกลับหัวกลับหาง)
        recipe = find_recipe(report.repo_id)
        if recipe is not None:
            plan = apply_recipe(plan, recipe, fit.memory_model.value)
        plan = harden_plan(plan, report, fit)
        attempts.append({"raw": redact(raw)[:8000], "error": ""})
        _log_session(report, fit, attempts, plan)
        return plan

    _log_session(report, fit, attempts, None)
    raise PlanError(
        f"LLM ({provider.name}) ให้ plan ที่ไม่ผ่าน schema ครบ {max_attempts} ครั้ง — "
        "ลองใหม่, เปลี่ยน provider, หรือใช้ --no-llm"
    )


def _log_session(
    report: ModelReport,
    fit: FitReport,
    attempts: list[dict[str, str]],
    plan: DeploymentPlan | None,
) -> None:
    """audit log ต่อการวางแผนหนึ่งครั้ง (NFR auditability) — เขียนไม่ได้ก็ไม่ทำให้งานหลักพัง"""
    try:
        directory = sessions_dir()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        payload = {
            "time": stamp,
            "repo_id": report.repo_id,
            "target": fit.target_name,
            "attempts": attempts,
            "plan": plan.model_dump(mode="json") if plan else None,
            "outcome": "ok" if plan else "failed",
        }
        path = directory / f"plan-{stamp}-{report.repo_id.replace('/', '--')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        pass
