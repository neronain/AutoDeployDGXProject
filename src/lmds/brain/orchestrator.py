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

    # llama.cpp: --ctx-size คือ pool ที่แบ่งให้ทุก slot ส่วน fit คิดต่อ 1 sequence ที่ concurrency
    # ที่ขอ → เพดานของแผนคือ recommended × concurrency ไม่ใช่ recommended เฉย ๆ (ไม่งั้น
    # --concurrency 4 ถูกบีบกลับเหลือ context เดียวแล้วแบ่งสี่) · vLLM/SGLang แชร์ KV แบบ dynamic
    ceiling = fit.recommended_context or 0
    if ceiling and plan.runtime.engine is Engine.LLAMACPP:
        ceiling *= max(1, int(getattr(fit, "concurrency", 1) or 1))
    if ceiling and plan.serving.context > ceiling:
        plan.warnings.append(
            f"ลด context จาก {plan.serving.context:,} เหลือ {ceiling:,} ตาม fit analysis"
        )
        plan.serving.context = ceiling

    # ต้องมา *หลัง* clamp — max_output_tokens ถูกจัดให้พอดี slot จาก context ที่ใช้จริง
    # เดิมจัดก่อนแล้วค่อยลด context → output ที่แผนสัญญาอาจโตกว่า slot (รีวิว 2026-09-04)
    _fit_output_into_slots(plan)

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
    _harden_parsers(plan)

    plan.artifact_type = report.artifact_type
    plan.selected_gguf = plan.selected_gguf or report.selected_gguf
    return plan


# ชื่อ parser ที่ engine แต่ละตัวรู้จักจริง — อ่านมาจากรายการที่ตัวมันเองพิมพ์ออกมาตอน
# ใส่ชื่อผิด · vLLM กับ SGLang **ใช้คนละชุด** และชื่อคล้ายกันจนสับสนได้ง่าย
_VLLM_TOOL_PARSERS = {
    "apertus", "cohere_command3", "cohere_command4", "deepseek_v3", "deepseek_v31",
    "deepseek_v32", "deepseek_v4", "dots", "ernie45", "functiongemma", "gemma4",
    "gigachat3", "glm45", "glm47", "granite", "granite-20b-fc", "granite4", "hermes",
    "hunyuan_a13b", "hy_v3", "inkling", "internlm", "jamba", "kimi_k2", "kimi_k3",
    "lfm2", "ling3", "llama3_json", "llama4_json", "llama4_pythonic", "longcat", "mimo",
    "minicpm5", "minimax_m2", "minimax_m3", "mistral", "muse_glimmer", "olmo3", "openai",
    "phi4_mini_json", "poolside_v1", "pythonic", "qwen3_coder", "qwen3_xml", "seed_oss",
    "step3", "step3p5", "xlam",
}
_VLLM_REASONING_PARSERS = {
    "cohere_command3", "cohere_command4", "deepseek_r1", "deepseek_v3", "deepseek_v4",
    "ernie45", "gemma4", "glm45", "glm47", "granite", "holo2", "hunyuan_a13b", "hy_v3",
    "inkling", "kimi_k2", "kimi_k3", "ling3", "mimo", "minimax_m2",
    "minimax_m2_append_think", "minimax_m3", "mistral", "muse_glimmer", "nemotron_v3",
    "olmo3", "openai_gptoss", "poolside_v1", "qwen3", "seed_oss", "step3", "step3p5",
}
_SGLANG_TOOL_PARSERS = {
    "deepseekv3", "deepseekv31", "deepseekv32", "deepseekv4", "gemma4", "gigachat3",
    "glm", "glm45", "glm47", "gpt-oss", "hermes", "hunyuan", "interns1", "kimi_k2",
    "lfm2", "llama3", "mimo", "minimax-m2", "minimax-m3", "mistral", "poolside_v1",
    "pythonic", "qwen", "qwen25", "qwen3_coder", "step3", "step3p5", "trinity",
}
_SGLANG_REASONING_PARSERS = {
    "deepseek-r1", "deepseek-v3", "deepseek-v4", "gemma4", "glm45", "gpt-oss", "hunyuan",
    "interns1", "kimi", "kimi_k2", "mimo", "minimax", "minimax-append-think", "minimax-m3",
    "mistral", "nemotron_3", "poolside_v1", "qwen3", "qwen3-thinking", "step3", "step3p5",
}


def _harden_parsers(plan: DeploymentPlan) -> None:
    """ชื่อ parser ที่ engine ไม่รู้จัก = เซิร์ฟเวอร์ไม่ขึ้นเลย — ต้องจับตั้งแต่ตอนวางแผน

    เคสจริง 2026-09-01 บน spark-worker: LLM ที่วางแผน (qwen3-coder) เสนอ
    `--tool-call-parser qwen25` และ `--reasoning-parser qwen25` สำหรับโมเดล Qwen3.5
    ไม่มีใครตรวจกับรายชื่อจริง · container ตายทันทีที่ start หลังโหลดน้ำหนักครบแล้ว

        KeyError: 'invalid tool call parser: qwen25'
        KeyError: "Reasoning parser 'qwen25' not found"

    ที่ทำให้พลาดง่ายคือ **qwen25 เป็นชื่อจริงของ SGLang** แค่ไม่ใช่ของ vLLM — สองเครื่องยนต์
    ใช้คนละชุดและตั้งชื่อคล้ายกันมาก (qwen3_coder มีทั้งคู่ · minimax_m3 vs minimax-m3)

    ตัดทิ้งดีกว่าปล่อยผ่าน: เสิร์ฟได้โดยไม่มี tool calling ยังใช้งานได้ ส่วนชื่อผิด
    = start ไม่ขึ้นสักครั้ง · ตั้งเองภายหลังได้ด้วย TOOL_CALL_PARSER/REASONING_PARSER
    """
    if plan.runtime.engine is Engine.LLAMACPP:
        return  # llama.cpp ไม่มี parser แบบนี้ — ฝั่งนั้นใช้ --jinja ของ template เอง
    sglang = plan.runtime.engine is Engine.SGLANG
    engine_name = "SGLang" if sglang else "vLLM"
    other_name = "vLLM" if sglang else "SGLang"

    for kind, cfg, mine, theirs, flag in (
        ("tool-call", plan.tool_calling,
         _SGLANG_TOOL_PARSERS if sglang else _VLLM_TOOL_PARSERS,
         _VLLM_TOOL_PARSERS if sglang else _SGLANG_TOOL_PARSERS, "TOOL_CALL_PARSER"),
        ("reasoning", plan.reasoning,
         _SGLANG_REASONING_PARSERS if sglang else _VLLM_REASONING_PARSERS,
         _VLLM_REASONING_PARSERS if sglang else _SGLANG_REASONING_PARSERS, "REASONING_PARSER"),
    ):
        name = (cfg.parser or "").strip()
        if not name or name in mine:
            continue
        hint = (f" — {name!r} เป็นชื่อของ {other_name} ไม่ใช่ {engine_name}"
                if name in theirs else f" — {engine_name} ไม่รู้จักชื่อนี้")
        plan.warnings.append(
            f"ตัด {kind} parser {name!r} ออก{hint} · ถ้ารู้ชื่อที่ถูก สั่งตอน start ได้: "
            f"{flag}=<ชื่อ> ./controller start")
        cfg.parser = None
        cfg.enabled = False

    # flag ที่ LLM ใส่มาเองซ้ำกับที่ plan ตั้งไว้ = ส่งไปสองชุด vLLM เตือน duplicate keys
    # และถ้าชื่อผิดก็ตายทั้งที่ตัด parser ข้างบนไปแล้ว
    # extra_flags เก็บเป็นสตริงเดียวรวมค่า ("--tool-call-parser qwen25") บ้าง
    # และแยกเป็นสองอิลิเมนต์บ้าง — รองรับทั้งสองรูป
    owned = ("--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice")
    dropped: list[str] = []
    kept: list[str] = []
    skip_next = False
    for item in plan.serving.extra_flags:
        if skip_next:
            skip_next = False
            dropped.append(item)
            continue
        head = item.replace("=", " ").split()[0] if item.strip() else ""
        if head in owned:
            dropped.append(item)
            # "--tool-call-parser qwen25" มีค่าอยู่ในสตริงเดียวกันแล้ว ไม่ต้องข้ามตัวถัดไป
            skip_next = head != "--enable-auto-tool-choice" and len(item.split()) == 1 and "=" not in item
            continue
        kept.append(item)
    if dropped:
        plan.serving.extra_flags = kept
        plan.warnings.append(
            "ตัด flag ที่ซ้ำกับค่าใน plan ออก: " + " ".join(dropped)
            + " — controller ส่งให้เองอยู่แล้ว การใส่ซ้ำทำให้ vLLM เตือน duplicate keys")


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
        plan.speculative.embedded = False
        return

    # บาง repo ไม่แถมไฟล์ draft แยกแต่ "คง" MTP head ไว้ในไฟล์เป้าหมายเอง (nextn_predict_layers)
    # เคสจริง: SC117/Qwen3.6-35B-A3B-...-Native-MTP-Preserved-APEX — llama.cpp เปิดด้วย
    # --spec-type draft-mtp เฉย ๆ ไม่มี --spec-draft-model · ดูแต่ไฟล์อย่างเดียวจะพลาดทั้งตระกูล
    plan.speculative.embedded = bool(report.mtp_embedded)

    available = [v for v in report.gguf_variants if v.is_mtp]
    if not available:
        if plan.speculative.draft_files:
            plan.warnings.append("ตัด draft_files ออก — ไม่พบไฟล์ MTP ใน repo จริง")
            plan.speculative.draft_files = []
        if plan.speculative.embedded:
            plan.warnings.append(
                "GGUF มี MTP head ฝังในตัว (nextn) — เปิด speculative decoding ให้อัตโนมัติ "
                "โดยไม่ต้องมีไฟล์ draft แยก"
            )
        return

    # หัวที่ฝังมาในไฟล์เป้าหมายแล้วไม่ต้องมี draft แยก — ส่งทั้งสองอย่างคือชี้ผิดไฟล์
    if plan.speculative.embedded:
        if plan.speculative.draft_files:
            plan.warnings.append(
                "ตัด draft_files ออก — ไฟล์ที่เลือกฝัง MTP head มาแล้ว ส่ง --spec-type อย่างเดียวพอ"
            )
            plan.speculative.draft_files = []
        plan.warnings.append(
            "GGUF มี MTP head ฝังในตัว (nextn) — เปิด speculative decoding ให้อัตโนมัติ "
            "โดยไม่ต้องมีไฟล์ draft แยก"
        )
        return

    # ปฏิเสธเฉพาะตัวที่อ่าน header แล้วรู้ว่าเป็นหัวล้วน — อ่านไม่ได้ (None) ให้ผ่านไปตามเดิม
    # ดีกว่าปิดฟีเจอร์เงียบ ๆ เพราะเน็ตสะดุดตอน inspect
    usable = [v for v in available if v.is_standalone_draft is not False]
    if not usable:
        plan.speculative.draft_files = []
        plan.warnings.append(
            f"repo มี {available[0].filename.rsplit(chr(47), 1)[-1]} แต่เป็นหัว MTP ล้วน ๆ "
            f"llama.cpp โหลดเป็น draft model แยกไม่ได้ — {_mtp_sibling_hint(plan, report)}"
        )
        return

    chosen = usable[0].filename
    if plan.speculative.draft_files != [chosen]:
        plan.warnings.append(
            f"repo มีไฟล์ MTP — เปิด speculative decoding ให้อัตโนมัติด้วย "
            f"{chosen.rsplit(chr(47), 1)[-1]} (output เหมือนเดิม ได้มาแต่ความเร็ว)"
        )
    plan.speculative.draft_files = [chosen]  # llama-server รับ draft ได้ไฟล์เดียว


def _mtp_sibling_hint(plan: DeploymentPlan, report: ModelReport) -> str:
    """ชี้ชื่อ variant ที่ฝัง MTP มาแล้วของไฟล์ที่เลือก ถ้ามีอยู่จริงใน repo

    "เลือกตัวที่ลงท้าย -mtp" ไม่พอเมื่อ repo มีเป็นร้อย variant — ผู้ใช้ต้องมานั่งเดาว่า
    ตัวไหนคู่กับที่ตัวเองเลือก คำแนะนำที่ระบุชื่อไฟล์ตรง ๆ ทำตามได้ทันที
    """
    chosen = (plan.selected_gguf or "").rsplit("/", 1)[-1]
    if chosen.endswith(".gguf"):
        wanted = f"{chosen[: -len('.gguf')]}-mtp.gguf"
        if any(v.filename.rsplit("/", 1)[-1] == wanted for v in report.gguf_variants):
            return f"เลือก {wanted} แทนเพื่อเปิด speculative decoding"
    return "เลือก variant ที่ลงท้าย -mtp เพื่อเปิด speculative decoding"


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
