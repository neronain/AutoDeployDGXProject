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
from lmds.secrets import redact

from .allowlists import is_known_image, split_flags
from .plan_schema import DeploymentPlan, Engine, PlanError
from .prompts import build_system_prompt, build_user_prompt
from .providers import LlmProvider
from .rulebased import rule_based_plan

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


def harden_plan(plan: DeploymentPlan, report: ModelReport, fit: FitReport) -> DeploymentPlan:
    """บังคับข้อเท็จจริงกลับเข้า plan — LLM output เป็น untrusted input (PRD §9.3)"""
    if plan.revision != report.revision_sha:
        plan.warnings.append(f"แก้ revision จาก {plan.revision!r} เป็น SHA ที่ pin จริง")
        plan.revision = report.revision_sha
    if plan.model_id != report.repo_id:
        plan.warnings.append(f"แก้ model_id จาก {plan.model_id!r} เป็น {report.repo_id}")
        plan.model_id = report.repo_id

    expected_engine = Engine.LLAMACPP if report.artifact_type is ArtifactType.GGUF else Engine.VLLM
    if plan.runtime.engine is not expected_engine:
        plan.warnings.append(
            f"แก้ engine จาก {plan.runtime.engine.value} เป็น {expected_engine.value} ตาม artifact จริง"
        )
        plan.runtime.engine = expected_engine

    if not is_known_image(plan.runtime.engine, plan.runtime.image_ref):
        from .rulebased import DEFAULT_IMAGES

        plan.warnings.append(
            f"image ที่แผนเสนอ ({plan.runtime.image_ref}) ไม่อยู่ใน registry ที่ยอมรับ — "
            f"เปลี่ยนเป็น {DEFAULT_IMAGES[plan.runtime.engine]}"
        )
        plan.runtime.image_ref = DEFAULT_IMAGES[plan.runtime.engine]
        plan.runtime.image_pin = None

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

    allowed, needs_approval = split_flags(plan.runtime.engine, plan.serving.extra_flags)
    plan.serving.extra_flags = allowed
    if needs_approval:
        plan.flags_needing_approval = sorted(set(plan.flags_needing_approval + needs_approval))
        plan.warnings.append(
            "flag นอก allowlist ต้องได้รับอนุมัติจากผู้ใช้ก่อนใช้จริง: " + ", ".join(needs_approval)
        )
    plan.artifact_type = report.artifact_type
    plan.selected_gguf = plan.selected_gguf or report.selected_gguf
    return plan


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


def build_plan(
    report: ModelReport,
    fit: FitReport,
    provider: LlmProvider | None,
    max_attempts: int = MAX_ATTEMPTS,
) -> DeploymentPlan:
    """provider=None → rule-based (degraded/--no-llm); มี provider → LLM + validate + retry"""
    if provider is None:
        plan = harden_plan(rule_based_plan(report, fit), report, fit)
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
