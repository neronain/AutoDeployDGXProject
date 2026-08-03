"""Deploy wizard ฝั่งเว็บ — วางลิงก์ → วิเคราะห์ → ยืนยัน → generate

เรียก core ชุดเดียวกับ `lmds deploy` ทุกขั้น (resolver → inspector → fit → brain →
generator → validator → packager) ต่างกันแค่ตรงที่ CLI ถามผ่าน prompt ส่วนเว็บส่งค่ากลับมาทีเดียว

ผลลัพธ์ต้องเหมือน CLI เป๊ะ: bundle ที่ไม่ผ่าน quality gates จะไม่มี ZIP ออกมา
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Session:
    """ผลวิเคราะห์ที่รอผู้ใช้ยืนยัน — อยู่ในหน่วยความจำเท่านั้น ไม่เขียนลงดิสก์

    ไม่เก็บ HF token ไว้ในนี้: ใช้ตอน inspect รอบเดียวแล้วทิ้ง (controller อ่านจาก env เอง)
    """
    source: Any
    report: Any
    fit: Any
    plan: Any
    created_by: str = ""
    warnings: list[str] = field(default_factory=list)


_SESSIONS: dict[str, Session] = {}
_MAX_SESSIONS = 20


class DeployError(Exception):
    """ปัญหาที่ผู้ใช้แก้ได้ — ส่งข้อความกลับไปตรง ๆ พร้อม kind ให้ UI ตัดสินใจ"""

    def __init__(self, kind: str, message: str, extra: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.extra = extra or {}


def targets() -> list[dict]:
    from lmds.fit import PRESETS

    return [
        {"name": name, "memory_gb": spec.total_gpu_memory_gb, "tested": spec.tested,
         "memory_model": spec.memory_model.value, "gpus": spec.gpu_count}
        for name, spec in PRESETS.items()
    ]


def _plan_payload(session: Session) -> dict:
    plan, fit, report = session.plan, session.fit, session.report
    return {
        "model_id": plan.model_id,
        "revision": plan.revision,
        "served_model_name": plan.served_model_name,
        "engine": plan.runtime.engine.value,
        "image": plan.runtime.image_ref,
        "topology": plan.topology.value,
        "generator": plan.generator,
        "selected_gguf": plan.selected_gguf,
        "context": plan.serving.context,
        "max_output_tokens": plan.serving.max_output_tokens,
        "gpu_memory_utilization": plan.serving.gpu_memory_utilization,
        "max_num_seqs": plan.serving.max_num_seqs,
        "extra_flags": plan.serving.extra_flags,
        "flags_needing_approval": plan.flags_needing_approval,
        "assets_needing_approval": [
            {"filename": a.filename, "url": a.url} for a in plan.assets_needing_approval
        ],
        "warnings": plan.warnings,
        "features": {
            "tool_calling": plan.tool_calling.enabled,
            "reasoning": plan.reasoning.enabled,
            "modalities": plan.multimodal.modalities,
        },
        "fit": {
            "target": fit.target_name,
            "verdict": fit.verdict.value,
            "budget_gb": fit.budget_gb,
            "weights_gb": fit.weights_gb,
            # เพดานจริงของเครื่อง — มักสูงกว่าค่าที่แผนเสนอ ต้องให้ผู้ใช้เห็นก่อนกดยืนยัน
            "max_safe_context": fit.max_safe_context,
            "notes": fit.notes,
        },
        "gated": report.gated,
    }


def analyze(
    model: str,
    target: Optional[str] = None,
    revision: Optional[str] = None,
    no_llm: bool = False,
    hf_token: str = "",
    selected_gguf: str = "",
) -> dict:
    """วิเคราะห์จนได้ Deployment Plan ที่รอยืนยัน — ยังไม่เขียนไฟล์อะไรเลย"""
    from lmds.brain import PlanError, ProviderError, build_plan, make_provider
    from lmds.config import Settings
    from lmds.fit import PRESETS, Verdict, analyze as analyze_fit
    from lmds.fit.targets import from_hardware_report
    from lmds.inspector import AuthRequired, HfClient, HfError, RepoNotFound, inspect_model
    from lmds.resolver import SourceError, parse_source

    try:
        source = parse_source(model)
    except SourceError as exc:
        raise DeployError("input", str(exc)) from exc
    if revision:
        from dataclasses import replace

        source = replace(source, revision=revision)

    from lmds.secrets import get_secret

    token = hf_token or get_secret("hf") or ""
    try:
        report = inspect_model(source, HfClient(token=token or None))
    except AuthRequired as exc:
        raise DeployError("gated", str(exc)) from exc
    except RepoNotFound as exc:
        raise DeployError("not-found", str(exc)) from exc
    except HfError as exc:
        raise DeployError("hub", str(exc)) from exc

    # repo GGUF หลาย variant ต้องเลือกไฟล์ก่อน ไม่งั้นไปพังตอนท้าย
    weights = [v for v in report.gguf_variants if not v.is_mmproj]
    if weights and not (selected_gguf or report.selected_gguf):
        if len(weights) == 1:
            report.selected_gguf = weights[0].filename
        else:
            raise DeployError(
                "choose-gguf", "repo นี้มี GGUF หลายไฟล์ — เลือกก่อนหนึ่งไฟล์",
                {"variants": [
                    {"filename": v.filename,
                     "size_gb": round(v.size_bytes / 1024**3, 1) if v.size_bytes else None}
                    for v in sorted(weights, key=lambda v: v.size_bytes or 0)
                ]},
            )
    if selected_gguf:
        # inspect ซ้ำด้วยไฟล์ที่เลือก → ได้ GGUF header (architecture/context/kv dims) มาคำนวณ fit จริง
        # ถ้าไม่ทำขั้นนี้ fit จะได้แค่ verdict "unknown" กับ context อนุรักษ์นิยมสุด ๆ (เหมือน CLI ทำ)
        from dataclasses import replace as dc_replace

        chosen = next((v for v in weights if v.filename == selected_gguf), None)
        try:
            report = inspect_model(
                dc_replace(source, filename=selected_gguf), HfClient(token=token or None)
            )
        except HfError:
            report.selected_gguf = selected_gguf
            if chosen is not None:
                report.weight_bytes = chosen.size_bytes

    if target:
        if target not in PRESETS:
            raise DeployError("input", f"ไม่รู้จัก target '{target}'")
        spec = PRESETS[target]
    else:
        from lmds.hardware import probe

        spec = from_hardware_report(probe()) or PRESETS["dgx-spark-single"]

    fit = analyze_fit(report, spec)
    if fit.verdict in (Verdict.NO_FIT, Verdict.NEEDS_SMALLER_QUANT):
        raise DeployError(
            "no-fit", f"โมเดลไม่ fit กับ {fit.target_name} ({fit.verdict.value})",
            {"alternatives": fit.alternatives, "budget_gb": fit.budget_gb, "weights_gb": fit.weights_gb},
        )

    provider = None
    notes: list[str] = []
    if not no_llm:
        settings = Settings.load()
        if settings.provider is not None:
            try:
                provider = make_provider(settings.provider, get_secret(settings.provider.name.value))
            except Exception as exc:  # noqa: BLE001 — provider ล้มไม่ควรทำ deploy ตาย
                notes.append(f"ตั้ง provider ไม่สำเร็จ ({exc}) — ใช้ rule-based แทน")
        else:
            notes.append("ยังไม่ได้ตั้ง LLM provider — ใช้ rule-based mode")

    try:
        plan = build_plan(report, fit, provider)
    except (PlanError, ProviderError) as exc:
        notes.append(f"LLM ใช้ไม่ได้ ({exc}) — สลับเป็น rule-based")
        plan = build_plan(report, fit, None)

    if len(_SESSIONS) >= _MAX_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = Session(source, report, fit, plan, warnings=notes)
    return {"id": session_id, "notes": notes, "plan": _plan_payload(_SESSIONS[session_id])}


def generate(
    session_id: str,
    context: Optional[int] = None,
    approved_flags: Optional[list[str]] = None,
    approved_assets: Optional[list[str]] = None,
    output: str = "./bundles",
) -> dict:
    """render → 9 gates → checksums → ZIP · ไม่ผ่าน gates = ไม่มี ZIP (เหมือน CLI)"""
    from lmds.brain import apply_asset_approvals, apply_flag_approvals
    from lmds.generator import render_bundle
    from lmds.packager import make_zip, write_checksums
    from lmds.validator import all_passed, run_gates

    session = _SESSIONS.get(session_id)
    if session is None:
        raise DeployError("expired", "ผลวิเคราะห์หมดอายุแล้ว — วิเคราะห์ใหม่อีกครั้ง")

    plan, report, fit = session.plan, session.report, session.fit
    if approved_flags:
        apply_flag_approvals(plan, approved_flags)
    if approved_assets:
        apply_asset_approvals(plan, approved_assets)

    if context:
        ceiling = fit.max_safe_context or plan.serving.context
        plan.serving.context = min(int(context), ceiling)

    try:
        bundle = render_bundle(plan, report, fit, Path(output))
    except ValueError as exc:
        raise DeployError("render", str(exc)) from exc

    results = run_gates(bundle.directory, include_checksums=False)
    gates = [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results]
    if not all_passed(results):
        raise DeployError("gates", "bundle ไม่ผ่าน quality gates — ไม่สร้าง ZIP", {"gates": gates})

    checksums = write_checksums(bundle.directory)
    zip_path = make_zip(bundle.directory)
    # ลงทะเบียนทันที ไม่งั้น bundle ที่เพิ่งสร้างจะไม่โผล่ในรายการ แล้วผู้ใช้ไปต่อไม่ถูก
    from lmds.fleet import register_bundle

    register_bundle(bundle.controller)
    _SESSIONS.pop(session_id, None)
    return {
        "slug": bundle.directory.name,
        "directory": str(bundle.directory),
        "context": plan.serving.context,
        "gates": gates,
        "files": [str(p) for p in [*bundle.files, checksums, zip_path]],
        "zip": str(zip_path),
    }
