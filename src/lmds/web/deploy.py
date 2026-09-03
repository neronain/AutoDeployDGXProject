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


def _engine_reason(plan, report) -> str:
    """อธิบายว่าทำไมแผนนี้ได้ engine ตัวนี้ — ชนิดไฟล์ในรีโปเป็นตัวตัดสินหลัก"""
    from lmds.brain.plan_schema import Engine
    from lmds.inspector.report import ArtifactType

    artifact = getattr(report, "artifact_type", None)
    if artifact is ArtifactType.GGUF:
        return ("repo เป็น GGUF → llama.cpp เสมอ · vLLM กับ SGLang อ่านไฟล์ GGUF ไม่ได้ "
                "เลือกเป็นอย่างอื่นก็จะได้ bundle ที่ start ไม่ขึ้น")
    if artifact is ArtifactType.SAFETENSORS:
        return (f"repo เป็น safetensors → {plan.runtime.engine.value} · "
                "llama.cpp ใช้กับ repo นี้ไม่ได้เพราะมันอ่านได้เฉพาะ .gguf "
                "(ถ้าอยากใช้ llama.cpp ต้องหา repo ที่แปลงเป็น GGUF แล้ว)")
    if artifact is ArtifactType.MIXED:
        chosen = plan.runtime.engine
        other = "llama.cpp" if chosen is not Engine.LLAMACPP else "vLLM/SGLang"
        return (f"repo มีทั้ง safetensors และ GGUF → เลือก {chosen.value} · "
                f"อีกทางคือ {other} ซึ่งใช้ไฟล์คนละชุดในรีโปเดียวกัน")
    return f"engine: {plan.runtime.engine.value}"


def _plan_payload(session: Session) -> dict:
    from lmds.recipes import find_recipe

    plan, fit, report = session.plan, session.fit, session.report
    # สูตรถูกเติมลงแผนแบบเงียบ ๆ มาตลอด ผู้ใช้จึงไม่รู้ว่าค่าที่เห็นมาจากของที่เคยรันผ่านจริง
    # หรือมาจากค่าตั้งต้น — ต่างกันมากตอนตัดสินใจว่าจะเชื่อแผนนี้ไหม
    recipe = find_recipe(plan.model_id)
    return {
        "recipe": None if recipe is None else {
            "label": recipe.label or recipe.match,
            "match": recipe.match,
            "validated_on": recipe.validated_on,
            "source": recipe.source,
            "controller": recipe.controller,
        },
        # อ่านจากไฟล์ของโมเดลตอน inspect — คนกดยืนยันตรงหน้านี้ ควรเห็นตรงนี้
        "capabilities": report.capabilities,
        "model_id": plan.model_id,
        "revision": plan.revision,
        "served_model_name": plan.served_model_name,
        "engine": plan.runtime.engine.value,
        # ทำไมถึงได้ engine ตัวนี้ — ผู้ใช้เห็นแค่ชื่อ engine แล้วเดาเองไม่ออกว่าเลือกเองได้ไหม
        # เคสจริง 2026-08-19: repo safetensors ล้วน ผู้ใช้หา llama.cpp ในช่อง Engine ไม่เจอ
        # แล้วสรุปว่า LMDS พัง ทั้งที่ llama.cpp อ่าน safetensors ไม่ได้ตั้งแต่แรก
        "engine_reason": _engine_reason(plan, report),
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
            # ภาพรวมหน่วยความจำ — ให้หน้าเว็บวาดแถบ capacity / ใช้อยู่แล้ว / weights / KV / เหลือ
            # ตัวเลข "ใช้อยู่แล้ว" คือของที่หายไปจากหน้าเว็บทั้งที่ analyzer หักให้ได้ตั้งแต่ 2026-08-28
            "capacity_gb": fit.capacity_gb,
            "reserved_gb": fit.reserved_gb,
            "reserved_source": fit.reserved_source,
            "kv_budget_gb": fit.kv_budget_gb,
            "kv_bytes_per_token": fit.kv_bytes_per_token,
            "kv_at_context_gb": _kv_at(fit, plan.serving.context),
            # ชั้น "ตอนนี้" — None ทั้งชุดเมื่อเครื่องว่าง/ไม่รู้
            "now_verdict": fit.now_verdict,
            "now_budget_gb": fit.now_budget_gb,
            "now_max_safe_context": fit.now_max_safe_context,
            "now_short_gb": fit.now_short_gb,
            "running_now": fit.running_now,
            "notes": fit.notes,
        },
        "gated": report.gated,
    }


def _occupancy(name: str) -> tuple[float, list[str]] | None:
    """(GB ที่ถืออยู่, โมเดลที่รันอยู่) บนเครื่อง `name` — อ่านจากแคช inventory ของ hub

    ไม่ยิง SSH ตรงนี้: refresher สำรวจทุกเครื่องอยู่แล้วทุก 15 วิ และ analyze ก็รอ Hub
    อยู่หลายวินาทีแล้ว · None = ยังไม่มีข้อมูลของเครื่องนั้น (เพิ่งแอด / ต่อไม่ได้) ซึ่งต่างจาก
    (0, []) = สำรวจแล้วไม่มีใครถือ GPU — ผู้เรียกต้องบอกผู้ใช้ต่างกัน
    """
    from lmds.web import state

    entry = state.STORE.snapshot()["nodes"].get(name)
    if not entry or not entry.get("data"):
        return None
    data = entry["data"]
    gpus = ((data.get("host") or {}).get("gpus")) or []
    if not gpus:
        return None
    held = float(sum((g.get("vram_used_gb") or 0.0) for g in gpus))
    running = [m.get("slug", "") for m in (data.get("models") or []) if m.get("running")]
    running += [f.get("name", "") for f in ((data.get("host") or {}).get("foreign") or [])]
    return held, [r for r in running if r]


def _held_on_node(name: str) -> float | None:
    occ = _occupancy(name)
    return None if occ is None else occ[0]


def _running_on(machine: str, worker: str) -> list[str]:
    names: list[str] = []
    for m in [machine] + ([worker] if worker and worker != machine else []):
        occ = _occupancy(m)
        if occ:
            names += [f"{slug}@{m}" for slug in occ[1]]
    return names


def _reserved_on_target(spec, target: str, machine: str, worker: str) -> tuple[float, str, list[str]]:
    """หน่วยความจำที่ต้องหักออกจาก budget ก่อนวางแผน + ที่มาของตัวเลข + โน้ตถึงผู้ใช้

    เคสจริง 2026-08-28 (msi-5): deploy Gemma-4-31B ทับเครื่องที่ Qwen3.8-27B (Q8_0, ctx 256K)
    รันอยู่ · หน้าเว็บคิดจาก 114.5 GB เต็มแล้วตอบ "fits" เลือก Q8_0 + ctx 262K · เครื่องขึ้นไป
    107/121 GB ทั้งสองโมเดลคลาน 5-7 tok/s · analyzer รับ `reserved_gb` ได้ตั้งแต่วันนั้น
    แต่หน้าเว็บไม่เคยส่ง — CLI เองก็ส่งเฉพาะตอน target คือเครื่องตัวเอง · ฟลีตที่มีคนรัน
    vLLM หนึ่ง llama.cpp สองบนเครื่องเดียวจึงถูกวางแผนทับกันทุกครั้ง

    กติกา:
      เลือกเครื่องในฟลีต     → อ่านจากแคช inventory ของเครื่องนั้น
                              stacked = เครื่องที่แน่นสุด × จำนวนเครื่อง เพราะ tensor parallel
                              แบ่งเท่ากัน เครื่องที่เหลือน้อยสุดจึงเป็นตัวจำกัดทั้งคลัสเตอร์
      ไม่เลือก + ไม่มี target → เครื่องนี้เอง (เหมือน CLI)
      ไม่เลือก + target preset → เครื่องสมมติ ไม่มีอะไรให้หัก
    """
    notes: list[str] = []
    if machine:
        members = [machine] + ([worker] if worker and worker != machine else [])
        held = [_held_on_node(m) for m in members]
        unknown = [m for m, h in zip(members, held) if h is None]
        if unknown:
            notes.append(
                f"ยังไม่มีข้อมูลหน่วยความจำที่ใช้อยู่ของ {', '.join(unknown)} — คิดจากความจุเต็ม · "
                "ถ้าเครื่องนั้นมีโมเดลรันอยู่ budget นี้สูงเกินจริง "
                "(กด refresh ที่การ์ดเครื่องนั้นแล้ววิเคราะห์ใหม่)"
            )
        known = [h for h in held if h is not None]
        if not known:
            return 0.0, "", notes
        worst = max(known)
        if spec.node_count > 1:
            if worst > 0:
                notes.append(
                    f"stacked: หักตามเครื่องที่ใช้อยู่มากสุด {worst:.1f} GB × {spec.node_count} เครื่อง — "
                    "tensor parallel แบ่งเท่ากัน เครื่องที่แน่นสุดจึงเป็นตัวจำกัดทั้งคลัสเตอร์"
                )
            return worst * spec.node_count, " + ".join(members), notes
        return worst, machine, notes
    if not target:
        from lmds.hardware.profiler import memory_held_gb

        return memory_held_gb(), "this machine", notes
    return 0.0, "", notes


def _note_start_now(fit, plan) -> None:
    """ป้าย "deploy ได้ แต่ start ตอนนี้ได้ไหม" — เติมหลังรู้ context ที่แผนเสนอ"""
    if fit.now_verdict is None or fit.now_budget_gb is None:
        return
    who = ", ".join(fit.running_now) if fit.running_now else fit.reserved_source
    need = (fit.weights_gb or 0.0) + (_kv_at(fit, plan.serving.context) or 0.0)
    short = round(need - fit.now_budget_gb, 1)
    fit.now_short_gb = max(short, 0.0)
    if short > 0 and fit.now_max_safe_context:
        fit.notes.append(
            f"deploy ได้ แต่ start ตอนนี้ที่ context {plan.serving.context:,} ไม่ได้ — "
            f"{fit.reserved_source} เหลือ {fit.now_budget_gb} GB ต้องใช้ {need:.1f} GB (ขาด {short} GB) · "
            f"start ได้ทันทีถ้าลด context เหลือ {fit.now_max_safe_context:,} "
            f"หรือหยุด {who} ก่อน"
        )
    elif short > 0:
        fit.notes.append(
            f"deploy ได้ แต่ start ตอนนี้ไม่ได้ — {fit.reserved_source} เหลือ {fit.now_budget_gb} GB "
            f"แค่ weights ก็ {fit.weights_gb} GB แล้ว (ขาด {short} GB) · หยุด {who} ก่อน start"
        )
    else:
        fit.notes.append(
            f"start ตอนนี้ได้เลย — {fit.reserved_source} เหลือ {fit.now_budget_gb} GB "
            f"พอสำหรับ {need:.1f} GB ที่ context {plan.serving.context:,} (ของอื่นที่รันอยู่: {who})"
        )


def _kv_at(fit, context: int) -> Optional[float]:
    """KV cache ที่ context นี้ (GB) สำหรับ 1 sequence · None เมื่อไม่รู้มิติ KV"""
    from lmds.fit.analyzer import GIB

    if not fit.kv_bytes_per_token or not context:
        return None
    return round(fit.kv_bytes_per_token * context / GIB, 1)


def analyze(
    model: str,
    target: Optional[str] = None,
    revision: Optional[str] = None,
    no_llm: bool = False,
    hf_token: str = "",
    selected_gguf: str = "",
    engine: str = "",
    machine: str = "",
    worker: str = "",
) -> dict:
    """วิเคราะห์จนได้ Deployment Plan ที่รอยืนยัน — ยังไม่เขียนไฟล์อะไรเลย

    `machine`/`worker` = เครื่องในฟลีตที่จะส่ง bundle ไป — ใช้หักหน่วยความจำที่เครื่องนั้น
    ใช้อยู่แล้วออกจาก budget (ดู `_reserved_on_target`) · ว่าง = วิเคราะห์สำหรับเครื่องนี้/preset
    """
    from lmds.brain import PlanError, ProviderError, build_plan, make_provider
    from lmds.brain.plan_schema import Engine
    from lmds.config import Settings
    from lmds.fit import PRESETS, Verdict, analyze as analyze_fit
    from lmds.fit.targets import from_hardware_report
    from lmds.inspector import AuthRequired, HfClient, HfError, RepoNotFound, inspect_model
    from lmds.resolver import SourceError, parse_source

    # ตรวจ input ที่ตรวจได้ทันทีให้หมดก่อนแตะเครือข่าย — ชื่อ engine พิมพ์ผิดไม่ควร
    # ต้องรอผลดึง metadata สามสิบวินาทีก่อนถึงจะรู้
    chosen = None
    if engine:
        try:
            chosen = Engine(engine.strip().lower())
        except ValueError:
            raise DeployError(
                "input",
                f"ไม่รู้จัก engine '{engine}' — มีให้เลือก: "
                + ", ".join(e.value for e in Engine),
            ) from None

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
    weights = [v for v in report.gguf_variants if not v.is_mmproj and not v.is_mtp]
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

        # ชื่อต้องไม่ชนกับ `chosen` (Engine) ข้างบน — เดิมทับกัน GgufVariant เลยถูกส่งเป็น
        # engine= ให้ build_plan ซึ่งตีความว่า "ผู้ใช้เลือก engine เอง" แล้วข้าม LLM ไปเงียบ ๆ
        # ทุกครั้งที่ deploy repo GGUF หลายไฟล์ผ่านหน้าเว็บ
        variant = next((v for v in weights if v.filename == selected_gguf), None)
        try:
            report = inspect_model(
                dc_replace(source, filename=selected_gguf), HfClient(token=token or None)
            )
        except HfError:
            report.selected_gguf = selected_gguf
            if variant is not None:
                report.weight_bytes = variant.size_bytes

    if target:
        if target not in PRESETS:
            raise DeployError("input", f"ไม่รู้จัก target '{target}'")
        spec = PRESETS[target]
    else:
        from lmds.hardware import probe

        spec = from_hardware_report(probe()) or PRESETS["dgx-spark-single"]

    # สองชั้น — ชั้นแรกคิดจาก *เครื่องเปล่า*: ตัดสินว่าสร้าง bundle ได้ไหม (ฮาร์ดแวร์ใส่ได้จริงไหม)
    # ชั้นสองคิดจาก *ตอนนี้* (หักของที่รันอยู่): บอกว่า start ได้เลยไหม ขาดเท่าไร
    #
    # ผู้ใช้ 2026-09-04: "จริงต้องทำได้ เพราะลูกค้าอาจจะยังไม่ได้รัน เพียงแต่ต้องการรู้ค่าและ deploy
    # ลงไปก่อน" — deploy คือวาง bundle ไว้ที่เครื่อง ของอื่นหยุดทีหลังได้ · เวอร์ชันแรกของ 0.5.2
    # เอาความแน่นชั่วคราวไปบล็อกการสร้างทั้งก้อน ซึ่งผิดความหมายของ deploy
    reserved_gb, reserved_source, reserve_notes = _reserved_on_target(spec, target or "", machine, worker)
    fit = analyze_fit(report, spec)
    if fit.verdict in (Verdict.NO_FIT, Verdict.NEEDS_SMALLER_QUANT):
        raise DeployError(
            "no-fit", f"โมเดลไม่ fit กับ {fit.target_name} ({fit.verdict.value}) แม้เครื่องว่าง",
            {"alternatives": fit.alternatives, "budget_gb": fit.budget_gb, "weights_gb": fit.weights_gb},
        )
    fit.reserved_gb = round(reserved_gb, 1)
    fit.reserved_source = reserved_source
    fit.notes.extend(reserve_notes)
    if reserved_gb > 0:
        now = analyze_fit(report, spec, reserved_gb=reserved_gb)
        fit.now_verdict = now.verdict.value
        fit.now_budget_gb = now.budget_gb
        fit.now_max_safe_context = now.max_safe_context
        fit.running_now = _running_on(machine, worker)

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
        plan = build_plan(report, fit, provider, engine=chosen)
    except (PlanError, ProviderError) as exc:
        notes.append(f"LLM ใช้ไม่ได้ ({exc}) — สลับเป็น rule-based")
        plan = build_plan(report, fit, None, engine=chosen)

    _note_start_now(fit, plan)

    if len(_SESSIONS) >= _MAX_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = Session(source, report, fit, plan, warnings=notes)
    return {"id": session_id, "notes": notes, "plan": _plan_payload(_SESSIONS[session_id])}


def context_advice(session_id: str, value: int, kv_dtype: str = "bf16") -> dict:
    """ค่าที่กรอกในหน้า wizard นี้ควรไหม — ตอบจาก session ที่ analyze ไว้แล้ว

    ไม่ต้องยิง Hub ซ้ำและไม่ต้องรอ bundle ถูกสร้างก่อน: `report.kv_dims` กับ `fit`
    อยู่ในมือตั้งแต่ตอน analyze แล้ว · โมเดลที่กำลังจะ deploy จึงได้คำแนะนำทันที
    ไม่ต้องรอให้ bundle รุ่นใหม่ไปบันทึกอะไรไว้ก่อน
    """
    from lmds.fit import advise, ladder

    session = _SESSIONS.get(session_id)
    if session is None:
        raise DeployError("expired", "ผลวิเคราะห์หมดอายุแล้ว — วิเคราะห์ใหม่อีกครั้ง")

    report, fit = session.report, session.fit
    if report.kv_dims is None or fit.weights_gb is None:
        # ตอบว่าคำนวณไม่ได้ ดีกว่าเงียบ — หน้าเว็บจะได้บอกผู้ใช้ว่าทำไมไม่มีคำแนะนำ
        return {"available": False, "reason": "kv-dims-unknown", "ladder": [], "advice": []}

    steps = ladder(fit, report.kv_dims, kv_dtype, report.context_length)
    return {
        "available": True,
        "asked": value,
        "kv_dtype": kv_dtype,
        "kv_bytes_per_token": report.kv_dims.elements_per_token
        * (1 if kv_dtype == "fp8" else 2),
        "native_context": report.context_length,
        "ladder": [
            {"context": s.context, "kv_gb": s.kv_gb,
             "concurrency": round(s.concurrency, 1), "fits": s.fits}
            for s in steps
        ],
        "advice": [
            {"kind": a.kind, "level": a.level, "facts": a.facts}
            for a in advise(fit, report.kv_dims, value, kv_dtype,
                            report.context_length, fit.node_count)
        ],
    }


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
