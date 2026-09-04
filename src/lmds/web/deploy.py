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
    # พอร์ตที่แผนเลือกให้ (ว่างของเครื่องปลายทาง) — generate เขียนลง bundle.env
    port: int = 0


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
    # stacked: กี่เครื่อง / TP เท่าไร — ตัวเดียวกับที่ renderer ใช้เขียน controller (จาก preset ของ target)
    from lmds.fit import PRESETS

    spec = PRESETS.get(fit.target_name)
    is_stacked = plan.topology.value == "stacked"
    node_count = (spec.node_count if spec else 2) if is_stacked else 1
    tensor_parallel = (spec.gpu_count if spec else 2) if is_stacked else (
        2 if plan.topology.value == "multi-gpu" else 1)
    nodes = max(1, node_count)
    kv_at = _kv_at(fit, plan.serving.context)
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
        "task": plan.task,
        # ทำไมถึงได้ engine ตัวนี้ — ผู้ใช้เห็นแค่ชื่อ engine แล้วเดาเองไม่ออกว่าเลือกเองได้ไหม
        # เคสจริง 2026-08-19: repo safetensors ล้วน ผู้ใช้หา llama.cpp ในช่อง Engine ไม่เจอ
        # แล้วสรุปว่า LMDS พัง ทั้งที่ llama.cpp อ่าน safetensors ไม่ได้ตั้งแต่แรก
        "engine_reason": _engine_reason(plan, report),
        "image": plan.runtime.image_ref,
        "topology": plan.topology.value,
        # stacked ต้องบอกว่ากี่เครื่อง/TP เท่าไร — หน้าเว็บเคยเห็นแค่คำว่า "stacked"
        "node_count": node_count,
        "tensor_parallel": tensor_parallel,
        "generator": plan.generator,
        "selected_gguf": plan.selected_gguf,
        "context": plan.serving.context,
        "max_output_tokens": plan.serving.max_output_tokens,
        "gpu_memory_utilization": plan.serving.gpu_memory_utilization,
        "max_num_seqs": plan.serving.max_num_seqs,
        # ค่าที่สูตรตั้งให้ (kv-cache layout ของ DeepSeek V4 · env marlin ของ NVFP4) เคยไม่โผล่ในแผนเลย
        # ผู้ใช้จึงสรุปว่า "สูตรไม่ทำงาน" ทั้งที่ค่าอยู่ใน bundle (audit stacked 2026-09-04)
        "kv_cache_dtype": plan.serving.kv_cache_dtype,
        "extra_env": dict(plan.serving.extra_env or {}),
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
            "kv_at_context_gb": kv_at,
            # ต่อเครื่อง — stacked แบ่งเท่ากันทุกเครื่อง (single = ค่าเดียวกับยอดรวม) · เครื่องที่แน่นสุด
            # เป็นตัวจำกัด reserved จึงเป็น "ของเครื่องที่ใช้อยู่มากสุด" ไม่ใช่ค่าเฉลี่ย
            "node_count": nodes,
            "per_node": {
                "capacity_gb": fit.per_node_capacity_gb or round(fit.capacity_gb / nodes, 1),
                "os_reserve_gb": _os_reserve_gb(fit),
                "engine_overhead_gb": _engine_overhead_gb(fit),
                "comm_buffer_gb": round(fit.comm_buffer_gb / nodes, 1),
                "budget_gb": fit.per_node_budget_gb or round(fit.budget_gb / nodes, 1),
                "weights_gb": fit.per_node_weights_gb,
                "kv_budget_gb": fit.per_node_kv_budget_gb,
                "kv_at_context_gb": None if kv_at is None else round(kv_at / nodes, 1),
                "reserved_gb": round(fit.reserved_gb / nodes, 1),
                "now_budget_gb": None if fit.now_budget_gb is None else round(fit.now_budget_gb / nodes, 1),
            },
            # ชั้น "ตอนนี้" — None ทั้งชุดเมื่อเครื่องว่าง/ไม่รู้
            "now_verdict": fit.now_verdict,
            "now_budget_gb": fit.now_budget_gb,
            "now_max_safe_context": fit.now_max_safe_context,
            "now_short_gb": fit.now_short_gb,
            "running_now": fit.running_now,
            "notes": fit.notes,
        },
        "gated": report.gated,
        # พอร์ตว่างตัวแรกของเครื่องปลายทาง — เดิมทุก bundle เกิดมาที่ 8000 แล้วชนกันบนเครื่องเดียว
        "port": session.port,
    }


DEFAULT_PORT = 8000


def _cached_inventory(machine: str) -> dict | None:
    """inventory ที่ refresher เก็บไว้ของเครื่องนั้น (ว่าง = hub เอง) — None เมื่อยังไม่มี"""
    from lmds.web import state

    snap = state.STORE.snapshot()
    entry = snap["host"] if not machine else snap["nodes"].get(machine)
    return (entry or {}).get("data") or None


def used_ports(machine: str = "") -> set[int]:
    """พอร์ตที่ bundle/container บนเครื่องนั้นถืออยู่ — จากแคช inventory ไม่ยิง SSH

    นับทุก bundle ไม่ใช่แค่ที่รันอยู่: bundle ที่หยุดอยู่ก็จะกลับมาที่พอร์ตเดิมตอน start/autostart
    · container นอกระบบ (`external`) อยู่ในรายการ models อยู่แล้วพร้อมพอร์ตฝั่ง host
    """
    data = _cached_inventory(machine) or {}
    ports: set[int] = set()
    for model in data.get("models") or []:
        port = model.get("port")
        if isinstance(port, int) and port > 0:
            ports.add(port)
    return ports


def suggest_port(machine: str = "", start: int = DEFAULT_PORT, worker: str = "") -> int:
    """พอร์ตว่างตัวแรกตั้งแต่ `start` บนเครื่องนั้น — ไม่มีข้อมูล = `start` (ค่าเดิม ไม่เดา)

    เคสจริง 2026-09-04: deploy ตัวที่สองลงเครื่องเดิมได้ 8000 ซ้ำ หน้าภาพรวมขึ้น "port shared" ทันที
    และ autostart หลัง reboot ก็ชนกันจนขึ้นได้ตัวเดียว · แผนรู้อยู่แล้วว่าไปเครื่องไหน จึงเลือกให้ได้

    stacked: API เปิดที่ head แต่ bundle เดียวกันถูก push ไปทั้งสองเครื่อง (worker ถือ controller
    เดียวกันและอาจถูกสลับเป็น head ในคู่ถัดไป) — เลือกพอร์ตที่ว่างบน *ทั้งคู่* จะได้ไม่ชนตอนสลับบทบาท
    """
    taken = used_ports(machine)
    if worker and worker != machine:
        taken |= used_ports(worker)
    port = start
    while port in taken and port < 65535:
        port += 1
    return port


def _target_for_machine(machine: str) -> tuple[str, str]:
    """(ชื่อ preset, โน้ต) จากฮาร์ดแวร์ที่ refresher เห็นของเครื่องนั้น — ว่างเมื่อเดาไม่ได้

    เลือกเครื่องในฟลีตแต่ไม่เลือก preset เคยตกไปใช้ฮาร์ดแวร์ของ *hub* (VM ไม่มี GPU → dgx-spark-single
    128 GB) ทั้งที่ปลายทางคือ RTX 5090 32 GB → แผนเสนอ context ที่การ์ดนั้นรับไม่ไหว
    """
    from lmds.fit.targets import PRESETS

    host = (_cached_inventory(machine) or {}).get("host") or {}
    gpus = host.get("gpus") or []
    if not gpus:
        return "", (f"ยังไม่มีข้อมูลฮาร์ดแวร์ของ {machine} — วิเคราะห์ด้วยฮาร์ดแวร์ของเครื่องนี้แทน "
                    "· เลือก target ให้ตรงเครื่องปลายทาง หรือกด refresh ที่การ์ดเครื่องนั้นแล้ววิเคราะห์ใหม่")
    if host.get("memory_model") == "unified":
        return "dgx-spark-single", ""
    name = (gpus[0].get("name") or "").lower().replace("nvidia ", "").replace("geforce ", "").strip()
    slug = name.replace(" ", "-")
    for preset in PRESETS:
        if preset == slug or slug.endswith(preset):
            return preset, ""
    return "", (f"ไม่รู้จัก GPU '{gpus[0].get('name')}' ของ {machine} เป็น preset ไหน — "
                "วิเคราะห์ด้วยฮาร์ดแวร์ของเครื่องนี้แทน · เลือก target เองให้ตรงเครื่องปลายทาง")


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


def _os_reserve_gb(fit) -> float:
    from lmds.fit.analyzer import UNIFIED_OS_RESERVE_GB

    return UNIFIED_OS_RESERVE_GB if fit.memory_model.value == "unified" else 0.0


def _engine_overhead_gb(fit) -> float:
    from lmds.fit.analyzer import LLAMACPP_OVERHEAD_GB, VLLM_OVERHEAD_GB_PER_GPU

    return VLLM_OVERHEAD_GB_PER_GPU if fit.engine_assumed == "vllm" else LLAMACPP_OVERHEAD_GB


def _check_stacked_pair(target: str, machine: str, worker: str, node_count: int) -> list[str]:
    """เครื่องคู่ stacked ต้องจับคู่กันได้จริง — ตอบ 4xx พร้อมเหตุผลตั้งแต่วิเคราะห์ ไม่ใช่ไปล้มตอน push

    ลูกค้า 2026-09-04: "deploy หลายเครื่องไม่เคยขึ้น" · ไล่ดูแล้วครึ่งหนึ่งคือคู่ที่จับกันไม่ได้ตั้งแต่ต้น
    (ไม่ได้เลือก worker · worker ไม่มี cluster IP · ปิด stack · คนละไซต์) แต่ analyze ตอบ 200 ด้วยแผน
    2 เครื่องที่ไม่รู้ว่าเครื่องที่สองคือใคร แล้วไปล้มที่ cluster.env/NCCL ด้วยข้อความที่อ่านไม่รู้เรื่อง
    · คืนโน้ต (ไม่ใช่ error) สำหรับสิ่งที่ยังเดินต่อได้ เช่น stacked-4 ที่ต้องเติม worker ที่เหลือใน cluster.env
    """
    from lmds.nodes import find

    notes: list[str] = []
    if node_count <= 1 or not machine:
        return notes
    if not worker:
        raise DeployError(
            "input",
            f"target {target} ใช้ {node_count} เครื่อง — เลือก worker (เครื่องที่สอง) ในช่อง Worker ก่อน "
            "· ต้องการเครื่องเดียวให้เลือก target แบบ single (dgx-spark-single)",
        )
    if worker == machine:
        raise DeployError("input", f"worker เป็นเครื่องเดียวกันกับ head ({machine}) — เลือกเครื่องที่สองที่ต่อสาย 200G ถึงกัน")
    head, second = find(machine), find(worker)
    for role, name, node in (("head", machine, head), ("worker", worker, second)):
        if node is None:
            # ไม่อยู่ในทะเบียนแต่ refresher เคยสำรวจได้ = เรียกตรงจากเทส/สคริปต์ที่ป้อนแคชเอง — ปล่อยผ่าน
            # (ตรวจ cluster IP/site ไม่ได้ก็จริง แต่ชั้น route ของหน้าเว็บตรวจกลุ่ม cluster อีกชั้นอยู่แล้ว)
            if _cached_inventory(name):
                continue
            raise DeployError("input", f"{role} '{name}' ไม่มีในทะเบียนเครื่อง — เพิ่มเครื่องก่อน (หน้าเว็บ: Add machine / lmds node add)")
        if not node.cluster_ip:
            raise DeployError(
                "input",
                f"{role} '{name}' ยังไม่มี cluster IP (ที่อยู่บนสาย 200G/ConnectX) — ตั้งที่การ์ดเครื่องนั้น "
                "หรือ lmds node update --cluster-ip · NCCL ต้องวิ่งบนสายเร็ว ไม่ใช่สายบริหารจัดการ",
            )
        if not getattr(node, "stack", True):
            raise DeployError(
                "input",
                f"{role} '{name}' ถูกตั้งไม่ให้จับกลุ่ม stacked (stack: false) — เปิดที่การ์ดเครื่องนั้นก่อน "
                "หรือเลือกเครื่องอื่น",
            )
    if head and second and head.site and second.site and head.site != second.site:
        raise DeployError(
            "input",
            f"{machine} (ไซต์ {head.site}) กับ {worker} (ไซต์ {second.site}) อยู่คนละไซต์ — stacked ข้ามไซต์ทำไม่ได้ "
            "(NCCL วิ่งบนสายในแร็ค ไม่ใช่ WAN) · เลือก worker ในไซต์เดียวกัน",
        )
    if node_count > 2:
        notes.append(
            f"target {target} ใช้ {node_count} เครื่อง แต่หน้าเว็บเลือก worker ได้ตัวเดียว — เครื่องที่เหลือต้องระบุ "
            "ใน cluster.env (WORKER_IPS) บน head ก่อน start"
        )
    return notes


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
    from lmds.inspector.report import ArtifactType
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

    # เลือก target/เครื่องปลายทางให้จบก่อนแตะเครือข่าย — คู่ stacked ที่จับกันไม่ได้ (ไม่มี worker ·
    # คนละไซต์ · ไม่มี cluster IP) รู้ได้ทันทีจากทะเบียน ไม่ต้องรอดึง metadata ก่อน
    target_notes: list[str] = []
    guessed_target = False
    if not target and machine:
        # เครื่องปลายทางคือเครื่องอื่น — ฮาร์ดแวร์ของ hub ไม่เกี่ยว ต้องคิดจากของเครื่องนั้น
        target, why = _target_for_machine(machine)
        guessed_target = True
        if why:
            target_notes.append(why)
    if worker and worker != machine and (not target or (guessed_target and PRESETS.get(target) and PRESETS[target].node_count <= 1)):
        # เลือก worker มา = ตั้งใจ stacked · target ที่ "เดา" จากเครื่องเดียวเป็น single เสมอ จึงต้องไม่ทับเจตนา
        target = "dgx-spark-stacked"
        target_notes = [n for n in target_notes if "วิเคราะห์ด้วยฮาร์ดแวร์ของเครื่องนี้" not in n]
    if target:
        if target not in PRESETS:
            raise DeployError("input", f"ไม่รู้จัก target '{target}'")
        spec = PRESETS[target]
    else:
        from lmds.hardware import probe

        spec = from_hardware_report(probe()) or PRESETS["dgx-spark-single"]
    if worker and worker != machine and spec.node_count <= 1:
        # เลือก worker แล้วแต่ target เป็น single = แผนเครื่องเดียวที่ push ไป head เฉย ๆ ผู้ใช้เข้าใจว่า
        # deploy 2 เครื่องแล้ว — ขัดกันต้องบอก ไม่ใช่เลือกให้เงียบ ๆ
        raise DeployError(
            "input",
            f"เลือก worker ({worker}) แต่ target '{spec.name}' เป็นเครื่องเดียว — เลือก target แบบ stacked "
            "(dgx-spark-stacked = 2 เครื่อง, dgx-spark-stacked-4 = 4 เครื่อง) หรือเอา worker ออก",
        )
    if chosen is Engine.SGLANG and spec.node_count > 1:
        raise DeployError(
            "input",
            "SGLang ยังไม่มี controller แบบ stacked (หลายเครื่อง) — เลือก engine vllm กับ target stacked "
            "หรือใช้ SGLang กับ target แบบ single",
        )
    target_notes += _check_stacked_pair(spec.name, machine, worker, spec.node_count)

    from lmds.secrets import get_secret

    token = hf_token or get_secret("hf") or ""
    try:
        report = inspect_model(source, HfClient(token=token or None))
    except AuthRequired as exc:
        # บอกทางแก้ที่กดได้เลย — "ต้องใช้ token" เฉย ๆ ผู้ใช้ไม่รู้ว่าใส่ที่ไหน (ลูกค้า 2026-09-04: "analyze ล้ม")
        raise DeployError(
            "gated",
            f"{exc} · ใส่ token: หน้าเว็บกรอกช่อง HF token (ติ๊ก Save) แล้ววิเคราะห์ใหม่ · CLI: "
            "lmds config set-key hf · repo ที่ต้องกดยอมรับเงื่อนไข ให้กดบนเว็บ Hugging Face ด้วยบัญชีของ token นั้นก่อน",
        ) from exc
    except RepoNotFound as exc:
        raise DeployError("not-found", str(exc)) from exc
    except HfError as exc:
        raise DeployError("hub", str(exc)) from exc

    if spec.node_count > 1 and (
        report.artifact_type is ArtifactType.GGUF
        or (report.artifact_type is ArtifactType.MIXED and (selected_gguf or report.selected_gguf))
    ):
        # เดิมผ่าน analyze (200) แล้วไปตาย ValueError ตอน generate — llama.cpp ไม่ทำ tensor parallel
        # ข้ามเครื่อง (ไม่มี reference ที่รันผ่าน) · บอกตั้งแต่ตรงนี้พร้อมทางออก
        raise DeployError(
            "input",
            f"repo นี้เป็น GGUF → llama.cpp ซึ่งทำ tensor parallel ข้ามเครื่องไม่ได้ — target {spec.name} "
            "ใช้ไม่ได้ · เลือก target แบบ single (dgx-spark-single) หรือหา repo safetensors ของโมเดลนี้เพื่อใช้ vLLM stacked",
        )


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
    notes: list[str] = list(target_notes)
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
        if provider is None:
            # rule-based เองปฏิเสธ (embedding/SGLang กับ stacked) — ไม่มีอะไรให้ fallback อีก
            # เดิมเรียก rule-based ซ้ำแล้วยกซ้ำ → 500 เปล่า ๆ ทั้งที่เป็น input ของผู้ใช้
            raise DeployError("input", str(exc)) from exc
        notes.append(f"LLM ใช้ไม่ได้ ({exc}) — สลับเป็น rule-based")
        try:
            plan = build_plan(report, fit, None, engine=chosen)
        except PlanError as again:
            raise DeployError("input", str(again)) from again

    _note_start_now(fit, plan)

    # พอร์ตว่างของเครื่องปลายทาง (hub เองเมื่อไม่เลือกเครื่อง) — บอกไว้ในโน้ตให้เห็นว่าไม่ใช่ 8000
    # ตามเคย และทำไม · stacked: API อยู่ที่ head (`machine`) แต่เลือกที่ว่างบน worker ด้วย (ดู suggest_port)
    port = suggest_port(machine, worker=worker if spec.node_count > 1 else "")
    if port != DEFAULT_PORT:
        who = machine or "เครื่องนี้"
        if spec.node_count > 1 and worker:
            who = f"{machine}/{worker}"
        fit.notes.append(f"พอร์ต {port} — {DEFAULT_PORT} ถึง {port - 1} มี bundle/container บน {who} ถืออยู่แล้ว")

    if len(_SESSIONS) >= _MAX_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = Session(source, report, fit, plan, warnings=notes, port=port)
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
    port: Optional[int] = None,
) -> dict:
    """render → 9 gates → checksums → ZIP · ไม่ผ่าน gates = ไม่มี ZIP (เหมือน CLI)

    `port` = ผู้ใช้แก้พอร์ตในหน้า wizard · None = ใช้ที่ analyze เลือกไว้ (พอร์ตว่างของเครื่องปลายทาง)
    """
    from lmds.brain import apply_asset_approvals, apply_flag_approvals
    from lmds.generator import render_bundle
    from lmds.packager import make_zip, write_checksums
    from lmds.validator import all_passed, run_gates

    session = _SESSIONS.get(session_id)
    if session is None:
        raise DeployError("expired", "ผลวิเคราะห์หมดอายุแล้ว — วิเคราะห์ใหม่อีกครั้ง")

    # ตรวจค่าจาก body ก่อนแตะแผน — เดิม int("abc") ระเบิดเป็น 500 เปล่า ๆ ทั้งที่แค่พิมพ์ผิด
    context = _positive_int("context", context, 256, 10_000_000)
    port = _positive_int("port", port, 1, 65535)

    plan, report, fit = session.plan, session.report, session.fit
    if approved_flags:
        apply_flag_approvals(plan, approved_flags)
    if approved_assets:
        apply_asset_approvals(plan, approved_assets)

    if context:
        ceiling = fit.max_safe_context or plan.serving.context
        plan.serving.context = min(int(context), ceiling)
    chosen_port = port or session.port or DEFAULT_PORT

    try:
        bundle = render_bundle(plan, report, fit, Path(output))
    except ValueError as exc:
        raise DeployError("render", str(exc)) from exc

    results = run_gates(bundle.directory, include_checksums=False)
    gates = [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results]
    if not all_passed(results):
        raise DeployError("gates", "bundle ไม่ผ่าน quality gates — ไม่สร้าง ZIP", {"gates": gates})

    checksums = write_checksums(bundle.directory)
    # พอร์ตอยู่ใน bundle.env (ทางเดียวกับ `lmds set --port`) ไม่ใช่ใน template — controller ทุก engine
    # อ่านไฟล์นี้ก่อน default ทั้ง start/autostart/ปุ่ม test-* จึงได้พอร์ตเดียวกันบนเครื่องปลายทาง ·
    # เขียน *หลัง* checksums (ไฟล์นี้ผู้ใช้แก้ได้ ไม่ควรทำให้ verify-files ล้ม) แต่ *ก่อน* zip ให้ push ติดไปด้วย
    from lmds.fleet.bundle_settings import write as write_settings

    if chosen_port != DEFAULT_PORT:
        write_settings(bundle.directory, {"port": chosen_port})
    zip_path = make_zip(bundle.directory)
    # ลงทะเบียนทันที ไม่งั้น bundle ที่เพิ่งสร้างจะไม่โผล่ในรายการ แล้วผู้ใช้ไปต่อไม่ถูก
    from lmds.fleet import register_bundle

    register_bundle(bundle.controller)
    _SESSIONS.pop(session_id, None)
    return {
        "slug": bundle.directory.name,
        "directory": str(bundle.directory),
        "context": plan.serving.context,
        "port": chosen_port,
        "gates": gates,
        "files": [str(p) for p in [*bundle.files, checksums, zip_path]],
        "zip": str(zip_path),
    }


def _positive_int(name: str, value, low: int, high: int) -> Optional[int]:
    """ค่าจาก JSON ของหน้าเว็บ — None/ว่าง = ไม่ระบุ · อย่างอื่นต้องเป็นจำนวนเต็มในช่วง ไม่งั้น 4xx"""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise DeployError("input", f"{name} ต้องเป็นตัวเลข (ได้ {str(value)[:20]!r})") from None
    if not low <= number <= high:
        raise DeployError("input", f"{name} ต้องอยู่ระหว่าง {low:,} ถึง {high:,}")
    return number
