"""ที่นั่งของ KV cache — "ใส่ได้" กับ "ใช้ได้จริง" ไม่ใช่คำถามเดียวกัน

Fit Analyzer ตอบว่า *ที่ concurrency นี้ ตั้ง context ได้สูงสุดเท่าไร* ซึ่งอ่านแล้ว
เหมือนว่าตั้งตามนั้นแล้วใช้งานได้ตามปกติ · แต่ค่าที่มันเสนอคือค่าที่ **คนเดียว**
กิน KV pool หมดพอดี — ตั้งตามแล้วคนที่สองต้องรอคิว โดยไม่มีอะไรบอก

ไฟล์นี้ตอบคำถามกลับด้าน:
  - ถ้าตั้ง context เท่านี้ จะมีกี่คนใช้พร้อมกันได้
  - ค่าที่กรอกมานั้นควรเป็นเท่าไร และควรเลี่ยงอะไร

คำนวณล้วน ไม่มี LLM (หลักเดียวกับ fit/doctor) · คืน**รหัสกับตัวเลข ไม่ใช่ประโยค**
เพราะปลายทางมีสามที่ที่พูดคนละภาษา: CLI (ไทย) หน้าเว็บ (อังกฤษ) และผู้ช่วย LLM
ที่ต้องตอบเป็นภาษาเดียวกับที่ผู้ใช้พิมพ์มา
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lmds.inspector.report import KvDims

from .analyzer import CONTEXT_STEPS, GIB, FitReport, kv_replication

# KV cache เก็บด้วย dtype อะไรก็ได้ ไม่จำเป็นต้องเท่ากับ dtype ของ weight
# fp8 ลดครึ่งหนึ่งตรง ๆ และเป็นสวิตช์ตอนรัน ไม่ต้อง quantize checkpoint ใหม่
KV_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}

# ต่ำกว่านี้แปลว่าคนเดียวกินหมด — ยังรันได้ แต่ต้องรู้ตัวก่อน ไม่ใช่รู้ตอนคนที่สองบ่น
MIN_USEFUL_CONCURRENCY = 2.0

# เผื่อไว้ให้ CUDA graph, activation ของ chunked prefill, MoE workspace และ NCCL
# buffer ของ TP ข้ามเครื่อง — ของพวกนี้ไม่ได้อยู่ใน budget ของ analyzer
SAFE_MARGIN_FRACTION = 0.15


@dataclass(frozen=True)
class ContextPlan:
    """ตั้ง context เท่านี้แล้วได้อะไร"""

    context: int
    kv_dtype: str
    per_token_bytes: int
    kv_gb: float            # KV ของ "หนึ่ง" sequence ที่ยาวเต็ม context
    kv_budget_gb: float     # ที่ว่างหลังหัก weight ออกจาก budget ของ target
    # ไม่ปัดเศษตรงนี้ — ปัดที่ขอบ (CLI/เว็บ/facts) เท่านั้น · 1.96 ที่ถูกปัดเป็น 2.0
    # ตั้งแต่ในนี้จะผ่านเกณฑ์ MIN_USEFUL_CONCURRENCY ไปโดยไม่มีใครรู้
    concurrency: float      # กี่ sequence ยาวเต็ม context ที่อยู่พร้อมกันได้
    fits: bool


@dataclass(frozen=True)
class Advice:
    """ข้อสังเกตหนึ่งข้อ — รหัส + ตัวเลข ปลายทางเรียบเรียงเป็นภาษาของตัวเอง"""

    kind: str
    level: str  # ok | warn | bad
    facts: dict = field(default_factory=dict)


def bytes_per_token(dims: KvDims, kv_dtype: str = "bf16") -> int:
    """K + V ต่อ token ทุก layer รวมกัน

    `KvDims.bytes_per_token_fp16` คิดที่ 2 ไบต์ตายตัว — ตัวนี้เปิดให้เลือก dtype
    เพราะ `--kv-cache-dtype fp8` เป็นสวิตช์ที่เปลี่ยนคำตอบทั้งหน้า
    """
    width = KV_DTYPE_BYTES.get(kv_dtype)
    if width is None:
        raise ValueError(f"ไม่รู้จัก kv dtype '{kv_dtype}' — มีให้เลือก: {', '.join(KV_DTYPE_BYTES)}")
    # รูปทรงของ KV (GQA หรือ MLA) ตัดสินที่ KvDims ที่เดียว — ที่นี่แค่คูณความกว้าง
    return dims.elements_per_token * width


def kv_budget_gb(fit: FitReport) -> float | None:
    """ที่ว่างสำหรับ KV — None เมื่อยังคำนวณไม่ได้ (ไม่รู้ขนาด weight)"""
    if fit.weights_gb is None:
        return None
    return max(fit.budget_gb - fit.weights_gb, 0.0)


def plan(fit: FitReport, dims: KvDims, context: int, kv_dtype: str = "bf16") -> ContextPlan | None:
    """ตั้ง context เท่านี้ที่ target นี้ จะได้ concurrency เท่าไร"""
    budget = kv_budget_gb(fit)
    if budget is None or context <= 0:
        return None
    # stacked ที่ kv_heads < TP: ทุกเครื่องถือสำเนา KV เต็ม — งบรวมของคลัสเตอร์ต้องจ่ายหลายชุด
    # (ดู analyzer.kv_replication) ไม่งั้นตารางบอกว่า "รับได้ 2 คน" ทั้งที่ได้คนเดียว
    per_token = bytes_per_token(dims, kv_dtype) * kv_replication(dims, fit.node_count)
    kv_gb = context * per_token / GIB
    # ที่ว่าง 0 = หารไม่ได้ · ถือว่าไม่มีใครใช้ได้เลย ไม่ใช่ infinity
    concurrency = (budget / kv_gb) if kv_gb > 0 else 0.0
    return ContextPlan(
        context=context,
        kv_dtype=kv_dtype,
        per_token_bytes=per_token,
        kv_gb=round(kv_gb, 1),
        kv_budget_gb=round(budget, 1),
        concurrency=concurrency,
        fits=kv_gb <= budget,
    )


def ladder(fit: FitReport, dims: KvDims, kv_dtype: str = "bf16",
           native_context: int | None = None) -> list[ContextPlan]:
    """ทั้งบันได — ให้เห็นว่าแลก context กับจำนวนคนกันตรงไหน

    ตารางเดียวตอบคำถามที่ตัวเลขเดี่ยว ๆ ตอบไม่ได้: ลด context ลงหนึ่งขั้น
    ได้คนเพิ่มเท่าตัว · ตัวเลขเดี่ยวทำให้ดูเหมือนมีคำตอบเดียว
    """
    steps = [s for s in CONTEXT_STEPS if native_context is None or s <= native_context]
    plans = [plan(fit, dims, step, kv_dtype) for step in steps]
    return [p for p in plans if p is not None]


def max_context(fit: FitReport, dims: KvDims, concurrency: float = 1.0,
                kv_dtype: str = "bf16") -> int:
    """context สูงสุดที่รองรับ concurrency ตามที่ขอ — ค่าดิบ ยังไม่ปัดเป็นขั้น"""
    budget = kv_budget_gb(fit)
    if budget is None or concurrency <= 0:
        return 0
    per_token = bytes_per_token(dims, kv_dtype) * kv_replication(dims, fit.node_count)
    return int(budget * GIB / (per_token * concurrency))


def advise(fit: FitReport, dims: KvDims | None, context: int,
           kv_dtype: str = "bf16", native_context: int | None = None,
           gpu_count: int = 1) -> list[Advice]:
    """ค่าที่กรอกมานี้ควรเป็นเท่าไร และควรเลี่ยงอะไร

    ทุกข้อมาจากเลขที่คำนวณได้ ไม่มีข้อไหนมาจากการตีความ — ผู้ช่วย LLM เอารายการนี้
    ไปเรียบเรียงเป็นประโยคได้ แต่ห้ามคิดเลขเอง
    """
    out: list[Advice] = []
    if dims is None:
        return [Advice("kv-dims-unknown", "warn", {"context": context})]

    current = plan(fit, dims, context, kv_dtype)
    if current is None:
        return [Advice("weights-unknown", "warn", {"context": context})]

    if native_context and context > native_context:
        out.append(Advice("over-native", "bad",
                          {"context": context, "native": native_context}))

    if not current.fits:
        out.append(Advice("over-memory", "bad", {
            "context": context, "kv_gb": current.kv_gb,
            "kv_budget_gb": current.kv_budget_gb,
        }))
    elif current.concurrency < MIN_USEFUL_CONCURRENCY:
        out.append(Advice("single-user", "warn", {
            "context": context, "concurrency": round(current.concurrency, 1),
        }))

    # เหลือน้อยกว่า 15% = ไม่มีที่ให้ CUDA graph/activation/NCCL ซึ่งไม่ได้อยู่ในบัญชี
    if current.fits:
        spare = current.kv_budget_gb - current.kv_gb
        if spare < current.kv_budget_gb * SAFE_MARGIN_FRACTION:
            out.append(Advice("thin-margin", "warn", {
                "spare_gb": round(spare, 1), "kv_budget_gb": current.kv_budget_gb,
            }))

    # fp8 คือสวิตช์ตอนรัน ไม่ต้อง quantize ใหม่ — เสนอเมื่อมันเปลี่ยนคำตอบจริง
    if kv_dtype != "fp8":
        cheaper = plan(fit, dims, context, "fp8")
        if cheaper and (not current.fits or current.concurrency < MIN_USEFUL_CONCURRENCY) \
                and cheaper.fits:
            out.append(Advice("fp8-would-help", "ok", {
                "context": context,
                "concurrency_now": round(current.concurrency, 1),
                "concurrency_fp8": round(cheaper.concurrency, 1),
                "kv_gb_now": current.kv_gb, "kv_gb_fp8": cheaper.kv_gb,
            }))

    # ยังเหลือที่พอจะขยับขึ้นอีกขั้นโดยที่ concurrency ยังไม่ต่ำกว่าเกณฑ์
    if current.fits and current.concurrency >= MIN_USEFUL_CONCURRENCY:
        bigger = [s for s in CONTEXT_STEPS if s > context
                  and (not native_context or s <= native_context)]
        for step in bigger:
            ahead = plan(fit, dims, step, kv_dtype)
            if ahead and ahead.concurrency >= MIN_USEFUL_CONCURRENCY:
                out.append(Advice("room-to-grow", "ok", {
                    "context": context, "suggest": step,
                    "concurrency": round(ahead.concurrency, 1),
                }))
                break

    if context not in CONTEXT_STEPS:
        out.append(Advice("odd-step", "warn", {"context": context}))

    if gpu_count > 1:
        out.append(Advice("stacked-comms-unbudgeted", "warn", {"nodes": gpu_count}))

    return out


# ── คำอธิบายรหัส — ให้ผู้ช่วย LLM รู้ว่าแต่ละรหัสแปลว่าอะไร โดยไม่ต้องคิดเลขเอง ──
# ข้อความในนี้ไม่ได้เอาไปแสดงตรง ๆ ที่ไหน · CLI กับหน้าเว็บมีสำนวนของตัวเอง
ADVICE_LEGEND = {
    "over-native": "context ที่ขอเกิน native context ของตัวโมเดล — ตัวโมเดลรับไม่ได้ ไม่ใช่เรื่องหน่วยความจำ",
    "over-memory": "KV ที่ context นี้ใหญ่กว่าที่ว่างที่เหลือหลังหัก weight",
    "single-user": "ใส่ได้ แต่หนึ่งคำสนทนากิน KV pool เกือบหมด คนที่สองต้องรอคิว",
    "thin-margin": "เหลือที่ว่างน้อยกว่า 15% — ไม่พอให้ CUDA graph, activation ของ chunked prefill, MoE workspace และ NCCL buffer ซึ่งไม่ได้อยู่ในงบนี้",
    "fp8-would-help": "เปลี่ยน KV cache เป็น fp8 (--kv-cache-dtype fp8_e5m2) ลด KV ครึ่งหนึ่ง แล้วค่านี้ใช้ได้",
    "room-to-grow": "ยังขยับ context ขึ้นได้อีกขั้นโดยที่ยังรับได้มากกว่าหนึ่งคน",
    "odd-step": "ไม่ใช่ค่ายกกำลังสองที่ engine ใช้เป็นขั้น — ตั้งได้ แต่บางตัวปัดลงเอง",
    "stacked-comms-unbudgeted": "เป็น target หลายเครื่อง งบนี้ยังไม่รวม NCCL buffer ของ tensor parallel ข้ามเครื่อง ให้เผื่อไว้",
    "kv-dims-unknown": "อ่านมิติ KV จาก config ไม่ได้ จึงคำนวณให้ไม่ได้",
    "weights-unknown": "ยังไม่รู้ขนาด weight (เช่น GGUF ที่ยังไม่เลือกไฟล์)",
}
