"""Fit Analyzer — คำนวณล้วน 100% ไม่ใช้ LLM (FR-3)

หลักการ:
- ใช้ขนาดไฟล์จริงจาก repo ไม่ประมาณจาก params
- KV cache คำนวณจากมิติจริง (layers × kv_heads × head_dim) — ถ้าไม่รู้ ใช้ reserve คงที่และบอกตรง ๆ
- unified (DGX Spark): weights + KV + runtime ≤ total − OS reserve
- discrete (RTX): weights + KV ≤ VRAM × gpu_mem_util − runtime; llama.cpp offload ไป RAM ได้บางส่วน
- target ที่ยังไม่เคยทดสอบจริง → หัก budget เพิ่ม (conservative mode ตาม PRD §12)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from lmds.hardware import MemoryModel
from lmds.inspector.report import ArtifactType, ModelReport

from .targets import TargetSpec

GIB = 1024**3

# ค่าคงที่ของสูตร — ปรับจากผลรันจริงตอน M7 (hardware validation)
VLLM_OVERHEAD_GB_PER_GPU = 2.5  # CUDA context + activations + graphs
LLAMACPP_OVERHEAD_GB = 1.5  # compute buffer + context
UNIFIED_OS_RESERVE_GB = 12.0  # OS + desktop + services บน DGX Spark
GPU_MEMORY_UTILIZATION = 0.85  # ตรงกับ default ของ controller v3.0.0
UNTESTED_BUDGET_FACTOR = 0.95  # หักเพิ่ม 5% เมื่อ target ไม่อยู่ในรายการทดสอบแล้ว
UNKNOWN_KV_RESERVE_FRAC = 0.20  # ไม่รู้มิติ KV → กัน budget 20%
RAM_OFFLOAD_FRAC = 0.70  # llama.cpp offload: ใช้ RAM ได้ไม่เกิน 70%
MIN_PRACTICAL_CONTEXT = 4096
# stacked (TP ข้ามเครื่อง): NCCL buffer + torch.distributed + CUDA graph pool ของ all-reduce ต่อเครื่อง
# — เดิมเป็นแค่โน้ต "ต้องเผื่อ communication buffer" ไม่เคยถูกหักจริง budget รวม 227 GB จึงสูงเกินจริง
# และผู้ใช้ไม่เห็นตัวเลขต่อเครื่องเลย (audit stacked 2026-09-04) · ค่านี้เผื่อจากที่วัดบน 2×Spark:
# ~2–3 GB ต่อ rank สำหรับ NCCL ring/tree + pool ของ TP all-reduce ที่ ctx ยาว
STACKED_COMM_BUFFER_GB_PER_NODE = 3.0
# vLLM: หลังโหลด weight ต้องเหลือให้ profiling run (activation ที่ max_num_batched_tokens) + cache blocks
# ต่ำกว่านี้ไม่ใช่ "context เล็ก" แต่คือ start ไม่ขึ้น ("No available memory for the cache blocks")
VLLM_MIN_KV_GB = 2.0

# บันไดต้องยาวถึงที่โมเดลรุ่นใหม่ไปได้จริง ไม่งั้นบันไดเองกลายเป็นเพดาน
#
# เคสจริง 2026-08-14: Kimi-K3 native 1,048,576 บน 2x Spark มีหน่วยความจำพอถึง
# 735,631 tokens แต่ถูกเสนอที่ 262,144 เพราะนั่นคือขั้นสูงสุดที่มีในลิสต์ —
# เสีย context ไป 2.8 เท่าโดยไม่มีอะไรบอก และตารางในหน้าเว็บก็จบที่ขั้นนั้น
# ทั้งที่ผู้ใช้กรอก 524,288 แล้วระบบตอบว่า "ใส่ได้"
#
# ขั้นที่เกิน native ของโมเดลถูกกรองออกอยู่แล้วทั้งใน _largest_step() (ผ่าน
# min(..., native)) และใน ladder() การเติมขั้นจึงไม่กระทบโมเดล context สั้น
CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
# ไม่มี cap แบบตั้งเลขเอาเองอีกแล้ว — ดู _recommend_context()
CLIENT_OUTPUT_DEFAULT = 8192
TEMPLATE_OVERHEAD_TOKENS = 2048  # chat template + tool schema + system prompt


class Verdict(str, Enum):
    FITS = "fits"
    FITS_REDUCED_CONTEXT = "fits-reduced-context"
    FITS_WITH_OFFLOAD = "fits-with-offload"  # llama.cpp แบ่ง layer ลง RAM — ช้าลงชัดเจน
    NEEDS_SMALLER_QUANT = "needs-smaller-quant"
    NO_FIT = "no-fit"
    UNKNOWN = "unknown"


class VariantFit(BaseModel):
    filename: str
    size_gb: float
    fits: bool


class FitReport(BaseModel):
    target_name: str
    memory_model: MemoryModel
    engine_assumed: str  # vllm | llamacpp
    # จำนวนเครื่อง — คนอ่านรายงานต้องรู้ว่าเป็น target ข้ามเครื่องไหม โดยไม่ต้อง
    # ไปหา TargetSpec กลับมาเทียบเอง (ผู้ช่วย LLM กับหน้าเว็บได้แค่รายงานก้อนนี้)
    node_count: int = 1
    weights_gb: Optional[float] = None
    budget_gb: float = 0.0
    # ภาพรวมหน่วยความจำ — หน้าเว็บเอาไปวาดแถบ "เครื่องมีเท่านี้ · ใช้อยู่แล้ว · weights · KV · เหลือ"
    # ไม่มีสามค่านี้ ผู้ใช้เห็นแค่ budget ก้อนเดียวแล้วเดาไม่ออกว่ามันมาจากอะไร และไม่รู้เลยว่า
    # โมเดลตัวอื่นบนเครื่องเดียวกันถูกนับไปแล้วหรือยัง
    capacity_gb: float = 0.0          # หน่วยความจำ GPU ทั้งหมดของ target
    reserved_gb: float = 0.0          # ที่โมเดลอื่นบนเครื่องเป้าหมายถืออยู่แล้ว (หักออกจาก budget แล้ว)
    reserved_source: str = ""         # อ่านมาจากไหน — ชื่อเครื่อง / "this machine" / "" = preset สมมติ
    kv_budget_gb: Optional[float] = None  # budget - weights = ที่เหลือให้ KV cache
    # ภาพ "ตอนนี้" เมื่อเครื่องเป้าหมายมีโมเดลอื่นรันอยู่ — ค่าข้างบนคิดจากเครื่องเปล่า (ตัดสินว่า
    # สร้าง bundle ได้ไหม) ส่วนชุดนี้บอกว่า *start ตอนนี้* ได้ไหม · deploy = วาง bundle ไว้ก่อน
    # ของอื่นหยุดทีหลังได้ จึงห้ามเอาความแน่นชั่วคราวไปบล็อกการสร้าง — แค่ต้องบอกให้ชัด
    now_verdict: Optional[str] = None          # verdict เมื่อหักของที่รันอยู่ · None = เครื่องว่าง/ไม่รู้
    now_budget_gb: Optional[float] = None      # budget หลังหัก
    now_max_safe_context: Optional[int] = None # context สูงสุดที่ start ได้ตอนนี้
    now_short_gb: Optional[float] = None       # ขาดอีกกี่ GB ถึงจะ start ที่ context ที่แผนเสนอ (0 = พอ)
    running_now: list[str] = Field(default_factory=list)  # โมเดลที่ถือหน่วยความจำอยู่บนเครื่องนั้น
    # ภาพ "ต่อเครื่อง" ของ stacked — tensor parallel แบ่ง weights/KV เท่ากันทุกเครื่อง ผู้ใช้ต้องเห็นว่า
    # แต่ละเครื่องจะถืออะไรเท่าไร ไม่ใช่แค่ยอดรวมของคลัสเตอร์ (single: เท่ากับค่ารวม)
    comm_buffer_gb: float = 0.0             # NCCL/communication buffer ที่หักแล้ว (รวมทุกเครื่อง)
    per_node_capacity_gb: float = 0.0
    per_node_budget_gb: float = 0.0
    per_node_weights_gb: Optional[float] = None
    per_node_kv_budget_gb: Optional[float] = None
    verdict: Verdict = Verdict.UNKNOWN
    kv_bytes_per_token: Optional[int] = None
    kv_estimated: bool = False
    concurrency: int = 1
    max_safe_context: Optional[int] = None
    recommended_context: Optional[int] = None
    client_input_budget: Optional[int] = None
    client_output_default: int = CLIENT_OUTPUT_DEFAULT
    variant_fits: list[VariantFit] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _engine_for(report: ModelReport) -> str:
    return "llamacpp" if report.artifact_type is ArtifactType.GGUF else "vllm"


def _budget_gb(target: TargetSpec, engine: str, reserved_gb: float = 0.0) -> tuple[float, list[str]]:
    notes: list[str] = []
    if target.memory_model is MemoryModel.UNIFIED:
        overhead = VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB
        budget = target.total_gpu_memory_gb - UNIFIED_OS_RESERVE_GB * target.gpu_count - overhead * target.gpu_count
        if target.node_count > 1:
            # หักจริง ไม่ใช่แค่โน้ต — ดู STACKED_COMM_BUFFER_GB_PER_NODE
            budget -= STACKED_COMM_BUFFER_GB_PER_NODE * target.node_count
            notes.append(
                f"stacked {target.node_count} เครื่อง: ตัวเลขเป็นยอดรวมของคลัสเตอร์ — หัก communication buffer "
                f"(NCCL/TP) {STACKED_COMM_BUFFER_GB_PER_NODE:.0f} GB ต่อเครื่องแล้ว · ดูค่าต่อเครื่องที่ per_node"
            )
    else:
        usable = target.total_gpu_memory_gb * GPU_MEMORY_UTILIZATION
        overhead = (VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB) * target.gpu_count
        budget = usable - overhead
        if target.gpu_count > 1:
            notes.append(f"multi-GPU ×{target.gpu_count}: ใช้ tensor parallel — โมเดลต้องแบ่ง layer/head ลงตัว")
    if not target.tested:
        budget *= UNTESTED_BUDGET_FACTOR
        notes.append("target ยังไม่เคยทดสอบจริง — ใช้โหมด conservative (หัก budget เพิ่ม 5%)")

    # หน่วยความจำที่โมเดล *ตัวอื่น* ถืออยู่แล้วบนเครื่องนี้
    #
    # เคสจริง 2026-08-28 บน msi-5: deploy Gemma-4-31B ลงเครื่องที่มี Qwen3.8-27B (Q8_0,
    # ctx 256K) รันอยู่ก่อน · fit คิดจากความจุเต็ม 114.5 GB แล้วตอบ "fits" จึงเลือก Q8_0
    # ตัวใหญ่สุดและ context สูงสุด 262,144 · ผลคือเครื่องขึ้นไป 107/121 GB และทั้งสอง
    # โมเดลคลานอยู่ที่ 5-7 tok/s
    #
    # ตัวเลขนี้มีให้อ่านอยู่แล้ว (`compute_apps()` ที่ inventory ใช้รายงาน foreign
    # workloads) แค่ไม่เคยถูกส่งเข้ามาถึงตรงนี้
    if reserved_gb > 0:
        budget -= reserved_gb
        notes.append(
            f"หักหน่วยความจำที่โมเดลอื่นบนเครื่องนี้ถืออยู่ {reserved_gb:.1f} GB — "
            "หยุดตัวที่ไม่ใช้แล้วค่อย deploy จะได้ quant/context ที่ดีกว่านี้"
        )
    return max(budget, 0.0), notes


def _largest_step(limit: float) -> int | None:
    fitting = [s for s in CONTEXT_STEPS if s <= limit]
    return fitting[-1] if fitting else None


def kv_replication(dims, nodes: int) -> int:
    """KV cache ถูกทำสำเนากี่ชุดทั่วคลัสเตอร์เมื่อ tensor parallel = nodes

    vLLM แบ่ง KV ตาม kv_heads: หัวหาร TP ลงตัว = แต่ละ rank ถือส่วนของตัวเอง (KV รวม = 1 ชุด)
    แต่ถ้า kv_heads < TP (MLA ของ DeepSeek/Kimi = 1 หัว · Qwen3-Coder-Next = 2 หัว บน 4 เครื่อง)
    หัวเดียวแบ่งไม่ได้ → **ทุก rank ถือสำเนาเต็ม** (`num_kv_heads = max(1, kv_heads // tp)`)
    = KV ทั้งคลัสเตอร์โตเป็น TP/kv_heads เท่า และ *ต่อเครื่อง* ต้องมีที่ให้ KV เต็ม context

    เดิมคิดว่า KV ทั้งคลัสเตอร์หาร N ได้เสมอ → DeepSeek-V4 บน 2×Spark ถูกเสนอ 524,288 ทั้งที่แต่ละเครื่อง
    มีที่ให้แค่ 262,144 → vLLM ตาย "No available memory for the cache blocks" ตอน profiling (audit 2026-09-05)
    """
    nodes = max(1, int(nodes or 1))
    heads = int(getattr(dims, "kv_heads", 0) or 0)
    if nodes <= 1 or heads <= 0 or heads >= nodes:
        return 1
    # แต่ละ rank ถือ max(1, heads // nodes) หัว → รวมทั้งคลัสเตอร์ = nodes × หัวต่อ rank ÷ หัวจริง
    return max(1, (nodes * max(1, heads // nodes)) // heads)


def analyze(report: ModelReport, target: TargetSpec, concurrency: int = 1,
            reserved_gb: float = 0.0) -> FitReport:
    """`reserved_gb` = หน่วยความจำที่โมเดลอื่นบนเครื่องเป้าหมายถืออยู่แล้ว

    ผู้เรียกที่รู้ว่าเครื่องเป้าหมายคือเครื่องจริง (ไม่ใช่ preset สมมติ) ควรส่งค่านี้มา —
    ดู `_budget_gb` ว่าทำไมการไม่ส่งถึงทำให้เลือก quant ใหญ่เกินเครื่อง
    """
    engine = _engine_for(report)
    budget, notes = _budget_gb(target, engine, reserved_gb)
    fit = FitReport(
        target_name=target.name,
        memory_model=target.memory_model,
        engine_assumed=engine,
        node_count=target.node_count,
        budget_gb=round(budget, 1),
        capacity_gb=round(target.total_gpu_memory_gb, 1),
        reserved_gb=round(reserved_gb, 1),
        concurrency=concurrency,
        notes=notes,
    )
    nodes = max(1, target.node_count)
    fit.comm_buffer_gb = round(STACKED_COMM_BUFFER_GB_PER_NODE * nodes, 1) if nodes > 1 else 0.0
    fit.per_node_capacity_gb = round(target.total_gpu_memory_gb / nodes, 1)
    fit.per_node_budget_gb = round(budget / nodes, 1)

    weights_bytes = report.weight_bytes
    if weights_bytes is None:
        # ไม่รู้ขนาด weight (เช่น GGUF หลาย variant ที่ยังไม่เลือก) — ประเมินรายไฟล์แทน
        _fill_variant_fits(report, fit, budget)
        if fit.variant_fits:
            fitting = [v for v in fit.variant_fits if v.fits]
            fit.verdict = Verdict.NEEDS_SMALLER_QUANT if not fitting else Verdict.UNKNOWN
            fit.notes.append(
                f"ยังไม่เลือกไฟล์ GGUF — variant ที่ขนาดผ่าน budget: {len(fitting)}/{len(fit.variant_fits)}"
            )
        else:
            fit.notes.append("ไม่ทราบขนาด weight — คำนวณไม่ได้")
        return fit

    weights_gb = weights_bytes / GIB
    fit.weights_gb = round(weights_gb, 1)
    kv_budget_gb = budget - weights_gb
    fit.kv_budget_gb = round(kv_budget_gb, 1)
    # tensor parallel แบ่ง weights และ KV เท่ากันทุกเครื่อง
    fit.per_node_weights_gb = round(weights_gb / nodes, 1)
    fit.per_node_kv_budget_gb = round(max(kv_budget_gb, 0.0) / nodes, 1)

    if kv_budget_gb <= 0:
        return _handle_no_headroom(report, target, fit, weights_gb, budget)
    # stacked: profiling run + cache blocks ต้องมีที่ *ทุกเครื่อง* — ยอดรวม 3 GB บน 2 เครื่อง = เครื่องละ 1.5
    # ซึ่งไม่พอ ทั้งที่ยอดรวมผ่านเกณฑ์ (audit 2026-09-05)
    per_node_kv_gb = kv_budget_gb / nodes
    if engine == "vllm" and per_node_kv_gb < VLLM_MIN_KV_GB:
        # weight ชิด budget จนเหลือ KV ไม่ถึงที่ profiling run ของ vLLM ต้องใช้ — vLLM ตายตอน start ด้วย
        # "No available memory for the cache blocks" ไม่ใช่ "ได้ context 4096" · เคสจริง audit 2026-09-04:
        # Qwen3-235B FP8 (220.2 GiB) บน 2×Spark เหลือ 0.8 GB แล้วรายงานว่า fits-reduced-context
        where = f" (เครื่องละ {per_node_kv_gb:.1f} GB บน {nodes} เครื่อง — ต้องพอต่อเครื่อง)" if nodes > 1 else ""
        return _handle_no_headroom(
            report, target, fit, weights_gb, budget,
            reason=(f"weight {weights_gb:.1f} GB เหลือที่ให้ KV แค่ {kv_budget_gb:.1f} GB{where} "
                    f"(vLLM ต้องการอย่างน้อย ~{VLLM_MIN_KV_GB:.0f} GB สำหรับ profiling run + cache blocks)"),
        )

    dims = report.kv_dims
    if dims is None:
        # ไม่รู้มิติ KV — ใช้ reserve คงที่และแนะนำ context อนุรักษ์นิยม
        fit.kv_estimated = True
        if weights_gb <= budget * (1 - UNKNOWN_KV_RESERVE_FRAC):
            fit.verdict = Verdict.FITS
            fit.recommended_context = min(report.context_length or 16384, 16384)
            fit.notes.append("ไม่ทราบมิติ KV cache — กัน budget 20% และแนะนำ context อนุรักษ์นิยม")
        else:
            fit.verdict = Verdict.FITS_REDUCED_CONTEXT
            fit.recommended_context = MIN_PRACTICAL_CONTEXT
            fit.notes.append("weight ชิด budget และไม่ทราบมิติ KV — เริ่มที่ context ต่ำแล้วทดสอบเพิ่มทีละขั้น")
        _client_budget(fit)
        return fit

    # ค่าต่อ token คิดทั้งคลัสเตอร์: kv_heads < TP → ทุก rank ถือสำเนาเต็ม (ดู kv_replication)
    # → หน้าเว็บ/profile ที่หารด้วยจำนวนเครื่องจะได้ค่าต่อเครื่องที่ถูก (= สำเนาเต็มต่อเครื่อง)
    replication = kv_replication(dims, nodes)
    per_token = dims.bytes_per_token_fp16 * replication
    fit.kv_bytes_per_token = per_token
    if replication > 1:
        fit.notes.append(
            f"KV cache ถูกทำสำเนา {replication} ชุด: โมเดลมี kv_heads={dims.kv_heads} แบ่งให้ {nodes} เครื่อง "
            f"(TP={nodes}) ไม่ลงตัว vLLM จึงให้ทุกเครื่องถือ KV เต็ม context — context สูงสุดคิดจากงบ "
            f"ต่อเครื่อง ({fit.per_node_kv_budget_gb} GB) ไม่ใช่ยอดรวม"
        )
    max_context_raw = (kv_budget_gb * GIB) / (per_token * concurrency)
    native = report.context_length

    limit = min(max_context_raw, native) if native else max_context_raw
    safe = _largest_step(limit)
    if safe is None and native and native < MIN_PRACTICAL_CONTEXT and max_context_raw >= native:
        # โมเดลที่ native สั้นกว่าขั้นล่างสุดของบันได (embedding เล็ก ๆ อย่าง MiniLM 512) — หน่วยความจำพอ
        # เต็ม native อยู่แล้ว ไม่ใช่ "ไม่พอ" · เดิมหาขั้น ≤ 512 ไม่เจอแล้วตอบ needs-smaller-quant
        safe = native
    if safe is None:
        return _handle_no_headroom(
            report, target, fit, weights_gb, budget,
            reason=f"KV เหลือพอแค่ ~{int(max_context_raw):,} tokens (ต่ำกว่า {MIN_PRACTICAL_CONTEXT:,})",
        )

    fit.max_safe_context = safe
    # แนะนำเท่าที่คำนวณได้จริง ไม่ตัดด้วยเลขที่ตั้งเอาเอง
    #
    # เดิมมี DEFAULT_CONTEXT_CAP = 65,536 คร่อมอยู่ตรงนี้ ผลคือทุก deploy ถูกตัด
    # ลงมาที่ 65,536 ไม่ว่าเครื่องจะไหวแค่ไหน แล้วโยนภาระให้ผู้ใช้ไปหา --context เอง
    # ซึ่งไม่มีใครทำเพราะไม่มีใครรู้ว่าเสียอะไรไป โค้ดเดิมถึงกับต้องเขียน note
    # เตือนตัวเองว่า "เสนอ 65,536 แต่รันได้ 262,144" — นั่นคือสัญญาณว่าค่า default ผิด
    # ไม่ใช่ว่า note ยังไม่ดีพอ
    #
    # safe ไม่ใช่การเดา: มันคือ _largest_step(min(หน่วยความจำที่เหลือ, native context))
    # หารด้วย concurrency มาแล้ว จึงเป็นค่าที่ทั้งเครื่องและตัวโมเดลรับไหวโดยนิยาม
    # (เคสจริง 2026-08-13: qwen3-coder-next-gguf บน spark-worker ตั้งมือที่ 131,072
    # แล้วใช้งานได้ ทั้งที่ analyser เคยเสนอ 65,536)
    fit.recommended_context = safe
    fit.notes.append(
        f"context {safe:,} — ค่าสูงสุดที่หน่วยความจำและตัวโมเดลรับไหว "
        f"(ลดได้ด้วย --context ถ้าต้องการเผื่อ concurrency มากกว่านี้)"
    )
    if native and max_context_raw < native:
        fit.verdict = Verdict.FITS_REDUCED_CONTEXT
        fit.notes.append(f"native context {native:,} แต่หน่วยความจำพอที่ ~{int(max_context_raw):,}")
    else:
        fit.verdict = Verdict.FITS
    _client_budget(fit)
    return fit


def _client_budget(fit: FitReport) -> None:
    if fit.recommended_context is None:
        return
    output = CLIENT_OUTPUT_DEFAULT
    input_budget = fit.recommended_context - output - TEMPLATE_OVERHEAD_TOKENS
    if input_budget <= 0:
        output = max(1024, fit.recommended_context // 4)
        input_budget = fit.recommended_context - output - TEMPLATE_OVERHEAD_TOKENS
        fit.notes.append(f"context เล็ก — ลด client output default เหลือ {output:,}")
    fit.client_output_default = output
    fit.client_input_budget = max(input_budget, 0)


def _handle_no_headroom(
    report: ModelReport,
    target: TargetSpec,
    fit: FitReport,
    weights_gb: float,
    budget: float,
    reason: str | None = None,
) -> FitReport:
    if reason:
        fit.notes.append(reason)

    # llama.cpp บนเครื่อง discrete: offload layer ลง RAM ได้ (ช้าลง)
    if (
        fit.engine_assumed == "llamacpp"
        and target.memory_model is MemoryModel.DISCRETE
        and target.system_ram_gb
        and weights_gb <= budget + target.system_ram_gb * RAM_OFFLOAD_FRAC
    ):
        fit.verdict = Verdict.FITS_WITH_OFFLOAD
        fit.recommended_context = MIN_PRACTICAL_CONTEXT
        fit.notes.append(
            f"weight {weights_gb:.1f} GB เกิน VRAM budget {budget:.1f} GB — "
            f"llama.cpp offload บางส่วนลง RAM ได้ แต่ throughput ลดลงมาก"
        )
        _client_budget(fit)
        return fit

    _fill_variant_fits(report, fit, budget)
    fitting_variants = [v for v in fit.variant_fits if v.fits]
    if fitting_variants:
        fit.verdict = Verdict.NEEDS_SMALLER_QUANT
        fit.alternatives.append(
            "เลือก GGUF quant เล็กกว่า เช่น: " + ", ".join(v.filename for v in fitting_variants[:3])
        )
    elif report.artifact_type is ArtifactType.SAFETENSORS:
        fit.verdict = Verdict.NEEDS_SMALLER_QUANT
        fit.alternatives.append("หา checkpoint ที่ quantize แล้ว (GGUF/AWQ/NVFP4) ของโมเดลเดียวกัน")
    else:
        fit.verdict = Verdict.NO_FIT

    if target.memory_model is MemoryModel.DISCRETE:
        fit.alternatives.append("เพิ่ม GPU (tensor parallel) หรือใช้เครื่องที่ VRAM มากกว่า")
    elif target.node_count <= 1:
        fit.alternatives.append("ใช้ DGX Spark แบบ stacked (2+ เครื่อง: target dgx-spark-stacked)")
    elif target.node_count < 4:
        # กำลังวิเคราะห์ stacked อยู่แล้ว — "ใช้ stacked" ไม่ได้บอกอะไร ต้องชี้ preset ที่พอจริง
        # (235B FP8 = 220 GiB ไม่ลง 2×128 แต่ลง 4×128 · audit stacked 2026-09-04)
        fit.alternatives.append(
            f"เพิ่มเครื่อง: target dgx-spark-stacked-4 (4 เครื่อง = 512 GB) — {target.node_count} เครื่อง "
            f"มี budget {budget:.0f} GB แต่ weight {weights_gb:.0f} GB"
        )
    return fit


def _fill_variant_fits(report: ModelReport, fit: FitReport, budget: float) -> None:
    for variant in report.gguf_variants:
        if variant.is_mmproj or variant.size_bytes is None:
            continue
        size_gb = variant.size_bytes / GIB
        fit.variant_fits.append(
            VariantFit(
                filename=variant.filename,
                size_gb=round(size_gb, 1),
                # กันที่ให้ KV/overhead ขั้นต่ำ ~10% ของ budget
                fits=size_gb <= budget * 0.9,
            )
        )
    fit.variant_fits.sort(key=lambda v: v.size_gb, reverse=True)
