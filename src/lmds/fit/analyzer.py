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

CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
DEFAULT_CONTEXT_CAP = 65536  # ค่าเริ่มต้นมาตรฐาน v3.0.0
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
    weights_gb: Optional[float] = None
    budget_gb: float = 0.0
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


def _budget_gb(target: TargetSpec, engine: str) -> tuple[float, list[str]]:
    notes: list[str] = []
    if target.memory_model is MemoryModel.UNIFIED:
        overhead = VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB
        budget = target.total_gpu_memory_gb - UNIFIED_OS_RESERVE_GB * target.gpu_count - overhead * target.gpu_count
        if target.gpu_count > 1:
            notes.append("stacked: ตัวเลขเป็นการประมาณรวม — ต้องเผื่อ communication buffer ต่อ node เพิ่ม")
    else:
        usable = target.total_gpu_memory_gb * GPU_MEMORY_UTILIZATION
        overhead = (VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB) * target.gpu_count
        budget = usable - overhead
        if target.gpu_count > 1:
            notes.append(f"multi-GPU ×{target.gpu_count}: ใช้ tensor parallel — โมเดลต้องแบ่ง layer/head ลงตัว")
    if not target.tested:
        budget *= UNTESTED_BUDGET_FACTOR
        notes.append("target ยังไม่เคยทดสอบจริง — ใช้โหมด conservative (หัก budget เพิ่ม 5%)")
    return max(budget, 0.0), notes


def _largest_step(limit: float) -> int | None:
    fitting = [s for s in CONTEXT_STEPS if s <= limit]
    return fitting[-1] if fitting else None


def analyze(report: ModelReport, target: TargetSpec, concurrency: int = 1) -> FitReport:
    engine = _engine_for(report)
    budget, notes = _budget_gb(target, engine)
    fit = FitReport(
        target_name=target.name,
        memory_model=target.memory_model,
        engine_assumed=engine,
        budget_gb=round(budget, 1),
        concurrency=concurrency,
        notes=notes,
    )

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

    if kv_budget_gb <= 0:
        return _handle_no_headroom(report, target, fit, weights_gb, budget)

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

    per_token = dims.bytes_per_token_fp16
    fit.kv_bytes_per_token = per_token
    max_context_raw = (kv_budget_gb * GIB) / (per_token * concurrency)
    native = report.context_length

    limit = min(max_context_raw, native) if native else max_context_raw
    safe = _largest_step(limit)
    if safe is None:
        return _handle_no_headroom(
            report, target, fit, weights_gb, budget,
            reason=f"KV เหลือพอแค่ ~{int(max_context_raw):,} tokens (ต่ำกว่า {MIN_PRACTICAL_CONTEXT:,})",
        )

    fit.max_safe_context = safe
    fit.recommended_context = min(safe, DEFAULT_CONTEXT_CAP)
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
    else:
        fit.alternatives.append("ใช้ DGX Spark แบบ stacked (2+ เครื่อง)")
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
