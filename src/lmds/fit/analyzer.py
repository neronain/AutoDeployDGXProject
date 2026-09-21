"""Fit Analyzer — คำนวณล้วน 100% ไม่ใช้ LLM (FR-3)

หลักการ:
- ใช้ขนาดไฟล์จริงจาก repo ไม่ประมาณจาก params
- KV cache คำนวณจากมิติจริง (layers × kv_heads × head_dim) — ถ้าไม่รู้ ใช้ reserve คงที่และบอกตรง ๆ
- unified (DGX Spark): weights + KV + runtime ≤ total − OS reserve
- discrete (RTX): weights + KV ≤ VRAM × gpu_mem_util − runtime; llama.cpp offload ไป RAM ได้บางส่วน
- target ที่ยังไม่เคยทดสอบจริง → หัก budget เพิ่ม (conservative mode ตาม PRD §12)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from lmds.hardware import MemoryModel
from lmds.inspector.report import ArtifactType, ModelReport

from .targets import TargetSpec

GIB = 1024**3

# ค่าคงที่ของสูตร — ปรับจากผลรันจริงตอน M7 (hardware validation)
VLLM_OVERHEAD_GB_PER_GPU = 2.5  # CUDA context + activations + graphs
# llama.cpp: CUDA primary context + compute buffer — ทั้งคู่เกิด **ต่อ device** ไม่ใช่ต่อโมเดล
# ทุกใบที่ llama.cpp แบ่ง layer ลงไปต้องสร้าง CUDA context ของตัวเอง (ไม่กี่ร้อย MB ต่อใบ เป็น
# ของ driver ไม่ใช่ของโมเดล) และจอง compute buffer ของตัวเอง — ค่านี้จึงโตตามจำนวนใบเสมอ
#
# เคสจริง 2026-09-20 เครื่อง BesthaiAi (3× RTX 3060 12 GB · Qwen3.6-35B-A3B MoE Q4_K
# ctx 262,144 · llama.cpp ใน docker · --n-gpu-layers 999):
#   `lmds fit` ทำนาย  weights 20.2 + overhead 1.5 + KV 5.5 = 27.2 GB
#   nvidia-smi วัดจริง 11475 + 10143 + 10995 MiB = 32,613 MiB = 31.8 GiB  → ต่ำไป 4.6 GB
# แยกสาเหตุได้ (ดู tests/test_multi_gpu_overhead.py):
#   3.0 GB = overhead ถูกคิดใบเดียวแทนที่จะคิดสามใบ — `_budget_gb()` ที่นี่คูณ gpu_count อยู่แล้ว
#            แต่ `plan_kv_pin()` ใน fit/sizing.py (ตัวที่พิมพ์ตาราง `lmds fit`) ไม่เคยคูณ
#   0.9 GB = mmproj (vision projector) ที่ไม่เคยถูกนับใน weights เลย — ดู companion_weight_bytes()
#   0.7 GB = เหลือแยกไม่ออกจากการวัดครั้งเดียว (allocator slack / KV ที่ปัดเป็น block / ค่า
#            per-token ใน profile เอง) — ยังไม่มีใครมีสิทธิ์ตั้งชื่อให้มัน
# **เลข 1.5 ไม่ได้ถูกแก้** — หน่วยของมันต่างหากที่ผิด · ชื่อใหม่บอกหน่วยไว้แล้ว
#
# รูปของการแก้ตรงกับหลักฐานอีกทางที่จดไว้แล้ว: docs/DGX-SPARK-VLLM-FIELD-NOTES.md §6 และ
# docs/UPGRADE-2026-09.md §1.5 — NCCL 8 ทางจอง ~24 GiB **ต่อ rank** นอกงบของ vLLM · ตัวเลขนั้น
# เป็นคำกล่าวอ้างของผู้เขียน (ไม่มี log ดิบ) ใช้ยืนยันได้แค่ "โตตามจำนวน device" · และ NCCL reserve
# เป็น **คนละก้อน** กับค่านี้ — ตอนนี้อยู่ที่ `nccl_reserve_gb_per_rank()` ด้านล่าง
LLAMACPP_OVERHEAD_GB_PER_GPU = 1.5
# ชื่อเดิม — web/deploy.py กับ web/memory.py ยัง import ชื่อนี้อยู่ ห้ามลบจนกว่าจะแก้ทั้งสองที่
LLAMACPP_OVERHEAD_GB = LLAMACPP_OVERHEAD_GB_PER_GPU
# OS + desktop + services บน DGX Spark
# **ระดับหลักฐาน: ค่าเดา — ไม่เคยวัด** · ตั้งไว้ตั้งแต่ต้นโครงและไม่มี artifact รองรับสักชิ้น
# docs/GPU-UTIL-RANK-PROTOCOL.md §6 จะส่ง `UNIFIED_OS_RESERVE_GB_measured` (= idle_used ที่วัดจาก
# /proc/meminfo ก่อน serving) มาแทนค่านี้ · จนกว่าจะถึงตอนนั้น 12.0 คือตัวเลขที่ควรสงสัยที่สุดในไฟล์
UNIFIED_OS_RESERVE_GB = 12.0
GPU_MEMORY_UTILIZATION = 0.85  # ตรงกับ default ของ controller v3.0.0
UNTESTED_BUDGET_FACTOR = 0.95  # หักเพิ่ม 5% เมื่อ target ไม่อยู่ในรายการทดสอบแล้ว
UNKNOWN_KV_RESERVE_FRAC = 0.20  # ไม่รู้มิติ KV → กัน budget 20%
RAM_OFFLOAD_FRAC = 0.70  # llama.cpp offload: ใช้ RAM ได้ไม่เกิน 70%
MIN_PRACTICAL_CONTEXT = 4096
# stacked (TP ข้ามเครื่อง): NCCL buffer + torch.distributed + CUDA graph pool ของ all-reduce
# — เดิมเป็นแค่โน้ต "ต้องเผื่อ communication buffer" ไม่เคยถูกหักจริง budget รวม 227 GB จึงสูงเกินจริง
# และผู้ใช้ไม่เห็นตัวเลขต่อเครื่องเลย (audit stacked 2026-09-04)
#
# ค่านี้เผื่อจากที่วัดบน 2×Spark: ~2–3 GB ต่อ rank สำหรับ NCCL ring/tree + pool ของ TP all-reduce
#
# ── ขอบเขตที่อ้างได้ (เพิ่ม 2026-09-21) ────────────────────────────────────────────────────
# **วัดที่ 2 rank เท่านั้น** → `STACKED_COMM_BUFFER_valid_for_node_counts = [1, 2]`
# (docs/GPU-UTIL-RANK-PROTOCOL.md §6) · ที่ 3+ เครื่องเรากำลังอ้างนอกช่วงที่วัด และ `_budget_gb()`
# ต้องพูดออกมาเป็นโน้ต ไม่ใช่เงียบ · ภาคสนามอ้างว่าก้อนนี้โตตามจำนวน rank — ดูพจน์ที่ปิดอยู่ข้างล่าง
STACKED_COMM_BUFFER_GB_PER_NODE = 3.0

# ── พจน์ที่จะสเกลตาม rank — **ยังปิดอยู่ โดยตั้งใจ** ────────────────────────────────────────
# ภาคสนาม (FIELD-NOTES §6) อ้างว่า NCCL 8 ทางจอง ~24 GiB **ต่อ rank** ซึ่งถ้าจริงแปลว่าค่าคงที่
# ต่อเครื่องข้างบนไม่พอ ต้องมีพจน์ที่โตตามจำนวน peer · **แต่เรายังใส่ไม่ได้** ด้วยสามเหตุผล:
#
#   1. หลักฐานเป็น **คำกล่าวอ้างของผู้เขียน** (README ไม่มี log ดิบ) และผู้เขียนคนเดียวกันเขียน
#      อีก README ที่ใช้ 0.80 ที่ TP8 ขัดกันเอง — รีโปนั้นไม่มีโฟลเดอร์ `results/` เลย
#   2. ตัวเลข ~24 GiB/rank อยู่ใกล้ ~27 GiB/rank ของ **COW break** (fused-MoE expert copy ที่
#      UPGRADE-2026-09.md §1.7 R8 วินิจฉัยไว้) มาก · นั่นเป็นคนละกลไกและ **ไม่โตตาม rank เลย**
#      มันโตตาม checkpoint · แยกกันได้ที่ N=1 ซึ่งไม่มี NCCL อยู่ในโปรเซส
#   3. ฟลีตเรา **รัน TP4/TP8 ไม่ได้** — recon ของเราเอง (docs/HCA-DUAL-TEST.md §3, 2026-09-20)
#      พบ world size สูงสุดบน fabric เดียวคือ **2** จึงจะไม่มีใครยืนยันรูปนี้ให้เราในเร็ว ๆ นี้
#
# docs/GPU-UTIL-RANK-PROTOCOL.md §5 จึงสั่งไว้ล่วงหน้า (ก่อนรู้ผล) ว่า **ห้ามเปลี่ยนค่าคงที่ข้างบน
# ให้สเกลตาม rank จนกว่าจะได้ verdict = R** เพราะถ้าผลออกมาเป็น C (checkpoint) หรือ H (host
# headroom) การเปลี่ยนนั้นคือการเข้ารหัสสมมติฐานที่ผิดลงในสินค้า แล้ว **ทุก stacked fit จะผิด
# ในแบบที่ดูน่าเชื่อถือกว่าเดิม** ซึ่งแย่กว่าการไม่มีพจน์นั้นเลย
#
# โครงสร้างจึงอยู่ครบแล้วแต่ **ค่าเริ่มต้นให้ผลเท่าพฤติกรรมเดิมทุกจำนวน rank** · เปิดใช้โดยตั้ง
# ค่านี้เป็นตัวเลข (`dx_rank_gib` จาก `fit-input.json` ที่โปรโตคอลจะส่งมา) เมื่อ verdict = R เท่านั้น
NCCL_COMM_BUFFER_GB_PER_PEER: float | None = None
# ถ้า verdict = C: ของที่ต้องเพิ่มคือ footprint ต่อ **checkpoint** ไม่ใช่ต่อ rank
# (`checkpoint_host_footprint_gib` ใน fit-input.json) — 0.0 = ยังไม่มีผลวัด ไม่เปลี่ยนอะไรวันนี้
CHECKPOINT_HOST_FOOTPRINT_GB = 0.0
# GPU-util ที่คนตั้งจริงเป็นขั้นละ 0.05 (0.85/0.80/0.75) — ปัด **ลง** เข้าขั้นเสมอ เพื่อให้เพดานที่เรา
# แนะนำไม่มีทางสูงกว่าค่าที่มีคนรันผ่านจริง และเพื่อให้คำตอบไม่ขยับตามการปัด GiB/decimal ของ capacity
GPU_UTIL_NOTCH = 0.05


def nccl_reserve_gb_per_rank(ranks: int) -> float:
    """หน่วยความจำนอกงบของ vLLM กี่ GB **ต่อ rank** เมื่อ world size = ``ranks``

    ค่าเริ่มต้น = พฤติกรรมเดิมเป๊ะ: 0 ที่ 1 rank · ``STACKED_COMM_BUFFER_GB_PER_NODE`` คงที่ที่ 2+
    (ค่าที่เราวัดเองบน 2×Spark · อ้างได้ถึงแค่ 2 rank — ดู valid_for_node_counts ในโปรโตคอล §6)

    ``NCCL_COMM_BUFFER_GB_PER_PEER`` ที่ไม่ใช่ None จะเปลี่ยนเป็นรูปที่โตตามจำนวน peer
    — **เปิดได้ต่อเมื่อ docs/GPU-UTIL-RANK-PROTOCOL.md ให้ verdict = R** เท่านั้น
    """
    ranks = max(1, int(ranks))
    if ranks == 1:
        return 0.0
    if NCCL_COMM_BUFFER_GB_PER_PEER is None:
        return STACKED_COMM_BUFFER_GB_PER_NODE
    return NCCL_COMM_BUFFER_GB_PER_PEER * (ranks - 1)


def gpu_util_ceiling(total_gb: float, ranks: int, *, os_reserve_gb: float = UNIFIED_OS_RESERVE_GB) -> float:
    """เพดาน --gpu-memory-utilization ที่ host นี้รับไหว — **เพดานคือ host headroom ไม่ใช่เพดานของ GPU**

    vLLM คิด gpu-util จากหน่วยความจำ *ทั้งเครื่อง* แต่มีของที่จองอยู่ *นอก* งบนั้น จึงต้องเหลือที่ให้:

        gpu_util × total + X(ranks) + OS ≤ total

    รูปเดียวกับที่ docs/GPU-UTIL-RANK-PROTOCOL.md §1 เขียนไว้ (``gpu_util_max = (MemTotal − X − safety)
    / MemTotal``) — ประเด็นทั้งหมดคือ **0.75/0.80 ไม่ใช่ค่าคงที่ของฮาร์ดแวร์ มันคือผลหาร** ใครที่
    รายงาน "0.75 ที่ TP8" กำลังรายงาน X ของเขาโดยไม่รู้ตัว · ฟังก์ชันนี้จึงเป็น "สูตร" ไม่ใช่ "ตาราง"

    **ด้วย X วันนี้ (คงที่ 3 GB ที่ 2+ rank) เพดานนี้จึงยัง *ไม่* ลดตามจำนวน rank** — 0.90 ที่ 1 rank
    และ 0.85 ที่ 2+ · นั่นคือสิ่งที่หลักฐานของเรารองรับจริง ไม่มากกว่านั้น · เมื่อ X ถูกวัดตามโปรโตคอล
    (ไม่ว่าผลจะเป็น R, C หรือ H) เพดานจะขยับเองโดยไม่ต้องแก้ฟังก์ชันนี้
    """
    if not total_gb or total_gb <= 0:
        return GPU_MEMORY_UTILIZATION
    head = float(total_gb) - nccl_reserve_gb_per_rank(ranks) - CHECKPOINT_HOST_FOOTPRINT_GB - os_reserve_gb
    return max(0.30, math.floor(head / float(total_gb) / GPU_UTIL_NOTCH) * GPU_UTIL_NOTCH)
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
    # ส่วนของ weights_gb ที่มาจากไฟล์คู่ (mmproj projector / MTP draft head) — แยกไว้ให้คนอ่าน
    # รายงานเห็นว่ายอด weights ไม่ได้เท่ากับขนาดไฟล์ GGUF ที่เลือก (0 = ไม่มีไฟล์คู่)
    companion_weights_gb: float = 0.0
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
    comm_buffer_gb_per_node: float = 0.0    # ต่อ rank — โตตามจำนวน peer (nccl_reserve_gb_per_rank)
    # เพดาน --gpu-memory-utilization ต่อเครื่อง (unified เท่านั้น) — ยิ่ง rank เยอะยิ่งต้องลด ดู gpu_util_ceiling()
    gpu_util_ceiling: Optional[float] = None
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
        overhead = VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB_PER_GPU
        budget = target.total_gpu_memory_gb - UNIFIED_OS_RESERVE_GB * target.gpu_count - overhead * target.gpu_count
        if target.node_count > 1:
            # หักจริง ไม่ใช่แค่โน้ต — และ **โตตามจำนวน rank** ไม่ใช่ค่าคงที่ต่อเครื่อง
            per_rank = nccl_reserve_gb_per_rank(target.node_count)
            budget -= per_rank * target.node_count
            notes.append(
                f"stacked {target.node_count} เครื่อง: ตัวเลขเป็นยอดรวมของคลัสเตอร์ — หัก communication buffer "
                f"(NCCL/TP) {per_rank:.0f} GB ต่อเครื่องแล้ว · ดูค่าต่อเครื่องที่ per_node"
            )
            if target.node_count > 2:
                # ค่านี้วัดที่ 2 rank · เกินกว่านั้นเป็นการอ้างนอกขอบเขต ต้องบอก ไม่ใช่เงียบ
                notes.append(
                    f"⚠️ communication buffer {per_rank:.0f} GB/เครื่อง วัดไว้ที่ 2 เครื่องเท่านั้น — "
                    f"ที่ {target.node_count} เครื่องยังไม่มีผลวัดรองรับ (ดู docs/GPU-UTIL-RANK-PROTOCOL.md) · "
                    "ถ้าภาคสนามถูกว่าก้อนนี้โตตามจำนวน rank ตัวเลขนี้จะ **ต่ำกว่าจริง**"
                )
            # "กี่เครื่อง" ไม่พอ ต้องบอก "ต่อกันอย่างไร" ด้วย — คนที่เลือก target 4 เครื่อง
            # แต่ไม่มี switch จะเสียบวงแหวนไม่ได้ (QSFP หมดตั้งแต่ 3 เครื่อง) แล้วไปเจอ
            # "unknown topology" ตอน `lmds cluster apply` ซึ่งไม่บอกว่าต้องซื้ออะไร ·
            # ตรงนี้คือจุดแรกสุดที่รู้จำนวนเครื่อง จึงเป็นจุดที่ควรเตือน
            notes.append(target.interconnect.note)
    else:
        usable = target.total_gpu_memory_gb * GPU_MEMORY_UTILIZATION
        per_gpu = VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB_PER_GPU
        overhead = per_gpu * target.gpu_count
        budget = usable - overhead
        if target.gpu_count > 1:
            notes.append(
                f"multi-GPU ×{target.gpu_count}: ใช้ tensor parallel — โมเดลต้องแบ่ง layer/head ลงตัว · "
                f"หัก overhead {per_gpu:.1f} GB ต่อใบ = {overhead:.1f} GB (CUDA context + compute buffer เกิดทุกใบ)"
            )
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


def companion_weight_bytes(report: ModelReport) -> int:
    """ไบต์ของไฟล์คู่ GGUF (mmproj / MTP draft) ที่ controller โหลดขึ้น GPU ด้วย แต่ไม่อยู่ใน weight_bytes

    `report.weight_bytes` ของ GGUF = ขนาดของ variant ที่เลือก **ไฟล์เดียว** (inspector/inspect.py
    `report.weight_bytes = selected.size_bytes`) ส่วน projector กับ draft head เป็นไฟล์แยก ที่
    generator/renderer.py ต่อท้าย MODEL_FILES ให้ controller โหลดและส่งเป็น `--mmproj` /
    `--spec-draft-model` — llama-server วางทั้งคู่ไว้บน GPU เหมือน weight แต่สูตร fit ไม่เคยนับ

    เคสจริง 2026-09-20 BesthaiAi: mmproj f16 0.9 GB หายไปจากยอด weights ทั้งก้อน (ดูคอมเมนต์
    ของ LLAMACPP_OVERHEAD_GB_PER_GPU) — เป็นคนละบั๊กกับเรื่อง overhead ต่อใบ และแก้แยกกัน

    นับ **ไฟล์เดียวต่อประเภท** เพราะ llama-server รับ `--mmproj` ได้ไฟล์เดียวและ
    `--spec-draft-model` ได้ไฟล์เดียว ไม่ใช่ผลรวมของทุก precision ที่ repo แถมมา (repo ที่ใส่
    mmproj ทั้ง BF16/F16/F32 ไว้ด้วยกันมีจริง — รวมทั้งสามจะเกินจริงเท่าตัว)
    เลือกไฟล์เล็กสุดให้ตรงกับที่ `brain._pick_projector()` เลือกจริง (มันเลือก min ตามขนาด) ·
    repo ที่เอา projector ของโมเดลคนละตัวมาปนกันอาจได้ตัวที่เล็กกว่าของจริงอยู่ระดับร้อย MB —
    ยอมรับไว้ก่อน เพราะไฟล์ที่ใช้จริงถูกตัดสินหลัง fit และที่นี่ยังไม่มีข้อมูลนั้น
    """
    total = 0
    for flag in ("is_mmproj", "is_mtp"):
        sizes = [v.size_bytes for v in report.gguf_variants if getattr(v, flag, False) and v.size_bytes]
        if sizes:
            total += min(sizes)
    return total


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
    fit.comm_buffer_gb = round(nccl_reserve_gb_per_rank(nodes) * nodes, 1) if nodes > 1 else 0.0
    fit.comm_buffer_gb_per_node = round(nccl_reserve_gb_per_rank(nodes), 1) if nodes > 1 else 0.0
    fit.per_node_capacity_gb = round(target.total_gpu_memory_gb / nodes, 1)
    fit.per_node_budget_gb = round(budget / nodes, 1)
    if target.memory_model is MemoryModel.UNIFIED:
        # เพดาน gpu-util ของ *เครื่องหนึ่งเครื่อง* — ไม่ใช่ของคลัสเตอร์ (vLLM ตั้งค่านี้ต่อ rank)
        fit.gpu_util_ceiling = round(gpu_util_ceiling(fit.per_node_capacity_gb, nodes), 2)
        if nodes > 1 and fit.gpu_util_ceiling < GPU_MEMORY_UTILIZATION:
            fit.notes.append(
                f"gpu-util ไม่ควรเกิน {fit.gpu_util_ceiling:.2f} ที่ {nodes} เครื่อง — เหลือที่ให้ของที่จอง "
                f"*นอก* งบของ vLLM {fit.comm_buffer_gb_per_node:.0f} GB/เครื่อง + OS {UNIFIED_OS_RESERVE_GB:.0f} GB"
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
    # ไฟล์คู่ (mmproj/MTP) กินหน่วยความจำ GPU เหมือน weight — ต้องอยู่ในยอดเดียวกัน ไม่ใช่โน้ตข้าง ๆ
    companion_gb = companion_weight_bytes(report) / GIB
    if companion_gb:
        weights_gb += companion_gb
        fit.companion_weights_gb = round(companion_gb, 1)
        fit.notes.append(
            f"รวมไฟล์คู่ (mmproj/MTP) {companion_gb:.1f} GB ไว้ในยอด weights แล้ว — "
            "llama-server โหลดขึ้น GPU พร้อมกับ weight ไม่ใช่ไฟล์ที่วางไว้เฉย ๆ"
        )
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
