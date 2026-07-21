"""GPU allowlist และ hardware profile

`tested=True` = ฮาร์ดแวร์ที่ทีมมีเครื่องจริงให้ทดสอบ (ดู PRD §12 — รุ่นนอก allowlist ใช้โหมด
conservative: เผื่อ headroom สูงกว่าปกติ)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryModel(str, Enum):
    UNIFIED = "unified"  # DGX Spark: CPU/GPU แชร์หน่วยความจำก้อนเดียว
    DISCRETE = "discrete"  # RTX: VRAM แยกจาก system RAM


@dataclass(frozen=True)
class KnownGpu:
    name_pattern: str  # substring จับกับชื่อจาก nvidia-smi (เทียบแบบ lowercase)
    vram_gb: float
    compute_capability: str
    memory_model: MemoryModel
    tested: bool


# เรียงจาก pattern เฉพาะเจาะจงมาก → น้อย เพราะจับแบบ substring ตัวแรกที่เจอ
KNOWN_GPUS: list[KnownGpu] = [
    # เครื่องทดสอบจริงของทีม (2026-07)
    KnownGpu("gb10", 128.0, "12.1", MemoryModel.UNIFIED, tested=True),          # DGX Spark
    KnownGpu("rtx pro 4000 blackwell", 24.0, "12.0", MemoryModel.DISCRETE, tested=True),
    KnownGpu("rtx 4070 ti super", 16.0, "8.9", MemoryModel.DISCRETE, tested=True),
    KnownGpu("rtx 4070 super", 12.0, "8.9", MemoryModel.DISCRETE, tested=True),
    # รุ่นที่รู้จักแต่ยังไม่มีเครื่องทดสอบ — โหมด conservative
    KnownGpu("rtx 5090", 32.0, "12.0", MemoryModel.DISCRETE, tested=False),
    KnownGpu("rtx 4090", 24.0, "8.9", MemoryModel.DISCRETE, tested=False),
    KnownGpu("rtx 6000 ada", 48.0, "8.9", MemoryModel.DISCRETE, tested=False),
    KnownGpu("a100", 80.0, "8.0", MemoryModel.DISCRETE, tested=False),
    KnownGpu("h100", 80.0, "9.0", MemoryModel.DISCRETE, tested=False),
]


def lookup_gpu(smi_name: str) -> KnownGpu | None:
    lowered = smi_name.lower()
    for gpu in KNOWN_GPUS:
        if gpu.name_pattern in lowered:
            return gpu
    return None


class TargetProfile(str, Enum):
    DGX_SPARK_SINGLE = "dgx-spark-single"
    DGX_SPARK_STACKED = "dgx-spark-stacked"
    RTX_SINGLE = "rtx-single"
    RTX_MULTI_GPU = "rtx-multi-gpu"
    UNKNOWN = "unknown"


def classify(gpu_names: list[str]) -> TargetProfile:
    """จำแนก target profile จากรายชื่อ GPU ที่ตรวจพบในเครื่องเดียว"""
    if not gpu_names:
        return TargetProfile.UNKNOWN
    known = [lookup_gpu(n) for n in gpu_names]
    if any(g and g.memory_model is MemoryModel.UNIFIED for g in known):
        return TargetProfile.DGX_SPARK_SINGLE
    if len(gpu_names) > 1:
        return TargetProfile.RTX_MULTI_GPU
    return TargetProfile.RTX_SINGLE
