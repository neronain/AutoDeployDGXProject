"""Target spec ของเครื่องเป้าหมาย — preset จากเครื่องจริงของทีม + สร้างจากผลตรวจเครื่องปัจจุบัน"""

from __future__ import annotations

from dataclasses import dataclass

from lmds.hardware import HardwareReport, MemoryModel, TargetProfile


@dataclass(frozen=True)
class TargetSpec:
    name: str
    memory_model: MemoryModel
    memory_gb: float  # VRAM ต่อ GPU (discrete) หรือ unified ทั้งก้อน
    gpu_count: int = 1
    system_ram_gb: float | None = None  # ใช้ประเมิน offload ของ llama.cpp
    tested: bool = False  # False → คำนวณแบบ conservative (ลด budget เพิ่ม)

    @property
    def total_gpu_memory_gb(self) -> float:
        return self.memory_gb * self.gpu_count


# preset ตามเครื่องทดสอบจริงของทีม (PRD §13 Decision Log 2026-07-21) + เครื่องอ้างอิงทั่วไป
PRESETS: dict[str, TargetSpec] = {
    "dgx-spark-single": TargetSpec(
        "dgx-spark-single", MemoryModel.UNIFIED, 128.0, 1, system_ram_gb=None, tested=True
    ),
    "dgx-spark-stacked": TargetSpec(
        "dgx-spark-stacked", MemoryModel.UNIFIED, 128.0, 2, system_ram_gb=None, tested=True
    ),
    "rtx-pro-4000": TargetSpec("rtx-pro-4000", MemoryModel.DISCRETE, 24.0, 1, tested=True),
    "rtx-pro-4000-dual": TargetSpec("rtx-pro-4000-dual", MemoryModel.DISCRETE, 24.0, 2, tested=True),
    "rtx-4070-super": TargetSpec("rtx-4070-super", MemoryModel.DISCRETE, 12.0, 1, tested=True),
    "rtx-4070-ti-super": TargetSpec("rtx-4070-ti-super", MemoryModel.DISCRETE, 16.0, 1, tested=True),
    # hardware-validated 2026-08-03: gemma-4-12b-it UD-Q8_K_XL (GGUF + vision) ที่ context 16,384
    "rtx-5090": TargetSpec("rtx-5090", MemoryModel.DISCRETE, 32.0, 1, tested=True),
    # ── รุ่นอ้างอิงยอดนิยม (ยังไม่ทดสอบจริง → conservative) ──
    # RTX 50 series (Blackwell) — 5090 ย้ายขึ้นไปกลุ่มที่ทดสอบจริงแล้ว
    "rtx-5080": TargetSpec("rtx-5080", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-5070-ti": TargetSpec("rtx-5070-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-5070": TargetSpec("rtx-5070", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-5060-ti": TargetSpec("rtx-5060-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    # RTX 40 series (Ada Lovelace)
    "rtx-4090": TargetSpec("rtx-4090", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-4080-super": TargetSpec("rtx-4080-super", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-4080": TargetSpec("rtx-4080", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-4070-ti": TargetSpec("rtx-4070-ti", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-4060-ti": TargetSpec("rtx-4060-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    # RTX 30 series (Ampere)
    "rtx-3090-ti": TargetSpec("rtx-3090-ti", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-3090": TargetSpec("rtx-3090", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-3080-ti": TargetSpec("rtx-3080-ti", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-3080": TargetSpec("rtx-3080", MemoryModel.DISCRETE, 10.0, 1, tested=False),
    "rtx-3060": TargetSpec("rtx-3060", MemoryModel.DISCRETE, 12.0, 1, tested=False),
}


def from_hardware_report(report: HardwareReport) -> TargetSpec | None:
    """สร้าง TargetSpec จากเครื่องที่ตรวจพบจริง — คืน None ถ้าไม่พบ GPU"""
    if not report.gpus:
        return None
    # nvidia-smi บนเครื่อง unified memory (DGX Spark GB10) มักรายงาน memory.total ไม่ได้ —
    # fallback ไปใช้สเปกจาก GPU allowlist
    memory_values: list[float] = []
    for gpu in report.gpus:
        if gpu.vram_mib:
            memory_values.append(gpu.vram_mib / 1024)
        elif gpu.known is not None:
            memory_values.append(gpu.known.vram_gb)
    if not memory_values:
        return None
    memory_model = (
        MemoryModel.UNIFIED
        if report.profile in (TargetProfile.DGX_SPARK_SINGLE, TargetProfile.DGX_SPARK_STACKED)
        else MemoryModel.DISCRETE
    )
    return TargetSpec(
        name="this-machine",
        memory_model=memory_model,
        memory_gb=round(min(memory_values), 1),
        gpu_count=len(report.gpus),
        system_ram_gb=report.ram_gb,
        tested=all(g.tested for g in report.gpus),
    )
