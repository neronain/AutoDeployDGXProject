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
    "rtx-4090": TargetSpec("rtx-4090", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-5090": TargetSpec("rtx-5090", MemoryModel.DISCRETE, 32.0, 1, tested=False),
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
