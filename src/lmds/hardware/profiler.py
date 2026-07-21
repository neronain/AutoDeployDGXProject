"""ตรวจฮาร์ดแวร์ของเครื่องปัจจุบันด้วย nvidia-smi / /proc — ทุกขั้น fail-safe

หมายเหตุ: โมดูลนี้ให้ข้อมูล "ตามที่ตรวจพบจริง" เท่านั้น ห้ามเดา — ถ้าตรวจไม่ได้ให้รายงานว่าตรวจไม่ได้
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from .profiles import KnownGpu, TargetProfile, classify, lookup_gpu

_SMI_QUERY = "--query-gpu=name,memory.total,compute_cap"


@dataclass
class DetectedGpu:
    name: str
    vram_mib: int | None
    compute_capability: str | None
    known: KnownGpu | None = None

    @property
    def tested(self) -> bool:
        return bool(self.known and self.known.tested)


@dataclass
class HardwareReport:
    arch: str
    gpus: list[DetectedGpu] = field(default_factory=list)
    ram_gb: float | None = None
    docker: bool = False
    nvidia_container_toolkit: bool = False
    profile: TargetProfile = TargetProfile.UNKNOWN
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def detect_gpus() -> tuple[list[DetectedGpu], list[str]]:
    notes: list[str] = []
    if shutil.which("nvidia-smi") is None:
        notes.append("ไม่พบ nvidia-smi — ตรวจ GPU ไม่ได้")
        return [], notes
    out = _run(["nvidia-smi", _SMI_QUERY, "--format=csv,noheader,nounits"])
    if out is None:
        notes.append("nvidia-smi รันไม่สำเร็จ — ตรวจ GPU ไม่ได้")
        return [], notes

    gpus: list[DetectedGpu] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        vram = int(float(parts[1])) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else None
        cc = parts[2] if len(parts) > 2 and parts[2] else None
        gpus.append(DetectedGpu(name=name, vram_mib=vram, compute_capability=cc, known=lookup_gpu(name)))

    for gpu in gpus:
        if gpu.known is None:
            notes.append(f"GPU นอก allowlist: {gpu.name} — Fit Analyzer จะใช้โหมด conservative")
    return gpus, notes


def detect_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 1)
    except OSError:
        pass
    return None


def detect_docker() -> tuple[bool, bool]:
    if shutil.which("docker") is None:
        return False, False
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    if info is None:
        return True, False
    return True, "nvidia" in info


def probe() -> HardwareReport:
    gpus, notes = detect_gpus()
    docker, toolkit = detect_docker()
    report = HardwareReport(
        arch=platform.machine(),
        gpus=gpus,
        ram_gb=detect_ram_gb(),
        docker=docker,
        nvidia_container_toolkit=toolkit,
        profile=classify([g.name for g in gpus]),
        notes=notes,
    )
    if docker and not toolkit and gpus:
        report.notes.append("Docker ใช้ได้แต่ไม่พบ nvidia runtime — ติดตั้ง NVIDIA Container Toolkit ก่อนรัน bundle")
    return report
