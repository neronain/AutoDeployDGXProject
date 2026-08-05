"""ตรวจฮาร์ดแวร์ของเครื่องปัจจุบันด้วย nvidia-smi / /proc — ทุกขั้น fail-safe

หมายเหตุ: โมดูลนี้ให้ข้อมูล "ตามที่ตรวจพบจริง" เท่านั้น ห้ามเดา — ถ้าตรวจไม่ได้ให้รายงานว่าตรวจไม่ได้
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .profiles import KnownGpu, TargetProfile, classify, lookup_gpu

_SMI_QUERY = "--query-gpu=name,memory.total,compute_cap,memory.used,utilization.gpu"


@dataclass
class DetectedGpu:
    name: str
    vram_mib: int | None
    compute_capability: str | None
    known: KnownGpu | None = None
    # ค่าสด ณ ตอนตรวจ — None เมื่อ nvidia-smi ไม่รายงาน (เช่น GB10 ที่ memory.total ว่าง)
    vram_used_mib: int | None = None
    utilization_pct: int | None = None

    @property
    def tested(self) -> bool:
        return bool(self.known and self.known.tested)


@dataclass
class HardwareReport:
    arch: str
    gpus: list[DetectedGpu] = field(default_factory=list)
    ram_gb: float | None = None
    disk_free_gb: float | None = None
    disk_total_gb: float | None = None
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

        def _int(index: int) -> int | None:
            if len(parts) > index and parts[index].replace(".", "").isdigit():
                return int(float(parts[index]))
            return None

        gpus.append(DetectedGpu(
            name=name, vram_mib=vram, compute_capability=cc, known=lookup_gpu(name),
            vram_used_mib=_int(3), utilization_pct=_int(4),
        ))

    for gpu in gpus:
        if gpu.known is None:
            notes.append(f"GPU นอก allowlist: {gpu.name} — Fit Analyzer จะใช้โหมด conservative")
    return gpus, notes



def detect_cpu() -> dict:
    """CPU: จำนวน core + โหลดเฉลี่ย 1 นาที แปลงเป็น % ของทั้งเครื่อง

    ใช้ loadavg แทนการวัด %util แบบ snapshot เพราะไม่ต้องรอสองครั้ง (หน้าเว็บ poll ทุก 5 วิ)
    · โหลด > 100% เป็นไปได้และหมายถึงมีงานรอคิว — ไม่ตัดเพดานให้ เพราะมันคือข้อมูล
    """
    import os as _os

    cores = _os.cpu_count()
    load1 = None
    try:
        load1 = round(_os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass
    percent = round(load1 / cores * 100) if load1 is not None and cores else None
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cores": cores, "load1": load1, "percent": percent, "model": model}


def detect_ram_gb() -> float | None:
    total, _ = detect_mem()
    return total


def detect_mem() -> tuple[float | None, float | None]:
    """คืน (RAM รวม GB, RAM ที่ว่างใช้ได้ GB) — None เมื่ออ่านไม่ได้ (เช่น ไม่ใช่ Linux)"""
    total = available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = round(int(line.split()[1]) / (1024 * 1024), 1)
                elif line.startswith("MemAvailable:"):
                    available = round(int(line.split()[1]) / (1024 * 1024), 1)
    except OSError:
        pass
    return total, available


def detect_disk_gb(path: str | None = None) -> tuple[float | None, float | None]:
    """คืน (พื้นที่ว่าง GB, พื้นที่ทั้งหมด GB) ของดิสก์ที่เก็บโมเดล

    ดูที่ $HOME เพราะ weight ลงที่ ~/.cache/huggingface (vLLM) หรือ ~/models (GGUF)
    ซึ่งเป็นสาเหตุ deploy ล้มที่พบบ่อยสุดกับโมเดลใหญ่
    """
    target = path or str(Path.home())
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return None, None
    return round(usage.free / (1024**3), 1), round(usage.total / (1024**3), 1)


def primary_ip() -> str:
    """IP หลักของเครื่อง (เส้นทางออก default) — ไม่มีการส่งข้อมูลจริง"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


@dataclass
class HostSummary:
    hostname: str
    ip: str
    arch: str
    profile: TargetProfile
    gpus: list[DetectedGpu] = field(default_factory=list)
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None

    @property
    def ram_used_gb(self) -> float | None:
        if self.ram_total_gb is None or self.ram_available_gb is None:
            return None
        return round(self.ram_total_gb - self.ram_available_gb, 1)


def host_summary() -> HostSummary:
    """ข้อมูลเครื่องแบบเบา (ไม่แตะ docker) — ใช้กับ lmds ps"""
    gpus, _ = detect_gpus()
    total, available = detect_mem()
    return HostSummary(
        hostname=platform.node(),
        ip=primary_ip(),
        arch=platform.machine(),
        profile=classify([g.name for g in gpus]),
        gpus=gpus,
        ram_total_gb=total,
        ram_available_gb=available,
    )



# ── network fabric: ConnectX / 200G / RDMA ────────────────────────────────────
# ใช้ตัดสินว่าเครื่องนี้ "ต่อ stacked ได้ไหม" — stacked (TP ข้ามเครื่อง) ยิง KV/activation
# ผ่านสายตลอดเวลา ถ้าไม่ใช่ RDMA 100G+ จะช้าจนไม่คุ้มเทียบกับรันแยกเครื่อง
_MELLANOX_VENDOR = "0x15b3"
_RDMA_DRIVERS = {"mlx5_core", "mlx4_core", "irdma", "ionic", "bnxt_en"}


def _sysfs(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _iface_addresses() -> dict[str, str]:
    """iface → IPv4 — ใช้เสนอค่าให้ผู้ใช้เลือกเป็น cluster IP ไม่ใช่ตั้งให้เอง

    stacked ต้องรู้ว่า rank คุยกันทาง IP ไหน ซึ่งมักคนละเส้นกับ IP ที่ใช้ SSH
    """
    out = _run(["ip", "-o", "-4", "addr", "show"], timeout=5)
    if not out:
        return {}
    addresses: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] != "inet":
            continue
        addresses.setdefault(parts[1], parts[3].split("/")[0])
    return addresses


def detect_fabric() -> dict:
    """การ์ดเครือข่ายความเร็วสูง + RDMA ของเครื่องนี้

    อ่านจาก /sys เท่านั้น (ไม่ต้อง root ไม่ต้องมี ethtool/ibstat) — บนเครื่องที่ไม่ใช่ Linux
    หรืออ่านไม่ได้จะได้ tier="unknown" ไม่ใช่การเดา
    """
    net = Path("/sys/class/net")
    if not net.is_dir():
        return {"links": [], "rdma_devices": [], "best_gbps": None, "tier": "unknown",
                "cluster_capable": False, "summary": "ตรวจไม่ได้บนเครื่องนี้"}

    rdma_devices = sorted(p.name for p in Path("/sys/class/infiniband").glob("*")) \
        if Path("/sys/class/infiniband").is_dir() else []

    addresses = _iface_addresses()
    links = []
    for iface in sorted(net.iterdir()):
        name = iface.name
        if name == "lo" or (iface / "device").exists() is False:
            continue
        driver = ""
        try:
            driver = (iface / "device" / "driver").resolve().name
        except OSError:
            pass
        vendor = _sysfs(str(iface / "device" / "vendor"))
        state = _sysfs(str(iface / "operstate")) or "unknown"
        raw_speed = _sysfs(str(iface / "speed"))
        # speed อ่านได้เฉพาะตอนลิงก์ขึ้น — ลิงก์ลงจะได้ -1 หรือ error
        gbps = None
        if raw_speed.lstrip("-").isdigit() and int(raw_speed) > 0:
            gbps = int(raw_speed) // 1000 or None
        address = addresses.get(name, "")
        links.append({
            "iface": name,
            "ip": address,
            # 169.254.x.x = ที่อยู่ที่เครื่องตั้งเองเพราะไม่มี DHCP/config — ลิงก์ขึ้นแต่ยังไม่ได้ตั้งค่า
            "link_local": address.startswith("169.254."),
            "speed_gbps": gbps,
            "driver": driver,
            "state": state,
            "connectx": vendor == _MELLANOX_VENDOR or driver.startswith("mlx"),
            "rdma": driver in _RDMA_DRIVERS and bool(rdma_devices),
        })

    up = [l for l in links if l["state"] == "up" and l["speed_gbps"]]
    best = max((l["speed_gbps"] for l in up), default=None)
    # เส้นที่ตั้งค่าแล้วมาก่อนเสมอ ต่อให้ link-local จะเร็วเท่ากัน — ยิง NCCL ไป 169.254 ไม่ถึงกัน
    fastest = max(up, key=lambda l: (not l["link_local"], l["speed_gbps"]), default=None)
    has_rdma = bool(rdma_devices) and any(l["rdma"] for l in up)

    if best is None:
        tier, summary = "unknown", "ไม่พบลิงก์ที่ขึ้นและรายงานความเร็วได้"
    elif has_rdma and best >= 100:
        tier = "rdma"
        summary = f"{fastest['iface']} {best}G RDMA ({fastest['driver']}) — stacked ได้เต็มที่"
    elif best >= 100:
        tier = "fast"
        summary = f"{fastest['iface']} {best}G แต่ยังไม่เห็น RDMA — stacked ได้แต่ควรเปิด RoCE ก่อน"
    elif best >= 25:
        tier = "fast"
        summary = f"{fastest['iface']} {best}G — พอ stacked ได้ แต่ช้ากว่า 100G ชัดเจน"
    else:
        tier = "basic"
        summary = f"{fastest['iface']} {best}G — เร็วไม่พอสำหรับ stacked ให้รันแยกเครื่อง"

    return {
        "links": [l for l in links if l["connectx"] or l["speed_gbps"]],
        "rdma_devices": rdma_devices,
        "best_gbps": best,
        "tier": tier,
        "cluster_capable": tier in {"rdma", "fast"},
        "summary": summary,
    }


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
    disk_free, disk_total = detect_disk_gb()
    report = HardwareReport(
        arch=platform.machine(),
        gpus=gpus,
        ram_gb=detect_ram_gb(),
        disk_free_gb=disk_free,
        disk_total_gb=disk_total,
        docker=docker,
        nvidia_container_toolkit=toolkit,
        profile=classify([g.name for g in gpus]),
        notes=notes,
    )
    if docker and not toolkit and gpus:
        report.notes.append("Docker ใช้ได้แต่ไม่พบ nvidia runtime — ติดตั้ง NVIDIA Container Toolkit ก่อนรัน bundle")
    if disk_free is not None and disk_free < 50:
        report.notes.append(
            f"ดิสก์เหลือ {disk_free} GB — โมเดลขนาดกลางขึ้นไปอาจโหลดไม่ลง "
            "(runtime image ~10-20 GB + weight) ย้ายที่เก็บด้วย HF_HOME/MODEL_DIR ได้"
        )
    return report
