"""ตรวจฮาร์ดแวร์ของเครื่องปัจจุบันด้วย nvidia-smi / /proc — ทุกขั้น fail-safe

หมายเหตุ: โมดูลนี้ให้ข้อมูล "ตามที่ตรวจพบจริง" เท่านั้น ห้ามเดา — ถ้าตรวจไม่ได้ให้รายงานว่าตรวจไม่ได้
"""

from __future__ import annotations

import ipaddress
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .profiles import KnownGpu, TargetProfile, classify, lookup_gpu

# ค่าที่ถามจาก nvidia-smi — เรียงตามนี้แล้วอ่านตามลำดับ
# GB10 (unified SoC) ตอบ [N/A] หลายตัว (power.limit, fan, clocks.mem, memory.total)
# ต้องแยก "ไม่มีค่า" ออกจาก "ศูนย์" ให้ชัด ไม่งั้นหน้าจอจะโชว์ 0W ทั้งที่การ์ดทำงานอยู่
_SMI_FIELDS = (
    "name", "memory.total", "compute_cap", "memory.used", "utilization.gpu",
    "temperature.gpu", "power.draw", "power.limit", "fan.speed",
    "clocks.gr", "clocks.max.gr", "clocks.mem", "clocks.sm",
    "pcie.link.gen.current", "pcie.link.width.current",
)
_SMI_QUERY = "--query-gpu=" + ",".join(_SMI_FIELDS)


@dataclass
class DetectedGpu:
    name: str
    vram_mib: int | None
    compute_capability: str | None
    known: KnownGpu | None = None
    # ค่าสด ณ ตอนตรวจ — None เมื่อ nvidia-smi ไม่รายงาน (เช่น GB10 ที่ memory.total ว่าง)
    vram_used_mib: int | None = None
    utilization_pct: int | None = None
    # telemetry — None = การ์ดรุ่นนี้ไม่รายงาน ไม่ใช่ค่าเป็นศูนย์
    temperature_c: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: int | None = None
    clock_graphics_mhz: int | None = None
    clock_graphics_max_mhz: int | None = None
    clock_memory_mhz: int | None = None
    clock_sm_mhz: int | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None

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



def compute_apps() -> list[tuple[int, str, int]]:
    """process ที่ถือหน่วยความจำ GPU อยู่ — (pid, ชื่อ, MiB)

    บนเครื่อง unified memory (GB10/DGX Spark) `--query-gpu=memory.used` คืน `[N/A]`
    ทั้ง memory.total ด้วย เพราะไม่มี VRAM แยกให้รายงาน แต่ `--query-compute-apps`
    ยังบอกได้ว่าใครถืออะไรไว้เท่าไร

    เคสจริง 2026-08-13 — msi-4 เพิ่งถูกแอดเข้าฟลีต รายงานว่า "0 โมเดล" และ
    vram_used_gb เป็น None ทั้งที่ container SGLang รันมา 32 ชั่วโมงและถือ GPU ไว้
    96,073 MiB · เครื่องที่เหลือจริงไม่ถึง 20 GB จึงดูเหมือนว่างทั้ง 121 GB
    """
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    apps: list[tuple[int, str, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            apps.append((int(parts[0]), parts[1], int(float(parts[2]))))
        except ValueError:
            continue  # [N/A] ในคอลัมน์ไหนก็ตาม = แถวนั้นใช้ไม่ได้ ไม่ใช่ศูนย์
    return apps


def memory_held_gb() -> float:
    """GPU memory ที่ process ทั้งหมดถืออยู่บนเครื่องนี้ (GB) — อ่านไม่ได้ = 0

    ตัวเดียวกับที่ `lmds ps` ใช้รายงาน foreign workload · fit เอาไปหักออกจาก budget
    ก่อนวางแผน deploy ตัวถัดไปลงเครื่องเดียวกัน · คืน 0 เมื่ออ่านไม่ได้ เพื่อให้พฤติกรรม
    ถอยกลับไปเท่าเดิม ไม่ใช่ล้มทั้งคำสั่ง
    """
    try:
        return sum(mib for _, _, mib in compute_apps() if mib) / 1024.0
    except Exception:  # noqa: BLE001 — nvidia-smi พังไม่ควรทำให้วางแผนไม่ได้
        return 0.0


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

        def _num(index: int):
            """ค่าที่อ่านไม่ได้/[N/A] → None — อย่าแปลงเป็น 0 เพราะความหมายคนละอย่าง"""
            if len(parts) <= index:
                return None
            raw = parts[index].strip()
            try:
                return float(raw)
            except ValueError:
                return None

        def _int(index: int) -> int | None:
            value = _num(index)
            return None if value is None else int(value)

        gpus.append(DetectedGpu(
            name=name, vram_mib=vram, compute_capability=cc, known=lookup_gpu(name),
            vram_used_mib=_int(3), utilization_pct=_int(4),
            temperature_c=_int(5), power_w=_num(6), power_limit_w=_num(7), fan_pct=_int(8),
            clock_graphics_mhz=_int(9), clock_graphics_max_mhz=_int(10),
            clock_memory_mhz=_int(11), clock_sm_mhz=_int(12),
            pcie_gen=_int(13), pcie_width=_int(14),
        ))

    # เครื่อง unified memory ไม่รายงาน memory.used — รวมจาก process ที่ถืออยู่แทน
    # ปล่อยให้เป็น None แปลว่า "ว่างทั้งเครื่อง" ในสายตาของ fit ซึ่งไม่จริงและอันตราย
    if gpus and all(gpu.vram_used_mib is None for gpu in gpus):
        apps = compute_apps()
        if apps:
            total = sum(mib for _, _, mib in apps)
            # การ์ดเดียวคือเคสที่รู้แน่ว่าของทั้งหมดอยู่ใบไหน หลายใบแล้วเดาไม่ได้ว่า
            # process ไหนอยู่ใบไหน จึงบอกเป็น note แทนที่จะหารเฉลี่ยมั่ว ๆ
            if len(gpus) == 1:
                gpus[0].vram_used_mib = total
            else:
                notes.append(
                    f"มี process ถือ GPU อยู่รวม {total:,} MiB แต่มีการ์ด {len(gpus)} ใบ "
                    "— แยกไม่ได้ว่าใบไหนเท่าไร"
                )

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
    # ที่อยู่ IPv4 ทุกเส้นของเครื่องนี้ — `ip` ข้างบนคือเส้นที่ออกเน็ตเส้นเดียว ซึ่งบ่อยครั้ง
    # ไม่ใช่เส้นที่คนอื่นใช้ยิงเข้ามา (VM/คอนเทนเนอร์ที่ NAT ออกไป, เครื่องที่มีทั้ง LAN
    # และ Tailscale) · ว่าง = ตรวจไม่ได้บนเครื่องนี้ ไม่ใช่ "ไม่มีเน็ต"
    addresses: list[dict] = field(default_factory=list)

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
        addresses=local_addresses(),
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


# การ์ดเสมือนที่ไม่ใช่ "ทางเข้าเครื่องนี้" — bridge ของ docker/k8s กับปลาย veth ของคอนเทนเนอร์
# ที่อยู่บนเส้นพวกนี้ยิงจากเครื่องอื่นไม่ถึง เอามาโชว์ปนกับ IP จริงคือชวนให้คนก๊อปผิดเส้น
_VIRTUAL_IFACE_PREFIXES = ("docker", "br-", "veth", "virbr", "cni", "flannel")

# `ifconfig` มีสองสำเนียง: ของใหม่เขียน `inet 10.0.0.5 netmask 255.255.255.0`
# ของ net-tools รุ่นเก่าเขียน `inet addr:10.0.0.5  Mask:255.255.255.0` — รับทั้งคู่
_INET_RE = re.compile(r"\binet\s+(?:addr:)?(\d+\.\d+\.\d+\.\d+)")
_MASK_RE = re.compile(r"(?:\bnetmask\s+|\bMask:)(\S+)")


def _netmask_bits(raw: str) -> int | None:
    """netmask → prefix length · macOS ให้มาเป็นเลขฐานสิบหก (`0xffffff00`) ไม่ใช่ 255.255.255.0"""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = int(raw, 16) if raw.lower().startswith("0x") else int(ipaddress.IPv4Address(raw))
    except ValueError:
        return None
    return bin(value).count("1")


def _ifconfig_addresses() -> dict[str, str]:
    """ทางสำรองเมื่อไม่มี iproute2 — คืนรูปเดียวกับ `ip -o -4 addr show` (iface → CIDR)

    เจอตอนเทสบน macOS/OrbStack: เครื่องที่ไม่มีคำสั่ง `ip` ทำให้ทั้ง `lmds info` และ
    การ์ดของเครื่องนั้นตอบว่า "ตรวจไม่ได้" ทั้งที่มี IP อยู่ครบ · `ifconfig` มีมาให้ทั้ง
    macOS และ image เก่าที่ยังไม่ย้ายไป iproute2 จึงใช้เป็นตาข่ายรับได้
    """
    out = _run(["ifconfig", "-a"], timeout=5)
    if not out:
        return {}
    addresses: dict[str, str] = {}
    iface = ""
    for line in out.splitlines():
        if not line.strip():
            continue
        # บรรทัดที่ไม่ย่อหน้า = ขึ้นการ์ดใบใหม่ ("en0: flags=…" หรือ "eth0  Link encap:…")
        if not line[0].isspace():
            iface = line.split()[0].rstrip(":")
            continue
        found = _INET_RE.search(line)
        if not iface or not found:
            continue
        mask = _MASK_RE.search(line)
        bits = _netmask_bits(mask.group(1)) if mask else None
        addresses.setdefault(iface, f"{found.group(1)}/{bits}" if bits is not None
                             else found.group(1))
    return addresses


def _iface_addresses() -> dict[str, str]:
    """iface → IPv4 — ใช้เสนอค่าให้ผู้ใช้เลือกเป็น cluster IP ไม่ใช่ตั้งให้เอง

    stacked ต้องรู้ว่า rank คุยกันทาง IP ไหน ซึ่งมักคนละเส้นกับ IP ที่ใช้ SSH
    """
    out = _run(["ip", "-o", "-4", "addr", "show"], timeout=5)
    if not out:
        return _ifconfig_addresses()
    addresses: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] != "inet":
            continue
        addresses.setdefault(parts[1], parts[3])  # เก็บ CIDR ไว้ทั้งก้อน เช่น 10.100.152.1/24
    return addresses or _ifconfig_addresses()


def local_addresses() -> list[dict]:
    """IPv4 ทุกเส้นของเครื่องนี้ — เส้นที่เครื่องอื่นยิงเข้ามาได้จริง เรียงเส้นที่น่าใช้ไว้บน

    ทะเบียนของ hub รู้จัก node จาก "ที่อยู่ที่ใช้ SSH" อย่างเดียว ซึ่งบ่อยครั้งไม่ใช่ IP
    (`orb`, `spark1.local`, ชื่อบน Tailscale) · พอถึงเวลาต้องบอกลูกค้าว่ายิง API ไปที่ไหน
    หรือจะตั้ง cluster IP ก็ไม่มีที่ไหนบนหน้าจอบอกได้ว่าเครื่องนั้นถือ IP อะไรอยู่จริง

    `primary_ip()` ตอบได้ทีละเส้นเดียว (เส้นที่ default route ออก) ซึ่งบนเครื่องที่ NAT
    ออกไป — คอนเทนเนอร์, VM ของ OrbStack, เครื่องที่มีทั้ง LAN และ Tailscale — ไม่ใช่
    เส้นที่คนอื่นใช้เข้ามา · ตัดเส้นเสมือนของ docker/k8s ทิ้งเพราะยิงจากข้างนอกไม่ถึง
    """
    default = primary_ip()
    out: list[dict] = []
    for iface, cidr in _iface_addresses().items():
        if iface == "lo" or iface.startswith(_VIRTUAL_IFACE_PREFIXES):
            continue
        address, _, prefix = cidr.partition("/")
        try:
            parsed = ipaddress.IPv4Address(address)
        except ValueError:
            continue
        if parsed.is_loopback:
            continue
        out.append({
            "iface": iface,
            "ip": address,
            # ต้องมี prefix ถึงจะบอกได้ว่าสองเครื่องอยู่วงเดียวกันไหม
            "prefix": int(prefix) if prefix.isdigit() else None,
            # 169.254.x.x = เครื่องตั้งเองเพราะไม่มี DHCP — ลิงก์ขึ้นแต่ยังคุยกับใครไม่ได้
            "link_local": parsed.is_link_local,
            # เส้นที่ออกเน็ต = ตัวเดียวกับที่เคยรายงานเป็น `ip` โดด ๆ
            "primary": address == default,
        })
    out.sort(key=lambda a: (not a["primary"], a["link_local"], a["iface"]))
    # เครื่องที่อ่านรายชื่อการ์ดไม่ได้เลยยังต้องบอก IP ที่ใช้ออกเน็ตได้ ไม่งั้นหน้าจอว่างเปล่า
    # ทั้งที่ `lmds info` ตอบ IP ตัวนี้มาตลอด — ว่างต้องแปลว่า "ไม่มีเลยจริง ๆ" เท่านั้น
    if default and default != "127.0.0.1" and not any(a["primary"] for a in out):
        out.insert(0, {"iface": "", "ip": default, "prefix": None,
                       "link_local": False, "primary": True})
    return out


# ชื่อ interface ของ ConnectX-7 บน DGX Spark เหมือนกันทุกเครื่อง: พอร์ต QSFP หนึ่งช่องคือ
# PCIe Gen5 x4 สองเส้น = สอง interface (f0/f1) · พอร์ต 1 (ข้าง RJ45) อยู่ PCI domain 0000
# (`enp1s0f0np0`/`enp1s0f1np1`) · พอร์ต 2 อยู่ domain 0002 (`enP2p1s0f0np0`/`enP2p1s0f1np1`)
_SPARK_IFACE_RE = re.compile(r"^en(?P<domain>P\d+)?p(?P<bus>\d+)s(?P<slot>\d+)f(?P<fn>\d)(?:np\d)?$")
NVIDIA_SYNC_NETPLAN = "99-nvidia-sync-cluster.yaml"
LMDS_NETPLAN = "99-lmds-cluster.yaml"


def _pci_domain_and_function(iface: Path) -> tuple[int | None, int | None]:
    """(PCI domain, PCI function) ของการ์ด — จาก symlink device (`0002:01:00.1` → (2, 1))"""
    try:
        address = (iface / "device").resolve().name
    except OSError:
        return None, None
    match = re.match(r"^([0-9a-f]{4}):[0-9a-f]{2}:[0-9a-f]{2}\.(\d)$", address, re.I)
    if not match:
        return None, None
    return int(match.group(1), 16), int(match.group(2))


def spark_port_of(name: str, domain: int | None = None) -> int | None:
    """พอร์ต QSFP (1 หรือ 2) ของ interface ชื่อนี้ — None เมื่อไม่ใช่รูปแบบของ Spark

    domain มาก่อนชื่อ (ของจริงจาก /sys) · ไม่มี /sys (node รุ่นเก่าส่งแต่ชื่อมา) เดาจากชื่อ:
    `enp…` = domain 0 = พอร์ต 1 · `enP2…` = domain 2 = พอร์ต 2
    """
    if domain is not None:
        return {0: 1, 2: 2}.get(domain)
    match = _SPARK_IFACE_RE.match(name or "")
    if not match:
        return None
    return 2 if match.group("domain") else 1


def spark_function_of(name: str) -> int | None:
    match = _SPARK_IFACE_RE.match(name or "")
    return int(match.group("fn")) if match else None


def _netplan_mentions(netplan_dir: Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """(ไฟล์ทั้งหมดใน /etc/netplan, iface → ไฟล์ที่เอ่ยถึง, ไฟล์ที่อ่านไม่ได้)

    ไฟล์ netplan บน Spark เป็น 0600 ของ root (NVIDIA Sync ก็เขียนแบบนั้น) — อ่านไม่ได้แปลว่า
    "ไม่รู้" ไม่ใช่ "ไม่ได้ถูกจัดการ" · ผู้เรียกจึงได้ทั้งรายชื่อไฟล์ (ls อ่านได้เสมอ) และรายการที่อ่านไม่ออก
    """
    if not netplan_dir.is_dir():
        return [], {}, []
    files = sorted(p.name for p in netplan_dir.glob("*.yaml"))
    mentions: dict[str, list[str]] = {}
    unreadable: list[str] = []
    for name in files:
        try:
            text = (netplan_dir / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            unreadable.append(name)
            continue
        # interface ใน netplan อยู่ใต้ ethernets: เป็น key ย่อหน้า — จับแค่ "<ชื่อ>:" ต้นบรรทัดพอ
        for found in re.findall(r"^\s+([A-Za-z0-9_.-]+):\s*$", text, re.M):
            mentions.setdefault(found, []).append(name)
    return files, mentions, unreadable


def detect_fabric(sys_net: str | Path = "/sys/class/net", sys_ib: str | Path = "/sys/class/infiniband",
                  netplan_dir: str | Path = "/etc/netplan", addresses: dict[str, str] | None = None) -> dict:
    """การ์ดเครือข่ายความเร็วสูง + RDMA ของเครื่องนี้

    อ่านจาก /sys เท่านั้น (ไม่ต้อง root ไม่ต้องมี ethtool/ibstat) — บนเครื่องที่ไม่ใช่ Linux
    หรืออ่านไม่ได้จะได้ tier="unknown" ไม่ใช่การเดา

    ตั้งแต่ 0.6.0 รายงานต่อ interface เพิ่ม: พอร์ต QSFP · function (f0/f1) · carrier (มีสายเสียบ
    และลิงก์ขึ้น) · RDMA device · ถูก netplan จัดการอยู่ไหม — และสรุปเป็นรายพอร์ต (`qsfp_ports`)
    เพื่อให้ hub วางแผนต่อสาย/ตั้ง IP ให้ Spark ที่ยังไม่ได้ตั้งค่าได้ · คีย์เดิมยังอยู่ครบ
    (pass path มาได้เพื่อเทสกับ /sys ปลอม)
    """
    net = Path(sys_net)
    if not net.is_dir():
        return {"links": [], "rdma_devices": [], "best_gbps": None, "tier": "unknown",
                "cluster_capable": False, "summary": "ตรวจไม่ได้บนเครื่องนี้",
                "qsfp_ports": [], "netplan_files": [], "nvidia_sync_netplan": False}

    ib_root = Path(sys_ib)
    rdma_devices = sorted(p.name for p in ib_root.glob("*")) if ib_root.is_dir() else []
    netplan_files, netplan_mentions, netplan_unreadable = _netplan_mentions(Path(netplan_dir))

    addresses = _iface_addresses() if addresses is None else addresses
    links = []
    for iface in sorted(net.iterdir()):
        name = iface.name
        # NIC เสมือน (VM/cloud) ไม่มี symlink device/ — เดิมข้ามทิ้งทั้งใบ ทำให้เครื่องแบบนั้น
        # รายงานว่า "ไม่มีเครือข่ายเลย" ทั้งที่มี IP อยู่ · ข้ามเฉพาะ loopback กับ virtual bridge
        # ที่ไม่ใช่ทางออกจริงก็พอ
        if name == "lo" or name.startswith(_VIRTUAL_IFACE_PREFIXES):
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
        # carrier = "1" คือมีสายเสียบและอีกฝั่งขึ้น (LOWER_UP ของ `ip link`) · อ่านไม่ได้ (EINVAL ตอน
        # interface ถูก down) = ไม่รู้ ไม่ใช่ "ไม่มีสาย"
        raw_carrier = _sysfs(str(iface / "carrier"))
        carrier = True if raw_carrier == "1" else False if raw_carrier == "0" else None
        domain, function = _pci_domain_and_function(iface)
        connectx = vendor == _MELLANOX_VENDOR or driver.startswith("mlx")
        rdma_device = ""
        try:
            rdma_device = sorted(p.name for p in (iface / "device" / "infiniband").iterdir())[0]
        except (OSError, IndexError):
            pass
        cidr = addresses.get(name, "")
        address, _, prefix = cidr.partition("/")
        managed_by = netplan_mentions.get(name, [])
        links.append({
            "iface": name,
            "ip": address,
            # ต้องรู้ prefix ถึงจะบอกได้ว่าสองเครื่องอยู่วงเดียวกันไหม (Spark มี fabric หลายวง)
            "prefix": int(prefix) if prefix.isdigit() else None,
            # 169.254.x.x = ที่อยู่ที่เครื่องตั้งเองเพราะไม่มี DHCP/config — ลิงก์ขึ้นแต่ยังไม่ได้ตั้งค่า
            "link_local": address.startswith("169.254."),
            "speed_gbps": gbps,
            "driver": driver,
            "state": state,
            "connectx": connectx,
            "rdma": driver in _RDMA_DRIVERS and bool(rdma_devices),
            "carrier": carrier,
            "qsfp_port": spark_port_of(name, domain) if connectx else None,
            "function": function if function is not None else spark_function_of(name),
            "rdma_device": rdma_device,
            # True/False เมื่ออ่านไฟล์ netplan ได้ · None = มีไฟล์ที่อ่านไม่ได้ (root 0600) จึงตอบไม่ได้
            "netplan_managed": (True if managed_by else (None if netplan_unreadable else False)),
            "netplan_files": managed_by,
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
        # เก็บลิงก์ที่ "มีความหมาย" ไว้ทั้งหมด: การ์ดจริง หรือรู้ความเร็ว หรืออย่างน้อยมี IP
        # เครื่อง controller บน VM มีแต่ eth0 ที่ไม่มี device/ — ถ้ากรองทิ้งจะดูเหมือนเครื่องพัง
        "links": [l for l in links if l["connectx"] or l["speed_gbps"] or l["ip"]],
        "rdma_devices": rdma_devices,
        "best_gbps": best,
        "tier": tier,
        "cluster_capable": tier in {"rdma", "fast"},
        "summary": summary,
        # มุมมองรายพอร์ต QSFP — สิ่งที่คนเสียบสายมองเห็น (interface สองตัวต่อพอร์ตคือรายละเอียดของ PCIe)
        "qsfp_ports": group_qsfp_ports(links),
        "netplan_files": netplan_files,
        "netplan_unreadable": netplan_unreadable,
        # NVIDIA Sync เขียนไฟล์ชื่อนี้ — มีอยู่แปลว่าเครื่องเคยถูกตั้งคลัสเตอร์ด้วยเครื่องมือของ NVIDIA
        "nvidia_sync_netplan": NVIDIA_SYNC_NETPLAN in netplan_files,
    }


def group_qsfp_ports(links: list[dict]) -> list[dict]:
    """จัด interface ของ ConnectX เป็นรายพอร์ต QSFP — ใช้ได้กับ payload จาก node รุ่นเก่าด้วย

    (รุ่นเก่าไม่มี `qsfp_port`/`carrier` → เดาพอร์ตจากชื่อ และถือว่า state=up + มี speed = มีสาย)
    """
    by_port: dict[int, list[dict]] = {}
    for link in links:
        if not link.get("connectx"):
            continue
        port = link.get("qsfp_port") or spark_port_of(link.get("iface") or "")
        if port is None:
            continue
        by_port.setdefault(port, []).append(link)
    out = []
    for port, members in sorted(by_port.items()):
        members = sorted(members, key=lambda l: (l.get("function") if l.get("function") is not None
                                                 else spark_function_of(l.get("iface") or "") or 0))
        carriers = [l.get("carrier") for l in members]
        if any(c is not None for c in carriers):
            carrier = any(c is True for c in carriers)
        else:
            carrier = any((l.get("state") == "up") and bool(l.get("speed_gbps")) for l in members)
        configured = [l for l in members if l.get("ip") and not str(l["ip"]).startswith("169.254.")]
        speed = max((l.get("speed_gbps") or 0 for l in members), default=0) or None
        out.append({
            "port": port,
            "ifaces": [l["iface"] for l in members],
            "carrier": carrier,
            "speed_gbps": speed,
            # interface ที่ถือ IP จริงอยู่บนพอร์ตนี้ (ว่าง = ยังไม่ได้ตั้ง)
            "configured": configured[0]["iface"] if configured else "",
            "ip": configured[0]["ip"] if configured else "",
            "prefix": configured[0].get("prefix") if configured else None,
            "rdma_devices": [l.get("rdma_device") for l in members if l.get("rdma_device")],
            "netplan_managed": any(l.get("netplan_managed") is True for l in members) or (
                None if any(l.get("netplan_managed") is None for l in members) else False),
        })
    return out


def detect_docker() -> tuple[bool, bool]:
    """(docker ใช้ได้, มี NVIDIA Container Toolkit)

    "มี toolkit" ไม่เท่ากับ "runtime ชื่อ nvidia ลงทะเบียนกับ docker" — Docker ≥25 ส่ง GPU ให้
    container ด้วย `--gpus` ผ่าน CDI ได้โดยไม่ต้องลงทะเบียน runtime · เคสจริง 2026-09-03 RTX4000
    (Docker 29, toolkit 1.20): `docker run --gpus all … nvidia-smi -L` เห็น GPU ครบสองใบ แต่ตรวจ
    แบบเดิมรายงาน ❌ แล้วบอกให้ไปลง toolkit ที่ลงอยู่แล้ว
    """
    if shutil.which("docker") is None:
        return False, False
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    if info and "nvidia" in info:
        return True, True
    toolkit_binary = shutil.which("nvidia-ctk") or shutil.which("nvidia-container-cli")
    cdi_spec = any(Path(p).is_file() for p in ("/etc/cdi/nvidia.yaml", "/var/run/cdi/nvidia.yaml"))
    return True, bool(toolkit_binary or cdi_spec)


def nvidia_runtime_registered() -> bool:
    """runtime ชื่อ nvidia อยู่ใน docker info ไหม — ใช้แยกโน้ต ไม่ใช่เงื่อนไขว่าใช้ GPU ได้"""
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    return bool(info and "nvidia" in info)


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
        report.notes.append("Docker ใช้ได้แต่ไม่พบ NVIDIA Container Toolkit — ติดตั้งก่อนรัน bundle "
                            "(sudo apt install nvidia-container-toolkit)")
    elif docker and toolkit and gpus and not nvidia_runtime_registered():
        report.notes.append("มี toolkit แต่ runtime nvidia ยังไม่ลงทะเบียนกับ docker — Docker ≥25 ใช้ --gpus "
                            "ผ่าน CDI ได้อยู่แล้ว · ถ้า start แล้ว container ไม่เห็น GPU: "
                            "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")
    if disk_free is not None and disk_free < 50:
        report.notes.append(
            f"ดิสก์เหลือ {disk_free} GB — โมเดลขนาดกลางขึ้นไปอาจโหลดไม่ลง "
            "(runtime image ~10-20 GB + weight) ย้ายที่เก็บด้วย HF_HOME/MODEL_DIR ได้"
        )
    return report


def suggest_target(gpus: list) -> str:
    """ชื่อ preset ของ `lmds deploy --target` ที่ตรงกับการ์ดในเครื่อง — "" ถ้าเดาไม่ได้

    หน้า hardware โชว์ profile (`rtx-multi-gpu`) ซึ่งไม่ใช่ชื่อ target · เคสจริง 2026-09-03: เอาชื่อนั้น
    ไปใส่ --target แล้วโดนปฏิเสธ ทั้งที่ preset `rtx-pro-4000-dual` มีอยู่ — ระบบควรบอกชื่อที่ใช้ได้เอง
    """
    from lmds.fit import PRESETS

    if not gpus:
        return ""
    name = (gpus[0].name or "").lower()
    if "gb10" in name or "dgx spark" in name:
        return "dgx-spark-single"
    m = re.search(r"rtx\s*(pro)?\s*(\d{4})\s*(ti\s*super|ti|super)?", name)
    if not m:
        return ""
    base = f"rtx-{'pro-' if m.group(1) else ''}{m.group(2)}"
    suffix = (m.group(3) or "").replace(" ", "")
    if suffix:
        base += "-" + ("ti-super" if suffix == "tisuper" else suffix)
    if len(gpus) >= 2 and f"{base}-dual" in PRESETS:
        return f"{base}-dual"
    return base if base in PRESETS else ""
