"""จับคู่เครื่องที่ stacked ด้วยกันได้ — จาก host payload ของแต่ละเครื่อง

stacked (TP ข้ามเครื่อง) ต่างจาก fleet ตรงที่หลายเครื่องกลายเป็นโมเดลเดียว
NCCL แบ่งงานเท่ากันทุก rank เครื่องจึงต้อง "เหมือนกัน" จริง ๆ ไม่ใช่แค่ต่อถึงกัน:
GPU รุ่นเดียวกัน จำนวนเท่ากัน สถาปัตยกรรมเดียวกัน และมีสายเร็วพอ

โมดูลนี้ไม่ยิง SSH เอง — รับ payload ที่ hub เก็บมาแล้ว เพื่อให้ทั้ง CLI และเว็บ
ตัดสินด้วยกติกาชุดเดียวกัน
"""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from typing import Iterable

# LMDS product-policy threshold for suggesting a stack, not an NCCL protocol limit or a measured
# performance boundary.  The UI still reports slower links; it just does not mark them ready.
MIN_STACK_GBPS = 25

# จำนวน netdev เป็นเพียงสัญญาณว่าเครื่อง *อาจ* ต่อสองพอร์ตอยู่ ไม่ใช่หลักฐาน topology:
# สี่ลิงก์อาจต่อผ่านสวิตช์หรือไปคนละปลายทาง จึงห้ามใช้ค่านี้ฉีด NCCL env อัตโนมัติ.
MESH_ACTIVE_LINKS = 4

_SAFE_IFACE = re.compile(r"^[A-Za-z0-9_.:@-]{1,64}$")
_SAFE_DEVICE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_LINKS = 64


def _text(value: object, limit: int = 200) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _number(value: object, *, low: int = 0, high: int = 1_000_000) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = int(value)
    except (OverflowError, ValueError):
        return None
    return number if value == number and low <= number <= high else None


def _ipv4(value: object) -> str:
    text = _text(value, 64)
    if not text:
        return ""
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return ""
    if parsed.version != 4 or parsed.is_loopback or parsed.is_multicast or parsed.is_unspecified:
        return ""
    return str(parsed)


def _links(host: object) -> list[dict]:
    """Normalize untrusted/stale agent JSON before cluster logic or presentation consumes it."""
    if not isinstance(host, dict):
        return []
    fabric = host.get("fabric")
    raw_links = fabric.get("links") if isinstance(fabric, dict) else None
    if not isinstance(raw_links, list):
        return []
    links = []
    for raw in raw_links[:_MAX_LINKS]:
        if not isinstance(raw, dict):
            continue
        iface = _text(raw.get("iface"), 64)
        if iface == "lo" or not _SAFE_IFACE.fullmatch(iface):
            continue
        ip = _ipv4(raw.get("ip"))
        prefix = _number(raw.get("prefix"), low=0, high=32) if ip else None
        device = _text(raw.get("rdma_device"), 64)
        if device and not _SAFE_DEVICE.fullmatch(device):
            device = ""
        state = _text(raw.get("state"), 16)
        if state not in {"up", "down", "unknown"}:
            state = "unknown"
        links.append({
            "iface": iface,
            "ip": ip,
            "prefix": prefix,
            "link_local": bool(ip and ipaddress.ip_address(ip).is_link_local),
            "speed_gbps": _number(raw.get("speed_gbps")),
            "rdma_device": device,
            "driver": _text(raw.get("driver"), 64),
            "state": state,
            "connectx": raw.get("connectx") is True,
            "rdma": raw.get("rdma") is True,
        })
    return links


def machine_signature(host: dict) -> tuple:
    """ลายเซ็นฮาร์ดแวร์ที่ต้องตรงกันทุก rank"""
    host = host if isinstance(host, dict) else {}
    gpus = host.get("gpus")
    gpus = [gpu for gpu in gpus if isinstance(gpu, dict)] if isinstance(gpus, list) else []
    first = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
    return (
        _text(host.get("arch"), 64),
        _text(host.get("profile"), 64),
        _text(first.get("name"), 200),
        len(gpus),
    )


def fabric_tier(host: dict) -> str:
    active = _active_links(host)
    fastest = best_link_speed(host)
    best = [link for link in active if link["speed_gbps"] == fastest]
    return "rdma" if best and all(link["rdma"] for link in best) else "ethernet"


def _active_links(host: object) -> list[dict]:
    """All physical/virtual links that are up and report a positive speed, after normalization."""
    return [
        link for link in _links(host)
        if link.get("state") == "up" and (link.get("speed_gbps") or 0) > 0
    ]


def best_link_speed(host: object) -> int:
    """Fastest confirmed-up link in the normalized node payload (zero means unknown)."""
    return max((link["speed_gbps"] for link in _active_links(host)), default=0)


def stack_ready(host: dict) -> bool:
    """เครื่องนี้พร้อมเป็นสมาชิก stack ไหม — ดูแค่ฮาร์ดแวร์ ยังไม่ดูว่าตั้ง cluster IP หรือยัง"""
    host = host if isinstance(host, dict) else {}
    gpus = host.get("gpus")
    valid_gpus = [gpu for gpu in gpus if isinstance(gpu, dict)] if isinstance(gpus, list) else []
    return bool(valid_gpus) and best_link_speed(host) >= MIN_STACK_GBPS


def _is_link_local(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_link_local
    except (TypeError, ValueError):
        return False


def fabric_links(host: dict) -> list[dict]:
    """ลิงก์ที่เสนอให้เลือกเป็น cluster interface — เร็วพอ มี IP และไม่ใช่ link-local

    เครื่องจริง (DGX Spark) มีพอร์ต ConnectX หลายเส้น เส้นที่ยังไม่ได้ตั้งค่าจะได้ 169.254.x.x
    มาเอง ลิงก์ขึ้นและเร็ว 200G เหมือนกัน แต่ยิง NCCL ข้ามเครื่องไม่ถึง — ห้ามเสนอ
    """
    # ตัดสินจาก IP เอง ไม่พึ่งแฟล็ก link_local ของ node — node เวอร์ชันเก่ายังไม่ส่งฟิลด์นี้มา
    links = _links(host)
    return [
        link for link in links
        if link.get("state") == "up" and link.get("ip") and not _is_link_local(link["ip"])
        and (link.get("speed_gbps") or 0) >= MIN_STACK_GBPS
    ]


def active_fabric_links(host: dict) -> list[dict]:
    """ลิงก์ ConnectX ที่ **ลิงก์ขึ้นจริง** — ไม่สนว่าตั้ง IP แล้วหรือยัง

    ต่างจาก fabric_links() ที่กรองเอาเฉพาะเส้นที่พร้อมใช้ · ตรงนี้ต้องเห็นเส้นที่ขึ้นแต่
    ยังไม่ได้ตั้งค่าด้วย เพราะนั่นคือ "อาการ" ที่ต้องเตือน ไม่ใช่สิ่งที่ควรกรองทิ้งเงียบ ๆ
    """
    links = _active_links(host)
    return [
        link for link in links
        if link.get("connectx")
    ]


def is_mesh(host: dict) -> bool:
    """Local signal that this host may be dual-port/mesh; never a topology certificate."""
    return len(active_fabric_links(host)) >= MESH_ACTIVE_LINKS


def nccl_ib_hca(host: dict) -> str:
    """ค่าที่ต้องใส่ให้ NCCL_IB_HCA — RoCE ทุกตัวที่ลิงก์ขึ้น คั่นด้วยจุลภาค

    บอก NCCL แค่ตัวเดียวเท่ากับใช้สายเส้นเดียวจากสองเส้นที่ต่ออยู่ = ได้แบนด์วิดท์ครึ่งเดียว
    โดยไม่มีอะไรฟ้อง เพราะงานก็ยังรันได้ (controller ตรวจเองตอน start อยู่แล้ว —
    ตรงนี้ทำให้ hub บอกล่วงหน้าได้โดยไม่ต้องรอถึงตอนรัน)
    """
    active = [link for link in active_fabric_links(host) if link.get("rdma_device")]
    if not active:
        return ""
    fastest = max(link.get("speed_gbps") or 0 for link in active)
    devices = [link["rdma_device"] for link in active if (link.get("speed_gbps") or 0) == fastest]
    return "=" + ",".join(dict.fromkeys(devices))


def oob_link(host: dict) -> dict | None:
    """Local non-ConnectX candidate for management/bootstrap traffic.

    A local interface cannot prove that every peer can reach it; callers must present this as a
    candidate/warning, never as an end-to-end topology certificate.
    """
    links = _links(host)
    candidates = [
        link for link in links
        if not link.get("connectx") and link.get("state") == "up"
        and link.get("ip") and not _is_link_local(link["ip"])
    ]
    if not candidates:
        return None
    # มีสายจริงชนะ wifi เสมอ แล้วค่อยเรียงตามความเร็ว
    return max(candidates, key=lambda link: (not _is_wireless(link), link.get("speed_gbps") or 0))


def _is_wireless(link: dict) -> bool:
    name = link.get("iface") or ""
    return name.startswith(("wl", "wlan", "wlp", "wlP"))


def fabric_warnings(host: dict) -> list[dict]:
    """ปัญหาการเดินสาย/ตั้งค่าที่ทำให้ stacked ช้าลงหรือต่อไม่ติด **โดยไม่มีอะไรฟ้อง**

    ทุกข้อคือเคสที่ "ก็รันได้" จึงไม่มีใครไปไล่หา — เป็นกลุ่มที่ไล่สาเหตุยากที่สุด
    คืนรหัส ไม่ใช่ประโยค · CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเอง
    """
    warnings: list[dict] = []
    active = active_fabric_links(host)

    # 1. ลิงก์ขึ้นแล้วแต่ไม่มี IP — NCCL ใช้เส้นนี้ไม่ได้เลย ทั้งที่สายเสียบอยู่
    no_ip = [link["iface"] for link in active if not link.get("ip")]
    if no_ip:
        warnings.append({"kind": "link-without-ip", "ifaces": no_ip})

    # 2. สองเส้นอยู่วงเดียวกัน — routing สับสน แพ็กเก็ตออกผิดเส้น
    #    ที่มา: eugr/spark-vllm-docker@42b3a793 docs/NETWORKING.md
    by_network: dict[str, list[str]] = {}
    for link in active:
        network = link_network(link)
        if network:
            by_network.setdefault(network, []).append(link["iface"])
    duplicates = {net: ifaces for net, ifaces in by_network.items() if len(ifaces) > 1}
    if duplicates:
        warnings.append({
            "kind": "shared-subnet",
            "networks": sorted(duplicates),
            "ifaces": sorted(i for ifaces in duplicates.values() for i in ifaces),
        })

    # 3. ลิงก์ขึ้นแต่ไม่มี RoCE device คู่กัน — NCCL_IB_HCA ตั้งไม่ได้ → ตกไปใช้ TCP
    no_hca = [link["iface"] for link in active if not link.get("rdma_device")]
    if no_hca:
        warnings.append({"kind": "no-rdma-device", "ifaces": no_hca})

    # 4. สี่ลิงก์ local บอกได้แค่ว่าเป็น mesh candidate; topology/OOB reachability ต้องตรวจข้ามเครื่อง
    if is_mesh(host):
        warnings.append({"kind": "mesh-topology-unverified"})
        oob = oob_link(host)
        if oob is None:
            warnings.append({"kind": "mesh-without-oob"})
        elif _is_wireless(oob):
            warnings.append({"kind": "mesh-oob-wireless", "ifaces": [oob["iface"]]})
    return warnings


def suggest_cluster_ip(host: dict) -> str:
    """IP ที่ควรใช้เป็น cluster IP — เส้นที่เร็วที่สุดที่มี IP อยู่แล้ว (ว่าง = ต้องกรอกเอง)"""
    links = fabric_links(host)
    if not links:
        return ""
    return max(links, key=lambda link: link.get("speed_gbps") or 0).get("ip", "")


def check_cluster_ip(host: dict, cluster_ip: str) -> dict:
    """ตรวจว่า cluster IP ที่ตั้งไว้ตรงกับการ์ดจริงไหม

    คืนทั้ง state (ให้หน้าเว็บเรียบเรียงข้อความเองเป็นอังกฤษ) และ message ภาษาไทยสำหรับ CLI
    state: ok / unset / mismatch / down / slow / link-local
    """
    if not cluster_ip:
        suggestion = suggest_cluster_ip(host)
        hint = f" — เสนอ {suggestion}" if suggestion else ""
        return {"state": "unset", "message": f"ยังไม่ได้ตั้ง cluster IP{hint}",
                "iface": "", "speed_gbps": None}
    links = _links(host)
    match = next((link for link in links if link.get("ip") == cluster_ip), None)
    if match is None:
        # ไม่ใช่ error เสมอไป: IP อาจอยู่บนการ์ดที่ตรวจไม่ได้ แต่ต้องเตือนเพราะพิมพ์ผิดก็มาทางนี้
        return {"state": "mismatch", "iface": "", "speed_gbps": None,
                "message": f"{cluster_ip} ไม่ตรงกับการ์ดที่ตรวจพบบนเครื่องนี้ — ตรวจอีกครั้ง"}
    if match.get("state") != "up":
        return {"state": "down", "iface": match["iface"],
                "speed_gbps": match.get("speed_gbps"),
                "message": f"{cluster_ip} อยู่บน {match['iface']} แต่ลิงก์ไม่ได้ขึ้น — ตรวจสาย/route"}
    if _is_link_local(cluster_ip):
        return {"state": "link-local", "iface": match["iface"],
                "speed_gbps": match.get("speed_gbps"),
                "message": f"{cluster_ip} เป็น link-local (169.254.x.x) — เส้นนี้ยังไม่ได้ตั้งค่า IP จริง"}
    speed = match.get("speed_gbps") or 0
    if speed < MIN_STACK_GBPS:
        return {"state": "slow", "iface": match["iface"], "speed_gbps": speed,
                "message": f"{cluster_ip} อยู่บน {match['iface']} {speed}G — ต่ำกว่าเกณฑ์แนะนำของ LMDS {MIN_STACK_GBPS}G"}
    return {"state": "ok", "iface": match["iface"], "speed_gbps": speed,
            "message": f"{cluster_ip} บน {match['iface']} {speed}G"}


def link_network(link: dict) -> str:
    """วงของลิงก์นี้ เช่น 10.100.152.0/24 — ว่างเมื่อ IP/prefix ยืนยันไม่ได้."""
    ip = _ipv4(link.get("ip")) if isinstance(link, dict) else ""
    prefix = _number(link.get("prefix"), low=0, high=32) if isinstance(link, dict) else None
    if not ip or prefix is None:
        return ""
    try:
        return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))
    except ValueError:
        return ""


def shared_fabric(members: list[dict]) -> tuple[str, dict[str, str]]:
    """วงที่ทุกเครื่องในกลุ่มมีขาอยู่ด้วยกัน → (ชื่อวง, {ชื่อเครื่อง: IP ในวงนั้น})

    DGX Spark มี fabric มากกว่าหนึ่งวง (เช่น 10.100.152.0/24 กับ 10.100.153.0/24)
    ถ้าปล่อยให้แต่ละเครื่องเลือกเองอาจได้คนละวง — ต่อกันไม่ติดทั้งที่ทุกอย่างดูถูก
    """
    per_machine: list[dict[str, dict]] = []
    for machine in members:
        by_network = {}
        for link in fabric_links(machine["host"]):
            network = link_network(link)
            if network:
                by_network.setdefault(network, link)
        per_machine.append(by_network)

    if not per_machine:
        return "", {}
    common = set(per_machine[0])
    for by_network in per_machine[1:]:
        common &= set(by_network)
    if not common:
        return "", {}

    # วงไหนก็ได้ที่เร็วที่สุด — เท่ากันหมดก็เอาเลขน้อยสุดเพื่อให้ผลคงที่ทุกครั้งที่เรียก
    def rank(network: str) -> tuple:
        speeds = [by_network[network].get("speed_gbps") or 0 for by_network in per_machine]
        return (min(speeds), [-int(part) for part in network.split("/")[0].split(".")])

    chosen = max(common, key=rank)
    return chosen, {
        machine["name"]: per_machine[index][chosen]["ip"]
        for index, machine in enumerate(members)
    }



# การหาร attention heads เป็นคุณสมบัติของโมเดล ไม่ใช่ topology. Power-of-two เป็นเพียง heuristic;
# หน้าจอ cluster ยังไม่รู้ model config จึงห้ามอ้างว่า non-power-of-two รองรับ pipeline.
def tensor_parallel_fits(world_size: int) -> bool:
    return world_size > 0 and (world_size & (world_size - 1)) == 0


def parallelism_note(world_size: int) -> dict:
    """บอกว่าขนาดคลัสเตอร์นี้ใช้ TP ได้ไหม และถ้าไม่ได้ควรทำอย่างไร

    คืนรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเอง
    """
    if tensor_parallel_fits(world_size):
        return {"kind": "tensor-parallel", "world_size": world_size, "usable": True}
    return {"kind": "model-dependent", "world_size": world_size, "usable": True,
            "largest_tp": 1 << (world_size.bit_length() - 1)}


def cluster_groups(machines: Iterable[dict]) -> list[dict]:
    """จัดกลุ่มเครื่องที่ stacked ด้วยกันได้

    machines: [{"name": str, "host": host payload, "cluster_ip": str}] — รวมเครื่อง hub เองได้
    คืนเฉพาะกลุ่มที่มีสมาชิก >= 2 เพราะกลุ่มเครื่องเดียว stacked ไม่ได้อยู่แล้ว
    """
    buckets: dict[tuple, list[dict]] = {}
    for machine in machines:
        host = machine.get("host") or {}
        if not stack_ready(host):
            continue
        buckets.setdefault(machine_signature(host), []).append(machine)

    groups = []
    for signature, members in buckets.items():
        if len(members) < 2:
            continue
        arch, profile, gpu, gpu_count = signature
        tiers = {fabric_tier(m["host"]) for m in members}
        speeds = [best_link_speed(m["host"]) for m in members]

        # เสนอ IP จากวงที่ทุกเครื่องมีขาร่วมกัน ไม่ใช่ให้แต่ละเครื่องเลือกเองอิสระ
        network, shared_ips = shared_fabric(members)
        detail = []
        for machine in members:
            check = check_cluster_ip(machine["host"], machine.get("cluster_ip", ""))
            detail.append({
                "name": machine["name"],
                "cluster_ip": machine.get("cluster_ip", ""),
                "suggested_ip": shared_ips.get(machine["name"]) or suggest_cluster_ip(machine["host"]),
                "network": next((link_network(link) for link in _links(machine["host"])
                                 if link["ip"] == machine.get("cluster_ip", "")), ""),
                **check,
            })

        addresses = [d["cluster_ip"] for d in detail if d["cluster_ip"]]
        # blockers เป็นรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเองคนละภาษา
        blockers = []
        missing = [d["name"] for d in detail if d["state"] == "unset"]
        if missing:
            blockers.append({"kind": "missing-ip", "names": missing})
        duplicate_addresses = {address for address, count in Counter(addresses).items() if count > 1}
        if duplicate_addresses:
            blockers.append({"kind": "duplicate-ip", "names": [
                d["name"] for d in detail if d["cluster_ip"] in duplicate_addresses
            ]})
        invalid_ip = [d["name"] for d in detail if d["state"] not in {"ok", "unset"}]
        if invalid_ip:
            blockers.append({"kind": "invalid-ip", "names": invalid_ip})
        unknown_network = [
            d["name"] for d in detail
            if d["state"] == "ok" and d["cluster_ip"] and not d["network"]
        ]
        if unknown_network:
            blockers.append({"kind": "unknown-prefix", "names": unknown_network})
        # ตั้งครบแล้วแต่คนละวง = ต่อกันไม่ติด ทั้งที่แต่ละเครื่องดูถูกหมด
        set_networks = {d["network"] for d in detail if d["cluster_ip"] and d["network"]}
        if len(addresses) == len(detail) and len(set_networks) > 1:
            blockers.append({"kind": "split-fabric", "names": [d["name"] for d in detail]})

        groups.append({
            "members": detail,
            "arch": arch,
            "profile": profile,
            "gpu": gpu,
            "gpus_per_node": gpu_count,
            # ทั้งกลุ่มวิ่งเร็วเท่าเครื่องที่ช้าที่สุด — NCCL รอ rank ที่ช้าที่สุดเสมอ
            "link_gbps": min(speeds),
            "rdma": tiers == {"rdma"},
            "quality": "rdma" if tiers == {"rdma"} else "ethernet",
            "world_size": gpu_count * len(members),
            "fabric_network": network,
            "parallelism": parallelism_note(gpu_count * len(members)),
            "blockers": blockers,
            "ready": not blockers,
        })
    groups.sort(key=lambda g: (-len(g["members"]), g["gpu"]))
    return groups


def cluster_note(host: dict) -> str:
    """ข้อความสั้น ๆ สำหรับแสดงต่อท้ายเครื่องหนึ่งเครื่อง"""
    host = host if isinstance(host, dict) else {}
    fabric = host.get("fabric") if isinstance(host.get("fabric"), dict) else {}
    if not isinstance(host.get("gpus"), list) or not host["gpus"]:
        return "ไม่พบ GPU — stacked ไม่ได้"
    return _text(fabric.get("summary"), 300) or "ตรวจสายเชื่อมไม่ได้"
