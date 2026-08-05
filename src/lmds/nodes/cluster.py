"""จับคู่เครื่องที่ stacked ด้วยกันได้ — จาก host payload ของแต่ละเครื่อง

stacked (TP ข้ามเครื่อง) ต่างจาก fleet ตรงที่หลายเครื่องกลายเป็นโมเดลเดียว
NCCL แบ่งงานเท่ากันทุก rank เครื่องจึงต้อง "เหมือนกัน" จริง ๆ ไม่ใช่แค่ต่อถึงกัน:
GPU รุ่นเดียวกัน จำนวนเท่ากัน สถาปัตยกรรมเดียวกัน และมีสายเร็วพอ

โมดูลนี้ไม่ยิง SSH เอง — รับ payload ที่ hub เก็บมาแล้ว เพื่อให้ทั้ง CLI และเว็บ
ตัดสินด้วยกติกาชุดเดียวกัน
"""

from __future__ import annotations

import ipaddress
from typing import Iterable

# ต่ำกว่านี้ stacked จะช้ากว่ารันแยกเครื่องจนไม่คุ้ม (activation/KV วิ่งข้ามเครื่องทุก token)
MIN_STACK_GBPS = 25

# หมายเหตุ MTU: คู่มือ setup ของ DGX Spark ทุกฉบับสั่งตั้ง mtu 9000 แต่**วัดบนเครื่องจริง
# แล้วไม่ต่างเลย** (2 × DGX Spark GB10, perftest ผ่าน RoCE):
#   netdev 1500 (RoCE MTU 1024) → 111.71 Gb/s · latency 1.98 µs
#   netdev 9000 (RoCE MTU 4096) → 111.71 Gb/s · latency 1.98 µs
# คอขวดคือ PCIe 5.0 x4 ต่อ RoCE device (~112 Gb/s) ไม่ใช่ขนาดเฟรม · จึงจงใจ**ไม่**เตือน
# เรื่อง MTU — คำเตือนที่ไม่มีผลจริงทำให้คำเตือนข้ออื่นถูกมองข้ามไปด้วย

# DGX Spark: ConnectX ใบเดียว สายเส้นเดียว = RoCE คู่แฝดสองตัว (PCIe 5.0 x4 ต่อตัว)
# ต่อสองพอร์ต (mesh 3 เครื่องแบบไม่ใช้สวิตช์) = สี่ตัวขึ้นพร้อมกัน
MESH_ACTIVE_LINKS = 4

# ค่าที่ mesh 3 เครื่องต้องใช้ ต่างจากคลัสเตอร์ผ่านสวิตช์/สองเครื่อง
# ที่มา: eugr/spark-vllm-docker (MIT) docs/NETWORKING.md — ผ่าน NCCL all_gather บน mesh จริง
MESH_NCCL_ENV = {
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_MERGE_NICS": "0",
}


def machine_signature(host: dict) -> tuple:
    """ลายเซ็นฮาร์ดแวร์ที่ต้องตรงกันทุก rank"""
    gpus = host.get("gpus") or []
    return (
        host.get("arch") or "",
        host.get("profile") or "",
        gpus[0].get("name", "") if gpus else "",
        len(gpus),
    )


def fabric_tier(host: dict) -> str:
    return (host.get("fabric") or {}).get("tier") or "unknown"


def stack_ready(host: dict) -> bool:
    """เครื่องนี้พร้อมเป็นสมาชิก stack ไหม — ดูแค่ฮาร์ดแวร์ ยังไม่ดูว่าตั้ง cluster IP หรือยัง"""
    fabric = host.get("fabric") or {}
    best = fabric.get("best_gbps")
    return bool(host.get("gpus")) and best is not None and best >= MIN_STACK_GBPS


def _is_link_local(ip: str) -> bool:
    return ip.startswith("169.254.")


def fabric_links(host: dict) -> list[dict]:
    """ลิงก์ที่เสนอให้เลือกเป็น cluster interface — เร็วพอ มี IP และไม่ใช่ link-local

    เครื่องจริง (DGX Spark) มีพอร์ต ConnectX หลายเส้น เส้นที่ยังไม่ได้ตั้งค่าจะได้ 169.254.x.x
    มาเอง ลิงก์ขึ้นและเร็ว 200G เหมือนกัน แต่ยิง NCCL ข้ามเครื่องไม่ถึง — ห้ามเสนอ
    """
    # ตัดสินจาก IP เอง ไม่พึ่งแฟล็ก link_local ของ node — node เวอร์ชันเก่ายังไม่ส่งฟิลด์นี้มา
    links = (host.get("fabric") or {}).get("links") or []
    return [
        link for link in links
        if link.get("ip") and not _is_link_local(link["ip"])
        and (link.get("speed_gbps") or 0) >= MIN_STACK_GBPS
    ]


def active_fabric_links(host: dict) -> list[dict]:
    """ลิงก์ ConnectX ที่ **ลิงก์ขึ้นจริง** — ไม่สนว่าตั้ง IP แล้วหรือยัง

    ต่างจาก fabric_links() ที่กรองเอาเฉพาะเส้นที่พร้อมใช้ · ตรงนี้ต้องเห็นเส้นที่ขึ้นแต่
    ยังไม่ได้ตั้งค่าด้วย เพราะนั่นคือ "อาการ" ที่ต้องเตือน ไม่ใช่สิ่งที่ควรกรองทิ้งเงียบ ๆ
    """
    links = (host.get("fabric") or {}).get("links") or []
    return [
        link for link in links
        if link.get("connectx") and link.get("state") == "up"
        and (link.get("speed_gbps") or 0) >= MIN_STACK_GBPS
    ]


def is_mesh(host: dict) -> bool:
    """เครื่องนี้เดินสายแบบ mesh (ต่อสองพอร์ต) หรือแบบปกติ (พอร์ตเดียว)

    ConnectX ของ DGX Spark: หนึ่งพอร์ต QSFP = RoCE คู่แฝดสองตัว เพราะ SoC ให้ PCIe 5.0
    ได้แค่ x4 ต่อ device จึงต้องใช้สอง device ต่อสายหนึ่งเส้นถึงจะได้ 200G
    ขึ้นครบสี่ = ต่อสองพอร์ต = mesh 3 เครื่องแบบไม่ใช้สวิตช์
    """
    return len(active_fabric_links(host)) >= MESH_ACTIVE_LINKS


def nccl_ib_hca(host: dict) -> str:
    """ค่าที่ต้องใส่ให้ NCCL_IB_HCA — RoCE ทุกตัวที่ลิงก์ขึ้น คั่นด้วยจุลภาค

    บอก NCCL แค่ตัวเดียวเท่ากับใช้สายเส้นเดียวจากสองเส้นที่ต่ออยู่ = ได้แบนด์วิดท์ครึ่งเดียว
    โดยไม่มีอะไรฟ้อง เพราะงานก็ยังรันได้ (controller ตรวจเองตอน start อยู่แล้ว —
    ตรงนี้ทำให้ hub บอกล่วงหน้าได้โดยไม่ต้องรอถึงตอนรัน)
    """
    devices = [link.get("rdma_device") or "" for link in active_fabric_links(host)]
    return ",".join(dict.fromkeys(d for d in devices if d))


def oob_link(host: dict) -> dict | None:
    """สายที่ใช้คุยกันนอกเหนือจาก RoCE (out-of-band) — mesh บังคับว่าต้องไม่ใช่ QSFP

    mesh 3 เครื่องต่อกันเป็นวงแหวน แต่ละคู่เห็นกันตรง ๆ แค่คู่ที่มีสายถึงกัน · NCCL/Ray
    ต้องมีเส้นที่ **ทุกเครื่องเห็นกันหมด** ไว้คุยกันตอน bootstrap ซึ่งคือพอร์ต RJ-45 10G
    (หรือ wifi ถ้าไม่มีจริง ๆ) ไม่ใช่ QSFP
    """
    links = (host.get("fabric") or {}).get("links") or []
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
    #    ที่มา: eugr/spark-vllm-docker docs/NETWORKING.md ("DO NOT use the same subnet on both twins")
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

    # 4. mesh ต้องมีเส้น out-of-band ที่ทุกเครื่องเห็นกัน
    if is_mesh(host):
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
    state: ok / unset / mismatch / slow / link-local
    """
    if not cluster_ip:
        suggestion = suggest_cluster_ip(host)
        hint = f" — เสนอ {suggestion}" if suggestion else ""
        return {"state": "unset", "message": f"ยังไม่ได้ตั้ง cluster IP{hint}",
                "iface": "", "speed_gbps": None}
    links = (host.get("fabric") or {}).get("links") or []
    match = next((link for link in links if link.get("ip") == cluster_ip), None)
    if match is None:
        # ไม่ใช่ error เสมอไป: IP อาจอยู่บนการ์ดที่ตรวจไม่ได้ แต่ต้องเตือนเพราะพิมพ์ผิดก็มาทางนี้
        return {"state": "mismatch", "iface": "", "speed_gbps": None,
                "message": f"{cluster_ip} ไม่ตรงกับการ์ดที่ตรวจพบบนเครื่องนี้ — ตรวจอีกครั้ง"}
    if _is_link_local(cluster_ip):
        return {"state": "link-local", "iface": match["iface"],
                "speed_gbps": match.get("speed_gbps"),
                "message": f"{cluster_ip} เป็น link-local (169.254.x.x) — เส้นนี้ยังไม่ได้ตั้งค่า IP จริง"}
    speed = match.get("speed_gbps") or 0
    if speed < MIN_STACK_GBPS:
        return {"state": "slow", "iface": match["iface"], "speed_gbps": speed,
                "message": f"{cluster_ip} อยู่บน {match['iface']} {speed}G — ช้าเกินไปสำหรับ stacked"}
    return {"state": "ok", "iface": match["iface"], "speed_gbps": speed,
            "message": f"{cluster_ip} บน {match['iface']} {speed}G"}


def link_network(link: dict) -> str:
    """วงของลิงก์นี้ เช่น 10.100.152.0/24 — ว่างเมื่อคำนวณไม่ได้

    node เวอร์ชันเก่ายังไม่ส่ง prefix มา จึงเดาเป็น /24 ซึ่งตรงกับ fabric ของ DGX Spark
    """
    ip = link.get("ip") or ""
    if not ip:
        return ""
    prefix = link.get("prefix") or 24
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



# vLLM ต้องแบ่ง attention head ให้ทุก rank เท่ากัน — TP ที่หาร head ไม่ลงตัวจะถูกปฏิเสธ
# ตั้งแต่ start โมเดลส่วนใหญ่มี head เป็นเลขยกกำลังสอง (Llama 3.3 70B = 64) จำนวนเครื่อง
# ที่เป็นเลขยกกำลังสองจึงใช้ tensor-parallel ได้ตรง ๆ ส่วนจำนวนอื่นต้องใช้ pipeline แทน
def tensor_parallel_fits(world_size: int) -> bool:
    return world_size > 0 and (world_size & (world_size - 1)) == 0


def parallelism_note(world_size: int) -> dict:
    """บอกว่าขนาดคลัสเตอร์นี้ใช้ TP ได้ไหม และถ้าไม่ได้ควรทำอย่างไร

    คืนรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเอง
    """
    if tensor_parallel_fits(world_size):
        return {"kind": "tensor-parallel", "world_size": world_size, "usable": True}
    # 3, 5, 6, 7 … : TP หารไม่ลง ต้อง pipeline (ช้ากว่าเพราะ token ไหลเป็นทอด)
    return {"kind": "pipeline-parallel", "world_size": world_size, "usable": True,
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
        speeds = [(m["host"].get("fabric") or {}).get("best_gbps") or 0 for m in members]

        # เสนอ IP จากวงที่ทุกเครื่องมีขาร่วมกัน ไม่ใช่ให้แต่ละเครื่องเลือกเองอิสระ
        network, shared_ips = shared_fabric(members)
        detail = []
        for machine in members:
            check = check_cluster_ip(machine["host"], machine.get("cluster_ip", ""))
            detail.append({
                "name": machine["name"],
                "cluster_ip": machine.get("cluster_ip", ""),
                "suggested_ip": shared_ips.get(machine["name"]) or suggest_cluster_ip(machine["host"]),
                **check,
            })

        addresses = [d["cluster_ip"] for d in detail if d["cluster_ip"]]
        # blockers เป็นรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเองคนละภาษา
        blockers = []
        missing = [d["name"] for d in detail if d["state"] == "unset"]
        if missing:
            blockers.append({"kind": "missing-ip", "names": missing})
        if len(set(addresses)) != len(addresses):
            blockers.append({"kind": "duplicate-ip", "names": [d["name"] for d in detail]})
        # ตั้งครบแล้วแต่คนละวง = ต่อกันไม่ติด ทั้งที่แต่ละเครื่องดูถูกหมด
        set_networks = {
            link_network({"ip": d["cluster_ip"], "prefix": None}) for d in detail if d["cluster_ip"]
        }
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
    fabric = host.get("fabric") or {}
    if not host.get("gpus"):
        return "ไม่พบ GPU — stacked ไม่ได้"
    return fabric.get("summary") or "ตรวจสายเชื่อมไม่ได้"
