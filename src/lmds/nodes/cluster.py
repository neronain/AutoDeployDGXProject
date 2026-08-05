"""จับคู่เครื่องที่ stacked ด้วยกันได้ — จาก host payload ของแต่ละเครื่อง

stacked (TP ข้ามเครื่อง) ต่างจาก fleet ตรงที่หลายเครื่องกลายเป็นโมเดลเดียว
NCCL แบ่งงานเท่ากันทุก rank เครื่องจึงต้อง "เหมือนกัน" จริง ๆ ไม่ใช่แค่ต่อถึงกัน:
GPU รุ่นเดียวกัน จำนวนเท่ากัน สถาปัตยกรรมเดียวกัน และมีสายเร็วพอ

โมดูลนี้ไม่ยิง SSH เอง — รับ payload ที่ hub เก็บมาแล้ว เพื่อให้ทั้ง CLI และเว็บ
ตัดสินด้วยกติกาชุดเดียวกัน
"""

from __future__ import annotations

from typing import Iterable

# ต่ำกว่านี้ stacked จะช้ากว่ารันแยกเครื่องจนไม่คุ้ม (activation/KV วิ่งข้ามเครื่องทุก token)
MIN_STACK_GBPS = 25


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

        detail = []
        for machine in members:
            check = check_cluster_ip(machine["host"], machine.get("cluster_ip", ""))
            detail.append({
                "name": machine["name"],
                "cluster_ip": machine.get("cluster_ip", ""),
                "suggested_ip": suggest_cluster_ip(machine["host"]),
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
