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

# NVIDIA ตรวจรับลิงก์ระหว่าง DGX Spark สองเครื่องที่ >=184 Gbit/s ต่อเส้น
# https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
SPARK_LINK_GBPS = 184


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
    """เครื่องนี้พร้อมเป็นสมาชิก stack ไหม — ยังไม่ดูว่าเลือก cluster IP ตัวไหนแล้ว

    ต้องมี "สายที่ยิง NCCL ถึงกันได้จริง" ไม่ใช่แค่ลิงก์เร็ว: `best_gbps` นับพอร์ตที่ลิงก์ขึ้น
    ทุกเส้นรวมเส้นที่ยังไม่ได้ตั้ง IP (169.254.x.x) ด้วย เครื่องที่มีแต่พอร์ตแบบนั้นจึงเคยถูก
    นับเข้ากลุ่มแล้วไปโผล่เป็นเตือน "ยังไม่ได้ตั้ง cluster IP" ทั้งที่ยังจับคู่กับใครไม่ได้เลย
    """
    return bool(host.get("gpus")) and bool(fabric_links(host))


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


def _is_spark(host: dict) -> bool:
    """เครื่องนี้เป็น DGX Spark ไหม — ดูจากชื่อ GPU (GB10) ไม่ใช่ชื่อเครื่องที่ตั้งเองได้"""
    return any("gb10" in (gpu.get("name") or "").lower() for gpu in host.get("gpus") or [])


def link_warning(host: dict, link: dict) -> dict | None:
    """ลิงก์นี้ negotiate ได้ต่ำกว่าที่การ์ดควรทำได้ไหม — None = ไม่มีอะไรต้องเตือน

    `/sys/class/net/*/speed` คือความเร็วที่ **negotiate ได้** ไม่ใช่ความสามารถของการ์ด
    พอร์ต 200G ที่ต่อผ่าน switch แล้ว auto-negotiate ลงมาเหลือ 50G จะรายงาน 50 ซึ่งผ่าน
    MIN_STACK_GBPS ไปได้สบาย — คลัสเตอร์ช้ากว่าที่ควรสี่เท่าโดยไม่มีอะไรตรงไหนบอกเลย

    คืนรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเอง
    """
    if not _is_spark(host) or not link.get("connectx"):
        return None
    speed = link.get("speed_gbps") or 0
    if not 0 < speed < SPARK_LINK_GBPS:
        return None
    return {"kind": "under-negotiated", "speed_gbps": speed, "expected_gbps": SPARK_LINK_GBPS}


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
                "iface": "", "speed_gbps": None, "warning": None}
    links = (host.get("fabric") or {}).get("links") or []
    match = next((link for link in links if link.get("ip") == cluster_ip), None)
    if match is None:
        # ไม่ใช่ error เสมอไป: IP อาจอยู่บนการ์ดที่ตรวจไม่ได้ แต่ต้องเตือนเพราะพิมพ์ผิดก็มาทางนี้
        return {"state": "mismatch", "iface": "", "speed_gbps": None, "warning": None,
                "message": f"{cluster_ip} ไม่ตรงกับการ์ดที่ตรวจพบบนเครื่องนี้ — ตรวจอีกครั้ง"}
    if _is_link_local(cluster_ip):
        return {"state": "link-local", "iface": match["iface"], "warning": None,
                "speed_gbps": match.get("speed_gbps"),
                "message": f"{cluster_ip} เป็น link-local (169.254.x.x) — เส้นนี้ยังไม่ได้ตั้งค่า IP จริง"}
    speed = match.get("speed_gbps") or 0
    if speed < MIN_STACK_GBPS:
        return {"state": "slow", "iface": match["iface"], "speed_gbps": speed, "warning": None,
                "message": f"{cluster_ip} อยู่บน {match['iface']} {speed}G — ช้าเกินไปสำหรับ stacked"}
    # เร็วพอจะ stacked แต่ยังต่ำกว่าที่การ์ดควรทำได้ = ใช้ได้จริง ห้ามตัดออกจากกลุ่ม
    # จึงเป็นคนละช่องกับ state — `usable_world_size` นับจาก state == "ok" เท่านั้น
    return {"state": "ok", "iface": match["iface"], "speed_gbps": speed,
            "warning": link_warning(host, match),
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


def networks_of(host: dict) -> dict[str, dict]:
    """วง → ลิงก์ที่พาไปวงนั้น (เอาเส้นแรกที่เจอต่อหนึ่งวง)"""
    by_network: dict[str, dict] = {}
    for link in fabric_links(host):
        network = link_network(link)
        if network:
            by_network.setdefault(network, link)
    return by_network


def machine_identity(host: dict) -> tuple:
    """ตัวตนของ "เครื่องจริง" — ใช้จับกรณีเครื่องเดียวถูกลงทะเบียนไว้สองชื่อ

    ทะเบียนกันซ้ำได้แค่คู่ host+user ที่พิมพ์เหมือนกันเป๊ะ เครื่องเดิมที่เพิ่มด้วยที่อยู่อีกทาง
    (Tailscale/ชื่อ DNS/IP) จึงเข้ามาเป็นสมาชิกที่สองได้ แล้ว world size ก็บวกเกินไปหนึ่ง

    hostname อย่างเดียวไม่พอเพราะตั้งชื่อซ้ำกันได้ จึงผูกกับชุด IP บนสายเร็วด้วย
    คืน () เมื่อข้อมูลไม่พอจะตัดสิน — ถือว่าเป็นคนละเครื่อง ดีกว่าเดาแล้วยุบเครื่องจริงทิ้ง
    """
    hostname = (host.get("hostname") or "").strip().lower()
    addresses = tuple(sorted(link["ip"] for link in fabric_links(host)))
    if not hostname or not addresses:
        return ()
    return (hostname, addresses)


def shared_fabric(members: list[dict]) -> tuple[str, dict[str, str]]:
    """วงที่ทุกเครื่องในกลุ่มมีขาอยู่ด้วยกัน → (ชื่อวง, {ชื่อเครื่อง: IP ในวงนั้น})

    DGX Spark มี fabric มากกว่าหนึ่งวง (เช่น 10.100.152.0/24 กับ 10.100.153.0/24)
    ถ้าปล่อยให้แต่ละเครื่องเลือกเองอาจได้คนละวง — ต่อกันไม่ติดทั้งที่ทุกอย่างดูถูก
    """
    per_machine: list[dict[str, dict]] = [networks_of(machine["host"]) for machine in members]

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


def drop_duplicate_machines(members: list[dict]) -> tuple[list[dict], list[dict]]:
    """เครื่องเดียวที่ถูกลงทะเบียนหลายชื่อ → เก็บชื่อแรก ที่เหลือคืนออกมาเป็นตัวซ้ำ"""
    seen: dict[tuple, str] = {}
    kept, twins = [], []
    for machine in members:
        identity = machine_identity(machine["host"])
        first = seen.get(identity) if identity else None
        if first is not None:
            twins.append({"name": machine["name"], "reason": "same-machine", "same_as": first})
            continue
        if identity:
            seen[identity] = machine["name"]
        kept.append(machine)
    return kept, twins


def connected_subsets(members: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """**ทุก** กลุ่มย่อยที่มีขาอยู่ในวงเดียวกัน → (รายการกลุ่ม, เครื่องที่ไม่มีวงร่วมกับใครเลย)

    ฮาร์ดแวร์ตรงกันไม่ได้แปลว่าคุยกันได้ — เครื่องที่ไม่มีวงร่วมกับใครเลยต้องไม่ถูกนับเข้า
    world size เพราะมันเปลี่ยนแผน parallel ทั้งกลุ่ม (2 เครื่องใช้ TP=2 ได้ พอนับเป็น 3
    กลายเป็นต้อง pipeline ทั้งที่เครื่องที่สามเข้าร่วมไม่ได้จริง)

    เดิมคืนแค่ **กลุ่มเดียวที่ใหญ่ที่สุด** แล้วโยนที่เหลือทิ้งเป็น "ไม่มี subnet ร่วม" ·
    ฟลีตที่มีเครื่องรุ่นเดียวกันหลายคู่บนคนละวง จึงเห็นได้ทีละคู่เท่านั้น

    เคสจริง 2026-08-31: spark-head + spark-worker ตั้ง cluster IP ครบแล้ว (10.100.152.x
    200G ทั้งคู่) แต่ระบบไปเสนอ msi-1 + msi-2 ที่ยังไม่ได้ตั้ง IP เลย เพราะทั้งสี่เครื่อง
    ฮาร์ดแวร์เหมือนกันจึงอยู่ถังเดียวกัน แล้วสองคู่มีสมาชิกเท่ากัน ตัวตัดสินจึงไปตกที่เลขวง ·
    ผู้ใช้ตั้ง IP แล้วกด Save ก็ไม่มีอะไรเกิดขึ้น เพราะคู่ของเขาถูกทิ้งไปตั้งแต่ขั้นนี้
    """
    by_network: dict[str, list[str]] = {}
    for machine in members:
        for network in networks_of(machine["host"]):
            by_network.setdefault(network, []).append(machine["name"])
    if not by_network:
        return [], list(members)

    # วงที่มีสมาชิกมากที่สุดก่อน — เท่ากันเอาเลขวงน้อยสุดเพื่อให้ผลคงที่ทุกครั้งที่เรียก
    def rank(network: str) -> tuple:
        return (len(by_network[network]), [-int(part) for part in network.split("/")[0].split(".")])

    subsets: list[list[dict]] = []
    placed: set[str] = set()
    for network in sorted(by_network, key=rank, reverse=True):
        names = [n for n in by_network[network] if n not in placed]
        if len(names) < 2:
            continue          # เครื่องเดียวในวง = ยังไม่ใช่กลุ่ม
        placed.update(names)
        subsets.append([m for m in members if m["name"] in names])

    outsiders = [m for m in members if m["name"] not in placed]
    return subsets, outsiders


def connected_subset(members: list[dict]) -> tuple[list[dict], list[dict]]:
    """กลุ่มเดียวที่ใหญ่ที่สุด — เก็บไว้ให้ผู้เรียกเดิมที่ต้องการคำตอบเดียว"""
    subsets, outsiders = connected_subsets(members)
    if not subsets:
        return [], list(members)
    inside = {m["name"] for m in subsets[0]}
    return subsets[0], [m for m in members if m["name"] not in inside]


def cluster_groups(machines: Iterable[dict]) -> list[dict]:
    """จัดกลุ่มเครื่องที่ stacked ด้วยกันได้

    machines: [{"name": str, "host": host payload, "cluster_ip": str, "stack": bool}]
    — รวมเครื่อง hub เองได้ · `stack=False` = ผู้ใช้สั่งไม่ให้จับกลุ่มเครื่องนี้ (ค่าเริ่มต้นคือจับ)
    คืนเฉพาะกลุ่มที่มีสมาชิก >= 2 เพราะกลุ่มเครื่องเดียว stacked ไม่ได้อยู่แล้ว

    เข้ากลุ่มต้องครบสามอย่าง: ฮาร์ดแวร์ตรงกัน · เป็นคนละเครื่องจริง ๆ · มีขาอยู่ในวงเดียวกัน
    เครื่องที่ตกข้อหลังไปอยู่ใน "excluded" — บอกให้เห็นว่าทำไมไม่ถูกนับ ไม่ใช่หายเงียบ
    """
    buckets: dict[tuple, list[dict]] = {}
    for machine in machines:
        host = machine.get("host") or {}
        # ปิดไว้ = ไม่ต้องเสนอเลย แม้ฮาร์ดแวร์จะตรงเป๊ะ — เจตนาของคนต้องชนะการตรวจอัตโนมัติ
        if machine.get("stack") is False:
            continue
        if not stack_ready(host):
            continue
        # ไซต์เป็นส่วนหนึ่งของถัง ไม่ใช่แค่ป้ายแสดงผล — stacked ข้ามไซต์ทำไม่ได้จริง
        # (NCCL ต้องวิ่งบนสายในแร็ค ไม่ใช่ผ่าน WAN/VPN) และการเอามารวมถังเดียวกันทำให้
        # คู่ที่อยู่คนละที่มาแย่งกันเป็น "กลุ่มที่ถูกเลือก" ทั้งที่ไม่มีวันได้ทำงานร่วมกัน
        # ชื่อคลัสเตอร์ที่ตั้งเอง = แบ่งด้วยมือ · ว่าง = ให้ระบบแบ่งเองตาม subnet
        #
        # ระบบแบ่งอัตโนมัติได้เฉพาะตอนที่แต่ละคู่อยู่คนละวง — เครื่องรุ่นเดียวกันสี่เครื่อง
        # บนวงเดียวกันจะถูกมองเป็นก้อนเดียว TP=4 ซึ่งบางทีไม่ใช่สิ่งที่ต้องการ
        # (อยากได้สองคู่แยกกันเพื่อรันคนละโมเดล หรือให้คู่หนึ่งเป็นตัวสำรอง)
        buckets.setdefault(
            (machine.get("site") or "", machine.get("cluster_name") or "", *machine_signature(host)),
            [],
        ).append(machine)

    groups = []
    for key, bucket in buckets.items():
        site, cluster_name, signature = key[0], key[1], key[2:]
        candidates, excluded = drop_duplicate_machines(bucket)
        # ตั้งชื่อไว้เอง = เจตนาของคนชัดแล้ว ไม่ต้องไปแบ่งซ้ำตาม subnet อีก
        # (ยังต้องมีวงร่วมกันจริงอยู่ดี ไม่งั้นขึ้นเป็น blocker ให้เห็น)
        if cluster_name:
            subsets, outsiders = [candidates], []
        else:
            subsets, outsiders = connected_subsets(candidates)
        excluded += [{"name": m["name"], "reason": "no-shared-fabric"} for m in outsiders]
        for members in subsets:
            if len(members) < 2:
                continue
            groups.append(_group_payload(site, cluster_name, signature, members, excluded))
    groups.sort(key=lambda g: (-len(g["members"]), g["gpu"]))
    return groups


def _group_payload(site: str, cluster_name: str, signature: tuple, members: list[dict],
                   excluded: list[dict]) -> dict:
    """สร้างข้อมูลของกลุ่มหนึ่งกลุ่ม — แยกออกมาเพราะตอนนี้หนึ่งถังให้ได้หลายกลุ่ม"""
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

    # เตือน ≠ บล็อก · กลุ่มยังพร้อมและยังนับ world size เต็ม แค่ต้องรู้ว่าวิ่งไม่เต็มสาย
    warnings = []
    under = [d for d in detail if (d.get("warning") or {}).get("kind") == "under-negotiated"]
    if under:
        warnings.append({
            "kind": "under-negotiated",
            "names": [d["name"] for d in under],
            "speed_gbps": min(d["warning"]["speed_gbps"] for d in under),
            "expected_gbps": SPARK_LINK_GBPS,
        })

    addresses = [d["cluster_ip"] for d in detail if d["cluster_ip"]]
    # blockers เป็นรหัส ไม่ใช่ประโยค — CLI (ไทย) กับหน้าเว็บ (อังกฤษ) เรียบเรียงเองคนละภาษา
    blockers = []
    missing = [d["name"] for d in detail if d["state"] == "unset"]
    if missing:
        blockers.append({"kind": "missing-ip", "names": missing})
    # ทะเบียนค้าง: cluster IP ที่ไม่มี interface ไหนบนเครื่องนั้นถืออยู่ (เปลี่ยน IP หลัง apply / แก้มือ / พิมพ์ผิด)
    # เดิมเป็นแค่ state=mismatch บนสมาชิกแล้วกลุ่มยัง "พร้อม" → cluster.env ได้ IP ที่ไม่มีใครถือ → start
    # ค้างที่ NCCL init · doctor เห็นอยู่แล้วแต่ push/wizard ไม่เคยถาม doctor — ต้องหยุดตั้งแต่จับกลุ่ม
    # (สมาชิกทุกตัวมี fabric link จริงถึงเข้ากลุ่มได้ — IP ที่ไม่อยู่ในนั้นจึงเก่าจริง ไม่ใช่การ์ดที่ตรวจไม่เจอ)
    stale = [d["name"] for d in detail if d["state"] in {"mismatch", "link-local"}]
    if stale:
        blockers.append({"kind": "stale-ip", "names": stale})
    # IP อยู่บนสายที่ช้ากว่า MIN_STACK_GBPS (เช่นสายบริหาร 1G) — NCCL วิ่งบนนั้นช้ากว่ารันเครื่องเดียว
    slow = [d["name"] for d in detail if d["state"] == "slow"]
    if slow:
        blockers.append({"kind": "slow-link", "names": slow})
    if len(set(addresses)) != len(addresses):
        blockers.append({"kind": "duplicate-ip", "names": [d["name"] for d in detail]})
    # ตั้งครบแล้วแต่คนละวง = ต่อกันไม่ติด ทั้งที่แต่ละเครื่องดูถูกหมด
    set_networks = {
        link_network({"ip": d["cluster_ip"], "prefix": None}) for d in detail if d["cluster_ip"]
    }
    if len(addresses) == len(detail) and len(set_networks) > 1:
        blockers.append({"kind": "split-fabric", "names": [d["name"] for d in detail]})
    # ตั้งชื่อคลัสเตอร์เองแล้วแต่เครื่องในกลุ่มไม่มีวงร่วมกันเลย = ยิง NCCL ถึงกันไม่ได้
    # ต้องบอก ไม่ใช่ปล่อยให้ไปค้นพบตอน start แล้วค้างที่ NCCL init
    if cluster_name and not shared_fabric(members)[0]:
        blockers.append({"kind": "no-shared-fabric", "names": [d["name"] for d in detail]})

    return {
        "site": site,
        # ว่าง = ระบบแบ่งเองตาม subnet · มีค่า = คนตั้งชื่อไว้เอง
        "cluster_name": cluster_name,
        "members": detail,
        # ฮาร์ดแวร์ตรงกันแต่เข้ากลุ่มไม่ได้ — ไม่นับใน world size และไม่ทำให้กลุ่ม "ไม่พร้อม"
        "excluded": excluded,
        "arch": arch,
        "profile": profile,
        "gpu": gpu,
        "gpus_per_node": gpu_count,
        # ทั้งกลุ่มวิ่งเร็วเท่าเครื่องที่ช้าที่สุด — NCCL รอ rank ที่ช้าที่สุดเสมอ
        "link_gbps": min(speeds),
        "rdma": tiers == {"rdma"},
        "quality": "rdma" if tiers == {"rdma"} else "ethernet",
        "world_size": gpu_count * len(members),
        # เครื่องที่ตั้ง cluster IP ถูกต้องแล้วจริง ๆ — ต่างจาก world size ตอนที่ยังตั้งไม่ครบ
        "usable_world_size": gpu_count * sum(1 for d in detail if d["state"] == "ok"),
        "fabric_network": network,
        "parallelism": parallelism_note(gpu_count * len(members)),
        "blockers": blockers,
        "warnings": warnings,
        "ready": not blockers,
    }


def cluster_note(host: dict) -> str:
    """ข้อความสั้น ๆ สำหรับแสดงต่อท้ายเครื่องหนึ่งเครื่อง"""
    fabric = host.get("fabric") or {}
    if not host.get("gpus"):
        return "ไม่พบ GPU — stacked ไม่ได้"
    best = fabric.get("best_gbps") or 0
    # แยกให้ชัดระหว่าง "สายช้าเกินไป" กับ "สายเร็วพอแต่ยังไม่ได้ตั้ง IP" — คนละวิธีแก้กันคนละเรื่อง
    if best >= MIN_STACK_GBPS and not fabric_links(host):
        return f"มีสาย {best}G แต่ยังไม่ได้ตั้ง IP จริง (ได้แต่ 169.254.x.x) — stacked ไม่ได้จนกว่าจะตั้ง"
    return fabric.get("summary") or "ตรวจสายเชื่อมไม่ได้"
