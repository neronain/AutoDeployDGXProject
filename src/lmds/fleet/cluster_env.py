"""เขียน cluster.env ลง bundle — ค่าที่ controller ต้องรู้ก่อน start แบบ stacked

แยกออกมาจาก CLI เพราะหน้าเว็บต้องเขียนไฟล์นี้ด้วย · เดิมตรรกะทั้งก้อนอยู่ใน
`lmds/cli/main.py` แล้วเรียก `typer.Exit` กับ `err_console` ตรง ๆ ซึ่งเรียกจาก
FastAPI ไม่ได้ — หน้าเว็บจึงทำได้แค่ *พิมพ์คำสั่ง CLI ให้ผู้ใช้ไปก็อป* ทั้งที่ทุกอย่าง
ที่ต้องใช้อยู่ในมือ server แล้ว

ที่นี่ไม่พิมพ์อะไรและไม่ออกจากโปรเซส — ล้มก็โยน ClusterEnvError ให้ผู้เรียกตัดสินใจเอง
"""

from __future__ import annotations

import base64
import ipaddress
import re
import shlex
from dataclasses import dataclass

# ── cluster.env schema v2 (multi-link) ──
#
# DGX Spark มี QSFP สองพอร์ต · 3 เครื่องต่อสายตรงถึงกันเป็นวงแหวน (A.p1→B.p2 · B.p1→C.p2 · C.p1→A.p2)
# = ทุกเครื่องมี 2 สาย 2 วง และ head ไปถึง worker แต่ละตัวด้วย **คนละ interface/IP** · schema เดิม
# (MASTER_IP/WORKER_IPS/TRANSPORT_IP_*/NCCL_SOCKET_IFNAME ตัวเดียว) บอกเรื่องนี้ไม่ได้เลย → NCCL ได้
# interface เดียวแล้วหาทางไป worker อีกตัวไม่เจอ · v2 เพิ่มคีย์ต่อ rank โดยยังเขียนคีย์เดิมครบ
# (controller เก่าอ่านได้เหมือนเดิม · controller ใหม่ที่ไม่เห็นคีย์ v2 ก็ทำงานแบบเดิม)
#
# คีย์ v2 (rank 0 = head · rank 1..N-1 = worker เรียงตาม node-rank):
#   CLUSTER_ENV_SCHEMA=2
#   CLUSTER_TOPOLOGY=direct-2 | ring-3 | switch-N
#   CLUSTER_NODES="<ชื่อ rank0> <ชื่อ rank1> …"
#   LINKS_<rank>="iface:ip/prefix:peer_rank:peer_ip …"   (peer_rank `*` + peer_ip `-` = สายเข้า switch)
#   NCCL_SOCKET_IFNAMES_<rank>=iface1,iface2               (comma list — NCCL/GLOO รับได้ตรง ๆ)
#   NCCL_IB_HCAS_<rank>=hca1,hca2                          (เฉพาะที่ทะเบียนรู้ · ว่าง = controller หาจาก sysfs)
#   HEAD_TO_WORKER_IP_<rank>=<IP ของ worker rank นี้ที่ head ใช้ถึง — ssh · rsync · VLLM_HOST_IP ของ worker>
#   WORKER_HEAD_IP_<rank>=<IP ของ head ที่ worker rank นี้ใช้ต่อกลับ — --master-addr ของ worker>
#   NCCL_CROSS_NIC=1                                       (ring เท่านั้น — คู่ต่างกันใช้ NIC ต่างกันได้)
LINK_SWITCH_PEER = "*"


class ClusterEnvError(Exception):
    """ปัญหาที่ผู้ใช้แก้ได้ — ข้อความพร้อมแสดงทั้งบน CLI และหน้าเว็บ"""


@dataclass
class ClusterEnvResult:
    target: str            # path ที่เขียน (หรือ "<node>:<path>" ถ้าเขียนข้ามเครื่อง)
    head_ip: str
    worker_ips: list[str]
    nnodes: int
    iface: str
    body: str


def build_cluster_env(groups, head_name: str, worker_name: str | None = None):
    """เลือกกลุ่ม/head/worker แล้วคืนเนื้อไฟล์ — ไม่แตะดิสก์ ทดสอบได้ตรง ๆ"""
    from lmds.nodes import find as find_node

    # hub ไม่จำเป็นต้องเป็นสมาชิกของคลัสเตอร์ — เครื่องที่คุมอาจเป็นโน้ตบุ๊กที่ไม่มี GPU
    ready = [g for g in groups if g["ready"] and any(m["name"] == head_name for m in g["members"])]
    if not ready:
        names = sorted({m["name"] for g in groups if g["ready"] for m in g["members"]})
        raise ClusterEnvError(
            f"ไม่มีกลุ่มที่พร้อมและมี '{head_name}' เป็นสมาชิก — "
            + (f"เลือก head ได้จาก: {', '.join(names)}" if names
               else "ต้องตั้ง cluster IP ให้ครบก่อน")
        )
    group = ready[0]

    others = [m for m in group["members"] if m["name"] != head_name]
    if worker_name:
        chosen = next((m for m in others if m["name"] == worker_name), None)
        if chosen is None:
            raise ClusterEnvError(f"'{worker_name}' ไม่ได้อยู่ในกลุ่มที่พร้อมของ '{head_name}'")
        workers = [chosen]
    elif others:
        # เกิน 2 เครื่องก็เขียนได้ — worker ทุกตัวลง WORKER_IPS เรียงตาม node-rank
        chosen = others[0]
        workers = others
    else:
        raise ClusterEnvError("กลุ่มนี้ไม่มี worker")

    head = next(m for m in group["members"] if m["name"] == head_name)
    node = find_node(chosen["name"])
    # controller มี SSH_USER ค่าเดียวใช้กับ worker ทุก rank — worker ที่ login คนละชื่อกันจึงเป็นไปไม่ได้
    # เดิมหยิบ user ของ worker ตัวแรกเงียบ ๆ แล้ว rank 2 ตายด้วย Permission denied ตอน sync-worker
    users = {}
    for member in workers:
        found = find_node(member["name"])
        if found is not None and found.user:
            users.setdefault(found.user, []).append(member["name"])
    if len(users) > 1:
        listed = " · ".join(f"{user}: {', '.join(names)}" for user, names in users.items())
        raise ClusterEnvError(
            f"worker ในกลุ่มนี้ login คนละ user ({listed}) — controller ใช้ SSH_USER ค่าเดียวกับ worker ทุกตัว · "
            f"เพิ่มเครื่องใหม่ด้วย user เดียวกัน (lmds node add … --user) หรือเลือก worker ที่ user ตรงกัน"
        )
    topology = topology_from_members([head, *workers])
    body = render_cluster_env(topology, ssh_user=(node.user if node else ""))
    head_node = topology["nodes"][0]
    return {
        "body": body,
        "head_ip": _head_ip_legacy(topology),
        "worker_ips": [_head_to_worker_ip(topology, n["rank"]) for n in topology["nodes"][1:]],
        "nnodes": len(topology["nodes"]),
        "iface": ",".join(_ifnames(head_node)),
        "topology": topology,
    }


# ── ค่าที่ลง cluster.env: ตรวจว่าเป็นของที่มันควรเป็น แล้วค่อย quote ──
#
# ไฟล์นี้ถูก controller `source` (`set -a; . "$CLUSTER_ENV"`) ทุกครั้งที่ start · ชื่อ interface, IP และชื่อ
# HCA มาจาก inventory ที่ **แต่ละ node รายงานตัวเอง** (`host.fabric.links[].iface`) ซึ่ง threat model ถือเป็น
# ข้อมูลที่ไม่น่าเชื่อถือ · audit 2026-10: node ที่ตอบชื่อ interface เป็น `x$(…)` ทำให้ hub เขียน
# `NCCL_SOCKET_IFNAME=x$(…)` ลงบน head แล้วคำสั่งข้างในรันตอน controller อ่านไฟล์
# สองชั้น ไม่ใช่ชั้นเดียว: (1) ทุกค่าต้องตรงรูปของสิ่งที่มันเป็นได้จริง ไม่ตรง = ปฏิเสธพร้อมบอกว่าฟิลด์ไหนของ
# เครื่องไหน (2) ตัวเขียนบรรทัด (_assign) quote ทุกค่าที่มีอักขระนอกชุดปลอดภัย — ชั้นแรกหลุดชั้นสองยังกันไว้
_IFACE_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,14}")   # ชื่อ interface ของ kernel (IFNAMSIZ 15) · ไม่มี `:` — ตัวคั่นของ LINKS_<rank>
_HCA_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")     # ชื่ออุปกรณ์ RDMA: mlx5_0 · rocep1s0f1
_USER_OK = re.compile(r"[A-Za-z_][A-Za-z0-9._-]{0,31}\$?")   # user ของ Linux
_KIND_OK = re.compile(r"direct-2|ring-3|switch-[0-9]{1,3}")
_SLUG_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")    # รูปเดียวกับ web/api._check_slug
_BARE_OK = re.compile(r"[A-Za-z0-9._,:/*@%+=-]*")            # bash อ่านตรงตัวโดยไม่ต้อง quote
_DQ_OK = re.compile(r"[A-Za-z0-9._,:/*@%+= -]*")             # ปลอดภัยใน "…" (ไม่มี $ ` \ ")


def _assign(key: str, value, quoted: bool = False) -> str:
    """บรรทัด `KEY=VALUE` ที่ bash อ่านกลับได้ค่าเดิมตรงตัว — ไม่ว่าในค่าจะมีอะไร

    ค่าที่ถูกรูปออกมาหน้าตาเดิมทุกตัวอักษร (ไฟล์ที่ 0.6.0 เขียนกับไฟล์วันนี้เหมือนกัน) · ค่าที่มีอักขระนอก
    ชุดปลอดภัยถูกห่อด้วย shlex.quote ซึ่ง bash ไม่ขยายอะไรข้างใน
    """
    text = str(value)
    if quoted and _DQ_OK.fullmatch(text):
        return f'{key}="{text}"'
    if not quoted and _BARE_OK.fullmatch(text):
        return f"{key}={text}"
    return f"{key}={shlex.quote(text)}"


def _prefix(value):
    """prefix จากทะเบียน/inventory เป็นอะไรก็ได้ ("24" · 24 · ขยะ) — เป็นเลขถ้าทำได้ ที่เหลือให้ด่านตรวจรายงาน"""
    try:
        return int(value or 24)
    except (TypeError, ValueError):
        return value


def _shown(value) -> str:
    text = str(value)
    return repr(text if len(text) <= 60 else text[:57] + "…")


def _check_ip(value, what: str, problems: list[str]) -> None:
    if value in ("", None):
        return
    try:
        if not isinstance(value, str) or ipaddress.ip_address(value).version != 4:
            raise ValueError
    except ValueError:
        problems.append(f"{what} {_shown(value)} ไม่ใช่ที่อยู่ IPv4")


def _validate_topology(topology: dict, ssh_user: str) -> None:
    """ทุกค่าที่กำลังจะถูกเขียนต้องเป็นของที่มันควรเป็น — ไม่ผ่าน = ClusterEnvError บอกฟิลด์และเครื่อง"""
    from lmds.nodes.registry import name_ok

    problems: list[str] = []
    if not _KIND_OK.fullmatch(str(topology.get("kind") or "")):
        problems.append(f"topology {_shown(topology.get('kind'))}")
    if ssh_user and not (isinstance(ssh_user, str) and _USER_OK.fullmatch(ssh_user)):
        problems.append(f"SSH user {_shown(ssh_user)} ไม่ใช่ชื่อ user ของ Linux")
    count = len(topology["nodes"])
    for node in topology["nodes"]:
        name = node.get("name")
        where = f"เครื่อง {_shown(name)}:"
        if not (isinstance(name, str) and name_ok(name)):
            problems.append(f"{where} ชื่อเครื่องใช้ใน cluster.env ไม่ได้ (ตัวอักษร ตัวเลข . _ - เท่านั้น)")
        for link in node["links"]:
            iface, hca = link.get("iface"), link.get("hca")
            if iface and not (isinstance(iface, str) and _IFACE_OK.fullmatch(iface)):
                problems.append(f"{where} interface {_shown(iface)} ไม่ใช่ชื่อ interface ของ Linux "
                                "(ตัวอักษร ตัวเลข . _ - ยาวไม่เกิน 15)")
            if hca and not (isinstance(hca, str) and _HCA_OK.fullmatch(hca)):
                problems.append(f"{where} RDMA device {_shown(hca)} ไม่ใช่ชื่ออุปกรณ์")
            _check_ip(link.get("ip"), f"{where} IP ของสาย", problems)
            _check_ip(link.get("peer_ip"), f"{where} IP ของปลายสาย", problems)
            prefix = link.get("prefix")
            if isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 <= prefix <= 32:
                problems.append(f"{where} prefix {_shown(prefix)} ต้องเป็นจำนวนเต็ม 0-32")
            peer = link.get("peer_rank")
            if peer != LINK_SWITCH_PEER and (isinstance(peer, bool) or not isinstance(peer, int)
                                             or not 0 <= peer < count):
                problems.append(f"{where} peer rank {_shown(peer)} ไม่ใช่ลำดับเครื่องในกลุ่ม")
    if problems:
        more = f" · และอีก {len(problems) - 6} จุด" if len(problems) > 6 else ""
        raise ClusterEnvError(
            "ไม่เขียน cluster.env — ค่าที่ได้มาไม่ใช่ของที่ควรอยู่ในไฟล์นี้ (controller source ไฟล์นี้ตอน start): "
            + " · ".join(problems[:6]) + more
            + " — ค่าพวกนี้มาจากที่เครื่องนั้นรายงานตัวเอง ตรวจเครื่องนั้นก่อนเขียนใหม่"
        )


# ─────────────────────────── topology ───────────────────────────
def _normalise_link(link: dict, rank_of: dict[str, int]) -> dict:
    """ลิงก์จากทะเบียน (`cluster_links`) → รูปที่ renderer ใช้ · peer ที่ไม่รู้จัก = สายเข้า switch"""
    peer_node = link.get("peer_node") or ""
    peer_rank = rank_of.get(peer_node)
    return {
        "iface": link.get("iface") or "",
        "ip": link.get("ip") or "",
        "prefix": _prefix(link.get("prefix")),
        "peer_rank": LINK_SWITCH_PEER if peer_rank is None else peer_rank,
        "peer_ip": (link.get("peer_ip") or "") if peer_rank is not None else "",
        "link_id": link.get("link_id") or "",
        "hca": link.get("hca") or "",
    }


def topology_from_members(members: list[dict]) -> dict:
    """สมาชิกกลุ่ม (head ก่อน แล้ว worker เรียงตาม rank) → {kind, nodes:[{name, rank, links:[…]}]}

    สมาชิกที่มี `cluster_links` (ทะเบียน 0.6.1+) ได้ทุกสาย · ที่ไม่มีได้สายเดียวจาก cluster_ip/iface
    ซึ่งคือพฤติกรรมเดิม — `direct-2` จากสมาชิกแบบเก่าจึง render ออกมาเหมือน 0.6.0 ทุกตัวอักษร
    """
    rank_of = {m["name"]: rank for rank, m in enumerate(members)}
    nodes = []
    for rank, member in enumerate(members):
        raw = member.get("cluster_links") or []
        if raw:
            links = [_normalise_link(link, rank_of) for link in raw]
        else:
            links = [{
                "iface": member.get("iface") or "", "ip": member.get("cluster_ip") or "",
                "prefix": 24, "peer_rank": LINK_SWITCH_PEER, "peer_ip": "", "link_id": "", "hca": "",
            }]
        nodes.append({"name": member["name"], "rank": rank, "links": links, "legacy": not raw})
    return {"kind": _kind(nodes), "nodes": nodes}


def _kind(nodes: list[dict]) -> str:
    n = len(nodes)
    if n == 2:
        return "direct-2"
    if n == 3:
        # วงแหวน = ทุกเครื่องมีสายตรงถึงอีกสองเครื่อง (peer_rank ระบุครบ) ไม่ใช่แค่ "มี 2 สาย"
        ring = all(
            {l["peer_rank"] for l in node["links"]} >= ({0, 1, 2} - {node["rank"]})
            for node in nodes
        )
        if ring:
            return "ring-3"
    return f"switch-{n}"


def _is_multilink(topology: dict) -> bool:
    """ต้องเขียนคีย์ v2 ไหม — 2 เครื่องแบบเก่า (สายเดียว ไม่มี cluster_links) = ไฟล์เดิมเป๊ะ"""
    return topology["kind"] != "direct-2" or any(not n["legacy"] for n in topology["nodes"])


def _same_subnet(ip: str, prefix: int, other: str) -> bool:
    try:
        return ipaddress.ip_address(other) in ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
    except ValueError:
        return False


def _pair(topology: dict, rank: int) -> tuple[dict | None, dict | None]:
    """(สายของ head, สายของ worker rank) ที่ต่อถึงกัน — สายตรงก่อน ไม่มีค่อยดูวงเดียวกัน"""
    head, worker = topology["nodes"][0], topology["nodes"][rank]
    for hl in head["links"]:
        if hl["peer_rank"] == rank:
            wl = next((l for l in worker["links"] if l["peer_rank"] == 0), None)
            if wl is None:
                wl = next((l for l in worker["links"] if l["ip"] == hl["peer_ip"]), None)
            return hl, wl
    for hl in head["links"]:
        for wl in worker["links"]:
            if hl["ip"] and wl["ip"] and _same_subnet(hl["ip"], hl["prefix"], wl["ip"]):
                return hl, wl
    return (head["links"][0] if head["links"] else None,
            worker["links"][0] if worker["links"] else None)


def _head_to_worker_ip(topology: dict, rank: int) -> str:
    hl, wl = _pair(topology, rank)
    if hl is not None and hl["peer_ip"]:
        return hl["peer_ip"]
    return wl["ip"] if wl is not None else ""


def _worker_head_ip(topology: dict, rank: int) -> str:
    hl, wl = _pair(topology, rank)
    if wl is not None and wl["peer_ip"]:
        return wl["peer_ip"]
    return hl["ip"] if hl is not None else ""


def _head_ip_legacy(topology: dict) -> str:
    """MASTER_IP ของไฟล์เดิม = IP ของ head บนสายไป worker rank 1"""
    if len(topology["nodes"]) > 1:
        ip = _worker_head_ip(topology, 1)
        if ip:
            return ip
    head = topology["nodes"][0]
    return head["links"][0]["ip"] if head["links"] else ""


def _ifnames(node: dict) -> list[str]:
    return [l["iface"] for l in node["links"] if l["iface"]]


def _hcas(node: dict) -> list[str]:
    return [l["hca"] for l in node["links"] if l["hca"]]


def _links_field(node: dict) -> str:
    parts = []
    for l in node["links"]:
        peer_ip = l["peer_ip"] or "-"
        parts.append(f"{l['iface'] or '-'}:{l['ip']}/{l['prefix']}:{l['peer_rank']}:{peer_ip}")
    return " ".join(parts)


def render_cluster_env(topology: dict, ssh_user: str = "") -> str:
    """เนื้อ cluster.env จาก topology — คีย์เดิมครบเสมอ · คีย์ v2 เมื่อมีหลายสายหรือเกิน 2 เครื่อง

    ตัวเขียนของ backend (nodes/cli/web) เรียกตรงนี้ได้ด้วย topology ที่สร้างจาก `cluster_links`
    ของทะเบียน — ชื่อคีย์ทั้งหมดอยู่ที่หัวไฟล์นี้
    """
    nodes = topology["nodes"]
    if len(nodes) < 2:
        raise ClusterEnvError("กลุ่มนี้ไม่มี worker")
    for node in nodes:
        for link in node["links"]:
            link["prefix"] = _prefix(link.get("prefix"))
    # ตรวจก่อนประกอบบรรทัดแรก — ค่าที่ไม่ใช่ของที่ควรอยู่ในไฟล์นี้ต้องไม่ถูกเขียนเลย ไม่ใช่เขียนแบบ quote ไว้
    _validate_topology(topology, ssh_user)
    head = nodes[0]
    worker_ranks = [n["rank"] for n in nodes[1:]]
    worker_ips = [_head_to_worker_ip(topology, r) for r in worker_ranks]
    master_ip = _head_ip_legacy(topology)
    lines = [
        "# สร้างโดย lmds (node cluster --write / หน้าเว็บ) — แก้มือได้ ค่า env ภายนอกยังชนะไฟล์นี้",
        _assign("MASTER_IP", master_ip),
        _assign("WORKER_IP", worker_ips[0]),
        # worker ทุกตัวเรียงตาม node-rank 1..N-1 — controller วนจากตัวแปรนี้
        _assign("WORKER_IPS", " ".join(worker_ips), quoted=True),
        _assign("NNODES", len(nodes)),
        _assign("TENSOR_PARALLEL_SIZE", len(nodes)),
        _assign("SSH_USER", ssh_user),
        _assign("TRANSPORT_IP_MASTER", master_ip),
        _assign("TRANSPORT_IP_WORKER", worker_ips[0]),
    ]
    head_ifnames = _ifnames(head)
    if head_ifnames:
        # NCCL เลือก interface เองแล้วมักได้เส้นบริหารจัดการที่ช้ากว่า — ระบุให้ชัด
        lines.append(_assign("NCCL_SOCKET_IFNAME", ",".join(head_ifnames)))
    if not _is_multilink(topology):
        return "\n".join(lines) + "\n"

    # ── v2 ──
    lines += [
        "",
        "# schema v2 — สายต่อ rank (iface:ip/prefix:peer_rank:peer_ip · peer_rank * = สายเข้า switch)",
        "CLUSTER_ENV_SCHEMA=2",
        _assign("CLUSTER_TOPOLOGY", topology["kind"]),
        _assign("CLUSTER_NODES", " ".join(n["name"] for n in nodes), quoted=True),
        # worker ทุกตัวใช้ transport IP ตามสายที่ head มองเห็น (rank order) — controller เก่าอ่านตัวนี้
        _assign("TRANSPORT_IPS_WORKER", " ".join(worker_ips), quoted=True),
    ]
    for node in nodes:
        rank = int(node["rank"])
        lines.append(_assign(f"LINKS_{rank}", _links_field(node), quoted=True))
        lines.append(_assign(f"NCCL_SOCKET_IFNAMES_{rank}", ",".join(_ifnames(node))))
        lines.append(_assign(f"NCCL_IB_HCAS_{rank}", ",".join(_hcas(node))))
        if rank > 0:
            lines.append(_assign(f"HEAD_TO_WORKER_IP_{rank}", _head_to_worker_ip(topology, rank)))
            lines.append(_assign(f"WORKER_HEAD_IP_{rank}", _worker_head_ip(topology, rank)))
    if topology["kind"] == "ring-3":
        # แต่ละคู่คุยกันคนละสาย — ต้องอนุญาตให้ NCCL ใช้ NIC คนละตัวในแต่ละคู่ ไม่งั้นบังคับ NIC เดียว
        # แล้วหาทางไป rank ที่อยู่อีกสายไม่เจอ
        lines.append("NCCL_CROSS_NIC=1")
    return "\n".join(lines) + "\n"


def parse_cluster_env(path) -> dict[str, str]:
    """อ่าน cluster.env เป็น dict — ค่าถูกถอด quote แล้ว · ไฟล์ไม่มี/อ่านไม่ได้ = dict ว่าง

    ใช้โดย `lmds remove` (ต้องรู้ว่า worker คือใคร) และหน้าเว็บ — ไม่ source ด้วย bash เพราะไฟล์นี้ผู้ใช้
    แก้มือได้ และ remove ไม่ควรรันอะไรจากมัน
    """
    from pathlib import Path

    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def worker_targets(values: dict[str, str]) -> list[dict]:
    """worker ทุกตัวเรียงตาม rank จาก cluster.env — v2 ใช้ HEAD_TO_WORKER_IP_<rank> (IP ที่ head ถึงได้จริง)
    ไฟล์เก่าใช้ WORKER_IPS/WORKER_IP · คืน [{rank, ip, ssh_user}]
    """
    ips: list[str] = []
    rank = 1
    while values.get(f"HEAD_TO_WORKER_IP_{rank}"):
        ips.append(values[f"HEAD_TO_WORKER_IP_{rank}"])
        rank += 1
    if not ips:
        ips = (values.get("WORKER_IPS") or values.get("WORKER_IP") or "").split()
    user = values.get("SSH_USER") or ""
    return [{"rank": i + 1, "ip": ip, "ssh_user": user} for i, ip in enumerate(ips)]


def write_cluster_env(slug: str, groups, head_name: str,
                      worker_name: str | None = None,
                      on_node: str | None = None) -> ClusterEnvResult:
    """เขียนไฟล์จริง — บนเครื่องนี้ หรือข้าม SSH ไปยังเครื่องที่ถือ bundle อยู่"""
    from lmds.fleet import bundle_roots

    # slug ถูกต่อเป็นคำสั่ง shell บนเครื่องปลายทาง (ls -d ~/bundles/<slug>) และเป็น path บนเครื่องนี้ —
    # หน้าเว็บตรวจที่ปากทางแล้ว แต่ CLI (`node cluster --write`) เรียกตรงมาที่นี่ จึงตรวจที่ sink ด้วย
    if not isinstance(slug, str) or not _SLUG_OK.fullmatch(slug):
        raise ClusterEnvError(
            f"ชื่อโมเดล (slug) ไม่ถูกต้อง: {_shown(slug)} — ใช้ได้เฉพาะ a-z A-Z 0-9 . _ - "
            "ขึ้นต้นด้วยตัวอักษร/ตัวเลข ยาวไม่เกิน 64")
    built = build_cluster_env(groups, head_name, worker_name)
    body = built["body"]

    if on_node:
        # bundle อยู่บนเครื่องที่จะรันมันจริง ไม่ใช่บน hub — เขียนข้ามเครื่องผ่าน SSH
        from lmds.nodes import NodeError, find as find_node, run as run_remote

        remote = find_node(on_node)
        if remote is None:
            raise ClusterEnvError(f"ไม่รู้จักเครื่อง '{on_node}' — ดู: lmds node list")
        encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
        script = (
            f"dir=\"$(ls -d ~/bundles/{slug} ~/*/bundles/{slug} ./bundles/{slug} 2>/dev/null | head -1)\"; "
            f"[ -n \"$dir\" ] || {{ echo 'ไม่พบ bundle {slug}' >&2; exit 1; }}; "
            f"echo {encoded} | base64 -d > \"$dir/cluster.env\" && "
            f"chmod 600 \"$dir/cluster.env\" && echo \"$dir/cluster.env\""
        )
        try:
            result = run_remote(remote, script, timeout=60)
        except NodeError as exc:
            raise ClusterEnvError(str(exc)) from exc
        if not result.ok:
            raise ClusterEnvError((result.stderr or result.stdout).strip()[:400])
        target = f"{remote.name}:{result.stdout.strip()}"
    else:
        bundle = next((root / slug for root in bundle_roots() if (root / slug).is_dir()), None)
        if bundle is None:
            raise ClusterEnvError(
                f"ไม่พบ bundle ของ '{slug}' บนเครื่องนี้ — "
                f"ถ้า bundle อยู่บนเครื่องอื่นให้ระบุเครื่องนั้น"
            )
        path = bundle / "cluster.env"
        path.write_text(body, encoding="utf-8")
        path.chmod(0o600)
        target = str(path)

    return ClusterEnvResult(target=target, head_ip=built["head_ip"],
                            worker_ips=built["worker_ips"], nnodes=built["nnodes"],
                            iface=built["iface"], body=body)
