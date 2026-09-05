"""ตั้งค่าสาย ConnectX ระหว่าง DGX Spark จาก hub — เทียบเท่า "Cluster Assistant" ของ NVIDIA Sync

เดิม LMDS แค่ *จด* cluster_ip/cluster_iface ลงทะเบียน ไม่เคยตั้งค่า interface ให้ใคร · ลูกค้าที่ได้
Spark มาใหม่เสียบสายแล้วก็ยังต้องไปตั้ง IP เองทีละเครื่อง (หรือใช้ NVIDIA Sync บนเครื่องอื่นอีกตัว)
ก่อนที่ stacked จะเริ่มได้ · โมดูลนี้ปิดช่องนั้น: อ่านว่าพอร์ตไหนมีสาย → เดา topology → แจก IP →
เขียน netplan ผ่าน SSH ด้วยรหัส sudo ที่ใช้ครั้งเดียว → ping ยืนยัน → จับคู่กุญแจ head→worker →
อัปเดตทะเบียน

ข้อเท็จจริงของ Spark (คู่มือ NVIDIA "ConnectX-7 Networking" + NVIDIA Sync):
- QSFP 2 ช่อง/เครื่อง (200G ต่อช่อง, Ethernet เท่านั้น) · **หนึ่งช่องคือสอง interface** (PCIe Gen5 x4 สองเส้น)
  พอร์ต 1 = `enp1s0f0np0` + `enp1s0f1np1` · พอร์ต 2 = `enP2p1s0f0np0` + `enP2p1s0f1np1` — ชื่อเหมือนกันทุกเครื่อง
- ลิงก์ที่ตั้งค่าใช้ **function เดียว** ของช่องนั้น (ฟลีตนี้ใช้ f1: spark-head `enp1s0f1np1` = 10.100.152.1)
- 2 เครื่องต่อตรง (1 สาย) · 3 เครื่องต่อตรงเป็นวง (3 สาย ใช้ทั้งสองช่องทุกเครื่อง: A.p1→B.p2, B.p1→C.p2,
  C.p1→A.p2) · 4 เครื่องต้องผ่าน switch (สายละเครื่อง ตั้ง port ที่ switch เป็น 200G ตายตัว) · 2–3 เครื่อง
  ผ่าน switch ก็ได้ · ห้ามปนตรงกับ switch
- NVIDIA Sync เขียน `/etc/netplan/99-nvidia-sync-cluster.yaml` (IP ส่วนตัวต่อลิงก์ แยกจากสายบริหาร) ·
  ถอนด้วยการย้ายไฟล์ไป /root/netplan-disabled แล้ว netplan generate/apply
- throughput ที่วัดได้จริง ~100 Gb/s ต่อลิงก์คือเพดาน PCIe x4 ไม่ใช่ความผิด

การตั้งค่าจริงทุกขั้นผ่าน `runner` (= `lmds.nodes.run`) จึงเทสด้วย SSH ปลอมได้ทั้งเส้นทาง · รหัส sudo
เดินทางทาง stdin เท่านั้น ไม่อยู่ใน argv/log/ทะเบียน (แนวเดียวกับ `ssh.run_privileged`)
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import time

from lmds.hardware.profiler import group_qsfp_ports, spark_function_of

from .registry import Node

NETPLAN_FILE = "/etc/netplan/99-lmds-cluster.yaml"
NVIDIA_SYNC_FILE = "/etc/netplan/99-nvidia-sync-cluster.yaml"
# NVIDIA ถอนคลัสเตอร์ด้วยการย้ายไฟล์มาที่นี่ ไม่ลบ — เราทำเหมือนกันเพื่อให้ถอยกลับได้เสมอ
DISABLED_DIR = "/root/netplan-disabled"
DEFAULT_BASE_SUBNET = "10.100.152.0/24"
# ฟลีตปัจจุบันตั้งลิงก์บน f1 ทุกเครื่อง — ตั้งใหม่ก็ใช้ f1 จะได้ไม่มีสองแบบปนกันในเอกสาร/หมอ
PREFERRED_FUNCTION = 1
# iperf3 ต่ำกว่านี้ค่อยเตือน — เพดาน PCIe x4 คือ ~100 จึงไม่เอา 184 (ค่า negotiate) มาวัด throughput
IPERF_WARN_GBPS = 90
VERIFY_ATTEMPTS = 6
VERIFY_PAUSE_S = 3.0
TOPOLOGY_CHOICES = ("direct", "ring", "switch")


class NetplanError(Exception):
    """ผู้ใช้แก้ได้ — ข้อความภาษาอังกฤษเพราะหน้าเว็บโชว์ตรง ๆ (CLI แปะคำอธิบายไทยเอง)"""


# ── มุมมองรายพอร์ตจาก host payload ───────────────────────────────────────────
def ports_of(host: dict | None) -> list[dict]:
    """พอร์ต QSFP ของเครื่องนี้จาก payload ที่ hub มีอยู่ — node รุ่นเก่าไม่ส่ง qsfp_ports มา ก็จัดกลุ่มให้เอง"""
    fabric = (host or {}).get("fabric") or {}
    ports = fabric.get("qsfp_ports")
    if ports is None:
        ports = group_qsfp_ports(fabric.get("links") or [])
    return [p for p in ports if p.get("port") in (1, 2)]


def cabled_ports(host: dict | None) -> list[int]:
    return [p["port"] for p in ports_of(host) if p.get("carrier")]


def iface_for(port: dict, prefer_function: int = PREFERRED_FUNCTION) -> str:
    """เลือก interface หนึ่งตัวของพอร์ตนี้ — ตัวที่ตั้ง IP ไว้แล้วชนะ ไม่งั้น f1 ตามฟลีต"""
    if port.get("configured"):
        return port["configured"]
    ifaces = list(port.get("ifaces") or [])
    for name in ifaces:
        if spark_function_of(name) == prefer_function:
            return name
    return ifaces[-1] if ifaces else ""


# ── topology ───────────────────────────────────────────────────────────────────
def _unknown(reason: str, order: list[str]) -> dict:
    return {"topology": "unknown", "links": [], "reason": reason, "order": list(order)}


def infer_topology(cabled: dict[str, list[int]], order: list[str], forced: str = "") -> dict:
    """เดาว่าเสียบสายแบบไหนจาก "พอร์ตไหนมี carrier" ของแต่ละเครื่อง

    คืน {"topology": direct-2 | ring-3 | switch-N | unknown, "links": [...], "reason", "order"}
    link = {"id", "ends": [{"node", "port"}, ...]} · switch มี link เดียวที่มีปลายทุกเครื่อง

    carrier บอกได้แค่ว่า "ช่องนี้มีสายและอีกฝั่งขึ้น" ไม่บอกว่าปลายอีกข้างคือใคร — วง 3 เครื่องและคู่
    ที่เสียบสองสายจึงเป็น *สมมติฐาน* ตามผังของ NVIDIA ที่ต้อง ping ยืนยันตอน apply · `forced`
    (direct/ring/switch) ใช้เมื่อหลักฐานตีความได้สองทาง (2 เครื่องสายละช่อง = ต่อตรงหรือผ่าน switch ก็ได้)
    """
    order = [n for n in order if n]
    n = len(order)
    # หน้าเว็บส่งค่าที่ inspect ตอบกลับมาทั้งก้อน ("direct-2" / "switch-4") — เอาแค่ชนิด
    forced = (forced or "").strip().lower().split("-")[0]
    if forced and forced not in TOPOLOGY_CHOICES:
        return _unknown(f"unknown topology '{forced}' — use one of {', '.join(TOPOLOGY_CHOICES)}", order)
    if n < 2 or n > 4:
        return _unknown(f"a cluster is 2–4 machines (got {n})", order)
    if len(set(order)) != n:
        return _unknown("the same machine is listed twice", order)

    counts = {name: len(cabled.get(name) or []) for name in order}
    missing = [name for name in order if counts[name] == 0]
    if missing:
        return _unknown(
            f"no cable detected on {', '.join(missing)} (both QSFP ports show NO-CARRIER) — "
            "plug the QSFP cable and check the link LED", order)

    one_each = all(c == 1 for c in counts.values())
    two_each = all(c == 2 for c in counts.values())

    # switch: สายละเครื่อง — 4 เครื่องเป็นทางเดียว · 3 เครื่องที่ไม่ครบสองช่องทุกเครื่องก็คือ switch ·
    # 2 เครื่องสายละช่องเป็น direct โดยปริยาย (ผลเหมือนกัน: วงเดียว .1/.2) เว้นแต่สั่งว่า switch
    if one_each and (n >= 3 or forced == "switch"):
        if forced == "ring":
            return _unknown("a ring needs both QSFP ports cabled on every machine — "
                            "each machine has one cable, which is a switch layout", order)
        if forced == "direct" and n > 2:
            return _unknown(f"{n} machines cannot be cabled directly with one cable each — "
                            "this is a switch layout", order)
        link = {"id": 0, "ends": [{"node": name, "port": (cabled[name] or [1])[0]} for name in order]}
        return {"topology": f"switch-{n}", "links": [link], "reason": "", "order": order}

    if n == 2:
        if forced == "ring":
            return _unknown("a ring is three machines — two machines connect directly", order)
        a, b = order
        pa, pb = sorted(cabled[a]), sorted(cabled[b])
        if len(pa) != len(pb):
            return _unknown(
                f"mixed cabling: {a} has {len(pa)} cabled port(s) but {b} has {len(pb)} — "
                "a direct pair uses the same number of cables on both machines", order)
        if len(pa) == 1:
            links = [{"id": 0, "ends": [{"node": a, "port": pa[0]}, {"node": b, "port": pb[0]}]}]
        else:
            # สองสายระหว่างคู่เดียวกัน (สายจริงของ spark-head/spark-worker): สมมติช่องเดียวกันชนกัน
            # (1↔1, 2↔2) — ถ้าเสียบไขว้ ping ตอน apply จะบอกเอง
            links = [{"id": i, "ends": [{"node": a, "port": p}, {"node": b, "port": p}]}
                     for i, p in enumerate(pa)]
        return {"topology": "direct-2", "links": links, "reason": "", "order": order}

    if n == 3:
        if two_each and forced != "switch":
            a, b, c = order
            links = [
                {"id": 0, "ends": [{"node": a, "port": 1}, {"node": b, "port": 2}]},
                {"id": 1, "ends": [{"node": b, "port": 1}, {"node": c, "port": 2}]},
                {"id": 2, "ends": [{"node": c, "port": 1}, {"node": a, "port": 2}]},
            ]
            return {"topology": "ring-3", "links": links, "reason": "", "order": order}
        if two_each and forced == "switch":
            return _unknown("every machine has both ports cabled — that is a ring, not a switch "
                            "(a switch takes one cable per machine)", order)
        odd = [f"{name} ({counts[name]})" for name in order]
        return _unknown(
            "mixed cabling: a 3-machine ring needs both ports on every machine, a switch needs "
            f"exactly one cable per machine — cabled ports: {', '.join(odd)}", order)

    both = [name for name in order if counts[name] == 2]
    return _unknown(
        f"4 machines must go through a switch (one cable per machine) — {', '.join(both)} "
        "has both ports cabled", order)


# ── IP allocation ───────────────────────────────────────────────────────────────
def _network_of(ip: str, prefix: int | None, fallback_prefix: int) -> ipaddress.IPv4Network | None:
    try:
        return ipaddress.ip_network(f"{ip}/{prefix or fallback_prefix}", strict=False)
    except ValueError:
        return None


def allocate_links(topology: dict, hosts: dict[str, dict | None], base_subnet: str = DEFAULT_BASE_SUBNET,
                   prefer_function: int = PREFERRED_FUNCTION) -> tuple[list[dict], list[str]]:
    """ใส่ interface + IP ให้ทุกปลายของทุกลิงก์ — deterministic และเก็บ IP เดิมไว้เมื่อมันเข้ากันได้

    กติกา: direct/ring หนึ่ง /24 ต่อลิงก์ นับจาก base (10.100.152.0/24 → .153 → .154) ปลาย .1/.2 ตามลำดับ
    ในลิงก์ · switch วงเดียว เครื่องที่ i ได้ .i+1 · ปลายที่มี IP อยู่แล้วบนวงเดียวกัน (ไม่ซ้ำกัน) ถูกเก็บไว้
    และวงนั้นถูกกันไม่ให้ลิงก์อื่นเอาไปใช้ · สายบริหาร (ไม่ใช่ ConnectX) ไม่ถูกแตะเลย
    """
    try:
        base = ipaddress.ip_network(base_subnet, strict=False)
    except ValueError as exc:
        raise NetplanError(f"base subnet '{base_subnet}' is not valid: {exc}") from exc
    if base.version != 4 or base.prefixlen > 30:
        raise NetplanError("base subnet must be an IPv4 network of /30 or larger (default 10.100.152.0/24)")

    warnings: list[str] = []
    links: list[dict] = []
    for link in topology.get("links") or []:
        ends = []
        for end in link["ends"]:
            port = next((p for p in ports_of(hosts.get(end["node"])) if p.get("port") == end["port"]), None)
            if port is None:
                raise NetplanError(f"{end['node']} has no QSFP port {end['port']} in its inventory")
            iface = iface_for(port, prefer_function)
            if not iface:
                raise NetplanError(f"{end['node']} port {end['port']} has no Linux interface")
            ends.append({"node": end["node"], "port": end["port"], "iface": iface,
                         "existing_ip": port.get("ip") or "", "existing_prefix": port.get("prefix")})
        links.append({"id": link["id"], "ends": ends})

    # รอบแรก: ลิงก์ที่มี IP เดิมเข้ากันได้ (ทุกปลายที่มี IP อยู่วงเดียวกันและไม่ซ้ำ) ยึดวงนั้นไว้ก่อน
    taken: set[ipaddress.IPv4Network] = set()
    for link in links:
        nets = {}
        for end in link["ends"]:
            if end["existing_ip"]:
                net = _network_of(end["existing_ip"], end["existing_prefix"], base.prefixlen)
                if net is not None:
                    nets[end["node"]] = net
        ips = [e["existing_ip"] for e in link["ends"] if e["existing_ip"]]
        if nets and len(set(nets.values())) == 1 and len(set(ips)) == len(ips):
            net = next(iter(nets.values()))
            if net in taken:
                warnings.append(f"link {link['id']}: existing addresses reuse {net} already used by "
                                "another link — re-allocating")
                continue
            link["subnet"] = net
            taken.add(net)
        elif nets:
            named = ", ".join(f"{e['node']}={e['existing_ip']}" for e in link["ends"] if e["existing_ip"])
            warnings.append(f"link {link['id']}: existing addresses do not fit one subnet ({named}) — "
                            "re-allocating from the base subnet")

    # รอบสอง: ลิงก์ที่เหลือได้วงถัดไปจาก base ที่ยังว่าง — ลำดับคงที่จึงรันซ้ำได้ผลเดิม
    candidate = base
    for link in links:
        if link.get("subnet") is not None:
            continue
        while candidate in taken:
            candidate = _next_subnet(candidate)
        link["subnet"] = candidate
        taken.add(candidate)

    for link in links:
        net = link["subnet"]
        usable = [str(h) for h in net.hosts()]
        kept = {e["existing_ip"] for e in link["ends"]
                if e["existing_ip"] and _network_of(e["existing_ip"], e["existing_prefix"], net.prefixlen) == net}
        free = [ip for ip in usable if ip not in kept]
        for end in link["ends"]:
            if end["existing_ip"] in kept:
                end["ip"] = end["existing_ip"]
            else:
                if not free:
                    raise NetplanError(f"subnet {net} has no free address left for {end['node']}")
                end["ip"] = free.pop(0)
            end["prefix"] = net.prefixlen
            end["changed"] = end["ip"] != end["existing_ip"]
            end.pop("existing_prefix", None)
        link["subnet"] = str(net)
    return links, warnings


def _next_subnet(net: ipaddress.IPv4Network) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(f"{net.broadcast_address + 1}/{net.prefixlen}", strict=False)


# ── netplan YAML ────────────────────────────────────────────────────────────────
def render_netplan(assignments: list[dict], renderer: str = "networkd") -> str:
    """YAML ของ /etc/netplan/99-lmds-cluster.yaml — เฉพาะ interface ที่แผนตั้ง ที่เหลือไม่แตะ

    ไม่มี routes/gateway โดยตั้งใจ: วงคลัสเตอร์เป็น point-to-point ไม่ควรมีอะไรวิ่งออกไปทางนี้
    `optional: true` เพราะสายที่ถอดออกต้องไม่ทำให้บูตค้างรอ network-online
    """
    lines = [
        "# Managed by LMDS (lmds cluster apply) — ConnectX cluster links. Do not edit by hand;",
        "# re-run `lmds cluster apply` or remove with `lmds cluster remove-net <node>`.",
        "network:",
        "  version: 2",
        f"  renderer: {renderer}",
        "  ethernets:",
    ]
    for item in sorted(assignments, key=lambda a: a["iface"]):
        lines += [
            f"    {item['iface']}:",
            "      dhcp4: no",
            f"      addresses: [{item['ip']}/{item['prefix']}]",
            "      optional: true",
        ]
    return "\n".join(lines) + "\n"


# ── plan ────────────────────────────────────────────────────────────────────────
def build_plan(order: list[str], hosts: dict[str, dict | None], *, base_subnet: str = DEFAULT_BASE_SUBNET,
               topology: str = "", nodes: dict[str, Node] | None = None, renderer: str = "networkd") -> dict:
    """แผนทั้งหมดจาก payload ที่ hub มีอยู่ — ไม่แตะเครื่องเลย

    รูป JSON (คงที่ — หน้าเว็บ (wizard) และ CLI `--json` ใช้ร่วมกัน):
      {"ok", "topology": "direct-2|ring-3|switch-N|unknown", "reason", "order", "base_subnet", "warnings": [str],
       "links": [{"id", "link_id", "subnet",
                  "ends": [{"node", "port", "iface", "ip", "prefix", "changed"}, ...],   # switch = ปลายทุกเครื่อง
                  "a": ends[0], "b": ends[1]}],
       "nodes": {name: {"links"/"iface_ips": [{"iface", "ip", "prefix", "peer_node", "peer_ip", "link_id", "qsfp_port"}],
                        "netplan"/"netplan_yaml": "<yaml>", "cluster_ip", "cluster_iface", "changed": bool,
                        "changes": [str]}},
       "per_node": (ตัวเดียวกับ nodes),
       "registry": {name: {"cluster_ip", "cluster_iface", "cluster_links": [...]}}}
    """
    order = [n for n in order if n]
    unreachable = [n for n in order if not hosts.get(n)]
    if unreachable:
        result = _unknown(f"no inventory for {', '.join(unreachable)} — the hub has not reached "
                          "the machine yet (refresh and retry)", order)
        return {"ok": False, **result, "base_subnet": base_subnet, "warnings": [],
                "nodes": {}, "registry": {}}
    cabled = {name: cabled_ports(hosts[name]) for name in order}
    no_ports = [name for name in order if not ports_of(hosts[name])]
    if no_ports:
        result = _unknown(f"{', '.join(no_ports)} reports no ConnectX QSFP ports — not a DGX Spark, "
                          "or the node runs an LMDS too old to report them", order)
        return {"ok": False, **result, "base_subnet": base_subnet, "warnings": [],
                "nodes": {}, "registry": {}}

    inferred = infer_topology(cabled, order, topology)
    if inferred["topology"] == "unknown":
        return {"ok": False, **inferred, "base_subnet": base_subnet, "warnings": [],
                "nodes": {}, "registry": {}}

    links, warnings = allocate_links(inferred, hosts, base_subnet)
    per_node: dict[str, dict] = {}
    for link in links:
        for end in link["ends"]:
            entry = per_node.setdefault(end["node"], {"links": [], "assignments": {}})
            entry["assignments"][end["iface"]] = end
            for peer in link["ends"]:
                if peer["node"] == end["node"]:
                    continue
                entry["links"].append({
                    "iface": end["iface"], "ip": end["ip"], "prefix": end["prefix"],
                    "peer_node": peer["node"], "peer_ip": peer["ip"], "link_id": link["id"],
                    "qsfp_port": end["port"],
                })

    head = order[0]
    registry: dict[str, dict] = {}
    nodes_out: dict[str, dict] = {}
    for name in order:
        entry = per_node[name]
        # cluster_ip = เส้นที่ head↔worker ใช้: worker เอาลิงก์ที่ไปถึง head · head เอาลิงก์ไป worker ตัวแรก
        if name == head:
            main = next((l for l in entry["links"] if l["peer_node"] == order[1]), entry["links"][0])
        else:
            main = next((l for l in entry["links"] if l["peer_node"] == head), entry["links"][0])
        assignments = list(entry["assignments"].values())
        current = nodes.get(name) if nodes else None
        changed = any(a["changed"] for a in assignments) or (
            current is not None and (current.cluster_ip != main["ip"] or current.cluster_iface != main["iface"]))
        # ประโยคสั้น ๆ ต่อการเปลี่ยนแปลง — หน้าเว็บโชว์ใต้ชื่อเครื่องให้คนอ่านก่อนกด apply
        changes = [f"write {NETPLAN_FILE}"]
        for a in sorted(assignments, key=lambda x: x["iface"]):
            before = a.get("existing_ip") or ""
            if not a["changed"]:
                changes.append(f"{a['iface']}: keep {a['ip']}/{a['prefix']}")
            elif before:
                changes.append(f"{a['iface']}: {before} → {a['ip']}/{a['prefix']}")
            else:
                changes.append(f"{a['iface']}: set {a['ip']}/{a['prefix']} (replaces the 169.254.x.x link-local address)")
        if (hosts[name].get("fabric") or {}).get("nvidia_sync_netplan"):
            changes.append(f"move {NVIDIA_SYNC_FILE.rsplit('/', 1)[-1]} to {DISABLED_DIR}")
        if current is not None and current.cluster_ip != main["ip"]:
            changes.append(f"registry: cluster_ip {current.cluster_ip or '(unset)'} → {main['ip']}")
        yaml_text = render_netplan(assignments, renderer)
        nodes_out[name] = {
            "links": entry["links"],
            "iface_ips": entry["links"],
            "netplan": yaml_text,
            "netplan_yaml": yaml_text,
            "cluster_ip": main["ip"],
            "cluster_iface": main["iface"],
            "changed": changed,
            "changes": changes,
        }
        registry[name] = {"cluster_ip": main["ip"], "cluster_iface": main["iface"],
                          "cluster_links": entry["links"]}

    if inferred["topology"] == "ring-3":
        warnings.append("ring-3: every machine has two links on two subnets — the registry's cluster_ip "
                        "is the link toward the head; multi-link NCCL uses cluster_links")
    for name in order:
        if (hosts[name].get("fabric") or {}).get("nvidia_sync_netplan"):
            warnings.append(f"{name}: {NVIDIA_SYNC_FILE} exists — apply moves it to {DISABLED_DIR} "
                            "so the LMDS file is the only one claiming these interfaces")
    links_out = []
    for l in links:
        ends = [{k: e[k] for k in ("node", "port", "iface", "ip", "prefix", "changed")} for e in l["ends"]]
        links_out.append({"id": l["id"], "link_id": f"L{l['id']}", "subnet": l["subnet"], "ends": ends,
                          "a": ends[0], "b": ends[1] if len(ends) > 1 else ends[0]})
    return {
        "ok": True,
        "topology": inferred["topology"],
        "reason": "",
        "order": order,
        "base_subnet": base_subnet,
        "warnings": warnings,
        "links": links_out,
        "nodes": nodes_out,
        "per_node": nodes_out,
        "registry": registry,
    }


# ── apply ───────────────────────────────────────────────────────────────────────
def _q(text: str) -> str:
    return shlex.quote(text)


def stage_script() -> str:
    """เขียน YAML (ทาง stdin) ลงไฟล์ชั่วคราวของ user แล้วพิมพ์ path — ไม่ต้อง sudo"""
    return "f=$(mktemp /tmp/lmds-netplan.XXXXXX) && cat > \"$f\" && chmod 600 \"$f\" && echo \"$f\""


def apply_script(staged: str, ifaces: list[str], stamp: str) -> str:
    """สคริปต์ที่รันใต้ sudo: สำรองของเดิม → ปลดไฟล์อื่นที่อ้าง interface เดียวกัน → ติดตั้ง → generate → apply

    ไฟล์ netplan ถูก merge ตามชื่อ: `99-lmds` แพ้ `99-nvidia-sync` ถ้าปล่อยไว้ทั้งคู่ → IP ของเราไม่เคยขึ้น
    เงียบ ๆ · จึงย้ายไฟล์ *ที่เอ่ยถึง interface ของเรา* ไป /root/netplan-disabled (ทางเดียวกับที่ NVIDIA
    ใช้ถอนคลัสเตอร์) ประทับ stamp ให้ rollback หาเจอ · ไฟล์ของสายบริหาร (ไม่เอ่ยถึง ConnectX) ไม่ถูกแตะ
    """
    pattern = "|".join(re.escape(i) for i in sorted(ifaces))
    return (
        "set -e; "
        f"f={NETPLAN_FILE}; d={DISABLED_DIR}; s={_q(stamp)}; mkdir -p \"$d\"; "
        "if [ -f \"$f\" ]; then cp -p \"$f\" \"$d/$(basename \"$f\").$s\"; echo \"backup $d/$(basename \"$f\").$s\"; fi; "
        "for g in /etc/netplan/*.yaml; do [ -f \"$g\" ] || continue; [ \"$g\" = \"$f\" ] && continue; "
        f"if grep -qE {_q(f'^[[:space:]]+({pattern}):')} \"$g\"; then "
        "mv \"$g\" \"$d/$(basename \"$g\").$s\"; echo \"disabled $g\"; fi; done; "
        f"install -m 0600 -o root -g root {_q(staged)} \"$f\"; rm -f {_q(staged)}; "
        "netplan generate && netplan apply && echo LMDS_NETPLAN_APPLIED"
    )


def rollback_script(stamp: str) -> str:
    """ถอยกลับไปก่อน apply รอบที่มี stamp นี้: เอาไฟล์ของเราออก คืนไฟล์ที่ย้ายไป แล้ว apply ใหม่"""
    return (
        f"f={NETPLAN_FILE}; d={DISABLED_DIR}; s={_q(stamp)}; rm -f \"$f\"; "
        "for b in \"$d\"/*.\"$s\"; do [ -f \"$b\" ] || continue; "
        "mv \"$b\" \"/etc/netplan/$(basename \"$b\" .\"$s\")\"; echo \"restored $(basename \"$b\" .\"$s\")\"; done; "
        "netplan generate; netplan apply; echo LMDS_NETPLAN_ROLLED_BACK"
    )


def remove_script(stamp: str) -> str:
    """ถอนแบบ NVIDIA: ย้ายไฟล์ไป /root/netplan-disabled (ไม่ลบ) แล้ว generate/apply"""
    return (
        f"f={NETPLAN_FILE}; d={DISABLED_DIR}; s={_q(stamp)}; mkdir -p \"$d\"; "
        "if [ -f \"$f\" ]; then mv \"$f\" \"$d/$(basename \"$f\").$s\"; echo \"moved to $d/$(basename \"$f\").$s\"; "
        "else echo LMDS_NETPLAN_ABSENT; fi; netplan generate; netplan apply; echo LMDS_NETPLAN_REMOVED"
    )


def sudo_wrap(script: str) -> str:
    # -S = อ่านรหัสจาก stdin · -p '' = ไม่พิมพ์ prompt ปน output · รหัสไม่เคยอยู่ในสตริงนี้
    return f"sudo -S -p '' bash -c {_q(script)}"


def verify_script(assignments: list[dict]) -> str:
    """พิมพ์ addr + link ของทุก interface ที่ตั้ง — ผู้เรียกดูว่า CIDR ขึ้นและ LOWER_UP อยู่"""
    parts = []
    for item in assignments:
        parts.append(f"ip -br addr show dev {_q(item['iface'])}; ip -br link show dev {_q(item['iface'])}")
    return "; ".join(parts)


def verify_output_ok(text: str, assignments: list[dict]) -> list[str]:
    """รายการปัญหาจากผล verify_script — ว่าง = ครบทุก interface"""
    problems = []
    for item in assignments:
        rows = [line for line in (text or "").splitlines() if line.split() and line.split()[0] == item["iface"]]
        cidr = f"{item['ip']}/{item['prefix']}"
        if not any(cidr in row for row in rows):
            problems.append(f"{item['iface']} does not show {cidr}")
        if not any("LOWER_UP" in row for row in rows):
            problems.append(f"{item['iface']} lost carrier (no LOWER_UP)")
    return problems


def _unreachable(result) -> bool:
    return getattr(result, "exit_code", 1) in (124, 255)


def _scrub(text: str, secrets: list[str]) -> str:
    text = (text or "").strip()
    for secret in secrets:
        if secret:
            text = text.replace(secret, "•••")
    return text[-300:]


def apply_plan(plan: dict, passwords: dict[str, str], *, nodes: dict[str, Node] | None = None,
               runner=None, progress=None, pair: bool = True, speed_test: bool = True,
               update_registry=None, sleep=time.sleep, stamp: str = "") -> dict:
    """ตั้งค่าตามแผนบนทุกเครื่อง — คืน {"ok", "applied", "steps", "nodes", "pings", "pairing", "speed", "registry"}

    ลำดับ: ตรวจรหัส sudo ของทุกเครื่อง *ก่อน* แตะเครื่องแรก (รหัสผิดเครื่องที่สองไม่ควรทิ้งเครื่องแรกไว้
    ครึ่งทาง) → ต่อเครื่อง: stage YAML → sudo apply → ยืนยัน (IP ขึ้น + LOWER_UP, ลองซ้ำเพราะลิงก์กระพริบ
    หลัง netplan apply) → ล้ม = rollback → ping ทุกลิงก์จากทั้งสองปลาย → กุญแจ head→worker → iperf3
    (ถ้ามีทั้งสองฝั่ง เตือนอย่างเดียว) → ทะเบียน · รันซ้ำได้: ไฟล์เดิมถูกสำรองแล้วเขียนทับด้วยเนื้อหาเดิม

    `passwords` = {ชื่อเครื่อง: รหัส sudo} ใช้ทาง stdin ของ `sudo -S` เท่านั้น · step detail ถูกกรอง
    ค่ารหัสออกก่อนคืน · `runner`/`update_registry`/`sleep` แทนของจริงในเทส
    """
    from . import ssh

    run = runner or ssh.run
    secrets = [p for p in passwords.values() if p]
    if nodes is None:
        from .registry import load

        nodes = {n.name: n for n in load()}
    if update_registry is None:
        from .registry import update as update_registry
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")

    steps: list[dict] = []
    report = {"ok": False, "applied": False, "steps": steps, "nodes": {}, "pings": [],
              "pairing": [], "speed": [], "registry": {}, "stamp": stamp}

    def step(node: str, what: str, ok: bool, detail: str = "", level: str = "") -> dict:
        item = {"node": node, "step": what, "ok": ok, "detail": _scrub(detail, secrets),
                "level": level or ("pass" if ok else "fail")}
        steps.append(item)
        if progress is not None:
            progress(item)
        return item

    if not plan.get("ok") or not plan.get("nodes"):
        step("", "plan", False, plan.get("reason") or "the plan is not applicable")
        return report
    order = list(plan["order"])
    missing = [n for n in order if n not in nodes]
    if missing:
        step("", "registry", False, f"{', '.join(missing)} not in the registry")
        return report
    passwords = {n: (passwords.get(n) or "") for n in order}

    # 1. sudo ทุกเครื่องก่อน — `sudo -S -v` ไม่ทำอะไรนอกจากตรวจรหัส · ไม่ส่งรหัสมา = เครื่องนั้นต้องมี
    #    NOPASSWD (เคสจริง msi-5 2026-09-05) ตรวจด้วย `sudo -n` ไม่มี = ล้มตั้งแต่ตรงนี้ ยังไม่แตะอะไร
    for name in order:
        ok = _sudo_ok(run, nodes[name], passwords[name], step)
        if not ok:
            return report

    # 2. ต่อเครื่อง
    for name in order:
        node = nodes[name]
        spec = plan["nodes"][name]
        assignments = [{"iface": l["iface"], "ip": l["ip"], "prefix": l["prefix"]}
                       for l in _unique_assignments(spec["links"])]
        outcome = {"ok": False, "rolled_back": False}
        report["nodes"][name] = outcome

        staged = run(node, stage_script(), timeout=30, stdin_text=spec["netplan"])
        path = (staged.stdout or "").strip().splitlines()[-1].strip() if staged.ok and (staged.stdout or "").strip() else ""
        if not staged.ok or not path.startswith("/tmp/lmds-netplan."):
            step(name, "stage netplan file", False, staged.stderr or staged.stdout or "mktemp failed")
            return report
        step(name, "stage netplan file", True, path)

        applied = run(node, sudo_wrap(apply_script(path, [a["iface"] for a in assignments], stamp)),
                      timeout=180, stdin_text=passwords[name] + "\n")
        applied_ok = applied.ok and "LMDS_NETPLAN_APPLIED" in (applied.stdout or "")
        notes = [line for line in (applied.stdout or "").splitlines() if line.startswith(("backup", "disabled"))]
        step(name, f"write {NETPLAN_FILE} + netplan apply", applied_ok,
             "; ".join(notes) if applied_ok else (applied.stderr or applied.stdout or "netplan apply failed"))
        if not applied_ok:
            _rollback(name, node, passwords[name], stamp, run, step, outcome)
            return report

        problems = _verify_with_retries(node, assignments, run, sleep)
        if problems:
            step(name, "verify addresses + carrier", False, "; ".join(problems))
            _rollback(name, node, passwords[name], stamp, run, step, outcome)
            return report
        step(name, "verify addresses + carrier", True,
             ", ".join(f"{a['iface']} {a['ip']}/{a['prefix']} LOWER_UP" for a in assignments))
        outcome["ok"] = True
    report["applied"] = True

    # 3. ping ทุกลิงก์จากทุกปลาย — บอกได้ว่าสายเสียบตรงผังไหม (ring/สองสายเป็นสมมติฐานจนถึงตรงนี้)
    pings_ok = True
    for name in order:
        for link in plan["nodes"][name]["links"]:
            pinged = run(nodes[name], f"ping -c 3 -W 2 -I {_q(link['iface'])} {_q(link['peer_ip'])} >/dev/null 2>&1",
                         timeout=30)
            item = {"node": name, "iface": link["iface"], "peer_node": link["peer_node"],
                    "peer_ip": link["peer_ip"], "link_id": link["link_id"], "ok": pinged.ok}
            report["pings"].append(item)
            pings_ok = pings_ok and pinged.ok
            step(name, f"ping {link['peer_node']} {link['peer_ip']} via {link['iface']}", pinged.ok,
                 "" if pinged.ok else "no reply — if the cabling is crossed (port 1 ↔ port 2) swap the cable "
                                      "or re-run with the machines listed in cable order")

    # 4. กุญแจ head → worker บนที่อยู่ใหม่ (controller ใช้ IP นี้)
    pairing_ok = True
    if pair and len(order) > 1:
        from .cluster_ssh import pair_workers

        head = nodes[order[0]]
        workers = [(nodes[w], plan["registry"][w]["cluster_ip"]) for w in order[1:]]
        pairing = pair_workers(head, workers, runner=run)
        report["pairing"] = pairing
        pairing_ok = bool(pairing) and all(s["ok"] for s in pairing)
        for item in pairing:
            step(item.get("node") or order[0], item["step"], item["ok"], item.get("detail", ""))

    # 5. iperf3 — มีก็วัด ไม่มีก็ข้าม ต่ำกว่า 90 แค่เตือน (เพดาน PCIe x4 ~100 ไม่ใช่ 200)
    if speed_test:
        report["speed"] = _speed_tests(plan, nodes, run, step)

    # 6. ทะเบียน — เขียนเมื่อ IP ขึ้นครบ ต่อให้ ping/กุญแจยังไม่ผ่าน (ค่าที่ตั้งจริงบนเครื่องคือความจริง)
    for name in order:
        fields = plan["registry"][name]
        try:
            update_registry(name, cluster_ip=fields["cluster_ip"], cluster_iface=fields["cluster_iface"],
                            cluster_links=fields["cluster_links"])
            report["registry"][name] = fields
            step(name, "registry updated", True, f"cluster_ip {fields['cluster_ip']} on {fields['cluster_iface']}")
        except Exception as exc:  # noqa: BLE001 — ทะเบียนล้มไม่ควรทำให้รายงานทั้งก้อนหาย
            step(name, "registry updated", False, str(exc))
            pings_ok = False
    report["ok"] = report["applied"] and pings_ok and pairing_ok
    return report


def _unique_assignments(links: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for link in links:
        seen.setdefault(link["iface"], link)
    return list(seen.values())


def _verify_with_retries(node: Node, assignments: list[dict], run, sleep) -> list[str]:
    problems = ["not verified"]
    for attempt in range(VERIFY_ATTEMPTS):
        shown = run(node, verify_script(assignments), timeout=30)
        if _unreachable(shown):
            problems = ["management ssh session failed after netplan apply"]
        else:
            problems = verify_output_ok(shown.stdout, assignments)
        if not problems:
            return []
        if attempt + 1 < VERIFY_ATTEMPTS:
            sleep(VERIFY_PAUSE_S)
    return problems


def _sudo_ok(run, node: Node, password: str, step) -> bool:
    """ตรวจว่า sudo บนเครื่องนี้ใช้ได้ — มีรหัส = `sudo -S -v` · ไม่มีรหัส = ต้อง NOPASSWD (`sudo -n`)

    ชื่อ step มีคำว่า "sudo password" เสมอ — wizard บนหน้าเว็บจับติ๊กช่องแรกด้วย regex นี้"""
    if password:
        checked = run(node, "sudo -S -p '' -v && echo LMDS_SUDO_OK", timeout=30, stdin_text=password + "\n")
        ok = checked.ok and "LMDS_SUDO_OK" in (checked.stdout or "")
        step(node.name, "sudo password accepted", ok,
             "" if ok else ("unreachable" if _unreachable(checked) else (checked.stderr or checked.stdout or "rejected")))
        return ok
    checked = run(node, "sudo -n true && echo LMDS_SUDO_OK", timeout=30)
    ok = checked.ok and "LMDS_SUDO_OK" in (checked.stdout or "")
    step(node.name, "sudo password not needed (passwordless sudo)" if ok else "sudo password", ok,
         "" if ok else ("unreachable" if _unreachable(checked)
                        else f"missing sudo password for {node.name} — sudo asks for one there"))
    return ok


def sudo_needs_password(node: Node, runner=None) -> bool | None:
    """True = ต้องใส่รหัส · False = NOPASSWD · None = ต่อไม่ติด — หน้าเว็บใช้ตัดสินว่าจะโชว์ช่องรหัสไหม"""
    from . import ssh

    run = runner or ssh.run
    checked = run(node, "sudo -n true && echo LMDS_SUDO_OK", timeout=15)
    if checked.ok and "LMDS_SUDO_OK" in (checked.stdout or ""):
        return False
    return None if _unreachable(checked) else True


def _rollback(name: str, node: Node, password: str, stamp: str, run, step, outcome: dict) -> None:
    rolled = run(node, sudo_wrap(rollback_script(stamp)), timeout=180, stdin_text=password + "\n")
    ok = rolled.ok and "LMDS_NETPLAN_ROLLED_BACK" in (rolled.stdout or "")
    outcome["rolled_back"] = ok
    step(name, "rollback to the previous netplan", ok,
         "" if ok else (rolled.stderr or rolled.stdout or f"could not roll back — previous files are in {DISABLED_DIR}"),
         level="warn" if ok else "fail")


def _speed_tests(plan: dict, nodes: dict[str, Node], run, step) -> list[dict]:
    """iperf3 5 วิ ต่อลิงก์ (ปลาย a เป็น server ครั้งเดียว, ปลาย b เป็น client) — ไม่มีก็ข้าม ไม่ล้มเพราะมัน"""
    import json

    results = []
    has = {}
    for name in plan["order"]:
        has[name] = run(nodes[name], "command -v iperf3 >/dev/null 2>&1", timeout=20).ok
    for link in plan["links"]:
        ends = link["ends"]
        if len(ends) < 2:
            continue
        pairs = [(ends[0], e) for e in ends[1:]]
        for server, client in pairs:
            if not (has.get(server["node"]) and has.get(client["node"])):
                results.append({"link_id": link["id"], "from": client["node"], "to": server["node"],
                                "gbps": None, "skipped": "iperf3 not installed on both ends"})
                continue
            run(nodes[server["node"]], f"iperf3 -s -1 -D -B {_q(server['ip'])} -p 5201 >/dev/null 2>&1", timeout=20)
            measured = run(nodes[client["node"]], f"iperf3 -c {_q(server['ip'])} -t 5 -J -p 5201", timeout=60)
            gbps = None
            try:
                data = json.loads(measured.stdout or "{}")
                gbps = round(float(data["end"]["sum_received"]["bits_per_second"]) / 1e9, 1)
            except (ValueError, KeyError, TypeError):
                pass
            item = {"link_id": link["id"], "from": client["node"], "to": server["node"], "gbps": gbps,
                    "skipped": "" if gbps is not None else "iperf3 gave no result"}
            results.append(item)
            if gbps is None:
                step(client["node"], f"iperf3 → {server['node']}", True, item["skipped"], level="warn")
            else:
                slow = gbps < IPERF_WARN_GBPS
                step(client["node"], f"iperf3 → {server['node']} {server['ip']}", True,
                     f"{gbps} Gbit/s" + (" — below 90 (PCIe x4 ceiling is ~100); check the switch port speed"
                                         if slow else ""),
                     level="warn" if slow else "pass")
    return results


def remove_net(node: Node, password: str, *, runner=None, update_registry=None, stamp: str = "") -> dict:
    """ถอนไฟล์ของเรา (ย้ายไป /root/netplan-disabled แบบ NVIDIA) แล้วล้าง cluster_* ในทะเบียน"""
    from . import ssh

    run = runner or ssh.run
    if update_registry is None:
        from .registry import update as update_registry
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    steps: list[dict] = []
    password = password or ""

    def step(name: str, what: str, ok: bool, detail: str = "", level: str = "") -> None:
        steps.append({"node": name, "step": what, "ok": ok, "detail": _scrub(detail, [password] if password else []),
                      "level": level or ("pass" if ok else "fail")})

    if not _sudo_ok(run, node, password, step):
        return {"ok": False, "removed": False, "absent": False, "steps": steps}
    removed = run(node, sudo_wrap(remove_script(stamp)), timeout=180, stdin_text=password + "\n")
    out = removed.stdout or ""
    ok = removed.ok and "LMDS_NETPLAN_REMOVED" in out
    absent = "LMDS_NETPLAN_ABSENT" in out
    steps.append({"node": node.name, "step": f"move {NETPLAN_FILE} to {DISABLED_DIR} + netplan apply", "ok": ok,
                  "detail": ("no LMDS file was present" if absent else
                             next((l for l in out.splitlines() if l.startswith("moved")), "")) if ok
                  else _scrub(removed.stderr or out, [password]),
                  "level": "pass" if ok else "fail"})
    if ok:
        try:
            update_registry(node.name, cluster_ip="", cluster_iface="", cluster_links=[])
            steps.append({"node": node.name, "step": "registry cleared", "ok": True, "detail": "", "level": "pass"})
        except Exception as exc:  # noqa: BLE001
            steps.append({"node": node.name, "step": "registry cleared", "ok": False, "detail": str(exc),
                          "level": "fail"})
            ok = False
    return {"ok": ok, "removed": ok and not absent, "absent": absent, "steps": steps}


def _ui_port(port: dict, links: list[dict]) -> dict:
    """พอร์ตหนึ่งช่องในรูปที่ wizard วาด: {qsfp_port, interfaces: [{iface, function, carrier, speed_gbps, ip, prefix,
    rdma_device, netplan_managed}]} — interface ที่ node รุ่นเก่าไม่บอก carrier ใช้ค่าของพอร์ต"""
    by_name = {l.get("iface"): l for l in links}
    interfaces = []
    for iface in port.get("ifaces") or []:
        link = by_name.get(iface) or {}
        fn = link.get("function")
        if fn is None:
            fn = spark_function_of(iface)
        carrier = link.get("carrier")
        ip = link.get("ip") or ""
        interfaces.append({
            "iface": iface,
            "function": f"f{fn}" if fn is not None else "",
            "carrier": bool(port.get("carrier")) if carrier is None else bool(carrier),
            "speed_gbps": link.get("speed_gbps") or (port.get("speed_gbps") if port.get("carrier") else None),
            # link-local 169.254.x = ยังไม่ได้ตั้ง — หน้าเว็บโชว์ว่า "no IP" ตรงกว่าโชว์ที่อยู่ที่ใช้ไม่ได้
            "ip": "" if str(ip).startswith("169.254.") else ip,
            "prefix": None if str(ip).startswith("169.254.") else link.get("prefix"),
            "rdma_device": link.get("rdma_device") or "",
            "netplan_managed": link.get("netplan_managed"),
        })
    return {"qsfp_port": port.get("port"), "carrier": bool(port.get("carrier")), "speed_gbps": port.get("speed_gbps"),
            "configured": port.get("configured") or "", "ip": port.get("ip") or "", "prefix": port.get("prefix"),
            "interfaces": interfaces}


def inspect_nodes(order: list[str], hosts: dict[str, dict | None], errors: dict[str, str] | None = None,
                  topology: str = "") -> dict:
    """สรุปที่หน้าเว็บ/CLI ใช้ก่อนวางแผน — รายพอร์ตต่อเครื่อง + topology ที่เดาได้ (ไม่แตะเครื่อง)

    {"nodes": {name: {"reachable", "error", "hostname", "spark", "sudo_needed": true,
                      "fabric": {"ports": [{"qsfp_port", "carrier", "speed_gbps", "configured", "ip", "prefix",
                                            "interfaces": [{"iface", "function", "carrier", "speed_gbps", "ip", "prefix",
                                                            "rdma_device", "netplan_managed"}]}],
                                 "links": [...ดิบจาก agent...], "netplan_files", "nvidia_sync"},
                      "ports": [...group_qsfp_ports...], "netplan_files", "nvidia_sync"}},
     "topology": {"kind", "topology", "links", "reason", "order", "cabled": {name: [port]}}}
    """
    errors = errors or {}
    out: dict[str, dict] = {}
    cabled: dict[str, list[int]] = {}
    for name in order:
        host = hosts.get(name)
        if not host:
            out[name] = {"reachable": False, "error": errors.get(name) or "no data yet", "hostname": "",
                         "spark": False, "sudo_needed": True, "ports": [], "netplan_files": [], "nvidia_sync": False,
                         "fabric": {"ports": [], "links": [], "netplan_files": [], "nvidia_sync": False}}
            continue
        fabric = host.get("fabric") or {}
        ports = ports_of(host)
        raw_links = fabric.get("links") or []
        cabled[name] = [p["port"] for p in ports if p.get("carrier")]
        out[name] = {
            "reachable": True, "error": "", "hostname": host.get("hostname") or "",
            "spark": any("gb10" in (g.get("name") or "").lower() for g in host.get("gpus") or []),
            # ทุกขั้นที่เขียน netplan ต้องใช้ sudo — หน้าเว็บใช้ตัดสินว่าจะขอรหัส
            "sudo_needed": True,
            "ports": ports,
            "fabric": {"ports": [_ui_port(p, raw_links) for p in ports], "links": raw_links,
                       "netplan_files": fabric.get("netplan_files") or [],
                       "nvidia_sync": bool(fabric.get("nvidia_sync_netplan"))},
            "netplan_files": fabric.get("netplan_files") or [],
            "nvidia_sync": bool(fabric.get("nvidia_sync_netplan")),
        }
    if all(out[n]["reachable"] for n in order) and order:
        inferred = infer_topology(cabled, list(order), topology)
    else:
        down = [n for n in order if not out[n]["reachable"]]
        inferred = _unknown(f"no inventory for {', '.join(down)}", list(order))
    return {"nodes": out, "topology": {**inferred, "kind": inferred["topology"], "cabled": cabled}}
