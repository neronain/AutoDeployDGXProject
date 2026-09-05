"""หมอของคู่ stacked — บอก *เหตุผล* ที่คู่นี้ยังใช้ไม่ได้ ไม่ใช่แค่ "ไม่ผ่าน"

ลูกค้าเห็นแค่ป้าย "not ready"/"ไม่ผ่าน" บนหน้า Cluster · เหตุผลจริงกระจายอยู่หลายที่:
cluster IP ยังไม่ตั้ง · IP ตั้งแล้วแต่คนละวง · การ์ดขึ้นแค่ 50G · head ไม่มีกุญแจไป worker ·
ดิสก์ worker ไม่พอรับ weight · bundle ยังไม่มี cluster.env — แต่ละอย่างแก้คนละวิธี

ทุกข้อตรวจเป็น **รหัส** (`kind`) + ข้อมูลดิบ · ประโยคมีสองภาษาในตารางเดียว (`describe`)
เพราะ CLI พูดไทยและหน้าเว็บพูดอังกฤษ · ตรวจอ่านอย่างเดียว: ไม่ start/stop/เขียนอะไรทั้งสิ้น
"""

from __future__ import annotations

import shlex

from .cluster import (
    MIN_STACK_GBPS,
    check_cluster_ip,
    link_network,
    link_warning,
    machine_signature,
)
from .registry import Node

# ต่ำกว่านี้ weight ของโมเดล 70B (~140 GB bf16) ลงไม่ครบทั้งสองเครื่อง — เตือน ไม่บล็อก
LOW_DISK_GB = 150

_TEXT = {
    "registered": ("{names} is not in the registry — add it first (lmds node add)",
                   "{names} ไม่อยู่ในทะเบียน — เพิ่มก่อน (lmds node add)"),
    "reachable": ("{names} is unreachable from the hub: {error}",
                  "hub ต่อ {names} ไม่ได้: {error}"),
    "gpu": ("{names} reports no GPU — cannot be a stacked member",
            "{names} ไม่พบ GPU — เป็นสมาชิก stacked ไม่ได้"),
    "opted-out": ("{names} is excluded from stacked groups (stack = off)",
                  "{names} ถูกตั้งไม่เอาเข้ากลุ่ม stacked (stack = off)"),
    "same-site": ("machines are in different sites ({sites}) — stacked needs one rack, not a WAN",
                  "เครื่องอยู่คนละไซต์ ({sites}) — stacked ต้องอยู่แร็คเดียวกัน ไม่ใช่ข้าม WAN"),
    "hardware": ("hardware differs: {signatures} — every rank must be the same GPU model and count",
                 "ฮาร์ดแวร์ไม่ตรงกัน: {signatures} — ทุก rank ต้อง GPU รุ่นเดียวกัน จำนวนเท่ากัน"),
    "cluster-ip": ("cluster IP on {names}: {state}{hint}",
                   "cluster IP ของ {names}: {state}{hint}"),
    "same-subnet": ("cluster IPs are on different subnets ({networks}) — NCCL cannot connect",
                    "cluster IP อยู่คนละวง ({networks}) — NCCL ต่อกันไม่ติด"),
    "iface-up": ("the fast link on {names} ({iface}) is not up",
                 "สายเร็วของ {names} ({iface}) ยังไม่ขึ้น"),
    "link-speed": ("link on {names} negotiated at {speed}G (expected ≥{expected}G) — works but slow; "
                   "check the switch port speed",
                   "สายของ {names} ได้แค่ {speed}G (ควร ≥{expected}G) — ใช้ได้แต่ช้า ตรวจ port speed ที่ switch"),
    "ssh-head-to-worker": ("{head} cannot ssh to {user}@{ip} without a password: {error}",
                           "{head} ssh ไป {user}@{ip} แบบไม่ถามรหัสไม่ได้: {error}"),
    "fabric-ping": ("{head} cannot ping {ip} over the cluster link (ICMP may be blocked — verify with ssh)",
                    "{head} ping {ip} บนสายคลัสเตอร์ไม่ถึง (อาจแค่บล็อก ICMP — ยืนยันด้วย ssh)"),
    "disk": ("{names} has only {free} GB free — stacked weights land on every machine",
             "{names} เหลือดิสก์ {free} GB — weight ของ stacked ต้องอยู่ทุกเครื่อง"),
    "bundle-on-head": ("bundle '{slug}' is not on {head} — push it first",
                       "ไม่มี bundle '{slug}' บน {head} — push ไปก่อน"),
    "cluster-env": ("bundle '{slug}' on {head} has no cluster.env — start would prompt for IPs "
                    "(which cannot happen from the web)",
                    "bundle '{slug}' บน {head} ยังไม่มี cluster.env — start จะถาม IP (ซึ่งทำจากหน้าเว็บไม่ได้)"),
    "cluster-env-match": ("cluster.env on {head} points at {found} but the registry says {expected}",
                          "cluster.env บน {head} ชี้ไป {found} แต่ทะเบียนบอก {expected}"),
    "ok": ("{what}", "{what}"),
    # ── สายและพอร์ต ConnectX (lmds cluster inspect/plan) ──
    "spark-ports": ("{names} reports no ConnectX QSFP ports — not a DGX Spark, or its LMDS is too old to report them",
                    "{names} ไม่รายงานพอร์ต QSFP ของ ConnectX — ไม่ใช่ DGX Spark หรือ LMDS บนเครื่องเก่าเกินไป"),
    "cabling": ("{names}: no cable detected — both QSFP ports show NO-CARRIER",
                "{names}: ไม่พบสาย — พอร์ต QSFP ทั้งสองช่องขึ้น NO-CARRIER"),
    "topology": ("cabling does not match a supported layout: {reason}",
                 "การเสียบสายไม่ตรงผังที่รองรับ: {reason}"),
    "port-function": ("{names} port {port} has addresses on both functions ({ifaces}) — only one should carry the cluster IP",
                      "{names} พอร์ต {port} มี IP ทั้งสอง function ({ifaces}) — ควรมีตัวเดียวที่ถือ cluster IP"),
    "port-speed": ("{names} port {port} negotiated {speed}G (expected ≥{expected}G) — force the switch port to 200G",
                   "{names} พอร์ต {port} ได้ {speed}G (ควร ≥{expected}G) — ตั้ง port ที่ switch เป็น 200G ตายตัว"),
    "netplan-managed": ("{names} still has {file} — `lmds cluster apply` moves it to {disabled} before writing its own file",
                        "{names} ยังมี {file} — `lmds cluster apply` จะย้ายไป {disabled} ก่อนเขียนไฟล์ของตัวเอง"),
    "link-ping": ("{node} cannot ping {peer} ({peer_ip}) over {iface}",
                  "{node} ping {peer} ({peer_ip}) ทาง {iface} ไม่ถึง"),
    "firewall": ("{names}: ufw is active and does not allow traffic in on {iface} — the worker cannot reach "
                 "the head's master port even though ping works",
                 "{names}: ufw เปิดอยู่และไม่ได้ปล่อยทางเข้า {iface} — worker ต่อพอร์ต master ของ head ไม่ได้ทั้งที่ ping ถึง"),
}

# ประโยคตอน "ผ่าน" — ต้องมีคนละชุด ไม่งั้น ✓ ตามด้วยประโยคของอาการล้ม อ่านแล้วขัดกันเอง
_PASS = {
    "registered": ("both machines are in the registry", "ทั้งสองเครื่องอยู่ในทะเบียน"),
    "reachable": ("the hub reaches both machines", "hub ต่อได้ทั้งสองเครื่อง"),
    "gpu": ("GPU present on both", "มี GPU ทั้งสองเครื่อง"),
    "opted-out": ("both allowed in stacked groups", "ทั้งคู่ยอมเข้ากลุ่ม stacked"),
    "same-site": ("same site ({sites})", "ไซต์เดียวกัน ({sites})"),
    "hardware": ("same hardware on every rank ({signatures})", "ฮาร์ดแวร์ตรงกันทุก rank ({signatures})"),
    "cluster-ip": ("cluster IP on {names}: {state}", "cluster IP ของ {names}: {state}"),
    "same-subnet": ("cluster IPs share one subnet ({networks})", "cluster IP อยู่วงเดียวกัน ({networks})"),
    "iface-up": ("fast link on {names} ({iface}) is up", "สายเร็วของ {names} ({iface}) ขึ้นแล้ว"),
    "link-speed": ("link speed on {names} is as expected", "สายของ {names} ได้ความเร็วตามที่ควร"),
    "ssh-head-to-worker": ("{head} reaches {user}@{ip} over ssh without a password",
                           "{head} ssh ไป {user}@{ip} ได้โดยไม่ถามรหัส"),
    "fabric-ping": ("{head} pings {ip} over the cluster link", "{head} ping {ip} บนสายคลัสเตอร์ถึง"),
    "disk": ("enough disk on {names}", "ดิสก์ของ {names} พอ"),
    "bundle-on-head": ("bundle '{slug}' is on {head}", "มี bundle '{slug}' บน {head}"),
    "cluster-env": ("bundle '{slug}' on {head} has a cluster.env", "bundle '{slug}' บน {head} มี cluster.env"),
    "cluster-env-match": ("cluster.env on {head} matches the registry ({found})",
                          "cluster.env บน {head} ตรงกับทะเบียน ({found})"),
    "spark-ports": ("{names} has both ConnectX QSFP ports", "{names} มีพอร์ต QSFP ของ ConnectX ครบสองช่อง"),
    "cabling": ("{names}: cable on port {ports}", "{names}: มีสายที่พอร์ต {ports}"),
    "topology": ("cabling matches {topology}", "การเสียบสายตรงผัง {topology}"),
    "port-function": ("{names} port {port} uses one function ({ifaces})", "{names} พอร์ต {port} ใช้ function เดียว ({ifaces})"),
    "port-speed": ("{names} port {port} negotiated {speed}G", "{names} พอร์ต {port} ได้ {speed}G"),
    "netplan-managed": ("{names}: no foreign netplan file claims the cluster ports",
                        "{names}: ไม่มีไฟล์ netplan ของคนอื่นถือพอร์ตคลัสเตอร์"),
    "link-ping": ("{node} pings {peer} ({peer_ip}) over {iface}", "{node} ping {peer} ({peer_ip}) ทาง {iface} ถึง"),
    "firewall": ("{names}: {state}", "{names}: {state}"),
}

_FIX = {
    "cluster-ip": "lmds node set {name} --cluster-ip {suggested}",
    "same-subnet": "lmds node set <name> --cluster-ip <ip on the shared subnet>",
    "ssh-head-to-worker": "lmds cluster pair {head} {worker}   (or the Pair SSH button)",
    "cluster-env": "lmds cluster write {slug} --head {head} --worker {worker}",
    "cluster-env-match": "lmds cluster write {slug} --head {head} --worker {worker}",
    "opted-out": "lmds node set {name} --stack",
    "same-site": "lmds node set <name> --site <same site>",
    "cabling": "plug a QSFP cable into {name} and check the link LED, then: lmds cluster inspect …",
    "topology": "re-cable to one of: 2 direct · 3 ring (both ports) · 2–4 via switch (one cable each)",
    "port-speed": "set the switch port to 200G fixed (no auto-negotiation)",
    "netplan-managed": "lmds cluster apply {names}   (moves the NVIDIA Sync file aside automatically)",
    "firewall": "sudo ufw allow in on {iface}   (on {node}; lmds cluster apply does this for you)",
}


def describe(finding: dict, lang: str = "en") -> str:
    """ประโยคของ finding หนึ่งข้อ — en สำหรับหน้าเว็บ · th สำหรับ CLI"""
    table = _PASS if finding.get("ok") else _TEXT
    en, th = table.get(finding["kind"], ("{kind}", "{kind}"))
    template = th if lang == "th" else en
    data = {"kind": finding["kind"], "names": ", ".join(finding.get("names") or []),
            **(finding.get("data") or {})}
    try:
        return template.format(**data)
    except (KeyError, IndexError):
        return template


def _fix_for(kind: str, data: dict) -> str:
    template = _FIX.get(kind, "")
    try:
        return template.format(**data)
    except (KeyError, IndexError):
        return template


def _finding(kind: str, ok: bool, names: list[str], level: str = "fail", **data) -> dict:
    item = {"kind": kind, "ok": ok, "level": "pass" if ok else level, "names": names,
            "data": data}
    if not ok:
        item["fix"] = _fix_for(kind, data)
    return item


def _fast_link(host: dict, cluster_ip: str) -> dict | None:
    for link in (host.get("fabric") or {}).get("links") or []:
        if link.get("ip") == cluster_ip:
            return link
    return None


def diagnose_pair(head_name: str, worker_name: str, *, nodes: dict[str, Node],
                  hosts: dict[str, dict | None], errors: dict[str, str] | None = None,
                  runner=None, slug: str = "", bundle_dir: str = "") -> dict:
    """ตรวจคู่ (head, worker) ทีละข้อ — คืน {"ok", "findings": [...], "head", "worker"}

    `hosts` = host payload ที่ hub มีอยู่แล้ว (แคชของ refresher) · None = ต่อไม่ได้ (เหตุผลใน
    `errors`) · `runner` = ตัวรันคำสั่งบน node (แทน `lmds.nodes.run` ในเทส) — ใช้เฉพาะข้อที่ต้อง
    ถาม head จริง (ssh/ping/cluster.env) และไม่ถูกเรียกเลยเมื่อของพื้นฐานยังไม่ผ่าน
    """
    errors = errors or {}
    findings: list[dict] = []
    pair = [head_name, worker_name]

    missing = [n for n in pair if n not in nodes]
    findings.append(_finding("registered", not missing, missing or pair))
    if missing:
        return _result(head_name, worker_name, findings)

    down = [n for n in pair if not hosts.get(n)]
    findings.append(_finding("reachable", not down, down or pair,
                             error="; ".join(errors.get(n, "no data yet") for n in down)))
    if down:
        return _result(head_name, worker_name, findings)

    head, worker = nodes[head_name], nodes[worker_name]
    head_host, worker_host = hosts[head_name] or {}, hosts[worker_name] or {}

    no_gpu = [n for n in pair if not (hosts[n] or {}).get("gpus")]
    findings.append(_finding("gpu", not no_gpu, no_gpu or pair))

    opted_out = [n for n in pair if nodes[n].stack is False]
    findings.append(_finding("opted-out", not opted_out, opted_out or pair,
                             name=(opted_out or [""])[0]))

    sites = {head.site or "(no site)", worker.site or "(no site)"}
    findings.append(_finding("same-site", len(sites) == 1, pair, sites=" vs ".join(sorted(sites))))

    signatures = [machine_signature(head_host), machine_signature(worker_host)]
    findings.append(_finding("hardware", signatures[0] == signatures[1], pair,
                             signatures=" vs ".join(f"{n}: {s[2]} ×{s[3]} {s[0]}"
                                                    for n, s in zip(pair, signatures))))

    checks = {n: check_cluster_ip(hosts[n] or {}, nodes[n].cluster_ip) for n in pair}
    for name in pair:
        check = checks[name]
        ok = check["state"] == "ok"
        hint = ""
        if not ok:
            from .cluster import suggest_cluster_ip

            suggested = suggest_cluster_ip(hosts[name] or {})
            hint = f" — suggested {suggested}" if suggested else ""
            findings.append(_finding("cluster-ip", False, [name], state=check["state"], hint=hint,
                                     name=name, suggested=suggested or "<ip>"))
        else:
            findings.append(_finding("cluster-ip", True, [name], state=f"{nodes[name].cluster_ip} on "
                                     f"{check['iface']} {check['speed_gbps']}G", hint=""))
    ips_ok = all(checks[n]["state"] == "ok" for n in pair)

    if ips_ok:
        networks = {n: link_network(_fast_link(hosts[n], nodes[n].cluster_ip) or
                                    {"ip": nodes[n].cluster_ip}) for n in pair}
        findings.append(_finding("same-subnet", len(set(networks.values())) == 1, pair,
                                 networks=" vs ".join(f"{n}: {v}" for n, v in networks.items())))
        for name in pair:
            link = _fast_link(hosts[name], nodes[name].cluster_ip) or {}
            state = (link.get("state") or "up").lower()
            findings.append(_finding("iface-up", state in {"up", "unknown", ""}, [name],
                                     iface=link.get("iface") or "?"))
            warning = link_warning(hosts[name], link)
            if warning:
                findings.append(_finding("link-speed", False, [name], level="warn",
                                         speed=warning["speed_gbps"], expected=warning["expected_gbps"]))

    for name in pair:
        free = (hosts[name] or {}).get("disk_free_gb")
        if isinstance(free, (int, float)) and free < LOW_DISK_GB:
            findings.append(_finding("disk", False, [name], level="warn", free=round(free)))

    # ข้อที่ต้องถาม head จริง — ต่อเมื่อ IP ครบและเครื่องคุยกับ hub ได้ ไม่งั้นผลจะเป็น noise
    basics_ok = not no_gpu and not opted_out and len(sites) == 1 and signatures[0] == signatures[1]
    if runner is not None and ips_ok and basics_ok:
        worker_ip = worker.cluster_ip
        target = shlex.quote(f"{worker.user}@{worker_ip}")
        reached = runner(head, f"ssh -o BatchMode=yes -o ConnectTimeout=8 {target} true", timeout=40)
        findings.append(_finding("ssh-head-to-worker", reached.ok, [head_name],
                                 head=head_name, worker=worker_name, user=worker.user, ip=worker_ip,
                                 error=(reached.stderr or reached.stdout or "").strip()[-200:]))
        if not reached.ok:
            pinged = runner(head, f"ping -c1 -W2 {shlex.quote(worker_ip)} >/dev/null 2>&1", timeout=20)
            findings.append(_finding("fabric-ping", pinged.ok, [head_name], level="warn",
                                     head=head_name, ip=worker_ip))
        if slug:
            quoted = shlex.quote(slug)
            probe = runner(head, (
                f"dir=\"$(ls -d ~/bundles/{quoted} ~/*/bundles/{quoted} 2>/dev/null | head -1)\"; "
                f"[ -n \"$dir\" ] || {{ echo NOBUNDLE; exit 0; }}; "
                f"[ -f \"$dir/cluster.env\" ] && cat \"$dir/cluster.env\" || echo NOENV"
            ), timeout=30)
            text = (probe.stdout or "").strip()
            if not probe.ok or text == "NOBUNDLE":
                findings.append(_finding("bundle-on-head", False, [head_name], slug=slug, head=head_name))
            elif text in {"NOENV", ""}:
                findings.append(_finding("cluster-env", False, [head_name], slug=slug,
                                         head=head_name, worker=worker_name))
            else:
                env = dict(line.split("=", 1) for line in text.splitlines()
                           if "=" in line and not line.startswith("#"))
                found = f"MASTER_IP={env.get('MASTER_IP', '?')} WORKER_IPS={env.get('WORKER_IPS', '?').strip(chr(34))}"
                expected = f"MASTER_IP={head.cluster_ip} WORKER_IPS={worker_ip}"
                matches = (env.get("MASTER_IP") == head.cluster_ip
                           and worker_ip in env.get("WORKER_IPS", "").strip('"').split())
                findings.append(_finding("cluster-env-match", matches, [head_name], slug=slug,
                                         head=head_name, worker=worker_name, found=found,
                                         expected=expected))
    return _result(head_name, worker_name, findings)


def _result(head: str, worker: str, findings: list[dict]) -> dict:
    return {
        "head": head,
        "worker": worker,
        "ok": not any(f["level"] == "fail" for f in findings),
        "findings": findings,
    }


# ── หมอของ "สาย" — ก่อนที่จะมี cluster IP ให้ตรวจ (lmds cluster inspect / plan) ─────────
def diagnose_network(order: list[str], *, nodes: dict[str, Node], hosts: dict[str, dict | None],
                     errors: dict[str, str] | None = None, runner=None, topology: str = "",
                     plan: dict | None = None) -> dict:
    """ตรวจสาย/พอร์ต/function/netplan ของกลุ่ม 2–4 เครื่อง — คืน {"ok", "findings", "names", "topology"}

    ต่างจาก diagnose_pair ตรงที่ทำงานได้ตั้งแต่ยังไม่มี cluster IP: คำถามคือ "เสียบสายถูกไหม" ไม่ใช่
    "IP ตรงกันไหม" · `plan` (จาก netplan.build_plan) ทำให้ ping ต่อลิงก์ได้ผ่าน `runner`
    """
    from .cluster import SPARK_LINK_GBPS
    from .netplan import DISABLED_DIR, NVIDIA_SYNC_FILE, cabled_ports, infer_topology, ports_of

    errors = errors or {}
    order = [n for n in order if n]
    findings: list[dict] = []

    missing = [n for n in order if n not in nodes]
    findings.append(_finding("registered", not missing, missing or order))
    if missing:
        return _net_result(order, findings, "unknown")
    down = [n for n in order if not hosts.get(n)]
    findings.append(_finding("reachable", not down, down or order,
                             error="; ".join(errors.get(n, "no data yet") for n in down)))
    if down:
        return _net_result(order, findings, "unknown")

    no_ports = [n for n in order if len(ports_of(hosts[n])) < 2]
    findings.append(_finding("spark-ports", not no_ports, no_ports or order))
    if no_ports:
        return _net_result(order, findings, "unknown")

    cabled = {n: cabled_ports(hosts[n]) for n in order}
    for name in order:
        findings.append(_finding("cabling", bool(cabled[name]), [name], name=name,
                                 ports=" + ".join(str(p) for p in cabled[name]) or "-"))
    inferred = infer_topology(cabled, order, topology)
    findings.append(_finding("topology", inferred["topology"] != "unknown", order,
                             topology=inferred["topology"], reason=inferred.get("reason", "")))

    for name in order:
        fabric = hosts[name].get("fabric") or {}
        links = fabric.get("links") or []
        for port in ports_of(hosts[name]):
            if not port.get("carrier"):
                continue
            with_ip = [l["iface"] for l in links if l.get("iface") in (port.get("ifaces") or [])
                       and l.get("ip") and not str(l["ip"]).startswith("169.254.")]
            findings.append(_finding("port-function", len(with_ip) <= 1, [name], level="warn",
                                     port=port["port"], ifaces=", ".join(with_ip) or port.get("configured") or "-"))
            speed = port.get("speed_gbps") or 0
            if 0 < speed < SPARK_LINK_GBPS:
                findings.append(_finding("port-speed", False, [name], level="warn", port=port["port"],
                                         speed=speed, expected=SPARK_LINK_GBPS))
        if fabric.get("nvidia_sync_netplan"):
            findings.append(_finding("netplan-managed", False, [name], level="warn", file=NVIDIA_SYNC_FILE,
                                     disabled=DISABLED_DIR))
        else:
            findings.append(_finding("netplan-managed", True, [name]))

    if runner is not None and plan and plan.get("ok"):
        for name in order:
            for link in (plan["nodes"].get(name) or {}).get("links") or []:
                pinged = runner(nodes[name], f"ping -c1 -W2 -I {shlex.quote(link['iface'])} "
                                            f"{shlex.quote(link['peer_ip'])} >/dev/null 2>&1", timeout=20)
                findings.append(_finding("link-ping", pinged.ok, [name], level="warn", node=name,
                                         peer=link["peer_node"], peer_ip=link["peer_ip"], iface=link["iface"]))
    # ไฟร์วอลล์หลัง ping: ping ถึงแต่ TCP ไม่ถึงคืออาการของมัน (เคสจริง 2026-09-05)
    if runner is not None:
        for name in order:
            findings.append(_firewall_finding(name, nodes[name], hosts[name], runner))
    return _net_result(order, findings, inferred["topology"])


def _firewall_finding(name: str, node: Node, host: dict, runner) -> dict:
    """ufw บนเครื่องนี้กันสายคลัสเตอร์ไหม — เคสจริง 2026-09-05 (cynbangkok): ping ถึงแต่ worker ต่อ head:25000 ไม่ได้

    `ufw status` ต้อง root: ลอง `sudo -n` (NOPASSWD) ก่อน · ไม่ได้ก็อ่าน /etc/ufw/ufw.conf (644) ว่า ENABLED=yes
    → เตือนว่าต้องปล่อยเอง เพราะดู rule ไม่ได้ · ไม่มี ufw/ปิดอยู่ = ผ่าน"""
    from .netplan import ports_of

    ifaces = [i for p in ports_of(host) if p.get("carrier") for i in [p.get("configured") or (p.get("ifaces") or [""])[-1]] if i]
    probe = runner(node, "if sudo -n ufw status 2>/dev/null; then :; elif grep -qs '^ENABLED=yes' /etc/ufw/ufw.conf; "
                         "then echo LMDS_UFW_ENABLED_UNREADABLE; else echo LMDS_UFW_OFF; fi", timeout=20)
    out = probe.stdout or ""
    if not probe.ok and not out:
        return _finding("firewall", True, [name], state="firewall not checked (unreachable)")
    if "LMDS_UFW_OFF" in out or "Status: inactive" in out:
        return _finding("firewall", True, [name], state="no active firewall (ufw off)")
    if "LMDS_UFW_ENABLED_UNREADABLE" in out:
        return _finding("firewall", False, [name], level="warn", node=name, iface=" / ".join(ifaces) or "<cluster iface>",
                        state="ufw enabled (rules not readable without sudo)")
    if "Status: active" in out:
        missing = [i for i in ifaces if f" on {i}" not in out and f"on {i}" not in out]
        if missing:
            return _finding("firewall", False, [name], level="warn", node=name, iface=" / ".join(missing))
        return _finding("firewall", True, [name], state=f"ufw active, cluster interfaces allowed ({', '.join(ifaces) or '-'})")
    return _finding("firewall", True, [name], state="no active firewall")


def _net_result(names: list[str], findings: list[dict], topology: str) -> dict:
    return {
        "names": names,
        "topology": topology,
        "ok": not any(f["level"] == "fail" for f in findings),
        "findings": findings,
    }
