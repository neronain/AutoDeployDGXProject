"""เขียน cluster.env ลง bundle — ค่าที่ controller ต้องรู้ก่อน start แบบ stacked

แยกออกมาจาก CLI เพราะหน้าเว็บต้องเขียนไฟล์นี้ด้วย · เดิมตรรกะทั้งก้อนอยู่ใน
`lmds/cli/main.py` แล้วเรียก `typer.Exit` กับ `err_console` ตรง ๆ ซึ่งเรียกจาก
FastAPI ไม่ได้ — หน้าเว็บจึงทำได้แค่ *พิมพ์คำสั่ง CLI ให้ผู้ใช้ไปก็อป* ทั้งที่ทุกอย่าง
ที่ต้องใช้อยู่ในมือ server แล้ว

ที่นี่ไม่พิมพ์อะไรและไม่ออกจากโปรเซส — ล้มก็โยน ClusterEnvError ให้ผู้เรียกตัดสินใจเอง
"""

from __future__ import annotations

import base64
from dataclasses import dataclass


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
    iface = head.get("iface") or ""
    worker_ips = [m["cluster_ip"] for m in workers]
    lines = [
        "# สร้างโดย lmds (node cluster --write / หน้าเว็บ) — แก้มือได้ ค่า env ภายนอกยังชนะไฟล์นี้",
        f"MASTER_IP={head['cluster_ip']}",
        f"WORKER_IP={worker_ips[0]}",
        # worker ทุกตัวเรียงตาม node-rank 1..N-1 — controller วนจากตัวแปรนี้
        f'WORKER_IPS="{" ".join(worker_ips)}"',
        f"NNODES={len(workers) + 1}",
        f"TENSOR_PARALLEL_SIZE={len(workers) + 1}",
        f"SSH_USER={node.user if node else ''}",
        f"TRANSPORT_IP_MASTER={head['cluster_ip']}",
        f"TRANSPORT_IP_WORKER={worker_ips[0]}",
    ]
    if iface:
        # NCCL เลือก interface เองแล้วมักได้เส้นบริหารจัดการที่ช้ากว่า — ระบุให้ชัด
        lines.append(f"NCCL_SOCKET_IFNAME={iface}")
    return {
        "body": "\n".join(lines) + "\n",
        "head_ip": head["cluster_ip"],
        "worker_ips": worker_ips,
        "nnodes": len(workers) + 1,
        "iface": iface,
    }


def write_cluster_env(slug: str, groups, head_name: str,
                      worker_name: str | None = None,
                      on_node: str | None = None) -> ClusterEnvResult:
    """เขียนไฟล์จริง — บนเครื่องนี้ หรือข้าม SSH ไปยังเครื่องที่ถือ bundle อยู่"""
    from lmds.fleet import bundle_roots

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
