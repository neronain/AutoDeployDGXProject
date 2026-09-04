"""กุญแจ head → worker สำหรับ stacked — ข้อต่อที่หายไประหว่าง hub กับ controller

controller แบบ stacked รันบน head แล้ว `ssh ${SSH_USER}@${WORKER_IP}` ไปสั่ง docker/rsync
บน worker ด้วย **กุญแจของ head เอง** · แต่ `lmds node setup`/หน้าเว็บติดตั้งแค่กุญแจของ *hub*
ลงทุกเครื่อง — head ไม่เคยได้กุญแจไป worker เลย · README ของ bundle จึงต้องเขียนว่า
"ตั้ง passwordless SSH master→worker เอง" ซึ่งคนที่ deploy จากหน้าเว็บไม่มีวันเห็น แล้ว
sync-worker/start ก็ตายด้วย "Permission denied (publickey)" — เหตุผลอันดับต้น ๆ ที่
ลูกค้าบอกว่า "multi-node ไม่เคยติด"

วิธี: กุญแจ **เกิดบน head** (ssh-keygen ที่นั่น) ไม่ผ่าน hub · hub แค่ถือ public key ไปวาง
ใน authorized_keys ของ worker และเขียน stanza ใน ~/.ssh/config ของ head ให้ ssh ธรรมดา
(ที่ controller ใช้ ไม่มี -i) หยิบกุญแจนี้เองเมื่อคุยกับ IP ของ worker · ทำซ้ำได้ (idempotent)
ไม่แตะกุญแจอื่นของผู้ใช้ และถอนได้ด้วยการลบ block ที่มี marker
"""

from __future__ import annotations

import base64
import shlex

from .registry import Node

KEY_PATH = "~/.ssh/id_lmds_cluster"
_BEGIN = "# lmds-cluster begin {name}"
_END = "# lmds-cluster end {name}"


def _q(text: str) -> str:
    return shlex.quote(text)


def head_key_script(head_name: str) -> str:
    """สร้างกุญแจบน head ถ้ายังไม่มี แล้วพิมพ์ public key บรรทัดสุดท้าย"""
    comment = _q(f"lmds-cluster-{head_name}")
    return (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"([ -f {KEY_PATH} ] || ssh-keygen -q -t ed25519 -N '' -C {comment} -f {KEY_PATH}) && "
        f"chmod 600 {KEY_PATH} && cat {KEY_PATH}.pub"
    )


def authorize_script(public_key: str) -> str:
    """วาง public key ลง authorized_keys ของ worker — ไม่ซ้ำถ้ามีอยู่แล้ว"""
    key = _q(public_key)
    return (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && "
        f"(grep -qxF {key} ~/.ssh/authorized_keys || echo {key} >> ~/.ssh/authorized_keys)"
    )


def config_stanza(worker: Node, worker_ip: str) -> str:
    """stanza ของ ~/.ssh/config บน head — ครอบทั้ง IP บนสายเร็วและที่อยู่ที่ hub ใช้เข้า

    `StrictHostKeyChecking accept-new` จำเป็น: controller ใช้ BatchMode=yes ซึ่งไม่ยอมตอบ
    คำถาม host key ครั้งแรก → "Host key verification failed" ทั้งที่กุญแจถูกทุกอย่าง
    """
    hosts = [h for h in [worker_ip, *worker.all_hosts] if h]
    seen: list[str] = []
    for host in hosts:
        if host not in seen:
            seen.append(host)
    lines = [
        _BEGIN.format(name=worker.name),
        f"Host {' '.join(seen)}",
        f"  User {worker.user}",
        f"  IdentityFile {KEY_PATH}",
        "  StrictHostKeyChecking accept-new",
        "  ConnectTimeout 10",
    ]
    if worker.port and worker.port != 22:
        lines.append(f"  Port {worker.port}")
    lines.append(_END.format(name=worker.name))
    return "\n".join(lines) + "\n"


def config_script(worker: Node, worker_ip: str) -> str:
    """เขียน stanza ทับ block เดิมของ worker ตัวเดียวกัน (marker) — บรรทัดอื่นของผู้ใช้ไม่ถูกแตะ"""
    stanza = base64.b64encode(config_stanza(worker, worker_ip).encode("utf-8")).decode("ascii")
    begin = _q(_BEGIN.format(name=worker.name))
    end = _q(_END.format(name=worker.name))
    return (
        "f=~/.ssh/config; mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch \"$f\" && chmod 600 \"$f\" && "
        f"awk -v b={begin} -v e={end} '$0==b{{skip=1}} !skip{{print}} $0==e{{skip=0}}' \"$f\" > \"$f.lmds\" && "
        "mv \"$f.lmds\" \"$f\" && "
        f"echo {stanza} | base64 -d >> \"$f\" && chmod 600 \"$f\""
    )


def verify_script(worker: Node, worker_ip: str) -> str:
    """ทดสอบจาก head จริง ๆ ด้วยคำสั่งเดียวกับที่ controller ใช้ (ssh เปล่า ๆ + BatchMode)"""
    target = _q(f"{worker.user}@{worker_ip}")
    return f"ssh -o BatchMode=yes -o ConnectTimeout=8 {target} true"


def pair_workers(head: Node, workers: list[tuple[Node, str]], runner=None) -> list[dict]:
    """ให้ head เข้า worker ทุกตัวได้โดยไม่ถามรหัส — คืนรายการขั้นตอนพร้อมผล

    `workers` = [(Node ของ worker, IP ที่ controller จะใช้ = cluster IP)] · `runner` แทน
    `lmds.nodes.run` ในเทส
    """
    from . import ssh

    run = runner or ssh.run
    steps: list[dict] = []

    def step(what: str, ok: bool, detail: str = "", node: str = "") -> None:
        steps.append({"step": what, "ok": ok, "detail": detail.strip()[-300:], "node": node})

    made = run(head, head_key_script(head.name), timeout=60)
    public_key = (made.stdout or "").strip().splitlines()[-1].strip() if made.ok and made.stdout.strip() else ""
    if not made.ok or not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")):
        step(f"cluster key on {head.name} ({KEY_PATH})", False,
             (made.stderr or made.stdout or "ssh-keygen produced no public key"), head.name)
        return steps
    step(f"cluster key on {head.name} ({KEY_PATH})", True, public_key.split()[-1], head.name)

    for worker, worker_ip in workers:
        allowed = run(worker, authorize_script(public_key), timeout=60)
        step(f"authorize {head.name} on {worker.name} (~/.ssh/authorized_keys)", allowed.ok,
             "" if allowed.ok else (allowed.stderr or allowed.stdout), worker.name)
        if not allowed.ok:
            continue
        configured = run(head, config_script(worker, worker_ip), timeout=60)
        step(f"ssh config on {head.name} for {worker_ip} (~/.ssh/config)", configured.ok,
             "" if configured.ok else (configured.stderr or configured.stdout), head.name)
        if not configured.ok:
            continue
        checked = run(head, verify_script(worker, worker_ip), timeout=40)
        step(f"{head.name} → {worker.user}@{worker_ip} without a password", checked.ok,
             "" if checked.ok else (checked.stderr or checked.stdout), head.name)
    return steps
