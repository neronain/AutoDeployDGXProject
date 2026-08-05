"""ทะเบียนเครื่องที่ hub คุมอยู่ — `~/.config/lmds/nodes.yaml` (สิทธิ์ 0600)

**ไม่เก็บรหัสผ่านเด็ดขาด** — password ใช้ครั้งเดียวตอนติดตั้ง SSH key แล้วทิ้งทันที
ทะเบียนเก็บแค่ว่าเครื่องอยู่ที่ไหนและ login ด้วย user อะไร ตัวที่ใช้เข้าจริงคือ key ของ LMDS เอง
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from lmds.config.paths import config_dir, ensure_config_dir

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


class NodeError(Exception):
    pass


@dataclass
class Node:
    name: str
    host: str
    user: str
    port: int = 22
    # ไม่มีฟิลด์ password โดยตั้งใจ — ดู docstring ของโมดูล
    note: str = ""
    added_at: str = ""
    last_seen: str = ""
    last_error: str = ""
    lmds_version: str = ""
    labels: list[str] = field(default_factory=list)
    # เส้นทางที่ใช้ "คุยกันตอน stacked" — คนละเส้นกับ host ที่ใช้ SSH โดยตั้งใจ
    # NCCL ต้องยิงผ่านการ์ดเร็ว (ConnectX/200G) ไม่ใช่สายบริหารจัดการ
    cluster_ip: str = ""
    cluster_iface: str = ""
    # ที่อยู่สำรองของ "เครื่องเดียวกัน" — เช่น Tailscale/VPN ที่ใช้ตอนออกนอกออฟฟิศ
    # ต่างจากการเปลี่ยน host (= คนละเครื่อง) ตรงที่นี่คือทางเข้าอีกทางของเครื่องเดิม
    alt_hosts: list[str] = field(default_factory=list)

    @property
    def all_hosts(self) -> list[str]:
        return [self.host, *[h for h in self.alt_hosts if h and h != self.host]]

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def nodes_file() -> Path:
    return config_dir() / "nodes.yaml"


def load() -> list[Node]:
    path = nodes_file()
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise NodeError(f"อ่าน {path} ไม่ได้: {exc}") from exc
    entries = raw.get("nodes") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    out: list[Node] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        known = {f: entry.get(f) for f in Node.__dataclass_fields__ if f in entry}
        known.setdefault("host", "")
        known.setdefault("user", "")
        out.append(Node(**known))
    return out


def save(nodes: list[Node]) -> Path:
    ensure_config_dir()
    path = nodes_file()
    payload = {"nodes": [asdict(n) for n in nodes]}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)  # มีชื่อ user/host ของเครื่องภายใน — ไม่ควรให้ user อื่นอ่าน
    return path


def validate_cluster_ip(value: str) -> str:
    """cluster IP ต้องเป็น IPv4 ที่ใช้ได้จริง — พิมพ์ผิดตรงนี้ทำให้ stacked ค้างตอน NCCL init"""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise NodeError(f"cluster IP '{value}' ไม่ถูกต้อง: {exc}") from exc
    if address.version != 4:
        raise NodeError("รองรับ IPv4 เท่านั้นสำหรับ cluster IP (NCCL/RoCE ใช้ IPv4)")
    if address.is_loopback or address.is_multicast:
        raise NodeError(f"cluster IP '{value}' ใช้ไม่ได้ — ต้องเป็นที่อยู่บนสายจริงระหว่างเครื่อง")
    return value


def find(name: str) -> Node | None:
    return next((n for n in load() if n.name == name), None)


def add(node: Node) -> Node:
    if not _NAME_RE.match(node.name):
        raise NodeError(
            f"ชื่อ '{node.name}' ใช้ไม่ได้ — ใช้ a-z 0-9 . _ - เท่านั้น (ขึ้นต้นด้วยตัวอักษรหรือตัวเลข)"
        )
    if not node.host or not node.user:
        raise NodeError("ต้องระบุทั้ง host และ user")
    node.cluster_ip = validate_cluster_ip(node.cluster_ip)
    existing = load()
    if any(n.name == node.name for n in existing):
        raise NodeError(f"มีเครื่องชื่อ '{node.name}' อยู่แล้ว — ลบก่อนหรือใช้ชื่ออื่น")
    clash = next((n for n in existing if n.host == node.host and n.user == node.user), None)
    if clash is not None:
        raise NodeError(f"{node.target} ถูกเพิ่มไว้แล้วในชื่อ '{clash.name}'")
    existing.append(node)
    save(existing)
    return node


def remove(name: str) -> Node:
    existing = load()
    target = next((n for n in existing if n.name == name), None)
    if target is None:
        raise NodeError(f"ไม่รู้จักเครื่อง '{name}' — ดูรายชื่อ: lmds node list")
    save([n for n in existing if n.name != name])
    return target


def update(name: str, **changes) -> Node:
    """อัปเดตฟิลด์สถานะ (last_seen / last_error / lmds_version) — ไม่แตะ host/user"""
    existing = load()
    target = next((n for n in existing if n.name == name), None)
    if target is None:
        raise NodeError(f"ไม่รู้จักเครื่อง '{name}'")
    for key, value in changes.items():
        if key in {"name", "host", "user", "port"}:
            continue  # เปลี่ยนที่อยู่ต้องลบแล้วเพิ่มใหม่ ไม่ใช่แก้เงียบ ๆ
        if key == "cluster_ip":
            value = validate_cluster_ip(value)
        if key in Node.__dataclass_fields__:
            setattr(target, key, value)
    save(existing)
    return target


def suggest_name(host: str, taken: set[str] | None = None) -> str:
    """ตั้งชื่อเริ่มต้นจาก host — 10.0.0.5 → node-10-0-0-5, spark1.local → spark1"""
    taken = taken or {n.name for n in load()}
    base = host.split(".")[0] if not host.replace(".", "").isdigit() else "node-" + host.replace(".", "-")
    base = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-.") or "node"
    if base not in taken:
        return base
    for i in range(2, 100):
        candidate = f"{base}-{i}"
        if candidate not in taken:
            return candidate
    raise NodeError("ตั้งชื่ออัตโนมัติไม่ได้ — ระบุ --name เอง")
