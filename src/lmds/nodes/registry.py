"""ทะเบียนเครื่องที่ hub คุมอยู่ — `~/.config/lmds/nodes.yaml` (สิทธิ์ 0600)

**ไม่เก็บรหัสผ่านเด็ดขาด** — password ใช้ครั้งเดียวตอนติดตั้ง SSH key แล้วทิ้งทันที
ทะเบียนเก็บแค่ว่าเครื่องอยู่ที่ไหนและ login ด้วย user อะไร ตัวที่ใช้เข้าจริงคือ key ของ LMDS เอง
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from lmds.config.paths import config_dir, ensure_config_dir, write_atomic

NAME_MAX = 63
_NAME_EXTRA = "._-"


def name_ok(name: str) -> bool:
    """ชื่อเครื่องใช้ได้ไหม

    ชื่อเป็นของผู้ใช้ตั้ง — ไม่ควรบังคับให้พิมพ์ตัวเล็กทั้งที่ป้ายบนเครื่องเขียน "MSI6"
    รับตัวอักษรของภาษาไหนก็ได้ (ไทยใช้ได้) ตัวเลข และ `. _ -`

    ที่ยัง **ไม่** รับ: ช่องว่าง (พังตอน `lmds node run <ชื่อ> …`) และอักขระที่ shell ตีความ
    — ชื่อถูกต่อเป็นคำสั่งจริง จึงต้องปลอดภัยตั้งแต่ตอนรับเข้า ไม่ใช่หวังพึ่ง quote ปลายทาง

    ใช้ฟังก์ชันแทน regex เพราะ ``\\w`` ของ Python ไม่นับสระบน/ล่างของไทย (Unicode category M)
    ทำให้ "ปลาย-01" ผ่านแต่ "เครื่องหลัก" ตก ซึ่งอธิบายให้ผู้ใช้ไม่ได้
    """
    if not name or len(name) > NAME_MAX:
        return False
    if not name[0].isalnum():
        return False
    return all(ch.isalnum() or ch in _NAME_EXTRA or unicodedata.category(ch).startswith("M")
               for ch in name)


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
    # IP ที่ "เครื่องนั้น" รายงานว่าตัวเองถืออยู่ — คนละอย่างกับ host ที่ใช้ SSH เข้าไป
    # host เป็นชื่อได้ (`orb`, `spark1.local`, ชื่อบน Tailscale) และเป็นที่อยู่ที่ hub
    # มองเห็น ไม่ใช่ที่อยู่บนวงของเครื่องนั้น · เก็บไว้เพื่อให้รายชื่อเครื่องบอก IP ได้
    # ทันทีโดยไม่ต้อง SSH ใหม่ และยังบอกได้ตอนเครื่องดับ (ค่าล่าสุดที่เคยเห็น)
    local_ip: str = ""
    # ป้ายจัดกลุ่มตามที่ตั้งเครื่อง (เช่น ชื่อไซต์/ลูกค้า) — ใช้ "แสดงผลและกรอง" อย่างเดียว
    # ตั้งแต่ 2026-08-31 ฟิลด์นี้ **เป็นตัวบังคับ** ตอนจับกลุ่ม stacked ด้วย ไม่ใช่แค่ป้าย
    # แสดงผลอย่างเดิม — stacked ข้ามไซต์ทำไม่ได้จริง (NCCL วิ่งบนสายในแร็ค ไม่ใช่ผ่าน WAN)
    # และการปล่อยให้เครื่องคนละที่มาอยู่ถังเดียวกันทำให้คู่ที่ตั้ง IP ครบแล้วถูกคู่อื่นแย่งที่ไป
    site: str = ""
    # แบ่ง "หลายคลัสเตอร์ในไซต์เดียวกัน" ด้วยมือ — ว่าง = ให้ระบบแบ่งเองตาม subnet
    #
    # ระบบแบ่งอัตโนมัติได้เฉพาะตอนที่แต่ละคู่อยู่คนละวง · เครื่องรุ่นเดียวกันสี่เครื่องบนวง
    # เดียวกันจะถูกมองเป็นก้อนเดียว TP=4 ซึ่งบางทีไม่ใช่สิ่งที่ต้องการ (อยากได้สองคู่แยกกัน
    # เพื่อรันคนละโมเดล/ทำ failover) · ตั้งชื่อเดียวกันให้เครื่องที่ต้องการอยู่คลัสเตอร์เดียวกัน
    cluster_name: str = ""
    labels: list[str] = field(default_factory=list)
    # เส้นทางที่ใช้ "คุยกันตอน stacked" — คนละเส้นกับ host ที่ใช้ SSH โดยตั้งใจ
    # NCCL ต้องยิงผ่านการ์ดเร็ว (ConnectX/200G) ไม่ใช่สายบริหารจัดการ
    cluster_ip: str = ""
    cluster_iface: str = ""
    # ที่อยู่สำรองของ "เครื่องเดียวกัน" — เช่น Tailscale/VPN ที่ใช้ตอนออกนอกออฟฟิศ
    # ต่างจากการเปลี่ยน host (= คนละเครื่อง) ตรงที่นี่คือทางเข้าอีกทางของเครื่องเดิม
    alt_hosts: list[str] = field(default_factory=list)
    # ยอมให้เครื่องนี้ถูกจับกลุ่ม stacked ไหม — กลุ่มเป็นสิ่งที่ระบบ "เสนอ" จากฮาร์ดแวร์ที่ตรงกัน
    # ไม่ใช่สิ่งที่ประกาศไว้ เครื่องที่ตั้งใจให้รันงานของตัวเองจึงต้องปิดได้ ไม่งั้นพอวันหนึ่ง
    # มันถูกตั้ง IP บนวงเดียวกันก็จะเด้งเข้ากลุ่มเองแล้วเปลี่ยนแผน parallel ของทั้งกลุ่ม
    stack: bool = True

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
        # `stack: null` ในไฟล์ที่แก้ด้วยมือต้องแปลว่า "ค่าเริ่มต้น" ไม่ใช่ "ห้าม stacked"
        if known.get("stack") is None:
            known.pop("stack", None)
        out.append(Node(**known))
    return out


def save(nodes: list[Node]) -> Path:
    ensure_config_dir()
    payload = {"nodes": [asdict(n) for n in nodes]}
    # เขียนแบบ atomic — สองคำขอจากหน้าเว็บที่บันทึกพร้อมกันเคยเขียนทับกันกลางคันจนไฟล์พัง
    # สิทธิ์ 0600 เพราะมีชื่อ user/host ของเครื่องภายใน — ไม่ควรให้ user อื่นอ่าน
    return write_atomic(nodes_file(),
                        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


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
    if not name_ok(node.name):
        raise NodeError(
            f"ชื่อ '{node.name}' ใช้ไม่ได้ — ใช้ตัวอักษร ตัวเลข และ . _ - "
            f"(ขึ้นต้นด้วยตัวอักษรหรือตัวเลข · ห้ามมีช่องว่าง · ยาวไม่เกิน 63 ตัว)"
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
        if key == "stack":
            value = bool(value)
        if key in Node.__dataclass_fields__:
            setattr(target, key, value)
    save(existing)
    return target


def status_from_probe(info: dict) -> dict:
    """ฟิลด์สถานะที่ทะเบียนจำจาก `lmds agent info` — จุดเดียวที่ทุกคนเรียกใช้

    มีที่ probe อยู่หกจุด (CLI สามที่, หน้าเว็บสองที่, ตัว refresh เบื้องหลังอีกหนึ่ง)
    ก่อนหน้านี้ต่างคนต่างหยิบคีย์เอง — เพิ่มฟิลด์ใหม่ทีหนึ่งต้องไล่แก้ให้ครบทุกจุด และจุดที่
    ลืมจะเงียบ ไม่ใช่พัง · ผลคือเครื่องเดียวกันแสดง IP บ้างไม่แสดงบ้าง แล้วแต่ว่ารอบล่าสุด
    ใครเป็นคนอัปเดต
    """
    host = info.get("host") or {}
    fields = {
        "lmds_version": host.get("lmds_version") or "",
        "local_ip": host.get("ip") or "",
    }
    # คีย์ที่ปลายทางไม่ได้ส่งมาแปลว่า "ไม่รู้" ไม่ใช่ "ไม่มี" — เขียนทับด้วยค่าว่างคือทิ้งของ
    # ที่เคยรู้จริงไปเพราะ node รุ่นเก่ารุ่นเดียวที่ยังไม่ส่งฟิลด์นั้น
    return {key: value for key, value in fields.items() if value}


def in_saved_order(nodes: list[Node], order: list[str]) -> list[Node]:
    """เรียงเครื่องตามลำดับที่ผู้ใช้จัดไว้เอง (ลากในหน้าเว็บ) — ที่เหลือต่อท้ายตามเดิม

    ลำดับที่เก็บไว้กับทะเบียนไม่จำเป็นต้องตรงกัน: เครื่องที่เพิ่งเพิ่มยังไม่มีในลำดับ และ
    เครื่องที่ลบไปแล้วยังค้างชื่ออยู่ — ทั้งสองกรณีต้องไม่ทำให้ลิสต์หายหรือซ้ำ
    """
    rank = {name: index for index, name in enumerate(order)}
    return sorted(nodes, key=lambda node: rank.get(node.name, len(rank)))


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
