"""อ่านไฟล์ไลเซนส์จากดิสก์แล้วบอกว่าตอนนี้อยู่ในสถานะไหน

สถานะมีสี่แบบเท่านั้น — ทุกจุดในโปรแกรมตัดสินใจจากสี่ค่านี้ ไม่ต้องตีความไฟล์เอง

    free     ไม่มีไฟล์ไลเซนส์ = โหมดฟรีตาม LICENSE §2 (1 เครื่องที่เสิร์ฟได้)
             **ไม่ใช่ error** และไม่ควรมีข้อความบ่นใด ๆ — คนส่วนใหญ่อยู่โหมดนี้ถาวร
    active   มีไฟล์ ลายเซ็นถูก ยังไม่หมดอายุ
    expired  ลายเซ็นถูก แต่เลยวันหมดอายุแล้ว → อ่านอย่างเดียว ไม่ใช่หยุดทำงาน
    invalid  ไฟล์เสีย/ลายเซ็นไม่ผ่าน/กุญแจไม่รู้จัก → ตกกลับไปสิทธิ์เท่าโหมดฟรี

เรื่องสำคัญ: `invalid` **ไม่เคยทำให้โปรแกรมหยุด** มันแค่ทำให้สิทธิ์เท่าคนที่ไม่มีไฟล์เลย
เพราะไฟล์พังไม่ควรทำให้เครื่องที่กำลังเสิร์ฟลูกค้าอยู่ใช้ไม่ได้ (ดูกฎข้อ 2 ใน docs/LICENSING.md)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lmds.licensing import ed25519, keys
from lmds.licensing.model import FREE_SERVING_MACHINES, License, from_payload

FILENAME = "license.yaml"


def license_path() -> Path:
    """ที่อยู่ของไฟล์ไลเซนส์ — เคารพ LMDS_CONFIG_DIR เหมือนไฟล์ตั้งค่าอื่นของ LMDS"""
    from lmds.config.paths import config_dir

    return config_dir() / FILENAME


@dataclass(frozen=True)
class Status:
    """คำตอบเดียวที่ทุกที่ในโปรแกรมใช้ — อย่าให้ที่อื่นไปตีความไฟล์เอง"""

    state: str                     # free | active | expired | invalid
    license: License | None = None
    reason: str = ""               # ทำไมถึงเป็นสถานะนี้ (ภาษาคน ใช้โชว์ได้เลย)
    path: Path | None = None

    @property
    def machines_allowed(self) -> int:
        """จำนวนเครื่องที่เสิร์ฟได้ซึ่งใบนี้ครอบคลุม — 0 = ไม่จำกัด

        `expired` และ `invalid` ได้สิทธิ์เท่าโหมดฟรี ไม่ใช่ศูนย์ — ศูนย์แปลว่า
        "ห้ามรันอะไรเลย" ซึ่งไม่เคยเป็นสิ่งที่เราต้องการกับลูกค้าที่จ่ายเงินมาแล้ว
        """
        if self.state == "active" and self.license is not None:
            return self.license.machines
        return FREE_SERVING_MACHINES

    @property
    def unlimited(self) -> bool:
        return self.state == "active" and self.license is not None and self.license.unlimited

    @property
    def read_only(self) -> bool:
        """หมดอายุ = ของที่รันอยู่ยังรันต่อ แต่ขยายฟลีตเพิ่มไม่ได้"""
        return self.state == "expired"

    def describe(self) -> str:
        if self.state == "free":
            return f"โหมดฟรี — {FREE_SERVING_MACHINES} เครื่องที่เสิร์ฟได้ (ไม่ต้องมีไฟล์ไลเซนส์)"
        if self.license is None:
            return self.reason or self.state
        lic = self.license
        limit = "ไม่จำกัด" if lic.unlimited else f"{lic.machines} เครื่อง"
        return f"{lic.tier} · {lic.licensed_to} · {limit} · {self.reason}".strip(" ·")


def _read_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("ไฟล์ไลเซนส์ต้องเป็น mapping ที่มีบล็อก license: และ signature:")
    return loaded


def verify_document(document: dict, *, today: date | None = None) -> Status:
    """ตรวจเนื้อไฟล์ที่อ่านมาแล้ว — แยกจาก load() เพื่อให้เทสและตัวเซ็นเรียกใช้ซ้ำได้"""
    today = today or date.today()
    block = document.get("license")
    signature_block = document.get("signature") or {}
    if not isinstance(signature_block, dict):
        return Status("invalid", reason="บล็อก signature ผิดรูปแบบ")

    try:
        lic = from_payload(block)
    except ValueError as exc:
        return Status("invalid", reason=str(exc))

    key_id = str(signature_block.get("key") or "")
    public = keys.public_key(key_id)
    if public is None:
        return Status("invalid", license=lic,
                      reason=f"ไม่รู้จักกุญแจ {key_id!r} — ไฟล์นี้อาจเซ็นด้วยกุญแจที่ถอนไปแล้ว "
                             f"หรือ lmds รุ่นนี้เก่ากว่ากุญแจที่ใช้เซ็น")

    import base64

    try:
        raw_signature = base64.b64decode(str(signature_block.get("value") or ""), validate=True)
    except Exception:
        return Status("invalid", license=lic, reason="ลายเซ็นไม่ใช่ base64 ที่ถูกต้อง")

    if not ed25519.verify(public, lic.signing_bytes(), raw_signature):
        return Status("invalid", license=lic,
                      reason="ลายเซ็นไม่ตรงกับเนื้อไลเซนส์ — ไฟล์ถูกแก้หลังออกใบ "
                             "หรือก๊อปมาไม่ครบ")

    if lic.expired_on(today):
        return Status("expired", license=lic,
                      reason=f"หมดอายุเมื่อ {lic.expires.isoformat()}")

    left = lic.days_left(today)
    reason = "ไม่มีวันหมดอายุ" if left is None else f"เหลืออีก {left} วัน"
    return Status("active", license=lic, reason=reason)


def load(path: Path | None = None, *, today: date | None = None) -> Status:
    """อ่าน + ตรวจไฟล์ไลเซนส์ — ไม่เคยโยน exception ออกไปหาผู้เรียก

    ทุกคำสั่งของ lmds เรียกทางนี้ได้โดยไม่ต้องดักอะไร · ไฟล์หาย/พัง/สิทธิ์ไม่พอ ล้วนกลาย
    เป็น Status ที่อธิบายตัวเองได้ ไม่ใช่ traceback กลางงานของผู้ใช้
    """
    target = path or license_path()
    try:
        if not target.exists():
            return Status("free", reason="ไม่พบไฟล์ไลเซนส์", path=target)
    except OSError as exc:
        return Status("free", reason=f"อ่านที่อยู่ไฟล์ไลเซนส์ไม่ได้: {exc}", path=target)

    try:
        document = _read_yaml(target)
    except Exception as exc:
        return Status("invalid", reason=f"อ่านไฟล์ไลเซนส์ไม่ได้: {exc}", path=target)

    status = verify_document(document, today=today)
    return Status(status.state, status.license, status.reason, target)


def permissions_warning(path: Path | None = None) -> str:
    """เตือนเมื่อไฟล์ไลเซนส์คนอื่นอ่านได้ — เป็นคำเตือน ไม่ใช่ตัวบล็อก

    ไฟล์นี้ไม่ใช่ความลับที่ทำให้ระบบพังถ้าหลุด (เป็นใบอนุญาตที่มีลายเซ็น ไม่ใช่กุญแจ)
    แต่มีชื่อลูกค้าและเงื่อนไขทางการค้าอยู่ จึงไม่ควรให้ทุกคนบนเครื่องอ่านได้
    """
    target = path or license_path()
    try:
        if not target.exists():
            return ""
        mode = target.stat().st_mode
    except OSError:
        return ""
    if mode & (0o077):
        return (f"{target} เปิดให้ผู้ใช้อื่นบนเครื่องนี้อ่านได้ "
                f"(โหมด {oct(mode & 0o777)}) — แนะนำ chmod 600 {target}")
    return ""


def install(document_text: str, path: Path | None = None) -> Status:
    """เขียนไฟล์ไลเซนส์ลงเครื่องนี้ — ตรวจก่อนเขียนเสมอ

    ตรวจก่อนเพื่อไม่ให้ไฟล์ที่ใช้ไม่ได้ไปทับใบที่ยังดีอยู่ · เคสจริงที่กลัว: ลูกค้าวางใบ
    ต่ออายุที่ก๊อปมาไม่ครบทับใบเดิมตอนตี 3 แล้วเช้ามาทั้งฟลีตกลายเป็นโหมดฟรี
    """
    import yaml

    target = path or license_path()
    try:
        document = yaml.safe_load(document_text)
    except Exception as exc:
        return Status("invalid", reason=f"ไฟล์ที่ให้มาไม่ใช่ YAML ที่อ่านได้: {exc}", path=target)
    if not isinstance(document, dict):
        return Status("invalid", reason="ไฟล์ที่ให้มาไม่มีบล็อก license:/signature:", path=target)

    status = verify_document(document)
    if status.state == "invalid":
        return Status("invalid", status.license, status.reason, target)

    target.parent.mkdir(parents=True, exist_ok=True)
    # เขียนไฟล์ใหม่ด้วยสิทธิ์ 0600 ตั้งแต่ตอนสร้าง — เขียนก่อนแล้ว chmod ทีหลังจะมีช่วงสั้น ๆ
    # ที่ไฟล์เปิดให้คนอื่นอ่านได้
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(document_text if document_text.endswith("\n") else document_text + "\n")
    os.chmod(target, 0o600)
    return Status(status.state, status.license, status.reason, target)
