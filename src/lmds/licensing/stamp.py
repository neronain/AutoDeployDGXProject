"""ตราประทับบน bundle — บอกว่า bundle นี้ถูก generate บนเครื่องที่ถือไลเซนส์ใบไหน

**ไม่ใช่การล็อก** · `LICENSE` ให้สิทธิ์ bundle กับคนที่ generate มันอยู่แล้ว และคนใช้ฟรี
ก็ generate ได้ไม่ต่างกัน · ตรานี้มีไว้ตอบคำถามเดียว: *bundle ก้อนนี้มาจากใคร*

ความเสี่ยงเชิงพาณิชย์ตัวจริงไม่ใช่นักศึกษาที่ลบ `if` ทิ้ง แต่คือ **พาร์ตเนอร์ A เอา bundle
ไปขายต่อในนาม B** ซึ่งเป็นเงินก้อนใหญ่และเกิดเงียบ ๆ · ตรานี้ทำให้ตรวจได้ทันที

ทำไมแก้ชื่อในตราไม่ได้
----------------------
ตราไม่ได้เก็บแค่ชื่อ แต่เก็บ **ลายเซ็นของไลเซนส์ใบนั้นทั้งดวง** (ซึ่งเซ็นด้วยกุญแจของเรา
บนเครื่องของเรา) · ใครแก้ `licensed_to` ในไฟล์ ลายเซ็นจะไม่ตรงกับเนื้อทันที และ
`lmds validate` จับได้ตั้งแต่ gate

สิ่งที่ตรานี้ **ไม่ได้** อ้าง
--------------------------
- ไม่ได้บอกว่า bundle ถูกรันที่ไหน — ก๊อปไปวางเครื่องอื่นได้ และเราไม่ได้ห้าม
- ไม่ได้ผูกกับฮาร์ดแวร์ — เปลี่ยน mainboard แล้วลูกค้าโทรมาด่าไม่คุ้มกับที่ได้
- ไม่ได้กันการก๊อปตรา — ก๊อปทั้งก้อนได้ แต่จะยังเป็นชื่อเจ้าของเดิมติดไปด้วย
  ซึ่งก็คือสิ่งที่เราต้องการพอดี

bundle ที่ generate จากเครื่องโหมดฟรีจะมีตราที่บอกว่า `community` ตรง ๆ ไม่มีลายเซ็น
— ไม่ใช่ความผิด และ gate ปล่อยผ่าน
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

from lmds.licensing import ed25519, keys
from lmds.licensing.model import canonical
from lmds.licensing.store import Status

FIELD = "origin"


def build(status: Status, *, lmds_version: str = "") -> dict:
    """ก้อน `origin:` ที่จะเขียนลง MODEL_PROFILE.yaml"""
    stamped = {
        "stamped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lmds_version": lmds_version,
    }
    lic = status.license
    if status.state != "active" or lic is None:
        # โหมดฟรี (หรือใบหมดอายุ/ใช้ไม่ได้) — บอกตรง ๆ ว่าไม่มีใบ ไม่ต้องแกล้งว่ามี
        stamped["tier"] = "community"
        stamped["licensed_to"] = None
        stamped["license_id"] = None
        return stamped

    stamped["tier"] = lic.tier
    stamped["licensed_to"] = lic.licensed_to
    stamped["license_id"] = lic.id
    # เก็บ payload ที่ถูกเซ็น + ลายเซ็น เพื่อให้ตรวจซ้ำได้โดยไม่ต้องมีไฟล์ไลเซนส์ตัวจริง
    # (ผู้รับ bundle ไม่มีไฟล์นั้นอยู่แล้ว — นั่นคือประเด็นทั้งหมดของการมีตรา)
    stamped["attestation"] = {
        "key": status.signature_key or "",
        "payload": lic.payload(),
        "signature": status.signature_value or "",
    }
    return stamped


def verify(stamped: dict | None) -> tuple[bool, str]:
    """ตรวจตราหนึ่งดวง — คืน (ผ่านไหม, เหตุผลที่คนอ่านได้)

    ไม่มีตรา = ผ่าน (bundle เก่าที่ generate ก่อนมีฟีเจอร์นี้ ต้องไม่กลายเป็นของเสีย)
    ตรา community = ผ่าน (เครื่องโหมดฟรี generate เองได้ตามสัญญา)
    มี attestation = ต้องตรวจผ่านจริง ๆ ไม่งั้นคือถูกแก้
    """
    if not stamped:
        return True, "ไม่มีตรา (bundle generate ก่อนมีฟีเจอร์นี้)"
    if not isinstance(stamped, dict):
        return False, "ก้อน origin ผิดรูปแบบ"

    attestation = stamped.get("attestation")
    if not attestation:
        tier = stamped.get("tier") or "community"
        if stamped.get("licensed_to"):
            # อ้างชื่อเจ้าของแต่ไม่มีลายเซ็นมายืนยัน = อาการของการเติมชื่อเข้าไปเอง
            return False, (f"ตราอ้างว่าเป็นของ {stamped['licensed_to']!r} แต่ไม่มี attestation "
                           f"มายืนยัน — ตราที่ออกโดย lmds จริงจะมีลายเซ็นมาด้วยเสมอ")
        return True, f"ตรา {tier} (ไม่มีไลเซนส์ — ถูกต้องสำหรับโหมดฟรี)"

    if not isinstance(attestation, dict):
        return False, "ก้อน attestation ผิดรูปแบบ"

    payload = attestation.get("payload")
    if not isinstance(payload, dict):
        return False, "attestation ไม่มี payload ของไลเซนส์"

    public = keys.public_key(str(attestation.get("key") or ""))
    if public is None:
        return False, (f"attestation เซ็นด้วยกุญแจ {attestation.get('key')!r} ที่ lmds รุ่นนี้ไม่รู้จัก "
                       f"หรือถูกถอนไปแล้ว")
    try:
        signature = base64.b64decode(str(attestation.get("signature") or ""), validate=True)
    except Exception:
        return False, "ลายเซ็นใน attestation ไม่ใช่ base64 ที่ถูกต้อง"

    if not ed25519.verify(public, canonical(payload), signature):
        return False, ("ลายเซ็นใน attestation ไม่ตรงกับ payload — ตรานี้ถูกแก้หลัง generate")

    # ชื่อที่โชว์ด้านบนต้องตรงกับ payload ที่เซ็นไว้ ไม่งั้นคนอ่านจะเชื่อบรรทัดที่แก้ได้
    for field, source in (("licensed_to", "licensed_to"), ("license_id", "id"), ("tier", "tier")):
        shown, signed = stamped.get(field), payload.get(source)
        if shown is not None and str(shown) != str(signed):
            return False, (f"{field} ที่เขียนไว้ ({shown!r}) ไม่ตรงกับในลายเซ็น ({signed!r}) "
                           f"— มีคนแก้ชื่อเจ้าของ")

    return True, f"ออกให้ {payload.get('licensed_to')} · {payload.get('id')}"


def describe(stamped: dict | None) -> str:
    """บรรทัดเดียวสำหรับหัว controller"""
    if not stamped or not isinstance(stamped, dict):
        return "unlicensed build"
    who = stamped.get("licensed_to")
    if not who:
        return f"{stamped.get('tier') or 'community'} build (ไม่มีไลเซนส์ผูกกับ bundle นี้)"
    return f"{who} · {stamped.get('license_id')} · {stamped.get('tier')}"
