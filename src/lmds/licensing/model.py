"""รูปแบบไฟล์ไลเซนส์และการแปลงเป็นไบต์ที่เอาไปเซ็น

ไฟล์เป็น YAML อ่านออกด้วยตาเปล่าโดยตั้งใจ — ลูกค้าองค์กรที่ audit ต้องเปิดดูได้ว่าตัวเอง
ซื้ออะไรไว้ โดยไม่ต้องเชื่อคำโปรแกรม · ลายเซ็นอยู่แยกเป็นบล็อกของตัวเอง

    license:
      id: LMDS-2026-0007
      tier: team
      licensed_to: บริษัท ตัวอย่าง จำกัด
      machines: 8              # 0 = ไม่จำกัด
      issued: 2026-09-20
      expires: 2027-09-20      # ว่าง = ไม่มีวันหมดอายุ
      contact: ops@example.com
    signature:
      key: k1
      value: <base64 ของลายเซ็น Ed25519 บน canonical JSON ของบล็อก license>

ทำไมเซ็น canonical JSON ไม่ใช่ตัว YAML
--------------------------------------
YAML ตัวเดียวกันเขียนได้หลายหน้าตา (เรียงคีย์สลับ, ใส่/ไม่ใส่ quote, comment, ตัวขึ้นบรรทัด)
ถ้าเซ็นข้อความ YAML ดิบ การจัดรูปแบบใหม่ที่ไม่เปลี่ยนความหมายเลยจะทำให้ลายเซ็นพัง แล้ว
ลูกค้าจะเจอ "ไลเซนส์ใช้ไม่ได้" ทั้งที่ไม่มีใครแก้อะไร · canonical JSON (เรียงคีย์ · ไม่มี
ช่องว่าง) ให้ไบต์ชุดเดียวจากความหมายชุดเดียว ไฟล์จึงจัดรูปแบบใหม่ได้โดยลายเซ็นยังใช้ได้
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

# ระดับที่ขายจริง — ตัวเลขคือ "เครื่องที่เสิร์ฟได้" ตาม LICENSE §1.1 · 0 = ไม่จำกัด
#
# ค่าพวกนี้เป็นเพียง "ชื่อที่ใช้เรียก" — ตัวที่บังคับใช้จริงคือ `machines` ในไฟล์ไลเซนส์
# แต่ละใบ เพื่อให้ออกใบที่ตกลงกันเป็นพิเศษได้โดยไม่ต้องแก้โค้ดและปล่อยรุ่นใหม่ทั้งฟลีต
TIERS: dict[str, int] = {
    "community": 1,      # ไม่ต้องมีไฟล์ — ตรงกับ LICENSE §2
    "trial": 0,          # ใบทดลอง มีวันหมดอายุเสมอ จำนวนเครื่องระบุในใบ
    "team": 8,
    "enterprise": 0,     # ไม่จำกัด
    "partner": 0,        # ออก sub-license ได้ (ดู LICENSE) — ไม่จำกัดเครื่อง
}

# LICENSE §2: "up to one (1) serving machine managed together ... No licence file
# is required in this mode." — ตัวเลขนี้ต้องตรงกับสัญญาเสมอ
#
# เอกสารแผน WS-7 เคยเสนอระดับ "Maker ฟรี 2 เครื่อง" (เพราะ stacked pair ใช้ 2 พอดี)
# **ยังไม่ได้ทำ** เพราะ LICENSE ที่ประกาศออกไปแล้วเขียนว่า 1 — ถ้าจะเปลี่ยนต้องแก้ LICENSE
# ก่อน แล้วค่อยแก้เลขตรงนี้ ไม่ใช่ทางกลับกัน
FREE_SERVING_MACHINES = 1


@dataclass(frozen=True)
class License:
    """เนื้อของไลเซนส์หนึ่งใบ — ตรงกับบล็อก `license:` ในไฟล์"""

    id: str
    tier: str
    licensed_to: str
    machines: int                      # 0 = ไม่จำกัด
    issued: date
    expires: date | None = None
    contact: str = ""
    note: str = ""
    # เผื่อไว้สำหรับใบที่ตกลงกันเป็นพิเศษ — ตัวโปรแกรมยังไม่ได้ใช้ แต่เซ็นรวมไปด้วยแล้ว
    # จึงเพิ่มความหมายทีหลังได้โดยใบเก่ายังตรวจผ่าน
    features: list[str] = field(default_factory=list)

    @property
    def unlimited(self) -> bool:
        return self.machines <= 0

    def expired_on(self, today: date) -> bool:
        return self.expires is not None and today > self.expires

    def days_left(self, today: date) -> int | None:
        return None if self.expires is None else (self.expires - today).days

    def payload(self) -> dict:
        """dict ที่เอาไปทำ canonical JSON — ต้องตรงกับที่เขียนลงไฟล์เป๊ะ"""
        out: dict = {
            "id": self.id,
            "tier": self.tier,
            "licensed_to": self.licensed_to,
            "machines": int(self.machines),
            "issued": self.issued.isoformat(),
        }
        # ใส่เฉพาะที่มีค่า — ฟิลด์ที่ไม่ได้ใช้ไม่ควรโผล่ในไบต์ที่เซ็น ไม่งั้นการเพิ่มฟิลด์
        # ใหม่ในอนาคตจะทำให้ใบเก่าที่ไม่มีฟิลด์นั้นตรวจไม่ผ่าน
        if self.expires is not None:
            out["expires"] = self.expires.isoformat()
        if self.contact:
            out["contact"] = self.contact
        if self.note:
            out["note"] = self.note
        if self.features:
            out["features"] = list(self.features)
        return out

    def signing_bytes(self) -> bytes:
        return canonical(self.payload())


def canonical(payload: dict) -> bytes:
    """ไบต์ชุดเดียวจากความหมายชุดเดียว — เรียงคีย์ ไม่มีช่องว่าง UTF-8 ไม่ escape

    `ensure_ascii=False` เพราะชื่อลูกค้าเป็นภาษาไทยได้ และเราอยากให้ไบต์ที่เซ็นตรงกับ
    ที่คนอ่านเห็นในไฟล์ · ต้อง encode เป็น utf-8 เองเพื่อให้ผลเท่ากันทุกแพลตฟอร์ม
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _as_date(value, fieldname: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{fieldname} ต้องเป็นวันที่แบบ YYYY-MM-DD — ได้ {value!r}") from exc


def from_payload(payload: dict) -> License:
    """สร้าง License จาก dict ที่อ่านมาจากไฟล์ — โยน ValueError เมื่อรูปแบบไม่ถูก"""
    if not isinstance(payload, dict):
        raise ValueError("บล็อก license ต้องเป็น mapping")
    missing = [k for k in ("id", "tier", "licensed_to", "machines", "issued") if k not in payload]
    if missing:
        raise ValueError(f"บล็อก license ขาดฟิลด์: {', '.join(missing)}")
    try:
        machines = int(payload["machines"])
    except (TypeError, ValueError) as exc:
        raise ValueError("machines ต้องเป็นจำนวนเต็ม (0 = ไม่จำกัด)") from exc
    expires = payload.get("expires")
    return License(
        id=str(payload["id"]),
        tier=str(payload["tier"]),
        licensed_to=str(payload["licensed_to"]),
        machines=machines,
        issued=_as_date(payload["issued"], "issued"),
        expires=_as_date(expires, "expires") if expires else None,
        contact=str(payload.get("contact") or ""),
        note=str(payload.get("note") or ""),
        features=[str(f) for f in (payload.get("features") or [])],
    )
