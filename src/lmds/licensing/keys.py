"""กุญแจสาธารณะที่ใช้ตรวจไลเซนส์ — ฝังมากับโปรแกรม

ทำไมฝังสองดอก
-------------
`k1` คือดอกที่เซ็นอยู่ตอนนี้ · `k2` ว่างไว้สำหรับวันที่ต้องหมุนกุญแจ

ถ้าฝังดอกเดียวแล้ววันหนึ่งต้องเปลี่ยน (กุญแจหลุด หรือถึงรอบเปลี่ยนตามนโยบาย) ใบที่เซ็น
ด้วยดอกใหม่จะตรวจไม่ผ่านบนทุกเครื่องที่ยังไม่อัปเดต — แปลว่าต้องบังคับให้ลูกค้าทั้งฟลีต
อัปเกรดพร้อมกันก่อนถึงจะต่ออายุได้ ซึ่งบนเครื่อง air-gapped คือเรื่องใหญ่ · ฝัง k2 ไว้
ตั้งแต่ต้นทำให้เปลี่ยนไปเซ็นด้วย k2 ได้ทันทีโดยเครื่องที่มีรุ่นนี้อยู่แล้วตรวจผ่านเลย

**ในนี้มีแต่กุญแจสาธารณะ** — เอาไปตรวจลายเซ็นได้อย่างเดียว ปลอมลายเซ็นไม่ได้
private key อยู่นอกรีโปที่ `~/.config/lmds/license-signing-<id>.key` (0600) บนเครื่อง
ที่ออกใบเท่านั้น และอยู่ใน .gitignore

ถ้าวันหนึ่งต้องถอนกุญแจดอกหนึ่ง ให้เอา id ของมันออกจาก ACTIVE แล้วปล่อยรุ่นใหม่ —
ใบที่เซ็นด้วยดอกนั้นจะตรวจไม่ผ่านทันทีที่เครื่องอัปเดต
"""

from __future__ import annotations

import base64

# id → public key (base64 ของ 32 ไบต์ดิบ)
PUBLIC_KEYS: dict[str, str] = {
    "k1": "X7oRD7Y9UjdI+ml+JyGkznGhke3N6sex5uAnCAAp7dI=",
    "k2": "ZNuN5vSBanFlqWF540MKuF3H+KgTEPuCizs83FgL/9I=",
}

# ดอกที่ยอมรับตอนนี้ — ถอนดอกไหนก็เอาออกจากชุดนี้
ACTIVE: frozenset[str] = frozenset({"k1", "k2"})


def public_key(key_id: str) -> bytes | None:
    """คืนกุญแจสาธารณะ 32 ไบต์ของ id นี้ — None เมื่อไม่รู้จักหรือถูกถอนแล้ว"""
    if key_id not in ACTIVE:
        return None
    encoded = PUBLIC_KEYS.get(key_id)
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    return raw if len(raw) == 32 else None
