"""ระบบไลเซนส์ของ LMDS — ไฟล์ลงนามออฟไลน์ล้วน ไม่มี phone-home

ออกแบบตาม LICENSE ที่ประกาศไปแล้ว: เครื่องที่เสิร์ฟได้ 1 เครื่อง (บวก control plane
กี่เครื่องก็ได้) ใช้ฟรีทุกวัตถุประสงค์รวมเชิงพาณิชย์ **โดยไม่ต้องมีไฟล์ไลเซนส์เลย**
ตั้งแต่ 2 เครื่องที่บริหารร่วมกันขึ้นไปจึงต้องมีใบ

ไม่มีการเชื่อมต่อออกไปไหนทั้งสิ้น — ตรวจลายเซ็น Ed25519 บนเครื่องนั้นเอง เพราะ
`SECURITY.md` ของเราสัญญาว่าไม่มี telemetry และตลาด DGX ในองค์กรคือตลาด air-gapped

อ่านลำดับนี้: model.py (รูปแบบไฟล์) → store.py (อ่าน+ตรวจ) → seats.py (นับเครื่อง)
→ enforce.py (จุดบังคับใช้ที่เดียว)
"""

from lmds.licensing.enforce import (
    CAPABILITIES,
    FLEET_GROW,
    FLEET_WRITE,
    Decision,
    LicenseRequired,
    check,
    require,
)
from lmds.licensing.model import FREE_SERVING_MACHINES, TIERS, License
from lmds.licensing.seats import Count, count_fleet, count_here
from lmds.licensing.store import Status, install, license_path, load

__all__ = [
    "CAPABILITIES", "FLEET_GROW", "FLEET_WRITE", "FREE_SERVING_MACHINES", "TIERS",
    "Count", "Decision", "License", "LicenseRequired", "Status",
    "check", "count_fleet", "count_here", "install", "license_path", "load", "require",
]
