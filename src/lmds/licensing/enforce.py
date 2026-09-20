"""จุดบังคับใช้ไลเซนส์ — **ที่เดียวในทั้งโปรแกรม**

`grep -rn "licensing.require(" src/` ครั้งเดียวต้องเห็นรายการล็อกทั้งหมด ไม่มีที่อื่นอีก
ลูกค้าองค์กรที่ต้อง audit โค้ดก่อนซื้อจะอ่านไฟล์นี้ไฟล์เดียวจบ และนั่นคือสิ่งที่เราอยากให้เป็น

กฎเหล็กสามข้อ (ละเมิดข้อไหนคือบั๊ก ไม่ใช่ทางเลือกในการออกแบบ)
------------------------------------------------------------
**1. ห้ามล็อกอะไรที่กันผู้ใช้เจ็บตัว**
   quality gates · doctor · repair · validate · preflight · secret redaction · allowlist ·
   approval ticket → **ฟรีตลอดกาล** คนใช้ฟรีที่ deploy พังแล้วโดนแฮ็กจะด่าชื่อเรา
   ไม่ใช่ด่าตัวเองที่ไม่ได้จ่ายเงิน

**2. ห้ามหยุดของที่รันอยู่ เด็ดขาด**
   หมดอายุ = เข้าโหมดอ่านอย่างเดียว (ขยายฟลีตไม่ได้) แต่ `ps` `logs` `doctor` `stop`
   `start` `restart` ยังทำได้ทั้งหมด · ถ้าลูกค้ารีบูตตี 3 แล้ว autostart ไม่ทำงานเพราะ
   ไลเซนส์หมดเมื่อวาน = เสียลูกค้ารายนั้นถาวร และสมควรแล้ว

**3. ล็อก "ขนาด" ไม่ล็อก "ความสามารถ"**
   คนใช้ฟรีทำได้ครบทั้งลูปบนเครื่องเดียว — deploy, stacked ก็ทำได้ถ้ามีเครื่องพอ,
   ทุกคำสั่ง, ทุก engine · สิ่งเดียวที่ต่างคือ *จำนวนเครื่องที่บริหารร่วมกัน*

สิ่งที่ **ไม่เคย** ถูกล็อก (เทส test_licensing.py คุมรายการนี้ไว้)
----------------------------------------------------------------
    ps · logs · doctor · repair · validate · smoke · fit · inspect · hardware · scan ·
    list · version · recipes · bench · start · stop · restart · enable · disable ·
    remove · set · adopt · generate · deploy (บนเครื่องตัวเอง) · web · config

สิ่งที่ถูกล็อก — ทั้งหมดอยู่ใน CAPABILITIES ข้างล่าง
"""

from __future__ import annotations

from dataclasses import dataclass

from lmds.licensing import store
from lmds.licensing.model import FREE_SERVING_MACHINES

# ── รายการล็อกทั้งหมดของผลิตภัณฑ์นี้ ────────────────────────────────────────────
#
# เพิ่มบรรทัดในนี้ = เพิ่มการล็อกหนึ่งอย่าง · ต้องมีเหตุผลที่เขียนให้ลูกค้าอ่านได้
FLEET_GROW = "fleet.grow"
FLEET_WRITE = "fleet.write"

CAPABILITIES: dict[str, str] = {
    FLEET_GROW: (
        "เพิ่มเครื่องที่เสิร์ฟได้เข้าฟลีตที่บริหารร่วมกัน เกินจำนวนที่ไลเซนส์ครอบคลุม "
        "(LICENSE §1.3 · §3)"
    ),
    FLEET_WRITE: (
        "ส่งของไปเครื่องอื่นในฟลีต (push · install · clone) ขณะที่ไลเซนส์หมดอายุแล้ว "
        "— ของที่รันอยู่ไม่ถูกแตะ"
    ),
}


class LicenseRequired(Exception):
    """ต้องมีไลเซนส์ถึงจะทำสิ่งนี้ได้ — ข้อความในตัวมันพร้อมโชว์ให้ผู้ใช้อ่านเลย"""

    def __init__(self, message: str, *, capability: str = "", status: store.Status | None = None):
        super().__init__(message)
        self.capability = capability
        self.status = status


@dataclass(frozen=True)
class Decision:
    allowed: bool
    message: str = ""

    def __bool__(self) -> bool:
        return self.allowed


# ช่องทางขอไลเซนส์ — รวมไว้ที่เดียว เพราะข้อความนี้โผล่ทั้งใน CLI, หน้าเว็บ และข้อความ
# ตอนถูกบล็อก · แยกกันเขียนเมื่อไรจะมีที่ใดที่หนึ่งตกรุ่นแล้วลูกค้าติดต่อไม่ได้
CONTACT_EMAIL = "tananan99@icloud.com"
CONTACT_URL = "fb.com/neronain.minidev"


def _buy_line() -> str:
    return (f"ดูเงื่อนไขใน LICENSE · ติดต่อขอไลเซนส์: {CONTACT_EMAIL} · {CONTACT_URL}\n"
            f"ติดตั้งไฟล์ที่ได้มาด้วย:  lmds license install <ไฟล์>")


def check(capability: str, *, serving_now: int | None = None,
          adding: int = 1, status: store.Status | None = None) -> Decision:
    """ตัดสินว่าทำได้ไหม — ไม่โยน exception ใช้ตอนอยากโชว์สถานะโดยไม่ขัดจังหวะ

    `serving_now` = จำนวนเครื่องที่เสิร์ฟได้ *ตอนนี้* · `adding` = กำลังจะเพิ่มอีกกี่เครื่อง
    """
    status = status or store.load()

    if capability == FLEET_WRITE:
        if status.read_only:
            lic = status.license
            return Decision(False,
                f"ไลเซนส์หมดอายุแล้ว ({status.reason}) — ส่งของไปเครื่องอื่นไม่ได้จนกว่าจะต่ออายุ\n"
                f"ของที่รันอยู่ทุกเครื่อง **ไม่ถูกแตะ** · ps · logs · doctor · start · stop "
                f"ยังใช้ได้ตามปกติ\n"
                + (f"ใบเดิม: {lic.id} · {lic.licensed_to}\n" if lic else "")
                + _buy_line())
        return Decision(True)

    if capability == FLEET_GROW:
        if status.unlimited:
            return Decision(True)
        allowed = status.machines_allowed
        if serving_now is None:
            return Decision(True)            # ไม่รู้ก็ไม่บล็อก — ดู docstring ของ seats.py
        after = serving_now + max(0, adding)
        if after <= allowed:
            return Decision(True)
        return Decision(False, _too_many_message(status, serving_now, after, allowed))

    # ความสามารถที่ไม่รู้จัก = ไม่ล็อก · เพิ่มชื่อใหม่โดยลืมใส่ใน CAPABILITIES ต้องไม่กลายเป็น
    # การบล็อกผู้ใช้แบบเงียบ ๆ
    return Decision(True)


def _too_many_message(status: store.Status, now: int, after: int, allowed: int) -> str:
    if status.state == "free":
        head = (f"โหมดฟรีครอบคลุม {FREE_SERVING_MACHINES} เครื่องที่เสิร์ฟได้ "
                f"(LICENSE §2) — ตอนนี้มี {now} เครื่อง คำสั่งนี้จะทำให้เป็น {after}")
    elif status.state == "expired":
        head = (f"ไลเซนส์หมดอายุแล้ว ({status.reason}) จึงเหลือสิทธิ์เท่าโหมดฟรี "
                f"{allowed} เครื่อง — ตอนนี้มี {now} คำสั่งนี้จะทำให้เป็น {after}")
    elif status.state == "invalid":
        head = (f"ไฟล์ไลเซนส์ใช้ไม่ได้ ({status.reason}) จึงเหลือสิทธิ์เท่าโหมดฟรี "
                f"{allowed} เครื่อง — ตอนนี้มี {now} คำสั่งนี้จะทำให้เป็น {after}")
    else:
        lic = status.license
        head = (f"ไลเซนส์ของ {lic.licensed_to if lic else '-'} ครอบคลุม {allowed} เครื่อง "
                f"— ตอนนี้มี {now} คำสั่งนี้จะทำให้เป็น {after}")
    return (f"{head}\n\n"
            f"control plane (เครื่องที่ไม่มี GPU/llama-server ใช้วางแผนและ push) **ไม่นับ** "
            f"— เพิ่มได้ไม่จำกัด\n"
            f"นับเฉพาะเครื่องที่เสิร์ฟได้และบริหารร่วมกัน (LICENSE §1.1 · §1.3)\n"
            f"ดูตัวเลขที่เรานับได้:  lmds license seats\n\n"
            + _buy_line())


def require(capability: str, *, serving_now: int | None = None,
            adding: int = 1, status: store.Status | None = None) -> None:
    """เหมือน check() แต่โยน LicenseRequired เมื่อทำไม่ได้ — ใช้ที่จุดบังคับใช้จริง"""
    decision = check(capability, serving_now=serving_now, adding=adding, status=status)
    if not decision.allowed:
        raise LicenseRequired(decision.message, capability=capability, status=status)
