"""การอนุมัติงานที่เปลี่ยนสภาพเครื่อง — คนเลือกเสมอว่าจะให้ทำแค่ไหน

ผู้ใช้เลือกได้ 3 แบบต่อ *หนึ่งงาน* ไม่ใช่ตั้งครั้งเดียวแล้วใช้ตลอด เพราะงานคนละงาน
มีความเสี่ยงไม่เท่ากัน — "ล้างแคช FlashInfer" กับ "เปลี่ยน bind เป็น 0.0.0.0"
ไม่ควรใช้มาตรฐานเดียวกัน:

    APPLY  แก้เลย            รันทุกขั้นรวดเดียว
    STEP   ทีละขั้น          รันทีละขั้น หยุดให้ดูผลก่อนไปขั้นถัดไป
    HOLD   ยังไม่ทำ          แสดงคำสั่งไว้เฉย ๆ ให้เอาไปรันเองหรือค่อยตัดสินใจ

ทำไมต้องมีตั๋ว (ticket) แทนที่จะให้ LLM เรียก endpoint ตรง ๆ:

ตั๋วออกโดย **เซิร์ฟเวอร์** ตอนเสนองาน และใช้ได้ต่อเมื่อ **เบราว์เซอร์ของผู้ใช้**
ส่งกลับมาพร้อมโหมดที่เลือก · ต่อให้โมเดลถูก prompt injection จนพยายามสั่งรันอะไร
ก็ตาม มันไม่มีทางออกตั๋วให้ตัวเองได้ ทางเดียวที่คำสั่งจะทำงานคือมีคนกดปุ่ม

ตั๋วหมดอายุ และแต่ละขั้นใช้ได้ครั้งเดียว — กันการกดซ้ำจนรีสตาร์ตโมเดลสองรอบซ้อน
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from .runner import Outcome, expand_action, preview_action, run_action, run_probe

APPLY = "apply"
STEP = "step"
HOLD = "hold"
MODES = (APPLY, STEP, HOLD)

# นานพอให้คนอ่านคำสั่งจบแล้วตัดสินใจ แต่ไม่นานจนตั๋วที่ลืมไว้เมื่อเช้ายังใช้ได้ตอนเย็น
TICKET_TTL_SECONDS = 30 * 60
MAX_TICKETS = 40
MAX_STEPS = 8


class PolicyError(Exception):
    pass


class PolicyBusy(PolicyError):
    """ตั๋วนี้มีขั้นที่กำลังทำงานอยู่ — คำขอที่มาซ้อนต้องรอ ไม่ใช่เริ่มขั้นถัดไป"""


class PolicyHalted(PolicyError):
    """ขั้นก่อนหน้าล้ม — ตั๋วนี้เดินต่อไม่ได้ ต้องให้ผู้ช่วยดูผลแล้วเสนอใหม่"""


@dataclass
class PlanStep:
    action: str
    target: str
    params: dict[str, str]
    title: str
    command: str
    risk: str
    impact: str
    done: bool = False
    result: dict | None = None

    def payload(self) -> dict:
        return {
            "action": self.action, "target": self.target, "params": self.params,
            "title": self.title, "command": self.command, "risk": self.risk,
            "impact": self.impact, "done": self.done, "result": self.result,
        }


@dataclass
class Ticket:
    id: str
    why: str                      # ผู้ช่วยอธิบายว่าทำไมถึงเสนองานนี้
    steps: list[PlanStep]
    created: float = field(default_factory=time.time)
    mode: str = ""                # ว่าง = ผู้ใช้ยังไม่ได้เลือกจากเมนู
    # มีคำขอหนึ่งกำลังเดินตั๋วนี้อยู่ — ถือได้ทีละคำขอ (ดู advance)
    busy: bool = False

    @property
    def expired(self) -> bool:
        return time.time() - self.created > TICKET_TTL_SECONDS

    @property
    def finished(self) -> bool:
        return all(step.done for step in self.steps)

    @property
    def destructive(self) -> bool:
        """มีขั้นที่ลบของถาวร (risk high) — ค่าตั้งต้นของเมนูคือ "ยังไม่ทำ" และ "แก้เลย" ต้องยืนยันซ้ำ"""
        return any(step.risk == "high" for step in self.steps)

    @property
    def default_mode(self) -> str:
        """ปุ่มที่หน้าเว็บเน้นให้ — ลบถาวร → HOLD · หลายขั้น → STEP (ดูผลก่อนไปต่อ) · งานเดียว → APPLY"""
        if self.destructive:
            return HOLD
        return STEP if len(self.steps) > 1 else APPLY

    def next_index(self) -> int:
        for index, step in enumerate(self.steps):
            if not step.done:
                return index
        return -1

    def failed_index(self) -> int:
        """ขั้นแรกที่เริ่มไปแล้วแต่ไม่สำเร็จ (ล้ม หรือระเบิดจนไม่มีผล) — -1 = ไม่มี · ใช้ตอนไม่มีขั้นไหนกำลังรัน"""
        for index, step in enumerate(self.steps):
            result = step.result or {}
            # "bundle นี้ไม่มีคำสั่งนั้น" ไม่ใช่ขั้นที่ล้ม — ไม่ควรหยุดตั๋วทั้งใบ (ดู Outcome.unsupported)
            if step.done and not result.get("ok") and not result.get("unsupported"):
                return index
        return -1

    def payload(self) -> dict:
        return {
            "ticket": self.id,
            "why": self.why,
            "mode": self.mode,
            "expired": self.expired,
            "finished": self.finished,
            # busy = มีขั้นกำลังทำงาน (ปุ่ม "ขั้นถัดไป" ต้องรอ) · halted = ขั้นก่อนหน้าล้ม ตั๋วเดินต่อไม่ได้
            "busy": self.busy,
            "halted": (not self.busy) and self.failed_index() >= 0,
            "destructive": self.destructive,
            "default_mode": self.default_mode,
            "next_index": self.next_index(),
            "steps": [step.payload() for step in self.steps],
            # เมนูที่หน้าเว็บเอาไปวาดปุ่ม — ข้อความอยู่ที่เดียวกับกติกา ไม่ใช่ไปเขียนซ้ำใน JS
            "menu": menu(),
        }


def menu() -> list[dict]:
    return [
        {"mode": APPLY, "label": "แก้เลย",
         "detail": "ให้ผู้ช่วยรันทุกขั้นให้จบในครั้งเดียว"},
        {"mode": STEP, "label": "ทีละขั้น",
         "detail": "รันทีละขั้น หยุดให้ดูผลก่อนไปต่อทุกครั้ง"},
        {"mode": HOLD, "label": "ยังไม่ทำ",
         "detail": "แสดงคำสั่งไว้เฉย ๆ ไม่แตะเครื่อง"},
    ]


_LOCK = threading.Lock()
_TICKETS: dict[str, Ticket] = {}


def _prune() -> None:
    for key in [k for k, t in _TICKETS.items() if t.expired]:
        _TICKETS.pop(key, None)
    while len(_TICKETS) > MAX_TICKETS:
        oldest = min(_TICKETS.values(), key=lambda t: t.created)
        _TICKETS.pop(oldest.id, None)


def propose(steps: list[dict], why: str = "") -> Ticket:
    """สร้างตั๋วจากรายการงานที่ผู้ช่วยเสนอ — ยังไม่รันอะไรทั้งนั้น

    ประกอบคำสั่งจริงตั้งแต่ตอนนี้ เพื่อให้สิ่งที่ผู้ใช้เห็นตอนอนุมัติคือสิ่งเดียวกับ
    ที่จะรัน ไม่ใช่คำอธิบายที่อาจไม่ตรงกับของจริง
    """
    if not steps:
        raise PolicyError("ไม่มีขั้นตอนให้อนุมัติ")
    # งานประกอบ (deploy_model) ขยายเป็นขั้นจริงก่อนนับ — ผู้ใช้ต้องเห็นทุกคำสั่ง ไม่ใช่ชื่อรวม
    expanded: list[dict] = []
    for raw in steps:
        expanded.extend(expand_action(raw.get("action", ""), raw.get("target", ""), raw.get("params") or {}))
    steps = expanded
    if len(steps) > MAX_STEPS:
        raise PolicyError(f"เสนอมาเกิน {MAX_STEPS} ขั้น — ซอยงานให้เล็กลงก่อน")

    built: list[PlanStep] = []
    for raw in steps:
        preview = preview_action(
            raw.get("action", ""), raw.get("target", ""), raw.get("params") or {}
        )
        built.append(PlanStep(
            action=preview["name"], target=preview["target"], params=preview["params"],
            title=preview["title"], command=preview["command"],
            risk=preview["risk"], impact=preview["impact"],
        ))

    ticket = Ticket(id=secrets.token_urlsafe(12), why=why.strip()[:400], steps=built)
    with _LOCK:
        _prune()
        _TICKETS[ticket.id] = ticket
    return ticket


def get(ticket_id: str) -> Ticket:
    with _LOCK:
        ticket = _TICKETS.get(ticket_id)
    if ticket is None:
        raise PolicyError("ไม่พบตั๋วนี้ — อาจหมดอายุไปแล้ว ให้ถามผู้ช่วยใหม่อีกครั้ง")
    if ticket.expired:
        raise PolicyError("ตั๋วหมดอายุแล้ว — ถามผู้ช่วยใหม่เพื่อให้เสนองานอีกรอบ")
    return ticket


def choose_once(ticket_id: str, mode: str, confirm: bool = False) -> tuple[Ticket, bool]:
    """ผู้ใช้เลือกจากเมนู — คืน (ตั๋ว, คำขอนี้เป็นคนเลือกหรือเปล่า)

    การเลือกเกิดได้ **ครั้งเดียว** ต่อตั๋ว และตัดสินใต้ล็อก: สองคำขอที่มาเกือบพร้อมกัน (กดปุ่มซ้ำ) มีคำขอเดียวที่ได้
    True · คำขอที่ได้ False ต้องไม่ไปเริ่มงาน — เดิมทั้งคู่เรียก advance ต่อ ขั้นที่ 1 กับขั้นที่ 2 จึงรันซ้อนกัน

    งานลบถาวร (destructive) รับ "แก้เลย" ต่อเมื่อยืนยันซ้ำ (`confirm`) — หน้าเว็บถามอีกชั้นก่อนส่ง ·
    "ทีละขั้น" ไม่ต้อง เพราะแต่ละขั้นต้องกดอยู่แล้ว
    """
    if mode not in MODES:
        raise PolicyError(f"โหมด '{mode}' ไม่มีอยู่จริง")
    ticket = get(ticket_id)
    if mode == APPLY and ticket.destructive and not confirm:
        raise PolicyError("งานนี้ลบของถาวร — ต้องยืนยันซ้ำก่อน 'แก้เลย' หรือเลือก 'ทีละขั้น'")
    with _LOCK:
        if ticket.mode and ticket.mode != mode:
            raise PolicyError(f"ตั๋วนี้เลือกไปแล้วว่า '{ticket.mode}'")
        first = not ticket.mode
        ticket.mode = mode
    return ticket, first


def choose(ticket_id: str, mode: str, confirm: bool = False) -> Ticket:
    """เลือกโหมด — คืนตั๋ว (ผู้เรียกที่ต้องรู้ว่าตัวเองเป็นคนเลือกหรือเปล่าใช้ choose_once)"""
    return choose_once(ticket_id, mode, confirm)[0]


def explain_failure(step: PlanStep) -> str:
    """ขั้นที่ล้มมี slug → ไปดึงสาเหตุจริงจาก log ของโมเดลนั้น (probe last_failure) มาแปะไว้กับผล

    ผู้ใช้ไม่ควรต้องถามผู้ช่วยอีกรอบว่า "แล้วทำไมล้ม" — คำตอบอยู่ใน server.log/docker logs อยู่แล้ว
    · ล้มเองก็แค่ไม่มีคำอธิบาย ไม่ทำให้ตั๋วพัง
    """
    from .catalog import ACTIONS

    slug = (step.params or {}).get("slug")
    spec = ACTIONS.get(step.action)
    # งานของ hub (push/deploy_plan/node_install) ไม่มี log ของโมเดลบน hub ให้อ่าน · เทสที่ล้มคือผลของมันเอง
    if not slug or spec is None or spec.hub_only or step.action == "run_test":
        return ""
    try:
        outcome = run_probe("last_failure", step.target, {"slug": slug})
    except Exception:
        return ""
    return (outcome.output or "")[-1500:] if outcome.ok else ""


def advance(ticket_id: str) -> tuple[Ticket, list[Outcome]]:
    """ลงมือทำตามโหมดที่เลือกไว้

    APPLY รันจนจบ · STEP รันขั้นเดียวแล้วหยุด · HOLD ไม่รันอะไรเลย
    """
    ticket = get(ticket_id)
    if not ticket.mode:
        raise PolicyError("ยังไม่ได้เลือกจากเมนูว่าจะให้ทำแบบไหน")
    if ticket.mode == HOLD:
        return ticket, []

    # การเดินตั๋วเป็นของคำขอเดียวในแต่ละขณะ — จองใต้ล็อกก่อนแตะขั้นไหนทั้งนั้น
    #
    # audit 2026-10: เดิมแต่ละขั้นถูก "ทำเครื่องหมายว่าเสร็จ" ก่อนรัน เพื่อกันรันซ้ำ แต่นั่นทำให้คำขอที่มาซ้อน
    # มองว่าขั้นที่ 1 เสร็จแล้ว จึงหยิบขั้นที่ 2 ไปรันทันที · ตั๋ว "stop แล้ว remove" ที่กดปุ่มซ้ำ: remove เริ่มขณะที่
    # stop ยังไม่จบ และก่อนจะรู้ด้วยซ้ำว่า stop ล้ม · ตอนนี้ (1) มีขั้นกำลังรัน = PolicyBusy (2) ขั้นก่อนหน้าไม่สำเร็จ
    # = PolicyHalted ไม่ว่าจะโหมดไหน — ขั้นถัดไปเริ่มได้ต่อเมื่อขั้นก่อนหน้า **จบและสำเร็จ** เท่านั้น
    with _LOCK:
        if ticket.busy:
            running = next((s for s in ticket.steps if s.done and s.result is None), None)
            raise PolicyBusy("กำลังทำ" + (f" '{running.title}'" if running else "ขั้นก่อนหน้า")
                             + " อยู่ — รอให้จบก่อน แล้วดูผลก่อนไปขั้นถัดไป")
        failed = ticket.failed_index()
        if failed >= 0:
            raise PolicyHalted(
                f"ขั้นที่ {failed + 1} ('{ticket.steps[failed].title}') ไม่สำเร็จ — ขั้นถัดไปตั้งอยู่บนผลของมัน จึงไม่ทำต่อ · "
                "ดูผลของขั้นนั้นแล้วถามผู้ช่วยใหม่เพื่อวางแผนต่อ")
        if ticket.finished:
            return ticket, []
        ticket.busy = True

    outcomes: list[Outcome] = []
    try:
        while True:
            with _LOCK:
                index = ticket.next_index()
                if index < 0:
                    break
                step = ticket.steps[index]
                # ทำเครื่องหมายว่าขั้นนี้ถูกหยิบไปแล้ว (ใช้ได้ครั้งเดียว) — ผลยังเป็น None จนกว่าจะรันจบ
                step.done = True
            outcome = run_action(step.action, step.target, step.params)
            result = outcome.payload()
            if not outcome.ok and not outcome.unsupported:
                result["explain"] = explain_failure(step)
            step.result = result
            outcomes.append(outcome)
            if ticket.mode == STEP:
                break
            # "bundle นี้ไม่มีคำสั่งนั้น" (เทสที่ engine นี้ไม่มี) ไม่ใช่ขั้นที่ล้ม — ขั้นถัดไปยังเดินต่อได้
            if not outcome.ok and not outcome.unsupported:
                # ขั้นถัดไปมักตั้งอยู่บนสมมติฐานว่าขั้นก่อนหน้าสำเร็จ — หยุดแล้วให้คนดู
                break
    finally:
        # ปล่อยเสมอ แม้ขั้นจะระเบิด — ขั้นที่ระเบิดไม่มีผล (result None) จึงนับเป็นไม่สำเร็จและตั๋วหยุดที่นั่น
        with _LOCK:
            ticket.busy = False
    return ticket, outcomes


def forget(ticket_id: str) -> None:
    with _LOCK:
        _TICKETS.pop(ticket_id, None)


def reset() -> None:
    """ล้างตั๋วทั้งหมด — สำหรับเทสต์"""
    with _LOCK:
        _TICKETS.clear()
