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

from .runner import Outcome, preview_action, run_action

APPLY = "apply"
STEP = "step"
HOLD = "hold"
MODES = (APPLY, STEP, HOLD)

# นานพอให้คนอ่านคำสั่งจบแล้วตัดสินใจ แต่ไม่นานจนตั๋วที่ลืมไว้เมื่อเช้ายังใช้ได้ตอนเย็น
TICKET_TTL_SECONDS = 30 * 60
MAX_TICKETS = 40
MAX_STEPS = 6


class PolicyError(Exception):
    pass


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

    @property
    def expired(self) -> bool:
        return time.time() - self.created > TICKET_TTL_SECONDS

    @property
    def finished(self) -> bool:
        return all(step.done for step in self.steps)

    def next_index(self) -> int:
        for index, step in enumerate(self.steps):
            if not step.done:
                return index
        return -1

    def payload(self) -> dict:
        return {
            "ticket": self.id,
            "why": self.why,
            "mode": self.mode,
            "expired": self.expired,
            "finished": self.finished,
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


def choose(ticket_id: str, mode: str) -> Ticket:
    """ผู้ใช้เลือกจากเมนู — เลือกได้ครั้งเดียวต่อตั๋ว"""
    if mode not in MODES:
        raise PolicyError(f"โหมด '{mode}' ไม่มีอยู่จริง")
    ticket = get(ticket_id)
    with _LOCK:
        if ticket.mode and ticket.mode != mode:
            raise PolicyError(f"ตั๋วนี้เลือกไปแล้วว่า '{ticket.mode}'")
        ticket.mode = mode
    return ticket


def advance(ticket_id: str) -> tuple[Ticket, list[Outcome]]:
    """ลงมือทำตามโหมดที่เลือกไว้

    APPLY รันจนจบ · STEP รันขั้นเดียวแล้วหยุด · HOLD ไม่รันอะไรเลย
    """
    ticket = get(ticket_id)
    if not ticket.mode:
        raise PolicyError("ยังไม่ได้เลือกจากเมนูว่าจะให้ทำแบบไหน")
    if ticket.mode == HOLD:
        return ticket, []
    if ticket.finished:
        return ticket, []

    outcomes: list[Outcome] = []
    while True:
        with _LOCK:
            index = ticket.next_index()
            if index < 0:
                break
            step = ticket.steps[index]
            # ทำเครื่องหมายก่อนรัน — คนกดปุ่มซ้ำระหว่างที่ขั้นนี้ยังทำงานอยู่จะได้ไม่รันซ้ำ
            step.done = True
        outcome = run_action(step.action, step.target, step.params)
        step.result = outcome.payload()
        outcomes.append(outcome)
        if ticket.mode == STEP:
            break
        if not outcome.ok:
            # ขั้นถัดไปมักตั้งอยู่บนสมมติฐานว่าขั้นก่อนหน้าสำเร็จ — หยุดแล้วให้คนดู
            break
    return ticket, outcomes


def forget(ticket_id: str) -> None:
    with _LOCK:
        _TICKETS.pop(ticket_id, None)


def reset() -> None:
    """ล้างตั๋วทั้งหมด — สำหรับเทสต์"""
    with _LOCK:
        _TICKETS.clear()
