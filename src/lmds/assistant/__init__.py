"""ผู้ช่วยที่ลงไปดูเครื่องจริงได้ — แคตตาล็อก, ตัวรัน, การอนุมัติ, ความรู้

แยกออกมาจาก `lmds.web.assistant` เพราะสองอย่างนี้คนละเรื่องกัน: ตรงนั้นคือกล่องแชท
ของหน้าเว็บ (ประกอบ prompt, สถานะ, สตรีม) ส่วนตรงนี้คือ *สิ่งที่ผู้ช่วยทำได้* ซึ่ง
ไม่ผูกกับหน้าเว็บเลย และควรทดสอบได้โดยไม่ต้องยก FastAPI ขึ้นมาทั้งตัว
"""

from __future__ import annotations

from .catalog import ACTIONS, PROBES, Action, ParamError, Probe, action_menu, probe_menu
from .knowledge import doc_index, playbook, search_docs
from .policy import APPLY, HOLD, MODES, STEP, PolicyError, Ticket
from .runner import LOCAL, Outcome, RunError, preview_action, run_action, run_probe

__all__ = [
    "ACTIONS",
    "APPLY",
    "Action",
    "HOLD",
    "LOCAL",
    "MODES",
    "Outcome",
    "PROBES",
    "ParamError",
    "PolicyError",
    "Probe",
    "RunError",
    "STEP",
    "Ticket",
    "action_menu",
    "doc_index",
    "playbook",
    "preview_action",
    "probe_menu",
    "run_action",
    "run_probe",
    "search_docs",
]
