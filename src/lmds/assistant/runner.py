"""รัน probe/action ที่เลือกจากแคตตาล็อก — บนเครื่องนี้หรือบนเครื่องปลายทางผ่าน SSH

ใช้ทางเดินเดิมทั้งหมด (`lmds.nodes.ssh`) ไม่เปิดช่องใหม่: key เดียวกัน ทะเบียนเดียวกัน
timeout เดียวกัน · ผู้ช่วยจึงไม่มีสิทธิ์อะไรมากกว่าที่ปุ่มบนหน้าเว็บมีอยู่แล้ว

สองอย่างที่ทำเสมอกับผลลัพธ์ ก่อนส่งต่อให้ LLM:

  1. **ตัดให้สั้น** — log 400 บรรทัดกิน context จนไม่เหลือที่ให้คำถาม เก็บ *ท้าย*
     ไว้เพราะ error อยู่ท้ายเสมอ
  2. **redact** — ผลจากเครื่องจริงมี API key, token, endpoint ภายใน ปนมาได้ และ
     ผลนี้กำลังจะถูกส่งออกไปหา LLM provider ข้างนอก
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from .catalog import ACTIONS, PROBES, Action, ParamError, Probe

# ต่อผลหนึ่งชิ้น — มากกว่านี้ context ของคำตอบจะถูกเบียดจนหมด
MAX_OUTPUT_CHARS = 4000
LOCAL = "this"


class RunError(Exception):
    pass


@dataclass
class Outcome:
    """ผลของ probe/action หนึ่งครั้ง — โครงเดียวกันทั้งสองแบบ เพื่อให้ prompt อ่านง่าย"""

    name: str
    title: str
    target: str
    params: dict[str, str] = field(default_factory=dict)
    exit_code: int = 0
    output: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.exit_code == 0

    def payload(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "target": self.target,
            "params": self.params,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "output": self.output,
            "error": self.error,
        }


def _trim(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    # เก็บท้ายไว้ — สาเหตุที่คำสั่งล้มอยู่บรรทัดท้าย ๆ เสมอ
    return "…(ตัดส่วนต้นออก)\n" + text[-MAX_OUTPUT_CHARS:]


def _clean_output(stdout: str, stderr: str) -> str:
    from lmds.secrets.redact import redact

    joined = (stdout or "") + (("\n" + stderr) if stderr and stderr.strip() else "")
    return _trim(redact(joined))


def _run_local(command: str, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"หมดเวลา {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


def _run_remote(target: str, command: str, timeout: int) -> tuple[int, str, str]:
    from lmds.nodes import NodeError, find, run

    node = find(target)
    if node is None:
        raise RunError(f"ไม่รู้จักเครื่อง '{target}' — ดูรายชื่อด้วย `lmds node list`")
    try:
        result = run(node, command, timeout=timeout)
    except NodeError as exc:
        raise RunError(str(exc)) from exc
    return result.exit_code, result.stdout, result.stderr


def _execute(name: str, title: str, target: str, command: str,
             params: dict[str, str], timeout: int) -> Outcome:
    outcome = Outcome(name=name, title=title, target=target or LOCAL, params=params)
    try:
        if not target or target == LOCAL:
            code, out, err = _run_local(command, timeout)
        else:
            code, out, err = _run_remote(target, command, timeout)
    except RunError as exc:
        outcome.error = str(exc)
        outcome.exit_code = 255
        return outcome
    outcome.exit_code = code
    outcome.output = _clean_output(out, err)
    if code != 0 and not outcome.output:
        outcome.output = f"คำสั่งจบด้วยรหัส {code} แต่ไม่มีข้อความอธิบาย"
    return outcome


def run_probe(name: str, target: str = LOCAL, params: dict | None = None) -> Outcome:
    """รัน probe หนึ่งตัว — อ่านอย่างเดียว จึงไม่ต้องขออนุมัติ"""
    probe: Probe | None = PROBES.get(name)
    if probe is None:
        return Outcome(name=name, title=name, target=target,
                       error=f"ไม่มี probe ชื่อ '{name}' ในแคตตาล็อก", exit_code=255)
    try:
        command, clean = probe.command(params or {})
    except ParamError as exc:
        return Outcome(name=name, title=probe.title, target=target,
                       error=str(exc), exit_code=255)
    return _execute(probe.name, probe.title, target, command, clean, probe.timeout)


def preview_action(name: str, target: str = LOCAL, params: dict | None = None) -> dict:
    """ประกอบคำสั่งของ action ให้ดู **โดยไม่รัน** — ผู้ใช้ต้องเห็นของจริงก่อนอนุมัติ

    คืนคำสั่งเต็มที่จะรันจริง ไม่ใช่คำอธิบาย · คนที่กดอนุมัติควรตรวจได้ว่ามันตรงกับที่บอก
    """
    action: Action | None = ACTIONS.get(name)
    if action is None:
        raise RunError(f"ไม่มีคำสั่ง '{name}' ในแคตตาล็อก")
    command, clean = action.command(params or {})
    return {
        "name": action.name,
        "title": action.title,
        "target": target or LOCAL,
        "params": clean,
        "risk": action.risk,
        "impact": action.impact,
        "steps": list(action.steps),
        "command": command,
    }


def run_action(name: str, target: str = LOCAL, params: dict | None = None) -> Outcome:
    """รัน action จริง — เรียกได้เฉพาะหลังผู้ใช้อนุมัติแล้วเท่านั้น (ดู policy.py)

    ฟังก์ชันนี้ไม่ตรวจสิทธิ์เอง โดยตั้งใจ: การอนุมัติเป็นเรื่องของชั้น policy ที่ถือ
    ตั๋วและอายุของมัน · ถ้าเอาสองอย่างมาปนกัน จะไม่มีที่ไหนเลยที่อ่านแล้วรู้ว่ากติกา
    การอนุมัติทั้งหมดคืออะไร
    """
    action: Action | None = ACTIONS.get(name)
    if action is None:
        return Outcome(name=name, title=name, target=target,
                       error=f"ไม่มีคำสั่ง '{name}' ในแคตตาล็อก", exit_code=255)
    try:
        command, clean = action.command(params or {})
    except ParamError as exc:
        return Outcome(name=name, title=action.title, target=target,
                       error=str(exc), exit_code=255)
    return _execute(action.name, action.title, target, command, clean, action.timeout)
