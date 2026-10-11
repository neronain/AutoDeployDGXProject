"""ประกอบ health findings ของทั้งฟลีตจากแคชของ refresher — ไม่ยิง SSH เพิ่มเอง

ใช้แคชเดียวกับหน้าเว็บ (`state.STORE.snapshot()`) เหตุผลเดียวกับทุกอย่างใน web/state.py:
เครื่องที่ตอบช้า/ต่อไม่ได้ต้องไม่ทำให้ endpoint นี้ช้าตาม และข้อมูลเก่าต้องบอกว่าเก่าแค่ไหน
ไม่ใช่เงียบ ๆ โชว์เป็นของสด
"""

from __future__ import annotations

from lmds.hardware.health import evaluate_host
from lmds.web import state


def _entry_findings(entry: dict) -> dict:
    data = entry.get("data")
    host = (data or {}).get("host")
    findings = evaluate_host(host) if host else []
    return {
        "findings": [f.payload() for f in findings],
        "error": entry.get("error") or "",
        "stale": entry.get("stale", True),
        "age_seconds": entry.get("age_seconds", 0.0),
    }


def local_findings() -> dict:
    snapshot = state.STORE.snapshot()
    return _entry_findings(snapshot["host"])


def fleet_findings() -> dict:
    """findings ของเครื่องนี้ + ทุก node ที่ refresher เคยสำรวจได้ — เครื่องที่ไม่มีแคชเลยถูกข้าม

    (ไม่ใช่รายงานว่า "ปกติ" — แค่ยังไม่มีข้อมูลให้ประเมิน ต่างจากประเมินแล้วไม่พบปัญหา)
    """
    snapshot = state.STORE.snapshot()
    return {
        "host": _entry_findings(snapshot["host"]),
        "nodes": {name: _entry_findings(entry) for name, entry in snapshot["nodes"].items()},
    }
