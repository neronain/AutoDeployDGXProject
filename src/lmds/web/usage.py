"""รวม token/request usage ของทั้งฟลีตจากแคชของ refresher

ไม่ได้เก็บอะไรใหม่ — `inventory.model_payload()` แปะ `usage` (จาก `request_usage()`)
ไว้ในทุก model dict อยู่แล้ว ทั้งของเครื่องนี้และของทุก node (ผ่าน `lmds agent info`)
ไฟล์นี้แค่รวมยอดจากของที่แคชไว้แล้ว เหตุผลเดียวกับ web/health.py

โมเดลที่ไม่ได้รัน หรือเครื่องที่ต่อไม่ได้ — `usage` เป็น None แปลว่าไม่มีข้อมูลให้รวม
ไม่ใช่รวมเป็นศูนย์ (ดู inventory.request_usage: None = ไม่ได้รัน)
"""

from __future__ import annotations

from lmds.web import state


def _node_usage(entry: dict) -> dict:
    data = entry.get("data")
    models = []
    for model in (data or {}).get("models") or []:
        usage = model.get("usage")
        if usage is None:
            continue
        models.append({"slug": model.get("slug"), **usage})
    return {
        "models": models,
        "error": entry.get("error") or "",
        "stale": entry.get("stale", True),
        "age_seconds": entry.get("age_seconds", 0.0),
    }


def _sum_field(rows: list[dict], field: str) -> int | None:
    values = [r[field] for r in rows if r.get(field) is not None]
    return sum(values) if values else None


def local_usage() -> dict:
    snapshot = state.STORE.snapshot()
    return _node_usage(snapshot["host"])


def fleet_usage() -> dict:
    """usage ของเครื่องนี้ + ทุก node ที่ refresher สำรวจได้ พร้อมยอดรวมทั้งฟลีต"""
    snapshot = state.STORE.snapshot()
    host = _node_usage(snapshot["host"])
    nodes = {name: _node_usage(entry) for name, entry in snapshot["nodes"].items()}
    all_rows = [*host["models"], *(row for entry in nodes.values() for row in entry["models"])]
    return {
        "host": host,
        "nodes": nodes,
        "fleet_total": {
            "requests_24h": _sum_field(all_rows, "requests_24h"),
            "prompt_tokens_24h": _sum_field(all_rows, "prompt_tokens_24h"),
            "generated_tokens_24h": _sum_field(all_rows, "generated_tokens_24h"),
            "models_counted": len(all_rows),
        },
    }
