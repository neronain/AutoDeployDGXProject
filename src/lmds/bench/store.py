"""เก็บผลวัดไว้เทียบข้ามเวลา ข้ามเครื่อง ข้ามเวอร์ชัน engine

ตัวเลขความเร็วโดด ๆ ไม่มีความหมาย — "32 tok/s" ดีหรือไม่ดีขึ้นกับว่าเครื่องอะไร quant อะไร
context เท่าไร build ไหน · ทุกผลจึงเก็บสภาพแวดล้อมไปด้วยทั้งชุด ไม่งั้นสามเดือนผ่านไป
จะไม่มีทางรู้ว่าที่เร็วขึ้นเพราะอัปเกรด llama.cpp หรือเพราะบังเอิญวัดตอนเครื่องว่าง

รูปแบบเป็น JSON หนึ่งไฟล์ต่อหนึ่งรอบวัด ไม่ใช่ฐานข้อมูล — อ่านด้วยตาได้ ก๊อปข้ามเครื่องได้
และไม่ต้องมี migration ตอนเพิ่มฟิลด์
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path


def bench_root() -> Path:
    return Path.home() / ".lmds" / "bench"


def _machine_facts() -> dict:
    from lmds.hardware import probe
    from lmds.hardware.profiler import detect_cpu, host_summary

    report = probe()
    summary = host_summary()
    return {
        "hostname": summary.hostname,
        "arch": report.arch,
        "profile": report.profile.value,
        "ram_total_gb": summary.ram_total_gb,
        "cpu": (detect_cpu() or {}).get("model", ""),
        "gpus": [
            {"name": gpu.name,
             "vram_gb": round(gpu.vram_mib / 1024, 1) if gpu.vram_mib else None}
            for gpu in report.gpus
        ],
        "os": platform.platform(),
    }


def record(slug: str, model_id: str, engine: str, served_name: str,
           workloads: list[dict], probes: list[dict], environment: dict,
           stamped_at: str) -> Path:
    """เขียนผลหนึ่งรอบลงไฟล์ แล้วคืน path

    `stamped_at` รับมาจากผู้เรียกแทนที่จะเรียก datetime เอง — ให้เทสต์กำหนดเวลาได้
    และให้ผลที่วัดพร้อมกันหลายเครื่องใช้ตราเวลาเดียวกัน
    """
    directory = bench_root() / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamped_at.replace(':', '').replace('-', '')}.json"
    payload = {
        "version": 1,
        "slug": slug,
        "model_id": model_id,
        "engine": engine,
        "served_name": served_name,
        "stamped_at": stamped_at,
        "machine": _machine_facts(),
        # build ของ engine, quant, context, MoE/MTP — ตัวแปรที่เปลี่ยนผลมากที่สุด
        "environment": environment,
        "workloads": workloads,
        "probes": probes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runs_for(slug: str) -> list[Path]:
    """ผลทุกรอบของ slug นี้ ใหม่สุดก่อน"""
    directory = bench_root() / slug
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), reverse=True)


def latest_merged(slug: str) -> dict | None:
    """ภาพรวมล่าสุดของโมเดลหนึ่ง — เติมด้านที่รอบล่าสุดไม่ได้วัดจากรอบก่อนหน้า

    `lmds bench run --caps-only` ไม่มีข้อมูลความเร็วในตัวมันเอง ถ้าเอารอบล่าสุดมาตรง ๆ
    ตารางคะแนนจะกลายเป็นขีดกลางทั้งคอลัมน์ ทั้งที่เพิ่งวัดความเร็วไปเมื่อสิบนาทีก่อน —
    อ่านแล้วเหมือนโมเดลถอยหลัง ทั้งที่เราแค่ถามคำถามที่แคบลง

    รอบที่ถูกยืมมาแนบตราเวลาของมันเองไว้ ผู้ใช้จะได้รู้ว่าตัวเลขนั้นเก่ากว่าที่เห็นข้างบน
    """
    paths = runs_for(slug)
    if not paths:
        return None
    try:
        merged = load(paths[0])
    except (OSError, json.JSONDecodeError):
        return None
    for field, stamp_key in (("workloads", "speed_from"), ("probes", "probes_from")):
        if merged.get(field):
            continue
        for path in paths[1:]:
            try:
                older = load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if older.get(field):
                merged[field] = older[field]
                merged[stamp_key] = older.get("stamped_at", "")
                break
    return merged


def remove(slug: str, keep_last: int = 0) -> int:
    """ลบผลวัดของโมเดลหนึ่ง คืนจำนวนไฟล์ที่ลบ

    `keep_last` > 0 = เก็บรอบล่าสุดไว้เท่านั้น · ผลสะสมเร็วกว่าที่คิดเพราะการวัดซ้ำเป็น
    เรื่องปกติ (ก่อน/หลังเปลี่ยน flag, ก่อน/หลังอัปเกรด engine) แล้วไม่มีใครกลับมาลบเอง
    """
    directory = bench_root() / slug
    if not directory.is_dir():
        return 0
    runs = sorted(directory.glob("*.json"), reverse=True)
    doomed = runs[keep_last:] if keep_last > 0 else runs
    removed = 0
    for path in doomed:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    # โฟลเดอร์ว่างที่ค้างไว้ทำให้ตารางคะแนนยังนับโมเดลนั้นอยู่ทั้งที่ไม่มีข้อมูลแล้ว
    if not any(directory.iterdir()):
        directory.rmdir()
    return removed


def all_runs() -> list[dict]:
    """ผลล่าสุดของทุกโมเดลที่เคยวัด — ใช้ทำตารางคะแนนรวม

    ตารางคะแนนตอบคำถามว่า "ตอนนี้ตัวไหนดีกว่า" ไม่ใช่ประวัติ จึงเอารอบล่าสุดของแต่ละตัว
    (แต่เติมด้านที่รอบล่าสุดไม่ได้วัด — ดู latest_merged)
    """
    root = bench_root()
    if not root.is_dir():
        return []
    latest = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        merged = latest_merged(directory.name)
        if merged:
            latest.append(merged)
    return latest
