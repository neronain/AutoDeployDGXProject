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
import re
from datetime import datetime
from pathlib import Path

# slug มาจาก argv (`lmds bench remove <slug>`) และจาก URL ของหน้าเว็บ (`DELETE /api/bench/{slug}`)
# แล้วถูกต่อเป็น path ใต้ ~/.lmds/bench ตรง ๆ — ตรวจที่นี่จุดเดียว ทุกฟังก์ชันที่รับ slug ผ่านด่านนี้
#
# เคสจริง (audit 2026-10-06): `lmds bench remove ../../myproject` ลบ package.json กับ
# tsconfig.json ในโฟลเดอร์นั้นทิ้ง แล้วรายงานว่า "ลบผลวัด … ไป 2 รอบ" — เพราะ remove()
# ต่อ path ดิบแล้วกวาด *.json ทุกไฟล์ที่เจอ
#
# อักขระชุดเดียวกับ slug ที่อื่นของ LMDS (fleet/apikey._SLUG · web/api._check_slug ·
# generator.check_slug_name) · ความยาวปล่อยถึงเพดานชื่อไฟล์ ไม่ใช่ 64 เพราะ bundle ที่สร้างก่อนมี
# กติกาความยาวยังมีผลวัดเก็บอยู่จริง (check_slug_name(allow_long=True) ยอมด้วยเหตุผลเดียวกัน)
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
# ชื่อไฟล์ที่ record() เขียนเองเท่านั้น (ตราเวลา YYYYMMDDTHHMMSS) — remove() ลบเฉพาะรูปนี้
# ไฟล์ .json อื่นที่บังเอิญอยู่ในโฟลเดอร์ไม่ใช่ของเรา ไม่อ่าน ไม่นับ ไม่ลบ
_RUN_FILE = re.compile(r"\d{8}T\d{6}\.json")


class BenchStoreError(ValueError):
    """slug หรือชื่อไฟล์ที่จะพาออกนอกที่เก็บผลวัด — ปฏิเสธก่อนแตะดิสก์"""


def bench_root() -> Path:
    return Path.home() / ".lmds" / "bench"


def check_slug(slug: str) -> str:
    """slug ต้องเป็นชื่อโฟลเดอร์ชั้นเดียว — ไม่ผ่าน = BenchStoreError (ไม่ใช่คืนค่าว่างเงียบ ๆ)"""
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        raise BenchStoreError(
            f"ชื่อโมเดล (slug) ไม่ถูกต้อง: {str(slug)[:80]!r} — ใช้ได้เฉพาะ a-z A-Z 0-9 . _ - "
            f"และขึ้นต้นด้วยตัวอักษร/ตัวเลข (ดูชื่อที่มีผลวัด: lmds bench list)")
    return slug


def slug_dir(slug: str) -> Path:
    """โฟลเดอร์ผลวัดของ slug นี้ — รับประกันว่าอยู่ใต้ bench_root() ชั้นเดียวพอดี

    ตรวจสองชั้นโดยตั้งใจ: รูปแบบชื่อ (กัน `..` กับ `/`) และ path ที่ resolve แล้ว (กัน symlink
    ที่ชื่อถูกแต่ชี้ออกไปข้างนอก — โฟลเดอร์นี้ถูกก๊อปข้ามเครื่องได้ จึงไม่ใช่ของที่เราสร้างเองเสมอ)
    """
    check_slug(slug)
    root = bench_root()
    directory = root / slug
    try:
        inside = directory.resolve().parent == root.resolve()
    except OSError as exc:
        raise BenchStoreError(f"อ่านที่เก็บผลวัดของ {slug} ไม่ได้: {exc}") from exc
    if not inside:
        raise BenchStoreError(
            f"ที่เก็บผลวัดของ {slug} ชี้ออกนอก {root} (symlink?) — ไม่แตะ")
    return directory


def _run_files(directory: Path) -> list[Path]:
    """ไฟล์ผลวัดที่ record() เขียนเอง ใหม่สุดก่อน — ไฟล์อื่นในโฟลเดอร์ไม่นับ"""
    return sorted((p for p in directory.iterdir()
                   if _RUN_FILE.fullmatch(p.name) and p.is_file()), reverse=True)


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
    directory = slug_dir(slug)
    name = f"{stamped_at.replace(':', '').replace('-', '')}.json"
    if not _RUN_FILE.fullmatch(name):
        # ตราเวลาที่ไม่ใช่รูป YYYY-MM-DDTHH:MM:SS จะได้ไฟล์ที่ runs_for()/remove() มองไม่เห็น
        raise BenchStoreError(f"ตราเวลาของผลวัดไม่ถูกรูปแบบ: {stamped_at!r} (ต้องเป็น YYYY-MM-DDTHH:MM:SS)")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
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
    """ผลทุกรอบของ slug นี้ ใหม่สุดก่อน — slug ที่ไม่ใช่ชื่อโฟลเดอร์ชั้นเดียว = BenchStoreError"""
    directory = slug_dir(slug)
    if not directory.is_dir():
        return []
    return _run_files(directory)


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
        if _has_result(merged.get(field)):
            continue
        for path in paths[1:]:
            try:
                older = load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if _has_result(older.get(field)):
                merged[field] = older[field]
                merged[stamp_key] = older.get("stamped_at", "")
                break
    return merged


def _has_result(rows) -> bool:
    """ด้านนี้ของรอบนั้น "วัดได้จริง" อย่างน้อยหนึ่งแถวไหม

    แถวที่ไม่ได้วัด (`unmeasured` — คำขอไปไม่ถึงโมเดล) กับข้อที่ถูกข้ามไม่ใช่ผลวัด · ด้านที่มีแต่
    แถวแบบนั้นต้องถูกเติมจากรอบก่อนเหมือนด้านที่ว่าง ไม่งั้นรอบที่เซิร์ฟเวอร์ตอบ 401 ครึ่งทาง
    จะเอาขีดกลางไปทับคะแนนที่วัดได้จริงเมื่อวาน
    """
    return any(isinstance(row, dict) and not row.get("unmeasured") and not row.get("skipped")
               for row in rows or [])


def remove(slug: str, keep_last: int = 0) -> int:
    """ลบผลวัดของโมเดลหนึ่ง คืนจำนวนไฟล์ที่ลบ

    `keep_last` > 0 = เก็บรอบล่าสุดไว้เท่านั้น · ผลสะสมเร็วกว่าที่คิดเพราะการวัดซ้ำเป็น
    เรื่องปกติ (ก่อน/หลังเปลี่ยน flag, ก่อน/หลังอัปเกรด engine) แล้วไม่มีใครกลับมาลบเอง
    """
    directory = slug_dir(slug)
    if not directory.is_dir():
        return 0
    # เฉพาะไฟล์ที่ record() เขียนเอง — ไม่ใช่ทุก *.json ที่เจอ (ดูหัวไฟล์: เคส package.json)
    runs = _run_files(directory)
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
        try:
            merged = latest_merged(directory.name)
        except BenchStoreError:
            # โฟลเดอร์ที่ชื่อไม่ใช่ slug หรือเป็น symlink ชี้ออกนอก — ไม่ใช่ของเรา ข้ามไป
            continue
        if merged:
            latest.append(merged)
    return latest
