"""ร่องรอยของคำสั่งที่เปลี่ยนสถานะ — ใครสั่งอะไรกับเครื่องไหนเมื่อไหร่

คอนโซลนี้ start/stop/ลบโมเดลได้ทุกเครื่องในทะเบียน และติดตั้ง LMDS ลง node ใหม่ได้
ผ่าน SSH ด้วยกุญแจของ hub · เวลามีอะไรผิดปกติ คำถามแรกคือ "ใครสั่ง" ซึ่งเดิมตอบไม่ได้เลย
— log ของ uvicorn ถูกตั้งไว้ที่ warning และถึงเปิดก็ไม่ได้เก็บไว้ที่ไหน

เก็บเฉพาะสิ่งที่ตอบคำถามนั้น: เวลา · IP ที่ยิงมา · method+path · ผลลัพธ์ · เวลาที่ใช้
**ไม่เก็บ** body, query string หรือ header — `require_token` รับ token ทาง `?token=`
ได้ด้วย การเก็บ query string จึงเท่ากับเขียน token ลงไฟล์ที่มีไว้ให้คนอ่าน

ไฟล์อยู่ที่ `~/.lmds/audit.log` โหมด 0600 · `$LMDS_AUDIT_LOG` ย้ายได้ (เช่นไปไว้ใน
ที่ที่ตัวเก็บ log ของศูนย์ดูดไปต่อ) · `$LMDS_AUDIT=0` ปิดทั้งหมด
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ไฟล์นี้มีไว้อ่านย้อนหลัง ไม่ใช่เก็บถาวร — เกินขนาดนี้หมุนไปเป็น .1 แล้วเริ่มใหม่
# เก็บรุ่นเดียวพอ: ประวัติยาวกว่านี้เป็นงานของตัวเก็บ log จริง ไม่ใช่ของ hub
MAX_BYTES = 5 * 1024 * 1024

_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.environ.get("LMDS_AUDIT", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}


def log_path() -> Path:
    override = os.environ.get("LMDS_AUDIT_LOG")
    if override:
        return Path(override)
    from lmds.fleet import run_root

    return run_root().parent / "audit.log"


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size < MAX_BYTES:
            return
    except OSError:
        return
    try:
        os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def record(method: str, path: str, *, ip: str = "", status: int = 0, ms: int = 0,
           actor: str = "") -> None:
    """เขียนหนึ่งบรรทัด — เงียบเสมอเมื่อเขียนไม่ได้

    ดิสก์เต็มหรือสิทธิ์ผิดไม่ควรทำให้คำสั่งที่ผู้ใช้สั่งล้มตาม: audit ที่หายไปหนึ่งบรรทัด
    แย่กว่าไม่มี audit นิดเดียว แต่คำสั่ง start ที่ล้มเพราะเขียน log ไม่ได้แย่กว่ามาก
    """
    if not enabled():
        return
    entry = {
        "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "ip": ip or "?",
        "method": method,
        "path": path,
        "status": status,
        "ms": ms,
    }
    if actor:
        entry["actor"] = actor
    try:
        target = log_path()
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            _rotate(target)
            # เปิดด้วย 0600 ตั้งแต่แรก — ไฟล์นี้บอกว่า IP ไหนคุยกับ hub นี้บ้าง
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            finally:
                os.close(fd)
    except OSError:
        pass


def read(limit: int = 50, *, path: Path | None = None) -> list[dict]:
    """รายการล่าสุด เรียงเก่า→ใหม่ · บรรทัดที่อ่านไม่ออกถูกข้าม ไม่ใช่ทำให้ทั้งไฟล์ใช้ไม่ได้"""
    target = path or log_path()
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines[-max(1, limit):]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


class Timer:
    """วัดเวลาของคำขอหนึ่ง — คืนเป็นมิลลิวินาที"""

    def __enter__(self) -> "Timer":
        self.started = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    @property
    def ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)
