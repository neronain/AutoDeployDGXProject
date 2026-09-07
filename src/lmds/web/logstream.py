"""ตาม log แบบเกือบ realtime ออกหน้าเว็บ (SSE) — `controller logs N -f` หนึ่ง process ต่อหนึ่งแผงที่เปิดอยู่

เจ้าของ 2026-09-07: "ส่วนของ log ทำให้เป็นการแสดง detail แบบเกือบ realtime ได้ไหม user จะได้ดูว่า error
อะไรด้วย แทนการกด 1 ครั้งแสดง 1 รอบ" — ปุ่ม logs เดิมเรียก controller ครั้งเดียวแล้วโชว์ก้อนนิ่ง ๆ ·
ตอน start ที่ตายกลางทาง ผู้ใช้ต้องกดซ้ำ ๆ แล้วเดาเอาว่าบรรทัดไหนคือสาเหตุ

กติกาที่ต้องรักษา:
- child ต้องตายเมื่อผู้ใช้ปิดแผง/แท็บ — ไม่งั้น docker logs -f / ssh ค้างสะสมบน hub ทั้งวัน · ฆ่าทั้ง
  process group (bash + docker logs/tail ข้างใน) เมื่อเราเป็นคนตั้ง session เอง · ssh ของ node ใช้
  terminate ธรรมดา (ไม่ใช่ group ของเรา — killpg ผิดตัวจะโดน hub เอง)
- keepalive ทุก ~15 วิ กัน proxy ปิดสายที่เงียบ (โมเดลที่ health แล้วอาจไม่พิมพ์อะไรเป็นนาที)
- จำกัดจำนวนสตรีมต่อ bundle และต่อ hub — แต่ละสตรีมคือ 1 process (+1 ssh) ค้างอยู่ตลอดที่แผงเปิด
- บรรทัดยาวเกิน (tokenizer dump, argv ยาวเป็นหน้า) ตัดที่ LINE_CAP — หน้าเว็บมีแผงเดียวไม่ใช่ที่เก็บ log
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
from typing import Callable

MAX_PER_BUNDLE = 3      # แผง Follow ของ bundle เดียวกันพร้อมกัน (หลายแท็บ/หลายคน)
MAX_PER_HUB = 12        # รวมทั้ง hub
LINE_CAP = 2000         # อักขระต่อบรรทัด
KEEPALIVE = 15.0        # วินาทีที่เงียบได้ก่อนส่ง comment กัน proxy ตัดสาย
POLL = 0.5              # ถี่แค่ไหนที่เช็คว่าผู้ใช้ปิดสายไปหรือยัง ระหว่างที่ log เงียบ
GRACE = 2.0             # รอ TERM ก่อน KILL
TAIL_DEFAULT = 200
RETRY_MS = 2000         # บอกเบราว์เซอร์ว่าถ้าสายจบให้ต่อใหม่ในกี่ ms (หน้าเว็บใช้ตอนรอ container โผล่ระหว่าง start)

_LOCK = threading.Lock()
_ACTIVE: dict[str, int] = {}


class TooManyStreams(Exception):
    """เกินเพดานสตรีม — ผู้เรียกแปลงเป็น 429 พร้อมข้อความนี้"""


def acquire(key: str) -> None:
    with _LOCK:
        total = sum(_ACTIVE.values())
        if total >= MAX_PER_HUB:
            raise TooManyStreams(
                f"hub ตาม log อยู่ {total} แผงแล้ว (เพดาน {MAX_PER_HUB}) — ปิดแผง Follow ที่ไม่ได้ดูก่อน")
        mine = _ACTIVE.get(key, 0)
        if mine >= MAX_PER_BUNDLE:
            raise TooManyStreams(
                f"{key} ถูกตามอยู่ {mine} แผงแล้ว (เพดาน {MAX_PER_BUNDLE} ต่อโมเดล) — ปิดแผง Follow ที่ไม่ได้ดูก่อน")
        _ACTIVE[key] = mine + 1


def release(key: str) -> None:
    with _LOCK:
        left = _ACTIVE.get(key, 0) - 1
        if left > 0:
            _ACTIVE[key] = left
        else:
            _ACTIVE.pop(key, None)


def active(key: str | None = None) -> int:
    with _LOCK:
        return _ACTIVE.get(key, 0) if key else sum(_ACTIVE.values())


def local_process(argv: list[str], cwd: str | None = None) -> subprocess.Popen:
    """spawn controller/docker/tail ในเครื่องนี้ — session ใหม่ให้ฆ่าได้ทั้งกลุ่ม"""
    return subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, cwd=cwd or None,
    )


def _own_group(proc: subprocess.Popen) -> bool:
    """เราเป็นคนตั้ง session ให้ process นี้ไหม (pgid == pid) — ไม่ใช่ = ห้าม killpg (จะโดนกลุ่มของ hub เอง)"""
    try:
        return os.getpgid(proc.pid) == proc.pid
    except (ProcessLookupError, OSError, AttributeError, TypeError):
        return False


def stop(proc: subprocess.Popen, grace: float = GRACE) -> None:
    """ปิด child ให้เรียบร้อย: TERM (ทั้งกลุ่มถ้าเป็นของเรา) → รอ grace → KILL · ไม่ระเบิดถ้ามันตายไปก่อนแล้ว"""
    if proc.poll() is not None:
        return
    # ssh ของ node: ปิด stdin ก่อน — ปลายทางถือ `cat` ไว้รอ EOF เพื่อฆ่า controller ของตัวเอง
    stdin = getattr(proc, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except OSError:
            pass
    group = _own_group(proc)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if group:
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def frame(line: str, ts: float | None = None) -> str:
    return "data: " + json.dumps(
        {"line": line[:LINE_CAP], "ts": round(ts if ts is not None else time.time(), 3)},
        ensure_ascii=False) + "\n\n"


def end_frame(code: int | None) -> str:
    return "event: end\ndata: " + json.dumps({"exit": code}) + "\n\n"


async def frames(request, key: str, spawn: Callable[[], subprocess.Popen]):
    """async generator ของ SSE — ผู้เรียกจองโควตา (acquire) มาแล้ว · คืนโควตาและปิด child ที่นี่เสมอ

    อ่าน stdout ใน thread แล้วส่งเข้า queue ของ event loop — readline() บล็อก ถ้าอ่านตรง ๆ ใน
    coroutine จะค้างทั้ง hub · ระหว่างที่ log เงียบวนเช็ค is_disconnected() ทุก POLL วิ ไม่ใช่รอบรรทัดถัดไป
    (โมเดลที่ health แล้วอาจเงียบเป็นนาที — ผู้ใช้ปิดแท็บไปแล้ว child ยังอยู่)
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    proc: subprocess.Popen | None = None
    try:
        yield f"retry: {RETRY_MS}\n\n"
        try:
            proc = spawn()
        except Exception as exc:  # noqa: BLE001 — ทุกอย่างที่ spawn ล้ม (ไม่มี ssh, NodeError) ต้องถึงผู้ใช้เป็นบรรทัด
            yield frame(f"เปิดสตรีม log ไม่ได้: {exc}")
            yield end_frame(-1)
            return
        assert proc.stdout is not None

        def pump() -> None:
            try:
                for raw in iter(proc.stdout.readline, b""):
                    text = raw.decode("utf-8", "replace").rstrip("\r\n")
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            except (OSError, ValueError, RuntimeError):
                pass
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except RuntimeError:
                    pass    # loop ปิดไปแล้ว (hub กำลังดับ) — ไม่มีใครรออ่าน

        threading.Thread(target=pump, daemon=True, name=f"logstream:{key}").start()
        idle = 0.0
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=POLL)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    break
                idle += POLL
                if idle >= KEEPALIVE:
                    idle = 0.0
                    yield ": keepalive\n\n"
                continue
            if item is None:
                yield end_frame(proc.wait())
                break
            idle = 0.0
            yield frame(item)
    finally:
        # ปิดใน thread แยก — finally นี้วิ่งตอน client ตัดสาย (CancelledError/GeneratorExit) ซึ่งห้ามบล็อก event loop
        if proc is not None and proc.poll() is None:
            threading.Thread(target=stop, args=(proc,), daemon=True, name=f"logstream-stop:{key}").start()
        release(key)


def response(request, key: str, spawn: Callable[[], subprocess.Popen]):
    """StreamingResponse ของ SSE — 429 ถ้าเกินเพดาน (ก่อนเปิดสาย เบราว์เซอร์จึงเห็นสถานะจริง ไม่ใช่สายที่จบเงียบ)"""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    try:
        acquire(key)
    except TooManyStreams as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return StreamingResponse(frames(request, key, spawn), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })
