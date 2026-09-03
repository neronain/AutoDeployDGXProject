"""แคชสถานะฝั่ง server + รีเฟรชเบื้องหลัง — หน้าเว็บไม่ต้องรอ SSH อีกต่อไป

ปัญหาที่แก้: เดิมหน้าเว็บ poll ทุก 5 วินาที และแต่ละรอบยิง SSH ไป *ทุกเครื่อง*
บน LAN 0.5ms ยังพอไหว แต่ผ่าน Tailscale relay ที่วัดได้จริง 82–154ms ต่อเครื่อง
หน้าเว็บจะกระตุกตลอดเวลาและยิ่งเพิ่มเครื่องยิ่งแย่ — ปัญหาเชิงสถาปัตยกรรม ไม่ใช่เรื่องหน้าตา

วิธีแก้: เครื่องเดียวที่คุยกับ node คือ refresher เบื้องหลัง · endpoint ทุกตัวอ่านจากแคช
จึงตอบทันทีเสมอ ไม่ว่าจะมีกี่เครื่องหรือเครื่องนั้นจะช้าแค่ไหน

หลักที่ยึด:
  - **หน้าเว็บต้องไม่บล็อกเพราะเครื่องหนึ่งช้า** — node ที่ช้าหรือล่มถูกถี่น้อยลงเอง
  - **บอกเสมอว่าข้อมูลเก่าแค่ไหน** (`age_seconds`) ดีกว่าโชว์ค่าที่อาจไม่จริงโดยไม่บอก
  - แคชอยู่ในหน่วยความจำของ process เดียว — `lmds web` เป็น process เดียวอยู่แล้ว
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# ทุกกี่วินาทีถึงจะรีเฟรช — เครื่องนี้ถูกกว่ามาก จึงถี่กว่า node ได้
LOCAL_INTERVAL = 3.0
NODE_INTERVAL = 15.0
# node ที่ต่อไม่ได้: ถอยออกไปเรื่อย ๆ จะได้ไม่เผา SSH ทิ้งทุก 15 วิ กับเครื่องที่ปิดอยู่
NODE_BACKOFF_MAX = 120.0


@dataclass
class Entry:
    """สถานะของแหล่งข้อมูลหนึ่งแหล่ง (เครื่องนี้ หรือ node หนึ่งเครื่อง)"""

    data: dict | None = None
    error: str = ""
    updated_at: float = 0.0
    interval: float = NODE_INTERVAL
    refreshing: bool = False

    @property
    def age_seconds(self) -> float:
        return 0.0 if not self.updated_at else round(time.time() - self.updated_at, 1)

    def payload(self) -> dict:
        return {
            "data": self.data,
            "error": self.error,
            "age_seconds": self.age_seconds,
            "stale": self.updated_at == 0.0,
            "refreshing": self.refreshing,
        }


class Store:
    """แคชกลาง + ตัวนับเวอร์ชัน — ตัวนับใช้บอก SSE ว่ามีอะไรเปลี่ยนจริงหรือเปล่า"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = Entry(interval=LOCAL_INTERVAL)
        self._nodes: dict[str, Entry] = {}
        self._version = 0
        # นับว่าแคชของเครื่องนี้ถูกทิ้งไปกี่ครั้ง — ใช้ตัดผลของ refresh ที่ออกตัวก่อนของจะเปลี่ยน
        self._local_epoch = 0
        self._changed = threading.Event()

    # ── อ่าน ────────────────────────────────────────────────────────────
    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def local_epoch(self) -> int:
        """เลขรุ่นของแคชเครื่องนี้ — refresher อ่านไว้ *ก่อน* ไปสำรวจ แล้วส่งคืนตอนเขียน"""
        with self._lock:
            return self._local_epoch

    def snapshot(self) -> dict:
        with self._lock:
            snap = {
                "version": self._version,
                "host": self._local.payload(),
                "nodes": {name: entry.payload() for name, entry in self._nodes.items()},
            }
        # งานที่กำลังรันไม่ได้อยู่ในแคช (มันเปลี่ยนทุกวินาที) — แปะทีหลังทุกครั้งที่ส่งออกไป
        # ไม่งั้นแถบความคืบหน้าจะหายทุกครั้งที่ snapshot มาถึง
        from lmds.web import jobs

        for name, entry in snap["nodes"].items():
            for model in ((entry.get("data") or {}).get("models") or []):
                job = jobs.active_for(model.get("slug", ""), name)
                model["job"] = job.payload() if job else None
        return snap

    def wait_for_change(self, since: int, timeout: float) -> bool:
        """รอจนกว่าจะมีอะไรเปลี่ยน — SSE ใช้แทนการให้เบราว์เซอร์ถามซ้ำ ๆ"""
        if self.version != since:
            return True
        self._changed.clear()
        return self._changed.wait(timeout)

    # ── เขียน ───────────────────────────────────────────────────────────
    def _bump(self) -> None:
        self._version += 1
        self._changed.set()

    def set_local(self, data: dict | None, error: str = "", *, epoch: int | None = None) -> bool:
        """เขียนผลสำรวจเครื่องนี้ลงแคช — คืน False ถ้าผลนั้นเก่าเกินกว่าจะเชื่อได้แล้ว

        `epoch` คือเลขรุ่นที่ผู้เรียกอ่านไว้ *ก่อน* เริ่มสำรวจ · ถ้าระหว่างที่สำรวจอยู่มีคำสั่ง
        ที่เปลี่ยนสถานะจริงมาทิ้งแคช (invalidate_local) เลขจะไม่ตรงกัน แปลว่าสิ่งที่ถืออยู่
        เป็นภาพก่อนการเปลี่ยนแปลง — ต้องทิ้ง ไม่ใช่เขียนทับ

        เดิมเขียนทับเสมอ จึงมีจังหวะที่ refresher สำรวจเสร็จ *ก่อน* ผู้ใช้กดลบ แล้วมาเขียน
        ผลเก่าลงแคช *หลัง* ลบเสร็จ — โมเดลที่ลบไปแล้วโผล่กลับมาในหน้าเว็บจนกว่าจะครบรอบถัดไป
        """
        with self._lock:
            if epoch is not None and epoch != self._local_epoch:
                return False
            self._local.data, self._local.error = data, error
            self._local.updated_at = time.time()
            self._local.refreshing = False
            self._bump()
            return True

    def set_node(self, name: str, data: dict | None, error: str = "") -> None:
        with self._lock:
            entry = self._nodes.setdefault(name, Entry())
            entry.data, entry.error = data, error
            entry.updated_at = time.time()
            entry.refreshing = False
            # ต่อไม่ได้ = ถอยห่างขึ้นเรื่อย ๆ · ต่อได้เมื่อไรกลับมาถี่ปกติทันที
            entry.interval = min(entry.interval * 2, NODE_BACKOFF_MAX) if error else NODE_INTERVAL
            self._bump()

    def mark_refreshing(self, name: str | None) -> None:
        """บอกหน้าเว็บว่ากำลังไปเอาข้อมูลอยู่ — ผู้ใช้จะได้รู้ว่ากดแล้วมีอะไรเกิดขึ้น"""
        with self._lock:
            entry = self._local if name is None else self._nodes.setdefault(name, Entry())
            entry.refreshing = True
            self._bump()

    def due(self, name: str | None) -> bool:
        with self._lock:
            entry = self._local if name is None else self._nodes.get(name)
            if entry is None:
                return True
            return time.time() - entry.updated_at >= entry.interval

    def force(self, name: str | None = None) -> None:
        """ทำให้ครบกำหนดทันที — ใช้ตอนผู้ใช้กด refresh หรือหลังสั่งงานที่เปลี่ยนสถานะ"""
        with self._lock:
            entry = self._local if name is None else self._nodes.setdefault(name, Entry())
            entry.updated_at = 0.0
            entry.interval = NODE_INTERVAL if name is not None else LOCAL_INTERVAL

    def invalidate_local(self) -> None:
        """ทิ้งแคชของเครื่องนี้ — ใช้หลังคำสั่งที่เปลี่ยนสถานะจริง (start/stop/remove)

        ต่างจาก force() ตรงที่ force แค่ "ถึงกำหนดแล้ว" ซึ่งยังอ่านค่าเก่าได้จนกว่า refresher
        จะวน · อันนี้ลบทิ้งเลย คำขอถัดไปจึงคำนวณสด ผู้ใช้ไม่มีทางเห็นของที่เพิ่งลบไป

        ต้องเรียก **หลัง** คำสั่งทำงานเสร็จเสมอ ไม่ใช่ก่อน — ทิ้งแคชก่อนของจริงเปลี่ยน
        เท่ากับเปิดช่องให้ refresher มาเติมภาพก่อนเปลี่ยนกลับเข้าไปใหม่ได้ทัน
        """
        with self._lock:
            self._local.data = None
            self._local.updated_at = 0.0
            # ขยับเลขรุ่น: ผลสำรวจใด ๆ ที่ออกตัวไปก่อนบรรทัดนี้ถือเป็นของเก่า เขียนลงไม่ได้แล้ว
            self._local_epoch += 1
            self._bump()

    def drop_missing(self, keep: set[str]) -> None:
        with self._lock:
            for name in [n for n in self._nodes if n not in keep]:
                del self._nodes[name]
            self._bump()


STORE = Store()


def _refresh_local() -> None:
    from lmds.inventory import host_payload, model_payload
    from lmds.fleet import discover
    from lmds.web import jobs

    # อ่านเลขรุ่นก่อนลงมือสำรวจ — discover() ใช้เวลาหลายสิบ ms ซึ่งนานพอให้ผู้ใช้กด
    # ลบ/หยุดคั่นกลางได้ · ถ้าเกิดขึ้นจริง เลขจะไม่ตรงตอนเขียน แล้ว set_local จะทิ้งผลนี้ให้
    epoch = STORE.local_epoch
    try:
        models = [model_payload(s, _job_payload(jobs, s.slug)) for s in discover()]
        STORE.set_local({"host": host_payload(), "models": models}, epoch=epoch)
    except Exception as exc:  # noqa: BLE001 — refresher ต้องไม่ตายเพราะเคสเดียว
        STORE.set_local(None, str(exc)[:300], epoch=epoch)


def _job_payload(jobs_module, slug: str) -> dict | None:
    job = jobs_module.active_for(slug)
    return job.payload() if job else None


def _refresh_node(name: str) -> None:
    from lmds.nodes import NodeError, find, probe, status_from_probe, update

    node = find(name)
    if node is None:
        return
    try:
        info = probe(node)
        STORE.set_node(name, info)
        update(name, last_error="", **status_from_probe(info))
    except NodeError as exc:
        STORE.set_node(name, None, str(exc)[:300])
        try:
            update(name, last_error=str(exc)[:200])
        except NodeError:
            pass
    except Exception as exc:  # noqa: BLE001 — ต้องลงแคชเสมอ ไม่งั้น due() เป็นจริงทุกวิ
        # probe ที่ล้มด้วยอย่างอื่นที่ไม่ใช่ NodeError (เช่น UnicodeDecodeError จาก motd ที่ไม่ใช่
        # UTF-8) เดิมหลุดขึ้นไปให้ _submit_node กลืนเงียบ ๆ โดยไม่เขียนอะไรลงแคช → เครื่องนั้น
        # ครบกำหนดตลอด ถูก SSH ซ้ำทุกวินาที และการ์ดขึ้น "ไม่มีข้อมูล" ตลอดกาลโดยไม่บอกสาเหตุ
        STORE.set_node(name, None, f"probe ล้มเหลว ({type(exc).__name__}): {exc}"[:300])


# ── สำรวจ node พร้อมกันหลายเครื่อง ────────────────────────────────────────
#
# เดิมลูปเดียวไล่ SSH ทีละเครื่องแบบเรียงคิว · บน LAN ไม่รู้สึกอะไร แต่ผ่าน Tailscale
# relay การ probe หนึ่งเครื่องกินเวลาระดับวินาที (ต่อ SSH ใหม่ + รันคำสั่งปลายทาง)
# 14 เครื่องจึงใช้เวลารอบละหลายสิบวินาที — NODE_INTERVAL=15 ที่ตั้งไว้ไม่เคยเป็นจริง
# เพราะรอบหนึ่งยังไม่ทันจบก็เลยกำหนดของทุกเครื่องไปแล้ว
#
# ที่แย่กว่านั้นคือ `_refresh_local()` อยู่หัวลูปเดียวกัน · ครบกำหนดทุก 3 วิ แต่กว่าลูป
# จะวนกลับมาถึงต้องรอ SSH ครบทั้ง 14 เครื่องก่อน — กราฟของ *เครื่องนี้เอง* จึงตามหลัง
# ของจริงหลายวินาที ซึ่งเป็นอาการที่ผู้ใช้รายงานเข้ามาตรง ๆ
#
# แก้ด้วยการโยน probe ลง thread pool: ลูปไม่ต้องรอใคร กราฟเครื่องนี้เดินตาม 3 วิจริง ๆ
# และรอบของ node ก็จบใน ceil(14/8) ชุด ไม่ใช่ 14 ชุด
NODE_WORKERS = 8

_inflight: set[str] = set()
_inflight_lock = threading.Lock()


def _submit_node(pool, name: str) -> None:
    """สั่ง probe หนึ่งเครื่องแบบไม่บล็อก — เครื่องที่ยังค้างอยู่จะไม่ถูกสั่งซ้ำ

    ต้องกันซ้ำเอง เพราะ `due()` ยังเป็นจริงตลอดจนกว่าผลจะเขียนกลับ · ถ้าไม่กัน
    เครื่องที่ช้าจะโดนสั่งซ้ำทุกวินาทีจนคิวเต็มไปด้วยงานของเครื่องเดียว
    """
    with _inflight_lock:
        if name in _inflight:
            return
        _inflight.add(name)

    def run() -> None:
        try:
            _refresh_node(name)
        except Exception:  # noqa: BLE001 — เครื่องเดียวล้มไม่ควรฆ่า worker ทั้งตัว
            pass
        finally:
            with _inflight_lock:
                _inflight.discard(name)

    try:
        pool.submit(run)
    except RuntimeError:      # pool ปิดไปแล้วระหว่างที่กำลังหยุด refresher
        with _inflight_lock:
            _inflight.discard(name)


def _loop(stop: threading.Event) -> None:
    """ตัวเดียวที่สั่งงาน — ทุก endpoint อ่านจากแคชที่ worker เติมให้"""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=NODE_WORKERS, thread_name_prefix="lmds-probe") as pool:
        while not stop.is_set():
            try:
                if STORE.due(None):
                    _refresh_local()

                from lmds.nodes import load

                names = {n.name for n in load()}
                STORE.drop_missing(names)
                for name in names:
                    if stop.is_set():
                        break
                    if STORE.due(name):
                        _submit_node(pool, name)
            except Exception:  # noqa: BLE001 — วนต่อเสมอ ล้มรอบเดียวไม่ควรหยุดทั้งระบบ
                pass
            stop.wait(1.0)


_thread: threading.Thread | None = None
_stop = threading.Event()


def start_refresher() -> None:
    """เริ่ม refresher เบื้องหลัง — เรียกซ้ำได้ ไม่สร้างซ้อน"""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(_stop,), daemon=True, name="lmds-refresh")
    _thread.start()


def stop_refresher(wait: float = 5.0) -> None:
    """หยุด refresher แล้ว **รอให้รอบที่ค้างอยู่จบจริง**

    ตั้ง event เฉย ๆ ไม่พอ — thread อาจกำลังอยู่กลาง `_refresh_local()` แล้วเขียนผลลง
    STORE หลังจากที่ผู้เรียกคิดว่าหยุดไปแล้ว · เป็นอาการเดียวกับที่แก้ไปหลายรอบในรอบนี้:
    บอกว่าเสร็จทั้งที่ยังไม่เสร็จ
    """
    global _thread
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=wait)
    _thread = None
