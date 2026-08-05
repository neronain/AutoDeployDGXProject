"""งานที่ใช้เวลานาน (download หลายสิบ GB, start ที่โหลดโมเดลเป็นนาที)

HTTP request เดียวรอไม่ไหว และผู้ใช้ต้องเห็นว่ามันคืบหน้าอยู่ ไม่ใช่ค้าง —
จึงรัน controller เป็น subprocess แล้วให้หน้าเว็บ poll เอาบรรทัดล่าสุดไปแสดง

หนึ่ง slug รันได้ทีละงานเท่านั้น: download ซ้อน start คือทางลัดไปสู่ไฟล์พัง
"""

from __future__ import annotations

import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# คำสั่งที่ยอมให้หน้าเว็บสั่งได้ — ไม่รับชื่อคำสั่งจาก client ตรง ๆ
ALLOWED = {
    "prepare-runtime", "download", "verify-files", "start", "stop", "restart", "repair",
    # stacked (multi-node)
    "sync-worker", "verify-worker", "clear-fi-cache",
    # ทดสอบว่าโมเดลตอบจริง — CLI มีมาตลอด เว็บเพิ่งได้
    "test-text", "test-vision", "test-reasoning", "test-tools", "bench", "stress",
    "props", "info", "network-info", "client-config", "status", "wait-health", "doctor",
}

# download อย่างเดียวไม่พอที่จะบอกว่า "ไฟล์มาครบ" — CLI ให้รัน verify-files ต่อเสมอ
# หน้าเว็บจึงต่อให้เลย ไม่งั้นผู้ใช้ไม่มีทางรู้ว่าโหลดครบจริงไหม
CHAINS = {
    "download": ["download", "verify-files"],
    "repair": ["download", "verify-files"],  # repair = โหลดที่ขาด (resume) แล้วตรวจซ้ำ
}
_TAIL_LINES = 400


@dataclass
class Job:
    id: str
    slug: str
    command: str
    steps: list = field(default_factory=list)
    step_index: int = 0
    lines: deque = field(default_factory=lambda: deque(maxlen=_TAIL_LINES))
    exit_code: int | None = None
    process: subprocess.Popen | None = None

    def __setattr__(self, name: str, value) -> None:
        """งานจบ = สถานะบนดิสก์เปลี่ยนแล้ว (weight โหลดเสร็จ, server ขึ้น) — ทิ้งแคชทันที

        ผูกไว้กับการตั้ง exit_code แทนที่จะทำหลัง thread จบ เพราะ "จบ" ในสายตาคนอื่นคือ
        ตอน exit_code ไม่ใช่ None · ทำทีหลังจะมีช่วงที่หน้าเว็บเห็นว่างานจบแล้วแต่ยังได้ค่าเก่า
        """
        super().__setattr__(name, value)
        if name == "exit_code" and value is not None:
            from lmds.web.state import STORE

            STORE.invalidate_local()

    @property
    def running(self) -> bool:
        return self.exit_code is None

    def payload(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "command": self.command,
            "steps": self.steps,
            "step": self.steps[self.step_index] if self.step_index < len(self.steps) else "",
            "running": self.running,
            "exit_code": self.exit_code,
            "output": "".join(self.lines),
        }


class JobError(Exception):
    pass


_JOBS: dict[str, Job] = {}
_ACTIVE: dict[str, str] = {}  # slug -> job id
_LOCK = threading.Lock()


def active_for(slug: str) -> Job | None:
    with _LOCK:
        job = _JOBS.get(_ACTIVE.get(slug, ""))
    return job if job and job.running else None


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def controller_env(options: dict | None) -> dict:
    """แปลงตัวเลือกจากหน้าเว็บเป็น env ที่ controller อ่าน — ชุดเดียวกับที่ CLI ใช้

    ตั้งทั้ง CTX_SIZE และ MAX_MODEL_LEN เพราะ llama.cpp กับ vLLM อ่านคนละชื่อ
    (แต่ละสคริปต์อ่านแค่ของตัวเอง ตัวที่เกินมาไม่มีผล)
    """
    options = options or {}
    env: dict[str, str] = {}
    if options.get("port"):
        env["API_PORT"] = str(int(options["port"]))
    if options.get("bind"):
        env["API_HOST"] = str(options["bind"])
    if options.get("context"):
        env["CTX_SIZE"] = env["MAX_MODEL_LEN"] = str(int(options["context"]))
    if options.get("api_key"):
        env["API_KEY"] = str(options["api_key"])
    if options.get("slots"):
        # llama.cpp แบ่ง context เท่า ๆ กันให้ทุก slot — ตัวนี้คือ knob ที่ client-config บ่นถึง
        env["PARALLEL_SEQS"] = env["MAX_NUM_SEQS"] = str(int(options["slots"]))
    return env


def start_task(key: str, command: str, work) -> Job:
    """งานยาวที่ไม่ใช่ controller — เช่นติดตั้ง LMDS บนเครื่องอื่นผ่าน SSH

    `work()` คืน `(exit_code, output)` · ไม่ stream ทีละบรรทัดเพราะ ssh ฝั่งนี้รอผลทีเดียว
    (CLI ก็บล็อกแล้วพิมพ์ทีเดียวเหมือนกัน) — สิ่งที่ได้จากการเป็น job คือหน้าเว็บไม่ค้าง
    และมีที่ให้กันไม่ให้สั่งซ้อนกับงานเดิมของ key เดียวกัน

    key ต้องไม่ชนกับ slug ของโมเดล — ผู้เรียกใส่ prefix เอง (เช่น "node:spark2")
    """
    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(key, ""))
        if current and current.running:
            raise JobError(f"{key} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=key, command=command, steps=[command])
        _JOBS[job.id] = job
        _ACTIVE[key] = job.id

    def run() -> None:
        try:
            code, output = work()
        except Exception as exc:                     # noqa: BLE001 — งานเบื้องหลังต้องไม่ทำให้เว็บล้ม
            job.lines.append(f"{type(exc).__name__}: {exc}\n")
            job.exit_code = 1
            return
        for line in (output or "").splitlines(keepends=True):
            job.lines.append(line)
        job.exit_code = code

    threading.Thread(target=run, daemon=True).start()
    return job


def start(slug: str, command: str, controller: str, options: dict | None = None) -> Job:
    if command not in ALLOWED:
        raise JobError(f"คำสั่ง '{command}' ไม่อยู่ในรายการที่อนุญาต")
    path = Path(controller)
    if not path.is_file():
        raise JobError(f"ไม่พบ controller ของ {slug}")

    steps = CHAINS.get(command, [command])
    extra_env = controller_env(options)

    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(slug, ""))
        if current and current.running:
            raise JobError(f"{slug} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=slug, command=command, steps=steps)
        _JOBS[job.id] = job
        _ACTIVE[slug] = job.id

    def run() -> None:
        import os

        env = {**os.environ, **extra_env}
        for index, step in enumerate(steps):
            job.step_index = index
            if len(steps) > 1:
                job.lines.append(f"\n── {step} ({index + 1}/{len(steps)}) ──\n")
            try:
                proc = subprocess.Popen(
                    [str(path), step], cwd=str(path.parent), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
            except OSError as exc:
                job.lines.append(f"เรียก controller ไม่ได้: {exc}\n")
                job.exit_code = 127
                return
            job.process = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                job.lines.append(line)
            code = proc.wait()
            if code != 0:
                # ขั้นแรกล้ม = ไม่ต้องทำขั้นถัดไป (verify ไฟล์ที่โหลดไม่จบไม่มีประโยชน์)
                job.exit_code = code
                return
        job.exit_code = 0

    threading.Thread(target=run, daemon=True).start()
    return job
