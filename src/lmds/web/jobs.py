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
ALLOWED = {"prepare-runtime", "download", "verify-files", "start", "stop", "restart", "repair"}
_TAIL_LINES = 400


@dataclass
class Job:
    id: str
    slug: str
    command: str
    lines: deque = field(default_factory=lambda: deque(maxlen=_TAIL_LINES))
    exit_code: int | None = None
    process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self.exit_code is None

    def payload(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "command": self.command,
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


def start(slug: str, command: str, controller: str) -> Job:
    if command not in ALLOWED:
        raise JobError(f"คำสั่ง '{command}' ไม่อยู่ในรายการที่อนุญาต")
    path = Path(controller)
    if not path.is_file():
        raise JobError(f"ไม่พบ controller ของ {slug}")

    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(slug, ""))
        if current and current.running:
            raise JobError(f"{slug} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=slug, command=command)
        _JOBS[job.id] = job
        _ACTIVE[slug] = job.id

    def run() -> None:
        try:
            proc = subprocess.Popen(
                [str(path), command],
                cwd=str(path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            job.lines.append(f"เรียก controller ไม่ได้: {exc}\n")
            job.exit_code = 127
            return
        job.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            job.lines.append(line)
        job.exit_code = proc.wait()

    threading.Thread(target=run, daemon=True).start()
    return job
