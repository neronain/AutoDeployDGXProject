"""Smoke test — พิสูจน์บนเครื่องจริงว่า bundle ใช้งานได้ ไม่ใช่แค่ผ่าน gate (roadmap เฟส 2 ข้อ 5)

gates ทั้ง 10 ด่านตรวจได้แค่ว่าสคริปต์ *ถูกต้อง* — ตอบไม่ได้ว่าโมเดลโหลดขึ้นจริงไหม
สถานะที่ bundle ได้จากขั้น generate จึงเป็น `static-validated` เสมอ และ README ก็เขียนค้างไว้ว่า
"สถานะจะอัปเดตเมื่อรัน acceptance tests" โดยไม่เคยมีคำสั่งไหนอัปเดตให้

โมดูลนี้เดินลำดับ acceptance เต็ม (download → verify → start/health → test-* → stop) แล้ว
**บันทึกผลไว้** — เป็นทางเดียวในโปรแกรมที่ทำให้ bundle เป็น `hardware-validated` ได้ ตามกฎข้อ 3
("ห้ามอ้าง hardware-validated โดยไม่ได้รันจริง")

สองอย่างที่ทำให้คำอ้างนี้เชื่อถือได้จริง:

- ผลผูกกับ **sha256 ของ controller ที่รัน** — แก้สคริปต์ทีหลังแล้วสถานะตกกลับเป็น
  static-validated เอง ไม่ต้องมีใครไปจำว่าเคยแก้อะไร
- ผลเก็บใน `~/.lmds/run/<slug>/` **ไม่ใช่ในโฟลเดอร์ bundle** เพราะ hardware-validated เป็น
  คุณสมบัติของ (bundle × เครื่อง) ไม่ใช่ของ bundle เดี่ยว ๆ · ส่ง ZIP ไปเครื่องอื่นแล้วสถานะ
  ต้องไม่ตามไปด้วย (และการเพิ่มไฟล์ในโฟลเดอร์ bundle จะทำให้ gate checksums ตกทันที)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lmds
from lmds.fleet import run_root

SMOKE_FILE = "smoke.json"

STATUS_HARDWARE = "hardware-validated"
STATUS_STATIC = "static-validated"

# ลำดับ acceptance ที่ README ของทุก bundle เขียนไว้ — ผิดลำดับแล้วพังแบบไม่บอกสาเหตุ
# (start ก่อน download = ไม่มีไฟล์ · ข้าม verify-files = ไฟล์ครึ่งเดียวแล้วไปตายตอนโหลด)
BASE_STEPS = [
    ("download", "ดาวน์โหลดไฟล์โมเดล"),
    ("verify-files", "ตรวจไฟล์ครบและถูกต้อง"),
    ("start", "เปิดเซิร์ฟเวอร์แล้วรอ /health"),
]

# คำอธิบายของ test-* ที่รู้จัก — ตัวที่ไม่รู้จักยังถูกรัน แค่ไม่มีคำอธิบายไทยกำกับ
TEST_LABELS = {
    "test-text": "ให้โมเดลตอบหนึ่งคำถาม",
    "test-reasoning": "ตรวจว่า reasoning parser แยก chain-of-thought ได้",
    "test-tools": "ตรวจว่า tool-call parser แปลงเป็น tool_calls ได้",
    "test-anthropic": "ตรวจว่า endpoint ตอบ /v1/messages ของ Anthropic ได้",
}

# dispatch ท้าย controller: `  test-text)    test_text ;;`
_DISPATCH_TEST = re.compile(r"(?m)^\s*(test-[a-z0-9-]+)\)")


def controller_fingerprint(controller: Path | str) -> str:
    """sha256 ของไฟล์ controller — ผลการรันผูกกับสคริปต์ตัวที่รันจริง ไม่ใช่แค่ชื่อ slug"""
    return hashlib.sha256(Path(controller).read_bytes()).hexdigest()


def available_tests(controller_text: str) -> list[str]:
    """คำสั่ง test-* ที่ bundle นี้มีจริง — อ่านจาก dispatch ไม่ใช่รายการที่ hardcode ไว้

    แต่ละ bundle มีไม่เท่ากัน (template ใส่ test-reasoning/test-tools ให้เฉพาะเมื่อ plan เปิดไว้
    และ llama.cpp มีแค่ test-text) · อ่านจากสคริปต์จริงแปลว่า test ที่เพิ่มเข้า template ทีหลัง
    ถูกรันให้เองโดยไม่ต้องกลับมาแก้ที่นี่
    """
    found = _DISPATCH_TEST.findall(controller_text)
    ordered = [t for t in ("test-text",) if t in found]
    ordered += sorted(t for t in set(found) if t not in ordered)
    return ordered


def plan_steps(controller_text: str) -> list[tuple[str, str]]:
    """ลำดับที่ smoke จะเดิน — ไม่รวม stop ที่ต้องรันเสมอแม้ขั้นก่อนหน้าจะล้ม"""
    steps = list(BASE_STEPS)
    # llama.cpp บน ARM64 ไม่มี image ทางการ ต้อง build เองก่อนหนึ่งครั้ง · ดูจาก RUNTIME_MODE
    # ที่ render ลงสคริปต์ ไม่ใช่จากการมีคำสั่ง prepare-runtime เพราะ template ใส่ dispatch case
    # นั้นไว้ทุก mode แม้ mode docker จะไม่ต้องใช้
    if "RUNTIME_MODE:-native" in controller_text:
        steps.insert(2, ("prepare-runtime", "เตรียม engine (ครั้งแรกครั้งเดียว — ใช้ sudo)"))
    steps += [(name, TEST_LABELS.get(name, name)) for name in available_tests(controller_text)]
    return steps


@dataclass
class StepResult:
    command: str
    label: str
    code: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.code == 0


@dataclass
class SmokeRecord:
    slug: str
    controller: str
    fingerprint: str
    started_at: str
    finished_at: str = ""
    lmds_version: str = ""
    steps: list[StepResult] = field(default_factory=list)
    stopped: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def failed_step(self) -> StepResult | None:
        return next((s for s in self.steps if not s.ok), None)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)


def record_path(slug: str) -> Path:
    return run_root() / slug / SMOKE_FILE


def write_record(record: SmokeRecord) -> Path:
    """เขียนผลลง run dir — ล้มเหลวก็ไม่ทำให้คำสั่งหลักพัง (เหมือน session log ของ brain)"""
    path = record_path(record.slug)
    payload = asdict(record)
    payload["passed"] = record.passed
    payload["status"] = STATUS_HARDWARE if record.passed else STATUS_STATIC
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return path


def read_record(slug: str) -> SmokeRecord | None:
    path = record_path(slug)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("slug"):
        return None
    steps = [
        StepResult(
            command=str(s.get("command", "")),
            label=str(s.get("label", "")),
            code=int(s.get("code", 1)),
            seconds=float(s.get("seconds", 0.0)),
        )
        for s in data.get("steps") or []
        if isinstance(s, dict)
    ]
    return SmokeRecord(
        slug=str(data["slug"]),
        controller=str(data.get("controller", "")),
        fingerprint=str(data.get("fingerprint", "")),
        started_at=str(data.get("started_at", "")),
        finished_at=str(data.get("finished_at", "")),
        lmds_version=str(data.get("lmds_version", "")),
        steps=steps,
        stopped=bool(data.get("stopped")),
    )


def validation_status(slug: str, controller: Path | str | None) -> tuple[str, str]:
    """คืน (สถานะ, เหตุผล) ของ bundle หนึ่งตัว — ตัวเดียวที่ยกเป็น hardware-validated ได้

    เงื่อนไขครบทั้งสามข้อเท่านั้น: มีผลบันทึกไว้ · ผลนั้นผ่านทุกขั้น · sha256 ของ controller
    ยังตรงกับตัวที่รันตอนนั้น
    """
    record = read_record(slug)
    if record is None:
        return STATUS_STATIC, "ยังไม่เคยรัน smoke test บนเครื่องนี้"
    if not record.passed:
        failed = record.failed_step
        where = f" (ตกที่ {failed.command})" if failed else ""
        return STATUS_STATIC, f"smoke test ล่าสุดไม่ผ่าน{where} เมื่อ {record.started_at}"
    if controller is None or not Path(controller).is_file():
        return STATUS_STATIC, "ไม่พบไฟล์ controller ที่เคยรัน — ยืนยันผลเดิมไม่ได้"
    if controller_fingerprint(controller) != record.fingerprint:
        return STATUS_STATIC, (
            f"controller ถูกแก้หลังรัน smoke test เมื่อ {record.started_at} — "
            "ผลเดิมใช้ยืนยันสคริปต์ตัวปัจจุบันไม่ได้"
        )
    return STATUS_HARDWARE, f"smoke test ผ่านครบทุกขั้นเมื่อ {record.started_at}"


def new_record(slug: str, controller: Path | str) -> SmokeRecord:
    return SmokeRecord(
        slug=slug,
        controller=str(controller),
        fingerprint=controller_fingerprint(controller),
        started_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        lmds_version=lmds.__version__,
    )


def finish_record(record: SmokeRecord) -> SmokeRecord:
    record.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return record


def timed(func) -> tuple[int, float]:
    """รัน callable ที่คืน exit code แล้วคืน (code, วินาทีที่ใช้)"""
    began = time.monotonic()
    code = func()
    return code, time.monotonic() - began


def run_smoke(
    controller: Path | str,
    slug: str,
    run_step=None,
    on_step=None,
) -> SmokeRecord:
    """เดิน acceptance เต็มลำดับแล้ว stop เสมอ — คืนผลที่บันทึกลง run dir แล้ว

    `run_step(command) -> exit code` แยกออกมาเป็นพารามิเตอร์เพื่อให้ CLI (เห็น output สด)
    กับ Web (capture ลง job log) ใช้ตัวเดียวกันได้ · ค่าเริ่มต้นคือรันตรงแบบเห็น output

    `stop` รันเสมอแม้ขั้นก่อนหน้าจะล้ม — smoke ที่ล้มหลัง start แล้วทิ้งเซิร์ฟเวอร์ค้างไว้
    คือของแถมที่ไม่มีใครอยากได้ (และทำให้ smoke รอบถัดไปชน port ตัวเอง)
    """
    controller = Path(controller)
    if run_step is None:
        def run_step(command: str) -> int:
            import subprocess

            return subprocess.run([str(controller), command]).returncode

    record = new_record(slug, controller)
    steps = plan_steps(controller.read_text(encoding="utf-8"))
    started = False

    for index, (command, label) in enumerate(steps, start=1):
        if on_step is not None:
            on_step(index, len(steps), command, label)
        code, seconds = timed(lambda: run_step(command))
        record.steps.append(StepResult(command, label, code, seconds))
        if command == "start" and code == 0:
            started = True
        if code != 0:
            break

    if started:
        record.stopped = run_step("stop") == 0

    write_record(finish_record(record))
    return record


__all__ = [
    "BASE_STEPS",
    "SMOKE_FILE",
    "STATUS_HARDWARE",
    "STATUS_STATIC",
    "SmokeRecord",
    "StepResult",
    "available_tests",
    "controller_fingerprint",
    "finish_record",
    "new_record",
    "plan_steps",
    "read_record",
    "record_path",
    "run_smoke",
    "timed",
    "validation_status",
    "write_record",
]
