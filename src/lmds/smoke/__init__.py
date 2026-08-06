"""Smoke test — พิสูจน์บนเครื่องจริงว่า bundle ใช้งานได้ ไม่ใช่แค่ผ่าน gate (roadmap เฟส 2 ข้อ 5)

gates ทั้ง 10 ด่านตรวจได้แค่ว่าสคริปต์ *ถูกต้อง* — ตอบไม่ได้ว่าโมเดลโหลดขึ้นจริงไหม
สถานะใน bundle artifact จึงเป็น `static-validated` เสมอ ส่วนผล runtimeเป็น machine-local
evidence ที่โมดูลนี้บันทึกนอก bundleและ `lmds doctor` คำนวณใหม่จากหลักฐานต้นทาง

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
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lmds
from lmds.fleet import run_root

SMOKE_FILE = "smoke.json"
SMOKE_SCHEMA_VERSION = 1
SKIP_CODE = 2
SKIPPABLE_TESTS = frozenset({"test-anthropic"})

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
_DISPATCH_COMMAND = re.compile(r"(?m)^\s*([a-z][a-z0-9-]+)\)")


class SmokeError(RuntimeError):
    """Smoke cannot run or persist a trustworthy result."""


class SmokeBusy(SmokeError):
    """Another smoke run for the same bundle owns the lifecycle lock."""


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


def available_commands(controller_text: str) -> set[str]:
    """Commands exposed by the controller's final dispatch."""
    return set(_DISPATCH_COMMAND.findall(controller_text))


def plan_steps(controller_text: str) -> list[tuple[str, str]]:
    """ลำดับที่ smoke จะเดิน — ไม่รวม stop ที่ต้องรันเสมอแม้ขั้นก่อนหน้าจะล้ม"""
    commands = available_commands(controller_text)
    stacked = {"sync-worker", "verify-worker"}.issubset(commands)
    native = "RUNTIME_MODE:-native" in controller_text
    has_runtime_assets = "prepare-runtime" in commands and not (
        "RUNTIME_MODE:-docker" in controller_text
    )

    if stacked:
        steps = [
            ("prepare-runtime", "เตรียม runtime ให้ตรงกันทุกเครื่อง"),
            ("download", "ดาวน์โหลดไฟล์โมเดล"),
            ("verify-files", "ตรวจไฟล์บนเครื่องหลัก"),
            ("sync-worker", "ส่งไฟล์โมเดลไปเครื่อง worker"),
            ("verify-worker", "ตรวจไฟล์บนเครื่อง worker"),
            ("start", "เปิดเซิร์ฟเวอร์แล้วรอ /health"),
        ]
    else:
        steps = [("download", "ดาวน์โหลดไฟล์โมเดล")]
        # Approved runtime assets are checked by verify-files, so prepare them first.
        # Native llama.cpp also needs its runtime before start; keeping one order makes
        # lmds up --smoke and standalone smoke deterministic.
        if native or has_runtime_assets:
            steps.append(("prepare-runtime", "เตรียม engine/runtime ที่อนุมัติแล้ว"))
        steps.extend(BASE_STEPS[1:])
    steps += [(name, TEST_LABELS.get(name, name)) for name in available_tests(controller_text)]
    return steps


@dataclass
class StepResult:
    command: str
    label: str
    code: int
    seconds: float
    detail: str = ""

    @property
    def skipped(self) -> bool:
        # Exit 2 is a capability skip only for explicitly optional probes. Treating
        # every test's exit 2 as a skip would let curl/test-text failures validate.
        return self.code == SKIP_CODE and self.command in SKIPPABLE_TESTS

    @property
    def ok(self) -> bool:
        return self.code == 0 or self.skipped


@dataclass
class SmokeRecord:
    slug: str
    controller: str
    fingerprint: str
    started_at: str
    finished_at: str = ""
    lmds_version: str = ""
    planned_steps: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    completed: bool = False
    stop_code: int | None = None
    stop_seconds: float = 0.0
    stop_detail: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.completed
            and bool(self.started_at)
            and bool(self.finished_at)
            and bool(self.planned_steps)
            and [s.command for s in self.steps] == self.planned_steps
            and all(s.ok for s in self.steps)
            and self.stop_code == 0
        )

    @property
    def stopped(self) -> bool:
        return self.stop_code == 0

    @property
    def failed_step(self) -> StepResult | None:
        return next((s for s in self.steps if not s.ok), None)

    @property
    def failure_code(self) -> int:
        failed = self.failed_step
        if failed is not None:
            return failed.code
        if self.stop_code not in (None, 0):
            return self.stop_code
        return 0 if self.passed else 1

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps) + self.stop_seconds


def record_path(slug: str) -> Path:
    return run_root() / slug / SMOKE_FILE


@contextmanager
def smoke_lock(slug: str):
    """Serialize destructive controller lifecycle actions for one bundle."""
    import fcntl

    directory = record_path(slug).parent
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = directory / ".smoke.lock"
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise SmokeError(f"สร้าง smoke lock ไม่สำเร็จ: {directory}: {exc}") from exc
    try:
        try:
            os.chmod(lock_path, 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SmokeBusy(f"มี smoke test ของ {slug} รันอยู่แล้ว") from exc
        except OSError as exc:
            raise SmokeError(f"ล็อก smoke ของ {slug} ไม่สำเร็จ: {exc}") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def write_record(record: SmokeRecord) -> Path:
    """Atomically persist the evidence; never claim success if this fails."""
    path = record_path(record.slug)
    payload = asdict(record)
    payload["schema_version"] = SMOKE_SCHEMA_VERSION
    tmp_name = ""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = ""
    except (OSError, TypeError, ValueError) as exc:
        raise SmokeError(f"บันทึกผล smoke ไม่สำเร็จ: {path}: {exc}") from exc
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass
    return path


def read_record(slug: str) -> SmokeRecord | None:
    path = record_path(slug)
    try:
        if path.stat().st_size > 1_000_000:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != SMOKE_SCHEMA_VERSION or data.get("slug") != slug:
            return None
        raw_steps = data.get("steps")
        raw_planned = data.get("planned_steps")
        if not isinstance(raw_steps, list) or not isinstance(raw_planned, list):
            return None
        if not all(isinstance(command, str) and command for command in raw_planned):
            return None

        steps: list[StepResult] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                return None
            code = item.get("code")
            seconds = item.get("seconds")
            if isinstance(code, bool) or not isinstance(code, int) or not -255 <= code <= 255:
                return None
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                return None
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds < 0:
                return None
            command = item.get("command")
            label = item.get("label")
            detail = item.get("detail", "")
            if not all(isinstance(value, str) for value in (command, label, detail)) or not command:
                return None
            steps.append(StepResult(command, label, code, seconds, detail))

        stop_code = data.get("stop_code")
        if isinstance(stop_code, bool) or not (
            stop_code is None or isinstance(stop_code, int) and -255 <= stop_code <= 255
        ):
            return None
        stop_seconds = data.get("stop_seconds", 0.0)
        if isinstance(stop_seconds, bool) or not isinstance(stop_seconds, (int, float)):
            return None
        stop_seconds = float(stop_seconds)
        if not math.isfinite(stop_seconds) or stop_seconds < 0:
            return None
        stop_detail = data.get("stop_detail", "")
        if not isinstance(stop_detail, str):
            return None
    except (OSError, ValueError, TypeError, OverflowError):
        return None
    return SmokeRecord(
        slug=str(data["slug"]),
        controller=str(data.get("controller", "")),
        fingerprint=str(data.get("fingerprint", "")),
        started_at=str(data.get("started_at", "")),
        finished_at=str(data.get("finished_at", "")),
        lmds_version=str(data.get("lmds_version", "")),
        planned_steps=list(raw_planned),
        steps=steps,
        completed=data.get("completed") is True,
        stop_code=stop_code,
        stop_seconds=stop_seconds,
        stop_detail=stop_detail,
    )


def validation_status(slug: str, controller: Path | str | None) -> tuple[str, str]:
    """คืน (สถานะ, เหตุผล) ของ bundle หนึ่งตัว — ตัวเดียวที่ยกเป็น hardware-validated ได้

    เงื่อนไขครบทั้งสามข้อเท่านั้น: มีผลบันทึกไว้ · ผลนั้นผ่านทุกขั้น · sha256 ของ controller
    ยังตรงกับตัวที่รันตอนนั้น
    """
    record = read_record(slug)
    if record is None:
        return STATUS_STATIC, "ยังไม่เคยรัน smoke test บนเครื่องนี้"
    if controller is None or not Path(controller).is_file():
        return STATUS_STATIC, "ไม่พบไฟล์ controller ที่เคยรัน — ยืนยันผลเดิมไม่ได้"
    try:
        controller_path = Path(controller)
        controller_text = controller_path.read_text(encoding="utf-8")
        fingerprint = controller_fingerprint(controller_path)
    except (OSError, UnicodeError):
        return STATUS_STATIC, "อ่านไฟล์ controller ไม่ได้ — ยืนยันผลเดิมไม่ได้"
    if fingerprint != record.fingerprint:
        return STATUS_STATIC, (
            f"controller ถูกแก้หลังรัน smoke test เมื่อ {record.started_at} — "
            "ผลเดิมใช้ยืนยันสคริปต์ตัวปัจจุบันไม่ได้"
        )
    failed = record.failed_step
    if failed is not None:
        return STATUS_STATIC, f"smoke test ล่าสุดไม่ผ่าน (ตกที่ {failed.command}) เมื่อ {record.started_at}"
    if record.stop_code not in (None, 0):
        return STATUS_STATIC, f"smoke test ล่าสุดหยุดเซิร์ฟเวอร์ไม่สำเร็จเมื่อ {record.started_at}"
    expected = [command for command, _ in plan_steps(controller_text)]
    if record.planned_steps != expected or [step.command for step in record.steps] != expected:
        return STATUS_STATIC, f"smoke test ล่าสุดรันไม่ครบทุกขั้นเมื่อ {record.started_at}"
    if record.stop_code != 0:
        return STATUS_STATIC, f"smoke test ล่าสุดหยุดเซิร์ฟเวอร์ไม่สำเร็จเมื่อ {record.started_at}"
    if not record.passed:
        return STATUS_STATIC, f"smoke test ล่าสุดไม่สมบูรณ์เมื่อ {record.started_at}"
    return STATUS_HARDWARE, f"smoke test ผ่านครบทุกขั้นเมื่อ {record.started_at}"


def new_record(slug: str, controller: Path | str, planned_steps: list[str]) -> SmokeRecord:
    return SmokeRecord(
        slug=slug,
        controller=str(controller),
        fingerprint=controller_fingerprint(controller),
        started_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        lmds_version=lmds.__version__,
        planned_steps=planned_steps,
    )


def finish_record(record: SmokeRecord) -> SmokeRecord:
    record.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return record


def timed(func) -> tuple[int, float]:
    """รัน callable ที่คืน exit code แล้วคืน (code, วินาทีที่ใช้)"""
    began = time.monotonic()
    code = func()
    return code, time.monotonic() - began


def _run_timed_step(run_step, command: str) -> tuple[int, float, str]:
    began = time.monotonic()
    try:
        code = run_step(command)
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError(f"run_step คืน exit code ไม่ถูกต้อง: {code!r}")
        return code, time.monotonic() - began, ""
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # subprocess launch/adapter failure is a failed step, not a traceback
        return 125, time.monotonic() - began, f"{type(exc).__name__}: {exc}"


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
    controller = Path(controller).resolve()
    if run_step is None:
        def run_step(command: str) -> int:
            import subprocess

            return subprocess.run(
                [str(controller), command],
                cwd=str(controller.parent),
            ).returncode

    try:
        controller_text = controller.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SmokeError(f"อ่าน controller ไม่สำเร็จ: {controller}: {exc}") from exc
    steps = plan_steps(controller_text)
    planned = [command for command, _ in steps]

    with smoke_lock(slug):
        try:
            record = new_record(slug, controller, planned)
        except OSError as exc:
            raise SmokeError(f"อ่าน controller ไม่สำเร็จ: {controller}: {exc}") from exc
        # Invalidate any older success before the first destructive action. An
        # interruption then leaves explicit incomplete evidence, not stale success.
        write_record(record)
        start_attempted = False
        finished_loop = False
        try:
            for index, (command, label) in enumerate(steps, start=1):
                if on_step is not None:
                    on_step(index, len(steps), command, label)
                if command == "start":
                    start_attempted = True
                code, seconds, detail = _run_timed_step(run_step, command)
                step = StepResult(command, label, code, seconds, detail)
                record.steps.append(step)
                write_record(record)
                if not step.ok:
                    break
            finished_loop = True
        finally:
            if start_attempted:
                code, seconds, detail = _run_timed_step(run_step, "stop")
                record.stop_code = code
                record.stop_seconds = seconds
                record.stop_detail = detail
            record.completed = finished_loop
            write_record(finish_record(record))
        return record


__all__ = [
    "BASE_STEPS",
    "SMOKE_FILE",
    "SMOKE_SCHEMA_VERSION",
    "SKIP_CODE",
    "SKIPPABLE_TESTS",
    "STATUS_HARDWARE",
    "STATUS_STATIC",
    "SmokeRecord",
    "SmokeBusy",
    "SmokeError",
    "StepResult",
    "available_tests",
    "available_commands",
    "controller_fingerprint",
    "finish_record",
    "new_record",
    "plan_steps",
    "read_record",
    "record_path",
    "run_smoke",
    "smoke_lock",
    "timed",
    "validation_status",
    "write_record",
]
