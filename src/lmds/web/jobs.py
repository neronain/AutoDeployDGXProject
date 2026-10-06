"""งานที่ใช้เวลานาน (download หลายสิบ GB, start ที่โหลดโมเดลเป็นนาที)

HTTP request เดียวรอไม่ไหว และผู้ใช้ต้องเห็นว่ามันคืบหน้าอยู่ ไม่ใช่ค้าง —
จึงรัน controller เป็น subprocess แล้วให้หน้าเว็บ poll เอาบรรทัดล่าสุดไปแสดง

หนึ่ง slug รันได้ทีละงานเท่านั้น: download ซ้อน start คือทางลัดไปสู่ไฟล์พัง
"""

from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import threading
import time
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
    # llama.cpp: build/image รู้จัก arch ของโมเดลไหม · update-runtime = prepare-runtime ที่ข้าม lock
    # (LLAMA_CPP_UPDATE=1) — ปุ่มแก้ของป้าย "runtime older than model" (spark-worker 2026-09-06)
    "check-runtime", "update-runtime",
}

# download อย่างเดียวไม่พอที่จะบอกว่า "ไฟล์มาครบ" — CLI ให้รัน verify-files ต่อเสมอ
# หน้าเว็บจึงต่อให้เลย ไม่งั้นผู้ใช้ไม่มีทางรู้ว่าโหลดครบจริงไหม
CHAINS = {
    "download": ["download", "verify-files"],
    "repair": ["download", "verify-files"],  # repair = โหลดที่ขาด (resume) แล้วตรวจซ้ำ
    "update-runtime": ["prepare-runtime"],
}

# env ที่คำสั่งของหน้าเว็บบางตัวต้องตั้งให้ controller — ชื่อปุ่มพูดกับผู้ใช้ ส่วน env พูดกับสคริปต์
COMMAND_ENV = {
    "update-runtime": {"LLAMA_CPP_UPDATE": "1"},
}
_TAIL_LINES = 400

# ตัวจบบรรทัดที่นับ — \r ด้วย ไม่ใช่ \n อย่างเดียว
#
# `for line in proc.stdout` ตัดที่ \n เท่านั้น · progress bar ของ huggingface_hub,
# docker pull, rsync และ curl เลื่อนตัวเลขด้วย \r โดยไม่ขึ้นบรรทัดใหม่เลย
# download 50 GB จึงไม่เคยส่งบรรทัดไหนออกมาสักบรรทัด — หน้าเว็บได้แผงว่าง ๆ นิ่ง
# อยู่ครึ่งชั่วโมง แล้วผู้ใช้สรุปว่างานค้าง ทั้งที่ไฟล์กำลังไหลเข้าเครื่องอยู่
_LINE_END = re.compile(r"\r\n|\r|\n")
_READ_CHUNK = 8192
_PUMP_POLL = 0.25         # ถี่แค่ไหนที่ _pump เงยหน้าดูว่างานถูกสั่งให้เลิกอ่านหรือยัง ระหว่างที่ท่อเงียบ


def _pump(job: "Job", proc: subprocess.Popen, secrets: list[str] | None = None) -> None:
    """ย้ายผลจาก process เข้า job ทีละบรรทัด — นับ \\r เป็นตัวจบบรรทัดด้วย

    บรรทัดที่จบด้วย \\r คือ "เฟรม" ของ progress bar ตัวถัดไปตั้งใจจะทับของเดิม
    ไม่ใช่ต่อท้าย · ถ้า append ทุกเฟรม deque 400 บรรทัดจะเต็มไปด้วยเลข % ของ
    วินาทีที่แล้ว แล้วดันบรรทัดที่บอกสาเหตุจริงหายไปหมด

    `secrets` = ค่าที่ต้องไม่โผล่ในผลงาน (token ที่ยืมให้ node) — กรอง *ตอนรับแต่ละบรรทัด*
    ไม่ใช่หลังท่อปิด: download กินเวลาเป็นสิบนาที หน้าเว็บ poll ทุกวิและได้บรรทัดที่มี token
    เต็ม ๆ ไปแสดง (curl -v, สคริปต์ที่ echo env) ก่อนที่ _scrub_secrets จะได้ทำงาน
    """
    # read1() ไม่ใช่ read(): คืนเท่าที่มีอยู่ทันที ส่วน read(n) จะรอจนครบ n
    # ซึ่งแปลว่าไม่สตรีม · ไม่ใช้ os.read(fileno) เพราะผูกกับ pipe จริงโดยไม่จำเป็น
    # แล้วทำให้เทสที่จำลอง subprocess ต้องมี fd จริงตามไปด้วย
    stream = proc.stdout
    read = getattr(stream, "read1", None) or stream.read
    pending = ""
    overwrite = False   # เฟรมก่อนหน้าจบด้วย \r → บรรทัดถัดไปทับตัวเดิม
    values = [v for v in (secrets or []) if v]
    if values:
        from lmds.secrets import redact

    def emit(text: str) -> None:
        nonlocal overwrite
        if values:
            text = redact(text, values)
        if overwrite and job.lines:
            job.lines[-1] = text
        else:
            job.lines.append(text)

    # ท่อจริง (มี fd) รอด้วย select ทีละช่วงสั้น ๆ แทนการบล็อกใน read ไม่มีกำหนด — งานที่ถูกยกเลิกจน
    # process ทั้งกลุ่มหายแล้ว แต่มีตัวที่หนีออกนอกกลุ่ม (setsid) ถือปลายท่อค้างไว้ จะได้เลิกอ่านและปล่อยล็อกได้
    # · ของปลอมในเทสที่ไม่มี fd ใช้ทางเดิม
    try:
        fd = stream.fileno()
    except Exception:  # noqa: BLE001 — ไม่มี fd (ของปลอม · BytesIO) ไม่ใช่ข้อผิดพลาด
        fd = None
    while True:
        if fd is not None:
            try:
                ready, _, _ = select.select([fd], [], [], _PUMP_POLL)
            except (OSError, ValueError):
                break
            if not ready:
                if getattr(job, "abandon_output", False):
                    break
                continue
        try:
            data = read(_READ_CHUNK)
        except (OSError, ValueError):   # process ตายกลางคัน — ท่อปิดไปแล้ว
            break
        if not data:
            break
        pending += data.decode("utf-8", "replace")
        while True:
            match = _LINE_END.search(pending)
            if match is None:
                break
            emit(pending[: match.start()] + "\n")
            overwrite = match.group() == "\r"
            pending = pending[match.end() :]
    if pending:        # เศษท้ายที่ไม่มีตัวจบบรรทัด — อย่าให้หายไปเฉย ๆ
        emit(pending + "\n")


@dataclass
class Job:
    id: str
    slug: str
    command: str
    # ว่าง = โมเดลในเครื่องนี้ · มีค่า = ชื่อเครื่องในทะเบียนที่งานนี้ไปรัน
    node: str = ""
    steps: list = field(default_factory=list)
    step_index: int = 0
    lines: deque = field(default_factory=lambda: deque(maxlen=_TAIL_LINES))
    exit_code: int | None = None
    process: subprocess.Popen | None = None
    # งานที่เงียบสนิทกับงานที่ตายไปแล้ว หน้าตาเหมือนกันเป๊ะถ้าไม่บอกเวลา — และบางขั้น
    # (verify-files ของ shard 50 GB) เงียบจริง ๆ โดยไม่มีอะไรผิด
    started_at: float = field(default_factory=time.monotonic)
    # ผู้ใช้กดยกเลิกแล้ว — งานหลายขั้น (download → verify-files) ต้องไม่เริ่มขั้นถัดไป
    cancel_requested: bool = False
    # process ของงานหายหมดแล้วแต่ยังมีคนนอกกลุ่มถือท่อ — _pump เลิกรออ่าน งานจบและล็อกหลุดได้
    abandon_output: bool = False

    def __setattr__(self, name: str, value) -> None:
        """งานจบ = สถานะบนดิสก์เปลี่ยนแล้ว (weight โหลดเสร็จ, server ขึ้น) — ทิ้งแคชทันที

        ผูกไว้กับการตั้ง exit_code แทนที่จะทำหลัง thread จบ เพราะ "จบ" ในสายตาคนอื่นคือ
        ตอน exit_code ไม่ใช่ None · ทำทีหลังจะมีช่วงที่หน้าเว็บเห็นว่างานจบแล้วแต่ยังได้ค่าเก่า
        """
        super().__setattr__(name, value)
        if name == "exit_code" and value is not None:
            from lmds.web.state import STORE

            # งานบนเครื่องอื่นไม่ได้เปลี่ยนสถานะของเครื่องนี้ — ทิ้งแคชของเครื่องนั้นแทน
            if getattr(self, "node", ""):
                STORE.force(self.node)
            else:
                STORE.invalidate_local()

    @property
    def running(self) -> bool:
        return self.exit_code is None

    def payload(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "node": self.node,
            "command": self.command,
            "steps": self.steps,
            "step": self.steps[self.step_index] if self.step_index < len(self.steps) else "",
            "running": self.running,
            "exit_code": self.exit_code,
            "output": "".join(self.lines),
            "elapsed": int(time.monotonic() - self.started_at),
        }


class JobError(Exception):
    pass


_JOBS: dict[str, Job] = {}
_ACTIVE: dict[str, str] = {}  # slug -> job id
_LOCK = threading.Lock()


def _key(slug: str, node: str = "") -> str:
    """หนึ่งงานต่อ (เครื่อง, โมเดล) — slug เดียวกันบนคนละเครื่องคือคนละงาน"""
    return f"{node}/{slug}" if node else slug


def active_for(slug: str, node: str = "") -> Job | None:
    with _LOCK:
        job = _JOBS.get(_ACTIVE.get(_key(slug, node), ""))
    return job if job and job.running else None


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


# ── ยกเลิกงาน ──
#
# audit 2026-10: cancel เดิมเรียก proc.terminate() ตัวเดียว และงานถูกสตาร์ตโดยไม่มี session ของตัวเอง
# จึงไม่มี process group ให้ส่งสัญญาณ · ที่ตายคือ bash ของ controller เท่านั้น ลูกของมัน (docker · curl ·
# aria2c · python ที่กำลังโหลด 70 GB) รันต่อ และยังถือปลายท่อ stdout ไว้ — _pump จึงอ่านต่อ งานค้าง
# `running` และล็อก (เครื่อง, โมเดล) ไม่หลุดจนกว่าตัวกำพร้าจะจบเอง ขณะที่ HTTP ตอบ `cancelled: true` ไปแล้ว
# ตอนนี้: งานรันใน session/process group ของตัวเอง · cancel ส่ง TERM ทั้งกลุ่ม รอ แล้ว KILL · รอจนกลุ่ม
# หายจริงก่อนจะตอบว่ายกเลิกแล้ว — หรือตอบตามจริงว่ายังมี process เหลือ (แบบเดียวกับ logstream.stop)
# คำสั่งที่สั่ง docker สร้าง container — ตัว container เป็นลูกของ dockerd ไม่ใช่ของงาน ฆ่ากลุ่มของงานแล้ว
# `docker run -d` ของ start (และ container โหลดที่ไม่รับ SIGTERM) ยังอยู่ได้ · ผลของ cancel ต้องบอกเรื่องนี้
MAY_LEAVE_CONTAINERS = {"start", "restart", "download", "repair", "prepare-runtime", "update-runtime"}
CANCEL_GRACE = 3.0       # รอหลัง TERM — docker CLI ต้องมีเวลาส่งสัญญาณต่อให้ container ก่อนโดน KILL
CANCEL_KILL_WAIT = 5.0    # รอหลัง KILL ให้ kernel เก็บกวาดและ thread ของงานปล่อยล็อก


def _group_of(proc) -> int | None:
    """pgid ของกลุ่มที่เราตั้งให้ process นี้เอง — None เมื่อไม่ใช่ (ssh ของ node · ของปลอมในเทส)

    ไม่ใช่กลุ่มของเรา = ห้าม killpg: จะไปโดนกลุ่มของ hub เอง
    """
    try:
        pid = int(proc.pid)
        return pid if os.getpgid(pid) == pid else None
    except (ProcessLookupError, OSError, AttributeError, TypeError, ValueError):
        return None


def _alive(proc, group: int | None) -> bool:
    """ยังมี process ของงานนี้เหลืออยู่ไหม — ทั้งกลุ่ม ไม่ใช่แค่ตัวหัว"""
    try:
        exited = proc.poll() is not None      # เก็บศพตัวหัวด้วย — zombie ทำให้กลุ่มดูเหมือนยังอยู่
    except Exception:  # noqa: BLE001 — ของปลอม/ท่อพัง: ถือว่าจบ
        return False
    if group is None:
        return not exited
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                           # มีลูกที่เป็นของ user อื่น (sudo) — ยังอยู่ แต่เราฆ่าไม่ได้
    except OSError:
        return not exited
    return True


def _signal(proc, group: int | None, sig: int) -> None:
    try:
        if group is not None:
            os.killpg(group, sig)
        elif sig == signal.SIGKILL:
            proc.kill()
        else:
            getattr(proc, "terminate", proc.kill)()
    except (ProcessLookupError, OSError):
        pass


def _adopt(job: "Job", proc) -> None:
    """ผูก process ที่เพิ่ง spawn เข้ากับงาน — ถ้าผู้ใช้กดยกเลิกไปก่อนแล้ว ฆ่าทั้งกลุ่มทันที

    ช่องว่างระหว่าง Popen กับการผูก: cancel ที่มาถึงตอนนั้นมองไม่เห็น process นี้ จึงไม่ได้ฆ่ามัน
    """
    job.process = proc
    if job.cancel_requested:
        _signal(proc, _group_of(proc), signal.SIGKILL)


def _wait_gone(proc, group: int | None, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _alive(proc, group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def cancel(job: Job, grace: float | None = None) -> dict:
    """หยุดงานที่ยังรันอยู่ทั้งกลุ่ม แล้วรายงานตามที่เกิดขึ้นจริง

    คืน {"cancelled", "signalled", "still_running", "lock_released", "detail"}:
      · `cancelled` = True เฉพาะเมื่อ process ของงาน **ทุกตัว** หายแล้ว (TERM → รอ → KILL → รอ)
      · `signalled` = มีอะไรให้ฆ่า (False = งานจบไปก่อนแล้ว)
      · `still_running` = ส่งสัญญาณแล้วแต่ยังมี process เหลือ — ล็อกยังไม่หลุด อย่าบอกผู้ใช้ว่ายกเลิกแล้ว
      · `lock_released` = thread ของงานตั้ง exit_code แล้ว (ล็อก (เครื่อง, โมเดล) หลุด)

    ไม่ตั้ง exit_code เอง: กลุ่มตาย → ท่อปิด → _pump จบ → thread ของงานเป็นคนตั้ง ซึ่งคือจังหวะที่ล็อกหลุด
    ตั้งจากตรงนี้ซ้อนกันจะได้ค่าสองรอบและ invalidate แคชสองครั้ง
    """
    if not job.running:
        return {"cancelled": False, "signalled": False, "still_running": False,
                "lock_released": True, "detail": "the job had already finished"}
    job.cancel_requested = True                 # ขั้นถัดไป (หรือขั้นแรกที่ยังไม่ทัน spawn) ต้องไม่เริ่ม
    job.lines.append("\n── ผู้ใช้กดยกเลิก ──\n")
    proc = job.process
    if proc is None:
        # กดเร็วกว่าที่ thread ของงานจะ spawn — ไม่มีอะไรให้ฆ่า · thread เห็นธงแล้วจบงานเองโดยไม่เริ่ม
        deadline = time.monotonic() + 2.0
        while job.running and job.process is None and time.monotonic() < deadline:
            time.sleep(0.02)
        proc = job.process
        if proc is None:
            return {"cancelled": not job.running, "signalled": False, "still_running": job.running,
                    "lock_released": not job.running,
                    "detail": "" if not job.running else
                    "nothing to signal — this job runs inside the hub (or has not spawned its command yet) "
                    "and is still running"}
    group = _group_of(proc)
    _signal(proc, group, signal.SIGTERM)
    gone = _wait_gone(proc, group, CANCEL_GRACE if grace is None else grace)
    if not gone:
        _signal(proc, group, signal.SIGKILL)
        gone = _wait_gone(proc, group, CANCEL_KILL_WAIT)
    if not gone:
        job.lines.append("ยกเลิกไม่สำเร็จ: ยังมี process ของงานนี้เหลืออยู่หลัง SIGKILL — ล็อกยังไม่หลุด\n")
        return {"cancelled": False, "signalled": True, "still_running": True, "lock_released": False,
                "detail": "processes of this job are still alive after SIGKILL — the job is still running "
                          "and the model stays locked; check the machine before retrying"}
    # กลุ่มหายแล้ว — thread ของงานจะตั้ง exit_code ทันทีที่ท่อปิด · รอช่วงสั้น ๆ ให้คำตอบตรงกับสถานะล็อก
    def settled(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while job.running and time.monotonic() < deadline:
            time.sleep(0.02)
        return not job.running

    detail = ""
    if not settled(1.5):
        # ท่อยังไม่ปิดทั้งที่กลุ่มของงานไม่เหลือใคร = มี process ที่หนีออกนอกกลุ่ม (setsid/daemonize) ถือปลายท่ออยู่
        # เราฆ่ามันไม่ได้ (ไม่รู้ว่าเป็นใคร) แต่ไม่ควรให้มันถือล็อกของโมเดลไว้ด้วย — เลิกอ่านผลของมัน
        job.abandon_output = True
        job.lines.append("มี process นอกกลุ่มของงานยังถือท่อผลลัพธ์อยู่ — เลิกอ่านแล้ว (มันอาจยังรันอยู่บนเครื่องนี้)\n")
        detail = ("the job's own processes are gone, but something that detached from it is still alive on this "
                  "machine — its output is no longer followed")
        settled(1.5)
    return {"cancelled": True, "signalled": True, "still_running": False,
            "lock_released": not job.running, "detail": detail}


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
    if options.get("gpu_util"):
        env["GPU_MEMORY_UTILIZATION"] = str(float(options["gpu_util"]))
    if options.get("served_name"):
        env["SERVED_MODEL_NAME"] = str(options["served_name"])
    # ปิดฟีเจอร์ = ส่งค่าว่าง ซึ่งเป็นค่าที่ "ตั้งใจส่ง" จึงเช็ก is not None ไม่ใช่ truthy
    # controller ใช้ ${VAR-default} (ไม่มี :) ค่าว่างจึงไม่ถูกแทนด้วย default
    if options.get("mtp") is not None:
        env["MTP_FILE"] = str(options["mtp"])
    if options.get("mmproj") is not None:
        env["MMPROJ_FILE"] = str(options["mmproj"])
    # ชื่อ parser ของ vLLM — controller เปิด/ปิดจาก env ตัวนี้ ค่าว่างคือปิด
    # ซึ่งเป็นค่าที่ต้องส่งได้จริง (เอา parser ที่ใส่ผิดออก) จึงเช็ก `is not None`
    # ไม่ใช่ truthy เหมือนตัวอื่น
    if options.get("tool_parser") is not None:
        env["TOOL_CALL_PARSER"] = str(options["tool_parser"])
    if options.get("reasoning_parser") is not None:
        env["REASONING_PARSER"] = str(options["reasoning_parser"])
    if options.get("image"):
        # controller อ่านคนละชื่อตาม engine — ตั้งทั้งคู่ ตัวที่เกินมาไม่มีผล
        env["VLLM_IMAGE"] = env["LLAMACPP_IMAGE"] = str(options["image"])
    return env


# ช่วงที่รับได้ของแต่ละค่า — ตรวจที่เดียวกันทั้งเครื่องนี้และเครื่องอื่น จะได้ไม่มีสองมาตรฐาน
OPTION_RANGES = {
    "port": (1, 65535, int),
    "context": (256, 10_000_000, int),
    "slots": (1, 1024, int),
    "gpu_util": (0.3, 0.98, float),
}


_PARSER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def clean_options(options: dict | None) -> dict:
    """ตรวจค่าที่มาจากหน้าเว็บก่อนเอาไปต่อเป็นคำสั่ง — คืน dict ที่ปลอดภัยแล้ว

    ค่าพวกนี้ถูกส่งข้ามเครื่องผ่าน SSH · ตรวจที่ฝั่ง server เท่านั้นที่นับ
    """
    options = options or {}
    cleaned: dict = {}
    for key, (low, high, cast) in OPTION_RANGES.items():
        raw = options.get(key)
        if raw in (None, ""):
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} ต้องเป็นตัวเลข") from None
        if not low <= value <= high:
            raise ValueError(f"{key} ต้องอยู่ระหว่าง {low} ถึง {high}")
        cleaned[key] = value
    # ชื่อ parser ไปอยู่ในคำสั่งที่รันข้ามเครื่อง จำกัดชุดตัวอักษรไว้ให้แคบที่สุดที่ยัง
    # ครอบชื่อจริงทั้งหมด (qwen3_coder, llama3_json, deepseek_r1, hermes, …)
    # ค่าว่างผ่านได้ เพราะมันแปลว่า "ปิด" ซึ่งเป็นสิ่งที่ต้องสั่งได้
    for key in ("tool_parser", "reasoning_parser"):
        if key not in options:
            continue
        value = str(options[key] or "")
        if value and not _PARSER_NAME.fullmatch(value):
            raise ValueError(f"{key} ต้องเป็นชื่อ parser (a-z, 0-9, _ และ -)")
        cleaned[key] = value

    if options.get("bind") in ("0.0.0.0", "127.0.0.1"):
        cleaned["bind"] = options["bind"]
    elif options.get("bind"):
        raise ValueError("bind รับได้เฉพาะ 0.0.0.0 หรือ 127.0.0.1")
    if options.get("api_key"):
        key = str(options["api_key"])
        if any(ch.isspace() or ord(ch) < 32 for ch in key):
            raise ValueError("API key ต้องไม่มีช่องว่างหรือตัวควบคุม")
        cleaned["api_key"] = key
    if options.get("image"):
        cleaned["image"] = _clean_image(str(options["image"]))
    if options.get("served_name"):
        cleaned["served_name"] = _clean_served_name(str(options["served_name"]))
    # ค่าที่รับคือชื่อไฟล์ หรือค่าว่าง (= ปิด) — ชื่อไฟล์ไปอยู่ในคำสั่งข้ามเครื่อง
    # จึงรับเฉพาะ basename ห้ามมี path หรืออักขระที่แตกคำสั่งได้
    for key in ("mtp", "mmproj"):
        if key not in options:
            continue
        value = str(options[key] or "")
        if value and not _COMPANION_FILE.fullmatch(value):
            raise ValueError(f"{key} ต้องเป็นชื่อไฟล์ .gguf (ไม่มี path)")
        cleaned[key] = value
    return cleaned


# ชื่อไฟล์คู่โมเดล (mmproj / mtp) — basename เท่านั้น
_COMPANION_FILE = re.compile(r"[A-Za-z0-9._-]+\.gguf")


def _clean_served_name(name: str) -> str:
    """ชื่อโมเดลที่ API เสิร์ฟ — ผู้ใช้ตั้งเองได้แทบทุกอย่าง แต่ต้องไม่พังคำสั่ง/URL

    ลูกค้าที่ย้ายมาจากระบบเดิมต้องใช้ชื่อเดิมเป๊ะ (เช่น `vllm-msi-03/aeon-ultimate`
    ที่มี `/` อยู่ข้างใน) — บังคับรูปแบบแคบเกินไปจะใช้กับของจริงไม่ได้
    """
    name = name.strip()
    if not name:
        raise ValueError("ชื่อโมเดลว่างไม่ได้")
    if len(name) > 200:
        raise ValueError("ชื่อโมเดลยาวเกิน 200 ตัว")
    if any(ch.isspace() or ord(ch) < 32 for ch in name):
        raise ValueError("ชื่อโมเดลต้องไม่มีช่องว่างหรือตัวควบคุม")
    return name


def _clean_image(image: str) -> str:
    """image ที่ผู้ใช้พิมพ์เอง — ต้องอยู่ใน registry ที่ยอมรับ และ tag ต้องมีอยู่จริง

    ค่านี้กลายเป็น `docker run <image>` บนเครื่องปลายทาง จะรับอะไรก็ได้ไม่ได้ ·
    ใช้ allowlist ตัวเดียวกับที่ใช้ตอน harden แผน จะได้ไม่มีสองมาตรฐาน
    """
    image = image.strip()
    if any(ch.isspace() or ord(ch) < 32 for ch in image):
        raise ValueError("ชื่อ image ต้องไม่มีช่องว่าง")
    from lmds.brain.allowlists import KNOWN_IMAGE_REPOS, image_repo
    from lmds.brain.registry import tag_exists

    allowed = set().union(*KNOWN_IMAGE_REPOS.values())
    if image_repo(image) not in allowed:
        raise ValueError(f"image '{image}' ไม่อยู่ใน registry ที่ยอมรับ")
    if tag_exists(image) is False:
        raise ValueError(f"tag ของ '{image}' ไม่มีอยู่จริงบน registry")
    return image


def _node_hint() -> str:
    """ชื่อเครื่องปลายทางสักตัวไว้เติมในคำแนะนำ — ไม่มีทะเบียนก็ไม่เป็นไร"""
    try:
        from lmds.nodes import load

        nodes = load()
        return nodes[0].name if nodes else ""
    except Exception:
        return ""


def _tell_watchdog(slug: str, event: str, ok: bool = True) -> None:
    """ส่งต่อให้ `watchdog.operator_event` — ไม่ทำให้งานล้มไม่ว่ากรณีใด"""
    try:
        from lmds.fleet import watchdog

        watchdog.operator_event(slug, event, ok=ok)
    except Exception:  # noqa: BLE001
        pass


def start(slug: str, command: str, controller: str, options: dict | None = None) -> Job:
    if command not in ALLOWED:
        raise JobError(f"คำสั่ง '{command}' ไม่อยู่ในรายการที่อนุญาต")
    path = Path(controller)
    if not path.is_file():
        raise JobError(f"ไม่พบ controller ของ {slug}")

    # เครื่องที่รันโมเดลไม่ได้ ไม่ควรถูกกดปุ่มให้ดูด weight สิบ ๆ GB ลงมา
    # ปฏิเสธที่นี่ = ครอบทุกทางที่หน้าเว็บสั่ง controller (ปุ่มเดี่ยวและ chain)
    from lmds.hardware import serving

    for step in CHAINS.get(command, [command]) + [command]:
        blocked = serving.guard(slug, step, _node_hint())
        if blocked:
            raise JobError(blocked)

    steps = CHAINS.get(command, [command])
    extra_env = {**controller_env(options), **COMMAND_ENV.get(command, {})}

    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(slug, ""))
        if current and current.running:
            raise JobError(f"{slug} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=slug, command=command, steps=steps)
        _JOBS[job.id] = job
        _ACTIVE[slug] = job.id

    def run() -> None:
        # ปุ่ม stop/start/restart ของหน้าเว็บเรียก controller ตรง ๆ ไม่ผ่าน `manager.stop_server` —
        # ต้องบอก watchdog เองว่า **คน** สั่ง ไม่งั้นมันปลุกโมเดลที่เพิ่งกดหยุดขึ้นมาใหม่ (audit 2026-10-06)
        note = {"stop": "stop", "start": "start", "restart": "start"}.get(command)
        if note is None:
            steps_run()
            return
        _tell_watchdog(slug, f"{note}-begin")
        try:
            steps_run()
        finally:
            _tell_watchdog(slug, f"{note}-end", ok=job.exit_code == 0)

    def steps_run() -> None:
        # PYTHONUNBUFFERED: ขั้น download เรียก python ในคอนเทนเนอร์ ซึ่ง stdout ที่ปลาย
        # ท่อ (ไม่ใช่ tty) ถูก block-buffer ไว้ — progress ค้างอยู่ในบัฟเฟอร์จนงานจบ
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **extra_env}
        for index, step in enumerate(steps):
            if job.cancel_requested:
                # ยกเลิกตกในช่องว่างระหว่างสองขั้น — ไม่มี process ให้ฆ่า แต่ขั้นถัดไปต้องไม่เริ่ม
                job.exit_code = 130
                return
            job.step_index = index
            if len(steps) > 1:
                job.lines.append(f"\n── {step} ({index + 1}/{len(steps)}) ──\n")
            try:
                # session ใหม่ = process group ของงานนี้เอง — cancel ฆ่าได้ทั้งกลุ่ม (controller + docker/
                # curl/aria2c/python ที่มันแตกออกมา) ไม่ใช่แค่ bash ตัวหัวแล้วทิ้งตัวโหลดไว้เป็นกำพร้า
                proc = subprocess.Popen(
                    [str(path), step], cwd=str(path.parent), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                job.lines.append(f"เรียก controller ไม่ได้: {exc}\n")
                job.exit_code = 127
                return
            _adopt(job, proc)
            assert proc.stdout is not None
            _pump(job, proc)
            code = proc.wait()
            if code != 0:
                # ขั้นแรกล้ม = ไม่ต้องทำขั้นถัดไป (verify ไฟล์ที่โหลดไม่จบไม่มีประโยชน์)
                job.exit_code = code
                return
        job.exit_code = 0

    threading.Thread(target=run, daemon=True).start()
    return job


# คำสั่งข้ามเครื่องที่ยาวพอจะต้องเห็นความคืบหน้า — สั้น ๆ อย่าง doctor/logs ตอบตรง ๆ เร็วกว่า
# stop อยู่ในนี้ด้วย: docker stop ของ vLLM ที่กำลังโหลด weight รอได้เป็นนาที และ ssh ที่ค้าง
# ระหว่างนั้นยึด thread ของเว็บไว้ทั้งก้อน — เป็น job แล้วอย่างน้อยกดยกเลิกได้
REMOTE_LONG = {"start", "stop", "restart", "repair", "remove"}


def explain_failure(output: str, exit_code: int | None = None, node: str = "", slug: str = "") -> str:
    """แปล error ที่เจอบ่อยให้เป็นสิ่งที่กดทำต่อได้ — ไม่ใช่ให้ไปนั่งอ่าน log ของ rsync เอง"""
    text = output or ""
    if "Permission denied" in text and ("rsync" in text or "failed to open" in text):
        return (
            "แก้ยังไง: ไฟล์ในแคชโมเดลบางส่วนเป็นของ root (มักเกิดจาก container ที่รันเป็น root "
            "แล้วโหลด weight ลงมา) — คัดลอกไป worker ในฐานะ user จึงอ่านไม่ได้\n"
            "กดปุ่ม \"แก้สิทธิ์ไฟล์\" ที่การ์ดของเครื่องนี้ (ถามรหัส sudo ครั้งเดียว) "
            "หรือรันบนเครื่องนั้นเอง: sudo chown -R $USER:$USER ~/.cache/huggingface"
        )
    # 255 คือ ssh เอง ไม่ใช่คำสั่งปลายทาง (สายตายเงียบ → ServerAlive ยอมแพ้ใน ~60 วิ) · คำสั่งที่ไม่มี
    # tty ไม่ได้ SIGHUP จึงมักรันต่อบนเครื่องนั้น — รายงานว่า "ล้ม" เฉย ๆ ทำให้ผู้ใช้สั่งซ้ำแล้วชน
    # "กำลังรันอยู่" หรือ download ซ้อน download (audit 2026-09-04)
    if exit_code == 255 and node:
        where = f"{node}" + (f" ({slug})" if slug else "")
        return (
            f"สาย ssh ไป {where} ขาดกลางงาน (exit 255) — งานบนเครื่องนั้นอาจยังรันอยู่ ไม่ได้ล้ม\n"
            f"ดูก่อนสั่งซ้ำ: lmds node run {node} logs {slug or '<slug>'}  · หรือ refresh การ์ดเครื่องนั้น"
        )
    return ""


def _scrub_secrets(job: "Job", secret_env: dict[str, str] | None) -> None:
    """ลบค่า secret ที่ยืมไปออกจาก log ของงาน ก่อนที่ใครจะได้อ่าน

    คำสั่งปลายทางไม่ควรพิมพ์ token ออกมา แต่ "ไม่ควร" กับ "ไม่เคย" คนละเรื่อง — สคริปต์
    ที่ echo env ทั้งชุดเพื่อ debug, curl ที่ใส่ -v, หรือ error ที่แนบ URL พร้อม token
    ล้วนทำให้ค่าไปโผล่ใน log ที่ถูกส่งกลับมาแสดงบนหน้าเว็บและถูกเก็บไว้

    เคสจริง 2026-08-20: สคริปต์ตรวจสอบของเราเองพิมพ์ token ออกมาเต็มค่าเพราะเขียน
    fallback ผิด — ถ้าไม่กรองตรงนี้ มันจะไปนอนอยู่ใน log ของ job ด้วย
    """
    if not secret_env:
        return
    from lmds.secrets import redact

    values = [v for v in secret_env.values() if v]
    if not values:
        return
    job.lines = deque((redact(line, values) for line in job.lines), maxlen=job.lines.maxlen)


def start_remote(node_name: str, slug: str, command: str, remote_command: str,
                 secret_env: dict[str, str] | None = None, stdin_text: str = "",
                 on_done=None) -> Job:
    """รันคำสั่งบนเครื่องอื่นเป็นงานเบื้องหลัง แล้วสตรีมผลกลับมาทีละบรรทัด

    `download` โมเดล 70 GB ใช้เวลาเป็นสิบนาที — ถ้ารอใน HTTP request เดียวผู้ใช้จะเห็น
    หน้าค้างโดยไม่รู้ว่าคืบหน้าหรือตายไปแล้ว
    """
    from lmds.nodes import NodeError, find, stream

    node = find(node_name)
    if node is None:
        raise JobError(f"ไม่รู้จักเครื่อง {node_name}")

    key = _key(slug, node_name)
    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(key, ""))
        if current and current.running:
            raise JobError(f"{slug} บน {node_name} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=slug, node=node_name, command=command, steps=[command])
        _JOBS[job.id] = job
        _ACTIVE[key] = job.id

    self_node = node_name

    def run() -> None:
        if job.cancel_requested:
            job.exit_code = 130
            return
        try:
            proc = stream(node, remote_command, secret_env, stdin_text=stdin_text)
        except NodeError as exc:
            job.lines.append(f"{exc}\n")
            job.exit_code = 127
            return
        _adopt(job, proc)
        assert proc.stdout is not None
        # กรองตั้งแต่ตอนรับ (ดู _pump) · _scrub_secrets ยังอยู่เป็นด่านสุดท้ายเผื่อค่าที่คร่อมสอง chunk
        _pump(job, proc, list((secret_env or {}).values()))
        _scrub_secrets(job, secret_env)
        code = proc.wait()
        # error ของ git/rsync อ่านแล้วไม่รู้ว่าต้องทำอะไร — แปลให้ตรงจุดก่อนจบงาน
        if code != 0 and self_node:
            from lmds.nodes import explain_install_failure, find

            target = find(self_node)
            output = "".join(job.lines)
            hint = (explain_install_failure(output, target) if target else "") or explain_failure(
                output, exit_code=code, node=self_node, slug=slug)
            if hint:
                job.lines.append("\n" + hint + "\n")
        job.exit_code = code

    def run_and_finish() -> None:
        try:
            run()
        finally:
            # เก็บกวาดต้องเกิดเสมอ แม้งานจะล้ม — clone ฝากกุญแจชั่วคราวไว้ที่ปลายทาง
            # ถ้าไม่ถอนออกก็เท่ากับเปิดประตูทิ้งไว้
            if on_done is not None:
                try:
                    for line in on_done(job) or []:
                        job.lines.append(line)
                except Exception as exc:  # noqa: BLE001 — เก็บกวาดล้มไม่ควรฆ่างานทั้งตัว
                    job.lines.append(f"เก็บกวาดหลังงานจบล้มเหลว: {exc}\n")

    threading.Thread(target=run_and_finish, daemon=True).start()
    return job


def start_shell(slug: str, command: str, script: str, cwd: str = "") -> Job:
    """รันสคริปต์บน hub เองเป็นงานเบื้องหลัง แล้วสตรีมผลกลับมาทีละบรรทัด

    ใช้กับงานที่ไม่ใช่ controller ของโมเดล — ตอนนี้มีอยู่ตัวเดียวคือ "อัปเดตตัว hub เอง"
    (`git pull` + `install.sh` ซึ่งกินเวลาเป็นนาทีและ log ยาว)
    """
    with _LOCK:
        current = _JOBS.get(_ACTIVE.get(slug, ""))
        if current and current.running:
            raise JobError(f"{slug} กำลังรัน '{current.command}' อยู่ — รอให้จบก่อน")
        job = Job(id=uuid.uuid4().hex, slug=slug, command=command, steps=[command])
        _JOBS[job.id] = job
        _ACTIVE[slug] = job.id

    def run() -> None:
        if job.cancel_requested:
            job.exit_code = 130
            return
        try:
            proc = subprocess.Popen(
                ["bash", "-s"], cwd=cwd or None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,       # กลุ่มของงานเอง — cancel ฆ่า git/pip/install.sh ที่มันแตกออกมาได้ด้วย
            )
        except OSError as exc:
            job.lines.append(f"รันไม่ได้: {exc}\n")
            job.exit_code = 127
            return
        _adopt(job, proc)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(script.encode("utf-8"))
        proc.stdin.close()
        _pump(job, proc)
        job.exit_code = proc.wait()

    threading.Thread(target=run, daemon=True).start()
    return job
