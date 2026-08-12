"""งานที่ใช้เวลานาน (download หลายสิบ GB, start ที่โหลดโมเดลเป็นนาที)

HTTP request เดียวรอไม่ไหว และผู้ใช้ต้องเห็นว่ามันคืบหน้าอยู่ ไม่ใช่ค้าง —
จึงรัน controller เป็น subprocess แล้วให้หน้าเว็บ poll เอาบรรทัดล่าสุดไปแสดง

หนึ่ง slug รันได้ทีละงานเท่านั้น: download ซ้อน start คือทางลัดไปสู่ไฟล์พัง
"""

from __future__ import annotations

import os
import re
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
}

# download อย่างเดียวไม่พอที่จะบอกว่า "ไฟล์มาครบ" — CLI ให้รัน verify-files ต่อเสมอ
# หน้าเว็บจึงต่อให้เลย ไม่งั้นผู้ใช้ไม่มีทางรู้ว่าโหลดครบจริงไหม
CHAINS = {
    "download": ["download", "verify-files"],
    "repair": ["download", "verify-files"],  # repair = โหลดที่ขาด (resume) แล้วตรวจซ้ำ
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


def _pump(job: "Job", proc: subprocess.Popen) -> None:
    """ย้ายผลจาก process เข้า job ทีละบรรทัด — นับ \\r เป็นตัวจบบรรทัดด้วย

    บรรทัดที่จบด้วย \\r คือ "เฟรม" ของ progress bar ตัวถัดไปตั้งใจจะทับของเดิม
    ไม่ใช่ต่อท้าย · ถ้า append ทุกเฟรม deque 400 บรรทัดจะเต็มไปด้วยเลข % ของ
    วินาทีที่แล้ว แล้วดันบรรทัดที่บอกสาเหตุจริงหายไปหมด
    """
    # read1() ไม่ใช่ read(): คืนเท่าที่มีอยู่ทันที ส่วน read(n) จะรอจนครบ n
    # ซึ่งแปลว่าไม่สตรีม · ไม่ใช้ os.read(fileno) เพราะผูกกับ pipe จริงโดยไม่จำเป็น
    # แล้วทำให้เทสที่จำลอง subprocess ต้องมี fd จริงตามไปด้วย
    stream = proc.stdout
    read = getattr(stream, "read1", None) or stream.read
    pending = ""
    overwrite = False   # เฟรมก่อนหน้าจบด้วย \r → บรรทัดถัดไปทับตัวเดิม

    def emit(text: str) -> None:
        nonlocal overwrite
        if overwrite and job.lines:
            job.lines[-1] = text
        else:
            job.lines.append(text)

    while True:
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
    if options.get("served_name"):
        env["SERVED_MODEL_NAME"] = str(options["served_name"])
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
    return cleaned


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
        # PYTHONUNBUFFERED: ขั้น download เรียก python ในคอนเทนเนอร์ ซึ่ง stdout ที่ปลาย
        # ท่อ (ไม่ใช่ tty) ถูก block-buffer ไว้ — progress ค้างอยู่ในบัฟเฟอร์จนงานจบ
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **extra_env}
        for index, step in enumerate(steps):
            job.step_index = index
            if len(steps) > 1:
                job.lines.append(f"\n── {step} ({index + 1}/{len(steps)}) ──\n")
            try:
                proc = subprocess.Popen(
                    [str(path), step], cwd=str(path.parent), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                job.lines.append(f"เรียก controller ไม่ได้: {exc}\n")
                job.exit_code = 127
                return
            job.process = proc
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
REMOTE_LONG = {"start", "restart", "repair", "remove"}


def explain_failure(output: str) -> str:
    """แปล error ที่เจอบ่อยให้เป็นสิ่งที่กดทำต่อได้ — ไม่ใช่ให้ไปนั่งอ่าน log ของ rsync เอง"""
    text = output or ""
    if "Permission denied" in text and ("rsync" in text or "failed to open" in text):
        return (
            "แก้ยังไง: ไฟล์ในแคชโมเดลบางส่วนเป็นของ root (มักเกิดจาก container ที่รันเป็น root "
            "แล้วโหลด weight ลงมา) — คัดลอกไป worker ในฐานะ user จึงอ่านไม่ได้\n"
            "กดปุ่ม \"แก้สิทธิ์ไฟล์\" ที่การ์ดของเครื่องนี้ (ถามรหัส sudo ครั้งเดียว) "
            "หรือรันบนเครื่องนั้นเอง: sudo chown -R $USER:$USER ~/.cache/huggingface"
        )
    return ""


def start_remote(node_name: str, slug: str, command: str, remote_command: str) -> Job:
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
        try:
            proc = stream(node, remote_command)
        except NodeError as exc:
            job.lines.append(f"{exc}\n")
            job.exit_code = 127
            return
        job.process = proc
        assert proc.stdout is not None
        _pump(job, proc)
        code = proc.wait()
        # error ของ git/rsync อ่านแล้วไม่รู้ว่าต้องทำอะไร — แปลให้ตรงจุดก่อนจบงาน
        if code != 0 and self_node:
            from lmds.nodes import explain_install_failure, find

            target = find(self_node)
            output = "".join(job.lines)
            hint = (explain_install_failure(output, target) if target else "") or explain_failure(output)
            if hint:
                job.lines.append("\n" + hint + "\n")
        job.exit_code = code

    threading.Thread(target=run, daemon=True).start()
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
        try:
            proc = subprocess.Popen(
                ["bash", "-s"], cwd=cwd or None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            job.lines.append(f"รันไม่ได้: {exc}\n")
            job.exit_code = 127
            return
        job.process = proc
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(script.encode("utf-8"))
        proc.stdin.close()
        _pump(job, proc)
        job.exit_code = proc.wait()

    threading.Thread(target=run, daemon=True).start()
    return job
