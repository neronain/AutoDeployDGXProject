"""ผนึก process ของ `lmds mcp` ให้อ่านอย่างเดียว — เขียนไฟล์ไม่ได้ สั่งคำสั่งที่เปลี่ยนสถานะไม่ได้ ทั้งบนเครื่องนี้และเครื่องอื่น

ทำไมไม่พอแค่ "ผูกเครื่องมือกับฟังก์ชันอ่าน": ฟังก์ชันอ่านของ LMDS มีทางเขียนซ่อนอยู่จริง (สำรวจ 2026-10-09 ตอนสร้าง MCP):

  - `fleet.discover()` ลบทะเบียนของ bundle ที่ตายแล้ว (`_drop_dead_registration`)
  - `inventory.request_usage()` เขียน `run/<slug>/usage.samples` ทุกครั้งที่ถาม
  - `brain.build_plan(..., None)` (rule-based) เขียน session log ลง config dir
  - `doctor` รัน container ชั่วคราวของ image ไปถามสถาปัตยกรรม แล้วจดผลลง `run/<slug>/runtime-arch.json`
  - `git status` ของ "hub มีไฟล์แก้ค้างไหม" เขียน index ของ repo
  - `lmds node list --check` / `lmds fleet check --check` เขียนทะเบียนเครื่อง

และวันหน้าจะมีเพิ่ม โดยคนที่แก้ฟังก์ชันอ่านไม่รู้ว่ามีผู้ถามที่สัญญาว่าอ่านอย่างเดียวเรียกมันอยู่ · รายการ "ห้ามเรียกฟังก์ชันนี้"
ตามไม่ทัน — จึงกั้นที่ขอบของ process แทน ด้วย audit hook ของ Python (PEP 578): ทุกการเปิดไฟล์เพื่อเขียน ทุกการลบ/ย้าย
ทุกการ spawn process และทุกสัญญาณที่ส่งออก ผ่าน hook นี้ก่อนถึงระบบปฏิบัติการ ไม่ว่าโค้ดชั้นไหนเป็นคนสั่ง

สิ่งที่เกิดเมื่อโดนกั้น: `ReadOnlyViolation` ซึ่งเป็น `PermissionError` (errno EROFS) — เหมือนดิสก์ถูก mount แบบ read-only ·
ทางเขียนที่ซ่อนในฟังก์ชันอ่านทุกจุดข้างบนห่อ `except OSError: pass` ไว้อยู่แล้ว (เขียนไม่ได้ไม่ควรทำให้การอ่านล้ม) จึงข้าม
การเขียนไปเองและคำตอบยังครบ · โค้ดที่ไม่ได้เผื่อไว้จะล้มดัง ๆ ซึ่งคือสิ่งที่ต้องการ · ทุกครั้งที่กั้นถูกจดไว้ (`drain()`)
ให้เครื่องมือแนบไปกับคำตอบ — ผู้ช่วยเห็นว่ามีอะไรถูกปฏิเสธ ไม่ใช่ได้คำตอบที่แหว่งโดยไม่รู้ตัว

ฝั่งเครื่องอื่น: hub ส่งคำสั่งอ่านไปเป็น `LMDS_READ_ONLY=1 lmds …` — `lmds` รุ่นที่รู้จักตัวแปรนี้ผนึก process ของตัวเอง
ด้วยฟังก์ชันเดียวกันตั้งแต่ก่อนเข้าคำสั่ง (cli/main._entry) ทางเขียนที่ซ่อนอยู่ข้างบนจึงถูกกั้นบนเครื่องนั้นด้วย ·
`lmds` รุ่นเก่ากว่าไม่รู้จักตัวแปรนี้และทำงานตามเดิม (คำสั่งยังเป็นคำสั่งอ่านจากรายการ REMOTE_READS — สิ่งที่ยังเกิดได้บน
เครื่องรุ่นเก่าคือ bookkeeping ของ `agent info`/`logs` ชุดเดียวกับที่หน้าเว็บของ hub ทำให้เกิดทุก 15 วิอยู่แล้ว)

audit hook ถอดออกไม่ได้ตลอดอายุ process — เรียก `seal()` เฉพาะใน process ที่ตั้งใจให้อ่านอย่างเดียวทั้งชีวิต
(`lmds mcp` · `lmds` ที่ถูกเรียกด้วย LMDS_READ_ONLY=1) · เทสเรียกผ่าน subprocess เท่านั้น
"""

from __future__ import annotations

import errno
import os
import re
import shlex
import sys
import threading

READ_ONLY_ENV = "LMDS_READ_ONLY"
# หน้าคำสั่งที่ hub ส่งไปเครื่องอื่น — เครื่องที่ lmds รู้จักจะผนึกตัวเอง · รุ่นเก่ามองเป็นตัวแปรแวดล้อมที่ไม่มีใครอ่าน
REMOTE_PREFIX = f"{READ_ONLY_ENV}=1"
_SLUG = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
_NUMBER = r"[0-9]{1,9}"
_RO = REMOTE_PREFIX + " "

# ── คำสั่งที่ hub ส่งไปรันบนเครื่องอื่นได้ — ครบทั้งรายการ ไม่มี wildcard ──────────────────────────────
# ทุกบรรทัดคือ `lmds …` แบบอ่าน ที่ฟังก์ชันใน reads.py ประกอบ · จะเพิ่มเครื่องมือที่ถามเครื่องอื่น ต้องเพิ่มบรรทัดที่นี่
# (ไม่เพิ่ม = ssh ถูกกั้นก่อนออกจากเครื่อง) · ไม่มี start/stop/restart/set/remove/push/install และจะไม่มี
REMOTE_READS = tuple(re.compile(pattern) for pattern in (
    rf"{_RO}lmds agent info",                              # nodes.probe — สถานะเครื่อง + โมเดล
    r"lmds version 2>/dev/null \| head -1",                # nodes.probe ถามเมื่อ lmds ปลายทางเก่าเกินจะมี `agent`
    rf"{_RO}lmds fit {_SLUG} --json(?: --slots {_NUMBER})?(?: --context {_NUMBER})?",   # dry run เสมอ — ไม่ใช่ `set --fit`
    rf"{_RO}lmds logs {_SLUG} -n {_NUMBER}",
    rf"{_RO}lmds doctor {_SLUG} --json --no-probe",
    rf"{_RO}lmds watchdog status(?: {_SLUG})? --json",
))

# ── โปรแกรมที่ process นี้ spawn ได้ — ที่ไม่อยู่ในสองตารางนี้ถูกกั้น (ปฏิเสธเป็นค่าเริ่มต้น) ─────────────
# โปรแกรมที่มีทั้งท่าอ่านและท่าเขียน: ระบุท่าอ่านที่รับ (คำแรกที่ไม่ใช่ option · สองคำสำหรับ `docker image inspect`)
_READ_VERBS = {
    "docker": {"ps", "inspect", "logs", "info", "images", "version", "container inspect", "image inspect"},
    "systemctl": {"is-active", "is-enabled", "show", "status", "cat", "list-units", "list-unit-files"},
    "loginctl": {"show-user", "show-session", "list-users", "list-sessions"},
    "git": {"rev-parse", "log", "status", "describe", "diff", "show", "symbolic-ref", "ls-files", "rev-list",
            "merge-base", "for-each-ref", "cat-file"},
}
# โปรแกรมที่ถามสภาพเครื่องอย่างเดียว — รับทุก argument ยกเว้นคำที่ทำให้มันกลายเป็นคำสั่งตั้งค่า
_PROBES = {
    "nvidia-smi": {"-r", "--gpu-reset", "-pm", "--persistence-mode", "-e", "-c", "--compute-mode", "-pl", "-lgc", "-rgc"},
    "ip": {"add", "del", "delete", "set", "flush", "replace", "change", "append"},
    "ifconfig": {"up", "down", "add", "del", "netmask", "mtu", "inet"},
    "ss": set(), "netstat": set(), "lsof": set(), "pgrep": set(), "ps": set(), "id": set(), "free": set(),
    "df": set(), "uname": set(), "nproc": set(), "lscpu": set(), "vm_stat": set(), "sw_vers": set(),
    "sysctl": {"-w", "--write", "-p", "--load", "--system"},
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_WRITABLE = {os.devnull, "/dev/tty"}

_sealed = False
_lock = threading.Lock()
_refused: list[str] = []
_logged: set[str] = set()
# เหตุการณ์ที่การถูกกั้นทำให้ *คำตอบ* ต่างไป (ข้อมูลหาย) — ที่เหลือคือการเขียนไฟล์/ลบ/ย้าย
_TELL_CALLER = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system", "os.fork",
                          "os.forkpty", "os.kill", "os.killpg"})


class ReadOnlyViolation(PermissionError):
    """process นี้ถูกผนึกให้อ่านอย่างเดียว — สิ่งที่ถูกกั้นอยู่ในข้อความ"""


def is_sealed() -> bool:
    return _sealed


def drain() -> list[str]:
    """คำสั่งที่ถูกกั้นตั้งแต่ครั้งก่อนที่เรียก — เครื่องมือแนบไปกับคำตอบ (ไม่ซ้ำ ไม่เกิน 20 บรรทัด)

    เฉพาะการ spawn และการส่งสัญญาณ: ตัวถามสภาพเครื่องที่ถูกกั้นทำให้คำตอบขาดข้อมูล ผู้ถามต้องรู้ · การเขียนไฟล์ที่ถูกกั้น
    ไม่อยู่ในนี้ — ทางเขียนที่ซ่อนในฟังก์ชันอ่านถูกกั้นทุกครั้งที่ถามเป็นเรื่องปกติและคำตอบไม่เปลี่ยน (ถ้าโค้ดไม่ได้เผื่อไว้
    มันล้มเป็น error ของเครื่องมือเองอยู่แล้ว) · ทั้งสองแบบลง stderr
    """
    with _lock:
        seen = list(dict.fromkeys(_refused))
        _refused.clear()
    return seen[:20]


def _refuse(what: str, *, tell_caller: bool):
    with _lock:
        if tell_caller and len(_refused) < 200:
            _refused.append(what)
        first_time = what not in _logged
        if first_time and len(_logged) < 2000:
            _logged.add(what)
    if first_time:                             # บรรทัดเดิมซ้ำทุกคำขอ (usage.samples) ไม่ช่วยใคร — ครั้งแรกพอ
        print(f"lmds-mcp: read-only — refused: {what}", file=sys.stderr, flush=True)
    raise ReadOnlyViolation(errno.EROFS, f"this lmds process is read-only (lmds mcp / {READ_ONLY_ENV}=1): {what}")


def remote_problem(command: str) -> str:
    """เหตุที่คำสั่งนี้ส่งไปรันบนเครื่องอื่นไม่ได้ — สตริงว่าง = อยู่ในรายการอ่าน"""
    if any(pattern.fullmatch(command) for pattern in REMOTE_READS):
        return ""
    return f"remote command is not on the read list: {command[:120]!r}"


def _first_words(args: list[str], skip_values: tuple[str, ...] = ()) -> list[str]:
    """คำที่ไม่ใช่ option ตามลำดับ — ข้าม `-x` และค่าของ option ใน skip_values (`git -C <dir>`)"""
    words, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in skip_values:
            skip = True
            continue
        if arg.startswith("-"):
            continue
        words.append(arg)
    return words


def spawn_problem(argv: list[str]) -> str:
    """เหตุที่ spawn คำสั่งนี้ไม่ได้ — สตริงว่าง = ได้ · กติกาเดียวของทั้ง process (ดูตารางข้างบน)"""
    if not argv:
        return "empty command"
    program = os.path.basename(str(argv[0]))
    rest = [str(a) for a in argv[1:]]
    shown = " ".join([program, *rest])[:160]
    if program == "ssh":
        if "-G" in rest:                       # `ssh -G` พิมพ์ config ที่จะใช้ ไม่ต่อไปไหน (nodes.ssh._dialed_names)
            return ""
        try:
            wrapped = shlex.split(rest[-1]) if rest else []
        except ValueError:
            wrapped = []
        # รูปเดียวที่ nodes.ssh.run ประกอบ: `bash -lc '<คำสั่ง>'` — อย่างอื่น (คำสั่งดิบ · -t · port forward) ไม่รับ
        if len(wrapped) != 3 or wrapped[:2] != ["bash", "-lc"]:
            return f"ssh without a recognised remote command: {shown}"
        return remote_problem(wrapped[2])
    if program in _READ_VERBS:
        words = _first_words(rest, skip_values=("-C", "-c", "--format", "-f", "-p", "--since", "--tail", "-n"))
        allowed = _READ_VERBS[program]
        verb = words[0] if words else ""
        if verb in allowed or " ".join(words[:2]) in allowed:
            return ""
        return f"{program} {verb or '(no verb)'} is not a read command: {shown}"
    if program in _PROBES:
        hit = sorted(set(rest) & _PROBES[program])
        return f"{program} with {hit[0]} changes settings: {shown}" if hit else ""
    if program.endswith(".sh"):
        # controller ของ bundle ไม่ถูกรันจาก process ที่ผนึกเลย แม้แต่ `logs`: มันเป็นสคริปต์ bash ที่อยู่นอกเอื้อมของ
        # audit hook และคำสั่งอ่านของมันก็เขียนได้ — เทสไม่เปลี่ยนสถานะจับได้ตอนสร้าง (2026-10-09): `<controller> logs 5`
        # ของ llama.cpp แบบ native ลบ `server.pid` ที่มันเห็นว่าค้าง (_our_server_pid) · log อ่านจากแหล่งตรงแทน
        # (fleet.logs_text(direct=True): docker logs / server.log)
        return f"bundle controllers are never run from a read-only process: {shown}"
    if program == "llama-server" and rest == ["--version"]:
        return ""                              # inventory ถาม build ของ llama.cpp — พิมพ์แล้วจบ ไม่โหลดโมเดล
    return f"{program} is not on the list of read-only programs: {shown}"


def _open_problem(path, flags) -> str:
    if isinstance(path, int) or not isinstance(flags, int) or not flags & _WRITE_FLAGS:
        return ""
    name = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path)
    return "" if name in _WRITABLE else f"open for writing: {name}"


def _hook(event: str, args) -> None:
    handler = _HANDLERS.get(event)
    if handler is not None:
        problem = handler(args)
        if problem:
            _refuse(problem, tell_caller=event in _TELL_CALLER)


def _kill_problem(args) -> str:
    pid, sig = args[0], args[1]
    if sig == 0:                               # `kill -0` = ถามว่า process ยังอยู่ไหม (fleet._pid_alive)
        return ""
    # subprocess ฆ่าลูกของตัวเองเมื่อหมดเวลา (timeout ของ ssh/docker) — ลูกที่ process นี้ spawn เองเท่านั้นที่ผ่านทางนั้น
    if sys._getframe(2).f_globals.get("__name__") == "subprocess":
        return ""
    return f"signal {sig} to pid {pid}"


def _mkdir_problem(args) -> str:
    # mkdir(exist_ok=True) ของโฟลเดอร์ที่มีอยู่แล้วไม่ได้เปลี่ยนอะไร — ปล่อยให้ OS ตอบ FileExistsError ตามปกติ
    try:
        return "" if os.path.isdir(args[0]) else f"mkdir {os.fsdecode(args[0])}"
    except (TypeError, ValueError):
        return "mkdir"


def _named(action: str):
    def problem(args) -> str:
        target = args[0] if args else ""
        try:
            return f"{action} {os.fsdecode(target)}"
        except TypeError:
            return action
    return problem


_HANDLERS = {
    "open": lambda args: _open_problem(args[0], args[2]),
    "subprocess.Popen": lambda args: spawn_problem(list(args[1]) if not isinstance(args[1], (str, bytes)) else [args[1]]),
    "os.posix_spawn": lambda args: spawn_problem([os.fsdecode(a) for a in args[1]]),
    "os.exec": lambda args: spawn_problem([os.fsdecode(a) for a in args[1]]),
    "os.spawn": lambda args: spawn_problem([os.fsdecode(a) for a in args[2]]),
    "os.system": lambda args: f"os.system: {os.fsdecode(args[0])[:120]}",
    "os.fork": lambda args: "fork",
    "os.forkpty": lambda args: "forkpty",
    "os.kill": _kill_problem,
    "os.killpg": lambda args: "" if args[1] == 0 else f"signal {args[1]} to process group {args[0]}",
    "os.mkdir": _mkdir_problem,
    "os.remove": _named("remove"),
    "os.rmdir": _named("rmdir"),
    "os.rename": _named("rename"),
    "os.chmod": _named("chmod"),
    "os.chown": _named("chown"),
    "os.symlink": lambda args: f"symlink {os.fsdecode(args[1])}",
    "os.link": lambda args: f"link {os.fsdecode(args[1])}",
    "os.truncate": _named("truncate"),
    "os.utime": _named("utime"),
    "os.setxattr": _named("setxattr"),
    "os.removexattr": _named("removexattr"),
    "shutil.rmtree": _named("rmtree"),
    "shutil.move": _named("move"),
    "shutil.copyfile": lambda args: f"copy to {os.fsdecode(args[1])}",
    "shutil.copytree": lambda args: f"copy to {os.fsdecode(args[1])}",
    "shutil.make_archive": _named("archive"),
    "shutil.unpack_archive": _named("unpack"),
}


def seal() -> None:
    """ผนึก process นี้ — เรียกครั้งเดียวตอน `lmds mcp` เริ่ม ก่อนอ่านคำขอแรก · ถอดไม่ได้"""
    global _sealed
    if _sealed:
        return
    # `git status` (hub มีไฟล์แก้ค้างไหม) เขียน index ของ repo เพื่อ refresh stat — บอก git ว่าไม่ต้อง
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    # `lmds` ที่ process นี้ (หรือลูกของมัน) เรียกต่อ ต้องผนึกตัวเองเหมือนกัน
    os.environ[READ_ONLY_ENV] = "1"
    # import หลังผนึกจะพยายามเขียน __pycache__ — ไม่ต้องลอง (ถึงลองก็ถูกกั้นและ importlib ข้ามเอง)
    sys.dont_write_bytecode = True
    sys.addaudithook(_hook)
    _sealed = True
