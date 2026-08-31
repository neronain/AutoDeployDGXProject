"""ชั้น SSH ของ hub — สร้าง key, ติดตั้ง key ด้วยรหัสผ่านครั้งเดียว, แล้วสั่งงาน node

หลักการด้านความปลอดภัย:
- **รหัสผ่านไม่ถูกเขียนลงดิสก์เลย** ใช้ครั้งเดียวตอน `ssh-copy-id` แล้วหลุดจากหน่วยความจำไป
- ใช้ **keypair เฉพาะของ LMDS** (`~/.config/lmds/id_lmds`) ไม่ยืม key ส่วนตัวของผู้ใช้ —
  เพิกถอนได้โดยไม่กระทบ key อื่น และรู้ได้ว่าใครเข้าเครื่องด้วยสิทธิ์อะไร
- ไม่ต้องใช้ root: user ปกติที่อยู่ในกลุ่ม docker ทำได้ทุกอย่างที่ LMDS ต้องการ
  (root จำเป็นเฉพาะ systemd autostart ซึ่งบอกคำสั่งให้ไปรันเองได้)
"""

from __future__ import annotations

import json
import os
import pty
import select
import shutil
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lmds.config.paths import config_dir, ensure_config_dir

from .registry import Node, NodeError

# ปิด ControlMaster ไว้ก่อน: เร็วขึ้นก็จริงแต่ทิ้ง socket ค้างเวลา process ตายกลางคัน
_SSH_BASE = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
]


def key_path() -> str:
    return str(config_dir() / "id_lmds")


def public_key_path() -> str:
    return key_path() + ".pub"


def ensure_key() -> str:
    """สร้าง keypair ของ LMDS ถ้ายังไม่มี — คืน path ของ public key"""
    ensure_config_dir()
    private = key_path()
    if not os.path.exists(private):
        if shutil.which("ssh-keygen") is None:
            raise NodeError("ไม่พบ ssh-keygen — ติดตั้ง openssh-client ก่อน")
        proc = subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "lmds-hub", "-f", private],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise NodeError(f"สร้าง SSH key ไม่สำเร็จ: {proc.stderr.strip()}")
        os.chmod(private, 0o600)
    return public_key_path()


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def run(node: Node, command: str, timeout: int = 60, stdin_text: str = "") -> Result:
    """รันคำสั่งบน node ด้วย key — ไม่ถามรหัสผ่าน (BatchMode) ถ้า key ใช้ไม่ได้จะล้มทันที

    ห่อด้วย `bash -lc` เพราะ `ssh host cmd` เป็น shell แบบ non-interactive ที่ไม่อ่าน
    `.profile`/`.bashrc` (ของ Ubuntu return ทันทีเมื่อไม่ interactive) — `lmds` ที่ติดตั้งไว้ที่
    `~/.local/bin` จึงหาไม่เจอ ทั้งที่ login เข้าไปเองแล้วใช้ได้ปกติ
    """
    wrapped = f"bash -lc {shlex.quote(command)}"
    # เครื่องเดียวกันอาจเข้าได้หลายทาง (LAN ตอนอยู่ออฟฟิศ, Tailscale ตอนออกนอก)
    # ลองทีละทางจนกว่าจะติด — ล้มเพราะ "ต่อไม่ถึง" เท่านั้นที่ถือว่าควรลองทางถัดไป
    hosts = getattr(node, "all_hosts", [node.host])
    last: Result | None = None
    for index, host in enumerate(hosts):
        result = _run_ssh(f"{node.user}@{host}", node.port, wrapped, timeout, stdin_text)
        if result.ok or not _looks_unreachable(result):
            return result
        last = result
        if index + 1 < len(hosts):
            continue
    return last if last is not None else Result(255, "", "ไม่มีที่อยู่ให้ต่อ")


def _looks_unreachable(result: Result) -> bool:
    """แยก "ต่อไม่ถึง" ออกจาก "ต่อได้แต่คำสั่งล้ม" — อย่างหลังไม่ควรไปลองทางอื่นซ้ำ"""
    if result.exit_code not in (124, 255):
        return False
    text = (result.stderr or "").lower()
    markers = ("no route to host", "connection refused", "timed out", "หมดเวลา",
               "could not resolve", "network is unreachable", "connection timed out")
    return result.exit_code == 124 or any(m in text for m in markers) or not text


def _run_ssh(target: str, port: int, wrapped: str, timeout: int, stdin_text: str = "") -> Result:
    args = ["ssh", *_SSH_BASE, "-i", key_path(), "-p", str(port), target, wrapped]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            # ไม่มี stdin_text = DEVNULL ไม่งั้น ssh ไปกิน stdin ของคนเรียกแล้วค้าง
            **({"input": stdin_text} if stdin_text else {"stdin": subprocess.DEVNULL}),
        )
    except FileNotFoundError as exc:
        raise NodeError("ไม่พบคำสั่ง ssh — ติดตั้ง openssh-client ก่อน") from exc
    except subprocess.TimeoutExpired:
        return Result(124, "", f"หมดเวลา {timeout}s — เครื่องอาจปิดอยู่หรือเน็ตช้า")
    return Result(proc.returncode, proc.stdout, proc.stderr)


def stream(node: Node, command: str, secret_env: dict[str, str] | None = None,
           stdin_text: str = ""):
    """เปิด ssh แบบอ่านผลทีละบรรทัด — ใช้กับงานยาว (download หลายสิบ GB) ที่ต้องเห็นความคืบหน้า

    ต่างจาก run() ที่รอจนจบแล้วค่อยคืนทั้งก้อน · คืน Popen ให้ผู้เรียกวนอ่าน stdout เอง

    คืนท่อแบบ **ไบต์** ไม่ใช่ข้อความ — ผู้เรียกต้องตัดบรรทัดที่ \\r ด้วย (progress bar
    ไม่ขึ้นบรรทัดใหม่) ซึ่ง text=True + bufsize=1 ทำให้ทำไม่ได้: มันตัดที่ \\n อย่างเดียว

    ปลายทางไม่มี tty (ไม่ได้ขอ -t เพราะงานนี้ต้องรันแบบไม่โต้ตอบ) python ฝั่งโน้นจึง
    block-buffer stdout ของตัวเอง — สั่ง unbuffered ไว้ ไม่งั้นผลโผล่มาทีเดียวตอนจบ
    """
    prelude = "export PYTHONUNBUFFERED=1; "
    if secret_env:
        # ความลับเดินทางทาง **stdin** ไม่ใช่ argv และไม่ใช่ไฟล์
        #
        # argv ของ ssh มองเห็นได้จาก `ps` ของทุก user บนเครื่อง hub · เขียนลงไฟล์บน node
        # แปลว่า secret ไปนอนอยู่อีกเครื่องถาวรโดยเจ้าของไม่ได้สั่ง · `read` ในเชลล์ปลายทาง
        # รับค่าแล้วจบ ไม่มีร่องรอยเหลือหลังคำสั่งจบ
        names = " ".join(secret_env)
        prelude += "".join(f"read -r {name}; export {name}; " for name in secret_env)
        prelude = f"# borrowed: {names}\n" + prelude
    wrapped = f"bash -lc {shlex.quote(prelude + command)}"
    host = node.all_hosts[0]
    args = ["ssh", *_SSH_BASE, "-i", key_path(), "-p", str(node.port),
            f"{node.user}@{host}", wrapped]
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if (secret_env or stdin_text) else subprocess.DEVNULL,
        )
        if proc.stdin is not None:
            if secret_env:
                for value in secret_env.values():
                    proc.stdin.write((value + "\n").encode())
            # ข้อความหลายบรรทัด (เช่น private key) ส่งดิบ ๆ ให้คำสั่งปลายทางอ่านเอง —
            # `read -r` ของ secret_env อ่านได้ทีละบรรทัดจึงใช้กับกุญแจไม่ได้
            if stdin_text:
                proc.stdin.write(stdin_text.encode())
            proc.stdin.flush()
            proc.stdin.close()
        return proc
    except FileNotFoundError as exc:
        raise NodeError("ไม่พบคำสั่ง ssh — ติดตั้ง openssh-client ก่อน") from exc


def push_file(node: Node, local: str, remote: str, timeout: int = 1800) -> Result:
    """ส่งไฟล์ไปเครื่องปลายทางด้วย scp — ใช้ key เดียวกับ run()

    ทำไมต้องส่งไฟล์: bundle ที่ผู้ใช้ตรวจแผนแล้วอนุมัติต้องเป็น *ตัวเดียวกัน* กับที่ไปรัน
    ถ้าไปสั่ง `lmds deploy` ซ้ำบนเครื่องปลายทาง มันจะวางแผนใหม่เองซึ่งอาจได้คนละค่า
    — ผู้ใช้อนุมัติแผนหนึ่งแล้วได้อีกแผนหนึ่งไปรัน
    """
    source = Path(local)
    if not source.is_file():
        raise NodeError(f"ไม่พบไฟล์ที่จะส่ง: {local}")
    for host in node.all_hosts:
        args = ["scp", *_SSH_BASE, "-i", key_path(), "-P", str(node.port),
                str(source), f"{node.user}@{host}:{remote}"]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise NodeError("ไม่พบคำสั่ง scp — ติดตั้ง openssh-client ก่อน") from exc
        except subprocess.TimeoutExpired:
            return Result(124, "", f"หมดเวลา {timeout}s ระหว่างส่งไฟล์")
        result = Result(proc.returncode, proc.stdout, proc.stderr)
        if result.ok or not _looks_unreachable(result):
            return result
    return result


def install_key(host: str, user: str, password: str, port: int = 22) -> None:
    """ติดตั้ง public key ของ LMDS ไปยังเครื่องปลายทางโดยใช้รหัสผ่าน **ครั้งเดียว**

    ใช้ pty ขับ ssh-copy-id เพราะ OpenSSH ไม่ยอมรับรหัสผ่านทาง stdin ธรรมดา
    (จึงไม่ต้องพึ่ง sshpass ที่หลายเครื่องไม่ได้ติดตั้งมา)
    """
    ensure_key()
    if shutil.which("ssh-copy-id") is None:
        raise NodeError(
            "ไม่พบ ssh-copy-id — ติดตั้ง openssh-client หรือคัดลอก key เอง:\n"
            f"  ssh-copy-id -i {public_key_path()} -p {port} {user}@{host}"
        )

    args = [
        "ssh-copy-id", "-i", public_key_path(), "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        f"{user}@{host}",
    ]
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.execvp(args[0], args)
        os._exit(127)

    buffer = b""
    sent = False
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 60)
            if not ready:
                break
            try:
                chunk = os.read(fd, 1024)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            lowered = buffer.lower()
            if not sent and b"password" in lowered.rsplit(b"\n", 1)[-1]:
                os.write(fd, password.encode() + b"\n")
                sent = True
            if b"permission denied" in lowered or b"too many authentication" in lowered:
                break
    finally:
        os.close(fd)
        _, status = os.waitpid(pid, 0)

    output = buffer.decode(errors="replace")
    code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status
    if code != 0 or "permission denied" in output.lower():
        hint = "รหัสผ่านไม่ถูกต้อง" if "permission denied" in output.lower() else output.strip()[-300:]
        raise NodeError(f"ติดตั้ง SSH key ไม่สำเร็จ: {hint}")


def _json_object(output: str) -> dict | None:
    """ดึง JSON object ออกจาก stdout ที่อาจมีอย่างอื่นปนมาข้างหน้า

    เคสจริง 2026-08-19: dgx-70 (praisit@10.2.1.70) มี rc ที่พ่น `declare -x …` ของทุก
    ตัวแปรออก stdout ทุกครั้งที่ login shell เริ่ม — JSON ของ `lmds agent info` อยู่ครบ
    และถูกต้องทุกตัวอักษร แต่เริ่มที่ไบต์ที่ 858 · hub รายงานว่า "เวอร์ชันไม่ตรงกัน"
    แล้วผู้ใช้ก็ไปไล่หาเวอร์ชันที่ไม่ได้ผิดอะไรเลย

    เครื่องของลูกค้ามี banner, motd, คำเตือน conda, และ rc แปลก ๆ เป็นเรื่องปกติ —
    การยืนกรานว่า stdout ต้องเป็น JSON ล้วนคือข้อสมมติที่ภาคสนามไม่เคยจริง
    """
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, dict) else None

    # ไล่ทีละ "{" เพราะขยะข้างหน้าอาจมีวงเล็บปีกกาของมันเอง (LS_COLORS, prompt, JSON ของ tool อื่น)
    end = output.rfind("}")
    if end < 0:
        return None
    start = output.find("{")
    while 0 <= start < end:
        try:
            parsed = json.loads(output[start:end + 1])
        except json.JSONDecodeError:
            start = output.find("{", start + 1)
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def probe(node: Node, timeout: int = 30) -> dict:
    """ดึงภาพรวมของ node ผ่าน `lmds agent info` — node ไม่ต้องรัน daemon อะไรเลย"""
    result = run(node, "lmds agent info", timeout=timeout)
    if not result.ok:
        stderr = (result.stderr or result.stdout).strip()
        # แยกสามอย่างที่ต่างกันคนละเรื่อง — บอกผิดแล้วผู้ใช้ไปแก้ผิดที่
        #   1. ต่อ SSH ไม่ได้เลย (เครื่องปิด/เน็ตไม่ถึง)
        #   2. ต่อได้แต่ไม่มี lmds
        #   3. ต่อได้ มี lmds แต่ **เวอร์ชันเก่าเกินไป** — ไม่มีคำสั่ง agent จึงพิมพ์ usage ออกมา
        #      เคสนี้เดิมถูกรายงานว่า "ต่อไม่ได้" ทั้งที่ต่อได้สบาย (เจอจริงกับเครื่องที่มี 0.1.0)
        if "command not found" in stderr or "No such file" in stderr:
            raise NodeError(
                f"{node.target} ยังไม่ได้ติดตั้ง LMDS (หรือ lmds ไม่อยู่ใน PATH ของ SSH session)\n"
                "ติดตั้งบนเครื่องนั้น: git clone …/AutoDeployDGXProject && ./install.sh"
            )
        if "Usage: lmds" in stderr or "No such command" in stderr:
            version = run(node, "lmds version 2>/dev/null | head -1", timeout=20).stdout.strip()
            raise NodeError(
                f"{node.target} มี LMDS แต่เก่าเกินไป ({version or 'ไม่ทราบเวอร์ชัน'}) — "
                f"ยังไม่มีคำสั่ง `agent` ที่ hub ใช้อ่านสถานะ\n"
                f"อัปเดตจากที่นี่ได้เลย: lmds node install {node.name}"
            )
        raise NodeError(f"ต่อ {node.target} ไม่ได้: {stderr[:300] or 'ไม่มีข้อความ'}")
    payload = _json_object(result.stdout)
    if payload is None:
        raise NodeError(
            f"{node.target} ตอบกลับไม่ใช่ JSON — เวอร์ชัน LMDS อาจไม่ตรงกัน\n"
            f"ที่ได้มา: {result.stdout.strip()[:200] or '(ว่าง)'}"
        )
    return payload


# ติดตั้ง/อัปเดต LMDS บน node — hub ไม่ได้ push โค้ดไปเอง แต่สั่งให้เครื่องนั้น clone จาก GitHub
# (ไม่ push เพราะ node อาจอยู่คนละสถาปัตยกรรม และ install.sh ต้องรันบนเครื่องนั้นอยู่ดี)
# repo ที่ node ดึงโค้ดไปติดตั้ง — ตั้งใหม่ได้ด้วย $LMDS_REPO_URL
#
# repo ส่วนตัวดึงผ่าน HTTPS แบบไม่ล็อกอินไม่ได้ (GitHub เลิกรับรหัสผ่านตั้งแต่ 2021) —
# ไซต์ที่ใช้ repo ส่วนตัวจึงต้องชี้ไปที่ SSH remote (`git@github.com:org/repo.git`
# คู่กับ deploy key บนเครื่องนั้น) หรือ mirror ภายในของตัวเอง
# ค่าตายตัวตัวเดียวแปลว่า `lmds node install` ใช้กับ repo ส่วนตัวไม่ได้เลย
REPO_URL = os.environ.get("LMDS_REPO_URL") or "https://github.com/neronain/AutoDeployDGXProject"
_INSTALL_SCRIPT = """
set -e
cd "$HOME"
if [ -d AutoDeployDGXProject/.git ]; then
  cd AutoDeployDGXProject && git pull --ff-only
else
  git clone --depth 1 {repo} AutoDeployDGXProject && cd AutoDeployDGXProject
fi
LMDS_ASSUME_YES=1 {skip}./install.sh
"$HOME/.local/bin/lmds" version
"""


# ขั้นที่ต้องใช้สิทธิ์ root บนเครื่องปลายทาง — ทำครั้งเดียวต่อเครื่อง
# แต่ละขั้นเป็น (คำสั่ง, คำอธิบาย, ตัวตรวจว่าสำเร็จจริง)
def privileged_steps(user: str) -> list[tuple[str, str, str]]:
    return [
        (f"sudo -S -p '' loginctl enable-linger {shlex.quote(user)}",
         "ให้ service ของผู้ใช้ขึ้นตั้งแต่บูต (ไม่ต้องรอ login)",
         f"loginctl show-user {shlex.quote(user)} -p Linger | grep -q Linger=yes"),
    ]


def ownership_steps(user: str) -> list[tuple[str, str, str]]:
    """คืนแคชโมเดลให้เป็นของ user — แก้เคส "container เคยรันเป็น root แล้วโหลด weight ลงมา"

    ผลของเคสนั้นคือไฟล์ในแคชเป็นของ root ปนอยู่ · รันบนเครื่องเดียวยังผ่านเพราะข้างใน
    container เป็น root อยู่แล้ว แต่ `sync-worker` คัดลอกในฐานะ user ผ่าน SSH เจอไฟล์
    โหมด 600 ของ root ก็ตายทันที (rsync exit 23) — เจอจริงกับ DeepSeek-V4-Flash บน spark-head

    แตะเฉพาะแคชโมเดลใน home ของ user เอง ไม่ยุ่งกับอย่างอื่นบนเครื่อง
    """
    quoted = shlex.quote(user)
    # `find ... -print -quit` = หยุดทันทีที่เจอไฟล์แรกที่ไม่ใช่ของ user — แคชใหญ่มาก
    # ไล่ทั้งต้นไม้ทุกครั้งช้าเกินจำเป็น · verify ผ่านเมื่อ "ไม่เหลือของ root แล้ว"
    verify = (
        "! find ~/.cache/huggingface ~/.cache/flashinfer -maxdepth 6 "
        f"! -user {quoted} -print -quit 2>/dev/null | grep -q ."
    )
    return [(
        f"sudo -S -p '' chown -R {quoted}:{quoted} ~/.cache/huggingface ~/.cache/flashinfer",
        "คืนสิทธิ์แคชโมเดล (~/.cache/huggingface, ~/.cache/flashinfer) ให้เป็นของผู้ใช้",
        verify,
    )]


def run_privileged(node: Node, password: str, with_prereq: bool = False,
                   steps: list[tuple[str, str, str]] | None = None) -> list[dict]:
    """ทำขั้นที่ต้องใช้ root บนเครื่องปลายทาง — รหัสผ่านส่งทาง stdin ใช้ครั้งเดียว

    ทำไมต้องมี: ขั้นพวกนี้ทำผ่าน SSH ไม่ได้เพราะ sudo ไม่มี tty ให้กรอกรหัส เดิมจึงได้แค่
    พิมพ์คำสั่งให้ผู้ใช้ไป ssh ทำเอง — ซึ่งขัดกับเหตุผลที่มี hub ตั้งแต่แรก

    รหัสผ่านไม่ถูกเขียนลงดิสก์ ไม่อยู่ใน argv (คนอื่นบนเครื่องอ่าน /proc ได้) และไม่ถูก
    เก็บในทะเบียน — ทะเบียนไม่มีฟิลด์ให้เก็บด้วยซ้ำ
    """
    results = []
    # ผู้เรียกระบุขั้นตอนเองได้ (เช่นแก้สิทธิ์ไฟล์อย่างเดียว) — ไม่ระบุ = ชุดตั้งค่าปกติ
    steps = list(steps) if steps is not None else privileged_steps(node.user)
    if with_prereq:
        steps.append((
            "sudo -S -p '' bash -c 'cd ~/AutoDeployDGXProject && LMDS_ASSUME_YES=1 ./install.sh'",
            "ติดตั้ง Docker / NVIDIA container toolkit",
            "docker info >/dev/null 2>&1",
        ))
    for command, what, verify in steps:
        # ตรวจ *ก่อน* ทำ — ถ้าเรียบร้อยอยู่แล้วก็ไม่ต้องแตะ sudo เลย
        # และที่สำคัญกว่า: รหัสผ่านผิดจะได้ไม่ขึ้น ✓ เพราะบังเอิญสถานะถูกอยู่ก่อนแล้ว
        # (ซึ่งชวนให้เข้าใจว่ารหัสผ่านใช้ได้ ทั้งที่ไม่ได้ใช้)
        if run(node, verify, timeout=60).ok:
            results.append({"step": what, "ok": True, "detail": "", "skipped": True})
            continue
        outcome = run(node, command, timeout=1800, stdin_text=password + "\n")
        # ไม่เชื่อ exit code อย่างเดียว — ตรวจผลจริงอีกที (sudo ที่รหัสผิดคืน 1 เหมือนกัน
        # กับคำสั่งที่ล้มด้วยเหตุอื่น และบางคำสั่งคืน 0 ทั้งที่ไม่ได้ทำอะไร)
        confirmed = run(node, verify, timeout=60).ok
        results.append({
            "step": what,
            "ok": confirmed,
            "skipped": False,
            "detail": "" if confirmed else (outcome.stderr or outcome.stdout).strip()[-300:],
        })
    return results


def install_script(with_prereq: bool = False) -> str:
    """สคริปต์ติดตั้ง — แยกออกมาเพื่อให้ทั้งแบบรอผลและแบบสตรีมใช้ตัวเดียวกัน"""
    return _INSTALL_SCRIPT.format(
        repo=REPO_URL, skip="" if with_prereq else "LMDS_SKIP_PREREQ=1 ",
    )


def explain_install_failure(output: str, node: Node) -> str:
    """แปล error ของ git ให้เป็นสิ่งที่ทำต่อได้

    "could not read Username for 'https://github.com'" อ่านแล้วไม่รู้เลยว่าต้องทำอะไร —
    ความหมายจริงคือ repo เป็น private และเครื่องนั้นไม่มีสิทธิ์เข้าถึง
    """
    text = output or ""
    if "could not read Username" in text or "Authentication failed" in text:
        return (
            f"{node.name} เข้าถึง repo ไม่ได้ — repo เป็น private และเครื่องนั้นยังไม่มีสิทธิ์\n"
            "แก้ได้สองทาง:\n"
            f"  1. ใส่ deploy key บน {node.name} แล้วชี้ remote ไป SSH:\n"
            f"     ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_lmds_github   (บนเครื่องนั้น)\n"
            "     เอา .pub ไปใส่ที่ GitHub → repo → Settings → Deploy keys\n"
            "  2. ตั้ง $LMDS_REPO_URL ให้ชี้ SSH remote หรือ mirror ภายใน แล้วสั่งใหม่"
        )
    return ""


def install_lmds(node: Node, timeout: int = 1800, with_prereq: bool = False) -> Result:
    """ติดตั้งหรืออัปเดต LMDS บน node ผ่าน SSH

    ค่าเริ่มต้นข้ามขั้นตอน prerequisite (docker/toolkit) เพราะขั้นนั้นต้องใช้ sudo ซึ่งไม่มี tty
    ให้กรอกรหัสผ่าน — เครื่องที่ยังไม่มี Docker ต้องไปรัน install.sh เองบนเครื่องนั้น
    """
    return run(node, install_script(with_prereq), timeout=timeout)


def check_login(host: str, user: str, port: int = 22) -> bool:
    """key ใช้ได้แล้วหรือยัง — ใช้ตรวจก่อนถามรหัสผ่าน จะได้ไม่ถามซ้ำโดยไม่จำเป็น"""
    node = Node(name="_probe", host=host, user=user, port=port)
    return run(node, "true", timeout=15).ok
