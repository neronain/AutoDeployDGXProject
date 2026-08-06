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


def run(node: Node, command: str, timeout: int = 60) -> Result:
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
        result = _run_ssh(f"{node.user}@{host}", node.port, wrapped, timeout)
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


def _run_ssh(target: str, port: int, wrapped: str, timeout: int) -> Result:
    args = ["ssh", *_SSH_BASE, "-i", key_path(), "-p", str(port), target, wrapped]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,  # ไม่งั้น ssh ไปกิน stdin ของคนเรียกแล้วค้าง
        )
    except FileNotFoundError as exc:
        raise NodeError("ไม่พบคำสั่ง ssh — ติดตั้ง openssh-client ก่อน") from exc
    except subprocess.TimeoutExpired:
        return Result(124, "", f"หมดเวลา {timeout}s — เครื่องอาจปิดอยู่หรือเน็ตช้า")
    return Result(proc.returncode, proc.stdout, proc.stderr)


def stream(node: Node, command: str):
    """เปิด ssh แบบอ่านผลทีละบรรทัด — ใช้กับงานยาว (download หลายสิบ GB) ที่ต้องเห็นความคืบหน้า

    ต่างจาก run() ที่รอจนจบแล้วค่อยคืนทั้งก้อน · คืน Popen ให้ผู้เรียกวนอ่าน stdout เอง
    """
    wrapped = f"bash -lc {shlex.quote(command)}"
    host = node.all_hosts[0]
    args = ["ssh", *_SSH_BASE, "-i", key_path(), "-p", str(node.port),
            f"{node.user}@{host}", wrapped]
    try:
        return subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, stdin=subprocess.DEVNULL,
        )
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


def probe(node: Node, timeout: int = 30) -> dict:
    """ดึงภาพรวมของ node ผ่าน `lmds agent info` — node ไม่ต้องรัน daemon อะไรเลย"""
    result = run(node, "lmds agent info", timeout=timeout)
    if not result.ok:
        stderr = (result.stderr or result.stdout).strip()
        if "command not found" in stderr or "not found" in stderr:
            raise NodeError(
                f"{node.target} ยังไม่ได้ติดตั้ง LMDS (หรือ lmds ไม่อยู่ใน PATH ของ SSH session)\n"
                "ติดตั้งบนเครื่องนั้น: git clone …/AutoDeployDGXProject && ./install.sh"
            )
        raise NodeError(f"ต่อ {node.target} ไม่ได้: {stderr[:300] or 'ไม่มีข้อความ'}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise NodeError(f"{node.target} ตอบกลับไม่ใช่ JSON — เวอร์ชัน LMDS อาจไม่ตรงกัน") from exc


# ติดตั้ง/อัปเดต LMDS บน node — hub ไม่ได้ push โค้ดไปเอง แต่สั่งให้เครื่องนั้น clone จาก GitHub
# (ไม่ push เพราะ node อาจอยู่คนละสถาปัตยกรรม และ install.sh ต้องรันบนเครื่องนั้นอยู่ดี)
REPO_URL = "https://github.com/neronain/AutoDeployDGXProject"
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


def install_script(with_prereq: bool = False) -> str:
    """สคริปต์ติดตั้ง — แยกออกมาเพื่อให้ทั้งแบบรอผลและแบบสตรีมใช้ตัวเดียวกัน"""
    return _INSTALL_SCRIPT.format(
        repo=REPO_URL, skip="" if with_prereq else "LMDS_SKIP_PREREQ=1 ",
    )


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
