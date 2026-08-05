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
import subprocess
from dataclasses import dataclass

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
    """รันคำสั่งบน node ด้วย key — ไม่ถามรหัสผ่าน (BatchMode) ถ้า key ใช้ไม่ได้จะล้มทันที"""
    args = ["ssh", *_SSH_BASE, "-i", key_path(), "-p", str(node.port), node.target, command]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise NodeError("ไม่พบคำสั่ง ssh — ติดตั้ง openssh-client ก่อน") from exc
    except subprocess.TimeoutExpired:
        return Result(124, "", f"หมดเวลา {timeout}s — เครื่องอาจปิดอยู่หรือเน็ตช้า")
    return Result(proc.returncode, proc.stdout, proc.stderr)


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


def check_login(host: str, user: str, port: int = 22) -> bool:
    """key ใช้ได้แล้วหรือยัง — ใช้ตรวจก่อนถามรหัสผ่าน จะได้ไม่ถามซ้ำโดยไม่จำเป็น"""
    node = Node(name="_probe", host=host, user=user, port=port)
    return run(node, "true", timeout=15).ok
