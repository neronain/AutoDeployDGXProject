"""จัดการหน้าเว็บที่รันเบื้องหลัง — รู้ว่ามีตัวไหนรันอยู่ และรันอยู่ด้วย token อะไร

ปัญหาที่แก้ (เจอจริงบน controller): `lmds web -b` ซ้ำ ๆ แล้วหน้าเว็บ "ใช้ได้บ้างไม่ได้บ้าง"

สาเหตุคือรอบที่สองขึ้นไป uvicorn bind ไม่ได้ (`address already in use`) แล้วตายทันที
แต่ CLI เขียน PID ของศพนั้นทับลง `web.pid` และพิมพ์ token ใหม่ออกมาให้ผู้ใช้ ผลคือ:

  - ตัวที่ยังเสิร์ฟจริงเป็นตัวเก่า ที่ถือ token *คนละตัว* กับที่เพิ่งพิมพ์
    → เปิดลิงก์ที่พิมพ์มาแล้วเจอ "A token is required"
  - `lmds web --stop` ฆ่า PID ที่ตายไปแล้ว → รายงานว่าหยุดสำเร็จ ทั้งที่ของจริงยังรันอยู่

หลักที่ยึด:
  - **บอกสถานะจริง** — รันซ้ำต้องบอกว่า "มีตัวรันอยู่แล้ว นี่คือลิงก์ของมัน" ไม่ใช่พิมพ์
    ลิงก์ที่ใช้ไม่ได้ · ตายตอนสตาร์ตต้องบอกว่าตาย พร้อมเหตุผลจาก log
  - **จำ token ของตัวที่รันอยู่ไว้** ผู้ใช้จะได้เปิดลิงก์เดิมซ้ำได้โดยไม่ต้อง restart
    (ไฟล์เป็น 0600 · log ไฟล์เดิมก็มี token อยู่แล้ว การเก็บแบบจำกัดสิทธิ์จึงรัดกุมกว่า)
  - **ไม่ฆ่า PID ที่ไม่ใช่ของเรา** — PID ถูกใช้ซ้ำได้ ตรวจ cmdline ก่อนเสมอ
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

# ตัวชี้ว่า process หนึ่งเป็นหน้าเว็บของ LMDS จริง ไม่ใช่ PID ที่ถูกใช้ซ้ำ
_CMDLINE_MARK = "lmds.cli.main"

# ตั้ง token จาก environment ได้ — สำหรับเครื่องที่รันด้วย systemd/compose ซึ่งไม่มีใคร
# นั่งตอบคำถามตอนสตาร์ต และไม่ควรต้องพึ่งไฟล์ในโฮมของ user คนใดคนหนึ่ง
TOKEN_ENV = "LMDS_WEB_TOKEN"

# สั้นกว่านี้เดาได้ในเวลาที่มีความหมาย — หน้านี้สั่ง start/stop โมเดลได้ทุกเครื่องในทะเบียน
MIN_TOKEN_LEN = 8


class TokenError(ValueError):
    """token ที่ผู้ใช้ตั้งเองใช้ไม่ได้ — ข้อความอธิบายว่าทำไม"""


def validate_token(token: str) -> str:
    """token ที่ผู้ใช้ตั้งเอง: อะไรก็ได้ แต่ต้องยาวพอ และไม่มีช่องว่าง/ตัวควบคุม

    ไม่บังคับรูปแบบตัวอักษร (ผู้ใช้อยากใช้ passphrase ภาษาไทยก็ได้) แต่ช่องว่างกับ
    ตัวควบคุมทำให้ copy ไป paste แล้วเพี้ยนโดยไม่มีใครรู้ตัว จึงกันไว้
    """
    token = token.strip()
    if len(token) < MIN_TOKEN_LEN:
        raise TokenError(f"token ต้องยาวอย่างน้อย {MIN_TOKEN_LEN} ตัว (ได้มา {len(token)})")
    if any(ch.isspace() or ord(ch) < 32 for ch in token):
        raise TokenError("token ต้องไม่มีช่องว่างหรือตัวควบคุม — copy/paste แล้วเพี้ยนโดยไม่รู้ตัว")
    return token


def new_token() -> str:
    import secrets

    return secrets.token_urlsafe(24)


def token_file() -> Path:
    """token ที่ใช้ซ้ำได้ข้ามการ start/stop — อยู่ใน config ไม่ใช่ run/ เพราะต้องอยู่ยาว"""
    from lmds.config import config_dir

    return config_dir() / "web-token"


def remembered_token() -> str:
    """token เดิมของเครื่องนี้ — ว่างถ้ายังไม่เคยมี

    ทำไมต้องจำ: token ที่สุ่มใหม่ทุกครั้งแปลว่าลิงก์ที่ bookmark ไว้ตายทุกครั้งที่ restart
    ผู้ใช้ต้องกลับไปหา terminal ทุกรอบ ซึ่งขัดกับเหตุผลที่มีหน้าเว็บตั้งแต่แรก
    """
    try:
        return token_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def remember_token(token: str) -> None:
    path = token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def forget_token() -> None:
    token_file().unlink(missing_ok=True)


def state_file() -> Path:
    from lmds.fleet import run_root

    return run_root() / "web.json"


def log_file() -> Path:
    from lmds.fleet import run_root

    return run_root() / "web.log"


def _cmdline(pid: int) -> str:
    """cmdline ของ process — ว่างถ้าอ่านไม่ได้ (เช่น macOS ที่ไม่มี /proc)"""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def alive(pid: int) -> bool:
    """process นี้ยังอยู่ และเป็นหน้าเว็บของ LMDS จริงหรือเปล่า

    บนเครื่องที่อ่าน `/proc` ไม่ได้ (macOS) ตรวจได้แค่ว่า PID ยังอยู่ — ยอมรับได้
    เพราะเคสที่เจอจริงเป็น Linux และการเดาว่า "ตายแล้ว" อันตรายกว่าเดาว่า "ยังอยู่"
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    cmdline = _cmdline(pid)
    return _CMDLINE_MARK in cmdline if cmdline else True


def read_state() -> dict | None:
    try:
        data = json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_state(pid: int, port: int, bind: str, token: str) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": pid, "port": port, "bind": bind, "token": token, "started_at": time.time()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        path.chmod(0o600)  # มี token อยู่ข้างใน — ผู้ใช้อื่นบนเครื่องเดียวกันไม่ควรอ่านได้
    except OSError:
        pass


def clear_state() -> None:
    state_file().unlink(missing_ok=True)
    # ไฟล์เก่าจากเวอร์ชันก่อนหน้า — ทิ้งไปด้วยจะได้ไม่มีสองแหล่งความจริง
    (state_file().parent / "web.pid").unlink(missing_ok=True)


def running() -> dict | None:
    """สถานะของหน้าเว็บที่รันอยู่จริง — None ถ้าไม่มี (ล้างไฟล์ค้างให้ด้วย)"""
    state = read_state()
    if state is None:
        return None
    if not alive(int(state.get("pid") or 0)):
        clear_state()
        return None
    return state


def port_busy(host: str, port: int, timeout: float = 0.4) -> bool:
    """มีใครยึดพอร์ตนี้อยู่ไหม — ใช้ตอบกรณีที่ผู้ใช้เปิดหน้าเว็บด้วยวิธีอื่น"""
    target = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_until_serving(host: str, port: int, pid: int, timeout: float = 12.0) -> bool:
    """รอจนกว่าหน้าเว็บจะรับ connection จริง — เลิกรอทันทีถ้า process ตายไปก่อน

    ต้องรอจริง ไม่ใช่พิมพ์ว่าสำเร็จแล้วเดินจากไป เพราะเคสที่พังคือ process ตายหลัง
    สตาร์ต 0.2 วินาที ซึ่ง `Popen` มองว่าสำเร็จ
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_busy(host, port):
            return True
        if not alive(pid):
            return False
        time.sleep(0.2)
    return port_busy(host, port)


def wait_until_free(host: str, port: int, timeout: float = 8.0) -> bool:
    """รอให้พอร์ตว่างจริงหลังสั่งหยุด — SIGTERM คืน socket ไม่ทันที

    ไม่รอแล้วสตาร์ตต่อทันที จะเจอ "พอร์ตไม่ว่าง" ทั้งที่เราเป็นคนสั่งหยุดเอง
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_busy(host, port):
            return True
        time.sleep(0.2)
    return not port_busy(host, port)


def stop(sig: int = 15) -> dict | None:
    """หยุดตัวที่รันอยู่ — คืนสถานะที่หยุดไป หรือ None ถ้าไม่มีอะไรให้หยุด"""
    state = running()
    if state is None:
        clear_state()
        return None
    try:
        os.kill(int(state["pid"]), sig)
    except OSError:
        clear_state()
        return None
    clear_state()
    return state


def url(state: dict, host: str = "") -> str:
    """ลิงก์ล้วน ไม่มี token — token อยู่ใน URL แปลว่ามันไปอยู่ใน history/log/referrer"""
    return f"http://{host or '127.0.0.1'}:{state.get('port', 8600)}/"


def log_tail(lines: int = 12) -> str:
    """ท้าย log — ใช้บอกสาเหตุตอนสตาร์ตไม่ขึ้น แทนที่จะให้ผู้ใช้ไปเปิดไฟล์เอง"""
    try:
        content = log_file().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.rstrip().splitlines()[-lines:])


# ── ให้หน้าเว็บขึ้นเองหลัง reboot / ตายแล้วฟื้นเอง ──────────────────────────────
# รันเบื้องหลังด้วย `-b` อยู่ได้จนกว่าเครื่องจะรีบูตหรือ process ตาย — สำหรับเครื่องที่
# ทำหน้าที่เป็น controller ประจำ นั่นไม่พอ ต้องเป็นบริการที่ระบบดูแลให้
#
# ใช้ **systemd user service** ไม่ใช่ system service เพราะ:
#   - ไม่ต้อง sudo (system service ต้องใช้ ซึ่ง SSH ไม่มี tty ให้กรอกรหัส)
#   - หน้าเว็บอ่าน ~/.config/lmds และ ~/.lmds ของผู้ใช้คนนั้นอยู่แล้ว
#   - `loginctl enable-linger` ทำให้มันขึ้นตั้งแต่บูตโดยไม่ต้องมีใคร login
UNIT_NAME = "lmds-web.service"

_UNIT = """[Unit]
Description=LMDS web console
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={python} -m lmds.cli.main web --port {port} --bind {bind}
Environment=LMDS_WEB_TOKEN={token}
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
"""


def unit_path():
    from pathlib import Path as _Path

    return _Path.home() / ".config" / "systemd" / "user" / UNIT_NAME


def render_unit(port: int, bind: str, token: str) -> str:
    import sys

    return _UNIT.format(python=sys.executable, port=port, bind=bind, token=token)
