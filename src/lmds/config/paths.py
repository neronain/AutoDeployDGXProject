"""ตำแหน่งไฟล์ config ของ LMDS — override ได้ด้วย env LMDS_CONFIG_DIR (ใช้ในเทสด้วย)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def config_dir() -> Path:
    root = os.environ.get("LMDS_CONFIG_DIR")
    if root:
        return Path(root)
    return Path.home() / ".config" / "lmds"


def config_file() -> Path:
    return config_dir() / "config.yaml"


def credentials_file() -> Path:
    return config_dir() / "credentials"


def profile_file() -> Path:
    return config_dir() / "profile.yaml"


def sessions_dir() -> Path:
    return config_dir() / "sessions"


def write_atomic(path: Path, text: str, mode: int = 0o600) -> Path:
    """เขียนไฟล์ config แบบ "เห็นได้ทีเดียวทั้งไฟล์" — เขียนลงไฟล์ชั่วคราวแล้ว replace

    หน้าเว็บรัน endpoint แบบ sync ใน threadpool สองคำขอที่บันทึกไฟล์เดียวกันพร้อมกัน
    (เช่นลากจัดลำดับเครื่องสองครั้งติด ๆ) จะเขียนทับกันกลางคัน ได้ไฟล์ที่เป็นเนื้อของ
    ครั้งใหม่ต่อด้วยหางของครั้งเก่า — YAML พังทั้งไฟล์แล้วหน้าเว็บ 500 ทั้งหน้า (เจอจริง)

    os.replace เป็น atomic บนไฟล์ระบบเดียวกัน คนอ่านจึงเห็นของเก่าหรือของใหม่เท่านั้น
    ไฟล์ชั่วคราวอยู่โฟลเดอร์เดียวกันเพื่อไม่ให้ข้ามไฟล์ระบบ (ข้ามแล้ว replace ไม่ atomic)
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.chmod(mode)
    os.replace(temporary, path)
    return path


def ensure_config_dir() -> Path:
    d = config_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d
