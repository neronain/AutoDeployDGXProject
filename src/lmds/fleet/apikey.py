"""API key ของ model server — เก็บไว้นอกโฟลเดอร์ bundle

`bundle.env` ถือ knob ทุกตัวที่ controller อ่าน **ยกเว้น key** เพราะโฟลเดอร์ bundle ถูก
zip แจกต่อได้ (เหตุผลเต็มอยู่ใน `fleet/bundle_settings.py`) ซึ่งถูกแล้ว — แต่ผลข้างเคียง
คือ key ไม่เคยถูกเก็บไว้ที่ไหนเลย:

  * หน้าเว็บส่ง `API_KEY=` ไปกับการกด start ครั้งนั้นครั้งเดียว
  * systemd unit เรียก `<controller> start` เปล่า ๆ (ไม่มี `Environment=API_KEY=`)

พอเครื่อง reboot โมเดลจึงกลับมาเปิดโล่งบน `0.0.0.0` ทั้งที่ผู้ใช้ตั้ง key ไว้แล้ว
และไม่มีอะไรบอกว่าเกิดขึ้น — เป็นความพังแบบเดียวกับที่ `bundle.env` แก้ให้ port/context
ไปแล้ว ต่างกันแค่ตรงที่อันนี้พังเงียบ ๆ ในทางที่เปิดช่องให้คนอื่น ไม่ใช่แค่ทำให้ใช้งานไม่ได้

ที่นี่คือที่เก็บ: `~/.lmds/keys/<slug>` โหมด 0600 ใต้โฟลเดอร์ 0700 — อยู่นอก bundle
จึงไม่ติดไปกับ zip · controller อ่านไฟล์นี้เมื่อไม่มี `API_KEY` ส่งมาจากภายนอก
ลำดับความสำคัญจึงเป็น: flag/env > ไฟล์นี้ > ไม่มี key

ไม่มีไฟล์ = พฤติกรรมเหมือนเดิมทุกประการ · bundle ที่ติดตั้งไปแล้วจึงไม่ถูกแตะ
"""

from __future__ import annotations

import os
import re
import secrets
import stat
from pathlib import Path

# slug เดินทางมาจาก URL ของหน้าเว็บและ argv — กันชื่อที่พาออกนอกโฟลเดอร์ไว้ตรงนี้จุดเดียว
# (รูปแบบเดียวกับ assistant/catalog._SLUG)
_SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class ApiKeyError(ValueError):
    """slug หรือ key ที่ใช้ไม่ได้ — บอกไปตรง ๆ ดีกว่าเขียนไฟล์ผิดที่"""


def key_root() -> Path:
    """โฟลเดอร์ที่เก็บ key ของเครื่องนี้ — `$LMDS_KEY_ROOT` ทับได้ (เทสใช้)"""
    return Path(os.environ.get("LMDS_KEY_ROOT", Path.home() / ".lmds" / "keys"))


def path_for(slug: str) -> Path:
    if not _SLUG.match(slug or ""):
        raise ApiKeyError(f"slug ใช้ไม่ได้: {slug!r}")
    return key_root() / slug


def mint() -> str:
    """key ใหม่ — hex ล้วนเพราะต้องผ่าน header, .env, YAML และช่องกรอกของ client ทุกตัว

    token_urlsafe มี `-`/`_` ซึ่งปลอดภัยเท่ากันแต่เคยทำให้คนก๊อปไม่ครบเวลาดับเบิลคลิก
    """
    return secrets.token_hex(24)


def read(slug: str) -> str:
    """key ที่เก็บไว้ของ bundle นี้ — คืนค่าว่างเมื่อไม่มี/อ่านไม่ได้

    อ่านไม่ได้ไม่ใช่ข้อยกเว้น: ไฟล์อาจเป็นของ user อื่นบนเครื่องที่แชร์กัน ซึ่งแปลว่า
    "ไม่มี key สำหรับเรา" ไม่ใช่ "ระบบพัง"
    """
    try:
        return path_for(slug).read_text(encoding="utf-8").strip()
    except (OSError, ApiKeyError):
        return ""


def write(slug: str, key: str) -> Path:
    """เก็บ key ของ bundle นี้ — คืน path ที่เขียน

    เขียนผ่านไฟล์ชั่วคราวแล้ว replace เพื่อไม่ให้มีช่วงที่ไฟล์มีอยู่แต่ยังว่าง
    (controller ที่ start พอดีช่วงนั้นจะอ่านได้ key ว่าง = รันแบบไม่มี auth)
    """
    text = (key or "").strip()
    if not text:
        raise ApiKeyError("key ว่างไม่ได้ — ใช้ clear() ถ้าตั้งใจจะเอาออก")
    if any(ch in text for ch in "\n\r\0"):
        raise ApiKeyError("key มีอักขระขึ้นบรรทัดใหม่ไม่ได้")
    target = path_for(slug)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, stat.S_IRWXU)
    temporary = target.with_name(f".{target.name}.new")
    # สร้างด้วย 0600 ตั้งแต่แรก ไม่ใช่เขียนก่อนแล้ว chmod ทีหลัง — ระหว่างสองขั้นนั้น
    # ไฟล์อ่านได้ทั้งเครื่อง
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (text + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(temporary, target)
    return target


def clear(slug: str) -> bool:
    """เอา key ที่เก็บไว้ออก — True เมื่อมีไฟล์ให้ลบจริง"""
    try:
        path_for(slug).unlink()
        return True
    except (OSError, ApiKeyError):
        return False


def listing() -> dict[str, bool]:
    """{slug: มี key ไหม} ของทุกไฟล์ในที่เก็บ — ไม่คืนตัว key ออกมา"""
    root = key_root()
    if not root.is_dir():
        return {}
    found: dict[str, bool] = {}
    for item in sorted(root.iterdir()):
        if item.is_file() and _SLUG.match(item.name):
            found[item.name] = bool(read(item.name))
    return found
