"""ด่านสุดท้ายก่อนคำตอบออกจาก `lmds mcp` — ไม่มี API key · HF token · token ของหน้าเว็บ · ลายเซ็นไลเซนส์ หลุดไปหาผู้ช่วย AI

คำตอบของเครื่องมือถูกส่งต่อให้ LLM provider ข้างนอกทันที และค้างอยู่ใน transcript ของผู้ช่วย — ของที่หลุดตรงนี้เอาคืนไม่ได้
ที่มาของความลับในคำตอบมีจริงทุกทาง: log ของ vLLM พิมพ์ argument ทั้งชุดตอน start (`api_key=…`) · `bundle.args` ของ
bundle ที่ adopt มามี `--api-key …` · error ของ Hub แนบ URL พร้อม token · `lmds fit` คืน `extra_args` ที่ตั้งอยู่

ใช้ตัวกรองเดียวกับที่หน้าเว็บใช้กับ log ของงาน (web/jobs._pump → `secrets.redact(text, ค่าที่รู้)`) แล้วเสริมสองอย่างที่
ทางนั้นไม่ต้องมี เพราะมันรู้ค่าที่ยืมไปล่วงหน้า แต่ที่นี่คำตอบมาจากเครื่องอื่นที่ hub ไม่รู้ key ของมัน:

  1. ค่าที่ตามหลังชื่อที่บอกว่าเป็นความลับ (`--api-key X` · `API_KEY=X` · `"hf_token": "X"` · `api_key=['X']`) — ปิดจากชื่อ
     ไม่ต้องรู้ค่า · ชุดชื่อเดียวกับที่ `lmds adopt` ใช้ถอดความลับออกจาก env (fleet/adopt._SECRET_ENV)
  2. ค่าของคีย์ที่ชื่อบอกว่าเป็นความลับใน JSON ของคำตอบเอง (เผื่อ payload วันหน้ามีฟิลด์แบบนั้น)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from lmds.secrets import MASK, redact

# ชื่อที่บอกว่าค่าเป็นความลับ — TOKEN ไม่นับ TOKENS/TOKENIZER (`max_tokens` · `tokenizer_mode` เป็นค่าตั้งธรรมดา)
_SECRET_NAME = r"(?:[A-Za-z0-9]+[_-])*(?:TOKEN(?!S|IZER)|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIALS?)(?:[_-][A-Za-z0-9]+)*"
_SECRET_KEY = re.compile(_SECRET_NAME, re.IGNORECASE)
_VALUE = r"(?P<value>[A-Za-z0-9_\-.+/=~:@%]{8,})"
# `ชื่อ=ค่า` · `ชื่อ: ค่า` · `"ชื่อ": "ค่า"` · `ชื่อ=['ค่า']` — ปิดเฉพาะค่า ชื่อยังอยู่ให้อ่านรู้เรื่อง
_ASSIGNMENT = re.compile(
    r"(?P<name>(?<![A-Za-z0-9])-{0,2}" + _SECRET_NAME + r")(?P<sep>[\"']?\s*[=:]\s*[\[(]?\s*[\"']?)" + _VALUE,
    re.IGNORECASE)
# `--api-key ค่า` — คั่นด้วยช่องว่างรับเฉพาะรูป flag (ขึ้นต้น --) ไม่งั้นประโยคธรรมดาที่มีคำว่า token จะโดนไปด้วย
_FLAG = re.compile(r"(?P<name>(?<![A-Za-z0-9-])--" + _SECRET_NAME + r")(?P<sep>\s+[\"']?)" + _VALUE, re.IGNORECASE)
# คำที่ตามหลังชื่อได้โดยไม่ใช่ค่าลับ — ข้อความของ doctor ("token: required") ต้องไม่กลายเป็น [REDACTED]
_NOT_A_VALUE = re.compile(r"(?i)(none|null|true|false|\[REDACTED\]|<[^>]*>|\$\{?[A-Za-z_]+\}?)$")


def _looks_secret(value: str) -> bool:
    """ค่าหลังชื่อลับหน้าตาเป็น key จริงไหม — มีตัวเลขปน หรือยาวตั้งแต่ 20 ตัว · คำธรรมดา ("required", "missing") ไม่ใช่"""
    if _NOT_A_VALUE.match(value):
        return False
    return len(value) >= 20 or any(ch.isdigit() for ch in value)


# รหัสผ่านที่ฝังใน URL (`http://user:pass@proxy:3128`)
_URL_PASSWORD = re.compile(r"(://[^/\s:@]+:)[^/\s@]+@")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


_KEYRING_TTL = 300.0
_keyring_cache: tuple[float, list[str]] | None = None


def _keyring_values() -> list[str]:
    """ค่าที่อยู่ใน OS keyring ของ hub — ถามไม่เกินครั้งละ 5 นาที

    keyring บนเครื่องที่มี desktop คุยผ่าน D-Bus ซึ่งค้างได้ (เคส spark-head 2026-09-07 — secrets.store จำกัดไว้ 5 วิ
    ต่อชื่อ) · ถามใหม่ทุกคำขอของเครื่องมือ = ทุกคำขอช้าได้ถึงครึ่งนาทีบนเครื่องแบบนั้น
    """
    global _keyring_cache
    now = time.monotonic()
    if _keyring_cache is not None and now - _keyring_cache[0] < _KEYRING_TTL:
        return _keyring_cache[1]
    values: list[str] = []
    try:
        from lmds.secrets import SECRET_ENV_VARS
        from lmds.secrets.store import _keyring, _kr_get

        ring = _keyring()
        if ring is not None:
            values = [value for value in (_kr_get(ring, name) for name in SECRET_ENV_VARS) if value]
    except Exception:  # noqa: BLE001 — ไม่มี keyring/ถามไม่ได้ = ไม่มีค่าจากทางนี้
        values = []
    _keyring_cache = (now, values)
    return values


def known_secrets() -> list[str]:
    """ค่าลับที่ hub เครื่องนี้ถืออยู่ — ปิดแบบรู้ค่า ไม่ว่ามันจะโผล่ในบริบทไหน

    ไฟล์อ่านใหม่ทุกครั้ง (key ถูกหมุนได้ระหว่างที่ server เปิดอยู่): key ของ bundle ทุกใบ (`lmds key`) · token ของหน้าเว็บ ·
    ไฟล์ credentials (provider + HF) · token ของ huggingface-cli · ตัวแปรแวดล้อมของ secret ทุกตัว · ลายเซ็นไลเซนส์
    """
    values: list[str] = []
    try:
        from lmds.fleet import apikey

        root = apikey.key_root()
        for item in sorted(root.iterdir()) if root.is_dir() else []:
            if item.is_file() and not item.name.startswith("."):
                values.append(_read(item))
    except OSError:
        pass
    try:
        from lmds.config.paths import config_dir
        # ตัวอ่านของ secrets.store เอง — ไฟล์เป็น `ชื่อ=ค่า` ที่คนแก้มือได้ กติกาข้ามบรรทัดเสียอยู่ที่นั่นที่เดียว
        from lmds.secrets.store import _hf_cli_token, _read_credentials_file

        values.append(_read(config_dir() / "web-token"))
        values += list(_read_credentials_file().values())
        values.append(_hf_cli_token() or "")
    except Exception:  # noqa: BLE001 — อ่านไม่ได้ = ไม่มีค่าที่รู้ ตัวกรองแบบรูปร่างยังทำงาน
        pass
    try:
        from lmds.secrets import SECRET_ENV_VARS

        names = {name for group in SECRET_ENV_VARS.values() for name in group} | {"LMDS_WEB_TOKEN", "API_KEY"}
        values += [os.environ.get(name, "") for name in sorted(names)]
    except Exception:  # noqa: BLE001
        pass
    values += _keyring_values()
    try:
        from lmds.licensing import store

        values.append(getattr(store.load(), "signature_value", "") or "")
    except Exception:  # noqa: BLE001 — ไฟล์ไลเซนส์เสีย/ไม่มี = ไม่มีลายเซ็นให้ปิด
        pass
    return sorted({v for v in values if v and len(v) >= 8}, key=len, reverse=True)


def scrub_text(text: str, known: list[str]) -> str:
    if not text:
        return text
    text = redact(text, known)

    def hide(match: re.Match) -> str:
        if not _looks_secret(match.group("value")):
            return match.group(0)
        return f"{match.group('name')}{match.group('sep')}{MASK}"

    text = _FLAG.sub(hide, _ASSIGNMENT.sub(hide, text))
    return _URL_PASSWORD.sub(lambda m: f"{m.group(1)}{MASK}@", text)


def scrub(value, known: list[str] | None = None):
    """คำตอบของเครื่องมือทั้งก้อน (dict/list ซ้อนกันได้) → ก้อนเดิมที่ไม่มีความลับ · โครงและคีย์ไม่เปลี่ยน"""
    known = known_secrets() if known is None else known
    if isinstance(value, str):
        return scrub_text(value, known)
    if isinstance(value, list):
        return [scrub(item, known) for item in value]
    if isinstance(value, tuple):
        return [scrub(item, known) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, str) and _SECRET_KEY.fullmatch(str(key)) and _looks_secret(item):
                out[key] = MASK
            else:
                out[key] = scrub(item, known)
        return out
    return value
