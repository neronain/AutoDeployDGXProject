"""Secret store ของ LMDS

ลำดับการ resolve (สูง → ต่ำ):
  1. environment variable
  2. OS keyring (ถ้าติดตั้ง package `keyring` และใช้งานได้)
  3. credentials file (~/.config/lmds/credentials, สิทธิ์ 0600, รูปแบบ KEY=VALUE)

กติกา FR-7: secret ห้ามลง config.yaml, bundle, log หรือ output ใด ๆ — ใช้ redact() คุมทุกทางออก
"""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Optional

from lmds.config.paths import credentials_file, ensure_config_dir

KEYRING_SERVICE = "lmds"

# ชื่อ secret ที่ระบบรู้จัก → env var ที่ยอมรับ (เรียงตามลำดับความสำคัญ)
SECRET_ENV_VARS: dict[str, list[str]] = {
    "openai": ["LMDS_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "gemini": ["LMDS_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["LMDS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"],
    "minimax": ["LMDS_MINIMAX_API_KEY", "MINIMAX_API_KEY"],
    "openai-compat": ["LMDS_OPENAI_COMPAT_API_KEY"],
    "hf": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
}


def _keyring():
    # LMDS_NO_KEYRING=1 = ไม่แตะ OS keyring เลย — install.sh ใช้ตอน Update ทั้งฟลีต เพราะ keyring บนเครื่องที่มี
    # desktop (GNOME) คุยผ่าน D-Bus Secret Service ซึ่งค้างได้ไม่มีกำหนดเมื่อ session ไม่ได้ unlock
    if os.environ.get("LMDS_NO_KEYRING"):
        return None
    try:
        import keyring  # type: ignore

        return keyring
    except Exception:
        return None


def _keyring_timeout() -> float:
    try:
        return float(os.environ.get("LMDS_KEYRING_TIMEOUT") or 5)
    except ValueError:
        return 5.0


def _kr_get(kr, name: str) -> Optional[str]:
    """อ่านจาก keyring แบบมีเวลาจำกัด — เคสจริง 2026-09-07 spark-head: `lmds config show` ที่ install.sh เรียก
    ค้าง 10 นาทีใน ep_poll (D-Bus) จน `lmds node install --all` ทั้งฟลีตหยุดรอ · เกินเวลา = ถือว่าไม่มีค่า
    (thread ที่ค้างเป็น daemon จึงไม่ขวางตอนโปรแกรมจบ)"""
    import threading

    box: dict[str, Optional[str]] = {}

    def run() -> None:
        try:
            box["value"] = kr.get_password(KEYRING_SERVICE, name)
        except Exception:
            box["value"] = None

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(_keyring_timeout())
    return None if worker.is_alive() else box.get("value")


# ── รูปของสิ่งที่ลงไฟล์ credentials ได้ ──
#
# ไฟล์เป็น `KEY=VALUE` บรรทัดละค่า · audit 2026-10: `POST /api/secrets/hf` ด้วย token
# "hf_…\nopenai=sk-attacker" เขียนบรรทัดที่สองลงไปด้วย — ตั้ง secret ตัวเดียวแต่ทับ/เพิ่ม key ของ provider
# อีกตัวได้ (เครื่อง server ไม่มี keyring จึงลงไฟล์เสมอ) · ตรวจที่ set_secret ซึ่งเป็นทางเดียวที่ทุกผู้เรียก
# (หน้าเว็บ · CLI) ผ่าน และตัวเขียนไฟล์ปฏิเสธซ้ำอีกชั้น — fleet/apikey.py ทำแบบเดียวกันกับ key ของโมเดล
_NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def validate_secret(name: str, value: str) -> str:
    """ค่าที่เก็บเป็น secret ได้ — ไม่ผ่าน = ValueError พร้อมเหตุผล · คืนค่าเดิมเมื่อผ่าน

    ปฏิเสธตัวควบคุมทั้งหมวด (CR · LF · NUL · TAB · ESC · DEL · C1): ไม่มี API key/token จริงตัวไหนมี และ
    LF คือสิ่งที่แทรกบรรทัดใหม่ลงไฟล์ได้ · ปฏิเสธช่องว่างหัวท้ายเพราะตอนอ่านกลับจะถูกตัด — ค่าที่เก็บกับค่าที่
    ได้คืนต้องเป็นตัวเดียวกัน
    """
    if not isinstance(name, str) or not _NAME_OK.fullmatch(name):
        raise ValueError(f"ชื่อ secret ไม่ถูกต้อง: {name!r}")
    if not isinstance(value, str) or not value:
        raise ValueError("ค่า secret ว่างเปล่า")
    if any(unicodedata.category(ch) == "Cc" for ch in value):
        raise ValueError("ค่า secret มีตัวควบคุม (ขึ้นบรรทัดใหม่ · tab · NUL ฯลฯ) — วางเฉพาะตัว key/token บรรทัดเดียว")
    if value != value.strip():
        raise ValueError("ค่า secret ขึ้นต้นหรือลงท้ายด้วยช่องว่าง — ตัดออกก่อน")
    return value


def _read_credentials_file() -> dict[str, str]:
    """อ่านไฟล์ credentials — บรรทัดที่ผิดรูปถูกข้าม ไม่ทำให้ทั้งไฟล์อ่านไม่ได้

    ไฟล์นี้คนแก้มือได้ และรุ่นก่อนเคยเขียนบรรทัดแปลก ๆ ลงไปได้ · ไบต์ที่ไม่ใช่ UTF-8, บรรทัดไม่มี `=`,
    ชื่อ key ผิดรูป, อ่านไฟล์ไม่ได้ = ข้าม/ถือว่าไม่มี — `lmds` ทั้งตัวต้องไม่ตายเพราะบรรทัดเดียว
    """
    path = credentials_file()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not _NAME_OK.fullmatch(key) or not value or "�" in value \
                or any(unicodedata.category(ch) == "Cc" for ch in value):
            continue
        out[key] = value
    return out


def _write_credentials_file(values: dict[str, str]) -> None:
    """เขียนทั้งไฟล์ใหม่แบบ atomic และเป็น 0600 **ตั้งแต่ open**

    เดิม write_text() แล้ว chmod(0o600) ทีหลัง — ระหว่างสองขั้นนั้นไฟล์เกิดมาด้วย umask ปกติ (มักอ่านได้ทั้ง
    เครื่อง) ใครอ่านจังหวะนั้นพอดีได้ key ไป · ทางเดียวกับ web-token และ license store
    """
    for key, value in values.items():
        validate_secret(key, value)             # ชั้นสุดท้าย: บรรทัดที่ผิดรูปต้องไม่ถูกเขียนไม่ว่าใครเรียก
    ensure_config_dir()
    path = credentials_file()
    body = "# LMDS credentials — อย่า commit ไฟล์นี้\n"
    body += "".join(f"{k}={v}\n" for k, v in sorted(values.items()))
    temporary = path.with_name(f".{path.name}.new")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)                    # ไฟล์ชั่วคราวค้างจากรอบก่อนอาจมีสิทธิ์อื่น — O_CREAT ไม่แก้ให้
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(temporary, path)


def check_credentials_permissions() -> Optional[str]:
    """คืนข้อความเตือนถ้าสิทธิ์ไฟล์ credentials หลวมเกินไป (group/other อ่านได้)"""
    path = credentials_file()
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"คำเตือน: {path} มีสิทธิ์ {oct(mode)} — ควรเป็น 0600 (chmod 600)"
    return None



def _hf_cli_token() -> Optional[str]:
    """token ที่ `huggingface-cli login` เขียนไว้ — เครื่องที่เคยโหลดโมเดล gated มีอยู่แล้ว

    ไม่อ่านตรงนี้ = บังคับให้ผู้ใช้กรอก token ซ้ำทั้งที่เครื่องมีอยู่แล้ว ซึ่งไม่มีเหตุผล
    ลำดับตาม huggingface_hub: HF_TOKEN_PATH > HF_HOME/token > ~/.cache/huggingface/token
    """
    candidates = []
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        candidates.append(Path(explicit))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "token")
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")

    for path in candidates:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def get_secret(name: str) -> Optional[str]:
    """resolve secret ตามลำดับ env > keyring > credentials file > (hf) ไฟล์ของ huggingface-cli"""
    for env_name in SECRET_ENV_VARS.get(name, []):
        value = os.environ.get(env_name)
        if value:
            return value

    kr = _keyring()
    if kr is not None:
        value = _kr_get(kr, name)
        if value:
            return value

    stored = _read_credentials_file().get(name)
    if stored:
        return stored
    # ท้ายสุดสำหรับ hf เท่านั้น — ของที่ผู้ใช้ตั้งกับ LMDS เองต้องชนะไฟล์ของเครื่องมือตัวอื่นเสมอ
    return _hf_cli_token() if name == "hf" else None


def set_secret(name: str, value: str) -> str:
    """เก็บ secret; คืน backend ที่ใช้ ('keyring' หรือ 'file') · ค่าที่ผิดรูป = ValueError (ดู validate_secret)"""
    validate_secret(name, value)

    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, name, value)
            return "keyring"
        except Exception:
            pass

    values = _read_credentials_file()
    values[name] = value
    _write_credentials_file(values)
    return "file"


def delete_secret(name: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, name)
        except Exception:
            pass
    values = _read_credentials_file()
    if name in values:
        del values[name]
        _write_credentials_file(values)


def secret_source(name: str) -> Optional[str]:
    """บอกว่า secret มาจากไหน: 'env' / 'keyring' / 'file' / None — ใช้แสดงใน config show"""
    for env_name in SECRET_ENV_VARS.get(name, []):
        if os.environ.get(env_name):
            return "env"
    kr = _keyring()
    if kr is not None and _kr_get(kr, name):
        return "keyring"
    if name in _read_credentials_file():
        return "file"
    if name == "hf" and _hf_cli_token():
        return "huggingface-cli"
    return None
