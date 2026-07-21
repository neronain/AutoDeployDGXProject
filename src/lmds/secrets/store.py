"""Secret store ของ LMDS

ลำดับการ resolve (สูง → ต่ำ):
  1. environment variable
  2. OS keyring (ถ้าติดตั้ง package `keyring` และใช้งานได้)
  3. credentials file (~/.config/lmds/credentials, สิทธิ์ 0600, รูปแบบ KEY=VALUE)

กติกา FR-7: secret ห้ามลง config.yaml, bundle, log หรือ output ใด ๆ — ใช้ redact() คุมทุกทางออก
"""

from __future__ import annotations

import os
import stat
from typing import Optional

from lmds.config.paths import credentials_file, ensure_config_dir

KEYRING_SERVICE = "lmds"

# ชื่อ secret ที่ระบบรู้จัก → env var ที่ยอมรับ (เรียงตามลำดับความสำคัญ)
SECRET_ENV_VARS: dict[str, list[str]] = {
    "openai": ["LMDS_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "gemini": ["LMDS_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["LMDS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"],
    "openai-compat": ["LMDS_OPENAI_COMPAT_API_KEY"],
    "hf": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
}


def _keyring():
    try:
        import keyring  # type: ignore

        return keyring
    except Exception:
        return None


def _read_credentials_file() -> dict[str, str]:
    path = credentials_file()
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _write_credentials_file(values: dict[str, str]) -> None:
    ensure_config_dir()
    path = credentials_file()
    body = "# LMDS credentials — อย่า commit ไฟล์นี้\n"
    body += "".join(f"{k}={v}\n" for k, v in sorted(values.items()))
    path.write_text(body, encoding="utf-8")
    path.chmod(0o600)


def check_credentials_permissions() -> Optional[str]:
    """คืนข้อความเตือนถ้าสิทธิ์ไฟล์ credentials หลวมเกินไป (group/other อ่านได้)"""
    path = credentials_file()
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"คำเตือน: {path} มีสิทธิ์ {oct(mode)} — ควรเป็น 0600 (chmod 600)"
    return None


def get_secret(name: str) -> Optional[str]:
    """resolve secret ตามลำดับ env > keyring > credentials file; คืน None ถ้าไม่พบ"""
    for env_name in SECRET_ENV_VARS.get(name, []):
        value = os.environ.get(env_name)
        if value:
            return value

    kr = _keyring()
    if kr is not None:
        try:
            value = kr.get_password(KEYRING_SERVICE, name)
            if value:
                return value
        except Exception:
            pass

    return _read_credentials_file().get(name)


def set_secret(name: str, value: str) -> str:
    """เก็บ secret; คืน backend ที่ใช้ ('keyring' หรือ 'file')"""
    if not value:
        raise ValueError("ค่า secret ว่างเปล่า")

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
    if kr is not None:
        try:
            if kr.get_password(KEYRING_SERVICE, name):
                return "keyring"
        except Exception:
            pass
    if name in _read_credentials_file():
        return "file"
    return None
