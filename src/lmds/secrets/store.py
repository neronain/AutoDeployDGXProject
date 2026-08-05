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
        try:
            value = kr.get_password(KEYRING_SERVICE, name)
            if value:
                return value
        except Exception:
            pass

    stored = _read_credentials_file().get(name)
    if stored:
        return stored
    # ท้ายสุดสำหรับ hf เท่านั้น — ของที่ผู้ใช้ตั้งกับ LMDS เองต้องชนะไฟล์ของเครื่องมือตัวอื่นเสมอ
    return _hf_cli_token() if name == "hf" else None


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
    if name == "hf" and _hf_cli_token():
        return "huggingface-cli"
    return None
