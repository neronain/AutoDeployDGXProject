"""Redaction filter — ทางออกทุกทาง (log, README, session log, stdout ที่บันทึก) ต้องผ่านตัวนี้"""

from __future__ import annotations

import re
from typing import Iterable

MASK = "[REDACTED]"

# รูปแบบ token ที่รู้จัก — กันกรณี secret หลุดมากับข้อความอื่น (เช่น error จาก provider)
_TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),        # OpenAI / Anthropic style
    re.compile(r"hf_[A-Za-z0-9]{16,}"),           # Hugging Face
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),       # Google API key
    re.compile(r"(?i)(bearer\s+)[a-z0-9_\-.~+/]{16,}=*"),
]


def redact(text: str, known_secrets: Iterable[str] = ()) -> str:
    """ลบ secret ที่รู้ค่า + pattern token ที่รู้จัก ออกจากข้อความ"""
    if not text:
        return text
    for secret in known_secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, MASK)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) + MASK) if (m.lastindex or 0) >= 1 else MASK,
            text,
        )
    return text


def mask_preview(secret: str | None) -> str:
    """แสดงตัวอย่าง secret แบบปลอดภัยใน UI เช่น 'sk-…f3ab' — ไม่เกิน 4 ตัวท้าย"""
    if not secret:
        return "(ไม่ได้ตั้งค่า)"
    if len(secret) < 12:
        return "****"
    return f"{secret[:3]}…{secret[-4:]}"
