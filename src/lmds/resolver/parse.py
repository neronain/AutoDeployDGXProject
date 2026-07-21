"""แปลง input ของผู้ใช้ (URL หรือ model ID) → ModelSource

รองรับเฟสนี้: Hugging Face (repo, ลิงก์ tree/blob/resolve, ลิงก์ไฟล์ .gguf ตรง)
Ollama / NGC: โครงไว้แล้ว แจ้งชัดว่ายังไม่รองรับ (เฟสถัดไป)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")


class SourceError(ValueError):
    """input ไม่ใช่รูปแบบที่ระบบรู้จัก"""


class UnsupportedSource(SourceError):
    """รู้จักแต่ยังไม่รองรับ (เช่น Ollama ในเฟสนี้)"""


@dataclass(frozen=True)
class ModelSource:
    kind: str  # "huggingface"
    repo_id: str
    revision: str | None = None  # branch/tag/sha ที่ผู้ใช้ระบุมากับลิงก์
    filename: str | None = None  # กรณีลิงก์ชี้ไฟล์เดียว (เช่น .gguf)

    @property
    def display(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        file = f" [{self.filename}]" if self.filename else ""
        return f"{self.repo_id}{rev}{file}"


def parse_source(text: str) -> ModelSource:
    text = text.strip().rstrip("/")
    if not text:
        raise SourceError("input ว่างเปล่า")

    if "://" in text or text.startswith(("huggingface.co/", "hf.co/", "ollama.com/", "www.")):
        url = urlparse(text if "://" in text else f"https://{text}")
        host = (url.hostname or "").lower()

        if host in _HF_HOSTS:
            return _parse_hf_path(url.path)
        if host in {"ollama.com", "www.ollama.com", "registry.ollama.ai"}:
            raise UnsupportedSource(
                "ลิงก์ Ollama ยังไม่รองรับในเฟสนี้ (อยู่ใน roadmap เฟส 2) — "
                "ใช้ลิงก์ Hugging Face ของ GGUF ตัวเดียวกันแทนได้"
            )
        if host in {"catalog.ngc.nvidia.com", "ngc.nvidia.com"}:
            raise UnsupportedSource("ลิงก์ NVIDIA NGC ยังไม่รองรับในเฟสนี้ (roadmap เฟส 2)")
        raise SourceError(f"ไม่รู้จักโดเมน: {host}")

    if _REPO_ID_RE.match(text):
        return ModelSource(kind="huggingface", repo_id=text)

    raise SourceError(
        f"ไม่เข้าใจ input: {text!r} — ใช้รูปแบบ org/model หรือลิงก์ huggingface.co เต็ม"
    )


def _parse_hf_path(path: str) -> ModelSource:
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise SourceError("ลิงก์ Hugging Face ไม่มี path")
    if parts[0] in {"datasets", "spaces", "collections"}:
        raise SourceError(f"ลิงก์เป็น {parts[0]} — ระบบรองรับเฉพาะ model repository")
    if len(parts) < 2:
        raise SourceError("ลิงก์ Hugging Face ต้องมีรูปแบบ /org/model")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision: str | None = None
    filename: str | None = None

    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve", "raw"}:
        revision = parts[3] if parts[3] != "main" else None
        rest = parts[4:]
        if rest and parts[2] != "tree":
            filename = "/".join(rest)
    elif len(parts) > 2:
        raise SourceError(f"ไม่เข้าใจ path ของลิงก์: {path}")

    return ModelSource(kind="huggingface", repo_id=repo_id, revision=revision, filename=filename)
