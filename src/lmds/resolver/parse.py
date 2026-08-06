"""แปลง input ของผู้ใช้ (URL หรือ model ID) → ModelSource

รองรับ: Hugging Face (repo, ลิงก์ tree/blob/resolve, ลิงก์ไฟล์ .gguf ตรง)
        Ollama (ollama.com / registry.ollama.ai — resolve เป็น GGUF blob ใน registry)
NGC: โครงไว้แล้ว แจ้งชัดว่ายังไม่รองรับ (เฟสถัดไป)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")
_OLLAMA_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,79}$")
_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")


class SourceError(ValueError):
    """input ไม่ใช่รูปแบบที่ระบบรู้จัก"""


class UnsupportedSource(SourceError):
    """รู้จักแต่ยังไม่รองรับ (เช่น Ollama ในเฟสนี้)"""


@dataclass(frozen=True)
class ModelSource:
    kind: str  # "huggingface" | "ollama"
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

    if "://" in text or text.startswith(
        (
            "huggingface.co/", "hf.co/",
            "ollama.com/", "registry.ollama.ai/",
            "catalog.ngc.nvidia.com/", "ngc.nvidia.com/",
            "www.",
        )
    ):
        url = urlparse(text if "://" in text else f"https://{text}")
        host = (url.hostname or "").lower()

        if host in _HF_HOSTS:
            return _parse_hf_path(url.path)
        if host in {"ollama.com", "www.ollama.com", "registry.ollama.ai"}:
            return _parse_ollama_path(url.path)
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


def _parse_ollama_path(path: str) -> ModelSource:
    """ollama.com/qwen3 · ollama.com/library/qwen3:8b · registry.ollama.ai/v2/<ns>/<name>/manifests/<tag>

    namespace ที่ไม่ระบุคือ `library` และ tag ที่ไม่ระบุคือ `latest` — ตรงกับที่ CLI ของ Ollama
    ตีความ `ollama pull qwen3` · tag เก็บใน revision เพราะมันคือ "เวอร์ชันที่ผู้ใช้ขอ" เหมือน
    branch/tag ของ HF และเป็นสิ่งที่ต้องใช้เรียก manifest
    """
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise SourceError("ลิงก์ Ollama ไม่มีชื่อโมเดล")

    # path ของ registry API: /v2/<namespace>/<name>/manifests/<tag>
    if parts[0] == "v2":
        if len(parts) != 5 or parts[3] != "manifests":
            raise SourceError(f"ไม่เข้าใจ path ของ registry API: {path}")
        return _ollama_source(parts[1], parts[2], parts[4])

    if parts[0] == "search":
        raise SourceError("ลิงก์เป็นหน้าค้นหา — ใส่ลิงก์ของโมเดลตัวใดตัวหนึ่ง")

    # Ollama รับ ref แบบ hf.co/<org>/<model>[:quant] ด้วย ซึ่งชี้ไป Hugging Face ไม่ใช่ registry
    # ไม่แปลงให้เอง เพราะ tag ของ ref นั้นคือ quant ที่ผู้ใช้เลือก แต่ ModelSource ต้องการ
    # *ชื่อไฟล์* จริงซึ่งรู้ไม่ได้จนกว่าจะ list repo — เดาแล้วผิดเงียบแย่กว่าบอกให้ใส่ลิงก์ HF ตรง ๆ
    if parts[0] in {"hf.co", "huggingface.co"}:
        raise SourceError(
            "ref แบบ hf.co/... คือโมเดลบน Hugging Face ไม่ใช่ของ Ollama registry — "
            "ใส่ลิงก์ huggingface.co/<org>/<model> โดยตรง แล้วเลือกไฟล์ quant ตอน deploy"
        )

    if len(parts) == 1:
        namespace, name = "library", parts[0]
    elif len(parts) == 2:
        namespace, name = parts
    else:
        raise SourceError(f"ไม่เข้าใจ path ของลิงก์ Ollama: {path}")

    name, _, tag = name.partition(":")
    if not name:
        raise SourceError("ลิงก์ Ollama ไม่มีชื่อโมเดล")

    return _ollama_source(namespace, name, tag or "latest")


def _ollama_source(namespace: str, name: str, tag: str) -> ModelSource:
    """validate ตาม Ollama types/model.Name ก่อนประกอบ registry path

    ไม่ใช่ขอบเขต security (origin ถูกตรึงไว้แล้ว) แต่ช่วยให้ input เสียจบเป็น SourceError
    แทน HTTP/manifest error ที่ชวนเข้าใจผิดภายหลัง
    """
    if not _OLLAMA_NAMESPACE_RE.fullmatch(namespace):
        raise SourceError(f"Ollama namespace ไม่ถูกต้อง: {namespace!r}")
    if not _OLLAMA_NAME_RE.fullmatch(name):
        raise SourceError(f"Ollama model name ไม่ถูกต้อง: {name!r}")
    if not _OLLAMA_NAME_RE.fullmatch(tag):
        raise SourceError(f"Ollama tag ไม่ถูกต้อง: {tag!r}")
    return ModelSource(kind="ollama", repo_id=f"{namespace}/{name}", revision=tag)
