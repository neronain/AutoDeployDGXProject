"""แปลง input ของผู้ใช้ (URL หรือ model ID) → ModelSource

รองรับเฟสนี้: Hugging Face (repo, ลิงก์ tree/blob/resolve, ลิงก์ไฟล์ .gguf ตรง)
Ollama / NGC: โครงไว้แล้ว แจ้งชัดว่ายังไม่รองรับ (เฟสถัดไป)

input ที่ผิดรูปแบบ **ทุกแบบ** ต้องออกมาเป็น SourceError — ผู้เรียก (CLI/เว็บ) จับตัวนี้ตัวเดียวแล้วแสดงเป็นข้อความ
(เคสจริง 2026-10-06: `lmds inspect 'https://[::1/unsloth/x'` ได้ traceback `ValueError: Invalid IPv6 URL`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")
# path แรกของ huggingface.co ที่ไม่ใช่ชื่อ org — เดิม `huggingface.co/settings/tokens` ถูกรับเป็น repo "settings/tokens"
_NOT_A_NAMESPACE = {
    "settings", "docs", "blog", "models", "api", "organizations", "login", "join", "logout", "pricing", "papers",
    "posts", "tasks", "learn", "new", "chat", "enterprise", "notifications", "search", "welcome", "inference",
    "inference-endpoints", "storage", "billing", "changelog", "brand", "jobs", "support", "security",
    "terms-of-service", "privacy", "content-guidelines",
}
# ref ที่ Hub ตั้งชื่อแบบมี / เสมอ: refs/pr/<เลข> (PR) · refs/convert/<ชื่อ> (parquet/duckdb) · refs/heads|tags/<ชื่อ>
_REF_KINDS = {"pr", "convert", "heads", "tags"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class SourceError(ValueError):
    """input ไม่ใช่รูปแบบที่ระบบรู้จัก"""


class UnsupportedSource(SourceError):
    """รู้จักแต่ยังไม่รองรับ (เช่น Ollama ในเฟสนี้)"""


@dataclass(frozen=True)
class ModelSource:
    kind: str  # "huggingface"
    repo_id: str
    revision: str | None = None  # branch/tag/sha ที่ผู้ใช้ระบุมากับลิงก์
    filename: str | None = None  # กรณีลิงก์ชี้ไฟล์ .gguf ไฟล์เดียว = เลือกไฟล์นั้น (→ llama.cpp)
    # สิ่งที่ parser *ตีความเอง* จากลิงก์ที่กำกวม (branch ที่มี / · ลิงก์ชี้ไฟล์ที่ไม่ใช่ weight) — ผู้เรียกแสดงให้ผู้ใช้เห็น
    # ไม่นับในการเทียบว่า source สองตัวเท่ากันไหม
    notes: tuple[str, ...] = field(default=(), compare=False)

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
        try:
            url = urlparse(text if "://" in text else f"https://{text}")
            host = (url.hostname or "").lower()
        except ValueError as exc:
            # urlparse โยน ValueError เองกับ URL ที่ผิดรูป (วงเล็บ IPv6 ไม่ปิด ฯลฯ)
            raise SourceError(f"ลิงก์ผิดรูปแบบ: {text!r} ({exc}) — ใช้รูปแบบ org/model หรือลิงก์ huggingface.co เต็ม") from None
        if url.scheme not in ("http", "https"):
            raise SourceError(f"ไม่รองรับลิงก์แบบ {url.scheme}:// — ใช้ https://huggingface.co/org/model")

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

    return ModelSource(kind="huggingface", repo_id=_repo_id(text))


def _repo_id(text: str) -> str:
    """ตรวจ org/model — ใช้ทั้งกับ id ที่พิมพ์ตรง ๆ และ path ของลิงก์ (เดิมลิงก์ข้ามการตรวจนี้ไปทั้งหมด)"""
    repo_id = text.removesuffix(".git")      # ลิงก์ clone: huggingface.co/org/model.git
    if _REPO_ID_RE.match(repo_id):
        return repo_id
    if ":" in repo_id and _REPO_ID_RE.match(repo_id.split(":", 1)[0]):
        base, quant = repo_id.split(":", 1)
        raise SourceError(
            f"รูปแบบ {base}:{quant} (repo:quant แบบ ollama / llama.cpp -hf) ยังไม่รองรับ — ระบุไฟล์ GGUF ด้วย "
            f"--gguf {quant} หรือใส่ลิงก์ไฟล์ .gguf ตรง ๆ"
        )
    if "@" in repo_id and _REPO_ID_RE.match(repo_id.split("@", 1)[0]):
        base, rev = repo_id.split("@", 1)
        raise SourceError(f"รูปแบบ {base}@{rev} ยังไม่รองรับ — ระบุ revision ด้วย --revision {rev}")
    raise SourceError(
        f"ไม่เข้าใจ input: {text!r} — ใช้รูปแบบ org/model หรือลิงก์ huggingface.co เต็ม"
    )


def _parse_hf_path(path: str) -> ModelSource:
    # แยก segment ก่อนแล้วค่อย decode: `tree/refs%2Fpr%2F3` ต้องได้ revision "refs/pr/3" เป็นก้อนเดียว
    # และชื่อไฟล์ที่มี %20 ต้องกลับเป็นช่องว่าง (เดิมไม่ decode เลย → หาไฟล์ใน repo ไม่เจอ)
    parts = [unquote(p) for p in path.split("/") if p]
    if not parts:
        raise SourceError("ลิงก์ Hugging Face ไม่มี path")
    if parts[0] in {"datasets", "spaces", "collections"}:
        raise SourceError(f"ลิงก์เป็น {parts[0]} — ระบบรองรับเฉพาะ model repository")
    if parts[0] in _NOT_A_NAMESPACE:
        raise SourceError(
            f"ลิงก์นี้เป็นหน้า /{parts[0]} ของ Hugging Face ไม่ใช่ model repository — ใช้ลิงก์รูปแบบ huggingface.co/org/model")
    if len(parts) < 2:
        raise SourceError("ลิงก์ Hugging Face ต้องมีรูปแบบ /org/model")

    repo_id = _repo_id(f"{parts[0]}/{parts[1]}")
    revision: str | None = None
    filename: str | None = None
    notes: list[str] = []

    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve", "raw"}:
        rest = parts[3:]
        if rest[0] == "refs" and len(rest) >= 3 and rest[1] in _REF_KINDS:
            # refs/pr/1 · refs/convert/parquet — เดิมเอา segment เดียว ได้ revision "refs" แล้ว Hub ตอบ "ไม่พบ model repo"
            revision, rest = "/".join(rest[:3]), rest[3:]
        else:
            revision, rest = rest[0], rest[1:]
        is_file_link = parts[2] != "tree"
        if rest and not _is_pinned(revision) and (not is_file_link or len(rest) > 1):
            # huggingface.co/<repo>/tree/feature/foo: branch "feature" + โฟลเดอร์ "foo" หรือ branch "feature/foo"?
            # ลิงก์อย่างเดียวบอกไม่ได้ — เลือกแบบแรก (branch ที่มี / หายากกว่าโฟลเดอร์ย่อยมาก) แล้วบอกว่าเลือกอะไร
            guess = "/".join([revision, *rest[:-1]]) if is_file_link else "/".join([revision, *rest])
            notes.append(
                f"ตีความลิงก์เป็น revision {revision!r} + path {'/'.join(rest)!r} — ถ้า branch ชื่อมี / "
                f"(เช่น {guess!r}) ให้ระบุเองด้วย --revision"
            )
        if is_file_link and rest:
            name = "/".join(rest)
            if name.lower().endswith(".gguf"):
                filename = name
            else:
                # ลิงก์ชี้ไฟล์อื่นใน repo (README.md · config.json · model.safetensors) — ไม่ใช่ "ไฟล์ของโมเดลที่เลือก"
                # เดิมเก็บเป็น filename แล้ว inspector เอาไปหาใน GGUF variant
                notes.append(f"ลิงก์ชี้ไฟล์ {name} ซึ่งไม่ใช่ไฟล์ .gguf — ใช้เป็นลิงก์ของ repo {repo_id} ทั้งตัว")
        if revision == "main":
            revision = None
    elif len(parts) > 2:
        raise SourceError(f"ไม่เข้าใจ path ของลิงก์: {path}")

    return ModelSource(kind="huggingface", repo_id=repo_id, revision=revision, filename=filename, notes=tuple(notes))


def _is_pinned(revision: str) -> bool:
    """revision ที่ไม่กำกวมว่าจบตรงไหน: main · commit hash · refs/…"""
    return revision == "main" or revision.startswith("refs/") or _COMMIT_RE.match(revision) is not None
