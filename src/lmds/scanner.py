"""ค้นหา weight ที่มีอยู่แล้วบนเครื่อง — ไม่ว่าจะถูกเก็บไว้แบบไหน

เครื่องลูกค้าส่วนใหญ่มีโมเดลอยู่ก่อนติดตั้ง LMDS และไม่ได้จัดระเบียบแบบเดียวกับเรา:
HF cache มีสองเลย์เอาต์ (`$HF_HOME/hub/models--X` กับ `$HF_HOME/models--X`), บางคนตั้ง
`HF_HUB_CACHE` ไปที่ดิสก์อื่น, ไฟล์ GGUF มักวางไว้เป็นโฟลเดอร์ธรรมดา

โมดูลนี้ "หาให้เจอ" เท่านั้น — ไม่ย้าย ไม่ลบ ไม่แก้อะไรทั้งสิ้น เพราะของพวกนี้เป็นของผู้ใช้
และการย้าย 150 GB เงียบ ๆ คือสิ่งที่ยอมรับไม่ได้
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ที่ที่โมเดลไปโผล่ได้จริงบนเครื่องที่ไม่เคยจัดระเบียบ
_ENV_ROOTS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "MODEL_DIR", "LLAMA_CACHE")
_COMMON_ROOTS = (
    "~/.cache/huggingface", "~/.cache/huggingface/hub", "~/models", "~/data/models",
    "/models", "/opt/models", "/srv/models", "/data/models", "/mnt/models",
)
# ไฟล์เล็กที่ไม่ควรถูกนับเป็น weight (โหลดไม่ครบ/ไฟล์ตัวอย่าง)
_MIN_GGUF_BYTES = 32 * 1024 * 1024


@dataclass
class FoundModel:
    """โมเดลหนึ่งตัวที่เจอบนดิสก์ — path เป็นของจริงที่ชี้ไปได้เลย"""

    kind: str  # "hf" | "gguf"
    name: str  # org/model สำหรับ hf · ชื่อไฟล์สำหรับ gguf
    path: str
    size_bytes: int = 0
    revisions: list[str] = field(default_factory=list)
    shard_count: int = 0
    # เลย์เอาต์ของ HF cache ที่พบ: "hub" (ปัจจุบัน) หรือ "root" (เก่า) — ต่างกันตอนบอก
    # HF_HUB_CACHE ให้คอนเทนเนอร์ ถ้าบอกผิดจะได้ LocalEntryNotFoundError ทั้งที่ไฟล์ครบ
    layout: str = ""

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1024**3, 1)

    @property
    def hub_cache_root(self) -> str:
        """ค่าที่ควรตั้งเป็น HF_HUB_CACHE เพื่อให้ไลบรารีของ HF มองเห็นโมเดลนี้"""
        if self.kind != "hf":
            return ""
        return str(Path(self.path).parent)


def _dir_size(path: Path, limit_files: int = 20000) -> int:
    total = seen = 0
    for entry in path.rglob("*"):
        seen += 1
        if seen > limit_files:
            break
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
            elif entry.is_symlink():
                # HF cache ใช้ symlink ชี้ไป blobs/ — ขนาดจริงอยู่ปลายทาง
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def candidate_roots(extra: list[str] | None = None) -> list[Path]:
    """ที่ที่จะไปค้น — env ของผู้ใช้มาก่อน แล้วค่อยที่ที่นิยมวางกัน"""
    roots: list[Path] = []
    for name in _ENV_ROOTS:
        value = os.environ.get(name)
        if value:
            roots.append(Path(value).expanduser())
    for raw in list(extra or []) + list(_COMMON_ROOTS):
        roots.append(Path(raw).expanduser())

    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in unique:
            unique.append(resolved)
    return unique


def _scan_hf_root(root: Path) -> list[FoundModel]:
    """หา models--org--name/ ที่มี snapshots จริง — รองรับทั้ง <root>/hub และ <root> ตรง ๆ"""
    found = []
    for base, layout in ((root / "hub", "hub"), (root, "root")):
        if not base.is_dir():
            continue
        for entry in sorted(base.glob("models--*")):
            snapshots = entry / "snapshots"
            if not snapshots.is_dir():
                continue
            revisions = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
            if not revisions:
                continue
            shards = 0
            for revision in revisions:
                shards = max(shards, len(list((snapshots / revision).glob("*.safetensors"))))
            found.append(FoundModel(
                kind="hf",
                name=entry.name.replace("models--", "", 1).replace("--", "/"),
                path=str(entry),
                size_bytes=_dir_size(entry),
                revisions=revisions,
                shard_count=shards,
                layout=layout,
            ))
    return found


def _scan_gguf_root(root: Path, max_depth: int = 3) -> list[FoundModel]:
    found = []
    for path in root.rglob("*.gguf"):
        try:
            if len(path.relative_to(root).parts) > max_depth + 1:
                continue
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        if size < _MIN_GGUF_BYTES:
            continue
        found.append(FoundModel(kind="gguf", name=path.name, path=str(path), size_bytes=size))
    return found


def scan(extra_roots: list[str] | None = None) -> list[FoundModel]:
    """weight ทั้งหมดที่เจอบนเครื่องนี้ เรียงจากใหญ่ไปเล็ก

    ตัวซ้ำ (path เดียวกันมาจากหลาย root) ถูกรวมให้แล้ว
    """
    results: dict[str, FoundModel] = {}
    for root in candidate_roots(extra_roots):
        for model in _scan_hf_root(root) + _scan_gguf_root(root):
            results.setdefault(model.path, model)
    return sorted(results.values(), key=lambda m: -m.size_bytes)


def find_model(repo_id: str, extra_roots: list[str] | None = None) -> FoundModel | None:
    """หาโมเดลตัวหนึ่งโดยเฉพาะ — ใช้ตอบว่า "ต้องโหลดใหม่ไหม" ก่อน download หลายสิบ GB"""
    wanted = repo_id.lower()
    for model in scan(extra_roots):
        if model.kind == "hf" and model.name.lower() == wanted:
            return model
    return None
