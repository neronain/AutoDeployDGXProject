"""ค้นหา weight ที่มีอยู่แล้วบนเครื่อง — ไม่ว่าจะถูกเก็บไว้แบบไหน

เครื่องลูกค้าส่วนใหญ่มีโมเดลอยู่ก่อนติดตั้ง LMDS และไม่ได้จัดระเบียบแบบเดียวกับเรา:
HF cache มีสองเลย์เอาต์ (`$HF_HOME/hub/models--X` กับ `$HF_HOME/models--X`), บางคนตั้ง
`HF_HUB_CACHE` ไปที่ดิสก์อื่น, ไฟล์ GGUF มักวางไว้เป็นโฟลเดอร์ธรรมดา

โมดูลนี้ "หาให้เจอ" เท่านั้น — ไม่ย้าย ไม่ลบ ไม่แก้อะไรทั้งสิ้น เพราะของพวกนี้เป็นของผู้ใช้
และการย้าย 150 GB เงียบ ๆ คือสิ่งที่ยอมรับไม่ได้
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ที่ที่โมเดลไปโผล่ได้จริงบนเครื่องที่ไม่เคยจัดระเบียบ
_ENV_ROOTS = (
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "MODEL_DIR", "LLAMA_CACHE",
    "OLLAMA_MODELS",
)
_COMMON_ROOTS = (
    "~/.cache/huggingface", "~/.cache/huggingface/hub", "~/models", "~/data/models",
    "/models", "/opt/models", "/srv/models", "/data/models", "/mnt/models",
    # Ollama desktop/user install และ Linux service install ตามลำดับ
    "~/.ollama/models", "/usr/share/ollama/.ollama/models",
)
# ไฟล์เล็กที่ไม่ควรถูกนับเป็น weight (โหลดไม่ครบ/ไฟล์ตัวอย่าง)
_MIN_GGUF_BYTES = 32 * 1024 * 1024
_OLLAMA_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OLLAMA_MANIFEST_CAP = 4 * 1024 * 1024
_OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"


@dataclass
class FoundModel:
    """โมเดลหนึ่งตัวที่เจอบนดิสก์ — path เป็นของจริงที่ชี้ไปได้เลย"""

    kind: str  # "hf" | "gguf" | "ollama"
    name: str  # org/model สำหรับ hf · ชื่อไฟล์สำหรับ gguf · namespace/model:tag สำหรับ ollama
    path: str
    size_bytes: int = 0
    revisions: list[str] = field(default_factory=list)
    shard_count: int = 0
    # เลย์เอาต์ของ HF cache ที่พบ: "hub" (ปัจจุบัน) หรือ "root" (เก่า) — ต่างกันตอนบอก
    # HF_HUB_CACHE ให้คอนเทนเนอร์ ถ้าบอกผิดจะได้ LocalEntryNotFoundError ทั้งที่ไฟล์ครบ
    layout: str = ""
    aliases: list[str] = field(default_factory=list)

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
            # HF cache เก็บไฟล์จริงใน blobs/ แล้ว snapshots/ เป็น symlink ชี้มา — นับทั้งสองอย่าง
            # จะได้ขนาดเป็นเท่าตัว ซึ่งทำให้วางแผนพื้นที่ผิด · นับเฉพาะไฟล์จริง
            if entry.is_file() and not entry.is_symlink():
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


def _scan_ollama_root(root: Path) -> list[FoundModel]:
    """จับคู่ local manifest กับ content-addressed blob ที่ไม่มีนามสกุล

    Ollama เก็บ manifest เป็น
    manifests/registry.ollama.ai/<namespace>/<model>/<tag> และ blob เป็น
    blobs/sha256-<hex> จึงห้ามใช้ rglob ไฟล์ไร้นามสกุลทั่ว root (ทั้งช้าและ false-positive สูง)
    """
    manifest_root = root / "manifests" / "registry.ollama.ai"
    blob_root = root / "blobs"
    if not manifest_root.is_dir() or not blob_root.is_dir():
        return []

    found: list[FoundModel] = []
    for path in sorted(manifest_root.glob("*/*/*")):
        try:
            rel = path.relative_to(manifest_root)
            namespace, model, tag = rel.parts
            if not path.is_file() or path.stat().st_size > _OLLAMA_MANIFEST_CAP:
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            layers = doc.get("layers") if isinstance(doc, dict) else None
            if not isinstance(layers, list):
                continue
            model_layers = [
                layer for layer in layers
                if isinstance(layer, dict) and layer.get("mediaType") == _OLLAMA_MODEL_LAYER
            ]
            if len(model_layers) != 1:
                continue
            layer = model_layers[0]
            digest = layer.get("digest")
            expected_size = layer.get("size")
            if not isinstance(digest, str) or not _OLLAMA_DIGEST_RE.fullmatch(digest):
                continue
            if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
                continue
            blob = blob_root / digest.replace(":", "-", 1)
            actual_size = blob.stat().st_size
            if not blob.is_file() or actual_size != expected_size:
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        found.append(FoundModel(
            kind="ollama",
            name=f"{namespace}/{model}:{tag}",
            path=str(blob),
            size_bytes=actual_size,
            revisions=[digest],
        ))
    return found


def scan(extra_roots: list[str] | None = None) -> list[FoundModel]:
    """weight ทั้งหมดที่เจอบนเครื่องนี้ เรียงจากใหญ่ไปเล็ก

    ตัวซ้ำ (path เดียวกันมาจากหลาย root) ถูกรวมให้แล้ว
    """
    results: dict[str, FoundModel] = {}
    for root in candidate_roots(extra_roots):
        for model in _scan_hf_root(root) + _scan_gguf_root(root) + _scan_ollama_root(root):
            existing = results.setdefault(model.path, model)
            if model.name != existing.name and model.name not in existing.aliases:
                existing.aliases.append(model.name)
    return sorted(results.values(), key=lambda m: -m.size_bytes)


def find_model(
    repo_id: str,
    extra_roots: list[str] | None = None,
    *,
    revision: str | None = None,
    digest: str | None = None,
) -> FoundModel | None:
    """หาโมเดลตัวหนึ่งโดยเฉพาะ — ใช้ตอบว่า "ต้องโหลดใหม่ไหม" ก่อน download หลายสิบ GB"""
    wanted = repo_id.lower()
    for model in scan(extra_roots):
        if model.kind == "hf" and model.name.lower() == wanted:
            return model
        if model.kind == "ollama":
            names = [model.name, *model.aliases]
            if revision:
                names = [name for name in names if name.lower() == f"{wanted}:{revision.lower()}"]
            else:
                names = [name for name in names if name.rsplit(":", 1)[0].lower() == wanted]
            if names and (
                digest is None
                or digest.removeprefix("sha256:").lower()
                in {value.removeprefix("sha256:").lower() for value in model.revisions}
            ):
                return model
    return None
