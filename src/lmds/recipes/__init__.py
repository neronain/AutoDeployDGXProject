"""สูตรที่ผ่านการรันจริง — ใช้เมื่อเครื่องลูกค้าไม่มี LLM provider

ที่มา: ทีม SI และลูกค้าหลายรายไม่มี API key ของ LLM ทำให้ `lmds deploy` ตกไปใช้ rule-based
ซึ่งรู้แค่ "GGUF → llama.cpp, safetensors → vLLM" ไม่รู้เรื่องเฉพาะรุ่น เช่น DeepSeek V4
บังคับ kv-cache fp8 หรือ Qwen3-Coder NVFP4 ต้องใช้ image ที่มี kernel ตรงรุ่น — deploy ผ่าน
แต่ start ไม่ขึ้น

แฟ้มนี้จึงเก็บสิ่งที่ **รันผ่านจริงบนฮาร์ดแวร์แล้ว** ไม่ใช่การเดา ทุกสูตรมี `source` และ
`validated_on` กำกับ · เป็นความรู้ที่กำหนดไว้ตายตัว (deterministic) จึงไม่ขัดกับหลักการของ
โปรเจกต์ที่ว่า LLM ห้ามเขียน Bash — สูตรแค่เติมค่าลง DeploymentPlan เหมือนที่ LLM ทำ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")


@dataclass
class Recipe:
    match: str
    label: str = ""
    engine: str = ""
    image: str = ""
    # image ที่ทดสอบมาผูกกับสถาปัตยกรรม — build ของ DGX Spark (ARM64/SM121) ใช้กับ RTX ไม่ได้
    # และกลับกัน · ว่าง = ใช้ได้ทุกเครื่อง
    image_for: list[str] = field(default_factory=list)
    serving: dict = field(default_factory=dict)
    tool_calling: dict = field(default_factory=dict)
    reasoning: dict = field(default_factory=dict)
    speculative: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    validated_on: str = ""
    # เฉพาะสูตรที่ดึงมาจากรีโป controller ของทีม — บอกว่ามาจากสคริปต์ตัวไหนและออกแบบมาให้รันแบบไหน
    controller: str = ""
    topology: str = ""

    def image_applies_to(self, memory_model: str) -> bool:
        return not self.image_for or memory_model in self.image_for

    @property
    def summary(self) -> str:
        bits = [self.engine or "?"]
        if self.image:
            bits.append(self.image)
        if self.tool_calling.get("parser"):
            bits.append(f"tools:{self.tool_calling['parser']}")
        if self.reasoning.get("parser"):
            bits.append(f"reasoning:{self.reasoning['parser']}")
        return " · ".join(bits)


def _read(path: Path) -> list[dict]:
    """รายการสูตรดิบจากไฟล์ YAML หนึ่งไฟล์ — ไฟล์พัง/ไม่มี ต้องไม่ทำให้ deploy ล้ม"""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    entries = raw.get("recipes")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("match")]


def synced_path() -> Path:
    """สูตรที่ดึงมาจากรีโป controller ภายนอก (`lmds recipes sync`)"""
    from lmds.config.paths import config_dir

    return config_dir() / "recipes-synced.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> list[Recipe]:
    """สูตรทั้งหมด = ที่มากับ LMDS + ที่ดึงมาจากรีโป controller ของทีม

    ของที่ดึงมาชนะของที่มากับตัวโปรแกรมเมื่อชนกัน — รีโปของทีมคือต้นทางที่รันจริง
    และอัปเดตบ่อยกว่า catalog ที่ฝังมากับเวอร์ชันที่ติดตั้งไว้
    """
    known = set(Recipe.__dataclass_fields__)
    merged: dict[str, dict] = {}
    for entry in [*_read(CATALOG_PATH), *_read(synced_path())]:
        merged[str(entry["match"]).lower()] = entry
    return [Recipe(**{k: v for k, v in entry.items() if k in known}) for entry in merged.values()]


def find_recipe(repo_id: str) -> Recipe | None:
    """สูตรของโมเดลนี้ — เทียบแบบไม่สนตัวพิมพ์ และยอมให้ `match` เป็น prefix

    prefix ใช้ครอบ variant ที่ผู้ให้บริการ quantize ออกมาหลายตัวจากฐานเดียวกัน
    ตัวที่ยาวกว่าชนะเสมอ เพื่อให้สูตรเฉพาะเจาะจงมาก่อนสูตรกว้าง
    """
    wanted = (repo_id or "").lower()
    if not wanted:
        return None
    matches = [r for r in load_catalog() if wanted.startswith(r.match.lower())]
    return max(matches, key=lambda r: len(r.match)) if matches else None
