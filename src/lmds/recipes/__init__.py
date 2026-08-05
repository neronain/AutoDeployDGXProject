"""สูตรที่ผ่านการรันจริงและ settings hints ที่มีที่มา — ใช้เมื่อไม่มี LLM provider

ที่มา: ทีม SI และลูกค้าหลายรายไม่มี API key ของ LLM ทำให้ `lmds deploy` ตกไปใช้ rule-based
ซึ่งรู้แค่ "GGUF → llama.cpp, safetensors → vLLM" ไม่รู้เรื่องเฉพาะรุ่น เช่น DeepSeek V4
บังคับ kv-cache fp8 หรือ Qwen3-Coder NVFP4 ต้องใช้ image ที่มี kernel ตรงรุ่น — deploy ผ่าน
แต่ start ไม่ขึ้น

รายการ `hardware-validated` เก็บสิ่งที่รันผ่านจริงบนฮาร์ดแวร์ ส่วน `settings-only` เก็บ hint
แบบ portable จากแหล่งอ้างอิง แต่ไม่อ้างว่า bundle ของ LMDS รันผ่านแล้ว ทุกสูตรมี `source`,
`status` และ `validated_on` กำกับ · เป็นความรู้ deterministic จึงไม่ขัดกับหลักการที่ LLM
ห้ามเขียน Bash — สูตรแค่เติมค่าลง DeploymentPlan เหมือนที่ LLM ทำ
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
    status: str = "hardware-validated"

    def image_applies_to(self, memory_model: str) -> bool:
        return not self.image_for or memory_model in self.image_for

    @property
    def is_hardware_validated(self) -> bool:
        return self.status == "hardware-validated"

    @property
    def summary(self) -> str:
        bits = [self.status, self.engine or "?"]
        if self.image:
            bits.append(self.image)
        if self.tool_calling.get("parser"):
            bits.append(f"tools:{self.tool_calling['parser']}")
        if self.reasoning.get("parser"):
            bits.append(f"reasoning:{self.reasoning['parser']}")
        return " · ".join(bits)


@lru_cache(maxsize=1)
def load_catalog() -> list[Recipe]:
    """อ่านสูตรทั้งหมด — แฟ้มพังต้องไม่ทำให้ deploy ล้ม แค่ไม่มีสูตรให้ใช้"""
    try:
        raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    entries = raw.get("recipes")
    if not isinstance(entries, list):
        return []
    known = set(Recipe.__dataclass_fields__)
    return [
        Recipe(**{k: v for k, v in entry.items() if k in known})
        for entry in entries
        if isinstance(entry, dict) and entry.get("match")
    ]


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
