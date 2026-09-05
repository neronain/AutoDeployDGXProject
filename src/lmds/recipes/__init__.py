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
    # แฟล็กเพิ่มที่ controller ที่พิสูจน์แล้วส่งให้ engine (`EXTRA_SERVE_ARGS_DEFAULT` ที่ publish พับมา)
    # — รูป "--flag value" ต่อรายการ · ยังผ่าน allowlist ของ harden เหมือนแฟล็กจาก LLM
    extra_flags: list[str] = field(default_factory=list)
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


# ความสามารถของโมเดล ไม่ใช่การตั้งค่าของเครื่อง — สูตรที่ดึงมาทับได้ถ้ามันระบุเอง
# แต่ถ้าเงียบ ต้องไม่ไปลบของที่ catalog รู้อยู่แล้ว (ดูเหตุผลใน load_catalog)
CAPABILITY_FIELDS = ("tool_calling", "reasoning")
# ฟิลด์ที่เป็น dict — รวมทีละคีย์ ไม่ใช่ทับทั้งก้อน (ดู _merge_entries)
_DICT_FIELDS = ("serving", "env", "tool_calling", "reasoning", "speculative")


def _parse_env(text: str) -> dict[str, str]:
    """`"K=V K2=V2"` (รูปที่ controller เก็บใน ENGINE_ENV) → {K: V}"""
    import shlex

    out: dict[str, str] = {}
    try:
        tokens = shlex.split(str(text))
    except ValueError:
        tokens = str(text).split()
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            if key:
                out[key] = value
    return out


def _parse_flags(text: str) -> list[str]:
    """`"--a 1 --b"` → ["--a 1", "--b"] — รูปเดียวกับ Serving.extra_flags"""
    import shlex

    try:
        tokens = shlex.split(str(text))
    except ValueError:
        tokens = str(text).split()
    out: list[str] = []
    for token in tokens:
        if token.startswith("-") or not out:
            out.append(token)
        else:
            out[-1] = f"{out[-1]} {token}"
    return [t for t in out if t.startswith("-")]


def _normalize(entry: dict) -> dict:
    """แปลงคีย์ที่ controller/publish เขียน (ระดับบนสุด) ให้เป็นฟิลด์ที่ Recipe รู้จัก

    controller ที่ publish มาเก็บ env ไว้ที่ `engine_env: "K=V K=V"` · parser ที่ `tool_parser` /
    `reasoning_parser` · แฟล็กเพิ่มที่ `extra_args` — ไม่ใช่ `env:`/`tool_calling:` แบบ catalog
    ของเดิมกรองด้วย "ชื่อฟิลด์ที่ Recipe มี" จึงทิ้งทั้งหมดเงียบ ๆ

    เคสจริง 2026-09-04: ucbye/Qwen3-Coder-Next-NVFP4-GB10 รันบน spark-head ได้ก็เพราะ
    `ENGINE_ENV=VLLM_NVFP4_GEMM_BACKEND=marlin …` สี่ตัว (อยู่ในสูตรที่ sync มาครบ) แต่แผนที่ hub
    สร้างให้ไม่มี env เลย → ptxas ตายตอน start บน image ที่ไม่มี FP4 kernel
    """
    out = dict(entry)
    if out.get("engine_env") and not out.get("env"):
        out["env"] = _parse_env(out["engine_env"])
    if out.get("tool_parser") and not out.get("tool_calling"):
        out["tool_calling"] = {"enabled": True, "parser": str(out["tool_parser"])}
    if out.get("reasoning_parser") and not out.get("reasoning"):
        out["reasoning"] = {"enabled": True, "parser": str(out["reasoning_parser"])}
    if out.get("extra_args") and not out.get("extra_flags"):
        out["extra_flags"] = _parse_flags(out["extra_args"])
    for name in _DICT_FIELDS:
        if not isinstance(out.get(name), dict):
            out.pop(name, None)
    return out


def _merge_entries(previous: dict, entry: dict) -> dict:
    """สูตรที่ดึงมา (entry) ทับของ catalog (previous) เฉพาะ *สิ่งที่มันพูดถึง* — ทีละคีย์ ไม่ใช่ทั้งก้อน

    สูตรที่ sync มาสร้างจาก header ของ controller จึงรู้แค่ "สั่งรันยังไง" (image · gpu_util · max_num_seqs
    · env) ส่วน catalog รู้ "โมเดลนี้ต้องการอะไร" (kv_cache_dtype ของ DeepSeek V4 · block-size 256 ·
    compilation_config PIECEWISE · env ปิด profiler) · เคสจริง 2026-09-04 บน hub: entry ของ DeepSeek ที่
    sync มามี serving แค่สองคีย์ แล้วทับทั้ง dict → ทุกอย่างที่ catalog รู้หายหมด แผนออกมา
    extra_flags=[] extra_env={} ทั้ง single และ stacked → vLLM ตาย "Expected 7 but got 8 arguments"
    """
    merged = dict(entry)
    for name in _DICT_FIELDS:
        merged[name] = {**(previous.get(name) or {}), **(entry.get(name) or {})}
    flags = list(previous.get("extra_flags") or []) + list(entry.get("extra_flags") or [])
    merged["extra_flags"] = list(dict.fromkeys(flags))
    # image_for ผูกกับ image — คงของ catalog ไว้เฉพาะเมื่อ image ยังเป็นตัวเดียวกัน (หรือ entry ไม่พูดถึง image)
    same_image = not entry.get("image") or entry.get("image") == previous.get("image")
    if "image_for" not in entry and same_image and previous.get("image_for"):
        merged["image_for"] = previous["image_for"]
    return merged


def load_catalog() -> list[Recipe]:
    """สูตรทั้งหมด — cache ตามไฟล์ที่อ่านจริง ไม่ใช่ cache ตลอดอายุ process

    ของเดิมเป็น lru_cache แบบไม่มี argument จึงจำคำตอบแรกไว้ตลอด: แก้
    recipes-synced.yaml แล้ว daemon ที่รันอยู่ไม่เห็น และเทสที่สลับ
    LMDS_CONFIG_DIR ก็อ่านของเทสก่อนหน้า ทำให้ผลเทสไม่คงที่
    """
    path = synced_path()
    try:
        stamp = path.stat().st_mtime
    except OSError:
        stamp = 0.0
    return _load_catalog_cached(str(path), stamp)


@lru_cache(maxsize=8)
def _load_catalog_cached(synced: str, stamp: float) -> list[Recipe]:
    """สูตรทั้งหมด = ที่มากับ LMDS + ที่ดึงมาจากรีโป controller ของทีม

    ของที่ดึงมาชนะของที่มากับตัวโปรแกรมเมื่อชนกัน — รีโปของทีมคือต้นทางที่รันจริง
    และอัปเดตบ่อยกว่า catalog ที่ฝังมากับเวอร์ชันที่ติดตั้งไว้
    """
    known = set(Recipe.__dataclass_fields__)
    merged: dict[str, dict] = {}
    for entry in [*_read(CATALOG_PATH), *_read(Path(synced))]:
        key = str(entry["match"]).lower()
        entry = _normalize(entry)
        previous = merged.get(key)
        if previous:
            # ของที่ดึงมาชนะทุกอย่างที่มันพูดถึง (ตามเดิม) แต่สิ่งที่มันเงียบไว้ต้องไม่หายไป —
            # สูตรที่ดึงมาสร้างจากสคริปต์ controller จึงบอกว่า "สั่งรันยังไง" ส่วน catalog
            # บอกว่า "โมเดลทำอะไรได้/ต้องการอะไร"
            #
            # ของจริง (1): สูตรที่ดึงมาของ ucbye/Qwen3-Coder-Next-NVFP4-GB10 เขียน
            # "tools (qwen3_coder)" ไว้ใน notes ซึ่งเป็นข้อความให้คนอ่าน ไม่มี
            # field tool_calling · พอทับทั้งก้อน สิ่งที่ catalog ยืนยันไว้ก็หาย
            # โมเดลจึง deploy ออกมาโดยไม่มี --enable-auto-tool-choice
            # ของจริง (2) 2026-09-04: serving/env ของ DeepSeek V4 หายทั้งชุด — ดู _merge_entries
            entry = _merge_entries(previous, entry)
        merged[key] = entry
    return [Recipe(**{k: v for k, v in entry.items() if k in known}) for entry in merged.values()]


# เดิม load_catalog เป็น lru_cache เอง จึงมี cache_clear ให้เรียก · การเปลี่ยน
# วิธี cache ข้างในไม่ควรทำให้ชื่อสาธารณะหายไป เทสที่ล้างแคชก่อนเขียนไฟล์เรียกผ่านชื่อนี้
load_catalog.cache_clear = _load_catalog_cached.cache_clear


def find_recipe(repo_id: str) -> Recipe | None:
    """สูตรของโมเดลนี้ — เทียบแบบไม่สนตัวพิมพ์ และยอมให้ `match` เป็น prefix

    prefix ใช้ครอบ variant ที่ผู้ให้บริการ quantize ออกมาหลายตัวจากฐานเดียวกัน
    ตัวที่ยาวกว่าชนะเสมอ เพื่อให้สูตรเฉพาะเจาะจงมาก่อนสูตรกว้าง
    """
    wanted = (repo_id or "").lower()
    if not wanted:
        return None
    matches = [r for r in load_catalog() if _prefix_matches(wanted, r.match.lower())]
    return max(matches, key=lambda r: len(r.match)) if matches else None


# ขอบของชื่อที่ prefix ต้องหยุด — `zai-org/GLM-4.7-Flash` ครอบ `…-Flash-NVFP4`/`…-Flash_v2` แต่ไม่ครอบ
# `…-Flashlight` ซึ่งเป็นคนละโมเดล (startswith ล้วนครอบแล้วยัด image/env/parser ของโมเดลอื่นให้เงียบ ๆ)
_NAME_BOUNDARY = "-_./"


def _prefix_matches(wanted: str, match: str) -> bool:
    if not match or not wanted.startswith(match):
        return False
    return len(wanted) == len(match) or wanted[len(match)] in _NAME_BOUNDARY
