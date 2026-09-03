"""เติมค่า settings ให้ตามโมเดล — สำหรับ bundle ที่มีอยู่แล้ว (ปุ่ม "เติมให้ตามโมเดล" / `lmds set --auto`)

ลำดับความน่าเชื่อถือ: สูตรที่รันผ่านจริง (recipe) > กฎตระกูล (families) > กฎฮาร์ดแวร์ (NVFP4/SM121)
คืนแค่ *ข้อเสนอ* พร้อมที่มาของแต่ละค่า — ผู้เรียกตัดสินใจเองว่าจะบันทึกไหม
"""

from __future__ import annotations

from lmds.brain.families import nvfp4_on_sm121, parsers_for
from lmds.recipes import find_recipe


def suggest_settings(model_id: str, engine: str, *, architecture: str = "",
                     quantization: str = "", memory_model: str = "") -> dict:
    """→ {"values": {field: value}, "sources": {field: ที่มา}, "notes": [...]}

    field ใช้ชื่อเดียวกับ `bundle_settings.FIELDS` จึงส่งเข้า write()/PUT /settings ได้ตรง ๆ
    """
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []

    def put(field: str, value, why: str) -> None:
        if value in (None, "") or field in values:
            return
        values[field] = str(value)
        sources[field] = why

    recipe = find_recipe(model_id)
    if recipe is not None and (not recipe.engine or recipe.engine == engine):
        label = f"สูตรที่รันผ่านจริง: {recipe.label or recipe.match}" + (
            f" ({recipe.validated_on})" if recipe.validated_on else "")
        put("tool_parser", (recipe.tool_calling or {}).get("parser"), label)
        put("reasoning_parser", (recipe.reasoning or {}).get("parser"), label)
        if recipe.image and recipe.image_applies_to(memory_model):
            put("image", recipe.image, label)
        if recipe.env:
            put("engine_env", " ".join(f"{k}={v}" for k, v in recipe.env.items()), label)
    elif recipe is not None:
        notes.append(f"มีสูตรของโมเดลนี้แต่เป็น engine {recipe.engine} ไม่ใช่ {engine} — ไม่เอาค่ามาใช้")

    choice = parsers_for(model_id, architecture, engine)
    put("tool_parser", choice.tool, f"กฎตระกูล: {choice.why}")
    put("reasoning_parser", choice.reasoning, f"กฎตระกูล: {choice.why}")

    hint = nvfp4_on_sm121(model_id, quantization, engine, memory_model)
    put("image", hint.image, hint.why)
    put("engine_env", hint.engine_env, hint.why)

    if not values:
        notes.append("ไม่รู้จักตระกูลนี้และไม่มีสูตรที่รันผ่าน — ไม่เดา · ดูคำเตือนใน plan หรือถามผู้ช่วย")
    return {"values": values, "sources": sources, "notes": notes}
