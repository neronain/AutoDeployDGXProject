"""header ของ controller: อ่านไฟล์ GGUF ที่ทดสอบผ่านให้เจอด้วย

สูตรของโมเดล GGUF ที่ไม่บอกว่า *ไฟล์ quant ไหน* ที่รันผ่าน แทบไม่มีค่า — รีโปหนึ่ง
มี Q3/Q4/Q6/Q8 ปนกันสิบกว่าไฟล์ และตัวที่ทดสอบมาแล้วมีตัวเดียว

เคสจริง (2026-08-28): controller ที่ LMDS สร้างเก็บรายชื่อ shard ไว้ในอาร์เรย์
`MODEL_FILES=( "…gguf" )` แล้วตั้ง `MODEL_FILE="${MODEL_FILES[0]}"` ต่อ · ตัวอ่าน header
รับเฉพาะบรรทัด `KEY="ค่า"` และทิ้งค่าที่มี `$` ทิ้ง ผลคือสูตรของ GGUF **ทุกตัว** ใน
script-update ไม่มี `gguf_file` เลย ทั้งที่ไฟล์เขียนไว้ชัดเจนอยู่บรรทัดบน
"""

from __future__ import annotations

from lmds.recipes.controllers import parse_header, recipe_from_controller

CONTROLLER = '''#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-0.4.0}"
MODEL_LABEL="${MODEL_LABEL:-org/Some-Model-GGUF}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp (native build)}"
MODEL_ID="org/Some-Model-GGUF"
MODEL_REVISION="0123456789abcdef"

MODEL_FILES=(
  "Some-Model-UD-Q8_K_XL-00001-of-00003.gguf"
  "Some-Model-UD-Q8_K_XL-00002-of-00003.gguf"
  "Some-Model-UD-Q8_K_XL-00003-of-00003.gguf"
)
MODEL_FILE="${MODEL_FILES[0]}"

start() {
  # ตัวแปรในฟังก์ชันต้องไม่ถูกอ่านเป็น header
  INNER_FILES=(
    "ห้ามอ่านตัวนี้.gguf"
  )
}
'''


def test_the_first_shard_of_an_array_is_read():
    meta = parse_header(CONTROLLER)
    assert meta["MODEL_FILES"] == "Some-Model-UD-Q8_K_XL-00001-of-00003.gguf"


def test_the_recipe_says_which_quant_was_validated():
    recipe = recipe_from_controller("some-model-gguf-single.sh", CONTROLLER)
    assert recipe is not None
    assert recipe["gguf_file"] == "Some-Model-UD-Q8_K_XL-00001-of-00003.gguf"
    assert recipe["engine"] == "llamacpp"


def test_an_array_inside_a_function_is_still_ignored():
    """กติกาเดิมของ parser: ตัวแปรที่ย่อหน้า = อยู่ในฟังก์ชัน ไม่ใช่ค่าตั้งต้นของสคริปต์"""
    assert "INNER_FILES" not in parse_header(CONTROLLER)


def test_a_plain_model_file_still_wins():
    """controller รุ่นเก่าตั้ง MODEL_FILE ตรง ๆ — ต้องไม่ถูกอาร์เรย์แย่งที่"""
    text = CONTROLLER.replace('MODEL_FILE="${MODEL_FILES[0]}"', 'MODEL_FILE="chosen.gguf"')
    recipe = recipe_from_controller("x-single.sh", text)
    assert recipe["gguf_file"] == "chosen.gguf"


def test_a_non_gguf_value_is_not_reported_as_one():
    """อาร์เรย์อื่นที่บังเอิญอยู่ระดับบนสุดต้องไม่กลายเป็นไฟล์โมเดล"""
    text = CONTROLLER.replace("MODEL_FILES=(", "EXTRA_FLAGS=(").replace(
        'MODEL_FILE="${MODEL_FILES[0]}"', ""
    )
    recipe = recipe_from_controller("x-single.sh", text)
    assert "gguf_file" not in recipe
