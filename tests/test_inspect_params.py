"""จำนวนพารามิเตอร์ของ checkpoint ที่อัด 4-bit ต้องไม่ถูกรายงานเป็นครึ่งเดียว"""

from lmds.inspector.inspect import _params_of


def test_nvfp4_packed_u8_counts_two_params_per_byte():
    """ตัวเลขจริงจาก Hub ของ Sehyo/Qwen3.5-122B-A10B-NVFP4 (2026-09-03)

    Hub บอก total 71.2B ทั้งที่โมเดลคือ 122B · U8 58.68B คือ NVFP4 สองตัวต่อไบต์ และ
    F8_E4M3 7.34B คือ scale (117.36B / 16 = 7.335B ตรงเป๊ะ) ไม่ใช่พารามิเตอร์
    """
    info = {
        "tags": ["nvfp4", "compressed-tensors", "qwen3_5_moe"],
        "safetensors": {
            "total": 71_217_533_040,
            "parameters": {"F32": 74_112, "BF16": 7_725_676_784,
                           "F8_E4M3": 7_335_051_264, "U8": 58_680_410_112},
        },
    }
    params = _params_of(info)
    assert params == 58_680_410_112 * 2 + 7_725_676_784 + 74_112
    assert 118e9 < params < 130e9, "ต้องอยู่แถว 122B ไม่ใช่ 71B"


def test_unquantized_checkpoint_keeps_the_hub_total():
    """BF16 ล้วน — Hub นับถูกอยู่แล้ว ห้ามไปคูณอะไรเพิ่ม"""
    info = {"tags": ["transformers"], "safetensors": {"total": 70_553_706_496,
            "parameters": {"BF16": 70_553_706_496}}}
    assert _params_of(info) == 70_553_706_496


def test_u8_without_a_4bit_tag_is_left_alone():
    """U8 อาจเป็นอย่างอื่น (เช่น int8) — ไม่มีแท็กบอกว่าอัด 4-bit ก็อย่าเดา"""
    info = {"tags": ["int8"], "safetensors": {"total": 1_000, "parameters": {"U8": 1_000}}}
    assert _params_of(info) == 1_000


def test_missing_total_is_none():
    assert _params_of({"safetensors": {}}) is None
