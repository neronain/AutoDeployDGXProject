"""ไฟล์ mmproj ต้องถูกจำได้แม้ชื่อไม่ขึ้นต้นด้วย mmproj — ไม่งั้น vision หายเงียบ ๆ

เคสจริง 2026-09-04 ตอนเลือกไฟล์ในหน้าเว็บ: llmfan46/gemma-4-31B-it-uncensored-heretic-NVFP4-GGUF
มี "gemma-4-31B-it-uncensored-heretic-mmproj-BF16.gguf" (1.1 GB) · ตรวจแค่ startswith("mmproj") จึง
  1) โผล่ในรายการ "เลือกไฟล์ weights" ให้ผู้ใช้เลือกผิดได้
  2) has_mmproj=False → capabilities บอก "โหลดภาพไม่ได้" และ controller ไม่ได้ --mmproj
     ทั้งที่ Gemma-4 เป็นโมเดลภาพ

ส่วน mtp ยังต้องตรวจเฉพาะขึ้นต้น: "…-Native-MTP-Preserved-APEX-….gguf" คือ weights ที่เก็บหัว MTP
ไว้ในตัว ไม่ใช่ไฟล์ mtp แยก — จับกลางชื่อจะทิ้ง weights ตัวจริงออกจากรายการ
"""

from lmds.inspector.inspect import _group_gguf_variants

GIB = 1024**3
REAL_REPO = [
    ("gemma-4-31B-it-uncensored-heretic-mmproj-BF16.gguf", int(1.1 * GIB), "a" * 64),
    ("gemma-4-31B-it-uncensored-heretic-NVFP4-Q8_0.gguf", int(16.8 * GIB), "b" * 64),
    ("gemma-4-31B-it-uncensored-heretic-NVFP4-BF16.gguf", 18 * GIB, "c" * 64),
]


def _roles(files):
    return {v.filename: (v.is_mmproj, v.is_mtp) for v in _group_gguf_variants(files)}


def test_a_projector_named_after_the_model_is_still_a_projector():
    roles = _roles(REAL_REPO)
    assert roles["gemma-4-31B-it-uncensored-heretic-mmproj-BF16.gguf"] == (True, False)
    assert roles["gemma-4-31B-it-uncensored-heretic-NVFP4-Q8_0.gguf"] == (False, False)
    assert roles["gemma-4-31B-it-uncensored-heretic-NVFP4-BF16.gguf"] == (False, False)


def test_the_weights_picker_no_longer_offers_the_projector():
    """รายการที่หน้าเว็บให้เลือกคือ variant ที่ไม่ใช่ mmproj/mtp — ต้องเหลือแค่ weights สองไฟล์"""
    variants = _group_gguf_variants(REAL_REPO)
    weights = [v.filename for v in variants if not v.is_mmproj and not v.is_mtp]
    assert weights == [
        "gemma-4-31B-it-uncensored-heretic-NVFP4-BF16.gguf",
        "gemma-4-31B-it-uncensored-heretic-NVFP4-Q8_0.gguf",
    ]


def test_prefix_and_dotted_forms_still_work():
    roles = _roles([
        ("mmproj-BF16.gguf", GIB, None),
        ("vision/MMPROJ-F16.gguf", GIB, None),
        ("model.mmproj.gguf", GIB, None),
    ])
    assert all(is_mmproj for is_mmproj, _ in roles.values())


def test_mtp_inside_a_model_name_is_not_an_mtp_side_file():
    roles = _roles([
        ("Qwen3.6-35B-A3B-Native-MTP-Preserved-APEX-Q4_K_M.gguf", 20 * GIB, None),
        ("mtp-Qwen3.6-35B.gguf", GIB, None),
    ])
    assert roles["Qwen3.6-35B-A3B-Native-MTP-Preserved-APEX-Q4_K_M.gguf"] == (False, False)
    assert roles["mtp-Qwen3.6-35B.gguf"] == (False, True)
