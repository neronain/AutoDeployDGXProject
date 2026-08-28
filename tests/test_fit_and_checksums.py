"""สองอย่างที่ทำให้ deploy บนเครื่องที่มีของรันอยู่แล้วออกมาผิด

เคสจริง 2026-08-28 — deploy `wangzhang/gemma-4-31B-it-abliterated-GGUF` ลง msi-5
ที่มี Qwen3.8-27B (Q8_0, ctx 256K) รันอยู่ก่อน:

  1. fit คิดจากความจุเต็ม 114.5 GB แล้วตอบ "fits" จึงเลือก Q8_0 (32.6 GB) และ
     context สูงสุด 262,144 · เครื่องขึ้นไป 107/121 GB และทั้งสองโมเดลเหลือ 5-7 tok/s
  2. `EXPECTED_SHAS` ในทุก controller ว่างเปล่า เพราะตัวอ่าน Hub มองหาคีย์ `oid`
     ขณะที่ `/api/models/<id>?blobs=true` ส่ง `sha256` — verify-files จึงลดเหลือ
     "ขนาดตรงไหม" อย่างเดียวมาตลอด
"""

from __future__ import annotations

from lmds.fit.analyzer import _budget_gb
from lmds.fit.targets import PRESETS
from lmds.inspector.inspect import _sibling_files


# ---------------------------------------------------------------------------
# 1. งบหน่วยความจำต้องรู้ว่าเครื่องไม่ได้ว่าง
# ---------------------------------------------------------------------------
def test_memory_already_in_use_comes_off_the_budget():
    spark = PRESETS["dgx-spark-single"]
    empty, _ = _budget_gb(spark, "llamacpp")
    busy, notes = _budget_gb(spark, "llamacpp", reserved_gb=30.0)

    assert busy == empty - 30.0
    assert any("โมเดลอื่น" in n for n in notes), "ต้องบอกผู้ใช้ว่าหักเพราะอะไร"


def test_a_machine_that_is_already_full_gets_no_budget_not_a_negative_one():
    spark = PRESETS["dgx-spark-single"]
    budget, _ = _budget_gb(spark, "llamacpp", reserved_gb=999.0)
    assert budget == 0.0


def test_nothing_changes_when_the_machine_is_idle():
    """ค่าเริ่มต้นต้องเท่าพฤติกรรมเดิมเป๊ะ — ไม่งั้น target preset ทุกตัวขยับตาม"""
    spark = PRESETS["dgx-spark-single"]
    assert _budget_gb(spark, "llamacpp")[0] == _budget_gb(spark, "llamacpp", 0.0)[0]


# ---------------------------------------------------------------------------
# 2. checksum ต้องเดินทางมาจาก Hub จริง
# ---------------------------------------------------------------------------
SHA = "f70913d592f33fa383a3ea656222573fccda04d31404d477cad8624876ac1e95"


def test_the_checksum_from_the_models_endpoint_is_read():
    """`/api/models/<id>?blobs=true` ส่งคีย์ `sha256` — รูปแบบที่ LMDS ใช้จริง"""
    info = {"siblings": [
        {"rfilename": "model-Q8_0.gguf",
         "size": 32635674752,
         "lfs": {"sha256": SHA, "size": 32635674752, "pointerSize": 136}},
    ]}
    assert _sibling_files(info) == [("model-Q8_0.gguf", 32635674752, SHA)]


def test_the_checksum_from_the_tree_endpoint_still_works():
    """endpoint ของ file tree ส่งคีย์ `oid` — ของเดิมต้องไม่พัง"""
    info = {"siblings": [
        {"rfilename": "model-Q8_0.gguf", "size": 10, "lfs": {"oid": SHA, "size": 10}},
    ]}
    assert _sibling_files(info)[0][2] == SHA


def test_a_file_without_lfs_reports_no_checksum_rather_than_a_wrong_one():
    info = {"siblings": [{"rfilename": "config.json", "size": 42}]}
    assert _sibling_files(info) == [("config.json", 42, None)]


def test_a_non_string_checksum_is_refused():
    """ค่าเพี้ยนจาก Hub ต้องกลายเป็น 'ไม่มี checksum' ไม่ใช่หลุดไปอยู่ในสคริปต์"""
    info = {"siblings": [
        {"rfilename": "x.gguf", "size": 1, "lfs": {"sha256": {"unexpected": True}, "size": 1}},
    ]}
    assert _sibling_files(info)[0][2] is None
