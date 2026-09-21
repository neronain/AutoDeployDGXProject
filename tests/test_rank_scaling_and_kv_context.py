"""สองรูในโมเดลคำนวณ fit ที่ภาคสนามชี้ไว้ (docs/DGX-SPARK-VLLM-FIELD-NOTES.md §6 และ §11)

รูที่ 1 — **ก้อนที่จองนอกงบของ vLLM หายไปจาก `plan_kv_pin()` ทั้งก้อน**
  `analyzer._budget_gb()` หัก 3 GB ต่อเครื่องมาตลอด แต่ `fit/sizing.plan_kv_pin()` (ตัวที่พิมพ์ตาราง
  `lmds fit` และตัวที่ตอบว่า "รับได้กี่คน") ไม่เคยหักเลย และพอ node_count>1 ก็เอา **ยอดของทั้ง
  คลัสเตอร์** ไปเทียบกับความจุของ **เครื่องเดียว** แล้วปิดท้ายด้วยโน้ต

  **สิ่งที่เทสชุดนี้ *ไม่* ยืนยัน**: ภาคสนาม (FIELD-NOTES §6) อ้างว่าก้อนนั้นโตตามจำนวน rank
  (~24 GiB/rank ที่ TP8) · เราใส่โครงสร้างไว้แต่ **ปิดไว้** เพราะ (ก) หลักฐานเป็น README ไม่มี log
  (ข) ~24 GiB อยู่ใกล้ ~27 GiB/rank ของ COW break ซึ่งโตตาม *checkpoint* ไม่ใช่ตาม rank และ
  (ค) ฟลีตเรารัน TP4/TP8 ไม่ได้เลย (world size สูงสุดบน fabric เดียว = 2 · HCA-DUAL-TEST §3)
  ดู docs/GPU-UTIL-RANK-PROTOCOL.md §5 ซึ่งสั่งไว้ล่วงหน้าว่าห้ามใส่พจน์นั้นจนกว่า verdict = R

รูที่ 2 — **หารด้วยเลขที่อาจเป็นเลขปลอม**
  "GPU KV cache size: N tokens" ของ vLLM = max_concurrency × max_model_len (kv_cache_utils.py:2306)
  → N ผูกกับ --max-model-len ที่ตั้งไว้ ไม่ใช่ค่าคงที่ของโมเดล

ระดับหลักฐานของตัวเลขในไฟล์นี้:
  * **artifact-backed (ของเราเอง)** — log ของ Nemotron/Qwopus ใน tests/test_kv_sizing.py
    (probe อ่านอย่างเดียว 2026-09-07) · ค่า NCCL 3 GB/rank ที่ 2×Spark
  * **artifact-backed (ภายนอก)** — gpu-util 0.80 ที่ TP4 (Tech2wild boot 10, มี head log + args.json)
  * **คำกล่าวอ้างของผู้เขียน** — NCCL ~24 GiB/rank ที่ TP8 และ gpu-util 0.75 ที่ TP8 (README ล้วน)
    → **ไม่ถูกเข้ารหัสลงในสูตร** · เทสข้างล่างยืนยันว่ามันยังไม่มีผลกับตัวเลขใด ๆ
  * **ยังไม่ได้วัดเอง** — ทุกอย่างที่ ranks ≥ 3 · ฟลีตเราต่อได้สูงสุด 2 เครื่องต่อ fabric
"""

from __future__ import annotations

import pytest

from lmds.fit.analyzer import (
    CHECKPOINT_HOST_FOOTPRINT_GB,
    NCCL_COMM_BUFFER_GB_PER_PEER,
    STACKED_COMM_BUFFER_GB_PER_NODE,
    UNIFIED_OS_RESERVE_GB,
    gpu_util_ceiling,
    nccl_reserve_gb_per_rank,
)
from lmds.fit.sizing import measured_from_log, plan_kv_pin

GIB = 1024**3
SPARK = {"memory_model": "unified", "total_gb": 121.0, "free_gb": 4.0, "held_gb": 0.0}

# log จริงของ spark-head (Nemotron-3-Super-120B NVFP4, hybrid Mamba) — ก้อนเดียวกับ test_kv_sizing.py
NEMOTRON_LOG = """(EngineCore pid=175) INFO 09-07 03:05:51 [model_runner.py:408] Model loading took 69.62 GiB memory and 470.931960 seconds
(EngineCore pid=175) INFO 09-07 03:06:51 [gpu_worker.py:551] Initial free memory 114.28 GiB, reserved 6.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.
(EngineCore pid=175) INFO 09-07 03:06:51 [kv_cache_utils.py:2312] GPU KV cache size: 1,179,648 tokens, Maximum concurrency for 262,144 tokens per request: 4.50x"""


def nemotron(**over):
    info = {"slug": "nemotron", "engine": "vllm", "weight_bytes": 80317948856, "kv_bytes_per_token": 8192,
            "native_context": 262144, "context": 262144, "slots": 2,
            "extra_args": "--kv-cache-dtype fp8 --kv-cache-memory 6442450944", "running": True,
            "own_gb": 78.5, "measured": measured_from_log(NEMOTRON_LOG)}
    info.update(over)
    return info


def stacked(ranks: int, *, cluster_weight_gb: float, cluster_kv_per_token: int, slots: int = 6, **over):
    """โมเดล stacked สังเคราะห์ที่เลขกลม — ตรวจเลขด้วยมือได้ทุกบรรทัด

    ค่าใน MODEL_PROFILE เป็น **ยอดของทั้งคลัสเตอร์** (TP แบ่งเท่ากันทุก rank) เหมือนของจริง
    """
    info = {"slug": "stack", "engine": "vllm", "weight_bytes": int(cluster_weight_gb * GIB),
            "kv_bytes_per_token": cluster_kv_per_token, "native_context": 262144, "context": 262144,
            "slots": slots, "extra_args": "", "running": False, "node_count": ranks}
    info.update(over)
    return info


# ── รูที่ 1 · โครงสร้างพร้อม แต่พจน์ที่สเกลตาม rank ยังปิดอยู่ ─────────────────────────────
def test_the_rank_scaling_term_is_present_but_off_and_changes_nothing_today():
    """**หัวใจของการตัดสินใจนี้** — โครงสร้างอยู่ครบ ค่าเริ่มต้นให้ผลเท่าพฤติกรรมเดิมทุกจำนวน rank

    docs/GPU-UTIL-RANK-PROTOCOL.md §5 สั่งไว้ล่วงหน้า (ก่อนรู้ผลวัด) ว่าห้ามเปลี่ยน
    `STACKED_COMM_BUFFER_GB_PER_NODE` ให้สเกลตาม rank จนกว่า verdict = R เพราะถ้าผลเป็น
    C (checkpoint) หรือ H (host headroom) การเปลี่ยนนั้นจะทำให้ทุก stacked fit ผิด
    **ในแบบที่ดูน่าเชื่อถือกว่าเดิม** · เทสนี้คือด่านที่กันไม่ให้ใครเปิดมันโดยไม่ตั้งใจ
    """
    assert NCCL_COMM_BUFFER_GB_PER_PEER is None, (
        "เปิดพจน์ที่สเกลตาม rank ได้ต่อเมื่อ GPU-UTIL-RANK-PROTOCOL ให้ verdict = R เท่านั้น"
    )
    assert CHECKPOINT_HOST_FOOTPRINT_GB == 0.0, "พจน์ต่อ checkpoint ก็ยังไม่มีผลวัดเช่นกัน"
    # พฤติกรรมเดิมเป๊ะ: 0 ที่ 1 rank · คงที่ 3.0 ที่ 2 ขึ้นไป — **ไม่โตตามจำนวน rank**
    assert nccl_reserve_gb_per_rank(1) == 0.0
    for ranks in (2, 3, 4, 8):
        assert nccl_reserve_gb_per_rank(ranks) == STACKED_COMM_BUFFER_GB_PER_NODE == 3.0
    # ตัวเลขที่ภาคสนามอ้าง (~24 GiB/rank ที่ TP8) ต้อง **ยังไม่** อยู่ในสูตร
    assert nccl_reserve_gb_per_rank(8) < 24.0


def test_turning_the_term_on_gives_the_per_peer_shape_without_touching_one_or_two_ranks():
    """เมื่อผลวัดมาถึง เปิดได้ด้วยค่าคงที่ตัวเดียว — เทสนี้พิสูจน์ว่าสวิตช์ต่อสายไว้ถูกแล้ว
    และที่สำคัญ: เปิดแล้ว **1 กับ 2 rank ยังเท่าเดิม** (2 rank คือหมุดเดียวที่เราวัดเอง)"""
    import lmds.fit.analyzer as an

    before = {n: nccl_reserve_gb_per_rank(n) for n in (1, 2, 4, 8)}
    try:
        an.NCCL_COMM_BUFFER_GB_PER_PEER = 3.0      # ค่าสมมติ — แทน dx_rank_gib จาก fit-input.json
        assert nccl_reserve_gb_per_rank(1) == before[1] == 0.0
        assert nccl_reserve_gb_per_rank(2) == before[2] == 3.0   # หมุดที่วัดเอง ห้ามขยับ
        assert nccl_reserve_gb_per_rank(4) == 9.0 != before[4]
        assert nccl_reserve_gb_per_rank(8) == 21.0 != before[8]
    finally:
        an.NCCL_COMM_BUFFER_GB_PER_PEER = None
    assert {n: nccl_reserve_gb_per_rank(n) for n in (1, 2, 4, 8)} == before


def test_gpu_util_ceiling_is_a_formula_not_a_lookup_table():
    """§1 ของโปรโตคอล: **0.75/0.80 ไม่ใช่ค่าคงที่ของฮาร์ดแวร์ มันคือผลหาร**
    `gpu_util_max = (MemTotal − X − safety) / MemTotal` · ฟังก์ชันนี้จึงต้องเป็นสูตร ไม่ใช่ตาราง

    **และด้วย X วันนี้ เพดานยังไม่ลดตามจำนวน rank** — 0.85 เท่ากันหมดที่ 2 ขึ้นไป
    นั่นคือสิ่งที่หลักฐานของเรารองรับจริง · เทสนี้จะเปลี่ยนก็ต่อเมื่อ X ถูกวัด ไม่ใช่ก่อนหน้านั้น
    """
    assert gpu_util_ceiling(128.0, 1) == pytest.approx(0.90)
    for ranks in (2, 4, 8):
        assert gpu_util_ceiling(128.0, ranks) == pytest.approx(0.85)
    # 0.70/0.75 ที่ภาคสนามอ้างที่ TP8 **ยังไม่ถูกเข้ารหัส** — เราไม่แกล้งรู้
    assert gpu_util_ceiling(128.0, 8) != pytest.approx(0.75)
    # หน่วยไม่ทำให้คำตอบแกว่ง: 128 GB decimal กับ 121.7 GiB ที่ free -g เห็น ได้ขั้นเดียวกัน
    for ranks in (1, 2, 4, 8):
        assert gpu_util_ceiling(121.7, ranks) == pytest.approx(gpu_util_ceiling(128.0, ranks))
    # สมการต้นทางต้องเป็นจริงที่เพดานนั้นเสมอ ไม่ว่า X จะเป็นเท่าไร
    for ranks in (1, 2, 4, 8):
        used = gpu_util_ceiling(128.0, ranks) * 128.0 + nccl_reserve_gb_per_rank(ranks) + UNIFIED_OS_RESERVE_GB
        assert used <= 128.0
    # และเมื่อ X ใหญ่ขึ้น เพดานต้องลดลงเอง โดยไม่ต้องแก้ฟังก์ชัน
    import lmds.fit.analyzer as an
    try:
        an.NCCL_COMM_BUFFER_GB_PER_PEER = 3.0
        assert gpu_util_ceiling(128.0, 8) == pytest.approx(0.70)
    finally:
        an.NCCL_COMM_BUFFER_GB_PER_PEER = None


def test_a_single_machine_is_completely_unchanged_by_the_rank_term():
    """เครื่องเดียว = ไม่มี peer = ไม่มี NCCL — ตารางเดิมทุกช่องต้องเท่าเดิม ไม่มีแถวใหม่โผล่"""
    p = plan_kv_pin(nemotron(), {**SPARK, "held_gb": 98.0, "others": [{"slug": "gemma4", "gb": 19.5}]})
    assert p["ranks"] == 1 and p["nccl_reserve_gb"] == 0.0 and p["per_rank"] is False
    assert [s["kind"] for s in p["stack"]] == ["os", "other", "weights", "overhead", "kv", "free"]
    # ตัวเลขชุดเดิมจาก test_kv_sizing.py — ต้องไม่ขยับแม้แต่ตำแหน่งเดียว
    assert p["kv_per_token_bytes"] == 5461 and p["kv_per_request_gb"] == 1.33
    assert p["kv_pin_gb"] == 3.5 and p["ram_needed_gb"] == 76.1 and p["gpu_util_equivalent"] == 0.65


def test_stacked_is_per_rank_arithmetic_not_a_note_that_says_do_it_yourself():
    """เดิม: weights ของทั้งคลัสเตอร์ถูกเอาไปลบกับความจุของ **เครื่องเดียว** แล้วปิดท้ายด้วยโน้ต
    ผลคือ stacked ตอบ "ไม่พอ" แทบทุกครั้งทั้งที่รันได้จริง · ตอนนี้ต้องแบ่งต่อ rank จริง ๆ

    เลขกลมให้ตรวจด้วยมือ (2 rank · คลัสเตอร์ weights 180 GiB · KV 8192 B/token ต่อคลัสเตอร์):
      weights 180 ÷ 2 = 90 · KV 8192 ÷ 2 = 4096 B/token → 1 GiB ต่อคำขอ 262,144 ต่อ rank
      pin = 6 slots × 1.0 × 1.2 = 7.2 → ปัดขึ้นครึ่ง GiB = 7.5
      ram/rank = 90 + overhead 3 + NCCL 3 + KV 7.5 = 103.5 · usable = 121 − 12 = 109 → เหลือ 5.5
    """
    p = plan_kv_pin(stacked(2, cluster_weight_gb=180.0, cluster_kv_per_token=8192), SPARK)
    assert p["ranks"] == 2 and p["per_rank"] is True
    assert p["cluster_weights_gb"] == 180.0 and p["weights_gb"] == 90.0
    assert p["kv_per_token_bytes"] == 4096 and p["kv_per_request_gb"] == 1.0
    assert p["nccl_reserve_gb"] == 3.0
    assert p["kv_pin_gb"] == 7.5
    assert p["ram_needed_gb"] == 103.5 and p["usable_gb"] == 109.0
    assert p["ram_after_gb"] == 5.5 and p["fits"] is True and p["verdict"] == "fits"
    assert [s["kind"] for s in p["stack"]] == ["os", "weights", "overhead", "nccl", "kv", "free"]
    assert any("ต่อ rank" in n for n in p["notes"])


def test_a_stacked_table_beyond_two_machines_says_it_is_out_of_measured_range():
    """3 เครื่องขึ้นไป = เราอ้างนอกช่วงที่วัด (valid_for_node_counts = [1, 2])

    ตัวเลขยังออกมาให้ใช้ได้ แต่ **ต้องติดป้ายว่าอาจต่ำกว่าจริง** ไม่ใช่ปล่อยให้ดูแน่นอนเท่ากับ
    ตัวเลขที่ 2 เครื่อง · นี่คือวินัยเดียวกับที่ FIELD-NOTES แยก artifact-backed ออกจากคำกล่าวอ้าง
    """
    two = plan_kv_pin(stacked(2, cluster_weight_gb=180.0, cluster_kv_per_token=8192), SPARK)
    four = plan_kv_pin(stacked(4, cluster_weight_gb=360.0, cluster_kv_per_token=16384), SPARK)
    assert not any("ยังไม่มีผลวัด" in n for n in two["notes"])
    assert any("ยังไม่มีผลวัด" in n and "ต่ำกว่าจริง" in n for n in four["notes"])
    # ต่อ rank เท่ากันเป๊ะ — เพราะ X ยังคงที่ · **ถ้าวันหนึ่งบรรทัดนี้ล้ม แปลว่ามีคนเปิดพจน์ rank แล้ว**
    assert two["nccl_reserve_gb"] == four["nccl_reserve_gb"] == 3.0
    assert two["weights_gb"] == four["weights_gb"] == 90.0
    assert two["suggested_slots_max"] == four["suggested_slots_max"]


def test_the_concurrency_answer_changes_only_from_the_unit_fix_not_from_new_guesses():
    """"รับผู้ใช้พร้อมกันกี่คน" ของ stacked — ตัวเลขที่ขยับวันนี้มาจาก **การแก้หน่วย** ล้วน ๆ
    (ยอดคลัสเตอร์ → ต่อ rank + หักก้อนที่ `analyzer` หักอยู่แล้ว) ไม่ได้มาจากสมมติฐานใหม่

    ต่อ rank: weights 90 · KV 1.0 GiB ต่อคำขอ 262,144 · overhead 3 · NCCL 3 · usable 109
      room = 109 − 90 − 3 − 3 = 13.0 → 13.0 ÷ 1.2 = 10 คำขอเต็ม context ต่อ rank
    ของเดิมเอา weights 180 (ทั้งคลัสเตอร์) มาลบกับ 109 ของเครื่องเดียว → ติดลบ → 0 เสมอ
    """
    p = plan_kv_pin(stacked(2, cluster_weight_gb=180.0, cluster_kv_per_token=8192, slots=1), SPARK)
    assert p["suggested_slots_max"] == 10
    # ก่อนแก้: room = 109 − 180 − 3 = ติดลบ → ตอบ 0 ช่องทั้งที่ recipe นี้รันอยู่จริงบน 2×Spark
    assert (109.0 - 180.0 - 3.0) < 0


# ── รูที่ 2 · KV ที่วัดได้ผูกกับ context ที่วัด ────────────────────────────────────────────
def test_the_measured_kv_rate_is_identical_at_the_context_it_was_measured_at():
    """**โค้ดเดิมไม่ได้ผิดตรงนี้** — ที่ context เดิม max_model_len ตัดกันพอดี:
        pool ÷ (conc × ctx) × ctx  ≡  pool ÷ conc  = GiB ต่อคำขอจริง
    สูตรใหม่จึงต้องให้คำตอบ **เท่าเดิมเป๊ะ** ที่ ctx เดิม ไม่งั้นแปลว่าเราไปทำของที่ถูกอยู่ให้พัง
    """
    p = plan_kv_pin(nemotron(), SPARK)
    assert p["kv_measured_context"] == 262144 and p["kv_source"] == "measured"
    assert p["kv_per_request_gb"] == 1.33 == round(6.0 / 4.50, 2)   # pool ÷ concurrency
    assert p["kv_per_token_bytes"] == 5461                          # pool × GIB ÷ 1,179,648
    assert not any("คิดใหม่ที่" in n for n in p["notes"])


def test_asking_for_a_shorter_context_no_longer_shrinks_kv_linearly():
    """จุดที่โค้ดเดิม **ผิดจริง** — `lmds fit --context 65536` บนโมเดลที่ log วัดไว้ที่ 262,144

    โมเดล hybrid (Mamba/SSM) มี state ที่ไม่โตตาม context ปนอยู่ใน KV ต่อคำขอ:
        KV(ctx) = a × ctx + b   ·   a = 4,096 B/token จาก config (fp8) · b = 1.333 − 1.000 = 0.333 GiB
    เดิมคูณอัตราเฉลี่ย 5,461 B/token ลงไปตรง ๆ → 0.333 GiB ต่อคำขอ ซึ่ง **ต่ำกว่าจริง 75%**
    และแปลว่าเราจะบอกลูกค้าว่ารับได้ราว 3 เท่าของที่รับได้จริง
    """
    p = plan_kv_pin(nemotron(), SPARK, context=65536)
    naive = 5461 * 65536 / GIB                      # สิ่งที่โค้ดเดิมตอบ
    assert round(naive, 2) == 0.33
    assert p["kv_state_gb"] == 0.33                 # b ที่ถอดออกมาได้
    assert p["kv_per_request_gb"] == 0.58           # = 4096×65536/GIB + 0.333
    assert p["kv_per_request_gb"] / naive == pytest.approx(1.75, abs=0.02)
    assert any("คิดใหม่ที่" in n and "262,144" in n for n in p["notes"])
    # ทางกลับกัน: ขอ context ยาวกว่าที่วัด ผลต้อง *น้อยลง* กว่าการคูณเชิงเส้น (b ไม่โตตาม ctx)
    long = plan_kv_pin(nemotron(native_context=524288), SPARK, context=524288)
    assert long["kv_per_request_gb"] < 5461 * 524288 / GIB


def test_a_log_without_the_concurrency_line_says_which_context_it_assumed():
    """log เก่าที่มีแต่ "GPU KV cache size" — ยังใช้ได้ แต่ต้องบอกว่าเดา ctx จากค่าที่ตั้งอยู่
    ไม่ใช่เงียบ ๆ แล้วให้คนอ่านคิดว่าเป็นค่าที่วัดมา"""
    old_log = ("Model loading took 69.62 GiB memory and 1 seconds\n"
               "reserved 6.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config\n"
               "GPU KV cache size: 1,179,648 tokens")
    p = plan_kv_pin(nemotron(measured=measured_from_log(old_log)), SPARK)
    assert p["kv_measured_context"] == 262144          # = context ที่ตั้งอยู่
    assert p["kv_per_request_gb"] == 1.33              # ผลเท่ากับตอนมีบรรทัด concurrency
    assert any("ไม่มีบรรทัด 'Maximum concurrency'" in n for n in p["notes"])


def test_without_kv_dims_in_the_profile_we_say_the_number_may_be_low():
    """Qwopus: MODEL_PROFILE มี kv_bytes_per_token: null → แยก a กับ b ไม่ออก
    ยังต้องตอบได้ (ค่าวัดดีกว่าไม่มีอะไร) แต่ห้ามเงียบว่ามันคือการคาดนอกช่วงที่วัด"""
    qwopus_log = ("Model loading took 71.32 GiB memory and 1 seconds\n"
                  "reserved 12.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config\n"
                  "GPU KV cache size: 508,031 tokens\n"
                  "Maximum concurrency for 262,144 tokens per request: 1.94x")
    info = {"slug": "qwopus", "engine": "vllm", "weight_bytes": 81488272728, "kv_bytes_per_token": None,
            "native_context": 262144, "slots": 3, "extra_args": "--kv-cache-memory 12884901888",
            "running": True, "own_gb": 84.2, "measured": measured_from_log(qwopus_log)}
    same = plan_kv_pin(dict(info), {**SPARK, "free_gb": 6.0, "held_gb": 84.2}, context=262144)
    assert same["kv_source"] == "measured-linear" and same["kv_per_request_gb"] == 6.19
    assert not any("อาจต่ำกว่าจริง" in n for n in same["notes"])
    shorter = plan_kv_pin(dict(info), {**SPARK, "free_gb": 6.0, "held_gb": 84.2}, context=65536)
    assert any("อาจต่ำกว่าจริง" in n for n in shorter["notes"])


# ── การป้องกัน recipe ที่พิสูจน์บนฮาร์ดแวร์แล้ว ────────────────────────────────────────────
def test_a_stacked_recipe_pin_is_still_never_recomputed_at_deploy_time():
    """`catalog.yaml` มี `kv_cache_memory` ของ recipe ที่รันผ่านจริงบน 2×Spark — ด่านที่กันไม่ให้
    สูตรใหม่ไปทับมันคือ `sizing_for_plan()` ที่คืน None ทันทีเมื่อ node_count != 1 · ต้องยังคืน None"""
    from types import SimpleNamespace

    from lmds.fit.sizing import sizing_for_plan
    from lmds.hardware import MemoryModel

    fit = SimpleNamespace(memory_model=MemoryModel.UNIFIED, node_count=2, capacity_gb=256.0,
                          kv_bytes_per_token=8192, weights_gb=180.0)
    plan = SimpleNamespace(runtime=SimpleNamespace(engine="vllm"),
                           serving=SimpleNamespace(context=262144, max_num_seqs=6, kv_cache_dtype="auto",
                                                   extra_flags=["--kv-cache-memory 3758096384"]))
    assert sizing_for_plan(fit, plan) is None
