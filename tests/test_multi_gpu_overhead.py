"""overhead ของ engine เกิด **ต่อ GPU** และไฟล์คู่ (mmproj/MTP) ก็กิน GPU เหมือน weight

เคสจริงที่ทำให้ต้องมีไฟล์นี้ — วัดบนเครื่อง BesthaiAi 2026-09-20:

    3× RTX 3060 12 GB (รวม 36 GB) · 24 cores · RAM 125 GB · docker
    Qwen3.6-35B-A3B MoE · GGUF Q4_K 20.7 GB + mmproj f16 0.9 GB · llama.cpp
    context 262,144 · 1 slot · --n-gpu-layers 999 · --spec-type draft-mtp

    `lmds fit` ทำนาย   weights 20.2 + overhead 1.5 + KV 5.5   = 27.2 GB
    nvidia-smi วัดจริง  11475 + 10143 + 10995 MiB = 32,613 MiB = 31.8 GiB
                                                       ต่ำไป 4.6 GB

เครื่องนี้ 36 GB จึงยัง "ใส่ได้" แต่บนเครื่องที่ตึงกว่า ช่องว่างขนาดนี้คือระยะห่างระหว่าง
แผนที่บอกว่า fits กับ start ที่ OOM

ระบุสาเหตุได้สองข้อ (คนละบั๊ก แก้คนละที่ · เทสในไฟล์นี้ตรึงไว้แยกกัน) เหลืออีกส่วนที่ยังระบุไม่ได้:

  3.0 GB  overhead ถูกคิดครั้งเดียวแทนที่จะคิดสามใบ · `fit/analyzer._budget_gb()` คูณ
          gpu_count มาตั้งแต่ต้นและ `web/memory.py` ก็คูณ แต่ `fit/sizing.plan_kv_pin()`
          (ตัวที่พิมพ์ตาราง `lmds fit`) ไม่เคยได้รับจำนวน GPU เลย — `host_info_from()`
          ยุบ `gpus` เหลือแค่ผลรวม VRAM · **ค่าคงที่ 1.5 ไม่ได้ถูกแก้ หน่วยของมันต่างหากที่ผิด**
  0.9 GB  mmproj ไม่เคยถูกนับใน weights · `model.weight_bytes` = ขนาดไฟล์ GGUF ที่เลือก
          ไฟล์เดียว ส่วน projector เป็นไฟล์แยกที่ controller โหลดขึ้น GPU ด้วย

  0.7 GB  เหลือแยกไม่ออกจากการวัดครั้งเดียว — **ห้ามยัดเข้าค่าคงที่ตัวไหน** (ดู
          test_the_remaining_gap_is_left_unexplained ท้ายไฟล์)

รูปของการแก้ตรงกับหลักฐานอีกทางหนึ่งที่จดไว้แล้ว: docs/DGX-SPARK-VLLM-FIELD-NOTES.md §6 และ
docs/UPGRADE-2026-09.md §1.5 — NCCL 8 ทางจอง ~24 GiB **ต่อ rank** นอกงบของ vLLM จึงต้องลด
gpu-util จาก 0.80 ที่ TP4 เหลือ 0.75 ที่ TP8 · ตัวเลขนั้นเป็นคำกล่าวอ้างของผู้เขียน ไม่มี log
ดิบ จึงใช้ยืนยันได้แค่ "โตตามจำนวน device" ไม่ใช่เอาตัวเลขมาใช้
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmds.fit.analyzer import (
    GIB,
    LLAMACPP_OVERHEAD_GB,
    LLAMACPP_OVERHEAD_GB_PER_GPU,
    VLLM_OVERHEAD_GB_PER_GPU,
    _budget_gb,
    analyze,
    companion_weight_bytes,
)
from lmds.fit.sizing import (
    LLAMACPP_OVERHEAD_GB_PER_GPU as SIZING_LLAMACPP_PER_GPU,
)
from lmds.fit.sizing import (
    VLLM_RUNTIME_OVERHEAD_GB_PER_GPU,
    plan_kv_pin,
)
from lmds.fit.targets import PRESETS, TargetSpec
from lmds.fleet.sizing import host_info_from, model_info_for
from lmds.hardware import MemoryModel
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

# ── ตัวเลขจากเครื่องจริง (2026-09-20) ────────────────────────────────────────────
BESTHAIAI_GPUS = 3
BESTHAIAI_MIB = (11475, 10143, 10995)           # nvidia-smi ต่อใบ ตอน serve อยู่
MEASURED_GIB = sum(BESTHAIAI_MIB) / 1024        # 31.849…
WEIGHTS_BYTES = int(20.2 * GIB)                 # ไฟล์ GGUF ที่เลือก ตามที่ profile เก็บไว้
MMPROJ_BYTES = int(0.9 * GIB)                   # mmproj f16 — คนละไฟล์ ไม่อยู่ใน weight_bytes
CONTEXT = 262_144
KV_PER_TOKEN = 22_528                           # × 262,144 = 5.5 GiB พอดี (ตรงกับที่ fit รายงาน)


def besthaiai_model(**overrides) -> dict:
    info = {
        "slug": "qwen36-35b-a3b", "engine": "llamacpp",
        "weight_bytes": WEIGHTS_BYTES, "companion_weight_bytes": MMPROJ_BYTES,
        "kv_bytes_per_token": KV_PER_TOKEN, "native_context": CONTEXT,
        "context": CONTEXT, "slots": 1,
        "extra_args": "--n-gpu-layers 999 --spec-type draft-mtp",
        "running": True, "node_count": 1,
    }
    info.update(overrides)
    return info


def besthaiai_host(**overrides) -> dict:
    host = {"memory_model": "discrete", "total_gb": 36.0, "free_gb": 3.4,
            "held_gb": MEASURED_GIB, "gpu_count": BESTHAIAI_GPUS, "others": []}
    host.update(overrides)
    return host


# ── เทสที่ตรึงการวัดจริงไว้ ────────────────────────────────────────────────────────
def test_besthaiai_prediction_lands_within_a_gigabyte_of_the_measurement():
    """ตัวเลขทั้งชุดของ 2026-09-20 — เทสนี้คือสิ่งที่จับได้ถ้าสูตรเลื่อนออกจากความจริง

    ห้ามผ่อนขอบเขตนี้เพื่อให้การแก้ครั้งหน้าผ่าน · ถ้ามันเริ่มไม่ผ่านแปลว่าสูตรเปลี่ยนไป
    ในทางที่ขัดกับเครื่องจริง — ต้องไปวัดใหม่ ไม่ใช่มาขยายตัวเลขตรงนี้
    """
    plan = plan_kv_pin(besthaiai_model(), besthaiai_host())

    assert plan["weights_gb"] == pytest.approx(21.1, abs=0.05)   # 20.2 ไฟล์ GGUF + 0.9 mmproj
    assert plan["overhead_gb"] == pytest.approx(4.5, abs=0.05)   # 1.5 ต่อใบ × 3 ใบ
    assert plan["kv_gb"] == pytest.approx(5.5, abs=0.05)
    assert plan["ram_needed_gb"] == pytest.approx(31.1, abs=0.05)

    # ต่ำกว่าที่วัดได้ แต่ต่ำแบบรู้ตัวและอธิบายได้ ไม่ใช่ 4.6 GB แบบเดิม
    short = MEASURED_GIB - plan["ram_needed_gb"]
    assert 0 < short < 1.0, f"เหลือช่องว่าง {short:.2f} GB — เดิม 4.65"


def test_the_two_causes_are_independent_and_each_one_is_worth_what_it_says():
    """แยกส่วนของช่องว่าง 4.6 GB ออกเป็นข้อ ๆ — ไม่ใช่ก้อนเดียวที่ถูก fit ด้วยเลขเดียว"""
    before = plan_kv_pin(besthaiai_model(companion_weight_bytes=0),
                         besthaiai_host(gpu_count=1))
    per_device_only = plan_kv_pin(besthaiai_model(companion_weight_bytes=0), besthaiai_host())
    both = plan_kv_pin(besthaiai_model(), besthaiai_host())

    assert before["ram_needed_gb"] == pytest.approx(27.2, abs=0.05)      # ที่ `lmds fit` เคยพิมพ์
    assert per_device_only["ram_needed_gb"] - before["ram_needed_gb"] == pytest.approx(3.0, abs=0.05)
    assert both["ram_needed_gb"] - per_device_only["ram_needed_gb"] == pytest.approx(0.9, abs=0.05)


def test_the_gap_used_to_be_blamed_on_a_process_that_does_not_exist():
    """หลักฐานอิสระว่า 4.6 GB นั้นเป็นของโมเดลตัวนี้เอง ไม่ใช่ของอย่างอื่นบนเครื่อง

    `plan_kv_pin` จับคู่ยอดจาก nvidia-smi กับสิ่งที่มันคิดว่าโมเดลใช้ · ส่วนต่างที่จับคู่ไม่ได้
    เกิน 1 GB จะถูกตั้งชื่อว่า "other GPU processes" แล้วโผล่ในตาราง · บน BesthaiAi
    ที่โมเดลรันอยู่ตัวเดียว ตารางเดิมจึงรายงานว่ามีอะไรอีก 4.6 GB ถืออยู่ — ซึ่งไม่มีอยู่จริง
    แถมยังชวนให้ผู้ใช้ไปไล่หาโปรเซสผี · แก้แล้วส่วนต่างลงมา 0.7 GB จนไม่มีแถวนั้นอีก
    """
    running_host = besthaiai_host(held_gb=MEASURED_GIB)
    before = plan_kv_pin(besthaiai_model(companion_weight_bytes=0),
                         {**running_host, "gpu_count": 1})
    after = plan_kv_pin(besthaiai_model(), running_host)

    assert [o["slug"] for o in before["others"]] == ["other GPU processes"]
    assert before["others_running_gb"] == pytest.approx(4.6, abs=0.1)
    assert after["others"] == []
    assert after["others_running_gb"] < 1.0


def test_the_remaining_gap_is_left_unexplained():
    """~0.7 GB ที่เหลือ **ต้องไม่** ถูกยัดเข้าค่าคงที่ตัวไหน — การวัดครั้งเดียวแยกมันไม่ออก

    ผู้สมัครที่แยกไม่ออกจากข้อมูลชุดนี้: compute buffer ต่อใบที่ใหญ่กว่าที่คิด · allocator slack ·
    KV ที่ llama.cpp ปัดเป็น block · ค่า kv_bytes_per_token ใน profile เองที่อาจคลาดเคลื่อน
    จะตัดสินได้ต้องมี log ของ llama-server (`CUDAn compute buffer size`, `KV self size`) ต่อใบ
    ซึ่งการวัดครั้งนี้ไม่ได้เก็บไว้

    เทสนี้จึงล็อก **ค่าคงที่** ไว้ที่ของเดิม: ถ้ามีใครขยับ 1.5 → 2.0 เพื่อให้ยอดลงตัวพอดี
    เทสนี้จะล้มและบังคับให้อธิบายที่มาก่อน
    """
    assert LLAMACPP_OVERHEAD_GB_PER_GPU == 1.5
    assert VLLM_OVERHEAD_GB_PER_GPU == 2.5
    assert VLLM_RUNTIME_OVERHEAD_GB_PER_GPU == 3.0
    # ชื่อเดิมที่ web/deploy.py กับ web/memory.py ยัง import อยู่ ต้องเป็นค่าเดียวกันเสมอ
    assert LLAMACPP_OVERHEAD_GB == LLAMACPP_OVERHEAD_GB_PER_GPU
    assert SIZING_LLAMACPP_PER_GPU == LLAMACPP_OVERHEAD_GB_PER_GPU


# ── overhead ต่อ device ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("devices", [1, 2, 3, 4])
def test_overhead_grows_with_the_number_of_devices(devices):
    plan = plan_kv_pin(besthaiai_model(), besthaiai_host(gpu_count=devices, total_gb=12.0 * devices))
    assert plan["gpu_count"] == devices
    assert plan["overhead_per_gpu_gb"] == LLAMACPP_OVERHEAD_GB_PER_GPU
    assert plan["overhead_gb"] == pytest.approx(LLAMACPP_OVERHEAD_GB_PER_GPU * devices, abs=0.05)


def test_vllm_overhead_is_per_device_too():
    info = {"slug": "x", "engine": "vllm", "weight_bytes": 40 * GIB, "kv_bytes_per_token": 4096,
            "native_context": 32768, "slots": 1, "running": False, "node_count": 1}
    one = plan_kv_pin(info, {"memory_model": "discrete", "total_gb": 96.0, "gpu_count": 1})
    four = plan_kv_pin(info, {"memory_model": "discrete", "total_gb": 96.0, "gpu_count": 4})
    assert one["overhead_gb"] == pytest.approx(VLLM_RUNTIME_OVERHEAD_GB_PER_GPU, abs=0.05)
    assert four["overhead_gb"] == pytest.approx(VLLM_RUNTIME_OVERHEAD_GB_PER_GPU * 4, abs=0.05)


def test_single_gpu_hosts_are_untouched():
    """DGX Spark = GPU ใบเดียวต่อเครื่อง · ตัวเลขที่ตรึงไว้ใน test_kv_sizing.py ต้องไม่ขยับ"""
    spark = {"memory_model": "unified", "total_gb": 121.0, "free_gb": 4.0, "held_gb": 98.0}
    info = {"slug": "nemotron", "engine": "vllm", "weight_bytes": 80_317_948_856,
            "kv_bytes_per_token": 8192, "native_context": 262144, "slots": 4,
            "running": False, "node_count": 1}
    assert plan_kv_pin(info, spark)["overhead_gb"] == plan_kv_pin(info, {**spark, "gpu_count": 1})["overhead_gb"]
    # ไม่ส่ง gpu_count มาเลย (ผู้เรียกเก่า / sizing_for_plan) = 1 ใบ ไม่ใช่ 0
    assert plan_kv_pin(info, spark)["gpu_count"] == 1
    assert plan_kv_pin(info, {**spark, "gpu_count": 0})["gpu_count"] == 1


def test_a_multi_gpu_plan_says_where_the_overhead_came_from():
    """ตัวเลขที่โตขึ้นเฉย ๆ โดยไม่บอกเหตุผล = คนอ่านเดาว่าโปรแกรมพัง"""
    plan = plan_kv_pin(besthaiai_model(), besthaiai_host())
    assert any("3 ใบ" in n for n in plan["notes"])
    assert any("mmproj" in n for n in plan["notes"])
    overhead_row = next(row for row in plan["stack"] if row["kind"] == "overhead")
    assert "3" in overhead_row["label"]


def test_analyzer_and_plan_kv_pin_agree_about_overhead_on_the_same_machine():
    """สองสูตรนี้เคยตอบไม่ตรงกันบนเครื่องเดียวกัน — นั่นคือรูที่ปล่อยให้บั๊กนี้อยู่ได้

    `_budget_gb()` หัก 1.5 × 3 = 4.5 ให้ถูกมาตลอด ส่วน `plan_kv_pin()` หัก 1.5 · ไม่มีเทสไหน
    เอาสองค่านี้มาวางข้างกัน ความต่างเลยไม่มีใครเห็นจนกระทั่งมีคนไปอ่าน nvidia-smi
    """
    target = TargetSpec("besthaiai", MemoryModel.DISCRETE, 12.0, BESTHAIAI_GPUS,
                        system_ram_gb=125.0, tested=False, gpus_per_node=BESTHAIAI_GPUS)
    budget, notes = _budget_gb(target, "llamacpp")
    analyzer_overhead = LLAMACPP_OVERHEAD_GB_PER_GPU * target.gpu_count
    plan = plan_kv_pin(besthaiai_model(), besthaiai_host())

    assert plan["overhead_gb"] == pytest.approx(analyzer_overhead, abs=0.05)
    # budget = 36 × 0.85 − 4.5 แล้วคูณ 0.95 เพราะ preset rtx-3060 ยัง tested=False
    assert budget == pytest.approx((36.0 * 0.85 - analyzer_overhead) * 0.95, abs=0.05)
    assert any("conservative" in n for n in notes)
    assert any("ต่อใบ" in n for n in notes)


# ── ไฟล์คู่: mmproj / MTP ─────────────────────────────────────────────────────────
def vision_gguf(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=WEIGHTS_BYTES,
        selected_gguf="Qwen3.6-35B-A3B-Q4_K_M.gguf",
        context_length=CONTEXT,
        kv_dims=KvDims(layers=44, kv_heads=1, head_dim=128),
        gguf_variants=[
            GgufVariant(filename="Qwen3.6-35B-A3B-Q4_K_M.gguf", size_bytes=WEIGHTS_BYTES),
            GgufVariant(filename="mmproj-F16.gguf", size_bytes=MMPROJ_BYTES, is_mmproj=True),
        ],
    )
    base.update(overrides)
    return ModelReport(**base)


def test_the_projector_is_part_of_the_weight_total():
    fit = analyze(vision_gguf(), PRESETS["dgx-spark-single"])
    assert fit.weights_gb == pytest.approx(21.1, abs=0.05)
    assert fit.companion_weights_gb == pytest.approx(0.9, abs=0.05)
    assert any("mmproj" in n for n in fit.notes)


def test_a_text_only_model_gains_nothing():
    plain = vision_gguf(gguf_variants=[
        GgufVariant(filename="Qwen3.6-35B-A3B-Q4_K_M.gguf", size_bytes=WEIGHTS_BYTES)])
    fit = analyze(plain, PRESETS["dgx-spark-single"])
    assert fit.weights_gb == pytest.approx(20.2, abs=0.05)
    assert fit.companion_weights_gb == 0.0
    assert not any("mmproj" in n for n in fit.notes)


def test_only_one_projector_and_one_draft_head_are_reserved():
    """repo ที่แถม mmproj หลาย precision มีจริง — llama-server รับ `--mmproj` ได้ไฟล์เดียว

    รวมทุกไฟล์ = เกินจริงเท่าตัวแล้วไปตัด context ของผู้ใช้ทิ้งด้วยเหตุผลที่ไม่มีอยู่จริง
    """
    report = vision_gguf(gguf_variants=[
        GgufVariant(filename="w-Q4_K_M.gguf", size_bytes=WEIGHTS_BYTES),
        GgufVariant(filename="mmproj-F32.gguf", size_bytes=3 * GIB, is_mmproj=True),
        GgufVariant(filename="mmproj-BF16.gguf", size_bytes=1 * GIB, is_mmproj=True),
        GgufVariant(filename="mmproj-F16.gguf", size_bytes=1 * GIB, is_mmproj=True),
        GgufVariant(filename="mtp-head.gguf", size_bytes=GIB // 4, is_mtp=True),
    ])
    # เล็กสุดของแต่ละประเภท = ตรงกับที่ brain._pick_projector() เลือกจริง
    assert companion_weight_bytes(report) == 1 * GIB + GIB // 4


def test_variants_without_a_size_are_skipped_not_guessed():
    report = vision_gguf(gguf_variants=[
        GgufVariant(filename="w-Q4_K_M.gguf", size_bytes=WEIGHTS_BYTES),
        GgufVariant(filename="mmproj-F16.gguf", size_bytes=None, is_mmproj=True),
    ])
    assert companion_weight_bytes(report) == 0


def test_the_projector_can_change_the_verdict_on_a_tight_card():
    """ไม่ใช่แค่เลขสวยขึ้น — 0.9 GB บนการ์ด 12 GB คือความต่างระหว่างแผนที่จริงกับแผนที่หลอก"""
    small = vision_gguf(
        weight_bytes=int(6.9 * GIB), context_length=65536,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        gguf_variants=[
            GgufVariant(filename="w-Q4_K_M.gguf", size_bytes=int(6.9 * GIB)),
            GgufVariant(filename="mmproj-F16.gguf", size_bytes=int(0.9 * GIB), is_mmproj=True),
        ],
    )
    text_only = small.model_copy(update={"gguf_variants": small.gguf_variants[:1]})
    card = PRESETS["rtx-4070-ti-super"]  # 16 GB → budget 12.1 GB
    # KV เหลือ 5.2 GB (ข้อความล้วน) พอถึงขั้น 32,768 · 4.3 GB (มี projector) ไม่ถึง ตกมาที่ 16,384
    assert analyze(text_only, card).max_safe_context == 32768
    assert analyze(small, card).max_safe_context == 16384


# ── ฝั่งที่แตะเครื่องจริง: fleet/sizing.py ──────────────────────────────────────────
def _bundle_with_projector(tmp_path: Path) -> Path:
    """bundle llama.cpp ที่มี mmproj — รูปเดียวกับที่ generator/renderer.py เขียนออกมา"""
    d = tmp_path / "bundle"
    d.mkdir()
    ctl = d / "qwen36-single.sh"
    ctl.write_text(
        "#!/usr/bin/env bash\n"
        'BUNDLE_ENV="${BUNDLE_ENV:-x}"\n'
        "MODEL_FILES=(\n"
        '  "Qwen3.6-35B-A3B-Q4_K_M.gguf"\n'
        '  "mmproj-F16.gguf"\n'
        ")\n"
        "EXPECTED_SIZES=(\n"
        f'  "{WEIGHTS_BYTES}"\n'
        f'  "{MMPROJ_BYTES}"\n'
        ")\n"
        'MMPROJ_FILE="${MMPROJ_FILE-mmproj-F16.gguf}"\n',
        encoding="utf-8",
    )
    ctl.chmod(0o755)
    (d / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": "unsloth/Qwen3.6-35B-A3B-GGUF", "weight_bytes": WEIGHTS_BYTES,
                  "native_context": CONTEXT, "kv_bytes_per_token": KV_PER_TOKEN},
        "runtime": {"engine": "llamacpp"}, "topology": "single",
        "serving": {"context": CONTEXT, "max_num_seqs": 1},
        "features": {"multimodal": {"projector_files": ["mmproj-F16.gguf"]}},
    }), encoding="utf-8")
    return ctl


def _server(ctl: Path):
    from lmds.fleet import ServerInfo

    return ServerInfo(slug="qwen36", model_id="unsloth/Qwen3.6-35B-A3B-GGUF", engine="llamacpp",
                      mode="docker", port=8080, container="lmds-qwen36",
                      controller=str(ctl), running=True, healthy=True)


def test_model_info_reads_the_projector_size_out_of_the_controller(tmp_path):
    """ขนาดของ mmproj มีอยู่แล้วใน EXPECTED_SIZES ของ controller — ไม่ต้องยิงเน็ต ไม่ต้องแตะดิสก์ของโมเดล"""
    ctl = _bundle_with_projector(tmp_path)
    profile = yaml.safe_load((ctl.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    info = model_info_for(_server(ctl), profile, {}, [], with_logs=False)

    assert info["weight_bytes"] == WEIGHTS_BYTES          # ของเดิมไม่ถูกแตะ
    assert info["companion_weight_bytes"] == MMPROJ_BYTES  # ส่วนที่เคยหายไป


def test_a_bundle_without_a_projector_adds_nothing(tmp_path):
    ctl = _bundle_with_projector(tmp_path)
    ctl.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_FILES=(\n"
        '  "Qwen3.6-35B-A3B-Q4_K_M.gguf"\n'
        ")\n"
        f'EXPECTED_SIZES=(\n  "{WEIGHTS_BYTES}"\n)\n',
        encoding="utf-8",
    )
    profile = {"model": {"weight_bytes": WEIGHTS_BYTES}, "runtime": {"engine": "llamacpp"}}
    assert model_info_for(_server(ctl), profile, {}, [], with_logs=False)["companion_weight_bytes"] == 0


def test_a_controller_we_cannot_read_reports_zero_not_a_guess(tmp_path):
    missing = tmp_path / "gone" / "x-single.sh"
    profile = {"model": {"weight_bytes": WEIGHTS_BYTES}, "runtime": {"engine": "llamacpp"}}
    assert model_info_for(_server(missing), profile, {}, [], with_logs=False)["companion_weight_bytes"] == 0


def test_host_info_carries_the_gpu_count_through():
    """ช่องว่างที่แท้จริง: จำนวน GPU ถูกยุบหายไปตอนรวม VRAM แล้วไม่มีใครรู้ว่าเครื่องมีกี่ใบ"""
    host = {"memory_model": "discrete", "gpus": [
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": 11.2},
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": 9.9},
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": 10.7},
    ]}
    out = host_info_from(host, [], "qwen36")
    assert out["gpu_count"] == BESTHAIAI_GPUS
    assert out["total_gb"] == pytest.approx(36.0, abs=0.05)


def test_a_host_with_no_readable_gpus_falls_back_to_one():
    out = host_info_from({"memory_model": "unified", "ram_total_gb": 121.0, "ram_used_gb": 100.0}, [], "x")
    assert plan_kv_pin({"slug": "x", "engine": "vllm", "weight_bytes": GIB,
                        "kv_bytes_per_token": 4096, "native_context": 8192},
                       out)["gpu_count"] == 1


def test_the_whole_besthaiai_path_end_to_end(tmp_path):
    """จาก bundle บนดิสก์ + host payload ของ inventory → ตาราง `lmds fit` ที่ตรงกับเครื่องจริง"""
    ctl = _bundle_with_projector(tmp_path)
    profile = yaml.safe_load((ctl.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    host = {"memory_model": "discrete", "gpus": [
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": mib / 1024}
        for mib in BESTHAIAI_MIB
    ]}
    info = model_info_for(_server(ctl), profile, {}, [], with_logs=False)
    plan = plan_kv_pin(info, host_info_from(host, [], "qwen36"), slots=1, context=CONTEXT)

    assert plan["gpu_count"] == BESTHAIAI_GPUS
    assert plan["ram_needed_gb"] == pytest.approx(31.1, abs=0.05)
    assert 0 < MEASURED_GIB - plan["ram_needed_gb"] < 1.0


def test_the_download_lock_is_not_mistaken_for_a_broken_weight_file(tmp_path, monkeypatch):
    """เคสจริงบนเครื่องลูกค้า 2026-09-20 — doctor ฟ้อง lock ของตัวเองว่าเป็นไฟล์เสีย

    `.download.lock` คือ flock ที่ controller สร้างด้วย `exec 9>` จึงขนาด 0 เสมอ
    doctor เดิมนับรวมในกติกา "ไฟล์ขนาด 0 ไบต์ = เสีย" แล้วแนะให้ `lmds repair`
    ซึ่งโหลด weight 21 GB ใหม่ หยุดโมเดลที่กำลังเสิร์ฟอยู่ แล้วจบด้วยการทิ้ง lock ไว้อีก
    — วนไม่รู้จบทั้งที่ weight มี .sha256-ok ครบตลอด
    """
    from lmds.doctor import checks

    directory = tmp_path / "models" / "demo"
    directory.mkdir(parents=True)
    (directory / "model.gguf").write_bytes(b"x" * 64)
    (directory / ".download.lock").touch()          # 0 ไบต์ — ปกติ ไม่ใช่อาการเสีย
    monkeypatch.setattr(checks, "_model_dir", lambda slug: directory)

    profile = {"runtime": {"engine": "llamacpp"}, "model": {"selected_gguf": "model.gguf"}}
    statuses = {f.name: f.status.name for f in checks._check_weights(profile, "demo")}
    assert statuses["weights"] == "OK", "lock ของ LMDS เองไม่ควรถูกนับว่าเป็น weight เสีย"

    # ไฟล์ 0 ไบต์ที่ *ไม่ใช่* ของ LMDS ยังต้องถูกจับได้เหมือนเดิม
    (directory / "mmproj.gguf").touch()
    statuses = {f.name: f.status.name for f in checks._check_weights(profile, "demo")}
    assert statuses["weights"] == "FAIL", "ไฟล์ 0 ไบต์จริง ๆ ต้องยังฟ้องอยู่"
