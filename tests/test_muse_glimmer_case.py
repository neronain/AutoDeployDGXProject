"""เคสศึกษา Muse-Glimmer-30B-GGUF (2026-08-13)

โมเดล multimodal ที่มาเป็น GGUF + mmproj รันบน llama.cpp — คนละเส้นทางกับ vLLM
ที่ล้มไปรอบก่อน เคสนี้เปิดโปงสามอย่างที่ฝั่ง no-LLM ยังไม่ครบ:

1. รันไทม์ผูกกับเครื่อง ไม่ได้ผูกกับโมเดล — llama.cpp มี build เดียวใช้ร่วมกันหมด
2. เลือก mmproj ด้วยขนาดอย่างเดียว — repo ที่มี projector ของหลายโมเดลจะหยิบผิด
3. preflight สถาปัตยกรรมข้าม native ทั้งหมด — ซึ่งคือโหมดที่ llama.cpp ใช้จริง
"""

import pytest

from lmds.brain.orchestrator import _pick_projector
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport


def _variant(name, size, mmproj=False):
    return GgufVariant(filename=name, size_bytes=size, is_mmproj=mmproj)


def _report(selected):
    return ModelReport(
        repo_id="unsloth/Muse-Glimmer-30B-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        selected_gguf=selected,
        context_length=131072,
        kv_dims=KvDims(layers=52, kv_heads=2, head_dim=128),
    )


# ไฟล์จริงใน repo
PROJECTORS = [
    _variant("mmproj-Muse-Glimmer-30B-BF16.gguf", 1_500_000_000, mmproj=True),
    _variant("mmproj-Muse-Glimmer-30B-Q8_0.gguf", 900_000_000, mmproj=True),
    _variant("mmproj-kquant.gguf", 300_000_000, mmproj=True),  # คู่กับ dflash-kquant
]


def test_projector_matches_the_weight_family_not_the_smallest_file():
    """mmproj-kquant เล็กที่สุด แต่คู่กับ dflash-kquant ซึ่งเป็นคนละโมเดล"""
    chosen = _pick_projector(PROJECTORS, _report("Muse-Glimmer-30B-UD-Q8_K_XL.gguf"))
    assert chosen.filename == "mmproj-Muse-Glimmer-30B-Q8_0.gguf"


def test_smallest_wins_within_the_matching_family():
    """เมื่อชื่อร่วมตระกูลกันแล้ว ค่อยใช้กติกาเดิม — เล็กกว่าไม่เสียคุณภาพพอจะคุ้ม"""
    chosen = _pick_projector(PROJECTORS, _report("Muse-Glimmer-30B-Q8_0.gguf"))
    assert chosen.filename == "mmproj-Muse-Glimmer-30B-Q8_0.gguf"


def test_falls_back_to_smallest_when_nothing_shares_a_name():
    """ไม่มีตัวไหนชื่อใกล้เคียง — กลับไปใช้พฤติกรรมเดิม ดีกว่าไม่เลือกอะไรเลย"""
    unrelated = [
        _variant("mmproj-a.gguf", 800, mmproj=True),
        _variant("mmproj-b.gguf", 400, mmproj=True),
    ]
    assert _pick_projector(unrelated, _report("Totally-Other-Model.gguf")).filename == "mmproj-b.gguf"


def test_a_single_projector_is_always_the_answer():
    only = [_variant("mmproj-kquant.gguf", 300, mmproj=True)]
    assert _pick_projector(only, _report("Muse-Glimmer-30B-UD-Q8_K_XL.gguf")).filename == "mmproj-kquant.gguf"


# ---- รันไทม์ผูกกับโมเดล ไม่ใช่ผูกกับเครื่อง ----

def test_a_pinned_native_build_reaches_the_controller(tmp_path):
    """โมเดลสองตัวบนเครื่องเดียวต้องชี้ llama.cpp คนละ build ได้"""
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.fit.analyzer import GIB
    from lmds.generator import render_bundle

    report = _report("Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    report.weight_bytes = int(30.1 * GIB)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.runtime.native_dir = "/home/x/src/llama.cpp-muse"

    bundle = render_bundle(plan, report, fit, tmp_path)
    script = next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")
    assert 'LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/x/src/llama.cpp-muse}"' in script


def test_without_a_pin_the_shared_build_is_still_used(tmp_path):
    """ไม่ได้ pin = ใช้ของกลางตามเดิม ไม่บังคับให้ทุกคนต้อง build แยก"""
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.fit.analyzer import GIB
    from lmds.generator import render_bundle

    report = _report("Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    report.weight_bytes = int(30.1 * GIB)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    assert plan.runtime.native_dir is None

    bundle = render_bundle(plan, report, fit, tmp_path)
    script = next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")
    assert 'LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/src/llama.cpp}"' in script


# ---- preflight สถาปัตยกรรมสำหรับ llama.cpp ----

def test_a_build_that_does_not_know_the_architecture_fails_the_check(tmp_path):
    """จับให้ได้ก่อนผู้ใช้เสียเวลาโหลด 30 GB"""
    from lmds.doctor.checks import _lib_knows

    lib = tmp_path / "libllama.so"
    lib.write_bytes(b"\x00" * 4096 + b"gemma4\x00qwen3moe\x00llama\x00" + b"\x00" * 4096)
    assert _lib_knows(lib, "gemma4") is True
    assert _lib_knows(lib, "muse-glimmer") is False


def test_the_needle_is_found_across_a_chunk_boundary(tmp_path):
    """ชื่อ architecture ที่คร่อมรอยต่อ 1 MiB ต้องยังเจอ ไม่งั้นรายงานผิดเป็นครั้งคราว"""
    from lmds.doctor.checks import _lib_knows

    lib = tmp_path / "libllama.so"
    chunk = 1 << 20
    padding = b"\x00" * (chunk - 5)
    lib.write_bytes(padding + b"muse-glimmer" + b"\x00" * 128)
    assert _lib_knows(lib, "muse-glimmer") is True
