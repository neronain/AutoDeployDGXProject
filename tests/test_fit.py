"""เทส Fit Analyzer — เคสอ้างอิงคำนวณมือไว้ล่วงหน้า

Qwen3-32B (BF16): weights 65.0 GB, layers 64, kv_heads 8, head_dim 128, native ctx 40,960
→ KV ต่อ token = 2×64×8×128×2 = 524,288 bytes? ไม่ใช่ — 2(K,V)×64×8×128×2(fp16) = 262,144 bytes/token
"""

from lmds.fit import PRESETS, TargetSpec, Verdict, analyze, from_hardware_report
from lmds.fit.analyzer import GIB
from lmds.hardware import MemoryModel
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport


def qwen32b(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="abc",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB,
        context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
    )
    base.update(overrides)
    return ModelReport(**base)


def test_kv_bytes_per_token_formula():
    dims = KvDims(layers=64, kv_heads=8, head_dim=128)
    assert dims.bytes_per_token_fp16 == 2 * 64 * 8 * 128 * 2  # 262,144


def test_qwen32b_fits_on_spark():
    fit = analyze(qwen32b(), PRESETS["dgx-spark-single"])
    # budget = 128 − 12 (OS) − 2.5 (vllm) = 113.5 GB → KV เหลือ 48.5 GB
    # max ctx = 48.5 GiB / 262144 ≈ 198k > native 40960 → safe step = 32768
    assert fit.verdict is Verdict.FITS
    assert fit.max_safe_context == 32768
    assert fit.recommended_context == 32768
    assert fit.client_input_budget == 32768 - 8192 - 2048


def test_qwen32b_bf16_needs_quant_on_24gb():
    fit = analyze(qwen32b(), PRESETS["rtx-pro-4000"])
    assert fit.verdict is Verdict.NEEDS_SMALLER_QUANT
    assert any("quantize" in a or "GGUF" in a for a in fit.alternatives)
    assert any("tensor parallel" in a or "VRAM" in a for a in fit.alternatives)


def test_small_gguf_fits_on_4070():
    report = ModelReport(
        repo_id="unsloth/Qwen3-8B-GGUF",
        revision_sha="abc",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB,
        context_length=40960,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf="q4.gguf",
    )
    fit = analyze(report, PRESETS["rtx-4070-super"])
    # budget = 12×0.85 − 1.5 = 8.7 GB → KV เหลือ 3.7 GB → max ctx ≈ 3.7GiB/147456 ≈ 26k → step 16384
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert fit.recommended_context >= 16384
    assert fit.client_input_budget > 0


def test_offload_path_for_gguf_larger_than_vram():
    report = ModelReport(
        repo_id="org/big-GGUF",
        revision_sha="abc",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=20 * GIB,
        selected_gguf="q4.gguf",
    )
    target = TargetSpec("rtx-4070-super-64ram", MemoryModel.DISCRETE, 12.0, 1, system_ram_gb=64.0, tested=True)
    fit = analyze(report, target)
    assert fit.verdict is Verdict.FITS_WITH_OFFLOAD
    assert fit.recommended_context == 4096


def test_vllm_never_offloads():
    report = qwen32b()  # safetensors → vllm
    target = TargetSpec("rtx-24g-ram", MemoryModel.DISCRETE, 24.0, 1, system_ram_gb=128.0, tested=True)
    fit = analyze(report, target)
    assert fit.verdict is not Verdict.FITS_WITH_OFFLOAD


def test_unknown_kv_dims_conservative():
    fit = analyze(qwen32b(kv_dims=None), PRESETS["dgx-spark-single"])
    assert fit.verdict is Verdict.FITS
    assert fit.kv_estimated is True
    assert fit.recommended_context == 16384
    assert any("KV" in n for n in fit.notes)


def test_untested_target_gets_headroom_cut():
    tested = analyze(qwen32b(), PRESETS["rtx-pro-4000"])
    untested = analyze(qwen32b(), PRESETS["rtx-4090"])  # VRAM เท่ากัน แต่ tested=False
    assert untested.budget_gb < tested.budget_gb
    assert any("conservative" in n for n in untested.notes)


def test_multi_variant_gguf_reports_fitting_subset():
    report = ModelReport(
        repo_id="org/multi-GGUF",
        revision_sha="abc",
        artifact_type=ArtifactType.GGUF,
        gguf_variants=[
            GgufVariant(filename="m-Q8_0.gguf", size_bytes=30 * GIB),
            GgufVariant(filename="m-Q4_K_M.gguf", size_bytes=18 * GIB),
            GgufVariant(filename="m-Q2_K.gguf", size_bytes=10 * GIB),
            GgufVariant(filename="mmproj-F16.gguf", size_bytes=1 * GIB, is_mmproj=True),
        ],
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    assert len(fit.variant_fits) == 3  # mmproj ไม่นับ
    assert all(v.fits for v in fit.variant_fits)  # budget 114 GB (llamacpp) ทุกตัวผ่าน

    fit_small = analyze(report, PRESETS["rtx-4070-ti-super"])
    # budget = 16×0.85 − 1.5 = 12.1 GB → ผ่านเฉพาะ Q2_K (10 GB ≤ 12.1×0.9 = 10.9)
    fitting = [v.filename for v in fit_small.variant_fits if v.fits]
    assert fitting == ["m-Q2_K.gguf"]
    # ยังไม่เลือกไฟล์ → verdict เป็น unknown (รอผู้ใช้เลือก variant ที่ผ่าน)
    assert fit_small.verdict is Verdict.UNKNOWN
    assert any("1/3" in n for n in fit_small.notes)

    # ไม่มี variant ไหนผ่านเลย → needs-smaller-quant
    tiny = TargetSpec("tiny-8g", MemoryModel.DISCRETE, 8.0, 1, tested=True)
    fit_none = analyze(report, tiny)
    assert all(not v.fits for v in fit_none.variant_fits)
    assert fit_none.verdict is Verdict.NEEDS_SMALLER_QUANT


def test_dual_gpu_aggregates_budget():
    single = analyze(qwen32b(), PRESETS["rtx-pro-4000"])
    dual = analyze(qwen32b(), PRESETS["rtx-pro-4000-dual"])
    assert dual.budget_gb > single.budget_gb
    # 65 GB บน dual 24GB (budget = 48×0.85 − 5 = 35.8) ยังไม่พอ → ต้อง quant
    assert dual.verdict is Verdict.NEEDS_SMALLER_QUANT
    assert any("tensor parallel" in n for n in dual.notes)


def test_from_hardware_report():
    from lmds.hardware.profiler import DetectedGpu, HardwareReport
    from lmds.hardware.profiles import TargetProfile, lookup_gpu

    report = HardwareReport(
        arch="x86_64",
        gpus=[
            DetectedGpu("NVIDIA RTX PRO 4000 Blackwell", 24564, "12.0", lookup_gpu("rtx pro 4000 blackwell")),
            DetectedGpu("NVIDIA RTX PRO 4000 Blackwell", 24564, "12.0", lookup_gpu("rtx pro 4000 blackwell")),
        ],
        ram_gb=128.0,
        profile=TargetProfile.RTX_MULTI_GPU,
    )
    spec = from_hardware_report(report)
    assert spec is not None
    assert spec.gpu_count == 2
    assert spec.memory_model is MemoryModel.DISCRETE
    assert spec.tested is True
    assert 23.5 <= spec.memory_gb <= 24.5


def test_no_gpu_returns_none():
    from lmds.hardware.profiler import HardwareReport

    assert from_hardware_report(HardwareReport(arch="arm64")) is None


def test_context_cap_is_reported_not_hidden():
    """เคสจริง Qwen3-Coder-30B บน DGX Spark (2026-08-02): แผนเสนอ 65,536 แต่รันได้จริง 262,144

    สูตรคำนวณถูกอยู่แล้ว (max_safe_context = 262,144) — ที่ผิดคือค่านั้นไม่เคยถูกแสดง
    ผู้ใช้จึงเสีย context ไป 4 เท่าโดยไม่รู้ตัว
    """
    from lmds.fit.analyzer import GIB
    from lmds.inspector.report import ArtifactType, KvDims, ModelReport

    report = ModelReport(
        repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=int(32.5 * GIB),
        selected_gguf="UD-Q8_K_XL.gguf",
        context_length=262144,
        kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])

    assert fit.max_safe_context == 262144, "สูตรเดิมคำนวณถูก — อย่าไปแก้สูตร"
    assert fit.recommended_context == 65536  # ค่าเริ่มต้นมาตรฐาน v3.0.0
    assert any("262,144" in n for n in fit.notes), "ต้องบอกผู้ใช้ว่าเครื่องรับได้มากกว่านี้"


def test_no_headroom_note_when_cap_not_binding():
    """โมเดลที่ max_safe เท่ากับค่าที่แนะนำอยู่แล้วต้องไม่มี note รกขึ้นมา"""
    fit = analyze(qwen32b(), PRESETS["dgx-spark-single"])
    assert fit.max_safe_context == fit.recommended_context
    assert not any("รองรับได้ถึง" in n for n in fit.notes)
