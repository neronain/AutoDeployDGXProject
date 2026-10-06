"""weight ใน .safetensors ที่ vLLM/SGLang โหลดไม่ได้ ต้องถูกปฏิเสธก่อนดาวน์โหลด — ตระกูล MLX ที่หลุด และ EXL2/EXL3

ต่อจาก tests/test_mlx_checkpoint.py (d0d1ec1) · audit 2026-10-06 กับ repo จริง ~90 ตัวพบว่าการแก้นั้นปิดได้เคสเดียว:
- `library_name == "mlx"` เทียบตรงตัว แต่ของจริงใช้ `mlx-lm` · `mlx-vlm` · `mlx-audio` · `mlx-serve` · `mlx-qwen3-asr`:
  `salakash/Minimalism` (mlx-lm · LoRA ของ MLX) กับ `nativ-community/sapiens2-seg-0.4b-bf16` (mlx-vlm) → "fits (vllm)"
  plan exit 0 (ตัวหลัง context 1,048,576)
- quant ของ ExLlama: `lmds generate doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw` → bundle vLLM context 217,088 เปิด tool
  calling ผ่านทุก gate (108 GB) · `icefog72/IceWhiskeyRP-7b-8bpw-exl2` → fits (vllm) · `turboderp/Qwen3.8-27B-exl3`
  (main มีแค่ cal_trace.safetensors 4 MB) → "weights 0.0 GB · fits (vllm)"

ทุกข้อปฏิเสธมีเทสคู่ที่พิสูจน์ว่า repo หน้าตาคล้ายกันแต่เสิร์ฟได้ยัง deploy ได้ — ปฏิเสธผิดคือขวางลูกค้าโดยไม่มีทางข้าม
"""

import pytest
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector import ArtifactType
from tests.real_hub import RealHub, flat

runner = CliRunner()
SPARK = PRESETS["dgx-spark-single"]

MLX_LORA = "salakash/Minimalism"
MLX_VLM = "nativ-community/sapiens2-seg-0.4b-bf16"
EXL3 = "doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw"
EXL2 = "icefog72/IceWhiskeyRP-7b-8bpw-exl2"
EXL3_BRANCHES = "turboderp/Qwen3.8-27B-exl3"
BNB = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
GPTQ = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
NVFP4 = "Inferact/Qwen3.8-27B-NVFP4"
MLX_TAGGED = "mlx-community/gemma-3-12b-it-bf16"
REFUSED = (MLX_LORA, MLX_VLM, EXL3, EXL2, EXL3_BRANCHES)
SERVABLE = (BNB, GPTQ, NVFP4, MLX_TAGGED)


@pytest.fixture
def hub(isolated_config, monkeypatch):
    fake = RealHub(*REFUSED, *SERVABLE)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    return fake


# ── ตระกูล MLX: library_name ที่ไม่ใช่คำว่า "mlx" เป๊ะ ───────────────────────────────────────────────


@pytest.mark.parametrize("repo, library", [(MLX_LORA, "mlx-lm"), (MLX_VLM, "mlx-vlm")])
def test_mlx_family_library_names_are_mlx(repo, library):
    report = RealHub(repo).inspect(repo)
    assert report.library_name == library
    assert report.unsupported_format == "mlx"
    assert f"library_name={library}" in report.unsupported_evidence

    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.UNSUPPORTED and fit.engine_assumed == "none"
    with pytest.raises(PlanError, match="MLX"):
        build_plan(report, fit, None)


@pytest.mark.parametrize("library, expected", [
    ("mlx", True), ("mlx-lm", True), ("mlx-vlm", True), ("mlx-audio", True), ("mlx-serve", True),
    ("mlx-qwen3-asr", True), ("MLX-LM", True),
    # ไม่นับจากชื่อไลบรารี: mflux/mtplx (เจอในตัวอย่าง — ถูกจับจากบล็อก quantization แทน) และของที่แค่ขึ้นต้นคล้าย
    ("mflux", False), ("mtplx", False), ("mlxtend", False), ("transformers", False), ("", False), (None, False),
])
def test_which_library_names_count_as_mlx(library, expected):
    from lmds.inspector.formats import is_mlx_library

    assert is_mlx_library(library) is expected


def test_a_transformers_repo_that_only_carries_the_mlx_tag_still_deploys():
    """mlx-community/gemma-3-12b-it-bf16: library_name=transformers + tag mlx — ไม่มีหลักฐานชี้ขาด จึงเตือน ไม่ปฏิเสธ"""
    report = RealHub(MLX_TAGGED).inspect(MLX_TAGGED)
    assert report.unsupported_format is None
    assert any("MLX" in w for w in report.warnings)
    fit = analyze(report, SPARK)
    assert fit.verdict is not Verdict.UNSUPPORTED
    assert build_plan(report, fit, None).runtime.engine is Engine.VLLM


# ── EXL2 / EXL3 ──────────────────────────────────────────────────────────────────────────────────────


def test_exl3_quant_is_refused_with_the_evidence_and_the_alternatives():
    report = RealHub(EXL3).inspect(EXL3)
    assert report.artifact_type is ArtifactType.SAFETENSORS
    assert report.unsupported_format == "exl3"
    assert report.quantization == "exl3-4.05bpw", "เดิมรายงานแค่ 'exl3' แล้ววางแผน vLLM ต่อ"
    seen = " ".join(report.unsupported_evidence)
    assert "quant_method=exl3" in seen and "tag exl3" in seen and "library_name=trellis" in seen

    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.UNSUPPORTED and fit.recommended_context is None
    for engine in (None, Engine.VLLM, Engine.SGLANG):
        with pytest.raises(PlanError) as err:
            build_plan(report, fit, None, engine=engine)
        said = str(err.value)
        assert "ExLlama" in said and "EXL3" in said, "ต้องบอกสาเหตุ"
        assert "GGUF" in said and "NVFP4" in said and "Qwen/Qwen3.8-Flash-Next" in said, "ต้องชี้รุ่นที่ใช้แทนได้"
        assert "67.7 GB" in said, "ต้องบอกขนาดที่จะโหลดฟรี"


def test_exl2_is_caught_from_the_tag_and_the_repo_name_when_config_says_nothing():
    """icefog72/IceWhiskeyRP-7b-8bpw-exl2: config.json ลอกมาจากต้นฉบับ (library_name=transformers ไม่มี quantization_config)
    ไฟล์จริงคือ output.safetensors ไฟล์เดียว · index ชี้ shard 3 ตัวที่ไม่มีอยู่"""
    report = RealHub(EXL2).inspect(EXL2)
    assert report.library_name == "transformers"
    assert report.unsupported_format == "exl2"
    assert analyze(report, SPARK).verdict is Verdict.UNSUPPORTED


def test_an_exl3_repo_whose_quants_live_on_branches_no_longer_fits_with_zero_gb():
    """turboderp/Qwen3.8-27B-exl3: main มีแค่ cal_trace.safetensors 4 MB — เดิม "weights 0.0 GB · fits (vllm)" bundle ✅"""
    report = RealHub(EXL3_BRANCHES).inspect(EXL3_BRANCHES)
    assert report.unsupported_format == "exl3"
    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.UNSUPPORTED and fit.weights_gb is None


def test_exl3_layers_mixed_into_a_modelopt_checkpoint_are_not_read_as_nvfp4():
    """brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 (Hub 2026-10-06): library_name=trellis · quantization_config มีทั้ง
    config_groups ของ ModelOpt และ `exl3_dense` — เดิมถูกจัดเป็น NVFP4"""
    from lmds.inspector.formats import exl_evidence

    tags = ["trellis", "safetensors", "glm_moe_dsa", "exl3", "quantization", "glm", "moe", "mixed-precision",
            "base_model:zai-org/GLM-5.2", "base_model:quantized:zai-org/GLM-5.2", "modelopt"]
    config = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
              "quantization_config": {"config_groups": {"group_0": {"targets": ["Linear"]}},
                                      "exl3_dense": {"bitrates": {"default": 3}}}}
    evidence, kind = exl_evidence("brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78", "trellis", tags, config)
    assert kind == "exl3"
    assert any("exl3_dense" in e for e in evidence) and "library_name=trellis" in evidence


@pytest.mark.parametrize("repo, library, tags", [
    # สัญญาณเดียว = ร่องรอย ไม่ใช่คำตัดสิน
    ("org/Qwen3-32B", "transformers", ["transformers", "safetensors", "exl3"]),     # tag อย่างเดียว
    ("org/Qwen3-32B-exl2", "transformers", ["transformers", "safetensors"]),        # ชื่ออย่างเดียว
])
def test_one_exl_signal_alone_is_a_trace_not_a_verdict(repo, library, tags):
    from lmds.inspector.formats import exl_evidence

    evidence, kind = exl_evidence(repo, library, tags, {"architectures": ["Qwen3ForCausalLM"]})
    assert evidence and kind is None


@pytest.mark.parametrize("repo, library, tags", [
    ("org/Excel-Assistant-7B", "transformers", ["transformers", "safetensors"]),    # "exl" กลางคำ
    ("org/Llama-3-8B-4.0bpw", "transformers", ["transformers", "safetensors"]),     # bpw ลำพังไม่ได้พูดถึง ExLlama
    ("org/Trellis-Coder-7B", "trellis", ["safetensors"]),                           # library trellis ลำพัง
])
def test_lookalikes_carry_no_exl_evidence_at_all(repo, library, tags):
    from lmds.inspector.formats import exl_evidence

    assert exl_evidence(repo, library, tags, {"architectures": ["LlamaForCausalLM"]}) == ([], None)


# ── quantizer ฝั่ง transformers ที่หน้าตาคล้ายกัน ยังต้อง deploy ได้ ─────────────────────────────────


@pytest.mark.parametrize("repo", [GPTQ, NVFP4])
def test_transformers_quantized_checkpoints_still_deploy(repo):
    report = RealHub(repo).inspect(repo)
    assert report.unsupported_format is None
    assert not any("ExLlama" in w or "MLX" in w for w in report.warnings)
    fit = analyze(report, SPARK)
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert build_plan(report, fit, None).runtime.engine is Engine.VLLM


def test_bitsandbytes_deploys_with_a_warning_because_the_image_was_never_proven_to_lack_it():
    report = RealHub(BNB).inspect(BNB)
    assert report.unsupported_format is None
    assert any("bitsandbytes" in w for w in report.warnings)
    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.FITS
    assert build_plan(report, fit, None).runtime.engine is Engine.VLLM


# ── CLI + หน้าเว็บ พูดเรื่องเดียวกัน ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("repo, word", [(EXL3, "ExLlama"), (EXL2, "ExLlama"), (EXL3_BRANCHES, "ExLlama"),
                                        (MLX_LORA, "MLX"), (MLX_VLM, "MLX")])
def test_cli_generate_refuses_before_anything_is_written(hub, tmp_path, repo, word):
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", repo, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)])
    assert result.exit_code == 1, result.output
    said = flat(result.output)
    assert "ไม่รองรับ" in said and word in said and "GGUF" in said
    assert not out.exists(), "ไม่มี bundle = ไม่มี controller ให้ใครกด download"
    assert not hub.weight_requests


def test_cli_inspect_says_unsupported_and_shows_no_fit_table(hub):
    result = runner.invoke(app, ["inspect", EXL3, "--target", "dgx-spark-single"])
    assert result.exit_code == 0
    said = flat(result.output)
    assert "ExLlama" in said and "ไม่รองรับ" in said and "FitAnalysis" not in said


@pytest.mark.parametrize("repo, word", [(EXL3, "ExLlama"), (MLX_LORA, "MLX")])
def test_web_analyze_refuses_with_alternatives(hub, repo, word):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        dep.analyze(repo, target="dgx-spark-single", no_llm=True)
    assert err.value.kind == "unsupported" and word in err.value.message
    assert err.value.extra["alternatives"]
    assert not hub.weight_requests
