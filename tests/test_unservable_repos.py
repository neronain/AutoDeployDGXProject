"""repo ที่ไม่มีอะไรให้ engine ของเราเสิร์ฟ ต้องถูกปฏิเสธ — ไม่ใช่ได้ "fits (vllm)" กับ bundle ที่ผ่านทุก gate

audit 2026-10-06 (repo จริง ~90 ตัว · hub 6e2b474) — `lmds generate` ออก exit 0 + "Bundle (static-validated ✅)" ให้:
`IFM/K2-Horizon-7B-Uno` (adapter PEFT ล้วน ไม่มี config.json) · `onnx-community/Qwen3-0.6B-ONNX` (ไม่มี safetensors/GGUF
เลย → vLLM เปิด tool calling ให้) · `stabilityai/sdxl-turbo` (pipeline diffusers → vLLM "weights 38.8 GiB") ·
`city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf` (`general.architecture=flux` → llama.cpp)
inspect + plan exit 0 "fits (vllm)" ให้ Z-Image-Turbo · Qwen3-TTS · chronos-2 · vit-base · bert-base-NER ·
bert-base-uncased · t5-small · LoRA ของ Qwen3-0.6B · และ plan vLLM ให้ RKLLM / MNN / pytorch_model.bin ล้วน
task เพี้ยน: `nvidia/parakeet-tdt-0.6b-v3` (ASR) กับ classifier แบบ Prompt-Guard → embed เพราะ tag ลอย ๆ ชนะ pipeline_tag

ทุกข้อปฏิเสธมีเทสคู่ที่พิสูจน์ว่า repo หน้าตาคล้ายกันแต่เสิร์ฟได้ยัง deploy ได้ (ท้ายไฟล์)
"""

import json

import pytest
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan, rule_based_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector import ArtifactType
from lmds.inspector.report import ModelReport
from tests.real_hub import RealHub, flat

runner = CliRunner()
SPARK = PRESETS["dgx-spark-single"]

FLUX_FILE = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf"
PARAKEET_FILE = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/main/parakeet-tdt-0.6b-v3.q8_0.gguf"
WHISPER_FILE = "https://huggingface.co/handy-computer/whisper-medium-gguf/blob/main/whisper-medium-Q4_K_M.gguf"
TTS_FILE = "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/blob/main/qwen-talker-0.6b-base-BF16.gguf"

# (source, unsupported_format, คำที่ต้องอยู่ในหลักฐาน/ข้อความ, คำที่ต้องอยู่ในทางออก)
REFUSED = [
    # ── adapter ล้วน: ชี้ไป base model จาก card ──
    ("IFM/K2-Horizon-7B-Uno", "adapter", "adapter_model.safetensors", "IFM/K2-Horizon-7B"),
    ("icml2026-7516/ThinkSafe-Qwen3-0.6B", "adapter", "adapter_config.json", "Qwen/Qwen3-0.6B"),
    # ── ไม่มีไฟล์ weight ในรูปแบบที่เสิร์ฟ: บอกว่า repo มีอะไรแทน ──
    ("onnx-community/Qwen3-0.6B-ONNX", "no-weights", "ONNX", "Qwen/Qwen3-0.6B"),
    ("HanzoHuang/Qwen2.5-3B-Instruct-RKLLM", "no-weights", "RKLLM", "Qwen/Qwen2.5-3B-Instruct"),
    ("darkmaniac7/Qwen3.5-4B-uncensored-MNN", "no-weights", "MNN", "huihui-ai/Huihui-Qwen3.5-4B-abliterated"),
    ("facebook/opt-125m", "no-weights", "PyTorch pickle", ".gguf"),
    # ── pipeline ของ diffusers ──
    ("stabilityai/sdxl-turbo", "diffusers", "model_index.json", "ComfyUI"),
    ("Tongyi-MAI/Z-Image-Turbo", "diffusers", "model_index.json", "ComfyUI"),
    # ── GGUF ที่ไม่ใช่ LLM (header จริง) ──
    (FLUX_FILE, "non-llm-gguf", "general.architecture=flux", "ComfyUI"),
    (PARAKEET_FILE, "non-llm-gguf", "general.architecture=asr", "whisper.cpp"),
    (WHISPER_FILE, "non-llm-gguf", "general.architecture=whisper", "whisper.cpp"),
    (TTS_FILE, "non-llm-gguf", "general.architecture=qwen3-tts", "whisper.cpp"),
    # ── งานที่ LMDS ไม่มีโหมดเสิร์ฟ ──
    ("city96/FLUX.1-dev-gguf", "no-serving-mode", "pipeline_tag=text-to-image", "--task"),
    ("unsloth/Wan2.2-TI2V-5B-GGUF", "no-serving-mode", "pipeline_tag=text-to-video", "--task"),
    ("unsloth/Z-Image-Turbo-GGUF", "no-serving-mode", "pipeline_tag=text-to-image", "--task"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "no-serving-mode", "Qwen3TTSForConditionalGeneration", "--task"),
    ("amazon/chronos-2", "no-serving-mode", "pipeline_tag=time-series-forecasting", "--task"),
    ("google/vit-base-patch16-224", "no-serving-mode", "ViTForImageClassification", "--task"),
    ("dslim/bert-base-NER", "no-serving-mode", "BertForTokenClassification", "--task"),
    ("google-bert/bert-base-uncased", "no-serving-mode", "BertForMaskedLM", "--task"),
    ("google-t5/t5-small", "no-serving-mode", "is_encoder_decoder=true", "--task"),
    ("openai/whisper-large-v3-turbo", "no-serving-mode", "pipeline_tag=automatic-speech-recognition", "--task"),
    ("nvidia/parakeet-tdt-0.6b-v3", "no-serving-mode", "pipeline_tag=automatic-speech-recognition", "--task"),
    ("protectai/deberta-v3-base-prompt-injection-v2", "no-serving-mode", "DebertaV2ForSequenceClassification", "--task"),
]


def repo_of(source: str) -> str:
    return "/".join(source.removeprefix("https://huggingface.co/").split("/")[:2])


def inspect(source: str, **kw):
    return RealHub(repo_of(source), **kw).inspect(source)


@pytest.fixture
def hub(isolated_config, monkeypatch):
    fake = RealHub(*sorted({repo_of(s) for s, *_ in REFUSED} | set(SERVABLE)))
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    return fake


# ── ทุกชั้นปฏิเสธด้วยเหตุผลเดียวกัน ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source, kind, evidence, way_out", REFUSED)
def test_inspect_fit_and_plan_all_refuse(source, kind, evidence, way_out):
    report = inspect(source)
    assert report.unsupported_format == kind
    assert evidence in " · ".join(report.unsupported_evidence)

    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.UNSUPPORTED
    assert fit.engine_assumed == "none", "เดิม 'vllm' / 'llamacpp' — อ่านเหมือนมี engine เสิร์ฟได้"
    assert fit.recommended_context is None

    for planner in (rule_based_plan, lambda r, f, e: build_plan(r, f, None, engine=e)):
        with pytest.raises(PlanError) as err:
            planner(report, fit, None)
        said = str(err.value)
        assert evidence in said, "ข้อความต้องบอกว่ารู้ได้จากอะไร"
        assert way_out in said, "ข้อความต้องบอกว่าใช้อะไรแทน"


@pytest.mark.parametrize("source, kind, evidence, way_out", [
    REFUSED[0], REFUSED[2], REFUSED[6], REFUSED[8], REFUSED[15], REFUSED[20],
], ids=["adapter", "onnx", "diffusers", "flux-gguf", "tts", "t5"])
def test_cli_generate_and_web_refuse_before_anything_is_written(hub, tmp_path, source, kind, evidence, way_out):
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", source, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)])
    assert result.exit_code == 1, result.output
    said = flat(result.output)
    assert "ไม่รองรับ" in said and flat(evidence) in said and flat(way_out) in said
    assert not out.exists()

    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        dep.analyze(source, target="dgx-spark-single", no_llm=True)
    assert err.value.kind == "unsupported" and evidence in err.value.message
    assert any(way_out in alt for alt in err.value.extra["alternatives"])
    assert not hub.weight_requests


def test_cli_inspect_json_carries_the_verdict_for_scripts(hub):
    result = runner.invoke(app, ["inspect", "onnx-community/Qwen3-0.6B-ONNX", "--target", "dgx-spark-single", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["model"]["artifact_type"] == "unknown"
    assert payload["model"]["unsupported_format"] == "no-weights"
    assert [(f["verdict"], f["engine_assumed"]) for f in payload["fit"]] == [("unsupported", "none")]


# ── โครงสร้างที่หา repo จริงแบบ "ล้วน ๆ" ไม่ได้: ตัดไฟล์ออกจาก metadata จริง ──────────────────────────


def test_safetensors_at_the_root_without_config_json_is_refused():
    """gpt2 ที่ไม่มี config.json — vLLM/SGLang อ่านสถาปัตยกรรมไม่ได้ (เดิมเป็นแค่คำเตือน "ไม่พบ config.json")"""
    def drop_config(fixture):
        del fixture["files"]["config.json"]

    report = inspect("openai-community/gpt2", edits={"openai-community/gpt2": drop_config})
    assert report.unsupported_format == "no-config"
    assert analyze(report, SPARK).verdict is Verdict.UNSUPPORTED


def test_a_checkpoint_that_only_lives_in_a_sub_folder_is_refused():
    """LiquidAI/LFM2.5-2.6B-GGUF ที่ไม่มีไฟล์ GGUF เหลือ = มีแต่ qad/model.safetensors — engine ถูกชี้ไปที่ราก repo เสมอ"""
    def drop_ggufs(fixture):
        fixture["info"]["siblings"] = [s for s in fixture["info"]["siblings"] if not s["rfilename"].endswith(".gguf")]

    report = inspect("LiquidAI/LFM2.5-2.6B-GGUF", edits={"LiquidAI/LFM2.5-2.6B-GGUF": drop_ggufs})
    assert report.unsupported_format == "no-root-checkpoint"
    assert "qad/" in " ".join(report.unsupported_evidence)
    with pytest.raises(PlanError, match="ราก repo"):
        build_plan(report, analyze(report, SPARK), None)


# ── "ไม่รู้ชนิดไฟล์" ไม่ถูกเดาเป็น vLLM อีก ──────────────────────────────────────────────────────────


def test_an_unknown_artifact_is_never_assumed_to_be_vllm():
    report = ModelReport(repo_id="org/whatever", revision_sha="a" * 40)       # artifact_type = unknown
    fit = analyze(report, SPARK)
    assert fit.engine_assumed == "none" and fit.verdict is Verdict.UNKNOWN
    with pytest.raises(PlanError, match="ไม่พบไฟล์ weight"):
        rule_based_plan(report, fit)


# ── task: pipeline_tag ที่เจาะจงชนะ tag ลอย ๆ ────────────────────────────────────────────────────────


def test_a_specific_pipeline_tag_outranks_bare_tags():
    from lmds.inspector.inspect import task_of

    # nvidia/parakeet-tdt-0.6b-v3: tag feature-extraction + pipeline_tag=automatic-speech-recognition → เดิม embed
    assert task_of({"pipeline_tag": "automatic-speech-recognition", "tags": ["feature-extraction", "nemo"]},
                   "nvidia/parakeet-tdt-0.6b-v3") == "other"
    # meta-llama/Prompt-Guard-86M: tag text-embeddings-inference + pipeline_tag=text-classification → เดิม embed
    assert task_of({"pipeline_tag": "text-classification", "tags": ["text-embeddings-inference", "deberta-v2"]},
                   "meta-llama/Prompt-Guard-86M") == "generate", "ยังไม่ปฏิเสธจาก tag — config.json เป็นคนตัดสิน"
    # ไม่มี pipeline_tag: tag ยังใช้ได้เหมือนเดิม (repo ที่คนแปลงเองมักไม่มี pipeline_tag)
    assert task_of({"tags": ["sentence-transformers"]}, "org/some-model") == "embed"
    assert task_of({"pipeline_tag": "feature-extraction", "tags": []}, "org/some-model") == "embed"
    assert task_of({"pipeline_tag": "text-generation", "tags": ["feature-extraction"]}, "org/some-llm") == "generate"
    assert task_of({"tags": []}, "VesNFF/Qwen3-VL-Embedding-8B-GGUF") == "embed"
    assert task_of({"pipeline_tag": "text-classification", "tags": []}, "BAAI/bge-reranker-v2-m3") == "rerank"


def test_the_refusal_for_an_unserved_task_can_be_overridden_with_task():
    """ปฏิเสธจากการจัดประเภทงาน = เดาได้ผิด ต้องมีทางข้าม: `lmds deploy --task embed` ตั้ง report.task หลัง inspect"""
    report = inspect("google-bert/bert-base-uncased")
    assert report.task == "other" and analyze(report, SPARK).verdict is Verdict.UNSUPPORTED
    report.task = "embed"
    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.FITS
    plan = build_plan(report, fit, None)
    assert plan.task == "embed" and plan.runtime.engine is Engine.VLLM


def test_structural_refusals_cannot_be_overridden_with_task():
    report = inspect("stabilityai/sdxl-turbo")
    report.task = "generate"
    assert analyze(report, SPARK).verdict is Verdict.UNSUPPORTED


# ── เทสคู่: หน้าตาคล้ายกันแต่เสิร์ฟได้ ต้องยัง deploy ได้ ────────────────────────────────────────────

SERVABLE = {
    # pipeline_tag=translation แต่เป็น causal LM (HunYuanDenseV1ForCausalLM) — คู่ของ t5-small
    "tencent/Hunyuan-MT-7B": ("generate", Engine.VLLM),
    # pipeline_tag=text-classification + XLMRobertaForSequenceClassification label เดียว — คู่ของ classifier
    "BAAI/bge-reranker-v2-m3": ("rerank", Engine.VLLM),
    # MPNetForMaskedLM แต่ pipeline_tag=sentence-similarity — คู่ของ bert-base-uncased (BertForMaskedLM + fill-mask)
    # และมี pytorch_model.bin / onnx / openvino ข้าง model.safetensors — คู่ของ opt-125m ที่มีแต่ .bin
    "sentence-transformers/all-mpnet-base-v2": ("embed", Engine.VLLM),
    # pipeline_tag=text-to-speech แต่ไฟล์ GGUF เป็นสถาปัตยกรรม LLM (qwen3) — คู่ของ Qwen3-TTS-GGUF
    "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf": ("generate", Engine.LLAMACPP),
    # ไม่มี config.json แต่เป็นรูปแบบ mistral (params.json + consolidated.safetensors) — คู่ของ no-config
    "mistralai/Pixtral-12B-2409": ("generate", Engine.VLLM),
    # GGUF สถาปัตยกรรมที่ไม่อยู่ในรายการ "ไม่ใช่ LLM" (lfm2) — คู่ของ flux/wan
    "LiquidAI/LFM2.5-2.6B-GGUF": ("generate", Engine.LLAMACPP),
}


@pytest.mark.parametrize("repo", sorted(SERVABLE))
def test_servable_lookalikes_still_deploy(repo):
    task, engine = SERVABLE[repo]
    source = repo if repo != "LiquidAI/LFM2.5-2.6B-GGUF" else (
        "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/blob/main/LFM2.5-2.6B-Q4_K_M.gguf")
    report = inspect(source)
    assert report.unsupported_format is None
    assert report.task == task
    fit = analyze(report, SPARK)
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    plan = build_plan(report, fit, None)
    assert plan.runtime.engine is engine and plan.task == task


def test_a_tts_model_that_is_an_llm_is_planned_with_a_warning_not_refused():
    report = inspect("pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf")
    assert report.gguf_architecture == "qwen3"
    assert any("text-to-speech" in w and "ไม่ใช่โมเดล chat ทั่วไป" in w for w in report.warnings)


def test_a_multi_file_audio_gguf_repo_waits_for_the_file_before_deciding():
    """Serveurperso/Qwen3-TTS-GGUF: 24 ไฟล์ ยังไม่เลือก — ตัดสินไม่ได้จนกว่าจะอ่าน header (อาจเป็น LLM ที่เสิร์ฟได้)"""
    report = inspect("Serveurperso/Qwen3-TTS-GGUF")
    assert report.unsupported_format is None and report.selected_gguf is None
    assert any("text-to-speech" in w and "header" in w for w in report.warnings)


def test_mistral_native_repo_reads_its_dimensions_from_params_json():
    report = inspect("mistralai/Pixtral-12B-2409")
    assert [s.filename for s in report.safetensor_shards] == ["consolidated.safetensors"]
    assert report.context_length == 131072
    assert (report.kv_dims.layers, report.kv_dims.kv_heads, report.kv_dims.head_dim) == (40, 8, 128)
    assert any("params.json" in w for w in report.warnings)


def test_an_adapter_file_next_to_a_full_checkpoint_does_not_make_the_repo_an_adapter():
    def add_adapter(fixture):
        fixture["info"]["siblings"] += [{"rfilename": "adapter_model.safetensors", "size": 1000},
                                        {"rfilename": "adapter_config.json", "size": 100}]

    report = inspect("openai-community/gpt2", edits={"openai-community/gpt2": add_adapter})
    assert report.unsupported_format is None
    assert [s.filename for s in report.safetensor_shards] == ["model.safetensors"]


def test_a_transformers_repo_tagged_diffusers_is_not_a_diffusers_pipeline():
    def tag_it(fixture):
        fixture["info"]["tags"].append("diffusers")

    report = inspect("tencent/Hunyuan-MT-7B", edits={"tencent/Hunyuan-MT-7B": tag_it})
    assert report.unsupported_format is None and report.artifact_type is ArtifactType.SAFETENSORS
