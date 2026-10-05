"""checkpoint รูปแบบ MLX ต้องถูกปฏิเสธก่อนดาวน์โหลด — ไม่ใช่ถูกส่งให้ vLLM เพราะไฟล์ลงท้าย .safetensors

เคสจริง 2026-10-05 (hub 965eb56): `lmds inspect Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` ตอบ
Artifact=safetensors · "dgx-spark-single (vllm) ✅ fits" และ `lmds plan … --no-llm` เลือก engine vllm ด้วยเหตุผล
"safetensors→vLLM" · vLLM โหลด weight ที่ quantize แบบ MLX ไม่ได้ — `deploy` + `node push --download` จะโหลด
113 GB แล้วล้มตอน start

metadata ข้างล่างคือของจริงจาก Hub วันนั้น (Hub redirect ชื่อ Vontra/… ไป TensorFold/…) ตัดเฉพาะส่วนที่ไม่เกี่ยว:
config.json เหลือคีย์ที่ inspector อ่าน · index ของ safetensors ย่อเหลือ tensor ละตัวต่อ shard
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan, rule_based_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector import ArtifactType, HfClient, inspect_model
from lmds.inspector.formats import mlx_evidence
from lmds.resolver import parse_source

runner = CliRunner()

REPO = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
SHA = "2b170fa6309d5d1ee380b35636075fac7945f286"
BASE_MODEL = "Qwen/Qwen3.8-Flash-Next"

SHARD_SIZES = [
    5283195887, 5250035137, 5250035154, 5250035160, 5250035164, 5250035184, 5067850140, 5337131618,
    5337131558, 4970130103, 5341257482, 4967968541, 5337131651, 5337131799, 4970130195, 5341257440,
    4967968481, 5337131695, 5337131801, 4970130135, 5341257436, 3715570974,
]
SHARDS = [f"model-{i:05d}-of-00022.safetensors" for i in range(1, 23)]

MODEL_INFO = {
    "id": "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "sha": SHA,
    "private": False,
    "gated": False,
    "pipeline_tag": "image-text-to-text",
    "library_name": "mlx",
    "tags": [
        "mlx", "safetensors", "qwen4_exp", "mlx-vlm", "omlx", "mtp", "speculative-decoding", "qwen", "qwen3.8",
        "qwen4-exp", "mixture-of-experts", "vision-language", "quantized", "apple-silicon", "4-bit",
        "image-text-to-text", "conversational", "base_model:Qwen/Qwen3.8-Flash-Next",
        "base_model:quantized:Qwen/Qwen3.8-Flash-Next", "license:other", "region:us",
    ],
    "cardData": {
        "library_name": "mlx", "license": "other", "license_name": "qwen-community-1.0",
        "base_model": BASE_MODEL, "base_model_relation": "quantized", "pipeline_tag": "image-text-to-text",
    },
    "safetensors": {"parameters": {"U32": 179484224000, "BF16": 515757424, "I64": 35}, "total": 179999981459},
    "siblings": [
        {"rfilename": "README.md", "size": 10306},
        {"rfilename": "chat_template.jinja", "size": 8952},
        {"rfilename": "config.json", "size": 5361},
        {"rfilename": "model.safetensors.index.json", "size": 420813},
        {"rfilename": "tokenizer.json", "size": 12809320},
        {"rfilename": "tokenizer_config.json", "size": 17928},
        *[{"rfilename": name, "size": size, "lfs": {"size": size}} for name, size in zip(SHARDS, SHARD_SIZES, strict=True)],
    ],
}

CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    # mlx-lm เขียนทั้งสองคีย์ — ไม่มี quant_method/quant_algo/format กำกับเหมือน quantizer ฝั่ง transformers
    "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
    "quantization_config": {"group_size": 32, "bits": 4, "mode": "affine"},
    "text_config": {
        "model_type": "qwen4_exp_text", "max_position_embeddings": 262144, "hidden_size": 2560,
        "num_hidden_layers": 48, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
        "full_attention_interval": 4, "num_experts": 512, "num_experts_per_tok": 10,
    },
    "vision_config": {"model_type": "qwen4_exp_vision"},
}

FILES = {
    "config.json": json.dumps(CONFIG),
    "model.safetensors.index.json": json.dumps({"weight_map": {f"layer.{i}": name for i, name in enumerate(SHARDS)}}),
    "chat_template.jinja": "{%- for message in messages %}{{ message.content }}{%- endfor %}",
}


class FakeHub:
    """Hub ปลอมที่จดทุกพาธที่ถูกขอ — ใช้ยืนยันว่าไม่มีใครแตะไฟล์ weight"""

    def __init__(self, info=None, files=None):
        self.info = info or MODEL_INFO
        self.files = FILES if files is None else files
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=self.info)
        name = request.url.path.split(f"/{SHA}/")[-1]
        if name in self.files:
            return httpx.Response(200, content=self.files[name].encode())
        return httpx.Response(404)

    def client(self, token=None) -> HfClient:
        return HfClient(token=token, client=httpx.Client(transport=httpx.MockTransport(self.handler)))

    @property
    def weight_requests(self) -> list[str]:
        return [p for p in self.paths if p.endswith(".safetensors") or p.endswith(".gguf")]


@pytest.fixture
def hub(isolated_config, monkeypatch):
    """CLI กับหน้าเว็บสร้าง HfClient เอง — สลับให้ชี้ Hub ปลอม แล้วให้ inspect_model ตัวจริงทำงานทั้งเส้น"""
    fake = FakeHub()
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    return fake


def mlx_report(info=None, files=None, repo=REPO):
    fake = FakeHub(info, files)
    return inspect_model(parse_source(repo), fake.client())


def flat(text: str) -> str:
    """rich ตัดบรรทัดตามความกว้างจอ — เทียบข้อความโดยไม่ขึ้นกับว่าตัดตรงไหน"""
    return "".join(text.split())


# ── inspect: ชนิดไฟล์ยังเป็น safetensors (ตามนามสกุล) แต่ต้องถูกหมายว่า MLX ───────────────────────────


def test_the_real_repo_is_marked_as_mlx_with_the_evidence_that_says_so():
    report = mlx_report()
    assert report.artifact_type is ArtifactType.SAFETENSORS
    assert report.library_name == "mlx"
    assert report.unsupported_format == "mlx"
    assert report.quantization == "mlx-4bit", "เดิมรายงานแค่ 'quantized' ซึ่งไม่บอกว่าเป็นของ MLX"
    seen = " ".join(report.unsupported_evidence)
    assert "library_name=mlx" in seen and "tag mlx" in seen and "bits=4 group_size=32" in seen
    assert report.weight_bytes == sum(SHARD_SIZES)


def test_fit_stops_claiming_the_model_fits_under_vllm():
    """ตัวเลขถูกทุกตัว (105 GiB < budget 113 GB) แต่ "fits (vllm)" คือสิ่งที่พาคนไปโหลด 113 GB ฟรี"""
    fit = analyze(mlx_report(), PRESETS["dgx-spark-single"])
    assert fit.verdict is Verdict.UNSUPPORTED
    assert fit.engine_assumed != "vllm"
    assert fit.recommended_context is None and fit.weights_gb is None
    assert any(BASE_MODEL in alt for alt in fit.alternatives)


# ── plan: decision matrix ต้องไม่ส่ง MLX ไปให้ engine ไหนเลย ─────────────────────────────────────────


@pytest.mark.parametrize("engine", [None, Engine.VLLM, Engine.SGLANG])
def test_the_decision_matrix_refuses_instead_of_choosing_vllm(engine):
    """--engine ก็ช่วยไม่ได้ — ไม่มี engine ไหนโหลด weight แบบ MLX"""
    report = mlx_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    for planner in (rule_based_plan, lambda r, f, e: build_plan(r, f, None, engine=e)):
        with pytest.raises(PlanError) as err:
            planner(report, fit, engine)
        said = str(err.value)
        assert "MLX" in said, "ต้องบอกสาเหตุ"
        assert "GGUF" in said and "NVFP4" in said and BASE_MODEL in said, "ต้องชี้ไปรุ่นที่ใช้แทนได้"


def test_the_llm_is_never_asked_to_plan_an_mlx_checkpoint():
    class Provider:
        name, model = "fake", "fake"

        def complete_json(self, system, user):
            raise AssertionError("ไม่ควรเรียก LLM — ไม่มีอะไรให้วางแผน")

    report = mlx_report()
    with pytest.raises(PlanError):
        build_plan(report, analyze(report, PRESETS["dgx-spark-single"]), Provider())


# ── CLI: inspect บอกชัด · plan/generate/deploy หยุดก่อนมีอะไรถูกเขียนหรือโหลด ─────────────────────────


def test_cli_inspect_says_unsupported_and_shows_no_fit_table(hub):
    result = runner.invoke(app, ["inspect", REPO, "--target", "dgx-spark-single"])
    assert result.exit_code == 0, "inspect สำเร็จ — แค่คำตอบคือ 'ใช้ไม่ได้'"
    said = flat(result.output)
    assert "MLX" in said and "ไม่รองรับ" in said
    assert "GGUF" in said and "NVFP4" in said and BASE_MODEL in said
    assert "FitAnalysis" not in said and "fits" not in said
    assert not hub.weight_requests


def test_cli_inspect_json_carries_the_verdict_for_scripts(hub):
    result = runner.invoke(app, ["inspect", REPO, "--target", "dgx-spark-single", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["model"]["unsupported_format"] == "mlx"
    assert [f["verdict"] for f in payload["fit"]] == ["unsupported"]


@pytest.mark.parametrize("command", ["plan", "generate", "deploy"])
def test_cli_refuses_before_anything_is_written_or_downloaded(hub, tmp_path, command):
    out = tmp_path / "bundles"
    args = [command, REPO, "--target", "dgx-spark-single", "--no-llm"]
    args += ["--json"] if command == "plan" else ["--output", str(out)]
    args += ["--yes"] if command == "deploy" else []
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert not result.stdout.strip().startswith("{"), "ห้ามมี plan ออกมาให้ script เอาไปใช้ต่อ"
    said = flat(result.output)
    assert "MLX" in said and "GGUF" in said and "NVFP4" in said and BASE_MODEL in said
    assert not out.exists(), "ไม่มี bundle = ไม่มี controller ให้ใครกด download"
    assert not hub.weight_requests


# ── หน้าเว็บ: ลูกค้าส่วนใหญ่ deploy จากตรงนี้ ────────────────────────────────────────────────────────


def test_web_analyze_refuses_with_alternatives(hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        dep.analyze(REPO, target="dgx-spark-single", no_llm=True)
    assert err.value.kind == "unsupported"
    assert "MLX" in err.value.message
    assert any(BASE_MODEL in alt for alt in err.value.extra["alternatives"])
    assert not hub.weight_requests


# ── เกณฑ์: สัญญาณเดียวไม่พอทั้งสองทิศ ────────────────────────────────────────────────────────────────


def test_mlx_weights_published_under_library_transformers_are_still_caught():
    """lmstudio-community/Qwen3.8-27B-MLX-4bit (Hub 2026-10-05): library_name=transformers ทั้งที่เป็น MLX 4-bit"""
    info = {**MODEL_INFO, "library_name": "transformers", "tags": ["transformers", "safetensors", "mlx"],
            "cardData": {"library_name": "transformers"}}
    files = {**FILES, "config.json": json.dumps({
        **CONFIG, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}})}
    report = mlx_report(info, files, repo="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    assert report.unsupported_format == "mlx"


def test_unquantized_mlx_conversion_is_caught_by_library_name():
    """mlx-community/Qwen3-4B-bf16: ไม่มีบล็อก quantization เลย — เหลือ library_name เป็นหลักฐาน"""
    plain = {k: v for k, v in CONFIG.items() if not k.startswith("quantization")}
    report = mlx_report(files={**FILES, "config.json": json.dumps(plain)}, repo="mlx-community/Qwen3-4B-bf16")
    assert report.unsupported_format == "mlx"
    assert report.quantization is None


@pytest.mark.parametrize("quant", [
    {"quant_method": "gptq", "bits": 4, "group_size": 128},          # GPTQ ก็มี bits + group_size
    {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "zero_point": True},
    {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"},
])
def test_transformers_quantizers_are_not_mistaken_for_mlx(quant):
    """ปฏิเสธผิดตัว = ขวางโมเดลที่ vLLM รันได้จริง โดยไม่มีทางข้าม"""
    evidence, decisive = mlx_evidence("org/Qwen3-32B-AWQ", "transformers", ["transformers", "safetensors"],
                                      {"quantization_config": quant})
    assert not decisive and not evidence


def test_an_mlx_tag_alone_on_a_transformers_repo_warns_but_does_not_block():
    """tag mlx บอกแค่ "ใช้กับ MLX ได้" — repo transformers ปกติก็ติดได้ ต้องยัง deploy ได้"""
    info = {**MODEL_INFO, "library_name": "transformers", "tags": ["transformers", "safetensors", "mlx"],
            "cardData": {"library_name": "transformers"}}
    plain = {k: v for k, v in CONFIG.items() if not k.startswith("quantization")}
    report = mlx_report(info, {**FILES, "config.json": json.dumps(plain)}, repo="org/Qwen3.8-Flash-Next")
    assert report.unsupported_format is None
    assert any("MLX" in w for w in report.warnings)
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    assert fit.verdict is not Verdict.UNSUPPORTED
    assert rule_based_plan(report, fit).runtime.engine is Engine.VLLM


def test_a_gguf_repo_tagged_mlx_still_goes_to_llamacpp():
    """ทางของ GGUF ไม่เกี่ยวกับ MLX เลย — llama.cpp อ่านไฟล์ .gguf ไม่ได้อ่าน safetensors"""
    # สอง variant = inspector ยังไม่เปิด header ของไฟล์ไหน (รอผู้ใช้เลือก) — เทสนี้จึงไม่ต้องมี GGUF จริง
    info = {**MODEL_INFO, "siblings": [{"rfilename": f"model-{q}.gguf", "size": 1000, "lfs": {"size": 1000}}
                                       for q in ("Q4_K_M", "Q8_0")]}
    report = mlx_report(info, files={})
    assert report.artifact_type is ArtifactType.GGUF
    assert report.unsupported_format is None
