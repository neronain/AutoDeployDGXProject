"""context ที่เสนอต้องไม่เกินที่ *ตัวโมเดล* รับได้ — ไม่รู้เพดานของโมเดลต้องบอกว่าเดา ไม่ใช่เสนอเท่าที่ RAM เหลือ

เคสจริง 2026-10-06 (audit · hub 6e2b474): inspector อ่านแค่ max_position_embeddings / max_sequence_length / n_positions
และ fit ใช้ `limit = … if native else max_context_raw` พร้อมโน้ต "ค่าสูงสุดที่หน่วยความจำและตัวโมเดลรับไหว":
- `tiiuae/falcon-7b-instruct` (config ไม่มีคีย์ context · tokenizer_config.model_max_length = 2048) → เสนอ 131,072
- `google/vit-base-patch16-224` → 1,048,576 · `THUDM/glm-4-9b-chat` เก็บไว้ที่ `seq_length`
- `mlx-community/gemma-3-12b-it-bf16`: text_config ไม่มี max_position_embeddings · model_max_length = 1e30
"""

import json

import pytest
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from tests.real_hub import RealHub

runner = CliRunner()
SPARK = PRESETS["dgx-spark-single"]
GIB = 1024**3

FALCON = "tiiuae/falcon-7b-instruct"
GLM4 = "THUDM/glm-4-9b-chat"
UNKNOWN = "mlx-community/gemma-3-12b-it-bf16"
HUNYUAN_MT = "tencent/Hunyuan-MT-7B"


def inspect(repo):
    return RealHub(repo).inspect(repo)


def test_falcon_gets_its_limit_from_the_tokenizer_config_not_from_free_memory():
    report = inspect(FALCON)
    assert report.context_length == 2048
    assert any("tokenizer_config.json" in w for w in report.warnings), "ต้องบอกว่าค่านี้มาจากแหล่งรอง"

    fit = analyze(report, SPARK)
    assert fit.recommended_context == 2048, "เดิมเสนอ 131,072 — Falcon รับได้ 2,048"
    assert fit.context_is_guess is False


def test_glm4_keeps_its_limit_in_seq_length():
    report = inspect(GLM4)
    assert report.context_length == 131072
    assert analyze(report, SPARK).recommended_context == 131072


def test_a_config_limit_wins_over_the_tokenizer_one():
    """Hunyuan-MT-7B: config.json บอก 32,768 · tokenizer_config.model_max_length บอก 262,144 — config คือของโมเดล"""
    report = inspect(HUNYUAN_MT)
    assert report.context_length == 32768
    assert not any("tokenizer_config.json" in w for w in report.warnings)


def test_an_unset_tokenizer_limit_is_not_a_limit():
    """model_max_length = int(1e30) คือ "ไม่ได้ตั้ง" — ถ้าเอามาใช้ context จะกลายเป็นเท่าที่ RAM เหลือเหมือนเดิม"""
    assert inspect(UNKNOWN).context_length is None


def test_unknown_native_context_is_capped_and_called_a_guess():
    report = inspect(UNKNOWN)
    fit = analyze(report, SPARK)

    assert fit.verdict is Verdict.FITS
    assert fit.recommended_context == 16384, "เดิมเสนอ 131,072 จากหน่วยความจำที่เหลือ"
    assert fit.context_is_guess is True
    assert fit.max_safe_context > 16384, "เพดานของเครื่องยังอยู่ ให้คนที่รู้ค่าจริงจาก model card ตั้งเองได้"
    said = " ".join(fit.notes)
    assert "ค่าเดา" in said and "--context" in said
    assert "ตัวโมเดลรับไหว" not in said, "ห้ามอ้างว่าโมเดลรับไหวเมื่อไม่รู้เพดานของมัน"

    plan = build_plan(report, fit, None)
    assert plan.serving.context == 16384


def test_a_gguf_without_a_context_key_gets_the_same_cap():
    report = ModelReport(repo_id="org/model-GGUF", revision_sha="a" * 40, artifact_type=ArtifactType.GGUF,
                         weight_bytes=8 * GIB, selected_gguf="m-Q4_K_M.gguf",
                         kv_dims=KvDims(layers=32, kv_heads=8, head_dim=128))
    fit = analyze(report, SPARK)
    assert fit.recommended_context == 16384 and fit.context_is_guess is True


def test_known_native_context_is_untouched():
    report = ModelReport(repo_id="org/model", revision_sha="a" * 40, artifact_type=ArtifactType.SAFETENSORS,
                         weight_bytes=8 * GIB, context_length=262144, kv_dims=KvDims(layers=32, kv_heads=8, head_dim=128))
    fit = analyze(report, SPARK)
    assert fit.recommended_context == 262144 and fit.context_is_guess is False
    assert any("ตัวโมเดลรับไหว" in n for n in fit.notes)


@pytest.mark.parametrize("config, expected", [
    ({"max_position_embeddings": 40960}, 40960),
    ({"n_positions": 1024}, 1024),                       # GPT-2 / T5
    ({"seq_length": 131072}, 131072),                    # ChatGLM / GLM-4
    ({"n_ctx": 2048}, 2048),
    ({"max_seq_len": 8192}, 8192),                       # MPT / mistral params
    ({"max_sequence_length": 4096}, 4096),
    ({"max_position_embeddings": 10**30}, None),         # ไม่ใช่เพดานจริง
    ({"max_position_embeddings": True}, None),
    ({"max_position_embeddings": 0, "seq_length": 512}, 512),
    ({"model_type": "mystery"}, None),
])
def test_which_config_keys_carry_the_native_context(config, expected):
    from lmds.inspector.inspect import native_context_from_config

    assert native_context_from_config(config) == expected


def test_cli_plan_carries_the_capped_context(isolated_config, monkeypatch):
    fake = RealHub(UNKNOWN, FALCON)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    result = runner.invoke(app, ["plan", UNKNOWN, "--target", "dgx-spark-single", "--no-llm", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["serving"]["context"] == 16384

    result = runner.invoke(app, ["inspect", FALCON, "--target", "dgx-spark-single", "--json"])
    assert json.loads(result.stdout)["fit"][0]["recommended_context"] == 2048
