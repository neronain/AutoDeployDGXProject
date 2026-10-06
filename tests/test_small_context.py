"""แผนต้องไม่ขัดกับ bundle ของตัวเอง: output ที่ตั้งต้องเหลือที่ให้คำถามเป็นบวก — ทุก engine ไม่ใช่แค่ llama.cpp

เคสจริง 2026-10-06 (audit · hub 6e2b474): `lmds generate microsoft/phi-2` → context 2,048 · max output 1,024 · bundle ✅
แต่ `client-config` ของ bundle เองตาย "context เล็กเกิน: input budget = -1024" (2,048 − 1,024 − template overhead 2,048)
· fit ตัด budget เป็น 0 เงียบ ๆ และ `_fit_output_into_slots` ข้าม engine ที่ไม่ใช่ llama.cpp
"""

import json
import subprocess

import pytest
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from tests.real_hub import RealHub

runner = CliRunner()
SPARK = PRESETS["dgx-spark-single"]
GIB = 1024**3
PHI2 = "microsoft/phi-2"
OVERHEAD = 2048        # TEMPLATE_OVERHEAD_TOKENS ของ controller


def phi2_with_context(tokens: int) -> RealHub:
    def edit(fixture):
        fixture["files"]["config.json"]["max_position_embeddings"] = tokens
        fixture["files"]["tokenizer_config.json"]["model_max_length"] = tokens

    return RealHub(PHI2, edits={PHI2: edit})


def test_phi2_is_refused_with_the_reason_instead_of_a_bundle_that_rejects_itself():
    report = RealHub(PHI2).inspect(PHI2)
    assert report.context_length == 2048
    fit = analyze(report, SPARK)
    assert any("เล็กกว่าที่ bundle chat ต้องใช้" in n for n in fit.notes), "ตาราง fit ต้องบอกล่วงหน้า ไม่ใช่ตัดเป็น 0 เงียบ ๆ"

    with pytest.raises(PlanError) as err:
        build_plan(report, fit, None)
    said = str(err.value)
    assert "2,048" in said and "client-config" in said and "--task" in said


def test_web_analyze_reports_the_same_refusal(isolated_config, monkeypatch):
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.HfClient", RealHub(PHI2).client)
    with pytest.raises(dep.DeployError) as err:
        dep.analyze(PHI2, target="dgx-spark-single", no_llm=True)
    assert err.value.kind == "input" and "2,048" in err.value.message


@pytest.mark.parametrize("context, output", [(4096, 1024), (3000, 512), (2600, 512)])
def test_a_small_but_usable_context_gets_an_output_that_leaves_room_for_the_question(context, output):
    report = phi2_with_context(context).inspect(PHI2)
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.runtime.engine is Engine.VLLM
    assert plan.serving.context == context and plan.serving.max_output_tokens == output
    assert plan.serving.context - plan.serving.max_output_tokens - OVERHEAD > 0
    if output < 1024:      # 4096: fit ตั้ง 1,024 ให้พอดีอยู่แล้ว · เล็กกว่านั้น harden ต้องลดเองและบอก
        assert any("ลด max_output_tokens" in w for w in plan.warnings)


def test_the_generated_bundle_agrees_with_its_own_client_config(isolated_config, monkeypatch, tmp_path):
    """รัน `client-config` ของ controller ที่ render ออกมาจริง — เดิมตายด้วย input budget ติดลบ"""
    monkeypatch.setattr("lmds.inspector.HfClient", phi2_with_context(4096).client)
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", PHI2, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)])
    assert result.exit_code == 0, result.output

    controller = next(out.glob("*/*-single.sh"))
    done = subprocess.run(["bash", str(controller), "client-config"], capture_output=True, text=True,
                          cwd=controller.parent, timeout=60)
    assert done.returncode == 0, done.stderr
    config = json.loads(done.stdout[done.stdout.index("{"):])
    assert config["max_output_tokens"] == 1024
    assert config["max_input_tokens"] == 4096 - 1024 - OVERHEAD > 0


def test_llamacpp_refuses_too_when_the_model_itself_is_too_small():
    report = ModelReport(repo_id="org/tiny-GGUF", revision_sha="a" * 40, artifact_type=ArtifactType.GGUF,
                         weight_bytes=1 * GIB, selected_gguf="tiny-Q4_K_M.gguf", context_length=2048,
                         kv_dims=KvDims(layers=12, kv_heads=4, head_dim=64))
    with pytest.raises(PlanError, match="2,048"):
        build_plan(report, analyze(report, SPARK), None)


def test_embedding_models_with_a_tiny_context_are_not_refused():
    """mxbai-embed-large-v1 (context 512): ไม่มี output token — กฎนี้เป็นของแผน chat เท่านั้น"""
    repo = "mixedbread-ai/mxbai-embed-large-v1"
    report = RealHub(repo).inspect(repo)
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.task == "embed" and plan.serving.context == 512


def test_a_normal_context_is_left_alone():
    report = phi2_with_context(32768).inspect(PHI2)
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.serving.max_output_tokens == 8192
    assert not any("ลด max_output_tokens" in w for w in plan.warnings)
