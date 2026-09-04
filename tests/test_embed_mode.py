"""โหมด embedding — โมเดลที่เสิร์ฟ /v1/embeddings ไม่ใช่ chat

ผู้ใช้ 2026-09-04: "ทำโหมด embedding ใน LMDS ให้ด้วยเลย" · เคสจริง: VesNFF/Qwen3-VL-Embedding-8B-GGUF → dgx-spark03
"""

from __future__ import annotations

import subprocess

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine, PlanError
from lmds.brain.rulebased import rule_based_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.inspect import task_of
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


# ── ตรวจจับจาก repo ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("info, repo, expected", [
    ({"pipeline_tag": "feature-extraction", "tags": []}, "Qwen/Qwen3-Embedding-0.6B", "embed"),
    ({"pipeline_tag": "sentence-similarity", "tags": []}, "google/embeddinggemma-300m", "embed"),
    ({"pipeline_tag": None, "tags": ["sentence-transformers", "gguf"]}, "BAAI/bge-m3", "embed"),
    ({"pipeline_tag": None, "tags": ["gguf"]}, "VesNFF/Qwen3-VL-Embedding-8B-GGUF", "embed"),  # GGUF ที่คนแปลงเอง ไม่มี tag
    ({"pipeline_tag": "text-generation", "tags": ["conversational"]}, "Qwen/Qwen3-8B", "generate"),
    ({"pipeline_tag": None, "tags": []}, "unsloth/Muse-Glimmer-30B-GGUF", "generate"),
])
def test_task_is_read_from_pipeline_tags_then_name(info, repo, expected):
    assert task_of(info, repo) == expected


# ── แผน ─────────────────────────────────────────────────────────────────────────

def _gguf_embed_report():
    return ModelReport(
        repo_id="VesNFF/Qwen3-VL-Embedding-8B-GGUF", revision_sha="sha", task="embed",
        artifact_type=ArtifactType.GGUF, weight_bytes=int(8.7 * GIB),
        selected_gguf="Qwen3-VL-Embedding-8B-Q8_0.gguf", architecture="qwen3vl",
        context_length=32768, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        gguf_variants=[], tags=["gguf"],
    )


def _st_embed_report():
    return ModelReport(
        repo_id="Qwen/Qwen3-Embedding-4B", revision_sha="sha", task="embed",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(8 * GIB),
        architecture="Qwen3ForCausalLM", context_length=32768,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
    )


def test_embedding_plan_has_no_chat_machinery():
    report = _gguf_embed_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    assert plan.task == "embed"
    assert plan.runtime.engine is Engine.LLAMACPP
    assert not plan.tool_calling.enabled and plan.tool_calling.parser is None
    assert not plan.reasoning.enabled
    assert any("embedding" in w for w in plan.warnings)


def test_safetensors_embedding_goes_to_vllm_even_when_sglang_was_asked():
    report = _st_embed_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit, Engine.SGLANG)
    assert plan.runtime.engine is Engine.VLLM and plan.task == "embed"


def test_embedding_refuses_a_stacked_target():
    report = _st_embed_report()
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    with pytest.raises(PlanError):
        rule_based_plan(report, fit, None)


def test_the_planner_llm_cannot_turn_an_embedding_model_into_a_chat_model():
    """task เป็นข้อเท็จจริงจาก repo — แผนจาก LLM ที่เปิด tool calling ให้ต้องถูกปิด"""
    from lmds.brain.orchestrator import harden_plan

    report = _st_embed_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit, None)
    plan.task = "generate"
    plan.tool_calling.enabled = True
    plan.tool_calling.parser = "hermes"
    hardened = harden_plan(plan, report, fit)
    assert hardened.task == "embed" and not hardened.tool_calling.enabled


# ── controller ───────────────────────────────────────────────────────────────────

def _render(report, tmp_path):
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    script = next(bundle.directory.glob("*-single.sh"))
    return plan, bundle, script, script.read_text(encoding="utf-8")


def test_llamacpp_embedding_controller(tmp_path):
    plan, bundle, script, text = _render(_gguf_embed_report(), tmp_path)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert 'POOLING="${POOLING:-last}"' in text, "Qwen → last-token pooling"
    assert 'SERVER_ARGS+=(--embedding --pooling "$POOLING" --batch-size "$EMBED_UBATCH" --ubatch-size "$EMBED_UBATCH")' in text
    assert "test-embed)" in text and "/v1/embeddings" in text
    assert "  test-text)    test_text ;;" not in text, "โมเดล embedding ไม่มี chat ให้ทดสอบ"
    assert '"task": "embed"' in text
    # dispatch จริง: test-text ต้องบอกให้ไปใช้ test-embed ไม่ใช่ยิง chat แล้วงง
    done = subprocess.run(["bash", str(script), "test-text"], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert done.returncode == 2 and "test-embed" in done.stderr
    profile = (bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    assert "task: embed" in profile and "pooling: last" in profile


def test_vllm_embedding_controller(tmp_path):
    plan, bundle, script, text = _render(_st_embed_report(), tmp_path)
    assert plan.runtime.engine is Engine.VLLM
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert "serve_args+=(--runner pooling --convert embed)" in text
    assert "test-embed)" in text and "  test-text)    test_text ;;" not in text
    assert "--tool-call-parser" not in text.split("serve_args+=(--runner pooling")[0].split("local serve_args=(")[-1]


def test_chat_models_are_untouched(tmp_path):
    report = ModelReport(
        repo_id="unsloth/Muse-Glimmer-30B-GGUF", revision_sha="sha", artifact_type=ArtifactType.GGUF,
        weight_bytes=int(30.1 * GIB), selected_gguf="Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        context_length=131072, kv_dims=KvDims(layers=52, kv_heads=2, head_dim=128),
    )
    plan, bundle, script, text = _render(report, tmp_path)
    assert plan.task == "generate"
    assert "--embedding" not in text and "test-embed)" not in text
    assert "  test-text)    test_text ;;" in text and '"task": "generate"' in text


def test_pooling_follows_the_model_family():
    from lmds.generator.renderer import embed_pooling_for

    def rep(repo, arch=""):
        return ModelReport(repo_id=repo, revision_sha="s", architecture=arch or None)

    assert embed_pooling_for(rep("Qwen/Qwen3-Embedding-0.6B", "Qwen3ForCausalLM")) == "last"
    assert embed_pooling_for(rep("BAAI/bge-m3", "XLMRobertaModel")) == "cls"
    assert embed_pooling_for(rep("google/embeddinggemma-300m", "Gemma3TextModel")) == "mean"


def test_feature_summary_shows_embedding():
    from lmds.fleet.manager import feature_summary

    assert feature_summary({"features": {"embedding": {"pooling": "last"}}}) == "embedding (last)"
    assert feature_summary({"features": {"embedding": None}}) == "text"
