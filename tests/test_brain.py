import json

import pytest

from lmds.brain import (
    Confidence,
    DeploymentPlan,
    Engine,
    PlanError,
    build_plan,
    harden_plan,
    rule_based_plan,
    slugify,
    split_flags,
)
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def qwen_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="sha-pinned-123",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB,
        context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
        license="apache-2.0",
    )
    base.update(overrides)
    return ModelReport(**base)


def spark_fit(report):
    return analyze(report, PRESETS["dgx-spark-single"])


def valid_plan_dict(**overrides) -> dict:
    base = {
        "plan_version": 1,
        "model_id": "Qwen/Qwen3-32B",
        "revision": "sha-pinned-123",
        "served_model_name": "qwen3-32b",
        "artifact_type": "safetensors",
        "facts": [{"claim": "artifact เป็น safetensors", "source": "hub-api", "confidence": "verified"}],
        "runtime": {"engine": "vllm", "image_ref": "vllm/vllm-openai:v0.8.5", "rationale": "ตาม matrix"},
        "topology": "single",
        "serving": {"context": 32768, "max_output_tokens": 8192},
    }
    base.update(overrides)
    return base


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.responses.pop(0)


# ---------- slugify / allowlists ----------

def test_slugify():
    assert slugify("Qwen/Qwen3-32B") == "qwen3-32b"
    assert slugify("meta-llama/Llama-3.3-70B-Instruct") == "llama-3-3-70b-instruct"


def test_split_flags_allowlist():
    allowed, needs = split_flags(
        Engine.VLLM,
        ["--kv-cache-dtype=fp8", "--trust-remote-code", "--enable-prefix-caching", "--evil-flag x"],
    )
    assert "--kv-cache-dtype=fp8" in allowed
    assert "--enable-prefix-caching" in allowed
    assert "--trust-remote-code" in needs  # จงใจไม่อยู่ใน allowlist
    assert "--evil-flag x" in needs


# ---------- rule-based ----------

def test_rule_based_plan_safetensors(isolated_config):
    report = qwen_report()
    plan = rule_based_plan(report, spark_fit(report))
    assert plan.runtime.engine is Engine.VLLM
    assert plan.revision == "sha-pinned-123"
    assert plan.serving.context == 32768
    assert plan.generator == "rule-based"
    assert all(f.confidence is Confidence.VERIFIED for f in plan.facts)
    assert plan.tool_calling.enabled is False


def test_rule_based_plan_gguf(isolated_config):
    report = qwen_report(
        artifact_type=ArtifactType.GGUF, weight_bytes=20 * GIB, selected_gguf="m-Q4_K_M.gguf"
    )
    plan = rule_based_plan(report, spark_fit(report))
    assert plan.runtime.engine is Engine.LLAMACPP
    assert plan.selected_gguf == "m-Q4_K_M.gguf"


# ---------- harden ----------

def test_harden_fixes_revision_and_context(isolated_config):
    report = qwen_report()
    fit = spark_fit(report)
    plan = DeploymentPlan.model_validate(
        valid_plan_dict(revision="wrong-sha", serving={"context": 999999, "max_output_tokens": 8192})
    )
    hardened = harden_plan(plan, report, fit)
    assert hardened.revision == "sha-pinned-123"
    assert hardened.serving.context == fit.recommended_context
    assert any("revision" in w for w in hardened.warnings)


def test_harden_forces_engine_to_match_artifact(isolated_config):
    report = qwen_report(artifact_type=ArtifactType.GGUF, weight_bytes=20 * GIB, kv_dims=None)
    plan = DeploymentPlan.model_validate(valid_plan_dict(artifact_type="gguf"))  # engine=vllm ผิด
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime.engine is Engine.LLAMACPP


def test_harden_moves_unknown_flags_to_approval(isolated_config):
    report = qwen_report()
    plan = DeploymentPlan.model_validate(
        valid_plan_dict(
            serving={"context": 16384, "max_output_tokens": 8192,
                     "extra_flags": ["--enable-prefix-caching", "--trust-remote-code", "--weird"]}
        )
    )
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.serving.extra_flags == ["--enable-prefix-caching"]
    assert "--trust-remote-code" in hardened.flags_needing_approval
    assert "--weird" in hardened.flags_needing_approval


def test_harden_forces_parallel_tools_off(isolated_config):
    report = qwen_report()
    plan = DeploymentPlan.model_validate(
        valid_plan_dict(tool_calling={"enabled": True, "parser": "hermes", "parallel": True})
    )
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.tool_calling.parallel is False


def test_harden_replaces_hallucinated_image(isolated_config):
    """เคสจริงจาก gigabyte02: LLM มโน ghcr.io/lmds/llamacpp-ubuntu-rtx → ต้องแทนด้วย image จริง"""
    report = qwen_report(artifact_type=ArtifactType.GGUF, weight_bytes=36 * GIB,
                         selected_gguf="m-Q8.gguf", kv_dims=None)
    plan = DeploymentPlan.model_validate(
        valid_plan_dict(
            artifact_type="gguf",
            runtime={"engine": "llamacpp", "image_ref": "ghcr.io/lmds/llamacpp-ubuntu-rtx", "rationale": "x"},
        )
    )
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime.image_ref == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert any("registry" in w for w in hardened.warnings)


def test_harden_keeps_known_image_with_tag(isolated_config):
    report = qwen_report()
    plan = DeploymentPlan.model_validate(
        valid_plan_dict(runtime={"engine": "vllm", "image_ref": "vllm/vllm-openai:v0.9.2", "rationale": "x"})
    )
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime.image_ref == "vllm/vllm-openai:v0.9.2"  # tag ใดก็ได้ ขอแค่ repo อยู่ใน allowlist


# ---------- build_plan / orchestrator ----------

def test_build_plan_without_provider_uses_rule_based(isolated_config):
    report = qwen_report()
    plan = build_plan(report, spark_fit(report), provider=None)
    assert plan.generator == "rule-based"


def test_build_plan_with_valid_llm_response(isolated_config):
    report = qwen_report()
    provider = FakeProvider([json.dumps(valid_plan_dict())])
    plan = build_plan(report, spark_fit(report), provider)
    assert plan.generator == "llm:fake/fake-1"
    assert plan.revision == "sha-pinned-123"


def test_build_plan_retries_on_invalid_then_succeeds(isolated_config):
    report = qwen_report()
    provider = FakeProvider(["not json at all", json.dumps(valid_plan_dict())])
    plan = build_plan(report, spark_fit(report), provider)
    assert plan.generator == "llm:fake/fake-1"
    assert len(provider.calls) == 2
    assert "failed validation" in provider.calls[1][1]  # feedback ถูกส่งกลับ


def test_build_plan_strips_markdown_fences(isolated_config):
    report = qwen_report()
    provider = FakeProvider(["```json\n" + json.dumps(valid_plan_dict()) + "\n```"])
    plan = build_plan(report, spark_fit(report), provider)
    assert plan.model_id == "Qwen/Qwen3-32B"


def test_build_plan_raises_after_max_attempts(isolated_config):
    report = qwen_report()
    provider = FakeProvider(["bad", "bad", "bad"])
    with pytest.raises(PlanError, match="no-llm"):
        build_plan(report, spark_fit(report), provider)


def test_session_log_written(isolated_config):
    from lmds.config.paths import sessions_dir

    report = qwen_report()
    build_plan(report, spark_fit(report), provider=None)
    logs = list(sessions_dir().glob("plan-*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["outcome"] == "ok"
    assert payload["plan"]["generator"] == "rule-based"


def _asset(**kw):
    from lmds.brain.plan_schema import RuntimeAsset

    base = dict(filename="parser.py", url="https://raw.githubusercontent.com/o/r/main/parser.py")
    base.update(kw)
    return RuntimeAsset(**base)


def test_runtime_assets_always_need_approval(isolated_config):
    """LLM ใส่ runtime_assets มาตรง ๆ ก็ห้ามเข้า bundle เอง — ต้องรออนุมัติเสมอ"""
    report = qwen_report()
    plan = DeploymentPlan.model_validate(valid_plan_dict())
    plan.runtime_assets = [_asset()]

    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime_assets == []
    assert [a.filename for a in hardened.assets_needing_approval] == ["parser.py"]


def test_runtime_asset_from_bad_source_dropped(isolated_config):
    """URL นอก allowlist / ชื่อไฟล์มี path / ไม่ใช่ https → ทิ้งทั้งหมด"""
    report = qwen_report()
    plan = DeploymentPlan.model_validate(valid_plan_dict())
    plan.assets_needing_approval = [
        _asset(filename="evil.py", url="https://evil.example.com/evil.py"),
        _asset(filename="../escape.py", url="https://raw.githubusercontent.com/o/r/main/x.py"),
        _asset(filename="plain.py", url="http://raw.githubusercontent.com/o/r/main/x.py"),
    ]

    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.assets_needing_approval == []
    assert hardened.runtime_assets == []
    assert sum("ตัดไฟล์ runtime" in w for w in hardened.warnings) == 3


def test_apply_asset_approvals_moves_only_approved(isolated_config):
    from lmds.brain import apply_asset_approvals

    report = qwen_report()
    plan = DeploymentPlan.model_validate(valid_plan_dict())
    plan.assets_needing_approval = [_asset(filename="a.py"), _asset(filename="b.py")]
    plan = harden_plan(plan, report, spark_fit(report))

    apply_asset_approvals(plan, ["a.py"])
    assert [a.filename for a in plan.runtime_assets] == ["a.py"]
    assert [a.filename for a in plan.assets_needing_approval] == ["b.py"]
