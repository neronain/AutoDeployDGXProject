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
    # Qwen3 บน vLLM: planner รู้ parser แน่ → เปิดให้เลย (เดิมพิมพ์เป็นคำเตือนแล้วปล่อย None
    # ผู้ใช้ต้องไปพิมพ์เองซึ่งคนไม่รู้ก็ไม่กล้า — 2026-09-04)
    assert plan.tool_calling.enabled is True and plan.tool_calling.parser == "qwen3_xml"
    assert plan.reasoning.enabled is True and plan.reasoning.parser == "qwen3"
    assert any("เปิด tool calling ให้แล้ว" in w for w in plan.warnings)


def test_rule_based_plan_does_not_guess_parsers_for_unknown_families(isolated_config):
    """ตระกูลที่ไม่รู้ → ห้ามเดา (เดาผิด = tool call กลายเป็นข้อความโดยไม่มี error)"""
    # ต้องเป็นตระกูลที่ *ไม่มี* ทั้งกฎและสูตร — Llama มีสูตรในคลัง (llama3_json) จึงใช้ไม่ได้
    report = qwen_report(repo_id="acme/mystery-model-7b")
    plan = rule_based_plan(report, spark_fit(report))
    assert plan.tool_calling.enabled is False and plan.tool_calling.parser is None
    assert plan.reasoning.enabled is False


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


def test_system_prompt_documents_runtime_assets():
    """LLM ต้องรู้ว่ามีช่อง runtime_assets และรู้กติกา (host allowlist / mount point)"""
    from lmds.brain.plan_schema import DeploymentPlan
    from lmds.brain.prompts import build_system_prompt

    prompt = build_system_prompt(DeploymentPlan.model_json_schema())
    assert "runtime_assets" in prompt
    assert "/opt/lmds/plugins" in prompt          # mount point ที่ flag ต้องชี้ไป
    assert "raw.githubusercontent.com" in prompt  # host allowlist
    assert "must approve" in prompt               # บอกว่าผู้ใช้ต้องอนุมัติ


# ── ตรวจว่า image tag มีอยู่จริง ────────────────────────────────────────────────
# LLM เสนอ `vllm/vllm-openai:v0.6.3.ss` ซึ่งไม่มีอยู่จริง · allowlist เดิมตรวจแค่ repo
# bundle จึงผ่าน gate ทุกด่านแล้วไปตายตอนรันด้วย "manifest unknown" (ผู้ใช้เจอจริง)

def test_split_image_ref_handles_every_registry_shape():
    from lmds.brain.registry import split_ref

    assert split_ref("vllm/vllm-openai:v0.6.3.ss") == ("registry-1.docker.io", "vllm/vllm-openai", "v0.6.3.ss")
    assert split_ref("ubuntu") == ("registry-1.docker.io", "library/ubuntu", "latest")
    assert split_ref("ghcr.io/ggml-org/llama.cpp:server-cuda") == ("ghcr.io", "ggml-org/llama.cpp", "server-cuda")
    assert split_ref("nvcr.io/nvidia/vllm:26.05-py3") == ("nvcr.io", "nvidia/vllm", "26.05-py3")


def test_a_tag_that_does_not_exist_is_reported_as_missing(monkeypatch):
    import httpx

    from lmds.brain.registry import SKIP_ENV, tag_exists

    monkeypatch.delenv(SKIP_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if "token" in str(request.url):
            return httpx.Response(200, json={"token": "t"})
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert tag_exists("vllm/vllm-openai:v0.6.3.ss", client=client) is False


def test_a_registry_we_cannot_query_is_not_treated_as_missing(monkeypatch):
    """registry ที่ถามแบบ anonymous ไม่ได้ · เครื่อง air-gapped — ไม่ใช่เหตุผลที่จะห้าม deploy

    เคสนี้ตัดสินได้ก่อนแตะเน็ต (registry ไม่อยู่ในรายการที่ถามแบบ anonymous ได้)

    เดิมใช้ nvcr.io เป็นตัวอย่าง แต่ตั้งแต่ 2026-09-01 NGC ถามได้แล้วผ่าน /proxy_auth
    (ดูเทสถัดไป) — ตัวอย่างของ "ถามไม่ได้" จึงต้องเป็น registry อื่น
    """
    from lmds.brain.registry import SKIP_ENV, tag_exists

    monkeypatch.delenv(SKIP_ENV, raising=False)
    assert tag_exists("registry.example.internal/team/vllm:v1") is None


def test_ngc_tags_are_checked_so_a_nonexistent_one_never_ships(monkeypatch):
    """`nvcr.io/nvidia/vllm:latest` ไม่มีอยู่จริง — ต้องถูกจับตั้งแต่ตอนวางแผน

    เคสจริงที่ลูกค้าเจอ 2026-09-01: แผนเสนอ tag นี้ ผ่านทุกด่าน แล้วไปตายตอน deploy
    ด้วย `manifest for nvcr.io/nvidia/vllm:latest not found` · ตัวตรวจ tag มีอยู่แล้ว
    แต่ nvcr.io ไม่อยู่ใน _ANON_TOKEN จึงคืน None = "ตรวจไม่ได้" แล้วปล่อยผ่าน

    NGC ใช้ /proxy_auth (ไม่ใช่ /token ซึ่งตอบ 401) — ยืนยันกับ registry จริงแล้ว
    """
    from lmds.brain.registry import _ANON_TOKEN

    assert "nvcr.io" in _ANON_TOKEN, "NGC ต้องอยู่ในรายการที่ถามได้"
    assert "proxy_auth" in _ANON_TOKEN["nvcr.io"], "/token ของ NGC ตอบ 401 — ต้องใช้ /proxy_auth"


def test_an_unreachable_registry_is_not_treated_as_missing(monkeypatch):
    import httpx

    from lmds.brain.registry import SKIP_ENV, tag_exists

    monkeypatch.delenv(SKIP_ENV, raising=False)

    def boom(request):
        raise httpx.ConnectError("no network")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    assert tag_exists("vllm/vllm-openai:latest", client=client) is None


def test_harden_replaces_an_image_whose_tag_does_not_exist(isolated_config, monkeypatch):
    """repo ถูกไม่ได้แปลว่า tag มีอยู่ — เคสจริง: `vllm/vllm-openai:v0.6.3.ss`
    ผ่าน gate ทุกด่านแล้วไปตายตอนรันด้วย "manifest unknown"
    """
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: False)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    report = qwen_report()
    plan = DeploymentPlan.model_validate(valid_plan_dict(
        runtime={"engine": "vllm", "image_ref": "vllm/vllm-openai:v0.6.3.ss", "rationale": "x"}))
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime.image_ref != "vllm/vllm-openai:v0.6.3.ss"
    assert any("ไม่มีอยู่จริง" in w for w in hardened.warnings)


def test_a_corrected_image_matches_the_target_machine(isolated_config, monkeypatch):
    """DGX Spark ต้องได้ image ของ NGC ไม่ใช่ upstream — upstream มี manifest arm64
    แต่ไม่ได้ build kernel ให้ SM121 · fallback เดิมคืนค่าเดียวไม่สนเครื่องเป้าหมาย
    """
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: False)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    report = qwen_report()
    plan = DeploymentPlan.model_validate(valid_plan_dict(
        runtime={"engine": "vllm", "image_ref": "vllm/vllm-openai:nope", "rationale": "x"}))
    hardened = harden_plan(plan, report, spark_fit(report))
    assert hardened.runtime.image_ref.startswith("nvcr.io/nvidia/vllm")


def test_no_warning_when_nothing_actually_changed(isolated_config):
    """"ลด max_output_tokens จาก 1,024 เหลือ 1,024" ไม่ได้บอกอะไรใคร — และทำให้
    ผู้ใช้ไล่หาว่าอะไรเปลี่ยนทั้งที่ไม่มีอะไรเปลี่ยน
    """
    report = qwen_report(artifact_type=ArtifactType.GGUF, weight_bytes=8 * GIB,
                         selected_gguf="m-Q8.gguf", kv_dims=None)
    plan = DeploymentPlan.model_validate(valid_plan_dict(
        artifact_type="gguf",
        runtime={"engine": "llamacpp", "image_ref": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                 "rationale": "x"},
        serving={"context": 16384, "max_output_tokens": 1024, "max_num_seqs": 4,
                 "extra_flags": []}))
    hardened = harden_plan(plan, report, spark_fit(report))
    assert not [w for w in hardened.warnings if "max_output_tokens" in w and "1,024 เหลือ 1,024" in w]
