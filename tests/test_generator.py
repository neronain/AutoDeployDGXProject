"""เทส Generator — bundle ที่ render ต้องผ่านมาตรฐาน v3.0.0 ทุกข้อ

audit rules ที่เช็คที่นี่สะท้อน audit-controllers.py ของ repo เดิม:
- bash -n ผ่าน
- ไม่มี numeric underscore literal ใน arithmetic
- ไม่มี pattern `| grep -q` (pipefail-unsafe)
- flags ครบตาม controller contract
- ไม่มี secret ฝังในไฟล์
"""

from __future__ import annotations

import re
import subprocess

import pytest
import yaml

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

REQUIRED_FLAGS = [
    "--context",
    "--port",
    "--bind",
    "--advertise-ip",
    "--interface",
    "--client-input",
    "--client-output",
]
REQUIRED_COMMANDS = [
    "download",
    "verify-files",
    "start",
    "stop",
    "restart",
    "status",
    "logs",
    "client-config",
    "network-info",
    "test-text",
]


def safetensors_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="sha-pinned-123",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB,
        shard_count=17,
        context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
        license="apache-2.0",
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def gguf_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF",
        revision_sha="sha-gguf-456",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB,
        context_length=40960,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        gguf_variants=[
            GgufVariant(
                filename="Qwen3-8B-Q4_K_M.gguf",
                size_bytes=5 * GIB,
                sha256="a" * 64,
            )
        ],
        has_chat_template=True,
        license="apache-2.0",
    )
    base.update(overrides)
    return ModelReport(**base)


def make_bundle(report, target="dgx-spark-single", tmp_path=None):
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None)
    return render_bundle(plan, report, fit, tmp_path), plan, fit


def audit_script(text: str) -> list[str]:
    problems = []
    if re.search(r"\(\(\s*[^)]*\b\d+_\d+", text):
        problems.append("numeric underscore ใน arithmetic")
    if re.search(r"\|\s*grep\s+-q", text):
        problems.append("pipefail-unsafe: | grep -q")
    for flag in REQUIRED_FLAGS:
        if flag + ")" not in text:
            problems.append(f"ขาด flag {flag}")
    for command in REQUIRED_COMMANDS:
        if f"{command})" not in text:
            problems.append(f"ขาดคำสั่ง {command}")
    return problems


@pytest.mark.parametrize("kind", ["vllm", "llamacpp"])
def test_generated_controller_passes_bash_n(isolated_config, tmp_path, kind):
    report = safetensors_report() if kind == "vllm" else gguf_report()
    bundle, _, _ = make_bundle(report, tmp_path=tmp_path)
    result = subprocess.run(["bash", "-n", str(bundle.controller)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n ล้มเหลว:\n{result.stderr}"


@pytest.mark.parametrize("kind", ["vllm", "llamacpp"])
def test_generated_controller_passes_audit_rules(isolated_config, tmp_path, kind):
    report = safetensors_report() if kind == "vllm" else gguf_report()
    bundle, _, _ = make_bundle(report, tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert audit_script(text) == []
    assert "set -Eeuo pipefail" in text


def test_vllm_controller_pins_revision(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'MODEL_REVISION="sha-pinned-123"' in text
    assert "--revision" in text


def test_llamacpp_controller_exact_verification(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert f'"{5 * GIB}"' in text  # exact size ใน EXPECTED_SIZES
    assert f'"{"a" * 64}"' in text  # SHA-256 ใน EXPECTED_SHAS
    assert 'magic="$(head -c 4' in text
    assert "--jinja" in text  # chat template ฝังใน GGUF


def test_llamacpp_without_selected_gguf_rejected(isolated_config, tmp_path):
    report = gguf_report(selected_gguf=None)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.selected_gguf = None
    with pytest.raises(ValueError, match="GGUF"):
        render_bundle(plan, report, fit, tmp_path)


def test_bundle_contains_delivery_contract_files(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    names = {f.name for f in bundle.files}
    assert "README.md" in names
    assert "MODEL_PROFILE.yaml" in names
    assert "SPECIAL_FILES.md" in names  # มี gguf → ต้องมี
    assert bundle.controller.name == "qwen3-8b-gguf-single.sh"
    assert bundle.controller.stat().st_mode & 0o111  # executable


def test_model_profile_yaml_valid_and_complete(isolated_config, tmp_path):
    bundle, plan, fit = make_bundle(safetensors_report(), tmp_path=tmp_path)
    profile = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["model"]["revision"] == "sha-pinned-123"
    assert profile["runtime"]["engine"] == "vllm"
    assert profile["serving"]["context"] == plan.serving.context
    assert profile["validation"] == {"static": True, "hardware": False}
    assert profile["target"]["name"] == fit.target_name


def test_readme_has_delivery_contract_sections(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    for section in [
        "Requirements",
        "Persistent paths",
        "Runtime pin",
        "First-run",
        "Conflict shutdown",
        "Start after reboot",
        "Status & logs",
        "Context tuning",
        "Security",
        "Validation scope",
    ]:
        assert section in readme, f"README ขาด section: {section}"
    assert "static-validated" in readme
    assert "sha-pinned-123" in readme


def test_approved_flags_rendered_but_unapproved_not(isolated_config, tmp_path):
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.serving.extra_flags = ["--enable-prefix-caching"]
    plan.flags_needing_approval = ["--trust-remote-code"]
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "--enable-prefix-caching" in text
    assert "--trust-remote-code" not in text  # ห้ามโผล่ในสคริปต์
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "--trust-remote-code" in readme  # แต่ต้องแจ้งใน README


def test_no_secrets_in_bundle(isolated_config, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETTOKEN123456789")
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    for file_path in bundle.files:
        content = file_path.read_text(encoding="utf-8")
        assert "hf_SECRETTOKEN123456789" not in content
    # token ใช้ผ่าน env เท่านั้น
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'HF_TOKEN:+' in text


def test_multi_gpu_target_gets_tensor_parallel(isolated_config, tmp_path):
    report = safetensors_report(weight_bytes=30 * GIB)
    fit = analyze(report, PRESETS["rtx-pro-4000-dual"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "TENSOR_PARALLEL_SIZE" in text
    assert "--tensor-parallel-size" in text


def test_llamacpp_spark_uses_native_build_mode(isolated_config, tmp_path):
    """Spark (unified): ไม่มี docker image ทางการ → native source build + prepare-runtime"""
    bundle, _, _ = make_bundle(gguf_report(), target="dgx-spark-single", tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'RUNTIME_MODE:-native' in text
    assert "prepare-runtime" in text
    assert "121a-real" in text
    assert "cmake" in text


def test_llamacpp_rtx_uses_docker_mode(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(weight_bytes=5 * GIB), target="rtx-pro-4000", tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'RUNTIME_MODE:-docker' in text
    assert "ghcr.io/ggml-org/llama.cpp" in text


def test_split_gguf_all_parts_in_controller(isolated_config, tmp_path):
    from lmds.inspector.report import GgufPart

    report = gguf_report(
        selected_gguf="BF16/m-BF16-00001-of-00002.gguf",
        weight_bytes=62 * GIB,
        gguf_variants=[
            GgufVariant(
                filename="BF16/m-BF16-00001-of-00002.gguf",
                size_bytes=62 * GIB,
                parts=[
                    GgufPart(filename="BF16/m-BF16-00001-of-00002.gguf", size_bytes=50 * GIB, sha256="a" * 64),
                    GgufPart(filename="BF16/m-BF16-00002-of-00002.gguf", size_bytes=12 * GIB, sha256="b" * 64),
                ],
            )
        ],
    )
    bundle, _, _ = make_bundle(report, tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "m-BF16-00001-of-00002.gguf" in text
    assert "m-BF16-00002-of-00002.gguf" in text  # ทุก part ถูก download/verify
    assert f'"{50 * GIB}"' in text and f'"{12 * GIB}"' in text
    assert ("a" * 64) in text and ("b" * 64) in text
    result = subprocess.run(["bash", "-n", str(bundle.controller)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gated_repo_noted_in_readme(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(gated=True), tmp_path=tmp_path)
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "HF_TOKEN" in readme
    assert "gated" in readme
