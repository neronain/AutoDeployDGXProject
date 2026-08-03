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


@pytest.mark.parametrize("target", ["dgx-spark-single", "rtx-pro-4000-dual"])
def test_no_broken_line_continuation_in_vllm_controller(isolated_config, tmp_path, target):
    """regression เคส gigabyte02: jinja block ทิ้งบรรทัดว่างกลาง docker run ที่ต่อด้วย backslash"""
    import re

    report = safetensors_report(weight_bytes=30 * GIB)
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert re.search(r"\\\n[ \t]*\n", text) is None
    assert '"${serve_args[@]}"' in text  # args array ไม่ใช่ line continuation


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
    # ติดตั้ง build deps อัตโนมัติ — ผู้ใช้ไม่ต้อง apt install เอง
    assert "install_build_dependencies" in text
    assert "apt-get install -y" in text
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "prepare-runtime" in readme  # first-run ต้องบอกขั้นนี้ตั้งแต่แรก
    assert "ติดตั้งให้อัตโนมัติ" in readme


def test_llamacpp_client_budget_accounts_parallel_slots(isolated_config, tmp_path):
    """llama.cpp แบ่ง ctx ให้ทุก slot — client budget ต้องคิดจาก context ต่อ slot"""
    bundle, _, _ = make_bundle(gguf_report(), target="dgx-spark-single", tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "CTX_SIZE / PARALLEL_SEQS" in text
    assert "context_per_slot" in text


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


def test_health_timeout_scales_with_model_size(isolated_config, tmp_path):
    """โมเดลใหญ่ต้องได้ timeout นานขึ้นอัตโนมัติ + มีคำสั่ง wait-health สำหรับตามต่อ"""
    small, _, _ = make_bundle(gguf_report(weight_bytes=5 * GIB), tmp_path=tmp_path / "s")
    big, _, _ = make_bundle(gguf_report(weight_bytes=100 * GIB), target="dgx-spark-single", tmp_path=tmp_path / "b")

    small_text = small.controller.read_text(encoding="utf-8")
    big_text = big.controller.read_text(encoding="utf-8")
    assert 'HEALTH_TIMEOUT:-600}' in small_text  # 5GB → ขั้นต่ำ 600
    assert 'HEALTH_TIMEOUT:-3300}' in big_text  # 100GB → 100×30+300
    for text in (small_text, big_text):
        assert "wait-health)" in text
        assert "ไม่ได้ถูกหยุด" in text  # timeout ต้องบอกว่าเซิร์ฟเวอร์ยังโหลดต่อ


def test_gated_repo_noted_in_readme(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(gated=True), tmp_path=tmp_path)
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "HF_TOKEN" in readme
    assert "gated" in readme


def test_verify_files_checks_shards_and_sizes(tmp_path):
    """download ที่ขาด shard ต้องถูกจับตอน verify-files ไม่ใช่ไปพังตอน start"""
    from lmds.inspector.report import ShardFile

    report = safetensors_report(
        shard_count=2,
        safetensor_shards=[
            ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=32_500_000_000),
            ShardFile(filename="model-00002-of-00002.safetensors", size_bytes=32_400_000_000),
        ],
        tokenizer_files=["tokenizer.json", "tokenizer_config.json"],
    )
    bundle, _, _ = make_bundle(report, tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "SHARD_FILES=(" in script
    assert "model-00001-of-00002.safetensors" in script
    assert "32500000000" in script
    assert "ขนาดไม่ตรง" in script
    assert "tokenizer.json" in script  # tokenizer ที่ repo มีจริง ต้องอยู่ในไฟล์จำเป็นด้วย


def test_runtime_asset_fetched_and_mounted(tmp_path):
    """ไฟล์ runtime ที่อนุมัติแล้วต้องมี prepare-runtime + bind-mount + ตรวจ sha"""
    from lmds.brain.plan_schema import RuntimeAsset

    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.runtime_assets = [
        RuntimeAsset(
            filename="super_v3_reasoning_parser.py",
            url="https://raw.githubusercontent.com/example/repo/main/super_v3_reasoning_parser.py",
            sha256="a" * 64,
            purpose="reasoning parser plugin",
        )
    ]
    bundle = render_bundle(plan, report, fit, tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "prepare_runtime()" in script
    assert "prepare-runtime) prepare_runtime" in script
    assert "super_v3_reasoning_parser.py" in script
    assert "${PLUGIN_MOUNT}:ro" in script
    assert "a" * 64 in script
    assert not audit_script(script)


def test_no_runtime_assets_keeps_script_clean(tmp_path):
    """bundle ปกติต้องไม่มีโค้ด plugin ปนเข้ามา"""
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "PLUGIN_DIR" not in script
    assert "prepare-runtime" not in script


def mmproj_gguf_report(**overrides) -> ModelReport:
    """repo GGUF ที่มีไฟล์ mmproj แยก — เคสจริง unsloth/gemma-4-12b-it-GGUF"""
    return gguf_report(
        repo_id="unsloth/gemma-4-12b-it-GGUF",
        selected_gguf="gemma-4-12b-it-UD-Q8_K_XL.gguf",
        gguf_variants=[
            GgufVariant(
                filename="gemma-4-12b-it-UD-Q8_K_XL.gguf", size_bytes=13 * GIB, sha256="a" * 64
            ),
            GgufVariant(filename="mmproj-F32.gguf", size_bytes=3 * GIB, sha256="c" * 64, is_mmproj=True),
            GgufVariant(filename="mmproj-BF16.gguf", size_bytes=1 * GIB, sha256="b" * 64, is_mmproj=True),
        ],
        **overrides,
    )


def test_multimodal_gguf_downloads_and_loads_projector(tmp_path):
    """เคสจริง 2026-08-03: profile บอกว่าต้องมี mmproj แต่ controller ไม่มีคำว่า mmproj เลย
    → download ได้ไฟล์เดียว, start ผ่าน, /health เขียว แต่โมเดลรับแต่ข้อความ ไม่มี error ให้เห็น
    """
    report = mmproj_gguf_report()
    bundle, plan, _ = make_bundle(report, tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    # เล็กสุดในกลุ่ม mmproj — BF16 (1 GB) ไม่ใช่ F32 (3 GB)
    assert plan.multimodal.projector_files == ["mmproj-BF16.gguf"]

    assert "mmproj-BF16.gguf" in script, "controller ต้องดาวน์โหลดไฟล์ projector ด้วย"
    assert "b" * 64 in script, "projector ต้องถูก verify ด้วย SHA-256 เหมือน weight"
    assert "--mmproj" in script, "ไม่ส่ง --mmproj = โมเดลกลายเป็น text-only แบบเงียบ"
    # MODEL_FILE (ตัวที่ส่งเป็น -m) ต้องยังเป็น weight ไม่ใช่ projector
    assert 'MODEL_FILE="${MODEL_FILES[0]}"' in script
    assert script.index("gemma-4-12b-it-UD-Q8_K_XL.gguf") < script.index("mmproj-BF16.gguf")
    assert not audit_script(script)

    profile = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["features"]["multimodal"]["projector_files"] == ["mmproj-BF16.gguf"]


def test_text_only_gguf_has_no_projector_flag(tmp_path):
    """repo ที่ไม่มี mmproj ต้องไม่มี --mmproj โผล่มา (ค่าว่างจะทำให้ llama-server ล้ม)"""
    bundle, plan, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert plan.multimodal.projector_files == []
    assert "mmproj" not in script.lower()
    assert not audit_script(script)


def test_usage_documents_options_and_api_token(tmp_path):
    """help ของ controller ต้องอธิบาย port/context/bind และวิธีตั้ง API token ให้ครบ"""
    for report in (safetensors_report(), gguf_report()):
        bundle, _, _ = make_bundle(report, tmp_path=tmp_path / report.artifact_type.value)
        script = bundle.controller.read_text(encoding="utf-8")

        assert "API TOKEN (authentication)" in script
        assert "API_KEY=my-secret-token" in script
        assert "Authorization: Bearer" in script
        assert "ENVIRONMENT VARIABLES" in script
        assert "EXAMPLES" in script
        for opt in ("--port N", "--context N", "--bind ADDR", "--advertise-ip ADDR"):
            assert opt in script, f"{opt} ไม่มีใน usage ของ {report.artifact_type.value}"
        # เตือนเรื่อง endpoint เปิดโล่งต้องอยู่ใน help ด้วย ไม่ใช่แค่ตอน start
        assert "127.0.0.1" in script


def test_test_text_survives_reasoning_models(tmp_path):
    """max_tokens 64 ทำให้โมเดลสาย reasoning คืน content ว่าง + finish_reason length

    เจอจริงกับ gemma-4-12b-it (2026-08-03): reasoning_content กิน budget หมดก่อนจะได้ตอบ
    ผู้ใช้เห็นแล้วนึกว่าโมเดลพัง
    """
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert '\\"max_tokens\\": 512' in script
    assert "reasoning_content" in script, "ต้องแยก 'ยังคิดไม่จบ' ออกจาก 'ตอบว่าง' ให้ผู้ใช้"
    assert "test-text: OK" in script
    assert not audit_script(script)
