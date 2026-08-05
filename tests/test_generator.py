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
from lmds.generator import renderer
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


@pytest.mark.parametrize("kind", ["vllm", "llamacpp"])
def test_test_text_survives_reasoning_models(tmp_path, kind):
    """max_tokens 64 ทำให้โมเดลสาย reasoning คืนคำตอบว่าง + finish_reason length

    เจอจริงสองรอบวันเดียวกัน (2026-08-03) — gemma-4-12b-it ฝั่ง llama.cpp (reasoning_content
    แยก field) และ Qwen3-8B ฝั่ง vLLM (<think> อยู่ใน content) · ผู้ใช้เห็นแล้วนึกว่าโมเดลพัง
    ทั้งที่เซิร์ฟเวอร์ทำงานปกติ จึงต้องแก้ให้ครบทุก template ไม่ใช่เฉพาะตัวที่เจอ
    """
    report = safetensors_report() if kind == "vllm" else gguf_report()
    bundle, _, _ = make_bundle(report, tmp_path=tmp_path / kind)
    script = bundle.controller.read_text(encoding="utf-8")

    assert '\\"max_tokens\\": 512' in script
    assert "reasoning_content" in script, "ต้องแยก 'ยังคิดไม่จบ' ออกจาก 'ตอบว่าง' ให้ผู้ใช้"
    assert "test-text: OK" in script
    assert not audit_script(script)


def test_stacked_test_text_also_handles_reasoning(tmp_path):
    """template stacked ถูกลืมบ่อยเพราะรันจริงยาก — ต้องได้การแก้เดียวกับ single"""
    text = (renderer.TEMPLATES_DIR / "stacked-vllm-controller.sh.j2").read_text(encoding="utf-8")
    assert '\\"max_tokens\\": 512' in text
    assert "test-text: OK" in text


def test_gated_repo_controller_checks_token_before_download(tmp_path):
    """เคสจริง Llama-3.1-8B (2026-08-03): ไม่มี HF_TOKEN → huggingface_hub โยน traceback
    60 บรรทัดใส่หน้าผู้ใช้ ทั้งที่สาเหตุคือ "ยังไม่ได้ตั้ง token" ประโยคเดียว
    """
    bundle, _, _ = make_bundle(
        safetensors_report(repo_id="meta-llama/Llama-3.1-8B-Instruct", gated=True),
        tmp_path=tmp_path,
    )
    script = bundle.controller.read_text(encoding="utf-8")

    assert "gated repo" in script
    assert 'if [[ -z "${HF_TOKEN:-}" ]]; then' in script
    assert "settings/tokens" in script
    assert not audit_script(script)


def test_public_repo_has_no_token_gate(tmp_path):
    """repo สาธารณะต้องไม่ถูกบังคับให้มี token (ข้อความ help เรื่อง HF_TOKEN ยังอยู่ได้)"""
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert 'if [[ -z "${HF_TOKEN:-}" ]]; then' not in script
    assert "settings/tokens" not in script


def test_download_falls_back_when_xet_transfer_fails(tmp_path):
    """Xet backend ของ Hub พังกับบาง repo ('Unable to parse string as hex hash value')

    env บน host ไม่ถึงคอนเทนเนอร์เอง — ต้องส่ง HF_HUB_DISABLE_XET เข้าไปและลองซ้ำให้อัตโนมัติ
    """
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "HF_HUB_DISABLE_XET" in script
    assert "-e HF_HUB_DISABLE_XET=1" in script


def test_usage_block_keeps_indentation(tmp_path):
    """`{% endif -%}` ของ Jinja กิน whitespace ของบรรทัดถัดไป — help เคยพิมพ์ `start` ชิดขอบ
    ทั้งที่คำสั่งอื่นย่อหน้า 2 ช่อง (เห็นในผลรันจริงบน RTX 5090)
    """
    for report in (safetensors_report(), gguf_report()):
        bundle, _, _ = make_bundle(report, tmp_path=tmp_path / report.artifact_type.value)
        script = bundle.controller.read_text(encoding="utf-8")
        in_commands = False
        for line in script.splitlines():
            if line == "COMMANDS":
                in_commands = True
                continue
            if in_commands:
                if not line.strip():
                    break
                assert line.startswith("  "), f"บรรทัด help หลุดการย่อหน้า: {line!r}"


def test_stop_reports_truthfully(tmp_path):
    """`stop` เคยพิมพ์ "stopped" เสมอแม้ไม่มีอะไรรันอยู่ — ถ้าชื่อ container/PID เพี้ยน
    ผู้ใช้จะเชื่อว่าหยุดแล้วทั้งที่ยังรันอยู่ (เห็นในผลรันจริงบน RTX 5090)
    """
    for report in (safetensors_report(), gguf_report()):
        bundle, _, _ = make_bundle(report, tmp_path=tmp_path / report.artifact_type.value)
        script = bundle.controller.read_text(encoding="utf-8")
        assert "ไม่มีอะไรให้หยุด" in script
        assert not audit_script(script)


def test_readme_surfaces_context_headroom(tmp_path):
    """คนที่รับ bundle ต่อ (SI → ลูกค้า) ต้องเห็นว่าเครื่องรับ context ได้มากกว่าค่าเริ่มต้น"""
    from lmds.fit.analyzer import GIB
    from lmds.inspector.report import KvDims

    report = gguf_report(
        weight_bytes=int(32.5 * GIB),
        context_length=262144,
        kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    bundle, plan, fit = make_bundle(report, tmp_path=tmp_path)
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    profile = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))

    assert fit.max_safe_context > plan.serving.context
    assert "262,144" in readme
    assert profile["target"]["max_safe_context"] == fit.max_safe_context


def _vllm_with_features(tmp_path):
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.tool_calling.enabled = True
    plan.tool_calling.parser = "hermes"
    plan.reasoning.enabled = True
    plan.reasoning.parser = "deepseek_r1"
    return render_bundle(plan, report, fit, tmp_path).controller.read_text(encoding="utf-8")


def test_enabled_parsers_get_an_acceptance_test(tmp_path):
    """เรา emit --tool-call-parser / --reasoning-parser ให้ vLLM แต่ไม่เคยมีทางพิสูจน์ว่าใช้ได้จริง
    parser ผิดตัวจะเงียบจนกว่าลูกค้าจะเจอเอง (ช่องว่างเทียบ reference v8.2)
    """
    script = _vllm_with_features(tmp_path)
    assert "test-reasoning)" in script and "test_reasoning()" in script
    assert "test-tools)" in script and "test_tools()" in script
    assert "1591" in script  # 37×43 — ตรวจว่าคิดเลขถูกจริง ไม่ใช่แค่ตอบอะไรมา
    assert "tool_calls" in script


def test_features_off_keeps_controller_clean(tmp_path):
    """โมเดลที่ไม่เปิด parser ต้องไม่มีคำสั่งทดสอบที่ใช้ไม่ได้ติดมา"""
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert "test_reasoning" not in script
    assert "test_tools" not in script
    assert not audit_script(script)


def test_multimodal_bundle_can_prove_vision_works(tmp_path):
    """mmproj มาครบไม่ได้แปลว่า vision ทำงาน — ต้องมีคำสั่งพิสูจน์ได้เอง
    (เดิมต้องให้คนเขียน curl + base64 เอง ซึ่งไม่มีใครทำ)
    """
    bundle, _, _ = make_bundle(mmproj_gguf_report(), tmp_path=tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "test_vision()" in script
    assert "test-vision)" in script
    assert "test-vision       Send a generated red image" in script
    # สร้างภาพเองด้วย stdlib — ไม่ต้องมีรูปในเครื่อง ไม่ต้องต่อเน็ต
    assert "import base64, json, struct, sys, urllib.error, urllib.request, zlib" in script
    assert not audit_script(script)


def test_text_only_bundle_has_no_vision_test(tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    assert "test_vision" not in bundle.controller.read_text(encoding="utf-8")


def test_spark_targets_default_to_the_ngc_vllm_image():
    """image upstream มี manifest arm64 แต่ไม่ได้ build kernel ให้ SM121 (GB10)
    — controller ที่รันจริงบน Spark ทุกตัวใช้ NGC"""
    from lmds.brain.rulebased import rule_based_plan
    from lmds.fit import PRESETS, analyze
    from lmds.inspector.report import ArtifactType, ModelReport

    report = ModelReport(repo_id="meta-llama/Llama-3.3-70B-Instruct", revision_sha="sha",
                         artifact_type=ArtifactType.SAFETENSORS, weight_bytes=140 * 1024**3,
                         shard_count=30, has_chat_template=True)
    spark = rule_based_plan(report, analyze(report, PRESETS["dgx-spark-stacked"]))
    rtx = rule_based_plan(report, analyze(report, PRESETS["rtx-5090"]))

    assert spark.runtime.image_ref.startswith("nvcr.io/nvidia/vllm")
    assert rtx.runtime.image_ref.startswith("vllm/vllm-openai")


def test_deepseek_v4_forces_fp8_kv_cache():
    """DeepSeek V4 ใช้ attention layout ที่บังคับ kv-cache fp8 — ปล่อย auto แล้ว vLLM
    ตายตอน load_model ด้วย 'fp8_ds_mla layout only supports fp8 kv-cache, got auto'
    (เจอจากการรันจริงบน DGX Spark)"""
    from lmds.brain.rulebased import rule_based_plan
    from lmds.fit import PRESETS, analyze
    from lmds.inspector.report import ArtifactType, ModelReport

    def report_for(repo):
        return ModelReport(repo_id=repo, revision_sha="sha", artifact_type=ArtifactType.SAFETENSORS,
                           weight_bytes=150 * 1024**3, shard_count=46, has_chat_template=True)

    ds = report_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    plan = rule_based_plan(ds, analyze(ds, PRESETS["dgx-spark-stacked"]))
    # สูตรระบุ layout เฉพาะของ DeepSeek V4 ทับค่ากว้าง ๆ ที่ ARCH_REQUIREMENTS ตั้งไว้
    assert plan.serving.kv_cache_dtype == "nvfp4_ds_mla"

    other = report_for("meta-llama/Llama-3.3-70B-Instruct")
    plan2 = rule_based_plan(other, analyze(other, PRESETS["dgx-spark-stacked"]))
    assert plan2.serving.kv_cache_dtype == "auto"


def test_metadata_cap_fits_real_moe_quant_configs():
    """config.json ของ MoE + NVFP4 โตตามจำนวนชั้น — Nemotron-3-Super-120B = 7.4MB
    เพดาน 4MB เดิมทำให้ inspect โมเดลกลุ่มนี้ไม่ผ่านเลย"""
    from lmds.inspector.hf_api import SMALL_FILE_CAP

    assert SMALL_FILE_CAP >= 8 * 1024 * 1024
