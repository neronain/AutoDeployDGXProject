import zipfile

from typer.testing import CliRunner

from lmds.brain import apply_flag_approvals, build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from tests.test_generator import gguf_report, safetensors_report

runner = CliRunner()


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)


def test_deploy_yes_produces_validated_zip(isolated_config, tmp_path, monkeypatch):
    _patch_inspect(monkeypatch, gguf_report())
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 0, result.output
    zip_path = tmp_path / "qwen3-8b-gguf.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert any(n.endswith("PACKAGE_SHA256SUMS") for n in archive.namelist())
    assert "static-validated" in result.output


def test_deploy_no_fit_exit_3(isolated_config, tmp_path, monkeypatch):
    # BF16 65GB บน RTX 24GB เดี่ยว → needs-smaller-quant → exit 3 พร้อมทางเลือก
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(
        app,
        ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "rtx-pro-4000",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 3
    assert "ไม่ fit" in result.output


def test_deploy_non_tty_skips_confirmation(isolated_config, tmp_path, monkeypatch):
    # CliRunner ไม่ใช่ tty → ต้องไม่ค้างรอ input แม้ไม่มี --yes
    _patch_inspect(monkeypatch, gguf_report())
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
         "--output", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output


def test_deploy_falls_back_to_rule_based_on_provider_error(isolated_config, tmp_path, monkeypatch):
    """เคสจริงจากเครื่อง gigabyte02: Gemini 429 → ต้องสลับ rule-based ไม่ใช่หยุดทำงาน"""
    from lmds.brain import ProviderError
    from lmds.config import ProviderName, Settings

    class QuotaExceededProvider:
        name = "gemini"
        model = "gemini-2.5-pro"

        def complete_json(self, system, user):
            raise ProviderError("gemini ตอบ HTTP 429: quota exceeded")

    settings = Settings.load()
    settings.set_provider(ProviderName.GEMINI)
    settings.save()
    monkeypatch.setattr("lmds.brain.make_provider", lambda c, k, client=None: QuotaExceededProvider())
    monkeypatch.setattr("lmds.secrets.store.get_secret", lambda n: "AIzaFakeKey123" if n == "gemini" else None)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda n: "AIzaFakeKey123" if n == "gemini" else None)
    _patch_inspect(monkeypatch, gguf_report())

    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--target", "dgx-spark-single",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "rule-based" in result.output
    assert "429" in result.output  # แจ้งสาเหตุที่สลับ


def test_spark_unified_memory_from_allowlist_when_smi_reports_none(isolated_config):
    """GB10 บนเครื่องจริง: nvidia-smi ไม่รายงาน memory.total → ใช้สเปก 128GB จาก allowlist"""
    from lmds.fit import from_hardware_report
    from lmds.hardware import MemoryModel
    from lmds.hardware.profiler import DetectedGpu, HardwareReport
    from lmds.hardware.profiles import TargetProfile, lookup_gpu

    report = HardwareReport(
        arch="aarch64",
        gpus=[DetectedGpu("NVIDIA GB10", None, "12.1", lookup_gpu("NVIDIA GB10"))],
        ram_gb=121.7,
        profile=TargetProfile.DGX_SPARK_SINGLE,
    )
    spec = from_hardware_report(report)
    assert spec is not None
    assert spec.memory_gb == 128.0
    assert spec.memory_model is MemoryModel.UNIFIED
    assert spec.tested is True


def test_deploy_multi_variant_gguf_fails_early_with_guidance(isolated_config, tmp_path, monkeypatch):
    """เคสจริงจาก gigabyte02: repo หลาย variant ต้องแจ้งตั้งแต่ต้น flow ไม่ใช่หลังยืนยันแผน"""
    from lmds.inspector.report import GgufVariant

    report = gguf_report(
        selected_gguf=None,
        weight_bytes=None,
        gguf_variants=[
            GgufVariant(filename="m-Q4_K_M.gguf", size_bytes=18 * GIB),
            GgufVariant(filename="m-Q8_0.gguf", size_bytes=32 * GIB),
        ],
    )
    _patch_inspect(monkeypatch, report)
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "--no-llm",
         "--target", "dgx-spark-single", "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 1
    assert "m-Q4_K_M.gguf" in result.output       # แสดงรายการ variant
    assert "blob/main" in result.output           # บอกวิธีระบุไฟล์ตรง
    assert "ยืนยัน" not in result.output           # ต้องจบก่อนถึงขั้นยืนยัน


def test_ensure_gguf_selected_interactive_reinspects(isolated_config, monkeypatch):
    """โหมด interactive: เลือกหมายเลขแล้วต้อง inspect ซ้ำด้วยไฟล์ที่เลือก"""
    import typer

    from lmds.cli.main import _ensure_gguf_selected
    from lmds.inspector.report import GgufVariant
    from lmds.resolver import parse_source

    report = gguf_report(
        selected_gguf=None,
        weight_bytes=None,
        gguf_variants=[
            GgufVariant(filename="m-Q4_K_M.gguf", size_bytes=18 * GIB),
            GgufVariant(filename="m-Q8_0.gguf", size_bytes=32 * GIB),
        ],
    )
    seen = {}

    def fake_inspect(source, client):
        seen["filename"] = source.filename
        return gguf_report(selected_gguf=source.filename, weight_bytes=18 * GIB)

    monkeypatch.setattr("lmds.inspector.inspect_model", fake_inspect)
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: 1)  # เลือกข้อ 1 (เรียงจากเล็กไปใหญ่)

    source = parse_source("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
    out = _ensure_gguf_selected(source, report, interactive=True)
    assert seen["filename"] == "m-Q4_K_M.gguf"
    assert out.selected_gguf == "m-Q4_K_M.gguf"


def test_ensure_gguf_selected_passthrough_when_selected(isolated_config):
    from lmds.cli.main import _ensure_gguf_selected
    from lmds.resolver import parse_source

    report = gguf_report()  # selected_gguf ตั้งแล้ว
    out = _ensure_gguf_selected(parse_source("unsloth/Qwen3-8B-GGUF"), report, interactive=False)
    assert out is report


def test_apply_flag_approvals_moves_flag(isolated_config):
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.flags_needing_approval = ["--trust-remote-code", "--weird"]

    apply_flag_approvals(plan, ["--trust-remote-code"])
    assert "--trust-remote-code" in plan.serving.extra_flags
    assert plan.flags_needing_approval == ["--weird"]
    assert any("อนุมัติ" in w for w in plan.warnings)


def test_approved_flag_survives_into_script(isolated_config, tmp_path):
    from lmds.generator import render_bundle

    report = gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.flags_needing_approval = ["--no-mmap"]
    apply_flag_approvals(plan, ["--no-mmap"])

    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "--no-mmap" in text


def test_rebuild_reuses_the_stored_plan_and_fixes_the_image(tmp_path, monkeypatch, isolated_config):
    """bundle ที่ image ใช้ไม่ได้ (tag หายไป) แก้ไม่ได้เลยนอกจากเดินผ่าน wizard ใหม่ทั้งชุด
    — ทั้งที่ MODEL_PROFILE.yaml เก็บทุกอย่างที่ต้องใช้ไว้แล้ว (เจอจริงกับ v0.6.3.ss)
    """
    from lmds.brain import registry

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    _patch_inspect(monkeypatch, gguf_report())
    first = runner.invoke(app, ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm",
                                "--target", "dgx-spark-single", "--output", str(tmp_path), "--yes"])
    assert first.exit_code == 0, first.output

    bundle_dir = tmp_path / "qwen3-8b-gguf"
    profile = bundle_dir / "MODEL_PROFILE.yaml"
    text = profile.read_text(encoding="utf-8")
    current = text.split("  image: ")[1].split("\n")[0]
    profile.write_text(text.replace(current, "vllm/vllm-openai:v0.6.3.ss"), encoding="utf-8")

    monkeypatch.setattr(registry, "tag_exists",
                        lambda ref, client=None: False if "v0.6.3.ss" in ref else True)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    result = runner.invoke(app, ["rebuild", "qwen3-8b-gguf", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    rebuilt = (tmp_path / "out" / "qwen3-8b-gguf" / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    assert "v0.6.3.ss" not in rebuilt, "image ที่ใช้ไม่ได้ต้องถูกแทนตอน rebuild"


def test_rebuild_says_so_when_there_is_no_profile(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    result = runner.invoke(app, ["rebuild", "not-a-bundle"])
    assert result.exit_code == 1
    assert "ไม่พบ bundle" in result.output
