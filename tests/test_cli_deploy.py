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
