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
