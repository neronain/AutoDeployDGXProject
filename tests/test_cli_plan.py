from typer.testing import CliRunner

from lmds.cli.main import app
from tests.test_cli_inspect import fake_report

runner = CliRunner()


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)


def test_plan_no_llm_rule_based(isolated_config, monkeypatch):
    _patch_inspect(monkeypatch, fake_report(kv_dims={"layers": 64, "kv_heads": 8, "head_dim": 128}))
    result = runner.invoke(app, ["plan", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-single"])
    assert result.exit_code == 0
    assert "rule-based" in result.output
    assert "vllm" in result.output


def test_plan_json_output_validates_schema(isolated_config, monkeypatch):
    import json

    from lmds.brain import DeploymentPlan

    _patch_inspect(monkeypatch, fake_report())
    result = runner.invoke(
        app, ["plan", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-single", "--json"]
    )
    assert result.exit_code == 0
    plan = DeploymentPlan.model_validate(json.loads(result.output))
    assert plan.revision == "abc123"


def test_plan_without_provider_config_falls_back(isolated_config, monkeypatch):
    # ไม่ตั้ง provider เลย (ไม่ใช้ --no-llm) → ต้อง fallback rule-based พร้อมข้อความแจ้ง
    _patch_inspect(monkeypatch, fake_report())
    result = runner.invoke(app, ["plan", "Qwen/Qwen3-32B", "--target", "dgx-spark-single"])
    assert result.exit_code == 0
    assert "rule-based" in result.output


def test_plan_unknown_target_exit_1(isolated_config, monkeypatch):
    _patch_inspect(monkeypatch, fake_report())
    result = runner.invoke(app, ["plan", "Qwen/Qwen3-32B", "--target", "does-not-exist"])
    assert result.exit_code == 1
