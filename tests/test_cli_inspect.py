from typer.testing import CliRunner

import lmds.cli.main as cli_main
from lmds.cli.main import app
from lmds.inspector import ArtifactType, AuthRequired, ModelReport

runner = CliRunner()


def fake_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="abc123",
        artifact_type=ArtifactType.SAFETENSORS,
        license="apache-2.0",
        params_total=32_800_000_000,
        weight_bytes=65_000_000_000,
        architecture="Qwen3ForCausalLM",
        context_length=40960,
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def test_inspect_renders_table(isolated_config, monkeypatch):
    monkeypatch.setattr("lmds.inspector.inspect.inspect_model", lambda s, c: fake_report())
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: fake_report())
    result = runner.invoke(app, ["inspect", "Qwen/Qwen3-32B"])
    assert result.exit_code == 0
    assert "abc123" in result.output
    assert "apache-2.0" in result.output


def test_inspect_json_output(isolated_config, monkeypatch):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: fake_report())
    result = runner.invoke(app, ["inspect", "Qwen/Qwen3-32B", "--json"])
    assert result.exit_code == 0
    assert '"revision_sha"' in result.output


def test_inspect_bad_input_exit_1(isolated_config):
    result = runner.invoke(app, ["inspect", "not a model"])
    assert result.exit_code == 1


def test_inspect_gated_noninteractive_exit_4(isolated_config, monkeypatch):
    def raise_auth(source, client):
        raise AuthRequired("meta-llama/Llama-3.3-70B-Instruct", 401, had_token=False)

    monkeypatch.setattr("lmds.inspector.inspect_model", raise_auth)
    # CliRunner ไม่ใช่ tty → path non-interactive → exit 4 พร้อมคำแนะนำ
    result = runner.invoke(app, ["inspect", "meta-llama/Llama-3.3-70B-Instruct"])
    assert result.exit_code == 4
    assert "set-hf-token" in result.output


def test_inspect_revision_option_pins_request(isolated_config, monkeypatch):
    captured = {}

    def capture(source, client):
        captured["revision"] = source.revision
        return fake_report()

    monkeypatch.setattr("lmds.inspector.inspect_model", capture)
    result = runner.invoke(app, ["inspect", "Qwen/Qwen3-32B", "--revision", "v2.0"])
    assert result.exit_code == 0
    assert captured["revision"] == "v2.0"
