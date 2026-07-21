from typer.testing import CliRunner

import lmds
from lmds.cli.main import app
from lmds.secrets import get_secret

runner = CliRunner()


def test_version(isolated_config):
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert lmds.__version__ in result.output
    assert "v3.0.0" in result.output


def test_config_set_provider_and_show(isolated_config):
    result = runner.invoke(app, ["config", "set-provider", "openai"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_config_set_key_via_stdin_and_masked_in_show(isolated_config):
    secret = "sk-test1234567890abcdefgh"
    result = runner.invoke(app, ["config", "set-key", "openai", "--stdin"], input=secret + "\n")
    assert result.exit_code == 0
    assert secret not in result.output  # ค่า key ห้ามโผล่ใน output
    assert get_secret("openai") == secret

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert secret not in result.output  # show ต้อง mask เสมอ


def test_set_hf_token_optional_skip(isolated_config):
    result = runner.invoke(app, ["config", "set-hf-token", "--stdin"], input="\n")
    assert result.exit_code == 0
    assert get_secret("hf") is None


def test_set_hf_token_via_stdin(isolated_config):
    token = "hf_ABCDEFGHIJKLMNOP1234"
    result = runner.invoke(app, ["config", "set-hf-token", "--stdin"], input=token + "\n")
    assert result.exit_code == 0
    assert get_secret("hf") == token
    assert token not in result.output


def test_openai_compat_without_base_url_fails(isolated_config):
    result = runner.invoke(app, ["config", "set-provider", "openai-compat", "--model", "qwen"])
    assert result.exit_code == 1


def test_hardware_command_runs_anywhere(isolated_config):
    # บนเครื่อง dev ที่ไม่มี nvidia-smi ต้องไม่ crash — แค่รายงานว่าตรวจไม่พบ
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Profile" in result.output or "profile" in result.output
