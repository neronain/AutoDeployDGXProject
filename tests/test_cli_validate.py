from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.packager import write_checksums
from tests.test_generator import gguf_report, make_bundle

runner = CliRunner()


def test_validate_passes_on_good_bundle(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    write_checksums(bundle.directory)
    result = runner.invoke(app, ["validate", str(bundle.directory)])
    assert result.exit_code == 0
    assert "static-validated" in result.output


def test_validate_fix_regenerates_checksums(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    result = runner.invoke(app, ["validate", str(bundle.directory), "--fix"])
    assert result.exit_code == 0
    assert (bundle.directory / "PACKAGE_SHA256SUMS").exists()


def test_validate_exit_2_on_tampered_bundle(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    write_checksums(bundle.directory)
    (bundle.directory / "README.md").write_text("tampered", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bundle.directory)])
    assert result.exit_code == 2


def test_validate_missing_dir_exit_1(isolated_config):
    result = runner.invoke(app, ["validate", "/nonexistent/path"])
    assert result.exit_code == 1
