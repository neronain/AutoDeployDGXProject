import os
from pathlib import Path

from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.fleet import discover, find, stop_server

runner = CliRunner()


def make_meta(root: Path, slug: str, mode: str = "native", pid: int | None = None,
              port: int = 8000, controller: str = "") -> Path:
    run_dir = root / slug
    run_dir.mkdir(parents=True)
    pid_file = run_dir / "server.pid"
    if pid is not None:
        pid_file.write_text(str(pid), encoding="utf-8")
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={slug}-model\n"
        f"model_id=org/{slug}\n"
        f"engine=llamacpp\n"
        f"mode={mode}\n"
        f"port={port}\n"
        f"container=lmds-{slug}\n"
        f"pid_file={pid_file if pid is not None else ''}\n"
        f"controller={controller}\n"
        f"started_at=2026-07-21T12:00:00\n",
        encoding="utf-8",
    )
    return run_dir


def test_discover_native_running_and_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-a", pid=os.getpid(), port=8000)   # pid ของ pytest เอง = alive
    make_meta(tmp_path, "model-b", pid=999999999, port=8001)     # pid ไม่มีจริง = stopped

    servers = {s.slug: s for s in discover()}
    assert servers["model-a"].running is True
    assert servers["model-b"].running is False
    assert servers["model-a"].port == 8000
    assert servers["model-a"].endpoint == "http://127.0.0.1:8000/v1"


def test_find_by_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-a", pid=os.getpid())
    assert find("model-a") is not None
    assert find("no-such") is None


def test_stop_native_fallback_kills_pid(tmp_path, monkeypatch):
    """controller หาย → fallback ส่ง SIGTERM ตรง — ใช้ fake kill กันไม่ให้ฆ่า pytest"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-a", pid=os.getpid(), controller="/no/such/controller.sh")
    killed = {}

    real_kill = os.kill

    def fake_kill(pid, sig):
        if sig == 0:
            return real_kill(pid, 0)
        killed["pid"], killed["sig"] = pid, sig

    monkeypatch.setattr(os, "kill", fake_kill)
    server = find("model-a")
    method = stop_server(server)
    assert method == "kill"
    assert killed == {"pid": os.getpid(), "sig": 15}


def test_stop_via_controller_when_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\necho stopped > " + str(tmp_path / "stopped.flag") + "\n")
    controller.chmod(0o755)
    make_meta(tmp_path, "model-a", pid=os.getpid(), controller=str(controller))

    method = stop_server(find("model-a"))
    assert method == "controller"
    assert (tmp_path / "stopped.flag").exists()


def test_cli_ps_lists_servers(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "qwen3-coder", pid=os.getpid(), port=8000)
    make_meta(tmp_path, "nvfp4-27b", pid=999999999, port=8001)

    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0
    assert "qwen3-coder" in result.output
    assert "nvfp4-27b" in result.output
    assert "stopped" in result.output


def test_cli_ps_empty(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "empty"))
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0
    assert "deploy" in result.output


def test_cli_stop_requires_slug_or_all(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 1


def test_cli_stop_all(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\nexit 0\n")
    controller.chmod(0o755)
    make_meta(tmp_path, "model-a", pid=os.getpid(), controller=str(controller))
    make_meta(tmp_path, "model-b", pid=999999999)  # ไม่รัน — ต้องถูกข้าม

    result = runner.invoke(app, ["stop", "--all"])
    assert result.exit_code == 0
    assert "model-a" in result.output
    assert "model-b" not in result.output


def test_cli_list_shows_missing_controller(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-a", pid=999999999, controller="/gone/ctl.sh")
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "หาย" in result.output


def test_generated_controller_writes_meta(isolated_config, tmp_path):
    from tests.test_generator import gguf_report, make_bundle

    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "server.meta" in text
    assert "write_meta" in text
    for key in ["slug=", "engine=", "port=", "controller="]:
        assert key in text
