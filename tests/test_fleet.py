import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.fleet import discover, find, stop_server

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_orphan_scan(monkeypatch):
    """ปิดการสแกน process/container จริงของเครื่อง dev — เทสที่ต้องการ orphan override เอง"""
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])


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


def test_orphan_native_detected(tmp_path, monkeypatch):
    """เคสจริงจาก gigabyte02: โมเดลที่ start จาก bundle รุ่นเก่า (ไม่มี meta) ต้องโผล่ใน ps"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-registered", pid=os.getpid(), port=8000)
    monkeypatch.setattr(
        "lmds.fleet.manager._pgrep_llama",
        lambda: [
            (os.getpid(), "llama-server -m /x.gguf --port 8000"),  # pid ตรง meta → ไม่ซ้ำ
            (424242, "/home/u/src/llama.cpp/build/bin/llama-server -m /home/u/models/q/Qwen3-old.gguf "
                     "--alias Qwen3-Coder-Old --port 8001"),
        ],
    )

    servers = {s.slug: s for s in discover()}
    assert "model-registered" in servers
    orphan = servers["Qwen3-Coder-Old"]
    assert orphan.registered is False
    assert orphan.running is True
    assert orphan.port == 8001
    assert orphan.pid == 424242
    assert len(servers) == 2  # pid ที่ลงทะเบียนแล้วต้องไม่ถูกนับซ้ำ


def test_orphan_docker_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    from lmds.fleet.manager import ServerInfo

    monkeypatch.setattr(
        "lmds.fleet.manager._orphan_docker",
        lambda known: [ServerInfo(slug="old-nvfp4", engine="?", mode="docker",
                                  container="lmds-old-nvfp4", running=True, registered=False)],
    )
    servers = {s.slug: s for s in discover()}
    assert servers["old-nvfp4"].registered is False


def test_stop_orphan_native_by_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmds.fleet.manager._pgrep_llama",
        lambda: [(424242, "llama-server -m /m.gguf --alias orphan-model --port 8001")],
    )
    killed = {}
    real_kill = os.kill

    def fake_kill(pid, sig):
        if sig == 0:
            return real_kill(pid, 0)
        killed["pid"] = pid

    monkeypatch.setattr(os, "kill", fake_kill)
    server = find("orphan-model")
    assert server is not None
    assert stop_server(server) == "kill"
    assert killed["pid"] == 424242


def test_cli_ps_marks_unregistered(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmds.fleet.manager._pgrep_llama",
        lambda: [(424242, "llama-server -m /m.gguf --alias old-model --port 8001")],
    )
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0
    assert "old-model" in result.output
    assert "ไม่ลงทะเบียน" in result.output
    assert "regenerate" in result.output


def test_host_summary_fields(monkeypatch):
    from lmds.hardware import host_summary
    from lmds.hardware.profiler import DetectedGpu
    from lmds.hardware.profiles import lookup_gpu

    monkeypatch.setattr(
        "lmds.hardware.profiler.detect_gpus",
        lambda: ([DetectedGpu("NVIDIA GB10", None, "12.1", lookup_gpu("NVIDIA GB10"))], []),
    )
    monkeypatch.setattr("lmds.hardware.profiler.detect_mem", lambda: (121.7, 76.5))
    monkeypatch.setattr("lmds.hardware.profiler.primary_ip", lambda: "10.2.1.138")
    monkeypatch.setattr("platform.node", lambda: "gigabyte02")

    host = host_summary()
    assert host.hostname == "gigabyte02"
    assert host.ip == "10.2.1.138"
    assert host.profile.value == "dgx-spark-single"
    assert host.ram_used_gb == 45.2  # 121.7 - 76.5


def test_cli_ps_shows_host_panel(tmp_path, monkeypatch, isolated_config):
    from lmds.hardware.profiler import HostSummary
    from lmds.hardware.profiles import TargetProfile

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "model-a", pid=os.getpid(), port=8000)
    monkeypatch.setattr(
        "lmds.hardware.host_summary",
        lambda: HostSummary(
            hostname="gigabyte02", ip="10.2.1.138", arch="aarch64",
            profile=TargetProfile.DGX_SPARK_SINGLE,
            ram_total_gb=121.7, ram_available_gb=40.0,
        ),
    )
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0
    assert "gigabyte02" in result.output
    assert "10.2.1.138" in result.output
    assert "dgx-spark-single" in result.output
    assert "81.7" in result.output  # RAM ใช้ไป
    assert "▰" in result.output  # แถบ RAM


def test_ram_bar_colors():
    from lmds.cli.main import _ram_bar

    assert "green" in _ram_bar(10, 100)
    assert "yellow" in _ram_bar(80, 100)
    assert "red" in _ram_bar(95, 100)
    assert "100%" in _ram_bar(100, 100)


def test_generated_controller_writes_meta(isolated_config, tmp_path):
    from tests.test_generator import gguf_report, make_bundle

    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "server.meta" in text
    assert "write_meta" in text
    for key in ["slug=", "engine=", "port=", "controller="]:
        assert key in text


_TAB = chr(9)
_NL = chr(10)
DOCKER_PS_SAMPLE = _NL.join([
    _TAB.join(["my-vllm", "vllm/vllm-openai:latest", "0.0.0.0:9000->8000/tcp"]),
    _TAB.join(["lmds-qwen3", "vllm/vllm-openai:latest", "0.0.0.0:8000->8000/tcp"]),
    _TAB.join(["postgres", "postgres:16", "5432/tcp"]),
]) + _NL


def test_docker_ps_adopts_model_servers_only():
    """container ที่คนอื่นรันไว้ต้องมองเห็นได้ แต่ container อื่น (db ฯลฯ) ต้องไม่ปนเข้ามา"""
    from lmds.fleet import manager

    found = manager._parse_docker_ps(DOCKER_PS_SAMPLE, set())
    by_slug = {s.slug: s for s in found}

    assert set(by_slug) == {"my-vllm", "qwen3"}  # postgres ไม่ใช่ model server
    assert by_slug["my-vllm"].external is True
    assert by_slug["my-vllm"].engine == "vllm"
    assert by_slug["my-vllm"].port == 9000
    assert by_slug["qwen3"].external is False  # ชื่อ lmds-* = ของเรา
    assert by_slug["qwen3"].container == "lmds-qwen3"


def test_docker_ps_skips_known_containers():
    from lmds.fleet import manager

    found = manager._parse_docker_ps(DOCKER_PS_SAMPLE, {"lmds-qwen3"})
    assert [s.slug for s in found] == ["my-vllm"]


def test_stop_external_container_uses_docker_stop(monkeypatch):
    """ของคนอื่น: หยุดอย่างเดียว ห้าม docker rm -f ทิ้ง"""
    from lmds.fleet import manager

    calls = []
    monkeypatch.setattr(
        manager.subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0, stdout=""),
    )
    external = manager.ServerInfo(
        slug="my-vllm", mode="docker", container="my-vllm", running=True,
        registered=False, external=True,
    )
    assert manager.stop_server(external) == "docker-stop"
    assert ["docker", "stop", "my-vllm"] in calls
    assert not any(c[:3] == ["docker", "rm", "-f"] for c in calls)


def test_stop_lmds_container_still_removes(monkeypatch):
    from lmds.fleet import manager

    calls = []
    monkeypatch.setattr(
        manager.subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0, stdout=""),
    )
    ours = manager.ServerInfo(slug="qwen3", mode="docker", container="lmds-qwen3", running=True)
    assert manager.stop_server(ours) == "docker-rm"
    assert ["docker", "rm", "-f", "lmds-qwen3"] in calls


def test_adopted_container_autostart_unit_uses_docker_start():
    from lmds.fleet import manager

    info = manager.ServerInfo(
        slug="my-vllm", mode="docker", container="my-vllm", running=True,
        registered=False, external=True,
    )
    unit = manager.render_unit(info)
    assert "ExecStart=/usr/bin/docker start my-vllm" in unit
    assert "ExecStop=/usr/bin/docker stop my-vllm" in unit
    assert "WantedBy=multi-user.target" in unit


def _bundle_like(tmp_path, slug="demo"):
    """สร้างโครงไฟล์เหมือน bundle จริง + ทะเบียน + weight ปลอม"""
    bundle_dir = tmp_path / "bundles" / slug
    bundle_dir.mkdir(parents=True)
    controller = bundle_dir / f"{slug}-single.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (bundle_dir / "README.md").write_text("x" * 100, encoding="utf-8")
    (tmp_path / "bundles" / f"{slug}.zip").write_text("y" * 50, encoding="utf-8")

    run_dir = tmp_path / "run" / slug
    run_dir.mkdir(parents=True)
    (run_dir / "server.log").write_text("z" * 10, encoding="utf-8")
    return bundle_dir, controller, run_dir


def test_removal_plan_lists_bundle_zip_and_registry(tmp_path, monkeypatch):
    from lmds.fleet import manager

    bundle_dir, controller, run_dir = _bundle_like(tmp_path)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    info = manager.ServerInfo(slug="demo", controller=str(controller), engine="vllm", mode="docker")

    labels = {i.label: i.path for i in manager.removal_plan(info)}
    assert labels["bundle"] == bundle_dir
    assert labels["zip"] == tmp_path / "bundles" / "demo.zip"
    assert labels["ทะเบียน/log"] == run_dir


def test_remove_deletes_everything_and_keep_weights_skips_them(tmp_path, monkeypatch):
    from lmds.fleet import manager

    bundle_dir, controller, run_dir = _bundle_like(tmp_path)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_text("w" * 1000, encoding="utf-8")
    monkeypatch.setattr(manager, "weights_path", lambda info: weights)
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)

    info = manager.ServerInfo(slug="demo", controller=str(controller), engine="vllm", mode="docker")

    # --keep-weights: ลบ bundle แต่ weight ต้องอยู่ครบ
    manager.remove_server(info, include_weights=False)
    assert not bundle_dir.exists() and not run_dir.exists()
    assert weights.is_dir()

    # รอบเต็ม: weight ต้องหายด้วย
    _bundle_like(tmp_path)
    manager.remove_server(info, include_weights=True)
    assert not weights.exists()


def test_repair_without_controller_explains_how_to_rebuild(tmp_path, monkeypatch):
    from lmds.fleet import FleetError, manager

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    info = manager.ServerInfo(slug="gone", controller="/no/such/ctl.sh")
    with pytest.raises(FleetError, match="lmds deploy"):
        manager.repair_server(info)


def test_repair_runs_download_then_verify(tmp_path, monkeypatch):
    from lmds.fleet import manager

    _, controller, _ = _bundle_like(tmp_path, "demo2")
    calls = []
    monkeypatch.setattr(manager, "_run_controller",
                        lambda info, cmd, extra=None: calls.append(cmd) or 0)
    info = manager.ServerInfo(slug="demo2", controller=str(controller))

    assert manager.repair_server(info) == 0
    assert calls == ["download", "verify-files"]
