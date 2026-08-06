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


def _docker_server_with_host_visible_process(tmp_path, monkeypatch, in_container: bool):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "gemma-4-12b-it-gguf", mode="docker", port=8000)
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda container: True)
    monkeypatch.setattr("lmds.fleet.manager._health_ok", lambda port: False)
    monkeypatch.setattr(
        "lmds.fleet.manager._pgrep_llama",
        lambda: [(4242, "llama-server --alias gemma-4-12b-it-ud-q8_k-xl --port 8000")],
    )
    monkeypatch.setattr("lmds.fleet.manager._in_container", lambda pid: in_container)
    return discover()


def test_docker_llamacpp_server_is_not_listed_twice(tmp_path, monkeypatch):
    """เคสจริง RTX 5090 (2026-08-03): llama.cpp โหมด docker โผล่ใน pgrep ของ host ด้วย

    ผลคือ `lmds list` ขึ้นสองแถวสำหรับเซิร์ฟเวอร์ตัวเดียว — แถวปลอมใช้ค่า --alias เป็น slug
    จึงดูเหมือนคนละโมเดล และสั่ง stop ตามชื่อนั้นไม่ได้
    """
    servers = _docker_server_with_host_visible_process(tmp_path, monkeypatch, in_container=True)
    assert [s.slug for s in servers] == ["gemma-4-12b-it-gguf"]


def test_same_port_never_counted_as_two_servers(tmp_path, monkeypatch):
    """กันชั้นที่สอง เผื่ออ่าน /proc/<pid>/cgroup ไม่ได้ (เช่น rootless/แพลตฟอร์มอื่น)"""
    servers = _docker_server_with_host_visible_process(tmp_path, monkeypatch, in_container=False)
    assert [s.slug for s in servers] == ["gemma-4-12b-it-gguf"]


def test_real_native_orphan_on_another_port_still_shows(tmp_path, monkeypatch):
    """ของจริงที่ต้องเก็บตกต้องไม่หายไปกับการกันซ้ำ"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "registered", mode="docker", port=8000)
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda container: True)
    monkeypatch.setattr("lmds.fleet.manager._health_ok", lambda port: False)
    monkeypatch.setattr(
        "lmds.fleet.manager._pgrep_llama",
        lambda: [(777, "llama-server --alias other-model --port 8001")],
    )
    monkeypatch.setattr("lmds.fleet.manager._in_container", lambda pid: False)

    slugs = {s.slug for s in discover()}
    assert slugs == {"registered", "other-model"}


def _make_bundle(directory, slug, model_id="org/demo", engine="llamacpp"):
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    controller = directory / f"{slug}-single.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    controller.chmod(0o755)
    (directory / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": model_id, "revision": "sha", "served_name": slug},
        "runtime": {"engine": engine}, "serving": {"context": 16384}, "topology": "single",
    }), encoding="utf-8")
    return controller


def test_bundles_on_disk_are_found_without_a_registry(tmp_path, monkeypatch):
    """รายงานจากผู้ใช้: deploy เสร็จแล้ว `lmds list`/หน้าเว็บไม่เห็นอะไรเลย ไปต่อไม่ถูก

    ทะเบียนถูกเขียนโดย controller ตอน start เท่านั้น — bundle ที่ยังไม่เคย start จึงหายไป
    """
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.chdir(tmp_path)
    _make_bundle(tmp_path / "bundles" / "demo-a", "demo-a")
    # เคสจริงบนเครื่องผู้ใช้: deploy จากในโฟลเดอร์ bundles เลยได้ bundles/bundles/
    _make_bundle(tmp_path / "bundles" / "bundles" / "demo-b", "demo-b")

    slugs = {s.slug for s in discover()}
    assert {"demo-a", "demo-b"} <= slugs


def test_scanned_bundle_is_marked_unregistered(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.chdir(tmp_path)
    _make_bundle(tmp_path / "bundles" / "demo", "demo", model_id="unsloth/demo-GGUF")

    found = next(s for s in discover() if s.slug == "demo")
    assert found.registered is False
    assert found.model_id == "unsloth/demo-GGUF"
    assert found.controller_exists is True


def test_registry_wins_over_disk_scan(tmp_path, monkeypatch):
    """เคย start แล้ว = ทะเบียนของ controller ละเอียดกว่า ห้ามถูกของที่สแกนมาทับ"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.chdir(tmp_path)
    controller = _make_bundle(tmp_path / "bundles" / "demo", "demo")
    make_meta(tmp_path / "run", "demo", mode="docker", port=8123, controller=str(controller))

    matches = [s for s in discover() if s.slug == "demo"]
    assert len(matches) == 1
    assert matches[0].registered is True
    assert matches[0].port == 8123


def test_register_bundle_does_not_overwrite_controller_registry(tmp_path, monkeypatch):
    from lmds.fleet import register_bundle

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    controller = _make_bundle(tmp_path / "bundles" / "demo", "demo")
    make_meta(tmp_path / "run", "demo", port=9999, controller=str(controller))

    register_bundle(controller)
    assert "port=9999" in (tmp_path / "run" / "demo" / "server.meta").read_text(encoding="utf-8")


def test_extra_bundle_dirs_from_env(tmp_path, monkeypatch):
    """bundle ที่ย้ายไปไว้ที่อื่น (เช่นดิสก์ลูกอื่น) ต้องบอก LMDS ได้"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "data" / "my-bundles"
    _make_bundle(elsewhere / "far-away", "far-away")

    assert "far-away" not in {s.slug for s in discover()}
    monkeypatch.setenv("LMDS_BUNDLE_DIRS", str(elsewhere))
    assert "far-away" in {s.slug for s in discover()}


def test_dead_registration_is_dropped(tmp_path, monkeypatch):
    """ทะเบียนที่ชี้ไป controller ที่ไม่มีแล้วและไม่ได้รันอยู่ = ทำอะไรกับมันไม่ได้เลย
    เกิดตอน generate bundle ไว้ที่อื่นแล้วลบทิ้ง — ปล่อยไว้จะเต็มหน้าจอด้วยรายการปลอม"""
    from lmds.fleet import discover

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._scan_bundles", lambda known: [])

    # ไม่มี started_at = ไม่เคยถูก start มาก่อน (generate อย่างเดียว)
    run_dir = tmp_path / "run" / "ghost"
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        "slug=ghost\nengine=vllm\nmode=docker\nport=8000\ncontroller=/ไม่มี/ที่นี่.sh\n",
        encoding="utf-8")

    assert [s.slug for s in discover()] == []
    assert not run_dir.exists(), "ทะเบียนที่ตายแล้วต้องถูกเก็บกวาด"


def test_running_server_survives_even_without_its_controller(tmp_path, monkeypatch):
    """ยังรันอยู่ = ต้องเห็นและสั่ง stop ได้ ถึงไฟล์ controller จะหายไปแล้วก็ตาม"""
    from lmds.fleet import discover

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run2"))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: True)
    monkeypatch.setattr("lmds.fleet.manager._health_ok", lambda port: False)
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._scan_bundles", lambda known: [])

    run_dir = tmp_path / "run2" / "alive"
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        "slug=alive\nengine=vllm\nmode=docker\nport=8000\ncontainer=c\ncontroller=/ไม่มี.sh\n",
        encoding="utf-8")

    assert [s.slug for s in discover()] == ["alive"]


def test_previously_started_model_keeps_its_warning(tmp_path, monkeypatch):
    """เคยรันจริงแล้ว controller หายไป = เรื่องที่ผู้ใช้ต้องรู้ ไม่ใช่เก็บกวาดเงียบ ๆ
    (ต่างจาก bundle ที่แค่ generate ไว้แล้วลบทิ้ง)"""
    from lmds.fleet import discover

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run3"))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._scan_bundles", lambda known: [])

    run_dir = tmp_path / "run3" / "was-running"
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        "slug=was-running\nengine=vllm\nmode=docker\nport=8000\ncontroller=/ไม่มี.sh\n"
        "started_at=2026-08-01T10:00:00\n", encoding="utf-8")

    found = discover()
    assert [s.slug for s in found] == ["was-running"]
    assert not found[0].controller_exists


def test_prune_removes_only_dead_registrations(tmp_path, monkeypatch, isolated_config):
    """เครื่องที่ใช้จัดการอย่างเดียวจะสะสมทะเบียนของ bundle ที่ย้าย/ลบไปแล้ว
    ล้างได้ต้องลบเฉพาะไฟล์ทะเบียน ไม่แตะของที่ยังใช้ได้"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._scan_bundles", lambda known: [])

    live = tmp_path / "ctl.sh"
    live.write_text("#!/bin/bash\n", encoding="utf-8")
    for slug, controller in (("dead", "/ไม่มี.sh"), ("alive", str(live))):
        run_dir = tmp_path / "run" / slug
        run_dir.mkdir(parents=True)
        (run_dir / "server.meta").write_text(
            f"slug={slug}\nengine=vllm\nmode=docker\nport=8000\ncontroller={controller}\n"
            "started_at=2026-08-01T10:00:00\n", encoding="utf-8")

    result = runner.invoke(app, ["prune", "--yes"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "run" / "dead").exists()
    assert (tmp_path / "run" / "alive" / "server.meta").exists()
    assert live.exists(), "prune ต้องไม่แตะไฟล์ controller"


def test_tests_never_touch_the_real_run_root():
    """เทสเคยเขียนทะเบียนลง ~/.lmds/run ของเครื่องจริง แล้วทิ้งรายการค้างให้ผู้ใช้เห็น"""
    import os
    from pathlib import Path

    from lmds.fleet import run_root

    real = Path.home() / ".lmds" / "run"
    assert os.environ.get("LMDS_RUN_ROOT"), "conftest ต้องแยก LMDS_RUN_ROOT ทุกเทส"
    assert run_root() != real


def test_start_passes_unknown_flags_to_the_controller(tmp_path, monkeypatch, isolated_config):
    """`lmds start x --port 8001` เคยตอบ "No such option: --port" ทั้งที่ controller รองรับ
    — คนอ่านคำแนะนำจาก lmds list แล้วพิมพ์ตามจะเจอ error ทันที (เจอจริงบน dgx-msi)"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\necho \"$@\"\n", encoding="utf-8")
    controller.chmod(0o755)
    make_meta(tmp_path, "m", controller=str(controller))

    seen = {}
    monkeypatch.setattr("lmds.fleet.manager._run_controller",
                        lambda info, command, extra=None: seen.update(cmd=command, extra=extra) or 0)
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    result = runner.invoke(app, ["start", "m", "--port", "8001", "--gpu-util", "0.8"])
    assert result.exit_code == 0, result.output
    assert seen["cmd"] == "start"
    assert seen["extra"] == ["--port", "8001", "--gpu-util", "0.8"]


def test_restart_passes_unknown_flags_to_the_controller(tmp_path, monkeypatch, isolated_config):
    """start ส่ง flag ต่อได้แล้ว แต่ restart เคยรับ flag มาแล้วทิ้งเงียบ — ผู้ใช้เห็น
    "restarted" ทั้งที่ port ไม่ได้เปลี่ยน ซึ่งแย่กว่า error เพราะไม่มีอะไรบอกว่าพลาด
    """
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    controller.chmod(0o755)
    make_meta(tmp_path, "m", controller=str(controller))

    seen = {}
    monkeypatch.setattr("lmds.fleet.manager._run_controller",
                        lambda info, command, extra=None: seen.update(cmd=command, extra=extra) or 0)
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    result = runner.invoke(app, ["restart", "m", "--port", "8001"])
    assert result.exit_code == 0, result.output
    assert seen == {"cmd": "restart", "extra": ["--port", "8001"]}


def test_logs_explains_a_model_that_never_ran(tmp_path, monkeypatch, isolated_config):
    """เดิมปล่อย "tail: cannot open ..." ดิบ ๆ ออกไป ซึ่งอ่านเหมือนระบบพัง
    ทั้งที่แค่ยังไม่เคยสตาร์ต (เจอจริงตอนกดปุ่ม logs บนหน้าเว็บ)
    """
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    controller.chmod(0o755)
    # ไม่มี meta เลย — เหมือน bundle ที่ deploy ไว้แต่ยังไม่เคย start (เคสจริงบน dgx-veerasiam)
    meta = make_meta(tmp_path, "m", controller=str(controller)) / "server.meta"
    meta.write_text(meta.read_text(encoding="utf-8").replace(
        "started_at=2026-07-21T12:00:00", "started_at="), encoding="utf-8")
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    called = []
    monkeypatch.setattr("lmds.fleet.manager._run_controller",
                        lambda *a, **k: called.append(a) or 0)

    result = runner.invoke(app, ["logs", "m"])
    assert result.exit_code == 1
    assert "ยังไม่เคยรัน" in result.output
    assert "lmds start m" in result.output
    assert not called, "ไม่ควรไปเรียก controller ให้ tail ไฟล์ที่ไม่มี"


def test_running_context_beats_the_bundle_value(tmp_path, monkeypatch, isolated_config):
    """ผู้ใช้ตั้ง context ตอน start (65,600) แต่หน้าเว็บโชว์ค่าใน bundle (16,384) ต่อไปเรื่อย ๆ
    — ดูแล้วเหมือนช่องที่กรอกไม่ทำงาน ทั้งที่ค่าไปถึง llama-server ถูกต้อง (ผู้ใช้เจอจริง)
    """
    from lmds.fleet import running_context

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    make_meta(tmp_path, "m", mode="docker")
    server = SimpleNamespace(running=True, mode="docker", container="lmds-m", pid=0, pid_file="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="--ctx-size 65600 --parallel 1", stderr=""))
    assert running_context(server) == 65600


def test_running_context_reads_vllm_flag_too(monkeypatch):
    from lmds.fleet import running_context

    server = SimpleNamespace(running=True, mode="docker", container="lmds-m", pid=0, pid_file="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="--max-model-len 32768", stderr=""))
    assert running_context(server) == 32768


def test_running_context_is_none_when_stopped():
    """ไม่ได้รันอยู่ = ไม่มีค่าที่ "กำลังใช้" ให้รายงาน · ต้องตกไปใช้ค่าใน bundle"""
    from lmds.fleet import running_context

    assert running_context(SimpleNamespace(running=False, mode="docker", container="x",
                                           pid=0, pid_file="")) is None


def test_running_context_survives_a_docker_that_says_nothing(monkeypatch):
    from lmds.fleet import running_context

    server = SimpleNamespace(running=True, mode="docker", container="lmds-m", pid=0, pid_file="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="no such container"))
    assert running_context(server) is None


def test_remove_does_not_claim_success_when_files_survive(tmp_path, monkeypatch, isolated_config):
    """rmtree ลบสิ่งที่ลบได้แล้วโยน error ตัวเดียว — ของที่เหลือยังอยู่จริง
    เดิม CLI พิมพ์ "ลบเรียบร้อย" ทับบรรทัด "ลบไม่ได้" ผู้ใช้จะรู้ตัวตอนดิสก์ไม่ลด
    (เคสจริง: weight ที่ container โหลดมาเป็น root เหลือ 23 GB)
    """
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    controller.chmod(0o755)
    make_meta(tmp_path / "run", "m", controller=str(controller))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    stubborn = tmp_path / "weights"
    stubborn.mkdir()
    (stubborn / "big.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr("lmds.fleet.manager.removal_plan", lambda info, include_weights=True: [
        SimpleNamespace(label="weight ของโมเดล", path=stubborn, size_bytes=4096)])
    monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)   # เงียบ ๆ ไม่ลบจริง

    result = runner.invoke(app, ["remove", "m", "-y"])
    assert result.exit_code == 2, result.output
    assert "ไม่ครบ" in result.output
    assert "sudo rm -rf" in result.output, "ต้องบอกคำสั่งที่ทำให้จบได้จริง"
    assert "เรียบร้อย" not in result.output


def test_remove_reports_success_only_when_everything_is_gone(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    controller.chmod(0o755)
    make_meta(tmp_path / "run", "m", controller=str(controller))
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    gone = tmp_path / "weights"
    gone.mkdir()
    monkeypatch.setattr("lmds.fleet.manager.removal_plan", lambda info, include_weights=True: [
        SimpleNamespace(label="weight ของโมเดล", path=gone, size_bytes=0)])

    result = runner.invoke(app, ["remove", "m", "-y"])
    assert result.exit_code == 0, result.output
    assert "เรียบร้อย" in result.output
