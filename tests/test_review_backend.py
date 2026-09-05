"""รีวิวโค้ดทั้งระบบ 2026-09-04 (ชุด backend: fleet / nodes / brain / fit)

แต่ละเทสคือบั๊กที่ยืนยันแล้วก่อนแก้ — ชื่อเทสบอกว่าเดิมพังอย่างไร
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


# ── H1: bundle.env ถูก source เป็น bash — ค่าจากหน้าเว็บต้องรันคำสั่งไม่ได้ ──────────

@pytest.mark.parametrize("bad", ["x$(id)y", "a`id`b", 'q"uote', "it's", "brace}x", "back\\slash"])
def test_bundle_env_refuses_a_served_name_the_shell_would_execute(tmp_path, bad):
    from lmds.fleet.bundle_settings import FILENAME, SettingsError, write

    with pytest.raises(SettingsError):
        write(tmp_path, {"served_name": bad})
    assert not (tmp_path / FILENAME).exists()


def test_bundle_env_refuses_an_image_and_engine_env_that_break_out_of_the_quote(tmp_path):
    from lmds.fleet.bundle_settings import SettingsError, write

    with pytest.raises(SettingsError):
        write(tmp_path, {"image": "vllm/vllm-openai:latest$(id)"})
    # `}` ปิด ${VAR:-…} ก่อนเวลา — ที่เหลือกลายเป็นคำสั่ง
    with pytest.raises(SettingsError):
        write(tmp_path, {"engine_env": "A=b}c"})


def test_a_plain_served_name_is_written_verbatim_and_read_back(tmp_path):
    from lmds.fleet.bundle_settings import FILENAME, read, write

    write(tmp_path, {"served_name": "muse-glimmer.v2_x/8b", "image": "ghcr.io/org/vllm:26.05"})
    text = (tmp_path / FILENAME).read_text(encoding="utf-8")
    assert "$(" not in text and "`" not in text
    assert read(tmp_path)["served_name"] == "muse-glimmer.v2_x/8b"


# ── H2: ทะเบียนเครื่อง — load/แก้/save ต้องไม่แข่งกันจนค่าหาย ────────────────────────

def test_concurrent_registry_updates_do_not_lose_each_other(tmp_path, monkeypatch):
    from lmds.nodes import registry

    for name in ("n1", "n2", "n3"):
        registry.add(registry.Node(name=name, host=f"{name}.local", user="u"))

    def churn():
        for i in range(150):
            registry.update("n1", last_seen=f"t{i}")

    def edit():
        registry.update("n2", cluster_ip="10.0.0.2")
        registry.remove("n3")

    threads = [threading.Thread(target=churn), threading.Thread(target=edit)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    nodes = {n.name: n for n in registry.load()}
    assert nodes["n2"].cluster_ip == "10.0.0.2", "PATCH ระหว่าง refresher เขียนทับ = ค่าหาย"
    assert "n3" not in nodes, "forget แล้วเครื่องโผล่กลับ"
    assert nodes["n1"].last_seen == "t149"


# ── L3 / L4: ไฟล์ที่แก้ด้วยมือ ────────────────────────────────────────────────────────

def test_null_list_fields_in_a_hand_edited_nodes_yaml_fall_back_to_defaults():
    from lmds.nodes import registry

    registry.ensure_config_dir()
    registry.nodes_file().write_text(
        "nodes:\n  - name: n1\n    host: h\n    user: u\n    alt_hosts:\n    labels: null\n",
        encoding="utf-8")
    node = registry.load()[0]
    assert node.alt_hosts == [] and node.labels == []  # เดิมได้ None → TypeError ทุกที่ที่วน


def test_a_link_local_address_is_not_a_cluster_ip():
    from lmds.nodes.registry import NodeError, validate_cluster_ip

    with pytest.raises(NodeError):
        validate_cluster_ip("169.254.10.1")
    assert validate_cluster_ip("10.100.152.1") == "10.100.152.1"


# ── M1 / M2: llama.cpp กับ concurrency ──────────────────────────────────────────────

def _gguf_report():
    return ModelReport(
        repo_id="unsloth/Muse-Glimmer-30B-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=int(30.1 * GIB),
        selected_gguf="Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        context_length=131072,
        kv_dims=KvDims(layers=52, kv_heads=2, head_dim=128),
    )


def test_asking_for_concurrency_gives_that_many_slots_and_a_pool_sized_for_them():
    """เดิม --concurrency 4 ได้ slot เดียว + context หารสี่ — แย่กว่าไม่ใส่"""
    from lmds.brain import build_plan

    report = _gguf_report()
    single = analyze(report, PRESETS["dgx-spark-single"])
    fit = analyze(report, PRESETS["dgx-spark-single"], concurrency=4)
    plan = build_plan(report, fit, provider=None)
    assert plan.serving.max_num_seqs == 4
    assert plan.serving.context == fit.recommended_context * 4
    # แต่ละ slot ไม่เกินที่โมเดลรับได้ และ pool ทั้งก้อนต้องอยู่ใน KV budget
    assert fit.recommended_context <= single.recommended_context
    assert plan.serving.context // plan.serving.max_num_seqs == fit.recommended_context
    assert plan.serving.context * fit.kv_bytes_per_token <= fit.kv_budget_gb * GIB


def test_harden_clamps_the_context_before_sizing_the_output():
    """เดิมจัด output ให้พอดี slot ก่อน แล้วค่อยลด context → output ที่สัญญาโตกว่า slot"""
    from lmds.brain import build_plan
    from lmds.brain.orchestrator import harden_plan

    report = _gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.serving.context = fit.recommended_context * 4
    plan.serving.max_output_tokens = fit.recommended_context * 2
    hardened = harden_plan(plan, report, fit)
    assert hardened.serving.context == fit.recommended_context
    assert hardened.serving.max_output_tokens < hardened.serving.context


# ── M3: ขั้น prereq ใต้ sudo ต้องหาโฟลเดอร์ของผู้ใช้เจอ ───────────────────────────────

def test_the_prereq_step_does_not_cd_into_roots_home(monkeypatch):
    from lmds.nodes import Node, run_privileged

    seen = []

    def fake_run(node, command, timeout=60, stdin_text=""):
        seen.append(command)
        return SimpleNamespace(ok=bool(stdin_text), exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    run_privileged(Node(name="n", host="h", user="u"), "pw", with_prereq=True)
    install = next(c for c in seen if "install.sh" in c)
    assert "cd ~/AutoDeployDGXProject" not in install, "~ ใต้ sudo คือ /root"
    assert 'HOME="$HOME"' in install and "sudo -S" in install
    assert "chown -R" in install, "venv ที่ root สร้างต้องกลับเป็นของผู้ใช้"


def test_ssh_gives_up_on_a_dead_tcp_session():
    from lmds.nodes.ssh import _SSH_BASE

    assert "ServerAliveInterval=15" in _SSH_BASE


# ── M4: `lmds remove` ต้องไม่หยิบ MODEL_DIR ของโมเดลอื่นไปลบ ──────────────────────────

def test_weights_path_ignores_a_model_dir_left_in_the_environment(tmp_path, monkeypatch):
    from lmds.fleet.manager import weights_path

    monkeypatch.setenv("HOME", str(tmp_path))
    mine = tmp_path / "models" / "demo"
    mine.mkdir(parents=True)
    other = tmp_path / "models" / "someone-else"
    other.mkdir()
    monkeypatch.setenv("MODEL_DIR", str(other))
    info = SimpleNamespace(slug="demo", controller=None, controller_exists=False,
                           engine="llamacpp", model_id="")
    assert weights_path(info) == mine


# ── M5: slug คือชื่อโฟลเดอร์/ไฟล์/container — ห้ามมี path ──────────────────────────────

@pytest.mark.parametrize("bad", ["../x", "a/b", "has space", "UPPER", "", "x" * 64])
def test_adopt_refuses_a_slug_that_is_not_a_slug_before_touching_docker(monkeypatch, bad):
    import importlib

    adopt_mod = importlib.import_module("lmds.fleet.adopt")  # lmds.fleet.adopt ชื่อชนกับฟังก์ชัน
    from lmds.fleet.manager import FleetError

    def boom(*a, **k):
        raise AssertionError("ต้องปฏิเสธ slug ก่อนเรียก docker/proc")

    monkeypatch.setattr(adopt_mod, "inspect_container", boom)
    monkeypatch.setattr(adopt_mod, "inspect_process", boom)
    if bad:
        with pytest.raises(FleetError):
            adopt_mod.adopt("c", slug=bad)
        with pytest.raises(FleetError):
            adopt_mod.adopt_process(pid=1, slug=bad)
    else:
        assert adopt_mod._derive_slug("Org/Model_X") == "org-model-x"


# ── M6: dual-GPU ในเครื่องเดียว = node เดียว ──────────────────────────────────────────

def test_two_gpus_in_one_box_are_one_node_not_two():
    from lmds.fit.targets import from_hardware_report
    from lmds.hardware.profiler import DetectedGpu, HardwareReport
    from lmds.hardware.profiles import TargetProfile, lookup_gpu

    gpu = lookup_gpu("rtx pro 4000 blackwell")
    report = HardwareReport(
        arch="x86_64",
        gpus=[DetectedGpu("NVIDIA RTX PRO 4000 Blackwell", 24564, "12.0", gpu),
              DetectedGpu("NVIDIA RTX PRO 4000 Blackwell", 24564, "12.0", gpu)],
        ram_gb=128.0,
        profile=TargetProfile.RTX_MULTI_GPU,
    )
    spec = from_hardware_report(report)
    assert spec.gpu_count == 2 and spec.node_count == 1


# ── L1: adopt ต้องเก็บ HostIp ของ port ที่ container เคย bind ───────────────────────────

def test_adopt_keeps_the_host_ip_of_a_port_binding():
    from lmds.fleet.adopt import Adopted, render_controller

    adopted = Adopted(container="c", image="vllm/vllm-openai:v0.19.2",
                      args=["--port", "8000"],
                      ports={"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}],
                             "9000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9000"}]})
    script = render_controller(adopted, "c")
    assert "--publish 127.0.0.1:8000:8000" in script
    assert "--publish 9000:9000" in script


# ── พบระหว่าง rollout 0.6.0 ─────────────────────────────────────────────────────────

def test_the_refresher_stamps_last_seen_like_the_cli(monkeypatch):
    """เดิมมีแต่ CLI ที่เขียน last_seen → `lmds node list` โชว์ "เห็นล่าสุด" ค้างเป็นวัน"""
    from lmds.nodes import registry
    from lmds.web import state

    registry.add(registry.Node(name="n1", host="h", user="u"))
    monkeypatch.setattr("lmds.nodes.probe", lambda node, timeout=30: {"host": {"lmds_version": "0.6.0"}})
    state._refresh_node("n1")
    node = registry.find("n1")
    assert node.last_seen and node.last_seen[:4].isdigit()
    assert node.lmds_version == "0.6.0"


def test_the_web_server_does_not_wait_forever_for_sse_clients_on_shutdown(monkeypatch):
    """systemd ต้อง SIGKILL ทุก restart เพราะ uvicorn รอ /api/events ที่ไม่มีวันปิด"""
    import uvicorn

    from lmds.web import api

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("os._exit", lambda code: seen.setdefault("exit", code))
    api.serve(host="127.0.0.1", port=0, token="t")
    assert seen.get("timeout_graceful_shutdown", 0) <= 5
    assert seen.get("exit") == 0


def test_weights_path_prefers_the_location_adopt_recorded(tmp_path, monkeypatch):
    """adopt บันทึก profile["weights"]["path"] จาก bind mount — remove ต้องใช้ค่านั้น ไม่เดาจากชื่อ"""
    from lmds.fleet import manager

    monkeypatch.setenv("HOME", str(tmp_path))
    recorded = tmp_path / ".cache" / "huggingface" / "hub" / "models--nvidia--X"
    recorded.mkdir(parents=True)
    ctl = tmp_path / "bundles" / "x" / "x-adopted.sh"
    ctl.parent.mkdir(parents=True); ctl.write_text("#!/bin/bash\n")
    monkeypatch.setattr(manager, "bundle_profile",
                        lambda c: {"weights": {"path": str(recorded)}, "runtime": {"engine": "tensorrt-llm"}})
    info = SimpleNamespace(slug="x", controller=str(ctl), controller_exists=True, engine="tensorrt-llm", model_id="")
    assert manager.weights_path(info) == recorded
    # path นอก home ต้องไม่ถูกเชื่อ
    monkeypatch.setattr(manager, "bundle_profile", lambda c: {"weights": {"path": "/etc"}, "runtime": {"engine": "llamacpp"}})
    assert manager.weights_path(info) is None


# ── docker ใช้ไม่ได้เพราะอะไร + ปุ่มแก้บน hub (2026-09-05 ลูกค้า cynbangkok) ─────────────────────────

def test_docker_access_tells_the_group_problem_apart_from_a_missing_daemon(tmp_path, monkeypatch):
    """ติดตั้งผ่านปุ่ม Update แล้ว "มี Docker แต่ user ปัจจุบันเรียกไม่ได้" — การ์ดบนเว็บเดิมบอกแค่ not ready
    docker_access ต้องแยก: ไม่ได้ติดตั้ง / ไม่อยู่ในกลุ่ม / อยู่ในกลุ่มแต่ session เก่า / daemon ไม่รัน — พร้อมคำสั่งแก้"""
    from lmds.hardware import profiler

    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text('#!/bin/bash\necho "${FAKE_DOCKER_ERR}" >&2; exit "${FAKE_DOCKER_RC:-1}"\n', encoding="utf-8"); fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_DOCKER_ERR", "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock")
    monkeypatch.setattr(profiler.os, "getgroups", lambda: [])
    import grp as _grp

    class G:  # noqa: D401 — โครง struct_group ปลอม
        gr_mem, gr_gid = [], 999
    monkeypatch.setattr(_grp, "getgrnam", lambda name: G)
    denied = profiler.docker_access("cynbangkok")
    assert denied["installed"] and not denied["usable"] and not denied["in_group"]
    assert "not in the docker group" in denied["reason"] and "usermod -aG docker cynbangkok" in denied["fix"]
    assert "systemctl restart user@" in denied["fix"], "กลุ่มใหม่ไม่มีผลกับ session ที่รันอยู่ — ต้องบอกให้รีสตาร์ต"

    G.gr_mem = ["cynbangkok"]
    stale = profiler.docker_access("cynbangkok")
    assert stale["in_group"] and "re-login" in stale["reason"]

    monkeypatch.setenv("FAKE_DOCKER_ERR", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?")
    down = profiler.docker_access("cynbangkok")
    assert "daemon is not running" in down["reason"] and "enable --now docker" in down["fix"]

    monkeypatch.setenv("FAKE_DOCKER_RC", "0"); monkeypatch.setenv("FAKE_DOCKER_ERR", "")
    assert profiler.docker_access("cynbangkok")["usable"] is True
    monkeypatch.setenv("PATH", "/nonexistent")
    assert profiler.docker_access("x")["installed"] is False


def test_hub_fix_docker_route_adds_the_group_and_schedules_a_session_restart(monkeypatch):
    """POST /api/hub/fix-docker: usermod ผ่าน sudo -S (รหัสทาง stdin เท่านั้น) · ยังใช้ไม่ได้ทันทีเพราะ session เก่า
    → ถ้า lmds-web เป็น unit ที่ enable ไว้ ให้ systemd-run หน่วง 5 วิ restart user@<uid> · รหัสไม่โผล่ใน argv/response"""
    import subprocess as sp

    from fastapi.testclient import TestClient

    from lmds.web.api import create_app
    import lmds.web.api as api_mod

    calls: list[tuple[list[str], str]] = []
    cur = {"installed": True, "usable": False, "in_group": False, "user": "cyn", "reason": "cyn is not in the docker group", "fix": ""}

    def fake_run(args, **kw):
        calls.append((list(args), kw.get("input") or ""))
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            return sp.CompletedProcess(args, 0, stdout="enabled\n", stderr="")
        if "usermod" in args:   # เพิ่มกลุ่มแล้ว แต่ session นี้ยังไม่เห็นผล
            cur.update(in_group=True, reason="cyn is in the docker group but this session started before that — re-login needed")
        return sp.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sp, "run", fake_run)
    monkeypatch.setattr("lmds.hardware.profiler.docker_access", lambda user=None: dict(cur))
    monkeypatch.setattr("getpass.getuser", lambda: "cyn")
    client = TestClient(create_app())
    assert client.post("/api/hub/fix-docker", json={}).status_code == 400
    got = client.post("/api/hub/fix-docker", json={"password": "s3cret"})
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["ok"] and body["restart_scheduled"] and [s["step"] for s in body["steps"]][0].startswith("add cyn to the docker group")
    usermod = next(c for c in calls if "usermod" in c[0])
    assert usermod[0][:4] == ["sudo", "-S", "-p", ""] and usermod[1] == "s3cret\n" and "s3cret" not in " ".join(usermod[0])
    restart = next(c for c in calls if "systemd-run" in c[0])
    assert "--on-active=5" in restart[0] and any(a.startswith("user@") for a in restart[0])
    assert "s3cret" not in got.text
