"""ฟลีต "ตรง hub" 3 มิติ — `lmds fleet check` · `/api/fleet/consistency` · `lmds node install` · hub dirty guard"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.nodes import Node, add, status_from_probe, update

HUB_HASH = "feedfacecafe"


def _info(name: str, *, supported=True, ctl_ok=True, commit="dcefd91"):
    return {
        "host": {"lmds_version": "0.6.1", "lmds_commit": commit, "lmds_installed_commit": commit, "ip": "10.0.0.5",
                 "runtimes": {"llamacpp": [{"dir": f"/home/{name}/src/llama.cpp", "present": True, "build": "10495",
                                            "commit": "3dc7285b4", "date": "2026-08-18", "lock_state": "stale"}]}},
        "models": [{"slug": "qwen3-8-27b-gguf", "engine": "llamacpp", "downloaded": True, "generated_by": "lmds 0.6.1",
                    "template_hash": HUB_HASH if ctl_ok else "0ld0ld0ld0ld",
                    "controller": {"state": "ok" if ctl_ok else "stale", "generated_by": "0.6.1"},
                    "runtime_arch": {"arch": "qwen35", "mode": "native", "supported": supported,
                                     "runtime": "llama.cpp ~/src/llama.cpp", "fix": "LLAMA_CPP_UPDATE=1 ./x prepare-runtime"}}],
        "summary": {"total": 1, "running": 0, "healthy": 0, "not_downloaded": 0},
    }


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setattr("lmds.fleet.consistency.hub_facts",
                        lambda: {"version": "0.6.1", "commit": "dcefd91", "template_hash": HUB_HASH, "dirty": []})
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)


def test_fleet_check_cli_and_api_agree(hub, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.web import create_app, state

    infos = {"good": _info("good"), "rt": _info("rt", supported=False), "ctl": _info("ctl", ctl_ok=False),
             "behind": _info("behind", commit="0000000")}
    for name, info in infos.items():
        add(Node(name=name, host=f"10.0.0.{len(name)}", user="ops"))
        update(name, **status_from_probe(info))       # ทะเบียน (CLI อ่าน)
        state.STORE.set_node(name, info)              # แคช (API อ่าน)
    add(Node(name="never", host="10.0.0.99", user="ops"))   # ไม่เคย probe

    client = TestClient(create_app())
    api = client.get("/api/fleet/consistency").json()
    runner = CliRunner(env={"COLUMNS": "300"})   # rich ตัดคำตามความกว้างจอ — ข้อความสรุปต้องอยู่บรรทัดเดียว
    done = runner.invoke(app, ["fleet", "check", "--json"])
    assert done.exit_code == 1, done.output          # มีเครื่องแดง
    cli = json.loads(done.output.strip().splitlines()[-1])

    def states(report):
        return {n["name"]: (n["code"]["state"], n["controllers"]["state"], n["runtimes"]["state"], n["consistent"], n["level"])
                for n in report["nodes"]}

    assert states(api) == states(cli)
    assert states(api)["good"] == ("ok", "ok", "ok", True, "ok")
    assert states(api)["rt"] == ("ok", "ok", "stale", False, "bad")
    assert states(api)["ctl"] == ("ok", "stale", "ok", False, "bad")
    assert states(api)["behind"][0] == "behind" and states(api)["behind"][3] is False
    assert states(api)["never"] == ("unknown", "unknown", "unknown", False, "warn")
    # API มีรายละเอียดถึงชื่อ bundle (จากแคช) · CLI มีแค่ตัวนับ (จากทะเบียน) — สรุปตรงกัน
    assert next(n for n in api["nodes"] if n["name"] == "rt")["runtimes"]["items"] == ["qwen3-8-27b-gguf"]
    assert api["summary"]["consistent"] == cli["summary"]["consistent"] == 1
    assert api["summary"]["runtime_stale"] == 1 and api["summary"]["controllers_stale"] == 1
    assert "ตรง hub 1" in api["summary"]["line"] and "runtime ค้าง 1 (rt)" in api["summary"]["line"]

    # ตาราง (ไม่ใช่ --json) ก็ต้อง exit 1 และบอกว่าใครค้าง
    table = runner.invoke(app, ["fleet", "check"])
    assert table.exit_code == 1 and "Fleet consistency" in table.output and "rt" in table.output

    # `lmds node list` โชว์คอลัมน์ bundles/llama.cpp จากทะเบียนโดยไม่ SSH
    listing = runner.invoke(app, ["node", "list"])
    assert listing.exit_code == 0 and "runtime ค้าง 1" in listing.output and "10495" in listing.output, listing.output


def test_node_install_prints_three_axes_and_only_says_matches_when_all_ok(hub, monkeypatch):
    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    monkeypatch.setattr("lmds.nodes.install_lmds",
                        lambda node, with_prereq=False, force=False, timeout=1800: SimpleNamespace(ok=True, stdout="installed", stderr=""))
    rebuilt: list[str] = []
    monkeypatch.setattr("lmds.cli.main._update_runtime_on_node",
                        lambda node, slug: rebuilt.append(slug) or (0, "runtime พร้อม (pinned: newcommit)"))
    # probe แรก: build ไม่รู้จัก arch · หลัง update-runtime แล้ว: รู้จัก
    monkeypatch.setattr("lmds.nodes.probe", lambda node: _info("msi-4", supported=bool(rebuilt)))

    runner = CliRunner(env={"COLUMNS": "300"})   # rich ตัดคำตามความกว้างจอ — ข้อความสรุปต้องอยู่บรรทัดเดียว
    done = runner.invoke(app, ["node", "install", "msi-4"])
    assert done.exit_code == 0, done.output
    out = done.output
    assert "code" in out and "controllers" in out and "runtime" in out and "สรุป" in out
    assert "build llama.cpp ใหม่ให้ qwen3-8-27b-gguf" in out and rebuilt == ["qwen3-8-27b-gguf"]
    assert "ตรง hub ✓ (code ✓ · controller ✓ · runtime ✓)" in out

    rebuilt.clear()
    done = runner.invoke(app, ["node", "install", "msi-4", "--no-runtimes"])
    assert done.exit_code == 1, done.output
    assert rebuilt == [] and "ยังไม่ตรง hub" in done.output and "runtime ค้าง 1 (qwen3-8-27b-gguf)" in done.output
    assert "ตรง hub ✓" not in done.output

    done = runner.invoke(app, ["node", "install", "--all", "--no-runtimes"])
    assert done.exit_code == 1 and "อัปเดตครบ 1 เครื่อง · ตรง hub 0" in done.output and "runtime ค้าง 1 (msi-4)" in done.output


def test_node_install_refuses_when_hub_is_dirty_unless_forced(hub, monkeypatch):
    from lmds.nodes import HubDirtyError

    add(Node(name="msi-4", host="10.0.0.4", user="ops"))

    def install(node, with_prereq=False, force=False, timeout=1800):
        if not force:
            raise HubDirtyError(__import__("pathlib").Path("/hub"), ["src/lmds/inventory.py"])
        return SimpleNamespace(ok=True, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.install_lmds", install)
    monkeypatch.setattr("lmds.nodes.probe", lambda node: _info("msi-4"))
    runner = CliRunner(env={"COLUMNS": "300"})   # rich ตัดคำตามความกว้างจอ — ข้อความสรุปต้องอยู่บรรทัดเดียว
    done = runner.invoke(app, ["node", "install", "msi-4"])
    assert done.exit_code == 1 and "ไฟล์แก้ค้าง" in done.output and "--force" in done.output
    assert runner.invoke(app, ["node", "install", "msi-4", "--force"]).exit_code == 0


def test_web_install_refuses_dirty_hub_and_forces_a_reprobe_when_done(hub, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.nodes import HubDirtyError
    from lmds.web import create_app, jobs, state
    from tests.test_web import FakeStream, wait_for_job

    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    dirty = {"on": True}

    def prepare(node, with_prereq=False, force=False):
        if dirty["on"] and not force:
            raise HubDirtyError(__import__("pathlib").Path("/hub"), ["src/lmds/inventory.py"])
        return "echo install"

    monkeypatch.setattr("lmds.nodes.prepare_install", prepare)
    monkeypatch.setattr("lmds.nodes.stream", lambda node, command, *_, **__: FakeStream())
    client = TestClient(create_app())
    refused = client.post("/api/nodes/msi-4/install", json={})
    assert refused.status_code == 409 and "ไฟล์แก้ค้าง" in refused.json()["detail"]

    state.STORE.set_node("msi-4", _info("msi-4"))
    before = state.STORE.node_epoch("msi-4")
    forced = client.post("/api/nodes/msi-4/install", json={"force": True})
    assert forced.status_code == 200
    wait_for_job(client, forced.json()["job"]["id"])
    import time
    for _ in range(50):
        if state.STORE.node_epoch("msi-4") > before:
            break
        time.sleep(0.02)
    assert state.STORE.node_epoch("msi-4") > before, "งานจบต้อง STORE.force(name) — การ์ดห้ามโชว์ของเก่าอีก 15 วิ"
    assert state.STORE.due("msi-4")
    assert jobs.get(forced.json()["job"]["id"]).exit_code == 0
