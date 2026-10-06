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
        # ทะเบียน (CLI อ่าน) — พร้อม last_seen อย่างที่ทุกจุดที่ probe จริงเขียน: ไม่มีเวลา = "ยังไม่เคย probe" ซึ่งไม่นับว่าตรง
        update(name, last_seen=_stamp(), **status_from_probe(info))
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


def test_control_plane_hub_runtime_axis_is_not_applicable():
    """เคสจริง 2026-09-07: hub (control-plane ไม่มี GPU) ถือ bundle GGUF 6 ใบไว้ push → มิติ runtime เคย "ตรวจไม่ได้" → hub
    เป็น warn ทั้งที่ทุก node ตรง · ต้องเป็น n/a และ hub นับว่า consistent"""
    from lmds.fleet.consistency import node_verdict

    hub = {"version": "0.6.1", "commit": "34f83bb", "template_hash": "f5ce29c2d47a", "dirty": []}
    host = {"lmds_version": "0.6.1", "lmds_commit": "34f83bb", "lmds_installed_commit": "34f83bb",
            "template_hash": "f5ce29c2d47a", "role": {"control_plane": True}, "runtimes": {"llamacpp": []}}
    models = [{"slug": "qwen3-8-flash-next-uncensored-gguf", "engine": "llamacpp", "template_hash": "f5ce29c2d47a",
               "generated_by": "0.6.1", "runtime_arch": {"arch": "qwen4exp", "supported": None}}]
    v = node_verdict({"host": host, "models": models}, hub)
    assert v.runtimes.state == "n/a", v.runtimes
    assert v.consistent and v.level == "ok", v.payload()
    # เครื่องธรรมดาที่ runtime ยังตอบไม่ได้ ยังต้องเป็น unknown เหมือนเดิม
    host_gpu = {**host, "role": {"control_plane": False}}
    v2 = node_verdict({"host": host_gpu, "models": models}, hub)
    assert v2.runtimes.state == "unknown" and not v2.consistent


def test_discovered_containers_do_not_make_a_node_look_inconsistent():
    """เคสจริง 2026-09-08: container ที่คนอื่น start เอง (`vllm-gemma4` บน dgx-spark01, `vllm-qwen3-122b` บน msi-5)
    โผล่ใน inventory เป็น external=True ไม่มี controller → ถูกนับเป็น "ตรวจไม่ได้" ทำให้ทั้งเครื่องขึ้น "ยังไม่ตรง hub"
    ทั้งที่ bundle ของ LMDS ตรง template หมด"""
    from lmds.fleet.consistency import controllers_axis

    hub = {"version": "0.6.1", "commit": "b9112bd", "template_hash": "f5ce29c2d47a", "dirty": []}
    models = [
        {"slug": "nvidia-nemotron-3-super-120b-a12b-nvfp4", "generated_by": "lmds 0.6.1",
         "template_hash": "f5ce29c2d47a", "controller_exists": True},
        {"slug": "vllm-gemma4", "external": True, "controller_exists": False},
    ]
    axis = controllers_axis(models, hub)
    assert axis.state == "ok", axis
    assert "1 ใบ" in axis.detail and "container ที่ค้นพบเอง 1" in axis.detail

    only_external = controllers_axis([{"slug": "vllm-gemma4", "external": True, "controller_exists": False}], hub)
    assert only_external.state == "n/a", only_external


# ── ความสดของข้อมูล: `fleet check` เคยอ่านแต่ทะเบียนเสมอ ─────────────────────────────
#
# หน้าเว็บมีตัว refresh เบื้องหลังคอยอุ่นแคชไว้ · CLI เป็น process ครั้งเดียวจบจึงส่ง {}
# เข้า fleet_report ทุกครั้ง แล้วตกไปใช้ตัวนับจากทะเบียน = ของรอบที่ probe ล่าสุด
# เคสจริง: `lmds node install` เสร็จแล้ว `fleet check` ยังบอกเวอร์ชันเก่า คนอ่านคิดว่า install ไม่ติด

def test_fleet_check_without_check_says_out_loud_that_the_numbers_are_not_live(hub, monkeypatch):
    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    update("msi-4", **status_from_probe(_info("msi-4")))
    monkeypatch.setattr("lmds.nodes.probe", lambda node: (_ for _ in ()).throw(AssertionError("ต้องไม่ SSH")))

    done = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check"])
    assert done.exit_code == 0, done.output
    assert "มาจากทะเบียน ไม่ได้ต่อเข้าเครื่องจริง" in done.output
    assert "lmds fleet check --check" in done.output


def test_fleet_check_with_check_probes_every_machine_and_reports_now(hub, monkeypatch):
    """ตัวเลขต้องมาจากการต่อจริงรอบนี้ ไม่ใช่จากที่ทะเบียนจำไว้ก่อนหน้า"""
    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    update("msi-4", **status_from_probe(_info("msi-4", ctl_ok=False)))   # ทะเบียนจำว่า controller ค้าง

    probed: list[str] = []

    def fake_probe(node):
        probed.append(node.name)
        return _info(node.name)                                          # ของจริงตอนนี้: ตรงแล้ว
    monkeypatch.setattr("lmds.nodes.probe", fake_probe)

    stale = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check", "--json"])
    assert json.loads(stale.output.strip().splitlines()[-1])["nodes"][0]["controllers"]["state"] == "stale"
    assert not probed, "ไม่ใส่ --check ต้องไม่ SSH"

    live = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check", "--check", "--json"])
    assert live.exit_code == 0, live.output
    node = json.loads(live.output.strip().splitlines()[-1])["nodes"][0]
    assert probed == ["msi-4"]
    assert node["controllers"]["state"] == "ok" and node["source"] == "cache"


def test_probing_writes_the_result_back_so_the_next_run_is_not_stale_again(hub, monkeypatch):
    """เดิมผู้ใช้ต้องสั่ง `lmds node list --check` ก่อนเองทุกครั้ง — ทำให้แทนเลย"""
    from lmds.nodes import find as find_node

    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    update("msi-4", **status_from_probe(_info("msi-4", ctl_ok=False)))
    assert find_node("msi-4").controllers_stale == 1

    monkeypatch.setattr("lmds.nodes.probe", lambda node: _info(node.name))
    assert CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check", "--check"]).exit_code == 0

    after = find_node("msi-4")
    assert after.controllers_stale == 0 and after.last_seen, "ผลต้องถูกเขียนกลับทะเบียน"


def test_a_machine_that_cannot_be_reached_falls_back_instead_of_failing_the_whole_report(hub, monkeypatch):
    """เครื่องหนึ่งดับไม่ควรทำให้รายงานทั้งฟลีตใช้ไม่ได้ — ถอยไปใช้ของที่ทะเบียนจำไว้"""
    from lmds.nodes import NodeError, find as find_node

    add(Node(name="up", host="10.0.0.1", user="ops"))
    add(Node(name="down", host="10.0.0.2", user="ops"))
    for name in ("up", "down"):
        update(name, **status_from_probe(_info(name)))

    def fake_probe(node):
        if node.name == "down":
            raise NodeError("ssh: connect to host 10.0.0.2 port 22: No route to host")
        return _info(node.name)
    monkeypatch.setattr("lmds.nodes.probe", fake_probe)

    done = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check", "--check", "--json"])
    report = json.loads(done.output.strip().splitlines()[-1])
    sources = {n["name"]: n["source"] for n in report["nodes"]}
    assert sources == {"up": "cache", "down": "registry"}
    assert "No route to host" in find_node("down").last_error


def test_every_row_carries_when_its_data_was_taken(hub):
    """ตัวเลขที่เก่าอ่านเหมือนตัวเลขที่ใหม่ — ผู้อ่านรายงานต้องบอกได้ว่าเป็นของเมื่อไหร่"""
    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    update("msi-4", last_seen="2026-09-20T10:00:00", **status_from_probe(_info("msi-4")))

    done = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check", "--json"])
    node = json.loads(done.output.strip().splitlines()[-1])["nodes"][0]
    assert node["last_seen"] == "2026-09-20T10:00:00"

    table = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check"])
    assert "ข้อมูลเมื่อ" in table.output


def test_a_machine_that_was_never_probed_is_labelled_as_such(hub):
    add(Node(name="never", host="10.0.0.99", user="ops"))
    table = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check"])
    assert "ยังไม่เคย probe" in table.output


# ── ทะเบียนต้องไม่เปลี่ยน "ตรวจไม่ได้" เป็น "ตรง hub ✓" · ของที่จำไว้ไม่ใช่ของที่ตรวจตอนนี้ ──────────
#
# audit 2026-10-06: probe ก้อนเดียวกันได้คำตัดสินสองแบบ — ทางเต็ม (การ์ดบนเว็บ/แคช) บอก
# "ยังไม่ตรง hub — controller ตรวจไม่ได้ … (code ✓ · controller ? · runtime ?)" · ทางทะเบียน (`lmds fleet check`
# ไม่ใส่ --check และทางสำรองของทุกเครื่องที่แคชไม่มีข้อมูล) บอก "ตรง hub ✓ (code ✓ · controller ✓ · runtime ✓)"
# เพราะทะเบียนจำแค่จำนวน `stale` กับ `supported is False` — `unknown` / `supported: None` นับเป็น 0 แล้ว 0 = ผ่าน
# ขัดกับกติกาของโมดูลเอง ("null = ตรวจไม่ได้ ไม่ใช่ผ่าน") · และเครื่องที่ต่อไม่ได้ตอนนี้ถูกนับเข้า "ตรง hub N"
# จากตัวเลขที่จำไว้

HUB = {"version": "0.9.0", "commit": "6e2b474", "template_hash": "aaaa1111", "dirty": []}


def _stamp(minutes_ago: float = 0) -> str:
    from datetime import datetime, timedelta

    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M")   # รูปแบบที่ CLI/refresher เขียน


def _host(**extra) -> dict:
    return {"lmds_version": "0.9.0", "lmds_commit": "6e2b474", "lmds_installed_commit": "6e2b474",
            "template_hash": "aaaa1111", "ip": "10.2.1.11", **extra}


def _bundle(slug="qwen3.8-flash-gguf", *, engine="llamacpp", ctl="ok", supported=True, **extra) -> dict:
    """หนึ่งโมเดลในรูปที่ `lmds agent info` ส่งจริง (inventory.model_payload)"""
    model = {"slug": slug, "engine": engine, "downloaded": True, "controller_exists": True, "external": False,
             "generated_by": "lmds 0.9.0", "template_hash": "aaaa1111" if ctl != "stale" else "0ld0ld0ld0ld",
             "controller": {"state": ctl, "generated_by": "0.9.0", "template_hash": "aaaa1111"},
             "runtime_arch": None}
    if engine == "llamacpp":
        model["runtime_arch"] = {"arch": "qwen4exp", "mode": "native", "supported": supported,
                                 "runtime": "llama.cpp ~/src/llama.cpp", "fix": "LLAMA_CPP_UPDATE=1 ./x prepare-runtime"}
    model.update(extra)
    return model


# โปรไฟล์อ่านไม่ได้: inventory ส่ง controller=None และ generated_by=None (ไม่ใช่ {"state": "unknown"})
_UNREADABLE = {"slug": "broken-profile", "engine": "vllm", "downloaded": True, "controller_exists": True,
               "external": False, "generated_by": None, "template_hash": None, "controller": None, "runtime_arch": None}

_AUDIT_PROBE = {   # ก้อนที่ผู้ตรวจใช้ repro: runtime ยังไม่ได้ build (supported=None) + โปรไฟล์ใบที่สองอ่านไม่ได้
    "host": _host(),
    "models": [_bundle(supported=None),
               {"slug": "broken-profile", "engine": "vllm",
                "controller": {"state": "unknown", "generated_by": "", "reason": "อ่าน MODEL_PROFILE.yaml ไม่ได้"}}],
}


def _remembered(info: dict, **fields) -> Node:
    """Node อย่างที่ทะเบียนจะจำหลัง probe ก้อนนี้ — เพิ่ง probe เมื่อครู่ เว้นแต่จะบอกเป็นอย่างอื่น"""
    fields.setdefault("last_seen", _stamp())
    return Node(name="msi-1", host="10.2.1.11", user="tkc", **{**status_from_probe(info), **fields})


def _picture(verdict) -> tuple:
    """สิ่งที่ผู้ใช้เห็น: ✓ / ? / ชื่อปัญหา ต่อมิติ + นับว่าตรงไหม + สีของแถว (n/a กับ ok คือ ✓ เหมือนกัน)"""
    def shown(axis):
        return "ok" if axis.ok else axis.state
    return (shown(verdict.code), shown(verdict.controllers), shown(verdict.runtimes), verdict.consistent, verdict.level)


def test_the_registry_does_not_turn_cannot_verify_into_matches_the_hub():
    from lmds.fleet.consistency import fleet_report, node_verdict, summary_line, verdict_from_registry

    full = node_verdict(_AUDIT_PROBE, HUB)
    assert _picture(full) == ("ok", "unknown", "unknown", False, "warn")

    node = _remembered(_AUDIT_PROBE)
    remembered = verdict_from_registry(node, HUB)
    assert _picture(remembered) == _picture(full), summary_line(remembered)
    assert "ตรง hub ✓" not in summary_line(remembered)
    assert "controller ?" in summary_line(remembered) and "runtime ?" in summary_line(remembered)

    report = fleet_report({}, [node], None, HUB)
    assert report["summary"]["consistent"] == 0 and report["summary"]["unknown"] == 1
    assert "ตรง hub 0" in report["summary"]["line"] and "ตรวจไม่ได้ 1 (msi-1)" in report["summary"]["line"]


@pytest.mark.parametrize("label, info", [
    ("ทุกอย่างตรง", {"host": _host(), "models": [_bundle(), _bundle("gemma-4-vllm", engine="vllm")]}),
    ("โปรไฟล์อ่านไม่ได้ (controller=None)", {"host": _host(), "models": [_bundle(), _UNREADABLE]}),
    ("runtime ยังไม่ได้ build (supported=None)", {"host": _host(), "models": [_bundle(supported=None)]}),
    ("llama.cpp ที่อ่าน arch ไม่ได้ (runtime_arch=None)", {"host": _host(), "models": [_bundle(runtime_arch=None)]}),
    ("controller ค้าง + อีกใบตรวจไม่ได้", {"host": _host(), "models": [_bundle(ctl="stale"), _UNREADABLE]}),
    ("runtime ไม่รู้จัก arch", {"host": _host(), "models": [_bundle(supported=False)]}),
    ("adopted — ไม่มี template", {"host": _host(), "models": [_bundle("coder-next", engine="vllm", ctl="adopted")]}),
    ("container ที่ค้นพบเอง ไม่ใช่ bundle", {"host": _host(), "models": [
        _bundle("gemma-4-vllm", engine="vllm"),
        {"slug": "vllm-gemma4", "engine": "vllm", "external": True, "controller_exists": False,
         "generated_by": None, "template_hash": None, "controller": None, "runtime_arch": None}]}),
    ("control plane ถือ bundle ไว้ push", {"host": _host(role={"control_plane": True}),
                                         "models": [_bundle(supported=None)]}),
    ("เครื่องใหม่ยังไม่มี bundle", {"host": _host(), "models": []}),
])
def test_what_the_registry_remembers_gives_the_same_verdict_as_the_probe_it_came_from(label, info):
    """นิยามเดียว: probe ก้อนไหนก็ตาม ทางทะเบียนต้องตัดสินเหมือนทางเต็ม — ไม่มีทางไหน "ใจดีกว่า" """
    from lmds.fleet.consistency import node_verdict, summary_line, verdict_from_registry

    full = node_verdict(info, HUB)
    remembered = verdict_from_registry(_remembered(info), HUB)
    assert _picture(remembered) == _picture(full), f"{label}: ทะเบียน → {summary_line(remembered)} · เต็ม → {summary_line(full)}"


def test_a_state_the_registry_has_no_word_for_is_never_rendered_as_ok():
    """ahead (controller ใหม่กว่า lmds บนเครื่องนั้น) หรือ state ที่เพิ่มมาวันหน้า — ทะเบียนนับเฉพาะที่ *รู้ว่าผ่าน* เป็นผ่าน"""
    from lmds.fleet.consistency import verdict_from_registry

    for odd in ("ahead", "something-new", "", None):
        info = {"host": _host(), "models": [_bundle(engine="vllm", ctl="ok"),
                                           {**_bundle("odd", engine="vllm"), "controller": {"state": odd}}]}
        verdict = verdict_from_registry(_remembered(info), HUB)
        assert not verdict.controllers.ok and not verdict.consistent, odd


def test_a_registry_written_before_this_release_loads_and_reads_as_unknown_not_ok(hub):
    """nodes.yaml เดิมมีแต่ controllers_stale/runtime_stale — 0 ในไฟล์เก่าแปลว่า "ไม่มีใบที่ stale" ไม่ได้แปลว่า
    "ไม่มีใบที่ตรวจไม่ได้" (ตอนนั้นยังไม่มีใครนับ) → ต้องอ่านได้ ไม่ล้ม และขึ้น ? จนกว่าจะ probe ใหม่"""
    from lmds.fleet.consistency import verdict_from_registry
    from lmds.nodes import load
    from lmds.nodes.registry import nodes_file

    nodes_file().parent.mkdir(parents=True, exist_ok=True)
    nodes_file().write_text(
        "nodes:\n- name: msi-1\n  host: 10.2.1.11\n  user: tkc\n  port: 22\n"
        f"  last_seen: '{_stamp()}'\n  last_error: ''\n  lmds_version: 0.6.1\n  lmds_commit: dcefd91\n"
        "  controllers_stale: 0\n  runtime_stale: 0\n  restart_pending: 0\n  llamacpp_build: 10495 · 2026-08-18\n",
        encoding="utf-8")

    (node,) = load()
    assert node.controllers_unknown is None and node.runtime_unknown is None
    verdict = verdict_from_registry(node)
    assert _picture(verdict) == ("ok", "unknown", "unknown", False, "warn")

    runner = CliRunner(env={"COLUMNS": "300"})
    checked = runner.invoke(app, ["fleet", "check"])
    assert checked.exit_code == 0, checked.output              # ไม่รู้ ≠ ผิด — ไม่ exit 1
    assert "ตรง hub 0" in checked.output and "ตรวจไม่ได้ 1 (msi-1)" in checked.output, checked.output
    listed = runner.invoke(app, ["node", "list"])
    assert listed.exit_code == 0 and "ตรง hub" not in listed.output, listed.output

    # ทะเบียนเก่าที่จำว่ามีใบ stale ยังเป็น "ค้าง" (รู้ว่าผิด) ไม่ถูกลดเป็นแค่ "ไม่รู้"
    update("msi-1", controllers_stale=2)
    assert verdict_from_registry(load()[0]).controllers.state == "stale"


def test_a_machine_that_cannot_be_reached_now_is_not_counted_as_matching_from_memory(hub, monkeypatch):
    """เครื่องดับ: ตัวเลขที่จำไว้ยังสวยทุกช่อง — แต่ "ตรง hub" คือคำยืนยันของตอนนี้ ไม่ใช่ของเมื่อสามชั่วโมงก่อน"""
    from lmds.nodes import NodeError

    add(Node(name="up", host="10.0.0.1", user="ops"))
    add(Node(name="down", host="10.0.0.2", user="ops"))
    update("up", last_seen=_stamp(), **status_from_probe(_info("up")))
    update("down", last_seen=_stamp(minutes_ago=185), **status_from_probe(_info("down")))

    def fake_probe(node):
        if node.name == "down":
            raise NodeError("ต่อ ops@10.0.0.2 ไม่ได้: ssh: connect to host 10.0.0.2 port 22: No route to host")
        return _info(node.name)
    monkeypatch.setattr("lmds.nodes.probe", fake_probe)

    runner = CliRunner(env={"COLUMNS": "300"})
    done = runner.invoke(app, ["fleet", "check", "--check", "--json"])
    report = json.loads(done.output.strip().splitlines()[-1])
    rows = {n["name"]: n for n in report["nodes"]}
    assert rows["up"]["consistent"] is True and rows["up"]["verified"] is True
    down = rows["down"]
    assert down["consistent"] is False and down["verified"] is False and down["level"] == "warn"
    assert (down["code"]["state"], down["controllers"]["state"], down["runtimes"]["state"]) == ("ok", "ok", "ok"), \
        "สิ่งที่จำไว้ยังโชว์ตามที่จำ — แค่ไม่นับว่ายืนยันแล้ว"
    assert "3 ชม." in down["unverified"] and "No route to host" in down["unverified"], down["unverified"]
    assert report["summary"]["consistent"] == 1
    assert "ตรง hub 1" in report["summary"]["line"] and "ยังไม่ได้ตรวจตอนนี้ 1 (down)" in report["summary"]["line"]

    table = runner.invoke(app, ["fleet", "check", "--check"])
    assert table.exit_code == 0, table.output                  # ต่อไม่ได้ ≠ ไม่ตรง — เหลือง ไม่ใช่แดง
    assert "ยังไม่ได้ตรวจตอนนี้" in table.output and "ชม.ก่อน" in table.output


def test_the_web_report_does_not_count_an_unreachable_machine_either(hub):
    """การ์ด Fleet consistency: refresher ต่อไม่ได้ → แคชไม่มี data → ตกไปทางทะเบียน — ทางเดียวกับข้างบน"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.web import create_app, state

    for name in ("up", "down"):
        add(Node(name=name, host=f"10.0.0.{len(name)}", user="ops"))
        update(name, last_seen=_stamp(), **status_from_probe(_info(name)))
    state.STORE.set_node("up", _info("up"))
    state.STORE.set_node("down", None, "ต่อ ops@10.0.0.4 ไม่ได้: ssh: connect to host 10.0.0.4 port 22: No route to host")
    update("down", last_error="ต่อ ops@10.0.0.4 ไม่ได้: ssh: connect to host 10.0.0.4 port 22: No route to host")

    report = TestClient(create_app()).get("/api/fleet/consistency").json()
    rows = {n["name"]: n for n in report["nodes"]}
    assert rows["down"]["consistent"] is False and rows["down"]["verified"] is False
    assert report["summary"]["consistent"] == 1 and report["summary"]["unverified"] == 1


def test_remembered_numbers_count_only_while_they_are_fresh(hub):
    """ไม่ใส่ --check ยังต้องมีประโยชน์: hub ที่เปิดเว็บ refresher เขียนทะเบียนทุก 15 วิ — ของที่เพิ่ง probe = ของตอนนี้
    · แต่ของเมื่อวานไม่ใช่ ไม่ว่าตัวเลขจะสวยแค่ไหน"""
    from lmds.fleet.consistency import fleet_report
    from lmds.nodes import load

    add(Node(name="fresh", host="10.0.0.1", user="ops"))
    add(Node(name="yesterday", host="10.0.0.2", user="ops"))
    update("fresh", last_seen=_stamp(minutes_ago=1), **status_from_probe(_info("fresh")))
    update("yesterday", last_seen=_stamp(minutes_ago=60 * 26), **status_from_probe(_info("yesterday")))

    rows = {n["name"]: n for n in fleet_report({}, load(), None)["nodes"]}
    assert rows["fresh"]["consistent"] is True and rows["fresh"]["verified"] is True
    assert rows["yesterday"]["consistent"] is False and rows["yesterday"]["verified"] is False
    assert "26 ชม.ก่อน" in rows["yesterday"]["unverified"], rows["yesterday"]["unverified"]

    out = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["fleet", "check"]).output
    assert "ตรง hub 1" in out and "ยังไม่ได้ตรวจตอนนี้ 1 (yesterday)" in out, out


def test_node_list_does_not_print_matches_for_bundles_it_could_not_check(hub):
    add(Node(name="msi-1", host="10.2.1.11", user="tkc"))
    update("msi-1", last_seen=_stamp(), **status_from_probe(_AUDIT_PROBE))

    out = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["node", "list"]).output
    assert "controller ตรวจไม่ได้ 1" in out and "runtime ตรวจไม่ได้ 1" in out, out
    assert "ตรง hub" not in out, out
