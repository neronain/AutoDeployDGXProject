"""Audit รอบ 2 (2026-09-05) — hub orchestration + คอนโซลของ stacked deployment

ไล่เส้นทางของลูกค้าตั้งแต่กด "Deploy stacked to this group" จนโมเดลขึ้นบน head+worker และงานวันที่สอง
จากคอนโซล · ทุกข้อในไฟล์นี้ล้มก่อนแก้ · SSH ทุกสายเป็นของปลอม ไม่แตะเครื่องจริง · ตัวช่วย (spark/register/
FakeSSH/stacked_bundle) ยืมจาก tests/test_audit_stacked_orchestration.py — ฟลีตปลอมชุดเดียวกับรอบแรก
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from lmds.nodes import Node, add  # noqa: E402
from lmds.web import state  # noqa: E402
from tests.test_audit_stacked_orchestration import (  # noqa: E402
    PAGE,
    PUB,
    FakeSSH,
    cli,
    client,
    diagnose,
    failing,
    four_nodes,
    group_of,
    register,
    spark,
    stacked_bundle,
)


# ── fixtures (ชุดเดียวกับรอบแรก — autouse ของโมดูลนั้นไม่ข้ามไฟล์มา) ────────────────
@pytest.fixture(autouse=True)
def fresh_jobs():
    from lmds.web import jobs

    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    state.STORE.__init__()
    yield
    for job in jobs._JOBS.values():
        if job.process and job.running:
            job.process.kill()
    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    state.stop_refresher()
    state.STORE.__init__()


@pytest.fixture(autouse=True)
def no_host_scan(monkeypatch):
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "")
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: {
        "hostname": "hub", "gpus": [], "arch": "x86_64", "profile": "generic",
        "fabric": {"links": [{"iface": "eth0", "ip": "192.168.139.92", "prefix": 24,
                              "speed_gbps": 10, "state": "up", "connectx": False, "rdma": False}],
                   "best_gbps": 10, "tier": "basic", "summary": "eth0 10G"},
        "role": {"control_plane": True, "engines": []},
    })


@pytest.fixture
def pair():
    head = register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    worker = register("spark-worker", "10.2.1.194", "10.100.152.2", spark("10.100.152.2", "10.2.1.194"))
    return head, worker


def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def run_script(script: str, home: Path, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    """รันสคริปต์ที่ hub ส่งไป node จริง ๆ ใต้ bash — `~` ชี้ไป home ปลอม"""
    env = {**os.environ, "HOME": str(home)}
    if extra_path is not None:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)


def bundle_with_two_controllers(home: Path, slug: str, topology: str) -> Path:
    """bundle จริงบน spark-head (nvidia-nemotron-3-super-120b-a12b-nvfp4): มีทั้ง -single.sh กับ -stacked.sh
    (deploy single ก่อน แล้ว deploy stacked ทับก่อนที่ renderer จะรู้จัก .replaced-*) · profile บอกว่า stacked"""
    bundle = home / "bundles" / slug
    bundle.mkdir(parents=True)
    for kind in ("single", "stacked"):
        ctl = bundle / f"{slug}-{kind}.sh"
        ctl.write_text(f'#!/usr/bin/env bash\necho "{kind} $*"\n', encoding="utf-8")
        ctl.chmod(0o755)
    (bundle / "MODEL_PROFILE.yaml").write_text(
        f"slug: {slug}\nmodel:\n  id: org/{slug}\ntopology: {topology}\nruntime:\n  engine: vllm\n", encoding="utf-8")
    return bundle


# ── 1. wizard ตั้งค่าเครือข่าย: ติ๊กเดินสดระหว่าง apply + ขั้น firewall (0.6.1) มีชื่อของตัวเอง ──────────
def test_apply_status_streams_structured_steps_and_names_the_firewall_step(monkeypatch):
    """ลูกค้ากด Apply แล้วนั่งดูติ๊กต่อเครื่องเป็น "·" กับ log "Started…" อยู่หลายนาทีทั้งที่ hub กำลังเขียน
    netplan อยู่ — wizard อ่าน `steps` จาก GET /api/cluster/apply/{id} แต่ route นั้นส่งแค่ {id, running, job,
    result} (result มาตอนจบ) · และขั้น "firewall: allow cluster interfaces" ที่เพิ่มวันนี้ตกไปอยู่ในสาขา
    else ของ _net_step_lines → log ขึ้น "[a] pair SSH — firewall: … : paired" ซึ่งอ่านแล้วเข้าใจผิดว่าเป็น ssh"""
    register("a", "10.2.2.1", "", spark("10.100.152.1", "10.2.2.1"), site="TKC")
    register("b", "10.2.2.2", "", spark("10.100.152.2", "10.2.2.2"), site="TKC")
    steps = [
        {"node": "a", "step": "sudo password accepted", "ok": True, "detail": "", "level": "pass"},
        {"node": "a", "step": "stage netplan file", "ok": True, "detail": "/tmp/x", "level": "pass"},
        {"node": "a", "step": "firewall: allow cluster interfaces", "ok": True,
         "detail": "ufw active — allowed in on enp1s0f1np1", "level": "pass"},
        {"node": "b", "step": "firewall: allow cluster interfaces", "ok": False,
         "detail": "ufw allow failed for enp1s0f1np1 — run: sudo ufw allow in on <iface>", "level": "warn"},
    ]
    seen = {}

    def fake_apply(plan, passwords, *, nodes=None, runner=None, progress=None, **options):
        seen["progress"] = progress
        for step in steps:
            progress(step)
        return {"ok": True, "applied": True, "steps": steps, "nodes": {}, "pings": [], "pairing": [],
                "speed": [], "registry": {}}
    monkeypatch.setattr("lmds.nodes.netplan.apply_plan", fake_apply)

    api = client()
    plan = {"nodes": {"a": {}, "b": {}}, "order": ["a", "b"], "topology": "direct-2"}
    body = api.post("/api/cluster/apply", json={"plan": plan, "passwords": {}, "wait": True}).json()
    job_id = body["job"]["id"]

    status = api.get(f"/api/cluster/apply/{job_id}").json()
    assert status["steps"] == steps, "wizard ต้องได้ step แบบโครงสร้างระหว่างงานเดิน ไม่ใช่รอ result ตอนจบ"
    lines = api.get(f"/api/jobs/{job_id}").json()["output"].splitlines()
    assert "[a] firewall: ok — ufw active — allowed in on enp1s0f1np1" in lines, lines
    assert "[b] firewall: failed — ufw allow failed for enp1s0f1np1 — run: sudo ufw allow in on <iface>" in lines
    assert not any("pair SSH — firewall" in line for line in lines), lines


def test_wizard_ticks_the_firewall_step_live_while_apply_runs(tmp_path):
    """ฝั่งหน้าเว็บ: CNW_JOB_STEPS ไม่มีหมวด firewall → step นั้นไม่ขึ้นเป็นติ๊กเลย ลูกค้าไม่มีทางรู้ว่า hub เปิด
    ufw ให้แล้วหรือยัง (เคสจริง 2026-09-05: ping ถึงแต่ worker ต่อ master port ไม่ได้)"""
    from tests.test_cluster_wizard_ui import FLEET, HELPERS
    from tests.test_console_shell import run_scenario

    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        H.stepsPartial.push({ node: "spark-01", step: "firewall: allow cluster interfaces", ok: true,
                              detail: "ufw active — allowed in on enp1s0f1np1", level: "pass" });
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        for (const i of body().querySelectorAll("input.cnw-pw")) i.value = "hunter2-" + i.dataset.node;
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(12);
        const ticks = n => body().querySelectorAll(`.cnw-prog-row[data-node="${n}"] .cnw-tick`).map(t => t.dataset.step + "=" + t.dataset.state);
        const running = { s1: ticks("spark-01"), s2: ticks("spark-02"),
          log: document.getElementById("cnw-log").textContent.split("\\n") };
        await H.sleep(1500); await H.tick(16);
        console.log(JSON.stringify({ running, after: body().dataset.step, errors: H.errors, alerts: H.alerts }));""")
    r = out["running"]
    assert "firewall=ok" in r["s1"], r["s1"]
    assert "firewall=wait" in r["s2"], r["s2"]
    assert any(line.startswith("✓ [spark-01] firewall: allow cluster interfaces") for line in r["log"]), r["log"]
    assert out["after"] == "verify" and out["errors"] == [] and out["alerts"] == []


# ── 2. target 4 เครื่องจาก wizard: เลือก worker ได้ตัวเดียว ที่เหลือต้องมาจากกลุ่ม ───────────────
def test_a_wizard_that_picks_one_worker_for_a_four_node_bundle_gets_the_rest_from_the_group(tmp_path, monkeypatch):
    """analyze บอกว่า "ที่เหลือเติมตอนเขียน cluster.env" แต่ select_members ถือว่า worker ที่ระบุต้องครบพอดี →
    wizard stacked-4 จบด้วย 400 "built for 4 machines … 1 worker was chosen" ทุกครั้ง ไม่มีทางไปต่อจากหน้าเว็บ ·
    ตอนนี้ worker ที่ระบุมาก่อน (rank ตามลำดับที่เลือก) แล้วเติมที่เหลือจากกลุ่มตาม rank · เกินหรือไม่พอยังปฏิเสธ"""
    from lmds.nodes.stacked import StackedError, select_members

    trimmed = select_members([group_of("a", "b", "c", "d")], "a", workers=["c"], nnodes=4)
    assert [m["name"] for m in trimmed["members"]] == ["a", "c", "b", "d"]
    trimmed = select_members([group_of("a", "b", "c", "d")], "a", workers=["d", "b"], nnodes=4)
    assert [m["name"] for m in trimmed["members"]] == ["a", "d", "b", "c"]
    with pytest.raises(StackedError, match="built for 2 machines"):
        select_members([group_of("a", "b", "c", "d")], "a", workers=["b", "c"], nnodes=2)
    with pytest.raises(StackedError, match="built for 4 machines"):
        select_members([group_of("a", "b", "c")], "a", workers=["b"], nnodes=4)

    four_nodes()
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path / "bundles"])
    stacked_bundle(tmp_path, "four", 4)
    monkeypatch.setattr("lmds.nodes.run", FakeSSH(lambda node, cmd: (0, "/home/nvidia/bundles/four/cluster.env", "")))
    r = client().post("/api/cluster/write", json={"slug": "four", "head": "n1", "worker": "n3", "on": "n1"})
    assert r.status_code == 200, r.text
    assert r.json()["workers"] == ["n3", "n2", "n4"]
    assert r.json()["worker_ips"] == ["10.100.152.3", "10.100.152.2", "10.100.152.4"]


def test_the_wizard_pairs_every_worker_that_cluster_env_names():
    """หลังเขียน cluster.env ให้ bundle 4 เครื่อง wizard จับกุญแจให้แค่ draft.worker ตัวเดียว — worker rank 2–3
    ไม่มีกุญแจของ head แล้ว sync-worker ตายด้วย Permission denied ที่เครื่องที่สาม"""
    html = page()
    fn = html[html.index("async function writeClusterEnv"):html.index("async function pushAfterBuild")]
    assert "workers: d.workers" in fn, "writeClusterEnv ต้องคืนรายชื่อ worker ที่เขียนลง cluster.env"
    push = html[html.index("async function pushAfterBuild"):html.index("function wizBusy")]
    assert "pairClusterSsh(machine, cluster.workers" in push, "pair ต้องใช้รายชื่อจาก cluster.env ไม่ใช่ draft.worker ตัวเดียว"
    pairing = html[html.index("async function pairClusterSsh"):html.index("document.addEventListener", html.index("async function pairClusterSsh"))]
    assert "workers" in pairing and "JSON.stringify({ head, workers" in pairing


def test_analyze_note_for_a_four_node_target_says_the_wizard_fills_the_rest():
    """โน้ตของ analyze เคยบอกให้ไป "ระบุใน cluster.env (WORKER_IPS) บน head ก่อน start" เอง — ตรงข้ามกับสิ่งที่
    /api/cluster/write ทำให้แล้ว"""
    from lmds.web.deploy import _check_stacked_pair

    four_nodes()
    (note,) = _check_stacked_pair("dgx-spark-stacked-4", "n1", "n2", 4)
    assert "WORKER_IPS" not in note and "cluster.env" in note and "เติม" in note


# ── 3. ทะเบียนค้าง: cluster IP ที่ interface ไม่มีแล้ว ต้องหยุดก่อน push/start ─────────────────────
def test_a_stale_cluster_ip_blocks_the_group_and_the_cluster_env_write():
    """ทะเบียนบอก 10.100.152.9 แต่ enp1s0f1np1 ถือ 10.100.152.2 (เปลี่ยน IP หลัง apply/แก้มือ) · check_cluster_ip
    ตอบ mismatch แต่กลุ่มยัง ready → cluster.env ได้ IP ที่ไม่มีใครถือ → start ค้างที่ NCCL init · doctor เห็น
    แต่ push/wizard ไม่เคยถาม doctor"""
    from lmds.nodes import cluster as cl
    from lmds.nodes.stacked import StackedError, select_members

    machines = [
        {"name": "a", "host": spark("10.100.152.1", "10.2.2.1"), "cluster_ip": "10.100.152.1"},
        {"name": "b", "host": spark("10.100.152.2", "10.2.2.2"), "cluster_ip": "10.100.152.9"},
    ]
    (group,) = cl.cluster_groups(machines)
    assert not group["ready"]
    assert group["blockers"] == [{"kind": "stale-ip", "names": ["b"]}]
    assert group["members"][1]["suggested_ip"] == "10.100.152.2", "ต้องเสนอ IP ที่การ์ดถือจริงให้กดแก้"
    with pytest.raises(StackedError, match="stale-ip: b"):
        select_members([group], "a")

    slow = [
        {"name": "a", "host": spark("10.100.152.1", "10.2.2.1"), "cluster_ip": "10.100.152.1"},
        {"name": "b", "host": spark("10.100.152.2", "10.2.2.2"), "cluster_ip": "10.2.2.2"},
    ]
    (group,) = cl.cluster_groups(slow)
    assert "slow-link" in [b["kind"] for b in group["blockers"]]

    html = page()
    header = html[html.index("function groupHeader"):html.index("function layoutClusterGroups")]
    assert '"stale-ip"' in header and '"slow-link"' in header, "หน้าเว็บต้องเรียบเรียง blocker ใหม่เป็นประโยค"


def test_cli_cluster_show_survives_blocker_kinds_it_did_not_map(monkeypatch):
    """`lmds cluster show` เรียบเรียง blocker ด้วย dict[...] ตรง ๆ — กลุ่มที่ตั้งชื่อเองแต่ไม่มีวงร่วมกัน
    (`no-shared-fabric`) หรือ `stale-ip` ใหม่ ทำให้คำสั่งตายด้วย KeyError ทั้งตาราง"""
    register("a", "10.2.2.1", "10.100.152.1", None, site="TKC", cluster_name="lab")
    register("b", "10.2.2.2", "10.100.153.2", None, site="TKC", cluster_name="lab")
    hosts = {"a": spark("10.100.152.1", "10.2.2.1"), "b": spark("10.100.153.2", "10.2.2.2")}
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": hosts[node.name]})
    result = cli("cluster", "show")
    assert result.exit_code == 0, result.output
    assert "no-shared-fabric" in result.output or "วงร่วม" in result.output


# ── 4. SSH_USER เดียวของ controller vs ทะเบียนที่ user ต่างกัน ────────────────────────────────
def test_cluster_env_refuses_workers_that_log_in_as_different_users():
    """controller มี SSH_USER ค่าเดียวใช้กับ worker ทุกตัว · build_cluster_env หยิบ user ของ worker ตัวแรกเงียบ ๆ
    → rank 2 ที่ login ด้วยอีกชื่อได้ Permission denied ตอน sync-worker โดยไม่มีอะไรบอกว่าเพราะ user"""
    from lmds.fleet.cluster_env import ClusterEnvError, build_cluster_env

    add(Node(name="w1", host="10.2.2.2", user="nvidia", cluster_ip="10.100.152.2"))
    add(Node(name="w2", host="10.2.2.3", user="ops", cluster_ip="10.100.152.3"))
    with pytest.raises(ClusterEnvError) as err:
        build_cluster_env([group_of("h", "w1", "w2")], "h")
    assert "nvidia" in str(err.value) and "ops" in str(err.value) and "SSH_USER" in str(err.value)


def test_doctor_warns_when_head_and_worker_log_in_as_different_users(monkeypatch):
    """head login เป็น nvidia · worker เป็น ops → controller rsync ไป /home/nvidia/.cache/huggingface บน worker
    (WORKER_HF_HOME default = HF_HOME ของ head) ซึ่งไม่มี · doctor ต้องเตือนพร้อมชื่อตัวแปร และถ้า cluster.env
    บน head บอก SSH_USER คนละคนกับทะเบียน ต้องนับว่าไม่ตรง"""
    from lmds.nodes.doctor import describe

    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    node = add(Node(name="spark-worker", host="10.2.1.194", user="ops", cluster_ip="10.100.152.2", site="Neronain"))
    state.STORE.set_node(node.name, {"host": spark("10.100.152.2", "10.2.1.194"), "models": [], "summary": {}})
    good = "MASTER_IP=10.100.152.1\nWORKER_IPS=\"10.100.152.2\"\nSSH_USER=ops\n"
    report = diagnose(runner=FakeSSH(lambda n, cmd: (0, good if "cluster.env" in cmd else "", "")), slug="big")
    warns = {f["kind"]: f for f in report["findings"] if f["level"] == "warn"}
    assert "ssh-user" in warns, [f["kind"] for f in report["findings"]]
    text = describe(warns["ssh-user"], "en")
    assert "nvidia" in text and "ops" in text and "WORKER_HF_HOME" in text
    assert report["ok"] is True, "user ต่างกันเป็นเรื่องเตือน ไม่ใช่บล็อก"

    stale = "MASTER_IP=10.100.152.1\nWORKER_IPS=\"10.100.152.2\"\nSSH_USER=nvidia\n"
    bad = failing(diagnose(runner=FakeSSH(lambda n, cmd: (0, stale if "cluster.env" in cmd else "", "")), slug="big"))
    assert [k for k, f in bad.items() if f["level"] == "fail"] == ["cluster-env-match"], list(bad)
    assert "SSH_USER=nvidia" in bad["cluster-env-match"]["data"]["found"]
    assert "SSH_USER=ops" in bad["cluster-env-match"]["data"]["expected"]


def test_ssh_config_stanza_covers_every_link_ip_of_the_worker():
    """ring 3 เครื่อง: head ถึง worker rank 2 ด้วย IP อีกสาย (HEAD_TO_WORKER_IP_2) ไม่ใช่ cluster_ip ในทะเบียน ·
    stanza ที่ครอบแค่ cluster_ip + host บริหาร → ssh ไป IP สายนั้นไม่หยิบกุญแจ → Permission denied ทั้งที่ pair ผ่าน"""
    from lmds.nodes.cluster_ssh import config_stanza

    worker = Node(name="w", host="10.2.2.3", user="nvidia", cluster_ip="10.100.153.3",
                  cluster_links=[{"iface": "enp1s0f1np1", "ip": "10.100.153.3", "prefix": 30, "peer_node": "h"},
                                 {"iface": "enP2p1s0f1np1", "ip": "10.100.154.3", "prefix": 30, "peer_node": "x"}])
    host_line = next(l for l in config_stanza(worker, "10.100.153.3").splitlines() if l.startswith("Host "))
    assert host_line.split()[1:] == ["10.100.153.3", "10.100.154.3", "10.2.2.3"]


# ── 5. ปุ่มของ controller จาก hub บน bundle ที่มีสอง controller ─────────────────────────────────
def test_controller_buttons_pick_the_controller_the_profile_names(tmp_path, monkeypatch):
    """เคสจริง spark-head (nvidia-nemotron-3-super-120b-a12b-nvfp4): deploy single ก่อน แล้ว deploy stacked ทับ →
    โฟลเดอร์มีทั้ง -single.sh กับ -stacked.sh · hub เลือกด้วย `ls ./*-single.sh ./*-stacked.sh | head -1`
    = single เสมอ → กด sync-worker/verify-worker/logs-worker จากหน้าเว็บได้ "unknown command" ของ controller เดี่ยว"""
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    fake = FakeSSH()
    monkeypatch.setattr("lmds.nodes.run", fake)
    r = client().post("/api/nodes/spark-head/models/nemo/ctl/logs-worker")
    assert r.status_code == 200, r.text
    (script,) = fake.on("spark-head")

    bundle_with_two_controllers(tmp_path / "stacked-home", "nemo", "stacked")
    ran = run_script(script, tmp_path / "stacked-home")
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "stacked logs worker 200", ran.stdout

    bundle_with_two_controllers(tmp_path / "single-home", "nemo", "single")
    ran = run_script(script, tmp_path / "single-home")
    assert ran.stdout.strip() == "single logs worker 200", ran.stdout

    # bundle ที่ renderer รุ่นใหม่จัดการแล้ว (.replaced-*) ต้องไม่ถูกหยิบเป็น controller
    home = tmp_path / "replaced-home"
    bundle_with_two_controllers(home, "nemo", "stacked")
    (home / "bundles/nemo/nemo-single.sh").rename(home / "bundles/nemo/nemo-single.sh.replaced-20260905-120000")
    ran = run_script(script, home)
    assert ran.stdout.strip() == "stacked logs worker 200", ran.stdout


def test_download_on_a_stacked_head_syncs_the_worker_even_when_the_cache_has_not_seen_the_bundle(tmp_path, monkeypatch):
    """wizard push เสร็จแล้วบอกให้กด download ที่การ์ด head · hub ต่อ sync-worker/verify-worker ให้ *ถ้า* แคชของ
    refresher บอกว่า bundle นี้ stacked — แต่แคชเพิ่งถูก force หลัง push ยังไม่ทันสำรวจ → ได้ `lmds repair` เปล่า ๆ
    แล้ว start ตายที่ worker ด้วย "snapshot missing" · การตัดสินใจต้องเกิดบน node จาก MODEL_PROFILE ไม่ใช่จากแคช"""
    from lmds.web import jobs

    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"), models=[])
    captured = {}

    def fake_start_remote(node_name, slug, command, remote_command, secrets=None):
        captured["remote"] = remote_command
        return SimpleNamespace(payload=lambda: {"id": "j1", "running": True, "command": command})
    monkeypatch.setattr(jobs, "start_remote", fake_start_remote)
    r = client().post("/api/nodes/spark-head/models/nemo/repair", json={})
    assert r.status_code == 200, r.text
    remote = captured["remote"]
    assert "sync-worker" in remote and "verify-worker" in remote, remote

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "lmds").write_text('#!/usr/bin/env bash\necho "lmds $*"\n', encoding="utf-8")
    (bin_dir / "lmds").chmod(0o755)
    bundle_with_two_controllers(tmp_path / "stacked-home", "nemo", "stacked")
    ran = run_script(remote, tmp_path / "stacked-home", bin_dir)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.split("\n")[:3] == ["lmds repair nemo", "stacked sync-worker", "stacked verify-worker"], ran.stdout

    bundle_with_two_controllers(tmp_path / "single-home", "nemo", "single")
    ran = run_script(remote, tmp_path / "single-home", bin_dir)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "lmds repair nemo", "bundle เดี่ยวต้องไม่แตะ worker"


# ── 6. cluster.env ต้องไปอยู่กับ bundle บน head ไม่ใช่สำเนาบน hub ────────────────────────────────
def test_cluster_write_defaults_to_the_head_when_on_is_omitted(tmp_path, monkeypatch):
    """ผู้เรียก API ที่ไม่ส่ง `on` (สคริปต์/ผู้ช่วย) ได้ cluster.env ในสำเนา bundle บน hub ซึ่งไม่มีวันถูกรัน ·
    bundle ที่รันจริงอยู่บน head เสมอ — ไม่ระบุ = head"""
    four_nodes()
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path / "bundles"])
    bundle = stacked_bundle(tmp_path, "two", 2)
    fake = FakeSSH(lambda node, cmd: (0, "/home/nvidia/bundles/two/cluster.env", ""))
    monkeypatch.setattr("lmds.nodes.run", fake)
    r = client().post("/api/cluster/write", json={"slug": "two", "head": "n1"})
    assert r.status_code == 200, r.text
    assert r.json()["target"].startswith("n1:"), r.json()
    assert any("cluster.env" in c for c in fake.on("n1"))
    assert not (bundle / "cluster.env").exists(), "ห้ามเขียนลงสำเนาบน hub เมื่อไม่ได้ขอ"


# ── 7. การ์ดโมเดลบน hub เอง ────────────────────────────────────────────────────────────────
def test_the_local_model_card_says_how_many_nodes_the_stacked_bundle_spans():
    """หัวข้อ "Stacked (2 nodes)" ถูกเขียนตายตัว — bundle stacked-4 บนเครื่องนี้ก็ขึ้น 2 nodes"""
    html = page()
    assert "Stacked (2 nodes)" not in html
    card = html[html.index('<span class="sec-title">Stacked'):]
    assert "cluster.nnodes" in card[:400]
