"""Audit ฝั่ง hub ของ stacked deployment (vLLM TP ข้าม 2–4 DGX Spark) — 2026-09-04

ลูกค้ารายงาน: "download ไม่ผ่าน · analyze ไม่ผ่าน · runtime ไม่ผ่าน · multi-node ไม่เคยติด"
ไล่เส้นทางจากหน้าเว็บ/CLI ไปจนถึง controller บน head แล้วพบว่าข้อต่อระหว่าง hub กับ
controller ขาดหลายจุด — ทุกข้อในไฟล์นี้ล้มก่อนแก้ · SSH ทุกสายเป็นของปลอม ไม่แตะเครื่องจริง
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.nodes import Node, add  # noqa: E402
from lmds.web import create_app, state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src/lmds/web/static/index.html"
HARNESS = Path(__file__).with_name("console_shell_dom.js")


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def fresh_jobs():
    from lmds.web import jobs

    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    yield
    for job in jobs._JOBS.values():
        if job.process and job.running:
            job.process.kill()
    jobs._JOBS.clear()
    jobs._ACTIVE.clear()


@pytest.fixture(autouse=True)
def no_host_scan(monkeypatch):
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "")
    # hub = VM ควบคุม ไม่มี GPU สาย 10G — ตรงกับ hub จริง (Autodeploy) ที่ลูกค้าใช้
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: {
        "hostname": "hub", "gpus": [], "arch": "x86_64", "profile": "generic",
        "fabric": {"links": [{"iface": "eth0", "ip": "192.168.139.92", "prefix": 24,
                              "speed_gbps": 10, "state": "up", "connectx": False, "rdma": False}],
                   "best_gbps": 10, "tier": "basic", "summary": "eth0 10G"},
        "role": {"control_plane": True, "engines": []},
    })


def spark(ip: str, mgmt: str, *, disk: float = 900.0, speed: int = 200, state_: str = "up") -> dict:
    """host payload ของ DGX Spark หนึ่งเครื่อง — สายเร็ว + สายบริหาร + พอร์ต link-local ที่ยังไม่ตั้ง"""
    return {
        "hostname": f"spark-{ip.rsplit('.', 1)[-1]}", "arch": "aarch64", "profile": "dgx_spark",
        "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128}], "disk_free_gb": disk,
        "fabric": {"links": [
            {"iface": "enp1s0f1np1", "ip": ip, "prefix": 24, "speed_gbps": speed, "state": state_,
             "driver": "mlx5_core", "connectx": True, "rdma": True},
            {"iface": "enp1s0f0np0", "ip": "169.254.9.9", "prefix": 16, "speed_gbps": 200,
             "state": "up", "connectx": True, "rdma": True, "link_local": True},
            {"iface": "enP7s7", "ip": mgmt, "prefix": 24, "speed_gbps": 1, "state": "up",
             "connectx": False, "rdma": False},
        ], "rdma_devices": ["rocep1s0f1"], "best_gbps": 200, "tier": "rdma", "cluster_capable": True},
    }


def register(name: str, mgmt: str, cluster_ip: str, host: dict | None, *, site="Neronain",
             models: list | None = None, **fields) -> Node:
    node = add(Node(name=name, host=mgmt, user="nvidia", cluster_ip=cluster_ip, site=site, **fields))
    if host is not None:
        state.STORE.set_node(name, {"host": host, "models": models or [],
                                    "summary": {"total": len(models or []), "running": 0}})
    else:
        state.STORE.set_node(name, None, "ssh: connect timed out")
    return node


@pytest.fixture
def pair():
    """spark-head + spark-worker ตั้ง cluster IP ครบ อยู่ไซต์เดียวกัน — คู่จริงที่ hub มีอยู่"""
    head = register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    worker = register("spark-worker", "10.2.1.194", "10.100.152.2", spark("10.100.152.2", "10.2.1.194"))
    return head, worker


class FakeSSH:
    """`lmds.nodes.run` ปลอม — จำทุกคำสั่งพร้อมชื่อเครื่อง และให้เทสกำหนดคำตอบตามคำสั่ง"""

    def __init__(self, answer=None):
        self.calls: list[tuple[str, str]] = []
        self.answer = answer or (lambda node, command: (0, "", ""))

    def __call__(self, node, command, timeout=60, stdin_text=""):
        self.calls.append((node.name, command))
        code, out, err = self.answer(node, command)
        return SimpleNamespace(ok=code == 0, exit_code=code, stdout=out, stderr=err)

    def on(self, name: str) -> list[str]:
        return [c for n, c in self.calls if n == name]


def stacked_bundle(root: Path, slug: str, nnodes: int = 2) -> Path:
    bundle = root / "bundles" / slug
    bundle.mkdir(parents=True)
    (bundle / f"{slug}-stacked.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'TENSOR_PARALLEL_SIZE="${{TENSOR_PARALLEL_SIZE:-{nnodes}}}"\n'
        f'NNODES="${{NNODES:-{nnodes}}}"\n'
        'case "${1:-}" in sync-worker|verify-worker|start|download) echo "$1";; esac\n',
        encoding="utf-8")
    (bundle / "MODEL_PROFILE.yaml").write_text(
        f"slug: {slug}\ntopology: stacked\nruntime:\n  engine: vllm\nserving:\n  context: 8192\n",
        encoding="utf-8")
    return bundle


# ── 1. cluster.env ต้องตรงกับจำนวนเครื่องที่ bundle ถูก render มา ─────────────────
def group_of(*names: str, ready=True) -> dict:
    return {"ready": ready, "site": "Neronain", "blockers": [], "members": [
        {"name": n, "cluster_ip": f"10.100.152.{i + 1}", "iface": "enp1s0f1np1", "state": "ok"}
        for i, n in enumerate(names)]}


def test_bundle_node_count_is_read_from_the_controller_that_will_run(tmp_path):
    from lmds.nodes.stacked import bundle_node_count

    assert bundle_node_count(stacked_bundle(tmp_path, "four", 4)) == 4
    assert bundle_node_count(stacked_bundle(tmp_path, "two", 2)) == 2
    single = tmp_path / "bundles" / "one"
    single.mkdir()
    (single / "one-single.sh").write_text("#!/bin/bash\n")
    assert bundle_node_count(single) == 1
    assert bundle_node_count(tmp_path / "nope") == 1


def test_a_two_node_bundle_in_a_four_node_group_gets_exactly_one_worker():
    """เดิม build_cluster_env เขียน worker ทุกตัวในกลุ่ม → NNODES=4/TP=4 ทับแผนที่คิดมาสำหรับ 2"""
    from lmds.fleet.cluster_env import build_cluster_env
    from lmds.nodes.stacked import select_members

    trimmed = select_members([group_of("a", "b", "c", "d")], "a", nnodes=2)
    assert [m["name"] for m in trimmed["members"]] == ["a", "b"]
    body = build_cluster_env([trimmed], "a")["body"]
    assert "NNODES=2\n" in body and "TENSOR_PARALLEL_SIZE=2\n" in body
    assert 'WORKER_IPS="10.100.152.2"' in body


def test_a_four_node_bundle_writes_every_worker_in_rank_order():
    from lmds.fleet.cluster_env import build_cluster_env
    from lmds.nodes.stacked import select_members

    trimmed = select_members([group_of("a", "b", "c", "d")], "a", workers=["c", "b", "d"], nnodes=4)
    body = build_cluster_env([trimmed], "a")["body"]
    assert "NNODES=4\n" in body
    assert 'WORKER_IPS="10.100.152.3 10.100.152.2 10.100.152.4"' in body, "ลำดับที่เลือก = node-rank"


def test_a_four_node_bundle_with_a_two_machine_group_is_refused_with_the_reason():
    from lmds.nodes.stacked import StackedError, select_members

    with pytest.raises(StackedError) as err:
        select_members([group_of("a", "b")], "a", nnodes=4)
    assert "built for 4 machines" in str(err.value) and "dgx-spark-stacked-4" in str(err.value)

    with pytest.raises(StackedError, match="cannot be both head and worker"):
        select_members([group_of("a", "b")], "a", workers=["a"])
    with pytest.raises(StackedError, match="not a member of any ready cluster group"):
        select_members([group_of("a", "b")], "zzz")
    with pytest.raises(StackedError, match="missing-ip: b"):
        blocked = group_of("a", "b", ready=False)
        blocked["blockers"] = [{"kind": "missing-ip", "names": ["b"]}]
        select_members([blocked], "a")


# ── 2. กุญแจ head → worker ───────────────────────────────────────────────────
PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE lmds-cluster-spark-head"


def test_pairing_makes_the_key_on_the_head_and_authorizes_it_on_the_worker(pair):
    """hub ติดตั้งแต่กุญแจของตัวเองลง node — head ไม่เคยมีทางไป worker · นี่คือข้อต่อที่หายไป"""
    from lmds.nodes.cluster_ssh import pair_workers

    head, worker = pair
    fake = FakeSSH(lambda node, cmd: (0, PUB + "\n", "") if "ssh-keygen" in cmd else (0, "", ""))
    steps = pair_workers(head, [(worker, worker.cluster_ip)], runner=fake)

    assert all(s["ok"] for s in steps), steps
    assert [n for n, _ in fake.calls] == ["spark-head", "spark-worker", "spark-head", "spark-head"]
    keygen, authorize, config, verify = [c for _, c in fake.calls]
    assert "ssh-keygen" in keygen and "id_lmds_cluster" in keygen and "-N ''" in keygen
    assert "authorized_keys" in authorize and PUB in authorize
    assert "~/.ssh/config" in config or "f=~/.ssh/config" in config
    assert verify == "ssh -o BatchMode=yes -o ConnectTimeout=8 nvidia@10.100.152.2 true", \
        "ต้องทดสอบด้วยคำสั่งแบบเดียวกับที่ controller ใช้ (ssh เปล่า ๆ ไม่มี -i)"
    # กุญแจส่วนตัวต้องไม่ผ่าน hub เลย — ไม่มีคำสั่งไหน cat ไฟล์ private
    assert not any("cat ~/.ssh/id_lmds_cluster " in c or c.endswith("id_lmds_cluster") for _, c in fake.calls)


def test_ssh_config_stanza_is_idempotent_and_keeps_the_users_own_lines(tmp_path, pair):
    """รันซ้ำกี่ครั้ง block ของ worker ตัวเดิมต้องมีอันเดียว และของเดิมในไฟล์ต้องอยู่ครบ"""
    from lmds.nodes.cluster_ssh import config_script

    if shutil.which("bash") is None or shutil.which("awk") is None:
        pytest.skip("ต้องมี bash + awk")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host github.com\n  User git\n", encoding="utf-8")
    _, worker = pair
    for _ in range(3):
        done = subprocess.run(["bash", "-c", config_script(worker, "10.100.152.2")],
                              env={**os.environ, "HOME": str(home)}, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    text = (home / ".ssh" / "config").read_text(encoding="utf-8")
    assert text.count("# lmds-cluster begin spark-worker") == 1
    assert text.startswith("Host github.com\n  User git\n"), "บรรทัดของผู้ใช้ต้องไม่ถูกแตะ"
    assert "Host 10.100.152.2 10.2.1.194" in text
    assert "IdentityFile ~/.ssh/id_lmds_cluster" in text
    assert "StrictHostKeyChecking accept-new" in text, "BatchMode ตอบคำถาม host key ครั้งแรกไม่ได้"
    assert "User nvidia" in text
    assert (home / ".ssh" / "config").stat().st_mode & 0o077 == 0


def test_pairing_reports_the_failing_step_instead_of_pretending(pair):
    from lmds.nodes.cluster_ssh import pair_workers

    head, worker = pair
    fake = FakeSSH(lambda node, cmd: (0, PUB + "\n", "") if "ssh-keygen" in cmd
                   else (255, "", "Permission denied (publickey)") if cmd.startswith("ssh ")
                   else (0, "", ""))
    steps = pair_workers(head, [(worker, worker.cluster_ip)], runner=fake)
    assert [s["ok"] for s in steps] == [True, True, True, False]
    assert "Permission denied" in steps[-1]["detail"]


# ── 6. หมอของคู่ — เหตุผล ไม่ใช่แค่ "ไม่ผ่าน" ────────────────────────────────
def diagnose(head="spark-head", worker="spark-worker", runner=None, slug=""):
    from lmds.nodes import load
    from lmds.nodes.doctor import diagnose_pair

    nodes = {n.name: n for n in load()}
    snap = state.STORE.snapshot()["nodes"]
    hosts = {n: ((e.get("data") or {}).get("host")) for n, e in snap.items()}
    errors = {n: e.get("error") or "" for n, e in snap.items()}
    return diagnose_pair(head, worker, nodes=nodes, hosts=hosts, errors=errors, runner=runner, slug=slug)


def failing(report) -> dict[str, dict]:
    return {f["kind"]: f for f in report["findings"] if not f["ok"]}


def test_doctor_explains_a_missing_cluster_ip_with_the_command_that_fixes_it():
    from lmds.nodes.doctor import describe

    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    register("spark-worker", "10.2.1.194", "", spark("10.100.152.2", "10.2.1.194"))
    report = diagnose()
    assert report["ok"] is False
    bad = failing(report)
    assert bad["cluster-ip"]["names"] == ["spark-worker"]
    assert bad["cluster-ip"]["fix"] == "lmds node set spark-worker --cluster-ip 10.100.152.2"
    assert "suggested 10.100.152.2" in describe(bad["cluster-ip"], "en")
    assert "เสนอ" not in describe(bad["cluster-ip"], "en") and "cluster IP" in describe(bad["cluster-ip"], "th")


def test_doctor_flags_split_subnet_wrong_site_and_a_down_link():
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    register("spark-worker", "10.2.1.194", "10.100.153.2",
             spark("10.100.153.2", "10.2.1.194", state_="down"), site="TKC")
    bad = failing(diagnose())
    assert set(bad) >= {"same-site", "same-subnet", "iface-up"}
    assert "Neronain vs TKC" in bad["same-site"]["data"]["sites"]
    assert "10.100.152.0/24" in bad["same-subnet"]["data"]["networks"]


def test_doctor_reports_unreachable_and_gpu_less_machines_before_anything_else():
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"))
    register("spark-worker", "10.2.1.194", "10.100.152.2", None)
    bad = failing(diagnose())
    assert list(bad) == ["reachable"] and "timed out" in bad["reachable"]["data"]["error"]
    assert list(failing(diagnose(worker="ghost"))) == ["registered"]


def test_doctor_checks_head_to_worker_ssh_and_points_at_pair(pair):
    fake = FakeSSH(lambda node, cmd: (255, "", "Permission denied (publickey)") if cmd.startswith("ssh ")
                   else (1, "", "") if cmd.startswith("ping") else (0, "", ""))
    bad = failing(diagnose(runner=fake))
    assert set(bad) == {"ssh-head-to-worker", "fabric-ping"}
    assert bad["ssh-head-to-worker"]["fix"].startswith("lmds cluster pair spark-head spark-worker")
    assert bad["fabric-ping"]["level"] == "warn"
    assert fake.on("spark-head")[0] == "ssh -o BatchMode=yes -o ConnectTimeout=8 nvidia@10.100.152.2 true"
    assert fake.on("spark-worker") == [], "หมอถามแต่ head — worker ไม่ต้องรับสาย"


def test_doctor_reads_cluster_env_on_the_head_and_compares_it_with_the_registry(pair):
    answers = {"env": "MASTER_IP=10.100.152.1\nWORKER_IP=10.100.152.9\nWORKER_IPS=\"10.100.152.9\"\n"}
    fake = FakeSSH(lambda node, cmd: (0, answers["env"], "") if "cluster.env" in cmd else (0, "", ""))
    bad = failing(diagnose(runner=fake, slug="big-model"))
    assert list(bad) == ["cluster-env-match"]
    assert "10.100.152.9" in bad["cluster-env-match"]["data"]["found"]
    assert bad["cluster-env-match"]["fix"] == "lmds cluster write big-model --head spark-head --worker spark-worker"

    answers["env"] = "NOENV"
    assert list(failing(diagnose(runner=fake, slug="big-model"))) == ["cluster-env"]
    answers["env"] = "NOBUNDLE"
    assert list(failing(diagnose(runner=fake, slug="big-model"))) == ["bundle-on-head"]


def test_doctor_passes_a_healthy_pair_and_warns_about_disk_and_negotiated_speed():
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195", disk=80))
    register("spark-worker", "10.2.1.194", "10.100.152.2", spark("10.100.152.2", "10.2.1.194", speed=50))
    good = "MASTER_IP=10.100.152.1\nWORKER_IP=10.100.152.2\nWORKER_IPS=\"10.100.152.2\"\nNNODES=2\n"
    report = diagnose(runner=FakeSSH(lambda node, cmd: (0, good if "cluster.env" in cmd else "", "")),
                      slug="big-model")
    assert report["ok"] is True, [f for f in report["findings"] if not f["ok"]]
    warns = {f["kind"]: f for f in report["findings"] if f["level"] == "warn"}
    assert set(warns) == {"disk", "link-speed"}
    assert warns["disk"]["names"] == ["spark-head"] and warns["link-speed"]["names"] == ["spark-worker"]


# ── API ──────────────────────────────────────────────────────────────────────
def client() -> TestClient:
    return TestClient(create_app())


def test_cluster_view_marks_a_gpu_less_hub_as_control_plane_not_a_candidate(pair):
    """hub จริง (VM) ขึ้นแถวแรกว่า ready:false 10G too slow — คนอ่านคิดว่าเป็นเครื่องที่ต้องแก้"""
    view = client().get("/api/cluster").json()
    hub = next(m for m in view["machines"] if m["self"])
    assert hub["candidate"] is False and hub["reason"] == "control-plane"
    assert all(m["candidate"] is True for m in view["machines"] if not m["self"])
    # กลุ่มที่พร้อมยังอยู่ครบ — hub ไม่ได้ทำให้ใครหายไป
    assert [tuple(m["name"] for m in g["members"]) for g in view["groups"]] == [("spark-head", "spark-worker")]


def test_cluster_view_suggests_the_configured_ip_when_it_is_already_fine():
    """spark-head จริงตั้ง 10.100.152.1 (ok) แต่ช่องเสนอ 10.100.153.1 — ดูเหมือนตั้งผิดทั้งที่ถูก"""
    host = spark("10.100.152.1", "10.2.1.195")
    host["fabric"]["links"].insert(0, {"iface": "enP2p1s0f1np1", "ip": "10.100.153.1", "prefix": 24,
                                       "speed_gbps": 200, "state": "up", "connectx": True, "rdma": True})
    register("spark-head", "10.2.1.195", "10.100.152.1", host)
    row = next(m for m in client().get("/api/cluster").json()["machines"] if m["name"] == "spark-head")
    assert row["ip"]["state"] == "ok" and row["suggested_ip"] == "10.100.152.1"


def four_nodes():
    for i in range(1, 5):
        register(f"n{i}", f"10.2.2.{i}", f"10.100.152.{i}", spark(f"10.100.152.{i}", f"10.2.2.{i}"), site="TKC")


def test_cluster_write_from_the_web_follows_the_bundle_node_count(tmp_path, monkeypatch):
    four_nodes()
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path / "bundles"])
    stacked_bundle(tmp_path, "two", 2)
    stacked_bundle(tmp_path, "four", 4)
    written = {}

    def answer(node, cmd):
        if "cluster.env" in cmd:
            import base64
            import re
            blob = re.search(r"echo (\S+) \| base64 -d", cmd).group(1)
            written[node.name] = base64.b64decode(blob).decode()
            return 0, f"/home/nvidia/bundles/x/cluster.env", ""
        return 0, "", ""
    monkeypatch.setattr("lmds.nodes.run", FakeSSH(answer))
    api = client()

    r = api.post("/api/cluster/write", json={"slug": "two", "head": "n1", "on": "n1"})
    assert r.status_code == 200, r.text
    assert r.json()["nnodes"] == 2 and r.json()["worker_ips"] == ["10.100.152.2"]
    assert "NNODES=2\n" in written["n1"]

    r = api.post("/api/cluster/write", json={"slug": "four", "head": "n1", "on": "n1"})
    assert r.status_code == 200, r.text
    assert r.json()["worker_ips"] == ["10.100.152.2", "10.100.152.3", "10.100.152.4"]
    assert 'WORKER_IPS="10.100.152.2 10.100.152.3 10.100.152.4"' in written["n1"]

    r = api.post("/api/cluster/write", json={"slug": "four", "head": "n1", "worker": "n2", "on": "n1"})
    assert r.status_code == 400 and "built for 4 machines" in r.json()["detail"]
    r = api.post("/api/cluster/write", json={"slug": "four", "head": "n1", "workers": ["n4", "n3", "n2"], "on": "n1"})
    assert r.json()["worker_ips"] == ["10.100.152.4", "10.100.152.3", "10.100.152.2"]


def test_cluster_pair_endpoint_runs_the_pairing_and_reports_each_step(pair, monkeypatch):
    fake = FakeSSH(lambda node, cmd: (0, PUB + "\n", "") if "ssh-keygen" in cmd else (0, "", ""))
    monkeypatch.setattr("lmds.nodes.run", fake)
    r = client().post("/api/cluster/pair", json={"head": "spark-head", "worker": "spark-worker"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and len(r.json()["steps"]) == 4
    assert r.json()["steps"][-1]["step"] == "spark-head → nvidia@10.100.152.2 without a password"

    assert client().post("/api/cluster/pair", json={"head": "spark-head"}).status_code == 400
    r = client().post("/api/cluster/pair", json={"head": "spark-head", "worker": "nobody"})
    assert r.status_code == 404


def test_cluster_doctor_endpoint_returns_english_findings_with_fixes(pair, monkeypatch):
    monkeypatch.setattr("lmds.nodes.run", FakeSSH(
        lambda node, cmd: (255, "", "Permission denied (publickey)") if cmd.startswith("ssh ") else (0, "", "")))
    r = client().get("/api/cluster/doctor", params={"head": "spark-head", "worker": "spark-worker"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    ssh = next(f for f in body["findings"] if f["kind"] == "ssh-head-to-worker")
    assert ssh["text"] == ("spark-head cannot ssh to nvidia@10.100.152.2 without a password: "
                           "Permission denied (publickey)")
    assert ssh["fix"].startswith("lmds cluster pair")
    assert all("text" in f for f in body["findings"])
    assert client().get("/api/cluster/doctor", params={"head": "spark-head"}).status_code == 400


def analyze_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_analyze(model, **kw):
        calls.append(kw)
        return {"id": "s1", "notes": [], "plan": {}}
    monkeypatch.setattr("lmds.web.deploy.analyze", fake_analyze)
    return calls


def test_analyze_refuses_a_stacked_target_without_a_valid_pair(pair, monkeypatch):
    """เดิม analyze รับ machine/worker อะไรก็ได้ → ไปตายที่ push/cluster.env ทีหลังโดยไม่มีเหตุผล"""
    register("msi-6", "192.168.10.147", "", spark("10.10.1.6", "192.168.10.147"), site="TKC")
    calls = analyze_calls(monkeypatch)
    api = client()
    base = {"model": "org/model", "target": "dgx-spark-stacked", "machine": "spark-head"}

    r = api.post("/api/deploy/analyze", json=base)
    assert r.status_code == 422 and r.json()["detail"]["kind"] == "cluster"
    assert "worker" in r.json()["detail"]["message"]
    r = api.post("/api/deploy/analyze", json={**base, "worker": "spark-head"})
    assert r.status_code == 422 and "both head and worker" in r.json()["detail"]["message"]
    r = api.post("/api/deploy/analyze", json={**base, "worker": "msi-6"})
    assert r.status_code == 422 and "msi-6" in r.json()["detail"]["message"]
    assert calls == [], "ต้องปฏิเสธก่อนไปดึง metadata จาก Hugging Face"

    r = api.post("/api/deploy/analyze", json={**base, "worker": "spark-worker"})
    assert r.status_code == 200, r.text
    assert calls[-1]["machine"] == "spark-head" and calls[-1]["worker"] == "spark-worker"


def test_analyze_for_a_single_target_ignores_a_stale_worker(pair, monkeypatch):
    """เลือก stacked แล้วเปลี่ยนใจเป็น single — draft.worker ค้าง → budget หักหน่วยความจำของ worker ด้วย"""
    calls = analyze_calls(monkeypatch)
    r = client().post("/api/deploy/analyze", json={"model": "org/model", "target": "dgx-spark-single",
                                                   "machine": "spark-head", "worker": "spark-worker"})
    assert r.status_code == 200, r.text
    assert calls[-1]["worker"] == ""


def stacked_model(slug="big-model", **extra) -> dict:
    return {"slug": slug, "engine": "vllm", "port": 8000, "running": False, "healthy": False,
            "downloaded": False, "topology": "stacked", "controller_exists": True,
            "commands": ["download", "sync-worker", "verify-worker", "start", "logs"],
            "cluster": {"master_ip": "10.100.152.1", "worker_ips": ["10.100.152.2"], "nnodes": 2},
            **extra}


def test_download_on_a_stacked_head_chains_sync_and_verify_worker(monkeypatch):
    """ปุ่ม download บนการ์ด head โหลดแค่ head — worker ไม่เคยได้ไฟล์ แล้ว start ตายที่ worker"""
    from lmds.web import jobs

    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"),
             models=[stacked_model()])
    started = []
    monkeypatch.setattr(jobs, "start_remote",
                        lambda node, slug, command, remote, secret_env=None, **kw:
                        started.append((command, remote)) or SimpleNamespace(payload=lambda: {"id": "j"}))
    monkeypatch.setattr("lmds.hardware.serving.guard", lambda *a, **k: "")
    r = client().post("/api/nodes/spark-head/models/big-model/repair", json={})
    assert r.status_code == 200, r.text
    command, remote = started[0]
    assert command == "download + sync-worker + verify-worker"
    assert "lmds repair big-model" in remote
    assert remote.index("sync-worker") < remote.index("verify-worker")
    assert "&&" in remote, "ขั้นถัดไปต้องรันต่อเมื่อขั้นก่อนสำเร็จเท่านั้น"


def test_logs_worker_is_a_controller_button_that_reaches_the_worker(monkeypatch):
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"),
             models=[stacked_model()])
    fake = FakeSSH(lambda node, cmd: (0, "===== worker 10.100.152.2 =====\nINFO ok", ""))
    monkeypatch.setattr("lmds.nodes.run", fake)
    r = client().post("/api/nodes/spark-head/models/big-model/ctl/logs-worker")
    assert r.status_code == 200, r.text
    assert "worker 10.100.152.2" in r.json()["output"]
    assert fake.calls[-1][1].endswith('"$ctl" logs worker 200')
    assert PAGE.read_text(encoding="utf-8").count('ctl("logs-worker"') == 1


def test_cancelling_a_stacked_start_says_the_workers_may_still_be_running(monkeypatch):
    from lmds.web import jobs

    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"),
             models=[stacked_model()])
    proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
    job = jobs.Job(id="j1", slug="big-model", node="spark-head", command="start", steps=["start"])
    job.process = proc
    jobs._JOBS[job.id] = job
    r = client().post("/api/jobs/j1/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True
    proc.wait(timeout=5)
    text = "".join(job.lines)
    assert "worker" in text and "stop" in text, text


# ── 3. inventory: โมเดล stacked ต้องขึ้นทั้งสองการ์ด พร้อมบทบาท ─────────────────
def test_the_node_agent_reads_cluster_env_next_to_the_controller(tmp_path):
    from lmds.inventory import read_cluster_env

    bundle = stacked_bundle(tmp_path, "big-model")
    assert read_cluster_env(str(bundle / "big-model-stacked.sh")) is None
    (bundle / "cluster.env").write_text(
        "# made by lmds\nMASTER_IP=10.100.152.1\nWORKER_IP=10.100.152.2\n"
        'WORKER_IPS="10.100.152.2 10.100.152.3"\nNNODES=3\nSSH_USER=nvidia\n', encoding="utf-8")
    assert read_cluster_env(str(bundle / "big-model-stacked.sh")) == {
        "master_ip": "10.100.152.1", "worker_ips": ["10.100.152.2", "10.100.152.3"],
        "nnodes": 3, "ssh_user": "nvidia"}


def test_snapshot_shows_the_stacked_model_on_the_worker_card_with_its_role():
    """เดิมการ์ด worker ว่างเปล่า (bundle อยู่ที่ head) — คนดูไม่รู้ว่าเครื่องนี้ถูกใช้อยู่และพอร์ตไหน"""
    register("spark-head", "10.2.1.195", "10.100.152.1", spark("10.100.152.1", "10.2.1.195"),
             models=[stacked_model(running=True)])
    register("spark-worker", "10.2.1.194", "10.100.152.2", spark("10.100.152.2", "10.2.1.194"))
    snap = state.STORE.snapshot()
    head_model = snap["nodes"]["spark-head"]["data"]["models"][0]
    assert head_model["stacked_role"] == "head" and head_model["stacked_peers"] == ["spark-worker"]
    worker_models = snap["nodes"]["spark-worker"]["data"]["models"]
    assert len(worker_models) == 1
    shadow = worker_models[0]
    assert (shadow["slug"], shadow["stacked_role"], shadow["stacked_head"], shadow["port"],
            shadow["running"]) == ("big-model", "worker", "spark-head", 8000, True)
    assert shadow["commands"] == [] and shadow["controller_exists"] is False
    # ทางที่หน้าเว็บอ่านการ์ด (แคช) ต้องได้เหมือน SSE
    api = client().get("/api/nodes/spark-worker/inventory").json()
    assert api["models"][0]["stacked_role"] == "worker"
    # ผลสำรวจรอบถัดไปของ worker ต้องไม่สะสมเงาซ้ำ — snapshot คำนวณใหม่ทุกครั้งจากของจริง
    snap = state.STORE.snapshot()
    assert len(snap["nodes"]["spark-worker"]["data"]["models"]) == 1


# ── 4. หน้าเว็บ (รันสคริปต์จริงของหน้าใน node) ─────────────────────────────────
def _node() -> str:
    found = shutil.which("node")
    if found:
        return found
    home = Path(os.environ.get("REAL_HOME") or Path.home())
    for cand in [*sorted((home / ".nvm/versions/node").glob("*/bin/node"), reverse=True),
                 Path("/opt/homebrew/bin/node"), Path("/usr/local/bin/node")]:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    pytest.skip("ไม่มี node บนเครื่องนี้")


def run_scenario(tmp_path: Path, prelude: str, body: str) -> list:
    script = tmp_path / "scenario.js"
    script.write_text(prelude + "\n// ---- boot ----\n" + body, encoding="utf-8")
    result = subprocess.run([_node(), str(HARNESS), str(PAGE), str(script)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"scenario failed:\n{result.stderr[-2500:]}\n--- stdout ---\n{result.stdout[-1500:]}"
    return [json.loads(line) for line in result.stdout.splitlines()
            if line.strip().startswith(("{", "["))]


FLEET = """const shadow = { slug: "big-model", stacked_role: "worker", stacked_head: "spark-head", port: 8000,
                 running: true, healthy: true, engine: "vllm", topology: "stacked", commands: [], controller_exists: false };
const headModel = { slug: "big-model", stacked_role: "head", stacked_peers: ["spark-worker"], port: 8000, running: true,
                    healthy: true, engine: "vllm", topology: "stacked", downloaded: true,
                    commands: ["sync-worker", "verify-worker", "logs-worker", "clear-fi-cache", "status"] };
const member = (name, ip) => ({ name, cluster_ip: ip, suggested_ip: ip, state: "ok", iface: "enp1s0f1np1", speed_gbps: 200, warning: null });
const fx = { nodes: [
  { name: "spark-head", site: "Neronain", models: [headModel] },
  { name: "spark-worker", site: "Neronain", models: [shadow] } ],
  cluster: {
    machines: [
      { name: "hub", self: true, reachable: true, ready: false, has_gpu: false, stack: true, candidate: false, reason: "control-plane",
        hostname: "hub", fabric: { summary: "eth0 10G", best_gbps: 10 }, cluster_ip: "", suggested_ip: "", ip: { state: "unset" } },
      { name: "spark-head", self: false, reachable: true, ready: true, has_gpu: true, stack: true, candidate: true, hostname: "spark-head",
        fabric: { summary: "200G", best_gbps: 200 }, cluster_ip: "10.100.152.1", suggested_ip: "10.100.152.1", ip: { state: "ok" } },
      { name: "spark-worker", self: false, reachable: true, ready: true, has_gpu: true, stack: true, candidate: true, hostname: "spark-worker",
        fabric: { summary: "200G", best_gbps: 200 }, cluster_ip: "10.100.152.2", suggested_ip: "10.100.152.2", ip: { state: "ok" } } ],
    groups: [ { site: "Neronain", cluster_name: "", members: [member("spark-head", "10.100.152.1"), member("spark-worker", "10.100.152.2")],
                excluded: [], gpu: "NVIDIA GB10", gpus_per_node: 1, link_gbps: 200, rdma: true, world_size: 2, usable_world_size: 2,
                fabric_network: "10.100.152.0/24", parallelism: { kind: "tensor-parallel", world_size: 2 }, blockers: [], warnings: [], ready: true } ] } };
H.fx = fx;
H.doctorCalls = [];
H.routes = [...H.defaultRoutes(fx),
  [/^\\/api\\/cluster\\/doctor/, url => { H.doctorCalls.push(url); return { head: "spark-head", worker: "spark-worker", ok: false, findings: [
      { kind: "cluster-ip", ok: true, level: "pass", names: ["spark-head"], text: "cluster IP on spark-head: 10.100.152.1 on enp1s0f1np1 200G" },
      { kind: "ssh-head-to-worker", ok: false, level: "fail", names: ["spark-head"], text: "spark-head cannot ssh to nvidia@10.100.152.2 without a password: Permission denied", fix: "lmds cluster pair spark-head spark-worker" } ] }; }],
  [/^\\/api\\/cluster\\/pair/, () => ({ ok: true, steps: [{ step: "cluster key on spark-head", ok: true, detail: "" }] })]];
"""


def test_the_hub_line_calls_a_gpu_less_hub_a_control_plane(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.go("#/nodes"); await H.tick(10);
        const line = document.querySelector("#nodes .hubstack");
        console.log(JSON.stringify({ text: line ? line.textContent.replace(/\\s+/g, " ").trim() : null,
                                     toggle: !!(line && line.querySelector("button")) }));""")
    assert out["text"] and "control plane" in out["text"] and "GPU" in out["text"], out
    assert out["toggle"] is False, "VM ที่ไม่มี GPU ไม่มีอะไรให้ Include/Exclude"


def test_the_worker_card_shows_the_stacked_model_as_a_worker_row_without_buttons(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.go("#/node/spark-worker"); await H.tick(10);
        const card = nodeRows.get("spark-worker").block;
        const row = card.querySelector('[data-stacked-role="worker"]');
        await H.go("#/node/spark-head"); await H.tick(10);
        const headCard = nodeRows.get("spark-head").block;
        headCard.querySelector('button[data-nact="menu"]').click(); await H.tick(10);
        console.log(JSON.stringify({
          row: row ? row.textContent.replace(/\\s+/g, " ").trim() : null,
          buttons: row ? row.querySelectorAll("button").length : -1,
          headTag: !!headCard.querySelector('[data-stacked-role="head"]'),
          headText: headCard.textContent.replace(/\\s+/g, " "),
          logsWorker: !!headCard.querySelector('button[data-nact="ctl:logs-worker"]') }));""")
    assert out["row"] and "big-model" in out["row"] and "worker" in out["row"] and "spark-head" in out["row"]
    assert "8000" in out["row"]
    assert out["buttons"] == 0, "แถวเงาบน worker ไม่มีปุ่ม — สั่งงานที่ head เท่านั้น"
    assert out["headTag"] is True and "spark-worker" in out["headText"]
    assert out["logsWorker"] is True


def test_the_group_header_has_a_doctor_button_that_renders_the_reasons(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.go("#/nodes"); await H.tick(10);
        const btn = document.querySelector('#nodes button[data-cact="doctor"]');
        H.assert(btn, "no doctor button in the group header");
        btn.click(); await H.tick(20);
        const panel = document.querySelector("#nodes .gdoctor");
        const pairBtn = document.querySelector('#nodes button[data-cact="pair-ssh"]');
        console.log(JSON.stringify({ calls: H.doctorCalls, text: panel ? panel.textContent.replace(/\\s+/g, " ").trim() : null,
                                     fails: panel ? panel.querySelectorAll(".warn-line").length : -1, pair: !!pairBtn,
                                     errors: H.errors, alerts: H.alerts }));""")
    assert out["calls"] and "head=spark-head" in out["calls"][0] and "worker=spark-worker" in out["calls"][0]
    assert out["text"] and "cannot ssh" in out["text"] and "lmds cluster pair" in out["text"]
    assert out["fails"] == 1 and out["pair"] is True
    assert out["errors"] == [] and out["alerts"] == []


def test_the_wizard_never_sends_a_worker_for_a_single_target():
    page = PAGE.read_text(encoding="utf-8")
    assert "worker: isStackedTarget(target) ? (draft.worker || \"\") : \"\"" in page


def test_the_stacked_push_flow_pairs_ssh_after_writing_cluster_env():
    page = PAGE.read_text(encoding="utf-8")
    flow = page[page.index("async function pushAfterBuild"):page.index("function wizBusy")]
    assert "writeClusterEnv(" in flow and "pairClusterSsh(" in flow
    assert flow.index("writeClusterEnv(") < flow.index("pairClusterSsh(")


# ── 5. CLI ───────────────────────────────────────────────────────────────────
def cli(*args, **kw):
    from typer.testing import CliRunner

    from lmds.cli.main import app

    return CliRunner().invoke(app, list(args), **kw)


def test_cli_cluster_doctor_prints_the_reasons_in_thai(pair, monkeypatch):
    head_host, worker_host = spark("10.100.152.1", "10.2.1.195"), spark("10.100.152.2", "10.2.1.194")
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": head_host if node.name == "spark-head" else worker_host})
    monkeypatch.setattr("lmds.nodes.run", FakeSSH(
        lambda node, cmd: (255, "", "Permission denied (publickey)") if cmd.startswith("ssh ") else (0, "", "")))
    result = cli("cluster", "doctor", "spark-head", "spark-worker")
    assert result.exit_code == 1, result.output
    assert "ssh ไป nvidia@10.100.152.2" in result.output
    assert "lmds cluster pair spark-head spark-worker" in result.output
    assert "✓" in result.output, "ข้อที่ผ่านต้องเห็นด้วย ไม่ใช่เห็นแต่ที่ล้ม"


def test_cli_cluster_pair_and_write_use_the_registry(pair, tmp_path, monkeypatch):
    fake = FakeSSH(lambda node, cmd: (0, PUB + "\n", "") if "ssh-keygen" in cmd
                   else (0, "/home/nvidia/bundles/big/cluster.env", "") if "cluster.env" in cmd else (0, "", ""))
    monkeypatch.setattr("lmds.nodes.run", fake)
    head_host, worker_host = spark("10.100.152.1", "10.2.1.195"), spark("10.100.152.2", "10.2.1.194")
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": head_host if node.name == "spark-head" else worker_host})
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path / "bundles"])
    stacked_bundle(tmp_path, "big", 2)

    result = cli("cluster", "pair", "spark-head", "spark-worker")
    assert result.exit_code == 0, result.output
    assert len(fake.on("spark-head")) == 3 and len(fake.on("spark-worker")) == 1

    result = cli("cluster", "write", "big", "--head", "spark-head", "--worker", "spark-worker", "--on", "spark-head")
    assert result.exit_code == 0, result.output
    assert "10.100.152.2" in result.output
    assert any("cluster.env" in c for c in fake.on("spark-head"))


def test_node_push_of_a_stacked_bundle_writes_cluster_env_pairs_and_syncs(pair, tmp_path, monkeypatch):
    """`lmds node push head slug --download --start` เคยส่ง zip แล้วโหลดที่ head อย่างเดียว —
    ไม่มี cluster.env (start ถาม IP ที่ไม่มีใครตอบ) ไม่มีกุญแจ ไม่มี sync-worker"""
    import lmds.cli.main as main

    from lmds.packager.bundle import make_zip

    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path / "bundles"])
    make_zip(stacked_bundle(tmp_path, "big", 2))
    head_host, worker_host = spark("10.100.152.1", "10.2.1.195"), spark("10.100.152.2", "10.2.1.194")
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": head_host if node.name == "spark-head" else worker_host})
    fake = FakeSSH(lambda node, cmd: (0, PUB + "\n", "") if "ssh-keygen" in cmd
                   else (0, "/home/nvidia/bundles/big/cluster.env", "") if "cluster.env" in cmd
                   else (0, "/home/nvidia/bundles/big", ""))
    monkeypatch.setattr("lmds.nodes.run", fake)
    monkeypatch.setattr("lmds.nodes.push_file", lambda node, local, remote, timeout=0: SimpleNamespace(ok=True, stderr=""))
    detached = []
    monkeypatch.setattr(main, "_run_detached", lambda node, command, log_name, **kw: detached.append(command) or 0)

    result = cli("node", "push", "spark-head", "big", "--download", "--start")
    assert result.exit_code == 0, result.output
    assert any("cluster.env" in c for c in fake.on("spark-head")), "cluster.env ต้องถูกเขียนก่อน start"
    assert any("authorized_keys" in c for c in fake.on("spark-worker")), "worker ต้องได้กุญแจของ head"
    assert len(detached) == 2
    assert "lmds repair big" in detached[0] and "sync-worker" in detached[0] and "verify-worker" in detached[0]
    assert "lmds start big" in detached[1]
