"""Multi-link ConnectX-7 บน DGX Spark — 3 เครื่องต่อสายตรงเป็นวงแหวน · 2–4 เครื่องผ่าน switch

ทุกเครื่องมี QSFP สองพอร์ต (port 1 = enp1s0f0np0/enp1s0f1np1 · port 2 = enP2p1s0f0np0/enP2p1s0f1np1 ·
RoCE rocep1s0f0 / roceP2p1s0f0) · วงแหวน A.p1→B.p2 · B.p1→C.p2 · C.p1→A.p2 = ทุกเครื่องมี 2 สาย 2 วง
และ head ถึง worker แต่ละตัวด้วยคนละ interface/IP · cluster.env เดิมบอกได้แค่ IP เดียวต่อเครื่อง

เทสฝั่ง Python คุม schema v2 ของ cluster.env (direct-2 ต้องเท่าเดิมทุกตัวอักษร) · เทสฝั่ง bash รัน controller
ที่ render แล้วจริงใต้ `set -Eeuo pipefail` กับ ssh/docker/ip/ping ปลอมที่ตอบ **ต่อ node** (ssh ปลอมรันคำสั่ง
ปลายทางในเครื่องนี้ · ip/ping/sysfs ของแต่ละเครื่องแยกกัน) แล้วดูว่า worker แต่ละ rank ได้ env ของตัวเอง
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.targets import TargetSpec
from lmds.fleet.cluster_env import (
    build_cluster_env,
    parse_cluster_env,
    render_cluster_env,
    topology_from_members,
    worker_targets,
)
from lmds.generator import render_bundle
from lmds.hardware import MemoryModel
from tests.test_audit_stacked_controller import _DF, _DOCKER, _report, _seed_head_cache, _shim

SAFE_PATH = "/usr/bin:/bin"

# ── วงแหวน 3 เครื่อง: a(head) · b(rank 1) · c(rank 2) ──
#   สาย ab: a.port1 10.0.1.1 ↔ b.port2 10.0.1.2      สาย bc: b.port1 10.0.2.1 ↔ c.port2 10.0.2.2
#   สาย ca: c.port1 10.0.3.1 ↔ a.port2 10.0.3.2
P1, P2 = "enp1s0f0np0", "enP2p1s0f0np0"
H1, H2 = "rocep1s0f0", "roceP2p1s0f0"


def _link(iface, ip, peer, peer_ip, link_id):
    return {"iface": iface, "ip": ip, "prefix": 24, "peer_node": peer, "peer_ip": peer_ip, "link_id": link_id}


RING = [
    {"name": "a", "cluster_ip": "10.0.1.1",
     "cluster_links": [_link(P1, "10.0.1.1", "b", "10.0.1.2", "ab"), _link(P2, "10.0.3.2", "c", "10.0.3.1", "ca")]},
    {"name": "b", "cluster_ip": "10.0.1.2",
     "cluster_links": [_link(P2, "10.0.1.2", "a", "10.0.1.1", "ab"), _link(P1, "10.0.2.1", "c", "10.0.2.2", "bc")]},
    {"name": "c", "cluster_ip": "10.0.3.1",
     "cluster_links": [_link(P2, "10.0.2.2", "b", "10.0.2.1", "bc"), _link(P1, "10.0.3.1", "a", "10.0.3.2", "ca")]},
]
# ── switch 4 เครื่อง: ทุกเครื่องสายเดียว วงเดียว 10.100.152.0/24 ──
SWITCH4 = [
    {"name": f"s{i}", "cluster_ip": f"10.100.152.{i}",
     "cluster_links": [{"iface": P1, "ip": f"10.100.152.{i}", "prefix": 24, "peer_node": "", "peer_ip": "", "link_id": "sw"}]}
    for i in range(1, 5)
]
# ── 2 เครื่องแบบเดิม (ทะเบียนก่อน 0.6.1 ไม่มี cluster_links) ──
LEGACY2 = [{"name": "head", "cluster_ip": "10.100.152.1", "iface": "enp1s0f1np1"},
           {"name": "worker", "cluster_ip": "10.100.152.2"}]


# ═════════════════════ 1. cluster.env schema v2 ═════════════════════
def test_direct_2_from_a_legacy_registry_renders_exactly_the_0_6_0_file():
    """ไฟล์ของคู่ที่รันอยู่จริง (spark-head/worker) ต้องไม่เปลี่ยนแม้แต่ตัวอักษรเดียว"""
    body = build_cluster_env([{"ready": True, "members": LEGACY2}], "head", "worker")["body"]
    assert body == (
        "# สร้างโดย lmds (node cluster --write / หน้าเว็บ) — แก้มือได้ ค่า env ภายนอกยังชนะไฟล์นี้\n"
        "MASTER_IP=10.100.152.1\nWORKER_IP=10.100.152.2\nWORKER_IPS=\"10.100.152.2\"\nNNODES=2\n"
        "TENSOR_PARALLEL_SIZE=2\nSSH_USER=\nTRANSPORT_IP_MASTER=10.100.152.1\nTRANSPORT_IP_WORKER=10.100.152.2\n"
        "NCCL_SOCKET_IFNAME=enp1s0f1np1\n"
    )
    assert "CLUSTER_TOPOLOGY" not in body and "LINKS_0" not in body
    assert topology_from_members(LEGACY2)["kind"] == "direct-2"


def test_ring_3_writes_per_rank_links_and_the_ip_each_side_uses():
    topology = topology_from_members(RING)
    assert topology["kind"] == "ring-3"
    body = render_cluster_env(topology, ssh_user="neronain")
    values = parse_cluster_env_text(body)
    # คีย์เดิมยังครบ และชี้สายตรง head→worker แต่ละตัว (คนละวง)
    assert values["MASTER_IP"] == "10.0.1.1" and values["WORKER_IP"] == "10.0.1.2"
    assert values["WORKER_IPS"] == "10.0.1.2 10.0.3.1" and values["NNODES"] == "3" == values["TENSOR_PARALLEL_SIZE"]
    assert values["TRANSPORT_IP_MASTER"] == "10.0.1.1" and values["TRANSPORT_IPS_WORKER"] == "10.0.1.2 10.0.3.1"
    assert values["NCCL_SOCKET_IFNAME"] == f"{P1},{P2}"
    # v2
    assert values["CLUSTER_ENV_SCHEMA"] == "2" and values["CLUSTER_TOPOLOGY"] == "ring-3"
    assert values["CLUSTER_NODES"] == "a b c"
    assert values["LINKS_0"] == f"{P1}:10.0.1.1/24:1:10.0.1.2 {P2}:10.0.3.2/24:2:10.0.3.1"
    assert values["LINKS_1"] == f"{P2}:10.0.1.2/24:0:10.0.1.1 {P1}:10.0.2.1/24:2:10.0.2.2"
    assert values["LINKS_2"] == f"{P2}:10.0.2.2/24:1:10.0.2.1 {P1}:10.0.3.1/24:0:10.0.3.2"
    assert values["NCCL_SOCKET_IFNAMES_0"] == f"{P1},{P2}"
    assert values["NCCL_SOCKET_IFNAMES_1"] == f"{P2},{P1}" == values["NCCL_SOCKET_IFNAMES_2"]
    assert values["NCCL_IB_HCAS_1"] == ""            # ทะเบียนไม่รู้ HCA → controller เดิน sysfs บน worker เอง
    assert values["HEAD_TO_WORKER_IP_1"] == "10.0.1.2" and values["WORKER_HEAD_IP_1"] == "10.0.1.1"
    assert values["HEAD_TO_WORKER_IP_2"] == "10.0.3.1" and values["WORKER_HEAD_IP_2"] == "10.0.3.2"
    assert values["NCCL_CROSS_NIC"] == "1"
    assert worker_targets(values) == [{"rank": 1, "ip": "10.0.1.2", "ssh_user": "neronain"},
                                      {"rank": 2, "ip": "10.0.3.1", "ssh_user": "neronain"}]


def test_switch_4_is_one_subnet_one_interface_and_no_cross_nic():
    topology = topology_from_members(SWITCH4)
    assert topology["kind"] == "switch-4"
    values = parse_cluster_env_text(render_cluster_env(topology))
    assert values["WORKER_IPS"] == "10.100.152.2 10.100.152.3 10.100.152.4"
    assert values["LINKS_0"] == f"{P1}:10.100.152.1/24:*:-"
    for rank in (1, 2, 3):
        assert values[f"HEAD_TO_WORKER_IP_{rank}"] == f"10.100.152.{rank + 1}"
        assert values[f"WORKER_HEAD_IP_{rank}"] == "10.100.152.1"
        assert values[f"NCCL_SOCKET_IFNAMES_{rank}"] == P1
    assert "NCCL_CROSS_NIC" not in values
    # ทะเบียนเก่า 4 เครื่อง (ไม่มี cluster_links) ก็ได้คีย์ v2 — IP เดิม สายเดียว
    legacy = [{"name": f"s{i}", "cluster_ip": f"10.100.152.{i}", "iface": P1} for i in range(1, 5)]
    old = parse_cluster_env_text(render_cluster_env(topology_from_members(legacy)))
    assert old["CLUSTER_TOPOLOGY"] == "switch-4" and old["HEAD_TO_WORKER_IP_3"] == "10.100.152.4"


def test_hcas_from_the_registry_are_written_when_known():
    ring = [dict(m, cluster_links=[dict(l, hca=(H1 if l["iface"] == P1 else H2)) for l in m["cluster_links"]])
            for m in RING]
    values = parse_cluster_env_text(render_cluster_env(topology_from_members(ring)))
    assert values["NCCL_IB_HCAS_0"] == f"{H1},{H2}" and values["NCCL_IB_HCAS_1"] == f"{H2},{H1}"


def parse_cluster_env_text(body: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8") as handle:
        handle.write(body)
    try:
        return parse_cluster_env(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def test_parse_cluster_env_reads_a_string_like_object_and_files(tmp_path):
    path = tmp_path / "cluster.env"
    path.write_text('MASTER_IP=1.1.1.1\nWORKER_IPS="2.2.2.2 3.3.3.3"\n# comment\nSSH_USER=me\n', encoding="utf-8")
    values = parse_cluster_env(path)
    assert values == {"MASTER_IP": "1.1.1.1", "WORKER_IPS": "2.2.2.2 3.3.3.3", "SSH_USER": "me"}
    assert worker_targets(values) == [{"rank": 1, "ip": "2.2.2.2", "ssh_user": "me"},
                                      {"rank": 2, "ip": "3.3.3.3", "ssh_user": "me"}]
    assert parse_cluster_env(tmp_path / "missing") == {}


# ═════════════════════ 2. bundle ของ 3 เครื่อง ═════════════════════
def _spec(nodes: int) -> TargetSpec:
    name = {2: "dgx-spark-stacked", 4: "dgx-spark-stacked-4"}.get(nodes, f"dgx-spark-stacked-{nodes}")
    return PRESETS.get(name) or TargetSpec(name, MemoryModel.UNIFIED, 128.0, nodes, system_ram_gb=None, tested=False)


def _bundle(tmp_path, nodes: int):
    report = _report()
    fit = analyze(report, _spec(nodes))
    plan = build_plan(report, fit, provider=None)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def test_a_three_node_bundle_renders_tp_3_and_passes_bash_n(tmp_path):
    """ยังไม่มี preset dgx-spark-stacked-3 — ชื่อ target แบบนั้นต้องได้ 3 เครื่อง ไม่ใช่ถอยไป 2 เงียบ ๆ"""
    bundle = _bundle(tmp_path, 3)
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'NNODES="${NNODES:-3}"' in text and 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-3}"' in text
    for nodes in (2, 3, 4):
        done = subprocess.run(["bash", "-n", str(_bundle(tmp_path / str(nodes), nodes).controller)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


# ═════════════════════ 3. controller ที่ render แล้ว กับ node ปลอมทีละเครื่อง ═════════════════════
# ssh ปลอม: จำ host · รันคำสั่งปลายทางในเครื่องนี้ในนาม FAKE_NODE=<ip> · /tmp/lmds-<slug> และ
# /sys/class/infiniband ของ worker ถูกเบี่ยงไปโฟลเดอร์ต่อ node (FAKE_REMOTE/<ip>/…)
_SSH = '''
host=""; cmd=""
while (( $# )); do
  case "$1" in
    -o) shift 2 ;;
    -*) shift ;;
    *) if [[ -z "$host" ]]; then host="$1"; shift; else cmd="$*"; break; fi ;;
  esac
done
node="${host#*@}"
echo "ssh ${host} :: ${cmd%%$'\\n'*}" >> "$FAKE_LOG"
for down in ${FAKE_SSH_DOWN:-}; do [[ "$node" == "$down" ]] && { echo "ssh: connect to host $node port 22: No route to host" >&2; exit 255; }; done
cmd="${cmd//\\/tmp\\/lmds-/${FAKE_REMOTE}/${node}/tmp/lmds-}"
cmd="${cmd//\\/sys\\/class\\/infiniband/${FAKE_REMOTE}/${node}/infiniband}"
cmd="${cmd//\\/sys\\/class\\/net/${FAKE_REMOTE}/${node}/net}"
export FAKE_NODE="$node"
unset IB_SYSFS_ROOT NET_SYSFS_ROOT          # ssh จริงไม่พา env ของ head ไปด้วย
exec bash -c "$cmd"
'''

# ip ปลอม: ที่อยู่ของแต่ละเครื่องอ่านจาก FAKE_REMOTE/<node>/ip.txt (บรรทัดรูป `ip -o -4 addr show`)
_IP = '''
node="${FAKE_NODE:-head}"
table="${FAKE_REMOTE}/${node}/ip.txt"
[[ -f "$table" ]] || exit 0
dev=""
while (( $# )); do case "$1" in dev) dev="$2"; shift 2 ;; *) shift ;; esac; done
if [[ -n "$dev" ]]; then awk -v d="$dev" '$2 == d' "$table"; else cat "$table"; fi
exit 0
'''

_PING = '''
node="${FAKE_NODE:-head}"
echo "ping[${node}] $*" >> "$FAKE_LOG"
for bad in ${FAKE_PING_FAIL:-}; do [[ "${@: -1}" == "$bad" ]] && exit 1; done
exit 0
'''


def _addr_lines(entries):
    return "".join(f"{i + 2}: {iface}  inet {ip}/24 brd 0.0.0.0 scope global {iface}\n"
                   for i, (iface, ip) in enumerate(entries))


def _node(remote: Path, name: str, addrs, hcas):
    """เครื่องปลอมหนึ่งเครื่อง: ที่อยู่ + sysfs ของ RoCE (hca → iface)"""
    d = remote / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "ip.txt").write_text(_addr_lines(addrs), encoding="utf-8")
    for hca, iface in hcas:
        (d / "infiniband" / hca / "device" / "net" / iface).mkdir(parents=True, exist_ok=True)
    for iface, _ip in addrs:
        (d / "net" / iface).mkdir(parents=True, exist_ok=True)
        (d / "net" / iface / "operstate").write_text("up\n", encoding="utf-8")
        (d / "net" / iface / "speed").write_text("200000\n", encoding="utf-8")


def _ring_fixture(tmp_path: Path, bundle) -> dict:
    remote = tmp_path / "remote"
    _node(remote, "head", [(P1, "10.0.1.1"), (P2, "10.0.3.2")], [(H1, P1), (H2, P2)])
    _node(remote, "10.0.1.2", [(P2, "10.0.1.2"), (P1, "10.0.2.1")], [(H2, P2), (H1, P1)])
    _node(remote, "10.0.3.1", [(P2, "10.0.2.2"), (P1, "10.0.3.1")], [(H2, P2), (H1, P1)])
    (bundle.directory / "cluster.env").write_text(render_cluster_env(topology_from_members(RING), "neronain"),
                                                  encoding="utf-8")
    return _env(tmp_path, bundle)


def _switch_fixture(tmp_path: Path, bundle) -> dict:
    remote = tmp_path / "remote"
    _node(remote, "head", [(P1, "10.100.152.1")], [(H1, P1)])
    for i in (2, 3, 4):
        _node(remote, f"10.100.152.{i}", [(P1, f"10.100.152.{i}")], [(H1, P1)])
    (bundle.directory / "cluster.env").write_text(render_cluster_env(topology_from_members(SWITCH4), "neronain"),
                                                  encoding="utf-8")
    return _env(tmp_path, bundle)


def _env(tmp_path: Path, bundle) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, "ssh", _SSH)
    _shim(bin_dir, "docker", _DOCKER)
    _shim(bin_dir, "ip", _IP)
    _shim(bin_dir, "df", _DF)
    _shim(bin_dir, "ping", _PING)
    _shim(bin_dir, "curl", "exit 0\n")
    _shim(bin_dir, "sudo", "exit 0\n")
    _shim(bin_dir, "rsync", 'echo "rsync $*" >> "$FAKE_LOG"; exit 0\n')
    (tmp_path / "home").mkdir(exist_ok=True)
    return {
        "PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(tmp_path / "home"),
        "FAKE_LOG": str(tmp_path / "calls.log"), "FAKE_REMOTE": str(tmp_path / "remote"),
        "IB_SYSFS_ROOT": str(tmp_path / "remote" / "head" / "infiniband"),
        "NET_SYSFS_ROOT": str(tmp_path / "remote" / "head" / "net"),
        "WORKER_INIT_WAIT": "0", "WORKER_CHECK_INTERVAL": "0", "RUN_DIR": str(tmp_path / "run"),
    }


def _run(bundle, cmd, env, extra=None, timeout=90):
    return subprocess.run(["bash", str(bundle.controller), *cmd], env={**env, **(extra or {})},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def _calls(env) -> str:
    log = Path(env["FAKE_LOG"])
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _worker_sh(env, ip, slug) -> str:
    return (Path(env["FAKE_REMOTE"]) / ip / "tmp" / f"lmds-{slug}" / "worker.sh").read_text(encoding="utf-8")


def test_serve_args_of_a_ring_shows_tp_3_and_two_workers_with_their_own_env(tmp_path):
    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    done = _run(bundle, ["serve-args"], env)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert out.count("--tensor-parallel-size\n3\n") == 3      # head + 2 worker
    assert f"# worker rank 1 @ 10.0.1.2 — VLLM_HOST_IP=10.0.1.2 NCCL_SOCKET_IFNAME={P2},{P1}" in out
    assert f"# worker rank 2 @ 10.0.3.1 — VLLM_HOST_IP=10.0.3.1 NCCL_SOCKET_IFNAME={P2},{P1}" in out
    assert "NCCL_CROSS_NIC=1" in out
    # worker แต่ละตัวต่อ head ที่ IP ของสายตัวเอง · head ประกาศ IP ฝั่งสาย rank 1
    head, w1, w2 = out.split("# worker rank ")[0], out.split("# worker rank 1")[1].split("# worker rank 2")[0], out.split("# worker rank 2")[1]
    assert "--master-addr\n10.0.1.1\n" in head and "--node-rank\n0\n" in head
    assert "--master-addr\n10.0.1.1\n" in w1 and "--node-rank\n1\n" in w1 and "--headless" in w1
    assert "--master-addr\n10.0.3.2\n" in w2 and "--node-rank\n2\n" in w2
    assert "ssh" not in _calls(env) and "docker" not in _calls(env)


def test_start_of_a_ring_gives_every_rank_its_own_interfaces_hcas_and_ips(tmp_path):
    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(env)
    slug = bundle.directory.name

    head_run = next(l for l in calls.splitlines() if l.startswith("docker[head] run -d"))
    assert "-e VLLM_HOST_IP=10.0.1.1" in head_run
    assert f"-e NCCL_SOCKET_IFNAME={P1},{P2}" in head_run and f"-e GLOO_SOCKET_IFNAME={P1},{P2}" in head_run
    assert f"-e NCCL_IB_HCA={H1},{H2}" in head_run and "-e NCCL_IB_DISABLE=0" in head_run
    assert "-e NCCL_CROSS_NIC=1" in head_run
    assert "--master-addr 10.0.1.1" in head_run

    w1 = _worker_sh(env, "10.0.1.2", slug)
    assert "export VLLM_HOST_IP=10.0.1.2" in w1
    assert f"export NCCL_SOCKET_IFNAME={P2},{P1}" in w1 and f"export NCCL_IB_HCA={H2},{H1}" in w1
    assert "export NCCL_CROSS_NIC=1" in w1 and "--master-addr 10.0.1.1" in w1 and "--node-rank 1" in w1
    w2 = _worker_sh(env, "10.0.3.1", slug)
    assert "export VLLM_HOST_IP=10.0.3.1" in w2
    assert f"export NCCL_SOCKET_IFNAME={P2},{P1}" in w2 and f"export NCCL_IB_HCA={H2},{H1}" in w2
    assert "--master-addr 10.0.3.2" in w2 and "--node-rank 2" in w2
    assert "--tensor-parallel-size 3" in w1 and "--nnodes 3" in w2

    # head คุยกับ worker แต่ละตัวทาง IP ของสายตรง (rank 2 ไม่ใช่ 10.0.1.x) ทั้ง ssh และ container
    assert "ssh neronain@10.0.1.2 ::" in calls and "ssh neronain@10.0.3.1 ::" in calls
    assert "docker[10.0.1.2] run -d" in calls and "docker[10.0.3.1] run -d" in calls
    assert "ssh neronain@10.0.2.1" not in calls and "ssh neronain@10.0.2.2" not in calls
    # network-info ท้าย start: ping คู่ปลายสายทางสายนั้น ๆ ทั้งจาก head และจาก worker
    assert f"ping[head] -c 1 -W 1 -I {P1} 10.0.1.2" in calls and f"ping[head] -c 1 -W 1 -I {P2} 10.0.3.1" in calls
    assert f"ping[10.0.3.1] -c 1 -W 1 -I {P1} 10.0.3.2" in calls and f"ping[10.0.3.1] -c 1 -W 1 -I {P2} 10.0.2.1" in calls
    assert "ping=ok" in done.stdout and "Topology   : ring-3 (NCCL_CROSS_NIC=1)" in done.stdout


def test_start_of_a_switch_4_gives_every_worker_the_same_interface_and_no_cross_nic(tmp_path):
    bundle = _bundle(tmp_path, 4)
    env = _switch_fixture(tmp_path, bundle)
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(env)
    slug = bundle.directory.name
    head_run = next(l for l in calls.splitlines() if l.startswith("docker[head] run -d"))
    assert "-e VLLM_HOST_IP=10.100.152.1" in head_run and f"-e NCCL_SOCKET_IFNAME={P1}" in head_run
    assert "NCCL_CROSS_NIC" not in head_run and f"-e NCCL_IB_HCA={H1}" in head_run
    for rank, ip in ((1, "10.100.152.2"), (2, "10.100.152.3"), (3, "10.100.152.4")):
        w = _worker_sh(env, ip, slug)
        assert f"export VLLM_HOST_IP={ip}" in w and f"export NCCL_SOCKET_IFNAME={P1}" in w
        assert f"export NCCL_IB_HCA={H1}" in w and "NCCL_CROSS_NIC" not in w
        assert "--master-addr 10.100.152.1" in w and f"--node-rank {rank}" in w and "--tensor-parallel-size 4" in w
        assert f"docker[{ip}] run -d" in calls


def test_dry_run_prints_every_rank_without_touching_docker_or_ssh(tmp_path):
    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    done = _run(bundle, ["start", "--dry-run"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert "DRY RUN: ring-3" in out and "NNODES=3 TP=3" in out
    assert "== worker rank 1 @ 10.0.1.2" in out and "== worker rank 2 @ 10.0.3.1" in out and "== head rank 0" in out
    assert "export VLLM_HOST_IP=10.0.3.1" in out and f"export NCCL_SOCKET_IFNAME={P2},{P1}" in out
    assert "--master-addr 10.0.3.2" in out and "--master-addr 10.0.1.1" in out
    assert "--tensor-parallel-size 3" in out
    assert "ssh" not in _calls(env) and "docker[" not in _calls(env)
    # ไม่มี cluster.env เลยก็ยังประกอบให้ดูได้ (ค่าตัวอย่าง) — ไม่ถูก _require_cluster_config ปฏิเสธ
    bare = _run(bundle, ["start", "--dry-run"], env, extra={"CLUSTER_ENV": "/nonexistent"})
    assert bare.returncode == 0, bare.stderr
    assert "10.100.152.2" in bare.stdout and "ssh" not in _calls(env)


def test_network_info_and_doctor_report_every_link_and_fail_on_a_dead_one(tmp_path):
    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    info = _run(bundle, ["network-info"], env)
    assert info.returncode == 0, info.stderr
    assert f"NCCL if    : {P1},{P2}" in info.stdout and f"NCCL HCA   : {H1},{H2}" in info.stdout
    assert "rank 1 (10.0.1.2)" in info.stdout and "rank 2 (10.0.3.1)" in info.stdout and "head-ip=10.0.3.2" in info.stdout
    assert info.stdout.count("ping=ok") == 6                      # 2 สาย × 3 เครื่อง
    assert info.stdout.count("link=up speed=200000 ip=yes") == 6
    assert f"hca={H2}" in info.stdout and f"hca={H1}" in info.stdout

    status = _run(bundle, ["status"], env)
    assert status.returncode == 0 and "--- Links ---" in status.stdout and "ping=ok" in status.stdout

    ok = _run(bundle, ["doctor"], env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "ทุกสายขึ้น" in ok.stdout
    # สาย c→a ขาด: doctor ต้องล้มพร้อมชี้บรรทัด ไม่ใช่ผ่านแล้วไปค้างที่ NCCL init
    dead = _run(bundle, ["doctor"], env, extra={"FAKE_PING_FAIL": "10.0.3.2"})
    assert dead.returncode != 0
    assert "ping=FAIL" in dead.stdout and "สายบางเส้นไม่พร้อม" in dead.stderr and "RUNBOOK-MULTI-NODE" in dead.stderr
    # สายหลุด (operstate down) บน worker ก็ต้องล้ม แม้ ping ปลอมจะผ่าน
    (tmp_path / "remote" / "10.0.1.2" / "net" / P1 / "operstate").write_text("down\n", encoding="utf-8")
    down = _run(bundle, ["doctor"], env)
    assert down.returncode != 0 and "link=down" in down.stdout


def test_sync_worker_and_ssh_checks_use_the_direct_link_of_each_rank(tmp_path):
    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["sync-worker"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    rsyncs = [l for l in _calls(env).splitlines() if l.startswith("rsync ")]
    assert len(rsyncs) == 2
    assert "neronain@10.0.1.2:" in rsyncs[0] and "neronain@10.0.3.1:" in rsyncs[1]
    # head ที่ไม่ได้ถือ IP ของสายใดสายหนึ่งใน LINKS_0 = สายหลุด/รันผิดเครื่อง — start ต้องหยุดก่อนแตะ worker
    (tmp_path / "remote" / "head" / "ip.txt").write_text(_addr_lines([(P1, "10.0.1.1")]), encoding="utf-8")
    wrong = _run(bundle, ["start"], env)
    assert wrong.returncode != 0 and "10.0.3.2" in wrong.stderr and "LINKS_0" in wrong.stderr
    assert "docker[10.0.1.2] run" not in _calls(env)


def test_a_legacy_two_node_cluster_env_keeps_the_single_link_behaviour(tmp_path):
    """คู่ที่รันอยู่จริงต้องไม่เห็นความต่าง: ไม่มีคีย์ v2 = serve-args/start เหมือน 0.6.0"""
    bundle = _bundle(tmp_path, 2)
    env = _env(tmp_path, bundle)
    _node(tmp_path / "remote", "head", [("mgmt0", "10.1.1.1"), ("fabric0", "10.200.0.1")], [("mlx5_0", "fabric0")])
    _node(tmp_path / "remote", "10.1.1.2", [("mgmt1", "10.1.1.2"), ("fabric1", "10.200.0.2")], [("mlx5_1", "fabric1")])
    (bundle.directory / "cluster.env").write_text(
        "MASTER_IP=10.1.1.1\nWORKER_IP=10.1.1.2\nSSH_USER=neronain\n"
        "TRANSPORT_IP_MASTER=10.200.0.1\nTRANSPORT_IP_WORKER=10.200.0.2\n", encoding="utf-8")
    args = _run(bundle, ["serve-args"], env)
    assert args.returncode == 0 and "# worker rank 1..1 (headless" in args.stdout and "# head (rank 0):" in args.stdout
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    w = _worker_sh(env, "10.1.1.2", bundle.directory.name)
    assert "export VLLM_HOST_IP=10.200.0.2" in w and "export NCCL_SOCKET_IFNAME=fabric1" in w
    assert "export NCCL_IB_HCA=mlx5_1" in w and "NCCL_CROSS_NIC" not in w and "--master-addr 10.200.0.1" in w
    head_run = next(l for l in _calls(env).splitlines() if l.startswith("docker[head] run -d"))
    assert "-e NCCL_SOCKET_IFNAME=fabric0" in head_run and "-e NCCL_IB_HCA=mlx5_0" in head_run
    assert "Topology   :" not in done.stdout and "ping[" not in _calls(env)
