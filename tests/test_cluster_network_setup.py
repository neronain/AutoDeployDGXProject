"""ตั้งค่าสาย ConnectX ระหว่าง DGX Spark จาก hub — inspect → plan → apply → doctor (2026-09-05)

ลูกค้าได้ Spark มาใหม่ เสียบสายแล้วยังต้องตั้ง IP เองทีละเครื่องก่อน stacked จะเริ่มได้ · ชุดนี้คุม:
การอ่าน /sys ปลอมของทั้งสองพอร์ต · การเดา topology · การแจก IP ที่ทำซ้ำได้และเก็บของเดิม · YAML
ของ netplan · ลำดับคำสั่งตอน apply กับ SSH/sudo/netplan ปลอม (rollback · รหัสไม่โผล่ใน argv/log) ·
บรรทัดของหมอ · รูป JSON ของ API — ไม่แตะเครื่องจริงเลย
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmds.hardware.profiler import detect_fabric, group_qsfp_ports
from lmds.nodes import Node, add, find
from lmds.nodes.netplan import (
    DISABLED_DIR,
    NETPLAN_FILE,
    NetplanError,
    allocate_links,
    apply_plan,
    build_plan,
    infer_topology,
    inspect_nodes,
    remove_net,
    render_netplan,
)

PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE lmds-cluster-a"


# ── /sys ปลอมของ Spark ─────────────────────────────────────────────────────────
def fake_sys(root: Path, *, port1_carrier=True, port2_carrier=False, addresses=None,
             netplan: dict[str, str] | None = None, unreadable: list[str] = ()) -> dict:
    """/sys/class/net + /sys/class/infiniband + /etc/netplan ตามหน้าตาจริงของ spark-head"""
    net = root / "sys/class/net"
    ib = root / "sys/class/infiniband"
    devices = root / "devices"
    net.mkdir(parents=True)
    ib.mkdir(parents=True)
    drivers = root / "drivers"
    (drivers / "mlx5_core").mkdir(parents=True)
    (drivers / "r8127").mkdir()
    layout = {
        "enp1s0f0np0": ("0000:01:00.0", "rocep1s0f0", port1_carrier),
        "enp1s0f1np1": ("0000:01:00.1", "rocep1s0f1", port1_carrier),
        "enP2p1s0f0np0": ("0002:01:00.0", "roceP2p1s0f0", port2_carrier),
        "enP2p1s0f1np1": ("0002:01:00.1", "roceP2p1s0f1", port2_carrier),
    }
    for iface, (pci, rdma, carrier) in layout.items():
        d = net / iface
        d.mkdir()
        dev = devices / pci
        (dev / "infiniband" / rdma).mkdir(parents=True)
        (dev / "vendor").write_text("0x15b3\n")
        os.symlink(drivers / "mlx5_core", dev / "driver")
        os.symlink(dev, d / "device")
        (d / "operstate").write_text("up\n" if carrier else "down\n")
        (d / "speed").write_text("200000\n" if carrier else "-1\n")
        (d / "carrier").write_text("1\n" if carrier else "0\n")
        (ib / rdma).mkdir()
    # สายบริหาร 1G — ไม่ใช่ ConnectX ต้องไม่ถูกจัดเข้าพอร์ต QSFP
    mgmt = net / "enP7s7"
    mgmt.mkdir()
    dev = devices / "0007:01:00.0"
    dev.mkdir()
    (dev / "vendor").write_text("0x10ec\n")
    os.symlink(drivers / "r8127", dev / "driver")
    os.symlink(dev, mgmt / "device")
    (mgmt / "operstate").write_text("up\n")
    (mgmt / "speed").write_text("1000\n")
    (mgmt / "carrier").write_text("1\n")
    (net / "lo").mkdir()
    etc = root / "etc/netplan"
    etc.mkdir(parents=True)
    for name, text in (netplan or {}).items():
        (etc / name).write_text(text)
    for name in unreadable:
        (etc / name).write_text("network: {}\n")
        (etc / name).chmod(0)
    return {"sys_net": net, "sys_ib": ib, "netplan_dir": etc, "addresses": addresses or {
        "enP7s7": "10.2.1.195/24", "enp1s0f0np0": "169.254.21.127/16", "enp1s0f1np1": "10.100.152.1/24",
        "enP2p1s0f0np0": "169.254.31.110/16",
    }}


def test_detect_fabric_reports_both_spark_ports_with_carrier_function_and_rdma(tmp_path):
    fabric = detect_fabric(**fake_sys(tmp_path, port1_carrier=True, port2_carrier=False,
                                      netplan={"99-nvidia-sync-cluster.yaml":
                                               "network:\n  ethernets:\n    enp1s0f1np1:\n      addresses: [10.100.152.1/24]\n"}))
    by_name = {l["iface"]: l for l in fabric["links"]}
    p1 = by_name["enp1s0f1np1"]
    assert (p1["qsfp_port"], p1["function"], p1["carrier"], p1["rdma_device"]) == (1, 1, True, "rocep1s0f1")
    assert p1["netplan_managed"] is True and p1["netplan_files"] == ["99-nvidia-sync-cluster.yaml"]
    p2 = by_name["enP2p1s0f0np0"]
    assert (p2["qsfp_port"], p2["function"], p2["carrier"]) == (2, 0, False), "พอร์ต 2 ไม่มีสาย = NO-CARRIER"
    assert p2["netplan_managed"] is False
    assert by_name["enP7s7"]["qsfp_port"] is None, "สายบริหารไม่ใช่พอร์ต QSFP"
    # คีย์เดิมยังอยู่ครบ — หน้า Cluster และ doctor คู่ยังใช้ของเดิม
    for key in ("ip", "prefix", "link_local", "speed_gbps", "driver", "state", "connectx", "rdma"):
        assert key in p1
    assert fabric["tier"] == "rdma" and fabric["best_gbps"] == 200 and fabric["cluster_capable"]
    assert fabric["nvidia_sync_netplan"] is True
    ports = {p["port"]: p for p in fabric["qsfp_ports"]}
    assert ports[1]["ifaces"] == ["enp1s0f0np0", "enp1s0f1np1"] and ports[1]["carrier"] is True
    assert ports[1]["configured"] == "enp1s0f1np1" and ports[1]["ip"] == "10.100.152.1"
    assert ports[2]["carrier"] is False and ports[2]["configured"] == "" and ports[2]["speed_gbps"] is None
    assert ports[1]["rdma_devices"] == ["rocep1s0f0", "rocep1s0f1"]


def test_detect_fabric_reports_unreadable_netplan_as_unknown_not_unmanaged(tmp_path):
    """ไฟล์ netplan บน Spark เป็น 0600 ของ root — อ่านไม่ได้ต้องตอบ "ไม่รู้" ไม่ใช่ "ไม่ได้ถูกจัดการ" """
    if os.geteuid() == 0:
        pytest.skip("root อ่านไฟล์ mode 0 ได้")
    fabric = detect_fabric(**fake_sys(tmp_path, unreadable=["99-nvidia-sync-cluster.yaml", "00-installer-config.yaml"]))
    assert fabric["netplan_files"] == ["00-installer-config.yaml", "99-nvidia-sync-cluster.yaml"]
    assert fabric["netplan_unreadable"] == ["00-installer-config.yaml", "99-nvidia-sync-cluster.yaml"]
    assert fabric["nvidia_sync_netplan"] is True
    assert all(l["netplan_managed"] is None for l in fabric["links"] if l["connectx"])


def test_group_qsfp_ports_understands_old_node_payloads_without_carrier_keys():
    """node รุ่นก่อน 0.6 ส่งแค่ iface/state/speed — hub ต้องยังจัดพอร์ตได้จากชื่อ"""
    links = [
        {"iface": "enp1s0f0np0", "ip": "169.254.9.9", "prefix": 16, "speed_gbps": 200, "state": "up", "connectx": True},
        {"iface": "enp1s0f1np1", "ip": "10.100.152.2", "prefix": 24, "speed_gbps": 200, "state": "up", "connectx": True},
        {"iface": "enP2p1s0f0np0", "ip": "", "prefix": None, "speed_gbps": None, "state": "down", "connectx": True},
        {"iface": "enP2p1s0f1np1", "ip": "", "prefix": None, "speed_gbps": None, "state": "down", "connectx": True},
        {"iface": "enP7s7", "ip": "10.2.1.194", "prefix": 24, "speed_gbps": 1, "state": "up", "connectx": False},
    ]
    ports = {p["port"]: p for p in group_qsfp_ports(links)}
    assert ports[1]["carrier"] is True and ports[1]["configured"] == "enp1s0f1np1"
    assert ports[2]["carrier"] is False and ports[2]["ifaces"] == ["enP2p1s0f0np0", "enP2p1s0f1np1"]


# ── host payload ที่ hub มีในแคช ───────────────────────────────────────────────
def spark(mgmt: str, p1: str = "", p2: str = "", *, carrier1=True, carrier2=False, nvidia_sync=False,
          speed=200, hostname="") -> dict:
    """payload ของ Spark หนึ่งเครื่อง — p1/p2 = IP ที่ตั้งไว้แล้วบน f1 ของพอร์ตนั้น (ว่าง = link-local)"""
    def link(iface, port, fn, carrier, ip):
        return {"iface": iface, "ip": ip, "prefix": 24 if ip and not ip.startswith("169.254.") else (16 if ip else None),
                "link_local": ip.startswith("169.254."), "speed_gbps": speed if carrier else None,
                "driver": "mlx5_core", "state": "up" if carrier else "down", "connectx": True, "rdma": True,
                "carrier": carrier, "qsfp_port": port, "function": fn, "rdma_device": f"roce{iface[2:]}",
                "netplan_managed": None, "netplan_files": []}
    links = [
        link("enp1s0f0np0", 1, 0, carrier1, "169.254.1.1" if carrier1 else ""),
        link("enp1s0f1np1", 1, 1, carrier1, p1 or ("169.254.1.2" if carrier1 else "")),
        link("enP2p1s0f0np0", 2, 0, carrier2, "169.254.2.1" if carrier2 else ""),
        link("enP2p1s0f1np1", 2, 1, carrier2, p2 or ("169.254.2.2" if carrier2 else "")),
        {"iface": "enP7s7", "ip": mgmt, "prefix": 24, "link_local": False, "speed_gbps": 1, "driver": "r8127",
         "state": "up", "connectx": False, "rdma": False, "carrier": True, "qsfp_port": None, "function": None,
         "rdma_device": "", "netplan_managed": None, "netplan_files": []},
    ]
    return {
        "hostname": hostname or f"spark-{mgmt.rsplit('.', 1)[-1]}", "arch": "aarch64", "profile": "dgx_spark",
        "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128}], "disk_free_gb": 900,
        "fabric": {"links": links, "rdma_devices": [l["rdma_device"] for l in links if l.get("rdma_device")],
                   "best_gbps": 200, "tier": "rdma", "cluster_capable": True,
                   "qsfp_ports": group_qsfp_ports(links),
                   "netplan_files": ["00-installer-config.yaml"] + (["99-nvidia-sync-cluster.yaml"] if nvidia_sync else []),
                   "netplan_unreadable": ["00-installer-config.yaml"], "nvidia_sync_netplan": nvidia_sync},
    }


# ── topology ────────────────────────────────────────────────────────────────────
def test_topology_two_machines_one_cable_is_direct():
    got = infer_topology({"a": [1], "b": [1]}, ["a", "b"])
    assert got["topology"] == "direct-2"
    assert got["links"] == [{"id": 0, "ends": [{"node": "a", "port": 1}, {"node": "b", "port": 1}]}]
    # เสียบไขว้ (พอร์ต 1 ของ a ไปพอร์ต 2 ของ b) ก็ยังเป็นคู่ตรง
    crossed = infer_topology({"a": [1], "b": [2]}, ["a", "b"])
    assert crossed["topology"] == "direct-2" and crossed["links"][0]["ends"][1]["port"] == 2


def test_topology_two_machines_two_cables_is_direct_with_two_links():
    """คู่จริง spark-head/spark-worker: ทั้งสองพอร์ตมีสาย → สองลิงก์ สมมติพอร์ตเดียวกันชนกัน"""
    got = infer_topology({"a": [1, 2], "b": [1, 2]}, ["a", "b"])
    assert got["topology"] == "direct-2" and [l["id"] for l in got["links"]] == [0, 1]
    assert got["links"][1]["ends"] == [{"node": "a", "port": 2}, {"node": "b", "port": 2}]


def test_topology_three_machines_both_ports_is_a_ring_in_nvidia_order():
    got = infer_topology({"a": [1, 2], "b": [1, 2], "c": [1, 2]}, ["a", "b", "c"])
    assert got["topology"] == "ring-3"
    assert [(l["ends"][0]["node"], l["ends"][0]["port"], l["ends"][1]["node"], l["ends"][1]["port"])
            for l in got["links"]] == [("a", 1, "b", 2), ("b", 1, "c", 2), ("c", 1, "a", 2)]


def test_topology_one_cable_each_is_a_switch_for_three_or_four():
    four = infer_topology({n: [1] for n in "abcd"}, list("abcd"))
    assert four["topology"] == "switch-4" and len(four["links"]) == 1
    assert [e["node"] for e in four["links"][0]["ends"]] == ["a", "b", "c", "d"]
    assert infer_topology({n: [2] for n in "abc"}, list("abc"))["topology"] == "switch-3"
    # 2 เครื่องสายละช่องเป็น direct โดยปริยาย — บังคับเป็น switch ได้
    assert infer_topology({"a": [1], "b": [1]}, ["a", "b"], forced="switch")["topology"] == "switch-2"


def test_topology_rejects_missing_cable_and_mixed_layouts_with_a_reason():
    missing = infer_topology({"a": [1], "b": []}, ["a", "b"])
    assert missing["topology"] == "unknown" and "no cable detected on b" in missing["reason"]
    mixed = infer_topology({"a": [1, 2], "b": [1]}, ["a", "b"])
    assert mixed["topology"] == "unknown" and "mixed cabling" in mixed["reason"]
    mixed3 = infer_topology({"a": [1, 2], "b": [1, 2], "c": [1]}, ["a", "b", "c"])
    assert mixed3["topology"] == "unknown" and "c (1)" in mixed3["reason"]
    four = infer_topology({"a": [1, 2], "b": [1], "c": [1], "d": [1]}, list("abcd"))
    assert four["topology"] == "unknown" and "switch" in four["reason"] and "a has both" in four["reason"]
    assert infer_topology({"a": [1]}, ["a"])["topology"] == "unknown"
    assert "unknown topology 'mesh'" in infer_topology({"a": [1], "b": [1]}, ["a", "b"], forced="mesh")["reason"]
    assert "ring needs both" in infer_topology({n: [1] for n in "abc"}, list("abc"), forced="ring")["reason"]


# ── IP allocation ────────────────────────────────────────────────────────────────
def test_allocation_is_deterministic_and_never_touches_management_interfaces():
    hosts = {"a": spark("10.2.1.1"), "b": spark("10.2.1.2")}
    topo = infer_topology({"a": [1], "b": [1]}, ["a", "b"])
    first, _ = allocate_links(topo, hosts)
    second, _ = allocate_links(topo, hosts)
    assert first == second
    (link,) = first
    assert link["subnet"] == "10.100.152.0/24"
    assert [(e["node"], e["iface"], e["ip"], e["prefix"], e["changed"]) for e in link["ends"]] == [
        ("a", "enp1s0f1np1", "10.100.152.1", 24, True), ("b", "enp1s0f1np1", "10.100.152.2", 24, True)]
    assert all("enP7s7" not in e["iface"] for e in link["ends"])


def test_allocation_keeps_existing_addresses_that_already_match():
    """คู่จริง: 152.x บนพอร์ต 1 และ 153.x บนพอร์ต 2 ตั้งไว้แล้ว — แผนต้องบอกว่า "ไม่เปลี่ยน" """
    hosts = {"spark-head": spark("10.2.1.195", "10.100.152.1", "10.100.153.1", carrier2=True),
             "spark-worker": spark("10.2.1.194", "10.100.152.2", "10.100.153.2", carrier2=True)}
    topo = infer_topology({"spark-head": [1, 2], "spark-worker": [1, 2]}, ["spark-head", "spark-worker"])
    links, warnings = allocate_links(topo, hosts)
    assert warnings == []
    assert [l["subnet"] for l in links] == ["10.100.152.0/24", "10.100.153.0/24"]
    assert all(not e["changed"] for l in links for e in l["ends"])


def test_a_new_worker_joins_the_subnet_the_head_already_uses():
    hosts = {"head": spark("10.2.1.1", "10.100.152.1"), "new": spark("10.2.1.9")}
    links, _ = allocate_links(infer_topology({"head": [1], "new": [1]}, ["head", "new"]), hosts)
    ends = {e["node"]: e for e in links[0]["ends"]}
    assert ends["head"]["ip"] == "10.100.152.1" and not ends["head"]["changed"]
    assert ends["new"]["ip"] == "10.100.152.2" and ends["new"]["changed"]


def test_ring_gets_one_subnet_per_link_and_switch_gets_one_subnet_for_all():
    ring_hosts = {n: spark(f"10.2.1.{i}", carrier2=True) for i, n in enumerate("abc", 1)}
    ring, _ = allocate_links(infer_topology({n: [1, 2] for n in "abc"}, list("abc")), ring_hosts,
                             base_subnet="10.200.0.0/24")
    assert [l["subnet"] for l in ring] == ["10.200.0.0/24", "10.200.1.0/24", "10.200.2.0/24"]
    assert [(e["node"], e["iface"], e["ip"]) for e in ring[2]["ends"]] == [
        ("c", "enp1s0f1np1", "10.200.2.1"), ("a", "enP2p1s0f1np1", "10.200.2.2")]
    switch_hosts = {n: spark(f"10.2.1.{i}") for i, n in enumerate("abcd", 1)}
    switch, _ = allocate_links(infer_topology({n: [1] for n in "abcd"}, list("abcd")), switch_hosts)
    assert [e["ip"] for e in switch[0]["ends"]] == ["10.100.152.1", "10.100.152.2", "10.100.152.3", "10.100.152.4"]
    with pytest.raises(NetplanError, match="base subnet"):
        allocate_links(infer_topology({"a": [1], "b": [1]}, ["a", "b"]), switch_hosts, base_subnet="nope")


def test_conflicting_existing_addresses_are_reallocated_with_a_warning():
    hosts = {"a": spark("10.2.1.1", "10.100.152.1"), "b": spark("10.2.1.2", "10.100.170.2")}
    links, warnings = allocate_links(infer_topology({"a": [1], "b": [1]}, ["a", "b"]), hosts)
    assert warnings and "do not fit one subnet" in warnings[0]
    assert [e["ip"] for e in links[0]["ends"]] == ["10.100.152.1", "10.100.152.2"]


# ── netplan YAML + plan ──────────────────────────────────────────────────────────
def test_netplan_yaml_lists_only_cluster_interfaces_with_static_addresses():
    text = render_netplan([{"iface": "enP2p1s0f1np1", "ip": "10.100.153.1", "prefix": 24},
                           {"iface": "enp1s0f1np1", "ip": "10.100.152.1", "prefix": 24}])
    assert text == (
        "# Managed by LMDS (lmds cluster apply) — ConnectX cluster links. Do not edit by hand;\n"
        "# re-run `lmds cluster apply` or remove with `lmds cluster remove-net <node>`.\n"
        "network:\n  version: 2\n  renderer: networkd\n  ethernets:\n"
        "    enP2p1s0f1np1:\n      dhcp4: no\n      addresses: [10.100.153.1/24]\n      optional: true\n"
        "    enp1s0f1np1:\n      dhcp4: no\n      addresses: [10.100.152.1/24]\n      optional: true\n"
    )
    assert "gateway" not in text and "routes" not in text
    import yaml

    parsed = yaml.safe_load(text)
    assert set(parsed["network"]["ethernets"]) == {"enp1s0f1np1", "enP2p1s0f1np1"}


def test_plan_shape_for_a_ring_and_the_registry_fields_it_derives():
    hosts = {n: spark(f"10.2.1.{i}", carrier2=True, nvidia_sync=(n == "c")) for i, n in enumerate("abc", 1)}
    plan = build_plan(list("abc"), hosts)
    assert plan["ok"] and plan["topology"] == "ring-3" and plan["order"] == ["a", "b", "c"]
    assert set(plan) >= {"links", "nodes", "registry", "warnings", "base_subnet", "reason"}
    a = plan["nodes"]["a"]
    assert [(l["iface"], l["ip"], l["peer_node"], l["peer_ip"], l["link_id"], l["qsfp_port"]) for l in a["links"]] == [
        ("enp1s0f1np1", "10.100.152.1", "b", "10.100.152.2", 0, 1),
        ("enP2p1s0f1np1", "10.100.154.2", "c", "10.100.154.1", 2, 2)]
    assert a["cluster_ip"] == "10.100.152.1" and a["cluster_iface"] == "enp1s0f1np1"
    # worker c ถึง head ทางลิงก์ 2 — cluster_ip ของมันต้องเป็นปลายที่หันไปหา head
    assert plan["registry"]["c"] == {"cluster_ip": "10.100.154.1", "cluster_iface": "enp1s0f1np1",
                                     "cluster_links": plan["nodes"]["c"]["links"]}
    assert "enp1s0f1np1" in a["netplan"] and "enP2p1s0f1np1" in a["netplan"] and "enP7s7" not in a["netplan"]
    assert any("ring-3" in w for w in plan["warnings"]) and any("99-nvidia-sync" in w and "c:" in w for w in plan["warnings"])


def test_plan_reports_why_when_cabling_is_wrong_or_the_node_is_not_a_spark():
    hosts = {"a": spark("10.2.1.1"), "b": spark("10.2.1.2", carrier1=False)}
    plan = build_plan(["a", "b"], hosts)
    assert plan["ok"] is False and "no cable detected on b" in plan["reason"] and plan["nodes"] == {}
    down = build_plan(["a", "b"], {"a": hosts["a"], "b": None})
    assert down["ok"] is False and "no inventory for b" in down["reason"]
    hub = {"hostname": "hub", "gpus": [], "fabric": {"links": [{"iface": "eth0", "ip": "10.0.0.1", "prefix": 24,
                                                               "speed_gbps": 10, "state": "up", "connectx": False}]}}
    assert "no ConnectX QSFP ports" in build_plan(["a", "hub"], {"a": hosts["a"], "hub": hub})["reason"]


def test_plan_marks_nothing_changed_when_the_real_pair_is_already_configured():
    hosts = {"spark-head": spark("10.2.1.195", "10.100.152.1", "10.100.153.1", carrier2=True),
             "spark-worker": spark("10.2.1.194", "10.100.152.2", "10.100.153.2", carrier2=True)}
    reg = {"spark-head": Node("spark-head", "10.2.1.195", "nvidia", cluster_ip="10.100.152.1", cluster_iface="enp1s0f1np1"),
           "spark-worker": Node("spark-worker", "10.2.1.194", "nvidia", cluster_ip="10.100.152.2", cluster_iface="enp1s0f1np1")}
    plan = build_plan(["spark-head", "spark-worker"], hosts, nodes=reg)
    assert plan["ok"] and plan["topology"] == "direct-2"
    assert all(not entry["changed"] for entry in plan["nodes"].values())
    assert plan["registry"]["spark-worker"]["cluster_ip"] == "10.100.152.2"


# ── apply กับ ssh/sudo/netplan ปลอม ─────────────────────────────────────────────
class FakeFleet:
    """`lmds.nodes.run` ปลอม: จำทุกคำสั่ง + stdin ต่อเครื่อง และเล่นบทของ sudo/netplan/ip/ping/iperf3"""

    def __init__(self, passwords: dict[str, str], *, lose_carrier_on: str = "", ping_fail: bool = False):
        self.passwords = passwords
        self.calls: list[tuple[str, str, str]] = []
        self.staged: dict[str, str] = {}
        self.lose_carrier_on = lose_carrier_on
        self.ping_fail = ping_fail

    def __call__(self, node, command, timeout=60, stdin_text=""):
        self.calls.append((node.name, command, stdin_text))
        code, out, err = self.answer(node.name, command, stdin_text)
        return SimpleNamespace(ok=code == 0, exit_code=code, stdout=out, stderr=err)

    def on(self, name):
        return [c for n, c, _ in self.calls if n == name]

    def answer(self, name, command, stdin):
        if command.startswith("sudo -S"):
            if stdin != self.passwords[name] + "\n":
                return 1, "", "Sorry, try again."
            if command.endswith("-v && echo LMDS_SUDO_OK"):
                return 0, "LMDS_SUDO_OK\n", ""
            if "netplan generate && netplan apply" in command:
                return 0, "disabled /etc/netplan/99-nvidia-sync-cluster.yaml\nLMDS_NETPLAN_APPLIED\n", ""
            if "LMDS_NETPLAN_ROLLED_BACK" in command:
                return 0, "restored 99-nvidia-sync-cluster.yaml\nLMDS_NETPLAN_ROLLED_BACK\n", ""
            if "LMDS_NETPLAN_REMOVED" in command:
                return 0, f"moved to {DISABLED_DIR}/99-lmds-cluster.yaml.x\nLMDS_NETPLAN_REMOVED\n", ""
            return 1, "", f"unexpected sudo command: {command}"
        if command.startswith("f=$(mktemp /tmp/lmds-netplan."):
            self.staged[name] = stdin
            return 0, f"/tmp/lmds-netplan.{name}\n", ""
        if command.startswith("ip -br addr show dev"):
            import yaml

            ethernets = yaml.safe_load(self.staged[name])["network"]["ethernets"]
            rows = []
            for iface, spec in ethernets.items():
                rows.append(f"{iface:16} UP {spec['addresses'][0]} fe80::1/64")
                flags = "<BROADCAST,MULTICAST,UP>" if iface == self.lose_carrier_on else "<BROADCAST,MULTICAST,UP,LOWER_UP>"
                rows.append(f"{iface:16} UP 48:21:0b:96:4d:fc {flags}")
            return 0, "\n".join(rows) + "\n", ""
        if command.startswith("ping "):
            return (1 if self.ping_fail else 0), "", ""
        if "ssh-keygen" in command:
            return 0, PUB + "\n", ""
        if command.startswith("command -v iperf3"):
            return 1, "", ""
        return 0, "", ""


def register(name: str, mgmt: str, **fields) -> Node:
    return add(Node(name=name, host=mgmt, user="nvidia", site="Neronain", **fields))


@pytest.fixture
def fresh_pair():
    """สองเครื่องใหม่: สายที่พอร์ต 1 ทั้งคู่ ยังไม่มี IP เลย · worker เคยผ่าน NVIDIA Sync มา"""
    register("a", "10.2.1.1")
    register("b", "10.2.1.2")
    hosts = {"a": spark("10.2.1.1"), "b": spark("10.2.1.2", nvidia_sync=True)}
    plan = build_plan(["a", "b"], hosts)
    assert plan["ok"]
    return plan


def test_apply_runs_the_sequence_per_node_and_keeps_the_password_out_of_argv_and_logs(fresh_pair):
    plan = fresh_pair
    passwords = {"a": "s3cret-A", "b": "s3cret-B"}
    fleet = FakeFleet(passwords)
    result = apply_plan(plan, dict(passwords), nodes={"a": find("a"), "b": find("b")}, runner=fleet,
                        sleep=lambda s: None, stamp="20260905-120000")

    assert result["ok"] and result["applied"], [s for s in result["steps"] if not s["ok"]]
    names = [n for n, _, _ in fleet.calls]
    # รหัสของทุกเครื่องถูกตรวจก่อนแตะเครื่องแรก
    assert names[:2] == ["a", "b"] and all(c.endswith("-v && echo LMDS_SUDO_OK") for _, c, _ in fleet.calls[:2])
    a_cmds = fleet.on("a")
    assert a_cmds[1].startswith("f=$(mktemp /tmp/lmds-netplan.")
    assert a_cmds[2].startswith("sudo -S -p '' bash -c ") and "netplan generate && netplan apply" in a_cmds[2]
    assert "/root/netplan-disabled" in a_cmds[2] and "install -m 0600 -o root -g root /tmp/lmds-netplan.a" in a_cmds[2]
    assert "enp1s0f1np1" in a_cmds[2] and "20260905-120000" in a_cmds[2]
    assert a_cmds[3].startswith("ip -br addr show dev enp1s0f1np1")
    # b ถูกแตะหลัง a เสร็จครบ (ไม่สลับกัน) และ ping/pair มาหลังทั้งคู่ขึ้น
    order = [(n, c.split()[0]) for n, c, _ in fleet.calls]
    idx_b_apply = next(i for i, (n, c, _) in enumerate(fleet.calls) if n == "b" and "netplan generate" in c)
    idx_a_verify = next(i for i, (n, c, _) in enumerate(fleet.calls) if n == "a" and c.startswith("ip -br addr"))
    assert idx_a_verify < idx_b_apply
    pings = [(n, c) for n, c, _ in fleet.calls if c.startswith("ping ")]
    assert pings == [("a", "ping -c 3 -W 2 -I enp1s0f1np1 10.100.152.2 >/dev/null 2>&1"),
                     ("b", "ping -c 3 -W 2 -I enp1s0f1np1 10.100.152.1 >/dev/null 2>&1")]
    assert any("ssh-keygen" in c for c in fleet.on("a")) and any("authorized_keys" in c for c in fleet.on("b"))
    assert any(c == "ssh -o BatchMode=yes -o ConnectTimeout=8 nvidia@10.100.152.2 true" for c in a_cmds)
    # YAML ที่ส่งไปคือของแผน
    assert fleet.staged["a"] == plan["nodes"]["a"]["netplan"]
    # รหัสไม่อยู่ใน argv ไหนเลย และไม่อยู่ในรายงาน
    for _, command, _ in fleet.calls:
        assert "s3cret" not in command
    assert "s3cret" not in json.dumps(result, ensure_ascii=False)
    # ทะเบียนได้ค่าจริง
    assert find("a").cluster_ip == "10.100.152.1" and find("a").cluster_iface == "enp1s0f1np1"
    assert find("b").cluster_ip == "10.100.152.2"
    assert find("b").cluster_links == plan["nodes"]["b"]["links"]
    assert result["registry"]["b"]["cluster_ip"] == "10.100.152.2"
    assert result["speed"] and result["speed"][0]["skipped"] == "iperf3 not installed on both ends"
    kinds = [s["step"] for s in result["steps"] if s["node"] == "a"]
    assert kinds[:4] == ["sudo password accepted", "stage netplan file",
                         f"write {NETPLAN_FILE} + netplan apply", "verify addresses + carrier"]
    assert "disabled /etc/netplan/99-nvidia-sync-cluster.yaml" in [s["detail"] for s in result["steps"] if s["node"] == "b"][2]


def test_apply_is_idempotent_on_a_second_run(fresh_pair):
    passwords = {"a": "pw", "b": "pw"}
    nodes = {"a": find("a"), "b": find("b")}
    first = apply_plan(fresh_pair, passwords, nodes=nodes, runner=FakeFleet(passwords), sleep=lambda s: None)
    second = apply_plan(fresh_pair, passwords, nodes=nodes, runner=FakeFleet(passwords), sleep=lambda s: None)
    assert first["ok"] and second["ok"]
    assert [s["step"] for s in first["steps"]] == [s["step"] for s in second["steps"]]
    assert find("a").cluster_ip == "10.100.152.1"


def test_apply_rolls_back_the_node_when_the_interface_loses_carrier(fresh_pair):
    passwords = {"a": "pw", "b": "pw"}
    fleet = FakeFleet(passwords, lose_carrier_on="enp1s0f1np1")
    calls_sleep = []
    result = apply_plan(fresh_pair, passwords, nodes={"a": find("a"), "b": find("b")}, runner=fleet,
                        sleep=calls_sleep.append, stamp="S1")
    assert not result["ok"] and not result["applied"]
    assert result["nodes"]["a"] == {"ok": False, "rolled_back": True}
    assert "b" not in result["nodes"], "เครื่องที่สองต้องไม่ถูกแตะเมื่อเครื่องแรกล้ม"
    rollback = [c for c in fleet.on("a") if "LMDS_NETPLAN_ROLLED_BACK" in c]
    assert len(rollback) == 1 and "S1" in rollback[0] and rollback[0].startswith("sudo -S -p '' bash -c ")
    assert len(calls_sleep) >= 3, "ต้องลองซ้ำก่อนตัดสินว่าสายหาย (ลิงก์กระพริบหลัง netplan apply)"
    failed = [s for s in result["steps"] if not s["ok"]]
    assert failed and "lost carrier" in failed[0]["detail"]
    assert find("a").cluster_ip == "", "ทะเบียนต้องไม่ถูกเขียนเมื่อ apply ล้ม"
    assert not any(c.startswith("ping ") for _, c, _ in fleet.calls)


def test_apply_stops_before_touching_anything_when_a_sudo_password_is_wrong(fresh_pair):
    fleet = FakeFleet({"a": "right", "b": "right"})
    result = apply_plan(fresh_pair, {"a": "right", "b": "wrong"}, nodes={"a": find("a"), "b": find("b")},
                        runner=fleet, sleep=lambda s: None)
    assert not result["applied"]
    assert [s["ok"] for s in result["steps"]] == [True, False]
    assert result["steps"][1]["node"] == "b" and "Sorry" in result["steps"][1]["detail"]
    assert not any("netplan" in c for _, c, _ in fleet.calls), "ห้ามเขียนอะไรถ้ารหัสเครื่องใดเครื่องหนึ่งผิด"
    missing = apply_plan(fresh_pair, {"a": "x"}, nodes={"a": find("a"), "b": find("b")}, runner=fleet)
    assert missing["steps"][-1]["step"] == "sudo password" and "b" in missing["steps"][-1]["detail"]


def test_apply_reports_ping_failure_as_crossed_cabling_but_keeps_the_addresses(fresh_pair):
    passwords = {"a": "pw", "b": "pw"}
    result = apply_plan(fresh_pair, passwords, nodes={"a": find("a"), "b": find("b")},
                        runner=FakeFleet(passwords, ping_fail=True), sleep=lambda s: None, pair=False, speed_test=False)
    assert result["applied"] and not result["ok"]
    assert all(not p["ok"] for p in result["pings"]) and result["pairing"] == []
    assert any("crossed" in s["detail"] for s in result["steps"] if s["step"].startswith("ping"))
    assert find("a").cluster_ip == "10.100.152.1", "IP ขึ้นจริงแล้ว ทะเบียนต้องตรงกับเครื่อง"


def test_remove_net_moves_the_file_aside_and_clears_the_registry():
    register("a", "10.2.1.1", cluster_ip="10.100.152.1", cluster_iface="enp1s0f1np1",
             cluster_links=[{"iface": "enp1s0f1np1", "ip": "10.100.152.1"}])
    fleet = FakeFleet({"a": "pw"})
    result = remove_net(find("a"), "pw", runner=fleet, stamp="S2")
    assert result["ok"] and result["removed"]
    moved = [c for c in fleet.on("a") if "LMDS_NETPLAN_REMOVED" in c]
    assert len(moved) == 1 and f"mv \"$f\" \"$d/$(basename \"$f\").$s\"" in moved[0] and "S2" in moved[0]
    assert "netplan generate; netplan apply" in moved[0]
    node = find("a")
    assert (node.cluster_ip, node.cluster_iface, node.cluster_links) == ("", "", [])
    assert "pw" not in json.dumps(result)
    denied = remove_net(find("a"), "nope", runner=FakeFleet({"a": "pw"}))
    assert not denied["ok"] and "nope" not in json.dumps(denied)


def test_registry_round_trips_cluster_links_and_tolerates_null():
    from lmds.nodes import load, nodes_file, update

    register("a", "10.2.1.1")
    update("a", cluster_links=[{"iface": "enp1s0f1np1", "ip": "10.100.152.1", "peer_node": "b"}])
    assert load()[0].cluster_links[0]["peer_node"] == "b"
    text = nodes_file().read_text(encoding="utf-8").replace("cluster_links:\n  - iface: enp1s0f1np1\n    ip: 10.100.152.1\n    peer_node: b",
                                                            "cluster_links:")
    nodes_file().write_text(text, encoding="utf-8")
    assert load()[0].cluster_links == []


# ── doctor ──────────────────────────────────────────────────────────────────────
def test_network_doctor_names_the_missing_cable_and_the_nvidia_file_in_both_languages():
    from lmds.nodes.doctor import describe, diagnose_network

    reg = {"a": register("a", "10.2.1.1"), "b": register("b", "10.2.1.2")}
    hosts = {"a": spark("10.2.1.1", nvidia_sync=True), "b": spark("10.2.1.2", carrier1=False)}
    report = diagnose_network(["a", "b"], nodes=reg, hosts=hosts)
    by_kind = {}
    for f in report["findings"]:
        by_kind.setdefault(f["kind"], []).append(f)
    assert not report["ok"] and report["topology"] == "unknown"
    cabling = {f["names"][0]: f for f in by_kind["cabling"]}
    assert cabling["a"]["ok"] and cabling["a"]["data"]["ports"] == "1"
    assert not cabling["b"]["ok"] and "plug a QSFP cable into b" in cabling["b"]["fix"]
    assert describe(cabling["b"], "th") == "b: ไม่พบสาย — พอร์ต QSFP ทั้งสองช่องขึ้น NO-CARRIER"
    assert describe(cabling["b"], "en") == "b: no cable detected — both QSFP ports show NO-CARRIER"
    topo = by_kind["topology"][0]
    assert not topo["ok"] and "no cable detected on b" in describe(topo, "en")
    managed = {f["names"][0]: f for f in by_kind["netplan-managed"]}
    assert managed["a"]["level"] == "warn" and "99-nvidia-sync-cluster.yaml" in describe(managed["a"], "en")
    assert managed["b"]["ok"]


def test_network_doctor_pings_every_planned_link_when_given_a_runner():
    from lmds.nodes.doctor import diagnose_network

    reg = {n: register(n, f"10.2.1.{i}") for i, n in enumerate("abc", 1)}
    hosts = {n: spark(f"10.2.1.{i}", carrier2=True, speed=50 if n == "c" else 200) for i, n in enumerate("abc", 1)}
    plan = build_plan(list("abc"), hosts)
    fleet = FakeFleet({})
    report = diagnose_network(list("abc"), nodes=reg, hosts=hosts, runner=fleet, plan=plan)
    assert report["ok"] and report["topology"] == "ring-3"
    pings = [f for f in report["findings"] if f["kind"] == "link-ping"]
    assert len(pings) == 6 and all(f["ok"] for f in pings)
    assert [c for _, c, _ in fleet.calls][0] == "ping -c1 -W2 -I enp1s0f1np1 10.100.152.2 >/dev/null 2>&1"
    slow = [f for f in report["findings"] if f["kind"] == "port-speed"]
    assert [(f["names"][0], f["level"], f["data"]["speed"]) for f in slow] == [("c", "warn", 50), ("c", "warn", 50)]


def test_inspect_summarises_ports_per_node_in_the_shape_the_wizard_reads():
    hosts = {"a": spark("10.2.1.1", "10.100.152.1"), "b": None}
    view = inspect_nodes(["a", "b"], hosts, {"b": "ssh: timed out"})
    a = view["nodes"]["a"]
    assert a["spark"] and a["sudo_needed"] and a["ports"][0]["configured"] == "enp1s0f1np1"
    ports = a["fabric"]["ports"]
    assert [p["qsfp_port"] for p in ports] == [1, 2]
    f1 = ports[0]["interfaces"][1]
    assert f1 == {"iface": "enp1s0f1np1", "function": "f1", "carrier": True, "speed_gbps": 200, "ip": "10.100.152.1",
                  "prefix": 24, "rdma_device": "rocep1s0f1np1", "netplan_managed": None}
    assert ports[0]["interfaces"][0]["ip"] == "", "169.254.x = ยังไม่ตั้ง ต้องโชว์ว่าไม่มี IP"
    assert ports[1]["interfaces"][0]["carrier"] is False and ports[1]["carrier"] is False
    b = view["nodes"]["b"]
    assert b["reachable"] is False and b["error"] == "ssh: timed out" and b["fabric"]["ports"] == []
    topo = view["topology"]
    assert topo["kind"] == topo["topology"] == "unknown" and "no inventory for b" in topo["reason"]
    both = inspect_nodes(["a", "b"], {"a": hosts["a"], "b": spark("10.2.1.2")})
    assert both["topology"]["kind"] == "direct-2" and both["topology"]["cabled"] == {"a": [1], "b": [1]}


def test_plan_carries_the_wizard_aliases_and_change_summaries():
    hosts = {"a": spark("10.2.1.1", "10.100.152.1"), "b": spark("10.2.1.2", nvidia_sync=True)}
    reg = {"a": Node("a", "10.2.1.1", "nvidia", cluster_ip="10.100.152.1"), "b": Node("b", "10.2.1.2", "nvidia")}
    plan = build_plan(["a", "b"], hosts, nodes=reg)
    link = plan["links"][0]
    assert link["link_id"] == "L0" and link["a"]["node"] == "a" and link["b"]["ip"] == "10.100.152.2"
    assert plan["per_node"] is plan["nodes"]
    b = plan["per_node"]["b"]
    assert b["iface_ips"] == b["links"] and b["netplan_yaml"] == b["netplan"]
    assert b["changes"] == [f"write {NETPLAN_FILE}",
                            "enp1s0f1np1: set 10.100.152.2/24 (replaces the 169.254.x.x link-local address)",
                            f"move 99-nvidia-sync-cluster.yaml to {DISABLED_DIR}",
                            "registry: cluster_ip (unset) → 10.100.152.2"]
    assert plan["nodes"]["a"]["changes"] == [f"write {NETPLAN_FILE}", "enp1s0f1np1: keep 10.100.152.1/24"]
    # หน้าเว็บส่ง kind ที่ inspect ตอบกลับมาเป็น topology — ต้องรับได้
    assert build_plan(["a", "b"], hosts, topology="direct-2")["ok"]
    assert build_plan(["a", "b"], hosts, topology="switch-2")["topology"] == "switch-2"


# ── API ─────────────────────────────────────────────────────────────────────────
@pytest.fixture
def web():
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app, state

    register("a", "10.2.1.1")
    register("b", "10.2.1.2")
    state.STORE.set_node("a", {"host": spark("10.2.1.1"), "models": [], "summary": {"total": 0, "running": 0}})
    state.STORE.set_node("b", {"host": spark("10.2.1.2", nvidia_sync=True), "models": [], "summary": {"total": 0, "running": 0}})
    return TestClient(create_app())


def test_api_inspect_and_plan_shapes(web):
    got = web.post("/api/cluster/inspect", json={"nodes": ["a", "b"]})
    assert got.status_code == 200, got.text
    body = got.json()
    assert set(body) == {"nodes", "topology", "findings", "ok"}
    assert body["topology"]["kind"] == "direct-2" and body["nodes"]["b"]["nvidia_sync"] is True
    assert body["nodes"]["a"]["fabric"]["ports"][0]["interfaces"][1]["carrier"] is True and body["findings"][0]["text"]
    assert all("text" in f for f in body["findings"])

    plan = web.post("/api/cluster/plan", json={"nodes": ["a", "b"], "base_subnet": "10.77.0.0/24"})
    assert plan.status_code == 200
    body = plan.json()
    assert body["ok"] and body["links"][0]["subnet"] == "10.77.0.0/24"
    assert body["nodes"]["a"]["cluster_ip"] == "10.77.0.1" and "addresses: [10.77.0.1/24]" in body["nodes"]["a"]["netplan"]
    assert body["registry"]["b"]["cluster_links"][0]["peer_ip"] == "10.77.0.1"

    assert web.post("/api/cluster/plan", json={"nodes": ["a"]}).status_code == 400
    assert web.post("/api/cluster/plan", json={"nodes": ["a", "zzz"]}).status_code == 404
    assert web.post("/api/cluster/plan", json={"nodes": ["a", "b"], "base_subnet": "junk"}).status_code == 400
    bad = web.post("/api/cluster/plan", json={"nodes": ["a", "b"], "topology": "ring"}).json()
    assert bad["ok"] is False and "ring is three machines" in bad["reason"]


def test_api_apply_runs_as_a_job_and_never_echoes_the_password(web, monkeypatch):
    import lmds.nodes

    passwords = {"a": "pw-a", "b": "pw-b"}
    fleet = FakeFleet(passwords)
    monkeypatch.setattr(lmds.nodes, "run", fleet)
    monkeypatch.setattr("lmds.nodes.netplan.VERIFY_PAUSE_S", 0.0)
    plan = web.post("/api/cluster/plan", json={"nodes": ["a", "b"]}).json()

    missing = web.post("/api/cluster/apply", json={"plan": plan, "passwords": {"a": "pw-a"}})
    assert missing.status_code == 400 and "b" in missing.json()["detail"]
    assert web.post("/api/cluster/apply", json={"plan": {}, "passwords": passwords}).status_code == 400

    done = web.post("/api/cluster/apply", json={"plan": plan, "passwords": passwords, "wait": True,
                                                "speed_test": False})
    assert done.status_code == 200, done.text
    body = done.json()
    assert set(body) == {"job", "result"} and body["job"]["running"] is False and body["job"]["exit_code"] == 0
    assert body["result"]["ok"] and body["result"]["registry"]["b"]["cluster_ip"] == "10.100.152.2"
    assert "pw-" not in done.text
    # หน้าเว็บตามงานผ่าน /api/jobs/{id} และอ่าน output เป็นติ๊กต่อเครื่อง — บรรทัดต้องอยู่ในภาษาที่มันรู้จัก
    job = web.get(f"/api/jobs/{body['job']['id']}")
    assert job.status_code == 200 and job.json()["running"] is False and job.json()["exit_code"] == 0
    lines = job.json()["output"].splitlines()
    for expected in ("[a] write netplan: ok", "[a] netplan apply: ok", "[a] verify addresses: ok",
                     "[a] ping b (10.100.152.2) via enp1s0f1np1: ok", "[b] ping a (10.100.152.1) via enp1s0f1np1: ok",
                     "[a] registry: cluster_ip 10.100.152.1 on enp1s0f1np1 saved"):
        assert any(l.startswith(expected) for l in lines), (expected, lines)
    assert any(l.startswith("[a] pair SSH — ") and l.endswith(": paired") for l in lines), lines
    assert "[b] netplan apply: ok — disabled /etc/netplan/99-nvidia-sync-cluster.yaml" in lines
    assert "pw-" not in job.text
    status = web.get(f"/api/cluster/apply/{body['job']['id']}")
    assert status.status_code == 200 and status.json()["running"] is False and status.json()["result"]["ok"]
    assert web.get("/api/cluster/apply/nope").status_code == 404
    assert find("a").cluster_ip == "10.100.152.1"

    # แบบเบื้องหลัง: ตอบ job ทันที แล้ว poll จนจบ
    started = web.post("/api/cluster/apply", json={"plan": plan, "passwords": passwords, "speed_test": False})
    assert started.status_code == 200 and started.json()["job"]["running"] is True
    import time

    for _ in range(200):
        polled = web.get(f"/api/jobs/{started.json()['job']['id']}").json()
        if not polled["running"]:
            break
        time.sleep(0.02)
    assert not polled["running"] and polled["exit_code"] == 0 and "done — cluster IPs saved" in polled["output"]
    assert "pw-" not in json.dumps(polled)


def test_api_apply_reports_a_failed_node_with_exit_1_and_a_rollback_line(web, monkeypatch):
    import lmds.nodes

    passwords = {"a": "pw-a", "b": "pw-b"}
    monkeypatch.setattr(lmds.nodes, "run", FakeFleet(passwords, lose_carrier_on="enp1s0f1np1"))
    monkeypatch.setattr("lmds.nodes.netplan.VERIFY_PAUSE_S", 0.0)
    plan = web.post("/api/cluster/plan", json={"nodes": ["a", "b"]}).json()
    body = web.post("/api/cluster/apply", json={"plan": plan, "passwords": passwords, "wait": True}).json()
    assert body["job"]["exit_code"] == 1 and body["result"]["nodes"]["a"]["rolled_back"]
    output = body["job"]["output"]
    assert "[a] verify addresses: failed — enp1s0f1np1 lost carrier" in output
    assert "[a] rolled back to the previous netplan: ok" in output and "[b] netplan apply" not in output
    assert find("a").cluster_ip == ""


def test_api_remove_net(web, monkeypatch):
    import lmds.nodes

    fleet = FakeFleet({"a": "pw"})
    monkeypatch.setattr(lmds.nodes, "run", fleet)
    from lmds.nodes import update

    update("a", cluster_ip="10.100.152.1", cluster_iface="enp1s0f1np1")
    assert web.post("/api/cluster/remove-net", json={"node": "zzz", "password": "x"}).status_code == 404
    assert web.post("/api/cluster/remove-net", json={"node": "a"}).status_code == 400
    got = web.post("/api/cluster/remove-net", json={"node": "a", "password": "pw"})
    assert got.status_code == 200, got.text
    assert got.json()["ok"] and got.json()["removed"] and got.json()["node"] == "a"
    assert got.json()["detail"] == f"netplan file moved to {DISABLED_DIR} on a, ports released, registry cleared"
    assert find("a").cluster_ip == "" and "pw" not in got.text.replace("password", "")
