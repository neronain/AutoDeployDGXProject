"""cluster fabric — ตรวจสายเร็ว + จับคู่เครื่องที่ stacked ด้วยกันได้"""

from __future__ import annotations

import pytest

from lmds.nodes import cluster as cl


def host(gpu="NVIDIA GB10", count=1, gbps=200, tier="rdma", ip="10.10.0.1", arch="aarch64",
         profile="dgx_spark"):
    links = []
    if gbps:
        links.append({"iface": "enp1s0f0np0", "ip": ip, "speed_gbps": gbps,
                      "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": tier == "rdma"})
    return {
        "arch": arch,
        "profile": profile,
        "gpus": [{"name": gpu} for _ in range(count)],
        "fabric": {"links": links, "rdma_devices": ["mlx5_0"] if tier == "rdma" else [],
                   "best_gbps": gbps, "tier": tier, "cluster_capable": bool(gbps and gbps >= 25)},
    }


def test_stack_ready_needs_gpu_and_fast_link():
    assert cl.stack_ready(host())
    assert not cl.stack_ready(host(count=0))
    assert not cl.stack_ready(host(gbps=1, tier="basic"))


def test_slow_link_is_not_stackable():
    """1G ต่อถึงกันได้ก็จริง แต่ stacked จะช้ากว่ารันแยก — ต้องไม่ถูกเสนอ"""
    assert not cl.stack_ready(host(gbps=10, tier="basic"))


def test_suggest_cluster_ip_picks_fastest_link_with_an_address():
    payload = host(gbps=200)
    payload["fabric"]["links"].append(
        {"iface": "eno1", "ip": "192.168.1.20", "speed_gbps": 1000 // 1000,
         "driver": "e1000e", "state": "up", "connectx": False, "rdma": False})
    assert cl.suggest_cluster_ip(payload) == "10.10.0.1"


def test_suggest_is_empty_when_no_link_has_an_address():
    payload = host()
    payload["fabric"]["links"][0]["ip"] = ""
    assert cl.suggest_cluster_ip(payload) == ""


@pytest.mark.parametrize(
    "cluster_ip, state",
    [("", "unset"), ("10.10.0.1", "ok"), ("10.99.0.1", "mismatch")],
)
def test_check_cluster_ip_states(cluster_ip, state):
    assert cl.check_cluster_ip(host(), cluster_ip)["state"] == state


def test_check_cluster_ip_flags_a_slow_interface():
    payload = host(gbps=10, tier="basic", ip="192.168.1.5")
    check = cl.check_cluster_ip(payload, "192.168.1.5")
    assert check["state"] == "slow"
    assert check["iface"] == "enp1s0f0np0"


def test_two_matching_machines_form_a_group():
    machines = [
        {"name": "spark1", "host": host(ip="10.10.0.1"), "cluster_ip": "10.10.0.1"},
        {"name": "spark2", "host": host(ip="10.10.0.2"), "cluster_ip": "10.10.0.2"},
    ]
    (group,) = cl.cluster_groups(machines)
    assert group["ready"]
    assert group["world_size"] == 2
    assert group["rdma"]
    assert [m["name"] for m in group["members"]] == ["spark1", "spark2"]


def test_mixed_hardware_never_groups():
    """GPU คนละรุ่นห้ามอยู่กลุ่มเดียวกัน — NCCL แบ่งงานเท่ากันทุก rank"""
    machines = [
        {"name": "spark", "host": host(), "cluster_ip": "10.10.0.1"},
        {"name": "rtx", "host": host(gpu="NVIDIA GeForce RTX 5090", arch="x86_64",
                                     profile="rtx", ip="10.10.0.2"), "cluster_ip": "10.10.0.2"},
    ]
    assert cl.cluster_groups(machines) == []


def test_different_gpu_count_never_groups():
    machines = [
        {"name": "a", "host": host(count=1), "cluster_ip": "10.10.0.1"},
        {"name": "b", "host": host(count=2, ip="10.10.0.2"), "cluster_ip": "10.10.0.2"},
    ]
    assert cl.cluster_groups(machines) == []


def test_group_reports_the_slowest_member_speed():
    """NCCL รอ rank ที่ช้าที่สุด — ตัวเลขที่แสดงต้องเป็นของเครื่องที่ช้าสุด ไม่ใช่เร็วสุด"""
    machines = [
        {"name": "a", "host": host(gbps=200), "cluster_ip": "10.10.0.1"},
        {"name": "b", "host": host(gbps=100, ip="10.10.0.2"), "cluster_ip": "10.10.0.2"},
    ]
    (group,) = cl.cluster_groups(machines)
    assert group["link_gbps"] == 100


def test_missing_cluster_ip_blocks_the_group():
    machines = [
        {"name": "a", "host": host(), "cluster_ip": "10.10.0.1"},
        {"name": "b", "host": host(ip="10.10.0.2"), "cluster_ip": ""},
    ]
    (group,) = cl.cluster_groups(machines)
    assert not group["ready"]
    assert group["blockers"] == [{"kind": "missing-ip", "names": ["b"]}]
    # ต้องเสนอค่าที่ตรวจพบให้ ไม่ใช่ปล่อยให้ผู้ใช้ไปหาเอง
    assert group["members"][1]["suggested_ip"] == "10.10.0.2"


def test_duplicate_cluster_ip_blocks_the_group():
    machines = [
        {"name": "a", "host": host(), "cluster_ip": "10.10.0.1"},
        {"name": "b", "host": host(ip="10.10.0.1"), "cluster_ip": "10.10.0.1"},
    ]
    (group,) = cl.cluster_groups(machines)
    assert not group["ready"]
    assert group["blockers"][0]["kind"] == "duplicate-ip"


def test_a_single_machine_is_not_a_group():
    assert cl.cluster_groups([{"name": "solo", "host": host(), "cluster_ip": "10.10.0.1"}]) == []


def test_blockers_carry_codes_not_sentences():
    """CLI เป็นไทย หน้าเว็บเป็นอังกฤษ — โมดูลนี้ห้ามผูกภาษาไว้ใน blockers"""
    machines = [
        {"name": "a", "host": host(), "cluster_ip": ""},
        {"name": "b", "host": host(ip="10.10.0.2"), "cluster_ip": ""},
    ]
    (group,) = cl.cluster_groups(machines)
    assert all(set(b) == {"kind", "names"} for b in group["blockers"])


# ── เคสจากเครื่องจริง (DGX Spark, ConnectX 4 พอร์ต) ──────────────────────────
def multi_port_host():
    """DGX Spark จริง: ConnectX 200G สี่เส้น สองเส้นยังไม่ได้ตั้ง IP (ได้ 169.254 มาเอง)"""
    links = [
        {"iface": "enP2p1s0f0np0", "ip": "169.254.31.110", "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
        {"iface": "enP2p1s0f1np1", "ip": "10.100.153.1", "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
        {"iface": "enP7s7", "ip": "192.168.79.9", "speed_gbps": 1,
         "driver": "r8127", "state": "up", "connectx": False, "rdma": False},
        {"iface": "enp1s0f1np1", "ip": "10.100.152.1", "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
    ]
    return {"arch": "aarch64", "profile": "dgx_spark", "gpus": [{"name": "NVIDIA GB10"}],
            "fabric": {"links": links, "rdma_devices": ["rocep1s0f1"], "best_gbps": 200,
                       "tier": "rdma", "cluster_capable": True}}


def test_link_local_is_never_suggested():
    """169.254.x.x ลิงก์ขึ้นและเร็ว 200G เหมือนกัน แต่ยิง NCCL ข้ามเครื่องไม่ถึง"""
    assert cl.suggest_cluster_ip(multi_port_host()) in {"10.100.152.1", "10.100.153.1"}


def test_link_local_ip_is_flagged():
    check = cl.check_cluster_ip(multi_port_host(), "169.254.31.110")
    assert check["state"] == "link-local"


def test_management_ip_on_a_slow_port_is_flagged():
    """เผลอกรอก IP ที่ใช้ SSH (สาย 1G) เป็นกรณีที่เกิดง่ายที่สุด"""
    assert cl.check_cluster_ip(multi_port_host(), "192.168.79.9")["state"] == "slow"


def test_link_local_filtering_does_not_need_the_node_flag():
    """node เวอร์ชันเก่ายังไม่ส่งฟิลด์ link_local มา — hub ต้องตัดสินจาก IP เองได้"""
    host = multi_port_host()
    for link in host["fabric"]["links"]:
        link.pop("link_local", None)
    assert not cl.suggest_cluster_ip(host).startswith("169.254.")


def two_fabric_host(a, b):
    """DGX Spark จริงมี fabric สองวง — ปล่อยให้แต่ละเครื่องเลือกเองอาจได้คนละวง"""
    links = [
        {"iface": "enp1s0f1np1", "ip": a, "prefix": 24, "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
        {"iface": "enP2p1s0f1np1", "ip": b, "prefix": 24, "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
    ]
    return {"arch": "aarch64", "profile": "dgx_spark", "gpus": [{"name": "NVIDIA GB10"}],
            "fabric": {"links": links, "rdma_devices": ["r"], "best_gbps": 200,
                       "tier": "rdma", "cluster_capable": True}}


def test_suggestion_uses_a_fabric_both_machines_share():
    machines = [
        {"name": "head", "host": two_fabric_host("10.100.152.1", "10.100.153.1"), "cluster_ip": ""},
        {"name": "worker", "host": two_fabric_host("10.100.152.2", "10.100.153.2"), "cluster_ip": ""},
    ]
    (group,) = cl.cluster_groups(machines)
    suggested = {m["name"]: m["suggested_ip"] for m in group["members"]}
    # ต้องอยู่วงเดียวกันทั้งคู่ ไม่ใช่ต่างคนต่างเลือก
    assert {ip.rsplit(".", 1)[0] for ip in suggested.values()} == {"10.100.152"}
    assert group["fabric_network"] == "10.100.152.0/24"


def test_cluster_ips_on_different_subnets_are_blocked():
    """ตั้งครบทั้งคู่แต่คนละวง = ต่อกันไม่ติด ทั้งที่แต่ละเครื่องดูถูกหมด"""
    machines = [
        {"name": "head", "host": two_fabric_host("10.100.152.1", "10.100.153.1"),
         "cluster_ip": "10.100.152.1"},
        {"name": "worker", "host": two_fabric_host("10.100.152.2", "10.100.153.2"),
         "cluster_ip": "10.100.153.2"},
    ]
    (group,) = cl.cluster_groups(machines)
    assert not group["ready"]
    assert [b["kind"] for b in group["blockers"]] == ["split-fabric"]


def test_shared_fabric_is_stable_across_calls():
    """สองวงเร็วเท่ากัน — ต้องเลือกเหมือนเดิมทุกครั้ง ไม่งั้นค่าที่เสนอเปลี่ยนไปมา"""
    machines = [
        {"name": "head", "host": two_fabric_host("10.100.153.1", "10.100.152.1"), "cluster_ip": ""},
        {"name": "worker", "host": two_fabric_host("10.100.152.2", "10.100.153.2"), "cluster_ip": ""},
    ]
    picks = {cl.shared_fabric([{"name": m["name"], "host": m["host"]} for m in machines])[0]
             for _ in range(3)}
    assert picks == {"10.100.152.0/24"}


def test_shared_fabric_does_not_crash_on_ipv6_telemetry():
    """agent อาจรายงาน IPv6 แม้ cluster.env ยังรับ IPv4; หน้าคลัสเตอร์ต้องไม่ล้มทั้ง endpoint"""
    members = [
        {"name": "a", "host": host(ip="2001:db8:1::1")},
        {"name": "b", "host": host(ip="2001:db8:1::2")},
    ]
    for member in members:
        member["host"]["fabric"]["links"][0]["prefix"] = 64
    network, addresses = cl.shared_fabric(members)
    assert network == "2001:db8:1::/64"
    assert addresses == {"a": "2001:db8:1::1", "b": "2001:db8:1::2"}


# ── ขยายเป็น 3–4 เครื่อง ────────────────────────────────────────────────────
def test_four_machines_form_one_group():
    machines = [
        {"name": f"spark{i}", "host": host(ip=f"10.100.152.{i}"), "cluster_ip": f"10.100.152.{i}"}
        for i in range(1, 5)
    ]
    (group,) = cl.cluster_groups(machines)
    assert group["ready"] and group["world_size"] == 4
    assert group["parallelism"]["kind"] == "tensor-parallel"


def test_three_machines_need_pipeline_parallel():
    """TP=3 หาร attention head ของโมเดลส่วนใหญ่ไม่ลงตัว (Llama 3.3 70B = 64 heads)"""
    machines = [
        {"name": f"spark{i}", "host": host(ip=f"10.100.152.{i}"), "cluster_ip": f"10.100.152.{i}"}
        for i in range(1, 4)
    ]
    (group,) = cl.cluster_groups(machines)
    note = group["parallelism"]
    assert note["kind"] == "pipeline-parallel"
    assert note["largest_tp"] == 2  # TP=2 แล้วเหลือเครื่องที่สามไว้ทำ pipeline


@pytest.mark.parametrize("size, kind", [(2, "tensor-parallel"), (4, "tensor-parallel"),
                                        (8, "tensor-parallel"), (3, "pipeline-parallel"),
                                        (6, "pipeline-parallel")])
def test_parallelism_note_by_size(size, kind):
    assert cl.parallelism_note(size)["kind"] == kind


def test_hub_ip_follows_the_subnet_peers_actually_use():
    """DGX Spark มี fabric สองวงที่เร็วเท่ากัน และทุกเครื่องมักมีขาอยู่ทั้งสองวง

    เคสจริงบนเครื่องสองเครื่อง: s2 ตั้ง 10.100.16.2 ไว้ แต่ hub ถูกเสนอ 10.100.17.1
    แล้วระบบก็ฟ้อง split-fabric ที่ **ตัวเองสร้างขึ้นมาเอง** — ผู้ใช้ไม่ได้ทำอะไรผิดเลย
    """
    from lmds.nodes import suggest_cluster_ip

    host = {"fabric": {"links": [
        {"iface": "enp1s0f0np0", "ip": "10.100.16.1", "prefix": 24, "speed_gbps": 200,
         "state": "up", "connectx": True},
        {"iface": "enP2p1s0f0np0", "ip": "10.100.17.1", "prefix": 24, "speed_gbps": 200,
         "state": "up", "connectx": True},
    ]}}
    assert suggest_cluster_ip(host, ["10.100.16.2"]) == "10.100.16.1"
    assert suggest_cluster_ip(host, ["10.100.17.2"]) == "10.100.17.1"


def test_hub_ip_without_peers_still_picks_the_fastest():
    """ยังไม่มีเครื่องอื่นในทะเบียน = ไม่มีอะไรให้ตาม — พฤติกรรมเดิมต้องไม่เปลี่ยน"""
    from lmds.nodes import suggest_cluster_ip

    host = {"fabric": {"links": [
        {"iface": "eth0", "ip": "192.168.1.5", "prefix": 24, "speed_gbps": 25,
         "state": "up", "connectx": False},
        {"iface": "enp1s0f0np0", "ip": "10.100.16.1", "prefix": 24, "speed_gbps": 200,
         "state": "up", "connectx": True},
    ]}}
    assert suggest_cluster_ip(host) == "10.100.16.1"
    assert suggest_cluster_ip(host, []) == "10.100.16.1"


def test_peer_ip_on_a_subnet_the_hub_cannot_reach_is_ignored():
    """peer ตั้ง IP ที่ hub ไม่มีขาอยู่ — เสนอตามความเร็วต่อ ไม่ใช่คืนค่าว่าง"""
    from lmds.nodes import suggest_cluster_ip

    host = {"fabric": {"links": [
        {"iface": "enp1s0f0np0", "ip": "10.100.16.1", "prefix": 24, "speed_gbps": 200,
         "state": "up", "connectx": True},
    ]}}
    assert suggest_cluster_ip(host, ["192.168.99.2"]) == "10.100.16.1"
