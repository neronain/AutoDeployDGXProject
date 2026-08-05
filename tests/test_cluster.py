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
