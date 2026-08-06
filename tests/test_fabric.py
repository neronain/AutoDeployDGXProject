"""fabric ที่ตั้งผิดแล้ว "ก็ยังรันได้" — กลุ่มปัญหาที่ไล่สาเหตุยากที่สุด

ชื่อ interface/HCA และค่าพื้นฐานด้านล่างมาจาก DGX Spark payload; test cases หลายชุดดัดแปลง
ค่าพื้นฐานอย่างจงใจเพื่อครอบคลุม invalid/stale/mixed-speed input ไม่ใช่ hardware captures:

    rocep1s0f0  ==> enp1s0f0np0   (Up, 200000, 10.100.16.1/24)
    rocep1s0f1  ==> enp1s0f1np1   (Down)
    roceP2p1s0f0 ==> enP2p1s0f0np0 (Up, 200000, 10.100.17.1/24)
    roceP2p1s0f1 ==> enP2p1s0f1np1 (Down)
    enP7s7       10G RJ-45, 192.168.1.75/24
"""

from __future__ import annotations

import io

from rich.console import Console

from lmds.nodes import (
    active_fabric_links,
    fabric_warnings,
    is_mesh,
    nccl_ib_hca,
    oob_link,
)


def link(iface, *, ip="", prefix=24, speed=200, state="up", connectx=True,
         rdma_device=""):
    return {
        "iface": iface, "ip": ip, "prefix": prefix if ip else None,
        "link_local": ip.startswith("169.254."), "speed_gbps": speed,
        "rdma_device": rdma_device, "driver": "mlx5_core" if connectx else "nvethernet",
        "state": state, "connectx": connectx, "rdma": connectx,
    }


def host(*links):
    return {"fabric": {"links": list(links), "best_gbps": 200, "tier": "rdma"}}


OOB = link("enP7s7", ip="192.168.1.75", speed=10, connectx=False)

# สายเส้นเดียวเสียบพอร์ตเดียว = RoCE คู่แฝดสองตัวขึ้น (ของจริงบน spark1)
TWO_PORT = host(
    link("enp1s0f0np0", ip="10.100.16.1", rdma_device="rocep1s0f0"),
    link("enP2p1s0f0np0", ip="10.100.17.1", rdma_device="roceP2p1s0f0"),
    link("enp1s0f1np1", state="down", speed=None),
    link("enP2p1s0f1np1", state="down", speed=None),
    OOB,
)

# ขึ้นครบสี่ = local mesh candidate; ปลายสาย/topology ยังไม่ทราบจาก payload เครื่องเดียว
MESH = host(
    link("enp1s0f0np0", ip="10.100.16.1", rdma_device="rocep1s0f0"),
    link("enP2p1s0f0np0", ip="10.100.17.1", rdma_device="roceP2p1s0f0"),
    link("enp1s0f1np1", ip="10.100.18.1", rdma_device="rocep1s0f1"),
    link("enP2p1s0f1np1", ip="10.100.19.1", rdma_device="roceP2p1s0f1"),
    OOB,
)


def kinds(payload) -> set[str]:
    return {w["kind"] for w in fabric_warnings(payload)}


# ── การนับลิงก์ที่ใช้งานอยู่ ────────────────────────────────────────────────────

def test_only_links_that_are_actually_up_count():
    """สายที่ไม่ได้เสียบยังโผล่ใน /sys/class/net ครบทุกตัว — นับรวมแล้วเข้าใจผิดว่าเป็น mesh"""
    assert [l["iface"] for l in active_fabric_links(TWO_PORT)] == [
        "enp1s0f0np0", "enP2p1s0f0np0",
    ]


def test_oob_interface_is_not_counted_as_fabric():
    """RJ-45 10G ไม่ใช่ ConnectX — นับรวมแล้วเลข active เพี้ยน"""
    assert all(l["iface"] != "enP7s7" for l in active_fabric_links(MESH))


def test_two_active_links_is_not_mesh():
    assert not is_mesh(TWO_PORT)


def test_four_active_links_is_mesh_candidate():
    assert is_mesh(MESH)
    assert "mesh-topology-unverified" in kinds(MESH)


# ── NCCL_IB_HCA ──────────────────────────────────────────────────────────────

def test_hca_lists_every_active_twin():
    """สายเส้นเดียวมีคู่แฝดสองตัว — บอก NCCL ตัวเดียวคือได้แบนด์วิดท์ครึ่งเดียว"""
    assert nccl_ib_hca(TWO_PORT) == "=rocep1s0f0,roceP2p1s0f0"


def test_hca_covers_all_four_in_mesh():
    assert nccl_ib_hca(MESH) == (
        "=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"
    )


def test_hca_skips_links_without_a_roce_device():
    """การ์ดที่ไม่มี RoCE คู่กันใส่ลง NCCL_IB_HCA ไม่ได้ — จะได้ชื่อว่างคั่นจุลภาค"""
    payload = host(
        link("enp1s0f0np0", ip="10.100.16.1", rdma_device="rocep1s0f0"),
        link("enp2s0", ip="10.100.17.1", rdma_device=""),
    )
    assert nccl_ib_hca(payload) == "=rocep1s0f0"


def test_hca_uses_only_equal_fastest_rails_and_exact_match_prefix():
    payload = host(
        link("enp1s0f0np0", speed=200, rdma_device="mlx5_1"),
        link("enp2s0f0np0", speed=25, rdma_device="mlx5_10"),
    )
    assert nccl_ib_hca(payload) == "=mlx5_1"


def test_hca_matches_controller_when_only_positive_slow_rail_exists():
    payload = host(link("enp1s0f0np0", speed=10, rdma_device="mlx5_4"))
    assert nccl_ib_hca(payload) == "=mlx5_4"


def test_hca_is_empty_when_nothing_is_up():
    assert nccl_ib_hca(host(link("enp1s0f0np0", state="down", speed=None))) == ""


# ── กับดักที่พังเงียบ ────────────────────────────────────────────────────────

def test_link_up_without_ip_is_reported():
    """สายเสียบอยู่ ลิงก์ขึ้น 200G แต่ไม่มี IP — NCCL ใช้เส้นนี้ไม่ได้เลย

    ของเดิมกรองเส้นที่ไม่มี IP ทิ้งเงียบ ๆ ใน fabric_links() ผู้ใช้จึงเห็นแค่ว่า
    "มีสายน้อยกว่าที่เสียบไว้" โดยไม่มีอะไรบอกว่าทำไม
    """
    payload = host(
        link("enp1s0f0np0", ip="10.100.16.1", rdma_device="rocep1s0f0"),
        link("enP2p1s0f0np0", rdma_device="roceP2p1s0f0"),  # ขึ้นแต่ไม่มี IP
    )
    assert "link-without-ip" in kinds(payload)
    assert fabric_warnings(payload)[0]["ifaces"] == ["enP2p1s0f0np0"]


def test_twins_sharing_one_subnet_is_reported():
    """คู่แฝดต้องคนละวง — วงเดียวกันทำให้ routing สับสน แพ็กเก็ตออกผิดเส้น

    ที่มา: eugr/spark-vllm-docker@42b3a793 docs/NETWORKING.md เขียนตัวหนาไว้ว่า
    "DO NOT use the same subnet on both twins"
    """
    payload = host(
        link("enp1s0f0np0", ip="10.100.16.1", rdma_device="rocep1s0f0"),
        link("enP2p1s0f0np0", ip="10.100.16.2", rdma_device="roceP2p1s0f0"),
    )
    warning = next(w for w in fabric_warnings(payload) if w["kind"] == "shared-subnet")
    assert warning["networks"] == ["10.100.16.0/24"]
    assert warning["ifaces"] == ["enP2p1s0f0np0", "enp1s0f0np0"]


def test_correct_wiring_has_no_subnet_warning():
    assert "shared-subnet" not in kinds(TWO_PORT)
    assert "shared-subnet" not in kinds(MESH)


def test_link_without_roce_device_is_reported():
    payload = host(link("enp2s0", ip="10.100.16.1", rdma_device=""))
    assert "no-rdma-device" in kinds(payload)


def test_healthy_two_port_setup_has_no_warnings():
    """เครื่องที่ตั้งถูกต้องต้องเงียบสนิท — คำเตือนที่ขึ้นตลอดเวลาไม่มีใครอ่าน"""
    assert kinds(TWO_PORT) == set()


# ── mesh: เส้น out-of-band ────────────────────────────────────────────────────

def test_mesh_needs_an_out_of_band_link():
    """A local mesh candidate without even a local OOB candidate must be flagged.

    This does not prove that an OOB interface found locally is reachable by every peer.
    """
    without_oob = host(*[l for l in MESH["fabric"]["links"] if l["iface"] != "enP7s7"])
    assert "mesh-without-oob" in kinds(without_oob)


def test_mesh_with_rj45_is_fine():
    assert "mesh-without-oob" not in kinds(MESH)
    assert oob_link(MESH)["iface"] == "enP7s7"


def test_wireless_out_of_band_is_flagged_but_allowed():
    wifi = link("wlP9s9", ip="192.168.1.90", speed=1, connectx=False)
    payload = host(*[l for l in MESH["fabric"]["links"] if l["iface"] != "enP7s7"], wifi)
    assert "mesh-oob-wireless" in kinds(payload)
    assert "mesh-without-oob" not in kinds(payload)


def test_wired_out_of_band_beats_wireless():
    wifi = link("wlP9s9", ip="192.168.1.90", speed=1, connectx=False)
    payload = host(*MESH["fabric"]["links"], wifi)
    assert oob_link(payload)["iface"] == "enP7s7"
    assert "mesh-oob-wireless" not in kinds(payload)


def test_link_local_is_not_an_out_of_band_candidate():
    """169.254.x.x = ตั้ง IP เองเพราะไม่มี DHCP — ยิงข้ามเครื่องไม่ถึง"""
    self_assigned = link("enP7s7", ip="169.254.3.4", speed=10, connectx=False)
    payload = host(*[l for l in MESH["fabric"]["links"] if l["iface"] != "enP7s7"], self_assigned)
    assert oob_link(payload) is None
    assert "mesh-without-oob" in kinds(payload)


def test_non_mesh_host_is_not_asked_for_out_of_band():
    """คลัสเตอร์ปกติ (2 เครื่อง / ผ่านสวิตช์) คุยกันบน QSFP ได้ ไม่ต้องมี OOB"""
    without_oob = host(*[l for l in TWO_PORT["fabric"]["links"] if l["iface"] != "enP7s7"])
    assert "mesh-without-oob" not in kinds(without_oob)


# ── ทนต่อ payload ที่ไม่ครบ ──────────────────────────────────────────────────

def test_empty_payload_does_not_crash():
    assert fabric_warnings({}) == []
    assert not is_mesh({})
    assert nccl_ib_hca({}) == ""
    assert oob_link({}) is None


def test_payload_from_an_older_node_without_new_fields():
    """node เวอร์ชันเก่าส่ง link มาไม่มี mtu/rdma_device — ต้องไม่ระเบิด"""
    old = {"fabric": {"links": [
        {"iface": "enp1s0f0np0", "ip": "10.100.16.1", "prefix": 24,
         "speed_gbps": 200, "state": "up", "connectx": True, "rdma": True},
    ]}}
    assert "no-rdma-device" in kinds(old)
    assert nccl_ib_hca(old) == ""


def test_untrusted_remote_payload_is_normalized_and_never_raises():
    hostile = {"gpus": [{"name": "GB10"}], "fabric": {"links": [
        {"iface": "enp1s0[/dim][bold red]boom", "ip": ["not", "text"],
         "speed_gbps": "200", "state": {"bad": True}, "connectx": True,
         "rdma_device": "[red]mlx5_0[/red]"},
        None,
        "not-a-link",
        {"iface": "enp2s0", "ip": "10.0.0.1", "prefix": 24,
         "speed_gbps": float("nan"), "state": "up", "connectx": True},
        {"iface": "lo", "ip": "127.0.0.1", "prefix": 8,
         "speed_gbps": 200, "state": "up", "connectx": True},
    ]}}
    assert active_fabric_links(hostile) == []
    assert fabric_warnings(hostile) == []
    assert nccl_ib_hca(hostile) == ""


def test_cli_escapes_remote_rich_markup(monkeypatch):
    from lmds.cli import main

    stream = io.StringIO()
    monkeypatch.setattr(main, "console", Console(file=stream, color_system=None, width=200))
    main._render_fabric_warnings("[red]remote[/red][blink]", MESH)
    rendered = stream.getvalue()
    assert "[red]remote[/red][blink]" in rendered
    assert "NCCL_IB_HCA='=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1'" in rendered
