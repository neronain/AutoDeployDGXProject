"""ตั้ง cluster IP แล้วกด Save ต้องได้กลุ่มที่พร้อม deploy — ไม่ใช่เงียบ

เคสจริง 2026-08-31 (ผู้ใช้รายงาน): "พอผมกดตั้ง ip ที่เครื่อง A และ B พอกด save
แล้วไม่มีคำสั่ง deploy cluster"

ที่เกิดขึ้นจริงบนฟลีต: spark-head + spark-worker ตั้ง cluster IP ครบแล้ว
(10.100.152.1/.2, 200G ทั้งคู่, ขึ้นว่า "stacked ได้" ทั้งคู่) แต่กลุ่มที่ระบบเสนอกลับเป็น
msi-1 + msi-2 ที่ **ยังไม่ได้ตั้ง IP เลย** ส่วนคู่ที่ตั้งแล้วถูกเขี่ยออกเป็น
"ฮาร์ดแวร์ตรงกัน แต่ไม่มี subnet ร่วมกับกลุ่มนี้"

สาเหตุ: ทั้งสี่เครื่องเป็น GB10 เหมือนกันหมดจึงตกอยู่ถังเดียวกัน แล้ว connected_subset()
คืน **กลุ่มเดียวที่ใหญ่ที่สุด** โยนที่เหลือทิ้ง · สองคู่มีสมาชิกเท่ากัน ตัวตัดสินจึงไปตกที่
เลขวง ซึ่งไม่เกี่ยวอะไรกับว่าใครตั้ง IP ไว้แล้ว
"""

from pathlib import Path as _Path

from lmds.nodes.cluster import cluster_groups


def _spark(ip_on_fast: str, mgmt: str):
    """host payload ของ DGX Spark หนึ่งเครื่อง — มีขา 200G กับขาบริหารจัดการ"""
    links = [
        {"iface": "enp1s0f1np1", "ip": ip_on_fast, "speed_gbps": 200,
         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True},
        {"iface": "enP7s7", "ip": mgmt, "speed_gbps": 1,
         "driver": "r8127", "state": "up", "connectx": False, "rdma": False},
    ]
    return {"arch": "aarch64", "profile": "dgx_spark",
            "gpus": [{"name": "NVIDIA GB10"}],
            "fabric": {"links": links, "rdma_devices": ["rocep1s0f1"],
                       "best_gbps": 200, "tier": "rdma", "cluster_capable": True}}


def _fleet():
    return [
        # ไซต์ Neronain — ตั้ง IP ครบแล้วทั้งคู่
        {"name": "spark-head", "site": "Neronain", "cluster_ip": "10.100.152.1",
         "host": _spark("10.100.152.1", "10.2.1.195")},
        {"name": "spark-worker", "site": "Neronain", "cluster_ip": "10.100.152.2",
         "host": _spark("10.100.152.2", "10.2.1.194")},
        # ไซต์ TKC — ยังไม่ได้ตั้ง IP
        {"name": "msi-1", "site": "TKC", "cluster_ip": "",
         "host": _spark("10.55.0.1", "10.2.2.11")},
        {"name": "msi-2", "site": "TKC", "cluster_ip": "",
         "host": _spark("10.55.0.2", "10.2.2.12")},
    ]


def test_the_pair_that_has_ips_set_is_not_thrown_away():
    """คู่ที่ตั้ง IP ครบต้องขึ้นเป็นกลุ่มที่ 'พร้อม' — ของเดิมหายไปทั้งคู่"""
    groups = cluster_groups(_fleet())
    names = {tuple(sorted(m["name"] for m in g["members"])) for g in groups}

    assert ("spark-head", "spark-worker") in names, \
        "คู่ที่ตั้ง IP ครบแล้วหายไป — นี่คืออาการที่ผู้ใช้เจอ"

    ready = [g for g in groups if g["ready"]]
    assert [tuple(sorted(m["name"] for m in g["members"])) for g in ready] \
        == [("spark-head", "spark-worker")]


def test_every_pair_shows_up_not_only_the_biggest():
    """ฟลีตที่มีเครื่องรุ่นเดียวกันหลายคู่ ต้องเห็นครบทุกคู่"""
    groups = cluster_groups(_fleet())
    names = {tuple(sorted(m["name"] for m in g["members"])) for g in groups}
    assert names == {("spark-head", "spark-worker"), ("msi-1", "msi-2")}


def test_machines_in_different_sites_are_never_grouped_together():
    """stacked ข้ามไซต์ทำไม่ได้จริง — NCCL ต้องวิ่งบนสายในแร็ค ไม่ใช่ผ่าน WAN

    ผู้ใช้ย้ำเองว่าฟีเจอร์นี้ต้องอยู่ในไซต์เดียวกัน · ถึงสองไซต์จะบังเอิญใช้เลขวงเดียวกัน
    (10.55.0.x ซ้ำกันได้สบายในเน็ตส่วนตัว) ก็ต้องไม่ถูกจับคู่ข้ามกัน
    """
    fleet = [
        {"name": "a1", "site": "กรุงเทพ", "cluster_ip": "10.55.0.1",
         "host": _spark("10.55.0.1", "10.2.2.1")},
        {"name": "b1", "site": "เชียงใหม่", "cluster_ip": "10.55.0.2",
         "host": _spark("10.55.0.2", "10.2.2.2")},
    ]
    groups = cluster_groups(fleet)
    assert groups == [], "จับคู่ข้ามไซต์ไม่ได้ แม้เลขวงจะบังเอิญตรงกัน"


def test_each_group_says_which_site_it_belongs_to():
    for group in cluster_groups(_fleet()):
        assert group["site"] in {"Neronain", "TKC"}


# ── หลายคลัสเตอร์ในไซต์เดียวกัน ──────────────────────────────────────────
#
# ผู้ใช้ถาม 2026-08-31: "ถ้าใน site นั้น ผมจะทำ cluster มากกว่า 1 cluster ต้องทำอย่างไร"
#
# ระบบแบ่งเองได้เฉพาะตอนที่แต่ละคู่อยู่คนละวง · เครื่องรุ่นเดียวกันสี่เครื่องบนวงเดียวกัน
# จะถูกมองเป็นก้อนเดียว TP=4 ซึ่งบางทีไม่ใช่สิ่งที่ต้องการ — ต้องตั้งชื่อคลัสเตอร์เอง


def _four_on_one_subnet():
    """สี่เครื่องรุ่นเดียวกัน ไซต์เดียวกัน วงเดียวกันหมด"""
    return [
        {"name": f"n{i}", "site": "TKC", "cluster_ip": f"10.100.152.{i}", "cluster_name": "",
         "host": _spark(f"10.100.152.{i}", f"10.2.2.{i}")}
        for i in (1, 2, 3, 4)
    ]


def test_without_a_name_four_machines_on_one_subnet_become_one_cluster():
    """พฤติกรรมเดิมต้องไม่เปลี่ยน — ไม่ตั้งชื่อ = ระบบแบ่งเองตาม subnet"""
    groups = cluster_groups(_four_on_one_subnet())
    assert len(groups) == 1
    assert groups[0]["world_size"] == 4
    assert groups[0]["cluster_name"] == ""


def test_naming_splits_one_subnet_into_two_clusters():
    """ตั้งชื่อเดียวกันให้เครื่องที่ต้องการอยู่ด้วยกัน — วงเดียวกันก็แยกได้"""
    fleet = _four_on_one_subnet()
    fleet[0]["cluster_name"] = fleet[1]["cluster_name"] = "ทีมค้นหา"
    fleet[2]["cluster_name"] = fleet[3]["cluster_name"] = "ทีมสำรอง"

    groups = cluster_groups(fleet)
    by_name = {g["cluster_name"]: sorted(m["name"] for m in g["members"]) for g in groups}

    assert by_name == {"ทีมค้นหา": ["n1", "n2"], "ทีมสำรอง": ["n3", "n4"]}
    assert all(g["world_size"] == 2 for g in groups)
    assert all(g["ready"] for g in groups), "ตั้ง IP ครบแล้วทั้งสี่ ทั้งสองกลุ่มต้องพร้อม"


def test_a_named_cluster_whose_machines_share_no_network_says_so():
    """ตั้งชื่อรวมกันได้ แต่ถ้ายิง NCCL ถึงกันไม่ได้ต้องบอก ไม่ใช่ปล่อยไปค้างตอน start"""
    fleet = [
        {"name": "a", "site": "TKC", "cluster_ip": "10.1.0.1", "cluster_name": "รวมมิตร",
         "host": _spark("10.1.0.1", "10.2.2.1")},
        {"name": "b", "site": "TKC", "cluster_ip": "10.9.0.1", "cluster_name": "รวมมิตร",
         "host": _spark("10.9.0.1", "10.2.2.2")},
    ]
    groups = cluster_groups(fleet)
    assert len(groups) == 1
    kinds = {b["kind"] for b in groups[0]["blockers"]}
    assert "no-shared-fabric" in kinds or "split-fabric" in kinds
    assert groups[0]["ready"] is False


def test_a_name_never_pulls_machines_across_sites():
    """ชื่อคลัสเตอร์แบ่งได้ภายในไซต์ — ข้ามไซต์ยังห้ามเหมือนเดิม"""
    fleet = [
        {"name": "a", "site": "กรุงเทพ", "cluster_ip": "10.100.152.1", "cluster_name": "รวมมิตร",
         "host": _spark("10.100.152.1", "10.2.2.1")},
        {"name": "b", "site": "เชียงใหม่", "cluster_ip": "10.100.152.2", "cluster_name": "รวมมิตร",
         "host": _spark("10.100.152.2", "10.2.2.2")},
    ]
    assert cluster_groups(fleet) == []


def test_the_console_can_set_a_cluster_name():
    """ผู้ใช้ย้ำเรื่อง GUI — ฟีเจอร์ที่ตั้งได้แต่ใน CLI เท่ากับไม่มีสำหรับทีมที่ใช้หน้าเว็บ"""
    page = (_Path(__file__).resolve().parents[1]
            / "src" / "lmds" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'class="clname"' in page, "ไม่มีช่องตั้งชื่อคลัสเตอร์"
    assert 'data-cact="save-clname"' in page, "ไม่มีปุ่ม Save ของชื่อคลัสเตอร์"
    assert "cluster_name:" in page or "cluster_name" in page
    # หัวกลุ่มต้องบอกว่าแบ่งเองหรือระบบแบ่ง ไม่งั้นดูไม่ออกว่าทำไมกลุ่มออกมาแบบนี้
    assert "แบ่งอัตโนมัติ" in page


def test_the_api_accepts_a_cluster_name_change():
    from fastapi.testclient import TestClient
    from lmds.hardware import serving
    from lmds.web import state
    from lmds.web.api import create_app

    try:
        client = TestClient(create_app())
        r = client.patch("/api/nodes/ไม่มีเครื่องนี้", json={"cluster_name": "x"})
        assert r.status_code == 404, r.text
    finally:
        state.stop_refresher()
        state.STORE.__init__()
        serving.reset_cache()

