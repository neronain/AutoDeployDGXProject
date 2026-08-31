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
