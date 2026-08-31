"""สถานะคลัสเตอร์ต้องอยู่บนหน้าจอตลอด ไม่ใช่ต้องกดปุ่มทุกครั้งถึงจะเห็น

ผู้ใช้ขอ 2026-08-31: "รบกวนเพิ่ม ให้เห็นสถานะ cluster ให้เห็นตลอดการใช้งานหน่อย
ไม่ต้องค่อยกด check ตลอดถึงจะเห็น"

ของเดิม `/api/cluster` ไปต่อทุกเครื่องสด ๆ ทุกครั้ง จึงช้าเกินกว่าจะเรียกถี่ ๆ ได้ และ
หน้าเว็บเรียกเฉพาะตอนกดปุ่ม · แต่ refresher probe ทุกเครื่องรอบละ 15 วิเพื่อทำการ์ด
สถานะอยู่แล้ว — ข้อมูลที่ cluster_groups ต้องใช้อยู่ในแคชครบ ไม่ต้องไปถามซ้ำ
"""

from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from lmds.hardware import serving
    from lmds.web import state
    from lmds.web.api import create_app

    state.STORE.__init__()
    made = TestClient(create_app())
    yield made
    state.stop_refresher()
    state.STORE.__init__()
    serving.reset_cache()


def test_the_default_read_never_touches_the_machines(client, monkeypatch):
    """อ่านจากแคช = ตอบทันที · ถ้ายังไป probe อยู่ก็เรียกถี่ไม่ได้อยู่ดี"""
    probed = []
    monkeypatch.setattr("lmds.nodes.probe",
                        lambda node, timeout=30: probed.append(node.name) or {"host": {}},
                        raising=False)

    r = client.get("/api/cluster")
    assert r.status_code == 200, r.text
    assert probed == [], f"ยังไป probe อยู่: {probed}"
    assert r.json()["live"] is False


def test_the_button_can_still_force_a_live_check(client, monkeypatch):
    """ปุ่ม Check cluster ต้องยังบังคับอ่านสดได้ — ใช้ตอนเพิ่งเสียบสายใหม่"""
    probed = []

    def fake_probe(node, timeout=30):
        probed.append(node.name)
        return {"host": {}}

    monkeypatch.setattr("lmds.nodes.probe", fake_probe, raising=False)
    from lmds.nodes import Node, add

    add(Node(name="n1", host="10.0.0.1", user="ops"))
    r = client.get("/api/cluster?refresh=true")
    assert r.status_code == 200, r.text
    assert r.json()["live"] is True
    assert probed == ["n1"], "refresh=true ต้องไปต่อเครื่องจริง"


def test_a_node_with_nothing_cached_yet_says_so_instead_of_stalling(client):
    """เครื่องที่ refresher ยังไม่เคยสำรวจสำเร็จ ต้องรายงานตามจริง ไม่ใช่ไปต่อสดแล้วค้าง"""
    from lmds.nodes import Node, add

    add(Node(name="ยังไม่เคยเจอ", host="10.0.0.9", user="ops"))
    rows = client.get("/api/cluster").json()["machines"]
    row = next(m for m in rows if m["name"] == "ยังไม่เคยเจอ")
    assert row["reachable"] is False
    assert "ยังไม่มีข้อมูล" in row["error"]


def test_the_console_refreshes_the_cluster_every_cycle():
    page = INDEX.read_text(encoding="utf-8")
    # refreshNodes จบด้วยการอ่านสถานะคลัสเตอร์ใหม่ ไม่ใช่แค่จัดรั้วของเดิม
    tail = page[page.index("async function refreshNodes"):]
    body = tail[:tail.index("\ndocument.addEventListener")]
    assert "loadCluster()" in body, "รอบ refresh ไม่ได้ดึงสถานะคลัสเตอร์ใหม่"
    # ปุ่มยังเป็นการตรวจสด
    assert 'loadCluster(true)' in page
    assert '"/api/cluster" + (live ? "?refresh=true" : "")' in page


def test_a_failed_cached_read_keeps_what_is_already_on_screen():
    """เน็ตสะดุดหนึ่งรอบต้องไม่ทำให้รั้วกลุ่มกะพริบหาย — เรียกทุกรอบแล้วยิ่งสำคัญ"""
    page = INDEX.read_text(encoding="utf-8")
    assert "if (live) clusterData = null;" in page
