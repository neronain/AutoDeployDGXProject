"""deploy แบบ stacked ต้อง "กดได้" จากหน้าเว็บ ไม่ใช่พิมพ์คำสั่งให้ไปก็อป

ผู้ใช้รายงาน 2026-08-30 (วันโชว์เคส): "การ Deploy Stack ไม่เจอ"

ของเดิมหน้า Cluster ตรวจเจอคู่ที่ stacked ได้ รู้ด้วยว่าใครควรเป็น head — แล้วก็
พิมพ์ออกมาเป็นข้อความว่าให้ไปรัน `lmds deploy --target dgx-spark-stacked` กับ
`lmds node cluster --write` เองที่เทอร์มินัล · แปลว่าเส้นทาง deploy แบบ stacked
ไม่มีอยู่จริงในหน้าเว็บ ทั้งที่ข้อมูลที่ต้องใช้อยู่ในมือ server ครบแล้ว

เทสนี้ยึดสามอย่างไว้:
  1. หน้า Cluster มี *ปุ่ม* ไม่ใช่แค่คำสั่งให้ก็อป
  2. เลือกเครื่องแล้วต้องไม่ทับ target ที่ผู้ใช้ปักไว้เอง (stacked → single เงียบ ๆ)
  3. มี endpoint ให้เขียน cluster.env ได้จากหน้าเว็บ
"""

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_ready_cluster_offers_a_button_not_a_command_to_copy(page):
    assert 'button class="deploy-stack"' in page, "หน้า Cluster ยังไม่มีปุ่ม deploy"
    assert 'data-head=' in page and 'data-worker=' in page, "ปุ่มต้องพก head/worker ไปด้วย"
    # ปุ่มต้องเปิด wizard โดยตั้ง target ให้แล้ว ไม่ใช่ให้ผู้ใช้ไปเลือกเอง
    handler = page[page.index('button.deploy-stack'):]
    assert 'dgx-spark-stacked' in handler[:600]


def test_choosing_a_machine_does_not_silently_undo_a_stacked_target(page):
    """เลือกเครื่อง = ระบบเดา target ให้ · แต่ห้ามทับของที่ผู้ใช้ตั้งใจเลือกเอง

    เดิมทับเสมอ: กด Deploy จากหน้า Cluster (ได้ stacked มา) แล้วช่อง Run on ถูกเซ็ต
    ตามไปด้วย → onchange ยิง → target กลายเป็น dgx-spark-single ทันทีโดยไม่มีอะไรบอก
    """
    assert "targetPinned" in page, "ไม่มีตัวแยกว่า target มาจากผู้ใช้หรือจากการเดา"
    m = re.search(r"if \(suggested && target[^)]*\)", page)
    assert m, "ไม่เจอจุดที่เซ็ต target จากเครื่องที่เลือก"
    assert "!targetPinned" in m.group(0), "ยังทับ target ที่ปักไว้อยู่"


def test_worker_picker_appears_only_for_stacked(page):
    assert 'id="w-worker"' in page, "ไม่มีช่องเลือก worker"
    assert "function syncStackedFields" in page
    assert "function isStackedTarget" in page
    # ค่าเริ่มต้นต้องซ่อน — คน deploy เครื่องเดียวไม่ควรเห็นช่องที่ไม่เกี่ยวกับตัวเอง
    assert re.search(r'id="w-worker"[^>]*hidden', page), "ช่อง worker ต้องซ่อนไว้ก่อน"


def test_cluster_env_is_written_after_a_stacked_push(page):
    """ขั้นที่หายไปจริง ๆ — bundle ที่ไม่มี cluster.env จะไปค้างที่ NCCL init เงียบ ๆ"""
    assert "async function writeClusterEnv" in page
    assert '"/api/cluster/write"' in page
    assert "pushAfterBuild(d, draft.machine, isStackedTarget(draft.target))" in page


def test_cluster_write_endpoint_exists_and_validates_input():
    """`create_app()` สตาร์ต refresher เบื้องหลังด้วย — ต้องหยุดให้เรียบร้อย

    ไม่หยุด = thread นั้นวิ่งต่อไปสำรวจเครื่องจริงตลอดเทสที่เหลือ แล้วผลสำรวจของ
    เครื่องที่ไม่มี GPU ไปนอนอยู่ในแคชกลาง · เทสที่รันทีหลังจึงเจอ `_guard_serving`
    ปฏิเสธคำสั่งอย่าง repair/download ทั้งที่ตัวมัน monkeypatch ไว้แล้ว
    (เจอจริงตอนเพิ่มไฟล์นี้: 11 เทสใน test_fleet/test_web ล้มเพราะไฟล์นี้ชื่อขึ้นต้น
    ด้วย d จึงรันก่อนพวกนั้น — ของเดิมรอดเพราะ test_web อยู่ท้ายสุดอยู่แล้ว)
    """
    from fastapi.testclient import TestClient
    from lmds.hardware import serving
    from lmds.web import state
    from lmds.web.api import create_app

    try:
        client = TestClient(create_app())
        r = client.post("/api/cluster/write", json={})
        assert r.status_code == 400, r.text
        assert "slug" in r.json()["detail"]
    finally:
        state.stop_refresher()
        state.STORE.__init__()
        # `serving._detect` เป็น lru_cache(maxsize=1) ที่อ่าน LMDS_ROLE ตอนถูกเรียก
        # *ครั้งแรก* ของ process — create_app() ไปเรียกมันก่อนใคร คำตอบ "เครื่องนี้เป็น
        # hub" จึงถูกตรึงไว้ทั้งรัน แล้วเทสที่ setenv ทีหลังก็ไม่มีผลอีกเลย
        serving.reset_cache()


def test_build_cluster_env_reports_the_reason_instead_of_exiting():
    """หน้าเว็บเรียกใช้ได้ = ห้าม typer.Exit / ห้ามพิมพ์ลง console"""
    from lmds.fleet.cluster_env import ClusterEnvError, build_cluster_env

    groups = [{"ready": True, "members": [
        {"name": "head", "cluster_ip": "10.100.152.1", "iface": "enp1s0f1np1"},
        {"name": "worker", "cluster_ip": "10.100.152.2"},
    ]}]

    built = build_cluster_env(groups, "head", "worker")
    assert "MASTER_IP=10.100.152.1" in built["body"]
    assert "WORKER_IP=10.100.152.2" in built["body"]
    assert "NNODES=2" in built["body"]
    assert "TENSOR_PARALLEL_SIZE=2" in built["body"]
    assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in built["body"]

    with pytest.raises(ClusterEnvError) as err:
        build_cluster_env(groups, "head", "ไม่มีเครื่องนี้")
    assert "ไม่ได้อยู่ในกลุ่ม" in str(err.value)

    with pytest.raises(ClusterEnvError):
        build_cluster_env([], "head", None)
