"""ป้าย "รอรีสตาร์ต" และปุ่มรีสตาร์ตหน้าเว็บ

เคสจริงบนเครื่องลูกค้า: ป้ายติดค้างถาวร ปิด-เปิดเบราว์เซอร์ก็ไม่หาย reboot ก็ไม่หาย
ทำให้ไม่มีทางรู้ว่าโค้ดที่อัปไปแล้วทำงานหรือยัง
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from lmds.web.api import create_app


def test_a_git_checkout_does_not_look_like_a_pending_restart(monkeypatch):
    """git pull ขยับ HEAD แต่ไม่แตะ _build.py — เทียบสองค่านั้นตรง ๆ ป้ายจะติดถาวร"""
    from lmds import inventory

    monkeypatch.setattr(inventory, "_BOOT_COMMIT", None)
    monkeypatch.setattr(inventory, "_git_head", lambda: "cb87a7c")

    running = inventory.source_commit()
    on_disk = inventory.installed_commit()

    assert running == on_disk == "cb87a7c"
    assert not (running and on_disk and running != on_disk), "ป้ายต้องไม่ขึ้น"


def test_a_real_pending_restart_is_still_reported(monkeypatch):
    """pull แล้วยังไม่รีสตาร์ต = ต้องขึ้น · ไม่งั้นแก้บั๊กแรกแล้วป้ายก็ไร้ประโยชน์"""
    from lmds import inventory

    monkeypatch.setattr(inventory, "_BOOT_COMMIT", "aaaaaaa")
    monkeypatch.setattr(inventory, "_git_head", lambda: "bbbbbbb")

    assert inventory.source_commit() == "aaaaaaa"
    assert inventory.installed_commit() == "bbbbbbb"


def test_the_console_offers_a_restart_button():
    """เดิมทางเดียวคือ ssh เข้าไปพิมพ์ systemctl — คนที่ใช้ผ่านเว็บล้วน ๆ ทำไม่ได้"""
    page = TestClient(create_app()).get("/").text
    assert 'id="restart-web"' in page
    assert "restartConsole" in page
    assert "/api/restart" in page


def test_restart_reports_which_unit_it_will_restart(monkeypatch):
    calls: list[list[str]] = []

    class _Popen:
        def __init__(self, argv, **kw):
            calls.append(argv)

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    res = TestClient(create_app()).post("/api/restart")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["restarting"] is True
    assert body["unit"].endswith(".service") or body["unit"]
    # ต้องหลุดจาก process นี้ ไม่งั้น systemd ฆ่าตัวที่สั่งก่อนคำตอบจะกลับถึงเบราว์เซอร์
    # (มี Popen ตัวอื่นของ fleet ปนอยู่ด้วย จึงหาเอาจากทั้งรายการ ไม่ใช่ตัวแรก)
    detached = [c for c in calls if c and c[0] == "setsid"]
    assert detached, f"ไม่เจอคำสั่งแบบ setsid ใน {calls}"
    assert "systemctl --user restart" in " ".join(detached[0])


def test_a_transient_network_error_does_not_open_a_modal():
    """poll ทุก 5 วิ reject ทุกครั้งที่เซิร์ฟเวอร์ไม่ตอบ — เด้ง modal ทุกครั้งคือทำให้การ
    รีสตาร์ตตามปกติกลายเป็นเรื่องน่าตกใจ และบังคับให้กด OK โดยไม่มีอะไรให้ทำ
    """
    page = TestClient(create_app()).get("/").text
    assert "function isNetworkError" in page
    assert "consoleRestarting" in page
    # ตอนกดรีสตาร์ตเอง เน็ตหลุดคือสิ่งที่ตั้งใจให้เกิด ต้องเงียบสนิท
    assert "if (consoleRestarting && isNetworkError(reason)) { event.preventDefault(); return; }" in page


def test_show_error_actually_exists():
    """เคยเรียก showError() ไว้สองที่ทั้งที่ยังไม่มีฟังก์ชันนี้ — error path จะโยน
    ReferenceError ทับ error จริง แล้วผู้ใช้ไม่มีทางรู้ว่าเกิดอะไรขึ้น
    """
    page = TestClient(create_app()).get("/").text
    assert "function showError" in page
    assert 'id="toast"' in page
