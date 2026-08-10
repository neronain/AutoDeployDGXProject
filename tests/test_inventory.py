"""ภาพรวมเครื่องที่ hub ดึงผ่าน `lmds agent info`"""

from lmds import inventory
def test_cache_health_flags_root_owned_entries(tmp_path, monkeypatch):
    """แคชที่กลายเป็นของ root ทำให้ download/remove/sync ล้มโดยไม่มีสาเหตุที่มองเห็น
    — เดิมไม่มีอะไรตรวจเลย ผู้ใช้เห็นแค่คำสั่งที่ล้ม (เจอจริงบน msi-5: hub/ 73 GB เป็นของ root)
    """
    hub = tmp_path / "hub"
    (hub / "models--org--name").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert inventory.cache_health()["owner_ok"] is True

    monkeypatch.setattr(inventory.os, "getuid", lambda: -1)   # ทุกอย่างกลายเป็นของคนอื่น
    health = inventory.cache_health()
    assert health["owner_ok"] is False and health["foreign_entries"] > 0


def test_cache_health_is_quiet_on_a_fresh_machine(tmp_path, monkeypatch):
    """เครื่องที่ยังไม่มีแคช ไม่ใช่เครื่องที่มีปัญหา — ต้องไม่ขึ้นเตือน"""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert inventory.cache_health()["owner_ok"] is None
