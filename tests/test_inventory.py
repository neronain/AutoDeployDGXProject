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


def _server(controller):
    from lmds.fleet.manager import ServerInfo
    import inspect
    kwargs = {}
    for name, param in inspect.signature(ServerInfo).parameters.items():
        if param.default is inspect.Parameter.empty:
            kwargs[name] = ""
    kwargs.update(slug="demo", controller=str(controller))
    return ServerInfo(**kwargs)


def test_a_controller_without_download_is_treated_as_self_managed(tmp_path):
    """สคริปต์เองคือความจริงสุดท้าย — ไม่มี `download` แปลว่า LMDS โหลด weight ให้ไม่ได้

    bundle ที่ adopt มาบางตัวมี model id เป็นรูป org/name ตามปกติ จึงหลุดตัวกรองที่เดาจาก
    profile แล้วหน้าเว็บยื่นปุ่ม download/repair ที่กดไปเจอ usage ของ bash (ผู้ใช้รายงานว่า
    "กด repair แล้วไม่ทำงาน" หลังแอด node ที่มี vllm อยู่ก่อน)
    """
    controller = tmp_path / "ctl.sh"
    controller.write_text("case $1 in\n  start)  start ;;\n  logs)   logs ;;\nesac\n")
    payload = inventory.model_payload(_server(controller))
    assert payload["self_managed_weights"] is True
    assert payload["downloaded"] is True     # ปุ่มที่ควรได้คือ start ไม่ใช่ download
