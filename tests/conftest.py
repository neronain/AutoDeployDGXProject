import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """แยก config dir ต่อเทส + ปิด keyring และ env จริงของเครื่อง ไม่ให้เทสไปแตะของผู้ใช้"""
    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path / "lmds-config"))
    # ทะเบียน runtime ต้องแยกด้วย — เทสที่ลืมตั้งเองเคยเขียนลง ~/.lmds/run ของเครื่องจริง
    # แล้วทิ้งรายการค้างไว้ให้ผู้ใช้เห็นในหน้าเว็บ (เจอจริง: qwen3-32b, qwen3-8b-gguf)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "lmds-run"))
    for env_names in __import__("lmds.secrets.store", fromlist=["SECRET_ENV_VARS"]).SECRET_ENV_VARS.values():
        for env_name in env_names:
            monkeypatch.delenv(env_name, raising=False)
    # บังคับ path แบบไฟล์: ทำให้ import keyring ล้มเหลวในเทส
    monkeypatch.setitem(sys.modules, "keyring", None)
    yield tmp_path / "lmds-config"


@pytest.fixture(autouse=True)
def fresh_web_state():
    """ล้างแคชของหน้าเว็บทุกเทส — STORE เป็น global ค้างข้ามเทสได้

    เทสที่รันทีหลังจะได้ snapshot ของเทสก่อน แล้วเห็นโมเดลที่ไม่ใช่ของตัวเอง
    (อาการเดียวกับที่ผู้ใช้เจอบนเครื่อง controller เป๊ะ) · หยุด refresher ด้วย
    ไม่งั้น thread เบื้องหลังจะไปยิง SSH ระหว่างเทส
    """
    try:
        from lmds.web import state
    except ImportError:      # ยังไม่ได้ติดตั้ง extra ของเว็บ
        yield
        return
    state.stop_refresher()
    state.STORE.__init__()
    yield
    state.stop_refresher()
    state.STORE.__init__()


@pytest.fixture(autouse=True)
def no_registry_lookups(monkeypatch):
    """เทสต้องไม่ยิงเน็ตจริง — การตรวจ image tag ทำให้ชุดเทสช้าจาก 12 วิเป็น 90 วิ
    และผลจะเปลี่ยนไปตามว่าตอนนั้นต่อเน็ตได้ไหม ซึ่งไม่ใช่สิ่งที่เทสควรวัด

    เทสที่ตั้งใจตรวจพฤติกรรมนี้ patch ทับเองได้ตามปกติ
    """
    try:
        from lmds.brain.registry import SKIP_ENV
    except ImportError:
        yield
        return
    # ตั้ง env แทนการ patch ตัวฟังก์ชัน — patch แล้วเทสของ tag_exists เองจะไปทดสอบ stub
    monkeypatch.setenv(SKIP_ENV, "1")
    yield
