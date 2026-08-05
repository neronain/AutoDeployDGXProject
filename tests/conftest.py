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
def fresh_web_state(monkeypatch):
    """ล้างแคชของหน้าเว็บทุกเทส — STORE เป็น global ค้างข้ามเทสได้

    เทสที่รันทีหลังจะได้ snapshot ของเทสก่อน แล้วเห็นโมเดลที่ไม่ใช่ของตัวเอง
    (อาการเดียวกับที่ผู้ใช้เจอบนเครื่อง controller เป๊ะ) · หยุด refresher ด้วย
    ไม่งั้น thread เบื้องหลังจะไปยิง SSH ระหว่างเทส

    ล้างแคชอย่างเดียวไม่พอ: create_app() สตาร์ท refresher ใหม่ทุกครั้ง แล้ว thread นั้น
    เขียน STORE ต่อได้หลังเราล้างไปแล้ว เทสจึงล้มสลับตัวไปมาแล้วแต่จังหวะ · ไม่ให้สตาร์ทเลย
    ตรงกว่าไล่หยุดทีหลัง — endpoint คำนวณสดเมื่อแคชว่าง ซึ่งเป็นสิ่งที่เทสตรวจอยู่แล้ว
    """
    try:
        from lmds.web import state
    except ImportError:      # ยังไม่ได้ติดตั้ง extra ของเว็บ
        yield
        return
    # node probe มี subprocess timeout 30 วิ — 5 วิ default ยังปล่อย thread เก่าเขียน STORE
    # ตามหลัง reset ได้ ถ้าหยุดไม่ลงในกรอบนี้ให้ fail ตรง ๆ แทนการกลับไป flaky
    assert state.stop_refresher(timeout=35.0), "web refresher did not stop before STORE reset"
    monkeypatch.setattr(state, "start_refresher", lambda: None)
    state.STORE.__init__()
    yield
    assert state.stop_refresher(timeout=35.0), "web refresher did not stop before STORE reset"
    state.STORE.__init__()
