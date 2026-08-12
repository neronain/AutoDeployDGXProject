import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ─── sandbox ของทั้ง session ────────────────────────────────────────────────
# ตั้งตอน import conftest คือ "ก่อน" pytest จะ collect/import ไฟล์เทส — เร็วกว่า
# fixture ทุกตัว · โค้ดระดับ module ในไฟล์เทสที่แตะ config ตอน import จึงถูกครอบด้วย
#
# ย้าย HOME ไม่ใช่แค่ LMDS_CONFIG_DIR เพราะรูที่ทำให้ nodes.yaml ของผู้ใช้พังสองครั้ง
# คือโค้ดที่ resolve home เอง (Path.home() / expanduser) ซึ่งไม่สนใจ env ของเรา
# ย้าย HOME แล้วรูนั้นปิดสนิท: path จริงกลายเป็นสิ่งที่ process นี้ "ชี้ไปไม่ถึง"
# แทนที่จะเป็นสิ่งที่เราคอยเฝ้าดูว่าพังหรือยัง
REAL_HOME = Path.home()
REAL_CONFIG_DIR = REAL_HOME / ".config" / "lmds"
REAL_RUN_ROOT = REAL_HOME / ".lmds" / "run"

SANDBOX_HOME = Path(tempfile.mkdtemp(prefix="lmds-tests-home-"))

# เก็บกวาดที่ atexit ไม่ใช่ที่ fixture teardown — pytest ที่ import conftest แล้วไม่รัน
# fixture (เช่น --collect-only หรือ collect พัง) จะทิ้งโฟลเดอร์ค้างใน /tmp ทุกครั้ง
atexit.register(shutil.rmtree, SANDBOX_HOME, ignore_errors=True)

os.environ["HOME"] = str(SANDBOX_HOME)
os.environ["LMDS_CONFIG_DIR"] = str(SANDBOX_HOME / ".config" / "lmds")
os.environ["LMDS_RUN_ROOT"] = str(SANDBOX_HOME / ".lmds" / "run")


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


@pytest.fixture(autouse=True)
def no_systemd_lookups(monkeypatch):
    """เทสต้องไม่ไปถาม systemd จริงของเครื่อง

    `lmds web --stop/--restart` เรียก daemon.service_active() ซึ่งรัน
    `systemctl --user is-active lmds-web.service` จริง · ผลเทสจึงขึ้นกับว่าเครื่องที่รัน
    เปิด service ไว้หรือเปล่า — ถ้าเปิดอยู่ เทสจะเดินไปสาย systemd แทนสายที่ตั้งใจทดสอบ

    ยังทำให้ `subprocess.run` ข้างใน service_active() ไม่ถูกเรียกด้วย ซึ่งสำคัญเพราะ
    เทสที่ stub `subprocess.Popen` ไว้จะพังทันที: subprocess.run ใช้ Popen เป็น
    context manager แต่ stub ของเทสไม่มี __enter__/__exit__

    เทสที่ตั้งใจทดสอบสาย systemd patch ทับเป็น True เองอยู่แล้ว
    """
    try:
        from lmds.web import daemon
    except ImportError:      # ยังไม่ได้ติดตั้ง extra ของเว็บ
        yield
        return
    monkeypatch.setattr(daemon, "service_active", lambda: False)
    yield


def _live_paths() -> dict[str, Path]:
    """path ที่เทสกำลังจะเขียนจริง ๆ — ถามตัวโปรแกรม ไม่ใช่เดาจาก env

    import ในฟังก์ชันเพราะ conftest ถูก import ก่อนที่ sys.path จะพร้อมเสมอไป
    """
    from lmds.config.paths import config_dir
    from lmds.fleet import run_root

    return {"config_dir": config_dir(), "run_root": run_root(), "home": Path.home()}


def _escaped(where: str) -> list[str]:
    """path ไหนบ้างที่หลุดออกไปนอก sandbox แล้วชี้กลับมาที่ของจริง

    สองที่ที่เทสเคยทำพัง (config dir / run root) เช็คแบบ "เท่ากับหรืออยู่ข้างใน"
    ส่วน home เช็คแค่ "เท่ากับ" — tmp ของ pytest อยู่ใต้ home ได้ถ้า TMPDIR ตั้งไว้
    แบบนั้น เช็คแบบอยู่ข้างในจะเตือนผิดทุกเทสบนเครื่องแบบนั้น · ไม่เสียการป้องกัน
    เพราะถ้า home ถูกย้ายกลับไปของจริง config dir จะตกไปอยู่ใน REAL_CONFIG_DIR เอง
    """
    leaks = []
    for label, path in _live_paths().items():
        resolved = Path(path).expanduser()
        if resolved == REAL_HOME:
            leaks.append(f"{label} = {resolved}  (คือ home จริงของเครื่อง)")
            continue
        for real in (REAL_CONFIG_DIR, REAL_RUN_ROOT):
            if resolved == real or real in resolved.parents:
                leaks.append(f"{label} = {resolved}  (อยู่ใน {real})")
                break
    if leaks:
        leaks.append(f"— ตรวจตอน: {where}")
    return leaks


@pytest.fixture(scope="session", autouse=True)
def never_touch_the_real_config():
    """กันเทสเขียนทับ config จริงของเครื่อง — เคยลบ nodes.yaml ของผู้ใช้มาแล้ว

    เดิมกันด้วยการ snapshot ~/.config/lmds ตอนเริ่ม แล้วเทียบ+เขียนคืนตอนจบ ซึ่ง
    **แยกไม่ออก**ว่าไฟล์เปลี่ยนเพราะเทสหรือเพราะ daemon: `lmds web` เขียน nodes.yaml
    ทุก 1-3 วิ (วัดแล้ว) และ write_atomic ทิ้ง `.nodes.yaml.*.tmp` ไว้ชั่วขณะ ซึ่ง
    snapshot จับติดได้ ~0.25% ของครั้งที่มอง · ผลคือ session ล้มด้วยข้อความที่โทษเทส
    ผิด ๆ แถมสร้างไฟล์ .tmp ค้างไว้ และถ้าผู้ใช้เพิ่ม node ในหน้าเว็บระหว่างเทสรัน
    ของที่เพิ่งเพิ่มจะถูกเขียนทับหายไปเงียบ ๆ (ทดสอบแล้ว: หายจริง)

    ตอนนี้จึงไม่เฝ้าดูของจริงอีกต่อไป — ย้าย HOME ทั้ง session ให้ path ของจริง
    "ชี้ไปไม่ถึง" ตั้งแต่ต้น แล้วตรวจว่าการย้ายยังอยู่จริง · ไม่ต้องเดาว่าใครเขียน
    เพราะเทสเขียนไปที่นั่นไม่ได้ และ daemon จะเขียนของมันต่อไปโดยไม่มีใครไปยุ่ง
    """
    leaks = _escaped("เริ่ม session")
    assert not leaks, "sandbox ไม่ทำงานตั้งแต่ต้น:\n  " + "\n  ".join(leaks)
    yield
    leaks = _escaped("จบ session")
    assert not leaks, "มีเทสย้าย path กลับไปที่ของจริง:\n  " + "\n  ".join(leaks)


@pytest.fixture(autouse=True)
def _sandbox_holds(isolated_config):
    """ตรวจรายเทส — เทสที่ตั้ง env กลับไปที่ของจริงจะถูกจับตรงเทสนั้น ไม่ใช่ตอนจบ session

    ข้อความจึงชี้ตัวคนผิดได้ทันที ต่างจากของเดิมที่บอกแค่ว่า "ไฟล์เปลี่ยน" ตอนจบ

    ตรวจหลัง yield ด้วยเพราะเทสที่ย้าย env *ในตัวเทสเอง* ผ่านด่านก่อนเทสไปแล้ว ·
    fixture นี้ขึ้นกับ isolated_config จึง teardown ก่อน monkeypatch ถูกถอน —
    ตอนตรวจจึงยังเห็นค่าที่เทสตั้งไว้จริง ๆ ไม่ใช่ค่าที่ถูกคืนแล้ว
    """
    leaks = _escaped("ก่อนเทส")
    assert not leaks, "sandbox ไม่ทำงานในเทสนี้:\n  " + "\n  ".join(leaks)
    yield
    leaks = _escaped("ระหว่างเทส")
    assert not leaks, "เทสนี้ย้าย path ไปที่ของจริง:\n  " + "\n  ".join(leaks)
