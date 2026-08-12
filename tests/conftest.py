import sys
from pathlib import Path

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


def _real_config_files() -> dict[str, bytes]:
    """Contents of the operator's real config, if any exists on this machine.

    Contents rather than checksums: a checksum tells you the file changed, the
    bytes let you put it back. The directory is a few kilobytes.
    """
    real = Path.home() / ".config" / "lmds"
    if not real.is_dir():
        return {}
    snapshot: dict[str, bytes] = {}
    for path in sorted(real.rglob("*")):
        if path.is_file():
            try:
                snapshot[str(path)] = path.read_bytes()
            except OSError:
                continue
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def never_touch_the_real_config():
    """กันเทสเขียนทับ config จริงของเครื่อง — เคยลบ nodes.yaml ของผู้ใช้มาแล้ว

    isolated_config กัน in-process ได้ แต่ subprocess ที่ไม่ได้รับ env ไปด้วย
    หรือโค้ดที่ resolve home เองยังหลุดได้ ซึ่งไฟล์ที่พังคือทะเบียนเครื่องจริง
    ของผู้ใช้ · เทียบ checksum ก่อน/หลัง session จับได้ทุกทาง
    """
    before = _real_config_files()
    yield
    after = _real_config_files()
    if before == after:
        return

    damaged = sorted({*before, *after}, key=str)
    restored: list[str] = []
    for name in damaged:
        original = before.get(name)
        if original is None:
            continue  # เทสสร้างไฟล์ใหม่ — ปล่อยไว้ให้เห็น ไม่ใช่ของเดิมที่หาย
        if after.get(name) == original:
            continue
        try:
            path = Path(name)
            path.write_bytes(original)
            path.chmod(0o600)
            restored.append(name)
        except OSError:  # noqa: PERF203
            pass

    raise AssertionError(
        "เทสไปแก้ config จริงของเครื่องนี้ — กู้คืนให้แล้ว แต่ต้องอุดรูก่อนรันอีก:\n  "
        + "\n  ".join(restored or [n for n in damaged if n in before])
    )
