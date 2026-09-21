"""แคชของ pip ต้องใช้ได้เสมอตอนติดตั้ง — ไม่งั้นได้ WARNING สี่บรรทัดทุกครั้งที่อัปเดต node

เคสจริง (2026-09-21, spark-head): ทุกครั้งที่อัปเดต node ผ่านหน้าเว็บ ขึ้น

    WARNING: The directory '/home/neronain/.cache/pip' or its parent directory is not
    owned or is not writable by the current user. The cache has been disabled. …
    If executing pip with sudo, you should use sudo's -H flag.

สองบรรทัดต่อการเรียก pip หนึ่งครั้ง และ `install.sh` เรียกสองครั้ง

**สาเหตุอยู่ในตัวเราเอง** ไม่ใช่เครื่องลูกค้า: ขั้นติดตั้ง prerequisite เรียก installer ผ่าน
`sudo env HOME="$HOME"` (`nodes/ssh.py`) เพราะ `~` ใต้ sudo คือ `/root` ไม่ใช่ home ของผู้ใช้
pip จึงรันเป็น root แต่เห็นแคชที่เป็นของผู้ใช้ธรรมดา — ตรงกับที่ข้อความของ pip แนะนำให้ใช้
`sudo -H` พอดี แต่เราใช้ `-H` ไม่ได้เพราะทั้ง installer พึ่ง `HOME` ของผู้ใช้

ที่สำคัญกว่าความรก: **แคชที่หายไปไม่ใช่แค่ช้าลง** · คอมเมนต์ใน `install.sh` เองบันทึกไว้ว่า
ไซต์ที่ PyPI ตอบช้าเคยทำให้ขั้น pip ล้มจนเครื่องเหลือแบบไม่มี lmds เลย (2026-09-04)
การแก้ด้วย `--no-cache-dir` จึงแก้อาการแล้วทำให้สาเหตุแย่ลง
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "install.sh"


def _block() -> str:
    """ดึงเฉพาะส่วนที่ตัดสินใจเรื่องแคชออกมารัน — ไม่ต้องรัน installer ทั้งไฟล์"""
    text = INSTALL_SH.read_text(encoding="utf-8")
    start = text.index("pip_cache_is_writable_by_us() {")
    end = text.index("\nfi\n", text.index("PIP_CACHE_DIR=\"${INSTALL_DIR}/pip-cache\"")) + 4
    return text[start:end]


def _decide(home: Path, install_dir: Path, env: dict | None = None) -> str:
    """คืนค่า PIP_CACHE_DIR ที่บล็อกนั้นตัดสินใจ — ว่าง = ปล่อยให้ pip ใช้ค่าเริ่มต้น"""
    script = (
        "set -Eeuo pipefail\n"
        f'HOME={home!s}\n'
        f'INSTALL_DIR={install_dir!s}\n'
        + _block()
        + '\nprintf "%s" "${PIP_CACHE_DIR:-}"\n'
    )
    full = dict(os.environ)
    full.pop("PIP_CACHE_DIR", None)
    full.update(env or {})
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env=full, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""


def test_a_cache_we_own_is_left_alone(tmp_path):
    """เครื่องปกติต้องใช้แคชเดิมต่อ — ย้ายทุกเครื่องแปลว่าทิ้งแคชที่มีอยู่แล้วของทุกคน"""
    home = tmp_path / "home"
    (home / ".cache" / "pip").mkdir(parents=True)
    assert _decide(home, tmp_path / "lmds") == ""


def test_a_machine_with_no_cache_yet_is_left_alone(tmp_path):
    """ยังไม่มีอะไรเลย = pip สร้างเองได้ · ย้ายตรงนี้คือแก้ปัญหาที่ยังไม่เกิด"""
    home = tmp_path / "home"
    home.mkdir()
    assert _decide(home, tmp_path / "lmds") == ""


def test_a_cache_owned_by_somebody_else_is_moved_not_disabled(tmp_path):
    """หัวใจของไฟล์นี้ · จำลองด้วยการถอดสิทธิ์เขียน ซึ่งเป็นผลเดียวกับที่ root เป็นเจ้าของ
    (เทสรันเป็น user ธรรมดา จะ chown เป็น root ไม่ได้)"""
    home = tmp_path / "home"
    cache = home / ".cache" / "pip"
    cache.mkdir(parents=True)
    cache.chmod(0o500)
    try:
        chosen = _decide(home, tmp_path / "lmds")
    finally:
        cache.chmod(0o700)
    assert chosen == str(tmp_path / "lmds" / "pip-cache"), (
        "ต้องย้ายไปที่ที่เราเพิ่งสร้างเอง ไม่ใช่ปิดแคชทิ้ง — แคชที่หายเคยทำให้ติดตั้งล้มมาแล้ว"
    )


def test_a_cache_parent_owned_by_somebody_else_is_caught_too(tmp_path):
    """`~/.cache` เป็นของ root จาก sudo ครั้งเก่า แต่ `~/.cache/pip` ยังไม่มี —
    pip จะสร้างไม่ได้ และเตือนเรื่อง 'or its parent directory' ตรงตามข้อความของมันเอง"""
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    (home / ".cache").chmod(0o500)
    try:
        chosen = _decide(home, tmp_path / "lmds")
    finally:
        (home / ".cache").chmod(0o700)
    assert chosen == str(tmp_path / "lmds" / "pip-cache")


def test_an_operator_who_set_the_cache_themselves_wins(tmp_path):
    """ไซต์ที่ชี้แคชไปยัง mirror/ดิสก์ของตัวเองไว้แล้ว ต้องไม่ถูกเราเขียนทับ"""
    home = tmp_path / "home"
    cache = home / ".cache" / "pip"
    cache.mkdir(parents=True)
    cache.chmod(0o500)
    try:
        chosen = _decide(home, tmp_path / "lmds", env={"PIP_CACHE_DIR": "/srv/pip"})
    finally:
        cache.chmod(0o700)
    assert chosen == "/srv/pip"


def test_the_installer_never_reaches_for_no_cache_dir():
    """กันการ 'แก้' ในอนาคตที่ทำให้สาเหตุแย่ลง — ปิดแคชคือทางที่เร็วและผิด

    ดูเฉพาะบรรทัดที่เป็นโค้ด: คอมเมนต์ที่อธิบายว่า *ทำไมถึงไม่ใช้* ต้องไม่ทำให้เทสแดง
    (เทสรุ่นแรกแดงเพราะคอมเมนต์ของตัวเองพอดี)
    """
    code = [line for line in INSTALL_SH.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]
    assert not [line for line in code if "--no-cache-dir" in line]
