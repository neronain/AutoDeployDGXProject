"""commit ที่ *รันอยู่* กับ commit ที่ *ติดตั้งไว้* เป็นคนละค่า และต้องแยกให้ออก

`install.sh` เขียน `_build.py` ทับได้ตลอด แต่ process ที่รันอยู่ import โมดูลนั้นไปแล้ว
python cache ไว้ใน sys.modules ค่าจึงไม่ขยับจนกว่าจะรีสตาร์ต

เคสจริงที่พังเพราะเรื่องนี้: header โชว์ commit เก่าค้างหลัง install.sh แล้วหน้าเว็บเอาเลขนั้น
ไปเทียบกับ node — node ทุกเครื่องที่เพิ่งอัปเดตถูกต้องเลยโดนติดป้ายว่ารันโค้ดเก่ากันหมด
"""

from __future__ import annotations

import lmds
from lmds.inventory import installed_commit
from lmds.web import daemon


def test_installed_commit_reads_the_file_not_the_import_cache(monkeypatch, tmp_path):
    """เขียน _build.py ทับแล้วต้องเห็นค่าใหม่ทันที โดยไม่ต้องรีสตาร์ต process"""
    package = tmp_path / "lmds"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    build = package / "_build.py"
    build.write_text('COMMIT = "aaaaaaa"\nSOURCE = "/x"\n', encoding="utf-8")
    monkeypatch.setattr(lmds, "__file__", str(package / "__init__.py"))

    assert installed_commit() == "aaaaaaa"

    build.write_text('COMMIT = "bbbbbbb"\nSOURCE = "/x"\n', encoding="utf-8")
    assert installed_commit() == "bbbbbbb", "อ่านผ่าน import cache อยู่ — ค่าจะไม่ขยับ"


def test_installed_commit_survives_a_missing_or_odd_build_file(monkeypatch, tmp_path):
    """เครื่องที่ยังไม่เคยผ่าน install.sh ไม่มีไฟล์นี้ — ต้องคืนค่าว่าง ไม่ใช่ระเบิดทั้งหน้า"""
    package = tmp_path / "lmds"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(lmds, "__file__", str(package / "__init__.py"))
    assert installed_commit() == ""

    (package / "_build.py").write_text("# ไม่มี COMMIT เลย\n", encoding="utf-8")
    assert installed_commit() == ""


def test_the_unit_refuses_to_restart_on_the_already_running_code():
    """ไม่มี RestartPreventExitStatus ที่ตรงกับรหัสนี้ = service วน restart ทุก 3 วิไม่รู้จบ

    เจอจริง 144 รอบ: มีคนรัน `lmds web` ด้วยมือค้างถือพอร์ตไว้ ตัวที่ systemd สั่งขึ้นจึง
    เจอ "มีหน้าเว็บรันอยู่แล้ว" แล้วจบด้วย 0 ซึ่ง Restart=always อ่านว่าสำเร็จ
    """
    unit = daemon.render_unit(port=8600, bind="0.0.0.0", token="t")
    assert f"RestartPreventExitStatus={daemon.EXIT_ALREADY_RUNNING}" in unit
    assert daemon.EXIT_ALREADY_RUNNING != 0, "จบด้วย 0 คือบอก systemd ว่าสำเร็จ"


def test_the_unit_gives_up_instead_of_looping_forever():
    """เพดานการปลุกซ้ำ — กันทุกสาเหตุที่ทำให้ล้มซ้ำ ไม่ใช่แค่เคส 'มีตัวรันอยู่แล้ว'

    ค่าปริยายของ systemd (5 ครั้ง/10 วิ) ดักไม่ได้เพราะ RestartSec=3 อยู่ใต้เพดานตลอด
    ต้องตั้งเองถึงจะมีวันหยุด · ที่หยุดแล้วไปอยู่สถานะ failed สำคัญ เพราะมันคือสิ่งเดียว
    ที่ทำให้คนเห็นว่าพัง — ตอนวนอยู่ หน้าเว็บยังเปิดได้ปกติจากตัวที่ถือพอร์ตอยู่
    """
    unit = daemon.render_unit(port=8600, bind="0.0.0.0", token="t")
    assert "StartLimitBurst=" in unit and "StartLimitIntervalSec=" in unit
