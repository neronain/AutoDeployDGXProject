"""node push ต้องส่งของที่อยู่ในโฟลเดอร์ตอนนี้ ไม่ใช่ zip ที่แพ็กไว้ตอน generate

เคสจริง 2026-09-03: `lmds set --engine-env ...` แล้ว `lmds node push spark-head` — container
ขึ้นมาโดยไม่มี env สักตัว เพราะ zip ถูกสร้างตอน deploy ก่อน set · bundle.args ของ Sehyo ก็ไม่ถึง
spark-worker ทั้งที่ hub รายงานว่าบันทึกแล้ว · อาการนี้เงียบสนิท: push บอกสำเร็จ start ก็สำเร็จ
"""

import zipfile

from lmds.packager.bundle import make_zip


def test_make_zip_includes_files_written_after_generate(tmp_path):
    bundle = tmp_path / "demo"
    bundle.mkdir()
    (bundle / "demo-single.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    stale = make_zip(bundle)                      # zip ตอน generate
    (bundle / "bundle.env").write_text('API_PORT="${API_PORT:-8001}"\n', encoding="utf-8")
    (bundle / "bundle.args").write_text("--speculative-config x\n", encoding="utf-8")

    assert "demo/bundle.env" not in zipfile.ZipFile(stale).namelist()
    fresh = make_zip(bundle)                      # สิ่งที่ push ต้องทำก่อนส่ง
    names = zipfile.ZipFile(fresh).namelist()
    assert "demo/bundle.env" in names and "demo/bundle.args" in names
    assert "demo/demo.zip" not in names           # ห้ามยัด zip ซ้อน zip


def test_push_command_repacks_before_sending():
    """คุมที่ซอร์สของคำสั่ง — push ต้องเรียก make_zip เมื่อมีโฟลเดอร์ bundle อยู่ข้าง zip"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "lmds" / "cli" / "main.py").read_text(encoding="utf-8")
    i = src.index("def node_push(") if "def node_push(" in src else src.index("ส่ง bundle ที่สร้างไว้ในเครื่องนี้ไปติดตั้งบนเครื่องอื่น")
    body = src[i:i + 3000]
    assert "make_zip(" in body, "push ต้องแพ็กใหม่จากโฟลเดอร์ก่อนส่ง"
