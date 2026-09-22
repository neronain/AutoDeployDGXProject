"""หน้าเว็บของ hub ต้อง parse ผ่าน — ไม่งั้นทั้งคอนโซลตาย ไม่ใช่แค่ปุ่มเดียว

บทเรียนจาก LiteGate 2026-09-21: ตัวแปรชนกันตัวเดียวทำให้ทั้งไฟล์ parse ไม่ผ่าน
หน้าเว็บขึ้นว่า "not connected" ทุกแผงว่างเปล่า ส่วนชุดเทส 881 ตัวผ่านหมดเพราะ
ไม่มีใครเคยลอง parse ไฟล์นั้นเลย — LMDS ก็มีหน้าเว็บก้อนเดียวกันแบบนั้น
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

CONSOLE = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


def inline_scripts() -> list[str]:
    html = CONSOLE.read_text(encoding="utf-8")
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)


@pytest.mark.skipif(shutil.which("node") is None, reason="ต้องมี node")
def test_every_inline_script_parses(tmp_path):
    blocks = inline_scripts()
    assert blocks, "ไม่เจอ <script> เลย — regex หรือไฟล์เปลี่ยนไป เทสนี้จะกลายเป็นเทสเปล่า"
    for i, body in enumerate(blocks):
        f = tmp_path / f"block{i}.mjs"
        f.write_text(body, encoding="utf-8")
        done = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert done.returncode == 0, f"บล็อกที่ {i} parse ไม่ผ่าน:\n{done.stderr[:2000]}"


# ── ปุ่ม Fix docker access ของ node ──────────────────────────────────────────
#
# ลูกค้า cynbangkok เจอ 2026-09-22: pull ตายด้วย permission denied ที่ docker.sock
# hub มีปุ่มนี้มาตั้งแต่ 2026-09-05 แต่ node ไม่มี — ใช้หน้าเว็บอย่างเดียวแล้วตัน
def test_the_node_card_offers_fix_docker_access():
    html = CONSOLE.read_text(encoding="utf-8")
    assert 'data-nact="fix-docker"' in html


def test_the_form_routes_each_job_to_its_own_endpoint():
    """ฟอร์มเดียวใช้สามงาน — ถ้า routing พลาด รหัส sudo จะไปเรียกงานผิดตัว"""
    html = CONSOLE.read_text(encoding="utf-8")
    assert 'btn.dataset.endpoint || "setup"' in html
    for endpoint in ("fix-docker", "fix-permissions"):
        assert f'"{endpoint}"' in html
    # แบบเก่าที่เป็น boolean ต้องไม่เหลือ ไม่งั้นสองทางแย่งกันคุม
    assert "dataset.perms" not in html


def test_it_says_docker_login_will_not_help():
    """สาเหตุที่ทำปุ่มนี้คือมีคนถูกส่งไป docker login — หน้าจอต้องตัดทางนั้นทิ้งให้ชัด"""
    html = CONSOLE.read_text(encoding="utf-8")
    assert "does not fix it" in html
    assert "/var/run/docker.sock" in html


# ── docker ใช้ไม่ได้ต้องเห็นตั้งแต่การ์ดขึ้น ───────────────────────────────────
#
# ข้อมูล host.docker_access ส่งมาถึง hub ตั้งแต่ทักเครื่องครั้งแรก แต่เดิมแสดงเฉพาะ
# การ์ด hub — เครื่องอื่นตกสำรวจ ลูกค้าจึงรู้ตอนกด download แล้ว pull ล้มเท่านั้น
def _warning_fn() -> str:
    html = CONSOLE.read_text(encoding="utf-8")
    body = html[html.index("function dockerAccessWarning("):]
    return body[:body.index("\nfunction ")]


def test_the_node_card_warns_before_a_pull_can_fail():
    html = CONSOLE.read_text(encoding="utf-8")
    assert "dockerAccessWarning(name, host)" in html, "ต้องถูกเรียกตอนวาดการ์ดเครื่อง"


def test_it_stays_quiet_when_docker_works_or_the_node_is_too_old_to_say():
    """เครื่องรุ่นเก่าไม่ส่ง docker_access มา — ห้ามเดาว่าพังแล้วขึ้นเตือนมั่ว"""
    fn = _warning_fn()
    assert "!da.user || da.usable !== false" in fn
    # `!da.usable` เฉย ๆ จะจริงตอน undefined ด้วย ซึ่งคือ "ไม่รู้" ไม่ใช่ "พัง"
    assert "!da.usable)" not in fn


def test_the_button_only_appears_when_it_can_actually_help():
    """ปุ่มแก้เรื่องกลุ่ม docker — ถ้ายังไม่ได้ติดตั้ง docker กดไปก็ไม่ช่วย ต้องบอกคำสั่งติดตั้งแทน"""
    fn = _warning_fn()
    assert "da.installed === true" in fn
    assert 'data-nact="fix-docker"' in fn
    assert "da.fix" in fn, "กรณีกดปุ่มไม่ได้ ต้องยังบอกวิธีแก้"


def test_it_heads_off_the_wrong_conclusion_about_denied():
    """คำว่า denied ในข้อความของ docker คือเหตุที่ทำให้เข้าใจผิดว่าเป็นเรื่อง registry"""
    fn = _warning_fn()
    assert "not the registry" in fn
