"""เครื่องที่ "ตรวจไม่ได้" ต้องไม่ถูกรายงานเหมือนเครื่องที่ "ไม่ตรง"

เคสจริง (2026-09-21, อัปเดตทั้งฟลีตจากหน้าเว็บ) · `msi-3` ขึ้นว่า

    does not match the hub yet — runtime could not be checked (llama.cpp … ·
    ตรวจไม่ได้ pid-…: profile เก่าไม่มี gguf_architecture และยังไม่ download)
    (code ✓ · controllers ✓ · runtime ?)

code ✓ · controllers ✓ · **ไม่มีอะไรผิดสักอย่าง** — runtime เช็คไม่ได้เพราะ weight
ยังไม่ถูก download ซึ่งไม่ใช่ความผิดของเครื่อง · แต่มันขึ้นเรียงอยู่กลางเครื่องอื่นอีก 15 ตัว
ที่เขียนว่า "matches the hub" คนกวาดตาดูจะอ่านว่าเครื่องนี้เสีย

ฝั่ง server แยกไว้ถูกแล้วด้วย `Verdict.level` (`bad` = มีอะไรผิดจริง · `warn` = ยังไม่รู้)
— **หน้าเว็บทิ้งค่านั้นแล้วพิมพ์ข้อความเดียวกันทั้งสองแบบ**

หลักเดียวกับที่ `lmds burn` ยึดไว้แล้ว: "ไม่เกี่ยว" กับ "วัดไม่ได้" ไม่นับเป็นล้มเหลว ·
คอลัมน์ที่ร้องหมาป่าทุกแถว คือคอลัมน์ที่ไม่มีใครอ่านภายในสัปดาห์เดียว แล้วเครื่องที่พังจริง
จะถูกมองข้ามตอนนั้น

เทสชุดนี้เรียก `verdictText()` ตัวจริงจากหน้าเว็บ ไม่ได้เขียนตรรกะซ้ำในเทส
"""

from __future__ import annotations

import json

from tests.test_console_shell import run_scenario

FLEET = """const fx = { nodes: [{ name: "spark-01", site: "TKC", models: [] }], localModels: [] };
H.fx = fx;
H.routes = [...H.defaultRoutes(fx)];
"""

OK = {"state": "ok", "detail": "", "items": []}
NA = {"state": "n/a", "detail": "", "items": []}


def _ask(tmp_path, verdict, lang: str = "en") -> str:
    """ถาม `verdictText()` ตัวจริงว่าจะพูดว่าอะไรกับคำตัดสินก้อนนี้"""
    (out,) = run_scenario(tmp_path, FLEET, f"""
        await H.tick(4);
        // สลับด้วยปุ่มจริงเหมือนที่ผู้ใช้กด ไม่ใช่ยัดค่าตัวแปรเอง
        while (window.LANG !== {json.dumps(lang)}) {{ document.getElementById("lang").click(); await H.tick(6); }}
        console.log(JSON.stringify({{ text: verdictText({json.dumps(verdict)}), errors: H.errors }}));
    """)
    assert out["errors"] == []
    return out["text"]


def test_a_machine_that_only_could_not_be_checked_is_not_called_a_mismatch(tmp_path):
    """หัวใจของไฟล์นี้ · code ✓ controllers ✓ runtime ? = ไม่มีอะไรผิด"""
    text = _ask(tmp_path, {
        "code": OK, "controllers": OK,
        "runtimes": {"state": "unknown", "detail": "weights not downloaded yet", "items": []},
        "consistent": False, "level": "warn",
    })
    assert "does not match" not in text, text
    assert "as far as it could be checked" in text
    assert "could not be checked" in text and "weights not downloaded yet" in text
    # เครื่องหมายยังต้องบอกความจริงว่ามิติไหนยังไม่รู้
    assert "runtime ?" in text and "code ✓" in text


def test_a_real_mismatch_still_says_so(tmp_path):
    """อีกด้านของเหรียญ — ผ่อนคำให้ของที่ผิดจริงคือการทำให้คอลัมน์นี้ไร้ความหมาย"""
    text = _ask(tmp_path, {
        "code": {"state": "behind", "detail": "hub is at abc1234", "items": []},
        "controllers": OK, "runtimes": OK, "consistent": False, "level": "bad",
    })
    assert "does not match the hub yet" in text
    assert "code ✗" in text


def test_a_machine_that_is_both_wrong_and_unmeasurable_is_reported_as_wrong(tmp_path):
    """มีอะไรผิดจริงแม้ข้อเดียว = ไม่ตรง · ที่ตรวจไม่ได้ตามมาทีหลังในประโยคเดียวกัน"""
    text = _ask(tmp_path, {
        "code": OK,
        "controllers": {"state": "stale", "detail": "", "items": ["demo"]},
        "runtimes": {"state": "unknown", "detail": "no weights", "items": []},
        "consistent": False, "level": "bad",
    })
    assert "does not match the hub yet" in text
    assert "demo" in text and "could not be checked" in text


def test_a_fully_matching_machine_is_unchanged(tmp_path):
    text = _ask(tmp_path, {"code": OK, "controllers": OK, "runtimes": NA,
                           "consistent": True, "level": "ok"})
    assert "matches the hub" in text
    # n/a ไม่ใช่ความล้มเหลว — ต้องขึ้น ✓ เหมือน ok
    assert "runtime ✓" in text


def test_the_whole_line_follows_the_language_of_the_console(tmp_path):
    """ในภาพหน้าจอที่เจอ ประโยคเดียวมีทั้งอังกฤษและไทยปนกัน เพราะกรอบนอกไม่ได้ผ่าน `T()`
    ส่วน `detail` มาจาก server เป็นไทย · คอนโซลสองภาษาแล้ว ข้อความนี้ต้องตามภาษาด้วย"""
    text = _ask(tmp_path, {
        "code": OK, "controllers": OK,
        "runtimes": {"state": "unknown", "detail": "ยังไม่ download", "items": []},
        "consistent": False, "level": "warn",
    }, lang="th")
    assert "ตรวจไม่ได้" in text
    assert "could not be checked" not in text and "as far as" not in text
