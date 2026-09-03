"""ช่อง image min tokens ต้องตั้งได้จากหน้าเว็บ ไม่ใช่แค่ CLI

เคสจริง 2026-09-04: 5 bundle vision ที่ไม่ใช่ Qwen ทั้งฟลีต start ไม่ขึ้นเพราะ 1024 เกินเพดานของ
projector · คนหน้างานใช้หน้าเว็บล้วน — ถ้าแก้ได้แต่ทาง `lmds set` ก็เท่ากับแก้ไม่ได้
"""

from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


def test_settings_form_has_the_field_and_sends_it():
    page = INDEX.read_text(encoding="utf-8")
    assert 'class="n-image-min-tokens"' in page, "ฟอร์ม settings ยังไม่มีช่อง image min tokens"
    assert 'image_min_tokens: num(".n-image-min-tokens")' in page, "กรอกแล้วต้องถูกส่งไป PUT /settings"
    # โชว์เฉพาะ llama.cpp ที่มี projector — โมเดลข้อความล้วน/vLLM ไม่ควรเห็นช่องนี้
    form = page[page.index('class="n-image-min-tokens"') - 400:page.index('class="n-image-min-tokens"')]
    assert 'm.engine === "llamacpp" && m.projector' in form
