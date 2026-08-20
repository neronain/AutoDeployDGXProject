"""หน้ารายละเอียดคะแนน — ลูกค้าเอาไปใช้นำเสนอ ช่องที่ว่างเปล่าจึงมีราคา"""

import pytest

from lmds.cli.main import _quant_from_filename


@pytest.mark.parametrize("filename,expected", [
    ("Qwen3.8-27B-Uncensored-Q4_K_M.gguf", "Q4_K_M"),
    ("model-IQ3_XXS.gguf", "IQ3_XXS"),
    ("m-F16-00001-of-00002.gguf", "F16"),
    ("Muse-Glimmer-30B-UD-Q8_K_XL.gguf", "Q8_K_XL"),
])
def test_quant_read_from_the_gguf_filename(filename, expected):
    """profile ของ bundle GGUF ไม่ได้จด quantization ไว้ — ชื่อไฟล์บอกอยู่แล้ว"""
    assert _quant_from_filename(filename) == expected


@pytest.mark.parametrize("filename", ["plain.gguf", "", "model.safetensors"])
def test_unknown_quant_is_empty_not_a_guess(filename):
    assert _quant_from_filename(filename) == ""


def test_detail_view_is_wired_into_the_page():
    from pathlib import Path

    import lmds.web as web

    page = (Path(web.__file__).parent / "static/index.html").read_text(encoding="utf-8")
    assert "function benchDetailMarkup" in page
    assert 'id="bench-detail"' in page
    assert "data-bench-detail=" in page
    # ต้องปิดได้ ไม่งั้นค้างอยู่กับโมเดลตัวเดิม
    assert "data-bench-close" in page
    # ตัวเลขทุกแกนที่ลูกค้าอยากเห็น
    for panel in (">Decode <", ">Prefill <", ">TTFT <", ">Capability<"):
        assert panel in page, panel


def test_sections_can_be_folded():
    """หน้าเดียวมีหลายหมวดยาว ๆ — ต้องซ่อนของที่ยังไม่ดูได้ และต้องจำไว้ด้วย"""
    from pathlib import Path

    import lmds.web as web

    page = (Path(web.__file__).parent / "static/index.html").read_text(encoding="utf-8")
    assert 'data-fold="fleet"' in page and 'data-fold="bench"' in page
    assert "lmds-folded" in page, "สถานะพับต้องถูกจำไว้ ไม่ใช่กางกลับทุกครั้งที่รีเฟรช"
    assert "function applyFold" in page
