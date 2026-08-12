"""ความสามารถที่อ่านได้จากไฟล์ ก่อน deploy

สิ่งที่เทสจริง ๆ ไม่ใช่ "ตอบ yes ถูกไหม" แต่คือ **ตอบ yes ตอนที่ควรตอบ likely
หรือเปล่า** · yes ที่ควรเป็น likely คือคนไปวางแผนงานบนสิ่งที่ยังไม่ได้พิสูจน์
"""

from __future__ import annotations

from lmds.inspector.capabilities import detect

QWEN_TOOLS = """
{%- if tools %}
  <|im_start|>system
  {% for tool in tools %}{{ tool | tojson }}{% endfor %}
  <tool_call>
{%- endif %}
{%- for message in messages %}{% if message['role'] == 'system' %}...{% endif %}{% endfor %}
"""

PLAIN = """
{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}
"""

THINKING = PLAIN + "\n{% if add_generation_prompt %}<think>{% endif %}"


# ---------------------------------------------------------------------------
# tool calling — ความต่างที่สำคัญที่สุดในไฟล์นี้
# ---------------------------------------------------------------------------
def test_a_template_with_tools_is_likely_not_yes():
    """template รับ tool ได้ ไม่ได้แปลว่าเซิร์ฟเวอร์จะแปลงคำตอบเป็น tool_calls

    ต้องมี --tool-call-parser ที่ตรงตระกูลด้วย และพิสูจน์ได้ด้วยการยิงจริงเท่านั้น
    ตอบ yes ตรงนี้คือชวนให้คนไปวางแผนบนสิ่งที่ยังไม่ได้พิสูจน์
    """
    cap = detect({}, QWEN_TOOLS).get("tool_calling")
    assert cap.status == "likely"
    assert "tool-call-parser" in cap.caveat


def test_a_template_without_tools_is_a_definite_no():
    """อันนี้ตอบ no เต็มปากได้ — ไม่มีทางป้อน tool ให้โมเดลตั้งแต่ต้นทาง"""
    cap = detect({}, PLAIN).get("tool_calling")
    assert cap.status == "no"
    assert "ไม่มีอะไรให้ parse" in cap.caveat


def test_no_template_at_all_is_unknown_not_no():
    """ไม่มีข้อมูล กับ มีข้อมูลว่าไม่มี เป็นคนละเรื่อง"""
    assert detect({}, "").get("tool_calling").status == "unknown"


# ---------------------------------------------------------------------------
# vision
# ---------------------------------------------------------------------------
def test_a_vision_config_is_a_definite_yes():
    cap = detect({"vision_config": {"model_type": "qwen2_vl"}}, PLAIN).get("vision")
    assert cap.status == "yes"
    assert "qwen2_vl" in cap.evidence


def test_a_conditional_generation_architecture_is_only_likely():
    """ชื่อ architecture เป็นร่องรอย ไม่ใช่หลักฐาน"""
    cap = detect({"architectures": ["FooForConditionalGeneration"]}, PLAIN).get("vision")
    assert cap.status == "likely"


def test_a_text_only_config_is_a_definite_no():
    assert detect({"model_type": "llama"}, PLAIN).get("vision").status == "no"


def test_a_gguf_without_mmproj_cannot_see():
    cap = detect({"vision_config": {}}, PLAIN, has_mmproj=False).get("vision")
    assert cap.status == "no"
    assert "mmproj" in cap.evidence


def test_a_gguf_with_mmproj_still_needs_the_flag():
    cap = detect({}, PLAIN, has_mmproj=True).get("vision")
    assert cap.status == "yes"
    assert "--mmproj" in cap.caveat


# ---------------------------------------------------------------------------
# reasoning
# ---------------------------------------------------------------------------
def test_a_thinking_template_warns_about_the_parser():
    """โมเดลคิดได้ แต่ถ้าไม่ตั้ง parser ความคิดจะปนมาในคำตอบ"""
    cap = detect({}, THINKING).get("reasoning")
    assert cap.status == "likely"
    assert "reasoning-parser" in cap.caveat


def test_a_plain_template_does_not_think():
    assert detect({}, PLAIN).get("reasoning").status == "no"


# ---------------------------------------------------------------------------
# system prompt
# ---------------------------------------------------------------------------
def test_a_template_handling_system_says_yes():
    assert detect({}, QWEN_TOOLS).get("system_prompt").status == "yes"


def test_a_template_ignoring_system_says_what_happens():
    cap = detect({}, PLAIN).get("system_prompt")
    assert cap.status == "no"
    assert "ถูกเมิน" in cap.caveat


# ---------------------------------------------------------------------------
# ของเซิร์ฟเวอร์ ไม่ใช่ของโมเดล
# ---------------------------------------------------------------------------
def test_streaming_and_json_mode_are_not_model_properties():
    """vLLM และ llama.cpp ทำได้กับทุกโมเดล — บอกว่าโมเดลนี้ทำไม่ได้คือผิดตั้งแต่คำถาม"""
    report = detect({}, PLAIN, server="vllm")
    assert report.get("streaming").status == "server"
    assert report.get("json_mode").status == "server"
    assert "vLLM" in report.get("streaming").evidence


def test_json_mode_says_small_models_still_struggle():
    """บังคับรูปแบบได้ ไม่ได้แปลว่าเนื้อหาจะถูก"""
    assert "schema" in detect({}, PLAIN).get("json_mode").caveat


def test_every_capability_is_reported_even_with_no_input():
    """ตารางที่หายไปหนึ่งแถว อ่านเหมือน 'ไม่รองรับ' ทั้งที่แปลว่า 'ไม่รู้'"""
    names = set(detect().to_dict())
    assert names == {"vision", "tool_calling", "reasoning", "system_prompt",
                     "streaming", "json_mode"}
