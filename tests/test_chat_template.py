"""chat template ที่สูตรระบุ ต้องไปถึง vLLM จริง

catalog บอกมาตั้งแต่ต้นว่า Llama-3.3 ต้องใช้ tool_chat_template_llama3.1_json.jinja
เพราะ template ที่มากับโมเดลไม่ได้ออกรูปแบบ tool ที่ vLLM แปลงได้ · ค่าถูกเขียนลง
plan แล้วไม่มี controller ตัวไหนอ่าน — tool calling จึงเปิดไม่ติดถึงจะตั้ง parser ถูก
"""

from __future__ import annotations

import subprocess

import pytest

from tests.test_generator import gguf_report, make_bundle, safetensors_report  # noqa: F401


def _controller(tmp_path, **recipe_tools):
    from lmds.brain.plan_schema import DeploymentPlan
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    return bundle.controller.read_text(encoding="utf-8")


def test_the_controller_has_a_chat_template_knob(isolated_config, tmp_path):
    text = _controller(tmp_path)
    assert 'CHAT_TEMPLATE="${CHAT_TEMPLATE:-' in text
    assert "--chat-template" in text


def test_a_bare_template_name_is_resolved_to_the_image_path(isolated_config, tmp_path):
    """สูตรเขียนชื่อไฟล์เปล่า — vLLM ต้องได้ path ที่หาเจอในคอนเทนเนอร์"""
    text = _controller(tmp_path)
    assert "_resolve_chat_template" in text
    assert "/opt/vllm/vllm-src/examples/" in text


def test_a_full_path_is_left_alone(isolated_config, tmp_path):
    """คนที่ระบุ path เองต้องได้ตามนั้น ไม่ใช่โดนเติม prefix ทับ"""
    text = _controller(tmp_path)
    body = text[text.index("_resolve_chat_template() {"):]
    assert '"$value" == */*' in body.split("}")[0]


def test_the_generated_script_still_parses(isolated_config, tmp_path):
    from tests.test_generator import make_bundle as mb
    bundle, _, _ = mb(safetensors_report(), tmp_path=tmp_path)
    result = subprocess.run(["bash", "-n", str(bundle.controller)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_recipe_value_reaches_the_plan():
    """apply_recipe เคยเติมค่านี้ถูกอยู่แล้ว — กันไม่ให้หายอีก"""
    from lmds.brain.plan_schema import DeploymentPlan
    from lmds.brain.rulebased import apply_recipe
    from lmds.recipes import find_recipe

    recipe = find_recipe("meta-llama/Llama-3.3-70B-Instruct")
    assert recipe is not None, "สูตร Llama-3.3 หายไปจาก catalog"

    plan = DeploymentPlan.model_validate({
        "model_id": "meta-llama/Llama-3.3-70B-Instruct", "revision": "main",
        "served_model_name": "l", "artifact_type": "safetensors",
        "runtime": {"engine": "vllm", "image_ref": "x"}, "topology": "stacked",
        "serving": {"context": 65536, "max_output_tokens": 4096},
    })
    plan = apply_recipe(plan, recipe, "unified")
    assert plan.tool_calling.parser == "llama3_json"
    assert plan.tool_calling.chat_template_override == "tool_chat_template_llama3.1_json.jinja"
