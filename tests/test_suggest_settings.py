"""ระบบเติมค่า settings ให้ตามโมเดล — คนที่ไม่รู้ต้องไม่ต้องเดาชื่อ parser เอง

ผู้ใช้ 2026-09-04: "ทำให้ระบบกรอกให้เองตาม model ได้ไหม … กลัวใส่ผิด แล้วไม่มีให้ใช้งาน"
ความรู้มีอยู่แล้วใน arch_notes (ข้อความ) และ recipes (ค่า) — เทสชุดนี้ยึดว่ามันไหลไปถึงช่องกรอก
"""

import pytest

from lmds.brain.families import NVFP4_SM121_ENGINE_ENV, nvfp4_on_sm121, parsers_for
from lmds.brain.orchestrator import _VLLM_REASONING_PARSERS, _VLLM_TOOL_PARSERS, _SGLANG_REASONING_PARSERS, _SGLANG_TOOL_PARSERS
from lmds.fleet.suggest import suggest_settings


@pytest.mark.parametrize("model_id, engine, tool, reasoning", [
    ("unsloth/Qwen3.6-35B-A3B-NVFP4", "vllm", "qwen3_xml", "qwen3"),
    ("Qwen/Qwen3-32B", "vllm", "qwen3_xml", "qwen3"),
    ("Qwen/Qwen3-32B", "sglang", "qwen", "qwen3"),           # SGLang ใช้คนละชุดชื่อ
    ("ucbye/Qwen3-Coder-Next-NVFP4-GB10", "vllm", "qwen3_coder", None),
    ("google/gemma-4-31b-it", "vllm", "gemma4", None),
    ("llmfan46/gemma-4-31B-it-uncensored-heretic-NVFP4-GGUF", "llamacpp", None, None),  # llama.cpp ไม่มีแฟล็กนี้
    ("meta-llama/Llama-3.3-70B-Instruct", "vllm", None, None),  # ไม่รู้ = ไม่เดา
])
def test_family_parsers(model_id, engine, tool, reasoning):
    c = parsers_for(model_id, "", engine)
    assert (c.tool, c.reasoning) == (tool, reasoning)


def test_every_family_parser_survives_hardening():
    """ชื่อที่เราเติมให้ต้องอยู่ในรายการที่ engine รู้จัก — ไม่งั้น _harden_parsers ตัดทิ้งเงียบ ๆ"""
    for mid in ("Qwen/Qwen3-32B", "Qwen/Qwen3-Coder-480B", "google/gemma-4-12b-it"):
        v = parsers_for(mid, "", "vllm")
        assert v.tool is None or v.tool in _VLLM_TOOL_PARSERS, (mid, v.tool)
        assert v.reasoning is None or v.reasoning in _VLLM_REASONING_PARSERS, (mid, v.reasoning)
        s = parsers_for(mid, "", "sglang")
        assert s.tool is None or s.tool in _SGLANG_TOOL_PARSERS, (mid, s.tool)
        assert s.reasoning is None or s.reasoning in _SGLANG_REASONING_PARSERS, (mid, s.reasoning)


def test_nvfp4_hint_only_for_vllm_on_unified():
    h = nvfp4_on_sm121("unsloth/Qwen3.6-35B-A3B-NVFP4", "", "vllm", "unified")
    assert h.image and h.engine_env == NVFP4_SM121_ENGINE_ENV
    assert nvfp4_on_sm121("unsloth/Qwen3.6-35B-A3B-NVFP4", "", "vllm", "discrete").image is None
    assert nvfp4_on_sm121("unsloth/Qwen3.6-35B-A3B-NVFP4", "", "llamacpp", "unified").image is None
    assert nvfp4_on_sm121("Qwen/Qwen3-32B", "", "vllm", "unified").image is None


def test_suggest_for_the_real_bundle_on_veerasiam():
    """qwen3-6-35b-a3b-nvfp4 (vLLM · GB10) — เคสที่ผู้ใช้ถามถึงตรง ๆ"""
    s = suggest_settings("unsloth/Qwen3.6-35B-A3B-NVFP4", "vllm",
                         architecture="Qwen3_5MoeForConditionalGeneration", memory_model="unified")
    v = s["values"]
    assert v["tool_parser"] == "qwen3_xml" and v["reasoning_parser"] == "qwen3"
    # 2026-09-04: image community ถูกใส่ให้อัตโนมัติแล้ว start ล้ม (ไม่รู้จัก qwen3_5_moe) ทั้งที่ image เดิมรันได้
    # → เหลือเป็นหมายเหตุให้คนตัดสินใจ ไม่ใส่เป็นค่า
    assert "image" not in v and "engine_env" not in v
    assert any("avarok/dgx-vllm-nvfp4-kernel" in n and "marlin" in n for n in s["notes"])
    assert all(s["sources"].get(k) for k in v)   # ทุกค่าบอกที่มา


def test_recipe_beats_family_rule():
    """สูตรที่รันผ่านจริงชนะกฎตระกูล — Coder-Next มี recipe ระบุ qwen3_coder + image ของมันเอง"""
    s = suggest_settings("ucbye/Qwen3-Coder-Next-NVFP4-GB10", "vllm", memory_model="unified")
    assert s["values"]["tool_parser"] == "qwen3_coder"
    assert s["sources"]["tool_parser"].startswith("สูตรที่รันผ่านจริง")


def test_unknown_family_suggests_nothing_and_says_so():
    s = suggest_settings("acme/mystery-model-7b", "vllm", memory_model="discrete")
    assert s["values"] == {}
    assert any("ไม่เดา" in n for n in s["notes"])


def test_a_family_we_do_not_know_can_still_come_from_a_recipe():
    """Llama ไม่อยู่ในกฎตระกูล แต่มีสูตรในคลัง — ต้องได้ค่าจากสูตร ไม่ใช่ว่าง"""
    s = suggest_settings("meta-llama/Llama-3.3-70B-Instruct", "vllm", memory_model="discrete")
    assert s["values"].get("tool_parser") == "llama3_json"
    assert s["sources"]["tool_parser"].startswith("สูตรที่รันผ่านจริง")


def test_llamacpp_bundle_gets_no_parser_fields():
    s = suggest_settings("unsloth/gemma-4-12b-it-GGUF", "llamacpp", memory_model="unified")
    assert "tool_parser" not in s["values"] and "reasoning_parser" not in s["values"]


def test_llamacpp_vision_bundles_get_image_min_tokens_by_family():
    """เคส 2026-09-04: Gemma-4 ต้อง auto · Qwen-VL ต้อง 1024 · ไม่มี projector = ไม่เสนอ"""
    g = suggest_settings("unsloth/gemma-4-12b-it-GGUF", "llamacpp", memory_model="unified", projector=True)
    assert g["values"]["image_min_tokens"] == "auto"
    q = suggest_settings("unsloth/Qwen3-VL-8B-Instruct-GGUF", "llamacpp", memory_model="unified", projector=True)
    assert q["values"]["image_min_tokens"] == "1024"
    t = suggest_settings("unsloth/Qwen3-8B-GGUF", "llamacpp", memory_model="unified", projector=False)
    assert "image_min_tokens" not in t["values"]
