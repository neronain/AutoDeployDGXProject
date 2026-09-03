"""Deploy-time warnings for DGX Spark SM121 gotchas (from the Qwen3.5-122B research)."""

from __future__ import annotations

from lmds.brain.rulebased import arch_notes


def _joined(repo_id: str, quant: str = "", hybrid: bool = False) -> str:
    return " || ".join(arch_notes(repo_id, quant, hybrid_attention=hybrid))


def test_nvfp4_warns_about_fp4_kernel_image():
    assert "FP4 kernel" in _joined("nvidia/Llama-3.3-70B-Instruct-NVFP4")
    assert "FP4 kernel" in _joined("some/model", quant="nvfp4")


def test_nvfp4_says_it_can_fail_outright_not_just_run_slow():
    """เคสจริง 2026-08-20 บน msi-6: ptxas ปฏิเสธ cvt.e2m1x2 บน sm_121 แล้ว engine ตายก่อน health

    คำเตือนเดิมพูดถึงแต่ "fallback Marlin ช้ามาก" ซึ่งอ่านแล้วเข้าใจว่าอย่างแย่ก็แค่ช้า
    ผู้ใช้จึงกด deploy ไปแล้วเจอ container ตายโดยไม่รู้ว่าเกี่ยวกัน
    """
    notes = _joined("some/model-NVFP4")
    assert "e2m1x2" in notes
    assert "sm_121" in notes


def test_hybrid_detected_from_files_warns_even_when_the_name_is_new():
    """Qwen3.8 เป็น hybrid เหมือน Qwen3.5 แต่ชื่อไม่ตรง — คำเตือนที่ผูกกับชื่อพลาดรุ่นใหม่เสมอ"""
    by_name = _joined("orcarouter/Qwen3.8-27B-Uncensored")
    assert "prefix-caching" not in by_name

    detected = _joined("orcarouter/Qwen3.8-27B-Uncensored", hybrid=True)
    assert "prefix-caching" in detected


def test_qwen35_warns_prefix_caching_and_tool_parser():
    notes = _joined("Intel/Qwen3.5-122B-A10B-int4-AutoRound")
    assert "prefix-caching" in notes
    assert "qwen3_xml" in notes


def test_qwen35_suggests_mtp_speculative_decoding():
    notes = _joined("Intel/Qwen3.5-122B-A10B-int4-AutoRound")
    assert "MTP" in notes
    assert "speculative" in notes


def test_qwen3_coder_gets_no_xml_hint():
    notes = _joined("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    # Coder must not be told to use qwen3_xml
    assert "qwen3_xml" not in notes


def test_gemma4_is_told_to_use_its_own_parser():
    """เคสจริง msi-2 (2026-09-02): deploy ด้วย --tool-call-parser hermes

    LMDS ไม่เคยแนะ Gemma ไว้ คนจึงใส่ hermes ซึ่งเป็นค่าที่คนมักหยิบมาใช้เป็นค่าเริ่มต้น
    ผลคือโมเดลถูกใช้งานมาเป็นสัปดาห์โดยไม่มีใครรู้ว่า tool calling พังอยู่
    """
    notes = _joined("google/gemma-4-31B-it")
    assert "gemma4" in notes
    assert "hermes" in notes  # ต้องบอกด้วยว่าตัวไหนคือตัวที่ผิด ไม่ใช่บอกแต่ตัวที่ถูก


def test_gemma4_note_says_the_failure_is_silent():
    """คำเตือนที่ไม่บอกว่า "พังแบบเงียบ" จะถูกข้าม เพราะ deploy แล้วดูเหมือนสำเร็จ

    vLLM ขึ้นปกติ /health เขียว ตอบ 200 ทุก request — สัญญาณเดียวที่มีคือ
    finish_reason ที่ควรเป็น tool_calls กลับเป็น stop
    """
    notes = _joined("google/gemma-4-31B-it")
    assert "finish_reason" in notes
    assert "tool_calls" in notes


def test_non_gemma_models_do_not_get_the_gemma_note():
    for repo in ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "nvidia/Llama-3.3-70B-Instruct-NVFP4"):
        assert "gemma4" not in _joined(repo)


def test_nvfp4_moe_note_names_the_working_recipe_and_the_failing_one():
    """เคยสรุปว่า MoE+NVFP4 บน sm_121 เป็นทางตัน (msi-6, 2026-08-20) — ผิด

    2026-09-03 บน spark-head: Qwen3-Coder-Next-NVFP4-GB10 (MoE 512 expert) รันได้ 61 tok/s
    บน cu130-nightly + env marlin ครบชุด · ที่ msi-6 ล้มเพราะขาด VLLM_USE_FLASHINFER_MOE_FP4=0
    ไม่ใช่เพราะทางตัน · คำเตือนต้องบอกทั้งสูตรที่ผ่านและเงื่อนไขที่ล้ม ไม่ใช่บอกให้เลิก
    """
    notes = _joined("Qwen/Qwen3.6-35B-A3B-NVFP4")
    assert "61 tok/s" in notes and "VLLM_USE_FLASHINFER_MOE_FP4=0" in notes
    assert "e2m1x2" in notes and "sm_121" in notes   # เคสที่ล้มยังต้องอยู่ ให้คนรู้จักอาการ
    assert "ทางตัน" not in notes and "ไม่ช่วย" not in notes


def test_nemotron_3x_warns_hybrid_mamba():
    notes = _joined("nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    assert "Mamba" in notes
    assert "--mamba-backend" in notes


def test_deepseek_v4_serving_notes():
    notes = _joined("deepseek-ai/DeepSeek-V4-Flash-0731")
    assert "ds_mla" in notes  # kv-cache dtype nuance
    assert "dspark" in notes  # spec method, not plain mtp
    assert "block-size 256" in notes


def test_plain_model_has_no_spurious_notes():
    assert arch_notes("meta-llama/Llama-3.3-70B-Instruct") == []


def test_qwen36_gets_the_hybrid_warnings_from_its_name_alone():
    """เดิม 3.6 เข้าเงื่อนไขได้ทางเดียวคือ hybrid_attention ที่มาจากการอ่านไฟล์

    inspect ที่อ่านไฟล์ไม่ครบ (เน็ตสะดุด / repo ไม่มี config) จะได้ plan ที่เงียบสนิท
    ทั้งที่รู้จากชื่อได้อยู่แล้วว่าเป็นตระกูล DeltaNet
    """
    notes = _joined("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    assert "prefix-caching" in notes


def test_mtp_note_carries_both_runtimes_flags():
    """เคสจริง spark-02 (2026-09-03): plan เป็น GGUF/llama.cpp แต่โน้ตยกแต่แฟล็กของ vLLM

    ผู้ใช้อ่านแล้วเอาไปใช้ไม่ได้ — llama.cpp ไม่มี --speculative-config
    """
    notes = _joined("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    assert "--spec-type draft-mtp" in notes
    assert "--speculative-config" in notes


def test_mtp_vllm_method_stays_the_current_name():
    """vLLM deprecate ชื่อเจาะจงรุ่นแล้ว (qwen3_next_mtp -> mtp)

    model card ของ Qwen ยังเขียนชื่อเก่าอยู่ · ถ้า LMDS ลอกตามจะพาคนไปหา
    deprecation warning โดยไม่จำเป็น
    """
    notes = _joined("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    assert '"method":"mtp"' in notes


def test_qwen_tool_note_says_the_two_names_are_the_same_parser():
    """เคยเข้าใจผิดว่า 3.6 ต้องใช้ qwen3_coder เท่านั้น — จริง ๆ สองชื่อ map ไปคลาสเดียวกัน

    คำเตือนต้องบอกให้ชัด ไม่งั้นคนจะไล่แก้ของที่ไม่ได้พัง และ --reasoning-parser
    ที่ขาดจริง ๆ จะถูกมองข้าม
    """
    notes = _joined("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    assert "qwen3_xml" in notes and "qwen3_coder" in notes
    assert "ตัวเดียวกัน" in notes
    assert "--reasoning-parser qwen3" in notes
