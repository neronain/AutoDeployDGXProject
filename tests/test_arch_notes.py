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


def test_nvfp4_note_names_the_command_that_actually_fixes_it():
    """คำแนะนำเดิมชี้ให้ไป cutlass/b12x — ซึ่งเป็น path ที่ *พัง* บน sm_121 พอดี

    ยืนยันบน msi-6 2026-08-20: VLLM_NVFP4_GEMM_BACKEND=marlin ทำให้ engine ขึ้นได้จริง
    (ptxas error หายเกลี้ยง) ส่วน cutlass คือตัวที่ JIT แล้วตาย · คำเตือนต้องบอกคำสั่ง
    ที่กดตามแล้วได้ผล ไม่ใช่ชื่อ backend ให้ไปลองเอง
    """
    notes = _joined("Qwen/Qwen3.6-35B-A3B-NVFP4")
    assert "VLLM_NVFP4_GEMM_BACKEND=marlin" in notes
    assert "lmds set" in notes and "--engine-env" in notes
    # ยังต้องบอกราคาที่จ่าย ไม่ใช่ขายว่าฟรี
    assert "42%" in notes
    assert "VLLM_MARLIN_USE_ATOMIC_ADD" in notes


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
