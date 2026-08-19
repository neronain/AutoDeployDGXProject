"""Deploy-time warnings for DGX Spark SM121 gotchas (from the Qwen3.5-122B research)."""

from __future__ import annotations

from lmds.brain.rulebased import arch_notes


def _joined(repo_id: str, quant: str = "") -> str:
    return " || ".join(arch_notes(repo_id, quant))


def test_nvfp4_warns_about_fp4_kernel_image():
    assert "FP4 CUTLASS" in _joined("nvidia/Llama-3.3-70B-Instruct-NVFP4")
    assert "FP4 CUTLASS" in _joined("some/model", quant="nvfp4")


def test_qwen35_warns_prefix_caching_and_tool_parser():
    notes = _joined("Intel/Qwen3.5-122B-A10B-int4-AutoRound")
    assert "prefix-caching" in notes
    assert "qwen3_xml" in notes


def test_qwen3_coder_gets_no_xml_hint():
    notes = _joined("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    # Coder must not be told to use qwen3_xml
    assert "qwen3_xml" not in notes


def test_plain_model_has_no_spurious_notes():
    assert arch_notes("meta-llama/Llama-3.3-70B-Instruct") == []
