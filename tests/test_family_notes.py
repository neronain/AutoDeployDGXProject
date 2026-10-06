"""คำเตือนของตระกูลหนึ่งต้องไม่ไปโผล่ในแผนของอีกตระกูล

เคสจริง 2026-10-06 (audit · hub 6e2b474): `lmds plan openai/gpt-oss-120b --target rtx-4090 --no-llm` พิมพ์
- โน้ต NVFP4 / SM121 / Marlin — เครื่องเป็น RTX และ quant เป็น **mxfp4** (`"fp4" in quant` กิน mxfp4 ด้วย)
- "Qwen3.5 (DeltaNet hybrid attention): อย่าเปิด --enable-prefix-caching" กับ "Qwen3.5/3.6 มี MTP head …
  --speculative-config" — เพราะ `layer_types` ที่ปน `full_attention` กับ `sliding_attention` ถูกนับเป็น hybrid
เหมือนกันกับ `openai/gpt-oss-20b` และ `unsloth/gemma-3n-E4B-it`
"""

import json

import pytest
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from tests.real_hub import RealHub, flat

runner = CliRunner()

GPT_OSS_120B = "openai/gpt-oss-120b"
GPT_OSS_20B = "openai/gpt-oss-20b"
GEMMA_3N = "unsloth/gemma-3n-E4B-it"
QWEN_NVFP4 = "Inferact/Qwen3.8-27B-NVFP4"          # qwen3_5: linear_attention สลับ full_attention + NVFP4 ของแท้

QWEN35_ONLY = ("DeltaNet", "MTP head", "--enable-prefix-caching")
NVFP4_ONLY = ("FP4 kernel", "Marlin", "sm_121")


def warnings_of(repo: str, target: str) -> str:
    report = RealHub(repo).inspect(repo)
    return " || ".join(build_plan(report, analyze(report, PRESETS[target]), None).warnings)


@pytest.mark.parametrize("repo, full_layers", [(GPT_OSS_120B, 18), (GPT_OSS_20B, 12), (GEMMA_3N, 7)])
def test_sliding_window_models_are_not_hybrid_but_keep_their_kv_sizing(repo, full_layers):
    """sliding-window ยังเป็น attention ปกติ ไม่ใช่ linear/SSM — แต่ KV ที่โตตาม context มีแค่ layer full_attention
    (gpt-oss-120b: 36 layer → 18) จึงต้องยังนับเท่าเดิม ไม่งั้นประเมิน KV เกินจริงสองเท่า"""
    report = RealHub(repo).inspect(repo)
    assert report.hybrid_attention is False
    assert report.kv_dims.layers == full_layers


@pytest.mark.parametrize("repo", [GPT_OSS_120B, GPT_OSS_20B, GEMMA_3N])
@pytest.mark.parametrize("target", ["rtx-4090", "dgx-spark-single"])
def test_sliding_window_and_mxfp4_models_get_no_qwen35_or_nvfp4_advice(repo, target):
    said = warnings_of(repo, target)
    for phrase in (*QWEN35_ONLY, *NVFP4_ONLY):
        assert phrase not in said, f"{phrase!r} ไม่ใช่เรื่องของ {repo}"


def test_a_real_linear_attention_nvfp4_model_still_gets_both_on_a_dgx_spark():
    report = RealHub(QWEN_NVFP4).inspect(QWEN_NVFP4)
    assert report.hybrid_attention is True
    assert report.kv_dims.layers == 16, "64 layer · full attention ทุก ๆ 4"

    said = warnings_of(QWEN_NVFP4, "dgx-spark-single")
    for phrase in (*QWEN35_ONLY, *NVFP4_ONLY):
        assert phrase in said


def test_the_sm121_kernel_note_is_not_shown_for_a_discrete_gpu_target():
    """FP4 kernel ของ sm_121 เป็นเรื่องของ GB10 — บน RTX โน้ตนี้ไม่เกี่ยว ส่วนโน้ตของสถาปัตยกรรม (DeltaNet) ยังต้องอยู่"""
    said = warnings_of(QWEN_NVFP4, "rtx-5090")
    assert not any(phrase in said for phrase in NVFP4_ONLY)
    assert all(phrase in said for phrase in QWEN35_ONLY)


@pytest.mark.parametrize("quant, expected", [
    ("mxfp4", False), ("nvfp4", True), ("fp4", True), ("nvfp4-pack-quantized", True), ("fp8", False), ("", False),
])
def test_only_nvfp4_quantization_triggers_the_nvfp4_note(quant, expected):
    from lmds.brain.rulebased import arch_notes

    said = " || ".join(arch_notes("org/some-model", quant, memory_model="unified"))
    assert ("FP4 kernel" in said) is expected


def test_cli_plan_for_gpt_oss_on_an_rtx_no_longer_prints_other_families_advice(isolated_config, monkeypatch):
    fake = RealHub(GPT_OSS_120B)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    result = runner.invoke(app, ["plan", GPT_OSS_120B, "--target", "rtx-4090", "--no-llm"])
    assert result.exit_code == 0, result.output
    said = flat(result.output)
    assert "DeltaNet" not in said and "MTPhead" not in said and "NVFP4" not in said

    result = runner.invoke(app, ["plan", GPT_OSS_120B, "--target", "rtx-4090", "--no-llm", "--json"])
    assert not any("Qwen3.5" in w for w in json.loads(result.stdout)["warnings"])
