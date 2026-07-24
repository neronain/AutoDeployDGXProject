"""เทส normalize/coalesce ของ extra_flags — กันบั๊กจริง:
llama.cpp ใหม่ --flash-attn ต้องมีค่า + LLM แยก flag/value เป็นคนละ item
"""

from __future__ import annotations

import pytest

from lmds.brain.allowlists import coalesce_flag_tokens, normalize_llamacpp_flags
from lmds.brain.orchestrator import harden_plan
from lmds.brain.rulebased import rule_based_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


@pytest.mark.parametrize("inp,exp", [
    (["--flash-attn"], ["--flash-attn on"]),
    (["-fa"], ["--flash-attn on"]),
    (["--flash-attn on"], ["--flash-attn on"]),
    (["--flash-attn=auto"], ["--flash-attn auto"]),
    (["--flash-attn=off"], ["--flash-attn off"]),
    (["--n-gpu-layers 99"], ["--n-gpu-layers 99"]),
])
def test_normalize_flash_attn(inp, exp):
    assert normalize_llamacpp_flags(inp) == exp


@pytest.mark.parametrize("inp,exp", [
    (["--threads", "4"], ["--threads 4"]),
    (["--flash-attn", "--threads", "4"], ["--flash-attn", "--threads 4"]),
    (["--enable-prefix-caching", "--enable-chunked-prefill"],
     ["--enable-prefix-caching", "--enable-chunked-prefill"]),  # flag ล้วน ไม่รวม
    (["--kv-cache-dtype", "fp8"], ["--kv-cache-dtype fp8"]),
    (["--n-gpu-layers 99"], ["--n-gpu-layers 99"]),  # มีค่าอยู่แล้ว ไม่แตะ
])
def test_coalesce_flag_tokens(inp, exp):
    assert coalesce_flag_tokens(inp) == exp


def _gguf_report():
    return ModelReport(
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        revision_sha="sha-fa",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=16 * GIB,
        selected_gguf="gemma-4-26B-A4B-it-Q4_K_M.gguf",
        context_length=131072,
        kv_dims=KvDims(layers=48, kv_heads=8, head_dim=128),
        has_chat_template=True,
    )


def test_harden_fixes_llm_flag_split_and_flash_attn():
    """เคสจริง: LLM ให้ ['--flash-attn','--threads','4'] → ต้องได้ค่าครบ ไม่หลุดไป needs_approval"""
    report = _gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit)
    plan.serving.extra_flags = ["--flash-attn", "--threads", "4", "--n-gpu-layers", "99"]
    hardened = harden_plan(plan, report, fit)
    assert hardened.serving.extra_flags == ["--flash-attn on", "--threads 4", "--n-gpu-layers 99"]
    assert hardened.flags_needing_approval == []


def test_rendered_controller_has_flash_attn_value(tmp_path):
    """สคริปต์ที่ render ต้องมี '--flash-attn on' ไม่ใช่ '--flash-attn' เปล่า (บั๊กที่ทำ start ล้ม)"""
    report = _gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit)
    plan.serving.extra_flags = ["--flash-attn"]
    plan = harden_plan(plan, report, fit)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "--flash-attn on" in text
    assert "SERVER_ARGS+=(--flash-attn)" not in text  # ต้องไม่มี bare
