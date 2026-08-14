"""SGLang เป็นทางเลือกที่สาม — safetensors เสิร์ฟได้ทั้ง vLLM และ SGLang

ที่ต้องมี engine นี้: checkpoint NVFP4 บางตระกูล (sparkarena/MiniMax-M3) calibrate ด้วย
w1/w3 scale ซึ่งรันถูกต้องเฉพาะบน SGLang · ถ้าไม่มีก็ต้องยกทั้งตระกูลออกจากระบบ

ธงทุกตัวในเทสนี้ยืนยันจาก `sglang serve --help` ของ scitrera/dgx-spark-sglang-mm:v0
ที่รันบน spark-head จริง ไม่ได้อ่านจากเอกสาร
"""

from __future__ import annotations

import subprocess

import yaml

from lmds.brain import build_plan
from lmds.brain.allowlists import _BY_ENGINE, KNOWN_IMAGE_REPOS
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/gpt-oss-120b",
        revision_sha="sha-pinned-123",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=61 * GIB,
        shard_count=9,
        context_length=131072,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=64),
        license="apache-2.0",
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def bundle(tmp_path, rep=None, engine=Engine.SGLANG, target="dgx-spark-single"):
    rep = rep or report()
    fit = analyze(rep, PRESETS[target])
    plan = build_plan(rep, fit, provider=None, engine=engine)
    return render_bundle(plan, rep, fit, tmp_path), plan


def controller(built) -> str:
    return built.controller.read_text()


# ── ธงที่ผู้ใช้เลือกต้องเดินไปถึงไฟล์ ────────────────────────────────────────
# ข้อนี้คือข้อที่จับบั๊กจริงตอนสร้างฟีเจอร์: gates เขียวทุกด่าน แต่ไฟล์ที่ออกมายังเป็น
# vLLM เพราะ engine หายกลางทางระหว่าง CLI กับ rule_based_plan

def test_choosing_sglang_actually_produces_an_sglang_controller(tmp_path):
    built, plan = bundle(tmp_path)
    assert plan.runtime.engine is Engine.SGLANG
    text = controller(built)
    assert "--entrypoint sglang" in text
    assert "--entrypoint vllm" not in text
    assert "VLLM_IMAGE" not in text


def test_the_bundle_records_which_engine_it_runs_on(tmp_path):
    """`lmds ps` กับหน้าเว็บอ่านค่านี้ · เขียนผิดคือทั้งระบบเข้าใจ bundle ผิดตัว"""
    built, _ = bundle(tmp_path)
    assert "engine=sglang" in controller(built)
    profile = built.directory / "MODEL_PROFILE.yaml"
    assert yaml.safe_load(profile.read_text())["runtime"]["engine"] == "sglang"


def test_the_launch_command_speaks_sglang_not_vllm(tmp_path):
    text = controller(bundle(tmp_path)[0])
    # ตัดคอมเมนต์ออกก่อน — ตารางเทียบชื่อธงกับ vLLM อยู่ในคอมเมนต์โดยตั้งใจ
    # สิ่งที่ต้องตรวจคือธงที่ "ถูกส่งจริง" ไม่ใช่คำที่ปรากฏในไฟล์
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    for flag in ("--model-path", "--context-length", "--mem-fraction-static",
                 "--max-running-requests", "--chunked-prefill-size"):
        assert flag in code, flag
    for flag in ("--max-model-len", "--gpu-memory-utilization", "--max-num-seqs",
                 "--enable-auto-tool-choice"):
        assert flag not in code, f"ธงของ vLLM หลุดมา: {flag}"


def test_the_knobs_keep_their_names_across_engines(tmp_path):
    """SGLang เรียกธงคนละชื่อ แต่ชื่อ knob ของ controller ต้องเหมือนเดิมทุก engine

    `lmds set`, bundle.env, หน้าเว็บ และธง --context/--gpu-util ใช้ชื่อพวกนี้ร่วมกัน
    เปลี่ยนชื่อเมื่อไหร่ --context จะไปตั้งตัวแปรที่ไม่มีใครอ่าน = พังเงียบ
    """
    text = controller(bundle(tmp_path)[0])
    for knob in ("MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS"):
        assert knob in text, knob
    assert '--context)        MAX_MODEL_LEN="$2"' in text


def test_the_generated_script_is_valid_bash(tmp_path):
    path = bundle(tmp_path)[0].controller
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


# ── คำขอที่เป็นไปไม่ได้ต้องถูกปฏิเสธ ไม่ใช่ทำตาม ─────────────────────────────

def test_gguf_never_becomes_sglang_however_hard_you_ask(tmp_path):
    """SGLang อ่าน GGUF ไม่ได้ · ยอมตามคำขอ = ส่ง bundle ที่ start ไม่ขึ้นให้"""
    from lmds.brain.rulebased import rule_based_plan

    gguf = report(artifact_type=ArtifactType.GGUF, selected_gguf="model-Q8_0.gguf",
                  weight_bytes=30 * GIB)
    fit = analyze(gguf, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(gguf, fit, Engine.SGLANG)
    assert plan.runtime.engine is Engine.LLAMACPP


def test_an_image_outside_the_list_is_not_accepted_for_sglang():
    assert "lmsysorg/sglang" in KNOWN_IMAGE_REPOS[Engine.SGLANG]
    assert "scitrera/dgx-spark-sglang-mm" in KNOWN_IMAGE_REPOS[Engine.SGLANG]
    assert "ghcr.io/ggml-org/llama.cpp" not in KNOWN_IMAGE_REPOS[Engine.SGLANG]


def test_sglang_has_its_own_flag_allowlist():
    """ธงของ vLLM ไม่ใช่ธงของ SGLang — ปล่อยผ่านคือ start แล้วตายด้วย unknown argument"""
    flags = _BY_ENGINE[Engine.SGLANG]
    assert "--moe-runner-backend" in flags
    assert "--fp4-gemm-backend" in flags
    assert "--gpu-memory-utilization" not in flags


def test_the_spark_gets_the_build_made_for_its_chip(tmp_path):
    """kernel ของ SM121 ต้องมากับ image ที่ build ให้เครื่องนี้ เหมือนกติกาของ vLLM"""
    _, plan = bundle(tmp_path)
    assert "nvcr.io/nvidia/sglang" in plan.runtime.image_ref


# ── หน้าเว็บต้องเลือกได้เท่ากับ CLI ─────────────────────────────────────────
# ตอนแรกฟีเจอร์นี้มีเฉพาะฝั่ง CLI · หน้าเว็บเรียก planner โดยไม่ส่ง engine เลย
# ผลคือคนที่ใช้หน้าเว็บไม่มีทางเลือก SGLang ได้ ทั้งที่ระบบรองรับแล้ว

def test_the_web_rejects_an_engine_it_does_not_know_before_touching_the_network():
    """พิมพ์ผิดต้องรู้ทันที ไม่ใช่หลังรอดึง metadata สามสิบวินาที"""
    import pytest

    from lmds.web import deploy

    with pytest.raises(deploy.DeployError) as caught:
        deploy.analyze("org/model", engine="nonsense")
    assert "nonsense" in str(caught.value)
    assert "sglang" in str(caught.value)


def test_the_web_and_the_cli_offer_the_same_engines():
    """สองทางเข้าที่ให้ตัวเลือกไม่เท่ากันคือกับดัก — คนหนึ่งทำได้ อีกคนทำไม่ได้"""
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "src/lmds/web/static/index.html").read_text()
    assert 'id="w-engine"' in page
    for value in ('value="vllm"', 'value="sglang"'):
        assert value in page, value
    # llama.cpp ต้องไม่อยู่ในรายการ — เลือกแล้วไม่มีผล เพราะ GGUF บังคับใช้มันอยู่แล้ว
    assert 'value="llamacpp"' not in page
