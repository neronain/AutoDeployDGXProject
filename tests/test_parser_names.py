"""ชื่อ parser ที่ engine ไม่รู้จัก = start ไม่ขึ้นเลย — ต้องจับตั้งแต่ตอนวางแผน

เคสจริง 2026-09-01 บน spark-worker: LLM ที่วางแผน (qwen3-coder ที่รันอยู่บน msi-1)
เสนอ `--tool-call-parser qwen25` + `--reasoning-parser qwen25` ให้โมเดล Qwen3.5 MoE
ไม่มีใครตรวจกับรายชื่อจริง · container ตายทันทีที่ start หลังโหลดน้ำหนัก 122B ครบแล้ว

    KeyError: 'invalid tool call parser: qwen25'
    KeyError: "Reasoning parser 'qwen25' not found"

ที่ทำให้พลาดง่ายคือ **qwen25 เป็นชื่อจริงของ SGLang** แค่ไม่ใช่ของ vLLM — สองเครื่องยนต์
ใช้คนละชุด และ LLM จำข้ามกัน · แถม LLM ยังใส่ flag เดิมซ้ำใน extra_flags อีกชุด
ทำให้ vLLM เตือน "Found duplicate keys"
"""

import pytest

from lmds.brain.orchestrator import _harden_parsers
from lmds.brain.plan_schema import DeploymentPlan, Engine, RuntimeChoice, Serving
from lmds.inspector.report import ArtifactType


def _plan(engine=Engine.VLLM, tool=None, reasoning=None, extra=None) -> DeploymentPlan:
    p = DeploymentPlan(
        model_id="org/m", revision="sha", served_model_name="m",
        artifact_type=ArtifactType.SAFETENSORS,
        runtime=RuntimeChoice(engine=engine, image_ref="img:1"), serving=Serving(context=8192),
    )
    if tool:
        p.tool_calling.enabled, p.tool_calling.parser = True, tool
    if reasoning:
        p.reasoning.enabled, p.reasoning.parser = True, reasoning
    p.serving.extra_flags = list(extra or [])
    return p


def test_a_sglang_parser_name_on_vllm_is_dropped_and_explained():
    p = _plan(tool="qwen25", reasoning="qwen25")
    _harden_parsers(p)
    assert p.tool_calling.parser is None and p.tool_calling.enabled is False
    assert p.reasoning.parser is None
    msg = " ".join(p.warnings)
    assert "qwen25" in msg and "SGLang" in msg, msg
    assert "TOOL_CALL_PARSER" in msg, "ต้องบอกวิธีตั้งเองด้วย"


def test_the_same_name_is_fine_on_sglang():
    """qwen25 ถูกต้องสำหรับ SGLang — ห้ามตัดทิ้งเพราะ vLLM ไม่มี"""
    p = _plan(engine=Engine.SGLANG, tool="qwen25")
    _harden_parsers(p)
    assert p.tool_calling.parser == "qwen25"
    assert not p.warnings


def test_a_name_nobody_knows_is_dropped_too():
    p = _plan(tool="totally_made_up")
    _harden_parsers(p)
    assert p.tool_calling.parser is None
    assert "ไม่รู้จักชื่อนี้" in " ".join(p.warnings)


@pytest.mark.parametrize("name", ["qwen3_coder", "qwen3_xml", "hermes", "glm47"])
def test_valid_vllm_names_pass_through_untouched(name):
    p = _plan(tool=name)
    _harden_parsers(p)
    assert p.tool_calling.parser == name and p.tool_calling.enabled is True
    assert not p.warnings


def test_duplicate_flags_from_the_llm_are_removed():
    """LLM ใส่ทั้งใน plan และใน extra_flags — controller ส่งเองอยู่แล้ว จึงไปสองชุด"""
    # รูปที่ LLM ส่งมาจริง: ค่ารวมอยู่ในสตริงเดียว
    p = _plan(tool="qwen3_xml",
              extra=["--enable-auto-tool-choice", "--tool-call-parser qwen3_xml",
                     "--max-num-seqs 8"])
    _harden_parsers(p)
    assert p.serving.extra_flags == ["--max-num-seqs 8"]
    assert "duplicate" in " ".join(p.warnings).lower()


def test_duplicate_flags_split_across_two_elements_also_removed():
    """อีกรูปที่เจอได้: flag กับค่าถูกแยกเป็นคนละอิลิเมนต์"""
    p = _plan(tool="qwen3_xml",
              extra=["--tool-call-parser", "qwen3_xml", "--max-num-seqs", "8"])
    _harden_parsers(p)
    assert p.serving.extra_flags == ["--max-num-seqs", "8"]


def test_llamacpp_is_left_alone():
    """llama.cpp ไม่มี parser แบบนี้ — อย่าไปยุ่ง"""
    p = _plan(engine=Engine.LLAMACPP, tool="whatever")
    _harden_parsers(p)
    assert p.tool_calling.parser == "whatever"
    assert not p.warnings
