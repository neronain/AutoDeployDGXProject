"""test-tools ต้องวัดโหมดที่ agent ใช้จริง ไม่ใช่โหมดที่ผ่านง่าย

เคสจริง 2026-08-14 — Nemotron-3-Super บน spark-head: `test-tools` รายงาน PASS
แต่ Claude Code เห็นเป็นข้อความเปล่า เพราะเทสยิงด้วย `tool_choice: "required"`
ซึ่ง vLLM บังคับรูปแบบให้ด้วย guided decoding ผลจึงเป็น JSON ที่ parser อ่านออก
เสมอ ไม่ว่า --tool-parser จะตรงกับโมเดลหรือไม่

agent ทุกตัวส่ง "auto" มา โมเดลจึงเขียนตามรูปแบบของมันเอง (ตัวนั้นเขียน XML แบบ
Qwen) แล้ว parser `hermes` แปลไม่ออก · เทสที่ผ่านทั้งที่ของจริงพัง แย่กว่าไม่มีเทส
เพราะมันทำให้เลิกสงสัย
"""

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(tmp_path) -> str:
    """safetensors -> vLLM · tool parser มีอยู่ใน engine นี้ ไม่ใช่ llama.cpp"""
    report = ModelReport(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="sha-pinned-123",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB,
        shard_count=17,
        context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
        has_chat_template=True,
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    return next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")


def test_the_default_run_measures_auto(tmp_path):
    """ค่าเริ่มต้นต้องครอบคลุม auto — คนกด `test-tools` เฉย ๆ ต้องได้คำตอบที่ใช้ได้"""
    text = _controller(tmp_path)

    assert 'test_tools "${2:-both}"' in text
    assert 'test_tools "${2:-required}"' not in text, "required อย่างเดียวคือเทสที่ผ่านง่ายเกินจริง"


def test_a_failure_under_auto_is_a_failure(tmp_path):
    """auto ไม่ผ่าน = agent ใช้ไม่ได้ ต้อง exit ไม่ใช่ WARN แล้วจบ 0"""
    text = _controller(tmp_path)
    block = text[text.index("test_tools() {") : text.index("PYEOF\n}", text.index("test_tools() {"))]

    assert "FAIL(auto)" in block
    assert "WARN(auto)" not in block, "เตือนแล้วคืน 0 = สคริปต์อัตโนมัติจะเดินต่อทั้งที่พัง"
    assert "sys.exit(1)" in block


def test_the_raw_shape_is_printed_when_it_fails(tmp_path):
    """บอกว่าโมเดลเขียนอะไรออกมาจริง ไม่งั้นต้องไปไล่ลอง parser ทีละตัวเอง"""
    block = _controller(tmp_path)
    assert "qwen3_xml" in block and "hermes" in block and "pythonic" in block


def test_the_engine_is_asked_for_parser_names(tmp_path):
    """ชื่อไฟล์ qwen3xml.py แต่ชื่อที่ลงทะเบียนคือ qwen3_xml — เดาจากซอร์สแล้วพัง"""
    text = _controller(tmp_path)

    assert "list_parsers() {" in text
    assert "parsers)         list_parsers ;;" in text
    assert "lazy" in text, "registry ปกติว่างเปล่า ชื่อจริงอยู่ใน lazy registry"


def test_reasoning_accepts_both_field_names(tmp_path):
    """vLLM เปลี่ยนชื่อฟิลด์ระหว่างรุ่น — ดูชื่อเดียวคือรายงานผิดว่า parser ไม่ทำงาน"""
    block = _controller(tmp_path)

    assert 'msg.get("reasoning_content") or msg.get("reasoning")' in block


def test_an_empty_reasoning_is_not_reported_as_a_broken_parser(tmp_path):
    """ไม่มี chain-of-thought อาจแปลว่าโมเดลไม่ได้คิด ไม่ใช่ว่า parser ผิด"""
    text = _controller(tmp_path)
    block = text[text.index("test_reasoning() {"):text.index("PYEOF\n}", text.index("test_reasoning() {"))]

    assert "ไม่ใช่หลักฐานว่า parser ผิด" in block
    assert "</think>" in block, "ต้องแยกเคสที่โมเดลพ่น think ออกมาจริงจากเคสที่ไม่ได้คิดเลย"
