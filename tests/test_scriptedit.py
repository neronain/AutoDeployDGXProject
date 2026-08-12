"""แก้ script ด้วย LLM — สิ่งที่เทสจริง ๆ คือ "มันปฏิเสธเมื่อไหร่"

ฟีเจอร์นี้เขียนไฟล์ที่รันโมเดลบนเครื่องที่มีคนใช้งานอยู่ · เทสที่พิสูจน์ว่ามันแก้ได้
สำเร็จมีค่าน้อยกว่าเทสที่พิสูจน์ว่ามันไม่ยอมแก้ตอนที่ยังไม่แน่ใจ
"""

from __future__ import annotations

import json

import pytest

from lmds.web import scriptedit

SCRIPT = '''#!/usr/bin/env bash
set -euo pipefail
API_PORT="${API_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"

start() {
  serve_args=(--port "$API_PORT" --max-model-len "$MAX_MODEL_LEN")
  docker run "${serve_args[@]}"
}

case "${1:-help}" in
  start)        start ;;
  stop)         stop ;;
  restart)      stop; start ;;
  test-text)    test_text ;;
  *)            usage ;;
esac
'''


def _script(content: str = SCRIPT) -> scriptedit.Script:
    return scriptedit.Script(
        slug="demo", path="/home/x/bundles/demo/demo-single.sh",
        content=content, commands=["restart", "start", "stop", "test-text"],
    )


class _Provider:
    """provider ปลอมที่ตอบตามที่เทสกำหนด — ไม่ยิงเน็ตจริงในเทส"""

    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user):
        self.system, self.user = system, user
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


@pytest.fixture
def brain(monkeypatch):
    """ผูก provider ปลอมเข้ากับ propose() แล้วคืน setter ให้เทสกำหนดคำตอบ"""
    from types import SimpleNamespace

    holder = {}

    def use(payload):
        holder["provider"] = _Provider(payload)
        return holder["provider"]

    monkeypatch.setattr(
        "lmds.config.Settings.load",
        staticmethod(lambda: SimpleNamespace(
            provider=SimpleNamespace(name=SimpleNamespace(value="openai-compat"), model="m")
        )),
    )
    monkeypatch.setattr("lmds.secrets.get_secret", lambda name: "")
    monkeypatch.setattr(
        "lmds.brain.providers.make_provider", lambda *a, **k: holder["provider"]
    )
    return use


# ---------------------------------------------------------------------------
# knob มาก่อนการแก้ไฟล์
# ---------------------------------------------------------------------------
def test_an_option_answer_needs_no_file_change(brain):
    """แก้ port ด้วย --port ได้ ก็ไม่ควรไปยุ่งกับไฟล์"""
    brain({"kind": "option", "explanation": "มี option อยู่แล้ว", "command": "restart --port 9000"})

    result = scriptedit.propose(_script(), "อยากเปลี่ยน port เป็น 9000")
    assert result["kind"] == "option"
    assert result["command"] == "restart --port 9000"
    assert "edits" not in result


def test_the_prompt_says_options_come_first(brain):
    provider = brain({"kind": "unsupported", "explanation": "ไม่รู้"})
    scriptedit.propose(_script(), "อะไรสักอย่าง")
    assert "อย่าเสนอแก้ไฟล์" in provider.system
    # และสคริปต์จริงต้องถูกส่งไปด้วย ไม่งั้น LLM เดาโครงสร้างเอาเอง
    assert "MAX_MODEL_LEN" in provider.user


def test_an_option_the_controller_does_not_have_is_refused(brain):
    """bundle เก่าไม่มีคำสั่งใหม่ — เสนอไปก็กดแล้วล้ม"""
    brain({"kind": "option", "explanation": "ใช้ตัวนี้", "command": "test-reasoning"})

    with pytest.raises(scriptedit.ScriptError, match="ไม่มี"):
        scriptedit.propose(_script(), "ตรวจ reasoning")


# ---------------------------------------------------------------------------
# การแก้ไฟล์ต้องยึดกับไฟล์จริง
# ---------------------------------------------------------------------------
def test_an_anchor_that_is_not_in_the_file_kills_the_whole_proposal(brain):
    """LLM แต่งข้อความเดิมขึ้นมาเอง = ไม่รู้ว่ากำลังแก้ตรงไหน"""
    brain({"kind": "edit", "explanation": "เพิ่ม flag",
           "edits": [{"find": 'serve_args=(--port "$API_PORT" --enforce-eager)',
                      "replace": "x", "why": ""}]})

    with pytest.raises(scriptedit.ScriptError, match="เจอ 0 ครั้ง"):
        scriptedit.propose(_script(), "เพิ่ม --enforce-eager")


def test_an_ambiguous_anchor_is_refused_rather_than_guessed(brain):
    """เจอสองที่แล้วเดาเอาว่าอันไหน คือวิธีแก้ไฟล์ผิดจุดแบบเงียบ ๆ"""
    doubled = SCRIPT + "\nstart() {\n  :\n}\n"
    brain({"kind": "edit", "explanation": "แก้",
           "edits": [{"find": "start() {", "replace": "start() {  # patched", "why": ""}]})

    with pytest.raises(scriptedit.ScriptError, match="เจอ 2 ครั้ง"):
        scriptedit.propose(_script(doubled), "แก้ start")


def test_a_valid_edit_comes_back_with_a_diff_to_read(brain):
    """คนต้องได้อ่าน diff ก่อนกด ไม่ใช่ได้แค่คำอธิบายของ LLM"""
    brain({"kind": "edit", "explanation": "เพิ่ม --enforce-eager",
           "edits": [{"find": '--max-model-len "$MAX_MODEL_LEN")',
                      "replace": '--max-model-len "$MAX_MODEL_LEN" --enforce-eager)',
                      "why": "ลด VRAM ตอน start"}]})

    result = scriptedit.propose(_script(), "เพิ่ม --enforce-eager")
    assert result["kind"] == "edit"
    assert "--enforce-eager" in result["preview"]
    assert "+" in result["diff"] and "--enforce-eager" in result["diff"]
    # แก้จุดเดียว ไม่ใช่เขียนไฟล์ใหม่ทั้งก้อน
    assert result["preview"].count("\n") == SCRIPT.count("\n")


def test_a_no_op_edit_is_refused(brain):
    brain({"kind": "edit", "explanation": "เหมือนเดิม",
           "edits": [{"find": 'GPU_UTIL="${GPU_UTIL:-0.90}"',
                      "replace": 'GPU_UTIL="${GPU_UTIL:-0.90}"', "why": ""}]})

    with pytest.raises(scriptedit.ScriptError, match="เหมือนกัน"):
        scriptedit.propose(_script(), "อะไรก็ได้")


def test_a_proposal_too_big_to_review_is_refused(brain):
    edits = [{"find": f"line{i}", "replace": "x", "why": ""} for i in range(20)]
    brain({"kind": "edit", "explanation": "ยกเครื่อง", "edits": edits})

    with pytest.raises(scriptedit.ScriptError, match="ใหญ่เกิน"):
        scriptedit.propose(_script(), "rewrite ทั้งไฟล์")


def test_a_reply_that_is_not_json_is_reported_not_guessed(brain):
    brain("ผมคิดว่าน่าจะแก้ตรง start นะครับ")

    with pytest.raises(scriptedit.ScriptError, match="ไม่ใช่ JSON"):
        scriptedit.propose(_script(), "แก้หน่อย")


def test_a_json_reply_wrapped_in_a_code_fence_still_parses(brain):
    """โมเดล local หลายตัวห่อ JSON ด้วย ``` แม้จะสั่งว่าอย่าห่อ"""
    brain('```json\n{"kind": "unsupported", "explanation": "ทำไม่ได้"}\n```')

    assert scriptedit.propose(_script(), "อะไรสักอย่าง")["kind"] == "unsupported"


# ---------------------------------------------------------------------------
# ตอนเขียนจริง
# ---------------------------------------------------------------------------
def test_applying_recomputes_from_the_file_as_it_is_now():
    """ไฟล์เปลี่ยนไปหลังเสนอ = preview เก่าจะเขียนทับงานของคนอื่น"""
    edits = [{"find": 'API_PORT="${API_PORT:-8001}"',
              "replace": 'API_PORT="${API_PORT:-9000}"', "why": ""}]
    changed = SCRIPT.replace('API_PORT="${API_PORT:-8001}"', 'API_PORT="${API_PORT:-7000}"')

    with pytest.raises(scriptedit.ScriptError, match="ไฟล์เปลี่ยนไป"):
        scriptedit.apply_edits(changed, edits)


def test_a_backup_is_written_before_the_file_is(tmp_path):
    target = tmp_path / "demo-single.sh"
    target.write_text(SCRIPT, encoding="utf-8")
    script = scriptedit.Script("demo", str(target), SCRIPT, ["start"])

    result = scriptedit.apply(
        script,
        [{"find": 'GPU_UTIL="${GPU_UTIL:-0.90}"',
          "replace": 'GPU_UTIL="${GPU_UTIL:-0.80}"', "why": ""}],
    )

    from pathlib import Path

    assert Path(result["backup"]).read_text(encoding="utf-8") == SCRIPT
    assert "0.80" in target.read_text(encoding="utf-8")


def test_a_syntax_error_is_never_written_to_disk(tmp_path):
    """สคริปต์ที่ syntax เสีย = โมเดลตัวนั้น start ไม่ขึ้นอีกเลย"""
    target = tmp_path / "demo-single.sh"
    target.write_text(SCRIPT, encoding="utf-8")
    script = scriptedit.Script("demo", str(target), SCRIPT, ["start"])

    with pytest.raises(scriptedit.ScriptError, match="syntax"):
        scriptedit.apply(
            script,
            [{"find": "case \"${1:-help}\" in", "replace": "case \"${1:-help}\" in\n  ((", "why": ""}],
        )

    assert target.read_text(encoding="utf-8") == SCRIPT


def test_commands_are_read_from_the_dispatch_table():
    """ปุ่มและการตรวจ option ต้องอิงสิ่งที่สคริปต์รองรับจริง ไม่ใช่รายการที่ hardcode ไว้"""
    assert scriptedit._commands_from_text(SCRIPT) == ["restart", "start", "stop", "test-text"]


def test_an_env_prefixed_command_is_accepted(brain):
    """`VAR=value ./ctl start` คือรูปที่เอกสารของ LMDS แนะนำเอง ไม่ใช่คำสั่งที่แต่งขึ้น"""
    brain({"kind": "option", "explanation": "ตั้งผ่าน env",
           "command": "VLLM_LOGGING_LEVEL=DEBUG ./demo-single.sh restart"})

    result = scriptedit.propose(_script(), "อยากเห็น log ละเอียดขึ้น")
    assert result["kind"] == "option"
    assert result["command"].startswith("VLLM_LOGGING_LEVEL=DEBUG")


def test_the_script_name_is_not_mistaken_for_the_command(brain):
    brain({"kind": "option", "explanation": "", "command": "./demo-single.sh restart --port 9000"})

    assert scriptedit.propose(_script(), "เปลี่ยน port")["kind"] == "option"


def test_env_values_alone_are_not_a_command(brain):
    brain({"kind": "option", "explanation": "", "command": "FOO=1 BAR=2"})

    with pytest.raises(scriptedit.ScriptError, match="ไม่มีคำสั่งให้รัน"):
        scriptedit.propose(_script(), "อะไรสักอย่าง")
