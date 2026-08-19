"""ตัววัดต้องไม่โกหก — เคสที่รอบวัดจริงเคยจับได้ ต้องไม่กลับมาอีก"""

import struct

import pytest

from lmds.bench import capability, score, workloads
from lmds.bench.runner import Sample, WorkloadResult


# ── ชุดงาน ─────────────────────────────────────────────────────────────────
def test_prompt_length_lands_near_target():
    """เคยเล็งด้วย 0.75 token/คำ ทั้งที่ของจริง 1.31 — งาน 512 ยิงไป 893"""
    for workload in workloads.FULL:
        if workload.kind == "code":
            continue
        words = len(workload.prompt().split())
        estimated = words * 1.3
        assert 0.6 * workload.input_tokens <= estimated <= 1.4 * workload.input_tokens, workload.key


def test_nonce_goes_at_the_very_front():
    """nonce ต้องอยู่หัว prompt ไม่งั้น prefix cache ยังใช้ต่อได้ทั้งก้อน"""
    workload = workloads.FULL[0]
    assert workload.prompt("abc").startswith("[อ้างอิง abc]")
    assert workload.prompt("abc") != workload.prompt("xyz")


def test_workloads_too_long_for_the_context_are_dropped():
    chosen = workloads.select("full", 4096)
    assert all(w.input_tokens + w.output_tokens + 512 <= 4096 for w in chosen)
    assert workloads.select("full", 0) == workloads.FULL


# ── การคิดเลข ──────────────────────────────────────────────────────────────
def test_decode_excludes_the_first_token():
    """token แรกถูกนับใน TTFT แล้ว — ไม่ลบออกจะได้ตัวเลขสูงเกินจริง"""
    sample = Sample(ttft_s=1.0, decode_s=10.0, total_s=11.0,
                    prompt_tokens=100, completion_tokens=101)
    assert sample.decode_tps == pytest.approx(10.0)


def test_single_token_reply_reports_zero_not_infinity():
    sample = Sample(ttft_s=1.0, decode_s=0.0, total_s=1.0,
                    prompt_tokens=100, completion_tokens=1)
    assert sample.decode_tps == 0.0


def test_median_not_mean():
    """รอบแรกมักช้ากว่า — ค่าเฉลี่ยจะถูกมันดึงลงทั้งชุด"""
    result = WorkloadResult("k", "l", 512, samples=[
        Sample(1.0, 10.0, 11.0, 100, 101),   # 10 tok/s
        Sample(1.0, 10.0, 11.0, 100, 101),   # 10 tok/s
        Sample(1.0, 100.0, 101.0, 100, 101),  # 1 tok/s — รอบที่ช้าผิดปกติ
    ])
    assert result.as_dict()["decode_tps"] == pytest.approx(10.0)


def test_cached_tokens_are_surfaced():
    result = WorkloadResult("k", "l", 512, samples=[
        Sample(1.0, 10.0, 11.0, 100, 101, cached_tokens=40),
        Sample(1.0, 10.0, 11.0, 100, 101, cached_tokens=60),
    ])
    assert result.as_dict()["cached_tokens"] == 100


# ── คะแนน ──────────────────────────────────────────────────────────────────
def _probe(key, passed, skipped=False):
    return {"key": key, "label": key, "passed": passed, "detail": "", "skipped": skipped}


def test_skipped_probes_are_not_counted_against_the_model():
    """โมเดลที่ไม่มี mmproj ไม่ควรโดนหักคะแนนเรื่องภาพ"""
    probes = [_probe("instructions", True), _probe("tools", True),
              _probe("vision", False, skipped=True)]
    result = score.capability_score(probes)
    assert result["score"] == 100
    assert result["counted"] == 2
    assert result["skipped"] == ["vision"]


def test_failing_tools_hurts_more_than_failing_vision():
    with_tools = score.capability_score([_probe("tools", False), _probe("vision", True)])
    with_vision = score.capability_score([_probe("tools", True), _probe("vision", False)])
    assert with_tools["score"] < with_vision["score"]


def test_score_names_what_failed():
    result = score.capability_score([_probe("tools", False), _probe("thai", True)])
    assert result["failed"] == ["tools"]
    assert result["passed"] == ["thai"]


def test_no_probes_gives_no_score_not_zero():
    """ไม่มีข้อที่วัดได้ ≠ ได้ศูนย์ — ศูนย์แปลว่าสอบตกทุกข้อ"""
    assert score.capability_score([])["score"] is None


def test_speed_summary_separates_short_from_long():
    workload_rows = [
        {"key": "a", "target_input": 512, "decode_tps": 30.0, "ttft_s": 0.5},
        {"key": "b", "target_input": 8192, "decode_tps": 10.0, "ttft_s": 4.0},
    ]
    summary = score.speed_summary(workload_rows)
    assert summary["decode_tps_avg"] == pytest.approx(20.0)
    assert summary["decode_tps_long"] == pytest.approx(10.0)
    assert summary["longest_context"] == 8192
    assert summary["ttft_s_short"] == pytest.approx(0.5)


def test_speed_summary_survives_a_total_failure():
    summary = score.speed_summary([{"key": "a", "target_input": 512, "error": "timeout"}])
    assert summary["decode_tps_avg"] is None
    assert summary["failed"] == 1


# ── ภาพทดสอบ ───────────────────────────────────────────────────────────────
def test_test_image_is_big_enough_for_a_vision_encoder():
    """ภาพ 8x8 ถูก preprocess ทิ้งเงียบ ๆ — ตัววัดเคยกล่าวหาว่าเซิร์ฟเวอร์ไม่รองรับภาพ"""
    png = capability._RED_PNG
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert width >= 224 and height >= 224


def test_answer_separates_wrong_from_out_of_budget():
    """"ตอบผิด" กับ "ไม่ทันได้ตอบ" คนละเรื่อง — โมเดล reasoning เคยถูกตัดสินว่าโง่เพราะข้อนี้"""
    exhausted = {"choices": [{"finish_reason": "length",
                              "message": {"content": "", "reasoning_content": "คิดอยู่…"}}]}
    text, blocked = capability._answer(exhausted)
    assert text == ""
    assert "งบ token หมด" in blocked

    answered = {"choices": [{"finish_reason": "stop", "message": {"content": "ปารีส"}}]}
    assert capability._answer(answered) == ("ปารีส", "")


def test_slow_thinker_gets_a_second_chance(monkeypatch):
    """โมเดลที่คิดยาวจนงบหมด ต้องได้ลองอีกครั้งด้วยงบที่ใหญ่ขึ้น ไม่ใช่ถูกตัดสินว่าทำไม่ได้"""
    budgets = []

    def fake_chat(client, endpoint, model, **body):
        budgets.append(body["max_tokens"])
        if len(budgets) == 1:
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": "", "reasoning_content": "คิด" * 900}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"city": "กรุงเทพ"}'}}]}

    monkeypatch.setattr(capability, "_chat", fake_chat)
    text, blocked = capability._ask(None, "http://x/v1", "m", [{"role": "user", "content": "hi"}])
    assert blocked == ""
    assert text == '{"city": "กรุงเทพ"}'
    assert budgets[1] == budgets[0] * 3


def test_second_chance_is_not_infinite(monkeypatch):
    calls = []

    def always_out_of_budget(client, endpoint, model, **body):
        calls.append(body["max_tokens"])
        return {"choices": [{"finish_reason": "length",
                             "message": {"content": "", "reasoning_content": "คิด"}}]}

    monkeypatch.setattr(capability, "_chat", always_out_of_budget)
    text, blocked = capability._ask(None, "http://x/v1", "m", [{"role": "user", "content": "hi"}])
    assert text == ""
    assert len(calls) == 2
    assert "แม้ให้งบ" in blocked
