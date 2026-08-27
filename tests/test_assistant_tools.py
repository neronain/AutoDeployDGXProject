"""เทสของผู้ช่วยที่ลงไปดูเครื่องจริง — แคตตาล็อก, ตัวรัน, การอนุมัติ, ความรู้

สิ่งที่เทสชุดนี้ปกป้อง เรียงตามความเสียหายถ้าพลาด:

  1. **ค่าที่ LLM ส่งมาไม่กลายเป็นคำสั่ง** — slug ที่มี `;` หรือ `$( )` ต้องถูกปฏิเสธ
     ตั้งแต่ชั้นตรวจ ไม่ใช่ไปพึ่ง quote อย่างเดียว
  2. **งานที่เปลี่ยนเครื่องไม่ทำงานเองโดยไม่มีคนกด** — ตั๋วต้องมีโหมดก่อนถึงจะเดิน
  3. **action ผิดขั้นเดียวทิ้งทั้งชุด** — ขั้นถัดไปตั้งอยู่บนสมมติฐานว่าขั้นก่อนสำเร็จ
"""

from __future__ import annotations

import pytest

from lmds.assistant import catalog, knowledge, policy, router, runner


# ---------------------------------------------------------------------------
# แคตตาล็อก: ค่าที่ไม่น่าเชื่อถือต้องตายตั้งแต่ประตู
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("evil", [
    "a; rm -rf ~",
    "$(reboot)",
    "`id`",
    "../../etc/passwd",
    "a b",
    "x" * 200,
    "",
])
def test_a_slug_that_is_not_a_slug_is_refused(evil):
    with pytest.raises(catalog.ParamError):
        catalog.Param("slug", "slug").clean(evil)


def test_the_command_is_built_from_the_cleaned_values():
    """คำสั่งประกอบด้วยโค้ด — LLM ให้ได้แค่ค่าที่ผ่านการตรวจแล้ว

    ค่าที่ผ่านการตรวจไม่มีอักขระที่ต้อง quote เหลืออยู่แล้ว (`shlex.quote` จึงคืนค่าเดิม)
    การ quote เป็นชั้นสำรอง ไม่ใช่ด่านแรก — ด่านแรกคือ Param.clean
    """
    probe = catalog.PROBES["model_logs"]
    command, clean = probe.command({"slug": "qwen3-coder", "lines": "50"})
    assert clean == {"slug": "qwen3-coder", "lines": "50"}
    assert command.endswith('"$ctl" logs 50')
    assert "bundles/qwen3-coder" in command


def test_a_log_request_cannot_ask_for_the_whole_file():
    """log ทั้งไฟล์กิน context จนไม่เหลือที่ให้คำตอบ — บีบให้อยู่ในช่วงที่ใช้ได้"""
    assert catalog.Param("lines", "lines").clean("999999") == "400"
    assert catalog.Param("lines", "lines").clean("1") == "20"


def test_bind_and_ratio_stay_inside_what_the_controller_accepts():
    assert catalog.Param("bind", "bind").clean("127.0.0.1") == "127.0.0.1"
    with pytest.raises(catalog.ParamError):
        catalog.Param("bind", "bind").clean("example.com")
    assert catalog.Param("ratio", "ratio").clean("0.75") == "0.75"
    for bad in ("0", "1.5", "-0.2", "abc"):
        with pytest.raises(catalog.ParamError):
            catalog.Param("ratio", "ratio").clean(bad)


def test_a_survey_probe_survives_a_missing_tool(monkeypatch):
    """เครื่องที่ไม่มี nvidia-smi/docker/แคช ยังต้องได้ผลสำรวจ ไม่ใช่คำว่า "ล้ม"

    เจอจริงตอนทดสอบ: `disk` ขึ้นว่าล้มเพราะ `du` เจอ path ที่ยังไม่มี ทั้งที่ผล
    `df` ที่ต้องการอยู่ครบแล้ว — ผู้ช่วยเห็นธง "ล้ม" แล้วทิ้งข้อมูลที่ใช้ได้
    """
    for name in ("disk", "gpu", "docker", "system", "ports", "network", "bundles"):
        command, _ = catalog.PROBES[name].command({})
        assert command.endswith("|| true"), f"{name} ควรเป็น probe สำรวจที่ไม่ล้ม"


def test_every_action_tells_the_user_what_it_will_cost_them():
    """ผู้ใช้ต้องอ่านผลกระทบได้ก่อนกดอนุมัติ — action ที่ไม่มีข้อความนี้คือปุ่มที่กดมั่ว"""
    for action in catalog.ACTIONS.values():
        assert action.impact.strip(), f"{action.name} ไม่ได้บอกผลกระทบ"


# ---------------------------------------------------------------------------
# ตัวรัน: ล้มแบบรายงาน ไม่ใช่ล้มแบบโยน exception ใส่กล่องแชท
# ---------------------------------------------------------------------------
def test_an_unknown_probe_is_reported_not_raised():
    outcome = runner.run_probe("ไม่มีอยู่จริง")
    assert not outcome.ok
    assert "แคตตาล็อก" in outcome.error


def test_a_bad_parameter_never_reaches_the_shell(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(runner, "_run_local", lambda *a, **k: called.append(a) or (0, "", ""))

    outcome = runner.run_probe("model_logs", runner.LOCAL, {"slug": "a; reboot"})
    assert not outcome.ok
    assert not called, "คำสั่งไม่ควรถูกรันเลยเมื่อพารามิเตอร์ไม่ผ่าน"


def test_output_is_trimmed_and_secrets_are_masked(monkeypatch):
    monkeypatch.setattr(
        runner, "_run_local",
        lambda *a, **k: (0, "sk-" + "a" * 48 + "\n" + "z" * 20000, ""),
    )
    outcome = runner.run_probe("bundles")
    assert len(outcome.output) <= runner.MAX_OUTPUT_CHARS + 100
    assert "sk-" + "a" * 48 not in outcome.output


def test_a_probe_on_an_unknown_node_says_so(monkeypatch):
    monkeypatch.setattr("lmds.nodes.find", lambda name: None)
    outcome = runner.run_probe("overview", "เครื่องที่ไม่มีในทะเบียน")
    assert not outcome.ok
    assert "ไม่รู้จักเครื่อง" in outcome.error


# ---------------------------------------------------------------------------
# การอนุมัติ: จุดเดียวที่งานเปลี่ยนเครื่องเริ่มทำงานได้
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_tickets():
    policy.reset()
    yield
    policy.reset()


@pytest.fixture
def ran(monkeypatch):
    """จับว่ามีอะไรถูกสั่งรันจริงบ้าง โดยไม่แตะเครื่องจริง"""
    log: list[str] = []

    def fake(name, target="this", params=None):
        log.append(name)
        return runner.Outcome(name=name, title=name, target=target, exit_code=0, output="ok")

    monkeypatch.setattr(policy, "run_action", fake)
    return log


def test_a_proposal_runs_nothing_until_the_user_picks_from_the_menu(ran):
    ticket = policy.propose([{"action": "model_restart", "target": "this",
                              "params": {"slug": "demo"}}], why="ลองดู")
    assert ran == []
    assert ticket.mode == ""
    with pytest.raises(policy.PolicyError):
        policy.advance(ticket.id)
    assert ran == []


def test_the_user_sees_the_real_command_before_approving():
    ticket = policy.propose([{"action": "set_context", "target": "this",
                              "params": {"slug": "demo", "context": "32768"}}])
    step = ticket.payload()["steps"][0]
    assert "--context 32768" in step["command"]
    assert step["impact"]


def test_apply_runs_every_step_and_step_mode_stops_after_one(ran):
    steps = [
        {"action": "clear_fi_cache", "target": "this", "params": {"slug": "demo"}},
        {"action": "model_restart", "target": "this", "params": {"slug": "demo"}},
    ]

    stepwise = policy.propose(steps)
    policy.choose(stepwise.id, policy.STEP)
    policy.advance(stepwise.id)
    assert ran == ["clear_fi_cache"], "โหมดทีละขั้นต้องหยุดให้ดูผลก่อน"
    policy.advance(stepwise.id)
    assert ran == ["clear_fi_cache", "model_restart"]

    ran.clear()
    whole = policy.propose(steps)
    policy.choose(whole.id, policy.APPLY)
    policy.advance(whole.id)
    assert ran == ["clear_fi_cache", "model_restart"]


def test_hold_means_the_machine_is_never_touched(ran):
    ticket = policy.propose([{"action": "model_stop", "target": "this",
                              "params": {"slug": "demo"}}])
    policy.choose(ticket.id, policy.HOLD)
    policy.advance(ticket.id)
    assert ran == []


def test_a_failed_step_stops_the_rest(monkeypatch):
    log: list[str] = []

    def failing(name, target="this", params=None):
        log.append(name)
        return runner.Outcome(name=name, title=name, target=target,
                              exit_code=1, output="พัง")

    monkeypatch.setattr(policy, "run_action", failing)
    ticket = policy.propose([
        {"action": "clear_fi_cache", "target": "this", "params": {"slug": "demo"}},
        {"action": "model_restart", "target": "this", "params": {"slug": "demo"}},
    ])
    policy.choose(ticket.id, policy.APPLY)
    policy.advance(ticket.id)
    assert log == ["clear_fi_cache"], "ขั้นถัดไปตั้งอยู่บนสมมติฐานว่าขั้นก่อนสำเร็จ"


def test_a_mode_cannot_be_swapped_after_the_fact(ran):
    ticket = policy.propose([{"action": "model_start", "target": "this",
                              "params": {"slug": "demo"}}])
    policy.choose(ticket.id, policy.HOLD)
    with pytest.raises(policy.PolicyError):
        policy.choose(ticket.id, policy.APPLY)


def test_an_unknown_or_expired_ticket_is_refused(ran):
    with pytest.raises(policy.PolicyError):
        policy.get("ไม่เคยออกตั๋วนี้")

    ticket = policy.propose([{"action": "model_start", "target": "this",
                              "params": {"slug": "demo"}}])
    ticket.created -= policy.TICKET_TTL_SECONDS + 1
    with pytest.raises(policy.PolicyError):
        policy.get(ticket.id)


def test_a_proposal_with_an_unknown_action_is_refused():
    with pytest.raises(runner.RunError):
        policy.propose([{"action": "format_disk", "target": "this", "params": {}}])


# ---------------------------------------------------------------------------
# router: LLM เลือกได้เฉพาะของที่มีจริง
# ---------------------------------------------------------------------------
def test_only_catalog_entries_survive_validation():
    plan = router.validate({
        "probes": [
            {"name": "gpu", "target": "this"},
            {"name": "rm_rf", "target": "this"},
            {"name": "overview", "target": "เครื่องที่ไม่มี"},
        ],
        "docs": ["port ชน"],
    }, targets={"spark-head"})
    assert [p["name"] for p in plan.probes] == ["gpu"]
    assert plan.docs == ["port ชน"]


def test_an_action_is_all_or_nothing():
    plan = router.validate({
        "action": {"why": "x", "steps": [
            {"name": "model_restart", "target": "this", "params": {"slug": "demo"}},
            {"name": "ยิงจรวด", "target": "this", "params": {}},
        ]},
    }, targets=set())
    assert plan.action_steps == [], "ขั้นเดียวผิดต้องทิ้งทั้งชุด"


def test_a_probe_asking_for_an_impossible_slug_is_dropped():
    plan = router.validate(
        {"probes": [{"name": "doctor", "target": "this", "params": {"slug": "a;b"}}]},
        targets=set(),
    )
    assert plan.probes == []


def test_json_wrapped_in_a_fence_is_still_read():
    parsed = router._parse('```json\n{"probes": [{"name": "gpu"}]}\n```')
    assert parsed["probes"][0]["name"] == "gpu"


def test_a_provider_that_breaks_does_not_break_the_chat():
    class Broken:
        def complete_json(self, system, user):
            raise RuntimeError("โควตาหมด")

    plan = router.choose("เครื่องเป็นไง", [], [], provider=Broken())
    assert plan.empty
    assert "โควตาหมด" in plan.note


def test_the_router_prompt_lists_what_can_actually_be_chosen():
    prompt = router.build_prompt(["spark-head"], ["demo"])
    for name in ("model_logs", "doctor", "set_context"):
        assert name in prompt
    assert "spark-head" in prompt and "demo" in prompt


# ---------------------------------------------------------------------------
# ความรู้: วิธีคิดติดมากับแพ็กเกจ ส่วนข้อเท็จจริงค้นจากเอกสารจริง
# ---------------------------------------------------------------------------
def test_the_playbook_ships_with_the_package():
    text = knowledge.playbook()
    assert "หลักฐานชนะความจำ" in text


def test_docs_search_finds_a_real_section():
    hits = knowledge.search_docs("port")
    assert hits, "ควรหาหัวข้อที่พูดถึง port เจอในเอกสารจริง"
    assert all(hit["doc"].endswith(".md") for hit in hits)
    assert all(len(hit["text"]) <= knowledge.MAX_SECTION_CHARS + 20 for hit in hits)


def test_an_empty_query_returns_nothing():
    assert knowledge.search_docs("") == []
