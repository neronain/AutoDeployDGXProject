"""เทสชั้นเว็บของผู้ช่วยที่ลงไปดูเครื่องได้

กฎที่เทสไว้: **ไม่มีทางอื่นเลยที่งานเปลี่ยนสภาพเครื่องจะเริ่มทำงาน นอกจากมีคนกดปุ่ม**
ตั๋วออกโดยเซิร์ฟเวอร์ตอนเสนอ และเดินได้ก็ต่อเมื่อ endpoint ถูกเรียกพร้อมโหมดที่เลือก
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.assistant import policy, runner  # noqa: E402
from lmds.web import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def clean_tickets():
    policy.reset()
    yield
    policy.reset()


@pytest.fixture
def ran(monkeypatch):
    log: list[str] = []

    def fake(name, target="this", params=None):
        log.append(name)
        return runner.Outcome(name=name, title=name, target=target, exit_code=0, output="ok")

    monkeypatch.setattr(policy, "run_action", fake)
    return log


def _ticket() -> str:
    return policy.propose(
        [{"action": "model_restart", "target": "this", "params": {"slug": "demo"}}],
        why="โมเดลค้าง",
    ).id


def test_an_unknown_ticket_is_a_404(ran):
    r = TestClient(create_app()).get("/api/assistant/ticket/ไม่มีจริง")
    assert r.status_code == 404


def test_the_ticket_shows_the_command_and_the_menu(ran):
    body = TestClient(create_app()).get(f"/api/assistant/ticket/{_ticket()}").json()
    assert body["why"] == "โมเดลค้าง"
    assert body["mode"] == ""
    assert [m["mode"] for m in body["menu"]] == ["apply", "step", "hold"]
    assert '"$ctl" restart' in body["steps"][0]["command"]
    assert ran == []


def test_hold_records_the_choice_without_touching_the_machine(ran):
    client = TestClient(create_app())
    body = client.post(f"/api/assistant/ticket/{_ticket()}/choose",
                       json={"mode": "hold"}).json()
    assert body["mode"] == "hold"
    assert ran == []


def test_apply_runs_the_work_only_after_the_button(ran):
    client = TestClient(create_app())
    ticket = _ticket()
    assert ran == []
    body = client.post(f"/api/assistant/ticket/{ticket}/choose", json={"mode": "apply"}).json()
    assert ran == ["model_restart"]
    assert body["finished"] is True
    assert body["steps"][0]["result"]["ok"] is True


def test_step_mode_waits_for_a_second_press(ran):
    client = TestClient(create_app())
    ticket = policy.propose([
        {"action": "clear_fi_cache", "target": "this", "params": {"slug": "demo"}},
        {"action": "model_restart", "target": "this", "params": {"slug": "demo"}},
    ]).id

    body = client.post(f"/api/assistant/ticket/{ticket}/choose", json={"mode": "step"}).json()
    assert ran == ["clear_fi_cache"]
    assert body["finished"] is False

    body = client.post(f"/api/assistant/ticket/{ticket}/advance").json()
    assert ran == ["clear_fi_cache", "model_restart"]
    assert body["finished"] is True


def test_a_made_up_mode_is_refused(ran):
    r = TestClient(create_app()).post(
        f"/api/assistant/ticket/{_ticket()}/choose", json={"mode": "แก้ให้หมดเลย"})
    assert r.status_code == 400
    assert ran == []


def test_advancing_before_choosing_is_refused(ran):
    r = TestClient(create_app()).post(f"/api/assistant/ticket/{_ticket()}/advance")
    assert r.status_code == 400
    assert ran == []


# ---------------------------------------------------------------------------
# สตรีมของแชท: หลักฐานและตั๋วเดินทางมากับคำตอบ
# ---------------------------------------------------------------------------
def _events(raw: str) -> list[dict]:
    out = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if chunk and chunk != "[DONE]":
                out.append(json.loads(chunk))
    return out


@pytest.fixture
def chatty(monkeypatch):
    """ผู้ช่วยที่พร้อมคุย — ตั้ง provider จริงในแซนด์บ็อกซ์ แต่ไม่ยิงออกเน็ตจริง

    ตั้งผ่าน endpoint เดียวกับที่ผู้ใช้ใช้ แทนการ monkeypatch `available()` — ด่าน
    "มีสมองให้คุยไหม" จึงถูกทดสอบไปด้วย ไม่ใช่ถูกข้าม · openai-compat ชี้เข้าบ้าน
    จึงไม่ต้องมี API key
    """
    from lmds.web import assistant

    TestClient(create_app()).put("/api/provider", json={
        "name": "openai-compat", "model": "local", "base_url": "http://127.0.0.1:9/v1",
    })
    monkeypatch.setattr(assistant, "build_messages", lambda history, evidence=None, context=None: ("sys", []))

    class Fake:
        def stream_chat(self, system, messages):
            yield "ดูให้แล้วครับ"

    monkeypatch.setattr("lmds.brain.providers.make_provider", lambda *a, **k: Fake())
    return assistant


def test_the_stream_says_what_it_checked(chatty, monkeypatch):
    monkeypatch.setattr(chatty, "investigate", lambda question, state=None, context=None: {
        "probes": [{"name": "gpu", "title": "GPU", "target": "this", "ok": True,
                    "output": "ความลับยาว ๆ", "params": {}}],
        "docs": [{"query": "port ชน", "sections": []}],
        "note": "",
    })
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat",
                                 json={"messages": [{"role": "user", "content": "เครื่องเป็นไง"}]}).text)

    checked = next(e["evidence"] for e in events if "evidence" in e)
    assert checked["probes"][0]["title"] == "GPU"
    assert checked["docs"] == ["port ชน"]
    # ผลดิบของ probe ไม่ต้องเดินทางไปหน้าเว็บ — มันไปอยู่ใน prompt แล้ว
    assert "output" not in checked["probes"][0]
    assert any(e.get("delta") for e in events)


def test_a_proposal_reaches_the_page_as_a_ticket(chatty, monkeypatch, ran):
    ticket = policy.propose([{"action": "model_restart", "target": "this",
                              "params": {"slug": "demo"}}], why="ค้าง")
    monkeypatch.setattr(chatty, "investigate",
                        lambda question, state=None, context=None: {"probes": [], "docs": [],
                                                      "ticket": ticket.payload()})
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat",
                                 json={"messages": [{"role": "user", "content": "แก้ให้ที"}]}).text)

    sent = next(e["ticket"] for e in events if "ticket" in e)
    assert sent["ticket"] == ticket.id
    assert sent["mode"] == ""
    assert ran == [], "ส่งข้อเสนอมาแสดงผล ไม่ใช่ลงมือทำ"


def test_a_broken_investigation_still_answers(chatty, monkeypatch):
    def explode(question, state=None, context=None):
        raise RuntimeError("ssh ล่ม")

    monkeypatch.setattr(chatty, "investigate", explode)
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat",
                                 json={"messages": [{"role": "user", "content": "hi"}]}).text)

    checked = next(e["evidence"] for e in events if "evidence" in e)
    assert "ssh ล่ม" in checked["note"]
    assert any(e.get("delta") for e in events), "ตรวจไม่ได้ ≠ ตอบไม่ได้"


# ---------------------------------------------------------------------------
# ผู้ช่วยแบบ operator: บริบทของการ์ด · คำแนะนำถัดไป · ยืนยันงานลบถาวร · สมองจากฟลีต
# ---------------------------------------------------------------------------
def test_the_context_of_the_open_card_travels_with_the_question(chatty, monkeypatch):
    seen: dict = {}

    def investigate(question, state=None, context=None):
        seen["context"] = context
        return {"probes": [{"name": "last_failure", "title": "x", "target": "spark-head", "ok": True,
                            "params": {"slug": "nemo"}, "output": ""}], "docs": [], "context": context}

    def build(history, evidence=None, context=None):
        seen["built_with"] = context
        return "sys", []

    monkeypatch.setattr(chatty, "investigate", investigate)
    monkeypatch.setattr(chatty, "build_messages", build)
    monkeypatch.setattr(chatty, "clean_context", lambda raw, state=None: {"node": "spark-head", "slug": "nemo"})
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat", json={
        "messages": [{"role": "user", "content": "ตัวนี้ทำไมไม่ขึ้น"}],
        "context": {"node": "spark-head", "slug": "nemo"},
    }).text)

    assert seen["context"] == {"node": "spark-head", "slug": "nemo"}
    assert seen["built_with"] == {"node": "spark-head", "slug": "nemo"}
    suggest = next(e["suggest"] for e in events if "suggest" in e)
    assert suggest and "nemo" in suggest[0], "ชิป 'ทำอะไรได้ต่อ' ต้องผูกกับสิ่งที่เพิ่งดู"


def test_the_status_carries_capabilities_and_says_whose_brain_it_is(chatty, monkeypatch):
    monkeypatch.setattr("lmds.assistant.brain.owner", lambda provider, snapshot=None: {"node": "spark-head", "slug": "nemo"})
    body = TestClient(create_app()).get("/api/assistant").json()
    assert body["available"] is True
    assert body["brain"]["provider"] == "openai-compat"
    assert body["brain"]["from_fleet"] == {"node": "spark-head", "slug": "nemo"}
    assert [g["group"] for g in body["capabilities"]["groups"]] == ["ตรวจ", "แก้", "ตั้งค่า", "deploy", "แนะนำโมเดล"]
    assert body["capabilities"]["actions"] >= 20


def test_a_running_fleet_model_becomes_the_brain_through_the_provider_settings(monkeypatch):
    from types import SimpleNamespace

    from lmds.web import state

    snap = {"host": {"data": {"models": []}},
            "nodes": {"spark-head": {"data": {"models": [
                {"slug": "nemo", "model_id": "nvidia/Nemotron", "engine": "vllm", "port": 8000, "running": True,
                 "features": "tools", "served_name": "nemotron-3"},
                {"slug": "asleep", "model_id": "x/y", "engine": "vllm", "port": 8010, "running": False},
            ]}}}}
    monkeypatch.setattr(state.STORE, "snapshot", lambda: snap)
    monkeypatch.setattr("lmds.nodes.find", lambda name: SimpleNamespace(local_ip="192.168.10.30", host="spark-head")
                        if name == "spark-head" else None)
    client = TestClient(create_app())

    r = client.post("/api/assistant/brain", json={"node": "spark-head", "slug": "nemo"})
    assert r.status_code == 200, r.text
    assert r.json()["brain"]["base_url"] == "http://192.168.10.30:8000/v1"
    provider = client.get("/api/provider").json()
    assert provider["name"] == "openai-compat" and provider["model"] == "nemotron-3"
    assert provider["base_url"] == "http://192.168.10.30:8000/v1"
    assert client.get("/api/assistant").json()["available"] is True, "openai-compat ในบ้านไม่ต้องมี key"

    assert client.post("/api/assistant/brain", json={"node": "spark-head", "slug": "asleep"}).status_code == 409
    assert client.post("/api/assistant/brain", json={"node": "spark-head", "slug": "a;b"}).status_code == 400


def test_a_destructive_ticket_refuses_apply_without_the_second_confirmation(ran):
    client = TestClient(create_app())
    ticket = policy.propose([{"action": "remove_model", "target": "this", "params": {"slug": "old"}}]).id
    shown = client.get(f"/api/assistant/ticket/{ticket}").json()
    assert shown["destructive"] is True and shown["default_mode"] == "hold"

    r = client.post(f"/api/assistant/ticket/{ticket}/choose", json={"mode": "apply"})
    assert r.status_code == 400 and ran == []
    body = client.post(f"/api/assistant/ticket/{ticket}/choose", json={"mode": "apply", "confirm": True}).json()
    assert ran == ["remove_model"] and body["finished"] is True


def test_a_composite_ticket_walks_step_by_step_and_a_failure_explains_itself(monkeypatch):
    log: list[str] = []

    def run(name, target="this", params=None):
        log.append(name)
        ok = name != "model_start"
        return runner.Outcome(name=name, title=name, target=target, exit_code=0 if ok else 1,
                              output="ok" if ok else "start failed")

    monkeypatch.setattr(policy, "run_action", run)
    monkeypatch.setattr(policy, "run_probe", lambda name, target="this", params=None:
                        runner.Outcome(name=name, title=name, target=target, exit_code=0,
                                       output="สาเหตุจาก log: CUDA out of memory"))
    client = TestClient(create_app())
    ticket = policy.propose([{"action": "deploy_model", "target": "spark-head",
                              "params": {"repo": "org/Model"}}]).id

    body = client.post(f"/api/assistant/ticket/{ticket}/choose", json={"mode": "step"}).json()
    assert log == ["deploy_plan"] and len(body["steps"]) == 6
    for _ in range(3):
        body = client.post(f"/api/assistant/ticket/{ticket}/advance").json()
    assert log == ["deploy_plan", "push_bundle", "model_download", "model_start"]
    failed = body["steps"][3]
    assert failed["result"]["ok"] is False
    assert "CUDA out of memory" in failed["result"]["explain"]
