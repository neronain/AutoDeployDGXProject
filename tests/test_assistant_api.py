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
    monkeypatch.setattr(assistant, "build_messages", lambda history, evidence=None: ("sys", []))

    class Fake:
        def stream_chat(self, system, messages):
            yield "ดูให้แล้วครับ"

    monkeypatch.setattr("lmds.brain.providers.make_provider", lambda *a, **k: Fake())
    return assistant


def test_the_stream_says_what_it_checked(chatty, monkeypatch):
    monkeypatch.setattr(chatty, "investigate", lambda question, state=None: {
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
                        lambda question, state=None: {"probes": [], "docs": [],
                                                      "ticket": ticket.payload()})
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat",
                                 json={"messages": [{"role": "user", "content": "แก้ให้ที"}]}).text)

    sent = next(e["ticket"] for e in events if "ticket" in e)
    assert sent["ticket"] == ticket.id
    assert sent["mode"] == ""
    assert ran == [], "ส่งข้อเสนอมาแสดงผล ไม่ใช่ลงมือทำ"


def test_a_broken_investigation_still_answers(chatty, monkeypatch):
    def explode(question, state=None):
        raise RuntimeError("ssh ล่ม")

    monkeypatch.setattr(chatty, "investigate", explode)
    client = TestClient(create_app())
    events = _events(client.post("/api/assistant/chat",
                                 json={"messages": [{"role": "user", "content": "hi"}]}).text)

    checked = next(e["evidence"] for e in events if "evidence" in e)
    assert "ssh ล่ม" in checked["note"]
    assert any(e.get("delta") for e in events), "ตรวจไม่ได้ ≠ ตอบไม่ได้"
