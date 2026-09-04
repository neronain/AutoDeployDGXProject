"""เทสจาก code review 2026-09-04 — ชั้นเว็บ (api / state / jobs / memory / scriptedit / clone)

แต่ละเทสตรงกับข้อค้นพบหนึ่งข้อ: บั๊กที่ยังไม่มีใครเจอบนเครื่องจริงแต่เจอตอนอ่านโค้ด
เขียนไว้ให้มันเป็น "เจอบนเครื่องจริง" ไม่ได้อีกในอนาคต
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.web import create_app  # noqa: E402
from tests.test_web import (  # noqa: E402,F401 — fixture ใช้ร่วมกัน (autouse ด้วย)
    FakeStream, fleet, fresh_jobs, no_host_scan, registered, wait_for_job,
)

PAGE = Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html"


def _never_run(*_args, **_kwargs):
    raise AssertionError("ห้ามแตะ SSH — slug ที่ผิดต้องถูกปฏิเสธก่อนถึงตรงนี้")


# ── 1. slug จาก URL กลายเป็นคำสั่ง shell ─────────────────────────────────
def test_shell_metacharacters_in_slug_are_rejected_before_any_ssh(registered, monkeypatch):
    """`x';id;'` เคยทะลุเข้า echo '… {slug} …' ในสคริปต์ที่รันบนเครื่องอื่นได้"""
    monkeypatch.setattr("lmds.nodes.run", _never_run)
    monkeypatch.setattr("lmds.nodes.stream", _never_run)
    monkeypatch.setattr("lmds.nodes.push_file", _never_run)
    client = TestClient(create_app())
    bad = "x'%3Bid%3B'"
    assert client.post(f"/api/nodes/spark2/models/{bad}/ctl/status").status_code == 400
    assert client.post(f"/api/nodes/spark2/models/{bad}/clone", json={"to": "y"}).status_code == 400
    assert client.get(f"/api/nodes/spark2/models/{bad}/memory").status_code == 400
    assert client.post(f"/api/nodes/spark2/models/{bad}/doctor").status_code == 400
    assert client.post(f"/api/nodes/spark2/models/{bad}/bench").status_code == 400
    assert client.post(f"/api/models/{bad}/push/spark2").status_code == 400
    assert client.get(f"/api/models/{bad}/script?node=spark2").status_code == 400
    # slug ปกติของ bundle ยังผ่าน — ตัวกรองต้องไม่กว้างจนใช้งานจริงไม่ได้
    from lmds.web.api import _SLUG_OK

    for ok in ("qwen3-8b-gguf", "Qwen3.6-35B_nvfp4", "a"):
        assert _SLUG_OK.fullmatch(ok), ok
    for nope in ("", "-x", "a b", "a/b", "$(id)", "a" * 65):
        assert not _SLUG_OK.fullmatch(nope), nope


def test_locate_snippets_keep_the_slug_quoted_everywhere(registered, monkeypatch):
    """ทุกที่ที่ slug โผล่ในสคริปต์ — รวม echo — ต้องเป็นตัวที่ shlex.quote แล้ว"""
    import shlex

    from lmds.web import memory, scriptedit

    def only_quoted(script: str, slug: str) -> None:
        quoted = shlex.quote(slug)
        assert quoted in script
        assert script.count(slug) == script.count(quoted), script

    only_quoted(scriptedit._locate("$(id)"), "$(id)")

    captured = []
    monkeypatch.setattr("lmds.nodes.run", lambda node, script, timeout=0: captured.append(script)
                        or SimpleNamespace(ok=False, stdout="", stderr="no"))
    from lmds.nodes import NodeError, find

    with pytest.raises(NodeError):
        memory.read_node_profile(find("spark2"), "$(id)")
    only_quoted(captured[-1], "$(id)")

    from lmds.fleet.clone import CloneError, inspect_source, plan_clone
    from lmds.nodes import Node, add

    add(Node(name="spark3", host="10.0.0.7", user="ops"))
    with pytest.raises(CloneError):
        inspect_source(plan_clone("$(id)", "spark2", "spark3"))
    only_quoted(captured[-1], "$(id)")


# ── 3. os.environ ของทั้ง process ถูกใช้ส่ง option ────────────────────────
def test_overlapping_local_starts_leave_the_environment_clean(fleet, monkeypatch):
    """สอง start ซ้อนกันเคย restore ค่าของกันและกัน → API_PORT ค้างใน env ถาวร"""
    monkeypatch.delenv("API_PORT", raising=False)
    seen = []

    def slow_start(server):
        seen.append(os.environ.get("API_PORT"))
        time.sleep(0.2)
        # ค่าต้องเป็นของตัวเองตลอดที่รันอยู่ ไม่ถูกอีกคำขอเขียนทับกลางคัน
        assert os.environ.get("API_PORT") == seen[-1]
        return 0

    monkeypatch.setattr("lmds.fleet.start_server", slow_start)
    client = TestClient(create_app())
    results = []

    def go(port):
        results.append(client.post(f"/api/models/{fleet}/start", json={"port": port}).status_code)

    threads = [threading.Thread(target=go, args=(p,)) for p in (8101, 8102)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == [200, 200]
    assert sorted(seen) == ["8101", "8102"]
    assert os.environ.get("API_PORT") is None


# ── 4. งานที่ ssh ค้างต้องยกเลิกได้ ────────────────────────────────────────
def test_a_hung_remote_job_can_be_cancelled_and_releases_the_lock(registered, monkeypatch):
    monkeypatch.setattr(
        "lmds.nodes.stream",
        lambda node, command, *_, **__: subprocess.Popen(
            ["sleep", "60"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT),
    )
    client = TestClient(create_app())
    assert client.post("/api/jobs/nope/cancel").status_code == 404

    job = client.post("/api/nodes/spark2/models/demo/start").json()["job"]
    assert job["running"] is True
    # ระหว่างค้าง โมเดลเดียวกันสั่งอะไรไม่ได้เลย — นี่คือสิ่งที่ปุ่มยกเลิกมีไว้ปลดล็อก
    assert client.post("/api/nodes/spark2/models/demo/start").status_code == 409

    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json() == {"id": job["id"], "cancelled": True}
    done = wait_for_job(client, job["id"], tries=200)
    assert done["running"] is False and done["exit_code"] != 0
    assert "ยกเลิก" in done["output"]
    # ล็อก (เครื่อง, โมเดล) ต้องหลุดแล้ว
    assert client.post("/api/nodes/spark2/models/demo/start").status_code == 200
    # ยกเลิกซ้ำงานที่จบแล้ว = ไม่มีอะไรให้ฆ่า ไม่ใช่ error
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["cancelled"] is False


def test_cancel_route_is_token_guarded():
    client = TestClient(create_app(token="s3cret"))
    assert client.post("/api/jobs/x/cancel").status_code == 401


# ── 5. probe ที่ล้มด้วยอย่างอื่นที่ไม่ใช่ NodeError ─────────────────────────
def test_a_crashing_probe_lands_in_the_cache_instead_of_storming_ssh(monkeypatch):
    """UnicodeDecodeError จาก motd ที่ไม่ใช่ UTF-8 เคยทำให้เครื่องนั้นถูก SSH ทุกวินาที
    และการ์ดขึ้น "ไม่มีข้อมูล" ตลอดกาล เพราะไม่มีอะไรถูกเขียนลงแคชเลย
    """
    from lmds.nodes import Node, add
    from lmds.web import state

    add(Node(name="mojibake", host="10.0.0.8", user="ops"))

    def broken(node):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr("lmds.nodes.probe", broken)
    state._refresh_node("mojibake")
    entry = state.STORE.snapshot()["nodes"]["mojibake"]
    assert entry["data"] is None
    assert "UnicodeDecodeError" in entry["error"]
    assert state.STORE.due("mojibake") is False


# ── 6. refresh=true ต้องเขียนแคชด้วย ────────────────────────────────────────
def test_manual_refresh_updates_the_cache_the_sse_stream_reads(registered, monkeypatch):
    from lmds.web import state

    client = TestClient(create_app())
    client.get("/api/nodes/spark2/inventory")   # เติมแคชด้วยของเก่า
    fresh = {**registered, "host": {**registered["host"], "hostname": "spark2-renamed"}}
    monkeypatch.setattr("lmds.nodes.probe", lambda node: fresh)
    data = client.get("/api/nodes/spark2/inventory?refresh=true").json()
    assert data["host"]["hostname"] == "spark2-renamed"
    cached = state.STORE.snapshot()["nodes"]["spark2"]
    assert cached["data"]["host"] == data["host"]
    assert cached["refreshing"] is False and cached["error"] == ""

    from lmds.nodes import NodeError

    monkeypatch.setattr("lmds.nodes.probe", lambda node: (_ for _ in ()).throw(NodeError("boom")))
    assert client.get("/api/nodes/spark2/inventory?refresh=true").json()["reachable"] is False
    cached = state.STORE.snapshot()["nodes"]["spark2"]
    assert cached["data"] is None and "boom" in cached["error"] and cached["refreshing"] is False


# ── 7. restart/update นอก systemd ─────────────────────────────────────────
def test_restart_and_update_refuse_when_not_under_systemd(monkeypatch, tmp_path):
    """เดิมสั่ง systemctl ไปทั้งที่ไม่ได้อยู่ใต้ systemd แล้วบอกว่า "กำลังรีสตาร์ต…" เงียบ ๆ"""
    from lmds.web import selfupdate

    monkeypatch.setattr("lmds.web.api._running_unit", lambda: "")
    monkeypatch.setattr(selfupdate, "source_root", lambda: tmp_path)
    monkeypatch.setattr(selfupdate, "dirty_files", lambda root: [])
    client = TestClient(create_app())
    for route in ("/api/restart", "/api/update"):
        r = client.post(route, json={})
        assert r.status_code == 409, route
        assert "lmds web --restart" in r.json()["detail"]


def test_update_script_restarts_the_unit_that_is_actually_running(monkeypatch):
    from lmds.web import selfupdate

    assert "restart x.service" in selfupdate.update_script(unit="x.service")
    assert "restart lmds-web.service" in selfupdate.update_script()

    sent = {}
    monkeypatch.setattr("lmds.web.api._running_unit", lambda: "lmds-console.service")
    monkeypatch.setattr(selfupdate, "source_root", lambda: Path("/nonexistent-but-mocked"))
    monkeypatch.setattr(selfupdate, "dirty_files", lambda root: [])
    monkeypatch.setattr("lmds.web.jobs.start_shell",
                        lambda slug, command, script, cwd="": sent.update(script=script)
                        or SimpleNamespace(payload=lambda: {"id": "j"}))
    assert TestClient(create_app()).post("/api/update", json={}).status_code == 200
    assert "restart lmds-console.service" in sent["script"]


def test_console_restart_waits_for_a_new_process_not_just_a_ping():
    page = PAGE.read_text(encoding="utf-8")
    body = page.split("async function restartConsole()")[1].split("\n}\n")[0]
    assert "waitForHub(hubBuild.boot)" in body
    assert 'api("/api/version")' not in body


# ── 8. คำสั่งยาวบน thread ของ request ────────────────────────────────────
def test_fleet_scan_asks_every_node_at_once(monkeypatch):
    from lmds.nodes import Node, add

    for i in range(8):
        add(Node(name=f"n{i}", host=f"10.0.0.{10 + i}", user="ops"))
    monkeypatch.setattr("lmds.scanner.scan", lambda: [])

    def slow(node, command, timeout=0):
        time.sleep(1.0)
        return SimpleNamespace(ok=True, stdout='{"host": [{"kind": "gguf", "name": "m", "path": "/x", '
                                                '"size_gb": 1, "shards": 1, "layout": "hub"}]}', stderr="")

    monkeypatch.setattr("lmds.nodes.run", slow)
    started = time.monotonic()
    payload = TestClient(create_app()).get("/api/scan?all_nodes=true").json()
    assert time.monotonic() - started < 3.0, "ต้องถามพร้อมกัน ไม่ใช่เรียงคิว 8 วิ"
    assert all(payload[f"n{i}"][0]["name"] == "m" for i in range(8))


def test_remote_stop_streams_as_a_job(registered, monkeypatch):
    """docker stop ของ vLLM ที่กำลังโหลดรอได้เป็นนาที — เป็น job ถึงจะกดยกเลิกได้"""
    monkeypatch.setattr("lmds.nodes.run", _never_run)
    monkeypatch.setattr("lmds.nodes.stream", lambda node, command, *_, **__: FakeStream(["stopped\n"]))
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/models/demo/stop")
    assert r.status_code == 200 and "job" in r.json()
    assert wait_for_job(client, r.json()["job"]["id"])["exit_code"] == 0


def test_short_remote_commands_no_longer_get_half_an_hour(registered, monkeypatch):
    seen = {}
    monkeypatch.setattr("lmds.nodes.run", lambda node, command, timeout=0: seen.update(timeout=timeout)
                        or SimpleNamespace(exit_code=0, stdout="ok", stderr=""))
    assert TestClient(create_app()).post("/api/nodes/spark2/models/demo/doctor").status_code == 200
    assert seen["timeout"] <= 120


def test_scan_panel_reports_failures_instead_of_hanging():
    page = PAGE.read_text(encoding="utf-8")
    body = page.split("async function loadScan()")[1].split("\n}\n")[0]
    assert "try {" in body and "catch" in body and "Scan failed" in body


# ── 9. selected_gguf เคยถูกส่งเป็น engine= ────────────────────────────────
def test_choosing_a_gguf_file_does_not_masquerade_as_an_engine_choice(monkeypatch):
    """GgufVariant หลุดเข้า build_plan(engine=…) → "ผู้ใช้เลือก engine เอง" → ข้าม LLM เงียบ ๆ"""
    import lmds.brain as brain
    from lmds.inspector.report import GgufVariant
    from lmds.web import deploy as dep
    from tests.test_generator import gguf_report

    variants = [GgufVariant(filename="demo-Q4.gguf", size_bytes=4 * 1024**3),
                GgufVariant(filename="demo-Q8.gguf", size_bytes=8 * 1024**3)]
    monkeypatch.setattr("lmds.inspector.inspect_model",
                        lambda source, client: gguf_report(selected_gguf=getattr(source, "filename", None),
                                                           gguf_variants=variants))
    seen = {}
    real = brain.build_plan

    def spy(report, fit, provider, **kwargs):
        seen.update(kwargs)
        return real(report, fit, provider, **kwargs)

    monkeypatch.setattr(brain, "build_plan", spy)
    result = dep.analyze("unsloth/demo-GGUF", target="dgx-spark-single", no_llm=True,
                         selected_gguf="demo-Q4.gguf")
    assert result["plan"]["selected_gguf"] == "demo-Q4.gguf"
    assert seen["engine"] is None, "ไม่ได้เลือก engine → ต้องส่ง None ไม่ใช่ GgufVariant"


# ── 10. ผู้ช่วยอ่าน repo ผิดคีย์ ─────────────────────────────────────────
def test_assistant_sees_the_model_repo(monkeypatch):
    from lmds.web import assistant, state

    monkeypatch.setattr("lmds.nodes.load", lambda: [])
    state.STORE.set_local({"host": {}, "models": [
        {"slug": "demo", "running": True, "port": 8000, "model_id": "org/name"}]})
    got = assistant.gather_state()["models_here"]
    assert got[0]["repo"] == "org/name"


# ── 11. หน้าเว็บ: error ที่หายไปเงียบ ๆ ───────────────────────────────────
def test_page_surfaces_errors_it_used_to_swallow():
    page = PAGE.read_text(encoding="utf-8")
    # (a) detail แบบสตริง (400/401/429/500) ต้องโชว์ ไม่ใช่ "Analysis failed"
    assert page.count('typeof d.detail === "string"') >= 2
    # (b) เลิกตาม job ต้องบอก ไม่ใช่หยุดเงียบ
    assert page.count("Lost track of job") >= 2
    # (c) doctor ตรวจ r.ok ก่อนแตะ findings
    doctor = page.split("/doctor`);")[1][:400]
    assert "if (!r.ok)" in doctor and "(d.findings || []).filter" in doctor
    # ปุ่มยกเลิกงาน — ทั้งโมเดลในเครื่องนี้และบนเครื่องอื่น
    assert "async function cancelJob(" in page
    assert page.count("cancelJob('") >= 2
    assert "/cancel`" in page


# ── 12. ป้าย "มีอัปเดต" บนเครื่องที่ติดตั้งจริง ─────────────────────────────
def test_upstream_check_uses_the_stamped_checkout(monkeypatch, tmp_path):
    """เดิมมอง .git จากตำแหน่งโมดูล — บน hub ที่ติดตั้งจริงโค้ดอยู่ใน site-packages จึงว่างเสมอ"""
    from lmds.web import selfupdate

    git = ["git", "-c", "user.email=t@example", "-c", "user.name=t", "-C", str(tmp_path)]
    subprocess.run(git + ["init", "-q"], check=True)
    (tmp_path / "f").write_text("x")
    subprocess.run(git + ["add", "f"], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    subprocess.run(git + ["remote", "add", "origin", str(tmp_path)], check=True)

    monkeypatch.setattr(selfupdate, "source_root", lambda: tmp_path)
    assert TestClient(create_app()).get("/api/version?check_repo=true").json()["upstream"] != ""
    monkeypatch.setattr(selfupdate, "source_root", lambda: None)
    assert TestClient(create_app()).get("/api/version?check_repo=true").json()["upstream"] == ""


# ── 13. option ของโมเดลในเครื่องนี้ต้องผ่านตัวตรวจเดียวกับ node ────────────
def test_bad_local_options_are_400_not_500(fleet, monkeypatch):
    monkeypatch.setattr("lmds.fleet.start_server", _never_run)
    client = TestClient(create_app())
    for route in (f"/api/models/{fleet}/start", f"/api/models/{fleet}/run/start"):
        r = client.post(route, json={"port": "abc"})
        assert r.status_code == 400, (route, r.text)
        assert "port" in r.json()["detail"]
    r = client.post(f"/api/models/{fleet}/run/start", json={"image": "evil.example/x:latest"})
    assert r.status_code == 400 and "image" in r.json()["detail"]


# ── ฟอนต์อยู่ในแพ็กเกจ เสิร์ฟจาก hub เอง (air-gapped) ──────────────────────────────

def test_the_console_typefaces_are_served_by_the_hub_not_the_internet():
    """ผู้ใช้ 2026-09-04: "โหลดฟอนต์มาเพิ่มได้เพื่อให้ธีมสมบูรณ์" — แต่กฎห้ามดึงจากเน็ตยังอยู่"""
    import re

    from fastapi.testclient import TestClient

    from lmds.web.api import create_app

    client = TestClient(create_app())
    page = client.get("/").text
    urls = re.findall(r"url\((/fonts/[^)]+\.woff2)\)", page)
    assert {"/fonts/geist-latin.woff2", "/fonts/geist-mono-latin.woff2"} <= set(urls)
    assert "fonts.gstatic.com" not in page and "fonts.googleapis.com" not in page
    assert '--sans: "Geist"' in page and '--mono: "Geist Mono"' in page
    for url in set(urls):
        r = client.get(url)                       # ไม่ต้องมี token — หน้า login ก็ใช้
        assert r.status_code == 200, url
        assert r.headers["content-type"].startswith("font/woff2")
        assert "immutable" in r.headers["cache-control"]
        assert r.content[:4] == b"wOF2"
    assert client.get("/fonts/nope.woff2").status_code == 404
    assert client.get("/fonts/..%2F..%2Findex.html").status_code == 404


def test_commit_badges_tolerate_short_hashes_of_different_length():
    """spark-head ประทับ 0ad1a59e (8 ตัว) hub เป็น 0ad1a59 (7) — เทียบเป๊ะ = ขึ้น "ยังไม่ตรง" หลังอัปเดตสำเร็จ"""
    from pathlib import Path

    page = Path(__file__).resolve().parents[1].joinpath("src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "function sameCommit(a, b)" in page
    assert "commit !== hubBuild.commit" not in page
    assert "reg.lmds_commit !== hubBuild.commit" not in page
    assert "hubBuild.installed !== hubBuild.commit" not in page
    text = Path(__file__).resolve().parents[1].joinpath("install.sh").read_text(encoding="utf-8")
    assert "rev-parse --short=7 HEAD" in text


def test_advanced_options_are_neither_remembered_nor_sent_unless_the_section_is_open():
    """dgx-veerasiam 2026-09-04: image ชุมชนที่ค้างใน localStorage ถูกส่งทับ bundle ทุกครั้งที่กด start"""
    from pathlib import Path

    page = Path(__file__).resolve().parents[1].joinpath("src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert 'const ADV_OPTS = ["image", "engine_env", "extra_args", "tool_parser", "reasoning_parser", "image_min_tokens"]' in page
    assert "return stripAdv(JSON.parse(localStorage.getItem(\"lmds:node:\" + key)" in page
    assert 'if (!adv || !adv.open) stripAdv(o);' in page
    assert 'JSON.stringify(stripAdv({ ...o }))' in page
    assert 'localStorage.removeItem("lmds:node:" + node + "/" + slug)' in page
