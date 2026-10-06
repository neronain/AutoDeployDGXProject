"""audit 2026-10-06 — `lmds bench run`: คะแนนของโมเดลที่ไม่เคยได้รับคำถาม

`lmds bench run` ไม่เคยส่ง API key ของ bundle · ทุก bundle ใหม่ได้ key ตอน deploy และ controller
บังคับใช้ → ทุกคำขอได้ 401 → บันทึก "คะแนนความสามารถ 0/100" exit 0 แล้วตารางคะแนนก็โชว์ว่า
โมเดลทำอะไรไม่ได้เลย ทั้งที่ไม่มีคำขอไหนไปถึงโมเดลสักคำขอ — รูปร่างเดียวกับทุกบั๊กของรอบก่อน:
*สิ่งที่รายงานไม่ตรงกับของจริง*

ทุกเทสรันคำสั่งจริงผ่าน CLI กับเซิร์ฟเวอร์ HTTP จริง (ปลอมแค่ตัวโมเดล) — ไม่มีเทสไหนยืนยันด้วย
ข้อความในซอร์ส · เรื่อง path ของที่เก็บผลวัดอยู่ที่ tests/test_audit3_bench_store.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.bench import score, store
from lmds.cli.main import app
from lmds.fleet import ServerInfo, apikey

runner = CliRunner()


# ── เซิร์ฟเวอร์ OpenAI ปลอมที่บังคับ key เหมือน vLLM/llama.cpp จริง ──────────────────
class _Model:
    """ตัวโมเดลปลอมหลัง HTTP จริง — นับว่าคำขอไหนมาพร้อม key ที่ถูก"""

    def __init__(self, key: str = "", fail_after: int | None = None, fail_status: int = 503):
        self.key = key
        self.with_key = 0
        self.without_key = 0
        self.fail_after = fail_after      # ตอบปกติได้กี่คำขอก่อนเริ่มตอบ fail_status
        self.fail_status = fail_status
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(model):  # noqa: N805 — closure ของ handler ต้องเห็นตัวโมเดล
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if model.key and self.headers.get("authorization") != f"Bearer {model.key}":
                    model.without_key += 1
                    return self._json(401, {"error": "Unauthorized"})
                model.with_key += 1
                if model.fail_after is not None and model.with_key > model.fail_after:
                    return self._json(model.fail_status, {"error": "Loading model"})
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for token in ("a", "b", "c"):
                        self.wfile.write(("data: " + json.dumps(
                            {"choices": [{"delta": {"content": token}}]}) + "\n\n").encode())
                    self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":10,'
                                     b'"completion_tokens":3}}\n\ndata: [DONE]\n\n')
                    return None
                # คำตอบเดียวที่ผ่านข้อ instructions/thai/recall/json ไม่ผ่านข้อ tools/reasoning
                # — ให้มีทั้ง ✅ และ ❌ จริง จะได้เห็นว่า "ไม่ได้วัด" ไม่ถูกนับปนกับสองอย่างนี้
                answer = ('{"city": "ปารีส Paris กรุงเทพมหานคร QUAIL-7742 '
                          + "ท้องฟ้าเป็นสีฟ้าเพราะแสงกระเจิงในชั้นบรรยากาศของโลก" + '"}')
                return self._json(200, {"choices": [{"finish_reason": "stop",
                                                     "message": {"content": answer}}]})

        return Handler


@pytest.fixture
def bench_home(tmp_path, monkeypatch):
    """ที่เก็บผลวัดและที่เก็บ key แยกต่อเทส · ไม่มี API_KEY ของเครื่อง dev ปนเข้ามา"""
    root = tmp_path / "home" / ".lmds" / "bench"
    root.mkdir(parents=True)
    monkeypatch.setattr(store, "bench_root", lambda: root)
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "home" / ".lmds" / "keys"))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    return root


def _serve(monkeypatch, slug: str, port: int) -> None:
    """ให้ `lmds bench run` เห็น bundle นี้ว่ารันอยู่ที่ port ของเซิร์ฟเวอร์ปลอม"""
    import lmds.fleet
    import lmds.fleet.manager as manager

    real = manager.find

    def running(wanted):
        found = real(wanted)
        if found is None and wanted == slug:
            found = ServerInfo(slug=slug, model=slug, engine="vllm")
        if found is not None:
            found.running, found.port = True, port
        return found

    monkeypatch.setattr(lmds.fleet, "find", running)


def _run(slug: str, *extra: str):
    return runner.invoke(app, ["bench", "run", slug, "--quick", "--runs", "1", "--skip-burn", *extra])


def _said(result) -> str:
    """ข้อความที่ผู้ใช้เห็น โดยไม่ขึ้นกับว่า rich ตัดบรรทัดตรงไหน (ความกว้างจอของเครื่องที่รันเทส)"""
    return " ".join(result.output.split())


# ── 1. key ─────────────────────────────────────────────────────────────────────────
def test_bench_sends_the_key_that_deploy_minted_for_the_bundle(tmp_path, monkeypatch, bench_home):
    """เส้นทางของลูกค้าจริง: deploy (ได้ key) → start (เซิร์ฟเวอร์บังคับ key) → bench run

    ก่อนแก้: คำขอที่ไม่มี key 9 · มี key 0 · "คะแนนความสามารถ 0/100" · exit 0
    """
    from tests.test_generator import safetensors_report

    report = safetensors_report(weight_bytes=20 * 2**30)
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: report)
    deployed = runner.invoke(app, ["deploy", report.repo_id, "--no-llm", "--target", "dgx-spark-single",
                                   "--yes", "--output", str(tmp_path / "bundles")])
    assert deployed.exit_code == 0, deployed.output
    slug = "qwen3-32b"
    key = apikey.read(slug)
    assert key, "deploy ต้องตั้ง key ให้ bundle ใหม่ — ถ้าไม่ตั้งแล้ว เทสนี้ไม่ได้ทดสอบอะไร"

    model = _Model(key)
    try:
        _serve(monkeypatch, slug, model.port)
        result = _run(slug)
    finally:
        model.close()

    assert result.exit_code == 0, result.output
    assert model.without_key == 0, f"มี {model.without_key} คำขอที่ไม่ได้ส่ง key ของ bundle"
    assert model.with_key >= 8
    assert key not in result.output, "ห้ามพิมพ์ key ออกจอ"

    stored = store.load(store.runs_for(slug)[0])
    assert key not in json.dumps(stored), "ห้ามเก็บ key ลงไฟล์ผลวัด"
    capability = score.capability_score(stored["probes"])
    assert capability["score"] and capability["score"] > 0
    assert "instructions" in capability["passed"] and not capability["unmeasured"]
    assert all(not w["error"] for w in stored["workloads"])


def test_the_environment_key_wins_over_the_stored_one_like_the_controller_does(monkeypatch, bench_home):
    """controller: flag/env > ที่เก็บ · คนที่ start ด้วย `API_KEY=… lmds start` ต้องวัดได้ด้วยวิธีเดียวกัน"""
    apikey.write("m1", "stored-but-not-what-the-server-uses")
    monkeypatch.setenv("API_KEY", "the-key-the-server-was-started-with")
    model = _Model("the-key-the-server-was-started-with")
    try:
        _serve(monkeypatch, "m1", model.port)
        result = _run("m1", "--caps-only")
    finally:
        model.close()
    assert result.exit_code == 0, result.output
    assert model.without_key == 0 and model.with_key >= 5


# ── 2. ไม่ได้วัด ≠ ได้ศูนย์ ─────────────────────────────────────────────────────────
def _good_run(root: Path, slug: str, stamp: str = "2026-10-01T09:00:00") -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stamp.replace('-', '').replace(':', '')}.json").write_text(json.dumps({
        "slug": slug, "stamped_at": stamp, "engine": "vllm", "machine": {"hostname": "h"},
        "workloads": [{"key": "chat-short", "label": "chat", "target_input": 512, "error": "",
                       "decode_tps": 30.0, "ttft_s": 0.5, "prompt_tokens": 500, "runs": 3}],
        "probes": [{"key": "tools", "label": "Tool calling", "passed": True, "detail": "",
                    "skipped": False}],
    }), encoding="utf-8")


def test_a_server_that_rejects_every_request_is_not_measured_and_nothing_is_stored(monkeypatch, bench_home):
    """ทุกคำขอ 401 = ไม่มีข้อมูลของโมเดลเลย — ต้อง exit ไม่เป็นศูนย์ บอกสาเหตุ และไม่เก็บอะไร

    ที่ต้องไม่เก็บ: `latest_merged` เอารอบล่าสุดขึ้นตารางคะแนน รอบ 401 ล้วนจึงทับรอบดีเมื่อวานด้วย 0/100
    """
    _good_run(bench_home, "m1")
    apikey.write("m1", "a-key-the-server-does-not-accept")
    model = _Model("the-real-key")
    try:
        _serve(monkeypatch, "m1", model.port)
        result = _run("m1")
    finally:
        model.close()

    assert result.exit_code == 1, result.output
    assert "ไม่ได้วัด" in _said(result) and "401" in _said(result)
    assert "lmds restart m1" in _said(result), "ต้องบอกทางแก้ของเคส key ใหม่หลัง start"
    assert "0/100" not in _said(result), "ห้ามมีคะแนนของรอบที่ไม่ได้วัด"
    assert len(store.runs_for("m1")) == 1, "รอบที่ไม่ได้วัดต้องไม่ถูกเก็บ"

    board = runner.invoke(app, ["bench", "list"])
    assert "100/100" in board.output and "30.0" in board.output, board.output


def test_a_server_that_demands_a_key_we_do_not_have_says_how_to_supply_one(monkeypatch, bench_home):
    """ไม่มี key เก็บไว้เลย (เซิร์ฟเวอร์ถูก start ด้วย API_KEY ของผู้ใช้เอง) — ข้อความต้องต่างจากเคส key ผิด"""
    model = _Model("only-the-operator-knows-this")
    try:
        _serve(monkeypatch, "m1", model.port)
        result = _run("m1", "--speed-only")
    finally:
        model.close()
    assert result.exit_code == 1, result.output
    assert "lmds key set m1" in _said(result)
    assert store.runs_for("m1") == []


def test_a_server_nobody_listens_on_is_not_measured(monkeypatch, bench_home):
    """connection refused ทุกคำขอ — เดิมบันทึก 0/100 เหมือนกัน"""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    _serve(monkeypatch, "m1", free_port)
    result = _run("m1")
    assert result.exit_code == 1, result.output
    assert "ต่อเซิร์ฟเวอร์ไม่ได้" in _said(result) and "lmds doctor m1" in _said(result)
    assert store.runs_for("m1") == []


def test_a_run_that_dies_half_way_records_what_was_not_measured_instead_of_zeros(monkeypatch, bench_home):
    """เซิร์ฟเวอร์ตอบได้ช่วงแรกแล้วเริ่มตอบ 503 (โมเดลถูก unload/restart กลางรอบ)

    ข้อที่วัดได้ต้องถูกเก็บ · ข้อที่วัดไม่ได้ต้องถูกจดว่า "ไม่ได้วัด" ไม่ใช่ถูกนับเป็นสอบตก
    """
    model = _Model(fail_after=3)       # instructions · thai · json ผ่านไปได้ ที่เหลือ 503
    try:
        _serve(monkeypatch, "m1", model.port)
        result = _run("m1", "--caps-only")
    finally:
        model.close()

    assert result.exit_code == 0, result.output
    assert "ไม่ได้วัด 3 จาก 6" in _said(result), result.output
    stored = store.load(store.runs_for("m1")[0])
    by_key = {p["key"]: p for p in stored["probes"]}
    assert [k for k, p in by_key.items() if p.get("unmeasured")] == ["tools", "reasoning", "recall"]
    assert by_key["tools"]["unmeasured"] == "not-ready" and "503" in by_key["tools"]["detail"]
    assert by_key["vision"]["skipped"] and not by_key["vision"].get("unmeasured"), "ไม่มี mmproj = ข้าม ไม่ใช่ไม่ได้วัด"

    capability = score.capability_score(stored["probes"])
    assert capability["score"] == 100 and capability["counted"] == 3, capability
    assert capability["unmeasured"] == ["tools", "reasoning", "recall"]
    assert capability["failed"] == [], "ข้อที่ไม่ได้วัดต้องไม่อยู่ในรายการสอบตก"
    assert capability["skipped"] == ["vision"]


def test_a_real_refusal_from_the_model_server_still_counts_as_a_failed_probe(monkeypatch, bench_home):
    """500 คือเซิร์ฟเวอร์ตอบเรื่อง *คำขอนั้น* (llama.cpp ที่ไม่เปิด --jinja ตอบแบบนี้กับ tools) — ยังเป็นผลวัดจริง"""
    model = _Model(fail_after=3, fail_status=500)
    try:
        _serve(monkeypatch, "m1", model.port)
        result = _run("m1", "--caps-only")
    finally:
        model.close()
    assert result.exit_code == 0, result.output
    capability = score.capability_score(store.load(store.runs_for("m1")[0])["probes"])
    assert capability["unmeasured"] == [] and "tools" in capability["failed"]
    assert capability["score"] < 100


def test_the_scoreboard_keeps_yesterdays_numbers_when_todays_side_was_not_measured(bench_home):
    """รอบใหม่ที่ด้านความสามารถไม่ได้วัดเลย ต้องไม่เอาขีดกลางไปทับคะแนนที่วัดได้จริงของรอบก่อน"""
    _good_run(bench_home, "m1")
    (bench_home / "m1" / "20261002T090000.json").write_text(json.dumps({
        "slug": "m1", "stamped_at": "2026-10-02T09:00:00",
        "workloads": [{"key": "chat-short", "target_input": 512, "error": "", "decode_tps": 31.0,
                       "ttft_s": 0.4}],
        "probes": [{"key": "tools", "passed": False, "skipped": True, "unmeasured": "auth"},
                   {"key": "vision", "passed": False, "skipped": True}],
    }), encoding="utf-8")
    merged = store.latest_merged("m1")
    assert merged["workloads"][0]["decode_tps"] == 31.0
    assert merged["probes"][0]["passed"] is True
    assert merged["probes_from"] == "2026-10-01T09:00:00"
