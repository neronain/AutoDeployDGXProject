"""เทส Web UI — ชั้นนี้ต้องไม่มี logic ของตัวเอง

กฎที่เทสไว้: อะไรที่ CLI ทำได้ เว็บต้องได้ผลเหมือนกัน และหน้าที่สั่ง start/stop ได้
ต้องไม่เปิดโล่งให้ทั้งวง network โดยบังเอิญ
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient as _TestClient  # noqa: E402

from lmds.web import create_app  # noqa: E402


def TestClient(app, *args, **kwargs):
    """production no-token mode accepts loopback Host only; keep every test on that real origin."""
    kwargs.setdefault("base_url", "http://127.0.0.1")
    return _TestClient(app, *args, **kwargs)


@pytest.fixture(autouse=True)
def fresh_jobs():
    """งานค้างจากเทสก่อนทำให้เทสถัดไปได้ 409 — ล้างทะเบียนงานทุกครั้ง"""
    from lmds.web import jobs

    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    yield
    for job in jobs._JOBS.values():
        if job.process and job.running:
            job.process.kill()
    jobs._JOBS.clear()
    jobs._ACTIVE.clear()


@pytest.fixture(autouse=True)
def no_host_scan(monkeypatch):
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "")


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    slug = "demo-gguf"
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    model_dir = tmp_path / "models" / slug
    model_dir.mkdir(parents=True)
    (model_dir / "demo-Q8.gguf").write_bytes(b"x" * 1024)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    bundle = tmp_path / "bundles" / slug
    bundle.mkdir(parents=True)
    controller = bundle / f"{slug}-single.sh"
    controller.write_text("#!/usr/bin/env bash\necho log-line\n", encoding="utf-8")
    controller.chmod(0o755)
    (bundle / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": "unsloth/demo-GGUF", "revision": "sha", "selected_gguf": "demo-Q8.gguf"},
        "runtime": {"engine": "llamacpp", "image": "img"},
        "serving": {"context": 16384},
        "topology": "single",
    }), encoding="utf-8")

    run_dir = tmp_path / "run" / slug
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\nmodel={slug}\nmodel_id=unsloth/demo-GGUF\nengine=llamacpp\n"
        f"mode=docker\nport=8000\ncontainer=lmds-{slug}\npid_file=\n"
        f"controller={controller}\nstarted_at=2026-08-03T10:00:00\n",
        encoding="utf-8",
    )
    return slug


def test_page_is_self_contained(fleet):
    """เครื่องลูกค้าอาจอยู่หลัง proxy/air-gapped — หน้าเว็บต้องไม่ดึงอะไรจากเน็ต"""
    body = TestClient(create_app()).get("/").text
    assert "<title>LMDS</title>" in body
    for remote in ("https://", "http://cdn", "cdnjs", "googleapis", "unpkg"):
        assert remote not in body, f"หน้าเว็บอ้างถึงแหล่งภายนอก: {remote}"


def test_models_endpoint_matches_fleet(fleet):
    data = TestClient(create_app()).get("/api/models").json()
    assert [m["slug"] for m in data["models"]] == [fleet]
    model = data["models"][0]
    assert model["engine"] == "llamacpp"
    assert model["context"] == 16384
    assert model["running"] is False


def test_doctor_endpoint_matches_cli(fleet):
    """ผลจากเว็บต้องเป็นชุดเดียวกับ lmds doctor ไม่ใช่คำนวณซ้ำคนละทาง"""
    from lmds.doctor import diagnose

    web = TestClient(create_app()).get(f"/api/models/{fleet}/doctor").json()
    cli = diagnose(fleet)
    assert web["healthy"] == cli.healthy
    assert [f["name"] for f in web["findings"]] == [f.name for f in cli.findings]


def test_logs_endpoint_returns_text(fleet):
    data = TestClient(create_app()).get(f"/api/models/{fleet}/logs?lines=10").json()
    assert "log-line" in data["text"]


def test_unknown_slug_is_404(fleet):
    client = TestClient(create_app())
    assert client.get("/api/models/ไม่มีจริง/logs").status_code == 404
    assert client.post("/api/models/ไม่มีจริง/start").status_code == 404


def test_token_guards_every_api_route(fleet):
    """หน้านี้สั่ง start/stop ได้ — ตั้ง token แล้วต้องกันได้ทุกเส้นทาง ไม่ใช่แค่บางอัน"""
    client = TestClient(create_app(token="s3cret"))
    for path in ("/api/host", "/api/models", f"/api/models/{fleet}/doctor", f"/api/models/{fleet}/logs"):
        assert client.get(path).status_code == 401, path
    assert client.post(f"/api/models/{fleet}/start").status_code == 401

    ok = client.get("/api/models", headers={"x-lmds-token": "s3cret"})
    assert ok.status_code == 200
    assert client.get("/api/models?token=s3cret").status_code == 200


def test_wrong_token_rejected(fleet):
    client = TestClient(create_app(token="s3cret"))
    assert client.get("/api/models", headers={"x-lmds-token": "s3cre"}).status_code == 401
    assert client.get("/api/models", headers={"x-lmds-token": "s3cretX"}).status_code == 401


def test_token_guard_covers_every_api_route():
    """เพิ่ม endpoint ใหม่แล้วลืม dependency = สิทธิ์หลุดทันที; ตรวจ route table ทั้งก้อน"""
    app = create_app(token="s3cret")
    client = TestClient(app)
    replacements = {
        "{slug}": "missing", "{session_id}": "missing", "{command}": "missing",
        "{job_id}": "missing", "{name}": "missing",
    }
    checked = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/"):
            continue
        for marker, value in replacements.items():
            path = path.replace(marker, value)
        for method in sorted((getattr(route, "methods", set()) or set()) - {"HEAD", "OPTIONS"}):
            response = client.request(method, path, json={} if method != "GET" else None)
            assert response.status_code == 401, f"{method} {path} ไม่ถูก token guard"
            checked.append((method, path))
    assert checked


def test_cross_site_state_changes_are_rejected_without_breaking_cli_calls(fleet):
    """loopback ไม่มี token โดยตั้งใจ แต่เว็บอื่นต้อง submit POST มาหยุดโมเดลไม่ได้"""
    client = TestClient(create_app())
    path = f"/api/models/{fleet}/stop"
    assert client.post(path, headers={
        "origin": "https://evil.example", "sec-fetch-site": "cross-site",
    }).status_code == 403
    assert client.post(path, headers={"origin": "https://evil.example"}).status_code == 403

    # same-origin browser และ client/CLI ที่ไม่มี browser metadata ยังใช้ API ได้
    assert client.post(path, headers={"origin": "http://127.0.0.1"}).status_code == 200
    assert client.post(path).status_code == 200


def test_no_token_mode_rejects_dns_rebinding_hostnames():
    client = _TestClient(create_app(), base_url="http://attacker.example")
    assert client.get("/api/models").status_code == 421


def test_csrf_guard_covers_every_mutating_api_route():
    app = create_app()
    client = TestClient(app)
    replacements = {
        "{slug}": "missing", "{session_id}": "missing", "{command}": "missing",
        "{job_id}": "missing", "{name}": "missing",
    }
    checked = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/"):
            continue
        for marker, value in replacements.items():
            path = path.replace(marker, value)
        for method in sorted((getattr(route, "methods", set()) or set()) - {"GET", "HEAD", "OPTIONS"}):
            response = client.request(method, path, json={}, headers={
                "origin": "https://evil.example", "sec-fetch-site": "cross-site",
            })
            assert response.status_code == 403, f"{method} {path} ไม่ถูก CSRF guard"
            checked.append((method, path))
    assert checked


def test_index_sets_browser_security_headers():
    response = TestClient(create_app()).get("/")
    csp = response.headers["content-security-policy"]
    assert "script-src 'nonce-" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert '<script nonce="' in response.text


def test_no_token_configured_means_open(fleet):
    """bind 127.0.0.1 (ค่าเริ่มต้น) ไม่ต้องมี token — เข้าถึงได้เฉพาะเครื่องนี้อยู่แล้ว"""
    assert TestClient(create_app()).get("/api/models").status_code == 200


def test_actions_call_fleet_and_report_failure(fleet, monkeypatch):
    calls = []
    monkeypatch.setattr("lmds.fleet.start_server", lambda info, **kwargs: calls.append("start") or 0)
    monkeypatch.setattr("lmds.fleet.stop_server", lambda info: calls.append("stop") or "controller")
    client = TestClient(create_app())

    assert client.post(f"/api/models/{fleet}/start").json()["ok"] is True
    assert client.post(f"/api/models/{fleet}/stop").json()["ok"] is True
    assert calls == ["start", "stop"]

    monkeypatch.setattr("lmds.fleet.start_server", lambda info, **kwargs: 1)
    failed = client.post(f"/api/models/{fleet}/start")
    assert failed.status_code == 500
    assert failed.json()["ok"] is False


def test_fleet_error_becomes_409(fleet, monkeypatch):
    from lmds.fleet import FleetError

    def boom(info, **kwargs):
        raise FleetError("ไม่พบ controller")

    monkeypatch.setattr("lmds.fleet.restart_server", boom)
    r = TestClient(create_app()).post(f"/api/models/{fleet}/restart")
    assert r.status_code == 409
    assert "controller" in r.json()["detail"]


def test_static_page_ships_with_the_package():
    from lmds.web import api

    assert (Path(api.STATIC) / "index.html").is_file()


# ── deploy wizard ─────────────────────────────────────────────────────────────

def test_targets_endpoint_lists_presets():
    from lmds.fit import PRESETS

    data = TestClient(create_app()).get("/api/targets").json()
    assert {t["name"] for t in data["targets"]} == set(PRESETS)
    rtx = next(t for t in data["targets"] if t["name"] == "rtx-5090")
    assert rtx["tested"] is True  # validated 2026-08-03


def test_multi_variant_gguf_asks_before_guessing(monkeypatch):
    """repo GGUF หลาย variant ต้องให้ผู้ใช้เลือก ไม่ใช่เดาให้ — เดาผิดคือโหลดผิดไฟล์หลาย GB"""
    from tests.test_generator import gguf_report
    from lmds.inspector.report import GgufVariant
    from lmds.web import deploy as dep

    report = gguf_report(selected_gguf=None, gguf_variants=[
        GgufVariant(filename="demo-Q4.gguf", size_bytes=4 * 1024**3),
        GgufVariant(filename="demo-Q8.gguf", size_bytes=8 * 1024**3),
        GgufVariant(filename="mmproj-BF16.gguf", size_bytes=1024**3, is_mmproj=True),
    ])
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: report)

    with pytest.raises(dep.DeployError) as err:
        dep.analyze("unsloth/demo-GGUF", target="rtx-5090")
    assert err.value.kind == "choose-gguf"
    names = [v["filename"] for v in err.value.extra["variants"]]
    assert names == ["demo-Q4.gguf", "demo-Q8.gguf"]  # เรียงจากเล็กไปใหญ่ ไม่มี mmproj ปน


def test_single_variant_gguf_needs_no_question(monkeypatch):
    from tests.test_generator import gguf_report
    from lmds.web import deploy as dep

    report = gguf_report(selected_gguf=None)
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: report)
    result = dep.analyze("unsloth/demo-GGUF", target="dgx-spark-single", no_llm=True)
    assert result["plan"]["selected_gguf"] == "Qwen3-8B-Q4_K_M.gguf"


def test_no_fit_returns_alternatives(monkeypatch):
    """โมเดลใหญ่เกินเครื่อง — ต้องบอกทางเลือก ไม่ใช่ปล่อยให้ผู้ใช้เดา"""
    from tests.test_generator import safetensors_report
    from lmds.fit.analyzer import GIB
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model",
                        lambda source, client: safetensors_report(weight_bytes=400 * GIB))
    with pytest.raises(dep.DeployError) as err:
        dep.analyze("Qwen/Huge", target="rtx-5090", no_llm=True)
    assert err.value.kind == "no-fit"
    assert err.value.extra["alternatives"]


def test_generate_produces_a_validated_bundle(tmp_path, monkeypatch):
    """เว็บต้องได้ bundle คุณภาพเดียวกับ CLI — ผ่าน gates ครบและมี ZIP"""
    from tests.test_generator import safetensors_report
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True)

    result = dep.generate(analyzed["id"], context=16384, output=str(tmp_path))
    assert result["context"] == 16384
    assert all(g["passed"] for g in result["gates"])
    assert Path(result["zip"]).is_file()
    assert (Path(result["directory"]) / "MODEL_PROFILE.yaml").is_file()


def test_context_cannot_exceed_the_safe_ceiling(tmp_path, monkeypatch):
    """ผู้ใช้พิมพ์ context เกินเพดานได้ในช่อง input — ฝั่ง server ต้องตัดให้ ไม่เชื่อค่าจากหน้าเว็บ"""
    from tests.test_generator import safetensors_report
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True)
    ceiling = analyzed["plan"]["fit"]["max_safe_context"] or analyzed["plan"]["context"]

    result = dep.generate(analyzed["id"], context=9_000_000, output=str(tmp_path))
    assert result["context"] == ceiling


def test_session_is_single_use(tmp_path, monkeypatch):
    """generate แล้ว session ต้องหมดอายุ — กันกด 'สร้าง bundle' ซ้ำแล้วได้ของซ้อนกัน"""
    from tests.test_generator import safetensors_report
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True)
    dep.generate(analyzed["id"], output=str(tmp_path))

    with pytest.raises(dep.DeployError) as err:
        dep.generate(analyzed["id"], output=str(tmp_path))
    assert err.value.kind == "expired"


def test_analyze_endpoint_requires_a_model():
    r = TestClient(create_app()).post("/api/deploy/analyze", json={"model": "  "})
    assert r.status_code == 400


def test_deploy_endpoints_are_token_guarded(fleet):
    client = TestClient(create_app(token="s3cret"))
    assert client.get("/api/targets").status_code == 401
    assert client.post("/api/deploy/analyze", json={"model": "x"}).status_code == 401
    assert client.post("/api/deploy/abc/generate", json={}).status_code == 401


# ── งานที่ใช้เวลานาน (download / start) ───────────────────────────────────────

@pytest.fixture
def runnable(tmp_path, monkeypatch):
    """bundle ที่ controller รันได้จริง — download แล้วสร้างไฟล์ weight ให้"""
    from lmds.fleet import register_bundle

    slug = "demo"
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    model_dir = tmp_path / "models" / slug
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    bundle = tmp_path / "bundles" / slug
    bundle.mkdir(parents=True)
    controller = bundle / f"{slug}-single.sh"
    controller.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        f'  download) echo "โหลดอยู่"; mkdir -p "{model_dir}"; echo x > "{model_dir}/demo-Q8.gguf";;\n'
        '  fail) echo "พัง"; exit 3;;\n'
        '  *) echo "cmd $1";;\n'
        "esac\n",
        encoding="utf-8",
    )
    controller.chmod(0o755)
    (bundle / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": "org/demo", "revision": "sha", "selected_gguf": "demo-Q8.gguf", "served_name": slug},
        "runtime": {"engine": "llamacpp"}, "serving": {"context": 8192}, "topology": "single",
    }), encoding="utf-8")
    register_bundle(controller)
    return slug


def _wait(client, job_id, tries=60):
    import time

    for _ in range(tries):
        data = client.get(f"/api/jobs/{job_id}").json()
        if not data["running"]:
            return data
        time.sleep(0.1)
    raise AssertionError("งานไม่จบในเวลาที่กำหนด")


def test_generated_bundle_appears_before_first_start(runnable):
    """เคสจริงจากหน้าเว็บ: สร้าง bundle เสร็จแล้วไปต่อไม่ถูก เพราะ fleet เห็นเฉพาะตัวที่เคย start"""
    models = TestClient(create_app()).get("/api/models").json()["models"]
    assert [m["slug"] for m in models] == [runnable]
    assert models[0]["running"] is False


def test_download_button_shows_until_weights_exist(runnable):
    client = TestClient(create_app())
    assert client.get("/api/models").json()["models"][0]["downloaded"] is False

    job = client.post(f"/api/models/{runnable}/run/download").json()
    assert _wait(client, job["id"])["exit_code"] == 0

    assert client.get("/api/models").json()["models"][0]["downloaded"] is True


def test_job_output_is_streamed_back(runnable):
    client = TestClient(create_app())
    job = client.post(f"/api/models/{runnable}/run/download").json()
    assert "โหลดอยู่" in _wait(client, job["id"])["output"]


def test_failed_job_reports_exit_code(runnable, monkeypatch):
    from lmds.web import jobs

    monkeypatch.setattr(jobs, "ALLOWED", jobs.ALLOWED | {"fail"})
    client = TestClient(create_app())
    job = client.post(f"/api/models/{runnable}/run/fail").json()
    done = _wait(client, job["id"])
    assert done["exit_code"] == 3
    assert "พัง" in done["output"]


def test_only_one_job_per_model(runnable):
    """download ซ้อน start = ไฟล์พัง — ต้องกันไว้ ไม่ใช่หวังว่าผู้ใช้จะไม่กดซ้ำ"""
    client = TestClient(create_app())
    client.post(f"/api/models/{runnable}/run/download")
    second = client.post(f"/api/models/{runnable}/run/download")
    assert second.status_code == 409
    assert "กำลังรัน" in second.json()["detail"]


def test_only_allowlisted_commands_can_run(runnable):
    """ชื่อคำสั่งมาจาก URL — ห้ามส่งต่อไปให้ shell ตรง ๆ"""
    client = TestClient(create_app())
    for bad in ("rm-rf", "help;whoami", "../../etc/passwd"):
        assert client.post(f"/api/models/{runnable}/run/{bad}").status_code in (404, 409)


def test_job_routes_are_token_guarded(runnable):
    client = TestClient(create_app(token="s3cret"))
    assert client.post(f"/api/models/{runnable}/run/download").status_code == 401
    assert client.get("/api/jobs/whatever").status_code == 401


def test_finished_job_history_is_bounded(monkeypatch):
    """เว็บรันเป็น daemon หลายวันได้; ผลงานเก่าห้ามสะสมใน memory ตลอดอายุ process"""
    from lmds.web import jobs

    monkeypatch.setattr(jobs, "_MAX_FINISHED_JOBS", 2)
    for index in range(4):
        job = jobs.Job(id=f"done-{index}", slug=f"model-{index}", command="status",
                       exit_code=0)
        jobs._JOBS[job.id] = job
        jobs._ACTIVE[job.slug] = job.id
    running = jobs.Job(id="live", slug="live-model", command="download")
    jobs._JOBS[running.id] = running
    jobs._ACTIVE[running.slug] = running.id

    with jobs._LOCK:
        jobs._prune_finished_locked()

    assert list(jobs._JOBS) == ["done-2", "done-3", "live"]
    assert set(jobs._ACTIVE) == {"model-2", "model-3", "live-model"}


def test_job_payload_can_be_read_while_output_is_appended():
    """worker เขียน log พร้อม HTTP thread อ่าน payload; snapshot ต้องไม่ขว้าง mutation error"""
    from lmds.web import jobs

    job = jobs.Job(id="race", slug="demo", command="download")
    writer_done = threading.Event()
    errors = []

    def write() -> None:
        try:
            for index in range(5_000):
                job.append(f"line {index}\n")
        except Exception as exc:  # pragma: no cover - assertion captures an unexpected thread failure
            errors.append(exc)
        finally:
            writer_done.set()

    thread = threading.Thread(target=write)
    thread.start()
    while not writer_done.is_set():
        assert isinstance(job.payload()["output"], str)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []


def test_download_always_verifies_afterwards(runnable):
    """"กด download แล้วชัวร์ไหมว่าไฟล์มาครบ" — CLI ให้รัน verify-files ต่อเสมอ
    เว็บจึงต้องต่อให้ ไม่งั้นผู้ใช้ไม่มีทางรู้
    """
    client = TestClient(create_app())
    job = client.post(f"/api/models/{runnable}/run/download").json()
    assert job["steps"] == ["download", "verify-files"]
    done = _wait(client, job["id"])
    assert done["exit_code"] == 0
    assert "cmd verify-files" in done["output"]


def test_chain_stops_when_the_first_step_fails(runnable, monkeypatch):
    """download ล้ม = ไม่ต้อง verify ต่อ (verify ไฟล์ที่โหลดไม่จบไม่มีประโยชน์)"""
    from lmds.web import jobs

    monkeypatch.setitem(jobs.CHAINS, "download", ["fail", "verify-files"])
    monkeypatch.setattr(jobs, "ALLOWED", jobs.ALLOWED | {"fail"})
    client = TestClient(create_app())
    done = _wait(client, client.post(f"/api/models/{runnable}/run/download").json()["id"])
    assert done["exit_code"] == 3
    assert "cmd verify-files" not in done["output"]


def test_options_reach_the_controller_as_env(runnable, tmp_path):
    """เทียบเท่า `API_KEY=… ./x.sh start --port … --context …` ของ CLI"""
    from lmds.web import jobs

    env = jobs.controller_env({"port": 8001, "api_key": "s3cret", "context": 4096, "bind": "127.0.0.1"})
    assert env == {"API_PORT": "8001", "API_HOST": "127.0.0.1",
                   "CTX_SIZE": "4096", "MAX_MODEL_LEN": "4096", "API_KEY": "s3cret"}


def test_empty_options_change_nothing(runnable):
    """ไม่ได้ตั้งอะไร = ใช้ค่า default ของ controller ไม่ใช่ยัดค่าว่างทับ"""
    from lmds.web import jobs

    assert jobs.controller_env({}) == {}
    assert jobs.controller_env({"port": None, "api_key": "", "context": None}) == {}


@pytest.mark.parametrize("options", [
    {"port": 0}, {"port": 65536}, {"port": True}, {"port": "not-a-port"},
    {"context": 0}, {"context": 1.5}, {"slots": -1},
    {"bind": "evil.example"}, {"api_key": 123}, {"api_key": "bad\x00key"},
    {"extra": "ignored-before"},
])
def test_local_controller_options_are_validated(options):
    from lmds.web import jobs

    with pytest.raises(jobs.JobError):
        jobs.controller_env(options)


def test_stop_rejects_options_it_cannot_apply(runnable):
    response = TestClient(create_app()).post(f"/api/models/{runnable}/stop", json={"port": 8123})
    assert response.status_code == 400


def test_start_passes_options_through_without_mutating_process_env(runnable, monkeypatch):
    seen = {}

    def fake_start(info, *, env=None):
        seen.update(env or {})
        return 0

    monkeypatch.setattr("lmds.fleet.start_server", fake_start)
    TestClient(create_app()).post(f"/api/models/{runnable}/start",
                                  json={"port": 8123, "api_key": "abc"})
    assert seen == {"API_PORT": "8123", "API_KEY": "abc"}


def test_parallel_starts_keep_request_environments_isolated(runnable, monkeypatch):
    """sync FastAPI handlers รันคนละ thread; API key ของ request หนึ่งห้ามไหลไปอีก request"""
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2)
    seen = []

    def fake_start(info, *, env=None):
        barrier.wait(timeout=5)
        seen.append(dict(env or {}))
        return 0

    monkeypatch.setattr("lmds.fleet.start_server", fake_start)
    client = TestClient(create_app())
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda item: client.post(f"/api/models/{runnable}/start", json=item),
            ({"port": 8101, "api_key": "alpha"}, {"port": 8102, "api_key": "beta"}),
        ))
    assert [r.status_code for r in responses] == [200, 200]
    assert sorted(seen, key=lambda env: env["API_PORT"]) == [
        {"API_KEY": "alpha", "API_PORT": "8101"},
        {"API_KEY": "beta", "API_PORT": "8102"},
    ]


# ── ปิดช่องว่างเทียบ CLI: remove / autostart / test / stacked ────────────────

def test_removal_plan_shows_what_will_be_deleted(runnable):
    """ลบแล้วกู้ไม่ได้ — ต้องเห็นรายการกับขนาดก่อนยืนยัน เหมือนที่ CLI ทำ"""
    data = TestClient(create_app()).get(f"/api/models/{runnable}/removal-plan").json()
    labels = {i["label"] for i in data["items"]}
    assert "bundle" in labels
    assert data["total_bytes"] > 0


def test_removal_plan_respects_keep_weights(runnable):
    client = TestClient(create_app())
    client.post(f"/api/models/{runnable}/run/download")
    # อ่านครั้งเดียวแล้วใช้ค่านั้น — เดิมเรียก API สองรอบแล้วสมมติว่าผลเหมือนเดิม
    # งานอาจจบระหว่างสองรอบจน job กลายเป็น None (Linux เร็วกว่าจึงแพ้ race ประจำ)
    job = client.get("/api/models").json()["models"][0].get("job")
    if job:
        _wait(client, job["id"])

    withw = client.get(f"/api/models/{runnable}/removal-plan").json()
    keep = client.get(f"/api/models/{runnable}/removal-plan?keep_weights=true").json()
    assert not any(i["is_weights"] for i in keep["items"])
    assert keep["total_bytes"] <= withw["total_bytes"]


def test_remove_deletes_and_model_disappears(runnable, tmp_path):
    client = TestClient(create_app())
    assert client.post(f"/api/models/{runnable}/remove",
                       json={"keep_weights": True, "confirm": runnable}).status_code == 200
    assert [m["slug"] for m in client.get("/api/models").json()["models"]] == []


def test_local_remove_requires_the_exact_slug(runnable, monkeypatch):
    called = []
    monkeypatch.setattr("lmds.fleet.remove_server", lambda *args, **kwargs: called.append(args) or [])
    client = TestClient(create_app())
    for body in ({}, {"confirm": True}, {"confirm": "wrong"}, {"confirm": runnable.upper()}):
        assert client.post(f"/api/models/{runnable}/remove", json=body).status_code == 400
    assert called == []


@pytest.mark.parametrize("body", [
    {"confirm": "demo", "keep_weights": "false"},
    {"confirm": "demo", "unexpected": True},
])
def test_local_remove_rejects_ambiguous_options(runnable, body):
    body["confirm"] = runnable
    assert TestClient(create_app()).post(f"/api/models/{runnable}/remove", json=body).status_code == 400


def test_autostart_failure_tells_you_the_manual_commands(runnable, monkeypatch):
    """เว็บไม่มี tty ให้กรอกรหัส sudo — ต้องส่งคำสั่งกลับไปให้ผู้ใช้รันเอง ไม่ใช่ 500 เปล่า ๆ"""
    from lmds.fleet import FleetError

    def boom(info, timeout=1800, start_now=False):
        raise FleetError("ติดตั้ง autostart ไม่สำเร็จ\nลองรันมือ:\n  sudo systemctl enable lmds-demo")

    monkeypatch.setattr("lmds.fleet.enable_autostart", boom)
    r = TestClient(create_app()).post(f"/api/models/{runnable}/autostart", json={"enabled": True})
    assert r.status_code == 409
    assert "sudo systemctl enable" in r.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"enabled": "false"}, {"enabled": 0},
                                   {"enabled": True, "extra": 1}])
def test_autostart_requires_an_explicit_boolean(runnable, body):
    assert TestClient(create_app()).post(
        f"/api/models/{runnable}/autostart", json=body,
    ).status_code == 400


def test_test_text_runs_from_the_web(runnable, monkeypatch):
    """CLI มี test-text มาตลอด — เว็บต้องเรียกได้ด้วย ไม่งั้นต้องสลับกลับไป terminal"""
    from lmds.web import jobs

    assert "test-text" in jobs.ALLOWED
    client = TestClient(create_app())
    done = _wait(client, client.post(f"/api/models/{runnable}/run/test-text").json()["id"])
    assert done["exit_code"] == 0
    assert "cmd test-text" in done["output"]


def test_stacked_commands_are_allowed(runnable):
    """stacked ต้องสั่ง sync-worker / verify-worker ได้ ไม่งั้นใช้เว็บกับ 2 node ไม่ได้เลย"""
    from lmds.web import jobs

    assert {"sync-worker", "verify-worker", "prepare-runtime", "clear-fi-cache"} <= jobs.ALLOWED


def test_repair_re_downloads_then_verifies(runnable):
    from lmds.web import jobs

    assert jobs.CHAINS["repair"] == ["download", "verify-files"]


def test_remove_and_autostart_are_token_guarded(runnable):
    client = TestClient(create_app(token="s3cret"))
    assert client.get(f"/api/models/{runnable}/removal-plan").status_code == 401
    assert client.post(f"/api/models/{runnable}/remove", json={}).status_code == 401
    assert client.post(f"/api/models/{runnable}/autostart", json={"enabled": True}).status_code == 401


def test_slots_option_reaches_the_controller(runnable):
    """client-config บ่นว่า context ต่อ slot เล็กเกิน — knob ที่มันบอกให้ปรับต้องปรับได้จากเว็บ"""
    from lmds.web import jobs

    env = jobs.controller_env({"slots": 2})
    assert env == {"PARALLEL_SEQS": "2", "MAX_NUM_SEQS": "2"}


def test_vision_test_is_allowed(runnable):
    from lmds.web import jobs

    assert "test-vision" in jobs.ALLOWED


# ── ปุ่มต้องตรงกับสิ่งที่ controller ทำได้จริง ─────────────────────────────────

def test_commands_are_read_from_the_controller_itself(tmp_path, monkeypatch):
    """bundle เก่าไม่มีคำสั่งใหม่ ๆ — เดาจาก profile ทำให้ปุ่มขึ้นแล้วกดล้ม
    (เจอจริง: test-vision ไม่โผล่ให้ผู้ใช้เพราะ bundle สร้างก่อนมีคำสั่งนี้)
    """
    from lmds.fleet import register_bundle
    from lmds.inventory import controller_commands as _controller_commands
    from tests.test_generator import gguf_report, make_bundle, mmproj_gguf_report

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    vision, _, _ = make_bundle(mmproj_gguf_report(), tmp_path=tmp_path / "mm")
    plain, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path / "plain")

    assert "test-vision" in _controller_commands(str(vision.controller))
    assert "test-vision" not in _controller_commands(str(plain.controller))
    # คำสั่งพื้นฐานต้องเจอครบทั้งคู่
    for command in ("download", "verify-files", "start", "stop", "logs", "test-text"):
        assert command in _controller_commands(str(plain.controller)), command


def test_unknown_commands_are_not_reported(tmp_path):
    """dispatch table มี help|--help|-h ปนอยู่ — ต้องไม่หลุดออกมาเป็นปุ่ม"""
    from lmds.inventory import controller_commands as _controller_commands

    script = tmp_path / "x-single.sh"
    script.write_text("#!/usr/bin/env bash\ncase \"$1\" in\n  start) start ;;\n"
                      "  rm-rf) danger ;;\n  help|--help) usage ;;\nesac\n", encoding="utf-8")
    assert _controller_commands(str(script)) == ["start"]


def test_missing_controller_reports_no_commands():
    from lmds.inventory import controller_commands as _controller_commands

    assert _controller_commands("/ไม่มี/ที่นี่.sh") == []


def test_model_payload_exposes_commands(runnable):
    model = TestClient(create_app()).get("/api/models").json()["models"][0]
    assert isinstance(model["commands"], list)


# ── fleet หลายเครื่อง ────────────────────────────────────────────────────────
@pytest.fixture
def registered(monkeypatch):
    """เครื่องหนึ่งเครื่องในทะเบียน + SSH ที่ถูก mock ทั้งหมด — เทสห้ามต่อออกนอกเครื่องจริง"""
    from lmds.nodes import Node, add

    add(Node(name="spark2", host="10.0.0.6", user="ops", cluster_ip="10.10.0.2"))

    fabric = {"links": [{"iface": "enp1s0f0np0", "ip": "10.10.0.2", "speed_gbps": 200,
                         "driver": "mlx5_core", "state": "up", "connectx": True, "rdma": True}],
              "rdma_devices": ["mlx5_0"], "best_gbps": 200, "tier": "rdma", "cluster_capable": True}
    payload = {
        "host": {"lmds_version": "0.1.0", "hostname": "spark2", "arch": "aarch64",
                 "profile": "dgx_spark", "gpus": [{"name": "NVIDIA GB10"}], "fabric": fabric,
                 "cpu": {"cores": 20, "percent": 8}, "ram_used_gb": 30, "ram_total_gb": 119},
        "models": [], "summary": {"total": 0, "running": 0, "healthy": 0, "not_downloaded": 0},
    }
    monkeypatch.setattr("lmds.nodes.probe", lambda node: payload)
    monkeypatch.setattr("lmds.nodes.ssh.probe", lambda node: payload)
    return payload


def test_nodes_list_exposes_cluster_ip(registered):
    node = TestClient(create_app()).get("/api/nodes").json()["nodes"][0]
    assert (node["name"], node["cluster_ip"]) == ("spark2", "10.10.0.2")
    assert "password" not in node


def test_node_registry_remove_requires_the_exact_name(registered, monkeypatch):
    from lmds.nodes import find

    client = TestClient(create_app())
    for body in ({}, {"confirm": True}, {"confirm": "SPARK2"}, {"confirm": "other"}):
        assert client.request("DELETE", "/api/nodes/spark2", json=body).status_code == 400
        assert find("spark2") is not None
    assert client.request("DELETE", "/api/nodes/spark2", json={"confirm": "spark2"}).status_code == 200
    assert find("spark2") is None


def test_node_registry_remove_rejects_unknown_options(registered):
    response = TestClient(create_app()).request(
        "DELETE", "/api/nodes/spark2", json={"confirm": "spark2", "purge": True},
    )
    assert response.status_code == 400


def test_node_inventory_reports_resources(registered):
    data = TestClient(create_app()).get("/api/nodes/spark2/inventory").json()
    assert data["reachable"] is True
    assert data["host"]["cpu"]["cores"] == 20
    assert data["host"]["fabric"]["best_gbps"] == 200


def test_unreachable_node_degrades_instead_of_failing(monkeypatch):
    """เครื่องล่มหนึ่งเครื่องต้องไม่ทำให้ทั้งหน้าเว็บพัง"""
    from lmds.nodes import Node, NodeError, add

    add(Node(name="down", host="10.0.0.9", user="ops"))
    monkeypatch.setattr("lmds.nodes.probe", lambda node: (_ for _ in ()).throw(NodeError("timeout")))
    data = TestClient(create_app()).get("/api/nodes/down/inventory").json()
    assert data["reachable"] is False
    assert "timeout" in data["error"]


def test_patch_sets_cluster_ip(registered):
    client = TestClient(create_app())
    r = client.patch("/api/nodes/spark2", json={"cluster_ip": "10.10.0.7"})
    assert r.status_code == 200
    assert client.get("/api/nodes").json()["nodes"][0]["cluster_ip"] == "10.10.0.7"


def test_patch_rejects_a_bad_cluster_ip(registered):
    r = TestClient(create_app()).patch("/api/nodes/spark2", json={"cluster_ip": "10.10.0"})
    assert r.status_code == 400


def test_patch_cannot_change_the_address(registered):
    """host/user แก้ที่นี่ไม่ได้ — ที่อยู่เปลี่ยน = คนละเครื่อง"""
    r = TestClient(create_app()).patch("/api/nodes/spark2", json={"host": "1.2.3.4"})
    assert r.status_code == 400


def test_patch_unknown_node_is_404(registered):
    assert TestClient(create_app()).patch("/api/nodes/nope", json={"note": "x"}).status_code == 404


def test_cluster_view_sends_codes_not_thai_sentences(registered, monkeypatch):
    """หน้าเว็บเป็นภาษาอังกฤษล้วน — API ต้องส่งรหัสสถานะให้ JS เรียบเรียงเอง"""
    data = TestClient(create_app()).get("/api/cluster").json()
    node = next(m for m in data["machines"] if m["name"] == "spark2")
    assert node["ip"]["state"] in {"ok", "unset", "mismatch", "slow"}
    assert node["fabric"]["best_gbps"] == 200
    for group in data["groups"]:
        for blocker in group["blockers"]:
            assert set(blocker) == {"kind", "names"}


def test_cluster_view_groups_matching_machines(registered, monkeypatch):
    from lmds.inventory import host_payload as real_host_payload

    hub = dict(real_host_payload())
    hub.update(registered["host"], hostname="spark1")
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: hub)

    data = TestClient(create_app()).get("/api/cluster").json()
    (group,) = data["groups"]
    assert sorted(m["name"] for m in group["members"]) == ["spark1", "spark2"]
    assert group["world_size"] == 2


def test_node_command_allowlist_blocks_anything_else(registered, monkeypatch):
    """ปุ่มบนหน้าเว็บสั่งข้ามเครื่องได้ — ต้องจำกัดคำสั่งไว้เท่าที่จำเป็น"""
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/rm-rf")
    assert r.status_code == 400


def test_node_remove_previews_before_it_deletes(registered, monkeypatch):
    """กดปุ่มเดียวแล้ว weight 71 GB หายไม่ได้ — คำขอแรกต้องเป็น --dry-run เสมอ"""
    seen = []
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: seen.append(command)
                        or SimpleNamespace(exit_code=0, stdout="จะลบ 71.2 GB", stderr=""))
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/remove")
    assert r.status_code == 200
    assert seen == ["lmds remove demo --dry-run"], "คำขอที่ไม่มี confirm ต้องไม่ลบอะไรเลย"


def test_node_remove_needs_the_exact_slug_to_confirm(registered, monkeypatch):
    """confirm ที่เดาได้ (เช่น true) ทำให้ปุ่มพลาดกลายเป็นการลบจริง — ต้องพิมพ์ชื่อให้ตรง"""
    called = []
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: called.append(command)
                        or SimpleNamespace(exit_code=0, stdout="", stderr=""))
    client = TestClient(create_app())
    for bad in ({"confirm": "yes"}, {"confirm": True}, {"confirm": "Demo"}):
        assert client.post("/api/nodes/spark2/models/demo/remove", json=bad).status_code == 400, bad
    assert not called

    ok = client.post("/api/nodes/spark2/models/demo/remove", json={"confirm": "demo"})
    assert ok.status_code == 200
    assert called == ["lmds remove demo -y"]


def test_node_logs_command_asks_for_a_bounded_number_of_lines(registered, monkeypatch):
    """logs ที่ไม่จำกัดบรรทัดคือการดึง log ทั้งไฟล์ข้ามเครือข่ายมาใส่เบราว์เซอร์"""
    sent = {}

    def fake_run(node, command, timeout=0):
        sent["command"] = command
        return SimpleNamespace(exit_code=0, stdout="log บรรทัดหนึ่ง", stderr="")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/logs")
    assert r.status_code == 200
    assert sent["command"] == "lmds logs demo -n 300"


def test_node_menu_commands_all_reach_the_node(registered, monkeypatch):
    """ทุกปุ่มในเมนู ⋯ ของหน้าเว็บต้องผ่าน allowlist จริง — ปุ่มที่กดแล้ว 400 แย่กว่าไม่มีปุ่ม"""
    seen = []

    def fake_run(node, command, timeout=0):
        seen.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    client = TestClient(create_app())
    menu = ["start", "stop", "restart", "doctor", "logs", "repair", "enable", "disable"]
    for command in menu:
        assert client.post(f"/api/nodes/spark2/models/demo/{command}").status_code == 200, command
    assert len(seen) == len(menu)


def test_page_only_suggests_commands_that_exist():
    """หน้าเว็บเคยแนะนำ `lmds deploy --topology stacked` ซึ่งไม่มีอยู่จริง —
    คนที่เชื่อหน้าเว็บจะพิมพ์แล้วเจอ error ทันที"""
    import re

    from typer.main import get_command

    from lmds.cli.main import app

    body = TestClient(create_app()).get("/").text
    known = set(get_command(app).commands)
    for match in re.finditer(r"lmds ([a-z-]+)((?: --?[a-z-]+)*)", body):
        command, flags = match.group(1), match.group(2)
        assert command in known, f"หน้าเว็บแนะนำคำสั่งที่ไม่มี: lmds {command}"
        if command == "deploy":
            assert "--topology" not in flags, "ไม่มี --topology — topology มาจาก --target"


def test_node_cards_can_be_collapsed():
    """เครื่องที่มีโมเดลเยอะทำให้หน้าจอยาวจนหาของไม่เจอ — ต้องย่อได้และจำสถานะไว้
    ไม่งั้น poll ทุก 5 วินาทีจะกางกลับเอง"""
    body = TestClient(create_app()).get("/").text
    assert 'class="ntoggle"' in body
    assert ".nbody.collapsed { display: none; }" in body
    assert "collapsedNodes" in body


def test_page_stays_readable_in_both_themes():
    """หน้าเว็บรันบนเครื่องลูกค้าที่ตั้ง theme มาแล้ว — ต้องอ่านออกทั้งสองโหมด
    ไม่ใช่ออกแบบให้สวยเฉพาะโหมดที่เราใช้เอง"""
    body = TestClient(create_app()).get("/").text
    assert "prefers-color-scheme: dark" in body
    # สีทุกตัวต้องมาจาก token — hex ที่ hard-code ในกฎ CSS จะเพี้ยนในอีกโหมดหนึ่ง
    import re

    inside_dark = body.split("prefers-color-scheme: dark")[1].split("}\n}")[0]
    assert "--bg:" in inside_dark and "--fg:" in inside_dark


def test_page_respects_accessibility_basics():
    """ops console ใช้คีย์บอร์ดเป็นหลักตอนแก้ปัญหา — focus ต้องเห็น
    และ motion ต้องปิดได้สำหรับคนที่ตั้งค่าไว้"""
    body = TestClient(create_app()).get("/").text
    assert ":focus-visible" in body
    assert "prefers-reduced-motion: reduce" in body


def test_destructive_button_is_quiet_until_hovered():
    """ปุ่มลบไม่ควรดึงสายตาไปกว่าปุ่มที่ใช้ทุกวัน — เด่นตอนจะกดจริงก็พอ"""
    body = TestClient(create_app()).get("/").text
    assert "button.danger { color: var(--fg2)" in body
    # ปุ่มที่ลบของจริง ๆ ต้องติด class danger ไว้ — ตัวที่ลบเครื่องออกจากทะเบียนอยู่ในเมนู ⋯
    assert 'class="danger">Remove from registry</button>' in body
    assert 'style="border-color:var(--bad);color:var(--bad)"' in body   # ยืนยันการลบโมเดล


def test_events_endpoint_exists_as_a_stream():
    """หน้าเว็บต้องไม่ poll — server push ให้แทน · เดิม poll ทุก 5 วิ = SSH ทุกเครื่องทุกรอบ

    ไม่เปิดสตรีมจริงในเทส เพราะ TestClient จะค้างรอ generator ที่ออกแบบให้ไม่จบเอง
    (มันจบเมื่อ "ลูกค้าตัดสาย" ซึ่ง TestClient ไม่ทำ) — ตรวจว่า route มีจริงและเป็น async
    ส่วนตรรกะจริงอยู่ใน state.Store ซึ่งเทสแยกไว้แล้ว
    """
    import inspect

    app = create_app()
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/events")
    assert inspect.iscoroutinefunction(route.endpoint), \
        "ต้องเป็น async ไม่งั้น thread ค้างหลังผู้ใช้ปิดแท็บ"


def test_store_notifies_only_when_something_changes():
    """SSE ส่งเฉพาะตอนมีของเปลี่ยนจริง — ไม่งั้นก็แค่ poll ที่ย้ายไปอยู่ฝั่ง server"""
    from lmds.web.state import Store

    store = Store()
    first = store.version
    store.set_local({"host": {}, "models": []})
    assert store.version != first

    after = store.version
    assert store.wait_for_change(first, timeout=0.1) is True     # เปลี่ยนไปแล้ว รู้ทันที
    assert store.wait_for_change(after, timeout=0.1) is False    # ยังไม่มีอะไรใหม่


def test_unreachable_node_backs_off():
    """เครื่องที่ปิดอยู่ต้องไม่ถูกยิง SSH ทุก 15 วิไปเรื่อย ๆ — ถอยห่างขึ้นเรื่อย ๆ
    แล้วกลับมาถี่ปกติทันทีที่ต่อได้"""
    from lmds.web.state import NODE_INTERVAL, Store

    store = Store()
    store.set_node("down", None, "หมดเวลา")
    first = store._nodes["down"].interval
    store.set_node("down", None, "หมดเวลา")
    assert store._nodes["down"].interval > first

    store.set_node("down", {"host": {}, "models": []})
    assert store._nodes["down"].interval == NODE_INTERVAL


def test_endpoints_never_block_on_a_slow_node(monkeypatch):
    """เครื่องหนึ่งช้าหรือล่มต้องไม่ทำให้ทั้งหน้าเว็บรอ — endpoint อ่านจากแคชเสมอ"""
    import time

    from lmds.nodes import Node, add
    from lmds.web import state

    add(Node(name="slowpoke", host="10.0.0.9", user="ops"))
    state.STORE.set_node("slowpoke", None, "หมดเวลา 60s")

    started = time.monotonic()
    data = TestClient(create_app()).get("/api/nodes/slowpoke/inventory").json()
    assert time.monotonic() - started < 2.0, "endpoint ไปรอ SSH แทนที่จะอ่านแคช"
    assert data["reachable"] is False


def test_cache_is_dropped_after_a_state_change(fleet):
    """กด stop แล้วต้องเห็นผลทันที ไม่ใช่รอ refresher รอบถัดไปอีก 15 วิ"""
    from lmds.web import state

    client = TestClient(create_app())
    client.get("/api/models")
    state.STORE.set_local({"host": {}, "models": [{"slug": "ของเก่า"}]})
    client.post(f"/api/models/{fleet}/stop")
    slugs = [m["slug"] for m in client.get("/api/models").json()["models"]]
    assert "ของเก่า" not in slugs


def test_gpu_telemetry_hides_values_the_card_does_not_report():
    """GB10 (unified SoC) ไม่รายงาน power limit / fan / memory clock —
    โชว์ 0 ตรงนั้นคือการโกหก · เกจต้องหายไปเลยเมื่อค่าเป็น null"""
    body = TestClient(create_app()).get("/").text
    assert "function gauge(" in body
    assert 'if (value === null) return "";' in body, "ต้องซ่อนเกจที่ไม่มีค่า ไม่ใช่วาดเป็น 0"
    assert "N/A (SoC)" in body, "unified memory ต้องบอกว่าเป็น SoC ไม่ใช่โชว์ 0 GB"


def test_temperature_is_colour_coded():
    """อุณหภูมิเป็นค่าเดียวที่ 'สูง = อันตราย' จริง — ต้องเห็นได้โดยไม่ต้องอ่านตัวเลข"""
    body = TestClient(create_app()).get("/").text
    assert "function tempColour(" in body
    assert "var(--bad)" in body.split("function tempColour(")[1][:200]


def test_command_output_is_not_wiped_by_live_updates():
    """SSE ส่งสถานะทุก 1-3 วิ — ถ้าเขียนทับทั้งแถว ผลของ doctor จะหายก่อนผู้ใช้อ่านจบ
    (เจอจริงหลังเปลี่ยนมาใช้ SSE: กด doctor แล้วเหมือนไม่มีอะไรเกิดขึ้น)"""
    body = TestClient(create_app()).get("/").text
    assert "pinnedOutput" in body
    assert "busy || pinnedOutput.has(name)" in body
    # ต้องมีทางปิดกลับไปหน้าปกติ ไม่งั้นค้างอยู่กับผลเก่า
    assert 'data-nact="close-output"' in body


def test_manual_refresh_bypasses_the_cache():
    """กด refresh แล้วต้องได้ของสด ไม่ใช่ค่าเดิมจากแคชที่เพิ่งอ่านมา"""
    body = TestClient(create_app()).get("/").text
    assert "/inventory?refresh=true" in body


def test_page_javascript_parses():
    """JS พังหนึ่งตัวอักษร = หน้าเว็บขาวทั้งหน้า ไม่มี error ให้เห็นนอกจากเปิด console

    เทสอื่นเรียก API ได้หมดโดยที่หน้าเว็บใช้ไม่ได้เลย — gate นี้จึงเป็นตัวเดียวที่จับได้
    """
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("ไม่มี node บนเครื่องนี้")
    html = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert blocks, "หน้าเว็บต้องมี <script> — ถ้าไม่มีแปลว่าโครงหน้าเปลี่ยนไปแล้ว"
    for i, block in enumerate(blocks):
        result = subprocess.run([node, "--check", "-"], input=block, capture_output=True, text=True)
        assert result.returncode == 0, f"script block {i} พัง:\n{result.stderr[:800]}"


def test_node_start_passes_validated_port_and_context(registered, monkeypatch):
    """หน้าเว็บต้องตั้ง port/context ตอนสั่งรันบนเครื่องอื่นได้ — ไม่งั้นต้องไป ssh แก้ .sh เอง"""
    sent = {}

    def fake_run(node, command, timeout=0):
        sent["command"] = command
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    r = TestClient(create_app()).post(
        "/api/nodes/spark2/models/demo/start", json={"port": 8001, "context": 32768, "gpu_util": 0.8})
    assert r.status_code == 200
    assert sent["command"] == "lmds start demo --port 8001 --context 32768 --gpu-util 0.8"


def test_node_options_are_validated_on_the_server(registered, monkeypatch):
    """ค่าพวกนี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นผ่าน SSH — ตรวจที่ server เท่านั้นที่นับ
    ฝากไว้กับ JS ในเบราว์เซอร์ไม่ได้ ใครก็ยิง API ตรงข้ามได้
    """
    called = []
    monkeypatch.setattr("lmds.nodes.run",
                        lambda *a, **k: called.append(a) or SimpleNamespace(exit_code=0, stdout="", stderr=""))
    client = TestClient(create_app())
    for bad in ({"port": "8001; rm -rf /"}, {"port": 0}, {"port": 70000}, {"port": True},
                {"port": 8001.5}, {"context": -1}, {"gpu_util": 1.5}, {"context": "abc"},
                {"unexpected": 1}):
        assert client.post("/api/nodes/spark2/models/demo/start", json=bad).status_code == 400, bad
    assert not called, "ค่าที่ไม่ผ่านการตรวจต้องไม่ถูกส่งไปเครื่องปลายทางเลย"


def test_node_options_are_rejected_for_commands_that_ignore_them(registered, monkeypatch):
    """ส่ง port ไปกับ doctor แล้วเงียบ ๆ ทิ้ง = ผู้ใช้เข้าใจว่าตั้งค่าแล้วทั้งที่ไม่ได้ตั้ง"""
    monkeypatch.setattr("lmds.nodes.run",
                        lambda *a, **k: SimpleNamespace(exit_code=0, stdout="", stderr=""))
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/doctor", json={"port": 8001})
    assert r.status_code == 400


def test_live_updates_pause_while_a_node_row_is_in_use():
    """SSE ส่ง snapshot ทุก ~1 วิ แล้ววาดทับ body ของแถวเครื่อง — ตัวเลข port/context
    ที่ผู้ใช้เพิ่งพิมพ์จะหายทุกวินาทีจนกรอกไม่ทัน (ผู้ใช้เจอจริง)

    เทสนี้ตรวจว่า guard ยังอยู่ครบ · JS จริงต้องรันในเบราว์เซอร์ถึงจะทดสอบได้ แต่การลบ
    เงื่อนไขทิ้งโดยไม่ตั้งใจคือสิ่งที่จับได้ที่นี่
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "nodeIsInUse(name)" in page, "applySnapshot ต้องข้ามแถวที่ผู้ใช้กำลังใช้อยู่"
    assert "openModelMenus" in page and "document.activeElement" in page, \
        "ต้องเช็กทั้งเมนูที่กางอยู่และช่องกรอกที่โฟกัสอยู่"
    assert "markStale" in page, "หยุดอัปเดตแล้วต้องบอกผู้ใช้ว่าตัวเลขเริ่มเก่า"


def test_autostart_badge_is_not_shown_for_every_model():
    """autostart เป็นสตริง enabled|disabled|absent|n/a — ทุกตัว truthy ใน JS

    เช็กแบบ `m.autostart ? …` จึงติดป้าย "autostart" ให้ทุกโมเดลและเสนอปุ่ม disable
    ทั้งที่ไม่เคยเปิดเลย (ผู้ใช้เจอจริงบน dgx-veerasiam) — สถานะที่ผิดแย่กว่าไม่มีสถานะ
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert 'm.autostart ? `<span class="tag">autostart</span>`' not in page
    assert 'auto === "enabled"' in page, "ต้องเทียบค่ากับ enabled ตรง ๆ"
    assert 'auto === "n/a"' in page, "เครื่องที่ไม่มี systemd ต้องไม่มีปุ่มที่กดแล้วล้มแน่ ๆ"


# ── ติดตั้ง LMDS บนเครื่องอื่นจากหน้าเว็บ ─────────────────────────────────────
def _wait_job(client, job_id, tries=200):
    for _ in range(tries):
        data = client.get(f"/api/jobs/{job_id}").json()
        if not data["running"]:
            return data
        time.sleep(0.02)
    raise AssertionError("job ไม่จบสักที")


def test_web_can_install_lmds_on_a_node(registered, monkeypatch):
    """เครื่องที่ยังไม่มี lmds ขึ้นว่าติดต่อไม่ได้ตลอด — คนใช้หน้าเว็บอย่างเดียวเคยไม่มีทางออก

    hub ไม่ได้ส่ง agent ไปรัน แต่เรียก `lmds agent info` บนเครื่องนั้น เครื่องที่ยังไม่ได้ลง
    จึงต้องมีคน ssh เข้าไปลงเองก่อน ซึ่งขัดกับที่หน้าเว็บมีไว้เพื่อไม่ต้อง ssh
    """
    from lmds.nodes.ssh import Result

    calls = []

    def fake_install(node, timeout=1800, with_prereq=False):
        calls.append((node.name, with_prereq))
        return Result(0, "Cloning...\nlmds 0.1.0\n", "")

    monkeypatch.setattr("lmds.nodes.install_lmds", fake_install)

    client = TestClient(create_app())
    job = client.post("/api/nodes/spark2/install", json={}).json()
    done = _wait_job(client, job["id"])

    assert done["exit_code"] == 0
    assert calls == [("spark2", False)]
    assert "lmds 0.1.0" in done["output"]
    assert "พร้อมแล้ว" in done["output"]


def test_install_defaults_to_skipping_sudo_steps(registered, monkeypatch):
    """--with-prereq ต้องใช้ sudo แบบไม่ถามรหัส — หน้าเว็บไม่มี tty ให้กรอก ค้างแล้ว timeout"""
    from lmds.nodes.ssh import Result

    seen = {}
    monkeypatch.setattr(
        "lmds.nodes.install_lmds",
        lambda node, timeout=1800, with_prereq=False: (
            seen.update(with_prereq=with_prereq) or Result(0, "ok", "")
        ),
    )
    client = TestClient(create_app())
    _wait_job(client, client.post("/api/nodes/spark2/install", json={}).json()["id"])
    assert seen["with_prereq"] is False


def test_failed_install_reports_the_real_output(registered, monkeypatch):
    """exit code เปล่า ๆ ไม่ช่วยอะไร — ต้องเห็นว่า install.sh บ่นอะไร"""
    from lmds.nodes.ssh import Result

    monkeypatch.setattr(
        "lmds.nodes.install_lmds",
        lambda node, timeout=1800, with_prereq=False: Result(1, "", "git: command not found"),
    )
    client = TestClient(create_app())
    done = _wait_job(client, client.post("/api/nodes/spark2/install", json={}).json()["id"])
    assert done["exit_code"] == 1
    assert "git: command not found" in done["output"]


def test_install_on_an_unknown_node_is_404(registered):
    assert TestClient(create_app()).post("/api/nodes/nope/install", json={}).status_code == 404


@pytest.mark.parametrize("body", [
    {"with_prereq": "false"}, {"with_prereq": 1}, {"unknown": True},
])
def test_node_install_rejects_ambiguous_options(registered, body):
    assert TestClient(create_app()).post("/api/nodes/spark2/install", json=body).status_code == 400


@pytest.mark.parametrize("port", [0, 65536, True, "not-a-port"])
def test_node_add_rejects_invalid_ssh_ports(monkeypatch, port):
    called = []
    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: called.append("key"))
    response = TestClient(create_app()).post("/api/nodes", json={
        "host": "node.example", "user": "ops", "port": port,
    })
    assert response.status_code == 400
    assert called == [], "invalid input must be rejected before any SSH/key side effect"


@pytest.mark.parametrize("body", [
    {"host": ["node.example"], "user": "ops"},
    {"host": "node.example", "user": {"name": "ops"}},
    {"host": "node.example", "user": "ops", "password": True},
    {"host": "node.example", "user": "ops", "note": ["prod"]},
    {"host": "node.example", "user": "ops", "unexpected": "ignored-before"},
])
def test_node_add_rejects_malformed_or_unknown_fields_before_side_effects(monkeypatch, body):
    called = []
    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: called.append("key"))
    response = TestClient(create_app()).post("/api/nodes", json=body)
    assert response.status_code == 400
    assert called == []


def test_node_add_rejects_invalid_registry_name_before_side_effects(monkeypatch):
    called = []
    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: called.append("key"))
    response = TestClient(create_app()).post("/api/nodes", json={
        "host": "node.example", "user": "ops", "name": "bad name",
    })
    assert response.status_code == 422
    assert called == []


@pytest.mark.parametrize("host,user", [
    ("node.example", "-oProxyCommand=bad"),
    ("node.example", "ops user"),
    ("-Fmalicious", "ops"),
    ("ops@node.example", "ops"),
    ("node\x00.example", "ops"),
])
def test_node_add_rejects_ambiguous_ssh_destinations_before_side_effects(monkeypatch, host, user):
    called = []
    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: called.append("key"))
    response = TestClient(create_app()).post("/api/nodes", json={
        "host": host, "user": user, "port": 22,
    })
    assert response.status_code == 422
    assert called == []


def test_two_installs_on_the_same_node_do_not_overlap(registered, monkeypatch):
    """git pull ซ้อน git pull บนเครื่องเดียวกัน = repo พัง"""
    from lmds.nodes.ssh import Result

    release = threading.Event()
    monkeypatch.setattr(
        "lmds.nodes.install_lmds",
        lambda node, timeout=1800, with_prereq=False: (release.wait(5), Result(0, "ok", ""))[1],
    )
    client = TestClient(create_app())
    first = client.post("/api/nodes/spark2/install", json={})
    assert first.status_code == 200
    second = client.post("/api/nodes/spark2/install", json={})
    assert second.status_code == 409
    release.set()
    _wait_job(client, first.json()["id"])


def test_node_job_key_does_not_collide_with_a_model_slug(registered, monkeypatch):
    """งานของเครื่องใช้ key คนละ namespace กับ slug ของโมเดล ไม่งั้นบล็อกกันเอง"""
    from lmds.nodes.ssh import Result

    monkeypatch.setattr(
        "lmds.nodes.install_lmds",
        lambda node, timeout=1800, with_prereq=False: Result(0, "ok", ""),
    )
    client = TestClient(create_app())
    done = _wait_job(client, client.post("/api/nodes/spark2/install", json={}).json()["id"])
    assert done["slug"] == "node:spark2"


# ── ที่อยู่สำรอง (alt_hosts) ──────────────────────────────────────────────────
def test_alt_hosts_can_be_set_from_the_web(registered):
    """เครื่องเดียวกันเข้าได้หลายทาง (LAN ที่ออฟฟิศ, VPN ตอนออกนอก) — เดิมมีแต่ `lmds node set`"""
    client = TestClient(create_app())
    data = client.patch("/api/nodes/spark2", json={"alt_hosts": "100.64.0.6, spark2.local"}).json()
    assert data["alt_hosts"] == ["100.64.0.6", "spark2.local"]
    assert client.get("/api/nodes").json()["nodes"][0]["alt_hosts"] == ["100.64.0.6", "spark2.local"]


def test_alt_hosts_accepts_a_list_too(registered):
    client = TestClient(create_app())
    data = client.patch("/api/nodes/spark2", json={"alt_hosts": ["a.local", "b.local"]}).json()
    assert data["alt_hosts"] == ["a.local", "b.local"]


def test_alt_hosts_can_be_cleared(registered):
    client = TestClient(create_app())
    client.patch("/api/nodes/spark2", json={"alt_hosts": "x.local"})
    assert client.patch("/api/nodes/spark2", json={"alt_hosts": ""}).json()["alt_hosts"] == []


def test_blank_entries_are_dropped(registered):
    """พิมพ์จุลภาคเกินมาไม่ควรกลายเป็นที่อยู่ว่างที่ ssh พยายามต่อ"""
    client = TestClient(create_app())
    data = client.patch("/api/nodes/spark2", json={"alt_hosts": "a.local, , ,b.local,"}).json()
    assert data["alt_hosts"] == ["a.local", "b.local"]


@pytest.mark.parametrize("body", [
    {"alt_hosts": {"host": "a.local"}},
    {"alt_hosts": ["a.local", 3]},
    {"alt_hosts": "bad host"},
    {"cluster_ip": ["10.10.0.2"]},
    {"note": "ok", "host": "replacement.example"},
])
def test_node_patch_rejects_malformed_or_unknown_fields(registered, body):
    assert TestClient(create_app()).patch("/api/nodes/spark2", json=body).status_code == 400


# ── หน้าเว็บต้อง parse ได้ ────────────────────────────────────────────────────
def _page_script() -> str:
    from lmds.web import api as web_api

    html = (Path(web_api.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return html.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def test_page_script_parses():
    """สคริปต์ทั้งหน้าอยู่ในไฟล์เดียว — พิมพ์ผิดจุดเดียวคือทั้งหน้าตาย ไม่ใช่แค่ฟีเจอร์เดียว

    เคสจริงตอนเขียน PR นี้: template literal ที่ควรเป็น "\n" กลายเป็นขึ้นบรรทัดใหม่จริง
    ผลคือ "Other machines" ค้างที่ Loading… ตลอดกาล และไม่มีเทสไหนจับได้เลย
    เพราะเทสฝั่ง API ผ่านหมด — เบราว์เซอร์ต่างหากที่พัง
    """
    script = _page_script()
    checker = shutil.which("node") or shutil.which("nodejs")
    if checker is None:
        # ไม่มี node ก็ยังจับเคสที่เจอจริงได้: backtick/`${` ที่ไม่ปิด
        assert script.count("`") % 2 == 0, "backtick ไม่ครบคู่ — template literal ค้าง"
        pytest.skip("ไม่มี node ในเครื่องนี้ — ตรวจได้แค่ backtick")
    # ส่งเป็นไฟล์ UTF-8 ไม่ใช่ stdin — โค้ดหน้าเว็บมีทั้งไทยและอังกฤษ ส่วน stdin ของ
    # subprocess ใช้ encoding ของ locale ซึ่งบน Windows ไทยคือ cp874 แล้วพังตั้งแต่ยังไม่ตรวจ
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run([checker, "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:800]


def test_page_has_no_native_dialogs():
    """alert()/confirm()/prompt() บล็อกทั้งหน้า จัดสไตล์ไม่ได้ และบนมือถือขึ้นชื่อโดเมนนำหน้า

    มี openModal() ที่ใช้ <dialog> อยู่แล้ว — เทสนี้กันไม่ให้เผลอกลับไปใช้ของเดิม
    """
    script = _page_script()
    code = "\n".join(line for line in script.splitlines() if not line.strip().startswith("//"))
    for name in ("alert(", "confirm(", "prompt("):
        assert name not in code, f"ยังมี {name} อยู่ — ใช้ say()/ask()/askFor() แทน"


def test_untrusted_web_values_are_escaped_and_urls_are_scheme_checked():
    """remote agent/profile/catalog data must not become active HTML in the operator browser"""
    checker = shutil.which("node") or shutil.which("nodejs")
    if checker is None:
        pytest.skip("ไม่มี node สำหรับรัน helper ฝั่ง browser")
    script = _page_script()
    helpers = script[script.index("const esc ="):script.index("const open =")]
    payload = '<img src=x onerror="globalThis.pwned=1">'
    probe = helpers + f"\nconsole.log(JSON.stringify({{escaped: esc({payload!r}), " \
        f"number: finite({payload!r}), unsafe: safeHref('javascript:alert(1)'), " \
        "safe: safeHref('https://example.com/model')}));\n"
    result = subprocess.run([checker, "-e", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import json

    data = json.loads(result.stdout)
    assert data["escaped"].startswith("&lt;img") and "<img" not in data["escaped"]
    assert data["number"] is None
    assert data["unsafe"] == ""
    assert data["safe"] == "https://example.com/model"

    # modal body is the common sink for API/SSH errors; job/log output has separate safe sinks
    assert '${body ? `<div>${esc(body)}</div>` : ""}' in script
    assert '${esc(d.output || "starting…")}' in script
    assert "box.textContent = body" in script


def test_job_polling_does_not_strand_ui_on_network_failure():
    """fetch rejection must be handled: otherwise watching/button state remains stuck until reload."""
    script = _page_script()
    follow = script[script.index("async function followJob"):script.index("async function waitForJob")]
    wait = script[script.index("async function waitForJob"):script.index("// ── ตัวดู log")]
    assert "catch (_)" in follow
    assert "watching.delete(slug)" in follow
    assert "failures < 3" in follow
    assert "catch (_)" in wait
    assert "Lost contact while waiting for the job" in wait


def test_page_ships_without_any_external_request():
    """เครื่องเป้าหมายมักอยู่หลัง proxy หรือตัดเน็ต — CDN ตัวเดียวคือหน้าเว็บพังตรงที่ต้องใช้"""
    from lmds.web import api as web_api

    html = (Path(web_api.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    for marker in ("http://", "https://", "//cdn", "src=", "@import"):
        if marker in ("http://", "https://"):
            # ยอมให้มีลิงก์ที่ผู้ใช้กดเอง (target=_blank) แต่ห้ามโหลดทรัพยากรจากข้างนอก
            continue
        assert marker not in html.replace('src="${', ""), f"หน้าเว็บอ้างของข้างนอก: {marker}"
