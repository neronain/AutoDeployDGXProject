"""เทส Web UI — ชั้นนี้ต้องไม่มี logic ของตัวเอง

กฎที่เทสไว้: อะไรที่ CLI ทำได้ เว็บต้องได้ผลเหมือนกัน และหน้าที่สั่ง start/stop ได้
ต้องไม่เปิดโล่งให้ทั้งวง network โดยบังเอิญ
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.web import create_app  # noqa: E402


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


def test_no_token_configured_means_open(fleet):
    """bind 127.0.0.1 (ค่าเริ่มต้น) ไม่ต้องมี token — เข้าถึงได้เฉพาะเครื่องนี้อยู่แล้ว"""
    assert TestClient(create_app()).get("/api/models").status_code == 200


def test_actions_call_fleet_and_report_failure(fleet, monkeypatch):
    calls = []
    monkeypatch.setattr("lmds.fleet.start_server", lambda info: calls.append("start") or 0)
    monkeypatch.setattr("lmds.fleet.stop_server", lambda info: calls.append("stop") or "controller")
    client = TestClient(create_app())

    assert client.post(f"/api/models/{fleet}/start").json()["ok"] is True
    assert client.post(f"/api/models/{fleet}/stop").json()["ok"] is True
    assert calls == ["start", "stop"]

    monkeypatch.setattr("lmds.fleet.start_server", lambda info: 1)
    failed = client.post(f"/api/models/{fleet}/start")
    assert failed.status_code == 500
    assert failed.json()["ok"] is False


def test_fleet_error_becomes_409(fleet, monkeypatch):
    from lmds.fleet import FleetError

    def boom(info):
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
    assert jobs.controller_env({"port": None, "api_key": "", "context": 0}) == {}


def test_start_passes_options_through(runnable, monkeypatch):
    import os

    seen = {}

    def fake_start(info):
        seen.update({k: os.environ.get(k) for k in ("API_PORT", "API_KEY")})
        return 0

    monkeypatch.setattr("lmds.fleet.start_server", fake_start)
    TestClient(create_app()).post(f"/api/models/{runnable}/start",
                                  json={"port": 8123, "api_key": "abc"})
    assert seen == {"API_PORT": "8123", "API_KEY": "abc"}
    # ต้องคืน environment ให้เหมือนเดิม ไม่ทิ้งค่าไว้กระทบคำสั่งถัดไป
    assert os.environ.get("API_PORT") is None


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
    _wait(client, client.get("/api/models").json()["models"][0]["job"]["id"]) \
        if client.get("/api/models").json()["models"][0].get("job") else None

    withw = client.get(f"/api/models/{runnable}/removal-plan").json()
    keep = client.get(f"/api/models/{runnable}/removal-plan?keep_weights=true").json()
    assert not any(i["is_weights"] for i in keep["items"])
    assert keep["total_bytes"] <= withw["total_bytes"]


def test_remove_deletes_and_model_disappears(runnable, tmp_path):
    client = TestClient(create_app())
    assert client.post(f"/api/models/{runnable}/remove", json={"keep_weights": True}).status_code == 200
    assert [m["slug"] for m in client.get("/api/models").json()["models"]] == []


def test_autostart_failure_tells_you_the_manual_commands(runnable, monkeypatch):
    """เว็บไม่มี tty ให้กรอกรหัส sudo — ต้องส่งคำสั่งกลับไปให้ผู้ใช้รันเอง ไม่ใช่ 500 เปล่า ๆ"""
    from lmds.fleet import FleetError

    def boom(info, timeout=1800, start_now=False):
        raise FleetError("ติดตั้ง autostart ไม่สำเร็จ\nลองรันมือ:\n  sudo systemctl enable lmds-demo")

    monkeypatch.setattr("lmds.fleet.enable_autostart", boom)
    r = TestClient(create_app()).post(f"/api/models/{runnable}/autostart", json={"enabled": True})
    assert r.status_code == 409
    assert "sudo systemctl enable" in r.json()["detail"]


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
    assert 'class="danger">forget</button>' in body
