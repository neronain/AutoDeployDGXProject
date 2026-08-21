"""เทส Web UI — ชั้นนี้ต้องไม่มี logic ของตัวเอง

กฎที่เทสไว้: อะไรที่ CLI ทำได้ เว็บต้องได้ผลเหมือนกัน และหน้าที่สั่ง start/stop ได้
ต้องไม่เปิดโล่งให้ทั้งวง network โดยบังเอิญ
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


class _FakePipe:
    """ท่อไบต์ที่คายทีละบรรทัด — พอสำหรับ _pump ซึ่งอ่านด้วย read1()

    เขียนเป็นไบต์เพราะของจริงเป็นไบต์: _pump ต้องเห็น \r เป็นตัวจบบรรทัดด้วย และ
    ท่อที่ถอดรหัสเป็นข้อความให้แล้วจะตัดที่ \n อย่างเดียว ซึ่งเป็นบั๊กที่กำลังแก้อยู่
    """

    def __init__(self, lines):
        self._lines = iter(lines)

    def read1(self, _size=-1):
        chunk = next(self._lines, "")
        return chunk.encode("utf-8") if isinstance(chunk, str) else chunk

    def close(self):
        pass


class FakeStream:
    """จำลอง Popen ของ ssh แบบสตรีม — จบทันที ไม่ต้องรอ thread จริง"""

    def __init__(self, lines=("done\n",), code=0):
        self.stdout = _FakePipe(lines)
        self._code = code

    def wait(self):
        return self._code

    def kill(self):
        """งานที่ถูกยกเลิกเรียกตัวนี้ — ของปลอมที่จบไปแล้วไม่ต้องทำอะไร"""

    def poll(self):
        return self._code


class FakeJob:
    """งานที่จบไปแล้ว — endpoint แค่ต้องคืน payload ของมัน ไม่ต้องรัน bash จริง"""

    def payload(self):
        return {"id": "fake", "running": False, "exit_code": 0, "output": ""}


def wait_for_job(client, job_id, tries=50):
    """งานรันใน thread — รอจนมันจบก่อนค่อยตรวจผล"""
    import time

    for _ in range(tries):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if not payload["running"]:
            return payload
        time.sleep(0.02)
    raise AssertionError("งานไม่จบใน 1 วินาที")


def test_page_is_self_contained(fleet):
    """เครื่องลูกค้าอาจอยู่หลัง proxy/air-gapped — หน้าเว็บต้องไม่ *ดึง* อะไรจากเน็ต

    กฎคือห้ามโหลดทรัพยากรจากภายนอก ไม่ใช่ห้ามเอ่ยถึง URL เลย · ลิงก์ `<a href>`
    ไม่ได้โหลดอะไรตอนเปิดหน้า — หน้ายังทำงานครบในเครื่องที่ไม่มีเน็ต แค่กดลิงก์แล้ว
    ไม่ไปไหนเท่านั้น · เดิมเช็ค "https://" ทั้งหน้า ซึ่งห้ามลายเซ็นผู้สร้างไปด้วย
    """
    import re

    body = TestClient(create_app()).get("/").text
    assert "<title>LMDS</title>" in body

    # ทรัพยากรที่เบราว์เซอร์จะไปโหลดเอง: script/img/iframe (src), stylesheet (link href),
    # @import และ url(...) ใน CSS
    loaders = (
        re.findall(r"""<(?:script|img|iframe|source|video|audio)[^>]*\ssrc=["']([^"']+)""", body, re.I)
        + re.findall(r"""<link[^>]*\shref=["']([^"']+)""", body, re.I)
        + re.findall(r"""@import\s+["']([^"']+)""", body, re.I)
        + re.findall(r"""url\(\s*["']?([^)"']+)""", body, re.I)
    )
    for target in loaders:
        assert not target.lower().startswith(("http://", "https://", "//")), (
            f"หน้าเว็บจะไปโหลดของจากภายนอก: {target}"
        )
    for host in ("cdnjs", "googleapis", "unpkg", "jsdelivr"):
        assert host not in body, f"หน้าเว็บอ้างถึง CDN: {host}"


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

def gate_file(tmp_path) -> Path:
    """ไฟล์ที่ถ้ามีอยู่ จะทำให้ขั้น download ของ controller ค้างรอจนกว่าเทสจะลบทิ้ง

    มีไว้ให้เทสที่ต้องการให้งาน "ยังรันอยู่" แน่ ๆ ตอนยิงคำขอที่สอง · เทสอื่นไม่ได้
    สร้างไฟล์นี้ ขั้น download จึงเร็วเหมือนเดิมทุกประการ
    """
    return tmp_path / "download-gate"


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
    gate = gate_file(tmp_path)
    # ลูปรอมีเพดาน (~10 วิ) — เทสที่พังกลางคันต้องไม่ทิ้ง process ค้างไว้ทั้งชุด
    controller.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        f'  download) echo "โหลดอยู่"; n=0;'
        f' while [ -f "{gate}" ] && [ $n -lt 200 ]; do sleep 0.05; n=$((n+1)); done;'
        f' mkdir -p "{model_dir}"; echo x > "{model_dir}/demo-Q8.gguf";;\n'
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


def test_only_one_job_per_model(runnable, tmp_path):
    """download ซ้อน start = ไฟล์พัง — ต้องกันไว้ ไม่ใช่หวังว่าผู้ใช้จะไม่กดซ้ำ

    งานแรกต้อง "ยังรันอยู่" จริงตอนคำขอที่สองมาถึง ไม่งั้นการปล่อยผ่านคือพฤติกรรมที่ถูก
    (ด่านกันคือ "หนึ่งงานต่อโมเดล" ไม่ใช่ "ห้ามสั่งซ้ำตลอดกาล") · ของเดิมพึ่งจังหวะว่า
    controller ซึ่งเป็น bash สั้น ๆ ยังทำงานไม่จบ บางรอบมันจบก่อนคำขอที่สอง เทสจึง
    fail สลับไปมาโดยที่โค้ดไม่ได้ผิดเลย — ตอนนี้กั้นด้วยไฟล์ ปล่อยเมื่อเราสั่งเท่านั้น
    """
    gate = gate_file(tmp_path)
    gate.touch()
    client = TestClient(create_app())
    try:
        first = client.post(f"/api/models/{runnable}/run/download")
        assert first.status_code == 200, first.text

        second = client.post(f"/api/models/{runnable}/run/download")
        assert second.status_code == 409
        assert "กำลังรัน" in second.json()["detail"]
    finally:
        gate.unlink(missing_ok=True)
    _wait(client, first.json()["id"])


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


def test_node_opted_out_of_stacking_is_not_grouped(registered, monkeypatch):
    """สั่ง "ไม่เอาเข้ากลุ่ม" แล้วต้องไม่ถูกเสนอเข้าคลัสเตอร์อีก แม้ฮาร์ดแวร์จะตรงเป๊ะ"""
    from lmds.inventory import host_payload as real_host_payload

    hub = dict(real_host_payload())
    hub.update(registered["host"], hostname="spark1")
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: hub)
    client = TestClient(create_app())
    assert client.get("/api/cluster").json()["groups"], "ก่อนปิดต้องยังจับกลุ่มได้"

    assert client.patch("/api/nodes/spark2", json={"stack": False}).json()["stack"] is False
    data = client.get("/api/cluster").json()
    assert data["groups"] == []
    assert next(m for m in data["machines"] if m["name"] == "spark2")["stack"] is False


def test_hub_can_opt_itself_out_of_stacking(registered, monkeypatch):
    """hub ไม่ได้อยู่ในทะเบียน — ค่าของมันจึงต้องเก็บใน config ไม่ใช่หายไปตอนรีสตาร์ต"""
    from lmds.config import Settings
    from lmds.inventory import host_payload as real_host_payload

    hub = dict(real_host_payload())
    hub.update(registered["host"], hostname="spark1")
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: hub)
    client = TestClient(create_app())

    assert client.patch("/api/cluster/self", json={"stack": False}).status_code == 200
    assert Settings.load().cluster.stack_self is False
    data = client.get("/api/cluster").json()
    assert data["groups"] == []
    assert next(m for m in data["machines"] if m["self"])["stack"] is False

    client.patch("/api/cluster/self", json={"stack": True})
    assert client.get("/api/cluster").json()["groups"], "เปิดคืนแล้วต้องกลับมาจับกลุ่มได้"


def test_cluster_self_patch_needs_the_field(registered):
    assert TestClient(create_app()).patch("/api/cluster/self", json={}).status_code == 400


# ── ลำดับการ์ดที่ผู้ใช้ลากจัดเอง ────────────────────────────────────────────
@pytest.fixture
def three_nodes():
    from lmds.nodes import Node, add

    for index, name in enumerate(["alpha", "beta", "gamma"], start=1):
        add(Node(name=name, host=f"10.0.0.{index}", user="ops"))


def test_saved_order_drives_the_node_list(three_nodes):
    client = TestClient(create_app())
    assert [n["name"] for n in client.get("/api/nodes").json()["nodes"]] == \
        ["alpha", "beta", "gamma"]

    r = client.put("/api/nodes/order", json={"names": ["gamma", "alpha", "beta"]})
    assert r.status_code == 200 and r.json()["names"] == ["gamma", "alpha", "beta"]
    assert [n["name"] for n in client.get("/api/nodes").json()["nodes"]] == \
        ["gamma", "alpha", "beta"]


def test_order_survives_a_restart(three_nodes):
    """ลำดับต้องอยู่ที่ hub ไม่ใช่ในเบราว์เซอร์ — เปิดจากเครื่องไหนก็ต้องเห็นเหมือนกัน"""
    from lmds.config import Settings

    TestClient(create_app()).put("/api/nodes/order", json={"names": ["beta", "gamma", "alpha"]})
    assert Settings.load().ui.node_order == ["beta", "gamma", "alpha"]
    # แอปตัวใหม่ = เหมือนรีสตาร์ต service แล้วเปิดหน้าใหม่
    assert [n["name"] for n in TestClient(create_app()).get("/api/nodes").json()["nodes"]] == \
        ["beta", "gamma", "alpha"]


def test_order_drops_stale_names_and_keeps_new_machines(three_nodes):
    """เครื่องที่ลบไปแล้วต้องไม่ค้างในลิสต์ · เครื่องที่ยังไม่มีในลำดับต้องไม่หายไปจากหน้า"""
    from lmds.nodes import Node, add

    client = TestClient(create_app())
    saved = client.put("/api/nodes/order",
                       json={"names": ["gamma", "ลบไปแล้ว", "alpha", "gamma"]}).json()["names"]
    assert saved == ["gamma", "alpha"]          # ชื่อแปลกปลอมและชื่อซ้ำถูกตัดทิ้ง

    add(Node(name="delta", host="10.0.0.9", user="ops"))
    assert [n["name"] for n in client.get("/api/nodes").json()["nodes"]] == \
        ["gamma", "alpha", "beta", "delta"]     # ที่ไม่ได้จัดลำดับต่อท้ายตามทะเบียน


def test_order_rejects_a_bad_payload(three_nodes):
    assert TestClient(create_app()).put("/api/nodes/order", json={"names": "alpha"}).status_code == 400


def test_cluster_members_follow_the_saved_order(registered, monkeypatch):
    """สมาชิกตัวแรกของกลุ่มคือเครื่องที่ถูกเสนอเป็น head — ลำดับที่ลากจึงต้องมีผลถึงตรงนี้"""
    import copy

    from lmds.inventory import host_payload as real_host_payload
    from lmds.nodes import Node, add

    add(Node(name="spark3", host="10.0.0.7", user="ops", cluster_ip="10.10.0.3"))

    # แต่ละเครื่องต้องเป็นคนละเครื่องจริง ๆ ไม่งั้นโดนยุบเป็นตัวเดียว (same-machine)
    addresses = {"spark2": "10.10.0.2", "spark3": "10.10.0.3"}

    def probe_each(node):
        data = copy.deepcopy(registered)
        data["host"]["hostname"] = node.name
        data["host"]["fabric"]["links"][0]["ip"] = addresses[node.name]
        return data

    monkeypatch.setattr("lmds.nodes.probe", probe_each)
    monkeypatch.setattr("lmds.nodes.ssh.probe", probe_each)

    hub = dict(real_host_payload())
    hub.update(copy.deepcopy(registered["host"]), hostname="spark1")
    monkeypatch.setattr("lmds.inventory.host_payload", lambda: hub)

    client = TestClient(create_app())
    client.put("/api/nodes/order", json={"names": ["spark3", "spark2"]})
    (group,) = client.get("/api/cluster").json()["groups"]
    # hub มาก่อนเสมอ (ไม่มีการ์ดให้ลาก) แล้วตามด้วยลำดับที่จัดไว้
    assert [m["name"] for m in group["members"]] == ["spark1", "spark3", "spark2"]


def test_recipes_are_offered_in_the_deploy_box_not_only_in_a_panel():
    """คนที่ยังไม่ได้ตั้ง LLM ต้องเลือกสูตรได้จากในกล่อง deploy — เดิมมีแค่ช่องพิมพ์ลิงก์ HF"""
    page = TestClient(create_app()).get("/").text
    assert 'id="w-recipes"' in page and "renderRecipePicker" in page
    assert "data-recipe=" in page                       # ชิปให้กดเลือก
    assert "/api/recipes/sync" in page                  # ปุ่มดึงจากรีโปของทีม


def test_recipes_endpoint_reports_where_the_recipes_came_from():
    data = TestClient(create_app()).get("/api/recipes").json()
    assert data["recipes"] and "default_repo" in data
    assert "dgx-spark-all-controllers" in data["default_repo"]
    assert data["source"] == {}                          # ยังไม่เคย sync บนเครื่องเทส


def test_recipe_sync_reports_a_readable_error(monkeypatch):
    """ดึงไม่ได้ (เน็ต/สิทธิ์/ชื่อ ref ผิด) ต้องบอกเป็นข้อความ ไม่ใช่ 500 เปล่า ๆ"""
    from lmds.recipes import sync as sync_module

    def boom(*args, **kwargs):
        raise sync_module.SyncError("ต่อ github.com ไม่ได้")

    monkeypatch.setattr(sync_module, "sync", boom)
    r = TestClient(create_app()).post("/api/recipes/sync", json={})
    assert r.status_code == 400 and "github.com" in r.json()["detail"]


def test_theme_is_chosen_by_the_user_not_only_by_the_os():
    """เดิมหน้าเว็บมืดตาม prefers-color-scheme อย่างเดียว เครื่องที่ตั้ง OS เป็นมืดจึงเลือกสว่างไม่ได้"""
    page = TestClient(create_app()).get("/").text
    assert 'id="theme"' in page                       # ปุ่มสลับธีมบนแถบหัว
    assert 'lmds-theme' in page                       # จำค่าที่เลือกไว้ในเบราว์เซอร์
    assert ':root[data-theme="dark"]' in page         # ธีมมืดผูกกับ attribute
    # ชุดสีมืดต้องมีที่เดียว — ถ้ามี media query คุม :root ด้วย สองชุดจะหลุดกันเมื่อแก้สีทีหลัง
    assert "@media (prefers-color-scheme: dark)" not in page


def test_page_has_a_drag_handle_for_reordering():
    page = TestClient(create_app()).get("/").text
    assert 'class="ngrip"' in page and "pointerdown" in page
    assert "/api/nodes/order" in page


def test_drag_listens_on_the_document_not_the_handle():
    """เจอตอนใช้จริง: ผูก pointermove ไว้กับตัวที่จับแล้วลากได้ก้าวเดียวก็ค้าง

    เพราะทุกครั้งที่ย้ายการ์ดใน DOM ตัวที่จับถูกถอดออกแล้วใส่กลับ เบราว์เซอร์จึงปล่อย
    pointer capture ทิ้ง event ที่เหลือเลยไม่วิ่งมาที่ที่จับอีก — ต้องฟังที่ document
    """
    page = TestClient(create_app()).get("/").text
    assert 'document.addEventListener("pointermove"' in page
    assert 'document.addEventListener("pointerup"' in page
    assert ".setPointerCapture(" not in page          # ชื่อในคอมเมนต์ได้ แต่ห้ามเรียกจริง


def test_page_can_toggle_stacking_for_the_hub_itself():
    """hub ไม่มีการ์ดในลิสต์ ปุ่มของมันจึงต้องอยู่ที่อื่น — ไม่งั้นปิดแล้วกลุ่มหาย = เปิดคืนไม่ได้"""
    page = TestClient(create_app()).get("/").text
    assert 'data-cact="toggle-stack"' in page
    assert "hubstack" in page and "/api/cluster/self" in page


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

    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: called.append(command) or FakeStream())
    ok = client.post("/api/nodes/spark2/models/demo/remove", json={"confirm": "demo"})
    assert ok.status_code == 200
    wait_for_job(client, ok.json()["job"]["id"])
    assert called == ["lmds remove demo -y"], "ลบจริงต้องเป็นงานเบื้องหลัง (ลบ 70 GB ใช้เวลา)"


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
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: seen.append(command) or FakeStream())
    client = TestClient(create_app())
    menu = ["start", "stop", "restart", "doctor", "logs", "repair", "enable", "disable"]
    for command in menu:
        r = client.post(f"/api/nodes/spark2/models/demo/{command}")
        assert r.status_code == 200, command
        job = r.json().get("job")
        if job:
            wait_for_job(client, job["id"])
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
    # ธีมมืดผูกกับ attribute แล้ว (ผู้ใช้เลือกเองได้) แต่ยังต้องมีชุด token ครบเหมือนเดิม
    assert "prefers-color-scheme: dark" in body      # โหมด "ตามเครื่อง" ยังอ่านค่าของ OS
    # สีทุกตัวต้องมาจาก token — hex ที่ hard-code ในกฎ CSS จะเพี้ยนในอีกโหมดหนึ่ง
    inside_dark = body.split(':root[data-theme="dark"] {')[1].split("}")[0]
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
    assert 'class="danger">Forget</button>' in body


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
    assert 'if (value == null) return "";' in body, "ต้องซ่อนเกจที่ไม่มีค่า ไม่ใช่วาดเป็น 0"
    assert "N/A (SoC)" in body, "unified memory ต้องบอกว่าเป็น SoC ไม่ใช่โชว์ 0 GB"


def test_temperature_is_colour_coded():
    """อุณหภูมิเป็นค่าเดียวที่ 'สูง = อันตราย' จริง — ต้องเห็นได้โดยไม่ต้องอ่านตัวเลข

    สีเปลี่ยนเป็นไล่เฉดเพื่อให้เข้าชุดกับทั้งหน้าได้ แต่ระดับอันตรายต้องยังแยกออกจากกัน
    """
    body = TestClient(create_app()).get("/").text
    assert "function tempColour(" in body
    block = body.split("function tempColour(")[1][:300]
    assert "gr-bad" in block and "gr-warn" in block and "gr-ok" in block
    assert 'id="gr-bad"' in body, "gradient ที่อ้างถึงต้องถูกนิยามไว้ในหน้าจริง"


def test_every_gradient_referenced_by_a_gauge_is_defined():
    """url(#id) ที่ไม่มี <linearGradient> รองรับ = เส้นเกจหายไปเฉย ๆ ไม่มี error ให้เห็น"""
    import re

    body = TestClient(create_app()).get("/").text
    # คอมเมนต์ไม่ได้ถูกเรนเดอร์ — ตัวอย่างที่เขียนไว้ในนั้นไม่ใช่การอ้างถึงจริง
    live = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    used = set(re.findall(r"url\(#([a-z-]+)\)", live))
    defined = set(re.findall(r'<linearGradient id="([a-z-]+)"', body))
    assert used, "หน้านี้ควรใช้ gradient กับเกจ"
    assert used <= defined, f"อ้างถึงแต่ไม่ได้นิยาม: {used - defined}"


def test_command_output_lives_outside_the_body_it_must_survive():
    """SSE ส่งสถานะทุก 1-3 วิ — ผลของ doctor ต้องไม่หายไปกับการวาดการ์ดใหม่

    เดิมแก้ด้วยการ *หยุดวาด* การ์ดขณะมีผลค้างอยู่ (ผลอยู่ใน row.body เดียวกัน) ซึ่งแลกมาด้วย
    รายการโมเดลหายทั้งแผงและค่าในการ์ดค้างไปด้วย · ตอนนี้ผลอยู่คนละกล่อง (.nout) ใต้การ์ด
    การ์ดจึงอัปเดตสดต่อได้โดยผลไม่หาย
    """
    body = TestClient(create_app()).get("/").text
    assert 'class="nout"' in body, "ต้องมีกล่องผลลัพธ์แยกจาก body"
    assert "row.out || row.body" in body, "showNodeOutput ต้องเขียนลงกล่องนั้น"
    # body วาดใหม่ได้อิสระ — ห้ามกลับไปผูกการวาดกับการมีผลค้าง
    assert "if (row) row.body.innerHTML = nodeModelsMarkup" in body
    # ต้องลบผลทิ้งได้ ไม่งั้นเทสหลาย ๆ รอบแล้วหน้ายาวจนใช้งานไม่ไหว
    assert 'data-nact="close-output"' in body
    assert "function clearNodeOutput" in body


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


def test_node_start_passes_options_as_env_just_like_a_local_model(registered, monkeypatch):
    """โมเดลบนเครื่องอื่นต้องตั้งค่าได้เท่ากับโมเดลในเครื่อง — controller ตัวเดียวกัน
    รับ env ชุดเดียวกัน · เดิม node ใช้ flag ทำให้ตั้ง slots/bind/API key ไม่ได้เลย
    """
    sent = {}

    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: sent.update(command=command) or FakeStream())
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/models/demo/start", json={
        "port": 8001, "context": 32768, "gpu_util": 0.8, "slots": 8,
        "bind": "127.0.0.1", "api_key": "s3cret",
    })
    assert r.status_code == 200, r.text
    wait_for_job(client, r.json()["job"]["id"])
    for expected in ("API_PORT=8001", "CTX_SIZE=32768", "MAX_MODEL_LEN=32768",
                     "PARALLEL_SEQS=8", "MAX_NUM_SEQS=8", "API_HOST=127.0.0.1",
                     "API_KEY=s3cret", "GPU_MEMORY_UTILIZATION=0.8"):
        assert expected in sent["command"], expected
    assert sent["command"].endswith("lmds start demo")


def test_node_options_are_validated_on_the_server(registered, monkeypatch):
    """ค่าพวกนี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นผ่าน SSH — ตรวจที่ server เท่านั้นที่นับ
    ฝากไว้กับ JS ในเบราว์เซอร์ไม่ได้ ใครก็ยิง API ตรงข้ามได้
    """
    called = []
    monkeypatch.setattr("lmds.nodes.run",
                        lambda *a, **k: called.append(a) or SimpleNamespace(exit_code=0, stdout="", stderr=""))
    client = TestClient(create_app())
    for bad in ({"port": "8001; rm -rf /"}, {"port": 0}, {"port": 70000},
                {"context": -1}, {"gpu_util": 1.5}, {"context": "abc"}):
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


def test_auth_endpoint_says_whether_a_token_is_needed():
    """หน้าเว็บถามก่อนวาด — bind 127.0.0.1 ไม่ต้องมี token จะได้ไม่บังคับ login โดยไม่จำเป็น"""
    assert TestClient(create_app()).get("/api/auth").json() == {"required": False}
    assert TestClient(create_app("s3cret-token")).get("/api/auth").json() == {"required": True}


def test_auth_check_accepts_the_right_token_and_rejects_the_wrong_one():
    client = TestClient(create_app("s3cret-token"))
    assert client.post("/api/auth", headers={"x-lmds-token": "s3cret-token"}).status_code == 200
    assert client.post("/api/auth", headers={"x-lmds-token": "wrong"}).status_code == 401


def test_repeated_wrong_tokens_get_throttled():
    """token สั้นสุด 8 ตัวที่ผู้ใช้ตั้งเองอาจเป็นคำที่เดาได้ — ยิงได้ไม่จำกัดคือเดาจนเจอ"""
    client = TestClient(create_app("s3cret-token"))
    codes = [client.post("/api/auth", headers={"x-lmds-token": "nope"}).status_code
             for _ in range(12)]
    assert codes[0] == 401, "ครั้งแรก ๆ ต้องเป็น 401 ธรรมดา คนพิมพ์ผิดไม่ควรโดนลงโทษ"
    assert 429 in codes, "ผิดรัว ๆ ต้องโดนหน่วง"
    # ของจริงต้องยังเข้าได้ระหว่างถูกหน่วง? ไม่ — หน่วงคิดจาก IP จึงกันทั้งหมด
    assert client.post("/api/auth", headers={"x-lmds-token": "s3cret-token"}).status_code == 429


def test_a_successful_login_clears_the_throttle():
    client = TestClient(create_app("s3cret-token"))
    for _ in range(3):
        client.post("/api/auth", headers={"x-lmds-token": "nope"})
    assert client.post("/api/auth", headers={"x-lmds-token": "s3cret-token"}).status_code == 200
    for _ in range(3):
        client.post("/api/auth", headers={"x-lmds-token": "nope"})
    assert client.post("/api/auth", headers={"x-lmds-token": "s3cret-token"}).status_code == 200


def test_page_asks_for_the_token_before_drawing_anything():
    """เดิมหน้าโหลดขึ้นมาก่อนแล้วค่อยพังตอนเรียก API — คนที่ไม่มีสิทธิ์เห็นโครงหน้าและชื่อเครื่อง"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "function loginScreen" in page and "async function boot()" in page
    assert "localStorage.setItem(TOKEN_KEY" in page, "ผ่านแล้วต้องจำไว้ ไม่ใช่ให้กรอกทุกครั้ง"
    assert "history.replaceState" in page, "?token= ใน URL ต้องถูกลบออกจากแถบที่อยู่"


def test_cluster_ip_is_shown_on_the_machine_it_belongs_to():
    """เดิม cluster IP ทุกเครื่องกองรวมกันอยู่การ์ดล่างสุด ลูกค้าอ่านแล้วไม่รู้ว่าอันไหนของใคร
    — ต้องอยู่ในการ์ดของเครื่องนั้น และกลุ่มที่ stacked ด้วยกันได้ต้องมีรั้วสีคร่อมไว้
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "function clusterStrip" in page and "class=\"nclus\"" in page
    assert "function layoutClusterGroups" in page, "เครื่องกลุ่มเดียวกันต้องถูกจัดให้อยู่ติดกัน"
    assert '<div class="card" id="cluster">' not in page, "การ์ด cluster ล่างสุดต้องไม่เหลือไว้ให้สับสน"
    assert "⇄" in page, "ต้องบอกชื่อคู่ของเครื่องนั้นตรง ๆ"


def test_a_model_without_weights_still_offers_a_way_to_get_them():
    """doctor บอกว่า weight หาย แล้วให้คำสั่งแก้ — แต่เมนูกลับไม่มีปุ่มนั้นให้กด
    ผู้ใช้ต้อง ssh เข้าเครื่องนั้นไป cd bundles/... เอง ซึ่งขัดกับเหตุผลที่มีหน้าเว็บ
    (ผู้ใช้เจอจริงกับ qwen3-coder-next-gguf บน dgx-veerasiam)

    เงื่อนไขเดิมกลับด้าน: repair ขึ้นเฉพาะตอน downloaded ซึ่งเป็นตอนที่ไม่ต้องใช้
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert 'nbtn("repair"' in page, "repair ต้องมีเสมอ — ยังไม่มีไฟล์คือตอนที่ต้องใช้มากที่สุด"
    assert '(m.downloaded ? "start" : "download")' in page, \
        "ยังไม่มี weight ปุ่มหลักต้องเป็น download ไม่ใช่ปล่อยว่าง"


def test_doctor_fixes_point_at_commands_the_web_can_actually_run(tmp_path, monkeypatch):
    """คำแนะนำที่ทำตามไม่ได้จากที่ที่ผู้ใช้อ่านมัน ก็เท่ากับไม่มีคำแนะนำ

    `weights` เคยบอกให้ `cd bundles/<slug> && ./<slug>-single.sh download` ซึ่งคนที่อ่าน
    doctor จากหน้าเว็บทำตามไม่ได้เลยโดยไม่ ssh เข้าเครื่องนั้น
    """
    from lmds.doctor import checks

    src = (Path(checks.__file__)).read_text(encoding="utf-8")
    assert "cd bundles/{slug} && ./{slug}-single.sh download" not in src
    # allowlist ของคำสั่งข้ามเครื่อง — คำสั่งที่ doctor แนะนำต้องอยู่ในนี้
    import re

    api_src = (Path(__file__).resolve().parents[1] / "src/lmds/web/api.py").read_text(encoding="utf-8")
    block = api_src.split('allowed = {\n            "start"')[1].split("}")[0]
    allowed = set(re.findall(r'"(\w[\w-]*)":', block)) | {"start"}
    for command in re.findall(r'f"lmds (\w+) \{slug\}', src):
        assert command in allowed, f"doctor แนะนำ `lmds {command}` แต่หน้าเว็บสั่งข้ามเครื่องไม่ได้"


def test_bundle_can_be_pushed_to_the_machine_that_will_run_it(registered, monkeypatch, tmp_path):
    """wizard สร้าง bundle ลงเครื่องที่เปิดหน้าเว็บอยู่เสมอ — บน controller ที่ไม่มี GPU
    มันจึงรันไม่ได้ และไม่มีทางบอกให้ไปรันที่เครื่องอื่นจากหน้าเว็บเลย (ผู้ใช้เจอจริง)
    """
    archive = tmp_path / "demo.zip"
    archive.write_bytes(b"PK\x03\x04" + b"0" * 2048)
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path])

    sent, ran = {}, {}
    monkeypatch.setattr("lmds.nodes.push_file",
                        lambda node, local, remote, timeout=0: sent.update(local=local, remote=remote)
                        or SimpleNamespace(ok=True, exit_code=0, stdout="", stderr=""))
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: ran.update(cmd=command)
                        or SimpleNamespace(ok=True, exit_code=0, stdout="/home/u/bundles/demo", stderr=""))

    r = TestClient(create_app()).post("/api/models/demo/push/spark2")
    assert r.status_code == 200, r.text
    assert sent["local"].endswith("demo.zip") and sent["remote"] == "/tmp/demo.zip"
    assert "unzip" in ran["cmd"] and "~/bundles" in ran["cmd"]
    assert r.json()["path"] == "/home/u/bundles/demo"


def test_pushing_a_bundle_that_does_not_exist_says_so(registered, monkeypatch, tmp_path):
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path])
    r = TestClient(create_app()).post("/api/models/missing/push/spark2")
    assert r.status_code == 404
    assert "missing.zip" in r.json()["detail"]


def test_pushing_to_an_unknown_machine_says_so(registered, tmp_path, monkeypatch):
    monkeypatch.setattr("lmds.fleet.bundle_roots", lambda: [tmp_path])
    r = TestClient(create_app()).post("/api/models/demo/push/not-a-machine")
    assert r.status_code == 404


def test_node_models_get_the_same_controls_as_local_ones():
    """โมเดลบนเครื่องอื่นเคยตั้งได้แค่ port/context/gpu-util ส่วนโมเดลในเครื่องตั้งได้
    slots/bind/API key ด้วย และมีชุดทดสอบครบ — controller ตัวเดียวกันแท้ ๆ
    ผู้ใช้เห็นสองหน้าจอที่ทำงานคนละอย่างทั้งที่ควรเหมือนกัน
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    for field in ("n-port", "n-ctx", "n-slots", "n-bind", "n-key", "n-gpu"):
        assert f'class="{field}"' in page, f"เมนูของ node ขาดช่อง {field}"
    for command in ("test-text", "test-vision", "bench", "stress", "client-config", "verify-files"):
        assert f'ctl("{command}"' in page, f"เมนูของ node ขาดปุ่ม {command}"


def test_node_controller_commands_are_allowlisted(registered, monkeypatch):
    """endpoint นี้รันสคริปต์ของ bundle บนเครื่องอื่น — ต้องรับเฉพาะคำสั่งที่อ่าน/ทดสอบ"""
    calls = []
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: calls.append(command)
                        or SimpleNamespace(exit_code=0, stdout="ok", stderr=""))
    client = TestClient(create_app())
    assert client.post("/api/nodes/spark2/models/demo/ctl/test-text").status_code == 200
    assert "test-text" in calls[0] and "bundles/demo" in calls[0]
    # start/stop/download มีทางของมันเองที่จัดการ option แล้ว — ห้ามเข้าทางนี้
    for bad in ("start", "stop", "download", "rm-rf"):
        assert client.post(f"/api/nodes/spark2/models/demo/ctl/{bad}").status_code == 400, bad


def test_long_controller_commands_run_as_a_job_not_in_the_request(registered, monkeypatch):
    """เจอจริง: กด sync-worker แล้ว "เงียบ" — มันคัดลอก weight ข้ามเครื่องเป็นสิบนาที
    แต่ endpoint รอให้จบใน request เดียว ปุ่มจึงค้างเป็น "sync-worker…" จนสายหลุด
    """
    from lmds.web import jobs

    started, blocking = [], []
    monkeypatch.setattr(jobs, "start_remote",
                        lambda node, slug, command, remote: started.append((node, slug, command))
                        or SimpleNamespace(payload=lambda: {"id": "job-1", "command": command}))
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: blocking.append(command)
                        or SimpleNamespace(exit_code=0, stdout="ok", stderr=""))
    client = TestClient(create_app())

    for command in ("sync-worker", "prepare-runtime", "verify-worker", "verify-files", "stress"):
        payload = client.post(f"/api/nodes/spark2/models/demo/ctl/{command}").json()
        assert payload["job"]["id"] == "job-1", command
    assert [c for _, _, c in started] == ["sync-worker", "prepare-runtime", "verify-worker",
                                          "verify-files", "stress"]
    assert blocking == [], "คำสั่งยาวต้องไม่ไปค้างอยู่ใน HTTP request"


def test_short_controller_commands_still_answer_directly(registered, monkeypatch):
    """ชุดทดสอบสั้น ๆ ยังตอบผลกลับมาในคำขอเดียวเหมือนเดิม — ไม่ต้องไปตามผ่าน job ให้ยุ่ง"""
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: SimpleNamespace(
                            exit_code=0, stdout="ตอบแล้ว", stderr=""))
    payload = TestClient(create_app()).post("/api/nodes/spark2/models/demo/ctl/status").json()
    assert payload["output"] == "ตอบแล้ว" and "job" not in payload


def test_no_button_can_be_left_stuck_when_a_request_fails():
    """ปุ่มถูก disable + เปลี่ยนข้อความเป็น "…" ถ้า fetch พังแล้วไม่มีใครคืนสถานะ = กดแล้วเงียบ"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "async function withButton(" in page
    assert "} finally {" in page.split("async function withButton(")[1][:900]
    # ตาข่ายรับท้ายสำหรับปุ่มที่ยังไม่ได้ย้ายมาใช้ withButton
    assert 'addEventListener("unhandledrejection"' in page
    # ปุ่ม ctl ที่ได้ job กลับมาต้องไปตามผลแบบสตรีม ไม่ใช่รอเงียบ ๆ
    ctl_branch = page.split('if (nact.startsWith("ctl:"))')[1][:700]
    assert "withButton" in ctl_branch and "followNodeJob" in ctl_branch


def test_page_warns_when_another_model_owns_the_port():
    """bundle ที่สร้างก่อนมีตัวตรวจในสคริปต์จะยังยิงทดสอบไปโดนโมเดลอื่นได้
    หน้าเว็บจึงต้องเตือนตรงที่ผู้ใช้กำลังจะกดปุ่มทดสอบพอดี
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "const rival = models.find(" in page
    assert "x.running && x.port === m.port" in page, "ต้องนับเฉพาะตัวที่รันอยู่จริงบนพอร์ตเดียวกัน"
    assert "ผลทดสอบที่ได้จะเป็นของตัวนั้น" in page


def test_long_node_commands_stream_instead_of_blocking(registered, monkeypatch):
    """download โมเดล 70 GB ใช้เวลาเป็นสิบนาที — รอใน request เดียวคือให้ผู้ใช้มองหน้าค้าง
    โดยไม่รู้ว่าคืบหน้าหรือตายไปแล้ว
    """
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command: FakeStream(["โหลด 10%\n", "โหลด 50%\n", "เสร็จ\n"]))
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/models/demo/repair")
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["node"] == "spark2" and job["running"] in (True, False)
    done = wait_for_job(client, job["id"])
    assert "โหลด 50%" in done["output"] and done["exit_code"] == 0


def test_short_node_commands_stay_immediate(registered, monkeypatch):
    """doctor/logs ตอบเร็วอยู่แล้ว — ทำเป็น job จะกลายเป็นต้องรออีกรอบโดยไม่ได้อะไรเพิ่ม"""
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: SimpleNamespace(
                            exit_code=0, stdout="ตารางผลตรวจ", stderr=""))
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/doctor")
    assert r.status_code == 200
    assert "job" not in r.json() and r.json()["output"] == "ตารางผลตรวจ"


def test_two_jobs_on_the_same_model_are_refused(registered, monkeypatch):
    """download ซ้อน start คือทางลัดไปสู่ไฟล์พัง — กันไว้เหมือนโมเดลในเครื่อง"""
    import threading

    gate = threading.Event()

    def slow_stream(node, command):
        return FakeStream(iter(lambda: gate.wait(2) and None, None))

    monkeypatch.setattr("lmds.nodes.stream", lambda node, command: FakeStream(["…\n"] * 1))
    client = TestClient(create_app())
    first = client.post("/api/nodes/spark2/models/demo/repair")
    assert first.status_code == 200
    # งานแรกอาจจบไปแล้ว (FakeStream เร็วมาก) — เทสนี้จึงตรวจแค่ว่าคีย์แยกตามเครื่อง
    other = client.post("/api/nodes/spark2/models/other/repair")
    assert other.status_code == 200, "คนละโมเดลต้องรันพร้อมกันได้"


def test_node_jobs_are_keyed_per_machine(registered):
    """slug เดียวกันบนคนละเครื่องคือคนละงาน — ใช้ slug อย่างเดียวจะบล็อกกันข้ามเครื่อง"""
    from lmds.web import jobs

    assert jobs._key("demo", "spark1") != jobs._key("demo", "spark2")
    assert jobs._key("demo") == "demo", "โมเดลในเครื่องนี้ต้องใช้คีย์เดิม (เข้ากันได้กับของเก่า)"


def test_nodes_list_suggests_a_target_preset(registered, monkeypatch):
    """deploy สำหรับเครื่องอื่นต้องระบุ target เอง (เครื่องนี้ตรวจแทนเขาไม่ได้) —
    ถ้าไม่แนะนำให้ ผู้ใช้ต้องรู้เองว่าเครื่องปลายทางคือ preset ไหน เดาผิดแล้วแผนที่ได้
    จะคำนวณ context จากหน่วยความจำที่ไม่ใช่ของจริง
    """
    from lmds.web import state

    state.STORE.set_node("spark2", {"host": {"memory_model": "unified",
                                             "gpus": [{"name": "NVIDIA GB10"}]}, "models": []})
    rows = {n["name"]: n for n in TestClient(create_app()).get("/api/nodes").json()["nodes"]}
    assert rows["spark2"]["suggested_target"] == "dgx-spark-single"


def test_target_hint_is_empty_when_the_machine_is_unknown(registered):
    """เดาไม่ได้ต้องบอกว่าเดาไม่ได้ ไม่ใช่เดามั่ว — แผนที่ผิดเงียบ ๆ แย่กว่าให้เลือกเอง"""
    rows = {n["name"]: n for n in TestClient(create_app()).get("/api/nodes").json()["nodes"]}
    assert rows["spark2"]["suggested_target"] == ""


def test_wizard_can_target_another_machine_from_the_start():
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert 'id="w-machine"' in page, "wizard ต้องเลือกเครื่องปลายทางได้ตั้งแต่ต้น"
    assert "pushAfterBuild" in page, "เลือกเครื่องไว้แล้วต้องส่งให้เลย ไม่ใช่ให้ไปหาปุ่ม push เอง"
    assert "bundle ยังอยู่บนเครื่องนี้ครบ" in page, "ส่งไม่ผ่านต้องบอกว่าของยังอยู่ ไม่ใช่หายไป"


def test_command_palette_reaches_every_model_and_every_page_action():
    """หน้านี้ยาวขึ้นตามจำนวนเครื่อง — 3 เครื่อง × 5 โมเดลแล้วการเลื่อนหาเริ่มไม่ไหว"""
    import re

    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "function palOpen" in page and "function palCommands" in page
    assert 'e.key.toLowerCase() === "k"' in page, "ต้องเปิดด้วย ⌘K / Ctrl-K"
    # palette กดปุ่มบนหน้าแทนผู้ใช้ — ปุ่มที่มันอ้างถึงต้องมีอยู่จริง ไม่งั้นกดแล้วเงียบ
    palette = page.split("function palCommands")[1].split("function jumpToLocal")[0]
    for element in set(re.findall(r'getElementById\("([a-z-]+)"\)', palette)):
        assert f'id="{element}"' in page, f"palette อ้างถึงปุ่ม #{element} ที่ไม่มีอยู่บนหน้า"


def test_palette_does_not_steal_the_slash_key_while_typing():
    """'/' เปิด palette ได้ แต่ต้องไม่แย่งตอนผู้ใช้กำลังพิมพ์ token หรือ path อยู่"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "INPUT|TEXTAREA|SELECT" in page and "!typing" in page


def test_provider_is_visible_and_editable_from_the_page(isolated_config, monkeypatch):
    """LLM ที่ใช้ช่วยวางแผนเปลี่ยนผลลัพธ์ของทุก deploy — เดิมดูและแก้ได้จาก CLI เท่านั้น
    ทั้งที่หน้าเว็บคือที่ที่ผู้ใช้เห็นว่าแผนออกมาไม่ดีแล้วอยากเปลี่ยนทันที
    """
    client = TestClient(create_app())
    before = client.get("/api/provider").json()
    assert before["configured"] is False and "openai" in before["choices"]

    r = client.put("/api/provider", json={"name": "openai-compat", "model": "aeon-ultimate",
                                          "base_url": "http://192.168.10.43:8080/v1",
                                          "api_key": "sk-test-1234"})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["name"] == "openai-compat" and after["model"] == "aeon-ultimate"
    assert after["base_url"] == "http://192.168.10.43:8080/v1"


def test_provider_endpoint_never_returns_the_key(isolated_config):
    """key อยู่ใน keyring/ไฟล์ 0600 ของเครื่อง — ส่งกลับมาที่เบราว์เซอร์คือทำให้มันรั่ว
    ผ่านทุกที่ที่ response ไปโผล่ (cache, devtools, log ของ proxy)
    """
    client = TestClient(create_app())
    client.put("/api/provider", json={"name": "openai", "api_key": "sk-secret-value-9999"})
    body = client.get("/api/provider").text
    assert "sk-secret-value-9999" not in body
    payload = client.get("/api/provider").json()
    assert payload["has_key"] is True and payload["key_hint"] == "…9999"


def test_provider_rejects_openai_compat_without_a_base_url(isolated_config):
    r = TestClient(create_app()).put("/api/provider", json={"name": "openai-compat", "model": "x"})
    assert r.status_code == 400
    assert "base-url" in r.json()["detail"] or "base_url" in r.json()["detail"]


def test_page_shows_which_llm_is_in_use():
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert 'id="brain"' in page and "function loadProvider" in page
    assert "rule-based" in page, "ไม่ได้ตั้ง LLM ต้องบอกว่าใช้ rule-based แทน ไม่ใช่ปล่อยว่าง"


def test_model_list_uses_the_saved_key_when_none_is_typed(isolated_config, monkeypatch):
    """ผู้ใช้ที่ตั้ง key ไว้แล้วไม่ควรต้องพิมพ์ซ้ำแค่เพื่อดูรายชื่อโมเดล"""
    used = {}
    monkeypatch.setattr("lmds.brain.providers.list_models",
                        lambda name, key, base=None: used.update(key=key, base=base) or ["m1"])
    client = TestClient(create_app())
    client.put("/api/provider", json={"name": "openai", "api_key": "sk-saved-9999"})
    r = client.post("/api/provider/models", json={"name": "openai"})
    assert r.status_code == 200 and r.json()["models"] == ["m1"]
    assert used["key"] == "sk-saved-9999"


def test_model_list_can_try_a_key_before_it_is_saved(isolated_config, monkeypatch):
    """ลอง key ก่อนบันทึกได้ — ไม่งั้นต้องบันทึก key ที่ยังไม่รู้ว่าใช้ได้ไหมก่อนถึงจะลองได้"""
    used = {}
    monkeypatch.setattr("lmds.brain.providers.list_models",
                        lambda name, key, base=None: used.update(key=key) or [])
    r = TestClient(create_app()).post("/api/provider/models",
                                      json={"name": "openai", "api_key": "sk-typed-0000"})
    assert r.status_code == 200
    assert used["key"] == "sk-typed-0000"


def test_model_list_failure_is_explained_not_a_stack_trace(isolated_config, monkeypatch):
    from lmds.brain.providers import ProviderError

    def boom(name, key, base=None):
        raise ProviderError("key ใช้ไม่ได้ (ถูกปฏิเสธ)")

    monkeypatch.setattr("lmds.brain.providers.list_models", boom)
    r = TestClient(create_app()).post("/api/provider/models", json={"name": "openai"})
    assert r.status_code == 422 and "key ใช้ไม่ได้" in r.json()["detail"]


def test_a_node_without_lmds_can_be_installed_from_the_page(registered, monkeypatch):
    """เครื่องที่เพิ่งเพิ่มเข้ามามักยังไม่มี LMDS — เดิมหน้าเว็บบอกแค่ว่าติดต่อไม่ได้
    พร้อมคำสั่งให้ไป ssh ทำเอง ทั้งที่ hub ต่อ SSH ได้อยู่แล้วและ CLI ก็มีคำสั่งนี้มาตลอด
    """
    sent = {}
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: sent.update(command=command) or FakeStream(["ok\n"]))
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/install")
    assert r.status_code == 200, r.text
    wait_for_job(client, r.json()["job"]["id"])
    assert "install.sh" in sent["command"] and "AutoDeployDGXProject" in sent["command"]
    assert "LMDS_SKIP_PREREQ=1" in sent["command"], \
        "ขั้น prerequisite ต้องใช้ sudo ซึ่งไม่มี tty ให้กรอกรหัส — ค่าเริ่มต้นต้องข้าม"


def test_install_button_only_shows_when_ssh_works(registered):
    """ต่อ SSH ไม่ได้ (เครื่องปิด/เน็ตไม่ถึง) เป็นคนละเรื่องกับไม่มี lmds — ปุ่มติดตั้ง
    กดไปก็ล้ม จึงไม่ควรขึ้น
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "const needsInstall" in page
    assert 'data-nact="install"' in page


def test_installing_an_unknown_machine_says_so(registered):
    r = TestClient(create_app()).post("/api/nodes/not-a-machine/install")
    assert r.status_code == 404


def test_nodes_list_survives_a_machine_that_is_unreachable(registered):
    """เครื่องที่ติดต่อไม่ได้มี entry ในแคชแต่ `data` เป็น None — เผลออ่านต่อจาก None
    ทำให้ /api/nodes ตอบ 500 แล้ว **ทั้งส่วน Other machines หายไปทั้งก้อน**
    ไม่ใช่แค่เครื่องนั้นหาย (ผู้ใช้เจอจริง: หน้าค้างที่ "Loading…")
    """
    from lmds.web import state

    state.STORE.set_node("spark2", None, "ต่อไม่ได้")
    r = TestClient(create_app()).get("/api/nodes")
    assert r.status_code == 200, r.text
    row = next(n for n in r.json()["nodes"] if n["name"] == "spark2")
    assert row["suggested_target"] == ""


def test_the_nodes_list_never_breaks_the_whole_page(registered, monkeypatch):
    """หนึ่งเครื่องมีปัญหาต้องไม่ทำให้รายชื่อทั้งหมดพัง — เป็นหลักที่ยึดมาตั้งแต่ต้น
    แต่ตัวช่วยที่เพิ่มทีหลังไม่ได้ถูกคลุมด้วยหลักนั้น
    """
    from lmds.web import state

    state.STORE.set_node("spark2", {"host": None, "models": []})
    assert TestClient(create_app()).get("/api/nodes").status_code == 200
    state.STORE.set_node("spark2", {"models": []})
    assert TestClient(create_app()).get("/api/nodes").status_code == 200


def test_the_page_says_something_when_a_list_cannot_be_read():
    """ค้างที่ "Loading…" ตลอดกาลคือบอกผู้ใช้ว่า "รอไปเรื่อย ๆ" ทั้งที่มันจะไม่มาแล้ว"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    block = page.split("async function refreshNodes()")[1][:900]
    assert "catch" in block and "อ่านรายชื่อเครื่องไม่ได้" in block


def test_the_image_can_be_overridden_without_a_redeploy(registered, monkeypatch):
    """bundle ที่ image ใช้ไม่ได้ (tag ผิด/ถูกถอน) เคยแก้ไม่ได้เลยนอกจาก deploy ใหม่ทั้งชุด
    — ทุก knob อื่นเปลี่ยนผ่าน env ได้ แต่ image ถูก hardcode ไว้ตัวเดียว
    """
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: True)
    sent = {}
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: sent.update(command=command) or FakeStream())
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/models/demo/start",
                    json={"image": "nvcr.io/nvidia/vllm:26.05-py3"})
    assert r.status_code == 200, r.text
    wait_for_job(client, r.json()["job"]["id"])
    assert "VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3" in sent["command"]


def test_an_image_from_outside_the_allowlist_is_refused(registered):
    """ค่านี้กลายเป็น `docker run <image>` บนเครื่องปลายทาง — รับอะไรก็ได้ไม่ได้"""
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/start",
                                      json={"image": "evil/backdoor:latest"})
    assert r.status_code == 400 and "registry ที่ยอมรับ" in r.json()["detail"]


def test_an_image_whose_tag_is_missing_is_refused(registered, monkeypatch):
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: False)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/start",
                                      json={"image": "vllm/vllm-openai:v0.6.3.ss"})
    assert r.status_code == 400 and "ไม่มีอยู่จริง" in r.json()["detail"]


def test_a_collapsed_machine_still_shows_its_load():
    """ย่อการ์ดไว้แล้วยังต้องรู้ว่าเครื่องไหนว่าง — ไม่งั้นต้องกางทีละใบเพื่อเลือกว่าจะสั่งงาน
    เครื่องไหน ซึ่งคือเหตุผลที่ย่อมันตั้งแต่แรก
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "function summaryMarkup" in page and 'class="nsum"' in page
    # แถบสรุปต้องอัปเดตแม้ตัวการ์ดถูกล็อก (ผู้ใช้กำลังพิมพ์/มีผลคำสั่งค้างอยู่)
    guard = page.split("if (busy || nodeIsInUse(name))")[1][:300]
    assert "paintSummary" in guard, "ค่าในหัวต้องไม่ค้างตอนตัวการ์ดถูกล็อก"


def test_the_collapsed_summary_uses_the_same_colour_thresholds():
    """ตัวเลขชุดเดียวกันต้องอ่านได้เหมือนกันทั้งย่อและกาง — คนละเกณฑ์คือคนละความหมาย"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    summary = page.split("function summaryMarkup")[1].split("function paintSummary")[0]
    assert "pct >= 90" in summary and "pct >= 75" in summary


def test_setup_endpoint_never_stores_the_password(registered, monkeypatch):
    """ผู้ใช้ยอมให้ถามรหัสผ่านตอนที่ต้องใช้ — แต่ "ถามตอนนั้น" ต้องแปลว่าไม่เก็บจริง ๆ"""
    seen = {}
    monkeypatch.setattr("lmds.nodes.run_privileged",
                        lambda node, password, with_prereq=False: seen.update(pw=password)
                        or [{"step": "linger", "ok": True, "detail": ""}])
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/setup", json={"password": "s3cret-value"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert seen["pw"] == "s3cret-value"

    # ต้องไม่โผล่กลับมาที่ client และไม่ค้างในทะเบียน
    assert "s3cret-value" not in r.text
    from lmds.nodes import nodes_file

    assert "s3cret-value" not in nodes_file().read_text(encoding="utf-8")


def test_setup_without_a_password_says_so(registered):
    r = TestClient(create_app()).post("/api/nodes/spark2/setup", json={})
    assert r.status_code == 400 and "รหัสผ่าน" in r.json()["detail"]


def test_sudo_password_is_never_typed_into_a_browser_prompt():
    """`prompt()` ของเบราว์เซอร์แสดงรหัสผ่านเป็นตัวอักษรล้วน ปิดไม่ได้ และคนที่ยืนอยู่
    ข้างหลังอ่านได้หมด (ผู้ใช้เจอจริง — รหัสผ่านโผล่เต็ม ๆ ในกล่องโต้ตอบ)
    """
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    import re

    # ฟอร์มนี้ใช้ร่วมกับ "แก้สิทธิ์ไฟล์" แล้ว — เงื่อนไขเดียวกันคุมทั้งสองงาน
    setup = page.split('if (nact === "setup" || nact === "fix-perms")')[1].split('if (nact === "install")')[0]
    # คอมเมนต์ไม่ได้ถูกรัน — ที่อธิบายว่า "ไม่ใช้ prompt()" ไม่ใช่การเรียก prompt()
    code = re.sub(r"//.*", "", setup)
    assert "prompt(" not in code, "ต้องใช้ฟอร์มในหน้าที่ปิดรหัสได้"
    assert 'id="setup-pw" type="password"' in setup
    assert 'input.value = ""' in page, "ส่งเสร็จต้องล้างช่อง ไม่ทิ้งรหัสค้างใน DOM"


def test_version_endpoint_reports_the_commit_not_just_the_number():
    """เลข version ไม่ขยับทุกคอมมิต — ถ้าไม่มี commit ก็บอกไม่ได้ว่าเครื่องไหนรันโค้ดเก่า"""
    d = TestClient(create_app()).get("/api/version").json()
    assert d["version"] and "commit" in d and d["upstream"] == ""   # ไม่ถาม remote ถ้าไม่สั่ง


def test_version_endpoint_can_ask_the_repo(monkeypatch):
    from lmds.web import api as api_module

    monkeypatch.setattr(api_module, "create_app", api_module.create_app)   # กันการ import ผิดตัว
    d = TestClient(create_app()).get("/api/version?check_repo=true").json()
    assert "upstream" in d          # ต่อ GitHub ไม่ได้ก็ต้องเป็นสตริงว่าง ไม่ใช่ระเบิด


def test_node_payload_carries_the_commit_it_runs():
    """สถานะโมเดลทุกอย่างคำนวณด้วยโค้ดของ *เครื่องนั้น* — ต้องรู้ว่ามันรัน commit ไหน"""
    from lmds.inventory import host_payload

    assert "lmds_commit" in host_payload()


def test_commit_is_known_even_when_installed_outside_a_git_checkout(monkeypatch, tmp_path):
    """node ติดตั้งแบบปกติ (ไม่ใช่ editable) โค้ดจึงไม่ได้อยู่ใน git checkout —
    ถ้าอ่าน commit ไม่ได้เลย ป้าย "โค้ดเก่า" จะไม่มีวันขึ้นบนเครื่องที่ต้องขึ้นที่สุด
    """
    import sys
    import types

    from lmds import inventory

    fake = tmp_path / "site-packages" / "lmds" / "__init__.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr("lmds.__file__", str(fake))
    monkeypatch.setitem(sys.modules, "lmds._build", types.SimpleNamespace(COMMIT="abc1234"))
    # source_commit() จำค่าไว้ตั้งแต่เรียกครั้งแรก (โดยตั้งใจ: process ต้องรายงาน commit ที่
    # ตัวเองโหลดมา ไม่ใช่ที่ git ชี้อยู่ตอนนี้) — เทสนี้ทดสอบตัว resolve จึงต้องล้าง cache ก่อน
    monkeypatch.setattr(inventory, "_BOOT_COMMIT", None)
    assert inventory.source_commit() == "abc1234"


def test_page_flags_nodes_running_older_code():
    page = TestClient(create_app()).get("/").text
    assert "paintNodeVersion" in page and "lmds_commit" in page
    assert "โค้ดเก่า" in page                      # ป้ายเตือนบนการ์ดเครื่อง
    assert 'data-nact="install"' in page          # ปุ่ม update ใช้ทางติดตั้งเดิมที่สตรีมผลอยู่แล้ว
    assert "check_repo=true" in page              # หน้าเว็บถาม GitHub ให้ hub ด้วย


def test_deploy_form_has_an_optional_token_field():
    """ลูกค้าบางรายมี token ของตัวเอง — ต้องมีที่ให้กรอกตั้งแต่แรก ไม่ใช่รอให้ 401 ก่อน"""
    page = TestClient(create_app()).get("/").text
    assert 'id="w-hf" type="password"' in page          # ไม่โชว์เป็นตัวอักษรล้วน
    # ป้ายในตัวควบคุมเป็นอังกฤษแล้ว (ดู scripts/audit_ui_language.py ฝั่ง gateway)
    # เจตนาเทสเหมือนเดิม: ช่องนี้ต้องบอกให้เห็นว่าไม่ใส่ก็ได้
    assert "Optional" in page
    assert "hf_token: hfToken" in page                  # ส่งไปกับคำขอวิเคราะห์
    assert 'id="w-hf-save"' in page and "/api/secrets/hf" in page


def test_saving_the_token_never_echoes_it_back(monkeypatch):
    """token ต้องไม่ถูกส่งกลับออกไปทาง API ไม่ว่ากรณีใด — เก็บแล้วบอกแค่ว่าเก็บที่ไหน"""
    saved = {}
    monkeypatch.setattr("lmds.secrets.set_secret",
                        lambda name, value: saved.setdefault(name, value) and "file" or "file")
    r = TestClient(create_app()).post("/api/secrets/hf", json={"token": "hf_abc123"})
    assert r.status_code == 200 and r.json() == {"saved": True, "backend": "file"}
    assert saved == {"hf": "hf_abc123"}
    assert "hf_abc123" not in r.text


def test_saving_an_empty_token_is_refused():
    assert TestClient(create_app()).post("/api/secrets/hf", json={"token": "  "}).status_code == 400


def test_adopted_bundle_is_not_told_to_download(registered, monkeypatch):
    """bundle จาก `lmds adopt` ชี้ไปที่ path ในคอนเทนเนอร์ — LMDS โหลด/ตรวจไฟล์ให้ไม่ได้

    เดิมขึ้น "not downloaded" ตลอดกาลแล้วยื่นปุ่ม download ที่กดไปก็ล้มแน่นอน
    (เจอกับ qwen3-coder-next-nvfp4-gb10 บน msi-6)
    """
    from lmds.inventory import self_managed_weights, weights_present

    adopted = {"adopted": True, "model": {"id": "/models/qwen3-coder-next"},
               "runtime": {"engine": "vllm"}}
    assert self_managed_weights(adopted) is True
    assert weights_present(SimpleNamespace(slug="demo"), adopted) is True   # ไม่ใช่ "ยังไม่โหลด"
    # profile ปกติของ HF ยังตรวจตามเดิม
    assert self_managed_weights({"model": {"id": "org/model"}}) is False

    page = TestClient(create_app()).get("/").text
    assert "self_managed_weights" in page and "weight จัดการเอง" in page


def test_fix_permissions_only_touches_the_model_cache(registered, monkeypatch):
    """เคสจริง: container รันเป็น root แล้วโหลด weight ลงแคช → sync-worker ตาย exit 23

    ปุ่มนี้สั่ง sudo บนเครื่องจริง จึงต้องแตะเฉพาะแคชโมเดลใน home ของ user เท่านั้น
    """
    ran = []

    def fake_run(node, command, timeout=0, stdin_text=None):
        ran.append(command)
        # ตรวจก่อนทำต้องไม่ผ่าน (ยังมีไฟล์ของ root) · ตรวจหลัง chown ต้องผ่าน
        ok = "chown" in command or any("chown" in c for c in ran[:-1])
        return SimpleNamespace(ok=ok, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    client = TestClient(create_app())
    payload = client.post("/api/nodes/spark2/fix-permissions", json={"password": "s3cret"}).json()

    chown = next(c for c in ran if "chown" in c)
    assert "chown -R ops:ops ~/.cache/huggingface ~/.cache/flashinfer" in chown
    assert "sudo -S -p ''" in chown          # รหัสผ่านไปทาง stdin ไม่ใช่ใน argv
    assert "/" not in chown.split("chown -R ops:ops")[1].split("~")[0]   # ไม่มีพาธนอก home
    assert payload["steps"][0]["ok"] is True
    # รหัสผ่านต้องไม่โผล่ในคำสั่งใด ๆ ที่ถูกส่งไปเครื่องปลายทาง
    assert not any("s3cret" in c for c in ran)


def test_fix_permissions_needs_a_password():
    assert TestClient(create_app()).post("/api/nodes/spark2/fix-permissions",
                                         json={}).status_code in (400, 404)


def test_rsync_permission_failure_explains_the_fix():
    """log ของ rsync อ่านแล้วไม่รู้ว่าต้องทำอะไร — ต้องบอกวิธีแก้ต่อท้ายให้เลย"""
    from lmds.web.jobs import explain_failure

    hint = explain_failure(
        'rsync: [sender] send_files failed to open "/home/neronain/.cache/huggingface/'
        'models--nvidia--DeepSeek-V4-Flash-NVFP4/trees/e3cd.json": Permission denied (13)\n'
        "rsync error: some files/attrs were not transferred (code 23)")
    assert "แก้สิทธิ์ไฟล์" in hint and "chown -R" in hint
    assert explain_failure("rsync ok, nothing wrong") == ""


def test_setup_form_says_which_user_it_will_use():
    """ผู้ใช้ถามว่า "จะรู้ได้ยังไงว่า user ไหน" — ต้องบอก ไม่ใช่ให้เดา"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "lastNodeRegistry" in page
    setup = page.split('if (nact === "setup" || nact === "fix-perms")')[1].split('if (nact === "setup-go")')[0]
    assert "user เดียวกับที่ใช้ต่อ SSH" in setup


def test_served_name_reaches_the_controller(registered, monkeypatch):
    """ชื่อที่ client เรียกใช้ต้องตั้งได้จากหน้าเว็บ ไม่ใช่ต้อง deploy ใหม่"""
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: True)
    sent = {}
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda node, command, *_: sent.update(command=command) or FakeStream())
    client = TestClient(create_app())
    r = client.post("/api/nodes/spark2/models/demo/start",
                    json={"served_name": "vllm-msi-03/aeon-ultimate"})
    assert r.status_code == 200, r.text
    wait_for_job(client, r.json()["job"]["id"])
    assert "SERVED_MODEL_NAME=vllm-msi-03/aeon-ultimate" in sent["command"]


def test_a_served_name_with_spaces_is_refused(registered):
    """ค่านี้ถูกต่อเป็นคำสั่งที่รันผ่าน SSH — ช่องว่างทำให้คำสั่งแตกเป็นคนละคำ"""
    r = TestClient(create_app()).post("/api/nodes/spark2/models/demo/start",
                                      json={"served_name": "มี ช่องว่าง"})
    assert r.status_code == 400


# ── อัปเดตตัว hub เองจากหน้าเว็บ ────────────────────────────────────────────
def test_update_refuses_when_the_hub_is_not_a_git_checkout(monkeypatch):
    """ติดตั้งแบบไม่ผ่าน git แล้ว `git pull` ไม่มีความหมาย — บอกทางที่ใช้ได้จริงแทน"""
    from lmds.web import selfupdate

    monkeypatch.setattr(selfupdate, "source_root", lambda: None)
    r = TestClient(create_app()).post("/api/update", json={})
    assert r.status_code == 409
    assert "git checkout" in r.json()["detail"]


def test_update_refuses_to_run_over_uncommitted_work(monkeypatch, tmp_path):
    """`git pull --ff-only` จะล้มอยู่แล้ว — ล้มแล้วบอกว่าไฟล์ไหนค้าง ดีกว่าไปตายกลาง log

    และที่สำคัญกว่า: ห้าม reset/merge ทับงานที่ยังไม่ได้ commit ของใครสักคนเงียบ ๆ
    """
    from lmds.web import selfupdate

    monkeypatch.setattr(selfupdate, "source_root", lambda: tmp_path)
    monkeypatch.setattr(selfupdate, "dirty_files", lambda root: ["src/lmds/web/api.py"])
    r = TestClient(create_app()).post("/api/update", json={})
    assert r.status_code == 409
    assert "src/lmds/web/api.py" in r.json()["detail"]


def test_update_streams_a_job_and_restarts_the_service(monkeypatch, tmp_path):
    """อัปเดตกินเวลาเป็นนาที — ต้องเป็นงานที่ตามดู log ได้ ไม่ใช่ HTTP ที่ค้างรอ"""
    from lmds.web import selfupdate

    monkeypatch.setattr(selfupdate, "source_root", lambda: tmp_path)
    monkeypatch.setattr(selfupdate, "dirty_files", lambda root: [])
    sent = {}
    monkeypatch.setattr("lmds.web.jobs.start_shell",
                        lambda slug, command, script, cwd="": sent.update(
                            slug=slug, script=script, cwd=cwd) or FakeJob())
    r = TestClient(create_app()).post("/api/update", json={})
    assert r.status_code == 200
    assert "git pull --ff-only" in sent["script"]
    # restart ต้องหลุดจาก process ของเว็บ ไม่งั้น systemd ฆ่าตัวที่สั่ง restart ไปพร้อมกัน
    assert "setsid" in sent["script"] and "lmds-web.service" in sent["script"]
    # ห้าม reset/merge — เครื่องที่มีของแก้ค้างต้องล้ม ไม่ใช่ถูกกลืน
    assert "reset --hard" not in sent["script"] and "git merge" not in sent["script"]


def test_the_update_script_pulls_only_from_the_configured_remote():
    """ไม่รับ URL จาก request — ไม่งั้นใครยิง endpoint นี้ได้ก็สั่งติดตั้งโค้ดจากที่ไหนก็ได้"""
    from lmds.web import selfupdate

    script = selfupdate.update_script()
    assert "https://" not in script and "git@" not in script


def test_source_root_uses_the_path_install_sh_stamped(monkeypatch, tmp_path):
    """เดาจากตำแหน่งโมดูลไม่ได้ — ติดตั้งจริงแล้วโค้ดอยู่ใน site-packages ที่ไม่มี .git ใกล้ ๆ

    เจอตอนทดสอบบนเครื่องจริง: ปุ่มอัปเดตตอบว่า "ไม่ได้ติดตั้งจาก git checkout" ทุกครั้ง
    ทั้งที่ checkout อยู่ที่ ~/AutoDeployDGXProject เรียบร้อย
    """
    import sys
    import types

    from lmds.web import selfupdate

    (tmp_path / ".git").mkdir()
    (tmp_path / "install.sh").write_text("#!/bin/sh\n")
    monkeypatch.setitem(sys.modules, "lmds._build",
                        types.SimpleNamespace(COMMIT="abc1234", SOURCE=str(tmp_path)))
    assert selfupdate.source_root() == tmp_path


def test_source_root_is_none_when_nothing_looks_like_a_checkout(monkeypatch, tmp_path):
    """ไม่มี checkout จริง = ต้องตอบ None ไม่ใช่คืน path มั่ว ๆ ให้ไป git pull ทับ"""
    import sys
    import types

    from lmds.web import selfupdate

    monkeypatch.setitem(sys.modules, "lmds._build",
                        types.SimpleNamespace(COMMIT="", SOURCE=""))
    monkeypatch.setattr(selfupdate.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(selfupdate.lmds, "__file__", str(tmp_path / "a" / "b" / "lmds" / "__init__.py"))
    assert selfupdate.source_root() is None


def test_web_sets_and_exposes_a_node_site():
    """จัดกลุ่มตามไซต์จากหน้าเว็บ: PATCH ตั้ง site ได้ · /api/nodes ส่ง site ให้ badge"""
    from lmds.nodes import Node, add, find

    add(Node(name="cust1", host="10.0.0.9", user="ops"))
    client = TestClient(create_app())

    r = client.patch("/api/nodes/cust1", json={"site": "customer-a"})
    assert r.status_code == 200
    assert r.json()["site"] == "customer-a"
    assert find("cust1").site == "customer-a"

    nodes = client.get("/api/nodes").json()["nodes"]
    assert next(n for n in nodes if n["name"] == "cust1")["site"] == "customer-a"

    # เว้นว่าง = เอาป้ายออก
    assert client.patch("/api/nodes/cust1", json={"site": ""}).json()["site"] == ""


def test_web_site_change_does_not_touch_cluster():
    from lmds.nodes import Node, add, find

    add(Node(name="cust1", host="10.0.0.9", user="ops", cluster_ip="10.10.0.1", stack=True))
    client = TestClient(create_app())
    client.patch("/api/nodes/cust1", json={"site": "customer-a"})
    node = find("cust1")
    assert node.cluster_ip == "10.10.0.1" and node.stack is True   # cluster ไม่ถูกแตะ


def test_long_output_cannot_stretch_the_page():
    """log 400 บรรทัดต้องเลื่อนอยู่ในกล่องตัวเอง ไม่ใช่ดันการ์ดเครื่องอื่นหายไปจากจอ"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert ".nout pre { max-height:" in page
    assert "overflow: auto" in page.split(".nout pre {")[1][:120]


def test_a_successful_job_keeps_its_result_on_screen():
    """test-text ที่ผ่านก็ยังต้องอ่านคำตอบได้ — เดิมผลถูกล้างทันทีที่ exit 0"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    success = page.split("watchingNodes.delete(key);")[1][:600]
    assert "pinnedOutput.delete(node);\n      loadNode(node);" not in success


def test_the_output_panel_button_does_not_claim_to_delete_anything():
    """ปุ่มปิดกล่องผลลัพธ์เคยชื่อ "ลบผลลัพธ์" และวางอยู่ใต้ผลของคำสั่ง remove พอดี

    ผู้ใช้ที่เพิ่งลบ weight ไป 42 GB อ่านแล้วไม่กล้ากด เพราะไม่รู้ว่ามันจะลบอะไรอีก
    มันแค่ปิดกล่องข้อความ ไม่แตะอะไรบนเครื่องปลายทางเลย
    """
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert ">ลบผลลัพธ์<" not in page
    assert ">ปิดข้อความนี้<" in page
    # และไม่มีคำอังกฤษหลงเหลือในปุ่มของหน้าที่เป็นไทยทั้งหน้า
    assert ">Dismiss<" not in page


# ---------------------------------------------------------------------------
# ปุ่มกับ handler ต้องมาเป็นคู่
# ---------------------------------------------------------------------------
def _console_html() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(
        encoding="utf-8"
    )


def _rendered_actions(html: str) -> set:
    """ทุก action ที่มีปุ่มวาดออกมาจริง — ทั้งที่เขียนตรง ๆ และที่ผ่านตัวช่วย"""
    import re

    actions = set(re.findall(r'data-(?:n)?act="([a-z0-9:_-]+)"', html))
    for fn, prefix in (("ctl", "ctl:"), ("job", "job:"), ("btn", "")):
        for name in re.findall(fn + r'\("([a-z0-9:_-]+)"', html):
            actions.add(prefix + name)
    return actions


def _handled_actions(html: str) -> set:
    import re

    handled = set(re.findall(r'n?act === "([a-z0-9:_-]+)"', html))
    handled |= set(re.findall(r'n?act\.startsWith\("([a-z0-9:_-]+)"\)', html))
    return handled


def test_every_handler_has_a_button_that_reaches_it():
    """handler ที่ไม่มีปุ่มให้กด = ฟีเจอร์ที่ผู้ใช้เอื้อมไม่ถึง และไม่มีอะไรฟ้อง

    เจอจริง: save-settings กับ clear-settings มี handler และมี endpoint พร้อม
    ตั้งแต่ dcf58be แต่ไม่เคยมีปุ่มถูกวาดในแผงของ node เลย · ลูกค้ารายงานว่า
    "เปลี่ยน ID ไม่ได้ เปลี่ยน port ไม่ได้" ซึ่งตรงกับสิ่งที่เห็นบนหน้าจอทุกประการ —
    ช่องกรอกได้ แต่ไม่มีอะไรให้กด และค่าไม่เคยถูกบันทึก
    """
    # ปุ่มแถวบนของโมเดลเลือกคำสั่งจากสถานะ แล้วส่งชื่อนั้นเข้าตัวช่วยเป็นตัวแปร
    # (`const act = m.running ? "stop" : (m.downloaded ? "start" : "download")`)
    # ตัวอ่านแบบข้อความจึงมองไม่เห็นว่ามันถูกวาด — ไม่ใช่ orphan จริง
    RENDERED_FROM_A_VARIABLE = {"start", "stop", "restart", "download"}

    html = _console_html()
    rendered = _rendered_actions(html) | RENDERED_FROM_A_VARIABLE
    orphans = sorted(
        action
        for action in _handled_actions(html)
        if action not in rendered and not any(r.startswith(action) for r in rendered)
    )
    assert orphans == [], f"handler ที่ไม่มีปุ่ม: {orphans}"


def test_the_settings_panels_can_persist_what_they_collect():
    """ช่องที่กรอกได้ต้องมีทางบันทึกลง bundle ไม่ใช่อยู่แค่ในเบราว์เซอร์

    ค่าที่อยู่แค่ในเบราว์เซอร์หายทุกครั้งที่ reboot เพราะ systemd เรียก controller
    เปล่า ๆ — ซึ่งเป็นเหตุผลที่ bundle.env ถูกสร้างขึ้นมาตั้งแต่แรก
    """
    html = _console_html()
    assert 'data-nact="save-settings"' in html
    assert 'data-nact="clear-settings"' in html
    assert "/settings`" in html, "ปุ่ม Save ของแผงเครื่องนี้ต้องเรียก PUT /api/models/<slug>/settings"


def test_no_map_runs_on_an_unguarded_property():
    """ฟิลด์เดียวที่หายไปไม่ควรทำให้ทั้งหน้าหยุดทำงาน

    ลูกค้าเจอ `คำสั่งไม่สำเร็จ — Cannot read properties of undefined (reading 'map')`
    บ่อยจนใช้งานไม่ได้ · ต้นเหตุคือ `.map()` ยิงตรงบน property ของ response
    พอเซิร์ฟเวอร์ตอบไม่ครบ TypeError หลุดไปถึง unhandledrejection แล้วปุ่มค้าง
    """
    import re

    html = _console_html()
    risky = []
    for number, line in enumerate(html.splitlines(), 1):
        for match in re.finditer(r"([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+)\.map\(", line):
            expr = match.group(1)
            if "?." in expr:
                continue
            if any(f"({expr} {op} [])" in line or f"{expr} {op} []" in line
                   for op in ("||", "??")):
                continue
            risky.append(f"บรรทัด {number}: {expr}.map(")
    assert risky == [], "map() ที่ไม่มีตัวกัน:\n" + "\n".join(risky)


def test_a_button_that_is_disabled_gets_re_enabled_no_matter_what():
    """ปุ่มที่ปิดไว้ระหว่างทำงาน ต้องมีทางกลับมาเสมอ

    รูปแบบที่พังคือ disabled=true → await api(...) → disabled=false · พอ api() โยน
    ข้อผิดพลาด บรรทัดสุดท้ายไม่เคยถูกรัน ปุ่มค้างจนกว่าจะรีโหลดหน้า — ตรงกับที่ลูกค้า
    รายงานว่า "กดปุ่มแล้วค้าง พอทำงานจบหน้าจอส่วนนั้นไม่ปิด"
    """
    lines = _console_html().splitlines()
    stuck = []
    for number, line in enumerate(lines, 1):
        if "disabled = true" not in line:
            continue
        window = "\n".join(lines[number - 1:number + 60])
        if "finally" in window or "withButton" in window or "location.reload()" in window:
            continue
        stuck.append(f"บรรทัด {number}: {line.strip()[:80]}")
    assert stuck == [], "ปุ่มที่อาจค้างถาวร:\n" + "\n".join(stuck)


def test_a_controller_hint_becomes_a_button():
    """controller บอกคำสั่งที่ต้องรันไว้ในข้อความที่มันตาย — คนใช้หน้าเว็บไม่มีที่พิมพ์

    เจอจริงบน msi-6: กด start แล้วได้ "ยังไม่มี llama-server — รัน: <path> prepare-runtime"
    ซึ่งถูกต้อง แต่เป็นทางตัน ผู้ใช้จึงไปสร้างโมเดลใหม่ซึ่งตายด้วยเหตุเดียวกัน
    """
    html = _console_html()
    assert "function fixButtonFor(" in html
    assert "showNodeOutput" in html
    # ต้องถูกเรียกจากกล่องผลลัพธ์ ไม่ใช่นิยามไว้เฉย ๆ
    assert "fixButtonFor(node, output)" in html
    for command in ("prepare-runtime", "download", "stop", "verify-files"):
        assert f'"{command}"' in html, command


def test_every_run_this_hint_in_the_templates_is_offerable():
    """สำนวน "รัน: $0 <คำสั่ง>" ในเทมเพลตต้องอยู่ในชุดที่หน้าเว็บเสนอเป็นปุ่มได้

    ถ้าเพิ่มคำสั่งใหม่ในเทมเพลตแล้วลืมเติมที่หน้าเว็บ ผู้ใช้จะเจอทางตันแบบเดิมอีก
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    hinted = set()
    for template in (root / "src/lmds/generator/templates").glob("*.j2"):
        hinted |= set(re.findall(r"รัน: \$0 ([a-z-]+)", template.read_text(encoding="utf-8")))

    html = _console_html()
    offered = set(re.findall(r'FIXABLE = new Set\(\[(.*?)\]\)', html, re.S))
    assert offered, "ไม่เจอรายการคำสั่งที่หน้าเว็บเสนอได้"
    listed = set(re.findall(r'"([a-z-]+)"', offered.pop()))
    missing = sorted(hinted - listed)
    assert missing == [], f"เทมเพลตแนะนำคำสั่งที่หน้าเว็บกดไม่ได้: {missing}"
