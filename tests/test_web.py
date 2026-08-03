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
