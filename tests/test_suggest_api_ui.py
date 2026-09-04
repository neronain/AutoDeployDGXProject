"""ปุ่ม "เติมให้ตามโมเดล" — จากหน้าเว็บถึง bundle.env โดยไม่ต้องรู้ชื่อ parser

ผู้ใช้ 2026-09-04: "ทำให้ระบบกรอกให้เองตาม model ได้ไหม … กลัวใส่ผิด แล้วไม่มีให้ใช้งาน"
"""

from pathlib import Path

from fastapi.testclient import TestClient

from lmds.web.api import create_app

INDEX = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


def test_node_suggest_reads_the_cached_inventory_not_ssh():
    from lmds.web import state

    state.STORE.set_node("dgx-veerasiam", {
        "host": {"memory_model": "unified", "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128.0}]},
        "models": [{"slug": "qwen3-6-35b-a3b-nvfp4", "model_id": "unsloth/Qwen3.6-35B-A3B-NVFP4",
                    "engine": "vllm", "running": False}],
    })
    r = TestClient(create_app()).get("/api/nodes/dgx-veerasiam/models/qwen3-6-35b-a3b-nvfp4/settings/suggest")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["values"]["tool_parser"] == "qwen3_xml"
    assert d["values"]["reasoning_parser"] == "qwen3"
    assert "engine_env" not in d["values"] and "image" not in d["values"]   # แนะนำเป็น note เท่านั้น
    assert any("marlin" in n for n in d["notes"])
    assert set(d["sources"]) == set(d["values"])


def test_node_suggest_for_a_model_not_in_cache_is_404_with_advice():
    r = TestClient(create_app()).get("/api/nodes/spark-09/models/nothing/settings/suggest")
    assert r.status_code == 404
    assert "refresh" in r.json()["detail"]


def test_local_suggest_uses_the_bundle_profile(monkeypatch):
    class Srv:
        controller = "/tmp/x/x-single.sh"
        model_id = "unsloth/Qwen3.6-35B-A3B-NVFP4"
        engine = "vllm"

    monkeypatch.setattr("lmds.fleet.find", lambda slug: Srv())
    monkeypatch.setattr("lmds.fleet.bundle_profile", lambda controller: {
        "model": {"id": "unsloth/Qwen3.6-35B-A3B-NVFP4", "architecture": "Qwen3_5MoeForConditionalGeneration"},
        "runtime": {"engine": "vllm"}, "target": {"memory_model": "unified"},
    })
    r = TestClient(create_app()).get("/api/models/qwen3-6-35b-a3b-nvfp4/settings/suggest")
    assert r.status_code == 200, r.text
    assert r.json()["values"]["tool_parser"] == "qwen3_xml"


def test_node_set_forwards_image_min_tokens(monkeypatch):
    """ช่อง image min tokens ในฟอร์ม node ต้องไปถึง `lmds set` บนเครื่องนั้น ไม่ใช่หายกลางทาง"""
    from lmds.nodes.ssh import Result

    seen = {}

    class N:
        name = "n1"

    monkeypatch.setattr("lmds.nodes.find", lambda name: N())

    def fake_run(node, cmd, timeout=0):
        seen["cmd"] = cmd
        return Result(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    # ไม่มีแคชของเครื่องนี้ → _require_controller ปล่อยผ่าน (ให้ controller เป็นคนบอกเอง)
    r = TestClient(create_app()).post("/api/nodes/n1/models/gemma/set", json={"image_min_tokens": "auto"})
    assert r.status_code == 200, r.text
    assert "--image-min-tokens auto" in seen["cmd"]


def test_page_has_the_button_the_env_field_and_the_handler():
    page = INDEX.read_text(encoding="utf-8")
    assert 'data-nact="suggest-settings"' in page, "ยังไม่มีปุ่มเติมให้ตามโมเดล"
    assert 'class="n-engine-env"' in page and 'engine_env: num(".n-engine-env")' in page, "engine env ยังตั้งจากหน้าเว็บไม่ได้"
    handler = page[page.index('nact === "suggest-settings"'):][:1500]
    assert "/settings/suggest" in handler and ".n-tool-parser" in handler and "press Save" in handler   # ป้ายอังกฤษตั้งแต่ 0.6.0
