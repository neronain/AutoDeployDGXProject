"""บรรทัด "ต้องใช้แรมเท่าไร" ใต้ฟอร์ม settings — รู้ก่อนกด start ไม่ใช่รู้ตอนพัง

ผู้ใช้ 2026-09-04: "หลังกรอกแล้ว จะมีให้ดูว่าต้องการแรมเท่าไร เพราะถ้าใส่ gpu-util น้อยกว่าที่
ต้องการ ระบบจะทำงานไม่ดีไหม" — vLLM จอง gpu-util × ทั้งเครื่อง เอา weights ออก ที่เหลือคือ KV
เหลือไม่พอ 1 คำขอ = ไม่ start · จองมากกว่าที่ว่าง = ไม่ start
"""

from pathlib import Path

from fastapi.testclient import TestClient

from lmds.web.api import create_app
from lmds.web.memory import memory_facts

INDEX = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"
GIB = 1024**3

PROFILE = {
    "model": {"id": "unsloth/Qwen3.6-35B-A3B-NVFP4", "weight_bytes": 26489446568,
              "native_context": 262144, "kv_bytes_per_token": 20480},
    "runtime": {"engine": "vllm"},
    "serving": {"context": 262144, "gpu_memory_utilization": 0.85, "max_num_seqs": 4},
}
HOST = {"memory_model": "unified", "ram_total_gb": 122.0, "ram_used_gb": 95.0,
        "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": 85.8}]}


def test_facts_come_from_profile_and_live_host():
    f = memory_facts(PROFILE, HOST)
    assert f["engine"] == "vllm" and f["capacity_gb"] == 128.0
    assert f["weights_gb"] == 24.7 and f["kv_bytes_per_token"] == 20480 and f["kv_source"] == "profile"
    # ว่างตอนนี้บน unified = แรมว่างของทั้งระบบ (122-95) ไม่ใช่ 128-85.8 เพราะ OS กินจาก pool เดียวกัน
    assert f["free_gb_now"] == 27.0
    assert f["overhead_gb"] == 2.5
    assert f["defaults"] == {"gpu_util": 0.85, "max_num_seqs": 4, "context": 262144}


def test_old_bundle_without_kv_asks_the_hub_once(monkeypatch):
    calls = []

    def fake(model_id, revision=None):
        calls.append(model_id)
        return 20480, "hub"

    monkeypatch.setattr("lmds.web.memory.kv_bytes_from_hub", fake)
    prof = {**PROFILE, "model": {k: v for k, v in PROFILE["model"].items() if k != "kv_bytes_per_token"}}
    f = memory_facts(prof, HOST)
    assert f["kv_bytes_per_token"] == 20480 and f["kv_source"] == "hub"
    assert calls == ["unsloth/Qwen3.6-35B-A3B-NVFP4"]


def test_when_kv_is_unknown_the_line_still_gets_weights_and_says_why(monkeypatch):
    monkeypatch.setattr("lmds.web.memory.kv_bytes_from_hub", lambda m, r=None: (None, "ถาม Hugging Face ไม่ได้: offline"))
    prof = {**PROFILE, "model": {k: v for k, v in PROFILE["model"].items() if k != "kv_bytes_per_token"}}
    f = memory_facts(prof, HOST)
    assert f["kv_bytes_per_token"] is None and f["weights_gb"] == 24.7
    assert any("offline" in n for n in f["notes"])


def test_llamacpp_uses_its_own_overhead():
    prof = {**PROFILE, "runtime": {"engine": "llamacpp"}}
    assert memory_facts(prof, HOST)["overhead_gb"] == 1.5


def test_node_endpoint_reads_profile_over_ssh_and_host_from_cache(monkeypatch):
    import yaml

    from lmds.nodes.ssh import Result
    from lmds.web import state

    state.STORE.set_node("dgx-veerasiam", {"host": HOST, "models": [
        {"slug": "qwen3-6-35b-a3b-nvfp4", "model_id": PROFILE["model"]["id"], "engine": "vllm"}]})

    class N:
        name = "dgx-veerasiam"

    monkeypatch.setattr("lmds.nodes.find", lambda name: N())
    seen = {}

    def fake_run(node, cmd, timeout=0):
        seen["cmd"] = cmd
        return Result(exit_code=0, stdout=yaml.safe_dump(PROFILE), stderr="")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    r = TestClient(create_app()).get("/api/nodes/dgx-veerasiam/models/qwen3-6-35b-a3b-nvfp4/memory")
    assert r.status_code == 200, r.text
    d = r.json()
    assert "MODEL_PROFILE.yaml" in seen["cmd"] and "qwen3-6-35b-a3b-nvfp4" in seen["cmd"]
    assert d["weights_gb"] == 24.7 and d["free_gb_now"] == 27.0 and d["kv_bytes_per_token"] == 20480


def test_node_endpoint_explains_when_the_bundle_is_missing(monkeypatch):
    from lmds.nodes.ssh import Result

    class N:
        name = "n"

    monkeypatch.setattr("lmds.nodes.find", lambda name: N())
    monkeypatch.setattr("lmds.nodes.run", lambda node, cmd, timeout=0: Result(exit_code=1, stdout="", stderr="ไม่พบ bundle x"))
    r = TestClient(create_app()).get("/api/nodes/n/models/x/memory")
    assert r.status_code == 409 and "ไม่พบ bundle" in r.json()["detail"]


def test_new_bundles_write_kv_bytes_into_the_profile(tmp_path):
    """bundle ใหม่ต้องพก kv_bytes_per_token — จะได้ไม่ต้องถาม Hugging Face ทุกครั้งที่เปิดแผง"""
    import yaml

    from tests.test_generator import make_bundle, safetensors_report

    bundle, _, fit = make_bundle(safetensors_report(), tmp_path=tmp_path)
    prof = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert prof["model"]["kv_bytes_per_token"] == fit.kv_bytes_per_token > 0


def test_page_has_the_live_line_and_recomputes_on_input():
    page = INDEX.read_text(encoding="utf-8")
    assert 'class="n-mem dim"' in page, "ยังไม่มีบรรทัดคำนวณใต้ฟอร์ม"
    assert "function paintMemoryLine(" in page and "/memory" in page
    hydrate = page[page.index("async function hydrateMemoryLine("):][:1500]
    assert '".n-ctx, .n-slots, .n-gpu"' in hydrate and 'addEventListener("input"' in hydrate
    # ข้อความสำคัญสองแบบที่ผู้ใช้ถามถึงต้องมี: จองเกินที่ว่าง และ KV ไม่พอ 1 คำขอ
    assert "start ตอนนี้ไม่ได้" in page and "vLLM จะไม่ start" in page
    # แถว port/context/slots/gpu-util ต้องไม่ห่อบรรทัดจน gpu-util ตกไปอยู่แถวล่าง
    row = page[page.index('class="n-port"') - 200:page.index('class="n-port"')]
    assert "flex-wrap:nowrap" in row
