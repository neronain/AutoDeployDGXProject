"""ภาพรวมเครื่องที่ hub ดึงผ่าน `lmds agent info`"""

import http.server
import json
import threading

from lmds import inventory
def test_cache_health_flags_root_owned_entries(tmp_path, monkeypatch):
    """แคชที่กลายเป็นของ root ทำให้ download/remove/sync ล้มโดยไม่มีสาเหตุที่มองเห็น
    — เดิมไม่มีอะไรตรวจเลย ผู้ใช้เห็นแค่คำสั่งที่ล้ม (เจอจริงบน msi-5: hub/ 73 GB เป็นของ root)
    """
    hub = tmp_path / "hub"
    (hub / "models--org--name").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert inventory.cache_health()["owner_ok"] is True

    monkeypatch.setattr(inventory.os, "getuid", lambda: -1)   # ทุกอย่างกลายเป็นของคนอื่น
    health = inventory.cache_health()
    assert health["owner_ok"] is False and health["foreign_entries"] > 0


def test_cache_health_is_quiet_on_a_fresh_machine(tmp_path, monkeypatch):
    """เครื่องที่ยังไม่มีแคช ไม่ใช่เครื่องที่มีปัญหา — ต้องไม่ขึ้นเตือน"""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert inventory.cache_health()["owner_ok"] is None


def _server(controller):
    from lmds.fleet.manager import ServerInfo
    import inspect
    kwargs = {}
    for name, param in inspect.signature(ServerInfo).parameters.items():
        if param.default is inspect.Parameter.empty:
            kwargs[name] = ""
    kwargs.update(slug="demo", controller=str(controller))
    return ServerInfo(**kwargs)


def test_a_controller_without_download_is_treated_as_self_managed(tmp_path):
    """สคริปต์เองคือความจริงสุดท้าย — ไม่มี `download` แปลว่า LMDS โหลด weight ให้ไม่ได้

    bundle ที่ adopt มาบางตัวมี model id เป็นรูป org/name ตามปกติ จึงหลุดตัวกรองที่เดาจาก
    profile แล้วหน้าเว็บยื่นปุ่ม download/repair ที่กดไปเจอ usage ของ bash (ผู้ใช้รายงานว่า
    "กด repair แล้วไม่ทำงาน" หลังแอด node ที่มี vllm อยู่ก่อน)
    """
    controller = tmp_path / "ctl.sh"
    controller.write_text("case $1 in\n  start)  start ;;\n  logs)   logs ;;\nesac\n")
    payload = inventory.model_payload(_server(controller))
    assert payload["self_managed_weights"] is True
    assert payload["downloaded"] is True     # ปุ่มที่ควรได้คือ start ไม่ใช่ download


def test_agent_info_reports_runtimes_and_images(tmp_path, monkeypatch, request):
    """host.runtimes.llamacpp[] จาก llama-server --version + git (ไม่ fetch) · host.images[] เฉพาะ image ที่ bundle อ้าง
    · ห้าม docker run · models[] บอก generated_by/template_hash/controller.state/runtime ที่ผูก"""
    import subprocess

    import yaml

    from lmds.fleet import find
    from lmds.generator.renderer import template_hash

    # build llama.cpp ปลอม: llama-server ตอบ version · lock เก่ากว่า build (stale) · git ปลอมตอบ merge-base
    llama = tmp_path / "src" / "llama.cpp"
    (llama / ".git").mkdir(parents=True)
    (llama / "build" / "bin").mkdir(parents=True)
    server = llama / "build" / "bin" / "llama-server"
    server.write_text("#!/bin/bash\necho 'version: 10495 (3dc7285b4)'\n", encoding="utf-8")
    server.chmod(0o755)
    (llama / "build" / "bin" / "libllama.so").write_bytes(b"\x00llama\x00qwen35\x00")
    (llama / "build" / "runtime.lock").write_text("ece963f41\n", encoding="utf-8")
    shims = tmp_path / "shims"
    shims.mkdir()
    calls = tmp_path / "calls.log"
    (shims / "git").write_text(
        "#!/bin/bash\n"
        f'echo "git $*" >> {calls}\n'
        'case "$*" in *fetch*) exit 9 ;; *"log -1"*) echo 2026-08-18 ;; *merge-base*) exit 0 ;; esac\nexit 0\n',
        encoding="utf-8")
    (shims / "git").chmod(0o755)
    docker_calls = []
    real_run = subprocess.run

    def fake_run(args, **kw):
        if list(args)[:1] == ["docker"]:
            docker_calls.append(list(args))
            assert args[1] != "run", "agent info ห้าม docker run"
            return subprocess.CompletedProcess(
                args, 0, "sha256:abcdef0123456789|vllm/vllm-openai@sha256:61fc8a8aaaaa|2026-09-01T10:00:00Z\n", "")
        return real_run(args, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PATH", f"{shims}:/usr/bin:/bin")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    # serving.detect() แคชทั้ง process — ผลที่ตรวจใต้ which ปลอมต้องไม่ไปหลอกเทส doctor ที่รันทีหลัง
    from lmds.hardware import serving

    serving.reset_cache()
    request.addfinalizer(serving.reset_cache)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models" / "none"))
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)

    def register(slug, profile, mode):
        d = tmp_path / "bundles" / slug
        d.mkdir(parents=True)
        ctl = d / f"{slug}-single.sh"
        ctl.write_text('#!/usr/bin/env bash\nSCRIPT_VERSION="${SCRIPT_VERSION:-0.5.1}"\ncase $1 in\n  download) : ;;\n  start) : ;;\nesac\n',
                       encoding="utf-8")
        (d / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
        run_dir = tmp_path / "run" / slug
        run_dir.mkdir(parents=True)
        (run_dir / "server.meta").write_text(
            f"slug={slug}\nmodel={slug}\nmodel_id=o/{slug}\nengine={profile['runtime']['engine']}\nmode={mode}\n"
            f"port=8000\ncontainer=lmds-{slug}\npid_file=\ncontroller={ctl}\nstarted_at=\n", encoding="utf-8")

    register("qwen3-8-27b-gguf", {
        "generated_by": "lmds 0.5.1",
        "model": {"id": "o/Qwen-GGUF", "selected_gguf": "q.gguf", "gguf_architecture": "qwen35"},
        "runtime": {"engine": "llamacpp", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda", "native_build": True},
        "target": {"memory_model": "unified", "llamacpp_dir": str(llama)}, "topology": "single"}, "native")
    register("glm-vllm", {
        "generated_by": "lmds 0.6.0", "template_hash": "0ld", "model": {"id": "o/GLM"},
        "runtime": {"engine": "vllm", "image": "vllm/vllm-openai:latest", "image_pin": "sha256:61fc8a8aaaaa"},
        "topology": "single"}, "docker")

    models = [inventory.model_payload(find("qwen3-8-27b-gguf")), inventory.model_payload(find("glm-vllm"))]
    gguf, vllm = models
    assert gguf["generated_by"] == "lmds 0.5.1" and gguf["script_version"] == "0.5.1" and gguf["template_hash"] is None
    assert gguf["controller"]["state"] == "stale" and "0.5.1" in gguf["controller"]["reason"]
    assert gguf["runtime"] == {"engine": "llamacpp", "mode": "native", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                               "image_pin": "", "llamacpp_dir": str(llama)}
    assert gguf["runtime_arch"]["supported"] is True
    assert vllm["controller"]["state"] == "stale" and vllm["runtime"]["mode"] == "docker"
    assert vllm["runtime"]["image_pin"] == "sha256:61fc8a8aaaaa"

    host = inventory.with_runtimes(inventory.host_payload(), models)
    assert host["template_hash"] == template_hash() and host["lmds_installed_commit"] == inventory.installed_commit()
    assert isinstance(host["source_dirty"], list)
    (build,) = host["runtimes"]["llamacpp"]
    assert build["dir"] == str(llama) and build["present"] and build["build"] == "10495" and build["commit"] == "3dc7285b4"
    assert build["date"] == "2026-08-18" and build["lock"] == "ece963f41" and build["lock_state"] == "stale"
    assert build["used_by"] == ["qwen3-8-27b-gguf"]
    assert "fetch" not in calls.read_text(encoding="utf-8")
    (image,) = host["images"]
    assert image["ref"] == "vllm/vllm-openai@sha256:61fc8a8aaaaa" and image["present"]
    assert image["digest"] == "sha256:61fc8a8aaaaa" and image["used_by"] == ["glm-vllm"]
    assert image["created"].startswith("2026-09-01")
    assert any(c[:3] == ["docker", "image", "inspect"] for c in docker_calls)
    assert not any(c[:2] == ["docker", "run"] for c in docker_calls)
    summary = inventory.summary_of(models)
    assert summary["controllers_stale"] == 2 and summary["runtime_stale"] == 0
    # ไม่มี doctor ในรายการคำสั่งของ controller (มันเป็นคำสั่งของ lmds ไม่ใช่ของ template)
    assert "doctor" not in inventory.KNOWN_COMMANDS


def test_llama_server_version_line_parses_both_formats(tmp_path):
    """เคสจริง 2026-09-06 ทั้งฟลีตขึ้น "build ?" — llama-server รุ่นใหม่พิมพ์
    `version: 0.1.2-dev (build 10495, commit 3dc7285b4)` แต่ regex จับได้แต่แบบเก่า `version: 10495 (3dc7285b4)`"""
    from lmds import inventory

    for text, build, commit in (
        ("version: 10495 (3dc7285b4)", "10495", "3dc7285b4"),
        ("version: 0.1.2-dev (build 10495, commit 3dc7285b4)\nbuilt with GNU 13.3.0 for Linux aarch64", "10495", "3dc7285b4"),
        ("version: 0.4.0-dev (build 10826, commit 73a43d1f6)", "10826", "73a43d1f6"),
    ):
        found = inventory._VERSION_LINE.search(text)
        assert found and found.group(1) == build and found.group(2) == commit, text

    llama = tmp_path / "src" / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    server = llama / "build" / "bin" / "llama-server"
    server.write_text("#!/bin/bash\necho 'version: 0.4.0-dev (build 10826, commit 73a43d1f6)'\necho 'built with GNU 13.3.0 for Linux aarch64'\n", encoding="utf-8")
    server.chmod(0o755)
    info = inventory.llamacpp_runtime_info(llama)
    assert info["build"] == "10826" and info["commit"] == "73a43d1f6", info


def test_llamacpp_context_per_request_is_context_divided_by_slots(tmp_path, monkeypatch):
    """เคสจริง 2026-09-07 dgx-veerasiam: gemma-4-12b ตั้ง context 131,072 slots 2 → llama.cpp ให้ 65,536 ต่อ request แต่การ์ด
    โชว์ 131,072 · vLLM ไม่แบ่ง (max-model-len เป็นต่อคำขอ)"""
    from lmds import inventory

    controller = tmp_path / "demo-single.sh"
    controller.write_text("#!/bin/bash\ncase \"$1\" in download) ;; start) ;; esac\n", encoding="utf-8")
    server = _server(controller)
    server.engine = "llamacpp"
    server.running = True
    import lmds.fleet as fleet

    monkeypatch.setattr(fleet, "running_context", lambda s: 131072)
    monkeypatch.setattr(fleet, "running_slots", lambda s: 2)
    payload = inventory.model_payload(server)
    assert payload["context"] == 131072 and payload["slots"] == 2 and payload["context_per_request"] == 65536

    server.engine = "vllm"
    monkeypatch.setattr(fleet, "running_context", lambda s: 262144)
    monkeypatch.setattr(fleet, "running_slots", lambda s: 3)
    payload = inventory.model_payload(server)
    assert payload["context_per_request"] == 262144 and payload["slots"] == 3


def test_memory_by_slug_counts_host_rss_of_native_servers(tmp_path, monkeypatch):
    """dgx-veerasiam 2026-09-07: gemma-4-12b (llama.cpp native) ถือ GPU 17.7 GB + VmRSS 11.2 GB (mmap weight ค้าง) ขณะ
    `free` บอก used 119 GB — นับแค่ GPU แล้ว Fit คิดว่าเหลือที่ทั้งที่เครื่องเริ่ม swap · unified memory ต้องรวม RSS"""
    from lmds import inventory
    from lmds.hardware import profiler

    controller = tmp_path / "demo-single.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    server = _server(controller)
    server.running = True
    server.mode = "native"
    server.pid = 4242
    monkeypatch.setattr(profiler, "compute_apps", lambda: [(4242, "llama-server", 17705)])
    monkeypatch.setattr(inventory, "_rss_gb", lambda pid: 11.2 if pid == 4242 else 0.0)
    assert inventory.memory_by_slug([server]) == {"demo": round(17705 / 1024 + 11.2, 1)}


# ───────────────────────── audit 2026-09-08: features ที่เป็นจริง · MTP จาก argv · ตั้งค่าแล้วยังไม่ restart · usage ─────────────────────────

def _profile_vllm(**extra):
    return {"model": {"id": "nvidia/Qwen3.6-35B-A3B-NVFP4", "task": "generate"},
            "runtime": {"engine": "vllm"}, "features": {"moe": {"experts": 128, "experts_active": 8}}, **extra}


def test_vision_is_derived_from_config_json_argv_or_adopt_profile_for_vllm(tmp_path, monkeypatch):
    """gemma-4-31B-it (adopt) และ nvidia/Qwen3.6-35B-A3B-NVFP4 บน vLLM ขึ้น text/MoE ทั้งที่ตอบภาพได้ — plan ไม่เคยจด modalities
    ให้ safetensors · inventory ต้องดู config.json (vision_config/image_token_id/mm_*), argv (--limit-mm-per-prompt) และ
    multimodal.projector ที่ adopt เขียน"""
    from lmds import inventory

    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    server = _server(controller)
    server.engine = "vllm"
    server.running = True
    # config.json ของ Qwen3.6 จริง (dgx-spark04): vision_config + image_token_id + vision_start_token_id
    cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
           "vision_config": {"depth": 27}, "image_token_id": 248056, "vision_start_token_id": 248053}
    assert inventory.vision_from_config(cfg) is True
    assert inventory.vision_from_config({"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {"model_type": "siglip"}}) is True
    assert inventory.vision_from_config({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}) is False
    assert inventory.vision_from_config({"architectures": ["Gemma3nForConditionalGeneration"]}) is False, "ชื่อ arch อย่างเดียวไม่พอ"
    assert inventory.vision_from_config({"mm_tokens_per_image": 256}) is True

    features, note = inventory.effective_features(_profile_vllm(), server, argv=[], config=cfg)
    assert features == "image, MoE 128e/8a" and note == ""
    # ไม่มี config.json แต่ argv เปิด multimodal ไว้
    features, _ = inventory.effective_features(_profile_vllm(), server, argv=["serve", "x", "--limit-mm-per-prompt", '{"image":4}'], config={})
    assert "image" in features
    # adopt เขียน multimodal.projector ไว้ (โปรไฟล์เก่าที่ยังไม่มี modalities)
    features, _ = inventory.effective_features(_profile_vllm(features={"multimodal": {"projector": True}}), server, argv=[], config={})
    assert "image" in features
    # ไม่มีหลักฐานเลย = text (ไม่เดา)
    features, _ = inventory.effective_features(_profile_vllm(features={}), server, argv=[], config={})
    assert features == "text"
    # hf_config อ่านจากแคช HF ตาม profile (ทั้ง hub/ layout)
    snap = tmp_path / "hf" / "hub" / "models--nvidia--Qwen3.6-35B-A3B-NVFP4" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert inventory.hf_config(_profile_vllm(), server).get("image_token_id") == 248056
    features, _ = inventory.effective_features(_profile_vllm(), server, argv=[])
    assert "image" in features


def test_llamacpp_vl_embedding_does_not_claim_image_and_says_why(tmp_path):
    """qwen3-vl-embedding-8b-gguf: mmproj มากับไฟล์ แต่ llama-server ทิ้งภาพบน /v1/embeddings (vector เท่ากันมี/ไม่มีภาพ)
    — ป้าย vision บนการ์ดโกหก client ที่หวัง multimodal embedding · ตัด image ออก + หมายเหตุ · projector ไม่ขึ้นป้ายแทน"""
    from lmds import inventory

    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\ncase $1 in\n  start) ;;\n  download) ;;\n  test-embed) ;;\nesac\n", encoding="utf-8")
    server = _server(controller)
    server.engine = "llamacpp"
    profile = {"model": {"id": "Qwen/Qwen3-VL-Embedding-8B-GGUF", "task": "embed", "selected_gguf": "q.gguf"},
               "runtime": {"engine": "llamacpp"},
               "features": {"multimodal": {"modalities": ["image", "text"], "projector_files": ["mmproj-F16.gguf"]},
                            "embedding": {"pooling": "last"}}}
    features, note = inventory.effective_features(profile, server, argv=[])
    assert features == "text, embedding (last)" and "mmproj" in note and "/v1/embeddings" in note
    import lmds.fleet as fleet
    payload_profile = profile
    import lmds.inventory as inv
    orig = fleet.bundle_profile
    fleet.bundle_profile = lambda c: payload_profile
    try:
        payload = inv.model_payload(server)
    finally:
        fleet.bundle_profile = orig
    assert payload["features"] == "text, embedding (last)" and payload["projector"] is False
    assert "mmproj" in payload["feature_note"]
    # โมเดล chat ที่มี mmproj ยังขึ้น image ตามเดิม
    chat = {**profile, "model": {**profile["model"], "task": "generate"}, "features": {"multimodal": profile["features"]["multimodal"]}}
    features, note = inventory.effective_features(chat, server, argv=[])
    assert features == "image, text" and note == ""


def test_speculative_is_read_from_the_running_argv_and_bundle_args(tmp_path):
    """dgx-spark04: Qwen3.6 vLLM รัน --speculative-config {"method":"mtp",…} แต่การ์ดบอกว่าไม่มี MTP"""
    from lmds import inventory

    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    server = _server(controller)
    server.engine = "vllm"
    server.running = True
    argv = ["serve", "nvidia/Qwen3.6-35B-A3B-NVFP4", "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
    assert inventory.speculative_active(_profile_vllm(), server, argv) is True
    assert inventory.speculative_active(_profile_vllm(), server, ["serve", "x"]) is False
    (tmp_path / "bundle.args").write_text('--speculative-config {"method":"mtp"}\n', encoding="utf-8")
    assert inventory.speculative_active(_profile_vllm(), server, ["serve", "x"]) is True, "bundle.args นับด้วย (มีผลรอบ start ถัดไป)"
    (tmp_path / "bundle.args").unlink()
    assert inventory.speculative_active(_profile_vllm(features={"speculative": {"embedded": True}}), server, []) is True
    assert inventory.speculative_active({"serving": {"extra_flags": ["--speculative-config", "{}"]}}, server, []) is True
    features, _ = inventory.effective_features(_profile_vllm(), server, argv=argv, config={})
    assert features.endswith("MTP")


def test_pending_restart_compares_saved_settings_with_the_running_argv(tmp_path):
    """msi-6 2026-09-08: `lmds set --model-id` แล้ว API ยังตอบชื่อเก่า — bundle.env ≠ argv จน restart · ป้าย custom คือกรณีกลับกัน"""
    from lmds import inventory
    from lmds.fleet.bundle_settings import write

    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    server = _server(controller)
    server.engine = "vllm"
    server.running = True
    running = ["serve", "org/m", "--served-model-name", "old-name", "--max-model-len", "65536", "--max-num-seqs", "4",
               "--port", "8000", "--gpu-memory-utilization", "0.85", "--tool-call-parser", "qwen3_xml"]
    assert inventory.pending_restart(server, argv=running) is None, "ไม่มี bundle.env = ไม่มีอะไรค้าง"
    write(tmp_path, {"served_name": "new-name", "context": 65536, "slots": 4, "port": 8000, "gpu_util": "0.85",
                     "tool_parser": "qwen3_xml"})
    drift = inventory.pending_restart(server, argv=running)
    assert drift == {"pending": True, "changes": [{"field": "served_name", "saved": "new-name", "running": "old-name"}]}
    write(tmp_path, {"served_name": "old-name", "context": 65536, "slots": 4, "port": 8000, "gpu_util": "0.85",
                     "tool_parser": "qwen3_xml", "extra_args": '--speculative-config {"method":"mtp"}'})
    drift = inventory.pending_restart(server, argv=running)
    assert drift["pending"] and drift["changes"][0]["field"] == "extra_args" and "speculative" in drift["changes"][0]["saved"]
    with_extra = running + ["--speculative-config", '{"method":"mtp"}']
    assert inventory.pending_restart(server, argv=with_extra) == {"pending": False, "changes": []}
    # flag ที่ argv ไม่มี (controller เก่าไม่ส่ง --max-num-seqs) = ไม่นับว่าค้าง · flag=value ก็อ่านได้
    write(tmp_path, {"slots": 8, "context": 65536})
    assert inventory.pending_restart(server, argv=["serve", "x", "--max-model-len=65536"])["pending"] is False
    assert inventory.pending_restart(server, argv=["serve", "x", "--max-model-len=32768"])["changes"][0]["field"] == "context"
    # llama.cpp: --alias / --ctx-size / --parallel
    server.engine = "llamacpp"
    write(tmp_path, {"served_name": "qwen", "context": 131072, "slots": 2})
    llama = ["llama-server", "-m", "x.gguf", "--alias", "qwen", "--ctx-size", "131072", "--parallel", "1"]
    assert inventory.pending_restart(server, argv=llama)["changes"] == [{"field": "slots", "saved": "2", "running": "1"}]
    server.running = False
    assert inventory.pending_restart(server, argv=llama) is None
    server.running = True
    payload = inventory.model_payload(server)      # argv ของ process ปลอมอ่านไม่ได้ → None ไม่ล้ม
    assert payload["pending_restart"] is None
    assert inventory.summary_of([{"running": True, "healthy": True, "downloaded": True, "pending_restart": {"pending": True}},
                                 {"running": True, "healthy": True, "downloaded": True, "pending_restart": None}])["restart_pending"] == 1


class _FakeMetrics(http.server.BaseHTTPRequestHandler):
    body = ""

    def log_message(self, *_a):
        pass

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        if type(self).body is None:
            raw = b'{"error":{"code":501,"message":"This server does not support metrics endpoint. Start it with `--metrics`"}}'
            self.send_response(501)
        else:
            raw = type(self).body.encode()
            self.send_response(200)
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)


def test_usage_counts_llamacpp_from_metrics_with_a_24h_window_and_falls_back_to_launch_slot(tmp_path):
    """server.log ของ build ปัจจุบันไม่มีบรรทัด POST (0 POST · 653 launch_slot_ บน dgx-spark02) — นับจาก /metrics เก็บตัวอย่าง
    แล้วคิดส่วนต่างในหน้าต่าง 24 ชม. · ไม่มี /metrics (501) = นับ launch_slot_ ตั้งแต่ start · แคช 5 นาที"""
    from lmds import inventory

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeMetrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        controller = tmp_path / "ctl.sh"
        controller.write_text("#!/bin/bash\n", encoding="utf-8")
        info = _server(controller)
        info.engine = "llamacpp"; info.mode = "native"; info.running = True; info.port = port
        info.run_dir = tmp_path / "run"; info.run_dir.mkdir()
        (info.run_dir / "server.log").write_text("srv  launch_slot_: id 0\nslot release\nsrv  launch_slot_: id 0\n" * 3, encoding="utf-8")
        inventory._USAGE_CACHE.clear()

        _FakeMetrics.body = ("# HELP llamacpp:prompt_tokens_total x\nllamacpp:prompt_tokens_total 1000\n"
                             "llamacpp:tokens_predicted_total 500\nllamacpp:requests_processing 0\n")
        t0 = 1_700_000_000.0
        first = inventory.request_usage(info, now=t0)
        assert first["source"] == "metrics" and first["requests_24h"] == 6 and first["prompt_tokens_24h"] == 1000
        assert "ตั้งแต่ start" in first["note"]
        assert first["generated_tokens_24h"] == 500
        # 5 นาทีถัดไป = แคช
        assert inventory.request_usage(info, now=t0 + 60) == first
        # ผ่านไป 2 ชม.: ตัวนับโต → ส่วนต่างในหน้าต่าง
        _FakeMetrics.body = "llamacpp:prompt_tokens_total 4000\nllamacpp:tokens_predicted_total 1500\n"
        (info.run_dir / "server.log").write_text("launch_slot_\n" * 10, encoding="utf-8")
        later = inventory.request_usage(info, now=t0 + 7200)
        assert later["requests_24h"] == 4 and later["prompt_tokens_24h"] == 3000 and later["generated_tokens_24h"] == 1000
        assert later["note"] == ""
        # 30 ชม. ถัดมา: ตัวอย่างแรกหลุดหน้าต่าง 24 ชม. — ส่วนต่างคิดจากตัวอย่างที่ยังอยู่
        _FakeMetrics.body = "llamacpp:prompt_tokens_total 4100\nllamacpp:tokens_predicted_total 1550\n"
        (info.run_dir / "server.log").write_text("launch_slot_\n" * 11, encoding="utf-8")
        much_later = inventory.request_usage(info, now=t0 + 30 * 3600)
        assert much_later["requests_24h"] == 1 and much_later["prompt_tokens_24h"] == 100
        rows = [json.loads(l) for l in (info.run_dir / "usage.samples").read_text(encoding="utf-8").splitlines()]
        assert [r["t"] for r in rows] == [t0, t0 + 7200, t0 + 30 * 3600], "เก็บถึง 48 ชม. — ฐาน 'เมื่อ 24 ชม. ก่อน' ต้องมีเสมอ"
        # server restart: ตัวนับถอยหลัง → ประวัติเดิมไม่ต่อกัน นับใหม่ตั้งแต่ start
        _FakeMetrics.body = "llamacpp:prompt_tokens_total 50\nllamacpp:tokens_predicted_total 5\n"
        (info.run_dir / "server.log").write_text("launch_slot_\n" * 2, encoding="utf-8")
        inventory._USAGE_CACHE.clear()
        restarted = inventory.request_usage(info, now=t0 + 31 * 3600)
        assert restarted["requests_24h"] == 2 and restarted["prompt_tokens_24h"] == 50
        # ไม่มี /metrics (start ก่อนมี --metrics) = launch_slot_ ตั้งแต่ start
        _FakeMetrics.body = None
        inventory._USAGE_CACHE.clear()
        fallback = inventory.request_usage(info, now=t0 + 32 * 3600)
        assert fallback["source"] == "server-log" and fallback["requests_24h"] == 2 and "--metrics" in fallback["note"]
        info.running = False
        assert inventory.request_usage(info) is None
    finally:
        server.shutdown()
    assert inventory.parse_metrics('vllm:request_success_total{finished_reason="stop"} 30\nvllm:request_success_total{finished_reason="length"} 7\n')["vllm:request_success_total"] == 37


def test_usage_counts_vllm_from_docker_logs(tmp_path, monkeypatch):
    import subprocess

    from lmds import inventory

    controller = tmp_path / "ctl.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    info = _server(controller)
    info.engine = "vllm"; info.mode = "docker"; info.container = "lmds-demo"; info.running = True; info.port = 8000
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, 'INFO: 10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 200\n'
                                                    'INFO: "GET /health" 200\nINFO: "POST /v1/embeddings HTTP/1.1" 200\n', "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    inventory._USAGE_CACHE.clear()
    usage = inventory.request_usage(info, now=1.0)
    assert usage == {"requests_24h": 2, "prompt_tokens_24h": None, "generated_tokens_24h": None, "source": "docker-log", "note": ""}
    assert calls == [["docker", "logs", "--since", "24h", "lmds-demo"]]
    assert inventory.request_usage(info, now=100.0) == usage and len(calls) == 1, "แคช 5 นาที — ไม่ docker logs ทุกรอบ refresh"
