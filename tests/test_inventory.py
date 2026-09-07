"""ภาพรวมเครื่องที่ hub ดึงผ่าน `lmds agent info`"""

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
