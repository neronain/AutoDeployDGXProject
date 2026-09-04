"""เทสจาก audit 2026-09-04 (ชุด backend: web / nodes / fleet) — ตามหลัง review รอบใหญ่ของ 0.6.0

แต่ละเทสคือข้อบกพร่องหนึ่งข้อที่ผู้ใช้เจอจริงวันนี้หรือที่ไล่เส้นทางแล้วเจอ · เขียนให้ล้มก่อน
แล้วค่อยแก้ — เหมือน tests/test_review_*.py
"""

from __future__ import annotations

import os
import stat
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.web import create_app  # noqa: E402
from tests.test_web import (  # noqa: E402,F401 — fixture ใช้ร่วมกัน (autouse ด้วย)
    fleet, fresh_jobs, no_host_scan, registered, wait_for_job,
)


# ── 1. remove: weight ที่ container เขียนไว้เป็น root ลบผ่าน docker แทน sudo ─────────
def _readonly_tree(root: Path) -> Path:
    """โฟลเดอร์ที่ user ลบไม่ได้ — จำลอง weight ของ root โดยไม่ต้องเป็น root (chmod 500 ที่ parent)"""
    inner = root / "snapshots"
    inner.mkdir(parents=True)
    (inner / "model.safetensors").write_bytes(b"x" * 64)
    inner.chmod(stat.S_IRUSR | stat.S_IXUSR)   # unlink ลูกข้างในล้มด้วย EACCES
    return inner


def _fake_docker(bin_dir: Path, log: Path) -> None:
    """docker ปลอม: จด argv แล้วทำหน้าที่ของ `docker run … rm -rf /x/<name>` ในฐานะ user นี้

    ของจริงรันเป็น root ในคอนเทนเนอร์ จึงลบไฟล์ของ root ได้ · ที่นี่คืนสิทธิ์ให้ตัวเองก่อนลบ
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "docker"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        'if [ "$1" = images ]; then echo "alpine:3.20"; echo "<none>:<none>"; exit 0; fi\n'
        'if [ "$1" = run ]; then\n'
        '  host=""; while [ $# -gt 0 ]; do case "$1" in -v) host="${2%%:*}"; shift 2;; *) shift;; esac; done\n'
        f'  target="$(tail -1 {log} | sed -E "s#.* rm -rf -- /x/##")"\n'
        '  chmod -R u+rwx "$host/$target" 2>/dev/null; rm -rf -- "$host/$target"; exit 0\n'
        'fi\n'
        'exit 1\n',
        encoding="utf-8",
    )
    script.chmod(0o755)


def test_root_owned_weights_are_removed_through_docker_when_plain_rm_fails(tmp_path, monkeypatch):
    """เคสจริง 2026-09-04: `lmds remove` ของ bundle ที่ adopt มา — weight ที่ container โหลดเป็น root
    ลบไม่ออก CLI พิมพ์ "ต้องใช้ sudo rm -rf" แล้วยอมแพ้ ทั้งที่ผู้ใช้อยู่ในกลุ่ม docker
    ซึ่ง `docker run --rm -v <dir>:/x <image> rm -rf /x/<name>` ลบได้อยู่แล้ว
    """
    from lmds.fleet import manager

    home = Path.home()
    weights = home / "models" / "adopted-demo"
    inner = _readonly_tree(weights)
    log = tmp_path / "docker.log"
    _fake_docker(tmp_path / "bin", log)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")

    info = SimpleNamespace(slug="adopted-demo", running=False, controller="", controller_exists=False,
                           engine="llamacpp", model_id="", mode="docker", container="", run_dir=Path())
    monkeypatch.setattr(manager, "removal_plan",
                        lambda info, include_weights=True: [manager.RemovalItem("weight ของโมเดล", weights, 64, True)])
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    try:
        lines = manager.remove_server(info)
    finally:
        if inner.exists():
            inner.chmod(0o700)
    assert not weights.exists(), lines
    assert not manager.removal_failed(lines), lines
    assert any("docker" in line for line in lines), lines
    calls = log.read_text(encoding="utf-8").splitlines()
    run_line = next(c for c in calls if c.startswith("run "))
    assert f"-v {weights.parent}:/x" in run_line and run_line.endswith("rm -rf -- /x/adopted-demo")
    assert "alpine:3.20" in run_line            # image ในเครื่อง ไม่ pull ของใหม่


def test_docker_fallback_only_touches_the_users_own_trees(tmp_path, monkeypatch):
    """path นอก home / HF cache ห้ามส่งให้ docker ลบเป็น root เด็ดขาด — ต่อให้ rm ล้ม"""
    from lmds.fleet import manager

    log = tmp_path / "docker.log"
    _fake_docker(tmp_path / "bin", log)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    assert manager._docker_rm(Path("/etc/lmds-nope")) is False
    assert manager._docker_rm(Path.home()) is False       # ทั้ง home ก็ไม่ใช่ของที่ remove ควรลบ
    assert not log.exists()
    hf = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf))
    victim = hf / "hub" / "models--org--x"
    victim.mkdir(parents=True)
    assert manager._docker_rm(victim) is True and not victim.exists()


# ── 2. port ชนกัน: bundle ใหม่ต้องได้พอร์ตว่างของเครื่องปลายทาง ────────────────
def _seed_node(name: str, ports: list[int], gpu: str = "NVIDIA GB10", memory_model: str = "unified") -> None:
    from lmds.web import state

    state.STORE.set_node(name, {
        "host": {"hostname": name, "gpus": [{"name": gpu, "vram_gb": 32.0, "vram_used_gb": 0.0}],
                 "memory_model": memory_model, "foreign": []},
        "models": [{"slug": f"m{p}", "port": p, "running": i == 0} for i, p in enumerate(ports)],
    })


def test_new_bundle_gets_the_first_free_port_on_the_target_machine(tmp_path, monkeypatch):
    """เคสจริง 2026-09-04: ทุก bundle ใหม่ได้ 8000 ทั้งที่เครื่องนั้นมี 8000 อยู่แล้ว → หน้าภาพรวมขึ้น
    "port shared" ทันทีหลัง deploy · แผนต้องเลือกพอร์ตว่างจาก inventory ของเครื่องปลายทาง
    และเขียนลง bundle.env เพื่อให้ start/autostart บนเครื่องนั้นได้พอร์ตเดียวกัน
    """
    from tests.test_generator import safetensors_report
    from lmds.fleet.bundle_settings import read as read_settings
    from lmds.web import deploy as dep

    _seed_node("spark2", [8000, 8001, 8003])
    assert dep.suggest_port("spark2") == 8002
    assert dep.suggest_port("never-probed") == 8000   # ไม่มีข้อมูล = ค่าเดิม ไม่เดามั่ว

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True, machine="spark2")
    assert analyzed["plan"]["port"] == 8002
    assert "8002" in " ".join(analyzed["plan"]["fit"]["notes"] + analyzed["notes"])

    result = dep.generate(analyzed["id"], context=16384, output=str(tmp_path))
    assert result["port"] == 8002
    saved = read_settings(Path(result["directory"]))
    assert saved["port"] == "8002"
    assert f"{result['slug']}/bundle.env" in zipfile.ZipFile(result["zip"]).namelist()


def test_a_never_started_bundle_reports_the_port_it_will_actually_use(tmp_path, monkeypatch):
    """ทะเบียนที่ register_bundle เขียนเคยตั้ง port=8000 เสมอ — bundle ที่ wizard ให้ 8002 จึงขึ้นการ์ด
    ว่า 8000 และหน้าภาพรวมแจ้ง "port shared" กับตัวที่ 8000 จริง จนกว่าจะ start ครั้งแรก
    """
    from lmds.fleet import discover, register_bundle
    from lmds.fleet.bundle_settings import write as write_settings

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    bundle = tmp_path / "bundles" / "later"
    bundle.mkdir(parents=True)
    controller = bundle / "later-single.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    (bundle / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": "org/later"}, "runtime": {"engine": "llamacpp"}, "serving": {}}), encoding="utf-8")
    write_settings(bundle, {"port": 8002})
    register_bundle(controller)
    entry = next(s for s in discover() if s.slug == "later")
    assert entry.port == 8002


def test_hub_deploy_avoids_ports_already_used_on_the_hub_itself():
    from lmds.web import deploy as dep, state

    state.STORE.set_local({"host": {"gpus": []},
                           "models": [{"slug": "a", "port": 8000}, {"slug": "b", "port": 8000}]})
    assert dep.suggest_port("") == 8001


def test_port_override_in_generate_is_validated(tmp_path, monkeypatch):
    from tests.test_generator import safetensors_report
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(f"/api/deploy/{analyzed['id']}/generate", json={"port": "abc", "output": str(tmp_path)})
    assert r.status_code == 422 and r.json()["detail"]["kind"] == "input"
    assert "port" in r.json()["detail"]["message"]
    r = client.post(f"/api/deploy/{analyzed['id']}/generate", json={"port": 9100, "output": str(tmp_path)})
    assert r.status_code == 200 and r.json()["port"] == 9100


# ── 3. เลือกเครื่องในฟลีตโดยไม่ระบุ target = วิเคราะห์ด้วยฮาร์ดแวร์ของ hub ──────────
def test_machine_without_target_uses_that_machines_hardware_not_the_hubs(monkeypatch):
    """hub เป็น VM ไม่มี GPU · ผู้ใช้เลือก msi-5 (RTX 5090) แต่ไม่ได้เลือก preset → เดิมแผนคิดจาก
    ฮาร์ดแวร์ของ hub (ตกไป dgx-spark-single 128 GB) แล้วเสนอ context ที่การ์ด 32 GB รับไม่ไหว
    """
    from tests.test_generator import safetensors_report
    from lmds.fit.analyzer import GIB
    from lmds.web import deploy as dep

    _seed_node("msi-5", [8000], gpu="NVIDIA GeForce RTX 5090", memory_model="discrete")
    monkeypatch.setattr("lmds.inspector.inspect_model",
                        lambda source, client: safetensors_report(weight_bytes=8 * GIB))
    analyzed = dep.analyze("Qwen/Small", no_llm=True, machine="msi-5")
    assert analyzed["plan"]["fit"]["target"] == "rtx-5090"


# ── 4. analyze: Xet first-byte ช้า 20–60 วิ ต้องไม่โดน timeout ภายใน และ HfError = 4xx ──
def test_hub_client_read_timeout_survives_xet_first_byte_latency():
    """ไฟล์บน Xet bridge ตอบ byte แรกช้าได้ถึงนาที · read timeout 30 วิเดิม = analyze ล้มทั้งที่ Hub ปกติ"""
    from lmds.inspector import HfClient

    timeout = HfClient()._client.timeout
    assert timeout.read is not None and timeout.read >= 90
    assert timeout.connect is not None and timeout.connect <= 30   # ต่อไม่ติดยังต้องรู้เร็ว


def test_hub_errors_reach_the_browser_as_4xx_with_the_reason(monkeypatch):
    from lmds.inspector import HfError

    def down(source, client):
        raise HfError("ต่อ Hugging Face ไม่ได้ (ReadTimeout) — เช็คอินเทอร์เน็ต")

    monkeypatch.setattr("lmds.inspector.inspect_model", down)
    r = TestClient(create_app(), raise_server_exceptions=False).post(
        "/api/deploy/analyze", json={"model": "Qwen/Qwen3-32B", "target": "dgx-spark-single", "no_llm": True})
    assert r.status_code == 422
    assert r.json()["detail"]["kind"] == "hub"
    assert "ReadTimeout" in r.json()["detail"]["message"]


# ── 5. concurrency ────────────────────────────────────────────────────────────
class _BlockingPipe:
    """ท่อที่คายบรรทัดแรกแล้วค้างจนกว่าเทสจะปล่อย — งานจึง "รันอยู่" นานพอให้ยิงซ้อน"""

    def __init__(self, first: bytes, release: threading.Event):
        self._first = first
        self._release = release

    def read1(self, _size=-1):
        if self._first:
            chunk, self._first = self._first, b""
            return chunk
        self._release.wait(5)
        return b""

    def close(self):
        pass


class _BlockingStream:
    def __init__(self, first: bytes, release: threading.Event, code: int = 0):
        self.stdout = _BlockingPipe(first, release)
        self._code = code

    def wait(self):
        return self._code

    def terminate(self):
        pass


def test_two_tabs_starting_the_same_remote_job_get_exactly_one_job(registered, monkeypatch):
    from lmds.web import jobs

    release = threading.Event()
    monkeypatch.setattr("lmds.nodes.stream", lambda *a, **k: _BlockingStream(b"go\n", release))
    outcomes: list = []

    def fire():
        try:
            outcomes.append(jobs.start_remote("spark2", "demo", "start", "lmds start demo"))
        except jobs.JobError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=fire) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    release.set()
    started = [o for o in outcomes if isinstance(o, jobs.Job)]
    assert len(started) == 1 and len(outcomes) == 6
    for job in started:
        for _ in range(100):
            if not job.running:
                break
            time.sleep(0.02)


def test_a_remote_command_finishing_during_a_probe_does_not_leave_the_old_picture_for_15s():
    """refresher ออกตัว probe เครื่อง A → ระหว่างนั้น job start บน A จบ (STORE.force) → probe เก่าเขียนทับ
    ด้วยภาพ *ก่อน* start แล้วนอนอยู่ 15 วิ · เหมือน epoch ของเครื่องนี้ (set_local) แต่ node ไม่เคยมี
    """
    from lmds.web import state

    store = state.STORE
    epoch = store.node_epoch("a")
    store.force("a")                                   # งานเปลี่ยนสถานะจริงเพิ่งจบ
    store.set_node("a", {"host": {}, "models": []}, epoch=epoch)   # ผล probe ที่ออกตัวก่อนหน้านั้น
    snap = store.snapshot()["nodes"]["a"]
    assert snap["data"] is not None                    # ยังแสดงได้ ดีกว่าว่างเปล่า
    assert store.due("a") is True                      # แต่ต้องไปสำรวจใหม่ทันที ไม่ใช่อีก 15 วิ

    fresh = store.node_epoch("a")
    store.set_node("a", {"host": {}, "models": [{"slug": "x"}]}, epoch=fresh)
    assert store.due("a") is False


def test_sse_version_is_stable_when_nothing_changed():
    """drop_missing() ขยับ version ทุกวิแม้ไม่ได้ลบอะไร → /api/events ส่ง snapshot ทั้งก้อนวินาทีละครั้ง
    ให้ทุกเบราว์เซอร์ที่เปิดอยู่ ทั้งที่ไม่มีอะไรเปลี่ยน
    """
    from lmds.web import state

    store = state.STORE
    store.set_node("a", {"host": {}, "models": []})
    before = store.version
    store.drop_missing({"a"})
    store.drop_missing({"a", "b"})
    assert store.version == before
    store.drop_missing(set())
    assert store.version == before + 1 and "a" not in store.snapshot()["nodes"]


# ── 6. token ที่ยืมให้ node โผล่ในผลงานสด ๆ ก่อนถูกกรอง ─────────────────────────
def test_borrowed_secrets_never_show_in_the_live_job_output(registered, monkeypatch):
    """_scrub_secrets กรอง *หลัง* ท่อปิด — ระหว่าง download 20 นาที หน้าเว็บ poll ทุกวิและได้บรรทัด
    ที่มี token เต็ม ๆ (curl -v / สคริปต์ที่ echo env) ไปแสดงและถูกเก็บในประวัติแท็บ
    """
    from lmds.web import jobs

    release = threading.Event()
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda *a, **k: _BlockingStream(b"Authorization: Bearer hf_SECRET123\n", release))
    job = jobs.start_remote("spark2", "demo", "repair", "lmds repair demo", {"HF_TOKEN": "hf_SECRET123"})
    try:
        for _ in range(100):
            if job.payload()["output"]:
                break
            time.sleep(0.01)
        live = job.payload()["output"]
        assert live and "hf_SECRET123" not in live, live
    finally:
        release.set()
    for _ in range(100):
        if not job.running:
            break
        time.sleep(0.02)
    assert "hf_SECRET123" not in job.payload()["output"]


# ── 7. input ผิดชนิดต้องเป็น 400 ไม่ใช่ 500 ───────────────────────────────────
def test_generate_with_a_non_numeric_context_is_400(tmp_path, monkeypatch):
    from tests.test_generator import safetensors_report
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True)
    r = TestClient(create_app(), raise_server_exceptions=False).post(
        f"/api/deploy/{analyzed['id']}/generate", json={"context": "abc", "output": str(tmp_path)})
    assert r.status_code == 422 and r.json()["detail"]["kind"] == "input"
    assert "context" in r.json()["detail"]["message"]


def test_adding_a_node_with_a_bad_port_is_400(monkeypatch):
    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: "k")
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post("/api/nodes", json={"host": "10.0.0.9", "user": "ops", "port": "ssh"})
    assert r.status_code == 400 and "port" in r.json()["detail"]
    r = client.post("/api/nodes", json={"host": "10.0.0.9", "user": "ops", "port": 70000})
    assert r.status_code == 400


# ── 8. push จากหน้าเว็บส่ง zip เก่า — ค่าที่ตั้งผ่านหน้าเว็บไม่ถึงเครื่องปลายทาง ────────
def test_web_push_repacks_the_bundle_so_saved_settings_travel(registered, monkeypatch):
    """CLI `node push` แพ็กใหม่ตั้งแต่ 2026-09-03 แต่ปุ่ม push บนหน้าเว็บยังส่ง zip ที่แพ็กตอน generate
    → bundle.env / bundle.args ที่ตั้งจากหน้าเว็บไม่เคยไปถึง node แล้ว start ก็ "สำเร็จ" ด้วยค่า default
    """
    from lmds.packager.bundle import make_zip

    bundle = Path.home() / "bundles" / "demo"
    bundle.mkdir(parents=True)
    (bundle / "demo-single.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    make_zip(bundle)
    (bundle / "bundle.env").write_text('API_PORT="${API_PORT:-8001}"\n', encoding="utf-8")

    sent: dict = {}

    def fake_push(node, local, remote, timeout=1800):
        sent["names"] = zipfile.ZipFile(local).namelist()
        return SimpleNamespace(ok=True, stderr="", stdout="")

    monkeypatch.setattr("lmds.nodes.push_file", fake_push)
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, cmd, timeout=60, **k: SimpleNamespace(ok=True, stdout="~/bundles/demo\n", stderr=""))
    r = TestClient(create_app()).post("/api/models/demo/push/spark2")
    assert r.status_code == 200, r.text
    assert "demo/bundle.env" in sent["names"]


# ── 9. สาย ssh ขาดกลางงาน: ต้องบอกว่างานบนเครื่องนั้นอาจยังรันอยู่ ────────────────
def test_lost_ssh_link_explains_that_the_remote_job_may_still_be_running():
    from lmds.web import jobs

    hint = jobs.explain_failure("", exit_code=255, node="spark2", slug="demo")
    assert "spark2" in hint and "ยังรัน" in hint and "logs" in hint
    assert jobs.explain_failure("all good", exit_code=0, node="spark2", slug="demo") == ""
