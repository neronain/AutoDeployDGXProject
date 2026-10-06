"""audit 2026-10-06 — `lmds doctor` ขึ้น ✅ ให้สองเรื่องที่ไม่จริง (และหนึ่งเรื่องที่ไม่ได้ตรวจ)

1. **weights** — bundle safetensors: รายการไฟล์บังคับว่าง และเมื่อไม่มี snapshot ของ revision ที่ pin
   ไว้ก็ "ยอมรับ snapshot ตัวไหนก็ได้" · ผล: snapshot เดียวที่มีเป็นของ revision อื่น มีแค่
   `config.json` กับ blob `.incomplete` ค้าง ทั้งที่ profile บอก 21 GB → "✅ weights …
   ไม่พบปัญหาที่บล็อกการรัน" exit 0 · controller ของ bundle เดียวกัน (`verify-files`) ปฏิเสธทันที
2. **endpoint** — ตอบ ✅ "มี API key เก็บไว้" เพราะ *มีไฟล์ key* โดยไม่ดูเซิร์ฟเวอร์ที่รันอยู่ ·
   หลัง `lmds key new` บนเซิร์ฟเวอร์ที่เปิดโล่งอยู่ มันยังเปิดโล่งจนกว่าจะมีคน restart
3. **port** — รายงาน "ว่าง" เมื่อทั้ง `ss` และ `netstat` ใช้ไม่ได้ (คือไม่ได้ตรวจเลย)

ทั้งสามคือรูปเดียวกัน: *เขียวเพราะไม่ได้ดู ไม่ใช่เขียวเพราะดี*

เทสรันของจริง: bundle render จาก template จริง · แคช HF เลย์เอาต์จริง (snapshot เป็น symlink ไป
blobs) · เซิร์ฟเวอร์ HTTP จริงที่บังคับ/ไม่บังคับ key · คำสั่ง `lmds doctor` ผ่าน CLI
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.doctor import Status, diagnose
from lmds.fleet import apikey
from lmds.fleet.manager import register_bundle
from lmds.inspector.report import ShardFile
from tests.test_generator import make_bundle, safetensors_report

runner = CliRunner()
REVISION = "sha-pinned-123"            # ค่าที่ safetensors_report() pin ไว้
REPO = "Qwen/Qwen3-32B"
SLUG = "qwen3-32b"
# ขนาดจริงระดับ GB (ตัววางแผน fit ปฏิเสธโมเดลขนาดไม่กี่ KB) — ไฟล์บนดิสก์เป็น sparse จึงไม่กินที่จริง
SHARDS = {f"model-0000{i}-of-00003.safetensors": 7 * 2**30 + i for i in (1, 2, 3)}


@pytest.fixture(autouse=True)
def a_machine_that_serves_models(tmp_path, monkeypatch):
    """เครื่องที่รันโมเดลได้ (ไม่งั้น doctor ลดข้อ weights เป็นคำเตือนของ control plane) และไม่แตะ docker จริง"""
    from lmds.hardware import serving

    monkeypatch.setenv(serving.ROLE_ENV, "serving")
    serving.reset_cache()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks.shutil.which", lambda name: "/usr/bin/" + name)
    import shutil as _shutil

    monkeypatch.setattr("lmds.doctor.checks.shutil.disk_usage",
                        lambda path: _shutil._ntuple_diskusage(2_000 * 1024**3, 1_000 * 1024**3, 1_000 * 1024**3))
    yield
    serving.reset_cache()


def _bundle(tmp_path: Path, *, shards: bool = True, target: str = "dgx-spark-single",
            weight_bytes: int | None = None) -> Path:
    """bundle จริงจาก template จริง ลงทะเบียนแบบเดียวกับ `lmds deploy` — คืน path ของ controller"""
    overrides: dict = {"weight_bytes": weight_bytes or sum(SHARDS.values()), "shard_count": len(SHARDS)}
    if shards:
        overrides["safetensor_shards"] = [ShardFile(filename=n, size_bytes=s) for n, s in SHARDS.items()]
    bundle, _plan, _fit = make_bundle(safetensors_report(**overrides), target=target,
                                      tmp_path=tmp_path / "bundles")
    register_bundle(bundle.controller)
    return bundle.controller


def _snapshot(tmp_path: Path, revision: str, files: dict[str, int], *, layout: str = "hub") -> Path:
    """แคช HF เลย์เอาต์จริง: ไฟล์ใน snapshots/<rev>/ เป็น symlink ไปที่ blobs/<hash>"""
    base = tmp_path / "hf" / "hub" if layout == "hub" else tmp_path / "hf"
    repo = base / f"models--{REPO.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo / "blobs").mkdir(exist_ok=True)
    for name, size in files.items():
        blob = repo / "blobs" / hashlib.sha1(f"{revision}/{name}".encode()).hexdigest()
        with blob.open("wb") as handle:
            handle.truncate(size)              # sparse: ขนาดตามที่ Hub รายงาน โดยไม่เขียนข้อมูลจริง
        link = snapshot / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return snapshot


def _interrupted_download(tmp_path: Path, size: int = 1000) -> Path:
    blob = tmp_path / "hf" / "hub" / f"models--{REPO.replace('/', '--')}" / "blobs" / "deadbeef.incomplete"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"x" * size)
    return blob


SMALL = {"config.json": 40, "model.safetensors.index.json": 60, "tokenizer.json": 80}


def _finding(slug: str, name: str):
    return next(f for f in diagnose(slug).findings if f.name == name)


# ── 1. weights ───────────────────────────────────────────────────────────────────────
def test_another_revisions_half_downloaded_snapshot_is_not_a_green_tick(tmp_path):
    """เคสของ audit เป๊ะ: snapshot เดียวที่มีเป็นของ revision อื่น มี config.json + blob .incomplete

    ก่อนแก้: "✅ weights … ไม่พบปัญหาที่บล็อกการรัน" exit 0 · controller ตัวเดียวกันตอบ
    "ยังไม่ได้ download" ทันทีที่สั่ง start
    """
    _bundle(tmp_path, shards=False, weight_bytes=20 * 2**30)
    _snapshot(tmp_path, "an-older-revision-0000", {"config.json": 40})
    _interrupted_download(tmp_path)

    result = runner.invoke(app, ["doctor", SLUG])

    said = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert "ไม่พบปัญหาที่บล็อกการรัน" not in said
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL
    assert REVISION in weights.detail, "ต้องบอกว่า revision ไหนที่ยังไม่มี"
    assert "an-older-revis" in weights.detail, "ต้องบอกว่าที่มีอยู่เป็นของ revision อื่น"
    assert ".incomplete" in weights.detail, "download ที่ค้างอยู่คือคำอธิบายที่ผู้ใช้ต้องการที่สุด"
    assert f"lmds repair {SLUG}" in weights.fix


def test_the_downloaded_badge_on_the_web_agrees_with_doctor(tmp_path):
    """หน้าเว็บใช้ตัวตรวจชุดเดียวกัน (inventory.weights_present) — ป้าย downloaded ต้องไม่ขึ้นเหมือนกัน"""
    from lmds.fleet import bundle_profile, find
    from lmds.inventory import weights_present

    controller = _bundle(tmp_path)
    _snapshot(tmp_path, "an-older-revision-0000", {**SMALL, **SHARDS})
    assert weights_present(find(SLUG), bundle_profile(str(controller))) is False

    _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    assert weights_present(find(SLUG), bundle_profile(str(controller))) is True


def test_a_complete_download_of_the_pinned_revision_passes(tmp_path):
    """ต้องไม่แก้เกิน: โหลดครบตาม shard list ที่ controller ถืออยู่ (ไฟล์เป็น symlink ไป blobs) = ✅"""
    _bundle(tmp_path)
    snapshot = _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.OK, weights
    assert str(snapshot) in weights.detail and "3" in weights.detail


def test_a_missing_shard_is_named(tmp_path):
    _bundle(tmp_path)
    present = dict(list(SHARDS.items())[:2])
    _snapshot(tmp_path, REVISION, {**SMALL, **present})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL
    assert "model-00003-of-00003.safetensors" in weights.detail and "1" in weights.detail
    assert f"lmds repair {SLUG}" in weights.fix


def test_a_truncated_shard_is_caught_by_its_size(tmp_path):
    """download ที่ขาดกลางไฟล์ — ไฟล์ *มีอยู่* แต่สั้นกว่าที่ Hub รายงาน (controller เก็บขนาดไว้ใน SHARD_SIZES)"""
    _bundle(tmp_path)
    broken = dict(SHARDS)
    broken["model-00002-of-00003.safetensors"] = 100
    _snapshot(tmp_path, REVISION, {**SMALL, **broken})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL
    assert "model-00002-of-00003.safetensors" in weights.detail
    assert "ได้ 100 " in weights.detail
    assert f"{SHARDS['model-00002-of-00003.safetensors']:,}" in weights.detail, "ต้องบอกขนาดที่ควรเป็นด้วย"


def test_a_dangling_symlink_counts_as_a_missing_shard(tmp_path):
    """blob ถูกลบ (prune/ดิสก์เต็ม) แต่ symlink ใน snapshot ยังอยู่ — `ls` ยังเห็นชื่อไฟล์ครบ"""
    _bundle(tmp_path)
    snapshot = _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    (snapshot / "model-00001-of-00003.safetensors").resolve().unlink()
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL and "model-00001-of-00003.safetensors" in weights.detail


def test_without_a_shard_list_the_total_size_is_compared_to_the_profile(tmp_path):
    """bundle ที่ Hub ไม่ได้ให้รายการ shard มา — เหลือ `weight_bytes` ใน MODEL_PROFILE ให้เทียบ"""
    _bundle(tmp_path, shards=False, weight_bytes=sum(SHARDS.values()))
    half = dict(list(SHARDS.items())[:1])
    _snapshot(tmp_path, REVISION, {**SMALL, **half})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL, weights
    assert "ไม่ครบ" in weights.detail

    _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    assert _finding(SLUG, "weights").status is Status.OK


def test_a_snapshot_with_no_weight_files_at_all_is_not_downloaded(tmp_path):
    """snapshot ของ revision ที่ถูก แต่มีแค่ config — download ถูกขัดหลังไฟล์เล็กไฟล์แรก"""
    _bundle(tmp_path, shards=False, weight_bytes=20 * 2**30)
    _snapshot(tmp_path, REVISION, {"config.json": 40})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.FAIL


def test_leftover_incomplete_blobs_beside_a_complete_download_are_a_warning_not_a_blocker(tmp_path):
    """ไฟล์ครบตาม shard list แล้ว แต่มี .incomplete ค้าง — บอกให้รู้ (กินดิสก์) ไม่ใช่ห้าม start"""
    _bundle(tmp_path)
    _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    _interrupted_download(tmp_path, size=5000)
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.WARN, weights
    assert ".incomplete" in weights.detail
    assert not [f for f in diagnose(SLUG).failed if f.name == "weights"]


def test_the_legacy_cache_layout_is_still_found(tmp_path):
    """$HF_HOME/models--X (HF รุ่นเก่า) — เคส DeepSeek V4 บน spark-head ต้องไม่กลับมา"""
    _bundle(tmp_path)
    _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS}, layout="legacy")
    assert _finding(SLUG, "weights").status is Status.OK


def test_a_stacked_bundle_follows_its_own_controller_which_takes_any_snapshot(tmp_path):
    """controller แบบ stacked (`_snapshot_path`) ถอยไปใช้ snapshot ที่มีอยู่เมื่อไม่เจอ revision ที่ pin

    doctor ต้องตรวจ snapshot *ตัวที่ controller จะใช้จริง* — ครบก็ไม่ใช่ FAIL แต่ต้องบอกว่าเป็นคนละ
    revision · ไม่ครบก็ยัง FAIL เหมือนกัน
    """
    controller = _bundle(tmp_path, target="dgx-spark-stacked")
    assert yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text())["topology"] == "stacked"

    _snapshot(tmp_path, "an-older-revision-0000", {**SMALL, **SHARDS})
    weights = _finding(SLUG, "weights")
    assert weights.status is Status.WARN, weights
    assert "an-older-revis" in weights.detail and REVISION in weights.detail

    (tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-32B" / "snapshots" / "an-older-revision-0000"
     / "model-00003-of-00003.safetensors").unlink()
    assert _finding(SLUG, "weights").status is Status.FAIL


# ── 2. endpoint ──────────────────────────────────────────────────────────────────────
class _Server:
    """เซิร์ฟเวอร์โมเดลปลอม — `engine` กำหนดว่า endpoint ไหนอยู่หลัง key เหมือนของจริง

    vllm:     ทุก path ใต้ /v1 อยู่หลัง key (รวม /v1/models) · ไม่มี /props
    llamacpp: /v1/models และ /health **เปิดเสมอ** แม้ตั้ง --api-key · /props อยู่หลัง key
    """

    def __init__(self, key: str = "", engine: str = "vllm"):
        self.key, self.engine, self.paths = key, engine, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, status: int) -> None:
                body = json.dumps({"data": [{"id": SLUG}]}).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                outer.paths.append((self.path, bool(self.headers.get("authorization"))))
                authorised = not outer.key or self.headers.get("authorization") == f"Bearer {outer.key}"
                if outer.engine == "vllm":
                    if self.path.startswith("/v1") and not authorised:
                        return self._answer(401)
                    return self._answer(200 if self.path.startswith(("/v1/models", "/health")) else 404)
                if self.path in ("/v1/models", "/models", "/health"):
                    return self._answer(200)
                return self._answer(200 if authorised else 401)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def model(tmp_path):
    """bundle จริง (controller อ่านที่เก็บ key เอง) + weight ครบ"""
    controller = _bundle(tmp_path)
    _snapshot(tmp_path, REVISION, {**SMALL, **SHARDS})
    return controller


def _running_on(monkeypatch, port: int) -> None:
    import lmds.doctor.checks as checks
    import lmds.fleet.manager as manager

    real = manager.find

    def running(slug):
        found = real(slug)
        if found is not None:
            found.running, found.healthy, found.port = True, True, port
        return found

    monkeypatch.setattr(checks, "find", running)
    monkeypatch.setattr(checks, "_listening_on", lambda p: f"LISTEN 0 4096 0.0.0.0:{p} ")


def test_a_key_set_after_start_does_not_make_an_open_server_green(model, monkeypatch):
    """`lmds key new` บนเซิร์ฟเวอร์ที่รันแบบเปิดอยู่ — ไฟล์ key มีแล้ว แต่เซิร์ฟเวอร์ยังตอบทุกคนจนกว่าจะ restart

    ก่อนแก้: "✅ endpoint │ ผูกกับ 0.0.0.0 และมี API key เก็บไว้" ขณะที่ GET /v1/models ไม่มี key ได้ 200
    """
    apikey.write(SLUG, apikey.mint())
    server = _Server(key="")                      # start ก่อนมี key = ไม่บังคับ
    try:
        _running_on(monkeypatch, server.port)
        endpoint = _finding(SLUG, "endpoint")
    finally:
        server.close()

    assert endpoint.status is Status.WARN, endpoint
    assert "ยังไม่บังคับ" in endpoint.detail
    assert f"lmds restart {SLUG}" in endpoint.fix
    assert all(not sent_key for _path, sent_key in server.paths), "ต้องยิงแบบไม่ส่ง key — ไม่งั้นไม่ได้พิสูจน์อะไร"


@pytest.mark.parametrize("engine", ["vllm", "llamacpp"])
def test_a_server_that_refuses_keyless_requests_is_the_only_green_tick(model, monkeypatch, engine):
    """เขียวได้ทางเดียว: เห็นเซิร์ฟเวอร์ปฏิเสธคำขอที่ไม่มี key ด้วยตาตัวเอง

    llama.cpp คือกับดัก: `/v1/models` เปิดสาธารณะเสมอแม้ตั้ง --api-key — ดู 200 จาก path นั้นแล้วสรุปว่า
    "เปิดโล่ง" จะกล่าวหาเซิร์ฟเวอร์ที่ป้องกันถูกต้องแล้วทุกตัว
    """
    key = apikey.mint()
    apikey.write(SLUG, key)
    server = _Server(key=key, engine=engine)
    try:
        _running_on(monkeypatch, server.port)
        endpoint = _finding(SLUG, "endpoint")
    finally:
        server.close()
    assert endpoint.status is Status.OK, endpoint
    assert "401" in endpoint.detail
    assert key not in endpoint.detail + endpoint.fix


@pytest.mark.parametrize("engine", ["vllm", "llamacpp"])
def test_an_open_server_without_any_key_keeps_its_warning(model, monkeypatch, engine):
    server = _Server(key="", engine=engine)
    try:
        _running_on(monkeypatch, server.port)
        endpoint = _finding(SLUG, "endpoint")
    finally:
        server.close()
    assert endpoint.status is Status.WARN
    assert "ไม่มี API key" in endpoint.detail and f"lmds key new {SLUG}" in endpoint.fix


def test_a_stopped_model_is_never_reported_as_protected(model):
    """ไม่ได้รัน = ไม่มีเซิร์ฟเวอร์ให้ถาม — บอกสิ่งที่รู้จากไฟล์ แต่ห้ามขึ้น ✅ ให้ข้ออ้างที่ไม่ได้ตรวจ"""
    apikey.write(SLUG, apikey.mint())
    endpoint = _finding(SLUG, "endpoint")
    assert endpoint.status is not Status.OK, endpoint
    assert "ยังไม่ได้รัน" in endpoint.detail and "มี API key เก็บไว้" in endpoint.detail
    assert not [f for f in diagnose(SLUG).failed if f.name == "endpoint"], "ไม่ใช่ตัวบล็อกการรัน"


def test_a_running_server_that_cannot_be_reached_is_unverified_not_green(model, monkeypatch):
    """ทะเบียนบอกว่ารัน แต่ยิงไม่ติด (กำลังโหลด/พอร์ตเพี้ยน) — ตรวจไม่ได้ ≠ ป้องกันแล้ว"""
    import socket

    apikey.write(SLUG, apikey.mint())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    _running_on(monkeypatch, dead_port)
    endpoint = _finding(SLUG, "endpoint")
    assert endpoint.status is Status.WARN and "ตรวจไม่ได้" in endpoint.detail


def test_the_probe_goes_to_the_address_the_bundle_binds(model, monkeypatch):
    """0.0.0.0/ว่าง → loopback · IP เจาะจง → IP นั้น (เซิร์ฟเวอร์ที่ผูก IP เดียวไม่ฟังที่ 127.0.0.1)"""
    import lmds.doctor.checks as checks

    asked = []
    monkeypatch.setattr(checks, "_answers_without_key",
                        lambda host, port: (asked.append(host), ("enforced", "GET /v1/models → 401"))[1])
    apikey.write(SLUG, apikey.mint())
    _running_on(monkeypatch, 8000)

    diagnose(SLUG)
    (model.parent / "bundle.env").write_text('API_HOST="${API_HOST:-10.20.30.40}"\n', encoding="utf-8")
    diagnose(SLUG)
    (model.parent / "bundle.env").write_text('API_HOST="${API_HOST:-::}"\n', encoding="utf-8")
    diagnose(SLUG)
    assert asked == ["127.0.0.1", "10.20.30.40", "[::1]"]


# ── 3. port ──────────────────────────────────────────────────────────────────────────
def test_a_port_nobody_could_check_is_not_reported_free(model, monkeypatch):
    """ไม่มีทั้ง `ss` และ `netstat` (container ตัดเครื่องมือ / macOS) — เดิมตอบ "8000 ว่าง" ทั้งที่ไม่ได้ตรวจ"""
    monkeypatch.setattr("lmds.doctor.checks.shutil.which", lambda name: None)
    port = _finding(SLUG, "port")
    assert port.status is Status.WARN, port
    assert "ตรวจไม่ได้" in port.detail and "ว่าง" not in port.detail.replace("ว่างไหม", "")


def test_a_tool_that_exists_but_fails_is_also_not_a_free_port(model, monkeypatch):
    """macOS มี `netstat` แต่ไม่รู้จัก `-tlnp` → exit ไม่เป็นศูนย์ = ยังไม่ได้ตรวจ"""
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (1, "netstat: illegal option"))
    assert "ตรวจไม่ได้" in _finding(SLUG, "port").detail


def test_a_port_that_was_checked_and_is_free_still_says_so(model, monkeypatch):
    monkeypatch.setattr("lmds.doctor.checks._run",
                        lambda args, timeout=10: (0, "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n"))
    port = _finding(SLUG, "port")
    assert port.status is Status.OK and "ว่าง" in port.detail
