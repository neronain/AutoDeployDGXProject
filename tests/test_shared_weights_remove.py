"""`lmds remove` ต้องไม่ลบ weight ที่ bundle อื่นยังเสิร์ฟอยู่ — audit 2026-10-06 (ข้อมูลหาย)

weight ถูกหาจาก **ชื่อโมเดล** (`$HF_HOME/hub/models--org--name`) ไม่ใช่จากชื่อ bundle · โมเดลเดียวกันที่ deploy
สองใบจึงชี้โฟลเดอร์เดียวกันเสมอ และไม่มีใครเช็คก่อนลบ:

  * `lmds deploy Qwen/Qwen3.6-35B-A3B --name qwen36-test` ข้าง `qwen36-prod` ที่รันอยู่ → `lmds remove qwen36-test`
    พิมพ์ "ลบ weight ของโมเดล: …/models--Qwen--Qwen3.6-35B-A3B" แล้ว prod รันต่อโดยไม่มี weight บนดิสก์
  * `lmds deploy … --also-stacked` สร้าง `<slug>` + `<slug>-stacked` บน weight ก้อนเดียว (ทางที่เจอบ่อยที่สุด)
  * stacked: weight ที่ sync-worker คัดลอกไป worker ก็เป็นโฟลเดอร์เดียวกันของ stacked ใบอื่น / ของ bundle ที่ worker มีเอง

ทุกเทสในไฟล์นี้วาง bundle จริงบนดิสก์ (controller + MODEL_PROFILE.yaml + ทะเบียนจาก `register_bundle`) แล้วสั่งผ่าน
ทางที่ผู้ใช้ใช้จริง — `manager.remove_server`, `lmds remove` (CliRunner), `/api/models/<slug>/removal-plan|remove`
— และยืนยันจาก **ดิสก์** ว่าไฟล์อยู่/หาย ไม่ใช่จากข้อความที่พิมพ์
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.fleet import manager
from tests.test_audit_stacked_controller import _shim
from tests.test_stacked_remove import _DOCKER, _SSH

SAFE_PATH = "/usr/bin:/bin"
QWEN = "Qwen/Qwen3.6-35B-A3B"
QWEN_CACHE = "models--Qwen--Qwen3.6-35B-A3B"

# docker ปลอมของ head: ไม่มี container ไหนรันอยู่ (`container inspect` = ไม่เจอ) · `ps` ว่าง — `find()` ถาม docker จริง
# บนเครื่อง dev ไม่ได้ ไม่งั้นผลเทสขึ้นกับว่าเครื่องนั้นรันอะไรอยู่
_HEAD_DOCKER = '''
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  container) exit 1 ;;
  ps) exit 0 ;;
  images) echo "alpine:3.20"; exit 0 ;;
  *) exit 0 ;;
esac
'''


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """เครื่องเปล่าหนึ่งเครื่อง: HOME · HF cache · ทะเบียน · PATH ที่มีแต่ docker ปลอม"""
    home = tmp_path / "home"
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    (home / "bundles").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LMDS_BUNDLE_DIRS", raising=False)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(home / ".lmds" / "run"))
    monkeypatch.chdir(home)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _shim(bin_dir, "docker", _HEAD_DOCKER)
    monkeypatch.setenv("PATH", f"{bin_dir}:{SAFE_PATH}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    monkeypatch.setattr(manager, "_pgrep_llama", lambda: [])
    return home


def _hf_cache(home: Path, cache: str = QWEN_CACHE, size: int = 4096) -> Path:
    repo = home / ".cache" / "huggingface" / "hub" / cache
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "blobs" / "weights.safetensors").write_bytes(b"x" * size)
    return repo


def _bundle(home: Path, slug: str, model_id: str = QWEN, engine: str = "vllm", register: bool = True,
            suffix: str = "single", extra_profile: str = "", running: bool = False) -> Path:
    """bundle แบบที่ `lmds deploy` วางไว้: โฟลเดอร์ + controller + profile (+ ทะเบียนจาก register_bundle จริง)"""
    directory = home / "bundles" / slug
    directory.mkdir(parents=True)
    controller = directory / f"{slug}-{suffix}.sh"
    controller.write_text(f'#!/bin/bash\necho "ctl[{slug}] $*" >> "$FAKE_LOG"\n', encoding="utf-8")
    controller.chmod(0o755)
    (directory / "MODEL_PROFILE.yaml").write_text(
        f"generated_by: lmds 0.9.0\nruntime:\n  engine: {engine}\nmodel:\n  id: {model_id}\n  served_name: {slug}\n"
        + extra_profile, encoding="utf-8")
    if register:
        meta = manager.register_bundle(controller)
        if running:
            # "กำลังเสิร์ฟ": โหมด native ที่ pid file ชี้ process ที่มีชีวิตจริง (ตัวเทสเอง) — find() จะรายงาน running
            pid_file = meta.parent / "server.pid"
            pid_file.write_text(str(os.getpid()), encoding="utf-8")
            meta.write_text(meta.read_text(encoding="utf-8").replace("mode=docker", "mode=native")
                            .replace("pid_file=\n", f"pid_file={pid_file}\n")
                            .replace("started_at=\n", "started_at=2026-10-01T09:00:00\n"), encoding="utf-8")
    return directory


def _weights_items(info) -> list:
    return [item for item in manager.removal_plan(info) if item.is_weights and not item.node]


# ═════════════════════ เครื่องเดียว: HF cache ก้อนเดียว สอง bundle ═════════════════════
def test_removing_the_idle_copy_keeps_the_weights_the_running_bundle_serves_from(home):
    """รีโปรของ auditor ตรงตัว: qwen36-prod (รัน) + qwen36-test (ว่าง) บน Qwen3.6-35B-A3B ก้อนเดียว"""
    cache = _hf_cache(home)
    _bundle(home, "qwen36-prod", running=True)
    test_dir = _bundle(home, "qwen36-test")
    prod = manager.find("qwen36-prod")
    assert prod.running and manager.weights_path(prod) == cache

    test = manager.find("qwen36-test")
    [item] = _weights_items(test)
    assert item.path == cache and item.kept and item.shared_with == ["qwen36-prod"]

    lines = manager.remove_server(test)
    assert not manager.removal_failed(lines), lines
    # ดิสก์คือหลักฐาน: weight ของ prod อยู่ครบ · bundle/ทะเบียนของตัวที่ลบหายจริง
    assert (cache / "blobs" / "weights.safetensors").stat().st_size == 4096
    assert not test_dir.exists() and not (home / ".lmds/run/qwen36-test").exists()
    assert manager.find("qwen36-prod").running and manager.weights_path(manager.find("qwen36-prod")) == cache
    kept = [line for line in lines if line.startswith("เก็บ ")]
    assert len(kept) == 1 and "qwen36-prod" in kept[0] and str(cache) in kept[0]
    assert not any(line.startswith("ลบ weight") for line in lines)


def test_the_last_bundle_using_the_weights_removes_them_as_before(home):
    cache = _hf_cache(home)
    _bundle(home, "qwen36-prod")
    _bundle(home, "qwen36-test")
    manager.remove_server(manager.find("qwen36-test"))
    assert cache.is_dir()

    last = manager.find("qwen36-prod")
    [item] = _weights_items(last)
    assert not item.kept and item.shared_with == []
    lines = manager.remove_server(last)
    assert not manager.removal_failed(lines), lines
    assert not cache.exists()
    assert f"ลบ weight ของโมเดล: {cache}" in lines


def test_a_bundle_of_another_model_never_blocks_removal(home):
    cache = _hf_cache(home)
    other = _hf_cache(home, "models--google--gemma-4-31B-it")
    _bundle(home, "gemma4", model_id="google/gemma-4-31B-it")
    _bundle(home, "qwen36-test")
    lines = manager.remove_server(manager.find("qwen36-test"))
    assert not manager.removal_failed(lines), lines
    assert not cache.exists() and other.is_dir()


def test_a_bundle_on_disk_without_a_registration_still_counts_as_a_user(home):
    """bundle ที่ copy มา/สร้างก่อนมีทะเบียน (ไม่มี server.meta) ก็ใช้ weight ก้อนนั้นเหมือนกัน"""
    cache = _hf_cache(home)
    _bundle(home, "qwen36-copied", register=False)
    _bundle(home, "qwen36-test")
    [item] = _weights_items(manager.find("qwen36-test"))
    assert item.shared_with == ["qwen36-copied"]
    manager.remove_server(manager.find("qwen36-test"))
    assert cache.is_dir()


def test_keep_weights_still_lists_nothing_about_weights(home):
    _hf_cache(home)
    _bundle(home, "qwen36-prod")
    _bundle(home, "qwen36-test")
    plan = manager.removal_plan(manager.find("qwen36-test"), include_weights=False)
    assert not [item for item in plan if item.is_weights]


def test_llamacpp_bundles_of_the_same_gguf_repo_keep_their_own_folders(home):
    """llama.cpp เก็บที่ ~/models/<slug> — สองใบของ repo เดียวกัน (Q4_K_M กับ Q8_0) ไม่ได้ใช้โฟลเดอร์ร่วมกัน
    จึงลบของใครของมันได้เต็ม ๆ และห้ามแตะของอีกใบ"""
    for slug, name in (("qwen3-8b-q4", "Qwen3-8B-Q4_K_M.gguf"), ("qwen3-8b-q8", "Qwen3-8B-Q8_0.gguf")):
        _bundle(home, slug, model_id="unsloth/Qwen3-8B-GGUF", engine="llamacpp")
        (home / "models" / slug).mkdir(parents=True)
        (home / "models" / slug / name).write_bytes(b"g" * 2048)
    info = manager.find("qwen3-8b-q4")
    [item] = _weights_items(info)
    assert item.path == home / "models" / "qwen3-8b-q4" and not item.kept
    lines = manager.remove_server(info)
    assert not manager.removal_failed(lines), lines
    assert not (home / "models" / "qwen3-8b-q4").exists()
    assert (home / "models" / "qwen3-8b-q8" / "Qwen3-8B-Q8_0.gguf").stat().st_size == 2048


# ═════════════════════ bundle ที่ `lmds adopt` รับมา — path จดจาก bind mount ═════════════════════
def _adopted(home: Path, slug: str, weights: Path, kind: str, model_id: str = "") -> Path:
    return _bundle(home, slug, model_id=model_id or f"adopted/{slug}", suffix="adopted",
                   extra_profile=f"weights:\n  path: {weights}\n  kind: {kind}\n  source: bind-mount\n")


def test_an_adopted_snapshot_inside_the_repo_folder_is_protected_in_both_directions(home):
    """adopt จด `…/models--org--m/snapshots/<sha>` (path ที่ container ใช้จริง) — อยู่ **ใน** โฟลเดอร์ที่ bundle ปกติ
    ของ repo เดียวกันจะลบทั้งก้อน · เท่ากันเป๊ะไม่พอ ต้องดูว่าทับกันไหม"""
    cache = _hf_cache(home)
    snapshot = cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    _adopted(home, "coder-next", snapshot, "dir")
    _bundle(home, "qwen36-test")

    [mine] = _weights_items(manager.find("qwen36-test"))
    assert mine.path == cache and mine.shared_with == ["coder-next"]
    [theirs] = _weights_items(manager.find("coder-next"))
    assert theirs.path == snapshot and theirs.shared_with == ["qwen36-test"]

    manager.remove_server(manager.find("qwen36-test"))
    assert (snapshot / "config.json").is_file() and (cache / "blobs" / "weights.safetensors").is_file()
    # เหลือ coder-next ใบเดียว — ลบได้ และลบเฉพาะ snapshot ที่มันจดไว้
    lines = manager.remove_server(manager.find("coder-next"))
    assert not manager.removal_failed(lines), lines
    assert not snapshot.exists() and (cache / "blobs").is_dir()


def test_adopted_gguf_bundles_on_different_files_of_one_folder_remove_only_their_own_file(home):
    """bundle A ใช้ Q4_K_M.gguf · bundle B ใช้ Q8_0.gguf ในโฟลเดอร์เดียวกัน — ลบ A ต้องไม่แตะไฟล์ของ B"""
    folder = home / "ggufs" / "Qwen3-8B-GGUF"
    folder.mkdir(parents=True)
    q4, q8 = folder / "Qwen3-8B-Q4_K_M.gguf", folder / "Qwen3-8B-Q8_0.gguf"
    q4.write_bytes(b"4" * 1000)
    q8.write_bytes(b"8" * 3000)
    _adopted(home, "llama-q4", q4, "file")
    _adopted(home, "llama-q8", q8, "file")

    [item] = _weights_items(manager.find("llama-q4"))
    assert item.path == q4 and not item.kept and item.size_bytes == 1000
    lines = manager.remove_server(manager.find("llama-q4"))
    assert not manager.removal_failed(lines), lines
    assert not q4.exists() and q8.stat().st_size == 3000


def test_adopted_bundles_recorded_at_the_same_repo_folder_keep_the_whole_folder(home):
    """สองใบจดไว้ที่ระดับโฟลเดอร์ repo เดียวกัน (ใช้คนละไฟล์ข้างใน) — แยกไม่ออกว่าไฟล์ไหนของใคร = เก็บทั้งโฟลเดอร์ และบอก"""
    repo = home / ".cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF"
    repo.mkdir(parents=True)
    (repo / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"4" * 1000)
    (repo / "Qwen3-8B-Q8_0.gguf").write_bytes(b"8" * 3000)
    _adopted(home, "llama-q4", repo, "hf-cache", model_id="unsloth/Qwen3-8B-GGUF")
    _adopted(home, "llama-q8", repo, "hf-cache", model_id="unsloth/Qwen3-8B-GGUF")

    lines = manager.remove_server(manager.find("llama-q4"))
    assert not manager.removal_failed(lines), lines
    assert sorted(p.name for p in repo.iterdir()) == ["Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q8_0.gguf"]
    [kept] = [line for line in lines if line.startswith("เก็บ ")]
    assert "ทั้งก้อน" in kept and "llama-q8" in kept and str(repo) in kept


# ═════════════════════ CLI — ตาราง dry-run / ก่อนยืนยัน ต้องบอกว่าใครใช้ร่วม ═════════════════════
def test_cli_dry_run_names_the_sharing_bundle_and_does_not_count_the_weights_as_deleted(home):
    from lmds.cli.main import app

    cache = _hf_cache(home, size=5 * 1024 * 1024)
    _bundle(home, "qwen36-prod", running=True)
    _bundle(home, "qwen36-test")
    result = CliRunner().invoke(app, ["remove", "qwen36-test", "--dry-run"], env={"COLUMNS": "260"})
    assert result.exit_code == 0, result.output
    delete_part, _, keep_part = result.output.partition("เก็บไว้ ไม่ลบ")
    assert keep_part, result.output
    # ตาราง "เก็บไว้": path ของ weight + ชื่อ bundle ที่ยังใช้ · ตาราง "จะลบ": ไม่มี weight และยอดรวมไม่นับ 5 MB นั้น
    assert QWEN_CACHE in keep_part and "qwen36-prod" in keep_part
    assert QWEN_CACHE not in delete_part
    total = next(line for line in delete_part.splitlines() if line.startswith("รวม "))
    assert "MB" not in total and "GB" not in total, total
    assert cache.is_dir()


def test_cli_remove_keeps_the_shared_weights_says_so_and_still_succeeds(home):
    from lmds.cli.main import app

    cache = _hf_cache(home)
    _bundle(home, "qwen36-prod", running=True)
    test_dir = _bundle(home, "qwen36-test")
    result = CliRunner().invoke(app, ["remove", "qwen36-test", "-y"], env={"COLUMNS": "260"})
    assert result.exit_code == 0, result.output
    assert cache.is_dir() and not test_dir.exists()
    said = [line for line in result.output.splitlines() if line.strip().startswith("เก็บ weight")]
    assert said and "qwen36-prod" in said[0]
    assert "ลบ qwen36-test เรียบร้อย" in result.output
    # prod ไม่ถูกแตะ — ไม่มีใครเรียก controller ของมัน (stop) เลย
    calls = Path(os.environ["FAKE_LOG"]).read_text(encoding="utf-8") if Path(os.environ["FAKE_LOG"]).exists() else ""
    assert "ctl[qwen36-prod]" not in calls


def test_deploy_also_stacked_pair_shares_one_weight_folder_and_removing_one_keeps_it(home, monkeypatch):
    """ทางที่เจอบ่อยที่สุด: `lmds deploy … --also-stacked` = `<slug>` + `<slug>-stacked` บน weight ก้อนเดียว
    เดิม `lmds remove qwen3-32b-stacked -y` ลบ weight ของ `qwen3-32b` ที่ยังรันอยู่"""
    from lmds.cli.main import app
    from tests.test_generator import safetensors_report

    report = safetensors_report(weight_bytes=20 * 2**30)
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)
    runner = CliRunner()
    made = runner.invoke(app, ["deploy", report.repo_id, "--no-llm", "--target", "dgx-spark-single", "--yes",
                               "--output", str(home / "bundles"), "--also-stacked"])
    assert made.exit_code == 0, made.output
    assert (home / "bundles/qwen3-32b").is_dir() and (home / "bundles/qwen3-32b-stacked").is_dir()
    cache = _hf_cache(home, "models--Qwen--Qwen3-32B", size=8192)

    plan = runner.invoke(app, ["remove", "qwen3-32b-stacked", "--dry-run"], env={"COLUMNS": "260"})
    assert plan.exit_code == 0, plan.output
    keep_part = plan.output.partition("เก็บไว้ ไม่ลบ")[2]
    assert "models--Qwen--Qwen3-32B" in keep_part and "qwen3-32b" in keep_part, plan.output

    runner.invoke(app, ["remove", "qwen3-32b-stacked", "-y"], env={"COLUMNS": "260"})
    assert (cache / "blobs" / "weights.safetensors").stat().st_size == 8192
    assert not (home / "bundles/qwen3-32b-stacked").exists() and (home / "bundles/qwen3-32b").is_dir()
    # และกลับกัน: ใบ single ที่เหลืออยู่ใบเดียวลบ weight ได้ตามปกติ
    last = runner.invoke(app, ["remove", "qwen3-32b", "-y"], env={"COLUMNS": "260"})
    assert last.exit_code == 0, last.output
    assert not cache.exists()


# ═════════════════════ หน้าเว็บ — กล่องยืนยันและปุ่มลบ ═════════════════════
def test_web_removal_preview_marks_shared_weights_as_kept_and_remove_leaves_them(home):
    from fastapi.testclient import TestClient

    from lmds.web.api import create_app

    cache = _hf_cache(home, size=8192)
    _bundle(home, "qwen36-prod", running=True)
    test_dir = _bundle(home, "qwen36-test")
    client = TestClient(create_app())

    data = client.get("/api/models/qwen36-test/removal-plan").json()
    [weights] = [item for item in data["items"] if item["is_weights"]]
    assert weights["kept"] is True and weights["shared_with"] == ["qwen36-prod"] and weights["path"] == str(cache)
    assert all(item["kept"] is False for item in data["items"] if not item["is_weights"])
    # ยอด "Deletes N GB" = ของที่จะหายจริง ไม่รวม weight ที่เก็บไว้
    assert data["total_bytes"] == sum(item["bytes"] for item in data["items"] if not item["kept"])
    assert data["total_bytes"] < 8192

    done = client.post("/api/models/qwen36-test/remove", json={"keep_weights": False}).json()
    assert done["failed"] == [], done
    assert any("qwen36-prod" in line and line.startswith("เก็บ ") for line in done["done"]), done
    assert (cache / "blobs" / "weights.safetensors").is_file() and not test_dir.exists()

    # ตัวสุดท้าย: preview บอกว่าจะลบ และลบจริง
    last = client.get("/api/models/qwen36-prod/removal-plan").json()
    [weights] = [item for item in last["items"] if item["is_weights"]]
    assert weights["kept"] is False and weights["shared_with"] == []


# ═════════════════════ stacked — weight บน worker ═════════════════════
BIG = "org/Big"
BIG_CACHE = "models--org--Big"
# ssh ปลอมของ test_stacked_remove + HOME ต่อ node — worker มีทะเบียน (~/.lmds/run) ของตัวเอง ไม่ใช่ของ head
_SSH_OWN_HOME = _SSH.replace('export FAKE_NODE="$node"',
                             'export FAKE_NODE="$node" HOME="${FAKE_REMOTE}/${node}/home"')


def _stacked(home: Path, tmp_path: Path, slug: str, workers: list[str]) -> None:
    directory = _bundle(home, slug, model_id=BIG, register=False, suffix="stacked", extra_profile="topology: stacked\n")
    (directory / "cluster.env").write_text(
        "\n".join(["MASTER_IP=10.1.1.1", f"WORKER_IP={workers[0]}", f'WORKER_IPS="{" ".join(workers)}"',
                   f"NNODES={len(workers) + 1}", "SSH_USER=neronain", "WORKER_HF_HOME=/wk/hf",
                   "WORKER_FLASHINFER_CACHE=/wk/fi"]) + "\n", encoding="utf-8")
    for ip in workers:
        node = tmp_path / "remote" / ip
        (node / "wk/hf/hub" / BIG_CACHE / "blobs").mkdir(parents=True, exist_ok=True)
        (node / "wk/hf/hub" / BIG_CACHE / "blobs" / "shard-1").write_bytes(b"w" * 4000)
        (node / "wk/hf/hub/.locks" / BIG_CACHE).mkdir(parents=True, exist_ok=True)
        (node / "tmp" / f"lmds-{slug}").mkdir(parents=True)
        (node / "home").mkdir(exist_ok=True)
        (node / "container").write_text("running", encoding="utf-8")


@pytest.fixture
def cluster(home, tmp_path, monkeypatch) -> Path:
    bin_dir = tmp_path / "bin"
    _shim(bin_dir, "ssh", _SSH_OWN_HOME)
    _shim(bin_dir, "docker", _DOCKER)
    monkeypatch.setenv("FAKE_REMOTE", str(tmp_path / "remote"))
    return tmp_path / "remote"


def test_two_stacked_bundles_of_one_model_share_the_worker_weights_until_the_last_is_removed(home, cluster, tmp_path):
    """stacked สองใบของโมเดลเดียวกัน (เช่น `--name` เพื่อลอง context ยาวขึ้น) ชี้ WORKER_HF_HOME เดียวกันบน worker
    ตัวเดียวกัน — ลบใบหนึ่งแล้ว rm -rf บน worker = อีกใบต้อง sync 75–173 GB ใหม่"""
    _hf_cache(home, BIG_CACHE)
    _stacked(home, tmp_path, "big-a", ["10.1.1.2"])
    _stacked(home, tmp_path, "big-b", ["10.1.1.2"])
    worker = cluster / "10.1.1.2"

    info = manager.find("big-a")
    plan = manager.removal_plan(info)
    remote_weights = [item for item in plan if item.node and item.is_weights]
    assert remote_weights and all(item.shared_with == ["big-b"] for item in remote_weights)
    # ของ bundle นี้ใบเดียว (container / สคริปต์) ยังอยู่ในรายการลบตามเดิม
    assert [item.kind for item in plan if item.node and not item.kept] == ["container", "path"]

    lines = manager.remove_server(info)
    assert not manager.removal_failed(lines), lines
    assert (worker / "wk/hf/hub" / BIG_CACHE / "blobs/shard-1").stat().st_size == 4000
    assert (worker / "wk/hf/hub/.locks" / BIG_CACHE).is_dir()
    assert not (worker / "tmp/lmds-big-a").exists() and (worker / "tmp/lmds-big-b").is_dir()
    assert (home / ".cache/huggingface/hub" / BIG_CACHE).is_dir()             # ของ head ก็ใช้ร่วมกัน
    kept = [line for line in lines if line.startswith("เก็บ ") and "10.1.1.2:" in line]
    assert kept and all("big-b" in line for line in kept)

    last = manager.remove_server(manager.find("big-b"))
    assert not manager.removal_failed(last), last
    assert not (worker / "wk/hf/hub" / BIG_CACHE).exists() and not (worker / "wk/hf/hub/.locks" / BIG_CACHE).exists()
    assert not (home / ".cache/huggingface/hub" / BIG_CACHE).exists()


def test_a_stacked_bundle_on_other_workers_does_not_hold_this_workers_weights(home, cluster, tmp_path):
    """ใช้ร่วมกันต้องเป็น worker ตัวเดียวกัน — ใบอื่นที่ชี้ worker อีกเครื่องไม่ได้ถือของบนเครื่องนี้"""
    _stacked(home, tmp_path, "big-a", ["10.1.1.2"])
    _stacked(home, tmp_path, "big-b", ["10.1.1.3"])
    lines = manager.remove_server(manager.find("big-a"))
    assert not manager.removal_failed(lines), lines
    assert not (cluster / "10.1.1.2/wk/hf/hub" / BIG_CACHE).exists()
    assert (cluster / "10.1.1.3/wk/hf/hub" / BIG_CACHE / "blobs/shard-1").is_file()


def _worker_registration(cluster: Path, ip: str, slug: str, model_id: str, engine: str) -> None:
    run = cluster / ip / "home/.lmds/run" / slug
    run.mkdir(parents=True)
    (run / "server.meta").write_text(
        f"slug={slug}\nmodel={slug}\nmodel_id={model_id}\nengine={engine}\nmode=docker\nport=8000\n"
        f"container=lmds-{slug}\npid_file=\ncontroller=/home/u/bundles/{slug}/{slug}-single.sh\nstarted_at=\n",
        encoding="utf-8")


def test_a_workers_own_bundle_of_the_same_model_keeps_the_weights_on_that_worker(home, cluster, tmp_path):
    """worker มีใบ single ของโมเดลเดียวกันลงทะเบียนอยู่เอง (push ไปก่อนจะทำ stacked) — มันเสิร์ฟจาก HF cache โฟลเดอร์
    เดียวกับที่ sync-worker คัดลอกลงไป · head ต้องถาม worker ก่อนลบ และ worker อีกเครื่องที่ไม่มีก็ลบตามปกติ"""
    _stacked(home, tmp_path, "big-a", ["10.1.1.2", "10.1.1.3"])
    _worker_registration(cluster, "10.1.1.2", "big-single", BIG, "vllm")
    _worker_registration(cluster, "10.1.1.2", "big-a", BIG, "vllm")            # ทะเบียนของใบนี้เองบน worker — ไม่นับ
    _worker_registration(cluster, "10.1.1.3", "big-gguf", BIG, "llamacpp")     # llama.cpp ไม่ใช้ HF cache
    _worker_registration(cluster, "10.1.1.3", "other", "org/Other", "vllm")    # คนละโมเดล

    plan = manager.removal_plan(manager.find("big-a"))
    held = {item.node: item.shared_with for item in plan if item.is_weights and item.node and "เลย์เอาต์" not in item.label
            and "lock" not in item.label}
    assert held["10.1.1.3"] == []
    assert len(held["10.1.1.2"]) == 1 and held["10.1.1.2"][0].startswith("big-single") and "10.1.1.2" in held["10.1.1.2"][0]

    lines = manager.remove_server(manager.find("big-a"))
    assert not manager.removal_failed(lines), lines
    assert (cluster / "10.1.1.2/wk/hf/hub" / BIG_CACHE / "blobs/shard-1").is_file()
    assert not (cluster / "10.1.1.3/wk/hf/hub" / BIG_CACHE).exists()
    assert not (cluster / "10.1.1.2/container").exists() and not (cluster / "10.1.1.2/tmp/lmds-big-a").exists()


# ═════════════════════ หน้าเว็บ — JS จริงของกล่องยืนยัน (node) ═════════════════════
def test_the_web_confirmation_box_lists_kept_weights_apart_and_names_who_still_uses_them(tmp_path):
    """กด "Remove from machine" บนการ์ด — กล่องยืนยันต้องแยกของที่ "จะหาย" กับของที่ "เก็บไว้เพราะ bundle อื่นใช้"
    รัน handler จริงของ index.html กับ payload รูปเดียวกับที่ /api/models/<slug>/removal-plan ตอบ (เทสข้างบน)"""
    from tests.test_console_shell import FLEET, run_scenario

    cache = "/home/u/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B"
    prelude = FLEET + """
        fx.localModels = [{ slug: "qwen36-test", model: "qwen36-test", model_id: "Qwen/Qwen3.6-35B-A3B", running: false,
                            healthy: false, engine: "vllm", port: 8001, context: 32768, features: "text",
                            controller_exists: true, registered: true }];
        H.routes = [["/api/models/qwen36-test/removal-plan", () => ({ slug: "qwen36-test", total_bytes: 2 * 1024 ** 3, items: [
            { label: "bundle", path: "/home/u/bundles/qwen36-test", bytes: 2 * 1024 ** 3, is_weights: false, kept: false, shared_with: [] },
            { label: "weight", path: "%s", bytes: 70 * 1024 ** 3, is_weights: true, kept: true, shared_with: ["qwen36-prod"] }] })],
          ...H.defaultRoutes(fx)];""" % cache
    (out,) = run_scenario(tmp_path, prelude, """
        await H.tick(10);
        const acts = () => document.querySelectorAll('button[data-slug="qwen36-test"]').map(b => b.dataset.act);
        const opts = document.querySelector('button[data-act="opts"][data-slug="qwen36-test"]');
        H.assert(opts, "the local model card must have a manage panel: " + JSON.stringify(acts()));
        opts.click(); await H.tick(10);
        const button = document.querySelector('button[data-act="removeask"][data-slug="qwen36-test"]');
        H.assert(button, "the manage panel must offer Remove: " + JSON.stringify(acts()));
        button.click(); await H.tick(10);
        const rows = document.getElementById("rm-qwen36-test").querySelectorAll("div.mono").map(d => d.textContent);
        console.log(JSON.stringify({ rows, text: document.getElementById("rm-qwen36-test").textContent, errors: H.errors }));""")
    assert out["errors"] == []
    kept = [row for row in out["rows"] if cache in row]
    doomed = [row for row in out["rows"] if "/home/u/bundles/qwen36-test" in row]
    assert len(kept) == 1 and len(doomed) == 1, out["rows"]
    assert "Kept" in kept[0] and "qwen36-prod" in kept[0]
    assert "Kept" not in doomed[0] and "qwen36-prod" not in doomed[0]
    assert "Deletes 2.0 GB" in out["text"]
