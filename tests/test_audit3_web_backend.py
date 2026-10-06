"""audit 2026-10 ชั้นเว็บ (backend) — ทุกข้อในรายงานมีเทสที่ล้มก่อนแก้

เทสในไฟล์นี้ยิงผ่าน endpoint จริง (TestClient) ด้วย input ชนิดเดียวกับที่ผู้ตรวจใช้ แล้วยืนยันที่
**ผลที่เกิดขึ้นจริง** — ไฟล์ยังอยู่ไหม · คำสั่งถูกรันไหม · process ตายจริงไหม — ไม่ใช่ที่ข้อความในซอร์ส
· ที่ไหนมี sink (ฟังก์ชันที่ลงมือทำของอันตราย) เทส sink ตรง ๆ ด้วยอีกชั้น เพราะ endpoint ไม่ใช่ผู้เรียกคนเดียว

ลำดับตามรายงาน:
  1. POST /api/recipes/sync เชื่อ repo/ref จาก body → ลบ config dir · รันคำสั่งผ่าน option ของ git
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.config.paths import config_dir  # noqa: E402
from lmds.web import create_app  # noqa: E402
from tests.test_recipe_sync import VLLM_CONTROLLER  # noqa: E402
from tests.test_web import (  # noqa: E402,F401 — fixture ใช้ร่วมกัน (autouse ด้วย)
    fleet, fresh_jobs, no_host_scan,
)

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}


# ══ 1. recipes sync ═══════════════════════════════════════════════════════════

def _controller_repo(tmp_path: Path) -> str:
    """รีโป controller จริง (git จริง · ไม่แตะเน็ต) → URL แบบ file:// ที่ hub ตั้งเป็นต้นทางได้"""
    src = tmp_path / "ctlrepo"
    src.mkdir()
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True, env=env)
    (src / "qwen3-coder-next-single.sh").write_text(VLLM_CONTROLLER, encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "commit", "-qm", "init"], check=True, env=env)
    return src.as_uri()


def _configure_recipe_source(repo: str, ref: str = "") -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    (config_dir() / "config.yaml").write_text(
        yaml.safe_dump({"recipes": {"sync_repo": repo, "sync_ref": ref}}), encoding="utf-8")


def _precious_config() -> list[str]:
    """ของที่อยู่ใน config dir ของ hub จริง — ทะเบียนเครื่อง กุญแจ ความลับ token"""
    cfg = config_dir()
    cfg.mkdir(parents=True, exist_ok=True)
    for name in ("nodes.yaml", "id_lmds", "id_lmds.pub", "credentials", "web-token"):
        (cfg / name).write_text("precious\n", encoding="utf-8")
    (cfg / "sessions").mkdir()
    (cfg / "sessions" / "s1.json").write_text("{}", encoding="utf-8")
    return sorted(p.name for p in cfg.iterdir())


@pytest.fixture
def offline_default(monkeypatch):
    """รีโปตั้งต้นของทีมอยู่บน GitHub — เทสห้ามออกเน็ต ไม่ว่าโค้ดจะเลือกต้นทางผิดแค่ไหน"""
    from lmds.recipes import sync as sync_module

    monkeypatch.setattr(sync_module, "DEFAULT_REPO", "file:///nonexistent/lmds-test-default")
    from lmds.recipes import load_catalog

    load_catalog.cache_clear()
    yield
    load_catalog.cache_clear()


def test_a_repo_ending_in_dotdot_cannot_empty_the_config_directory(tmp_path, offline_default):
    """ผู้ตรวจ: `{"repo": "<path>/.."}` → checkout_dir = controllers/.. = config dir → rmtree ทั้งโฟลเดอร์

    POST เดียว ทะเบียนเครื่อง · กุญแจ SSH · credentials · web-token · sessions หายหมด (HTTP ตอบ 400 เฉย ๆ)
    """
    before = _precious_config()
    r = TestClient(create_app()).post("/api/recipes/sync", json={"repo": str(tmp_path / "nope") + "/.."})

    assert r.status_code == 400
    assert sorted(p.name for p in config_dir().iterdir()) == before
    assert (config_dir() / "sessions" / "s1.json").is_file()


def test_the_sync_endpoint_uses_the_configured_source_and_real_git_accepts_it(tmp_path, offline_default):
    """ต้นทางมาจาก config.yaml — `{}` (สิ่งที่หน้าเว็บส่ง) ดึงจากรีโปที่ hub ตั้งไว้ ทั้งรอบ clone และรอบ fetch

    รันกับ git จริง: `--` ที่เพิ่มเข้าไปในทุกคำสั่งต้องเป็นรูปที่ git รับ ไม่ใช่แค่ดูปลอดภัยบนกระดาษ
    """
    url = _controller_repo(tmp_path)
    _configure_recipe_source(url)
    client = TestClient(create_app())

    first = client.post("/api/recipes/sync", json={})           # clone
    assert first.status_code == 200, first.text
    assert first.json()["repo"] == url and first.json()["count"] == 1
    again = client.post("/api/recipes/sync", json={})           # remote set-url + fetch + reset
    assert again.status_code == 200, again.text
    assert again.json()["commit"] == first.json()["commit"]

    listing = client.get("/api/recipes").json()
    assert listing["default_repo"] == url                       # หน้าเว็บโชว์ต้นทางที่จะดึงจริง
    assert listing["source"]["repo"] == url and listing["source"]["count"] == 1


def test_a_ref_shaped_like_a_git_option_runs_nothing(tmp_path, offline_default):
    """ผู้ตรวจ: ref `--upload-pack=touch <marker>;` → git fetch อ่านเป็น option แล้วรันคำสั่งนั้นบน hub"""
    url = _controller_repo(tmp_path)
    _configure_recipe_source(url)
    client = TestClient(create_app())
    assert client.post("/api/recipes/sync", json={"repo": url, "ref": "main"}).status_code == 200

    marker = tmp_path / "RAN_ON_HUB"
    r = client.post("/api/recipes/sync", json={"repo": url, "ref": f"--upload-pack=touch {marker};"})

    assert r.status_code == 400
    assert not marker.exists(), "คำสั่งที่ฝังมาใน ref ถูกรันบน hub"


def test_a_request_cannot_point_the_hub_at_another_repo(tmp_path, offline_default, monkeypatch):
    """repo/ref ใน body ที่ไม่ตรงกับค่าตั้งของ hub = 400 และ git ไม่ถูกเรียกเลย"""
    from lmds.recipes import sync as sync_module

    url = _controller_repo(tmp_path)
    _configure_recipe_source(url)
    calls: list[tuple] = []
    monkeypatch.setattr(sync_module, "_git", lambda *args, **kwargs: calls.append(args) or "")
    client = TestClient(create_app())

    for body in ({"repo": "https://example.invalid/evil/repo"}, {"ref": "other-branch"},
                 {"repo": url, "ref": "v2"}, {"repo": 5}, {"ref": ["main"]}):
        r = client.post("/api/recipes/sync", json=body)
        assert r.status_code == 400, (body, r.text)
    assert calls == []


def test_the_sink_refuses_a_hostile_ref_for_every_caller(tmp_path, monkeypatch):
    """CLI และ publish เรียก fetch() ตรง ๆ — ด่านต้องอยู่ที่ sink ไม่ใช่ที่ endpoint อย่างเดียว

    ชั้นที่สอง: ต่อให้ด่านตรวจ ref หลุด (จำลองด้วยการถอดด่านออก) `--` ยังกันไม่ให้ git อ่านเป็น option
    """
    from lmds.recipes import sync as sync_module

    url = _controller_repo(tmp_path)
    sync_module.fetch(url, "main")
    marker = tmp_path / "RAN_ON_HUB"
    hostile = f"--upload-pack=touch {marker};"

    with pytest.raises(sync_module.SyncError):
        sync_module.fetch(url, hostile)
    assert not marker.exists()

    monkeypatch.setattr(sync_module, "validate_ref", lambda ref: ref)
    with pytest.raises(sync_module.SyncError):
        sync_module.fetch(url, hostile)
    assert not marker.exists(), "ไม่มี `--` คั่น — git ยังอ่าน ref เป็น option"


def test_the_sink_never_removes_anything_but_its_own_checkout(tmp_path, monkeypatch):
    """rmtree ก่อน clone ต้องพิสูจน์ก่อนว่าเป้าหมายเป็นลูกตรงของแคช — ชั้นบนพลาดแค่ไหนก็ไม่ลบ config dir"""
    from lmds.recipes import sync as sync_module

    before = _precious_config()
    url = _controller_repo(tmp_path)

    with pytest.raises(sync_module.SyncError):                   # input ของผู้ตรวจ ผ่าน sink ตรง ๆ
        sync_module.fetch(str(tmp_path / "nope") + "/..")
    assert sorted(p.name for p in config_dir().iterdir()) == before

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x", encoding="utf-8")
    sync_module.cache_root().mkdir(parents=True, exist_ok=True)
    link = sync_module.cache_root() / "repo-0123456789abcdef"    # ชื่อถูกรูป แต่เป็น symlink ออกนอกแคช
    link.symlink_to(outside)
    for target in (config_dir(), sync_module.cache_root(), sync_module.cache_root() / "..",
                   sync_module.cache_root() / "published-local", link, outside):
        monkeypatch.setattr(sync_module, "checkout_dir", lambda repo, target=target: target)
        with pytest.raises(sync_module.SyncError):
            sync_module.fetch(url, "main")
    assert sorted(p.name for p in config_dir().iterdir()) == sorted([*before, "controllers"])
    assert (outside / "keep").is_file()


def test_the_checkout_directory_is_never_derived_from_caller_text():
    from lmds.recipes import sync as sync_module
    from lmds.recipes.publish import default_local_repo

    seen = set()
    for repo in ("https://example.com/team/published-local", "https://example.com/a/controllers.git",
                 "git@example.com:team/x.git", "ssh://git@example.com:2222/team/x"):
        target = sync_module.checkout_dir(repo)
        assert target.parent == sync_module.cache_root()
        assert sync_module._CHECKOUT_NAME.fullmatch(target.name)
        assert target != default_local_repo()                    # เดิมรีโปชื่อนี้ทับ local store ของ publish
        seen.add(target)
    assert len(seen) == 4
    # สะกดต่างกันแต่เป็นรีโปเดียวกัน = สำเนาเดียวกัน (ไม่ clone ซ้ำเพราะ .git ต่อท้าย)
    assert sync_module.checkout_dir("https://example.com/a/b") == sync_module.checkout_dir("https://example.com/a/b.git/")


@pytest.mark.parametrize("repo", [
    "", "-oProxyCommand=id", "--upload-pack=id", "ext::sh -c id", "https://user:secret@example.com/a/b",
    "ssh://-oProxyCommand=id/x", "git@example.com:-x/y", "/abs/path/..", "https://example.com/a/../../b",
    "https://example.com/a b", "https://example.com/a\nb", "file://host/share", "file:///a/../b",
    "../relative", "example.com/a/b", 5, None, ["https://example.com/a/b"],
])
def test_repo_values_git_must_never_see(repo):
    from lmds.recipes.sync import SyncError, validate_repo

    with pytest.raises(SyncError):
        validate_repo(repo)


@pytest.mark.parametrize("repo", [
    "https://github.com/neronain/dgx-spark-all-controllers", "https://git.example.com:8443/team/recipes.git",
    "http://gitea.lan/team/recipes", "git@github.com:neronain/script-update.git",
    "ssh://git@git.example.com:2222/team/recipes.git", "file:///srv/mirror/controllers",
])
def test_repo_forms_that_real_sites_use_still_pass(repo):
    from lmds.recipes.sync import DEFAULT_REPO, validate_repo

    assert validate_repo(repo) == repo
    assert validate_repo(DEFAULT_REPO) == DEFAULT_REPO


@pytest.mark.parametrize("ref", [
    "", "-x", "--upload-pack=touch /tmp/x;", "a..b", "a b", "main;id", "x.lock", "refs/heads/", "a//b",
    "$(id)", "main\n", "@{-1}", 7, None,
])
def test_ref_values_git_must_never_see(ref):
    from lmds.recipes.sync import SyncError, validate_ref

    with pytest.raises(SyncError):
        validate_ref(ref)


def test_ordinary_branch_and_tag_names_still_pass():
    from lmds.recipes.sync import validate_ref

    for ref in ("main", "v1.2.3", "release/2026-09", "feature_x-2"):
        assert validate_ref(ref) == ref


def test_a_wrong_shaped_config_file_is_explained_not_a_bare_crash():
    """ค่าใหม่ `recipes.sync_repo` อยู่ใน config.yaml ที่คนแก้มือ — พิมพ์ผิดชนิดต้องได้ข้อความที่บอกไฟล์และคีย์

    เดิม YAML ที่ถูกไวยากรณ์แต่ผิดรูป (ทั้งไฟล์เป็นข้อความ · ค่าเป็น list) หลุดเป็น ValidationError ของ
    pydantic ที่ไม่มีใครจับ → 500 เปล่า ๆ ทุก route ที่อ่าน config
    """
    from lmds.config import Settings, SettingsError

    config_dir().mkdir(parents=True, exist_ok=True)
    for text in ("precious\n", "recipes:\n  sync_repo: [1, 2]\n"):
        (config_dir() / "config.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(SettingsError) as caught:
            Settings.load()
        assert "config.yaml" in str(caught.value)
        r = TestClient(create_app(), raise_server_exceptions=False).post("/api/recipes/sync", json={})
        assert r.status_code == 500 and "config.yaml" in r.json()["detail"]


def test_publish_goes_through_the_same_gate_before_touching_git(tmp_path, monkeypatch):
    """publish ใช้ sink เดียวกัน (checkout_dir + _git) — repo/ref ที่ไม่ผ่านด่านต้องไม่ถึง git"""
    from lmds.recipes import publish as publish_module
    from lmds.recipes.sync import SyncError

    controller = tmp_path / "demo-single.sh"
    controller.write_text(VLLM_CONTROLLER, encoding="utf-8")
    calls: list[tuple] = []
    monkeypatch.setattr(publish_module, "_git", lambda *args, **kwargs: calls.append(args) or "")

    for repo, ref in (("git@example.com:team/x.git", "--upload-pack=id"),
                      ("https://example.com/team/..", "main"),
                      ("ssh://-oProxyCommand=id/x", "main")):
        with pytest.raises(SyncError):
            publish_module.publish("demo", controller, {}, repo=repo, ref=ref, host="h", now="d")
    assert calls == []
