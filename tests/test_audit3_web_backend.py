"""audit 2026-10 ชั้นเว็บ (backend) — ทุกข้อในรายงานมีเทสที่ล้มก่อนแก้

เทสในไฟล์นี้ยิงผ่าน endpoint จริง (TestClient) ด้วย input ชนิดเดียวกับที่ผู้ตรวจใช้ แล้วยืนยันที่
**ผลที่เกิดขึ้นจริง** — ไฟล์ยังอยู่ไหม · คำสั่งถูกรันไหม · process ตายจริงไหม — ไม่ใช่ที่ข้อความในซอร์ส
· ที่ไหนมี sink (ฟังก์ชันที่ลงมือทำของอันตราย) เทส sink ตรง ๆ ด้วยอีกชั้น เพราะ endpoint ไม่ใช่ผู้เรียกคนเดียว

ลำดับตามรายงาน:
  1. POST /api/recipes/sync เชื่อ repo/ref จาก body → ลบ config dir · รันคำสั่งผ่าน option ของ git
  2. token ที่ไม่ใช่ ASCII → ทุกคำขอ 500 · การเดาด้วยค่าที่ไม่ใช่ ASCII ไม่ถูกนับเข้า lockout
  3. ชื่อ interface ที่ node รายงานเองลง cluster.env ดิบ ๆ → รันตอน controller source ไฟล์
  4. cancel ฆ่าแค่ bash ของ controller → ตัวโหลดเป็นกำพร้า งานยัง running ล็อกไม่หลุด แต่ตอบว่ายกเลิกแล้ว
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
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


# ══ 2. token ที่ไม่ใช่ ASCII ════════════════════════════════════════════════════

THAI_TOKEN = "รหัสผ่านของฮับ"


@pytest.fixture
def audit_log(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "audit.log"
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(path))
    monkeypatch.delenv("LMDS_AUDIT", raising=False)
    return path


def _audit_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_thai_passphrase_is_verified_instead_of_crashing_every_request(audit_log):
    """ผู้ตรวจ: `validate_token` รับ passphrase ไทย แต่ทุกคำขอที่ผ่าน guard ตอบ 500 — รวมทั้งตัวที่ถูก"""
    from lmds.web import daemon

    token = daemon.validate_token(THAI_TOKEN)
    client = TestClient(create_app(token), raise_server_exceptions=False)
    wire = token.encode("utf-8")                       # สิ่งที่ curl / หน้าเว็บส่งจริงใน header

    assert client.post("/api/auth", headers={"x-lmds-token": wire}).status_code == 200
    assert client.get("/api/version", headers={"x-lmds-token": wire}).status_code == 200
    assert client.post("/api/auth", params={"token": token}).status_code == 200      # ?token=
    assert client.get("/api/version", params={"token": token}).status_code == 200
    assert client.get("/api/version", params={"token": "wrongwrong"}).status_code == 401
    assert client.get("/api/version", headers={"x-lmds-token": "รหัสผิดแน่นอน".encode()}).status_code == 401
    assert client.get("/api/version").status_code == 401


def test_a_thai_passphrase_works_over_real_http_including_the_event_stream(audit_log):
    """ผ่าน uvicorn จริง (ตัวแยก header ตัวจริง) — header · ?token= · SSE ของ /api/events"""
    import http.client
    from urllib.parse import quote

    from tests.test_logs_follow import LiveServer

    with LiveServer(create_app(THAI_TOKEN)) as srv:
        def ask(method: str, path: str, header: bytes | None = None) -> int:
            conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
            conn.putrequest(method, path)
            if header is not None:
                conn.putheader("x-lmds-token", header)
            conn.putheader("content-length", "0")
            conn.endheaders()
            status = conn.getresponse().status
            conn.close()                                  # ปิดสาย = SSE จบ (สิ่งที่ TestClient ทำไม่ได้)
            return status

        assert ask("POST", "/api/auth", THAI_TOKEN.encode("utf-8")) == 200
        assert ask("GET", "/api/version?token=" + quote(THAI_TOKEN)) == 200
        assert ask("GET", "/api/events?token=" + quote(THAI_TOKEN)) == 200
        assert ask("GET", "/api/events?token=" + quote("ไม่ใช่รหัสนี้")) == 401
        assert ask("POST", "/api/auth", "ไม่ใช่รหัสนี้".encode()) == 401


def test_two_spellings_of_the_same_passphrase_are_one_passphrase(audit_log):
    """"é" จุดรหัสเดียว (NFC) กับ e + ◌́ (NFD) คือสิ่งเดียวกันที่คนพิมพ์ — เทอร์มินัลกับเบราว์เซอร์ให้ไม่เหมือนกันได้"""
    import unicodedata

    typed_on_the_hub = unicodedata.normalize("NFD", "caféprivé-hub")
    typed_in_the_browser = unicodedata.normalize("NFC", "caféprivé-hub")
    assert typed_on_the_hub != typed_in_the_browser
    client = TestClient(create_app(typed_on_the_hub), raise_server_exceptions=False)
    assert client.post("/api/auth", headers={"x-lmds-token": typed_in_the_browser.encode()}).status_code == 200


def test_guesses_of_any_shape_count_towards_the_lockout(audit_log):
    """ผู้ตรวจ: token ASCII + เดาด้วยค่าที่ไม่ใช่ ASCII → 500 ก่อนถึงตัวนับ จึงเดาได้ไม่จำกัด"""
    from lmds.web import api

    client = TestClient(create_app("s3cret-token"), raise_server_exceptions=False)
    guesses = [client.get("/api/version", params={"token": "ก" * 8}).status_code for _ in range(4)]
    guesses += [client.get("/api/version", headers={"x-lmds-token": b"\xff\xfe\xfd-not-utf8"}).status_code
                for _ in range(4)]
    guesses += [client.get("/api/version", headers={"x-lmds-token": "ข้อความ".encode()}).status_code
                for _ in range(4)]

    assert guesses[: api._FAIL_FREE + 1] == [401] * (api._FAIL_FREE + 1), guesses
    assert set(guesses[api._FAIL_FREE + 1:]) == {429}, guesses
    # ถูกล็อกจริง: แม้ token ที่ถูกก็ต้องรอ — เดิมตรงนี้ยังเข้าได้เพราะ 12 ครั้งข้างบนไม่เคยถูกนับ
    assert client.get("/api/version", params={"token": "s3cret-token"}).status_code == 429
    rows = _audit_rows(audit_log)
    assert len(rows) == 13 and {row["status"] for row in rows} == {401, 429}


def test_the_console_does_not_start_with_a_token_its_guard_cannot_verify(monkeypatch):
    """self-check ตอนสตาร์ต: guard ตัวจริงต้องรับ token ของตัวเองได้ ไม่งั้น `lmds web` ต้องไม่ขึ้น

    จำลอง guard ที่ถอยกลับไปเป็นแบบเดิม (เทียบ str ตรง ๆ ซึ่งรับได้แค่ ASCII) — ของที่เคยเกิดคือ
    หน้าเว็บ "เปิดแล้ว" แต่ทุกคำขอตอบ 500 · ต้องล้มตั้งแต่สร้างแอป และ serve() ต้องจบก่อน bind พอร์ต
    """
    import secrets

    import uvicorn

    from lmds.web import api

    class Started(Exception):
        pass

    def never(*_args, **_kwargs):
        raise Started("uvicorn.run ถูกเรียก — หน้าเว็บกำลังจะรายงานว่าเปิดแล้ว")

    monkeypatch.setattr(uvicorn, "run", never)
    monkeypatch.setattr(api, "token_matches", lambda supplied, token: secrets.compare_digest(supplied, token))

    with pytest.raises(api.TokenUnusable):
        create_app(THAI_TOKEN)
    with pytest.raises(SystemExit) as stopped:
        api.serve(port=1, token=THAI_TOKEN)
    assert stopped.value.code == 1
    create_app("ascii-token-still-fine")                # guard แบบเก่ายืนยัน ASCII ได้ — ไม่ล้มพร่ำเพรื่อ


@pytest.mark.parametrize("token", ["abc\udcffdefgh", " padded-token ", "has\ttab-inside", "line\nbreak-token"])
def test_tokens_no_client_could_ever_send_are_refused_at_startup(token):
    from lmds.web import api

    with pytest.raises(api.TokenUnusable):
        create_app(token)


def test_validate_token_refuses_bytes_that_are_not_text_and_keeps_thai():
    from lmds.web import daemon

    assert daemon.validate_token(THAI_TOKEN) == THAI_TOKEN
    for bad in ("abc\udcffdefgh", "delete\x7fchar-token", "c1\x85control-token"):
        with pytest.raises(daemon.TokenError):
            daemon.validate_token(bad)


def test_lmds_web_refuses_an_environment_token_it_could_never_verify(monkeypatch, tmp_path):
    """`$LMDS_WEB_TOKEN` ที่มีไบต์เสีย (locale ผิดตอนตั้ง unit) → CLI ต้องจบด้วย error ไม่ใช่ไปถึง serve()"""
    from typer.testing import CliRunner

    import lmds.web
    from lmds.cli.main import app
    from tests.test_web_auth_audit import _web_sandbox

    _web_sandbox(monkeypatch, tmp_path)
    served: list = []
    monkeypatch.setattr(lmds.web, "serve", lambda **kwargs: served.append(kwargs))
    monkeypatch.setenv("LMDS_WEB_TOKEN", "abc\udcffdefgh")

    result = CliRunner().invoke(app, ["web"])
    assert result.exit_code == 1, result.output
    assert served == []
    # ปฏิเสธแบบบอกเหตุ (typer.Exit) พร้อมชื่อที่มาของค่า — ไม่ใช่ traceback จากการพยายามพิมพ์ token ที่พิมพ์ไม่ได้
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "LMDS_WEB_TOKEN" in result.output


def _page_token_headers(tmp_path, prelude: str, body: str) -> list[list]:
    """บูตหน้าเว็บจริงใน node แล้วคืน [ชื่อคำขอ, จุดรหัสของ header x-lmds-token ที่หน้าเว็บจะส่ง]"""
    from tests.test_console_shell import run_scenario

    setup = """
const fx = { nodes: [] };
H.seen = [];
const note = (url, opts) => { const h = (opts && opts.headers) || {};
  if ("x-lmds-token" in h) H.seen.push([String(url).split("?")[0], Array.from(String(h["x-lmds-token"]), c => c.charCodeAt(0))]); };
H.routes = [
  ["/api/auth", (url, opts) => { note(url, opts); return { required: true }; }],
  ...H.defaultRoutes(fx).filter(r => r[0] !== "/api/auth").map(([pat, handler]) => [pat, (url, opts) => { note(url, opts); return handler(url, opts); }]),
];
"""
    (out,) = run_scenario(tmp_path, setup + prelude, body + "\nconsole.log(JSON.stringify(H.seen));")
    return out


def _assert_the_hub_accepts_what_the_page_sends(seen: list[list]) -> None:
    assert seen, "หน้าเว็บไม่ได้ส่ง token เลย"
    client = TestClient(create_app(THAI_TOKEN), raise_server_exceptions=False)
    for name, codes in seen:
        # fetch ของเบราว์เซอร์จริงโยน TypeError เมื่อ header มีจุดรหัสเกิน 255 — DOM ย่อส่วนไม่ตรวจให้ เทสจึงตรวจเอง
        assert max(codes) <= 255, f"{name}: header มีอักขระนอก ISO-8859-1 — เบราว์เซอร์จะไม่ส่งคำขอนี้เลย"
        assert client.post("/api/auth", headers={"x-lmds-token": bytes(codes)}).status_code == 200, name


def test_the_page_sends_a_thai_token_in_a_form_the_browser_allows_and_the_hub_accepts(tmp_path):
    """ครบวง: หน้าเว็บจริง (JS จริงใน node) → header ที่ได้ → guard ตัวจริง · ทั้งตอนบูตและทุกคำขอผ่าน api()"""
    seen = _page_token_headers(tmp_path, f'localStorage.setItem("lmds:token", {THAI_TOKEN!r});', "await H.tick();")
    names = {name for name, _codes in seen}
    assert "/api/auth" in names and len(names) > 1, names     # ตอนบูต + คำขอปกติของหน้า
    _assert_the_hub_accepts_what_the_page_sends(seen)


def test_the_login_box_can_submit_a_thai_token(tmp_path):
    """กรอก passphrase ไทยในหน้า login แล้วกด Sign in — เดิม fetch โยน TypeError หน้าจึงเงียบ"""
    seen = _page_token_headers(tmp_path, "", f"""
        document.getElementById("tok").value = {THAI_TOKEN!r};
        await document.getElementById("tok-go").onclick();
        await H.tick();
        H.assert(localStorage.getItem("lmds:token") === {THAI_TOKEN!r}, "token ที่ผ่านแล้วต้องถูกจำไว้");""")
    assert [name for name, _codes in seen] == ["/api/auth"]
    _assert_the_hub_accepts_what_the_page_sends(seen)


# ══ 3. cluster.env — ค่าที่ node รายงานเองถูก source เป็น shell ══════════════════════

def _link(iface: str, ip: str, peer_rank, peer_ip: str, hca: str = "rocep1s0f1") -> dict:
    return {"iface": iface, "ip": ip, "prefix": 24, "peer_rank": peer_rank, "peer_ip": peer_ip,
            "link_id": "", "hca": hca}


def _two_node_topology() -> dict:
    return {"kind": "direct-2", "nodes": [
        {"name": "spark-head", "rank": 0, "legacy": False,
         "links": [_link("enp1s0f1np1", "10.100.152.1", 1, "10.100.152.2")]},
        {"name": "spark-worker", "rank": 1, "legacy": False,
         "links": [_link("enp1s0f1np1", "10.100.152.2", 0, "10.100.152.1")]},
    ]}


def _source_under_bash(tmp_path: Path, body: str, names: list[str]) -> dict[str, str]:
    """อ่านไฟล์แบบเดียวกับ controller (`set -a; . "$CLUSTER_ENV"`) แล้วคืนค่าที่ bash เห็นจริง"""
    env_file = tmp_path / "cluster.env"
    env_file.write_text(body, encoding="utf-8")
    script = 'set -a; . "$1"; set +a; shift; for name in "$@"; do printf "%s\\0" "${!name-<unset>}"; done'
    done = subprocess.run(["bash", "-c", script, "bash", str(env_file), *names], capture_output=True,
                          cwd=tmp_path, timeout=30)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return dict(zip(names, done.stdout.decode("utf-8").split("\0"), strict=False))


def test_an_interface_name_reported_by_a_node_cannot_reach_cluster_env(tmp_path, monkeypatch):
    """ผู้ตรวจ: `host.fabric.links[].iface` = `x$(touch${IFS}<marker>)` → hub เขียนลง cluster.env บน head
    → คำสั่งรันตอน controller source ไฟล์ · ตอนนี้ endpoint ต้องปฏิเสธ และไม่มีอะไรถูกเขียนข้ามเครื่อง"""
    import lmds.fleet
    import lmds.nodes
    from tests.test_audit_stacked_orchestration import FakeSSH, register, spark, stacked_bundle

    monkeypatch.setattr("lmds.inventory.host_payload", lambda: {
        "hostname": "hub", "gpus": [], "arch": "x86_64", "profile": "generic",
        "fabric": {"links": [], "best_gbps": 10, "tier": "basic"}, "role": {"control_plane": True, "engines": []}})
    marker = tmp_path / "RAN_ON_HEAD"
    evil = f"x$(touch${{IFS}}{marker})"
    head, worker = spark("10.100.152.1", "10.2.2.1"), spark("10.100.152.2", "10.2.2.2")
    for link in head["fabric"]["links"]:
        link["iface"] = evil
    register("n1", "10.2.2.1", "10.100.152.1", head, site="TKC")
    register("n2", "10.2.2.2", "10.100.152.2", worker, site="TKC")
    monkeypatch.setattr(lmds.fleet, "bundle_roots", lambda: [tmp_path / "bundles"])
    stacked_bundle(tmp_path, "two", 2)
    ssh = FakeSSH(lambda node, command: (0, "/home/nvidia/bundles/two/cluster.env", ""))
    monkeypatch.setattr(lmds.nodes, "run", ssh)

    r = TestClient(create_app()).post("/api/cluster/write", json={"slug": "two", "head": "n1", "worker": "n2", "on": "n1"})

    assert r.status_code == 400, r.text
    assert "n1" in r.json()["detail"]                       # บอกว่าเครื่องไหนรายงานค่าแปลก
    assert not any("base64 -d" in command for _node, command in ssh.calls), "ไฟล์ถูกเขียนไปแล้ว"
    assert not marker.exists()


HOSTILE = "x$(touch${IFS}%s)`touch${IFS}%s`\";touch %s;\"';touch %s;'\ntouch %s\n"


@pytest.mark.parametrize("field", ["iface", "hca", "ip", "peer_ip", "prefix", "peer_rank", "name", "kind", "ssh_user"])
def test_every_field_is_checked_against_what_it_can_legitimately_be(field, tmp_path):
    """ด่านที่ sink: ค่าที่ไม่ใช่ชื่อ interface / IP / อุปกรณ์ / จำนวนเต็ม ถูกปฏิเสธพร้อมบอกฟิลด์ — ไม่ใช่เขียนแบบ quote ไว้"""
    from lmds.fleet.cluster_env import ClusterEnvError, render_cluster_env

    topology, ssh_user = _two_node_topology(), "nvidia"
    evil = "x$(touch /tmp/never)"
    if field in ("iface", "hca", "ip", "peer_ip", "prefix", "peer_rank"):
        topology["nodes"][1]["links"][0][field] = evil
    elif field == "name":
        topology["nodes"][1]["name"] = evil
    elif field == "kind":
        topology["kind"] = evil
    else:
        ssh_user = evil

    with pytest.raises(ClusterEnvError):
        render_cluster_env(topology, ssh_user=ssh_user)


def test_hostile_values_in_every_field_run_nothing_when_the_file_is_sourced(tmp_path, monkeypatch):
    """ชั้นที่สอง: ถอดด่านตรวจออก (จำลองว่ามีทางหลุด) แล้วใส่ค่าร้ายทุกฟิลด์พร้อมกัน — bash ต้องอ่านเป็นข้อความ

    รันใต้ bash จริงแบบที่ controller ทำ: ไม่มี marker สักไฟล์ และตัวแปรได้ค่าตรงตัวอักษร
    """
    from lmds.fleet import cluster_env

    monkeypatch.setattr(cluster_env, "_validate_topology", lambda *args, **kwargs: None, raising=False)
    markers = iter(tmp_path / f"MARK{i}" for i in range(400))

    def evil() -> str:
        return HOSTILE % tuple(next(markers) for _ in range(5))

    topology = _two_node_topology()
    topology["kind"] = evil()
    values = {"ssh_user": evil()}
    for node in topology["nodes"]:
        node["name"] = evil()
        for link in node["links"]:
            for key in ("iface", "hca", "ip", "peer_ip", "prefix"):
                link[key] = evil()
    values["iface0"] = topology["nodes"][0]["links"][0]["iface"]
    values["hca1"] = topology["nodes"][1]["links"][0]["hca"]

    body = cluster_env.render_cluster_env(topology, ssh_user=values["ssh_user"])
    seen = _source_under_bash(tmp_path, body, [
        "NCCL_SOCKET_IFNAME", "SSH_USER", "NCCL_IB_HCAS_1", "CLUSTER_TOPOLOGY", "LINKS_0", "CLUSTER_NODES",
        "MASTER_IP", "WORKER_IPS", "HEAD_TO_WORKER_IP_1", "WORKER_HEAD_IP_1", "NNODES"])

    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("MARK")) == []
    assert seen["NCCL_SOCKET_IFNAME"] == values["iface0"]
    assert seen["SSH_USER"] == values["ssh_user"]
    assert seen["NCCL_IB_HCAS_1"] == values["hca1"]
    assert seen["CLUSTER_TOPOLOGY"] == topology["kind"]
    assert seen["NNODES"] == "2"


def test_a_legitimate_cluster_env_is_unchanged_and_thai_machine_names_survive(tmp_path):
    """ค่าที่ถูกรูปต้องออกมาหน้าตาเดิม (controller เก่าและเทสอื่นอ่านรูปนี้) · ชื่อเครื่องภาษาไทยที่ทะเบียนรับ
    ต้องผ่านด่านและ bash ต้องอ่านกลับได้ตรงตัว"""
    from lmds.fleet.cluster_env import render_cluster_env

    topology = _two_node_topology()
    body = render_cluster_env(topology, ssh_user="nvidia")
    for line in ("MASTER_IP=10.100.152.1", 'WORKER_IPS="10.100.152.2"', "SSH_USER=nvidia",
                 "NCCL_SOCKET_IFNAME=enp1s0f1np1", 'CLUSTER_NODES="spark-head spark-worker"',
                 'LINKS_0="enp1s0f1np1:10.100.152.1/24:1:10.100.152.2"', "NCCL_IB_HCAS_1=rocep1s0f1"):
        assert line in body.splitlines(), line

    topology["nodes"][1]["name"] = "เครื่องสอง"
    seen = _source_under_bash(tmp_path, render_cluster_env(topology, ssh_user="nvidia"), ["CLUSTER_NODES", "WORKER_IP"])
    assert seen == {"CLUSTER_NODES": "spark-head เครื่องสอง", "WORKER_IP": "10.100.152.2"}


@pytest.mark.parametrize("slug", ["x';touch /tmp/never;'", "../../etc", "a b", "$(id)", ""])
def test_the_cluster_env_writer_checks_the_slug_itself(slug, monkeypatch):
    """slug ถูกต่อเป็นคำสั่งบนเครื่องปลายทาง — CLI เรียก sink ตรง ๆ โดยไม่ผ่านด่านของ endpoint"""
    import lmds.nodes
    from lmds.fleet.cluster_env import ClusterEnvError, write_cluster_env
    from tests.test_audit_stacked_orchestration import FakeSSH, group_of

    lmds.nodes.add(lmds.nodes.Node(name="n1", host="10.2.2.1", user="nvidia"))
    ssh = FakeSSH()
    monkeypatch.setattr(lmds.nodes, "run", ssh)
    with pytest.raises(ClusterEnvError):
        write_cluster_env(slug, [group_of("n1", "n2")], "n1", None, "n1")
    assert ssh.calls == []


# ══ 4. ยกเลิกงาน — ฆ่าทั้งกลุ่ม รอจนหายจริง แล้วค่อยบอกว่ายกเลิกแล้ว ══════════════════

def _pid_alive(pid: int) -> bool:
    """process ยังรันอยู่จริงไหม — zombie ที่รอคนเก็บไม่นับ (มันไม่ได้ทำอะไรแล้ว)"""
    done = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = done.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_dead(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


@pytest.fixture
def downloading(tmp_path, monkeypatch):
    """controller ตัวแทนของ `download` จริง: bash ที่แตกตัวโหลด (docker/curl/python) เป็นลูกแล้วรอ

    คืน start(child_command) → (job, pid ของตัวโหลด, path ของ controller)
    """
    from lmds.hardware import serving
    from lmds.web import jobs

    monkeypatch.setattr(serving, "guard", lambda *args, **kwargs: "")      # เครื่องเทสไม่มี GPU — ไม่เกี่ยวกับข้อนี้
    pidfile = tmp_path / "child.pid"
    ctl = tmp_path / "bundles" / "demo" / "demo-single.sh"
    ctl.parent.mkdir(parents=True)
    started: list[int] = []

    def start(child: str = "sleep 300", trap: str = ""):
        ctl.write_text(f"""#!/usr/bin/env bash
{trap}
case "$1" in
  download) echo "downloading 70 GB"; {child} & echo $! > {pidfile}; wait $!; echo "download finished" ;;
  verify-files) echo "verify ok" ;;
  start) echo "started" ;;
esac
""", encoding="utf-8")
        ctl.chmod(0o755)
        job = jobs.start("demo", "download", str(ctl))
        deadline = time.time() + 5
        while not pidfile.exists() or not pidfile.read_text().strip():
            assert time.time() < deadline, "controller ไม่ได้แตกตัวโหลด"
            time.sleep(0.02)
        started.append(int(pidfile.read_text()))
        return job, started[-1], ctl

    yield start
    for pid in started:                                                   # เก็บกวาดของเทสเอง ไม่ว่าเทสจะผ่านไหม
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def test_cancel_stops_the_downloader_not_just_the_controller_shell(downloading):
    """ผู้ตรวจ: cancel ตอบ `cancelled: true` แต่ตัวโหลดยังรันเป็นกำพร้า งานยัง running และล็อกโมเดลไม่หลุด"""
    from lmds.web import jobs

    job, child, ctl = downloading()
    client = TestClient(create_app())

    result = client.post(f"/api/jobs/{job.id}/cancel").json()

    assert result["cancelled"] is True, result
    assert _wait_dead(child, 1.0), "ตัวโหลดยังรันอยู่หลัง hub ตอบว่ายกเลิกแล้ว"
    payload = client.get(f"/api/jobs/{job.id}").json()                     # ทันที — ไม่ใช่ "อีกสักพัก"
    assert payload["running"] is False and payload["exit_code"] not in (0, None)
    assert result["still_running"] is False and result["lock_released"] is True
    assert "verify ok" not in payload["output"], "ขั้นถัดไปของ chain เริ่มทั้งที่ผู้ใช้กดยกเลิก"
    assert result["notes"] and "Stop" in result["notes"][0]                # container ที่ docker ถืออยู่ไม่ได้ถูกหยุด — บอกตามจริง
    again = jobs.start("demo", "start", str(ctl))                          # ล็อกหลุดจริง: งานใหม่เริ่มได้
    assert again.id != job.id


def test_cancel_escalates_to_kill_when_the_job_ignores_term(downloading, monkeypatch):
    from lmds.web import jobs

    monkeypatch.setattr(jobs, "CANCEL_GRACE", 0.4)
    job, child, _ctl = downloading(child="""bash -c 'trap "" TERM; sleep 300'""", trap="trap '' TERM")
    result = TestClient(create_app()).post(f"/api/jobs/{job.id}/cancel").json()

    assert result["cancelled"] is True and result["lock_released"] is True, result
    assert _wait_dead(child, 1.0)
    assert job.running is False


def test_a_process_that_left_the_group_cannot_keep_the_model_locked(downloading):
    """ตัวที่ daemonize ตัวเอง (setsid) ไม่อยู่ในกลุ่มของงาน — ฆ่ากลุ่มแล้วมันยังถือท่ออยู่

    ล็อกต้องหลุดเมื่อกลุ่มของงานหาย และผลต้องบอกว่ายังมีอะไรเหลือ ไม่ใช่เงียบ
    """
    import sys

    from lmds.web import jobs

    job, escapee, ctl = downloading(child=f"{sys.executable} -c 'import os, time; os.setsid(); time.sleep(300)'")
    result = TestClient(create_app()).post(f"/api/jobs/{job.id}/cancel").json()

    assert _pid_alive(escapee), "ตัวอย่างในเทสเปลี่ยนไป — ตัวที่หนีออกนอกกลุ่มควรรอดจาก killpg"
    assert result["cancelled"] is True and result["lock_released"] is True, result
    assert "detached" in result["detail"]
    assert job.running is False
    jobs.start("demo", "start", str(ctl))


def test_cancelling_a_finished_job_says_so_and_cancelling_a_remote_job_says_what_survives(monkeypatch):
    """งานบนเครื่องอื่น: ที่ถูกฆ่าคือ ssh บน hub — คำสั่งบนเครื่องนั้นอาจรันต่อ ผลต้องบอก ไม่ใช่ปล่อยให้เข้าใจว่าหยุดแล้ว"""
    import lmds.nodes
    from lmds.web import jobs

    lmds.nodes.add(lmds.nodes.Node(name="spark2", host="10.2.2.2", user="nvidia"))
    monkeypatch.setattr("lmds.nodes.stream", lambda node, command, *_, **__: subprocess.Popen(
        ["sleep", "60"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    job = jobs.start_remote("spark2", "demo", "start", "true")
    client = TestClient(create_app())
    deadline = time.time() + 5
    while job.process is None and time.time() < deadline:
        time.sleep(0.02)

    result = client.post(f"/api/jobs/{job.id}/cancel").json()
    assert result["cancelled"] is True and result["lock_released"] is True, result
    assert any("spark2" in note and "may still be running" in note for note in result["notes"])

    after = client.post(f"/api/jobs/{job.id}/cancel").json()
    assert after["cancelled"] is False and after["signalled"] is False and after["still_running"] is False


# ══ 5. ลบโมเดลในเครื่อง — ยืนยันที่ server ไม่ใช่ที่ JS ═══════════════════════════════

def _bundle_and_weights(tmp_path: Path, slug: str) -> tuple[Path, Path]:
    return tmp_path / "bundles" / slug, tmp_path / "models" / slug / "demo-Q8.gguf"


def test_local_remove_without_a_matching_confirm_deletes_nothing(fleet, tmp_path, monkeypatch):
    """ผู้ตรวจ: `POST /api/models/{slug}/remove` body ว่าง → ลบ bundle **และ weight** ทันที

    SECURITY.md บอกว่าหน้าเว็บต้องสองขั้น (ดูแผน แล้วยืนยันด้วย slug) และทางของ node บังคับอยู่แล้ว —
    ทางของเครื่องนี้พึ่ง JS ของหน้าเว็บอย่างเดียว
    """
    import lmds.fleet

    monkeypatch.chdir(tmp_path)                       # bundle_roots มองหา ./bundles
    bundle, weights = _bundle_and_weights(tmp_path, fleet)
    calls: list = []
    real = lmds.fleet.remove_server
    monkeypatch.setattr(lmds.fleet, "remove_server",
                        lambda server, include_weights=True: calls.append(include_weights) or real(server, include_weights))
    client = TestClient(create_app())

    attempts = [client.post(f"/api/models/{fleet}/remove")] + [
        client.post(f"/api/models/{fleet}/remove", json=body)
        for body in ({}, {"keep_weights": False}, {"confirm": "another-model"}, {"confirm": [fleet]},
                     {"confirm": True}, {"confirm": fleet.upper()})]

    assert [r.status_code for r in attempts] == [400] * 7, [r.text for r in attempts]
    assert calls == [], "ตัวลบถูกเรียกทั้งที่ไม่มีการยืนยัน"
    assert bundle.is_dir() and weights.is_file()


def test_local_remove_keeps_the_weights_unless_asked_otherwise(fleet, tmp_path, monkeypatch):
    """ค่าตั้งต้นตามช่องติ๊กบนหน้าเว็บ ("Keep the weights" ติ๊กไว้ก่อน) — ลบ weight ต้องขอเองตรง ๆ"""
    import lmds.fleet

    monkeypatch.chdir(tmp_path)
    asked: list = []
    monkeypatch.setattr(lmds.fleet, "remove_server",
                        lambda server, include_weights=True: asked.append(include_weights) or [])
    client = TestClient(create_app())

    assert client.post(f"/api/models/{fleet}/remove", json={"confirm": fleet}).status_code == 200
    assert client.post(f"/api/models/{fleet}/remove", json={"confirm": fleet, "keep_weights": True}).status_code == 200
    assert client.post(f"/api/models/{fleet}/remove", json={"confirm": fleet, "keep_weights": False}).status_code == 200
    assert asked == [False, False, True]                     # include_weights
    # ชนิดผิดต้องไม่ถูกเดาเป็น "ลบ weight": "no" · 0 · null
    for odd in ("no", 0, None):
        r = client.post(f"/api/models/{fleet}/remove", json={"confirm": fleet, "keep_weights": odd})
        assert r.status_code == 400, (odd, r.text)
    assert asked == [False, False, True]


def test_the_remove_button_on_the_page_still_works_against_the_real_endpoint(fleet, tmp_path, monkeypatch):
    """ครบวง: กด Remove → Confirm removal บนหน้าเว็บจริง (JS ใน node) → body ที่ได้ยิงเข้า endpoint จริง

    ช่อง "Keep the weights" ติ๊กมาแต่แรก: bundle หาย weight อยู่
    """
    from tests.test_console_shell import run_scenario

    prelude = """const fx = { nodes: [], localModels: [{ slug: "%s", running: false, healthy: false,
      controller_exists: true, downloaded: true, engine: "llamacpp", port: 8000, topology: "single", context: 8192 }] };
H.sent = [];
H.routes = H.defaultRoutes(fx);
H.routes.unshift(["/api/models/%s/removal-plan", () => ({ slug: "%s", items: [], total_bytes: 0 })]);
H.routes.unshift(["/api/models/%s/remove", (url, opts) => { H.sent.push(opts.body); return { slug: "%s", done: [], failed: [] }; }]);
""" % ((fleet,) * 5)
    (sent,) = run_scenario(tmp_path, prelude, f"""
        document.querySelector('button[data-act="opts"][data-slug="{fleet}"]').click(); await H.tick();
        document.querySelector('button[data-act="removeask"][data-slug="{fleet}"]').click(); await H.tick();
        H.assert(document.getElementById("keep-{fleet}").checked === true, "ช่อง Keep the weights ต้องติ๊กมาแต่แรก");
        document.querySelector('button[data-act="removego"][data-slug="{fleet}"]').click(); await H.tick();
        console.log(JSON.stringify(H.sent));""")
    assert len(sent) == 1, sent

    monkeypatch.chdir(tmp_path)
    bundle, weights = _bundle_and_weights(tmp_path, fleet)
    r = TestClient(create_app()).post(f"/api/models/{fleet}/remove", content=sent[0],
                                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["failed"] == []
    assert not bundle.exists(), "bundle ต้องถูกลบ"
    assert weights.is_file(), "ช่อง Keep the weights ติ๊กอยู่ — weight ต้องยังอยู่"


# ══ 6. secret ที่มีบรรทัดใหม่แทรกบรรทัดลงไฟล์ credentials ═══════════════════════════

REAL_KEY = "sk-proj-REALBILLINGKEY-0123456789abcdef"


def _credentials_text() -> str:
    from lmds.config.paths import credentials_file

    return credentials_file().read_text(encoding="utf-8") if credentials_file().exists() else ""


def test_a_newline_in_the_hf_token_cannot_rewrite_another_secret():
    """ผู้ตรวจ: `POST /api/secrets/hf` ด้วย "hf_…\\nopenai=sk-attacker" → key ของ openai ถูกทับ"""
    client = TestClient(create_app())
    assert client.put("/api/provider", json={"name": "openai", "model": "gpt-4o", "api_key": REAL_KEY}).status_code == 200
    before = _credentials_text()

    r = client.post("/api/secrets/hf", json={"token": "hf_abcdefghijklmnopqrstu\nopenai=sk-attacker-controlled"})

    assert r.status_code == 400
    assert _credentials_text() == before
    assert client.get("/api/provider").json()["key_hint"] == f"…{REAL_KEY[-4:]}"
    # token ปกติ (วางแล้วมีบรรทัดใหม่ติดท้าย) ยังเก็บได้
    assert client.post("/api/secrets/hf", json={"token": "hf_abcdefghijklmnopqrstu\n"}).json()["saved"] is True


def test_a_bad_provider_key_is_refused_before_anything_is_saved():
    """key ผิดรูปต้องได้ 400 และต้องไม่ทิ้ง provider ที่เปลี่ยนไปครึ่งเดียว (config บันทึกแล้วแต่ key ไม่ได้เก็บ)"""
    client = TestClient(create_app())
    client.put("/api/provider", json={"name": "openai", "model": "gpt-4o", "api_key": REAL_KEY})

    r = client.put("/api/provider", json={"name": "gemini", "model": "gemini-2.5-pro",
                                          "api_key": "AIza-good\nopenai=sk-attacker-controlled"})

    assert r.status_code == 400
    now = client.get("/api/provider").json()
    assert now["name"] == "openai" and now["key_hint"] == f"…{REAL_KEY[-4:]}"
    assert "attacker" not in _credentials_text()


@pytest.mark.parametrize("value", ["a\nb", "a\rb", "a\x00b", "a\tb", "a\x1bb", "a\x7fb", "a\x85b", " padded", "padded ", ""])
def test_the_secret_store_refuses_values_that_cannot_be_one_line(value):
    """ด่านที่ sink — CLI (`lmds config set-key`) เรียก set_secret ตรง ๆ"""
    from lmds.secrets import get_secret, set_secret

    set_secret("openai", REAL_KEY)
    with pytest.raises(ValueError):
        set_secret("hf", value)
    assert get_secret("openai") == REAL_KEY and "hf" not in _credentials_text()


@pytest.mark.parametrize("name", ["openai\nhf", "a=b", "", "../x", "has space"])
def test_the_secret_store_refuses_names_that_are_not_names(name):
    from lmds.secrets import set_secret

    with pytest.raises(ValueError):
        set_secret(name, "value-1234567890")
    assert _credentials_text() == ""


def test_a_damaged_credentials_file_is_read_around_not_crashed_on():
    """ไฟล์ที่คนแก้มือ/รุ่นเก่าเขียนเพี้ยน: ไบต์ที่ไม่ใช่ UTF-8 · บรรทัดไม่มี = · ชื่อผิดรูป — ข้ามบรรทัดนั้น ที่เหลือยังอ่านได้"""
    from lmds.config.paths import credentials_file
    from lmds.secrets import get_secret, secret_source, set_secret

    credentials_file().parent.mkdir(parents=True, exist_ok=True)
    credentials_file().write_bytes(
        b"# comment\njust some words\n=novalue\nbad name=x\n\xff\xfe\xfd garbage \xff\n"
        b"gemini=\xff\xfebroken-bytes\nopenai=" + REAL_KEY.encode() + b"\nhf=\n")

    assert get_secret("openai") == REAL_KEY
    assert get_secret("gemini") is None and secret_source("gemini") is None
    assert set_secret("hf", "hf_abcdefghijklmnop1234") == "file"            # เขียนทับไฟล์ที่เพี้ยนได้ ไม่ตาย
    assert get_secret("openai") == REAL_KEY and get_secret("hf") == "hf_abcdefghijklmnop1234"
    assert "garbage" not in _credentials_text()


def test_the_credentials_file_is_never_wider_than_0600_not_even_briefly(monkeypatch):
    """เดิม write_text() แล้ว chmod(0o600) — ไฟล์เกิดมาด้วย umask ปกติก่อน · พิสูจน์ด้วยการตัด chmod ทีหลังออก:
    ถ้าสิทธิ์ถูกต้องเพราะ chmod ตามหลัง ไฟล์จะกว้างตาม umask ทันที"""
    import stat

    from lmds.config.paths import credentials_file
    from lmds.secrets import set_secret

    old = os.umask(0)
    try:
        monkeypatch.setattr(Path, "chmod", lambda self, mode, **kwargs: None)
        set_secret("hf", "hf_abcdefghijklmnop1234")
        assert stat.S_IMODE(credentials_file().stat().st_mode) == 0o600
        os.chmod(credentials_file(), 0o644)                                # ไฟล์เดิมที่สิทธิ์หลวม — เขียนรอบถัดไปต้องกลับมาแคบ
        set_secret("openai", REAL_KEY)
        assert stat.S_IMODE(credentials_file().stat().st_mode) == 0o600
        leftovers = [p.name for p in credentials_file().parent.iterdir() if p.name.startswith(".credentials")]
        assert leftovers == []
    finally:
        os.umask(old)


# ══ 7. key ที่บันทึกไว้ถูกส่งไปยัง base_url ที่คำขอระบุ ═══════════════════════════════

@pytest.fixture
def listener():
    """เซิร์ฟเวอร์ HTTP จริงบน 127.0.0.1 ที่จด header Authorization ของทุกคำขอ — ตัวแทนของ "ปลายทางที่คำขอชี้ไป" """
    import http.server
    import threading

    def start():
        seen: list = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — ชื่อตาม http.server
                seen.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"local-model"}]}')

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/v1", seen

    servers: list = []
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_the_saved_provider_key_is_not_sent_to_a_url_named_by_the_request(listener):
    """ผู้ตรวจ: บันทึก key ของ openai แล้ว `POST /api/provider/models {"name":"openai","base_url":"<เครื่องฉัน>"}`
    → เครื่องนั้นได้ `Authorization: Bearer <key จริง>`"""
    elsewhere, seen = listener()
    client = TestClient(create_app())
    client.put("/api/provider", json={"name": "openai", "model": "gpt-4o", "api_key": REAL_KEY})

    r = client.post("/api/provider/models", json={"name": "openai", "base_url": elsewhere})

    assert r.status_code == 200, r.text
    assert seen == [None], "key ที่บันทึกไว้ถูกส่งไปยังที่อยู่ที่คำขอระบุ"
    assert r.json() == {"models": ["local-model"], "key_sent": "none"}

    # ปลายทางอื่นใช้ key ได้เมื่อมากับคำขอเดียวกัน — และ key นั้นไม่ถูกบันทึกทับของเดิม
    r = client.post("/api/provider/models", json={"name": "openai", "base_url": elsewhere, "api_key": "sk-typed-now"})
    assert seen[-1] == "Bearer sk-typed-now" and r.json()["key_sent"] == "request"
    assert client.get("/api/provider").json()["key_hint"] == f"…{REAL_KEY[-4:]}"


def test_the_saved_key_still_reaches_the_address_it_was_saved_with(listener, monkeypatch):
    """ปุ่ม "List models" ของ provider ที่ตั้งไว้ต้องยังใช้ได้โดยไม่ต้องกรอก key ซ้ำ — ไปที่อยู่ที่บันทึกคู่กัน ไม่ไปที่อื่น"""
    from lmds.brain import providers

    saved_url, at_saved = listener()
    other_url, at_other = listener()
    client = TestClient(create_app())
    client.put("/api/provider", json={"name": "openai-compat", "model": "m", "base_url": saved_url, "api_key": "sk-lan-key-1234"})

    for spelling in (saved_url, saved_url + "/", saved_url.replace("http://", "HTTP://")):
        r = client.post("/api/provider/models", json={"name": "openai-compat", "base_url": spelling})
        assert r.status_code == 200 and r.json()["key_sent"] == "saved", (spelling, r.text)
    assert at_saved == ["Bearer sk-lan-key-1234"] * 3

    assert client.post("/api/provider/models", json={"name": "openai-compat", "base_url": other_url}).json()["key_sent"] == "none"
    assert at_other == [None]

    # provider ที่ไม่ได้ตั้ง base URL: key ที่บันทึกไว้ไปที่อยู่ทางการของ provider นั้นเท่านั้น
    official, at_official = listener()
    monkeypatch.setattr(providers, "OPENAI_BASE", official)
    client.put("/api/provider", json={"name": "openai", "model": "gpt-4o", "api_key": REAL_KEY})
    assert client.post("/api/provider/models", json={"name": "openai"}).json()["key_sent"] == "saved"
    assert at_official == [f"Bearer {REAL_KEY}"]
    assert client.post("/api/provider/models", json={"name": "openai", "base_url": other_url}).json()["key_sent"] == "none"
    assert at_other == [None, None]
