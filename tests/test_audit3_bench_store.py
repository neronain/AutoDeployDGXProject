"""audit 2026-10-06 — ที่เก็บผลวัด (`~/.lmds/bench/<slug>/`) ลบของคนอื่นได้

`lmds bench remove ../../myproject` ลบ `package.json` กับ `tsconfig.json` ของโฟลเดอร์นั้น แล้ว
พิมพ์ว่า "ลบผลวัดของ ../../myproject ไป 2 รอบ" — `bench/store.py` ต่อ slug เป็น path ดิบ ๆ แล้ว
กวาด `*.json` ทุกไฟล์ที่เจอ · `DELETE /api/bench/{slug}` เรียกฟังก์ชันเดียวกันโดยไม่ตรวจ slug

ทุกเทสรันคำสั่ง/endpoint จริงกับไฟล์จริงใต้ tmp แล้วดูว่าไฟล์ยังอยู่ไหม
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.bench import store
from lmds.cli.main import app
from lmds.fleet import apikey

runner = CliRunner()


@pytest.fixture
def bench_home(tmp_path, monkeypatch):
    """ที่เก็บผลวัดแยกต่อเทส วางไว้ที่ <tmp>/home/.lmds/bench เหมือนของจริงใต้ HOME"""
    root = tmp_path / "home" / ".lmds" / "bench"
    root.mkdir(parents=True)
    monkeypatch.setattr(store, "bench_root", lambda: root)
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "home" / ".lmds" / "keys"))
    return root


def _said(result) -> str:
    """ข้อความที่ผู้ใช้เห็น โดยไม่ขึ้นกับว่า rich ตัดบรรทัดตรงไหน"""
    return " ".join(result.output.split())


def _good_run(root: Path, slug: str, stamp: str = "2026-10-01T09:00:00") -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stamp.replace('-', '').replace(':', '')}.json").write_text(json.dumps({
        "slug": slug, "stamped_at": stamp, "engine": "vllm", "machine": {"hostname": "h"},
        "workloads": [], "probes": [],
    }), encoding="utf-8")


def _someone_elses_project(tmp_path: Path) -> Path:
    project = tmp_path / "home" / "myproject"
    project.mkdir(parents=True, exist_ok=True)
    for name in ("package.json", "tsconfig.json", "keep.txt"):
        (project / name).write_text("{}", encoding="utf-8")
    return project


@pytest.mark.parametrize("argument", ["../../myproject", "..", "m1/../../../myproject", "ABSOLUTE"])
def test_bench_remove_refuses_a_path_and_deletes_nothing(tmp_path, bench_home, argument):
    """`lmds bench remove ../../myproject` ลบ package.json + tsconfig.json แล้วรายงานว่า "ลบผลวัด … 2 รอบ" """
    project = _someone_elses_project(tmp_path)
    (bench_home.parent / "watchdog.json").write_text("{}", encoding="utf-8")   # ของ ~/.lmds เอง
    argument = str(project) if argument == "ABSOLUTE" else argument

    result = runner.invoke(app, ["bench", "remove", "--", argument])

    assert result.exit_code == 2, result.output
    assert "ไม่ถูกต้อง" in _said(result) and "ลบผลวัดของ" not in _said(result)
    assert sorted(p.name for p in project.iterdir()) == ["keep.txt", "package.json", "tsconfig.json"]
    assert (bench_home.parent / "watchdog.json").exists()


@pytest.mark.parametrize("command", [["bench", "show"], ["agent", "bench", "--slug"]])
def test_the_read_commands_refuse_a_path_too(tmp_path, bench_home, command):
    project = _someone_elses_project(tmp_path)
    (project / "package.json").write_text('{"secret": "token-from-another-project"}', encoding="utf-8")
    result = runner.invoke(app, [*command, "../../myproject"])
    assert result.exit_code == 2, result.output
    assert "token-from-another-project" not in result.output


def test_remove_only_deletes_the_result_files_the_store_wrote(bench_home):
    """แม้ slug ถูกต้อง ก็ลบเฉพาะไฟล์รูป <ตราเวลา>.json — ไฟล์อื่นในโฟลเดอร์ไม่ใช่ของเรา"""
    _good_run(bench_home, "m1")
    (bench_home / "m1" / "notes.json").write_text("{}", encoding="utf-8")
    (bench_home / "m1" / "package.json").write_text("{}", encoding="utf-8")

    assert [p.name for p in store.runs_for("m1")] == ["20261001T090000.json"]
    assert store.remove("m1") == 1
    assert sorted(p.name for p in (bench_home / "m1").iterdir()) == ["notes.json", "package.json"]


def test_a_slug_folder_that_is_a_symlink_out_of_the_store_is_not_followed(tmp_path, bench_home):
    """ชื่อถูกรูปแบบแต่โฟลเดอร์ชี้ออกไปข้างนอก — ที่เก็บนี้ถูกก๊อปข้ามเครื่องได้ จึงไม่ใช่ของที่เราสร้างเองเสมอ"""
    project = _someone_elses_project(tmp_path)
    (project / "20261001T090000.json").write_text("{}", encoding="utf-8")
    (bench_home / "evil").symlink_to(project, target_is_directory=True)

    with pytest.raises(store.BenchStoreError):
        store.remove("evil")
    with pytest.raises(store.BenchStoreError):
        store.runs_for("evil")
    assert (project / "20261001T090000.json").exists()
    assert store.all_runs() == [], "ตารางคะแนนต้องข้ามโฟลเดอร์แบบนี้ ไม่ใช่ล้มทั้งตาราง"

    result = runner.invoke(app, ["bench", "remove", "evil"])
    assert result.exit_code == 2 and (project / "20261001T090000.json").exists()


@pytest.mark.parametrize("slug", ["../x", "..", ".", "", "a/b", "ABSOLUTE", "a b", "-rf", ".hidden", "ไทย", "a\n"])
def test_every_store_function_that_takes_a_slug_refuses_a_bad_one(tmp_path, bench_home, slug):
    # path เต็มต้องอยู่ใต้ tmp เสมอ — เทสนี้ถูกรันกับโค้ดก่อนแก้ด้วย ซึ่งจะลบ *.json ใน path นั้นจริง
    slug = str(tmp_path / "home" / "elsewhere") if slug == "ABSOLUTE" else slug
    for call in (lambda: store.runs_for(slug), lambda: store.latest_merged(slug),
                 lambda: store.remove(slug),
                 lambda: store.record(slug, "org/m", "vllm", "m", [], [], {}, "2026-10-01T09:00:00")):
        with pytest.raises(store.BenchStoreError):
            call()
    assert list(bench_home.parent.parent.rglob("*.json")) == [], "ต้องไม่มีอะไรถูกเขียนออกไป"


def test_the_store_accepts_exactly_the_names_the_rest_of_lmds_calls_a_slug(bench_home):
    """กติกาเดียวกับที่เก็บ key (fleet/apikey) — ชื่อที่ deploy ตั้งให้ต้องเก็บผลวัดได้ทุกชื่อ"""
    for name in ("qwen3-32b", "Qwen3.6-35B-A3B", "gemma_4-31b.it", "a", "x" * 64):
        assert store.check_slug(name) == name
        assert apikey.path_for(name).name == name
    for name in ("../x", "a/b", "", ".x", "-x"):
        with pytest.raises(store.BenchStoreError):
            store.check_slug(name)
        with pytest.raises(apikey.ApiKeyError):
            apikey.path_for(name)
    # bundle ที่สร้างก่อนมีกติกาความยาว 64 ยังมีผลวัดอยู่จริง — ต้องอ่าน/ลบได้
    assert store.check_slug("x" * 120)


def test_a_result_with_an_odd_timestamp_is_refused_rather_than_written_where_remove_cannot_see_it(bench_home):
    with pytest.raises(store.BenchStoreError):
        store.record("m1", "org/m", "vllm", "m", [], [], {}, "../../escape")
    assert not (bench_home / "m1").exists() or list((bench_home / "m1").iterdir()) == []


# ── หน้าเว็บ: ปุ่มลบ/ดูผลวัดเรียกฟังก์ชันชุดเดียวกัน ─────────────────────────────────
def test_the_web_delete_endpoint_validates_the_slug_before_touching_the_disk(tmp_path, bench_home):
    """`DELETE /api/bench/{slug}` เรียก remove() ตรง ๆ โดยไม่ผ่าน _check_slug — `..` ลบ *.json ใน ~/.lmds"""
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    victim = bench_home.parent / "watchdog.json"            # ~/.lmds/watchdog.json
    victim.write_text("{}", encoding="utf-8")
    _good_run(bench_home, "m1")
    client = TestClient(create_app())

    for path in ("/api/bench/%2E%2E", "/api/bench/.hidden", "/api/bench/a%20b"):
        assert client.delete(path).status_code == 400, path
        assert client.get(path).status_code == 400, path
    assert victim.exists()

    assert client.get("/api/bench/m1").json()["run"]["slug"] == "m1"
    assert client.delete("/api/bench/m1").json() == {"slug": "m1", "removed": 1}
    assert client.post("/api/bench/%2E%2E/run").status_code == 400
