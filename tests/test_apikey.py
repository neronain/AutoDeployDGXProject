"""API key ของ model server ที่เก็บกับเครื่อง — ~/.lmds/keys/<slug>

ทำไมต้องมีไฟล์นี้: key ไม่อยู่ใน bundle.env (โฟลเดอร์ bundle ถูก zip แจกต่อได้) ซึ่งถูก
แต่เดิมไม่ได้เก็บไว้ที่ไหนเลย — หน้าเว็บส่ง API_KEY= ไปกับการกด start ครั้งเดียว ส่วน
systemd unit เรียก `<controller> start` เปล่า ๆ · reboot แล้วโมเดลกลับมาเปิดโล่งบน
0.0.0.0 เงียบ ๆ ทั้งที่ผู้ใช้ตั้ง key ไว้แล้ว
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.fleet import apikey

runner = CliRunner()


@pytest.fixture(autouse=True)
def key_root(tmp_path, monkeypatch):
    root = tmp_path / "keys"
    monkeypatch.setenv("LMDS_KEY_ROOT", str(root))
    return root


def test_a_key_is_written_only_readable_by_its_owner(key_root):
    """คนอื่นบนเครื่องเดียวกันอ่าน key ของเราไม่ได้ — ทั้งไฟล์และโฟลเดอร์"""
    path = apikey.write("demo", "s3cret-value")
    assert path == key_root / "demo"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert apikey.read("demo") == "s3cret-value"


def test_a_minted_key_is_long_and_has_no_characters_that_break_a_copy_paste():
    key = apikey.mint()
    assert len(key) >= 32 and key.isalnum() and key == key.strip()
    assert apikey.mint() != key, "ต้องสุ่มใหม่ทุกครั้ง"


def test_a_slug_cannot_walk_out_of_the_key_folder():
    """slug มาจาก URL ของหน้าเว็บและ argv — ../ ต้องไม่พาไปเขียนที่อื่น"""
    for bad in ("../escape", "a/b", "/etc/passwd", "", ".", ".."):
        with pytest.raises(apikey.ApiKeyError):
            apikey.path_for(bad)


def test_reading_a_key_that_is_not_there_is_not_an_error():
    """ไม่มี key = "เสิร์ฟแบบเปิด" ซึ่งเป็นสถานะที่ถูกต้อง ไม่ใช่ระบบพัง"""
    assert apikey.read("never-set") == ""
    assert apikey.read("../nonsense") == ""
    assert apikey.clear("never-set") is False


def test_an_empty_or_multiline_key_is_refused():
    """เขียนค่าว่างลงไป = controller อ่านได้ key ว่างแล้วรันแบบไม่มี auth โดยที่ไฟล์มีอยู่"""
    for bad in ("", "   ", "\n"):
        with pytest.raises(apikey.ApiKeyError):
            apikey.write("demo", bad)
    with pytest.raises(apikey.ApiKeyError):
        apikey.write("demo", "line-one\nline-two")


def test_overwriting_a_key_never_leaves_the_file_half_written(key_root):
    """เขียนทับผ่าน temp + replace — controller ที่ start พอดีช่วงนั้นต้องไม่เจอไฟล์ว่าง"""
    apikey.write("demo", "first-key")
    apikey.write("demo", "second-key")
    assert apikey.read("demo") == "second-key"
    leftovers = [p.name for p in key_root.iterdir() if p.name.startswith(".")]
    assert not leftovers, f"ไฟล์ชั่วคราวค้าง: {leftovers}"


def test_listing_says_which_bundles_have_a_key_without_revealing_any(key_root):
    apikey.write("alpha", "key-a")
    apikey.write("beta", "key-b")
    found = apikey.listing()
    assert found == {"alpha": True, "beta": True}
    assert "key-a" not in str(found)


def test_the_listing_ignores_temporary_and_hidden_files(key_root):
    """ไฟล์ชั่วคราวของ write() ขึ้นต้นด้วยจุด — ต้องไม่โผล่เป็น bundle ปลอม"""
    apikey.write("real", "key")
    (key_root / ".real.new").write_text("half written", encoding="utf-8")
    (key_root / ".DS_Store").write_bytes(b"\0")
    assert set(apikey.listing()) == {"real"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_the_cli_never_prints_the_whole_key_unless_asked(key_root):
    apikey.write("demo", "abcdefgh12345678")
    shown = runner.invoke(app, ["key", "show", "demo"])
    assert shown.exit_code == 0
    assert "abcdefgh12345678" not in shown.stdout, "ค่าเต็มต้องไม่หลุดออกมาเอง"
    assert "--reveal" in shown.stdout

    revealed = runner.invoke(app, ["key", "show", "demo", "--reveal"])
    assert "abcdefgh12345678" in revealed.stdout


def test_show_exits_non_zero_when_there_is_no_key(key_root):
    """สคริปต์ของลูกค้าเช็กได้ว่า bundle นี้ปิดอยู่จริงไหมโดยไม่ต้องอ่านข้อความ"""
    result = runner.invoke(app, ["key", "show", "demo"])
    assert result.exit_code == 1 and "ไม่มี key" in result.stdout


def test_new_refuses_to_silently_replace_an_existing_key(key_root):
    apikey.write("demo", "in-use-by-clients")
    blocked = runner.invoke(app, ["key", "new", "demo"])
    assert blocked.exit_code == 1 and apikey.read("demo") == "in-use-by-clients"

    forced = runner.invoke(app, ["key", "new", "demo", "--force"])
    assert forced.exit_code == 0 and apikey.read("demo") != "in-use-by-clients"


def test_set_takes_the_key_on_stdin_not_on_the_command_line(key_root):
    """`ps` อ่าน argv ของทุก process บนเครื่องได้ และ shell เก็บบรรทัดคำสั่งลง history —
    เหตุผลเดียวกับที่ controller ส่ง key ทาง --api-key-file ไม่ใช่ --api-key"""
    result = runner.invoke(app, ["key", "set", "demo"], input="key-from-stdin\n")
    assert result.exit_code == 0 and apikey.read("demo") == "key-from-stdin"

    empty = runner.invoke(app, ["key", "set", "demo"], input="")
    assert empty.exit_code == 1


def test_clear_says_plainly_that_the_endpoint_becomes_open(key_root):
    apikey.write("demo", "key")
    result = runner.invoke(app, ["key", "clear", "demo"])
    assert result.exit_code == 0 and apikey.read("demo") == ""
    # คำเตือนไปทาง stderr โดยตั้งใจ — สคริปต์ที่ pipe stdout ต่อจะได้ไม่กลืนมันไป
    assert "เสิร์ฟแบบเปิด" in (result.stdout + (result.stderr or ""))


def test_the_key_root_follows_the_home_of_the_user_running_the_controller(monkeypatch, tmp_path):
    """systemd system-scope ตั้ง HOME ให้ unit เอง — ที่เก็บต้องเดินตาม ไม่ใช่ฝัง path ตอน generate"""
    monkeypatch.delenv("LMDS_KEY_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "someone"))
    assert apikey.key_root() == tmp_path / "someone" / ".lmds" / "keys"


def test_the_store_is_not_inside_any_bundle_folder(monkeypatch, tmp_path):
    """โฟลเดอร์ bundle ถูก zip แจกต่อได้ — key ที่หลุดเข้าไปคือ key ที่แจกออกไปด้วย"""
    monkeypatch.delenv("LMDS_KEY_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert "bundles" not in apikey.key_root().parts
    assert apikey.key_root() == tmp_path / ".lmds" / "keys"
    assert os.environ.get("LMDS_KEY_ROOT") is None
