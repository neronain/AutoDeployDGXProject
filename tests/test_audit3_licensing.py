"""audit 2026-10-06 — ไฟล์ไลเซนส์ที่ผิดรูปต้องเป็นสถานะ `invalid` ไม่ใช่ traceback

`store.load()` สัญญาในบรรทัดแรกของ docstring ว่า "ไม่เคยโยน exception ออกไปหาผู้เรียก" เพราะ
ทุกคำสั่งเรียกมันโดยไม่ดักอะไร · แต่สองรูปหลุดออกไปได้:

    machines: .inf     → int(inf)        → OverflowError   (`lmds license show` ตาย)
    features: 5        → for f in 5      → TypeError       (`lmds license install` ตาย)

ทั้งคู่ไม่ใช่ ValueError จึงหลุด `except ValueError` ของ verify_document · และเพราะ
`lmds node install` ถามไลเซนส์ก่อนทุกครั้ง (nodes/ssh.py `_require_licence_for`) ไฟล์ที่พังแบบนี้
จึงบล็อกการติดตั้ง/อัปเดตทั้งฟลีตด้วย traceback — ตรงข้ามกับกฎข้อ 2 ของ docs/LICENSING.md เป๊ะ

เทสรันผ่านทางที่ผู้ใช้เดินจริง: ไฟล์บนดิสก์ → `store.load()` / คำสั่ง `lmds license …` /
`/api/license` — ไม่มีเทสไหนยืนยันด้วยข้อความในซอร์ส
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.licensing import keys, store
from tests.test_licensing import _issue

runner = CliRunner()


@pytest.fixture(autouse=True)
def _accept_the_test_key(monkeypatch):
    monkeypatch.setattr(keys, "ACTIVE", frozenset({"k1", "k2", "test"}))


def _good(tmp_path, **overrides) -> str:
    """ใบที่เซ็นถูกต้องและยังไม่หมดอายุ ณ วันที่รันเทส"""
    overrides.setdefault("expires", date.today() + timedelta(days=30))
    text, _ = _issue(tmp_path, **overrides)
    return text


def _mutations(good: str) -> dict[str, str]:
    """ไฟล์ผิดรูปทุกแบบที่นึกออก — สองแบบแรกคือที่ audit เจอว่าหลุดเป็น exception"""
    signature = good[good.index("signature:"):]
    licence = good[:good.index("signature:")]
    return {
        "machines: .inf": good.replace("machines: 8", "machines: .inf"),
        "features: 5": good.replace("  id: LMDS-TEST-1", "  id: LMDS-TEST-1\n  features: 5"),
        "machines: -.inf": good.replace("machines: 8", "machines: -.inf"),
        "machines: .nan": good.replace("machines: 8", "machines: .nan"),
        "machines: 1e400": good.replace("machines: 8", "machines: 1e400"),
        "machines: [8]": good.replace("machines: 8", "machines: [8]"),
        "machines: {n: 8}": good.replace("machines: 8", "machines: {n: 8}"),
        "machines: null": good.replace("machines: 8", "machines: null"),
        "machines: yes": good.replace("machines: 8", "machines: yes"),
        "machines: eight": good.replace("machines: 8", "machines: eight"),
        "features: a string": good.replace("  id: LMDS-TEST-1", "  id: LMDS-TEST-1\n  features: everything"),
        "features: {a: 1}": good.replace("  id: LMDS-TEST-1", "  id: LMDS-TEST-1\n  features: {a: 1}"),
        "id is a mapping": good.replace("id: LMDS-TEST-1", "id: {a: 1}"),
        "issued is a list": good.replace("issued: '2026-01-01'", "issued: [2026, 1, 1]"),
        "issued is a datetime": good.replace("issued: '2026-01-01'", "issued: 2026-01-01 00:00:00"),
        "expires is a number": licence.replace("expires: '", "expires: 12345 # '") + signature,
        "expires is a mapping": licence.replace("expires: '", "expires: {y: 2099} # '") + signature,
        "license is a list": "license: [1, 2]\n" + signature,
        "license is a string": "license: nope\n" + signature,
        "license is missing": signature,
        "signature is a string": licence + "signature: nope\n",
        "signature is a list": licence + "signature: [k1, AAAA]\n",
        "signature is a number": licence + "signature: 5\n",
        "signature.value is a list": licence + "signature: {key: test, value: [1, 2]}\n",
        "signature.value is a number": licence + "signature: {key: test, value: 5}\n",
        "signature.key is a mapping": licence + "signature: {key: {a: 1}, value: AAAA}\n",
        "signature too short": licence + "signature: {key: test, value: AAAA}\n",
        "top-level list": "- a\n- b\n",
        "top-level scalar": "just a sentence\n",
        "empty file": "",
        "only a comment": "# nothing here\n",
        "binary garbage": "\x00\x01\x02",
        "not yaml": "license: [unclosed\n",
        "yaml alias bomb": "a: &a [x, x]\nlicense: [*a, *a]\nsignature: {key: test, value: AAAA}\n",
    }


_NAMES = list(_mutations("license:\n  id: LMDS-TEST-1\n  machines: 8\n  issued: '2026-01-01'\n"
                         "  expires: '2027-01-01'\nsignature:\n  key: test\n  value: AAAA\n"))


@pytest.mark.parametrize("name", _NAMES)
def test_a_malformed_licence_file_is_invalid_with_a_reason_never_an_exception(tmp_path, name):
    """`load()` สัญญาว่าไม่เคยโยน — ทุกรูปต้องได้สถานะ invalid + เหตุผลที่คนอ่านได้ + สิทธิ์เท่าโหมดฟรี"""
    good = _good(tmp_path)
    text = _mutations(good)[name]
    assert text != good, f"เทสนี้ล้าสมัย: การแก้ไฟล์แบบ {name!r} ไม่ได้เปลี่ยนอะไร"
    path = tmp_path / "license.yaml"
    path.write_text(text, encoding="utf-8")

    status = store.load(path)                    # ต้องไม่โยนอะไรออกมาเลย

    assert status.state == "invalid", (name, status)
    assert status.reason.strip(), "invalid ต้องบอกเหตุผลเสมอ"
    assert "Traceback" not in status.reason
    assert status.machines_allowed == 1 and not status.unlimited and not status.read_only
    assert status.describe()                     # สิ่งที่ `license show` และ /api/license เอาไปโชว์


def test_the_reason_names_the_field_that_is_wrong(tmp_path):
    """"ไลเซนส์ใช้ไม่ได้" เฉย ๆ ไม่ช่วยใคร — ลูกค้าต้องรู้ว่าบรรทัดไหนในไฟล์"""
    cases = _mutations(_good(tmp_path))
    path = tmp_path / "license.yaml"
    for name, field in (("machines: .inf", "machines"), ("features: 5", "features"),
                        ("issued is a datetime", "issued"), ("expires is a mapping", "expires")):
        path.write_text(cases[name], encoding="utf-8")
        assert field in store.load(path).reason, name


def test_a_signed_licence_whose_expiry_is_a_yaml_datetime_does_not_crash(tmp_path):
    """`expires: 2099-01-01T00:00:00` — YAML อ่านเป็น datetime ซึ่งเป็นคลาสลูกของ date

    ต้องเป็นใบที่ **เซ็นทั้งที่เป็น datetime** ถึงจะไปถึงจุดที่พัง: ถ้าแค่แก้ไฟล์เติมเวลาเข้าไป ไบต์ที่
    เซ็นเปลี่ยน ลายเซ็นไม่ผ่านก่อน (ได้ invalid ตามปกติ) · แต่ใบที่ผู้ออกเซ็นจาก datetime ผ่าน
    isinstance(value, date) เข้าไปได้ ลายเซ็นตรง แล้วตายที่ `today > expires` — เทียบ date กับ
    datetime ไม่ได้ → TypeError กลาง `lmds license show` และ `lmds node install`
    """
    from datetime import datetime

    text, _ = _issue(tmp_path, expires=datetime(2099, 1, 1))
    assert "expires: '2099-01-01T00:00:00'" in text, "เทสนี้ล้าสมัย: รูปของบรรทัด expires เปลี่ยนไป"
    path = tmp_path / "license.yaml"
    path.write_text(text.replace("expires: '2099-01-01T00:00:00'", "expires: 2099-01-01T00:00:00"),
                    encoding="utf-8")

    status = store.load(path)

    assert status.state == "invalid"
    assert "expires" in status.reason and "YYYY-MM-DD" in status.reason

    # วันที่แบบไม่มี quote (รูปที่ docs/LICENSING.md ใช้เป็นตัวอย่าง) ยังต้องใช้ได้ตามเดิม
    good = _good(tmp_path)
    expiry = (date.today() + timedelta(days=30)).isoformat()
    assert f"expires: '{expiry}'" in good
    path.write_text(good.replace(f"expires: '{expiry}'", f"expires: {expiry}"), encoding="utf-8")
    assert store.load(path).state == "active"


def test_the_shapes_that_used_to_work_still_do(tmp_path):
    """กันแก้เกิน: ใบจริงที่เคยตรวจผ่านต้องยังผ่าน — ไฟล์พังไม่ควรทำให้ของลูกค้าที่จ่ายแล้วใช้ไม่ได้"""
    path = tmp_path / "license.yaml"
    for overrides in ({}, {"machines": 0}, {"expires": None}, {"features": ["sso", "audit-export"]},
                      {"contact": "ops@example.com", "note": "ต่ออายุรอบสอง"}):
        text, lic = _issue(tmp_path, **overrides)
        path.write_text(text, encoding="utf-8")
        status = store.load(path, today=date(2026, 6, 1))
        assert status.state == "active", (overrides, status.reason)
        assert status.license == lic


# ── ผ่านคำสั่งจริง ───────────────────────────────────────────────────────────────────
def _licence_file(isolated_config):
    isolated_config.mkdir(parents=True, exist_ok=True)
    return isolated_config / "license.yaml"


@pytest.mark.parametrize("name", ["machines: .inf", "features: 5", "issued is a datetime", "binary garbage"])
def test_license_show_reports_a_broken_file_instead_of_dying(tmp_path, isolated_config, name):
    """ก่อนแก้: `lmds license show` → OverflowError: cannot convert float infinity to integer"""
    _licence_file(isolated_config).write_text(_mutations(_good(tmp_path))[name], encoding="utf-8")

    result = runner.invoke(app, ["license", "show"])

    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output
    assert "invalid" in result.output


@pytest.mark.parametrize("name", ["features: 5", "machines: .inf", "license is a list", "top-level list"])
def test_license_install_refuses_a_bad_file_and_keeps_the_licence_that_was_there(tmp_path, isolated_config, name):
    """ก่อนแก้: `lmds license install bad.yaml` → TypeError: 'int' object is not iterable

    เคสที่กลัว: ลูกค้าวางใบต่ออายุที่พังทับใบเดิมที่ยังดีอยู่ แล้วทั้งฟลีตตกเป็นโหมดฟรี
    """
    good = _good(tmp_path)
    installed = _licence_file(isolated_config)
    assert runner.invoke(app, ["license", "install", str(_write(tmp_path / "good.yaml", good))]).exit_code == 0
    before = installed.read_bytes()

    bad = _write(tmp_path / "bad.yaml", _mutations(good)[name])
    result = runner.invoke(app, ["license", "install", str(bad)])

    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2, result.output
    assert "ไม่ได้เขียนทับของเดิม" in " ".join(result.output.split())
    assert installed.read_bytes() == before, "ใบเดิมต้องไม่ถูกแตะแม้แต่ไบต์เดียว"
    assert store.load(installed).state == "active"
    assert sorted(p.name for p in installed.parent.iterdir() if "license" in p.name) == ["license.yaml"]


def test_license_install_refuses_a_file_that_is_not_text(tmp_path, isolated_config):
    """ก๊อปผิดไฟล์ (zip/รูป) — เดิมเป็น traceback ของ UnicodeDecodeError"""
    binary = tmp_path / "license.yaml.zip"
    binary.write_bytes(b"PK\x03\x04\xff\xfe\x00garbage")
    result = runner.invoke(app, ["license", "install", str(binary)])
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2
    assert not _licence_file(isolated_config).exists()


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_a_broken_licence_file_does_not_block_installing_a_node(tmp_path, isolated_config):
    """`lmds node install` ถามไลเซนส์ก่อนทุกครั้ง — ไฟล์พังต้องตกเป็นสิทธิ์โหมดฟรี ไม่ใช่ traceback

    เรียกจุดบังคับใช้จริง (nodes/ssh.py) กับทะเบียนว่าง: เครื่องแรกอยู่ในสิทธิ์โหมดฟรีเสมอ
    """
    from lmds.nodes import Node, ssh

    _licence_file(isolated_config).write_text(_mutations(_good(tmp_path))["machines: .inf"], encoding="utf-8")
    ssh._require_licence_for(Node(name="gpu-1", host="192.0.2.10", user="u"))     # ต้องไม่โยน


# ── หน้าเว็บเดินทางเดียวกัน ──────────────────────────────────────────────────────────
def test_the_web_licence_endpoints_survive_a_broken_file(tmp_path, isolated_config):
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    cases = _mutations(_good(tmp_path))
    installed = _licence_file(isolated_config)
    client = TestClient(create_app())

    installed.write_text(cases["machines: .inf"], encoding="utf-8")
    shown = client.get("/api/license")
    assert shown.status_code == 200 and shown.json()["state"] == "invalid"
    assert shown.json()["machines_allowed"] == 1

    good = _good(tmp_path)
    assert client.post("/api/license", json={"text": good}).status_code == 200
    before = installed.read_bytes()
    refused = client.post("/api/license", json={"text": _mutations(good)["features: 5"]})
    assert refused.status_code == 400 and "features" in refused.json()["detail"]
    assert installed.read_bytes() == before
