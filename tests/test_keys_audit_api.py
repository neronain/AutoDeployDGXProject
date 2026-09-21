"""key ของโมเดลและ audit ต้องใช้ได้จากคอนโซล ไม่ใช่มีแต่ทาง CLI

ลูกค้าที่เริ่มเอาไปทดสอบใช้งานผ่าน GUI เป็นหลัก · `lmds key` กับ `lmds audit` มีแต่ทาง
CLI คนกลุ่มนั้นจึงตั้ง key ไม่ได้และดูร่องรอยไม่ได้เลย — ทั้งที่เป็นงานความปลอดภัยที่
ทำมาเพื่อพวกเขาโดยตรง · ที่ย้อนแย้งที่สุดคือ wizard deploy บนหน้าเว็บพิมพ์ข้อความบอกให้
ไปรัน `lmds key show <slug> --reveal` ซึ่งเป็นคำสั่งที่คนกลุ่มนั้นพิมพ์ไม่ได้อยู่แล้ว
"""

from __future__ import annotations

import stat

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lmds.web import create_app  # noqa: E402

TOKEN = "tok-abcdefgh"
AUTH = {"x-lmds-token": TOKEN}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.delenv("LMDS_AUDIT", raising=False)
    return TestClient(create_app(TOKEN))


# ── key ───────────────────────────────────────────────────────────────────────

def test_status_says_whether_there_is_a_key_without_handing_it_over(client):
    """หน้าเว็บ poll สถานะตลอด — คืนค่าเต็มทุกครั้งแปลว่า key เดินทางผ่านเครือข่าย
    และไปนอนใน memory ของเบราว์เซอร์ตลอดเวลาโดยไม่มีใครขอ"""
    created = client.post("/api/models/demo/key", headers=AUTH, json={}).json()["key"]

    body = client.get("/api/models/demo/key", headers=AUTH).json()
    assert body["has_key"] is True
    assert "key" not in body, "ค่าเต็มต้องไม่ติดมากับสถานะ"
    assert created not in str(body)
    assert body["hint"].startswith(created[:4]) and body["hint"].endswith(created[-4:])


def test_revealing_the_key_is_its_own_endpoint(client):
    client.post("/api/models/demo/key", headers=AUTH, json={})
    full = client.get("/api/models/demo/key/reveal", headers=AUTH).json()["key"]
    assert len(full) >= 32
    assert client.get("/api/models/never-set/key/reveal", headers=AUTH).status_code == 404


def test_creating_a_key_returns_it_once_and_says_a_restart_is_needed(client):
    body = client.post("/api/models/demo/key", headers=AUTH, json={}).json()
    assert body["generated"] is True and len(body["key"]) >= 32
    assert body["restart_required"] is True, "ตัวที่รันอยู่ยังใช้ key เดิมจนกว่าจะ restart"


def test_an_existing_key_is_not_overwritten_by_accident(client):
    first = client.post("/api/models/demo/key", headers=AUTH, json={}).json()["key"]

    clash = client.post("/api/models/demo/key", headers=AUTH, json={})
    assert clash.status_code == 409, "ทับโดยไม่ยืนยัน = client ทุกตัวที่ถือใบเก่าพังพร้อมกัน"
    assert client.get("/api/models/demo/key/reveal", headers=AUTH).json()["key"] == first

    forced = client.post("/api/models/demo/key", headers=AUTH, json={"force": True}).json()
    assert forced["key"] != first


def test_a_key_the_operator_already_has_can_be_set(client):
    body = client.post("/api/models/demo/key", headers=AUTH,
                       json={"key": "brought-from-elsewhere"}).json()
    assert body["generated"] is False
    assert client.get("/api/models/demo/key/reveal", headers=AUTH).json()["key"] == "brought-from-elsewhere"


def test_clearing_says_plainly_that_the_endpoint_becomes_open(client):
    client.post("/api/models/demo/key", headers=AUTH, json={})
    body = client.request("DELETE", "/api/models/demo/key", headers=AUTH).json()
    assert body["removed"] is True and "เสิร์ฟแบบเปิด" in body["warning"]
    assert client.get("/api/models/demo/key", headers=AUTH).json()["has_key"] is False


def test_the_key_file_written_through_the_api_is_owner_only(client, tmp_path):
    """ทางเว็บกับทาง CLI ต้องเขียนไฟล์แบบเดียวกัน ไม่ใช่ทางหนึ่งเข้มอีกทางหลวม"""
    client.post("/api/models/demo/key", headers=AUTH, json={})
    path = tmp_path / "keys" / "demo"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_slug_cannot_walk_out_of_the_key_folder(client):
    """slug มาจาก URL — ../ ต้องไม่พาไปเขียนที่อื่น (ใช้ด่านเดียวกับ route อื่นทั้งระบบ)"""
    for bad in ("..%2F..%2Fetc", "a%2Fb", "x';id;'"):
        assert client.get(f"/api/models/{bad}/key", headers=AUTH).status_code in (400, 404)


def test_every_key_route_needs_the_token(client):
    assert client.get("/api/models/demo/key").status_code == 401
    assert client.get("/api/models/demo/key/reveal").status_code == 401
    assert client.post("/api/models/demo/key", json={}).status_code == 401
    assert client.request("DELETE", "/api/models/demo/key").status_code == 401


# ── audit ─────────────────────────────────────────────────────────────────────

def test_audit_shows_what_changed_state(client):
    client.post("/api/models/demo/key", headers=AUTH, json={})
    body = client.get("/api/audit", headers=AUTH).json()
    assert body["enabled"] is True and body["path"].endswith("audit.log")
    assert any(e["path"] == "/api/models/demo/key" and e["method"] == "POST"
               for e in body["entries"])


def test_audit_can_show_only_what_was_refused(client):
    client.post("/api/models/demo/key", headers=AUTH, json={})   # ผ่าน
    client.get("/api/models/demo/key")                            # 401

    every = client.get("/api/audit", headers=AUTH).json()
    refused = client.get("/api/audit?failed_only=true", headers=AUTH).json()
    assert refused["refused"] >= 1
    assert all(int(e["status"]) in {401, 403, 429} for e in refused["entries"])
    assert len(refused["entries"]) < len(every["entries"])


def test_the_audit_never_carries_the_token(client):
    """`require_token` รับ token ทาง ?token= ได้ — audit ที่เก็บ query string
    คือการเขียน token ลงไฟล์ที่มีไว้ให้คนอ่าน"""
    client.post(f"/api/models/demo/key?token={TOKEN}", json={})
    assert TOKEN not in str(client.get("/api/audit", headers=AUTH).json())


def test_audit_needs_the_token_too(client):
    assert client.get("/api/audit").status_code == 401
