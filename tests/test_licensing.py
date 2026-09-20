"""ระบบไลเซนส์ — ลายเซ็น · การนับเครื่อง · และ **กฎที่ห้ามละเมิด**

เทสในไฟล์นี้แบ่งเป็นสองพวก:

1. พวกที่ตรวจว่าโค้ดทำงานถูก (ลายเซ็น, วันหมดอายุ, การนับ)
2. พวกที่ตรวจว่า **เราไม่ได้ล็อกในสิ่งที่สัญญาว่าจะไม่ล็อก** — พวกนี้สำคัญกว่า
   เพราะบั๊กประเภทนั้นไม่ทำให้อะไรพัง มันแค่ทำให้เราผิดคำพูดกับลูกค้าอย่างเงียบ ๆ
"""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pytest

from lmds.licensing import ed25519, enforce, keys, seats, store
from lmds.licensing.model import FREE_SERVING_MACHINES, License

ROOT = Path(__file__).resolve().parents[1]

# RFC 8032 §7.1 — ชุดทดสอบทางการของ Ed25519
RFC_8032 = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
]


@pytest.mark.parametrize("secret,public,message,signature", RFC_8032)
@pytest.mark.parametrize("force_pure", [False, True])
def test_ed25519_matches_the_rfc_test_vectors(secret, public, message, signature, force_pure, monkeypatch):
    """ทั้งสองทาง (cryptography และตัวที่เราแปะเอง) ต้องให้ผลตรง RFC เป๊ะ

    ถ้าทางใดทางหนึ่งเพี้ยน ลูกค้าครึ่งหนึ่งจะตรวจใบผ่านและอีกครึ่งไม่ผ่าน ขึ้นกับว่า
    เครื่องนั้นบังเอิญมี cryptography ลงไว้ไหม — อาการที่หาสาเหตุยากที่สุดแบบหนึ่ง
    """
    if force_pure:
        monkeypatch.setattr(ed25519, "HAVE_CRYPTOGRAPHY", False)
    elif not ed25519.HAVE_CRYPTOGRAPHY:
        pytest.skip("เครื่องนี้ไม่มี cryptography — ทางนั้นถูกทดสอบที่อื่น")

    secret_b, public_b = bytes.fromhex(secret), bytes.fromhex(public)
    message_b, signature_b = bytes.fromhex(message), bytes.fromhex(signature)

    assert ed25519.public_from_secret(secret_b) == public_b
    assert ed25519.sign(secret_b, message_b) == signature_b
    assert ed25519.verify(public_b, message_b, signature_b)
    assert not ed25519.verify(public_b, message_b + b"\x00", signature_b)


def test_a_malformed_signature_is_false_not_an_exception():
    """ไฟล์พังต้องได้คำตอบว่า "ไม่ผ่าน" ไม่ใช่ traceback กลางคำสั่งที่ผู้ใช้กำลังทำอย่างอื่น"""
    assert ed25519.verify(b"", b"x", b"") is False
    assert ed25519.verify(b"\x00" * 32, b"x", b"\x00" * 64) is False
    assert ed25519.verify(b"short", b"x", b"also-short") is False


def test_the_two_backends_never_disagree():
    """ทั้งสองทางต้องตอบเหมือนกันทุกอินพุต — รวมอินพุตพิลึกที่ไม่มีใครตั้งใจส่งมา

    เทสนี้เกิดจากบั๊กจริง 2026-09-20: public key ศูนย์ 32 ไบต์ + ลายเซ็นศูนย์ 64 ไบต์
    ผ่านบนทางที่เราแปะเอง แต่ `cryptography` ปฏิเสธ — เครื่อง air-gapped (ซึ่งใช้ทางนั้น
    พอดี) จะยอมรับไฟล์ที่เครื่องอื่นในฟลีตเดียวกันปฏิเสธ · ความต่างแบบนี้ต้องจับให้ได้
    อัตโนมัติ ไม่ใช่รอให้คนสังเกต
    """
    if not ed25519.HAVE_CRYPTOGRAPHY:
        pytest.skip("เครื่องนี้ไม่มี cryptography — เทียบสองทางไม่ได้")

    import secrets as _secrets

    cases: list[tuple[bytes, bytes, bytes]] = [
        (b"\x00" * 32, b"x", b"\x00" * 64),                  # จุด small-order ล้วน
        (b"\x01" + b"\x00" * 31, b"x", b"\x00" * 64),
        (b"\xff" * 32, b"x", b"\xff" * 64),                  # y ใหญ่เกิน p
        (b"", b"x", b""),
        (b"\x00" * 31, b"x", b"\x00" * 63),                  # ความยาวผิด
    ]
    # ใบจริงหนึ่งใบ + ใบที่ถูกแก้ไปทีละไบต์
    secret = _secrets.token_bytes(32)
    public = ed25519.public_from_secret(secret)
    good = ed25519.sign(secret, b"payload")
    cases.append((public, b"payload", good))
    for index in (0, 31, 32, 63):
        broken = bytearray(good)
        broken[index] ^= 0x01
        cases.append((public, b"payload", bytes(broken)))
    for _ in range(10):
        cases.append((_secrets.token_bytes(32), _secrets.token_bytes(8), _secrets.token_bytes(64)))

    for public_key, message, signature in cases:
        with_lib = ed25519.verify(public_key, message, signature)
        saved = ed25519.HAVE_CRYPTOGRAPHY
        try:
            ed25519.HAVE_CRYPTOGRAPHY = False
            pure = ed25519.verify(public_key, message, signature)
        finally:
            ed25519.HAVE_CRYPTOGRAPHY = saved
        assert with_lib == pure, (
            f"สองทางตอบไม่ตรงกัน: cryptography={with_lib} pure={pure}\n"
            f"  key={public_key.hex()[:32]}… sig={signature.hex()[:32]}…")


def _issue(tmp_path, **overrides) -> tuple[str, License]:
    """ออกใบจริงด้วยกุญแจชั่วคราว แล้วแทน public key ที่ฝังไว้ด้วยของชุดนั้น"""
    import secrets as _secrets

    import yaml

    secret = _secrets.token_bytes(32)
    public = ed25519.public_from_secret(secret)
    keys.PUBLIC_KEYS["test"] = base64.b64encode(public).decode()
    fields = dict(id="LMDS-TEST-1", tier="team", licensed_to="ลูกค้าทดสอบ",
                  machines=8, issued=date(2026, 1, 1), expires=date(2027, 1, 1))
    fields.update(overrides)
    lic = License(**fields)
    signature = ed25519.sign(secret, lic.signing_bytes())
    document = {"license": lic.payload(),
                "signature": {"key": "test", "value": base64.b64encode(signature).decode()}}
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=True), lic


@pytest.fixture(autouse=True)
def _accept_the_test_key(monkeypatch):
    monkeypatch.setattr(keys, "ACTIVE", frozenset({"k1", "k2", "test"}))


def test_a_signed_licence_verifies_and_reports_its_limit(tmp_path):
    text, lic = _issue(tmp_path)
    path = tmp_path / "license.yaml"
    status = store.install(text, path)
    assert status.state == "active"
    assert status.machines_allowed == 8
    assert store.load(path, today=date(2026, 6, 1)).state == "active"


@pytest.mark.parametrize("old,new", [
    ("machines: 8", "machines: 999"),
    ("licensed_to: ลูกค้าทดสอบ", "licensed_to: คนอื่น"),
    ("tier: team", "tier: enterprise"),
    ("expires: '2027-01-01'", "expires: '2099-01-01'"),
])
def test_editing_any_field_breaks_the_signature(tmp_path, old, new):
    """แก้ฟิลด์ไหนก็ตามในไฟล์ต้องทำให้ลายเซ็นไม่ผ่าน — ไม่ใช่แค่ฟิลด์ machines"""
    text, _ = _issue(tmp_path)
    assert old in text, f"เทสนี้ล้าสมัย: ไม่มี {old!r} ในไฟล์ที่ออกมา"
    status = store.install(text.replace(old, new), tmp_path / "license.yaml")
    assert status.state == "invalid"


def test_an_invalid_file_never_overwrites_a_good_one(tmp_path):
    """เคสที่กลัว: ลูกค้าวางใบต่ออายุที่ก๊อปมาไม่ครบทับใบเดิมตอนตี 3"""
    good, _ = _issue(tmp_path)
    path = tmp_path / "license.yaml"
    assert store.install(good, path).state == "active"
    before = path.read_text(encoding="utf-8")

    assert store.install(good.replace("machines: 8", "machines: 99"), path).state == "invalid"
    assert path.read_text(encoding="utf-8") == before, "ใบที่ใช้ไม่ได้ต้องไม่ถูกเขียนลงไป"
    assert store.load(path).state == "active"


def test_the_licence_file_is_written_private(tmp_path):
    text, _ = _issue(tmp_path)
    path = tmp_path / "license.yaml"
    store.install(text, path)
    assert path.stat().st_mode & 0o077 == 0, "ไฟล์ไลเซนส์ไม่ควรให้ผู้ใช้อื่นบนเครื่องอ่านได้"


# ── กฎข้อ 2: ห้ามหยุดของที่รันอยู่ ─────────────────────────────────────────────

def test_an_expired_licence_falls_back_to_the_free_tier_not_to_zero(tmp_path):
    """หมดอายุ = เหลือสิทธิ์เท่าคนใช้ฟรี ไม่ใช่ศูนย์

    ศูนย์แปลว่า "ห้ามรันอะไรเลย" ซึ่งไม่เคยเป็นสิ่งที่เราต้องการกับลูกค้าที่จ่ายเงินมาแล้ว
    """
    text, _ = _issue(tmp_path, expires=date(2026, 1, 2))
    path = tmp_path / "license.yaml"
    store.install(text, path)
    status = store.load(path, today=date(2026, 6, 1))
    assert status.state == "expired"
    assert status.machines_allowed == FREE_SERVING_MACHINES
    assert status.read_only


def test_a_broken_file_also_falls_back_to_the_free_tier(tmp_path):
    path = tmp_path / "license.yaml"
    path.write_text("license: {ไม่ใช่: ของจริง}\n", encoding="utf-8")
    status = store.load(path)
    assert status.state == "invalid"
    assert status.machines_allowed == FREE_SERVING_MACHINES


def test_no_licence_file_is_a_normal_state_not_an_error(tmp_path):
    """คนส่วนใหญ่อยู่โหมดนี้ถาวร — ต้องไม่ใช่ error และต้องไม่มีอะไรหมดอายุ"""
    status = store.load(tmp_path / "ไม่มีไฟล์นี้.yaml")
    assert status.state == "free"
    assert status.machines_allowed == FREE_SERVING_MACHINES
    assert not status.read_only


# ── การนับเครื่องตาม LICENSE §1.1 ──────────────────────────────────────────────

class _FakeNode:
    def __init__(self, name, serving):
        self.name, self.serving = name, serving


def test_a_control_plane_hub_is_free_and_does_not_count():
    """LICENSE §1.1: "a control-plane machine ... 0 — it is free and does not count"

    ฟลีตฟรีตามปกติคือ hub ไม่มี GPU + เครื่องเสิร์ฟหนึ่งเครื่อง = นับได้ 1 = ฟรี
    """
    count = seats.count_fleet([_FakeNode("node-1", True)], hub_serves=False)
    assert count.serving == 1
    assert count.control_plane == 1
    assert enforce.check(enforce.FLEET_GROW, serving_now=count.serving, adding=0,
                         status=store.Status("free")).allowed


def test_machines_not_yet_probed_are_not_counted_but_are_reported():
    """ผิดพลาดไปทางไม่บล็อกเสมอ — แต่ต้องบอกให้เห็นว่านับไม่ครบ ไม่ใช่เงียบ"""
    count = seats.count_fleet(
        [_FakeNode("known", True), _FakeNode("ยังไม่ตรวจ", None)], hub_serves=False)
    assert count.serving == 1
    assert count.unknown == 1
    assert "ยังไม่ตรวจ" in count.explain()


def test_the_probe_remembers_a_control_plane_as_false_not_as_missing():
    """False ต้องรอดตัวกรองท้าย status_from_probe ไม่ถูกตัดทิ้งเหมือน "" กับ None"""
    from lmds.nodes.registry import status_from_probe

    assert status_from_probe({"host": {"role": {"control_plane": True}}})["serving"] is False
    assert status_from_probe({"host": {"role": {"control_plane": False}}})["serving"] is True
    # node รุ่นเก่าไม่ส่ง role มา = "ไม่รู้" ต้องไม่เขียนทับของที่เคยรู้
    assert "serving" not in status_from_probe({"host": {}})


# ── กฎข้อ 1 และ 3: ล็อกเฉพาะขนาด และห้ามล็อกของที่กันผู้ใช้เจ็บตัว ──────────────

# คำสั่งที่ **ต้องทำงานได้เสมอ** ไม่ว่าไลเซนส์จะเป็นอะไร — ดู docstring ของ enforce.py
NEVER_GATED = (
    "ps", "logs", "doctor", "repair", "validate", "smoke", "fit", "inspect", "hardware",
    "scan", "list", "version", "recipes", "bench", "start", "stop", "restart",
    "enable", "disable", "remove", "set", "adopt", "generate", "web", "config", "prune",
)


def test_every_lock_in_the_product_is_declared_in_one_place():
    """`require()` ต้องถูกเรียกจากที่ที่ประกาศไว้เท่านั้น — ไม่มีล็อกลับ

    ลูกค้าองค์กรที่ audit โค้ดอ่าน enforce.py ไฟล์เดียวต้องเห็นครบ ถ้ามีคนเพิ่ม
    `require()` ที่อื่นโดยไม่อัปเดตเอกสาร เทสนี้จะจับได้
    """
    import subprocess

    found = subprocess.run(
        ["grep", "-rn", "--include=*.py", r"licensing\.require(\|enforce\.require(\|^\s*require(",
         str(ROOT / "src")],
        capture_output=True, text=True).stdout.strip().splitlines()
    callers = {line.split(":")[0].replace(str(ROOT) + "/", "") for line in found if line}
    allowed = {"src/lmds/licensing/enforce.py", "src/lmds/nodes/ssh.py"}
    assert callers <= allowed, (
        f"มี require() นอกจุดที่ประกาศไว้: {sorted(callers - allowed)}\n"
        f"ถ้าตั้งใจเพิ่มล็อกใหม่ ให้เพิ่มใน enforce.CAPABILITIES และอัปเดต docs/LICENSING.md ด้วย")


def test_the_capabilities_table_documents_every_lock():
    """ทุกความสามารถที่ล็อกได้ต้องมีคำอธิบายที่เอาไปโชว์ลูกค้าได้"""
    assert set(enforce.CAPABILITIES) == {enforce.FLEET_GROW, enforce.FLEET_WRITE}
    for name, reason in enforce.CAPABILITIES.items():
        assert len(reason) > 30, f"{name} ต้องอธิบายให้ลูกค้าอ่านรู้เรื่อง"


def test_an_unknown_capability_is_never_a_silent_block():
    """พิมพ์ชื่อความสามารถผิดต้องไม่กลายเป็นการบล็อกผู้ใช้แบบเงียบ ๆ"""
    assert enforce.check("ชื่อที่ไม่มีอยู่จริง", serving_now=99).allowed


def test_serving_operations_are_never_gated_by_the_licence():
    """กฎข้อ 1 และ 2 — คำสั่งพวกนี้ต้องไม่มีทางถูกไลเซนส์ห้าม ไม่ว่าสถานะไหน"""
    for state in ("free", "expired", "invalid"):
        status = store.Status(state, reason="ทดสอบ")
        for command in NEVER_GATED:
            assert enforce.check(command, serving_now=999, status=status).allowed, (
                f"`lmds {command}` ถูกล็อกในสถานะ {state} — ผิดกฎข้อ 1/2 ของ enforce.py")


def test_an_expired_licence_still_allows_running_things_but_not_growing(tmp_path):
    text, _ = _issue(tmp_path, expires=date(2026, 1, 2))
    path = tmp_path / "license.yaml"
    store.install(text, path)
    status = store.load(path, today=date(2026, 6, 1))

    assert not enforce.check(enforce.FLEET_WRITE, status=status).allowed
    message = enforce.check(enforce.FLEET_WRITE, status=status).message
    assert "ไม่ถูกแตะ" in message, "ข้อความต้องบอกชัดว่าของที่รันอยู่ปลอดภัย"
    assert "start" in message and "stop" in message


def test_the_free_tier_matches_the_published_licence_text():
    """เลขในโค้ดต้องตรงกับ LICENSE ที่ประกาศไปแล้ว — ถ้าจะเปลี่ยนต้องแก้สัญญาก่อน"""
    licence_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "up to one (1) serving machine managed together" in licence_text
    assert FREE_SERVING_MACHINES == 1


def test_unlimited_licences_never_block_growth(tmp_path):
    text, _ = _issue(tmp_path, tier="enterprise", machines=0)
    status = store.install(text, tmp_path / "license.yaml")
    assert status.unlimited
    assert enforce.check(enforce.FLEET_GROW, serving_now=10_000, status=status).allowed


def test_a_revoked_key_stops_verifying(tmp_path, monkeypatch):
    """ถอนกุญแจ = ใบที่เซ็นด้วยดอกนั้นตรวจไม่ผ่านทันทีที่เครื่องอัปเดต"""
    text, _ = _issue(tmp_path)
    monkeypatch.setattr(keys, "ACTIVE", frozenset({"k1", "k2"}))   # ถอน "test"
    status = store.install(text, tmp_path / "license.yaml")
    assert status.state == "invalid"
    assert "กุญแจ" in status.reason


def test_the_shipped_public_keys_are_real_keys():
    """กุญแจที่ฝังไปกับโปรแกรมต้องถอดรหัสได้และยาว 32 ไบต์"""
    for key_id in keys.ACTIVE:
        assert keys.public_key(key_id) is not None, f"{key_id} ใน ACTIVE แต่ใช้ไม่ได้"
        assert len(keys.public_key(key_id)) == 32


def test_no_private_key_ever_ships_with_the_package():
    """กันวันที่มีคนก๊อป private key มาวางในรีโปแล้วลืม"""
    import subprocess

    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                             capture_output=True, text=True).stdout.splitlines()
    bad = [f for f in tracked if "license-signing" in f or f.endswith(".license-key")]
    assert not bad, f"private key หลุดเข้ารีโป: {bad}"


# ── ตราประทับบน bundle ─────────────────────────────────────────────────────────

def _stamped(tmp_path, **overrides) -> dict:
    """ตราจากใบจริง — ผ่านเส้นทางเดียวกับตอน generate ของจริง"""
    from lmds.licensing.stamp import build

    text, _ = _issue(tmp_path, **overrides)
    status = store.install(text, tmp_path / "license.yaml")
    assert status.state == "active"
    return build(status, lmds_version="0.7.0")


def test_a_stamp_from_a_real_licence_verifies(tmp_path):
    from lmds.licensing.stamp import verify as verify_stamp

    stamped = _stamped(tmp_path)
    assert stamped["licensed_to"] == "ลูกค้าทดสอบ"
    ok, detail = verify_stamp(stamped)
    assert ok, detail
    assert "ลูกค้าทดสอบ" in detail


def test_renaming_the_owner_on_a_stamp_is_caught(tmp_path):
    """เหตุผลทั้งหมดที่ฟีเจอร์นี้มีอยู่ — พาร์ตเนอร์ A เอา bundle ไปขายต่อในนาม B"""
    from lmds.licensing.stamp import verify as verify_stamp

    stamped = _stamped(tmp_path)
    stamped["licensed_to"] = "พาร์ตเนอร์ที่แอบอ้าง"
    ok, detail = verify_stamp(stamped)
    assert not ok
    assert "แก้ชื่อเจ้าของ" in detail


def test_editing_the_signed_payload_of_a_stamp_is_caught(tmp_path):
    from lmds.licensing.stamp import verify as verify_stamp

    stamped = _stamped(tmp_path)
    stamped["attestation"]["payload"]["licensed_to"] = "คนอื่น"
    stamped["licensed_to"] = "คนอื่น"
    ok, detail = verify_stamp(stamped)
    assert not ok
    assert "ลายเซ็น" in detail


def test_a_free_tier_stamp_is_honest_and_passes(tmp_path):
    """เครื่องโหมดฟรี generate ได้ตามสัญญา — ตราบอกตรง ๆ ว่าไม่มีใบ ไม่ใช่แกล้งว่ามี"""
    from lmds.licensing.stamp import build, verify as verify_stamp

    stamped = build(store.Status("free"), lmds_version="0.7.0")
    assert stamped["tier"] == "community"
    assert stamped["licensed_to"] is None
    assert "attestation" not in stamped
    ok, _ = verify_stamp(stamped)
    assert ok


def test_a_bundle_with_no_stamp_at_all_still_passes():
    """bundle ที่ generate ก่อนมีฟีเจอร์นี้ต้องไม่กลายเป็นของเสียข้ามคืน"""
    from lmds.licensing.stamp import verify as verify_stamp

    assert verify_stamp(None)[0]
    assert verify_stamp({})[0]


def test_claiming_an_owner_without_an_attestation_is_caught():
    """เติมชื่อเจ้าของเข้าไปเองโดยไม่มีลายเซ็นมายืนยัน"""
    from lmds.licensing.stamp import verify as verify_stamp

    ok, detail = verify_stamp({"tier": "enterprise", "licensed_to": "ใครก็ไม่รู้",
                               "license_id": "LMDS-ของปลอม"})
    assert not ok
    assert "attestation" in detail


def test_an_expired_licence_stamps_as_community_not_as_active(tmp_path):
    """ใบหมดอายุไม่ควรประทับตราว่ายังเป็นลูกค้าอยู่ — แต่ก็ต้อง generate ได้ตามปกติ"""
    from lmds.licensing.stamp import build, verify as verify_stamp

    text, _ = _issue(tmp_path, expires=date(2026, 1, 2))
    path = tmp_path / "license.yaml"
    store.install(text, path)
    status = store.load(path, today=date(2026, 6, 1))
    assert status.state == "expired"

    stamped = build(status, lmds_version="0.7.0")
    assert stamped["licensed_to"] is None
    assert verify_stamp(stamped)[0]


def test_the_stamp_gate_runs_before_the_checksum_gate():
    """run_gates() ตัดตัวสุดท้ายด้วย ALL_GATES[:-1] — gate_checksums ต้องอยู่ท้ายเสมอ

    แทรก gate ต่อท้ายโดยไม่ดูบรรทัดนี้ = ตอน include_checksums=False จะตัดผิดตัวเงียบ ๆ
    """
    from lmds.validator.gates import ALL_GATES, gate_checksums, gate_origin_stamp

    assert ALL_GATES[-1] is gate_checksums
    assert gate_origin_stamp in ALL_GATES[:-1]


def test_the_gate_reads_a_real_bundle_profile(tmp_path):
    """ด่านต้องอ่าน MODEL_PROFILE.yaml จริง ไม่ใช่แค่ทำงานกับ dict ในเทส"""
    import yaml as _yaml

    from lmds.validator.gates import gate_origin_stamp

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    stamped = _stamped(tmp_path)

    (bundle / "MODEL_PROFILE.yaml").write_text(
        _yaml.safe_dump({"model": {"id": "x"}, "origin": stamped}, allow_unicode=True),
        encoding="utf-8")
    assert gate_origin_stamp(bundle).passed

    stamped["licensed_to"] = "คนที่แอบอ้าง"
    (bundle / "MODEL_PROFILE.yaml").write_text(
        _yaml.safe_dump({"model": {"id": "x"}, "origin": stamped}, allow_unicode=True),
        encoding="utf-8")
    assert not gate_origin_stamp(bundle).passed

    # bundle ที่ไม่มี profile เลย = ไม่ใช่เรื่องของด่านนี้
    (bundle / "MODEL_PROFILE.yaml").unlink()
    assert gate_origin_stamp(bundle).passed
