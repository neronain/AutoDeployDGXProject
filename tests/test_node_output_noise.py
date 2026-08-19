"""stdout ของ node อาจมีขยะจาก rc/banner ปนหน้า JSON — hub ต้องยังอ่านออก

เคสจริง 2026-08-19: dgx-70 มี rc ที่พ่น `declare -x …` ทุกครั้งที่ login shell เริ่ม
JSON ถูกต้องครบถ้วนแต่เริ่มที่ไบต์ที่ 858 · hub รายงานว่า "เวอร์ชัน LMDS อาจไม่ตรงกัน"
"""

from lmds.nodes.ssh import _json_object

NOISE = (
    'declare -x HOME="/home/praisit"\n'
    'declare -x LANG="en_US.utf8"\n'
    'declare -x PATH="/home/praisit/.local/bin:/usr/bin"\n'
)


def test_plain_json():
    assert _json_object('{"a": 1}') == {"a": 1}


def test_json_after_rc_noise():
    assert _json_object(NOISE + '{"models": [], "summary": {"total": 0}}') == {
        "models": [], "summary": {"total": 0}}


def test_json_with_trailing_motd():
    assert _json_object('{"a": 1}\nLast login: Wed Aug 19\n') == {"a": 1}


def test_nested_braces_survive():
    payload = '{"outer": {"inner": [1, 2]}, "z": "}"}'
    assert _json_object(NOISE + payload)["outer"] == {"inner": [1, 2]}


def test_no_json_at_all():
    assert _json_object("Usage: lmds [OPTIONS]") is None
    assert _json_object("") is None


def test_non_object_json_rejected():
    """agent info คืน object เสมอ — array/ตัวเลขแปลว่าคุยกันคนละเรื่อง"""
    assert _json_object("[1, 2, 3]") is None


def test_braces_inside_the_noise():
    """ขยะข้างหน้าอาจมีปีกกาของมันเอง — เช่น LS_COLORS หรือ prompt ที่มี {}"""
    from lmds.nodes.ssh import _json_object

    noisy = 'declare -x LS_COLORS="ow=01;{34}:"\ndeclare -x PS1="{\\u}"\n'
    assert _json_object(noisy + '{"ok": true}') == {"ok": True}
