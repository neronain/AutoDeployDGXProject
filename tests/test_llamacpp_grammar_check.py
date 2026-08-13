"""llama.cpp ที่เก่ากว่าที่ agent client ต้องการ

เคสจริง 2026-08-13 — gpt-oss-120b บน spark-worker ขึ้นปกติทุกอย่าง: chat ได้
streaming ได้ เรียก tool ด้วย schema ง่าย ๆ ก็ได้ แต่พอ Claude Code ส่งชุด tool
จริงมาก็ 400 ทันที:

    parse: error parsing grammar: number of repetitions exceeds sane defaults
    srv send_error: Failed to initialize samplers: failed to parse grammar

llama.cpp แปลง JSON schema เป็น GBNF แล้ว maxLength/maxItems ค่าสูงถูกขยายเป็น
repetition จนชน MAX_REPETITION_THRESHOLD (2000) · upstream แก้ที่ cd0fa6051
โดยเปลี่ยนจาก throw เป็นลด max เหลือ unbounded

ตรวจจากข้อความ error ในไบนารีไม่ได้ เพราะรุ่นที่แก้แล้วก็ยังมีสตริงนั้น (min_times
ยัง throw อยู่) จึงต้องดูที่ commit
"""

import lmds.doctor.checks as checks
from lmds.doctor.checks import Status, _check_llamacpp_grammar


def _profile(engine="llamacpp", llamacpp_dir=None):
    p = {"runtime": {"engine": engine}, "target": {}}
    if llamacpp_dir:
        p["target"]["llamacpp_dir"] = str(llamacpp_dir)
    return p


def _fake_checkout(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    return root


def test_a_build_without_the_fix_is_flagged(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    # git merge-base --is-ancestor คืน 1 = ไม่ใช่บรรพบุรุษ = ยังไม่มีคอมมิตนั้น
    monkeypatch.setattr(checks, "_run", lambda *a, **k: (1, ""))
    findings = _check_llamacpp_grammar(_profile(llamacpp_dir=root), "slug")
    assert len(findings) == 1
    assert findings[0].status is Status.WARN
    assert "cd0fa6051" in findings[0].detail
    assert "failed to parse grammar" in findings[0].detail


def test_a_build_with_the_fix_passes(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    monkeypatch.setattr(checks, "_run", lambda *a, **k: (0, ""))
    findings = _check_llamacpp_grammar(_profile(llamacpp_dir=root), "slug")
    assert len(findings) == 1
    assert findings[0].status is Status.OK


def test_an_unknown_commit_is_not_judged(tmp_path, monkeypatch):
    """checkout ตื้นหรือคนละ remote — git คืนโค้ดอื่น ห้ามเดาว่าแย่"""
    root = _fake_checkout(tmp_path)
    monkeypatch.setattr(checks, "_run", lambda *a, **k: (128, "fatal: Not a valid object name"))
    assert _check_llamacpp_grammar(_profile(llamacpp_dir=root), "slug") == []


def test_a_non_git_install_is_skipped(tmp_path, monkeypatch):
    """ติดตั้งจาก tarball/แพ็กเกจ — ไม่มี .git ให้ถาม ก็ไม่ตัดสิน"""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    monkeypatch.setattr(checks, "_run", lambda *a, **k: (0, ""))
    assert _check_llamacpp_grammar(_profile(llamacpp_dir=root), "slug") == []


def test_vllm_is_not_asked_about_llamacpp(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "_run", lambda *a, **k: (1, ""))
    assert _check_llamacpp_grammar(_profile(engine="vllm"), "slug") == []


def test_the_pinned_build_is_what_gets_checked(tmp_path, monkeypatch):
    """โมเดลที่ผูกกับ build ของตัวเอง ต้องถูกถามที่ build นั้น ไม่ใช่ของกลาง"""
    root = _fake_checkout(tmp_path)
    seen = []

    def fake_run(cmd, timeout=15):
        seen.append(cmd)
        return (0, "")

    monkeypatch.setattr(checks, "_run", fake_run)
    _check_llamacpp_grammar(_profile(llamacpp_dir=root), "slug")
    assert str(root) in seen[0]
