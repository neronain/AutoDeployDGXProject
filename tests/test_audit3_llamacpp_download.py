"""Audit รอบ 3 (2026-10-06) ข้อ 8 — สามจุดที่ auditor สงสัยในทาง download ของ llama.cpp (ยืนยันแล้วว่าจริงทั้งสาม)

(i)   aria2c ล้มแต่ทิ้งไฟล์ขนาดครบไว้ (เขียนไม่เรียงลำดับ ข้างในเป็นรู) → ข้าม curl → "download complete" → verify-files
      ผ่านเมื่อ Hub ไม่ให้ sha256
(ii)  ล็อก download ถูกถืออยู่ แต่ fuser/lsof หาเจ้าของไม่เจอ → pipeline คืน 1 ใต้ set -e/pipefail → จบ exit 1 เงียบสนิท
(iii) Hub ไม่บอกขนาด + ไฟล์ครบอยู่แล้ว → `curl -C -` ได้ 416 (curl รุ่นเก่า exit 22) → die ทั้งที่ไม่มีอะไรผิด

รัน `download` / `verify-files` ของ controller ที่ render แล้วทั้งไฟล์ กับ aria2c/curl/flock/fuser ปลอมบน PATH
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.single_controller_harness import GGUF_BYTES, GGUF_NAME, Box, gguf_report, render, write_exe

REAL = b"GGUF-real-bytes!"            # เนื้อจริงของไฟล์ (16 ไบต์) — GGUF_BYTES ของ harness คือ "รู" (ศูนย์ล้วน) ขนาดเท่ากันไม่ได้
HOLES = b"GGUF" + b"\0" * (len(REAL) - 4)

# aria2c ปลอมตาม repro_aria2_holes.sh: วางไฟล์ยาวเท่าขนาดจริง (ท่อนท้ายมาถึงแล้ว ท่อนก่อนหน้าเป็นรู) แล้วตายแบบสายหลุด
_ARIA2_HOLES = r'''
echo "aria2c $*" >> "$FAKE_LOG"
d=""; o=""; while (( $# )); do case "$1" in -d) d="$2"; shift ;; -o) o="$2"; shift ;; esac; shift; done
printf 'GGUF' > "$d/$o"; head -c 12 /dev/zero >> "$d/$o"
[[ -n "${FAKE_ARIA2_CONTROL:-}" ]] && : > "$d/$o.aria2"
echo "aria2c: [ERROR] CUID#7 - Download aborted. errorCode=2 Timeout." >&2
exit 2
'''

# curl ปลอม: โหลดสำเร็จ = เขียนเนื้อจริงลง -o · FAKE_CURL_FAIL = ต่อไม่ได้
_CURL = r'''
case " $* " in *" --help "*) echo "--retry-all-errors"; exit 0 ;; esac
out=""; prev=""; range=""
for a in "$@"; do [[ "$prev" == "-o" ]] && out="$a"; [[ "$prev" == "-r" ]] && range="$a"; prev="$a"; done
cat >/dev/null 2>&1 || true                       # -K - : config (token) มาทาง stdin
echo "curl out=$out range=$range resume=$([[ " $* " == *" -C - "* ]] && echo yes || echo no)" >> "$FAKE_LOG"
if [[ -n "${FAKE_CURL_416:-}" ]]; then
  # เซิร์ฟเวอร์ที่ไฟล์ครบแล้ว: ขอ range หลังไบต์สุดท้าย = 416 · curl รุ่นเก่า + -f = exit 22
  if [[ -n "$range" ]]; then printf '416'; exit 0; fi
  echo "curl: (22) The requested URL returned error: 416" >&2; exit 22
fi
[[ -n "${FAKE_CURL_FAIL:-}" ]] && exit 7
printf 'GGUF-real-bytes!' > "$out"
'''


def _box(tmp_path, sha256=None, size=len(REAL)) -> Box:
    from lmds.inspector.report import GgufVariant

    report = gguf_report(gguf_variants=[GgufVariant(filename=GGUF_NAME, size_bytes=size, sha256=sha256)])
    made = Box(tmp_path, "llamacpp", render(tmp_path, "llamacpp", report=report))
    (made.home / "models" / made.bundle.directory.name / GGUF_NAME).unlink()      # ยังไม่เคยโหลด
    write_exe(made.bin / "curl", _CURL)
    write_exe(made.bin / "sleep", "exit 0\n")           # ลูป resume หน่วง 5 วิต่อรอบ — ไม่ใช่สิ่งที่วัด
    return made


def _model(made: Box):
    return made.home / "models" / made.bundle.directory.name / GGUF_NAME


# ═════════════════════ (i) aria2c ที่ล้มต้องไม่ถูกนับว่าสำเร็จ ═════════════════════
def test_a_failed_aria2c_that_left_a_full_length_file_is_redone_not_reported_complete(tmp_path):
    made = _box(tmp_path)
    try:
        write_exe(made.bin / "aria2c", _ARIA2_HOLES)
        done = made.run("download")
        assert done.returncode == 0, done.stdout + done.stderr
        assert _model(made).read_bytes() == REAL, "ได้ไฟล์ที่เป็นรูของ aria2c — ขนาดครบแต่เนื้อไม่ใช่ของจริง"
        assert "ทิ้งไฟล์บางส่วน" in done.stdout and "resume=yes" in made.calls(), done.stdout + made.calls()
    finally:
        made.close()


def test_when_every_downloader_fails_download_fails_and_verify_does_not_pass(tmp_path):
    """ทั้ง aria2c และ curl ล้ม — เดิม "download complete" rc 0 แล้ว verify-files: OK (Hub ไม่ให้ sha256) กับไฟล์ที่เป็นรู"""
    made = _box(tmp_path)
    try:
        write_exe(made.bin / "aria2c", _ARIA2_HOLES)
        done = made.run("download", env={"FAKE_CURL_FAIL": "1"})
        assert done.returncode != 0, done.stdout + done.stderr
        assert "download complete" not in done.stdout
        verified = made.run("verify-files")
        assert verified.returncode != 0 and "verify-files: OK" not in verified.stdout, verified.stdout
        assert not _model(made).exists() or _model(made).read_bytes() != HOLES, "ไฟล์รูของ aria2c ยังถูกทิ้งไว้ให้คนเข้าใจผิด"
    finally:
        made.close()


def test_aria2c_gets_to_resume_its_own_partial_before_curl_takes_over(tmp_path):
    """มี .aria2 (สมุดจดว่าท่อนไหนได้แล้ว) = aria2c ต่อเองได้ถูก — ให้ลองก่อน · ยังล้มจึงทิ้งทั้งไฟล์และสมุดจดแล้วใช้ curl"""
    made = _box(tmp_path)
    try:
        write_exe(made.bin / "aria2c", _ARIA2_HOLES)
        done = made.run("download", env={"FAKE_ARIA2_CONTROL": "1"})
        assert done.returncode == 0, done.stdout + done.stderr
        assert made.calls().count("aria2c ") == 3, "ล้มครั้งแรก + ต่อเองได้อีกสองรอบ"
        assert _model(made).read_bytes() == REAL
        assert not (_model(made).parent / (GGUF_NAME + ".aria2")).exists()
    finally:
        made.close()


def test_an_aria2c_that_fails_without_touching_the_file_keeps_the_resumable_partial(tmp_path):
    """aria2c ที่ตายก่อนเขียนอะไร (option ไม่รู้จัก ฯลฯ): ของเดิมที่ curl รอบก่อนโหลดค้างไว้ยังต่อได้ — ห้ามทิ้งหลาย GB ฟรี ๆ"""
    made = _box(tmp_path)
    try:
        write_exe(made.bin / "aria2c", 'echo "aria2c $*" >> "$FAKE_LOG"; exit 1\n')
        _model(made).write_bytes(REAL[:6])
        done = made.run("download")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ทิ้งไฟล์บางส่วน" not in done.stdout and "ถอยไป curl" in done.stdout
        assert _model(made).read_bytes() == REAL
    finally:
        made.close()


# ═════════════════════ (ii) ล็อกที่หาเจ้าของไม่เจอ ═════════════════════
@pytest.mark.parametrize("finder", ["fuser", "lsof"])
def test_a_held_download_lock_whose_owner_cannot_be_found_still_says_so(tmp_path, finder):
    made = _box(tmp_path)
    try:
        write_exe(made.bin / "flock", "exit 1\n")                 # ล็อกถูกคนอื่นถืออยู่
        write_exe(made.bin / finder, "exit 1\n")                  # มีเครื่องมือ แต่มองไม่เห็น process ที่ถือ
        # PATH ที่มีเฉพาะเครื่องมือค้นหาตัวที่กำลังทดสอบ — macOS มี fuser/lsof จริงติดเครื่อง ซึ่งจะเห็น fd ของสคริปต์เอง
        real = tmp_path / "realbin"
        real.mkdir()
        for src_dir in ("/usr/bin", "/bin", "/sbin", "/usr/sbin"):
            for tool in Path(src_dir).iterdir():
                if tool.name in ("fuser", "lsof", "flock") or (real / tool.name).exists():
                    continue
                try:
                    (real / tool.name).symlink_to(tool)
                except OSError:
                    pass
        done = made.run("download", path=f"{made.bin}:{real}")
        assert done.returncode != 0
        assert "กำลังรันอยู่แล้ว" in done.stderr, f"จบเงียบ: stdout={done.stdout!r} stderr={done.stderr!r}"
        assert "curl out=" not in made.calls(), "ต้องไม่เริ่มโหลดซ้อน"
    finally:
        made.close()


# ═════════════════════ (iii) ไม่รู้ขนาด + ไฟล์ครบอยู่แล้ว ═════════════════════
def test_redownloading_a_complete_file_of_unknown_size_is_not_an_error(tmp_path):
    made = _box(tmp_path, size=None)
    try:
        _model(made).write_bytes(REAL)
        done = made.run("download", env={"FAKE_CURL_416": "1"})
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ครบอยู่แล้ว" in done.stdout and "download complete" in done.stdout, done.stdout
        assert _model(made).read_bytes() == REAL
        assert f"range={len(REAL)}-{len(REAL)}" in made.calls(), "ถาม 1 ไบต์ที่ท้ายไฟล์ ไม่ใช่โหลดซ้ำ"
    finally:
        made.close()


def test_a_real_failure_with_unknown_size_still_fails(tmp_path):
    made = _box(tmp_path, size=None)
    try:
        _model(made).write_bytes(REAL[:5])
        done = made.run("download", env={"FAKE_CURL_FAIL": "1"})
        assert done.returncode != 0 and "ดึงไฟล์ไม่สำเร็จ" in done.stderr, done.stdout + done.stderr
    finally:
        made.close()


def test_the_harness_bytes_really_differ_from_the_holes():
    assert len(HOLES) == len(REAL) and HOLES != REAL and GGUF_BYTES.startswith(b"GGUF")
