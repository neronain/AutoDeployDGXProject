"""โหลดไฟล์ใหญ่หลายส่วนพร้อมกันด้วย curl ล้วน เมื่อไม่มี aria2c

เคสจริง 2026-09-04: Qwen3-VL-Embedding f16 14 GB → dgx-spark03 · HF เสิร์ฟผ่าน Xet bridge สตรีมเดี่ยว
0.3–1.4 MB/s (= 11 ชม.) แต่ range 8 ส่วนพร้อมกันได้ ~50 MB/s · node ไม่มี aria2c
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location("_dl", Path(__file__).with_name("test_download_resume.py"))
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)
_controller, _extract = _dl._controller, _dl._extract

# curl ปลอมที่รองรับ -r start-end จากไฟล์ต้นทาง (FAKE_SRC) และ -K - (อ่าน config จาก stdin แล้วทิ้ง)
# · FAKE_FAIL_PART=<i> ทำให้ส่วนนั้นตายกลางคัน (เขียนแค่ครึ่ง) เพื่อทดสอบ resume
_CURL = r'''#!/bin/bash
[[ "$1" == "--help" ]] && { echo "--retry-all-errors"; exit 0; }
range=""; out=""; url=""
while (( $# )); do
  case "$1" in
    -r) range="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    -K) cat >/dev/null; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
echo "curl range=${range:-none}" >> "$FAKE_LOG"
[[ -n "$out" ]] && exec > "$out"
src="$FAKE_SRC"; size=$(stat -c %s "$src")
if [[ -n "$range" ]]; then
  start="${range%-*}"; end="${range#*-}"; len=$(( end - start + 1 ))
  if [[ -n "${FAKE_FAIL_PART:-}" && "$start" == "${FAKE_FAIL_PART}" ]]; then
    len=$(( len / 2 )); tail -c +$(( start + 1 )) "$src" | head -c "$len"; exit 18
  fi
  tail -c +$(( start + 1 )) "$src" | head -c "$len"
else
  cat "$src"
fi
'''


def _bin(tmp_path: Path) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    (b / "curl").write_text(_CURL, encoding="utf-8")
    (b / "curl").chmod(0o755)
    (b / "aria2c").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")   # ไม่มี aria2c จริง
    (b / "aria2c").chmod(0o755)
    return b


def _script(tmp_path: Path, out: Path, want: int, extra_env: str = "") -> str:
    text = _controller(tmp_path)
    funcs = "\n".join(_extract(text, n) for n in ("file_size", "curl_retry_all", "fetch_parallel", "fetch_one"))
    return (
        "set -euo pipefail\nFETCH_MAX_ATTEMPTS=3\ndie() { echo \"DIE: $*\" >&2; exit 9; }\n"
        f"{extra_env}\n{funcs}\n"
        f'fetch_one "https://example.invalid/big.gguf" "{out}" {want}\n'
    )


def _run(tmp_path: Path, src: Path, want: int, env_extra: dict | None = None, extra_env: str = "") -> tuple[subprocess.CompletedProcess, Path]:
    out = tmp_path / "models" / "big.gguf"
    out.parent.mkdir(exist_ok=True)
    log = tmp_path / "curl.log"
    log.write_text("")
    if not (tmp_path / "bin").exists():
        _bin(tmp_path)
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
           "HOME": str(tmp_path), "FAKE_SRC": str(src), "FAKE_LOG": str(log), **(env_extra or {})}
    done = subprocess.run(["bash", "-c", _script(tmp_path, out, want, extra_env)],
                          capture_output=True, text=True, env=env, timeout=120)
    return done, out


def _source(tmp_path: Path, size: int) -> Path:
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(size))
    return src


def test_a_big_file_is_fetched_in_parallel_parts_and_reassembled_byte_exact(tmp_path):
    src = _source(tmp_path, 300 * 1024 * 1024 + 12345)      # ≥ 256 MB → ทางขนาน
    done, out = _run(tmp_path, src, src.stat().st_size)
    assert done.returncode == 0, done.stderr + done.stdout
    assert out.read_bytes() == src.read_bytes()
    ranges = [l for l in (tmp_path / "curl.log").read_text().splitlines() if "range=" in l and "none" not in l]
    assert len(ranges) == 8, ranges
    assert not (Path(str(out) + ".parts")).exists(), "โฟลเดอร์ส่วนต้องถูกเก็บกวาดเมื่อครบ"
    assert "โหลดขนาน 8 ส่วน" in done.stdout


def test_a_part_that_dies_halfway_is_resumed_from_where_it_stopped(tmp_path):
    src = _source(tmp_path, 264 * 1024 * 1024)
    want = src.stat().st_size
    chunk = (want + 7) // 8
    fail_start = 3 * chunk                                   # ส่วนที่ 3 ตายกลางคันในรอบแรก
    done, out = _run(tmp_path, src, want, env_extra={"FAKE_FAIL_PART": str(fail_start)})
    # รอบแรก ส่วน 3 ไม่ครบ → fetch_parallel คืน 1 → ถอยไป curl เดี่ยว "ต่อจากที่มี" ซึ่งของเรา
    # ไฟล์ out ยังไม่มี → curl เดี่ยวโหลดทั้งไฟล์ (fake ไม่รองรับ -C แต่ cat ทั้งไฟล์) → ครบ
    assert done.returncode == 0, done.stderr + done.stdout
    assert out.read_bytes() == src.read_bytes()
    log = (tmp_path / "curl.log").read_text()
    assert f"range={fail_start}-" in log, "ส่วนที่ 3 ต้องถูกยิงในรอบแรก"


def test_small_files_and_fetch_parts_1_keep_the_single_stream_path(tmp_path):
    src = _source(tmp_path, 2 * 1024 * 1024)                 # 2 MB < 256 MB
    done, out = _run(tmp_path, src, src.stat().st_size)
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == src.read_bytes()
    assert "range=" not in (tmp_path / "curl.log").read_text().replace("range=none", "")
    # ปิดด้วย FETCH_PARTS=1 แม้ไฟล์ใหญ่
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(260 * 1024 * 1024))
    done, out = _run(tmp_path, big, big.stat().st_size, extra_env="FETCH_PARTS=1")
    assert done.returncode == 0, done.stderr
    assert "โหลดขนาน" not in done.stdout
