"""audit 2026-10-06 — `install.sh` ต้องไม่ทิ้งเครื่องไว้แบบไม่มี `lmds` และต้องไม่ประกาศ "เสร็จ" ลอย ๆ

install.sh ย้าย venv เดิมไป `venv.old` ก่อนสร้างใหม่ แล้วย้ายกลับเมื่อล้ม — แต่ผูกการย้ายกลับไว้กับ
แค่สองจุด (สร้าง venv · pip หลัก) ขณะที่ระหว่างนั้นมีคำสั่งอื่นที่ล้มได้ใต้ `set -e`:

  A. เขียน `src/lmds/_build.py` ไม่ได้ ("Permission denied" — checkout เป็นของ root จาก
     `sudo env HOME=… ./install.sh` รอบก่อน) → exit 1 · `venv/bin` มี pip แต่ไม่มี lmds ·
     `lmds version` → no such file · **รันซ้ำ** → บรรทัด `rm -rf venv.old` ลบสำเนาสุดท้ายที่ใช้ได้ทิ้ง
  D. pip สำเร็จแต่ `lmds` ตัวใหม่รันไม่ขึ้น → พิมพ์ "ติดตั้งเสร็จ: " (เวอร์ชันว่าง) exit 0 และ
     `venv.old` ถูกลบไปแล้วก่อนถึงบรรทัดที่ลองรัน
  E. `LMDS_BIN_DIR=…/tools/bin` → ประกาศ "เพิ่ม …/tools/bin ลง PATH" แต่เขียน `${HOME}/.local/bin`

ทุกเทสรัน **install.sh ตัวจริงทั้งไฟล์** ใน HOME ชั่วคราว กับ `python3`/`pip`/`git` ปลอมบน PATH
(ไม่มีเน็ต ไม่มี venv จริง ไม่แตะ HOME ของเครื่อง) แล้วดูสิ่งเดียวที่สำคัญ: หลังจบ `lmds version` ยังรันได้ไหม
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[1] / "install.sh"

# pip ปลอมที่ venv ปลอมวางไว้ — พฤติกรรมคุมด้วย $FAKE_FAIL
_FAKE_PIP = r'''#!/bin/bash
echo "pip $*" >> "$FAKE_LOG"
case "$*" in
  *"--upgrade pip"*)
    if [ "${FAKE_FAIL:-}" = upgrade-pip ]; then
      echo "ERROR: Could not fetch https://pypi.org/simple/pip/ (network unreachable)" >&2; exit 1
    fi ;;
  *keyring*|*fastapi*) exit 0 ;;
  *)
    case "${FAKE_FAIL:-}" in
      install-lmds) echo "ERROR: build failed" >&2; exit 1 ;;
      killed)       kill -TERM "$PPID"; sleep 1; exit 1 ;;
    esac
    d="$(dirname "$0")"
    if [ "${FAKE_FAIL:-}" = broken-lmds ]; then
      printf '#!/bin/bash\necho "ModuleNotFoundError: No module named pydantic_core" >&2\nexit 1\n' > "$d/lmds"
    elif [ "${FAKE_FAIL:-}" = silent-lmds ]; then
      printf '#!/bin/bash\nexit 0\n' > "$d/lmds"
    else
      printf '#!/bin/bash\necho "lmds 2.0.0 (NEW)"\n' > "$d/lmds"
    fi
    chmod +x "$d/lmds" ;;
esac
exit 0
'''

# python3 ปลอม: พอสำหรับเส้นทางของ install.sh — ตรวจเวอร์ชัน (stdin), ensurepip, `-m venv`
_FAKE_PYTHON = r'''#!/bin/bash
case "$1" in
  -)  cat >/dev/null; exit 0 ;;
  -c) exit 0 ;;
  -m) if [ "$2" = venv ]; then
        [ "$3" = "--help" ] && exit 0
        dir="${@: -1}"
        [ "${FAKE_FAIL:-}" = venv ] && { echo "Error: cannot create venv" >&2; exit 1; }
        rm -rf "$dir"; mkdir -p "$dir/bin"
        cp "$FAKE_PIP_SRC" "$dir/bin/pip"; chmod +x "$dir/bin/pip"
        exit 0
      fi ;;
esac
exit 0
'''


class Box:
    """เครื่องจำลองหนึ่งเครื่อง: checkout + HOME + PATH ที่มีแต่ของปลอม"""

    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.home = root / "home"
        self.shims = root / "shims"
        (self.repo / "src" / "lmds").mkdir(parents=True)
        (self.repo / ".git").mkdir()
        shutil.copy2(INSTALL_SH, self.repo / "install.sh")
        self.home.mkdir()
        self.shims.mkdir()
        self._shim("python3", _FAKE_PYTHON)
        self._shim("git", "#!/bin/bash\necho abc1234\n")
        (root / "fake-pip").write_text(_FAKE_PIP, encoding="utf-8")
        self.install_dir = self.home / ".local" / "share" / "lmds"
        self.bin_dir = self.home / ".local" / "bin"

    def _shim(self, name: str, body: str) -> None:
        path = self.shims / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def existing_install(self, where: str = "venv") -> None:
        """รุ่นเดิมที่ใช้ได้อยู่ — `where="venv.old"` = สภาพหลังรอบก่อนถูก kill กลางทาง"""
        bin_dir = self.install_dir / where / "bin"
        bin_dir.mkdir(parents=True)
        lmds = bin_dir / "lmds"
        lmds.write_text('#!/bin/bash\necho "lmds 1.0.0 (OLD, working)"\n', encoding="utf-8")
        lmds.chmod(0o755)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        link = self.bin_dir / "lmds"
        if not link.is_symlink():
            link.symlink_to(self.install_dir / "venv" / "bin" / "lmds")

    def half_built_venv(self) -> None:
        """venv ที่มี pip แต่ยังไม่มี lmds — สิ่งที่รอบที่ล้มกลางทางทิ้งไว้"""
        bin_dir = self.install_dir / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "pip").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    def run(self, **env: str) -> subprocess.CompletedProcess:
        full = {"PATH": f"{self.shims}:/usr/bin:/bin", "HOME": str(self.home), "SHELL": "/bin/bash",
                "USER": "tester", "LMDS_SKIP_PREREQ": "1", "FAKE_LOG": str(self.root / "pip.log"),
                "FAKE_PIP_SRC": str(self.root / "fake-pip"), **env}
        return subprocess.run(["bash", "./install.sh"], cwd=self.repo, env=full, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    def lmds_version(self) -> str:
        """สิ่งที่ผู้ใช้ได้เมื่อพิมพ์ `lmds version` หลัง installer จบ — ว่าง = ไม่มี lmds ให้รันแล้ว"""
        link = self.bin_dir / "lmds"
        if not os.access(link, os.X_OK):
            return ""
        done = subprocess.run([str(link), "version"], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    def pip_calls(self) -> str:
        log = self.root / "pip.log"
        return log.read_text(encoding="utf-8") if log.exists() else ""


@pytest.fixture
def box(tmp_path) -> Box:
    return Box(tmp_path)


OLD = "lmds 1.0.0 (OLD, working)"
NEW = "lmds 2.0.0 (NEW)"


# ── เส้นทางปกติ ───────────────────────────────────────────────────────────────────────
def test_a_normal_upgrade_ends_with_the_new_version_running(box):
    box.existing_install()
    done = box.run()
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"ติดตั้งเสร็จ: {NEW}" in done.stdout
    assert box.lmds_version() == NEW
    assert not (box.install_dir / "venv.old").exists(), "รุ่นใหม่รันได้แล้ว — ของเดิมไม่ต้องเก็บ"


def test_a_fresh_machine_installs(box):
    done = box.run()
    assert done.returncode == 0, done.stdout + done.stderr
    assert box.lmds_version() == NEW


# ── A. ล้มระหว่างย้าย venv เดิมกับ pip หลัก ───────────────────────────────────────────
@pytest.mark.skipif(os.geteuid() == 0, reason="root เขียนไฟล์ 0444 ได้ — จำลองเคสนี้ไม่ได้")
def test_an_unwritable_build_stamp_leaves_the_working_install_alone_even_when_retried(box):
    """checkout ที่ root เป็นเจ้าของ: `printf … > src/lmds/_build.py` → Permission denied

    ก่อนแก้: exit 1 · venv/bin มี pip แต่ไม่มี lmds · รอบที่สอง `rm -rf venv.old` ลบสำเนาสุดท้ายทิ้ง
    """
    box.existing_install()
    stamp = box.repo / "src" / "lmds" / "_build.py"
    stamp.write_text('COMMIT = "old"\n', encoding="utf-8")
    stamp.chmod(0o444)
    try:
        for attempt in (1, 2, 3):
            done = box.run()
            assert done.returncode != 0, f"รอบ {attempt}: ต้องล้ม ไม่ใช่ผ่านเงียบ ๆ\n{done.stdout}"
            assert "ติดตั้งเสร็จ" not in done.stdout
            assert box.lmds_version() == OLD, f"รอบ {attempt}: รุ่นเดิมหายไป\n{done.stdout}{done.stderr}"
        assert "_build.py" in done.stderr and "chown" in done.stderr, "ต้องบอกไฟล์ที่เขียนไม่ได้และทางแก้"
    finally:
        stamp.chmod(0o644)


def test_failing_to_upgrade_pip_itself_does_not_abort_the_install(box):
    """`pip install --upgrade pip` เป็นแค่การขยับ pip — ล้ม (เน็ตสะดุด) ไม่ควรฆ่าทั้งการติดตั้ง

    ก่อนแก้: บรรทัดนี้อยู่ใต้ `set -e` โดยไม่มีการย้าย venv กลับ → เครื่องเหลือแบบไม่มี lmds
    """
    box.existing_install()
    done = box.run(FAKE_FAIL="upgrade-pip")
    assert done.returncode == 0, done.stdout + done.stderr
    assert box.lmds_version() == NEW


def test_a_failed_venv_creation_restores_the_old_install(box):
    box.existing_install()
    done = box.run(FAKE_FAIL="venv")
    assert done.returncode != 0
    assert box.lmds_version() == OLD


def test_a_failed_pip_install_restores_the_old_install(box):
    """เคสเดิม (spark-head 2026-09-04: PyPI ช้า) — ต้องยังทำงานเหมือนเดิมหลังจัดโครงใหม่"""
    box.existing_install()
    done = box.run(FAKE_FAIL="install-lmds")
    assert done.returncode != 0
    assert box.lmds_version() == OLD
    assert OLD in done.stderr, "ต้องบอกว่ารุ่นเดิมยังอยู่และเป็นรุ่นไหน"


def test_being_killed_in_the_middle_restores_the_old_install(box):
    """ssh หลุด / หมดเวลา `lmds node install` (timeout 420) ระหว่าง pip — SIGTERM ต้องไม่ทิ้ง venv ครึ่งตัว"""
    box.existing_install()
    done = box.run(FAKE_FAIL="killed")
    assert done.returncode != 0
    assert box.lmds_version() == OLD


# ── re-run หลังรอบที่ล้ม ───────────────────────────────────────────────────────────────
def test_a_rerun_never_deletes_the_last_working_copy(box):
    """สภาพหลังรอบที่ตายแบบย้ายกลับไม่ทัน (kill -9 / ไฟดับ): venv ครึ่งตัว + venv.old คือตัวที่ใช้ได้

    ก่อนแก้: บรรทัดแรกของส่วนติดตั้งคือ `rm -rf venv.old` — ลบตัวที่ใช้ได้ แล้วเก็บ venv ครึ่งตัวไว้แทน
    """
    box.existing_install(where="venv.old")
    box.half_built_venv()
    assert box.lmds_version() == ""                       # ตอนนี้ไม่มี lmds ให้รัน

    done = box.run(FAKE_FAIL="install-lmds")              # รอบซ่อมก็ยังล้ม (เน็ตยังไม่มา)
    assert done.returncode != 0
    assert box.lmds_version() == OLD, "ล้มซ้ำต้องได้รุ่นเดิมกลับมา ไม่ใช่เสียมันไปถาวร"

    done = box.run()                                      # เน็ตมาแล้ว
    assert done.returncode == 0, done.stdout + done.stderr
    assert box.lmds_version() == NEW


# ── D. pip ผ่าน แต่ของที่ได้รันไม่ขึ้น ─────────────────────────────────────────────────
@pytest.mark.parametrize("failure", ["broken-lmds", "silent-lmds"])
def test_success_is_only_announced_after_the_new_lmds_actually_ran(box, failure):
    """ก่อนแก้: "ติดตั้งเสร็จ: " (เวอร์ชันว่าง) exit 0 และ venv.old ถูกลบไปก่อนจะลองรันด้วยซ้ำ

    broken = lmds ตัวใหม่ตายตอน import · silent = รันจบแต่ไม่พิมพ์เวอร์ชัน (ไม่ใช่ lmds ที่ใช้ได้)
    """
    box.existing_install()
    done = box.run(FAKE_FAIL=failure)
    assert done.returncode != 0, done.stdout
    assert "ติดตั้งเสร็จ" not in done.stdout
    assert box.lmds_version() == OLD, "ของใหม่รันไม่ได้ ต้องได้ของเดิมกลับมา"
    assert OLD in done.stderr
    if failure == "broken-lmds":
        assert "pydantic_core" in done.stderr, "ต้องโชว์สิ่งที่ lmds ตัวใหม่พูดเอง ไม่ใช่แค่คำวินิจฉัยของเรา"


def test_a_fresh_machine_whose_new_lmds_cannot_start_is_a_failure_not_a_success(box):
    done = box.run(FAKE_FAIL="broken-lmds")
    assert done.returncode != 0
    assert "ติดตั้งเสร็จ" not in done.stdout
    assert box.lmds_version() == ""


# ── E. PATH ที่เขียนต้องตรงกับที่ประกาศ ────────────────────────────────────────────────
def test_the_path_line_written_names_the_directory_that_was_announced(box):
    """`LMDS_BIN_DIR=…/tools/bin` → ประกาศ "เพิ่ม …/tools/bin ลง PATH" แต่เขียน `${HOME}/.local/bin`"""
    custom = box.home / "tools" / "bin"
    done = box.run(LMDS_BIN_DIR=str(custom))
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"เพิ่ม {custom} ลง PATH" in done.stdout
    rc = (box.home / ".bashrc").read_text(encoding="utf-8")
    assert ".local/bin" not in rc

    # พิสูจน์ด้วยของจริง: shell ใหม่ที่ source rc นี้ต้องหา lmds เจอ
    found = subprocess.run(["bash", "-c", 'source "$HOME/.bashrc"; command -v lmds && lmds version'],
                           env={"HOME": str(box.home), "PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True)
    assert found.returncode == 0 and str(custom / "lmds") in found.stdout and NEW in found.stdout, found

    box.run(LMDS_BIN_DIR=str(custom))                     # รันซ้ำต้องไม่เติมบรรทัดซ้ำ
    assert (box.home / ".bashrc").read_text(encoding="utf-8").count("export PATH=") == 1


def test_the_default_location_keeps_the_portable_home_form(box):
    """ที่ติดตั้งปริยาย: บรรทัดเดิม `${HOME}/.local/bin` (ย้าย home แล้วยังถูก) และไม่ซ้ำกับของที่รุ่นก่อนเขียนไว้"""
    (box.home / ".bashrc").write_text('export PATH="${HOME}/.local/bin:${PATH}"\n', encoding="utf-8")
    done = box.run()
    assert done.returncode == 0, done.stdout + done.stderr
    rc = (box.home / ".bashrc").read_text(encoding="utf-8")
    assert rc.count('export PATH="${HOME}/.local/bin:${PATH}"') == 1
    assert "ลง PATH" not in done.stdout, "มีอยู่แล้วต้องไม่ประกาศว่าเพิ่ม"
