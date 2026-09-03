"""install.sh ต้องสร้าง venv ได้บนเครื่องที่ไม่มี python3-venv โดยไม่ต้องใช้ sudo

เคสจริง 2026-09-03: เพิ่ม node RTX4000 (Ubuntu 24.04, Python 3.12.3) จากหน้าเว็บ · `python3 -m venv
--help` ผ่านแต่ `python3 -m venv DIR` ล้มด้วย "ensurepip is not available" · install.sh ตายพร้อมคำแนะนำ
ให้ลบ venv ทิ้งแล้วลองใหม่ ซึ่งไม่เกี่ยวกับสาเหตุเลย · หน้าเว็บรัน install.sh แบบไม่มี sudo จึงต้องมีทาง
ที่ไม่ต้อง sudo — venv --without-pip แล้วดึง pip มาเอง
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "install.sh"


def _function(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    start = text.index(f"\n{name}() {{") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _shim(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _fake_python_without_ensurepip(shims: Path, log: Path) -> None:
    # python3 ที่ "มี venv แต่ไม่มี ensurepip" เหมือน Ubuntu ที่ยังไม่ลง python3-venv
    _shim(shims, "python3", f'''
echo "python3 $*" >> "{log}"
if [[ "$1" == "-c" && "$2" == "import ensurepip" ]]; then exit 1; fi
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
  shift 2; without=0; dir=""
  for a in "$@"; do case "$a" in --without-pip) without=1 ;; --clear) ;; *) dir="$a" ;; esac; done
  if (( without == 0 )); then
    echo "Error: Command '[...]' returned non-zero exit status 1." >&2
    echo "The virtual environment was not created successfully because ensurepip is not available." >&2
    exit 1
  fi
  mkdir -p "$dir/bin"
  printf '#!/usr/bin/env bash\\n# fake venv python: running get-pip installs pip\\ncase "$1" in *get-pip*|/tmp/*) touch "$(dirname "$0")/pip" ;; esac\\n' > "$dir/bin/python"
  chmod +x "$dir/bin/python"
  exit 0
fi
exit 0
''')


def _run(shims: Path, script: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(shims), **(env_extra or {})}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)


def _prelude() -> str:
    return "\n".join([_function("die"), _function("ask_yes"), _function("sudo_run"), _function("make_venv")])


def test_venv_is_created_without_pip_and_pip_is_bootstrapped_when_ensurepip_is_missing(tmp_path):
    shims = tmp_path / "shims"; shims.mkdir()
    log = tmp_path / "calls.log"
    _fake_python_without_ensurepip(shims, log)
    _shim(shims, "curl", 'echo "curl $*" >> "%s"; out=""; while (( $# )); do [[ "$1" == "-o" ]] && out="$2"; shift; done; echo "# fake get-pip" > "$out"' % log)
    _shim(shims, "sudo", 'echo "SUDO WAS CALLED: $*" >> "%s"; exit 1' % log)
    venv = tmp_path / "venv"
    out = _run(shims, _prelude() + f'\nmake_venv "{venv}"', {"LMDS_SKIP_PREREQ": "1", "LMDS_ASSUME_YES": "1"})
    assert out.returncode == 0, out.stderr + out.stdout
    assert (venv / "bin" / "pip").exists(), "pip ไม่ถูก bootstrap หลัง --without-pip"
    calls = log.read_text()
    assert "--without-pip" in calls
    assert "SUDO WAS CALLED" not in calls, "หน้าเว็บรันแบบไม่มี sudo — ห้ามพยายาม sudo"
    assert "bootstrap.pypa.io/get-pip.py" in calls


def test_plain_venv_is_used_when_ensurepip_exists(tmp_path):
    shims = tmp_path / "shims"; shims.mkdir()
    log = tmp_path / "calls.log"
    _shim(shims, "python3", f'echo "python3 $*" >> "{log}"; [[ "$1" == "-m" && "$2" == "venv" ]] && mkdir -p "${{@: -1}}/bin"; exit 0')
    _shim(shims, "curl", 'echo "CURL WAS CALLED" >> "%s"; exit 1' % log)
    venv = tmp_path / "venv"
    out = _run(shims, _prelude() + f'\nmake_venv "{venv}"')
    assert out.returncode == 0, out.stderr
    calls = log.read_text()
    assert "--without-pip" not in calls and "CURL WAS CALLED" not in calls


def test_failure_message_names_the_real_fix_not_rm_rf():
    """คำแนะนำเดิม 'ลบทิ้งแล้วลองใหม่' ไม่เกี่ยวกับสาเหตุ — ต้องชี้ไปที่ python3-venv"""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "ลบทิ้งแล้วลองใหม่: rm -rf" not in text
    assert "sudo apt install python3-venv" in text
