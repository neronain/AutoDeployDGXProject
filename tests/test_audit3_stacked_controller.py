"""Audit รอบ 3 (2026-10-06) — stacked controller: สิ่งที่มันรายงาน ต้องตรงกับของจริงบนทุก node

รอบแรกไล่ "ล้มแล้วไม่บอกสาเหตุ" · รอบสองไล่ "ทำงานผิดเงียบ ๆ" · รอบนี้ไล่ **คำรายงานที่ไม่ตรงกับของจริง**:
stop ที่พิมพ์ stopped ทั้งที่ ssh ไป worker ไม่ถึง · info ที่บอก RUNNING แทนโมเดลของคนอื่นบนพอร์ตเดียวกัน ·
test-text ที่คืน 0 กับคำตอบว่าง · verify-files ที่รับ snapshot คนละ revision · knob ที่บันทึกผิดแล้วทำให้ stop ไม่ได้ ·
start ที่ล้มกลางทางแล้วทิ้ง worker ถือ GPU ไว้โดยไม่บอก

ทุกข้อ render bundle จริงแล้วรัน controller ทั้งสคริปต์ใต้ bash กับ docker/ssh ปลอมบน PATH · docker ปลอมของไฟล์นี้
**จำสถานะ container ต่อ node** (ไฟล์ใต้ FAKE_STATE/<node>/<ชื่อ container>) เพราะเรื่องที่ตรวจคือ "หลังสั่งแล้วของจริง
เป็นอย่างไร" — shim ที่ตอบสำเร็จเสมอพิสูจน์เรื่องนี้ไม่ได้
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

from tests.test_audit_stacked_controller import (
    SAFE_PATH, _SSH, _bundle, _calls, _seed_head_cache, _shim,
)

HEAD = "10.1.1.1"
W1, W2, W3 = "10.1.1.2", "10.1.1.3", "10.1.1.4"
MODEL = "nvidia/DeepSeek-V4-Flash-NVFP4"
# controller เป้าหมายคือ bash 5 บน Linux · macOS มีแต่ 3.2 ที่ /bin/bash — ตั้ง LMDS_TEST_BASH=/opt/homebrew/bin/bash
# เพื่อรันชุดนี้ใต้ bash 5 บนเครื่อง dev (CI เป็น Linux อยู่แล้ว)
BASH = os.environ.get("LMDS_TEST_BASH") or "bash"


# ───────────────────────── shims ─────────────────────────
# docker ปลอมที่จำสถานะ: `run -d --name X` สร้างไฟล์สถานะ · `rm -f X` ลบ · `ps`/`inspect` ตอบจากไฟล์นั้น
#   FAKE_DOCKER_DOWN="<node> …"   daemon ของ node นั้นไม่ตอบ (ทุกคำสั่งล้ม)
#   FAKE_RM_STUCK="<node> …"      rm -f คืน 0 แต่ container ยังอยู่
#   FAKE_RUN_FAIL="<node> …"      docker run -d ล้ม (GPU runtime หาย)
#   FAKE_RUN_DIES="<node> …"      docker run -d สำเร็จ แต่ container ตายทันที (Exited)
#   FAKE_HEAD_RUN_CUTS_SSH="<ip>" ตั้งแต่ head ถูกสั่ง run เป็นต้นไป ssh ไป worker ตัวนี้ล่ม (สายหลุดกลาง start)
#   FAKE_HEAD_RUN_KILLS="<ip>"    worker ตัวนี้ตายตอน head เริ่ม (OOM ระหว่างโหลด weight — หลังผ่านด่านก่อนเปิด head ไปแล้ว)
_DOCKER = r'''
node="${FAKE_NODE:-head}"
echo "docker[${node}] $*" >> "$FAKE_LOG"
state="${FAKE_STATE}/${node}"; mkdir -p "$state"
on() { case " $1 " in *" $node "*) return 0 ;; esac; return 1; }
if on "${FAKE_DOCKER_DOWN:-}"; then
  echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1
fi
last="${@: -1}"
case "$1" in
  image) exit 0 ;;
  ps)
    all=0; quiet=0; fmt=""; name=""; shift
    while (( $# )); do
      case "$1" in
        -a) all=1; shift ;;
        -q) quiet=1; shift ;;
        --filter) name="${2#name=^}"; name="${name%\$}"; shift 2 ;;
        --format) fmt="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    [[ -n "$name" && -f "$state/$name" ]] || exit 0
    st="$(cat "$state/$name")"
    [[ "$st" == running || "$all" == 1 ]] || exit 0
    if (( quiet )); then echo "cid-$name"; exit 0; fi
    status="Up 2 hours"; [[ "$st" == running ]] || status="Exited (1) 3 minutes ago"
    case "$fmt" in
      *Names*Status*) printf '%s\t%s\n' "$name" "$status" ;;
      *Status*) echo "$status" ;;
      *) echo "$name" ;;
    esac
    exit 0 ;;
  rm) on "${FAKE_RM_STUCK:-}" || rm -f "$state/$last"; exit 0 ;;
  inspect)
    if [[ "$*" == *".Id"* ]]; then echo "sha256:aaaa1111"; exit 0; fi
    if [[ "$*" == *".State.Running"* ]]; then
      [[ -f "$state/$last" ]] || { echo "Error: No such object: $last" >&2; exit 1; }
      if [[ "$(cat "$state/$last")" == running ]]; then echo true; else echo false; fi
    fi
    exit 0 ;;
  container) [[ -f "$state/$last" ]]; exit $? ;;
  run)
    name=""; prev=""; detached=0
    for a in "$@"; do
      [[ "$prev" == "--name" ]] && name="$a"
      [[ "$a" == "-d" ]] && detached=1
      prev="$a"
    done
    if (( detached )); then
      if on "${FAKE_RUN_FAIL:-}"; then
        echo 'docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].' >&2; exit 125
      fi
      st=running; on "${FAKE_RUN_DIES:-}" && st=exited
      echo "$st" > "$state/$name"; echo "cid-$name"
      if [[ "$node" == head && -n "${FAKE_HEAD_RUN_CUTS_SSH:-}" ]]; then echo "$FAKE_HEAD_RUN_CUTS_SSH" > "$FAKE_STATE/ssh-down"; fi
      if [[ "$node" == head && -n "${FAKE_HEAD_RUN_KILLS:-}" ]]; then
        for f in "$FAKE_STATE/$FAKE_HEAD_RUN_KILLS"/lmds-*-worker; do [[ -f "$f" ]] && echo exited > "$f"; done
      fi
    fi
    exit 0 ;;
  logs) echo "${FAKE_LOGS:-}"; exit 0 ;;
  *) exit 0 ;;
esac
'''

# ssh ปลอมของชุดเดิม + worker ที่ต่อไม่ถึง: FAKE_SSH_DOWN="<ip> …" (หรือไฟล์ FAKE_STATE/ssh-down ที่ docker ปลอมเขียน)
# → ข้อความจริงของ ssh แล้วคืน 255
_SSH_DOWN = '''
case " ${FAKE_SSH_DOWN:-} $(cat "$FAKE_STATE/ssh-down" 2>/dev/null) " in
  *" $node "*) echo "ssh: connect to host $node port 22: Connection timed out" >&2; exit 255 ;;
esac
'''
_SSH3 = _SSH.replace('export FAKE_NODE="$node"', _SSH_DOWN + 'export FAKE_NODE="$node"')

# ip ปลอม: ทุกเครื่องของคลัสเตอร์ทดสอบ (head + worker สามตัว) มี IP ของตัวเองบน interface สักตัว
_IP = '''
if [[ "$1" == "-o" ]]; then
  n=1
  for ip in 10.1.1.1 10.1.1.2 10.1.1.3 10.1.1.4; do
    n=$(( n + 1 )); echo "${n}: eth${n}    inet ${ip}/24 brd 10.1.1.255 scope global eth${n}"
  done
fi
exit 0
'''
_DF = '''
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "fake 1 1 999999999999 1% /"
'''


def _bin(tmp_path: Path, **extra: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in {"ssh": _SSH3, "docker": _DOCKER, "ip": _IP, "df": _DF, "sudo": "exit 0\n", **extra}.items():
        _shim(bin_dir, name, body)
    return bin_dir


def _env(tmp_path: Path, env: dict | None = None, workers: str = W1) -> dict:
    remote = tmp_path / "remote"
    remote.mkdir(exist_ok=True)
    (tmp_path / "home").mkdir(exist_ok=True)
    full = {
        "PATH": f"{tmp_path / 'bin'}:{SAFE_PATH}", "HOME": str(tmp_path / "home"),
        "FAKE_LOG": str(tmp_path / "calls.log"), "FAKE_REMOTE": str(remote), "FAKE_STATE": str(tmp_path / "state"),
        "CLUSTER_ENV": "/nonexistent/cluster.env", "WORKER_INIT_WAIT": "0", "WORKER_CHECK_INTERVAL": "0",
        "RUN_DIR": str(tmp_path / "run"),
        "MASTER_IP": HEAD, "WORKER_IP": workers.split()[0], "WORKER_IPS": workers, "SSH_USER": "neronain",
    }
    full.update(env or {})
    return full


def _run(bundle, cmd: list[str], tmp_path: Path, env: dict | None = None, workers: str = W1,
         timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(bundle.controller), *cmd], env=_env(tmp_path, env, workers),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _names(bundle) -> tuple[str, str]:
    slug = bundle.directory.name
    return f"lmds-{slug}-head", f"lmds-{slug}-worker"


def _container(tmp_path: Path, node: str, name: str) -> Path:
    return tmp_path / "state" / node / name


def _up(tmp_path: Path, bundle, *nodes: str, state: str = "running") -> None:
    """ตั้งต้นว่า container ของ bundle นี้อยู่บน node พวกนี้ ("head" หรือ IP ของ worker)"""
    head, worker = _names(bundle)
    for node in nodes:
        path = _container(tmp_path, node, head if node == "head" else worker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state + "\n", encoding="utf-8")


def _left(tmp_path: Path, bundle) -> set[str]:
    """node ที่ยังมี container ของ bundle นี้ **รันอยู่** หลังคำสั่งจบ — สิ่งที่ถือ GPU memory จริง"""
    head, worker = _names(bundle)
    root = tmp_path / "state"
    out = set()
    for node in (p.name for p in root.iterdir()) if root.exists() else ():
        path = root / node / (head if node == "head" else worker)
        if path.exists() and path.read_text(encoding="utf-8").strip() == "running":
            out.add(node)
    return out


# ═════════════════════ 1. stop: รายงานตามที่เห็นบนแต่ละ node ═════════════════════
def test_stop_names_the_worker_it_could_not_reach_and_still_stops_the_rest(tmp_path):
    """worker ตัวกลางต่อ ssh ไม่ถึง (255) → เดิมกลืนด้วย `|| true` แล้วพิมพ์ "stopped …-head + …-worker" rc 0 ทั้งที่
    container บนเครื่องนั้นยังรันและถือ GPU memory · ต้อง: หยุด node อื่นให้ครบ (ไม่หยุดที่ตัวแรกที่ล้ม) · บอกชื่อเครื่องที่
    ไปไม่ถึง · บอกว่าอาจยังรันอยู่ · exit ไม่เป็นศูนย์ · ไม่พิมพ์บรรทัดสรุปว่าหยุดครบ"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    workers = f"{W1} {W2} {W3}"
    _up(tmp_path, bundle, "head", W1, W2, W3)
    head, worker = _names(bundle)

    done = _run(bundle, ["stop"], tmp_path, workers=workers, env={"FAKE_SSH_DOWN": W2})

    assert done.returncode != 0, done.stdout + done.stderr
    assert _left(tmp_path, bundle) == {W2}, "ตัวที่ไปถึงต้องถูกหยุดทั้งหมด — รวมตัวที่อยู่ **หลัง** ตัวที่ล้ม"
    said_bad = [ln for ln in done.stderr.splitlines() if W2 in ln]
    assert said_bad and any("GPU" in ln for ln in said_bad), done.stderr
    assert "Connection timed out" in done.stderr, "ข้อความของ ssh เองต้องถึงผู้ใช้"
    assert not any(W1 in ln or W3 in ln for ln in done.stderr.splitlines() if "GPU" in ln), \
        "เครื่องที่หยุดสำเร็จต้องไม่ถูกเหมารวมว่าอาจยังรัน"
    assert f"stopped {head} + {worker}" not in done.stdout + done.stderr
    # ต่อ node ที่หยุดได้ ต้องพูดถึงเครื่องนั้น
    for node in (W1, W3):
        assert any(node in ln and worker in ln for ln in done.stdout.splitlines()), done.stdout


def test_stop_reports_success_only_after_seeing_every_container_gone(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    _up(tmp_path, bundle, "head", W1, W2)
    head, worker = _names(bundle)

    done = _run(bundle, ["stop"], tmp_path, workers=f"{W1} {W2}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _left(tmp_path, bundle) == set()
    assert f"stopped {head} + {worker}" in done.stdout

    # หลังสั่งลบแล้วต้อง "ดู" อีกครั้งบนเครื่องนั้น — ลำดับคือสาระ: ps → rm -f → ps
    for node in ("head", W1, W2):
        seq = [ln.split()[1] for ln in _calls(tmp_path).splitlines() if ln.startswith(f"docker[{node}] ")]
        assert seq.index("rm") < len(seq) - 1 - seq[::-1].index("ps"), f"{node}: ไม่ได้ตรวจซ้ำหลัง rm -f ({seq})"


def test_stop_says_nothing_was_running_instead_of_stopped(tmp_path):
    """ไม่มี container สักตัว → เดิมก็พิมพ์ "stopped …" · ชื่อ container ที่เพี้ยนจึงดูเหมือนหยุดสำเร็จ"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, ["stop"], tmp_path, workers=f"{W1} {W2}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "stopped " not in done.stdout, done.stdout
    assert "ไม่มีอะไรให้หยุด" in done.stdout


@pytest.mark.parametrize("fault", ["FAKE_RM_STUCK", "FAKE_DOCKER_DOWN"])
def test_stop_does_not_claim_a_node_it_could_not_confirm(tmp_path, fault):
    """ssh ถึง แต่ container ไม่หายหลัง rm -f (daemon ค้าง) หรือถาม docker บนเครื่องนั้นไม่ได้เลย = ยังไม่รู้ว่าหยุด"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    _up(tmp_path, bundle, "head", W1, W2)
    done = _run(bundle, ["stop"], tmp_path, workers=f"{W1} {W2}", env={fault: W1})
    assert done.returncode != 0, done.stdout + done.stderr
    assert _left(tmp_path, bundle) == {W1}
    assert any(W1 in ln and "GPU" in ln for ln in done.stderr.splitlines()), done.stderr
    assert not any(W2 in ln for ln in done.stderr.splitlines() if "GPU" in ln), done.stderr
    assert "stopped lmds-" not in done.stdout


def test_restart_does_not_start_on_top_of_a_worker_it_could_not_stop(tmp_path):
    """restart = stop แล้ว start · stop ที่หยุดไม่ครบต้องจบตรงนั้น ไม่ใช่เปิด worker ชุดใหม่ทับตัวที่ยังถือ GPU"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    _up(tmp_path, bundle, "head", W1, W2)
    done = _run(bundle, ["restart"], tmp_path, workers=f"{W1} {W2}", env={"FAKE_SSH_DOWN": W2})
    assert done.returncode != 0
    assert "run -d" not in _calls(tmp_path), "ห้าม start ต่อหลัง stop ไม่ครบ"
    assert any(W2 in ln and "GPU" in ln for ln in done.stderr.splitlines()), done.stderr
    assert "restart" in done.stderr.split("ERROR:")[-1]
    assert _left(tmp_path, bundle) == {W2}, "ตัวที่ไปถึงยังต้องถูกหยุด"


# ═════════════════════ 5. knob ที่บันทึกผิด ต้องไม่ทำให้หยุด/ดูสถานะไม่ได้ ═════════════════════
def _bad_knob(bundle) -> None:
    # สิ่งที่ `lmds set --gpu-util 0.99` เขียน (hub รับ 0–1 · controller รับ 0.3–0.98)
    (bundle.directory / "bundle.env").write_text('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.99}"\n',
                                                 encoding="utf-8")


def test_a_saved_out_of_range_knob_does_not_stop_the_controller_from_stopping(tmp_path):
    """เคส audit: bundle.env มี gpu-util 0.99 → validate_numbers รันก่อน dispatch → stop/status/logs ตายด้วย
    "invalid --gpu-util" · โมเดล stacked ที่รันอยู่หยุดผ่าน controller ของตัวเองไม่ได้"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n")
    _bad_knob(bundle)
    _up(tmp_path, bundle, "head", W1)

    status = _run(bundle, ["status"], tmp_path)
    assert status.returncode == 0, status.stderr
    logs = _run(bundle, ["logs", "50"], tmp_path)
    assert logs.returncode == 0, logs.stderr
    info = _run(bundle, ["info"], tmp_path)
    assert info.returncode == 0, info.stderr

    stop = _run(bundle, ["stop"], tmp_path)
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert _left(tmp_path, bundle) == set(), "stop ต้องหยุดได้จริง ไม่ใช่แค่ไม่ error"


@pytest.mark.parametrize("verb", ["start", "restart"])
def test_start_and_restart_still_refuse_the_bad_knob_and_say_how_to_fix_it(tmp_path, verb):
    """คำสั่งที่เปิด engine ยังต้องปฏิเสธ — และ restart ต้องปฏิเสธ **ก่อน** หยุดโมเดลที่รันอยู่"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    _bad_knob(bundle)
    _up(tmp_path, bundle, "head", W1)
    done = _run(bundle, [verb], tmp_path)
    assert done.returncode != 0
    assert "0.99" in done.stderr and "0.98" in done.stderr
    fixes = [ln for ln in done.stderr.splitlines() if "--gpu-util" in ln and ("lmds set" in ln or f" {verb} " in ln)]
    assert len(fixes) >= 2, f"ต้องบอกทั้งทางแก้ถาวรและทางแก้ครั้งนี้: {done.stderr}"
    assert _left(tmp_path, bundle) == {"head", W1}, "ค่าผิดต้องถูกปฏิเสธก่อนแตะ container ที่รันอยู่"
    assert "rm -f" not in _calls(tmp_path) and "run -d" not in _calls(tmp_path)

    # ค่าที่ใช้ได้ผ่านทาง flag = ไปต่อได้ (ไม่ใช่ปฏิเสธเหมา)
    ok = _run(bundle, ["restart", "--gpu-util", "0.9"], tmp_path)
    assert ok.returncode == 0, ok.stdout + ok.stderr
