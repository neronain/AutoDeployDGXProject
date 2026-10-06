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

import json
import os
import socket
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


# ═════════════════════ 6. start ล้มกลางทาง ═════════════════════
def test_a_worker_that_fails_to_launch_does_not_leave_the_earlier_workers_running(tmp_path):
    """คลัสเตอร์ 3 worker: rank 1 เปิดแล้ว → rank 2 `docker run` ล้ม → เดิม die "start worker container … ไม่สำเร็จ"
    แล้วออก ทิ้ง rank 1 ค้างรอ head ที่ไม่มีวันมา ถือ GPU memory โดยไม่มีบรรทัดไหนบอก"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2} {W3}", env={"FAKE_RUN_FAIL": W2})
    assert done.returncode != 0
    assert _left(tmp_path, bundle) == set(), f"worker ที่เปิดไปแล้วยังรันอยู่: {done.stderr}"
    assert f"docker[{W3}] run -d" not in _calls(tmp_path), "ล้มที่ rank 2 แล้วต้องไม่เปิด rank 3 ต่อ"
    assert any(W1 in ln and "หยุด" in ln for ln in done.stderr.splitlines()), done.stderr
    # สาเหตุต้องยังเป็นข้อความท้ายสุด (hub/คน อ่านท้าย) — รายงานการเก็บกวาดมาก่อน
    assert done.stderr.index("start ล้มกลางทาง") < done.stderr.rindex("ERROR:"), done.stderr
    assert W2 in done.stderr.split("ERROR:")[-1], "บรรทัดสาเหตุต้องชี้เครื่องที่ล้ม"


def test_a_worker_that_dies_before_the_head_starts_takes_the_other_workers_down_and_keeps_its_own_log(tmp_path):
    """worker หนึ่งตัวตายตอน init ("worker … หยุดก่อน head จะเริ่ม") → เดิมออกเลย worker อีกตัวยังรัน · ตัวที่ตายเอง
    ต้องไม่ถูกลบ — `logs worker` ยังต้องอ่าน log ของมันได้"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    _, worker = _names(bundle)
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}", env={"FAKE_RUN_DIES": W2})
    assert done.returncode != 0
    assert _left(tmp_path, bundle) == set(), done.stderr
    assert _container(tmp_path, W2, worker).exists(), "container ที่ตายเองคือหลักฐาน ต้องไม่ถูกลบ"
    assert "docker[head] run -d" not in _calls(tmp_path)
    assert any(W1 in ln and "หยุด" in ln for ln in done.stderr.splitlines()), done.stderr


def test_a_head_that_dies_before_health_does_not_leave_the_workers_holding_gpu_memory(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n")
    _seed_head_cache(tmp_path / "home")
    head, _ = _names(bundle)
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}", env={"FAKE_RUN_DIES": "head"})
    assert done.returncode != 0
    assert "head container" in done.stderr
    assert _left(tmp_path, bundle) == set(), done.stderr
    assert _container(tmp_path, "head", head).exists(), "head ที่ตายเองต้องเก็บไว้ให้ `logs head` อ่าน"
    for node in (W1, W2):
        assert any(node in ln and "หยุด" in ln for ln in done.stderr.splitlines()), done.stderr


def test_a_worker_that_dies_while_the_head_loads_takes_the_head_and_the_other_workers_down(tmp_path):
    """worker ตายระหว่างรอ head health → เดิมลบแค่ head แล้วออก: คลัสเตอร์ 3 เครื่องเหลือ worker อีกตัวถือ GPU ไว้
    โดยข้อความบอกแค่ "หยุด head แล้ว" · ตัวที่ตายเองเก็บไว้ให้ดู log"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n")
    _seed_head_cache(tmp_path / "home")
    head, worker = _names(bundle)
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}", env={"FAKE_HEAD_RUN_KILLS": W2})
    assert done.returncode != 0
    assert W2 in done.stderr.split("ERROR:")[-1]
    assert _left(tmp_path, bundle) == set(), done.stderr
    assert not _container(tmp_path, "head", head).exists(), "head ที่ยังรันต้องถูกหยุดจริง"
    assert _container(tmp_path, W2, worker).exists(), "worker ที่ตายเองต้องเก็บไว้ให้ `logs worker` อ่าน"
    for label in ("head", W1):
        assert any(label in ln and "หยุด" in ln for ln in done.stderr.split("ERROR:")[0].splitlines()), done.stderr


def test_a_rollback_that_cannot_reach_a_worker_says_which_one_is_still_running(tmp_path):
    """เก็บกวาดแล้วไปไม่ถึงบางเครื่อง (สาย management หลุดหลัง worker เปิดไปแล้ว) ต้องบอกชื่อเครื่องนั้น —
    ห้ามออกโดยทิ้ง container ไว้เงียบ ๆ และห้ามเหมาว่าเครื่องที่หยุดได้ก็ยังรัน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n")
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}",
                env={"FAKE_RUN_DIES": "head", "FAKE_HEAD_RUN_CUTS_SSH": W1})
    assert done.returncode != 0
    assert _left(tmp_path, bundle) == {W1}
    assert any(W1 in ln and "GPU" in ln for ln in done.stderr.splitlines()), done.stderr
    assert not any(W2 in ln and "GPU" in ln for ln in done.stderr.splitlines()), done.stderr


def test_a_health_timeout_leaves_the_cluster_loading_and_says_exactly_what_is_still_up(tmp_path):
    """หมดเวลารอ health ≠ ล้ม (โมเดลใหญ่โหลดเกินได้) — ตั้งใจไม่หยุด แต่ต้องบอกครบทุกเครื่องที่ยังถือ GPU และวิธีคืน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n", sleep="exit 0\n")      # sleep ปลอม: ลูปรอหมุนจนครบ STARTUP_TIMEOUT โดยไม่ต้องรอ 10 วิ/รอบ
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}", env={"STARTUP_TIMEOUT": "1", "STUCK_HINT_AFTER": "9999"})
    assert done.returncode != 0
    assert _left(tmp_path, bundle) == {"head", W1, W2}, "timeout ต้องไม่ฆ่าโมเดลที่กำลังโหลด"
    last = done.stderr.split("ERROR:")[-1]
    assert W1 in last and W2 in last and "head" in last and " stop" in last, last


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_cancelled_start_never_kills_the_cluster_that_is_still_loading(tmp_path, sig):
    """hub ยกเลิก job / สาย ssh ของ hub หลุด / Ctrl-C ระหว่างรอ health (โมเดลใหญ่ = ชั่วโมงกว่า) ต้องไม่ใช่เหตุให้เก็บกวาด —
    การหยุดของที่รอบนี้เปิดไว้เป็นของ "start ล้ม" เท่านั้น · container ต้องยังรันครบ และบอกว่าอะไรยังรันอยู่"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 7\n")
    _seed_head_cache(tmp_path / "home")
    proc = subprocess.Popen([BASH, str(bundle.controller), "start"], env=_env(tmp_path, workers=f"{W1} {W2}"),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while "docker[head] run -d" not in _calls(tmp_path):        # รอจนเข้าช่วงรอ health จริง
            assert proc.poll() is None and time.monotonic() < deadline, proc.communicate()
            time.sleep(0.1)
        time.sleep(0.5)
        if sig == signal.SIGINT:
            os.killpg(proc.pid, sig)      # Ctrl-C ไปทั้ง process group (bash + sleep ที่มันรออยู่) — ส่งให้ bash ตัวเดียวมันไม่ออก
        else:
            proc.send_signal(sig)         # hub: proc.terminate() / สาย ssh หลุด — ถึง controller ตัวเดียว
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode != 0
    assert _left(tmp_path, bundle) == {"head", W1, W2}, err
    assert "rm -f" not in _calls(tmp_path).split("docker[head] run -d")[-1], "ถูกขัดจังหวะ ≠ ล้ม — ห้ามลบอะไร"
    assert W1 in err and W2 in err and " stop" in err, f"ต้องบอกว่าอะไรยังรันอยู่และหยุดอย่างไร: {err!r}"


def test_a_successful_start_does_not_roll_anything_back(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], tmp_path, workers=f"{W1} {W2}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _left(tmp_path, bundle) == {"head", W1, W2}
    assert "start ล้มกลางทาง" not in done.stderr and "start จบก่อนเสร็จ" not in done.stderr


# ═════════════════════ 2. info / รอ health: 200 ที่พอร์ต ≠ โมเดลของเรา ═════════════════════
class _Api(BaseHTTPRequestHandler):
    """เซิร์ฟเวอร์ปลอมบนพอร์ตของโมเดล — ตั้งต่อคลาสย่อย: `models` = ชื่อที่ /v1/models ประกาศ · int = ตอบ status นั้น
    (401 = มี API key) · None = ไม่ใช่ OpenAI API (ตอบหน้าเว็บ 200 ทุก path แบบ portainer) · `reply` = body ของ chat"""
    models: list[str] | int | None = []
    reply: bytes = b"{}"

    def log_message(self, *_):
        pass

    def _send(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(b"")
        if self.models is None:
            return self._send(b"<html><body>portainer</body></html>")
        if isinstance(self.models, int):
            return self._send(b'{"error": "Unauthorized"}', self.models)
        self._send(json.dumps({"object": "list", "data": [
            {"id": m, "object": "model", "permission": [{"id": "modelperm-1"}]} for m in self.models]}).encode())

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(self.reply)


@pytest.fixture
def api():
    started = []

    def start(models, reply: bytes | dict = b"{}") -> str:
        body = json.dumps(reply).encode() if isinstance(reply, dict) else reply
        handler = type("Handler", (_Api,), {"models": models, "reply": body})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return str(server.server_address[1])

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _state_line(stdout: str) -> str:
    return next(ln for ln in stdout.splitlines() if ln.strip().startswith("State"))


def test_info_does_not_call_another_model_on_the_same_port_running(tmp_path, api):
    """เคส audit: ไม่มี container ของ bundle นี้เลย แต่โมเดลอื่นตอบ 200 ที่พอร์ตเดียวกัน → เดิม "State : RUNNING" """
    bundle = _bundle(tmp_path)
    _bin(tmp_path)                       # ไม่มี curl ปลอม — ใช้ curl จริงกับเซิร์ฟเวอร์ปลอม
    port = api(["some-other-model"])
    done = _run(bundle, ["info", "--port", port], tmp_path, env={"SERVED_MODEL_NAME": "ours"})
    assert done.returncode == 0, done.stderr
    assert "RUNNING" not in _state_line(done.stdout), done.stdout
    assert f"port {port} is answered by something else" in done.stdout and "some-other-model" in done.stdout


def test_info_says_running_only_for_our_container_and_our_served_name(tmp_path, api):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    env = {"SERVED_MODEL_NAME": "ours"}

    # (ก) container ของเรารัน + พอร์ตประกาศชื่อเรา (มากับชื่ออื่นก็ได้ — vLLM ประกาศ LoRA/alias ร่วมกัน)
    _up(tmp_path, bundle, "head")
    port = api(["an-alias", "ours"])
    ours = _run(bundle, ["info", "--port", port], tmp_path, env=env)
    assert "RUNNING" in _state_line(ours.stdout) and "something else" not in ours.stdout, ours.stdout

    # (ข) container ของเรารัน แต่พอร์ตประกาศชื่อคนอื่น = ยังไม่ใช่ของเรา
    port = api(["some-other-model"])
    other = _run(bundle, ["info", "--port", port], tmp_path, env=env)
    assert "RUNNING" not in _state_line(other.stdout) and "something else" in other.stdout, other.stdout

    # (ข2) container ของเรารัน แต่ตัวที่ตอบไม่ใช่ OpenAI API เลย (หน้าเว็บ 200 ทุก path) = ไม่ใช่ของเรา
    port = api(None)
    web = _run(bundle, ["info", "--port", port], tmp_path, env=env)
    assert "RUNNING" not in _state_line(web.stdout) and "something else" in web.stdout, web.stdout

    # (ข3) container ของเรารัน + /health ตอบ แต่ /v1/models ขอ key ที่คำสั่งนี้ไม่มี = ของเรา แต่ต้องบอกว่ายังไม่ได้เทียบชื่อ
    port = api(401)
    locked = _run(bundle, ["info", "--port", port], tmp_path, env=env)
    assert "RUNNING" in _state_line(locked.stdout) and "something else" not in locked.stdout, locked.stdout
    assert "API_KEY" in locked.stdout, "ต้องบอกว่ายืนยันชื่อโมเดลไม่ได้เพราะอะไร"

    # (ค) ไม่มีอะไรตอบเลย + container ของเรารัน = กำลังโหลด ไม่ใช่ stopped
    loading = _run(bundle, ["info", "--port", "1"], tmp_path, env=env)
    assert "stopped" not in _state_line(loading.stdout) and "RUNNING" not in _state_line(loading.stdout), loading.stdout

    # (ง) ไม่มีอะไรตอบ + ไม่มี container = stopped
    _container(tmp_path, "head", _names(bundle)[0]).unlink()
    down = _run(bundle, ["info", "--port", "1"], tmp_path, env=env)
    assert "stopped" in _state_line(down.stdout) and "something else" not in down.stdout, down.stdout


# curl ปลอม: "เซิร์ฟเวอร์อื่น" ที่ตอบ /health 200 และประกาศโมเดลชื่ออื่นที่พอร์ตของเรา (โผล่หลังด่านพอร์ตของ start —
# เซิร์ฟเวอร์จริงบนพอร์ตนั้นจะถูก check_port_free ขวางตั้งแต่ต้น ซึ่งเป็นคนละด่านกับที่เทสนี้ตรวจ)
_CURL_FOREIGN = '''
case "$*" in
  *"/v1/models"*) echo '{"object":"list","data":[{"id":"some-other-model","object":"model"}]}' ;;
esac
exit 0
'''


@pytest.mark.parametrize("head_alive", [False, True])
def test_start_does_not_report_success_because_another_server_answers_the_port(tmp_path, head_alive):
    """รอ health ของ start ใช้กติกาเดียวกับ info · เดิมเชื่อ /health 200 เปล่า ๆ → "Server พร้อม" rc 0 ทั้งที่
    (ก) head ของเราตายไปแล้ว (เช่น bind พอร์ตไม่ได้) หรือ (ข) head ยังโหลดอยู่ แต่ตัวที่ตอบคือโมเดลของคนอื่น"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl=_CURL_FOREIGN, sleep="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    env = {"SERVED_MODEL_NAME": "ours", "STARTUP_TIMEOUT": "1", "STUCK_HINT_AFTER": "9999", "API_PORT": _free_port()}
    if not head_alive:
        env["FAKE_RUN_DIES"] = "head"
    done = _run(bundle, ["start"], tmp_path, env=env)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "Server พร้อม" not in done.stdout and "started:" not in done.stdout
    assert f"port {env['API_PORT']} is answered by something else" in done.stdout + done.stderr
    assert "some-other-model" in done.stdout + done.stderr


# ═════════════════════ 3. test-text: exit code คือคำตอบ ═════════════════════
def _chat(content=None, **extra) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content, **extra}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("reply, passes", [
    (_chat("4"), True),
    (_chat("", reasoning_content="2+2 … still thinking"), True),          # คิดไม่จบ — ตามเดิม ไม่ใช่ความผิดพลาด
    (_chat(None, reasoning="2+2 … still thinking"), True),                # vLLM รุ่นใหม่ใช้ชื่อ reasoning
    (_chat(""), False),                                                   # เคส audit: คำตอบว่าง → เดิม rc 0
    (_chat(None), False),
    ({"data": [{"embedding": [0.1]}]}, False),                            # JSON ที่ไม่ใช่ chat completion
    ({"error": {"message": "model is overloaded", "code": 503}}, False),  # error object ใน 200
    ({"object": "error", "message": "boom"}, False),
    (b"<html>portainer</html>", False),                                   # อ่านเป็น JSON ไม่ได้
])
def test_test_text_fails_unless_the_model_actually_said_something(tmp_path, api, reply, passes):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    port = api(["ours"], reply)
    done = _run(bundle, ["test-text", "--port", port], tmp_path, env={"SERVED_MODEL_NAME": "ours"})
    assert (done.returncode == 0) is passes, f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    if not passes:
        assert "test-text: OK" not in done.stdout


def test_test_text_fails_when_it_cannot_check_the_answer_at_all(tmp_path, api):
    """ไม่มี python3 → เดิมข้ามการตรวจทั้งก้อน พิมพ์ JSON ดิบ แล้วคืน 0"""
    bundle = _bundle(tmp_path)
    bin_dir = _bin(tmp_path)
    port = api(["ours"], _chat("4"))
    # PATH ที่มีทุกอย่างที่ controller ใช้ ยกเว้น python3
    for tool in ("bash", "curl", "cat", "date", "dirname", "basename", "tr", "grep", "sed", "awk", "head", "tail", "id",
                 "env"):
        found = next((Path(d) / tool for d in ("/usr/bin", "/bin") if (Path(d) / tool).exists()), None)
        assert found, tool
        (bin_dir / tool).symlink_to(found)
    done = _run(bundle, ["test-text", "--port", port], tmp_path,
                env={"SERVED_MODEL_NAME": "ours", "PATH": str(bin_dir)})
    assert done.returncode != 0, done.stdout + done.stderr
    assert "python3" in done.stderr


# ═════════════════════ 4. verify-files / verify-worker: revision ที่ตรึงไว้เท่านั้น ═════════════════════
def test_verify_files_refuses_a_snapshot_of_another_revision(tmp_path):
    """เคส audit: cache มีแต่ snapshots/OLD-REVISION-0000 (ไฟล์ครบ ขนาดตรง) → เดิม "verify-files: OK (2 shards @
    …/OLD-REVISION-0000)" rc 0 แล้ว start ไปรัน `vllm serve --revision rev-ds4` แบบ offline ซึ่งหาไม่เจอ"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    _seed_head_cache(tmp_path / "home", rev="OLD-REVISION-0000")
    done = _run(bundle, ["verify-files"], tmp_path)
    assert done.returncode != 0, done.stdout
    assert "verify-files: OK" not in done.stdout
    assert "rev-ds4" in done.stderr and "OLD-REVISION-0000" in done.stderr and "download" in done.stderr

    # start ก็ต้องหยุดก่อนเปิด container ไหน
    _shim(tmp_path / "bin", "curl", "exit 0\n")
    started = _run(bundle, ["start"], tmp_path)
    assert started.returncode != 0 and "run -d" not in _calls(tmp_path)

    # revision ที่ตรึงมาอยู่ข้าง ๆ ของเก่า = ผ่าน และชี้ไปที่ตัวที่ตรึง
    _seed_head_cache(tmp_path / "home")
    ok = _run(bundle, ["verify-files"], tmp_path)
    assert ok.returncode == 0 and "snapshots/rev-ds4" in ok.stdout, ok.stdout + ok.stderr


def test_verify_files_still_follows_a_branch_ref_to_its_commit(tmp_path):
    """สิ่งที่ต้องไม่พัง: MODEL_REVISION เป็นชื่อ branch → refs/<branch> ชี้ commit → ใช้ snapshot ของ commit นั้น"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    snap = _seed_head_cache(tmp_path / "home", rev="c0ffee")
    refs = snap.parent.parent / "refs"
    refs.mkdir()
    (refs / "rev-ds4").write_text("c0ffee\n", encoding="utf-8")
    done = _run(bundle, ["verify-files"], tmp_path)
    assert done.returncode == 0 and "snapshots/c0ffee" in done.stdout, done.stdout + done.stderr


# docker ปลอมสำหรับ verify-worker: รันสคริปต์ python ที่ controller ส่งมาทาง stdin จริง ๆ โดยเบี่ยง /cache ไปที่ -v
_DOCKER_VERIFY = _DOCKER.replace('''  run)
''', '''  run)
    if [[ "$*" == *"--entrypoint python3"* && "$last" == "-" ]]; then
      shift; vol=""
      while (( $# )); do
        case "$1" in
          -v) vol="${2%%:*}"; shift 2 ;;
          -e) export "$2"; shift 2 ;;
          *) shift ;;
        esac
      done
      sed "s#/cache#${vol}#g" | python3 -
      exit $?
    fi
''')


def test_verify_worker_refuses_a_snapshot_of_another_revision_on_every_worker(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path, docker=_DOCKER_VERIFY)
    good, stale = tmp_path / "w-good", tmp_path / "w-stale"
    _seed_head_cache(good)
    _seed_head_cache(stale, rev="OLD-REVISION-0000")

    # worker เดียว ของครบ revision ตรง = ผ่าน (ยืนยันว่า shim รันสคริปต์จริง)
    ok = _run(bundle, ["verify-worker"], tmp_path, env={"WORKER_HF_HOME": str(good / ".cache/huggingface")})
    assert ok.returncode == 0 and "verify-worker: PASS" in ok.stdout, ok.stdout + ok.stderr

    done = _run(bundle, ["verify-worker"], tmp_path, env={"WORKER_HF_HOME": str(stale / ".cache/huggingface")})
    assert done.returncode != 0, done.stdout
    assert "verify-worker: PASS" not in done.stdout
    out = done.stdout + done.stderr
    assert "OLD-REVISION-0000" in out and "rev-ds4" in out and W1 in out, out
