"""ที่อยู่สำรองของ node ใช้ได้เฉพาะตอน **ต่อไม่ถึง** — ไม่ใช่ตอนคำสั่งช้า และไม่ใช่ตอนคำสั่งล้มเอง

`all_hosts` คือทางเข้าหลายทางของ **เครื่องเดียวกัน** (LAN + Tailscale) · audit 2026-10-06: `ssh.run`
ถือว่า exit 124 (timeout ของผู้เรียกเอง) และ 255 ที่ stderr ว่าง คือ "ต่อไม่ถึง" แล้วส่งคำสั่งเดิมไป
อีกที่อยู่หนึ่ง — เคสจริงที่ repro: `lmds start qwen3-coder --port 8001` ถูกรันสองรอบบน msi-1
(10.2.1.11 แล้ว 100.64.0.11) ผู้เรียกขอ timeout 3 วิ ได้ 6 วิ · process แรกยังรันอยู่บนเครื่องนั้น
(ไม่มี tty → ไม่มี SIGHUP) · start/install/remove และขั้น `sudo -S` ที่ถือรหัสผ่านโดนแบบเดียวกันหมด

เทสในไฟล์นี้รัน `ssh.run` / `push_file` / `stream` ของจริง กับ `ssh`/`scp` ปลอมบน PATH ที่บันทึกทุกครั้ง
ที่ถูกเรียก แล้วทำตัวตามแผนต่อที่อยู่ — ข้อความ error คัดจากที่ OpenSSH พิมพ์จริง (10.3p1 บน macOS:
"Operation timed out" · Linux: "Connection timed out")
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from lmds.nodes import Node, ssh

LAN, TAILNET = "10.2.1.11", "100.64.0.11"

# ssh/scp ปลอม — ตัวเดียวกันสองชื่อ · แผนต่อ host อยู่ในไฟล์ $FAKE_PLAN ("<host> <mode>" ต่อบรรทัด)
_FAKE = r"""#!/bin/bash
me=$(basename "$0")
host="" port=22 connect=10 prev="" last="" query=0
for a in "$@"; do
  case "$prev" in -p|-P) port="$a" ;; -o) case "$a" in ConnectTimeout=*) connect="${a#ConnectTimeout=}" ;; esac ;; esac
  case "$a" in -G) query=1 ;; *@*) [ -n "$host" ] || { host="${a#*@}"; host="${host%%:*}"; } ;; esac
  prev="$a"; last="$a"
done
mode=$(awk -v h="$host" '$1 == h { print $2; exit }' "$FAKE_PLAN")
if [ "$query" = 1 ]; then                       # ssh -G: พิมพ์ config ที่จะใช้ ไม่ต่อไปไหน
  echo "hostname ${mode#alias:}"; exit 0
fi
case "$mode" in alias:*) shown="${mode#alias:}"; mode=refused ;; *) shown="$host" ;; esac
echo "$me $host :: $last" >> "$FAKE_LOG"
case "$mode" in
  ok)          [ "$me" = ssh ] && { cat >/dev/null; echo "ran on $host"; }; exit 0 ;;
  refused)     echo "ssh: connect to host $shown port $port: Connection refused" >&2 ;;
  timedout)    echo "ssh: connect to host $shown port $port: Connection timed out" >&2 ;;
  timedout-mac) echo "ssh: connect to host $shown port $port: Operation timed out" >&2 ;;
  noroute)     echo "ssh: connect to host $shown port $port: No route to host" >&2 ;;
  netdown)     echo "ssh: connect to host $shown port $port: Network is unreachable" >&2 ;;
  unresolved)  echo "ssh: Could not resolve hostname $shown: Name or service not known" >&2 ;;
  blackhole)   sleep "$connect"; echo "ssh: connect to host $shown port $port: Connection timed out" >&2 ;;
  slow)        exec sleep 30 ;;                 # ต่อติดแล้ว — คำสั่งปลายทางแค่ใช้เวลานานกว่าที่ผู้เรียกรอ
  exit255)     exit 255 ;;                      # คำสั่งปลายทางออก 255 เอง ไม่มีอะไรบน stderr
  nested)      echo "ssh: connect to host 192.168.100.2 port 22: No route to host" >&2 ;;   # ssh ของ *คำสั่งปลายทาง* (head→worker) ล้ม
  denied)      echo "scp: dest open \"/x\": Permission denied" >&2; exit 1 ;;
  *)           echo "fake $me: no plan for '$host'" >&2; exit 99 ;;
esac
[ "$me" = scp ] && echo "scp: Connection closed" >&2
exit 255
"""


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """วาง ssh/scp ปลอมหน้า PATH · คืนฟังก์ชันตั้งแผน และตัวอ่าน log ว่าใครถูกเรียกด้วยคำสั่งอะไร"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("ssh", "scp"):
        fake = bin_dir / name
        fake.write_text(_FAKE, encoding="utf-8")
        fake.chmod(0o755)
    plan, log = tmp_path / "plan", tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_PLAN", str(plan))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    class Wire:
        def plan(self, **modes):
            plan.write_text("".join(f"{host} {mode}\n" for host, mode in modes.items()), encoding="utf-8")

        def hosts(self):
            return [line.split()[1] for line in log.read_text(encoding="utf-8").splitlines()]

        def commands(self):
            return [line.split(" :: ", 1)[1] for line in log.read_text(encoding="utf-8").splitlines()]

    return Wire()


def _node(**kw) -> Node:
    return Node(name="msi-1", host=LAN, user="tkc", alt_hosts=[TAILNET], **kw)


START = "lmds start qwen3-coder --port 8001"


# ── ต้องไม่ยิงซ้ำ ──────────────────────────────────────────────────────────────────────
def test_a_command_slower_than_the_callers_timeout_is_not_sent_to_the_other_address(wire):
    """timeout ของผู้เรียก = ต่อติดแล้ว คำสั่งช้า · เครื่องเดียวกันต้องไม่ได้ `lmds start` สองรอบ"""
    wire.plan(**{LAN: "slow", TAILNET: "slow"})
    began = time.time()
    result = ssh.run(_node(), START, timeout=2)
    took = time.time() - began

    assert result.exit_code == 124 and "หมดเวลา" in result.stderr
    assert wire.hosts() == [LAN], f"คำสั่งถูกส่งไป {wire.hosts()} — เครื่องเดียวกันสองรอบ"
    assert took < 3.5, f"ผู้เรียกขอ 2 วิ ได้ {took:.1f} วิ"


def test_a_step_that_carries_the_sudo_password_is_not_typed_twice(wire):
    """ขั้น `sudo -S` ส่งรหัสผ่านทาง stdin — ช้าแล้วยิงซ้ำ = สองคำสั่ง root พร้อมกันบนเครื่องเดียว"""
    wire.plan(**{LAN: "slow", TAILNET: "ok"})
    result = ssh.run(_node(), "sudo -S -p '' usermod -aG docker tkc", timeout=2, stdin_text="hunter2\n")
    assert result.exit_code == 124
    assert wire.hosts() == [LAN]


def test_a_remote_command_that_exits_255_by_itself_is_not_rerun(wire):
    """255 เปล่า ๆ ไม่ได้แปลว่าต่อไม่ถึง — คำสั่งปลายทางออก 255 เองได้ (bash ใต้ set -e ส่งต่อ exit ของลูก)"""
    wire.plan(**{LAN: "exit255", TAILNET: "ok"})
    result = ssh.run(_node(), START, timeout=10)
    assert result.exit_code == 255
    assert wire.hosts() == [LAN]


def test_a_remote_command_whose_own_ssh_failed_is_not_rerun(wire):
    """controller แบบ stacked ssh จาก head ไป worker เอง — worker ดับแล้ว stderr ของมันคือข้อความ
    "ssh: connect to host <worker> …" เป๊ะ ออก 255 เหมือนกัน · นั่นคือคำสั่งที่ *รันแล้ว* บน head
    ไม่ใช่ hub ที่ต่อ head ไม่ถึง — แยกได้จากชื่อ host ในบรรทัดนั้น"""
    wire.plan(**{LAN: "nested", TAILNET: "ok"})
    result = ssh.run(_node(), "lmds start stacked-model", timeout=10)
    assert result.exit_code == 255 and "192.168.100.2" in result.stderr
    assert wire.hosts() == [LAN]


# ── ต้องยังย้ายทางได้ ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("why", ["refused", "timedout", "timedout-mac", "noroute", "netdown", "unresolved"])
def test_an_address_that_cannot_be_reached_falls_through_to_the_next(wire, why):
    """อยู่นอกออฟฟิศ: LAN ต่อไม่ถึง → Tailscale · คำสั่งรันครั้งเดียว บนทางที่ต่อติด"""
    wire.plan(**{LAN: why, TAILNET: "ok"})
    result = ssh.run(_node(), START, timeout=10)
    assert result.ok and result.stdout.strip() == f"ran on {TAILNET}"
    assert wire.hosts() == [LAN, TAILNET]


def test_when_no_address_can_be_reached_the_last_real_error_comes_back(wire):
    wire.plan(**{LAN: "noroute", TAILNET: "refused"})
    result = ssh.run(_node(), START, timeout=10)
    assert result.exit_code == 255 and "Connection refused" in result.stderr
    assert wire.hosts() == [LAN, TAILNET]


def test_a_short_caller_timeout_still_leaves_room_to_try_the_next_address(wire):
    """ConnectTimeout ตายตัว 10 วิ + ผู้เรียกที่รอแค่ 4 วิ = หมดเวลาก่อน ssh จะบอกว่าต่อไม่ถึง →
    เดิมรอดเพราะนับ 124 เป็นต่อไม่ถึง · เลิกนับแล้วต้องให้ ssh ยอมแพ้เองทันเวลา"""
    wire.plan(**{LAN: "blackhole", TAILNET: "ok"})
    result = ssh.run(_node(), "true", timeout=4)
    assert result.ok, result.stderr
    assert wire.hosts() == [LAN, TAILNET]


def test_a_host_that_is_an_ssh_config_alias_still_fails_over(wire):
    """host ในทะเบียนเป็นชื่อใน ~/.ssh/config ได้ — ssh พิมพ์ HostName ที่แมปไว้ ไม่ใช่ชื่อที่เราส่ง"""
    wire.plan(**{"spark1": "alias:192.168.50.7", TAILNET: "ok"})
    node = Node(name="spark1", host="spark1", user="tkc", alt_hosts=[TAILNET])
    result = ssh.run(node, "true", timeout=10)
    assert result.ok and wire.hosts() == ["spark1", TAILNET]


# ── push_file: ลูปเดียวกัน ─────────────────────────────────────────────────────────────
def test_push_file_moves_on_only_when_the_address_cannot_be_reached(wire, tmp_path):
    payload = tmp_path / "bundle.tgz"
    payload.write_bytes(b"x" * 64)

    wire.plan(**{LAN: "refused", TAILNET: "ok"})
    assert ssh.push_file(_node(), str(payload), "bundle.tgz", timeout=10).ok
    assert wire.hosts() == [LAN, TAILNET]


def test_push_file_does_not_resend_after_a_timeout_or_a_real_scp_error(wire, tmp_path):
    payload = tmp_path / "bundle.tgz"
    payload.write_bytes(b"x" * 64)

    wire.plan(**{LAN: "slow", TAILNET: "ok"})
    slow = ssh.push_file(_node(), str(payload), "bundle.tgz", timeout=2)
    assert slow.exit_code == 124 and wire.hosts() == [LAN], "ไฟล์กำลังไปอยู่ทางแรก — ส่งซ้ำอีกทางคือสองตัวเขียนทับกัน"

    Path(os.environ["FAKE_LOG"]).write_text("", encoding="utf-8")
    wire.plan(**{LAN: "denied", TAILNET: "ok"})
    denied = ssh.push_file(_node(), str(payload), "/x", timeout=10)
    assert not denied.ok and "Permission denied" in denied.stderr
    assert wire.hosts() == [LAN]


# ── stream: เดิมใช้ all_hosts[0] อย่างเดียว ─────────────────────────────────────────────
def test_stream_reaches_a_node_that_only_answers_on_its_alternate_address(wire):
    """งานยาว (download · logs -f · job จากหน้าเว็บ) เคยล้มทันทีกับเครื่องที่เข้าได้ทาง Tailscale ทางเดียว
    ทั้งที่ `run` ต่อเครื่องเดียวกันได้ · คำสั่งจริงต้องถูกส่งครั้งเดียว — ที่เหลือคือการถามทาง (`true`)"""
    wire.plan(**{LAN: "noroute", TAILNET: "ok"})
    proc = ssh.stream(_node(), START)
    out = proc.stdout.read().decode()
    assert proc.wait() == 0 and f"ran on {TAILNET}" in out
    sent = [(host, cmd) for host, cmd in zip(wire.hosts(), wire.commands(), strict=True) if START in cmd]
    assert [host for host, _ in sent] == [TAILNET], "คำสั่งจริงต้องไปทางที่ต่อติด และไปครั้งเดียว"


def test_stream_on_a_single_address_node_makes_exactly_one_connection(wire):
    wire.plan(**{LAN: "ok"})
    proc = ssh.stream(Node(name="msi-1", host=LAN, user="tkc"), START)
    proc.stdout.read()
    assert proc.wait() == 0
    assert wire.hosts() == [LAN], "เครื่องที่มีทางเดียวไม่ต้องเสีย handshake ถามทาง"


def test_stream_stays_on_the_primary_when_it_answers(wire):
    wire.plan(**{LAN: "ok", TAILNET: "ok"})
    proc = ssh.stream(_node(), START)
    assert f"ran on {LAN}" in proc.stdout.read().decode() and proc.wait() == 0
    assert TAILNET not in wire.hosts()
