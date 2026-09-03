"""งานยาว ๆ บน node ต้องไม่ผูกกับ SSH session ของ hub

เคสจริง spark-worker 2026-09-03: `lmds node push --download` รันผ่าน run() ตรง ๆ · session หลุด →
controller บน node ตาย แต่ container ที่มันสั่งไว้ยังอยู่โดยไม่มีใครเฝ้า (watchdog กัน Xet ค้าง
อยู่ใน controller จึงตายไปด้วย) → container ค้างที่ "Fetching 33 files 0%" rx 0 MB/s อยู่ 90 นาที
ส่วน hub ค้างที่ "โหลด weight บน spark-worker…" ตลอด
"""

from types import SimpleNamespace

import pytest

from lmds.cli import main as cli


class _Result(SimpleNamespace):
    @property
    def ok(self):
        return self.exit_code == 0


def _fake_run(script):
    """script: ลำดับผลลัพธ์ที่ run() จะคืน — ตัวแรกคือการ launch ที่เหลือคือการอ่าน log"""
    calls = []

    def run(node, command, timeout=60, stdin_text=""):
        calls.append(command)
        out = script.pop(0)
        return _Result(exit_code=0, stdout=out, stderr="")

    return run, calls


def test_launch_is_detached_and_rc_is_read_from_the_log(monkeypatch, capsys):
    run, calls = _fake_run(["started\n", "โหลดอยู่ 10%\n", "โหลดอยู่ 10%\nโหลดอยู่ 60%\n", "โหลดอยู่ 10%\nโหลดอยู่ 60%\nเสร็จ\n__RC=0\n"])
    monkeypatch.setattr("lmds.nodes.run", run)
    rc = cli._run_detached(SimpleNamespace(name="n1"), "lmds repair demo", "demo.repair", timeout=60, poll=0)
    assert rc == 0
    # คำสั่งที่สั่งไปต้องหลุดจาก session: setsid + nohup + เขียน __RC ตอนจบ
    assert "setsid nohup" in calls[0] and "__RC=$?" in calls[0] and "< /dev/null" in calls[0]
    # พิมพ์เฉพาะส่วนใหม่ ไม่พิมพ์ซ้ำ
    out = capsys.readouterr().out
    assert out.count("โหลดอยู่ 10%") == 1 and "เสร็จ" in out and "__RC" not in out


def test_non_zero_rc_propagates(monkeypatch):
    run, _ = _fake_run(["started\n", "พัง\n__RC=3\n"])
    monkeypatch.setattr("lmds.nodes.run", run)
    assert cli._run_detached(SimpleNamespace(name="n1"), "lmds repair demo", "demo.repair", timeout=60, poll=0) == 3


def test_timeout_leaves_the_job_running_and_says_so(monkeypatch, capsys):
    run, _ = _fake_run(["started\n"] + ["ยังโหลด\n"] * 50)
    monkeypatch.setattr("lmds.nodes.run", run)
    rc = cli._run_detached(SimpleNamespace(name="n1"), "lmds repair demo", "demo.repair", timeout=0, poll=0)
    assert rc == 124
    assert "ยังรันอยู่บน node" in capsys.readouterr().err


def test_node_run_quotes_each_argument_for_the_remote_shell(monkeypatch):
    """เคสจริง RTX4000 (2026-09-03): `lmds node run X set s --extra-args "--a 1 --b 2"`
    ถึงปลายทางเป็น 4 argument แยกกัน typer จึงตอบ No such option: --b · ต้อง quote เหมือน node ctl"""
    from typer.testing import CliRunner

    from lmds.cli.main import app

    captured = {}

    class _Res:
        stdout, stderr, ok, exit_code = "", "", True, 0

    def fake_run(node, command, timeout=900):
        captured["command"] = command
        return _Res()

    class _Node:
        name = "x"

    monkeypatch.setattr("lmds.nodes.run", fake_run, raising=False)
    monkeypatch.setattr("lmds.nodes.find", lambda n: _Node(), raising=False)
    result = CliRunner().invoke(app, ["node", "run", "x", "set", "s", "--extra-args", "--kv-cache-dtype fp8 --max-num-batched-tokens 8192"])
    assert result.exit_code == 0, result.output
    assert captured["command"] == "lmds set s --extra-args '--kv-cache-dtype fp8 --max-num-batched-tokens 8192'"
