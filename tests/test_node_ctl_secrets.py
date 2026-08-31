"""`lmds node ctl` ต้องให้ node ยืม HF token ได้เหมือนหน้าเว็บ

เคสจริง 2026-08-31: `lmds node ctl spark-head <slug> download` ตกทันทีด้วย
"เป็น gated repo — ต้องมี HF_TOKEN ก่อน download" ทั้งที่ hub ถือ token ที่ใช้ได้อยู่
· กลไกให้ยืมมีมาตั้งแต่รอบก่อนแล้ว แต่ต่อไว้เฉพาะเส้นทางหน้าเว็บ (lmds.web.jobs)
เส้นทาง CLI จึง deploy โมเดล gated ข้ามเครื่องไม่ได้เลย

และของเดิมใช้ run() ที่รอจนจบแล้วค่อยพ่นผลออกมาทีเดียว — download 90 GB จึงเงียบ
สนิทเป็นสิบนาที แยกไม่ออกว่าทำงานอยู่หรือค้างไปแล้ว
"""

import subprocess

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app


class _FakeProc:
    def __init__(self, lines, code=0):
        self.stdout = _FakeStdout(lines)
        self._code = code

    def wait(self):
        return self._code


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else b""


@pytest.fixture
def node(monkeypatch):
    class Node:
        name, user, host, port = "spark-head", "neronain", "10.0.0.1", 22
        all_hosts = ["10.0.0.1"]

    monkeypatch.setattr("lmds.nodes.find", lambda name: Node(), raising=False)
    return Node


def test_ctl_lends_the_hub_token_to_the_node(node, monkeypatch):
    seen = {}

    def fake_stream(n, command, secret_env=None):
        seen["secret_env"] = secret_env
        seen["command"] = command
        return _FakeProc([b"download complete\n"])

    monkeypatch.setattr("lmds.nodes.stream", fake_stream, raising=False)
    monkeypatch.setattr("lmds.cli.main.get_secret",
                        lambda name: "hf_" + "x" * 34 if name == "hf" else None)

    result = CliRunner().invoke(app, ["node", "ctl", "spark-head", "demo", "download"])

    assert result.exit_code == 0, result.output
    assert seen["secret_env"] == {"HF_TOKEN": "hf_" + "x" * 34}, "ไม่ได้ให้ยืม token"
    assert "download complete" in result.output


def test_no_token_on_the_hub_is_not_an_error(node, monkeypatch):
    """เครื่องที่ไม่เคยตั้ง token ต้องยังสั่ง repo สาธารณะได้ตามปกติ"""
    seen = {}

    def fake_stream(n, command, secret_env=None):
        seen["secret_env"] = secret_env
        return _FakeProc([b"ok\n"])

    monkeypatch.setattr("lmds.nodes.stream", fake_stream, raising=False)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda name: None)

    result = CliRunner().invoke(app, ["node", "ctl", "spark-head", "demo", "status"])
    assert result.exit_code == 0, result.output
    assert seen["secret_env"] is None


def test_token_never_reaches_the_terminal(node, monkeypatch):
    """ปลายทาง "ไม่ควร" พิมพ์ token — แต่ไม่ควรกับไม่เคยคนละเรื่อง"""
    token = "hf_" + "z" * 34

    monkeypatch.setattr("lmds.nodes.stream",
                        lambda n, c, s=None: _FakeProc([f"using {token} now\n".encode()]),
                        raising=False)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda name: token)

    result = CliRunner().invoke(app, ["node", "ctl", "spark-head", "demo", "download"])
    assert token not in result.output
    assert "***" in result.output


def test_exit_code_of_the_remote_command_is_propagated(node, monkeypatch):
    monkeypatch.setattr("lmds.nodes.stream",
                        lambda n, c, s=None: _FakeProc([b"boom\n"], code=3), raising=False)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda name: None)

    result = CliRunner().invoke(app, ["node", "ctl", "spark-head", "demo", "start"])
    assert result.exit_code == 3
