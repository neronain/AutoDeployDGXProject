"""คำสั่งเดียวที่แขวนต้องไม่ล้มทั้งคอนโซล

เคสจริง 2026-08-14: กด Remove บนหน้าเว็บ · `docker stop` ไม่ยอมจบ · คำขอไม่เคยตอบ
· systemd ฆ่า lmds-web ทั้งตัวด้วย `Failed with result 'timeout'` · ผู้ใช้เห็นปุ่มค้าง
ที่ "Removing…" แล้วต้องรีเฟรชเอง โดยไม่รู้ว่าลบสำเร็จหรือไม่
"""

from __future__ import annotations

import subprocess

from lmds.fleet import manager


def test_a_command_that_never_finishes_gives_up(monkeypatch):
    def hang(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

    monkeypatch.setattr(manager.subprocess, "run", hang)
    done = manager._bounded(["docker", "stop", "x"], capture_output=True)
    assert done.returncode != 0, "ต้องรายงานว่าไม่สำเร็จ"


def test_it_reports_failure_instead_of_raising(monkeypatch):
    """โยน exception ขึ้นไป = ทั้งคำขอพัง · ผู้เรียกควรได้ผลลัพธ์ไปตัดสินใจต่อ"""
    def hang(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(manager.subprocess, "run", hang)
    manager._bounded(["docker", "stop", "x"])      # ต้องไม่โยนอะไรออกมา


def test_a_timeout_is_always_set(monkeypatch):
    seen = {}

    def spy(args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(manager.subprocess, "run", spy)
    manager._bounded(["docker", "stop", "x"])
    assert seen.get("timeout"), "ไม่มีเพดานเวลา = แขวนได้ตลอดกาล"


def test_a_caller_that_chose_its_own_limit_keeps_it(monkeypatch):
    seen = {}

    def spy(args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(manager.subprocess, "run", spy)
    manager._bounded(["docker", "stop", "x"], timeout=5)
    assert seen["timeout"] == 5


def test_stopping_a_wedged_container_still_returns(monkeypatch):
    """stop_server ต้องคืนค่าเสมอ — มันคือขั้นแรกของการลบ ถ้าค้างตรงนี้ก็ไม่มีอะไรเดินต่อ"""
    def hang(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(manager.subprocess, "run", hang)
    info = type("S", (), {"pid": None, "controller_exists": False, "mode": "docker",
                          "container": "stuck", "slug": "x", "controller": "",
                          "external": False, "running": True})()
    assert manager.stop_server(info)
