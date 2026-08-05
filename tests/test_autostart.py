"""เทส autostart (systemd) — ให้โมเดลกลับมาเองหลัง reboot

ทดสอบการ render unit + status parsing + enable/disable โดย mock subprocess/systemctl
(ไม่แตะ systemd จริงของเครื่องที่รันเทส)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lmds.fleet import manager
from lmds.fleet.manager import (
    FleetError,
    ServerInfo,
    autostart_status,
    disable_autostart,
    enable_autostart,
    render_unit,
    unit_name,
)


def _info(tmp_path, mode="docker") -> ServerInfo:
    ctl = tmp_path / "m-single.sh"
    ctl.write_text("#!/bin/bash\n")
    ctl.chmod(0o755)
    return ServerInfo(slug="m", model="org/M", engine="vllm", mode=mode,
                      port=8000, container="lmds-m", controller=str(ctl))


def test_unit_name():
    assert unit_name("gemma-4") == "lmds-gemma-4.service"


def test_render_unit_has_required_directives(tmp_path):
    unit = render_unit(_info(tmp_path), timeout=1234)
    assert "Type=oneshot" in unit
    assert "RemainAfterExit=yes" in unit
    assert "WantedBy=multi-user.target" in unit
    assert "TimeoutStartSec=1234" in unit
    # ExecStartPre stop กัน container ค้างหลัง reboot
    assert "ExecStartPre=-" in unit and " stop" in unit
    assert "ExecStart=" in unit and " start" in unit
    assert "ExecStop=" in unit and " stop" in unit
    assert "User=" in unit and "Environment=HOME=" in unit


def test_autostart_status_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    assert autostart_status("nope") == "absent"


def test_autostart_status_na_without_systemd(monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    assert autostart_status("anything") == "n/a"


def test_autostart_status_enabled(tmp_path, monkeypatch):
    sysd = tmp_path / "systemd"
    sysd.mkdir()
    (sysd / unit_name("m")).write_text("[Unit]\n")
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(sysd))

    class R:
        stdout = "enabled\n"
        returncode = 0

    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: R())
    assert autostart_status("m") == "enabled"


def test_enable_autostart_runs_sudo_steps(tmp_path, monkeypatch):
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    calls = []

    class OK:
        returncode = 0

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, *a, **k: (calls.append(cmd), OK())[1])
    name = enable_autostart(info, timeout=600, start_now=True)
    assert name == "lmds-m.service"
    # unit ถูก stage ลง bundle dir ก่อน
    assert (tmp_path / "lmds-m.service").exists()
    # ต้องเรียก install + daemon-reload + enable + start (เพราะ start_now=True)
    joined = [" ".join(c) for c in calls]
    assert any("install" in c for c in joined)
    assert any("daemon-reload" in c for c in joined)
    assert any("enable" in c for c in joined)
    assert any("systemctl start" in c for c in joined)


def test_enable_autostart_fails_without_systemd(tmp_path, monkeypatch):
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    with pytest.raises(FleetError, match="systemd"):
        enable_autostart(info)


def test_enable_autostart_reports_failed_step(tmp_path, monkeypatch):
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))

    class Fail:
        returncode = 1

    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: Fail())
    with pytest.raises(FleetError, match="ไม่สำเร็จ"):
        enable_autostart(info)


def test_disable_autostart_runs_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    calls = []
    states = iter(["enabled", "absent"])   # ก่อนสั่ง → หลังสั่ง
    monkeypatch.setattr(manager, "autostart_status", lambda slug: next(states))

    class OK:
        returncode = 0

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, *a, **k: (calls.append(cmd), OK())[1])
    name = disable_autostart("m")
    assert name == "lmds-m.service"
    joined = [" ".join(c) for c in calls]
    assert any("disable" in c for c in joined)
    assert any("rm -f" in c for c in joined)


def test_disable_autostart_reports_failure_instead_of_claiming_success(tmp_path, monkeypatch):
    """sudo ที่ขอรหัสผ่านไม่ได้ (เช่นถูกเรียกผ่าน SSH จาก hub) เคยถูกกลืนทั้งหมด แล้วรายงานว่า
    "ปิด autostart แล้ว" — ผู้ใช้จะรู้ตัวอีกทีตอน reboot แล้วโมเดลเด้งขึ้นมาเอง
    """
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    monkeypatch.setattr(manager, "autostart_status", lambda slug: "enabled")  # ไม่เปลี่ยนเลย

    class Fail:
        returncode = 1

    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: Fail())
    with pytest.raises(FleetError, match="ไม่สำเร็จ"):
        disable_autostart("m")


def test_disable_autostart_says_so_when_there_was_nothing_to_disable(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    monkeypatch.setattr(manager, "autostart_status", lambda slug: "absent")
    with pytest.raises(FleetError, match="ไม่ได้ตั้ง autostart"):
        disable_autostart("m")
