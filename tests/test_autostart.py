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


def test_user_scope_unit_has_no_user_directive(tmp_path):
    """`User=` ใน user unit ทำให้ systemd ตายก่อนเรียก controller ด้วยซ้ำ

    user manager รันเป็น user นั้นอยู่แล้ว จึงไม่มีสิทธิ์สลับ user ให้ตัวเอง:
    `Failed to determine supplementary groups` แล้ว `status=216/GROUP`
    """
    unit = render_unit(_info(tmp_path), scope="user")
    assert "User=" not in unit
    assert "WantedBy=default.target" in unit
    # ส่วนที่เหลือต้องยังครบ — ไม่ใช่ตัด User= แล้วทำ unit พังทางอื่นแทน
    assert "ExecStart=" in unit and " start" in unit
    assert "Type=oneshot" in unit


def test_adopted_docker_user_unit_has_no_user_directive(tmp_path):
    """container ที่ถูก adopt ก็ใช้ทางเดียวกัน — พังเงียบแบบเดียวกันถ้าลืม"""
    info = ServerInfo(slug="m", model="org/M", engine="vllm", mode="docker",
                      port=8000, container="lmds-m", controller=str(tmp_path / "gone.sh"))
    unit = render_unit(info, scope="user")
    assert "User=" not in unit
    assert "WantedBy=default.target" in unit
    assert "docker start lmds-m" in unit


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
        stdout = "active"    # is-active ตอบ active → smoke-check ผ่าน

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, *a, **k: (calls.append(cmd), OK())[1])
    name = enable_autostart(info, timeout=600, start_now=True, scope="system")
    assert name == "lmds-m.service"
    # unit ถูก stage ลง bundle dir ก่อน
    assert (tmp_path / "lmds-m.service").exists()
    # ต้องเรียก install + daemon-reload + enable + start (เพราะ start_now=True)
    joined = [" ".join(c) for c in calls]
    assert any("install" in c for c in joined)
    assert any("daemon-reload" in c for c in joined)
    assert any("enable" in c for c in joined)
    assert any("systemctl start" in c for c in joined)
    # --now ต้องพิสูจน์ต่อว่า unit ขึ้นจริง ไม่ใช่แค่ start คืน 0
    assert any("is-active" in c for c in joined)


def test_enable_now_fails_loudly_when_the_unit_will_not_start(tmp_path, monkeypatch):
    """เคสจริงที่เจ็บมาแล้ว: start คืน 0 แต่ unit ล้มทันที (เช่น User= ใน user unit)

    ก่อนหน้านี้ enable รายงานสำเร็จ แล้วเครื่องบูตมาไม่มีโมเดล โดยไม่มีใครรู้จนถึงตอนนั้น
    ตอนนี้ is-active = failed ต้องทำให้ enable ล้มพร้อมบอก log
    """
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))

    def fake_run(cmd, *a, **k):
        text = " ".join(cmd)
        out = "failed" if "is-active" in text else ("bad: status=216/GROUP" if "journalctl" in text else "")
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    with pytest.raises(FleetError, match="start ไม่ขึ้น"):
        enable_autostart(_info(tmp_path), start_now=True)


def test_enable_now_accepts_a_model_still_loading(tmp_path, monkeypatch):
    """โมเดลใหญ่ยัง activating (โหลด weight อยู่) ไม่ใช่ความล้มเหลว — ต้องไม่โยน error"""
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))

    def fake_run(cmd, *a, **k):
        out = "activating" if "is-active" in " ".join(cmd) else ""
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    # ไม่ควรโยน
    enable_autostart(info, start_now=True)


def test_enable_autostart_fails_without_systemd(tmp_path, monkeypatch):
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    with pytest.raises(FleetError, match="systemd"):
        enable_autostart(info, scope="system")


def test_enable_autostart_reports_failed_step(tmp_path, monkeypatch):
    info = _info(tmp_path)
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))

    class Fail:
        returncode = 1

    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: Fail())
    with pytest.raises(FleetError, match="ไม่สำเร็จ"):
        enable_autostart(info, scope="system")


def test_disable_autostart_runs_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))   # ไม่มี user unit
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


def test_autostart_needs_no_sudo_by_default(tmp_path, monkeypatch):
    """hub สั่งข้ามเครื่องผ่าน SSH ซึ่งไม่มี tty ให้กรอกรหัส sudo — ปุ่ม enable บนหน้าเว็บ
    จึงล้มเสมอบนเครื่องที่ sudo ต้องใช้รหัสผ่าน ซึ่งคือค่าปกติของ Ubuntu (ผู้ใช้เจอจริง)
    """
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))
    calls = []

    class OK:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), OK())[1])
    info = _info(tmp_path)
    name = enable_autostart(info)

    assert (tmp_path / "user-units" / name).exists()
    assert not any("sudo" in c for c in calls), "ค่าเริ่มต้นต้องไม่แตะ sudo เลย"
    assert ["systemctl", "--user", "enable", name] in calls


def test_user_unit_starts_at_boot_not_only_at_login(tmp_path, monkeypatch):
    """WantedBy=multi-user.target ใช้กับ user scope ไม่ได้ — ต้องเป็น default.target"""
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))

    class OK:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kw: OK())
    info = _info(tmp_path)
    unit = (tmp_path / "user-units" / enable_autostart(info)).read_text(encoding="utf-8")
    assert "WantedBy=default.target" in unit
    assert "multi-user.target" not in unit
    # unit ที่เขียนลงดิสก์จริงต้องไม่มี User= ด้วย ไม่ใช่แค่ที่ render_unit คืนมา
    assert "User=" not in unit


def test_status_sees_a_user_unit(tmp_path, monkeypatch):
    """เปิดแบบ user แล้วรายงานว่า absent = บอกผิด และหน้าเว็บจะเสนอปุ่ม enable ซ้ำ"""
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    units = tmp_path / "user-units"
    units.mkdir()
    (units / "lmds-m.service").write_text("x", encoding="utf-8")
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(units))

    class OK:
        returncode = 0
        stdout = "enabled\n"

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kw: OK())
    assert manager.autostart_status("m") == "enabled"


def test_disable_removes_a_user_unit_too(tmp_path, monkeypatch):
    """เปิดแบบ user แล้วสั่ง disable ต้องปิดได้จริง — ไม่งั้นมันยังขึ้นเองอยู่หลัง reboot
    ทั้งที่ผู้ใช้สั่งปิดไปแล้ว
    """
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    units = tmp_path / "user-units"
    units.mkdir()
    (units / "lmds-m.service").write_text("x", encoding="utf-8")
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(units))
    monkeypatch.setenv("LMDS_SYSTEMD_DIR", str(tmp_path / "systemd"))
    calls = []

    class OK:
        returncode = 0
        stdout = "enabled\n"

    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), OK())[1])
    manager.disable_autostart("m")
    assert not (units / "lmds-m.service").exists()
    assert ["systemctl", "--user", "disable", "--now", "lmds-m.service"] in calls
    assert not any("sudo" in c for c in calls), "user unit ปิดได้โดยไม่ต้อง sudo"


# --- port collision at enable time (autostart) --------------------------------
# หลายโมเดล default port 8000 เท่ากัน · enable หลายตัวแล้ว reboot = ชน port ตัวหลังล้ม

def _ctl_with_port(tmp_path, slug, port):
    d = tmp_path / slug
    d.mkdir()
    ctl = d / f"{slug}-single.sh"
    ctl.write_text(f'#!/bin/bash\nAPI_PORT="${{API_PORT:-{port}}}"\n')
    ctl.chmod(0o755)
    return manager.ServerInfo(slug=slug, model="m", engine="llamacpp", mode="native",
                              port=port, container="", controller=str(ctl))


def test_effective_port_prefers_saved_over_controller_default(tmp_path):
    info = _ctl_with_port(tmp_path, "m", 8000)
    assert manager.effective_autostart_port(info) == "8000"   # จาก default ใน controller
    (tmp_path / "m" / "bundle.env").write_text('API_PORT="${API_PORT:-8010}"\n')
    assert manager.effective_autostart_port(info) == "8010"   # ค่าที่ set ชนะ


def test_enable_refuses_a_port_already_claimed_by_another_autostart(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))
    mine = _ctl_with_port(tmp_path, "mine", 8000)
    other = _ctl_with_port(tmp_path, "other", 8000)
    monkeypatch.setattr(manager, "discover", lambda: [mine, other])
    monkeypatch.setattr(manager, "autostart_status", lambda slug: "enabled")

    with pytest.raises(FleetError, match="ชนกับ 'other'"):
        enable_autostart(mine)


def test_enable_allows_a_free_port(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))
    mine = _ctl_with_port(tmp_path, "mine", 8010)
    other = _ctl_with_port(tmp_path, "other", 8000)
    monkeypatch.setattr(manager, "discover", lambda: [mine, other])
    monkeypatch.setattr(manager, "autostart_status", lambda slug: "enabled")

    class OK:
        returncode = 0
        stdout = "active"
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: OK())
    # ไม่ชน → ผ่าน
    assert enable_autostart(mine) == "lmds-mine.service"


def test_a_stopped_neighbour_does_not_block_enable(tmp_path, monkeypatch):
    """เพื่อนบ้านที่ port เดียวกันแต่ไม่ได้ตั้ง autostart ไม่ชน — มันไม่ขึ้นตอนบูต"""
    monkeypatch.setattr(manager, "have_systemctl", lambda: True)
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "user-units"))
    mine = _ctl_with_port(tmp_path, "mine", 8000)
    other = _ctl_with_port(tmp_path, "other", 8000)
    monkeypatch.setattr(manager, "discover", lambda: [mine, other])
    monkeypatch.setattr(manager, "autostart_status",
                       lambda slug: "enabled" if slug == "mine" else "absent")

    class OK:
        returncode = 0
        stdout = "active"
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: OK())
    assert enable_autostart(mine) == "lmds-mine.service"
