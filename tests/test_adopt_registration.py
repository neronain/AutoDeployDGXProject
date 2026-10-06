"""bundle ที่ adopt มาต้องหาเจอจากทุกที่ที่ LMDS มองหา controller — ไม่ใช่แค่จากโฟลเดอร์ที่พิมพ์คำสั่ง

audit 2026-10-06: `lmds adopt coder-next` (ค่าตั้งต้น `--output ./bundles`) จด controller ลง server.meta
เป็น path **สัมพัทธ์** `bundles/coder-next/coder-next-adopted.sh`

  1. จาก cwd อื่น (lmds-web ใต้ systemd · เชลล์อีกหน้าต่าง · `lmds node run` จาก hub) `controller_exists`
     เป็น False → ปุ่ม start/restart/test หายหรือขึ้น "ไม่พบ controller"
  2. พอ container ไม่ได้รันอยู่ `discover()` เห็นว่า "ไม่เคย start + controller ไม่อยู่" แล้ว **ลบทะเบียนทิ้ง**
  3. ตัวสแกน bundle บนดิสก์ (`_pick_controller`) glob แค่ `*-single.sh`/`*-stacked.sh` — bundle ที่
     adopt มาจึงกลับเข้าระบบเองไม่ได้อีกเลย · โมเดลของลูกค้าหายจาก `lmds list` ทั้งที่ไฟล์อยู่ครบ

glob ชุดเดียวกันอยู่ใน `nodes/ssh.py::ctl_script` (hub ตาม log ของโมเดลบนเครื่องอื่น) และ
`fleet/clone.py::inspect_source` (ซึ่งตอบว่า "ไม่พบ controller" ทั้งที่เหตุจริงคือ clone ของที่ adopt มาไม่มีความหมาย)
"""

import importlib
import shutil
import subprocess

import pytest

from tests.adopt_fakes import adopt_from, adopt_mod, container_payload

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")

manager = importlib.import_module("lmds.fleet.manager")


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """เครื่องหนึ่งเครื่อง: HOME ของตัวเอง · container `coder-next` ที่เปิด/ปิดได้ · ไม่มี orphan อื่น"""
    home = tmp_path / "home"
    (home / "elsewhere").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    monkeypatch.delenv("LMDS_BUNDLE_DIRS", raising=False)
    state = {"running": True}
    monkeypatch.setattr(manager, "_container_running", lambda name: state["running"] and name == "coder-next")
    monkeypatch.setattr(manager, "_orphan_docker", lambda known: [])
    monkeypatch.setattr(manager, "_orphan_native", lambda *a, **k: [])
    monkeypatch.setattr(manager, "_health_ok", lambda *a, **k: False)
    monkeypatch.chdir(home)
    return home, state


def _adopt_like_the_cli(home):
    """`cd ~ && lmds adopt coder-next` — output เป็น ./bundles แบบสัมพัทธ์ตามค่าตั้งต้นของ CLI"""
    from pathlib import Path

    return adopt_from(container_payload(name="coder-next"), Path("./bundles"))


def test_the_registered_controller_path_does_not_depend_on_the_cwd(machine, monkeypatch):
    home, _ = machine
    controller = _adopt_like_the_cli(home)
    assert controller.is_absolute(), f"path สัมพัทธ์ใช้ได้เฉพาะจากโฟลเดอร์ที่พิมพ์คำสั่ง: {controller}"

    monkeypatch.chdir(home / "elsewhere")          # lmds-web · systemd · เชลล์อีกหน้าต่าง
    found = manager.find("coder-next")
    assert found is not None and found.controller_exists
    assert found.controller == str(home / "bundles" / "coder-next" / "coder-next-adopted.sh")


def test_a_stopped_adopted_model_does_not_vanish_from_the_fleet(machine, monkeypatch):
    home, state = machine
    _adopt_like_the_cli(home)
    monkeypatch.chdir(home / "elsewhere")
    state["running"] = False                        # `lmds stop` → controller ทำ docker rm -f
    found = manager.find("coder-next")
    assert found is not None, "โมเดลของลูกค้าหายจาก lmds list ทั้งที่ bundle อยู่ครบ"
    assert found.controller_exists and not found.running


def test_a_registration_written_with_a_relative_path_still_resolves(machine, monkeypatch):
    """ทะเบียนที่ adopt รุ่นก่อนเขียนไว้ (สามตัวบนฟลีต TKC) ยังเป็น path สัมพัทธ์อยู่บนดิสก์ —
    ต้องหา controller เจอโดยไม่ต้องให้ใครไป adopt ซ้ำ และต้องไม่ถูกลบทิ้งตอน container หยุด"""
    home, state = machine
    _adopt_like_the_cli(home)
    meta = manager.run_root() / "coder-next" / "server.meta"
    text = meta.read_text(encoding="utf-8")
    absolute = str(home / "bundles" / "coder-next" / "coder-next-adopted.sh")
    assert absolute in text
    meta.write_text(text.replace(absolute, "bundles/coder-next/coder-next-adopted.sh"), encoding="utf-8")

    monkeypatch.chdir(home / "elsewhere")
    found = manager.find("coder-next")
    assert found is not None and found.controller_exists and found.controller == absolute

    state["running"] = False
    found = manager.find("coder-next")
    assert found is not None and found.controller_exists
    assert meta.exists(), "ทะเบียนของ bundle ที่ยังอยู่บนดิสก์ต้องไม่ถูกเก็บกวาด"


def test_an_adopted_bundle_on_disk_is_rediscovered_as_what_it_is(machine, monkeypatch):
    """ทะเบียนหายไปแล้ว (เคยถูกลบโดยบั๊กข้างบน · หรือ ~/.lmds/run ถูกล้าง) — bundle ยังอยู่ใน ~/bundles
    ต้องกลับมาเอง **ด้วยชื่อ container เดิม** ไม่ใช่ `lmds-<slug>` ที่เป็นธรรมเนียมของ bundle ที่ LMDS สร้าง"""
    home, state = machine
    _adopt_like_the_cli(home)
    shutil.rmtree(manager.run_root() / "coder-next")
    state["running"] = False
    monkeypatch.chdir(home / "elsewhere")

    found = manager.find("coder-next")
    assert found is not None, "bundle ที่ adopt มาไม่ถูกสแกนเจอ"
    assert found.controller.endswith("coder-next-adopted.sh") and found.controller_exists
    assert found.container == "coder-next", f"ชื่อ container ผิดตัว: {found.container}"
    assert found.port == 8000 and found.mode == "docker"


def test_a_generated_bundle_still_wins_over_an_adopted_script_in_the_same_folder(tmp_path):
    """ลำดับเดิมต้องไม่เปลี่ยน: โฟลเดอร์ที่มี -single.sh/-stacked.sh ใช้ตัวนั้นเหมือนเดิม"""
    folder = tmp_path / "bundles" / "m"
    folder.mkdir(parents=True)
    for name in ("m-single.sh", "m-adopted.sh"):
        (folder / name).write_text("#!/bin/bash\n", encoding="utf-8")
    (folder / "MODEL_PROFILE.yaml").write_text("topology: single\n", encoding="utf-8")
    assert manager._pick_controller(folder, folder / "MODEL_PROFILE.yaml").name == "m-single.sh"


def test_a_native_adopted_bundle_is_rediscovered_as_a_process_not_a_container(machine, monkeypatch):
    home, _ = machine
    proc = adopt_mod.AdoptedProcess(pid=4242, argv=["/opt/llama/llama-server", "-m", "/models/x.gguf",
                                                     "--port", "8080"], exe="/opt/llama/llama-server", cwd="/opt")
    monkeypatch.setattr(adopt_mod, "inspect_process", lambda **kw: proc)
    monkeypatch.setattr(adopt_mod, "probe_server", lambda *a, **k: {})
    from pathlib import Path

    controller, _ = adopt_mod.adopt_process(pid=4242, slug="x", output=Path("./bundles"))
    assert controller.is_absolute()
    shutil.rmtree(manager.run_root() / "x")
    monkeypatch.chdir(home / "elsewhere")

    found = manager.find("x")
    assert found is not None and found.controller_exists
    assert found.mode == "native" and found.container == "" and found.port == 8080


# ── hub สั่ง controller บนเครื่องอื่น ──────────────────────────────────────────────────────
def _bundle_with(home, slug, script_name, body="#!/bin/bash\necho \"ran: $*\"\n"):
    folder = home / "bundles" / slug
    folder.mkdir(parents=True)
    script = folder / script_name
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    (folder / "MODEL_PROFILE.yaml").write_text("adopted: true\n", encoding="utf-8")
    return folder


def test_the_hub_can_reach_the_controller_of_an_adopted_bundle(tmp_path):
    """`ctl_script` คือสิ่งที่ hub ส่งไปรันบน node (ตาม log · สั่งคำสั่งของ controller) — รันมันจริงใต้ bash"""
    from lmds.nodes.ssh import ctl_script

    home = tmp_path / "home"
    _bundle_with(home, "coder-next", "coder-next-adopted.sh")
    done = subprocess.run(["bash", "-c", ctl_script("coder-next", "msi-1", "logs 50")],
                          capture_output=True, text=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran: logs 50"


def test_the_hub_still_prefers_a_generated_controller(tmp_path):
    from lmds.nodes.ssh import ctl_script

    home = tmp_path / "home"
    folder = _bundle_with(home, "m", "m-single.sh", "#!/bin/bash\necho single\n")
    (folder / "m-adopted.sh").write_text("#!/bin/bash\necho adopted\n", encoding="utf-8")
    (folder / "m-adopted.sh").chmod(0o755)
    done = subprocess.run(["bash", "-c", ctl_script("m", "msi-1", "status")],
                          capture_output=True, text=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
    assert done.stdout.strip() == "single"


def test_an_adopted_controller_refuses_commands_it_does_not_have(tmp_path):
    """พอ hub หา controller ของ bundle ที่ adopt มาเจอ คำสั่งที่มันไม่มี (prepare-runtime · download ·
    verify-files) ต้อง **ล้ม** ไม่ใช่พิมพ์วิธีใช้แล้วคืน 0 — hub อ่าน exit code แล้วรายงานว่า "สำเร็จ" """
    from tests.adopt_fakes import inspected, run_controller

    script = adopt_mod.render_controller(inspected(container_payload()), "m")
    for command in ("prepare-runtime", "download", "verify-files"):
        done, argv = run_controller(script, tmp_path, command)
        assert done.returncode != 0, f"{command} คืน 0 ทั้งที่ไม่ได้ทำอะไร"
        assert argv is None
    done, _ = run_controller(script, tmp_path, "help")
    assert done.returncode == 0 and "คำสั่ง:" in done.stdout

    native = adopt_mod.render_native_controller(
        adopt_mod.AdoptedProcess(pid=1, argv=["/x/llama-server", "-m", "/m.gguf"], exe="/x/llama-server"), "n")
    done, _ = run_controller(native, tmp_path, "prepare-runtime")
    assert done.returncode != 0


# ── clone ─────────────────────────────────────────────────────────────────────────────────
class _Node:
    def __init__(self, name):
        self.name, self.host, self.site, self.cluster_ip = name, "10.0.0.1", "TKC", ""


class _Result:
    def __init__(self, done):
        self.ok, self.stdout, self.stderr = done.returncode == 0, done.stdout, done.stderr


def test_cloning_an_adopted_bundle_says_why_it_cannot_instead_of_controller_not_found(tmp_path, monkeypatch):
    """bundle ที่ adopt มาชี้ weight/mount/image ของเครื่องต้นทางเอง — ไม่มี MODEL_DIR/HF_HOME ให้ clone อ่าน
    และคำสั่ง docker run ของมันใช้ที่เครื่องอื่นไม่ได้ · "ไม่พบ controller" ทำให้คนไปไล่หาไฟล์ที่ไม่ได้หาย"""
    from lmds.fleet.clone import CloneError, inspect_source, plan_clone

    home = tmp_path / "home"
    _bundle_with(home, "coder-next", "coder-next-adopted.sh")
    monkeypatch.setattr("lmds.nodes.find", lambda name: _Node(name), raising=False)
    monkeypatch.setattr("lmds.nodes.run", lambda node, script, timeout=120: _Result(subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"})), raising=False)

    with pytest.raises(CloneError) as err:
        inspect_source(plan_clone("coder-next", "msi-1", "msi-2"))
    message = str(err.value)
    assert "adopt" in message
    assert "ไม่พบ controller" not in message
