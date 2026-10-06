"""สิ่งที่ adopt ทำซ้ำไม่ได้ (หรือจงใจทำต่างจากเดิม) ต้องถูกบอกครบสามที่ — ไม่มีข้อไหนเงียบ

audit 2026-10-06: ทุกข้อที่หลุดจาก adopt มีรูปเดียวกันคือ "เขียน controller สำเร็จ ไม่มี error" แล้วของหาย
ตอน restart · ต่อให้ยกธงมาครบแค่ไหนก็ยังมีของที่ `docker run` คำสั่งเดียวแทนไม่ได้ (network ที่สอง ·
container ของ compose · HostConfig ที่ LMDS ยังไม่รู้จัก) — ของพวกนั้นต้องถูกเก็บเป็น **รายการ** แล้วไปโผล่ที่

  1. หน้าจอของ `lmds adopt` (คำเตือน)       — คนที่สั่งเห็นตอนนั้นเลย
  2. MODEL_PROFILE.yaml (`not_reproduced`)   — หน้าเว็บ/doctor/คนที่มาทีหลังอ่านได้
  3. หัว controller (บล็อกคอมเมนต์)          — คนที่กำลังจะกด restart เปิดไฟล์มาก็เจอ

และของเดิมที่ adopt ไว้แล้วสามตัวบนฟลีต: `bundles refresh`/`node install` **ข้าม** bundle ที่ adopt มา
(ไม่มี template) ทางเดียวที่สคริปต์จะถูกสร้างใหม่คือมีคนสั่ง `lmds adopt` ซ้ำ — ตรงนั้นต้องไม่เขียนทับเงียบ ๆ
"""

import shutil
import subprocess
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from tests.adopt_fakes import (
    adopt_from, adopt_mod, container_payload, fake_docker_inspect, image_payload, inspected,
    run_controller, started,
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")


@pytest.fixture(autouse=True)
def key_store(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))


# ของที่ `docker run` คำสั่งเดียวแทนไม่ได้ หรือ LMDS ยังไม่รู้วิธีใส่กลับ
HARD_TO_REPRODUCE = dict(
    config={"Labels": {"com.docker.compose.project": "llm", "com.docker.compose.service": "vllm",
                       "com.docker.compose.project.config_files": "/srv/llm/compose.yaml"},
            "Healthcheck": {"Test": ["CMD-SHELL", "curl -f http://localhost:8000/health"], "Interval": 30000000000}},
    host={"NetworkMode": "llm_default", "AutoRemove": False,
          "BlkioDeviceReadBps": [{"Path": "/dev/nvme0n1", "Rate": 104857600}],
          "DeviceRequests": [{"Driver": "", "Count": -1, "DeviceIDs": None, "Capabilities": [["gpu"]], "Options": {}},
                             {"Driver": "acme", "Count": 1, "DeviceIDs": None, "Capabilities": [["fpga"]], "Options": {}}]},
    NetworkSettings={"Ports": {}, "Networks": {
        "llm_default": {"Aliases": ["vllm", "9c1e7a5b3d2f"], "IPAMConfig": None},
        "monitoring": {"Aliases": ["vllm"], "IPAMConfig": None}}},
)


def _hard():
    return container_payload(**HARD_TO_REPRODUCE)


def test_each_thing_it_cannot_reproduce_is_named():
    said = "\n".join(adopt_mod.not_reproduced(inspected(_hard())))
    assert "monitoring" in said, "network ที่สอง — docker run ต่อได้ network เดียว"
    assert "docker network connect" in said, "ต้องบอกวิธีทำเองด้วย ไม่ใช่แค่บอกว่าขาด"
    assert "compose" in said and "llm" in said, "container ของ compose: compose ยังถือว่าเป็นของมัน"
    assert "BlkioDeviceReadBps" in said, "HostConfig ที่ LMDS ไม่รู้จัก ต้องถูกเอ่ยชื่อ"
    assert "fpga" in said, "device request ที่ไม่ใช่ GPU"
    assert "HEALTHCHECK" in said or "healthcheck" in said, "ของเดิมมี healthcheck ที่ผู้ใช้ตั้ง — เราปิดมัน"
    assert "unless-stopped" in said, "ของเดิมไม่มี restart policy — เราใส่ให้ ต้องบอก"


def test_what_can_be_carried_on_the_primary_network_is_carried(tmp_path):
    """alias ของ service ใน compose คือชื่อที่ container อื่น (gateway/open-webui) ใช้เรียก — หาย = ลูกค้าเรียกไม่ถึง"""
    argv = started(adopt_mod.render_controller(inspected(_hard()), "m"), tmp_path)
    assert "--network" in argv and argv[argv.index("--network") + 1] == "llm_default"
    assert [b for a, b in zip(argv, argv[1:], strict=False) if a == "--network-alias"] == ["vllm"]


def test_a_container_that_needs_nothing_special_reports_only_the_restart_default():
    lines = adopt_mod.not_reproduced(inspected(container_payload()))
    assert len(lines) == 1 and "unless-stopped" in lines[0], lines


def test_a_container_with_its_own_restart_policy_reports_nothing():
    adopted = inspected(container_payload(host={"RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0}}))
    assert adopt_mod.not_reproduced(adopted) == []


def test_the_list_is_at_the_top_of_the_controller_and_does_not_break_it(tmp_path):
    adopted = inspected(_hard())
    script = adopt_mod.render_controller(adopted, "m")
    head = script.split("set -Eeuo pipefail", 1)[0]
    for line in adopt_mod.not_reproduced(adopted):
        assert line in head, f"ไม่อยู่ในหัวไฟล์: {line}"
    assert all(ln.startswith("#") or not ln.strip() for ln in head.splitlines()), "หัวไฟล์ต้องเป็นคอมเมนต์ล้วน"
    assert subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True).returncode == 0
    started(script, tmp_path)


def test_a_value_with_a_newline_cannot_escape_the_comment_block(tmp_path):
    """label เป็นของที่ใครก็ตั้งได้ — ขึ้นบรรทัดใหม่ในค่าแล้วบรรทัดถัดไปกลายเป็นคำสั่ง ถ้าเขียนลงคอมเมนต์ดิบ ๆ"""
    payload = _hard()
    payload["Config"]["Labels"]["com.docker.compose.project"] = "llm\ntouch " + str(tmp_path / "pwned")
    script = adopt_mod.render_controller(inspected(payload), "m")
    done, _ = run_controller(script, tmp_path, "info")
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "pwned").exists()


def test_the_list_is_stored_in_the_profile(tmp_path):
    payload = _hard()
    controller = adopt_from(payload, tmp_path / "bundles")
    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["not_reproduced"] == adopt_mod.not_reproduced(inspected(payload))
    assert any("monitoring" in line for line in profile["not_reproduced"])


def test_lmds_adopt_prints_the_list_as_a_warning(tmp_path):
    from lmds.cli.main import app

    payload = _hard()
    with patch.object(adopt_mod.subprocess, "run", side_effect=fake_docker_inspect(payload, image_payload())):
        result = CliRunner().invoke(app, ["adopt", "vllm-gemma4", "--output", str(tmp_path / "bundles")],
                                    env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    for needle in ("monitoring", "BlkioDeviceReadBps", "compose"):
        assert needle in result.output, f"คำเตือนเรื่อง {needle} ไม่ขึ้นหน้าจอ:\n{result.output}"


# ── สั่ง adopt ซ้ำทับของเดิม ──────────────────────────────────────────────────────────────
def test_readopting_an_unchanged_container_leaves_no_backup_behind(tmp_path):
    payload = container_payload()
    first = adopt_from(payload, tmp_path / "bundles")
    adopt_from(payload, tmp_path / "bundles")
    assert [p.name for p in first.parent.iterdir() if ".replaced-" in p.name] == []


def test_readopting_over_a_different_script_keeps_the_old_one_and_says_so(tmp_path):
    """สคริปต์ที่ adopt รุ่นก่อนเขียน (หรือที่คนแก้มือไว้) ต่างจากตัวที่จะเขียนใหม่ — ต้องเหลือของเดิมให้ diff
    และบอกบนหน้าจอ · generator ของ adopt เปลี่ยนไปเรื่อย ๆ "ต้อง diff ก่อนเชื่อ" """
    from lmds.cli.main import app

    payload = container_payload()
    controller = adopt_from(payload, tmp_path / "bundles")
    controller.write_text(controller.read_text(encoding="utf-8") + "\n# แก้มือ: เพิ่ม --foo\n", encoding="utf-8")

    with patch.object(adopt_mod.subprocess, "run", side_effect=fake_docker_inspect(payload, image_payload())):
        result = CliRunner().invoke(app, ["adopt", "vllm-gemma4", "--output", str(tmp_path / "bundles")],
                                    env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    backups = [p for p in controller.parent.iterdir() if ".replaced-" in p.name]
    assert len(backups) == 1 and "# แก้มือ: เพิ่ม --foo" in backups[0].read_text(encoding="utf-8")
    assert "# แก้มือ" not in controller.read_text(encoding="utf-8")
    assert backups[0].name in result.output.replace("\n", ""), "ต้องบอกว่าของเดิมถูกเก็บไว้ที่ไหน"


def test_refresh_never_regenerates_an_adopted_controller(tmp_path, monkeypatch):
    """`lmds node install` เรียก `bundles refresh --all --if-older` ทุกครั้งที่อัปเดตเครื่อง —
    controller ของลูกค้าสามตัวต้องไม่ถูกแตะจากทางนั้น (ยืนยันพฤติกรรมเดิม · กันคนมาแก้ทีหลัง)"""
    from lmds.fleet.manager import ServerInfo
    from lmds.fleet.refresh import refresh_bundle

    controller = adopt_from(container_payload(), tmp_path / "bundles")
    before = controller.read_bytes()
    result = refresh_bundle(ServerInfo(slug="vllm-gemma4", controller=str(controller)), if_older=False)
    assert result.action == "adopted"
    assert controller.read_bytes() == before
    assert [p.name for p in controller.parent.iterdir() if ".replaced-" in p.name] == []


# ── ค่าที่มาจากของที่รันอยู่ ต้องไม่กลายเป็นคำสั่งในสคริปต์ ───────────────────────────────
@pytest.mark.parametrize("command", ["info", "banner", "status", "remove-plan"])
def test_a_model_path_with_shell_syntax_is_printed_not_executed(tmp_path, command):
    """`echo "model: <path>"` แทรก path ดิบ ๆ ในเครื่องหมายคำพูดคู่ — `$(…)` ใน path ถูกรัน และ `"` ทำไฟล์พัง
    ชื่อโมเดลมาจาก argv/env ของ container ที่ใครก็ตามบนเครื่องสั่งรันได้"""
    marker = tmp_path / "pwned"
    # ไม่มีช่องว่าง: argv ที่มีช่องว่างถูกแตกเป็นหลายคำตอนหาชื่อโมเดล · `$(>ไฟล์)` สร้างไฟล์ได้โดยไม่ต้องมีคำสั่ง
    evil = f'/models/a"b$(>{marker})`>{marker}`'
    adopted = inspected(container_payload(
        args=["-m", "vllm.entrypoints.openai.api_server", "--model", evil, "--port", "8000"]))
    script = adopt_mod.render_controller(adopted, "m")
    assert subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True).returncode == 0
    done, _ = run_controller(script, tmp_path, command)
    assert done.returncode == 0, done.stderr
    assert not marker.exists(), "path ของโมเดลถูกรันเป็นคำสั่ง"
    if command == "info":
        assert evil in done.stdout, "ต้องพิมพ์ path ตามตัวอักษร"


@pytest.mark.parametrize("command", ["info", "status", "network-info", "remove-plan"])
def test_native_model_names_with_shell_syntax_are_printed_not_executed(tmp_path, command):
    marker = tmp_path / "pwned"
    alias = f'q"x$(>{marker})'
    proc = adopt_mod.AdoptedProcess(
        pid=7, exe="/opt/llama/llama-server", cwd="/opt",
        argv=["/opt/llama/llama-server", "-m", f"/models/w$(>{marker}).gguf", "--alias", alias,
              "-c", f"$(>{marker})", "--host", f"$(>{marker})", "--port", "8080"])
    script = adopt_mod.render_native_controller(proc, "n")
    assert subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True).returncode == 0
    done, _ = run_controller(script, tmp_path, command,
                             extra_bins={"hostname": "#!/bin/bash\necho 10.0.0.9\n",
                                         "curl": "#!/bin/bash\nexit 7\n"})
    assert done.returncode == 0, done.stderr
    assert not marker.exists(), "ค่าจาก argv ของ process ถูกรันเป็นคำสั่ง"
    if command == "info":
        assert alias in done.stdout
