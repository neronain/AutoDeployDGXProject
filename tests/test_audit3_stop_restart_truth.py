"""audit 2026-10-06 — `lmds stop` / `restart` / `remove` รายงานว่าสำเร็จทั้งที่ controller ล้ม

เคสที่ auditor รันได้จริง: controller พิมพ์ error ของ docker
(`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`)
แล้วจบด้วย exit ≠ 0 · LMDS พิมพ์ต่อท้ายว่า `restart qwen แล้ว (controller)` และจบด้วย exit 0 ·
`lmds stop demo` → "หยุด demo แล้ว" ทั้งที่ process ยังอยู่ · `lmds remove` เดินต่อไปลบ bundle กับ
weight ของโมเดลที่ยังรันอยู่ — รูปเดียวกับทุกบั๊กในตาราง "หน้าจอบอกว่าผ่าน" ของ maintainer skill

ทุกเทสในไฟล์นี้รันของจริง: controller เป็นสคริปต์ bash จริง · "โมเดล" เป็น process `sleep` จริงที่
pid อยู่ในทะเบียน · สิ่งที่ยืนยันคือสภาพของ process/ไฟล์หลังคำสั่งจบ ไม่ใช่ข้อความที่พิมพ์
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.cli.main import app  # noqa: E402
from lmds.fleet import manager  # noqa: E402
from lmds.fleet.manager import FleetError  # noqa: E402

runner = CliRunner()

DOCKER_DENIED = ("permission denied while trying to connect to the docker API at "
                 "unix:///var/run/docker.sock")


@pytest.fixture(autouse=True)
def no_host_scan(monkeypatch):
    """ไม่สแกน process/container จริงของเครื่อง dev — เทสนี้มีแต่ของที่สร้างเอง"""
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setenv("LMDS_WATCHDOG_ROOT", "")     # ตั้งจริงใน fixture `fleet`


class Fleet:
    """โฟลเดอร์ทดลอง: bundle + ทะเบียน + process จริงที่ทำตัวเป็นโมเดล"""

    def __init__(self, root: Path):
        self.root = root
        self.pids: list[int] = []

    def spawn(self, script: str = "exec sleep 300") -> int:
        """process ที่ **ไม่ใช่ลูกของ pytest** — เหมือนโมเดลจริงซึ่งไม่ใช่ลูกของ `lmds`

        ลูกตรง ๆ ที่ถูกฆ่าจะค้างเป็น zombie จนกว่าเราจะ wait() และ `kill -0` ยังตอบว่า "อยู่" —
        controller ที่รอให้ตายจริงจะรอไม่มีวันจบ ซึ่งเป็นเรื่องของเทส ไม่ใช่ของโค้ด
        """
        out = subprocess.run(["bash", "-c", f"( {script} ) >/dev/null 2>&1 & echo $!"],
                             capture_output=True, text=True, check=True)
        pid = int(out.stdout.strip())
        self.pids.append(pid)
        return pid

    def model(self, slug: str, controller_body: str, *, mode: str = "native",
              weights: bool = False) -> int:
        bundle = self.root / "work" / slug
        bundle.mkdir(parents=True)
        pid = self.spawn()
        run_dir = self.root / "run" / slug
        run_dir.mkdir(parents=True)
        pid_file = run_dir / "server.pid"
        pid_file.write_text(str(pid), encoding="utf-8")
        controller = bundle / f"{slug}-single.sh"
        controller.write_text(
            "#!/usr/bin/env bash\n"
            f'PID_FILE="{pid_file}"\n'
            f'echo "$@" >> "{bundle}/calls.log"\n'
            + controller_body, encoding="utf-8")
        controller.chmod(0o755)
        model_dir = self.root / "models" / slug
        if weights:
            model_dir.mkdir(parents=True)
            (model_dir / "weights.gguf").write_bytes(b"w" * 2048)
        (bundle / "MODEL_PROFILE.yaml").write_text(
            f"model: {{id: org/{slug}, served_name: {slug}}}\n"
            "runtime: {engine: llamacpp, native_build: true}\nserving: {port: 59999}\n",
            encoding="utf-8")
        (run_dir / "server.meta").write_text(
            f"slug={slug}\nmodel={slug}\nmodel_id=org/{slug}\nengine=llamacpp\nmode={mode}\n"
            f"port=59999\ncontainer=\npid_file={pid_file}\ncontroller={controller}\n"
            "started_at=2026-10-06T10:00:00\n", encoding="utf-8")
        return pid

    def calls(self, slug: str) -> list[str]:
        log = self.root / "work" / slug / "calls.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def bundle(self, slug: str) -> Path:
        return self.root / "work" / slug


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@pytest.fixture
def fleet(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LMDS_WATCHDOG_ROOT", str(tmp_path / "watchdog"))
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.chdir(tmp_path)
    box = Fleet(tmp_path)
    yield box
    for pid in box.pids:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


# controller ที่ทำงานถูก: stop ฆ่า process ที่ pid file ชี้ แล้วรอจนมันตายจริง
GOOD = """case "$1" in
  stop|restart) pid="$(cat "$PID_FILE")"; kill "$pid" 2>/dev/null
        while kill -0 "$pid" 2>/dev/null; do sleep 0.05; done ;;
esac
exit 0
"""
# controller ที่ docker ปฏิเสธ — พิมพ์สิ่งที่ docker พูดจริง แล้วจบด้วย exit 1
DENIED = f"""echo "docker บอกว่า: {DOCKER_DENIED}" >&2
echo "แก้: sudo usermod -aG docker $USER แล้ว login ใหม่" >&2
exit 1
"""
# controller ที่บอกว่าสำเร็จ (exit 0) แต่ไม่ได้หยุดอะไรเลย
LIAR = "exit 0\n"


# ── restart ──────────────────────────────────────────────────────────────────
def test_a_restart_the_controller_failed_is_an_error_carrying_what_the_controller_said(fleet):
    proc = fleet.model("qwen", DENIED)

    with pytest.raises(FleetError) as caught:
        manager.restart_server(manager.find("qwen"))

    said = str(caught.value)
    assert "docker.sock" in said, "ข้อความต้องพาสิ่งที่ controller พูดเองมาด้วย ไม่ใช่แค่บอกว่าล้ม"
    assert "exit 1" in said
    assert fleet.calls("qwen") == ["restart"]
    assert alive(proc)


def test_cli_restart_exits_non_zero_and_does_not_claim_it_restarted(fleet):
    fleet.model("qwen", DENIED)

    result = runner.invoke(app, ["restart", "qwen"])

    assert result.exit_code == 1, result.output
    assert "restart qwen แล้ว" not in result.output
    assert "docker.sock" in result.output


def test_cli_restart_with_a_flag_the_controller_rejects_is_a_failure(fleet):
    """`lmds restart demo --port notaport` — controller ตรวจค่าเองแล้วปฏิเสธ · เดิม LMDS ทิ้ง exit code"""
    fleet.model("demo", """for a in "$@"; do
  if [ "$a" = notaport ]; then echo "ERROR: --port ต้องเป็นตัวเลข (ได้ notaport)" >&2; exit 2; fi
done
exit 0
""")

    result = runner.invoke(app, ["restart", "demo", "--port", "notaport"])

    assert result.exit_code == 1, result.output
    assert "restart demo แล้ว" not in result.output
    assert "notaport" in result.output
    assert fleet.calls("demo") == ["restart --port notaport"]


def test_cli_restart_that_works_still_says_so(fleet):
    fleet.model("qwen", GOOD)

    result = runner.invoke(app, ["restart", "qwen"])

    assert result.exit_code == 0, result.output
    assert "restart qwen แล้ว" in result.output


# ── stop ─────────────────────────────────────────────────────────────────────
def test_cli_stop_does_not_say_stopped_when_the_controller_failed(fleet):
    proc = fleet.model("demo", 'echo "controller: $1 FAILED (docker rm refused)" >&2\nexit 7\n')

    result = runner.invoke(app, ["stop", "demo"])

    assert result.exit_code == 1, result.output
    assert "หยุด demo แล้ว" not in result.output
    assert "docker rm refused" in result.output
    assert "exit 7" in result.output
    assert alive(proc), "เทสนี้ตั้งใจให้โมเดลยังรันอยู่ — นั่นคือสิ่งที่ข้อความเดิมโกหก"


def test_a_stop_that_exits_zero_but_left_the_model_running_is_not_a_stop(fleet, monkeypatch):
    """exit 0 ไม่ใช่หลักฐานว่าหยุด — ถามทะเบียน process อีกครั้งก่อนบอกว่าหยุดแล้ว"""
    monkeypatch.setattr(manager, "STOP_CONFIRM_SECONDS", 0.3)
    proc = fleet.model("demo", LIAR)

    with pytest.raises(FleetError) as caught:
        manager.stop_server(manager.find("demo"))

    assert "ยังรันอยู่" in str(caught.value)
    assert alive(proc)

    result = runner.invoke(app, ["stop", "demo"])
    assert result.exit_code == 1, result.output
    assert "หยุด demo แล้ว" not in result.output


def test_cli_stop_that_really_stopped_says_so(fleet):
    proc = fleet.model("demo", GOOD)

    result = runner.invoke(app, ["stop", "demo"])

    assert result.exit_code == 0, result.output
    assert "หยุด demo แล้ว (controller)" in result.output
    assert gone(proc)
    assert manager.find("demo").running is False


def test_stop_all_exits_non_zero_and_names_the_ones_that_did_not_stop(fleet):
    good = fleet.model("alpha", GOOD)
    bad = fleet.model("bravo", DENIED)

    result = runner.invoke(app, ["stop", "--all"])

    assert result.exit_code == 1, result.output
    assert "หยุด alpha แล้ว" in result.output
    assert "หยุด bravo แล้ว" not in result.output
    last = [line for line in result.output.splitlines() if line.strip()][-1]
    assert "bravo" in last and "alpha" not in last, f"บรรทัดสรุปต้องบอกว่าตัวไหนไม่หยุด: {last}"
    assert gone(good)
    assert alive(bad)


def test_stop_all_that_stopped_everything_exits_zero(fleet):
    fleet.model("alpha", GOOD)
    fleet.model("bravo", GOOD)

    result = runner.invoke(app, ["stop", "--all"])

    assert result.exit_code == 0, result.output


def test_the_kill_fallback_waits_for_the_process_to_die_before_saying_it_did(fleet):
    """controller หาย → SIGTERM ตรง ๆ · SIGTERM เป็นแค่คำขอ — ต้องเห็นว่าตายจริงก่อนรายงาน"""
    proc = fleet.model("demo", GOOD)
    (fleet.bundle("demo") / "demo-single.sh").unlink()

    assert manager.stop_server(manager.find("demo")) == "kill"
    assert not alive(proc), "คืนค่าแล้วต้องตายแล้ว ไม่ใช่ 'กำลังจะตาย'"


def test_the_kill_fallback_admits_it_when_the_process_ignores_sigterm(fleet, tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "STOP_CONFIRM_SECONDS", 0.5)
    stubborn = fleet.spawn("trap '' TERM; while :; do sleep 0.1; done")
    time.sleep(0.3)                      # ให้ bash ติดตั้ง trap ก่อน
    run_dir = tmp_path / "run" / "stuck"
    run_dir.mkdir(parents=True)
    (run_dir / "server.pid").write_text(str(stubborn), encoding="utf-8")
    (run_dir / "server.meta").write_text(
        f"slug=stuck\nmodel=stuck\nengine=llamacpp\nmode=native\nport=59998\ncontainer=\n"
        f"pid_file={run_dir / 'server.pid'}\ncontroller=/no/such/ctl.sh\n"
        "started_at=2026-10-06T10:00:00\n", encoding="utf-8")

    with pytest.raises(FleetError) as caught:
        manager.stop_server(manager.find("stuck"))

    assert "ยังรันอยู่" in str(caught.value)
    assert alive(stubborn)


def test_a_docker_rm_that_docker_refused_is_not_reported_as_removed(tmp_path, monkeypatch):
    """ทาง fallback ของ container (ไม่มี controller) ก็ทิ้ง exit code เหมือนกัน — docker ปลอมที่
    ปฏิเสธด้วยข้อความจริงของ docker · ต้องได้ error ที่พาข้อความนั้นมา ไม่ใช่ "docker-rm\""""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    docker = fakebin / "docker"
    docker.write_text(f'#!/usr/bin/env bash\necho "{DOCKER_DENIED}" >&2\nexit 1\n', encoding="utf-8")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ['PATH']}")
    ours = manager.ServerInfo(slug="qwen3", mode="docker", container="lmds-qwen3", running=True)

    with pytest.raises(FleetError) as caught:
        manager.stop_server(ours)

    assert "docker.sock" in str(caught.value)


# ── เว็บ: ทางเดียวกัน ผลเดียวกัน ───────────────────────────────────────────────
def test_web_stop_and_restart_return_an_error_with_the_controllers_words(fleet, monkeypatch):
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    proc = fleet.model("qwen", DENIED)
    client = TestClient(create_app())

    for verb in ("restart", "stop"):
        answer = client.post(f"/api/models/qwen/{verb}")
        assert answer.status_code == 409, (verb, answer.text)
        assert "docker.sock" in answer.json()["detail"], verb
    assert alive(proc)


# ── remove: หยุดไม่ได้ = ไม่ลบ ────────────────────────────────────────────────
def test_remove_stops_right_there_when_the_model_could_not_be_stopped(fleet, monkeypatch):
    """เดิม: "หยุดไม่สำเร็จ" เป็นแค่บรรทัดหนึ่งในรายงาน แล้วเดินต่อไปลบ bundle กับ weight
    ของโมเดลที่ยังรันอยู่ — process ที่ถือไฟล์ที่ถูกลบไปแล้วรันต่อจนกว่าจะ restart แล้วก็ไม่ขึ้นอีกเลย"""
    monkeypatch.setenv("MODEL_DIR", str(fleet.root / "models" / "qwen"))
    proc = fleet.model("qwen", DENIED, weights=True)
    info = manager.find("qwen")
    planned = [item.path for item in manager.removal_plan(info)]
    assert planned, "แผนลบต้องมีของ ไม่งั้นเทสนี้พิสูจน์อะไรไม่ได้"

    with pytest.raises(FleetError) as caught:
        manager.remove_server(info)

    said = str(caught.value)
    assert "ยังรันอยู่" in said and "docker.sock" in said
    assert alive(proc)
    for path in planned:
        assert path.exists(), f"ลบ {path} ไปแล้วทั้งที่โมเดลยังรันอยู่"
    assert (fleet.root / "run" / "qwen" / "server.meta").exists()


def test_cli_remove_reports_the_running_model_instead_of_suggesting_sudo_rm(fleet):
    proc = fleet.model("qwen", DENIED)

    result = runner.invoke(app, ["remove", "qwen", "-y"])

    assert result.exit_code != 0, result.output
    assert "ลบ qwen เรียบร้อย" not in result.output
    assert "ยังรันอยู่" in result.output
    assert "sudo rm -rf" not in result.output, "ไม่มีอะไรให้ sudo rm — ปัญหาคือโมเดลยังไม่หยุด"
    assert fleet.bundle("qwen").exists()
    assert alive(proc)


def test_web_remove_answers_409_and_keeps_the_bundle(fleet):
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    fleet.model("qwen", DENIED)

    answer = TestClient(create_app()).post("/api/models/qwen/remove", json={})

    assert answer.status_code == 409, answer.text
    assert "ยังรันอยู่" in answer.json()["detail"]
    assert fleet.bundle("qwen").exists()


def test_remove_still_removes_a_model_that_stops_cleanly(fleet):
    proc = fleet.model("qwen", GOOD)

    lines = manager.remove_server(manager.find("qwen"))

    assert not manager.removal_failed(lines), lines
    assert gone(proc)
    assert not fleet.bundle("qwen").exists()
