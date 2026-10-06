"""audit 2026-10-06 — watchdog: ของลูกค้าที่ adopt มา · โมเดลที่คนสั่งหยุด · restart ที่ล้ม · status ที่ไม่โกหก

ทุกข้อในไฟล์นี้คือรูปเดียวกับตาราง "หน้าจอบอกว่าผ่าน ไม่ใช่หลักฐานว่าผ่าน" ของ maintainer skill:

1. `lmds adopt` ทำให้ข้อห้าม "container ที่ LMDS ไม่ได้สร้าง" หายไป (`external` กลายเป็น False) —
   watchdog เปิดได้ แล้ว restart ของมันคือ `docker rm -f` + รันคำสั่งที่เราเขียนใหม่บน container ของลูกค้า
3. คนสั่ง `lmds stop` แล้วลูปเดิน `waiting, waiting, restart` — ปลุกโมเดลที่ตั้งใจหยุดขึ้นมาใหม่
5. restart ที่ controller ล้ม ถูกจดว่าสำเร็จ แล้วพัก settle ให้โมเดลที่ไม่เคยถูก restart
6. status ของ "ไม่เคยยิง probe สักรอบ" กับ "ยิงสำเร็จสามรอบ" เหมือนกันทุกตัวอักษร และไม่เคยถาม systemd
7. service ของ user ที่ไม่มี linger ตายพร้อมสาย SSH ทั้งที่สถานะบอกว่าเปิดอยู่

เทสรันของจริง: controller เป็น bash จริง · "โมเดล" เป็น process จริง · probe เป็น httpx จริงที่ยิงพอร์ตปิด ·
`docker` / `systemctl` / `loginctl` เป็นสคริปต์ปลอมบน PATH ที่จด argv ไว้ — สิ่งที่ยืนยันคือคำสั่งที่ถูกส่งออกไปจริง
และสภาพของ process หลังจบ ไม่ใช่ว่าฟังก์ชันภายในถูกเรียกกี่ครั้ง
"""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.cli.main import app  # noqa: E402
from lmds.fleet import manager  # noqa: E402
from lmds.fleet import watchdog as wd  # noqa: E402
from lmds.fleet.manager import FleetError  # noqa: E402

runner = CliRunner()

DOCKER_DENIED = ("permission denied while trying to connect to the docker API at "
                 "unix:///var/run/docker.sock")

# docker ปลอม: จดทุกคำสั่ง · ตอบจากไฟล์ข้าง ๆ — ไม่มีไฟล์ = เครื่องที่ไม่มี container อะไรรันอยู่
FAKE_DOCKER = r"""#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
echo "$*" >> "$here/docker.log"
case "$1" in
  inspect)   case " $* " in *" --format "*) echo "" ;; *) cat "$here/inspect.json" ;; esac ;;
  container) cat "$here/running.txt" 2>/dev/null || echo false ;;
  ps)        if [ "$2" = "--format" ]; then cat "$here/ps.txt" 2>/dev/null || true; fi ;;
esac
exit 0
"""
# systemctl ปลอม: `is-active` ตอบตาม active.txt (คำ + exit code เหมือนของจริง) · `show` ตอบตาม show.txt
FAKE_SYSTEMCTL = r"""#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
echo "$*" >> "$here/systemctl.log"
if [ -f "$here/bus-down" ]; then echo "Failed to connect to bus: No medium found" >&2; exit 1; fi
case " $* " in
  *" is-active "*) word="$(cat "$here/active.txt" 2>/dev/null || echo inactive)"; echo "$word"
                   [ "$word" = active ] && exit 0; exit 3 ;;
  *" show "*)      cat "$here/show.txt" 2>/dev/null || true ;;
esac
exit 0
"""
FAKE_LOGINCTL = r"""#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
echo "$*" >> "$here/loginctl.log"
echo "Linger=$(cat "$here/linger.txt" 2>/dev/null || echo yes)"
"""
# controller ที่จัดการ process จริง: start ปล่อย `sleep` · stop ฆ่าแล้วรอจนตาย · restart = ทั้งสอง
CONTROLLER = r"""#!/usr/bin/env bash
PID_FILE="__PID_FILE__"
echo "$@" >> "$(dirname "$0")/calls.log"
stop_it() {
  local pid; pid="$(cat "$PID_FILE" 2>/dev/null || true)"; [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do sleep 0.05; done
}
start_it() { ( exec sleep 300 ) >/dev/null 2>&1 & echo $! > "$PID_FILE"; }
denied() { echo "docker บอกว่า: __DENIED__" >&2; exit 1; }
case "$1" in
  stop)    [ -z "${FAIL_STOP:-}" ] || denied; stop_it ;;
  start)   start_it ;;
  restart) [ -z "${FAIL_RESTART:-}" ] || denied; stop_it; start_it ;;
esac
exit 0
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Box:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("systemctl", FAKE_SYSTEMCTL), ("loginctl", FAKE_LOGINCTL)):
            (self.bin / name).write_text(body, encoding="utf-8")
            (self.bin / name).chmod(0o755)
        self.start = time.time()
        self.now = self.start
        self.pid_files: list[Path] = []

    # นาฬิกาของลูป: เริ่มที่เวลาจริง (เวลาที่คนสั่ง stop ถูกจดด้วยเวลาจริง) แล้วเดินทีละรอบโดยไม่ต้องรอ
    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def model(self, slug: str = "qwen") -> int:
        """bundle native ที่รันอยู่จริง (process `sleep`) บนพอร์ตที่ไม่มีใครฟัง — probe ได้ connection refused"""
        bundle = self.root / "work" / slug
        bundle.mkdir(parents=True)
        run_dir = self.root / "run" / slug
        run_dir.mkdir(parents=True)
        pid_file = run_dir / "server.pid"
        self.pid_files.append(pid_file)
        controller = bundle / f"{slug}-single.sh"
        controller.write_text(CONTROLLER.replace("__PID_FILE__", str(pid_file))
                              .replace("__DENIED__", DOCKER_DENIED), encoding="utf-8")
        controller.chmod(0o755)
        (bundle / "MODEL_PROFILE.yaml").write_text(
            f"model: {{id: org/{slug}, served_name: {slug}}}\n"
            "runtime: {engine: llamacpp, native_build: true}\n", encoding="utf-8")
        subprocess.run([str(controller), "start"], check=True)
        (bundle / "calls.log").unlink()
        (run_dir / "server.meta").write_text(
            f"slug={slug}\nmodel={slug}\nmodel_id=org/{slug}\nengine=llamacpp\nmode=native\n"
            f"port={free_port()}\ncontainer=\npid_file={pid_file}\ncontroller={controller}\n"
            "started_at=2026-10-06T10:00:00\n", encoding="utf-8")
        return self.pid(slug)

    def pid(self, slug: str = "qwen") -> int:
        return int((self.root / "run" / slug / "server.pid").read_text(encoding="utf-8").strip())

    def controller(self, slug: str = "qwen") -> Path:
        return self.root / "work" / slug / f"{slug}-single.sh"

    def calls(self, slug: str = "qwen") -> list[str]:
        log = self.root / "work" / slug / "calls.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def log(self, tool: str) -> list[str]:
        path = self.bin / f"{tool}.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def loop(self, slug: str = "qwen", rounds: int = 1, **kwargs) -> list[str]:
        """ลูปจริง: หา ServerInfo จริงทุกรอบ · probe เป็น httpx จริง · restart เรียก controller จริง"""
        actions: list[str] = []
        self.reports: list[dict] = []

        def seen(report: dict) -> None:
            actions.append(report["action"])
            self.reports.append(report)

        wd.loop(slug, rounds=rounds, clock=self.clock, sleeper=self.sleep, on_tick=seen, **kwargs)
        return actions

    def customer_container(self, name: str = "coder-next") -> int:
        """เครื่องที่มี container ของลูกค้ารันอยู่ก่อน LMDS — สิ่งที่ `docker` ปลอมจะรายงาน"""
        port = free_port()
        (self.bin / "inspect.json").write_text(json.dumps([{
            "Name": f"/{name}", "Path": "python3",
            "Args": ["-m", "vllm.entrypoints.openai.api_server", "--model", "Qwen/Qwen3-Coder-Next",
                     "--port", "8000"],
            "Config": {"Image": "vllm/vllm-openai:v0.11.0", "Env": ["PATH=/usr/bin"],
                       "Entrypoint": ["python3", "-m", "vllm.entrypoints.openai.api_server"]},
            "HostConfig": {"Binds": ["/home/u/.cache/huggingface:/root/.cache/huggingface"],
                           "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": str(port)}]},
                           "NetworkMode": "bridge", "Runtime": "runc", "IpcMode": "host",
                           "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}],
                           "ShmSize": 67108864}}]), encoding="utf-8")
        (self.bin / "ps.txt").write_text(
            f"{name}\tvllm/vllm-openai:v0.11.0\t0.0.0.0:{port}->8000/tcp\trunning\n", encoding="utf-8")
        (self.bin / "running.txt").write_text("true\n", encoding="utf-8")
        return port


@pytest.fixture
def box(tmp_path, monkeypatch, isolated_config):
    from lmds.hardware import serving

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LMDS_WATCHDOG_ROOT", str(tmp_path / "watchdog"))
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    monkeypatch.setenv("LMDS_USER_SYSTEMD_DIR", str(tmp_path / "systemd-user"))
    monkeypatch.setenv("LMDS_ROLE", "serving")          # เครื่อง dev ไม่มี GPU — `lmds start` ต้องไม่ถูกปัดตก
    serving.reset_cache()
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.chdir(tmp_path)
    made = Box(tmp_path)
    monkeypatch.setenv("PATH", f"{made.bin}{os.pathsep}{os.environ['PATH']}")
    yield made
    serving.reset_cache()
    for pid_file in made.pid_files:
        try:
            os.kill(int(pid_file.read_text(encoding="utf-8").strip()), 9)
        except (OSError, ValueError):
            pass


def audit() -> list[dict]:
    from lmds.web import audit as log

    return log.read(500)


def policy(**over) -> wd.Policy:
    base = dict(interval=120, failures_before_restart=3, max_restarts=3, backoff_seconds=0,
                settle_seconds=600, probe_timeout=3)
    base.update(over)
    return wd.Policy(**base)


# ── 1. bundle ที่ adopt มา ────────────────────────────────────────────────────
def test_adopting_a_customer_container_does_not_open_it_to_the_watchdog(box):
    """`BEFORE adopt: external=True refusals:[…]` / `AFTER adopt: external=False refusals: []` — ข้อห้าม
    หายไปพอดีตอนที่ของลูกค้าเข้ามาอยู่ในมือเรา"""
    from lmds.fleet.adopt import adopt

    box.customer_container()
    before = manager.find("coder-next")
    assert before.external is True and wd.refusals(before), "ก่อน adopt ต้องถูกห้ามอยู่แล้ว (พฤติกรรมเดิม)"

    controller = adopt("coder-next", output=box.root / "work")
    info = manager.find("coder-next")

    assert controller.name == "coder-next-adopted.sh"
    assert info.external is False and info.registered is True and info.controller_exists
    reasons = wd.refusals(info)
    assert reasons, "adopt แล้วข้อห้ามต้องยังอยู่"
    assert "adopted" in reasons[0] and "--allow-adopted" in reasons[0], reasons
    with pytest.raises(FleetError) as caught:
        wd.arm(info)
    assert "--allow-adopted" in str(caught.value)
    assert wd.load("coder-next").armed is False


def test_the_opt_in_for_an_adopted_bundle_is_a_named_flag_and_leaves_a_trace(box):
    from lmds.fleet.adopt import adopt

    box.customer_container()
    adopt("coder-next", output=box.root / "work")

    refused = runner.invoke(app, ["watchdog", "arm", "coder-next"])
    assert refused.exit_code == 1, refused.output
    assert "--allow-adopted" in refused.output
    assert wd.load("coder-next").armed is False

    allowed = runner.invoke(app, ["watchdog", "arm", "coder-next", "--allow-adopted",
                                  "--failures", "2", "--settle", "60"])
    assert allowed.exit_code == 0, allowed.output
    state = wd.load("coder-next")
    assert state.armed is True and state.allow_adopted is True
    armed = [a for a in audit() if a["method"] == "WATCHDOG-ARM"]
    assert armed and "adopted" in armed[-1]["reason"]

    # เปิดด้วยความตั้งใจแล้ว มันต้องทำงานจริง: พลาดสองรอบ → restart ผ่าน controller ที่ adopt เขียน
    assert box.loop("coder-next", rounds=2) == ["waiting", "restart"]
    sent = box.log("docker")
    assert "rm -f coder-next" in sent
    assert any(line.startswith("run -d --name coder-next") for line in sent), sent


def test_a_watchdog_armed_before_this_rule_does_not_keep_restarting_the_customers_container(box):
    """สถานะที่เปิดไว้ก่อนมีกติกานี้ไม่มี `allow_adopted` — ลูปที่วนอยู่ต้องไม่ `docker rm -f` ของลูกค้าต่อ"""
    from lmds.fleet.adopt import adopt

    box.customer_container()
    adopt("coder-next", output=box.root / "work")
    wd.save(wd.State(slug="coder-next", armed=True, armed_at="2026-09-22T09:00:00",
                     policy=dataclasses.asdict(policy(failures_before_restart=2))))
    (box.bin / "docker.log").unlink(missing_ok=True)

    actions = box.loop("coder-next", rounds=5)

    assert actions == ["waiting", "blocked", "blocked", "blocked", "blocked"], actions
    touched = [line for line in box.log("docker") if line.split()[0] in {"rm", "run", "stop", "restart"}]
    assert touched == [], f"watchdog แตะ container ของลูกค้า: {touched}"
    state = wd.load("coder-next")
    assert state.restarts == []
    skipped = [a for a in audit() if a["method"] == "WATCHDOG-SKIPPED"]
    assert len(skipped) == 1, "ลง audit ครั้งเดียวต่อช่วงที่ล่ม ไม่ใช่ทุกรอบ"
    assert "--allow-adopted" in skipped[0]["reason"] and skipped[0]["actor"] == "watchdog"
    assert "--allow-adopted" in box.reports[-1]["reason"], "บรรทัดของลูปต้องบอกเหตุด้วย"

    shown = runner.invoke(app, ["watchdog", "status", "coder-next"])
    assert shown.exit_code == 0, shown.output
    said = " ".join(shown.output.split())
    assert "จะไม่ restart ตัวนี้ให้" in said and "--allow-adopted" in said
    as_json = json.loads(runner.invoke(app, ["watchdog", "status", "coder-next", "--json"]).output)
    assert "--allow-adopted" in as_json[0]["blocked"]


def test_status_names_the_block_even_before_the_loop_has_run_once(box):
    """ลูปยังไม่ทันวนหลังอัปเดต — status ต้องบอกเองจาก bundle ไม่รอให้ลูปจดเหตุผลลงไฟล์"""
    from lmds.fleet.adopt import adopt

    box.customer_container()
    adopt("coder-next", output=box.root / "work")
    state = wd.State(slug="coder-next", armed=True)
    info = manager.find("coder-next")

    assert not any("--allow-adopted" in line for line in wd.describe(state, "en"))
    assert any("will not restart" in line and "--allow-adopted" in line
               for line in wd.describe(state, "en", info=info))


# ── 3. คนสั่งหยุดเอง ───────────────────────────────────────────────────────────
def test_a_model_the_operator_stopped_on_purpose_stays_stopped(box):
    """เดิม: หลัง `lmds stop` → `waiting, waiting, restart` — หน่วยความจำที่เพิ่งเคลียร์ให้โมเดลถัดไปถูกเอาคืน"""
    pid = box.model()
    wd.arm(manager.find("qwen"), policy=policy())

    stopped = runner.invoke(app, ["stop", "qwen"])
    assert stopped.exit_code == 0, stopped.output
    assert not alive(pid)

    actions = box.loop(rounds=6)

    assert actions == ["paused"] * 6, actions
    assert box.calls() == ["stop"], "controller ต้องได้รับแค่ stop ของคน ไม่มี restart ตามมา"
    assert not alive(box.pid()), "ไม่มี process ใหม่ถูกปลุกขึ้นมา"
    state = wd.load("qwen")
    assert state.armed is True and state.paused_at and state.paused_down is True
    assert state.restarts == []

    shown = " ".join(runner.invoke(app, ["watchdog", "status", "qwen"]).output.split())
    assert "paused: stopped by operator at " + time.strftime("%Y-%m-%d") in shown
    assert "lmds start qwen" in shown
    as_json = json.loads(runner.invoke(app, ["watchdog", "status", "qwen", "--json"]).output)[0]
    assert as_json["paused"] is True and as_json["paused_at"] == state.paused_at
    methods = [a["method"] for a in audit()]
    assert methods.count("WATCHDOG-PAUSE") == 1 and "WATCHDOG-RESTART" not in methods


@pytest.mark.parametrize("verb", ["start", "restart"])
def test_an_operator_start_or_restart_puts_the_watchdog_back_on_duty(box, verb):
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    assert runner.invoke(app, ["stop", "qwen"]).exit_code == 0
    assert wd.load("qwen").paused_at

    again = runner.invoke(app, [verb, "qwen"])
    assert again.exit_code == 0, again.output

    state = wd.load("qwen")
    assert alive(box.pid())
    assert state.paused_at == 0 and state.paused_down is False
    assert state.settle_until > time.time(), "โมเดลที่เพิ่ง start ต้องได้เวลาโหลดก่อนถูกยิง"
    assert box.loop(rounds=1) == ["settling"]
    assert [a["method"] for a in audit() if a["method"].startswith("WATCHDOG-")] == [
        "WATCHDOG-ARM", "WATCHDOG-PAUSE", "WATCHDOG-RESUME"]
    # พ้น settle แล้วกลับมาเฝ้าจริง: พลาดครบสามรอบ → restart
    # เดินนาฬิกาของเทสให้พ้น settle_until ตัวจริง ไม่ใช่ "601 วิจากตอนสร้าง box": settle_until นับจากเวลาจริงตอน
    # `lmds start` จบ ส่วนนาฬิกาของ box เริ่มตอนสร้าง fixture — stop+start ที่กินเวลาจริงเกิน 1 วิ (เครื่องกำลังรัน
    # เทสอื่นอยู่) ทำให้รอบแรกยังเป็น "settling" (ล้มเป็นบางรอบ 2026-10-06 ทั้งรันชุดเต็มและรันไฟล์เดี่ยว)
    box.sleep(max(0.0, state.settle_until - box.clock()) + 1)
    assert box.loop(rounds=3) == ["waiting", "waiting", "restart"]
    assert box.calls()[-1] == "restart"


def test_a_model_that_died_on_its_own_is_still_restarted(box):
    """crash ≠ stop ของคน — ไม่มีใครสั่งหยุด จึงไม่มีอะไรพัก · และ restart ของ watchdog เองไม่นับเป็นคำสั่งของคน"""
    pid = box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    os.kill(pid, 9)

    actions = box.loop(rounds=3)

    assert actions == ["waiting", "waiting", "restart"], actions
    assert box.calls() == ["restart"]
    assert alive(box.pid()) and box.pid() != pid
    state = wd.load("qwen")
    assert state.restarts[0]["ok"] is True
    assert state.paused_at == 0
    assert state.settle_until > box.start, "restart ที่สำเร็จต้องเว้นช่วงให้โมเดลโหลด"
    methods = [a["method"] for a in audit()]
    assert "WATCHDOG-PAUSE" not in methods and "WATCHDOG-RESUME" not in methods


def test_a_stop_that_failed_does_not_leave_the_model_unwatched(box, monkeypatch):
    """controller stop ล้ม = โมเดลยังรันอยู่ — พักไว้คือเลิกเฝ้าของที่ยังเสิร์ฟ"""
    pid = box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    monkeypatch.setenv("FAIL_STOP", "1")

    failed = runner.invoke(app, ["stop", "qwen"])

    assert failed.exit_code == 1, failed.output
    assert alive(pid)
    assert wd.load("qwen").paused_at == 0
    assert "paused" not in box.loop(rounds=1)


def test_the_web_stop_button_and_the_web_job_pause_it_too(box):
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    from lmds.web import create_app, jobs

    box.model()
    wd.arm(manager.find("qwen"), policy=policy())

    answer = TestClient(create_app()).post("/api/models/qwen/stop")
    assert answer.status_code == 200, answer.text
    assert wd.load("qwen").paused_down is True
    assert box.loop(rounds=3) == ["paused"] * 3

    # ปุ่มบนการ์ดของ hub เดินอีกทาง: งานเบื้องหลังที่เรียก controller ตรง ๆ (ไม่ผ่าน manager.stop_server)
    box.model("other")
    wd.arm(manager.find("other"), policy=policy())
    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    job = jobs.start("other", "stop", str(box.controller("other")))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not wd.load("other").paused_down:
        time.sleep(0.05)
    assert job.exit_code == 0
    assert wd.load("other").paused_down is True
    assert box.loop("other", rounds=3) == ["paused"] * 3
    assert box.calls("other") == ["stop"]


def test_a_model_started_outside_lmds_is_watched_again(box):
    """คนเรียก controller ตรง ๆ / autostart ตอนบูต ไม่ผ่าน `lmds start` — โมเดลรันอยู่แต่ watchdog ยัง "พัก"
    คือเฝ้าในนาม · เห็นว่ากลับมารันแล้วต้องกลับมาเฝ้าเอง"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    assert runner.invoke(app, ["stop", "qwen"]).exit_code == 0
    assert box.loop(rounds=2) == ["paused", "paused"]

    subprocess.run([str(box.controller()), "start"], check=True)

    assert box.loop(rounds=2) == ["resumed", "settling"]
    state = wd.load("qwen")
    assert state.paused_at == 0
    assert "WATCHDOG-RESUME" in [a["method"] for a in audit()]


def test_a_stop_that_arrives_while_a_probe_is_in_flight_wins(box):
    """probe รอได้ถึง 30 วินาที — `lmds stop` ที่มาถึงในช่วงนั้นต้องไม่แพ้ให้กับ state ที่ลูปอ่านไว้ก่อน"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy(failures_before_restart=1))

    def slow_probe() -> wd.ProbeResult:
        wd.operator_event("qwen", "stop-begin", by="somchai")
        return wd.ProbeResult(False, False, 0, "ReadTimeout: no answer in 30s", 30000)

    actions = box.loop(rounds=2, prober=slow_probe, restarter=lambda: pytest.fail("ห้าม restart"))

    assert actions == ["paused", "paused"], actions
    state = wd.load("qwen")
    assert state.paused_at and state.paused_by == "somchai", "การพักที่คนเขียนต้องไม่ถูกลูปเขียนทับ"
    assert state.restarts == []


def test_disarm_reaches_a_loop_that_is_already_running(box):
    """เดิมลูปอ่านสถานะครั้งเดียวแล้วเขียนทับทุกรอบ — `disarm` ถูกเขียนกลับเป็น armed (ลูป nohup ไม่มีวันหยุด)"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    rounds: list[str] = []

    def once(report: dict) -> None:
        rounds.append(report["action"])
        if len(rounds) == 2:
            wd.disarm("qwen")

    wd.loop("qwen", rounds=10, clock=box.clock, sleeper=box.sleep, on_tick=once,
            prober=lambda: wd.ProbeResult(True, True, 200, "", 5))

    assert rounds == ["ok", "ok"], "ลูปต้องหยุดเองในรอบถัดจาก disarm"
    assert wd.load("qwen").armed is False


# ── 5. restart ที่ล้ม ─────────────────────────────────────────────────────────
def test_a_restart_the_controller_failed_is_recorded_and_shown_as_failed(box, monkeypatch):
    """เดิม: `restart_ok=True` · `restarts:[{'ok': True}]` · "1 automatic restarts so far" แล้วพัก settle
    ให้โมเดลที่ไม่เคยถูก restart"""
    pid = box.model()
    wd.arm(manager.find("qwen"), policy=policy(failures_before_restart=1, settle_seconds=900))
    os.kill(pid, 9)
    monkeypatch.setenv("FAIL_RESTART", "1")

    assert box.loop(rounds=1) == ["restart"]

    assert box.reports[0]["restart_ok"] is False
    state = wd.load("qwen")
    assert state.restarts[0]["ok"] is False
    assert "docker.sock" in state.restarts[0]["error"], "ต้องเก็บสิ่งที่ controller พูดไว้ด้วย"
    assert state.settle_until == 0, "ไม่มีอะไรกำลังโหลด — พัก settle คือเลิกเฝ้าโมเดลที่ยังพัง"
    entry = [a for a in audit() if a["method"] == "WATCHDOG-RESTART"][0]
    assert entry["status"] == 500 and "restart failed" in entry["reason"] and "docker.sock" in entry["reason"]

    thai = "\n".join(wd.describe(state, "th", now=box.now))
    english = "\n".join(wd.describe(state, "en", now=box.now))
    assert "ล้มเหลว" in thai and "docker.sock" in thai
    assert "(1 failed)" in english and "last FAILED" in english
    assert wd.report(state, now=box.now)["restarts_failed"] == 1

    # รอบถัดไปยังยิงอยู่ ไม่ได้หลับไป 15 นาที — และครั้งที่ล้มยังนับเข้าโควตา
    box.sleep(120)
    assert box.loop(rounds=4)[0] != "settling"
    assert len(wd.load("qwen").restarts) <= 3


# ── 6. status ────────────────────────────────────────────────────────────────
def test_status_tells_never_probed_apart_from_probing_fine(box):
    """เดิมสองสถานะนี้พิมพ์ออกมาเหมือนกันทุกตัวอักษร — "เปิดอยู่ · 0 restarts" ทั้งคู่"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    never = wd.describe(wd.load("qwen"), "en", now=box.now)

    ok = wd.ProbeResult(True, True, 200, "", 40)
    wd.loop("qwen", rounds=3, clock=box.clock, sleeper=box.sleep, prober=lambda: ok)
    state = wd.load("qwen")
    probed = wd.describe(state, "en", now=box.now)

    assert never != probed
    assert any("no probe has run yet" in line and "no probe has succeeded yet" in line for line in never)
    assert not any("no probe has" in line for line in probed)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state.last_ok_at))
    assert any(f"last successful probe {stamp}" in line and f"last probe {stamp}" in line for line in probed)

    facts = wd.report(state, now=box.now)
    assert facts["never_probed"] is False and facts["never_succeeded"] is False
    assert facts["seconds_since_last_ok"] == 0 and facts["probe_overdue"] is False
    assert wd.report(wd.State(slug="x", armed=True))["never_succeeded"] is True


def test_status_says_so_when_probes_run_but_none_ever_succeeded_and_when_they_stopped_coming(box):
    box.model()
    wd.arm(manager.find("qwen"), policy=policy(failures_before_restart=50))
    box.loop(rounds=2)                               # พอร์ตปิด — ยิงจริงแล้วพลาดจริง
    state = wd.load("qwen")

    thai = "\n".join(wd.describe(state, "th", now=box.now))
    assert "ยังไม่เคยมี probe ที่สำเร็จ" in thai and "no probe has succeeded yet" in thai
    assert "probe ล่าสุด " + time.strftime("%Y-%m-%d") in thai

    # ลูปตายไปแล้วชั่วโมงหนึ่ง: ไฟล์สถานะยัง "เปิดอยู่" — ต้องบอกว่าไม่มี probe ใหม่มา
    later = box.now + 3600
    late = "\n".join(wd.describe(state, "en", now=later))
    assert "no new probe for more than 3 intervals" in late and "not running" in late
    assert wd.report(state, now=later)["probe_overdue"] is True
    assert wd.report(state, now=box.now)["probe_overdue"] is False


def test_status_reports_what_systemd_says_about_the_service(box):
    """เคส msi-4 (2026-09-22): unit ตาย 203/EXEC วน 9 รอบ · status ที่อ่านแต่ไฟล์บอกว่า "เปิดอยู่" """
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    unit = Path(os.environ["LMDS_USER_SYSTEMD_DIR"]) / wd.unit_name("qwen")
    unit.parent.mkdir(parents=True)
    unit.write_text(wd.render_unit("qwen"), encoding="utf-8")
    (box.bin / "active.txt").write_text("activating\n", encoding="utf-8")
    (box.bin / "show.txt").write_text("Result=exit-code\nExecMainStatus=203\nNRestarts=9\n", encoding="utf-8")

    shown = runner.invoke(app, ["watchdog", "status", "qwen"])

    assert shown.exit_code == 0, shown.output
    said = " ".join(shown.output.split())
    assert "service lmds-watchdog-qwen.service: activating" in said
    assert "ไม่ได้รันอยู่" in said and "ExecMainStatus=203" in said and "NRestarts=9" in said
    assert "--user is-active lmds-watchdog-qwen.service" in box.log("systemctl")
    service = json.loads(runner.invoke(app, ["watchdog", "status", "qwen", "--json"]).output)[0]["service"]
    assert service["state"] == "activating" and service["installed"] is True
    assert "ExecMainStatus=203" in service["detail"]

    (box.bin / "active.txt").write_text("active\n", encoding="utf-8")
    healthy = " ".join(runner.invoke(app, ["watchdog", "status", "qwen"]).output.split())
    assert "service lmds-watchdog-qwen.service: active" in healthy and "ไม่ได้รันอยู่" not in healthy


def test_status_says_unknown_when_systemd_cannot_be_asked(box, monkeypatch):
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())

    (box.bin / "bus-down").write_text("", encoding="utf-8")
    refused = wd.service_state("qwen")
    assert refused["state"] == "unknown" and "Failed to connect to bus" in refused["detail"]
    shown = " ".join(runner.invoke(app, ["watchdog", "status", "qwen"]).output.split())
    assert "service lmds-watchdog-qwen.service: unknown" in shown
    assert "Failed to connect to bus" in shown and "ไม่ได้แปลว่าลูปรันอยู่" in shown

    empty = box.root / "no-tools"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    missing = wd.service_state("qwen")
    assert missing["state"] == "unknown" and "systemctl" in missing["detail"]
    lines = " ".join(wd.describe(wd.load("qwen"), "en", service=missing))
    assert "unknown" in lines and "does not mean the loop is running" in lines


def test_a_watchdog_with_no_unit_installed_is_not_described_as_served(box):
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())

    shown = " ".join(runner.invoke(app, ["watchdog", "status", "qwen"]).output.split())

    assert "service: ไม่ได้ติดตั้ง" in shown and "lmds watchdog run" in shown


# ── 7. linger ────────────────────────────────────────────────────────────────
def test_the_service_is_not_installed_for_a_user_whose_session_takes_it_down(box):
    """ไม่มี linger: user manager ถูกหยุดเมื่อ session สุดท้ายปิด — service ตายพร้อมสาย SSH ที่ใช้ติดตั้ง"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    (box.bin / "linger.txt").write_text("no\n", encoding="utf-8")
    user = getpass.getuser()

    with pytest.raises(FleetError) as caught:
        wd.install_service("qwen")

    said = str(caught.value)
    assert f"sudo loginctl enable-linger {user}" in said
    assert "lmds watchdog arm qwen --service" in said
    assert f"show-user {user} -p Linger" in box.log("loginctl")
    assert not (Path(os.environ["LMDS_USER_SYSTEMD_DIR"]) / wd.unit_name("qwen")).exists()
    assert not any("enable" in line for line in box.log("systemctl")), box.log("systemctl")

    cli = runner.invoke(app, ["watchdog", "arm", "qwen", "--service"])
    assert cli.exit_code == 1, cli.output
    assert f"sudo loginctl enable-linger {user}" in " ".join(cli.output.split())


def test_the_service_is_installed_and_checked_when_linger_is_on(box):
    box.model()
    (box.bin / "linger.txt").write_text("yes\n", encoding="utf-8")
    (box.bin / "active.txt").write_text("active\n", encoding="utf-8")

    cli = runner.invoke(app, ["watchdog", "arm", "qwen", "--service"])

    assert cli.exit_code == 0, cli.output
    unit = Path(os.environ["LMDS_USER_SYSTEMD_DIR"]) / wd.unit_name("qwen")
    assert unit.is_file()
    sent = box.log("systemctl")
    assert "--user daemon-reload" in sent and "--user enable --now lmds-watchdog-qwen.service" in sent
    assert sent.index("--user enable --now lmds-watchdog-qwen.service") < sent.index(
        "--user is-active lmds-watchdog-qwen.service"), "ต้องถาม systemd ซ้ำหลังติดตั้ง ไม่ใช่เชื่อ exit 0"
    assert "service lmds-watchdog-qwen.service: active" in " ".join(cli.output.split())


def test_status_warns_when_an_installed_service_runs_without_linger(box):
    """unit ที่ enable ไว้กลับมาเองทุกครั้งที่มีคน SSH เข้าไปดู — `active` ตอนเช็คไม่ได้แปลว่าเฝ้าทั้งคืน"""
    box.model()
    wd.arm(manager.find("qwen"), policy=policy())
    unit = Path(os.environ["LMDS_USER_SYSTEMD_DIR"]) / wd.unit_name("qwen")
    unit.parent.mkdir(parents=True)
    unit.write_text(wd.render_unit("qwen"), encoding="utf-8")
    (box.bin / "active.txt").write_text("active\n", encoding="utf-8")
    (box.bin / "linger.txt").write_text("no\n", encoding="utf-8")

    shown = " ".join(runner.invoke(app, ["watchdog", "status", "qwen"]).output.split())

    assert "service lmds-watchdog-qwen.service: active" in shown
    assert "linger" in shown and f"sudo loginctl enable-linger {getpass.getuser()}" in shown
