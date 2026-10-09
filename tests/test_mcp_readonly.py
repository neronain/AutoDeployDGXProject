"""`lmds mcp` อ่านอย่างเดียวเป็นโครงสร้าง — ไม่มีเครื่องมือไหนเปลี่ยนสถานะได้ และไม่มีความลับหลุดออกไปกับคำตอบ

สิ่งที่เทสไฟล์นี้กันไว้ (ทุกข้อเป็นของที่มีอยู่จริงในโค้ดอ่านของ LMDS — สำรวจ 2026-10-09 ตอนสร้าง MCP server):

  * ฟังก์ชัน "อ่าน" ที่เขียนเงียบ ๆ: `fleet.discover()` ลบทะเบียนที่ตายแล้ว · `inventory.request_usage()` เขียน
    usage.samples · `brain.build_plan()` แบบ rule-based เขียน session log · `fleet check --check` เขียนทะเบียนเครื่อง
  * คำตอบที่พาความลับไปด้วย: log ของ vLLM พิมพ์ `api_key` · bundle.args มี `--api-key` · log ของ node มี Bearer token
  * เครื่องมือในวันหน้าที่ถูกผูกกับฟังก์ชันที่เดินไปถึงทางเขียน — ต้องล้มที่ขอบของ process ไม่ใช่เขียนสำเร็จเงียบ ๆ

เทสที่แตะ process ที่ถูกผนึกรันเป็น process ลูกเสมอ (audit hook ถอดไม่ได้ — ผนึก pytest เองแล้วเทสที่เหลือพังทั้งชุด)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time

import pytest

from tests import mcp_fleet
from tests.mcp_fleet import (
    DOCKER_SLUG,
    GHOST_SLUG,
    HOSTS,
    MUTATING_DOCKER_VERBS,
    MUTATING_LMDS_VERBS,
    MUTATING_SYSTEMCTL_VERBS,
    NATIVE_SLUG,
    NODE_DOWN,
    NODE_HUNG,
    NODE_OK,
    PLAIN_REPO,
    PLANTED,
)

MASK = "[REDACTED]"
RO = "LMDS_READ_ONLY=1 "          # หน้าคำสั่งที่ hub ส่งไปเครื่องอื่น — lmds ปลายทางผนึกตัวเอง
NODE_SLUG = "gemma-3-27b"
# ทุกเครื่องมือ ทุกรูปแบบการเรียกที่ไปถึงโค้ดคนละเส้น — hub · เครื่องอื่น · probe ทั้งฟลีต · Hugging Face
EVERY_CALL = [
    ("lmds_version", {}),
    ("lmds_nodes", {}),
    ("lmds_models", {}),
    ("lmds_models", {"node": NODE_OK}),
    ("lmds_models", {"node": NODE_DOWN}),
    ("lmds_inspect", {"model": PLAIN_REPO}),
    ("lmds_inspect", {"model": PLAIN_REPO, "targets": ["dgx-spark-single"], "context": 32768}),
    ("lmds_plan", {"model": PLAIN_REPO, "target": "dgx-spark-single"}),
    ("lmds_plan", {"model": PLAIN_REPO, "target": "rtx-4090"}),
    ("lmds_fit", {"slug": DOCKER_SLUG}),
    ("lmds_fit", {"slug": NATIVE_SLUG, "slots": 2, "context": 8192}),
    ("lmds_fit", {"slug": NODE_SLUG, "node": NODE_OK, "slots": 2}),
    ("lmds_fleet_check", {}),
    ("lmds_fleet_check", {"check": True}),
    ("lmds_watchdog_status", {}),
    ("lmds_watchdog_status", {"slug": NATIVE_SLUG}),
    ("lmds_watchdog_status", {"node": NODE_OK}),
    ("lmds_logs", {"slug": DOCKER_SLUG}),
    ("lmds_logs", {"slug": NATIVE_SLUG, "lines": 500}),
    ("lmds_logs", {"slug": NODE_SLUG, "node": NODE_OK}),
    ("lmds_doctor", {"slug": DOCKER_SLUG}),
    ("lmds_doctor", {"slug": NATIVE_SLUG}),
    ("lmds_doctor", {"slug": GHOST_SLUG}),
    ("lmds_doctor", {"slug": NODE_SLUG, "node": NODE_OK}),
]
EXPECTED_REMOTE = {
    RO + "lmds agent info",
    RO + f"lmds fit {NODE_SLUG} --json --slots 2",
    RO + "lmds watchdog status --json",
    RO + f"lmds logs {NODE_SLUG} -n 100",
    RO + f"lmds doctor {NODE_SLUG} --json --no-probe",
}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    built = mcp_fleet.build(tmp_path, monkeypatch)
    yield built
    built.close()


def call_everything(fleet) -> tuple[list[dict], str]:
    client = fleet.mcp(fake_hub=True)
    client.initialize()
    client.request("tools/list")
    answers = [client.call(name, arguments) for name, arguments in EVERY_CALL]
    assert client.close() == 0
    return answers, client.stderr_text()


# ── ชั้นที่ 1: ทะเบียนผูกได้เฉพาะฟังก์ชันอ่านที่ประกาศไว้ ─────────────────────────────────────────────
def test_every_tool_is_bound_to_a_function_on_the_read_allowlist():
    from lmds.mcp import reads, tools

    assert len(tools.TOOLS) == len({tool.name for tool in tools.TOOLS}) == 10
    for tool in tools.TOOLS:
        name = tool.func.__name__
        assert name in reads.READ_ALLOWLIST, f"{tool.name} ผูกกับ {name} ซึ่งไม่อยู่ใน READ_ALLOWLIST"
        assert tool.func is getattr(reads, name) and tool.func.__module__ == "lmds.mcp.reads"
    assert {tool.func.__name__ for tool in tools.TOOLS} == set(reads.READ_ALLOWLIST), \
        "ชื่อใน READ_ALLOWLIST ที่ไม่มีเครื่องมือใช้ = ช่องที่เปิดทิ้งไว้"
    # ไม่มี argument ไหนของเครื่องมือไหนที่ชื่อบอกว่าสั่งให้ทำ
    for tool in tools.TOOLS:
        assert not set(tool.properties) & {"apply", "confirm", "force", "yes", "write", "command", "argv", "path", "env"}


@pytest.mark.parametrize("bind", ["start_server", "remove", "foreign"])
def test_binding_a_tool_to_anything_else_fails_at_import_time(bind):
    """เพิ่มเครื่องมือที่ผูกกับ `fleet.start_server` หรือ `os.remove` ต้องทำให้ `lmds mcp` ไม่ขึ้นเลย ไม่ใช่ขึ้นแล้วใช้ได้"""
    from lmds import fleet
    from lmds.mcp import reads, tools

    def nodes():                                   # ชื่อตรงกับของใน allowlist แต่ไม่ใช่ตัวที่อยู่ใน reads
        return {}

    func = {"start_server": fleet.start_server, "remove": os.remove, "foreign": nodes}[bind]
    assert bind != "foreign" or func.__name__ in reads.READ_ALLOWLIST
    with pytest.raises(RuntimeError, match="READ_ALLOWLIST"):
        tools.Tool("lmds_bad", func, "local", "x")


# ── ชั้นที่ 3: เรียกทุกเครื่องมือบนฟลีตจำลอง แล้วไม่มีอะไรเปลี่ยน ──────────────────────────────────────
def test_calling_every_tool_changes_nothing_on_the_hub_or_on_any_node(fleet):
    before = fleet.state()
    registry = (fleet.config / "nodes.yaml").read_bytes()
    assert {name for name, _ in EVERY_CALL} == {
        "lmds_version", "lmds_nodes", "lmds_models", "lmds_inspect", "lmds_plan", "lmds_fit", "lmds_fleet_check",
        "lmds_watchdog_status", "lmds_logs", "lmds_doctor"}

    answers, stderr = call_everything(fleet)

    # คำตอบมาจริง ไม่ใช่ error ทั้งชุด (เทสที่ผ่านเพราะไม่มีอะไรรันไม่ได้พิสูจน์อะไร)
    failed = {(name, json.dumps(args, ensure_ascii=False)) for (name, args), answer in zip(EVERY_CALL, answers, strict=True)
              if answer["error"]}
    assert failed == {("lmds_models", json.dumps({"node": NODE_DOWN})),
                      ("lmds_plan", json.dumps({"model": PLAIN_REPO, "target": "rtx-4090"}))}, failed

    # ไฟล์ของ hub: ทะเบียนเครื่อง · config · ทะเบียน runtime · โฟลเดอร์ bundle · key · สถานะ watchdog — เท่าเดิมทุกไบต์
    after = fleet.state()
    assert {path for path in after if path not in before} == set(), "มีไฟล์ใหม่"
    assert {path for path in before if path not in after} == set(), "มีไฟล์หาย"
    assert [path for path in before if before[path] != after[path]] == [], "มีไฟล์ถูกแก้"
    assert (fleet.config / "nodes.yaml").read_bytes() == registry
    assert not (fleet.bundles / DOCKER_SLUG / "bundle.env").exists()

    # เครื่องอื่น: ได้รับเฉพาะคำสั่งอ่านชุดนี้ ไม่มีกริยาที่เปลี่ยนสถานะ ไม่มีอย่างอื่นเลย
    remote = fleet.remote_commands()
    assert set(remote) == EXPECTED_REMOTE
    for command in remote:
        words = shlex.split(command)
        assert words[:2] == ["LMDS_READ_ONLY=1", "lmds"] and words[2] in {"agent", "fit", "logs", "doctor", "watchdog"}
        assert not set(words[2:]) & MUTATING_LMDS_VERBS, command
    for argv in fleet.calls("ssh"):
        assert "BatchMode=yes" in argv and argv[-1].startswith("bash -lc "), argv

    # เครื่องนี้: docker/systemctl ถูกถามอย่างเดียว
    docker = fleet.calls("docker")
    assert ["logs", "--tail", "100", f"lmds-{DOCKER_SLUG}"] in docker
    assert not [argv for argv in docker if argv and argv[0] in MUTATING_DOCKER_VERBS]
    assert not [argv for argv in fleet.calls("systemctl") if set(argv) & MUTATING_SYSTEMCTL_VERBS]

    # ทางเขียนที่ซ่อนในฟังก์ชันอ่านถูกกั้นจริง (ไม่ใช่บังเอิญไม่เกิด) — seal จดไว้บน stderr
    assert "refused: open for writing" in stderr and "usage.samples" in stderr
    assert f"refused: remove {fleet.run / GHOST_SLUG / 'server.meta'}" in stderr


def test_the_same_reads_through_the_cli_do_write_which_is_what_the_seal_is_for(fleet):
    """ตัวพิสูจน์ว่าเทสข้างบนไม่ได้ผ่านเพราะไม่มีอะไรจะเขียน: ฟังก์ชันชุดเดียวกันผ่าน CLI เขียนจริงทั้งสี่ที่"""
    samples = fleet.run / NATIVE_SLUG / "usage.samples"
    ghost = fleet.run / GHOST_SLUG / "server.meta"
    sessions = fleet.config / "sessions"
    registry = fleet.config / "nodes.yaml"
    registry_before = registry.read_bytes()

    plans_before = sorted(sessions.glob("plan-*.json"))          # ของ fixture เอง (render bundle = วางแผนสองครั้ง)
    call_everything(fleet)
    assert not samples.exists() and ghost.exists() and sorted(sessions.glob("plan-*.json")) == plans_before
    assert registry.read_bytes() == registry_before

    assert fleet.lmds("agent", "info").returncode == 0
    assert samples.is_file(), "`lmds agent info` เขียน usage.samples"
    assert not ghost.exists(), "`fleet.discover()` ลบทะเบียนที่ตายแล้ว"
    fleet.lmds("plan", PLAIN_REPO, "--target", "dgx-spark-single", "--no-llm", "--json", fake_hub=True)
    assert len(list(sessions.glob("plan-*.json"))) > len(plans_before), "`lmds plan --no-llm` เขียน session log"
    fleet.lmds("fleet", "check", "--check", "--json")
    assert registry.read_bytes() != registry_before, "`lmds fleet check --check` เขียนทะเบียนเครื่อง"


# ── ชั้นที่ 2: process ที่ถูกผนึก ────────────────────────────────────────────────────────────────────
SEAL_PROBE = r'''
import json, os, shutil, signal, subprocess, sys
from pathlib import Path

home = Path(os.environ["HOME"])
slug, pid = sys.argv[1], int(sys.argv[2])
victim = home / "victim.txt"
victim.write_text("before", encoding="utf-8")
(home / "keep-dir").mkdir()

import lmds.fleet.apikey as apikey
import lmds.fleet.bundle_settings as bundle_settings
import lmds.nodes.registry as registry
from lmds import fleet
from lmds.config import Settings
from lmds.mcp import seal
from lmds.nodes import ssh

RO = "LMDS_READ_ONLY=1 "
nodes = {node.name: node for node in registry.load()}
ok, hung = nodes[sys.argv[3]], nodes[sys.argv[4]]
controller = str(home / "bundles" / slug / (slug + "-single.sh"))

seal.seal()
out = {}


def attempt(name, action):
    try:
        value = action()
    except seal.ReadOnlyViolation:
        out[name] = "refused"
    except Exception as exc:
        out[name] = "%s: %s" % (type(exc).__name__, exc)
    else:
        out[name] = "allowed" if not isinstance(value, str) else "allowed: " + value


def spawn(*argv, **kw):
    return lambda: subprocess.run(list(argv), capture_output=True, **kw) and None


# ── ต้องถูกกั้น ──
attempt("write a file", lambda: victim.write_text("after", encoding="utf-8") and None)
attempt("append to a file", lambda: open(victim, "a").close())
attempt("create a file", lambda: (home / "new.txt").touch())
attempt("delete a file", lambda: victim.unlink())
attempt("rename a file", lambda: victim.rename(home / "moved.txt") and None)
attempt("mkdir", lambda: (home / "new-dir").mkdir())
attempt("rmdir", lambda: (home / "keep-dir").rmdir())
attempt("rmtree", lambda: shutil.rmtree(home / "keep-dir"))
attempt("chmod", lambda: victim.chmod(0o777))
attempt("registry.save", lambda: registry.save(registry.load()) and None)
attempt("registry.update", lambda: registry.update(ok.name, note="changed") and None)
attempt("registry.remove", lambda: registry.remove(ok.name) and None)
attempt("Settings.save", lambda: Settings.load().save())
attempt("bundle_settings.write", lambda: bundle_settings.write(home / "bundles" / slug, {"slots": 2}) and None)
attempt("apikey.write", lambda: apikey.write(slug, "x" * 24) and None)
attempt("fleet.stop_server", lambda: fleet.stop_server(fleet.find(sys.argv[5])) and None)
attempt("fleet.start_server", lambda: fleet.start_server(fleet.find(slug)) and None)
attempt("docker rm", spawn("docker", "rm", "-f", "lmds-" + slug))
attempt("docker run", spawn("docker", "run", "--rm", "busybox", "true"))
attempt("docker stop", spawn("docker", "stop", "lmds-" + slug))
attempt("docker exec", spawn("docker", "exec", "lmds-" + slug, "sh"))
attempt("systemctl restart", spawn("systemctl", "--user", "restart", "lmds-web"))
attempt("systemctl enable", spawn("systemctl", "--user", "enable", "--now", "x.service"))
attempt("git pull", spawn("git", "-C", str(home), "pull", "--ff-only"))
attempt("bash -c", spawn("bash", "-c", "echo hi"))
attempt("shell=True", spawn("echo hi", shell=True))
attempt("os.system", lambda: os.system("true") and None)
attempt("rm", spawn("rm", "-f", str(victim)))
attempt("scp", spawn("scp", str(victim), "tkc@10.9.0.1:"))
attempt("rsync", spawn("rsync", "-a", str(home), "tkc@10.9.0.1:"))
attempt("controller start", spawn(controller, "start"))
attempt("controller stop", spawn(controller, "stop"))
attempt("controller logs", spawn(controller, "logs", "5"))
attempt("ssh with a raw command", spawn("ssh", "tkc@10.9.0.1", "rm -rf ~"))
attempt("ssh.run lmds stop", lambda: ssh.run(ok, RO + "lmds stop gemma-3-27b") and None)
attempt("ssh.run lmds set --fit", lambda: ssh.run(ok, RO + "lmds set gemma-3-27b --fit --json") and None)
attempt("ssh.run read then write", lambda: ssh.run(ok, RO + "lmds agent info; lmds stop gemma-3-27b") and None)
attempt("ssh.run read with substitution", lambda: ssh.run(ok, RO + "lmds logs $(id) -n 5") and None)
attempt("ssh.run lmds node install", lambda: ssh.run(ok, RO + "lmds node install --all") and None)
attempt("ssh.run a read without the read-only prefix", lambda: ssh.run(ok, "lmds agent info") and None)
attempt("ssh.stream", lambda: ssh.stream(ok, "lmds start gemma-3-27b") and None)
attempt("ssh.push_file", lambda: ssh.push_file(ok, str(victim), "victim.txt") and None)
attempt("signal another process", lambda: os.kill(pid, signal.SIGTERM))

# ── ต้องยังทำได้ ──
attempt("read a file", lambda: victim.read_text(encoding="utf-8"))
attempt("mkdir of an existing dir", lambda: (home / "keep-dir").mkdir(exist_ok=True))
attempt("registry.load", lambda: str(len(registry.load())))
attempt("docker ps", spawn("docker", "ps"))
attempt("docker container inspect", spawn("docker", "container", "inspect", "--format", "{{.State.Running}}", "x"))
attempt("git rev-parse", spawn("git", "-C", str(home), "rev-parse", "--short", "HEAD"))
attempt("systemctl is-active", spawn("systemctl", "--user", "is-active", "x.service"))
attempt("docker logs", spawn("docker", "logs", "--tail", "5", "lmds-" + slug))
attempt("fleet.logs_text direct", lambda: fleet.logs_text(fleet.find(slug), 2, direct=True).splitlines()[-1][:8])
attempt("ssh -G", spawn("ssh", "-G", "tkc@10.9.0.1"))
attempt("ssh.run lmds agent info", lambda: str(ssh.run(ok, RO + "lmds agent info").exit_code))
attempt("ssh.run lmds fit", lambda: str(ssh.run(ok, RO + "lmds fit gemma-3-27b --json --slots 2 --context 4096").exit_code))
attempt("kill -0", lambda: os.kill(pid, 0))
attempt("timeout kills its own child", lambda: str(ssh.run(hung, RO + "lmds agent info", timeout=3).exit_code))
print(json.dumps({"attempts": out, "victim": victim.read_text(encoding="utf-8"), "sealed": seal.is_sealed()}))
'''

MUST_BE_REFUSED = [
    "write a file", "append to a file", "create a file", "delete a file", "rename a file", "mkdir", "rmdir", "rmtree",
    "chmod", "registry.save", "registry.update", "registry.remove", "Settings.save", "bundle_settings.write",
    "apikey.write", "fleet.stop_server", "fleet.start_server", "docker rm", "docker run", "docker stop", "docker exec",
    "systemctl restart", "systemctl enable", "git pull", "bash -c", "shell=True", "os.system", "rm", "scp", "rsync",
    "controller start", "controller stop", "controller logs", "ssh with a raw command",
    "ssh.run lmds stop", "ssh.run lmds set --fit", "ssh.run read then write", "ssh.run read with substitution",
    "ssh.run lmds node install", "ssh.run a read without the read-only prefix", "ssh.stream", "ssh.push_file",
    "signal another process",
]
MUST_STILL_WORK = {
    "read a file": "allowed: before", "mkdir of an existing dir": "allowed", "registry.load": "allowed: 3",
    "docker ps": "allowed", "docker container inspect": "allowed", "git rev-parse": "allowed",
    "systemctl is-active": "allowed", "docker logs": "allowed", "fleet.logs_text direct": "allowed: INFO 10-",
    "ssh -G": "allowed",
    "ssh.run lmds agent info": "allowed: 0", "ssh.run lmds fit": "allowed: 0", "kill -0": "allowed",
    "timeout kills its own child": "allowed: 124",
}


def test_a_sealed_process_cannot_write_spawn_a_mutating_command_or_send_one_to_a_node(fleet):
    """ไม่ผ่านทะเบียนเครื่องมือเลย: ผนึก process แล้วลองทางเขียนตรง ๆ ทีละทาง — ของที่เครื่องมือในวันหน้าอาจเดินไปถึง"""
    from lmds.nodes import Node
    from lmds.nodes.registry import load, save

    save([*load(), Node(name=NODE_HUNG, host=HOSTS[NODE_HUNG], user="tkc")])
    fleet.set_node(NODE_HUNG, "hung")
    before = fleet.state()

    done = subprocess.run(
        [sys.executable, "-c", SEAL_PROBE, DOCKER_SLUG, str(fleet.sleeper.pid), NODE_OK, NODE_HUNG, NATIVE_SLUG],
        cwd=fleet.home, env=fleet.env, capture_output=True, text=True, encoding="utf-8", timeout=180,
        stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    attempts = report["attempts"]

    assert report["sealed"] is True and report["victim"] == "before"
    # "refused" = ReadOnlyViolation ถึงผู้เรียกตรง ๆ · บางทางห่อมันเป็น error ของตัวเอง (FleetError ของ stop/start)
    # ซึ่งก็คือถูกกั้นเหมือนกัน — ที่ต้องไม่มีคือ "allowed"
    not_refused = {name: attempts[name] for name in MUST_BE_REFUSED
                   if attempts[name] != "refused" and "is read-only" not in attempts[name]}
    assert not_refused == {}
    assert sum(1 for name in MUST_BE_REFUSED if attempts[name] == "refused") >= len(MUST_BE_REFUSED) - 2
    assert {name: attempts[name] for name in MUST_STILL_WORK} == MUST_STILL_WORK
    assert set(attempts) == set(MUST_BE_REFUSED) | set(MUST_STILL_WORK)

    # ผลบนเครื่อง: มีแค่ไฟล์ที่ตัวทดสอบสร้างเองก่อนผนึก · โมเดลที่ "รันอยู่" ยังอยู่ · ไม่มีคำสั่งเขียนไปถึงโปรแกรมปลอม
    after = fleet.state()
    assert {path for path in after if path not in before} == {"victim.txt", "keep-dir"}
    assert [path for path in before if before[path] != after.get(path)] == []
    assert fleet.sleeper.poll() is None, "process ของโมเดลถูกฆ่า"
    assert sorted(set(fleet.remote_commands())) == [
        RO + "lmds agent info", RO + "lmds fit gemma-3-27b --json --slots 2 --context 4096"]
    docker = [argv[0] for argv in fleet.calls("docker")]
    assert {"ps", "container", "logs"} <= set(docker) and not set(docker) & MUTATING_DOCKER_VERBS
    assert ["--user", "is-active", "x.service"] in fleet.calls("systemctl")
    assert not [argv for argv in fleet.calls("systemctl") if set(argv) & MUTATING_SYSTEMCTL_VERBS]
    # การกั้นเป็น error แบบดิสก์ read-only — โค้ดที่ห่อ `except OSError` (ทางเขียนที่ซ่อนในฟังก์ชันอ่าน) ข้ามได้เอง
    assert "lmds-mcp: read-only — refused: docker rm" in done.stderr


def test_a_node_asked_with_the_read_only_prefix_seals_its_own_lmds(fleet):
    """hub ส่งคำสั่งอ่านไปเครื่องอื่นเป็น `LMDS_READ_ONLY=1 lmds …` — ให้ฟลีตจำลองเล่นเป็นเครื่องปลายทางแล้วรันคำสั่งชุดนั้นจริง

    `lmds agent info` ธรรมดาเขียน usage.samples และลบทะเบียนผี (เทสข้างบน) · ใต้ตัวแปรนี้ต้องได้คำตอบเดิมโดยไม่เขียนอะไร
    และคำสั่งที่ต้องเขียนต้องล้มพร้อมเหตุผล ไม่ใช่ทำไปครึ่งทาง
    """
    sealed = {"LMDS_READ_ONLY": "1"}
    before = fleet.state()
    info = fleet.lmds("agent", "info", env=sealed)
    assert info.returncode == 0, info.stderr[-2000:]
    payload = json.loads(info.stdout)
    assert {model["slug"] for model in payload["models"]} == {DOCKER_SLUG, NATIVE_SLUG}
    assert payload["models"] and payload["summary"]["running"] == 1

    logs = fleet.lmds("logs", NATIVE_SLUG, "-n", "2", env=sealed)
    assert logs.returncode == 0 and logs.stdout.splitlines()[-1].startswith("srv  launch_slot_")
    docker_logs = fleet.lmds("logs", DOCKER_SLUG, "-n", "3", env=sealed)
    assert docker_logs.returncode == 0 and "Avg prompt throughput" in docker_logs.stdout
    assert fleet.lmds("fit", DOCKER_SLUG, "--json", env=sealed).returncode == 0
    assert fleet.lmds("doctor", DOCKER_SLUG, "--json", "--no-probe", env=sealed).returncode in (0, 2)
    assert json.loads(fleet.lmds("watchdog", "status", "--json", env=sealed).stdout)[0]["slug"] == NATIVE_SLUG
    for table in (["ps"], ["list"], ["node", "list"], ["version"]):       # คำสั่งอ่านของคนก็ใช้ได้ใต้ตัวแปรนี้
        assert fleet.lmds(*table, env=sealed).returncode == 0, table
    assert fleet.state() == before, "คำสั่งอ่านใต้ LMDS_READ_ONLY=1 ต้องไม่แตะไฟล์ไหนเลย (รวม server.pid ที่ controller จะลบ)"

    # คำสั่งที่ต้องเขียน: ล้ม และไม่มีอะไรถูกเขียน
    for argv in (["set", DOCKER_SLUG, "--slots", "2"], ["watchdog", "arm", NATIVE_SLUG], ["stop", NATIVE_SLUG],
                 ["node", "remove", NODE_OK, "--yes"], ["key", "new", DOCKER_SLUG]):
        done = fleet.lmds(*argv, env=sealed)
        assert done.returncode != 0, argv
    assert fleet.state() == before
    assert fleet.sleeper.poll() is None

    # เทียบ: คำสั่งอ่านตัวเดียวกันโดยไม่มีตัวแปรนี้เขียนจริง
    fleet.lmds("logs", NATIVE_SLUG, "-n", "2")
    assert not (fleet.run / NATIVE_SLUG / "server.pid").exists(), \
        "controller ของ llama.cpp ลบ server.pid ที่มันเห็นว่าค้างแม้ถูกเรียกด้วย `logs` — เหตุที่เครื่องมือไม่รัน controller"


def test_the_seal_is_a_permission_error_so_tolerant_read_paths_keep_working():
    from lmds.mcp import seal

    assert issubclass(seal.ReadOnlyViolation, PermissionError) and issubclass(seal.ReadOnlyViolation, OSError)
    assert seal.is_sealed() is False, "pytest เองต้องไม่ถูกผนึก"


@pytest.mark.parametrize("command,allowed", [
    (RO + "lmds agent info", True),
    ("lmds version 2>/dev/null | head -1", True),
    (RO + "lmds fit qwen3-32b --json", True),
    (RO + "lmds fit qwen3-32b --json --slots 4 --context 32768", True),
    (RO + "lmds logs qwen3.6-35b-a3b -n 500", True),
    (RO + "lmds doctor qwen3-32b --json --no-probe", True),
    (RO + "lmds watchdog status --json", True),
    (RO + "lmds watchdog status qwen3-32b --json", True),
    ("lmds agent info", False),                       # ไม่มีหน้า LMDS_READ_ONLY=1 = ปลายทางไม่ถูกผนึก — ไม่ส่ง
    ("LMDS_READ_ONLY=0 lmds agent info", False),
    (RO + "lmds agent info ", False),
    (RO + "lmds agent info && lmds stop x", False),
    (RO + "lmds agent info\nlmds stop x", False),
    (RO + "lmds agent bench", False),
    (RO + "lmds set qwen3-32b --fit --json", False),
    (RO + "lmds fit qwen3-32b --json --apply", False),
    (RO + "lmds fit qwen3-32b; id --json", False),
    (RO + "lmds fit $(id) --json", False),
    (RO + "lmds logs qwen3-32b -n 5 -f", False),
    (RO + "lmds logs ../../etc/passwd -n 5", False),
    (RO + "lmds doctor qwen3-32b --json", False),
    (RO + "lmds doctor qwen3-32b", False),
    (RO + "lmds watchdog arm qwen3-32b", False),
    (RO + "lmds watchdog disarm qwen3-32b", False),
    (RO + "lmds watchdog run qwen3-32b --once", False),
    (RO + "lmds start qwen3-32b", False), (RO + "lmds stop --all", False), (RO + "lmds restart qwen3-32b", False),
    (RO + "lmds remove qwen3-32b -y", False), (RO + "lmds node install --all", False),
    (RO + "lmds bundles refresh --all", False), ("sudo reboot", False), ("", False),
])
def test_the_remote_read_list_is_exact(command, allowed):
    """คำสั่งที่ hub ส่งไปเครื่องอื่นได้คือรายการตายตัว เทียบเต็มบรรทัด — ต่อท้าย/แทรก/คล้าย ๆ ไม่ผ่าน"""
    from lmds.mcp import seal

    assert (seal.remote_problem(command) == "") is allowed
    wrapped = ["ssh", "-o", "BatchMode=yes", "-i", "/k", "-p", "22", "tkc@10.9.0.1", f"bash -lc {shlex.quote(command)}"]
    assert (seal.spawn_problem(wrapped) == "") is allowed


# ── ความลับ ───────────────────────────────────────────────────────────────────────────────────────
def test_no_planted_secret_appears_in_any_tool_result(fleet):
    answers, _stderr = call_everything(fleet)
    everything = "\n".join(answer["text"] for answer in answers)
    assert len(everything) > 5000
    leaked = sorted(name for name, value in PLANTED.items() if value in everything)
    assert leaked == [], f"หลุด: {leaked}"
    # ถูกปิด ไม่ใช่ถูกตัดทิ้งทั้งบรรทัด — บรรทัดที่บอกสาเหตุยังอ่านได้
    by_call = {(name, json.dumps(args, ensure_ascii=False, sort_keys=True)): answer
               for (name, args), answer in zip(EVERY_CALL, answers, strict=True)}

    def answer_of(name: str, **args) -> dict:
        return by_call[(name, json.dumps(args, ensure_ascii=False, sort_keys=True))]

    hub_log = answer_of("lmds_logs", slug=DOCKER_SLUG)["payload"]["text"]
    assert f"'api_key': ['{MASK}']" in hub_log and f"rejected key {MASK} for" in hub_log
    assert f"http://deploy:{MASK}@proxy.internal:3128" in hub_log and "Model loading took 61.20 GiB" in hub_log
    assert f"--api-key {MASK}" in answer_of("lmds_logs", slug=NATIVE_SLUG, lines=500)["payload"]["text"]
    node_log = answer_of("lmds_logs", slug=NODE_SLUG, node=NODE_OK)["payload"]["text"]
    assert f"--api-key {MASK} --port 8000" in node_log and f"Bearer {MASK}" in node_log
    assert f"token {MASK}" in node_log and f"session {MASK} opened" in node_log and "CUDA out of memory" in node_log
    drift = answer_of("lmds_models", node=NODE_OK)["payload"]["models"][0]["pending_restart"]["changes"][0]
    assert drift == {"field": "extra_args", "saved": f"--api-key {MASK}", "running": "(ไม่อยู่บน argv)"}
    assert answer_of("lmds_fit", slug=DOCKER_SLUG)["payload"]["current"]["extra_args"] == f"--api-key {MASK}"
    assert answer_of("lmds_fit", slug=NODE_SLUG, node=NODE_OK, slots=2)["payload"]["current"]["extra_args"] == \
        f"--api-key {MASK}"


def test_redaction_keeps_ordinary_text_and_numbers():
    """ตัวปิดต้องไม่กินของที่ไม่ใช่ความลับ — คำตอบที่ถูกปิดจนอ่านไม่ออกก็ใช้ไม่ได้เท่ากับคำตอบที่รั่ว"""
    from lmds.mcp.redaction import scrub

    plain = {
        "kv_bytes_per_token": 147456, "max_tokens": 8192, "prompt_tokens_24h": 3400, "token_source": "environment",
        "detail": "HF token: required for gated repos · ยังไม่ได้ตั้ง token", "fix": "lmds config set-hf-token",
        "image": "ghcr.io/ggml-org/llama.cpp@sha256:ccfd96bb2aba4ef77e3df656d713ce85bf3c6a886d7974243127c5ad66ca6d2a",
        "commit": "6b0d2201f3a94c0e8d7b6a5c4d3e2f1a0b9c8d7e", "args": "--max-num-seqs 4 --tokenizer-mode mistral",
        "note": "kv_bytes_per_token=141312 prompt_tokens_total 123456789",
    }
    assert scrub(plain, []) == plain
    assert scrub({"api_key": "0123456789abcdef", "x": ["API_KEY=0123456789abcdef"]}, []) == \
        {"api_key": MASK, "x": [f"API_KEY={MASK}"]}
    assert scrub("the key is 0123feedbeef4567", ["0123feedbeef4567"]) == f"the key is {MASK}"


# ── SSH ที่ค้าง ───────────────────────────────────────────────────────────────────────────────────
def test_a_node_whose_ssh_hangs_becomes_that_nodes_error_and_the_rest_is_reported(fleet):
    """เครื่องที่ต่อติดแล้วเงียบ (Tailscale relay หลุดกลางทาง) — timeout 30 วิของ probe ตัวเดิมทำงาน เครื่องอื่นได้ผลครบ"""
    from lmds.nodes import Node
    from lmds.nodes.registry import load, save

    save([*load(), Node(name=NODE_HUNG, host=HOSTS[NODE_HUNG], user="tkc", site="chiangmai")])
    fleet.set_node(NODE_HUNG, "hung")
    registry = (fleet.config / "nodes.yaml").read_bytes()
    client = fleet.mcp()
    client.initialize()

    started = time.monotonic()
    answer = client.call("lmds_fleet_check", {"check": True}, timeout=120)
    elapsed = time.monotonic() - started

    assert answer["error"] is False, "เครื่องเดียวค้างต้องไม่ทำให้ทั้งคำขอล้ม"
    assert 25 < elapsed < 90, f"timeout ของ probe คือ 30 วิ (ใช้ไป {elapsed:.0f} วิ)"
    nodes = {node["name"]: node for node in answer["payload"]["nodes"]}
    assert set(nodes) == {NODE_OK, NODE_DOWN, NODE_HUNG}
    assert nodes[NODE_OK]["reachable"] is True and nodes[NODE_OK]["source"] == "cache"
    assert nodes[NODE_OK]["code"]["state"] and nodes[NODE_OK]["controllers"]["state"] == "stale"
    assert nodes[NODE_HUNG]["reachable"] is False and "30" in nodes[NODE_HUNG]["error"]
    assert nodes[NODE_DOWN]["reachable"] is False and "Connection timed out" in nodes[NODE_DOWN]["error"]
    assert answer["payload"]["summary"]["total"] == 3
    assert (fleet.config / "nodes.yaml").read_bytes() == registry

    # server ยังใช้ได้ต่อ และถามเครื่องเดียวที่ค้างก็ได้ error ของเครื่องนั้น
    assert client.call("lmds_version")["error"] is False
