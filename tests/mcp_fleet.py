"""ฟลีตจำลองสำหรับเทสของ `lmds mcp` (tests/test_mcp_*.py) — hub หนึ่งเครื่องกับ node ปลอมหลัง `ssh` ปลอม

หลักเดียวกับ harness อื่นของ repo นี้: **รันของจริงทั้งเส้น** ของปลอมมีเฉพาะขอบที่แตะโลกภายนอก

  * hub = โฟลเดอร์ชั่วคราวที่เป็นทั้ง HOME · config dir · run root · ที่เก็บ key · ที่เก็บสถานะ watchdog ของ process ลูก
    ข้างในเป็นของที่สร้างด้วยโค้ดจริง: bundle ที่ render จริงสองใบ (vLLM แบบ docker ที่ยังไม่เคย start · llama.cpp แบบ
    native ที่ "รันอยู่") · ทะเบียนเครื่องที่เขียนด้วย `nodes.registry.save` · key ที่เขียนด้วย `apikey.write` ·
    watchdog ที่เขียนด้วย `watchdog.save` · ทะเบียนผี (`ghost`) ที่ `fleet.discover()` ของ CLI จะลบทิ้ง
  * `ssh` ปลอมบน PATH = เครื่องปลายทาง: จดทุกคำสั่งที่ hub ส่งมา (ssh.jsonl · remote.jsonl) แล้วตอบตามโหมดของ host นั้น
    — ok (ตอบ JSON ที่เตรียมไว้ของคำสั่งอ่านแต่ละตัว) · down (ข้อความ "Connection timed out" ของ ssh, exit 255) ·
    hung (ต่อติดแล้วเงียบ — ให้ timeout ของ hub ทำงานจริง) · คำสั่งที่ไม่รู้จัก = exit 127 และยังถูกจดไว้ให้เทสเห็น
  * `docker` · `systemctl` · `loginctl` · `pgrep` ปลอม = จดว่าถูกเรียกด้วยอะไร แล้วตอบแบบเครื่องที่ไม่มีของนั้น
    (เครื่อง dev มี docker/OrbStack จริง — เทสต้องไม่ไปถามมัน)
  * engine ปลอม = HTTP server ใน process ของเทส ตอบ `/health` กับ `/metrics` ให้ bundle llama.cpp ที่ "รันอยู่"
  * Hugging Face ปลอม = httpx.MockTransport แบบเดียวกับ tests/test_mlx_checkpoint.py — process ลูกเปิดผ่าน
    `python -m tests.mcp_fleet --fake-hub <คำสั่ง lmds>` ซึ่งสลับ `lmds.inspector.HfClient` ก่อนเรียก CLI ตัวจริง

ค่าที่หน้าตาเป็น key ทุกตัวในไฟล์นี้เป็นของปลอมที่ปลูกไว้ให้เทส redaction หา (`PLANTED`) — สั้นและเห็นชัดว่าไม่ใช่ของจริง
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKER_SLUG = "qwen3-32b"                 # vLLM (docker) — เคยรัน ตอนนี้หยุดอยู่
NATIVE_SLUG = "qwen3-8b-gguf"             # llama.cpp (native) — "รันอยู่" (pid จริง + engine ปลอมตอบ /health)
GHOST_SLUG = "ghost"                      # ทะเบียนของ bundle ที่ถูกลบไปแล้ว — discover() ของ CLI เก็บกวาดทิ้ง
NODE_OK = "spark-head-rack-a-unit-07-ชั้นบน"     # ชื่อยาว + ไทย: ตารางของ `lmds node list` ตัดชื่อแบบนี้จนอ่านไม่ออก
NODE_DOWN = "msi-down"
NODE_HUNG = "msi-hung"
HOSTS = {NODE_OK: "10.9.0.1", NODE_DOWN: "10.9.0.9", NODE_HUNG: "10.9.0.7"}

# ── ค่าปลอมที่ปลูกไว้ — ห้ามโผล่ในคำตอบของเครื่องมือไหนเลย ──
PLANTED = {
    "bundle_key": "0123feedbeef4567plantedkey89abcd",          # ~/.lmds/keys/<slug> ของ hub (hub รู้ค่า)
    "web_token": "webtokenPLANTED0001fake",                    # config/web-token
    "hf_token": "hf_PLANTEDfakeTOKEN0001",                     # env HF_TOKEN ของ process
    "args_key": "argsPLANTEDkey0001fake",                      # bundle.args: --api-key …
    "log_key": "logPLANTEDkey0001fake",                        # log ของ vLLM: 'api_key': ['…']
    "node_key": "nodePLANTEDkey0001fake",                      # log บน node (hub ไม่รู้ค่า)
    "node_bearer": "nodePLANTEDbearer0001fake",                # Authorization: Bearer … ใน log ของ node
    "node_args_key": "nodePLANTEDargs0002fake",                # extra_args ใน agent info ของ node
    "proxy_password": "proxyPLANTEDpw0001",                    # http://user:…@proxy ใน log
    "provider_key": "providerPLANTEDkey0001fake",              # ไฟล์ credentials ของ hub (key ของ LLM provider)
}

MUTATING_LMDS_VERBS = {
    "start", "stop", "restart", "remove", "deploy", "generate", "rebuild", "push", "clone", "install", "set",
    "adopt", "refresh", "arm", "disarm", "enable", "disable", "repair", "prune", "download", "setup", "run",
    "ctl", "apply", "write", "pair", "add", "smoke", "burn", "new", "clear",
}
MUTATING_DOCKER_VERBS = {"run", "rm", "stop", "start", "restart", "kill", "pull", "exec", "create", "cp", "rmi", "update"}
MUTATING_SYSTEMCTL_VERBS = {"start", "stop", "restart", "enable", "disable", "daemon-reload", "kill", "mask", "edit"}

VLLM_LOG = textwrap.dedent(f"""\
    INFO 10-09 09:00:01 [api_server.py:1234] vLLM API server version 0.9.2
    INFO 10-09 09:00:01 [api_server.py:1240] non-default args: {{'port': 8000, 'api_key': ['{PLANTED["log_key"]}'], 'max_model_len': 40960}}
    INFO 10-09 09:00:02 [launcher.py:28] HTTPS_PROXY=http://deploy:{PLANTED["proxy_password"]}@proxy.internal:3128
    INFO 10-09 09:00:41 [gpu_model_runner.py:1801] Model loading took 61.20 GiB and 38.1 seconds
    WARNING 10-09 09:02:13 [protocol.py:99] rejected key {PLANTED["bundle_key"]} for /v1/chat/completions
    INFO 10-09 09:02:14 [metrics.py:417] Avg prompt throughput: 0.0 tokens/s, Running: 0 reqs
    """)
NATIVE_LOG = textwrap.dedent(f"""\
    main: server is listening on http://0.0.0.0:8101
    srv  launch_slot_: id  0 | task 1 | processing task
    srv  log_server_r: request: GET /v1/models 127.0.0.1 200 --api-key {PLANTED["args_key"]}
    srv  launch_slot_: id  0 | task 2 | processing task
    """)
NODE_LOG = textwrap.dedent(f"""\
    INFO vllm serve started with --api-key {PLANTED["node_key"]} --port 8000
    INFO curl -H "Authorization: Bearer {PLANTED["node_bearer"]}" http://127.0.0.1:8000/v1/models
    INFO snapshot_download resumed with token {PLANTED["hf_token"]}
    INFO console session {PLANTED["web_token"]} opened the log panel
    WARN upstream planner rejected {PLANTED["provider_key"]} (401)
    ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
    """)
NODE_FIT = {"slug": "gemma-3-27b", "engine": "vllm", "context": 32768, "slots": 2, "weights_gb": 51.2,
            "kv_per_request_gb": 3.9, "ram_needed_gb": 63.5, "total_gb": 119.0, "usable_gb": 111.0, "ram_after_gb": 47.5,
            "fits": True, "verdict": "fits", "reason": "พอ — เหลือ 47.5 GB", "settings": {"slots": "2", "context": "32768"},
            "current": {"slots": "1", "extra_args": f"--api-key {PLANTED['node_args_key']}"}, "notes": []}
NODE_DOCTOR = {"slug": "gemma-3-27b", "healthy": False, "skipped": ["architecture"], "findings": [
    {"name": "controller", "status": "ok", "detail": "gemma-3-27b-single.sh", "fix": ""},
    {"name": "weights", "status": "fail", "detail": "ขาด 3 จาก 12 shard", "fix": "lmds repair gemma-3-27b"}]}
NODE_WATCHDOG = [{"slug": "gemma-3-27b", "armed": True, "armed_at": "2026-10-01 08:00", "never_probed": False,
                  "probe_overdue": False, "paused": False, "gave_up": False, "restarts": [],
                  "service": {"unit": "lmds-watchdog-gemma-3-27b.service", "installed": True, "state": "active"}}]


def node_agent_info() -> dict:
    """สิ่งที่ `lmds agent info` ของ node ปลอมพิมพ์ — รูปเดียวกับ inventory.snapshot() เท่าที่ hub อ่าน"""
    model = {
        "slug": "gemma-3-27b", "model_id": "google/gemma-3-27b-it", "engine": "vllm", "mode": "docker", "port": 8000,
        "running": True, "healthy": True, "registered": True, "external": False, "controller_exists": True,
        "endpoint": "http://127.0.0.1:8000/v1", "context": 32768, "slots": 1, "context_per_request": 32768,
        "autostart": "enabled", "downloaded": True, "features": "tools · vision", "commands": ["start", "stop", "logs"],
        "pending_restart": {"pending": True, "changes": [
            {"field": "extra_args", "saved": f"--api-key {PLANTED['node_args_key']}", "running": "(ไม่อยู่บน argv)"}]},
        "controller": {"state": "stale", "detail": "เก่ากว่า lmds บนเครื่อง"}, "runtime_arch": None,
        "generated_by": "lmds 0.9.4", "template_hash": "0ldhash00000", "usage": {"requests_24h": 412, "source": "docker-log"},
    }
    return {
        "host": {"hostname": "spark-head", "ip": "10.9.0.1", "lmds_version": "0.9.4", "lmds_commit": "0ad1a59e",
                 "template_hash": "0ldhash00000", "memory_model": "unified", "ram_total_gb": 119.0, "ram_used_gb": 71.5,
                 "gpus": [{"name": "NVIDIA GB10", "vram_gb": 119.0}], "role": {"control_plane": False},
                 "runtimes": {"llamacpp": []}, "foreign": []},
        "models": [model],
        "summary": {"total": 1, "running": 1, "healthy": 1, "not_downloaded": 0, "controllers_stale": 1,
                    "runtime_stale": 0, "restart_pending": 1},
    }


# ── โปรแกรมปลอมบน PATH ──────────────────────────────────────────────────────────────────────────
_FAKE_SSH = '''\
#!{python}
import json, os, re, shlex, sys, time
box = {box!r}
argv = sys.argv[1:]
with open(os.path.join(box, "log", "ssh.jsonl"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv, ensure_ascii=False) + "\\n")
if "-G" in argv:                                    # ssh -G: พิมพ์ config ที่จะใช้ ไม่ต่อไปไหน
    print("hostname", argv[-1].split("@")[-1])
    sys.exit(0)
host, wrapped = argv[-2].split("@")[-1], argv[-1]
try:
    parts = shlex.split(wrapped)
except ValueError:
    parts = []
command = parts[2] if len(parts) == 3 and parts[:2] == ["bash", "-lc"] else wrapped
with open(os.path.join(box, "log", "remote.jsonl"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"host": host, "command": command}}, ensure_ascii=False) + "\\n")
with open(os.path.join(box, "remote", host + ".json"), encoding="utf-8") as handle:
    node = json.load(handle)
if node["mode"] == "down":
    sys.stderr.write("ssh: connect to host %s port 22: Connection timed out\\n" % host)
    sys.exit(255)
if node["mode"] == "hung":
    time.sleep(90)
for pattern, answer in node["answers"]:
    if re.fullmatch(pattern, command):
        sys.stdout.write(answer.get("stdout", ""))
        sys.stderr.write(answer.get("stderr", ""))
        sys.exit(answer.get("exit", 0))
sys.stderr.write("bash: line 1: unexpected command on fake node: %s\\n" % command)
sys.exit(127)
'''

_FAKE_DOCKER = '''\
#!{python}
import json, os, sys
box = {box!r}
argv = sys.argv[1:]
with open(os.path.join(box, "log", "docker.jsonl"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv, ensure_ascii=False) + "\\n")
if argv[:1] == ["logs"]:
    path = os.path.join(box, "docker", argv[-1] + ".log")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines(keepends=True)
        if "--tail" in argv:
            lines = lines[-int(argv[argv.index("--tail") + 1]):]
        sys.stdout.write("".join(lines))
        sys.exit(0)
    sys.stderr.write("Error response from daemon: No such container: %s\\n" % argv[-1])
    sys.exit(1)
if argv[:1] in (["ps"], ["images"]):
    sys.exit(0)
sys.exit(1)                                         # inspect · info · image inspect: ไม่มีของนั้นบนเครื่องนี้
'''

_FAKE_QUIET = '''\
#!{python}
import json, os, sys
box = {box!r}
with open(os.path.join(box, "log", {name!r} + ".jsonl"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")
sys.stdout.write({stdout!r})
sys.exit({code})
'''


class _Engine(BaseHTTPRequestHandler):
    """llama-server ปลอม: /health พร้อม · /metrics มีตัวนับ (ทำให้ inventory พยายามเขียน usage.samples) · /v1/* ต้องมี key"""

    def do_GET(self):  # noqa: N802 — ชื่อตาม http.server
        if self.path == "/health":
            body, status, kind = b'{"status":"ok"}', 200, "application/json"
        elif self.path == "/metrics":
            body = (b"llamacpp:requests_total 12\nllamacpp:prompt_tokens_total 3400\n"
                    b"llamacpp:tokens_predicted_total 910\n")
            status, kind = 200, "text/plain"
        else:
            body, status, kind = b'{"error":{"message":"Invalid API Key"}}', 401, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


# ── ฟลีต ─────────────────────────────────────────────────────────────────────────────────────────
class Fleet:
    """สร้างด้วย `build(tmp_path, monkeypatch)` — env ของ process เทสถูกชี้มาที่กล่องนี้ระหว่างสร้าง แล้วส่งต่อให้ลูก"""

    def __init__(self, box: Path) -> None:
        self.box = box
        self.home = box / "home"
        self.config = self.home / ".config" / "lmds"
        self.run = self.home / ".lmds" / "run"
        self.keys = self.home / ".lmds" / "keys"
        self.watchdogs = self.home / ".lmds" / "watchdog"
        self.bundles = self.home / "bundles"
        self.bin = box / "bin"
        self.log = box / "log"
        self.sleeper: subprocess.Popen | None = None
        self.engine: ThreadingHTTPServer | None = None
        self.clients: list[Mcp] = []

    # env ของ process ลูก (CLI/MCP ตัวจริง)
    @property
    def env(self) -> dict:
        path = [str(self.bin)]
        path += [d for d in ("/opt/homebrew/bin", "/usr/local/bin") if Path(d, "bash").is_file()]   # bash 5 บน macOS
        path += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        keep = {k: v for k, v in os.environ.items() if k in ("LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")}
        return {
            **keep,
            "HOME": str(self.home), "LMDS_CONFIG_DIR": str(self.config), "LMDS_RUN_ROOT": str(self.run),
            "LMDS_KEY_ROOT": str(self.keys), "LMDS_WATCHDOG_ROOT": str(self.watchdogs),
            "PATH": os.pathsep.join(path), "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8",
            "HF_TOKEN": PLANTED["hf_token"], "LMDS_ROLE": "serving", "LMDS_SKIP_REGISTRY_CHECK": "1",
            "COLUMNS": "200", "NO_COLOR": "1",
        }

    def lmds(self, *args: str, fake_hub: bool = False, timeout: int = 180, env: dict | None = None):
        """รัน CLI ตัวจริงใน process ลูกบนฟลีตนี้ — คืน CompletedProcess (stdout/stderr แยกกัน)"""
        return subprocess.run(self.argv(*args, fake_hub=fake_hub), cwd=self.home, env={**self.env, **(env or {})},
                              capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                              stdin=subprocess.DEVNULL)

    def argv(self, *args: str, fake_hub: bool = False) -> list[str]:
        return [sys.executable, "-m", "tests.mcp_fleet", *(["--fake-hub"] if fake_hub else []), *args]

    def mcp(self, fake_hub: bool = False, env: dict | None = None, argv: list[str] | None = None, stderr=None) -> "Mcp":
        client = Mcp(argv or self.argv("mcp", fake_hub=fake_hub), cwd=self.home, env={**self.env, **(env or {})},
                     stderr=stderr)
        self.clients.append(client)
        return client

    # บันทึกของโปรแกรมปลอม
    def calls(self, program: str) -> list:
        path = self.log / f"{program}.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def remote_commands(self) -> list[str]:
        return [entry["command"] for entry in self.calls("remote")]

    def set_node(self, name: str, mode: str, answers: list | None = None) -> None:
        (self.box / "remote").mkdir(exist_ok=True)
        (self.box / "remote" / f"{HOSTS[name]}.json").write_text(
            json.dumps({"mode": mode, "answers": answers or []}, ensure_ascii=False), encoding="utf-8")

    def state(self) -> dict[str, str]:
        """ลายนิ้วมือของทุกไฟล์ที่ hub ถือ (ทะเบียน · config · run · bundle · key · watchdog) — เทียบก่อน/หลังเรียกเครื่องมือ"""
        out: dict[str, str] = {}
        for path in sorted(self.home.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(self.home))
            if path.is_symlink() or not path.is_file():
                out[rel] = "dir" if path.is_dir() else "other"
                continue
            stat = path.stat()
            out[rel] = f"{hashlib.sha256(path.read_bytes()).hexdigest()}:{stat.st_mtime_ns}:{stat.st_mode:o}"
        return out

    def close(self) -> None:
        for client in self.clients:
            client.close()
        if self.sleeper is not None:
            self.sleeper.kill()
            self.sleeper.wait()
        if self.engine is not None:
            self.engine.shutdown()
            self.engine.server_close()


def _fake(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def ok_answers() -> list:
    """คำสั่งอ่านที่ node ปลอมรู้จัก (regex เต็มบรรทัด) → คำตอบ · อย่างอื่นทั้งหมด = exit 127 และอยู่ใน remote.jsonl"""
    slug = "gemma-3-27b"
    return [
        [r"LMDS_READ_ONLY=1 lmds agent info", {"stdout": json.dumps(node_agent_info(), ensure_ascii=False)}],
        [r"lmds agent info", {"stdout": json.dumps(node_agent_info(), ensure_ascii=False)}],      # `--check` ของ CLI
        # rc ของเครื่องจริงพ่นของออก stdout ก่อน JSON ได้ (เคส dgx-70) — node ปลอมทำแบบนั้นกับ fit
        [rf"LMDS_READ_ONLY=1 lmds fit {slug} --json( --slots \d+)?( --context \d+)?",
         {"stdout": "declare -x LANG=\"C.UTF-8\"\n" + json.dumps(NODE_FIT, ensure_ascii=False)}],
        [rf"LMDS_READ_ONLY=1 lmds logs {slug} -n \d+", {"stdout": NODE_LOG}],
        [rf"LMDS_READ_ONLY=1 lmds doctor {slug} --json --no-probe",
         {"stdout": json.dumps(NODE_DOCTOR, ensure_ascii=False), "exit": 2}],
        [rf"LMDS_READ_ONLY=1 lmds watchdog status( {slug})? --json",
         {"stdout": json.dumps(NODE_WATCHDOG, ensure_ascii=False)}],
    ]


def build(tmp_path: Path, monkeypatch) -> Fleet:
    """สร้างฟลีตด้วยโค้ดจริงของ LMDS — ต้องเรียกใต้ monkeypatch เพราะย้าย HOME/config/run ของ process เทสมาที่กล่อง"""
    fleet = Fleet(tmp_path / "fleet")
    for directory in (fleet.home, fleet.bin, fleet.log, fleet.box / "docker", fleet.box / "remote"):
        directory.mkdir(parents=True, exist_ok=True)
    for key, value in fleet.env.items():
        if key not in ("PATH", "PYTHONPATH", "HF_TOKEN"):
            monkeypatch.setenv(key, value)
    python = sys.executable
    _fake(fleet.bin / "ssh", _FAKE_SSH.format(python=python, box=str(fleet.box)))
    _fake(fleet.bin / "docker", _FAKE_DOCKER.format(python=python, box=str(fleet.box)))
    for name, stdout, code in (("systemctl", "inactive\n", 3), ("loginctl", "Linger=no\n", 0), ("pgrep", "", 1)):
        _fake(fleet.bin / name, _FAKE_QUIET.format(python=python, box=str(fleet.box), name=name, stdout=stdout, code=code))

    # ── bundle สองใบ render ด้วย renderer ตัวจริง ──
    from tests import single_controller_harness as harness

    from lmds.fleet import apikey, register_bundle, watchdog
    from lmds.fleet.bundle_settings import ARGS_FILENAME

    docker_bundle = harness.render(fleet.home, "vllm")
    native_bundle = harness.render(fleet.home, "llamacpp")
    assert docker_bundle.directory.name == DOCKER_SLUG and native_bundle.directory.name == NATIVE_SLUG
    meta = register_bundle(docker_bundle.controller)
    # เคยรันมาแล้วและตอนนี้หยุดอยู่ — `lmds logs` ของ CLI ไม่ยอมอ่าน log ของ bundle ที่ไม่เคย start เลย
    meta.write_text(meta.read_text(encoding="utf-8").replace("started_at=\n", "started_at=2026-10-08T20:00:00\n"),
                    encoding="utf-8")
    (fleet.box / "docker" / f"lmds-{DOCKER_SLUG}.log").write_text(VLLM_LOG, encoding="utf-8")
    (docker_bundle.directory / ARGS_FILENAME).write_text(f"--api-key {PLANTED['args_key']}\n", encoding="utf-8")
    apikey.write(DOCKER_SLUG, PLANTED["bundle_key"])

    # llama.cpp ที่ "รันอยู่": pid จริง (sleep) + engine ปลอมที่ตอบ /health และ /metrics
    fleet.engine = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=fleet.engine.serve_forever, daemon=True).start()
    port = fleet.engine.server_address[1]
    fleet.sleeper = subprocess.Popen(["sleep", "900"], stdin=subprocess.DEVNULL)
    run_dir = fleet.run / NATIVE_SLUG
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.pid").write_text(str(fleet.sleeper.pid), encoding="utf-8")
    (run_dir / "server.log").write_text(NATIVE_LOG, encoding="utf-8")
    (run_dir / "server.meta").write_text(
        f"slug={NATIVE_SLUG}\nmodel={NATIVE_SLUG}\ndefault_model={NATIVE_SLUG}\nmodel_id=unsloth/Qwen3-8B-GGUF\n"
        f"engine=llamacpp\nmode=native\nport={port}\ncontainer=\npid_file={run_dir / 'server.pid'}\n"
        f"controller={native_bundle.controller}\nstarted_at=2026-10-09T08:00:00\n", encoding="utf-8")

    # ทะเบียนผี: ไม่เคย start และ controller ไม่อยู่แล้ว — `fleet.discover()` ลบไฟล์นี้ทุกครั้งที่ CLI/หน้าเว็บถาม
    ghost = fleet.run / GHOST_SLUG
    ghost.mkdir(parents=True)
    (ghost / "server.meta").write_text(
        f"slug={GHOST_SLUG}\nmodel={GHOST_SLUG}\nmodel_id=org/{GHOST_SLUG}\nengine=vllm\nmode=docker\nport=8999\n"
        f"container=lmds-{GHOST_SLUG}\npid_file=\ncontroller={fleet.home / 'gone' / 'ghost-single.sh'}\nstarted_at=\n",
        encoding="utf-8")

    # watchdog ของตัวที่รันอยู่ — เขียนด้วย save() ตัวจริง
    watchdog.save(watchdog.State(slug=NATIVE_SLUG, armed=True, armed_at="2026-10-08 21:00", armed_by="owner",
                                 last_probe_at=1_780_000_000.0, last_ok_at=1_780_000_000.0, last_detail="ok"))

    # ── ทะเบียนเครื่อง + token ของหน้าเว็บ ──
    from lmds.nodes import Node
    from lmds.nodes.registry import save

    save([
        Node(name=NODE_OK, host=HOSTS[NODE_OK], user="tkc", site="bangkok-dc1", note="head ของคู่ stacked",
             lmds_version="0.9.4", lmds_commit="0ad1a59e", last_seen="2026-10-01 08:15", local_ip="10.9.0.1",
             controllers_stale=1, controllers_unknown=0, runtime_stale=0, runtime_unknown=0, restart_pending=1,
             llamacpp_build="10495 · 2026-08-18", alt_hosts=["100.64.0.1"]),
        Node(name=NODE_DOWN, host=HOSTS[NODE_DOWN], user="tkc", site="chiangmai", lmds_version="0.10.0",
             lmds_commit="6b0d220", last_seen="2026-10-02 11:00", last_error="",
             controllers_stale=0, controllers_unknown=0, runtime_stale=0, runtime_unknown=0),
    ])
    (fleet.config / "web-token").write_text(PLANTED["web_token"] + "\n", encoding="utf-8")
    (fleet.config / "web-token").chmod(0o600)
    (fleet.config / "credentials").write_text(f"openai={PLANTED['provider_key']}\n", encoding="utf-8")
    (fleet.config / "credentials").chmod(0o600)

    fleet.set_node(NODE_OK, "ok", ok_answers())
    fleet.set_node(NODE_DOWN, "down")
    return fleet


# ── client ของ MCP ทาง stdio ─────────────────────────────────────────────────────────────────────
class Mcp:
    """คุยกับ `lmds mcp` ตัวจริงทาง stdin/stdout ด้วย JSON-RPC จริง — เก็บทุกบรรทัดของ stdout ไว้ให้เทสตรวจ"""

    def __init__(self, argv: list[str], cwd: Path, env: dict, stderr=None) -> None:
        self._own_stderr = stderr is None
        self._stderr_file = None
        if self._own_stderr:
            self._stderr_file = open(Path(cwd).parent / f"mcp-stderr-{os.getpid()}-{id(self)}.log", "w+b")  # noqa: SIM115
            stderr = self._stderr_file
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        self.stdout_lines: list[bytes] = []
        self._inbox: queue.Queue = queue.Queue()
        self._pending: dict = {}
        self._next_id = 0
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for raw in self.proc.stdout:
            self.stdout_lines.append(raw)
            self._inbox.put(raw)
        self._inbox.put(None)

    def send_raw(self, line: str) -> None:
        self.last_sent = line
        self.proc.stdin.write(line.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        self.send_raw(json.dumps({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})}))

    def next_message(self, timeout: float = 60):
        try:
            raw = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"ไม่มีคำตอบใน {timeout:g} วิ หลังส่ง: {self.last_sent[:300]}\n"
                                 f"stderr:\n{self.stderr_text()[-3000:]}") from None
        if raw is None:
            raise AssertionError(f"server ปิด stdout ก่อนตอบ — stderr:\n{self.stderr_text()[-3000:]}")
        return json.loads(raw)

    def request(self, method: str, params=None, timeout: float = 60, request_id=None) -> dict:
        """ส่งคำขอแล้วรอคำตอบของ id นั้น (คำตอบของคำขออื่นที่มาก่อนถูกเก็บไว้ให้คนที่รอมัน)"""
        if request_id is None:
            self._next_id += 1
            request_id = self._next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send_raw(json.dumps(message, ensure_ascii=False))
        return self.wait_for(request_id, timeout)

    def wait_for(self, request_id, timeout: float = 60) -> dict:
        while request_id not in self._pending:
            message = self.next_message(timeout)
            self._pending[message.get("id")] = message
        return self._pending.pop(request_id)

    def initialize(self, version: str = "2025-06-18") -> dict:
        reply = self.request("initialize", {"protocolVersion": version, "capabilities": {},
                                            "clientInfo": {"name": "pytest", "version": "0"}})
        self.notify("notifications/initialized")
        return reply

    def call(self, name: str, arguments: dict | None = None, timeout: float = 120) -> dict:
        """เรียกเครื่องมือ → {"payload": JSON ก้อนแรก, "notes": [...], "error": bool, "text": ทุกก้อนต่อกัน, "raw": คำตอบดิบ}"""
        reply = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        assert "result" in reply, f"{name} ตอบเป็น JSON-RPC error: {reply}"
        content = reply["result"]["content"]
        assert content and all(item["type"] == "text" for item in content)
        notes = json.loads(content[1]["text"])["notes"] if len(content) > 1 else []
        return {"payload": json.loads(content[0]["text"]), "notes": notes, "error": reply["result"]["isError"],
                "text": "\n".join(item["text"] for item in content), "raw": reply}

    def stderr_text(self) -> str:
        if self._stderr_file is None:
            return ""
        if not self._stderr_file.closed:
            self._stderr_file.flush()
        with open(self._stderr_file.name, "rb") as handle:
            return handle.read().decode("utf-8", "replace")

    def close(self, timeout: float = 30) -> int:
        """ปิด stdin (วิธีที่ client จริงบอกเลิก) แล้วรอ process จบ — คืน exit code"""
        if self._closed:
            return self.proc.returncode
        self._closed = True
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self._stderr_file is not None:
            self._stderr_file.close()
        return self.proc.returncode


# ── Hugging Face ปลอม (แบบเดียวกับ tests/test_mlx_checkpoint.FakeHub) ───────────────────────────────
PLAIN_REPO = "Qwen/Qwen3-32B"
PLAIN_SHA = "abc123def4567890abc123def4567890abc123de"
MISSING_REPO = "nobody/no-such-model"


def _plain_repo() -> tuple[dict, dict]:
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    info = {
        "id": PLAIN_REPO, "sha": PLAIN_SHA, "gated": False, "private": False, "tags": ["safetensors", "text-generation"],
        "cardData": {"license": "apache-2.0"}, "pipeline_tag": "text-generation", "library_name": "transformers",
        "safetensors": {"total": 32_800_000_000},
        "siblings": [*({"rfilename": name, "size": 32_500_000_000, "lfs": {"size": 32_500_000_000}} for name in shards),
                     {"rfilename": "config.json", "size": 700}, {"rfilename": "tokenizer_config.json", "size": 900},
                     {"rfilename": "model.safetensors.index.json", "size": 300}],
    }
    files = {
        "model.safetensors.index.json": json.dumps({"metadata": {"total_size": 65_000_000_000},
                                                    "weight_map": {"a": shards[0], "b": shards[1]}}),
        "config.json": json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                                   "max_position_embeddings": 40960, "num_hidden_layers": 64, "hidden_size": 5120,
                                   "num_attention_heads": 64, "num_key_value_heads": 8, "torch_dtype": "bfloat16"}),
        "tokenizer_config.json": json.dumps({"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}),
    }
    return info, files


def install_fake_hub() -> None:
    """สลับ `lmds.inspector.HfClient` ให้ชี้ Hub ปลอม — CLI/MCP ตัวจริงใน process นี้เดิน inspect ทั้งเส้นกับ metadata นี้"""
    import httpx

    import lmds.inspector
    from lmds.inspector import HfClient
    from tests import test_mlx_checkpoint as mlx

    plain_info, plain_files = _plain_repo()
    repos = {PLAIN_REPO: (plain_info, plain_files, PLAIN_SHA), mlx.REPO: (mlx.MODEL_INFO, mlx.FILES, mlx.SHA)}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for repo, (info, files, sha) in repos.items():
            if path == f"/api/models/{repo}" or path.startswith(f"/api/models/{repo}/"):
                return httpx.Response(200, json=info)
            if path.startswith(f"/{repo}/resolve/"):
                name = path.split(f"/{sha}/")[-1]
                return httpx.Response(200, content=files[name].encode()) if name in files else httpx.Response(404)
        return httpx.Response(401)                    # repo ที่ไม่มีอยู่ + ไม่ล็อกอิน = 401 เปล่า ๆ (วัดจาก Hub จริง)

    def client(token=None) -> HfClient:
        return HfClient(token=token, client=httpx.Client(transport=httpx.MockTransport(handler)))

    lmds.inspector.HfClient = client


def main(argv: list[str]) -> None:
    """`python -m tests.mcp_fleet [--fake-hub] <คำสั่ง lmds…>` — CLI ตัวจริง (รวม `lmds mcp`) กับ Hub ปลอมถ้าขอ"""
    if argv[:1] == ["--fake-hub"]:
        install_fake_hub()
        argv = argv[1:]
    from lmds.cli.main import app

    app(args=argv, prog_name="lmds")


if __name__ == "__main__":
    main(sys.argv[1:])
