"""เครื่องมือรัน single controller ทั้งสาม (vLLM · SGLang · llama.cpp) แบบ "ลูกค้ารัน" — ใช้ร่วมกันใน test_audit3_*

หลักของ harness นี้: **รันทั้งสคริปต์ที่ render แล้วจริง ๆ** ไม่ดึงฟังก์ชันออกมาแปะ stub · ของปลอมมีสองชั้นเท่านั้น

  * `docker` ปลอม — ทำตัวเหมือน docker เท่าที่ controller ใช้: `run -d` **รัน entrypoint จริง** (จาก PATH) ด้วย argv ที่ได้
    และ env เฉพาะที่ส่งผ่าน `-e` (ชื่อเปล่า = หยิบจาก env ของ docker เอง · NAME=VALUE = ตามนั้น) → สิ่งที่ engine
    "ได้รับจริง" จึงเป็นผลของ argv/env ที่ controller ประกอบ ไม่ใช่สิ่งที่เทสเชื่อเอาเอง · `ps`/`rm -f`/`inspect` ตอบจาก
    process ตัวนั้น
  * engine ปลอม (`vllm` · `sglang` · `llama-server`) — HTTP server จริงที่ **ฟังตาม --host/--port ที่ถูกสั่ง** เสิร์ฟ
    `/health` `/v1/models` `/v1/chat/completions` ตามชื่อโมเดลที่ถูกสั่ง และบังคับ API key ตามช่องทางที่ engine จริงรับ
    (vLLM = env VLLM_API_KEY · SGLang = --api-key · llama-server = --api-key-file) — key ที่ไปผิดช่อง = เซิร์ฟเวอร์เปิดโล่ง
    ซึ่งเทสจับได้ด้วยการยิงโดยไม่ใส่ key (บทเรียน LLAMA_ARG_API_KEY)

controller เล็ง Linux bash 5 — บน macOS ใช้ bash ของ homebrew ไม่ใช่ /bin/bash 3.2
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport, ShardFile

BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if Path(p).is_file()), "bash")
# /sbin กับ /usr/sbin: macOS วาง sha256sum/md5 ไว้ที่นั่น (ดู test_audit_cli_controllers.SAFE_PATH)
SAFE_PATH = "/usr/bin:/bin:/sbin:/usr/sbin"
KINDS = ["vllm", "sglang", "llamacpp"]
SHARDS = [("model-00001-of-00002.safetensors", 12), ("model-00002-of-00002.safetensors", 7)]
GGUF_NAME = "Qwen3-8B-Q4_K_M.gguf"
GGUF_BYTES = b"GGUF" + b"\0" * 8


def st_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned-123", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, shard_count=2, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
        safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in SHARDS],
    )
    base.update(overrides)
    return ModelReport(**base)


def gguf_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf=GGUF_NAME,
        gguf_variants=[GgufVariant(filename=GGUF_NAME, size_bytes=len(GGUF_BYTES), sha256=None)],
        has_chat_template=True, license="apache-2.0",
    )
    base.update(overrides)
    return ModelReport(**base)


def render(tmp_path: Path, kind: str, report: ModelReport | None = None, tweak=None):
    """render bundle ของ engine นั้นลง tmp_path/bundles — คืน Bundle (มี .controller · .directory)"""
    if kind == "llamacpp":
        report, engine = report or gguf_report(), None
    else:
        report, engine = report or st_report(), (Engine.SGLANG if kind == "sglang" else None)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=engine)
    if tweak:
        tweak(plan)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def served_name(bundle) -> str:
    for line in bundle.controller.read_text(encoding="utf-8").splitlines():
        if line.startswith("DEFAULT_SERVED_MODEL_NAME=") or line.startswith("SERVED_MODEL_NAME="):
            value = line.split("=", 1)[1].strip().strip('"')
            if value.startswith("${SERVED_MODEL_NAME:-"):
                value = value[len("${SERVED_MODEL_NAME:-"):-1]
            if not value.startswith("$"):
                return value
    raise AssertionError("หา SERVED_MODEL_NAME ใน controller ไม่เจอ")


def container_name(bundle) -> str:
    return f"lmds-{bundle.directory.name}"


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def specific_ip() -> str:
    """IP ของเครื่องนี้ที่ **ไม่ใช่** 127.0.0.1 — ให้ engine ปลอมผูกแค่ตัวนี้ (loopback จึงไม่มีใครฟัง เหมือน --bind ของจริง)

    Linux: ทั้ง 127/8 เป็น loopback → 127.0.0.2 ใช้ได้เลยแม้ไม่มีเน็ต · macOS: ต้องเป็น IP ของการ์ดจริง
    (127.0.0.2 ไม่ได้ตั้งไว้) · หาไม่ได้ทั้งคู่ = ข้ามพร้อมเหตุผล ไม่ใช่ผ่านเงียบ
    """
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.2", 0))
        return "127.0.0.2"
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))                # ไม่ส่งอะไรออกไปจริง — แค่ถาม kernel ว่าจะออกทาง IP ไหน
            ip = s.getsockname()[0]
        with socket.socket() as s:
            s.bind((ip, 0))
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    pytest.skip("เครื่องนี้ไม่มี IP อื่นนอกจาก 127.0.0.1 ให้ผูก (ไม่มี 127.0.0.2 และไม่มีการ์ดเครือข่ายที่ใช้ได้)")


# ───────────────────────── ของปลอม ─────────────────────────
_ENGINE = r'''
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
prog = os.path.basename(sys.argv[0])
log = os.environ.get("FAKE_LOG", os.devnull)


def note(line):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def flag(name, default=""):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def host_path(path):
    """path ในคอนเทนเนอร์ → path บนเครื่อง ตาม -v ที่ docker ปลอมจดไว้"""
    for pair in filter(None, os.environ.get("FAKE_MOUNTS", "").split(";")):
        src, dst = pair.split("=", 1)
        if path == dst or path.startswith(dst.rstrip("/") + "/"):
            return src + path[len(dst):]
    return path


note("engine[%s] argv: %s" % (prog, " ".join(argv)))
note("engine[%s] env: %s" % (prog, " ".join(sorted(
    "%s=%s" % (k, v) for k, v in os.environ.items()
    if not k.startswith("FAKE_") and k not in ("PATH", "PWD", "SHLVL", "_", "OLDPWD", "__CF_USER_TEXT_ENCODING", "LC_CTYPE")))))
if "--version" in argv or "--help" in argv or "-h" in argv:
    print("fake %s · --metrics --api-key-file" % prog)
    sys.exit(0)

model = flag("--served-model-name") or flag("--alias") or "unnamed"
host, port = flag("--host", "127.0.0.1"), int(flag("--port", "8000"))
# key ตามช่องที่ engine จริงรับ — ผิดช่อง = ไม่มี auth (เซิร์ฟเวอร์เปิดโล่ง) ซึ่งคือสิ่งที่เทสต้องจับให้ได้
key = ""
if prog == "vllm":
    key = os.environ.get("VLLM_API_KEY", "")
elif prog == "sglang":
    # SGLang จริง: ธง --api-key หรือคีย์ api-key ในไฟล์ --config (YAML → argument ภายใน process) · **ไม่มี env ของ server**
    # (SGLANG_API_KEY เป็นของ client ภายนอก) — ของปลอมจึงไม่อ่าน env เด็ดขาด: key ที่ไปผิดช่อง = เซิร์ฟเวอร์เปิดโล่ง
    key = flag("--api-key")
    config = flag("--config")
    if config and os.environ.get("FAKE_NO_CONFIG"):        # SGLang รุ่นก่อนมี --config: argparse ตาย
        sys.stderr.write("usage: sglang serve [-h] --model-path MODEL_PATH ...\nsglang: error: unrecognized arguments: --config %s\n" % config)
        sys.exit(2)
    if config and not key and not os.environ.get("FAKE_IGNORE_CONFIG_KEY"):
        for line in open(host_path(config), encoding="utf-8"):
            if line.startswith("api-key:"):
                key = json.loads(line.split(":", 1)[1].strip())
elif prog == "llama-server":
    key_file = flag("--api-key-file")
    if key_file:
        key = open(host_path(key_file), encoding="utf-8").read().strip()
note("engine[%s] auth: %s" % (prog, "key=" + key if key else "OPEN"))
content = os.environ.get("FAKE_CONTENT", "4")
reasoning = os.environ.get("FAKE_REASONING", "")
raw_chat = os.environ.get("FAKE_CHAT_BODY", "")
models = os.environ.get("FAKE_MODELS", model).split(",")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not key or self.path == "/health":
            return True
        if prog == "llama-server" and self.path in ("/v1/models", "/models"):
            return True                     # llama-server: /v1/models เป็น public endpoint
        return self.headers.get("Authorization", "") == "Bearer " + key

    def do_GET(self):
        note("engine[%s] GET %s" % (prog, self.path))
        if not self._authorized():
            return self._send({"error": {"message": "Unauthorized", "code": 401}}, 401)
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in models if m]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        note("engine[%s] POST %s" % (prog, self.path))
        if not self._authorized():
            return self._send({"error": {"message": "Unauthorized", "code": 401}}, 401)
        if raw_chat:
            return self._send(None, raw=raw_chat.encode())
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        self._send({"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})


bind = "127.0.0.1" if host in ("0.0.0.0", "::", "localhost", "") else host.strip("[]")
server = ThreadingHTTPServer((bind, port), H)
note("engine[%s] listening %s:%d model=%s" % (prog, bind, port, model))
threading.Timer(float(os.environ.get("FAKE_ENGINE_TTL", "90")), lambda: os._exit(0)).start()   # กันค้างถ้าเทสตายกลางทาง
server.serve_forever()
'''

_DOCKER = r'''
state="${FAKE_STATE:?}"; mkdir -p "$state"
echo "docker $*" >> "$FAKE_LOG"
alive() { [[ -f "$state/$1.pid" ]] && kill -0 "$(cat "$state/$1.pid")" 2>/dev/null; }
filter_name() {   # ชื่อจาก --filter name=^X$
  local a; for a in "$@"; do case "$a" in name=*) a="${a#name=}"; a="${a#^}"; echo "${a%\$}"; return ;; esac; done
}
[[ -z "${FAKE_DOCKER_DENIED:-}" ]] || { echo "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock" >&2; exit 1; }
case "${1:-}" in
  ps)
    name="$(filter_name "$@")"
    if [[ -n "$name" ]] && alive "$name"; then
      case " $* " in *" --format "*) echo "$name" ;; *) echo "CONTAINER ID   NAMES   STATUS"; echo "c0ffee   $name   Up 1 minute" ;; esac
    else
      case " $* " in *" --format "*) ;; *) echo "CONTAINER ID   NAMES   STATUS" ;; esac
    fi
    exit 0 ;;
  image|pull) exit 0 ;;
  logs) name="${@: -1}"; [[ -f "$state/$name.out" ]] && cat "$state/$name.out"; exit 0 ;;
  info) echo /var/lib/docker; exit 0 ;;
  inspect|container)
    name="${@: -1}"; alive "$name" && { echo true; exit 0; }; exit 1 ;;
  wait) echo 0; exit 0 ;;
  rm)
    name="${@: -1}"
    if [[ -f "$state/$name.pid" ]]; then
      kill "$(cat "$state/$name.pid")" 2>/dev/null || true; rm -f "$state/$name.pid"; exit 0
    fi
    exit 1 ;;
  run)
    shift
    detach="" name="" entry="" mounts=""; envs=()
    while (( $# )); do
      case "$1" in
        -d) detach=1; shift ;;
        --rm|--no-healthcheck) shift ;;
        --name) name="$2"; shift 2 ;;
        --entrypoint) entry="$2"; shift 2 ;;
        --network|--ipc|--gpus|--user) shift 2 ;;
        -v) mounts+="${2%%:*}=$(cut -d: -f2 <<< "$2");"; shift 2 ;;
        -e) if [[ "$2" == *=* ]]; then envs+=("$2"); elif [[ -n "${!2+x}" ]]; then envs+=("$2=${!2}"); fi; shift 2 ;;
        -*) echo "docker ปลอม: ไม่รู้จัก flag $1" >&2; exit 125 ;;
        *) break ;;
      esac
    done
    image="$1"; shift
    if [[ -z "$detach" ]]; then
      case "$entry" in nvidia-smi) echo "GPU 0: Fake GPU" ;; esac
      exit 0
    fi
    # image ของ llama.cpp ไม่ถูก override entrypoint — ตัว image เองรัน llama-server
    [[ -n "$entry" ]] || entry="llama-server"
    fakes=(); while IFS='=' read -r k _; do [[ "$k" == FAKE_* ]] && fakes+=("$k=${!k}"); done < <(env)
    nohup env -i PATH="$PATH" FAKE_MOUNTS="$mounts" "${fakes[@]}" ${envs[@]+"${envs[@]}"} "$entry" "$@" >> "$state/$name.out" 2>&1 &
    echo $! > "$state/$name.pid"
    echo "c0ffee$!"
    exit 0 ;;
esac
exit 0
'''


def write_exe(path: Path, body: str, shebang: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shebang = shebang or (f"#!{BASH}" if BASH.startswith("/") else "#!/usr/bin/env bash")
    path.write_text(shebang + "\n" + textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)


class Box:
    """โฟลเดอร์ทดลองหนึ่งชุด: HOME ปลอม · bin ของปลอม · สถานะ container · log ของทุกอย่างที่ถูกเรียก"""

    def __init__(self, tmp_path: Path, kind: str, bundle):
        self.tmp, self.kind, self.bundle = tmp_path, kind, bundle
        self.home, self.bin, self.state = tmp_path / "home", tmp_path / "bin", tmp_path / "state"
        self.log = tmp_path / "calls.log"
        for d in (self.home, self.bin, self.state):
            d.mkdir(exist_ok=True)
        self.model = served_name(bundle)
        self.container = container_name(bundle)
        self.run_dir = self.home / ".lmds" / "run" / bundle.directory.name
        self._procs: list[subprocess.Popen] = []
        write_exe(self.bin / "docker", _DOCKER)
        for engine in ("vllm", "sglang", "llama-server"):
            write_exe(self.bin / engine, _ENGINE, shebang=f"#!{sys.executable}")
        for quiet in ("ip", "nvidia-smi"):
            write_exe(self.bin / quiet, "exit 0\n")
        write_exe(self.bin / "ss", "exit 1\n")
        self.seed_weights()

    # ── ไฟล์โมเดลที่ verify-files ต้องเห็น ──
    def seed_weights(self) -> None:
        if self.kind == "llamacpp":
            model_dir = self.home / "models" / self.bundle.directory.name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / GGUF_NAME).write_bytes(GGUF_BYTES)
            return
        snap = self.home / ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/sha-pinned-123"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        for name, size in SHARDS:
            (snap / name).write_bytes(b"x" * size)

    # ── env / รัน controller ──
    def env(self, extra: dict | None = None, path: str | None = None) -> dict:
        env = {
            "PATH": path or f"{self.bin}:{Path(sys.executable).parent}:{SAFE_PATH}",
            "HOME": str(self.home), "FAKE_LOG": str(self.log), "FAKE_STATE": str(self.state),
            "ADVERTISE_IP": "10.9.9.9", "HEALTH_TIMEOUT": "20", "STOP_TIMEOUT": "5",
        }
        if self.kind == "llamacpp":
            env["LLAMA_SERVER"] = str(self.bin / "llama-server")
        env.update(extra or {})
        return env

    def run(self, *args: str, env: dict | None = None, path: str | None = None, timeout: int = 90):
        return subprocess.run([BASH, str(self.bundle.controller), *args], capture_output=True, text=True,
                              env=self.env(env, path), timeout=timeout)

    def calls(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""

    def clear_calls(self) -> None:
        self.log.unlink(missing_ok=True)

    # ── เซิร์ฟเวอร์ที่ "รันอยู่ก่อนแล้ว" โดยไม่ผ่าน start ──
    def serve(self, port: int, model: str | None = None, host: str = "127.0.0.1", ours: bool = True,
              program: str | None = None, **fake_env: str) -> subprocess.Popen:
        """รัน engine ปลอมตรง ๆ · ours=True = ลงทะเบียนเป็น container/process ของ bundle นี้ด้วย
        (docker ปลอมตอบว่ารันอยู่ · native: เขียน server.pid + server.meta อย่างที่ start ทำ)"""
        program = program or {"vllm": "vllm", "sglang": "sglang", "llamacpp": "llama-server"}[self.kind]
        name_flag = "--alias" if program == "llama-server" else "--served-model-name"
        env = {"PATH": f"{self.bin}:{SAFE_PATH}", "FAKE_LOG": str(self.log),
               **{f"FAKE_{k.upper()}": v for k, v in fake_env.items()}}
        proc = subprocess.Popen([str(self.bin / program), name_flag, model or self.model, "--host", host,
                                 "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._procs.append(proc)
        if ours:
            (self.state / f"{self.container}.pid").write_text(str(proc.pid), encoding="utf-8")
            if self.kind == "llamacpp":
                self.run_dir.mkdir(parents=True, exist_ok=True)
                (self.run_dir / "server.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
                (self.run_dir / "server.meta").write_text(
                    f"slug={self.bundle.directory.name}\nmodel={model or self.model}\nport={port}\n"
                    f"pid_file={self.run_dir / 'server.pid'}\n", encoding="utf-8")
        wait_listening(host, port)
        return proc

    def close(self) -> None:
        pids = [p.pid for p in self._procs]
        for pid_file in [*self.state.glob("*.pid"), *self.home.glob(".lmds/run/*/server.pid")]:
            try:
                pids.append(int(pid_file.read_text().strip()))
            except (OSError, ValueError):
                pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for proc in self._procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def wait_listening(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"ไม่มีใครฟังที่ {host}:{port} ภายใน {timeout:.0f} วิ")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def box(tmp_path, request):
    """Box ของ engine ที่เทสขอผ่าน parametrize("kind") — เก็บกวาด process ปลอมทุกตัวตอนจบ
    (ไฟล์เทส import fixture นี้ไปใช้: `from tests.single_controller_harness import box`)"""
    kind = request.getfixturevalue("kind")
    made = Box(tmp_path, kind, render(tmp_path, kind))
    try:
        yield made
    finally:
        made.close()
