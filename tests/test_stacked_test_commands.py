"""ชุดทดสอบของ controller vLLM — test-tools/test-reasoning/test-vision/parsers/bench/stress ทั้ง stacked และ single

stacked เคยมีแค่ test-text: `./<slug>-stacked.sh test-tools` และ `bench` พิมพ์ usage เฉย ๆ (เจอจริง 2026-09-04)
ทั้งที่ปุ่มบนหน้าเว็บกับ allowlist ของ hub มีชื่อพวกนี้อยู่แล้ว · ทุกข้อรัน controller ที่ render แล้วจริงใต้ bash
(`set -Eeuo pipefail`) กับเซิร์ฟเวอร์ HTTP ปลอมที่ตอบรูป OpenAI — ชุดทดสอบคุยกับ head ที่ 127.0.0.1 เท่านั้น
จึงต้องรันได้โดย **ไม่มี cluster.env** และไม่มี ssh/docker/ip ปลอมบน PATH
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lmds.inventory import controller_commands
from tests.test_audit_stacked_controller import SAFE_PATH, _bundle as _stacked_bundle, _run, _shim
from tests.test_review_templates import _bundle as _render_single, _safetensors_report

MODEL = "test-model"
NEW_COMMANDS = {"test-reasoning", "test-tools", "parsers", "bench", "stress"}


# ───────────────────────── bundles ─────────────────────────
def _multimodal(plan):
    plan.multimodal.modalities = ["image", "text"]


def _single_bundle(tmp_path, tweak=None):
    bundle = _render_single(tmp_path / "single", _safetensors_report(), tweak=tweak)
    assert bundle.controller.name.endswith("-single.sh")
    return bundle


def _bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    return bin_dir


def _run_single(bundle, cmd: list[str], tmp_path: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    # controller เดี่ยวเช็ค `need docker` ตั้งแต่ต้นแม้แค่ help — ใส่ตัวปลอมเงียบ ๆ ถ้าเทสไม่ได้วางของตัวเองไว้
    if not (_bin(tmp_path) / "docker").exists():
        _shim(tmp_path / "bin", "docker", "exit 0\n")
    full = {"PATH": f"{tmp_path / 'bin'}:{SAFE_PATH}", "HOME": str(tmp_path), **(env or {})}
    return subprocess.run(["bash", str(bundle.controller), *cmd], env=full, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=90)


# ───────────────────────── fake OpenAI server ─────────────────────────
class _Server(ThreadingHTTPServer):
    daemon_threads = True


def _serve(mode: str, model: str = MODEL):
    """เซิร์ฟเวอร์ปลอมบนพอร์ตว่าง · mode = tool_calls | text | reasoning | vision · stream ตอบเป็น SSE เสมอ"""

    class Handler(BaseHTTPRequestHandler):
        requests: list[dict] = []

        def log_message(self, *_):  # เงียบ — ไม่ให้ log ของ http.server ปนใน output ของ pytest
            pass

        def _json(self, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # /v1/models — assert_our_server เทียบ id ตัวแรกกับ SERVED_MODEL_NAME
            self._json({"data": [{"id": model}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
            Handler.requests.append(req)
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in ("one", "two", "three", "four"):
                    chunk = {"choices": [{"delta": {"content": word + " "}}]}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 4}}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            if mode == "tool_calls":
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Bangkok"}'}}]}
            elif mode == "reasoning":
                msg = {"role": "assistant", "content": "37 × 43 = 1591",
                       "reasoning_content": "37*40 = 1480, 37*3 = 111, so 1591"}
            elif mode == "vision":
                msg = {"role": "assistant", "content": "Red"}
            else:  # โมเดลเขียน tool call เป็นข้อความรูป Qwen3-Coder — parser ไม่ได้แปลง
                msg = {"role": "assistant",
                       "content": "<function=get_weather>\n<parameter=location>Bangkok</parameter>\n</function>"}
            self._json({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 12}})

    server = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Handler


@pytest.fixture
def fake_server():
    started = []

    def start(mode: str, model: str = MODEL):
        server, handler = _serve(mode, model)
        started.append(server)
        return {"API_PORT": str(server.server_address[1]), "SERVED_MODEL_NAME": MODEL}, handler

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


# ───────────────────────── usage ↔ dispatch ─────────────────────────
def _usage_commands(stdout: str) -> set[str]:
    section = stdout.split("COMMANDS", 1)[1].split("\nOPTIONS", 1)[0]
    names: set[str] = set()
    for line in section.splitlines():
        m = re.match(r"^  ([a-z][a-z-]*(?: \| [a-z][a-z-]*)*)", line)
        if m:
            names.update(x.strip() for x in m.group(1).split("|"))
    return names


def _dispatch_commands(controller: Path) -> set[str]:
    text = controller.read_text(encoding="utf-8")
    block = text[text.index('case "${1:-help}" in'):]
    block = block[:block.index("\nesac")]
    names: set[str] = set()
    for m in re.finditer(r"(?m)^  ([a-z][a-z|-]*)\)", block):
        names.update(m.group(1).split("|"))
    return names - {"banner"}       # alias ของ info — ไม่โฆษณาใน usage โดยตั้งใจ


@pytest.mark.parametrize("multimodal", [False, True])
def test_every_usage_command_is_dispatched_and_vice_versa_in_both_vllm_templates(tmp_path, multimodal):
    tweak = _multimodal if multimodal else None
    stacked = _stacked_bundle(tmp_path, tweak=tweak)
    single = _single_bundle(tmp_path, tweak=tweak)
    for name, bundle, usage in (
        ("stacked", stacked, _run(stacked, ["help"], tmp_path, cluster=False)),
        ("single", single, _run_single(single, ["help"], tmp_path)),
    ):
        assert usage.returncode == 0, usage.stderr
        advertised, dispatched = _usage_commands(usage.stdout), _dispatch_commands(bundle.controller)
        assert advertised == dispatched, f"{name}: usage {advertised ^ dispatched} ต่างจาก dispatch"
        assert NEW_COMMANDS <= advertised, name
        assert ("test-vision" in advertised) is multimodal, name
        # ปุ่มบนหน้าเว็บอ่านจาก dispatch table ตัวเดียวกัน — ต้องเห็นชุดใหม่ด้วย (parsers ไม่ใช่ปุ่ม)
        assert NEW_COMMANDS - {"parsers"} <= set(controller_commands(str(bundle.controller))), name
        assert subprocess.run(["bash", "-n", str(bundle.controller)]).returncode == 0, name


# ───────────────────────── test-tools ─────────────────────────
def test_stacked_test_tools_passes_without_cluster_env_when_the_head_returns_tool_calls(tmp_path, fake_server):
    env, handler = fake_server("tool_calls")
    bundle = _stacked_bundle(tmp_path)
    done = _run(bundle, ["test-tools"], tmp_path, env=env, cluster=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS: โมเดลเรียก tool ได้เองในโหมด auto" in done.stdout
    assert "ยังไม่ได้ตั้งค่าคลัสเตอร์" not in done.stdout + done.stderr
    assert [r["tool_choice"] for r in handler.requests] == ["auto", "required"]
    assert all(r["model"] == MODEL and r["tools"][0]["function"]["name"] == "get_weather" for r in handler.requests)

    handler.requests.clear()
    only_required = _run(bundle, ["test-tools", "required"], tmp_path, env=env, cluster=False)
    assert only_required.returncode == 0 and "PASS(required)" in only_required.stdout
    assert [r["tool_choice"] for r in handler.requests] == ["required"]


def test_stacked_test_tools_fails_and_names_the_parser_when_the_call_comes_back_as_text(tmp_path, fake_server):
    env, _ = fake_server("text")
    done = _run(_stacked_bundle(tmp_path), ["test-tools"], tmp_path, env=env, cluster=False)
    assert done.returncode == 1
    assert "FAIL(auto)" in done.stdout and "qwen3_coder" in done.stdout
    assert "restart --tool-parser" in done.stdout


def test_test_commands_refuse_a_port_that_answers_for_another_model(tmp_path, fake_server):
    """พอร์ตเดียวกันทุก bundle — ถ้าโมเดลอื่นยึดอยู่ ผลทดสอบจะเป็นของมัน ไม่ใช่ตัวนี้"""
    env, _ = fake_server("tool_calls", model="someone-elses-model")
    done = _run(_stacked_bundle(tmp_path), ["test-tools"], tmp_path, env=env, cluster=False)
    assert done.returncode == 1
    assert "someone-elses-model" in done.stderr and "restart --port 8001" in done.stderr


# ───────────────────────── test-reasoning / test-vision ─────────────────────────
def test_stacked_test_reasoning_reads_reasoning_content(tmp_path, fake_server):
    env, _ = fake_server("reasoning")
    done = _run(_stacked_bundle(tmp_path), ["test-reasoning"], tmp_path, env=env, cluster=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("PASS: 1591 ถูกต้อง")


def test_stacked_test_vision_exists_only_for_multimodal_plans_and_sends_a_generated_png(tmp_path, fake_server):
    env, handler = fake_server("vision")
    text_only = _run(_stacked_bundle(tmp_path / "text"), ["test-vision"], tmp_path, env=env, cluster=False)
    assert "USAGE" in text_only.stdout and not handler.requests, "plan ไม่มี multimodal ต้องไม่มีคำสั่ง"

    done = _run(_stacked_bundle(tmp_path / "mm", tweak=_multimodal), ["test-vision"], tmp_path, env=env, cluster=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("PASS")
    image = handler.requests[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")


# ───────────────────────── bench / stress ─────────────────────────
def test_stacked_bench_and_stress_talk_to_the_head_only(tmp_path, fake_server):
    env, handler = fake_server("tool_calls")
    bundle = _stacked_bundle(tmp_path)
    bench = _run(bundle, ["bench", "2", "8"], tmp_path, env=env, cluster=False)
    assert bench.returncode == 0, bench.stdout + bench.stderr
    assert "run 1:" in bench.stdout and "run 2:" in bench.stdout and "2 runs × 8 tokens" in bench.stdout
    assert "4 tokens" in bench.stdout, "ต้องใช้ usage.completion_tokens จาก stream ไม่ใช่นับ chunk เดา"
    assert all(r["stream"] and r["max_tokens"] == 8 for r in handler.requests)

    handler.requests.clear()
    stress = _run(bundle, ["stress", "6", "3"], tmp_path, env=env, cluster=False)
    assert stress.returncode == 0, stress.stdout + stress.stderr
    assert "ok 6/6" in stress.stdout and "stress: PASS" in stress.stdout and len(handler.requests) == 6


# ───────────────────────── parsers ─────────────────────────
# docker ปลอม: `docker run … --entrypoint python3 IMAGE -c SCRIPT` รัน SCRIPT ในเครื่องนี้ (PYTHONPATH ชี้ vllm ปลอม)
_DOCKER_PY = '''
case "$1" in
  run) exec python3 -c "${@: -1}" ;;
  *) exit 0 ;;
esac
'''

_VLLM_HELP = '''
cat <<'EOF'
usage: vllm serve [options]

options:
  --tool-call-parser {deepseek_v3,hermes,llama3_json,qwen3_coder}
                        Select the tool call parser depending on the model
  --reasoning-parser {deepseek_r1,qwen3}
                        Select the reasoning parser
EOF
'''


def _fake_vllm(root: Path, layout: str) -> Path:
    """vllm ปลอม · new = registry ที่ vllm.tool_parsers (0.28) · old = ที่ vllm.entrypoints.openai.tool_parsers · bare = ไม่มี"""
    pkg = root / "site" / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    tool = textwrap.dedent('''
        class ToolParserManager:
            tool_parsers = {"hermes": object(), "qwen3_xml": object()}
    ''')
    if layout == "new":
        (pkg / "tool_parsers.py").write_text(tool, encoding="utf-8")
    elif layout == "old":
        (pkg / "entrypoints" / "openai").mkdir(parents=True)
        (pkg / "entrypoints" / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "entrypoints" / "openai" / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "entrypoints" / "openai" / "tool_parsers.py").write_text(tool, encoding="utf-8")
    if layout != "bare":
        (pkg / "reasoning.py").write_text(textwrap.dedent('''
            class ReasoningParserManager:
                reasoning_parsers = {"deepseek_r1": object()}
        '''), encoding="utf-8")
    return pkg.parent


@pytest.mark.parametrize("layout", ["new", "old"])
def test_parsers_reads_the_registry_wherever_this_vllm_keeps_it(tmp_path, layout):
    _shim(_bin(tmp_path), "docker", _DOCKER_PY)
    env = {"PYTHONPATH": str(_fake_vllm(tmp_path, layout))}
    for done in (_run(_stacked_bundle(tmp_path), ["parsers"], tmp_path, env=env, cluster=False),
                 _run_single(_single_bundle(tmp_path), ["parsers"], tmp_path, env=env)):
        assert done.returncode == 0, done.stdout + done.stderr
        assert "tool parsers  (--tool-parser):\n  hermes qwen3_xml" in done.stdout
        assert "reasoning parsers  (--reasoning-parser):\n  deepseek_r1" in done.stdout
        assert "--help" not in done.stdout


def test_parsers_falls_back_to_vllm_serve_help_when_the_registry_moved_again(tmp_path):
    """0.28 ย้าย module แล้วครั้งหนึ่ง — ครั้งหน้าอย่างน้อยยังอ่าน choices จาก argparse ได้"""
    _shim(_bin(tmp_path), "docker", _DOCKER_PY)
    _shim(_bin(tmp_path), "vllm", _VLLM_HELP)
    env = {"PYTHONPATH": str(_fake_vllm(tmp_path, "bare"))}
    for done in (_run(_stacked_bundle(tmp_path), ["parsers"], tmp_path, env=env, cluster=False),
                 _run_single(_single_bundle(tmp_path), ["parsers"], tmp_path, env=env)):
        assert done.returncode == 0, done.stdout + done.stderr
        assert "(จาก vllm serve --help)\n  deepseek_v3 hermes llama3_json qwen3_coder" in done.stdout
        assert "(จาก vllm serve --help)\n  deepseek_r1 qwen3" in done.stdout


def test_parsers_says_so_when_nothing_can_be_read(tmp_path):
    _shim(_bin(tmp_path), "docker", _DOCKER_PY)
    done = _run(_stacked_bundle(tmp_path), ["parsers"], tmp_path,
                env={"PYTHONPATH": str(_fake_vllm(tmp_path, "bare"))}, cluster=False)
    assert done.returncode == 0
    assert done.stdout.count("อ่านจาก engine ไม่ได้") == 2
