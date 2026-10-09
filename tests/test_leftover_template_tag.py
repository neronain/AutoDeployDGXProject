"""template tag ที่หลุดออกมาเป็นตัวหนังสือ และด่านที่มองไม่เห็นมัน — audit templates 2026-10-06 ข้อ 2

controller stacked เขียนข้อความ "port ถูกใช้อยู่แล้ว" ไว้ในบล็อก `{% raw %}` · `{{ slug }}` ในนั้นจึงไม่เคยถูกแทนค่า
ผู้ใช้ที่ start แล้ว port ชนได้คำแนะนำว่า

    (ถ้าตั้งค่าไว้ให้ถาวร: lmds set {{ slug }} --port <PORT>)

และบรรทัดเดียวกัน (ทั้ง vLLM · SGLang · stacked) พิมพ์ชื่อโมเดลที่ยึด port เป็น '''other-model''' เพราะลำดับ '"'"'
(ท่า escape single quote ของ shell) ถูกเขียนในสตริงที่อยู่ใน double quote อยู่แล้ว · gate `template-rendered` ผ่าน
เพราะมองหาแต่ `{%` — bash -n ก็ผ่าน เพราะมันเป็นแค่ตัวหนังสือในข้อความ

เทสรัน check_port_free ตัวจริงโดยมีเซิร์ฟเวอร์ HTTP จริงยึด port อยู่ แล้วอ่านข้อความที่ผู้ใช้จะเห็น
"""

from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport, ShardFile
from lmds.validator.gates import gate_template_rendered

BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if os.access(p, os.X_OK)), "bash")
KINDS = ("vllm", "sglang", "llamacpp", "stacked")


def _bundle(tmp_path: Path, kind: str):
    if kind == "llamacpp":
        report = ModelReport(
            repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
            weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
            selected_gguf="Qwen3-8B-Q4_K_M.gguf",
            gguf_variants=[GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=12, sha256=None)],
            has_chat_template=True, license="apache-2.0")
    else:
        big = kind == "stacked"
        report = ModelReport(
            repo_id="nvidia/DeepSeek-V4-Flash-NVFP4" if big else "Qwen/Qwen3-32B", revision_sha="sha-pinned-123",
            artifact_type=ArtifactType.SAFETENSORS, weight_bytes=(157 if big else 65) * GIB, shard_count=2,
            context_length=131072 if big else 40960, kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
            has_chat_template=True,
            safetensor_shards=[ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=12)])
    fit = analyze(report, PRESETS["dgx-spark-stacked" if kind == "stacked" else "dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def _extract(text: str, name: str) -> str:
    start = text.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอปีกกาปิดของ {name}")


class _Models(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — ชื่อที่ http.server กำหนด
        body = json.dumps({"data": [{"id": "other-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def held_port():
    """เซิร์ฟเวอร์จริงที่ยึด port และตอบ /v1/models ว่าเป็น other-model — สิ่งที่ check_port_free ถามจริง"""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("kind", ["vllm", "sglang", "stacked"])
def test_the_port_in_use_message_names_the_bundle_and_the_model_holding_the_port(tmp_path, kind, held_port):
    bundle = _bundle(tmp_path, kind)
    text = bundle.controller.read_text(encoding="utf-8")
    slug = bundle.directory.name
    declared = re.search(r'^SLUG="[^"\n]*"$', text, re.M)        # stacked ประกาศ SLUG ไว้ที่หัวไฟล์ — ใช้บรรทัดจริงของมัน
    harness = "\n".join([
        "set -Eeuo pipefail",
        declared.group(0) if declared else "",
        f"API_PORT={held_port}",
        'die() { echo "ERROR: $*" >&2; exit 9; }',
        # controller เดี่ยวถามพอร์ตผ่าน _tcp_open (เพดานเวลา) — stacked ถามแค่ 127.0.0.1 จึงไม่มีตัวนี้
        _extract(text, "_tcp_open") if "_tcp_open() {" in text else "",
        _extract(text, "check_port_free"),
        "check_port_free",
        "echo GUARD_PASSED",
    ])
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")

    done = subprocess.run([BASH, str(script), "start"], capture_output=True, text=True, timeout=60)
    said = done.stderr

    assert done.returncode == 9 and "GUARD_PASSED" not in done.stdout, done.stdout + said
    assert f"lmds set {slug} --port <PORT>" in said, said
    assert "{{" not in said and "}}" not in said, said
    assert "โดยโมเดล 'other-model'" in said and "'''" not in said, said


@pytest.mark.parametrize("kind", KINDS)
def test_rendered_controllers_have_no_leftover_tags(tmp_path, kind):
    """ทั้ง 4 template ต้องผ่าน — และต้องผ่านทั้งที่มี `{{.Names}}` ของ docker อยู่จริงในไฟล์ (ไม่ใช่ผ่านเพราะไม่มี {{ เลย)"""
    bundle = _bundle(tmp_path, kind)
    result = gate_template_rendered(bundle.directory)
    assert result.passed, result.detail
    if kind != "llamacpp" or "{{" in bundle.controller.read_text(encoding="utf-8"):
        assert "{{." in bundle.controller.read_text(encoding="utf-8")


@pytest.mark.parametrize("leftover", [
    '  echo "ตั้งถาวร: lmds set {{ slug }} --port <PORT>"',
    'MODEL_ID="{{ plan.model_id }}"',
    '  "{{ shard.filename }}"',
    'X="${{ \'{\' }}X:-1}"',
    "  echo {{- slug -}}",
])
def test_the_gate_sees_a_variable_tag_that_was_never_rendered(tmp_path, leftover):
    bundle = _bundle(tmp_path, "vllm")
    text = bundle.controller.read_text(encoding="utf-8")
    lines = text.split("\n")
    lines.insert(40, leftover)
    bundle.controller.write_text("\n".join(lines), encoding="utf-8")

    result = gate_template_rendered(bundle.directory)

    assert not result.passed
    assert f"{bundle.controller.name}:41:" in result.detail, result.detail


@pytest.mark.parametrize("legit", [
    """  docker inspect --format '{{index .RepoDigests 0}}' "$img\"""",
    """  docker ps --format 'table {{.Names}}\\t{{.Status}}'""",
    """  docker inspect -f '{{json .Mounts}}' "$c\"""",
    """  docker inspect -f '{{ .State.Status }}' "$c\"""",
    """  docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' "$c\"""",
    """  x="${a:-${b:-${c}}}\"""",
    """  body='{"a":{"b":{"c":1}}}'""",
    """  awk '{{ print $1 }}' /dev/null""",
])
def test_braces_the_controller_emits_on_purpose_are_not_mistaken_for_tags(tmp_path, legit):
    """Go template ของ docker · ${a:-${b}} · JSON ซ้อน — ทั้งหมดมี {{ หรือ }} โดยชอบ"""
    bundle = _bundle(tmp_path, "vllm")
    lines = bundle.controller.read_text(encoding="utf-8").split("\n")
    lines.insert(40, legit)
    bundle.controller.write_text("\n".join(lines), encoding="utf-8")
    result = gate_template_rendered(bundle.directory)
    assert result.passed, result.detail
