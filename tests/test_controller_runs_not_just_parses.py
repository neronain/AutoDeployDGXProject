"""controller ต้อง *รันได้* ไม่ใช่แค่ผ่าน bash -n

เคสจริง 2026-08-13 — check_port_free รอบแรกเขียน awk ด้วยลำดับ escape single quote
ของ shell ที่ฝังมาผิด ทำให้ $4 กับ $NF ของ awk กลายเป็น positional parameter ของ
bash พอเจอ set -u ก็ตายทันทีที่ผู้ใช้กด start:

    muse-glimmer-30b-gguf-single.sh: line 344: $4: unbound variable

bash -n ผ่านฉลุย เพราะ syntax ถูกต้องทุกประการ — ปัญหาอยู่ที่ตอนรัน เทสที่ตรวจแค่
syntax จึงไม่มีวันจับได้ ต้องดึงฟังก์ชันออกมารันจริงภายใต้ set -euo pipefail
"""

import re
import socket
import subprocess
import textwrap

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(tmp_path) -> str:
    report = ModelReport(
        repo_id="unsloth/Muse-Glimmer-30B-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=int(30.1 * GIB),
        selected_gguf="Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        context_length=131072,
        kv_dims=KvDims(layers=52, kv_heads=2, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    return next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")


def _extract(text: str, name: str) -> str:
    """ตัดฟังก์ชันออกมาด้วยการนับปีกกา — regex พลาดกับ { } ที่ซ้อนกัน"""
    start = text.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอปีกกาปิดของ {name}")


def _run_guard(tmp_path, port: int) -> subprocess.CompletedProcess:
    body = _extract(_controller(tmp_path), "check_port_free")
    harness = textwrap.dedent(f"""
        set -euo pipefail
        API_PORT={port}
        die() {{ echo "DIED: $*"; exit 9; }}
        {{body}}
        check_port_free
        echo GUARD_PASSED
    """).replace("{body}", body)
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30)


def test_the_guard_runs_without_unbound_variables(tmp_path):
    """port ว่าง → ผ่าน และต้องไม่มี unbound variable ระหว่างทาง"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    # ปิด socket แล้ว port ว่างจริง

    result = _run_guard(tmp_path, free_port)
    combined = result.stdout + result.stderr
    assert "unbound variable" not in combined, combined
    assert result.returncode == 0, combined
    assert "GUARD_PASSED" in result.stdout


def test_the_guard_refuses_a_port_somebody_else_holds(tmp_path):
    """port ที่มีคนฟังอยู่ → ต้อง die ไม่ใช่ปล่อยผ่าน"""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        result = _run_guard(tmp_path, port)

    combined = result.stdout + result.stderr
    assert "unbound variable" not in combined, combined
    assert result.returncode == 9, combined
    assert str(port) in combined


def test_no_shell_quote_escaping_sequences_survive_templating(tmp_path):
    """ลำดับ '\"'\"' เป็นท่าของ shell ไม่ใช่ของ Python/Jinja

    ถ้ามันโผล่ในไฟล์ที่ render ออกมา แปลว่ามีคนเขียนท่านี้ในสตริงที่ไม่ได้ผ่าน shell
    แล้วมันจะกลายเป็นตัวอักษรจริง ทำให้ตัวแปรที่ตั้งใจ quote หลุดออกมาเปล่า ๆ
    """
    assert "'\"'\"'" not in _controller(tmp_path)
