"""แฟล็กที่บันทึกด้วย `lmds set --extra-args` ต้องไปถึง engine ตามที่พิมพ์ — audit templates 2026-10-06 ข้อ 3

controller แตก EXTRA_SERVE_ARGS เป็น argv ด้วย `for extra_arg in $EXTRA_SERVE_ARGS` ซึ่งนอกจากแตกคำแล้ว bash ยัง
*ขยายชื่อไฟล์* ให้ด้วย · `lmds set` ยอมให้มี `*` (CORS ของ vLLM ใช้ `--allowed-origins *`) ผลคือ

    --allowed-origins * --max-log-len 100
    → --allowed-origins MODEL_PROFILE.yaml README.md bundle.args … --max-log-len 100

engine ได้ชื่อไฟล์ในโฟลเดอร์ที่ยืนอยู่แทน `*` — แล้วแต่ว่ารันจากโฟลเดอร์ไหน (systemd autostart กับคนพิมพ์เองได้ผลต่างกัน)

เทสบันทึกค่าด้วย `bundle_settings.write` ตัวจริง แล้วรัน controller จริงจากโฟลเดอร์ที่มีไฟล์ ดู argv ที่มันจะส่งให้ engine
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.fleet import bundle_settings
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport, ShardFile

BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if os.access(p, os.X_OK)), "bash")
SAFE_PATH = "/usr/bin:/bin:/sbin:/usr/sbin"
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
            safetensor_shards=[ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=12),
                               ShardFile(filename="model-00002-of-00002.safetensors", size_bytes=7)])
    fit = analyze(report, PRESETS["dgx-spark-stacked" if kind == "stacked" else "dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def _engine_argv(tmp_path: Path, kind: str, extra_args: str) -> list[str]:
    """argv ที่ controller จะส่งให้ engine (ทีละบรรทัด) — รันจากโฟลเดอร์ที่มีไฟล์ให้ glob ไปโดน"""
    bundle = _bundle(tmp_path, kind)
    saved = bundle_settings.write(bundle.directory, {"extra_args": extra_args})
    assert saved == {"extra_args": extra_args}, "lmds set ต้องรับค่านี้ — ไม่งั้นเทสนี้ไม่ได้ทดสอบทางที่ผู้ใช้ใช้จริง"
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": SAFE_PATH, "HOME": str(home), "MASTER_IP": "10.0.0.1", "WORKER_IP": "10.0.0.2", "SSH_USER": "u"}
    if kind in ("llamacpp", "stacked"):
        command = [BASH, str(bundle.controller), "serve-args"]
    else:
        command = [BASH, str(bundle.controller), "start"]
        env["DRY_RUN"] = "1"
    # โฟลเดอร์ bundle เองคือที่ที่ systemd และปุ่มบนเว็บรัน controller — มี MODEL_PROFILE.yaml README.md bundle.args อยู่แล้ว
    for name in ("a", "x1", "ab"):
        (bundle.directory / name).write_text("", encoding="utf-8")
    done = subprocess.run(command, cwd=bundle.directory, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.splitlines()


@pytest.mark.parametrize("kind", KINDS)
def test_a_star_in_saved_extra_args_reaches_the_engine_as_a_star(tmp_path, kind):
    argv = _engine_argv(tmp_path, kind, "--allowed-origins * --max-log-len 100")
    at = argv.index("--allowed-origins")
    assert argv[at:at + 4] == ["--allowed-origins", "*", "--max-log-len", "100"], argv[at:at + 8]
    assert "README.md" not in argv and "MODEL_PROFILE.yaml" not in argv


@pytest.mark.parametrize("kind", KINDS)
def test_question_marks_and_brackets_are_not_matched_against_files_either(tmp_path, kind):
    """`?` กับ `[…]` ขยายเฉพาะเมื่อมีไฟล์ที่ตรง — บั๊กแบบที่ขึ้นกับว่าในโฟลเดอร์มีไฟล์ชื่ออะไร (ที่นี่มี a · x1 · ab)"""
    argv = _engine_argv(tmp_path, kind, "--stop ? --pattern x[0-9] --chars [ab][ab]")
    at = argv.index("--stop")
    assert argv[at:at + 6] == ["--stop", "?", "--pattern", "x[0-9]", "--chars", "[ab][ab]"], argv[at:at + 8]


@pytest.mark.parametrize("kind", KINDS)
def test_json_without_spaces_is_still_one_argument(tmp_path, kind):
    """กติกาเดิมต้องอยู่: `--hf-overrides {...}` ที่เขียนติดกันเป็น argv ตัวเดียว (มี * และ { } อยู่ข้างในก็ตาม)"""
    overrides = '{"architectures":["Qwen3ForSequenceClassification"],"glob":"*","classifier_from_token":["no","yes"]}'
    argv = _engine_argv(tmp_path, kind, f"--hf-overrides {overrides} --seed 7")
    at = argv.index("--hf-overrides")
    assert argv[at:at + 4] == ["--hf-overrides", overrides, "--seed", "7"], argv[at:at + 5]
