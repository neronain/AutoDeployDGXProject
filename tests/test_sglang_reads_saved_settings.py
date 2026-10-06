"""SGLang ต้องใช้ served name / image ที่บันทึกไว้ใน bundle.env — audit templates 2026-10-06 ข้อ 4

ทุกบรรทัดใน bundle.env เป็นรูป `VAR="${VAR:-ค่า}"` ซึ่งไม่ทำอะไรเลยถ้า VAR ถูกตั้งไปแล้ว · หัว controller ของ SGLang
ประกาศ SERVED_MODEL_NAME กับ SGLANG_IMAGE **ก่อน** จุดที่ source bundle.env ค่าที่ผู้ใช้บันทึกจึงถูกเมินเงียบ ๆ:
`lmds set --served-name prod-alias` เขียนไฟล์สำเร็จ แต่ API ยังเสิร์ฟชื่อเดิม (port ที่ประกาศใต้จุด source ใช้ได้ปกติ
จึงดูเหมือนไฟล์ถูกอ่านแล้ว) · vLLM กับ llama.cpp แก้เรื่องเดียวกันไปเมื่อ 2026-09-03 — SGLang ตกไป

เทสเขียน bundle.env ด้วยมือ (รูปเดียวกับที่ bundle_settings.write เขียน) แล้วรัน `DRY_RUN=1 start` จริง ดู argv
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
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport, ShardFile

BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if os.access(p, os.X_OK)), "bash")
SAFE_PATH = "/usr/bin:/bin:/sbin:/usr/sbin"


def _bundle_env(image_var: str) -> str:
    """รูปเดียวกับที่ fleet/bundle_settings.write เขียน — ทุกบรรทัดเป็น ${VAR:-value}"""
    return (
        'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-prod-alias}"\n'
        f'{image_var}="${{{image_var}:-registry.local/custom:1}}"\n'
        'API_PORT="${API_PORT:-8011}"\n'
    )


def _dry_run(tmp_path: Path, engine: Engine | None, extra_env: dict | None = None) -> list[str]:
    report = ModelReport(
        repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned-123", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, shard_count=2, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
        safetensor_shards=[ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=12)])
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=engine)
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles")
    image_var = "SGLANG_IMAGE" if engine is Engine.SGLANG else "VLLM_IMAGE"
    (bundle.directory / "bundle.env").write_text(_bundle_env(image_var), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    done = subprocess.run([BASH, str(bundle.controller), "start"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=60, env={"PATH": SAFE_PATH, "HOME": str(home), "DRY_RUN": "1", **(extra_env or {})})
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.splitlines()


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize("engine", [Engine.SGLANG, None], ids=["sglang", "vllm"])
def test_saved_served_name_image_and_port_reach_the_engine(tmp_path, engine):
    argv = _dry_run(tmp_path, engine)
    assert _after(argv, "--served-model-name") == "prod-alias"
    assert "IMAGE=registry.local/custom:1" in argv
    assert _after(argv, "--port") == "8011"


def test_the_environment_still_wins_over_the_saved_file(tmp_path):
    """ลำดับความสำคัญต้องเหมือน engine อื่น: flag > env ภายนอก > bundle.env > ค่าของ bundle"""
    argv = _dry_run(tmp_path, Engine.SGLANG, {"SERVED_MODEL_NAME": "from-env", "SGLANG_IMAGE": "env/image:2"})
    assert _after(argv, "--served-model-name") == "from-env"
    assert "IMAGE=env/image:2" in argv
    assert _after(argv, "--port") == "8011", "ค่าที่ env ไม่ได้ตั้ง ยังต้องมาจากไฟล์"
