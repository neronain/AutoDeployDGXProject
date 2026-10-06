"""bundle ที่ render ด้วย renderer รุ่นก่อนต้องถูกนับว่า "เก่า" แล้ว regenerate ได้ controller ที่ใช้งานได้ — audit 2026-10-06 ข้อ 5

ตัวตัดสินว่า controller บน node เก่ากว่าแพ็กเกจคือ `template_hash` (ฝังใน MODEL_PROFILE.yaml และ `TEMPLATE_HASH=` ที่หัว
controller) · `lmds node install` จบด้วย `lmds bundles refresh --all --if-older` ซึ่ง regenerate เฉพาะใบที่ hash ไม่ตรง

เดิม hash คิดจากไฟล์ `templates/*.j2` อย่างเดียว · รอบนี้การ escape ค่าย้ายไปทำที่ renderer (ไม่ใช่ใน template) — ถ้าวันหนึ่ง
แก้ renderer โดยไม่แตะ template เลย bundle ที่ render ด้วยตัวเก่าจะได้ hash เดียวกับตัวใหม่ hub รายงาน "ตรง template"
และ refresh ข้ามไป ทั้งที่ controller บนเครื่องยังเป็นของเก่า → RENDER_REVISION ถูกนับรวมใน hash

เทสที่สอง regenerate bundle "รุ่นก่อน" จริง ๆ ผ่าน refresh_bundles แล้ว **รัน controller ใหม่ใต้ bash** — ค่าที่ผู้ใช้ตั้งไว้
(bundle.env · bundle.args) ต้องยังไปถึง argv และไฟล์ของผู้ใช้ต้องไม่ถูกแตะ
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest
import yaml

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle, renderer
from lmds.inspector.report import ArtifactType, KvDims, ModelReport, ShardFile

BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if os.access(p, os.X_OK)), "bash")
SAFE_PATH = "/usr/bin:/bin:/sbin:/usr/sbin"
SLUG = "qwen3-32b"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """regenerate เป็นงานออฟไลน์ — ต่อเน็ต = เทสล้ม · และไม่ถามสถานะ process/docker ของเครื่องที่รันเทส"""
    def boom(*a, **k):
        raise AssertionError("regenerate ห้ามต่อเน็ต")
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.fleet.manager._health_ok", lambda port, engine="": False)


def _render_with_revision(tmp_path: Path, monkeypatch, revision: int) -> Path:
    """bundle vLLM ที่ render ตอน RENDER_REVISION = revision แล้วลงทะเบียนให้ fleet เห็น (แบบเดียวกับ bundle บน node)"""
    root = tmp_path / "bundles"
    monkeypatch.setenv("LMDS_BUNDLE_DIRS", str(root))
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    report = ModelReport(
        repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned-123", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, shard_count=2, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
        tokenizer_files=["tokenizer.json", "tokenizer_config.json"],
        safetensor_shards=[ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=12),
                           ShardFile(filename="weights/model 00002+of=00002.safetensors", size_bytes=7)])
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    with monkeypatch.context() as patch:
        patch.setattr(renderer, "RENDER_REVISION", revision)
        patch.setattr(renderer, "_TEMPLATE_HASH", {})
        bundle = render_bundle(plan, report, fit, root)
    run_dir = tmp_path / "run" / SLUG
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        f"slug={SLUG}\nmodel={SLUG}\nmodel_id=Qwen/Qwen3-32B\nengine=vllm\nmode=docker\nport=8011\n"
        f"container=lmds-{SLUG}\npid_file=\ncontroller={bundle.controller}\nstarted_at=\n", encoding="utf-8")
    return bundle.directory


def test_a_renderer_only_change_makes_existing_bundles_stale_and_refresh_picks_them_up(tmp_path, monkeypatch):
    from lmds.fleet.consistency import controller_state
    from lmds.fleet.refresh import refresh_bundles
    from lmds.generator.renderer import template_hash

    directory = _render_with_revision(tmp_path, monkeypatch, renderer.RENDER_REVISION - 1)
    controller = directory / f"{SLUG}-single.sh"
    profile = yaml.safe_load((directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    old_hash = profile["template_hash"]

    # ไฟล์ template ชุดเดียวกันเป๊ะ — ต่างกันแค่รุ่นของ renderer
    assert old_hash != template_hash()
    assert f'TEMPLATE_HASH="{old_hash}"' in controller.read_text(encoding="utf-8")
    state = controller_state(profile, controller)
    assert state["state"] == "stale" and old_hash in state["reason"]

    (result,) = refresh_bundles([SLUG], if_older=True)       # สิ่งที่ `lmds node install` เรียกตอนท้าย
    assert result.action == "refreshed", result.detail
    profile = yaml.safe_load((directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["template_hash"] == template_hash()
    assert controller_state(profile, controller)["state"] == "ok"
    (again,) = refresh_bundles([SLUG], if_older=True)
    assert again.action == "current", "รอบถัดไปต้องไม่ regenerate ซ้ำ — ไม่งั้นทุก node install ทิ้งไฟล์ .replaced-* เพิ่มหนึ่งใบ"


def test_refreshing_a_bundle_from_the_previous_renderer_yields_a_controller_that_runs(tmp_path, monkeypatch):
    """ค่าปกติ render ออกมาเหมือนเดิมทุกไบต์ทั้งก่อนและหลังมี escape (เทียบ 1,416 ไฟล์ตอนแก้) — เทสนี้พิสูจน์ปลายทาง:
    regenerate แล้วรันได้ · ตารางไฟล์ยกมาครบ (ชื่อที่มีช่องว่าง + = ด้วย) · ค่าที่ผู้ใช้ตั้งยังมีผล · ไฟล์ของผู้ใช้ไม่ถูกแตะ"""
    from lmds.fleet.refresh import refresh_bundles
    from lmds.validator.gates import run_gates

    directory = _render_with_revision(tmp_path, monkeypatch, renderer.RENDER_REVISION - 1)
    controller = directory / f"{SLUG}-single.sh"
    before = controller.read_text(encoding="utf-8")
    mine = {
        "bundle.env": 'API_PORT="${API_PORT:-8011}"\nSERVED_MODEL_NAME="${SERVED_MODEL_NAME:-prod-alias}"\n'
                      'VLLM_IMAGE="${VLLM_IMAGE:-registry.local/custom:1}"\n',
        "bundle.args": "--allowed-origins * --max-log-len 100\n",
        "cluster.env": "MASTER_IP=10.0.0.1\n",
    }
    for name, content in mine.items():
        (directory / name).write_text(content, encoding="utf-8")
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / SLUG).write_text("sekrit-key\n", encoding="utf-8")

    (result,) = refresh_bundles([SLUG], if_older=True)

    assert result.action == "refreshed", result.detail
    after = controller.read_text(encoding="utf-8")
    assert after != before
    # ของเดิมเก็บไว้ครบ · ไฟล์ของผู้ใช้เหมือนเดิมทุกไบต์
    assert Path(result.replaced).read_text(encoding="utf-8") == before
    assert {name: (directory / name).read_text(encoding="utf-8") for name in mine} == mine
    assert (keys / SLUG).read_text(encoding="utf-8") == "sekrit-key\n"
    assert all(g.passed for g in run_gates(directory, include_checksums=True)), "bundle ที่ regenerate ต้องผ่านทุกด่านรวม checksum"

    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": SAFE_PATH, "HOME": str(home), "LMDS_KEY_ROOT": str(keys), "DRY_RUN": "1"}
    done = subprocess.run([BASH, str(controller), "start"], cwd=directory, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    argv = done.stdout.splitlines()
    assert argv[argv.index("--served-model-name") + 1] == "prod-alias"
    assert argv[argv.index("--port") + 1] == "8011"
    assert "IMAGE=registry.local/custom:1" in argv
    at = argv.index("--allowed-origins")
    assert argv[at:at + 4] == ["--allowed-origins", "*", "--max-log-len", "100"]

    # ตารางไฟล์ที่ refresh ยกมาจากหัว controller เดิม ต้องไปถึง bash ครบตัวอักษร
    probe = ('trap \'printf "%s\\n" "${SHARD_FILES[@]}" "${SHARD_SIZES[@]}" > "$OUT"\' EXIT\n'
             'set -- help\n. "$CONTROLLER" >/dev/null 2>&1\n')
    out = tmp_path / "tables.out"
    subprocess.run([BASH, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                   env={"PATH": SAFE_PATH, "HOME": str(home), "OUT": str(out), "CONTROLLER": str(controller)})
    assert out.read_text(encoding="utf-8").splitlines() == [
        "model-00001-of-00002.safetensors", "weights/model 00002+of=00002.safetensors", "12", "7"]
