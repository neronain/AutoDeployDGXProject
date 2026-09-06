"""regenerate controller แบบออฟไลน์ — `lmds bundles refresh` (Update path 2026-09-06 §5.4)

bundle บน node 46/47 ใบเป็น controller ของ 0.3.0–0.6.0 · `lmds rebuild` ต้องถึง HF และมีแค่ CLI · เทสนี้ใช้ profile 0.5.1
จริงของ qwen3-8-flash-next-uncensored-gguf (spark-head) กับหัว controller เดิม แล้วพิสูจน์ว่า render ใหม่ได้โดยไม่แตะเน็ต
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import yaml

SLUG = "qwen3-8-flash-next-uncensored-gguf"

PROFILE_051 = {
    "profile_version": 1, "generated_by": "lmds 0.5.1", "generator": "rule-based",
    "model": {"id": "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF", "revision": "06756566a4b4a29d0dee62ccb405914a15fdf80d",
              "served_name": SLUG, "artifact_type": "gguf",
              "selected_gguf": "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf", "gated": True,
              "license": "apache-2.0", "architecture": "qwen4exp", "params_total": None, "weight_bytes": 97473155200,
              "native_context": 262144},
    "runtime": {"engine": "llamacpp", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                "image_pin": "sha256:8557e3d273aa6010d46f355e826348b691ba3ddffccae8eaf0150596bbc3ec42"},
    "topology": "single",
    "target": {"name": "dgx-spark-single", "memory_model": "unified", "budget_gb": 114.5, "verdict": "fits",
               "max_safe_context": 262144, "llamacpp_dir": None, "fit_notes": ["context 262,144 — ค่าสูงสุด"]},
    "serving": {"context": 262144, "max_output_tokens": 8192, "gpu_memory_utilization": 0.85, "kv_cache_dtype": "auto",
                "max_num_seqs": 1, "extra_flags": [], "extra_env": {}},
    "features": {"tool_calling": {"enabled": False, "parser": None, "chat_template_override": None, "parallel": False},
                 "reasoning": {"enabled": False, "parser": None},
                 "multimodal": {"modalities": ["image", "text"], "projector_files": ["mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"]},
                 "moe": {"experts": 512, "experts_active": 10}, "speculative": {"draft_files": [], "embedded": False}},
    "facts": [{"claim": "artifact เป็น gguf", "source": "hub-api", "confidence": "verified"}],
    "warnings": [], "flags_needing_approval": [], "validation": {"static": True, "hardware": False},
}

CONTROLLER_051 = """#!/usr/bin/env bash
# qwen3-8-flash-next-uncensored-gguf — llama.cpp single-node controller (GGUF)
set -Eeuo pipefail
SCRIPT_VERSION="${SCRIPT_VERSION:-0.5.1}"
MODEL_ID="orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"
RUNTIME_MODE="${RUNTIME_MODE:-native}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/src/llama.cpp}"
MODEL_FILES=(
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf"
  "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf"
  "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
)
MODEL_URLS=(
  "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/06756566a4b4a29d0dee62ccb405914a15fdf80d/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
  "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/06756566a4b4a29d0dee62ccb405914a15fdf80d/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf"
  "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/06756566a4b4a29d0dee62ccb405914a15fdf80d/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf"
  "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/06756566a4b4a29d0dee62ccb405914a15fdf80d/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
)
EXPECTED_SIZES=(
  "44766155936"
  "44735995008"
  "7971004256"
  "907543296"
)
EXPECTED_SHAS=(
  "50dc0856abd4a8ecea97a47ffa197bde3ea8d7d0f49d0e1fea7f71c97e8a70d1"
  "2a309e0b112fde96ba3bcba5a6b58cc05e5df7bb7fad5a990eaa51df335b0e43"
  "ebc43c58e2eaeba1d5bdf62c8cb1f0eac198c4dc01941f771921edeebf574bc3"
  "f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9"
)
MODEL_FILE="${MODEL_FILES[0]}"
MMPROJ_FILE="${MMPROJ_FILE-mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf}"
case "${1:-help}" in
  download) : ;;
  start) : ;;
esac
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """regenerate ต้องไม่แตะเน็ต — socket ใด ๆ = เทสล้ม (lmds rebuild ต่างจากนี้ตรงที่ inspect ซ้ำจาก HF)"""
    def boom(*a, **k):
        raise AssertionError("regenerate ห้ามต่อเน็ต")
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.fleet.manager._health_ok", lambda port, engine="": False)   # ไม่ยิง /health ของเครื่องจริง


def _bundle(tmp_path: Path, monkeypatch, profile: dict = PROFILE_051, controller: str = CONTROLLER_051,
            slug: str = SLUG, mode: str = "native", running: bool = False) -> Path:
    root = tmp_path / "bundles"
    monkeypatch.setenv("LMDS_BUNDLE_DIRS", str(root))
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    directory = root / slug
    directory.mkdir(parents=True)
    ctl = directory / f"{slug}-single.sh"
    ctl.write_text(controller, encoding="utf-8")
    ctl.chmod(0o755)
    (directory / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (directory / "bundle.env").write_text('API_PORT="${API_PORT:-8011}"\nCTX_SIZE="${CTX_SIZE:-65536}"\n', encoding="utf-8")
    (directory / "bundle.args").write_text("--flash-attn on\n", encoding="utf-8")
    (directory / "cluster.env").write_text("MASTER_IP=10.0.0.1\n", encoding="utf-8")
    run_dir = tmp_path / "run" / slug
    run_dir.mkdir(parents=True)
    pid = ""
    if running:
        import os
        (run_dir / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
        pid = str(run_dir / "server.pid")
    (run_dir / "server.meta").write_text(
        f"slug={slug}\nmodel={slug}\nmodel_id={profile['model']['id']}\nengine={profile['runtime']['engine']}\n"
        f"mode={mode}\nport=8011\ncontainer=lmds-{slug}\npid_file={pid}\ncontroller={ctl}\nstarted_at=\n", encoding="utf-8")
    return directory


def test_bundles_refresh_offline_keeps_settings(tmp_path, monkeypatch):
    from lmds.fleet.refresh import format_result, refresh_bundles
    from lmds.generator.renderer import template_hash

    directory = _bundle(tmp_path, monkeypatch, running=True)
    (result,) = refresh_bundles([SLUG])
    assert result.action == "refreshed", format_result(result)
    assert result.before == "0.5.1" and result.after and result.after != "0.5.1"
    assert result.running and "restart" in format_result(result)

    ctl = directory / f"{SLUG}-single.sh"
    text = ctl.read_text(encoding="utf-8")
    assert "check_architecture()" in text and "explain_crash()" in text and "check-runtime)" in text
    assert 'TEMPLATE_HASH="' + template_hash() + '"' in text
    # ตารางไฟล์จาก controller เดิมต้องยกมาครบ — verify-files ยังเทียบขนาด/sha ได้
    assert '"44766155936"' in text and '"50dc0856abd4a8ecea97a47ffa197bde3ea8d7d0f49d0e1fea7f71c97e8a70d1"' in text
    assert "resolve/06756566a4b4a29d0dee62ccb405914a15fdf80d/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf" in text
    assert 'MMPROJ_FILE="${MMPROJ_FILE-mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf}"' in text
    # ของเดิมเก็บไว้ · ค่าต่อเครื่องไม่แตะ
    replaced = list(directory.glob(f"{SLUG}-single.sh.replaced-*"))
    assert len(replaced) == 1 and replaced[0].read_text(encoding="utf-8") == CONTROLLER_051
    assert (directory / "bundle.env").read_text(encoding="utf-8") == 'API_PORT="${API_PORT:-8011}"\nCTX_SIZE="${CTX_SIZE:-65536}"\n'
    assert (directory / "bundle.args").read_text(encoding="utf-8") == "--flash-attn on\n"
    assert (directory / "cluster.env").read_text(encoding="utf-8") == "MASTER_IP=10.0.0.1\n"
    # profile ใหม่: ค่าที่อนุมัติไว้คงเดิม + คีย์ใหม่ (template_hash · gguf_architecture จาก architecture เดิม)
    profile = yaml.safe_load((directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["template_hash"] == template_hash() and profile["generated_by"] != "lmds 0.5.1"
    assert profile["serving"]["context"] == 262144 and profile["serving"]["max_num_seqs"] == 1
    assert profile["model"]["gguf_architecture"] == "qwen4exp" and profile["runtime"]["native_build"] is True
    assert profile["runtime"]["min_llamacpp"]["commit"] == "6c84c7d5d"
    assert profile["target"]["max_safe_context"] == 262144 and profile["target"]["verdict"] == "fits"
    assert profile["features"]["multimodal"]["projector_files"] == ["mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"]
    assert (directory / "PACKAGE_SHA256SUMS").exists()
    # รอบสองด้วย --if-older = ตรงแล้ว ไม่ทำซ้ำ
    (again,) = refresh_bundles([SLUG], if_older=True)
    assert again.action == "current" and len(list(directory.glob("*.replaced-*"))) == 1


def test_bundles_refresh_skips_adopted_and_reports(tmp_path, monkeypatch):
    from lmds.fleet.refresh import format_result, refresh_bundles

    profile = {"generated_by": "lmds adopt", "adopted": True,
               "model": {"id": "/models/coder-next", "served_name": "coder-next", "artifact_type": "safetensors"},
               "runtime": {"engine": "vllm", "image": "vllm/vllm-openai:latest"}, "topology": "single"}
    directory = _bundle(tmp_path, monkeypatch, profile=profile,
                        controller="#!/usr/bin/env bash\ncase $1 in\n  start) : ;;\nesac\n", slug="coder-next", mode="docker")
    (result,) = refresh_bundles(["coder-next"])
    assert result.action == "adopted" and "ไม่มี template" in format_result(result)
    assert not list(directory.glob("*.replaced-*")) and (directory / "coder-next-single.sh").read_text().startswith("#!/usr/bin/env bash\ncase")


def test_bundles_refresh_reports_old_profiles_that_need_an_online_rebuild(tmp_path, monkeypatch):
    """profile 0.3.0/0.4.0 ขาดคีย์ (dgx-veerasiam, msi-5) — ต้องรายงาน ไม่ล้มทั้งงาน และไม่แตะไฟล์"""
    from lmds.fleet.refresh import format_result, refresh_bundles

    profile = {"generated_by": "lmds 0.3.0", "model": {"id": "unsloth/Old-GGUF", "served_name": "old", "artifact_type": "gguf",
                                                      "selected_gguf": "old.gguf"},
               "runtime": {"engine": "llamacpp", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda"}, "topology": "single",
               "serving": {"context": 8192}}
    directory = _bundle(tmp_path, monkeypatch, profile=profile,
                        controller="#!/usr/bin/env bash\nSCRIPT_VERSION=\"${SCRIPT_VERSION:-0.3.0}\"\ncase $1 in\n  download) : ;;\n  start) : ;;\nesac\n",
                        slug="old")
    results = refresh_bundles(None)   # --all
    result = next(r for r in results if r.slug == "old")
    assert result.action == "needs-online" and "lmds rebuild" in format_result(result), format_result(result)
    assert not list(directory.glob("*.replaced-*"))
    assert not result.ok


def test_bundles_refresh_cli_and_local_api(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from lmds.cli.main import app

    directory = _bundle(tmp_path, monkeypatch)
    runner = CliRunner()
    done = runner.invoke(app, ["bundles", "refresh", "--all", "--if-older"])
    assert done.exit_code == 0, done.output
    assert "regenerate แล้ว" in done.output and "0.5.1 →" in done.output
    assert len(list(directory.glob("*.replaced-*"))) == 1
    # ไม่มี bundle = ไม่ล้ม (สคริปต์ install บน node เรียกเสมอ)
    monkeypatch.setenv("LMDS_BUNDLE_DIRS", str(tmp_path / "empty"))
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "empty-run"))
    assert runner.invoke(app, ["bundles", "refresh", "--all", "--if-older"]).exit_code == 0

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    monkeypatch.setenv("LMDS_BUNDLE_DIRS", str(tmp_path / "bundles"))
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    client = TestClient(create_app())
    answer = client.post(f"/api/models/{SLUG}/regenerate", json={}).json()
    assert answer["action"] == "refreshed" and "regenerate แล้ว" in answer["line"], answer
    assert len(list(directory.glob("*.replaced-*"))) == 2
    assert client.post("/api/models/nope/regenerate", json={}).status_code == 404


def test_regenerate_on_node_runs_bundles_refresh_there(monkeypatch):
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from lmds.nodes import Node, add
    from lmds.web import create_app

    add(Node(name="spark-worker", host="10.0.0.7", user="ops"))
    seen = []
    monkeypatch.setattr("lmds.nodes.run", lambda node, command, timeout=0: seen.append(command)
                        or SimpleNamespace(exit_code=0, stdout="x: regenerate แล้ว", stderr=""))
    client = TestClient(create_app())
    answer = client.post(f"/api/nodes/spark-worker/models/{SLUG}/regenerate").json()
    assert answer["exit_code"] == 0 and seen == [f"lmds bundles refresh {SLUG}"]
    assert client.post("/api/nodes/spark-worker/models/x%27%3Bid%3B%27/regenerate").status_code == 400
