"""เทสหน้า lmds list — คอลัมน์ port / context / support (ฟีเจอร์ที่โมเดลรองรับ)"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.fleet import bundle_profile, feature_summary, profile_context
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _report():
    return ModelReport(
        repo_id="Qwen/Qwen3-32B",
        revision_sha="sha-list",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB,
        shard_count=17,
        context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
        has_chat_template=True,
    )


def _render(tmp_path, with_features=True):
    report = _report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    if with_features:
        plan.tool_calling.enabled = True
        plan.tool_calling.parser = "hermes"
        plan.reasoning.enabled = True
        plan.reasoning.parser = "deepseek_r1"
    return render_bundle(plan, report, fit, tmp_path)


def test_feature_summary_variants():
    assert feature_summary(None) == "text"
    assert feature_summary({}) == "text"
    assert feature_summary({"features": {"tool_calling": {"enabled": True}}}) == "tools"
    prof = {"features": {
        "tool_calling": {"enabled": True},
        "reasoning": {"enabled": True},
        "multimodal": {"modalities": ["image"]},
    }}
    assert feature_summary(prof) == "tools, reasoning, image"


def test_profile_helpers_from_rendered_bundle(tmp_path):
    bundle = _render(tmp_path)
    prof = bundle_profile(str(bundle.controller))
    assert prof is not None
    assert profile_context(prof) == 32768
    assert feature_summary(prof) == "tools, reasoning"
    assert (prof.get("runtime") or {}).get("engine") == "vllm"


def test_bundle_profile_missing_returns_none(tmp_path):
    assert bundle_profile(str(tmp_path / "nope.sh")) is None
    assert bundle_profile("") is None


def _write_meta(run_root: Path, slug: str, controller: Path, port: int):
    d = run_root / slug
    d.mkdir(parents=True)
    (d / "server.meta").write_text(
        f"slug={slug}\nmodel=Qwen/Qwen3-32B\nmodel_id=Qwen/Qwen3-32B\n"
        f"engine=vllm\nmode=docker\nport={port}\ncontainer=lmds-{slug}\n"
        f"pid_file=\ncontroller={controller}\nstarted_at=2026-07-24T00:00:00\n",
        encoding="utf-8",
    )


def test_list_shows_port_context_support(tmp_path, monkeypatch):
    bundle = _render(tmp_path)
    run_root = tmp_path / "run"
    _write_meta(run_root, bundle.controller.parent.name, bundle.controller, 8000)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(run_root))

    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    out = result.stdout
    assert "context" in out and "support" in out.lower()
    assert "8000" in out            # port
    assert "32,768" in out          # context (มี comma)
    assert "tools" in out and "reasoning" in out  # support


def test_list_hint_shows_one_command_per_line_including_repair_remove(tmp_path, monkeypatch):
    """คำใบ้ต้องอ่านง่าย: หนึ่งคำสั่งต่อบรรทัด และต้องมี repair/remove ด้วย

    เดิมเอาหลายคำสั่งมาต่อกันด้วย · ในบรรทัดเดียว ผู้ใช้อ่านแล้วเข้าใจผิดว่าต้องพิมพ์ทั้งบรรทัด
    """
    bundle = _render(tmp_path)
    run_root = tmp_path / "run"
    slug = bundle.controller.parent.name
    _write_meta(run_root, slug, bundle.controller, 8000)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(run_root))

    out = CliRunner().invoke(app, ["list"]).stdout
    for command in ("start", "stop", "restart", "logs", "enable", "repair", "remove"):
        assert f"lmds {command} {slug}" in out or f"lmds {command} {slug} -f" in out, command

    # แต่ละคำสั่งต้องอยู่คนละบรรทัด ไม่ใช่ต่อกันด้วย ·
    command_lines = [ln for ln in out.splitlines() if ln.strip().startswith("lmds ")]
    assert len(command_lines) >= 7
    assert all("·" not in ln for ln in command_lines)


def test_anthropic_command_reaches_ui_allowlists():
    """คำสั่งใหม่ต้องอยู่ในลิสต์ทั้งสองที่ ไม่งั้นปุ่มไม่ขึ้น (CLI) หรือหน้าเว็บสั่งไม่ได้"""
    from lmds.inventory import KNOWN_COMMANDS
    from lmds.web.jobs import ALLOWED

    assert "test-anthropic" in KNOWN_COMMANDS
    assert "test-anthropic" in ALLOWED
