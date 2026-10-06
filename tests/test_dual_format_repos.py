"""repo ที่มีทั้ง .safetensors และ .gguf — ทุกชั้นต้องเห็นตรงกันว่าแผนนี้ใช้ฝั่งไหน

เคสจริง 2026-10-06 (audit ~90 repo บน Hub · hub 6e2b474) — เดิม repo ที่มี .safetensors แม้ไฟล์เดียวกลายเป็น
"mixed" → vLLM เสมอ เลือก GGUF ไม่ได้ และ bundle จะโหลดทั้ง repo:
- `LiquidAI/LFM2.5-2.6B-GGUF` (GGUF 8 ไฟล์ + `qad/model.safetensors` · ไม่มี config.json ที่ราก):
  `lmds generate … --no-llm` ได้ bundle vLLM `SHARD_FILES=("qad/model.safetensors")` ผ่านทุก gate ·
  ใส่ `--gguf Q4_K_M` ได้ "ไม่ใช่ repo GGUF"
- `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` (.gguf 105 ไฟล์ 1.6 TB + `GSQ-3bit/`) เหมือนกัน
- หน้าเว็บบังคับให้เลือกไฟล์ GGUF แล้ววางแผน vLLM อยู่ดี ("แก้ engine จาก llamacpp เป็น vllm ตาม artifact จริง")
- `OBLITERATUS/Qwen3.8-27B-OBLITERATED` (GGUF 7 quant + mmproj · safetensors ข้าง ๆ เป็น MLX): ชี้ไฟล์
  `…-Q4_K_M.gguf` ตรง ๆ ยังได้ exit 1 "⛔ ไม่รองรับ … MLX … จะดาวน์โหลด 111.1 GB"

repo สองรูปแบบของจริงที่ใช้เป็นตัวยืน: `CMSManhattan/JiRackUltra_1b` (model.safetensors + config.json ที่ราก + GGUF 5
ไฟล์) และ `mixedbread-ai/mxbai-embed-large-v1` (embedding · GGUF ไฟล์เดียวใน gguf/)
"""

import json

import pytest
import yaml
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan, rule_based_plan
from lmds.brain.orchestrator import harden_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector import ArtifactType
from tests.real_hub import RealHub, flat, load, shard_files_of

runner = CliRunner()

LFM = "LiquidAI/LFM2.5-2.6B-GGUF"
HERETIC = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
JIRACK = "CMSManhattan/JiRackUltra_1b"
MXBAI = "mixedbread-ai/mxbai-embed-large-v1"
OBLITERATED = "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
THESTAGE = "TheStageAI/gemma-4-E4B-it"
MBAKGUN = "mbakgun/Qwen2.5-Coder-14B-n8n-Workflow-Generator"
ALL = (LFM, HERETIC, JIRACK, MXBAI, OBLITERATED, THESTAGE, MBAKGUN)

JIRACK_GGUF = "JiRackUltra_1b_Q4_K_M.gguf"
OBLITERATED_GGUF = "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
SPARK = PRESETS["dgx-spark-single"]


def size_of(repo: str, name: str) -> int:
    return next(s["size"] for s in load(repo)["info"]["siblings"] if s["rfilename"] == name)


def blob(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/blob/main/{name}"


@pytest.fixture
def hub(isolated_config, monkeypatch):
    fake = RealHub(*ALL)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    return fake


def profile_of(out) -> dict:
    return yaml.safe_load(next(out.glob("*/MODEL_PROFILE.yaml")).read_text())


# ── (a) .safetensors ที่เสิร์ฟไม่ได้ ไม่ทำให้ repo GGUF กลายเป็นงานของ vLLM ──────────────────────────


@pytest.mark.parametrize("repo, variants, stray", [
    (LFM, 8, "qad/"),                  # safetensors อยู่แต่ในโฟลเดอร์ย่อย · ไม่มี config.json ที่ราก
    (HERETIC, 103, "GSQ-3bit/"),       # .gguf 105 ไฟล์ = weight 103 + mmproj/mtp 2
    (MBAKGUN, 2, "MLX"),               # library_name=mlx · รากเป็น adapter · merged-hf/ กับ mlx-q4/ เป็นโฟลเดอร์ย่อย
])
def test_a_gguf_repo_with_stray_safetensors_is_a_gguf_repo(repo, variants, stray):
    report = RealHub(repo).inspect(repo)
    assert report.artifact_type is ArtifactType.GGUF, "เดิมเป็น mixed → vLLM"
    assert len([v for v in report.gguf_variants if not v.is_mmproj and not v.is_mtp]) == variants
    assert not report.safetensor_shards, "ไม่มีไฟล์ safetensors ไหนไปถึง SHARD_FILES ของ bundle"
    assert stray in report.format_note and "llama.cpp" in report.format_note
    assert analyze(report, SPARK).engine_assumed == "llamacpp"


def test_cli_generate_picks_a_gguf_from_the_repo_that_used_to_say_not_a_gguf_repo(hub, tmp_path):
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", LFM, "--target", "dgx-spark-single", "--no-llm",
                                 "--gguf", "Q4_K_M", "--output", str(out)])
    assert result.exit_code == 0, result.output
    profile = profile_of(out)
    assert profile["runtime"]["engine"] == "llamacpp"
    assert profile["model"]["artifact_type"] == "gguf"
    assert profile["model"]["selected_gguf"] == "LFM2.5-2.6B-Q4_K_M.gguf"
    assert profile["model"]["weight_bytes"] == size_of(LFM, "LFM2.5-2.6B-Q4_K_M.gguf")
    assert not hub.weight_requests


def test_cli_generate_without_a_choice_asks_for_one_instead_of_building_a_vllm_bundle(hub, tmp_path):
    """เดิม exit 0 + bundle vLLM ที่ผ่านทุก gate ทั้งที่ repo ไม่มี config.json ที่ราก (ทั้ง repo 33.5 GB)"""
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", LFM, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)])
    assert result.exit_code == 1
    said = flat(result.output)
    assert "--gguf" in said and "LFM2.5-2.6B-Q8_0.gguf" in said
    assert not out.exists()


# ── (b) repo สองรูปแบบของจริง: ไม่เลือก = safetensors/vLLM · เลือกไฟล์ GGUF = llama.cpp ───────────────


def test_a_dual_format_repo_defaults_to_its_safetensors_checkpoint():
    report = RealHub(JIRACK).inspect(JIRACK)
    assert report.artifact_type is ArtifactType.MIXED
    assert report.selected_gguf is None
    assert [s.filename for s in report.safetensor_shards] == ["model.safetensors"]
    assert report.weight_bytes == size_of(JIRACK, "model.safetensors")
    assert report.architecture == "Qwen2ForCausalLM"
    assert "safetensors" in report.format_note and "GGUF 5 ไฟล์" in report.format_note and "--gguf" in report.format_note

    fit = analyze(report, SPARK)
    assert fit.engine_assumed == "vllm"
    plan = build_plan(report, fit, None)
    assert plan.runtime.engine is Engine.VLLM
    assert plan.artifact_type is ArtifactType.MIXED and plan.selected_gguf is None
    assert report.format_note in plan.warnings, "แผนต้องบอกเองว่าใช้ฝั่งไหน อีกฝั่งคืออะไร"


def test_pointing_at_a_gguf_file_means_llamacpp_with_that_file_as_the_weight():
    hub = RealHub(JIRACK)
    report = hub.inspect(blob(JIRACK, JIRACK_GGUF))
    header = load(JIRACK)["gguf"][JIRACK_GGUF]["metadata"]

    assert report.artifact_type is ArtifactType.GGUF
    assert report.selected_gguf == JIRACK_GGUF
    assert report.weight_bytes == size_of(JIRACK, JIRACK_GGUF), "เดิมยังเป็นขนาด safetensors แม้เลือก GGUF แล้ว"
    assert report.gguf_architecture == header["general.architecture"]
    assert report.context_length == header[f"{header['general.architecture']}.context_length"]
    assert not report.safetensor_shards
    assert "GGUF" in report.format_note and "safetensors" in report.format_note

    fit = analyze(report, SPARK)
    assert fit.engine_assumed == "llamacpp"
    plan = build_plan(report, fit, None)
    assert plan.runtime.engine is Engine.LLAMACPP
    assert plan.artifact_type is ArtifactType.GGUF and plan.selected_gguf == JIRACK_GGUF
    assert report.format_note in plan.warnings


@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG])
def test_choosing_vllm_or_sglang_on_a_dual_repo_uses_the_safetensors_checkpoint(engine):
    report = RealHub(JIRACK).inspect(JIRACK)
    plan = build_plan(report, analyze(report, SPARK), None, engine=engine)
    assert plan.runtime.engine is engine and plan.selected_gguf is None


def test_asking_for_llamacpp_without_naming_a_file_explains_how_to_pick_one():
    """เดิม harden แก้กลับเป็น vllm เงียบ ๆ — ผู้ใช้ที่ตั้งใจเลือก llama.cpp ได้ bundle vLLM"""
    report = RealHub(JIRACK).inspect(JIRACK)
    fit = analyze(report, SPARK)
    for planner in (rule_based_plan, lambda r, f, e: build_plan(r, f, None, engine=e)):
        with pytest.raises(PlanError) as err:
            planner(report, fit, Engine.LLAMACPP)
        said = str(err.value)
        assert "--gguf" in said and JIRACK_GGUF in said


def test_an_llm_plan_cannot_carry_a_gguf_file_into_a_vllm_bundle():
    report = RealHub(JIRACK).inspect(JIRACK)
    fit = analyze(report, SPARK)
    plan = rule_based_plan(report, fit)
    plan.runtime.engine = Engine.LLAMACPP          # สิ่งที่ LLM เสนอได้: หยิบ engine/ไฟล์จากรายการ GGUF
    plan.selected_gguf = JIRACK_GGUF
    hardened = harden_plan(plan, report, fit)
    assert hardened.runtime.engine is Engine.VLLM
    assert hardened.selected_gguf is None and hardened.artifact_type is ArtifactType.MIXED


def test_cli_generate_on_a_dual_repo_builds_whichever_side_was_asked_for(hub, tmp_path):
    safetensors_out, gguf_out = tmp_path / "st", tmp_path / "gguf"
    base = ["generate", JIRACK, "--target", "dgx-spark-single", "--no-llm"]

    result = runner.invoke(app, [*base, "--output", str(safetensors_out)])
    assert result.exit_code == 0, result.output
    profile = profile_of(safetensors_out)
    assert profile["runtime"]["engine"] == "vllm" and profile["model"]["selected_gguf"] is None
    assert shard_files_of(next(safetensors_out.glob("*/*-single.sh")).read_text()) == ["model.safetensors"]
    assert "GGUF5ไฟล์" in flat(result.output), "ต้องเห็นว่ามีอีกทาง"

    result = runner.invoke(app, [*base, "--gguf", "Q4_K_M", "--output", str(gguf_out)])
    assert result.exit_code == 0, result.output
    profile = profile_of(gguf_out)
    assert profile["runtime"]["engine"] == "llamacpp"
    assert profile["model"]["artifact_type"] == "gguf"
    assert profile["model"]["selected_gguf"] == JIRACK_GGUF
    assert profile["model"]["weight_bytes"] == size_of(JIRACK, JIRACK_GGUF)
    assert not hub.weight_requests


def test_a_single_gguf_in_a_dual_repo_is_not_selected_behind_the_users_back():
    """mxbai-embed-large-v1: GGUF ไฟล์เดียวใน gguf/ — เดิมถูกเลือกอัตโนมัติ (selected_gguf) ทั้งที่แผนเป็น vLLM"""
    report = RealHub(MXBAI).inspect(MXBAI)
    assert report.artifact_type is ArtifactType.MIXED
    assert report.selected_gguf is None
    assert report.task == "embed"
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.runtime.engine is Engine.VLLM and plan.task == "embed" and plan.selected_gguf is None


# ── หน้าเว็บ ─────────────────────────────────────────────────────────────────────────────────────────


def web_plan(model, **kw):
    from lmds.web import deploy as dep

    return dep.analyze(model, target=kw.pop("target", "dgx-spark-single"), no_llm=True, **kw)["plan"]


def test_web_plans_vllm_for_a_dual_repo_without_making_the_user_pick_a_gguf_first(hub):
    plan = web_plan(JIRACK)
    assert plan["engine"] == "vllm" and plan["selected_gguf"] is None
    assert any("GGUF 5 ไฟล์" in w for w in plan["warnings"])


def test_web_selected_gguf_means_llamacpp(hub):
    """เดิม: เลือกไฟล์แล้วได้แผน vLLM พร้อม "แก้ engine จาก llamacpp เป็น vllm ตาม artifact จริง" """
    plan = web_plan(JIRACK, selected_gguf=JIRACK_GGUF)
    assert plan["engine"] == "llamacpp" and plan["selected_gguf"] == JIRACK_GGUF
    assert plan["fit"]["weights_gb"] == pytest.approx(size_of(JIRACK, JIRACK_GGUF) / 1024**3, abs=0.06)
    assert not any("แก้ engine" in w for w in plan["warnings"])


def test_web_engine_llamacpp_on_a_dual_repo_asks_which_file_or_takes_the_only_one(hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        web_plan(JIRACK, engine="llamacpp")
    assert err.value.kind == "choose-gguf"
    assert JIRACK_GGUF in [v["filename"] for v in err.value.extra["variants"]]

    plan = web_plan(MXBAI, engine="llamacpp")
    assert plan["engine"] == "llamacpp" and plan["selected_gguf"] == "gguf/mxbai-embed-large-v1-f16.gguf"


def test_web_refuses_a_gguf_file_together_with_engine_vllm(hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        web_plan(JIRACK, selected_gguf=JIRACK_GGUF, engine="vllm")
    assert err.value.kind == "input" and "GGUF" in err.value.message


def test_web_stacked_target_refuses_the_gguf_side_but_not_the_safetensors_side(hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        web_plan(JIRACK, target="dgx-spark-stacked", selected_gguf=JIRACK_GGUF)
    assert err.value.kind == "input" and "llama.cpp" in err.value.message

    plan = web_plan(JIRACK, target="dgx-spark-stacked")
    assert plan["engine"] == "vllm" and plan["topology"] == "stacked"


def test_web_gguf_repo_with_stray_safetensors_goes_through_the_normal_choose_flow(hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        web_plan(LFM)
    assert err.value.kind == "choose-gguf" and len(err.value.extra["variants"]) == 8
    plan = web_plan(LFM, selected_gguf="LFM2.5-2.6B-Q4_K_M.gguf")
    assert plan["engine"] == "llamacpp"


# ── MLX ข้าง ๆ GGUF: รูปแบบที่ไม่รองรับตัดสิทธิ์เฉพาะฝั่ง safetensors ────────────────────────────────


def test_a_gguf_file_next_to_mlx_weights_is_planned_on_llamacpp(hub):
    """เดิม exit 1 "⛔ ไม่รองรับ … MLX … จะดาวน์โหลด 111.1 GB" ทั้งที่ผู้ใช้ชี้ไฟล์ GGUF 16.8 GB"""
    result = runner.invoke(app, ["plan", blob(OBLITERATED, OBLITERATED_GGUF), "--target", "dgx-spark-single",
                                 "--no-llm", "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["runtime"]["engine"] == "llamacpp"
    assert plan["selected_gguf"] == OBLITERATED_GGUF and plan["artifact_type"] == "gguf"
    assert any("MLX" in w for w in plan["warnings"]), "ยังต้องบอกว่า safetensors ข้าง ๆ ใช้ไม่ได้เพราะอะไร"
    assert not hub.weight_requests


def test_mlx_weights_do_not_disqualify_the_ggufs_in_the_same_repo():
    report = RealHub(OBLITERATED).inspect(OBLITERATED)
    assert report.library_name == "mlx"
    assert report.artifact_type is ArtifactType.GGUF
    assert report.unsupported_format is None
    assert len([v for v in report.gguf_variants if not v.is_mmproj]) == 7
    assert "MLX" in report.format_note and "library_name=mlx" in report.format_note

    fit = analyze(report, SPARK)
    assert fit.verdict is not Verdict.UNSUPPORTED and fit.engine_assumed == "llamacpp"


def test_cli_points_at_the_repos_own_ggufs_when_none_is_selected(hub, tmp_path):
    """ยังไม่เลือกไฟล์ = บอกว่า repo นี้ใช้ GGUF ของมันเองได้ และเลือกอย่างไร — ไม่ใช่ "ไม่รองรับ MLX" """
    out = tmp_path / "bundles"
    base = ["generate", OBLITERATED, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)]
    result = runner.invoke(app, base)
    assert result.exit_code == 1
    said = flat(result.output)
    assert "ไม่รองรับ" not in said
    assert "--gguf" in said and OBLITERATED_GGUF in said

    result = runner.invoke(app, [*base, "--gguf", "Q4_K_M"])
    assert result.exit_code == 0, result.output
    assert profile_of(out)["runtime"]["engine"] == "llamacpp"


def test_cli_inspect_shows_the_gguf_side_and_why_the_safetensors_are_unused(hub):
    result = runner.invoke(app, ["inspect", OBLITERATED, "--target", "dgx-spark-single"])
    assert result.exit_code == 0
    said = flat(result.output)
    assert "ไม่รองรับ" not in said and "MLX" in said and "llama.cpp" in said


def test_the_only_gguf_of_an_mlx_repo_is_used_directly():
    """TheStageAI/gemma-4-E4B-it: library_name=mlx · GGUF ไฟล์เดียวใน s-gguf-pq-ple/"""
    report = RealHub(THESTAGE).inspect(THESTAGE)
    assert report.artifact_type is ArtifactType.GGUF
    assert report.selected_gguf == "s-gguf-pq-ple/body.gguf"
    assert report.gguf_architecture == "gemma4"
    assert report.weight_bytes == size_of(THESTAGE, "s-gguf-pq-ple/body.gguf")
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.runtime.engine is Engine.LLAMACPP
