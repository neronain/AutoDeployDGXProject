"""สูตรที่รันผ่านจริง — สิ่งที่ทดแทน LLM ให้เครื่องลูกค้าที่ไม่มี API key"""

from __future__ import annotations

import pytest

from lmds.brain.rulebased import rule_based_plan
from lmds.fit import PRESETS, analyze
from lmds.inspector.report import ArtifactType, ModelReport
from lmds.recipes import find_recipe, load_catalog


def report_for(repo: str, artifact=ArtifactType.SAFETENSORS) -> ModelReport:
    return ModelReport(repo_id=repo, revision_sha="sha", artifact_type=artifact,
                       weight_bytes=100 * 1024**3, shard_count=20, has_chat_template=True)


def plan_for(repo: str, target="dgx-spark-stacked"):
    report = report_for(repo)
    return rule_based_plan(report, analyze(report, PRESETS[target]))


def test_catalog_loads():
    catalog = load_catalog()
    assert catalog, "แคตตาล็อกว่าง — สูตรหายไปหมด"


@pytest.mark.parametrize("recipe", load_catalog(), ids=lambda r: r.match)
def test_every_recipe_states_where_it_came_from(recipe):
    """สูตรที่ไม่มีที่มาคือการเดา — ห้ามมีในแคตตาล็อก"""
    assert recipe.source, f"{recipe.match} ไม่มี source"
    assert recipe.validated_on, f"{recipe.match} ไม่ได้บอกว่ารันผ่านบนอะไร"
    assert recipe.engine, f"{recipe.match} ไม่ได้ระบุ engine"


def test_deepseek_recipe_sets_what_the_hardware_run_needed():
    plan = plan_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    assert plan.runtime.image_ref == "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
    assert plan.serving.kv_cache_dtype == "fp8"
    # ค่าที่ไม่ใช่ฟิลด์ของ Serving ต้องกลายเป็น flag ไม่ใช่หายไปเงียบ ๆ
    assert "--moe-backend" in plan.serving.extra_flags


def test_llama_recipe_enables_tool_calling():
    plan = plan_for("meta-llama/Llama-3.3-70B-Instruct")
    assert plan.tool_calling.enabled and plan.tool_calling.parser == "llama3_json"
    assert plan.tool_calling.chat_template_override


def test_recipe_never_overrides_context_from_the_target():
    """context ต้องมาจากการวิเคราะห์หน่วยความจำของเครื่องเป้าหมาย ไม่ใช่ค่าคงที่ในสูตร"""
    small = plan_for("meta-llama/Llama-3.3-70B-Instruct", target="rtx-5090")
    big = plan_for("meta-llama/Llama-3.3-70B-Instruct", target="dgx-spark-stacked")
    assert small.serving.context != big.serving.context


def test_unknown_model_still_gets_a_plan():
    plan = plan_for("some-org/never-seen-before")
    assert plan.runtime.image_ref
    assert any("rule-based" in w for w in plan.warnings)


def test_recipe_replaces_the_no_research_warning():
    """สูตรมาจากการรันจริง คำเตือน 'ไม่มีการวิจัย parser' จึงไม่ตรงและต้องหายไป"""
    plan = plan_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    assert not any("ไม่มีการวิจัย parser" in w for w in plan.warnings)
    assert any("สูตรที่รันผ่านจริง" in w for w in plan.warnings)


def test_longest_match_wins():
    """สูตรเฉพาะเจาะจงต้องมาก่อนสูตรกว้าง เมื่อ prefix ซ้อนกัน"""
    assert find_recipe("nvidia/DeepSeek-V4-Flash-NVFP4-something").match.startswith("nvidia/DeepSeek")
    assert find_recipe("") is None


def test_spark_image_is_not_reused_on_a_different_architecture():
    """image ที่ build/ทดสอบบน DGX Spark (ARM64/SM121) ใช้กับ RTX ไม่ได้ — ต้องไม่ทับเงียบ ๆ"""
    rtx = plan_for("meta-llama/Llama-3.3-70B-Instruct", target="rtx-5090")
    assert rtx.runtime.image_ref.startswith("vllm/vllm-openai")
    assert any("คนละแบบ" in w for w in rtx.warnings)

    spark = plan_for("meta-llama/Llama-3.3-70B-Instruct")
    assert spark.runtime.image_ref.startswith("nvcr.io/nvidia/vllm")


def test_catalog_file_ships_with_the_package():
    """แคตตาล็อกเป็นไฟล์ข้อมูล ไม่ใช่โค้ด — ถ้าไม่ติดไปกับ wheel เครื่องลูกค้าจะได้แคตตาล็อกว่าง
    โดยไม่มีอาการอะไรบอกเลย"""
    from lmds.recipes import CATALOG_PATH

    assert CATALOG_PATH.is_file(), f"ไม่พบ {CATALOG_PATH}"
    assert CATALOG_PATH.suffix == ".yaml"


def test_every_recipe_image_passes_the_allowlist():
    """image ที่ไม่อยู่ใน allowlist จะถูก harden แทนที่เงียบ ๆ — สูตรก็ไร้ผลทันที
    เคสจริง: DeepSeek V4 ได้ image dspark จากสูตร แล้วถูกแทนด้วย vllm-openai:latest"""
    from lmds.brain.allowlists import is_known_image
    from lmds.brain.plan_schema import Engine

    engines = {"vllm": Engine.VLLM, "llamacpp": Engine.LLAMACPP}
    for recipe in load_catalog():
        if not recipe.image or recipe.engine not in engines:
            continue
        assert is_known_image(engines[recipe.engine], recipe.image), \
            f"{recipe.match}: image {recipe.image} ไม่อยู่ใน allowlist จะถูกแทนที่เงียบ ๆ"


def test_recipe_image_survives_hardening():
    from lmds.brain.orchestrator import harden_plan

    report = report_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"


def test_recipe_for_another_engine_is_not_applied():
    """สูตร SGLang ต้องไม่ถูกยัดลง controller ของ vLLM — bundle จะผ่าน gate ทุกด่าน
    แต่ start ไม่ขึ้นเลยเพราะ image คนละ engine"""
    plan = plan_for("zai-org/GLM-4.7-Flash", target="dgx-spark-single")
    assert "sglang" not in plan.runtime.image_ref
    assert any("ยังไม่ได้ generate" in w for w in plan.warnings)
