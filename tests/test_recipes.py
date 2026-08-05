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
    assert plan.serving.kv_cache_dtype == "nvfp4_ds_mla"
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


def test_recipe_env_reaches_the_container():
    """บาง runtime เลือก backend ผิดถ้าไม่ตั้ง env — DeepSeek V4 ตายที่
    determine_available_memory ถ้าไม่ปิด CUDA-graph memory profiler"""
    plan = plan_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    assert plan.serving.extra_env.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0"


def test_recipe_env_is_rendered_into_the_controller(tmp_path):
    import subprocess

    from lmds.brain.orchestrator import harden_plan
    from lmds.generator import render_bundle

    report = report_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")

    assert "EXTRA_DOCKER_ENV=(" in text
    assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in text
    assert subprocess.run(["bash", "-n", str(bundle.controller)]).returncode == 0


def test_every_recipe_flag_either_survives_or_asks_for_approval():
    """flag ที่ allowlist ไม่รู้จักจะถูก harden ตัดทิ้ง — สูตรก็ไร้ผลโดยไม่มีอาการอะไรบอก
    เคสจริง: DeepSeek V4 เสีย --compilation-config PIECEWISE ไปเงียบ ๆ แล้ว start ไม่ผ่านสามรอบ

    ข้อยกเว้นที่ถูกต้อง: flag อ่อนไหวอย่าง --trust-remote-code ต้องไม่ถูกเปิดเงียบ ๆ
    แต่ก็ต้องไม่หายไปเฉย ๆ — ต้องโผล่ใน flags_needing_approval ให้ผู้ใช้ตัดสิน
    """
    from lmds.brain.orchestrator import harden_plan

    for recipe in load_catalog():
        if recipe.engine != "vllm" or not recipe.serving:
            continue
        report = report_for(recipe.match)
        fit = analyze(report, PRESETS["dgx-spark-stacked"])
        plan = rule_based_plan(report, fit)
        wanted = [f for f in plan.serving.extra_flags if f.startswith("--")]
        hardened = harden_plan(plan, report, fit)
        kept = " ".join(hardened.serving.extra_flags)
        pending = " ".join(hardened.flags_needing_approval)

        for flag in wanted:
            assert flag in kept or flag in pending, \
                f"{recipe.match}: {flag} หายไปเงียบ ๆ ตอน harden"


def test_deepseek_compilation_config_is_not_dropped():
    """ตัวที่ทำให้ start ไม่ผ่านจริง — ต้องอยู่ในคำสั่งที่รันจริง ไม่ใช่รออนุมัติ"""
    from lmds.brain.orchestrator import harden_plan

    report = report_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    hardened = harden_plan(rule_based_plan(report, fit), report, fit)
    flags = " ".join(hardened.serving.extra_flags)
    assert "--compilation-config" in flags and "PIECEWISE" in flags
    assert "--moe-backend" in flags


def test_llm_path_also_gets_the_recipe():
    """ลูกค้าที่ "มี" API key ต้องไม่ได้ผลแย่กว่าคนที่ไม่มี — สูตรที่รันผ่านจริงชนะสิ่งที่ LLM ค้นมา
    ในส่วนที่ทับกัน ส่วนที่สูตรไม่ครอบคลุม LLM ยังคุมเหมือนเดิม"""
    from lmds.brain.orchestrator import build_plan
    from lmds.brain.plan_schema import DeploymentPlan

    report = report_for("nvidia/DeepSeek-V4-Flash-NVFP4")
    fit = analyze(report, PRESETS["dgx-spark-stacked"])

    class FakeProvider:
        name, model = "fake", "m1"

        def complete_json(self, system, user):
            # LLM เดา image และ kv-cache ผิด (image ทั่วไป, auto) แต่ตั้ง served name ที่ดีมา
            plan = rule_based_plan(report, fit).model_copy(deep=True)
            plan.runtime.image_ref = "vllm/vllm-openai:latest"
            plan.serving.kv_cache_dtype = "auto"
            plan.serving.extra_flags = []
            plan.served_model_name = "ds-v4-from-llm"
            return plan.model_dump_json()

    plan = build_plan(report, fit, FakeProvider())
    assert isinstance(plan, DeploymentPlan)
    # สูตรทับสิ่งที่ LLM เดาผิด
    assert plan.runtime.image_ref.startswith("ghcr.io/anemll/")
    assert plan.serving.kv_cache_dtype == "nvfp4_ds_mla"
    assert "--compilation-config" in " ".join(plan.serving.extra_flags)
    # แต่ไม่ไปยุ่งกับสิ่งที่สูตรไม่ได้พูดถึง
    assert plan.served_model_name == "ds-v4-from-llm"


@pytest.mark.parametrize(
    "repo_id, tool_parser",
    [
        ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3_coder"),
        ("Qwen/Qwen3.6-35B-A3B-FP8", "qwen3_xml"),
        ("Qwen/Qwen3.6-35B-A3B-NVFP4", "qwen3_xml"),
        ("QuantTrio/MiniMax-M2-AWQ", "minimax_m2"),
        ("openai/gpt-oss-120b", "openai"),
        ("stepfun-ai/Step-3.7-Flash-FP8", "step3p5"),
    ],
)
def test_new_families_get_the_tool_parser_that_was_tested(repo_id, tool_parser):
    """Qwen 3.5 กับ 3.6 เปลี่ยนรูปแบบ tool call — ใช้ parser ผิดแล้ว tool call หลุดเป็นข้อความ"""
    recipe = find_recipe(repo_id)
    assert recipe is not None, f"ไม่มีสูตรของ {repo_id}"
    assert recipe.tool_calling.get("parser") == tool_parser


def test_attention_backend_from_a_recipe_is_not_left_pending_approval():
    """backend ของ attention เป็นของเฉพาะรุ่น+สถาปัตยกรรม ไม่ใช่การจูน — ต้องผ่าน allowlist

    ถ้าไม่ผ่าน มันจะไปกอง flags_needing_approval แล้วผู้ใช้ที่ deploy แบบ -y จะไม่ได้ flag นั้น
    ซึ่งเป็นเคสที่พังเงียบ: bundle ออกมาครบ แต่ start แล้วช้ากว่าที่ทดสอบไว้หรือตายตอน init
    """
    from lmds.brain.orchestrator import harden_plan
    from lmds.brain.rulebased import apply_recipe

    report = report_for("Qwen/Qwen3.5-35B-A3B-FP8")
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = harden_plan(
        apply_recipe(rule_based_plan(report, fit), find_recipe(report.repo_id), fit.memory_model.value),
        report, fit,
    )
    # harden รวม flag กับค่าเป็นสตริงเดียว — เทียบด้วย prefix ไม่ใช่ความเท่ากันของ element
    assert any(f.startswith("--attention-backend") for f in plan.serving.extra_flags), plan.serving.extra_flags
    assert not [f for f in plan.flags_needing_approval if "attention-backend" in f]


def test_imported_recipes_say_who_actually_tested_them():
    """เอาสูตรของโปรเจกต์อื่นมาต้องบอกตรง ๆ ว่าใครทดสอบ — ไม่ใช่เขียนเหมือนเรารันเอง"""
    imported = [r for r in load_catalog() if "eugr/spark-vllm-docker" in r.source]
    assert imported, "ไม่มีสูตรที่นำเข้ามา"
    for recipe in imported:
        assert "โปรเจกต์ต้นทาง" in recipe.validated_on, recipe.match
        assert "MIT" in recipe.source, recipe.match
