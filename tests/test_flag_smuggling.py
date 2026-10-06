"""flag ที่ซ่อนอยู่หลัง flag ที่อนุญาตใน item เดียวกัน ต้องถูกตรวจด้วย — allowlist เคยดูแค่ token แรก

ผู้ตรวจยืนยัน 2026-10-06 (hub 6e2b474): `split_flags(VLLM, ["--dtype float16 --trust-remote-code"])` → allowed ทั้งก้อน
ไม่มีอะไรถูกส่งไปขออนุมัติ ทั้งที่ --trust-remote-code จงใจอยู่นอก allowlist (PRD §9.2: ต้องอนุมัติทุกครั้ง) และ controller
แยกคำ EXTRA_SERVE_ARGS ด้วยช่องว่างแล้วส่งทุกตัวเข้า engine · แผนจาก LLM คือ input ที่ไม่น่าเชื่อถือ
"""

import subprocess

import pytest

from lmds.brain.allowlists import coalesce_flag_tokens, split_flags
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import Engine
from lmds.brain.rulebased import QWEN3_RERANKER_HF_OVERRIDES, rule_based_plan
from lmds.fit import PRESETS, analyze
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport

GIB = 1024**3


@pytest.mark.parametrize("engine, item, allowed, needs_approval", [
    (Engine.VLLM, "--dtype float16 --trust-remote-code", ["--dtype float16"], ["--trust-remote-code"]),
    (Engine.VLLM, "--dtype=float16 --trust-remote-code --allowed-origins *",
     ["--dtype=float16"], ["--trust-remote-code", "--allowed-origins *"]),
    (Engine.VLLM, "--enable-prefix-caching --trust-remote-code", ["--enable-prefix-caching"], ["--trust-remote-code"]),
    (Engine.LLAMACPP, "--threads 4 --rpc 10.0.0.9:50052", ["--threads 4"], ["--rpc 10.0.0.9:50052"]),
    (Engine.LLAMACPP, "--threads 4 -ngl 99 --rpc x", ["--threads 4"], ["-ngl 99", "--rpc x"]),
])
def test_every_flag_inside_an_item_is_checked(engine, item, allowed, needs_approval):
    assert split_flags(engine, [item]) == (allowed, needs_approval)
    assert split_flags(engine, coalesce_flag_tokens([item])) == (allowed, needs_approval)


@pytest.mark.parametrize("engine, item", [
    (Engine.VLLM, '--speculative-config {"method": "mtp", "num_speculative_tokens": 2}'),   # JSON มีช่องว่าง
    (Engine.VLLM, QWEN3_RERANKER_HF_OVERRIDES),
    (Engine.VLLM, "--chat-template /cache/my templates/qwen.jinja"),                         # path มีช่องว่าง
    (Engine.VLLM, "--limit-mm-per-prompt image=4,video=1"),
    (Engine.LLAMACPP, "--n-gpu-layers -1"),                                                  # ค่าลบไม่ใช่ flag
    (Engine.LLAMACPP, "--rope-scaling yarn"),
])
def test_a_value_with_spaces_or_json_is_still_one_item_untouched(engine, item):
    from lmds.brain.allowlists import explode_flag_item

    assert explode_flag_item(item) == [item]
    allowed, needs_approval = split_flags(engine, [item])
    assert (allowed, needs_approval) == ([item], []) or needs_approval == [item], "ไม่ถูกแตกเป็นหลายชิ้น"


def report():
    return ModelReport(repo_id="org/some-model", revision_sha="a" * 40, artifact_type=ArtifactType.SAFETENSORS,
                       weight_bytes=10 * GIB, context_length=32768, kv_dims=KvDims(layers=32, kv_heads=8, head_dim=128),
                       trust_remote_code_files=["modeling_x.py"], has_chat_template=True)


def test_harden_sends_the_smuggled_flag_for_approval_and_strips_the_controller_owned_one():
    """สิ่งที่แผนจาก LLM ใส่มาได้: flag ที่ต้องอนุมัติ และ flag ที่ controller เป็นเจ้าของ ซ่อนหลัง flag ที่อนุญาต"""
    rep = report()
    fit = analyze(rep, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(rep, fit)
    plan.serving.extra_flags = ["--dtype float16 --trust-remote-code", "--kv-cache-dtype fp8 --tensor-parallel-size 2"]

    plan = harden_plan(plan, rep, fit)

    assert plan.serving.extra_flags == ["--dtype float16", "--kv-cache-dtype fp8"]
    assert plan.flags_needing_approval == ["--trust-remote-code"]
    assert any("--tensor-parallel-size 2" in w and "controller" in w for w in plan.warnings)


def test_the_rendered_controller_does_not_pass_the_unapproved_flag(tmp_path):
    """รัน bash อ่านค่าตั้งต้นของ EXTRA_SERVE_ARGS จาก controller ที่ render จริง — ไม่ grep ซอร์ส"""
    rep = report()
    fit = analyze(rep, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(rep, fit)
    plan.serving.extra_flags = ["--dtype float16 --trust-remote-code"]
    plan = harden_plan(plan, rep, fit)
    bundle = render_bundle(plan, rep, fit, tmp_path / "bundles")

    # ทุกบรรทัดที่ต่อ argv ของ `vllm serve` ใน controller → ให้ bash ประกอบ array จริงแล้วพิมพ์ออกมา
    lines = [ln.strip() for ln in bundle.controller.read_text().splitlines()
             if ln.strip().startswith("serve_args+=(") and "$(" not in ln]
    script = "serve_args=()\n" + "\n".join(lines) + '\nprintf "%s\\n" "${serve_args[@]}"\n'
    argv = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()
    assert "--dtype" in argv and "float16" in argv, argv
    assert not any("trust-remote-code" in arg for arg in argv)
