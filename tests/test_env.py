"""env ของ engine — สองปัญหาที่อยู่บนเส้นทางเดียวกัน

1. env ที่สูตรตั้งไว้ไม่เคยถึง controller ของ single-node (render เฉพาะ stacked)
   → สูตรที่เขียนไว้เพื่อกัน start ไม่ขึ้น กลับไม่มีผลอะไรเลย
2. extra_env ไม่เคยผ่าน allowlist และ harden_plan() ไม่แตะเลย
   → LLM ตั้ง LD_PRELOAD/PYTHONPATH ได้ ซึ่งคือการรันโค้ดใน container

แก้ข้อ 1 อย่างเดียวโดยไม่แก้ข้อ 2 = ขยายช่องโหว่ให้กว้างขึ้น จึงต้องมาคู่กัน
"""

from __future__ import annotations

import pytest

from lmds.brain.allowlists import is_allowed_env, split_env
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import Engine
from lmds.brain.rulebased import rule_based_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def safetensors_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-8B",
        revision_sha="sha-env-test",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=16 * GIB,
        shard_count=4,
        context_length=32768,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def gguf_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF",
        revision_sha="sha-env-gguf",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB,
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def _plan_with_env(report, env, target="dgx-spark-single"):
    fit = analyze(report, PRESETS[target])
    plan = rule_based_plan(report, fit)
    plan.serving.extra_env = dict(env)
    return harden_plan(plan, report, fit), report, fit


# ── allowlist ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("engine,name", [
    (Engine.VLLM, "VLLM_MARLIN_USE_ATOMIC_ADD"),
    (Engine.VLLM, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
    (Engine.VLLM, "NCCL_IB_HCA"),
    (Engine.VLLM, "TORCH_CUDA_ARCH_LIST"),
    (Engine.VLLM, "CUDA_VISIBLE_DEVICES"),
    (Engine.VLLM, "OMP_NUM_THREADS"),
    (Engine.LLAMACPP, "GGML_CUDA_FORCE_MMQ"),
])
def test_engine_variables_are_allowed(engine, name):
    """สูตรที่รันผ่านจริงต้องตั้ง env พวกนี้ได้ — ไม่งั้น allowlist ทำให้ระบบใช้ไม่ได้"""
    assert is_allowed_env(engine, name)


@pytest.mark.parametrize("name", [
    "LD_PRELOAD",         # โหลด .so เข้าโปรเซส = รันโค้ดใน container
    "LD_LIBRARY_PATH",
    "PYTHONPATH",         # แทรกโมดูลทับของจริง
    "PYTHONSTARTUP",
    "BASH_ENV",
    "PATH",               # สลับ binary ที่ถูกเรียก
    "IFS",
])
def test_loader_variables_are_rejected(name):
    """ทั้งหมดนี้คือการรันโค้ดใน container ซึ่งกฎข้อ 2 บอกว่าต้องผ่านการอนุมัติ

    ต่างจากไฟล์ runtime ภายนอกตรงที่ไฟล์มี URL+SHA ให้ตรวจ ส่วน env ไม่มีอะไรให้ตรวจเลย
    จึงปฏิเสธไปตรง ๆ ดีกว่าเปิดช่องให้กดผ่าน
    """
    assert not is_allowed_env(Engine.VLLM, name)


@pytest.mark.parametrize("name", [
    "HF_HUB_TOKEN", "VLLM_API_KEY", "NCCL_SECRET", "TORCH_PASSWORD",
])
def test_secretish_names_are_rejected_even_with_a_valid_prefix(name):
    """secret ห้ามมาจาก LLM หรือจากไฟล์แคตตาล็อก (กฎข้อ 4)

    HF_HUB_TOKEN ผ่าน prefix HF_HUB_ ได้ถ้าไม่มีข้อนี้ — แล้วก็จะไปโผล่ใน bundle
    """
    assert not is_allowed_env(Engine.VLLM, name)


@pytest.mark.parametrize("name", ["vllm_lower", "VLLM-DASH", "1VLLM", "", "VLLM_A B"])
def test_malformed_names_are_rejected(name):
    assert not is_allowed_env(Engine.VLLM, name)


@pytest.mark.parametrize("name", [
    "NCCL_ENV_PLUGIN",
    "NCCL_NET_PLUGIN",
    "NCCL_PROFILER_PLUGIN",
    "VLLM_ALLOW_INSECURE_SERIALIZATION",
    "VLLM_LOGGING_CONFIG_PATH",
    "VLLM_PLUGINS",
])
def test_code_loading_and_insecure_engine_env_are_rejected(name):
    """prefix ถูกไม่ได้แปลว่าปลอดภัย: บางตัวโหลด .so/config/pickle ได้"""
    assert not is_allowed_env(Engine.VLLM, name)


def test_env_is_scoped_to_the_selected_engine():
    assert is_allowed_env(Engine.VLLM, "VLLM_USE_FLASHINFER_SAMPLER")
    assert not is_allowed_env(Engine.LLAMACPP, "VLLM_USE_FLASHINFER_SAMPLER")
    assert is_allowed_env(Engine.LLAMACPP, "GGML_CUDA_FORCE_MMQ")
    assert not is_allowed_env(Engine.VLLM, "GGML_CUDA_FORCE_MMQ")


def test_values_with_newlines_are_rejected():
    """ค่าที่มีขึ้นบรรทัดใหม่แทรก -e ตัวถัดไปได้ทันทีที่มีใครเอาไปต่อสตริง"""
    allowed, rejected = split_env(Engine.VLLM, {
        "VLLM_USE_FLASHINFER_SAMPLER": "1\nNCCL_DEBUG=TRACE",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1\r0",
    })
    assert allowed == {}
    assert rejected == ["VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_USE_FLASHINFER_SAMPLER"]


def test_values_are_coerced_to_strings():
    """catalog.yaml เขียน 1 เปล่า ๆ ได้ — YAML จะ parse เป็น int แล้ว shlex.quote พัง"""
    allowed, _ = split_env(Engine.VLLM, {"VLLM_USE_FLASHINFER_SAMPLER": 1})
    assert allowed == {"VLLM_USE_FLASHINFER_SAMPLER": "1"}


def test_documented_structured_values_are_accepted():
    env = {
        "CUDA_VISIBLE_DEVICES": "GPU-8932f937,0,-1",
        "TORCH_CUDA_ARCH_LIST": "8.0 8.6+PTX",
        "NCCL_IB_HCA": "=mlx5_0:1,^mlx5_1:2",
    }
    allowed, rejected = split_env(Engine.VLLM, env)
    assert allowed == env
    assert rejected == []


@pytest.mark.parametrize("name,value", [
    ("VLLM_USE_FLASHINFER_SAMPLER", "yes"),
    ("OMP_NUM_THREADS", "0"),
    ("NCCL_IB_HCA", "/tmp/plugin.so"),
    ("VLLM_NVFP4_GEMM_BACKEND", "../../evil"),
    ("VLLM_NVFP4_GEMM_BACKEND", "not-a-real-backend"),
    ("CUDA_VISIBLE_DEVICES", "0; touch /tmp/pwn"),
    ("TORCH_CUDA_ARCH_LIST", "8.0; touch /tmp/pwn"),
])
def test_invalid_values_are_rejected(name, value):
    allowed, rejected = split_env(Engine.VLLM, {name: value})
    assert allowed == {}
    assert rejected == [name]


# ── harden ───────────────────────────────────────────────────────────────────

def test_harden_drops_env_outside_the_allowlist():
    plan, _, _ = _plan_with_env(safetensors_report(), {
        "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
        "LD_PRELOAD": "/tmp/evil.so",
    })
    assert plan.serving.extra_env == {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"}
    assert any("LD_PRELOAD" in w for w in plan.warnings)


def test_harden_keeps_a_clean_env_untouched():
    env = {"VLLM_NVFP4_GEMM_BACKEND": "marlin", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    plan, _, _ = _plan_with_env(safetensors_report(), env)
    assert plan.serving.extra_env == env
    assert not any("ตัด env" in w for w in plan.warnings)


def test_renderer_fails_closed_if_hardening_was_skipped(tmp_path):
    """renderer เป็น public API: ข้าม harden ก็ต้องสร้าง bundle ที่มี env อันตรายไม่ได้"""
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit)
    plan.serving.extra_env = {"LD_PRELOAD": "/tmp/evil.so"}

    with pytest.raises(ValueError, match="LD_PRELOAD"):
        render_bundle(plan, report, fit, tmp_path)
    assert not any(tmp_path.iterdir())


# ── env ต้องไปถึง controller จริง ─────────────────────────────────────────────

def test_env_reaches_the_single_node_vllm_controller(tmp_path):
    """เคสจริงที่พังเงียบ: สูตร DeepSeek V4 ตั้ง VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
    ไว้พร้อมคอมเมนต์ว่า "ตัวที่ทำให้ start ไม่ผ่านถ้าไม่ตั้ง" แต่ controller ของ single-node
    ไม่เคย render env เลย — MODEL_PROFILE.yaml บันทึกไว้ แผนก็มี แต่ตอนรันไม่มีใครตั้งให้
    """
    env = {"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"}
    plan, report, fit = _plan_with_env(safetensors_report(), env)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")

    assert "EXTRA_ENV=(" in text
    assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in text
    # ต้องถูกส่งเข้า container จริง ไม่ใช่แค่ประกาศตัวแปรทิ้งไว้
    assert 'docker_args+=(-e "$pair")' in text


def test_env_reaches_the_llamacpp_controller_in_both_modes(tmp_path):
    """llama.cpp มีสองโหมด — native ไม่มี container ให้ส่ง -e ต้อง export เอง"""
    env = {"GGML_CUDA_FORCE_MMQ": "1"}
    plan, report, fit = _plan_with_env(gguf_report(), env)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")

    assert "GGML_CUDA_FORCE_MMQ=1" in text
    assert 'export "${EXTRA_ENV[@]}"' in text        # native
    assert 'docker_args+=(-e "$pair")' in text       # docker


def test_no_env_means_no_leftover_machinery(tmp_path):
    """ไม่มี env = ต้องไม่มีตัวแปร/ลูปว่าง ๆ ค้างในสคริปต์"""
    plan, report, fit = _plan_with_env(safetensors_report(), {})
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "EXTRA_ENV" not in text


def test_bundle_with_env_passes_every_gate(tmp_path):
    from lmds.packager import write_checksums
    from lmds.validator import all_passed, run_gates

    plan, report, fit = _plan_with_env(
        safetensors_report(), {"VLLM_NVFP4_GEMM_BACKEND": "marlin"}
    )
    bundle = render_bundle(plan, report, fit, tmp_path)
    write_checksums(bundle.directory)
    results = run_gates(bundle.directory, include_checksums=True)
    assert all_passed(results), [f"{r.name}: {r.detail}" for r in results if not r.passed]


def test_every_env_in_the_catalog_passes_the_allowlist():
    """สูตรในแคตตาล็อกต้องใช้งานได้จริงหลังเพิ่ม allowlist

    ทรงเดียวกับเทสที่บังคับว่า flag ของสูตรต้องผ่าน allowlist หรือไปรออนุมัติ
    """
    from lmds.recipes import load_catalog

    for recipe in load_catalog():
        if not recipe.env:
            continue
        allowed, rejected = split_env(
            Engine(recipe.engine), {k: str(v) for k, v in recipe.env.items()}
        )
        assert not rejected, f"{recipe.match}: env ไม่ผ่าน allowlist — {rejected}"
