"""Audit stacked (multi-node) — ชุดวางแผน: inspector → fit → brain → recipes → web deploy.analyze

ลูกค้ารายงาน 2026-09-04: "analyze ล้ม" · "deploy หลายเครื่องไม่เคยขึ้น" · "DeepSeek-V4-Flash-NVFP4 ไม่ผ่าน"
ไล่จากเคสจริงบน hub (spark-head + spark-worker) แล้วเขียนเทสให้ล้มก่อนแก้ทุกข้อ — เหมือน
tests/test_audit_backend.py · แต่ละเทสคือข้อบกพร่องหนึ่งข้อพร้อมเหตุผลว่าทำไมถึงพัง
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from lmds.brain import build_plan
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import DeploymentPlan, Engine, Topology
from lmds.brain.rulebased import rule_based_plan
from lmds.fit import PRESETS, Verdict, analyze
from lmds.fit.analyzer import GIB
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from lmds.recipes import find_recipe, load_catalog, synced_path

DEEPSEEK = "nvidia/DeepSeek-V4-Flash-NVFP4"
QWEN_CODER = "ucbye/Qwen3-Coder-Next-NVFP4-GB10"
QWEN_235B = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
DIGEST = "sha256:365447a3b5a172e96c50e69f761a15e45fff2a7487f46172a84a4cf806f25f5d"
AVAROK_PINNED = f"avarok/dgx-vllm-nvfp4-kernel@{DIGEST}"


# ── รายงานสังเคราะห์จากค่าจริงที่ inspect บน hub 2026-09-04 ─────────────────────────────
def deepseek_report(**overrides) -> ModelReport:
    """nvidia/DeepSeek-V4-Flash-NVFP4: MoE 256/6 · 43 layers · 156.7 GiB · ctx 1M · ไม่มี chat template"""
    base = dict(
        repo_id=DEEPSEEK, revision_sha="e3cd60e7de98e9867116860d522499a728de1cf9",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=168_281_985_176, shard_count=46,
        context_length=1_048_576, quantization="fp8", architecture="DeepseekV4ForCausalLM",
        model_type="deepseek_v4", kv_dims=KvDims(layers=43, kv_heads=1, head_dim=512),
        moe_experts=256, moe_experts_active=6, has_chat_template=False,
    )
    base.update(overrides)
    return ModelReport(**base)


def qwen235b_report(**overrides) -> ModelReport:
    """Qwen/Qwen3-235B-A22B-Instruct-2507-FP8: 94 layers · GQA 4 · 220.2 GiB · ctx 262k"""
    base = dict(
        repo_id=QWEN_235B, revision_sha="sha-235b", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=236_426_193_880, shard_count=24, context_length=262_144, quantization="fp8",
        architecture="Qwen3MoeForCausalLM", kv_dims=KvDims(layers=94, kv_heads=4, head_dim=128),
        moe_experts=128, moe_experts_active=8, has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def qwen_coder_report(**overrides) -> ModelReport:
    base = dict(
        repo_id=QWEN_CODER, revision_sha="sha-coder", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=int(42.7 * GIB), shard_count=10, context_length=262_144, quantization="nvfp4",
        architecture="Qwen3NextForCausalLM", kv_dims=KvDims(layers=48, kv_heads=2, head_dim=256),
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def gguf_report(**overrides) -> ModelReport:
    base = dict(repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf", artifact_type=ArtifactType.GGUF,
                weight_bytes=5 * GIB, selected_gguf="Qwen3-8B-Q4_K_M.gguf", has_chat_template=True)
    base.update(overrides)
    return ModelReport(**base)


def _write_synced(entries: list[dict]) -> None:
    synced_path().parent.mkdir(parents=True, exist_ok=True)
    synced_path().write_text(yaml.safe_dump({"version": 1, "recipes": entries}, allow_unicode=True),
                             encoding="utf-8")
    load_catalog.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_catalog():
    load_catalog.cache_clear()
    yield
    load_catalog.cache_clear()


# ── 1. registry: image ที่ตรึง digest (`repo@sha256:…`) ถูกตัดสินว่า "ไม่มีอยู่จริง" ────────
class _Registry:
    """registry ปลอม: token ให้เสมอ · manifests/<ref> ตอบ 200 เฉพาะ ref ที่รู้จัก"""

    def __init__(self, known: set[str]):
        self.known = known
        self.asked: list[str] = []

    def get(self, url, **kw):
        return SimpleNamespace(status_code=200, json=lambda: {"token": "t"})

    def request(self, method, url, headers=None):
        self.asked.append(url)
        ref = url.rsplit("/manifests/", 1)[-1]
        ok = ref in self.known
        return SimpleNamespace(status_code=200 if ok else 404,
                               headers={"Docker-Content-Digest": ref if ref.startswith("sha256:") else DIGEST})

    def close(self):
        pass


def test_split_ref_keeps_a_digest_pinned_image_whole():
    """เคสจริง 2026-09-04 (ucbye/Qwen3-Coder-Next-NVFP4-GB10): สูตรที่ sync มาใช้
    `avarok/dgx-vllm-nvfp4-kernel@sha256:3654…` ซึ่งคือ image ที่รันอยู่จริงบน spark-head วันนี้
    แต่ split_ref ตัดที่ ':' ตัวสุดท้าย → repo กลายเป็น `…kernel@sha256` และ "tag" = เลข digest
    registry ตอบ 404 → แผนสรุปว่า image ไม่มีอยู่จริง แล้วเปลี่ยนเป็น nvcr ที่ไม่มี FP4 kernel
    """
    from lmds.brain.registry import split_ref

    assert split_ref(AVAROK_PINNED) == ("registry-1.docker.io", "avarok/dgx-vllm-nvfp4-kernel", DIGEST)
    assert split_ref("ghcr.io/anemll/dspark-vllm-gx10@" + DIGEST) == ("ghcr.io", "anemll/dspark-vllm-gx10", DIGEST)
    # ของเดิมต้องไม่เปลี่ยน
    assert split_ref("nvcr.io/nvidia/vllm:26.05-py3") == ("nvcr.io", "nvidia/vllm", "26.05-py3")
    assert split_ref("vllm/vllm-openai") == ("registry-1.docker.io", "vllm/vllm-openai", "latest")


def test_a_pinned_digest_is_checked_as_a_digest_not_as_a_tag(monkeypatch):
    from lmds.brain import registry

    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    fake = _Registry(known={DIGEST, "latest"})
    assert registry.tag_exists(AVAROK_PINNED, client=fake) is True
    assert fake.asked[-1].endswith(f"/v2/avarok/dgx-vllm-nvfp4-kernel/manifests/{DIGEST}")
    # digest ที่ไม่มีจริงยังต้องถูกจับได้
    assert registry.tag_exists("avarok/dgx-vllm-nvfp4-kernel@sha256:" + "0" * 64, client=fake) is False


def test_a_pinned_digest_resolves_to_itself_without_asking_the_registry(monkeypatch):
    """image ที่ตรึงมาแล้วไม่ต้องถาม registry ว่าชี้ไปไหน — มันคือคำตอบอยู่แล้ว
    (เครื่อง air-gapped ที่ใช้ digest ของ image ในเครื่องต้องไม่เสียเวลารอ timeout)"""
    from lmds.brain import registry

    monkeypatch.delenv(registry.SKIP_ENV, raising=False)

    class _Boom:
        def get(self, *a, **k):
            raise AssertionError("ไม่ควรถาม registry")

        request = get

        def close(self):
            pass

    assert registry.resolve_digest(AVAROK_PINNED, client=_Boom()) == DIGEST


# ── 2. harden: image ของสูตร (พิสูจน์แล้ว) ถูกลดรุ่นเงียบ ๆ เป็น image ที่ไม่มี FP4 kernel ────
def _stacked_fit(report):
    return analyze(report, PRESETS["dgx-spark-stacked"])


def test_a_recipe_image_survives_a_registry_miss(monkeypatch):
    """registry ตอบ "ไม่มี" กับ image ของสูตร — สูตรคือหลักฐานว่ามันรันจริงบนเครื่อง (อาจเป็น build
    ในเครื่อง/ digest ที่ registry ไม่ตอบ) · เดิมเปลี่ยนเป็น nvcr:26.05 เงียบ ๆ ซึ่ง NVFP4 บน SM121
    ตายตั้งแต่ start (`cvt .e2m1x2 not supported on sm_121`) · ต้องคง image ไว้แล้วเตือน"""
    from lmds.brain import registry

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: False)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    _write_synced([{"match": QWEN_CODER, "engine": "vllm", "image": AVAROK_PINNED,
                    "source": "team", "validated_on": "spark-head", "topology": "single"}])
    report = qwen_coder_report()
    fit = _stacked_fit(report)
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == AVAROK_PINNED
    assert not plan.runtime.image_ref.startswith("nvcr.io")
    assert any("สูตร" in w and "registry" in w for w in plan.warnings), plan.warnings


def test_nvfp4_on_a_spark_never_falls_back_to_the_ngc_image(monkeypatch):
    """image ที่ LLM เสนอไม่มีจริง + โมเดล NVFP4 + DGX Spark → ต้องได้ image ที่มี FP4 kernel
    ของ sm_121 (ตัวที่ spark04/veerasiam รันอยู่) ไม่ใช่ nvcr 26.05 ที่ไม่มี"""
    from lmds.brain import registry
    from lmds.brain.rulebased import SPARK_NVFP4_VLLM_IMAGE

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: False)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    report = qwen_coder_report()
    fit = _stacked_fit(report)
    plan = rule_based_plan(report, fit)
    plan.runtime.image_ref = "vllm/vllm-openai:v0.6.3.ss"     # LLM มโน tag
    hardened = harden_plan(plan, report, fit)
    assert hardened.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    assert "@sha256:" in SPARK_NVFP4_VLLM_IMAGE
    assert not hardened.runtime.image_ref.startswith("nvcr.io")


def test_nvfp4_on_a_spark_without_a_recipe_gets_the_proven_image_and_marlin_env():
    """NVFP4 บน GB10 ที่ไม่มีสูตร: rule-based เคยให้ nvcr:26.05 + ไม่มี env → ptxas ตายตอน start
    ค่าที่พิสูจน์แล้ว 2026-09-03 บน spark-head คือ vllm-openai (cu130) + env marlin สี่ตัว"""
    from lmds.brain.rulebased import SPARK_NVFP4_ENV, SPARK_NVFP4_VLLM_IMAGE

    report = qwen_coder_report(repo_id="someone/Other-Model-NVFP4", quantization="nvfp4")
    fit = _stacked_fit(report)
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.runtime.image_ref == SPARK_NVFP4_VLLM_IMAGE
    for key, value in SPARK_NVFP4_ENV.items():
        assert plan.serving.extra_env.get(key) == value, (key, plan.serving.extra_env)
    # เครื่องการ์ดแยกไม่เกี่ยว — kernel ของ sm_121 เป็นเรื่องของ Spark เท่านั้น
    rtx = harden_plan(rule_based_plan(report, analyze(report, PRESETS["rtx-5090"])), report,
                      analyze(report, PRESETS["rtx-5090"]))
    assert "VLLM_NVFP4_GEMM_BACKEND" not in rtx.serving.extra_env


def test_a_pinned_recipe_image_renders_as_one_reference(tmp_path, monkeypatch):
    """image_ref ที่มี @sha256 อยู่แล้ว + image_pin → template ต่อเป็น `repo@sha256@sha256:…` ไม่ได้"""
    from lmds.brain import registry
    from lmds.generator import render_bundle

    monkeypatch.setattr(registry, "tag_exists", lambda ref, client=None: True)
    monkeypatch.setattr(registry, "resolve_digest", lambda ref, client=None: DIGEST)
    monkeypatch.delenv(registry.SKIP_ENV, raising=False)
    _write_synced([{"match": QWEN_CODER, "engine": "vllm", "image": AVAROK_PINNED,
                    "source": "team", "validated_on": "spark-head", "topology": "single"}])
    report = qwen_coder_report()
    fit = _stacked_fit(report)
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert f"VLLM_IMAGE=\"${{VLLM_IMAGE:-{AVAROK_PINNED}}}\"" in text
    assert "@sha256@" not in text


# ── 3. recipes: สูตรที่ sync มาทับของ catalog ทั้งก้อน → env/flag ของ DeepSeek หายหมด ──────
def test_synced_recipe_keeps_catalog_serving_and_env_it_is_silent_about():
    """เคสจริง 2026-09-04 บน hub: recipes-synced.yaml มี DeepSeek-V4 แค่ `serving: {gpu_util, max_num_seqs}`
    (อ่านจาก header ของ controller) แล้วทับ entry ของ catalog ทั้งก้อน → kv_cache_dtype nvfp4_ds_mla ·
    block_size 256 · compilation_config PIECEWISE · env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS หายหมด
    แผนออกมา extra_flags=[] extra_env={} ทั้ง single และ stacked · vLLM ตาย "Expected 7 but got 8 arguments"
    """
    _write_synced([{"match": DEEPSEEK, "engine": "vllm", "image": "ghcr.io/anemll/dspark-vllm-gx10:0.1.1",
                    "serving": {"gpu_memory_utilization": 0.85, "max_num_seqs": 6},
                    "source": "team@bc22407", "validated_on": "2x spark", "topology": "stacked"}])
    recipe = find_recipe(DEEPSEEK)
    assert recipe.serving["max_num_seqs"] == 6                       # ของที่ sync มาชนะ
    assert recipe.serving["kv_cache_dtype"] == "nvfp4_ds_mla"        # ของ catalog ที่มันเงียบไว้ต้องอยู่
    assert recipe.serving["block_size"] == 256
    assert recipe.env.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0"
    assert recipe.image_for == ["unified"]

    report = deepseek_report()
    for target in ("dgx-spark-stacked", "dgx-spark-single"):
        fit = analyze(report, PRESETS[target])
        plan = harden_plan(rule_based_plan(report, fit), report, fit)
        assert plan.serving.kv_cache_dtype == "nvfp4_ds_mla", target
        assert "--block-size 256" in " ".join(plan.serving.extra_flags), (target, plan.serving.extra_flags)
        assert "--compilation-config" in " ".join(plan.serving.extra_flags), target
        assert plan.serving.extra_env.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0", target
        assert not plan.flags_needing_approval, plan.flags_needing_approval


def test_synced_engine_env_and_parsers_reach_the_plan():
    """controller ที่ publish มาเก็บ env ไว้ที่ `engine_env: "K=V K=V"` และ parser ที่ `tool_parser:`
    (ระดับบนสุด) — Recipe ไม่มีฟิลด์ชื่อนี้จึงถูกทิ้งตอนโหลด · ucbye/Qwen3-Coder-Next-NVFP4-GB10
    รันบน spark-head ได้ก็เพราะ env สี่ตัวนี้ แต่แผนที่ hub สร้างให้ไม่มีเลย"""
    _write_synced([{"match": QWEN_CODER, "engine": "vllm", "image": AVAROK_PINNED,
                    "serving": {"gpu_memory_utilization": 0.85, "max_num_seqs": 4},
                    "engine_env": "VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_TEST_FORCE_FP8_MARLIN=1 "
                                  "VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_MARLIN_USE_ATOMIC_ADD=1",
                    "tool_parser": "qwen3_coder", "extra_args": "--enable-prefix-caching",
                    "source": "team@bc22407", "validated_on": "spark-head", "topology": "single"}])
    recipe = find_recipe(QWEN_CODER)
    assert recipe.env == {"VLLM_NVFP4_GEMM_BACKEND": "marlin", "VLLM_TEST_FORCE_FP8_MARLIN": "1",
                          "VLLM_USE_FLASHINFER_MOE_FP4": "0", "VLLM_MARLIN_USE_ATOMIC_ADD": "1"}
    assert recipe.tool_calling.get("parser") == "qwen3_coder" and recipe.tool_calling.get("enabled")
    assert recipe.serving["kv_cache_dtype"] == "fp8"                 # จาก catalog

    report = qwen_coder_report()
    fit = _stacked_fit(report)
    plan = harden_plan(rule_based_plan(report, fit), report, fit)
    assert plan.topology is Topology.STACKED
    assert plan.serving.extra_env.get("VLLM_NVFP4_GEMM_BACKEND") == "marlin"
    assert plan.serving.extra_env.get("VLLM_MARLIN_USE_ATOMIC_ADD") == "1"
    assert "--enable-prefix-caching" in plan.serving.extra_flags
    assert plan.tool_calling.parser == "qwen3_coder"


# ── 4. harden: flag ที่ controller stacked เป็นเจ้าของ ต้องไม่หลุดมาจาก LLM ─────────────────
def test_controller_owned_parallelism_flags_are_stripped_from_the_plan():
    """controller ตั้ง --tensor-parallel-size/--nnodes/--node-rank เองจาก target · LLM ใส่
    `--tensor-parallel-size 1` มาด้วย (อยู่ใน allowlist) → vLLM เอาตัวหลังชนะ = TP=1 บน 2 เครื่อง
    head รอ worker ที่ไม่มีวันมา"""
    report = deepseek_report()
    fit = _stacked_fit(report)
    plan = rule_based_plan(report, fit)
    plan.serving.extra_flags += ["--tensor-parallel-size 1", "--nnodes 1", "--node-rank 0",
                                 "--distributed-executor-backend ray", "--enable-prefix-caching"]
    hardened = harden_plan(plan, report, fit)
    joined = " ".join(hardened.serving.extra_flags)
    for owned in ("--tensor-parallel-size", "--nnodes", "--node-rank", "--distributed-executor-backend"):
        assert owned not in joined, joined
    assert "--enable-prefix-caching" in hardened.serving.extra_flags
    assert any("controller" in w and "tensor-parallel" in w for w in hardened.warnings), hardened.warnings


def test_the_llm_is_told_how_many_nodes_and_who_sets_tensor_parallel():
    """prompt บอกแค่ชื่อ target · LLM ไม่รู้ว่า stacked = 2 เครื่อง TP ข้ามเครื่อง และ controller
    ตั้ง flag พวกนั้นเอง จึงชอบเติม --tensor-parallel-size มาให้"""
    from lmds.brain.prompts import build_user_prompt

    report = deepseek_report()
    fit = _stacked_fit(report)
    text = build_user_prompt({"model": {}, "fit": fit.model_dump(mode="json")}, fit.target_name, "")
    assert "2 nodes" in text or "node_count=2" in text, text[-600:]
    assert "--tensor-parallel-size" in text and "controller" in text.lower(), text[-600:]


# ── 5. fit: ตัวเลขต่อเครื่องของ stacked ─────────────────────────────────────────────────
def test_stacked_fit_reports_per_node_memory_and_budgets_the_comm_buffer():
    """หน้าเว็บ/CLI เห็นแค่ budget รวม 227 GB — ผู้ใช้ตัดสินใจไม่ได้ว่า "เครื่องละเท่าไร" และ NCCL
    buffer ของ TP ข้ามเครื่องไม่เคยถูกหักจริง (มีแต่โน้ต) · ต้องมีตัวเลขต่อเครื่องในรายงาน"""
    from lmds.fit.analyzer import STACKED_COMM_BUFFER_GB_PER_NODE

    report = deepseek_report()
    fit = _stacked_fit(report)
    assert fit.node_count == 2
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert fit.per_node_weights_gb == pytest.approx(156.7 / 2, abs=0.2)
    assert fit.per_node_budget_gb == pytest.approx(fit.budget_gb / 2, abs=0.1)
    assert fit.comm_buffer_gb == pytest.approx(STACKED_COMM_BUFFER_GB_PER_NODE * 2, abs=0.01)
    # budget รวมต้องหัก buffer แล้ว: 2×128 − 2×12 (OS) − 2×2.5 (engine) − 2×buffer
    assert fit.budget_gb == pytest.approx(256 - 24 - 5 - STACKED_COMM_BUFFER_GB_PER_NODE * 2, abs=0.1)
    assert fit.per_node_kv_budget_gb == pytest.approx(fit.kv_budget_gb / 2, abs=0.1)
    # single ไม่มีของพวกนี้
    single = analyze(qwen_coder_report(), PRESETS["dgx-spark-single"])
    assert single.node_count == 1 and single.comm_buffer_gb == 0.0
    assert single.per_node_budget_gb == single.budget_gb


def test_235b_fp8_needs_four_sparks_and_the_report_says_which_preset():
    """220 GiB บน 2×128 GB ไม่พอ — ทางเลือกต้องบอกชื่อ preset ที่พอ (dgx-spark-stacked-4)
    ไม่ใช่แค่ "ใช้ stacked" ทั้งที่กำลังวิเคราะห์ stacked อยู่แล้ว"""
    report = qwen235b_report()
    two = analyze(report, PRESETS["dgx-spark-stacked"])
    assert two.verdict is Verdict.NEEDS_SMALLER_QUANT
    assert any("dgx-spark-stacked-4" in a for a in two.alternatives), two.alternatives
    four = analyze(report, PRESETS["dgx-spark-stacked-4"])
    assert four.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert four.node_count == 4 and four.per_node_weights_gb == pytest.approx(220.2 / 4, abs=0.2)
    # 4 เครื่องแล้วยังไม่พอ ต้องไม่ชี้ไป preset ที่ใหญ่กว่าซึ่งไม่มี
    huge = analyze(qwen235b_report(weight_bytes=int(600 * GIB)), PRESETS["dgx-spark-stacked-4"])
    assert not any("stacked" in a for a in huge.alternatives), huge.alternatives


# ── 6. web deploy.analyze: ทุกทางที่ stacked ล้มต้องเป็น 4xx พร้อมข้อความที่ทำต่อได้ ─────────
@pytest.fixture
def stacked_pair(monkeypatch):
    """spark-head + spark-worker ในทะเบียน + inventory ในแคช (ไม่ยิง SSH)"""
    from lmds.nodes import Node, add
    from lmds.web import state

    add(Node(name="spark-head", host="10.2.1.195", user="ops", cluster_ip="10.100.153.1", site="hq"))
    add(Node(name="spark-worker", host="10.2.1.194", user="ops", cluster_ip="10.100.153.2", site="hq"))
    add(Node(name="spark-far", host="10.9.9.9", user="ops", cluster_ip="10.100.200.2", site="branch"))
    add(Node(name="spark-solo", host="10.2.1.190", user="ops", cluster_ip="10.100.153.3", site="hq", stack=False))
    add(Node(name="spark-nocluster", host="10.2.1.191", user="ops", site="hq"))
    for name, ports, used in (("spark-head", [8000, 8001], 40.0), ("spark-worker", [8000, 8002, 8003], 10.0)):
        state.STORE.set_node(name, {
            "host": {"hostname": name, "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": used}],
                     "memory_model": "unified", "foreign": []},
            "models": [{"slug": f"m{p}", "port": p, "running": i == 0} for i, p in enumerate(ports)],
        })
    return monkeypatch


def _mock_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: report)


def _analyze(**kw):
    from lmds.web import deploy

    return deploy.analyze(kw.pop("model", DEEPSEEK), no_llm=True, **kw)


def _expect(kind: str, needle: str, **kw):
    from lmds.web import deploy

    with pytest.raises(deploy.DeployError) as caught:
        _analyze(**kw)
    assert caught.value.kind == kind, (caught.value.kind, caught.value.message)
    assert needle in caught.value.message, caught.value.message
    return caught.value


def test_stacked_target_on_a_fleet_machine_requires_a_worker(stacked_pair):
    """หน้าเว็บส่ง target stacked โดยไม่มี worker ได้ (ช่อง worker ซ่อนอยู่/ยังไม่เลือก) → เดิมวิเคราะห์ผ่าน
    200 ด้วยแผน 2 เครื่องที่ไม่รู้ว่าเครื่องที่สองคือใคร แล้วไปล้มตอน push/cluster.env"""
    _mock_inspect(stacked_pair, deepseek_report())
    _expect("input", "worker", target="dgx-spark-stacked", machine="spark-head")


def test_worker_must_not_be_the_head(stacked_pair):
    _mock_inspect(stacked_pair, deepseek_report())
    _expect("input", "เครื่องเดียวกัน", target="dgx-spark-stacked", machine="spark-head", worker="spark-head")


def test_worker_that_cannot_stack_is_refused_with_the_reason(stacked_pair):
    _mock_inspect(stacked_pair, deepseek_report())
    _expect("input", "ไม่มีในทะเบียน", target="dgx-spark-stacked", machine="spark-head", worker="ghost")
    _expect("input", "cluster IP", target="dgx-spark-stacked", machine="spark-head", worker="spark-nocluster")
    _expect("input", "stack", target="dgx-spark-stacked", machine="spark-head", worker="spark-solo")
    _expect("input", "ไซต์", target="dgx-spark-stacked", machine="spark-head", worker="spark-far")


def test_a_worker_with_a_single_target_is_a_contradiction_not_a_silent_single_plan(stacked_pair):
    """เลือก worker แล้วแต่ target ยังเป็น single (หรือเดาจากเครื่อง = single) → เดิมได้แผน single เงียบ ๆ
    push ไป head เครื่องเดียว ผู้ใช้เข้าใจว่า deploy 2 เครื่องแล้ว"""
    _mock_inspect(stacked_pair, qwen_coder_report())
    _expect("input", "stacked", target="dgx-spark-single", machine="spark-head", worker="spark-worker")
    # ไม่ส่ง target มาเลย + มี worker = ตั้งใจ stacked → เลือก preset 2 เครื่องให้
    out = _analyze(model=QWEN_CODER, machine="spark-head", worker="spark-worker")
    assert out["plan"]["topology"] == "stacked"
    assert out["plan"]["fit"]["target"] == "dgx-spark-stacked"


def test_gguf_and_sglang_are_refused_for_stacked_at_analyze_time(stacked_pair):
    """เดิม GGUF+stacked ผ่าน analyze (200) แล้วไปตาย ValueError ตอน generate · llama.cpp ทำ TP ข้ามเครื่องไม่ได้
    · SGLang stacked ยังไม่มี controller — ต้องปฏิเสธตั้งแต่วิเคราะห์พร้อมทางออก"""
    _mock_inspect(stacked_pair, gguf_report())
    err = _expect("input", "llama.cpp", target="dgx-spark-stacked", machine="spark-head", worker="spark-worker")
    assert "single" in err.message
    _mock_inspect(stacked_pair, deepseek_report())
    _expect("input", "SGLang", target="dgx-spark-stacked", machine="spark-head", worker="spark-worker",
            engine="sglang")


def test_gated_repo_tells_the_user_exactly_how_to_add_the_token(stacked_pair):
    from lmds.inspector import AuthRequired

    def boom(source, client):
        raise AuthRequired("meta-llama/Llama-3.3-70B-Instruct", 401, had_token=False)

    stacked_pair.setattr("lmds.inspector.inspect_model", boom)
    err = _expect("gated", "lmds config set-key hf", model="meta-llama/Llama-3.3-70B-Instruct",
                  target="dgx-spark-stacked", machine="spark-head", worker="spark-worker")
    assert "HF token" in err.message or "Hugging Face token" in err.message


def test_stacked_payload_carries_per_node_memory_env_and_the_port_free_on_both(stacked_pair):
    """แผนที่หน้าเว็บได้ต้องมี: ตัวเลขต่อเครื่อง · kv_cache_dtype · extra_env · node_count · และพอร์ต
    ที่ว่างบน **ทั้งสอง** เครื่อง (8000 ถูกใช้ทั้งคู่, 8001 บน head, 8002-8003 บน worker → 8004)"""
    _mock_inspect(stacked_pair, deepseek_report())
    out = _analyze(target="dgx-spark-stacked", machine="spark-head", worker="spark-worker")
    plan, fit = out["plan"], out["plan"]["fit"]
    assert fit["node_count"] == 2
    assert fit["per_node"]["capacity_gb"] == 128.0
    assert fit["per_node"]["weights_gb"] == pytest.approx(156.7 / 2, abs=0.2)
    assert fit["per_node"]["budget_gb"] == pytest.approx(fit["budget_gb"] / 2, abs=0.1)
    assert fit["per_node"]["reserved_gb"] == 40.0          # เครื่องที่แน่นสุดเป็นตัวจำกัด
    assert fit["per_node"]["kv_at_context_gb"] == pytest.approx(fit["kv_at_context_gb"] / 2, abs=0.1)
    assert fit["reserved_gb"] == 80.0 and "spark-worker" in fit["reserved_source"]
    assert plan["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert plan["extra_env"].get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "0"
    assert plan["port"] == 8004
    assert plan["node_count"] == 2 and plan["tensor_parallel"] == 2


def test_a_plan_error_from_the_rule_based_planner_is_a_4xx_not_a_500(stacked_pair):
    """embedding + stacked → rule_based_plan ยก PlanError · analyze จับเฉพาะตอนมี LLM แล้ว fallback
    ไป rule-based ซ้ำ ซึ่งยกซ้ำ → 500 เปล่า ๆ"""
    _mock_inspect(stacked_pair, qwen_coder_report(task="embed"))
    _expect("input", "embedding", target="dgx-spark-stacked", machine="spark-head", worker="spark-worker")


def test_the_web_route_returns_a_4xx_for_every_stacked_refusal(stacked_pair):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    _mock_inspect(stacked_pair, gguf_report())
    client = TestClient(create_app())
    resp = client.post("/api/deploy/analyze", json={"model": "unsloth/Qwen3-8B-GGUF", "no_llm": True,
                                                    "target": "dgx-spark-stacked", "machine": "spark-head",
                                                    "worker": "spark-worker"})
    assert 400 <= resp.status_code < 500, resp.text
    detail = resp.json()["detail"]
    # ชั้น route (api.py) ตรวจคู่เครื่องจากกลุ่ม cluster ก่อน (kind "cluster") · ชั้น deploy.analyze ตรวจ
    # โมเดล/engine/ทะเบียน (kind "input") — ทางไหนก็ต้องเป็น 4xx พร้อมข้อความ ไม่ใช่ 500 หรือ 200 เงียบ ๆ
    assert detail["kind"] in ("input", "cluster") and detail["message"], detail
