"""โหมด rerank — reranker (cross-encoder) เสิร์ฟ /v1/rerank + /v1/score ไม่ใช่ /v1/embeddings และไม่ใช่ chat

เจ้าของ 2026-09-08: `lmds plan Qwen/Qwen3-Reranker-4B --target dgx-spark-single --no-llm` ออกมาเป็น task embed
(Qwen3-Reranker ติด tag sentence-transformers เหมือน Qwen3-Embedding) · reranker ต้องถูกเสิร์ฟเป็น score/rerank model:
vLLM --runner pooling --convert classify (+ --hf-overrides ของ Qwen3-Reranker ของแท้) · llama.cpp --reranking
"""

from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import httpx
import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine, PlanError
from lmds.brain.rulebased import (
    QWEN3_RERANKER_HF_OVERRIDES,
    is_pooling_task,
    qwen3_reranker_overrides,
    rerank_family_for,
    rule_based_plan,
)
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.inspect import GGUF_POOLING_RANK, inspect_model, task_from_config, task_of
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from lmds.resolver import parse_source
from tests.test_inspector import SHA, hub_response, make_client
from tests.test_review_templates import extract_fn

SAFE_PATH = "/usr/bin:/bin"


# ── ตรวจจับจาก Hub / config.json / GGUF ─────────────────────────────────────────────

@pytest.mark.parametrize("info, repo, expected", [
    # Qwen3-Reranker ของแท้: pipeline text-ranking แต่ tag sentence-transformers เหมือน embedding — ต้องเป็น rerank
    ({"pipeline_tag": "text-ranking", "tags": ["transformers", "sentence-transformers", "text-generation"]},
     "Qwen/Qwen3-Reranker-4B", "rerank"),
    # bge-reranker: Hub ติด text-classification + text-embeddings-inference — ชื่อบอกว่า reranker
    ({"pipeline_tag": "text-classification", "tags": ["sentence-transformers", "text-embeddings-inference"]},
     "BAAI/bge-reranker-v2-m3", "rerank"),
    ({"pipeline_tag": "text-ranking", "tags": ["reranker", "cross-encoder"]}, "jinaai/jina-reranker-v2-base-multilingual", "rerank"),
    ({"pipeline_tag": None, "tags": ["gguf"]}, "gpustack/bge-reranker-v2-m3-GGUF", "rerank"),   # GGUF ที่คนแปลง ไม่มี tag
    ({"pipeline_tag": None, "tags": ["cross-encoder"]}, "some-org/ms-marco-MiniLM-L6", "rerank"),  # tag บอกอย่างเดียว
    ({"pipeline_tag": "feature-extraction", "tags": ["sentence-transformers"]}, "Qwen/Qwen3-Embedding-4B", "embed"),
    ({"pipeline_tag": None, "tags": ["gguf"]}, "VesNFF/Qwen3-VL-Embedding-8B-GGUF", "embed"),
    ({"pipeline_tag": "text-generation", "tags": ["conversational"]}, "Qwen/Qwen3-8B", "generate"),
])
def test_rerankers_are_detected_before_the_embedding_rules(info, repo, expected):
    assert task_of(info, repo) == expected


@pytest.mark.parametrize("config, expected", [
    ({"architectures": ["XLMRobertaForSequenceClassification"], "id2label": {"0": "LABEL_0"}}, "rerank"),
    ({"architectures": ["XLMRobertaForSequenceClassification"], "num_labels": 1}, "rerank"),
    ({"architectures": ["ModernBertForSequenceClassification"]}, "rerank"),
    # จัดหมวดหลาย label (sentiment) ไม่ใช่ reranker
    ({"architectures": ["BertForSequenceClassification"], "id2label": {"0": "negative", "1": "positive"}}, None),
    ({"architectures": ["BertForSequenceClassification"], "num_labels": 3}, None),
    ({"architectures": ["Qwen3ForCausalLM"]}, None),
    ({}, None),
])
def test_a_single_label_sequence_classifier_in_config_is_a_reranker(config, expected):
    assert task_from_config(config) == expected


def test_inspect_reads_the_classifier_head_from_config_even_when_the_name_and_tags_say_nothing():
    """repo ชื่อกลาง ๆ ไม่มี tag — config.json เป็นหลักฐานตรงกว่า Hub"""
    config = {"architectures": ["XLMRobertaForSequenceClassification"], "model_type": "xlm-roberta",
              "max_position_embeddings": 8194, "id2label": {"0": "LABEL_0"}}
    files = {"config.json": json.dumps(config), "tokenizer_config.json": json.dumps({})}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response(
                [{"rfilename": "model.safetensors", "lfs": {"size": 2_200_000_000}}, {"rfilename": "config.json", "size": 700}],
                tags=["transformers", "xlm-roberta"]))
        name = request.url.path.split(f"/{SHA}/")[-1]
        if name in files:
            return httpx.Response(200, content=files[name].encode())
        return httpx.Response(404)

    report = inspect_model(parse_source("some-org/multilingual-cross-scorer"), make_client(handler))
    assert report.task == "rerank" and report.architecture == "XLMRobertaForSequenceClassification"


def test_a_gguf_converted_with_pooling_rank_is_a_reranker(monkeypatch):
    """convert_hf_to_gguf เขียน {arch}.pooling_type = 4 (RANK) ให้ไฟล์ reranker — ชื่อ repo อาจไม่บอก"""
    from lmds.inspector import inspect as inspect_mod
    from lmds.inspector.gguf import GgufInfo

    info = GgufInfo(version=3, tensor_count=200, metadata={
        "general.architecture": "bert", "bert.pooling_type": GGUF_POOLING_RANK, "bert.context_length": 8192,
        "bert.block_count": 24, "bert.attention.head_count": 16, "bert.embedding_length": 1024})
    assert info.pooling_type == GGUF_POOLING_RANK == 4
    monkeypatch.setattr(inspect_mod, "parse_gguf", lambda _src: info)

    class Client:
        def range_source(self, *_a, **_k):
            return None

    report = ModelReport(repo_id="some-org/multilingual-scorer-GGUF", revision_sha="sha", artifact_type=ArtifactType.GGUF)
    inspect_mod._inspect_gguf(report, parse_source("some-org/multilingual-scorer-GGUF"), Client(), "sha",
                              [("scorer-Q8_0.gguf", 1_100_000_000, None)])
    assert report.task == "rerank"
    assert GgufInfo(version=3, tensor_count=1, metadata={"general.architecture": "bert", "bert.pooling_type": 2}).pooling_type == 2
    assert GgufInfo(version=3, tensor_count=1, metadata={"general.architecture": "bert"}).pooling_type is None


# ── แผน ───────────────────────────────────────────────────────────────────────────

def _qwen_report(**over):
    base = dict(repo_id="Qwen/Qwen3-Reranker-4B", revision_sha="sha", task="rerank",
                artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(8 * GIB),
                architecture="Qwen3ForCausalLM", context_length=40960,
                kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128))
    base.update(over)
    return ModelReport(**base)


def _bge_report():
    return ModelReport(repo_id="BAAI/bge-reranker-v2-m3", revision_sha="sha", task="rerank",
                       artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(2.2 * GIB),
                       architecture="XLMRobertaForSequenceClassification", context_length=8194,
                       kv_dims=KvDims(layers=24, kv_heads=16, head_dim=64))


def _gguf_report():
    return ModelReport(repo_id="gpustack/bge-reranker-v2-m3-GGUF", revision_sha="sha", task="rerank",
                       artifact_type=ArtifactType.GGUF, weight_bytes=int(1.1 * GIB),
                       selected_gguf="bge-reranker-v2-m3-Q8_0.gguf", architecture="bert", context_length=8192,
                       kv_dims=KvDims(layers=24, kv_heads=16, head_dim=64), gguf_variants=[], tags=["gguf"])


def test_rerank_family_decides_who_needs_the_qwen3_overrides():
    assert rerank_family_for(_qwen_report()) == "qwen3"
    assert qwen3_reranker_overrides(_qwen_report()) == QWEN3_RERANKER_HF_OVERRIDES
    assert rerank_family_for(_bge_report()) == "seq-cls" and qwen3_reranker_overrides(_bge_report()) is None
    # ตัวที่คนแปลงเป็น seq-cls แล้ว (tomaarsen/Qwen3-Reranker-0.6B-seq-cls) ไม่ต้อง override อีก
    converted = _qwen_report(repo_id="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", architecture="Qwen3ForSequenceClassification")
    assert rerank_family_for(converted) == "seq-cls" and qwen3_reranker_overrides(converted) is None
    assert rerank_family_for(_gguf_report()) == "generic"
    assert is_pooling_task("embed") and is_pooling_task("rerank") and not is_pooling_task("generate")
    assert " " not in QWEN3_RERANKER_HF_OVERRIDES.split(" ", 1)[1], "JSON ต้องไม่มีช่องว่าง — EXTRA_SERVE_ARGS แยกคำด้วยช่องว่าง"


def test_qwen3_reranker_plan_is_a_pooling_plan_with_the_vllm_overrides_once():
    report = _qwen_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    assert plan.task == "rerank" and plan.runtime.engine is Engine.VLLM
    assert plan.serving.context <= 32768
    assert not plan.tool_calling.enabled and plan.tool_calling.parser is None and not plan.reasoning.enabled
    assert plan.serving.max_output_tokens == 16
    overrides = [f for f in plan.serving.extra_flags if f.startswith("--hf-overrides")]
    assert overrides == [QWEN3_RERANKER_HF_OVERRIDES], "rule-based + สูตร ต้องได้ตัวเดียว ไม่ซ้ำ"
    assert "Qwen3ForSequenceClassification" in overrides[0] and '"is_original_qwen3_reranker":true' in overrides[0]
    # สูตรใน catalog: image NGC เดียวกับ embedding ที่รันอยู่ + slots 8
    assert plan.runtime.image_ref == "nvcr.io/nvidia/vllm:26.05-py3" and plan.serving.max_num_seqs == 8
    assert any("test-rerank" in w for w in plan.warnings)
    assert not any("/v1/embeddings" in w for w in plan.warnings), "ต้องไม่ถูกเรียกว่า embedding อีก"


def test_a_native_sequence_classifier_needs_no_overrides():
    report = _bge_report()
    plan = build_plan(report, analyze(report, PRESETS["dgx-spark-single"]), provider=None)
    assert plan.task == "rerank" and plan.runtime.engine is Engine.VLLM
    assert not any(f.startswith("--hf-overrides") for f in plan.serving.extra_flags)


def test_rerank_follows_the_embedding_guard_rails():
    report = _bge_report()
    assert rule_based_plan(report, analyze(report, PRESETS["dgx-spark-single"]), Engine.SGLANG).runtime.engine is Engine.VLLM
    with pytest.raises(PlanError, match="reranker"):
        rule_based_plan(report, analyze(report, PRESETS["dgx-spark-stacked"]), None)


def test_harden_forces_task_and_the_overrides_back_and_dedupes_them():
    """แผนจาก LLM ตั้ง task เองไม่ได้ · ลืม --hf-overrides = harden เติมให้ · ใส่ซ้ำ (recipe + LLM) = เก็บตัวแรก"""
    from lmds.brain.orchestrator import harden_plan

    report = _qwen_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = rule_based_plan(report, fit, None)
    plan.task = "generate"
    plan.tool_calling.enabled = True
    plan.tool_calling.parser = "hermes"
    plan.serving.extra_flags = [f for f in plan.serving.extra_flags if not f.startswith("--hf-overrides")]
    hardened = harden_plan(plan, report, fit)
    assert hardened.task == "rerank" and not hardened.tool_calling.enabled
    assert [f for f in hardened.serving.extra_flags if f.startswith("--hf-overrides")] == [QWEN3_RERANKER_HF_OVERRIDES]

    plan2 = rule_based_plan(report, fit, None)
    plan2.serving.extra_flags = plan2.serving.extra_flags + [QWEN3_RERANKER_HF_OVERRIDES, '--hf-overrides {"architectures":["Other"]}']
    hardened2 = harden_plan(plan2, report, fit)
    assert [f for f in hardened2.serving.extra_flags if f.startswith("--hf-overrides")] == [QWEN3_RERANKER_HF_OVERRIDES]


def test_kv_pin_treats_rerank_exactly_like_embed():
    """prefill ล้วนทั้งคู่ — pin = slots × KV(context) × 1.2 · โมเดลรูปเดียวกันต้องได้ pin เท่ากัน"""
    from lmds.fit.sizing import parse_kv_pin, sizing_for_plan

    def twin(task, repo):
        return ModelReport(repo_id=repo, revision_sha="sha", task=task, artifact_type=ArtifactType.SAFETENSORS,
                           weight_bytes=int(2.2 * GIB), architecture="XLMRobertaForSequenceClassification",
                           context_length=8194, kv_dims=KvDims(layers=24, kv_heads=16, head_dim=64))

    plans = {}
    for task, repo in (("rerank", "some-org/scorer"), ("embed", "some-org/encoder")):
        report = twin(task, repo)
        fit = analyze(report, PRESETS["dgx-spark-single"])
        plans[task] = (build_plan(report, fit, provider=None), fit)
    (rerank, rfit), (embed, _efit) = plans["rerank"], plans["embed"]
    assert rerank.serving.max_num_seqs == embed.serving.max_num_seqs and rerank.serving.context == embed.serving.context
    r_pin, e_pin = parse_kv_pin(" ".join(rerank.serving.extra_flags)), parse_kv_pin(" ".join(embed.serving.extra_flags))
    assert r_pin and r_pin == e_pin
    sized = sizing_for_plan(rfit, rerank)
    assert sized["slots"] == rerank.serving.max_num_seqs and sized["context"] == rerank.serving.context
    assert sized["kv_pin_bytes"] == r_pin


# ── controller ────────────────────────────────────────────────────────────────────

def _render(report, tmp_path):
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    script = next(bundle.directory.glob("*-single.sh"))
    return plan, bundle, script, script.read_text(encoding="utf-8")


def _env(tmp_path, **extra):
    return {"PATH": SAFE_PATH, "HOME": str(tmp_path), "ADVERTISE_IP": "10.0.0.9", "RUN_DIR": str(tmp_path / "run"), **extra}


def test_vllm_qwen3_reranker_controller(tmp_path):
    plan, bundle, script, text = _render(_qwen_report(), tmp_path)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert "serve_args+=(--runner pooling --convert classify)" in text
    assert "--convert embed" not in text and "test-embed)" not in text.split('case "${1:-help}" in')[-1].split("test-text|")[0]
    assert re.search(r"^\s+test-rerank\)\s+test_rerank ;;", text, re.M) and "  test-text)    test_text ;;" not in text
    assert '"task": "rerank"' in text and '"score_endpoint": "/v1/score"' in text

    # score template ของ Qwen3-Reranker แนบมากับ bundle และถูก mount ให้ --chat-template
    template = bundle.directory / "score_template.jinja"
    assert template.exists() and template in bundle.files
    body = template.read_text(encoding="utf-8")
    assert '<Query>' in body and 'selectattr("role", "eq", "document")' in body and body.endswith("<think>\n\n</think>\n\n")
    assert 'docker_args+=(-v "${SCORE_TEMPLATE}:${SCORE_TEMPLATE_MOUNT}:ro")' in text

    # DRY_RUN: argv จริงมี --hf-overrides (JSON เป็น argv เดียว) + --chat-template ไปที่ mount
    dry = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True, env=_env(tmp_path, DRY_RUN="1"))
    assert dry.returncode == 0, dry.stderr
    argv = dry.stdout.splitlines()
    assert argv[argv.index("--runner") + 1] == "pooling" and argv[argv.index("--convert") + 1] == "classify"
    assert argv[argv.index("--hf-overrides") + 1] == QWEN3_RERANKER_HF_OVERRIDES.split(" ", 1)[1]
    assert argv[argv.index("--chat-template") + 1] == "/opt/lmds/score_template.jinja"
    assert "--tool-call-parser" not in argv and "--reasoning-parser" not in argv

    # SCORE_TEMPLATE="" = ปิด template ได้
    off = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True,
                         env=_env(tmp_path, DRY_RUN="1", SCORE_TEMPLATE=""))
    assert off.returncode == 0 and "--chat-template" not in off.stdout.splitlines()
    # ชี้ไฟล์ที่ไม่มี = die ไม่ใช่ start แล้วให้ vLLM ตายทีหลัง
    missing = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True,
                             env=_env(tmp_path, DRY_RUN="1", SCORE_TEMPLATE=str(tmp_path / "nope.jinja")))
    assert missing.returncode != 0 and "SCORE_TEMPLATE" in missing.stderr

    # test-text บน reranker บอกให้ไปใช้ test-rerank (mirror ของ embedding)
    done = subprocess.run(["bash", str(script), "test-text"], capture_output=True, text=True, env=_env(tmp_path))
    assert done.returncode == 2 and "test-rerank" in done.stderr and "reranker" in done.stderr
    embed = subprocess.run(["bash", str(script), "test-embed"], capture_output=True, text=True, env=_env(tmp_path))
    assert embed.returncode == 2 and "test-rerank" in embed.stderr

    cfg = json.loads(subprocess.run(["bash", str(script), "client-config"], capture_output=True, text=True,
                                    env=_env(tmp_path)).stdout)
    assert cfg["task"] == "rerank" and cfg["endpoint"] == "/v1/rerank" and cfg["score_endpoint"] == "/v1/score"
    assert cfg["score_template"] == "bundled"

    profile = (bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    assert "task: rerank" in profile and "family: qwen3" in profile and "score_template: true" in profile
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "test-rerank" in readme and "/v1/rerank" in readme and "-single.sh test-text" not in readme


def test_vllm_native_reranker_has_no_template_and_no_overrides(tmp_path):
    plan, bundle, script, text = _render(_bge_report(), tmp_path)
    assert not (bundle.directory / "score_template.jinja").exists()
    dry = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True,
                         env=_env(tmp_path, DRY_RUN="1", GPU_MEMORY_UTILIZATION="0.5"))
    assert dry.returncode == 0, dry.stderr
    argv = dry.stdout.splitlines()
    assert "--hf-overrides" not in argv and "--chat-template" not in argv
    assert argv[argv.index("--convert") + 1] == "classify"
    cfg = json.loads(subprocess.run(["bash", str(script), "client-config"], capture_output=True, text=True,
                                    env=_env(tmp_path, GPU_MEMORY_UTILIZATION="0.5")).stdout)
    assert cfg["score_template"] == "none"
    assert "family: seq-cls" in (bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")


def test_a_small_pooling_model_gets_gpu_util_0_3_and_the_controller_accepts_it(tmp_path):
    """sizing ตั้งพื้น gpu-util ที่ 0.3 (max(0.3, …)) แต่ controller เคยตรวจ `> 0.3` — โมเดลเล็ก (bge-reranker 2 GB,
    Qwen3-Embedding-0.6B) ได้ 0.3 พอดีแล้ว start ตาย "invalid --gpu-util: 0.3 (ต้องอยู่ระหว่าง 0.3 ถึง 0.98)" (พบตอนทำ rerank 2026-09-08)"""
    plan, _bundle, script, _text = _render(_bge_report(), tmp_path)
    assert plan.serving.gpu_memory_utilization == 0.3
    dry = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True, env=_env(tmp_path, DRY_RUN="1"))
    assert dry.returncode == 0, dry.stderr
    too_low = subprocess.run(["bash", str(script), "start"], capture_output=True, text=True,
                             env=_env(tmp_path, DRY_RUN="1", GPU_MEMORY_UTILIZATION="0.29"))
    assert too_low.returncode != 0 and "invalid --gpu-util" in too_low.stderr


def test_regenerating_as_a_non_qwen_model_removes_a_stale_score_template(tmp_path):
    """โฟลเดอร์เดิม เปลี่ยนเป็นโมเดลอื่น — template เก่าต้องไม่ค้างให้ controller หยิบไปใช้"""
    _plan, bundle, _script, _text = _render(_qwen_report(), tmp_path)
    assert (bundle.directory / "score_template.jinja").exists()
    report = _qwen_report(repo_id="Qwen/Qwen3-Reranker-4B", architecture="Qwen3ForSequenceClassification")
    fit = analyze(report, PRESETS["dgx-spark-single"])
    render_bundle(build_plan(report, fit, provider=None), report, fit, tmp_path)
    assert not (bundle.directory / "score_template.jinja").exists()


def test_llamacpp_reranker_controller(tmp_path):
    plan, bundle, script, text = _render(_gguf_report(), tmp_path)
    assert plan.runtime.engine is Engine.LLAMACPP
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert 'SERVER_ARGS+=(--reranking --batch-size "$EMBED_UBATCH" --ubatch-size "$EMBED_UBATCH")' in text
    assert "--embedding --pooling" not in text and "POOLING=" not in text
    out = subprocess.run(["bash", str(script), "serve-args"], capture_output=True, text=True,
                         env={**os.environ, "RUN_DIR": str(tmp_path / "run")}, timeout=60)
    assert out.returncode == 0, out.stderr
    argv = out.stdout.splitlines()
    assert "--reranking" in argv and "--embedding" not in argv
    done = subprocess.run(["bash", str(script), "test-text"], capture_output=True, text=True, env=_env(tmp_path))
    assert done.returncode == 2 and "test-rerank" in done.stderr
    cfg = json.loads(subprocess.run(["bash", str(script), "client-config"], capture_output=True, text=True,
                                    env=_env(tmp_path, CTX_SIZE="8192", PARALLEL_SEQS="2")).stdout)
    assert cfg["task"] == "rerank" and cfg["endpoint"] == "/v1/rerank" and cfg["max_input_tokens"] == 4096
    assert "pooling" not in cfg and "max_output_tokens" not in cfg
    assert "task: rerank" in (bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")


# ── test-rerank กับเซิร์ฟเวอร์ปลอม ────────────────────────────────────────────────────

class _FakeRerank(http.server.BaseHTTPRequestHandler):
    mode = "good"
    seen: list[dict] = []

    def log_message(self, *_a):  # เงียบ
        pass

    def do_GET(self):  # /v1/models ของ assert_our_server ถูก stub ทิ้ง — เผื่อไว้
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        type(self).seen.append({"path": self.path, "body": body})
        if type(self).mode == "http-error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        if type(self).mode == "garbage":
            payload = {"choices": []}
        else:
            docs = body.get("documents") or []
            scores = {"good": [0.02, 0.97, 0.05], "wrong": [0.91, 0.30, 0.10]}[type(self).mode]
            payload = {"id": "rerank-x", "model": body.get("model"),
                       "results": [{"index": i, "relevance_score": scores[i], "document": {"text": d}} for i, d in enumerate(docs)]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_rerank_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeRerank)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeRerank.seen = []
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()


def _test_rerank_script(controller_text: str) -> str:
    return ("set -Eeuo pipefail\nassert_our_server() { :; }\ndie() { echo \"ERROR: $*\" >&2; exit 1; }\n"
            + extract_fn(controller_text, "test_rerank") + "\ntest_rerank\n")


@pytest.mark.parametrize("engine", ["vllm", "llamacpp"])
@pytest.mark.parametrize("mode, rc, marker", [("good", 0, "PASS"), ("wrong", 2, "FAIL"), ("http-error", 1, "HTTP 500"),
                                              ("garbage", 1, "results")])
def test_test_rerank_passes_only_when_the_relevant_document_ranks_first(tmp_path, fake_rerank_server, engine, mode, rc, marker):
    report = _qwen_report() if engine == "vllm" else _gguf_report()
    _plan, _bundle, _script, text = _render(report, tmp_path)
    _FakeRerank.mode = mode
    done = subprocess.run(["bash", "-c", _test_rerank_script(text)], capture_output=True, text=True, timeout=60,
                          env={"PATH": os.environ["PATH"], "API_PORT": str(fake_rerank_server), "SERVED_MODEL_NAME": "rr",
                               "API_KEY": "secret-key"})
    assert done.returncode == rc, done.stdout + done.stderr
    assert marker in done.stdout
    if mode in ("good", "wrong"):
        assert done.stdout.count("score=") == 3, "ต้องพิมพ์คะแนนทุกชิ้น"
        assert "relevant" in done.stdout
    request = _FakeRerank.seen[-1]
    assert request["path"] == "/v1/rerank" and request["body"]["model"] == "rr"
    assert isinstance(request["body"]["query"], str) and len(request["body"]["documents"]) == 3


# ── hub / UI / recipe / assistant ────────────────────────────────────────────────

def test_feature_summary_and_inventory_know_rerank(tmp_path):
    from lmds.fleet.manager import feature_summary
    from lmds.inventory import KNOWN_COMMANDS, controller_commands

    assert feature_summary({"features": {"rerank": {"family": "qwen3"}}}) == "rerank (qwen3)"
    assert feature_summary({"features": {"rerank": {}}}) == "text"
    assert feature_summary({"features": {"embedding": {"pooling": "last"}, "rerank": None}}) == "embedding (last)"
    assert {"test-embed", "test-rerank"} <= KNOWN_COMMANDS, "ปุ่มบนหน้าเว็บขึ้นตามชุดนี้"
    _plan, _bundle, script, _text = _render(_qwen_report(), tmp_path)
    assert "test-rerank" in controller_commands(str(script)) and "test-embed" not in controller_commands(str(script))


def test_the_recipes_and_the_assistant_know_rerank():
    from lmds.assistant import brain, catalog
    from lmds.recipes import find_recipe

    for repo in ("Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B"):
        recipe = find_recipe(repo)
        assert recipe is not None and recipe.engine == "vllm" and recipe.image == "nvcr.io/nvidia/vllm:26.05-py3"
        assert recipe.serving["max_num_seqs"] == 8 and "is_original_qwen3_reranker" in recipe.serving["hf_overrides"]
        assert "pending" in recipe.validated_on
    assert find_recipe("Qwen/Qwen3-Reranker-0.6B") is None

    action = catalog.ACTIONS["run_test"]
    assert action.command({"slug": "rr", "test": "test-rerank"})[0].endswith('"$ctl" test-rerank')
    assert action.command({"slug": "ee", "test": "test-embed"})[0].endswith('"$ctl" test-embed')
    with pytest.raises(brain.BrainError, match="chat"):
        brain._pick({"slug": "rr", "running": True, "engine": "vllm", "port": 8011, "features": "rerank (qwen3)"}, "this", "127.0.0.1")


RERANK_CARD = """const fx = { nodes: [
  { name: "spark-04", site: "TKC", models: [{ slug: "qwen3-reranker-4b", running: true, healthy: true, engine: "vllm", port: 8011,
      context: 32768, features: "rerank (qwen3)", commands: ["start", "stop", "status", "logs", "client-config", "test-rerank"],
      controller_exists: true, downloaded: true }] } ],
  localModels: [{ slug: "rr-x", running: true, healthy: true, controller_exists: true, downloaded: true, engine: "vllm", port: 8011,
                  topology: "single", context: 32768, features: "rerank (qwen3)", model_id: "Qwen/Qwen3-Reranker-4B",
                  commands: ["start", "stop", "status", "logs", "client-config", "test-rerank"] },
                { slug: "chat-a", running: true, healthy: true, controller_exists: true, downloaded: true, engine: "vllm", port: 8000,
                  topology: "single", context: 32768, features: "tools", model_id: "org/chat-a",
                  commands: ["start", "stop", "status", "logs", "test-text", "test-tools"] }] };
H.fx = fx;
H.routes = H.defaultRoutes(fx);
"""


def test_the_card_shows_the_rerank_tag_and_the_test_rerank_button_but_no_brain_button(tmp_path):
    """หน้าเว็บจริง (DOM ย่อส่วน): การ์ด reranker ติดป้าย RERANK · แผง Tests มีปุ่ม test-rerank (ทั้งในเครื่องและ node) ·
    ไม่มีปุ่ม brain/test-text เพราะไม่ใช่ chat"""
    from tests.test_console_shell import run_scenario

    (out,) = run_scenario(tmp_path, RERANK_CARD, """
        const tagText = el => [...el.querySelectorAll(".tag")].map(t => t.textContent.trim());
        const row = document.querySelector('button[data-act="tests"][data-slug="rr-x"]').closest("div[id], li, tr, article, section, div");
        const tags = tagText(document);
        document.querySelector('button[data-act="tests"][data-slug="rr-x"]').click(); await H.tick(5);
        const rerankBtn = !!document.querySelector('#panel-rr-x button[data-act="job:test-rerank"]');
        const textBtn = !!document.querySelector('#panel-rr-x button[data-act="job:test-text"]');
        document.querySelector('button[data-act="opts"][data-slug="rr-x"]').click(); await H.tick(5);
        const brainBtn = !!document.querySelector('#panel-rr-x button[data-act="brain"]');
        document.querySelector('button[data-act="opts"][data-slug="chat-a"]').click(); await H.tick(5);
        const chatBrainBtn = !!document.querySelector('#panel-chat-a button[data-act="brain"]');
        await H.go("#/node/spark-04");
        document.querySelector('button[data-nact="menu"][data-node="spark-04"][data-slug="qwen3-reranker-4b"]').click(); await H.tick(8);
        const nodeRerankBtn = !!document.querySelector('button[data-nact="ctl:test-rerank"][data-node="spark-04"]');
        const nodeBrainBtn = !!document.querySelector('button[data-nact="brain"][data-node="spark-04"]');
        const nodeTags = tagText(document);
        console.log(JSON.stringify({ tags, rerankBtn, textBtn, brainBtn, chatBrainBtn, nodeRerankBtn, nodeBrainBtn, nodeTags }));
    """)
    assert "RERANK" in out["tags"] and "RERANK" in out["nodeTags"]
    assert out["rerankBtn"] is True and out["textBtn"] is False
    assert out["brainBtn"] is False and out["chatBrainBtn"] is True, "reranker เสิร์ฟ chat ไม่ได้ ไม่ควรมีปุ่ม brain"
    assert out["nodeRerankBtn"] is True and out["nodeBrainBtn"] is False


def test_the_web_ctl_allowlist_and_the_page_carry_test_rerank():
    api = Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "api.py"
    assert '"test-rerank"' in api.read_text(encoding="utf-8")
    page = (Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert page.count('"test-rerank"') >= 2, "ปุ่มทั้งการ์ดในเครื่องและการ์ดของ node"
    assert ">RERANK</span>" in page and "isPoolingModel(m)" in page and 'data-task-hint="rerank"' in page
