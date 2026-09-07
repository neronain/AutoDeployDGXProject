"""ตั้ง slots/context/KV ให้พอดี (Fit) — สูตรเดียวสำหรับ CLI · หน้าเว็บ · pin ตั้งต้นตอน deploy

เจ้าของ 2026-09-07: "ผมอยากทราบค่าที่ทำให้รัน 2 model บน vllm ผ่าน … ถ้าอนาคตรันแบบนี้อีก หรือกับ node อื่นหรือลูกค้า
จะทราบได้อย่างไรว่าควรใช้ค่าไหน เช่น ct = 262144 แต่ slot, gpu util จะตั้งค่าอย่างไรให้พอดี perfect"

ตัวเลขจริงที่ยึดไว้ (probe อ่านอย่างเดียว 2026-09-07 · docker logs + nvidia-smi + free -g):
  spark-head (gigabyte01) free -g: total 121 · used 108 · available 12
    VLLM::EngineCore 80,364 MiB = Nemotron-3-Super-120B-A12B NVFP4 (hybrid Mamba)
      "Model loading took 69.62 GiB" · "reserved 6.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes"
      "GPU KV cache size: 1,179,648 tokens, Maximum concurrency for 262,144 tokens per request: 4.50x"
      argv: --max-model-len 262144 --gpu-memory-utilization 0.85 --max-num-seqs 2 --kv-cache-dtype fp8 --kv-cache-memory 6442450944
      → 69.62 + 6 + ~2.9 overhead = 78.5 GiB ที่ nvidia-smi เห็น (overhead ~3 GB ตามสูตร)
    llama-server 19,921 MiB = Gemma-4 26B-A4B QAT Q4_K_M · ctx 65536 · 2 slots (weight_bytes 16,796,015,520 บนดิสก์)
  spark-worker (gigabyte02) free -g: total 121 · used 95 · available 25
    VLLM::EngineCore 86,263 MiB = Qwopus3.5-122B-A10B NVFP4
      "Model loading took 71.32 GiB" · pin 12 GiB · "GPU KV cache size: 508,031 tokens" · "Maximum concurrency … 1.94x"
      MODEL_PROFILE: weight_bytes 81,488,272,728 (75.9 GiB บนดิสก์) · kv_bytes_per_token: null → ต้องวัดจาก log
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lmds.fit.sizing import (
    KV_PIN_HEADROOM,
    kv_dtype_from_args,
    measured_from_log,
    parse_kv_pin,
    plan_kv_pin,
    settings_for,
    with_kv_pin,
)

GIB = 1024**3

NEMOTRON_LOG = """(EngineCore pid=175) INFO 09-07 03:05:51 [model_runner.py:408] Model loading took 69.62 GiB memory and 470.931960 seconds
(EngineCore pid=175) INFO 09-07 03:06:51 [gpu_worker.py:551] Initial free memory 114.28 GiB, reserved 6.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config.
(EngineCore pid=175) INFO 09-07 03:06:51 [kv_cache_utils.py:2312] GPU KV cache size: 1,179,648 tokens, Maximum concurrency for 262,144 tokens per request: 4.50x"""

QWOPUS_LOG = """(EngineCore pid=204) INFO 09-07 03:24:22 [gpu_model_runner.py:4879] Model loading took 71.32 GiB memory and 559.801924 seconds
(EngineCore pid=204) INFO 09-07 03:26:16 [gpu_worker.py:361] Initial free memory 114.29 GiB, reserved 12.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.
(EngineCore pid=204) INFO 09-07 03:26:16 [kv_cache_utils.py:1708] GPU KV cache size: 508,031 tokens
(EngineCore pid=204) INFO 09-07 03:26:16 [kv_cache_utils.py:1709] Maximum concurrency for 262,144 tokens per request: 1.94x"""

SPARK = {"memory_model": "unified", "total_gb": 121.0, "free_gb": 4.0, "held_gb": 98.0}


@pytest.fixture(autouse=True)
def _fresh_serving_cache():
    """`serving._detect` เป็น lru_cache ทั้ง process — create_app()/build_plan ในไฟล์นี้จะตรึงคำตอบ "เครื่องนี้เป็น hub"
    ไว้ให้เทสที่รันทีหลัง (test_llamacpp_runtime_arch / job ใน test_web) ซึ่งต้องตรวจใหม่ใต้ shim ของตัวเอง — ล้างทั้งก่อนและหลัง"""
    from lmds.hardware import serving

    serving.reset_cache()
    yield
    serving.reset_cache()


def nemotron(running=True, **over):
    info = {"slug": "nemotron", "engine": "vllm", "weight_bytes": 80317948856, "kv_bytes_per_token": 8192,
            "native_context": 262144, "context": 262144, "slots": 2,
            "extra_args": "--kv-cache-dtype fp8 --kv-cache-memory 6442450944", "running": running,
            "own_gb": 78.5 if running else None, "measured": measured_from_log(NEMOTRON_LOG)}
    info.update(over)
    return info


# ── log ของ vLLM ─────────────────────────────────────────────────────────────
def test_measured_from_log_reads_both_line_shapes():
    """หัว 0.10 พิมพ์ tokens+concurrency บรรทัดเดียว · รุ่นใหม่แยกสองบรรทัด — ต้องอ่านได้ทั้งคู่"""
    a = measured_from_log(NEMOTRON_LOG)
    assert a == {"loading_took_gib": 69.62, "initial_free_gib": 114.28, "reserved_kv_gib": 6.0, "kv_pool_gib": 6.0,
                 "kv_cache_tokens": 1179648, "concurrency_context": 262144, "concurrency": 4.5}
    b = measured_from_log(QWOPUS_LOG)
    assert b["kv_cache_tokens"] == 508031 and b["concurrency"] == 1.94 and b["kv_pool_gib"] == 12.0
    assert measured_from_log("") == {}
    # ไม่มี pin: profiling บอก "Available KV cache memory" — ใช้เป็น pool แทน
    c = measured_from_log("Available KV cache memory: 60.10 GiB\nGPU KV cache size: 393,936 tokens")
    assert c["kv_pool_gib"] == 60.1 and c["kv_cache_tokens"] == 393936


def test_pin_flag_round_trip_and_replacement():
    assert parse_kv_pin("--kv-cache-dtype fp8 --kv-cache-memory 6442450944") == 6442450944
    assert parse_kv_pin("--kv-cache-memory=12884901888") == 12884901888
    assert parse_kv_pin("--kv-cache-dtype fp8") is None
    assert with_kv_pin("--kv-cache-dtype fp8 --kv-cache-memory 1 --foo", 2) == "--kv-cache-dtype fp8 --foo --kv-cache-memory 2"
    assert with_kv_pin("", 5) == "--kv-cache-memory 5"
    assert kv_dtype_from_args("--kv-cache-dtype fp8_e4m3") == "fp8" and kv_dtype_from_args("--x 1") == "bf16"


# ── สูตร ──────────────────────────────────────────────────────────────────────
def test_hybrid_model_uses_measured_numbers_over_the_profile():
    """Nemotron (hybrid Mamba): profile บอก 8192 B/token (bf16) · log จริง 6 GiB ÷ 1,179,648 tokens = 5,461 B/token
    (รวม mamba state + block rounding ที่ config ไม่บอก) · weights 69.62 ที่โหลดจริง ไม่ใช่ 74.8 บนดิสก์"""
    p = plan_kv_pin(nemotron(), {**SPARK, "others": [{"slug": "gemma4", "gb": 19.5}]})
    assert p["weights_gb"] == 69.6 and p["weights_source"] == "measured"
    assert p["kv_per_token_bytes"] == 5461 and p["kv_source"] == "measured"
    assert p["kv_per_request_gb"] == 1.33
    # pin = 2 × 1.33 × 1.2 = 3.2 → ปัดขึ้นครึ่ง GiB = 3.5
    assert p["kv_pin_gb"] == 3.5 and p["kv_pin_bytes"] == int(3.5 * GIB)
    assert p["overhead_gb"] == 3.0 and p["ram_needed_gb"] == 76.1
    assert p["usable_gb"] == 109.0 and p["others_running_gb"] == 19.5
    assert p["ram_after_gb"] == 13.4 and p["fits"] is True and p["verdict"] == "fits"
    assert p["suggested_slots_max"] == 10
    assert p["gpu_util_equivalent"] == 0.65


def test_qwopus_without_kv_in_profile_is_sized_from_the_log():
    """spark-worker: MODEL_PROFILE มี kv_bytes_per_token: null — เดิมคำนวณไม่ได้เลย · log ให้ 12 GiB ÷ 508,031 = 25,362 B/token
    = 6.19 GiB ต่อคำขอ 262K (ตรงกับ "Maximum concurrency 1.94x")"""
    p = plan_kv_pin({"slug": "qwopus", "engine": "vllm", "weight_bytes": 81488272728, "kv_bytes_per_token": None,
                     "native_context": 262144, "slots": 3, "extra_args": "--kv-cache-memory 12884901888",
                     "running": True, "own_gb": 84.2, "measured": measured_from_log(QWOPUS_LOG)},
                    {"memory_model": "unified", "total_gb": 121.0, "free_gb": 6.0, "held_gb": 84.2})
    assert p["kv_per_token_bytes"] == 25362 and p["kv_per_request_gb"] == 6.19
    assert p["kv_pin_gb"] == 22.5 and p["ram_needed_gb"] == 96.8
    assert p["others_running_gb"] == 0.0 and p["ram_after_gb"] == 12.2 and p["verdict"] == "fits"
    assert p["free_now_corrected_gb"] == 90.2  # 6 ว่าง + 84.2 ที่ตัวมันเองถือ


def test_running_model_is_not_counted_against_itself():
    """เคสที่ทำให้เจ้าของถาม: การ์ดขึ้น "Cannot start now — 12.9 GB free" ให้โมเดลที่รันอยู่แล้ว เพราะนับหน่วยความจำของตัวมันเอง
    เป็น "ไม่ว่าง" · เครื่องเดียวกัน (held 98 = ตัวมัน 78.5 + Gemma 19.5): รันอยู่ → ใส่ได้ · ไม่ได้รัน (98 เป็นของคนอื่นทั้งหมด) → ไม่พอ"""
    host = {**SPARK}  # ไม่แยกรายตัว — ต้องหักตัวเองออกจากยอดรวมเอง
    running = plan_kv_pin(nemotron(running=True), host)
    assert running["others_running_gb"] == 19.5 and running["fits"] is True
    assert running["others"] == [{"slug": "other GPU processes", "gb": 19.5}]
    stopped = plan_kv_pin(nemotron(running=False), host)
    assert stopped["others_running_gb"] == 98.0 and stopped["fits"] is False and stopped["verdict"] == "no-fit"
    assert stopped["stop_suggestion"] == "other GPU processes"
    assert "หยุด other GPU processes" in stopped["reason"]


def test_dense_model_that_does_not_fit_suggests_the_slot_count_that_would():
    """Llama-3.3-70B แบบ dense (KV fp8 ≈ 20 GiB ต่อคำขอ 131K): slots 4 → pin 96 GiB → ไม่พอ · บอกว่า slots 1 พอ"""
    info = {"slug": "llama70b", "engine": "vllm", "weight_bytes": 70 * GIB, "kv_bytes_per_token": 327680,
            "native_context": 131072, "slots": 4, "extra_args": "--kv-cache-dtype fp8", "running": False}
    host = {"memory_model": "unified", "total_gb": 121.0, "free_gb": 100.0, "held_gb": 0.0}
    p = plan_kv_pin(info, host)
    assert p["kv_dtype"] == "fp8" and p["kv_per_request_gb"] == 20.0 and p["kv_pin_gb"] == 96.0
    assert p["fits"] is False and p["suggested_slots_max"] == 1 and "ลด slots เหลือ 1" in p["reason"]
    assert p["stack"][-1]["kind"] == "over"
    ok = plan_kv_pin(info, host, slots=1)
    assert ok["kv_pin_gb"] == 24.0 and ok["ram_needed_gb"] == 97.0 and ok["fits"] is True


def test_mla_sparse_model_pins_a_small_kv():
    """GLM-5.3-Flash (sparse MLA ~1.7 GiB ต่อ 262K bf16 · 89 GiB weights ต่อ rank) — pin เล็กมากแม้ 6 slots"""
    p = plan_kv_pin({"slug": "glm", "engine": "vllm", "weight_bytes": 89 * GIB, "kv_bytes_per_token": 6963,
                     "native_context": 262144, "slots": 6, "extra_args": "--kv-cache-dtype fp8_e4m3", "running": False},
                    {"memory_model": "unified", "total_gb": 121.0, "free_gb": 110.0, "held_gb": 0.0})
    assert p["kv_per_request_gb"] == 0.85 and p["kv_pin_gb"] == 6.5
    assert p["ram_needed_gb"] == 98.5 and p["verdict"] == "fits"


def test_llamacpp_uses_the_context_pool_not_slots():
    """Gemma-4 26B-A4B llama.cpp บน spark-head: ctx 65536 · 2 slots → 15.6 + KV(65536) 2.5 + 1.5 = 19.6 (วัดจริง 19,921 MiB = 19.5)"""
    p = plan_kv_pin({"slug": "gemma4", "engine": "llamacpp", "weight_bytes": 16796015520, "kv_bytes_per_token": 40960,
                     "native_context": 262144, "slots": 2, "context": 65536, "running": False},
                    {"memory_model": "unified", "total_gb": 121.0, "free_gb": 30.0, "held_gb": 78.5,
                     "others": [{"slug": "nemotron", "gb": 78.5}]})
    assert p["pin_supported"] is False and p["kv_pin_bytes"] is None
    assert p["kv_gb"] == 2.5 and p["ram_needed_gb"] == 19.6 and p["per_slot_context"] == 32768
    assert p["fits"] is True and p["ram_after_gb"] == 10.9
    assert settings_for(p, "") == {"slots": "2", "context": "65536"}
    # dense 70B Q4 (KV 256 KiB/token) เต็ม 262K = 64 GiB → ไม่พอข้าง Nemotron · บอก context ที่พอ (ขั้น 4096)
    too_big = plan_kv_pin({"slug": "dense", "engine": "llamacpp", "weight_bytes": 20 * GIB, "kv_bytes_per_token": 262144,
                           "native_context": 262144, "slots": 1, "running": False},
                          {"memory_model": "unified", "total_gb": 121.0, "held_gb": 78.5, "others": [{"slug": "nemotron", "gb": 78.5}]})
    assert too_big["fits"] is False and too_big["ram_needed_gb"] == 85.5
    # ที่เหลือ 109 − 78.5 − 20 − 1.5 = 9 GiB ÷ 256 KiB/token = 36,864 tokens (ลงตัวขั้น 4096 พอดี)
    assert too_big["suggested_context_max"] == 36864 and "ลด context เหลือ ≤ 36,864" in too_big["reason"]


def test_context_is_clamped_to_native_and_unattributed_memory_is_kept():
    p = plan_kv_pin({"slug": "x", "engine": "vllm", "weight_bytes": 20 * GIB, "kv_bytes_per_token": 4096,
                     "native_context": 131072, "running": False},
                    {"memory_model": "unified", "total_gb": 121.0, "held_gb": 50.0, "others": [{"slug": "a", "gb": 30.0}]},
                    slots=2, context=999999)
    assert p["context"] == 131072 and any("เกิน native" in n for n in p["notes"])
    # จับคู่รายตัวได้ 30 แต่ nvidia-smi เห็น 50 — ส่วนต่าง 20 ต้องยังถูกนับ
    assert p["others_running_gb"] == 50.0
    assert {"slug": "other GPU processes", "gb": 20.0} in p["others"]


def test_unknown_inputs_say_why_instead_of_guessing():
    p = plan_kv_pin({"slug": "x", "engine": "vllm", "weight_bytes": None, "kv_bytes_per_token": 4096, "native_context": 8192},
                    {"memory_model": "unified", "total_gb": 121.0})
    assert p["fits"] is None and p["verdict"] == "unknown" and "weights" in p["reason"]
    q = plan_kv_pin({"slug": "x", "engine": "vllm", "weight_bytes": GIB, "kv_bytes_per_token": None, "native_context": 8192},
                    {"memory_model": "unified", "total_gb": 121.0})
    assert q["verdict"] == "unknown" and "KV ต่อ token" in q["reason"]


def test_settings_for_vllm_writes_pin_and_equivalent_gpu_util():
    p = plan_kv_pin(nemotron(), {**SPARK, "others": [{"slug": "gemma4", "gb": 19.5}]})
    s = settings_for(p, "--kv-cache-dtype fp8 --kv-cache-memory 6442450944")
    assert s == {"slots": "2", "context": "262144", "extra_args": f"--kv-cache-dtype fp8 --kv-cache-memory {int(3.5 * GIB)}",
                 "gpu_util": "0.65"}


# ── CLI: lmds set --fit / lmds fit ──────────────────────────────────────────────
def _bundle(tmp_path, engine="vllm", kv=8192, extra=None, weight_bytes=80317948856):
    from lmds.fleet.bundle_settings import write

    d = tmp_path / "bundle"
    d.mkdir()
    ctl = d / "nemotron-single.sh"
    ctl.write_text('#!/usr/bin/env bash\nBUNDLE_ENV="${BUNDLE_ENV:-x}"\nAPI_PORT="${API_PORT:-8000}"\n', encoding="utf-8")
    ctl.chmod(0o755)
    (d / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "weight_bytes": weight_bytes,
                  "native_context": 262144, "kv_bytes_per_token": kv},
        "runtime": {"engine": engine}, "topology": "single",
        "serving": {"context": 262144, "gpu_memory_utilization": 0.85, "max_num_seqs": 4},
    }), encoding="utf-8")
    if extra:
        write(d, {"extra_args": extra})
    return ctl


def _fake_server(ctl: Path, running=True):
    from lmds.fleet import ServerInfo

    return ServerInfo(slug="nemotron", model_id="nvidia/x", engine="vllm", mode="docker", port=8000,
                      container="lmds-nemotron", controller=str(ctl), running=running, healthy=running)


HOST = {"memory_model": "unified", "ram_total_gb": 121.0, "ram_used_gb": 108.0, "foreign": [],
        "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": 98.0}]}
MODELS = [{"slug": "nemotron", "running": True, "engine": "vllm", "memory_gb": 78.5},
          {"slug": "gemma4", "running": True, "engine": "llamacpp", "memory_gb": 19.5}]


@pytest.fixture
def spark_head(monkeypatch, tmp_path):
    ctl = _bundle(tmp_path, extra="--kv-cache-dtype fp8 --kv-cache-memory 6442450944")
    server = _fake_server(ctl)
    monkeypatch.setattr("lmds.fleet.find", lambda slug: server if slug == "nemotron" else None)
    monkeypatch.setattr("lmds.fleet.sizing.local_facts", lambda: (HOST, MODELS))
    monkeypatch.setattr("lmds.fleet.sizing.measured_for", lambda s: measured_from_log(NEMOTRON_LOG))
    return ctl.parent


def test_set_fit_writes_and_replaces_the_kv_pin(spark_head):
    """`lmds set nemotron --fit --slots 2` บน spark-head: bundle.args เดิมมี pin 6 GiB → แทนด้วย 3.5 GiB · dtype คงอยู่ ·
    slots/context/gpu-util เทียบเท่าลง bundle.env"""
    from lmds.cli.main import app
    from lmds.fleet.bundle_settings import read

    r = CliRunner().invoke(app, ["set", "nemotron", "--fit", "--slots", "2"])
    assert r.exit_code == 0, r.output
    saved = read(spark_head)
    assert saved["extra_args"] == f"--kv-cache-dtype fp8 --kv-cache-memory {int(3.5 * GIB)}"
    assert saved["slots"] == "2" and saved["context"] == "262144" and saved["gpu_util"] == "0.65"
    assert "weights" in r.output and "69.6" in r.output and "lmds restart nemotron" in r.output
    # --json ให้ hub ใช้ผ่าน SSH — stdout เป็น JSON ล้วน
    r2 = CliRunner().invoke(app, ["set", "nemotron", "--fit", "--slots", "3", "--json"])
    assert r2.exit_code == 0, r2.output
    d = json.loads(r2.output)
    assert d["plan"]["slots"] == 3 and d["saved"]["slots"] == "3"
    assert read(spark_head)["extra_args"].count("--kv-cache-memory") == 1


def test_set_fit_refuses_when_it_does_not_fit_and_writes_nothing(spark_head, monkeypatch):
    """slots 40 บนเครื่องที่ Gemma ถือ 19.5 อยู่ → ไม่พอ · bundle.args ต้องเป็นของเดิม · บอกว่าลด slots เหลือเท่าไร"""
    from lmds.cli.main import app
    from lmds.fleet.bundle_settings import read

    before = read(spark_head)
    r = CliRunner().invoke(app, ["set", "nemotron", "--fit", "--slots", "40"])
    assert r.exit_code == 1
    assert "ไม่เขียนค่าให้" in r.output and "ลด slots เหลือ 10" in r.output
    assert read(spark_head) == before


def test_fit_command_is_a_dry_run(spark_head):
    from lmds.cli.main import app
    from lmds.fleet.bundle_settings import read

    before = read(spark_head)
    r = CliRunner().invoke(app, ["fit", "nemotron", "--slots", "2", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["verdict"] == "fits" and d["kv_pin_gb"] == 3.5 and d["settings"]["slots"] == "2"
    assert d["others"] == [{"slug": "gemma4", "gb": 19.5}] and d["own_gb_now"] == 78.5
    assert read(spark_head) == before


# ── API ──────────────────────────────────────────────────────────────────────────
def test_local_fit_route_previews_then_applies(spark_head, monkeypatch):
    from lmds.web import state
    from lmds.web.api import create_app
    from lmds.fleet.bundle_settings import read

    state.STORE.set_local({"host": HOST, "models": MODELS})
    c = TestClient(create_app())
    r = c.post("/api/models/nemotron/fit", json={"slots": 2})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["applied"] is False and d["plan"]["verdict"] == "fits" and d["plan"]["kv_pin_gb"] == 3.5
    assert [s["kind"] for s in d["plan"]["stack"]] == ["os", "other", "weights", "overhead", "kv", "free"]
    r = c.post("/api/models/nemotron/fit", json={"slots": 2, "apply": True})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] is True and r.json()["restart_needed"] is True
    assert read(spark_head)["slots"] == "2" and f"--kv-cache-memory {int(3.5 * GIB)}" in read(spark_head)["extra_args"]
    # ไม่พอ → 409 พร้อมตาราง ไม่เขียน
    r = c.post("/api/models/nemotron/fit", json={"slots": 40, "apply": True})
    assert r.status_code == 409 and r.json()["plan"]["verdict"] == "no-fit" and "ลด slots" in r.json()["detail"]
    assert read(spark_head)["slots"] == "2"
    assert c.post("/api/models/nemotron/fit", json={"slots": "abc"}).status_code == 400
    assert c.post("/api/models/nope/fit", json={}).status_code == 404


def test_node_fit_route_runs_the_cli_on_that_machine(monkeypatch):
    """เครื่องอื่น: hub ไม่ลอกสูตร — สั่ง `lmds fit --json` / `lmds set --fit --json` ผ่าน SSH แล้วส่งผลกลับ"""
    from lmds.nodes.ssh import Result
    from lmds.web import state
    from lmds.web.api import create_app

    class N:
        name = "spark-head"

    seen = []
    plan = {"verdict": "fits", "slots": 2, "kv_pin_gb": 3.5, "stack": []}

    def fake_run(node, cmd, timeout=0):
        seen.append(cmd)
        if "set" in cmd:
            return Result(exit_code=0, stdout=json.dumps({"plan": plan, "saved": {"slots": "2"}}), stderr="")
        return Result(exit_code=0, stdout=json.dumps(plan), stderr="")

    monkeypatch.setattr("lmds.nodes.find", lambda name: N() if name == "spark-head" else None)
    monkeypatch.setattr("lmds.nodes.run", fake_run)
    state.STORE.set_node("spark-head", {"host": HOST, "models": [{"slug": "nemotron", "running": True}]})
    c = TestClient(create_app())
    r = c.post("/api/nodes/spark-head/models/nemotron/fit", json={"slots": 2, "context": 262144})
    assert r.status_code == 200, r.text
    assert r.json()["plan"]["kv_pin_gb"] == 3.5 and r.json()["applied"] is False
    assert seen[-1] == "lmds fit nemotron --json --slots 2 --context 262144"
    r = c.post("/api/nodes/spark-head/models/nemotron/fit", json={"slots": 2, "apply": True})
    assert r.status_code == 200 and r.json()["saved"] == {"slots": "2"} and r.json()["restart_needed"] is True
    assert seen[-1] == "lmds set nemotron --fit --json --slots 2"
    # เครื่องที่ยังไม่ได้อัปเดต — typer ตอบ No such command
    monkeypatch.setattr("lmds.nodes.run", lambda node, cmd, timeout=0: Result(exit_code=2, stdout="", stderr="Error: No such command 'fit'."))
    r = c.post("/api/nodes/spark-head/models/nemotron/fit", json={})
    assert r.status_code == 409 and "อัปเดต LMDS" in r.json()["detail"]
    # ไม่พอบนเครื่องนั้น — CLI พิมพ์ {"plan", "error"} exit 1 → 409 พร้อมตาราง
    monkeypatch.setattr("lmds.nodes.run", lambda node, cmd, timeout=0: Result(
        exit_code=1, stdout=json.dumps({"plan": {"verdict": "no-fit"}, "error": "ไม่พอ — ขาด 5 GB"}), stderr=""))
    r = c.post("/api/nodes/spark-head/models/nemotron/fit", json={"apply": True})
    assert r.status_code == 409 and r.json()["plan"]["verdict"] == "no-fit" and "ขาด 5 GB" in r.json()["detail"]


# ── ค่าตั้งต้นตอน deploy บนเครื่อง unified ─────────────────────────────────────
def test_new_vllm_bundle_on_a_spark_pins_kv_instead_of_relying_on_gpu_util(tmp_path):
    """Qwen3-32B (65 GiB · 64 layer × 8 kv-head × 128 = 256 KiB/token bf16) บน dgx-spark-single: analyzer เสนอ context 32,768
    (ขั้นสูงสุด ≤ native 40,960) = 8 GiB ต่อคำขอ → pin = 4 × 8 × 1.2 = 38.4 → 38.5 GiB (พอในที่เหลือ 121 − 12 OS − 65 − 3 = 41) ·
    RAM 106.5 แทน 0.85 × 121 = 103 ที่ไม่มีใครรู้ว่าไปไหน · controller มี --kv-cache-memory · profile บันทึก memory.sizing"""
    from tests.test_generator import make_bundle, safetensors_report

    bundle, plan, fit = make_bundle(safetensors_report(), tmp_path=tmp_path)
    assert plan.serving.context == 32768
    flags = [f for f in plan.serving.extra_flags if f.startswith("--kv-cache-memory")]
    assert len(flags) == 1
    pin_bytes = int(flags[0].split()[1])
    prof = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    sizing = prof["memory"]["sizing"]
    assert sizing["kv_pin_bytes"] == pin_bytes == int(38.5 * GIB) and sizing["source"] == "sizing" and sizing["slots"] == 4
    assert sizing["usable_gb"] == 109.0 and sizing["weights_gb"] == 65.0 and sizing["overhead_gb"] == 3.0
    assert sizing["kv_per_request_gb"] == 8.0 and sizing["capped"] is False
    assert sizing["ram_needed_gb"] == 106.5 and sizing["full_context_requests"] == 4
    controller = next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")
    assert f"--kv-cache-memory {pin_bytes}" in controller
    assert any("KV pin ตั้งต้น 38.5 GiB" in w for w in plan.warnings)
    # gpu-util ต้องลงมาเท่าที่ใช้จริง (106.5/121.6 + 2% ≈ 0.90 > 0.85 → คงเดิม) — เคส Qwen3-Embedding-8B dgx-spark04 2026-09-08:
    # pin 22 GiB แต่ gpu-util 0.85 ค้าง → vLLM ปฏิเสธ "Free memory 66/121 GiB < desired 103 GiB" ทั้งที่ต้องการ 39 GB
    assert plan.serving.gpu_memory_utilization <= 0.85


def test_default_pin_lowers_gpu_util_to_the_equivalent_for_small_models(tmp_path):
    """Qwen3-Embedding-8B (15 GB) บน Spark: pin 22 GiB → RAM ≈ 39 GB → gpu-util เทียบเท่า ≈ 0.35 ไม่ใช่ 0.85 — ไม่งั้น
    start ล้มทันทีที่มีโมเดลอื่นรันอยู่ (vLLM เช็ค free ≥ gpu-util × ทั้งเครื่อง แม้ pin แล้ว)"""
    from tests.test_generator import make_bundle, safetensors_report

    _bundle, plan, _fit = make_bundle(safetensors_report(weight_bytes=15 * GIB), tmp_path=tmp_path)
    assert any(f.startswith("--kv-cache-memory") for f in plan.serving.extra_flags)
    assert plan.serving.gpu_memory_utilization < 0.6, plan.serving.gpu_memory_utilization
    prof_sizing = plan.serving.gpu_memory_utilization
    assert abs(prof_sizing - (plan.serving.gpu_memory_utilization)) < 1e-9


def test_discrete_targets_and_explicit_recipe_pins_are_left_alone(tmp_path):
    from lmds.fit.sizing import apply_default_pin, sizing_for_plan
    from tests.test_generator import make_bundle, safetensors_report

    _bundle_, plan, fit = make_bundle(safetensors_report(weight_bytes=20 * GIB), target="rtx-5090", tmp_path=tmp_path / "rtx")
    assert not any(f.startswith("--kv-cache-memory") for f in plan.serving.extra_flags)
    assert sizing_for_plan(fit, plan) is None
    # สูตรที่ตั้ง pin ไว้เอง (GLM 3.5 GiB) — ห้ามแตะ และ profile บอกว่า explicit
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze

    report = safetensors_report()
    fit2 = analyze(report, PRESETS["dgx-spark-single"])
    plan2 = build_plan(report, fit2, provider=None)
    plan2.serving.extra_flags = [f for f in plan2.serving.extra_flags if not f.startswith("--kv-cache-memory")]
    plan2.serving.extra_flags.append("--kv-cache-memory 3758096384")
    assert apply_default_pin(plan2, fit2) is None
    assert plan2.serving.extra_flags.count("--kv-cache-memory 3758096384") == 1
    rec = sizing_for_plan(fit2, plan2)
    assert rec["source"] == "explicit" and rec["kv_pin_bytes"] == 3758096384


def test_default_pin_is_capped_to_what_is_left_for_big_kv_models(tmp_path):
    """โมเดล KV โต (dense 70B: 327,680 B/token) ที่ context เต็มงบ — 4 × เต็ม context × 1.2 ใหญ่กว่าที่เหลือ → pin เท่าที่เหลือ
    (RAM เท่า gpu-util เดิม แต่มีเลขให้ดู) และบอกว่าได้กี่คำขอเต็ม context"""
    from lmds.inspector.report import KvDims
    from tests.test_generator import make_bundle, safetensors_report

    report = safetensors_report(weight_bytes=70 * GIB, context_length=131072, kv_dims=KvDims(layers=80, kv_heads=8, head_dim=128))
    bundle, plan, fit = make_bundle(report, tmp_path=tmp_path)
    prof = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    sizing = prof["memory"]["sizing"]
    assert sizing["capped"] is True and sizing["kv_pin_gb"] == 36.0 and sizing["ram_needed_gb"] == 109.0
    # 36 GiB ถือคำขอเต็ม 131,072 (40 GiB) ไม่ได้ — vLLM จะปฏิเสธตอน start · ลด context ให้เท่าที่ pin ถือได้ (ขั้น 4096)
    assert plan.serving.context == 114688 and sizing["context"] == 114688
    assert sizing["kv_per_request_gb"] == 35.0 and sizing["full_context_requests"] == 1
    assert any("ลด context จาก 131,072" in w for w in plan.warnings)
    assert KV_PIN_HEADROOM == 1.2
