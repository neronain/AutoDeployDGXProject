"""ดึงสูตรจากรีโป controller ของทีม — อ่าน controller จริงแบบไม่รันสคริปต์

เคสในไฟล์นี้ลอกโครงมาจาก controller จริงใน dgx-spark-all-controllers ทั้งหมด
(ส่วนหัวแบบ vLLM, แบบ llama.cpp, แบบ stacked ที่มีฟังก์ชันคั่นก่อนบล็อกตั้งค่า)
"""

from __future__ import annotations

import pytest
import yaml

from lmds.recipes import find_recipe, load_catalog, synced_path
from lmds.recipes.controllers import parse_header, recipe_from_controller, scan_directory

VLLM_CONTROLLER = '''#!/usr/bin/env bash
# Qwen3-Coder-Next — controller
set -Eeuo pipefail

SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3-Coder-Next (NVFP4-GB10)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-vLLM (Docker)}"
MODEL_FEATURES="${MODEL_FEATURES:-code · tools}"

HF_REPO="${HF_REPO:-saricles/Qwen3-Coder-Next-NVFP4-GB10}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-coder-next}"
VLLM_IMAGE="${VLLM_IMAGE:-avarok/dgx-vllm-nvfp4-kernel:v23}"
MAX_NUM_SEQS="2"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.68}"
KV_CACHE_DTYPE="fp8"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/models/qwen}"

main() {
  MODEL_LABEL="ค่าในฟังก์ชันต้องไม่ถูกอ่าน"
  echo run
}
main "$@"
'''

STACKED_CONTROLLER = '''#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Gemma-4-31B-IT (NVFP4) · 2-node}"
RUNTIME_LABEL="${RUNTIME_LABEL:-vLLM (Docker, stacked)}"

prompt_cluster_config() {
  read -r -p "master ip: " MASTER_IP
}

MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
VLLM_IMAGE="vllm/vllm-openai:gemma4-cu130"
'''

GGUF_CONTROLLER = '''#!/usr/bin/env bash
SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3-VL-32B-Thinking (GGUF)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
MODEL_FEATURES="${MODEL_FEATURES:-vision · thinking · tools}"
HF_REPO="${HF_REPO:-unsloth/Qwen3-VL-32B-Thinking-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-Q4_K_M}"
MODEL_FILE="${MODEL_FILE:-Qwen3-VL-32B-Thinking-Q4_K_M.gguf}"
'''

MULTI_MODEL_CONTROLLER = '''#!/usr/bin/env bash
MODEL_LABEL="${MODEL_LABEL:-Red-team (สองโมเดลในไฟล์เดียว)}"
RUNTIME_LABEL="${RUNTIME_LABEL:-llama.cpp}"
GLM_REPO_ID="DavidAU/GLM-4.7-Flash-GGUF"
QWEN_REPO_ID="DavidAU/Qwen3.6-40B-GGUF"
'''


def test_header_stops_at_the_edge_of_the_config_block():
    meta = parse_header(VLLM_CONTROLLER)
    assert meta["MODEL_LABEL"] == "Qwen3-Coder-Next (NVFP4-GB10)"   # ไม่ใช่ค่าที่เขียนทับในฟังก์ชัน
    assert meta["HF_REPO"] == "saricles/Qwen3-Coder-Next-NVFP4-GB10"
    assert "MODEL_DIR" not in meta                                  # ค่าที่อ้างตัวแปรอื่นแปลไม่ได้


def test_vllm_controller_becomes_a_recipe():
    recipe = recipe_from_controller("qwen3-coder-next.sh", VLLM_CONTROLLER, "controllers@abc1234")
    assert recipe["match"] == "saricles/Qwen3-Coder-Next-NVFP4-GB10"
    assert recipe["engine"] == "vllm"
    assert recipe["image"] == "avarok/dgx-vllm-nvfp4-kernel:v23"
    assert recipe["serving"] == {"max_num_seqs": 2, "gpu_memory_utilization": 0.68,
                                 "kv_cache_dtype": "fp8"}
    assert recipe["topology"] == "single"
    assert "controllers@abc1234" in recipe["source"] and "qwen3-coder-next.sh" in recipe["source"]


def test_context_is_never_taken_from_the_controller():
    """context ต้องมาจากการวิเคราะห์เครื่องเป้าหมาย ไม่ใช่ค่าคงที่ของเครื่องที่เคยรัน"""
    recipe = recipe_from_controller("x.sh", VLLM_CONTROLLER)
    assert "context" not in recipe["serving"] and "max_model_len" not in recipe["serving"]


def test_stacked_controller_is_read_past_the_helper_function():
    """controller แบบ stacked มีฟังก์ชันถามค่าคั่นก่อนบล็อกตั้งค่าจริง — ต้องอ่านต่อให้ถึง"""
    recipe = recipe_from_controller("gemma4-31b-stacked.sh", STACKED_CONTROLLER)
    assert recipe["match"] == "nvidia/Gemma-4-31B-IT-NVFP4"
    assert recipe["topology"] == "stacked"


def test_gguf_controller_keeps_the_tested_file():
    recipe = recipe_from_controller("qwen3-vl.sh", GGUF_CONTROLLER)
    assert recipe["engine"] == "llamacpp"
    assert recipe["gguf_file"] == "Qwen3-VL-32B-Thinking-Q4_K_M.gguf"
    assert "image" not in recipe          # llama.cpp ไม่ได้รันด้วย image ของ vLLM
    assert recipe["notes"] == ["vision", "thinking", "tools"]


def test_multi_model_controller_is_skipped_not_guessed():
    assert recipe_from_controller("redteam.sh", MULTI_MODEL_CONTROLLER) is None


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_scan_reports_what_it_skipped(tmp_path):
    _write(tmp_path, "a-single.sh", VLLM_CONTROLLER)
    _write(tmp_path, "redteam-modelctl.sh", MULTI_MODEL_CONTROLLER)
    _write(tmp_path, "verify-all.sh", "#!/usr/bin/env bash\necho tool\n")
    recipes, skipped = scan_directory(tmp_path, "controllers@abc1234")
    assert [r["match"] for r in recipes] == ["saricles/Qwen3-Coder-Next-NVFP4-GB10"]
    assert any("redteam-modelctl.sh" in line for line in skipped)
    assert not any("verify-all.sh" in line for line in skipped)   # เครื่องมือของรีโป ไม่ใช่ controller


def test_a_recipe_without_an_engine_never_reaches_the_catalog(tmp_path):
    """RUNTIME_LABEL ที่ไม่มีคำที่รู้จัก → engine ว่าง → bundle ที่ไม่รู้ว่าจะรันด้วยอะไร

    เดิมด่านนี้อยู่ในชุดเทส (parametrize ทับ catalog ที่รวมของที่ sync มา) ซึ่งแปลว่า
    ต้องมีคน "รันเทส" บนเครื่องที่ sync แล้วเท่านั้นถึงจะเจอ · ย้ายมาตรวจตอน sync
    คือตอนที่ยังบอกได้ว่า controller ไฟล์ไหนเป็นต้นเหตุ
    """
    mystery = VLLM_CONTROLLER.replace(
        '${RUNTIME_LABEL:-vLLM (Docker)}', '${RUNTIME_LABEL:-เครื่องยนต์ที่ยังไม่รู้จัก}')
    # เอาตัวแปรเฉพาะ engine ออกด้วย — ไม่งั้น _engine เดา vllm จาก VLLM_IMAGE ได้
    # (พฤติกรรมใหม่ที่ถูกต้อง) · เทสนี้จงใจให้ "ไม่มีสัญญาณ engine เลย"
    mystery = mystery.replace(
        'VLLM_IMAGE="${VLLM_IMAGE:-avarok/dgx-vllm-nvfp4-kernel:v23}"', '')
    assert "vLLM" not in mystery and "VLLM_IMAGE" not in mystery, \
        "ตัวอย่างในเทสเปลี่ยนไป — replace ไม่โดนแล้ว"
    _write(tmp_path, "mystery-single.sh", mystery)
    recipes, skipped = scan_directory(tmp_path, "controllers@abc1234")

    assert recipes == [], "สูตรที่ไม่รู้ engine ต้องไม่เข้าแคตตาล็อก"
    assert any("mystery-single.sh" in line and "engine" in line for line in skipped), skipped


def test_a_complete_recipe_still_gets_through(tmp_path):
    """ด่านใหม่ต้องไม่กันของดีทิ้ง — สูตรปกติต้องมีครบทั้งสามฟิลด์อยู่แล้ว"""
    from lmds.recipes.controllers import REQUIRED_FIELDS

    _write(tmp_path, "a-single.sh", VLLM_CONTROLLER)
    recipes, skipped = scan_directory(tmp_path, "controllers@abc1234")

    assert len(recipes) == 1 and skipped == []
    assert all(recipes[0].get(field) for field in REQUIRED_FIELDS)


def test_single_wins_over_stacked_for_the_same_model(tmp_path):
    """โมเดลเดียวกันมีทั้ง single และ stacked — LMDS เลือก topology เองจากเครื่องที่มี"""
    _write(tmp_path, "gemma4-31b-stacked.sh", STACKED_CONTROLLER)
    _write(tmp_path, "gemma4-31b-single.sh", STACKED_CONTROLLER.replace("· 2-node", "")
           .replace("vLLM (Docker, stacked)", "vLLM (Docker)"))
    recipes, skipped = scan_directory(tmp_path)
    assert len(recipes) == 1 and recipes[0]["topology"] == "single"
    assert any("ซ้ำกับ gemma4-31b-single.sh" in line for line in skipped)


def test_synced_recipes_win_over_the_bundled_catalog(isolated_config, tmp_path):
    """รีโปของทีมคือต้นทางที่รันจริงและอัปเดตบ่อยกว่า catalog ที่ฝังมากับเวอร์ชันที่ติดตั้งไว้"""
    bundled = find_recipe("meta-llama/Llama-3.3-70B-Instruct")
    assert bundled is not None and bundled.image != "team/custom-image:v9"

    synced_path().parent.mkdir(parents=True, exist_ok=True)
    synced_path().write_text(yaml.safe_dump({"version": 1, "recipes": [
        {"match": "meta-llama/Llama-3.3-70B-Instruct", "label": "ของทีม", "engine": "vllm",
         "image": "team/custom-image:v9", "controller": "llama33-70b-single.sh"},
    ]}, allow_unicode=True), encoding="utf-8")
    load_catalog.cache_clear()

    updated = find_recipe("meta-llama/Llama-3.3-70B-Instruct")
    assert updated.image == "team/custom-image:v9"
    assert updated.controller == "llama33-70b-single.sh"


@pytest.fixture(autouse=True)
def _fresh_catalog():
    """แคตตาล็อกถูก cache ไว้ — เทสที่เขียนไฟล์ต้องไม่ไปกวนเทสตัวอื่น"""
    load_catalog.cache_clear()
    yield
    load_catalog.cache_clear()


def test_old_controller_without_runtime_label_infers_engine(tmp_path):
    """controller รุ่นเก่าไม่มี RUNTIME_LABEL แต่มี LLAMACPP_IMAGE/LLAMA_CPP_REPO ครบ —
    ต้องเดา engine เป็น llamacpp ได้ ไม่ใช่ตกไปเป็น 'ไม่รู้ engine' แล้ว skip

    เจอจริง: qwen3-coder-30b-a3b-instruct ที่ generate ด้วย lmds เก่าถูกข้ามตอน sync
    """
    old = '''#!/bin/bash
MODEL_ID="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_LABEL="Qwen3 Coder 30B"
RUNTIME_MODE="${RUNTIME_MODE:-native}"
LLAMACPP_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
MODEL_FILE="Qwen3-Coder-30B.Q6_K.gguf"
'''
    _write(tmp_path, "qwen3-coder-old-single.sh", old)
    recipes, skipped = scan_directory(tmp_path, "controllers@abc1234")
    match = next((r for r in recipes if "Qwen3-Coder-30B" in r["match"]), None)
    assert match is not None, f"ควรอ่านได้ แต่ skip: {skipped}"
    assert match["engine"] == "llamacpp"
