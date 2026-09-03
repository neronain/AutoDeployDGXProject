"""publish controller ที่รันผ่านแล้วขึ้นคลัง — ทางกลับของ recipes/sync

ยึดกฎเดียว: ส่งเฉพาะค่าของโมเดล · ค่าของเครื่อง (port/context/slots) ต้องไม่หลุดขึ้นไป
"""

from __future__ import annotations

import subprocess

import yaml

from lmds.recipes.publish import (
    build_profile,
    measured_features,
    publish,
    stamp_features,
)

PROFILE = {
    "model": {"id": "org/Coder-30B-GGUF", "revision": "abc123",
              "selected_gguf": "Coder-30B.Q6_K.gguf"},
    "runtime": {"engine": "llamacpp"},
    "serving": {"context": 131072, "max_num_seqs": 1},   # ค่าของเครื่อง — ต้องไม่ไปโผล่
    "features": {
        "multimodal": {"modalities": ["image", "text"]},
        "tool_calling": {"enabled": True, "parser": "qwen3_coder"},
        "reasoning": {"enabled": False, "parser": None},
    },
}

CONTROLLER = '''#!/bin/bash
# coder-30b — llama.cpp single-node controller (GGUF)
RUNTIME_LABEL="llama.cpp (native build)"
MODEL_ID="org/Coder-30B-GGUF"
MODEL_LABEL="Coder 30B"
API_PORT="${API_PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-131072}"
'''


def _ctl(tmp_path):
    p = tmp_path / "coder-30b-single.sh"
    p.write_text(CONTROLLER)
    p.chmod(0o755)
    return p


def test_measured_features_reads_only_what_the_profile_confirms():
    assert measured_features(PROFILE) == ["vision", "tools (qwen3_coder)"]
    # reasoning enabled=false → ไม่ใส่


def test_stamp_features_inserts_then_replaces():
    once = stamp_features(CONTROLLER, ["vision", "tools (qwen3_coder)"])
    assert 'MODEL_FEATURES="vision · tools (qwen3_coder)"' in once
    twice = stamp_features(once, ["tools (qwen3_coder)"])
    assert twice.count("MODEL_FEATURES=") == 1          # แทน ไม่ใช่เพิ่มซ้ำ
    assert 'MODEL_FEATURES="tools (qwen3_coder)"' in twice


def test_profile_carries_no_machine_specific_values():
    text = build_profile("coder-30b", PROFILE, "2026-08-16", "msi-3")
    data = yaml.safe_load(text)
    assert data["source"]["revision"] == "abc123"
    assert data["measured_features"] == ["vision", "tools (qwen3_coder)"]
    # กฎเหล็ก: ไม่มี key/ค่าของเครื่องใน data (โน้ตอธิบายกฎมีคำว่า port ได้ — ดู values เท่านั้น)
    assert "serving" not in data
    assert not {"port", "context", "slots"} & set(data)
    values = yaml.safe_dump({k: v for k, v in data.items() if k != "note"})
    assert "8000" not in values and "131072" not in values


def test_publish_writes_controller_and_profile_and_commits(tmp_path):
    store = tmp_path / "store"
    result = publish("coder-30b", _ctl(tmp_path), PROFILE,
                     repo=str(store), now="2026-08-16", host="msi-3", push=False)
    assert result["committed"] is True and result["remote"] is False
    dest = store / "controllers" / "coder-30b"
    assert (dest / "coder-30b-single.sh").exists()
    assert (dest / "PROFILE.yaml").exists()
    # MODEL_FEATURES ถูก stamp ลง controller ที่ publish
    assert "MODEL_FEATURES=" in (dest / "coder-30b-single.sh").read_text()
    # commit จริง
    log = subprocess.run(["git", "-C", str(store), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    assert "publish coder-30b" in log


def test_publish_is_idempotent(tmp_path):
    store = tmp_path / "store"
    publish("coder-30b", _ctl(tmp_path), PROFILE, repo=str(store), now="2026-08-16",
            host="msi-3", push=False)
    again = publish("coder-30b", _ctl(tmp_path), PROFILE, repo=str(store),
                    now="2026-08-16", host="msi-3", push=False)
    assert again["committed"] is False        # ไม่มีอะไรเปลี่ยน = ไม่ commit ซ้ำ


def test_published_controller_never_carries_the_bundle_env(tmp_path):
    """bundle.env (port/context ของเครื่อง) เป็นคนละไฟล์ ต้องไม่ตามขึ้นไปกับ controller"""
    (tmp_path / "bundle.env").write_text('API_PORT="${API_PORT:-8010}"\n')
    store = tmp_path / "store"
    publish("coder-30b", _ctl(tmp_path), PROFILE, repo=str(store), now="d", host="h", push=False)
    dest = store / "controllers" / "coder-30b"
    assert not (dest / "bundle.env").exists()


def test_operator_supplied_features_override_the_rulebased_profile(tmp_path):
    """coder ที่มี tools จริงแต่ profile (rule-based) เขียน enabled=false —
    คนที่เพิ่ง test มา ระบุเองได้และต้องชนะ"""
    rulebased_says_no_tools = {
        **PROFILE,
        "features": {"tool_calling": {"enabled": False}, "reasoning": {"enabled": False}},
    }
    store = tmp_path / "store"
    result = publish("coder-30b", _ctl(tmp_path), rulebased_says_no_tools,
                     features=["tools (qwen3_coder)"], repo=str(store),
                     now="d", host="h", push=False)
    assert result["features"] == ["tools (qwen3_coder)"]
    ctl = (store / "controllers" / "coder-30b" / "coder-30b-single.sh").read_text()
    assert 'MODEL_FEATURES="tools (qwen3_coder)"' in ctl
    profile_yaml = yaml.safe_load(
        (store / "controllers" / "coder-30b" / "PROFILE.yaml").read_text())
    assert profile_yaml["measured_features"] == ["tools (qwen3_coder)"]


def test_publish_then_scan_reads_it_back(tmp_path):
    """ลูปต้องปิด: สิ่งที่ publish เขียน (controllers/<slug>/) ฝั่ง sync ต้องอ่านเป็นสูตรได้

    scan_directory เดิม glob เฉพาะ root — controller ที่ publish วางใน subdir จึงหาย
    เงียบ · ตอนนี้ rglob ครอบทั้งสองแบบ
    """
    from lmds.recipes.controllers import scan_directory

    store = tmp_path / "store"
    publish("coder-30b", _ctl(tmp_path), PROFILE, features=["tools (qwen3_coder)"],
            repo=str(store), now="d", host="h", push=False)
    recipes, _ = scan_directory(store)
    match = next((r for r in recipes if r["match"] == "org/Coder-30B-GGUF"), None)
    assert match is not None                      # อ่านเจอทั้งที่อยู่ใน controllers/<slug>/
    assert "tools (qwen3_coder)" in match["notes"]   # measured features เดินทางมาถึง


VLLM_CONTROLLER = '''#!/bin/bash
# q36 — vLLM single-node controller
RUNTIME_LABEL="vLLM (Docker)"
MODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_LABEL="Qwen3.6 35B"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm@sha256:654e}"
ENGINE_ENV="${ENGINE_ENV:-}"
EXTRA_SERVE_ARGS_DEFAULT=''
BUNDLE_ARGS="${BUNDLE_ARGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bundle.args}"
if [[ -z "${EXTRA_SERVE_ARGS:-}" && -f "$BUNDLE_ARGS" ]]; then
  EXTRA_SERVE_ARGS="$(tr '\\n' ' ' < "$BUNDLE_ARGS")"
fi
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-$EXTRA_SERVE_ARGS_DEFAULT}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
'''


def _vllm_bundle(tmp_path):
    ctl = tmp_path / "q36-single.sh"
    ctl.write_text(VLLM_CONTROLLER)
    ctl.chmod(0o755)
    (tmp_path / "bundle.env").write_text(
        'VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:61fc}"\n'
        'TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"\n'
        'REASONING_PARSER="${REASONING_PARSER:-qwen3}"\n'
        'API_PORT="${API_PORT:-8000}"\n'
        'MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"\n'
        'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"\n')
    (tmp_path / "bundle.args").write_text(
        '--kv-cache-dtype fp8\n--speculative-config {"method":"mtp","num_speculative_tokens":3}\n')
    return ctl


def test_publish_folds_proven_model_values_from_bundle_env_and_args(tmp_path):
    """เคสจริง spark04/spark-worker (2026-09-03): รันได้เพราะ lmds set --image/--tool-parser/
    --extra-args แต่ header ของ controller ยังเป็นค่า plan ที่ล้ม — publish แบบเดิมส่งค่าที่ล้มขึ้นคลัง
    """
    from lmds.recipes.controllers import parse_header

    store = tmp_path / "store"
    result = publish("q36", _vllm_bundle(tmp_path), PROFILE, repo=str(store), now="d", host="spark04",
                     push=False)
    assert result["unfolded"] == []
    ctl = (store / "controllers" / "q36" / "q36-single.sh").read_text()
    meta = parse_header(ctl)
    # ค่าของโมเดล — พับลงมา
    assert meta["VLLM_IMAGE"] == "vllm/vllm-openai@sha256:61fc"
    assert meta["TOOL_CALL_PARSER"] == "qwen3_xml" and meta["REASONING_PARSER"] == "qwen3"
    assert meta["EXTRA_SERVE_ARGS_DEFAULT"] == \
        '--kv-cache-dtype fp8 --speculative-config {"method":"mtp","num_speculative_tokens":3}'
    # ค่าของเครื่อง — ต้องยังเป็นค่าเดิมของ controller ไม่ใช่ของ spark04
    assert meta["API_PORT"] == "8000" and meta["MAX_MODEL_LEN"] == "131072"
    assert meta["GPU_MEMORY_UTILIZATION"] == "0.4"
    # provenance บอกคน review ว่าค่าไหนถูกทับ
    profile_yaml = yaml.safe_load((store / "controllers" / "q36" / "PROFILE.yaml").read_text())
    assert profile_yaml["overrides"]["VLLM_IMAGE"] == "vllm/vllm-openai@sha256:61fc"
    assert "API_PORT" not in profile_yaml["overrides"]


def test_folded_extra_args_survive_bash_with_json_braces(tmp_path):
    """JSON ใน ${VAR:-…} ถูก } ตัดขาด — นั่นคือเหตุผลของ single quote · ต้องพิสูจน์ด้วย bash จริง"""
    store = tmp_path / "store"
    publish("q36", _vllm_bundle(tmp_path), PROFILE, repo=str(store), now="d", host="h", push=False)
    ctl = store / "controllers" / "q36" / "q36-single.sh"
    header = "\n".join(l for l in ctl.read_text().splitlines()
                       if not l.startswith("#!")) + '\nprintf "%s" "$EXTRA_SERVE_ARGS"\n'
    out = subprocess.run(["bash", "-c", header], capture_output=True, text=True,
                         cwd=tmp_path / "elsewhere" if (tmp_path / "elsewhere").mkdir() is None else tmp_path,
                         env={"PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout == '--kv-cache-dtype fp8 --speculative-config {"method":"mtp","num_speculative_tokens":3}'


def test_publish_reports_keys_an_old_controller_cannot_hold(tmp_path):
    """controller รุ่นก่อน EXTRA_SERVE_ARGS_DEFAULT — บอกว่าพับไม่ได้ ดีกว่าหายเงียบ"""
    ctl = _ctl(tmp_path)                       # header เก่า ไม่มีบรรทัด image/parser/args
    (tmp_path / "bundle.args").write_text("--jinja\n")
    (tmp_path / "bundle.env").write_text('LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-ghcr.io/x@sha256:1}"\n')
    result = publish("coder-30b", ctl, PROFILE, repo=str(tmp_path / "store"), now="d", host="h",
                     push=False)
    assert set(result["unfolded"]) == {"EXTRA_SERVE_ARGS", "LLAMACPP_IMAGE"}


def test_header_parser_reads_single_quoted_defaults():
    from lmds.recipes.controllers import parse_header, recipe_from_controller

    text = VLLM_CONTROLLER.replace("EXTRA_SERVE_ARGS_DEFAULT=''",
                                   "EXTRA_SERVE_ARGS_DEFAULT='--a {\"b\":1}'")
    text = text.replace('TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"', 'TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"')
    meta = parse_header(text)
    assert meta["EXTRA_SERVE_ARGS_DEFAULT"] == '--a {"b":1}'
    recipe = recipe_from_controller("q36-single.sh", text)
    assert recipe["extra_args"] == '--a {"b":1}' and recipe["tool_parser"] == "qwen3_xml"


def test_publish_commits_into_a_cloned_repo_on_a_hub_with_no_git_identity(tmp_path, monkeypatch):
    """เคสจริง 2026-09-03: hub ไม่เคยตั้ง user.email · repo candidates เป็น clone ไม่ใช่ init เอง
    → "unable to auto-detect email address" ทั้ง 23 ตัว · identity ต้องถูกตั้งเป็น local ของ repo นั้น
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home-without-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    (tmp_path / "home-without-gitconfig").mkdir()
    origin = tmp_path / "origin"
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
    (origin / "README.md").write_text("candidates\n")
    subprocess.run(["git", "-C", str(origin), "-c", "user.email=a@b", "-c", "user.name=a", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "-m", "init"], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)

    result = publish("coder-30b", _ctl(tmp_path), PROFILE, repo=str(clone), now="d", host="hub-x", push=False)
    assert result["committed"] is True
    author = subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%an <%ae>"],
                            capture_output=True, text=True).stdout.strip()
    assert author == "lmds (hub-x) <lmds@hub-x>"
    # ไม่แตะ global ของเครื่อง
    assert not (tmp_path / "home-without-gitconfig" / ".gitconfig").exists()


def test_the_other_engines_image_key_is_not_reported_as_unfolded(tmp_path):
    """`lmds set --image` เขียนทั้ง VLLM_IMAGE และ LLAMACPP_IMAGE ลง bundle.env · controller vLLM
    มีแค่ VLLM_IMAGE — LLAMACPP_IMAGE ที่เหลือไม่ใช่ของที่พับไม่ได้ (เจอจริง spark04/spark-worker)"""
    from lmds.recipes.publish import bundle_overrides, fold_overrides

    _vllm_bundle(tmp_path)
    (tmp_path / "bundle.env").write_text(
        'VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:61fc}"\n'
        'LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-vllm/vllm-openai@sha256:61fc}"\n')
    text, applied, skipped = fold_overrides(VLLM_CONTROLLER, bundle_overrides(tmp_path))
    assert applied["VLLM_IMAGE"] == "vllm/vllm-openai@sha256:61fc"
    assert skipped == []
