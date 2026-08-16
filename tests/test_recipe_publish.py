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
