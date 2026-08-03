"""Regression เทียบมาตรฐาน controllers v3.0.0 — กฎชุดเดียวกับ `audit-controllers.py` ของ repo เดิม

ROADMAP ประกาศไว้ตั้งแต่ต้นว่า "ทุก PR ต้องผ่าน regression เทียบ controllers v3.0.0" แต่ไม่เคยมี
อะไรบังคับจริง ไฟล์นี้คือตัวบังคับนั้น — port กฎมาทั้งชุดแทนที่จะ vendor controller ตัวอ้างอิง
21 ไฟล์ (~400 KB) เข้ารีโป เพราะสิ่งที่ต้องคงไว้คือ *กฎ* ไม่ใช่ไฟล์

ยืนยันแล้วเมื่อ 2026-08-03: controller อ้างอิงทั้ง 21 ตัวผ่านกฎชุดนี้ 0 error 0 warning
และ bundle ที่ LMDS generate ทุกแบบก็ผ่านเท่ากัน

ที่มา: neronain/dgx-spark-all-controllers/audit-controllers.py
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tests.test_generator import gguf_report, make_bundle, safetensors_report

# ── regex ชุดเดียวกับ audit-controllers.py (คัดลอกมาตรง ๆ ห้ามแก้ให้หลวมลง) ──
SCRIPT_VERSION_DECL = re.compile(r'(?m)^SCRIPT_VERSION="\$\{SCRIPT_VERSION:-[0-9]+\.[0-9]+\.[0-9]+\}"')
BANNER_DEF = re.compile(r"(?m)^banner\(\) \{")
INFO_DEF = re.compile(r"(?m)^info\(\) \{")
INFO_DISPATCH = re.compile(r"(?m)^\s*info\|banner\)")
AUTHOR_USERNAME = re.compile(r"neronain")
AUTHOR_CREDIT_MARKER = "neronain.minidev"
CLUSTER_PROMPT = re.compile(r"(?m)^prompt_cluster_config\(\) \{")
PURE_NUMERIC_SEPARATOR = re.compile(r"\b\d+(?:_\d+)+\b")
PIPEFAIL_GREP_Q = re.compile(r"\|[^\n]*\bgrep\b[^\n]*-[A-Za-z]*q[A-Za-z]*")
FIXED_API_PORT = re.compile(r'(?m)^API_PORT="(?!\$\{API_PORT:-)')
FIXED_CONTEXT = re.compile(r'(?m)^(?:CTX_SIZE|MAX_MODEL_LEN)="(?!\$\{(?:CTX_SIZE|MAX_MODEL_LEN):-)')
FIXED_SINGLE_MASTER = re.compile(r'(?m)^MASTER_IP="[^"]+"')
DIRECT_FIRST_IP = re.compile(r"hostname -I[^\n]*awk[^\n]*\{print \$1\}")


def audit(path) -> list[str]:
    """คืนรายการ finding — ว่าง = ผ่านเหมือน controller อ้างอิง"""
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []

    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    if proc.returncode:
        findings.append(f"bash-syntax: {proc.stderr.strip()}")

    for match in PURE_NUMERIC_SEPARATOR.finditer(text):
        findings.append(f"numeric-separator: {match.group(0)}")

    if "set -Eeuo pipefail" in text or "set -o pipefail" in text:
        for match in PIPEFAIL_GREP_Q.finditer(text):
            findings.append(f"pipefail-grep-q: {match.group(0).strip()}")

    is_stacked = (
        "stacked" in path.name.lower()
        or "WORKER_IP=" in text
        or 'TENSOR_PARALLEL_SIZE="2"' in text
    )

    if not is_stacked and FIXED_SINGLE_MASTER.search(text):
        findings.append("fixed-single-master-ip")
    if FIXED_API_PORT.search(text):
        findings.append("non-overridable-api-port")
    if FIXED_CONTEXT.search(text):
        findings.append("non-overridable-context")
    if "--context" not in text:
        findings.append("missing-context-option")
    if "--port" not in text:
        findings.append("missing-port-option")
    if "network-info" not in text or "detect_advertise_ip" not in text:
        findings.append("missing-network-selection")
    if not SCRIPT_VERSION_DECL.search(text):
        findings.append("missing-script-version")
    if not (BANNER_DEF.search(text) and INFO_DEF.search(text) and INFO_DISPATCH.search(text)):
        findings.append("missing-banner-info")

    for number, line in enumerate(text.splitlines(), start=1):
        if AUTHOR_CREDIT_MARKER in line:
            continue
        if AUTHOR_USERNAME.search(line):
            findings.append(f"hard-coded-author-username:{number}")

    if is_stacked and not CLUSTER_PROMPT.search(text):
        findings.append("missing-cluster-prompt")

    for match in DIRECT_FIRST_IP.finditer(text):
        context = text[max(0, match.start() - 1200):match.start()]
        if "detect_advertise_ip()" not in context:
            findings.append("first-hostname-ip")

    return findings


CASES = [
    ("vllm-spark", "safetensors", "dgx-spark-single"),
    ("vllm-rtx", "safetensors", "rtx-5090"),
    ("llamacpp-spark", "gguf", "dgx-spark-single"),
    ("llamacpp-rtx", "gguf", "rtx-5090"),
    ("vllm-stacked", "safetensors", "dgx-spark-stacked"),
]


@pytest.mark.parametrize("name,kind,target", CASES)
def test_generated_controller_matches_v3_standard(isolated_config, tmp_path, name, kind, target):
    report = safetensors_report() if kind == "safetensors" else gguf_report()
    bundle, _, _ = make_bundle(report, target=target, tmp_path=tmp_path / name)
    findings = audit(bundle.controller)
    assert findings == [], f"{name} ไม่ผ่านมาตรฐาน v3.0.0: {findings}"


def test_banner_carries_required_metadata(isolated_config, tmp_path):
    """banner ต้องอ่านแล้วรู้ทันทีว่า controller นี้คืออะไร — ไม่ใช่แค่ ASCII art ผ่าน regex"""
    bundle, plan, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")

    assert f'MODEL_LABEL="${{MODEL_LABEL:-{plan.model_id}}}"' in text
    assert 'RUNTIME_LABEL="${RUNTIME_LABEL:-vLLM (Docker)}"' in text
    assert "MODEL_FEATURES=" in text
    # credit ของผู้เขียนต้องมี marker ไม่งั้น audit จะจับเป็น hard-coded username
    assert AUTHOR_CREDIT_MARKER in text


def test_llamacpp_on_spark_labels_native_build(isolated_config, tmp_path):
    """Spark ไม่มี image ทางการ — banner ต้องบอกว่าเป็น native build ไม่ใช่ Docker"""
    bundle, _, _ = make_bundle(gguf_report(), target="dgx-spark-single", tmp_path=tmp_path)
    assert "llama.cpp (native build)" in bundle.controller.read_text(encoding="utf-8")
