"""Quality gates ของ bundle — สืบทอด audit-controllers.py + quality-gates.md ของ v3.0.0

ทุก bundle ต้องผ่านทุก gate ก่อนถึงมือผู้ใช้ (PRD §10) — gate เขียนให้ตรวจได้ทั้ง
bundle ที่ LMDS generate เองและ bundle เดิม/แก้มือ (`lmds validate <dir>`)
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

CHECKSUM_FILE = "PACKAGE_SHA256SUMS"

REQUIRED_FLAGS = [
    "--context",
    "--port",
    "--bind",
    "--advertise-ip",
    "--interface",
    "--client-input",
    "--client-output",
]
REQUIRED_COMMANDS = [
    "download",
    "verify-files",
    "start",
    "stop",
    "restart",
    "status",
    "logs",
    "client-config",
    "network-info",
]

# pattern secret ที่ห้ามอยู่ใน bundle (สอดคล้อง redaction filter)
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
]

_NUMERIC_UNDERSCORE = re.compile(r"\(\(\s*[^)]*\b\d+_\d+")
_PIPE_GREP_Q = re.compile(r"\|\s*grep\s+-q")
# `\` ปิดบรรทัดแล้วตามด้วยบรรทัดว่าง = คำสั่งขาดตอนกลางทาง (bash -n จับไม่ได้ — เจอจริงบน gigabyte02)
_BROKEN_CONTINUATION = re.compile(r"\\\n[ \t]*\n")

_PROFILE_REQUIRED_PATHS = [
    ("model", "id"),
    ("model", "revision"),
    ("runtime", "engine"),
    ("serving", "context"),
    ("validation",),
]


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


def _controllers(bundle_dir: Path) -> list[Path]:
    return sorted(bundle_dir.glob("*.sh"))


def _text_files(bundle_dir: Path) -> list[Path]:
    return [
        p for p in sorted(bundle_dir.rglob("*"))
        if p.is_file() and p.suffix not in {".zip", ".gguf", ".safetensors"}
    ]


def gate_bash_syntax(bundle_dir: Path) -> GateResult:
    scripts = _controllers(bundle_dir)
    if not scripts:
        return GateResult("bash-syntax", False, "ไม่พบสคริปต์ .sh ใน bundle")
    for script in scripts:
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        if proc.returncode != 0:
            return GateResult("bash-syntax", False, f"{script.name}: {proc.stderr.strip()[:200]}")
    return GateResult("bash-syntax", True, f"{len(scripts)} สคริปต์")


def gate_numeric_underscore(bundle_dir: Path) -> GateResult:
    for script in _controllers(bundle_dir):
        if _NUMERIC_UNDERSCORE.search(script.read_text(encoding="utf-8")):
            return GateResult("numeric-underscore", False, f"{script.name}: numeric underscore ใน arithmetic")
    return GateResult("numeric-underscore", True)


def gate_pipefail_safe(bundle_dir: Path) -> GateResult:
    for script in _controllers(bundle_dir):
        text = script.read_text(encoding="utf-8")
        if _PIPE_GREP_Q.search(text):
            return GateResult("pipefail-safe", False, f"{script.name}: พบ '| grep -q'")
        if "set -Eeuo pipefail" not in text and "set -euo pipefail" not in text:
            return GateResult("pipefail-safe", False, f"{script.name}: ไม่มี set -Eeuo pipefail")
    return GateResult("pipefail-safe", True)


def gate_line_continuation(bundle_dir: Path) -> GateResult:
    for script in _controllers(bundle_dir):
        text = script.read_text(encoding="utf-8")
        match = _BROKEN_CONTINUATION.search(text)
        if match:
            line_no = text[: match.start()].count("\n") + 1
            return GateResult(
                "line-continuation", False,
                f"{script.name}:{line_no}: บรรทัดต่อด้วย \\ แล้วตามด้วยบรรทัดว่าง — คำสั่งขาดตอน",
            )
    return GateResult("line-continuation", True)


# marker ที่ audit-controllers.py ของ v3.0.0 บังคับ — ไม่ใช่แค่ flag/คำสั่ง
_VERSION_DECL = re.compile(r'(?m)^SCRIPT_VERSION="\$\{SCRIPT_VERSION:-[0-9]+\.[0-9]+\.[0-9]+\}"')
_BANNER_DEF = re.compile(r"(?m)^banner\(\) \{")
_INFO_DEF = re.compile(r"(?m)^info\(\) \{")
_INFO_DISPATCH = re.compile(r"(?m)^\s*info\|banner\)")


def gate_contract(bundle_dir: Path) -> GateResult:
    missing: list[str] = []
    for script in _controllers(bundle_dir):
        text = script.read_text(encoding="utf-8")
        for flag in REQUIRED_FLAGS:
            if flag + ")" not in text:
                missing.append(f"{script.name}: {flag}")
        for command in REQUIRED_COMMANDS:
            if f"{command})" not in text:
                missing.append(f"{script.name}: คำสั่ง {command}")
        if not _VERSION_DECL.search(text):
            missing.append(f'{script.name}: SCRIPT_VERSION="${{SCRIPT_VERSION:-X.Y.Z}}"')
        if not (_BANNER_DEF.search(text) and _INFO_DEF.search(text) and _INFO_DISPATCH.search(text)):
            missing.append(f"{script.name}: banner()/info() + dispatch info|banner)")
    if missing:
        return GateResult("controller-contract", False, "; ".join(missing[:5]))
    return GateResult("controller-contract", True)


# marker ที่ controller stacked (multi-node) ต้องมีจริง — กัน bundle ที่ตั้งใจ stacked
# แต่ถูก render เป็น single-node (จะ "ผ่าน" contract เดี่ยวแต่รันจริงไม่ได้)
_STACKED_FLAG_MARKERS = ["--nnodes", "--node-rank", "--headless", "--distributed-executor-backend"]
_STACKED_COMMAND_MARKERS = ["prepare-runtime", "sync-worker", "verify-worker"]
# stacked ต้องถาม IP/user ของคลัสเตอร์ก่อน start — ค่า default ในไฟล์เป็นแค่ตัวอย่าง
_STACKED_CLUSTER_PROMPT = re.compile(r"(?m)^prompt_cluster_config\(\) \{")


def gate_stacked_contract(bundle_dir: Path) -> GateResult:
    """ถ้า MODEL_PROFILE.yaml ระบุ topology: stacked — controller ต้องมี multi-node machinery จริง"""
    profile_path = bundle_dir / "MODEL_PROFILE.yaml"
    if not profile_path.exists():
        return GateResult("stacked-contract", True, "n/a (ไม่มี profile)")
    try:
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return GateResult("stacked-contract", True, "n/a (profile อ่านไม่ได้ — ปล่อยให้ gate อื่นจับ)")
    if not isinstance(data, dict) or data.get("topology") != "stacked":
        return GateResult("stacked-contract", True, "n/a (ไม่ใช่ stacked)")

    scripts = _controllers(bundle_dir)
    if not scripts:
        return GateResult("stacked-contract", False, "topology stacked แต่ไม่พบสคริปต์ controller")
    missing: list[str] = []
    combined = "\n".join(s.read_text(encoding="utf-8") for s in scripts)
    for flag in _STACKED_FLAG_MARKERS:
        if flag not in combined:
            missing.append(flag)
    for command in _STACKED_COMMAND_MARKERS:
        if f"{command})" not in combined:
            missing.append(f"คำสั่ง {command}")
    if "ssh" not in combined:
        missing.append("SSH orchestration (worker)")
    if not _STACKED_CLUSTER_PROMPT.search(combined):
        missing.append("prompt_cluster_config()")
    if missing:
        return GateResult(
            "stacked-contract", False,
            "topology stacked แต่ controller ขาด multi-node machinery: " + ", ".join(missing[:6]),
        )
    return GateResult("stacked-contract", True, "multi-node machinery ครบ")


def gate_multimodal_assets(bundle_dir: Path) -> GateResult:
    """profile ประกาศ mmproj ไว้ → controller ต้องโหลดไฟล์นั้นและส่ง --mmproj จริง

    เคสจริง (gemma-4-12b-it-GGUF, 2026-08-03): MODEL_PROFILE + SPECIAL_FILES บอกว่าต้องมี
    mmproj-BF16.gguf แต่ controller ไม่มีคำว่า mmproj เลย — download มาไฟล์เดียว, start ผ่าน,
    /health เขียว แต่โมเดลรับแต่ข้อความ ไม่มี error ให้เห็นเลยสักจุด
    """
    profile_path = bundle_dir / "MODEL_PROFILE.yaml"
    if not profile_path.exists():
        return GateResult("multimodal-assets", True, "n/a (ไม่มี profile)")
    try:
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return GateResult("multimodal-assets", True, "n/a (profile อ่านไม่ได้)")
    if not isinstance(data, dict):
        return GateResult("multimodal-assets", True, "n/a")

    features = data.get("features") or {}
    multimodal = features.get("multimodal") or {}
    projectors = multimodal.get("projector_files") or []
    if not projectors:
        return GateResult("multimodal-assets", True, "n/a (ไม่ใช่ multimodal)")
    if (data.get("runtime") or {}).get("engine") != "llamacpp":
        # vLLM โหลด vision tower มาจาก safetensors ของ repo อยู่แล้ว ไม่มีไฟล์ mmproj แยก
        return GateResult("multimodal-assets", True, "n/a (ไม่ใช่ llama.cpp)")

    scripts = _controllers(bundle_dir)
    if not scripts:
        return GateResult("multimodal-assets", False, "ประกาศ mmproj แต่ไม่พบสคริปต์ controller")
    combined = "\n".join(s.read_text(encoding="utf-8") for s in scripts)

    missing: list[str] = []
    if "--mmproj" not in combined:
        missing.append("ไม่ส่ง --mmproj ให้ llama-server (โมเดลจะกลายเป็น text-only)")
    for name in projectors:
        if name.rsplit("/", 1)[-1] not in combined:
            missing.append(f"ไม่ได้ดาวน์โหลด {name}")
    if missing:
        return GateResult("multimodal-assets", False, "; ".join(missing[:4]))
    return GateResult("multimodal-assets", True, f"mmproj ครบ {len(projectors)} ไฟล์")


def gate_profile_schema(bundle_dir: Path) -> GateResult:
    path = bundle_dir / "MODEL_PROFILE.yaml"
    if not path.exists():
        return GateResult("profile-schema", False, "ไม่พบ MODEL_PROFILE.yaml")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return GateResult("profile-schema", False, f"YAML ไม่ถูกต้อง: {exc}")
    if not isinstance(data, dict):
        return GateResult("profile-schema", False, "MODEL_PROFILE.yaml ไม่ใช่ mapping")
    for key_path in _PROFILE_REQUIRED_PATHS:
        node = data
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                return GateResult("profile-schema", False, f"ขาด field: {'.'.join(key_path)}")
            node = node[key]
    revision = data["model"]["revision"]
    if not revision or revision in {"main", "latest"}:
        return GateResult("profile-schema", False, f"revision ไม่ได้ pin: {revision!r}")
    return GateResult("profile-schema", True)


def gate_secret_scan(bundle_dir: Path) -> GateResult:
    for file_path in _text_files(bundle_dir):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                return GateResult(
                    "secret-scan", False,
                    f"{file_path.name}: พบ pattern secret ({match.group()[:8]}…)",
                )
    return GateResult("secret-scan", True)


def compute_checksums(bundle_dir: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for file_path in sorted(bundle_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(bundle_dir).as_posix()
        if rel == CHECKSUM_FILE or rel.endswith(".zip"):
            continue
        sums[rel] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return sums


def gate_checksums(bundle_dir: Path) -> GateResult:
    path = bundle_dir / CHECKSUM_FILE
    if not path.exists():
        return GateResult("checksums", False, f"ไม่พบ {CHECKSUM_FILE} — รัน lmds validate --fix")
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            recorded[parts[1].strip()] = parts[0].strip()
    actual = compute_checksums(bundle_dir)
    if recorded != actual:
        changed = sorted(set(recorded) ^ set(actual)) or sorted(
            k for k in actual if recorded.get(k) != actual[k]
        )
        return GateResult("checksums", False, f"ไม่ตรง: {', '.join(changed[:3])}")
    return GateResult("checksums", True, f"{len(actual)} ไฟล์")



# Jinja ที่หลุดออกมาเป็น bash ที่ syntax ถูกต้อง — `bash -n` ผ่าน แล้วไปตายตอนรันจริง
# เคสจริง: {% if shard_files %} ถูกวางไว้ใน {% raw %} จึงไม่เคยถูกแปลง และหลุดไปกับ bundle
# expression/comment/statement tag หลุดได้เหมือนกัน: rendered controller ไม่ควรมี Jinja
# opener เหลือเลย · ยกเว้นเฉพาะ Docker Go-template รูป field selector ที่ controller ใช้จริง
# (`{{.Id}}`, `{{ .Names }}`); Go-template แบบ function ที่เพิ่มในอนาคตต้องเพิ่ม allow case พร้อม test
_TEMPLATE_LEFTOVER = re.compile(
    r"\{%"
    # `${#array[@]}` เป็น Bash length expansion ไม่ใช่ Jinja comment
    r"|(?<!\$)\{#"
    r"|\{\{(?!-?\s*\.)"
)


def gate_template_rendered(bundle_dir: Path) -> GateResult:
    for script in _controllers(bundle_dir):
        text = script.read_text(encoding="utf-8")
        match = _TEMPLATE_LEFTOVER.search(text)
        if match:
            line = text[: match.start()].count("\n") + 1
            return GateResult(
                "template-rendered", False,
                f"{script.name}:{line}: มี Jinja tag เหลืออยู่ในไฟล์ผลลัพธ์ ({match.group(0).strip()})",
            )
    return GateResult("template-rendered", True)


ALL_GATES = [
    gate_bash_syntax,
    gate_template_rendered,
    gate_numeric_underscore,
    gate_pipefail_safe,
    gate_line_continuation,
    gate_contract,
    gate_stacked_contract,
    gate_multimodal_assets,
    gate_profile_schema,
    gate_secret_scan,
    gate_checksums,
]


def run_gates(bundle_dir: Path, include_checksums: bool = True) -> list[GateResult]:
    gates = ALL_GATES if include_checksums else ALL_GATES[:-1]
    return [gate(bundle_dir) for gate in gates]


def all_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)
