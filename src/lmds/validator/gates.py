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
_TEMPLATE_LEFTOVER = re.compile(r"(?m)^\s*\{%|\{%\s*(if|for|endif|endfor|raw|endraw)\b")
# `{{ ชื่อ … }}` / `{{ 'ข้อความ' }}` — หัวของ expression เท่านั้น ตัดสินว่าเป็นของ Jinja ไหมที่ _jinja_names()
_VARIABLE_TAG = re.compile(r"""\{\{-?\s*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<quote>['"]))""")
# ใช้เมื่ออ่าน template ของแพ็กเกจไม่ได้ (ไม่ควรเกิด) — ชื่อที่ controller template ใช้จริง ณ 2026-10-06
_JINJA_NAMES_FALLBACK = frozenset({
    "slug", "plan", "report", "fit", "lmds_version", "origin_label", "controller_version", "template_hash",
    "model_label", "runtime_label", "model_features", "shard", "part", "asset", "flag", "pair", "loop",
})
_jinja_names_cache: frozenset[str] | None = None


def _jinja_names() -> frozenset[str]:
    """ชื่อตัวแปรที่ template ของ controller อ้างถึงจริง (ตัวแปรของ renderer + ตัวแปรลูป) — อ่านจาก template เองด้วย Jinja

    ทำไมไม่จับ `{{` ทุกตัว: controller มี `{{` ที่ตั้งใจใส่ — Go template ของ docker (`--format '{{.Names}}'`,
    `{{.State.Running}}`, `{{.Repository}}:{{.Tag}}`) · และ `}}` ก็เป็นตัวปิดปกติของ `${a:-${b}}` กับ JSON
    สิ่งที่แยก Jinja ที่หลุดออกจากของพวกนั้นได้แน่นอนคือ "หัวของ expression เป็นชื่อที่ renderer รู้จัก" (`{{ slug }}`)
    Go template ขึ้นต้นด้วย `.` หรือคำสั่งของมันเอง (`index`, `json`, `range`) ซึ่งไม่ใช่ตัวแปรของเรา
    """
    global _jinja_names_cache
    if _jinja_names_cache is not None:
        return _jinja_names_cache
    names: set[str] = {"loop"}
    try:
        from jinja2 import Environment, meta, nodes

        from lmds.generator.renderer import TEMPLATES_DIR

        env = Environment()
        for path in sorted(Path(TEMPLATES_DIR).glob("*.sh.j2")):
            tree = env.parse(path.read_text(encoding="utf-8"))
            names |= meta.find_undeclared_variables(tree)
            for loop in tree.find_all(nodes.For):
                targets = [loop.target] if isinstance(loop.target, nodes.Name) else loop.target.find_all(nodes.Name)
                names |= {target.name for target in targets}
    except Exception:  # noqa: BLE001 — gate ต้องตัดสินได้เสมอ แม้อ่าน template ไม่ได้
        names |= _JINJA_NAMES_FALLBACK
    _jinja_names_cache = frozenset(names)
    return _jinja_names_cache


def gate_template_rendered(bundle_dir: Path) -> GateResult:
    for script in _controllers(bundle_dir):
        text = script.read_text(encoding="utf-8")
        match = _TEMPLATE_LEFTOVER.search(text)
        if match is None:
            # `{{ slug }}` ที่อยู่ใน {% raw %} หลุดออกมาเป็นตัวหนังสือ — เคสจริง (audit 2026-10-06): controller stacked พิมพ์
            # "lmds set {{ slug }} --port <PORT>" ให้ผู้ใช้ตอน port ชน และด่านนี้ผ่าน เพราะเดิมมองหาแต่ `{%`
            names = _jinja_names()
            match = next(
                (m for m in _VARIABLE_TAG.finditer(text) if m.group("quote") or m.group("name") in names), None)
        if match:
            line = text[: match.start()].count("\n") + 1
            shown = text[match.start(): text.find("\n", match.start())][:40].strip()
            return GateResult(
                "template-rendered", False,
                f"{script.name}:{line}: มี Jinja tag เหลืออยู่ในไฟล์ผลลัพธ์ ({shown})",
            )
    return GateResult("template-rendered", True)


_CANARY_TOKEN = re.compile("LMDS(?:CANARY|DFLTCNRY|EITHCNRY|WORDSCNRY|NUMBCNRY)")
_CANARY_KIND = {
    "LMDSCANARY": "dq", "LMDSDFLTCNRY": "default", "LMDSEITHCNRY": "either",
    "LMDSWORDSCNRY": "words", "LMDSNUMBCNRY": "number",
}
_ADJACENT_WORDS = re.compile(r"LMDSWORDSCNRY(?: +LMDSWORDSCNRY)+")


@dataclass
class _ValueFrame:
    """บรรทัดหนึ่งของ canary render ที่มีค่าแทรก — ข้อความของ template รอบ ๆ ค่า และชนิดของ encoding ที่แต่ละค่าต้องเป็น"""

    pattern: re.Pattern
    kinds: list[str]
    literal: str            # ข้อความของ template ทั้งบรรทัด (ไม่รวมค่า) — ใช้ตัดสินว่าบรรทัดนี้ "ชี้ตัวได้" แค่ไหน

    @property
    def distinctive(self) -> bool:
        # `  "…"` (สมาชิกของ array) ตรงกับบรรทัดไหนก็ได้ · `SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-…}"` ตรงได้บรรทัดเดียว
        return sum(1 for c in self.literal if c.isalnum()) >= 4


def _value_frame(canary_line: str) -> _ValueFrame | None:
    """แยกบรรทัดของ canary render เป็น "ข้อความของ template" กับ "ช่องของค่า" — None = บรรทัดนี้ไม่มีค่าแทรก"""
    if "LMDS" not in canary_line:
        return None
    # flag หลายตัวที่คั่นด้วยช่องว่าง (`ARGS=(W W W )`) คือรายการคำก้อนเดียว — แยกทีละช่องไม่ได้ เพราะคำที่ quote มีช่องว่างข้างในได้
    line = _ADJACENT_WORDS.sub("LMDSWORDSCNRY", canary_line)
    tokens = [m.group(0) for m in _CANARY_TOKEN.finditer(line)]
    if not tokens:
        return None
    literals = _CANARY_TOKEN.split(line)
    kinds: list[str] = []
    merged: list[str] = [literals[0]]
    for token, after in zip(tokens, literals[1:], strict=True):
        kind = _CANARY_KIND[token]
        if kinds and merged[-1] == "" and len(merged) > 1:
            # สองค่าติดกันโดยไม่มีข้อความคั่น (`{{ a }}{{ b }}`) แยกเขตกันไม่ได้ — ตรวจรวมเป็นช่องเดียวด้วยกติกาของตัวที่ไม่ใช่ตัวเลข
            if kinds[-1] == "number" or (kinds[-1] == "dq" and kind == "default"):
                kinds[-1] = kind
            merged[-1] = after
            continue
        kinds.append(kind)
        merged.append(after)
    pattern = re.compile("(.*?)".join(re.escape(part) for part in merged), re.S)
    return _ValueFrame(pattern, kinds, "".join(merged))


def _pair_lines(canary_lines: list[str], actual_lines: list[str], wanted: list[int]) -> dict[int, int]:
    """บรรทัดของ canary → บรรทัดของ controller จริงที่เป็นคู่กัน (เฉพาะบรรทัดใน `wanted`)

    render จาก template ชุดเดียวกัน = จำนวนบรรทัดเท่ากัน จับคู่ตามตำแหน่งได้เลย · ไม่เท่า (แก้มือ · template รุ่นอื่น · ค่าที่พา
    ขึ้นบรรทัดใหม่มา · ค่าที่ทำให้ตารางไฟล์ที่สร้างกลับมามีสมาชิกเกิน) ใช้ difflib หาช่วงที่ข้อความของ template ตรงกัน แล้วจับคู่
    ช่วงที่ต่างกันตามลำดับจากต้นช่วง — บรรทัดที่เกินมาฝั่งใดฝั่งหนึ่งไม่มีคู่ และคู่ที่โครงไม่ตรงผู้เรียกจะเห็นเอง
    (ทั้งสองแบบผู้เรียกต้องรายงาน ไม่ใช่ข้าม)
    """
    if len(canary_lines) == len(actual_lines):
        return {i: i for i in wanted}
    import difflib

    pairs: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, canary_lines, actual_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            for offset in range(min(i2 - i1, j2 - j1)):
                pairs[i1 + offset] = j1 + offset
    return {i: pairs[i] for i in wanted if i in pairs}


def gate_value_expansion(bundle_dir: Path) -> GateResult:
    """ค่าที่ renderer แทรกลง controller ต้องเป็น "ตัวหนังสือ" เสมอ — ไม่มี `$(` `${` backtick หรือ quote ที่ค่าพามาเอง

    วิธี: สร้างแผนกลับจาก bundle (MODEL_PROFILE.yaml + หัว controller — ทางเดียวกับ `lmds bundles refresh`) แล้ว render
    template เดิมอีกรอบโดยแทน *ทุกค่า* ด้วยคำ canary · บรรทัดที่มี canary คือบรรทัดที่มีค่าแทรก และส่วนที่เหลือของบรรทัดนั้น
    คือข้อความของ template → ตัดข้อความของ template ออกจากบรรทัดคู่กันใน controller จริง ที่เหลือคือ "ค่าตามที่ถูกเขียนลงไฟล์"
    ซึ่งต้องเป็นผลของ encoder ชนิดที่ renderer เลือกให้ตำแหน่งนั้นพอดี (`shellsafe.encoding_problem`)

    **ไม่มีการอ่าน quote ของทั้งไฟล์** — รุ่นแรกของด่านนี้ไล่นับ quote ตั้งแต่บรรทัดแรก พอ template มี
    `"$(… | grep -o '"id":"[^"]*"' …)"` (quote ใน `$( )` เริ่มนับใหม่ และ `'…'` ข้างในมี `"` จำนวนคี่) สถานะก็เพี้ยน
    ไปทั้งไฟล์: ค่าที่อยู่ในช่วงที่มัน "คิดว่าอยู่ใน single quote" ไม่ถูกตรวจเลยและด่านรายงานว่าผ่าน (2026-10-06 หลัง merge)
    บริบทของแต่ละตำแหน่งเป็นเรื่องของ template ไม่ใช่ของ bundle — พิสูจน์ด้วยการถาม bash เองในเทส
    (tests/test_shell_injection.py) ส่วนด่านนี้ตรวจสิ่งเดียวที่ขึ้นกับ bundle: ค่าถูก encode มาครบไหม

    ตรวจไม่ได้ต้องพูด: บรรทัดมีค่าที่หาคู่ไม่ได้/โครงไม่ตรง ใน bundle ที่อ้าง template_hash ชุดนี้ = ไม่ผ่าน
    (ค่าพาขึ้นบรรทัดใหม่มา หรือไฟล์ถูกแก้) · bundle จาก template รุ่นอื่น = n/a พร้อมจำนวนบรรทัดที่ตรวจได้

    เคสจริง (audit 2026-10-06): ไฟล์ใน repo ชื่อ `…$(touch PWNED_gguf).gguf` และ served_model_name
    `qwen$(touch PWNED_served_name)` ผ่านครบทุกด่านที่มีตอนนั้น แล้ว `controller help` สร้างไฟล์ PWNED บนเครื่อง
    """
    name = "value-expansion"
    profile_path = bundle_dir / "MODEL_PROFILE.yaml"
    scripts = [s for s in _controllers(bundle_dir) if s.name.endswith(("-single.sh", "-stacked.sh"))]
    if not profile_path.exists() or not scripts:
        return GateResult(name, True, "n/a (ไม่มี profile หรือไม่ใช่ controller ที่ LMDS render)")
    try:
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return GateResult(name, True, "n/a (profile อ่านไม่ได้ — ปล่อยให้ gate อื่นจับ)")
    if not isinstance(profile, dict):
        return GateResult(name, True, "n/a")

    from lmds.fleet.consistency import controller_header
    from lmds.fleet.refresh import RefreshError, plan_from_profile
    from lmds.generator.renderer import render_canary_controller, template_hash
    from lmds.shellsafe import encoding_problem

    checked = 0
    skipped: list[str] = []
    for script in scripts:
        actual = script.read_text(encoding="utf-8")
        try:
            try:
                plan, report, fit = plan_from_profile(profile, actual)
            except RefreshError:
                # repo ที่ Hub ไม่รายงาน shard: template ไม่พิมพ์ตาราง SHARD_FILES เลย และ refresh ถือว่า "ต้องออนไลน์" ·
                # สำหรับการเทียบโครง "ไม่มีตาราง" = "ตารางว่าง" พอดี (template ข้ามบล็อกนั้นเหมือนกัน) จึงยังเทียบได้
                plan, report, fit = plan_from_profile(profile, actual + "\nSHARD_FILES=(\n)\n")
            canary = render_canary_controller(plan, report, fit, slug=bundle_dir.name)
        except Exception as exc:  # noqa: BLE001 — bundle ที่สร้างแผนกลับไม่ได้ (adopt · profile รุ่นเก่า) เทียบไม่ได้ ไม่ใช่ไม่ผ่าน
            return GateResult(name, True, f"n/a (สร้างแผนกลับจาก bundle ไม่ได้: {str(exc)[:80]})")

        # bundle นี้อ้างว่า render จาก template + renderer ชุดที่แพ็กเกจถืออยู่ไหม — ถ้าใช่ ทุกบรรทัดมีค่าต้องตรวจได้
        claimed = str(profile.get("template_hash") or controller_header(script)["template_hash"] or "")
        current = claimed == template_hash()
        canary_lines, actual_lines = canary.split("\n"), actual.split("\n")
        frames = {i: frame for i, line in enumerate(canary_lines) if (frame := _value_frame(line)) is not None}
        pairs = _pair_lines(canary_lines, actual_lines, sorted(frames))
        unverified: list[int] = [i for i in frames if i not in pairs]
        for i, j in sorted(pairs.items()):
            frame = frames[i]
            found = frame.pattern.fullmatch(actual_lines[j])
            if found is None:
                unverified.append(i)
                continue
            if not current and not frame.distinctive:
                # template รุ่นอื่น: บรรทัดรูป `  "…"` จับคู่ตามตำแหน่งแล้วอาจเป็นคนละบรรทัดกัน — ไม่ตัดสินจากคู่ที่ไม่แน่ใจ
                unverified.append(i)
                continue
            for kind, text in zip(frame.kinds, found.groups(), strict=True):
                problem = encoding_problem(kind, text)
                if problem:
                    return GateResult(
                        name, False,
                        f"{script.name}:{j + 1}: {problem} — ค่าจาก repo/แผนจะถูกเชลล์ตีความบนเครื่องที่รัน")
            checked += 1
        if unverified and current:
            first = min(unverified)
            where = pairs.get(first)
            return GateResult(
                name, False,
                f"{script.name}:{(where if where is not None else first) + 1}: บรรทัดที่มีค่าแทรกไม่ตรงโครงของ template ที่ bundle "
                f"อ้างว่า render มา ({len(unverified)} บรรทัด) — ตรวจไม่ได้ว่าค่าเป็นตัวหนังสือ: ค่าพาขึ้นบรรทัดใหม่มา "
                "หรือไฟล์ถูกแก้มือ · regenerate: lmds bundles refresh")
        if unverified:
            skipped.append(f"{script.name}: ตรวจไม่ได้ {len(unverified)} จาก {len(frames)} บรรทัดที่มีค่าแทรก")
    if skipped:
        return GateResult(
            name, True,
            "n/a (controller จาก template รุ่นอื่น — " + "; ".join(skipped) + f" · ตรวจได้ {checked} บรรทัด ไม่พบค่าที่เป็นโค้ด · "
            "regenerate แล้วตรวจใหม่: lmds bundles refresh)")
    return GateResult(name, True, f"เทียบกับ canary แล้ว {checked} บรรทัดที่มีค่าแทรก")


def gate_serving_consistent(bundle_dir: Path) -> GateResult:
    """context / slots / max_output_tokens ต้องเป็นค่าที่เป็นไปได้จริงพร้อมกัน

    llama.cpp แบ่ง `--ctx-size` เท่า ๆ กันให้ทุก slot — output ที่ใหญ่กว่า context ต่อ slot
    คือค่าที่เป็นไปไม่ได้ · เจอจริง: context 16,384 · slots 4 · output 8,192 ผ่าน gate
    ทุกด่านแล้วไปตายที่ `client-config` ของ bundle ตัวเอง ("context ต่อ slot เล็กเกิน")
    """
    name = "serving-consistent"
    scripts = _controllers(bundle_dir)
    if not scripts:
        return GateResult(name, True, "ไม่มี controller ให้ตรวจ")
    text = scripts[0].read_text(encoding="utf-8")
    if "PARALLEL_SEQS" not in text:
        return GateResult(name, True, "ไม่ใช่ llama.cpp")

    def value(var: str) -> int | None:
        match = re.search(rf'^{var}="\$\{{{var}:-(\d+)\}}"', text, re.M)
        return int(match.group(1)) if match else None

    context, slots, output = value("CTX_SIZE"), value("PARALLEL_SEQS"), value("CLIENT_OUTPUT")
    overhead = 2048
    if None in (context, slots, output) or slots <= 0:
        return GateResult(name, True, "อ่านค่าไม่ครบ — ข้าม")
    # เงื่อนไขเดียวกับที่ controller ใช้เป๊ะ ๆ: input_budget = ต่อslot - output - overhead > 0
    per_slot = context // slots
    input_budget = per_slot - output - overhead
    if input_budget <= 0:
        return GateResult(
            name, False,
            f"context ต่อ slot {per_slot:,} ({context:,}/{slots}) − output {output:,} − "
            f"overhead {overhead:,} = {input_budget:,} — bundle นี้รัน client-config "
            f"ของตัวเองไม่ผ่าน",
        )
    return GateResult(name, True, f"input {input_budget:,} ต่อ slot")


def gate_origin_stamp(bundle_dir: Path) -> GateResult:
    """ตราประทับต้องเป็นของจริง ถ้ามีการอ้างชื่อเจ้าของ

    **ไม่ใช่ด่านไลเซนส์** — bundle ที่ไม่มีตรา และ bundle จากเครื่องโหมดฟรี ผ่านทั้งคู่
    ด่านนี้จับอย่างเดียวคือ *ตราที่ถูกแก้* เช่นพาร์ตเนอร์เปลี่ยน licensed_to เป็นชื่อตัวเอง
    แล้วเอาไปขายต่อ · ลายเซ็นในตราเซ็นด้วยกุญแจของเราบนเครื่องของเรา แก้ชื่อแล้วไม่ตรงทันที
    """
    name = "origin-stamp"
    path = bundle_dir / "MODEL_PROFILE.yaml"
    if not path.exists():
        return GateResult(name, True, "n/a (ไม่มี profile)")
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return GateResult(name, True, "n/a (profile อ่านไม่ได้ — ปล่อยให้ gate อื่นจับ)")

    from lmds.licensing.stamp import verify as verify_stamp

    ok, detail = verify_stamp(profile.get("origin"))
    return GateResult(name, ok, detail)


ALL_GATES = [
    gate_serving_consistent,
    gate_bash_syntax,
    gate_template_rendered,
    gate_value_expansion,
    gate_numeric_underscore,
    gate_pipefail_safe,
    gate_line_continuation,
    gate_contract,
    gate_stacked_contract,
    gate_multimodal_assets,
    gate_profile_schema,
    gate_secret_scan,
    gate_origin_stamp,
    # gate_checksums ต้องอยู่ท้ายสุดเสมอ — run_gates() ตัดตัวสุดท้ายออกด้วย ALL_GATES[:-1]
    # ตอน include_checksums=False · แทรกอะไรต่อท้ายนี่คือตัดผิดตัวแบบเงียบ ๆ
    gate_checksums,
]


def run_gates(bundle_dir: Path, include_checksums: bool = True) -> list[GateResult]:
    gates = ALL_GATES if include_checksums else ALL_GATES[:-1]
    return [gate(bundle_dir) for gate in gates]


def all_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)
