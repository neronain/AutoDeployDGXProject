"""ทะเบียนเครื่องมือของ `lmds mcp` — ที่เดียวที่ชื่อเครื่องมือถูกผูกกับฟังก์ชัน และที่เดียวที่ argument จากข้างนอกถูกตรวจ

argument มาจากผู้ช่วย AI ซึ่งส่ง "อะไรก็ตามที่หน้าเว็บ/README/issue ที่มันเพิ่งอ่านบอกให้ส่ง" (prompt injection) —
ถือเป็นข้อมูลที่ไม่ไว้ใจทุกตัว · ลำดับของทุกคำขอ: รูปร่างตาม inputSchema → ตัวตรวจของโปรเจกต์ (ชื่อเครื่องต้องอยู่ใน
ทะเบียน · slug ผ่าน shellsafe.BUNDLE_SLUG · repo ผ่าน resolver.parse_source ตัวเดียวกับ `lmds inspect`) → จึงเรียก
ฟังก์ชันอ่าน · ไม่ผ่านด่านไหน = ปฏิเสธก่อนมีคำสั่งถูกประกอบ · คำตอบทุกก้อนผ่าน redaction.scrub ก่อนออก

`cost` บอกราคาของการเรียกหนึ่งครั้ง อยู่ในคำอธิบายที่ผู้ช่วยอ่านตอนเลือกเครื่องมือ:
  local    อ่านจากเครื่อง hub เท่านั้น
  ssh-one  SSH ไปหนึ่งเครื่อง (เมื่อระบุ node) — ไม่ระบุ = local
  ssh-all  SSH ไปทุกเครื่องในทะเบียนพร้อมกัน (เฉพาะเมื่อเปิด check)
  network  ถาม Hugging Face ผ่านอินเทอร์เน็ต
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from . import reads, redaction, seal

MODEL_MAX_CHARS = 300          # ลิงก์ Hugging Face ที่ยาวสุดที่มีเหตุผล (org/model + /blob/<rev>/<ไฟล์ .gguf>)
REVISION_MAX_CHARS = 128
TARGETS_MAX = 8
RESULT_MAX_CHARS = 600_000     # เกินนี้ไม่ส่ง — คำตอบเข้า context ของผู้ช่วยทั้งก้อน

READ_ONLY_NOTE = "Read-only: never changes anything on the hub or on any node."
_NODE_ARG = {"type": "string", "maxLength": 63,
             "description": "Node name exactly as listed by lmds_nodes. Omit to ask the hub itself."}
_SLUG_ARG = {"type": "string", "maxLength": 64,
             "description": "Bundle slug exactly as listed by lmds_models (letters, digits, . _ -)."}
_MODEL_ARG = {"type": "string", "maxLength": MODEL_MAX_CHARS,
              "description": "Hugging Face repo id (org/model) or a huggingface.co link."}
_REVISION_ARG = {"type": "string", "maxLength": REVISION_MAX_CHARS, "description": "Branch, tag or commit (optional)."}


class Refused(Exception):
    """argument ไม่ผ่านด่าน — ปฏิเสธก่อนเรียกฟังก์ชันอ่าน (ยังไม่มีคำสั่งไหนถูกประกอบ)"""


@dataclass(frozen=True)
class Tool:
    name: str
    func: Callable
    cost: str
    summary: str
    properties: dict = field(default_factory=dict)
    required: tuple[str, ...] = ()
    checks: dict = field(default_factory=dict)      # {ชื่อ argument: ตัวตรวจ(value) -> value ที่ใช้ได้ หรือ raise Refused}

    def __post_init__(self) -> None:
        # ด่านโครงสร้างข้อแรก: ผูกได้เฉพาะฟังก์ชันที่ประกาศใน reads.READ_ALLOWLIST และเป็นตัวที่อยู่ใน reads จริง
        # (ชื่อเดียวกันจากโมดูลอื่นไม่นับ) — ผูกผิด = import ไม่ผ่าน server ไม่ขึ้น ไม่ใช่ไปรู้ตอนมีคนเรียก
        name = getattr(self.func, "__name__", "")
        if name not in reads.READ_ALLOWLIST or getattr(reads, name, None) is not self.func:
            raise RuntimeError(f"เครื่องมือ {self.name} ผูกกับ {self.func!r} ซึ่งไม่อยู่ใน reads.READ_ALLOWLIST — "
                               "`lmds mcp` ผูกได้เฉพาะฟังก์ชันอ่านที่ประกาศไว้")
        if self.cost not in ("local", "ssh-one", "ssh-all", "network"):
            raise RuntimeError(f"เครื่องมือ {self.name}: cost '{self.cost}' ไม่รู้จัก")

    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": list(self.required),
                "additionalProperties": False}

    def definition(self) -> dict:
        cost = {
            "local": "Cost: local — reads the hub only, no SSH, no network.",
            "ssh-one": "Cost: local when `node` is omitted; one SSH connection to that node when given.",
            "ssh-all": "Cost: local by default; with check=true one SSH connection to EVERY registered node in parallel.",
            "network": "Cost: network — metadata requests to Hugging Face (no weights are downloaded); no SSH.",
        }[self.cost]
        return {"name": self.name, "description": f"{self.summary} {READ_ONLY_NOTE} {cost}",
                "inputSchema": self.schema,
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                                "openWorldHint": self.cost == "network"}}


# ── ตัวตรวจ argument ─────────────────────────────────────────────────────────────────────────────
def _known_node(value: str) -> str:
    """ชื่อเครื่องต้องเป็นชื่อที่อยู่ในทะเบียนของ hub เป๊ะ ๆ — ชื่ออื่นไม่มีทางไปถึง ssh"""
    from lmds.nodes import NodeError, load
    from lmds.nodes.registry import name_ok

    if not name_ok(value):
        raise Refused(f"node {value[:80]!r} is not a valid node name")
    try:
        names = [node.name for node in load()]
    except NodeError as exc:
        raise Refused(f"cannot read the node registry: {exc}") from exc
    if value not in names:
        raise Refused(f"unknown node {value!r} — registered nodes: {', '.join(names) or '(none)'}")
    return value


def _bundle_slug(value: str) -> str:
    from lmds.shellsafe import is_bundle_slug, unsafe_chars

    if not is_bundle_slug(value) or unsafe_chars(value, allow_space=False):
        raise Refused(f"slug {value[:80]!r} is not a bundle name (letters, digits, . _ - only, max 64, "
                      "starting with a letter or digit)")
    return value


def _repo(value: str) -> str:
    """ลิงก์/repo id ผ่าน parser ตัวเดียวกับ `lmds inspect` — รูปที่มันไม่รู้จักไม่ถูกส่งต่อไปไหน"""
    from lmds.resolver import SourceError, parse_source
    from lmds.shellsafe import _control_chars

    if _control_chars(value):
        raise Refused("model contains control characters")
    try:
        parse_source(value)
    except SourceError as exc:
        raise Refused(f"model is not a Hugging Face repo or link: {exc}") from exc
    return value.strip()


def _revision(value: str) -> str:
    from lmds.shellsafe import unsafe_chars

    cleaned = value.strip()
    if not cleaned or cleaned.startswith(("-", "/", ".")) or ".." in cleaned \
            or unsafe_chars(cleaned, allow_space=False) or not all(c.isalnum() or c in "._-/" for c in cleaned):
        raise Refused(f"revision {value[:80]!r} is not a branch, tag or commit name")
    return cleaned


def _target(value: str) -> str:
    from lmds.fit import PRESETS

    if value not in PRESETS:
        raise Refused(f"unknown target {value[:80]!r} — choose one of: {', '.join(sorted(PRESETS))}")
    return value


def _targets(values: list) -> tuple[str, ...]:
    return tuple(_target(value) for value in values)


def _engine(value: str) -> str:
    from lmds.brain.plan_schema import Engine

    allowed = [engine.value for engine in Engine]
    if value not in allowed:
        raise Refused(f"unknown engine {value[:80]!r} — choose one of: {', '.join(allowed)}")
    return value


def _kv_dtype(value: str) -> str:
    from lmds.fit.memory import KV_DTYPE_BYTES

    if value not in KV_DTYPE_BYTES:
        raise Refused(f"unknown kv_dtype {value[:80]!r} — choose one of: {', '.join(KV_DTYPE_BYTES)}")
    return value


def _empty_is_absent(check: Callable) -> Callable:
    """`node: ""` = ไม่ได้ระบุ (hub) — ผู้ช่วยบางตัวส่งสตริงว่างแทนการไม่ส่งคีย์"""
    return lambda value: check(value) if value else ""


_INT = lambda low, high, what: {"type": "integer", "minimum": low, "maximum": high, "description": what}  # noqa: E731


# ── ทะเบียน ──────────────────────────────────────────────────────────────────────────────────────
TOOLS: tuple[Tool, ...] = (
    Tool("lmds_version", reads.hub_version, "local",
         "Identity of this LMDS hub: version, the git commit it runs, the commit installed on disk, the controller "
         "template standard and hash, and uncommitted files in the hub checkout. Call this first."),
    Tool("lmds_nodes", reads.nodes, "local",
         "Every machine in the hub's node registry with what the hub remembers from its last probe: full untruncated "
         "name, site, SSH host/user/port and alternate hosts, the IP the node reported, LMDS version and commit, "
         "last_seen, last_error, stale/unknown counters for controllers and runtimes, restart_pending, cluster "
         "settings. Remembered values, not a live check — use lmds_fleet_check with check=true or lmds_models with "
         "a node for the current state."),
    Tool("lmds_models", reads.models, "ssh-one",
         "Machine facts and every model bundle on the hub or on one node — the `lmds agent info` payload: host "
         "(GPUs, memory, role, runtimes) plus per model slug, model_id, engine, mode, port, running/healthy, "
         "endpoint, context and slots in effect, autostart, adopted/external, pending_restart, controller and "
         "runtime state, 24h request usage, and a summary.",
         {"node": _NODE_ARG}, checks={"node": _empty_is_absent(_known_node)}),
    Tool("lmds_inspect", reads.inspect_repo, "network",
         "Inspect a Hugging Face model repo without downloading weights — the `lmds inspect --json` payload: "
         "artifact type, size, architecture, native context, quantization, capabilities, gated flag, the "
         "unsupported-format verdict (e.g. MLX checkpoints no engine here can load, in model.unsupported_format), "
         "fit per target and context/concurrency advice.",
         {"model": _MODEL_ARG, "revision": _REVISION_ARG,
          "targets": {"type": "array", "items": {"type": "string", "maxLength": 64}, "maxItems": TARGETS_MAX,
                      "description": "Target presets to evaluate fit against (e.g. dgx-spark-single). "
                                     "Empty = this hub's own hardware plus dgx-spark-single."},
          "concurrency": _INT(1, 1024, "Concurrent requests used for the KV-cache estimate (default 1)."),
          "context": _INT(1, 10_000_000, "Ask whether this context length is advisable (optional)."),
          "kv_dtype": {"type": "string", "maxLength": 16, "description": "KV cache dtype: bf16 (default), fp16 or fp8."}},
         required=("model",),
         checks={"model": _repo, "revision": _revision, "targets": _targets, "kv_dtype": _kv_dtype}),
    Tool("lmds_plan", reads.plan, "network",
         "Rule-based deployment plan for a Hugging Face model on a target — the `lmds plan --no-llm --json` payload "
         "(engine, image, context, slots, flags, assets). Never calls an LLM and writes no bundle. A model that does "
         "not fit, or whose format no engine can load, returns an error with the reason and alternatives.",
         {"model": _MODEL_ARG, "revision": _REVISION_ARG,
          "target": {"type": "string", "maxLength": 64,
                     "description": "Target preset (e.g. dgx-spark-single, dgx-spark-stacked). "
                                    "Omit = this hub's own hardware, else dgx-spark-single."},
          "concurrency": _INT(1, 1024, "Concurrent requests the plan should serve (default 1)."),
          "engine": {"type": "string", "maxLength": 16, "description": "Force a runtime: vllm, sglang or llamacpp (optional)."}},
         required=("model",),
         checks={"model": _repo, "revision": _revision, "target": _target, "engine": _engine}),
    Tool("lmds_fit", reads.fit, "ssh-one",
         "Memory fit of an existing bundle on its machine for a given slots/context — the `lmds fit <slug> --json` "
         "payload: weights, overhead, KV per request, what other running models hold, RAM left, verdict, and the "
         "settings that WOULD be written. Always a dry run: it can never apply them. May ask Hugging Face once for "
         "an old bundle that lacks KV dimensions.",
         {"slug": _SLUG_ARG, "node": _NODE_ARG,
          "slots": _INT(1, 1024, "Concurrent request slots to evaluate (default: the bundle's current setting)."),
          "context": _INT(1, 10_000_000, "Context per request to evaluate (default: current setting / native).")},
         required=("slug",), checks={"slug": _bundle_slug, "node": _empty_is_absent(_known_node)}),
    Tool("lmds_fleet_check", reads.fleet_check, "ssh-all",
         "Whether every node matches the hub on three axes (code, controllers, runtime) — the `lmds fleet check "
         "--json` payload with one entry per node and a summary. Default uses what the registry remembers (entries "
         "older than 5 minutes are marked unverified). check=true probes every node now; unlike the CLI's --check "
         "it does not write the results back to the registry. An unreachable node appears with reachable=false and "
         "its error; the other nodes are still reported.",
         {"check": {"type": "boolean", "description": "Probe every node over SSH now (slower). Default false."}}),
    Tool("lmds_watchdog_status", reads.watchdog_status, "ssh-one",
         "Generate-probe watchdogs on the hub or on one node — the `lmds watchdog status --json` payload: per armed "
         "slug the policy, last probe and last success times, restarts, paused/blocked/gave_up, and what systemd "
         "says about the watchdog service. Empty list = none armed.",
         {"slug": {**_SLUG_ARG, "description": "Only this bundle (optional; default: every armed watchdog)."},
          "node": _NODE_ARG},
         checks={"slug": _empty_is_absent(_bundle_slug), "node": _empty_is_absent(_known_node)}),
    Tool("lmds_logs", reads.logs, "ssh-one",
         f"Tail of a model's server log on the hub or on one node (default {reads.LOG_LINES_DEFAULT} lines, hard cap "
         f"{reads.LOG_LINES_MAX}; long output keeps the end). Secrets such as API keys are redacted.",
         {"slug": _SLUG_ARG, "node": _NODE_ARG,
          "lines": _INT(1, reads.LOG_LINES_MAX, f"Lines from the end (default {reads.LOG_LINES_DEFAULT}).")},
         required=("slug",), checks={"slug": _bundle_slug, "node": _empty_is_absent(_known_node)}),
    Tool("lmds_doctor", reads.doctor, "ssh-one",
         "Why a bundle does not download or start — the `lmds doctor <slug> --json --no-probe` payload: findings "
         "(name, ok/warn/fail, detail, fix command) for role, controller, HF token, weights, permissions, disk, "
         "docker, image, port, open endpoint and server health. The one check that would run a throw-away container "
         "is skipped and listed under `skipped`.",
         {"slug": _SLUG_ARG, "node": _NODE_ARG},
         required=("slug",), checks={"slug": _bundle_slug, "node": _empty_is_absent(_known_node)}),
)

BY_NAME = {tool.name: tool for tool in TOOLS}
if len(BY_NAME) != len(TOOLS):
    raise RuntimeError("ชื่อเครื่องมือซ้ำกันใน lmds.mcp.tools.TOOLS")


def definitions() -> list[dict]:
    return [tool.definition() for tool in TOOLS]


# ── ตรวจรูปร่างตาม inputSchema (เฉพาะส่วนของ JSON Schema ที่ทะเบียนนี้ใช้) ──────────────────────────────
def _shape(name: str, spec: dict, value):
    kind = spec["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise Refused(f"{name} must be a string")
        if len(value) > spec.get("maxLength", 1024):
            raise Refused(f"{name} is too long ({len(value)} characters, limit {spec.get('maxLength', 1024)})")
    elif kind == "integer":
        # bool เป็น int ใน Python — `slots: true` ต้องไม่กลายเป็น 1
        if isinstance(value, bool) or not isinstance(value, int):
            raise Refused(f"{name} must be an integer")
        if value < spec.get("minimum", value) or value > spec.get("maximum", value):
            raise Refused(f"{name} must be between {spec.get('minimum')} and {spec.get('maximum')} (got {value})")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise Refused(f"{name} must be true or false")
    elif kind == "array":
        if not isinstance(value, list):
            raise Refused(f"{name} must be a list")
        if len(value) > spec.get("maxItems", 64):
            raise Refused(f"{name} has too many items ({len(value)}, limit {spec.get('maxItems', 64)})")
        for index, item in enumerate(value):
            _shape(f"{name}[{index}]", spec["items"], item)
    return value


def validate(tool: Tool, arguments) -> dict:
    """argument ที่ผ่านทุกด่านแล้ว พร้อมส่งให้ฟังก์ชันอ่าน — ไม่ผ่าน = Refused (ยังไม่มีอะไรถูกเรียก)"""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise Refused("arguments must be an object")
    unknown = sorted(set(arguments) - set(tool.properties))
    if unknown:
        raise Refused(f"unknown argument(s) for {tool.name}: {', '.join(str(key)[:40] for key in unknown)} — "
                      f"accepted: {', '.join(tool.properties) or '(none)'}")
    missing = [key for key in tool.required if arguments.get(key) in (None, "")]
    if missing:
        raise Refused(f"missing required argument(s) for {tool.name}: {', '.join(missing)}")
    clean: dict = {}
    for key, value in arguments.items():
        if value is None:
            continue                            # null = ไม่ได้ระบุ
        value = _shape(key, tool.properties[key], value)
        check = tool.checks.get(key)
        clean[key] = check(value) if check is not None else value
    return {key: value for key, value in clean.items() if value != ""}


# ── เรียก ────────────────────────────────────────────────────────────────────────────────────────
def _text(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def as_result(payload, *, error: bool, notes: list[str] | None = None) -> dict:
    known = redaction.known_secrets()
    text = _text(redaction.scrub(payload, known))
    if len(text) > RESULT_MAX_CHARS:
        text = _text({"error": f"result is too large to return ({len(text):,} characters, limit {RESULT_MAX_CHARS:,}) — "
                               "narrow the request (one node, one slug, fewer log lines)"})
        error = True
    content = [{"type": "text", "text": text}]
    if notes:
        # ก้อนแรกคือ payload ล้วน (เท่ากับ `--json`) · ก้อนที่สองคือสิ่งที่ CLI จะพิมพ์บน stderr และสิ่งที่ถูกกั้นไว้
        content.append({"type": "text", "text": _text({"notes": redaction.scrub(notes, known)})})
    return {"content": content, "isError": error}


def call(name: str, arguments) -> dict:
    """เรียกเครื่องมือหนึ่งครั้ง → ผลในรูป MCP (`content` + `isError`) · ชื่อที่ไม่รู้จัก = KeyError ให้ชั้น protocol ตอบ"""
    tool = BY_NAME[name]
    try:
        clean = validate(tool, arguments)
    except Refused as exc:
        return as_result({"error": str(exc), "refused": True}, error=True)
    seal.drain()
    try:
        payload = tool.func(**clean)
    except reads.ToolFailure as exc:
        return as_result({"error": str(exc), **exc.details}, error=True, notes=_blocked())
    except seal.ReadOnlyViolation as exc:
        return as_result({"error": str(exc), "read_only": True}, error=True)
    except _hub_file_errors() as exc:
        # ไฟล์ของ hub เองอ่านไม่ได้ (config.yaml / nodes.yaml ที่แก้มือแล้วพัง) — ข้อความของมันบอกไฟล์และวิธีแก้อยู่แล้ว
        # ส่งต่อให้ผู้ถามตรง ๆ ไม่ใช่ "internal error" พร้อม traceback บน stderr ที่ผู้ช่วยมองไม่เห็น
        return as_result({"error": str(exc)}, error=True, notes=_blocked())
    notes = list(getattr(payload, "notes", []) or []) + _blocked()
    return as_result(dict(payload) if isinstance(payload, dict) else payload, error=False, notes=notes)


def _hub_file_errors() -> tuple:
    from lmds.config import SettingsError
    from lmds.fleet import FleetError
    from lmds.nodes import NodeError

    return (SettingsError, NodeError, FleetError)


def _blocked() -> list[str]:
    return [f"read-only guard refused: {what}" for what in seal.drain()]
