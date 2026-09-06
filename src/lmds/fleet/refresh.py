"""regenerate controller ของ bundle เก่าแบบ *ออฟไลน์* — จากแผนที่เก็บใน MODEL_PROFILE.yaml ไม่ต้องถึง Hugging Face

ทำไมต้องมี: `lmds node install` อัปเดตแค่ตัวโปรแกรม · bundle บน node 46/47 ใบยังเป็น controller ของ 0.3.0–0.6.0
(audit 2026-09-06) ซึ่งไม่มี check_architecture/explain_crash/update-runtime — ทางเดียวที่เคยมีคือ `lmds rebuild`
ที่ต้อง inspect ซ้ำจาก HF (เน็ต + token + เป็นนาทีต่อใบ) และมีแค่ CLI · Update จึงไม่เคยแตะ bundle

กติกา:
  - ใช้ค่าที่ profile จดไว้ตามที่อนุมัติแล้ว (serving/features/target/runtime) — ไม่เรียก LLM ไม่ harden ซ้ำ ไม่ถามเน็ต
  - ตารางไฟล์ (ขนาด/sha ของ GGUF · shard ของ safetensors) อ่านจากหัว controller เดิม — profile ไม่ได้เก็บ
  - controller เดิมเก็บเป็น `<name>.replaced-<stamp>` ทุกครั้ง (renderer เก็บให้เฉพาะตอน topology เปลี่ยน)
  - `bundle.env` / `bundle.args` / `cluster.env` ไม่แตะ — เป็นค่าต่อเครื่องโดยตั้งใจ
  - `generated_by: lmds adopt` / weight ที่ผู้ใช้ดูแลเอง = ข้าม (ไม่มี template ให้ regenerate)
  - profile ไม่พอ render (รุ่นเก่าขาดคีย์) = รายงาน "ต้อง lmds rebuild ออนไลน์" ไม่ล้มทั้งงาน
  - โมเดลที่กำลังรัน regenerate ไฟล์ได้ (ปลอดภัย — docs/FLEET-MULTI-NODE.md) แต่ต้องบอกว่าใช้ตัวใหม่เมื่อ restart
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


class RefreshError(Exception):
    """render จาก profile ไม่ได้ — ต้อง `lmds rebuild` (ออนไลน์) แทน"""


@dataclass
class RefreshResult:
    slug: str
    action: str            # refreshed | current | adopted | needs-online | error | missing
    detail: str = ""
    replaced: str = ""     # path ของ controller เดิมที่เก็บไว้
    running: bool = False
    before: str = ""       # generated_by เดิม
    after: str = ""        # generated_by ใหม่
    gates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.action in ("refreshed", "current", "adopted")

    def payload(self) -> dict:
        return {"slug": self.slug, "action": self.action, "detail": self.detail, "replaced": self.replaced,
                "running": self.running, "before": self.before, "after": self.after, "gates": list(self.gates)}


def format_result(r: RefreshResult) -> str:
    """บรรทัดรายงานต่อ bundle — ใช้ทั้ง CLI, สคริปต์ install บน node และ job บนหน้าเว็บ"""
    label = {
        "refreshed": "regenerate แล้ว",
        "current": "ตรง template อยู่แล้ว",
        "adopted": "adopted — ไม่มี template ให้ regenerate",
        "needs-online": "controller-stale (ต้อง lmds rebuild ออนไลน์)",
        "error": "regenerate ไม่สำเร็จ",
        "missing": "ไม่พบ bundle",
    }.get(r.action, r.action)
    parts = [f"{r.slug}: {label}"]
    if r.before and r.after and r.before != r.after:
        parts.append(f"{r.before} → {r.after}")
    if r.detail:
        parts.append(r.detail)
    if r.running and r.action == "refreshed":
        parts.append("ตัวที่รันอยู่ใช้ controller เดิมจนกว่าจะ restart")
    return " · ".join(parts)


# ────────────────────────── อ่านตารางไฟล์จากหัว controller เดิม ──────────────────────────
def _bash_array(text: str, name: str) -> list[str] | None:
    """ค่าใน `NAME=(\n  "a"\n  "b"\n)` ของ controller — None เมื่อไม่มี array นั้น"""
    found = re.search(rf"^{re.escape(name)}=\((.*?)^\)", text, re.M | re.S)
    if not found:
        return None
    return re.findall(r'"([^"\n]*)"', found.group(1))


def _default_of(text: str, name: str) -> str:
    """ค่าตั้งต้นของ `NAME="${NAME-value}"` / `${NAME:-value}` ในหัว controller"""
    found = re.search(rf'^{re.escape(name)}="\$\{{{re.escape(name)}:?-([^}}]*)\}}"', text, re.M)
    return found.group(1) if found else ""


def _required_files(text: str) -> list[str]:
    found = re.search(r"^\s*for f in (.+?); do\s*$", text, re.M)
    if not found:
        return []
    import shlex

    try:
        return shlex.split(found.group(1))
    except ValueError:
        return found.group(1).split()


def _gguf_variants(profile: dict, text: str):
    """สร้าง gguf_variants (พร้อม part/ขนาด/sha) จาก MODEL_FILES/MODEL_URLS/EXPECTED_* ของ controller เดิม"""
    from lmds.inspector.report import GgufPart, GgufVariant

    files = _bash_array(text, "MODEL_FILES")
    if files is None:
        raise RefreshError("controller เดิมไม่มีตาราง MODEL_FILES/EXPECTED_SIZES ให้ยกมา")
    urls = _bash_array(text, "MODEL_URLS") or []
    sizes = _bash_array(text, "EXPECTED_SIZES") or []
    shas = _bash_array(text, "EXPECTED_SHAS") or []
    revision = str((profile.get("model") or {}).get("revision") or "")
    features = profile.get("features") or {}
    projector = {n.rsplit("/", 1)[-1] for n in ((features.get("multimodal") or {}).get("projector_files") or [])}
    drafts = {n.rsplit("/", 1)[-1] for n in ((features.get("speculative") or {}).get("draft_files") or [])}
    mmproj_default = _default_of(text, "MMPROJ_FILE")
    mtp_default = _default_of(text, "MTP_FILE")
    if mmproj_default:
        projector.add(mmproj_default)
    if mtp_default:
        drafts.add(mtp_default)

    def part(i: int) -> GgufPart:
        name = files[i]
        url = urls[i] if i < len(urls) else ""
        marker = f"/resolve/{revision}/" if revision else "/resolve/"
        filename = url.split(marker, 1)[1] if marker in url else name
        size = sizes[i] if i < len(sizes) else ""
        sha = shas[i] if i < len(shas) else ""
        return GgufPart(filename=filename, size_bytes=int(size) if size.isdigit() else None, sha256=sha or None)

    main_parts, mmproj_parts, mtp_parts = [], [], []
    for i, name in enumerate(files):
        base = name.rsplit("/", 1)[-1]
        (mmproj_parts if base in projector else mtp_parts if base in drafts else main_parts).append(part(i))
    if not main_parts:
        raise RefreshError("MODEL_FILES ของ controller เดิมไม่มีไฟล์ weight")
    variants = [GgufVariant(
        filename=main_parts[0].filename,
        size_bytes=sum(p.size_bytes or 0 for p in main_parts) or None,
        sha256=main_parts[0].sha256 if len(main_parts) == 1 else None,
        parts=main_parts if len(main_parts) > 1 else [],
    )]
    variants += [GgufVariant(filename=p.filename, size_bytes=p.size_bytes, sha256=p.sha256, is_mmproj=True) for p in mmproj_parts]
    variants += [GgufVariant(filename=p.filename, size_bytes=p.size_bytes, sha256=p.sha256, is_mtp=True) for p in mtp_parts]
    return variants, main_parts[0].filename


def _need(mapping: dict, *keys: str, where: str):
    for key in keys:
        if mapping.get(key) in (None, ""):
            raise RefreshError(f"profile ไม่มี {where}.{key} (รุ่นเก่า)")


def plan_from_profile(profile: dict, controller_text: str = ""):
    """(plan, report, fit) จาก MODEL_PROFILE.yaml + หัว controller เดิม — ไม่ถามเน็ต · ขาดคีย์ = RefreshError"""
    from lmds.brain.plan_schema import (
        DeploymentPlan, Engine, Fact, Moe, Multimodal, Reasoning, RuntimeAsset, RuntimeChoice, Serving,
        Speculative, ToolCalling, Topology,
    )
    from lmds.fit.analyzer import GIB, FitReport, Verdict
    from lmds.fit.targets import MemoryModel
    from lmds.inspector.report import ArtifactType, ModelReport, ShardFile

    model = profile.get("model") or {}
    runtime = profile.get("runtime") or {}
    target = profile.get("target") or {}
    serving = profile.get("serving") or {}
    features = profile.get("features") or {}
    _need(model, "id", "revision", "served_name", "artifact_type", where="model")
    _need(runtime, "engine", "image", where="runtime")
    _need(serving, "context", where="serving")
    _need(target, "name", "memory_model", where="target")
    if not profile.get("topology"):
        raise RefreshError("profile ไม่มี topology (รุ่นเก่า)")
    try:
        engine = Engine(runtime["engine"])
        artifact = ArtifactType(model["artifact_type"])
        topology = Topology(profile["topology"])
        memory_model = MemoryModel(target["memory_model"])
    except ValueError as exc:
        raise RefreshError(f"ค่าใน profile ไม่รู้จัก: {exc}") from exc

    gguf_variants, selected = [], model.get("selected_gguf")
    shards: list[ShardFile] = []
    tokenizer_files: list[str] = []
    if engine is Engine.LLAMACPP:
        if not selected:
            raise RefreshError("profile ไม่มี model.selected_gguf")
        gguf_variants, selected = _gguf_variants(profile, controller_text)
    else:
        names = _bash_array(controller_text, "SHARD_FILES")
        sizes = _bash_array(controller_text, "SHARD_SIZES") or []
        if names is None:
            raise RefreshError("controller เดิมไม่มีตาราง SHARD_FILES ให้ยกมา")
        shards = [ShardFile(filename=n, size_bytes=int(sizes[i]) if i < len(sizes) and sizes[i].isdigit() else None)
                  for i, n in enumerate(names)]
        tokenizer_files = [f for f in _required_files(controller_text)
                           if f not in ("config.json", "model.safetensors.index.json")]

    moe = features.get("moe") or {}
    spec = features.get("speculative") or {}
    report = ModelReport(
        repo_id=model["id"], revision_sha=str(model["revision"]), gated=bool(model.get("gated")),
        license=model.get("license"), artifact_type=artifact, task=model.get("task") or "generate",
        params_total=model.get("params_total"), weight_bytes=model.get("weight_bytes"),
        shard_count=len(shards) or None, safetensor_shards=shards, tokenizer_files=tokenizer_files,
        architecture=model.get("architecture"),
        # profile ก่อน 0.6.1 ไม่มี gguf_architecture — แต่ architecture ของ bundle GGUF คือ general.architecture อยู่แล้ว
        # (inspector อ่านจาก header) → เติมให้ตรงนี้ hub จะได้ติดป้าย "runtime older than model" ได้ก่อน download (P2)
        gguf_architecture=model.get("gguf_architecture") or (model.get("architecture") if engine is Engine.LLAMACPP else None),
        context_length=model.get("native_context"), quantization=model.get("quantization"),
        moe_experts=moe.get("experts"), moe_experts_active=moe.get("experts_active"),
        mtp_embedded=bool(spec.get("embedded")), has_chat_template=True,
        gguf_variants=gguf_variants, selected_gguf=selected,
    )
    try:
        verdict = Verdict(target.get("verdict") or "unknown")
    except ValueError:
        verdict = Verdict.UNKNOWN
    fit = FitReport(
        target_name=str(target["name"]), memory_model=memory_model,
        engine_assumed="llamacpp" if engine is Engine.LLAMACPP else "vllm",
        weights_gb=(model.get("weight_bytes") or 0) / GIB or None,
        budget_gb=float(target.get("budget_gb") or 0.0), verdict=verdict,
        max_safe_context=target.get("max_safe_context"), kv_bytes_per_token=model.get("kv_bytes_per_token"),
        notes=list(target.get("fit_notes") or []),
    )
    serving_kwargs = {k: serving[k] for k in Serving.model_fields if serving.get(k) is not None}
    assets = []
    names, urls, shas = (_bash_array(controller_text, "ASSET_FILES"), _bash_array(controller_text, "ASSET_URLS"),
                         _bash_array(controller_text, "ASSET_SHAS"))
    for i, name in enumerate(names or []):
        assets.append(RuntimeAsset(filename=name, url=(urls or [""] * len(names))[i],
                                   sha256=((shas or [])[i] if shas and i < len(shas) and shas[i] else None)))
    plan = DeploymentPlan(
        model_id=model["id"], revision=str(model["revision"]), served_model_name=model["served_name"],
        artifact_type=artifact, selected_gguf=selected, task=model.get("task") or "generate",
        facts=[Fact(**f) for f in (profile.get("facts") or []) if isinstance(f, dict)],
        runtime=RuntimeChoice(engine=engine, image_ref=runtime["image"], image_pin=runtime.get("image_pin"),
                              native_dir=target.get("llamacpp_dir")),
        topology=topology, serving=Serving(**serving_kwargs),
        tool_calling=ToolCalling(**(features.get("tool_calling") or {})),
        reasoning=Reasoning(**(features.get("reasoning") or {})),
        multimodal=Multimodal(**(features.get("multimodal") or {})),
        speculative=Speculative(**spec), moe=Moe(**moe),
        warnings=list(profile.get("warnings") or []),
        flags_needing_approval=list(profile.get("flags_needing_approval") or []),
        runtime_assets=assets, generator=str(profile.get("generator") or "rule-based"),
    )
    return plan, report, fit


# ────────────────────────── ทำจริง ──────────────────────────
def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def refresh_bundle(server, *, if_older: bool = True) -> RefreshResult:
    """regenerate controller ของ bundle นี้ (ServerInfo จาก fleet.find/discover) — คืนผล ไม่โยน"""
    from lmds.fleet.bundle_settings import ensure_controller_reads
    from lmds.fleet.consistency import controller_state
    from lmds.fleet.manager import bundle_profile
    from lmds.generator import render_bundle
    from lmds.generator.renderer import resolve_slug
    from lmds.packager import write_checksums
    from lmds.validator import run_gates

    slug = server.slug
    if not server.controller or not server.controller_exists:
        return RefreshResult(slug, "missing", "ไม่พบไฟล์ controller", running=server.running)
    controller = Path(server.controller)
    profile = bundle_profile(str(controller)) or {}
    state = controller_state(profile, controller)
    result = RefreshResult(slug, "error", running=server.running, before=state.get("generated_by", ""))
    if state["state"] == "adopted":
        result.action = "adopted"
        return result
    if not profile:
        result.action = "needs-online"
        result.detail = "อ่าน MODEL_PROFILE.yaml ไม่ได้"
        return result
    if if_older and state["state"] == "ok":
        result.action = "current"
        result.after = result.before
        return result
    try:
        text = controller.read_text(encoding="utf-8", errors="replace")
        plan, report, fit = plan_from_profile(profile, text)
    except RefreshError as exc:
        result.action = "needs-online"
        result.detail = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 — profile ที่เพี้ยน (ชนิดผิด) ก็เข้าข่าย "ต้อง rebuild ออนไลน์"
        result.action = "needs-online"
        result.detail = f"อ่านแผนจาก profile ไม่ได้ ({type(exc).__name__}: {str(exc)[:120]})"
        return result

    output_root = controller.parent.parent
    try:
        target_slug, _ = resolve_slug(output_root, plan.model_id)
    except ValueError as exc:
        result.detail = str(exc)
        return result
    if target_slug != controller.parent.name:
        result.action = "needs-online"
        result.detail = f"ชื่อโฟลเดอร์ ({controller.parent.name}) ไม่ตรง slug ของโมเดล ({target_slug}) — render ทับที่เดิมไม่ได้"
        return result

    backup = controller.with_name(f"{controller.name}.replaced-{_stamp()}")
    serial = 1
    while backup.exists():   # regenerate สองรอบในวินาทีเดียว (เทส/สคริปต์) ต้องไม่ทับหลักฐานรอบแรก
        serial += 1
        backup = controller.with_name(f"{controller.name}.replaced-{_stamp()}-{serial}")
    shutil.copy2(controller, backup)
    result.replaced = str(backup)
    try:
        bundle = render_bundle(plan, report, fit, output_root)
    except Exception as exc:  # noqa: BLE001 — คืนของเดิมทุกกรณี ห้ามทิ้ง bundle ไว้ครึ่ง ๆ
        shutil.copy2(backup, controller)
        result.detail = f"render ไม่สำเร็จ ({str(exc)[:160]}) — คืน controller เดิมแล้ว"
        return result
    gates = run_gates(bundle.directory, include_checksums=False)
    failed = [f"{g.name}: {g.detail}" for g in gates if not g.passed]
    if failed:
        shutil.copy2(backup, bundle.controller)
        result.gates = failed
        result.detail = "controller ใหม่ไม่ผ่าน quality gates — คืนของเดิมแล้ว: " + "; ".join(failed)[:300]
        return result
    ensure_controller_reads(bundle.controller)
    try:
        write_checksums(bundle.directory)
    except OSError:
        pass
    result.action = "refreshed"
    result.after = controller_state(bundle_profile(str(bundle.controller)) or {}, bundle.controller).get("generated_by", "")
    result.detail = f"เก็บของเดิมไว้ที่ {backup.name}"
    return result


def refresh_bundles(slugs: list[str] | None = None, *, if_older: bool = True) -> list[RefreshResult]:
    """ทุก bundle บนเครื่องนี้ (หรือเฉพาะ slug ที่ระบุ) — ไม่มี bundle = ลิสต์ว่าง ไม่ใช่ error"""
    from lmds.fleet.manager import discover

    servers = [s for s in discover() if s.controller]
    if slugs:
        wanted = set(slugs)
        found = [s for s in servers if s.slug in wanted]
        missing = [RefreshResult(s, "missing", "ไม่รู้จัก bundle นี้ — ดู: lmds list") for s in slugs if s not in {f.slug for f in found}]
        return [refresh_bundle(s, if_older=if_older) for s in found] + missing
    return [refresh_bundle(s, if_older=if_older) for s in servers]
