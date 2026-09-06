"""ฟลีต "ตรง hub" ต้องผ่าน 3 มิติ — code · controllers · runtimes (audit Update path 2026-09-06 §5)

เดิม "ตรง hub" = เทียบ commit ของตัวโปรแกรม lmds อย่างเดียว · spark-worker รายงาน "พร้อมแล้ว — ตรง hub" ครบ 14 เครื่อง
แล้ว start ตายด้วย `unknown model architecture: 'qwen4exp'` เพราะ build llama.cpp บนเครื่องมาจาก 18 ส.ค. และ controller
ของ bundle ถูกสร้างโดย LMDS 0.5.1 — ไม่มีอะไรใน Update path ดู bundle/controller/build/image เลย

นิยามที่ใช้ทั้ง CLI (`lmds node install`, `lmds fleet check`, `lmds node list`) หน้าเว็บ (ปุ่ม Update, การ์ด Fleet consistency)
และ doctor (`controller-stale`) — จุดเดียว ไม่ต่างคนต่างตัดสิน:

  CODE        node.lmds_commit == hub.commit (prefix ≥7) · node ติดตั้งแล้วรันของที่ติดตั้ง · hub ไม่มีไฟล์แก้ค้าง
  CONTROLLER  ทุก bundle: template_hash ที่ renderer ฝังไว้ == template_hash ของ hub (ไม่ใช่เลข version —
              0.6.0 ↔ 0.6.1 ที่ template ไม่เปลี่ยนคือตัวเดียวกัน) · bundle จาก `lmds adopt` ไม่มี template = n/a
  RUNTIME     ทุก bundle llama.cpp: build/image ที่ผูกไว้รู้จัก arch ของโมเดล (runtime_arch.supported == true)
              · supported == null = "ตรวจไม่ได้" ไม่ใช่ผ่าน

สีต่อเครื่อง: เขียว = ok ทั้งสาม · เหลือง = unknown มิติใด ๆ · แดง = stale/behind/dirty มิติใด ๆ
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_SCRIPT_VERSION_RE = re.compile(r'^SCRIPT_VERSION="\$\{SCRIPT_VERSION:-([0-9.]+)\}"', re.M)
_TEMPLATE_HASH_RE = re.compile(r'^TEMPLATE_HASH="([0-9a-f]+)"', re.M)


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    """'lmds 0.5.1' / '0.6.1-dev' → (0, 5, 1) · None เมื่อไม่มีเลขรุ่น (เช่น 'lmds adopt')"""
    found = _VERSION_RE.search(text or "")
    return tuple(int(x) for x in found.groups()) if found else None  # type: ignore[return-value]


def same_commit(a: str | None, b: str | None) -> bool:
    """hash ย่อสองตัวชี้ commit เดียวกันไหม — prefix ≥7 (กติกาเดียวกับ cli._same_commit / sameCommit ของหน้าเว็บ)"""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def controller_header(controller: str | Path | None) -> dict:
    """SCRIPT_VERSION / TEMPLATE_HASH จากหัว controller — อ่านแค่ 8 KB แรก (hub ถามทุก 15 วิ)"""
    if not controller:
        return {"script_version": "", "template_hash": ""}
    try:
        with open(controller, "r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(8192)
    except OSError:
        return {"script_version": "", "template_hash": ""}
    version = _SCRIPT_VERSION_RE.search(head)
    digest = _TEMPLATE_HASH_RE.search(head)
    return {"script_version": version.group(1) if version else "",
            "template_hash": digest.group(1) if digest else ""}


def controller_state(profile: dict | None, controller: str | Path | None = None, *,
                     package_version: str | None = None, package_hash: str | None = None) -> dict:
    """controller ของ bundle นี้เก่ากว่าแพ็กเกจ lmds ที่ถืออยู่ไหม

    คืน {state: ok|stale|adopted|ahead|unknown, generated_by, template_hash, script_version, reason}
    · adopted = ไม่มี template ให้ regenerate (นับเป็น n/a ไม่ใช่ stale)
    · ไม่มี template_hash ในโปรไฟล์ = render ก่อนจะมีลายเซ็น → stale เสมอ แม้เลขรุ่นเท่ากัน (template ชุดนี้ใหม่กว่าแน่)
    """
    import lmds
    from lmds.inventory import self_managed_weights

    if package_version is None:
        package_version = lmds.__version__
    if package_hash is None:
        from lmds.generator.renderer import template_hash

        package_hash = template_hash()
    profile = profile or {}
    header = controller_header(controller)
    generated_by = str(profile.get("generated_by") or "")
    out = {
        "generated_by": generated_by.replace("lmds ", "", 1) if generated_by.startswith("lmds ") else generated_by,
        "template_hash": str(profile.get("template_hash") or header["template_hash"] or ""),
        "script_version": header["script_version"],
        "state": "unknown",
        "reason": "",
    }
    if not profile:
        out["reason"] = "อ่าน MODEL_PROFILE.yaml ไม่ได้"
        return out
    if generated_by.strip().endswith("adopt") or profile.get("adopted") or self_managed_weights(profile):
        out["state"] = "adopted"
        out["reason"] = "adopted — ไม่มี template ให้ regenerate"
        return out
    if not generated_by:
        out["reason"] = "โปรไฟล์ไม่มี generated_by"
        return out
    if out["template_hash"]:
        if out["template_hash"] == package_hash:
            out["state"] = "ok"
        else:
            out["state"] = "stale"
            out["reason"] = f"template ต่างจาก lmds ({out['template_hash']} ≠ {package_hash})"
        return out
    ours, theirs = parse_version(package_version), parse_version(generated_by)
    if theirs is None:
        out["reason"] = f"อ่านรุ่นจาก generated_by ไม่ได้ ({generated_by})"
        return out
    if ours is not None and theirs > ours:
        out["state"] = "ahead"
        out["reason"] = f"สร้างโดย lmds {out['generated_by']} ใหม่กว่าเครื่องนี้ ({package_version})"
        return out
    out["state"] = "stale"
    out["reason"] = (f"สร้างโดย lmds {out['generated_by']} ไม่มี template_hash — ก่อน template ชุดของ {package_version}")
    return out


# ────────────────────────── ตัดสินต่อเครื่อง ──────────────────────────
@dataclass
class Axis:
    state: str            # ok | stale | behind | ahead | dirty | unknown | n/a
    detail: str = ""      # บรรทัดที่พิมพ์/โชว์ต่อจากชื่อมิติ
    items: list[str] = field(default_factory=list)   # bundle ที่ค้าง (stale) หรือตรวจไม่ได้ (unknown)

    @property
    def ok(self) -> bool:
        return self.state in ("ok", "n/a")

    @property
    def known(self) -> bool:
        return self.state != "unknown"

    def payload(self) -> dict:
        return {"state": self.state, "detail": self.detail, "items": list(self.items)}


@dataclass
class Verdict:
    code: Axis
    controllers: Axis
    runtimes: Axis
    version: str = ""
    commit: str = ""

    @property
    def consistent(self) -> bool:
        """"ตรง hub" — ok ครบสามมิติเท่านั้น · unknown ไม่นับว่าผ่าน"""
        return self.code.ok and self.controllers.ok and self.runtimes.ok

    @property
    def level(self) -> str:
        axes = (self.code, self.controllers, self.runtimes)
        if any(not a.ok and a.known for a in axes):
            return "bad"
        if any(not a.known for a in axes):
            return "warn"
        return "ok"

    def payload(self) -> dict:
        return {"code": self.code.payload(), "controllers": self.controllers.payload(),
                "runtimes": self.runtimes.payload(), "consistent": self.consistent, "level": self.level,
                "version": self.version, "commit": self.commit}


def hub_facts() -> dict:
    """สิ่งที่ hub ใช้เทียบ — commit/version/template_hash ของแพ็กเกจที่รันอยู่ + ไฟล์แก้ค้างใน checkout"""
    import lmds
    from lmds.generator.renderer import template_hash
    from lmds.inventory import source_commit

    dirty: list[str] = []
    try:
        from lmds.web.selfupdate import dirty_files, source_root

        root = source_root()
        if root is not None:
            dirty = dirty_files(root)
    except Exception:  # noqa: BLE001 — ไม่มีส่วนเว็บ/ไม่ใช่ checkout = ไม่มีของค้าง
        dirty = []
    return {"version": lmds.__version__, "commit": source_commit(), "template_hash": template_hash(), "dirty": dirty}


def code_axis(host: dict | None, hub: dict) -> Axis:
    host = host or {}
    version, commit = str(host.get("lmds_version") or ""), str(host.get("lmds_commit") or "")
    label = f"{version} ({commit})" if version and commit else (version or commit or "?")
    if not commit:
        return Axis("unknown", f"{label} · ตรวจไม่ได้ (เครื่องนั้นไม่รายงาน commit — lmds รุ่นเก่า)")
    if not hub.get("commit"):
        return Axis("unknown", f"{label} · ตรวจไม่ได้ (hub ไม่รู้ commit ของตัวเอง — ไม่ได้ติดตั้งจาก git)")
    if not same_commit(commit, hub["commit"]):
        return Axis("behind", f"{label} · ยังไม่ตรง hub (hub: {hub['commit']})")
    installed = str(host.get("lmds_installed_commit") or "")
    if installed and not same_commit(installed, commit):
        return Axis("behind", f"{label} · ติดตั้ง {installed} แล้วแต่ยังรันโค้ดเก่า")
    if hub.get("dirty"):
        names = ", ".join(hub["dirty"][:3]) + (" …" if len(hub["dirty"]) > 3 else "")
        return Axis("dirty", f"{label} · commit ตรง แต่ hub มีไฟล์แก้ค้าง {len(hub['dirty'])} ไฟล์ ({names}) — node ได้ commit ไม่ใช่โค้ดที่ hub รัน")
    return Axis("ok", f"{label} · ตรง hub")


def _controller_of(model: dict, hub: dict) -> dict:
    """state ของ controller หนึ่งใบเทียบกับ template ของ hub — คำนวณจากฟิลด์ดิบเมื่อมี ไม่งั้นเชื่อที่ node ตัดสินไว้"""
    own = model.get("controller") if isinstance(model.get("controller"), dict) else None
    if own and own.get("state") == "adopted":
        return own
    generated_by = model.get("generated_by")
    template = model.get("template_hash") or (own or {}).get("template_hash")
    if generated_by is None and own is None:
        return {"state": "unknown", "generated_by": "", "reason": "node รุ่นเก่าไม่รายงาน generated_by"}
    profile = {"generated_by": generated_by or (f"lmds {own['generated_by']}" if own and own.get("generated_by") else ""),
               "template_hash": template}
    if model.get("self_managed_weights"):
        profile["adopted"] = True
    return controller_state(profile, None, package_version=hub.get("version"), package_hash=hub.get("template_hash"))


def controllers_axis(models: list[dict] | None, hub: dict) -> Axis:
    models = models or []
    if not models:
        return Axis("n/a", "ไม่มี bundle")
    stale, unknown, adopted, ahead = [], [], [], []
    for m in models:
        if m.get("stacked_role") == "worker":
            continue  # bundle อยู่ที่ head — การ์ด worker เป็นเงา
        state = _controller_of(m, hub)
        tag = f"{m.get('slug')} ({state.get('generated_by') or '?'})"
        if state["state"] == "stale":
            stale.append(tag)
        elif state["state"] == "unknown":
            unknown.append(f"{m.get('slug')}: {state.get('reason') or 'ตรวจไม่ได้'}")
        elif state["state"] == "adopted":
            adopted.append(str(m.get("slug")))
        elif state["state"] == "ahead":
            ahead.append(tag)
    total = len([m for m in models if m.get("stacked_role") != "worker"])
    extra = f" · adopted {len(adopted)} (ไม่มี template)" if adopted else ""
    if stale:
        return Axis("stale", f"{total} ใบ · เก่ากว่า lmds {len(stale)} ใบ: {', '.join(stale)}{extra}", [s.split(" (")[0] for s in stale])
    if ahead:
        return Axis("ahead", f"{total} ใบ · ใหม่กว่า hub {len(ahead)} ใบ: {', '.join(ahead)}{extra}", [s.split(" (")[0] for s in ahead])
    if unknown:
        return Axis("unknown", f"{total} ใบ · ตรวจไม่ได้ {len(unknown)} ใบ ({'; '.join(unknown)}){extra}",
                    [u.split(":")[0] for u in unknown])
    return Axis("ok", f"{total} ใบ · ตรง template ของ hub{extra}")


def _runtime_label(host: dict | None) -> str:
    builds = ((host or {}).get("runtimes") or {}).get("llamacpp") or []
    parts = []
    for b in builds:
        dir_ = str(b.get("dir") or "").replace(str(Path.home()), "~")
        build, commit, date = b.get("build") or "?", (b.get("commit") or "")[:9], b.get("date") or ""
        lock = b.get("lock_state")
        parts.append(f"llama.cpp {dir_} build {build} ({commit}{', ' + date if date else ''})"
                     + (f" · lock {lock}" if lock and lock != "ok" else ""))
    return " / ".join(parts)


def runtimes_axis(models: list[dict] | None, host: dict | None) -> Axis:
    models = [m for m in (models or []) if m.get("stacked_role") != "worker"]
    llama = [m for m in models if (m.get("engine") or "") == "llamacpp"]
    label = _runtime_label(host)
    if not llama:
        return Axis("n/a", (label + " · " if label else "") + "ไม่มี bundle llama.cpp")
    stale, unknown, good = [], [], []
    for m in llama:
        arch = m.get("runtime_arch") if isinstance(m.get("runtime_arch"), dict) else None
        if arch is None:
            reason = "profile เก่าไม่มี gguf_architecture และยังไม่ download" if not m.get("downloaded") \
                else "อ่าน arch จากไฟล์ไม่ได้"
            unknown.append(f"{m.get('slug')}: {reason}")
        elif arch.get("supported") is False:
            stale.append(f"{m.get('slug')}: {arch.get('runtime')} ไม่รู้จัก {arch.get('arch')} → {arch.get('fix')}")
        elif arch.get("supported") is None:
            unknown.append(f"{m.get('slug')}: ยังไม่ได้ build/pull รันไทม์ — ถาม arch {arch.get('arch')} ไม่ได้")
        else:
            good.append(f"{m.get('slug')}: arch {arch.get('arch')} ✓")
    head = (label + " · ") if label else ""
    if stale:
        return Axis("stale", head + f"runtime ค้าง {len(stale)} — " + " · ".join(stale), [s.split(":")[0] for s in stale])
    if unknown:
        return Axis("unknown", head + "ตรวจไม่ได้ " + " · ".join(unknown), [u.split(":")[0] for u in unknown])
    return Axis("ok", head + " · ".join(good))


def node_verdict(info: dict | None, hub: dict | None = None) -> Verdict:
    """ตัดสิน 3 มิติจาก payload ของ `lmds agent info` (หรือ snapshot ในแคชของ hub) — ไม่ SSH"""
    hub = hub or hub_facts()
    info = info or {}
    host = info.get("host") or {}
    models = info.get("models") or []
    return Verdict(code=code_axis(host, hub), controllers=controllers_axis(models, hub),
                   runtimes=runtimes_axis(models, host),
                   version=str(host.get("lmds_version") or ""), commit=str(host.get("lmds_commit") or ""))


def verdict_from_registry(node, hub: dict | None = None) -> Verdict:
    """ตัดสินจากที่ทะเบียนจำไว้ (`status_from_probe`) — ใช้เมื่อไม่มี snapshot เต็ม เช่น `lmds fleet check` จาก CLI

    ทะเบียนเก็บแค่ตัวนับ (controllers_stale / runtime_stale / llamacpp_build) จึงบอกได้ว่าค้างกี่ใบ แต่ไม่รู้ชื่อ
    """
    hub = hub or hub_facts()
    host = {"lmds_version": node.lmds_version, "lmds_commit": node.lmds_commit}
    code = code_axis(host, hub)
    stale = getattr(node, "controllers_stale", None)
    if stale is None:
        controllers = Axis("unknown", "ตรวจไม่ได้ (ทะเบียนยังไม่มีข้อมูล controller — probe เครื่องนั้นก่อน)")
    elif stale > 0:
        controllers = Axis("stale", f"เก่ากว่า lmds {stale} ใบ")
    else:
        controllers = Axis("ok", "ตรง template ของ hub")
    runtime_stale = getattr(node, "runtime_stale", None)
    build = getattr(node, "llamacpp_build", "") or ""
    if runtime_stale is None:
        runtimes = Axis("unknown", "ตรวจไม่ได้ (ทะเบียนยังไม่มีข้อมูล runtime)")
    elif runtime_stale > 0:
        runtimes = Axis("stale", (f"llama.cpp {build} · " if build else "") + f"runtime ค้าง {runtime_stale}")
    else:
        runtimes = Axis("ok", f"llama.cpp {build}" if build else "ไม่มี bundle llama.cpp ที่ค้าง")
    return Verdict(code=code, controllers=controllers, runtimes=runtimes,
                   version=node.lmds_version, commit=node.lmds_commit)


# ────────────────────────── ข้อความ ──────────────────────────
def _mark(axis: Axis) -> str:
    return "✓" if axis.ok else ("?" if not axis.known else "✗")


def summary_line(verdict: Verdict) -> str:
    """บรรทัดสรุปต่อเครื่อง — "ตรง hub ✓" พิมพ์ได้ต่อเมื่อ 3 มิติ ok เท่านั้น"""
    marks = f"code {_mark(verdict.code)} · controller {_mark(verdict.controllers)} · runtime {_mark(verdict.runtimes)}"
    if verdict.consistent:
        return f"ตรง hub ✓ ({marks})"
    problems = []
    for name, axis in (("code", verdict.code), ("controller", verdict.controllers), ("runtime", verdict.runtimes)):
        if axis.ok:
            continue
        if not axis.known:
            problems.append(f"{name} ตรวจไม่ได้ ({axis.detail})")
        else:
            what = {"stale": "ค้าง", "behind": "ยังไม่ตรง", "dirty": "hub มีของแก้ค้าง", "ahead": "ใหม่กว่า hub"}.get(axis.state, axis.state)
            problems.append(f"{name} {what}" + (f" {len(axis.items)} ({', '.join(axis.items)})" if axis.items else f" ({axis.detail})"))
    return f"ยังไม่ตรง hub — {' · '.join(problems)} ({marks})"


def verdict_lines(verdict: Verdict) -> list[str]:
    """4 บรรทัดที่ Update job พิมพ์ต่อเครื่อง (spec §5.3) — code / controllers / runtime / สรุป"""
    return [
        f"  code        {verdict.code.detail}",
        f"  controllers {verdict.controllers.detail}",
        f"  runtime     {verdict.runtimes.detail}",
        f"  สรุป        {summary_line(verdict)}",
    ]


def fleet_summary_line(verdicts: dict[str, Verdict]) -> str:
    """บรรทัดท้าย `--all`: อัปเดตครบ N · ตรง hub n · controller ค้าง n (…) · runtime ค้าง n (…) · ตรวจไม่ได้ n"""
    total = len(verdicts)
    good = [n for n, v in verdicts.items() if v.consistent]
    ctl = [n for n, v in verdicts.items() if v.controllers.state in ("stale", "ahead")]
    rt = [n for n, v in verdicts.items() if v.runtimes.state == "stale"]
    code = [n for n, v in verdicts.items() if v.code.state in ("behind", "dirty")]
    unknown = [n for n, v in verdicts.items() if not v.consistent and v.level == "warn"]

    def group(label: str, names: list[str]) -> str:
        return f"{label} {len(names)}" + (f" ({', '.join(names)})" if names else "")

    return (f"อัปเดตครบ {total} เครื่อง · ตรง hub {len(good)} · {group('code ไม่ตรง', code)} · "
            f"{group('controller ค้าง', ctl)} · {group('runtime ค้าง', rt)} · {group('ตรวจไม่ได้', unknown)}")


def fleet_report(snapshot_nodes: dict, registry_nodes: list, local_info: dict | None, hub: dict | None = None) -> dict:
    """รายงานทั้งฟลีตจากสิ่งที่ hub ถืออยู่แล้ว (แคช `agent info` + ทะเบียน) — ไม่ SSH · ใช้ร่วมกันโดย CLI และ API"""
    hub = hub or hub_facts()
    out: dict = {"hub": {**hub, "verdict": None}, "nodes": [], "summary": {}}
    if local_info is not None:
        local = node_verdict({"host": {**(local_info.get("host") or {}), "lmds_commit": hub["commit"],
                                       "lmds_version": hub["version"]},
                              "models": local_info.get("models") or []}, hub)
        out["hub"]["verdict"] = local.payload()
    verdicts: dict[str, Verdict] = {}
    for node in registry_nodes:
        cached = (snapshot_nodes or {}).get(node.name) or {}
        data = cached.get("data")
        if data:
            verdict = node_verdict(data, hub)
            source = "cache"
        else:
            verdict = verdict_from_registry(node, hub)
            source = "registry"
        verdicts[node.name] = verdict
        out["nodes"].append({"name": node.name, "site": getattr(node, "site", ""), "source": source,
                             "reachable": bool(data) or not getattr(node, "last_error", ""),
                             "error": getattr(node, "last_error", "") if not data else "",
                             **verdict.payload()})
    out["summary"] = {
        "total": len(verdicts),
        "consistent": sum(1 for v in verdicts.values() if v.consistent),
        "controllers_stale": sum(len(v.controllers.items) for v in verdicts.values() if v.controllers.state in ("stale", "ahead")),
        "runtime_stale": sum(len(v.runtimes.items) for v in verdicts.values() if v.runtimes.state == "stale"),
        "code_behind": sum(1 for v in verdicts.values() if v.code.state in ("behind", "dirty")),
        "unknown": sum(1 for v in verdicts.values() if not v.consistent and v.level == "warn"),
        "line": fleet_summary_line(verdicts),
    }
    return out
