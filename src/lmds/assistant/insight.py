"""probe ที่คำนวณบน hub — จากแคช inventory + ทะเบียน + สูตร recipes โดยไม่ SSH

สองอย่างในนี้ตอบไม่ได้ด้วยคำสั่งบนเครื่องเดียว:

  **fleet_consistency** — "ตรง hub" คือการเทียบ *ทุกเครื่อง* กับ hub · เครื่องปลายทางไม่รู้ template_hash
  ของ hub และไม่รู้ว่าเครื่องข้าง ๆ ค้างอะไร · hub มีคำตอบอยู่แล้วในแคชเดียวกับการ์ด Fleet consistency

  **model_recommend** — "งานแบบนี้ควรใช้ตัวไหน" ต้องมองข้ามเครื่อง: weight อยู่ที่ไหน รันอยู่หรือยัง
  เครื่องนั้นเหลือที่ไหม · ให้ LLM เดาชื่อโมเดลจากความจำคือคำตอบที่ฟลีตนี้ไม่มี weight

ผลเป็น *ข้อความ* ไม่ใช่ JSON — ไปอยู่ใน prompt ของผู้ช่วยเหมือน probe อื่น และต้องบอกเสมอว่า
ค่าไหน "ตรวจไม่ได้" (unknown ≠ ok — บทเรียนจาก audit 2026-09-06)
"""

from __future__ import annotations

import re

# คำที่คนพิมพ์ → แง่มุมที่ให้คะแนน · ไทย/อังกฤษปนกันได้ · หนึ่งโจทย์มีหลายแง่มุมได้ ("ไทย + vision")
_ASPECTS: dict[str, tuple[str, ...]] = {
    "coding": ("code", "coding", "coder", "program", "โค้ด", "เขียนโปรแกรม", "dev"),
    "thai": ("thai", "ไทย", "ภาษาไทย"),
    "vision": ("vision", "image", "ocr", "ภาพ", "รูป", "multimodal", "vl"),
    "tools": ("tool", "tools", "function", "agent", "mcp", "เครื่องมือ"),
    "uncensored": ("uncensored", "abliterated", "nsfw", "ไม่เซ็นเซอร์", "ถอดเซ็นเซอร์", "ไม่กรอง"),
    "long_context": ("long", "context", "ยาว", "128k", "256k", "เอกสารยาว"),
    "reasoning": ("reason", "reasoning", "think", "คิด", "วิเคราะห์"),
    "embedding": ("embed", "embedding", "vector", "rag"),
}

_UNCENSORED = re.compile(r"uncensored|abliterated|orcarouter|huihui|heretic|nsfw", re.I)
_CODER = re.compile(r"coder|code|devstral|codestral|starcoder", re.I)
_THAI_STRONG = re.compile(r"typhoon|thai|openthai", re.I)
_MULTILINGUAL = re.compile(r"gemma|qwen|glm|nemotron|llama-?3|mistral|deepseek|kimi", re.I)


def _aspects(task: str) -> list[str]:
    low = (task or "").lower()
    found = [name for name, words in _ASPECTS.items() if any(w in low for w in words)]
    return found or ["general"]


def _fleet_models() -> list[dict]:
    """ทุกโมเดลที่ hub เห็น พร้อมชื่อเครื่องและสภาพเครื่อง — จากแคชเดียวกับหน้าเว็บ"""
    from lmds.web import state

    snap = state.STORE.snapshot()
    rows: list[dict] = []
    local = (snap.get("host") or {}).get("data") or {}
    for m in local.get("models") or []:
        rows.append({"node": "this", "host": local.get("host") or {}, **m})
    for name, entry in (snap.get("nodes") or {}).items():
        data = entry.get("data") or {}
        for m in data.get("models") or []:
            rows.append({"node": name, "host": data.get("host") or {}, **m})
    return rows


def _free_gb(host: dict) -> tuple[float | None, float | None]:
    """(เหลือ, ทั้งหมด) เป็น GB — unified memory ใช้ RAM ของเครื่อง · discrete ใช้ VRAM รวมทุกการ์ด"""
    gpus = host.get("gpus") or []
    if host.get("memory_model") == "unified" and host.get("ram_total_gb"):
        total = float(host["ram_total_gb"])
        used = float(host.get("ram_used_gb") or 0)
        return round(total - used, 1), round(total, 1)
    if gpus:
        total = sum(float(g.get("vram_gb") or 0) for g in gpus)
        used = sum(float(g.get("vram_used_gb") or 0) for g in gpus)
        return (round(total - used, 1) if total else None), (round(total, 1) if total else None)
    return None, None


def _score(model: dict, aspects: list[str], recipe) -> tuple[int, list[str], list[str]]:
    """คะแนน + เหตุผลที่ให้ + เหตุผลที่ตัด — ทุกแต้มต้องมีคำอธิบายที่ผู้ช่วยเอาไปอ้างได้"""
    score = 0
    why: list[str] = []
    against: list[str] = []
    ident = f"{model.get('model_id') or ''} {model.get('slug') or ''} {(getattr(recipe, 'label', '') or '')}"
    features = str(model.get("features") or "")
    context = int(model.get("context") or 0)

    for aspect in aspects:
        if aspect == "coding":
            if _CODER.search(ident):
                score += 3
                why.append("ชื่อรุ่นเป็นสาย coder")
            else:
                score -= 1
        elif aspect == "thai":
            if _THAI_STRONG.search(ident):
                score += 3
                why.append("รุ่นที่ฝึกภาษาไทยโดยตรง")
            elif _MULTILINGUAL.search(ident):
                score += 2
                why.append("ตระกูลหลายภาษาที่ใช้ไทยได้ (ยังไม่ได้วัด — ยืนยันด้วย bench_results)")
            else:
                against.append("ไม่มีหลักฐานว่ารองรับไทย")
        elif aspect == "vision":
            if "image" in features or model.get("projector"):
                score += 3
                why.append("รับภาพได้ (มี projector/modality image)")
            else:
                score -= 5
                against.append("ไม่รับภาพ")
        elif aspect == "tools":
            if "tools" in features:
                score += 3
                why.append("เปิด tool calling ไว้แล้ว")
            elif recipe is not None and getattr(recipe, "tool_calling", {}).get("parser"):
                score += 1
                why.append("recipe มี parser ของ tools แต่ bundle นี้ยังไม่เปิด")
            else:
                against.append("ยังไม่เปิด tool calling (ต้องตั้ง --tool-parser แล้วเทส test-tools)")
        elif aspect == "uncensored":
            if _UNCENSORED.search(ident):
                score += 3
                why.append("รุ่นถอดเซ็นเซอร์")
            else:
                score -= 5
                against.append("ไม่ใช่รุ่นถอดเซ็นเซอร์")
        elif aspect == "long_context":
            if context >= 131072:
                score += 2
                why.append(f"context {context:,}")
            elif context >= 65536:
                score += 1
                why.append(f"context {context:,} (กลาง ๆ)")
            else:
                against.append(f"context {context:,}" if context else "ไม่รู้ context")
        elif aspect == "reasoning":
            if "reasoning" in features:
                score += 2
                why.append("มี reasoning parser")
            elif recipe is not None and getattr(recipe, "reasoning", {}).get("parser"):
                score += 1
                why.append("recipe รู้จัก reasoning parser ของรุ่นนี้")
        elif aspect == "embedding":
            if "embedding" in features:
                score += 3
                why.append("โมเดล embedding")
            else:
                score -= 5
                against.append("ไม่ใช่โมเดล embedding")
    if "embedding" in features and "embedding" not in aspects:
        score -= 5
        against.append("เป็น embedding ไม่ใช่ chat")
    if model.get("running"):
        score += 1
        why.append("รันอยู่ตอนนี้")
    if recipe is not None:
        score += 1
        why.append("มีสูตรที่รันผ่านจริง (recipes)")
    return score, why, against


def model_recommend(params: dict[str, str], target: str) -> str:
    """จัดอันดับโมเดลที่ฟลีต *มี weight อยู่แล้ว* ตามโจทย์ — พร้อมบอกว่าอยู่เครื่องไหน รันอยู่ไหม เครื่องนั้นเหลือที่ไหม"""
    from lmds.recipes import find_recipe

    task = params.get("task") or ""
    aspects = _aspects(task)
    models = _fleet_models()
    if not models:
        return "แคชยังไม่มีโมเดลของเครื่องไหนเลย — รอ refresher หรือกด Refresh ที่การ์ดก่อน"

    ranked: list[tuple[int, dict, list[str], list[str]]] = []
    for m in models:
        if not (m.get("downloaded") or m.get("running")):
            continue
        try:
            recipe = find_recipe(str(m.get("model_id") or ""))
        except Exception:
            recipe = None
        score, why, against = _score(m, aspects, recipe)
        ranked.append((score, m, why, against))
    ranked.sort(key=lambda r: (-r[0], str(r[1].get("slug"))))

    lines = [f"โจทย์: {task or '(ไม่ระบุ)'} → แง่มุมที่ให้คะแนน: {', '.join(aspects)}",
             "นับเฉพาะโมเดลที่มี weight บนเครื่องแล้ว (downloaded/running) · คะแนนคือการจัดอันดับ ไม่ใช่ผลวัด — ยืนยันด้วย bench_results/run_test",
             ""]
    if not ranked:
        lines.append("ไม่มีโมเดลที่มี weight ในฟลีต — ต้อง deploy ก่อน (deploy_model)")
        return "\n".join(lines)
    for rank, (score, m, why, against) in enumerate(ranked[:8], start=1):
        where = "เครื่องนี้ (hub)" if m["node"] == "this" else m["node"]
        free, total = _free_gb(m.get("host") or {})
        state = "รันอยู่" if m.get("running") else "มี weight แต่ยังไม่ได้ start"
        mem = f" · ใช้ {m['memory_gb']} GB" if m.get("memory_gb") else ""
        room = f" · เครื่องเหลือ {free}/{total} GB" if free is not None else ""
        lines.append(f"{rank}. {m.get('slug')} ({m.get('model_id') or '?'}) @ {where} — {state}{mem}{room}")
        lines.append(f"   engine {m.get('engine') or '?'} · features: {m.get('features') or 'text'} · context {int(m.get('context') or 0):,} · คะแนน {score}")
        if why:
            lines.append("   + " + " · ".join(why))
        if against:
            lines.append("   − " + " · ".join(against))
    skipped = [m.get("slug") for m in models if not (m.get("downloaded") or m.get("running"))]
    if skipped:
        lines.append("")
        lines.append("ไม่นับ (ยังไม่มี weight): " + ", ".join(str(s) for s in skipped[:10]))
    return "\n".join(lines)


def fleet_consistency(params: dict[str, str], target: str) -> str:
    """'ตรง hub' 3 มิติต่อเครื่อง จากแคชเดียวกับการ์ด Fleet consistency — unknown รายงานเป็น 'ตรวจไม่ได้' ไม่ใช่ผ่าน"""
    from lmds.fleet.consistency import fleet_report
    from lmds.web import state

    from lmds.nodes import load

    try:
        from lmds.config import Settings
        from lmds.nodes import in_saved_order

        registry = in_saved_order(load(), Settings.load().ui.node_order)  # ลำดับเดียวกับหน้าเว็บ
    except Exception:
        registry = load()

    snap = state.STORE.snapshot()
    report = fleet_report(snap.get("nodes") or {}, registry, (snap.get("host") or {}).get("data"))
    hub = report.get("hub") or {}
    lines = [f"hub: lmds {hub.get('version')} ({hub.get('commit') or '?'}) · template {hub.get('template_hash')}"]
    if hub.get("dirty"):
        lines.append(f"  hub มีไฟล์แก้ค้าง {len(hub['dirty'])} ไฟล์ — node install จะถูกปฏิเสธจนกว่าจะ commit")
    verdict = hub.get("verdict") or {}
    if verdict:
        lines.append(f"  bundle บน hub: controllers {((verdict.get('controllers') or {}).get('detail') or '-')}")
    mark = {"ok": "✓", "n/a": "✓", "unknown": "?"}
    # หนึ่งบรรทัดต่อเครื่อง (15 เครื่อง × 4 บรรทัดเกินงบ 4,000 ตัวของ probe) · ขยายเฉพาะมิติที่ไม่ผ่าน/ตรวจไม่ได้
    for n in report.get("nodes") or []:
        summary = "ตรง hub" if n.get("consistent") else ("ตรวจไม่ได้ครบ" if n.get("level") == "warn" else "ยังไม่ตรง")
        src = " (จากทะเบียน ยังไม่มี probe ล่าสุด)" if n.get("source") == "registry" else ""
        axes = {axis: (n.get(axis) or {}) for axis in ("code", "controllers", "runtimes")}
        marks = " · ".join(f"{axis} {mark.get(a.get('state'), '✗')}" for axis, a in axes.items())
        lines.append(f"- {n.get('name')}: {summary}{src} — {marks}")
        for axis, a in axes.items():
            # hub dirty ซ้ำทุกเครื่อง — บอกครั้งเดียวที่หัวรายงานพอ (15 เครื่อง × บรรทัดเดียวกันกินงบหมด)
            if a.get("state") not in ("ok", "n/a") and not (axis == "code" and a.get("state") == "dirty"):
                label = "ตรวจไม่ได้" if a.get("state") == "unknown" else str(a.get("state"))
                lines.append(f"    {axis}: {label} — {(a.get('detail') or '-')[:160]}")
    lines.append((report.get("summary") or {}).get("line") or "")
    lines.append("แก้: controllers ค้าง → bundles_refresh บนเครื่องนั้น · runtime ค้าง → update_runtime ของ slug นั้น · ทั้งหมดในครั้งเดียว → node_install")
    return "\n".join(line for line in lines if line is not None)
