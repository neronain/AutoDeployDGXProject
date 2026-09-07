"""ปุ่ม Fit บนการ์ดโมเดล — POST /api/models/{slug}/fit และ /api/nodes/{name}/models/{slug}/fit

body: {slots, context, apply} · apply=false = ดูตาราง · apply=true = เขียนลง bundle เหมือน `lmds set --fit`
โมเดลในเครื่อง: คำนวณที่นี่ (host/models จากแคช inventory ของ hub · log ของ vLLM อ่านสด) · โมเดลบนเครื่องอื่น:
สั่ง `lmds fit --json` / `lmds set --fit --json` บนเครื่องนั้นผ่าน SSH — ได้ log จริงและ nvidia-smi ของเครื่องนั้นเอง
โดยไม่ต้องลอกสูตรมาไว้สองที่
"""

from __future__ import annotations

import json
import shlex


class FitUnavailable(Exception):
    """ทำไม่ได้ในสภาพนี้ (409) — ข้อความบอกทางแก้ · `plan` = ตารางที่คำนวณได้ (ถ้ามี) ให้หน้าเว็บโชว์ว่าทำไม"""

    def __init__(self, message: str, plan: dict | None = None) -> None:
        super().__init__(message)
        self.plan = plan


def _clean(body: dict | None) -> tuple[int | None, int | None, bool]:
    body = body or {}

    def _int(name: str) -> int | None:
        value = body.get(name)
        if value in (None, ""):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} ต้องเป็นจำนวนเต็ม (ได้ {value!r})") from exc
        if number < 1:
            raise ValueError(f"{name} ต้องมากกว่า 0")
        return number

    return _int("slots"), _int("context"), bool(body.get("apply"))


def local_fit(slug: str, body: dict | None) -> dict:
    from lmds.fleet import find
    from lmds.fleet.sizing import FitError, apply, preview
    from lmds.web import state

    slots, context, do_apply = _clean(body)
    server = find(slug)
    if server is None or not server.controller:
        raise LookupError(f"ไม่รู้จัก {slug}")
    local = (state.STORE.snapshot().get("host") or {}).get("data") or {}
    host = local.get("host") or None
    models = local.get("models") or None
    try:
        plan = preview(server, slots, context, host, models)
    except FitError as exc:
        raise FitUnavailable(str(exc)) from exc
    out = {"slug": slug, "plan": plan, "applied": False}
    if do_apply:
        try:
            out["saved"] = apply(server, plan)
        except FitError as exc:
            raise FitUnavailable(str(exc), plan) from exc
        out["applied"] = True
        out["restart_needed"] = bool(server.running)
        state.STORE.invalidate_local()
    return out


def node_fit(name: str, slug: str, body: dict | None) -> dict:
    from lmds.nodes import NodeError, find, run
    from lmds.web import state

    slots, context, do_apply = _clean(body)
    node = find(name)
    if node is None:
        raise LookupError(f"ไม่รู้จักเครื่อง {name}")
    argv = ["lmds", "set", slug, "--fit", "--json"] if do_apply else ["lmds", "fit", slug, "--json"]
    if slots:
        argv += ["--slots", str(slots)]
    if context:
        argv += ["--context", str(context)]
    try:
        result = run(node, " ".join(shlex.quote(a) for a in argv), timeout=180)
    except NodeError as exc:
        raise FitUnavailable(str(exc)) from exc
    text = (result.stdout or "").strip()
    start = text.find("{")
    payload = None
    if start >= 0:
        try:
            payload = json.loads(text[start:])
        except ValueError:
            payload = None
    if payload is None:
        err = (result.stderr or text or "").strip()
        if "No such command" in err or "no such option" in err.lower():
            raise FitUnavailable(f"{name} ยังไม่มี `lmds fit` — อัปเดต LMDS บนเครื่องนั้นก่อน (ปุ่ม Update ที่การ์ดเครื่อง)")
        raise FitUnavailable(err[-400:] or f"{name} ไม่ตอบผล fit")
    if payload.get("error"):
        # ไม่พอ / คำนวณไม่ได้ — ส่งตารางกลับไปด้วยถ้ามี ให้ผู้ใช้เห็นว่าทำไม
        raise FitUnavailable(payload["error"], payload.get("plan"))
    plan = payload.get("plan") if "plan" in payload else payload
    out = {"node": name, "slug": slug, "plan": plan, "applied": do_apply}
    if do_apply:
        out["saved"] = payload.get("saved") or {}
        entry = (state.STORE.snapshot()["nodes"].get(name) or {}).get("data") or {}
        running = next((m.get("running") for m in (entry.get("models") or []) if m.get("slug") == slug), None)
        out["restart_needed"] = bool(running)
        state.STORE.force(name)
    return out
