"""Web UI — หน้าเดียวสำหรับคุม fleet + doctor (เฟส 2)

ชั้นนี้ไม่มี logic ของตัวเอง: เรียก core เดิมทั้งหมด (hardware / fleet / doctor)
แล้วแปลงเป็น JSON เท่านั้น — อะไรที่ CLI ทำได้ เว็บต้องได้ผลเหมือนกันเป๊ะ

ความปลอดภัย (ตาม PRD §9): หน้านี้สั่ง start/stop โมเดลได้ จึง
- bind 127.0.0.1 เป็นค่าเริ่มต้น — ต้องตั้งใจเปิดออก network เอง
- ตั้ง token ได้ และถ้า bind ออก network โดยไม่มี token จะเตือนเสียงดัง
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import shlex
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import lmds
from lmds.web import state

STATIC = Path(__file__).parent / "static"

# ถี่แค่ไหนถึงจะเช็คว่ามีอะไรเปลี่ยน — แค่เทียบ int ในหน่วยความจำ ไม่แตะ SSH
_EVENT_TICK = 0.5
_EVENT_KEEPALIVE = 15.0


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _host_payload() -> dict:
    from lmds.inventory import host_payload

    return host_payload()


def _active_job(slug: str) -> dict | None:
    from . import jobs

    job = jobs.active_for(slug)
    return {"id": job.id, "command": job.command} if job else None


def _model_payload(server) -> dict:
    from lmds.inventory import model_payload

    return model_payload(server, _active_job(server.slug))


def create_app(token: str = "") -> FastAPI:
    app = FastAPI(title="LMDS", docs_url=None, redoc_url=None, openapi_url=None)

    def require_token(request: Request) -> None:
        if token:
            supplied = request.headers.get("x-lmds-token") or request.query_params.get("token", "")
            # compare_digest กัน timing attack — เทียบสตริงตรง ๆ รั่วความยาวและ prefix
            if not secrets.compare_digest(supplied, token):
                raise HTTPException(status_code=401, detail="token ไม่ถูกต้อง")
        elif request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            # โหมดไม่มี token มีไว้ให้ loopback เท่านั้น ต้องตรวจ Host ด้วย ไม่ใช่แค่เชื่อว่า
            # uvicorn bind loopback: DNS rebinding ทำให้ origin ของผู้โจมตี resolve มาที่ 127.0.0.1
            # แล้วส่ง Host ของตัวเองเข้ามาได้ ซึ่ง Origin comparison อย่างเดียวจะมองว่า same-origin
            raise HTTPException(status_code=421, detail="โหมดไม่มี token รับเฉพาะ localhost")

        # token ป้องกันคนที่ไม่รู้ secret แต่ค่าเริ่มต้นบน loopback จงใจไม่มี token — ถ้าไม่
        # กัน CSRF เว็บไซต์ใด ๆ ที่ผู้ใช้เปิดอยู่สามารถ submit form มาที่ 127.0.0.1 แล้วสั่ง
        # stop/remove/install ได้ ถึงอ่าน response ไม่ได้ก็เปลี่ยนสถานะสำเร็จแล้ว
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            fetch_site = request.headers.get("sec-fetch-site", "").lower()
            if fetch_site == "cross-site":
                raise HTTPException(status_code=403, detail="ไม่รับคำสั่งข้าม origin")
            origin = request.headers.get("origin", "").rstrip("/")
            if origin:
                expected = f"{request.url.scheme}://{request.headers.get('host', '')}".rstrip("/")
                if not secrets.compare_digest(origin, expected):
                    raise HTTPException(status_code=403, detail="origin ไม่ตรงกับหน้าเว็บ")

    guarded = [Depends(require_token)]

    # ตัวเดียวที่คุยกับ node จริง — endpoint ทุกตัวอ่านจากแคชที่มันเติมให้ จึงตอบทันทีเสมอ
    state.start_refresher()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        html = html.replace("<script>", f'<script nonce="{nonce}">')
        return HTMLResponse(html, headers={
            # script มีตัวเดียวและได้ nonce ใหม่ทุก response — HTML ที่หลุดเข้า innerHTML
            # จึงเรียก <script>/onerror/javascript: ไม่ได้ แม้ sink ใหม่จะเผลอหลุดมาในอนาคต
            "Content-Security-Policy": (
                "default-src 'none'; "
                f"script-src 'nonce-{nonce}'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
            ),
            # token อยู่ใน query string เพื่อให้ EventSource/bookmark ใช้ได้ — ห้ามส่ง URL นั้น
            # เป็น Referer ตอนกด source link ออกไปข้างนอก
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Cache-Control": "no-store",
        })

    @app.get("/api/version", dependencies=guarded)
    def version() -> dict:
        return {"version": lmds.__version__}

    @app.get("/api/host", dependencies=guarded)
    def host() -> dict:
        # อ่านจากแคช — แต่ครั้งแรกยังไม่มีข้อมูล ต้องคำนวณสด ไม่งั้นหน้าเว็บว่างเปล่าตอนเปิด
        cached = (state.STORE.snapshot()["host"] or {}).get("data")
        return cached["host"] if cached else _host_payload()

    @app.get("/api/models", dependencies=guarded)
    def models() -> dict:
        from lmds.fleet import discover

        cached = (state.STORE.snapshot()["host"] or {}).get("data")
        if cached:
            return {"models": cached["models"]}
        return {"models": [_model_payload(s) for s in discover()]}

    @app.get("/api/events", dependencies=guarded)
    async def events(request: Request) -> StreamingResponse:
        """สตรีมสถานะแทนการให้เบราว์เซอร์ถามซ้ำ ๆ

        เดิม poll ทุก 5 วิ = SSH ไปทุกเครื่องทุกรอบ · ผ่าน relay 150ms ต่อเครื่องแล้วกระตุก
        ตอนนี้ refresher เบื้องหลังคุยกับ node ตัวเดียว แล้ว push ให้ทุกหน้าที่เปิดอยู่

        เป็น async และเช็ก is_disconnected() เพราะ generator แบบ blocking จะค้างอยู่หลัง
        ผู้ใช้ปิดแท็บไปแล้ว — เปิดหน้าเว็บทิ้งไว้ทั้งวันจะสะสม thread ค้างเรื่อย ๆ
        """

        async def stream():
            last = -1
            idle = 0.0
            while not await request.is_disconnected():
                snapshot = state.STORE.snapshot()
                if snapshot["version"] != last:
                    last, idle = snapshot["version"], 0.0
                    yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                else:
                    idle += _EVENT_TICK
                    if idle >= _EVENT_KEEPALIVE:
                        idle = 0.0
                        yield ": keepalive\n\n"   # กัน proxy ปิด connection ที่เงียบนานเกิน
                # เทียบเลขเวอร์ชันในหน่วยความจำ ไม่ได้แตะ SSH — ถูกมากพอที่จะเช็คถี่ ๆ ได้
                await asyncio.sleep(_EVENT_TICK)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.get("/api/models/{slug}/doctor", dependencies=guarded)
    def doctor(slug: str) -> dict:
        from lmds.doctor import diagnose

        result = diagnose(slug)
        return {
            "slug": result.slug,
            "healthy": result.healthy,
            "findings": [
                {"name": f.name, "status": f.status.value, "detail": f.detail, "fix": f.fix}
                for f in result.findings
            ],
        }

    @app.get("/api/models/{slug}/logs", dependencies=guarded)
    def logs(slug: str, lines: int = Query(200, ge=1, le=2000)) -> dict:
        from lmds.fleet import FleetError, find, logs_text

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        try:
            return {"slug": slug, "text": logs_text(server, lines)}
        except FleetError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _action(slug: str, verb: str, options: dict | None = None) -> JSONResponse:
        from lmds.fleet import FleetError, find, restart_server, start_server, stop_server

        from . import jobs

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        runner = {"start": start_server, "stop": stop_server, "restart": restart_server}[verb]
        try:
            if verb == "stop" and options:
                raise jobs.JobError("stop ไม่รับตัวเลือก start/restart")
            # ส่ง env ให้ subprocess โดยตรง ห้ามแก้ os.environ ของ process เว็บ: FastAPI รัน
            # sync handlers พร้อมกันหลาย thread และ API_KEY/port ของ request หนึ่งจะไหลไปอีกตัวได้
            extra_env = jobs.controller_env(options)
            outcome = runner(server) if verb == "stop" or not extra_env else runner(server, env=extra_env)
        except jobs.JobError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FleetError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # start คืน exit code (int) ส่วน stop/restart คืนวิธีที่ใช้ (str)
        ok = outcome == 0 if isinstance(outcome, int) else True
        return JSONResponse(
            {"slug": slug, "action": verb, "ok": ok, "outcome": outcome},
            status_code=200 if ok else 500,
        )

    @app.post("/api/models/{slug}/start", dependencies=guarded)
    def start(slug: str, body: dict | None = None) -> JSONResponse:
        state.STORE.invalidate_local()
        return _action(slug, "start", body)

    @app.post("/api/models/{slug}/stop", dependencies=guarded)
    def stop(slug: str, body: dict | None = None) -> JSONResponse:
        state.STORE.invalidate_local()
        return _action(slug, "stop", body)

    @app.post("/api/models/{slug}/restart", dependencies=guarded)
    def restart(slug: str, body: dict | None = None) -> JSONResponse:
        state.STORE.invalidate_local()
        return _action(slug, "restart", body)

    # ── deploy wizard ──────────────────────────────────────────────────────
    @app.get("/api/targets", dependencies=guarded)
    def targets_list() -> dict:
        from .deploy import targets

        return {"targets": targets()}

    def _deploy_error(exc) -> HTTPException:
        # kind บอก UI ว่าให้ทำอะไรต่อ (ขอ token / ให้เลือกไฟล์ / แสดงทางเลือกอื่น)
        return HTTPException(status_code=422, detail={"kind": exc.kind, "message": exc.message, **exc.extra})

    @app.post("/api/deploy/analyze", dependencies=guarded)
    def deploy_analyze(body: dict) -> dict:
        from .deploy import DeployError, analyze

        model = (body.get("model") or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="ต้องระบุลิงก์โมเดล")
        try:
            return analyze(
                model,
                target=body.get("target") or None,
                revision=body.get("revision") or None,
                no_llm=bool(body.get("no_llm")),
                hf_token=body.get("hf_token") or "",
                selected_gguf=body.get("selected_gguf") or "",
            )
        except DeployError as exc:
            raise _deploy_error(exc) from exc

    @app.post("/api/deploy/{session_id}/generate", dependencies=guarded)
    def deploy_generate(session_id: str, body: dict) -> dict:
        from .deploy import DeployError, generate

        try:
            return generate(
                session_id,
                context=body.get("context"),
                approved_flags=body.get("approved_flags") or [],
                approved_assets=body.get("approved_assets") or [],
                output=body.get("output") or "./bundles",
            )
        except DeployError as exc:
            raise _deploy_error(exc) from exc

    # ── งานที่ใช้เวลานาน: download / start / verify ────────────────────────
    @app.post("/api/models/{slug}/run/{command}", dependencies=guarded)
    def run_command(slug: str, command: str, body: dict | None = None) -> dict:
        state.STORE.invalidate_local()
        from lmds.fleet import find

        from . import jobs

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        try:
            return jobs.start(slug, command, server.controller, body).payload()
        except jobs.JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}", dependencies=guarded)
    def job_status(job_id: str) -> dict:
        from . import jobs

        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบงานนี้")
        return job.payload()

    # ── ลบ / autostart — ทำผ่าน fleet ตรง ๆ ไม่ใช่ job เพราะไม่ใช่คำสั่งของ controller ──
    @app.get("/api/models/{slug}/removal-plan", dependencies=guarded)
    def removal_preview(slug: str, keep_weights: bool = False) -> dict:
        from lmds.fleet import find, removal_plan

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        items = removal_plan(server, include_weights=not keep_weights)
        return {
            "slug": slug,
            "items": [{"label": i.label, "path": str(i.path), "bytes": i.size_bytes,
                       "is_weights": i.is_weights} for i in items],
            "total_bytes": sum(i.size_bytes for i in items),
        }

    @app.post("/api/models/{slug}/remove", dependencies=guarded)
    def remove(slug: str, body: dict | None = None) -> dict:
        state.STORE.invalidate_local()
        from lmds.fleet import find, remove_server

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        unknown = sorted(set(body or {}) - {"confirm", "keep_weights"})
        if unknown:
            raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")
        if (body or {}).get("confirm") != slug:
            raise HTTPException(status_code=400, detail="ต้องยืนยันด้วยชื่อโมเดลที่จะลบ")
        keep = (body or {}).get("keep_weights", False)
        if not isinstance(keep, bool):
            raise HTTPException(status_code=400, detail="keep_weights ต้องเป็น true หรือ false")
        return {"slug": slug, "done": remove_server(server, include_weights=not keep)}

    @app.post("/api/models/{slug}/autostart", dependencies=guarded)
    def autostart(slug: str, body: dict | None = None) -> dict:
        state.STORE.invalidate_local()
        from lmds.fleet import FleetError, disable_autostart, enable_autostart, find

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        unknown = sorted(set(body or {}) - {"enabled"})
        if unknown:
            raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")
        enabled = (body or {}).get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="enabled ต้องเป็น true หรือ false")
        try:
            # ต้องใช้ sudo — เว็บไม่มี tty ให้กรอกรหัส ถ้าไม่ผ่าน FleetError จะบอกคำสั่งให้รันเอง
            name = enable_autostart(server) if enabled else disable_autostart(server)
        except FleetError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"slug": slug, "unit": name, "enabled": enabled}

    # ── fleet หลายเครื่อง — hub คุม node อื่นผ่าน SSH ────────────────────────
    @app.get("/api/nodes", dependencies=guarded)
    def nodes_list() -> dict:
        from lmds.nodes import load

        return {"nodes": [
            {"name": n.name, "host": n.host, "user": n.user, "port": n.port, "note": n.note,
             "lmds_version": n.lmds_version, "last_seen": n.last_seen, "last_error": n.last_error,
             "cluster_ip": n.cluster_ip, "cluster_iface": n.cluster_iface,
             # ที่อยู่สำรอง — หน้าเว็บต้องเติมค่าเดิมให้ตอนแก้ ไม่ให้พิมพ์ใหม่ทั้งหมด
             "alt_hosts": n.alt_hosts}
            for n in load()
        ]}

    @app.get("/api/nodes/{name}/inventory", dependencies=guarded)
    def node_inventory(name: str, refresh: bool = False) -> dict:
        """สถานะของ node หนึ่งเครื่อง — เครื่องล่มต้องไม่ทำให้ทั้งหน้าพัง จึงคืน reachable=false"""
        from lmds.nodes import NodeError, find, probe, update

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")

        # ปกติอ่านจากแคช (ตอบทันทีแม้เครื่องนั้นอยู่ไกล) · refresh=true = ผู้ใช้กดเอง
        if not refresh:
            cached = state.STORE.snapshot()["nodes"].get(name)
            if cached and not cached["stale"]:
                if cached["data"]:
                    return {"name": name, "reachable": True, "error": "",
                            "age_seconds": cached["age_seconds"], **cached["data"]}
                return {"name": name, "reachable": False, "error": cached["error"],
                        "age_seconds": cached["age_seconds"], "host": None, "models": []}
        state.STORE.mark_refreshing(name)
        try:
            info = probe(node)
        except NodeError as exc:
            update(name, last_error=str(exc)[:200])
            return {"name": name, "reachable": False, "error": str(exc), "host": None, "models": []}
        update(name, last_error="", lmds_version=(info.get("host") or {}).get("lmds_version", ""))
        return {"name": name, "reachable": True, "error": "", **info}

    @app.get("/api/scan", dependencies=guarded)
    def scan_weights(all_nodes: bool = False) -> dict:
        """weight ที่มีอยู่แล้วบนเครื่อง — อ่านอย่างเดียว ไม่ย้ายไม่ลบ"""
        import json as json_module

        from lmds.scanner import scan

        def rows(models) -> list[dict]:
            return [{"kind": m.kind, "name": m.name, "path": m.path, "size_gb": m.size_gb,
                     "shards": m.shard_count, "layout": m.layout} for m in models]

        payload = {"host": rows(scan())}
        if all_nodes:
            from lmds.nodes import NodeError, load, run as run_remote

            for node in load():
                try:
                    result = run_remote(node, "lmds scan --json", timeout=300)
                    payload[node.name] = json_module.loads(result.stdout).get("host", []) \
                        if result.ok else []
                except (NodeError, json_module.JSONDecodeError, ValueError):
                    payload[node.name] = []
        return payload

    @app.get("/api/recipes", dependencies=guarded)
    def recipes_list() -> dict:
        """สูตรที่รันผ่านจริง — สิ่งที่ใช้แทน LLM เมื่อเครื่องไม่มี provider"""
        from lmds.recipes import load_catalog

        return {"recipes": [
            {"match": r.match, "label": r.label, "engine": r.engine, "image": r.image,
             "serving": r.serving, "tools": r.tool_calling.get("parser"),
             "reasoning": r.reasoning.get("parser"), "notes": r.notes,
             "source": r.source, "validated_on": r.validated_on}
            for r in load_catalog()
        ]}

    @app.get("/api/cluster", dependencies=guarded)
    def cluster_view() -> dict:
        """เครื่องไหนจับคู่ stacked กันได้บ้าง — ต่อทุกเครื่องจริง จึงช้ากว่าหน้าอื่น

        เรียกเมื่อผู้ใช้กดเท่านั้น ไม่รวมอยู่ใน poll ปกติ
        """
        from lmds.inventory import host_payload
        from lmds.nodes import (
            NodeError, check_cluster_ip, cluster_groups, load, probe, stack_ready,
            suggest_cluster_ip,
        )

        def row(name, host, cluster_ip, is_self):
            # ส่งข้อมูลดิบอย่างเดียว — หน้าเว็บเป็นภาษาอังกฤษ จึงเรียบเรียงประโยคฝั่ง JS
            return {
                "name": name, "self": is_self, "reachable": True,
                "ready": stack_ready(host), "has_gpu": bool(host.get("gpus")),
                "fabric": host.get("fabric"), "cluster_ip": cluster_ip,
                "suggested_ip": suggest_cluster_ip(host),
                "ip": check_cluster_ip(host, cluster_ip),
            }

        # ต่อทุกเครื่องก่อนหนึ่งรอบ — ต้องรู้ว่าเครื่องอื่นอยู่วงไหนถึงจะเสนอ IP ของ hub ได้ถูก
        probed, unreachable = [], []
        for node in load():
            try:
                probed.append((node, (probe(node).get("host")) or {}))
            except NodeError as exc:
                unreachable.append((node, exc))

        local = host_payload()
        local_name = local.get("hostname") or "this machine"
        # hub เองไม่ได้อยู่ในทะเบียน จึงยังไม่มีที่เก็บ cluster IP ของตัวเอง — เสนอจากการ์ดที่ตรวจพบ
        # โดยเลือกวงที่ใช้ร่วมกับเครื่องอื่นได้ก่อน ไม่งั้นระบบจะฟ้อง split-fabric ที่ตัวเองสร้าง
        machines = [{"name": local_name, "host": local,
                     "cluster_ip": suggest_cluster_ip(local, [n.cluster_ip for n, _ in probed])}]
        rows = [row(local_name, local, machines[0]["cluster_ip"], True)]
        for node, host in probed:
            machines.append({"name": node.name, "host": host, "cluster_ip": node.cluster_ip})
            rows.append(row(node.name, host, node.cluster_ip, False))
        for node, exc in unreachable:
            rows.append({"name": node.name, "self": False, "reachable": False, "ready": False,
                         "has_gpu": False, "error": str(exc)[:200], "fabric": None,
                         "cluster_ip": node.cluster_ip, "suggested_ip": "",
                         "ip": {"state": "unset", "iface": "", "speed_gbps": None}})
        return {"machines": rows, "groups": cluster_groups(machines)}

    @app.patch("/api/nodes/{name}", dependencies=guarded)
    def node_patch(name: str, body: dict) -> dict:
        """แก้ค่าที่แก้ได้ของเครื่อง — ตอนนี้คือ cluster IP/interface และโน้ต"""
        from lmds.nodes import NodeError, find, update, validate_ssh_target

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        allowed = {"cluster_ip", "cluster_iface", "note", "alt_hosts"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail=f"ฟิลด์ที่แก้ไม่ได้: {', '.join(unknown)}")
        for key in ("cluster_ip", "cluster_iface", "note"):
            if key in body and not isinstance(body[key], str):
                raise HTTPException(status_code=400, detail=f"{key} ต้องเป็นข้อความ")
        changes = {k: body[k] for k in ("cluster_ip", "cluster_iface", "note") if k in body}
        if "alt_hosts" in body:
            # เครื่องเดียวกันเข้าได้หลายทาง (LAN ตอนอยู่ออฟฟิศ, Tailscale ตอนออกนอก)
            # รับได้ทั้ง list และสตริงคั่นจุลภาค เพราะช่องกรอกบนหน้าเว็บเป็นบรรทัดเดียว
            raw = body["alt_hosts"]
            if not isinstance(raw, (str, list)) or (
                isinstance(raw, list) and not all(isinstance(item, str) for item in raw)
            ):
                raise HTTPException(status_code=400, detail="alt_hosts ต้องเป็นข้อความหรือรายการข้อความ")
            items = raw if isinstance(raw, list) else raw.split(",")
            hosts = [h.strip() for h in items if h.strip()]
            try:
                for host in hosts:
                    validate_ssh_target(host, node.user, node.port)
            except NodeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            changes["alt_hosts"] = hosts
        if not changes:
            raise HTTPException(status_code=400, detail="ไม่มีฟิลด์ที่แก้ได้ในคำขอนี้")
        try:
            node = update(name, **changes)
        except NodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": node.name, "cluster_ip": node.cluster_ip,
                "cluster_iface": node.cluster_iface, "note": node.note,
                "alt_hosts": node.alt_hosts}

    @app.post("/api/nodes/{name}/install", dependencies=guarded)
    def node_install(name: str, body: dict | None = None) -> dict:
        """ติดตั้ง/อัปเดต LMDS บนเครื่องนั้นผ่าน SSH — เดิมทำได้แต่ทาง CLI

        ทุกเครื่องที่ hub คุมต้องมี `lmds` อยู่บนเครื่อง (hub ไม่ได้ส่ง agent ไปรัน แต่เรียก
        `lmds agent info` ผ่าน SSH) · เครื่องที่ยังไม่ได้ลงจึงขึ้นว่าติดต่อไม่ได้ตลอด
        และคนใช้หน้าเว็บอย่างเดียวก็ติดตรงนี้โดยไม่มีทางออก

        ใช้เวลาหลายนาที (git clone + pip install) จึงเป็น job ไม่ใช่ request ที่รอจนจบ
        """
        from lmds.nodes import NodeError, find, install_lmds, probe, update

        from . import jobs

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        unknown = sorted(set(body or {}) - {"with_prereq"})
        if unknown:
            raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")
        if "with_prereq" in (body or {}) and not isinstance(body["with_prereq"], bool):
            raise HTTPException(status_code=400, detail="with_prereq ต้องเป็น true หรือ false")
        # --with-prereq ต้องใช้ sudo แบบไม่ถามรหัสผ่าน ซึ่งหน้าเว็บไม่มี tty ให้กรอก
        # เปิดให้เลือกได้แต่บอกไว้ชัด ๆ ว่าเครื่องที่ sudo ถามรหัสจะค้างแล้ว timeout
        with_prereq = bool((body or {}).get("with_prereq"))

        def work() -> tuple[int, str]:
            result = install_lmds(node, with_prereq=with_prereq)
            output = (result.stdout or "") + (result.stderr or "")
            if not result.ok:
                return result.exit_code or 1, output
            try:
                info = probe(node)
            except NodeError as exc:
                return 1, f"{output}\nติดตั้งแล้วแต่ยังอ่านสถานะไม่ได้: {exc}\n"
            version = (info.get("host") or {}).get("lmds_version", "")
            update(name, lmds_version=version, last_error="")
            return 0, f"{output}\nพร้อมแล้ว — {name} รัน lmds {version}\n"

        try:
            job = jobs.start_task(f"node:{name}", "install", work)
        except jobs.JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.payload()

    @app.post("/api/nodes", dependencies=guarded)
    def node_add(body: dict) -> dict:
        """เพิ่มเครื่อง — รหัสผ่านใช้ครั้งเดียวเพื่อติดตั้ง key แล้วทิ้ง ไม่เขียนลงดิสก์"""
        from lmds.nodes import (
            Node, NodeError, add, check_login, ensure_key, install_key, load, probe, suggest_name,
            validate_node_name, validate_ssh_target,
        )

        allowed = {"host", "user", "name", "port", "password", "note"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail=f"ฟิลด์ที่ไม่รู้จัก: {', '.join(unknown)}")
        for key in ("host", "user", "name", "password", "note"):
            if key in body and not isinstance(body[key], str):
                raise HTTPException(status_code=400, detail=f"{key} ต้องเป็นข้อความ")
        host = (body.get("host") or "").strip()
        user = (body.get("user") or "").strip()
        if not host or not user:
            raise HTTPException(status_code=400, detail="ต้องระบุทั้ง host และ user")
        raw_port = body.get("port")
        if raw_port in (None, ""):
            raw_port = 22
        if isinstance(raw_port, bool):
            raise HTTPException(status_code=400, detail="port ต้องเป็นจำนวนเต็ม 1–65535")
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="port ต้องเป็นจำนวนเต็ม 1–65535") from None
        if not 1 <= port <= 65535:
            raise HTTPException(status_code=400, detail="port ต้องเป็นจำนวนเต็ม 1–65535")
        password = body.get("password") or ""

        try:
            host, user, port = validate_ssh_target(host, user, port)
            name = (body.get("name") or "").strip() or suggest_name(host, {n.name for n in load()})
            name = validate_node_name(name)
            ensure_key()
            node = Node(name=name, host=host, user=user, port=port, note=body.get("note") or "")
            if not check_login(host, user, port):
                if not password:
                    raise HTTPException(
                        status_code=422,
                        detail={"kind": "need-password",
                                "message": f"ยังเข้า {user}@{host} ด้วย key ไม่ได้ — ใส่รหัสผ่านครั้งเดียวเพื่อติดตั้ง key"},
                    )
                install_key(host, user, password, port)
                if not check_login(host, user, port):
                    raise HTTPException(status_code=422,
                                        detail={"kind": "key-failed", "message": "ติดตั้ง key แล้วแต่ยัง login ไม่ได้"})
            # ยังไม่ได้ติดตั้ง LMDS บนเครื่องนั้นก็เพิ่มได้ — key ติดตั้งไปแล้ว ค่อยไปลงทีหลังได้
            try:
                info = probe(node)
                reachable = True
            except NodeError as exc:
                info, reachable = {}, False
                node.last_error = str(exc)[:200]
            node.lmds_version = (info.get("host") or {}).get("lmds_version", "")
            node.last_seen = _timestamp() if reachable else ""
            add(node)
        except NodeError as exc:
            raise HTTPException(status_code=422, detail={"kind": "node", "message": str(exc)}) from exc
        finally:
            password = ""  # noqa: F841 — เคลียร์ทันที ไม่ให้ค้างในเฟรม
        return {"name": node.name, "reachable": reachable, "error": node.last_error, **info}

    @app.delete("/api/nodes/{name}", dependencies=guarded)
    def node_remove(name: str, body: dict | None = None) -> dict:
        from lmds.nodes import NodeError, remove

        unknown = sorted(set(body or {}) - {"confirm"})
        if unknown:
            raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")
        if (body or {}).get("confirm") != name:
            raise HTTPException(status_code=400, detail="ต้องยืนยันด้วยชื่อเครื่องที่จะเอาออก")
        try:
            node = remove(name)
        except NodeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"name": node.name, "removed": True}

    def _node_options(command: str, body: dict) -> list[str]:
        """แปลง option จากหน้าเว็บเป็น flag ของ controller — ตรวจค่าก่อนเสมอ

        ค่าพวกนี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นผ่าน SSH · ต่อให้ quote แล้วก็ยัง
        ต้องตรวจชนิดและช่วงที่นี่ ไม่ใช่ฝากไว้กับ JS ฝั่งเบราว์เซอร์ซึ่งใครก็ข้ามได้
        """
        if not body:
            return []
        if command == "remove":
            unknown = sorted(set(body) - {"confirm"})
            if unknown:
                raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")
            return []   # remove รับแค่ confirm ซึ่งจัดการแยกไปแล้ว
        if command not in {"start", "restart"}:
            raise HTTPException(status_code=400, detail=f"'{command}' ไม่รับ option (รับเฉพาะ start/restart)")
        unknown = sorted(set(body) - {"port", "context", "gpu_util"})
        if unknown:
            raise HTTPException(status_code=400, detail=f"ตัวเลือกที่ไม่รู้จัก: {', '.join(unknown)}")

        def number(key: str, low: float, high: float, integer: bool = True):
            raw = body.get(key)
            if raw in (None, ""):
                return None
            if isinstance(raw, bool):
                raise HTTPException(status_code=400, detail=f"{key} ต้องเป็นตัวเลข")
            try:
                value = int(raw) if integer else float(raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{key} ต้องเป็นตัวเลข") from None
            if integer and isinstance(raw, float) and not raw.is_integer():
                raise HTTPException(status_code=400, detail=f"{key} ต้องเป็นจำนวนเต็ม")
            if not low <= value <= high:
                raise HTTPException(status_code=400, detail=f"{key} ต้องอยู่ระหว่าง {low} ถึง {high}")
            return value

        flags: list[str] = []
        port = number("port", 1, 65535)
        if port is not None:
            flags += ["--port", str(port)]
        context = number("context", 256, 10_000_000)
        if context is not None:
            flags += ["--context", str(context)]
        # ช่วงเดียวกับที่ controller ตรวจเอง — ตรงกันจะได้ไม่มีค่าที่ผ่านที่นี่แล้วไปตายปลายทาง
        gpu_util = number("gpu_util", 0.3, 0.98, integer=False)
        if gpu_util is not None:
            flags += ["--gpu-util", f"{gpu_util:g}"]
        return flags

    @app.post("/api/nodes/{name}/models/{slug}/{command}", dependencies=guarded)
    def node_command(name: str, slug: str, command: str, body: dict | None = None) -> dict:
        """สั่งงานโมเดลบนเครื่องอื่น — ผ่าน CLI ของ node ตัวเดียวกับที่ผู้ใช้พิมพ์เอง"""
        from lmds.nodes import NodeError, find, run

        # allowlist ไม่ใช่พิธีกรรม — หน้าเว็บสั่งข้ามเครื่องได้ ทุกตัวที่เพิ่มต้องรันแบบ
        # ไม่โต้ตอบได้จริง (ssh ปิด stdin)
        allowed = {
            "start": "", "stop": "", "restart": "", "repair": "", "doctor": "",
            "logs": "-n 300", "enable": "", "disable": "",
            # remove ลบ weight หลายสิบ GB และกู้คืนไม่ได้ — หน้าเว็บต้องเรียก preview ก่อน
            # (`--dry-run` ไม่ลบอะไร) แล้วส่ง confirm ที่ตรงกับ slug กลับมาถึงจะลบจริง
            # ปุ่มเดียวจบไม่ได้ แต่ "ทำไม่ได้เลย" ก็ไม่ใช่คำตอบ — ผู้ใช้ต้องไป ssh เองอยู่ดี
            "remove": "--dry-run",
        }
        if command not in allowed:
            raise HTTPException(status_code=400, detail=f"คำสั่ง '{command}' ไม่อยู่ในรายการที่อนุญาต")
        if command == "remove" and (body or {}).get("confirm"):
            if (body or {}).get("confirm") != slug:
                raise HTTPException(status_code=400, detail="ชื่อยืนยันไม่ตรงกับโมเดลที่จะลบ")
            allowed = {**allowed, "remove": "-y"}
        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        parts = ["lmds", command, shlex.quote(slug)]
        if allowed[command]:
            parts.append(allowed[command])
        parts += _node_options(command, body or {})
        try:
            result = run(node, " ".join(parts), timeout=1800)
            state.STORE.force(name)   # สถานะเพิ่งเปลี่ยน — อย่าให้ผู้ใช้เห็นของเก่าอีก 15 วิ
        except NodeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"node": name, "slug": slug, "command": command,
                "exit_code": result.exit_code, "output": (result.stdout + result.stderr)[-8000:]}

    return app


def serve(host: str = "127.0.0.1", port: int = 8600, token: Optional[str] = None) -> None:
    import uvicorn

    uvicorn.run(create_app(token or ""), host=host, port=port, log_level="warning")
