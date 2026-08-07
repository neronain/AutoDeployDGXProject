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
import time
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


# กันเดา token: token สั้นสุด 8 ตัวที่ผู้ใช้ตั้งเองอาจเป็นคำที่เดาได้ ถ้ายิงได้ไม่จำกัด
# บอตในวง network เดาจนเจอได้ · หน่วงเป็นขั้นตามจำนวนครั้งที่ผิดติดกันจาก IP เดียวกัน
_FAIL_WINDOW = 300.0      # ลืมความผิดพลาดหลังเงียบไป 5 นาที
_FAIL_FREE = 5            # ผิดได้เท่านี้ก่อนโดนหน่วง (พิมพ์ผิดจริง ๆ ไม่ควรโดนลงโทษ)
_FAIL_LOCK_MAX = 60.0     # หน่วงสูงสุดต่อครั้ง


class _Attempts:
    """นับความพยายามที่ผิดต่อ IP — อยู่ในหน่วยความจำของ process เดียว พอสำหรับงานนี้"""

    def __init__(self) -> None:
        self._by_ip: dict[str, tuple[int, float]] = {}

    def locked_for(self, ip: str) -> float:
        count, last = self._by_ip.get(ip, (0, 0.0))
        if count <= _FAIL_FREE or time.time() - last > _FAIL_WINDOW:
            return 0.0
        wait = min(2.0 ** (count - _FAIL_FREE), _FAIL_LOCK_MAX)
        remaining = wait - (time.time() - last)
        return max(0.0, remaining)

    def failed(self, ip: str) -> None:
        count, last = self._by_ip.get(ip, (0, 0.0))
        if time.time() - last > _FAIL_WINDOW:
            count = 0
        self._by_ip[ip] = (count + 1, time.time())

    def passed(self, ip: str) -> None:
        self._by_ip.pop(ip, None)


def create_app(token: str = "") -> FastAPI:
    app = FastAPI(title="LMDS", docs_url=None, redoc_url=None, openapi_url=None)
    attempts = _Attempts()

    def _client_ip(request: Request) -> str:
        return request.client.host if request.client else "?"

    def require_token(request: Request) -> None:
        if not token:
            return
        ip = _client_ip(request)
        wait = attempts.locked_for(ip)
        if wait:
            raise HTTPException(status_code=429, detail=f"ผิดหลายครั้งเกินไป — รออีก {wait:.0f} วินาที")
        supplied = request.headers.get("x-lmds-token") or request.query_params.get("token", "")
        # compare_digest กัน timing attack — เทียบสตริงตรง ๆ รั่วความยาวและ prefix
        if not secrets.compare_digest(supplied, token):
            attempts.failed(ip)
            raise HTTPException(status_code=401, detail="token ไม่ถูกต้อง")
        attempts.passed(ip)

    guarded = [Depends(require_token)]

    @app.get("/api/auth")
    def auth_mode() -> dict:
        """หน้าเว็บถามก่อนวาดว่าเครื่องนี้ต้อง token ไหม — bind 127.0.0.1 ไม่ต้อง"""
        return {"required": bool(token)}

    @app.post("/api/auth")
    def auth_check(request: Request, _: None = Depends(require_token)) -> dict:
        """ตรวจ token ที่กรอกในหน้า login — ผ่านแล้วเบราว์เซอร์ถึงจะเก็บไว้ใช้ต่อ"""
        return {"ok": True}

    # ตัวเดียวที่คุยกับ node จริง — endpoint ทุกตัวอ่านจากแคชที่มันเติมให้ จึงตอบทันทีเสมอ
    state.start_refresher()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

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
        import os

        from lmds.fleet import FleetError, find, restart_server, start_server, stop_server

        from . import jobs

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        runner = {"start": start_server, "stop": stop_server, "restart": restart_server}[verb]
        # ตัวเลือก port/API key/context ส่งผ่าน env เหมือนที่ผู้ใช้พิมพ์หน้าคำสั่งบน CLI
        saved = {k: os.environ.get(k) for k in jobs.controller_env(options)}
        os.environ.update(jobs.controller_env(options))
        try:
            outcome = runner(server)
        except FleetError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
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
    def _suggest_target(host: dict | None) -> str:
        """เดา target preset จากฮาร์ดแวร์ที่ตรวจพบของเครื่องนั้น

        deploy สำหรับเครื่องอื่นต้องระบุ target เอง (เครื่องนี้ตรวจตัวเองไม่ได้แทนเขา) —
        ถ้าไม่แนะนำให้ ผู้ใช้ต้องรู้เองว่าเครื่องปลายทางคือ preset ไหน ซึ่งเดาผิดแล้ว
        แผนที่ได้จะคำนวณ context จากหน่วยความจำที่ไม่ใช่ของจริง
        """
        host = host or {}
        gpus = host.get("gpus") or []
        if not gpus:
            return ""
        if host.get("memory_model") == "unified":
            return "dgx-spark-single"
        name = (gpus[0].get("name") or "").lower().replace("nvidia ", "").strip()
        from lmds.fit.targets import PRESETS

        slug = name.replace("geforce ", "").replace(" ", "-")
        for preset in PRESETS:
            if preset == slug or slug.endswith(preset):
                return preset
        return ""

    @app.get("/api/provider", dependencies=guarded)
    def provider_get() -> dict:
        """LLM ที่ใช้เป็นสมองของระบบตอนนี้ — key ถูก mask เสมอ ไม่ส่งค่าจริงออกไป"""
        from lmds.config import DEFAULT_MODELS, ProviderName, Settings
        from lmds.secrets import get_secret

        settings = Settings.load()
        provider = settings.provider
        key = get_secret(provider.name.value) if provider else ""
        return {
            "configured": provider is not None,
            "name": provider.name.value if provider else "",
            "model": provider.model if provider else "",
            "base_url": provider.base_url if provider else "",
            # ไม่ส่ง key จริงกลับไปเด็ดขาด — บอกแค่ว่ามีหรือยัง และท้าย 4 ตัวไว้ยืนยันว่าใช่ตัวที่คิด
            "has_key": bool(key),
            "key_hint": f"…{key[-4:]}" if key else "",
            "choices": [n.value for n in ProviderName],
            "defaults": {n.value: DEFAULT_MODELS[n] for n in ProviderName},
        }

    @app.put("/api/provider", dependencies=guarded)
    def provider_set(body: dict) -> dict:
        """ตั้ง provider / model / base URL / API key จากหน้าเว็บ

        เดิมต้องกลับไป CLI ทุกครั้ง ทั้งที่หน้าเว็บคือที่ที่ผู้ใช้เห็นว่าแผนออกมาไม่ดี
        """
        from lmds.config import ProviderName, Settings
        from lmds.secrets import set_secret

        name = (body.get("name") or "").strip()
        try:
            provider_name = ProviderName(name)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"ไม่รู้จัก provider '{name}'") from None
        settings = Settings.load()
        try:
            settings.set_provider(provider_name, (body.get("model") or "").strip(),
                                  (body.get("base_url") or "").strip() or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        settings.save()
        key = (body.get("api_key") or "").strip()
        if key:
            set_secret(provider_name.value, key)
        return provider_get()

    @app.post("/api/provider/models", dependencies=guarded)
    def provider_models(body: dict) -> dict:
        """ถาม provider ว่า key นี้ใช้โมเดลอะไรได้บ้าง

        ผู้ใช้ที่ไม่ได้อยู่กับ provider นั้นทุกวันไม่มีทางรู้ชื่อโมเดล — พิมพ์ผิดตัวเดียว
        แล้วรู้ตอน deploy ล้มกลางทาง · key ที่ยังไม่ได้บันทึกก็ลองได้ (ส่งมากับ request)
        """
        from lmds.brain.providers import ProviderError, list_models
        from lmds.config import ProviderName
        from lmds.secrets import get_secret

        try:
            name = ProviderName((body.get("name") or "").strip())
        except ValueError:
            raise HTTPException(status_code=400, detail="ไม่รู้จัก provider นี้") from None
        # ยังไม่กรอก key ใหม่ = ใช้ตัวที่บันทึกไว้ · ไม่มีเลยก็ยังลองได้ (endpoint ในวงมักไม่ต้องใช้)
        key = (body.get("api_key") or "").strip() or get_secret(name.value)
        try:
            models = list_models(name, key, (body.get("base_url") or "").strip() or None)
        except ProviderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"models": models}

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

    @app.post("/api/nodes/{name}/models/{slug}/ctl/{command}", dependencies=guarded)
    def node_controller_command(name: str, slug: str, command: str) -> dict:
        """สั่ง *คำสั่งของ controller* บนเครื่องอื่น — ชุดทดสอบ/ข้อมูล ที่ `lmds` ไม่ได้ห่อไว้

        ปุ่มบนหน้าเว็บขึ้นตาม `commands` ที่ bundle นั้นรองรับจริงอยู่แล้ว แต่ allowlist
        ที่นี่กันไว้อีกชั้น: รับเฉพาะคำสั่งที่ **อ่านอย่างเดียวหรือทดสอบ** ไม่ใช่ทุกอย่าง
        ที่ dispatch table มี (download/start/stop มีทางของมันเองที่จัดการ option แล้ว)
        """
        from lmds.nodes import NodeError, find, run

        allowed = {
            "test-text", "test-vision", "test-reasoning", "test-tools",
            "bench", "stress", "client-config", "network-info", "status", "props",
            "verify-files", "prepare-runtime", "sync-worker", "verify-worker", "clear-fi-cache",
        }
        if command not in allowed:
            raise HTTPException(status_code=400, detail=f"คำสั่ง '{command}' ไม่อยู่ในรายการที่อนุญาต")
        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        quoted = shlex.quote(slug)
        script = (
            f"dir=\"$(ls -d ~/bundles/{quoted} ~/*/bundles/{quoted} 2>/dev/null | head -1)\"; "
            f"[ -n \"$dir\" ] || {{ echo 'ไม่พบ bundle {slug} บน {name}' >&2; exit 1; }}; "
            f"cd \"$dir\" || exit 1; "
            f"ctl=\"$(ls ./*-single.sh ./*-stacked.sh 2>/dev/null | head -1)\"; "
            f"[ -n \"$ctl\" ] || {{ echo 'ไม่พบ controller' >&2; exit 1; }}; "
            f"\"$ctl\" {shlex.quote(command)}"
        )
        try:
            result = run(node, script, timeout=3600)
        except NodeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"node": name, "slug": slug, "command": command,
                "exit_code": result.exit_code, "output": (result.stdout + result.stderr)[-8000:]}

    @app.post("/api/models/{slug}/push/{name}", dependencies=guarded)
    def push_bundle(slug: str, name: str) -> dict:
        """ส่ง bundle ที่สร้าง+ตรวจแผนไว้ในเครื่องนี้ ไปติดตั้งบนเครื่องอื่น

        ทำไมส่ง ZIP แทนการสั่งให้ปลายทาง `lmds deploy` เอง: ผู้ใช้ตรวจแผนและอนุมัติ flag
        ไปแล้วบน bundle *ตัวนี้* — ให้ปลายทางวางแผนใหม่เองอาจได้คนละค่า กลายเป็นอนุมัติ
        แผนหนึ่งแล้วได้อีกแผนหนึ่งไปรัน
        """
        from lmds.fleet import bundle_roots
        from lmds.nodes import NodeError, find, push_file, run

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        archive = next((r / f"{slug}.zip" for r in bundle_roots() if (r / f"{slug}.zip").is_file()), None)
        if archive is None:
            raise HTTPException(status_code=404, detail=f"ไม่พบ {slug}.zip ในเครื่องนี้")
        try:
            sent = push_file(node, str(archive), f"/tmp/{slug}.zip")
            if not sent.ok:
                raise HTTPException(status_code=409, detail=(sent.stderr or "ส่งไฟล์ไม่สำเร็จ")[:300])
            quoted = shlex.quote(slug)
            unpacked = run(node, (
                f"mkdir -p ~/bundles && cd ~/bundles && unzip -oq /tmp/{quoted}.zip && "
                f"rm -f /tmp/{quoted}.zip && chmod +x ~/bundles/{quoted}/*.sh 2>/dev/null; "
                f"ls -d ~/bundles/{quoted}"
            ), timeout=300)
        except NodeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not unpacked.ok:
            raise HTTPException(status_code=409, detail=(unpacked.stderr or "แตกไฟล์ไม่สำเร็จ")[:300])
        state.STORE.force(name)
        return {"node": name, "slug": slug, "path": unpacked.stdout.strip(),
                "size_mb": round(archive.stat().st_size / 1024**2, 1)}

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
        keep = bool((body or {}).get("keep_weights"))
        from lmds.fleet import removal_failed

        lines = remove_server(server, include_weights=not keep)
        return {"slug": slug, "done": lines, "failed": removal_failed(lines)}

    @app.post("/api/models/{slug}/autostart", dependencies=guarded)
    def autostart(slug: str, body: dict | None = None) -> dict:
        state.STORE.invalidate_local()
        from lmds.fleet import FleetError, disable_autostart, enable_autostart, find

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        enabled = bool((body or {}).get("enabled"))
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
             # deploy สำหรับเครื่องอื่นต้องระบุ target เอง — แนะนำจากฮาร์ดแวร์ที่เคยตรวจไว้
             "suggested_target": _node_target_hint(n.name)}
            for n in load()
        ]}

    @app.post("/api/nodes/{name}/install", dependencies=guarded)
    def node_install(name: str, body: dict | None = None) -> dict:
        """ติดตั้ง/อัปเดต LMDS บนเครื่องนั้นจากหน้าเว็บ

        เครื่องที่เพิ่งเพิ่มเข้ามามักยังไม่มี LMDS — เดิมหน้าเว็บบอกแค่ว่าติดต่อไม่ได้
        พร้อมคำสั่งให้ไป ssh ทำเอง ทั้งที่ hub ต่อ SSH ได้อยู่แล้วและ CLI ก็มี
        `lmds node install` มาตลอด
        """
        from lmds.nodes import find, install_script

        from . import jobs

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        # prerequisite (docker/toolkit) ต้องใช้ sudo ซึ่งไม่มี tty ให้กรอกรหัส — ค่าเริ่มต้นจึงข้าม
        script = install_script(with_prereq=bool((body or {}).get("with_prereq")))
        try:
            job = jobs.start_remote(name, "_install", "install", script)
        except jobs.JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"node": name, "job": job.payload()}

    @app.post("/api/nodes/{name}/setup", dependencies=guarded)
    def node_setup(name: str, body: dict) -> dict:
        """ตั้งค่าที่ต้องใช้ root บนเครื่องนั้น — รหัสผ่านมากับ request นี้ ใช้ครั้งเดียว

        **ไม่เก็บรหัสผ่าน**: ไม่เขียนลงดิสก์ ไม่ใส่ใน argv (ส่งทาง stdin ของ ssh)
        ไม่ log และทะเบียน node ไม่มีฟิลด์ให้เก็บ · ผู้ใช้ต้องกรอกใหม่ทุกครั้งที่ต้องใช้
        """
        from lmds.nodes import NodeError, find, run_privileged

        node = find(name)
        if node is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        password = (body or {}).get("password") or ""
        if not password:
            raise HTTPException(status_code=400, detail="ต้องใส่รหัสผ่าน sudo ของ user บนเครื่องนั้น")
        try:
            outcomes = run_privileged(node, password, with_prereq=bool((body or {}).get("with_prereq")))
        except NodeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state.STORE.force(name)
        return {"node": name, "steps": outcomes,
                "ok": all(step["ok"] for step in outcomes)}

    def _attach_node_jobs(name: str, payload: dict) -> dict:
        """แปะงานที่กำลังรันของแต่ละโมเดลลงไปใน payload — หน้าเว็บจะได้ตามต่อได้หลังรีเฟรช"""
        from . import jobs

        for model in payload.get("models") or []:
            job = jobs.active_for(model.get("slug", ""), name)
            if job is not None:
                model["job"] = job.payload()
        return payload

    def _node_target_hint(name: str) -> str:
        # เครื่องที่ติดต่อไม่ได้มี entry อยู่แต่ data เป็น None — `.get("data", {})` ไม่ช่วย
        # เพราะ default ใช้เฉพาะตอน "ไม่มีคีย์" ไม่ใช่ตอนค่าเป็น None · เคสนี้ทำให้ /api/nodes
        # ตอบ 500 แล้วทั้งส่วน Other machines หายไปทั้งก้อน
        cached = state.STORE.snapshot()["nodes"].get(name) or {}
        return _suggest_target((cached.get("data") or {}).get("host"))

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
                    return _attach_node_jobs(name, {
                        "name": name, "reachable": True, "error": "",
                        "age_seconds": cached["age_seconds"], **cached["data"]})
                return {"name": name, "reachable": False, "error": cached["error"],
                        "age_seconds": cached["age_seconds"], "host": None, "models": []}
        state.STORE.mark_refreshing(name)
        try:
            info = probe(node)
        except NodeError as exc:
            update(name, last_error=str(exc)[:200])
            return {"name": name, "reachable": False, "error": str(exc), "host": None, "models": []}
        update(name, last_error="", lmds_version=(info.get("host") or {}).get("lmds_version", ""))
        return _attach_node_jobs(name, {"name": name, "reachable": True, "error": "", **info})

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
        from lmds.config import Settings
        from lmds.inventory import host_payload
        from lmds.nodes import (
            NodeError, check_cluster_ip, cluster_groups, load, probe, stack_ready,
            suggest_cluster_ip,
        )

        def row(name, host, cluster_ip, is_self, stack):
            # ส่งข้อมูลดิบอย่างเดียว — หน้าเว็บเป็นภาษาอังกฤษ จึงเรียบเรียงประโยคฝั่ง JS
            return {
                "name": name, "self": is_self, "reachable": True,
                # ผู้ใช้สั่งไว้ว่าเครื่องนี้เอาไปจับกลุ่มได้ไหม — ต่างจาก ready ที่มาจากการตรวจ
                "stack": stack,
                # ชื่อจริงของเครื่อง — ต่างจากชื่อในทะเบียน จึงเป็นทางเดียวที่ผู้ใช้จะเห็นว่า
                # สองรายการนี้คือเครื่องเดียวกันที่ถูกเพิ่มไว้สองชื่อ
                "hostname": host.get("hostname") or "",
                "ready": stack_ready(host), "has_gpu": bool(host.get("gpus")),
                "fabric": host.get("fabric"), "cluster_ip": cluster_ip,
                "suggested_ip": suggest_cluster_ip(host),
                "ip": check_cluster_ip(host, cluster_ip),
            }

        local = host_payload()
        local_name = local.get("hostname") or "this machine"
        # hub เองไม่ได้อยู่ในทะเบียน จึงยังไม่มีที่เก็บ cluster IP ของตัวเอง — เสนอจากการ์ดที่ตรวจพบ
        # ส่วน "เอาเข้ากลุ่มไหม" เก็บใน config.yaml (ทะเบียนไม่มีแถวของ hub ให้เก็บ)
        stack_self = Settings.load().cluster.stack_self
        machines = [{"name": local_name, "host": local, "cluster_ip": suggest_cluster_ip(local),
                     "stack": stack_self}]
        rows = [row(local_name, local, machines[0]["cluster_ip"], True, stack_self)]
        for node in load():
            try:
                host = (probe(node).get("host")) or {}
            except NodeError as exc:
                rows.append({"name": node.name, "self": False, "reachable": False, "ready": False,
                             "hostname": "", "has_gpu": False, "error": str(exc)[:200], "fabric": None,
                             "stack": node.stack, "cluster_ip": node.cluster_ip, "suggested_ip": "",
                             "ip": {"state": "unset", "iface": "", "speed_gbps": None}})
                continue
            machines.append({"name": node.name, "host": host, "cluster_ip": node.cluster_ip,
                             "stack": node.stack})
            rows.append(row(node.name, host, node.cluster_ip, False, node.stack))
        return {"machines": rows, "groups": cluster_groups(machines)}

    @app.patch("/api/cluster/self", dependencies=guarded)
    def cluster_self_patch(body: dict) -> dict:
        """เปิด/ปิดการเอา hub เองเข้ากลุ่ม stacked — node อื่นใช้ PATCH /api/nodes/{name}"""
        from lmds.config import Settings

        if "stack" not in body:
            raise HTTPException(status_code=400, detail="ต้องระบุฟิลด์ stack (true/false)")
        settings = Settings.load()
        settings.cluster.stack_self = bool(body["stack"])
        settings.save()
        return {"stack": settings.cluster.stack_self}

    @app.patch("/api/nodes/{name}", dependencies=guarded)
    def node_patch(name: str, body: dict) -> dict:
        """แก้ค่าที่แก้ได้ของเครื่อง — ตอนนี้คือ cluster IP/interface และโน้ต"""
        from lmds.nodes import NodeError, find, update

        if find(name) is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จักเครื่อง {name}")
        changes = {k: body[k] for k in ("cluster_ip", "cluster_iface", "note", "stack") if k in body}
        if not changes:
            raise HTTPException(status_code=400, detail="ไม่มีฟิลด์ที่แก้ได้ในคำขอนี้")
        try:
            node = update(name, **changes)
        except NodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": node.name, "cluster_ip": node.cluster_ip,
                "cluster_iface": node.cluster_iface, "note": node.note, "stack": node.stack}

    @app.post("/api/nodes", dependencies=guarded)
    def node_add(body: dict) -> dict:
        """เพิ่มเครื่อง — รหัสผ่านใช้ครั้งเดียวเพื่อติดตั้ง key แล้วทิ้ง ไม่เขียนลงดิสก์"""
        from lmds.nodes import (
            Node, NodeError, add, check_login, ensure_key, install_key, load, probe, suggest_name,
        )

        host = (body.get("host") or "").strip()
        user = (body.get("user") or "").strip()
        if not host or not user:
            raise HTTPException(status_code=400, detail="ต้องระบุทั้ง host และ user")
        port = int(body.get("port") or 22)
        password = body.get("password") or ""

        try:
            ensure_key()
            name = (body.get("name") or "").strip() or suggest_name(host, {n.name for n in load()})
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
    def node_remove(name: str) -> dict:
        from lmds.nodes import NodeError, remove

        try:
            node = remove(name)
        except NodeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"name": node.name, "removed": True}

    def _node_env(command: str, body: dict) -> str:
        """แปลง option จากหน้าเว็บเป็น env ที่ controller อ่าน — **ชุดเดียวกับโมเดลในเครื่องนี้**

        เดิม node ใช้ flag ส่วน local ใช้ env ทำให้ตั้งค่าได้ไม่เท่ากัน (node ไม่มี slots,
        bind, API key) ทั้งที่ controller ตัวเดียวกันรับได้หมด — ผู้ใช้เห็นสองหน้าจอที่
        ทำงานคนละอย่างทั้งที่ควรเหมือนกัน
        """
        if not body:
            return ""
        if command not in {"start", "restart"}:
            raise HTTPException(status_code=400, detail=f"'{command}' ไม่รับ option (รับเฉพาะ start/restart)")
        from . import jobs

        try:
            env = jobs.controller_env(jobs.clean_options(body))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

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
        env = _node_env(command, {k: v for k, v in (body or {}).items() if k != "confirm"})
        parts = ([env] if env else []) + ["lmds", command, shlex.quote(slug)]
        if allowed[command]:
            parts.append(allowed[command])
        remote = " ".join(parts)

        from . import jobs

        # งานยาว (download หลายสิบ GB, start ที่โหลดโมเดลเป็นนาที) ตอบกลับเป็น job แล้วสตรีมผล
        # — รอใน request เดียวคือให้ผู้ใช้มองหน้าค้างโดยไม่รู้ว่าคืบหน้าหรือตายไปแล้ว
        # `remove --dry-run` ไม่นับว่ายาว: มันแค่คิดขนาดไฟล์แล้วจบ
        long_running = command in jobs.REMOTE_LONG and not (
            command == "remove" and not (body or {}).get("confirm"))
        if long_running:
            try:
                return {"node": name, "slug": slug, "command": command,
                        "job": jobs.start_remote(name, slug, command, remote).payload()}
            except jobs.JobError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            result = run(node, remote, timeout=1800)
            state.STORE.force(name)   # สถานะเพิ่งเปลี่ยน — อย่าให้ผู้ใช้เห็นของเก่าอีก 15 วิ
        except NodeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"node": name, "slug": slug, "command": command,
                "exit_code": result.exit_code, "output": (result.stdout + result.stderr)[-8000:]}

    return app


def serve(host: str = "127.0.0.1", port: int = 8600, token: Optional[str] = None) -> None:
    import uvicorn

    uvicorn.run(create_app(token or ""), host=host, port=port, log_level="warning")
