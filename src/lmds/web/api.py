"""Web UI — หน้าเดียวสำหรับคุม fleet + doctor (เฟส 2)

ชั้นนี้ไม่มี logic ของตัวเอง: เรียก core เดิมทั้งหมด (hardware / fleet / doctor)
แล้วแปลงเป็น JSON เท่านั้น — อะไรที่ CLI ทำได้ เว็บต้องได้ผลเหมือนกันเป๊ะ

ความปลอดภัย (ตาม PRD §9): หน้านี้สั่ง start/stop โมเดลได้ จึง
- bind 127.0.0.1 เป็นค่าเริ่มต้น — ต้องตั้งใจเปิดออก network เอง
- ตั้ง token ได้ และถ้า bind ออก network โดยไม่มี token จะเตือนเสียงดัง
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

import lmds

STATIC = Path(__file__).parent / "static"


def _host_payload() -> dict:
    from lmds.fit.targets import from_hardware_report
    from lmds.hardware import probe
    from lmds.hardware.profiler import host_summary

    report = probe()
    summary = host_summary()
    target = from_hardware_report(report)
    return {
        "hostname": summary.hostname,
        "ip": summary.ip,
        "arch": report.arch,
        "profile": report.profile.value,
        "ram_used_gb": summary.ram_used_gb,
        "ram_total_gb": summary.ram_total_gb,
        "disk_free_gb": report.disk_free_gb,
        "disk_total_gb": report.disk_total_gb,
        "docker": report.docker,
        "toolkit": report.nvidia_container_toolkit,
        # unified (Spark) ต้องแสดง memory คนละแบบกับ discrete (RTX) — ดู mockup
        "memory_model": target.memory_model.value if target else None,
        "gpus": [
            {
                "name": gpu.name,
                "vram_gb": round(gpu.vram_mib / 1024, 1) if gpu.vram_mib
                else (gpu.known.vram_gb if gpu.known else None),
                "compute": gpu.compute_capability,
                "tested": gpu.tested,
            }
            for gpu in report.gpus
        ],
    }


def _weights_present(server, profile) -> bool:
    """โหลด weight มาแล้วหรือยัง — ใช้ตัวตรวจชุดเดียวกับ lmds doctor ไม่คำนวณซ้ำคนละทาง"""
    from lmds.doctor.checks import _weight_paths

    if not profile:
        return False
    directory, wanted = _weight_paths(profile, server.slug)
    if not directory.is_dir():
        return False
    return all((directory / name).exists() for name in wanted)


def _active_job(slug: str) -> dict | None:
    from . import jobs

    job = jobs.active_for(slug)
    return {"id": job.id, "command": job.command} if job else None


def _model_payload(server) -> dict:
    from lmds.fleet import autostart_status, bundle_profile, feature_summary, profile_context

    profile = bundle_profile(server.controller)
    return {
        "slug": server.slug,
        "model_id": server.model_id or server.model,
        "engine": server.engine,
        "mode": server.mode,
        "port": server.port,
        "running": server.running,
        "healthy": server.healthy,
        "registered": server.registered,
        "external": server.external,
        "controller_exists": server.controller_exists,
        "endpoint": server.endpoint,
        "context": profile_context(profile),
        "features": feature_summary(profile),
        "autostart": autostart_status(server.slug),
        "topology": (profile or {}).get("topology"),
        "max_num_seqs": ((profile or {}).get("serving") or {}).get("max_num_seqs"),
        "started_at": server.started_at,
        "downloaded": _weights_present(server, profile),
        "job": _active_job(server.slug),
    }


def create_app(token: str = "") -> FastAPI:
    app = FastAPI(title="LMDS", docs_url=None, redoc_url=None, openapi_url=None)

    def require_token(request: Request) -> None:
        if not token:
            return
        supplied = request.headers.get("x-lmds-token") or request.query_params.get("token", "")
        # compare_digest กัน timing attack — เทียบสตริงตรง ๆ รั่วความยาวและ prefix
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="token ไม่ถูกต้อง")

    guarded = [Depends(require_token)]

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/version", dependencies=guarded)
    def version() -> dict:
        return {"version": lmds.__version__}

    @app.get("/api/host", dependencies=guarded)
    def host() -> dict:
        return _host_payload()

    @app.get("/api/models", dependencies=guarded)
    def models() -> dict:
        from lmds.fleet import discover

        return {"models": [_model_payload(s) for s in discover()]}

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
        return _action(slug, "start", body)

    @app.post("/api/models/{slug}/stop", dependencies=guarded)
    def stop(slug: str, body: dict | None = None) -> JSONResponse:
        return _action(slug, "stop", body)

    @app.post("/api/models/{slug}/restart", dependencies=guarded)
    def restart(slug: str, body: dict | None = None) -> JSONResponse:
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
        from lmds.fleet import find, remove_server

        server = find(slug)
        if server is None:
            raise HTTPException(status_code=404, detail=f"ไม่รู้จัก {slug}")
        keep = bool((body or {}).get("keep_weights"))
        return {"slug": slug, "done": remove_server(server, include_weights=not keep)}

    @app.post("/api/models/{slug}/autostart", dependencies=guarded)
    def autostart(slug: str, body: dict | None = None) -> dict:
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

    return app


def serve(host: str = "127.0.0.1", port: int = 8600, token: Optional[str] = None) -> None:
    import uvicorn

    uvicorn.run(create_app(token or ""), host=host, port=port, log_level="warning")
