"""รวบรวมข้อเท็จจริงให้ fit/sizing.py แล้วเขียนผลลง bundle — ฝั่งที่แตะเครื่องจริง

fit/sizing.py คำนวณล้วน · ไฟล์นี้หาของให้มัน: MODEL_PROFILE.yaml (weights · KV ต่อ token · native context) ·
bundle.env/bundle.args (slots/context/pin ที่ตั้งอยู่) · log ของ vLLM (ค่าที่โหลดจริง — ทำให้ตัวเลขแก้ตัวเองได้) ·
host (free -g / nvidia-smi) · โมเดลอื่นที่รันอยู่และถืออะไรเท่าไร (inventory.memory_by_slug)

ผู้เรียก:  `lmds fit` / `lmds set --fit` (CLI บนเครื่องนั้น) · POST /api/models/{slug}/fit (hub สำหรับโมเดลในเครื่อง) ·
          เครื่องอื่นผ่าน SSH → `lmds fit --json` บนเครื่องปลายทาง (ได้ log จริงของเครื่องนั้นฟรี)
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from lmds.fit.sizing import measured_from_log, plan_kv_pin, settings_for

# บรรทัดของ vLLM ที่ต้องใช้ — กรองตอนสตรีม ไม่โหลด log ทั้งก้อน (container ที่รันมาหลายวันมี log หลายร้อย MB)
_LOG_KEYS = ("Model loading took", "KV cache size", "Available KV cache memory", "memory for KV Cache",
             "Maximum concurrency", "Initial free memory")
_LOG_DEADLINE_S = 20.0


class FitError(ValueError):
    """คำนวณ/เขียนไม่ได้ด้วยเหตุที่ผู้ใช้แก้ได้ — ข้อความเป็นภาษาไทยพร้อมทางออก"""


def measured_for(server) -> dict:
    """ตัวเลขจริงจาก log ของ vLLM ตัวนี้ (docker) — ว่างเมื่อไม่ใช่ vLLM/ไม่มี container/อ่านไม่ได้"""
    if (server.engine or "") != "vllm" or not server.container or server.mode != "docker":
        return {}
    try:
        proc = subprocess.Popen(["docker", "logs", server.container], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
    except OSError:
        return {}
    keep: list[str] = []
    started = time.monotonic()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if any(k in line for k in _LOG_KEYS):
                keep.append(line.rstrip())
                if len(keep) > 60:
                    keep = keep[-60:]
            if time.monotonic() - started > _LOG_DEADLINE_S:
                proc.kill()
                break
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return measured_from_log("\n".join(keep))


def local_facts() -> tuple[dict, list[dict]]:
    """(host, models) ของเครื่องนี้ — รูปเดียวกับ `lmds agent info` แต่เบากว่า (ไม่อ่าน profile ทุก bundle)"""
    from lmds.fleet import discover
    from lmds.inventory import foreign_workloads, host_payload, memory_by_slug

    servers = discover()
    host = host_payload()
    if "foreign" not in host:
        host["foreign"] = foreign_workloads()
    held = memory_by_slug(servers)
    models = [{"slug": s.slug, "running": s.running, "engine": s.engine, "memory_gb": held.get(s.slug)}
              for s in servers]
    return host, models


def host_info_from(host: dict, models: list[dict], slug: str) -> dict:
    """แปลง payload ของ inventory เป็น host_info ของ plan_kv_pin — ทำงานกับทั้งเครื่องนี้และแคชของ node บน hub"""
    host = host or {}
    gpus = host.get("gpus") or []
    unified = host.get("memory_model") == "unified"
    vram_total = float(sum((g.get("vram_gb") or 0.0) for g in gpus)) or None
    vram_used = float(sum((g.get("vram_used_gb") or 0.0) for g in gpus))
    if unified and host.get("ram_total_gb"):
        total = float(host["ram_total_gb"])
        free = (total - float(host["ram_used_gb"])) if host.get("ram_used_gb") is not None else None
    else:
        total = vram_total
        free = (vram_total - vram_used) if vram_total else None
    others = []
    own = None
    for m in models or []:
        if not m.get("running"):
            continue
        if m.get("slug") == slug:
            own = m.get("memory_gb")
            continue
        if m.get("memory_gb"):
            others.append({"slug": m.get("slug"), "gb": round(float(m["memory_gb"]), 1)})
    for f in host.get("foreign") or []:
        if f.get("vram_mib"):
            others.append({"slug": f.get("name") or "foreign", "gb": round(float(f["vram_mib"]) / 1024.0, 1)})
    return {
        "memory_model": "unified" if unified else "discrete",
        "total_gb": total,
        "free_gb": free,
        "held_gb": vram_used,
        "others": others,
        "own_gb": own,
    }


def model_info_for(server, profile: dict | None, settings: dict, host_models: list[dict] | None = None,
                   *, with_logs: bool = True) -> dict:
    profile = profile or {}
    model = profile.get("model") or {}
    serving = profile.get("serving") or {}
    own = None
    for m in host_models or []:
        if m.get("slug") == server.slug:
            own = m.get("memory_gb")
    node_count = 1
    if (profile.get("topology") or "") == "stacked":
        try:
            from lmds.inventory import read_cluster_env

            node_count = int((read_cluster_env(server.controller) or {}).get("nnodes") or 2)
        except Exception:  # noqa: BLE001
            node_count = 2
    return {
        "slug": server.slug,
        "engine": (profile.get("runtime") or {}).get("engine") or server.engine or "vllm",
        "weight_bytes": model.get("weight_bytes"),
        "kv_bytes_per_token": model.get("kv_bytes_per_token"),
        "native_context": model.get("native_context"),
        "context": int(settings.get("context") or serving.get("context") or 0) or None,
        "slots": int(settings.get("slots") or serving.get("max_num_seqs") or 0) or None,
        "extra_args": settings.get("extra_args") or "",
        "running": bool(server.running),
        "own_gb": own,
        "measured": measured_for(server) if (with_logs and node_count == 1) else {},
        "node_count": node_count,
    }


def preview(server, slots: int | None, context: int | None, host: dict | None = None,
            models: list[dict] | None = None, *, with_logs: bool = True) -> dict:
    """ตาราง Fit ของ bundle นี้ — ไม่เขียนอะไร · host/models ไม่ส่งมา = อ่านจากเครื่องนี้"""
    from lmds.fleet import bundle_profile
    from lmds.fleet.bundle_settings import read

    if not server.controller:
        raise FitError(f"{server.slug} ไม่มี controller — fit ไม่ได้")
    bundle_dir = Path(server.controller).parent
    if host is None or models is None:
        host, models = local_facts()
    settings = read(bundle_dir)
    profile = bundle_profile(server.controller) or {}
    if not profile.get("model", {}).get("kv_bytes_per_token") and profile.get("model", {}).get("id"):
        # bundle เก่าไม่มี KV ต่อ token — ถาม Hub ครั้งเดียว (แคชใน web/memory.py) ถ้าไม่มี log ให้วัด
        try:
            from lmds.web.memory import kv_bytes_from_hub

            kv, _why = kv_bytes_from_hub(profile["model"]["id"], profile["model"].get("revision"))
            if kv:
                profile["model"]["kv_bytes_per_token"] = kv
        except Exception:  # noqa: BLE001 — ไม่มีเน็ตก็แค่ไม่มีค่านี้
            pass
    info = model_info_for(server, profile, settings, models, with_logs=with_logs)
    plan = plan_kv_pin(info, host_info_from(host, models, server.slug), slots=slots, context=context)
    plan["settings"] = settings_for(plan, settings.get("extra_args"))
    plan["current"] = {k: settings.get(k) for k in ("slots", "context", "gpu_util", "extra_args") if settings.get(k)}
    plan["bundle_dir"] = str(bundle_dir)
    return plan


def refusal(plan: dict) -> str | None:
    """เหตุผลที่ *ไม่* เขียนค่าให้ — None = เขียนได้"""
    if plan.get("fits") is None:
        return plan.get("reason") or "ยังคำนวณไม่ได้"
    if plan.get("fits") is False:
        return plan.get("reason") or "ไม่พอ"
    if plan.get("node_count", 1) > 1:
        return "bundle แบบ stacked — pin ต่อ rank ยังต้องตั้งเองผ่าน --extra-args"
    return None


def apply(server, plan: dict, extra: dict | None = None) -> dict[str, str]:
    """เขียน slots/context (+ gpu_util + --kv-cache-memory สำหรับ vLLM) ลง bundle เหมือน `lmds set`"""
    from lmds.fleet.bundle_settings import SettingsError, ensure_controller_reads, read, write

    why = refusal(plan)
    if why:
        raise FitError(why)
    bundle_dir = Path(server.controller).parent
    ensure_controller_reads(Path(server.controller))
    merged = {**read(bundle_dir), **{k: str(v) for k, v in (extra or {}).items() if v is not None}, **plan["settings"]}
    try:
        return write(bundle_dir, merged)
    except SettingsError as exc:
        raise FitError(str(exc)) from exc
