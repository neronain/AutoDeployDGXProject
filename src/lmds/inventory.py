"""สรุปสถานะเครื่องนี้เป็น JSON — ใช้ร่วมกันระหว่างหน้าเว็บกับ `lmds agent info`

hub อ่านข้อมูลของ node ผ่าน SSH โดยเรียก `lmds agent info` ไม่ใช่ยิง HTTP เข้าไป
node จึงไม่ต้องรัน daemon อะไรเลย และไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22
"""

from __future__ import annotations

import re
from pathlib import Path

# คำสั่งที่ controller รู้จัก — ใช้กรองผลจาก dispatch table ไม่ให้ help/-h หลุดมาเป็นปุ่ม
KNOWN_COMMANDS = {
    "prepare-runtime", "download", "verify-files", "start", "stop", "restart", "status",
    "logs", "client-config", "network-info", "test-text", "test-vision", "test-reasoning",
    "test-tools", "bench", "stress", "props", "info", "wait-health", "doctor",
    "sync-worker", "verify-worker", "clear-fi-cache", "repair",
}
_COMMAND_RE = re.compile(r"(?m)^\s{2}([a-z][a-z-]*)\)")


def controller_commands(controller: str) -> list[str]:
    """คำสั่งที่ controller ตัวนี้รองรับจริง — อ่านจาก dispatch table ของสคริปต์เอง

    bundle เก่าไม่มีคำสั่งใหม่ ๆ (เช่น test-vision) การเดาจาก profile ทำให้ปุ่มขึ้นแล้วกดล้ม
    """
    try:
        text = Path(controller).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return sorted({m for m in _COMMAND_RE.findall(text) if m in KNOWN_COMMANDS})


def self_managed_weights(profile) -> bool:
    """bundle นี้เป็นแบบที่ผู้ใช้ดูแล weight เอง ไม่ใช่ของที่ LMDS โหลดมาไหม

    bundle ที่มาจาก `lmds adopt` ชี้ไปที่ path *ในคอนเทนเนอร์* (เช่น /models/xxx) ซึ่งบน
    เครื่องโฮสต์ไม่มีอยู่จริง — เอาไปคิดเป็น repo id ของ Hugging Face ไม่ได้ ผลคือหน้าเว็บ
    ขึ้น "not downloaded" ตลอดกาลและยื่นปุ่ม download ที่กดไปก็ล้มแน่นอน (เจอกับ
    qwen3-coder-next-nvfp4-gb10 บน msi-6) · adopt ตั้งใจไม่รองรับ download อยู่แล้ว
    """
    model = (profile or {}).get("model") or {}
    identifier = str(model.get("id") or "")
    # repo id ของ HF เป็น "org/name" เสมอ — ขึ้นต้นด้วย / หรือ ~ คือ path ไม่ใช่ repo
    return bool((profile or {}).get("adopted")) or identifier.startswith(("/", "~", "./"))


def weights_present(server, profile) -> bool:
    """โหลด weight มาแล้วหรือยัง — ใช้ตัวตรวจชุดเดียวกับ lmds doctor ไม่คำนวณซ้ำคนละทาง"""
    from lmds.doctor.checks import _weight_paths

    if not profile:
        return False
    # weight ที่ผู้ใช้ดูแลเอง: ตอบว่า "ไม่รู้" ไม่ได้ จึงถือว่าพร้อม — ปุ่มที่ควรได้คือ start
    # ไม่ใช่ download ที่ทำอะไรไม่ได้จริง · ความจริงว่ามันมีไฟล์ครบไหม รู้ได้ตอน start เท่านั้น
    if self_managed_weights(profile):
        return True
    directory, wanted = _weight_paths(profile, server.slug)
    if not directory.is_dir():
        return False
    return all((directory / name).exists() for name in wanted)


def source_commit() -> str:
    """commit ของซอร์สที่ *ถูก import อยู่จริง* — ว่างเมื่อไม่ได้ติดตั้งจาก git checkout

    เลข version ไม่ขยับทุกคอมมิต (0.2.0 มาหลายสิบคอมมิตแล้ว) จึงบอกไม่ได้เลยว่าเครื่องไหน
    รันโค้ดเก่า — เคสจริง: แก้บั๊กบน hub แล้วเข้าใจว่าทั้งฟลีตได้ของใหม่ ทั้งที่ `lmds agent info`
    ที่คำนวณสถานะทุกอย่างรันด้วยโค้ดของ *เครื่องนั้น* ซึ่งยังเก่าอยู่
    """
    import subprocess

    import lmds

    root = Path(lmds.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return ""
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def host_payload() -> dict:
    import lmds
    from lmds.fit.targets import from_hardware_report
    from lmds.hardware import probe
    from lmds.hardware.profiler import detect_cpu, detect_fabric, host_summary

    report = probe()
    summary = host_summary()
    target = from_hardware_report(report)
    cpu = detect_cpu()
    fabric = detect_fabric()
    return {
        "lmds_version": lmds.__version__,
        # commit ของโค้ดที่เครื่องนี้รันอยู่ — hub เอาไปเทียบว่า node ไหนตามหลังแล้วต้องอัปเดต
        "lmds_commit": source_commit(),
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
        "cpu": cpu,
        # ConnectX/200G — ใช้บอกว่าเครื่องนี้จับคู่ stacked กับเครื่องอื่นได้ไหม
        "fabric": fabric,
        # unified (Spark) ต้องแสดง memory คนละแบบกับ discrete (RTX)
        "memory_model": target.memory_model.value if target else None,
        "gpus": [
            {
                "name": gpu.name,
                "vram_gb": round(gpu.vram_mib / 1024, 1) if gpu.vram_mib
                else (gpu.known.vram_gb if gpu.known else None),
                "compute": gpu.compute_capability,
                "tested": gpu.tested,
                # ค่าสด — GB10 (unified) มักไม่รายงาน memory.total/used จึงเป็น None ได้
                "vram_used_gb": round(gpu.vram_used_mib / 1024, 1) if gpu.vram_used_mib else None,
                "utilization_pct": gpu.utilization_pct,
                # telemetry — None = การ์ดรุ่นนี้ไม่รายงาน หน้าเว็บต้องซ่อนช่องนั้น ไม่ใช่โชว์ 0
                "temperature_c": gpu.temperature_c,
                "power_w": gpu.power_w,
                "power_limit_w": gpu.power_limit_w,
                "fan_pct": gpu.fan_pct,
                "clock_graphics_mhz": gpu.clock_graphics_mhz,
                "clock_graphics_max_mhz": gpu.clock_graphics_max_mhz,
                "clock_memory_mhz": gpu.clock_memory_mhz,
                "clock_sm_mhz": gpu.clock_sm_mhz,
                "pcie_gen": gpu.pcie_gen,
                "pcie_width": gpu.pcie_width,
            }
            for gpu in report.gpus
        ],
    }


def model_payload(server, active_job: dict | None = None) -> dict:
    from lmds.fleet import (
        autostart_status,
        bundle_profile,
        feature_summary,
        profile_context,
        running_context,
    )

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
        # ค่าที่ *กำลังรัน* ชนะค่าที่ bundle ตั้งไว้เสมอ — ผู้ใช้ตั้ง context ตอน start แล้ว
        # หน้าเว็บโชว์ค่าเก่าต่อไป ดูเหมือนช่องที่กรอกไม่ทำงาน ทั้งที่ทำงานถูกต้อง
        "context": running_context(server) or profile_context(profile),
        "context_configured": profile_context(profile),
        "features": feature_summary(profile),
        "autostart": autostart_status(server.slug),
        "topology": (profile or {}).get("topology"),
        "max_num_seqs": ((profile or {}).get("serving") or {}).get("max_num_seqs"),
        "commands": controller_commands(server.controller) if server.controller_exists else [],
        "started_at": server.started_at,
        "downloaded": weights_present(server, profile),
        # หน้าเว็บต้องแยกได้ว่า "โหลดครบแล้ว" กับ "weight ไม่ได้อยู่ในมือ LMDS" คนละเรื่อง
        "self_managed_weights": self_managed_weights(profile),
        "job": active_job,
    }


def snapshot() -> dict:
    """ภาพรวมทั้งเครื่อง — สิ่งที่ `lmds agent info` พิมพ์ออกมาให้ hub อ่าน"""
    from lmds.fleet import discover

    models = [model_payload(s) for s in discover()]
    return {
        "host": host_payload(),
        "models": models,
        # llama.cpp รันหลายโมเดลพร้อมกันได้ (คนละ port) — สรุปให้ hub ไม่ต้องนับเอง
        "summary": {
            "total": len(models),
            "running": sum(1 for m in models if m["running"]),
            "healthy": sum(1 for m in models if m["healthy"]),
            "not_downloaded": sum(1 for m in models if not m["downloaded"]),
        },
    }
