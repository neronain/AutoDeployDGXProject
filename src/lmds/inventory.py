"""สรุปสถานะเครื่องนี้เป็น JSON — ใช้ร่วมกันระหว่างหน้าเว็บกับ `lmds agent info`

hub อ่านข้อมูลของ node ผ่าน SSH โดยเรียก `lmds agent info` ไม่ใช่ยิง HTTP เข้าไป
node จึงไม่ต้องรัน daemon อะไรเลย และไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22
"""

from __future__ import annotations

import os
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



# engine ที่รู้จัก — ใช้จับว่า process/container ที่ถือ GPU อยู่คือ inference server
# ไม่ใช่งานอื่น (training, notebook, ตัดต่อวิดีโอ) ซึ่งไม่ควรชวนให้ adopt
_ENGINE_HINTS = ("sglang", "vllm", "llama-server", "llama_cpp", "ollama",
                 "text-generation", "tgi", "tensorrt")


def foreign_workloads() -> list[dict]:
    """งานที่ถือ GPU อยู่แต่ LMDS ไม่ได้เป็นคนสร้าง

    เครื่องที่เพิ่งถูกแอดเข้าฟลีตมักมีของรันอยู่ก่อนแล้ว — `lmds ps` เห็นเฉพาะ bundle
    ของตัวเอง เครื่องจึงดู "ว่าง" ทั้งที่หน่วยความจำเกือบหมด แล้ว fit ก็วางแผน deploy
    ทับลงไปบนที่ที่ไม่มีจริง

    เคสจริง 2026-08-13 — msi-4 แอดเข้ามาแล้วรายงาน 0 โมเดล ขณะที่ container SGLang
    (`Jackrong/Qwopus3.6-35B-A3B-Coder`, port 30000) รันมา 32 ชั่วโมงและถือ 96,073 MiB

    รายงานอย่างเดียว ไม่แตะอะไรทั้งนั้น — `lmds adopt <container>` มีอยู่แล้วสำหรับ
    คนที่ตัดสินใจว่าจะเอาเข้ามาอยู่ใต้การดูแล
    """
    from lmds.hardware.profiler import compute_apps

    managed = {slug for slug, _ in _running_slugs()}
    found: list[dict] = []

    for pid, name, mib in compute_apps():
        if not any(hint in name.lower() for hint in _ENGINE_HINTS):
            continue
        found.append({"kind": "process", "pid": pid, "name": name, "vram_mib": mib,
                      "detail": _cmdline(pid)})

    for container, image, status in _docker_containers():
        haystack = f"{container} {image}".lower()
        if not any(hint in haystack for hint in _ENGINE_HINTS):
            continue
        if container in managed or any(container.endswith(slug) for slug in managed):
            continue  # ของเราเอง
        found.append({"kind": "container", "name": container, "image": image,
                      "detail": status})
    return found


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(raw.decode("utf-8", "replace").split("\x00")).strip()[:200]


def _docker_containers() -> list[tuple[str, str, str]]:
    import subprocess

    try:
        done = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    rows = []
    for line in done.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def _running_slugs() -> list[tuple[str, str]]:
    """slug ของ bundle ที่ LMDS ดูแลอยู่ — ใช้คัดของตัวเองออกจากรายการ 'ของคนอื่น'"""
    from lmds.fleet import bundle_roots

    slugs = []
    for root in bundle_roots():
        try:
            slugs.extend((d.name, str(d)) for d in root.iterdir() if d.is_dir())
        except OSError:
            continue
    return slugs


# commit ที่ process นี้เริ่มมาด้วย — อ่านครั้งเดียวแล้วจำไว้
#
# ติดตั้งแบบ git checkout ทำให้ `git rev-parse HEAD` เปลี่ยนทันทีที่ `git pull` ทั้งที่
# process ยังรันโค้ดเก่าอยู่ · ถ้าอ่านสดทุกครั้ง "ตัวที่รันอยู่" กับ "ตัวบนดิสก์" จะเท่ากัน
# เสมอ ป้าย "รอรีสตาร์ต" จึงไม่มีวันขึ้นตอนที่ควรขึ้น — และ (เพราะ _build.py ไม่เคยถูก
# git pull อัปเดต) กลับขึ้นค้างถาวรตอนที่ไม่ควรขึ้น ซึ่งเป็นอาการที่เจอจริงบนเครื่องลูกค้า
_BOOT_COMMIT: str | None = None


def source_commit() -> str:
    """commit ของซอร์สที่ *ถูก import อยู่จริง* — ว่างเมื่อไม่ได้ติดตั้งจาก git checkout

    เลข version ไม่ขยับทุกคอมมิต (0.2.0 มาหลายสิบคอมมิตแล้ว) จึงบอกไม่ได้เลยว่าเครื่องไหน
    รันโค้ดเก่า — เคสจริง: แก้บั๊กบน hub แล้วเข้าใจว่าทั้งฟลีตได้ของใหม่ ทั้งที่ `lmds agent info`
    ที่คำนวณสถานะทุกอย่างรันด้วยโค้ดของ *เครื่องนั้น* ซึ่งยังเก่าอยู่
    """
    global _BOOT_COMMIT

    if _BOOT_COMMIT is None:
        _BOOT_COMMIT = _commit_on_disk()
    return _BOOT_COMMIT


def _git_head() -> str | None:
    """commit บนดิสก์จาก git — None เมื่อไม่ได้ติดตั้งจาก git checkout"""
    import subprocess

    import lmds

    root = Path(lmds.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _commit_on_disk() -> str:
    """commit ของโค้ดที่ process นี้โหลดมา (เรียกครั้งแรกครั้งเดียว)"""
    head = _git_head()
    if head is not None:
        return head

    # ติดตั้งแบบปกติ (node ทุกเครื่องเป็นแบบนี้) โค้ดไม่ได้อยู่ใน git checkout แล้ว —
    # ใช้ commit ที่ install.sh ประทับไว้ตอนติดตั้งแทน ซึ่งตรงกับโค้ดที่กำลังรันจริง
    #
    # จงใจ import (ไม่ใช่อ่านไฟล์): python cache โมดูลไว้ตั้งแต่ครั้งแรก ค่านี้จึงเป็นของ
    # "ตอนที่ process นี้เริ่ม" ซึ่งตรงกับโค้ดที่ถูกโหลดเข้าหน่วยความจำไปแล้วจริง ๆ
    # ถ้าอ่านสด ๆ จากดิสก์ เราจะรายงาน commit ของโค้ดที่ยังไม่ได้รัน — โกหกอีกทาง
    try:
        from lmds._build import COMMIT

        return str(COMMIT or "")
    except Exception:
        return ""


def installed_commit() -> str:
    """commit ที่ *ติดตั้งไว้บนดิสก์* ณ ตอนนี้ — ต่างจาก source_commit() ที่เป็นตัวที่รันอยู่

    อ่านไฟล์ตรง ๆ ไม่ผ่าน import เพราะ `lmds._build` ถูก cache ไว้ใน sys.modules ตั้งแต่
    ครั้งแรกที่ถูกเรียก · `install.sh` เขียนทับทีหลังไม่มีผลกับ process ที่รันอยู่

    สองค่านี้ต่างกันเมื่อไหร่ = ติดตั้งของใหม่แล้วแต่ยังไม่ได้รีสตาร์ต ซึ่งเป็นสถานะที่เคย
    หลอกคนมาแล้ว: header โชว์ commit เก่าค้าง แล้วทุก node ที่อัปเดตถูกต้องกลับโดนติดป้าย
    ว่า "โค้ดเก่า" เพราะไม่ตรงกับ hub — ทั้งที่ hub ต่างหากที่ต้องรีสตาร์ต
    """
    import re

    import lmds

    # git checkout: `git pull` ขยับ HEAD แต่ไม่เคยแตะ _build.py — อ่าน _build.py ที่นี่
    # จะได้ commit ตอนติดตั้งครั้งแรกซึ่งค้างอยู่อย่างนั้นตลอดไป แล้วป้าย "รอรีสตาร์ต"
    # ก็ติดถาวร รีสตาร์ตกี่ครั้งหรือ reboot ก็ไม่หาย (เจอจริงบนเครื่องลูกค้า)
    head = _git_head()
    if head is not None:
        return head

    # ไม่ใช่ git checkout — อ่าน **ไฟล์** ไม่ใช่โมดูลที่ python cache ไว้ ค่าที่ install.sh
    # เพิ่งเขียนทับจึงเห็นทันทีโดยไม่ต้องรีสตาร์ต ซึ่งคือทั้งหมดที่ค่านี้มีไว้บอก
    build = Path(lmds.__file__).resolve().parent / "_build.py"
    try:
        text = build.read_text(encoding="utf-8")
    except OSError:
        return ""
    found = re.search(r"""^COMMIT\s*=\s*["']([^"']*)["']""", text, re.MULTILINE)
    return found.group(1) if found else ""


def cache_health() -> dict:
    """แคชโมเดลบนเครื่องนี้ยังเป็นของ user อยู่ไหม — root-owned = โหลด/ลบ/ซิงก์ไม่ได้

    เจอจริงบน msi-5: `docker run` ที่ไม่ได้ใส่ `--user` โหลด weight ลงแคชในฐานะ root
    ผลคือ `~/.cache/huggingface/hub` ทั้งก้อน (73 GB) เป็นของ root — user เขียนไม่ได้
    โมเดลตัวถัดไปจึงโหลดไม่ลง, `remove` ลบไม่ออก, `sync-worker` ตายด้วย rsync exit 23

    เดิมอาการนี้เงียบสนิท: มีปุ่ม "แก้สิทธิ์" อยู่แล้วแต่ไม่มีอะไรบอกว่าต้องกด ผู้ใช้เห็นแค่
    คำสั่งที่ล้มโดยไม่มีสาเหตุ · ตรวจให้เห็นตั้งแต่หน้ารวมเครื่องแทน

    ค่าที่คืน `owner_ok=None` แปลว่ายังไม่มีแคช (เครื่องใหม่) ไม่ใช่ว่ามีปัญหา
    """
    root = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    if not root.is_dir():
        return {"path": str(root), "exists": False, "owner_ok": None, "writable": None}
    me = os.getuid()
    foreign = 0
    # ไล่ทั้งต้นไม้ไม่ไหว (แคชเป็นแสนไฟล์) — ดูถึงชั้นลูกของ models--X ก็พอ
    # ต้องลงถึงชั้นนั้นจริง ๆ: เคสที่เจอบ่อยคือตัวโฟลเดอร์โมเดลเป็นของ user แต่ refs/,
    # .no_exist/, .locks/ ข้างในเป็นของ root — พอสั่ง remove ก็ลบไม่ออกทั้งก้อน
    targets: list[Path] = []
    for base in (root, root / "hub", root / ".locks"):
        if not base.is_dir():
            continue
        targets.append(base)
        for model in base.glob("models--*"):
            targets.append(model)
            try:
                targets.extend(model.iterdir())
            except OSError:
                foreign += 1
    for entry in targets:
        try:
            if entry.stat().st_uid != me:
                foreign += 1
        except OSError:
            foreign += 1
    hub = root / "hub"
    writable = os.access(hub if hub.is_dir() else root, os.W_OK)
    return {
        "path": str(root),
        "exists": True,
        "owner_ok": foreign == 0,
        "foreign_entries": foreign,
        "writable": writable,
    }


def _role_payload(capability) -> dict:
    """บทบาทของเครื่องนี้ในรูปที่คอนโซลใช้ได้ — พร้อมหลักฐานให้คนเถียงกับข้อสรุปได้"""
    return {
        "control_plane": capability.is_control_plane,
        "engines": list(capability.engines),
        "evidence": capability.evidence(),
        "forced": capability.forced,
    }


def host_payload() -> dict:
    import lmds
    from lmds.fit.targets import from_hardware_report
    from lmds.hardware import probe, serving
    from lmds.hardware.profiler import detect_cpu, detect_fabric, host_summary

    report = probe()
    summary = host_summary()
    target = from_hardware_report(report)
    cpu = detect_cpu()
    fabric = detect_fabric()
    return {
        "lmds_version": lmds.__version__,
        # ของที่ถือ GPU อยู่แต่ไม่ได้มาจาก LMDS — เครื่องที่เพิ่งแอดเข้ามามักมี
        "foreign": foreign_workloads(),
        # commit ของโค้ดที่เครื่องนี้รันอยู่ — hub เอาไปเทียบว่า node ไหนตามหลังแล้วต้องอัปเดต
        "lmds_commit": source_commit(),
        "hostname": summary.hostname,
        "ip": summary.ip,
        # ที่อยู่ทุกเส้น ไม่ใช่แค่เส้นที่ออกเน็ต — hub รู้จักเครื่องนี้จากที่อยู่ SSH ซึ่งอาจ
        # เป็นชื่อ (`orb`, ชื่อบน Tailscale) จึงไม่มีทางรู้เลยว่าเครื่องถือ IP อะไรอยู่จริง
        "ips": summary.addresses,
        "arch": report.arch,
        "profile": report.profile.value,
        "ram_used_gb": summary.ram_used_gb,
        "ram_total_gb": summary.ram_total_gb,
        "disk_free_gb": report.disk_free_gb,
        "disk_total_gb": report.disk_total_gb,
        "docker": report.docker,
        "toolkit": report.nvidia_container_toolkit,
        # เครื่องนี้รันโมเดลเองได้ไหม หรือมีหน้าที่แค่สร้าง bundle แล้ว push ต่อ
        # คอนโซลเอาไปตัดสินใจว่าจะโชว์ปุ่ม Download/Start หรือชวนให้ push แทน
        "role": _role_payload(serving.detect()),
        # แคชโมเดลเป็นของ user อยู่ไหม — root-owned ทำให้ download/remove/sync ล้มเงียบ ๆ
        "cache": cache_health(),
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
    commands = controller_commands(server.controller) if server.controller_exists else []
    # ตัวสคริปต์เองคือความจริงสุดท้าย: ไม่มี `download` = LMDS โหลด weight ให้ไม่ได้ จบ
    # เดาจาก profile อย่างเดียวไม่พอ — bundle ที่ adopt มาแล้ว model id บังเอิญเป็นรูป org/name
    # จะหลุดตัวกรอง แล้วหน้าเว็บก็ยื่นปุ่ม download/repair ที่กดไปเจอ usage ของ bash
    self_managed = self_managed_weights(profile) or (
        bool(commands) and "download" not in commands
    )
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
        # การ์ดในเว็บโชว์ slug ซึ่งไม่เคยเปลี่ยน — ตั้งชื่อใหม่แล้วหน้าจอเลยดูเหมือนไม่มีอะไรเกิดขึ้น
        "served_name": server.model or ((profile or {}).get("model") or {}).get("served_name"),
        "default_served_name": server.default_model
        or ((profile or {}).get("model") or {}).get("served_name"),
        # ส่งเป็นตัวเลข ไม่ใช่สตริงรวม — หน้าเว็บจะได้จัดรูปเองได้ ไม่ต้องแกะข้อความ
        "moe": ((profile or {}).get("features") or {}).get("moe") or None,
        "speculative": bool(
            (((profile or {}).get("features") or {}).get("speculative") or {}).get("draft_files")
            or (((profile or {}).get("features") or {}).get("speculative") or {}).get("embedded")
        ),
        "projector": bool(
            (((profile or {}).get("features") or {}).get("multimodal") or {}).get("projector_files")
        ),
        "autostart": autostart_status(server.slug),
        "topology": (profile or {}).get("topology"),
        "max_num_seqs": ((profile or {}).get("serving") or {}).get("max_num_seqs"),
        "commands": commands,
        "started_at": server.started_at,
        "downloaded": True if self_managed else weights_present(server, profile),
        # หน้าเว็บต้องแยกได้ว่า "โหลดครบแล้ว" กับ "weight ไม่ได้อยู่ในมือ LMDS" คนละเรื่อง
        "self_managed_weights": self_managed,
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
