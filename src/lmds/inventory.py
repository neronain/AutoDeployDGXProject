"""สรุปสถานะเครื่องนี้เป็น JSON — ใช้ร่วมกันระหว่างหน้าเว็บกับ `lmds agent info`

hub อ่านข้อมูลของ node ผ่าน SSH โดยเรียก `lmds agent info` ไม่ใช่ยิง HTTP เข้าไป
node จึงไม่ต้องรัน daemon อะไรเลย และไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# คำสั่งที่ controller รู้จัก — ใช้กรองผลจาก dispatch table ไม่ให้ help/-h หลุดมาเป็นปุ่ม
# (`doctor` ไม่อยู่ในนี้: เป็นคำสั่งของ lmds ไม่ใช่ของ controller — dispatch ของ template ไม่มี · audit 2026-09-06 §6.9)
KNOWN_COMMANDS = {
    "prepare-runtime", "download", "verify-files", "start", "stop", "restart", "status",
    "logs", "client-config", "network-info", "test-text", "test-vision", "test-reasoning",
    "test-tools", "bench", "stress", "props", "info", "wait-health",
    "sync-worker", "verify-worker", "clear-fi-cache", "repair", "check-runtime",
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


def read_cluster_env(controller: str) -> dict | None:
    """ค่าคลัสเตอร์ที่ bundle stacked นี้จะใช้จริง — จาก cluster.env ข้าง controller (None = ยังไม่มี)

    hub ต้องรู้ว่า head ตัวนี้จับคู่กับ worker ไหน ถึงจะโชว์โมเดลบนการ์ดของ worker ด้วยได้ ·
    worker ไม่มี bundle ของตัวเอง (weight อยู่ในแคช HF + container ที่ head สั่ง) การ์ดของมันจึง
    ว่างเปล่าทั้งที่เครื่องถูกใช้อยู่ · อ่านเฉพาะคีย์ที่ hub ใช้ ไม่ส่ง env ทั้งไฟล์ออกไป
    """
    if not controller:
        return None
    path = Path(controller).parent / "cluster.env"
    if not path.is_file():
        return None
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return None
    master = values.get("MASTER_IP", "")
    workers = (values.get("WORKER_IPS") or values.get("WORKER_IP") or "").split()
    if not master and not workers:
        return None
    try:
        nnodes = int(values.get("NNODES") or (len(workers) + 1))
    except ValueError:
        nnodes = len(workers) + 1
    return {"master_ip": master, "worker_ips": workers, "nnodes": nnodes,
            "ssh_user": values.get("SSH_USER", "")}


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
    # container ที่ bundle ของเราสั่ง (รวม adopt) — ชื่อไม่จำเป็นต้องขึ้นต้นด้วย lmds-
    managed |= _managed_containers()
    managed_pids = _managed_pids()
    found: list[dict] = []

    for pid, name, mib in compute_apps():
        if not any(hint in name.lower() for hint in _ENGINE_HINTS):
            continue
        # llama-server ที่ bundle ของเรา start เอง หรือ EngineCore ของ vLLM ในคอนเทนเนอร์
        # ที่เราคุมอยู่ — nvidia-smi เห็นเป็น process เหมือนกันหมด · เดิมนับเป็น "นอกระบบ"
        # ทั้งที่เป็นของเรา: dgx-veerasiam ขึ้น "นอกระบบอีก 3" ทั้งที่ทั้ง 3 คือ bundle ของ LMDS
        # และทุกเครื่องที่รัน vLLM ขึ้นซ้ำสองรายการ (container + VLLM::EngineCore)
        if pid in managed_pids or _has_managed_ancestor(pid, managed_pids):
            continue
        owner = _container_of_pid(pid)
        if owner and (owner in managed or any(owner.endswith(slug) for slug in managed)):
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


def memory_by_slug(servers) -> dict[str, float]:
    """GB ที่ bundle แต่ละตัวที่รันอยู่ถือบน GPU — จับคู่ compute-apps กับ container/pid ของ bundle

    ทำไมต้องมี: Fit (fit/sizing.py) ต้องแยก "ที่ตัวเองถืออยู่" ออกจาก "ที่โมเดลอื่นถือ" — เดิมหน้าเว็บมีแค่ยอดรวม
    จึงขึ้น "Cannot start now" ให้โมเดลที่รันอยู่แล้ว (นับหน่วยความจำของตัวมันเองเป็น "ไม่ว่าง") · วัดจริง 2026-09-07
    spark-head: VLLM::EngineCore 80,364 MiB (Nemotron) + llama-server 19,921 MiB (Gemma-4) แยกกันได้จาก cgroup/pid
    · คืนเฉพาะที่จับคู่ได้ — ตัวที่ไม่ได้อยู่ในนี้ผู้เรียกคิดจากยอดรวมแทน
    """
    from lmds.hardware.profiler import compute_apps

    try:
        apps = compute_apps()
    except Exception:  # noqa: BLE001 — nvidia-smi พัง = ไม่รู้ ไม่ใช่ล้ม
        return {}
    running = [s for s in servers if getattr(s, "running", False)]
    if not apps or not running:
        return {}
    by_container = {s.container: s.slug for s in running if s.container}
    by_pid: dict[int, str] = {}
    for s in running:
        pid = int(getattr(s, "pid", 0) or 0)
        if not pid and getattr(s, "pid_file", ""):
            try:
                pid = int(Path(s.pid_file).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = 0
        if pid:
            by_pid[pid] = s.slug
    names = _container_names_by_id() if by_container else {}
    out: dict[str, float] = {}
    for pid, _name, mib in apps:
        slug = None
        owner = _container_of_pid(pid, names) if names else ""
        if owner and owner in by_container:
            slug = by_container[owner]
        else:
            cur, depth = pid, 8
            while cur > 1 and depth > 0 and slug is None:
                slug = by_pid.get(cur)
                cur = _parent_of(cur)
                depth -= 1
        if slug:
            # unified memory (DGX Spark): ที่ process ถือฝั่ง host (VmRSS) ก็กิน RAM ก้อนเดียวกับ GPU — llama.cpp native
            # mmap weight ค้างใน RSS ~เท่า weight (dgx-veerasiam 2026-09-07: gemma-4-12b GPU 17.7 GB + RSS 11.2 GB = 28.9 GB
            # ขณะ free -m บอก used 119 GB) · ไม่นับ = Fit คิดว่าเหลือที่ทั้งที่เครื่องเริ่ม swap แล้ว
            out[slug] = round(out.get(slug, 0.0) + mib / 1024.0 + _rss_gb(pid), 1)
    return out


def _rss_gb(pid: int) -> float:
    """VmRSS ของ pid (GB) — 0 ถ้าอ่านไม่ได้"""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _managed_pids() -> set[int]:
    """pid ของเซิร์ฟเวอร์ native ที่ bundle ของเรา start ไว้ (server.pid ใต้ run root)"""
    from lmds.fleet.manager import run_root

    pids: set[int] = set()
    try:
        for pid_file in run_root().glob("*/server.pid"):
            text = pid_file.read_text(encoding="utf-8", errors="replace").strip()
            if text.isdigit():
                pids.add(int(text))
    except OSError:
        pass
    return pids


def _managed_containers() -> set[str]:
    """ชื่อ container ที่ bundle ของเราลงทะเบียนไว้ (server.meta: container=…)"""
    from lmds.fleet.manager import run_root

    names: set[str] = set()
    try:
        for meta in run_root().glob("*/server.meta"):
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("container=") and line[10:].strip():
                    names.add(line[10:].strip())
    except OSError:
        pass
    return names


def _parent_of(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _has_managed_ancestor(pid: int, managed_pids: set[int], depth: int = 8) -> bool:
    """llama-server/vLLM แตกลูกได้ — ถ้าบรรพบุรุษเป็นของเรา ลูกก็ของเรา"""
    while depth > 0 and pid > 1:
        pid = _parent_of(pid)
        if pid in managed_pids:
            return True
        depth -= 1
    return False


def _container_of_pid(pid: int, names: dict[str, str] | None = None) -> str:
    """ชื่อ container ที่ process นี้อยู่ข้างใน — ว่างเมื่อรันบน host ตรง ๆ

    `names` = ผลของ _container_names_by_id() ที่ผู้เรียกอ่านไว้แล้ว — ไม่ส่งมา = ยิง docker ps ต่อ pid
    """
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for token in cgroup.replace("/", " ").replace("-", " ").replace(".scope", " ").split():
        if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
            return (names if names is not None else _container_names_by_id()).get(token, "")
    return ""


def _container_names_by_id() -> dict[str, str]:
    import subprocess

    try:
        done = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if done.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in done.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


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


def _docker_access(usable: bool) -> dict:
    from lmds.hardware.profiler import docker_access

    if usable:
        import getpass

        return {"installed": True, "usable": True, "in_group": True, "user": getpass.getuser(), "reason": "", "fix": ""}
    try:
        return docker_access()
    except Exception as exc:  # noqa: BLE001 — ฟิลด์ประกอบ ไม่ควรล้ม host payload ทั้งก้อน
        return {"installed": False, "usable": False, "in_group": False, "user": "", "reason": str(exc)[:200], "fix": ""}


def source_dirty() -> list[str]:
    """ไฟล์แก้ค้างใน checkout ที่โค้ดนี้ติดตั้งมา — ว่างเมื่อไม่ใช่ checkout

    hub ที่ dirty ติดตั้งโค้ดที่ยังไม่ commit ให้ตัวเอง (`pip install "$REPO_DIR"`) แต่ `git bundle` ส่งเฉพาะ commit
    ไป node → stamp เท่ากันทั้งที่โค้ดต่างกัน (audit 2026-09-06: hub dirty=8 · ทุก node dirty=0) · ส่งไปกับ agent info
    ทั้งสองฝั่ง จะได้เห็นว่าใครมีของค้าง
    """
    try:
        from lmds.web.selfupdate import dirty_files, source_root

        root = source_root()
        return dirty_files(root) if root is not None else []
    except Exception:  # noqa: BLE001 — ฟิลด์ประกอบ
        return []


# llama-server --version มี 2 รูปแบบ: เก่า `version: 10495 (3dc7285b4)` · ใหม่ (b10xxx+)
# `version: 0.1.2-dev (build 10495, commit 3dc7285b4)` — เคสจริง 2026-09-06 ทั้งฟลีตขึ้น "build ?" เพราะจับได้แต่แบบเก่า
_VERSION_LINE = re.compile(r"version:\s*(?:\S+\s*\(build\s*)?(\d+)[,\s]+(?:commit\s*)?\(?([0-9a-fA-F]{7,})\)")
_RUNTIME_CACHE: dict[str, tuple[tuple, dict]] = {}


def _git(directory: Path, *args: str, timeout: int = 5) -> str:
    import subprocess

    try:
        done = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _git_ok(directory: Path, *args: str) -> bool | None:
    """True/False ตาม exit code · None = git ไม่มี/ถามไม่ได้"""
    import subprocess

    try:
        done = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    # merge-base --is-ancestor: 0 = ใช่ · 1 = ไม่ใช่ · 128 = commit ไม่รู้จัก/ไม่ใช่ repo
    if done.returncode == 0:
        return True
    return False if done.returncode == 1 else None


def llamacpp_runtime_info(directory: Path | str, used_by: list[str] | None = None) -> dict:
    """build llama.cpp ในโฟลเดอร์นี้คือรุ่นไหน + lock ตรงกับ build ไหม — ถูกพอจะถามทุก 15 วิ

    อ่าน stamp `build/lmds-build.json` (prepare-runtime เขียน) · `llama-server --version` (ไม่โหลดโมเดล ~50 ms) ·
    `git log -1` ของ commit นั้น · แคชตาม mtime ของ binary/lock/stamp — ไม่ `git fetch` ไม่ `docker run`
    lock_state: ok = lock คือ build · stale = lock เก่ากว่า build (prepare-runtime แบบเดิมจะ downgrade — แก้แล้ว 0.6.1) ·
    ahead = lock ใหม่กว่า build (มีคน rollback build) · missing = ไม่มี lock · unknown = ถาม git ไม่ได้
    """
    import json
    import subprocess

    directory = Path(directory)
    server = directory / "build" / "bin" / "llama-server"
    lock_path = directory / "build" / "runtime.lock"
    stamp_path = directory / "build" / "lmds-build.json"

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    key = (mtime(server), mtime(lock_path), mtime(stamp_path))
    cached = _RUNTIME_CACHE.get(str(directory))
    if cached and cached[0] == key:
        return {**cached[1], "used_by": list(used_by or [])}

    out: dict = {"dir": str(directory), "present": server.is_file(), "build": "", "commit": "", "date": "",
                 "lock": "", "lock_state": "unknown", "stamp": None, "used_by": list(used_by or [])}
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if isinstance(stamp, dict):
            out["stamp"] = stamp
            out["commit"] = str(stamp.get("commit") or "")
            out["build"] = str(stamp.get("build") or "")
            out["date"] = str(stamp.get("date") or "")
    except (OSError, ValueError):
        pass
    if server.is_file():
        try:
            done = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=10)
            found = _VERSION_LINE.search((done.stdout or "") + (done.stderr or ""))
        except (OSError, subprocess.TimeoutExpired):
            found = None
        if found:
            out["build"] = out["build"] or found.group(1)
            # commit ที่ binary บอกเชื่อได้กว่า stamp (stamp เขียนตาม build · rebuild นอก controller ไม่แตะ stamp)
            if out["commit"] and not out["commit"].startswith(found.group(2)) and not found.group(2).startswith(out["commit"]):
                out["commit"] = found.group(2)
                out["date"] = ""
            out["commit"] = out["commit"] or found.group(2)
    if not out["commit"] and (directory / ".git").is_dir():
        out["commit"] = _git(directory, "rev-parse", "--short=9", "HEAD")
    if out["commit"] and not out["date"] and (directory / ".git").is_dir():
        out["date"] = _git(directory, "log", "-1", "--format=%cs", out["commit"])
    try:
        out["lock"] = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        out["lock"] = ""
    if not out["present"]:
        out["lock_state"] = "missing" if not out["lock"] else "unknown"
    elif not out["lock"]:
        out["lock_state"] = "missing"
    elif out["commit"] and (out["lock"].startswith(out["commit"]) or out["commit"].startswith(out["lock"])):
        out["lock_state"] = "ok"
    elif out["commit"] and (directory / ".git").is_dir():
        older = _git_ok(directory, "merge-base", "--is-ancestor", out["lock"], out["commit"])
        if older:
            out["lock_state"] = "stale"
        elif older is False and _git_ok(directory, "merge-base", "--is-ancestor", out["commit"], out["lock"]):
            out["lock_state"] = "ahead"
        else:
            out["lock_state"] = "unknown"
    _RUNTIME_CACHE[str(directory)] = (key, {k: v for k, v in out.items() if k != "used_by"})
    return out


_IMAGE_CACHE: dict[str, tuple[float, dict]] = {}
_IMAGE_TTL = 60.0


def docker_image_info(ref: str, used_by: list[str] | None = None) -> dict:
    """image ที่ bundle อ้าง มีในเครื่องไหม digest อะไร — `docker image inspect` เท่านั้น (ไม่ pull ไม่ run)"""
    import shutil
    import subprocess
    import time

    cached = _IMAGE_CACHE.get(ref)
    if cached and time.time() - cached[0] < _IMAGE_TTL:
        return {**cached[1], "used_by": list(used_by or [])}
    out = {"ref": ref, "present": False, "id": "", "digest": "", "created": "", "used_by": list(used_by or [])}
    if shutil.which("docker"):
        try:
            done = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}|{{join .RepoDigests \",\"}}|{{.Created}}", ref],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            done = None
        if done is not None and done.returncode == 0 and done.stdout.strip():
            image_id, _, rest = done.stdout.strip().partition("|")
            digests, _, created = rest.partition("|")
            out.update({"present": True, "id": image_id.replace("sha256:", "")[:12],
                        "digest": next((d.split("@", 1)[1] for d in digests.split(",") if "@" in d), ""),
                        "created": created[:19]})
    _IMAGE_CACHE[ref] = (time.time(), {k: v for k, v in out.items() if k != "used_by"})
    return out


def runtime_summary(profile, server=None) -> dict | None:
    """รันไทม์ที่ bundle นี้ผูกไว้ — อ่านจาก profile ไม่ต้องเปิดไฟล์ทีละใบที่ hub"""
    if not profile:
        return None
    runtime = profile.get("runtime") or {}
    engine = runtime.get("engine") or ""
    if not engine:
        return None
    out = {"engine": engine, "mode": "", "image": runtime.get("image") or "", "image_pin": runtime.get("image_pin") or "",
           "llamacpp_dir": ""}
    if engine == "llamacpp":
        from lmds.doctor.checks import llamacpp_mode, llamacpp_root

        out["mode"] = llamacpp_mode(profile, server)
        if out["mode"] == "native":
            out["llamacpp_dir"] = str(llamacpp_root(profile))
    else:
        out["mode"] = "docker"
    return out


def host_runtimes(models: list[dict]) -> dict:
    """build llama.cpp / image ที่ bundle บนเครื่องนี้อ้างถึง — เฉพาะที่มี bundle ใช้ ไม่สแกนทั้งเครื่อง"""
    dirs: dict[str, list[str]] = {}
    images: dict[str, list[str]] = {}
    for m in models:
        runtime = m.get("runtime") or {}
        if runtime.get("llamacpp_dir"):
            dirs.setdefault(runtime["llamacpp_dir"], []).append(m["slug"])
        elif runtime.get("mode") == "docker" and runtime.get("image"):
            image = runtime["image"]
            if runtime.get("image_pin"):
                from lmds.brain.allowlists import image_repo

                image = f"{image_repo(image)}@{runtime['image_pin']}"
            images.setdefault(image, []).append(m["slug"])
    return {
        "llamacpp": [llamacpp_runtime_info(d, slugs) for d, slugs in sorted(dirs.items())],
        "images": [docker_image_info(ref, slugs) for ref, slugs in sorted(images.items())],
    }


def with_runtimes(host: dict, models: list[dict]) -> dict:
    """เติม host.runtimes / host.images จาก bundle ที่สำรวจแล้ว — แยกจาก host_payload() เพื่อให้ผู้เรียกที่มี models อยู่แล้ว
    (snapshot / refresher) ไม่ต้อง discover ซ้ำ และผู้ที่ไม่มี (`/api/host` ครั้งแรก) ยังได้ payload ครบ"""
    runtimes = host_runtimes(models or [])
    host["runtimes"] = {"llamacpp": runtimes["llamacpp"]}
    host["images"] = runtimes["images"]
    return host


def host_payload() -> dict:
    import lmds
    from lmds.fit.targets import from_hardware_report
    from lmds.generator.renderer import template_hash
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
        # ของบนดิสก์ (ต่างจากที่รันอยู่ = ติดตั้งแล้วยังไม่รีสตาร์ต) · ไฟล์แก้ค้างใน checkout · ลายเซ็น template
        # ของแพ็กเกจนี้ — สามค่าที่ "ตรง hub" ต้องดูนอกจาก commit (audit 2026-09-06)
        "lmds_installed_commit": installed_commit(),
        "source_dirty": source_dirty(),
        "template_hash": template_hash(),
        # build llama.cpp / image ที่ bundle บนเครื่องนี้อ้าง (เฉพาะที่ถูกอ้าง) — hub เทียบข้ามเครื่องได้โดยไม่ SSH
        # (เติมจริงด้วย with_runtimes(host, models) เมื่อรู้รายการ bundle แล้ว)
        "runtimes": {"llamacpp": []},
        "images": [],
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
        # ทำไม docker ใช้ไม่ได้ + คำสั่งแก้ — การ์ด "Docker + GPU" บนหน้าเว็บโชว์และมีปุ่มแก้ให้
        "docker_access": _docker_access(report.docker),
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


def runtime_arch_status(server, profile) -> dict | None:
    """รันไทม์ llama.cpp บนเครื่องนี้รู้จัก arch ของโมเดลไหม — ให้ hub ติดป้าย "runtime เก่ากว่าโมเดล" ก่อนใครกด start

    เคสจริง 2026-09-06 spark-worker: bundle qwen4exp วางบนเครื่องที่ build llama.cpp ไว้ตั้งแต่ 18 ส.ค. —
    ทุกอย่างเขียว (ไฟล์ครบ port ว่าง) จน start ตาย exit 1 · อ่านหัว GGUF + สแกน libllama (ไม่กี่ MB) ถูกพอ
    ที่จะทำทุกครั้งที่ hub ถาม · docker mode ไม่ยิง `docker run` ตรงนี้ — อ่านผลที่ doctor จดไว้เท่านั้น
    """
    from lmds.doctor.checks import llamacpp_arch_support

    try:
        support = llamacpp_arch_support(profile or {}, server.slug, server, probe_docker=False)
    except Exception:  # noqa: BLE001 — ฟิลด์ประกอบ ไม่ควรล้ม payload ของโมเดลทั้งก้อน
        return None
    if support is None:
        return None
    return {k: support[k] for k in ("arch", "mode", "supported", "runtime", "fix")}


def model_payload(server, active_job: dict | None = None, memory_gb: float | None = None) -> dict:
    """`memory_gb` = ที่ตัวนี้ถือบน GPU ตอนนี้ (จาก memory_by_slug) — None = ไม่ได้รัน/จับคู่ไม่ได้"""
    from lmds.fleet.consistency import controller_header, controller_state
    from lmds.fleet import (
        autostart_status,
        bundle_profile,
        feature_summary,
        profile_context,
        running_context, running_slots,
    )

    profile = bundle_profile(server.controller)
    commands = controller_commands(server.controller) if server.controller_exists else []
    # ตัวสคริปต์เองคือความจริงสุดท้าย: ไม่มี `download` = LMDS โหลด weight ให้ไม่ได้ จบ
    # เดาจาก profile อย่างเดียวไม่พอ — bundle ที่ adopt มาแล้ว model id บังเอิญเป็นรูป org/name
    # จะหลุดตัวกรอง แล้วหน้าเว็บก็ยื่นปุ่ม download/repair ที่กดไปเจอ usage ของ bash
    self_managed = self_managed_weights(profile) or (
        bool(commands) and "download" not in commands
    )
    ctx_now = running_context(server) or profile_context(profile)
    slots = running_slots(server) or ((profile or {}).get("serving") or {}).get("max_num_seqs") or None
    if (server.engine or "") == "llamacpp" and ctx_now and slots and int(slots) > 1:
        context_per_request = int(ctx_now) // int(slots)
    else:
        context_per_request = ctx_now
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
        # llama.cpp แบ่ง --ctx-size ให้ทุก slot เท่ากัน → คำขอเดียวได้ context ÷ slots (vLLM: max-model-len เป็นต่อคำขออยู่แล้ว)
        # เคสจริง 2026-09-07 dgx-veerasiam: ตั้ง 131,071 slots 2 แล้ว Score บอก ctx max 65,536 — ผู้ใช้เข้าใจว่าค่าไม่ติด
        "slots": slots,
        "context_per_request": context_per_request,
        # เพดานของโมเดล — ช่อง context บนหน้าเว็บใส่ max/hint ให้ ไม่ปล่อยให้ตั้งเกินแล้วไปตายตอน start
        "native_context": ((profile or {}).get("model") or {}).get("native_context") or None,
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
        # stacked: head ตัวนี้จับคู่กับ worker ไหน (จาก cluster.env) — hub เอาไปวาดการ์ดของ worker
        "cluster": read_cluster_env(server.controller) if (profile or {}).get("topology") == "stacked" else None,
        "max_num_seqs": ((profile or {}).get("serving") or {}).get("max_num_seqs"),
        # ที่ตัวนี้ถือบน GPU ตอนนี้ (GB) — Fit หักออกจาก "ใช้อยู่แล้ว" ไม่ให้โมเดลที่รันอยู่ถูกนับเป็นคนอื่น
        "memory_gb": memory_gb,
        "commands": commands,
        "started_at": server.started_at,
        "downloaded": True if self_managed else weights_present(server, profile),
        # หน้าเว็บต้องแยกได้ว่า "โหลดครบแล้ว" กับ "weight ไม่ได้อยู่ในมือ LMDS" คนละเรื่อง
        "self_managed_weights": self_managed,
        # llama.cpp: build/image บนเครื่องรู้จัก arch ของโมเดลไหม (supported=false = ป้าย "runtime older than model"
        # + ปุ่ม update runtime) · None = ไม่ใช่ llama.cpp หรือยังบอกไม่ได้
        "runtime_arch": runtime_arch_status(server, profile),
        # ใครสร้าง controller นี้ (generated_by / SCRIPT_VERSION / template_hash) และเก่ากว่าแพ็กเกจบนเครื่องไหม —
        # hub ใช้ตัดสินมิติ "controller" ของ "ตรง hub" · runtime = build/image ที่ผูกไว้ (ไม่ต้องเปิดไฟล์ทีละใบ)
        "generated_by": (profile or {}).get("generated_by"),
        "template_hash": (profile or {}).get("template_hash"),
        "script_version": controller_header(server.controller)["script_version"] if server.controller_exists else "",
        "controller": controller_state(profile, server.controller if server.controller_exists else None)
        if profile else None,
        "runtime": runtime_summary(profile, server),
        "job": active_job,
    }


def snapshot() -> dict:
    """ภาพรวมทั้งเครื่อง — สิ่งที่ `lmds agent info` พิมพ์ออกมาให้ hub อ่าน"""
    from lmds.fleet import discover

    servers = discover()
    held = memory_by_slug(servers)
    models = [model_payload(s, memory_gb=held.get(s.slug)) for s in servers]
    return {
        "host": with_runtimes(host_payload(), models),
        "models": models,
        # llama.cpp รันหลายโมเดลพร้อมกันได้ (คนละ port) — สรุปให้ hub ไม่ต้องนับเอง
        "summary": summary_of(models),
    }


def summary_of(models: list[dict]) -> dict:
    return {
        "total": len(models),
        "running": sum(1 for m in models if m["running"]),
        "healthy": sum(1 for m in models if m["healthy"]),
        "not_downloaded": sum(1 for m in models if not m["downloaded"]),
        # controller เก่ากว่า lmds / runtime เก่ากว่าโมเดล — ตัวเลขที่ "ตรง hub" ต้องเป็น 0 ทั้งคู่
        "controllers_stale": sum(1 for m in models if ((m.get("controller") or {}).get("state")) == "stale"),
        "runtime_stale": sum(1 for m in models if ((m.get("runtime_arch") or {}).get("supported")) is False),
    }
