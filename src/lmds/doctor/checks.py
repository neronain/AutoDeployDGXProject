"""Doctor — วินิจฉัยว่าทำไมโมเดลถึง start/download ไม่ผ่าน (คำนวณล้วน ไม่ใช้ LLM)

ทุกข้อในไฟล์นี้มาจาก failure ที่เจอจริงตอน hardware validation 2026-08-03 บน RTX 5090
และจาก reference stacked v8.2 — ไม่ได้เดาว่า "น่าจะพังตรงไหน"

หลักการเดียวกับ Fit Analyzer: ตรวจข้อเท็จจริงบนเครื่อง แล้วบอกคำสั่งแก้ตรง ๆ
ไม่ส่งอะไรให้ LLM ตีความ (ตาม PRD §8 deterministic core)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lmds.fleet import ServerInfo, bundle_profile, find


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Finding:
    name: str
    status: Status
    detail: str
    fix: str = ""


@dataclass
class Diagnosis:
    slug: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if f.status is Status.FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.status is Status.WARN]

    @property
    def healthy(self) -> bool:
        return not self.failed


def _run(args: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def _model_dir(slug: str) -> Path:
    return Path(os.environ.get("MODEL_DIR") or Path.home() / "models" / slug)


def _free_gb(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1024**3
    except OSError:
        return None


# ── checks ────────────────────────────────────────────────────────────────────

def _check_docker(profile: dict, server: ServerInfo) -> list[Finding]:
    if server.mode == "native":
        return []  # llama.cpp native build ไม่ใช้ docker ตอนรัน
    if shutil.which("docker") is None:
        return [Finding("docker", Status.FAIL, "ไม่พบคำสั่ง docker",
                        "ติดตั้ง: curl -fsSL https://get.docker.com | sudo sh")]
    code, _ = _run(["docker", "info"])
    if code != 0:
        return [Finding("docker", Status.FAIL, "เรียก docker ไม่ได้ (daemon ไม่ขึ้น หรือ user ไม่อยู่ในกลุ่ม docker)",
                        "sudo systemctl enable --now docker · sudo usermod -aG docker $USER แล้ว newgrp docker")]
    return [Finding("docker", Status.OK, "ใช้งานได้")]


def _check_image(profile: dict, server: ServerInfo) -> list[Finding]:
    image = (profile.get("runtime") or {}).get("image") or ""
    if server.mode == "native" or not image or shutil.which("docker") is None:
        return []
    code, _ = _run(["docker", "image", "inspect", image])
    if code != 0:
        return [Finding("runtime-image", Status.WARN, f"ยังไม่มี image ในเครื่อง: {image}",
                        f"ดึงล่วงหน้าได้: docker pull {image} (ไม่ดึงเองก็ได้ start จะ pull ให้)")]
    return [Finding("runtime-image", Status.OK, image)]


def _check_hf_token(profile: dict) -> list[Finding]:
    if not (profile.get("model") or {}).get("gated"):
        return []
    if os.environ.get("HF_TOKEN"):
        return [Finding("hf-token", Status.OK, "มี HF_TOKEN ใน environment")]
    # token ที่เก็บใน lmds ใช้ได้แค่ตอน inspect — controller อ่านจาก env เท่านั้น
    return [Finding(
        "hf-token", Status.FAIL,
        "โมเดลนี้เป็น gated repo แต่ไม่มี HF_TOKEN ใน environment "
        "(token ที่เก็บด้วย lmds config ใช้ได้แค่ตอนวิเคราะห์ ไม่ถึง controller)",
        "export HF_TOKEN=hf_xxx  แล้วรัน download/start ใหม่",
    )]


def _weight_paths(profile: dict, slug: str) -> tuple[Path, list[str]]:
    """คืน (โฟลเดอร์ที่ควรมี weight, รายชื่อไฟล์ที่ต้องมี)"""
    model = profile.get("model") or {}
    engine = (profile.get("runtime") or {}).get("engine")
    if engine == "llamacpp":
        wanted = [n.rsplit("/", 1)[-1] for n in [model.get("selected_gguf")] if n]
        projectors = ((profile.get("features") or {}).get("multimodal") or {}).get("projector_files") or []
        wanted += [p.rsplit("/", 1)[-1] for p in projectors]
        return _model_dir(slug), wanted
    repo = (model.get("id") or "").replace("/", "--")
    revision = model.get("revision") or "main"
    return _hf_home() / "hub" / f"models--{repo}" / "snapshots" / revision, []


def _check_weights(profile: dict, slug: str) -> list[Finding]:
    directory, wanted = _weight_paths(profile, slug)
    if not directory.is_dir():
        return [Finding("weights", Status.FAIL, f"ยังไม่มีไฟล์โมเดลที่ {directory}",
                        f"cd bundles/{slug} && ./{slug}-single.sh download")]

    missing = [name for name in wanted if not (directory / name).exists()]
    if missing:
        # เคสจริง: mmproj ไม่ถูกโหลด → multimodal กลายเป็น text-only เงียบ ๆ
        return [Finding("weights", Status.FAIL, f"ไฟล์ที่ต้องมีหายไป: {', '.join(missing)}",
                        f"lmds repair {slug}  (โหลดเฉพาะส่วนที่ขาด)")]

    empty = [p.name for p in directory.glob("*") if p.is_file() and p.stat().st_size == 0]
    if empty:
        return [Finding("weights", Status.FAIL, f"มีไฟล์ขนาด 0 ไบต์: {', '.join(empty[:3])}",
                        f"lmds repair {slug}")]
    return [Finding("weights", Status.OK, str(directory))]


def _check_permissions(profile: dict, slug: str) -> list[Finding]:
    """docker เคยสร้าง cache เป็น root → รอบถัดไปเขียนไม่ได้ (เคสจริงจาก reference v8.2)"""
    findings = []
    directory, _ = _weight_paths(profile, slug)
    candidates = [d for d in {directory, _hf_home(), Path.home() / ".cache" / "flashinfer"} if d.exists()]
    unwritable = [str(d) for d in candidates if not os.access(d, os.W_OK)]
    if unwritable:
        uid, gid = os.getuid(), os.getgid()
        findings.append(Finding(
            "permissions", Status.FAIL,
            f"เขียนไม่ได้: {', '.join(unwritable)} (มักเกิดจาก container เคยสร้างเป็น root)",
            f"sudo chown -R {uid}:{gid} {unwritable[0]}",
        ))
    else:
        findings.append(Finding("permissions", Status.OK, "cache dir เขียนได้ทั้งหมด"))
    return findings


def _check_disk(profile: dict, slug: str) -> list[Finding]:
    directory, _ = _weight_paths(profile, slug)
    free = _free_gb(directory)
    if free is None:
        return []
    if free < 10:
        return [Finding("disk", Status.FAIL, f"ดิสก์เหลือ {free:.0f} GB — ไม่พอสำหรับ download/รัน",
                        "ลบ bundle เก่า (lmds remove) หรือย้ายด้วย HF_HOME / MODEL_DIR")]
    if free < 50:
        return [Finding("disk", Status.WARN, f"ดิสก์เหลือ {free:.0f} GB", "เผื่อไว้ ≥50 GB สำหรับ image + โมเดล")]
    return [Finding("disk", Status.OK, f"เหลือ {free:.0f} GB")]


def _listening_on(port: int) -> str:
    for cmd in (["ss", "-tlnp"], ["netstat", "-tlnp"]):
        if shutil.which(cmd[0]) is None:
            continue
        code, out = _run(cmd)
        if code != 0:
            continue
        for line in out.splitlines():
            if f":{port} " in line:
                return line.strip()
    return ""


def _check_port(server: ServerInfo) -> list[Finding]:
    if not server.port:
        return []
    holder = _listening_on(server.port)
    if server.running:
        if holder:
            return [Finding("port", Status.OK, f"{server.port} — เซิร์ฟเวอร์ตัวนี้ฟังอยู่")]
        return [Finding("port", Status.WARN, f"container ขึ้นแต่ยังไม่มีใครฟัง port {server.port}",
                        f"โมเดลอาจกำลังโหลดอยู่ — ดู: lmds logs {server.slug} -f")]
    if holder:
        return [Finding("port", Status.FAIL, f"port {server.port} ถูกใช้โดยโปรเซสอื่น: {holder[:100]}",
                        f"หยุดตัวที่ชน หรือย้าย port: ./{server.slug}-single.sh start --port 8001")]
    return [Finding("port", Status.OK, f"{server.port} ว่าง")]


def _check_server(server: ServerInfo) -> list[Finding]:
    if not server.running:
        return [Finding("server", Status.WARN, "ยังไม่ได้รัน", f"lmds start {server.slug}")]
    if server.healthy:
        return [Finding("server", Status.OK, f"running + /health ผ่าน ({server.endpoint})")]
    return [Finding("server", Status.WARN, "รันอยู่แต่ /health ยังไม่ผ่าน (อาจกำลังโหลดโมเดล)",
                    f"lmds logs {server.slug} -f")]


def _check_controller(server: ServerInfo) -> list[Finding]:
    if server.controller_exists:
        return [Finding("bundle", Status.OK, server.controller)]
    return [Finding("bundle", Status.FAIL, "ไม่พบไฟล์ controller ของ bundle นี้แล้ว",
                    f"deploy ใหม่ หรือ lmds remove {server.slug} เพื่อล้างทะเบียนทิ้ง")]


def diagnose(slug: str) -> Diagnosis:
    """ตรวจทุกข้อของ slug เดียว — ไม่แก้อะไรให้เอง แค่บอกสาเหตุกับคำสั่ง"""
    server = find(slug)
    if server is None:
        return Diagnosis(slug, [Finding(
            "bundle", Status.FAIL, f"ไม่รู้จัก '{slug}'",
            "ดูรายชื่อที่มี: lmds list",
        )])

    result = Diagnosis(slug)
    result.findings.extend(_check_controller(server))

    profile = bundle_profile(server.controller) or {}
    if not profile:
        result.findings.append(Finding(
            "profile", Status.WARN, "อ่าน MODEL_PROFILE.yaml ไม่ได้ — ตรวจได้ไม่ครบทุกข้อ",
            "ไฟล์อยู่ข้าง controller ใน bundle เดียวกัน",
        ))
    else:
        result.findings.extend(_check_hf_token(profile))
        result.findings.extend(_check_weights(profile, slug))
        result.findings.extend(_check_permissions(profile, slug))
        result.findings.extend(_check_disk(profile, slug))
        result.findings.extend(_check_docker(profile, server))
        result.findings.extend(_check_image(profile, server))

    result.findings.extend(_check_port(server))
    result.findings.extend(_check_server(server))
    return result
