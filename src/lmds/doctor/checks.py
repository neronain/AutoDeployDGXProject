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

from lmds.fleet import ServerInfo, bundle_profile, discover, find


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
        # "ยังไม่ได้ pull" กับ "ไม่มี tag นี้อยู่จริง" ต่างกันคนละเรื่อง — ข้อความเดิมบอกเหมือนกัน
        # ผู้ใช้จึงกด start ซ้ำแล้วเจอ "manifest unknown" โดยไม่รู้ว่าปัญหาอยู่ตรงไหน
        from lmds.brain.registry import tag_exists

        if tag_exists(image) is False:
            return [Finding(
                "runtime-image", Status.FAIL,
                f"image '{image}' ไม่มีอยู่จริงบน registry — pull ไม่ได้แน่นอน",
                f"เปลี่ยน image ตอน start: VLLM_IMAGE=<image ที่มีจริง> lmds start {server.slug}"
                f"  ·  หรือ deploy ใหม่เพื่อให้ระบบเลือก image ให้",
            )]
        return [Finding("runtime-image", Status.WARN, f"ยังไม่มี image ในเครื่อง: {image}",
                        f"ดึงล่วงหน้าได้: docker pull {image} (ไม่ดึงเองก็ได้ start จะ pull ให้)")]
    return [Finding("runtime-image", Status.OK, image)]


def _model_type(profile: dict, slug: str) -> str:
    """`model_type` จาก config.json ของ checkpoint ที่โหลดมาแล้ว

    อ่านจากไฟล์ในเครื่อง ไม่ถาม Hub — ตอน doctor เราสนใจว่าของที่อยู่บนดิสก์ตรงนี้
    รันได้ไหม ไม่ใช่ว่าของบน Hub เป็นยังไง
    """
    import json

    directory, _ = _weight_paths(profile, slug)
    config = directory / "config.json"
    if not config.is_file():
        return ""
    try:
        return str(json.loads(config.read_text(encoding="utf-8")).get("model_type") or "")
    except (OSError, ValueError):
        return ""


# โมเดลที่ออกใหม่กว่ารันไทม์เป็นเรื่องปกติ ไม่ใช่ความผิดของใคร — แต่มันจบด้วย
# container ที่ตายเงียบ ๆ หลังโหลด weight มาแล้วหลายสิบกิกะ ซึ่งแพงเกินกว่าจะ
# ปล่อยให้รู้ตอนนั้น
_ARCH_PROBE = (
    "import sys\n"
    "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES\n"
    "import transformers\n"
    "print('KNOWN' if sys.argv[1] in CONFIG_MAPPING_NAMES else 'UNKNOWN', transformers.__version__)\n"
)


def _gguf_architecture(path: Path) -> str:
    """`general.architecture` จากหัวไฟล์ GGUF ในเครื่อง

    อ่านผ่าน parser ตัวเดียวกับ inspector ไม่เขียนใหม่ — GGUF metadata มี vocab
    อยู่ด้วยจึงใหญ่เกินกว่าจะอ่านทั้งก้อนขึ้นหน่วยความจำ เลยป้อนเป็น stream
    """
    from lmds.inspector.gguf import GgufParseError, parse_gguf

    class _FileSource:
        def __init__(self, handle):
            self._handle = handle

        def read(self, n: int) -> bytes:
            data = self._handle.read(n)
            if len(data) < n:
                raise EOFError("ปลายไฟล์ก่อนอ่านครบ")
            return data

        def skip(self, n: int) -> None:
            self._handle.seek(n, 1)

    try:
        with path.open("rb") as handle:
            return parse_gguf(_FileSource(handle)).architecture or ""
    except (OSError, EOFError, GgufParseError):
        return ""


def _check_architecture_llamacpp(profile: dict, slug: str) -> list[Finding]:
    """llama.cpp build ตัวที่โมเดลนี้ผูกไว้ รู้จักสถาปัตยกรรมของมันไหม

    เคสจริง 2026-08-13: Muse-Glimmer-30B ใช้ architecture `muse-glimmer` ซึ่ง
    llama.cpp บน spark-head (23 ก.ค., ตามหลัง upstream 296 commit) ยังไม่รู้จัก
    ถ้าไม่ตรวจตรงนี้ ผู้ใช้จะโหลด 30 GB จบแล้วค่อยเจอตอน start ว่ารันไม่ได้

    เช็คเดิมข้ามทาง native ทั้งหมด (`if server.mode == "native": return []`)
    ทั้งที่ llama.cpp บน DGX Spark รัน native เป็นปกติ — เช็คที่มีอยู่จึงไม่เคย
    ทำงานกับ engine ที่ต้องการมันที่สุด
    """
    directory, wanted = _weight_paths(profile, slug)
    gguf = next((directory / name for name in wanted if name.endswith(".gguf")), None)
    if gguf is None or not gguf.is_file():
        return []  # ยังไม่ได้โหลด — _check_weights บอกไปแล้ว
    architecture = _gguf_architecture(gguf)
    if not architecture:
        return []

    pinned = (profile.get("target") or {}).get("llamacpp_dir")
    root = Path(pinned) if pinned else Path.home() / "src" / "llama.cpp"
    libs = sorted(root.glob("build/bin/libllama.so*")) + sorted(root.glob("build/src/libllama.so*"))
    if not libs:
        return [Finding(
            "architecture", Status.WARN,
            f"ไม่พบ libllama.so ใต้ {root} — ตรวจสถาปัตยกรรมไม่ได้",
        )]

    known = any(_lib_knows(lib, architecture) for lib in libs)
    if known:
        return [Finding("architecture", Status.OK, f"{architecture} (llama.cpp {root.name})")]
    return [Finding(
        "architecture", Status.FAIL,
        f"llama.cpp ที่ {root} ไม่รู้จักสถาปัตยกรรม '{architecture}' — "
        f"โมเดลใหม่กว่ารันไทม์ ต้อง build llama.cpp ใหม่ให้รองรับก่อน: "
        f"cd {root} && git pull && cmake --build build -j",
    )]


def _lib_knows(lib: Path, architecture: str) -> bool:
    """ชื่อ architecture โผล่ใน .so ไหม

    ไม่ใช้ `strings | grep -q` เพราะ grep ปิด pipe ทันทีที่เจอ แล้ว strings โดน
    SIGPIPE — ภายใต้ pipefail จะกลายเป็น "ไม่เจอ" ทั้งที่เจอ อ่านเองตรง ๆ ชัดกว่า
    """
    needle = architecture.encode()
    try:
        with lib.open("rb") as handle:
            tail = b""
            while chunk := handle.read(1 << 20):
                if needle in tail + chunk:
                    return True
                tail = chunk[-len(needle):]
    except OSError:
        return False
    return False



# llama.cpp แปลง JSON schema ของ tool เป็น GBNF · `maxLength`/`maxItems` ค่าสูงถูก
# ขยายเป็น repetition ตรง ๆ แล้วชน MAX_REPETITION_THRESHOLD (2000) จนโยน exception
#
#   parse: error parsing grammar: number of repetitions exceeds sane defaults
#   srv send_error: Failed to initialize samplers: failed to parse grammar
#
# upstream แก้ที่ cd0fa6051 (2026-08-05) — เปลี่ยนจาก throw เป็นลด max เหลือ unbounded
# ข้อความ error เดิมยังอยู่ในไบนารีทั้งสองรุ่น (min_times ยัง throw) จึงดูจากสตริงไม่ได้
# ต้องดูที่ commit
_GRAMMAR_FIX = "cd0fa6051"


def _check_llamacpp_grammar(profile: dict, slug: str) -> list[Finding]:
    """llama.cpp ตัวนี้รับ schema ที่ agent client ส่งมาไหว หรือจะตายตอนเรียก tool

    เคสจริง 2026-08-13 — gpt-oss-120b บน spark-worker เสิร์ฟได้ปกติ ตอบ chat ได้
    เรียก tool ด้วย schema ง่าย ๆ ก็ได้ แต่พอ Claude Code ส่งชุด tool จริงมาก็ 400
    ทันที · โมเดลไม่ผิด ไฟล์ไม่ขาด — llama.cpp เก่ากว่าที่ client ต้องการเท่านั้น

    อาการนี้จับตอน deploy ไม่ได้เลยถ้าไม่ตรวจ เพราะทุกอย่างขึ้นปกติหมด
    """
    if (profile.get("runtime") or {}).get("engine") != "llamacpp":
        return []
    pinned = (profile.get("target") or {}).get("llamacpp_dir")
    root = Path(pinned) if pinned else Path.home() / "src" / "llama.cpp"
    if not (root / ".git").exists():
        return []  # ไม่ใช่ checkout (ติดตั้งจาก tarball/แพ็กเกจ) — ตรวจ ancestry ไม่ได้

    code, _ = _run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                    _GRAMMAR_FIX, "HEAD"], timeout=20)
    if code == 0:
        return [Finding("grammar", Status.OK, f"llama.cpp มี {_GRAMMAR_FIX} — tool schema ใหญ่ผ่าน")]
    if code != 1:
        return []  # ไม่รู้จัก commit นั้น (checkout ตื้น/คนละ remote) — ไม่ตัดสิน

    return [Finding(
        "grammar", Status.WARN,
        f"llama.cpp ที่ {root} ยังไม่มี {_GRAMMAR_FIX} — tool ที่มี maxLength/maxItems "
        "เกิน 2000 จะทำให้ตอบ 400 'failed to parse grammar' (Claude Code ส่งแบบนั้นมา) · "
        f"แก้: cd {root} && git pull && cmake --build build -j",
    )]


def _check_architecture(profile: dict, server: ServerInfo, slug: str) -> list[Finding]:
    """รันไทม์ตัวนี้รู้จักสถาปัตยกรรมของ checkpoint นี้ไหม"""
    if (profile.get("runtime") or {}).get("engine") == "llamacpp":
        return _check_architecture_llamacpp(profile, slug)
    image = (profile.get("runtime") or {}).get("image") or ""
    model_type = _model_type(profile, slug)
    if server.mode == "native" or not image or not model_type or shutil.which("docker") is None:
        return []
    # image ที่ยังไม่ได้ pull — _check_image บอกไปแล้ว ไม่ต้องดึง 20 GB มาเพื่อถาม
    if _run(["docker", "image", "inspect", image])[0] != 0:
        return []

    code, out = _run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", _ARCH_PROBE, model_type],
        timeout=120,
    )
    if code != 0 or not out.strip():
        # ถามไม่ได้ ไม่ได้แปลว่าใช้ไม่ได้ — เงียบดีกว่าเตือนผิด
        return []

    verdict, _, version = out.strip().split()[0], None, (out.strip().split() + [""])[1]
    if verdict == "KNOWN":
        return [Finding("architecture", Status.OK, f"{model_type} (transformers {version})")]
    return [Finding(
        "architecture", Status.FAIL,
        f"image นี้ไม่รู้จักสถาปัตยกรรม '{model_type}' (transformers {version}) — "
        f"start แล้ว container จะตายทันทีที่โหลด config",
        "โมเดลใหม่กว่ารันไทม์ · ลอง image ที่ transformers ใหม่กว่า: "
        f"VLLM_IMAGE=<image ใหม่กว่า> lmds start {slug}"
        "  ·  เช็คก่อนได้ว่าตัวไหนรู้จัก: "
        "docker run --rm --entrypoint python3 <image> -c "
        "\"from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as m; "
        f"print('{model_type}' in m)\"",
    )]


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
    """คืน (โฟลเดอร์ที่ควรมี weight, ไฟล์ที่**ขาดไม่ได้**) — projector อยู่ใน _projectors()"""
    model = profile.get("model") or {}
    engine = (profile.get("runtime") or {}).get("engine")
    if engine == "llamacpp":
        wanted = [n.rsplit("/", 1)[-1] for n in [model.get("selected_gguf")] if n]
        return _model_dir(slug), wanted
    repo = (model.get("id") or "").replace("/", "--")
    revision = model.get("revision") or "main"
    # HF cache มีสองเลย์เอาต์: $HF_HOME/hub/models--X (ปัจจุบัน) และ $HF_HOME/models--X (เก่า)
    # controller รองรับทั้งคู่แล้ว แต่ตรงนี้เคยดูแค่ hub/ — โมเดลที่โหลดด้วย HF รุ่นเก่าจึงขึ้นว่า
    # "ยังไม่ download" ทั้งที่ไฟล์ครบทุกไฟล์ (เจอจริงกับ DeepSeek V4 บน spark-head)
    home = _hf_home()
    for base in (home / "hub", home):
        candidate = base / f"models--{repo}" / "snapshots" / revision
        if candidate.is_dir():
            return candidate, []
        # revision อาจถูกเก็บเป็น ref ไม่ใช่ชื่อโฟลเดอร์ — ยอมรับ snapshot ที่มีอยู่จริงตัวใดก็ได้
        snapshots = base / f"models--{repo}" / "snapshots"
        if snapshots.is_dir():
            existing = sorted(p for p in snapshots.iterdir() if p.is_dir())
            if existing:
                return existing[-1], []
    return home / "hub" / f"models--{repo}" / "snapshots" / revision, []


def _projectors(profile: dict) -> list[str]:
    """ไฟล์ mmproj ที่ profile ประกาศไว้ — เป็น **ทางเลือก** ไม่ใช่ของบังคับ

    llama-server รับ `--mmproj` ได้ไฟล์เดียว แต่ repo มักมีหลาย precision (BF16/F16/F32)
    ให้เลือก · profile รุ่นเก่าจึงลิสต์ไว้ทั้งหมด ทั้งที่ controller โหลดและใช้แค่ตัวเดียว
    """
    multimodal = (profile.get("features") or {}).get("multimodal") or {}
    return [p.rsplit("/", 1)[-1] for p in (multimodal.get("projector_files") or [])]


def _check_weights(profile: dict, slug: str) -> list[Finding]:
    directory, wanted = _weight_paths(profile, slug)
    if not directory.is_dir():
        # บอกคำสั่งระดับ lmds ก่อนเสมอ — ใช้ได้จากที่ไหนก็ได้ และเป็นปุ่มเดียวกับที่มีบนหน้าเว็บ
        # เดิมบอกให้ cd เข้า bundle ทั้งที่ `lmds repair` ทำงานเดียวกัน (download resume + verify)
        # ผู้ใช้ที่อ่าน doctor จากหน้าเว็บจึงไม่มีทางทำตามได้โดยไม่ ssh เข้าเครื่องนั้น
        return [Finding("weights", Status.FAIL, f"ยังไม่มีไฟล์โมเดลที่ {directory}",
                        f"lmds repair {slug}  (โหลด resume ได้ แล้วตรวจไฟล์ให้)")]

    missing = [name for name in wanted if not (directory / name).exists()]
    if missing:
        return [Finding("weights", Status.FAIL, f"ไฟล์ที่ต้องมีหายไป: {', '.join(missing)}",
                        f"lmds repair {slug}  (โหลดเฉพาะส่วนที่ขาด)")]

    empty = [p.name for p in directory.glob("*") if p.is_file() and p.stat().st_size == 0]
    if empty:
        return [Finding("weights", Status.FAIL, f"มีไฟล์ขนาด 0 ไบต์: {', '.join(empty[:3])}",
                        f"lmds repair {slug}")]

    findings = [Finding("weights", Status.OK, str(directory))]
    # mmproj ขาด = เสีย vision แต่โมเดล **ยังรันได้** เป็น text-only จึงเป็นคำเตือน ไม่ใช่ FAIL
    # เดิมนับรวมเป็นไฟล์บังคับและบังคับ *ครบทุก precision* → gemma-4-31b ที่โหลดครบแล้วขึ้นว่า
    # "ยังไม่ download" ตลอดกาล ปุ่ม start เลยไม่ขึ้นทั้งที่รันได้จริง (เจอบน dgx-veerasiam)
    projectors = _projectors(profile)
    if projectors and not any((directory / name).exists() for name in projectors):
        findings.append(Finding(
            "multimodal", Status.WARN,
            f"ไม่มีไฟล์ mmproj ({', '.join(projectors)}) — โมเดลจะรับแต่ข้อความ ภาพใช้ไม่ได้",
            f"lmds repair {slug}  (หรือรันแบบ text-only ต่อได้เลย)",
        ))
    return findings


def _check_permissions(profile: dict, slug: str) -> list[Finding]:
    """docker เคยสร้าง cache เป็น root → รอบถัดไปเขียนไม่ได้ (เคสจริงจาก reference v8.2)"""
    findings = []
    directory, _ = _weight_paths(profile, slug)
    candidates = [d for d in {directory, _hf_home(), Path.home() / ".cache" / "flashinfer"} if d.exists()]
    unwritable = [str(d) for d in candidates if not os.access(d, os.W_OK)]
    uid, gid = os.getuid(), os.getgid()
    if unwritable:
        findings.append(Finding(
            "permissions", Status.FAIL,
            f"เขียนไม่ได้: {', '.join(unwritable)} (มักเกิดจาก container เคยสร้างเป็น root)",
            f"sudo chown -R {uid}:{gid} {unwritable[0]}",
        ))
        return findings

    # โฟลเดอร์เขียนได้ไม่ได้แปลว่าไฟล์ข้างในอ่านได้ทุกตัว — container ที่รันเป็น root ทิ้ง
    # ไฟล์โหมด 600 ของ root ไว้ได้ในโฟลเดอร์ที่เราเขียนได้ · รันเครื่องเดียวไม่เจอ แต่
    # `sync-worker` ที่คัดลอกในฐานะ user จะตายด้วย rsync exit 23 (เจอจริงกับ DeepSeek-V4-Flash)
    blocked = _first_unreadable(candidates)
    if blocked:
        findings.append(Finding(
            "permissions", Status.FAIL,
            f"อ่านไฟล์ไม่ได้: {blocked} (มักเกิดจาก container เคยรันเป็น root) — "
            f"คัดลอกไป worker ไม่ได้",
            f"sudo chown -R {uid}:{gid} {_hf_home()}",
        ))
    else:
        findings.append(Finding("permissions", Status.OK, "cache dir เขียนได้และอ่านไฟล์ได้ครบ"))
    return findings


# แคชโมเดลมีไฟล์เป็นหมื่น — ไล่ทั้งต้นไม้ทุกครั้งช้าเกินไปสำหรับคำสั่งที่ควรตอบทันที
# หยุดทันทีที่เจอไฟล์แรกที่อ่านไม่ได้ และมีเพดานจำนวนไฟล์กันเคสแคชใหญ่ผิดปกติ
_READ_SCAN_LIMIT = 20_000


def _first_unreadable(roots: list[Path]) -> str:
    seen = 0
    for root in roots:
        for path, _, files in os.walk(root, onerror=lambda _e: None):
            for name in files:
                seen += 1
                if seen > _READ_SCAN_LIMIT:
                    return ""
                full = os.path.join(path, name)
                if not os.access(full, os.R_OK):
                    return full
    return ""


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
        # ส่วนใหญ่ตัวที่ยึด port คือโมเดล LMDS อีกตัวที่ยังรันอยู่ (ทุก bundle default 8000 เหมือนกัน)
        # บอกชื่อไปเลยดีกว่าให้ผู้ใช้ไปไล่หาเองจาก output ของ ss
        rival = next(
            (s.slug for s in discover()
             if s.slug != server.slug and s.running and s.port == server.port),
            "",
        )
        if rival:
            return [Finding("port", Status.FAIL,
                            f"port {server.port} ถูก {rival} ใช้อยู่ (ทุก bundle ตั้งต้นที่ 8000 เหมือนกัน)",
                            f"lmds stop {rival}   หรือรันคู่กันคนละ port: lmds start {server.slug} --port 8001")]
        return [Finding("port", Status.FAIL, f"port {server.port} ถูกใช้โดยโปรเซสอื่น: {holder[:100]}",
                        f"หยุดตัวที่ชน หรือย้าย port: lmds start {server.slug} --port 8001")]
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
        result.findings.extend(_check_architecture(profile, server, slug))
        result.findings.extend(_check_llamacpp_grammar(profile, slug))

    result.findings.extend(_check_port(server))
    result.findings.extend(_check_server(server))
    return result
