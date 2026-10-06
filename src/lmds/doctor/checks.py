"""Doctor — วินิจฉัยว่าทำไมโมเดลถึง start/download ไม่ผ่าน (คำนวณล้วน ไม่ใช้ LLM)

ทุกข้อในไฟล์นี้มาจาก failure ที่เจอจริงตอน hardware validation 2026-08-03 บน RTX 5090
และจาก reference stacked v8.2 — ไม่ได้เดาว่า "น่าจะพังตรงไหน"

หลักการเดียวกับ Fit Analyzer: ตรวจข้อเท็จจริงบนเครื่อง แล้วบอกคำสั่งแก้ตรง ๆ
ไม่ส่งอะไรให้ LLM ตีความ (ตาม PRD §8 deterministic core)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dataclasses import replace

from lmds.fleet import ServerInfo, bundle_profile, discover, find
from lmds.inventory import self_managed_weights


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


def llamacpp_root(profile: dict) -> Path:
    """โฟลเดอร์ build llama.cpp ที่ bundle นี้ใช้ — ค่าที่ pin ไว้ใน profile ไม่งั้นโฟลเดอร์กลางของเครื่อง"""
    pinned = (profile.get("target") or {}).get("llamacpp_dir")
    return Path(pinned) if pinned else Path.home() / "src" / "llama.cpp"


def llamacpp_mode(profile: dict, server: ServerInfo | None = None) -> str:
    """native (build จาก source — DGX Spark) หรือ docker (image ทางการ — RTX x86_64)

    server.meta ของ bundle ที่ยังไม่เคย start ถูกเขียนโดย register_bundle ซึ่งเดาเป็น docker เมื่อ profile
    ไม่บอก — profile รุ่นเก่าไม่มี runtime.native_build จึงต้องดู target.memory_model (unified = native)
    """
    runtime = profile.get("runtime") or {}
    if runtime.get("native_build") is not None:
        return "native" if runtime.get("native_build") else "docker"
    if (profile.get("target") or {}).get("memory_model") == "unified":
        return "native"
    if server is not None and server.mode in ("native", "docker"):
        return server.mode
    return "docker"


def _runtime_libs(root: Path) -> list[Path]:
    """ไฟล์ที่มีตาราง LLM_ARCH_NAMES — libllama.so (build แบบ shared) หรือตัว llama-server เอง (static)

    สแกน llama-server เฉพาะเมื่อไม่มี libllama: build static ตัวไบนารีใหญ่เป็นร้อย MB และตรงนี้ถูกเรียก
    ทุกครั้งที่ hub ถาม `agent info`
    """
    libs = sorted(root.glob("build/bin/libllama.so*")) + sorted(root.glob("build/src/libllama.so*"))
    if libs:
        return libs
    server = root / "build" / "bin" / "llama-server"
    return [server] if server.is_file() else []


def _arch_cache_path(slug: str) -> Path:
    from lmds.fleet.manager import run_root

    return run_root() / slug / "runtime-arch.json"


def _docker_knows_arch(image: str, architecture: str, slug: str, probe: bool) -> bool | None:
    """image llama.cpp ที่ pin ไว้รู้จัก arch นี้ไหม — None = ตอบไม่ได้

    `docker run` แพงเกินกว่าจะทำทุกครั้งที่ hub poll (`lmds agent info` ทุกสิบวิ) — doctor เป็นคนถาม
    แล้วจดผลไว้ที่ run/<slug>/runtime-arch.json · inventory อ่านแค่ที่จดไว้ (probe=False)
    """
    import json

    cache = _arch_cache_path(slug)
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("image") == image and saved.get("arch") == architecture:
            return saved.get("supported")
    except (OSError, ValueError):
        pass
    if not probe or shutil.which("docker") is None:
        return None
    if _run(["docker", "image", "inspect", image])[0] != 0:
        return None  # ยังไม่ได้ pull — _check_image บอกไปแล้ว ไม่ดึงหลาย GB มาเพื่อถาม
    # สคริปต์เดียวกับ runtime_knows_arch ของ controller — image ที่วางไฟล์ไว้ที่อื่นตอบ none (ถามไม่ได้) ไม่ใช่ 0
    code, out = _run(["docker", "run", "--rm", "--entrypoint", "sh", image, "-c",
                      'files="$(ls /app/libllama.so* /app/llama-server /app/*.so 2>/dev/null)"; '
                      '[ -n "$files" ] || { echo none; exit 0; }; cat $files | grep -acF -- "$1"',
                      "sh", architecture], timeout=120)
    last = (out.strip().splitlines() or [""])[-1].strip()
    if not last.isdigit():
        return None
    supported = int(last) > 0
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"image": image, "arch": architecture, "supported": supported}),
                         encoding="utf-8")
    except OSError:
        pass
    return supported


def llamacpp_arch_support(profile: dict, slug: str, server: ServerInfo | None = None,
                          probe_docker: bool = False) -> dict | None:
    """รันไทม์ llama.cpp ของ bundle นี้รู้จักสถาปัตยกรรมของโมเดลไหม — ใช้ร่วมกันโดย doctor / inventory / repair

    คืน None เมื่อไม่ใช่ bundle llama.cpp หรือไม่รู้ arch เลย · ไม่งั้น
    {"arch", "mode", "supported": True/False/None, "runtime", "fix"} — supported=None คือ "ตอบไม่ได้" ไม่ใช่เสีย

    arch มาจากหัวไฟล์ GGUF ในเครื่องเมื่อโหลดแล้ว ไม่งั้นจากที่ renderer จดไว้ใน MODEL_PROFILE.yaml
    (model.gguf_architecture) — จึงบอกได้ *ก่อน* download ว่า build บนเครื่องนี้เก่ากว่าโมเดล
    """
    if (profile.get("runtime") or {}).get("engine") != "llamacpp":
        return None
    directory, wanted = _weight_paths(profile, slug)
    gguf = next((directory / name for name in wanted if name.endswith(".gguf")), None)
    architecture = _gguf_architecture(gguf) if gguf is not None and gguf.is_file() else ""
    architecture = architecture or str((profile.get("model") or {}).get("gguf_architecture") or "")
    if not architecture:
        return None

    mode = llamacpp_mode(profile, server)
    controller = (server.controller if server is not None else "") or f"./{slug}-single.sh"
    if mode == "native":
        root = llamacpp_root(profile)
        libs = _runtime_libs(root)
        supported = any(_lib_knows(lib, architecture) for lib in libs) if libs else None
        return {
            "arch": architecture, "mode": mode, "supported": supported,
            "runtime": f"llama.cpp {root}" + ("" if libs else " (ยังไม่ได้ build)"),
            "fix": f"LLAMA_CPP_UPDATE=1 {controller} prepare-runtime  (หรือ lmds repair {slug})",
        }
    from lmds.brain.allowlists import image_repo

    image = (profile.get("runtime") or {}).get("image") or ""
    pin = (profile.get("runtime") or {}).get("image_pin") or ""
    if not image:
        return None
    # ตรึงที่ digest เหมือน controller (LLAMACPP_IMAGE) · image_repo ไม่ตัด registry ที่มีพอร์ตขาด
    pinned = f"{image_repo(image)}@{pin}" if pin else image
    return {
        "arch": architecture, "mode": mode,
        "supported": _docker_knows_arch(pinned, architecture, slug, probe_docker),
        "runtime": f"image {pinned}",
        "fix": f"lmds set {slug} --image <image llama.cpp ใหม่กว่า>  แล้ว lmds start {slug}",
    }


def _check_architecture_llamacpp(profile: dict, slug: str, server: ServerInfo | None = None) -> list[Finding]:
    """llama.cpp build/image ตัวที่โมเดลนี้ผูกไว้ รู้จักสถาปัตยกรรมของมันไหม

    เคสจริง 2026-08-13: Muse-Glimmer-30B ใช้ architecture `muse-glimmer` ซึ่ง
    llama.cpp บน spark-head (23 ก.ค., ตามหลัง upstream 296 commit) ยังไม่รู้จัก
    ถ้าไม่ตรวจตรงนี้ ผู้ใช้จะโหลด 30 GB จบแล้วค่อยเจอตอน start ว่ารันไม่ได้

    เช็คเดิมข้ามทาง native ทั้งหมด (`if server.mode == "native": return []`)
    ทั้งที่ llama.cpp บน DGX Spark รัน native เป็นปกติ — เช็คที่มีอยู่จึงไม่เคย
    ทำงานกับ engine ที่ต้องการมันที่สุด

    2026-09-06 spark-worker (qwen4exp บน build 18 ส.ค.): คำแนะนำเดิม `git pull && cmake --build` ข้าม lock
    ของ controller — build ผ่านแล้ว prepare-runtime รอบถัดไปก็ย้อนกลับ · บอกทางที่ controller รู้จักแทน
    """
    support = llamacpp_arch_support(profile, slug, server, probe_docker=True)
    if support is None:
        return []
    architecture, runtime = support["arch"], support["runtime"]
    if support["supported"] is None:
        return [Finding(
            "architecture", Status.WARN,
            f"ตรวจไม่ได้ว่า {runtime} รู้จักสถาปัตยกรรม '{architecture}' ไหม"
            + (" — ยังไม่ได้ build llama.cpp (start จะ build ให้เอง)" if support["mode"] == "native" else " — ยังไม่ได้ pull image"),
        )]
    if support["supported"]:
        return [Finding("architecture", Status.OK, f"{architecture} ({runtime})")]
    return [Finding(
        "architecture", Status.FAIL,
        f"{runtime} ไม่รู้จักสถาปัตยกรรม '{architecture}' — รันไทม์เก่ากว่าโมเดล "
        f"(upstream llama.cpp เพิ่ม arch นี้ทีหลัง) · start จะตายตอนโหลดด้วย "
        f"\"unknown model architecture: '{architecture}'\"",
        support["fix"],
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
        return _check_architecture_llamacpp(profile, slug, server)
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
    """คืน (โฟลเดอร์ที่ควรมี weight, ไฟล์ที่**ขาดไม่ได้**) — projector อยู่ใน _projectors()

    safetensors: โฟลเดอร์คือ snapshot **ตัวที่ controller ของ bundle นี้จะใช้จริง** (ดู _hf_snapshot)
    """
    model = profile.get("model") or {}
    engine = (profile.get("runtime") or {}).get("engine")
    if engine == "llamacpp":
        wanted = [n.rsplit("/", 1)[-1] for n in [model.get("selected_gguf")] if n]
        return _model_dir(slug), wanted
    return _hf_snapshot(profile)[0], []


def _hf_repo_dir(profile: dict) -> Path:
    """โฟลเดอร์ `models--<org>--<name>` ของ bundle นี้ในแคช HF — รูปเดียวกับ `_model_cache_dir` ของ controller

    HF cache มีสองเลย์เอาต์: $HF_HOME/hub/models--X (ปัจจุบัน) และ $HF_HOME/models--X (เก่า) ·
    เคยดูแค่ hub/ — โมเดลที่โหลดด้วย HF รุ่นเก่าจึงขึ้นว่า "ยังไม่ download" ทั้งที่ไฟล์ครบทุกไฟล์
    (เจอจริงกับ DeepSeek V4 บน spark-head)
    """
    repo = ((profile.get("model") or {}).get("id") or "").replace("/", "--")
    home = _hf_home()
    for base in (home / "hub", home):
        if (base / f"models--{repo}" / "snapshots").is_dir():
            return base / f"models--{repo}"
    return home / "hub" / f"models--{repo}"


def _hf_snapshot(profile: dict) -> tuple[Path, bool]:
    """(snapshot ที่ controller ของ bundle นี้จะใช้, เป็นของ revision ที่ pin ไว้ไหม)

    เดิมตรงนี้ "ยอมรับ snapshot ที่มีอยู่จริงตัวใดก็ได้" เมื่อไม่เจอ revision ที่ pin — ซึ่งไม่ตรงกับ
    controller แบบ single เลย: `snapshot_dir()` ของมันคือ `snapshots/$MODEL_REVISION` ตรง ๆ และ
    `verify-files`/`start` ตายด้วย "ยังไม่ได้ download" ถ้าไม่มี · doctor จึงขึ้น ✅ weights ให้
    เครื่องที่ controller ตัวเดียวกันปฏิเสธ (audit 2026-10-06: snapshot เดียวที่มีเป็นของ revision
    อื่น มี config.json กับ blob .incomplete ทั้งที่ profile บอก 21 GB)

    controller แบบ stacked (`_snapshot_path`) ถอยจริง: revision ที่ pin → refs/<revision> →
    snapshot ตัวแรกที่เจอ · เดินตามลำดับเดียวกันเฉพาะ topology นั้น แล้วบอกผู้เรียกว่าไม่ใช่ตัวที่ pin
    """
    declared = str((profile.get("model") or {}).get("revision") or "")
    revision = declared or "main"
    repo_dir = _hf_repo_dir(profile)
    pinned = repo_dir / "snapshots" / revision
    if pinned.is_dir():
        return pinned, True
    if not declared:
        # profile ไม่ได้ pin revision ไว้เลย (profile เก่า/เขียนมือ) — ไม่มี revision ให้ยึด จึงใช้ตัวที่มี
        # อยู่ตามเดิม · inventory.hf_config พึ่งทางนี้อ่าน config.json ของโมเดลที่ไม่มี revision ใน profile
        try:
            existing = sorted(p for p in (repo_dir / "snapshots").iterdir() if p.is_dir())
        except OSError:
            existing = []
        return (existing[-1], True) if existing else (pinned, True)
    if profile.get("topology") != "stacked":
        return pinned, True
    try:
        commit = (repo_dir / "refs" / revision).read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    if commit and (repo_dir / "snapshots" / commit).is_dir():
        return repo_dir / "snapshots" / commit, True      # ref ชี้มาที่ commit นี้ = revision เดียวกัน
    try:
        others = sorted(p for p in (repo_dir / "snapshots").iterdir() if p.is_dir())
    except OSError:
        others = []
    return (others[0], False) if others else (pinned, True)


# รายการ shard + ขนาดจาก Hub ที่ renderer ฝังไว้ในหัว controller (vLLM/SGLang ทั้ง single และ stacked):
#
#     SHARD_FILES=(
#       "model-00001-of-00009.safetensors"
#     )
#     SHARD_SIZES=(
#       "4976698672"
#     )
#
# อ่านจาก controller ไม่ใช่จาก MODEL_PROFILE เพราะมันคือชุดเดียวกับที่ `verify-files` ใช้ตัดสิน —
# doctor ที่ตัดสินด้วยข้อมูลอีกชุดจะกลับมาเห็นต่างจาก controller อีกรอบ
_SHARD_ARRAY = re.compile(r'^SHARD_(FILES|SIZES)=\(\n((?:[ \t]*"[^"\n]*"[ \t]*\n)*)\)', re.MULTILINE)


def _controller_shards(controller: str) -> list[tuple[str, int | None]]:
    """[(ชื่อไฟล์, ขนาดเป็นไบต์ หรือ None เมื่อ Hub ไม่รายงาน)] — ว่าง = controller ไม่มีรายการ"""
    try:
        text = Path(controller).read_text(encoding="utf-8", errors="replace") if controller else ""
    except OSError:
        return []
    arrays = {kind: re.findall(r'"([^"\n]*)"', body) for kind, body in _SHARD_ARRAY.findall(text)}
    names = arrays.get("FILES") or []
    sizes = arrays.get("SIZES") or []
    return [(name, int(sizes[i]) if i < len(sizes) and sizes[i].isdigit() else None)
            for i, name in enumerate(names) if name]


def _incomplete_blobs(repo_dir: Path) -> tuple[int, int]:
    """(จำนวน, ไบต์รวม) ของ blob `.incomplete` — ร่องรอยของ download ที่ถูกขัดกลางทาง"""
    count = total = 0
    try:
        for blob in (repo_dir / "blobs").glob("*.incomplete"):
            try:
                total += blob.stat().st_size
                count += 1
            except OSError:
                continue
    except OSError:
        pass
    return count, total


def _gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f} GB" if num_bytes >= 1024**3 else f"{num_bytes / 1024**2:.1f} MB"


def _size_on_disk(path: Path) -> int | None:
    """ขนาดของไฟล์จริงหลังตาม symlink (snapshot ของ HF เป็น symlink ไป blobs/) · None = ไม่มี/ลิงก์ขาด"""
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _projectors(profile: dict) -> list[str]:
    """ไฟล์ mmproj ที่ profile ประกาศไว้ — เป็น **ทางเลือก** ไม่ใช่ของบังคับ

    llama-server รับ `--mmproj` ได้ไฟล์เดียว แต่ repo มักมีหลาย precision (BF16/F16/F32)
    ให้เลือก · profile รุ่นเก่าจึงลิสต์ไว้ทั้งหมด ทั้งที่ controller โหลดและใช้แค่ตัวเดียว
    """
    multimodal = (profile.get("features") or {}).get("multimodal") or {}
    return [p.rsplit("/", 1)[-1] for p in (multimodal.get("projector_files") or [])]




def _check_adopted_weights(profile: dict) -> list[Finding]:
    """weight ของ bundle ที่รับเข้าระบบ — ตรวจไฟล์ตรง path ที่เซิร์ฟเวอร์ใช้จริง"""
    model = (profile or {}).get("model") or {}
    raw = str(model.get("id") or "")
    if not raw.startswith(("/", "~", "./")):
        # adopt จาก container: path อยู่ *ข้างใน* container โฮสต์มองไม่เห็น ตรวจแทนไม่ได้
        return [Finding("weights", Status.OK,
                        "ผู้ใช้ดูแลเอง (bundle ที่รับเข้าระบบ) — LMDS ไม่ได้โหลดไฟล์นี้มา")]

    path = Path(raw).expanduser()
    if path.exists():
        size = path.stat().st_size / 1e9 if path.is_file() else 0
        detail = f"{path}" + (f" ({size:.1f} GB)" if size else "")
        return [Finding("weights", Status.OK, detail)]
    return [Finding(
        "weights", Status.FAIL,
        f"ไฟล์ที่เซิร์ฟเวอร์ถูกสั่งให้ใช้ไม่อยู่แล้ว: {path}",
        "bundle นี้รับเข้าระบบมา LMDS ไม่ได้เป็นคนโหลดไฟล์ จึงโหลดคืนให้ไม่ได้ — "
        "หาไฟล์กลับมาไว้ที่เดิม หรือ deploy ใหม่จากรุ่นบน Hugging Face",
    )]

def _check_hf_weights(profile: dict, slug: str, controller: str) -> list[Finding]:
    """weight ในแคช Hugging Face (vLLM/SGLang) — ครบตามที่ controller จะตรวจตอน start ไหม

    ✅ ได้ต่อเมื่อ: มี snapshot ที่ controller จะใช้ · shard ครบทุกไฟล์และขนาดตรงกับที่ Hub รายงาน
    (รายการเดียวกับ `verify-files`) หรือถ้า bundle ไม่มีรายการ ขนาดรวมบนดิสก์ต้องไม่น้อยกว่า
    `weight_bytes` ใน profile · `.incomplete` ที่ค้างอยู่ถูกรายงานเสมอ
    """
    model = profile.get("model") or {}
    revision = str(model.get("revision") or "main")
    directory, is_pinned = _hf_snapshot(profile)
    repo_dir = _hf_repo_dir(profile)
    partial_count, partial_bytes = _incomplete_blobs(repo_dir)
    partial = (f".incomplete ค้าง {partial_count} ไฟล์ ({_gb(partial_bytes)}) ใน {repo_dir / 'blobs'}"
               if partial_count else "")
    repair = f"lmds repair {slug}  (โหลด resume ได้ แล้วตรวจไฟล์ให้)"

    if not directory.is_dir():
        try:
            others = sorted(p.name for p in (repo_dir / "snapshots").iterdir() if p.is_dir())
        except OSError:
            others = []
        if not others and not partial:
            # ยังไม่เคยโหลดอะไรเลย — ข้อความเดิม · บอกคำสั่งระดับ lmds ก่อนเสมอ: ใช้ได้จากที่ไหนก็ได้
            # และเป็นปุ่มเดียวกับบนหน้าเว็บ (เดิมบอกให้ cd เข้า bundle ซึ่งผู้ใช้หน้าเว็บทำตามไม่ได้)
            return [Finding("weights", Status.FAIL, f"ยังไม่มีไฟล์โมเดลที่ {directory}", repair)]
        detail = f"ยังไม่มี snapshot ของ revision ที่ pin ไว้ ({revision}) ที่ {directory}"
        if others:
            shown = ", ".join(name[:14] for name in others[:3])
            detail += (f" — ที่มีอยู่เป็นของ revision อื่น ({shown}) ซึ่ง controller ไม่ใช้ "
                       f"(start จะตอบว่า \"ยังไม่ได้ download\")")
        if partial:
            detail += f" · download ถูกขัดกลางทาง: {partial}"
        return [Finding("weights", Status.FAIL, detail, repair)]

    shards = _controller_shards(controller)
    problems: list[str] = []
    if shards:
        missing = [name for name, _ in shards if _size_on_disk(directory / name) is None]
        if missing:
            problems.append(f"shard ขาด {len(missing)} จาก {len(shards)} ไฟล์: {', '.join(missing[:3])}"
                            + (" …" if len(missing) > 3 else ""))
        wrong = [(name, got, want) for name, want in shards
                 if want is not None and (got := _size_on_disk(directory / name)) is not None and got != want]
        if wrong:
            name, got, want = wrong[0]
            problems.append(f"ขนาดไม่ตรงกับที่ Hub รายงาน {len(wrong)} ไฟล์ — {name}: ได้ {got:,} ต้องการ {want:,} ไบต์")
        verified = f"shard ครบ {len(shards)} ไฟล์ ขนาดตรง"
    else:
        # bundle นี้ไม่มีรายการ shard (Hub ไม่ได้ให้มา/ไฟล์เดียว) — เทียบขนาดรวมกับที่ profile จดไว้
        weights = [size for path in directory.rglob("*")
                   if path.suffix in (".safetensors", ".bin", ".gguf", ".pt")
                   and (size := _size_on_disk(path)) is not None]
        on_disk = sum(weights)
        expected = model.get("weight_bytes")
        expected = expected if isinstance(expected, int) and expected > 0 else 0
        if not weights:
            problems.append("snapshot นี้ไม่มีไฟล์ weight เลย (มีแต่ config) — download ไม่ครบ")
        elif expected and on_disk < expected:
            problems.append(f"download ไม่ครบ: บนดิสก์ {_gb(on_disk)} จากที่ profile บันทึกไว้ {_gb(expected)}")
        verified = (f"weight {len(weights)} ไฟล์ {_gb(on_disk)}"
                    + ("" if expected else " (bundle นี้ไม่มีรายการ shard/ขนาดให้เทียบ — ตรวจได้แค่ว่ามีไฟล์)"))

    # ไฟล์ขนาด 0 ไบต์ในชั้นบนของ snapshot (config/tokenizer ที่โหลดขาด) — กติกาเดิม
    empty = [p.name for p in directory.glob("*") if _size_on_disk(p) == 0]
    if empty:
        problems.append(f"มีไฟล์ขนาด 0 ไบต์: {', '.join(empty[:3])}")

    if problems:
        if partial:
            problems.append(f"download ถูกขัดกลางทาง: {partial}")
        return [Finding("weights", Status.FAIL, " · ".join(problems) + f" ({directory})",
                        f"lmds repair {slug}  (โหลดเฉพาะส่วนที่ขาด)")]

    notes: list[str] = []
    if not is_pinned:
        notes.append(f"controller (stacked) จะใช้ snapshot ของ revision {directory.name[:14]} "
                     f"เพราะไม่มีของ revision ที่ pin ไว้ ({revision}) — คนละ commit กับที่วางแผนไว้")
    if partial:
        notes.append(f"{partial} — เศษของ download ที่ถูกขัด ไฟล์ที่ต้องใช้ครบแล้ว แต่ยังกินดิสก์อยู่")
    if notes:
        fix = (repair if not is_pinned else
               f"ถ้าไม่มี download ของ repo นี้กำลังรันอยู่ ลบได้: rm {repo_dir / 'blobs'}/*.incomplete")
        return [Finding("weights", Status.WARN, f"{directory} · {verified} · " + " · ".join(notes), fix)]
    return [Finding("weights", Status.OK, f"{directory} · {verified}")]


def _check_weights(profile: dict, slug: str, controller: str = "") -> list[Finding]:
    # bundle ที่มาจาก `lmds adopt` ชี้ weight ไปที่ path เดิมของเจ้าของ ไม่ใช่ ~/models/<slug>
    # ตามธรรมเนียม LMDS · ตรวจด้วยกติกาปกติจะขึ้น "ยังไม่มีไฟล์โมเดล" ตลอดกาลทั้งที่เซิร์ฟเวอร์
    # กำลังเสิร์ฟไฟล์นั้นอยู่ แล้วยังแนะ `lmds repair` ซึ่ง controller ของ adopt ไม่มีคำสั่งนั้น
    # — คำแนะนำที่ทำตามแล้วล้มแน่นอนแย่กว่าไม่แนะอะไรเลย
    if self_managed_weights(profile):
        return _check_adopted_weights(profile)

    if (profile.get("runtime") or {}).get("engine") != "llamacpp":
        return _check_hf_weights(profile, slug, controller)

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

    # ข้ามไฟล์ทำงานของ LMDS เอง — `.download.lock` เป็น flock ที่ controller สร้างด้วย
    # `exec 9>` จึง **ขนาด 0 เสมอโดยธรรมชาติ** ไม่ใช่อาการเสีย (ดู single-llamacpp-controller
    # .sh.j2 และ fleet/clone.py ที่ exclude ไฟล์นี้อยู่แล้วเพราะถือว่าเป็นของชั่วคราว)
    #
    # เคสจริง 2026-09-20 บนเครื่องลูกค้า: repair เสร็จแล้วทิ้ง lock ไว้ → doctor ฟ้อง
    # "ไฟล์ขนาด 0 ไบต์" → แนะให้ repair → repair โหลด weight 21 GB ใหม่และหยุดโมเดลที่รันอยู่
    # → จบแล้วทิ้ง lock อีก → วนอยู่อย่างนั้น · weight ทั้งสองไฟล์มี .sha256-ok ครบตลอดทาง
    ignore = {".download.lock"}
    empty = [p.name for p in directory.glob("*")
             if p.is_file() and p.stat().st_size == 0 and p.name not in ignore]
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


def _listening_on(port: int) -> str | None:
    """บรรทัดของ ss/netstat ที่ฟัง port นี้ · "" = ตรวจแล้วไม่มีใครฟัง · **None = ไม่ได้ตรวจ**

    เดิมคืน "" ทั้งสองกรณี — เครื่องที่ไม่มีทั้ง ss และ netstat (หรือมีแต่รันไม่ผ่าน เช่น netstat
    ของ macOS ที่ไม่รู้จัก -tlnp) จึงได้ "✅ port 8000 ว่าง" ทั้งที่ไม่มีใครได้ดูเลย (audit 2026-10-06)
    """
    checked = False
    for cmd in (["ss", "-tlnp"], ["netstat", "-tlnp"]):
        if shutil.which(cmd[0]) is None:
            continue
        code, out = _run(cmd)
        if code != 0:
            continue
        checked = True
        for line in out.splitlines():
            if f":{port} " in line:
                return line.strip()
    return "" if checked else None


def _free_port(start: int) -> int:
    """port ถัดไปที่ว่างจริง — แนะเลขตายตัวแล้วเจอว่าไม่ว่างอีกคือทำให้เสียเวลาเปล่า"""
    for candidate in range(max(start, 8000) + 1, max(start, 8000) + 40):
        if not _listening_on(candidate):
            return candidate
    return start + 1


def _check_port(server: ServerInfo) -> list[Finding]:
    # `lmds set --port` เขียน bundle.env ซึ่ง controller อ่านก่อนตั้ง default — ตรวจด้วยค่าใน
    # profile จึงฟ้อง conflict ที่ไม่เกิดจริง แล้วแนะให้ย้ายไป port ที่อาจไม่ว่างอีกตัว
    from lmds.fleet.manager import effective_autostart_port

    port = server.port
    if not server.running:
        saved = effective_autostart_port(server)
        if saved and saved.isdigit():
            port = int(saved)
    if not port:
        return []
    server = replace(server, port=port)
    holder = _listening_on(server.port)
    if holder is None:
        # ไม่ได้ตรวจ ≠ ว่าง — ขึ้นเขียวตรงนี้คือบอกผู้ใช้ว่า start ได้ ทั้งที่ไม่มีใครดูว่ามีอะไรยึดพอร์ตอยู่ไหม
        return [Finding("port", Status.WARN,
                        f"ตรวจไม่ได้ว่ามีใครฟัง port {server.port} อยู่ไหม — เครื่องนี้ไม่มี ss/netstat ที่ใช้ได้",
                        "ติดตั้งแล้วตรวจใหม่: sudo apt install iproute2   (ให้คำสั่ง ss)")]
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
                            f"lmds stop {rival}   หรือรันคู่กันคนละ port: "
                            f"lmds set {server.slug} --port {_free_port(server.port)}")]
        return [Finding("port", Status.FAIL, f"port {server.port} ถูกใช้โดยโปรเซสอื่น: {holder[:100]}",
                        f"หยุดตัวที่ชน หรือย้าย port: lmds set {server.slug} --port {_free_port(server.port)}")]
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


def _check_controller_age(profile: dict, server: ServerInfo) -> list[Finding]:
    """controller เก่ากว่า template ของ lmds บนเครื่องนี้ไหม — bundle ที่ Update ไม่เคยแตะ (audit 2026-09-06)

    เทียบ template_hash ที่ renderer ฝังไว้ ไม่ใช่เลข version · ไม่มี hash (render ก่อน 0.6.1) = เก่าแน่ ·
    adopted = ไม่มี template ให้ regenerate (ไม่ใช่ปัญหา) · แก้ด้วย `lmds bundles refresh <slug>` (ออฟไลน์)
    """
    from lmds.fleet.consistency import controller_state

    state = controller_state(profile, server.controller if server.controller_exists else None)
    if state["state"] == "ok":
        return [Finding("controller-stale", Status.OK, f"controller ตรง template ของ lmds (lmds {state['generated_by']})")]
    if state["state"] == "adopted":
        return [Finding("controller-stale", Status.OK, "adopted — ไม่มี template ให้ regenerate")]
    if state["state"] in ("stale", "ahead"):
        return [Finding(
            "controller-stale", Status.WARN,
            f"controller {'เก่ากว่า' if state['state'] == 'stale' else 'ใหม่กว่า'} lmds บนเครื่องนี้ — {state['reason']} "
            f"· ไม่มีคำสั่ง/ตัวกันพลาดที่เพิ่มมาทีหลัง (เช่น check-runtime, explain_crash)",
            f"lmds bundles refresh {server.slug}   (ออฟไลน์ · เก็บของเดิมเป็น .replaced-* · bundle.env คงเดิม"
            f"{' · ตัวที่รันอยู่ใช้ controller ใหม่เมื่อ restart' if server.running else ''})",
        )]
    return [Finding("controller-stale", Status.WARN, f"ตรวจไม่ได้ว่า controller เก่าไหม — {state['reason']}",
                    f"lmds rebuild {server.slug}")]


def _check_role() -> list[Finding]:
    """เครื่องนี้มีไว้รันโมเดล หรือมีไว้สร้าง bundle แล้วส่งต่อ

    ต้องมาก่อนข้ออื่นเพราะมันเปลี่ยนความหมายของทุกข้อที่ตามมา: บน control plane
    "ไม่มี docker" กับ "ไม่พบ libllama.so" ไม่ใช่ปัญหาที่ต้องแก้ — มันคือนิยามของเครื่อง
    """
    from lmds.hardware import serving

    capability = serving.detect()
    if capability.can_serve:
        return [Finding("บทบาท", Status.OK,
                        f"เครื่องรันโมเดล — engine: {', '.join(capability.engines)}")]
    return [Finding(
        "บทบาท", Status.WARN,
        f"control plane — {capability.evidence()}",
        "สร้าง bundle ที่นี่แล้วส่งไปรันที่อื่น: lmds node push <เครื่อง> <slug> --download",
    )]


def _demote_for_control_plane(findings: list[Finding]) -> list[Finding]:
    """บน control plane ข้อที่บอกว่า "รันไม่ได้" ไม่ใช่ของเสีย — ไม่ควรนับเป็นตัวบล็อก

    เดิม `lmds doctor` บน hub VM สรุปว่า "พบ 1 ข้อที่ต้องแก้ก่อนถึงจะรันได้" (docker)
    ซึ่งชวนให้ไปติดตั้ง docker บนเครื่องที่ไม่มี GPU ให้มันใช้อยู่ดี
    """
    from lmds.hardware import serving

    if serving.detect().can_serve:
        return findings
    # runtime ที่ไม่มี = นิยามของเครื่อง ไม่ใช่ของเสีย · weight ที่ยังไม่โหลด = ปกติ
    # เพราะ bundle บน hub มีไว้ push ต่อ ไม่ได้มีไว้รันที่นี่
    softened = {"docker", "image", "architecture", "grammar", "weights", "server"}
    push_instead = "ส่งไปรันที่เครื่องอื่น: lmds node push <เครื่อง> <slug> --download"
    demoted = []
    for finding in findings:
        if finding.name in softened and finding.status is Status.FAIL:
            fix = push_instead if finding.name in ("weights", "server") else (
                finding.fix or "ไม่ต้องแก้ที่นี่ — เครื่องนี้ไม่ได้มีไว้รันโมเดล")
            finding = Finding(finding.name, Status.WARN, finding.detail, fix)
        elif finding.name in ("weights", "server") and finding.fix:
            # คำแนะนำเดิมชี้ไป repair/start ซึ่งเป็นคำสั่งที่เครื่องนี้ปฏิเสธอยู่แล้ว
            finding = Finding(finding.name, finding.status, finding.detail, push_instead)
        demoted.append(finding)
    return demoted


# ที่อยู่ที่แปลว่า "เฉพาะในเครื่องนี้" — ทุกตัวที่ไม่อยู่ในนี้ = คนอื่นในวง network ยิงถึง
_LOCAL_BINDS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _reads_key_store(controller: str) -> bool:
    """controller ตัวนี้ไปอ่าน ~/.lmds/keys/<slug> เองไหม

    อ่านจากตัวสคริปต์ ไม่ใช่เดาจากเวอร์ชัน — bundle ที่ adopt มาไม่มี template ให้
    regenerate และไม่ได้เดินตามเลขเวอร์ชันของ lmds เลย
    """
    try:
        return "LMDS_KEY_ROOT" in Path(controller).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _probe_host(bind: str) -> str:
    """ที่อยู่ที่จะยิงไปหาเซิร์ฟเวอร์ของ bundle นี้ จากเครื่องเดียวกัน

    ผูกทุก interface (0.0.0.0 / ว่าง / ::) → loopback · ผูก IP เจาะจง → IP นั้น เพราะเซิร์ฟเวอร์ที่ผูก
    IP เดียวไม่ฟังที่ 127.0.0.1 — ยิง loopback แล้วสรุปว่า "ต่อไม่ติด" คือวินิจฉัยผิดตัว
    """
    host = (bind or "").strip().strip("[]")
    if host in ("", "0.0.0.0", "*"):
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if ":" in host else host


def _answers_without_key(host: str, port: int) -> tuple[str, str]:
    """ยิงเซิร์ฟเวอร์ที่รันอยู่หนึ่งรอบ **โดยไม่ส่ง key** — ("enforced" | "open" | "unknown", สิ่งที่เห็น)

    อ่านอย่างเดียว ไม่มีคำขอไหนไปถึงโมเดล (GET ล้วน) · สองขั้นเพราะ engine วาง key ไว้คนละที่:

      vLLM / SGLang  ทุก path ใต้ /v1 อยู่หลัง key → `/v1/models` ตอบ 401 ถ้าบังคับ
      llama.cpp      `/v1/models` กับ `/health` **เปิดสาธารณะเสมอ** แม้ตั้ง --api-key ·
                     ตัวที่อยู่หลัง key คือ `/props` (และทุก endpoint ที่ทำงานจริง)

    ดู 200 จาก `/v1/models` อย่างเดียวจึงกล่าวหา llama-server ที่ป้องกันถูกต้องแล้วทุกตัวว่าเปิดโล่ง
    """
    import httpx

    base = f"http://{host}:{port}"
    try:
        # trust_env=False: ห้ามให้ HTTP_PROXY ของเครื่องพาคำขอ loopback ออกไปหา proxy
        with httpx.Client(timeout=3.0, trust_env=False) as client:
            models = client.get(f"{base}/v1/models").status_code
            if models in (401, 403):
                return "enforced", f"GET /v1/models ไม่มี key → {models}"
            if models != 200:
                return "unknown", f"GET /v1/models → {models}"
            props = client.get(f"{base}/props").status_code
            if props in (401, 403):
                return "enforced", f"GET /props ไม่มี key → {props}"
            return "open", "GET /v1/models ไม่มี key → 200"
    except httpx.HTTPError as exc:
        return "unknown", f"ยิง {base} ไม่ติด ({type(exc).__name__})"


def _check_open_endpoint(server: ServerInfo) -> list[Finding]:
    """เปิดให้ทั้งวง network โดยไม่ต้องยืนยันตัวตนไหม

    ค่าเริ่มต้นของ controller คือ bind 0.0.0.0 ซึ่งถูกสำหรับคลัสเตอร์ (head ต้องคุย worker)
    และสำหรับ gateway ที่อยู่คนละเครื่อง — ที่ไม่ถูกคือ *ไม่มี key* คู่กับมัน

    controller เตือนเรื่องนี้ตอน start อยู่แล้ว แต่ข้อความนั้นเลื่อนหายไปกับ log ของการ
    บูตเครื่อง · doctor คือที่ที่คนมาดูตอนสงสัย จึงต้องบอกซ้ำตรงนี้ด้วย

    **✅ ได้ทางเดียว: เห็นเซิร์ฟเวอร์ที่รันอยู่ปฏิเสธคำขอที่ไม่มี key** · เดิมเขียวเพราะ *มีไฟล์ key*
    ซึ่งเป็นสถานะของดิสก์ ไม่ใช่ของเซิร์ฟเวอร์: หลัง `lmds key new` บนเซิร์ฟเวอร์ที่เปิดอยู่ มันยัง
    ตอบทุกคนจนกว่าจะ restart ขณะที่ doctor บอกว่า "มี API key เก็บไว้" (audit 2026-10-06) ·
    ไม่ได้รัน = ไม่มีเซิร์ฟเวอร์ให้ถาม จึงบอกสิ่งที่รู้จากไฟล์ แต่ไม่ขึ้นเขียวให้ข้ออ้างที่ไม่ได้ตรวจ
    """
    from lmds.fleet import apikey
    from lmds.fleet.manager import _bundle_env_value

    slug = server.slug
    bind = _bundle_env_value(Path(server.controller).parent, "API_HOST") or "0.0.0.0"
    if bind in _LOCAL_BINDS:
        return [Finding("endpoint", Status.OK, f"ผูกกับ {bind} — เข้าถึงได้เฉพาะในเครื่องนี้")]

    has_key = bool(apikey.read(slug))
    # มีไฟล์ key ไม่ได้แปลว่า controller หยิบไปใช้ได้ · controller ที่ render ก่อนรุ่นนี้
    # และ bundle ที่ adopt มาจาก container ซึ่งไม่มีตัวแปรชื่อ *API_KEY* ให้เติม
    # ยังเสิร์ฟแบบเปิดอยู่ทั้งที่ `lmds key show` บอกว่ามี
    usable = has_key and _reads_key_store(server.controller)
    unusable = Finding(
        "endpoint", Status.WARN,
        f"ผูกกับ {bind} · มี API key เก็บไว้แต่ controller ตัวนี้หยิบไปใช้ไม่ได้ — ยังเสิร์ฟแบบเปิดอยู่",
        f"regenerate ให้รู้จักที่เก็บ: lmds bundles refresh {slug} · "
        "bundle ที่ adopt มาแล้วไม่มีตัวแปรชื่อ *API_KEY* ต้องตั้ง auth ที่คำสั่งของ engine เอง",
    )
    no_key = Finding(
        "endpoint", Status.WARN,
        f"ผูกกับ {bind} โดยไม่มี API key — ใครก็ตามที่ถึงเครื่องนี้ใช้โมเดลได้โดยไม่ต้องยืนยันตัวตน",
        f"ตั้ง key: lmds key new {slug} แล้ว lmds restart {slug} · "
        f"หรือถ้าตั้งใจให้ใช้เฉพาะในเครื่อง: lmds set {slug} --bind 127.0.0.1",
    )

    if not server.running or not server.port:
        if not has_key:
            return [no_key]
        if not usable:
            return [unusable]
        return [Finding(
            "endpoint", Status.WARN,
            f"ผูกกับ {bind} · มี API key เก็บไว้และ controller จะใช้ตอน start — แต่ยังไม่ได้รัน "
            "จึงยังไม่ได้ยืนยันกับเซิร์ฟเวอร์จริงว่าบังคับ key",
            f"lmds start {slug} แล้วรัน lmds doctor {slug} อีกครั้ง",
        )]

    verdict, seen = _answers_without_key(_probe_host(bind), server.port)
    if verdict == "enforced":
        if has_key:
            return [Finding("endpoint", Status.OK,
                            f"ผูกกับ {bind} · เซิร์ฟเวอร์ปฏิเสธคำขอที่ไม่มี key ({seen})")]
        # ถูก start ด้วย API_KEY= ของผู้ใช้เอง — วันนี้ปลอดภัย แต่ systemd ตอน autostart เรียก
        # controller เปล่า ๆ: reboot แล้วกลับมาเปิดโล่งโดยไม่มีอะไรบอก (เหตุผลที่มี fleet/apikey.py)
        return [Finding(
            "endpoint", Status.WARN,
            f"ผูกกับ {bind} · เซิร์ฟเวอร์บังคับ key อยู่ ({seen}) แต่ไม่มี key เก็บไว้กับเครื่อง — "
            "autostart หลัง reboot จะกลับมาเสิร์ฟแบบเปิด",
            f"เก็บ key ตัวที่ใช้อยู่: lmds key set {slug}",
        )]
    if verdict == "open":
        if not has_key:
            return [no_key]
        if not usable:
            return [unusable]
        return [Finding(
            "endpoint", Status.WARN,
            f"ผูกกับ {bind} · มี API key เก็บไว้ แต่เซิร์ฟเวอร์ที่รันอยู่ยังไม่บังคับ ({seen}) — "
            "key ถูกตั้งหลัง start จึงยังไม่มีผล ตอนนี้ใครถึงเครื่องนี้ก็ใช้โมเดลได้",
            f"lmds restart {slug}   (restart เพื่อให้ key มีผล)",
        )]
    state = ("มี API key เก็บไว้" if usable else
             "มี API key เก็บไว้แต่ controller ตัวนี้หยิบไปใช้ไม่ได้" if has_key else "ไม่มี API key เก็บไว้")
    return [Finding(
        "endpoint", Status.WARN,
        f"ผูกกับ {bind} · {state} · ตรวจไม่ได้ว่าเซิร์ฟเวอร์ที่รันอยู่บังคับ key ไหม — {seen}",
        f"โมเดลอาจกำลังโหลดอยู่: lmds logs {slug} -f แล้วรัน lmds doctor {slug} อีกครั้ง",
    )]


def diagnose(slug: str) -> Diagnosis:
    """ตรวจทุกข้อของ slug เดียว — ไม่แก้อะไรให้เอง แค่บอกสาเหตุกับคำสั่ง"""
    server = find(slug)
    if server is None:
        return Diagnosis(slug, [Finding(
            "bundle", Status.FAIL, f"ไม่รู้จัก '{slug}'",
            "ดูรายชื่อที่มี: lmds list",
        )])

    result = Diagnosis(slug)
    result.findings.extend(_check_role())
    result.findings.extend(_check_controller(server))

    profile = bundle_profile(server.controller) or {}
    if not profile:
        result.findings.append(Finding(
            "profile", Status.WARN, "อ่าน MODEL_PROFILE.yaml ไม่ได้ — ตรวจได้ไม่ครบทุกข้อ",
            "ไฟล์อยู่ข้าง controller ใน bundle เดียวกัน",
        ))
    else:
        result.findings.extend(_check_controller_age(profile, server))
        result.findings.extend(_check_hf_token(profile))
        result.findings.extend(_check_weights(profile, slug, server.controller))
        result.findings.extend(_check_permissions(profile, slug))
        result.findings.extend(_check_disk(profile, slug))
        result.findings.extend(_check_docker(profile, server))
        result.findings.extend(_check_image(profile, server))
        result.findings.extend(_check_architecture(profile, server, slug))
        result.findings.extend(_check_llamacpp_grammar(profile, slug))

    result.findings.extend(_check_port(server))
    result.findings.extend(_check_open_endpoint(server))
    result.findings.extend(_check_server(server))
    result.findings = _demote_for_control_plane(result.findings)
    return result
