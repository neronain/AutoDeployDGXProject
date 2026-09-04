"""รับ container ที่รันอยู่ก่อน LMDS เข้ามาอยู่ในระบบ

ลูกค้าจำนวนมากมี vLLM/llama.cpp รันอยู่ก่อนแล้วเพิ่งมาติดตั้ง LMDS ทีหลัง · `lmds ps`
มองเห็น container พวกนั้นและ stop/restart/logs ได้ แต่ทำอย่างอื่นไม่ได้เลย เพราะไม่มี
controller — กด repair ก็ได้แต่คำว่า "ไม่พบ controller"

ตัวนี้อ่านสิ่งที่ container กำลังใช้อยู่จริง (image, env, mount, port, args) แล้วเขียนเป็น
controller ที่ **รันคำสั่งเดิมซ้ำได้เป๊ะ** — ของที่รันอยู่ไม่ถูกแตะต้อง

**ไม่ใช่ทุกเครื่องที่รันด้วย Docker** — เคสที่เจอบ่อยพอ ๆ กันคือ `llama-server` ที่รันตรง ๆ
ใต้ systemd unit ที่ลูกค้าเขียนเอง · `lmds ps` มองเห็นมันอยู่แล้ว (`_orphan_native` อ่าน
cmdline) แต่ก็ตันตรงเดียวกันคือไม่มี controller · `inspect_process` ทำเรื่องเดียวกันกับ
process แทน container

หลักที่ยึด:
  - **สร้างจากสิ่งที่รันอยู่จริง ไม่ใช่เดา** — อ่านจาก `docker inspect` ตรง ๆ
  - **ไม่แกล้งทำเป็นมี `download`/`verify-files`** — weight ของ container พวกนี้เป็น path
    ที่ผู้ใช้จัดการเอง ไม่ได้มาจาก Hugging Face ที่เรารู้จัก · คำสั่งที่ทำอะไรไม่ได้จริง
    แต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มีคำสั่งนั้น
"""

from __future__ import annotations

import re

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .manager import FleetError, run_root

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def _check_slug(slug: str) -> str:
    """slug กลายเป็นชื่อโฟลเดอร์ ชื่อไฟล์ controller และชื่อ container — ต้องไม่มี / .. ช่องว่าง
    เดิมรับอะไรก็ได้: `--slug ../../x` เขียน controller นอก bundles/ ได้ (รีวิว 2026-09-04)
    """
    if not _SLUG_RE.fullmatch(slug or ""):
        raise FleetError(
            f"slug '{(slug or '')[:40]}' ใช้ไม่ได้ — ใช้ตัวพิมพ์เล็ก ตัวเลข และ . _ - (ไม่เกิน 63 ตัว)")
    return slug


def _derive_slug(text: str) -> str:
    """ชื่อ container/โมเดลที่ไม่ได้ขอ slug มา — บีบให้เข้ารูป slug (org/model → org-model)"""
    derived = re.sub(r"[^a-z0-9.-]+", "-", (text or "").lower()).strip("-.")[:63]
    return _check_slug(derived or "adopted")


@dataclass
class Adopted:
    """สิ่งที่อ่านได้จาก container ที่รันอยู่"""

    container: str
    image: str
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)
    ports: dict = field(default_factory=dict)
    network: str = ""
    runtime: str = ""
    entrypoint: list[str] = field(default_factory=list)
    ipc_mode: str = ""
    shm_size: int = 0

    @property
    def argv_tokens(self) -> list[str]:
        """คำสั่งของ container เป็น token — แตะสตริงที่ถูกห่อด้วย shell ออกมาด้วย

        image จำนวนมากสั่งงานผ่าน `bash -c "โน่นนี่ && เซิร์ฟเวอร์ --port 8355 …"` ทำให้
        argv ทั้งชุดยุบเหลือสามชิ้น (`bash`, `-c`, สตริงยาว) · การไล่หา `--port` แบบ
        เทียบทีละชิ้นจึงไม่มีวันเจอ ทั้งที่มันอยู่ในนั้น

        เจอจริง 2026-08-27 บน spark-03 (nvidia/tensorrt-llm:nemotron-fixed2)
        """
        tokens: list[str] = []
        for item in list(self.entrypoint) + list(self.args):
            if " " not in item:
                tokens.append(item)
                continue
            try:
                tokens.extend(shlex.split(item))
            except ValueError:
                tokens.extend(item.split())
        return tokens

    @property
    def port(self) -> int:
        for item in self.env:
            if item.startswith("PORT="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    break
        # --port บน argv คือคำสั่งที่เซิร์ฟเวอร์รับไปจริง ๆ ส่วน PortBindings เป็นแค่รูที่
        # เปิดไว้ ซึ่งมีได้หลายรูโดยที่ API อยู่รูเดียว
        #
        # เจอจริง 2026-08-27 บน spark-03: container เปิด 6006/8355/8888 (metrics, API,
        # notebook) · adopt คว้า 6006 มาเป็น port ของโมเดล แล้ว `lmds ps` ก็ค้างที่
        # "loading" ตลอดกาลเพราะ health check ไปเคาะผิดรู
        argv = self.argv_tokens
        for flag in ("--port", "-p", "--server-port"):
            if flag in argv:
                index = argv.index(flag) + 1
                if index < len(argv):
                    try:
                        return int(argv[index])
                    except ValueError:
                        break
        for spec in self.ports or {}:
            try:
                return int(spec.split("/")[0])
            except ValueError:
                continue
        return 0

    @property
    def model(self) -> str:
        for key in ("MODEL=", "MODEL_ID=", "MODEL_PATH=", "MODEL_HANDLE="):
            for item in self.env:
                if item.startswith(key):
                    return item.split("=", 1)[1]
        return self._model_from_argv()

    def _model_from_argv(self) -> str:
        """เซิร์ฟเวอร์หลายตัวรับชื่อโมเดลทาง argv ไม่ใช่ env — อ่านจาก env อย่างเดียวจึงแจ้ง
        "(ไม่ระบุใน env)" ทั้งที่ชื่ออยู่ตรงหน้า

        เจอจริง 2026-08-27 บน spark-03: `trtllm-serve
        nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --host 0.0.0.0 --port 8355`
        """
        argv = self.argv_tokens
        # `-m` ต้องมาท้ายสุด: `python3 -m sglang.launch_server` ทำให้ `-m` กลายเป็น
        # ชื่อโมดูล ไม่ใช่โมเดล — เจอจริง 2026-09-01 หน้าเว็บขึ้นชื่อรุ่นว่า
        # "sglang.launch_server" · ฝั่ง llama.cpp ที่ใช้ `-m` จริงยังตกมาถึงอยู่ดี
        for flag in ("--model-path", "--model_path", "--model", "-m"):
            if flag in argv:
                index = argv.index(flag) + 1
                if index < len(argv):
                    return argv[index]
        # ชื่อที่วางเป็น positional ตามหลังคำสั่ง serve — รับเฉพาะรูป org/name หรือ path
        for previous, item in zip(argv, argv[1:]):
            if previous.endswith(("serve", "-serve")) and not item.startswith("-"):
                return item
        return ""

    @property
    def context(self) -> int:
        for key in ("MAX_MODEL_LEN=", "CTX_SIZE="):
            for item in self.env:
                if item.startswith(key):
                    try:
                        return int(item.split("=", 1)[1])
                    except ValueError:
                        break
        # หลายเซิร์ฟเวอร์รับ context ทาง argv ไม่ใช่ env — SGLang ใช้ --context-length
        # ดูแต่ env อย่างเดียวจึงได้ 0 แล้วหน้าเว็บไม่โชว์ context ให้เลย
        value = _argv_value(self.argv_tokens, "--context-length", "--max-model-len",
                            "--ctx-size", "-c")
        return int(value) if value.isdigit() else 0

    @property
    def engine(self) -> str:
        """เดาเครื่องยนต์จาก **คำสั่งที่รันจริง** ก่อน แล้วค่อยดูชื่อ image

        ของเดิมดูแต่ชื่อ image และรู้จักคำเดียวคือ "vllm" — container ที่ชื่อ image
        ไม่มีคำนั้นจึงขึ้น engine=unknown ทั้งหมด เจอจริง 2026-09-01: MiniMax M3 บน
        image `scitrera/dgx-spark-sglang-mm:v0` ขึ้น "unknown" ในหน้าเว็บทั้งที่คำสั่ง
        เขียนว่า `python3 -m sglang.launch_server` ชัด ๆ
        """
        haystack = f"{self.image} {' '.join(self.argv_tokens)} {' '.join(self.entrypoint)}".lower()
        for needle, engine in (("sglang", "sglang"), ("vllm", "vllm"),
                               ("llama-server", "llamacpp"), ("llama.cpp", "llamacpp"),
                               ("llamacpp", "llamacpp"), ("trtllm", "tensorrt-llm")):
            if needle in haystack:
                return engine
        return "unknown"


def inspect_container(container: str) -> Adopted:
    """อ่านทุกอย่างที่ต้องใช้เพื่อรันซ้ำ — ล้มเหลวชัด ๆ ถ้าไม่มี container นั้น"""
    try:
        proc = subprocess.run(["docker", "inspect", container],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FleetError(f"เรียก docker inspect ไม่ได้: {exc}") from exc
    if proc.returncode != 0:
        raise FleetError(f"ไม่พบ container '{container}' — ดูรายชื่อ: docker ps")
    data = json.loads(proc.stdout)[0]
    config, host = data.get("Config") or {}, data.get("HostConfig") or {}
    return Adopted(
        container=data["Name"].lstrip("/"),
        image=config.get("Image") or "",
        args=list(data.get("Args") or []),
        env=list(config.get("Env") or []),
        binds=list(host.get("Binds") or []),
        ports=dict(host.get("PortBindings") or {}),
        network=host.get("NetworkMode") or "",
        runtime=host.get("Runtime") or "",
        entrypoint=list(config.get("Entrypoint") or []),
        ipc_mode=host.get("IpcMode") or "",
        shm_size=int(host.get("ShmSize") or 0),
    )



def _host_path(adopted: "Adopted", container_path: str) -> Path | None:
    """แปลง path ฝั่งคอนเทนเนอร์กลับเป็น path บนเครื่อง โดยใช้ -v ที่มันถูกรันมา

    ไม่มีตัวนี้ = อ่าน config.json ของโมเดลไม่ได้เลย เพราะ path ที่ adopt เห็น
    (เช่น /cache/models--org--m/snapshots/abc) มีอยู่แค่ในคอนเทนเนอร์
    """
    if not container_path.startswith("/"):
        return None
    best: tuple[int, Path] | None = None
    for bind in adopted.binds:
        parts = bind.split(":")
        if len(parts) < 2:
            continue
        host_dir, cont_dir = parts[0], parts[1]
        if container_path == cont_dir or container_path.startswith(cont_dir.rstrip("/") + "/"):
            rest = container_path[len(cont_dir.rstrip("/")):].lstrip("/")
            candidate = Path(host_dir) / rest if rest else Path(host_dir)
            if best is None or len(cont_dir) > best[0]:
                best = (len(cont_dir), candidate)
    if best:
        return best[1]
    direct = Path(container_path)
    return direct if direct.exists() else None


def _features_from_model(adopted: "Adopted") -> dict:
    """อ่านความสามารถจาก config.json ของโมเดลจริง — ไม่ได้เดาจากชื่อ

    หน้าเว็บติดป้าย vision/MoE/MTP จาก profile["features"] · adopt ไม่เคยเขียนคีย์นี้
    bundle ที่ adopt มาจึงโล่งไปทั้งแถว ทั้งที่ config.json อยู่บนดิสก์ให้อ่านอยู่แล้ว
    """
    path = _host_path(adopted, adopted.model)
    if path is None or not (path / "config.json").is_file():
        return {}
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    text = config.get("text_config") or config
    features: dict = {}

    experts = text.get("num_local_experts") or text.get("num_experts") or text.get("n_routed_experts")
    active = text.get("num_experts_per_tok") or text.get("moe_topk")
    if experts:
        features["moe"] = {"experts": int(experts),
                           **({"experts_active": int(active)} if active else {})}

    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    if config.get("vision_config") or config.get("processor_class") \
            or architecture.endswith("ForConditionalGeneration"):
        features["multimodal"] = {"projector": True}

    if text.get("num_nextn_predict_layers") or config.get("num_nextn_predict_layers"):
        features["speculative"] = {"embedded_mtp": True}
    return features

def weights_on_host(adopted: "Adopted") -> dict:
    """ที่เก็บ weight บนเครื่อง — อ่านจาก bind mount ที่ container ใช้อยู่จริง ไม่ได้เดา

    `lmds remove` ต้องรู้ว่าจะลบอะไร: bundle ที่ adopt มาไม่มี MODEL_DIR/HF cache แบบ bundle ปกติ
    weight อยู่ที่ไหนสักแห่งใน `-v` ของ docker run — HF cache ที่ mount เป็น /root/.cache/huggingface
    หรือโฟลเดอร์โมเดลตรง ๆ · เคสจริง 2026-09-04: remove บอกแค่ "ต้องใช้ sudo rm -rf …" โดยไม่มี path
    ให้ เพราะไม่มีใครจดไว้ตอน adopt · จดลง MODEL_PROFILE["weights"] ให้ remove/status ใช้ต่อ

    คืน {} เมื่อไม่รู้ — ดีกว่าเดามั่วแล้วลบผิดโฟลเดอร์
    """
    model = adopted.model or ""
    out: dict = {}
    if model.startswith("/"):
        host = _host_path(adopted, model)
        if host is not None:
            out = {"path": str(host), "kind": "dir" if host.is_dir() else "file", "source": "bind-mount"}
    elif "/" in model:
        slug = f"models--{model.replace('/', '--')}"
        for bind in adopted.binds:
            host_dir = Path(bind.split(":")[0])
            for candidate in (host_dir / "hub" / slug, host_dir / slug):
                if candidate.is_dir():
                    out = {"path": str(candidate), "kind": "hf-cache", "source": "bind-mount"}
                    break
            if out:
                break
        if not out:
            # container ใช้ cache ในตัวเอง (ไม่ได้ mount) หรือ weight ยังไม่มาถึงเครื่อง — จดชื่อ repo ไว้ให้
            # remove ค้นใน HF cache ของเครื่องต่อได้ ไม่ต้องเดาจาก container ที่ตายไปแล้ว
            out = {"hf_repo": model, "kind": "hf-cache"}
    if adopted.binds:
        out["binds"] = list(adopted.binds)
    return out


def _weights_label(weights: dict) -> str:
    """บรรทัดที่ controller/remove พิมพ์ — path จริงถ้ารู้ ไม่รู้ก็บอกว่าไม่รู้ ไม่พิมพ์ path เดา"""
    if weights.get("path"):
        return weights["path"]
    if weights.get("hf_repo"):
        return f"HF cache ของ {weights['hf_repo']} (ยังไม่พบบนเครื่อง — ดู lmds weights)"
    return "(ไม่ทราบ — ดู bind mount ใน docker inspect)"


# env ของ image เองมีเป็นร้อยตัว (PATH, CUDA_*, LD_*) — เอาไปใส่ใน docker run ซ้ำ
# ไม่ได้ช่วยอะไรและทำให้สคริปต์อ่านไม่รู้เรื่อง · เก็บเฉพาะที่ผู้ใช้ตั้งเองจริง ๆ
_KEEP_ENV_PREFIXES = ("MODEL", "PORT", "MAX_", "VLLM_", "HF_", "CTX_", "API_", "SERVED_",
                      "NCCL_", "CUDA_VISIBLE_DEVICES", "TOKENIZERS_")


def meaningful_env(adopted: Adopted) -> list[str]:
    return [e for e in adopted.env if e.split("=", 1)[0].startswith(_KEEP_ENV_PREFIXES)]




# ---------------------------------------------------------------------------
# process ที่รันตรง ๆ (ไม่ใช่ container)
# ---------------------------------------------------------------------------
@dataclass
class AdoptedProcess:
    """สิ่งที่อ่านได้จาก process ที่รันอยู่ — ทั้งหมดมาจาก /proc ไม่ได้เดา"""

    pid: int
    argv: list[str] = field(default_factory=list)
    exe: str = ""
    cwd: str = ""
    # systemd unit ที่เป็นเจ้าของ (ว่าง = ไม่ได้รันใต้ unit) — ตัวที่จะแย่ง port กลับ
    unit: str = ""

    @property
    def engine(self) -> str:
        name = (self.exe or (self.argv[0] if self.argv else "")).lower()
        if "llama" in name:
            return "llamacpp"
        argv = " ".join(self.argv).lower()
        if "sglang" in name or "sglang" in argv:
            return "sglang"
        if "vllm" in name or "vllm" in argv:
            return "vllm"
        return "unknown"

    @property
    def port(self) -> int:
        value = _argv_value(self.argv, "--port", "-p")
        return int(value) if value.isdigit() else 0

    @property
    def model_path(self) -> str:
        return _argv_value(self.argv, "-m", "--model")

    @property
    def model(self) -> str:
        alias = _argv_value(self.argv, "--alias", "--served-model-name")
        if alias:
            return alias
        return Path(self.model_path).stem if self.model_path else ""

    @property
    def context(self) -> int:
        value = _argv_value(self.argv, "-c", "--ctx-size", "--max-model-len")
        return int(value) if value.isdigit() else 0


def _argv_value(argv: list[str], *flags: str) -> str:
    """ค่าของ flag แรกที่เจอ — รองรับทั้ง `--flag value` และ `--flag=value`"""
    for index, item in enumerate(argv):
        for flag in flags:
            if item == flag and index + 1 < len(argv):
                return argv[index + 1]
            if item.startswith(f"{flag}="):
                return item.split("=", 1)[1]
    return ""


def _read_proc(pid: int, name: str) -> str:
    try:
        return Path(f"/proc/{pid}/{name}").read_text(errors="replace")
    except OSError:
        return ""


def owning_unit(pid: int) -> str:
    """systemd unit *ของคนอื่น* ที่เป็นเจ้าของ process — ตัวที่จะแย่ง port กลับ

    สนใจเฉพาะ unit ที่ไม่ใช่ของ LMDS · process ที่ถูก start จากคอนโซลจะสืบ cgroup ของ
    `lmds-web.service` มาด้วย ถ้าคว้ามาใช้จะได้คำเตือนที่ผิด ("unit เดิมยังคุมอยู่" ทั้งที่
    ไม่มีใครแย่ง) และร้ายกว่านั้นคือ controller จะปฏิเสธ start ตัวเองเพราะเห็นว่า unit
    ที่ตัวเองอ้างว่าเป็นเจ้าของยัง active — เจอจริงบนเครื่องลูกค้า บันทึกเป็น lmds-web.service
    """
    for line in _read_proc(pid, "cgroup").splitlines():
        part = line.rsplit("/", 1)[-1].strip()
        if part.endswith(".service") and not _is_own_unit(part):
            return part
    return ""


# unit ที่ LMDS สร้างเอง — ไม่ใช่ "เจ้าของเดิม" ที่ต้องระวัง
_FOREIGN_UNIT = ("lmds-",)


def _is_own_unit(unit: str) -> bool:
    return unit.startswith(_FOREIGN_UNIT)


def inspect_process(pid: int = 0, port: int = 0) -> AdoptedProcess:
    """อ่านคำสั่งที่ process กำลังรันอยู่จริง

    **จงใจไม่อ่าน /proc/<pid>/environ** — API key ของ backend อยู่ในนั้น การเขียนมันลง
    bundle คือทำให้ทุกคนที่อ่านไฟล์ได้เห็น secret · cmdline พอสำหรับรันซ้ำอยู่แล้ว ส่วน
    env ที่จำเป็นจริงให้คนตั้งเองใน bundle.env ซึ่งเป็นที่ของมัน
    """
    if not pid and not port:
        raise FleetError("ต้องระบุ --pid หรือ --port")
    if not pid:
        pid = _pid_on_port(port)
        if not pid:
            raise FleetError(f"ไม่มี process ไหนฟังอยู่ที่ port {port}")

    raw = _read_proc(pid, "cmdline")
    if not raw:
        raise FleetError(f"อ่าน /proc/{pid}/cmdline ไม่ได้ — process ยังอยู่ไหม?")
    argv = [a for a in raw.split("\0") if a]

    try:
        exe = str(Path(f"/proc/{pid}/exe").resolve())
    except OSError:
        exe = argv[0] if argv else ""
    try:
        cwd = str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        cwd = ""

    return AdoptedProcess(pid=pid, argv=argv, exe=exe, cwd=cwd, unit=owning_unit(pid))


def _pid_on_port(port: int) -> int:
    try:
        proc = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FleetError(f"เรียก ss ไม่ได้: {exc}") from exc
    for line in proc.stdout.splitlines():
        if f":{port} " not in line:
            continue
        marker = "pid="
        if marker in line:
            value = line.split(marker, 1)[1].split(",", 1)[0]
            if value.isdigit():
                return int(value)
    return 0



# ชื่อ env ที่เป็นความลับ — adopt ต้องไม่คัดลอกค่าลงสคริปต์ที่วางไว้บนดิสก์
# เคสจริง dgx-spark03 2026-09-03: `lmds adopt trtllm-nemotron` เขียน `--env HF_TOKEN=hf_…`
# ลง bundles/…-adopted.sh แบบ 0755 อ่านได้ทุก user บนเครื่อง · หลักของ LMDS คือความลับเดินทาง
# ทาง env/stdin เท่านั้น (ดู node ctl) สคริปต์ที่ adopt สร้างต้องอยู่ใต้กติกาเดียวกัน
_SECRET_ENV = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|CREDENTIAL)", re.IGNORECASE)


def redact_secrets(env_items: list[str]) -> tuple[list[str], list[str]]:
    """คืน (env ที่จะเขียนลงสคริปต์, ชื่อที่ถูกถอดค่าออก)

    ตัวที่เป็นความลับเหลือแค่ชื่อ — `docker run --env NAME` หยิบค่าจาก environment ของ
    เชลล์ที่สั่ง start ซึ่งเป็นที่ที่ค่านั้นควรอยู่ · ค่าที่ไม่มีอยู่จริงตอน start = docker ข้ามให้
    """
    kept, redacted = [], []
    for item in env_items:
        name, sep, _ = item.partition("=")
        if sep and _SECRET_ENV.search(name):
            kept.append(name)
            redacted.append(name)
        else:
            kept.append(item)
    return kept, redacted


# คำสั่ง start ที่ไป "ดาวน์โหลดก่อนแล้วค่อยเสิร์ฟ" (`hf download X && …serve X`) ทำงานได้ตอนแรก
# แต่กลายเป็นระเบิดเวลา: repo ที่ gated + token หมดอายุ = ดึง revision ใหม่ได้ครึ่งเดียว (401)
# แล้ว serve ชี้ไปที่ snapshot ที่ไม่ครบ · เคสจริง dgx-spark03 2026-09-03: สร้าง container ใหม่
# จากคำสั่งเดิมเป๊ะ → วนล้ม 15 รอบ ทั้งที่ snapshot ที่ครบอยู่บนดิสก์มาตั้งแต่ มิ.ย.
_DOWNLOAD_THEN_SERVE = re.compile(r"^\s*(hf|huggingface-cli)\s+download\s+(\S+)[^&]*&&", re.MULTILINE)


def download_before_serve(command: str) -> str:
    """repo id ที่คำสั่งจะไปดึงก่อนเสิร์ฟ — ว่างเมื่อไม่มีขั้นนั้น"""
    m = _DOWNLOAD_THEN_SERVE.search(command or "")
    return m.group(2) if m else ""

def render_controller(adopted: Adopted, slug: str) -> str:
    """สคริปต์ที่รัน container เดิมซ้ำได้ — คำสั่งเดียวกับที่มันรันอยู่ตอนนี้"""
    env_items, redacted = redact_secrets(meaningful_env(adopted))
    fetch_repo = download_before_serve(" ".join(adopted.args or []))
    if fetch_repo:
        env_lines_note = (f"  # ⚠ คำสั่งนี้ไป `hf download {fetch_repo}` ก่อนเสิร์ฟทุกครั้งที่ start — ถ้า repo\n"
                          f"  #   gated และ token หมดอายุ จะได้ snapshot ใหม่ที่ไม่ครบแล้วเสิร์ฟล้ม (dgx-spark03 2026-09-03)\n"
                          f"  #   ถ้า weight อยู่ครบแล้ว ให้ชี้ path ของ snapshot ตรง ๆ และตั้ง HF_HUB_OFFLINE=1\n")
    else:
        env_lines_note = ""
    env_lines = "".join(f'  --env {shlex.quote(e)} \\\n' for e in env_items)
    if redacted:
        names = " ".join(redacted)
        env_lines = (f"  # ค่าของ {names} ไม่ได้เก็บไว้ในไฟล์นี้ — export ไว้ในเชลล์ก่อน start "
                     f"(docker หยิบจาก environment ให้เอง)\n") + env_lines
    env_lines = env_lines_note + env_lines
    bind_lines = "".join(f'  --volume {shlex.quote(b)} \\\n' for b in adopted.binds)
    def _publish(spec: str, binding: list) -> str:
        # เดิมทิ้ง HostIp → container ที่เคย bind แค่ 127.0.0.1 กลายเป็นเปิดทุก interface หลัง adopt
        host_ip = str(binding[0].get("HostIp") or "").strip()
        prefix = f"{host_ip}:" if host_ip and host_ip not in ("0.0.0.0", "::") else ""
        return (f'  --publish {shlex.quote(prefix + str(binding[0]["HostPort"]))}'
                f':{shlex.quote(spec.split("/")[0])} \\\n')

    port_lines = "".join(
        _publish(spec, binding) for spec, binding in (adopted.ports or {}).items() if binding
    )
    entry = f'  --entrypoint {shlex.quote(adopted.entrypoint[0])} \\\n' if adopted.entrypoint else ""
    network = f'  --network {shlex.quote(adopted.network)} \\\n' if adopted.network not in ("", "default") else ""
    runtime = '  --gpus all \\\n' if adopted.runtime == "nvidia" else ""
    # NCCL/torch.distributed คุยกันผ่าน /dev/shm — docker ให้มาแค่ 64 MB โดยปริยาย
    #
    # เคสจริง 2026-09-01: adopt โมเดล stacked (MiniMax M3 บน SGLang 2 เครื่อง) แล้ว
    # ทิ้ง --ipc host --shm-size ของเดิมไป พอ start ใหม่ head ตายด้วย
    #   "creating shared memory segment /dev/shm/nccl-… No space left on device (28)"
    # ส่วน worker ที่ต่อกลับมาเจอ Connection refused ก็ตายตาม · ที่ร้ายกว่าคือ
    # --restart unless-stopped ปลุก head ซ้ำทุก 10 นาทีจนครบ 31 รอบโดยไม่มีใครรู้
    ipc = f'  --ipc {shlex.quote(adopted.ipc_mode)} \\\n' if adopted.ipc_mode not in ("", "private") else ""
    shm = f'  --shm-size {adopted.shm_size} \\\n' if adopted.shm_size > 67108864 else ""
    args = " ".join(shlex.quote(a) for a in adopted.args)
    # สิ่งที่ lmds remove จะแตะ — status/info/remove-plan พิมพ์ให้เห็นก่อน ไม่ใช่รู้ตอนที่ลบไปแล้ว
    weights_label = shlex.quote(_weights_label(weights_on_host(adopted)))

    return f'''#!/usr/bin/env bash
# LMDS adopted controller — สร้างจาก container ที่รันอยู่ก่อนหน้า ไม่ได้ deploy ผ่าน LMDS
#
# สคริปต์นี้ทำได้แค่ "รันคำสั่งเดิมซ้ำ" — weight เป็น path ที่ผู้ใช้จัดการเอง จึงไม่มี
# download/verify-files ให้ · คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มี
set -Eeuo pipefail

SCRIPT_VERSION="${{SCRIPT_VERSION:-1.0.0}}"
ADOPTED=1
CONTAINER_NAME="{adopted.container}"
IMAGE="${{IMAGE:-{adopted.image}}}"
API_PORT="${{API_PORT:-{adopted.port or 8000}}}"
SLUG="{slug}"

die() {{ echo "ERROR: $*" >&2; exit 1; }}

# ชื่อโมเดลที่ server เสิร์ฟอยู่จริง — /v1/models มีคีย์ "id" หลายตัว (ของ permission ด้วย)
# regex แบบ greedy จะคว้าตัวสุดท้ายมา แล้วขอ completion ด้วยชื่อที่ server ไม่รู้จัก → 404
served_model() {{
  local body
  body="$(curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models")" || return 1
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
  else
    printf '%s' "$body" | sed -E 's/^[^"]*"object":"list".*?"id":"([^"]+)".*/\\1/;q'
  fi
}}

banner() {{
  echo "LMDS adopted · {slug} · v${{SCRIPT_VERSION}}"
  echo "container: ${{CONTAINER_NAME}} · image: ${{IMAGE}}"
}}

info() {{
  banner
  echo "model:     {adopted.model or '(ไม่ระบุใน env)'}"
  echo "weights:   "{weights_label}
  echo "context:   {adopted.context or 0}"
  echo "port:      ${{API_PORT}}"
  echo "adopted:   ใช่ — สร้างจาก container ที่รันอยู่ก่อน LMDS"
}}

# สิ่งที่ `lmds remove {slug}` จะลบ — weight ของ bundle ที่ adopt มาอยู่นอกที่ที่ LMDS จัดการ
# จึงต้องบอกเป็น path ตรง ๆ ก่อนใครกดลบ (เดิมรู้ตอนที่ remove ตอบ "ต้องใช้ sudo rm -rf" แล้ว)
remove_plan() {{
  echo "lmds remove {slug} จะลบ:"
  echo "  bundle:    $(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
  echo "  ทะเบียน:   ${{LMDS_RUN_ROOT:-${{HOME}}/.lmds/run}}/{slug}"
  echo "  weights:   "{weights_label}
  echo "  container: {adopted.container} (หยุดและลบ · image {adopted.image} ไม่ถูกลบ)"
}}

start() {{
  local running
  running="$(docker ps --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  [[ -z "$running" ]] || die "container ${{CONTAINER_NAME}} กำลังรันอยู่ — รัน: $0 stop ก่อน"
  local leftover
  leftover="$(docker ps -a --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  if [[ -n "$leftover" ]]; then
    echo "เก็บซาก container จากรอบก่อน (${{CONTAINER_NAME}}) แล้วเริ่มใหม่"
    docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  fi
  docker run -d --name "${{CONTAINER_NAME}}" --restart unless-stopped \\
{runtime}{ipc}{shm}{network}{port_lines}{bind_lines}{env_lines}{entry}  "${{IMAGE}}" {args}
  echo "started: ${{CONTAINER_NAME}} (port ${{API_PORT}})"
}}

stop() {{
  docker stop "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  echo "stopped: ${{CONTAINER_NAME}}"
}}

restart() {{ stop; start; }}

status() {{
  docker ps -a --filter "name=^${{CONTAINER_NAME}}$" --format 'container: {{{{.Names}}}} · {{{{.Status}}}}'
  curl -fsS -m 5 "http://127.0.0.1:${{API_PORT}}/v1/models" >/dev/null 2>&1 \\
    && echo "api: ตอบปกติ" || echo "api: ยังไม่ตอบ"
  echo "weights: "{weights_label}"  (lmds remove {slug} ลบด้วย — ดู: $0 remove-plan)"
}}

logs() {{ docker logs --tail "${{1:-300}}" "${{CONTAINER_NAME}}"; }}

test_text() {{
  local served
  served="$(served_model)" || die "เรียก /v1/models ไม่ได้ — server ขึ้นหรือยัง? ดู: $0 logs"
  curl -fsS "http://127.0.0.1:${{API_PORT}}/v1/chat/completions" \\
    -H "Content-Type: application/json" \\
    -d "{{\\"model\\": \\"$served\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": \\"ตอบสั้น ๆ: 2+2 เท่ากับเท่าไร\\"}}], \\"max_tokens\\": 256}}" \\
    || die "เรียก /v1/chat/completions ไม่สำเร็จ — ดู: $0 logs"
  echo ""
}}

client_config() {{
  local served
  served="$(served_model)" || served="{slug}"
  echo "{{"
  echo "  \\"base_url\\": \\"http://$(hostname -I | awk '{{print $1}}'):${{API_PORT}}/v1\\","
  echo "  \\"model\\": \\"$served\\","
  echo "  \\"server_context\\": {adopted.context or 0}"
  echo "}}"
}}

usage() {{
  banner
  cat <<'USAGE'

คำสั่ง:
  start | stop | restart      รันคำสั่งเดิมของ container ซ้ำ
  status                      สถานะ container + API
  logs [N]                    log ล่าสุด N บรรทัด
  test-text                   ถามจริงแล้วดูว่าตอบไหม
  client-config               ค่าที่ client ต้องใช้
  remove-plan                 สิ่งที่ lmds remove จะลบ (bundle · ทะเบียน · weight · container)
  info | banner               ข้อมูลของ bundle นี้

ไม่มี download / verify-files: weight ของ container นี้เป็น path ที่คุณจัดการเอง
LMDS จึงไม่มีอะไรให้โหลดหรือตรวจ — ดูแลไฟล์เองเหมือนเดิม
USAGE
}}

case "${{1:-}}" in
  start)          start ;;
  stop)           stop ;;
  restart)        restart ;;
  status)         status ;;
  logs)           shift; logs "${{1:-300}}" ;;
  test-text)      test_text ;;
  client-config)  client_config ;;
  remove-plan)    remove_plan ;;
  info|banner)    info ;;
  *)              usage ;;
esac
'''


def adopt(container: str, slug: str = "", output: Path | None = None) -> Path:
    """สร้าง bundle จาก container ที่รันอยู่ แล้วลงทะเบียนกับ fleet — คืน path ของ controller"""
    if slug:
        _check_slug(slug)  # ก่อนแตะ docker — slug ผิดรูปไม่ควรได้ไปถึง inspect
    adopted = inspect_container(container)
    slug = slug or _derive_slug(adopted.container.replace("_", "-"))
    directory = (output or Path("./bundles")) / slug
    directory.mkdir(parents=True, exist_ok=True)

    controller = directory / f"{slug}-adopted.sh"
    controller.write_text(render_controller(adopted, slug), encoding="utf-8")
    controller.chmod(0o755)

    profile = {
        "profile_version": 1,
        "generated_by": "lmds adopt",
        "adopted": True,
        "model": {"id": adopted.model or adopted.container, "artifact_type": "unknown"},
        "runtime": {"engine": adopted.engine,
                    "image": adopted.image},
        "serving": {"context": adopted.context, "port": adopted.port},
        "source_container": adopted.container,
    }
    features = _features_from_model(adopted)
    if features:
        profile["features"] = features
    # ที่เก็บ weight — lmds remove/status อ่านจากตรงนี้ (ดู weights_on_host)
    weights = weights_on_host(adopted)
    if weights:
        profile["weights"] = weights
    import yaml

    (directory / "MODEL_PROFILE.yaml").write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_dir = run_root() / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={adopted.model or adopted.container}\n"
        f"model_id={adopted.model or adopted.container}\n"
        f"engine={profile['runtime']['engine']}\n"
        f"mode=docker\n"
        f"port={adopted.port}\n"
        f"container={adopted.container}\n"
        f"pid_file=\n"
        f"controller={controller}\n"
        f"started_at=\n",
        encoding="utf-8")
    return controller


# ---------------------------------------------------------------------------
# ถามเซิร์ฟเวอร์ที่รันอยู่ว่ามันทำอะไรได้บ้าง
# ---------------------------------------------------------------------------
# adopt มีของที่ deploy ปกติไม่มี: **เซิร์ฟเวอร์ตัวจริงรันอยู่ตรงหน้า** · llama.cpp บอก
# modalities, chat_template_caps และ n_ctx_train ของตัวเองได้ตรง ๆ จึงไม่ต้องเดาจากชื่อไฟล์
# และไม่ต้องให้ผู้ใช้ deploy ใหม่เพื่อให้ป้ายความสามารถขึ้นในคอนโซล
def probe_server(port: int, timeout: float = 5.0) -> dict:
    """ค่าที่เซิร์ฟเวอร์รายงานเอง — คืน {} เมื่อถามไม่ได้ (adopt ต้องไม่ล้มเพราะเรื่องนี้)"""
    import json as _json
    import urllib.request

    out: dict = {}
    for path, key in (("/props", "props"), ("/v1/models", "models")):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=timeout
            ) as response:
                out[key] = _json.loads(response.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 — ถามไม่ได้ก็แค่ไม่มีข้อมูล ไม่ใช่เหตุให้ adopt ล้ม
            continue
    return out


def features_from_probe(probe: dict, argv: list[str]) -> dict:
    """features block สำหรับ MODEL_PROFILE — จากสิ่งที่เซิร์ฟเวอร์บอก ไม่ใช่จากชื่อโมเดล"""
    props = probe.get("props") or {}
    caps = props.get("chat_template_caps") or {}
    modalities = props.get("modalities") or {}

    projector = _argv_value(argv, "--mmproj")
    vision = bool(modalities.get("vision")) or bool(projector)

    return {
        "tool_calling": {
            "enabled": bool(caps.get("supports_tools") or caps.get("supports_tool_calls")),
            "parser": None,
            "parallel": bool(caps.get("supports_parallel_tool_calls")),
        },
        "reasoning": {"enabled": bool(caps.get("supports_preserve_reasoning")), "parser": None},
        "multimodal": {
            "modalities": ["image", "text"] if vision else ["text"],
            "projector_files": [projector.rsplit("/", 1)[-1]] if projector else [],
        },
        # MTP ของ llama.cpp เปิดด้วย flag ไม่ใช่คุณสมบัติของไฟล์ — อ่านจาก argv ที่รันจริง
        "speculative": {
            "draft_files": (
                [_argv_value(argv, "-md", "--spec-draft-model").rsplit("/", 1)[-1]]
                if _argv_value(argv, "-md", "--spec-draft-model") else []
            ),
            "embedded": _argv_value(argv, "--spec-type") == "draft-mtp"
            and not _argv_value(argv, "-md", "--spec-draft-model"),
        },
    }


def _native_context(probe: dict) -> int:
    """เพดานจริงของตัวโมเดล (n_ctx_train) — ต่างจาก n_ctx ที่เป็นค่าที่สั่งรันครั้งนี้

    ไม่มีค่านี้ คอนโซลไม่รู้ว่าเพิ่ม context ได้ถึงไหน · เคสจริง: รันอยู่ 65,536 ทั้งที่
    โมเดลรับได้ 262,144
    """
    meta = ((probe.get("models") or {}).get("data") or [{}])[0].get("meta") or {}
    value = meta.get("n_ctx_train")
    return int(value) if isinstance(value, int) and value > 0 else 0


# flag ที่คอนโซล/CLI ต้องปรับได้ — ต้องถูกดึงออกจาก argv ที่ replay แล้วใส่กลับจากตัวแปร
# ไม่งั้นค่าที่ผู้ใช้ตั้งจะถูก argv เดิมทับทุกครั้ง (เจอจริง: ตั้ง context 131968 แล้วเด้งกลับ 65536)
_MANAGED_FLAGS = {
    "port": ("--port", "-p"),
    "ctx": ("-c", "--ctx-size"),
    "host": ("--host",),
}


def split_managed(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """คืน (argv ที่เหลือ, ค่าที่ดึงออกมา) — รองรับทั้ง `--flag value` และ `--flag=value`"""
    flat = {f: name for name, flags in _MANAGED_FLAGS.items() for f in flags}
    rest: list[str] = []
    found: dict[str, str] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        name = flat.get(item)
        if name and index + 1 < len(argv):
            found.setdefault(name, argv[index + 1])
            index += 2
            continue
        matched = False
        for flag, fname in flat.items():
            if item.startswith(f"{flag}="):
                found.setdefault(fname, item.split("=", 1)[1])
                matched = True
                break
        if matched:
            index += 1
            continue
        rest.append(item)
        index += 1
    return rest, found


def render_native_controller(proc: AdoptedProcess, slug: str) -> str:
    """สคริปต์ที่รันคำสั่งเดิมของ process ซ้ำได้ — argv ชุดเดียวกับที่มันรันอยู่ตอนนี้"""
    rest, managed = split_managed(proc.argv[1:])
    argv = " \\\n    ".join(shlex.quote(a) for a in rest)
    default_ctx = managed.get("ctx", "")
    default_host = managed.get("host", "0.0.0.0")
    exe = shlex.quote(proc.exe or (proc.argv[0] if proc.argv else ""))
    cwd = shlex.quote(proc.cwd or str(Path.home()))
    unit_note = (
        f"# unit เดิมที่เป็นเจ้าของ process นี้: {proc.unit}\n"
        f"# ถ้ามันยัง enable อยู่ มันจะแย่ง port กลับทุกครั้งที่ LMDS stop\n"
        if proc.unit else ""
    )
    weights_label = shlex.quote(proc.model_path or "(ไม่ระบุใน argv)")

    return f"""#!/usr/bin/env bash
# LMDS adopted controller (native) — สร้างจาก process ที่รันอยู่ก่อนหน้า ไม่ได้ deploy ผ่าน LMDS
#
# argv ข้างล่างคัดมาจาก /proc/{proc.pid}/cmdline ตอนรับเข้าระบบ ไม่ได้เดา
# ไม่มี download/verify-files: weight เป็น path ที่คุณจัดการเอง LMDS จึงไม่มีอะไรให้โหลดหรือตรวจ
{unit_note}set -Eeuo pipefail

SCRIPT_VERSION="${{SCRIPT_VERSION:-1.0.0}}"
ADOPTED=1
SLUG="{slug}"
API_PORT="${{API_PORT:-{proc.port or 8000}}}"
CTX_SIZE="${{CTX_SIZE:-{default_ctx}}}"
API_HOST="${{API_HOST:-{default_host}}}"
SERVER_BIN="${{SERVER_BIN:-{exe}}}"
WORK_DIR="${{WORK_DIR:-{cwd}}}"
RUN_DIR="${{RUN_DIR:-${{HOME}}/.lmds/run/{slug}}}"
PID_FILE="${{RUN_DIR}}/server.pid"
LOG_FILE="${{RUN_DIR}}/server.log"
OWNING_UNIT="{proc.unit}"

die() {{ echo "ERROR: $*" >&2; exit 1; }}

server_alive() {{ [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }}

served_model() {{
  local body
  body="$(curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models")" || return 1
  printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}}

banner() {{
  echo "LMDS adopted (native) · {slug} · v${{SCRIPT_VERSION}}"
  echo "binary: ${{SERVER_BIN}}"
}}

info() {{
  banner
  echo "model:     {proc.model or '(ไม่ระบุใน argv)'}"
  echo "weights:   {proc.model_path or '(ไม่ระบุ)'}"
  echo "context:   ${{CTX_SIZE:-ตามที่ argv เดิมตั้ง}}"
  echo "port:      ${{API_PORT}}"
  echo "adopted:   ใช่ — จาก process ที่รันอยู่ก่อน LMDS (pid {proc.pid} ตอนรับเข้า)"
  [[ -n "$OWNING_UNIT" ]] && echo "unit เดิม:  ${{OWNING_UNIT}} (ยัง enable อยู่ = แย่ง port กลับ)"
  true
}}

write_meta() {{
  mkdir -p "$RUN_DIR"
  local script_path
  script_path="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/$(basename "${{BASH_SOURCE[0]}}")"
  cat > "${{RUN_DIR}}/server.meta" <<META
slug={slug}
model={proc.model or slug}
default_model={proc.model or slug}
model_id={proc.model_path or slug}
engine={proc.engine}
mode=native
port=${{API_PORT}}
container=
pid_file=${{PID_FILE}}
controller=${{script_path}}
started_at=$(date +%Y-%m-%dT%H:%M:%S)
META
}}

start() {{
  server_alive && die "{slug} รันอยู่แล้ว (PID $(cat "$PID_FILE"))"
  # เจ้าของเดิมยังถือ port อยู่ = start ไปก็ชนกันเปล่า ๆ บอกให้ชัดดีกว่าปล่อยให้ล้มเอง
  if [[ -n "$OWNING_UNIT" ]] && systemctl is-active --quiet "$OWNING_UNIT" 2>/dev/null; then
    die "${{OWNING_UNIT}} ยังรันอยู่และถือ port ${{API_PORT}} — หยุดก่อน: sudo systemctl disable --now ${{OWNING_UNIT}}"
  fi
  [[ -x "$SERVER_BIN" ]] || die "ไม่พบ binary: $SERVER_BIN"
  mkdir -p "$RUN_DIR"
  cd "$WORK_DIR" || die "เข้า $WORK_DIR ไม่ได้"
  # argv เดิมถูกดึง --port/-c/--host ออกไปแล้ว ใส่กลับจากตัวแปรตรงนี้ เพื่อให้ค่าที่ตั้ง
  # จากคอนโซลหรือ flag บรรทัดคำสั่งชนะของเดิมได้จริง
  local args=({argv})
  args+=(--host "$API_HOST" --port "$API_PORT")
  [[ -n "$CTX_SIZE" ]] && args+=(-c "$CTX_SIZE")
  setsid nohup "$SERVER_BIN" "${{args[@]}}" >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  write_meta
  echo "started: {slug} (PID $(cat "$PID_FILE") · port ${{API_PORT}})"
}}

stop() {{
  if server_alive; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    for _ in $(seq 1 30); do server_alive || break; sleep 1; done
    server_alive && kill -9 "$(cat "$PID_FILE")" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "stopped: {slug}"
}}

restart() {{ stop; start; }}

# สิ่งที่ `lmds remove {slug}` จะลบ — weight เป็นไฟล์ที่คุณจัดการเอง จึงต้องเห็น path ก่อนกดลบ
remove_plan() {{
  echo "lmds remove {slug} จะลบ:"
  echo "  bundle:    $(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
  echo "  ทะเบียน:   ${{RUN_DIR}}"
  echo "  weights:   "{weights_label}
  [[ -n "$OWNING_UNIT" ]] && echo "  unit เดิม:  ${{OWNING_UNIT}} ไม่ถูกแตะ — ปิดเองถ้าไม่ใช้แล้ว: sudo systemctl disable --now ${{OWNING_UNIT}}"
  true
}}

status() {{
  echo "model:     {proc.model or slug}"
  echo "weights:   "{weights_label}"  (lmds remove {slug} ลบด้วย — ดู: $0 remove-plan)"
  if server_alive; then echo "process: running (PID $(cat "$PID_FILE"))"; else echo "process: stopped"; fi
  curl -fsS -m 5 "http://127.0.0.1:${{API_PORT}}/v1/models" >/dev/null 2>&1 \\
    && echo "api: ตอบปกติ" || echo "api: ยังไม่ตอบ"
  if [[ -n "$OWNING_UNIT" ]] && systemctl is-active --quiet "$OWNING_UNIT" 2>/dev/null; then
    echo "หมายเหตุ: ${{OWNING_UNIT}} ยังรันอยู่ — ตัวที่ตอบอาจเป็นของ unit นั้น ไม่ใช่ของ LMDS"
  fi
  true
}}

logs() {{ tail -n "${{1:-300}}" "$LOG_FILE" 2>/dev/null || echo "ยังไม่มี log (start ผ่าน LMDS ก่อน)"; }}

test_text() {{
  local served
  served="$(served_model)" || die "เรียก /v1/models ไม่ได้ — server ขึ้นหรือยัง? ดู: $0 logs"
  curl -fsS "http://127.0.0.1:${{API_PORT}}/v1/chat/completions" \\
    -H "Content-Type: application/json" \\
    -d "{{\\"model\\": \\"$served\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": \\"ตอบสั้น ๆ: 2+2 เท่ากับเท่าไร\\"}}], \\"max_tokens\\": 256}}" \\
    || die "เรียก /v1/chat/completions ไม่สำเร็จ — ดู: $0 logs"
  echo ""
}}

client_config() {{
  local served
  served="$(served_model)" || served="{slug}"
  echo "{{"
  echo "  \\"base_url\\": \\"http://$(hostname -I | awk '{{print $1}}'):${{API_PORT}}/v1\\","
  echo "  \\"model\\": \\"$served\\","
  echo "  \\"server_context\\": {proc.context or 0}"
  echo "}}"
}}

network_info() {{
  echo "Bind:      0.0.0.0:${{API_PORT}}"
  echo "Endpoint:  http://$(hostname -I | awk '{{print $1}}'):${{API_PORT}}/v1"
  echo "Model:     {proc.model or slug}"
}}

usage() {{
  banner
  cat <<'USAGE'

คำสั่ง:
  start | stop | restart      รันคำสั่งเดิมของ process ซ้ำ
  status                      สถานะ process + API
  logs [N]                    log ล่าสุด N บรรทัด
  test-text                   ถามจริงแล้วดูว่าตอบไหม
  client-config               ค่าที่ client ต้องใช้
  network-info                bind + endpoint
  remove-plan                 สิ่งที่ lmds remove จะลบ (bundle · ทะเบียน · weight)
  info | banner               ข้อมูลของ bundle นี้

ไม่มี download / verify-files: weight เป็น path ที่คุณจัดการเอง
LMDS จึงไม่มีอะไรให้โหลดหรือตรวจ — ดูแลไฟล์เองเหมือนเดิม
USAGE
}}

# flag ที่รับได้ตอน start/restart — ชุดเดียวกับ controller ปกติ
ARGS=()
while (( $# )); do
  case "$1" in
    --port)      API_PORT="$2"; shift 2 ;;
    --port=*)    API_PORT="${{1#*=}}"; shift ;;
    --context)   CTX_SIZE="$2"; shift 2 ;;
    --context=*) CTX_SIZE="${{1#*=}}"; shift ;;
    --bind)      API_HOST="$2"; shift 2 ;;
    --bind=*)    API_HOST="${{1#*=}}"; shift ;;
    *)           ARGS+=("$1"); shift ;;
  esac
done
set -- "${{ARGS[@]}}"

case "${{1:-}}" in
  start)          start ;;
  stop)           stop ;;
  restart)        restart ;;
  status)         status ;;
  logs)           shift; logs "${{1:-300}}" ;;
  test-text)      test_text ;;
  client-config)  client_config ;;
  network-info)   network_info ;;
  remove-plan)    remove_plan ;;
  info|banner)    info ;;
  *)              usage ;;
esac
"""


def adopt_process(pid: int = 0, port: int = 0, slug: str = "",
                  output: Path | None = None) -> tuple[Path, AdoptedProcess]:
    """สร้าง bundle จาก process ที่รันอยู่ — คืน (path ของ controller, สิ่งที่อ่านได้)"""
    if slug:
        _check_slug(slug)
    proc = inspect_process(pid=pid, port=port)
    if proc.engine == "unknown":
        raise FleetError(
            f"pid {proc.pid} ไม่ใช่ตัวเสิร์ฟโมเดลที่รู้จัก (argv: {' '.join(proc.argv[:3])} …)"
        )
    slug = slug or _derive_slug((proc.model or f"pid-{proc.pid}").replace("_", "-"))
    directory = (output or Path("./bundles")) / slug
    directory.mkdir(parents=True, exist_ok=True)

    controller = directory / f"{slug}-adopted.sh"
    controller.write_text(render_native_controller(proc, slug), encoding="utf-8")
    controller.chmod(0o755)

    # เซิร์ฟเวอร์ตัวจริงรันอยู่ตรงหน้า — ถามมันเลยว่าทำอะไรได้ ดีกว่าเดาจากชื่อไฟล์
    # ไม่มีบล็อกนี้ คอนโซลไม่มีข้อมูลจะแสดงป้ายความสามารถ และไม่รู้เพดาน context
    probe = probe_server(proc.port) if proc.port else {}
    native_context = _native_context(probe)
    meta = ((probe.get("models") or {}).get("data") or [{}])[0].get("meta") or {}

    profile = {
        "profile_version": 1,
        "generated_by": "lmds adopt (native)",
        "adopted": True,
        "model": {
            "id": proc.model_path or proc.model or slug,
            "served_name": proc.model or slug,
            "artifact_type": "gguf" if proc.model_path.endswith(".gguf") else "unknown",
            "params_total": meta.get("n_params"),
            "weight_bytes": meta.get("size"),
            "quantization": meta.get("ftype"),
            # เพดานของ *ตัวโมเดล* ไม่ใช่ค่าที่สั่งรันครั้งนี้ — คอนโซลใช้บอกว่าเพิ่มได้ถึงไหน
            "native_context": native_context or None,
        },
        "runtime": {
            "engine": proc.engine, "native_build": True, "binary": proc.exe,
            "build": ((probe.get("props") or {}).get("build_info")),
        },
        "serving": {"context": proc.context, "port": proc.port},
        "limits": {
            "context_tokens": native_context or proc.context or 0,
            "max_output_tokens": 8192,
        },
        "features": features_from_probe(probe, proc.argv),
        "source_process": {"pid": proc.pid, "unit": proc.unit, "argv": proc.argv},
    }
    if proc.model_path:
        # ไฟล์ที่ process ถืออยู่จริง (-m บน argv) — lmds remove ถามก่อนลบ ไม่เดาจาก ~/models/<slug>
        profile["weights"] = {"path": proc.model_path, "kind": "file", "source": "argv"}
    import yaml

    (directory / "MODEL_PROFILE.yaml").write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_dir = run_root() / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={proc.model or slug}\n"
        f"default_model={proc.model or slug}\n"
        f"model_id={proc.model_path or slug}\n"
        f"engine={proc.engine}\n"
        f"mode=native\n"
        f"port={proc.port}\n"
        f"container=\n"
        f"pid_file={run_dir / 'server.pid'}\n"
        f"controller={controller}\n"
        f"started_at=\n",
        encoding="utf-8")
    return controller, proc
