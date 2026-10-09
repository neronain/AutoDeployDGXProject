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
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lmds.secrets.redact import MASK, redact

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
    # `--gpus all` บน docker รุ่นใหม่ **ไม่ได้** ตั้ง Runtime=nvidia — มันไปอยู่ใน
    # HostConfig.DeviceRequests ส่วน Runtime ยังเป็น "runc" ตาม default ของ daemon
    device_requests: list = field(default_factory=list)
    entrypoint: list[str] = field(default_factory=list)
    ipc_mode: str = ""
    shm_size: int = 0
    # ── audit 2026-10-06: ทุกฟิลด์ข้างล่างคือของที่เคย "หายเงียบ ๆ" ตอน restart ──────────────
    # ค่าว่างทั้งชุด = Adopted ที่ปั้นขึ้นตรง ๆ (เทสรุ่นก่อน) → สคริปต์ที่ได้ต้องเหมือนเดิมทุกบรรทัด
    #
    # Path ของ docker inspect = argv[0] ที่ container รันจริง (Args คือที่เหลือ) · image ที่ไม่มี
    # ENTRYPOINT ตัว executable อยู่ตรงนี้ที่เดียว (Config.Entrypoint เป็น null)
    path: str = ""
    # env ที่ image อบมาเอง — None = ถาม image ไม่ได้ จึงแยกของผู้ใช้ออกจากของ image ไม่ได้
    image_env: list[str] | None = None
    # HostConfig.Mounts: `--mount …` และ volume แบบ long syntax ของ compose (ไม่อยู่ใน Binds)
    mounts: list[dict] = field(default_factory=list)
    # NetworkSettings.Ports: พอร์ตฝั่งเครื่องที่ docker แจกให้ *ตอนนี้* (ใช้เมื่อ -p ไม่ระบุฝั่งเครื่อง)
    published: dict = field(default_factory=dict)
    # restart policy เดิม ("no" · "always" · "unless-stopped" · "on-failure:3") · "" = ไม่รู้
    restart: str = ""
    # ธงของ docker run ที่ยกมาจาก HostConfig/Config ตรง ๆ — [["--ulimit", "memlock=-1:-1"], …]
    options: list[list[str]] = field(default_factory=list)
    # สิ่งที่ inspect เห็นแต่ใส่กลับใน docker run คำสั่งเดียวไม่ได้ — ดู not_reproduced()
    unreproduced: list[str] = field(default_factory=list)

    @property
    def wants_gpu(self) -> bool:
        """คอนเทนเนอร์นี้ถูกรันมาโดยขอ GPU หรือเปล่า — นับทั้งทางเก่าและทางใหม่"""
        if self.runtime == "nvidia":
            return True
        return any(_is_gpu_request(req) for req in self.device_requests or [])

    def env_value(self, name: str) -> str | None:
        """ค่าของ env ตัวนี้ใน container — None = ไม่ได้ตั้ง"""
        prefix = f"{name}="
        for item in self.env:
            if item.startswith(prefix):
                return item[len(prefix):]
        return None

    @property
    def shares_host_network(self) -> bool:
        """ไม่มีการ map พอร์ต: ใช้ network ของเครื่อง (หรือของ container อื่น) ตรง ๆ"""
        return self.network == "host" or self.network.startswith("container:")

    def bind_pairs(self) -> list[tuple[str, str]]:
        """(path บนเครื่อง, path ใน container) ของ bind mount ทุกตัว — ทั้ง `-v` และ `--mount type=bind`"""
        pairs: list[tuple[str, str]] = []
        for bind in self.binds:
            parts = bind.split(":")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
        for mount in self.mounts or []:
            if str(mount.get("Type") or "").lower() == "bind" and mount.get("Source") and mount.get("Target"):
                pairs.append((str(mount["Source"]), str(mount["Target"])))
        return pairs

    @property
    def argv_tokens(self) -> list[str]:
        """คำสั่งของ container เป็น token — แตะสตริงที่ถูกห่อด้วย shell ออกมาด้วย

        image จำนวนมากสั่งงานผ่าน `bash -c "โน่นนี่ && เซิร์ฟเวอร์ --port 8355 …"` ทำให้
        argv ทั้งชุดยุบเหลือสามชิ้น (`bash`, `-c`, สตริงยาว) · การไล่หา `--port` แบบ
        เทียบทีละชิ้นจึงไม่มีวันเจอ ทั้งที่มันอยู่ในนั้น

        เจอจริง 2026-08-27 บน spark-03 (nvidia/tensorrt-llm:nemotron-fixed2)
        """
        tokens: list[str] = []
        # Path + Args คือ argv ที่ container รันจริง · Config.Entrypoint ใช้เมื่อไม่มี Path (ปั้นขึ้นตรง ๆ)
        # — image ที่ไม่มี ENTRYPOINT ตัว executable (`vllm`) อยู่ใน Path ที่เดียว ไม่ดูตรงนี้ = engine "unknown"
        head = [self.path] if self.path else list(self.entrypoint)
        for item in head + list(self.args):
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
        """พอร์ตที่ **เครื่อง** เปิดให้เข้าถึง API — ตัวที่ health/status/watchdog/gateway ต้องใช้

        audit 2026-10-06: ของเดิมคืนพอร์ต *ข้างใน* container · `-p 8001:8000` จึงถูกจดเป็น 8000 ลง
        API_PORT, profile และ server.meta — `lmds ps` โชว์ 8001 ก่อน adopt แล้วกลายเป็น 8000 หลัง adopt
        และทุกอย่างไปเคาะ 127.0.0.1:8000 ซึ่งอาจเป็นบริการอื่นไปเลย (AI-Local-ISIT: 8000 คือ portainer)
        """
        inner = self.container_port
        if not inner or self.shares_host_network:
            return inner
        return self._host_port(inner) or inner

    def _host_port(self, inner: int) -> int:
        """พอร์ตฝั่งเครื่องที่ map เข้าพอร์ตนี้ของ container — 0 = ไม่ได้ publish

        HostConfig.PortBindings ก่อน (สิ่งที่ถูกสั่ง) แล้วค่อย NetworkSettings.Ports (สิ่งที่ docker
        แจกให้จริง — มีค่าเมื่อ `-p 8000` ไม่ระบุฝั่งเครื่อง)
        """
        for source in (self.ports, self.published):
            for spec, bindings in (source or {}).items():
                number, _, proto = str(spec).partition("/")
                if number != str(inner) or proto not in ("", "tcp"):
                    continue
                for binding in bindings or []:
                    value = str((binding or {}).get("HostPort") or "")
                    if value.isdigit() and int(value):
                        return int(value)
        return 0

    @property
    def host_port_is_ephemeral(self) -> bool:
        """`-p 8000` เฉย ๆ: docker สุ่มพอร์ตฝั่งเครื่องให้ และสุ่มใหม่ทุกครั้งที่สร้าง container"""
        for spec, bindings in (self.ports or {}).items():
            number, _, proto = str(spec).partition("/")
            if number == str(self.container_port) and proto in ("", "tcp"):
                return any(not str((b or {}).get("HostPort") or "").strip() for b in bindings or [])
        return False

    @property
    def container_port(self) -> int:
        """พอร์ตที่เซิร์ฟเวอร์ฟังอยู่ *ข้างใน* container — ใช้จับคู่กับ --publish เท่านั้น"""
        declared = self._declared_port()
        if declared:
            return declared
        for spec in self.ports or {}:
            try:
                return int(str(spec).split("/")[0])
            except ValueError:
                continue
        return 0

    def _declared_port(self) -> int:
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
        equals_form = _argv_value(argv, "--port", "--server-port")
        if equals_form.isdigit():
            return int(equals_form)
        # llama.cpp อ่าน LLAMA_ARG_* เมื่อ argv ไม่ได้บอก (argv ชนะ env) — image ทางการแนะนำทางนี้
        from_env = self.env_value("LLAMA_ARG_PORT") or ""
        return int(from_env) if from_env.isdigit() else 0

    @property
    def model(self) -> str:
        for key in ("MODEL=", "MODEL_ID=", "MODEL_PATH=", "MODEL_HANDLE="):
            for item in self.env:
                if item.startswith(key):
                    return item.split("=", 1)[1]
        # LLAMA_ARG_MODEL / LLAMA_ARG_HF_REPO: llama-server ที่ตั้งค่าผ่าน env ล้วน ๆ (argv ว่าง) —
        # audit 2026-10-06: adopt รายงาน `model: ''` ทั้งที่ชื่ออยู่ใน env ตรงหน้า
        return (self._model_from_argv() or self.env_value("LLAMA_ARG_MODEL")
                or self.env_value("LLAMA_ARG_HF_REPO") or "")

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
        for previous, item in zip(argv, argv[1:], strict=False):
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
                            "--ctx-size", "-c") or (self.env_value("LLAMA_ARG_CTX_SIZE") or "")
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


def _is_gpu_request(request) -> bool:
    """DeviceRequest ตัวนี้ขอ GPU ไหม — DeviceRequests ใช้กับ device อื่นได้ด้วย ห้ามเหมา"""
    for group in (request or {}).get("Capabilities") or []:
        if "gpu" in [str(c).lower() for c in (group or [])]:
            return True
    return False


def _one_line(text: object, limit: int = 400) -> str:
    """ข้อความบรรทัดเดียว — ค่าจาก docker inspect (label · ชื่อ env) เป็นของที่ใครก็ตั้งได้

    ขึ้นบรรทัดใหม่ในค่าแล้วเขียนลงบล็อกคอมเมนต์ของ controller ดิบ ๆ = บรรทัดถัดไปกลายเป็นคำสั่ง
    """
    flat = "".join(ch if ch.isprintable() else " " for ch in str(text))
    flat = " ".join(flat.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _csv(fields: list[str]) -> str:
    """ต่อ field แบบ CSV ตามที่ `--mount` / `--gpus` ของ docker อ่าน — field ที่มี , หรือ " ต้องห่อ"""
    out = []
    for item in fields:
        out.append('"' + item.replace('"', '""') + '"' if ("," in item or '"' in item) else item)
    return ",".join(out)


def _inspect_image(reference: str) -> dict | None:
    """Config ของ image ที่ container นี้สร้างมา — None เมื่อถามไม่ได้

    ทำไมต้องถาม image: `Config` ของ container คือ **ของ image รวมกับที่ผู้ใช้สั่ง** แล้ว · env ร้อยตัว
    (PATH, CUDA_*, LD_*) · label · WORKDIR · USER ปนกันจนแยกไม่ออกว่าตัวไหนผู้ใช้ตั้ง · ของเดิมแก้ด้วย
    การ *เดา* จาก prefix ของชื่อ env (MODEL/PORT/VLLM_/HF_/…) ซึ่งทิ้ง NVIDIA_VISIBLE_DEVICES ·
    OMP_NUM_THREADS · PYTORCH_CUDA_ALLOC_CONF · HTTPS_PROXY · TRANSFORMERS_OFFLINE · LLAMA_ARG_* ไปเงียบ ๆ
    (audit 2026-10-06) · ลบด้วยของ image แล้วที่เหลือคือของผู้ใช้พอดี ไม่ต้องเดา

    ใส่ env ของ image ลงสคริปต์ทั้งหมดแทนก็ "ครบ" เหมือนกัน แต่ตรึง PATH/LD_LIBRARY_PATH ของ image รุ่นนี้
    ไว้ — วันที่มีคนเปลี่ยน IMAGE ได้ของเก่ามาทับของใหม่
    """
    if not reference:
        return None
    try:
        proc = subprocess.run(["docker", "image", "inspect", reference],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    try:
        data = json.loads(proc.stdout)[0]
    except (ValueError, IndexError, KeyError, TypeError):
        return None
    # คำตอบต้องเป็น *image* จริง — ของปลอมในเทส (และ wrapper บางตัว) ตอบ payload ของ container กลับมา
    # ทุกคำสั่ง ถ้าเชื่อ จะสรุปว่า "env ทุกตัวเป็นของ image" แล้วไม่ใส่ env กลับสักตัว
    if not isinstance(data, dict) or "RootFS" not in data or "HostConfig" in data or "State" in data:
        return None
    config = data.get("Config")
    return config if isinstance(config, dict) else {}


# HostConfig ที่ยกเป็นธงได้ตรง ๆ · ค่าว่าง/0/false ของ docker = ไม่ได้ตั้ง → ไม่ออกธง
_HOST_LIST_FLAGS = (
    ("CapAdd", "--cap-add"), ("CapDrop", "--cap-drop"), ("Dns", "--dns"), ("DnsOptions", "--dns-option"),
    ("DnsSearch", "--dns-search"), ("ExtraHosts", "--add-host"), ("GroupAdd", "--group-add"),
    ("SecurityOpt", "--security-opt"), ("VolumesFrom", "--volumes-from"),
    ("DeviceCgroupRules", "--device-cgroup-rule"),
)
_HOST_MAP_FLAGS = (("Sysctls", "--sysctl"), ("StorageOpt", "--storage-opt"), ("Annotations", "--annotation"))
_HOST_SCALAR_FLAGS = (
    ("PidMode", "--pid"), ("UTSMode", "--uts"), ("UsernsMode", "--userns"),
    ("CgroupParent", "--cgroup-parent"), ("VolumeDriver", "--volume-driver"),
    ("Memory", "--memory"), ("MemoryReservation", "--memory-reservation"), ("MemorySwap", "--memory-swap"),
    ("CpusetCpus", "--cpuset-cpus"), ("CpusetMems", "--cpuset-mems"), ("CpuShares", "--cpu-shares"),
    ("CpuPeriod", "--cpu-period"), ("CpuQuota", "--cpu-quota"),
    ("CpuRealtimePeriod", "--cpu-rt-period"), ("CpuRealtimeRuntime", "--cpu-rt-runtime"),
    ("BlkioWeight", "--blkio-weight"), ("OomScoreAdj", "--oom-score-adj"), ("PidsLimit", "--pids-limit"),
)
_HOST_BOOL_FLAGS = (
    ("Privileged", "--privileged"), ("ReadonlyRootfs", "--read-only"), ("PublishAllPorts", "--publish-all"),
    ("OomKillDisable", "--oom-kill-disable"), ("Init", "--init"),
)
# คีย์ที่ render_controller จัดการเองอยู่แล้ว (ลำดับบรรทัดเดิมต้องไม่ขยับ) หรือมีตัวจัดการเฉพาะข้างล่าง
_HOST_HANDLED = frozenset({
    "Binds", "PortBindings", "NetworkMode", "IpcMode", "ShmSize", "Runtime", "DeviceRequests",
    "RestartPolicy", "AutoRemove", "Mounts", "LogConfig", "Tmpfs", "Ulimits", "Devices", "Links",
    "NanoCpus", "MemorySwappiness", "CgroupnsMode",
})
# คีย์ที่ไม่ใช่ "สิ่งที่ผู้ใช้สั่ง": ผลของการรันครั้งนั้น (ขนาดจอ · cidfile) · ของ Windows · หรือค่าที่ docker
# คำนวณจากธงอื่น (MaskedPaths/ReadonlyPaths ตาม --privileged/--security-opt ซึ่งยกมาแล้ว)
_HOST_NOT_AN_OPTION = frozenset({
    "ContainerIDFile", "ConsoleSize", "Isolation", "Cgroup", "MaskedPaths", "ReadonlyPaths",
    "CpuCount", "CpuPercent", "IOMaximumIOps", "IOMaximumBandwidth",
})


def _mount_option(mount: dict) -> tuple[str, list[str]]:
    """HostConfig.Mounts หนึ่งตัว → ค่าของ `--mount` ("" = แทนไม่ได้) + สิ่งที่ตกหล่น"""
    kind = str(mount.get("Type") or "").lower()
    target = str(mount.get("Target") or "")
    source = str(mount.get("Source") or "")
    if kind not in ("bind", "volume", "tmpfs") or not target:
        return "", [f"mount ชนิด {kind or '?'} ที่ {target or '?'} — LMDS ยังเขียนกลับเป็น --mount ไม่ได้ "
                    f"container ใหม่จะไม่มี mount นี้"]
    fields = [f"type={kind}"]
    if source and kind != "tmpfs":
        fields.append(f"source={source}")
    fields.append(f"target={target}")
    if mount.get("ReadOnly"):
        fields.append("readonly")
    known = {"Type", "Source", "Target", "ReadOnly", "Consistency"}
    missed: list[str] = []
    if kind == "bind":
        known.add("BindOptions")
        options = dict(mount.get("BindOptions") or {})
        propagation = options.pop("Propagation", "")
        if propagation:
            fields.append(f"bind-propagation={propagation}")
        if options.pop("NonRecursive", False):
            fields.append("bind-nonrecursive=true")
        options.pop("CreateMountpoint", None)     # path มีอยู่แล้ว (container รันอยู่) — ไม่มีผลตอนสร้างใหม่
        missed += [f"BindOptions.{k}" for k, v in options.items() if v]
    elif kind == "volume":
        known.add("VolumeOptions")
        options = dict(mount.get("VolumeOptions") or {})
        if options.pop("NoCopy", False):
            fields.append("volume-nocopy=true")
        subpath = options.pop("Subpath", "")
        if subpath:
            fields.append(f"volume-subpath={subpath}")
        driver = dict(options.pop("DriverConfig", None) or {})
        if driver.get("Name"):
            fields.append(f"volume-driver={driver['Name']}")
        for key, value in (driver.get("Options") or {}).items():
            fields.append(f"volume-opt={key}={value}")
        for key, value in (options.pop("Labels", None) or {}).items():
            fields.append(f"volume-label={key}={value}")
        missed += [f"VolumeOptions.{k}" for k, v in options.items() if v]
    else:
        known.add("TmpfsOptions")
        options = dict(mount.get("TmpfsOptions") or {})
        size = options.pop("SizeBytes", 0)
        if size:
            fields.append(f"tmpfs-size={int(size)}")
        mode = options.pop("Mode", 0)
        if mode:
            fields.append(f"tmpfs-mode={int(mode):o}")
        missed += [f"TmpfsOptions.{k}" for k, v in options.items() if v]
    missed += [k for k, v in mount.items() if k not in known and v]
    return _csv(fields), [f"{name} ของ mount {target} — LMDS ยังเขียนกลับเป็น --mount ไม่ได้" for name in missed]


def _host_options(host: dict) -> tuple[list[list[str]], list[str]]:
    """HostConfig → ธงของ docker run + รายการที่ใส่กลับไม่ได้ — **ไม่มีคีย์ไหนถูกข้ามเงียบ ๆ**

    คีย์ที่ไม่รู้จักและมีค่า (docker รุ่นใหม่เพิ่มคีย์เรื่อย ๆ) ตกไปอยู่ในรายการ ไม่ใช่หายไป
    """
    options: list[list[str]] = []
    notes: list[str] = []
    seen = set(_HOST_HANDLED) | set(_HOST_NOT_AN_OPTION)

    runtime = str(host.get("Runtime") or "")
    if runtime and runtime != "runc":
        # daemon ที่ตั้ง default-runtime=nvidia (Jetson/DGX บางเครื่อง) ให้ทุก container เป็น nvidia —
        # เขียนซ้ำให้ชัดไม่เปลี่ยนอะไร แต่ตัวที่ผู้ใช้สั่ง `--runtime nvidia` เองจะไม่หาย
        options.append(["--runtime", runtime])
    if str(host.get("CgroupnsMode") or "") == "host":
        # "private" คือค่าตั้งต้นของทุกเครื่องที่เป็น cgroup v2 (Ubuntu 22.04+/DGX OS) — ไม่ออกธง
        options.append(["--cgroupns", "host"])

    for key, flag in _HOST_BOOL_FLAGS:
        seen.add(key)
        if host.get(key) is True:
            options.append([flag])
    for key, flag in _HOST_SCALAR_FLAGS:
        seen.add(key)
        value = host.get(key)
        if value not in (None, "", 0, False):
            options.append([flag, str(value)])
    nano = host.get("NanoCpus") or 0
    if nano:
        options.append(["--cpus", f"{nano / 1e9:.3f}".rstrip("0").rstrip(".")])
    swappiness = host.get("MemorySwappiness")
    if isinstance(swappiness, int) and not isinstance(swappiness, bool) and swappiness >= 0:
        options.append(["--memory-swappiness", str(swappiness)])
    for key, flag in _HOST_LIST_FLAGS:
        seen.add(key)
        options += [[flag, str(item)] for item in host.get(key) or []]
    for key, flag in _HOST_MAP_FLAGS:
        seen.add(key)
        options += [[flag, f"{k}={v}"] for k, v in (host.get(key) or {}).items()]

    for limit in host.get("Ulimits") or []:
        options.append(["--ulimit", f"{limit.get('Name')}={limit.get('Soft')}:{limit.get('Hard')}"])
    for device in host.get("Devices") or []:
        spec = str(device.get("PathOnHost") or "")
        inside = str(device.get("PathInContainer") or "")
        perms = str(device.get("CgroupPermissions") or "")
        if inside and (inside != spec or perms):
            spec += f":{inside}"
        if perms:
            spec += f":{perms}"
        options.append(["--device", spec])
    for path, opts in (host.get("Tmpfs") or {}).items():
        options.append(["--tmpfs", f"{path}:{opts}" if opts else str(path)])
    for link in host.get("Links") or []:
        # inspect เก็บเป็น "/ตัวอื่น:/ตัวนี้/alias" — ธงรับ "ตัวอื่น:alias"
        other, _, alias = str(link).partition(":")
        other = other.lstrip("/")
        options.append(["--link", f"{other}:{alias.rsplit('/', 1)[-1]}" if alias else other])

    log = host.get("LogConfig") or {}
    log_type, log_opts = str(log.get("Type") or ""), dict(log.get("Config") or {})
    if log_opts or log_type not in ("", "json-file"):
        # ขีดจำกัดขนาด log (max-size/max-file) หายไป = ดิสก์เต็มในอีกสามเดือน โดยไม่มีอะไรเชื่อมกลับมาที่ adopt
        if log_type:
            options.append(["--log-driver", log_type])
        options += [["--log-opt", f"{k}={v}"] for k, v in log_opts.items()]

    for mount in host.get("Mounts") or []:
        value, missed = _mount_option(mount if isinstance(mount, dict) else {})
        if value:
            options.append(["--mount", value])
        notes += missed

    for request in host.get("DeviceRequests") or []:
        if _is_gpu_request(request):
            continue          # --gpus — render_controller ใส่ที่ตำแหน่งเดิมของมัน
        request = request or {}
        ids = [str(i) for i in request.get("DeviceIDs") or []]
        if str(request.get("Driver") or "") == "cdi" and ids:
            options += [["--device", i] for i in ids]
            continue
        caps = ",".join(str(c) for group in request.get("Capabilities") or [] for c in group or [])
        notes.append(f"device request ที่ไม่ใช่ GPU (driver={request.get('Driver') or '-'} · "
                     f"capabilities={caps or '-'}) — LMDS เขียนกลับเป็นธงไม่ได้ container ใหม่จะไม่ได้ device นี้")

    if host.get("AutoRemove"):
        notes.append("ของเดิมรันด้วย --rm — LMDS ไม่ใส่ --rm ให้ (container ที่ตายแล้วต้องยังอยู่ให้อ่าน log) "
                     "และใช้ --restart unless-stopped แทน")

    for key, value in host.items():
        if key in seen or value in (None, "", 0, False, [], {}):
            continue
        notes.append(f"HostConfig.{key} = {json.dumps(value, ensure_ascii=False, default=str)[:160]} — "
                     f"LMDS ยังไม่รู้วิธีใส่ตัวนี้กลับใน docker run · container ใหม่จะไม่มีค่านี้")
    return options, notes


def _subtract(mine: dict | None, baked: dict | None) -> dict:
    """คีย์ที่ container มีเกินจาก image (หรือค่าต่างกัน)"""
    baked = baked or {}
    return {k: v for k, v in (mine or {}).items() if k not in baked or baked[k] != v}


def _config_options(data: dict, image: dict | None) -> tuple[list[list[str]], list[str]]:
    """Config/NetworkSettings → ธง + รายการที่ใส่กลับไม่ได้ · ของที่ image ตั้งเองไม่ถูกเขียนซ้ำ

    image=None (ถาม image ไม่ได้): ถือว่าทุกค่าเป็นของผู้ใช้ — เขียนซ้ำค่าที่ image ตั้งอยู่แล้วให้ผลเท่าเดิม
    กับ image ตัวเดิม ซึ่งดีกว่าเดาแล้วทิ้ง
    """
    config = data.get("Config") or {}
    host = data.get("HostConfig") or {}
    base = image or {}
    options: list[list[str]] = []
    notes: list[str] = []
    name = str(data.get("Name") or "").lstrip("/")
    container_id = str(data.get("Id") or "")
    network = str(host.get("NetworkMode") or "")
    shared = network == "host" or network.startswith("container:")
    own_network = network not in ("", "default", "bridge", "none") and not shared

    if config.get("Tty"):
        options.append(["--tty"])
    if config.get("OpenStdin"):
        options.append(["--interactive"])
    user = str(config.get("User") or "")
    if user and user != str(base.get("User") or ""):
        options.append(["--user", user])
    workdir = str(config.get("WorkingDir") or "")
    if workdir and workdir != str(base.get("WorkingDir") or ""):
        options.append(["--workdir", workdir])
    hostname = str(config.get("Hostname") or "")
    # ค่าตั้งต้นของ Hostname คือ id 12 ตัวแรก · บน network ของเครื่อง docker ใส่ชื่อเครื่องให้เอง
    if hostname and container_id and not container_id.startswith(hostname) and not shared:
        options.append(["--hostname", hostname])
    if config.get("Domainname"):
        options.append(["--domainname", str(config["Domainname"])])
    if config.get("MacAddress"):
        options.append(["--mac-address", str(config["MacAddress"])])
    stop_signal = str(config.get("StopSignal") or "")
    if stop_signal and stop_signal != str(base.get("StopSignal") or ""):
        options.append(["--stop-signal", stop_signal])
    if config.get("StopTimeout") is not None:
        options.append(["--stop-timeout", str(config["StopTimeout"])])

    all_labels = config.get("Labels") or {}
    options += [["--label", f"{k}={v}"] for k, v in _subtract(all_labels, base.get("Labels")).items()]
    project = all_labels.get("com.docker.compose.project")
    if project:
        service = all_labels.get("com.docker.compose.service") or name
        files = all_labels.get("com.docker.compose.project.config_files") or ""
        notes.append(
            f"container นี้เป็นของ docker compose (project {project} · service {service}"
            + (f" · {files}" if files else "") + ") — label ของ compose ถูกคงไว้ compose จึงยังเห็นมัน: "
            "`docker compose up/down` ของ project นั้นจะสร้าง/ลบมันตามไฟล์ compose ไม่ใช่ตามสคริปต์นี้ "
            "และสิ่งที่ compose ทำให้นอกเหนือจาก docker run (depends_on · secrets · configs) ไม่มีในสคริปต์นี้")

    published = {str(spec) for spec in (host.get("PortBindings") or {})}
    for spec in _subtract(config.get("ExposedPorts"), base.get("ExposedPorts")):
        if str(spec) not in published:
            options.append(["--expose", str(spec)])

    # volume ไม่มีชื่อ (`-v /path` หรือ VOLUME ของ image): docker สร้างอันใหม่ทุกครั้งที่สร้าง container
    mounted = {str(m.get("Target") or "") for m in host.get("Mounts") or [] if isinstance(m, dict)}
    mounted |= {b.split(":")[1] for b in host.get("Binds") or [] if b.count(":") >= 1}
    if image is not None:     # ไม่รู้ว่า image ประกาศ VOLUME อะไร = ไม่รู้ว่าตัวไหนผู้ใช้สั่ง — docker สร้างของ image ให้เองอยู่แล้ว
        for path in _subtract(config.get("Volumes"), base.get("Volumes")):
            if str(path) not in mounted:
                options.append(["--volume", str(path)])
    for mount in data.get("Mounts") or []:
        if not isinstance(mount, dict) or str(mount.get("Type") or "") != "volume":
            continue
        if re.fullmatch(r"[0-9a-f]{64}", str(mount.get("Name") or "")):
            notes.append(f"volume ไม่มีชื่อที่ {mount.get('Destination')} — ข้อมูลในนั้นไม่ตามไปกับ container ใหม่ "
                         f"(stop ของสคริปต์นี้ทำ docker rm · start ได้ volume เปล่าอันใหม่)")

    test = list((config.get("Healthcheck") or {}).get("Test") or [])
    if test and test != ["NONE"]:
        notes.append("HEALTHCHECK ของเดิมถูกปิด (--no-healthcheck) — LMDS ตรวจสุขภาพเองที่พอร์ตของ API · "
                     "อะไรที่ผูกกับ health ของ docker (compose depends_on: service_healthy · autoheal) จะไม่เห็นสถานะ")

    networks = (data.get("NetworkSettings") or {}).get("Networks") or {}
    primary_name = network if network in networks else ("bridge" if network in ("", "default") else network)

    def _aliases(settings: dict) -> list[str]:
        # docker เติม id สั้นของ container เป็น alias ให้เอง และชื่อ container ก็ resolve ได้อยู่แล้ว
        return [a for a in (settings or {}).get("Aliases") or []
                if a and a != name and not (container_id and container_id.startswith(a))]

    if own_network:
        primary = networks.get(primary_name) or {}
        # alias ของ service ใน compose คือชื่อที่ container อื่น (gateway · open-webui) ใช้เรียกโมเดลนี้
        options += [["--network-alias", alias] for alias in _aliases(primary)]
        ipam = primary.get("IPAMConfig") or {}
        if ipam.get("IPv4Address"):
            options.append(["--ip", str(ipam["IPv4Address"])])
        if ipam.get("IPv6Address"):
            options.append(["--ip6", str(ipam["IPv6Address"])])
        options += [["--link-local-ip", str(ip)] for ip in ipam.get("LinkLocalIPs") or []]
    for other, settings in networks.items():
        if other == primary_name or shared:
            continue
        alias_flags = "".join(f" --alias {shlex.quote(a)}" for a in _aliases(settings))
        notes.append(
            f"container ต่ออยู่กับ network {other} ด้วย — docker run ต่อได้ network เดียว ({network or 'bridge'}) · "
            f"หลัง start ทุกครั้งต้องต่อเอง: docker network connect{alias_flags} {shlex.quote(other)} {shlex.quote(name)}")
    return options, notes


def _restart_policy(host: dict) -> str:
    policy = host.get("RestartPolicy")
    if not isinstance(policy, dict):
        return ""          # payload ไม่บอก — render_controller ใช้ unless-stopped เหมือนที่ทำมาตลอด
    name = str(policy.get("Name") or "no")
    retries = int(policy.get("MaximumRetryCount") or 0)
    return f"{name}:{retries}" if name == "on-failure" and retries > 0 else name


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
    # id ของ image ที่ container ใช้อยู่จริงก่อน (แท็กอาจถูกย้ายไปชี้ image อื่นแล้ว) → ไม่มีค่อยใช้ชื่อ
    image_config = _inspect_image(str(data.get("Image") or config.get("Image") or ""))
    host_options, host_notes = _host_options(host)
    config_options, config_notes = _config_options(data, image_config)
    return Adopted(
        path=str(data.get("Path") or ""),
        image_env=list(image_config.get("Env") or []) if image_config is not None else None,
        mounts=[m for m in host.get("Mounts") or [] if isinstance(m, dict)],
        published=dict((data.get("NetworkSettings") or {}).get("Ports") or {}),
        restart=_restart_policy(host),
        options=host_options + config_options,
        unreproduced=host_notes + config_notes,
        container=data["Name"].lstrip("/"),
        image=config.get("Image") or "",
        args=list(data.get("Args") or []),
        env=list(config.get("Env") or []),
        binds=list(host.get("Binds") or []),
        ports=dict(host.get("PortBindings") or {}),
        network=host.get("NetworkMode") or "",
        runtime=host.get("Runtime") or "",
        device_requests=list(host.get("DeviceRequests") or []),
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
    # ทั้ง `-v` และ `--mount type=bind` (HostConfig.Mounts) — โฟลเดอร์โมเดลที่ mount ด้วยแบบหลังเคยหาไม่เจอ
    for host_dir, cont_dir in adopted.bind_pairs():
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
        # modalities คือคีย์ที่ feature_summary/การ์ดอ่าน — `projector` เฉย ๆ ไม่มีใครแปลเป็นป้าย vision
        # (gemma-4-31B-it ที่ adopt บน vLLM ขึ้น text ล้วน · audit 2026-09-08)
        features["multimodal"] = {"projector": True, "modalities": ["image", "text"]}

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
        for source, _target in adopted.bind_pairs():
            host_dir = Path(source)
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
    mount_binds = [f"{m['Source']}:{m['Target']}" + (":ro" if m.get("ReadOnly") else "")
                   for m in adopted.mounts or []
                   if str(m.get("Type") or "").lower() == "bind" and m.get("Source") and m.get("Target")]
    if adopted.binds or mount_binds:
        out["binds"] = list(adopted.binds) + mount_binds
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
#
# รายการ prefix นี้เป็น **ทางสำรอง** เท่านั้น (audit 2026-10-06): มันเป็นการเดาว่าชื่อแบบไหนผู้ใช้ตั้งเอง
# และเดาผิดเงียบ ๆ มาตลอด · ทางหลักคือถาม image ว่ามันตั้งอะไรไว้เอง แล้วเอาที่เหลือทั้งหมด (meaningful_env)
# ใช้รายการนี้เฉพาะตอนถาม image ไม่ได้ และตอนนั้นตัวที่ไม่ได้ใส่ต้องถูกบอกชื่อ (env_left_out)
_KEEP_ENV_PREFIXES = ("MODEL", "PORT", "MAX_", "VLLM_", "HF_", "CTX_", "API_", "SERVED_",
                      "NCCL_", "CUDA_VISIBLE_DEVICES", "TOKENIZERS_",
                      # llama.cpp ทั้งตัวตั้งค่าผ่าน LLAMA_ARG_* ได้ — ทิ้งชุดนี้ = คำสั่งที่ไม่มีโมเดล
                      "LLAMA_", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "OMP_",
                      "PYTORCH_", "TORCH_", "TRANSFORMERS_", "SGLANG_",
                      "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def meaningful_env(adopted: Adopted) -> list[str]:
    """env ที่ผู้ใช้สั่งตอนรัน — ของ container ลบด้วยของ image · ถาม image ไม่ได้จึงถอยไปใช้รายการ prefix"""
    if adopted.image_env is not None:
        baked = set(adopted.image_env)
        return [e for e in adopted.env if e not in baked]
    return [e for e in adopted.env if e.split("=", 1)[0].startswith(_KEEP_ENV_PREFIXES)]


def env_left_out(adopted: Adopted) -> list[str]:
    """**ชื่อ** ของ env ที่ไม่ได้ใส่กลับเพราะแยกไม่ออกว่าเป็นของ image หรือของผู้ใช้ — ไม่มีค่า (อาจเป็นความลับ)"""
    if adopted.image_env is not None:
        return []
    kept = set(meaningful_env(adopted))
    return [e.split("=", 1)[0] for e in adopted.env if e not in kept]




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
    # สิ่งที่ adopt_process ทำให้/ทำไม่ได้ และคนสั่งควรรู้ (ค่าความลับที่ถอดออก · สคริปต์เดิมที่เก็บไว้)
    notes: list[str] = field(default_factory=list)

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
#
# `TOKEN(?!S|IZER)`: ของเดิมจับคำว่า TOKEN ที่ไหนก็ได้ในชื่อ → `TOKENIZERS_PARALLELISM=false` และ
# `MAX_TOKENS=…` ถูกนับเป็นความลับ ค่าถูกถอดออก เหลือ `--env TOKENIZERS_PARALLELISM` เฉย ๆ ซึ่ง docker
# ข้ามเมื่อเชลล์ไม่มีค่า = ตั้งค่าของผู้ใช้หายเงียบ ๆ อีกทางหนึ่ง (เจอตอน audit 2026-10-06)
_SECRET_ENV = re.compile(r"(TOKEN(?!S|IZER)|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|CREDENTIAL)", re.IGNORECASE)

# ในบรรดาความลับที่ถูกถอดค่าออก มีแค่ "API key ของ model server" ที่ระบบเรามีที่เก็บให้
# (`lmds key` → ~/.lmds/keys/<slug>) · HF token / รหัสผ่าน proxy เป็นของคนละเรื่อง
# เติมค่าจากที่เก็บลงตัวที่ไม่ใช่ API key = ส่งของผิดไปให้ engine
_API_KEY_ENV = re.compile(r"API_?KEY", re.IGNORECASE)

# รหัสผ่านที่ฝังใน URL (`http://user:pass@proxy:3128`) — ชื่อ env (HTTPS_PROXY) ไม่บอกว่าเป็นความลับ
_URL_PASSWORD = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
# รูปแบบ token ที่รู้จัก (ชุดเดียวกับ secrets.redact) แต่ต้อง **ขึ้นต้นคำ** — ตัวของ redact ไม่มีขอบ ซึ่งถูกสำหรับ
# การปิดข้อความใน log แต่ที่นี่ผลของการจับผิดคือ *ค่าถูกถอดออกจากคำสั่ง*: `/models/risk-assessment-model-v2`
# มี "sk-assessment-model-v2" อยู่ข้างใน ถ้าจับ = โมเดลหายจากคำสั่งแล้ว start ไม่ขึ้น
_TOKEN_SHAPES = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)(bearer\s+)[a-z0-9_\-.~+/]{16,}=*"),
)


def _looks_secret(value: str) -> bool:
    """ค่าหน้าตาเป็นความลับ ทั้งที่ชื่อไม่บอก — รูปแบบ token ที่รู้จัก หรือ URL ที่มีรหัสผ่าน"""
    return bool(_URL_PASSWORD.search(value)) or any(p.search(value) for p in _TOKEN_SHAPES)


def redact_secrets(env_items: list[str]) -> tuple[list[str], list[str]]:
    """คืน (env ที่จะเขียนลงสคริปต์, ชื่อที่ถูกถอดค่าออก)

    ตัวที่เป็นความลับเหลือแค่ชื่อ — `docker run --env NAME` หยิบค่าจาก environment ของ
    เชลล์ที่สั่ง start ซึ่งเป็นที่ที่ค่านั้นควรอยู่ · ค่าที่ไม่มีอยู่จริงตอน start = docker ข้ามให้
    """
    kept, redacted = [], []
    for item in env_items:
        name, sep, value = item.partition("=")
        if sep and (_SECRET_ENV.search(name) or _looks_secret(value)):
            kept.append(name)
            redacted.append(name)
        else:
            kept.append(item)
    return kept, redacted


# ── ความลับบน argv ────────────────────────────────────────────────────────────────────────
# audit 2026-10-06: ของเดิมถอดค่าออกเฉพาะ env · `--api-key sk-…` / `--hf-token hf_…` บน argv ถูกเขียนลง
# สคริปต์ 0755 ทั้งดุ้น (ทั้งทาง container และทาง process) แล้วติดไปกับ zip ที่ `lmds node push` ส่งข้ามเครื่อง
#
# ชื่อธงที่ค่าของมันคือความลับ · คำว่า token ต้องเป็นธงที่รู้จักเท่านั้น — `--max-tokens` `--tokenizer`
# `--max-num-batched-tokens` ไม่ใช่ความลับ และถอดค่าออก = คำสั่งพัง · `--api-key-file` เป็น path ไม่ใช่ key
_SECRET_FLAG = re.compile(
    r"^--?(?:(?:[a-z0-9]+[-_])*(?:api[-_]?keys?|secret(?:[-_]key)?|password|passwd)"
    r"|(?:hf|auth|access|hub|huggingface|bearer|api)[-_]token|token|hft)$", re.IGNORECASE)
# ชื่อตัวแปรแบบ `HF_TOKEN=…` ที่เขียนอยู่ *ในคำสั่ง* (`bash -c "export HF_TOKEN=… && serve"`)
_SECRET_ASSIGN = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|CREDENTIALS?)$", re.IGNORECASE)
_VALUE = r"""("[^"]*"|'[^']*'|[^\s;&|<>()'"]+)"""
_INLINE_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)(=|\s+)" + _VALUE)
_INLINE_ASSIGN = re.compile(r"(?<![\w-])([A-Za-z_][A-Za-z0-9_]*)=" + _VALUE)


@dataclass
class ArgSecret:
    """ค่าความลับหนึ่งตัวที่ถูกถอดออกจาก argv — สคริปต์เหลือแค่ชื่อตัวแปรที่ start จะอ่าน"""

    var: str        # ตัวแปรเชลล์ที่ start อ่านค่า (LMDS_ARG_API_KEY · HF_TOKEN)
    label: str      # ที่มา สำหรับข้อความถึงคน (--api-key · HF_TOKEN= ในคำสั่ง)
    value: str      # ค่าจริง — อยู่ในหน่วยความจำเท่านั้น (adopt ใช้เก็บ API key ลงที่เก็บของเครื่อง)
    required: bool  # ไม่มีค่าตอน start = ไม่ start (True) หรือข้ามธงนั้นไป (False)

    @property
    def is_api_key(self) -> bool:
        return bool(_API_KEY_ENV.search(self.var))


def _dq(text: str) -> str:
    """escape สำหรับวางในเครื่องหมายคำพูดคู่ของ bash — ได้ตัวอักษรเดิมกลับมาทุกตัว ไม่มีอะไรถูกรัน

    ค่าที่ไม่มีอักขระพิเศษได้ข้อความเดิมเป๊ะ สคริปต์ของ bundle ที่ไม่มีปัญหาจึงไม่เปลี่ยนสักบรรทัด
    """
    return re.sub(r'([\\"$`])', r"\\\1", text)


def render_args(args: list[str]) -> tuple[list[str], list[ArgSecret]]:
    """argv → (คำของเชลล์ทีละตัว พร้อมวางในสคริปต์, ความลับที่ถูกถอดค่าออก)

    กลไกเดียวกับ env (`redact_secrets`): ค่าไม่อยู่ในไฟล์ · start อ่านจากตัวแปรของเชลล์ และตัวที่เป็น
    API key ของ model server ถูกเติมจาก ~/.lmds/keys/<slug> (load_api_key) · ต่างกันข้อเดียว:

      - `--env NAME` ที่ไม่มีค่า docker ข้ามให้เอง · ธงบน argv ต้องตัดสินใจเอง:
        API key ไม่มีค่า = **ไม่ start** (ของเดิมมี auth — ขึ้นมาแบบเปิดโล่งเงียบ ๆ แย่กว่าไม่ขึ้น)
        ความลับอื่น (`--hf-token`) ไม่มีค่า = ข้ามธงนั้นไปเหมือน docker ข้าม env
      - ความลับที่ฝังอยู่กลางสตริงของ `bash -c "…"` ข้ามไม่ได้ จึงต้องมีค่าเสมอ
    """
    words: list[str] = []
    secrets: list[ArgSecret] = []
    used: dict[str, int] = {}

    def _name(base: str) -> str:
        used[base] = used.get(base, 0) + 1
        return base if used[base] == 1 else f"{base}_{used[base]}"

    def _flag_var(flag: str) -> str:
        return _name("LMDS_ARG_" + re.sub(r"[^A-Za-z0-9]+", "_", flag.lstrip("-")).upper().strip("_"))

    index = 0
    while index < len(args):
        item = str(args[index])
        following = str(args[index + 1]) if index + 1 < len(args) else None
        if _SECRET_FLAG.match(item) and following is not None and not following.startswith("--"):
            var = _flag_var(item)
            secret = ArgSecret(var, item, following, required=bool(_API_KEY_ENV.search(var)))
            secrets.append(secret)
            pair = f'{shlex.quote(item)} "${{{var}}}"'
            words.append(pair if secret.required else f"${{{var}:+{pair}}}")
            index += 2
            continue
        flag, equals, value = item.partition("=")
        if equals and value and _SECRET_FLAG.match(flag):
            var = _flag_var(flag)
            secret = ArgSecret(var, flag, value, required=bool(_API_KEY_ENV.search(var)))
            secrets.append(secret)
            whole = f'"{_dq(flag)}=${{{var}}}"'
            words.append(whole if secret.required else f"${{{var}:+{whole}}}")
            index += 1
            continue

        spans = _inline_secrets(item)
        if not spans:
            words.append(shlex.quote(item))
            index += 1
            continue
        out, position = ['"'], 0
        for start, end, base, label in spans:
            secret = ArgSecret(_name(base), label, item[start:end], required=True)
            secrets.append(secret)
            out.append(_dq(item[position:start]))
            out.append(f"${{{secret.var}}}")
            position = end
        out.append(_dq(item[position:]) + '"')
        words.append("".join(out))
        index += 1
    return words, secrets


def _inline_secrets(item: str) -> list[tuple[int, int, str, str]]:
    """ความลับที่ฝังอยู่ *ใน* argv ตัวเดียว → [(เริ่ม, จบ, ชื่อตัวแปร, ที่มา)] เรียงตามตำแหน่ง ไม่ทับกัน

    เคส spark-03 2026-08-27: ทั้งคำสั่งอยู่ในสตริงเดียวของ `bash -c "…"` — ไล่ทีละ argv ไม่มีวันเจอ --api-key
    """
    spans: list[tuple[int, int, str, str]] = []

    def take(start: int, end: int, base: str, label: str) -> None:
        if start < end and not any(start < e and s < end for s, e, _, _ in spans):
            spans.append((start, end, base, label))

    def unquoted(match: re.Match, group: int) -> tuple[int, int]:
        start, end = match.span(group)
        text = match.group(group)
        quoted = len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'"
        return (start + 1, end - 1) if quoted else (start, end)

    for match in _INLINE_FLAG.finditer(item):
        if _SECRET_FLAG.match(match.group(1)) and not match.group(3).startswith("-"):
            base = "LMDS_ARG_" + re.sub(r"[^A-Za-z0-9]+", "_", match.group(1).lstrip("-")).upper().strip("_")
            take(*unquoted(match, 3), base, match.group(1))
    for match in _INLINE_ASSIGN.finditer(item):
        if _SECRET_ASSIGN.search(match.group(1)):
            # ชื่อเดียวกับตัวแปรในคำสั่ง — export ไว้ในเชลล์ก่อน start เหมือนทาง `--env NAME`
            take(*unquoted(match, 2), match.group(1), f"{match.group(1)}= ในคำสั่ง")
    for pattern in _TOKEN_SHAPES:      # ตาข่ายชั้นสอง: รูปแบบ token ที่รู้จัก ใต้ธงที่ชื่อไม่บอกว่าเป็นความลับ
        for match in pattern.finditer(item):
            start = match.end(1) if (match.lastindex or 0) >= 1 else match.start()
            take(start, match.end(), "LMDS_ARG_SECRET", "ค่าที่หน้าตาเป็น token")
    return sorted(spans)


def _secret_guard(slug: str, secrets: list[ArgSecret]) -> str:
    """บรรทัดใน start ที่ตรวจว่าความลับของ argv มีค่าแล้ว — ก่อนแตะ docker/process"""
    lines = []
    for secret in secrets:
        if secret.required and secret.is_api_key:
            message = (f"ค่าของ {secret.label} ไม่ได้เก็บไว้ในไฟล์นี้ และที่เก็บ key ของเครื่องยังไม่มี — "
                       f"ของเดิมรันแบบมี API key จึงไม่ start แบบเปิดโล่งให้ · ตั้งก่อน: "
                       f'echo -n "$KEY" | lmds key set {slug}  (หรือ export {secret.var}=…)')
            lines.append(f'  [[ -n "${{{secret.var}:-}}" ]] || die {shlex.quote(message)}\n')
        elif secret.required:
            message = (f"ค่าของ {secret.label} ไม่ได้เก็บไว้ในไฟล์นี้ — export {secret.var}=… ในเชลล์ก่อน start")
            lines.append(f'  [[ -n "${{{secret.var}:-}}" ]] || die {shlex.quote(message)}\n')
        else:
            message = (f"หมายเหตุ: ไม่มีค่า {secret.var} ในเชลล์ — start โดยไม่ใส่ {secret.label} "
                       f"(ค่าเดิมไม่ได้เก็บไว้ในไฟล์นี้)")
            lines.append(f'  [[ -n "${{{secret.var}:-}}" ]] || echo {shlex.quote(message)} >&2\n')
    return "".join(lines)


def _scrub(text: str, known: list[str]) -> str:
    """ลบค่าความลับที่รู้ออกจากข้อความ (สคริปต์รุ่นเก่าที่จะเก็บสำรอง) + รูปแบบ token ที่รู้จัก"""
    for value in sorted({v for v in known if v and len(v) >= 4}, key=len, reverse=True):
        text = text.replace(value, MASK)
    return redact(text)


def _default_assign(name: str, value: str) -> str:
    """`NAME="${NAME:-value}"` — ค่าที่มีอักขระของเชลล์ถูกกำหนดแยกบรรทัดด้วย quote เดี่ยว

    `${NAME:-…}` ในเครื่องหมายคำพูดคู่ยัง expand `$(…)` ในค่าตั้งต้นอยู่ · ค่าพวกนี้ (path ของ binary ·
    cwd · ชื่อ image) มาจากของที่รันอยู่ ซึ่งใครก็ตามบนเครื่องตั้งได้ · ค่าธรรมดาได้บรรทัดเดิมเป๊ะ
    """
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=,~-]*", value):
        return f'{name}="${{{name}:-{value}}}"'
    return f'{name}="${{{name}:-}}"; [[ -n "${name}" ]] || {name}={shlex.quote(value)}'


def mask_args(argv: list[str], values: list[str]) -> list[str]:
    """argv ที่ค่าความลับถูกแทนด้วย [REDACTED] — สำหรับจดลงไฟล์ที่คนอ่าน (profile · ข้อความ error)"""
    secrets = {v for v in values if v}
    out = []
    for item in argv:
        if item in secrets:
            out.append(MASK)
            continue
        for value in sorted((v for v in secrets if len(v) >= 4), key=len, reverse=True):
            item = item.replace(value, MASK)
        out.append(item)
    return out


def _meta_value(value: object) -> str:
    """ค่าหนึ่งบรรทัดของ server.meta — ขึ้นบรรทัดใหม่ในชื่อโมเดล = แทรกคีย์อื่น (`controller=…`) ได้"""
    return " ".join(str(value).split())


# คำสั่ง start ที่ไป "ดาวน์โหลดก่อนแล้วค่อยเสิร์ฟ" (`hf download X && …serve X`) ทำงานได้ตอนแรก
# แต่กลายเป็นระเบิดเวลา: repo ที่ gated + token หมดอายุ = ดึง revision ใหม่ได้ครึ่งเดียว (401)
# แล้ว serve ชี้ไปที่ snapshot ที่ไม่ครบ · เคสจริง dgx-spark03 2026-09-03: สร้าง container ใหม่
# จากคำสั่งเดิมเป๊ะ → วนล้ม 15 รอบ ทั้งที่ snapshot ที่ครบอยู่บนดิสก์มาตั้งแต่ มิ.ย.
_DOWNLOAD_THEN_SERVE = re.compile(r"^\s*(hf|huggingface-cli)\s+download\s+(\S+)[^&]*&&", re.MULTILINE)


def download_before_serve(command: str) -> str:
    """repo id ที่คำสั่งจะไปดึงก่อนเสิร์ฟ — ว่างเมื่อไม่มีขั้นนั้น"""
    m = _DOWNLOAD_THEN_SERVE.search(command or "")
    return m.group(2) if m else ""

def _load_key_block(slug: str, names: list[str]) -> str:
    """บล็อกที่เติม API key จากที่เก็บของเครื่องให้ตัวแปรที่ถูกถอดค่าออก

    สคริปต์ที่ adopt สร้างเป็น 0755 อ่านได้ทุก user ค่าของ key จึงห้ามอยู่ในนั้น — ตัวที่เป็น
    ความลับถูกส่งเป็น `--env ชื่อ` เฉย ๆ แล้วให้ docker หยิบจาก environment ของเชลล์ที่สั่ง
    start ซึ่งถูกตามเจตนา แต่ systemd ตอน autostart ไม่มี environment นั้นให้ →
    docker ข้ามตัวแปรไป → container ขึ้นมาแบบไม่มี auth เงียบ ๆ ทุก reboot

    ที่เก็บอยู่คนละไฟล์ (~/.lmds/keys/<slug> โหมด 0600) หลักเดิมจึงไม่เสีย: สคริปต์ยังไม่มี
    ความลับอยู่ข้างใน แค่ไปอ่านตอน start · env ที่ตั้งมาจากภายนอกชนะไฟล์นี้เสมอ
    """
    if not names:
        # ไม่มีตัวแปรชื่อ *API_KEY* ใน container นี้ = เราไม่รู้ว่า engine ตัวนี้อ่าน key จากไหน
        # เขียนไว้ตรง ๆ ดีกว่าปล่อยให้คนคิดว่า `lmds key` คุมตัวนี้อยู่
        return ("# LMDS ไม่เห็นตัวแปรชื่อ *API_KEY* ใน container นี้ จึงไม่รู้ว่า engine อ่าน key จากไหน\n"
                "# — `lmds key {slug}` ไม่มีผลกับ bundle นี้ · ตั้ง auth ที่คำสั่งของ engine เอง\n"
                "load_api_key() {{ :; }}\n").format(slug=slug)
    loop = " ".join(shlex.quote(n) for n in names)
    return (
        "# API key ของโมเดลนี้เก็บอยู่ที่ ~/.lmds/keys/<slug> (0600) ไม่ได้อยู่ในไฟล์นี้ซึ่งเป็น 0755\n"
        "# systemd ตอน autostart เรียกสคริปต์นี้โดยไม่มี environment ของเชลล์ — ไม่เติมให้\n"
        "# container จะขึ้นมาแบบไม่มี auth ทุก reboot · env ที่ตั้งมาจากภายนอกชนะไฟล์นี้เสมอ\n"
        "load_api_key() {{\n"
        '  local store="${{LMDS_KEY_ROOT:-${{HOME:-}}/.lmds/keys}}/{slug}"\n'
        '  [[ -r "$store" ]] || return 0\n'
        '  local value; value="$(tr -d \'\\r\\n\' < "$store")"\n'
        '  [[ -n "$value" ]] || return 0\n'
        "  local name\n"
        "  for name in {loop}; do\n"
        '    [[ -n "${{!name:-}}" ]] || export "$name=$value"\n'
        "  done\n"
        "}}\n"
    ).format(slug=slug, loop=loop)


def _ask_block(slug: str, names: list[str]) -> str:
    """api_curl + api_state: ทุกคำถามที่สคริปต์ถาม server ของตัวเอง แนบ key ตัวเดียวกับที่ start ส่งให้

    เคสจริง AI-Local-ISIT 2026-10-10: adopt container ของ Strata ที่รันด้วย `-e API_KEY=…` · server
    ตอบ /health 200 และตอบคำขอที่มี key ปกติ แต่ `status` พิมพ์ "api: ยังไม่ตอบ" เพราะ status /
    test-text / client-config ยิง /v1/models เปล่า ๆ ได้ 401 ทุกครั้ง ทั้งที่ load_api_key ของไฟล์
    เดียวกันรู้จัก key ตัวนั้นอยู่แล้ว · test-text ล้มกับ server ที่ดีอยู่ และ client-config ตกไปใช้
    slug เป็นชื่อโมเดล (client เอาไปเรียกได้ 404) · vLLM/SGLang ที่ตั้ง --api-key เป็นแบบเดียวกัน —
    ที่ไม่เคยเห็นเพราะ llama.cpp เปิด /v1/models สาธารณะเสมอ (ดู doctor/checks.py)

    names ว่าง = ไม่รู้ว่า engine อ่าน key จากไหน จึงไม่แนบอะไร (และไม่แนบของที่บังเอิญอยู่ในที่เก็บ)
    """
    if names:
        loop = " ".join(shlex.quote(n) for n in names)
        ask = (
            "api_curl() {{\n"
            "  load_api_key\n"
            '  local name key=""\n'
            '  for name in {loop}; do key="${{!name:-}}"; [[ -z "$key" ]] || break; done\n'
            '  if [[ -n "$key" ]]; then curl -H "Authorization: Bearer ${{key}}" "$@"; else curl "$@"; fi\n'
            "}}\n"
        ).format(loop=loop)
        refused = f"ไม่รับ key ที่ LMDS เก็บไว้ (HTTP ${{code}}) — ตั้งให้ตรงกับของ server: lmds key set {slug}"
    else:
        ask = 'api_curl() { curl "$@"; }\n'
        refused = "ต้องใช้ key (HTTP ${code}) — LMDS ไม่รู้ว่า engine นี้อ่าน key จากไหน จึงถามแทนไม่ได้"
    # "ไม่ตอบ" กับ "ตอบแต่ไม่รับ key" แก้กันคนละที่ (restart กับตั้ง key) — รวมเป็นคำเดียวคือส่งคนไปผิดทาง
    return (
        "# คำถามที่สคริปต์นี้ถาม server ของตัวเอง (status · test-text · client-config) แนบ key ตัวเดียวกับที่\n"
        "# start ส่งให้ — ถามเปล่า ๆ กับ server ที่บังคับ key ได้ 401 แล้วรายงานว่า \"ยังไม่ตอบ\" ทั้งที่มันตอบอยู่\n"
        + ask +
        "api_state() {\n"
        "  local code\n"
        "  code=\"$(api_curl -s -o /dev/null -m 5 -w '%{http_code}' \"http://127.0.0.1:${API_PORT}/v1/models\" 2>/dev/null || true)\"\n"
        '  case "$code" in\n'
        '    200)     echo "api: ตอบปกติ" ;;\n'
        f'    401|403) echo "api: ตอบอยู่ แต่{refused}" ;;\n'
        '    ""|000)  echo "api: ยังไม่ตอบ" ;;\n'
        '    *)       echo "api: ตอบ HTTP ${code} ที่ /v1/models" ;;\n'
        "  esac\n"
        "}\n"
    )


def _gpus_value(adopted: Adopted) -> tuple[str, list[str]]:
    """ค่าของ `--gpus` ("" = ไม่ใส่) + สิ่งที่แทนไม่ได้

    GPU มาได้สองทาง และเช็คทางเดียวไม่พอ:
      · `--runtime=nvidia` (ทางเก่า)  → HostConfig.Runtime == "nvidia"
      · `--gpus all`       (ทางใหม่)  → HostConfig.DeviceRequests มี capability "gpu"
                                        ส่วน Runtime ยังเป็น "runc" ตาม default ของ daemon

    เคสจริง 2026-09-27 msi-1: คอนเทนเนอร์ vLLM ที่รันด้วย `--gpus all` ถูก adopt แล้ว
    controller ที่เขียนออกมา **ไม่มี --gpus เลย** เพราะเช็คแค่ Runtime == "nvidia"
    ครั้งถัดไปที่ใครสั่ง start โมเดลจะขึ้นโดยไม่เห็น GPU — เงียบสนิท ไม่มีอะไรฟ้องตอน adopt

    audit 2026-10-06 — ครึ่งหลังของเรื่องเดียวกัน: "ขอ GPU" ถูกเขียนกลับเป็น `--gpus all` เสมอ ·
    `--gpus device=1` (DeviceIDs ["1"]) บนเครื่องสองการ์ดจึงกลายเป็นทุกการ์ดหลัง restart แล้วไปจอง VRAM
    ของโมเดลอีกตัว · ทางเก่าก็เช่นกัน: `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=1` ถูกแปลงเป็น
    `--gpus all` ซึ่ง daemon เขียนทับ NVIDIA_VISIBLE_DEVICES เป็น all
    """
    notes: list[str] = []
    requests = [r for r in adopted.device_requests or [] if _is_gpu_request(r)]
    if requests:
        request = requests[0] or {}
        if len(requests) > 1:
            notes.append(f"container ขอ GPU ไว้ {len(requests)} ชุด — --gpus ใส่ได้ชุดเดียว ใช้ชุดแรก")
        ids = [str(i) for i in request.get("DeviceIDs") or [] if str(i)]
        count = request.get("Count")
        if ids:
            fields = ["device=" + ",".join(ids)]
        elif isinstance(count, int) and count > 0:
            fields = [str(count)]
        else:
            fields = ["all"]
        driver = str(request.get("Driver") or "")
        if driver and driver != "nvidia":
            fields.append(f"driver={driver}")
        extra = [c for group in request.get("Capabilities") or [] for c in group or [] if str(c).lower() != "gpu"]
        if extra:      # docker เติม "gpu" ให้เอง
            fields.append("capabilities=" + ",".join(dict.fromkeys(str(c) for c in extra)))
        if request.get("Options"):
            notes.append("DeviceRequests.Options ของ GPU — LMDS เขียนกลับเป็น --gpus ไม่ได้")
        return _csv(fields), notes
    if adopted.runtime == "nvidia":
        # ทางเก่า: runtime ตัดสินจาก NVIDIA_VISIBLE_DEVICES · เจาะจงการ์ดไว้ = ปล่อยให้ --runtime + env
        # (ซึ่งยกมาทั้งคู่แล้ว) ทำงานเหมือนเดิม ไม่ใส่ --gpus ไปทับ
        visible = adopted.env_value("NVIDIA_VISIBLE_DEVICES")
        return ("all" if visible in (None, "", "all") else ""), notes
    return "", notes


def _publish_lines(ports: dict) -> str:
    """`--publish` ครบทุก host binding พร้อม protocol

    เดิมทิ้ง HostIp → container ที่เคย bind แค่ 127.0.0.1 กลายเป็นเปิดทุก interface หลัง adopt (รีวิว 0.6.0) ·
    audit 2026-10-06: ยังเอาแค่ binding ตัวแรกของแต่ละพอร์ต (ตัวที่สองหาย) และทิ้ง `/udp`
    (พอร์ต UDP ถูก publish กลับเป็น TCP)
    """
    lines = []
    for spec, bindings in (ports or {}).items():
        number, _, proto = str(spec).partition("/")
        suffix = f"/{proto}" if proto and proto != "tcp" else ""
        for binding in bindings or []:
            host_ip = str((binding or {}).get("HostIp") or "").strip()
            host_port = str((binding or {}).get("HostPort") or "").strip()
            if host_ip in ("", "0.0.0.0"):
                prefix = ""
            elif ":" in host_ip:
                prefix = f"[{host_ip}]:"
            else:
                prefix = f"{host_ip}:"
            if host_port:
                value = f"{prefix}{host_port}:{number}{suffix}"
            else:          # `-p 8000` — docker สุ่มพอร์ตฝั่งเครื่องให้ (not_reproduced เตือนไว้แล้ว)
                value = f"{prefix}:{number}{suffix}" if prefix else f"{number}{suffix}"
            lines.append(f"  --publish {shlex.quote(value)} \\\n")
    return "".join(lines)


def not_reproduced(adopted: Adopted) -> list[str]:
    """ทุกอย่างที่ `docker run` ในสคริปต์ **ไม่เหมือน** กับที่ container ถูกสั่งรันมา — หนึ่งข้อหนึ่งบรรทัด

    หลักของ adopt (audit 2026-10-06): คำสั่งที่สร้างใหม่ต้องเท่ากับของเดิม **หรือบอกว่าตรงไหนไม่เท่า** ·
    บั๊กของ adopt ทุกตัวที่ผ่านมา (--gpus หาย · HEALTHCHECK · mount หาย) มีรูปเดียวกันคือเขียนสคริปต์สำเร็จ
    โดยไม่มี error แล้วของหายตอน restart · รายการนี้ไปโผล่สามที่: หน้าจอของ `lmds adopt` ·
    `not_reproduced` ใน MODEL_PROFILE.yaml · บล็อกคอมเมนต์ที่หัว controller

    ว่าง = ไม่มีอะไรต่างเท่าที่ LMDS มองเห็น · ข้อความไม่มีค่าความลับ (มีแต่ชื่อ)
    """
    lines: list[str] = []
    _, env_secrets = redact_secrets(meaningful_env(adopted))
    if env_secrets:
        lines.append(f"ค่าของ env {' '.join(env_secrets)} ไม่ได้เขียนลงสคริปต์ (ไฟล์ 0755) — ตอน start docker หยิบจาก "
                     f"environment ของเชลล์ · ตัวที่เป็น API key เติมจากที่เก็บ key ของเครื่องให้ (lmds key) · "
                     f"ไม่มีค่า = container ขึ้นมาโดยไม่มีตัวแปรนั้น")
    _, arg_secrets = render_args(list(adopted.args))
    for secret in arg_secrets:
        if secret.required and secret.is_api_key:
            fate = "เติมจากที่เก็บ key ของเครื่อง (lmds key) หรือ env · ไม่มีค่า = ไม่ start (ไม่ยอมขึ้นแบบไม่มี auth)"
        elif secret.required:
            fate = "ต้อง export ไว้ในเชลล์ก่อน start · ไม่มีค่า = ไม่ start"
        else:
            fate = "export ไว้ในเชลล์ก่อน start · ไม่มีค่า = start โดยไม่ใส่ธงนี้"
        lines.append(f"ค่าของ {secret.label} บน argv ไม่ได้เขียนลงสคริปต์ (ไฟล์ 0755) — อ่านจาก ${secret.var}: {fate}")
    if adopted.restart == "no":
        lines.append("ของเดิมไม่มี restart policy — สคริปต์นี้ใส่ --restart unless-stopped ให้ "
                     "(container กลับมาเองหลัง reboot/ล้ม ซึ่งของเดิมไม่ทำ)")
    lines += list(adopted.unreproduced)
    lines += _gpus_value(adopted)[1]
    dropped = env_left_out(adopted)
    if dropped:
        lines.append("ถาม image ไม่ได้ (docker image inspect) จึงแยก env ของผู้ใช้ออกจากของ image ไม่ได้ — "
                     f"ใส่กลับเฉพาะชื่อที่รู้จัก · ที่ไม่ได้ใส่: {' '.join(dropped)} · "
                     "ตัวไหนเป็นของที่ตั้งเอง ให้เพิ่ม --env ในสคริปต์")
    inner = adopted.container_port
    if inner and adopted.network and not adopted.shares_host_network:
        if adopted.host_port_is_ephemeral:
            lines.append(f"พอร์ต {inner} ถูก publish แบบให้ docker สุ่มพอร์ตฝั่งเครื่อง (ตอนนี้ {adopted.port}) — "
                         f"start ครั้งหน้าได้เลขใหม่ แต่สคริปต์และทะเบียนจดเลขตอนนี้ไว้ · ควรตรึงพอร์ตใน --publish")
        elif not adopted._host_port(inner):
            lines.append(f"พอร์ต {inner} ที่ API ฟังอยู่ไม่ได้ publish ออกมาที่เครื่อง — status/test-text/watchdog "
                         f"ของ LMDS เคาะ 127.0.0.1:{inner} ไม่ถึง (เข้าได้เฉพาะจาก network {adopted.network})")
    return [_one_line(redact(line)) for line in lines]


def _header_notes(lines: list[str]) -> str:
    """บล็อกคอมเมนต์ที่หัว controller — คนที่กำลังจะกด restart เปิดไฟล์มาก็เจอ"""
    if not lines:
        return ""
    body = "".join(f"#   - {_one_line(line)}\n" for line in lines)
    return ("#\n"
            f"# ⚠ ต่างจาก container เดิม {len(lines)} ข้อ — อ่านก่อนสั่ง start/restart (stop ของสคริปต์นี้ลบ container เดิมทิ้ง):\n"
            + body)


def render_controller(adopted: Adopted, slug: str, notes: list[str] | None = None) -> str:
    """สคริปต์ที่รัน container เดิมซ้ำได้ — คำสั่งเดียวกับที่มันรันอยู่ตอนนี้

    notes = ข้อที่ผู้เรียกรู้เพิ่ม (เช่น adopt() เก็บ API key ลงที่เก็บของเครื่องไม่ได้) ต่อท้ายรายการที่หัวไฟล์
    """
    env_items, redacted = redact_secrets(meaningful_env(adopted))
    fetch_repo = download_before_serve(" ".join(adopted.args or []))
    if fetch_repo:
        env_lines_note = (f"  # ⚠ คำสั่งนี้ไป `hf download {fetch_repo}` ก่อนเสิร์ฟทุกครั้งที่ start — ถ้า repo\n"
                          f"  #   gated และ token หมดอายุ จะได้ snapshot ใหม่ที่ไม่ครบแล้วเสิร์ฟล้ม (dgx-spark03 2026-09-03)\n"
                          f"  #   ถ้า weight อยู่ครบแล้ว ให้ชี้ path ของ snapshot ตรง ๆ และตั้ง HF_HUB_OFFLINE=1\n")
    else:
        env_lines_note = ""
    env_lines = "".join(f'  --env {shlex.quote(e)} \\\n' for e in env_items)
    # หมายเหตุต้องอยู่ *เหนือ* `docker run` ไม่ใช่แทรกกลาง — บรรทัดก่อนหน้าจบด้วย `\` ซึ่ง
    # ต่อเข้าบรรทัดคอมเมนต์ แล้วคำสั่งก็จบตรงนั้นทันที: docker run ถูกยิงโดยไม่มี image และ
    # ไม่มี --env สักตัว ส่วนบรรทัด `--env …` ที่เหลือกลายเป็นคำสั่งใหม่ ("--env: command not
    # found") · ผลคือ adopted bundle ที่มี env ความลับสักตัว (HF_TOKEN/API_KEY/PASSWORD)
    # `start` ไม่ขึ้นเลย — เจอตอนเขียนเทสที่รันสคริปต์จริง 2026-09-21
    run_notes = env_lines_note
    if redacted:
        names = _one_line(" ".join(redacted))
        run_notes += (f"  # ค่าของ {names} ไม่ได้เก็บไว้ในไฟล์นี้ — export ไว้ในเชลล์ก่อน start "
                      f"(docker หยิบจาก environment ให้เอง)\n")
    # argv ผ่านกลไกเดียวกับ env: ค่าความลับไม่อยู่ในไฟล์ (ดู render_args)
    arg_words, arg_secrets = render_args(list(adopted.args))
    key_names = [n for n in redacted if _API_KEY_ENV.search(n)]
    key_names += [s.var for s in arg_secrets if s.is_api_key and s.var not in key_names]
    load_key = _load_key_block(slug, key_names) + _ask_block(slug, key_names)
    secret_guard = _secret_guard(slug, arg_secrets)
    bind_lines = "".join(f'  --volume {shlex.quote(b)} \\\n' for b in adopted.binds)
    port_lines = _publish_lines(adopted.ports)
    # ธงที่ยกมาจาก HostConfig/Config ตรง ๆ (--mount · --ulimit · --cap-add · --user · --label …) — ดู _host_options
    extra_lines = "".join("  " + " ".join(shlex.quote(part) for part in option) + " \\\n"
                          for option in adopted.options)
    # --no-healthcheck: HEALTHCHECK ที่ฝังมาใน image ชี้พอร์ตของ *ตัวมันเอง* ไม่ใช่พอร์ตที่เราสั่งรัน
    #
    # เคสจริง 2026-09-27 msi-1: adopt คอนเทนเนอร์ vLLM (avarok/dgx-vllm-nvfp4-kernel:v23) ซึ่งฝัง
    # `HEALTHCHECK curl -f http://localhost:8888/health` มา แต่ LMDS รันบนพอร์ต 8000 → `docker ps`
    # ขึ้น "(unhealthy)" ตลอดอายุคอนเทนเนอร์ ทั้งที่ `curl :8000/health` ตอบ 200
    #
    # เทมเพลต controller ทุกตัวปิดทิ้งมาตั้งแต่ 2026-09-20 แล้ว (ดูเหตุผลเต็มใน llamacpp controller:
    # LMDS มี wait_health ของตัวเองที่รู้จัก engine · สัญญาณสุขภาพตัวที่สองที่โง่กว่าและขัดกันเองได้
    # แย่กว่าไม่มีเลย) — แต่ทางของ adopt ตกสำรวจ เพราะประกอบคำสั่ง docker run ขึ้นเองที่นี่
    #
    # --entrypoint: Path ของ docker inspect คือ argv[0] ที่ container รันจริง และ Args คือที่เหลือ ·
    # `--entrypoint <Path>` ตามด้วย Args จึงได้ argv เดิมเป๊ะ ไม่ว่า image จะมี ENTRYPOINT หรือไม่ ·
    # ของเดิมใส่ --entrypoint เฉพาะตอน Config.Entrypoint มีค่า (audit 2026-10-06): image ที่ไม่มี
    # ENTRYPOINT (`docker run img vllm serve …` → Path="vllm" · Config.Entrypoint=null) ได้คำสั่งที่
    # ไม่มีตัว `vllm` — start ครั้งหน้าได้ "serve: executable file not found"
    executable = adopted.path or (adopted.entrypoint[0] if adopted.entrypoint else "")
    entry = f'  --entrypoint {shlex.quote(executable)} \\\n' if executable else ""
    network = f'  --network {shlex.quote(adopted.network)} \\\n' if adopted.network not in ("", "default") else ""
    gpus, _gpu_notes = _gpus_value(adopted)
    runtime = f'  --gpus {shlex.quote(gpus)} \\\n' if gpus else ""
    # NCCL/torch.distributed คุยกันผ่าน /dev/shm — docker ให้มาแค่ 64 MB โดยปริยาย
    #
    # เคสจริง 2026-09-01: adopt โมเดล stacked (MiniMax M3 บน SGLang 2 เครื่อง) แล้ว
    # ทิ้ง --ipc host --shm-size ของเดิมไป พอ start ใหม่ head ตายด้วย
    #   "creating shared memory segment /dev/shm/nccl-… No space left on device (28)"
    # ส่วน worker ที่ต่อกลับมาเจอ Connection refused ก็ตายตาม · ที่ร้ายกว่าคือ
    # --restart unless-stopped ปลุก head ซ้ำทุก 10 นาทีจนครบ 31 รอบโดยไม่มีใครรู้
    ipc = f'  --ipc {shlex.quote(adopted.ipc_mode)} \\\n' if adopted.ipc_mode not in ("", "private") else ""
    shm = f'  --shm-size {adopted.shm_size} \\\n' if adopted.shm_size not in (0, 67108864) else ""
    # restart policy ของเดิมคงไว้ (on-failure:3 ถูกบังคับเป็น unless-stopped = ตัวที่ล้มซ้ำถูกปลุกไม่มีวันจบ) ·
    # ของเดิมไม่มี policy ("no") ยังใส่ unless-stopped ให้เหมือนที่ทำมาตลอด — และบอกไว้ใน not_reproduced
    restart = adopted.restart if adopted.restart not in ("", "no") else "unless-stopped"
    args = " ".join(arg_words)
    header_notes = _header_notes(not_reproduced(adopted) + list(notes or []))
    # ค่าที่มาจากของที่รันอยู่ (ชื่อโมเดล · ชื่อ container) ถูกวางในเครื่องหมายคำพูดคู่ของ echo —
    # `$(…)` หรือ `"` ในค่า = รันคำสั่ง/ไฟล์พัง (audit 2026-10-06) · _dq ทำให้เป็นตัวอักษรล้วน
    model_label = _dq(adopted.model or '(ไม่ระบุใน env)')
    container_label = _dq(adopted.container)
    image_label = _dq(adopted.image)
    image_line = _default_assign("IMAGE", adopted.image)
    # สิ่งที่ lmds remove จะแตะ — status/info/remove-plan พิมพ์ให้เห็นก่อน ไม่ใช่รู้ตอนที่ลบไปแล้ว
    weights_label = shlex.quote(_weights_label(weights_on_host(adopted)))

    return f'''#!/usr/bin/env bash
# LMDS adopted controller — สร้างจาก container ที่รันอยู่ก่อนหน้า ไม่ได้ deploy ผ่าน LMDS
#
# สคริปต์นี้ทำได้แค่ "รันคำสั่งเดิมซ้ำ" — weight เป็น path ที่ผู้ใช้จัดการเอง จึงไม่มี
# download/verify-files ให้ · คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มี
{header_notes}set -Eeuo pipefail

SCRIPT_VERSION="${{SCRIPT_VERSION:-1.0.0}}"
ADOPTED=1
CONTAINER_NAME="{container_label}"
{image_line}
API_PORT="${{API_PORT:-{adopted.port or 8000}}}"
SLUG="{slug}"

die() {{ echo "ERROR: $*" >&2; exit 1; }}

# ชื่อโมเดลที่ server เสิร์ฟอยู่จริง — /v1/models มีคีย์ "id" หลายตัว (ของ permission ด้วย)
# regex แบบ greedy จะคว้าตัวสุดท้ายมา แล้วขอ completion ด้วยชื่อที่ server ไม่รู้จัก → 404
served_model() {{
  local body
  body="$(api_curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models")" || return 1
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
  echo "model:     {model_label}"
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
  echo "  container: {container_label} (หยุดและลบ · image {image_label} ไม่ถูกลบ)"
}}

{load_key}
start() {{
  load_api_key
{secret_guard}  local running
  running="$(docker ps --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  [[ -z "$running" ]] || die "container ${{CONTAINER_NAME}} กำลังรันอยู่ — รัน: $0 stop ก่อน"
  local leftover
  leftover="$(docker ps -a --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  if [[ -n "$leftover" ]]; then
    echo "เก็บซาก container จากรอบก่อน (${{CONTAINER_NAME}}) แล้วเริ่มใหม่"
    docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  fi
{run_notes}  docker run -d --name "${{CONTAINER_NAME}}" --restart {restart} \\
  --no-healthcheck \\
{runtime}{ipc}{shm}{network}{port_lines}{bind_lines}{extra_lines}{env_lines}{entry}  "${{IMAGE}}" {args}
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
  api_state
  echo "weights: "{weights_label}"  (lmds remove {slug} ลบด้วย — ดู: $0 remove-plan)"
}}

logs() {{ docker logs --tail "${{1:-300}}" "${{CONTAINER_NAME}}"; }}

test_text() {{
  local served
  served="$(served_model)" || die "เรียก /v1/models ไม่ได้ — server ขึ้นหรือยัง? ดู: $0 logs"
  api_curl -fsS "http://127.0.0.1:${{API_PORT}}/v1/chat/completions" \\
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
  ""|help|-h|--help) usage ;;
  # คำสั่งที่สคริปต์นี้ไม่มี (download · verify-files · prepare-runtime) ต้อง **ล้ม** — hub เรียก controller
  # ของ bundle บนเครื่องอื่นแล้วอ่าน exit code · พิมพ์วิธีใช้แล้วคืน 0 = hub รายงานว่า "สำเร็จ" ทั้งที่ไม่ได้ทำอะไร
  *)              usage >&2; die "ไม่มีคำสั่ง '$1' ใน bundle ที่ adopt มา (ดูรายการข้างบน)" ;;
esac
'''


def _bundle_directory(output: Path | None, slug: str) -> Path:
    """โฟลเดอร์ของ bundle เป็น path **เต็ม** เสมอ

    audit 2026-10-06: ค่าตั้งต้นของ CLI คือ `--output ./bundles` และ path ของ controller ถูกจดลง server.meta
    ตามนั้น (`bundles/coder-next/coder-next-adopted.sh`) — ใช้ได้เฉพาะจากโฟลเดอร์ที่พิมพ์คำสั่ง · จาก cwd อื่น
    (lmds-web ใต้ systemd · `lmds node run` จาก hub) controller "ไม่มี" และพอ container หยุด ทะเบียนก็ถูกลบ

    absolute() ไม่ใช่ resolve(): ไม่ตาม symlink — path ที่คืนต้องเป็นตัวเดียวกับที่ผู้ใช้เห็น
    """
    return Path(os.path.abspath(Path(output or "./bundles").expanduser())) / slug


def _write_controller(controller: Path, text: str, known_secrets: list[str]) -> str:
    """เขียน controller · ของเดิมที่ **ต่าง** จากตัวใหม่ถูกเก็บเป็น `<ชื่อ>.replaced-<เวลา>` — คืน path ที่เก็บ ("" = ไม่มี)

    `bundles refresh`/`node install` ข้าม bundle ที่ adopt มา (ไม่มี template) ทางเดียวที่สคริปต์นี้ถูกสร้างใหม่คือ
    มีคนสั่ง `lmds adopt` ซ้ำ · generator ของ adopt เปลี่ยนไปเรื่อย ๆ และสคริปต์เดิมอาจถูกแก้มือไว้ —
    เขียนทับเงียบ ๆ คือทิ้งหลักฐานเดียวที่บอกว่าของเดิมรันด้วยอะไร

    สำเนาที่เก็บ: ถอดค่าความลับที่รู้ออก (สคริปต์รุ่นก่อนเขียน key จาก argv ไว้ตัวเต็ม · zip ของ bundle กวาดทุกไฟล์
    ในโฟลเดอร์) และเป็น 0600 เผื่อมีค่าที่เราไม่รู้จัก
    """
    replaced = ""
    try:
        old = controller.read_text(encoding="utf-8", errors="replace") if controller.is_file() else None
    except OSError:
        old = None
    if old is not None and old != text:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = controller.with_name(f"{controller.name}.replaced-{stamp}")
        serial = 1
        while backup.exists():      # สองรอบในวินาทีเดียวต้องไม่ทับหลักฐานรอบแรก
            serial += 1
            backup = controller.with_name(f"{controller.name}.replaced-{stamp}-{serial}")
        backup.write_text(_scrub(old, known_secrets), encoding="utf-8")
        backup.chmod(0o600)
        replaced = str(backup)
    controller.write_text(text, encoding="utf-8")
    controller.chmod(0o755)
    return replaced


def _remember_api_key(slug: str, values: list[str]) -> tuple[str, str]:
    """เก็บ API key ที่ของเดิมใช้อยู่ลงที่เก็บของเครื่อง — คืน (ข้อความแจ้ง, ข้อที่ทำซ้ำไม่ได้)

    ค่าถูกถอดออกจากสคริปต์ (0755) แล้ว ถ้าไม่เก็บไว้ที่ไหนเลย restart ครั้งแรกหลัง adopt จะไม่มี key:
    ทาง env = container ขึ้นมาแบบเปิดโล่งเงียบ ๆ · ทาง argv = start ไม่ขึ้น · ที่เก็บของเครื่อง
    (~/.lmds/keys/<slug> 0600 · นอก bundle · ไม่ติดไปกับ zip) คือที่ของ key ตัวนี้อยู่แล้ว และเป็นเครื่องเดียวกับ
    ที่ key ตัวนี้อยู่บน argv/env ของ container · ไม่เคยเขียนทับ key ที่มีคนตั้งไว้
    """
    from . import apikey

    distinct = list(dict.fromkeys(v.strip() for v in values if v and v.strip()))
    if not distinct:
        return "", ""
    stored = apikey.read(slug)
    if stored:
        if stored in distinct:
            return "", ""
        return "", ("ที่เก็บ key ของเครื่องมี key อยู่แล้วและไม่ตรงกับที่ container ใช้อยู่ — ไม่ได้เขียนทับ · "
                    f"start ครั้งหน้าจะใช้ key ในที่เก็บ (ดู: lmds key show {slug})")
    if len(distinct) > 1:
        return "", ("container ใช้ API key มากกว่าหนึ่งค่า — ที่เก็บของเครื่องเก็บได้ค่าเดียว จึงไม่ได้เก็บให้ · "
                    f'ตั้งเองก่อน restart: echo -n "$KEY" | lmds key set {slug} (หรือ export ตัวแปรในเชลล์)')
    try:
        path = apikey.write(slug, distinct[0])
    except (apikey.ApiKeyError, OSError) as exc:
        return "", (f"เก็บ API key ที่ container ใช้อยู่ลงที่เก็บของเครื่องไม่ได้ ({_one_line(exc, 120)}) — "
                    f'ตั้งเองก่อน restart: echo -n "$KEY" | lmds key set {slug}')
    return f"เก็บ API key ที่ container ใช้อยู่ไว้ที่ {path} (0600 · นอก bundle) — restart แล้วยังมี auth เหมือนเดิม", ""


@dataclass
class AdoptReport:
    """ผลของ adopt หนึ่งครั้ง — สิ่งที่หน้าจอ/หน้าเว็บต้องบอกคนสั่ง ไม่ใช่แค่ path ของไฟล์"""

    controller: Path
    slug: str
    adopted: Adopted
    # ต่างจาก container เดิมตรงไหน (ชุดเดียวกับที่หัว controller และ MODEL_PROFILE.yaml)
    not_reproduced: list[str] = field(default_factory=list)
    # controller เดิมที่ต่างจากตัวใหม่ ถูกเก็บไว้ที่ไหน ("" = ไม่มีของเดิม หรือเหมือนกันทุกตัวอักษร)
    replaced: str = ""
    # เรื่องที่ทำให้แล้วและควรรู้ (เก็บ key ลงที่เก็บของเครื่อง)
    notes: list[str] = field(default_factory=list)


def adopt(container: str, slug: str = "", output: Path | None = None) -> Path:
    """สร้าง bundle จาก container ที่รันอยู่ แล้วลงทะเบียนกับ fleet — คืน path ของ controller"""
    return adopt_with_report(container, slug=slug, output=output).controller


def adopt_with_report(container: str, slug: str = "", output: Path | None = None) -> AdoptReport:
    """adopt() ที่คืนสิ่งที่ต้องบอกคนสั่งด้วย — CLI และหน้าเว็บใช้ตัวนี้"""
    if slug:
        _check_slug(slug)  # ก่อนแตะ docker — slug ผิดรูปไม่ควรได้ไปถึง inspect
    adopted = inspect_container(container)
    slug = slug or _derive_slug(adopted.container.replace("_", "-"))
    directory = _bundle_directory(output, slug)
    directory.mkdir(parents=True, exist_ok=True)

    env_secret_values = [item.split("=", 1)[1] for item in meaningful_env(adopted)
                         if "=" in item and redact_secrets([item])[1]]
    api_key_values = [item.split("=", 1)[1] for item in meaningful_env(adopted)
                      if "=" in item and redact_secrets([item])[1] and _API_KEY_ENV.search(item.split("=", 1)[0])]
    arg_secrets = render_args(list(adopted.args))[1]
    api_key_values += [s.value for s in arg_secrets if s.is_api_key]
    key_note, key_problem = _remember_api_key(slug, api_key_values)

    extra = [key_problem] if key_problem else []
    differences = not_reproduced(adopted) + extra
    controller = directory / f"{slug}-adopted.sh"
    replaced = _write_controller(controller, render_controller(adopted, slug, notes=extra),
                                 env_secret_values + [s.value for s in arg_secrets])

    profile = {
        "profile_version": 1,
        "generated_by": "lmds adopt",
        "adopted": True,
        "model": {"id": adopted.model or adopted.container, "artifact_type": "unknown"},
        "runtime": {"engine": adopted.engine,
                    "image": adopted.image},
        # port = พอร์ตฝั่ง **เครื่อง** (ตัวที่ health/gateway ใช้) — ไม่ใช่พอร์ตข้างใน container
        "serving": {"context": adopted.context, "port": adopted.port},
        "source_container": adopted.container,
    }
    if adopted.container_port and adopted.container_port != adopted.port:
        profile["serving"]["container_port"] = adopted.container_port
    if differences:
        # ต่างจาก container เดิมตรงไหน — ชุดเดียวกับที่ `lmds adopt` พิมพ์และที่หัว controller
        profile["not_reproduced"] = differences
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
    model_name = _meta_value(adopted.model or adopted.container)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={model_name}\n"
        f"model_id={model_name}\n"
        f"engine={profile['runtime']['engine']}\n"
        f"mode=docker\n"
        f"port={adopted.port}\n"
        f"container={adopted.container}\n"
        f"pid_file=\n"
        f"controller={controller}\n"
        f"started_at=\n",
        encoding="utf-8")
    return AdoptReport(controller=controller, slug=slug, adopted=adopted, not_reproduced=differences,
                       replaced=replaced, notes=[key_note] if key_note else [])


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
    # ค่าความลับบน argv (`--api-key …` · `--hf-token …`) ไม่ลงสคริปต์ — กลไกเดียวกับทาง container (render_args) ·
    # audit 2026-10-06: /proc/<pid>/cmdline ถูกคัดมาทั้งดุ้น ทั้งที่ environ ถูกเว้นไว้ด้วยเหตุผลนี้พอดี
    arg_words, arg_secrets = render_args(rest)
    argv = " \\\n    ".join(arg_words)
    key_names = [s.var for s in arg_secrets if s.is_api_key]
    # ไม่มี key บน argv = สคริปต์เหมือนเดิมทุกบรรทัด (ไม่มี load_api_key ให้ doctor เข้าใจผิดว่า `lmds key` คุมอยู่)
    load_key = (_load_key_block(slug, key_names) if key_names else "") + _ask_block(slug, key_names) + "\n"
    secret_guard = ("  load_api_key\n" if key_names else "") + _secret_guard(slug, arg_secrets)
    secret_note = (
        "# ค่าของ " + _one_line(" ".join(dict.fromkeys(s.label for s in arg_secrets)))
        + " ไม่ได้เก็บไว้ในไฟล์นี้ (0755) — start อ่านจากตัวแปร "
        + " ".join(s.var for s in arg_secrets) + " (API key เติมจาก ~/.lmds/keys/<slug> ให้)\n"
        if arg_secrets else ""
    )
    # ค่าพวกนี้มาจาก argv ของ process ที่ใครก็ตามบนเครื่องสั่งรันได้ และถูกวางในเครื่องหมายคำพูดคู่ —
    # `$(…)` ใน `-c`/`--host`/ชื่อโมเดล = คำสั่งที่ถูกรันทุกครั้งที่เรียกสคริปต์ (audit 2026-10-06)
    default_ctx = managed.get("ctx", "") if re.fullmatch(r"-?\d+", managed.get("ctx", "")) else ""
    default_host = managed.get("host", "0.0.0.0")
    if not re.fullmatch(r"[0-9A-Za-z.:_\[\]-]+", default_host):
        default_host = "0.0.0.0"
    server_bin_line = _default_assign("SERVER_BIN", proc.exe or (proc.argv[0] if proc.argv else ""))
    work_dir_line = _default_assign("WORK_DIR", proc.cwd or str(Path.home()))
    model_label = _dq(proc.model or "")
    model_or_slug = _dq(proc.model or slug)
    model_path_label = _dq(proc.model_path or "")
    unit_label = _dq(proc.unit)
    # heredoc ของ server.meta ไม่ได้ quote: `$` และ backtick ยังถูกแปล และขึ้นบรรทัดใหม่ = แทรกคีย์อื่นได้
    meta_model = re.sub(r"([\\$`])", r"\\\1", _meta_value(proc.model or slug))
    meta_model_id = re.sub(r"([\\$`])", r"\\\1", _meta_value(proc.model_path or slug))
    unit_note = (
        f"# unit เดิมที่เป็นเจ้าของ process นี้: {_one_line(proc.unit)}\n"
        f"# ถ้ามันยัง enable อยู่ มันจะแย่ง port กลับทุกครั้งที่ LMDS stop\n"
        if proc.unit else ""
    )
    weights_label = shlex.quote(proc.model_path or "(ไม่ระบุใน argv)")

    return f"""#!/usr/bin/env bash
# LMDS adopted controller (native) — สร้างจาก process ที่รันอยู่ก่อนหน้า ไม่ได้ deploy ผ่าน LMDS
#
# argv ข้างล่างคัดมาจาก /proc/{proc.pid}/cmdline ตอนรับเข้าระบบ ไม่ได้เดา
# ไม่มี download/verify-files: weight เป็น path ที่คุณจัดการเอง LMDS จึงไม่มีอะไรให้โหลดหรือตรวจ
{unit_note}{secret_note}set -Eeuo pipefail

SCRIPT_VERSION="${{SCRIPT_VERSION:-1.0.0}}"
ADOPTED=1
SLUG="{slug}"
API_PORT="${{API_PORT:-{proc.port or 8000}}}"
CTX_SIZE="${{CTX_SIZE:-{default_ctx}}}"
API_HOST="${{API_HOST:-{default_host}}}"
{server_bin_line}
{work_dir_line}
RUN_DIR="${{RUN_DIR:-${{HOME}}/.lmds/run/{slug}}}"
PID_FILE="${{RUN_DIR}}/server.pid"
LOG_FILE="${{RUN_DIR}}/server.log"
OWNING_UNIT="{unit_label}"

die() {{ echo "ERROR: $*" >&2; exit 1; }}

server_alive() {{ [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }}

served_model() {{
  local body
  body="$(api_curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models")" || return 1
  printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}}

banner() {{
  echo "LMDS adopted (native) · {slug} · v${{SCRIPT_VERSION}}"
  echo "binary: ${{SERVER_BIN}}"
}}

info() {{
  banner
  echo "model:     {model_label or '(ไม่ระบุใน argv)'}"
  echo "weights:   {model_path_label or '(ไม่ระบุ)'}"
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
model={meta_model}
default_model={meta_model}
model_id={meta_model_id}
engine={proc.engine}
mode=native
port=${{API_PORT}}
container=
pid_file=${{PID_FILE}}
controller=${{script_path}}
started_at=$(date +%Y-%m-%dT%H:%M:%S)
META
}}

{load_key}start() {{
  server_alive && die "{slug} รันอยู่แล้ว (PID $(cat "$PID_FILE"))"
{secret_guard}
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
  echo "model:     {model_or_slug}"
  echo "weights:   "{weights_label}"  (lmds remove {slug} ลบด้วย — ดู: $0 remove-plan)"
  if server_alive; then echo "process: running (PID $(cat "$PID_FILE"))"; else echo "process: stopped"; fi
  api_state
  if [[ -n "$OWNING_UNIT" ]] && systemctl is-active --quiet "$OWNING_UNIT" 2>/dev/null; then
    echo "หมายเหตุ: ${{OWNING_UNIT}} ยังรันอยู่ — ตัวที่ตอบอาจเป็นของ unit นั้น ไม่ใช่ของ LMDS"
  fi
  true
}}

logs() {{ tail -n "${{1:-300}}" "$LOG_FILE" 2>/dev/null || echo "ยังไม่มี log (start ผ่าน LMDS ก่อน)"; }}

test_text() {{
  local served
  served="$(served_model)" || die "เรียก /v1/models ไม่ได้ — server ขึ้นหรือยัง? ดู: $0 logs"
  api_curl -fsS "http://127.0.0.1:${{API_PORT}}/v1/chat/completions" \\
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
  echo "Model:     {model_or_slug}"
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
  ""|help|-h|--help) usage ;;
  # คำสั่งที่สคริปต์นี้ไม่มีต้อง **ล้ม** ไม่ใช่พิมพ์วิธีใช้แล้วคืน 0 — hub อ่าน exit code (ดู controller ของ container)
  *)              usage >&2; die "ไม่มีคำสั่ง '$1' ใน bundle ที่ adopt มา (ดูรายการข้างบน)" ;;
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
    directory = _bundle_directory(output, slug)
    directory.mkdir(parents=True, exist_ok=True)

    # ความลับบน argv: ค่าไม่ลงสคริปต์/profile · API key ของเซิร์ฟเวอร์ย้ายไปที่เก็บของเครื่อง (ดู _remember_api_key)
    arg_secrets = render_args(split_managed(proc.argv[1:])[0])[1]
    key_note, key_problem = _remember_api_key(slug, [s.value for s in arg_secrets if s.is_api_key])
    for secret in arg_secrets:
        proc.notes.append(_one_line(
            f"ค่าของ {secret.label} บน argv ไม่ได้เขียนลงสคริปต์ (ไฟล์ 0755) — start อ่านจาก ${secret.var}"
            + (" · ไม่มีค่า = ไม่ start" if secret.required else " · ไม่มีค่า = start โดยไม่ใส่ธงนี้")))
    proc.notes += [note for note in (key_note, key_problem) if note]

    controller = directory / f"{slug}-adopted.sh"
    replaced = _write_controller(controller, render_native_controller(proc, slug),
                                 [s.value for s in arg_secrets])
    if replaced:
        proc.notes.append(f"controller เดิมต่างจากตัวที่สร้างใหม่ — เก็บของเดิมไว้ที่ {replaced}")

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
        # argv เก็บไว้ทั้งชุดเพื่อตรวจย้อนได้ว่า bundle นี้มาจากคำสั่งอะไร — ยกเว้นค่าความลับ (ไฟล์นี้ติดไปกับ zip)
        "source_process": {"pid": proc.pid, "unit": proc.unit,
                           "argv": mask_args(proc.argv, [s.value for s in arg_secrets])},
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
        f"model={_meta_value(proc.model or slug)}\n"
        f"default_model={_meta_value(proc.model or slug)}\n"
        f"model_id={_meta_value(proc.model_path or slug)}\n"
        f"engine={proc.engine}\n"
        f"mode=native\n"
        f"port={proc.port}\n"
        f"container=\n"
        f"pid_file={run_dir / 'server.pid'}\n"
        f"controller={controller}\n"
        f"started_at=\n",
        encoding="utf-8")
    return controller, proc
