"""Flag allowlist ต่อ engine — flag นอกรายการนี้ต้องให้ผู้ใช้อนุมัติรายตัวเสมอ

หมายเหตุ: `--trust-remote-code` จงใจไม่อยู่ใน allowlist — ต้องผ่านการอนุมัติทุกครั้ง (PRD §9.2)
"""

from __future__ import annotations

import re

from .plan_schema import Engine

VLLM_FLAGS = {
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "--reasoning-parser",
    "--chat-template",
    "--kv-cache-dtype",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--enforce-eager",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--tensor-parallel-size",
    "--quantization",
    "--dtype",
    "--swap-space",
    "--tokenizer-mode",
    "--limit-mm-per-prompt",
    # flag ที่โมเดลบางตระกูล "ต้อง" ใช้ ไม่ใช่การปรับจูน — เข้ามาทาง recipes/catalog.yaml
    # ซึ่งบังคับ source + validated_on อยู่แล้ว · เคสจริง: DeepSeek V4 ต้องใช้ PIECEWISE cudagraph
    # ไม่งั้น kernel init ของ DeepGEMM เรียกคนละ signature แล้วตายตอนโหลดโมเดล
    "--compilation-config",
    "--max-cudagraph-capture-size",
    "--moe-backend",
    "--block-size",
    "--generation-config",
    "--async-scheduling",
}

LLAMACPP_FLAGS = {
    "--flash-attn",
    "--n-gpu-layers",
    "--parallel",
    "--cont-batching",
    "--mmproj",
    "--jinja",
    "--cache-type-k",
    "--cache-type-v",
    "--threads",
    "--split-mode",
    "--batch-size",
    "--ubatch-size",
    "--rope-scaling",
}

_BY_ENGINE = {Engine.VLLM: VLLM_FLAGS, Engine.LLAMACPP: LLAMACPP_FLAGS}

# ── environment variable ──────────────────────────────────────────────────────
# Prefix ไม่ใช่ security boundary: NCCL_ENV_PLUGIN/NCCL_NET_PLUGIN รับ path ของ .so
# และ vLLM มี VLLM_ALLOW_INSECURE_SERIALIZATION ที่เปิด pickle โดยตรง · จึงรับเฉพาะ
# ชื่อที่ review แล้ว แยกตาม engine และตรวจทรงของค่าอีกชั้น
_COMMON_SAFE_ENV = {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}

_VLLM_SAFE_ENV = _COMMON_SAFE_ENV | {
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_NVFP4_GEMM_BACKEND",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
    "VLLM_USE_FLASHINFER_MOE_FP4",
    "VLLM_MARLIN_USE_ATOMIC_ADD",
    "TORCH_CUDA_ARCH_LIST",
    "NCCL_IB_HCA",
}

_LLAMACPP_SAFE_ENV = _COMMON_SAFE_ENV | {"GGML_CUDA_FORCE_MMQ"}

_ENV_BY_ENGINE = {
    Engine.VLLM: _VLLM_SAFE_ENV,
    Engine.LLAMACPP: _LLAMACPP_SAFE_ENV,
}

_BOOL_ENV = {
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_USE_FLASHINFER_MOE_FP4",
    "VLLM_MARLIN_USE_ATOMIC_ADD",
    "GGML_CUDA_FORCE_MMQ",
}

_NVFP4_GEMM_BACKENDS = {
    "flashinfer-cudnn",
    "flashinfer-trtllm",
    "flashinfer-cutlass",
    "cutlass",
    "marlin",
    "emulation",
}

_CUDA_DEVICE_LIST = re.compile(
    r"(?:MIG-GPU-[A-Fa-f0-9-]+/\d+/\d+|"
    r"(?:-?\d+|GPU-[A-Fa-f0-9-]+)(?:,(?:-?\d+|GPU-[A-Fa-f0-9-]+))*)"
)
_TORCH_ARCH_LIST = re.compile(r"\d+(?:\.\d+)?[a-z]?(?:\+PTX)?(?:[ ;]+\d+(?:\.\d+)?[a-z]?(?:\+PTX)?)*")
_SAFE_HCA_LIST = re.compile(r"^\^?=?[A-Za-z0-9_.+-]+(?::\d+)?(?:,[A-Za-z0-9_.+-]+(?::\d+)?)*$")


def is_allowed_env(engine: Engine, name: str) -> bool:
    return name in _ENV_BY_ENGINE[engine]


def _is_allowed_env_value(name: str, value: str) -> bool:
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        return False
    if name in _BOOL_ENV:
        return value in {"0", "1"}
    if name == "OMP_NUM_THREADS":
        return value.isdigit() and int(value) > 0
    if name == "VLLM_FLASHINFER_ALLREDUCE_BACKEND":
        return value in {"auto", "trtllm", "mnnvl"}
    if name == "VLLM_NVFP4_GEMM_BACKEND":
        return value in _NVFP4_GEMM_BACKENDS
    if name == "CUDA_VISIBLE_DEVICES":
        return bool(_CUDA_DEVICE_LIST.fullmatch(value))
    if name == "TORCH_CUDA_ARCH_LIST":
        return bool(_TORCH_ARCH_LIST.fullmatch(value))
    if name == "NCCL_IB_HCA":
        return bool(_SAFE_HCA_LIST.fullmatch(value))
    # เพิ่มชื่อเข้า allowlist แต่ลืมเพิ่ม validator ต้อง fail closed
    return False


def split_env(engine: Engine, env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """แยก env ที่ผ่าน exact per-engine allowlist และ value validator"""
    allowed: dict[str, str] = {}
    rejected: list[str] = []
    for name, value in (env or {}).items():
        text = str(value)
        if is_allowed_env(engine, name) and _is_allowed_env_value(name, text):
            allowed[name] = text
        else:
            rejected.append(name)
    return allowed, sorted(rejected)

# registry/repo ของ runtime image ที่ยอมรับ (เทียบส่วนก่อน :tag/@digest)
# LLM เสนอ image นอกรายการนี้ไม่ได้ — เคยเกิดจริง: มโน ghcr.io/lmds/llamacpp-ubuntu-rtx จน start พัง
# image ของชุมชนที่ "เราตรวจแล้ว" — เข้ามาได้ทางเดียวคือผ่าน recipes/catalog.yaml ซึ่งบังคับให้มี
# source + validated_on · โมเดลบางตัวรันได้เฉพาะ build เฉพาะทาง (DeepSeek V4 บน GB10) การกัน
# ทั้งหมดไว้เท่ากับกันโมเดลเหล่านั้นออกจากระบบ · มีเทสบังคับว่า image ในแคตตาล็อกต้องอยู่ในรายการนี้
VETTED_COMMUNITY_IMAGES = {
    "ghcr.io/anemll/dspark-vllm-gx10",   # DeepSeek V4 บน DGX Spark
    "avarok/dgx-vllm-nvfp4-kernel",      # NVFP4 kernel สำหรับ GB10
    "lmsysorg/sglang",                   # SGLang ทางการ
}

KNOWN_IMAGE_REPOS: dict[Engine, set[str]] = {
    Engine.VLLM: {"vllm/vllm-openai", "nvcr.io/nvidia/vllm", "docker.io/vllm/vllm-openai"}
    | VETTED_COMMUNITY_IMAGES,
    Engine.LLAMACPP: {"ghcr.io/ggml-org/llama.cpp", "ghcr.io/ggerganov/llama.cpp"},
}


# โฮสต์ที่ยอมให้ดึงไฟล์ runtime ภายนอก (parser plugin ฯลฯ) — ไฟล์พวกนี้เป็นโค้ดที่รันจริง
# ใน container จึงต้องจำกัดให้แคบ: แหล่งที่ทีมโมเดลเผยแพร่เองเท่านั้น
ALLOWED_ASSET_HOSTS = {
    "huggingface.co",
    "hf.co",
    "raw.githubusercontent.com",
    "github.com",
    "gitlab.com",
}


def is_allowed_asset_url(url: str) -> bool:
    """รับเฉพาะ https จากโฮสต์ใน allowlist — กัน LLM สั่งดึงไฟล์จากที่ไหนก็ได้"""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host in ALLOWED_ASSET_HOSTS


def is_safe_asset_filename(name: str) -> bool:
    """basename ล้วนเท่านั้น — กัน path traversal ตอนเขียนไฟล์ลง host"""
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


def image_repo(image_ref: str) -> str:
    """'ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:x' → 'ghcr.io/ggml-org/llama.cpp'"""
    without_digest = image_ref.split("@", 1)[0]
    head, _, tail = without_digest.rpartition(":")
    if head and "/" not in tail:  # ':' ตัวท้ายเป็น tag ไม่ใช่ port ของ registry
        return head
    return without_digest


def is_known_image(engine: Engine, image_ref: str) -> bool:
    return image_repo(image_ref) in KNOWN_IMAGE_REPOS[engine]


def flag_name(flag: str) -> str:
    """'--kv-cache-dtype=fp8' หรือ '--kv-cache-dtype fp8' → '--kv-cache-dtype'"""
    return flag.strip().split("=", 1)[0].split(None, 1)[0]


# llama.cpp รุ่นใหม่เปลี่ยน --flash-attn/-fa จาก boolean flag → ต้องมีค่า on|off|auto
# ถ้าถูกใส่มาแบบ bare (ไม่มีค่า) llama-server จะ 'กิน' flag ตัวถัดไปเป็นค่าแล้ว error
_FLASH_ATTN_ALIASES = {"--flash-attn", "-fa"}
_FLASH_ATTN_VALUES = {"on", "off", "auto"}


def normalize_llamacpp_flags(flags: list[str]) -> list[str]:
    """บังคับให้ --flash-attn/-fa มีค่าเสมอ (bare → 'on') กันไปกิน flag ถัดไป

    รองรับรูปแบบ: '--flash-attn', '-fa', '--flash-attn on', '--flash-attn=on'
    """
    out: list[str] = []
    for flag in flags:
        stripped = flag.strip()
        if not stripped:
            continue
        head = stripped.split("=", 1)[0].split(None, 1)[0]
        if head not in _FLASH_ATTN_ALIASES:
            out.append(stripped)
            continue
        # ดึงค่าที่อาจตามมา (ทั้งแบบ '=' และเว้นวรรค)
        if "=" in stripped:
            value = stripped.split("=", 1)[1].strip()
            trailing = ""
        else:
            parts = stripped.split(None, 1)
            value = parts[1].strip() if len(parts) == 2 else ""
            trailing = value
        if value in _FLASH_ATTN_VALUES:
            out.append(f"--flash-attn {value}")
        else:
            out.append("--flash-attn on")
            # กรณี LLM รวม flag อื่นมาใน item เดียว เช่น '--flash-attn --threads' → คืน token ที่เหลือ
            if trailing and trailing.startswith("-"):
                out.append(trailing)
    return out


def coalesce_flag_tokens(flags: list[str]) -> list[str]:
    """รวม '--flag' + ค่าที่ถูกแยกมาเป็นคนละ item → '--flag value'

    LLM บางครั้งให้ extra_flags แบบ ['--threads', '4'] แทน ['--threads 4'] ทำให้ '4'
    ถูกเข้าใจผิดเป็น flag แยก (ตกไป needs_approval) และ '--threads' เหลือค่าเปล่า
    เงื่อนไขรวม: item ปัจจุบันเป็น flag เดี่ยว (ขึ้นด้วย '-' ไม่มีค่าในตัว) และ item ถัดไป
    ไม่ใช่ flag (ไม่ขึ้นด้วย '-') → ถือเป็นค่าของ flag นั้น
    """
    out: list[str] = []
    i = 0
    n = len(flags)
    while i < n:
        cur = flags[i].strip()
        if not cur:
            i += 1
            continue
        if cur.startswith("-") and " " not in cur and "=" not in cur and i + 1 < n:
            nxt = flags[i + 1].strip()
            if nxt and not nxt.startswith("-"):
                out.append(f"{cur} {nxt}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def split_flags(engine: Engine, flags: list[str]) -> tuple[list[str], list[str]]:
    """คืน (allowed, needs_approval)"""
    allowlist = _BY_ENGINE[engine]
    allowed: list[str] = []
    needs_approval: list[str] = []
    for flag in flags:
        if not flag.strip():
            continue
        (allowed if flag_name(flag) in allowlist else needs_approval).append(flag.strip())
    return allowed, needs_approval
