"""Flag allowlist ต่อ engine — flag นอกรายการนี้ต้องให้ผู้ใช้อนุมัติรายตัวเสมอ

หมายเหตุ: `--trust-remote-code` จงใจไม่อยู่ใน allowlist — ต้องผ่านการอนุมัติทุกครั้ง (PRD §9.2)
"""

from __future__ import annotations

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

# registry/repo ของ runtime image ที่ยอมรับ (เทียบส่วนก่อน :tag/@digest)
# LLM เสนอ image นอกรายการนี้ไม่ได้ — เคยเกิดจริง: มโน ghcr.io/lmds/llamacpp-ubuntu-rtx จน start พัง
KNOWN_IMAGE_REPOS: dict[Engine, set[str]] = {
    Engine.VLLM: {"vllm/vllm-openai", "nvcr.io/nvidia/vllm", "docker.io/vllm/vllm-openai"},
    Engine.LLAMACPP: {"ghcr.io/ggml-org/llama.cpp", "ghcr.io/ggerganov/llama.cpp"},
}


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
