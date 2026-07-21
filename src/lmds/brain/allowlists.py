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
