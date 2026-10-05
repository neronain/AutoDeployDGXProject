"""รูปแบบ weight ที่รันไทม์ของ LMDS โหลดไม่ได้ — ตรวจจาก metadata ก่อนดาวน์โหลดสักไบต์

นามสกุล `.safetensors` บอกแค่ *ภาชนะ* ไม่ได้บอกว่าของข้างในเป็นของไลบรารีไหน · MLX (Apple Silicon)
เก็บ weight ลง `.safetensors` เหมือนกัน แต่จัดเก็บ/quantize ด้วยรูปแบบของตัวเอง (affine · `bits` +
`group_size` · อัดลง uint32) ซึ่ง vLLM / SGLang ไม่มี loader ให้ และ llama.cpp อ่านได้เฉพาะ `.gguf`

เคสจริง 2026-10-05 (hub 965eb56): `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (Hub redirect ไป
`TensorFold/…`) — inspect ตอบ Artifact=safetensors · fit "(vllm) ✅ fits" · plan เลือก vllm ด้วยเหตุผล
"safetensors→vLLM" · ถ้าปล่อยไป `deploy` + `node push --download` จะโหลด 113 GB แล้วไปล้มตอน start
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MLX = "mlx"

# จับเป็น token คั่นด้วย - _ . หรือหัว/ท้ายชื่อ — ไม่จับกลางคำ
_MLX_NAME_RE = re.compile(r"(?:^|[-_.])mlx(?:[-_.]|$)", re.IGNORECASE)
# คีย์ที่ quantizer ฝั่ง transformers ใส่กำกับเสมอ (GPTQ/AWQ: quant_method · ModelOpt: quant_algo ·
# llm-compressor: format) — GPTQ กับ AWQ ก็มี bits + group_size เหมือน MLX ต่างกันที่มีคีย์พวกนี้
_TRANSFORMERS_QUANT_KEYS = ("quant_method", "quant_algo", "format")


def mlx_quantization(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """บล็อก quantization แบบ MLX ใน config.json — None = ไม่ใช่

    mlx-lm เขียนคีย์ `quantization` (ไม่มี `_config`) ซึ่ง transformers ไม่เคยใช้ แล้วสำเนาไว้ที่
    `quantization_config` ให้ Hub อ่านได้ · ของจริงจากเคสข้างบน: `{"group_size": 32, "bits": 4,
    "mode": "affine"}` ทั้งสองคีย์
    """
    for key in ("quantization", "quantization_config"):
        block = (config or {}).get(key)
        if not isinstance(block, dict):
            continue
        if not all(isinstance(block.get(k), int) for k in ("bits", "group_size")):
            continue
        if any(block.get(k) for k in _TRANSFORMERS_QUANT_KEYS):
            continue
        return block
    return None


def mlx_evidence(repo_id: str, library_name: str | None, tags: Iterable[str],
                 config: dict[str, Any] | None) -> tuple[list[str], bool]:
    """(หลักฐานที่เจอ, ชี้ขาดได้ไหมว่าเป็น MLX)

    สัญญาณเดียวไม่พอทั้งสองทิศ (สำรวจ Hub 2026-10-05):
    - `library_name == "mlx"` อย่างเดียว **พลาด** `lmstudio-community/*-MLX-4bit` ซึ่งแจ้ง library_name
      เป็น `transformers` ทั้งที่ weight เป็น MLX 4-bit (tag `mlx` + บล็อก quantization บอกอยู่)
    - tag `mlx` อย่างเดียวบอกได้แค่ว่า "ใช้กับ MLX ได้" — repo transformers ปกติก็ติดได้ ปฏิเสธจาก
      tag ล้วนคือขวางโมเดลที่รันได้จริงโดยไม่มีทางข้าม จึงต้องมีชื่อ repo หรือการไม่มี library อื่นยืนยัน
    ชี้ขาด = library_name เป็น mlx · หรือ config มีบล็อก quantization แบบ MLX · หรือ tag `mlx` ที่มี
    ชื่อ repo ยืนยัน/ไม่มี library อื่นแย้ง · ที่เหลือเป็นแค่ร่องรอย ผู้เรียกเอาไปเตือน ไม่ใช่ปฏิเสธ
    """
    library = (library_name or "").strip().lower()
    tagged = any(isinstance(t, str) and t.lower() == MLX for t in tags)
    named = _MLX_NAME_RE.search(repo_id.split("/")[-1]) is not None
    quant = mlx_quantization(config)

    evidence: list[str] = []
    if library == MLX:
        evidence.append("library_name=mlx")
    if tagged:
        evidence.append("tag mlx")
    if quant is not None:
        evidence.append(f"config.json quantization bits={quant['bits']} group_size={quant['group_size']}")
    if named:
        evidence.append("ชื่อ repo มีคำว่า MLX")
    decisive = library == MLX or quant is not None or (tagged and (named or not library))
    return evidence, decisive


def base_model_of(tags: Iterable[str]) -> str:
    """repo ต้นทางจาก tag `base_model:<org>/<name>` / `base_model:quantized:<org>/<name>` — "" = ไม่ระบุ"""
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("base_model:"):
            repo = tag.rsplit(":", 1)[-1]
            if "/" in repo:
                return repo
    return ""


def unsupported_reason(report) -> str:
    """ทำไม deploy repo นี้ไม่ได้ — "" = รองรับ · ทางออกอยู่ที่ `unsupported_alternatives`"""
    if getattr(report, "unsupported_format", None) != MLX:
        return ""
    evidence = " · ".join(report.unsupported_evidence) or "metadata ของ repo"
    size = f" {report.weight_bytes / 1e9:.1f} GB" if report.weight_bytes else ""
    return (
        f"{report.repo_id} เป็น checkpoint รูปแบบ MLX (Apple Silicon) — LMDS deploy ไม่ได้ · "
        f"หลักฐาน: {evidence} · ไฟล์ลงท้าย .safetensors ก็จริง แต่ weight ข้างในจัดเก็บ/quantize แบบ MLX "
        "ซึ่ง vLLM และ SGLang โหลดไม่ได้ (llama.cpp อ่านได้เฉพาะ .gguf) — ถ้าปล่อยผ่านจะดาวน์โหลด"
        f"{size} แล้วไปล้มตอน start"
    )


def unsupported_alternatives(report) -> list[str]:
    """ของที่ใช้แทนได้ — รุ่นอื่นของ *โมเดลเดียวกัน* ที่ engine ของเราโหลดได้"""
    if getattr(report, "unsupported_format", None) != MLX:
        return []
    base = base_model_of(report.tags)
    out = [
        "ใช้รุ่นอื่นของโมเดลเดียวกันแทน: GGUF (→ llama.cpp) · NVFP4 (→ vLLM บน DGX Spark) · "
        "หรือ safetensors ต้นฉบับ" + (f" {base}" if base else " (ดู base model ใน model card)")
    ]
    if base:
        out.append(f"รุ่นที่ quantize จากต้นฉบับเดียวกัน: https://huggingface.co/models?other=base_model:quantized:{base}")
    return out
