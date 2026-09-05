"""ความรู้ต่อตระกูลโมเดลที่ *ต้องเป็นค่า* ไม่ใช่แค่ข้อความเตือน

เดิม `arch_notes()` รู้ว่า Qwen3 ต้องใช้ `qwen3_xml` + `qwen3` และ Gemma 4 ต้องใช้ `gemma4`
แต่พิมพ์ออกมาเป็นคำเตือนแล้วปล่อย `plan.tool_calling.parser = None` · ผู้ใช้จึงต้องอ่าน
คำเตือน แล้วไปพิมพ์ชื่อเองในช่อง settings — ซึ่งคนที่ไม่รู้ก็ไม่กล้าพิมพ์ (ผู้ใช้ 2026-09-04:
"กลัวใส่ผิด แล้วไม่มีให้ใช้งาน") · ส่วนคนที่เดาไปที่ hermes ก็ได้ tool call เป็นข้อความ
(msi-2 2026-09-02)

ไฟล์นี้เป็นที่เดียวที่เก็บความรู้นั้นในรูปค่า — planner ใช้ตอนสร้าง bundle ใหม่ และ
`fleet.suggest` ใช้เติมให้ bundle ที่มีอยู่แล้ว · ชื่อทุกตัวต้องอยู่ในรายการที่
`orchestrator._harden_parsers` ยอมรับ ไม่งั้นถูกตัดทิ้งตอนวางแผน (มีเทสคุม)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParserChoice:
    tool: str | None = None
    reasoning: str | None = None
    why: str = ""


def parsers_for(model_id: str, architecture: str = "", engine: str = "vllm") -> ParserChoice:
    """parser ที่ตระกูลนี้ต้องใช้บน engine นี้ — ว่าง = ไม่รู้ อย่าเดา

    engine สำคัญ: vLLM กับ SGLang ใช้คนละชุดชื่อ (Qwen3 บน vLLM = qwen3_xml · บน SGLang = qwen)
    llama.cpp ไม่มีแฟล็กพวกนี้เลย (ใช้ --jinja อ่าน template เอง)
    """
    if engine == "llamacpp":
        return ParserChoice()
    key = f"{model_id} {architecture}".lower()
    sglang = engine == "sglang"

    if "gemma-4" in key or "gemma4" in key:
        return ParserChoice(
            tool="gemma4",
            why="Gemma 4 พ่น <|tool_call>call:name{...} ซึ่งมีแต่ parser gemma4 อ่านออก (hermes ไม่ออก)",
        )
    if "qwen3" in key or "qwen-3" in key or "qwen3_5" in key:
        if "coder" in key:
            return ParserChoice(
                tool="qwen3_coder",
                why="Qwen3-Coder ใช้ qwen3_coder · เป็นโมเดลไม่คิด ไม่ต้องมี reasoning parser",
            )
        return ParserChoice(
            tool="qwen" if sglang else "qwen3_xml",
            reasoning="qwen3",
            why=("Qwen3/3.5/3.6: tool = " + ("qwen (SGLang)" if sglang else "qwen3_xml")
                 + " · reasoning = qwen3 ไม่งั้น thinking ปนมาในคำตอบ"),
        )
    return ParserChoice()


# ค่าที่พิสูจน์แล้วบน spark-head 2026-09-03 (ucbye/Qwen3-Coder-Next-NVFP4-GB10 · 61 tok/s เดี่ยว
# / 103 tok/s รวม 3 สาย) — อ่านจาก bundle.env ของเครื่องจริง ไม่ใช่จากข้อความเตือน
NVFP4_SM121_IMAGE = "avarok/dgx-vllm-nvfp4-kernel:latest"
NVFP4_SM121_ENGINE_ENV = (
    "VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_TEST_FORCE_FP8_MARLIN=1 "
    "VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_MARLIN_USE_ATOMIC_ADD=1"
)


# NVIDIA ตั้งชื่อ checkpoint NVFP4 ทางการว่า `…-FP4` (nvidia/Llama-3.3-70B-Instruct-FP4 · nvidia/DeepSeek-R1-FP4)
# ไม่ใช่ `…-NVFP4` และ config.json ของ ModelOpt บอกแค่ `quant_method: modelopt` (quant_algo อยู่ใน
# hf_quant_config.json) — จับเป็น token ของชื่อ (คั่นด้วย - _ .) ไม่จับกลางคำ · MXFP4 (gpt-oss) ไม่ใช่ NVFP4
# kernel คนละชุด อย่าเหมารวม
_FP4_TOKEN_RE = re.compile(r"(?:^|[-_.])(?:nv)?fp4(?:[-_.]|$)", re.IGNORECASE)


def looks_nvfp4(repo_id: str, quantization: str | None = "") -> bool:
    """checkpoint NVFP4 — ดูทั้งชื่อ repo (`nvfp4` / token `fp4`) และ quantization ที่อ่านจาก config

    DeepSeek-V4-Flash-NVFP4 รายงาน quantization="fp8" (hf_quant_config ของ ModelOpt บอก kv/attn เป็น fp8)
    ทั้งที่ weight เป็น NVFP4 — ชื่อ repo จึงเป็นหลักฐานที่ต้องดูด้วย
    """
    name = (repo_id or "").split("/")[-1]
    quant = (quantization or "").lower()
    if "mxfp4" in quant and "nvfp4" not in quant and "nvfp4" not in name.lower():
        return False
    if "nvfp4" in name.lower() or "nvfp4" in quant:
        return True
    if quant in ("fp4", "nvfp4") or quant.startswith("nvfp4"):
        return True
    return _FP4_TOKEN_RE.search(name) is not None


@dataclass(frozen=True)
class RuntimeHint:
    image: str | None = None
    engine_env: str | None = None
    why: str = ""
    extra: dict = field(default_factory=dict)


def nvfp4_on_sm121(model_id: str, quantization: str = "", engine: str = "vllm",
                   memory_model: str = "") -> RuntimeHint:
    """NVFP4 บน GB10 (unified/SM121) + vLLM: image ต้องมี FP4 kernel ของ sm_121 และต้องบังคับ Marlin

    ไม่ครบ = ล้มตั้งแต่ start ไม่ใช่แค่ช้า (msi-6 2026-08-20: ptxas ปฏิเสธ cvt .e2m1x2 ตอน JIT
    cutlass_fused_moe แล้ว engine core ตายก่อน health)
    """
    if engine != "vllm" or memory_model != "unified":
        return RuntimeHint()
    if not looks_nvfp4(model_id, quantization):
        return RuntimeHint()
    return RuntimeHint(
        image=NVFP4_SM121_IMAGE,
        engine_env=NVFP4_SM121_ENGINE_ENV,
        why="NVFP4 บน SM121 (GB10): image ที่มี FP4 kernel + บังคับ Marlin — พิสูจน์บน spark-head 2026-09-03",
    )
