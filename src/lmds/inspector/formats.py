"""รูปแบบ weight ที่รันไทม์ของ LMDS โหลดไม่ได้ — ตรวจจาก metadata ก่อนดาวน์โหลดสักไบต์

นามสกุล `.safetensors` บอกแค่ *ภาชนะ* ไม่ได้บอกว่าของข้างในเป็นของไลบรารีไหน · MLX (Apple Silicon)
เก็บ weight ลง `.safetensors` เหมือนกัน แต่จัดเก็บ/quantize ด้วยรูปแบบของตัวเอง (affine · `bits` +
`group_size` · อัดลง uint32) ซึ่ง vLLM / SGLang ไม่มี loader ให้ และ llama.cpp อ่านได้เฉพาะ `.gguf`

เคสจริง 2026-10-05 (hub 965eb56): `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (Hub redirect ไป
`TensorFold/…`) — inspect ตอบ Artifact=safetensors · fit "(vllm) ✅ fits" · plan เลือก vllm ด้วยเหตุผล
"safetensors→vLLM" · ถ้าปล่อยไป `deploy` + `node push --download` จะโหลด 113 GB แล้วไปล้มตอน start

EXL2 / EXL3 (ExLlamaV2/V3) คือบั๊กตัวเดียวกันอีกรอบ (audit 2026-10-06): quantizer ของ ExLlama เขียน weight เป็น
trellis/codebook (`….suh / .svh / .mul1 / .trellis`) ลง `.safetensors` — โหลดได้เฉพาะ ExLlama/TabbyAPI ·
`lmds generate doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw` ได้ bundle vLLM ผ่านทุก gate (108 GB)

ช่อง `ModelReport.unsupported_format` พูดถึง **ฝั่งที่แผนจะใช้** เท่านั้น: repo ที่มี GGUF อยู่ด้วยไม่ถูกปฏิเสธ
เพราะ safetensors ข้าง ๆ เป็น MLX/EXL — inspector จัดเป็น repo GGUF แล้วจดเหตุผลไว้ใน `format_note`
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MLX = "mlx"
EXL2 = "exl2"
EXL3 = "exl3"
# repo ที่ไม่มีอะไรให้ engine ของเราเสิร์ฟ — รู้ได้จากโครงสร้างของ repo ก่อนโหลดสักไบต์ (audit 2026-10-06:
# `lmds generate` เคยออก exit 0 + "Bundle (static-validated ✅)" ให้ทุกตัวในกลุ่มนี้)
DIFFUSERS = "diffusers"                    # pipeline ของ diffusers (model_index.json) — stabilityai/sdxl-turbo
ADAPTER = "adapter"                        # LoRA/PEFT ล้วน ไม่มี weight ของ base — IFM/K2-Horizon-7B-Uno
NO_WEIGHTS = "no-weights"                  # ไม่มี .safetensors/.gguf ของตัวโมเดลเลย — onnx-community/Qwen3-0.6B-ONNX
NO_ROOT_CHECKPOINT = "no-root-checkpoint"  # .safetensors อยู่แต่ในโฟลเดอร์ย่อย
NO_CONFIG = "no-config"                    # มี weight ที่รากแต่ไม่มี config.json
NON_LLM_GGUF = "non-llm-gguf"              # GGUF ที่ general.architecture ไม่ใช่ LLM — city96/FLUX.1-dev-gguf
NO_SERVING_MODE = "no-serving-mode"        # งานที่ LMDS ไม่มีโหมดเสิร์ฟ (TTS · ASR · classifier …) — task == "other"
INCOMPLETE_GGUF = "incomplete-gguf"        # split GGUF ที่ repo มีไม่ครบทุก part

# จับเป็น token คั่นด้วย - _ . หรือหัว/ท้ายชื่อ — ไม่จับกลางคำ
_MLX_NAME_RE = re.compile(r"(?:^|[-_.])mlx(?:[-_.]|$)", re.IGNORECASE)
# คีย์ที่ quantizer ฝั่ง transformers ใส่กำกับเสมอ (GPTQ/AWQ: quant_method · ModelOpt: quant_algo ·
# llm-compressor: format) — GPTQ กับ AWQ ก็มี bits + group_size เหมือน MLX ต่างกันที่มีคีย์พวกนี้
_TRANSFORMERS_QUANT_KEYS = ("quant_method", "quant_algo", "format")


def is_mlx_library(library_name: str | None) -> bool:
    """library_name ของ repo อยู่ในตระกูล MLX ไหม — `mlx` และไลบรารีที่ต่อยอดจากมัน (`mlx-<อะไรก็ตาม>`)

    เดิมเทียบ `== "mlx"` ตรงตัว · ของจริงบน Hub (สุ่ม 501 repo 2026-10-06): `mlx` 290 · `mlx-vlm` 4 · `mlx-serve` 3 ·
    `mlx-lm` 1 · `mlx-audio` 1 · `mlx-qwen3-asr` 1 — หกค่านี้คือคำประกาศของเจ้าของ repo เองว่า weight ทำมาให้ MLX ·
    `salakash/Minimalism` (`mlx-lm` · LoRA `adapters.safetensors`) กับ `nativ-community/sapiens2-seg-0.4b-bf16`
    (`mlx-vlm`) จึงหลุดไปเป็น "fits (vllm)" · ไลบรารีชื่ออื่นที่เจอในตัวอย่าง (`mflux` · `mtplx`) ไม่นับจากชื่อ —
    ทั้งสี่ repo นั้นถูกจับได้จากบล็อก quantization อยู่แล้ว และชื่อไลบรารีอย่างเดียวไม่บอกว่าเป็น MLX
    """
    library = (library_name or "").strip().lower()
    return library == MLX or library.startswith(("mlx-", "mlx_"))


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
    ชี้ขาด = library_name อยู่ในตระกูล mlx (`is_mlx_library`) · หรือ config มีบล็อก quantization แบบ MLX · หรือ tag
    `mlx` ที่มีชื่อ repo ยืนยัน/ไม่มี library อื่นแย้ง · ที่เหลือเป็นแค่ร่องรอย ผู้เรียกเอาไปเตือน ไม่ใช่ปฏิเสธ
    """
    library = (library_name or "").strip().lower()
    mlx_library = is_mlx_library(library)
    tagged = any(isinstance(t, str) and t.lower() == MLX for t in tags)
    named = _MLX_NAME_RE.search(repo_id.split("/")[-1]) is not None
    quant = mlx_quantization(config)

    evidence: list[str] = []
    if mlx_library:
        evidence.append(f"library_name={library}")
    if tagged:
        evidence.append("tag mlx")
    if quant is not None:
        evidence.append(f"config.json quantization bits={quant['bits']} group_size={quant['group_size']}")
    if named:
        evidence.append("ชื่อ repo มีคำว่า MLX")
    decisive = mlx_library or quant is not None or (tagged and (named or not library))
    return evidence, decisive


# ── EXL2 / EXL3 (ExLlama) ────────────────────────────────────────────────────────────────────────────
_EXL_TAGS = {"exl2": EXL2, "exllamav2": EXL2, "exl3": EXL3, "exllamav3": EXL3}
_EXL_LIBRARIES = {"exllamav2": EXL2, "exllama2": EXL2, "exllamav3": EXL3, "exllama3": EXL3}
# `trellis` = รันไทม์ที่เสิร์ฟ EXL3 (ชื่อมาจาก trellis coding ของตัว quantizer) — ไลบรารีเดียวไม่ชี้ขาด
# ต้องมีสัญญาณ EXL อื่นคู่กัน (ของจริงสองตัวที่เจอมี tag `exl3` คู่มาทั้งคู่)
_EXL_RUNTIME_LIBRARIES = {"trellis"}
_EXL_NAME_RE = re.compile(r"(?:^|[-_.])exl([23])(?:[-_.]|$)", re.IGNORECASE)
# bits-per-weight ในชื่อ repo ("4.05bpw" · "8bpw-h8") — สำนวนตั้งชื่อของ quant ตระกูล ExLlama
_BPW_NAME_RE = re.compile(r"(?:^|[-_.])\d+(?:\.\d+)?bpw(?:[-_.]|$)", re.IGNORECASE)


def exl_evidence(repo_id: str, library_name: str | None, tags: Iterable[str],
                 config: dict[str, Any] | None) -> tuple[list[str], str | None]:
    """(หลักฐานที่เจอ, "exl2" | "exl3" เมื่อชี้ขาดได้ · None = แค่ร่องรอย/ไม่เจอ)

    เกณฑ์แบบเดียวกับ `mlx_evidence` — สัญญาณเดี่ยวที่มาจากคนพิมพ์ (tag · ชื่อ) ต้องมีอีกอย่างยืนยัน:
    - `quantization_config.quant_method` เป็น exl2/exl3 หรือมีคีย์ `exl2*`/`exl3*` → ชี้ขาด (ตัว quantizer เขียนเอง)
      ของจริง: doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw `{quant_method: exl3, bits: 4.05, codebook: mul1}` ·
      brandonmusic/GLM-5.2-EXL3-…: มีคีย์ `exl3_dense` คู่กับ config_groups ของ ModelOpt (เดิมถูกอ่านเป็น NVFP4)
    - `library_name` exllamav2/exllamav3 → ชี้ขาด
    - สองอย่างขึ้นไปจาก: tag exl2/exl3/exllamav2/exllamav3 · token `exl2`/`exl3` ในชื่อ repo · `<n>bpw` ในชื่อ repo ·
      library_name trellis → ชี้ขาด (icefog72/IceWhiskeyRP-7b-8bpw-exl2: library_name=transformers ที่ลอกมาจาก
      ต้นฉบับ แต่มี tag exl2 + ชื่อ · turboderp/Qwen3.8-27B-exl3: ไม่มี config.json เลย มี tag + ชื่อ)
      · `bpw` กับ `trellis` นับเฉพาะเมื่อมี tag หรือชื่อ EXL อยู่แล้ว — สองตัวนี้ลำพังไม่ได้พูดถึง ExLlama
    - อย่างเดียว → ร่องรอย: ผู้เรียกเตือน ไม่ปฏิเสธ
    """
    library = (library_name or "").strip().lower()
    quant = (config or {}).get("quantization_config")
    quant = quant if isinstance(quant, dict) else {}
    method = str(quant.get("quant_method") or "").strip().lower()
    keyed = sorted(k for k in quant if isinstance(k, str) and k.lower().startswith(("exl2", "exl3")))
    tag_hits = sorted({t.lower() for t in tags if isinstance(t, str) and t.lower() in _EXL_TAGS})
    name = repo_id.split("/")[-1]
    named = _EXL_NAME_RE.search(name)
    bpw = _BPW_NAME_RE.search(name) is not None

    evidence: list[str] = []
    versions: list[str] = []
    if method in (EXL2, EXL3):
        evidence.append(f"config.json quantization_config.quant_method={method}")
        versions.append(method)
    if keyed:
        evidence.append(f"config.json quantization_config มีคีย์ {', '.join(keyed)}")
        versions.append(keyed[0][:4].lower())
    if library in _EXL_LIBRARIES:
        evidence.append(f"library_name={library}")
        versions.append(_EXL_LIBRARIES[library])
    decisive = bool(versions)

    soft = 0
    if tag_hits:
        evidence.append("tag " + ", ".join(tag_hits))
        versions.extend(_EXL_TAGS[t] for t in tag_hits)
        soft += 1
    if named:
        evidence.append(f"ชื่อ repo มีคำว่า EXL{named.group(1)}")
        versions.append(f"exl{named.group(1)}")
        soft += 1
    anchored = bool(tag_hits or named or decisive)
    if library in _EXL_RUNTIME_LIBRARIES and anchored:
        evidence.append(f"library_name={library}")
        soft += 1
    if bpw and anchored:
        evidence.append("ชื่อ repo ระบุ bits-per-weight (bpw)")
        soft += 1
    if not (decisive or soft >= 2):
        return evidence, None
    return evidence, EXL3 if EXL3 in versions else EXL2


def base_model_of(tags: Iterable[str]) -> str:
    """repo ต้นทางจาก tag `base_model:<org>/<name>` / `base_model:quantized:<org>/<name>` — "" = ไม่ระบุ"""
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("base_model:"):
            repo = tag.rsplit(":", 1)[-1]
            if "/" in repo:
                return repo
    return ""


_LABELS = {
    MLX: "MLX (Apple Silicon)",
    EXL2: "EXL2 (ExLlamaV2)",
    EXL3: "EXL3 (ExLlamaV3)",
    DIFFUSERS: "pipeline ของ diffusers",
    ADAPTER: "adapter (LoRA/PEFT) ล้วน",
    NO_WEIGHTS: "ไม่มีไฟล์ weight ที่เสิร์ฟได้",
    NO_ROOT_CHECKPOINT: "ไม่มี checkpoint ที่ราก repo",
    NO_CONFIG: "ไม่มี config.json ที่ราก repo",
    NON_LLM_GGUF: "GGUF ที่ไม่ใช่ LLM",
    NO_SERVING_MODE: "งานที่ LMDS ไม่มีโหมดเสิร์ฟ",
    INCOMPLETE_GGUF: "split GGUF ไม่ครบชุด",
}
_SERVES = "LMDS เสิร์ฟ LLM แบบ chat · embedding · rerank (vLLM/SGLang จาก safetensors · llama.cpp จาก GGUF)"


def unsupported_label(kind: str | None) -> str:
    """ชื่อรูปแบบสำหรับแสดงผล — "mlx" → "MLX (Apple Silicon)" · ไม่รู้จัก = ตัวพิมพ์ใหญ่ของค่าเดิม"""
    return _LABELS.get(kind or "", str(kind or "").upper())


def _wasted_download(report) -> str:
    """ "จะดาวน์โหลด X GB แล้วไปล้มตอน start" — ขนาดที่ลูกค้าจะเสียไปฟรี ๆ ถ้าปล่อยผ่าน"""
    size = f" {report.weight_bytes / 1e9:.1f} GB" if (report.weight_bytes or 0) >= 1e8 else ""
    return f"ถ้าปล่อยผ่านจะดาวน์โหลด{size} แล้วไปล้มตอน start"


def _why_mlx(report, evidence: str) -> str:
    return (
        f"{report.repo_id} เป็น checkpoint รูปแบบ MLX (Apple Silicon) — LMDS deploy ไม่ได้ · "
        f"หลักฐาน: {evidence} · ไฟล์ลงท้าย .safetensors ก็จริง แต่ weight ข้างในจัดเก็บ/quantize แบบ MLX "
        f"ซึ่ง vLLM และ SGLang โหลดไม่ได้ (llama.cpp อ่านได้เฉพาะ .gguf) — {_wasted_download(report)}"
    )


def _why_exl(report, evidence: str) -> str:
    label = unsupported_label(report.unsupported_format)
    return (
        f"{report.repo_id} เป็น checkpoint ที่ quantize แบบ {label} — LMDS deploy ไม่ได้ · "
        f"หลักฐาน: {evidence} · ไฟล์ลงท้าย .safetensors ก็จริง แต่ weight ข้างในเป็นรูปแบบของ ExLlama "
        "(trellis/codebook ต่อ tensor) ซึ่งโหลดได้เฉพาะ ExLlama/TabbyAPI — vLLM และ SGLang โหลดไม่ได้ "
        f"(llama.cpp อ่านได้เฉพาะ .gguf) — {_wasted_download(report)}"
    )


def _why_diffusers(report, evidence: str) -> str:
    return (
        f"{report.repo_id} เป็น pipeline ของ diffusers (โมเดลสร้างภาพ/วิดีโอ) — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · "
        "ไฟล์ .safetensors ใน repo เป็นชิ้นส่วนของ pipeline (unet/transformer · vae · text encoder) ไม่ใช่ checkpoint ของ LLM "
        f"vLLM และ SGLang โหลดไม่ได้ — {_wasted_download(report)}"
    )


def _why_adapter(report, evidence: str) -> str:
    return (
        f"{report.repo_id} เป็น adapter (LoRA/PEFT) ล้วน ไม่มี weight ของตัวโมเดล — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · "
        "adapter ต้องโหลดซ้อนบน base model ซึ่ง repo นี้ไม่มี (ไม่มี checkpoint ที่ราก) — bundle ที่สร้างจาก repo นี้ "
        "จะ start ไม่ขึ้น"
    )


def _why_no_weights(report, evidence: str) -> str:
    return (
        f"{report.repo_id} ไม่มีไฟล์ weight ในรูปแบบที่ LMDS เสิร์ฟ (.safetensors → vLLM/SGLang · .gguf → llama.cpp) — "
        f"deploy ไม่ได้ · ใน repo มี: {evidence} · engine ทั้งสามตัวอ่านรูปแบบพวกนี้ไม่ได้ ถ้าปล่อยผ่านจะได้ bundle ที่ "
        "start ไม่ขึ้นหลังดาวน์โหลดทั้ง repo"
    )


def _why_no_root_checkpoint(report, evidence: str) -> str:
    return (
        f"{report.repo_id} ไม่มี checkpoint ที่ราก repo — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · vLLM/SGLang ถูกชี้ไปที่ "
        "ราก repo เสมอ (config.json + model*.safetensors) ไฟล์ในโฟลเดอร์ย่อยจึงไม่ถูกโหลด — bundle จะ start ไม่ขึ้น"
    )


def _why_no_config(report, evidence: str) -> str:
    return (
        f"{report.repo_id} มีไฟล์ .safetensors ที่รากแต่ไม่มี config.json — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · "
        "vLLM/SGLang ต้องอ่านสถาปัตยกรรมจาก config.json (รูปแบบ mistral ใช้ params.json แทนได้ ซึ่ง repo นี้ก็ไม่มี) — "
        f"{_wasted_download(report)}"
    )


def _why_non_llm_gguf(report, evidence: str) -> str:
    return (
        f"ไฟล์ GGUF ที่เลือกของ {report.repo_id} ไม่ใช่ LLM — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · "
        ".gguf เป็นแค่ภาชนะ: โมเดลสร้างภาพ/วิดีโอและโมเดลเสียงก็เก็บเป็น GGUF ได้ แต่ llama.cpp (llama-server) "
        f"ไม่มี loader ให้สถาปัตยกรรมนี้ — {_wasted_download(report)}"
    )


def _why_no_serving_mode(report, evidence: str) -> str:
    return (
        f"{report.repo_id} ไม่ใช่โมเดล chat / embedding / rerank — LMDS ไม่มีโหมดเสิร์ฟงานนี้ · หลักฐาน: {evidence} · "
        "ถ้าปล่อยผ่านจะถูกวางแผนเป็น chat server ซึ่ง engine เสิร์ฟจาก weight แบบนี้ไม่ได้ — "
        f"{_wasted_download(report)}"
    )


def _why_incomplete_gguf(report, evidence: str) -> str:
    return (
        f"ไฟล์ GGUF ที่เลือกของ {report.repo_id} เป็น split ที่ repo มีไม่ครบชุด — LMDS deploy ไม่ได้ · หลักฐาน: {evidence} · "
        "llama.cpp ต้องมีทุก part ของ split GGUF ถึงจะโหลดได้ (ชื่อไฟล์ -0000N-of-0000M บอกจำนวนที่ต้องมี) — "
        f"{_wasted_download(report)}"
    )


_REASONS = {
    MLX: _why_mlx, EXL2: _why_exl, EXL3: _why_exl, DIFFUSERS: _why_diffusers, ADAPTER: _why_adapter,
    NO_WEIGHTS: _why_no_weights, NO_ROOT_CHECKPOINT: _why_no_root_checkpoint, NO_CONFIG: _why_no_config,
    NON_LLM_GGUF: _why_non_llm_gguf, NO_SERVING_MODE: _why_no_serving_mode, INCOMPLETE_GGUF: _why_incomplete_gguf,
}
# รูปแบบที่ "weight เป็นของโมเดลเดียวกันแต่คนละภาชนะ" — ทางออกคือรุ่นอื่นของโมเดลเดียวกัน
_REQUANTIZED = (MLX, EXL2, EXL3)


def unsupported_reason(report) -> str:
    """ทำไม deploy repo นี้ไม่ได้ — "" = รองรับ · ทางออกอยู่ที่ `unsupported_alternatives`

    `no-serving-mode` มาจากการจัดประเภทงาน (task == "other") — ผู้ใช้ที่รู้ว่าเราจัดผิดทับได้ด้วย `--task`
    (task ไม่ใช่ other แล้ว = ไม่ปฏิเสธ) · ข้ออื่นเป็นเรื่องของไฟล์ ทับไม่ได้
    """
    kind = getattr(report, "unsupported_format", None)
    explain = _REASONS.get(kind)
    if explain is None:
        return ""
    if kind == NO_SERVING_MODE and getattr(report, "task", "other") != "other":
        return ""
    return explain(report, " · ".join(report.unsupported_evidence) or "metadata ของ repo")


def unsupported_alternatives(report) -> list[str]:
    """ของที่ใช้แทนได้ — รุ่นอื่นของ *โมเดลเดียวกัน* ที่ engine ของเราโหลดได้ หรือทางที่ถูกสำหรับงานนั้น"""
    if not unsupported_reason(report):
        return []
    kind = report.unsupported_format
    base = base_model_of(report.tags)
    if kind == ADAPTER:
        return [
            (f"deploy ตัว base model แทน: {base}" if base else "deploy ตัว base model ที่ model card ระบุแทน")
            + " — LMDS ยังไม่โหลด LoRA adapter ซ้อนให้",
            "หรือใช้ repo ที่ merge adapter เข้ากับ base แล้ว (merged checkpoint หรือ GGUF ของรุ่น merged)",
        ]
    if kind in (DIFFUSERS, NON_LLM_GGUF):
        return [f"{_SERVES} — โมเดลสร้างภาพ/วิดีโอ/เสียงใช้รันไทม์ของมันเอง (ComfyUI · diffusers · "
                "stable-diffusion.cpp · whisper.cpp)"]
    if kind == NO_SERVING_MODE:
        return [
            f"{_SERVES} — งานอื่นใช้รันไทม์ของมันเอง",
            "ถ้าจัดประเภทผิด (โมเดลนี้ chat/embed/rerank ได้จริง): lmds deploy <repo> --task generate|embed|rerank",
        ]
    if kind == INCOMPLETE_GGUF:
        whole = [v.filename for v in getattr(report, "gguf_variants", [])
                 if not v.is_mmproj and not v.is_mtp and not v.missing_parts]
        return [
            "เลือก quant อื่นของ repo นี้ที่ครบชุด: " + ", ".join(whole[:4]) if whole
            else "repo นี้ไม่มี GGUF ที่ครบชุดเลย — หา repo อื่นของโมเดลเดียวกัน",
            "หรือแจ้งเจ้าของ repo ว่า upload ไม่ครบ (part ที่ขาดอยู่ในหลักฐานข้างบน)",
        ]
    if kind in (NO_WEIGHTS, NO_ROOT_CHECKPOINT, NO_CONFIG):
        out = ["ใช้ repo ของโมเดลเดียวกันที่มี checkpoint ที่ราก (config.json + model*.safetensors → vLLM/SGLang) "
               "หรือไฟล์ .gguf (→ llama.cpp)" + (f" เช่นต้นฉบับ {base}" if base else "")]
        if base:
            out.append(f"รุ่นที่แปลงจากต้นฉบับเดียวกัน: https://huggingface.co/models?other=base_model:quantized:{base}")
        return out
    out = [
        "ใช้รุ่นอื่นของโมเดลเดียวกันแทน: GGUF (→ llama.cpp) · NVFP4 (→ vLLM บน DGX Spark) · "
        "หรือ safetensors ต้นฉบับ" + (f" {base}" if base else " (ดู base model ใน model card)")
    ]
    if base:
        out.append(f"รุ่นที่ quantize จากต้นฉบับเดียวกัน: https://huggingface.co/models?other=base_model:quantized:{base}")
    return out
