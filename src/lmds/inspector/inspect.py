"""Orchestrator ของการ inspect: Hub API → จำแนก artifact → ไฟล์ metadata → ModelReport"""

from __future__ import annotations

import json
from typing import Any

from lmds.resolver import ModelSource

import re

from .formats import (
    ADAPTER, DIFFUSERS, MLX, NO_CONFIG, NO_ROOT_CHECKPOINT, NO_SERVING_MODE, NO_WEIGHTS, NON_LLM_GGUF,
    exl_evidence, mlx_evidence, mlx_quantization, unsupported_label,
)
from .gguf import GgufInfo, GgufParseError, parse_gguf
from .hf_api import INDEX_FILE_CAP, SMALL_FILE_CAP, BudgetExceeded, HfClient
from .report import ArtifactType, GgufPart, GgufVariant, KvDims, ModelReport, ShardFile

# projector ของ llama.cpp ไม่ได้ขึ้นต้นด้วย mmproj เสมอ — ผู้ quantize บางคนเอาชื่อโมเดลนำ
# เคสจริง 2026-09-04: llmfan46/gemma-4-31B-it-uncensored-heretic-NVFP4-GGUF มีไฟล์
# "gemma-4-31B-it-uncensored-heretic-mmproj-BF16.gguf" · ตรวจแค่ startswith จึง
#   1) โผล่ในรายการ "เลือกไฟล์ weights" ให้ผู้ใช้เลือกผิดได้ (1.1 GB)
#   2) has_mmproj=False → vision ถูกปิดเงียบ ๆ ทั้งที่ Gemma-4 เป็นโมเดลภาพ
# จับเป็น token คั่นด้วย - _ . หรือหัว/ท้ายชื่อ — ไม่จับกลางคำเพื่อไม่ชนชื่ออื่น
_MMPROJ_TOKEN_RE = re.compile(r"(?:^|[-_.])mmproj(?:[-_.]|$)", re.IGNORECASE)


def _is_mmproj(basename: str) -> bool:
    return _MMPROJ_TOKEN_RE.search(basename) is not None


_SPLIT_GGUF_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SHARD_NAME_RE = re.compile(r"^model-\d+-of-(\d+)\.safetensors$")
# .safetensors ที่ราก repo แต่ไม่ใช่ตัวโมเดล — PEFT เขียน adapter_model.safetensors · mlx-lm เขียน adapters.safetensors
_ADAPTER_WEIGHT_FILES = {"adapter_model.safetensors", "adapters.safetensors"}
# นามสกุลที่ถือว่าเป็น "ไฟล์ weight" เวลารายงานของที่ไม่ได้ใช้ (GGUF แยกไปอยู่ใน gguf_variants)
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".onnx_data", ".h5", ".msgpack",
                ".ot", ".mlmodel", ".tflite", ".rkllm", ".mnn", ".nemo")
# ไฟล์ tokenizer ที่ vLLM ต้องใช้จริงตอน serve — ถ้า repo มี ต้องโหลดมาครบด้วย
_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt"}


_EMBED_PIPELINES = {"feature-extraction", "sentence-similarity"}
_EMBED_TAGS = {"sentence-transformers", "sentence-similarity", "feature-extraction",
               "text-embeddings-inference", "embeddings", "embedding"}
_EMBED_NAME_RE = re.compile(r"embed", re.I)
# reranker (cross-encoder): รับ query + document แล้วให้คะแนน — เสิร์ฟ /v1/rerank + /v1/score ไม่ใช่ /v1/embeddings
# ต้องตรวจ *ก่อน* embedding: Qwen/Qwen3-Reranker-4B ติด tag sentence-transformers เหมือน Qwen3-Embedding
# และ BAAI/bge-reranker-v2-m3 มี pipeline_tag text-classification — ดูแค่ tag แล้วจะกลายเป็น embed ทั้งคู่
# (เคสจริง 2026-09-08: `lmds plan Qwen/Qwen3-Reranker-4B` ออกมาเป็น task embed)
_RERANK_PIPELINES = {"text-ranking"}
_RERANK_TAGS = {"reranker", "rerank", "reranking", "cross-encoder", "text-ranking"}
_RERANK_NAME_RE = re.compile(r"rerank", re.I)
# llama.cpp: {arch}.pooling_type = 4 (LLAMA_POOLING_TYPE_RANK) = ไฟล์ GGUF ที่แปลงมาพร้อมหัว classifier ของ reranker
GGUF_POOLING_RANK = 4
# general.architecture ของไฟล์ GGUF ที่ไม่ใช่ LLM — llama-server ไม่มี loader ให้ · เก็บเฉพาะค่าที่อ่านได้จาก header
# ของไฟล์จริงบน Hub (2026-10-06) ไม่เดาเพิ่ม: สถาปัตยกรรมที่ไม่รู้จักยังปล่อยผ่าน ให้ด่านตรวจตอนรันจัดการเหมือนเดิม
#   ภาพ/วิดีโอ (ComfyUI-GGUF / stable-diffusion.cpp): flux (city96/FLUX.1-dev-gguf) · sd3 (city96/stable-diffusion-3.5-large-gguf)
#   · qwen_image (city96/Qwen-Image-gguf) · hidream (city96/HiDream-I1-Dev-gguf) · aura (city96/AuraFlow-v0.3-gguf)
#   · lumina2 (unsloth/Z-Image-Turbo-GGUF) · wan (unsloth/Wan2.2-TI2V-5B-GGUF) · hyvid (city96/HunyuanVideo-gguf)
#   · ltxv (city96/LTX-Video-gguf)
#   เสียง: whisper (handy-computer/whisper-medium-gguf) · asr (nvidia/parakeet-tdt-0.6b-v3)
#   · qwen3-tts / qwen3-tts-tokenizer (Serveurperso/Qwen3-TTS-GGUF)
NON_LLM_GGUF_ARCHITECTURES = frozenset({
    "flux", "sd3", "qwen_image", "hidream", "aura", "lumina2", "wan", "hyvid", "ltxv",
    "whisper", "asr", "qwen3-tts", "qwen3-tts-tokenizer",
})


# pipeline_tag ที่ LMDS ไม่มีโหมดเสิร์ฟ (มีแค่ chat · embedding · rerank) — แยกสองชั้นตามความแน่นอน:
#
# สร้างภาพ/วิดีโอ/3D: ไม่มี engine ไหนของเราทำได้ ไม่ว่าสถาปัตยกรรมข้างในจะเป็นอะไร → ปฏิเสธเสมอ
PIPELINES_NEVER_SERVED = frozenset({
    "text-to-image", "image-to-image", "image-text-to-image", "unconditional-image-generation",
    "text-to-video", "image-to-video", "image-text-to-video", "video-to-video", "text-to-3d", "image-to-3d",
})
# เสียง · อนุกรมเวลา · vision classifier · งาน encoder: ปฏิเสธ **เว้นแต่** weight เป็น causal LM จริง (TTS ที่เป็น LLM พ่น
# audio token อย่าง Orpheus / VieNeu-TTS — engine เสิร์ฟส่วน LM เป็น completions ได้) ซึ่งจะวางแผนต่อพร้อมคำเตือน
PIPELINES_NOT_AN_LLM = frozenset({
    "text-to-speech", "text-to-audio", "audio-to-audio", "automatic-speech-recognition", "audio-classification",
    "voice-activity-detection", "time-series-forecasting", "image-classification", "zero-shot-image-classification",
    "object-detection", "zero-shot-object-detection", "image-segmentation", "mask-generation", "depth-estimation",
    "keypoint-detection", "video-classification", "image-feature-extraction", "token-classification", "fill-mask",
    "tabular-classification", "tabular-regression", "reinforcement-learning", "robotics", "graph-ml",
})
# ที่เหลือ (translation · summarization · question-answering · text-classification · image-to-text …) **ไม่ปฏิเสธจาก
# pipeline_tag**: LLM แบบ decoder-only ถูกติดป้ายพวกนี้อยู่จริง (tencent/Hunyuan-MT-7B = translation +
# HunYuanDenseV1ForCausalLM) — ตัดสินจากสถาปัตยกรรมใน config.json แทน (refine_task)

# หัวของ transformers ที่ไม่ใช่การ generate — แผน chat กับ weight แบบนี้ engine เสิร์ฟไม่ได้แน่นอน
_NON_CHAT_HEADS = (
    "ForSequenceClassification", "ForTokenClassification", "ForMaskedLM", "ForQuestionAnswering", "ForMultipleChoice",
    "ForNextSentencePrediction", "ForImageClassification", "ForObjectDetection", "ForSemanticSegmentation",
    "ForInstanceSegmentation", "ForUniversalSegmentation", "ForImageSegmentation", "ForDepthEstimation",
    "ForVideoClassification", "ForCTC", "ForAudioClassification", "ForAudioFrameClassification", "ForXVector",
    "ForSpeechSeq2Seq",
)
_CAUSAL_LM_HEADS = ("ForCausalLM", "LMHeadModel")


def task_of(info: dict, repo_id: str) -> str:
    """โมเดลนี้เอาไว้ทำอะไร — rerank · embed · generate (chat) · other (งานที่ LMDS ไม่มีโหมดเสิร์ฟ)

    pipeline_tag ที่เจาะจงชนะ tag ลอย ๆ: เดิม tag `feature-extraction` / `text-embeddings-inference` ทำให้
    nvidia/parakeet-tdt-0.6b-v3 (pipeline_tag=automatic-speech-recognition) และ meta-llama/Prompt-Guard-86M
    (text-classification) กลายเป็น task embed (audit 2026-10-06) · tag กลุ่ม embedding นับเฉพาะเมื่อ repo ไม่มี
    pipeline_tag หรือ pipeline_tag เองก็เป็นงาน embedding · ชื่อ repo ยังใช้ได้ (GGUF ที่คนแปลงเองมักไม่มี tag อะไรเลย
    เช่น VesNFF/Qwen3-VL-Embedding-8B-GGUF) · เดาผิดแก้ได้ด้วย `--task`
    """
    pipeline = info.get("pipeline_tag") or ""
    name = repo_id.split("/")[-1]
    tags = {t.lower() for t in info.get("tags", []) if isinstance(t, str)}
    unserved = pipeline in PIPELINES_NEVER_SERVED or pipeline in PIPELINES_NOT_AN_LLM
    if pipeline in _RERANK_PIPELINES or _RERANK_NAME_RE.search(name) or (tags & _RERANK_TAGS and not unserved):
        return "rerank"
    if unserved:
        return "other"
    if pipeline in _EMBED_PIPELINES:
        return "embed"
    if tags & _EMBED_TAGS and not pipeline:
        return "embed"
    if _EMBED_NAME_RE.search(name):
        return "embed"
    return "generate"


def task_from_config(config: dict) -> str | None:
    """งานที่ config.json บอกเอง — `*ForSequenceClassification` ที่มี label เดียว = cross-encoder/reranker

    (BAAI/bge-reranker-v2-m3: XLMRobertaForSequenceClassification · id2label 1 ตัว · Qwen3-Reranker ของแท้ยังเป็น
    Qwen3ForCausalLM — ตัวนั้นจับได้จากชื่อ/pipeline_tag แทน) · หลาย label = โมเดลจัดหมวด ไม่ใช่ reranker → None
    """
    architectures = config.get("architectures")
    arch = str(architectures[0]) if isinstance(architectures, list) and architectures else ""
    if not arch.endswith("ForSequenceClassification"):
        return None
    labels = config.get("id2label")
    num_labels = config.get("num_labels")
    if isinstance(labels, dict) and len(labels) > 1:
        return None
    if isinstance(num_labels, int) and num_labels > 1:
        return None
    return "rerank"


def refine_task(task: str, pipeline: str, config: dict) -> tuple[str, str]:
    """ปรับ task ด้วยสถาปัตยกรรมจริงใน config.json — (task, คำเตือน)

    - pipeline_tag บอกว่าไม่ใช่ LLM (TTS · ASR …) แต่ weight เป็น causal LM → เสิร์ฟเป็น completions ได้ จึงไม่ปฏิเสธ
    - encoder-decoder (T5 · BART · Whisper · Parakeet: `is_encoder_decoder: true`) และหัว classifier/encoder
      (`*ForTokenClassification` · `*ForMaskedLM` · `*ForImageClassification` · `*ForSequenceClassification` หลาย label …)
      ที่กำลังจะถูกวางแผนเป็น chat → other: vLLM/SGLang ไม่มีทาง generate จาก weight แบบนี้
    """
    architectures = config.get("architectures")
    arch = str(architectures[0]) if isinstance(architectures, list) and architectures else ""
    if task == "other":
        if pipeline in PIPELINES_NOT_AN_LLM and arch.endswith(_CAUSAL_LM_HEADS):
            return "generate", (
                f"pipeline_tag ของ repo คือ {pipeline} แต่ weight เป็น causal LM ({arch}) — วางแผนเป็น chat/completions "
                "ตามสถาปัตยกรรม · ไม่ใช่โมเดล chat ทั่วไป: ผลลัพธ์อาจเป็น token เฉพาะงาน (เช่น audio token) ที่ต้องมีตัวถอดรหัสเอง"
            )
        return task, ""
    if task == "generate" and (config.get("is_encoder_decoder") is True or arch.endswith(_NON_CHAT_HEADS)):
        return "other", ""
    return task, ""


def inspect_model(source: ModelSource, client: HfClient) -> ModelReport:
    info = client.model_info(source.repo_id, source.revision)
    revision_sha = info.get("sha") or (source.revision or "main")

    skipped: list[str] = []
    files = _sibling_files(info, skipped)
    safetensor_files = [(n, s) for n, s, _ in files if n.endswith(".safetensors")]
    gguf_files = [(n, s, sha) for n, s, sha in files if n.endswith(".gguf")]

    base = ModelReport(
        repo_id=source.repo_id,
        revision_requested=source.revision,
        revision_sha=revision_sha,
        gated=bool(info.get("gated")),
        private=bool(info.get("private")),
        license=_license_of(info),
        library_name=_library_of(info),
        params_total=_params_of(info),
        tags=[t for t in info.get("tags", []) if isinstance(t, str)],
        task=task_of(info, source.repo_id),
        file_count=len(files),
        trust_remote_code_files=sorted(
            name for name, _, _ in files
            if name.endswith(".py") and name.startswith(("configuration_", "modeling_", "processing_", "tokenization_"))
        ),
        tokenizer_files=sorted(
            name for name, _, _ in files
            if name in _TOKENIZER_FILES
        ),
    )
    if base.trust_remote_code_files:
        base.warnings.append(
            "repo มีไฟล์ Python (trust_remote_code) — ต้อง review ก่อน deploy: "
            + ", ".join(base.trust_remote_code_files)
        )
    if skipped:
        # repr() โดยเจตนา — ชื่อพวกนี้คือสิ่งที่เราไม่ไว้ใจ ห้ามพิมพ์ดิบลงที่ที่อาจถูก copy ไปวางในเชลล์
        report.warnings.append(
            f"ข้ามไฟล์ใน repo {len(skipped)} ไฟล์ที่ชื่อมีอักขระนอกชุดที่รองรับ (ตัวอักษร ตัวเลข . _ - + = @ , / ช่องว่าง) — "
            "LMDS จะไม่ดาวน์โหลด/ตรวจ/เสิร์ฟไฟล์เหล่านี้: "
            + ", ".join(repr(name[:80]) for name in skipped[:5]) + (" …" if len(skipped) > 5 else "")
        )

    variants = _group_gguf_variants(gguf_files)
    gguf_weights = [v for v in variants if not v.is_mmproj and not v.is_mtp]
    # ผู้ใช้ชี้ไฟล์ .gguf มาเอง (ลิงก์ blob/resolve · --gguf · selected_gguf ของหน้าเว็บ) = เลือกทาง llama.cpp
    wants_gguf = bool(source.filename and source.filename.lower().endswith(".gguf"))

    # ฝั่ง safetensors อ่านลงรายงานของมันเองก่อน แล้วค่อยตัดสินว่า repo นี้เสิร์ฟทางไหน — เดิมสองฝั่งเขียนทับ
    # รายงานเดียวกัน (architecture/context/kv_dims ของ config.json ชนะ header ของ GGUF เสมอ) และ repo ที่มี
    # .safetensors ไฟล์เดียวก็กลายเป็น "mixed" → vLLM ทั้งที่ของที่เสิร์ฟได้มีแต่ GGUF
    pipeline = str(info.get("pipeline_tag") or "")
    names = [name for name, _, _ in files]
    st: ModelReport | None = None
    st_problem = ""
    config: dict[str, Any] | None = None
    if safetensor_files:
        st = base.model_copy(deep=True)
        st.artifact_type = ArtifactType.SAFETENSORS
        config, origin = _inspect_safetensors(st, source, client, revision_sha, safetensor_files, pipeline)
        st_problem = _mark_unservable_safetensors(st, config, origin, names, safetensor_files)

    if gguf_weights and (wants_gguf or st is None or st_problem):
        report = base.model_copy(deep=True)
        report.artifact_type = ArtifactType.GGUF
        _inspect_gguf(report, source, client, revision_sha, gguf_files, pipeline)
        config = None
        if st is not None:
            _note_format_choice(report, st, st_problem, gguf_weights, safetensor_files)
    elif st is not None:
        report = st
        report.gguf_variants = variants
        if gguf_weights:
            report.artifact_type = ArtifactType.MIXED
            _note_format_choice(report, st, "", gguf_weights, safetensor_files)
    else:
        # ไม่มีทั้ง safetensors และ GGUF ของตัวโมเดล — เดิมเป็นแค่ artifact "unknown" แล้วทุกชั้นถัดไปเดาเป็น vLLM:
        # `lmds generate onnx-community/Qwen3-0.6B-ONNX` ได้ bundle vLLM ผ่านทุก gate เปิด tool calling ให้ด้วย
        report = base
        report.gguf_variants = variants
        if variants:
            report.warnings.append("พบเฉพาะไฟล์ mmproj/mtp — ไม่มี GGUF ของตัวโมเดล")
        report.unsupported_format = NO_WEIGHTS
        report.unsupported_evidence = _repo_contents(files) or ["ไม่มีไฟล์ weight ใน repo"]
    _note_unused_weights(report, files)
    _mark_unserved_task(report, pipeline, config)
    return report


def _repo_contents(files: list[tuple[str, int | None, str | None]]) -> list[str]:
    """repo ที่ไม่มี safetensors/GGUF มีอะไรอยู่แทน — "ONNX 11 ไฟล์ 9.4 GB" · ให้ข้อความปฏิเสธบอกได้ว่าเจออะไร"""
    kinds = (
        ("ONNX", (".onnx", ".onnx_data")), ("RKLLM (Rockchip NPU)", (".rkllm",)), ("MNN", (".mnn", ".mnn.weight")),
        ("PyTorch pickle (.bin/.pt/.pth/.ckpt)", (".bin", ".pt", ".pth", ".ckpt")), ("TensorFlow (.h5)", (".h5",)),
        ("Flax (.msgpack)", (".msgpack",)), ("CoreML", (".mlmodel", ".mlpackage")), ("TFLite/LiteRT", (".tflite", ".litertlm", ".task")),
        ("NeMo (.nemo)", (".nemo",)), ("mmproj/mtp GGUF", (".gguf",)),
    )
    out: list[str] = []
    for label, exts in kinds:
        hits = [size or 0 for name, size, _ in files if name.lower().endswith(exts)]
        if hits:
            out.append(f"{label} {len(hits)} ไฟล์ {sum(hits) / 1e9:.1f} GB")
    return out


def _mark_unservable_safetensors(st: ModelReport, config: dict[str, Any] | None, origin: str, names: list[str],
                                 safetensor_files: list[tuple[str, int | None]]) -> str:
    """ไฟล์ .safetensors ของ repo นี้เสิร์ฟด้วย vLLM/SGLang ได้ไหม — หมายลง `st.unsupported_format` แล้วคืนเหตุผลสั้น ๆ ("" = ได้)

    โครงสร้างของ repo บอกได้ก่อนโหลดสักไบต์ (audit 2026-10-06 · `lmds generate` เคยออก exit 0 + "static-validated ✅"):
    - pipeline ของ diffusers (`model_index.json` · library_name=diffusers): stabilityai/sdxl-turbo → vLLM "weights 38.8 GiB"
    - adapter ล้วน (`adapter_config.json` / `adapter_model.safetensors` · ไม่มี checkpoint ของ base): IFM/K2-Horizon-7B-Uno
    - checkpoint อยู่แต่ในโฟลเดอร์ย่อย: engine ถูกชี้ไปที่ราก repo เสมอ
    - ไม่มี config.json ที่ราก (และไม่ใช่รูปแบบ mistral ที่มี params.json): vLLM/SGLang อ่านสถาปัตยกรรมไม่ได้
    เหตุผลที่คืนใช้ตัดสิน repo ที่มี GGUF อยู่ด้วย: ฝั่ง safetensors ที่ใช้ไม่ได้ต้องไม่ลาก GGUF ที่ใช้ได้ลงไปด้วย
    (LiquidAI/LFM2.5-2.6B-GGUF · OBLITERATUS/Qwen3.8-27B-OBLITERATED)
    """
    root = {name for name in names if "/" not in name}
    tags = [t.lower() for t in st.tags]
    library = (st.library_name or "").lower()
    if not st.unsupported_format:
        adapter = sorted(root & {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "adapters.safetensors"})
        folders = sorted({name.split("/", 1)[0] + "/" for name, _ in safetensor_files if "/" in name})
        if "model_index.json" in root or library == "diffusers":
            st.unsupported_format = DIFFUSERS
            st.unsupported_evidence = (["model_index.json"] if "model_index.json" in root else []) + (
                [f"library_name={library}"] if library == "diffusers" else []) + sorted(
                t for t in st.tags if t.startswith("diffusers:"))
        elif not st.safetensor_shards and (adapter or library == "peft" or any(t.startswith("base_model:adapter:") for t in tags)):
            st.unsupported_format = ADAPTER
            st.unsupported_evidence = adapter + ([f"library_name={library}"] if library == "peft" else []) + sorted(
                t for t in st.tags if t.startswith("base_model:adapter:"))
        elif not st.safetensor_shards:
            st.unsupported_format = NO_ROOT_CHECKPOINT
            st.unsupported_evidence = [f"ไฟล์ .safetensors อยู่ใน {', '.join(folders[:4])}" if folders
                                       else "ไม่มีไฟล์ .safetensors ของตัวโมเดลที่ราก repo"]
        elif config is None and origin != "mistral":
            st.unsupported_format = NO_CONFIG
            st.unsupported_evidence = ["ไฟล์ที่ราก: " + ", ".join(s.filename for s in st.safetensor_shards[:3])]
    if not st.unsupported_format:
        return ""
    return f"{unsupported_label(st.unsupported_format)} ({' · '.join(st.unsupported_evidence) or 'metadata ของ repo'})"


def _mark_unserved_task(report: ModelReport, pipeline: str, config: dict[str, Any] | None) -> None:
    """งานของโมเดลที่ LMDS ไม่มีโหมดเสิร์ฟ (task = other) → ปฏิเสธ พร้อมหลักฐานว่ารู้ได้จากอะไร

    repo GGUF หลายไฟล์ที่ยังไม่เลือกไฟล์และ pipeline_tag อยู่กลุ่ม "ไม่ใช่ LLM" (TTS …): ยังตัดสินไม่ได้จนกว่าจะอ่าน
    header ของไฟล์ที่เลือก (อาจเป็น LLM พ่น audio token ที่ llama.cpp เสิร์ฟได้) — ปล่อยให้เลือกก่อนพร้อมคำเตือน
    """
    if report.unsupported_format or report.task != "other":
        return
    if (report.artifact_type is ArtifactType.GGUF and report.selected_gguf is None
            and pipeline in PIPELINES_NOT_AN_LLM):
        report.task = "generate"
        report.warnings.append(
            f"pipeline_tag ของ repo คือ {pipeline} — ไม่ใช่งาน chat/embedding/rerank · จะตัดสินจาก header ของไฟล์ GGUF "
            "ที่เลือก: ถ้าไม่ใช่สถาปัตยกรรม LLM จะถูกปฏิเสธ"
        )
        return
    evidence = [f"pipeline_tag={pipeline}"] if pipeline else []
    architectures = (config or {}).get("architectures")
    if isinstance(architectures, list) and architectures:
        evidence.append(f"config.json architectures={architectures[0]}")
    if (config or {}).get("is_encoder_decoder") is True:
        evidence.append("config.json is_encoder_decoder=true")
    report.unsupported_format = NO_SERVING_MODE
    report.unsupported_evidence = evidence


def _note_format_choice(report: ModelReport, st: ModelReport, st_problem: str, gguf_weights: list[GgufVariant],
                        safetensor_files: list[tuple[str, int | None]]) -> None:
    """repo ที่มีทั้ง .safetensors และ .gguf — บอกว่าแผนนี้ใช้ฝั่งไหน อีกฝั่งคืออะไร และสลับอย่างไร"""
    st_size = f"{st.weight_bytes / 1e9:.1f} GB" if st.weight_bytes else "ไม่ทราบขนาด"
    if report.artifact_type is ArtifactType.MIXED:
        example = min(gguf_weights, key=lambda v: abs((v.size_bytes or 0) - (st.weight_bytes or 0) / 4)).filename
        note = (
            f"repo มีทั้ง safetensors และ GGUF — ใช้ checkpoint safetensors {st_size} (→ vLLM/SGLang) · "
            f"อีกทางคือ GGUF {len(gguf_weights)} ไฟล์ (→ llama.cpp): ระบุ --gguf <quant> หรือใส่ลิงก์ไฟล์ .gguf ตรง ๆ "
            f"เช่น https://huggingface.co/{report.repo_id}/blob/main/{example}"
        )
    elif st_problem:
        total = sum(size or 0 for _, size in safetensor_files)
        note = (
            f"ไฟล์ .safetensors ใน repo นี้ ({len(safetensor_files)} ไฟล์ {total / 1e9:.1f} GB) ไม่ได้ใช้ — {st_problem} · "
            "vLLM/SGLang เสิร์ฟไม่ได้ · repo นี้เสิร์ฟทาง GGUF (→ llama.cpp)"
        )
    else:
        using = report.selected_gguf or "ไฟล์ที่เลือก"
        note = (
            f"repo มีทั้ง safetensors และ GGUF — ใช้ GGUF {using} (→ llama.cpp) · "
            f"อีกทางคือ checkpoint safetensors {st_size} (→ vLLM/SGLang): ใส่ชื่อ repo โดยไม่ระบุไฟล์ .gguf"
        )
    report.format_note = note
    report.warnings.append(note)


def _sibling_files(
    info: dict[str, Any], skipped: list[str] | None = None,
) -> list[tuple[str, int | None, str | None]]:
    """(ชื่อ, ขนาด, sha256) ของไฟล์ใน repo — ชื่อที่มีอักขระนอกชุดที่รองรับถูกข้าม (เก็บชื่อไว้ใน `skipped`)

    ชื่อไฟล์มาจาก Hub API ตรง ๆ และใครก็ตั้ง repo ได้ · ชื่อพวกนี้ไปจบใน controller (MODEL_FILES, MODEL_URLS,
    SHARD_FILES) ซึ่งเป็น bash ที่รันบนเครื่องลูกค้า — audit 2026-10-06: ไฟล์ชื่อ `Q4_K_M$(touch PWNED).gguf`
    รันคำสั่งบน node ทันทีที่เรียก controller · ข้ามตั้งแต่ตรงนี้ ไฟล์นั้นจึงไม่เคยเป็นตัวเลือก ไม่เคยถูกนับเป็น shard
    (renderer ปฏิเสธซ้ำอีกชั้น และ escape ทุกค่า — ดู lmds/shellsafe.py)
    """
    from lmds.shellsafe import is_safe_repo_filename

    out: list[tuple[str, int | None, str | None]] = []
    for sibling in info.get("siblings", []) or []:
        name = sibling.get("rfilename")
        if not name:
            continue
        if not isinstance(name, str) or not is_safe_repo_filename(name):
            if skipped is not None:
                skipped.append(str(name))
            continue
        size = sibling.get("size")
        lfs = sibling.get("lfs") or {}
        # Hub ใช้ชื่อคีย์ต่างกันตาม endpoint: `/api/models/<id>?blobs=true` ส่ง `sha256`
        # ส่วน endpoint ของ file tree ส่ง `oid` · เดิมอ่านแต่ `oid` ค่าจึงเป็น None เสมอ
        # กับเส้นทางที่ LMDS ใช้จริง — ผลคือ EXPECTED_SHAS ในทุก controller ว่างเปล่า
        # และ verify-files ลดเหลือ "ขนาดตรงไหม" อย่างเดียว
        #
        # ขนาดตรงแต่เนื้อในเสียเป็นเคสที่เกิดได้จริง (สายหลุดกลางทางแล้ว resume ทับ,
        # ดิสก์คืนบล็อกเสีย) และ GGUF ที่เสียบางไบต์จะโหลดขึ้นแต่ตอบเพี้ยน ซึ่งหาสาเหตุ
        # ยากกว่าไฟล์ที่โหลดไม่ขึ้นมาก
        sha = lfs.get("sha256") or lfs.get("oid")
        out.append((
            name,
            size if size is not None else lfs.get("size"),
            sha if isinstance(sha, str) else None,
        ))
    return out


def _library_of(info: dict[str, Any]) -> str | None:
    value = info.get("library_name") or (info.get("cardData") or {}).get("library_name")
    return str(value) if value else None


def _license_of(info: dict[str, Any]) -> str | None:
    card = info.get("cardData") or {}
    license_value = card.get("license")
    if isinstance(license_value, list):
        license_value = ", ".join(str(v) for v in license_value)
    if license_value:
        return str(license_value)
    for tag in info.get("tags", []) or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.removeprefix("license:")
    return None


# แท็กที่บอกว่า weight ถูกอัดสองพารามิเตอร์ต่อไบต์ (U8 หนึ่งตัว = 4-bit สองตัว)
_PACKED_4BIT_TAGS = {"nvfp4", "mxfp4", "4-bit", "awq", "gptq", "int4", "w4a16", "w4a4"}


def _params_of(info: dict[str, Any]) -> int | None:
    """จำนวนพารามิเตอร์ — Hub นับ element ไม่ใช่พารามิเตอร์ จึงต้องแก้ให้กับ checkpoint ที่อัด 4-bit

    เคสจริง 2026-09-03: scottgl/Qwen3.5-122B-A10B-NVFP4-GB10 ถูกรายงานว่า **26.8B** และ
    Sehyo/Qwen3.5-122B-A10B-NVFP4 ว่า 71.2B ทั้งที่ชื่อบอกอยู่ว่า 122B · Hub นับ U8 ที่อัด
    NVFP4 มาสองตัวต่อไบต์เป็นหนึ่ง และนับ scale F8_E4M3 (หนึ่งตัวต่อ 16 พารามิเตอร์) รวมเข้าไป
    ด้วย · ตัวเลขนี้ไม่กระทบ fit (ใช้ขนาดไฟล์) แต่โผล่บนหน้า inspect และ MODEL_PROFILE
    ให้คนเอาไปตัดสินใจผิด ๆ ว่าโมเดล "เล็ก"
    """
    st = info.get("safetensors") or {}
    total = st.get("total")
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    by_dtype = st.get("parameters") or {}
    packed = int(by_dtype.get("U8") or 0)
    tags = {str(t).lower() for t in (info.get("tags") or [])}
    if packed and tags & _PACKED_4BIT_TAGS:
        # U8 อัดสองตัว · F8_E4M3 ที่มาคู่กับ U8 คือ scale ของ NVFP4 (117B/16 = 7.3B พอดี
        # ในเคสจริง) ไม่ใช่พารามิเตอร์ — ส่วน BF16/F16/F32 คือชั้นที่ไม่ได้ quantize
        others = sum(int(v or 0) for k, v in by_dtype.items() if k not in ("U8", "F8_E4M3"))
        return packed * 2 + others
    return int(total)


def _fetch_json(
    client: HfClient, repo_id: str, revision: str, filename: str, cap: int = SMALL_FILE_CAP
) -> dict[str, Any] | None:
    raw = client.fetch_small_file(repo_id, revision, filename, cap=cap)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def checkpoint_files(names: list[str], index: dict[str, Any] | None,
                     mistral_native: bool = False) -> tuple[list[str], list[str], str]:
    """ไฟล์ .safetensors ชุดไหนคือ *ตัวโมเดล* ที่ vLLM/SGLang จะโหลด — (ไฟล์, shard ที่ index อ้างแต่ไม่มี, ที่มา)

    repo หนึ่งมี weight ได้หลายสำเนา แต่ engine โหลดชุดเดียว · เดิมรวมขนาดทุกไฟล์ .safetensors ทุกโฟลเดอร์:
    เคสจริง 2026-10-06 `openai/gpt-oss-120b` (มี `original/` อีกสำเนา) รายงาน 130.5 GB ทั้งที่ของจริง 65.25 GB
    → needs-smaller-quant บน dgx-spark-single (budget 113.5 GB) ทั้งที่ลงได้ · repo ของ mistralai 38 ตัววาง
    `consolidated*.safetensors` คู่กับ `model-*` ก็โดนนับสองเท่าเหมือนกัน

    ลำดับตรงกับ loader ของ vLLM เอง (`filter_duplicate_safetensors_files`: มี index ก็เชื่อ `weight_map`
    ไม่มีก็ glob `*.safetensors` ที่ **ราก** repo — ไม่ลงโฟลเดอร์ย่อย):
      1. `model.safetensors.index.json` ที่ราก → ไฟล์ที่ `weight_map` ชี้ (เฉพาะที่มีอยู่จริง)
         · index ที่ไม่มี shard ของมันเหลือสักไฟล์ = index ค้างมาจากต้นฉบับ (icefog72/…-exl2) → ข้ามไปข้อถัดไป
      2. `model.safetensors` ที่ราก (ไฟล์เดียว)
      3. `model-NNNNN-of-MMMMM.safetensors` ที่ราก (ไม่มี index)
      4. `consolidated*.safetensors` เมื่อ repo เป็นรูปแบบ mistral (มี `params.json`)
      5. `.safetensors` อื่นที่ราก ยกเว้นไฟล์ adapter — vLLM glob ไฟล์พวกนี้ทั้งหมด
    ไม่เข้าข้อไหนเลย = ไม่มี checkpoint ที่ราก (มีแต่ในโฟลเดอร์ย่อย / มีแต่ adapter) → คืนลิสต์ว่าง
    """
    present = set(names)
    root = sorted(n for n in present if "/" not in n)
    if index is not None:
        shards = sorted({v for v in (index.get("weight_map") or {}).values() if isinstance(v, str)})
        found = [s for s in shards if s in present]
        if found:
            return found, [s for s in shards if s not in present], "index"
    if "model.safetensors" in present:
        return ["model.safetensors"], [], "single"
    shards = [n for n in root if _SHARD_NAME_RE.match(n)]
    if shards:
        # สองชุดที่รากโดยไม่มี index บอกว่าชุดไหน (…-of-00018 กับ …-of-00028) → เอาชุดที่ครบและใหญ่สุด
        by_total: dict[int, list[str]] = {}
        for name in shards:
            by_total.setdefault(int(_SHARD_NAME_RE.match(name).group(1)), []).append(name)
        complete = [total for total, group in by_total.items() if len(group) == total]
        pick = max(complete) if complete else max(by_total, key=lambda total: len(by_total[total]))
        return by_total[pick], [], "shards"
    consolidated = [n for n in root if n.startswith("consolidated")]
    if mistral_native and consolidated:
        return consolidated, [], "mistral"
    rest = [n for n in root if n not in _ADAPTER_WEIGHT_FILES]
    return rest, [], "other" if rest else "none"


def _describe_unused(files: list[tuple[str, int | None]]) -> str:
    """สรุปไฟล์ weight ที่ไม่ได้ใช้เป็นกลุ่มตามโฟลเดอร์ — "original/ 7 ไฟล์ 65.2 GB · metal/model.bin 65.2 GB" """
    groups: dict[str, list[int]] = {}
    for name, size in files:
        key = name.split("/", 1)[0] + "/" if "/" in name else name
        groups.setdefault(key, []).append(size or 0)
    ranked = sorted(groups.items(), key=lambda item: -sum(item[1]))
    parts = [
        f"{key} {len(sizes)} ไฟล์ {sum(sizes) / 1e9:.1f} GB" if key.endswith("/") else f"{key} {sum(sizes) / 1e9:.1f} GB"
        for key, sizes in ranked[:4]
    ]
    if len(ranked) > 4:
        parts.append(f"และอีก {len(ranked) - 4} รายการ")
    return " · ".join(parts)


def _note_unused_weights(report: ModelReport, files: list[tuple[str, int | None, str | None]]) -> None:
    """ไฟล์ weight ใน repo ที่ไม่ใช่ชุดที่จะถูกโหลด — บอกจำนวน/ขนาด ให้เห็นว่าทำไม Weight size ไม่เท่าขนาด repo"""
    used = {s.filename for s in report.safetensor_shards}
    unused = [(name, size) for name, size, _ in files
              if name.endswith(_WEIGHT_EXTS) and name not in used]
    report.other_weight_files = len(unused)
    report.other_weight_bytes = sum(size or 0 for _, size in unused)
    if unused and report.safetensor_shards:
        report.warnings.append(
            f"repo มีไฟล์ weight อื่นอีก {len(unused)} ไฟล์ ({report.other_weight_bytes / 1e9:.1f} GB) ที่ไม่ใช่ "
            f"checkpoint ที่ engine โหลด — ไม่นับในขนาด weight และไม่อยู่ในรายการ shard: {_describe_unused(unused)}"
        )


def _inspect_safetensors(
    report: ModelReport,
    source: ModelSource,
    client: HfClient,
    revision: str,
    safetensor_files: list[tuple[str, int | None]],
    pipeline: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """อ่านฝั่ง safetensors ลง `report` · คืน (config.json ที่ราก repo หรือ None, ที่มาของชุด checkpoint)"""
    index = _fetch_json(client, source.repo_id, revision, _SAFETENSORS_INDEX, cap=INDEX_FILE_CAP)
    sizes_by_name = dict(safetensor_files)
    # รูปแบบ mistral (params.json + consolidated*.safetensors) — ถามเฉพาะเมื่อไม่มีชุดมาตรฐานให้ใช้
    names = list(sizes_by_name)
    chosen, missing, origin = checkpoint_files(names, index)
    mistral_params: dict[str, Any] | None = None
    if origin in ("other", "none") and any(n.startswith("consolidated") and "/" not in n for n in names):
        mistral_params = _fetch_json(client, source.repo_id, revision, "params.json")
        if mistral_params is not None:
            chosen, missing, origin = checkpoint_files(names, index, mistral_native=True)

    report.safetensor_shards = [ShardFile(filename=name, size_bytes=sizes_by_name.get(name)) for name in chosen]
    report.shard_count = len(chosen) or None
    sizes = [sizes_by_name[name] for name in chosen if sizes_by_name.get(name) is not None]
    if chosen and len(sizes) == len(chosen):
        report.weight_bytes = sum(sizes)
    elif chosen:
        report.warnings.append("Hub ไม่รายงานขนาดไฟล์ครบ — weight_bytes อาจไม่ครบถ้วน")
        # ขนาดรวมที่ index จดไว้เองยังดีกว่าผลรวมที่ขาด
        total = (index.get("metadata") or {}).get("total_size") if index is not None else None
        report.weight_bytes = int(total) if isinstance(total, (int, float)) and total > 0 else (sum(sizes) or None)
    if missing:
        report.warnings.append(f"index อ้าง shard ที่ไม่อยู่ใน repo: {', '.join(missing[:5])}")
    elif index is not None and origin != "index":
        report.warnings.append(
            "model.safetensors.index.json อ้าง shard ที่ไม่มีอยู่ใน repo เลย (index ค้างจากต้นฉบับ) — "
            "ใช้ไฟล์ .safetensors ที่มีอยู่จริงแทน"
        )
    if origin == "shards":
        report.warnings.append("มี shard model-*-of-* แต่ไม่มี model.safetensors.index.json — transformers โหลดไม่ได้ถ้าขาด index")
    elif origin == "other":
        report.warnings.append(
            "ไฟล์ weight ไม่ได้ตั้งชื่อตามมาตรฐาน transformers (model.safetensors / model-N-of-M.safetensors): "
            + ", ".join(chosen[:5])
        )
    elif origin == "none":
        report.warnings.append(
            "ไม่พบ checkpoint safetensors ที่ราก repo (มีแต่ในโฟลเดอร์ย่อยหรือเป็นไฟล์ adapter) — "
            "vLLM/SGLang โหลดจากราก repo เท่านั้น"
        )

    config = _fetch_json(client, source.repo_id, revision, "config.json")
    if config is not None:
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            report.architecture = str(architectures[0])
        report.model_type = config.get("model_type") or report.model_type
        # หัว classifier ในไฟล์คือหลักฐานตรงกว่า tag บน Hub — reranker ที่ Hub ติดป้าย text-classification
        # หรือไม่มีคำว่า rerank ในชื่อ ยังถูกจับได้จากตรงนี้
        report.task = task_from_config(config) or report.task
        report.task, task_note = refine_task(report.task, pipeline, config)
        if task_note:
            report.warnings.append(task_note)
        # โมเดล multimodal แยก config ของส่วนข้อความไว้ใต้ text_config — ค่า context
        # อยู่ในนั้น ไม่ใช่ระดับบนสุด · มองแค่ชั้นบนแล้วได้ None ซึ่งไม่ error อะไรเลย
        # แต่ทำให้ fit ถอยไปใช้ค่าตั้งต้น และ bundle ออกมาเล็กกว่าที่โมเดลทำได้หลายเท่า
        text_config = config.get("text_config")
        # `source` เป็นพารามิเตอร์ของฟังก์ชันนี้อยู่แล้ว — ตั้งชื่อชนกันเมื่อไหร่
        # การอ่าน config จะไปแทนที่ ModelSource เงียบ ๆ แล้วพังที่บรรทัดถัดไป
        candidates = [config, text_config] if isinstance(text_config, dict) else [config]
        for candidate in candidates:
            report.context_length = native_context_from_config(candidate)
            if report.context_length:
                break
        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            report.quantization = _quantization_label(quant)
        report.kv_dims = _kv_dims_from_config(config)
        report.hybrid_attention = config_is_hybrid(config)
        report.moe_experts, report.moe_experts_active = _moe_from_config(config)
    elif origin == "mistral" and mistral_params is not None:
        # รูปแบบ mistral ล้วน (params.json + consolidated*.safetensors · ไม่มี config.json): vLLM อ่าน params.json เอง
        # (config_format auto) — ไม่ปฏิเสธ แต่มิติ/context ต้องอ่านจาก params.json ไม่งั้น fit ถอยไปใช้ค่าเดา
        report.model_type = report.model_type or "mistral"
        derived = _config_from_mistral_params(mistral_params)
        report.kv_dims = _kv_dims_from_config(derived)
        report.context_length = derived.get("max_position_embeddings")
        report.warnings.append(
            "repo รูปแบบ mistral (params.json + consolidated*.safetensors · ไม่มี config.json) — vLLM อ่าน params.json เองได้ "
            "แต่ LMDS ยังไม่เคยรันรูปแบบนี้ผ่านบนเครื่องจริง · ถ้า start ไม่ขึ้นให้เพิ่ม --config-format mistral "
            "--load-format mistral --tokenizer-mode mistral หรือใช้ repo ที่มี config.json"
        )
    else:
        report.warnings.append("ไม่พบ config.json — ระบุสถาปัตยกรรมไม่ได้")
    _mark_mlx(report, config)
    _mark_exl(report, config)
    _warn_bitsandbytes(report, config)

    # ModelOpt เก็บ quant_algo (NVFP4/FP8) ไว้ใน hf_quant_config.json ส่วน config.json มักบอกแค่ "modelopt"
    # ซึ่งไม่บอกว่าเป็น FP4 หรือ FP8 → planner เลือก image ผิด (nvidia/Llama-3.3-70B-Instruct-FP4 ได้ nvcr
    # ที่ไม่มี FP4 kernel) · อ่านมาเติมเมื่อค่าจาก config ยังไม่บอกชนิด
    hf_quant = _fetch_json(client, source.repo_id, revision, "hf_quant_config.json")
    if hf_quant is not None and (report.quantization or "").lower() in ("", "modelopt", "quantized"):
        quant_cfg = hf_quant.get("quantization") or {}
        algo = quant_cfg.get("quant_algo")
        report.quantization = str(algo).lower() if algo else (report.quantization or "modelopt")

    tokenizer_config = _fetch_json(client, source.repo_id, revision, "tokenizer_config.json")
    if not report.context_length and tokenizer_config is not None:
        # config.json ไม่บอกเพดาน (Falcon · โมเดล custom code) — tokenizer_config.model_max_length คือแหล่งถัดไป
        # แต่ค่านี้มักเป็น "ไม่จำกัด" (1e30) จึงต้องมีขอบเขตความสมเหตุสมผล
        limit = tokenizer_config.get("model_max_length")
        if isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit <= MAX_SANE_CONTEXT:
            report.context_length = limit
            report.warnings.append(
                f"native context {limit:,} อ่านจาก tokenizer_config.json (model_max_length) — config.json ไม่ระบุ · "
                "ถ้า model card บอกค่าอื่นให้ตั้งด้วย --context"
            )
    template_text = ""
    if tokenizer_config is not None and tokenizer_config.get("chat_template"):
        report.has_chat_template = True
        template_text = str(tokenizer_config.get("chat_template") or "")
    else:
        template = client.fetch_small_file(source.repo_id, revision, "chat_template.jinja")
        report.has_chat_template = template is not None if tokenizer_config is not None else None
        # fetch_small_file คืน bytes — regex ของ capabilities ทำงานกับ str
        template_text = _as_text(template)

    # เนื้อ template คือหลักฐานว่าโมเดลรับ tool / system / thinking ได้ไหม · เดิมดึงมา
    # แล้วดูแค่ว่ามีไฟล์หรือเปล่า แล้วทิ้ง ทั้งที่คำตอบอยู่ในนั้นและตอบได้ก่อนดาวน์โหลด
    from lmds.inspector.capabilities import detect

    has_mmproj = None
    if report.gguf_variants:
        has_mmproj = any(v.is_mmproj for v in report.gguf_variants)
    report.capabilities = detect(
        config if config is not None else {},
        template_text,
        has_mmproj=has_mmproj,
        moe_experts=report.moe_experts,
        moe_experts_active=report.moe_experts_active,
    ).to_dict()
    return config, origin


# คีย์ที่ config.json ใช้บอกเพดาน context — ชื่อต่างกันตามตระกูล: max_position_embeddings (ส่วนใหญ่) ·
# n_positions (GPT-2/T5) · seq_length (ChatGLM/GLM-4: THUDM/glm-4-9b-chat) · n_ctx (GPT-2 รุ่นเก่า/CTRL) ·
# max_seq_len (MPT · mistral params) · max_sequence_length
_CONTEXT_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions", "seq_length", "n_ctx", "max_seq_len")
# เกินนี้ไม่ใช่เพดานจริง — tokenizer_config.model_max_length ที่ไม่ได้ตั้งเป็น int(1e30)
MAX_SANE_CONTEXT = 4 * 1024 * 1024


def native_context_from_config(config: dict[str, Any]) -> int | None:
    """เพดาน context ที่ config บอกเอง — None = ไม่บอก (ไม่เดา: "ไม่รู้" กับ "รู้ว่าเล็ก" เป็นคนละเรื่อง)"""
    for key in _CONTEXT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_SANE_CONTEXT:
            return value
    return None


def _config_from_mistral_params(params: dict[str, Any]) -> dict[str, Any]:
    """params.json ของ mistral → คีย์แบบ config.json เท่าที่ fit ใช้ (มิติ KV + context)"""
    context = next((params[k] for k in ("max_position_embeddings", "max_seq_len")
                    if isinstance(params.get(k), int) and params[k] > 0), None)
    return {
        "num_hidden_layers": params.get("n_layers"),
        "num_attention_heads": params.get("n_heads"),
        "num_key_value_heads": params.get("n_kv_heads"),
        "head_dim": params.get("head_dim"),
        "hidden_size": params.get("dim"),
        "max_position_embeddings": context,
    }


def _mark_mlx(report: ModelReport, config: dict[str, Any] | None) -> None:
    """checkpoint แบบ MLX เก็บใน .safetensors เหมือนกัน — ต้องแยกออกตรงนี้ ก่อนใครเอา artifact_type ไปเลือก engine

    เคสจริง 2026-10-05: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP ถูกรายงานเป็น safetensors ธรรมดา
    (Quantization "quantized") แล้ว planner ส่งให้ vLLM ซึ่งโหลดไม่ได้ — เกณฑ์อยู่ที่ formats.mlx_evidence
    """
    evidence, decisive = mlx_evidence(report.repo_id, report.library_name, report.tags, config)
    if decisive:
        report.unsupported_format = MLX
        report.unsupported_evidence = evidence
        quant = mlx_quantization(config)
        if quant is not None:
            report.quantization = f"mlx-{quant['bits']}bit"
    elif evidence:
        report.warnings.append(
            f"มีร่องรอยของ MLX ({' · '.join(evidence)}) แต่ไม่พอชี้ขาด — เช็ค model card ก่อน deploy: "
            "ถ้า weight เป็นรูปแบบ MLX จริง vLLM/SGLang จะโหลดไม่ได้"
        )


def _mark_exl(report: ModelReport, config: dict[str, Any] | None) -> None:
    """quant ของ ExLlama (EXL2/EXL3) เก็บใน .safetensors เหมือนกัน — บั๊กตัวเดียวกับ MLX อีกรอบ

    เคสจริง 2026-10-06: `lmds generate doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw` ได้ bundle vLLM context 217,088
    เปิด tool calling ให้ ผ่านทุก gate (108 GB) · เดิมอ่าน `quant_method: exl3` แล้วแค่พิมพ์ลงช่อง Quantization
    """
    if report.unsupported_format:
        return
    evidence, kind = exl_evidence(report.repo_id, report.library_name, report.tags, config)
    if kind is not None:
        report.unsupported_format = kind
        report.unsupported_evidence = evidence
        bits = ((config or {}).get("quantization_config") or {}).get("bits")
        report.quantization = f"{kind}-{bits}bpw" if isinstance(bits, (int, float)) else kind
    elif evidence:
        report.warnings.append(
            f"มีร่องรอยของ quant แบบ ExLlama ({' · '.join(evidence)}) แต่ไม่พอชี้ขาด — เช็ค model card ก่อน deploy: "
            "ถ้า weight เป็น EXL2/EXL3 จริง vLLM/SGLang จะโหลดไม่ได้"
        )


def _warn_bitsandbytes(report: ModelReport, config: dict[str, Any] | None) -> None:
    """bitsandbytes (bnb-4bit/8bit): vLLM โหลดได้ *ถ้า* image มีแพ็กเกจ bitsandbytes — ยังไม่เคยยืนยันบน image ของเรา

    ไม่ปฏิเสธ: พิสูจน์ไม่ได้ว่า image ที่ LMDS ใช้ไม่มีแพ็กเกจนี้ (ปฏิเสธผิด = ขวางลูกค้าโดยไม่มีทางข้าม) ·
    แต่ต้องบอก เพราะถ้าไม่มีจะรู้ตอน start หลังโหลดครบแล้ว และ vLLM ไม่ทำ tensor parallel ให้ bitsandbytes
    """
    quant = (config or {}).get("quantization_config")
    if isinstance(quant, dict) and str(quant.get("quant_method") or "").lower() == "bitsandbytes":
        report.warnings.append(
            "quantize ด้วย bitsandbytes — vLLM โหลดได้เฉพาะเมื่อ image มีแพ็กเกจ bitsandbytes (ยังไม่ได้ยืนยันกับ image "
            "ที่ LMDS ใช้) และไม่รองรับ tensor parallel (stacked / หลาย GPU) · ถ้า start ไม่ขึ้นให้ใช้ checkpoint ต้นฉบับ "
            "หรือรุ่น AWQ/GPTQ/NVFP4/GGUF ของโมเดลเดียวกันแทน"
        )


def _quantization_label(quant: dict) -> str:
    """ชนิด quantization จาก `quantization_config` — เอาตัวที่บอก *ชนิด* ก่อนตัวที่บอก *เครื่องมือ*

    `quant_method` คือเครื่องมือ (modelopt · compressed-tensors) ไม่ใช่ชนิด · ชนิดอยู่ที่ `quant_algo`
    (ModelOpt: NVFP4/FP8) หรือ `format` (llm-compressor: nvfp4-pack-quantized) · เดิมเอา quant_method ก่อน
    → checkpoint NVFP4 ของ NVIDIA/RedHat รายงานว่า "modelopt"/"compressed-tensors" แล้ว planner ไม่รู้ว่าเป็น FP4
    """
    algo = quant.get("quant_algo")
    if algo:
        return str(algo).lower()
    fmt = str(quant.get("format") or "").lower()
    if "fp4" in fmt:
        return fmt
    return str(quant.get("quant_method") or "quantized")


def _group_gguf_variants(gguf_files: list[tuple[str, int | None, str | None]]) -> list[GgufVariant]:
    """รวม split GGUF (-00001-of-N) เป็น variant เดียว — ขนาดรวมทุก part, download/verify ครบชุด"""
    singles: list[GgufVariant] = []
    groups: dict[str, list[tuple[int, GgufPart]]] = {}

    for name, size, sha in sorted(gguf_files):
        base = name.rsplit("/", 1)[-1]
        match = _SPLIT_GGUF_RE.match(base)
        part = GgufPart(filename=name, size_bytes=size, sha256=sha)
        if match:
            key = name[: len(name) - len(base)] + match.group("base")
            groups.setdefault(key, []).append((int(match.group("idx")), part))
        else:
            singles.append(
                GgufVariant(
                    filename=name, size_bytes=size, sha256=sha,
                    is_mmproj=_is_mmproj(base),
                    # mtp ตรวจเฉพาะขึ้นต้น — ชื่อโมเดลอย่าง "…-MTP-Preserved-APEX" คือ weights
                    # ที่เก็บหัว MTP ไว้ ไม่ใช่ไฟล์ mtp แยก จับกลางชื่อจะทิ้ง weights ตัวจริง
                    is_mtp=base.lower().startswith("mtp"),
                )
            )

    for parts_list in groups.values():
        parts_list.sort(key=lambda item: item[0])
        parts = [p for _, p in parts_list]
        sizes = [p.size_bytes for p in parts if p.size_bytes is not None]
        singles.append(
            GgufVariant(
                filename=parts[0].filename,
                size_bytes=sum(sizes) if len(sizes) == len(parts) else None,
                sha256=parts[0].sha256,
                parts=parts,
            )
        )
    return sorted(singles, key=lambda v: v.filename)


def _as_text(value: object) -> str:
    """ไฟล์เล็กจาก Hub มาเป็น bytes — แปลงเป็นข้อความแบบไม่ตายกับไบต์เสีย"""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value) if value else ""


def _moe_from_config(config: dict) -> tuple[int | None, int | None]:
    """จำนวน expert ทั้งหมด/ที่เปิดต่อ token — ชื่อคีย์ต่างกันไปตามตระกูล

    โมเดล multimodal ซุกไว้ใต้ text_config เหมือนที่ทำกับ context_length
    ถ้ามองแค่ชั้นบนจะได้ None เงียบ ๆ แล้ว MoE กลายเป็น dense ในสายตาระบบ
    """
    candidates = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config)
    total = active = None
    for candidate in candidates:
        for key in ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"):
            value = candidate.get(key)
            if isinstance(value, int) and value > 0:
                total = total or value
                break
        for key in ("num_experts_per_tok", "moe_topk", "num_experts_per_token"):
            value = candidate.get(key)
            if isinstance(value, int) and value > 0:
                active = active or value
                break
    return total, active


def _hybrid_attention_layers(scope: dict[str, Any], layers: int | None) -> int | None:
    """จำนวน layer ที่ KV โตตาม context สำหรับ arch แบบ hybrid linear-attention

    คืน None เมื่ออ่านรูปแบบไม่ออก — ให้ผู้เรียกตกไปทางปกติ ดีกว่าเดาแล้วได้ 0 layer
    """
    kinds = scope.get("layer_types")
    if isinstance(kinds, list) and kinds:
        full = [k for k in kinds if isinstance(k, str) and k == "full_attention"]
        if full and len(full) < len(kinds):
            return len(full)
        return None  # ทุก layer เป็น full attention อยู่แล้ว — ไม่ใช่ hybrid

    interval = scope.get("full_attention_interval")
    if isinstance(interval, int) and interval > 1 and isinstance(layers, int) and layers > 0:
        # ปัดขึ้นเหมือนทาง GGUF: 65 layer ทุก ๆ 4 = 17 ไม่ใช่ 16
        # ประเมินเกินหนึ่ง layer ปลอดภัยกว่าประเมินขาดแล้ว OOM ตอนโหลด
        return -(-layers // interval)
    return None


def config_is_hybrid(config: dict[str, Any]) -> bool:
    """arch นี้สลับ full attention กับ linear/SSM ไหม — ดูจาก config ไม่ใช่จากชื่อรุ่น"""
    scope = config
    if not config.get("num_hidden_layers") and isinstance(config.get("text_config"), dict):
        scope = config["text_config"]
    return _hybrid_attention_layers(scope, scope.get("num_hidden_layers")) is not None


def _kv_dims_from_config(config: dict[str, Any]) -> KvDims | None:
    """อ่านมิติ KV จาก config.json — รองรับ text_config ซ้อน (โมเดล multimodal)"""
    scope = config
    if not config.get("num_hidden_layers") and isinstance(config.get("text_config"), dict):
        scope = config["text_config"]

    layers = scope.get("num_hidden_layers")
    heads = scope.get("num_attention_heads")
    kv_heads = scope.get("num_key_value_heads") or heads
    head_dim = scope.get("head_dim")
    if head_dim is None and isinstance(scope.get("hidden_size"), int) and isinstance(heads, int) and heads:
        head_dim = scope["hidden_size"] // heads
    # hybrid Mamba (NemotronH, Jamba, Zamba): มีแค่บาง layer ที่เป็น attention จริง
    # ที่เหลือเป็น Mamba ซึ่ง state คงที่ไม่โตตาม context — นับรวมคือประเมินเกินหลายเท่า
    #
    # เคสจริง 2026-08-14: NVIDIA-Nemotron-3-Super-120B-A12B มี 88 layers แต่
    # `hybrid_override_pattern` บอกว่าเป็น attention แค่ 8 ตัว (M=Mamba 40, E=MLP 40, *=attn 8)
    # สูตรเดิมคิด 88 KiB/token ทั้งที่ของจริง 8 KiB/token — เกินจริง 11 เท่า
    #
    # นับเฉพาะ '*' เท่านั้น · pattern ที่ไม่มี '*' เลยแปลว่าเราอ่านรูปแบบนี้ไม่ออก
    # ปล่อยให้ตกไปทางปกติดีกว่าเดาแล้วได้ 0 layer
    pattern = scope.get("hybrid_override_pattern")
    if isinstance(pattern, str) and "*" in pattern:
        attention_layers = pattern.count("*")
        if isinstance(kv_heads, int) and isinstance(head_dim, int) and kv_heads > 0 and head_dim > 0:
            return KvDims(layers=attention_layers, kv_heads=kv_heads, head_dim=head_dim)

    # hybrid linear-attention (Qwen3.5, Qwen3-Next): full attention สลับกับ layer ที่เป็น
    # SSM/linear ซึ่ง state คงที่ไม่โตตาม context · HF config บอกด้วย `layer_types`
    # (ลิสต์ต่อ layer) หรือ `full_attention_interval` (ทุก ๆ N layer)
    #
    # เคสจริง 2026-08-19: orcarouter/Qwen3.8-27B-Uncensored-NVFP4 มี 64 layer แต่เป็น
    # full attention แค่ 16 (interval 4) · สูตรเดิมคิด 256 KiB/token ทั้งที่ของจริง 64 KiB
    # — เกินจริง 4 เท่า แล้วไปบอกว่าที่ context 262,144 รับได้ 1.4 คนพร้อมกัน ทั้งที่ได้ 5.8
    #
    # ทาง GGUF จับเคสนี้ได้มาตั้งแต่ `_interval_layers_only` แต่ทาง safetensors ไม่เคยมอง
    # — repo เดียวกันคนละรูปแบบไฟล์จึงให้คำตอบคนละอย่าง
    attention_layers = _hybrid_attention_layers(scope, layers)
    if attention_layers is not None:
        if isinstance(kv_heads, int) and isinstance(head_dim, int) and kv_heads > 0 and head_dim > 0:
            return KvDims(layers=attention_layers, kv_heads=kv_heads, head_dim=head_dim)

    # MLA (DeepSeek-V2/V3, Kimi K2/K3): บีบ K กับ V ให้เหลือ latent ก้อนเดียวต่อ layer
    # ขนาด kv_lora_rank + qk_rope_head_dim · สูตร GQA ปกติจะเกินจริงหลายสิบเท่า
    #
    # เคสจริง 2026-08-14: Kimi-K3-active-slice-32experts (93 layers, 96 heads) ถูกคิดเป็น
    # 2,581 KiB/token ทั้งที่ของจริงคือ 105 KiB/token — เกินจริง 24.7 เท่า แล้วไปตัด context
    # เหลือ 16,384 ทั้งที่โมเดลรองรับ 1,048,576 และหน่วยความจำพอถึงหลักแสน
    latent = scope.get("kv_lora_rank")
    if isinstance(latent, int) and latent > 0 and isinstance(layers, int) and layers > 0:
        rope = scope.get("qk_rope_head_dim")
        width = latent + (rope if isinstance(rope, int) and rope > 0 else 0)
        return KvDims(layers=layers, kv_heads=1, head_dim=width, latent_dim=width)

    if all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
        return KvDims(layers=layers, kv_heads=kv_heads, head_dim=head_dim)
    return None


def _kv_dims_from_gguf(gguf: GgufInfo) -> KvDims | None:
    arch = gguf.architecture
    if not arch:
        return None
    meta = gguf.metadata
    layers = meta.get(f"{arch}.block_count")
    heads = meta.get(f"{arch}.attention.head_count")
    kv_heads = meta.get(f"{arch}.attention.head_count_kv") or heads
    head_dim = meta.get(f"{arch}.attention.key_length")
    embedding = meta.get(f"{arch}.embedding_length")
    if head_dim is None and isinstance(embedding, int) and isinstance(heads, int) and heads:
        head_dim = embedding // heads

    # โมเดล sliding-window (gemma-4, และตัวอื่นที่ใช้ท่าเดียวกัน) เขียน head_count_kv
    # เป็น "ลิสต์ต่อ layer" ไม่ใช่เลขตัวเดียว เพราะแต่ละ layer ใช้ไม่เท่ากัน โค้ดเดิม
    # เช็ค isinstance(int) แล้วตกทันที คืน None → analyser ไปเข้าสาขา "ไม่รู้มิติ KV"
    # ที่ตั้ง context ไว้แค่ 16,384 ทั้งที่โมเดลรองรับ 262,144
    #
    # เคสจริง 2026-08-13: gemma-4-31B บน dgx-veerasiam รันมา 16,384 ด้วยเหตุนี้
    # ทั้งที่หน่วยความจำเหลือพอสำหรับ 262,144 — เสีย context ไป 16 เท่าโดยไม่มีใครรู้
    if isinstance(kv_heads, list):
        layers, kv_heads, head_dim = _scaling_layers_only(meta, arch, kv_heads, head_dim)
    else:
        layers = _interval_layers_only(meta, arch, layers)

    if all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
        return KvDims(layers=layers, kv_heads=kv_heads, head_dim=head_dim)
    return None


def _interval_layers_only(meta: dict, arch: str, layers: int | None) -> int | None:
    """เก็บเฉพาะ layer full-attention ของ arch แบบ hybrid ที่บอกด้วย "ทุก ๆ N layer"

    qwen3.5 / qwen3-next วาง full attention สลับกับ layer ที่เป็น SSM (linear attention)
    ซึ่ง state คงที่ไม่โตตาม context แล้วประกาศจังหวะไว้ที่ `full_attention_interval`
    แทนที่จะไล่เป็นลิสต์ต่อ layer อย่าง gemma-4 · `_scaling_layers_only` จับได้เฉพาะ
    แบบลิสต์ ของพวกนี้จึงถูกคูณด้วยจำนวน layer ทั้งหมด = ประเมิน KV เกินไป N เท่า

    เคสจริง 2026-08-15: Qwen3.8-27B (65 layer, interval 4) ถูกคิดเป็น 260 KiB/token
    → 95 GiB ที่ context 262,144 ทั้งที่ของจริง 64 KiB/token → 49 GiB (เครื่องวัดได้
    54 GB) ผลคือ fit ปฏิเสธการรันคู่กับโมเดลอื่นที่จริง ๆ แล้วรันได้สบาย
    """
    interval = meta.get(f"{arch}.full_attention_interval")
    if not isinstance(layers, int) or not isinstance(interval, int) or interval <= 1:
        return layers
    # ปัดขึ้น: 65 layer ทุก ๆ 4 = 17 layer ที่เป็น full attention ไม่ใช่ 16
    # ประเมินเกินหนึ่ง layer ปลอดภัยกว่าประเมินขาดแล้ว OOM ตอนโหลด
    return -(-layers // interval)


def _scaling_layers_only(
    meta: dict, arch: str, kv_heads: list, head_dim: int | None
) -> tuple[int | None, int | None, int | None]:
    """เก็บเฉพาะ layer ที่ KV โตตาม context.

    layer ที่เป็น sliding-window ใช้ KV คงที่เท่าขนาดหน้าต่าง (gemma-4 = 1024 token)
    ไม่ว่า context จะยาวแค่ไหน การนับมันรวมไปด้วยทำให้ประเมิน KV เกินจริงหลายเท่า
    แล้วไปตัด context ทิ้งโดยไม่จำเป็น — จึงนับเฉพาะ layer full-attention

    ที่เหลือคือส่วนคงที่ (gemma-4 ราว 800 MiB) ซึ่งไม่ได้บวกไว้ตรงนี้ เพราะ KvDims
    คิดเป็น bytes/token ล้วน ๆ ส่วนต่างนี้คงที่และเล็กกว่า reserve ของ preset มาก
    """
    pattern = meta.get(f"{arch}.attention.sliding_window_pattern")
    if isinstance(pattern, list) and len(pattern) == len(kv_heads):
        # pattern[i] เป็น True = layer นั้น sliding → ตัดออก เหลือแต่ full-attention
        full = [n for n, sliding in zip(kv_heads, pattern, strict=True) if not sliding]
        if full and all(isinstance(n, int) and n > 0 for n in full):
            # key_length เป็นของ layer full-attention อยู่แล้ว (SWA ใช้ key_length_swa)
            return len(full), max(full), head_dim

    # ไม่มี pattern ให้ดู — ไม่เดาว่า layer ไหนเป็นอะไร ใช้ค่ามากสุดกับทุก layer
    # ประเมินเกินจริงดีกว่าประเมินขาดแล้ว OOM ตอนโหลด
    usable = [n for n in kv_heads if isinstance(n, int) and n > 0]
    if not usable:
        return None, None, None
    return len(kv_heads), max(usable), head_dim


def _inspect_gguf(
    report: ModelReport,
    source: ModelSource,
    client: HfClient,
    revision: str,
    gguf_files: list[tuple[str, int | None, str | None]],
    pipeline: str = "",
) -> None:
    report.gguf_variants = _group_gguf_variants(gguf_files)
    weight_variants = [v for v in report.gguf_variants if not v.is_mmproj and not v.is_mtp]
    if not weight_variants:
        report.warnings.append("พบเฉพาะไฟล์ mmproj/mtp — ไม่มี GGUF ของตัวโมเดล")
        return

    if source.filename:
        selected = next(
            (
                v for v in weight_variants
                if v.filename == source.filename
                or any(p.filename == source.filename for p in v.parts)
            ),
            None,
        )
        if selected is None:
            report.warnings.append(f"ไม่พบไฟล์ {source.filename} ใน repo — ต้องเลือก variant ใหม่")
    elif len(weight_variants) == 1:
        selected = weight_variants[0]
    else:
        selected = None
        report.warnings.append(
            f"repo มี GGUF {len(weight_variants)} variant — ต้องเลือกไฟล์ตอน deploy (ยังไม่อ่าน header)"
        )

    if selected is None:
        return
    report.selected_gguf = selected.filename
    report.weight_bytes = selected.size_bytes

    try:
        gguf = parse_gguf(client.range_source(source.repo_id, revision, selected.filename))
    except (GgufParseError, BudgetExceeded, EOFError) as exc:
        report.warnings.append(f"อ่าน GGUF header ไม่สำเร็จ: {exc}")
        return
    report.architecture = report.architecture or gguf.architecture
    report.gguf_architecture = gguf.architecture
    if (gguf.architecture or "").lower() in NON_LLM_GGUF_ARCHITECTURES:
        # ไฟล์ .gguf เป็นแค่ภาชนะเหมือน .safetensors — โมเดล diffusion/เสียงก็เก็บเป็น GGUF (ComfyUI-GGUF ·
        # stable-diffusion.cpp · whisper.cpp) แต่ llama-server โหลดไม่ได้ · เคสจริง 2026-10-06:
        # `lmds generate …/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf` ได้ bundle llama.cpp ผ่านทุก gate
        report.unsupported_format = NON_LLM_GGUF
        report.unsupported_evidence = [f"general.architecture={gguf.architecture}", f"ไฟล์ {selected.filename}"]
        return
    if report.task == "other" and pipeline in PIPELINES_NOT_AN_LLM:
        # pipeline_tag บอกว่าเป็นงานเสียง ฯลฯ แต่ header เป็นสถาปัตยกรรม LLM ที่ llama.cpp โหลดได้ (VieNeu-TTS = qwen3)
        report.task = "generate"
        report.warnings.append(
            f"pipeline_tag ของ repo คือ {pipeline} แต่ไฟล์ GGUF เป็นสถาปัตยกรรม LLM ({gguf.architecture}) — วางแผนเป็น "
            "chat/completions · ไม่ใช่โมเดล chat ทั่วไป: ผลลัพธ์อาจเป็น token เฉพาะงานที่ต้องมีตัวถอดรหัสเอง"
        )
    report.context_length = report.context_length or gguf.context_length
    report.kv_dims = report.kv_dims or _kv_dims_from_gguf(gguf)
    if isinstance(gguf.metadata, dict) and report.architecture:
        interval = gguf.metadata.get(f"{report.architecture}.full_attention_interval")
        report.hybrid_attention = report.hybrid_attention or (
            isinstance(interval, int) and interval > 1)
    report.moe_experts = report.moe_experts or gguf.expert_count
    report.moe_experts_active = report.moe_experts_active or gguf.expert_used_count
    report.mtp_embedded = report.mtp_embedded or bool(gguf.nextn_layers)
    # ไฟล์ที่แปลงมาพร้อม pooling_type=RANK คือ reranker แม้ชื่อ repo จะไม่บอก (llama-server ต้อง --reranking)
    if gguf.pooling_type == GGUF_POOLING_RANK:
        report.task = "rerank"

    # ไฟล์ฝั่ง speculative ต้องถูกอ่าน header ด้วย ไม่ใช่เชื่อชื่อไฟล์: ถ้ามันเป็นหัวล้วน
    # การส่งเข้า --spec-draft-model ทำให้ start ไม่ขึ้น (ดู GgufInfo.is_standalone_model)
    for head in report.gguf_variants:
        if not head.is_mtp or head.is_standalone_draft is not None:
            continue
        try:
            head_info = parse_gguf(client.range_source(source.repo_id, revision, head.filename))
        except (GgufParseError, BudgetExceeded, EOFError) as exc:
            report.warnings.append(f"อ่าน header ของ {head.filename} ไม่สำเร็จ: {exc}")
            continue
        head.is_standalone_draft = head_info.is_standalone_model
    if report.has_chat_template is None:
        report.has_chat_template = gguf.chat_template is not None
    if gguf.file_type is not None and not report.quantization:
        report.quantization = f"gguf-file-type-{gguf.file_type}"
    name_upper = selected.filename.upper()
    for marker in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "F16", "BF16", "IQ1", "IQ2", "IQ3", "IQ4"):
        if f"-{marker}" in name_upper or f".{marker}" in name_upper or f"_{marker}" in name_upper:
            suffix = name_upper.split(marker, 1)[1].split(".GGUF", 1)[0]
            report.quantization = (marker + suffix).strip("-_.")
            break

    # repo GGUF ล้วนไม่ผ่าน _inspect_safetensors จึงไม่เคยมีใครเรียก detect() — capabilities
    # ว่างเปล่ามาตลอด ทั้งที่ chat template กับ metadata อยู่ในไฟล์ GGUF ครบแล้ว
    if not report.capabilities:
        from lmds.inspector.capabilities import detect

        report.capabilities = detect(
            {},
            gguf.chat_template or "",
            has_mmproj=any(v.is_mmproj for v in report.gguf_variants),
            moe_experts=report.moe_experts,
            moe_experts_active=report.moe_experts_active,
        ).to_dict()
    if gguf.partial:
        report.warnings.append("GGUF metadata อ่านได้บางส่วน (ชน budget) — ข้อมูลอาจไม่ครบ")
