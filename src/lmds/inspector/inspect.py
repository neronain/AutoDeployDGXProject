"""Orchestrator ของการ inspect: Hub API → จำแนก artifact → ไฟล์ metadata → ModelReport"""

from __future__ import annotations

import json
from typing import Any

from lmds.resolver import ModelSource

import re

from .gguf import GgufInfo, GgufParseError, parse_gguf
from .hf_api import INDEX_FILE_CAP, SMALL_FILE_CAP, BudgetExceeded, HfClient
from .report import ArtifactType, GgufPart, GgufVariant, KvDims, ModelReport, ShardFile

_SPLIT_GGUF_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

_SAFETENSORS_INDEX = "model.safetensors.index.json"
# ไฟล์ tokenizer ที่ vLLM ต้องใช้จริงตอน serve — ถ้า repo มี ต้องโหลดมาครบด้วย
_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt"}


def inspect_model(source: ModelSource, client: HfClient) -> ModelReport:
    info = client.model_info(source.repo_id, source.revision)
    revision_sha = info.get("sha") or (source.revision or "main")

    files = _sibling_files(info)
    safetensor_files = [(n, s) for n, s, _ in files if n.endswith(".safetensors")]
    gguf_files = [(n, s, sha) for n, s, sha in files if n.endswith(".gguf")]

    artifact = _classify(bool(safetensor_files), bool(gguf_files))
    report = ModelReport(
        repo_id=source.repo_id,
        revision_requested=source.revision,
        revision_sha=revision_sha,
        gated=bool(info.get("gated")),
        private=bool(info.get("private")),
        license=_license_of(info),
        artifact_type=artifact,
        params_total=_params_of(info),
        tags=[t for t in info.get("tags", []) if isinstance(t, str)],
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
    if report.trust_remote_code_files:
        report.warnings.append(
            "repo มีไฟล์ Python (trust_remote_code) — ต้อง review ก่อน deploy: "
            + ", ".join(report.trust_remote_code_files)
        )

    if artifact in (ArtifactType.SAFETENSORS, ArtifactType.MIXED):
        _inspect_safetensors(report, source, client, revision_sha, safetensor_files)
    if artifact in (ArtifactType.GGUF, ArtifactType.MIXED):
        _inspect_gguf(report, source, client, revision_sha, gguf_files)
    return report


def _sibling_files(info: dict[str, Any]) -> list[tuple[str, int | None, str | None]]:
    out: list[tuple[str, int | None, str | None]] = []
    for sibling in info.get("siblings", []) or []:
        name = sibling.get("rfilename")
        if not name:
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


def _classify(has_safetensors: bool, has_gguf: bool) -> ArtifactType:
    if has_safetensors and has_gguf:
        return ArtifactType.MIXED
    if has_safetensors:
        return ArtifactType.SAFETENSORS
    if has_gguf:
        return ArtifactType.GGUF
    return ArtifactType.UNKNOWN


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


def _params_of(info: dict[str, Any]) -> int | None:
    st = info.get("safetensors") or {}
    total = st.get("total")
    return int(total) if isinstance(total, (int, float)) and total > 0 else None


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


def _inspect_safetensors(
    report: ModelReport,
    source: ModelSource,
    client: HfClient,
    revision: str,
    safetensor_files: list[tuple[str, int | None]],
) -> None:
    report.safetensor_shards = [
        ShardFile(filename=name, size_bytes=size) for name, size in sorted(safetensor_files)
    ]
    sizes = [size for _, size in safetensor_files if size is not None]
    if len(sizes) == len(safetensor_files):
        report.weight_bytes = sum(sizes)
    else:
        report.warnings.append("Hub ไม่รายงานขนาดไฟล์ครบ — weight_bytes อาจไม่ครบถ้วน")
        report.weight_bytes = sum(sizes) if sizes else None

    index = _fetch_json(client, source.repo_id, revision, _SAFETENSORS_INDEX, cap=INDEX_FILE_CAP)
    if index is not None:
        weight_map = index.get("weight_map") or {}
        shards = {v for v in weight_map.values() if isinstance(v, str)}
        report.shard_count = len(shards) or None
        listed = {name for name, _ in safetensor_files}
        missing = sorted(shards - listed)
        if missing:
            report.warnings.append(f"index อ้าง shard ที่ไม่อยู่ใน repo: {', '.join(missing[:5])}")
        total = (index.get("metadata") or {}).get("total_size")
        if isinstance(total, int) and total > 0:
            report.weight_bytes = report.weight_bytes or total
    else:
        report.shard_count = len(safetensor_files)

    config = _fetch_json(client, source.repo_id, revision, "config.json")
    if config is not None:
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            report.architecture = str(architectures[0])
        report.model_type = config.get("model_type") or report.model_type
        # โมเดล multimodal แยก config ของส่วนข้อความไว้ใต้ text_config — ค่า context
        # อยู่ในนั้น ไม่ใช่ระดับบนสุด · มองแค่ชั้นบนแล้วได้ None ซึ่งไม่ error อะไรเลย
        # แต่ทำให้ fit ถอยไปใช้ค่าตั้งต้น และ bundle ออกมาเล็กกว่าที่โมเดลทำได้หลายเท่า
        text_config = config.get("text_config")
        # `source` เป็นพารามิเตอร์ของฟังก์ชันนี้อยู่แล้ว — ตั้งชื่อชนกันเมื่อไหร่
        # การอ่าน config จะไปแทนที่ ModelSource เงียบ ๆ แล้วพังที่บรรทัดถัดไป
        candidates = [config, text_config] if isinstance(text_config, dict) else [config]
        for candidate in candidates:
            for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
                value = candidate.get(key)
                if isinstance(value, int) and value > 0:
                    report.context_length = value
                    break
            if report.context_length:
                break
        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            report.quantization = str(quant.get("quant_method") or quant.get("quant_algo") or "quantized")
        report.kv_dims = _kv_dims_from_config(config)
        report.hybrid_attention = config_is_hybrid(config)
        report.moe_experts, report.moe_experts_active = _moe_from_config(config)
    else:
        report.warnings.append("ไม่พบ config.json — ระบุสถาปัตยกรรมไม่ได้")

    hf_quant = _fetch_json(client, source.repo_id, revision, "hf_quant_config.json")
    if hf_quant is not None and not report.quantization:
        quant_cfg = hf_quant.get("quantization") or {}
        report.quantization = str(quant_cfg.get("quant_algo") or "modelopt")

    tokenizer_config = _fetch_json(client, source.repo_id, revision, "tokenizer_config.json")
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
                    is_mmproj=base.lower().startswith("mmproj"),
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
        full = [n for n, sliding in zip(kv_heads, pattern) if not sliding]
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
    if report.artifact_type is ArtifactType.GGUF:
        report.weight_bytes = selected.size_bytes

    try:
        gguf = parse_gguf(client.range_source(source.repo_id, revision, selected.filename))
    except (GgufParseError, BudgetExceeded, EOFError) as exc:
        report.warnings.append(f"อ่าน GGUF header ไม่สำเร็จ: {exc}")
        return
    report.architecture = report.architecture or gguf.architecture
    report.context_length = report.context_length or gguf.context_length
    report.kv_dims = report.kv_dims or _kv_dims_from_gguf(gguf)
    if isinstance(gguf.metadata, dict) and report.architecture:
        interval = gguf.metadata.get(f"{report.architecture}.full_attention_interval")
        report.hybrid_attention = report.hybrid_attention or (
            isinstance(interval, int) and interval > 1)
    report.moe_experts = report.moe_experts or gguf.expert_count
    report.moe_experts_active = report.moe_experts_active or gguf.expert_used_count
    report.mtp_embedded = report.mtp_embedded or bool(gguf.nextn_layers)

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
