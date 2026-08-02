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
        sha = lfs.get("oid") if isinstance(lfs.get("oid"), str) else None
        out.append((name, size if size is not None else lfs.get("size"), sha))
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
        for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                report.context_length = value
                break
        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            report.quantization = str(quant.get("quant_method") or quant.get("quant_algo") or "quantized")
        report.kv_dims = _kv_dims_from_config(config)
    else:
        report.warnings.append("ไม่พบ config.json — ระบุสถาปัตยกรรมไม่ได้")

    hf_quant = _fetch_json(client, source.repo_id, revision, "hf_quant_config.json")
    if hf_quant is not None and not report.quantization:
        quant_cfg = hf_quant.get("quantization") or {}
        report.quantization = str(quant_cfg.get("quant_algo") or "modelopt")

    tokenizer_config = _fetch_json(client, source.repo_id, revision, "tokenizer_config.json")
    if tokenizer_config is not None and tokenizer_config.get("chat_template"):
        report.has_chat_template = True
    else:
        template = client.fetch_small_file(source.repo_id, revision, "chat_template.jinja")
        report.has_chat_template = template is not None if tokenizer_config is not None else None


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
    if all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
        return KvDims(layers=layers, kv_heads=kv_heads, head_dim=head_dim)
    return None


def _inspect_gguf(
    report: ModelReport,
    source: ModelSource,
    client: HfClient,
    revision: str,
    gguf_files: list[tuple[str, int | None, str | None]],
) -> None:
    report.gguf_variants = _group_gguf_variants(gguf_files)
    weight_variants = [v for v in report.gguf_variants if not v.is_mmproj]
    if not weight_variants:
        report.warnings.append("พบเฉพาะไฟล์ mmproj — ไม่มี GGUF ของตัวโมเดล")
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
    if gguf.partial:
        report.warnings.append("GGUF metadata อ่านได้บางส่วน (ชน budget) — ข้อมูลอาจไม่ครบ")
