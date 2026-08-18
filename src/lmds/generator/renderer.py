"""Renderer — แปลง DeploymentPlan + ModelReport + FitReport → deployment bundle

LLM ไม่มีสิทธิ์แตะขั้นนี้: ทุกไฟล์ render จาก template ที่ผ่านการตรวจแล้วเท่านั้น (PRD §8)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import lmds
from lmds.brain.plan_schema import DeploymentPlan, Engine, Topology
from lmds.fit import FitReport
from lmds.fit.targets import PRESETS
from lmds.inspector.report import ModelReport

TEMPLATES_DIR = Path(__file__).parent / "templates"

# ไฟล์เล็กที่ vLLM controller ต้อง verify ว่ามีจริงใน snapshot
BASE_REQUIRED_FILES = ["config.json"]


@dataclass
class Bundle:
    directory: Path
    controller: Path
    files: list[Path] = field(default_factory=list)


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    return env


def _client_input(plan: DeploymentPlan) -> int:
    return max(plan.serving.context - plan.serving.max_output_tokens - 2048, 0)


def _quote_flag(flag: str) -> str:
    """'--kv-cache-dtype=fp8' → '--kv-cache-dtype=fp8' (quoted ปลอดภัยสำหรับ bash array)"""
    parts = flag.replace("=", " ", 1).split(None, 1)
    if len(parts) == 1:
        return shlex.quote(parts[0])
    return f"{shlex.quote(parts[0])} {shlex.quote(parts[1])}"


def _context(plan: DeploymentPlan, report: ModelReport, fit: FitReport) -> dict:
    from lmds.brain.rulebased import slugify

    slug = slugify(plan.model_id)
    is_gguf = plan.runtime.engine is Engine.LLAMACPP

    gguf_size = None
    gguf_sha = None
    gguf_parts: list[dict] = []
    if plan.selected_gguf:
        selected_variant = next(
            (
                v for v in report.gguf_variants
                if v.filename == plan.selected_gguf
                or any(p.filename == plan.selected_gguf for p in v.parts)
            ),
            None,
        )
        if selected_variant is not None:
            gguf_size = selected_variant.size_bytes
            gguf_sha = selected_variant.sha256
            gguf_parts = [
                {
                    "filename": part.filename,
                    "basename": part.filename.rsplit("/", 1)[-1],
                    "size_bytes": part.size_bytes,
                    "sha256": part.sha256,
                }
                for part in selected_variant.all_parts
            ]
        else:
            gguf_parts = [
                {
                    "filename": plan.selected_gguf,
                    "basename": plan.selected_gguf.rsplit("/", 1)[-1],
                    "size_bytes": None,
                    "sha256": None,
                }
            ]

    # mmproj (multimodal projector) ของ llama.cpp อยู่คนละไฟล์กับ weight และ **ไม่ได้อยู่ใน
    # variant ที่ผู้ใช้เลือก** — ถ้าไม่ผนวกเข้า MODEL_FILES ตรงนี้ controller จะไม่โหลด ไม่ verify
    # และไม่ส่ง --mmproj ทำให้โมเดล multimodal กลายเป็น text-only เงียบ ๆ (เจอจริงกับ gemma-4-12b-it)
    mmproj_parts: list[dict] = []
    if is_gguf:
        for name in plan.multimodal.projector_files:
            base = name.rsplit("/", 1)[-1]
            variant = next(
                (v for v in report.gguf_variants if v.is_mmproj and v.filename.rsplit("/", 1)[-1] == base),
                None,
            )
            mmproj_parts.append(
                {
                    "filename": variant.filename if variant is not None else name,
                    "basename": base,
                    "size_bytes": variant.size_bytes if variant is not None else None,
                    "sha256": variant.sha256 if variant is not None else None,
                }
            )
    # ต่อท้ายเสมอ — MODEL_FILE ของ controller คือ MODEL_FILES[0] ซึ่งต้องเป็น weight ไม่ใช่ mmproj
    gguf_parts = gguf_parts + mmproj_parts
    mmproj_basename = mmproj_parts[0]["basename"] if mmproj_parts else ""

    # MTP draft head — เหตุผลเดียวกับ mmproj: อยู่คนละไฟล์กับ weight ไม่ได้อยู่ใน variant ที่เลือก
    # ไม่ผนวกตรงนี้ = ไม่โหลด ไม่ verify แล้ว speculative decoding เงียบหายไปทั้งที่ repo ทำมาให้
    mtp_parts: list[dict] = []
    if is_gguf:
        for name in plan.speculative.draft_files:
            base = name.rsplit("/", 1)[-1]
            variant = next(
                (v for v in report.gguf_variants if v.is_mtp and v.filename.rsplit("/", 1)[-1] == base),
                None,
            )
            mtp_parts.append(
                {
                    "filename": variant.filename if variant is not None else name,
                    "basename": base,
                    "size_bytes": variant.size_bytes if variant is not None else None,
                    "sha256": variant.sha256 if variant is not None else None,
                }
            )
    gguf_parts = gguf_parts + mtp_parts
    mtp_basename = mtp_parts[0]["basename"] if mtp_parts else ""

    required = list(BASE_REQUIRED_FILES)
    if report.shard_count and report.shard_count > 1:
        required.append("model.safetensors.index.json")
    required += list(report.tokenizer_files)
    required += [f for f in plan.special_files if "/" not in f]

    is_stacked = plan.topology.value == "stacked"
    # จำนวนเครื่องมาจาก target preset ไม่ใช่ค่าคงที่ — dgx-spark-stacked-4 = 4 เครื่อง
    # preset ที่ไม่รู้จัก (target กำหนดเอง) ถอยไปที่ 2 ซึ่งเป็นรูปแบบ stacked ที่ทดสอบแล้ว
    spec = PRESETS.get(fit.target_name)
    node_count = (spec.node_count if spec else 2) if is_stacked else 1
    tensor_parallel = 1
    if plan.topology.value == "multi-gpu":
        tensor_parallel = 2  # ค่าตั้งต้น dual-GPU — override ได้ผ่าน env ใน controller
    elif is_stacked:
        # 1 GPU ต่อเครื่องบน DGX Spark → TP = จำนวนเครื่อง
        tensor_parallel = (spec.gpu_count if spec else 2)

    weights_gb = fit.weights_gb or (report.weight_bytes or 0) / 1024**3
    disk_gb = int(weights_gb * 1.2 + 20)  # โมเดล + image + เผื่อ

    # health timeout สเกลตามขนาดโมเดลจริง (~30s/GB สำหรับ cold load + ฐาน 300s), ขั้นต่ำ 600 เพดาน 7200
    health_timeout = min(max(600, int(weights_gb * 30) + 300), 7200)

    # offload: ถ้า fit บอก fits-with-offload ให้เริ่มที่ค่ากลาง ปรับเองตาม log ได้
    n_gpu_layers = 999
    if fit.verdict.value == "fits-with-offload":
        n_gpu_layers = 32

    # ── metadata สำหรับ banner()/info() ตาม controller contract v3.0.0 ──
    # audit-controllers.py บังคับ SCRIPT_VERSION เป็น X.Y.Z เป๊ะ — ตัด suffix ของ dev build ออก
    version_match = re.match(r"\d+\.\d+\.\d+", lmds.__version__)
    features: list[str] = []
    if plan.tool_calling.enabled:
        features.append("tool-calling")
    if plan.reasoning.enabled:
        features.append("reasoning")
    if plan.multimodal.modalities:
        features.append("+".join(plan.multimodal.modalities))
    engine_name = {
        Engine.VLLM: "vLLM",
        Engine.SGLANG: "SGLang",
        Engine.LLAMACPP: "llama.cpp",
    }[plan.runtime.engine]
    native_build = is_gguf and fit.memory_model.value == "unified"

    return {
        "plan": plan,
        "report": report,
        "fit": fit,
        "slug": slug,
        "lmds_version": lmds.__version__,
        "controller_version": version_match.group(0) if version_match else "0.0.0",
        "model_label": plan.model_id,
        "runtime_label": f"{engine_name} ({'native build' if native_build else 'Docker'})",
        "model_features": ", ".join(features) or "text",
        "controller_name": f"{slug}-stacked.sh" if is_stacked else f"{slug}-single.sh",
        "is_stacked": is_stacked,
        "node_count": node_count,
        "shard_count": report.shard_count or 0,
        "total_size_gb": int(round(weights_gb)) if weights_gb else 0,
        "required_files": " ".join(shlex.quote(f) for f in dict.fromkeys(required)),
        # shard + ขนาดจาก Hub — ให้ verify-files จับ download ที่ไม่ครบได้เหมือนฝั่ง GGUF
        "shard_files": [
            {"filename": s.filename, "size": s.size_bytes or ""}
            for s in report.safetensor_shards
        ],
        # quote ทั้งก้อน KEY=VALUE — _quote_flag แยกช่องว่างเป็นคนละ token ซึ่งผิดสำหรับ env
        "extra_env_pairs": [
            shlex.quote(f"{key}={value}") for key, value in (plan.serving.extra_env or {}).items()
        ],
        "runtime_assets": [
            {"filename": a.filename, "url": a.url, "sha256": a.sha256 or ""}
            for a in plan.runtime_assets
        ],
        # --mmproj ถูกตัดออกจาก extra_flags เสมอ: path ของไฟล์เป็นของ controller (MODEL_DIR ต่างกัน
        # ระหว่าง native/docker) ค่าที่ LLM เดามาจะชี้ผิดที่ ส่วนตัวจริง emit จาก mmproj_basename
        "extra_flag_pairs": [
            _quote_flag(f)
            for f in plan.serving.extra_flags
            if not f.startswith(("--mmproj", "--spec-draft-model", "--spec-type"))
        ],
        "tensor_parallel": tensor_parallel,
        "gguf_basename": (plan.selected_gguf or "").rsplit("/", 1)[-1],
        "gguf_size": gguf_size,
        "gguf_sha256": gguf_sha,
        "gguf_parts": gguf_parts,
        "mmproj_basename": mmproj_basename,
        "mtp_basename": mtp_basename,
        # llama.cpp บน DGX Spark (unified/ARM64) ไม่มี docker image ทางการ — ใช้ native source build
        "runtime_mode": "native" if fit.memory_model.value == "unified" else "docker",
        "cuda_architectures": "121a-real" if fit.memory_model.value == "unified" else "native",
        "native_llamacpp": is_gguf and fit.memory_model.value == "unified",
        "has_chat_template": bool(report.has_chat_template),
        "n_gpu_layers": n_gpu_layers,
        "client_input": _client_input(plan),
        "context_env": "MAX_MODEL_LEN" if not is_gguf else "CTX_SIZE",
        "disk_gb": disk_gb,
        "health_timeout": health_timeout,
        "validation_status": "static-validated",
        "hardware_validated": False,
    }


def _model_profile_yaml(plan: DeploymentPlan, report: ModelReport, fit: FitReport) -> str:
    """MODEL_PROFILE.yaml — source of truth ภายใน bundle (ตาม template v3.0.0)"""
    profile = {
        "profile_version": 1,
        "generated_by": f"lmds {lmds.__version__}",
        "generator": plan.generator,
        "model": {
            "id": plan.model_id,
            "revision": plan.revision,
            "served_name": plan.served_model_name,
            "artifact_type": plan.artifact_type.value,
            "selected_gguf": plan.selected_gguf,
            # lmds doctor ใช้ตรวจว่าต้องมี HF_TOKEN ตอน download ไหม
            "gated": report.gated,
            "license": report.license,
            "architecture": report.architecture,
            "params_total": report.params_total,
            "weight_bytes": report.weight_bytes,
            "native_context": report.context_length,
        },
        "runtime": {
            "engine": plan.runtime.engine.value,
            "image": plan.runtime.image_ref,
            "image_pin": plan.runtime.image_pin,
        },
        "topology": plan.topology.value,
        "target": {
            "name": fit.target_name,
            "memory_model": fit.memory_model.value,
            "budget_gb": fit.budget_gb,
            "verdict": fit.verdict.value,
            # เพดานจริงของเครื่อง — ต่างจาก serving.context ที่เป็นค่าเริ่มต้นมาตรฐาน
            "max_safe_context": fit.max_safe_context,
            "llamacpp_dir": plan.runtime.native_dir,
            "fit_notes": fit.notes,
        },
        "serving": {
            "context": plan.serving.context,
            "max_output_tokens": plan.serving.max_output_tokens,
            "gpu_memory_utilization": plan.serving.gpu_memory_utilization,
            "kv_cache_dtype": plan.serving.kv_cache_dtype,
            "max_num_seqs": plan.serving.max_num_seqs,
            "extra_flags": plan.serving.extra_flags,
            "extra_env": plan.serving.extra_env,
        },
        "features": {
            "tool_calling": plan.tool_calling.model_dump(mode="json"),
            "reasoning": plan.reasoning.model_dump(mode="json"),
            "multimodal": plan.multimodal.model_dump(mode="json"),
        },
        "facts": [f.model_dump(mode="json") for f in plan.facts],
        "warnings": plan.warnings,
        "flags_needing_approval": plan.flags_needing_approval,
        "validation": {"static": True, "hardware": False},
    }
    return yaml.safe_dump(profile, allow_unicode=True, sort_keys=False)


def render_bundle(
    plan: DeploymentPlan,
    report: ModelReport,
    fit: FitReport,
    output_root: Path,
) -> Bundle:
    from lmds.brain.rulebased import slugify

    if plan.runtime.engine is Engine.LLAMACPP and not plan.selected_gguf:
        raise ValueError("llama.cpp bundle ต้องเลือกไฟล์ GGUF ก่อน (ใช้ลิงก์ไฟล์ตรง หรือระบุ variant)")

    is_stacked = plan.topology is Topology.STACKED
    if is_stacked and plan.runtime.engine is Engine.LLAMACPP:
        raise ValueError(
            "topology stacked (multi-node) รองรับเฉพาะ vLLM — GGUF/llama.cpp ยังไม่มี reference ที่ผ่านการทดสอบ "
            "(ใช้ single/multi-gpu กับ GGUF แทน)"
        )

    env = _environment()
    context = _context(plan, report, fit)
    slug = context["slug"]

    directory = output_root / slug
    directory.mkdir(parents=True, exist_ok=True)

    if is_stacked:
        template_name = "stacked-vllm-controller.sh.j2"
    elif plan.runtime.engine is Engine.LLAMACPP:
        template_name = "single-llamacpp-controller.sh.j2"
    elif plan.runtime.engine is Engine.SGLANG:
        template_name = "single-sglang-controller.sh.j2"
    else:
        template_name = "single-vllm-controller.sh.j2"
    controller_path = directory / context["controller_name"]
    controller_path.write_text(env.get_template(template_name).render(context), encoding="utf-8")
    controller_path.chmod(0o755)

    files = [controller_path]

    readme_path = directory / "README.md"
    readme_path.write_text(env.get_template("README.md.j2").render(context), encoding="utf-8")
    files.append(readme_path)

    profile_path = directory / "MODEL_PROFILE.yaml"
    profile_path.write_text(_model_profile_yaml(plan, report, fit), encoding="utf-8")
    files.append(profile_path)

    needs_special = (
        report.trust_remote_code_files
        or plan.selected_gguf
        or plan.special_files
        or plan.multimodal.projector_files
    )
    if needs_special:
        special_path = directory / "SPECIAL_FILES.md"
        special_path.write_text(env.get_template("SPECIAL_FILES.md.j2").render(context), encoding="utf-8")
        files.append(special_path)

    return Bundle(directory=directory, controller=controller_path, files=files)
