"""Renderer — แปลง DeploymentPlan + ModelReport + FitReport → deployment bundle

LLM ไม่มีสิทธิ์แตะขั้นนี้: ทุกไฟล์ render จาก template ที่ผ่านการตรวจแล้วเท่านั้น (PRD §8)
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import lmds
from lmds.brain.plan_schema import DeploymentPlan, Engine
from lmds.fit import FitReport
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
    if plan.selected_gguf:
        for variant in report.gguf_variants:
            if variant.filename == plan.selected_gguf:
                gguf_size = variant.size_bytes
                gguf_sha = variant.sha256
                break

    required = list(BASE_REQUIRED_FILES)
    if report.shard_count and report.shard_count > 1:
        required.append("model.safetensors.index.json")
    required += [f for f in plan.special_files if "/" not in f]

    tensor_parallel = 1
    if plan.topology.value == "multi-gpu":
        tensor_parallel = 2  # ค่าตั้งต้น dual-GPU — override ได้ผ่าน env ใน controller

    weights_gb = fit.weights_gb or (report.weight_bytes or 0) / 1024**3
    disk_gb = int(weights_gb * 1.2 + 20)  # โมเดล + image + เผื่อ

    # offload: ถ้า fit บอก fits-with-offload ให้เริ่มที่ค่ากลาง ปรับเองตาม log ได้
    n_gpu_layers = 999
    if fit.verdict.value == "fits-with-offload":
        n_gpu_layers = 32

    return {
        "plan": plan,
        "report": report,
        "fit": fit,
        "slug": slug,
        "lmds_version": lmds.__version__,
        "controller_name": f"{slug}-single.sh",
        "required_files": " ".join(shlex.quote(f) for f in required),
        "extra_flag_pairs": [_quote_flag(f) for f in plan.serving.extra_flags],
        "tensor_parallel": tensor_parallel,
        "gguf_basename": (plan.selected_gguf or "").rsplit("/", 1)[-1],
        "gguf_size": gguf_size,
        "gguf_sha256": gguf_sha,
        "has_chat_template": bool(report.has_chat_template),
        "n_gpu_layers": n_gpu_layers,
        "client_input": _client_input(plan),
        "context_env": "MAX_MODEL_LEN" if not is_gguf else "CTX_SIZE",
        "disk_gb": disk_gb,
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
        },
        "serving": {
            "context": plan.serving.context,
            "max_output_tokens": plan.serving.max_output_tokens,
            "gpu_memory_utilization": plan.serving.gpu_memory_utilization,
            "kv_cache_dtype": plan.serving.kv_cache_dtype,
            "max_num_seqs": plan.serving.max_num_seqs,
            "extra_flags": plan.serving.extra_flags,
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

    env = _environment()
    context = _context(plan, report, fit)
    slug = context["slug"]

    directory = output_root / slug
    directory.mkdir(parents=True, exist_ok=True)

    template_name = (
        "single-llamacpp-controller.sh.j2"
        if plan.runtime.engine is Engine.LLAMACPP
        else "single-vllm-controller.sh.j2"
    )
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
