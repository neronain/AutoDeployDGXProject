"""Deployment Plan — ขอบเขตการตัดสินใจทั้งหมดของ LLM (PRD §8.2)

กติกา: LLM ห้ามเขียน Bash — มันได้แค่เติมค่าลง schema นี้ แล้วทุกค่าถูก harden ซ้ำ
ด้วยข้อเท็จจริงจาก ModelReport/FitReport ก่อนส่งเข้า template (M5)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from lmds.inspector.report import ArtifactType


class Confidence(str, Enum):
    VERIFIED = "verified"  # อ่านจากไฟล์/API จริง
    INFERRED = "inferred"  # อนุมานจากหลักฐาน
    UNVERIFIED = "unverified"  # ข้อมูล community/model card ที่ยังไม่ยืนยัน


class Fact(BaseModel):
    claim: str
    source: str = ""
    confidence: Confidence


class Engine(str, Enum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


class Topology(str, Enum):
    SINGLE = "single"
    MULTI_GPU = "multi-gpu"
    STACKED = "stacked"


class RuntimeChoice(BaseModel):
    engine: Engine
    image_ref: str  # เช่น vllm/vllm-openai:v0.8.5 หรือ ghcr.io/ggml-org/llama.cpp:server-cuda
    image_pin: Optional[str] = None  # digest/commit — ต้อง pin ก่อน deploy จริง
    rationale: str = ""


class ToolCalling(BaseModel):
    enabled: bool = False
    parser: Optional[str] = None
    chat_template_override: Optional[str] = None
    parallel: bool = False  # ค่าเริ่มต้น false เสมอตาม SKILL.md Phase 6


class Reasoning(BaseModel):
    enabled: bool = False
    parser: Optional[str] = None


class Multimodal(BaseModel):
    modalities: list[str] = Field(default_factory=list)
    projector_files: list[str] = Field(default_factory=list)


class RuntimeAsset(BaseModel):
    """ไฟล์ที่ engine ต้องใช้แต่ **ไม่ได้อยู่ใน repo ของโมเดล**

    เคสจริง: Nemotron-3-Super ต้องมี `super_v3_reasoning_parser.py` วางบน host แล้ว
    bind-mount เข้า container คู่กับ flag `--reasoning-parser`

    ไฟล์พวกนี้เป็นโค้ดที่รันจริงใน container — ต้องผ่าน host allowlist และผู้ใช้ต้อง
    อนุมัติรายตัวเสมอ (เหมือน flag นอก allowlist) ห้าม inject อัตโนมัติเด็ดขาด
    """

    filename: str  # ชื่อไฟล์ปลายทาง (basename เท่านั้น — ห้ามมี path)
    url: str
    sha256: Optional[str] = None  # ถ้ามี controller จะตรวจให้ตอน prepare-runtime
    purpose: str = ""  # อธิบายให้ผู้ใช้ตัดสินใจตอนอนุมัติ


class Serving(BaseModel):
    context: int = Field(gt=0)
    max_output_tokens: int = Field(default=8192, gt=0)
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=0.98)
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = Field(default=4, gt=0, le=256)
    extra_flags: list[str] = Field(default_factory=list)


class DeploymentPlan(BaseModel):
    plan_version: Literal[1] = 1
    model_id: str
    revision: str
    served_model_name: str
    artifact_type: ArtifactType
    selected_gguf: Optional[str] = None

    facts: list[Fact] = Field(default_factory=list)
    runtime: RuntimeChoice
    topology: Topology = Topology.SINGLE
    serving: Serving
    tool_calling: ToolCalling = Field(default_factory=ToolCalling)
    reasoning: Reasoning = Field(default_factory=Reasoning)
    multimodal: Multimodal = Field(default_factory=Multimodal)

    special_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # flag ที่อยู่นอก allowlist — ห้าม inject อัตโนมัติ ผู้ใช้ต้องอนุมัติรายตัว (PRD §9.3)
    flags_needing_approval: list[str] = Field(default_factory=list)
    # ไฟล์ runtime ภายนอกที่ผู้ใช้อนุมัติแล้วเท่านั้น — harden ย้ายทุกตัวไปรออนุมัติก่อนเสมอ
    runtime_assets: list[RuntimeAsset] = Field(default_factory=list)
    assets_needing_approval: list[RuntimeAsset] = Field(default_factory=list)
    generator: str = ""  # "llm:<provider>/<model>" หรือ "rule-based"


class PlanError(Exception):
    """LLM ให้ plan ที่ไม่ผ่าน schema หลัง retry ครบ"""
