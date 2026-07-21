"""ModelReport — ผลการ inspect ที่เป็นข้อเท็จจริงล้วน (ยังไม่มีการตีความโดย LLM)

ทุก field มาจากไฟล์/ API จริงเท่านั้น — เป็นวัตถุดิบให้ Fit Analyzer (M3) และ Brain (M4)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class GgufVariant(BaseModel):
    filename: str
    size_bytes: Optional[int] = None
    is_mmproj: bool = False


class ModelReport(BaseModel):
    repo_id: str
    revision_requested: Optional[str] = None
    revision_sha: str  # pin จริง — commit SHA ณ เวลา inspect
    gated: bool = False
    private: bool = False
    license: Optional[str] = None
    artifact_type: ArtifactType = ArtifactType.UNKNOWN

    # ตัวเลขสำหรับ Fit Analyzer
    params_total: Optional[int] = None  # จาก Hub safetensors metadata
    weight_bytes: Optional[int] = None  # safetensors: รวมทุก shard / gguf: ไฟล์ที่เลือก
    shard_count: Optional[int] = None

    # จาก config.json / GGUF metadata
    architecture: Optional[str] = None
    model_type: Optional[str] = None
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    has_chat_template: Optional[bool] = None
    trust_remote_code_files: list[str] = Field(default_factory=list)  # configuration_*.py, modeling_*.py

    gguf_variants: list[GgufVariant] = Field(default_factory=list)
    selected_gguf: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    file_count: int = 0
    warnings: list[str] = Field(default_factory=list)
