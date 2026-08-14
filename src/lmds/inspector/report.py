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


class GgufPart(BaseModel):
    filename: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class GgufVariant(BaseModel):
    filename: str  # ไฟล์เดียว หรือ part แรก (-00001-of-N) ของ split GGUF
    size_bytes: Optional[int] = None  # split: ขนาดรวมทุก part
    sha256: Optional[str] = None  # จาก lfs.oid ของ Hub — ใช้ทำ exact hash check ใน controller
    is_mmproj: bool = False
    parts: list[GgufPart] = []  # ว่าง = ไฟล์เดียว; split = ทุก part เรียงลำดับ

    @property
    def all_parts(self) -> list[GgufPart]:
        if self.parts:
            return self.parts
        return [GgufPart(filename=self.filename, size_bytes=self.size_bytes, sha256=self.sha256)]


class KvDims(BaseModel):
    """มิติสำหรับคำนวณ KV cache — จาก config.json หรือ GGUF metadata เท่านั้น (ไม่เดา)"""

    layers: int
    kv_heads: int
    head_dim: int
    # MLA (DeepSeek-V2/V3, Kimi K2/K3): เก็บ latent ก้อนเดียวต่อ token ต่อ layer
    # ไม่ใช่ K กับ V แยกกันตาม head · ตั้งค่านี้เมื่อรู้ขนาด latent แล้วสูตรจะเปลี่ยนไปเลย
    latent_dim: Optional[int] = None

    @property
    def elements_per_token(self) -> int:
        """จำนวนค่าที่ต้องเก็บต่อ token — รูปทรงของ KV ตัดสินที่นี่ที่เดียว"""
        if self.latent_dim:
            return self.layers * self.latent_dim
        # K + V ต่อ layer ต่อ token: 2 × kv_heads × head_dim
        return 2 * self.layers * self.kv_heads * self.head_dim

    @property
    def bytes_per_token_fp16(self) -> int:
        return self.elements_per_token * 2  # fp16/bf16 = 2 ไบต์ต่อค่า


class ShardFile(BaseModel):
    filename: str
    size_bytes: Optional[int] = None  # None = Hub ไม่รายงานขนาด → ข้ามการเทียบขนาด


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
    # ไฟล์ .safetensors ทุก shard พร้อมขนาดจาก Hub — ใช้ให้ controller ตรวจ download ครบจริง
    safetensor_shards: list["ShardFile"] = Field(default_factory=list)
    tokenizer_files: list[str] = Field(default_factory=list)

    # จาก config.json / GGUF metadata
    architecture: Optional[str] = None
    model_type: Optional[str] = None
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    kv_dims: Optional[KvDims] = None
    has_chat_template: Optional[bool] = None
    # ความสามารถที่อ่านได้จากไฟล์ ก่อน deploy — ดู inspector/capabilities.py ว่าอะไร
    # ตอบได้จริงและอะไรต้องรอวัดตอนรัน
    capabilities: dict = Field(default_factory=dict)
    trust_remote_code_files: list[str] = Field(default_factory=list)  # configuration_*.py, modeling_*.py

    gguf_variants: list[GgufVariant] = Field(default_factory=list)
    selected_gguf: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    file_count: int = 0
    warnings: list[str] = Field(default_factory=list)
