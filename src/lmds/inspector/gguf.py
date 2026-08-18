"""อ่าน GGUF header/metadata จาก source แบบ read(n)/skip(n) — ไม่โหลด tensor data

สเปก GGUF v3: magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64 | kv pairs
kv: key(string) | value_type u32 | value — array ตัวเลขยาว ๆ (vocab scores ฯลฯ) ใช้ skip ไม่ดาวน์โหลด
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Protocol

from .hf_api import BudgetExceeded

GGUF_MAGIC = b"GGUF"

# value type → struct format (เฉพาะ scalar ขนาดคงที่)
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# เก็บ string array ทั้งชุดเมื่อสั้นพอเท่านั้น (เช่น รายชื่อ expert) — vocab ยาว ๆ เก็บแค่สรุป
_MAX_STORED_ARRAY = 64
_MAX_STRING_BYTES = 1024 * 1024  # chat template ยาวสุดที่ยอมเก็บ


class ByteSourceProtocol(Protocol):
    def read(self, n: int) -> bytes: ...
    def skip(self, n: int) -> None: ...


class ByteSource:
    """source จาก bytes ในหน่วยความจำ — ใช้กับไฟล์ local และในเทส"""

    def __init__(self, data: bytes):
        self._data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self._data):
            raise EOFError("ปลายไฟล์ก่อนอ่านครบ")
        out = self._data[self.pos : self.pos + n]
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self.pos += n


class GgufParseError(Exception):
    pass


@dataclass
class GgufInfo:
    version: int
    tensor_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    partial: bool = False  # True = อ่าน metadata ไม่ครบ (ชน budget)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def context_length(self) -> int | None:
        arch = self.architecture
        if arch:
            value = self.metadata.get(f"{arch}.context_length")
            if isinstance(value, int):
                return value
        return None

    @property
    def expert_count(self) -> int | None:
        arch = self.architecture
        value = self.metadata.get(f"{arch}.expert_count") if arch else None
        return value if isinstance(value, int) and value > 0 else None

    @property
    def expert_used_count(self) -> int | None:
        arch = self.architecture
        value = self.metadata.get(f"{arch}.expert_used_count") if arch else None
        return value if isinstance(value, int) and value > 0 else None

    @property
    def nextn_layers(self) -> int | None:
        """จำนวนชั้น NextN/MTP ที่ฝังมาในไฟล์ — >0 = เปิด speculative ได้โดยไม่ต้องมี draft แยก"""
        arch = self.architecture
        value = self.metadata.get(f"{arch}.nextn_predict_layers") if arch else None
        return value if isinstance(value, int) and value > 0 else None

    @property
    def chat_template(self) -> str | None:
        return self.metadata.get("tokenizer.chat_template")

    @property
    def file_type(self) -> int | None:
        value = self.metadata.get("general.file_type")
        return value if isinstance(value, int) else None


def _unpack(src: ByteSourceProtocol, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, src.read(size))[0]


def _read_string(src: ByteSourceProtocol, cap: int = _MAX_STRING_BYTES) -> str | None:
    """คืน None ถ้า string ยาวเกิน cap (ข้ามไปแล้ว ไม่เก็บ)"""
    length = _unpack(src, "<Q")
    if length > cap:
        src.skip(length)
        return None
    return src.read(length).decode("utf-8", errors="replace")


def _read_value(src: ByteSourceProtocol, value_type: int) -> Any:
    fmt = _SCALAR_FMT.get(value_type)
    if fmt is not None:
        return _unpack(src, fmt)
    if value_type == _TYPE_STRING:
        return _read_string(src)
    if value_type == _TYPE_ARRAY:
        elem_type = _unpack(src, "<I")
        count = _unpack(src, "<Q")
        elem_fmt = _SCALAR_FMT.get(elem_type)
        if elem_fmt is not None:
            if count <= _MAX_STORED_ARRAY:
                return [_unpack(src, elem_fmt) for _ in range(count)]
            src.skip(struct.calcsize(elem_fmt) * count)  # ไม่ดาวน์โหลด array ตัวเลขยาว
            return f"<array[{count}]>"
        if elem_type == _TYPE_STRING:
            if count <= _MAX_STORED_ARRAY:
                return [_read_string(src, cap=64 * 1024) for _ in range(count)]
            for _ in range(count):  # string array ต้องไล่อ่านความยาวทีละตัวเพื่อข้าม
                length = _unpack(src, "<Q")
                src.skip(length)
            return f"<array[{count}]>"
        raise GgufParseError(f"array element type ไม่รู้จัก: {elem_type}")
    raise GgufParseError(f"value type ไม่รู้จัก: {value_type}")


def parse_gguf(src: ByteSourceProtocol) -> GgufInfo:
    magic = src.read(4)
    if magic != GGUF_MAGIC:
        raise GgufParseError(f"ไม่ใช่ไฟล์ GGUF (magic: {magic!r})")
    version = _unpack(src, "<I")
    if version < 2:
        raise GgufParseError(f"GGUF version {version} เก่าเกินไป (รองรับ v2+)")
    tensor_count = _unpack(src, "<Q")
    kv_count = _unpack(src, "<Q")
    if kv_count > 100_000:
        raise GgufParseError(f"metadata_kv_count ผิดปกติ: {kv_count}")

    info = GgufInfo(version=version, tensor_count=tensor_count)
    try:
        for _ in range(kv_count):
            key = _read_string(src, cap=64 * 1024)
            value_type = _unpack(src, "<I")
            value = _read_value(src, value_type)
            if key is not None and value is not None:
                info.metadata[key] = value
    except (BudgetExceeded, EOFError):
        info.partial = True
    return info
