"""KV dims จาก GGUF ของโมเดล sliding-window

เคสจริง 2026-08-13: gemma-4-31B บน dgx-veerasiam รันอยู่ที่ context 16,384
ทั้งที่ GGUF บอก context_length = 262,144 และเครื่องมีหน่วยความจำเหลือพอ

สาเหตุ: gemma-4 เขียน head_count_kv เป็นลิสต์ต่อ layer (แต่ละ layer ใช้ไม่เท่ากัน)
ตัว parser เช็ค isinstance(int) แล้วคืน None → analyser เข้าสาขา "ไม่รู้มิติ KV"
ซึ่งตั้ง context ไว้แค่ 16,384 เสีย context ไป 16 เท่าโดยไม่มีอะไรร้อง
"""

from lmds.inspector.inspect import _kv_dims_from_gguf


class FakeGguf:
    def __init__(self, metadata, architecture="gemma4"):
        self.metadata = metadata
        self.architecture = architecture


# ค่าจริงที่อ่านจาก gemma-4-31B-it-UD-Q8_K_XL.gguf บน dgx-veerasiam
GEMMA4 = {
    "gemma4.block_count": 60,
    "gemma4.attention.head_count": 32,
    "gemma4.attention.head_count_kv": [16, 16, 16, 16, 16, 4] * 10,
    "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False] * 10,
    "gemma4.attention.key_length": 512,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.embedding_length": 5376,
}


def test_per_layer_kv_heads_no_longer_returns_none():
    """ลิสต์ต่อ layer ต้องอ่านได้ ไม่ใช่ยอมแพ้แล้วคืน None"""
    assert _kv_dims_from_gguf(FakeGguf(GEMMA4)) is not None


def test_only_full_attention_layers_scale_with_context():
    """layer ที่เป็น sliding-window ใช้ KV คงที่ ไม่ควรถูกนับรวมเป็น bytes/token"""
    dims = _kv_dims_from_gguf(FakeGguf(GEMMA4))
    assert dims.layers == 10, "60 layer แต่ full-attention มีแค่ 10 (ทุก ๆ ตัวที่ 6)"
    assert dims.kv_heads == 4
    assert dims.head_dim == 512


def test_bytes_per_token_matches_what_the_machine_really_allocated():
    """ยึดกับของจริง: วัดบน dgx-veerasiam ได้ ~80 KiB/token

    gemma ที่ 16,384 ใช้ KV+buffer 2.83 GB และตอนขึ้นเป็น 262,144 โตอีก 18.9 GiB
    → 18.9 GiB / (262144-16384) token ≈ 80 KiB/token
    """
    dims = _kv_dims_from_gguf(FakeGguf(GEMMA4))
    assert dims.bytes_per_token_fp16 == 81_920


def test_a_scalar_head_count_kv_still_works():
    """โมเดลปกติ (ไม่ sliding) ต้องไม่เปลี่ยนพฤติกรรม"""
    dims = _kv_dims_from_gguf(FakeGguf({
        "qwen3moe.block_count": 48,
        "qwen3moe.attention.head_count": 32,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.attention.key_length": 128,
    }, architecture="qwen3moe"))
    assert (dims.layers, dims.kv_heads, dims.head_dim) == (48, 4, 128)


def test_a_list_without_a_pattern_is_conservative():
    """ไม่รู้ว่า layer ไหน sliding → ใช้ค่ามากสุดกับทุก layer ประเมินเกินดีกว่า OOM"""
    dims = _kv_dims_from_gguf(FakeGguf({
        "mystery.block_count": 12,
        "mystery.attention.head_count": 32,
        "mystery.attention.head_count_kv": [8, 8, 2, 8, 8, 2, 8, 8, 2, 8, 8, 2],
        "mystery.attention.key_length": 128,
    }, architecture="mystery"))
    assert (dims.layers, dims.kv_heads) == (12, 8)
