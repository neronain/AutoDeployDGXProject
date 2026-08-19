"""arch แบบ full attention สลับ linear/SSM ต้องถูกตรวจจากไฟล์ ไม่ใช่จากชื่อรุ่น

เคสจริง 2026-08-19: orcarouter/Qwen3.8-27B-Uncensored-NVFP4 มี 64 layer แต่เป็น full
attention แค่ 16 · ทาง GGUF คิด KV ถูก (68 KiB/token) แต่ทาง safetensors คิด 256 KiB
— repo เดียวกันคนละรูปแบบไฟล์ให้คำตอบต่างกัน 4 เท่า แล้วไปบอกว่าที่ context 262,144
รับได้ 1.4 คนพร้อมกัน ทั้งที่ได้ 5.7
"""

from lmds.inspector.inspect import (_hybrid_attention_layers, _kv_dims_from_config,
                                    config_is_hybrid)

QWEN38 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    },
}


def test_counts_only_the_full_attention_layers():
    assert _hybrid_attention_layers(QWEN38["text_config"], 64) == 16


def test_falls_back_to_the_interval_when_there_is_no_layer_list():
    assert _hybrid_attention_layers({"full_attention_interval": 4}, 65) == 17  # ปัดขึ้น


def test_a_plain_model_is_not_treated_as_hybrid():
    assert _hybrid_attention_layers({"layer_types": ["full_attention"] * 32}, 32) is None
    assert _hybrid_attention_layers({}, 32) is None


def test_kv_matches_what_the_gguf_path_computes():
    dims = _kv_dims_from_config(QWEN38)
    assert dims is not None
    per_token = dims.layers * 2 * dims.kv_heads * dims.head_dim * 2
    assert per_token == 64 * 1024, f"ควรได้ 64 KiB/token ไม่ใช่ {per_token / 1024:.0f} KiB"


def test_hybrid_flag_reads_through_text_config():
    """โมเดล multimodal ซ่อนค่าไว้ใน text_config — มองแค่ชั้นบนสุดคือมองไม่เห็น"""
    assert config_is_hybrid(QWEN38) is True
    assert config_is_hybrid({"num_hidden_layers": 32, "layer_types": ["full_attention"] * 32}) is False
