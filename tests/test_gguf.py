import struct

import pytest

from lmds.inspector import ByteSource, GgufParseError, parse_gguf


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_f32_array(key: str, values: list[float]) -> bytes:
    body = struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", len(values))
    body += b"".join(struct.pack("<f", v) for v in values)
    return _string(key) + body


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(values))
    body += b"".join(_string(v) for v in values)
    return _string(key) + body


def build_gguf(kvs: list[bytes], tensor_count: int = 3) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def test_parse_basic_metadata():
    data = build_gguf([
        _kv_string("general.architecture", "qwen3"),
        _kv_u32("qwen3.context_length", 40960),
        _kv_u32("general.file_type", 15),
        _kv_string("tokenizer.chat_template", "{% for m in messages %}...{% endfor %}"),
    ])
    info = parse_gguf(ByteSource(data))
    assert info.version == 3
    assert info.tensor_count == 3
    assert info.architecture == "qwen3"
    assert info.context_length == 40960
    assert info.file_type == 15
    assert info.chat_template is not None
    assert info.partial is False


def test_long_numeric_array_skipped_not_stored():
    scores = [0.5] * 10_000
    data = build_gguf([
        _kv_f32_array("tokenizer.ggml.scores", scores),
        _kv_string("general.architecture", "llama"),
    ])
    info = parse_gguf(ByteSource(data))
    assert info.metadata["tokenizer.ggml.scores"] == "<array[10000]>"
    assert info.architecture == "llama"  # key หลัง array ยังอ่านได้


def test_long_string_array_skipped():
    vocab = [f"token{i}" for i in range(1000)]
    data = build_gguf([
        _kv_string_array("tokenizer.ggml.tokens", vocab),
        _kv_u32("llama.context_length", 8192),
        _kv_string("general.architecture", "llama"),
    ])
    info = parse_gguf(ByteSource(data))
    assert info.metadata["tokenizer.ggml.tokens"] == "<array[1000]>"
    assert info.context_length == 8192


def test_small_string_array_stored():
    data = build_gguf([_kv_string_array("split.tensors", ["a", "b"])])
    info = parse_gguf(ByteSource(data))
    assert info.metadata["split.tensors"] == ["a", "b"]


def test_bad_magic_raises():
    with pytest.raises(GgufParseError, match="GGUF"):
        parse_gguf(ByteSource(b"NOPE" + b"\x00" * 100))


def test_truncated_file_marks_partial():
    data = build_gguf([
        _kv_string("general.architecture", "llama"),
        _kv_string("general.name", "test"),
    ])
    info = parse_gguf(ByteSource(data[:-10]))  # ตัดท้าย — kv สุดท้ายอ่านไม่ครบ
    assert info.partial is True
    assert info.architecture == "llama"
