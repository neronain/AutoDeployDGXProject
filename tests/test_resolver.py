import pytest

from lmds.resolver import SourceError, UnsupportedSource, parse_source


def test_plain_repo_id():
    src = parse_source("Qwen/Qwen3-32B")
    assert src.kind == "huggingface"
    assert src.repo_id == "Qwen/Qwen3-32B"
    assert src.revision is None and src.filename is None


def test_full_hf_url():
    src = parse_source("https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct")
    assert src.repo_id == "meta-llama/Llama-3.3-70B-Instruct"


def test_hf_url_trailing_slash_and_no_scheme():
    assert parse_source("huggingface.co/Qwen/Qwen3-32B/").repo_id == "Qwen/Qwen3-32B"
    assert parse_source("hf.co/Qwen/Qwen3-32B").repo_id == "Qwen/Qwen3-32B"


def test_tree_url_with_revision():
    src = parse_source("https://huggingface.co/Qwen/Qwen3-32B/tree/abc123")
    assert src.revision == "abc123" and src.filename is None


def test_tree_main_revision_is_none():
    src = parse_source("https://huggingface.co/Qwen/Qwen3-32B/tree/main")
    assert src.revision is None


def test_direct_gguf_file_link():
    src = parse_source(
        "https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-Q4_K_M.gguf"
    )
    assert src.repo_id == "unsloth/Qwen3-32B-GGUF"
    assert src.filename == "Qwen3-32B-Q4_K_M.gguf"


def test_resolve_link_with_subdir():
    src = parse_source(
        "https://huggingface.co/org/repo/resolve/v1.0/sub/dir/model-Q8_0.gguf"
    )
    assert src.revision == "v1.0"
    assert src.filename == "sub/dir/model-Q8_0.gguf"


def test_ollama_link_resolves_to_registry_ref():
    """เดิมเคยตอบว่ายังไม่รองรับ — ตอนนี้ resolve เป็น ref ของ registry (เทสละเอียดใน test_ollama.py)"""
    src = parse_source("https://ollama.com/library/qwen3:32b")
    assert src.kind == "ollama"
    assert src.repo_id == "library/qwen3"
    assert src.revision == "32b"


@pytest.mark.parametrize("value", [
    "ollama.com/bad%20name",
    "ollama.com/library/name:bad%2Ftag",
    "registry.ollama.ai/v2/bad.name/qwen3/manifests/latest",
    "ollama.com/" + "a" * 81,
])
def test_invalid_ollama_name_parts_fail_before_network(value):
    with pytest.raises(SourceError, match="Ollama"):
        parse_source(value)


def test_ngc_rejected_with_clear_message():
    with pytest.raises(UnsupportedSource, match="NGC"):
        parse_source("https://catalog.ngc.nvidia.com/models/foo")


def test_dataset_link_rejected():
    with pytest.raises(SourceError, match="model"):
        parse_source("https://huggingface.co/datasets/foo/bar")


def test_garbage_rejected():
    with pytest.raises(SourceError):
        parse_source("not a model at all")
