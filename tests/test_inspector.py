import json
import struct

import httpx
import pytest

from lmds.inspector import ArtifactType, AuthRequired, HfClient, RepoNotFound, inspect_model
from lmds.resolver import parse_source
from tests.test_gguf import _kv_string, _kv_u32, build_gguf

SHA = "abc123def456"


def make_client(handler) -> HfClient:
    transport = httpx.MockTransport(handler)
    return HfClient(client=httpx.Client(transport=transport))


def hub_response(siblings, gated=False, card=None, safetensors=None, tags=None):
    return {
        "sha": SHA,
        "gated": gated,
        "private": False,
        "siblings": siblings,
        "cardData": card or {"license": "apache-2.0"},
        "tags": tags or [],
        **({"safetensors": safetensors} if safetensors else {}),
    }


def test_inspect_safetensors_repo():
    index = {
        "metadata": {"total_size": 65_000_000_000},
        "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
    }
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "max_position_embeddings": 40960}
    files = {
        "model.safetensors.index.json": json.dumps(index),
        "config.json": json.dumps(config),
        "tokenizer_config.json": json.dumps({"chat_template": "{{...}}"}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response(
                [
                    {"rfilename": "model-00001-of-00002.safetensors", "lfs": {"size": 32_500_000_000}},
                    {"rfilename": "model-00002-of-00002.safetensors", "lfs": {"size": 32_500_000_000}},
                    {"rfilename": "config.json", "size": 700},
                ],
                safetensors={"total": 32_800_000_000},
            ))
        name = request.url.path.split(f"/{SHA}/")[-1]
        if name in files:
            return httpx.Response(200, content=files[name].encode())
        return httpx.Response(404)

    report = inspect_model(parse_source("Qwen/Qwen3-32B"), make_client(handler))
    assert report.revision_sha == SHA
    assert report.artifact_type is ArtifactType.SAFETENSORS
    assert report.weight_bytes == 65_000_000_000
    assert report.shard_count == 2
    assert report.architecture == "Qwen3ForCausalLM"
    assert report.context_length == 40960
    assert report.params_total == 32_800_000_000
    assert report.license == "apache-2.0"
    assert report.has_chat_template is True
    assert report.warnings == []


def test_inspect_gguf_repo_single_variant():
    gguf_bytes = build_gguf([
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.context_length", 131072),
        _kv_u32("general.file_type", 15),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response(
                [{"rfilename": "model-Q4_K_M.gguf", "lfs": {"size": 19_800_000_000}}]
            ))
        range_header = request.headers.get("Range", "")
        if range_header:
            start, end = range_header.removeprefix("bytes=").split("-")
            return httpx.Response(206, content=gguf_bytes[int(start) : int(end) + 1])
        return httpx.Response(404)

    report = inspect_model(parse_source("unsloth/model-GGUF"), make_client(handler))
    assert report.artifact_type is ArtifactType.GGUF
    assert report.selected_gguf == "model-Q4_K_M.gguf"
    assert report.weight_bytes == 19_800_000_000
    assert report.architecture == "llama"
    assert report.context_length == 131072
    assert report.quantization == "Q4_K_M"


def test_inspect_gguf_many_variants_requires_choice():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response([
                {"rfilename": "m-Q4_K_M.gguf", "lfs": {"size": 10}},
                {"rfilename": "m-Q8_0.gguf", "lfs": {"size": 20}},
                {"rfilename": "mmproj-F16.gguf", "lfs": {"size": 5}},
            ]))
        return httpx.Response(404)

    report = inspect_model(parse_source("org/multi-GGUF"), make_client(handler))
    assert report.selected_gguf is None
    assert len([v for v in report.gguf_variants if v.is_mmproj]) == 1
    assert any("variant" in w for w in report.warnings)


def test_direct_file_link_selects_variant():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response([
                {"rfilename": "m-Q4_K_M.gguf", "lfs": {"size": 10}},
                {"rfilename": "m-Q8_0.gguf", "lfs": {"size": 20}},
            ]))
        range_header = request.headers.get("Range", "")
        if range_header:
            return httpx.Response(206, content=build_gguf([_kv_string("general.architecture", "llama")]))
        return httpx.Response(404)

    source = parse_source("https://huggingface.co/org/multi-GGUF/blob/main/m-Q8_0.gguf")
    report = inspect_model(source, make_client(handler))
    assert report.selected_gguf == "m-Q8_0.gguf"


def test_gated_repo_raises_auth_required():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "gated"})

    with pytest.raises(AuthRequired) as exc_info:
        inspect_model(parse_source("meta-llama/Llama-3.3-70B-Instruct"), make_client(handler))
    assert exc_info.value.had_token is False


def test_gated_with_bad_token_flagged():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "no access"})

    client = HfClient(token="hf_badtoken1234567890", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(AuthRequired) as exc_info:
        inspect_model(parse_source("meta-llama/Llama-3.3-70B-Instruct"), client)
    assert exc_info.value.had_token is True


def test_missing_repo_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(RepoNotFound):
        inspect_model(parse_source("nobody/does-not-exist"), make_client(handler))


def test_token_sent_as_bearer_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=hub_response([{"rfilename": "config.json", "size": 10}]))

    client = HfClient(token="hf_secret123456789012", client=httpx.Client(transport=httpx.MockTransport(handler)))
    inspect_model(parse_source("org/model"), client)
    assert seen["auth"] == "Bearer hf_secret123456789012"


def test_trust_remote_code_files_warned():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=hub_response([
                {"rfilename": "model.safetensors", "lfs": {"size": 100}},
                {"rfilename": "modeling_custom.py", "size": 5000},
            ]))
        return httpx.Response(404)

    report = inspect_model(parse_source("org/custom-model"), make_client(handler))
    assert report.trust_remote_code_files == ["modeling_custom.py"]
    assert any("trust_remote_code" in w for w in report.warnings)
