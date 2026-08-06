import httpx
import pytest
import json

from lmds.inspector import ArtifactType, HfClient, inspect_model
from lmds.inspector.ollama_api import (
    InvalidManifest,
    MANIFEST_CAP,
    MODEL_LAYER,
    ManifestNotFound,
    NoModelLayer,
    OllamaClient,
    OllamaError,
)
from lmds.resolver import parse_source
from lmds.resolver.parse import SourceError, UnsupportedSource
from tests.test_gguf import _kv_string, _kv_u32, build_gguf

BLOB = build_gguf([
    _kv_string("general.architecture", "qwen3"),
    _kv_u32("qwen3.context_length", 40960),
    _kv_u32("qwen3.block_count", 36),
    _kv_u32("qwen3.attention.head_count", 32),
    _kv_u32("qwen3.attention.head_count_kv", 8),
    _kv_u32("qwen3.attention.key_length", 128),
    _kv_u32("general.file_type", 15),
    _kv_string("tokenizer.chat_template", "{{ x }}"),
])
DIGEST = "sha256:" + "a3" * 32


def manifest(layers=None):
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "layers": layers if layers is not None else [
            {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": len(BLOB)},
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:bb", "size": 20},
        ],
    }


def make_client(handler) -> OllamaClient:
    return OllamaClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def registry_handler(body=BLOB, manifest_doc=None, blob_status=206):
    """จำลอง registry: /v2/<repo>/manifests/<tag> และ /v2/<repo>/blobs/<digest> ที่รองรับ Range"""
    def handler(request: httpx.Request) -> httpx.Response:
        if "/manifests/" in request.url.path:
            return httpx.Response(200, json=manifest_doc or manifest())
        if "/blobs/" in request.url.path:
            if blob_status != 206:
                return httpx.Response(blob_status, content=body)
            start, _, end = request.headers["Range"].removeprefix("bytes=").partition("-")
            actual_end = min(int(end), len(body) - 1)
            chunk = body[int(start) : actual_end + 1]
            return httpx.Response(
                206, content=chunk,
                headers={"Content-Range": f"bytes {start}-{actual_end}/{len(body)}"},
            )
        return httpx.Response(404)
    return handler


# ---------- parse ----------

@pytest.mark.parametrize("text, repo_id, revision", [
    ("ollama.com/qwen3", "library/qwen3", "latest"),
    ("ollama.com/library/qwen3:8b", "library/qwen3", "8b"),
    ("https://ollama.com/library/gemma3", "library/gemma3", "latest"),
    ("ollama.com/someuser/mymodel:q4", "someuser/mymodel", "q4"),
    ("registry.ollama.ai/v2/library/qwen3/manifests/8b", "library/qwen3", "8b"),
])
def test_parse_ollama_refs(text, repo_id, revision):
    source = parse_source(text)
    assert source.kind == "ollama"
    assert source.repo_id == repo_id
    assert source.revision == revision


def test_hf_passthrough_ref_is_rejected_with_guidance():
    """ollama.com/hf.co/... คือโมเดลบน HF และ tag คือ quant — เดาชื่อไฟล์ให้เองจะผิดเงียบ"""
    with pytest.raises(SourceError, match="huggingface.co"):
        parse_source("ollama.com/hf.co/unsloth/Qwen3-0.6B-GGUF:Q8_0")


def test_ngc_still_reports_unsupported():
    with pytest.raises(UnsupportedSource):
        parse_source("catalog.ngc.nvidia.com/models/x")


# ---------- inspect ----------

def test_inspect_ollama_reads_gguf_header():
    report = inspect_model(
        parse_source("ollama.com/library/qwen3:8b"), HfClient(),
        ollama_client=make_client(registry_handler()),
    )
    assert report.artifact_type is ArtifactType.GGUF
    assert report.source_kind == "ollama"
    assert report.repo_id == "library/qwen3:8b"
    assert report.architecture == "qwen3"
    assert report.context_length == 40960
    assert report.kv_dims.layers == 36
    assert report.kv_dims.kv_heads == 8
    assert report.has_chat_template is True
    assert report.quantization == "gguf-file-type-15"
    assert report.weight_bytes == len(BLOB)
    assert report.gguf_variants[0].download_url == (
        f"https://registry.ollama.ai/v2/library/qwen3/blobs/{DIGEST}"
    )
    assert not report.warnings


def test_digest_becomes_the_pin_not_the_tag():
    """tag อย่าง latest ชี้ blob คนละตัวได้เมื่อเวลาผ่านไป — digest คือสิ่งที่ pin ได้จริง"""
    report = inspect_model(
        parse_source("ollama.com/qwen3"), HfClient(),
        ollama_client=make_client(registry_handler()),
    )
    assert report.revision_requested == "latest"
    assert report.revision_sha == DIGEST.split(":", 1)[1]
    assert report.gguf_variants[0].sha256 == report.revision_sha
    assert report.selected_gguf == f"sha256-{report.revision_sha}"


def test_range_ignored_by_server_fails_loudly():
    """ถ้า registry ตอบ 200 แทน 206 แปลว่าส่งมาจากต้นไฟล์ — อ่านต่อจะได้ offset เพี้ยนแบบเงียบ"""
    report = inspect_model(
        parse_source("ollama.com/qwen3"), HfClient(),
        ollama_client=make_client(registry_handler(blob_status=200)),
    )
    assert any("Range" in w for w in report.warnings)
    assert report.architecture is None


def test_wrong_content_range_start_fails_loudly():
    """206 อย่างเดียวไม่พอ — chunk ที่เริ่มคนละ offset ทำให้ GGUF metadata ผิดแบบเงียบ"""
    def handler(request: httpx.Request) -> httpx.Response:
        if "/manifests/" in request.url.path:
            return httpx.Response(200, json=manifest())
        return httpx.Response(
            206,
            content=b"x" * 64,
            headers={"Content-Range": f"bytes 1000-1063/{len(BLOB)}"},
        )

    report = inspect_model(
        parse_source("ollama.com/qwen3"), HfClient(),
        ollama_client=make_client(handler),
    )
    assert report.architecture is None
    assert any("ไม่ตรงช่วง" in warning for warning in report.warnings)


def test_content_range_body_length_must_match():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/manifests/" in request.url.path:
            return httpx.Response(200, json=manifest())
        return httpx.Response(
            206,
            content=b"short",
            headers={"Content-Range": f"bytes 0-{len(BLOB) - 1}/{len(BLOB)}"},
        )

    report = inspect_model(
        parse_source("ollama.com/qwen3"), HfClient(),
        ollama_client=make_client(handler),
    )
    assert report.architecture is None
    assert any("ขนาด body" in warning for warning in report.warnings)


def test_manifest_without_model_layer():
    only_template = [{"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:bb", "size": 20}]
    client = make_client(registry_handler(manifest_doc=manifest(only_template)))
    with pytest.raises(NoModelLayer):
        inspect_model(parse_source("ollama.com/qwen3"), HfClient(), ollama_client=client)


@pytest.mark.parametrize("bad_layer", [
    {"mediaType": MODEL_LAYER, "size": len(BLOB)},
    {"mediaType": MODEL_LAYER, "digest": "sha256:not-a-digest", "size": len(BLOB)},
    {"mediaType": MODEL_LAYER, "digest": DIGEST + "\r\n", "size": len(BLOB)},
    {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": "large"},
    {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": {}},
    {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": 0},
    {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": -5},
])
def test_invalid_model_descriptor_fails_cleanly(bad_layer):
    client = make_client(registry_handler(manifest_doc=manifest([bad_layer])))
    with pytest.raises(InvalidManifest):
        inspect_model(parse_source("ollama.com/qwen3"), HfClient(), ollama_client=client)


def test_duplicate_model_layers_are_ambiguous():
    layer = {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": len(BLOB)}
    client = make_client(registry_handler(manifest_doc=manifest([layer, layer])))
    with pytest.raises(InvalidManifest, match="มากกว่าหนึ่ง"):
        inspect_model(parse_source("ollama.com/qwen3"), HfClient(), ollama_client=client)


@pytest.mark.parametrize("body", [b"[]", b'"not-an-object"', b"not-json"])
def test_manifest_root_and_json_are_validated(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(InvalidManifest):
        make_client(handler).manifest("library/qwen3", "latest")


@pytest.mark.parametrize("doc", [
    {"schemaVersion": 2, "layers": {}},
    {"schemaVersion": 2, "layers": ["not-an-object"]},
    {"schemaVersion": 1, "layers": []},
])
def test_manifest_schema_is_validated(doc):
    with pytest.raises(InvalidManifest):
        make_client(registry_handler(manifest_doc=doc)).manifest("library/qwen3", "latest")


def test_manifest_body_has_a_hard_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (MANIFEST_CAP + 1))

    with pytest.raises(OllamaError, match="เพดาน"):
        make_client(handler).manifest("library/qwen3", "latest")


def test_registry_transport_error_is_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(OllamaError, match="registry"):
        make_client(handler).manifest("library/qwen3", "latest")


def test_external_ollama_template_does_not_claim_embedded_jinja():
    blob = build_gguf([_kv_string("general.architecture", "qwen3")])
    doc = manifest([
        {"mediaType": MODEL_LAYER, "digest": DIGEST, "size": len(blob)},
        {
            "mediaType": "application/vnd.ollama.image.template",
            "digest": "sha256:" + "b" * 64,
            "size": 20,
        },
    ])
    report = inspect_model(
        parse_source("ollama.com/qwen3"), HfClient(),
        ollama_client=make_client(registry_handler(body=blob, manifest_doc=doc)),
    )
    assert report.has_chat_template is False
    assert any("ไม่ import/แปลง" in warning for warning in report.warnings)


def test_manifest_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(ManifestNotFound):
        inspect_model(parse_source("ollama.com/nope"), HfClient(), ollama_client=make_client(handler))


def test_registry_error_is_not_swallowed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(OllamaError):
        inspect_model(parse_source("ollama.com/qwen3"), HfClient(), ollama_client=make_client(handler))


# ---------- local Ollama store ----------

def test_scanner_finds_extensionless_ollama_blob(tmp_path, monkeypatch):
    from lmds import scanner

    root = tmp_path / "models"
    manifests = root / "manifests" / "registry.ollama.ai" / "library" / "qwen3"
    blobs = root / "blobs"
    manifests.mkdir(parents=True)
    blobs.mkdir()
    size = 33 * 1024 * 1024
    digest = "sha256:" + "c" * 64
    (manifests / "8b").write_text(json.dumps(manifest([
        {"mediaType": MODEL_LAYER, "digest": digest, "size": size},
    ])), encoding="utf-8")
    blob = blobs / digest.replace(":", "-", 1)
    with blob.open("wb") as handle:
        handle.truncate(size)
    monkeypatch.setattr(scanner, "candidate_roots", lambda extra=None: [root])

    found = scanner.scan()
    assert [(item.kind, item.name, item.path) for item in found] == [
        ("ollama", "library/qwen3:8b", str(blob)),
    ]
    assert scanner.find_model("library/qwen3", revision="8b", digest=digest) == found[0]


def test_scanner_rejects_incomplete_ollama_blob(tmp_path, monkeypatch):
    from lmds import scanner

    root = tmp_path / "models"
    manifests = root / "manifests" / "registry.ollama.ai" / "library" / "qwen3"
    blobs = root / "blobs"
    manifests.mkdir(parents=True)
    blobs.mkdir()
    digest = "sha256:" + "d" * 64
    (manifests / "latest").write_text(json.dumps(manifest([
        {"mediaType": MODEL_LAYER, "digest": digest, "size": 100},
    ])), encoding="utf-8")
    (blobs / digest.replace(":", "-", 1)).write_bytes(b"short")
    monkeypatch.setattr(scanner, "candidate_roots", lambda extra=None: [root])

    assert scanner.scan() == []
