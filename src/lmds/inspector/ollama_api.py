"""HTTP client สำหรับ Ollama registry — manifest + GGUF header ผ่าน HTTP Range

Ollama เก็บโมเดลเป็น OCI manifest ที่ชี้ไป blob หลายชั้น ชั้นที่ mediaType เป็น
`application/vnd.ollama.image.model` คือไฟล์ GGUF ล้วน (ขึ้นต้นด้วย magic `GGUF`)
จึงอ่าน header ด้วย parser ตัวเดียวกับ Hugging Face ได้เลย

หลักการเดียวกับ hf_api: ดึงเฉพาะ metadata ไม่โหลด weight
registry เปิดสาธารณะ ไม่มี auth — ไม่มีเคส gated/private เหมือน HF
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .hf_api import HttpRangeSource

OLLAMA_REGISTRY = "https://registry.ollama.ai"

# ชั้นที่เป็นไฟล์โมเดลจริง — ชั้นอื่น (template/license/params) เป็น metadata ตัวเล็ก
MODEL_LAYER = "application/vnd.ollama.image.model"
MANIFEST_CAP = 4 * 1024 * 1024
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")


class OllamaError(Exception):
    pass


class ManifestNotFound(OllamaError):
    def __init__(self, repo_id: str, tag: str):
        super().__init__(f"ไม่พบ Ollama model: {repo_id}:{tag}")


class NoModelLayer(OllamaError):
    """manifest มีอยู่แต่ไม่มีชั้นที่เป็นไฟล์โมเดล — ไม่ใช่ model ที่ deploy ได้"""


class InvalidManifest(OllamaError):
    """manifest/blob descriptor ผิด schema หรือกำกวม — ห้ามเดาต่อ"""


def _read_limited(resp: httpx.Response, cap: int, label: str) -> bytes:
    """อ่าน response แบบมีเพดาน แม้ peer จะโกหก/ไม่ส่ง Content-Length"""
    raw_length = resp.headers.get("Content-Length")
    if raw_length:
        try:
            if int(raw_length) > cap:
                raise OllamaError(f"{label} ใหญ่เกินเพดาน {cap:,} bytes")
        except ValueError as exc:
            raise OllamaError(f"{label} ส่ง Content-Length ไม่ถูกต้อง") from exc

    out = bytearray()
    for chunk in resp.iter_bytes():
        if len(out) + len(chunk) > cap:
            raise OllamaError(f"{label} ใหญ่เกินเพดาน {cap:,} bytes")
        out.extend(chunk)
    return bytes(out)


class OllamaClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def manifest(self, repo_id: str, tag: str) -> dict[str, Any]:
        url = f"{OLLAMA_REGISTRY}/v2/{repo_id}/manifests/{tag}"
        try:
            with self._client.stream("GET", url, headers={"User-Agent": "lmds"}) as resp:
                if resp.status_code == 404:
                    raise ManifestNotFound(repo_id, tag)
                if resp.status_code != 200:
                    raise OllamaError(
                        f"ดึง manifest ไม่สำเร็จ (HTTP {resp.status_code}): {repo_id}:{tag}"
                    )
                body = _read_limited(resp, MANIFEST_CAP, "Ollama manifest")
        except httpx.HTTPError as exc:
            raise OllamaError(f"ติดต่อ Ollama registry ไม่สำเร็จ: {exc}") from exc

        try:
            doc = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidManifest("Ollama manifest ไม่ใช่ JSON ที่ถูกต้อง") from exc
        if not isinstance(doc, dict):
            raise InvalidManifest("Ollama manifest ต้องเป็น JSON object")
        if doc.get("schemaVersion") != 2:
            raise InvalidManifest("Ollama manifest ไม่มี schemaVersion=2")
        if not isinstance(doc.get("layers"), list):
            raise InvalidManifest("Ollama manifest ไม่มี layers array")
        if any(not isinstance(layer, dict) for layer in doc["layers"]):
            raise InvalidManifest("Ollama manifest มี layer descriptor ที่ไม่ใช่ object")
        return doc

    @staticmethod
    def model_layer(manifest: dict[str, Any]) -> tuple[str, int]:
        """คืน (digest, size) ของชั้นที่เป็นไฟล์ GGUF"""
        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise InvalidManifest("Ollama manifest ไม่มี layers array")
        matches = [
            layer for layer in layers
            if isinstance(layer, dict) and layer.get("mediaType") == MODEL_LAYER
        ]
        if not matches:
            raise NoModelLayer(f"manifest ไม่มีชั้น {MODEL_LAYER}")
        if len(matches) != 1:
            raise InvalidManifest(f"manifest มีชั้น {MODEL_LAYER} มากกว่าหนึ่งชั้น")

        layer = matches[0]
        digest = layer.get("digest")
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise InvalidManifest("model layer มี digest ที่ไม่ใช่ sha256:<64 lowercase hex>")
        size = layer.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise InvalidManifest("model layer มี size ที่ไม่ใช่จำนวนเต็มบวก")
        return digest, size

    def blob_range_source(
        self,
        repo_id: str,
        digest: str,
        expected_size: int,
        budget: int = 64 * 1024 * 1024,
    ):
        if not _DIGEST_RE.fullmatch(digest):
            raise InvalidManifest("blob digest ไม่ใช่ sha256:<64 lowercase hex>")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise InvalidManifest("blob size ต้องเป็นจำนวนเต็มบวก")
        url = f"{OLLAMA_REGISTRY}/v2/{repo_id}/blobs/{digest}"

        def fetch(start: int, end: int) -> bytes:
            headers = {"User-Agent": "lmds", "Range": f"bytes={start}-{end}"}
            # ใช้ stream เพื่อดู status ก่อนอ่าน body — blob เป็นไฟล์ระดับ GB
            # ถ้าเผลออ่าน body ของ 200 จะดูดทั้งไฟล์เข้า memory
            try:
                with self._client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 404:
                        raise OllamaError(f"ไม่พบ blob: {digest}")
                    if resp.status_code == 200:
                        # เมิน Range แล้วส่งจากต้นไฟล์ — อ่านต่อจะได้ข้อมูลผิดตำแหน่งแบบเงียบ ๆ
                        raise OllamaError(
                            "registry ไม่รองรับ HTTP Range (ตอบ 200 แทน 206) — "
                            "อ่าน GGUF header ต่อไม่ได้เพราะ offset จะเพี้ยน"
                        )
                    if resp.status_code != 206:
                        raise OllamaError(f"Range request ล้มเหลว (HTTP {resp.status_code})")

                    raw_range = resp.headers.get("Content-Range", "")
                    match = _CONTENT_RANGE_RE.fullmatch(raw_range)
                    if not match:
                        raise OllamaError("Range response ไม่มี Content-Range ที่ถูกต้อง")
                    got_start, got_end, got_total = (int(value) for value in match.groups())
                    expected_end = min(end, expected_size - 1)
                    if (
                        got_start != start
                        or got_end != expected_end
                        or got_total != expected_size
                        or got_end < got_start
                    ):
                        raise OllamaError(
                            "Range response ไม่ตรงช่วงที่ขอ "
                            f"(ขอ bytes={start}-{end}; ได้ {raw_range})"
                        )
                    expected_length = got_end - got_start + 1
                    body = _read_limited(resp, expected_length, "Ollama blob range")
                    if len(body) != expected_length:
                        raise OllamaError(
                            "Range response มีขนาด body ไม่ตรง Content-Range "
                            f"(ได้ {len(body):,}; ต้องการ {expected_length:,} bytes)"
                        )
                    return body
            except httpx.HTTPError as exc:
                raise OllamaError(f"ติดต่อ Ollama blob ไม่สำเร็จ: {exc}") from exc

        return HttpRangeSource(fetch, budget=budget)
