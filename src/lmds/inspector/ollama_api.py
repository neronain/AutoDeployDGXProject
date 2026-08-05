"""HTTP client สำหรับ Ollama registry — manifest + GGUF header ผ่าน HTTP Range

Ollama เก็บโมเดลเป็น OCI manifest ที่ชี้ไป blob หลายชั้น ชั้นที่ mediaType เป็น
`application/vnd.ollama.image.model` คือไฟล์ GGUF ล้วน (ขึ้นต้นด้วย magic `GGUF`)
จึงอ่าน header ด้วย parser ตัวเดียวกับ Hugging Face ได้เลย

หลักการเดียวกับ hf_api: ดึงเฉพาะ metadata ไม่โหลด weight
registry เปิดสาธารณะ ไม่มี auth — ไม่มีเคส gated/private เหมือน HF
"""

from __future__ import annotations

from typing import Any

import httpx

from .hf_api import HttpRangeSource

OLLAMA_REGISTRY = "https://registry.ollama.ai"

# ชั้นที่เป็นไฟล์โมเดลจริง — ชั้นอื่น (template/license/params) เป็น metadata ตัวเล็ก
MODEL_LAYER = "application/vnd.ollama.image.model"


class OllamaError(Exception):
    pass


class ManifestNotFound(OllamaError):
    def __init__(self, repo_id: str, tag: str):
        super().__init__(f"ไม่พบ Ollama model: {repo_id}:{tag}")


class NoModelLayer(OllamaError):
    """manifest มีอยู่แต่ไม่มีชั้นที่เป็นไฟล์โมเดล — ไม่ใช่ model ที่ deploy ได้"""


class OllamaClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def manifest(self, repo_id: str, tag: str) -> dict[str, Any]:
        url = f"{OLLAMA_REGISTRY}/v2/{repo_id}/manifests/{tag}"
        resp = self._client.get(url, headers={"User-Agent": "lmds"})
        if resp.status_code == 404:
            raise ManifestNotFound(repo_id, tag)
        if resp.status_code != 200:
            raise OllamaError(f"ดึง manifest ไม่สำเร็จ (HTTP {resp.status_code}): {repo_id}:{tag}")
        return resp.json()

    @staticmethod
    def model_layer(manifest: dict[str, Any]) -> tuple[str, int]:
        """คืน (digest, size) ของชั้นที่เป็นไฟล์ GGUF"""
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == MODEL_LAYER:
                digest = layer.get("digest")
                if not digest:
                    break
                return digest, int(layer.get("size") or 0)
        raise NoModelLayer(f"manifest ไม่มีชั้น {MODEL_LAYER}")

    def blob_range_source(self, repo_id: str, digest: str, budget: int = 64 * 1024 * 1024):
        url = f"{OLLAMA_REGISTRY}/v2/{repo_id}/blobs/{digest}"

        def fetch(start: int, end: int) -> bytes:
            headers = {"User-Agent": "lmds", "Range": f"bytes={start}-{end}"}
            # ใช้ stream เพื่อดู status ก่อนอ่าน body — blob เป็นไฟล์ระดับ GB
            # ถ้าเผลออ่าน body ของ 200 จะดูดทั้งไฟล์เข้า memory
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
                resp.read()
                return resp.content

        return HttpRangeSource(fetch, budget=budget)
