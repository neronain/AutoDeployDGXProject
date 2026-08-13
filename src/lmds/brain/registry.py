"""ตรวจว่า image tag ที่แผนเสนอมีอยู่จริงบน registry ก่อนเขียนลง bundle

ปัญหาที่แก้ (เจอจริง): LLM เสนอ `vllm/vllm-openai:v0.6.3.ss` ซึ่งเป็น tag ที่ไม่มีอยู่จริง
allowlist เดิมตรวจแค่ **repo** (`vllm/vllm-openai` อยู่ในรายการที่ยอมรับ) ไม่ได้ตรวจ tag
ผู้ใช้จึงได้ bundle ที่ผ่าน gate ทุกด่านแล้วไปตายตอนรันจริงด้วย:

    docker: Error response from daemon: manifest for vllm/vllm-openai:v0.6.3.ss not found

หลักที่ยึด:
  - **ปฏิเสธเฉพาะตอนรู้แน่ว่าไม่มี** (404 จาก registry) · ตรวจไม่ได้เพราะเน็ต/สิทธิ์
    ไม่ใช่เหตุผลที่จะบล็อก — เครื่องหลัง proxy หรือ air-gapped ต้อง deploy ได้ต่อ
  - ไม่เก็บ credential ของ registry ที่ไหนทั้งสิ้น: ขอ token แบบ anonymous เท่านั้น
    ตรวจได้เฉพาะ image สาธารณะ ซึ่งเป็นทั้งหมดที่ allowlist ยอมรับอยู่แล้ว
"""

from __future__ import annotations

import os

import httpx

# ปิดการตรวจได้ด้วย env — เครื่องที่ตั้งใจไม่ให้ออกเน็ตเลยจะได้ไม่ต้องรอ timeout ทุกครั้ง
# (เทสก็ใช้ตัวนี้ ไม่ต้องไป patch ฟังก์ชันซึ่งทำให้เทสของตัวมันเองทดสอบ stub แทนของจริง)
SKIP_ENV = "LMDS_SKIP_REGISTRY_CHECK"

_TIMEOUT = 8.0
_ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])

# registry ที่ตรวจได้แบบไม่ต้องล็อกอิน — ที่ไม่อยู่ในนี้จะถูกข้าม (ไม่ฟันธงว่าไม่มี)
_ANON_TOKEN = {
    "registry-1.docker.io": "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull",
    "ghcr.io": "https://ghcr.io/token?scope=repository:{repo}:pull",
}


def split_ref(image_ref: str) -> tuple[str, str, str]:
    """`ghcr.io/org/name:tag` → (host, repo, tag) · ไม่มี host = Docker Hub"""
    ref = (image_ref or "").strip()
    tag = "latest"
    if ":" in ref.rsplit("/", 1)[-1]:
        ref, tag = ref.rsplit(":", 1)
    head = ref.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        host, repo = head, ref.split("/", 1)[1] if "/" in ref else ""
    else:
        host, repo = "registry-1.docker.io", ref
    if host in ("docker.io", "index.docker.io"):
        host = "registry-1.docker.io"
    if host == "registry-1.docker.io" and "/" not in repo:
        repo = f"library/{repo}"      # `ubuntu` = `library/ubuntu` บน Docker Hub
    return host, repo, tag


def resolve_digest(image_ref: str, client: httpx.Client | None = None) -> str | None:
    """digest ที่ tag นี้ชี้อยู่ ณ ตอนนี้ — `sha256:...` หรือ None ถ้าถามไม่ได้

    tag เคลื่อนที่ได้: `vllm/vllm-openai:latest` วันนี้กับเดือนหน้าเป็นคนละ image
    ซึ่งแปลว่า bundle ที่รันผ่านเมื่อวาน อาจรันไม่ผ่านวันนี้โดยไม่มีอะไรในไฟล์เปลี่ยนเลย
    digest ไม่เคลื่อน — ตรึงไว้แล้วสิ่งที่ทดสอบคือสิ่งที่รัน

    None ไม่ใช่ความล้มเหลว เหมือน tag_exists: registry ที่ต้องล็อกอิน (nvcr.io)
    เครื่องที่ไม่มีเน็ต หรือ proxy ที่บล็อก ล้วนถามไม่ได้ — และไม่ใช่เหตุผลที่จะ
    ห้าม deploy ผู้เรียกจึงต้องรับมือกับ None เสมอ
    """
    if os.environ.get(SKIP_ENV):
        return None
    host, repo, tag = split_ref(image_ref)
    token_url = _ANON_TOKEN.get(host)
    if not token_url or not repo:
        return None
    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT, follow_redirects=True)
    try:
        auth = client.get(token_url.format(repo=repo))
        if auth.status_code != 200:
            return None
        token = auth.json().get("token") or auth.json().get("access_token") or ""
        resp = client.request(
            "HEAD", f"https://{host}/v2/{repo}/manifests/{tag}",
            headers={"Authorization": f"Bearer {token}", "Accept": _ACCEPT},
        )
        if resp.status_code >= 400:
            return None
        digest = resp.headers.get("Docker-Content-Digest") or ""
        return digest if digest.startswith("sha256:") else None
    except httpx.HTTPError:
        return None
    finally:
        if owns_client:
            client.close()


def tag_exists(image_ref: str, client: httpx.Client | None = None) -> bool | None:
    """tag นี้มีอยู่จริงไหม — True มี · False ไม่มีแน่ ๆ · None ตรวจไม่ได้

    `None` ไม่ใช่ความล้มเหลว: registry ที่ต้องล็อกอิน (เช่น nvcr.io) เครื่องที่ไม่มีเน็ต
    หรือ proxy ที่บล็อกอยู่ ล้วนตรวจไม่ได้ — และไม่ใช่เหตุผลที่จะห้าม deploy
    """
    if os.environ.get(SKIP_ENV):
        return None
    host, repo, tag = split_ref(image_ref)
    token_url = _ANON_TOKEN.get(host)
    if not token_url or not repo:
        return None
    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT, follow_redirects=True)
    try:
        auth = client.get(token_url.format(repo=repo))
        if auth.status_code != 200:
            return None
        token = auth.json().get("token") or auth.json().get("access_token") or ""
        resp = client.request(
            "HEAD", f"https://{host}/v2/{repo}/manifests/{tag}",
            headers={"Authorization": f"Bearer {token}", "Accept": _ACCEPT},
        )
        if resp.status_code == 404:
            return False
        if resp.status_code < 400:
            return True
        return None            # 401/403/5xx = ตรวจไม่ได้ ไม่ใช่ไม่มี
    except httpx.HTTPError:
        return None
    finally:
        if owns_client:
            client.close()
