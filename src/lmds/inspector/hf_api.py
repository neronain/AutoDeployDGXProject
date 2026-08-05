"""HTTP client สำหรับ Hugging Face Hub API — ดึงเฉพาะ metadata และไฟล์เล็ก ไม่โหลด weight

หลักการ (FR-1.5, FR-1.6):
- 401/403 → AuthRequired ให้ชั้น CLI ตัดสินใจถาม token (optional)
- ไฟล์เล็กมีเพดานขนาด, ไฟล์ใหญ่ (GGUF header) อ่านผ่าน HTTP Range พร้อม budget
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

HF_BASE = "https://huggingface.co"
# 16MB — เหตุผลเดียวกับ INDEX_FILE_CAP: quant config ต่อชั้นของ MoE ตัวใหญ่ทำให้ config.json
# โตกว่าที่คาดมาก เคสจริง Nemotron-3-Super-120B NVFP4 = 7.4MB ซึ่งเป็น metadata ปกติ
# ยังคงเพดานไว้เพื่อกันไฟล์ผิดปกติ แต่ 4MB แคบเกินไปสำหรับโมเดลรุ่นใหม่
SMALL_FILE_CAP = 16 * 1024 * 1024

# model.safetensors.index.json โตตาม *จำนวน tensor* ไม่ใช่ขนาดโมเดล:
# MoE ตัวใหญ่ + quant ละเอียด (NVFP4/FP8 ที่มี scale ต่อ block) มีได้หลายแสน entry
# เช่น Qwen3.5-122B-A10B NVFP4 → index เกิน 4MB ทั้งที่เป็น metadata ปกติ
INDEX_FILE_CAP = 64 * 1024 * 1024


class HfError(Exception):
    pass


class AuthRequired(HfError):
    """repo เป็น gated/private — ต้องใช้ HF token (หรือ token ที่มียังไม่ได้รับสิทธิ์)"""

    def __init__(self, repo_id: str, status: int, had_token: bool):
        self.repo_id = repo_id
        self.status = status
        self.had_token = had_token
        detail = (
            "token ที่ให้มายังเข้าถึงไม่ได้ (อาจต้องกดยอมรับเงื่อนไขบนเว็บ Hugging Face ก่อน)"
            if had_token
            else "repo นี้เป็น gated/private — ต้องใช้ Hugging Face token"
        )
        super().__init__(f"{repo_id}: {detail} (HTTP {status})")


class RepoNotFound(HfError):
    def __init__(self, repo_id: str):
        super().__init__(f"ไม่พบ model repo: {repo_id}")


class BudgetExceeded(HfError):
    """อ่านข้อมูลเกิน budget ที่ตั้งไว้ (กันการโหลดไฟล์ใหญ่โดยไม่ตั้งใจ)"""


class HfClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        self.token = token
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "lmds"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _raise_for_access(self, repo_id: str, status: int) -> None:
        if status in (401, 403):
            raise AuthRequired(repo_id, status, had_token=self.token is not None)
        if status == 404:
            raise RepoNotFound(repo_id)

    def model_info(self, repo_id: str, revision: str | None = None) -> dict[str, Any]:
        """GET /api/models/... พร้อมขนาดไฟล์ (blobs=true) — คืน JSON ดิบของ Hub"""
        path = f"/api/models/{repo_id}"
        if revision:
            path += f"/revision/{revision}"
        resp = self._client.get(f"{HF_BASE}{path}", params={"blobs": "true"}, headers=self._headers())
        self._raise_for_access(repo_id, resp.status_code)
        if resp.status_code != 200:
            raise HfError(f"Hub API ตอบ HTTP {resp.status_code} สำหรับ {repo_id}")
        return resp.json()

    def fetch_small_file(
        self, repo_id: str, revision: str, filename: str, cap: int = SMALL_FILE_CAP
    ) -> bytes | None:
        """ดึงไฟล์เล็ก (config/tokenizer/index) — คืน None ถ้าไม่มีไฟล์นั้น"""
        url = f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"
        resp = self._client.get(url, headers=self._headers())
        if resp.status_code == 404:
            return None
        self._raise_for_access(repo_id, resp.status_code)
        if resp.status_code != 200:
            raise HfError(f"ดึง {filename} ไม่สำเร็จ (HTTP {resp.status_code})")
        if len(resp.content) > cap:
            raise BudgetExceeded(
                f"{filename} ใหญ่ {len(resp.content):,} bytes เกินเพดาน {cap:,} bytes — "
                "ไม่ใช่ไฟล์ metadata ปกติ"
            )
        return resp.content

    def range_source(self, repo_id: str, revision: str, filename: str, budget: int = 64 * 1024 * 1024):
        url = f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"
        headers = self._headers()

        def fetch(start: int, end: int) -> bytes:
            resp = self._client.get(url, headers={**headers, "Range": f"bytes={start}-{end}"})
            if resp.status_code == 404:
                raise RepoNotFound(repo_id)
            self._raise_for_access(repo_id, resp.status_code)
            if resp.status_code not in (200, 206):
                raise HfError(f"Range request ล้มเหลว (HTTP {resp.status_code})")
            return resp.content

        return HttpRangeSource(fetch, budget=budget)


class HttpRangeSource:
    """file-like source สำหรับ parser: read(n) / skip(n) โดยดึงข้อมูลเป็นช่วงตามต้องการ

    skip(n) ไม่ดาวน์โหลดข้อมูล — ใช้ข้ามส่วนที่ไม่สนใจ (เช่น numeric array ยาว ๆ ใน GGUF)
    """

    def __init__(self, fetch: Callable[[int, int], bytes], chunk: int = 1024 * 1024, budget: int = 64 * 1024 * 1024):
        self._fetch = fetch
        self._chunk = chunk
        self._budget = budget
        self.spent = 0
        self.pos = 0
        self._buf = b""
        self._buf_start = 0

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            buf_off = self.pos - self._buf_start
            if 0 <= buf_off < len(self._buf):
                take = min(n, len(self._buf) - buf_off)
                out += self._buf[buf_off : buf_off + take]
                self.pos += take
                n -= take
                continue
            fetch_len = max(n, self._chunk)
            if self.spent + fetch_len > self._budget:
                fetch_len = max(n, self._budget - self.spent)
            if self.spent + n > self._budget:
                raise BudgetExceeded(f"อ่านเกิน budget {self._budget} bytes")
            data = self._fetch(self.pos, self.pos + fetch_len - 1)
            if not data:
                raise EOFError("ปลายไฟล์ก่อนอ่านครบ")
            self.spent += len(data)
            self._buf = data
            self._buf_start = self.pos
        return bytes(out)

    def skip(self, n: int) -> None:
        self.pos += n
