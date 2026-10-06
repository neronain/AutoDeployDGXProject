"""HTTP client สำหรับ Hugging Face Hub API — ดึงเฉพาะ metadata และไฟล์เล็ก ไม่โหลด weight

หลักการ (FR-1.5, FR-1.6):
- 401/403 → AuthRequired ให้ชั้น CLI ตัดสินใจถาม token (optional) · แยก "gated" ออกจาก "ไม่พบหรือ private"
  ด้วย header `x-error-code` ของ Hub (ดู _raise_for_access)
- ไฟล์เล็กมีเพดานขนาด, ไฟล์ใหญ่ (GGUF header) อ่านผ่าน HTTP Range พร้อม budget
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import quote

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
    """เข้าไม่ถึง repo โดยไม่มี token ที่ใช้ได้ — `reason` บอกว่ารู้แค่ไหน

    - "gated"   = Hub บอกเองว่า repo มีอยู่และถูกจำกัดสิทธิ์ (`x-error-code: GatedRepo` หรือ 403)
    - "unknown" = 401 เปล่า ๆ: **repo ไม่มีอยู่ หรือเป็น private** — Hub ตอบเหมือนกันทั้งสองกรณีเมื่อไม่ได้ล็อกอิน
      (วัดจริง 2026-10-06: `/api/models/Qwen/Qwen3-8B-typo-zz` → 401 "Invalid username or password." ไม่มี x-error-code)
      เดิมทุก 401 ถูกรายงานว่า "repo นี้เป็น gated/private" — พิมพ์ชื่อผิดแล้วถูกถาม token
    """

    def __init__(self, repo_id: str, status: int, had_token: bool, reason: str = "gated"):
        self.repo_id = repo_id
        self.status = status
        self.had_token = had_token
        self.reason = reason
        if reason == "unknown":
            detail = (
                "ไม่พบ repo นี้ หรือ token ที่ให้มาไม่มีสิทธิ์เข้าถึง — เช็คชื่อ org/model และสิทธิ์ของ token"
                if had_token
                else "ไม่พบ repo นี้ หรือเป็น repo private — Hub ตอบเหมือนกันทั้งสองกรณีเมื่อไม่มี token · "
                     "เช็คชื่อ org/model ก่อน ถ้าเป็น repo private จึงใส่ Hugging Face token"
            )
        else:
            detail = (
                "token ที่ให้มายังเข้าถึงไม่ได้ (อาจต้องกดยอมรับเงื่อนไขบนเว็บ Hugging Face ก่อน)"
                if had_token
                else "repo นี้เป็น gated — ต้องใช้ Hugging Face token ของบัญชีที่ได้รับสิทธิ์แล้ว"
            )
        super().__init__(f"{repo_id}: {detail} (HTTP {status})")


class RepoNotFound(HfError):
    def __init__(self, repo_id: str):
        super().__init__(f"ไม่พบ model repo: {repo_id}")


class RevisionNotFound(RepoNotFound):
    """repo มีอยู่ แต่ไม่มี branch/tag/commit ที่ขอ — เป็น input ผิดแบบเดียวกับชื่อ repo ผิด (CLI exit 1)"""

    def __init__(self, repo_id: str, revision: str | None):
        HfError.__init__(self, f"ไม่พบ revision {revision!r} ใน {repo_id} — เช็คชื่อ branch/tag/commit "
                               "(PR ใช้รูป refs/pr/<เลข>)")


class BudgetExceeded(HfError):
    """อ่านข้อมูลเกิน budget ที่ตั้งไว้ (กันการโหลดไฟล์ใหญ่โดยไม่ตั้งใจ)"""


class HfClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        self.token = token
        # read นานกว่า connect: ไฟล์ที่ HF ย้ายไป Xet bridge ตอบ byte แรกช้าได้ 20–60 วิ (วัดจริง
        # 2026-09-04 ตอน analyze GGUF ที่ต้องอ่าน header ด้วย range request) · 30 วิเดิมทำให้ analyze
        # ล้มเป็น "ต่อ Hugging Face ไม่ได้" ทั้งที่ Hub ปกติดี — ส่วน connect ยัง 30 วิ จะได้รู้เร็วเมื่อไม่มีเน็ต
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "lmds"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, url: str, **kw) -> httpx.Response:
        """GET ที่แปล error ระดับเครือข่ายเป็น HfError — ไม่ปล่อย httpx.ConnectError หลุดขึ้นไป

        เคสจริง 2026-09-04: HF ล่มชั่วคราว (SSL EOF / handshake timeout) → หน้าเว็บได้ 500 "Internal
        Server Error" เปล่า ๆ และ CLI ได้ traceback ยาวของ httpx ทั้งที่ความหมายแค่ "ต่อ Hugging Face
        ไม่ได้ตอนนี้ ลองใหม่" · ทุกที่ที่จับ HfError อยู่แล้วจึงได้ข้อความที่ทำต่อได้
        """
        try:
            return self._client.get(url, **kw)
        except httpx.HTTPError as exc:
            raise HfError(
                f"ต่อ Hugging Face ไม่ได้ ({type(exc).__name__}: {exc or 'timeout'}) — "
                "เช็คอินเทอร์เน็ต/proxy ของเครื่องนี้ แล้วลองใหม่ · เครื่อง air-gapped ใช้ mirror ภายในผ่าน HF_ENDPOINT"
            ) from exc

    def _raise_for_access(self, repo_id: str, resp: httpx.Response, revision: str | None = None) -> None:
        """แปล status + `x-error-code` ของ Hub เป็นสาเหตุที่ถูก — ไม่เดาจาก status อย่างเดียว

        Hub ตอบ (วัดจริง 2026-10-06): repo ที่ไม่มีอยู่ + ไม่ล็อกอิน → 401 ไม่มี code · repo gated →
        `/api/models` 200 แล้ว `resolve` 401 `GatedRepo` · revision ผิด → 404 `RevisionNotFound` · ล็อกอินแล้ว
        repo ไม่มี → 404 `RepoNotFound`
        """
        status = resp.status_code
        code = resp.headers.get("x-error-code", "")
        if code == "RepoNotFound":
            raise RepoNotFound(repo_id)
        if code == "RevisionNotFound":
            raise RevisionNotFound(repo_id, revision)
        if code == "GatedRepo" or status == 403:
            raise AuthRequired(repo_id, status, had_token=self.token is not None, reason="gated")
        if status == 401:
            raise AuthRequired(repo_id, status, had_token=self.token is not None, reason="unknown")
        if status == 404:
            raise RepoNotFound(repo_id)

    def model_info(self, repo_id: str, revision: str | None = None) -> dict[str, Any]:
        """GET /api/models/... พร้อมขนาดไฟล์ (blobs=true) — คืน JSON ดิบของ Hub"""
        path = f"/api/models/{repo_id}"
        if revision:
            # revision ที่มี / (refs/pr/1 · refs/convert/…) ต้อง encode เป็น segment เดียว — ไม่ encode Hub ตอบ 404
            path += f"/revision/{quote(revision, safe='')}"
        resp = self._get(f"{HF_BASE}{path}", params={"blobs": "true"}, headers=self._headers())
        self._raise_for_access(repo_id, resp, revision)
        if resp.status_code != 200:
            raise HfError(f"Hub API ตอบ HTTP {resp.status_code} สำหรับ {repo_id}")
        return resp.json()

    def fetch_small_file(
        self, repo_id: str, revision: str, filename: str, cap: int = SMALL_FILE_CAP
    ) -> bytes | None:
        """ดึงไฟล์เล็ก (config/tokenizer/index) — คืน None ถ้าไม่มีไฟล์นั้น"""
        url = f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"
        resp = self._get(url, headers=self._headers())
        if resp.status_code == 404:
            return None
        self._raise_for_access(repo_id, resp, revision)
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
            resp = self._get(url, headers={**headers, "Range": f"bytes={start}-{end}"})
            if resp.status_code == 404:
                raise RepoNotFound(repo_id)
            self._raise_for_access(repo_id, resp, revision)
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
