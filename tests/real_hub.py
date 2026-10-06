"""Hub ปลอมที่เสิร์ฟ metadata **ของจริง** ให้ `inspect_model` ตัวจริงเดินทั้งเส้น — ไม่มีเครือข่าย ไม่มี weight

`tests/hub_fixtures/<org>--<name>.json` คือสิ่งที่ Hub ตอบจริง ณ วันที่ในช่อง `fetched` ตัดเหลือเฉพาะส่วนที่ inspector
อ่าน: รายชื่อไฟล์ + ขนาด · tags/library/pipeline · config.json · index ของ safetensors (tensor ละตัวต่อ shard) ·
และคีย์ใน header ของไฟล์ GGUF ที่ระบุ (ไม่มี vocab) · เทสที่ยืนยันกับของพวกนี้จึงพังเมื่อเกณฑ์ของเราเลิกตรงกับ repo จริง
ไม่ใช่เมื่อ fixture ที่แต่งเองเลิกตรงกับโค้ด

ทุกพาธที่ถูกขอถูกจดไว้ — `weight_requests` ใช้ยืนยันว่าไม่มีใครแตะไฟล์ weight เกินการอ่าน header
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from urllib.parse import unquote

import httpx

from lmds.inspector import HfClient, inspect_model
from lmds.resolver import parse_source

FIXTURES = Path(__file__).parent / "hub_fixtures"


def load(repo: str) -> dict:
    return json.loads((FIXTURES / (repo.replace("/", "--") + ".json")).read_text())


def _gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(value) -> bytes:
    """ค่าหนึ่งตัวในรูป (type u32 + payload) ของ GGUF v3 — ครอบชนิดที่ header จริงใน fixture ใช้"""
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 4) + struct.pack("<I", value) if 0 <= value < 2**32 else (
            struct.pack("<I", 11) + struct.pack("<q", value))
    if isinstance(value, float):
        return struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _gguf_string(value)
    if isinstance(value, list):
        head = struct.pack("<I", 9)
        if all(isinstance(v, bool) for v in value):
            return head + struct.pack("<I", 7) + struct.pack("<Q", len(value)) + b"".join(
                struct.pack("<?", v) for v in value)
        if all(isinstance(v, int) for v in value):
            return head + struct.pack("<I", 5) + struct.pack("<Q", len(value)) + b"".join(
                struct.pack("<i", v) for v in value)
        return head + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + b"".join(
            _gguf_string(str(v)) for v in value)
    raise TypeError(f"ชนิดที่ไม่รองรับใน fixture GGUF: {type(value).__name__}")


def gguf_bytes(header: dict) -> bytes:
    """header ของไฟล์ GGUF จาก fixture → ไบต์ที่ parser ตัวจริงอ่านได้ (ไม่มี tensor data)"""
    metadata = header["metadata"]
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", header.get("tensor_count", 0))
    out += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        out += _gguf_string(key) + _gguf_value(value)
    return out


class RealHub:
    """เสิร์ฟ fixture ของ repo ที่ระบุ · `renamed={"ชื่อเก่า": "ชื่อใหม่"}` = Hub redirect ชื่อที่ถูกย้าย"""

    def __init__(self, *repos: str, renamed: dict[str, str] | None = None, edits: dict | None = None):
        self.fixtures = {repo.lower(): load(repo) for repo in repos}
        for old, new in (renamed or {}).items():
            self.fixtures[old.lower()] = self.fixtures[new.lower()]
        # edits = แก้ fixture เฉพาะเทสนั้น (เช่น ตัดไฟล์ออกหนึ่งตัว) โดยไม่แตะไฟล์ของจริง
        for repo, change in (edits or {}).items():
            fixture = json.loads(json.dumps(self.fixtures[repo.lower()]))
            change(fixture)
            self.fixtures[repo.lower()] = fixture
        self.paths: list[str] = []
        self.ranges: list[tuple[str, str]] = []

    def _fixture(self, repo: str) -> dict | None:
        return self.fixtures.get(repo.lower())

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = unquote(request.url.path)
        self.paths.append(path)
        if path.startswith("/api/models/"):
            repo = "/".join(path.removeprefix("/api/models/").split("/")[:2])
            fixture = self._fixture(repo)
            if fixture is None:
                # Hub ตอบ 401 ให้ repo ที่ไม่มีอยู่จริงเมื่อไม่ได้ล็อกอิน — ไม่มี x-error-code (วัดจริง 2026-10-06)
                return httpx.Response(401, headers={"x-error-message": "Invalid username or password."},
                                      json={"error": "Invalid username or password."})
            return httpx.Response(200, json=fixture["info"])
        parts = path.lstrip("/").split("/")
        repo, name = "/".join(parts[:2]), "/".join(parts[4:])
        fixture = self._fixture(repo)
        if fixture is None or parts[2] != "resolve":
            return httpx.Response(404, headers={"x-error-code": "EntryNotFound"})
        header = (fixture.get("gguf") or {}).get(name)
        if header is not None:
            self.ranges.append((name, request.headers.get("Range", "")))
            data = gguf_bytes(header)
            start, _, end = request.headers.get("Range", "bytes=0-").removeprefix("bytes=").partition("-")
            return httpx.Response(206, content=data[int(start): int(end) + 1 if end else None])
        files = fixture.get("files") or {}
        if name in files:
            body = files[name]
            return httpx.Response(200, content=(body if isinstance(body, str) else json.dumps(body)).encode())
        return httpx.Response(404, headers={"x-error-code": "EntryNotFound"})

    def client(self, token=None) -> HfClient:
        return HfClient(token=token, client=httpx.Client(transport=httpx.MockTransport(self.handler)))

    def inspect(self, source: str):
        return inspect_model(parse_source(source), self.client())

    @property
    def weight_requests(self) -> list[str]:
        """ไฟล์ weight ที่ถูกขอ *เกิน* การอ่าน header ของ GGUF — ต้องว่างเสมอ"""
        headers = {name for fixture in self.fixtures.values() for name in (fixture.get("gguf") or {})}
        return [p for p in self.paths
                if p.endswith((".safetensors", ".bin")) or (p.endswith(".gguf") and not any(p.endswith(h) for h in headers))]


def flat(text: str) -> str:
    """rich ตัดบรรทัดตามความกว้างจอ — เทียบข้อความโดยไม่ขึ้นกับว่าตัดตรงไหน"""
    return "".join(text.split())


def shard_files_of(controller_text: str) -> list[str]:
    """รายการ SHARD_FILES ที่ controller จะตรวจ/โหลด — ให้ bash อ่าน array เอง ไม่ grep ข้อความ"""
    import subprocess

    start = controller_text.index("SHARD_FILES=(")
    end = controller_text.index(")", start) + 1
    script = controller_text[start:end] + '\nprintf "%s\\n" "${SHARD_FILES[@]}"\n'
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return [line for line in done.stdout.splitlines() if line]
