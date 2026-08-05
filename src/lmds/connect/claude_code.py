"""สร้างค่าตั้ง Claude Code จาก endpoint ที่รันอยู่ — และตรวจก่อนว่าใช้ได้จริง

ต่อด้วยมือพลาดได้สี่จุดที่อาการเหมือนกันหมด (Claude Code บอกว่าต่อไม่ได้) แต่คนละสาเหตุ:

- ใส่ `/v1` ต่อท้าย base URL → client เติม `/v1/messages` เองกลายเป็น `/v1/v1/messages`
- ตั้งชื่อโมเดลแค่ช่องเดียว → งานเบื้องหลัง (haiku) และ subagent ยิงชื่อโมเดลของ Anthropic
  มาที่เครื่องเรา แล้ว engine ที่ตรวจชื่อโมเดล (vLLM) ตอบ 404
- ใส่ token ผิดตัวแปร → `ANTHROPIC_AUTH_TOKEN` ไป header `Authorization: Bearer`
  ส่วน `ANTHROPIC_API_KEY` ไป `x-api-key` · ผิดช่องคือ 401
- ไม่ลดเพดาน context/output → โดนตัดกลางบทสนทนา เพราะค่าเริ่มต้นของ Claude Code
  สูงกว่าโมเดล local ทั่วไป

โมดูลนี้ไม่ยุ่งกับ terminal เลย — CLI และหน้าเว็บเรียกใช้ร่วมกันได้
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ANTHROPIC_VERSION = "2023-06-01"
PROBE_TIMEOUT = 60.0

# ตัวแปรที่ผู้ใช้ตั้งตอน start controller — ใช้ชื่อเดียวกันเพื่อไม่ให้ต้องจำสองชื่อ
KEY_ENV_VAR = "API_KEY"
TOKEN_ENV_KEY = "ANTHROPIC_AUTH_TOKEN"

# Claude Code clamp ค่านี้ไว้ที่อย่างน้อย 100,000 — โมเดล local ส่วนใหญ่ context เล็กกว่านั้น
# ใส่ไปก็ถูกดันขึ้นเป็น 100,000 เท่ากับไม่มีผล จึงไม่ใส่ให้ดีกว่าใส่บรรทัดที่หลอกผู้ใช้
AUTO_COMPACT_MIN = 100_000

# Claude Code ไม่ได้ยิงโมเดลเดียว — สี่ช่องนี้ต้องชี้โมเดลเดียวกันทั้งหมด
MODEL_ENV_KEYS = (
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


class ConnectError(Exception):
    pass


@dataclass
class ProbeResult:
    """แยก "ต่อไม่ได้" ออกจาก "ต่อได้แต่ใช้งานจริงไม่ได้" เพราะวิธีแก้คนละทาง"""

    messages_ok: bool = False
    tools_ok: bool = False
    detail: str = ""
    sample: str = ""


@dataclass
class ClaudeCodeConfig:
    base_url: str  # origin เปล่า ๆ ไม่มี /v1 — client เติม /v1/messages ให้เอง
    model: str
    context: int = 0
    max_output: int = 0
    api_key: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @property
    def needs_token(self) -> bool:
        return bool(self.api_key)

    @property
    def compact_hint(self) -> str:
        """ข้อความอธิบายเมื่อ context เล็กเกินกว่าจะบอก Claude Code ได้"""
        if not self.context or self.context >= AUTO_COMPACT_MIN:
            return ""
        return (
            f"context ของโมเดลนี้ {self.context:,} tokens ซึ่งต่ำกว่าขั้นต่ำ "
            f"{AUTO_COMPACT_MIN:,} ที่ Claude Code ยอมรับ — บอกให้ compact เองอัตโนมัติไม่ได้ "
            "ใช้ /compact ในเซสชันเมื่อบทสนทนายาว"
        )


def _origin(base_url: str, port: int) -> str:
    """ตัด /v1 ท้าย base_url ของผิว OpenAI ให้เหลือ origin

    bundle รุ่นก่อนที่ client-config ยังไม่มี anthropic_base_url ก็ยังใช้ได้
    """
    url = (base_url or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or f"http://127.0.0.1:{port}"


def build_config(client_config: dict, port: int = 0, api_key: str = "") -> ClaudeCodeConfig:
    """แปลงผลของ `client-config` เป็นค่าตั้งของ Claude Code"""
    model = (client_config.get("model") or "").strip()
    if not model:
        raise ConnectError("client-config ไม่มีชื่อโมเดล — bundle เสียหรือเก่าเกินไป")

    base = client_config.get("anthropic_base_url") or _origin(client_config.get("base_url", ""), port)

    # llama.cpp ใช้ server_context_total (แบ่งต่อ slot) ส่วน vLLM ใช้ server_context
    context = int(client_config.get("server_context_total") or client_config.get("server_context") or 0)
    max_output = int(client_config.get("max_output_tokens") or 0)

    key = api_key or client_config.get("api_key") or ""
    if key == "not-required":
        key = ""

    env = {"ANTHROPIC_BASE_URL": base}
    if key:
        # AUTH_TOKEN ไป Authorization: Bearer ซึ่งไม่ต้องกดอนุมัติตอนเปิด Claude Code
        # ต่างจาก ANTHROPIC_API_KEY ที่ต้องอนุมัติหนึ่งครั้ง — เลือกตัวที่สะดุดน้อยกว่า
        env[TOKEN_ENV_KEY] = key
    for name in MODEL_ENV_KEYS:
        env[name] = model
    if context >= AUTO_COMPACT_MIN:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(context)
    if max_output:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output)

    return ClaudeCodeConfig(
        base_url=base, model=model, context=context, max_output=max_output, api_key=key, env=env
    )


def env_lines(config: ClaudeCodeConfig, *, literal_token: bool = False) -> list[str]:
    """บล็อก export สำหรับ copy ไปวาง

    ค่า token อ้าง `$API_KEY` ไม่ใช่ค่าจริง — จะได้ไม่มี secret ขึ้นจอหรือค้างใน
    ประวัติเชลล์ · ผู้ใช้ต้องตั้ง API_KEY อยู่แล้วตอน start controller จึงเป็นค่าที่มีอยู่
    """
    lines = []
    for name, value in config.env.items():
        if name == TOKEN_ENV_KEY and not literal_token:
            lines.append(f'export {name}="${KEY_ENV_VAR}"')
        else:
            lines.append(f"export {name}={value}")
    return lines


def _headers(config: ClaudeCodeConfig) -> dict[str, str]:
    headers = {"anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    if config.api_key:
        headers["x-api-key"] = config.api_key
    return headers


def _text_blocks(payload: dict) -> str:
    """ต่อเฉพาะ block ชนิด text — โมเดลสาย reasoning ส่ง block ชนิด thinking มาด้วย
    ซึ่งไม่มี key "text" (อ่าน content[0].text ตรง ๆ จะพัง)
    """
    return "".join(
        block.get("text") or ""
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def probe_endpoint(config: ClaudeCodeConfig, client: httpx.Client | None = None) -> ProbeResult:
    """ยิงจริงสองครั้ง: ตอบข้อความได้ไหม และเรียก tool เป็นไหม

    ตรวจ tool ด้วยเพราะ Claude Code ใช้ tool แทบทุกเทิร์น — endpoint ที่ตอบข้อความได้
    แต่ไม่ออก tool_use block จะ "ต่อติดแต่ทำงานไม่ได้" ซึ่งหาสาเหตุยากกว่าต่อไม่ติด
    """
    http = client or httpx.Client(timeout=PROBE_TIMEOUT)
    url = f"{config.base_url}/v1/messages"
    result = ProbeResult()

    try:
        resp = http.post(
            url,
            headers=_headers(config),
            json={
                "model": config.model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "ตอบสั้น ๆ ว่า OK"}],
            },
        )
    except httpx.HTTPError as exc:
        result.detail = f"ต่อ {url} ไม่ได้ — {type(exc).__name__}: {exc}"
        return result

    if resp.status_code == 404:
        result.detail = "engine ที่รันอยู่ไม่มี /v1/messages — ต้องอัปเดต image ของ engine ก่อน"
        return result
    if resp.status_code in (401, 403):
        result.detail = (
            f"token ไม่ผ่าน (HTTP {resp.status_code}) — "
            f"รันใหม่ในเชลล์ที่ตั้ง {KEY_ENV_VAR} ไว้ หรือส่ง key ทาง --stdin"
        )
        return result
    if resp.status_code != 200:
        result.detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return result

    try:
        body = resp.json()
    except ValueError as exc:
        result.detail = f"ตอบกลับไม่ใช่ JSON ({exc}) — มี proxy คั่นอยู่หรือเปล่า"
        return result

    result.messages_ok = True
    result.sample = _text_blocks(body)[:120]

    try:
        tool_resp = http.post(
            url,
            headers=_headers(config),
            json={
                "model": config.model,
                "max_tokens": 512,
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file from disk",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Read /etc/hostname using the tool."}],
            },
        )
    except httpx.HTTPError:
        return result  # ตอบข้อความได้แล้ว — แค่ตรวจ tool ไม่สำเร็จ ไม่ใช่ต่อไม่ได้

    if tool_resp.status_code == 200:
        try:
            blocks = tool_resp.json().get("content") or []
        except ValueError:
            return result
        result.tools_ok = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
    return result


def settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def write_settings(config: ClaudeCodeConfig, path: Path | None = None) -> tuple[Path, Path | None]:
    """รวมค่าลง env ของ settings.json โดยไม่แตะคีย์อื่น — คืน (ไฟล์, ไฟล์สำรอง)

    สำรองของเดิมก่อนเสมอเพราะไฟล์นี้เป็นของผู้ใช้ ไม่ใช่ของเรา · ไฟล์นี้เก็บ token
    เป็นค่าจริง (Claude Code อ่านโดยไม่ผ่านเชลล์) จึงห้ามเอาไป commit
    """
    target = path or settings_path()
    backup: Path | None = None

    data: dict = {}
    if target.exists():
        raw = target.read_text(encoding="utf-8")
        try:
            data = json.loads(raw) if raw.strip() else {}
        except ValueError as exc:
            raise ConnectError(
                f"{target} ไม่ใช่ JSON ที่อ่านได้ ({exc}) — แก้ไฟล์ก่อน หรือ copy บล็อก export เอง"
            )
        if not isinstance(data, dict):
            raise ConnectError(f"{target} ระดับบนสุดไม่ใช่ object — ไม่เขียนทับให้")
        backup = target.with_name(target.name + ".lmds-bak")
        backup.write_text(raw, encoding="utf-8")

    env = data.get("env")
    if env is not None and not isinstance(env, dict):
        raise ConnectError(f"{target} มี env ที่ไม่ใช่ object — ไม่เขียนทับให้")
    data["env"] = {**(env or {}), **config.env}

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.chmod(0o600)  # มี token อยู่ข้างใน
    return target, backup
