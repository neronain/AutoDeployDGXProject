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
import os
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ANTHROPIC_VERSION = "2023-06-01"
PROBE_TIMEOUT = 60.0

# ตัวแปรที่ผู้ใช้ตั้งตอน start controller — ใช้ชื่อเดียวกันเพื่อไม่ให้ต้องจำสองชื่อ
KEY_ENV_VAR = "API_KEY"
TOKEN_ENV_KEY = "ANTHROPIC_AUTH_TOKEN"
NO_AUTH_TOKEN = "lmds-local-no-key"

# Claude Code clamp ค่านี้ไว้ที่อย่างน้อย 100,000 — โมเดล local ส่วนใหญ่ context เล็กกว่านั้น
# ใส่ไปก็ถูกดันขึ้นเป็น 100,000 เท่ากับไม่มีผล จึงไม่ใส่ให้ดีกว่าใส่บรรทัดที่หลอกผู้ใช้
AUTO_COMPACT_MIN = 100_000

# Claude Code มี main/alias/fallback/subagent หลายช่อง — ทุกช่องต้องชี้ local model เดียวกัน
MODEL_ENV_KEYS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)

# Claude Code เลือก provider กลุ่มนี้ก่อน ANTHROPIC_BASE_URL. บล็อก shell ต้อง unset และ
# --write ต้องเอาค่าที่ค้างใน user settings ออก ไม่เช่นนั้นค่าต่อ local ดูถูกแต่ client ไป cloud จริง
PROVIDER_ENV_KEYS = (
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_VERTEX",
)

# คีย์ที่คำสั่งนี้เป็นเจ้าของ ต้องลบค่ารอบเก่าก่อน merge รอบใหม่ เช่น compact-window ของ
# โมเดล 128k ต้องไม่ค้างเมื่อสลับไปโมเดล 32k
MANAGED_ENV_KEYS = frozenset(
    (*MODEL_ENV_KEYS, *PROVIDER_ENV_KEYS, "ANTHROPIC_BASE_URL", TOKEN_ENV_KEY,
     "ANTHROPIC_SMALL_FAST_MODEL",
     "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
     "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
     "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK", "CLAUDE_CODE_DISABLE_THINKING",
     "DISABLE_PROMPT_CACHING", "MAX_THINKING_TOKENS")
)

# export block ต้องล้างค่า LMDS รอบเก่าที่ config รอบใหม่ไม่ได้เขียนด้วย (เช่น compact
# ของโมเดลใหญ่) และล้าง x-api-key เพื่อให้ส่ง credential แบบ Bearer เพียงทางเดียว
SHELL_UNSET_KEYS = tuple(sorted((*MANAGED_ENV_KEYS, "ANTHROPIC_API_KEY")))


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
    if base_url is not None and not isinstance(base_url, str):
        raise ConnectError("client-config มี base URL ที่ไม่ใช่ข้อความ")
    url = (base_url or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    url = url or f"http://127.0.0.1:{port}"
    try:
        parsed = urlsplit(url)
        parsed.port  # validate malformed/out-of-range ports too
    except ValueError as exc:
        raise ConnectError(f"client-config มี base URL ที่ใช้ไม่ได้: {url!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in url)
    ):
        raise ConnectError(f"client-config มี base URL ที่ใช้ไม่ได้: {url!r}")
    return url


def _positive_int(client_config: dict, *names: str) -> int:
    for name in names:
        value = client_config.get(name)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            raise ConnectError(f"client-config มี {name} ที่ไม่ใช่จำนวนเต็มบวก")
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.strip().isdigit():
            number = int(value.strip())
        else:
            raise ConnectError(f"client-config มี {name} ที่ไม่ใช่จำนวนเต็มบวก")
        if number <= 0:
            raise ConnectError(f"client-config มี {name} ที่ไม่ใช่จำนวนเต็มบวก")
        return number
    return 0


def build_config(client_config: dict, port: int = 0, api_key: str = "") -> ClaudeCodeConfig:
    """แปลงผลของ `client-config` เป็นค่าตั้งของ Claude Code"""
    raw_model = client_config.get("model")
    model = raw_model.strip() if isinstance(raw_model, str) else ""
    if not model or any(ord(ch) < 32 or ord(ch) == 127 for ch in model):
        raise ConnectError("client-config ไม่มีชื่อโมเดล — bundle เสียหรือเก่าเกินไป")

    raw_base = client_config.get("anthropic_base_url") or client_config.get("base_url", "")
    base = _origin(raw_base, port)

    # max_input_tokens คือ budget ที่ controller หัก output/template overhead แล้วและปลอดภัยสุด
    # สำหรับ llama.cpp ต้องใช้ context_per_slot ไม่ใช่ server_context_total ซึ่งถูกหารให้หลาย slot
    context = _positive_int(
        client_config, "max_input_tokens", "context_per_slot", "server_context", "server_context_total"
    )
    max_output = _positive_int(client_config, "max_output_tokens")

    key = api_key or client_config.get("api_key") or ""
    if not isinstance(key, str):
        raise ConnectError("client-config มี API key ที่ไม่ใช่ข้อความ")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
        raise ConnectError("client-config มี API key ที่ใช้เป็น HTTP header ไม่ได้")
    if key == "not-required":
        key = ""

    env = {
        "ANTHROPIC_BASE_URL": base,
        # ต้องตั้ง credential แม้ endpoint ไม่บังคับ key มิฉะนั้น Claude Code ที่ login อยู่จะส่ง
        # credential subscription ไปยัง custom base URL. ค่าคงที่นี้ไม่ใช่ secret และ server no-auth ignore
        TOKEN_ENV_KEY: key or NO_AUTH_TOKEN,
        # local vLLM/llama.cpp เป็น Anthropic-compatible subset ไม่ใช่ Claude API เต็มรูปแบบ
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        # ถ้า stream ขาดกลาง tool call ห้าม fallback ไปยิง non-streaming ซ้ำ เพราะอาจทำ tool ซ้ำ
        "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
        # ตัวใหม่ตรงกว่า MAX_THINKING_TOKENS=0; เก็บทั้งคู่เพื่อรองรับ Claude Code รุ่นก่อนหน้า
        "CLAUDE_CODE_DISABLE_THINKING": "1",
        "DISABLE_PROMPT_CACHING": "1",
        "MAX_THINKING_TOKENS": "0",
    }
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

    endpoint ที่มี token จะอ้าง `$API_KEY` ไม่ใช่ค่าจริง — จะได้ไม่มี secret ขึ้นจอหรือค้างใน
    ประวัติเชลล์ · endpoint no-auth พิมพ์ dummy credential ที่ไม่ใช่ secret ได้ตรง ๆ
    """
    lines = ["unset " + " ".join(SHELL_UNSET_KEYS)]
    for name, value in config.env.items():
        if name == TOKEN_ENV_KEY and config.needs_token and not literal_token:
            lines.append(f'export {name}="${KEY_ENV_VAR}"')
        else:
            lines.append(f"export {name}={shlex.quote(value)}")
    return lines


def _headers(config: ClaudeCodeConfig) -> dict[str, str]:
    headers = {"anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    # Claude Code ใช้ ANTHROPIC_AUTH_TOKEN เป็น Bearer และ vLLM's VLLM_API_KEY ป้องกัน /v1
    # ด้วย Bearer เช่นกัน จึง probe auth path เดียวกับ client จริง ไม่ใช่ x-api-key คนละทาง
    headers["authorization"] = f"Bearer {config.api_key or NO_AUTH_TOKEN}"
    return headers


def _text_blocks(payload: dict) -> str:
    """ต่อเฉพาะ block ชนิด text — โมเดลสาย reasoning ส่ง block ชนิด thinking มาด้วย
    ซึ่งไม่มี key "text" (อ่าน content[0].text ตรง ๆ จะพัง)
    """
    return "".join(
        block.get("text") or ""
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") in {"text", "text_delta"}
    ).strip()


def _sse_blocks(response: httpx.Response) -> list[dict]:
    """อ่าน content blocks/deltas จาก Anthropic SSE; Claude Code ใช้ streaming จริง"""
    content_type = response.headers.get("content-type", "").lower()
    if not content_type.startswith("text/event-stream"):
        raise ConnectError(f"stream probe ได้ Content-Type {content_type or '(ไม่มี)'}")
    blocks: list[dict] = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except ValueError as exc:
            raise ConnectError("stream probe มี SSE data ที่ไม่ใช่ JSON") from exc
        if not isinstance(event, dict):
            continue
        block = event.get("content_block")
        if isinstance(block, dict):
            blocks.append(block)
        delta = event.get("delta")
        if isinstance(delta, dict):
            blocks.append(delta)
    return blocks


def _safe_detail(text: str, config: ClaudeCodeConfig, limit: int = 200) -> str:
    """ข้อความ server เป็น untrusted; ห้ามสะท้อน bearer token กลับ terminal"""
    value = text
    if config.api_key:
        value = value.replace(config.api_key, "***")
    value = value[:limit]
    # Rich markup ถูกปิดที่ caller แล้ว แต่ C0/DEL (โดยเฉพาะ ESC) ยังควบคุม terminal ได้
    return "".join(ch if ord(ch) >= 32 and ord(ch) != 127 else " " for ch in value)


def _probe_with_client(config: ClaudeCodeConfig, http: httpx.Client) -> ProbeResult:
    # Claude Code ส่ง inference ที่ path นี้พร้อม beta=true และต้องรับ SSE stream
    url = f"{config.base_url}/v1/messages?beta=true"
    result = ProbeResult()

    try:
        resp = http.post(
            url,
            headers=_headers(config),
            json={
                "model": config.model,
                "max_tokens": 256,
                "stream": True,
                "messages": [{"role": "user", "content": "ตอบสั้น ๆ ว่า OK"}],
            },
        )
    except httpx.HTTPError as exc:
        result.detail = _safe_detail(f"ต่อ {url} ไม่ได้ — {type(exc).__name__}: {exc}", config)
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
        result.detail = f"HTTP {resp.status_code}: {_safe_detail(resp.text, config)}"
        return result

    try:
        blocks = _sse_blocks(resp)
    except ConnectError as exc:
        result.detail = str(exc)
        return result

    result.sample = _safe_detail(_text_blocks({"content": blocks}), config, 120)
    if not result.sample:
        result.detail = "HTTP 200 แต่ไม่มี text block — ยังพิสูจน์ไม่ได้ว่าโมเดลตอบข้อความได้"
        return result
    result.messages_ok = True

    try:
        tool_resp = http.post(
            url,
            headers=_headers(config),
            json={
                "model": config.model,
                "max_tokens": 512,
                "stream": True,
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
                "tool_choice": {"type": "tool", "name": "read_file"},
                "messages": [{"role": "user", "content": "Read /etc/hostname using the tool."}],
            },
        )
    except httpx.HTTPError as exc:
        result.detail = _safe_detail(f"ตรวจ tool ไม่สำเร็จ — {type(exc).__name__}: {exc}", config)
        return result

    if tool_resp.status_code == 200:
        try:
            blocks = _sse_blocks(tool_resp)
        except ConnectError as exc:
            result.detail = str(exc)
            return result
        result.tools_ok = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
        if not result.tools_ok:
            result.detail = "tool probe ไม่คืน tool_use block แม้บังคับ tool_choice แล้ว"
    else:
        result.detail = (
            f"tool probe ตอบ HTTP {tool_resp.status_code}: "
            f"{_safe_detail(tool_resp.text, config)}"
        )
    return result


def probe_endpoint(config: ClaudeCodeConfig, client: httpx.Client | None = None) -> ProbeResult:
    """ยิงจริงสองครั้ง: ตอบ SSE ได้ไหม และ forced tool call ได้ไหม

    ตรวจ tool ด้วยเพราะ Claude Code ใช้ tool แทบทุกเทิร์น — endpoint ที่ตอบข้อความได้
    แต่ไม่ออก tool_use block จะ "ต่อติดแต่ทำงานไม่ได้" ซึ่งหาสาเหตุยากกว่าต่อไม่ติด
    """
    owned = client is None
    http = client or httpx.Client(timeout=PROBE_TIMEOUT)
    try:
        return _probe_with_client(config, http)
    finally:
        if owned:
            http.close()


def settings_path() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".claude"
    return root / "settings.json"


def _write_private_atomic(target: Path, text: str) -> None:
    """เขียนไฟล์ 0600 ใน directory เดียวแล้ว os.replace; failure ไม่แตะไฟล์เดิม"""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.lmds-", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_settings(config: ClaudeCodeConfig, path: Path | None = None) -> tuple[Path, Path | None]:
    """รวมค่าลง env ของ settings.json โดยไม่แตะคีย์อื่น — คืน (ไฟล์, ไฟล์สำรอง)

    สำรองของเดิมก่อนเสมอเพราะไฟล์นี้เป็นของผู้ใช้ ไม่ใช่ของเรา · ไฟล์นี้เก็บ token
    เป็นค่าจริง (Claude Code อ่านโดยไม่ผ่านเชลล์) จึงห้ามเอาไป commit
    """
    target = path or settings_path()
    effective_target = target.resolve(strict=False) if target.is_symlink() else target
    backup: Path | None = None
    original_raw: str | None = None

    data: dict = {}
    if target.exists():
        if not target.is_file():
            raise ConnectError(f"{target} ไม่ใช่ regular file — ไม่เขียนทับให้")
        raw = target.read_text(encoding="utf-8")
        original_raw = raw
        try:
            data = json.loads(raw) if raw.strip() else {}
        except ValueError as exc:
            raise ConnectError(
                f"{target} ไม่ใช่ JSON ที่อ่านได้ ({exc}) — แก้ไฟล์ก่อน หรือ copy บล็อก export เอง"
            )
        if not isinstance(data, dict):
            raise ConnectError(f"{target} ระดับบนสุดไม่ใช่ object — ไม่เขียนทับให้")
        # ชื่อไม่ซ้ำเพื่อให้ connect หลายครั้งไม่ทำลาย backup ต้นฉบับ; mode 0600 เพราะ
        # settings เดิมอาจมี token/credential ของผู้ใช้
        backup = target.with_name(f"{target.name}.lmds-bak.{time.time_ns()}")
        _write_private_atomic(backup, raw)

    env = data.get("env")
    if env is not None and not isinstance(env, dict):
        raise ConnectError(f"{target} มี env ที่ไม่ใช่ object — ไม่เขียนทับให้")
    merged_env = dict(env or {})
    for name in MANAGED_ENV_KEYS:
        merged_env.pop(name, None)
    merged_env.update(config.env)
    data["env"] = merged_env

    # อย่าทับการแก้ของ Claude Code/editor ที่เกิดหลังเราอ่านไฟล์และสร้าง backup
    if original_raw is None:
        if target.exists():
            raise ConnectError(f"{target} ถูกสร้างขึ้นระหว่างคำสั่ง — ไม่เขียนทับให้")
    elif not target.exists() or target.read_text(encoding="utf-8") != original_raw:
        raise ConnectError(f"{target} เปลี่ยนระหว่างคำสั่ง — ไม่เขียนทับให้")

    _write_private_atomic(
        effective_target, json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    return target, backup
