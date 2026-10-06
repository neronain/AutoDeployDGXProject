"""liveness watchdog — ยิง generate จริง ถ้าไม่ตอบถึงจะ restart · **opt-in เท่านั้น**

## ปัญหาที่แก้

`/health` ตอบ 200 ได้ทั้งที่ rank ค้าง (`docs/UPGRADE-2026-09.md` §1.9 ข้อ 2) · ผู้ดูแลเห็นไฟเขียว
ผู้ใช้เห็นคำขอค้าง และไม่มีใครรู้ว่าต้องกดอะไร · ของที่พิสูจน์ได้ว่า forward pass ยังเดินมีอย่างเดียว
คือ **สั่ง generate แล้วได้ token กลับมา** — 2 token greedy พอ

`manager._health_ok()` จงใจไม่ทำแบบนี้เพราะมันถูกเรียกทุกไม่กี่วินาทีจากทั้ง CLI และหน้าเว็บ
(ดูคอมเมนต์ของมัน: /health ของ SGLang รันโมเดลจริงทุกครั้ง → GPU 78% โดยไม่มีใครถามอะไร) ·
ที่นี่จึงเป็น **คนละตัว คนละจังหวะ**: ทุก 120 วินาที และเฉพาะ slug ที่ถูกสั่งเปิดไว้เท่านั้น

## ทำไมต้อง opt-in และทำไมต้องมีเพดาน

`README.md` (หัวข้อ `lmds adopt`) ตั้งใจให้ LMDS คุมเครื่องที่ **มีโมเดลของลูกค้ารันอยู่ก่อนแล้ว** ·
การมีอะไรบางอย่างในเครื่องที่ restart โมเดลของลูกค้าได้เองโดยไม่มีใครสั่ง เป็นเรื่องใหญ่กว่าการ
ปล่อยให้ค้างแล้วมีคนมาเห็นเอง · ข้อบังคับที่ตามมา:

- ต้องสั่งเปิดต่อ slug (`lmds watchdog arm <slug>`) ไม่มีอะไรเปิดเองทั้งสิ้น
- **ปฏิเสธ** ตั้งแต่ตอน arm: container ที่ไม่ได้มาจาก LMDS (`external`) · bundle ที่ `lmds adopt`
  รับเข้ามา (เว้นแต่สั่ง `--allow-adopted` เอง) · ตัวที่ไม่มีทะเบียน ·
  โมเดล embedding/rerank (ยิง generate ใส่มันไม่มีความหมาย ผลคือ restart เพราะเราถามผิด)
- **คนสั่ง stop = watchdog พัก** จนกว่าคนจะ start/restart เอง (หรือเห็นว่าโมเดลกลับมารันแล้ว) —
  ของที่ปลุกโมเดลซึ่งถูกสั่งหยุดโดยตั้งใจขึ้นมาใหม่ ไม่ใช่ watchdog แต่คือสิ่งที่ต้องมี watchdog ไว้กัน
- **ต้องพลาดติดกันหลายรอบ** ถึงจะนับว่าตาย ไม่ใช่รอบเดียว (GC pause / เน็ตสะดุดมีจริง)
- **เพดานจำนวนครั้งในกรอบเวลา** แล้วหยุดถาวร (`gave_up`) — ยังตรวจต่อ ยังรายงานต่อ แต่ไม่ restart อีก
  · โมเดลที่ start ไม่ขึ้นแล้วถูก restart ทุก 2 นาทีตลอดคืนคือความเสียหายที่ watchdog สร้างเอง
- **backoff** ระหว่างครั้ง และ **settle** หลัง restart (โมเดลใหญ่โหลดเป็นสิบนาที — ยิงใส่ระหว่างนั้น
  แล้วนับว่าตายคือการสร้างลูป)
- **ลง audit ทุกครั้งพร้อมเหตุผล** — คำถามแรกเวลามีอะไรผิดปกติคือ "ใครสั่ง" และคำตอบ
  "ระบบสั่งเอง" ต้องหาเจอที่เดียวกับคำสั่งของคน (`lmds audit`)

## 4xx = ยังไม่ตาย

เซิร์ฟเวอร์ที่ตอบ 404 (ไม่มี `/v1/completions`) หรือ 401 (ต้องใช้ key) **ตอบอยู่** — นั่นคือสิ่งที่
watchdog ถาม · restart ตรงนั้นคือการ restart เพราะ *เราถามผิด* ไม่ใช่เพราะมันพัง · เคสนั้นรายงานว่า
`misconfigured` แล้วดังไว้ ให้คนไปแก้ probe ไม่ใช่ไปรีสตาร์ตโมเดลของลูกค้า

## รูปแบบการรัน

แกนคือ `lmds watchdog run <slug>` ซึ่งเป็นลูป foreground ธรรมดา · `arm` เขียนไฟล์สถานะแล้วติดตั้ง
**systemd user service** ให้ถ้ามี — แต่ `docs` ของเราเองบอกว่าเครื่องลูกค้าบางรายอยู่ใน LXC/Docker
ที่ไม่มี init system เต็ม จึงต้องใช้งานได้โดยไม่มี systemd ด้วย (บอกคำสั่งให้ไปรันเองใต้ตัวคุม
process อะไรก็ได้) · เหมือนที่ `enable_autostart` เลือก user scope เป็นค่าเริ่มต้นเพราะ SSH ไม่มี tty
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .manager import FleetError, ServerInfo, bundle_profile, controller_startup_timeout

# ── ค่าเริ่มต้น ───────────────────────────────────────────────────────────────
# 120 วิ มาจาก §10 / §1.9 ข้อ 2 ("เขายิง 2-token greedy ทุก 120 วิ")
DEFAULT_INTERVAL = 120
# 3 ครั้งติด × 120 วิ ≈ 6 นาทีกว่าจะตัดสินว่าตาย — นานพอที่ GC/สลับ batch ใหญ่จะไม่โดน
DEFAULT_FAILURES = 3
# 3 ครั้งใน 6 ชั่วโมง: มากพอสำหรับอาการค้างชั่วคราวที่ restart แล้วหาย
# น้อยพอที่โมเดลซึ่ง start ไม่ขึ้นจะไม่ถูกวนทั้งคืน
DEFAULT_MAX_RESTARTS = 3
DEFAULT_WINDOW = 6 * 3600
DEFAULT_BACKOFF = 300
# โมเดลใหญ่/stacked โหลด 10-100 นาที (`controller_startup_timeout` ของจริงเคยอ่านได้ 6906 วิ)
# `arm()` ดึงค่าจาก controller มาทับให้เมื่อ bundle บอกไว้
DEFAULT_SETTLE = 900
DEFAULT_PROBE_TIMEOUT = 30

PROMPT = "ping"
MAX_TOKENS = 2      # "2-token greedy" — พอที่จะพิสูจน์ว่า forward pass เดินจริง


def root() -> Path:
    """ที่เก็บสถานะ watchdog ของเครื่องนี้ — `$LMDS_WATCHDOG_ROOT` ทับได้ (เทสใช้)"""
    return Path(os.environ.get("LMDS_WATCHDOG_ROOT", Path.home() / ".lmds" / "watchdog"))


@dataclass
class Policy:
    interval: int = DEFAULT_INTERVAL
    probe_timeout: int = DEFAULT_PROBE_TIMEOUT
    failures_before_restart: int = DEFAULT_FAILURES
    max_restarts: int = DEFAULT_MAX_RESTARTS
    window_seconds: int = DEFAULT_WINDOW
    backoff_seconds: int = DEFAULT_BACKOFF
    settle_seconds: int = DEFAULT_SETTLE


@dataclass
class State:
    """สิ่งที่ต้องอยู่รอดข้ามการ restart ของตัว watchdog เอง

    ถ้าไม่เก็บลงดิสก์ การ restart ตัว watchdog (หรือ reboot) จะล้างเพดานทิ้ง แล้วเครื่องที่
    ถูกสั่งหยุดไปแล้วเพราะเกินโควตาก็กลับมาวนใหม่ — ซึ่งคือสิ่งที่เพดานมีไว้กัน
    """

    slug: str = ""
    armed: bool = False
    armed_at: str = ""
    armed_by: str = ""
    restarts: list[dict] = field(default_factory=list)   # [{"at": epoch, "reason": str, "ok": bool}]
    gave_up_at: float = 0.0
    consecutive_failures: int = 0
    settle_until: float = 0.0
    last_probe_at: float = 0.0
    last_ok_at: float = 0.0
    last_detail: str = ""
    policy: dict = field(default_factory=lambda: asdict(Policy()))
    # bundle ที่ `lmds adopt` รับเข้ามา: เปิดได้ก็ต่อเมื่อคนสั่ง `--allow-adopted` เอง และต้องจำไว้ —
    # ลูปเช็คซ้ำทุกรอบ (สถานะที่ arm ไว้ก่อนมีกติกานี้ไม่มีค่านี้ จึงไม่ถูก restart ต่อเงียบ ๆ)
    allow_adopted: bool = False
    # เหตุที่ลูป "ควร restart แต่ไม่ทำ" ครั้งล่าสุด — ว่าง = ไม่มี · ให้ status บอกได้โดยไม่ต้องเดา
    blocked_reason: str = ""
    # คนสั่งหยุดโมเดลนี้เอง — ลูปไม่ยิง ไม่ restart จนกว่าคนจะ start/restart (ดู `operator_event`)
    paused_at: float = 0.0
    paused_by: str = ""
    # True เมื่อยืนยันแล้วว่าโมเดลหยุดจริง (คำสั่ง stop จบสำเร็จ หรือลูปเห็นเองว่าไม่ได้รัน) ·
    # ระหว่างที่ยังเป็น False คือ "กำลังหยุด" — เห็นว่ายังรันอยู่ไม่ได้แปลว่ามีคน start ใหม่
    paused_down: bool = False

    @property
    def gave_up(self) -> bool:
        return bool(self.gave_up_at)


def path_for(slug: str) -> Path:
    from .apikey import path_for as key_path      # ยืมกติกา slug เดียวกัน ไม่ตั้งใหม่ให้ต่างกัน

    return root() / (key_path(slug).name + ".json")


def load(slug: str) -> State:
    """สถานะของ slug นี้ — ไม่มีไฟล์/ไฟล์เสีย = ยังไม่เคยเปิด (ไม่ใช่ error)"""
    try:
        raw = json.loads(path_for(slug).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return State(slug=slug)
    if not isinstance(raw, dict):
        return State(slug=slug)
    known = set(State.__dataclass_fields__)
    fields = {k: v for k, v in raw.items() if k in known}
    # slug มาจากผู้เรียกเสมอ ไม่ใช่จากไฟล์ — ไฟล์ที่ถูกก๊อปข้ามชื่อมาต้องไม่พาเพดานของอีกตัวมาด้วย
    fields["slug"] = slug
    return State(**fields)


def save(state: State) -> Path:
    target = path_for(state.slug)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        pass
    # เขียนไฟล์ใหม่แล้วค่อยสลับ — ไฟล์นี้ถือเพดาน restart ไว้ ถ้าไฟดับกลางเขียนแล้วได้ JSON
    # ครึ่งท่อน `load()` จะอ่านเป็น "ยังไม่เคยเปิด" = เพดานหาย = วนได้อีกรอบ
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def policy_of(state: State) -> Policy:
    known = {f for f in Policy.__dataclass_fields__}
    return Policy(**{k: v for k, v in (state.policy or {}).items() if k in known})


# ── probe: generate จริง ─────────────────────────────────────────────────────
@dataclass
class ProbeResult:
    ok: bool            # ตอบ 200 และมี token ออกมาจริง
    alive: bool         # ตอบอะไรก็ได้ที่ไม่ใช่ 5xx/ต่อไม่ติด — "ยังไม่ตาย"
    status: int = 0
    detail: str = ""
    ms: int = 0


def probe(endpoint: str, model: str, *, api_key: str = "",
          timeout: int = DEFAULT_PROBE_TIMEOUT, poster=None) -> ProbeResult:
    """ยิง 2-token greedy ไปที่ `<endpoint>/completions` — `poster` แทน httpx ในเทส

    ใช้ `/v1/completions` ไม่ใช่ `/v1/chat/completions` เพราะไม่ต้องพึ่ง chat template
    (bundle ที่ template เพี้ยนจะตอบ 400 ทุกครั้ง แล้ว watchdog จะ restart ทั้งคืนโดยที่
    โมเดลไม่ได้พังเลย) · temperature=0 เพื่อให้ผลไม่ขึ้นกับ sampler และเบาที่สุดเท่าที่จะเป็นได้
    """
    url = endpoint.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
            "temperature": 0, "stream": False}

    started = time.monotonic()
    if poster is None:
        import httpx

        def poster(url, json, headers, timeout):  # noqa: A002 — ชื่อพารามิเตอร์ตาม httpx
            return httpx.post(url, json=json, headers=headers, timeout=timeout)

    try:
        response = poster(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — ต่อไม่ติด/หมดเวลา/DNS — ทุกแบบคือ "ไม่ตอบ"
        return ProbeResult(False, False, 0, f"{type(exc).__name__}: {str(exc)[:160]}",
                           int((time.monotonic() - started) * 1000))

    ms = int((time.monotonic() - started) * 1000)
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 200:
        # 4xx = เซิร์ฟเวอร์ตอบอยู่ (แค่ไม่ชอบคำถามของเรา) · 5xx = engine เองพัง → นับเป็นตาย
        alive = 400 <= status < 500
        return ProbeResult(False, alive, status,
                           ("probe rejected" if alive else "engine error") + f" (HTTP {status})", ms)
    try:
        payload = response.json()
        choices = payload.get("choices") or []
    except Exception:  # noqa: BLE001
        return ProbeResult(False, True, status, "200 but the body is not JSON", ms)
    if not choices:
        # ตอบ 200 โดยไม่มี choice = ไม่ได้ generate อะไรเลย ซึ่งคืออาการที่ /health มองไม่เห็นพอดี
        return ProbeResult(False, False, status, "200 with no completion in the body", ms)
    return ProbeResult(True, True, status, "", ms)


# ── การตัดสินใจ (บริสุทธิ์ — ไม่แตะเวลาจริง ไม่แตะเครื่อง) ────────────────────
def decide(state: State, policy: Policy, result: ProbeResult, now: float) -> dict:
    """คืน `{"action", "reason"}` — มี 7 แบบ และมีแบบเดียวที่แตะโมเดล

    `ok` · `misconfigured` (ตอบอยู่แต่ปฏิเสธ probe) · `settling` (เพิ่ง restart ยังโหลดอยู่) ·
    `waiting` (พลาดแล้วแต่ยังไม่ครบจำนวนครั้ง) · `backoff` · `restart` · `gave-up`

    แยกออกมาเป็นฟังก์ชันบริสุทธิ์เพราะนี่คือส่วนที่ตัดสินใจ restart โมเดลของลูกค้า — มันต้อง
    เทสได้ครบทุกทางโดยไม่ต้องมีโมเดล ไม่ต้องรอเวลาจริง และไม่ต้อง mock ตัวเอง
    """
    if result.ok:
        return {"action": "ok", "reason": ""}
    if result.alive:
        # ตอบอยู่แต่ไม่ยอม generate — เราถามผิด ไม่ใช่มันพัง · ห้ามนับเป็นความล้มเหลว
        return {"action": "misconfigured", "reason": result.detail}
    if now < state.settle_until:
        return {"action": "settling",
                "reason": f"{int(state.settle_until - now)}s of post-restart settle left"}

    failures = state.consecutive_failures + 1
    if failures < policy.failures_before_restart:
        return {"action": "waiting",
                "reason": f"{failures}/{policy.failures_before_restart} consecutive misses"}
    if state.gave_up:
        return {"action": "gave-up", "reason": "restart budget already spent — not restarting again"}

    recent = [r for r in state.restarts if now - float(r.get("at") or 0) <= policy.window_seconds]
    if len(recent) >= policy.max_restarts:
        return {"action": "gave-up",
                "reason": (f"{len(recent)} restarts in the last {policy.window_seconds // 3600}h "
                           f"did not fix it — stopping so it does not loop")}
    last = max((float(r.get("at") or 0) for r in recent), default=0.0)
    # backoff โตเป็นเท่าตัวตามจำนวนครั้งที่ทำไปแล้วในกรอบนี้: หลังครั้งที่ 1 รอ backoff_seconds
    # หลังครั้งที่ 2 รอสองเท่า หลังครั้งที่ 3 รอสี่เท่า — อาการที่ restart แล้วไม่หายจะยิ่งห่างขึ้น
    # เรื่อย ๆ แทนที่จะถี่เท่าเดิมจนหมดโควตาภายในไม่กี่นาที
    wait = policy.backoff_seconds * (2 ** max(0, len(recent) - 1))
    if last and now - last < wait:
        return {"action": "backoff", "reason": f"{int(wait - (now - last))}s before another restart"}
    return {"action": "restart",
            "reason": f"{failures} consecutive 2-token probes got no answer: {result.detail}"}


def advance(state: State, decision: dict, result: ProbeResult, now: float, *,
            restart_ok: bool | None = None, policy: Policy | None = None) -> State:
    """เขียนผลของรอบนี้กลับเข้า state — ตัวเดียวที่แก้ตัวนับ เพื่อให้กติกาอยู่ที่เดียว"""
    policy = policy or policy_of(state)
    state.last_probe_at = now
    state.last_detail = result.detail
    action = decision["action"]
    if action in {"ok", "misconfigured"}:
        state.consecutive_failures = 0
        if action == "ok":
            state.last_ok_at = now
            state.blocked_reason = ""
        return state
    if action == "settling":
        return state
    state.consecutive_failures += 1
    if action == "blocked":
        state.blocked_reason = decision["reason"]
    if action == "gave-up" and not state.gave_up_at:
        state.gave_up_at = now
    if action == "restart":
        entry = {"at": now, "reason": decision["reason"], "ok": bool(restart_ok)}
        if not restart_ok and decision.get("error"):
            entry["error"] = decision["error"]
        state.restarts.append(entry)
        # ตัดของเก่าทิ้งเพื่อไม่ให้ไฟล์โตไม่หยุด — เก็บเผื่อไว้สองเท่าของเพดานเพื่อให้ยังไล่ย้อนได้
        state.restarts = state.restarts[-max(10, policy.max_restarts * 2):]
        # settle มีไว้ให้โมเดลที่ *เพิ่งถูก restart* โหลดให้จบ · restart ที่ล้มไม่มีอะไรกำลังโหลด —
        # หยุดยิงไป 15 นาทีขึ้นไปตรงนั้นคือการไม่เฝ้าโมเดลที่ยังพังอยู่ (audit 2026-10-06) ·
        # ครั้งที่ล้มยังนับเข้าโควตาและยังต้องรอ backoff เหมือนเดิม จึงไม่วนถี่ขึ้น
        if restart_ok:
            state.settle_until = now + policy.settle_seconds
        state.consecutive_failures = 0
    return state


# ── arm / disarm ─────────────────────────────────────────────────────────────
ALLOW_ADOPTED_FLAG = "--allow-adopted"


def is_adopted(info: ServerInfo) -> bool:
    """bundle นี้มาจาก `lmds adopt` ไหม — container/process ที่ลูกค้าตั้งไว้ก่อน LMDS จะเข้าไป

    ดูสองทางเพราะแต่ละทางหายได้: ชื่อ controller (`<slug>-adopted.sh` — กติกาที่ maintainer ใช้แยก
    ด้วยตาเปล่า) และ `adopted: true` / `generated_by: lmds adopt…` ใน MODEL_PROFILE.yaml
    """
    controller = str(info.controller or "")
    if controller.endswith("-adopted.sh"):
        return True
    profile = bundle_profile(controller) if controller else None
    if not profile:
        return False
    return bool(profile.get("adopted")) or str(profile.get("generated_by") or "").startswith("lmds adopt")


def restart_block(info: ServerInfo, state: State | None = None) -> str:
    """เหตุที่ watchdog **ห้าม restart** ตัวนี้เอง แม้สถานะจะบอกว่าเปิดอยู่ — "" = ไม่มี

    เช็คตอน arm อย่างเดียวไม่พอ: สถานะที่เปิดไว้ก่อนมีกติกานี้ และ bundle ที่กลายเป็นของ adopt ทีหลัง
    ยังมีลูปวนอยู่ · ลูปจึงถามคำถามเดียวกันนี้ทุกครั้งก่อนจะ restart
    """
    if info.external:
        return (f"{info.slug} is a container LMDS did not create (adopted from this machine) — "
                f"a watchdog that restarts someone else's model is not ours to arm")
    if is_adopted(info) and not (state is not None and state.allow_adopted):
        return (f"{info.slug} was adopted from a model that was running before LMDS (lmds adopt) — "
                f"a watchdog restart there removes the customer's container (docker rm -f) and "
                f"re-runs a command LMDS regenerated, so it is refused unless the owner of that "
                f"model agreed · explicit opt-in: lmds watchdog arm {info.slug} {ALLOW_ADOPTED_FLAG}")
    return ""


def refusals(info: ServerInfo, *, allow_adopted: bool = False) -> list[str]:
    """เหตุที่ **ไม่ควรให้ watchdog แตะ** slug นี้ — [] = เปิดได้

    ปฏิเสธตั้งแต่ตอน arm ดีกว่าไปค้นพบตอนตีสามว่ามันเพิ่ง restart โมเดลของลูกค้าไปแล้ว

    `external` จับได้เฉพาะ container ที่ **ยังไม่มีทะเบียน** · หลัง `lmds adopt` bundle มีทะเบียนและ
    `external=False` — เดิมข้อห้ามจึงหายไปพอดีตอนที่ของลูกค้าเข้ามาอยู่ในมือเรา (audit 2026-10-06:
    `BEFORE adopt: external=True refusals:[…]` / `AFTER adopt: external=False refusals: []`) ทั้งที่
    restart ของ bundle แบบนั้นคือ `docker rm -f` + รันคำสั่งที่เราเขียนขึ้นใหม่บน container ของลูกค้า
    """
    out: list[str] = []
    blocked = restart_block(info, State(slug=info.slug, allow_adopted=allow_adopted))
    if blocked:
        out.append(blocked)
    if not info.registered:
        out.append(f"{info.slug} has no server.meta — LMDS cannot tell what restarting it would do")
    if not info.controller_exists and not (info.mode == "docker" and info.container):
        out.append(f"{info.slug} has neither a controller nor a container — there is nothing to restart")
    profile = bundle_profile(info.controller) if info.controller else None
    features = (profile or {}).get("features") or {}
    for kind in ("embedding", "rerank"):
        if features.get(kind):
            out.append(f"{info.slug} is an {kind} model — a 2-token generate probe means nothing "
                       f"there, so every probe would 'fail' and every restart would be wrong")
    return out


def warnings_for(info: ServerInfo) -> list[str]:
    """สิ่งที่ควรเห็นก่อนกด แต่ไม่ถึงกับห้าม"""
    out: list[str] = []
    profile = bundle_profile(info.controller) if info.controller else None
    stacked = ((profile or {}).get("topology") == "stacked"
               or str(info.controller or "").endswith("-stacked.sh"))
    if stacked:
        out.append("this bundle is stacked — a restart here takes the whole group down and back up, "
                   "including the workers on the other machines; arm the head only")
    return out


def arm(info: ServerInfo, *, policy: Policy | None = None, actor: str = "",
        allow_adopted: bool = False) -> State:
    """เปิด watchdog ของ slug นี้ — โยน FleetError เมื่อมีเหตุห้าม

    `allow_adopted=True` คือ `lmds watchdog arm <slug> --allow-adopted`: คนสั่งรับรองเองว่าเจ้าของ
    โมเดลที่ adopt มายอมให้ LMDS restart ได้ · ไม่มีผลกับ bundle ที่ไม่ได้ adopt
    """
    stop = refusals(info, allow_adopted=allow_adopted)
    if stop:
        raise FleetError("\n".join(stop))
    state = load(info.slug)
    policy = policy or policy_of(state)
    if policy.settle_seconds == DEFAULT_SETTLE and info.controller:
        # bundle บอกเองว่ามันรอ /health นานแค่ไหน — ใช้ค่านั้นแทนการเดา ไม่งั้นโมเดล 200 GB
        # ที่โหลด 40 นาทีจะถูกยิงใส่ระหว่างโหลดแล้วถูกนับว่าตาย
        floor = controller_startup_timeout(info.controller)
        if floor:
            policy.settle_seconds = max(policy.settle_seconds, floor + 120)
    state.armed = True
    state.armed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    state.armed_by = actor or os.environ.get("USER", "")
    state.policy = asdict(policy)
    state.allow_adopted = bool(allow_adopted and is_adopted(info))
    state.blocked_reason = ""
    # เปิดใหม่ = ให้โควตาใหม่ · คนที่กด arm อีกครั้งคือคนที่ตัดสินใจแล้วว่าให้ลองต่อ
    state.gave_up_at = 0.0
    state.consecutive_failures = 0
    save(state)
    _audit("WATCHDOG-ARM", info.slug, reason=f"interval {policy.interval}s · "
           f"max {policy.max_restarts} restarts / {policy.window_seconds // 3600}h"
           + (f" · adopted bundle — explicit opt-in ({ALLOW_ADOPTED_FLAG})" if state.allow_adopted else ""),
           actor=state.armed_by or "cli")
    return state


def disarm(slug: str, *, actor: str = "") -> State:
    state = load(slug)
    was = state.armed
    state.armed = False
    state.paused_at, state.paused_by, state.paused_down = 0.0, "", False
    save(state)
    if was:
        _audit("WATCHDOG-DISARM", slug, reason="", actor=actor or os.environ.get("USER", "") or "cli")
    return state


# ── คนสั่ง stop/start เอง ─────────────────────────────────────────────────────
# "กำลังหยุด" ที่ค้างนานกว่านี้โดยโมเดลยังรันอยู่ = คำสั่ง stop ตายกลางทาง (kill -9 / ไฟดับ)
# ไม่มีใครมายืนยันหรือยกเลิกให้ — เลิกพักเอง ดีกว่าเป็น watchdog ที่เปิดอยู่ในนามไปตลอด
PAUSE_PENDING_GRACE = 900


def operator_event(slug: str, event: str, *, ok: bool = True, by: str = "",
                   now: float | None = None) -> State | None:
    """คน (ไม่ใช่ watchdog) สั่ง stop/start/restart โมเดลนี้ — คืน state ใหม่ หรือ None เมื่อไม่ได้เปิดไว้

    audit 2026-10-06: ไม่มีอะไรใน `lmds stop` แตะสถานะ watchdog เลย — หลังคนสั่งหยุด ลูปเดิน
    `waiting, waiting, restart` แล้วปลุกโมเดลกลับขึ้นมา · บนเครื่อง unified-memory คนสั่ง stop
    เพื่อเอาหน่วยความจำไปให้โมเดลตัวถัดไป ผลคือสองตัวแย่งที่กันโดยไม่มีใครสั่ง

    เหตุการณ์ (เรียกจาก `manager.stop_server/start_server/restart_server` และงานของหน้าเว็บ):

    - `stop-begin`  พักทันที **ก่อน** เริ่มหยุด — ไม่งั้นรอบที่ยิงระหว่าง `docker stop` นับเป็นพลาด
    - `stop-end`    ok → ยืนยันว่าหยุดแล้ว · ไม่ ok → เลิกพัก (มันไม่ได้หยุด ยังต้องเฝ้าต่อ)
    - `start-begin` เว้นช่วง settle ให้โมเดลโหลด — ไม่งั้น probe ที่พลาดระหว่างโหลดจะพาไป restart ซ้อน
    - `start-end`   ok → กลับมาเฝ้าต่อ · ไม่ ok → คงสภาพเดิม (ที่พักอยู่ก็พักต่อ: โมเดลยังไม่ได้รัน)

    restart ของ watchdog เองไม่ผ่านตรงนี้ (`restart_server(..., operator=False)`) และโมเดลที่ตายเอง
    โดยไม่มีใครสั่ง stop ไม่มีเหตุการณ์ใดเกิดขึ้น — จึงยังถูก restart ตามปกติ
    """
    if not path_for(slug).is_file():
        return None
    state = load(slug)
    if not state.armed:
        return None
    now = time.time() if now is None else now
    who = by or os.environ.get("USER", "") or "operator"
    if event == "stop-begin":
        if not state.paused_at:
            state.paused_at, state.paused_by, state.paused_down = now, who, False
            _audit("WATCHDOG-PAUSE", slug, reason="stopped by operator — not probing or restarting "
                   "until an operator start/restart", actor=who)
    elif event == "stop-end":
        if ok:
            if not state.paused_at:
                state.paused_at, state.paused_by = now, who
            state.paused_down = True
            state.consecutive_failures = 0
        elif state.paused_at and not state.paused_down:
            state.paused_at, state.paused_by = 0.0, ""
            _audit("WATCHDOG-RESUME", slug, reason="the operator's stop failed — the model is still "
                   "running, so it is still watched", actor=who)
    elif event == "start-begin":
        state.settle_until = max(state.settle_until, now + policy_of(state).settle_seconds)
        state.consecutive_failures = 0
    elif event == "start-end":
        if ok and state.paused_at:
            state.paused_at, state.paused_by, state.paused_down = 0.0, "", False
            _audit("WATCHDOG-RESUME", slug, reason="started by operator", actor=who)
    else:
        return state
    save(state)
    return state


def armed_slugs() -> list[str]:
    try:
        files = sorted(root().glob("*.json"))
    except OSError:
        return []
    out = []
    for item in files:
        slug = item.name[: -len(".json")]
        if load(slug).armed:
            out.append(slug)
    return out


# ── ลูป ──────────────────────────────────────────────────────────────────────
def _audit(action: str, slug: str, *, reason: str, actor: str = "watchdog",
           status: int = 200, ms: int = 0) -> None:
    """ลง audit แบบไม่ทำให้ลูปล้ม — audit ที่หายหนึ่งบรรทัดดีกว่า watchdog ที่ตายทั้งตัว"""
    try:
        from lmds.web import audit

        audit.event(action, f"/models/{slug}", actor=actor, reason=reason, status=status, ms=ms)
    except Exception:  # noqa: BLE001 — extra ของเว็บอาจไม่ได้ติดตั้งบน node
        pass


def _stamp(at: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(at)) if at else "-"


def _paused_tick(info: ServerInfo, state: State, policy: Policy, now: float) -> dict | None:
    """รอบของ watchdog ที่พักอยู่ — คืน report เมื่อรอบนี้จบแค่นั้น · None เมื่อเลิกพักแล้วให้เดินต่อ"""
    if not info.running:
        state.paused_down = True
    elif state.paused_down or now - state.paused_at > PAUSE_PENDING_GRACE:
        # หยุดไปแล้วจริง แต่ตอนนี้กลับมารันอยู่ = มีคน start นอกทางของ `lmds start` (เรียก controller
        # ตรง ๆ · autostart ตอนบูต) — โมเดลที่รันอยู่โดยไม่มีใครเฝ้าทั้งที่สถานะบอกว่า "เปิดอยู่"
        # คือความล้มเหลวแบบเดียวกับ service ที่ตายเงียบ · กลับมาเฝ้า และเว้นช่วงให้มันโหลดก่อน
        state.paused_at, state.paused_by, state.paused_down = 0.0, "", False
        state.settle_until = max(state.settle_until, now + policy.settle_seconds)
        state.consecutive_failures = 0
        reason = "the model is running again (started outside lmds) — watching it again"
        _audit("WATCHDOG-RESUME", state.slug, reason=reason)
        return {"action": "resumed", "reason": reason, "probe": None}
    return {"action": "paused", "probe": None,
            "reason": f"stopped by operator at {_stamp(state.paused_at)} — not probing, not restarting · "
                      f"resume: lmds start {state.slug}"}


def _withdrawn(state: State) -> str:
    """คนเพิ่งสั่งหยุด/ปิด **ระหว่างที่รอบนี้กำลังยิง** ไหม — อ่านจากดิสก์ก่อนลงมือ restart

    probe รอได้ถึง `probe_timeout` วินาที · `lmds stop` ที่มาถึงในช่วงนั้นต้องชนะ ไม่ใช่แพ้ให้กับ
    state ที่ลูปอ่านไว้ก่อนหน้า
    """
    if state.paused_at or not path_for(state.slug).is_file():
        return ""
    fresh = load(state.slug)
    if fresh.paused_at:
        return f"an operator stopped {state.slug} while this probe was in flight — not restarting"
    if not fresh.armed:
        return f"the watchdog of {state.slug} was disarmed while this probe was in flight — not restarting"
    return ""


def tick(info: ServerInfo, state: State, policy: Policy, *, now: float,
         prober=None, restarter=None, api_key: str = "") -> tuple[State, dict]:
    """หนึ่งรอบ: ยิง → ตัดสิน → (อาจ) restart → ลง audit → คืน state ใหม่

    แยกจาก `loop()` เพื่อให้เทสเดินทีละรอบได้โดยไม่ต้องรอเวลาจริง — และเพื่อให้ `loop()`
    เหลือแค่ "นอนแล้วเรียก tick" ซึ่งไม่มีตรรกะให้พลาด
    """
    from .manager import restart_server

    prober = prober or (lambda: probe(info.endpoint, info.model or info.slug,
                                      api_key=api_key, timeout=policy.probe_timeout))
    # operator=False: restart ของ watchdog ไม่ใช่การตัดสินใจของคน — ห้ามไปเลิกพัก/ตั้ง settle แทนคน
    restarter = restarter or (lambda: restart_server(info, operator=False))

    if state.paused_at:
        report = _paused_tick(info, state, policy, now)
        if report is not None:
            return state, report

    if now < state.settle_until:
        # ไม่ยิงเลยระหว่าง settle — ยิงแล้วได้ 503 ตอนโมเดลโหลดอยู่ ไม่ได้ให้ข้อมูลอะไรเพิ่ม
        # นอกจากทำให้ตัวนับเดินไปทางที่ผิด
        decision = {"action": "settling",
                    "reason": f"{int(state.settle_until - now)}s of settle left"}
        return state, {**decision, "probe": None}

    result = prober()
    decision = decide(state, policy, result, now)
    restart_ok = None

    if decision["action"] == "restart":
        withdrawn = _withdrawn(state)
        if withdrawn:
            # ไม่แตะตัวนับ — รอบถัดไปลูปอ่านสถานะจากดิสก์ใหม่แล้วเห็นเองว่าพัก/ปิดอยู่
            return state, {"action": "paused", "reason": withdrawn, "probe": result}
        blocked = restart_block(info, state)
        if blocked:
            # ถึงเกณฑ์ restart แล้ว แต่ตัวนี้ห้ามแตะ — ลง audit ครั้งเดียวต่อหนึ่งช่วงที่ล่ม ไม่ใช่ทุกรอบ
            if not state.blocked_reason:
                _audit("WATCHDOG-SKIPPED", state.slug, status=403, ms=result.ms,
                       reason=f"would restart ({decision['reason']}) but refused: {blocked}")
            decision = {"action": "blocked", "reason": blocked}

    if decision["action"] == "restart":
        try:
            restarter()
            restart_ok = True
        except Exception as exc:  # noqa: BLE001 — restart ล้มไม่ควรฆ่า watchdog
            restart_ok = False
            # ข้อความของ `restart_server` พาบรรทัดท้าย ๆ ที่ controller พูดเองมาด้วย — เก็บให้พอเห็นสาเหตุ
            decision = {**decision, "error": f"{type(exc).__name__}: {' · '.join(str(exc).split())[:400]}"}
        _audit("WATCHDOG-RESTART", state.slug,
               reason=decision["reason"] + ("" if restart_ok else f" · restart failed: {decision.get('error', '')}"),
               status=200 if restart_ok else 500, ms=result.ms)
    elif decision["action"] == "gave-up" and not state.gave_up:
        # ครั้งเดียวตอนข้ามเส้น ไม่ใช่ทุกรอบ — ไม่งั้น audit จะเต็มไปด้วยบรรทัดเดียวกันทุก 2 นาที
        _audit("WATCHDOG-GIVE-UP", state.slug, reason=decision["reason"], status=429, ms=result.ms)
    elif decision["action"] == "misconfigured":
        _audit("WATCHDOG-SKIPPED", state.slug,
               reason=f"the server answered but refused the probe: {result.detail} — "
                      f"not restarting; fix the probe instead", status=409, ms=result.ms)

    state = advance(state, decision, result, now, restart_ok=restart_ok, policy=policy)
    return state, {**decision, "probe": result, "restart_ok": restart_ok}


def _commit(state: State, before: dict) -> None:
    """เขียนผลของรอบนี้ลงดิสก์ โดยไม่ทับสิ่งที่ **คนอื่นเขียนระหว่างรอบ**

    ไฟล์สถานะมีผู้เขียนสองฝั่ง: ลูปนี้ กับคำสั่งของคน (`arm` · `disarm` · `lmds stop/start`) ·
    เดิมลูปอ่านครั้งเดียวตอนเริ่มแล้วเขียน state ในหน่วยความจำทับทุกรอบ — `disarm` จึงถูกเขียนทับ
    กลับเป็น armed ในรอบถัดไป (ลูปที่รันด้วย nohup ไม่มีวันหยุด) และการพักจาก `lmds stop` ก็จะหาย
    แบบเดียวกัน · ฟิลด์ไหนบนดิสก์ต่างจากตอนเริ่มรอบ = มีคนเขียน → ของเขาชนะ
    """
    disk = asdict(load(state.slug))
    for name, value in disk.items():
        if name != "slug" and value != before.get(name):
            setattr(state, name, value)
    save(state)


def loop(slug: str, *, info: ServerInfo | None = None, rounds: int = 0, clock=time.time,
         sleeper=time.sleep, prober=None, restarter=None, on_tick=None,
         finder=None) -> State:
    """ลูปหลัก — `rounds=0` คือไม่มีที่สิ้นสุด (ค่าที่ service ใช้) · เทสส่งจำนวนรอบเข้ามา

    หา `ServerInfo` ใหม่ทุกรอบเมื่อไม่ได้ส่งเข้ามา เพราะ port/container เปลี่ยนได้ระหว่างทาง
    (คนสั่ง `lmds start --port` เอง หรือ restart ของเราเองเปลี่ยน container id) · อ่านไฟล์สถานะใหม่
    ทุกรอบด้วยเหตุผลเดียวกัน — คนสั่ง disarm / stop / arm ใหม่ได้ทุกเมื่อ (ดู `_commit`)
    """
    from . import apikey
    from .manager import find

    finder = finder or find
    state = load(slug)
    if not state.armed:
        raise FleetError(f"watchdog ของ {slug} ยังไม่ได้เปิด — สั่ง: lmds watchdog arm {slug}")
    key = ""
    try:
        key = apikey.read(slug)
    except Exception:  # noqa: BLE001 — ไม่มี key ไม่ใช่ความผิดพลาด
        key = ""

    count = 0
    while True:
        state = load(slug)
        if not state.armed:
            # disarm ระหว่างที่ลูปวนอยู่ — หยุดเอง ไม่เขียนอะไรทับ
            return state
        before = asdict(state)
        policy = policy_of(state)
        server = info or finder(slug)
        if server is None:
            # bundle ถูกลบไปแล้ว — หยุดเองอย่างเงียบ ๆ ดีกว่าวนบ่นทุก 2 นาทีตลอดกาล
            _audit("WATCHDOG-STOP", slug, reason="the bundle is gone from this machine", status=410)
            state.armed = False
            save(state)
            return state
        state, report = tick(server, state, policy, now=clock(), prober=prober,
                             restarter=restarter, api_key=key)
        _commit(state, before)
        if on_tick is not None:
            on_tick(report)
        count += 1
        if rounds and count >= rounds:
            return state
        sleeper(policy.interval)


# ── รายงาน ───────────────────────────────────────────────────────────────────
def _ago(seconds: float, lang: str) -> str:
    seconds = max(0, int(seconds))
    if seconds < 120:
        return f"{seconds} วินาทีก่อน" if lang == "th" else f"{seconds}s ago"
    if seconds < 7200:
        return f"{seconds // 60} นาทีก่อน" if lang == "th" else f"{seconds // 60} min ago"
    return f"{seconds // 3600} ชม.ก่อน" if lang == "th" else f"{seconds // 3600}h ago"


def report(state: State, *, info: ServerInfo | None = None, service: dict | None = None,
           now: float | None = None) -> dict:
    """สถานะแบบเครื่องอ่าน — `lmds watchdog status --json` · `describe()` สร้างประโยคจาก dict เดียวกันนี้

    เป็นทุกฟิลด์ของ `State` บวกข้อสรุปที่ **ต้องไม่ปล่อยให้คนอ่านเดาเอง**: ยังไม่เคยมี probe ที่สำเร็จไหม ·
    probe ล่าสุดเลยกำหนดมานานแค่ไหน · พักอยู่ไหม · มีเหตุห้าม restart ไหม · systemd พูดว่าอะไร
    (`service` มาจาก `service_state()` — ไม่ส่งมา = ไม่ได้ถาม ไม่ใช่ "ปกติ")
    """
    now = time.time() if now is None else now
    policy = policy_of(state)
    settling = max(0, int(state.settle_until - now))
    blocked = restart_block(info, state) if info is not None else state.blocked_reason
    since_probe = int(now - state.last_probe_at) if state.last_probe_at else None
    # เลยกำหนด = ไม่มีรอบใหม่มาเกินสามช่วง ทั้งที่ไม่ได้พักและไม่ได้ settle — ลูปไม่ได้รัน
    # (หรือกำลังอยู่กลาง restart ที่ controller รอ health นาน ซึ่งรอบนั้นยังไม่ถูกจด)
    overdue = bool(state.armed and not state.paused_at and not settling and since_probe is not None
                   and since_probe > 3 * policy.interval + policy.probe_timeout)
    failed = [r for r in state.restarts if not r.get("ok")]
    out = asdict(state)
    out.update({
        "gave_up": state.gave_up,
        "paused": bool(state.paused_at),
        "blocked": blocked,
        "settling_seconds": settling,
        "never_probed": not state.last_probe_at,
        "never_succeeded": not state.last_ok_at,
        "seconds_since_last_probe": since_probe,
        "seconds_since_last_ok": int(now - state.last_ok_at) if state.last_ok_at else None,
        "probe_overdue": overdue,
        "restarts_failed": len(failed),
        "service": service,
    })
    return out


def service_lines(service: dict, lang: str) -> list[str]:
    unit, state, detail = service.get("unit", ""), service.get("state", "unknown"), service.get("detail", "")
    why = f" ({detail})" if detail else ""
    user = service.get("user") or "$USER"
    lines: list[str] = []
    if state == "active":
        lines.append(f"service {unit}: active (ถาม systemd แล้ว)" if lang == "th"
                     else f"service {unit}: active (asked systemd)")
    elif state == "unknown":
        lines.append(
            f"service {unit}: unknown — ถาม systemd ไม่ได้{why} · ไม่ได้แปลว่าลูปรันอยู่ ดูเวลาของ probe ล่าสุดแทน"
            if lang == "th" else
            f"service {unit}: unknown — systemd could not be asked{why} · this does not mean the loop "
            f"is running; go by the probe times above")
    elif not service.get("installed"):
        lines.append(
            f"service: ไม่ได้ติดตั้ง ({unit}: {state}) — ไม่มีอะไรรันลูปให้ เว้นแต่รันเอง: lmds watchdog run …"
            if lang == "th" else
            f"service: not installed ({unit}: {state}) — nothing runs the loop unless you started it "
            f"yourself: lmds watchdog run …")
    else:
        lines.append(
            f"⚠ service {unit}: {state} — ลูป **ไม่ได้รันอยู่**{why} · ดู: journalctl --user -u {unit} -n 20"
            if lang == "th" else
            f"⚠ service {unit}: {state} — the loop is NOT running{why} · see: journalctl --user -u {unit} -n 20")
    if service.get("installed") and service.get("linger") == "no":
        lines.append(
            f"⚠ linger ของ {user} ปิดอยู่ — service นี้รันเฉพาะตอนมี session เปิดอยู่ และตายเมื่อ logout · "
            f"แก้: sudo loginctl enable-linger {user}"
            if lang == "th" else
            f"⚠ linger is off for {user} — this service only runs while a login session is open and "
            f"dies at logout · fix: sudo loginctl enable-linger {user}")
    return lines


def describe(state: State, lang: str = "th", now: float | None = None, *,
             info: ServerInfo | None = None, service: dict | None = None) -> list[str]:
    """สถานะเป็นประโยค — ใช้ทั้งใน `lmds watchdog status` และตอนสรุปท้าย arm

    ระหว่าง settle ต้องบอกว่ากำลัง settle **ไม่ใช่โชว์ผล probe ครั้งก่อนค้างไว้** —
    ผลครั้งก่อนคือตอนโมเดลล่ม ซึ่ง restart ไปแล้ว คนอ่านจะนึกว่ายังล่มอยู่ทั้งที่หายแล้ว

    **"เปิดอยู่ · 0 restarts" ไม่ใช่หลักฐานว่ามีใครเฝ้า** — เดิมข้อความของสถานะที่ไม่เคยยิง probe
    สักรอบกับสถานะที่ยิงสำเร็จสามรอบเหมือนกันทุกตัวอักษร (audit 2026-10-06) ซึ่งคือรูปเดียวกับเคส
    msi-4 (2026-09-22: service ตาย 203/EXEC วน 9 รอบ ส่วน status บอก "เปิดอยู่") · จึงพิมพ์เสมอว่า
    probe ล่าสุดรันเมื่อไร สำเร็จล่าสุดเมื่อไร และพูดตรง ๆ เมื่อยังไม่เคยสำเร็จเลย
    """
    now = time.time() if now is None else now
    th = lang == "th"
    policy = policy_of(state)
    facts = report(state, info=info, service=service, now=now)
    recent = len(state.restarts)
    settling = facts["settling_seconds"]

    lines = [f"{'เปิดอยู่' if state.armed else 'ปิดอยู่'} · ยิงทุก {policy.interval} วินาที · "
             f"พลาดติดกัน {policy.failures_before_restart} ครั้งถึงจะ restart · "
             f"restart ได้ไม่เกิน {policy.max_restarts} ครั้ง/{policy.window_seconds // 3600} ชม."
             if th else
             f"{'armed' if state.armed else 'disarmed'} · probe every {policy.interval}s · "
             f"{policy.failures_before_restart} consecutive misses before a restart · "
             f"at most {policy.max_restarts} restarts per {policy.window_seconds // 3600}h"]
    if facts["blocked"]:
        lines.append(("⚠ จะไม่ restart ตัวนี้ให้: " if th else "⚠ will not restart this one: ") + facts["blocked"])
    if facts["paused"]:
        when, who = _stamp(state.paused_at), state.paused_by or "operator"
        if th:
            lines.append(f"พักอยู่ — paused: stopped by operator at {when} (โดย {who}) · ไม่ยิง ไม่ restart "
                         f"จนกว่าจะสั่ง: lmds start {state.slug}")
        else:
            lines.append(f"paused: stopped by operator at {when} (by {who}) — not probing, not restarting "
                         f"until: lmds start {state.slug}")
    if state.gave_up:
        lines.append("เลิก restart แล้ว (เกินโควตา) — ยังตรวจอยู่แต่จะไม่แตะโมเดลอีก · "
                     f"เปิดโควตาใหม่: lmds watchdog arm {state.slug}"
                     if th else
                     "gave up restarting (budget spent) — still probing, will not touch the model "
                     f"again until: lmds watchdog arm {state.slug}")
    if recent:
        last = state.restarts[-1]
        bad = facts["restarts_failed"]
        if th:
            line = f"restart อัตโนมัติไปแล้ว {recent} ครั้ง" + (f" (ล้มเหลว {bad} ครั้ง)" if bad else "")
            line += " · ครั้งล่าสุด" + ("" if last.get("ok") else " **ล้มเหลว**") + f": {last.get('reason', '')}"
        else:
            line = f"{recent} automatic restarts so far" + (f" ({bad} failed)" if bad else "")
            line += " · last" + ("" if last.get("ok") else " FAILED") + f": {last.get('reason', '')}"
        if not last.get("ok") and last.get("error"):
            line += f" — {last['error']}"
        lines.append(line)
    if settling:
        lines.append(f"เพิ่ง restart ไป — พักให้โมเดลตั้งตัวก่อน เหลืออีก {settling} วินาทีจึงจะยิงรอบใหม่"
                     if th else
                     f"just restarted — letting the model settle, {settling}s before the next probe")
    elif state.last_detail:
        lines.append(f"รอบล่าสุด: {state.last_detail}" if th else f"last round: {state.last_detail}")

    if facts["never_probed"]:
        lines.append("ยังไม่เคยยิง probe สักรอบ — ยังไม่มี probe ที่สำเร็จ (no probe has succeeded yet) · "
                     "ลูปยังไม่เคยรัน หรือรันไม่ขึ้น"
                     if th else
                     "no probe has run yet — no probe has succeeded yet · the loop has never run, or cannot start")
    else:
        probe_at = f"{_stamp(state.last_probe_at)} ({_ago(facts['seconds_since_last_probe'], lang)})"
        if facts["never_succeeded"]:
            lines.append(f"probe ล่าสุด {probe_at} · ยังไม่เคยมี probe ที่สำเร็จ (no probe has succeeded yet)"
                         if th else f"last probe {probe_at} · no probe has succeeded yet")
        else:
            ok_at = f"{_stamp(state.last_ok_at)} ({_ago(facts['seconds_since_last_ok'], lang)})"
            lines.append(f"probe ล่าสุด {probe_at} · สำเร็จล่าสุด {ok_at}"
                         if th else f"last probe {probe_at} · last successful probe {ok_at}")
        if facts["probe_overdue"]:
            lines.append(f"⚠ ไม่มี probe ใหม่มานานกว่า 3 รอบ (ยิงทุก {policy.interval} วินาที) — ลูปไม่ได้รันอยู่ "
                         "หรือกำลังค้างอยู่กลาง restart"
                         if th else
                         f"⚠ no new probe for more than 3 intervals (every {policy.interval}s) — the loop is "
                         "not running, or is in the middle of a restart")
    if service is not None:
        lines += service_lines(service, lang)
    return lines


def linger_state() -> str:
    """`yes` | `no` | `unknown` — linger ของ user นี้ (`loginctl show-user <user> -p Linger`)

    ไม่มี linger = user manager ของ systemd ถูกหยุดเมื่อ session สุดท้ายปิด และ service ของ user
    ตายไปด้วย · watchdog ที่ติดตั้งผ่าน SSH (`lmds node run <เครื่อง> watchdog arm … --service`) จึงตาย
    ตอนสาย SSH หลุด แล้ว **กลับมาเองทุกครั้งที่มีคน SSH เข้าไปดู** (unit ยัง enabled อยู่) — คนที่เข้าไป
    เช็คจึงเห็น `active` เสมอ ทั้งที่ตลอดคืนไม่มีใครเฝ้า
    """
    import getpass
    import subprocess

    try:
        proc = subprocess.run(["loginctl", "show-user", getpass.getuser(), "-p", "Linger"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired, KeyError):
        return "unknown"
    if "Linger=yes" in proc.stdout:
        return "yes"
    # ไม่มี session และไม่ได้ linger: loginctl ตอบว่าไม่รู้จัก user นี้ — ซึ่งแปลว่าไม่ได้ linger
    if "Linger=no" in proc.stdout or "not logged in or lingering" in (proc.stderr or ""):
        return "no"
    return "unknown"


_UNIT_STATES = {"active", "inactive", "failed", "activating", "deactivating", "reloading"}


def service_state(slug: str) -> dict:
    """systemd พูดว่า service ของ watchdog นี้เป็นอย่างไร — `{unit, installed, state, detail, linger, user}`

    `state` คือคำตอบของ `systemctl --user is-active <unit>` · ถามไม่ได้ (ไม่มี systemctl · ต่อ user bus
    ไม่ได้ · หมดเวลา) = `unknown` พร้อมเหตุผลใน `detail` — **ไม่เดาว่าปกติ** · ไฟล์สถานะบนดิสก์บอกได้แค่ว่า
    "เคยสั่งเปิด" ไม่ได้บอกว่ามี process วนอยู่ (เคส msi-4 2026-09-22: 203/EXEC เก้ารอบ ส่วน status
    ที่อ่านแต่ไฟล์บอกว่า "เปิดอยู่")
    """
    import getpass
    import subprocess

    from .manager import have_systemctl, user_systemd_dir

    name = unit_name(slug)
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    out = {"unit": name, "installed": (user_systemd_dir() / name).is_file(),
           "state": "unknown", "detail": "", "linger": "unknown", "user": user}
    if not have_systemctl():
        out["detail"] = "no systemctl on this machine"
        return out
    try:
        proc = subprocess.run(["systemctl", "--user", "is-active", name],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        out["detail"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return out
    word = (proc.stdout.strip().splitlines() or [""])[0].strip()
    if word in _UNIT_STATES:
        out["state"] = word
    else:
        # "Failed to connect to bus" ฯลฯ — systemd ไม่ได้ตอบคำถาม ไม่ใช่ตอบว่า inactive
        out["detail"] = ((proc.stderr or "").strip().splitlines() or [word or f"exit {proc.returncode}"])[0][:200]
        return out
    if out["state"] != "active":
        # ทำไมถึงไม่ active — Result/ExecMainStatus/NRestarts คือสามค่าที่บอกเคส 203/EXEC ได้ในบรรทัดเดียว
        try:
            show = subprocess.run(["systemctl", "--user", "show", name, "-p", "Result",
                                   "-p", "ExecMainStatus", "-p", "NRestarts"],
                                  capture_output=True, text=True, timeout=10)
            facts = [line.strip() for line in show.stdout.splitlines() if "=" in line and line.strip()]
            out["detail"] = " · ".join(facts)[:200]
        except (OSError, subprocess.TimeoutExpired):
            pass
    out["linger"] = linger_state()
    return out


def unit_name(slug: str) -> str:
    return f"lmds-watchdog-{slug}.service"


def lmds_path() -> str:
    """พาธเต็มของคำสั่ง `lmds` บนเครื่องนี้

    **systemd ไม่ค้น `$PATH` ให้** — `ExecStart=lmds ...` จึงล้มด้วย `status=203/EXEC`
    ทุกครั้ง แล้ว `Restart=always` ก็พามันวนใหม่ทุก 30 วินาทีโดยไม่เคยรันสำเร็จเลย ·
    เจอจริงบน msi-4 (2026-09-22): unit วนไป 9 รอบ ส่วน `watchdog status` ยังรายงานว่า
    "เปิดอยู่" — **ดูเหมือนทำงาน แต่ไม่เคยยิง probe สักครั้ง** ซึ่งเป็นความล้มเหลวแบบที่
    แย่ที่สุดสำหรับฟีเจอร์นี้: เฝ้าอยู่ในนาม แต่ของจริงไม่มีใครเฝ้า

    หาจาก interpreter ที่กำลังรันอยู่ก่อน เพราะ console script อยู่ข้าง ๆ กันเสมอ
    """
    beside = Path(sys.executable).with_name("lmds")
    if beside.exists():
        return str(beside)
    found = shutil.which("lmds")
    if found:
        return str(Path(found).resolve())
    # ไม่เจอ console script = เรียกผ่าน interpreter ตรง ๆ · ยังเป็นพาธเต็มเหมือนกัน
    return f"{sys.executable} -m lmds.cli.main"


def manual_command(slug: str, executable: str = "") -> str:
    """คำสั่งให้ไปรันเองใต้ตัวคุม process อะไรก็ได้ — เครื่องที่ไม่มี systemd ใช้ทางนี้

    `docs` ของเราเองระบุว่าลูกค้าบางรายรันใน LXC/Docker ที่ไม่มี init system เต็ม · ถ้าฟีเจอร์นี้
    ผูกกับ systemd อย่างเดียว เครื่องกลุ่มนั้นจะไม่มีทางใช้ได้เลย ทั้งที่ลูปมันเป็นแค่ foreground process
    """
    return (f"nohup {executable or lmds_path()} watchdog run {slug} "
            f">> ~/.lmds/watchdog-{slug}.log 2>&1 &")


def install_service(slug: str, executable: str = "") -> str:
    """ติดตั้ง + enable + start systemd **user** service — คืนชื่อ unit

    โยน `FleetError` เมื่อไม่มี systemd โดยแนบคำสั่งทางเลือกไปด้วย ไม่ใช่แค่บอกว่าทำไม่ได้
    """
    import subprocess

    from .manager import have_systemctl, user_systemd_dir

    if not have_systemctl():
        raise FleetError(
            "เครื่องนี้ไม่มี systemd (systemctl) — watchdog ยังใช้ได้ แต่ต้องให้ตัวคุม process "
            f"ของเครื่องนี้เป็นคนดูแลแทน:\n    {manual_command(slug, executable)}")
    if linger_state() == "no":
        # ตรวจ **ก่อน** เขียน unit — ติดตั้งไปแล้วค่อยเตือนคือทิ้ง service ที่ตายพร้อมสาย SSH ไว้ให้
        # โดยที่สถานะบอกว่าเปิดอยู่ (ดู `linger_state`) · `unknown` ปล่อยผ่าน: ถามไม่ได้ไม่ใช่เหตุให้ห้าม
        import getpass

        user = getpass.getuser()
        raise FleetError(
            f"ยังไม่ติดตั้ง watchdog service ให้ — linger ของ {user} ปิดอยู่ (loginctl show-user {user} -p Linger)\n"
            f"systemd user service จะตายพร้อม session นี้ (สาย SSH หลุด/logout) ทั้งที่สถานะยังบอกว่าเปิดอยู่\n"
            f"เปิด linger ครั้งเดียวบนเครื่องนี้:\n    sudo loginctl enable-linger {user}\n"
            f"แล้วสั่งอีกครั้ง:\n    lmds watchdog arm {slug} --service\n"
            f"หรือให้ตัวคุม process อื่นของเครื่องนี้รันลูปแทน:\n    {manual_command(slug, executable)}")
    name = unit_name(slug)
    directory = user_systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(render_unit(slug, executable), encoding="utf-8")
    for cmd in (["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", name]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FleetError(f"ติดตั้ง watchdog service ไม่สำเร็จ: {' '.join(cmd)}\n"
                             f"{(proc.stderr or '').strip()[:300]}")
    return name


def remove_service(slug: str) -> str:
    """ถอน service ออก — เงียบเมื่อไม่เคยติดตั้ง (disarm ต้องทำงานได้เสมอ)"""
    import subprocess

    from .manager import have_systemctl, user_systemd_dir

    name = unit_name(slug)
    if have_systemctl():
        subprocess.run(["systemctl", "--user", "disable", "--now", name],
                       capture_output=True, text=True)
    (user_systemd_dir() / name).unlink(missing_ok=True)
    return name


def render_unit(slug: str, executable: str = "") -> str:
    """systemd **user** unit ของ watchdog — ไม่ต้อง sudo ด้วยเหตุผลเดียวกับ `manager.render_unit`

    ต่างจาก unit ของโมเดลตรงที่อันนี้เป็น `Type=simple` + `Restart=always`: ตัว watchdog เองตาย
    ต้องกลับมาเอง · **แต่นั่นไม่ใช่การเลี่ยงเพดาน** เพราะเพดานอยู่ในไฟล์สถานะบนดิสก์ ไม่ได้อยู่ใน
    หน่วยความจำของ process (ดู `State`)
    """
    return "\n".join([
        "[Unit]",
        f"Description=LMDS liveness watchdog: {slug}",
        "After=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={executable or lmds_path()} watchdog run {slug}",
        "Restart=always",
        "RestartSec=30",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])
