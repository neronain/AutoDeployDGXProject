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
- **ปฏิเสธ** ตั้งแต่ตอน arm: container ที่ไม่ได้มาจาก LMDS (`external`) · ตัวที่ไม่มีทะเบียน ·
  โมเดล embedding/rerank (ยิง generate ใส่มันไม่มีความหมาย ผลคือ restart เพราะเราถามผิด)
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
        return state
    if action == "settling":
        return state
    state.consecutive_failures += 1
    if action == "gave-up" and not state.gave_up_at:
        state.gave_up_at = now
    if action == "restart":
        state.restarts.append({"at": now, "reason": decision["reason"], "ok": bool(restart_ok)})
        # ตัดของเก่าทิ้งเพื่อไม่ให้ไฟล์โตไม่หยุด — เก็บเผื่อไว้สองเท่าของเพดานเพื่อให้ยังไล่ย้อนได้
        state.restarts = state.restarts[-max(10, policy.max_restarts * 2):]
        state.settle_until = now + policy.settle_seconds
        state.consecutive_failures = 0
    return state


# ── arm / disarm ─────────────────────────────────────────────────────────────
def refusals(info: ServerInfo) -> list[str]:
    """เหตุที่ **ไม่ควรให้ watchdog แตะ** slug นี้ — [] = เปิดได้

    ปฏิเสธตั้งแต่ตอน arm ดีกว่าไปค้นพบตอนตีสามว่ามันเพิ่ง restart โมเดลของลูกค้าไปแล้ว
    """
    out: list[str] = []
    if info.external:
        out.append(f"{info.slug} is a container LMDS did not create (adopted from this machine) — "
                   f"a watchdog that restarts someone else's model is not ours to arm")
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


def arm(info: ServerInfo, *, policy: Policy | None = None, actor: str = "") -> State:
    """เปิด watchdog ของ slug นี้ — โยน FleetError เมื่อมีเหตุห้าม"""
    stop = refusals(info)
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
    # เปิดใหม่ = ให้โควตาใหม่ · คนที่กด arm อีกครั้งคือคนที่ตัดสินใจแล้วว่าให้ลองต่อ
    state.gave_up_at = 0.0
    state.consecutive_failures = 0
    save(state)
    _audit("WATCHDOG-ARM", info.slug, reason=f"interval {policy.interval}s · "
           f"max {policy.max_restarts} restarts / {policy.window_seconds // 3600}h",
           actor=state.armed_by or "cli")
    return state


def disarm(slug: str, *, actor: str = "") -> State:
    state = load(slug)
    was = state.armed
    state.armed = False
    save(state)
    if was:
        _audit("WATCHDOG-DISARM", slug, reason="", actor=actor or os.environ.get("USER", "") or "cli")
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


def tick(info: ServerInfo, state: State, policy: Policy, *, now: float,
         prober=None, restarter=None, api_key: str = "") -> tuple[State, dict]:
    """หนึ่งรอบ: ยิง → ตัดสิน → (อาจ) restart → ลง audit → คืน state ใหม่

    แยกจาก `loop()` เพื่อให้เทสเดินทีละรอบได้โดยไม่ต้องรอเวลาจริง — และเพื่อให้ `loop()`
    เหลือแค่ "นอนแล้วเรียก tick" ซึ่งไม่มีตรรกะให้พลาด
    """
    from .manager import restart_server

    prober = prober or (lambda: probe(info.endpoint, info.model or info.slug,
                                      api_key=api_key, timeout=policy.probe_timeout))
    restarter = restarter or (lambda: restart_server(info))

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
        try:
            restarter()
            restart_ok = True
        except Exception as exc:  # noqa: BLE001 — restart ล้มไม่ควรฆ่า watchdog
            restart_ok = False
            decision = {**decision, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
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


def loop(slug: str, *, info: ServerInfo | None = None, rounds: int = 0, clock=time.time,
         sleeper=time.sleep, prober=None, restarter=None, on_tick=None,
         finder=None) -> State:
    """ลูปหลัก — `rounds=0` คือไม่มีที่สิ้นสุด (ค่าที่ service ใช้) · เทสส่งจำนวนรอบเข้ามา

    หา `ServerInfo` ใหม่ทุกรอบเมื่อไม่ได้ส่งเข้ามา เพราะ port/container เปลี่ยนได้ระหว่างทาง
    (คนสั่ง `lmds start --port` เอง หรือ restart ของเราเองเปลี่ยน container id)
    """
    from . import apikey
    from .manager import find

    finder = finder or find
    state = load(slug)
    if not state.armed:
        raise FleetError(f"watchdog ของ {slug} ยังไม่ได้เปิด — สั่ง: lmds watchdog arm {slug}")
    policy = policy_of(state)
    key = ""
    try:
        key = apikey.read(slug)
    except Exception:  # noqa: BLE001 — ไม่มี key ไม่ใช่ความผิดพลาด
        key = ""

    count = 0
    while True:
        server = info or finder(slug)
        if server is None:
            # bundle ถูกลบไปแล้ว — หยุดเองอย่างเงียบ ๆ ดีกว่าวนบ่นทุก 2 นาทีตลอดกาล
            _audit("WATCHDOG-STOP", slug, reason="the bundle is gone from this machine", status=410)
            state.armed = False
            save(state)
            return state
        state, report = tick(server, state, policy, now=clock(), prober=prober,
                             restarter=restarter, api_key=key)
        save(state)
        if on_tick is not None:
            on_tick(report)
        count += 1
        if rounds and count >= rounds:
            return state
        sleeper(policy.interval)


# ── รายงาน ───────────────────────────────────────────────────────────────────
def describe(state: State, lang: str = "th") -> list[str]:
    """สถานะเป็นประโยค — ใช้ทั้งใน `lmds watchdog status` และตอนสรุปท้าย arm"""
    policy = policy_of(state)
    recent = len(state.restarts)
    if lang == "th":
        lines = [f"{'เปิดอยู่' if state.armed else 'ปิดอยู่'} · ยิงทุก {policy.interval} วินาที · "
                 f"พลาดติดกัน {policy.failures_before_restart} ครั้งถึงจะ restart · "
                 f"restart ได้ไม่เกิน {policy.max_restarts} ครั้ง/{policy.window_seconds // 3600} ชม."]
        if state.gave_up:
            lines.append("เลิก restart แล้ว (เกินโควตา) — ยังตรวจอยู่แต่จะไม่แตะโมเดลอีก · "
                         f"เปิดโควตาใหม่: lmds watchdog arm {state.slug}")
        if recent:
            last = state.restarts[-1]
            lines.append(f"restart อัตโนมัติไปแล้ว {recent} ครั้ง · ครั้งล่าสุด: {last.get('reason', '')}")
        if state.last_detail:
            lines.append(f"รอบล่าสุด: {state.last_detail}")
        return lines
    lines = [f"{'armed' if state.armed else 'disarmed'} · probe every {policy.interval}s · "
             f"{policy.failures_before_restart} consecutive misses before a restart · "
             f"at most {policy.max_restarts} restarts per {policy.window_seconds // 3600}h"]
    if state.gave_up:
        lines.append("gave up restarting (budget spent) — still probing, will not touch the model "
                     f"again until: lmds watchdog arm {state.slug}")
    if recent:
        lines.append(f"{recent} automatic restarts so far · last: {state.restarts[-1].get('reason', '')}")
    if state.last_detail:
        lines.append(f"last round: {state.last_detail}")
    return lines


def unit_name(slug: str) -> str:
    return f"lmds-watchdog-{slug}.service"


def manual_command(slug: str, executable: str = "lmds") -> str:
    """คำสั่งให้ไปรันเองใต้ตัวคุม process อะไรก็ได้ — เครื่องที่ไม่มี systemd ใช้ทางนี้

    `docs` ของเราเองระบุว่าลูกค้าบางรายรันใน LXC/Docker ที่ไม่มี init system เต็ม · ถ้าฟีเจอร์นี้
    ผูกกับ systemd อย่างเดียว เครื่องกลุ่มนั้นจะไม่มีทางใช้ได้เลย ทั้งที่ลูปมันเป็นแค่ foreground process
    """
    return f"nohup {executable} watchdog run {slug} >> ~/.lmds/watchdog-{slug}.log 2>&1 &"


def install_service(slug: str, executable: str = "lmds") -> str:
    """ติดตั้ง + enable + start systemd **user** service — คืนชื่อ unit

    โยน `FleetError` เมื่อไม่มี systemd โดยแนบคำสั่งทางเลือกไปด้วย ไม่ใช่แค่บอกว่าทำไม่ได้
    """
    import subprocess

    from .manager import have_systemctl, user_systemd_dir

    if not have_systemctl():
        raise FleetError(
            "เครื่องนี้ไม่มี systemd (systemctl) — watchdog ยังใช้ได้ แต่ต้องให้ตัวคุม process "
            f"ของเครื่องนี้เป็นคนดูแลแทน:\n    {manual_command(slug, executable)}")
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


def render_unit(slug: str, executable: str = "lmds") -> str:
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
        f"ExecStart={executable} watchdog run {slug}",
        "Restart=always",
        "RestartSec=30",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])
