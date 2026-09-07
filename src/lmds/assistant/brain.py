"""สมองของผู้ช่วยจากโมเดลในฟลีต — ชี้ provider ไปที่ endpoint ของ bundle ที่รันอยู่

ลูกค้าจำนวนมากไม่มี API key ของ LLM ข้างนอก แต่มีโมเดลรันอยู่บนเครื่องตัวเองแล้ว (นั่นคือ
เหตุผลที่ติดตั้ง LMDS) · ปุ่ม "ใช้โมเดลนี้เป็นสมองของผู้ช่วย" บนการ์ดจึงเป็นทางที่สั้นที่สุด
จาก "ยังไม่มีสมอง" ไป "ผู้ช่วยทำงานได้" — และไม่มีอะไรออกนอกเครื่อง

ไม่มีทางลัดใหม่: ตั้งผ่าน `Settings.set_provider` ตัวเดียวกับหน้า Provider และ
`lmds config set-provider` · ที่นี่แค่ *หา* base URL กับชื่อโมเดลให้ถูกจากสิ่งที่ inventory รู้อยู่แล้ว
"""

from __future__ import annotations

from dataclasses import dataclass

# engine ที่เสิร์ฟ /v1/chat/completions แบบ OpenAI-compatible — embedding เสิร์ฟแต่ /v1/embeddings คุยไม่ได้
_CHAT_ENGINES = ("vllm", "llamacpp", "sglang")


class BrainError(ValueError):
    pass


@dataclass(frozen=True)
class Brain:
    node: str          # "this" = เครื่องที่รัน hub
    slug: str
    model: str         # served name ที่ /v1/models ประกาศ
    base_url: str

    def payload(self) -> dict:
        return {"node": self.node, "slug": self.slug, "model": self.model, "base_url": self.base_url}


def _pick(model: dict, node: str, host: str) -> Brain:
    if not model.get("running"):
        raise BrainError(f"{model.get('slug')} ยังไม่ได้รัน — start ก่อนถึงจะใช้เป็นสมองได้")
    engine = str(model.get("engine") or "")
    if engine not in _CHAT_ENGINES or "embedding" in str(model.get("features") or ""):
        raise BrainError(f"{model.get('slug')} ({engine or '?'}) ไม่ได้เสิร์ฟ chat completions — ใช้ vLLM/llama.cpp/SGLang ที่เป็นโมเดล chat")
    port = int(model.get("port") or 0)
    if not port:
        raise BrainError(f"ไม่รู้พอร์ตของ {model.get('slug')}")
    served = str(model.get("served_name") or model.get("default_served_name") or model.get("model_id") or model.get("slug"))
    return Brain(node=node, slug=str(model.get("slug")), model=served, base_url=f"http://{host}:{port}/v1")


def from_cache(node: str, slug: str) -> Brain:
    """หา endpoint จากแคชของ hub (เร็ว ไม่ SSH) — ใช้โดยปุ่มบนหน้าเว็บ"""
    from lmds.web import state

    snap = state.STORE.snapshot()
    node = (node or "").strip() or "this"
    if node in ("this", "local", "hub"):
        data = (snap.get("host") or {}).get("data") or {}
        host = "127.0.0.1"
        node = "this"
    else:
        from lmds.nodes import find

        entry = (snap.get("nodes") or {}).get(node)
        registered = find(node)
        if registered is None:
            raise BrainError(f"ไม่รู้จักเครื่อง {node}")
        data = (entry or {}).get("data") or {}
        if not data:
            raise BrainError(f"ยังไม่มีข้อมูลของ {node} ในแคช — กด Refresh ที่การ์ดก่อน")
        # ที่อยู่ที่เครื่องนั้นรายงานเอง (IP จริง) มาก่อนชื่อที่ hub ใช้ ssh (อาจเป็น alias ที่ python resolve ไม่ได้)
        host = registered.local_ip or registered.host
    for m in data.get("models") or []:
        if m.get("slug") == slug:
            return _pick(m, node, host)
    raise BrainError(f"ไม่พบ {slug} บน {node}")


def from_live(node: str, slug: str) -> Brain:
    """หา endpoint โดยถามเครื่องจริง — ใช้โดย CLI ที่ไม่มีแคชของ hub"""
    node = (node or "").strip() or "this"
    if node in ("this", "local", "hub"):
        from lmds.fleet import discover
        from lmds.inventory import model_payload

        for server in discover():
            if server.slug == slug:
                return _pick(model_payload(server), "this", "127.0.0.1")
        raise BrainError(f"ไม่พบ {slug} บนเครื่องนี้ — ดู: lmds ps")
    from lmds.nodes import NodeError, find, run
    from lmds.nodes.ssh import _json_object

    registered = find(node)
    if registered is None:
        raise BrainError(f"ไม่รู้จักเครื่อง {node} — ดู: lmds node list")
    try:
        result = run(registered, "lmds agent info", timeout=60)
    except NodeError as exc:
        raise BrainError(str(exc)) from exc
    data = _json_object(result.stdout) if result.ok else None
    if not data:
        raise BrainError(f"{node} ตอบ agent info ไม่ได้: {(result.stderr or result.stdout).strip()[:200]}")
    for m in data.get("models") or []:
        if m.get("slug") == slug:
            return _pick(m, node, registered.local_ip or registered.host)
    raise BrainError(f"ไม่พบ {slug} บน {node}")


def apply(brain: Brain):
    """ตั้งเป็น provider ของระบบ — ทางเดียวกับหน้า Provider (openai-compat ไม่ต้องใช้ key)"""
    from lmds.config import ProviderName, Settings

    settings = Settings.load()
    provider = settings.set_provider(ProviderName.OPENAI_COMPAT, model=brain.model, base_url=brain.base_url)
    settings.save()
    return provider


def owner(provider, snapshot: dict | None = None) -> dict | None:
    """provider ที่ตั้งอยู่ชี้ไปโมเดลตัวไหนในฟลีต — ให้หัวกล่องแชทบอกว่าสมองคือใคร

    เทียบจาก host:port ของ base_url กับโมเดลในแคช · ไม่ตรงกับตัวไหน = สมองข้างนอก (คืน None)
    """
    if provider is None or getattr(provider, "name", None) is None:
        return None
    if provider.name.value != "openai-compat" or not provider.base_url:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(provider.base_url)
    host, port = parsed.hostname or "", parsed.port
    if not port:
        return None
    if snapshot is None:
        from lmds.web import state

        snapshot = state.STORE.snapshot()
    local = (snapshot.get("host") or {}).get("data") or {}
    if host in ("127.0.0.1", "localhost", "::1", (local.get("host") or {}).get("ip")):
        for m in local.get("models") or []:
            if int(m.get("port") or 0) == port:
                return {"node": "this", "slug": m.get("slug")}
    from lmds.nodes import load

    for registered in load():
        if host not in (registered.local_ip, registered.host, *registered.alt_hosts):
            continue
        entry = (snapshot.get("nodes") or {}).get(registered.name) or {}
        for m in (entry.get("data") or {}).get("models") or []:
            if int(m.get("port") or 0) == port:
                return {"node": registered.name, "slug": m.get("slug")}
    return None
