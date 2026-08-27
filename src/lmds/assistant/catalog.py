"""แคตตาล็อกสิ่งที่ผู้ช่วยทำกับเครื่องได้ — ตรวจ (probe) และแก้ (action)

ทำไมต้องเป็นแคตตาล็อก ไม่ใช่ปล่อยให้ LLM เขียนคำสั่งเอง:

กติกาเดียวกับขั้นวางแผน deploy (PRD §8.2) — **LLM ไม่เขียน Bash** มันได้แค่เลือกชื่อ
รายการในแคตตาล็อกนี้ แล้วเติมพารามิเตอร์ที่ผ่านการตรวจด้วยโค้ด · คำสั่งจริงประกอบ
ที่นี่ทั้งหมด ค่าที่มาจากภายนอกผ่าน shlex.quote ทุกตัว

ผลที่ได้คือขอบเขตที่ตรวจสอบได้: อ่านไฟล์นี้จบก็รู้ครบว่าผู้ช่วยแตะอะไรได้บ้าง ไม่ต้อง
ไปไล่อ่าน prompt แล้วเดาว่าโมเดลจะคิดอะไรออก · และเวลาที่มันหลง (หรือโดน prompt
injection จากข้อความ error ของเครื่องปลายทาง) สิ่งที่แย่ที่สุดที่เกิดได้คือ "เลือก
รายการที่ไม่เกี่ยว" ไม่ใช่ "รันคำสั่งที่เราไม่เคยอนุญาต"

การแบ่ง probe/action ไม่ใช่แค่การจัดหมวด — probe รันได้เลยเพราะอ่านอย่างเดียว
ส่วน action ต้องผ่านการอนุมัติของคนเสมอ (policy.py)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Callable, Literal

# ── การตรวจพารามิเตอร์ ────────────────────────────────────────────────────────
# ค่าที่ LLM ส่งมาถือว่าไม่น่าเชื่อถือเท่ากับค่าที่ผู้ใช้พิมพ์ — ตรวจด้วยรูปแบบที่แคบ
# ที่สุดที่ยังใช้งานได้จริง แล้วค่อย quote อีกชั้นตอนประกอบคำสั่ง
_SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_WORD = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


class ParamError(ValueError):
    """พารามิเตอร์ไม่ผ่านการตรวจ — ปฏิเสธทั้งรายการ ไม่ใช่ตัดทิ้งเฉพาะตัวที่ผิด"""


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["slug", "word", "int", "lines", "bind", "ratio"]
    required: bool = True
    describe: str = ""

    def clean(self, raw) -> str:
        if raw is None or raw == "":
            raise ParamError(f"ขาดค่า '{self.name}'")
        text = str(raw).strip()
        if self.kind == "slug":
            if not _SLUG.match(text):
                raise ParamError(f"slug ไม่ถูกรูปแบบ: {text[:40]}")
            return text
        if self.kind == "word":
            if not _WORD.match(text):
                raise ParamError(f"ค่า '{self.name}' ไม่ถูกรูปแบบ: {text[:40]}")
            return text
        if self.kind in ("int", "lines"):
            if not text.isdigit():
                raise ParamError(f"'{self.name}' ต้องเป็นตัวเลข: {text[:40]}")
            value = int(text)
            if self.kind == "lines":
                # log ทั้งไฟล์ไม่ได้ — ทั้งช้าและกิน context ของคำตอบจนหมด
                value = max(20, min(value, 400))
            elif value <= 0:
                raise ParamError(f"'{self.name}' ต้องมากกว่า 0")
            return str(value)
        if self.kind == "bind":
            if text not in ("0.0.0.0", "127.0.0.1", "::") and not _IPV4.match(text):
                raise ParamError(f"bind address ไม่ถูกรูปแบบ: {text[:40]}")
            return text
        if self.kind == "ratio":
            try:
                value = float(text)
            except ValueError as exc:
                raise ParamError(f"'{self.name}' ต้องเป็นตัวเลขทศนิยม") from exc
            if not 0.0 < value <= 0.98:
                raise ParamError(f"'{self.name}' ต้องอยู่ระหว่าง 0 ถึง 0.98")
            return f"{value:.2f}"
        raise ParamError(f"ชนิดพารามิเตอร์ที่ไม่รู้จัก: {self.kind}")


def clean_params(params: tuple[Param, ...], given: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in params:
        raw = given.get(spec.name)
        if raw in (None, "") and not spec.required:
            continue
        out[spec.name] = spec.clean(raw)
    return out


# ── ตัวช่วยประกอบคำสั่ง ───────────────────────────────────────────────────────
def controller_prefix(slug: str) -> str:
    """shell snippet ที่ตั้ง $ctl ให้ชี้ controller ของ slug นี้

    ก็อปแนวเดียวกับ endpoint อื่นที่สั่งงานข้ามเครื่อง — "bundle อยู่ที่ไหน" ต้องมี
    คำตอบเดียวทั้งระบบ ไม่ใช่คนละแบบในแต่ละที่
    """
    quoted = shlex.quote(slug)
    return (
        f'dir="$(ls -d ~/bundles/{quoted} ~/*/bundles/{quoted} 2>/dev/null | head -1)"; '
        f'[ -n "$dir" ] || {{ echo "ไม่พบ bundle {slug}" >&2; exit 1; }}; '
        f'cd "$dir" || exit 1; '
        f'ctl="$(ls ./*-single.sh ./*-stacked.sh 2>/dev/null | head -1)"; '
        f'[ -n "$ctl" ] || {{ echo "ไม่พบ controller ของ {slug}" >&2; exit 1; }}; '
    )


def _ctl(slug: str, command: str) -> str:
    return controller_prefix(slug) + f'"$ctl" {command}'


# ── Probe: อ่านอย่างเดียว ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Probe:
    name: str
    title: str            # ชื่อที่ผู้ใช้เห็นในหน้าเว็บ
    answers: str          # คำถามแบบไหนที่ probe นี้ตอบได้ — router ใช้เลือก
    build: Callable[[dict[str, str]], str]
    params: tuple[Param, ...] = ()
    timeout: int = 60

    def command(self, given: dict) -> tuple[str, dict[str, str]]:
        clean = clean_params(self.params, given)
        return self.build(clean), clean


_SLUG_PARAM = Param("slug", "slug", describe="slug ของโมเดล")

PROBES: dict[str, Probe] = {}


def _probe(probe: Probe) -> Probe:
    PROBES[probe.name] = probe
    return probe


_probe(Probe(
    name="overview",
    title="ภาพรวมเครื่อง (สด)",
    answers="สถานะรวมของเครื่องนี้ ณ ตอนนี้: GPU, RAM, ดิสก์, docker, role, แคช, โมเดลที่มี "
            "— ใช้เมื่ออยากได้ข้อมูลสดแทนค่าที่แคชไว้ หรือเมื่อไม่แน่ใจว่าจะเริ่มดูตรงไหน",
    build=lambda _: "lmds agent info",
    timeout=45,
))

def _survey(command: str) -> str:
    """probe สำรวจ: เก็บทุกอย่างที่เก็บได้ แล้วจบด้วยสถานะสำเร็จเสมอ

    เครื่องมือย่อยตัวหนึ่งไม่มี (ไม่มี nvidia-smi, ไม่มี ~/.cache/huggingface, ไม่มี
    docker) ไม่ได้แปลว่าการสำรวจล้มเหลว — ข้อความที่ได้มาคือคำตอบอยู่แล้ว การรายงานว่า
    "คำสั่งนี้ล้ม" ทำให้ผู้ช่วยทิ้งข้อมูลที่ใช้ได้ แล้วไปบอกผู้ใช้ว่าตรวจไม่ได้
    """
    return f"{{ {command} ; }} 2>&1 || true"


_probe(Probe(
    name="gpu",
    title="GPU และงานที่ถือ VRAM อยู่",
    answers="การ์ดรุ่นอะไร ไดรเวอร์เวอร์ชันไหน ร้อนแค่ไหน ใช้ไฟเท่าไร และ process ไหนถือ VRAM อยู่",
    build=lambda _: _survey(
        "nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,"
        "utilization.gpu,temperature.gpu,power.draw --format=csv; echo; "
        "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv"
    ),
))

_probe(Probe(
    name="disk",
    title="พื้นที่ดิสก์",
    answers="ดิสก์เหลือเท่าไร และแคชโมเดลกินไปเท่าไร — ใช้เมื่อ download ล้มหรือก่อนโหลดโมเดลใหญ่",
    build=lambda _: _survey(
        "df -h -x tmpfs -x devtmpfs | head -20; echo; "
        "du -sh ~/.cache/huggingface ~/bundles 2>/dev/null"
    ),
    timeout=90,
))

_probe(Probe(
    name="memory",
    title="RAM และ swap",
    answers="RAM เหลือเท่าไร swap เปิดอยู่ไหม — บนเครื่อง unified memory (Spark) นี่คืองบเดียวกับ VRAM",
    build=lambda _: _survey("free -h; echo; swapon --show"),
))

_probe(Probe(
    name="system",
    title="ระบบปฏิบัติการและรันไทม์",
    answers="เคอร์เนล, distro, เวอร์ชัน docker/NVIDIA toolkit, uptime — ใช้เทียบว่าเครื่องนี้ต่างจากเครื่องอื่นตรงไหน",
    build=lambda _: _survey(
        "uname -a; echo; . /etc/os-release 2>/dev/null && echo \"$PRETTY_NAME\"; echo; "
        "uptime; echo; docker --version; nvidia-ctk --version | head -2"
    ),
))

_probe(Probe(
    name="ports",
    title="พอร์ตที่เปิดฟังอยู่",
    answers="มีอะไรฟังพอร์ตไหนอยู่ — ใช้เมื่อ start แล้วชนพอร์ต หรือหาว่าโมเดลเสิร์ฟที่พอร์ตอะไรจริง ๆ",
    build=lambda _: _survey("ss -tlnp 2>/dev/null || ss -tln"),
))

_probe(Probe(
    name="docker",
    title="คอนเทนเนอร์ทั้งหมด",
    answers="คอนเทนเนอร์ไหนรันอยู่/ตายไปแล้ว ใช้ image อะไร — ใช้เมื่อโมเดลไม่ขึ้นหรือสงสัยว่ามีของค้าง",
    build=lambda _: _survey(
        "docker ps -a --format '{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}' | head -30"
    ),
))

_probe(Probe(
    name="network",
    title="เน็ตเวิร์กและ fabric",
    answers="อินเทอร์เฟซและ IP ของเครื่องนี้ รวมถึงพอร์ต 200G/RoCE — จำเป็นตอนตั้ง stacked ข้ามเครื่อง",
    build=lambda _: _survey(
        "ip -br addr; echo; ip route | head -10; echo; "
        "ls /sys/class/infiniband 2>/dev/null || echo 'ไม่มี RoCE/InfiniBand'"
    ),
))

_probe(Probe(
    name="bundles",
    title="bundle ที่มีบนเครื่อง",
    answers="เครื่องนี้มี bundle ของโมเดลอะไรบ้าง — ใช้เมื่อไม่แน่ใจว่า slug ที่ผู้ใช้พูดถึงมีอยู่จริงไหม",
    build=lambda _: _survey("ls -1 ~/bundles 2>/dev/null || echo 'ยังไม่มี bundle'"),
))

_probe(Probe(
    name="model_status",
    title="สถานะโมเดล",
    answers="โมเดลตัวนี้รันอยู่ไหม เสิร์ฟที่พอร์ตอะไร API ตอบไหม — ถามถึงโมเดลตัวใดตัวหนึ่งให้ใช้อันนี้",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "status"),
    timeout=90,
))

_probe(Probe(
    name="model_logs",
    title="log ของโมเดล",
    answers="log ล่าสุดของโมเดล — ใช้เสมอเมื่อโมเดล start ไม่ขึ้น ตายกลางทาง หรือตอบช้าผิดปกติ",
    params=(_SLUG_PARAM, Param("lines", "lines", required=False, describe="จำนวนบรรทัด (20-400)")),
    build=lambda p: _ctl(p["slug"], f"logs {p.get('lines', '200')}"),
    timeout=90,
))

_probe(Probe(
    name="model_config",
    title="ค่าที่โมเดลตั้งไว้",
    answers="context, พอร์ต, bind address และ endpoint ที่ประกาศของโมเดลตัวนี้ — ใช้ก่อนเสนอให้เปลี่ยนค่าเสมอ",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "network-info"),
    timeout=60,
))

_probe(Probe(
    name="doctor",
    title="ตรวจสุขภาพโมเดล",
    answers="ผลตรวจอัตโนมัติของ LMDS ว่าโมเดลตัวนี้มีอะไรผิดปกติ — ใช้เป็นด่านแรกเมื่อผู้ใช้บอกว่า 'มันพัง'",
    params=(_SLUG_PARAM,),
    build=lambda p: f"lmds doctor {shlex.quote(p['slug'])}",
    timeout=120,
))


# ── Action: เปลี่ยนสภาพเครื่อง ────────────────────────────────────────────────
Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Action:
    name: str
    title: str
    answers: str
    build: Callable[[dict[str, str]], str]
    params: tuple[Param, ...] = ()
    risk: Risk = "medium"
    timeout: int = 600
    # ผลกระทบที่ผู้ใช้ต้องรู้ *ก่อน* กดอนุมัติ — เขียนเป็นภาษาคน ไม่ใช่ชื่อ flag
    impact: str = ""
    # ขั้นตอนย่อยสำหรับโหมด "ทีละขั้น" — ว่างไว้ = งานขั้นเดียวจบ
    steps: tuple[str, ...] = field(default_factory=tuple)

    def command(self, given: dict) -> tuple[str, dict[str, str]]:
        clean = clean_params(self.params, given)
        return self.build(clean), clean


ACTIONS: dict[str, Action] = {}


def _action(action: Action) -> Action:
    ACTIONS[action.name] = action
    return action


_action(Action(
    name="model_restart",
    title="รีสตาร์ตโมเดล",
    answers="ปิดแล้วเปิดโมเดลใหม่ด้วยค่าเดิม — ใช้เมื่อโมเดลค้างหรือหลังแก้ไฟล์",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "restart"),
    risk="medium",
    impact="โมเดลจะหยุดให้บริการระหว่างโหลดใหม่ (หลักนาที) คำขอที่ค้างอยู่จะขาด",
    steps=("หยุดโมเดล", "เปิดใหม่ด้วยค่าเดิม", "รอ /health ตอบ"),
))

_action(Action(
    name="model_stop",
    title="หยุดโมเดล",
    answers="หยุดโมเดล คืน VRAM ให้เครื่อง — ใช้เมื่อจะเปิดตัวอื่นแทนหรือเครื่องหน่วยความจำไม่พอ",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "stop"),
    risk="medium",
    impact="โมเดลจะไม่ให้บริการจนกว่าจะสั่งเปิดใหม่",
))

_action(Action(
    name="model_start",
    title="เปิดโมเดล",
    answers="เปิดโมเดลด้วยค่าที่ตั้งไว้",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "start"),
    risk="low",
    impact="เครื่องจะโหลด weight ขึ้น GPU ใช้ VRAM ตามขนาดโมเดล",
))

_action(Action(
    name="set_context",
    title="เปลี่ยน context แล้วรีสตาร์ต",
    answers="ตั้ง context ใหม่ให้โมเดล — ลด context = รับผู้ใช้พร้อมกันได้มากขึ้น",
    params=(_SLUG_PARAM, Param("context", "int", describe="จำนวน token")),
    build=lambda p: _ctl(p["slug"], f"restart --context {shlex.quote(p['context'])}"),
    risk="medium",
    impact="โมเดลรีสตาร์ต · ตั้งสูงเกินงบหน่วยความจำจะ start ไม่ขึ้น ให้ดูค่าที่ lmds inspect แนะนำก่อน",
    steps=("ตรวจค่าปัจจุบัน", "รีสตาร์ตด้วย context ใหม่", "ยืนยันว่า /health กลับมา"),
))

_action(Action(
    name="set_port",
    title="เปลี่ยนพอร์ตแล้วรีสตาร์ต",
    answers="ย้ายโมเดลไปฟังพอร์ตอื่น — ใช้เมื่อพอร์ตชนกับของเดิม",
    params=(_SLUG_PARAM, Param("port", "int", describe="พอร์ต 1-65535")),
    build=lambda p: _ctl(p["slug"], f"restart --port {shlex.quote(p['port'])}"),
    risk="medium",
    impact="client ที่ตั้ง endpoint เดิมไว้จะต่อไม่ติดจนกว่าจะแก้ค่าตาม",
))

_action(Action(
    name="set_bind",
    title="เปลี่ยน bind address แล้วรีสตาร์ต",
    answers="จำกัดให้เสิร์ฟเฉพาะในเครื่อง (127.0.0.1) หรือเปิดทั้งวง (0.0.0.0)",
    params=(_SLUG_PARAM, Param("bind", "bind", describe="127.0.0.1 หรือ 0.0.0.0")),
    build=lambda p: _ctl(p["slug"], f"restart --bind {shlex.quote(p['bind'])}"),
    risk="medium",
    impact="0.0.0.0 = ใครในเครือข่ายเดียวกันก็ยิงโมเดลได้ · 127.0.0.1 = เครื่องอื่นต่อไม่ได้อีก",
))

_action(Action(
    name="set_gpu_util",
    title="เปลี่ยนสัดส่วนหน่วยความจำ GPU แล้วรีสตาร์ต",
    answers="ปรับ gpu-memory-utilization — ใช้เมื่อจะรันหลายโมเดลร่วมเครื่อง หรือ OOM ตอนโหลด",
    params=(_SLUG_PARAM, Param("ratio", "ratio", describe="0.1-0.98")),
    build=lambda p: _ctl(p["slug"], f"restart --gpu-memory-utilization {shlex.quote(p['ratio'])}"),
    risk="medium",
    impact="ตั้งสูงเกินจะ OOM ตอนโหลด ตั้งต่ำเกิน KV cache จะเล็กจนรับคนได้น้อยลง",
))

_action(Action(
    name="clear_fi_cache",
    title="ล้างแคช FlashInfer",
    answers="ลบ kernel ที่ JIT ไว้ — ใช้เมื่อเจอ error เรื่อง signature ของ kernel ไม่ตรงหลังเปลี่ยน image",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "clear-fi-cache"),
    risk="low",
    impact="คำขอแรกหลังล้างจะช้าเพราะต้อง JIT ใหม่",
))

_action(Action(
    name="prepare_runtime",
    title="เตรียมรันไทม์ (ดึง image)",
    answers="ดึง/ล็อก container image ที่โมเดลตัวนี้ต้องใช้",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "prepare-runtime"),
    risk="low",
    timeout=3600,
    impact="ดาวน์โหลดหลาย GB ใช้เวลาและพื้นที่ดิสก์",
))


def probe_menu() -> list[dict]:
    """รายการ probe แบบย่อ — ใช้ทั้งใน prompt ของ router และหน้าเว็บ"""
    return [
        {"name": p.name, "title": p.title, "answers": p.answers,
         "params": [{"name": q.name, "required": q.required, "describe": q.describe}
                    for q in p.params]}
        for p in PROBES.values()
    ]


def action_menu() -> list[dict]:
    return [
        {"name": a.name, "title": a.title, "answers": a.answers, "risk": a.risk,
         "impact": a.impact,
         "params": [{"name": q.name, "required": q.required, "describe": q.describe}
                    for q in a.params]}
        for a in ACTIONS.values()
    ]
