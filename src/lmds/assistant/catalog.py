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
# repo ของ Hugging Face: org/name — ตัวอักษรชุดเดียวกับ slug ทั้งสองฝั่ง ไม่รับ URL (ให้ LLM ตัด https://huggingface.co/ ออกเอง)
_REPO = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}/[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
# ชื่อที่ API เสิร์ฟ (served name) — client ตั้งเป็น org/name หรือ name:tag ได้ จึงกว้างกว่า slug แต่ยังไม่มีช่องว่าง/quote
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,119}$")
# ข้อความสั้น ๆ ของคน (โจทย์ที่อยากได้โมเดล) — ใช้เฉพาะ probe ที่คำนวณในโปรเซส ไม่เคยไปถึง shell
_TEXT = re.compile(r"^[\w\s.,;:()/+\-\u0E00-\u0E7F]{1,160}$")


class ParamError(ValueError):
    """พารามิเตอร์ไม่ผ่านการตรวจ — ปฏิเสธทั้งรายการ ไม่ใช่ตัดทิ้งเฉพาะตัวที่ผิด"""


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["slug", "word", "int", "lines", "bind", "ratio", "repo", "name", "text", "choice", "bool"]
    required: bool = True
    describe: str = ""
    # สำหรับ kind="choice" — ค่าที่รับได้ทั้งหมด (ตัวแรกคือค่าตั้งต้นเมื่อไม่ระบุและไม่บังคับ)
    choices: tuple[str, ...] = ()

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
        if self.kind == "repo":
            text = re.sub(r"^https?://huggingface\.co/", "", text).strip("/")
            if not _REPO.match(text):
                raise ParamError(f"repo ไม่ถูกรูปแบบ org/name: {text[:60]}")
            return text
        if self.kind == "name":
            if not _NAME.match(text):
                raise ParamError(f"ชื่อ '{self.name}' ไม่ถูกรูปแบบ: {text[:40]}")
            return text
        if self.kind == "text":
            if not _TEXT.match(text):
                raise ParamError(f"ข้อความ '{self.name}' มีอักขระที่ไม่รับ หรือยาวเกิน 160 ตัว")
            return " ".join(text.split())
        if self.kind == "choice":
            if text not in self.choices:
                raise ParamError(f"'{self.name}' ต้องเป็นหนึ่งใน {', '.join(self.choices)}")
            return text
        if self.kind == "bool":
            low = text.lower()
            if low in ("1", "true", "yes", "on"):
                return "1"
            if low in ("0", "false", "no", "off"):
                return "0"
            raise ParamError(f"'{self.name}' ต้องเป็น true/false")
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
    build: Callable[[dict[str, str]], str] | None = None
    params: tuple[Param, ...] = ()
    timeout: int = 60
    # probe ที่ *คำนวณบน hub* จากแคช/ทะเบียน แทนการรัน shell — รับ (params, target) คืนข้อความ
    # ใช้กับของที่ hub รู้ดีกว่าเครื่องปลายทาง (ตรง hub 3 มิติ, แนะนำโมเดลข้ามเครื่อง) · ไม่มี SSH
    compute: Callable[[dict[str, str], str], str] | None = None

    def command(self, given: dict) -> tuple[str, dict[str, str]]:
        clean = clean_params(self.params, given)
        if self.build is None:
            return "", clean
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


# ── Probe ชุดที่สอง: สิ่งที่ผู้ช่วยต้องรู้ถึงจะ *ทำงานแทน* ได้ ไม่ใช่แค่ตอบ ──────────────
# ทุกตัวอ้างโค้ด/คำสั่งที่มีอยู่แล้ว (fleet check, fit, bench, explain_crash ของ controller)
# — ผู้ช่วยไม่ได้สิทธิ์ใหม่ แค่เห็นสิ่งเดียวกับที่ปุ่มบนหน้าเว็บเห็น

def _insight(name: str):
    """ผูก compute กับฟังก์ชันใน insight.py แบบขี้เกียจ — import ตอนใช้ ไม่ใช่ตอนโหลดแคตตาล็อก"""
    def run(params: dict[str, str], target: str) -> str:
        from . import insight

        return getattr(insight, name)(params, target)
    return run


_probe(Probe(
    name="fleet_consistency",
    title="ตรง hub ครบ 3 มิติไหม",
    answers="เครื่องไหน 'ตรง hub' ครบสามมิติ (code · controllers · runtime) เครื่องไหนค้าง ค้างมิติไหน "
            "— ใช้เมื่อถามว่าต้องอัปเดตอะไร หรือก่อนเสนอ node_install/bundles_refresh/update_runtime · ไม่ต้องระบุเครื่อง",
    compute=_insight("fleet_consistency"),
    timeout=30,
))

_probe(Probe(
    name="fit_preview",
    title="ตาราง Fit (slots/context/KV)",
    answers="โมเดลนี้ตั้ง slots/context เท่านี้ต้องใช้ RAM เท่าไร ใส่ได้ไหม ควรปักหมุด KV เท่าไร (lmds fit — ไม่เขียนอะไร) "
            "— ใช้ก่อนเสนอ set_fit/set_context/set_slots เสมอ",
    params=(_SLUG_PARAM,
            Param("slots", "int", required=False, describe="จำนวน request พร้อมกันที่อยากได้"),
            Param("context", "int", required=False, describe="context ต่อคำขอที่อยากได้")),
    build=lambda p: "lmds fit " + shlex.quote(p["slug"])
                    + (f" --slots {shlex.quote(p['slots'])}" if p.get("slots") else "")
                    + (f" --context {shlex.quote(p['context'])}" if p.get("context") else ""),
    timeout=120,
))

# รูปแบบ error ชุดเดียวกับ explain_crash ของ controller (llama.cpp + vLLM) — สาเหตุจริงอยู่บรรทัดเหล่านี้
_CRASH_PATTERNS = (
    "unknown model architecture|error loading model|failed to load model|failed to allocate|cudaMalloc failed|"
    "out of memory|Cannot allocate memory|CUDA error|check_tensor_dims|wrong shape|wrong number of tensors|"
    "No available memory for the cache blocks|Not enough memory|OutOfMemoryError|ValueError: |RuntimeError: |"
    "AssertionError: |ImportError: |ModuleNotFoundError: |NotImplementedError: |address already in use|"
    "Address already in use|ptxas|e2m1|Killed|Traceback"
)


def _last_failure(p: dict[str, str]) -> str:
    slug = shlex.quote(p["slug"])
    pattern = shlex.quote(_CRASH_PATTERNS)
    # llama.cpp native เขียน ~/.lmds/run/<slug>/server.log · vLLM/SGLang/llama.cpp docker อยู่ใน docker logs ของ lmds-<slug>
    # อ่านทั้งสองแหล่งที่มี — บอกด้วยว่าอ่านจากไหน ผู้ช่วยจะได้อ้างถูก
    return _survey(
        f'log=~/.lmds/run/{slug}/server.log; '
        f'if [ -f "$log" ]; then echo "== server.log ($log) — บรรทัด error ล่าสุด"; '
        f'tail -n 600 "$log" | grep -aE {pattern} | tail -8; echo; echo "== ท้าย server.log"; tail -n 12 "$log"; fi; '
        f'if docker inspect lmds-{slug} >/dev/null 2>&1; then echo; echo "== docker logs lmds-{slug} — บรรทัด error ล่าสุด"; '
        f'docker logs --tail 600 lmds-{slug} 2>&1 | grep -aE {pattern} | grep -vE "See root cause above" | tail -8; '
        f'echo; echo "== สถานะ container"; docker ps -a --filter name=^lmds-{slug}$ --format "{{{{.Status}}}}"; fi; '
        f'[ -f "$log" ] || docker inspect lmds-{slug} >/dev/null 2>&1 || echo "ไม่พบ log ของ {p["slug"]} บนเครื่องนี้ (ยังไม่เคย start หรือ bundle อยู่เครื่องอื่น)"'
    )


_probe(Probe(
    name="last_failure",
    title="สาเหตุที่ start ล้มครั้งล่าสุด",
    answers="ทำไม start/restart ครั้งล่าสุดถึงล้ม — ดึงบรรทัด error จริงจาก server.log/docker logs (แบบเดียวกับ explain_crash) "
            "ใช้แทน model_logs เมื่อคำถามคือ 'ทำไมไม่ขึ้น' จะได้ไม่ต้องอ่าน log ทั้งก้อน",
    params=(_SLUG_PARAM,),
    build=_last_failure,
    timeout=60,
))

_RUNTIME_PY = (
    # พิมพ์เป็นบรรทัดสั้น ๆ ไม่ใช่ JSON ทั้งก้อน — ผลถูกตัดที่ 4,000 ตัวโดยเก็บท้ายไว้ ถ้าพิมพ์ JSON สรุปรันไทม์ (อยู่ต้น) จะหาย
    "import json,sys; d=json.load(sys.stdin); h=d.get('host') or {}; r=h.get('runtimes') or {}; "
    "print('lmds', h.get('lmds_version'), h.get('lmds_commit'), '· template', h.get('template_hash')); "
    "[print('llama.cpp', x.get('dir'), '· build', x.get('build') or '?', x.get('date') or '', '· lock', x.get('lock_state'), "
    "'· ใช้โดย', ', '.join(x.get('used_by') or [])) for x in r.get('llamacpp') or []]; "
    "[print('image', x.get('ref') or x.get('image'), '·', (x.get('digest') or '')[:19], x.get('created') or '', "
    "'· ใช้โดย', ', '.join(x.get('used_by') or [])) for x in r.get('images') or []]; print(); "
    "[print(m.get('slug'), '·', m.get('engine'), (m.get('runtime') or {}).get('mode') or '', "
    "'· arch', (m.get('runtime_arch') or {}).get('arch') or '-', "
    "{True: 'รู้จัก', False: 'ไม่รู้จัก (รันไทม์เก่ากว่าโมเดล)', None: 'ตรวจไม่ได้'}[(m.get('runtime_arch') or {}).get('supported')], "
    "'· controller', (m.get('controller') or {}).get('state') or '?') for m in d.get('models') or []]"
)

_probe(Probe(
    name="runtime_info",
    title="รันไทม์บนเครื่อง (build/image/arch)",
    answers="llama.cpp build ไหน วันที่เท่าไร ล็อกไว้ไหม · vLLM ใช้ image/digest อะไร · รันไทม์รู้จัก arch ของแต่ละโมเดลไหม "
            "และ controller ของแต่ละ bundle ตรง template ไหม — ใช้เมื่อสงสัยว่า 'รันไทม์เก่ากว่าโมเดล'",
    build=lambda _: _survey(f"lmds agent info | python3 -c {shlex.quote(_RUNTIME_PY)}"),
    timeout=90,
))

_probe(Probe(
    name="bench_results",
    title="ผล Score/bench ล่าสุด",
    answers="โมเดลนี้เคยวัดคะแนน (lmds bench) ไหม ได้ tok/s เท่าไร ผ่าน capability อะไรบ้าง (tools/vision/thai/json/reasoning) "
            "— ใช้เมื่อเทียบโมเดลหรือถามว่าตัวไหนเร็ว/เก่งกว่า",
    params=(_SLUG_PARAM,),
    build=lambda p: _survey(f"lmds bench show {shlex.quote(p['slug'])}"),
    timeout=60,
))

_probe(Probe(
    name="model_recommend",
    title="แนะนำโมเดลจากที่มีในฟลีต",
    answers="งานแบบนี้ (coding / ภาษาไทย / vision / tool calling / uncensored / long-context / reasoning) ควรใช้โมเดลไหนที่ฟลีต "
            "*มี weight อยู่แล้ว* รันอยู่ที่ไหน หรือเครื่องไหนถือได้ — คำนวณจากแคช inventory + สูตร recipes บน hub ไม่ต้องระบุเครื่อง",
    params=(Param("task", "text", describe="โจทย์เป็นคำสั้น ๆ เช่น 'coding', 'ภาษาไทย vision', 'tools long-context'"),),
    compute=_insight("model_recommend"),
    timeout=30,
))

_probe(Probe(
    name="usage",
    title="จำนวนคำขอต่อโมเดล",
    answers="โมเดลไหนมีคนเรียกใช้จริงบ้างใน 24 ชั่วโมงที่ผ่านมา (นับ POST /v1/* จาก log) — ใช้ก่อนเสนอหยุด/ลบโมเดล หรือถามว่าตัวไหนไม่มีใครใช้",
    build=lambda _: _survey(
        'for c in $(docker ps --filter name=^lmds- --format "{{.Names}}" 2>/dev/null); do '
        'n=$(docker logs --since 24h "$c" 2>&1 | grep -acE "POST /v1/(chat/)?(completions|embeddings|responses)"); '
        'echo "$c (docker, 24h): $n คำขอ"; done; '
        'for f in ~/.lmds/run/*/server.log; do [ -f "$f" ] || continue; s=$(basename "$(dirname "$f")"); '
        'n=$(grep -acE "POST /v1/(chat/)?(completions|embeddings)" "$f"); '
        'echo "$s (native, นับตั้งแต่ start รอบนี้ — log ไม่มี timestamp): $n คำขอ"; done; '
        'echo; echo "(0 = ไม่มีใครเรียกในช่วงนั้น · โมเดลที่ไม่ได้รันไม่มี log ให้นับ)"'
    ),
    timeout=90,
))

_probe(Probe(
    name="weights_on_disk",
    title="weight ที่อยู่บนดิสก์",
    answers="เครื่องนี้มี weight ของโมเดลอะไรอยู่แล้ว ก้อนละกี่ GB (แคช Hugging Face + GGUF ใน bundle) และดิสก์เหลือเท่าไร "
            "— ใช้ก่อน deploy ซ้ำ, ก่อนลบ, หรือตอนถามว่า remove --keep-weights ทิ้งอะไรไว้",
    build=lambda _: _survey(
        "echo '== แคช Hugging Face'; du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | sort -h | tail -40; "
        "echo; echo '== ไฟล์ใน bundle'; du -sh ~/bundles/*/weights ~/bundles/*/*.gguf ~/*/bundles/*/weights 2>/dev/null | sort -h | tail -20; "
        "echo; echo '== ดิสก์'; df -h ~ | tail -1"
    ),
    timeout=180,
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
    # งานที่ต้องสั่งจาก hub เท่านั้น (เช่น `lmds node install`) — target ที่ LLM เลือกถูกบังคับเป็น this
    hub_only: bool = False
    # งานประกอบ: คืนรายการขั้นย่อย [{action, target, params}] ที่ policy จะขยายเป็นตั๋วขั้นต่อขั้น
    # (deploy_model = plan → push → download → start → test) · build ของตัวมันเองไม่ถูกใช้
    expand: Callable[[dict[str, str], str], list[dict]] | None = None

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


# ── Action ชุดที่สอง: งานประจำของฟลีตที่เดิมต้องไปกดเองทีละปุ่ม ─────────────────────
# กติกาเดิมทุกข้อ: คำสั่งประกอบที่นี่ · ผ่านตั๋วอนุมัติเสมอ · ใช้ CLI/controller ตัวเดียวกับปุ่มบนหน้าเว็บ

_action(Action(
    name="set_fit",
    title="ตั้ง slots/context/KV ให้พอดี (Fit) แล้วบันทึกลง bundle",
    answers="ให้ระบบคำนวณ slots/context/KV pin ที่พอดีกับเครื่องแล้วเขียนลง bundle (lmds set --fit) — ใช้เมื่อ OOM, "
            "อยากรับหลายคนพร้อมกัน, หรือรันสองโมเดลร่วมเครื่อง · ดู fit_preview ก่อนเสมอ",
    params=(_SLUG_PARAM,
            Param("slots", "int", required=False, describe="จำนวน request พร้อมกัน"),
            Param("context", "int", required=False, describe="context ต่อคำขอ")),
    build=lambda p: "lmds set " + shlex.quote(p["slug"]) + " --fit"
                    + (f" --slots {shlex.quote(p['slots'])}" if p.get("slots") else "")
                    + (f" --context {shlex.quote(p['context'])}" if p.get("context") else ""),
    risk="low",
    timeout=180,
    impact="เขียนค่าลง bundle เท่านั้น ยังไม่รีสตาร์ต · ไม่พอ = ไม่เขียนและบอกว่าต้องลด slots หรือหยุดตัวไหน · มีผลเมื่อ start/restart ครั้งถัดไป",
    steps=("คำนวณจากตัวเลขจริงของเครื่อง", "เขียน slots/context/pin ลง bundle"),
))

_action(Action(
    name="stop_to_fit",
    title="หยุดโมเดลที่ Fit บอกว่าต้องหยุด",
    answers="หยุดโมเดลอีกตัวเพื่อคืนหน่วยความจำให้ตัวที่จะรัน — ใช้เมื่อ fit_preview บอกว่า 'ไม่พอ ต้องหยุด <slug>'",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "stop"),
    risk="medium",
    impact="โมเดลที่หยุดจะไม่ให้บริการจนกว่าจะสั่งเปิดใหม่ · คำขอที่ค้างอยู่ขาดทันที",
))

_action(Action(
    name="set_slots",
    title="ตั้งจำนวน slots แล้วบันทึกลง bundle",
    answers="ตั้งจำนวน request พร้อมกัน (llama.cpp --parallel / vLLM max-num-seqs) — llama.cpp แบ่ง context ให้ทุก slot เท่ากัน",
    params=(_SLUG_PARAM, Param("slots", "int", describe="จำนวน request พร้อมกัน")),
    build=lambda p: f"lmds set {shlex.quote(p['slug'])} --slots {shlex.quote(p['slots'])}",
    risk="low",
    timeout=60,
    impact="มีผลเมื่อ start/restart ครั้งถัดไป · llama.cpp: context ต่อคำขอ = context ÷ slots",
))

_action(Action(
    name="set_served_name",
    title="ตั้งชื่อที่ API เสิร์ฟ (served name)",
    answers="เปลี่ยนชื่อโมเดลที่ /v1/models ประกาศ — ใช้เมื่อ client ตั้งชื่อเดิมไว้แล้ว",
    params=(_SLUG_PARAM, Param("name", "name", describe="ชื่อที่ client จะเรียก เช่น gpt-4o หรือ org/model")),
    build=lambda p: f"lmds set {shlex.quote(p['slug'])} --model-id {shlex.quote(p['name'])}",
    risk="low",
    timeout=60,
    impact="มีผลเมื่อ start/restart ครั้งถัดไป · client ที่เรียกชื่อเดิมอยู่จะได้ 404 หลังรีสตาร์ต",
))

_action(Action(
    name="update_runtime",
    title="build llama.cpp ใหม่ให้ bundle นี้ (update runtime)",
    answers="รันไทม์เก่ากว่าโมเดล (unknown model architecture) — build llama.cpp จาก master ใหม่ให้ bundle นี้ "
            "(LLAMA_CPP_UPDATE=1 prepare-runtime) · เฉพาะ llama.cpp native · docker ให้ set image ใหม่แทน",
    params=(_SLUG_PARAM,),
    build=lambda p: controller_prefix(p["slug"]) + 'LLAMA_CPP_UPDATE=1 "$ctl" prepare-runtime',
    risk="medium",
    timeout=5400,
    impact="ใช้เวลา 10–15 นาที กิน CPU ระหว่าง build · bundle อื่นบนเครื่องไม่ถูกแตะ · โมเดลนี้ต้อง restart หลัง build เสร็จ",
    steps=("ดึง source llama.cpp ล่าสุด", "build", "ตรวจว่า build ใหม่รู้จัก arch ของโมเดล"),
))

_action(Action(
    name="regenerate_controller",
    title="regenerate controller ของ bundle นี้",
    answers="controller ค้าง template เก่า (มิติ controllers ไม่ตรง hub) — render ใหม่จาก MODEL_PROFILE.yaml ออฟไลน์ (lmds bundles refresh)",
    params=(_SLUG_PARAM,),
    build=lambda p: f"lmds bundles refresh {shlex.quote(p['slug'])}",
    risk="low",
    timeout=300,
    impact="ไม่กี่วินาที · เก็บสคริปต์เดิมเป็น .replaced-<เวลา> · bundle.env/bundle.args ไม่แตะ · ตัวที่รันอยู่ใช้สคริปต์เก่าจนกว่าจะ restart",
))

_action(Action(
    name="bundles_refresh",
    title="regenerate controller ทุกใบที่ค้างบนเครื่องนี้",
    answers="ทำมิติ controllers ให้ตรง hub ทั้งเครื่องในครั้งเดียว (lmds bundles refresh --all --if-older)",
    build=lambda _: "lmds bundles refresh --all --if-older",
    risk="low",
    timeout=600,
    impact="regenerate เฉพาะใบที่เก่ากว่า template · bundle จาก adopt ข้าม · ตัวที่รันอยู่ใช้สคริปต์เก่าจนกว่าจะ restart",
))

_action(Action(
    name="node_install",
    title="อัปเดต LMDS บนเครื่อง (Update)",
    answers="ทำให้เครื่อง 'ตรง hub' ครบ 3 มิติ: ส่งโค้ดจาก hub → install.sh → regenerate controller → build llama.cpp ที่ค้าง "
            "(lmds node install) · ไม่ระบุเครื่อง = ทุกเครื่องในทะเบียน · สั่งจาก hub เสมอ",
    params=(Param("node", "word", required=False, describe="ชื่อเครื่องในทะเบียน (ว่าง = ทุกเครื่อง)"),),
    build=lambda p: "lmds node install " + (shlex.quote(p["node"]) if p.get("node") else "--all"),
    risk="medium",
    timeout=7200,
    hub_only=True,
    impact="โค้ดบนเครื่องนั้นเปลี่ยน · build llama.cpp ที่ค้างใช้ 10–15 นาที/เครื่อง · โมเดลที่รันอยู่ไม่ถูกหยุด แต่ต้อง restart ถึงจะใช้ controller ใหม่ "
           "· hub ที่มีไฟล์แก้ค้างจะถูกปฏิเสธ (commit ก่อน)",
    steps=("ส่งโค้ดจาก hub", "install.sh บนเครื่องนั้น", "regenerate controller ที่ค้าง", "build llama.cpp ที่ค้าง", "ตรวจ 3 มิติซ้ำ"),
))

_action(Action(
    name="enable_autostart",
    title="เปิด autostart (ขึ้นเองหลัง reboot)",
    answers="ให้โมเดลกลับมาเองหลังเปิด-ปิดเครื่อง (systemd user service — ไม่ต้อง sudo)",
    params=(_SLUG_PARAM,),
    build=lambda p: f"lmds enable {shlex.quote(p['slug'])}",
    risk="low",
    timeout=120,
    impact="สร้าง unit ของ user · เครื่องต้องเปิด linger ไม่งั้น service ตายตอน logout (คำสั่งจะบอกถ้าต้องตั้ง)",
))

_action(Action(
    name="disable_autostart",
    title="ปิด autostart",
    answers="เลิกให้โมเดลขึ้นเองหลัง reboot — ไม่ได้หยุดตัวที่รันอยู่",
    params=(_SLUG_PARAM,),
    build=lambda p: f"lmds disable {shlex.quote(p['slug'])}",
    risk="low",
    timeout=120,
    impact="ตัวที่รันอยู่ไม่ถูกหยุด · unit แบบ system ต้อง sudo ซึ่ง SSH ไม่มี tty — ถ้าล้มให้รันเองบนเครื่อง",
))

_action(Action(
    name="remove_model",
    title="ลบโมเดลออกจากเครื่อง",
    answers="ลบ bundle/controller/unit ของโมเดลนี้ (lmds remove) — ค่าตั้งต้นเก็บ weight ไว้ (keep_weights=1) "
            "· ลบ weight ด้วยต้องระบุ keep_weights=0 และผู้ใช้ต้องขอชัดเจน",
    params=(_SLUG_PARAM, Param("keep_weights", "bool", required=False, describe="1 = เก็บ weight ไว้ (ค่าตั้งต้น) · 0 = ลบ weight ด้วย")),
    build=lambda p: f"lmds remove {shlex.quote(p['slug'])} --yes"
                    + ("" if p.get("keep_weights") == "0" else " --keep-weights"),
    risk="high",
    timeout=600,
    impact="ถาวร — bundle และ controller หายไป · ถ้ารันอยู่จะถูกหยุด · keep_weights=0 = ลบ weight หลาย GB ที่ต้องโหลดใหม่ถ้าเปลี่ยนใจ",
))

_TESTS = ("test-text", "test-tools", "test-vision", "test-reasoning", "test-embed", "test-rerank", "score")

_action(Action(
    name="run_test",
    title="ทดสอบโมเดล",
    answers="พิสูจน์ว่าโมเดลใช้ได้จริง: test-text ตอบไหม · test-tools คืน tool_calls ไหม · test-vision เห็นภาพไหม · "
            "test-reasoning แยกความคิดออกไหม · test-embed (โมเดล embedding) vector ข้ามภาษาถูกไหม · "
            "test-rerank (โมเดล reranker) เอกสารที่เกี่ยวข้องได้อันดับหนึ่งไหม · score = lmds bench --quick (วัดความเร็ว+ความสามารถ · เฉพาะ chat)",
    params=(_SLUG_PARAM, Param("test", "choice", choices=_TESTS, describe=" | ".join(_TESTS))),
    build=lambda p: (f"lmds bench run {shlex.quote(p['slug'])} --quick" if p["test"] == "score"
                     else _ctl(p["slug"], p["test"])),
    risk="low",
    timeout=900,
    impact="ยิงคำขอจริงเข้าโมเดล (ใช้ GPU ไม่กี่วินาที ถึงหลายนาทีสำหรับ score) · โมเดลต้องรันอยู่",
))

_action(Action(
    name="model_download",
    title="ดาวน์โหลด weight",
    answers="โหลด weight ของโมเดลลงเครื่อง (controller download — resume ได้)",
    params=(_SLUG_PARAM,),
    build=lambda p: _ctl(p["slug"], "download"),
    risk="low",
    timeout=14400,
    impact="หลาย GB ถึงหลายสิบ GB ใช้เวลาและดิสก์ · ทำซ้ำได้ (ข้ามไฟล์ที่ครบแล้ว)",
))

_action(Action(
    name="push_bundle",
    title="ส่ง bundle จาก hub ไปเครื่องปลายทาง",
    answers="ส่ง bundle ที่สร้างบน hub ไปติดตั้งบนเครื่องอื่น (lmds node push) — ตัวเดียวกับที่อนุมัติแผน ไม่วางแผนใหม่ที่ปลายทาง",
    params=(Param("node", "word", describe="เครื่องปลายทาง"), _SLUG_PARAM),
    build=lambda p: f"lmds node push {shlex.quote(p['node'])} {shlex.quote(p['slug'])}",
    risk="low",
    timeout=600,
    hub_only=True,
    impact="แพ็ก zip ใหม่จากโฟลเดอร์ (ค่าจาก lmds set ติดไปด้วย) แล้ว scp · bundle เดิมชื่อเดียวกันบนปลายทางถูกทับ",
))

_action(Action(
    name="deploy_plan",
    title="วางแผน + สร้าง bundle บน hub",
    answers="สร้าง bundle จาก repo Hugging Face บน hub (lmds deploy --yes) — ขั้นแรกของ deploy_model · ปกติไม่ต้องเลือกเอง",
    params=(Param("repo", "repo", describe="org/name บน Hugging Face"),
            Param("target", "word", required=False, describe="target preset เช่น dgx-spark-single (ว่าง = auto)"),
            Param("gguf", "word", required=False, describe="quant ของ GGUF เช่น Q4_K_M เมื่อ repo มีหลายไฟล์")),
    build=lambda p: "lmds deploy " + shlex.quote(p["repo"]) + " --yes"
                    + (f" --target {shlex.quote(p['target'])}" if p.get("target") else "")
                    + (f" --gguf {shlex.quote(p['gguf'])}" if p.get("gguf") else ""),
    risk="low",
    timeout=1800,
    hub_only=True,
    impact="ถาม Hugging Face + ใช้สมอง (LLM) วางแผน · สร้างโฟลเดอร์ bundle บน hub ยังไม่โหลด weight · gated repo ต้องมี HF token ตั้งไว้",
))


def _deploy_steps(p: dict[str, str], target: str) -> list[dict]:
    """deploy_model = แผน → ส่ง → โหลด → start → เทส · แต่ละขั้นคือ action จริงในแคตตาล็อกนี้ ผู้ใช้เห็นทุกคำสั่ง

    slug ตั้งตามกติกาเดียวกับ `lmds deploy` (rulebased.slugify) — ผู้ใช้จึงเห็นชื่อที่จะได้ตั้งแต่ตอนอนุมัติ
    """
    from lmds.brain.rulebased import slugify

    slug = p.get("slug") or slugify(p["repo"])
    plan_params = {k: p[k] for k in ("repo", "target", "gguf") if p.get(k)}
    steps: list[dict] = [{"action": "deploy_plan", "target": "this", "params": plan_params}]
    if target and target != "this":
        steps.append({"action": "push_bundle", "target": "this", "params": {"node": target, "slug": slug}})
    steps += [
        {"action": "model_download", "target": target, "params": {"slug": slug}},
        {"action": "model_start", "target": target, "params": {"slug": slug}},
        {"action": "run_test", "target": target, "params": {"slug": slug, "test": "test-text"}},
        {"action": "run_test", "target": target, "params": {"slug": slug, "test": "test-tools"}},
    ]
    return steps


_action(Action(
    name="deploy_model",
    title="deploy โมเดลใหม่จนใช้ได้ (แผน → ส่ง → โหลด → start → เทส)",
    answers="เอาโมเดลจาก Hugging Face มารันบนเครื่องที่ระบุ ครบทุกขั้นในตั๋วเดียว — เหมาะกับโหมด 'ทีละขั้น' "
            "· ถ้ามี slug อยู่แล้วให้ใช้ model_start/model_download แทน",
    params=(Param("repo", "repo", describe="org/name บน Hugging Face"),
            Param("slug", "slug", required=False, describe="ชื่อ bundle ที่จะได้ (ว่าง = ตั้งจากชื่อ repo)"),
            Param("target", "word", required=False, describe="target preset (ว่าง = auto)"),
            Param("gguf", "word", required=False, describe="quant ของ GGUF เมื่อ repo มีหลายไฟล์")),
    build=lambda p: "lmds deploy " + shlex.quote(p["repo"]) + " --yes",
    expand=_deploy_steps,
    risk="medium",
    timeout=1800,
    impact="สร้าง bundle บน hub, ส่งไปเครื่องปลายทาง, โหลด weight หลาย GB, เปิดโมเดล (ใช้ VRAM) แล้วยิงเทส · ทุกขั้นเห็นคำสั่งก่อนกด",
    steps=("วางแผน + สร้าง bundle", "ส่งไปเครื่องปลายทาง", "โหลด weight", "start", "test-text", "test-tools"),
))


def probe_menu() -> list[dict]:
    """รายการ probe แบบย่อ — ใช้ทั้งใน prompt ของ router และหน้าเว็บ"""
    return [
        {"name": p.name, "title": p.title, "answers": p.answers,
         "params": [{"name": q.name, "required": q.required, "describe": q.describe,
                     "choices": list(q.choices)} for q in p.params]}
        for p in PROBES.values()
    ]


def action_menu() -> list[dict]:
    return [
        {"name": a.name, "title": a.title, "answers": a.answers, "risk": a.risk,
         "impact": a.impact, "hub_only": a.hub_only, "composite": a.expand is not None,
         "params": [{"name": q.name, "required": q.required, "describe": q.describe,
                     "choices": list(q.choices)} for q in a.params]}
        for a in ACTIONS.values()
    ]
