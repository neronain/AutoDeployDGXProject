"""ค่าจากข้างนอก (Hugging Face Hub · แผนของ LLM · profile เก่า) ที่จะไปอยู่ในสคริปต์ bash

controller ถูก render จาก template แล้ว *รันบนเครื่องลูกค้า* — ค่าทุกตัวที่ template แทรกลงไปจึงเป็นโค้ด
ถ้าไม่มีใครดูแลเรื่อง quote · audit 2026-10-06: ไฟล์ใน repo ที่ชื่อ `Q4_K_M$(touch PWNED).gguf` กับแผนที่
`served_model_name` เป็น `qwen$(touch PWNED)` ผ่าน gate ครบทุกด่าน แล้ว `controller help` ก็รันคำสั่งนั้นทันที
เพราะค่าถูกวางใน `"…"` ของ bash ดิบ ๆ

ป้องกันสามชั้น ไม่มีชั้นไหนพึ่งชั้นอื่น:

1. **escape ตอน render** (`generator/renderer.py` ใส่ให้ทุก `{{ … }}` เองตามบริบท — ฟังก์ชันอยู่ที่นี่)
2. **ปฏิเสธตั้งแต่ต้นทาง** — ค่าที่ไม่มีเหตุผลจะมี metacharacter ของเชลล์ (ชื่อไฟล์ ชื่อโมเดล image) ถูกตรวจด้วย
   กติกาในไฟล์นี้ ทั้งที่ inspector, `harden_plan` และ renderer
3. **gate** (`validator/gates.py` `gate_value_expansion`) เทียบ controller กับตัวที่ render ด้วยค่า canary

ไฟล์นี้ไม่ import อะไรจากแพ็กเกจ — inspector, brain และ generator ใช้ร่วมกันได้โดยไม่วนกัน
"""

from __future__ import annotations

import re
import shlex
import unicodedata

# คำที่ renderer ใส่แทน *ทุกค่า* ตอน render แบบ canary — ไม่มีอักขระที่เชลล์ตีความ อยู่ได้ทุกบริบท
# แยกตามชนิดของ encoding ที่ renderer เลือกให้ตำแหน่งนั้น: gate ใช้รู้ว่าค่าจริงตรงนั้นต้องมีหน้าตาแบบไหน
# (ห้ามมีคำไหนเป็นส่วนต้นของอีกคำ — gate แยกด้วย regex เดียว)
CANARY = "LMDSCANARY"              # ใน "…" · คอมเมนต์ · heredoc           → dq()
CANARY_DEFAULT = "LMDSDFLTCNRY"    # ค่าตั้งต้นใน "${VAR:-…}"              → dq_default()
CANARY_EITHER = "LMDSEITHCNRY"     # บรรทัดมี ' เปิดค้าง ('…' หรือข้อความ)   → quoted_either()
CANARY_WORDS = "LMDSWORDSCNRY"     # หลายคำที่ quote มาแล้ว วางนอก quote    → words()
CANARY_NUMBER = "LMDSNUMBCNRY"     # ตัวเลข — ไม่ถูก encode แต่ต้องเป็นตัวเลขจริง


class UnsafeValueError(ValueError):
    """ค่าที่ใส่ลง controller ไม่ได้ — ข้อความบอกว่าค่าไหน อักขระอะไร"""


class ShellWords(str):
    """ข้อความที่ถูก quote เป็น "คำ" ของ bash มาเรียบร้อยแล้ว (`shlex.quote`) — วางในบริบทที่ไม่มี quote ได้ตรง ๆ

    renderer ใช้ชนิดนี้บอก template ว่า "ไม่ต้อง escape ซ้ำ" · ห้ามห่อสตริงดิบด้วยชนิดนี้เอง ใช้ `words()` เสมอ
    """


def words(*values: str) -> ShellWords:
    """`shlex.quote` ทีละคำ คั่นด้วยช่องว่าง — สำหรับ `for f in …` / `args+=(…)` ที่ template วางโดยไม่มี quote

    อักขระควบคุมปฏิเสธเหมือน encoder ตัวอื่น: ขึ้นบรรทัดใหม่ใน `'…'` ถูกต้องตาม bash ก็จริง แต่ทำให้ controller มีบรรทัดเกิน
    จากที่ template เขียน — gate เทียบโครงบรรทัดต่อบรรทัดไม่ได้อีก และไม่มี flag/env ไหนที่ควรมีขึ้นบรรทัดใหม่อยู่แล้ว
    """
    for value in values:
        _reject_control(str(value))
    return ShellWords(" ".join(shlex.quote(str(v)) for v in values))


def _control_chars(value: str) -> list[str]:
    return sorted({c for c in value if ord(c) < 0x20 or ord(c) == 0x7F})


def _reject_control(value: str) -> None:
    # ขึ้นบรรทัดใหม่ในบรรทัดคอมเมนต์ (`# Model: …`) = บรรทัดถัดไปกลายเป็นคำสั่ง · ใน heredoc = ปิด heredoc ได้
    # ไม่มีค่าไหนที่ template แทรกแล้วควรมีอักขระควบคุม จึงปฏิเสธทั้งหมดแทนที่จะเดาว่าบริบทไหนทนได้
    bad = _control_chars(value)
    if bad:
        raise UnsafeValueError(
            f"ค่า {_short(value)} มีอักขระควบคุม ({', '.join(repr(c) for c in bad)}) — ใส่ลง controller ไม่ได้"
        )


def _short(value: str, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…'"


_DQ_ESCAPES = str.maketrans({"\\": "\\\\", '"': '\\"', "$": "\\$", "`": "\\`"})


def dq(value: object) -> str:
    """escape สำหรับวางใน `"…"` ของ bash (และ heredoc แบบไม่ quote ซึ่งตีความชุดเดียวกัน)

    ใน double quote มีแค่ `\\` `"` `$` `` ` `` ที่มีความหมาย — ค่าปกติ (ตัวอักษร ตัวเลข `._-/:@` ช่องว่าง ไทย) ออกมา
    เหมือนเดิมทุกไบต์ จึงไม่กระทบ bundle ที่มีอยู่
    """
    text = str(value)
    _reject_control(text)
    return text.translate(_DQ_ESCAPES)


def dq_default(value: object) -> str:
    """escape สำหรับวางเป็นค่าตั้งต้นใน `"${VAR:-…}"`

    บริบทนี้มีของเพิ่มสองตัวที่ **ไม่มีวิธี escape ที่ bash ทุกรุ่นตีความตรงกัน** (ทดสอบ 2026-10-06):

    - `}` ปิด expansion กลางคัน · `\\}` ได้ `}` บน bash 5.3 แต่ได้ `\\}` บน bash 3.2
    - `'` เปิด quote ซ้อน แม้ทั้งก้อนจะอยู่ใน double quote — `"${X:-a'b}"` คือ syntax error ทั้งสองรุ่น

    จึงปฏิเสธแทนการเดา · ค่าที่ลงบริบทนี้ (ชื่อโมเดล image parser path) ไม่มีเหตุผลจะมีสองตัวนี้อยู่แล้ว
    """
    text = str(value)
    for char in "}'":
        if char in text:
            raise UnsafeValueError(
                f"ค่า {_short(text)} มี {char} ซึ่งวางเป็นค่าตั้งต้นใน ${{VAR:-…}} ของ bash ไม่ได้ "
                "(ไม่มีวิธี escape ที่ bash ทุกรุ่นตีความตรงกัน)"
            )
    return dq(text)


def quoted_either(value: object) -> str:
    """escape สำหรับตำแหน่งที่บรรทัดของ template มี `'` เปิดค้างอยู่ก่อนถึงค่า — ปลอดภัยทั้งใน `'…'` และใน `"…"`/heredoc

    = dq() แล้วแทน `'` ด้วย `'\\''` · ทำไมไม่ escape แบบ single quote ล้วน ๆ: renderer รู้บริบทจากบรรทัดเดียว และ `'` บนบรรทัด
    อาจเป็น quote ของ bash จริง (`printf ' … lmds set <slug> %s …'` ของ _knob_fix — ตำแหน่งแรกที่มี 2026-10-06) หรือเป็นแค่
    เครื่องหมายในข้อความของ heredoc/สตริงหลายบรรทัด · เดาผิดทางไหนก็ต้องไม่กลายเป็นคำสั่ง:
      - ใน `'…'` จริง: `'\\''` ถูกต้อง · `\\$` กลายเป็นตัวหนังสือสองตัว (ค่าที่มี $ เพี้ยนหน้าตา แต่ไม่ถูกรัน)
      - ใน `"…"`/heredoc จริง: `\\$` ถูกต้อง · `'\\''` กลายเป็นตัวหนังสือสี่ตัว (เพี้ยนหน้าตา แต่ไม่หลุด quote)
    ค่าปกติ (ไม่มี `\\ " $ \\` '`) ออกมาเหมือนเดิมทุกไบต์ทั้งสองทาง
    """
    return dq(value).replace("'", "'\\''")


_NUMBER = re.compile(r"[-+]?[0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]+)?|True|False")
# อักขระที่ shlex.quote ปล่อยไว้นอก quote (ชุดเดียวกับ `shlex._find_unsafe`)
_BARE_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-")


def _expansion_named(text: str, at: int) -> str:
    if text[at] == "`":
        return "มี ` (command substitution) ที่ template ไม่ได้ใส่"
    follower = text[at + 1:at + 2]
    if follower == "(":
        return "มี $( ที่ template ไม่ได้ใส่"
    if follower == "{":
        return "มี ${ ที่ template ไม่ได้ใส่"
    return "มี $ (expansion) ที่ template ไม่ได้ใส่"


def encoding_problem(kind: str, text: str) -> str:
    """เหตุผลที่ `text` (ค่าตามที่เขียนอยู่ใน controller) ไม่ใช่ผลของ encoder ชนิด `kind` — สตริงว่าง = ใช่

    คู่ตรวจของ dq / dq_default / quoted_either / words: gate ใช้ยืนยันว่าค่าทุกตัวในไฟล์ถูก encode มาครบ โดยไม่ต้องอ่าน quote ของทั้งไฟล์
    `kind`: "dq" · "default" · "either" · "words" · "number"
    """
    if _control_chars(text):
        return "มีอักขระควบคุม (ขึ้นบรรทัดใหม่ ฯลฯ) ในค่า"
    if kind == "number":
        return "" if _NUMBER.fullmatch(text) else "ตำแหน่งของตัวเลขมีอย่างอื่น — quote/คำสั่งหลุดเข้ามา"
    if kind == "either":
        text = text.replace("'\\''", "")
        if "'" in text:
            return "quote: มี ' ที่ไม่ได้ escape — ค่าหลุดออกนอก single quote"
    if kind == "words":
        state = "out"
        for at, char in enumerate(text):
            if state == "sq":
                if char == "'":
                    state = "out"
            elif state == "dq":
                # shlex.quote เขียน double quote แบบเดียว: "'" (ตัว ' ที่อยู่ในค่า)
                if char == '"':
                    state = "out"
                elif char in "$`":
                    return _expansion_named(text, at)
                elif char != "'":
                    return 'quote: ใน "…" ของคำที่ quote มามีอย่างอื่นนอกจาก \''
            elif char == "'":
                state = "sq"
            elif char == '"':
                state = "dq"
            elif char in "$`":
                return _expansion_named(text, at)
            elif char != " " and char not in _BARE_WORD:
                return f"quote: มี {char!r} อยู่นอก quote — ค่าหลุดออกมาเป็นไวยากรณ์ของเชลล์"
        return "" if state == "out" else "quote: quote ของคำที่ quote มาปิดไม่ครบ"
    # dq / default: ทุก \ " $ ` ต้องมี backslash นำ — และ backslash นำได้แค่สี่ตัวนี้
    at, size = 0, len(text)
    while at < size:
        char = text[at]
        if char == "\\":
            if text[at + 1:at + 2] in ("\\", '"', "$", "`") and at + 1 < size:
                at += 2
                continue
            return "quote: มี \\ ที่ไม่ได้ escape ในค่า"
        if char == '"':
            return 'quote: มี " ที่ไม่ได้ escape — ค่าหลุดออกนอก quote'
        if char in "$`":
            return _expansion_named(text, at)
        if kind == "default" and char in "}'":
            return f"quote: มี {char} ในค่าตั้งต้นของ ${{VAR:-…}} — ปิด expansion/เปิด quote กลางคัน"
        at += 1
    return ""


# ────────────────────────── กติกาของค่าแต่ละชนิด ──────────────────────────
# อักขระที่เชลล์ตีความ ไม่ว่าจะอยู่บริบทไหนสักแห่ง — ค่าที่ไปถึง ssh ฝั่ง worker ถูก parse ซ้ำอีกรอบโดยเชลล์ปลายทาง
# (stacked: `ssh … "docker inspect '$VLLM_IMAGE'"`) การ escape ตอน render จึงคุมได้แค่ชั้นแรก ชั้นที่สองต้องไม่มีของพวกนี้เลย
SHELL_ACTIVE = "$`\"'\\;&|<>(){}"
# ชื่อไฟล์ใน repo: ตัวอักษร/ตัวเลขทุกภาษา (รวมสระ-วรรณยุกต์ที่เป็น combining mark) กับเครื่องหมายชุดนี้เท่านั้น
REPO_NAME_PUNCTUATION = "._-+=@,/ "


def unsafe_chars(value: str, *, allow_space: bool = True, extra_ok: str = "") -> list[str]:
    """อักขระใน `value` ที่เชลล์ตีความ (หรือเป็นอักขระควบคุม/ช่องว่างเมื่อไม่อนุญาต) — ลิสต์ว่าง = ใช้ได้"""
    bad: set[str] = set(_control_chars(value))
    for char in value:
        if char in SHELL_ACTIVE and char not in extra_ok:
            bad.add(char)
        elif char.isspace() and not (allow_space and char == " "):
            bad.add(char)
    return sorted(bad)


def repo_filename_problem(name: str) -> str:
    """เหตุผลที่ชื่อไฟล์จาก Hub นี้ใช้กับ controller ไม่ได้ — สตริงว่าง = ใช้ได้

    รับ: ตัวอักษรและตัวเลขทุกภาษา · `. _ - + = @ ,` · `/` คั่นโฟลเดอร์ · ช่องว่าง
    ไม่รับ: อย่างอื่นทั้งหมด (`$ ` " ' \\ ; & | < > ( ) { } [ ] * ? ! # ~ % ^ :` อักขระควบคุม) และ path ที่มี `..`
    """
    if not name:
        return "ชื่อว่าง"
    bad = sorted({
        c for c in name
        if c not in REPO_NAME_PUNCTUATION and unicodedata.category(c)[0] not in ("L", "M", "N")
    })
    if bad:
        return "มีอักขระนอกชุดที่รองรับ: " + " ".join(repr(c) for c in bad)
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "path ไม่ปกติ (มี .. หรือ / ซ้อน/นำหน้า/ต่อท้าย)"
    if name != name.strip():
        return "ขึ้นต้นหรือลงท้ายด้วยช่องว่าง"
    return ""


def is_safe_repo_filename(name: str) -> bool:
    return not repo_filename_problem(name)


# ชื่อที่ API เสิร์ฟ เมื่อมาจาก *แผน* (LLM เป็นคนเสนอได้) — แคบกว่าที่ `lmds set --served-name` ยอมให้คนพิมพ์เอง
def is_plain_served_name(name: str) -> bool:
    return bool(name) and all(c.isascii() and (c.isalnum() or c in "._:/-") for c in name)


# ชื่อ bundle (slug) ที่มาจากข้างนอก — URL ของหน้าเว็บ · argument ของเครื่องมือ MCP (`lmds mcp`) ที่ผู้ช่วย AI ส่งมา
# ตามที่หน้าเว็บ/เอกสารที่มันอ่านบอก · ค่านี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นและเป็นชื่อไฟล์ จึงตรวจรูปตั้งแต่ปากทาง:
# slug ที่ LMDS สร้างเองมีแค่ตัวอักษรชุดนี้ (ดูที่มาใน web/api.py · review 2026-09-04) · อยู่ที่นี่เพื่อให้ปากทางทุกบาน
# ใช้ตัวเดียวกัน — MCP import ของหน้าเว็บไม่ได้ (fastapi เป็น optional extra)
BUNDLE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def is_bundle_slug(slug: object) -> bool:
    return isinstance(slug, str) and bool(BUNDLE_SLUG.fullmatch(slug))
