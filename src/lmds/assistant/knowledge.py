"""ความรู้ของผู้ช่วย — วิธีคิด (playbook) กับเอกสารจริงของโปรเจกต์ (docs/)

แยกกันสองชั้นโดยตั้งใจ:

  **playbook.md** คือ *วิธีคิด* — สั้น อยู่ใน prompt ทุกครั้ง เปลี่ยนไม่บ่อย
  **docs/** คือ *ข้อเท็จจริง* — ยาวมาก (USAGE.md ไฟล์เดียวแสนกว่าตัวอักษร) ทีมแก้อยู่เรื่อย ๆ
  ค้นเอาเฉพาะตอนที่ต้องใช้

ทำไมไม่ยัด docs ลง prompt ให้หมด: นอกจากจะไม่พอที่แล้ว การคัดลอกตารางอาการ→วิธีแก้
มาไว้ใน prompt แปลว่าวันที่ทีมแก้ตารางจริง ผู้ช่วยจะยังตอบสูตรเก่าอย่างมั่นใจต่อไป
โดยไม่มีใครรู้ · ค้นจากไฟล์จริงทำให้คำตอบเก่าไม่ได้เกินกว่าที่ repo เป็น

ไม่มี embedding ไม่มี vector store โดยตั้งใจ — คลังนี้เล็กพอ (สิบกว่าไฟล์) ที่การให้
คะแนนด้วยคำตรง ๆ ทำงานได้ดีพอ และไม่ต้องพึ่ง service ภายนอกหรือไฟล์ index ที่ต้องสร้างใหม่
ทุกครั้งที่เอกสารเปลี่ยน
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MAX_SECTION_CHARS = 1800
MAX_RESULTS = 3

# ไฟล์ที่ให้ค้น — ไม่เอา CHANGELOG (ยาวและเป็นประวัติ ไม่ใช่วิธีทำ)
DOC_FILES = (
    "USAGE.md",
    "RUNBOOK-MULTI-NODE.md",
    "PREFLIGHT.md",
    "FLEET-MULTI-NODE.md",
    "NETWORK.md",
    "INSTALL.md",
    "DGX-SPARK-VLLM-FIELD-NOTES.md",
    "BENCH.md",
    "PRD.md",
    "CLI_SPEC.md",
)

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_WORD = re.compile(r"[a-zA-Z0-9_.\-]{2,}|[฀-๿]{3,}")


@dataclass(frozen=True)
class Section:
    doc: str
    heading: str
    body: str

    def payload(self) -> dict:
        text = self.body.strip()
        if len(text) > MAX_SECTION_CHARS:
            text = text[:MAX_SECTION_CHARS] + "\n…(ตัดต่อ)"
        return {"doc": self.doc, "heading": self.heading, "text": text}


def _docs_dir() -> Path:
    """หา docs/ — สำเนาที่มากับแพ็กเกจก่อน แล้วค่อยมองหา repo รอบตัว

    เครื่องที่ติดตั้งด้วย pip ไม่มี repo ให้ค้น มีแต่ site-packages · wheel จึงพา docs/
    ไปด้วย (ดู pyproject `force-include`) ไม่งั้นฟีเจอร์ค้นเอกสารตายเงียบ ๆ บนทุก node
    ที่ไม่ใช่เครื่องพัฒนา — เจอจริงตอนติดตั้งครั้งแรก: playbook มาครบ แต่ค้นได้ 0 ไฟล์

    เครื่องพัฒนายังได้ของสดจาก repo เพราะ `pip install -e .` ไม่ได้ก็อป docs เข้าไป
    การไล่หาขึ้นไปตามชั้นจึงยังจำเป็น ไม่ใช่โค้ดที่เหลือค้าง
    """
    here = Path(__file__).resolve()
    packaged = here.parent / "_docs"
    if (packaged / "USAGE.md").exists():
        return packaged
    for parent in here.parents:
        candidate = parent / "docs"
        if candidate.is_dir() and (candidate / "USAGE.md").exists():
            return candidate
    return packaged


def playbook() -> str:
    """วิธีคิดแบบ LMDS — ไฟล์ข้าง ๆ โมดูลนี้ จึงติดไปกับแพ็กเกจเสมอ"""
    path = Path(__file__).resolve().parent / "playbook.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        # ไม่มี playbook ก็ยังตอบได้ แค่ตอบแบบไม่มีวิธีคิดกำกับ — อย่าให้แชททั้งกล่องล้ม
        return ""


@lru_cache(maxsize=1)
def _sections() -> tuple[Section, ...]:
    """หั่นเอกสารทุกไฟล์ตามหัวข้อ — แคชไว้ เพราะอ่านใหม่ทุกคำถามคือหลายแสนตัวอักษร"""
    found: list[Section] = []
    root = _docs_dir()
    for name in DOC_FILES:
        path = root / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        marks = list(_HEADING.finditer(text))
        if not marks:
            continue
        for index, mark in enumerate(marks):
            start = mark.end()
            end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
            body = text[start:end].strip()
            if body:
                found.append(Section(doc=name, heading=mark.group(2).strip(), body=body))
    return tuple(found)


def _terms(query: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(query or "")][:12]


def search_docs(query: str, limit: int = MAX_RESULTS) -> list[dict]:
    """ค้นหัวข้อที่เกี่ยวข้องที่สุดจากเอกสารจริง

    ให้คะแนนแบบตรงไปตรงมา: คำที่อยู่ใน *หัวข้อ* มีน้ำหนักกว่าคำที่อยู่ในเนื้อ เพราะ
    หัวข้อของเอกสารชุดนี้เขียนเป็นอาการจริง ("start ทับ port ของโมเดลอื่นแล้วบอกว่าสำเร็จ")
    ซึ่งตรงกับสิ่งที่ผู้ใช้พิมพ์มามากกว่าคำที่บังเอิญโผล่กลางย่อหน้า
    """
    terms = _terms(query)
    if not terms:
        return []
    scored: list[tuple[int, Section]] = []
    for section in _sections():
        heading = section.heading.lower()
        body = section.body.lower()
        score = 0
        for term in terms:
            if term in heading:
                score += 5
            if term in body:
                score += 1
        if score:
            scored.append((score, section))
    scored.sort(key=lambda pair: (-pair[0], pair[1].doc, pair[1].heading))
    return [section.payload() for _, section in scored[: max(1, min(limit, 5))]]


def doc_index() -> list[str]:
    """ชื่อไฟล์ที่ค้นได้จริงตอนนี้ — ใส่ใน prompt ให้ผู้ช่วยรู้ว่ามีอะไรให้ค้น"""
    root = _docs_dir()
    return [name for name in DOC_FILES if (root / name).exists()]
