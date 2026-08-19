"""ตรวจว่าโมเดล *ทำอะไรได้จริง* ไม่ใช่แค่เร็วแค่ไหน

เร็วอย่างเดียวไม่พอ: โมเดลที่ decode 60 tok/s แต่เรียก tool ไม่ได้ ใช้กับ agent ไม่ได้เลย
ส่วนตัวที่ 25 tok/s แต่ครบทุกอย่างคือตัวที่เอาไปใช้งานจริงได้ · หน้าคะแนนจึงต้องมีสองแกน

ทุกข้อวัดจาก *พฤติกรรมจริงที่ปลายทาง* ไม่ใช่จากค่าที่เขียนไว้ใน MODEL_PROFILE — เพราะสิ่งที่
เจอบ่อยที่สุดคือ profile บอกว่ารองรับ แต่เซิร์ฟเวอร์ที่รันอยู่ไม่ได้เปิด flag ที่ต้องใช้
(เคสจริง: llama.cpp ที่ไม่ได้ใส่ `--jinja` → tool calling ตายเงียบ ๆ ทั้งที่ profile บอกว่ามี)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import httpx

# PNG สีแดงล้วน 8×8 — เล็กพอที่จะฝังในโค้ดและไม่กินเวลา encode
_RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAG0lEQVQoz2P8z8Dwn4"
    "GKgIlhVMOohlENoxpGNQAAaEwB/1PBpKQAAAAASUVORK5CYII="
)

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "ดูสภาพอากาศของเมืองหนึ่ง",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "ชื่อเมือง"}},
            "required": ["city"],
        },
    },
}


@dataclass
class Probe:
    key: str
    label: str
    passed: bool
    detail: str = ""
    skipped: bool = False


# งบ token ต่อหนึ่งข้อ — ใหญ่โดยตั้งใจ
#
# เคสจริง 2026-08-19: ตั้ง max_tokens=32 แล้วถาม muse-glimmer ว่าเมืองหลวงฝรั่งเศสคืออะไร
# ได้ finish_reason="length" กับ content ว่าง เพราะโมเดลใช้ทั้ง 32 token ไปกับการคิด
# ตัววัดสรุปว่า "ทำตามคำสั่งไม่ได้" · พอให้ 1024 token มันตอบ "ปารีส" ทันที
#
# โมเดลสาย reasoning คิดก่อนตอบเป็นปกติ — งบที่ตึงเกินไปไม่ได้วัดความสามารถ
# แต่วัดว่าเราให้เวลามันพอไหม
_BUDGET = 1024
_BUDGET_LONG = 2048


def _chat(client: httpx.Client, endpoint: str, model: str, **body) -> dict:
    body.setdefault("model", model)
    body.setdefault("max_tokens", _BUDGET)
    response = client.post(f"{endpoint}/chat/completions", json=body, timeout=300.0)
    response.raise_for_status()
    return response.json()


def _message(payload: dict) -> dict:
    choices = payload.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _finish_reason(payload: dict) -> str:
    choices = payload.get("choices") or []
    return (choices[0].get("finish_reason") or "") if choices else ""


def _answer(payload: dict) -> tuple[str, str]:
    """(คำตอบ, เหตุผลที่คำตอบว่าง) — แยกให้ชัดว่า "ตอบผิด" กับ "ไม่ทันได้ตอบ" คนละเรื่อง"""
    message = _message(payload)
    content = (message.get("content") or "").strip()
    if content:
        return content, ""
    thinking = message.get("reasoning_content") or ""
    if _finish_reason(payload) == "length":
        note = (f"งบ token หมดตอนกำลังคิด (คิดไป {len(thinking)} ตัวอักษร ยังไม่ทันตอบ)"
                if thinking else "งบ token หมดก่อนตอบ")
        return "", note
    return "", "เซิร์ฟเวอร์ตอบกลับมาว่าง"


def _probe_instructions(client, endpoint, model) -> Probe:
    """ทำตามคำสั่งที่คุมรูปแบบได้ไหม — ข้อพื้นฐานที่สุด ถ้าตกข้อนี้ที่เหลือไม่ต้องดู"""
    try:
        text, blocked = _answer(_chat(
            client, endpoint, model,
            messages=[{"role": "user", "content":
                       "ตอบด้วยคำเดียวเท่านั้น ห้ามมีเครื่องหมายวรรคตอน: เมืองหลวงของฝรั่งเศสคือ"}]))
        if blocked:
            return Probe("instructions", "ทำตามคำสั่ง", False, blocked)
        ok = "paris" in text.lower() or "ปารีส" in text
        return Probe("instructions", "ทำตามคำสั่ง", ok, text[:80])
    except Exception as exc:
        return Probe("instructions", "ทำตามคำสั่ง", False, str(exc)[:120])


def _probe_thai(client, endpoint, model) -> Probe:
    """ถามไทยแล้วตอบไทยไหม — โรงเรียนที่ใช้งานจริงต้องการข้อนี้มากกว่าคะแนนอังกฤษ"""
    try:
        text, blocked = _answer(_chat(
            client, endpoint, model,
            messages=[{"role": "user", "content":
                       "อธิบายสั้น ๆ ว่าทำไมท้องฟ้าถึงเป็นสีฟ้า ตอบเป็นภาษาไทย"}]))
        if blocked:
            return Probe("thai", "ตอบภาษาไทย", False, blocked)
        thai_chars = sum(1 for ch in text if "฀" <= ch <= "๿")
        ok = thai_chars >= 30
        return Probe("thai", "ตอบภาษาไทย", ok, f"อักษรไทย {thai_chars} ตัว")
    except Exception as exc:
        return Probe("thai", "ตอบภาษาไทย", False, str(exc)[:120])


def _probe_json(client, endpoint, model) -> Probe:
    """คืน JSON ที่ parse ได้จริงไหม — structured output คือฐานของ integration ส่วนใหญ่"""
    try:
        text, blocked = _answer(_chat(
            client, endpoint, model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       'ตอบเป็น JSON object ที่มีคีย์ "city" และ "country" สำหรับกรุงเทพมหานคร'}]))
        if blocked:
            return Probe("json", "JSON structured output", False, blocked)
        parsed = json.loads(text)
        ok = isinstance(parsed, dict) and "city" in parsed
        return Probe("json", "JSON structured output", ok, text[:80])
    except Exception as exc:
        return Probe("json", "JSON structured output", False, str(exc)[:120])


def _probe_tools(client, endpoint, model) -> Probe:
    """ออก tool_calls ที่มีชื่อฟังก์ชันและ arguments ที่ parse ได้ไหม

    ไม่ยอมรับการที่โมเดลพิมพ์ JSON ของ tool call ลงใน content — agent จริงอ่านจาก
    ฟิลด์ `tool_calls` เท่านั้น ตัวที่พิมพ์ลง content คือตัวที่ใช้กับ agent ไม่ได้
    """
    try:
        payload = _chat(
            client, endpoint, model, tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "อากาศที่เชียงใหม่ตอนนี้เป็นยังไง"}])
        message = _message(payload)
        calls = message.get("tool_calls") or []
        if not calls:
            return Probe("tools", "Tool calling", False, "ไม่มีฟิลด์ tool_calls ในคำตอบ")
        function = (calls[0].get("function") or {})
        arguments = json.loads(function.get("arguments") or "{}")
        ok = function.get("name") == "get_weather" and "city" in arguments
        return Probe("tools", "Tool calling", ok,
                     f"{function.get('name')}({arguments})"[:80])
    except Exception as exc:
        return Probe("tools", "Tool calling", False, str(exc)[:120])


def _probe_reasoning(client, endpoint, model) -> Probe:
    """แยกความคิดออกจากคำตอบไหม — ถ้าไม่แยก client จะเอา chain-of-thought ไปโชว์ผู้ใช้"""
    try:
        message = _message(_chat(
            client, endpoint, model, max_tokens=_BUDGET_LONG,
            messages=[{"role": "user", "content":
                       "ร้านขายส้ม 3 ลูก 25 บาท ถ้าซื้อ 12 ลูกจ่ายเท่าไร คิดทีละขั้น"}]))
        thinking = message.get("reasoning_content") or ""
        answer = message.get("content") or ""
        if not thinking:
            return Probe("reasoning", "แยก reasoning", False,
                         "ไม่มี reasoning_content (ไม่ได้ตั้ง --reasoning-parser หรือโมเดลไม่คิดก่อนตอบ)")
        ok = bool(answer.strip())
        return Probe("reasoning", "แยก reasoning", ok,
                     f"คิด {len(thinking)} ตัวอักษร · ตอบ {len(answer)} ตัวอักษร")
    except Exception as exc:
        return Probe("reasoning", "แยก reasoning", False, str(exc)[:120])


def _probe_vision(client, endpoint, model, has_projector: bool) -> Probe:
    if not has_projector:
        return Probe("vision", "รับภาพ", False, "ไม่มี mmproj — ข้าม", skipped=True)
    try:
        data_url = "data:image/png;base64," + base64.b64encode(_RED_PNG).decode()
        payload = _chat(
            client, endpoint, model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "ภาพนี้เป็นสีอะไร ตอบสั้น ๆ"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}])
        text, blocked = _answer(payload)
        if blocked:
            return Probe("vision", "รับภาพ", False, blocked)
        lowered = text.lower()
        ok = "red" in lowered or "แดง" in text
        return Probe("vision", "รับภาพ", ok, text[:80])
    except Exception as exc:
        return Probe("vision", "รับภาพ", False, str(exc)[:120])


def _probe_recall(client, endpoint, model, context_limit: int) -> Probe:
    """หาเข็มในกองฟาง — context ที่โฆษณาไว้ใช้ได้จริงถึงไหน

    วางรหัสไว้กลางข้อความยาว ๆ แล้วถามหา · โมเดลที่ตั้ง context 256K แต่ recall ที่ 8K
    ยังพลาด คือตัวเลขบนกระดาษที่เอาไปวางแผนงานไม่ได้
    """
    if context_limit and context_limit < 6000:
        return Probe("recall", "จำ context ยาว", False, "context สั้นเกินไป — ข้าม", skipped=True)
    needle = "รหัสยืนยันของโครงการคือ QUAIL-7742"
    filler = ("บันทึกการประชุมประจำสัปดาห์ระบุว่าทีมได้ทบทวนรายการงานค้าง "
              "และตกลงเลื่อนการทดสอบภาคสนามออกไปอีกหนึ่งรอบ ")
    half = filler * 120
    prompt = f"{half}\n\n{needle}\n\n{half}\n\nคำถาม: รหัสยืนยันของโครงการคืออะไร ตอบเฉพาะรหัส"
    try:
        text, blocked = _answer(_chat(
            client, endpoint, model, max_tokens=_BUDGET_LONG,
            messages=[{"role": "user", "content": prompt}]))
        if blocked:
            return Probe("recall", "จำ context ยาว", False, blocked)
        ok = "QUAIL-7742" in text.upper()
        return Probe("recall", "จำ context ยาว", ok, text[:60])
    except Exception as exc:
        return Probe("recall", "จำ context ยาว", False, str(exc)[:120])


def run_probes(endpoint: str, model: str, has_projector: bool = False,
               context_limit: int = 0, api_key: str = "", on_progress=None) -> list[Probe]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    probes: list[Probe] = []
    with httpx.Client(headers=headers) as client:
        steps = (
            ("instructions", lambda: _probe_instructions(client, endpoint, model)),
            ("thai", lambda: _probe_thai(client, endpoint, model)),
            ("json", lambda: _probe_json(client, endpoint, model)),
            ("tools", lambda: _probe_tools(client, endpoint, model)),
            ("reasoning", lambda: _probe_reasoning(client, endpoint, model)),
            ("vision", lambda: _probe_vision(client, endpoint, model, has_projector)),
            ("recall", lambda: _probe_recall(client, endpoint, model, context_limit)),
        )
        for name, step in steps:
            if on_progress:
                on_progress(name)
            probes.append(step())
    return probes
