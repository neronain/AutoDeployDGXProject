"""โมเดลตัวนี้ทำอะไรได้บ้าง — ตอบจากไฟล์ ก่อนดาวน์โหลด ก่อน deploy

เดิมรู้ตอนรันแล้วเท่านั้น ซึ่งแปลว่ารู้หลังจากโหลด weight หลายสิบกิกะ สร้าง bundle
ส่งข้ามเครื่อง แล้วนั่งรอ /health · ถ้ามันไม่มีสิ่งที่เราต้องการ ทั้งหมดนั้นเสียเปล่า
คำถาม "โมเดลนี้เรียก tool ได้ไหม" ควรตอบได้ตั้งแต่ตอนเลือกโมเดล ไม่ใช่ตอนติดตั้งเสร็จ

สิ่งที่ตอบได้จากไฟล์จริง ๆ:

  chat template คือหลักฐานที่ดีที่สุดและถูกมองข้ามบ่อยที่สุด — มันคือสิ่งที่กำหนดว่า
  โมเดลจะ *ถูกป้อน* tool/system/thinking ยังไง ถ้า template ไม่มีที่ทางให้ tools
  ต่อให้เปิด --tool-call-parser ก็ไม่มีอะไรให้ parse

  config.json บอกเรื่อง vision: vision_config, architectures ที่ลงท้ายด้วย
  ForConditionalGeneration, หรือมี processor

สิ่งที่ตอบจากไฟล์ **ไม่ได้** และต้องพูดให้ชัดว่าไม่ได้:

  JSON mode และ streaming เป็นความสามารถของ *เซิร์ฟเวอร์* ไม่ใช่ของโมเดล — vLLM
  และ llama.cpp ทำได้ทั้งคู่กับทุกโมเดล การไปบอกว่า "โมเดลนี้ทำ JSON mode ไม่ได้"
  จึงผิดตั้งแต่ตั้งคำถาม

  และ template ที่รองรับ tools ไม่ได้แปลว่าเซิร์ฟเวอร์จะ *แปลง* คำตอบเป็น tool_calls
  ให้ — อันนั้นต้องมี --tool-call-parser ที่ตรงตระกูล และพิสูจน์ได้ด้วยการยิงจริง
  เท่านั้น (ฝั่ง LiteGate วัดให้) · ที่นี่บอกได้แค่ว่า "มีทางเป็นไปได้" กับ "ไม่มีทาง"

ความต่างนั้นสำคัญพอที่จะแยกเป็นสองสถานะ ไม่ใช่ yes/no
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# yes    = ไฟล์ยืนยันชัด
# likely = มีร่องรอยแต่ยังต้องพิสูจน์ตอนรัน
# no     = ไฟล์บอกว่าไม่มี
# server = ขึ้นกับเซิร์ฟเวอร์ ไม่ใช่โมเดล
# unknown= ไม่มีข้อมูลพอจะบอก
STATUSES = ("yes", "likely", "no", "server", "unknown")


@dataclass
class Capability:
    name: str
    status: str
    evidence: str
    # สิ่งที่ยังต้องตรวจตอนรัน — เขียนไว้ตรงนี้ดีกว่าปล่อยให้คนคิดว่า yes แปลว่าจบ
    caveat: str = ""


@dataclass
class CapabilityReport:
    capabilities: list[Capability] = field(default_factory=list)

    def get(self, name: str) -> Capability | None:
        return next((c for c in self.capabilities if c.name == name), None)

    def to_dict(self) -> dict:
        return {
            c.name: {"status": c.status, "evidence": c.evidence, "caveat": c.caveat}
            for c in self.capabilities
        }


# ── chat template: หลักฐานหลัก ────────────────────────────────────────────
# มองหา "ที่ทางของ tools ใน template" ไม่ใช่แค่คำว่า tool โผล่ที่ไหนสักแห่ง
_TOOLS = re.compile(r"\btools\b|\btool_call|\btool_calls\b|\btool_use\b", re.I)
_THINK = re.compile(r"<think>|</think>|\bthinking\b|reasoning_content|<\|thought", re.I)
_SYSTEM = re.compile(r"['\"]system['\"]|\bsystem_message\b|\bsystem_prompt\b", re.I)


def _from_template(template: str) -> dict[str, bool]:
    return {
        "tools": bool(_TOOLS.search(template)),
        "thinking": bool(_THINK.search(template)),
        "system": bool(_SYSTEM.search(template)),
    }


def detect(
    config: dict | None = None,
    chat_template: str = "",
    *,
    has_mmproj: bool | None = None,
    server: str = "",
) -> CapabilityReport:
    """อ่านความสามารถจากไฟล์ของโมเดล

    `config`      config.json
    `chat_template` เนื้อ template (จาก chat_template.jinja หรือ tokenizer_config)
    `has_mmproj`  สำหรับ GGUF — มีไฟล์ projector ไหม (None = ไม่ใช่ GGUF)
    `server`      vllm | llamacpp — ใช้ตอบเรื่องที่เป็นของเซิร์ฟเวอร์
    """
    config = config or {}
    found = _from_template(chat_template) if chat_template else {}
    report = CapabilityReport()

    # ── vision ────────────────────────────────────────────────────────────
    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    vision_config = config.get("vision_config") or (config.get("text_config") and config.get("vision_config"))
    if has_mmproj is True:
        report.capabilities.append(Capability(
            "vision", "yes", "GGUF repo มีไฟล์ mmproj (projector)",
            "llama.cpp ต้องได้รับ --mmproj ตอน start ไม่งั้นภาพจะถูกปฏิเสธ"))
    elif has_mmproj is False:
        report.capabilities.append(Capability(
            "vision", "no", "GGUF repo ไม่มีไฟล์ mmproj — โหลดภาพไม่ได้", ""))
    elif vision_config:
        report.capabilities.append(Capability(
            "vision", "yes",
            f"config.json มี vision_config ({vision_config.get('model_type') or 'ไม่ระบุชนิด'})",
            "จำนวน visual token ต่อภาพกินโควตา context — ดูตอนตั้ง --max-model-len"))
    elif architecture.endswith("ForConditionalGeneration") or config.get("processor_class"):
        report.capabilities.append(Capability(
            "vision", "likely",
            f"architecture '{architecture}' เป็นรูปแบบของโมเดล multimodal",
            "ยังไม่พบ vision_config — ยืนยันด้วยการส่งภาพจริงหลัง start"))
    elif config:
        report.capabilities.append(Capability(
            "vision", "no", "config.json ไม่มี vision_config — เป็นโมเดลข้อความล้วน", ""))
    else:
        report.capabilities.append(Capability("vision", "unknown", "ไม่มี config.json ให้ดู", ""))

    # ── tool calling ──────────────────────────────────────────────────────
    if not chat_template:
        report.capabilities.append(Capability(
            "tool_calling", "unknown", "ไม่มี chat template ให้ดู",
            "ไม่มี template = ไม่มีทางป้อน tool ให้โมเดลตามรูปแบบ chat"))
    elif found["tools"]:
        report.capabilities.append(Capability(
            "tool_calling", "likely", "chat template มีที่ทางสำหรับ tools",
            "template รับ tool ได้ ไม่ได้แปลว่าเซิร์ฟเวอร์จะแปลงคำตอบเป็น tool_calls — "
            "vLLM ต้องได้ --enable-auto-tool-choice + --tool-call-parser ที่ตรงตระกูล "
            "แล้วพิสูจน์ด้วย test-tools"))
    else:
        report.capabilities.append(Capability(
            "tool_calling", "no", "chat template ไม่มีที่ทางสำหรับ tools",
            "เปิด parser ก็ไม่ช่วย — ไม่มีอะไรให้ parse ตั้งแต่ต้นทาง"))

    # ── reasoning / thinking ──────────────────────────────────────────────
    if not chat_template:
        report.capabilities.append(Capability("reasoning", "unknown", "ไม่มี chat template ให้ดู", ""))
    elif found["thinking"]:
        report.capabilities.append(Capability(
            "reasoning", "likely", "chat template มีร่องรอยของ thinking (<think> หรือเทียบเท่า)",
            "ถ้าไม่ตั้ง --reasoning-parser ความคิดจะปนมาในคำตอบ ทุกฝั่งที่เอาไปแสดง"
            "ต้องมาตัดเอง — ตั้ง parser แล้วพิสูจน์ด้วย test-reasoning"))
    else:
        report.capabilities.append(Capability(
            "reasoning", "no", "chat template ไม่มีร่องรอยของ thinking", ""))

    # ── system prompt ─────────────────────────────────────────────────────
    if not chat_template:
        report.capabilities.append(Capability("system_prompt", "unknown", "ไม่มี chat template ให้ดู", ""))
    elif found["system"]:
        report.capabilities.append(Capability(
            "system_prompt", "yes", "chat template รองรับ role 'system'", ""))
    else:
        report.capabilities.append(Capability(
            "system_prompt", "no", "chat template ไม่รับ role 'system'",
            "client ที่ส่ง system มาจะถูกเมิน หรือถูกยัดรวมกับ user แล้วแต่เซิร์ฟเวอร์"))

    # ── streaming / json mode: ของเซิร์ฟเวอร์ ไม่ใช่ของโมเดล ───────────────
    engine = {"vllm": "vLLM", "llamacpp": "llama.cpp"}.get(server, "เซิร์ฟเวอร์")
    report.capabilities.append(Capability(
        "streaming", "server", f"{engine} สตรีมได้กับทุกโมเดล",
        "ผ่าน proxy ต้องปิด buffering ไม่งั้น token จะมาทีเดียวตอนจบ"))
    report.capabilities.append(Capability(
        "json_mode", "server",
        "vLLM ใช้ guided decoding · llama.cpp ใช้ GBNF grammar — ทำได้กับทุกโมเดล"
        if not server else f"{engine} บังคับรูปแบบผลลัพธ์ได้กับทุกโมเดล",
        "โมเดลเล็กอาจทำตาม schema ที่ซับซ้อนได้แย่ ถึงจะบังคับรูปแบบได้ก็ตาม"))

    return report
