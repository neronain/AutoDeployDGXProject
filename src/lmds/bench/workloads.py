"""ชุดงานมาตรฐานที่ใช้วัดทุกโมเดลให้เหมือนกัน

ทำไมต้องมีชุดตายตัว: ตัวเลข tok/s ที่วัดคนละ prompt คนละความยาว เอามาเทียบกันไม่ได้เลย
ความเร็ว decode ของ llama.cpp ตกตามความยาว context อย่างชัดเจน — วัดที่ 512 token
แล้วบอกว่า "โมเดลนี้เร็วกว่า" ทั้งที่อีกตัววัดที่ 8K คือการเทียบคนละเรื่องกัน

prompt สร้างจากข้อความตายตัว ไม่สุ่ม — รันซ้ำวันหลังต้องได้ภาระงานเดียวกันเป๊ะ ถึงจะ
เอาผลก่อน/หลังอัปเกรด llama.cpp มาเทียบกันได้
"""

from __future__ import annotations

from dataclasses import dataclass

# ย่อหน้าฐานสำหรับสร้าง prompt ยาว — เนื้อหาเป็นกลาง ไม่พาโมเดลไปโหมด refusal
# และไม่ใช่ข้อความที่โมเดลน่าจะท่องจำมาแล้ว (ซึ่งจะทำให้ prefill cache ได้เปรียบผิดปกติ)
_FILLER = (
    "The maintenance log records that the pump on the second floor was inspected on a "
    "Tuesday morning. The technician noted a slight vibration at the coupling, measured "
    "the bearing temperature at forty-one degrees, and replaced the gasket on the outlet "
    "flange. Downstream pressure returned to the expected range within twenty minutes. "
    "A follow-up inspection was scheduled for the following month, and the spare gasket "
    "inventory was reduced to four units, which is below the reorder threshold of six. "
)

# โค้ดตัวอย่างสำหรับงานอ่านโค้ด — ต้องเป็นโค้ดที่ *มีบั๊กจริง* ไม่งั้นคำตอบจะกลวง
_CODE_UNIT = '''
def merge_ranges(ranges):
    """Merge overlapping [start, end] pairs."""
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start < last[1]:
            merged[-1] = (last[0], max(last[1], end))
        else:
            merged.append((start, end))
    return merged
'''


@dataclass(frozen=True)
class Workload:
    """งานหนึ่งชิ้น — ความยาว input ที่ตั้งใจ กับจำนวน token ที่ขอให้เขียนออกมา

    `input_tokens` เป็น *เป้าหมาย* ไม่ใช่ค่าจริง — tokenizer ของแต่ละโมเดลไม่เท่ากัน
    ผลที่บันทึกใช้ `prompt_tokens` ที่เซิร์ฟเวอร์รายงานกลับมาเสมอ
    """

    key: str
    label: str
    input_tokens: int
    output_tokens: int
    kind: str = "text"

    def prompt(self, nonce: str = "") -> str:
        """nonce ไปอยู่ *หัว* prompt เพื่อทำลาย prefix cache

        เคสจริง 2026-08-19: ยิง prompt เดิมซ้ำ 3 รอบบน llama.cpp ได้ `cached_tokens: 76`
        รอบหลัง ๆ ข้าม prefill ไปเลย — TTFT ที่วัดได้เหลือ 0.61 วิ สำหรับ 3,282 token
        (prefill 5,389 tok/s ซึ่งเป็นไปไม่ได้บน GB10) · ถ้าไม่ทำลาย cache เรากำลังวัด
        ความเร็วของ cache ไม่ใช่ของโมเดล
        """
        body = _build_prompt(self)
        return f"[อ้างอิง {nonce}]\n{body}" if nonce else body


def _pad_to(text: str, target_tokens: int) -> str:
    """ต่อ filler จนยาวประมาณ target_tokens

    ~1.3 token ต่อคำสำหรับ filler ชุดนี้ (วัดจริงจาก tokenizer ของ Qwen/Gemma: 682 คำ
    → 893 token) · ค่านี้เป็นแค่การเล็ง ผลที่บันทึกใช้ `prompt_tokens` จริงจากเซิร์ฟเวอร์เสมอ
    """
    words_needed = max(8, int(target_tokens / 1.3))
    words = text.split()
    filler = _FILLER.split()
    while len(words) < words_needed:
        words.extend(filler)
    return " ".join(words[:words_needed])


def _build_prompt(workload: Workload) -> str:
    if workload.kind == "code":
        body = _CODE_UNIT * max(1, workload.input_tokens // 120)
        return (
            "อ่านโค้ดข้างล่างนี้แล้วอธิบายว่ามันทำอะไร ชี้จุดที่เป็นบั๊ก และบอกวิธีแก้\n\n"
            f"```python\n{body}\n```"
        )
    if workload.kind == "write":
        return _pad_to(
            "เขียนบันทึกการเดินทางสั้น ๆ เกี่ยวกับเมืองริมทะเลที่ฝนเพิ่งหยุดตก "
            "ใช้ภาษาบรรยายที่เห็นภาพ ไม่ต้องมีหัวข้อย่อย บริบทประกอบ: ",
            workload.input_tokens,
        )
    return _pad_to(
        "สรุปเนื้อหาข้างล่างนี้เป็นย่อหน้าเดียว แล้วตามด้วยข้อสังเกตที่สำคัญที่สุดสามข้อ\n\n",
        workload.input_tokens,
    )


# ชุดเต็ม — ไล่ context จาก 512 ถึง 8K เพื่อให้เห็นว่าความเร็วตกตรงไหน
FULL: tuple[Workload, ...] = (
    Workload("content-512-256", "Content Generation", 512, 256),
    Workload("creative-512-512", "Creative Writing", 512, 512, kind="write"),
    Workload("summary-2k-256", "Summarization Light", 2048, 256),
    Workload("summary-4k-256", "Summarization Moderate", 4096, 256),
    Workload("code-4k-256", "Code Analysis", 4096, 256, kind="code"),
    Workload("summary-8k-256", "Summarization Long", 8192, 256),
)

# ชุดเร็ว — ใช้ตอนอยากรู้คร่าว ๆ ว่ายังทำงานอยู่ไหม ไม่ใช่ตอนจะเอาไปเทียบ
QUICK: tuple[Workload, ...] = (FULL[0], FULL[2])


def select(profile: str = "full", context_limit: int = 0) -> tuple[Workload, ...]:
    """เลือกชุดงานตามโปรไฟล์ และตัดงานที่ยาวเกิน context ของโมเดลออก

    โมเดลที่ context 4096 ถูกยิงงาน 8K จะได้ error กลับมาทุกครั้ง — นับเป็น "สอบตก"
    ทั้งที่มันแค่ไม่ได้ถูกตั้งให้ยาวขนาดนั้น
    """
    chosen = QUICK if profile == "quick" else FULL
    if context_limit <= 0:
        return chosen
    # เผื่อที่ให้ output + system prompt ด้วย ไม่ใช่แค่ input พอดี context
    return tuple(w for w in chosen if w.input_tokens + w.output_tokens + 512 <= context_limit)
