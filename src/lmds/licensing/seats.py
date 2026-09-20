"""นับ "เครื่องที่เสิร์ฟได้ซึ่งบริหารร่วมกัน" — ตัวเลขเดียวที่ไลเซนส์สนใจ

นับตาม LICENSE §1.1 ไม่ใช่ตามจำนวนแถวในทะเบียน:

- เครื่องที่ `llama-server` ใช้ได้ **หรือ** มี docker + NVIDIA GPU อย่างน้อยหนึ่งใบ → นับ 1
- control plane (ไม่มีทั้งสองอย่าง) → **นับ 0** · hub ที่ไม่มี GPU จึงฟรีเสมอ
- container หลายตัวบนเครื่องเดียว → ยังเป็น 1 (นับเครื่อง ไม่ใช่โมเดล)

เราไม่ตัดสินจากชื่อหรือ config — ใช้ผลตรวจจริงที่ `lmds agent info` ส่งกลับมา ซึ่งเป็น
ตัวเดียวกับที่ `lmds doctor` และคำสั่งอื่นใช้ (lmds.hardware.serving.ServingCapability)

เครื่องที่ยังไม่เคย probe = **ไม่นับ** โดยตั้งใจ
----------------------------------------------
ถ้าเดาว่าเครื่องที่ยังไม่รู้จักเป็นเครื่องเสิร์ฟ เราจะบล็อกคนที่ไม่ได้ทำอะไรผิดเลย
(เช่นเพิ่งเพิ่มเครื่องเข้าทะเบียนแต่ยังไม่ได้ติดตั้ง) — ผิดพลาดไปทางไม่บล็อกดีกว่าเสมอ
แต่ต้อง **บอกให้เห็น** ว่านับไม่ครบ ไม่ใช่เงียบ: `unknown` ในผลลัพธ์มีไว้เพื่อการนั้น
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Count:
    serving: int                                  # เครื่องที่ยืนยันแล้วว่าเสิร์ฟได้
    control_plane: int = 0                        # ไม่นับตามสัญญา — เก็บไว้อธิบาย
    unknown: int = 0                              # ยังไม่เคย probe → ไม่นับ
    serving_names: list[str] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return self.serving + self.control_plane + self.unknown

    def explain(self) -> str:
        parts = [f"เครื่องที่เสิร์ฟได้ {self.serving}"]
        if self.control_plane:
            parts.append(f"control plane {self.control_plane} (ไม่นับ)")
        if self.unknown:
            parts.append(f"ยังไม่ได้ตรวจ {self.unknown} (ไม่นับ): {', '.join(self.unknown_names[:4])}")
        return " · ".join(parts)


def count_fleet(nodes, *, hub_serves: bool | None = None, hub_name: str = "เครื่องนี้ (hub)") -> Count:
    """นับจากทะเบียน node + ตัว hub เอง

    `hub_serves=None` = ยังไม่ได้ตรวจ hub · ผู้เรียกที่รู้ควรส่งมาเสมอ เพราะ hub ที่มี GPU
    **นับด้วย** — เป็นเครื่องเสิร์ฟเครื่องหนึ่งไม่ต่างจาก node
    """
    serving_names: list[str] = []
    unknown_names: list[str] = []
    control = 0

    for node in nodes or []:
        value = getattr(node, "serving", None)
        name = getattr(node, "name", "?")
        if value is True:
            serving_names.append(name)
        elif value is False:
            control += 1
        else:
            unknown_names.append(name)

    if hub_serves is True:
        serving_names.insert(0, hub_name)
    elif hub_serves is False:
        control += 1
    else:
        unknown_names.insert(0, hub_name)

    return Count(len(serving_names), control, len(unknown_names), serving_names, unknown_names)


def count_here() -> Count:
    """นับจากสิ่งที่เครื่องนี้รู้ — ใช้ใน CLI และหน้าเว็บ"""
    from lmds.hardware.serving import detect
    from lmds.nodes import registry

    try:
        nodes = registry.load()
    except Exception:
        nodes = []
    try:
        hub_serves = detect().can_serve
    except Exception:
        hub_serves = None
    return count_fleet(nodes, hub_serves=hub_serves)
