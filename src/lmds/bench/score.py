"""สรุปผลดิบให้เป็นคะแนนที่เทียบกันได้ — โดยไม่แต่งตัวเลขที่ไม่ได้วัด

เจตนาสำคัญ: **ไม่มี "คะแนนความฉลาด"** ที่นี่ · เครื่องมืออย่าง Local-Bench ใช้ค่า IQ จาก
Artificial Analysis ซึ่งเป็นดัชนีภายนอกที่วัดจากโมเดลต้นทาง ไม่ใช่จาก quant ที่คุณรันจริง
บนเครื่องคุณ — เอามาแปะแล้วมันจะดูเหมือนเราวัดเอง ทั้งที่ไม่ได้วัด

สิ่งที่ให้คะแนนได้จริงมีสองแกน และทั้งคู่มาจากการยิงเครื่องจริง:
  ความเร็ว     decode tok/s ที่ context ต่าง ๆ + TTFT
  ความสามารถ   ทำได้/ทำไม่ได้ ทีละข้อ ไม่ใช่ความเห็น
"""

from __future__ import annotations

# น้ำหนักของแต่ละความสามารถ — tool calling หนักสุดเพราะมันคือเส้นแบ่งว่าเอาไปต่อ agent ได้ไหม
# ข้อที่ถูกข้าม (เช่น vision บนโมเดลที่ไม่มี mmproj) ไม่ถูกนับในตัวหาร
_WEIGHTS = {
    "instructions": 2.0,
    "tools": 3.0,
    "json": 2.0,
    "reasoning": 1.0,
    "thai": 2.0,
    "vision": 1.0,
    "recall": 2.0,
}


def capability_score(probes: list[dict]) -> dict:
    """0-100 จากข้อที่ *วัดได้* เท่านั้น พร้อมบอกว่าหารด้วยอะไร"""
    earned = 0.0
    possible = 0.0
    for probe in probes:
        if probe.get("skipped"):
            continue
        weight = _WEIGHTS.get(probe.get("key", ""), 1.0)
        possible += weight
        if probe.get("passed"):
            earned += weight
    if not possible:
        return {"score": None, "earned": 0.0, "possible": 0.0, "counted": 0}
    return {
        "score": round(100 * earned / possible),
        "earned": earned,
        "possible": possible,
        "counted": len([p for p in probes if not p.get("skipped")]),
        # ตัวเลขรวมบอกว่า "ดีแค่ไหน" แต่คนตัดสินใจต้องรู้ว่า *ตกข้อไหน* —
        # 85/100 ที่ตก tool calling ใช้กับ agent ไม่ได้ ส่วน 85 ที่ตก vision อาจไม่สำคัญเลย
        "passed": [p.get("key") for p in probes if p.get("passed") and not p.get("skipped")],
        "failed": [p.get("key") for p in probes if not p.get("passed") and not p.get("skipped")],
        "skipped": [p.get("key") for p in probes if p.get("skipped")],
    }


def speed_summary(workloads: list[dict]) -> dict:
    """ตัวเลขความเร็วที่ควรอ่านก่อน — ค่ากลาง กับค่าที่ context ยาวสุดที่วัดได้

    แยกสองตัวเพราะโมเดลที่เร็วตอน prompt สั้นแต่ตกฮวบตอน 8K เป็นเรื่องปกติมาก
    ค่าเฉลี่ยรวมจะกลบพฤติกรรมนั้นจนมองไม่เห็น
    """
    usable = [w for w in workloads if not w.get("error") and w.get("decode_tps")]
    if not usable:
        return {"decode_tps_avg": None, "decode_tps_long": None,
                "ttft_s_short": None, "longest_context": 0, "failed": len(workloads)}
    longest = max(usable, key=lambda w: w.get("target_input", 0))
    shortest = min(usable, key=lambda w: w.get("target_input", 0))
    return {
        "decode_tps_avg": round(sum(w["decode_tps"] for w in usable) / len(usable), 1),
        "decode_tps_long": round(longest["decode_tps"], 1),
        "ttft_s_short": round(shortest["ttft_s"], 2),
        "longest_context": longest.get("target_input", 0),
        "failed": len([w for w in workloads if w.get("error")]),
    }


def summarize(run: dict) -> dict:
    """หนึ่งบรรทัดของตารางคะแนน"""
    speed = speed_summary(run.get("workloads") or [])
    capability = capability_score(run.get("probes") or [])
    return {
        "slug": run.get("slug"),
        "model_id": run.get("model_id"),
        "engine": run.get("engine"),
        "hostname": (run.get("machine") or {}).get("hostname"),
        "stamped_at": run.get("stamped_at"),
        "speed": speed,
        "capability": capability,
        "environment": run.get("environment") or {},
        # ตัวเลขที่ยืมมาจากรอบก่อน ต้องติดป้ายว่ามาจากเมื่อไร ไม่งั้นดูเหมือนวัดพร้อมกันหมด
        "speed_from": run.get("speed_from", ""),
        "probes_from": run.get("probes_from", ""),
    }
