"""Health findings: เกณฑ์ตายตัวง่าย ๆ เหนือ telemetry ที่ profiler เก็บอยู่แล้ว (อุณหภูมิ/RAM/ดิสก์)

แนวคิดยืมมาจาก sparkDash (MiaAI-Lab, Apache-2.0) แต่ค่า threshold ในไฟล์นี้เป็นค่าเริ่มต้นแบบอนุรักษ์นิยม
ที่ยังไม่ผ่านการยืนยันกับพฤติกรรมจริงของ GB10/RTX แต่ละรุ่น — ต่างจาก doctor/checks.py ที่ทุกข้อมาจาก
failure ที่เจอจริงบนเครื่อง อย่าอ้างว่าเลขพวกนี้ "ถูกต้อง" จนกว่าจะมีเคสจริงมายืนยัน

หลักเดียวกับ profiler: None = ตรวจไม่ได้/การ์ดไม่รายงาน ไม่ใช่ศูนย์ — ข้ามเฉย ๆ ไม่ฟันธงว่าปกติ
"""

from __future__ import annotations

from dataclasses import dataclass

# GPU: NVIDIA แนะนำเริ่มเป็นห่วงที่ ~83°C สำหรับการ์ด consumer/datacenter ส่วนใหญ่ — ให้ส่วนต่าง
# ก่อนจุด throttle จริง (มักอยู่แถว 90-95°C ขึ้นกับรุ่น) เป็นค่าเริ่มต้นเท่านั้น ปรับได้ตามเครื่องจริง
GPU_TEMP_WARN_C = 85.0
GPU_TEMP_CRITICAL_C = 90.0

# RAM ว่าง (ram_total_gb - ram_used_gb) — ต่ำกว่านี้เสี่ยง OOM-kill ตอนมีงานใหม่เข้ามา
RAM_FREE_WARN_GB = 2.0
RAM_FREE_CRITICAL_GB = 1.0

# ดิสก์ว่าง — โมเดลขนาดกลางหนักหลักสิบ GB ต่อตัว เหลือน้อยกว่านี้ดาวน์โหลดตัวถัดไปไม่ผ่านแน่
DISK_FREE_WARN_GB = 20.0
DISK_FREE_CRITICAL_GB = 5.0


@dataclass
class HealthFinding:
    id: str
    severity: str  # "warn" | "critical"
    title: str
    detail: str

    def payload(self) -> dict:
        return {"id": self.id, "severity": self.severity, "title": self.title, "detail": self.detail}


def _level(value: float, warn: float, critical: float) -> str | None:
    """ค่ายิ่งสูงยิ่งแย่ (อุณหภูมิ) — คืน None เมื่อยังปกติ"""
    if value >= critical:
        return "critical"
    if value >= warn:
        return "warn"
    return None


def _level_low(value: float, warn: float, critical: float) -> str | None:
    """ค่ายิ่งต่ำยิ่งแย่ (พื้นที่ว่าง) — คืน None เมื่อยังปกติ"""
    if value <= critical:
        return "critical"
    if value <= warn:
        return "warn"
    return None


def evaluate_host(host: dict) -> list[HealthFinding]:
    """findings ของเครื่องเดียว จาก payload รูปเดียวกับ `lmds.inventory.host_payload()`

    ฟังก์ชัน pure ล้วน — ไม่แตะ SSH/subprocess เอง อ่านจากค่าที่ profiler/refresher เก็บไว้แล้วเท่านั้น
    จึงเรียกได้ถี่ (เช่นทุกครั้งที่ Prometheus scrape) โดยไม่เพิ่มภาระให้เครื่องจริง
    """
    findings: list[HealthFinding] = []

    for idx, gpu in enumerate(host.get("gpus") or []):
        temp = gpu.get("temperature_c")
        if temp is not None:
            level = _level(temp, GPU_TEMP_WARN_C, GPU_TEMP_CRITICAL_C)
            if level:
                name = gpu.get("name") or f"GPU {idx}"
                findings.append(HealthFinding(
                    id=f"gpu-{idx}-temperature", severity=level,
                    title=f"{name} (GPU {idx}) ร้อนเกิน",
                    detail=f"{temp:g}°C (เตือนที่ {GPU_TEMP_WARN_C:g}°C · วิกฤตที่ {GPU_TEMP_CRITICAL_C:g}°C)",
                ))

    ram_total = host.get("ram_total_gb")
    ram_used = host.get("ram_used_gb")
    if ram_total is not None and ram_used is not None:
        ram_free = ram_total - ram_used
        level = _level_low(ram_free, RAM_FREE_WARN_GB, RAM_FREE_CRITICAL_GB)
        if level:
            findings.append(HealthFinding(
                id="ram-free", severity=level, title="RAM เหลือน้อย",
                detail=f"เหลือ {ram_free:.1f}GB จาก {ram_total:.1f}GB "
                       f"(เตือนที่ {RAM_FREE_WARN_GB:g}GB · วิกฤตที่ {RAM_FREE_CRITICAL_GB:g}GB)",
            ))

    disk_free = host.get("disk_free_gb")
    if disk_free is not None:
        level = _level_low(disk_free, DISK_FREE_WARN_GB, DISK_FREE_CRITICAL_GB)
        if level:
            findings.append(HealthFinding(
                id="disk-free", severity=level, title="พื้นที่ดิสก์เหลือน้อย",
                detail=f"เหลือ {disk_free:.1f}GB "
                       f"(เตือนที่ {DISK_FREE_WARN_GB:g}GB · วิกฤตที่ {DISK_FREE_CRITICAL_GB:g}GB)",
            ))

    return findings
