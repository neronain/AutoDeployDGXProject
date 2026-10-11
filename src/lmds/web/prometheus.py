"""Prometheus text exposition (format 0.0.4) สำหรับ `GET /metrics`

Pure: รับ snapshot รูปเดียวกับ `state.STORE.snapshot()` แล้วคืนสตริง — ทดสอบได้โดยไม่ต้องมี
เครื่องจริง ไม่ต้องมี server ไม่ต้องรอ refresher · endpoint อ่านจากแคชเดิมที่หน้าเว็บใช้อยู่แล้ว
จึงไม่เพิ่มรอบ SSH ใหม่แม้ scrape ถี่แค่ไหน (ดู web/state.py)

ชื่อ metric ตาม convention ของ Prometheus/node_exporter: หน่วยฐาน (bytes, celsius, watts,
สัดส่วน 0-1 แทน %), `_total` เฉพาะ counter, หนึ่ง HELP/TYPE ต่อ family

**ไม่รู้ ไม่ใช่ศูนย์** — เหมือน profiler: ค่า None (การ์ด/เครื่องไม่รายงาน) ไม่ถูกเขียนเป็น series
เลย ไม่ใช่เขียนเป็น 0 · เครื่องที่ต่อไม่ได้ส่งออกแค่ `lmds_up 0` ไม่มี series อื่นของเครื่องนั้น
"""

from __future__ import annotations

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_GB = 1024 ** 3

METRIC_FAMILIES: list[tuple[str, str, str]] = [
    ("lmds_up", "gauge", "1 ถ้า refresher คุยกับเครื่องนี้ได้ล่าสุด มิฉะนั้น 0"),
    ("lmds_scrape_age_seconds", "gauge", "ข้อมูลของเครื่องนี้เก่าแค่ไหน (วินาทีนับจากที่ refresher สำรวจสำเร็จล่าสุด)"),
    ("lmds_gpu_temperature_celsius", "gauge", "อุณหภูมิ GPU"),
    ("lmds_gpu_power_watts", "gauge", "กำลังไฟที่ GPU ใช้อยู่"),
    ("lmds_gpu_power_limit_watts", "gauge", "เพดานกำลังไฟของ GPU"),
    ("lmds_gpu_utilization_ratio", "gauge", "สัดส่วนการใช้งาน GPU (0-1)"),
    ("lmds_gpu_vram_used_bytes", "gauge", "VRAM ที่ใช้ไปของ GPU"),
    ("lmds_gpu_vram_total_bytes", "gauge", "VRAM ทั้งหมดของ GPU"),
    ("lmds_host_ram_used_bytes", "gauge", "RAM ที่ใช้ไปของเครื่อง"),
    ("lmds_host_ram_total_bytes", "gauge", "RAM ทั้งหมดของเครื่อง"),
    ("lmds_host_disk_free_bytes", "gauge", "พื้นที่ดิสก์ว่างของเครื่อง"),
    ("lmds_host_disk_total_bytes", "gauge", "พื้นที่ดิสก์ทั้งหมดของเครื่อง"),
    ("lmds_model_running", "gauge", "1 ถ้าโมเดลนี้กำลังรันอยู่"),
    ("lmds_model_healthy", "gauge", "1 ถ้า health check ของโมเดลนี้ผ่าน"),
    ("lmds_health_finding", "gauge", "1 ต่อ health finding ที่กำลัง active (ดู /api/health/fleet)"),
]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: dict[str, str]) -> str:
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs.items())
    return "{" + inner + "}"


class _Writer:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._announced: set[str] = set()
        self._help = {name: (kind, help_text) for name, kind, help_text in METRIC_FAMILIES}

    def emit(self, name: str, value: float | bool | None, labels: dict[str, str]) -> None:
        if value is None:
            return
        if name not in self._announced:
            kind, help_text = self._help[name]
            self._lines.append(f"# HELP {name} {help_text}")
            self._lines.append(f"# TYPE {name} {kind}")
            self._announced.add(name)
        number = 1 if value is True else (0 if value is False else value)
        self._lines.append(f"{name}{_labels(labels)} {number!r}")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n" if self._lines else ""


def _node_label(name: str, data: dict | None) -> str:
    if name:
        return name
    host = (data or {}).get("host") or {}
    return host.get("hostname") or "local"


def _write_entry(w: _Writer, node: str, entry: dict) -> None:
    data = entry.get("data")
    up = data is not None and not entry.get("error")
    w.emit("lmds_up", up, {"node": node})
    w.emit("lmds_scrape_age_seconds", entry.get("age_seconds"), {"node": node})
    if not data:
        return

    host = data.get("host") or {}
    for idx, gpu in enumerate(host.get("gpus") or []):
        labels = {"node": node, "gpu": str(idx), "name": gpu.get("name") or ""}
        w.emit("lmds_gpu_temperature_celsius", gpu.get("temperature_c"), labels)
        w.emit("lmds_gpu_power_watts", gpu.get("power_w"), labels)
        w.emit("lmds_gpu_power_limit_watts", gpu.get("power_limit_w"), labels)
        pct = gpu.get("utilization_pct")
        w.emit("lmds_gpu_utilization_ratio", None if pct is None else pct / 100.0, labels)
        vram_used = gpu.get("vram_used_gb")
        w.emit("lmds_gpu_vram_used_bytes", None if vram_used is None else vram_used * _GB, labels)
        vram_total = gpu.get("vram_gb")
        w.emit("lmds_gpu_vram_total_bytes", None if vram_total is None else vram_total * _GB, labels)

    ram_used = host.get("ram_used_gb")
    w.emit("lmds_host_ram_used_bytes", None if ram_used is None else ram_used * _GB, {"node": node})
    ram_total = host.get("ram_total_gb")
    w.emit("lmds_host_ram_total_bytes", None if ram_total is None else ram_total * _GB, {"node": node})
    disk_free = host.get("disk_free_gb")
    w.emit("lmds_host_disk_free_bytes", None if disk_free is None else disk_free * _GB, {"node": node})
    disk_total = host.get("disk_total_gb")
    w.emit("lmds_host_disk_total_bytes", None if disk_total is None else disk_total * _GB, {"node": node})

    for model in data.get("models") or []:
        labels = {"node": node, "slug": model.get("slug") or ""}
        w.emit("lmds_model_running", bool(model.get("running")), labels)
        w.emit("lmds_model_healthy", bool(model.get("healthy")), labels)

    from lmds.hardware.health import evaluate_host

    for finding in evaluate_host(host):
        w.emit("lmds_health_finding", True, {"node": node, "id": finding.id, "severity": finding.severity})


def render(snapshot: dict) -> str:
    w = _Writer()
    host_entry = snapshot.get("host") or {}
    w_node = _node_label("", host_entry.get("data"))
    _write_entry(w, w_node, host_entry)
    for name, entry in (snapshot.get("nodes") or {}).items():
        _write_entry(w, name, entry)
    return w.render()
