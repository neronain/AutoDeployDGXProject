"""health findings — เกณฑ์ pure เหนือ host payload รูปเดียวกับ inventory.host_payload()"""

from __future__ import annotations

from lmds.hardware.health import evaluate_host


def _host(**overrides) -> dict:
    base = {
        "gpus": [],
        "ram_used_gb": 10.0,
        "ram_total_gb": 64.0,
        "disk_free_gb": 200.0,
        "disk_total_gb": 500.0,
    }
    base.update(overrides)
    return base


def test_no_findings_when_everything_is_comfortable():
    assert evaluate_host(_host()) == []


def test_gpu_temperature_warn_then_critical():
    warn = evaluate_host(_host(gpus=[{"name": "RTX 4090", "temperature_c": 86}]))
    assert [f.id for f in warn] == ["gpu-0-temperature"]
    assert warn[0].severity == "warn"

    critical = evaluate_host(_host(gpus=[{"name": "RTX 4090", "temperature_c": 95}]))
    assert critical[0].severity == "critical"


def test_gpu_without_temperature_reading_is_skipped_not_assumed_zero():
    """GB10 ไม่รายงานหลายฟิลด์ — None ต้องไม่ถูกตีความว่าปกติหรือผิดปกติ แค่ข้าม"""
    assert evaluate_host(_host(gpus=[{"name": "GB10", "temperature_c": None}])) == []


def test_multiple_gpus_report_their_own_index():
    findings = evaluate_host(_host(gpus=[
        {"name": "A", "temperature_c": 20},
        {"name": "B", "temperature_c": 92},
    ]))
    assert [f.id for f in findings] == ["gpu-1-temperature"]


def test_ram_free_warn_and_critical():
    warn = evaluate_host(_host(ram_total_gb=64.0, ram_used_gb=62.5))
    assert [f.id for f in warn] == ["ram-free"]
    assert warn[0].severity == "warn"

    critical = evaluate_host(_host(ram_total_gb=64.0, ram_used_gb=63.5))
    assert critical[0].severity == "critical"


def test_ram_missing_fields_skipped():
    assert evaluate_host(_host(ram_total_gb=None)) == []


def test_disk_free_warn_and_critical():
    warn = evaluate_host(_host(disk_free_gb=15.0))
    assert [f.id for f in warn] == ["disk-free"]

    critical = evaluate_host(_host(disk_free_gb=4.0))
    assert critical[0].severity == "critical"


def test_findings_serialize_to_plain_dicts():
    findings = evaluate_host(_host(disk_free_gb=1.0))
    assert findings[0].payload() == {
        "id": "disk-free", "severity": "critical",
        "title": "พื้นที่ดิสก์เหลือน้อย",
        "detail": findings[0].detail,
    }
    assert "1.0GB" in findings[0].detail
