"""รวม health findings จากแคชของ refresher — ไม่ยิง SSH เอง ไม่รายงานเครื่องที่ไม่มีข้อมูลว่า "ปกติ" """

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from lmds.web import health, state  # noqa: E402

_HOT_GPU_HOST = {"gpus": [{"name": "RTX 4090", "temperature_c": 95}],
                  "ram_used_gb": 1.0, "ram_total_gb": 64.0,
                  "disk_free_gb": 200.0, "disk_total_gb": 500.0}

_COOL_HOST = {"gpus": [{"name": "RTX 4090", "temperature_c": 40}],
              "ram_used_gb": 1.0, "ram_total_gb": 64.0,
              "disk_free_gb": 200.0, "disk_total_gb": 500.0}


def test_local_findings_reads_the_cached_host():
    state.STORE.set_local({"host": _HOT_GPU_HOST, "models": []})
    result = health.local_findings()
    assert result["stale"] is False
    assert [f["id"] for f in result["findings"]] == ["gpu-0-temperature"]


def test_local_findings_before_any_probe_is_empty_not_fabricated():
    result = health.local_findings()
    assert result["findings"] == []
    assert result["stale"] is True


def test_fleet_findings_covers_host_and_every_node():
    state.STORE.set_local({"host": _COOL_HOST, "models": []})
    state.STORE.set_node("spark-head", {"host": _HOT_GPU_HOST, "models": []})
    state.STORE.set_node("spark-worker", None, "ssh: connect timed out")

    result = health.fleet_findings()
    assert result["host"]["findings"] == []
    assert [f["id"] for f in result["nodes"]["spark-head"]["findings"]] == ["gpu-0-temperature"]
    # เครื่องที่ต่อไม่ได้: ไม่มี findings เพราะไม่มีข้อมูลให้ประเมิน ไม่ใช่เพราะมันปกติ
    assert result["nodes"]["spark-worker"]["findings"] == []
    assert result["nodes"]["spark-worker"]["error"] == "ssh: connect timed out"
