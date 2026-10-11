"""Prometheus exporter — pure function เหนือ snapshot รูปเดียวกับ state.STORE.snapshot()"""

from __future__ import annotations

from lmds.web.prometheus import render


def _snapshot(host_entry=None, nodes=None):
    return {
        "version": 1,
        "host": host_entry or {"data": None, "error": "", "age_seconds": 0.0, "stale": True},
        "nodes": nodes or {},
    }


def test_empty_snapshot_still_reports_up_zero():
    text = render(_snapshot())
    assert "lmds_up{node=\"local\"} 0" in text
    assert "lmds_gpu_temperature_celsius" not in text  # ไม่มีข้อมูล = ไม่มี series เลย ไม่ใช่ 0


def test_local_host_reports_gpu_ram_disk_in_bytes():
    entry = {
        "data": {
            "host": {
                "hostname": "Autodeploy",
                "gpus": [{"name": "RTX 4090", "temperature_c": 72, "power_w": 310.5,
                           "power_limit_w": 450.0, "utilization_pct": 80,
                           "vram_used_gb": 20.0, "vram_gb": 24.0}],
                "ram_used_gb": 32.0, "ram_total_gb": 128.0,
                "disk_free_gb": 100.0, "disk_total_gb": 500.0,
            },
            "models": [{"slug": "demo", "running": True, "healthy": False}],
        },
        "error": "", "age_seconds": 1.2, "stale": False,
    }
    nodes = {"spark-worker": {"data": {"host": {"gpus": [{"name": "GB10", "temperature_c": 50}]},
                                        "models": []},
                               "error": "", "age_seconds": 0.0, "stale": False}}
    text = render(_snapshot(host_entry=entry, nodes=nodes))

    assert 'lmds_up{node="Autodeploy"} 1' in text
    assert 'lmds_gpu_temperature_celsius{node="Autodeploy",gpu="0",name="RTX 4090"} 72' in text
    assert 'lmds_gpu_utilization_ratio{node="Autodeploy",gpu="0",name="RTX 4090"} 0.8' in text
    assert f'lmds_gpu_vram_used_bytes{{node="Autodeploy",gpu="0",name="RTX 4090"}} {20.0 * 1024**3!r}' in text
    assert f'lmds_host_ram_total_bytes{{node="Autodeploy"}} {128.0 * 1024**3!r}' in text
    assert 'lmds_model_running{node="Autodeploy",slug="demo"} 1' in text
    assert 'lmds_model_healthy{node="Autodeploy",slug="demo"} 0' in text
    assert 'lmds_gpu_temperature_celsius{node="spark-worker",gpu="0",name="GB10"} 50' in text
    # มี HELP/TYPE ครั้งเดียวต่อ family แม้มีหลาย node ใช้ metric เดียวกัน
    assert text.count("# TYPE lmds_gpu_temperature_celsius") == 1


def test_gpu_fields_the_card_does_not_report_are_omitted_not_zero():
    """GB10: power.limit/fan/clocks.mem มักเป็น None — ต้องไม่โผล่เป็น 0 ในกราฟ"""
    entry = {
        "data": {"host": {"gpus": [{"name": "GB10", "temperature_c": 55, "power_w": None,
                                      "power_limit_w": None, "utilization_pct": None,
                                      "vram_used_gb": None, "vram_gb": None}]},
                  "models": []},
        "error": "", "age_seconds": 0.0, "stale": False,
    }
    text = render(_snapshot(host_entry=entry))
    assert "lmds_gpu_temperature_celsius" in text
    assert "lmds_gpu_power_watts" not in text
    assert "lmds_gpu_utilization_ratio" not in text


def test_node_with_error_reports_down_without_stale_gpu_series():
    nodes = {"spark-worker": {"data": None, "error": "ssh: connect timed out", "age_seconds": 400.0, "stale": True}}
    text = render(_snapshot(nodes=nodes))
    assert 'lmds_up{node="spark-worker"} 0' in text
    assert 'gpu="0"' not in text


def test_active_health_findings_are_exported():
    entry = {
        "data": {"host": {"disk_free_gb": 2.0}, "models": []},
        "error": "", "age_seconds": 0.0, "stale": False,
    }
    text = render(_snapshot(host_entry=entry))
    assert 'lmds_health_finding{node="local",id="disk-free",severity="critical"} 1' in text


def test_label_values_are_escaped():
    entry = {
        "data": {"host": {"gpus": [{"name": 'Weird "GPU"\\x', "temperature_c": 90}]}, "models": []},
        "error": "", "age_seconds": 0.0, "stale": False,
    }
    text = render(_snapshot(host_entry=entry))
    assert 'name="Weird \\"GPU\\"\\\\x"' in text
