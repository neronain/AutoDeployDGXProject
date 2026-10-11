"""รวม token/request usage จากแคชของ refresher — ไม่เก็บอะไรใหม่ แค่รวมยอดจากที่มีอยู่แล้ว"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")

from lmds.web import state, usage  # noqa: E402


def _model(slug, requests=None, prompt=None, generated=None):
    if requests is prompt is generated is None:
        return {"slug": slug, "usage": None}
    return {"slug": slug, "usage": {"requests_24h": requests, "prompt_tokens_24h": prompt,
                                     "generated_tokens_24h": generated, "source": "metrics", "note": ""}}


def test_local_usage_reads_the_cached_models():
    state.STORE.set_local({"host": {}, "models": [_model("a", 10, 1000, 500), _model("b")]})
    result = usage.local_usage()
    assert result["models"] == [{"slug": "a", "requests_24h": 10, "prompt_tokens_24h": 1000,
                                  "generated_tokens_24h": 500, "source": "metrics", "note": ""}]


def test_local_usage_before_any_probe_is_empty():
    result = usage.local_usage()
    assert result["models"] == []
    assert result["stale"] is True


def test_fleet_usage_sums_across_host_and_nodes():
    state.STORE.set_local({"host": {}, "models": [_model("a", 10, 1000, 500)]})
    state.STORE.set_node("spark-head", {"host": {}, "models": [_model("b", 5, 300, 100)]})
    state.STORE.set_node("spark-worker", None, "ต่อไม่ได้")

    result = usage.fleet_usage()
    assert result["fleet_total"] == {"requests_24h": 15, "prompt_tokens_24h": 1300,
                                      "generated_tokens_24h": 600, "models_counted": 2}
    assert result["nodes"]["spark-worker"]["models"] == []
    assert result["nodes"]["spark-worker"]["error"] == "ต่อไม่ได้"


def test_fleet_total_is_none_when_nothing_has_usage_yet():
    state.STORE.set_local({"host": {}, "models": [_model("a")]})
    result = usage.fleet_usage()
    assert result["fleet_total"] == {"requests_24h": None, "prompt_tokens_24h": None,
                                      "generated_tokens_24h": None, "models_counted": 0}


def test_fleet_total_sums_partial_fields_independently():
    """โมเดลหนึ่งมีแต่ requests (llama.cpp ไม่มี /metrics) อีกตัวมีแต่ token — รวมแยกฟิลด์ ไม่ปนกัน"""
    state.STORE.set_local({"host": {}, "models": [
        _model("server-log-only", requests=7, prompt=None, generated=None),
        _model("metrics-full", requests=3, prompt=200, generated=80),
    ]})
    result = usage.fleet_usage()
    assert result["fleet_total"] == {"requests_24h": 10, "prompt_tokens_24h": 200,
                                      "generated_tokens_24h": 80, "models_counted": 2}
