"""รายชื่อเครื่องต้องบอก commit ไม่ใช่แค่เลข version — เลข version บอกไม่ได้ว่าใครตามหลัง"""

from lmds.cli.main import _version_label
from lmds.nodes.registry import status_from_probe


def test_label_shows_commit_when_the_node_reports_it():
    """เคสจริง 2026-09-03: ทุก node โชว์ 0.5.0 เท่ากันหมด ทั้งที่ 13 เครื่องยังอยู่คอมมิตเก่า"""
    assert _version_label("0.5.0", "f9181ab") == "0.5.0 (f9181ab)"


def test_label_falls_back_gracefully_for_old_nodes():
    """node รุ่นเก่าไม่ส่ง commit มา — ห้ามโชว์วงเล็บว่าง"""
    assert _version_label("0.4.1", "") == "0.4.1"
    assert _version_label("", "") == ""


def test_status_from_probe_keeps_the_commit_the_agent_sends():
    """host_payload ส่ง lmds_commit มานานแล้ว แต่ทะเบียนทิ้งไปตรงนี้ — จุดเดียวที่ทุกคนใช้"""
    fields = status_from_probe({"host": {"lmds_version": "0.5.0", "lmds_commit": "f9181ab", "ip": "10.0.0.5"}})
    assert fields["lmds_commit"] == "f9181ab"
    # ไม่ส่งมา = ไม่รู้ ไม่ใช่ว่าง — ห้ามเขียนทับของที่เคยรู้
    assert "lmds_commit" not in status_from_probe({"host": {"lmds_version": "0.4.1"}})
