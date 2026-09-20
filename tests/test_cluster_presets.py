"""preset 3/8 เครื่อง + "ต่อกันอย่างไร" — เพดานต่อสายตรง ≤3 และคำเตือนว่าต้องซื้ออะไร

เคสที่ต้องกัน: ลูกค้ามี DGX Spark 4 เครื่องแต่ไม่มี switch · Spark มี QSFP 2 ช่อง/เครื่อง
วงแหวน 3 เครื่องใช้ครบทั้งสองช่องพอดี เครื่องที่ 4 จึงไม่มีช่องเหลือให้เสียบ · เดิมไม่มีอะไร
บอกเขาเลยจนกว่าจะไปล้มที่ `lmds cluster apply` ด้วย "unknown topology" ที่ไม่บอกว่าต้องซื้ออะไร

ความซื่อสัตย์ของข้อมูล: preset ใหม่ทุกตัวต้อง `tested=False` — ในโค้ดนี้ tested=True แปลว่า
"ทีมเรารันบนเครื่องจริงแล้ว" ซึ่งปิดโหมด conservative ของ budget · คลัสเตอร์ 8 เครื่องที่อ้างถึง
เป็น **คำกล่าวอ้างของผู้เขียนรีโปภายนอก** (ไม่มี log ดิบ บางตัวไม่มีโฟลเดอร์ results/ เลย)
ดู docs/NVIDIA-CLUSTER-SOURCES.md §1
"""

from __future__ import annotations

import pytest

from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.fit.targets import MAX_DIRECT_NODES, MAX_SWITCH_NODES, interconnect_for
from lmds.hardware import MemoryModel
from lmds.hardware.profiler import group_qsfp_ports
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from lmds.nodes import cluster as cl
from lmds.nodes.netplan import infer_topology


# ── preset ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, nodes, total_gb",
    [
        ("dgx-spark-single", 1, 128.0),
        ("dgx-spark-stacked", 2, 256.0),
        ("dgx-spark-stacked-3", 3, 384.0),
        ("dgx-spark-stacked-4", 4, 512.0),
        ("dgx-spark-stacked-8", 8, 1024.0),
    ],
)
def test_spark_presets_scale_by_128gb_per_machine(name, nodes, total_gb):
    """ทุก preset ต่อยอดจากเครื่องละ 128 GB เท่ากัน — 3/8 ไม่ได้ใช้สเกลของตัวเอง"""
    spec = PRESETS[name]
    assert spec.memory_model is MemoryModel.UNIFIED
    assert spec.node_count == nodes and spec.gpu_count == nodes
    assert spec.total_gpu_memory_gb == total_gb


@pytest.mark.parametrize("name", ["dgx-spark-stacked-3", "dgx-spark-stacked-4", "dgx-spark-stacked-8"])
def test_presets_we_never_ran_stay_untested(name):
    """tested=True = "ทีมเรารันบนเครื่องจริงแล้ว" → ปิด conservative budget

    หลักฐานของ 3/8 เครื่องเป็นคำกล่าวอ้างในรีโปคนอื่น ไม่ใช่ artifact ที่เราตรวจได้ ·
    ตั้งเป็น True เพราะ README ของคนอื่น = ลูกค้า deploy แล้วพังตอนรัน · เทสนี้เป็นตัวกัน
    """
    assert PRESETS[name].tested is False


def test_only_one_and_two_machines_are_hardware_tested():
    """กันคนเผลอเลื่อน preset ที่ยังไม่ได้รันจริงขึ้นไปอยู่ในกลุ่ม tested"""
    tested = {n for n, s in PRESETS.items() if n.startswith("dgx-spark") and s.tested}
    assert tested == {"dgx-spark-single", "dgx-spark-stacked"}


# ── ต่อกันอย่างไร ─────────────────────────────────────────────────────────────
def test_up_to_three_machines_need_no_switch():
    assert MAX_DIRECT_NODES == 3
    for nodes in (1, 2, 3):
        link = interconnect_for(nodes)
        assert link.needs_switch is False and link.supported

    assert interconnect_for(2).cabling == "direct-2"
    # 3 เครื่อง = วงแหวน ใช้ QSFP ครบทั้งสองช่อง → ขยายต่อด้วยสายอย่างเดียวไม่ได้อีก
    assert interconnect_for(3).cabling == "ring-3"


def test_four_machines_need_a_switch_and_the_note_says_what_to_buy():
    link = interconnect_for(4)
    assert link.cabling == "switch" and link.needs_switch is True
    assert "switch" in link.note
    # "ต้องมี switch" เฉย ๆ ไม่ช่วยคนที่ยืนอยู่หน้าเครื่องสี่ตัว — ต้องมีชื่อรุ่นให้ไปหาซื้อ
    assert any("CRS804-4DDQ-HRM" in item for item in link.shopping)
    assert str(MAX_DIRECT_NODES) in link.note


def test_eight_is_the_largest_size_anyone_claims_to_have_run():
    assert MAX_SWITCH_NODES == 8
    assert interconnect_for(8).cabling == "switch" and interconnect_for(8).supported
    # เกินกว่านั้นไม่ใช่ "ยังไม่รองรับเพราะโค้ด" แต่เป็น "ไม่มีใครเคยเห็นของจริง"
    assert interconnect_for(9).cabling == "unsupported" and not interconnect_for(9).supported


def test_target_spec_answers_how_it_is_cabled():
    assert PRESETS["dgx-spark-stacked-3"].interconnect.needs_switch is False
    assert PRESETS["dgx-spark-stacked-4"].interconnect.needs_switch is True
    assert PRESETS["dgx-spark-stacked-8"].interconnect.needs_switch is True
    # dual-GPU อยู่ในเครื่องเดียว — ไม่ใช่คลัสเตอร์ ไม่ต้องต่อสายอะไร
    assert PRESETS["rtx-pro-4000-dual"].interconnect.cabling == "single"


# ── ผู้ใช้เห็นตอน analyse (จุดแรกสุดที่รู้จำนวนเครื่อง) ──────────────────────────
def _model() -> ModelReport:
    return ModelReport(
        repo_id="Qwen/Qwen3-32B", revision_sha="abc", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128),
    )


def test_fit_on_four_machines_warns_about_the_switch_before_anyone_cables_anything():
    """`fit.notes` ถูกพิมพ์ตรง ๆ ทั้ง CLI และหน้าเว็บ — ไม่ต้องรอ `cluster apply` ถึงจะรู้"""
    notes = " ".join(analyze(_model(), PRESETS["dgx-spark-stacked-4"]).notes)
    assert "switch" in notes and "CRS804-4DDQ-HRM" in notes


def test_fit_on_three_machines_does_not_ask_for_a_switch():
    notes = " ".join(analyze(_model(), PRESETS["dgx-spark-stacked-3"]).notes)
    assert "CRS804-4DDQ-HRM" not in notes
    assert "วงแหวน" in notes


# ── กลุ่มคลัสเตอร์: เห็นสายจริงแล้วยังเตือนได้ ─────────────────────────────────
def _spark(ip: str, ports: tuple[int, ...]) -> dict:
    """Spark หนึ่งเครื่องที่มีสายเสียบอยู่ที่ช่อง `ports` — cluster IP อยู่บนช่องแรกเสมอ"""
    links = []
    for port in (1, 2):
        iface = "enp1s0f1np1" if port == 1 else "enP2p1s0f1np1"
        cabled = port in ports
        links.append({
            "iface": iface, "ip": ip if (cabled and port == ports[0]) else ("169.254.9.9" if cabled else ""),
            "prefix": 24, "speed_gbps": 200 if cabled else None, "driver": "mlx5_core",
            "state": "up" if cabled else "down", "connectx": True, "rdma": True,
            "carrier": cabled, "qsfp_port": port, "function": 1, "rdma_device": f"roce{port}",
        })
    return {
        "hostname": f"spark-{ip.rsplit('.', 1)[-1]}", "arch": "aarch64", "profile": "dgx_spark",
        "gpus": [{"name": "NVIDIA GB10"}],
        "fabric": {"links": links, "best_gbps": 200, "tier": "rdma", "cluster_capable": True,
                   "qsfp_ports": group_qsfp_ports(links)},
    }


def _machines(count: int, ports: tuple[int, ...]) -> list[dict]:
    return [
        {"name": f"n{i}", "host": _spark(f"10.100.152.{i}", ports),
         "cluster_ip": f"10.100.152.{i}", "cluster_name": "lab"}
        for i in range(1, count + 1)
    ]


def test_cabling_is_read_from_the_ports_that_actually_have_a_cable():
    assert cl.cabled_qsfp_ports(_spark("10.100.152.1", (1,))) == [1]
    assert cl.cabled_qsfp_ports(_spark("10.100.152.1", (1, 2))) == [1, 2]
    assert cl.observed_cabling(_machines(4, (1,))) == "switch"
    assert cl.observed_cabling(_machines(3, (1, 2))) == "direct"


def test_a_group_of_four_wired_like_a_ring_is_told_to_buy_a_switch():
    """ตั้งชื่อคลัสเตอร์เองทำให้ 4 เครื่องมารวมกลุ่มได้แม้เดินสายแบบวงแหวน — ต้องเตือนตรงนี้"""
    group = cl.cluster_groups(_machines(4, (1, 2)))[0]
    warning = next(w for w in group["warnings"] if w["kind"] == "needs-switch")
    assert warning["node_count"] == 4 and warning["max_direct"] == MAX_DIRECT_NODES
    assert any("CRS804-4DDQ-HRM" in item for item in warning["shopping"])
    assert group["interconnect"]["needs_switch"] is True
    assert group["interconnect"]["cabling"] == "direct"


def test_a_group_of_four_on_a_switch_is_not_nagged():
    group = cl.cluster_groups(_machines(4, (1,)))[0]
    assert [w for w in group["warnings"] if w["kind"] == "needs-switch"] == []
    assert group["interconnect"]["cabling"] == "switch"


def test_a_pair_never_needs_a_switch():
    group = cl.cluster_groups(_machines(2, (1, 2)))[0]
    assert [w for w in group["warnings"] if w["kind"] == "needs-switch"] == []
    assert group["interconnect"]["needs_switch"] is False


# ── ผัง: เดาจากสายจริง ────────────────────────────────────────────────────────
def test_more_than_four_machines_on_a_switch_can_now_be_planned():
    """เพดาน "ผ่าน switch สูงสุด 4" ถูกถอนไปแล้ว — ปิดไว้ที่ 4 ทำให้ 8 เครื่องวางแผนไม่ได้เลย"""
    for count in (5, 6, 7, 8):
        names = [f"n{i}" for i in range(count)]
        got = infer_topology({name: [1] for name in names}, names)
        assert got["topology"] == f"switch-{count}"
        assert len(got["links"]) == 1 and len(got["links"][0]["ends"]) == count


def test_nine_machines_are_refused_because_nobody_has_shown_one():
    names = [f"n{i}" for i in range(9)]
    got = infer_topology({name: [1] for name in names}, names)
    assert got["topology"] == "unknown" and f"2–{MAX_SWITCH_NODES} machines" in got["reason"]


def test_forcing_a_ring_on_four_machines_says_what_to_buy():
    got = infer_topology({n: [1, 2] for n in "abcd"}, list("abcd"), forced="ring")
    assert got["topology"] == "unknown"
    assert "CRS804-4DDQ-HRM" in got["reason"] and "2 QSFP cages" in got["reason"]


def test_ring_cabling_on_four_machines_is_refused_with_the_shopping_list():
    """คนที่มี 4 เครื่องแต่ไม่มี switch: เดินวงแหวน 3 เครื่องแล้วเหลือเครื่องที่ 4 ต่อไม่ได้"""
    cabled = {"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1]}
    got = infer_topology(cabled, list("abcd"))
    assert got["topology"] == "unknown"
    assert "must go through a switch" in got["reason"]
    assert "CRS804-4DDQ-HRM" in got["reason"]
