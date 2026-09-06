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


# ── "ตรง hub" ต้องผ่าน 3 มิติ (audit Update path 2026-09-06 §5) ──
HUB = {"version": "0.6.1", "commit": "dcefd91", "template_hash": "hubhash00001", "dirty": []}


def _info(*, commit="dcefd91", ctl_hash="hubhash00001", supported=True, engine="llamacpp", generated_by="lmds 0.6.1"):
    return {
        "host": {"lmds_version": "0.6.1", "lmds_commit": commit, "lmds_installed_commit": commit,
                 "runtimes": {"llamacpp": [{"dir": "/home/x/src/llama.cpp", "present": True, "build": "10826",
                                            "commit": "73a43d1f6", "date": "2026-09-06", "lock_state": "ok"}]}},
        "models": [{"slug": "qwen3-8-flash", "engine": engine, "downloaded": True,
                    "generated_by": generated_by, "template_hash": ctl_hash,
                    "controller": {"state": "ok" if ctl_hash == HUB["template_hash"] else "stale", "generated_by": generated_by.split()[-1]},
                    "runtime_arch": None if supported is None else
                    {"arch": "qwen4exp", "mode": "native", "supported": supported,
                     "runtime": "llama.cpp /home/x/src/llama.cpp", "fix": "LLAMA_CPP_UPDATE=1 ./x prepare-runtime"}}],
    }


def test_update_consistency_needs_three_axes():
    """spark-worker 2026-09-06: "พร้อมแล้ว — ตรง hub" ทั้งที่ controller เป็น 0.5.1 และ build ไม่รู้จัก qwen4exp"""
    from lmds.fleet.consistency import node_verdict, summary_line, verdict_lines

    good = node_verdict(_info(), HUB)
    assert good.consistent and summary_line(good).startswith("ตรง hub ✓")
    lines = verdict_lines(good)
    assert lines[0].startswith("  code ") and "ตรง hub" in lines[0]
    assert lines[1].startswith("  controllers ") and lines[2].startswith("  runtime ") and "build 10826" in lines[2]

    # code ตรงแต่ runtime ค้าง — ห้ามพิมพ์ "ตรง hub"
    stale_rt = node_verdict(_info(supported=False), HUB)
    assert not stale_rt.consistent and stale_rt.runtimes.state == "stale" and stale_rt.runtimes.items == ["qwen3-8-flash"]
    assert summary_line(stale_rt).startswith("ยังไม่ตรง hub") and "runtime ค้าง 1" in summary_line(stale_rt)
    assert "ตรง hub ✓" not in summary_line(stale_rt)

    # controller เก่ากว่า template ของ hub (0.5.1 ไม่มี template_hash)
    stale_ctl = node_verdict(_info(ctl_hash=None, generated_by="lmds 0.5.1"), HUB)
    assert stale_ctl.controllers.state == "stale" and "0.5.1" in stale_ctl.controllers.detail and not stale_ctl.consistent
    # เลข version เท่ากันแต่ template ต่าง = stale เหมือนกัน (ตัดสินด้วย hash ไม่ใช่เลขรุ่น)
    assert node_verdict(_info(ctl_hash="other000000"), HUB).controllers.state == "stale"

    # commit ไม่ตรง
    behind = node_verdict(_info(commit="0000000"), HUB)
    assert behind.code.state == "behind" and "ยังไม่ตรง hub" in behind.code.detail

    # hub มีไฟล์แก้ค้าง = code ไม่ ok แม้ commit ตรง
    dirty = node_verdict(_info(), {**HUB, "dirty": ["src/lmds/x.py"]})
    assert dirty.code.state == "dirty" and not dirty.consistent

    # unknown → "ตรวจไม่ได้ (เหตุผล)" ไม่ใช่เงียบ และไม่นับว่าผ่าน
    unknown = node_verdict(_info(supported=None), HUB)
    assert unknown.runtimes.state == "unknown" and not unknown.consistent and unknown.level == "warn"
    assert "ตรวจไม่ได้" in summary_line(unknown)
    # ไม่มี bundle llama.cpp = n/a นับว่าผ่าน
    assert node_verdict(_info(engine="vllm", supported=None), HUB).runtimes.state == "n/a"


def test_status_from_probe_keeps_stale_counters():
    """ทะเบียนเก็บ controllers_stale / runtime_stale / llamacpp_build — และไม่เขียนทับด้วยค่าว่างจาก node รุ่นเก่า"""
    fields = status_from_probe(_info(ctl_hash=None, generated_by="lmds 0.5.1", supported=False))
    assert fields["controllers_stale"] == 1 and fields["runtime_stale"] == 1
    assert fields["llamacpp_build"] == "10826 · 2026-09-06"
    # 0 คือค่าจริง ต้องไม่ถูกตัดทิ้ง
    clean = status_from_probe(_info())
    assert clean["controllers_stale"] == 0 and clean["runtime_stale"] == 0
    # node รุ่นเก่าไม่ส่ง controller/runtime_arch/runtimes → ไม่มีคีย์ (= ไม่รู้ ไม่ใช่ 0)
    old = status_from_probe({"host": {"lmds_version": "0.6.0", "lmds_commit": "abc1234"},
                             "models": [{"slug": "x", "engine": "llamacpp"}]})
    assert "controllers_stale" not in old and "runtime_stale" not in old and "llamacpp_build" not in old
