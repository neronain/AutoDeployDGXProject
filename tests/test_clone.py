"""ทำสำเนาโมเดลจากเครื่องที่รันผ่านแล้วไปอีกเครื่อง โดยไม่โหลดจาก HF ใหม่

ผู้ใช้ขอ 2026-08-31: model A บน msi-1 อยากให้ไปรันที่ msi-2 ด้วย เพื่อทำ failover /
กระจายโหลดผ่าน gateway · ของเดิมทางเดียวคือ `download` ซึ่งดึงจาก Hugging Face ใหม่
ทั้งก้อน — IQ4_XS ของ Qwen3.8-Flash-Next คือ 90.8 GB ที่ 40 MB/s = 38 นาที
ทั้งที่เครื่องข้าง ๆ ในแร็คเดียวกันถือไฟล์ชุดเดียวกันอยู่แล้ว

สิ่งที่เทสนี้ยึดไว้:
  - ไฟล์วิ่ง *ตรง* ต้นทาง→ปลายทาง ไม่ผ่าน hub (hub มักเป็นเครื่องเล็กที่จะเป็นคอขวด)
  - เลือกสายเร็วที่สุดที่ทั้งคู่มี
  - กุญแจชั่วคราวถูกถอนออก **เสมอ** แม้การคัดลอกจะล้มกลางคัน
  - กุญแจของ hub ไม่เคยถูกส่งให้ node
"""

import pytest

from lmds.fleet.clone import (
    CloneError,
    _authorized_key_line,
    build_rsync_command,
    make_marker,
    plan_clone,
)


class _Node:
    def __init__(self, name, host, user="neronain", site="", cluster_ip=""):
        self.name, self.host, self.user, self.site = name, host, user, site
        self.cluster_ip, self.port = cluster_ip, 22
        self.all_hosts = [host]


@pytest.fixture
def fleet(monkeypatch):
    nodes = {
        "msi-1": _Node("msi-1", "100.86.7.95", site="TKC", cluster_ip="10.100.152.1"),
        "msi-2": _Node("msi-2", "100.92.145.101", site="TKC", cluster_ip="10.100.152.2"),
        "slow":  _Node("slow", "100.1.2.3", site="TKC"),
        "far":   _Node("far", "100.9.9.9", site="veerasiam"),
    }
    monkeypatch.setattr("lmds.nodes.find", lambda n: nodes.get(n), raising=False)
    return nodes


def test_uses_the_fast_link_when_both_machines_have_one(fleet):
    """cluster_ip คือขาบนการ์ด 200G ที่ตั้งไว้ให้ stacked อยู่แล้ว — copy ควรใช้เส้นเดียวกัน

    ถ้าไปวิ่งบน host (Tailscale relay ที่วัดได้ 82–154 ms) การคัดลอก 90 GB จะกลาย
    เป็นงานข้ามคืนทั้งที่สายในแร็คว่างอยู่
    """
    plan = plan_clone("demo", "msi-1", "msi-2")
    assert plan.link == "cluster"
    assert (plan.source_addr, plan.target_addr) == ("10.100.152.1", "10.100.152.2")
    assert plan.same_site is True


def test_falls_back_to_the_normal_address_when_one_side_has_no_fast_link(fleet):
    plan = plan_clone("demo", "msi-1", "slow")
    assert plan.link == "host"
    assert plan.target_addr == "100.1.2.3"


def test_a_cross_site_clone_is_flagged_not_blocked(fleet):
    """ข้ามไซต์ยังทำได้ แต่คนสั่งควรรู้ตัวว่ากำลังลาก 90 GB ข้ามเน็ตนอก"""
    plan = plan_clone("demo", "msi-1", "far")
    assert plan.same_site is False


def test_cloning_onto_itself_is_refused(fleet):
    with pytest.raises(CloneError, match="เครื่องเดียวกัน"):
        plan_clone("demo", "msi-1", "msi-1")


def test_unknown_machine_says_which_one(fleet):
    with pytest.raises(CloneError, match="ไม่รู้จักเครื่องปลายทาง"):
        plan_clone("demo", "msi-1", "ไม่มีเครื่องนี้")
    with pytest.raises(CloneError, match="ไม่รู้จักเครื่องต้นทาง"):
        plan_clone("demo", "ไม่มีเครื่องนี้", "msi-2")


def test_the_copy_runs_between_the_two_nodes_not_through_the_hub(fleet):
    plan = plan_clone("demo", "msi-1", "msi-2")
    plan.model_dir = "/home/neronain/models/demo"
    plan.bundle_dir = "/home/neronain/bundles/demo"

    cmd = build_rsync_command(plan, "neronain")

    # ปลายทางของ rsync คือเครื่องที่สอง — คำสั่งนี้รันบนเครื่องแรก
    assert "neronain@10.100.152.2" in cmd
    assert "/home/neronain/models/demo/" in cmd
    assert "--partial --append-verify" in cmd, "ไฟล์ 40 GB ขาดกลางคันต้องต่อได้ ไม่เริ่มใหม่"
    assert "ssh-agent" in cmd and "ssh-add -" in cmd


def test_the_private_key_never_touches_the_source_disk(fleet):
    """กุญแจเข้า ssh-agent ในหน่วยความจำเท่านั้น — ห้ามมีจังหวะที่มันถูกเขียนลงไฟล์"""
    plan = plan_clone("demo", "msi-1", "msi-2")
    plan.model_dir, plan.bundle_dir = "/m", "/b"
    cmd = build_rsync_command(plan, "neronain")

    assert 'KEY="$(cat)"' in cmd, "ต้องอ่านกุญแจจาก stdin"
    assert "unset KEY" in cmd
    # ห้ามมีการเขียนกุญแจลงไฟล์ไม่ว่ารูปแบบใด
    for bad in ("> ~/.ssh/", "id_rsa", "id_ed25519", "mktemp"):
        assert bad not in cmd, f"พบร่องรอยการเขียนกุญแจลงดิสก์: {bad}"


def test_temp_key_is_restricted_and_uniquely_marked():
    marker = make_marker()
    line = _authorized_key_line("ssh-ed25519 AAAAC3Nz... ", marker)
    assert line.startswith("restrict "), "กุญแจชั่วคราวต้องปิด port-forward/agent/pty"
    assert marker in line
    assert make_marker() != marker, "marker ต้องไม่ซ้ำ ไม่งั้นถอนผิดบรรทัด"


def test_revoke_matches_only_the_marker_line(monkeypatch, fleet):
    """authorized_keys มีกุญแจของ hub และของผู้ใช้อยู่ด้วย — ลบพลาดคือล็อกตัวเองออก"""
    from lmds.fleet import clone

    seen = {}

    class R:
        ok = True

    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, script, timeout=60: seen.update(script=script) or R(),
                        raising=False)
    marker = "lmds-clone-deadbeef"
    assert clone.revoke_temp_key(fleet["msi-2"], marker) is True
    assert "grep -v -F" in seen["script"], "ต้องตัดเฉพาะบรรทัดที่ตรงกับ marker แบบตรงตัว"
    assert marker in seen["script"]
