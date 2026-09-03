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


# ── หน้าเว็บ ──────────────────────────────────────────────────────────────
#
# ผู้ใช้รายงาน 2026-08-31: "หน้า gui ยังหาเมนูไม่เจอ" — เพราะรอบแรกทำแต่ CLI
# คนส่วนใหญ่ในทีมทำงานผ่านหน้าเว็บ ฟีเจอร์ที่มีแต่ใน CLI จึงเท่ากับไม่มี

from pathlib import Path as _Path

_INDEX = _Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"


def test_the_console_has_a_clone_control():
    page = _INDEX.read_text(encoding="utf-8")
    assert 'class="n-clone-to"' in page, "ไม่มีช่องเลือกเครื่องปลายทาง"
    assert 'data-nact="clone"' in page, "ไม่มีปุ่ม Clone"
    assert "function fillCloneTargets" in page, "ไม่ได้เติมรายชื่อเครื่องให้"
    assert "/clone/targets" in page and "/clone`" in page


def test_clone_endpoints_exist_and_validate():
    from fastapi.testclient import TestClient
    from lmds.hardware import serving
    from lmds.web import state
    from lmds.web.api import create_app

    try:
        client = TestClient(create_app())
        # ไม่ระบุปลายทาง = บอกให้ชัด ไม่ใช่ 500
        r = client.post("/api/nodes/msi-1/models/demo/clone", json={})
        assert r.status_code == 400, r.text
        assert "ปลายทาง" in r.json()["detail"]
        # เครื่องต้นทางที่ไม่มีในทะเบียน = 404 ไม่ใช่ลิสต์ว่าง ๆ ที่อ่านแล้วงง
        r = client.get("/api/nodes/ไม่มีเครื่องนี้/models/demo/clone/targets")
        assert r.status_code == 404, r.text
    finally:
        state.stop_refresher()
        state.STORE.__init__()
        serving.reset_cache()


def test_targets_put_the_same_site_and_fast_link_first(fleet):
    """สายในแร็คเร็วกว่าข้ามไซต์สิบเท่า — ตัวที่ควรเลือกต้องอยู่บนสุด"""
    from lmds.fleet.clone import plan_clone

    rows = []
    for name in ("far", "slow", "msi-2"):
        plan = plan_clone("demo", "msi-1", name)
        rows.append({"name": name, "same_site": plan.same_site, "link": plan.link})
    rows.sort(key=lambda r: (not r["same_site"], r["link"] != "cluster", r["name"]))

    assert [r["name"] for r in rows] == ["msi-2", "slow", "far"]


class _Result:
    def __init__(self, done):
        self.ok, self.stdout, self.stderr = done.returncode == 0, done.stdout, done.stderr


def _run_locally(cwd):
    import subprocess

    def run(node, script, timeout=120):
        return _Result(subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True, text=True, timeout=timeout))
    return run


def test_inspect_source_finds_a_vllm_model_in_the_hugging_face_cache(fleet, monkeypatch, tmp_path):
    """เคสจริง 2026-09-03: clone Qwen3.6-35B NVFP4 จาก spark04 → RTX4000 ตอบ "ยังไม่มีไฟล์โมเดล ()"

    controller ของ vLLM ไม่มี MODEL_DIR — weight อยู่ใน HF_HOME/hub/models--org--name
    (blobs/ + snapshots/ ที่เป็น symlink) · ต้องหาเจอเองจาก MODEL_ID และนับไฟล์ใน blobs
    """
    from lmds.fleet.clone import inspect_source

    home = tmp_path / "home"; bundle = tmp_path / "bundles" / "demo"; bundle.mkdir(parents=True)
    (bundle / "demo-single.sh").write_text(
        '#!/bin/bash\nMODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4"\nHF_HOME="${HF_HOME:-' + str(home) + '/.cache/huggingface}"\n')
    cache = home / ".cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
    (cache / "blobs").mkdir(parents=True); (cache / "snapshots/abc").mkdir(parents=True)
    (cache / "blobs" / "sha-1").write_bytes(b"x" * 1000)
    (cache / "snapshots/abc/model.safetensors").symlink_to("../../blobs/sha-1")
    monkeypatch.setattr("lmds.nodes.run", _run_locally(tmp_path), raising=False)

    plan = inspect_source(plan_clone("demo", "msi-1", "msi-2"))
    assert plan.model_dir == str(cache)
    assert plan.files == [("blobs/sha-1", 1000)], "ต้องนับไฟล์จริงใน blobs ไม่ใช่ symlink"
    assert plan.bundle_dir.endswith("bundles/demo")


def test_inspect_source_still_reads_model_dir_from_llamacpp_controllers(fleet, monkeypatch, tmp_path):
    from lmds.fleet.clone import inspect_source

    bundle = tmp_path / "bundles" / "gg"; bundle.mkdir(parents=True)
    models = tmp_path / "models" / "gg"; models.mkdir(parents=True)
    (models / "a.gguf").write_bytes(b"g" * 10)
    (bundle / "gg-single.sh").write_text('#!/bin/bash\nMODEL_DIR="' + str(models) + '"\nMODEL_ID="org/gg-GGUF"\n')
    monkeypatch.setattr("lmds.nodes.run", _run_locally(tmp_path), raising=False)

    plan = inspect_source(plan_clone("gg", "msi-1", "msi-2"))
    assert plan.model_dir == str(models) and plan.files == [("a.gguf", 10)]


def test_inspect_source_names_the_missing_location(fleet, monkeypatch, tmp_path):
    """ข้อความเดิม "ยังไม่มีไฟล์โมเดลบน host ()" — วงเล็บว่างบอกอะไรไม่ได้"""
    from lmds.fleet.clone import inspect_source

    bundle = tmp_path / "bundles" / "x"; bundle.mkdir(parents=True)
    (bundle / "x-single.sh").write_text('#!/bin/bash\nMODEL_ID="org/x"\nHF_HOME="' + str(tmp_path) + '/nohf"\n')
    monkeypatch.setattr("lmds.nodes.run", _run_locally(tmp_path), raising=False)
    with pytest.raises(CloneError) as err:
        inspect_source(plan_clone("x", "msi-1", "msi-2"))
    assert "models--org--x" in str(err.value)
