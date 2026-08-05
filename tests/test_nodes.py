"""ทะเบียนเครื่อง (fleet) + การตรวจสายเร็วบนเครื่องนั้น

เทสไม่ยิง SSH จริง — ส่วนที่ต่อออกนอกเครื่องถูก mock ทั้งหมด
"""

from __future__ import annotations

import pytest

from lmds.hardware import profiler
from lmds.nodes import (
    Node,
    NodeError,
    add,
    find,
    load,
    nodes_file,
    remove,
    suggest_name,
    update,
    validate_cluster_ip,
)


def make(name="spark1", host="10.0.0.5", user="ops", **kw) -> Node:
    return Node(name=name, host=host, user=user, **kw)


# ── ทะเบียน ────────────────────────────────────────────────────────────────
def test_add_then_find_roundtrip():
    add(make(cluster_ip="10.10.0.1", cluster_iface="enp1s0f0np0"))
    node = find("spark1")
    assert node.target == "ops@10.0.0.5"
    assert node.cluster_ip == "10.10.0.1"
    assert node.cluster_iface == "enp1s0f0np0"


def test_registry_file_is_not_world_readable():
    """ไฟล์มีชื่อ user/host ของเครื่องภายใน — ต้องไม่ให้ user อื่นอ่าน"""
    add(make())
    assert nodes_file().stat().st_mode & 0o077 == 0


def test_password_is_never_a_field():
    """กันการเผลอเพิ่มฟิลด์รหัสผ่านกลับเข้ามาในอนาคต"""
    assert not any("pass" in f for f in Node.__dataclass_fields__)


def test_missing_registry_reads_as_empty():
    assert load() == []


def test_duplicate_name_is_refused():
    add(make())
    with pytest.raises(NodeError, match="อยู่แล้ว"):
        add(make(host="10.0.0.9"))


def test_same_target_under_another_name_is_refused():
    add(make())
    with pytest.raises(NodeError, match="spark1"):
        add(make(name="spark1-again"))


def test_bad_name_is_refused():
    with pytest.raises(NodeError):
        add(make(name="Spark One!"))


def test_remove_unknown_name_raises():
    with pytest.raises(NodeError):
        remove("nope")


def test_update_cannot_silently_change_the_address():
    """ที่อยู่เปลี่ยน = คนละเครื่อง ต้อง remove แล้ว add ใหม่"""
    add(make())
    update("spark1", host="10.0.0.99", user="root", last_seen="now")
    node = find("spark1")
    assert (node.host, node.user) == ("10.0.0.5", "ops")
    assert node.last_seen == "now"


def test_update_validates_cluster_ip():
    add(make())
    with pytest.raises(NodeError):
        update("spark1", cluster_ip="10.10.0")
    assert find("spark1").cluster_ip == ""


@pytest.mark.parametrize("bad", ["10.10.0", "127.0.0.1", "224.0.0.1", "not-an-ip", "::1"])
def test_bad_cluster_ip_is_refused(bad):
    with pytest.raises(NodeError):
        validate_cluster_ip(bad)


def test_blank_cluster_ip_is_allowed():
    """ยังไม่ได้ตั้ง ≠ ตั้งผิด — เครื่องที่ไม่ได้ใช้ stacked ไม่ต้องกรอก"""
    assert validate_cluster_ip("") == ""
    assert validate_cluster_ip("  10.10.0.2 ") == "10.10.0.2"


@pytest.mark.parametrize(
    "host, expected",
    [("10.0.0.5", "node-10-0-0-5"), ("spark1.local", "spark1"), ("SPARK-A", "spark-a")],
)
def test_suggest_name(host, expected):
    assert suggest_name(host, set()) == expected


def test_suggest_name_avoids_collisions():
    assert suggest_name("spark1.local", {"spark1"}) == "spark1-2"


# ── ตรวจสายเร็วบนเครื่อง ────────────────────────────────────────────────────
def _fake_sysfs(tmp_path, monkeypatch, *, speed="200000", driver="mlx5_core",
                vendor="0x15b3", state="up", infiniband=True):
    net = tmp_path / "sys/class/net/enp1s0f0np0"
    (net / "device/driver").mkdir(parents=True)
    (net / "speed").write_text(speed)
    (net / "operstate").write_text(state)
    (net / "device/vendor").write_text(vendor)
    (net / "device/driver").rmdir()
    (tmp_path / f"sys/bus/pci/drivers/{driver}").mkdir(parents=True)
    (net / "device/driver").symlink_to(tmp_path / f"sys/bus/pci/drivers/{driver}")
    if infiniband:
        (tmp_path / "sys/class/infiniband/mlx5_0").mkdir(parents=True)

    real_path = profiler.Path

    class FakePath(type(real_path())):
        def __new__(cls, *args):
            text = str(args[0]) if args else ""
            if text.startswith("/sys/"):
                return real_path(str(tmp_path) + text)
            return real_path(*args)

    monkeypatch.setattr(profiler, "Path", FakePath)
    monkeypatch.setattr(profiler, "_run", lambda *a, **k:
                        "2: enp1s0f0np0    inet 10.10.0.1/24 brd 10.10.0.255 scope global\n")


def test_fabric_detects_connectx_rdma(tmp_path, monkeypatch):
    _fake_sysfs(tmp_path, monkeypatch)
    fabric = profiler.detect_fabric()
    assert fabric["tier"] == "rdma"
    assert fabric["best_gbps"] == 200
    assert fabric["cluster_capable"]
    link = fabric["links"][0]
    assert (link["iface"], link["ip"], link["connectx"]) == ("enp1s0f0np0", "10.10.0.1", True)


def test_fabric_without_rdma_is_only_fast(tmp_path, monkeypatch):
    """100G ที่ยังไม่เปิด RoCE ต้องบอกตรง ๆ ว่ายังไม่ใช่ RDMA"""
    _fake_sysfs(tmp_path, monkeypatch, speed="100000", infiniband=False)
    fabric = profiler.detect_fabric()
    assert fabric["tier"] == "fast"
    assert fabric["rdma_devices"] == []


def test_slow_nic_is_not_cluster_capable(tmp_path, monkeypatch):
    _fake_sysfs(tmp_path, monkeypatch, speed="1000", driver="e1000e", vendor="0x8086",
                infiniband=False)
    fabric = profiler.detect_fabric()
    assert fabric["tier"] == "basic"
    assert not fabric["cluster_capable"]


def test_link_that_is_down_reports_no_speed(tmp_path, monkeypatch):
    """ลิงก์ลง /sys/class/net/*/speed อ่านได้ -1 — ห้ามนับเป็นสายใช้งานได้"""
    _fake_sysfs(tmp_path, monkeypatch, speed="-1", state="down")
    fabric = profiler.detect_fabric()
    assert fabric["best_gbps"] is None
    assert fabric["tier"] == "unknown"
    assert not fabric["cluster_capable"]


def test_fabric_is_unknown_when_sysfs_is_absent(monkeypatch):
    """บนเครื่องที่ไม่ใช่ Linux ต้องรายงานว่าตรวจไม่ได้ ไม่ใช่เดาว่าไม่มี"""
    monkeypatch.setattr(profiler.Path, "is_dir", lambda self: False)
    fabric = profiler.detect_fabric()
    assert fabric["tier"] == "unknown"
    assert fabric["links"] == []


def test_detect_cpu_reports_cores():
    cpu = profiler.detect_cpu()
    assert cpu["cores"] and cpu["cores"] > 0
    assert cpu["percent"] is None or cpu["percent"] >= 0


def test_add_registers_a_machine_without_lmds(monkeypatch):
    """เครื่องที่ยังไม่ได้ลง LMDS ต้องเพิ่มเข้าทะเบียนได้ — ไม่งั้นวางลำดับกลับหัว
    (key ติดตั้งไปแล้ว และ hub ใช้ node run สั่งติดตั้งต่อได้)"""
    from typer.testing import CliRunner

    from lmds.cli.main import app

    monkeypatch.setattr("lmds.nodes.ensure_key", lambda: None)
    monkeypatch.setattr("lmds.nodes.check_login", lambda *a, **k: True)
    monkeypatch.setattr("lmds.nodes.probe",
                        lambda node: (_ for _ in ()).throw(NodeError("ยังไม่ได้ติดตั้ง LMDS")))

    result = CliRunner().invoke(app, ["node", "add", "10.0.0.5", "--user", "ops", "--name", "new"])
    assert result.exit_code == 0, result.output
    node = find("new")
    assert node is not None
    assert "ยังไม่ได้ติดตั้ง LMDS" in node.last_error
    assert node.last_seen == ""  # ยังไม่เคยอ่านสถานะได้จริง


# ── HF token ที่เครื่องมีอยู่แล้ว ────────────────────────────────────────────
def test_hf_token_falls_back_to_the_huggingface_cli_file(tmp_path, monkeypatch):
    """เครื่องที่เคยโหลดโมเดล gated มี ~/.cache/huggingface/token อยู่แล้ว — อย่าถามซ้ำ"""
    from lmds.secrets import store

    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("hf_fromcli\n", encoding="utf-8")
    monkeypatch.setattr(store.Path, "home", classmethod(lambda cls: home))

    assert store.get_secret("hf") == "hf_fromcli"
    assert store.secret_source("hf") == "huggingface-cli"


def test_lmds_own_hf_token_wins_over_the_cli_file(tmp_path, monkeypatch):
    """ของที่ผู้ใช้ตั้งกับ LMDS ต้องชนะไฟล์ของเครื่องมือตัวอื่นเสมอ"""
    from lmds.secrets import store

    home = tmp_path / "home2"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("hf_fromcli", encoding="utf-8")
    monkeypatch.setattr(store.Path, "home", classmethod(lambda cls: home))
    store.set_secret("hf", "hf_mine")

    assert store.get_secret("hf") == "hf_mine"


def test_other_secrets_never_read_the_hf_file(tmp_path, monkeypatch):
    from lmds.secrets import store

    home = tmp_path / "home3"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("hf_fromcli", encoding="utf-8")
    monkeypatch.setattr(store.Path, "home", classmethod(lambda cls: home))

    assert store.get_secret("openai") is None


def test_node_run_passes_flags_through(monkeypatch):
    """`node run x logs y -n 100` — flag ต้องไปถึงคำสั่งปลายทาง ไม่ใช่โดน typer กินเอง"""
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes.ssh import Result

    add(make(name="n1"))
    seen = {}

    def fake_run(node, command, timeout=60):
        seen["cmd"] = command
        return Result(0, "", "")

    monkeypatch.setattr("lmds.nodes.run", fake_run)

    result = CliRunner().invoke(app, ["node", "run", "n1", "logs", "my-model", "-n", "100"])
    assert result.exit_code == 0, result.output
    assert seen["cmd"] == "lmds logs my-model -n 100"


# ── ที่อยู่สำรอง (Tailscale/VPN) ─────────────────────────────────────────────
def test_alt_hosts_are_tried_after_the_primary():
    """เครื่องเดียวกันเข้าได้หลายทาง — LAN ตอนอยู่ออฟฟิศ, Tailscale ตอนออกนอก"""
    node = make(host="10.0.0.5", alt_hosts=["100.64.0.5"])
    assert node.all_hosts == ["10.0.0.5", "100.64.0.5"]


def test_alt_host_duplicate_of_primary_is_dropped():
    node = make(host="10.0.0.5", alt_hosts=["10.0.0.5", ""])
    assert node.all_hosts == ["10.0.0.5"]


def test_failover_only_happens_when_unreachable(monkeypatch):
    """คำสั่งที่ล้มเพราะ exit code ของตัวมันเอง ต้องไม่ถูกยิงซ้ำที่อยู่สำรอง
    ไม่งั้นคำสั่งที่มีผลข้างเคียงจะทำงานสองรอบ"""
    from lmds.nodes import ssh

    tried = []

    def fake(target, port, wrapped, timeout):
        tried.append(target)
        return ssh.Result(1, "", "boom")   # ต่อได้ แต่คำสั่งล้ม

    monkeypatch.setattr(ssh, "_run_ssh", fake)
    ssh.run(make(host="a", alt_hosts=["b"]), "true")
    assert tried == ["ops@a"]


def test_failover_happens_on_timeout(monkeypatch):
    from lmds.nodes import ssh

    tried = []

    def fake(target, port, wrapped, timeout):
        tried.append(target)
        if target.endswith("@a"):
            return ssh.Result(124, "", "หมดเวลา 60s")
        return ssh.Result(0, "ok", "")

    monkeypatch.setattr(ssh, "_run_ssh", fake)
    result = ssh.run(make(host="a", alt_hosts=["b"]), "true")
    assert tried == ["ops@a", "ops@b"] and result.stdout == "ok"
