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


def test_scanner_does_not_double_count_hf_symlinks(tmp_path, monkeypatch):
    """HF cache เก็บไฟล์จริงใน blobs/ แล้ว snapshots/ เป็น symlink ชี้มา —
    นับทั้งสองอย่างได้ขนาดเป็นเท่าตัว แล้ววางแผนพื้นที่ผิด"""
    from lmds import scanner

    root = tmp_path / "hub" / "models--org--model"
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "abc").mkdir(parents=True)
    blob = root / "blobs" / "deadbeef"
    blob.write_bytes(b"x" * 4096)
    (root / "snapshots" / "abc" / "model.safetensors").symlink_to(blob)

    monkeypatch.setattr(scanner, "candidate_roots", lambda extra=None: [tmp_path])
    found = [m for m in scanner.scan() if m.kind == "hf"]
    assert len(found) == 1
    assert found[0].size_bytes == 4096, "นับ symlink ซ้ำกับ blob"


def test_node_ctl_runs_the_bundle_controller(monkeypatch):
    """`lmds node run` สั่งคำสั่งของ lmds · `node ctl` สั่งสคริปต์ controller ในตัว bundle
    ซึ่งมีขั้นตอนที่ lmds ไม่ได้ห่อไว้ (prepare-runtime, sync-worker, test-text)"""
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes.ssh import Result

    add(make(name="n2"))
    seen = {}

    def fake_run(node, command, timeout=60):
        seen["cmd"] = command
        return Result(0, "ok", "")

    monkeypatch.setattr("lmds.nodes.run", fake_run)
    result = CliRunner().invoke(app, ["node", "ctl", "n2", "my-model", "start", "--gpu-util", "0.8"])
    assert result.exit_code == 0, result.output
    assert "bundles/my-model" in seen["cmd"]
    assert "start --gpu-util 0.8" in seen["cmd"]


def test_virtual_nic_without_device_link_is_still_reported(tmp_path, monkeypatch):
    """NIC เสมือน (VM/cloud) ไม่มี /sys/class/net/<if>/device — เดิมข้ามทิ้งทั้งใบ
    เครื่องแบบนั้นจึงรายงานว่า "ไม่มีเครือข่ายเลย" ทั้งที่มี IP อยู่
    (เจอจริงบน OrbStack VM ที่จะใช้เป็น controller)"""
    net = tmp_path / "sys/class/net/eth0"
    net.mkdir(parents=True)
    (net / "speed").write_text("10000")
    (net / "operstate").write_text("up")
    (tmp_path / "sys/class/net/lo").mkdir(parents=True)

    real_path = profiler.Path

    class FakePath(type(real_path())):
        def __new__(cls, *args):
            text = str(args[0]) if args else ""
            return real_path(str(tmp_path) + text) if text.startswith("/sys/") else real_path(*args)

    monkeypatch.setattr(profiler, "Path", FakePath)
    monkeypatch.setattr(profiler, "_run", lambda *a, **k:
                        "2: eth0    inet 192.168.139.92/24 brd 192.168.139.255 scope global\n")

    fabric = profiler.detect_fabric()
    assert [l["iface"] for l in fabric["links"]] == ["eth0"]
    assert fabric["links"][0]["ip"] == "192.168.139.92"
    assert fabric["best_gbps"] == 10


def test_docker_and_bridge_interfaces_are_ignored(tmp_path, monkeypatch):
    """docker0/veth/br- ไม่ใช่ทางออกจริงของเครื่อง — รกและทำให้เลือก cluster IP ผิด"""
    for name in ("docker0", "veth1234", "br-abc"):
        d = tmp_path / "sys/class/net" / name
        d.mkdir(parents=True)
        (d / "speed").write_text("10000")
        (d / "operstate").write_text("up")

    real_path = profiler.Path

    class FakePath(type(real_path())):
        def __new__(cls, *args):
            text = str(args[0]) if args else ""
            return real_path(str(tmp_path) + text) if text.startswith("/sys/") else real_path(*args)

    monkeypatch.setattr(profiler, "Path", FakePath)
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: "")
    assert profiler.detect_fabric()["links"] == []


# ── telemetry ของ GPU ───────────────────────────────────────────────────────
def _smi(monkeypatch, line: str) -> list:
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: line + "\n")
    return profiler.detect_gpus()[0]


def test_telemetry_is_read_from_nvidia_smi(monkeypatch):
    gpu = _smi(monkeypatch, "NVIDIA RTX 5090, 32768, 12.0, 8192, 55, 62, 210.5, 575, 41, "
                            "2100, 2520, 7001, 2100, 5, 16")[0]
    assert (gpu.temperature_c, gpu.power_w, gpu.power_limit_w, gpu.fan_pct) == (62, 210.5, 575.0, 41)
    assert (gpu.clock_graphics_mhz, gpu.clock_graphics_max_mhz) == (2100, 2520)
    assert (gpu.pcie_gen, gpu.pcie_width) == (5, 16)


def test_missing_telemetry_is_none_not_zero(monkeypatch):
    """GB10 (unified SoC) ตอบ [N/A] หลายตัว — โชว์ 0W ทั้งที่การ์ดทำงานอยู่คือการโกหก
    ค่าจริงจากเครื่อง: NVIDIA GB10, 43°C, 4.28W, power.limit/fan/clocks.mem = [N/A]"""
    gpu = _smi(monkeypatch, "NVIDIA GB10, [N/A], 12.1, [N/A], 0, 43, 4.28, [N/A], [N/A], "
                            "208, 3003, [N/A], 208, 1, 1")[0]
    assert gpu.temperature_c == 43 and gpu.power_w == 4.28
    assert gpu.power_limit_w is None, "ไม่รายงาน ≠ ไม่มีเพดาน"
    assert gpu.fan_pct is None, "ไม่รายงาน ≠ พัดลมหยุด"
    assert gpu.clock_memory_mhz is None
    assert gpu.clock_graphics_mhz == 208 and gpu.clock_graphics_max_mhz == 3003
