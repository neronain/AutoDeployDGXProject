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
    in_saved_order,
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


def test_stack_defaults_to_true_and_can_be_turned_off():
    """กลุ่ม stacked เป็นสิ่งที่ระบบเสนอเอง — ต้องสั่งไม่เอาเครื่องนี้เข้ากลุ่มได้"""
    add(make())
    assert find("spark1").stack is True
    assert update("spark1", stack=False).stack is False
    assert find("spark1").stack is False        # อ่านกลับจากไฟล์แล้วยังปิดอยู่


def test_registry_without_the_stack_field_still_joins():
    """ทะเบียนที่เขียนไว้ก่อนมีฟิลด์นี้ (หรือแก้มือเป็น null) ต้องไม่กลายเป็นปิดทั้งฟลีต"""
    add(make())
    nodes_file().write_text("nodes:\n- name: spark1\n  host: 10.0.0.5\n  user: ops\n  stack: null\n",
                            encoding="utf-8")
    assert load()[0].stack is True


def test_saved_order_puts_unknown_names_last_and_ignores_stale_ones():
    """ลำดับที่เก็บไว้กับทะเบียนไม่ตรงกันเป็นเรื่องปกติ — เครื่องใหม่ต่อท้าย ชื่อที่ลบไปแล้วข้าม"""
    nodes = [make(name=n, host=f"10.0.0.{i}") for i, n in enumerate(["a", "b", "c"], start=1)]
    ordered = in_saved_order(nodes, ["c", "ลบไปแล้ว", "a"])
    assert [n.name for n in ordered] == ["c", "a", "b"]


def test_saved_order_of_nothing_keeps_registry_order():
    nodes = [make(name=n, host=f"10.0.0.{i}") for i, n in enumerate(["a", "b"], start=1)]
    assert [n.name for n in in_saved_order(nodes, [])] == ["a", "b"]


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

    def fake(target, port, wrapped, timeout, stdin_text=""):
        tried.append(target)
        return ssh.Result(1, "", "boom")   # ต่อได้ แต่คำสั่งล้ม

    monkeypatch.setattr(ssh, "_run_ssh", fake)
    ssh.run(make(host="a", alt_hosts=["b"]), "true")
    assert tried == ["ops@a"]


def test_failover_happens_on_timeout(monkeypatch):
    from lmds.nodes import ssh

    tried = []

    def fake(target, port, wrapped, timeout, stdin_text=""):
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
    ซึ่งมีขั้นตอนที่ lmds ไม่ได้ห่อไว้ (prepare-runtime, sync-worker, test-text)

    ใช้ stream() ไม่ใช่ run() ตั้งแต่ 2026-08-31 — download 90 GB ใช้เวลาเป็นสิบนาที
    ของเดิมรอจนจบแล้วค่อยพ่นผลทีเดียว ระหว่างนั้นแยกไม่ออกว่าทำงานอยู่หรือค้างไปแล้ว
    (และ stream() คือทางเดียวที่ให้ node ยืม HF_TOKEN ได้อย่างปลอดภัย)
    """
    from typer.testing import CliRunner

    from lmds.cli.main import app

    add(make(name="n2"))
    seen = {}

    class FakeProc:
        stdout = iter([])

        def wait(self):
            return 0

    def fake_stream(node, command, secret_env=None):
        seen["cmd"] = command
        proc = FakeProc()
        proc.stdout = _Lines([b"ok\n"])
        return proc

    monkeypatch.setattr("lmds.nodes.stream", fake_stream)
    result = CliRunner().invoke(app, ["node", "ctl", "n2", "my-model", "start", "--gpu-util", "0.8"])
    assert result.exit_code == 0, result.output
    assert "bundles/my-model" in seen["cmd"]
    assert "start --gpu-util 0.8" in seen["cmd"]


class _Lines:
    """stdout ปลอมของ Popen — คืนไบต์ทีละบรรทัดแล้วจบด้วยค่าว่าง"""

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else b""


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


# ── ชื่อเครื่อง ────────────────────────────────────────────────────────────────
# ชื่อเป็นของผู้ใช้ตั้ง — บังคับพิมพ์ตัวเล็กทั้งที่ป้ายบนเครื่องเขียน "MSI6" คือกฎที่
# อธิบายไม่ได้ · แต่ชื่อถูกต่อเป็นคำสั่ง SSH จริง จึงต้องกันของที่ shell ตีความออกไป

@pytest.mark.parametrize("name", [
    "MSI6", "GPU-Rig-02", "dgx-veerasiam", "spark_head", "node.01",
    "เครื่องหลัก", "ปลาย-01", "机器2", "1node", "sv-01.local",
])
def test_names_people_actually_use_are_accepted(name):
    from lmds.nodes.registry import name_ok

    assert name_ok(name) is True


@pytest.mark.parametrize("name", [
    "", "_start", ".hidden", "-lead", "has space", "a;rm -rf /", "back`tick`",
    "pipe|it", "dollar$x", "quote'x", 'dq"x', "new\nline", "x" * 64,
])
def test_names_that_would_break_a_shell_command_are_refused(name):
    from lmds.nodes.registry import name_ok

    assert name_ok(name) is False, name


def test_thai_names_are_all_or_nothing():
    """`\\w` ของ Python ไม่นับสระบน/ล่าง — "ปลาย-01" เคยผ่านแต่ "เครื่องหลัก" ตก
    ซึ่งเป็นกฎที่อธิบายให้ผู้ใช้ไม่ได้เลย
    """
    from lmds.nodes.registry import name_ok

    assert name_ok("ปลาย-01") == name_ok("เครื่องหลัก") is True


def test_install_repo_can_point_somewhere_else(monkeypatch):
    """repo ส่วนตัวดึงผ่าน HTTPS แบบไม่ล็อกอินไม่ได้ — ค่าตายตัวตัวเดียวแปลว่า
    `lmds node install` ใช้กับ repo ส่วนตัวไม่ได้เลย · ไซต์ต้องชี้ไป SSH remote
    หรือ mirror ภายในของตัวเองได้
    """
    import importlib

    monkeypatch.setenv("LMDS_REPO_URL", "git@github.com:acme/lmds.git")
    ssh = importlib.reload(importlib.import_module("lmds.nodes.ssh"))
    try:
        assert ssh.REPO_URL == "git@github.com:acme/lmds.git"
        assert "git@github.com:acme/lmds.git" in ssh.install_script()
    finally:
        monkeypatch.delenv("LMDS_REPO_URL", raising=False)
        importlib.reload(ssh)


def test_node_install_all_updates_every_machine(tmp_path, monkeypatch, isolated_config):
    """อัปเดตทีละเครื่องด้วยมือแปลว่ามีวันลืมเครื่องหนึ่ง แล้วมันค้างเวอร์ชันเก่าอยู่เงียบ ๆ
    จนกว่าจะมีคนสังเกตเห็น (msi-6 ค้างที่ 0.1.0 อยู่หลายรอบ)
    """
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes import Node, add

    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path))
    for index, name in enumerate(("a", "b", "c"), 1):
        add(Node(name=name, host=f"10.0.0.{index}", user="u"))
    done = []
    monkeypatch.setattr("lmds.nodes.install_lmds",
                        lambda node, with_prereq=False: done.append(node.name)
                        or SimpleNamespace(ok=True, stdout="", stderr=""))
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": {"lmds_version": "0.2.0"}})

    result = CliRunner().invoke(app, ["node", "install", "--all"])
    assert result.exit_code == 0, result.output
    assert done == ["a", "b", "c"], "ต้องครบทุกเครื่อง ไม่ใช่หยุดที่ตัวแรก"


def test_node_install_all_reports_which_ones_failed(tmp_path, monkeypatch, isolated_config):
    """เครื่องหนึ่งล้มต้องไม่หยุดเครื่องที่เหลือ และต้องบอกชื่อตัวที่ล้ม ไม่ใช่แค่ exit ไม่เป็นศูนย์"""
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes import Node, add

    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path))
    for index, name in enumerate(("good", "bad"), 1):
        add(Node(name=name, host=f"10.0.0.{index}", user="u"))
    monkeypatch.setattr("lmds.nodes.install_lmds",
                        lambda node, with_prereq=False: SimpleNamespace(
                            ok=node.name != "bad", stdout="", stderr="ต่อไม่ได้"))
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {"host": {"lmds_version": "0.2.0"}})

    result = CliRunner().invoke(app, ["node", "install", "--all"])
    assert result.exit_code == 1
    assert "bad" in result.output and "พร้อมแล้ว" in result.output


def test_registry_still_has_no_password_field():
    """ผู้ใช้เสนอให้เก็บรหัสผ่านไว้ใช้ตอนต้องใช้สิทธิ์ — คำตอบคือถามใหม่ตอนนั้น ไม่ใช่เก็บไว้
    ทะเบียนต้องไม่มีที่ให้เก็บ ไม่งั้นวันหนึ่งจะมีคนใส่ลงไป
    """
    from dataclasses import fields

    from lmds.nodes import Node

    names = {f.name for f in fields(Node)}
    assert not {"password", "passwd", "sudo_password", "secret"} & names


def test_privileged_steps_verify_the_result_not_the_exit_code(monkeypatch, tmp_path):
    """sudo ที่รหัสผิดคืน 1 เหมือนกับคำสั่งที่ล้มด้วยเหตุอื่น และบางคำสั่งคืน 0 ทั้งที่ไม่ได้ทำอะไร
    — ต้องตรวจผลจริงอีกที ไม่ใช่เชื่อ exit code
    """
    from types import SimpleNamespace

    from lmds.nodes import Node, run_privileged

    seen = []

    def fake_run(node, command, timeout=60, stdin_text=""):
        seen.append((command, stdin_text))
        if "enable-linger" in command:
            return SimpleNamespace(ok=False, exit_code=1, stdout="", stderr="sudo: wrong password")
        return SimpleNamespace(ok=True, exit_code=0, stdout="Linger=yes", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    outcomes = run_privileged(Node(name="n", host="h", user="u"), "รหัสผ่าน")
    # คำสั่งล้ม แต่ตัวตรวจบอกว่าสำเร็จ → เชื่อตัวตรวจ (เช่น linger เปิดอยู่ก่อนแล้ว)
    assert outcomes[0]["ok"] is True


def test_the_sudo_password_goes_through_stdin_not_the_command_line(monkeypatch):
    """รหัสผ่านใน argv = คนอื่นบนเครื่องเดียวกันอ่านได้จาก /proc"""
    from types import SimpleNamespace

    from lmds.nodes import Node, run_privileged

    seen = []

    def fake_run(node, command, timeout=60, stdin_text=""):
        seen.append((command, stdin_text))
        # ตัวตรวจต้องบอกว่ายังไม่เรียบร้อย ไม่งั้นขั้นนั้นถูกข้ามไปเลย
        return SimpleNamespace(ok=bool(stdin_text), exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    run_privileged(Node(name="n", host="h", user="u"), "s3cret")
    command, stdin_text = next((c, s) for c, s in seen if s)
    assert "s3cret" not in command, "รหัสผ่านต้องไม่อยู่ในคำสั่ง"
    assert stdin_text.strip() == "s3cret"
    assert "sudo -S" in command, "ต้องบอก sudo ให้อ่านรหัสจาก stdin"


def test_a_wrong_password_does_not_get_a_tick_from_a_step_that_was_already_done(monkeypatch):
    """สถานะถูกอยู่ก่อนแล้ว + รหัสผ่านผิด = ขึ้น ✓ ซึ่งชวนให้เข้าใจว่ารหัสผ่านใช้ได้
    ทั้งที่ไม่ได้ถูกใช้เลย (ผู้ใช้เจอจริงกับ msi-6 ที่เปิด linger ไว้เองแล้ว)
    """
    from types import SimpleNamespace

    from lmds.nodes import Node, run_privileged

    used_sudo = []

    def fake_run(node, command, timeout=60, stdin_text=""):
        if stdin_text:
            used_sudo.append(command)
        return SimpleNamespace(ok=True, exit_code=0, stdout="Linger=yes", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    outcomes = run_privileged(Node(name="n", host="h", user="u"), "รหัสผิด")
    assert outcomes[0]["skipped"] is True, "เรียบร้อยอยู่แล้วต้องบอกว่าข้าม ไม่ใช่บอกว่าทำสำเร็จ"
    assert not used_sudo, "ไม่มีอะไรต้องทำก็ไม่ควรแตะ sudo เลย"


def test_a_wrong_password_fails_loudly_when_the_step_is_needed(monkeypatch):
    from types import SimpleNamespace

    from lmds.nodes import Node, run_privileged

    def fake_run(node, command, timeout=60, stdin_text=""):
        if stdin_text:
            return SimpleNamespace(ok=False, exit_code=1, stdout="",
                                   stderr="sudo: 1 incorrect password attempt")
        return SimpleNamespace(ok=False, exit_code=1, stdout="Linger=no", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    outcomes = run_privileged(Node(name="n", host="h", user="u"), "รหัสผิด")
    assert outcomes[0]["ok"] is False
    assert "incorrect password" in outcomes[0]["detail"]


def test_an_old_lmds_is_not_reported_as_unreachable(monkeypatch):
    """เครื่องที่มี lmds เก่า (ไม่มีคำสั่ง agent) พิมพ์ usage ออกมา — เดิมถูกรายงานว่า
    "ต่อ … ไม่ได้" ทั้งที่ SSH ต่อได้สบาย ผู้ใช้จึงไปไล่หาปัญหาเครือข่ายผิดที่
    (เจอจริงกับ AiTop100 ที่มี 0.1.0)
    """
    import pytest
    from types import SimpleNamespace

    from lmds.nodes import Node, NodeError, probe

    def fake_run(node, command, timeout=30, stdin_text=""):
        if "agent info" in command:
            return SimpleNamespace(ok=False, exit_code=2, stdout="",
                                   stderr="Usage: lmds [OPTIONS] COMMAND [ARGS]...")
        return SimpleNamespace(ok=True, exit_code=0, stdout="lmds 0.1.0", stderr="")

    monkeypatch.setattr("lmds.nodes.ssh.run", fake_run)
    with pytest.raises(NodeError) as caught:
        probe(Node(name="aitop100", host="h", user="u"))
    message = str(caught.value)
    assert "เก่าเกินไป" in message and "0.1.0" in message
    assert "lmds node install aitop100" in message, "ต้องบอกคำสั่งที่แก้ได้จริง"
    assert "ต่อ" not in message.split("\n")[0], "ต้องไม่บอกว่าต่อไม่ได้"


def test_a_private_repo_failure_says_what_to_do():
    """"could not read Username for 'https://github.com'" อ่านแล้วไม่รู้เลยว่าต้องทำอะไร
    ความหมายจริงคือ repo เป็น private และเครื่องนั้นไม่มีสิทธิ์ (ผู้ใช้เจอกับ AiTop100)
    """
    from lmds.nodes import Node, explain_install_failure

    hint = explain_install_failure(
        "fatal: could not read Username for 'https://github.com': No such device or address",
        Node(name="aitop100", host="h", user="u"))
    assert "private" in hint and "deploy key" in hint.lower()
    assert "LMDS_REPO_URL" in hint


def test_a_normal_failure_gets_no_made_up_explanation():
    """เดาความหมายผิดแล้วพาไปแก้ผิดที่ — ไม่รู้ก็ไม่ต้องเดา"""
    from lmds.nodes import Node, explain_install_failure

    assert explain_install_failure("disk full", Node(name="n", host="h", user="u")) == ""


def test_tests_never_touch_the_real_nodes_registry():
    """เทสเคยลบ nodes.yaml ของผู้ใช้สองครั้ง — 6 เครื่องจริงถูกแทนด้วย node ของ fixture

    เทสนี้ทำสิ่งที่เคยทำให้พังเป๊ะ ๆ (เพิ่ม node แล้วเขียนทะเบียน) แล้วพิสูจน์สองอย่าง:
    ของที่เขียนไปลงใน sandbox จริง และไฟล์ของผู้ใช้ไม่ถูกแตะ · เทียบกับ REAL_CONFIG_DIR
    ที่ conftest จำไว้ตอน import ไม่ใช่ Path.home() ซึ่งตอนนี้ชี้ไป sandbox แล้ว

    ของเดิมกันด้วยการ snapshot แล้วเขียนคืนตอนจบ session ซึ่งแยกไม่ออกว่าไฟล์เปลี่ยน
    เพราะเทสหรือเพราะ `lmds web` ที่รันอยู่เบื้องหลัง — เทสนี้ไม่ต้องแยก เพราะเทส
    เขียนไปที่ไฟล์ของจริงไม่ได้ตั้งแต่แรก
    """
    from tests.conftest import REAL_CONFIG_DIR

    real_registry = REAL_CONFIG_DIR / "nodes.yaml"
    before = real_registry.read_bytes() if real_registry.exists() else None

    add(Node(name="fixture-node", host="10.0.0.1", user="u"))
    written = nodes_file()

    assert written.exists(), "เทสต้องเขียนลงทะเบียนได้จริง ไม่ใช่แค่ไม่แตะของจริง"
    assert written != real_registry
    assert REAL_CONFIG_DIR not in written.parents, f"เขียนลง {written} ซึ่งอยู่ในของจริง"
    assert "fixture-node" in written.read_text(encoding="utf-8")

    after = real_registry.read_bytes() if real_registry.exists() else None
    if before is not None:
        assert after == before, "ทะเบียนจริงของผู้ใช้ถูกแตะ"


# ── site (จัดกลุ่มตามที่ตั้งเครื่อง — orthogonal กับ cluster) ──────────────────
def test_site_roundtrips_and_defaults_empty():
    add(make())
    assert find("spark1").site == ""          # ทะเบียนเดิมไม่มี site = ว่าง
    add(make(name="cust1", host="10.0.0.9", site="customer-a"))
    assert find("cust1").site == "customer-a"  # save/load เก็บค่าได้


def test_update_sets_and_clears_site():
    add(make())
    assert update("spark1", site="customer-b").site == "customer-b"
    assert update("spark1", site="").site == ""   # ว่าง = เอาป้ายออก


def test_site_does_not_touch_cluster_fields():
    """ตั้ง site ต้องไม่ไปแตะ stack/cluster_ip — cluster เป็นคนละเรื่องโดยสิ้นเชิง"""
    add(make(cluster_ip="10.10.0.1", stack=True))
    node = update("spark1", site="customer-a")
    assert node.cluster_ip == "10.10.0.1"     # ค่า cluster เดิมอยู่ครบ
    assert node.stack is True
