"""`lmds cluster apply` ต้องไม่ย้ายไฟล์ netplan ที่ถือสายบริหารออกจาก /etc/netplan — audit 2026-10-06 (เครื่องหลุดเครือข่าย)

apply ย้ายทุกไฟล์ที่ประกาศ interface ของคลัสเตอร์ไป /root/netplan-disabled *ทั้งไฟล์* โดยเชื่อว่า "ไฟล์ของสายบริหารไม่เอ่ยถึง
ConnectX" · เครื่องที่ installer ของ Ubuntu Server เขียนทุก NIC ลงไฟล์เดียว (`50-cloud-init.yaml`) ไม่เป็นแบบนั้น:

    eno1 (10.2.1.11/24 + default route) + enp1s0f1np1  →  หลัง apply เหลือ `99-lmds-cluster.yaml` ไฟล์เดียว
    สายบริหารไม่ถูกตั้งค่าที่ไหนเลย และทุกขั้นรายงาน [pass] เพราะ verify ดูแค่ interface ของคลัสเตอร์

เทสในไฟล์นี้รัน **สคริปต์จริงที่ apply_plan สร้าง** ใต้ bash (และ python3 จริงสำหรับ preflight) กับ /etc/netplan ที่เป็นโฟลเดอร์
จริงในแซนด์บ็อกซ์ต่อเครื่อง — มีแค่ sudo ที่ถูกถอดออก และ `netplan` / `install` ที่เป็นตัวปลอมจดว่าถูกเรียก · สิ่งที่ยืนยันคือ
**ไฟล์บนดิสก์หลัง apply** ไม่ใช่ข้อความในสคริปต์
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lmds.nodes.netplan import (
    NETPLAN_FILE,
    PREFLIGHT_STEP,
    NetplanError,
    apply_plan,
    apply_script,
    build_plan,
    inspect_nodes,
    read_preflight,
    render_netplan,
)
from lmds.nodes.registry import Node
from tests.test_cluster_network_setup import fake_sys, spark

IFACE = "enp1s0f1np1"
STAMP = "20261006-120000"

# ไฟล์เดียวรวมทุก NIC — แบบที่ installer ของ Ubuntu Server เขียน (รีโปรของ auditor)
SINGLE_FILE = f"""network:
  version: 2
  ethernets:
    eno1:
      addresses: [10.2.1.11/24]
      routes:
        - to: default
          via: 10.2.1.1
      nameservers:
        addresses: [10.2.1.1]
    {IFACE}:
      dhcp4: false
"""
MGMT_ONLY = """network:
  version: 2
  ethernets:
    eno1:
      dhcp4: true
"""
CLUSTER_ONLY = f"""network:
  version: 2
  ethernets:
    {IFACE}:
      addresses: [192.168.100.10/24]
"""
# ไฟล์ตามคู่มือ NVIDIA "connect two Sparks": สอง function ของพอร์ตเดียวกัน มีแต่ link-local
NVIDIA_CX7 = f"""network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    {IFACE}:
      link-local: [ ipv4 ]
"""


class Machines:
    """`lmds.nodes.run` ที่รันสคริปต์จริงในเครื่องนี้: /etc/netplan · /root/netplan-disabled · /tmp/lmds-netplan.* ของแต่ละ node
    ถูกเบี่ยงเข้าโฟลเดอร์ของ node นั้น · sudo ถูกถอด (รหัสไม่ถูกตรวจ — เป็นเรื่องของเทสอื่น) · ip/ping/ufw ตอบแบบเครื่องปกติ"""

    def __init__(self, root: Path, python: str = "real"):
        self.root = root
        self.calls: list[tuple[str, str]] = []
        self.bin = root / "bin"
        self.bin.mkdir(parents=True)
        self._shim("netplan", 'echo "$FAKE_NODE netplan $*" >> "$FAKE_CALLS"\n')
        # install -m 0600 -o root -g root <src> <dst> — เป็น root ไม่ได้ในเทส จึงคัดลอกเฉย ๆ
        self._shim("install", 'cp "${@: -2:1}" "${@: -1}"\n')
        if python == "real":
            self._shim("python3", f'exec {shlex.quote(sys.executable)} "$@"\n')
        elif python == "no-yaml":
            # -S = ไม่โหลด site-packages → `import yaml` ล้มเหมือนเครื่องที่ไม่มี python3-yaml
            self._shim("python3", f'exec {shlex.quote(sys.executable)} -S "$@"\n')
        self.path = f"{self.bin}:/usr/bin:/bin"
        if python == "missing":
            # PATH ที่มีแต่ของที่สคริปต์ใช้ และไม่มี python3 เลย
            for tool in ("grep", "basename", "cp", "mv", "mkdir", "cat", "chmod", "rm", "mktemp"):
                (self.bin / tool).symlink_to(shutil.which(tool))
            self.path = str(self.bin)

    def _shim(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def box(self, name: str) -> Path:
        return self.root / name

    def etc(self, name: str) -> Path:
        return self.box(name) / "etc/netplan"

    def disabled(self, name: str) -> Path:
        return self.box(name) / "root/netplan-disabled"

    def machine(self, name: str, files: dict[str, str]) -> Node:
        self.etc(name).mkdir(parents=True)
        for filename, text in files.items():
            (self.etc(name) / filename).write_text(text, encoding="utf-8")
        return Node(name=name, host=f"10.2.1.{len(list(self.root.iterdir()))}", user="tkc")

    def files(self, name: str) -> list[str]:
        return sorted(p.name for p in self.etc(name).iterdir())

    def netplan_calls(self) -> list[str]:
        log = self.root / "netplan.calls"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def __call__(self, node, command, timeout=60, stdin_text=""):
        self.calls.append((node.name, command))
        if command.startswith("sudo -S -p '' bash -c "):
            command = shlex.split(command)[-1]
            stdin_text = ""
        elif command.startswith("sudo "):
            return SimpleNamespace(ok=True, exit_code=0, stdout="LMDS_SUDO_OK\n", stderr="")
        if command.startswith("ip -br"):
            cidr = self.cidrs[node.name]
            return SimpleNamespace(ok=True, exit_code=0, stderr="",
                                   stdout=f"{IFACE} UP {cidr}\n{IFACE} UP aa:bb <BROADCAST,MULTICAST,UP,LOWER_UP>\n")
        if "LMDS_UFW" in command:
            return SimpleNamespace(ok=True, exit_code=0, stdout="LMDS_UFW_INACTIVE\n", stderr="")
        if command.startswith("ping "):
            return SimpleNamespace(ok=True, exit_code=0, stdout="", stderr="")
        box = str(self.box(node.name))
        command = (command.replace("/etc/netplan", f"{box}/etc/netplan")
                   .replace("/root/netplan-disabled", f"{box}/root/netplan-disabled")
                   .replace("/tmp/lmds-netplan.", f"{box}/lmds-netplan."))
        (self.box(node.name) / "root").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["/bin/bash", "-c", command], input=stdin_text, capture_output=True, text=True, timeout=60,
                              env={"PATH": self.path, "FAKE_NODE": node.name,
                                   "FAKE_CALLS": str(self.root / "netplan.calls")})
        out = proc.stdout.replace(f"{box}/lmds-netplan.", "/tmp/lmds-netplan.").replace(box, "")
        return SimpleNamespace(ok=proc.returncode == 0, exit_code=proc.returncode, stdout=out, stderr=proc.stderr)

    cidrs: dict[str, str] = {}


def _plan(order: list[str]) -> dict:
    """แผนจริงจาก build_plan: สองเครื่อง สายที่พอร์ต 1 ยังไม่มี IP"""
    hosts = {name: spark(f"10.2.1.{i}") for i, name in enumerate(order, 1)}
    plan = build_plan(order, hosts)
    assert plan["ok"], plan
    return plan


def _apply(machines: Machines, nodes: dict[str, Node], order: list[str]) -> dict:
    plan = _plan(order)
    machines.cidrs = {name: f"{plan['nodes'][name]['cluster_ip']}/24" for name in order}
    return apply_plan(plan, {name: "pw" for name in order}, nodes=nodes, runner=machines, pair=False, speed_test=False,
                      update_registry=lambda *a, **k: None, sleep=lambda s: None, stamp=STAMP)


def _steps(report: dict, node: str) -> list[tuple[str, str]]:
    return [(s["step"], s["level"]) for s in report["steps"] if s["node"] == node]


# ═════════════════════ ไฟล์เดียวรวมสายบริหาร ═════════════════════
def test_a_file_holding_the_management_nic_is_not_moved_and_nothing_is_changed_on_any_machine(tmp_path):
    """รีโปรของ auditor + "ก่อนแตะเครื่องไหนเลย": เครื่องที่สะอาด (msi-2) อยู่ **ก่อน** เครื่องที่มีปัญหา (msi-1) ในลำดับ —
    preflight ต้องจบทั้งแผนก่อน ไม่ใช่เขียน msi-2 ไปแล้วค่อยมาเจอ msi-1"""
    machines = Machines(tmp_path)
    nodes = {"msi-2": machines.machine("msi-2", {"50-cloud-init.yaml": MGMT_ONLY, "60-cluster.yaml": CLUSTER_ONLY}),
             "msi-1": machines.machine("msi-1", {"50-cloud-init.yaml": SINGLE_FILE})}
    report = _apply(machines, nodes, ["msi-2", "msi-1"])

    assert report["ok"] is False and report["applied"] is False
    # ดิสก์: ไม่มีอะไรถูกย้าย/เขียนบนเครื่องไหนเลย · สายบริหารของ msi-1 ยังถูกตั้งค่าอยู่ที่เดิม · netplan ไม่ถูกเรียกสักครั้ง
    assert machines.files("msi-1") == ["50-cloud-init.yaml"]
    assert (machines.etc("msi-1") / "50-cloud-init.yaml").read_text(encoding="utf-8") == SINGLE_FILE
    assert machines.files("msi-2") == ["50-cloud-init.yaml", "60-cluster.yaml"]
    assert not machines.disabled("msi-1").exists() and not machines.disabled("msi-2").exists()
    assert machines.netplan_calls() == []
    assert not any(c.startswith("f=$(mktemp") for _, c in machines.calls), "ห้าม stage ไฟล์เมื่อ preflight ไม่ผ่าน"

    # ขั้นที่ล้มคือ preflight ของ msi-1 — ระดับ fail ไม่ใช่ pass พร้อมหมายเหตุ · บอกไฟล์ · interface อื่น · และต้องทำอะไร
    assert _steps(report, "msi-1") == [("sudo password accepted", "pass"), (PREFLIGHT_STEP, "fail")]
    assert _steps(report, "msi-2") == [("sudo password accepted", "pass"), (PREFLIGHT_STEP, "pass")]
    [failed] = [s for s in report["steps"] if not s["ok"]]
    said = failed["detail"]
    assert "/etc/netplan/50-cloud-init.yaml" in said and "eno1" in said and IFACE in said
    assert "Nothing was changed" in said and "sudo netplan apply" in said and "On msi-1" in said


def test_separate_files_still_apply_and_only_the_cluster_file_is_moved(tmp_path):
    """เลย์เอาต์ที่ docstring เดิมสมมติไว้ยังต้องทำงานเหมือนเดิม: ไฟล์ของคลัสเตอร์ถูกย้าย ไฟล์ของสายบริหารไม่ถูกแตะ"""
    machines = Machines(tmp_path)
    nodes = {name: machines.machine(name, {"50-cloud-init.yaml": MGMT_ONLY, "60-cluster.yaml": CLUSTER_ONLY})
             for name in ("msi-1", "msi-2")}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert report["applied"] and report["ok"], [s for s in report["steps"] if not s["ok"]]
    for name in ("msi-1", "msi-2"):
        assert machines.files(name) == ["50-cloud-init.yaml", "99-lmds-cluster.yaml"]
        assert (machines.etc(name) / "50-cloud-init.yaml").read_text(encoding="utf-8") == MGMT_ONLY
        assert [p.name for p in machines.disabled(name).iterdir()] == [f"60-cluster.yaml.{STAMP}"]
        assert _steps(report, name)[:5] == [
            ("sudo password accepted", "pass"), (PREFLIGHT_STEP, "pass"), ("stage netplan file", "pass"),
            (f"write {NETPLAN_FILE} + netplan apply", "pass"), ("verify addresses + carrier", "pass")]
    assert machines.netplan_calls() == ["msi-1 netplan generate", "msi-1 netplan apply",
                                        "msi-2 netplan generate", "msi-2 netplan apply"]
    # ทุก preflight มาก่อนการเขียนครั้งแรก
    first_write = next(i for i, (_, c) in enumerate(machines.calls) if c.startswith("f=$(mktemp"))
    preflights = [i for i, (_, c) in enumerate(machines.calls) if "LMDS_PREFLIGHT_DONE" in c]
    assert len(preflights) == 2 and max(preflights) < first_write
    moved = next(s for s in report["steps"] if s["node"] == "msi-1" and s["step"] == PREFLIGHT_STEP)
    assert moved["detail"] == "will move /etc/netplan/60-cluster.yaml"


def test_a_machine_with_no_conflicting_file_passes_the_preflight_and_moves_nothing(tmp_path):
    machines = Machines(tmp_path)
    nodes = {name: machines.machine(name, {"50-cloud-init.yaml": MGMT_ONLY}) for name in ("msi-1", "msi-2")}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert report["ok"], report["steps"]
    assert machines.files("msi-1") == ["50-cloud-init.yaml", "99-lmds-cluster.yaml"]
    assert list(machines.disabled("msi-1").iterdir()) == []
    step = next(s for s in report["steps"] if s["node"] == "msi-1" and s["step"] == PREFLIGHT_STEP)
    assert step["ok"] and IFACE in step["detail"] and "no other netplan file" in step["detail"]


def test_flow_style_and_quoted_keys_are_understood_not_grepped(tmp_path):
    """YAML แบบ flow / คีย์ใส่ quote — grep `^\\s+iface:` มองไม่เห็นเลย · parser จริงเห็น และเห็นว่ามี eno1 อยู่ด้วย"""
    flow = '{network: {version: 2, ethernets: {eno1: {dhcp4: true}, "%s": {dhcp4: false}}}}\n' % IFACE
    machines = Machines(tmp_path)
    nodes = {"msi-1": machines.machine("msi-1", {"01-all.yaml": flow}),
             "msi-2": machines.machine("msi-2", {"50-cloud-init.yaml": MGMT_ONLY})}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert not report["applied"] and machines.files("msi-1") == ["01-all.yaml"] and machines.netplan_calls() == []
    [failed] = [s for s in report["steps"] if not s["ok"]]
    assert failed["step"] == PREFLIGHT_STEP and "01-all.yaml" in failed["detail"] and "eno1" in failed["detail"]


@pytest.mark.parametrize("stanza", [
    "      dhcp4: true\n",
    "      addresses: [172.16.0.5/24]\n",
    "      dhcp4: false\n      routes:\n        - to: 10.9.0.0/16\n          via: 10.2.1.1\n",
    "      match:\n        macaddress: aa:bb:cc:dd:ee:ff\n      set-name: lan0\n",
])
def test_any_real_configuration_on_another_interface_blocks_the_move(tmp_path, stanza):
    text = f"network:\n  version: 2\n  ethernets:\n    eth9:\n{stanza}    {IFACE}:\n      dhcp4: false\n"
    machines = Machines(tmp_path)
    nodes = {"msi-1": machines.machine("msi-1", {"10-lan.yaml": text}),
             "msi-2": machines.machine("msi-2", {})}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert not report["applied"] and machines.files("msi-1") == ["10-lan.yaml"]
    [failed] = [s for s in report["steps"] if not s["ok"]]
    assert "eth9" in failed["detail"] and "10-lan.yaml" in failed["detail"]


def test_a_bond_or_vlan_in_the_same_file_blocks_the_move(tmp_path):
    text = (f"network:\n  version: 2\n  ethernets:\n    {IFACE}:\n      dhcp4: false\n"
            "  vlans:\n    vlan40:\n      id: 40\n      link: eno1\n      addresses: [10.40.0.5/24]\n")
    machines = Machines(tmp_path)
    nodes = {"msi-1": machines.machine("msi-1", {"10-lan.yaml": text}), "msi-2": machines.machine("msi-2", {})}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert not report["applied"] and machines.files("msi-1") == ["10-lan.yaml"]
    assert "vlan40" in next(s for s in report["steps"] if not s["ok"])["detail"]


# ═════════════════════ ข้อยกเว้นที่ย้ายได้ — และต้องบอกว่าอะไรไปด้วย ═════════════════════
def test_a_link_local_only_sibling_stanza_rides_along_and_is_reported(tmp_path):
    """ไฟล์ `40-cx7.yaml` ตามคู่มือ NVIDIA ประกาศทั้ง f0 และ f1 เป็น link-local — ไม่มี address/DHCP/route ให้เสีย
    ปฏิเสธไฟล์นี้ = ทุกเครื่องที่ตั้งตามคู่มือ NVIDIA มาก่อนใช้ `cluster apply` ไม่ได้ โดยไม่ได้ป้องกันอะไร"""
    machines = Machines(tmp_path)
    nodes = {name: machines.machine(name, {"01-mgmt.yaml": MGMT_ONLY, "40-cx7.yaml": NVIDIA_CX7})
             for name in ("msi-1", "msi-2")}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert report["ok"], [s for s in report["steps"] if not s["ok"]]
    assert machines.files("msi-1") == ["01-mgmt.yaml", "99-lmds-cluster.yaml"]
    assert [p.name for p in machines.disabled("msi-1").iterdir()] == [f"40-cx7.yaml.{STAMP}"]
    step = next(s for s in report["steps"] if s["node"] == "msi-1" and s["step"] == PREFLIGHT_STEP)
    assert step["detail"] == "will move /etc/netplan/40-cx7.yaml (also drops its stanza for enp1s0f0np0)"


def test_the_nvidia_sync_file_is_moved_whole_as_the_plan_announces(tmp_path):
    """ไฟล์ของ NVIDIA Sync มีแต่ลิงก์คลัสเตอร์โดยนิยาม และแผนประกาศไว้แล้วว่าจะย้าย — ย้ายได้แม้มีพอร์ตอื่นที่ตั้ง IP ไว้"""
    sync = (f"network:\n  version: 2\n  ethernets:\n    {IFACE}:\n      addresses: [192.168.100.10/24]\n"
            "    enP2p1s0f1np1:\n      addresses: [192.168.101.10/24]\n")
    machines = Machines(tmp_path)
    nodes = {name: machines.machine(name, {"01-mgmt.yaml": MGMT_ONLY, "99-nvidia-sync-cluster.yaml": sync})
             for name in ("msi-1", "msi-2")}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert report["ok"], [s for s in report["steps"] if not s["ok"]]
    assert machines.files("msi-1") == ["01-mgmt.yaml", "99-lmds-cluster.yaml"]
    step = next(s for s in report["steps"] if s["node"] == "msi-1" and s["step"] == PREFLIGHT_STEP)
    assert "99-nvidia-sync-cluster.yaml" in step["detail"] and "enP2p1s0f1np1" in step["detail"]


# ═════════════════════ ตรวจไม่ได้ ≠ ผ่าน ═════════════════════
def test_a_file_that_is_not_valid_yaml_but_mentions_the_interface_stops_the_apply(tmp_path):
    broken = f"network:\n  ethernets:\n    {IFACE}:\n      addresses: [10.0.0.1/24\n    eno1:\n\tdhcp4: true\n"
    machines = Machines(tmp_path)
    nodes = {"msi-1": machines.machine("msi-1", {"50-broken.yaml": broken}), "msi-2": machines.machine("msi-2", {})}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert not report["applied"] and machines.files("msi-1") == ["50-broken.yaml"] and machines.netplan_calls() == []
    failed = next(s for s in report["steps"] if not s["ok"])
    assert failed["step"] == PREFLIGHT_STEP and "50-broken.yaml" in failed["detail"] and "not valid YAML" in failed["detail"]


@pytest.mark.parametrize("python, reason", [("no-yaml", "python3-yaml is missing"), ("missing", "python3 is missing")])
def test_without_a_yaml_parser_the_single_file_layout_is_still_refused(tmp_path, python, reason):
    """ไม่มี python3-yaml / ไม่มี python3 = บอกไม่ได้ว่าไฟล์นั้นมีอะไรอีก → ปฏิเสธ ไม่ถอยไปย้ายแบบ grep เหมือนเดิม"""
    machines = Machines(tmp_path, python=python)
    nodes = {"msi-1": machines.machine("msi-1", {"50-cloud-init.yaml": SINGLE_FILE}),
             "msi-2": machines.machine("msi-2", {"50-cloud-init.yaml": MGMT_ONLY})}
    report = _apply(machines, nodes, ["msi-1", "msi-2"])
    assert not report["applied"] and machines.files("msi-1") == ["50-cloud-init.yaml"]
    assert machines.netplan_calls() == []
    failed = next(s for s in report["steps"] if not s["ok"])
    assert failed["node"] == "msi-1" and reason in failed["detail"] and "50-cloud-init.yaml" in failed["detail"]
    # เครื่องที่ไม่มีไฟล์ไหนเอ่ยถึง interface ของแผนยังผ่าน preflight ได้ แม้ไม่มี parser
    assert (PREFLIGHT_STEP, "pass") in _steps(report, "msi-2")


def test_a_preflight_that_did_not_finish_is_a_failure_not_a_pass(tmp_path):
    move, problems, _ = read_preflight("msi-1", "LMDS_PREFLIGHT_FILE\t/etc/netplan/60-x.yaml\tenp1s0f1np1\t\t\n", [IFACE])
    assert move == [] and problems and "could not inspect" in problems[0]
    # ชื่อไฟล์ที่มาจากเครื่องปลายทางถูกวางในคำสั่ง root — รูปที่ไม่รู้จักต้องไม่ถูกย้าย
    out = "LMDS_PREFLIGHT_FILE\t/etc/netplan/x; rm -rf ~.yaml\tenp1s0f1np1\t\t\nLMDS_PREFLIGHT_DONE\n"
    move, problems, _ = read_preflight("msi-1", out, [IFACE])
    assert move == [] and len(problems) == 1
    with pytest.raises(NetplanError):
        apply_script("/tmp/lmds-netplan.abc", ["/etc/netplan/x; rm -rf ~.yaml"], STAMP)
    with pytest.raises(NetplanError):
        apply_script("/tmp/lmds-netplan.abc", [NETPLAN_FILE], STAMP)


def test_rollback_after_a_later_failure_restores_the_moved_file(tmp_path):
    """ย้ายตามรายชื่อจาก preflight แล้ว verify ล้ม → rollback ต้องคืนไฟล์เดิมกลับที่ (stamp เดียวกัน)"""
    machines = Machines(tmp_path)
    nodes = {name: machines.machine(name, {"50-cloud-init.yaml": MGMT_ONLY, "60-cluster.yaml": CLUSTER_ONLY})
             for name in ("msi-1", "msi-2")}
    plan = _plan(["msi-1", "msi-2"])
    machines.cidrs = {"msi-1": "10.9.9.9/24", "msi-2": "10.9.9.8/24"}        # IP ไม่ขึ้นตามแผน → verify ล้ม
    report = apply_plan(plan, {"msi-1": "pw", "msi-2": "pw"}, nodes=nodes, runner=machines, pair=False,
                        speed_test=False, update_registry=lambda *a, **k: None, sleep=lambda s: None, stamp=STAMP)
    assert not report["applied"] and report["nodes"]["msi-1"]["rolled_back"] is True
    assert machines.files("msi-1") == ["50-cloud-init.yaml", "60-cluster.yaml"]
    assert (machines.etc("msi-1") / "60-cluster.yaml").read_text(encoding="utf-8") == CLUSTER_ONLY
    assert machines.files("msi-2") == ["50-cloud-init.yaml", "60-cluster.yaml"], "เครื่องที่สองยังไม่ถูกแตะ"


# ═════════════════════ บอกล่วงหน้าตั้งแต่ inspect / plan ═════════════════════
def _host_from_real_detection(tmp_path: Path, netplan: dict[str, str]) -> dict:
    """payload ของ node จาก detect_fabric จริง กับ /sys และ /etc/netplan ปลอม — สายบริหารคือ enP7s7 (10.2.1.195)"""
    from lmds.hardware.profiler import detect_fabric

    fabric = detect_fabric(**fake_sys(tmp_path, netplan=netplan, addresses={
        "enP7s7": "10.2.1.195/24", "enp1s0f0np0": "169.254.21.127/16", "enp1s0f1np1": "169.254.21.128/16"}))
    return {"hostname": "msi-1", "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128}], "fabric": fabric}


def test_plan_and_inspect_warn_ahead_when_the_inventory_already_shows_a_shared_file(tmp_path):
    single = SINGLE_FILE.replace("eno1", "enP7s7")
    hosts = {"msi-1": _host_from_real_detection(tmp_path / "one", {"50-cloud-init.yaml": single}),
             "msi-2": _host_from_real_detection(tmp_path / "two", {"50-cloud-init.yaml": MGMT_ONLY.replace("eno1", "enP7s7"),
                                                                    "60-cluster.yaml": CLUSTER_ONLY})}
    plan = build_plan(["msi-1", "msi-2"], hosts)
    assert plan["ok"]
    assert plan["nodes"]["msi-1"]["netplan_shared"] == [{
        "file": "50-cloud-init.yaml", "cluster_ifaces": [IFACE], "other_ifaces": ["enP7s7"],
        "other_ips": {"enP7s7": "10.2.1.195"}}]
    assert plan["nodes"]["msi-2"]["netplan_shared"] == []
    [warning] = [w for w in plan["warnings"] if "50-cloud-init.yaml" in w]
    assert warning.startswith("msi-1:") and "enP7s7 (10.2.1.195)" in warning and IFACE in warning
    assert "stop before changing anything" in warning

    view = inspect_nodes(["msi-1", "msi-2"], hosts)
    [shared] = view["nodes"]["msi-1"]["netplan_shared"]
    assert shared["file"] == "50-cloud-init.yaml" and shared["other_ifaces"] == ["enP7s7"] and shared["text"] == warning
    assert view["nodes"]["msi-2"]["netplan_shared"] == []


def test_the_preview_stays_quiet_for_link_local_siblings_and_unreadable_files(tmp_path):
    """สิ่งที่ preflight ยอมให้ย้าย (stanza link-local ของ function ข้าง ๆ) ต้องไม่ขึ้นเป็นคำเตือนล่วงหน้า · ไฟล์ที่ผู้ใช้ของ node
    อ่านไม่ได้ (0600 ของ root — ส่วนใหญ่ของจริง) = ไม่รู้ ไม่เตือนมั่ว (preflight ใต้ sudo เป็นคนตัดสิน)"""
    host = _host_from_real_detection(tmp_path / "one", {"40-cx7.yaml": NVIDIA_CX7})
    plan = build_plan(["msi-1", "msi-2"], {"msi-1": host, "msi-2": spark("10.2.1.2")})
    assert plan["nodes"]["msi-1"]["netplan_shared"] == [] and not [w for w in plan["warnings"] if "40-cx7" in w]
    assert spark("10.2.1.9")["fabric"]["netplan_unreadable"] and plan["nodes"]["msi-2"]["netplan_shared"] == []


def test_web_inspect_and_plan_carry_the_warning_and_apply_reports_the_preflight_line(tmp_path, monkeypatch):
    pytest.importorskip("fastapi", reason="ส่วนเว็บเป็น optional extra")
    from fastapi.testclient import TestClient

    import lmds.nodes
    from lmds.nodes.registry import add
    from lmds.web import create_app, state

    machines = Machines(tmp_path / "boxes")
    single = SINGLE_FILE.replace("eno1", "enP7s7")
    for name, files in (("msi-1", {"50-cloud-init.yaml": single}), ("msi-2", {"50-cloud-init.yaml": MGMT_ONLY})):
        node = machines.machine(name, files)
        add(Node(name=name, host=node.host, user="tkc"))
        host = _host_from_real_detection(tmp_path / f"sys-{name}", files)
        host["fabric"]["qsfp_ports"][0]["carrier"] = True
        state.STORE.set_node(name, {"host": host, "models": [], "summary": {"total": 0, "running": 0}})
    monkeypatch.setattr(lmds.nodes, "run", machines)
    monkeypatch.setattr("lmds.nodes.netplan.VERIFY_PAUSE_S", 0.0)
    web = TestClient(create_app())

    seen = web.post("/api/cluster/inspect", json={"nodes": ["msi-2", "msi-1"]}).json()
    assert seen["nodes"]["msi-1"]["netplan_shared"][0]["file"] == "50-cloud-init.yaml"
    assert "enP7s7 (10.2.1.195)" in seen["nodes"]["msi-1"]["netplan_shared"][0]["text"]
    plan = web.post("/api/cluster/plan", json={"nodes": ["msi-2", "msi-1"]}).json()
    assert plan["ok"] and any("50-cloud-init.yaml" in w and "enP7s7" in w for w in plan["warnings"])

    done = web.post("/api/cluster/apply", json={"plan": plan, "passwords": {"msi-1": "pw", "msi-2": "pw"},
                                                "wait": True, "speed_test": False, "pair": False}).json()
    assert done["result"]["ok"] is False and done["result"]["applied"] is False and done["job"]["exit_code"] == 1
    lines = web.get(f"/api/jobs/{done['job']['id']}").json()["output"].splitlines()
    assert any(l.startswith("[msi-2] netplan preflight: ok") for l in lines), lines
    [failed] = [l for l in lines if l.startswith("[msi-1] netplan preflight: failed")]
    assert "50-cloud-init.yaml" in failed and "enP7s7" in failed
    assert not any("pair SSH" in l for l in lines), "ขั้น preflight ต้องไม่ถูกอ่านเป็น pair SSH"
    assert machines.files("msi-1") == ["50-cloud-init.yaml"] and machines.files("msi-2") == ["50-cloud-init.yaml"]
    assert machines.netplan_calls() == []


# ═════════════════════ renderer ต่อ interface ═════════════════════
def test_the_generated_file_scopes_the_renderer_to_its_own_interfaces():
    """`renderer` ระดับบนสุดของไฟล์ 99-… เขียนทับ backend ของทั้งเครื่อง (scalar คีย์เดียวกัน ไฟล์หลังชนะ) — ต้องอยู่ใต้
    interface ของเราเท่านั้น เพื่อไม่เปลี่ยน DGX OS ที่ใช้ NetworkManager เป็น networkd ทั้งเครื่อง"""
    text = render_netplan([{"iface": IFACE, "ip": "10.100.152.1", "prefix": 24},
                           {"iface": "enP2p1s0f1np1", "ip": "10.100.153.1", "prefix": 24}])
    data = yaml.safe_load(text)
    assert set(data["network"]) == {"version", "ethernets"}, "ห้ามมี renderer ระดับ network:"
    assert data["network"]["ethernets"] == {
        IFACE: {"renderer": "networkd", "dhcp4": False, "addresses": ["10.100.152.1/24"], "optional": True},
        "enP2p1s0f1np1": {"renderer": "networkd", "dhcp4": False, "addresses": ["10.100.153.1/24"], "optional": True}}
