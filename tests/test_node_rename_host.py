"""เปลี่ยน hostname ของ OS บนเครื่องปลายทางจาก hub (2026-09-21)

เคสจริง: เครื่องถูกส่งออกไปอยู่กับลูกค้าแล้วพบว่า **ตั้ง hostname ซ้ำกัน** และเข้าเครื่องได้ทาง
คอนโซล LMDS (พอร์ต 8600) ทางเดียว — ไม่มี SSH ให้ใครเข้าไปพิมพ์ `hostnamectl` เอง

ชุดนี้คุม: กติกาชื่อ (RFC 1123) · การกันชื่อซ้ำซึ่งเป็นเหตุผลทั้งหมดของฟีเจอร์ (รวมถึงเคสที่
"ไม่ใช่การชน" คือเครื่องเดียวที่ถูกลงทะเบียนไว้สองชื่อ) · ลำดับคำสั่งบนเครื่องกับ SSH/sudo ปลอม ·
รหัสไม่โผล่ใน argv/log · การถอยกลับเมื่อล้มกลางคัน · เคสที่เลือก **ปฏิเสธ** แทนที่จะทำครึ่งเดียว ·
และรูป JSON ของ API — **ไม่แตะเครื่องจริงเลย**
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.nodes import Node, add, find  # noqa: E402
from lmds.nodes.hostname import (  # noqa: E402
    BACKUP_DIR,
    HostnameError,
    blockers,
    collisions,
    preflight,
    rename_host,
    validate,
    verify_problems,
)


# ── /etc/hosts + hostnamectl ปลอมของเครื่องเดียว ────────────────────────────────
class FakeBox:
    """`lmds.nodes.run` ปลอม: จำทุกคำสั่ง + stdin แล้วเล่นบท sudo / hostnamectl / getent

    เก็บสถานะจริง (ชื่อปัจจุบัน + เนื้อ /etc/hosts + สำเนาที่สำรองไว้) เพื่อให้ verify/rollback
    ตอบตามความจริงของเครื่องนั้น ไม่ใช่ตอบ "ผ่าน" ไปเรื่อย ๆ — ไม่งั้นเทสจะไม่จับเคสถอยกลับเลย
    """

    HOSTS = "127.0.0.1\tlocalhost\n127.0.1.1\t{name} {name}.local\n10.2.1.9\tother-box\n"

    def __init__(self, hostname="spark1", password="", *, set_fails=False, breaks_resolution=False,
                 unreachable_after_set=False):
        self.name = hostname
        self.password = password
        self.hosts = self.HOSTS.format(name=hostname)
        self.backups: dict[str, tuple[str, str]] = {}
        self.calls: list[tuple[str, str]] = []
        self.set_fails = set_fails
        self.breaks_resolution = breaks_resolution      # ตั้งชื่อสำเร็จแต่ /etc/hosts ไม่ตามไปด้วย
        self.unreachable_after_set = unreachable_after_set
        self.renamed = False

    def __call__(self, node, command, timeout=60, stdin_text=""):
        self.calls.append((command, stdin_text))
        code, out, err = self.answer(command, stdin_text)
        return SimpleNamespace(ok=code == 0, exit_code=code, stdout=out, stderr=err)

    @property
    def commands(self) -> list[str]:
        return [c for c, _ in self.calls]

    def _stamp(self, command: str) -> str:
        return command.split("s=")[1].split(";")[0].strip("'\" ")

    def answer(self, command, stdin):
        if command.startswith("printf 'LMDS_HN"):
            if self.renamed and self.unreachable_after_set:
                return 255, "", "ssh: connect to host 10.2.1.9 port 22: No route to host"
            out = f"LMDS_HN {self.name}\nLMDS_STATIC {self.name}\n"
            if "LMDS_HOSTS" in command:                 # verify_script — ถามด้วยว่าชื่อนั้น resolve ไหม
                wanted = command.split("getent hosts ")[1].split(" ")[0].strip("'")
                line = "" if wanted not in self.hosts else f"127.0.1.1       {wanted}"
                out += f"LMDS_HOSTS {line}\n"
            return 0, out, ""
        if command.startswith("sudo -n true"):
            return (0, "LMDS_SUDO_OK\n", "") if self.password == "" else (1, "", "sudo: a password is required")
        if command.startswith("sudo -S"):
            if stdin != self.password + "\n":
                return 1, "", "Sorry, try again."
            if command.endswith("-v && echo LMDS_SUDO_OK"):
                return 0, "LMDS_SUDO_OK\n", ""
            if "LMDS_HOSTNAME_SET" in command:
                stamp = self._stamp(command)
                old = command.split("old=")[1].split(";")[0].strip("'\" ")
                new = command.split("new=")[1].split(";")[0].strip("'\" ")
                self.backups[stamp] = (self.name, self.hosts)
                if self.set_fails:
                    return 1, "", "Could not set property: Connection timed out"
                self.renamed = True
                if not self.breaks_resolution:
                    self.hosts = self.hosts.replace(old, new)
                self.name = new
                return 0, "LMDS_HOSTNAME_SET\n", ""
            if "LMDS_HOSTNAME_ROLLED_BACK" in command:
                stamp = self._stamp(command)
                if stamp in self.backups:
                    self.name, self.hosts = self.backups[stamp]
                self.renamed = False
                return 0, "restored /etc/hosts\nLMDS_HOSTNAME_ROLLED_BACK\n", ""
            return 1, "", f"unexpected sudo command: {command}"
        return 0, "", ""


def spark(hostname: str, *, fabric_ip="10.100.152.1") -> dict:
    """host payload เท่าที่ machine_identity/hostname ต้องใช้ — hostname + ชุด IP บนสายเร็ว"""
    return {"hostname": hostname, "gpus": [{"name": "NVIDIA GB10"}],
            "fabric": {"links": [{"iface": "enp1s0f1np1", "ip": fabric_ip, "prefix": 24,
                                  "speed_gbps": 200, "connectx": True}]}}


@pytest.fixture
def one_box():
    add(Node(name="msi-6", host="10.2.1.9", user="nvidia", local_ip="10.2.1.9"))
    return find("msi-6")


# ── กติกาชื่อ ────────────────────────────────────────────────────────────────────
def test_the_new_name_must_be_a_real_hostname_and_uppercase_is_folded_not_rejected():
    """RFC 1123: a-z 0-9 '-' ไม่เกิน 63 ตัว ไม่ขึ้น/ลงท้ายด้วย '-'

    ตัวพิมพ์ใหญ่ *ไม่ใช่* ความผิด — DNS ไม่สนตัวพิมพ์ · ลูกค้าที่พิมพ์ตามป้ายบนเครื่อง ("Spark-01")
    ต้องไม่โดนเด้งกลับด้วยเหตุผลที่อธิบายไม่ได้ · แต่ต้องคืนชื่อที่แปลงแล้วออกไปให้เห็นว่าตั้งอะไรจริง
    """
    assert validate("  Spark-01 ") == "spark-01"
    assert validate("s") == "s" and validate("9") == "9"
    for bad, because in [("", "ว่าง"), ("-x", "ขึ้นต้นด้วย -"), ("x-", "ลงท้ายด้วย -"),
                         ("a_b", "ขีดล่างใช้ไม่ได้"), ("a b", "ช่องว่าง"), ("a" * 64, "เกิน 63")]:
        with pytest.raises(HostnameError):
            validate(bad)
            raise AssertionError(because)
    # FQDN เป็นคนละเรื่องกับ static hostname — บอกให้ตรงแทนที่จะพูดรวม ๆ ว่า "อักขระไม่ถูกต้อง"
    with pytest.raises(HostnameError, match="FQDN"):
        validate("spark1.example.com")


# ── กันชื่อซ้ำ — เหตุผลทั้งหมดของฟีเจอร์นี้ ─────────────────────────────────────────
def test_a_name_another_machine_already_answers_to_is_refused(one_box):
    add(Node(name="msi-5", host="10.2.1.8", user="nvidia"))
    hosts = {"msi-6": spark("spark1", fabric_ip="10.100.152.1"),
             "msi-5": spark("spark2", fabric_ip="10.100.152.2")}
    nodes = {"msi-6": find("msi-6"), "msi-5": find("msi-5")}

    hit = collisions("spark2", name="msi-6", nodes=nodes, hosts=hosts)
    assert [(h["kind"], h["node"]) for h in hit] == [("hostname", "msi-5")]
    # ชื่อในทะเบียนของเครื่องอื่นก็นับ — คอนโซลจะอ่านเป็นเครื่องเดียวกันสองใบทันที
    assert [h["kind"] for h in collisions("msi-5", name="msi-6", nodes=nodes, hosts=hosts)] == ["registry-name"]
    # hub ไม่มีแถวในทะเบียน แต่ชื่อสมาชิกคลัสเตอร์ของมันคือ hostname ของ OS ตรง ๆ
    assert [h["kind"] for h in collisions("controller", name="msi-6", nodes=nodes, hosts=hosts,
                                          hub_hostname="controller")] == ["hub"]
    assert collisions("spark-7", name="msi-6", nodes=nodes, hosts=hosts, hub_hostname="controller") == []


def test_the_same_machine_registered_under_two_names_is_not_a_collision(one_box):
    """ทะเบียนของจริงมีเครื่องเดียวที่ถูก add ไว้สองชื่อ (นั่นคือเหตุผลที่ drop_duplicate_machines มีอยู่)

    ถ้าไม่ตัดเคสนี้ออก เครื่องพวกนั้นจะเปลี่ยนชื่อไม่ได้เลยเพราะมันชนกับตัวเอง — ใช้ machine_identity
    ตัวเดียวกับที่ยุบฝาแฝดอยู่แล้ว (hostname + ชุด IP บนสายเร็ว) ไม่ใช่กติกาใหม่อีกชุด
    """
    add(Node(name="msi-6-tailscale", host="100.64.0.6", user="nvidia"))
    twin = spark("spark1", fabric_ip="10.100.152.1")
    hosts = {"msi-6": twin, "msi-6-tailscale": dict(twin)}
    nodes = {"msi-6": find("msi-6"), "msi-6-tailscale": find("msi-6-tailscale")}
    assert collisions("spark-7", name="msi-6", nodes=nodes, hosts=hosts) == []
    # …แต่เครื่องคนละตัวที่บังเอิญชื่อซ้ำกันอยู่ตอนนี้ ยังต้องกันตามปกติ (IP สายเร็วคนละชุด)
    hosts["msi-6-tailscale"] = spark("spark1", fabric_ip="10.100.152.9")
    assert [h["kind"] for h in collisions("spark1", name="msi-6", nodes=nodes, hosts=hosts)] == ["hostname"]


# ── เคสที่เลือกปฏิเสธ ─────────────────────────────────────────────────────────────
def test_it_refuses_when_the_hub_reaches_this_machine_by_the_very_name_being_changed():
    """`lmds node add spark1.local` = hub เข้าเครื่องนี้ผ่านชื่อที่กำลังจะเปลี่ยน

    เปลี่ยนเสร็จก็เข้าไม่ได้อีก และเครื่องชุดนี้ไม่มี SSH ให้ไปแก้คืน — กับดักที่เปิดแล้วปิดไม่ได้
    """
    add(Node(name="spark1", host="spark1.local", user="nvidia", local_ip="10.2.1.9"))
    node = find("spark1")
    found = blockers(node, "spark1", "spark-7", nodes={"spark1": node}, hosts={"spark1": spark("spark1")})
    assert [b["kind"] for b in found] == ["ssh-address"]
    assert "10.2.1.9" in found[0]["text"], "ต้องบอกทางออก (ที่อยู่ IP ที่ใช้แทนได้) ไม่ใช่แค่ปฏิเสธ"
    # add ด้วย IP = ไม่เกี่ยว เปลี่ยนได้
    add(Node(name="by-ip", host="10.2.1.8", user="nvidia"))
    assert blockers(find("by-ip"), "spark1", "spark-7",
                    nodes={"by-ip": find("by-ip")}, hosts={"by-ip": spark("spark1")}) == []


def test_it_refuses_while_a_stacked_model_is_running_on_either_side_of_the_pair(one_box):
    """stacked ที่รันอยู่ = rendezvous ของ NCCL กับ /etc/hosts กำลังถูกใช้งาน

    จาก hub ยืนยันไม่ได้ว่ากลุ่มยังดีหลังเปลี่ยน — ปฏิเสธดีกว่าทิ้งเครื่องไว้ในสภาพที่ไม่มีใครรู้
    · เงาของ worker ที่ decorate_stacked เติมให้ก็มี topology: stacked จึงจับได้ทั้งสองฝั่งด้วยกติกาเดียว
    """
    ctx = {"nodes": {"msi-6": one_box}, "hosts": {"msi-6": spark("spark1")}}
    head = [{"slug": "glm-4.6", "running": True, "topology": "stacked", "stacked_role": "head"}]
    worker = [{"slug": "glm-4.6", "running": True, "topology": "stacked", "stacked_role": "worker"}]
    for models in (head, worker):
        found = blockers(one_box, "spark1", "spark-7", models=models, **ctx)
        assert [b["kind"] for b in found] == ["stacked-running"]
        assert "glm-4.6" in found[0]["text"]
    # หยุดแล้ว = เปลี่ยนได้ · โมเดลเดี่ยวที่รันอยู่ก็ไม่ห้าม (engine ผูกกับที่อยู่ ไม่ใช่ชื่อ)
    assert blockers(one_box, "spark1", "spark-7", models=[{**head[0], "running": False}], **ctx) == []
    assert blockers(one_box, "spark1", "spark-7",
                    models=[{"slug": "qwen3-8b", "running": True, "topology": "single"}], **ctx) == []


# ── ลำดับคำสั่งบนเครื่องจริง ──────────────────────────────────────────────────────
def test_rename_sets_the_hostname_and_etc_hosts_in_one_sudo_call_and_keeps_the_password_out_of_argv(one_box):
    """สองไฟล์ต้องไปด้วยกัน — ตั้งชื่อแล้วปล่อย /etc/hosts ค้างคือสาเหตุที่ sudo ช้าลงเป็นวินาที

    และเป็นสิ่งที่ผู้ใช้จะไม่มีทางไปแก้เองได้ เพราะเข้าเครื่องได้ทางคอนโซลนี้ทางเดียว
    """
    box = FakeBox("spark1", password="s3cret")
    result = rename_host(one_box, "Spark-7", "s3cret", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box, stamp="20260921-101500")

    assert result["ok"] and result["changed"]
    assert (result["old"], result["new"]) == ("spark1", "spark-7"), "ชื่อที่คืนออกไปต้องเป็นตัวที่ตั้งจริง"
    assert box.name == "spark-7"
    assert "spark-7 spark-7.local" in box.hosts and "spark1" not in box.hosts
    assert "other-box" in box.hosts, "บรรทัดที่ไม่เกี่ยวกับชื่อเดิมห้ามถูกแตะ"

    sudo = [c for c in box.commands if c.startswith("sudo -S -p '' bash -c")]
    assert len(sudo) == 1, "hostnamectl กับ /etc/hosts ต้องอยู่ในคำสั่ง sudo เดียว ไม่ใช่สองปุ่มให้ลืมกดที่สอง"
    assert "hostnamectl set-hostname" in sudo[0] and "/etc/hosts" in sudo[0]
    assert BACKUP_DIR in sudo[0], "ต้องสำรองของเดิมไว้ให้ถอยกลับด้วยมือได้"

    # รหัสผ่านอยู่ใน stdin เท่านั้น — ไม่อยู่ใน argv (คนอื่นบนเครื่องอ่าน /proc ได้) และไม่อยู่ในรายงาน
    assert all("s3cret" not in command for command in box.commands)
    assert "s3cret" not in repr(result["steps"])
    # ลำดับ: อ่านชื่อเดิม → ตรวจรหัส → เขียน → ยืนยัน
    assert [s["step"] for s in result["steps"]] == [
        "read the current hostname", "sudo password accepted",
        "set the hostname to spark-7 + update /etc/hosts", "verify the new name resolves"]


def test_a_machine_whose_current_name_has_capitals_still_gets_its_etc_hosts_line_rewritten(one_box):
    """เครื่องที่ตั้งชื่อว่า "DGX-Spark" มีคำนั้นใน /etc/hosts ตามตัวพิมพ์เดิม

    ถ้าเอาตัวพิมพ์เล็กไปเทียบก็ไม่เจอ แล้วบรรทัดเก่าจะค้างปนกับบรรทัดใหม่ที่ถูกเติมเข้าไป —
    ไม่ถึงกับพัง (ชื่อใหม่ resolve ได้) แต่คือขยะที่ไม่มีใครรู้ว่ามาจากไหนในไฟล์ที่ทั้งเครื่องพึ่งพา
    """
    box = FakeBox("DGX-Spark", password="s3cret")
    result = rename_host(one_box, "spark-7", "s3cret", hosts={"msi-6": spark("dgx-spark")},
                         nodes={"msi-6": one_box}, runner=box)
    assert result["ok"] and result["old"] == "dgx-spark"
    assert "DGX-Spark" not in box.hosts and "spark-7 spark-7.local" in box.hosts


def test_a_machine_with_passwordless_sudo_is_never_asked_for_one(one_box):
    """เคสจริง msi-5: NOPASSWD อยู่แล้ว — บังคับกรอกรหัสที่ไม่มีคือทางตันบนหน้าเว็บ"""
    box = FakeBox("spark1", password="")
    result = rename_host(one_box, "spark-7", "", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box)
    assert result["ok"] and box.name == "spark-7"
    assert any(c.startswith("sudo -n true") for c in box.commands)
    assert not any(c.endswith("-v && echo LMDS_SUDO_OK") for c in box.commands)
    assert "sudo password not needed (passwordless sudo)" in [s["step"] for s in result["steps"]]


def test_the_wrong_sudo_password_stops_before_anything_on_the_machine_is_touched(one_box):
    box = FakeBox("spark1", password="s3cret")
    result = rename_host(one_box, "spark-7", "nope", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box)
    assert not result["ok"] and not result["changed"]
    assert box.name == "spark1" and "spark1" in box.hosts
    assert not any("LMDS_HOSTNAME_SET" in c for c in box.commands)


def test_running_it_twice_is_not_an_error_and_asks_for_no_password_the_second_time(one_box):
    """กดซ้ำ/สองแท็บ/ลองใหม่หลังเน็ตหลุด — ต้องตอบว่าเรียบร้อยแล้ว ไม่ใช่ล้มแล้วชวนให้ไปแก้อะไรต่อ"""
    box = FakeBox("spark-7", password="s3cret")
    result = rename_host(one_box, "spark-7", "", hosts={"msi-6": spark("spark-7")},
                         nodes={"msi-6": one_box}, runner=box)
    assert result["ok"] and result["changed"] is False
    assert [s["step"] for s in result["steps"]] == ["read the current hostname", "nothing to change"]
    assert not any(c.startswith("sudo") for c in box.commands)


# ── ถอยกลับ ──────────────────────────────────────────────────────────────────────
def test_a_failed_hostnamectl_rolls_the_machine_back_to_the_name_it_had(one_box):
    box = FakeBox("spark1", password="s3cret", set_fails=True)
    result = rename_host(one_box, "spark-7", "s3cret", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box)
    assert not result["ok"] and result["rolled_back"]
    assert box.name == "spark1" and "spark1 spark1.local" in box.hosts
    assert [s["step"] for s in result["steps"]][-1] == "roll back to spark1"


def test_a_name_that_does_not_resolve_afterwards_is_treated_as_a_failure_not_a_success(one_box):
    """อาการที่เงียบที่สุดของงานนี้: ชื่อเปลี่ยนแล้วแต่ /etc/hosts ไม่ตามไป

    เครื่องยังทำงานปกติ แค่ `sudo` ทุกครั้งช้าลงเป็นวินาทีและขึ้น "unable to resolve host" —
    ไม่มีอะไรตรงไหนบอกเลยว่าเกิดจากปุ่มนี้ · ต้องนับเป็นล้มและถอยกลับ ไม่ใช่ตอบว่าสำเร็จ
    """
    box = FakeBox("spark1", password="s3cret", breaks_resolution=True)
    result = rename_host(one_box, "spark-7", "s3cret", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box)
    assert not result["ok"] and result["rolled_back"] and box.name == "spark1"
    failed = next(s for s in result["steps"] if s["step"] == "verify the new name resolves")
    assert "does not resolve" in failed["detail"] and "sudo" in failed["detail"]


def test_losing_the_machine_right_after_the_rename_is_reported_not_silently_passed(one_box):
    box = FakeBox("spark1", password="s3cret", unreachable_after_set=True)
    result = rename_host(one_box, "spark-7", "s3cret", hosts={"msi-6": spark("spark1")},
                         nodes={"msi-6": one_box}, runner=box)
    assert not result["ok"]
    assert "stopped answering" in next(s for s in result["steps"] if not s["ok"])["detail"]


# ── สคริปต์จริงกับ bash จริง ─────────────────────────────────────────────────────
def _sandboxed(root: Path, script: str) -> str:
    """ชี้ /etc และ /root ของสคริปต์ไปที่โฟลเดอร์ชั่วคราว แล้วทำสิ่งที่ต้องเป็น root ให้เป็น no-op

    เทสฝั่ง SSH ปลอมพิสูจน์ได้แค่ "ลำดับคำสั่งถูก" — ไม่ได้พิสูจน์ว่าสคริปต์ที่ส่งไปรันได้จริง ·
    /etc/hosts คือไฟล์ที่ทั้งเครื่องพึ่งพา เขียนพังบนเครื่องลูกค้าที่เข้าได้ทางเดียวแปลว่าจบ
    """
    return (script.replace("/etc/hosts", f"{root}/etc/hosts")
                  .replace("/etc/hostname", f"{root}/etc/hostname")
                  .replace("/root/lmds-hostname", f"{root}/root/lmds-hostname")
                  # เครื่องที่ไม่มี systemd จะไปทางนี้อยู่แล้ว — บังคับให้เทสเดินสาขานั้นเสมอ
                  .replace("command -v hostnamectl >/dev/null 2>&1", "false")
                  .replace('hostname "$new"', "true").replace('hostname "$old"', "true")
                  .replace('chown 0:0 "$t"', "true"))


@pytest.mark.parametrize("hosts_before, expect", [
    # บรรทัดของ Ubuntu/DGX OS ตามปกติ — ต้องแก้ทั้งชื่อเปล่าและ <ชื่อ>.local · บรรทัดอื่นห้ามขยับ
    ("127.0.0.1\tlocalhost\n127.0.1.1\tDGX-Spark DGX-Spark.local\n10.2.1.9\tother\n",
     "127.0.1.1 spark-7 spark-7.local"),
    # /etc/hosts ที่ถูกแก้มือจนไม่มีบรรทัดของ hostname เลย — ถ้าไม่เติมให้ ชื่อใหม่จะ resolve ไม่ได้
    # แล้ว sudo จะช้าลงทุกครั้ง ซึ่งคืออาการที่ฟีเจอร์นี้ตั้งใจจะไม่สร้าง
    ("127.0.0.1\tlocalhost\n10.2.1.9\tother\n", "127.0.1.1\tspark-7"),
])
def test_the_script_really_runs_under_bash_and_can_be_undone(tmp_path, hosts_before, expect):
    import subprocess

    from lmds.nodes.hostname import rename_script, rollback_script

    (tmp_path / "etc").mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "etc/hosts").write_text(hosts_before)
    (tmp_path / "etc/hostname").write_text("DGX-Spark\n")

    done = subprocess.run(["bash", "-c", _sandboxed(tmp_path, rename_script("DGX-Spark", "spark-7", "S1"))],
                          capture_output=True, text=True)
    assert done.returncode == 0 and "LMDS_HOSTNAME_SET" in done.stdout, done.stderr
    after = (tmp_path / "etc/hosts").read_text()
    assert expect in after and "DGX-Spark" not in after
    assert "10.2.1.9\tother" in after, "บรรทัดที่ไม่เกี่ยวกับชื่อเดิมต้องไม่ถูกแตะแม้แต่ช่องว่าง"
    assert (tmp_path / "etc/hostname").read_text().strip() == "spark-7"

    back = subprocess.run(["bash", "-c", _sandboxed(tmp_path, rollback_script("DGX-Spark", "S1"))],
                          capture_output=True, text=True)
    assert back.returncode == 0 and "LMDS_HOSTNAME_ROLLED_BACK" in back.stdout, back.stderr
    assert (tmp_path / "etc/hosts").read_text() == hosts_before, "ถอยกลับต้องได้ไฟล์เดิมเป๊ะ"
    assert (tmp_path / "etc/hostname").read_text().strip() == "DGX-Spark"


def test_a_failure_halfway_leaves_etc_hosts_exactly_as_it_was(tmp_path):
    """`set -e` + เขียนลง temp file ก่อนแล้วค่อย mv — ล้มตรงไหนก็ตาม /etc/hosts ต้องไม่เปลี่ยน

    ไฟล์นี้คือสิ่งที่ sudo และ ssh ของทั้งเครื่องพึ่งพา · ครึ่ง ๆ กลาง ๆ บนเครื่องที่เข้าได้ทาง
    คอนโซลนี้ทางเดียวแปลว่าไม่มีใครเข้าไปแก้ได้อีกเลย
    """
    import subprocess

    from lmds.nodes.hostname import rename_script

    (tmp_path / "etc").mkdir()
    (tmp_path / "root").mkdir()
    before = "127.0.0.1\tlocalhost\n127.0.1.1\tspark1\n"
    (tmp_path / "etc/hosts").write_text(before)
    (tmp_path / "etc/hostname").write_text("spark1\n")
    # ทำให้ awk ตัวแรกล้ม (เหมือนดิสก์เต็ม/ไฟล์อ่านไม่ได้) แล้วดูว่าไฟล์จริงยังเหมือนเดิมไหม
    broken = _sandboxed(tmp_path, rename_script("spark1", "spark-7", "S1")).replace("awk -v old=", "false -v old=")
    out = subprocess.run(["bash", "-c", broken], capture_output=True, text=True)
    assert out.returncode != 0 and "LMDS_HOSTNAME_SET" not in out.stdout
    assert (tmp_path / "etc/hosts").read_text() == before
    assert (tmp_path / "etc/hostname").read_text().strip() == "spark1"
    # สำเนาถูกทำไว้ *ก่อน* แตะอะไร — ถอยกลับด้วยมือได้เสมอ แม้ขั้นต่อไปจะไม่เคยเริ่ม
    assert sorted(p.name for p in (tmp_path / "root/lmds-hostname").iterdir()) == ["hostname.S1", "hosts.S1"]


def test_the_sudo_wrapper_round_trips_the_script_without_mangling_the_quoting():
    """สคริปต์นี้มีทั้ง single quote (โปรแกรม awk) และ `$` — ถูกห่อด้วย shlex สองชั้น

    quote พังแปลว่าคำสั่งที่รันบนเครื่องลูกค้าไม่ใช่คำสั่งที่เราเขียน · และรหัสผ่านต้องไม่อยู่ใน argv
    """
    import shlex

    from lmds.nodes.hostname import rename_script
    from lmds.nodes.netplan import sudo_wrap

    script = rename_script("DGX-Spark", "spark-7", "S1")
    parts = shlex.split(sudo_wrap(script))
    assert parts[:6] == ["sudo", "-S", "-p", "", "bash", "-c"]
    assert parts[6] == script and len(parts) == 7


def test_verify_checks_all_three_facts_not_just_the_name():
    ok = "LMDS_HN spark-7\nLMDS_STATIC spark-7\nLMDS_HOSTS 127.0.1.1       spark-7\n"
    assert verify_problems(ok, "spark-7") == []
    assert verify_problems("LMDS_HN spark1\nLMDS_STATIC spark-7\nLMDS_HOSTS x\n", "spark-7")
    assert verify_problems("LMDS_HN spark-7\nLMDS_STATIC spark1\nLMDS_HOSTS x\n", "spark-7")
    assert verify_problems("LMDS_HN spark-7\nLMDS_STATIC spark-7\nLMDS_HOSTS \n", "spark-7")


# ── preflight ที่หน้าเว็บใช้ก่อนโชว์ฟอร์ม ────────────────────────────────────────
def test_preflight_tells_the_form_what_it_needs_before_the_user_types_anything(one_box):
    ctx = {"nodes": {"msi-6": one_box}, "hosts": {"msi-6": spark("spark1")}}
    empty = preflight(one_box, "", sudo_needed=True, **ctx)
    assert empty["current"] == "spark1" and empty["valid"] is False and empty["blockers"] == []
    assert empty["sudo_needed"] is True and empty["backup_dir"] == BACKUP_DIR

    assert preflight(one_box, "Spark-7", **ctx)["new"] == "spark-7"
    assert "FQDN" in preflight(one_box, "a.b", **ctx)["error"]
    assert "already the name" in preflight(one_box, "spark1", **ctx)["error"]
    # ข้อห้ามที่ไม่ขึ้นกับชื่อใหม่ต้องขึ้นตั้งแต่ก่อนพิมพ์ ไม่ใช่รอให้พิมพ์จบแล้วค่อยบอก
    busy = preflight(one_box, "", models=[{"slug": "glm-4.6", "running": True, "topology": "stacked"}], **ctx)
    assert [b["kind"] for b in busy["blockers"]] == ["stacked-running"]


def test_warnings_name_the_things_that_go_stale_without_blocking(one_box):
    add(Node(name="w", host="10.2.1.7", user="nvidia", alt_hosts=["spark1.tailnet.ts.net", "spark1"],
             cluster_ip="10.100.152.2"))
    out = preflight(find("w"), "", nodes={"w": find("w")}, hosts={"w": spark("spark1")},
                    models=[{"slug": "qwen3-8b", "running": True, "topology": "single"}])
    joined = " · ".join(out["warnings"])
    assert out["blockers"] == [], "ที่อยู่สำรองและคู่ stacked ที่หยุดอยู่ไม่ใช่เหตุให้ห้าม"
    assert "qwen3-8b" in joined and "spark1" in joined and "cluster pair" in joined


# ── API ที่หน้าเว็บเรียก ──────────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient

    from lmds.web import create_app

    return TestClient(create_app())


def test_the_web_endpoint_renames_and_forces_a_fresh_probe_so_the_label_stops_lying(one_box, monkeypatch):
    from lmds.web import state

    state.STORE.set_node("msi-6", {"host": spark("spark1"), "models": []})
    box = FakeBox("spark1", password="s3cret")
    monkeypatch.setattr("lmds.nodes.run", box)
    forced: list[str] = []
    monkeypatch.setattr(state.STORE, "force", lambda name: forced.append(name))

    body = _client().post("/api/nodes/msi-6/rename-host",
                          json={"hostname": "SPARK-7", "password": "s3cret"}).json()
    assert body["ok"] and body["changed"] and body["new"] == "spark-7"
    assert box.name == "spark-7"
    assert forced == ["msi-6"], "ไม่สั่งสำรวจใหม่ = ป้ายบนจอค้างชื่อเก่าแล้วดูเหมือนกดไม่ติด"


def test_the_twin_row_of_the_same_machine_is_re_probed_too(one_box, monkeypatch):
    """เครื่องเดียวที่ถูก add ไว้สองชื่อ: ถ้าสำรวจใหม่แค่แถวเดียว อีกแถวจะถือ hostname เก่าอยู่ ≤15 วิ

    ช่วงนั้น machine_identity ของสองแถวไม่ตรงกัน → drop_duplicate_machines ไม่ยุบ → world size
    ของกลุ่มบวกเกินไปหนึ่ง ซึ่งเปลี่ยนแผน parallel ทั้งกลุ่ม (2 เครื่อง TP=2 กลายเป็น 3 ต้อง pipeline)
    """
    from lmds.web import state

    add(Node(name="msi-6-vpn", host="100.64.0.6", user="nvidia"))
    twin = spark("spark1")
    state.STORE.set_node("msi-6", {"host": twin, "models": []})
    state.STORE.set_node("msi-6-vpn", {"host": dict(twin), "models": []})
    monkeypatch.setattr("lmds.nodes.run", FakeBox("spark1", password="s3cret"))
    forced: list[str] = []
    monkeypatch.setattr(state.STORE, "force", lambda name: forced.append(name))

    body = _client().post("/api/nodes/msi-6/rename-host",
                          json={"hostname": "spark-7", "password": "s3cret"}).json()
    assert body["ok"] and sorted(forced) == ["msi-6", "msi-6-vpn"]


def test_the_web_endpoint_refuses_a_name_another_machine_already_uses(one_box, monkeypatch):
    from lmds.web import state

    add(Node(name="msi-5", host="10.2.1.8", user="nvidia"))
    state.STORE.set_node("msi-6", {"host": spark("spark1", fabric_ip="10.100.152.1"), "models": []})
    state.STORE.set_node("msi-5", {"host": spark("spark2", fabric_ip="10.100.152.2"), "models": []})
    box = FakeBox("spark1", password="s3cret")
    monkeypatch.setattr("lmds.nodes.run", box)

    body = _client().post("/api/nodes/msi-6/rename-host",
                          json={"hostname": "spark2", "password": "s3cret"}).json()
    assert body["ok"] is False and [b["kind"] for b in body["blockers"]] == ["taken"]
    assert "msi-5" in body["blockers"][0]["text"]
    assert box.name == "spark1" and not any("LMDS_HOSTNAME_SET" in c for c in box.commands)


def test_a_bad_name_is_a_400_with_the_reason_not_a_bare_500(one_box, monkeypatch):
    from lmds.web import state

    state.STORE.set_node("msi-6", {"host": spark("spark1"), "models": []})
    monkeypatch.setattr("lmds.nodes.run", FakeBox("spark1", password="s3cret"))
    r = _client().post("/api/nodes/msi-6/rename-host", json={"hostname": "-nope-", "password": "x"})
    assert r.status_code == 400 and "hostname" in r.json()["detail"]
    assert _client().post("/api/nodes/ghost/rename-host", json={"hostname": "a"}).status_code == 404


def test_the_cli_asks_before_it_touches_anything_and_never_echoes_the_password(one_box, monkeypatch):
    """CLI ยังต้องมี (ทุกอย่างในระบบนี้มีทั้งสองทาง) — แต่กติกาเดียวกับหน้าเว็บทุกข้อ

    ต้องสำรวจฟลีตก่อนเพราะต้องรู้ว่า hostname ไหนถูกใช้ไปแล้ว · และต้องบอกให้ชัดว่า
    "ชื่อในทะเบียนยังเหมือนเดิม" ไม่งั้นคนใช้จะคิดว่า `lmds node run msi-6` เปลี่ยนไปด้วย
    """
    from typer.testing import CliRunner

    from lmds.cli.main import app

    box = FakeBox("spark1", password="s3cret")
    monkeypatch.setattr("lmds.nodes.run", box)
    monkeypatch.setattr("lmds.nodes.probe", lambda node, timeout=30: {"host": spark("spark1"), "models": []})
    monkeypatch.setattr("lmds.cli.main._live_cluster_groups",
                        lambda names=None: ({"msi-6": one_box}, {"msi-6": spark("spark1")}, {}, []))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "s3cret")

    out = CliRunner().invoke(app, ["node", "rename-host", "msi-6", "SPARK-7", "--yes"],
                             env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert box.name == "spark-7"
    assert "s3cret" not in out.output
    assert "spark-7" in out.output and "msi-6" in out.output
    assert "ทะเบียน" in out.output, "ต้องบอกว่าชื่อในทะเบียนไม่ได้เปลี่ยนไปด้วย"


def test_the_cli_refuses_a_bad_name_before_it_connects_to_anything(one_box, monkeypatch):
    from typer.testing import CliRunner

    from lmds.cli.main import app

    def never(*a, **k):
        raise AssertionError("ต้องไม่ต่อเครื่องเลยเมื่อชื่อผิดกติกาตั้งแต่ต้น")

    monkeypatch.setattr("lmds.cli.main._live_cluster_groups", never)
    out = CliRunner().invoke(app, ["node", "rename-host", "msi-6", "spark1.example.com"])
    assert out.exit_code == 1 and "FQDN" in out.output


def test_the_preflight_endpoint_answers_without_touching_the_machine(one_box, monkeypatch):
    from lmds.web import state

    state.STORE.set_node("msi-6", {"host": spark("spark1"), "models": []})
    box = FakeBox("spark1", password="")
    monkeypatch.setattr("lmds.nodes.run", box)
    body = _client().get("/api/nodes/msi-6/rename-host?hostname=spark-7").json()
    assert body["current"] == "spark1" and body["new"] == "spark-7" and body["valid"] is True
    assert body["sudo_needed"] is False and body["blockers"] == []
    assert not any("LMDS_HOSTNAME_SET" in c for c in box.commands), "preflight ต้องไม่เขียนอะไรเลย"
