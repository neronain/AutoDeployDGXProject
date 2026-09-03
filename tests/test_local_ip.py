"""IP ของ "เครื่องนั้น" ต้องมองเห็นได้ — ทะเบียนรู้จัก node จากที่อยู่ที่ใช้ SSH เท่านั้น

ที่อยู่ SSH เป็นชื่อได้ (`orb`, `spark1.local`, ชื่อบนเครือข่าย Tailscale) และเป็นมุมมองของ
hub เครื่องเดียว · คำถามที่ตอบไม่ได้เลยก่อนหน้านี้คือ "เครื่องนี้อยู่ IP ไหนในวง"
ซึ่งเป็นคำถามแรกทั้งตอนจะยิง API เข้าไปและตอนจะตั้ง cluster IP
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.hardware import profiler

runner = CliRunner()

# `ip -o -4 addr show` ของ Ubuntu ที่มีทั้งสาย LAN, สาย ConnectX, docker0 และ veth
IP_CMD_OUTPUT = """\
1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: eth0    inet 10.2.1.70/24 brd 10.2.1.255 scope global eth0\\       valid_lft forever
3: enp1s0f0np0    inet 169.254.10.2/16 brd 169.254.255.255 scope global enp1s0f0np0\\    valid_lft forever
4: enp1s0f1np1    inet 10.10.0.2/24 brd 10.10.0.255 scope global enp1s0f1np1\\       valid_lft forever
5: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\       valid_lft forever
6: veth9a1b2c3    inet 172.18.0.1/16 scope global veth9a1b2c3\\       valid_lft forever
"""

# macOS/OrbStack — ไม่มี iproute2 เลย และ netmask มาเป็นเลขฐานสิบหก
IFCONFIG_MACOS = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether aa:bb:cc:dd:ee:ff
\tinet 192.168.50.145 netmask 0xffffff00 broadcast 192.168.50.255
utun5: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 100.78.9.105 --> 100.78.9.105 netmask 0xffffffff
"""

# net-tools รุ่นเก่าเขียนคนละสำเนียง — `inet addr:` กับ `Mask:`
IFCONFIG_NET_TOOLS = """\
eth0      Link encap:Ethernet  HWaddr aa:bb:cc:dd:ee:ff
          inet addr:10.2.1.70  Bcast:10.2.1.255  Mask:255.255.255.0
lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
"""


@pytest.fixture
def no_ip_command(monkeypatch):
    """เครื่องที่ไม่มี iproute2 — `_run` คืน None เหมือนคำสั่งไม่มีอยู่"""
    def fake_run(cmd, timeout=None):
        return None
    monkeypatch.setattr(profiler, "_run", fake_run)


def _fake_run(mapping):
    def run(cmd, timeout=None):
        return mapping.get(cmd[0])
    return run


# ── อ่านที่อยู่ ──────────────────────────────────────────────────────────────

def test_reads_every_real_ipv4_and_drops_the_virtual_ones(monkeypatch):
    monkeypatch.setattr(profiler, "_run", _fake_run({"ip": IP_CMD_OUTPUT}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    found = profiler.local_addresses()

    assert [a["ip"] for a in found] == ["10.2.1.70", "10.10.0.2", "169.254.10.2"]
    # docker0/veth ยิงจากเครื่องอื่นไม่ถึง — โชว์ปนไว้คือชวนให้ก๊อปผิดเส้น
    assert not any(a["iface"].startswith(("docker", "veth")) for a in found)
    assert all(a["ip"] != "127.0.0.1" for a in found)


def test_the_default_route_comes_first_and_link_local_last(monkeypatch):
    monkeypatch.setattr(profiler, "_run", _fake_run({"ip": IP_CMD_OUTPUT}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    found = profiler.local_addresses()

    assert found[0]["primary"] is True and found[0]["iface"] == "eth0"
    assert found[0]["prefix"] == 24
    # 169.254.x.x = ลิงก์ขึ้นแต่ยังไม่ได้ตั้งค่า — ไม่ควรเป็นตัวแรกที่ตาไปเจอ
    assert found[-1]["link_local"] is True
    assert [a["link_local"] for a in found] == [False, False, True]


def test_falls_back_to_ifconfig_when_iproute2_is_missing(monkeypatch):
    """เครื่องทดสอบบน macOS/OrbStack ไม่มีคำสั่ง `ip` — เดิมจึงรายงานว่า "ตรวจไม่ได้" """
    monkeypatch.setattr(profiler, "_run", _fake_run({"ifconfig": IFCONFIG_MACOS}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "192.168.50.145")

    found = profiler.local_addresses()

    assert [(a["iface"], a["ip"], a["prefix"]) for a in found] == [
        ("en0", "192.168.50.145", 24),
        ("utun5", "100.78.9.105", 32),
    ]


def test_ifconfig_of_old_net_tools_is_understood_too(monkeypatch):
    monkeypatch.setattr(profiler, "_run", _fake_run({"ifconfig": IFCONFIG_NET_TOOLS}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    found = profiler.local_addresses()

    assert [(a["iface"], a["ip"], a["prefix"]) for a in found] == [("eth0", "10.2.1.70", 24)]


def test_netmask_reads_both_hex_and_dotted():
    assert profiler._netmask_bits("0xffffff00") == 24
    assert profiler._netmask_bits("255.255.255.0") == 24
    assert profiler._netmask_bits("0xffffffff") == 32
    assert profiler._netmask_bits("ไม่ใช่ netmask") is None
    assert profiler._netmask_bits("") is None


def test_still_reports_the_egress_ip_when_no_interface_can_be_read(no_ip_command, monkeypatch):
    """อ่านรายชื่อการ์ดไม่ได้ ต้องไม่แปลว่า "เครื่องนี้ไม่มี IP" — `lmds info` ตอบค่านี้มาตลอด"""
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    found = profiler.local_addresses()

    assert [(a["ip"], a["primary"]) for a in found] == [("10.2.1.70", True)]


def test_empty_means_nothing_was_found_not_loopback(no_ip_command, monkeypatch):
    monkeypatch.setattr(profiler, "primary_ip", lambda: "127.0.0.1")
    assert profiler.local_addresses() == []


def test_agent_info_carries_every_address(monkeypatch):
    """`lmds agent info` คือช่องทางเดียวที่ hub ใช้อ่านสถานะเครื่องอื่น"""
    from lmds.inventory import host_payload

    monkeypatch.setattr(profiler, "_run", _fake_run({"ip": IP_CMD_OUTPUT}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    payload = host_payload()

    assert payload["ip"] == "10.2.1.70"
    assert [a["ip"] for a in payload["ips"]] == ["10.2.1.70", "10.10.0.2", "169.254.10.2"]


# ── ทะเบียนจำ IP ล่าสุดไว้ ───────────────────────────────────────────────────

def test_registry_remembers_the_ip_the_node_reported():
    from lmds.nodes import Node, add, find

    add(Node(name="spark2", host="orb", user="ops", local_ip="10.2.1.70"))

    assert find("spark2").local_ip == "10.2.1.70"


def test_status_from_probe_picks_up_version_and_ip():
    from lmds.nodes import status_from_probe

    assert status_from_probe({"host": {"lmds_version": "0.5.0", "ip": "10.2.1.70"}}) == {
        "lmds_version": "0.5.0", "local_ip": "10.2.1.70"}


def test_a_node_that_reports_no_ip_does_not_erase_what_we_knew():
    """คีย์ที่ไม่ได้ส่งมา = "ไม่รู้" ไม่ใช่ "ไม่มี" — node รุ่นเก่าไม่ควรลบของที่เคยรู้จริง"""
    from lmds.nodes import Node, add, find, status_from_probe, update

    add(Node(name="old", host="10.0.0.9", user="ops", local_ip="10.0.0.9"))
    update("old", **status_from_probe({"host": {"lmds_version": "0.1.0"}}))

    assert find("old").local_ip == "10.0.0.9"


def test_registry_written_before_this_field_existed_still_loads(monkeypatch, tmp_path):
    from lmds.nodes import load, nodes_file

    nodes_file().parent.mkdir(parents=True, exist_ok=True)
    nodes_file().write_text(
        "nodes:\n- name: spark1\n  host: 10.2.1.70\n  user: ops\n  port: 22\n",
        encoding="utf-8")

    (node,) = load()
    assert (node.name, node.local_ip) == ("spark1", "")


# ── หน้าจอ ───────────────────────────────────────────────────────────────────

def test_node_list_shows_the_ip_of_the_machine_itself():
    from lmds.nodes import Node, add

    add(Node(name="spark2", host="orb", user="ops", local_ip="10.2.1.70"))

    result = runner.invoke(app, ["node", "list"])

    assert result.exit_code == 0
    # ที่อยู่ SSH เป็นชื่อ — IP จริงต้องมาจากคอลัมน์ใหม่ ไม่ใช่จากคอลัมน์ปลายทาง
    assert "10.2.1.70" in result.output


def test_ps_lists_every_address_when_the_machine_has_more_than_one(monkeypatch):
    monkeypatch.setattr(profiler, "_run", _fake_run({"ip": IP_CMD_OUTPUT}))
    monkeypatch.setattr(profiler, "primary_ip", lambda: "10.2.1.70")

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert "10.10.0.2" in result.output and "enp1s0f1np1" in result.output


def test_console_serves_the_ip_of_each_node():
    from fastapi.testclient import TestClient

    from lmds.nodes import Node, add
    from lmds.web.api import create_app

    add(Node(name="spark2", host="orb", user="ops", local_ip="10.2.1.70"))

    (row,) = TestClient(create_app()).get("/api/nodes").json()["nodes"]
    assert (row["host"], row["local_ip"]) == ("orb", "10.2.1.70")


def test_console_paints_the_ip_on_the_machine_card():
    """หน้าเว็บเป็นไฟล์เดียว — ป้าย IP ต้องมีทั้งที่ว่างในการ์ดและตัวที่เขียนค่าลงไป"""
    page = Path("src/lmds/web/static/index.html").read_text(encoding="utf-8")

    assert 'class="nip"' in page
    assert "function paintNodeIp(" in page
    # ค่าจากทะเบียนขึ้นตั้งแต่การ์ดโผล่ ไม่ต้องรอ SSH รอบใหม่
    assert "node.local_ip" in page
    # และการ์ด System ของทั้งเครื่องนี้/เครื่องอื่นแจกแจงทุกเส้น
    assert "function addressRow(" in page
