"""เปลี่ยน hostname ของ OS บนเครื่องปลายทางจาก hub — ผ่าน SSH + sudo ที่ใช้ครั้งเดียว

เคสจริง 2026-09-21: เครื่องถูกส่งออกไปอยู่กับลูกค้าแล้วพบว่า **ตั้ง hostname ซ้ำกัน** หลายเครื่อง ·
ทางเข้าเครื่องพวกนั้นเหลือ **คอนโซล LMDS (พอร์ต 8600) ทางเดียว** ไม่มี SSH ให้ใครเข้าไปพิมพ์
`hostnamectl` เอง — งานนี้จึงมีหน้าเว็บเป็น *ของหลัก* ไม่ใช่ของต่อยอดจาก CLI

**ไม่ใช่เรื่องความถูกต้องของตัวนับ**: `cluster.machine_identity()` ผูก hostname เข้ากับชุด IP บน
สายเร็วอยู่แล้ว เครื่องคนละตัวที่ชื่อซ้ำกันจึงถูกนับแยกถูกต้องมาตั้งแต่ต้น (เทสคุมไว้ที่
`tests/test_cluster.py::test_same_hostname_on_different_machines_still_counts_twice`) · ที่พังคือ
*คนอ่าน*: prompt บนเครื่อง, journal, ป้ายในคอนโซล และ `lmds node list` ขึ้นชื่อเดียวกันสองใบ

## เปลี่ยน "ชื่อในทะเบียน" หรือ "hostname ของ OS" — ที่นี่เปลี่ยนอย่างหลังอย่างเดียว

สองอย่างนี้เป็นคนละอย่างโดยตั้งใจ (docs/FLEET-MULTI-NODE.md ข้อ "ชื่อในทะเบียนไม่จำเป็นต้องตรงกับ
hostname") · ชื่อในทะเบียนคือ *ป้ายของ hub*: `lmds node run <ชื่อ>`, `data-node=` ทุกปุ่มบนหน้าเว็บ,
`ui.node_order`, คีย์ของ job, รายชื่อสมาชิกกลุ่ม stacked และ `CLUSTER_NODES` ใน cluster.env ที่เขียน
ไปแล้วบนเครื่อง · `registry.update()` ปฏิเสธ `name`/`host`/`user`/`port` ไว้ตั้งแต่ต้นด้วยเหตุผลนั้น
— การแอบเปลี่ยนป้ายนี้คือการเปลี่ยนตัวตนของแถวในทะเบียน ไม่ใช่การแก้ข้อมูล

ที่ลูกค้าเจอซ้ำกันคือ hostname ของ OS ล้วน ๆ (ทะเบียนกันชื่อซ้ำอยู่แล้วที่ `registry.add`) ฟีเจอร์นี้
จึงแตะเฉพาะเครื่อง: `hostnamectl set-hostname` + `/etc/hosts` · **ทะเบียนไม่มีฟิลด์ hostname ของ OS
ให้ต้องอัปเดตด้วยซ้ำ** — ค่าที่คอนโซลโชว์มาจาก `lmds agent info` ทุกครั้ง ผู้เรียกจึงแค่สั่งสำรวจใหม่

## อะไรตามมาบ้าง (ไล่จากโค้ดจริง ไม่ใช่เดา)

- **`/etc/hosts`** — Ubuntu/DGX OS แม็ป `127.0.1.1 <hostname>` ไว้ · ตั้งชื่อใหม่แล้วไม่แก้บรรทัดนี้
  = `sudo` รอบถัดไป resolve ตัวเองไม่เจอ ขึ้น "unable to resolve host" และช้าลงเป็นวินาที ·
  ที่นี่จึงเขียนสองไฟล์นั้น **ในคำสั่ง sudo เดียวกัน** ไม่แยกเป็นสองปุ่มให้ใครลืมกดปุ่มที่สอง
- **cluster.env / NCCL / controller** — ไม่ต้องแตะ: ทุกค่าที่ควบคุม stacked เป็น IP
  (`MASTER_IP`, `WORKER_IPS`, `TRANSPORT_IP_*`, `--master-addr`, `--node-rank`) · `CLUSTER_NODES`
  เป็นชื่อในทะเบียนและเป็นข้อมูลประกอบ ไม่มีสคริปต์ไหนอ่าน · `hostname` ในเทมเพลต controller คือ
  `hostname -I` (หา IP) ซึ่งไม่เปลี่ยนตามชื่อ
- **server.meta / MODEL_PROFILE.yaml / bundle.env / netplan** — ไม่มี hostname อยู่ในนั้นเลย
- **`~/.ssh/config` ของ head (คู่ stacked)** — `cluster_ssh.pair_workers` เขียน `Host` line จาก
  `worker.all_hosts` ด้วย · ถ้าที่อยู่นั้นเป็นชื่อที่กำลังจะเปลี่ยน stanza จะเหลือแต่ IP ที่ยังตรง
  (controller ต่อด้วย IP อยู่แล้ว) — เตือนให้ `lmds cluster pair` ซ้ำ ไม่ถึงกับบล็อก
- **`~/.lmds/bench/*.json` และ PROFILE.yaml ของ recipes** — จดชื่อ ณ วันที่วัด/วันที่ publish
  เป็นหลักฐานย้อนหลัง **ตั้งใจไม่แก้** ประวัติที่ถูกแก้ย้อนหลังไม่ใช่ประวัติ

## ที่ปฏิเสธ ไม่ใช่ทำครึ่งเดียว

- hub เข้าเครื่องนี้ด้วย *ชื่อที่กำลังจะเปลี่ยน* (`node add spark1.local`) → เปลี่ยนเสร็จก็เข้าไม่ได้อีก
  และเครื่องนี้ไม่มี SSH ให้ไปแก้คืน — เป็นกับดักที่เปิดแล้วปิดไม่ได้
- มีโมเดล **stacked** รันอยู่ → rendezvous ของ NCCL กับ `/etc/hosts` กำลังถูกใช้งานอยู่ และจาก hub
  ยืนยันไม่ได้ว่ากลุ่มยังดีหลังเปลี่ยน · หยุดที่ head ก่อนแล้วค่อยเปลี่ยน
- ชื่อใหม่ชนกับเครื่องอื่น → นี่คือเหตุผลทั้งหมดของฟีเจอร์นี้ ห้ามสร้างปัญหาเดิมซ้ำจากปุ่มที่มีไว้แก้มัน

ทุกขั้นวิ่งผ่าน `runner` (= `lmds.nodes.run`) เหมือน `netplan.apply_plan` จึงเทสได้ทั้งเส้นทางด้วย SSH
ปลอม · รหัส sudo เดินทางทาง stdin เท่านั้น ไม่อยู่ใน argv/log/ทะเบียน
"""

from __future__ import annotations

import re
import shlex
import time

from .cluster import machine_identity
from .registry import Node

# ยืมของ netplan มาใช้แทนที่จะเขียนใหม่ให้ "คล้าย ๆ": ขั้น sudo ของงานนี้ต้องเหมือน wizard เครือข่าย
# คลัสเตอร์ทั้งข้อความและพฤติกรรม — หน้าเว็บจับชื่อขั้นที่มีคำว่า "sudo password" · detail ถูกกรอง
# รหัสออกและตัดเหลือ 300 ตัวท้ายชุดเดียวกัน · สองทางที่เพี้ยนกันทีหลังคือสิ่งที่ไม่มีใครเห็นจนลูกค้าเจอ
from .netplan import _scrub, _sudo_ok, _unreachable, sudo_wrap

HOSTNAME_MAX = 63
# สำรองไว้ที่เดียวกับที่ netplan เก็บไฟล์ที่ถูกปลด — ถอยกลับด้วยมือได้เสมอแม้ hub จะเข้าไม่ถึงแล้ว
BACKUP_DIR = "/root/lmds-hostname"
# RFC 1123 label: a-z 0-9 '-' · ห้ามขึ้นต้น/ลงท้ายด้วย '-'
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


class HostnameError(Exception):
    """ผู้ใช้แก้ได้ — ข้อความภาษาอังกฤษเพราะหน้าเว็บโชว์ตรง ๆ (CLI แปะคำอธิบายไทยเอง)"""


def _q(text: str) -> str:
    return shlex.quote(text)


def normalise(value: str) -> str:
    """ตัดช่องว่างแล้วลดเป็นตัวพิมพ์เล็ก — DNS ไม่สนตัวพิมพ์ แต่คนอ่านจอสน

    ลูกค้าพิมพ์ "Spark-01" มาแล้วเด้งกลับว่า "ต้องเป็น a-z" คือการกีดกันที่ไม่มีเหตุผลทางเทคนิค ·
    แปลงให้แล้ว **คืนค่าที่แปลงแล้วออกไปทุกทาง** (API/CLI/หน้าเว็บ) เพื่อให้ผู้ใช้เห็นชื่อที่ตั้งจริง
    ไม่ใช่ชื่อที่พิมพ์ไป
    """
    return (value or "").strip().lower()


def validate(value: str) -> str:
    """ชื่อใหม่ตามกติกา RFC 1123 — คืนชื่อที่ normalise แล้ว · ไม่ผ่าน = HostnameError

    ตรวจที่จุดเดียวแล้ว CLI/หน้าเว็บ/สคริปต์ใช้ร่วมกัน · ชื่อนี้ถูกต่อเข้าไปใน `hostnamectl` และ
    `/etc/hosts` บนเครื่องลูกค้า จึงต้องปลอดภัยตั้งแต่ตอนรับเข้า ไม่ใช่หวังพึ่ง quote ปลายทาง
    (แนวเดียวกับ `registry.name_ok`)
    """
    name = normalise(value)
    if not name:
        raise HostnameError("enter the new hostname")
    if "." in name:
        raise HostnameError(
            f"'{name}' contains a dot — set the short name only, not an FQDN "
            f"(the domain part belongs in DNS or in /etc/hosts)")
    if len(name) > HOSTNAME_MAX:
        raise HostnameError(f"'{name}' is {len(name)} characters — a hostname is at most {HOSTNAME_MAX}")
    if not _LABEL.match(name):
        raise HostnameError(
            f"'{name}' is not a valid hostname — use a-z, 0-9 and '-', "
            f"and start and end with a letter or a digit")
    return name


def hostname_of(host: dict | None) -> str:
    """hostname ที่ payload ของเครื่องนั้นรายงาน — "" เมื่อยังไม่เคยสำรวจ"""
    return normalise(((host or {}).get("hostname") or ""))


def collisions(new: str, *, name: str, nodes: dict, hosts: dict, hub_hostname: str = "") -> list[dict]:
    """ชื่อใหม่ไปชนกับใครบ้าง — [] = ว่าง ใช้ได้ · คืนรหัส ไม่ใช่ประโยค

    นี่คือเหตุผลทั้งหมดของฟีเจอร์นี้: ตั้งซ้ำแล้วแก้ไม่ได้ · ปุ่มที่มีไว้แก้ปัญหานั้นต้องไม่สร้างมันซ้ำ

    "เครื่องเดียวกันที่ถูกลงทะเบียนไว้สองชื่อ" **ไม่ใช่การชน** — ใช้ `machine_identity()` ตัวเดียวกับที่
    `drop_duplicate_machines` ใช้ยุบฝาแฝดตัดออก ไม่งั้นเครื่องที่ถูก add ไว้สองชื่อจะเปลี่ยนชื่อไม่ได้เลย
    เพราะมันชนกับตัวเอง (ซึ่งเป็นสภาพที่ทะเบียนของจริงมีอยู่ — นั่นคือเหตุผลที่ฟังก์ชันนั้นมีอยู่)

    hub เองก็นับด้วย: hub ไม่มีแถวในทะเบียน ชื่อสมาชิกคลัสเตอร์ของมันคือ hostname ของ OS ตรง ๆ
    (`web/api.py` → `local_name`) ตั้งชื่อ node ให้ชนกับ hub คือทำให้สองใบในกลุ่มเดียวกันชื่อเหมือนกัน
    """
    new = normalise(new)
    me = machine_identity(hosts.get(name) or {})
    found: list[dict] = []
    for other in nodes:
        if other == name:
            continue
        host = hosts.get(other) or {}
        identity = machine_identity(host)
        if me and identity and identity == me:
            continue                      # เครื่องเดียวกัน คนละแถวในทะเบียน — ไม่ใช่การชน
        if hostname_of(host) == new:
            found.append({"kind": "hostname", "node": other})
        elif normalise(other) == new:
            found.append({"kind": "registry-name", "node": other})
    if hub_hostname and normalise(hub_hostname) == new:
        found.append({"kind": "hub", "node": normalise(hub_hostname)})
    return found


def twin_names(name: str, hosts: dict) -> list[str]:
    """แถวอื่นในทะเบียนที่ชี้ไปที่ **เครื่องเดียวกัน** กับ `name` — ปกติว่าง

    ทำไมผู้เรียกต้องสนใจ: `machine_identity()` = (hostname, ชุด IP บนสายเร็ว) · พอ hostname เปลี่ยน
    แถวที่ถูก probe ใหม่จะได้ identity ใหม่ ส่วนแถวฝาแฝดยังถือค่าเก่าอยู่จนกว่าจะถึงรอบของมัน (≤15 วิ)
    — ช่วงนั้น `drop_duplicate_machines` มองเป็นคนละเครื่อง แล้ว world size ของกลุ่มบวกเกินไปหนึ่ง
    ซึ่งเปลี่ยนแผน parallel ทั้งกลุ่ม · สั่งสำรวจพร้อมกันทุกแถวแล้วช่องนั้นก็ไม่เคยเปิด
    """
    me = machine_identity(hosts.get(name) or {})
    if not me:
        return []
    return [other for other in hosts
            if other != name and machine_identity(hosts.get(other) or {}) == me]


def collision_text(item: dict) -> str:
    """ประโยคอังกฤษของการชนหนึ่งข้อ — หน้าเว็บโชว์ตรง ๆ · CLI แปะคำอธิบายไทยรอบนอก"""
    if item["kind"] == "hostname":
        return f"'{item['node']}' already has that hostname — pick another name"
    if item["kind"] == "registry-name":
        return (f"'{item['node']}' is another machine in this registry — a hostname that matches "
                f"someone else's registry name makes the console read as two of the same machine")
    return f"the hub itself is called '{item['node']}' — pick another name"


def host_addresses_using(node: Node, old: str) -> list[str]:
    """ที่อยู่ที่ hub ใช้เข้าเครื่องนี้ตัวไหน "เป็นชื่อเดิม" บ้าง — [] = ไม่มี (เป็น IP หมด)

    `lmds node add 10.2.1.9` เก็บเป็น IP ก็ไม่กระทบ · แต่คนที่ add ด้วย `spark1` / `spark1.local`
    (mDNS) กำลังเข้าเครื่องผ่าน **ชื่อที่กำลังจะเปลี่ยน** — เปลี่ยนเสร็จ hub ก็เข้าไม่ได้อีก และเครื่อง
    ชุดนี้ไม่มี SSH ให้ไปแก้คืน · เทียบเฉพาะ label แรก (`spark1.local` → `spark1`) เพราะโดเมนต่อท้าย
    ไม่ได้เปลี่ยนตาม static hostname
    """
    old = normalise(old)
    if not old:
        return []
    return [a for a in node.all_hosts
            if normalise(a).split(".")[0] == old]


def running_models(models: list[dict] | None) -> tuple[list[str], list[str]]:
    """(slug ของ stacked ที่รันอยู่, slug ของตัวอื่นที่รันอยู่) จาก payload ที่ hub มีอยู่

    เงาของ worker ที่ `web/state.decorate_stacked` เติมให้ก็มี `topology: "stacked"` — กติกาเดียว
    จึงจับได้ทั้ง head และ worker โดยไม่ต้องรู้ว่าเครื่องนี้เป็นฝั่งไหนของคู่
    """
    stacked, other = [], []
    for model in models or []:
        if not model.get("running"):
            continue
        (stacked if model.get("topology") == "stacked" else other).append(model.get("slug") or "?")
    return stacked, other


def blockers(node: Node, old: str, new: str, *, nodes: dict, hosts: dict,
             models: list[dict] | None = None, hub_hostname: str = "") -> list[dict]:
    """เหตุที่ยัง **ไม่ควรเปลี่ยน** — [] = ไปต่อได้ · คืนรหัส + ประโยคอังกฤษพร้อมโชว์

    ปฏิเสธดีกว่าทำครึ่งเดียวแล้วทิ้งเครื่องไว้ในสภาพที่ไม่มีใครรู้ว่าเป็นอะไร — โดยเฉพาะเครื่องที่
    เหลือทางเข้าทางเดียวคือคอนโซลนี้
    """
    out: list[dict] = []
    for item in collisions(new, name=node.name, nodes=nodes, hosts=hosts, hub_hostname=hub_hostname):
        out.append({"kind": "taken", "text": collision_text(item)})
    # เฉพาะ *ที่อยู่หลัก* เท่านั้นที่เป็นเหตุห้าม — ที่อยู่สำรองที่ค้างเป็นแค่คำเตือน (ดู warnings_for)
    if normalise(old) and normalise(node.host).split(".")[0] == normalise(old):
        hint = f" — re-add it by IP first ({node.local_ip})" if node.local_ip else ""
        out.append({"kind": "ssh-address", "text": (
            f"the hub reaches this machine at '{node.host}', which is the name being changed — "
            f"after the rename there would be no way back in{hint}")})
    stacked, _ = running_models(models)
    if stacked:
        out.append({"kind": "stacked-running", "text": (
            f"{', '.join(sorted(stacked))} is running stacked across machines — stop it on the head "
            f"first; the hostname and /etc/hosts are in use by the live NCCL rendezvous and the hub "
            f"cannot verify the group from here")})
    return out


def warnings_for(node: Node, old: str, models: list[dict] | None = None) -> list[str]:
    """สิ่งที่ตามมาแต่ไม่ถึงกับห้าม — ผู้ใช้ควรเห็นก่อนกด ไม่ใช่ไปเจอทีหลัง"""
    out: list[str] = []
    _, other = running_models(models)
    if other:
        out.append(f"{', '.join(sorted(other))} is running — it keeps serving, the engines bind to "
                   f"addresses, not to the hostname")
    stale = [a for a in host_addresses_using(node, old) if a != node.host]
    if stale:
        out.append(f"the alternate address {', '.join(stale)} stops resolving after the rename — "
                   f"drop it with: lmds node set {node.name} --alt-host ...")
    if node.cluster_ip:
        out.append("this machine is paired for stacked serving — the head's ~/.ssh/config lists it by "
                   "name as well as by IP; re-run `lmds cluster pair` if head→worker ssh ever complains")
    return out


# ── สคริปต์ที่รันบนเครื่องนั้น ────────────────────────────────────────────────────
def read_script() -> str:
    """อ่านชื่อปัจจุบันแบบไม่ต้อง sudo — static (ไฟล์) กับ transient (kernel) ต่างกันได้จริง"""
    return ("printf 'LMDS_HN %s\\n' \"$(hostname 2>/dev/null)\"; "
            "printf 'LMDS_STATIC %s\\n' \"$(cat /etc/hostname 2>/dev/null)\"")


def rename_script(old: str, new: str, stamp: str) -> str:
    """สคริปต์ใต้ sudo: สำรอง → เขียน /etc/hosts ใหม่ (atomic) → ตั้ง hostname → พิมพ์ sentinel

    **ลำดับสำคัญ**: เตรียม `/etc/hosts` ให้เสร็จและตรวจครบใน temp file *ก่อน* แล้วค่อย `mv` ทับ
    (ไฟล์เดียวกัน filesystem เดียวกัน = atomic ไม่มีเสี้ยววินาทีที่ /etc/hosts ว่าง) จากนั้นค่อยตั้งชื่อ ·
    ถ้าตั้งชื่อล้มหลังจากนั้น `sudo` ยังใช้ได้ (มันแค่ *เตือน* ว่า resolve ชื่อเดิมไม่ได้ ไม่ได้ปฏิเสธ)
    rollback จึงยังวิ่งได้ — ซึ่งเป็นเหตุผลที่ไม่สลับลำดับ

    แก้เฉพาะบรรทัด `127.*` และเฉพาะคำที่เป็นชื่อเดิมจริง ๆ (`spark1` หรือ `spark1.local` →
    `new`/`new.local`) · บรรทัดอื่นในไฟล์ไม่ถูกแตะเลย เพราะ /etc/hosts ของลูกค้ามักมีแถวของระบบ
    อื่นปนอยู่ และไฟล์นี้คือสิ่งที่ sudo/ssh ทั้งเครื่องพึ่งพา
    """
    awk = (
        '$1 ~ /^127\\./ {'
        ' ch = 0;'
        ' for (i = 2; i <= NF; i++) {'
        '  if ($i == old) { $i = new; ch = 1 }'
        '  else if (index($i, old ".") == 1) { $i = new substr($i, length(old) + 1); ch = 1 }'
        ' }'
        ' if (ch) { print; next }'
        '}'
        '{ print }'
    )
    # ไม่มีบรรทัด 127.* ไหนเอ่ยถึงชื่อเดิมเลย (เครื่องที่ /etc/hosts ถูกแก้มือ) = ต้องเติมเอง
    # ไม่งั้นได้ hostname ใหม่ที่ resolve ตัวเองไม่ได้ ซึ่งคืออาการที่ฟีเจอร์นี้ตั้งใจจะไม่สร้าง
    present = ('$1 ~ /^127\\./ { for (i = 2; i <= NF; i++) if ($i == new) found = 1 }'
               ' END { exit found ? 0 : 1 }')
    return (
        "set -e; "
        f"old={_q(old)}; new={_q(new)}; d={_q(BACKUP_DIR)}; s={_q(stamp)}; "
        "mkdir -p \"$d\"; chmod 700 \"$d\"; "
        "cp -p /etc/hostname \"$d/hostname.$s\" 2>/dev/null || :; "
        "cp -p /etc/hosts \"$d/hosts.$s\"; "
        "t=$(mktemp /etc/hosts.lmds.XXXXXX); "
        f"awk -v old=\"$old\" -v new=\"$new\" {_q(awk)} /etc/hosts > \"$t\"; "
        f"awk -v new=\"$new\" {_q(present)} \"$t\" || printf '127.0.1.1\\t%s\\n' \"$new\" >> \"$t\"; "
        # 0:0 ไม่ใช่ root:root — เลข uid/gid มีอยู่ทุกระบบ ส่วน *ชื่อ* กลุ่มไม่เหมือนกันทุกที่
        # (mktemp ให้ไฟล์โหมด 600 มาด้วย ปล่อยไว้ = ทุก process ที่ไม่ใช่ root อ่าน /etc/hosts ไม่ได้)
        "chown 0:0 \"$t\"; chmod 644 \"$t\"; mv \"$t\" /etc/hosts; "
        "if command -v hostnamectl >/dev/null 2>&1; then hostnamectl set-hostname \"$new\"; "
        "else printf '%s\\n' \"$new\" > /etc/hostname; hostname \"$new\"; fi; "
        "echo LMDS_HOSTNAME_SET"
    )


def rollback_script(old: str, stamp: str) -> str:
    """คืนไฟล์ทั้งสองจากสำเนาของรอบที่มี stamp นี้ แล้วตั้งชื่อเดิมกลับ

    ไม่มี `set -e` โดยตั้งใจ (เหมือน `netplan.rollback_script`): ขั้นไหนคืนไม่ได้ก็ต้องลองขั้นที่เหลือ
    ต่อ — ถอยได้ครึ่งเดียวยังดีกว่าหยุดกลางคัน และผู้เรียกเห็นจาก sentinel ว่าจบครบไหม
    """
    return (
        f"old={_q(old)}; d={_q(BACKUP_DIR)}; s={_q(stamp)}; "
        "if [ -f \"$d/hosts.$s\" ]; then cp -p \"$d/hosts.$s\" /etc/hosts && echo 'restored /etc/hosts'; fi; "
        "if [ -f \"$d/hostname.$s\" ]; then cp -p \"$d/hostname.$s\" /etc/hostname && echo 'restored /etc/hostname'; fi; "
        "if command -v hostnamectl >/dev/null 2>&1; then hostnamectl set-hostname \"$old\"; "
        "else hostname \"$old\"; fi; "
        "echo LMDS_HOSTNAME_ROLLED_BACK"
    )


def verify_script(new: str) -> str:
    """ยืนยันโดยไม่ใช้ sudo: ชื่อที่ kernel ถือ · ไฟล์ static · และ **ชื่อใหม่ resolve ได้จริง**

    ข้อสุดท้ายคือข้อที่สำคัญที่สุด — `getent hosts` ว่างแปลว่า /etc/hosts ยังไม่พาชื่อใหม่ไปไหน
    ซึ่งคืออาการ "sudo ช้าลงมาก" ที่ทำให้เครื่องดูเหมือนป่วยโดยไม่มีอะไรตรงไหนบอก
    """
    return (
        "printf 'LMDS_HN %s\\n' \"$(hostname 2>/dev/null)\"; "
        "printf 'LMDS_STATIC %s\\n' \"$(cat /etc/hostname 2>/dev/null)\"; "
        f"printf 'LMDS_HOSTS %s\\n' \"$(getent hosts {_q(new)} 2>/dev/null | head -1)\""
    )


def _tagged(text: str, tag: str) -> str:
    for line in (text or "").splitlines():
        if line.startswith(tag + " "):
            return line[len(tag) + 1:].strip()
    return ""


def verify_problems(text: str, new: str) -> list[str]:
    """ปัญหาจากผล verify_script — [] = ครบทั้งสามข้อ"""
    problems = []
    if normalise(_tagged(text, "LMDS_HN")) != normalise(new):
        problems.append(f"the machine still calls itself '{_tagged(text, 'LMDS_HN') or '(nothing)'}'")
    if normalise(_tagged(text, "LMDS_STATIC")) != normalise(new):
        problems.append(f"/etc/hostname says '{_tagged(text, 'LMDS_STATIC') or '(empty)'}'")
    if not _tagged(text, "LMDS_HOSTS"):
        problems.append(f"'{new}' does not resolve — /etc/hosts would make sudo slow on every call")
    return problems


def preflight(node: Node, new: str = "", *, nodes: dict, hosts: dict,
              models: list[dict] | None = None, hub_hostname: str = "",
              sudo_needed: bool | None = None) -> dict:
    """สิ่งที่หน้าเว็บต้องรู้ *ก่อน* โชว์ฟอร์ม — ชื่อปัจจุบัน · ต้องกรอกรหัส sudo ไหม · ติดอะไรอยู่

    ใช้ชื่อปัจจุบันจากแคชของ hub (เร็ว ไม่ต้องรอ SSH ตอนกดปุ่ม) — เป็นข้อมูลบอกล่วงหน้าเท่านั้น
    `rename_host` อ่านจากเครื่องจริงอีกครั้งแล้วตรวจซ้ำทุกข้อ เพราะแคชเก่าได้และคนสองคนกดพร้อมกันได้

    `new` ว่าง = ยังไม่ได้พิมพ์ชื่อ (โชว์ฟอร์มเปล่า) — ตรวจเฉพาะสิ่งที่ไม่ขึ้นกับชื่อใหม่
    """
    current = hostname_of(hosts.get(node.name))
    out = {"node": node.name, "current": current, "new": "", "valid": False, "error": "",
           "blockers": [], "warnings": warnings_for(node, current, models),
           "sudo_needed": sudo_needed, "backup_dir": BACKUP_DIR}
    if not (new or "").strip():
        # ข้อห้ามที่ไม่ขึ้นกับชื่อใหม่ (stacked รันอยู่ · hub เข้าเครื่องนี้ด้วยชื่อเดิม) ต้องขึ้น
        # ตั้งแต่ก่อนผู้ใช้พิมพ์ ไม่ใช่ให้พิมพ์จบแล้วค่อยบอกว่าทำไม่ได้ตั้งแต่แรกอยู่แล้ว
        out["blockers"] = [b for b in blockers(node, current, "lmds-preflight", nodes=nodes,
                                               hosts=hosts, models=models, hub_hostname=hub_hostname)
                           if b["kind"] != "taken"]
        return out
    try:
        out["new"] = validate(new)
        out["valid"] = True
    except HostnameError as exc:
        out["error"] = str(exc)
        return out
    if current and out["new"] == current:
        out["error"] = f"'{current}' is already the name of this machine — nothing to change"
        return out
    out["blockers"] = blockers(node, current, out["new"], nodes=nodes, hosts=hosts,
                               models=models, hub_hostname=hub_hostname)
    return out


# ── ทำจริง ────────────────────────────────────────────────────────────────────
def rename_host(node: Node, new: str, password: str = "", *, nodes: dict | None = None,
                hosts: dict | None = None, models: list[dict] | None = None,
                hub_hostname: str = "", runner=None, progress=None, stamp: str = "") -> dict:
    """เปลี่ยน hostname ของ `node` เป็น `new` — คืน {"ok", "changed", "old", "new", "steps", …}

    ลำดับ: อ่านชื่อปัจจุบันจากเครื่องจริง (ไม่เชื่อแคช — แคชเก่าได้และนี่คือค่าที่ใช้ตัดสินทุกข้อ) →
    ตรวจเหตุห้ามทั้งหมด *ก่อน* แตะ sudo → ตรวจรหัส sudo → เขียนสองไฟล์ในคำสั่งเดียว → ยืนยัน →
    ล้ม = ถอยกลับด้วยสำเนาของรอบนี้

    `password` ว่าง = เครื่องนั้นต้องมี NOPASSWD (ตรวจด้วย `sudo -n` ตั้งแต่ก่อนเขียนอะไร) เหมือน
    `netplan.apply_plan` ทุกประการ · `runner` แทน `lmds.nodes.run` ในเทส
    """
    from . import ssh

    run = runner or ssh.run
    secrets = [password] if password else []
    nodes = {node.name: node} if nodes is None else nodes
    hosts = hosts or {}
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")

    steps: list[dict] = []
    report = {"ok": False, "changed": False, "node": node.name, "old": "", "new": "",
              "steps": steps, "rolled_back": False, "blockers": [], "warnings": [], "stamp": stamp}

    def step(what: str, ok: bool, detail: str = "", level: str = "") -> dict:
        item = {"node": node.name, "step": what, "ok": ok, "detail": _scrub(detail, secrets),
                "level": level or ("pass" if ok else "fail")}
        steps.append(item)
        if progress is not None:
            progress(item)
        return item

    # ชื่อผิดกติกาต้องล้มตั้งแต่ก่อนต่อเครื่อง — ไม่ใช่ไปล้มบนเครื่องลูกค้า
    new = validate(new)
    report["new"] = new

    read = run(node, read_script(), timeout=30)
    # เก็บทั้งตัวดิบและตัวที่ normalise แล้ว: การตัดสิน (ชน/ซ้ำ/ที่อยู่ SSH) ใช้ตัวที่ normalise แล้ว
    # แต่การ **แก้ /etc/hosts ต้องเทียบตัวดิบ** — เครื่องที่ชื่อ "DGX-Spark" มีคำนั้นอยู่ในไฟล์ตามตัวพิมพ์
    # เดิม ถ้าเอาตัวพิมพ์เล็กไปเทียบก็ไม่เจอ แล้วบรรทัดเก่าจะค้างอยู่ปนกับบรรทัดใหม่ที่ถูกเติมเข้าไป
    raw = _tagged(read.stdout or "", "LMDS_HN") or _tagged(read.stdout or "", "LMDS_STATIC")
    old = normalise(raw)
    if not read.ok or not old:
        step("read the current hostname", False,
             "unreachable" if _unreachable(read) else (read.stderr or read.stdout or "no answer"))
        return report
    report["old"] = old
    step("read the current hostname", True, old)

    if old == new:
        # กดซ้ำ/สองแท็บ/ลองใหม่หลังเน็ตหลุด — ตอบว่าเรียบร้อยแล้ว ไม่ใช่ error และไม่ต้องขอรหัส sudo
        step("nothing to change", True, f"already '{new}'")
        report["ok"] = True
        return report

    report["blockers"] = blockers(node, old, new, nodes=nodes, hosts=hosts, models=models,
                                  hub_hostname=hub_hostname)
    if report["blockers"]:
        for item in report["blockers"]:
            step("refused before touching the machine", False, item["text"])
        return report
    report["warnings"] = warnings_for(node, old, models)

    if not _sudo_ok(run, node, password, lambda _name, what, ok, detail="", level="": step(what, ok, detail, level)):
        return report

    applied = run(node, sudo_wrap(rename_script(raw, new, stamp)), timeout=120,
                  stdin_text=password + "\n")
    applied_ok = applied.ok and "LMDS_HOSTNAME_SET" in (applied.stdout or "")
    step(f"set the hostname to {new} + update /etc/hosts", applied_ok,
         f"previous files kept in {BACKUP_DIR} as *.{stamp}" if applied_ok
         else (applied.stderr or applied.stdout or "hostnamectl failed"))
    if not applied_ok:
        _rollback(node, raw, password, stamp, run, step, report)
        return report

    shown = run(node, verify_script(new), timeout=30)
    problems = (["the machine stopped answering right after the rename"] if _unreachable(shown)
                else verify_problems(shown.stdout or "", new))
    if problems:
        step("verify the new name resolves", False, "; ".join(problems))
        _rollback(node, raw, password, stamp, run, step, report)
        return report
    step("verify the new name resolves", True, f"{new} · /etc/hostname · {_tagged(shown.stdout or '', 'LMDS_HOSTS')}")
    report["ok"] = True
    report["changed"] = True
    return report


def _rollback(node: Node, old: str, password: str, stamp: str, run, step, report: dict) -> None:
    rolled = run(node, sudo_wrap(rollback_script(old, stamp)), timeout=120, stdin_text=password + "\n")
    ok = rolled.ok and "LMDS_HOSTNAME_ROLLED_BACK" in (rolled.stdout or "")
    report["rolled_back"] = ok
    step(f"roll back to {old}", ok,
         "" if ok else (rolled.stderr or rolled.stdout
                        or f"could not roll back — the previous files are in {BACKUP_DIR} as *.{stamp}"),
         level="warn" if ok else "fail")
