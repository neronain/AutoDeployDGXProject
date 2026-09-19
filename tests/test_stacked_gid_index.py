"""RoCEv2 GID index ของ stacked controller ต้องมาจาก sysfs ของแต่ละเครื่อง ไม่ใช่เลขฝังตาย

เดิม `NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"` แล้วเลข 3 ถูก push ให้ทุก rank ·
field notes §6 (docs/DGX-SPARK-VLLM-FIELD-NOTES.md) บอกไว้ชัดว่าเลขนี้ดริฟท์ได้หลัง
อัปเฟิร์มแวร์หรือรีบูต แล้ว NCCL init จะค้างนิ่ง ๆ โดยไม่มี diagnostic ให้ไล่ —
เป็นอาการที่แพงที่สุดแบบหนึ่งเพราะ "ก็ไม่มี error สักบรรทัด"

เทสทุกข้อดึงฟังก์ชันจาก controller ที่ render แล้ว **มารันจริง** ใต้ `set -Eeuo pipefail`
กับ sysfs ปลอมใน tmp_path (bash -n จับ unbound variable / ตรรกะที่ตกม้าตายตอนรันไม่ได้)
"""

from __future__ import annotations

import pathlib
import subprocess
import textwrap

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from tests.test_review_templates import extract_fn

SAFE_PATH = "/usr/bin:/bin"

# ── เครื่องจริงในสนาม: DGX Spark สองพอร์ต QSFP · RoCE สองตัวชื่อ rocep… / roceP2p… ──
HCA1, IFACE1, IP1 = "rocep1s0f0", "enp1s0f0np0", "10.100.0.10"
HCA2, IFACE2, IP2 = "roceP2p1s0f0", "enP2p1s0f0np0", "10.100.1.10"
OTHER_IP = "10.100.0.20"          # IP ของ "เครื่องอื่น" บนวงเดียวกัน


def _gid_for(ip: str) -> str:
    """GID แบบ IPv4-mapped ที่ ConnectX เขียนไว้ใน sysfs (10.100.0.10 → …:ffff:0a64:000a)"""
    a, b, c, d = (int(x) for x in ip.split("."))
    return f"0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}"


def _link_local_gid() -> str:
    """GID ตัวแรก ๆ ของพอร์ต — link-local ไม่ใช่ IPv4-mapped จึงต้องไม่ถูกเลือกไม่ว่ากรณีใด"""
    return "fe80:0000:0000:0000:0a64:00ff:fe00:000a"


# ───────────────────────── bundle + sysfs ปลอม ─────────────────────────
def _stacked_controller(tmp_path) -> str:
    report = ModelReport(
        repo_id="nvidia/DeepSeek-V4-Flash-NVFP4",
        revision_sha="sha-gid",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=168 * GIB,
        shard_count=46,
        context_length=131072,
        kv_dims=KvDims(layers=61, kv_heads=128, head_dim=128),
        license="mit",
        has_chat_template=True,
    )
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles")
    return pathlib.Path(bundle.controller).read_text(encoding="utf-8")


def _gids(root: pathlib.Path, hca: str, gids: dict[int, tuple[str, str]], port: str = "1") -> None:
    """GID ของพอร์ตหนึ่งใน sysfs ปลอม — gids = {index: (gid, type)} ตรงตามเลย์เอาต์ของไดรเวอร์จริง"""
    gid_dir = root / hca / "ports" / port / "gids"
    type_dir = root / hca / "ports" / port / "gid_attrs" / "types"
    gid_dir.mkdir(parents=True, exist_ok=True)
    type_dir.mkdir(parents=True, exist_ok=True)
    for index, (gid, gid_type) in gids.items():
        (gid_dir / str(index)).write_text(gid + "\n", encoding="utf-8")
        (type_dir / str(index)).write_text(gid_type + "\n", encoding="utf-8")


def _device(root: pathlib.Path, hca: str, iface: str, gids: dict[int, tuple[str, str]], port: str = "1") -> None:
    """RoCE หนึ่งตัว: ผูกกับ interface แล้วใส่ GID ให้"""
    (root / hca / "device" / "net" / iface).mkdir(parents=True, exist_ok=True)
    _gids(root, hca, gids, port)


def _normal_port(ip: str, roce_v2_at: int) -> dict[int, tuple[str, str]]:
    """พอร์ตที่ตั้ง RoCEv2 ถูกต้อง: link-local สองตัว แล้ว IPv4-mapped คู่ v1/v2"""
    return {
        0: (_link_local_gid(), "IB/RoCE v1"),
        1: (_link_local_gid(), "RoCE v2"),
        roce_v2_at - 1: (_gid_for(ip), "IB/RoCE v1"),
        roce_v2_at: (_gid_for(ip), "RoCE v2"),
    }


def _ip_shim(bin_dir: pathlib.Path, table: dict[str, str]) -> None:
    """`ip -o -4 addr show dev <iface>` ปลอม — รูปแบบบรรทัดตรงกับของจริง (field 4 = ip/prefix)

    ตารางที่อยู่เขียนลงไฟล์ข้าง ๆ ไม่ฝังในสคริปต์ จะได้ไม่ต้องไล่ quote ซ้อนกันสองชั้น
    """
    (bin_dir / "addrs.txt").write_text(
        "".join(f"{i + 2}: {iface}  inet {ip}/24 brd 0.0.0.0 scope global {iface}\n"
                for i, (iface, ip) in enumerate(table.items())),
        encoding="utf-8",
    )
    shim = bin_dir / "ip"
    shim.write_text(textwrap.dedent("""\
        #!/bin/bash
        dev=""
        while (( $# )); do case "$1" in dev) dev="$2"; shift 2 ;; *) shift ;; esac; done
        table="$(dirname "$0")/addrs.txt"
        if [[ -n "$dev" ]]; then awk -v d="$dev" '$2 == d' "$table"; else cat "$table"; fi
        exit 0
        """), encoding="utf-8")
    shim.chmod(0o755)


def _harness(text: str, *names: str) -> str:
    """ฟังก์ชันจริงจาก controller + ตัวช่วยที่มันพึ่ง (die/log ของจริง ไม่ใช่ stub)"""
    return (
        "set -Eeuo pipefail\n"
        + "".join(extract_fn(text, name) for name in ("die", "log", *names))
    )


def _run(script: str, tmp_path: pathlib.Path, env: dict | None = None) -> subprocess.CompletedProcess:
    path = tmp_path / "gid_harness.sh"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(["bash", str(path)], env={"PATH": SAFE_PATH, **(env or {})},
                          capture_output=True, text=True, timeout=30)


# ═════════════════ 1. เลข 3 ต้องหายไปจาก controller ที่ render แล้ว ═════════════════
def test_the_rendered_controller_no_longer_carries_a_hardcoded_gid_default(tmp_path):
    """literal 3 เป็น default ของ GID = บั๊กตัวเดิม · ต้องไม่มีเหลือในไฟล์ที่ผู้ใช้รัน"""
    text = _stacked_controller(tmp_path)
    assert "NCCL_IB_GID_INDEX:-3" not in text
    assert 'NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"' in text
    # ไม่มีบรรทัดไหน "ตั้งค่า" GID เป็นเลขคงที่ (บรรทัดตัวอย่างในข้อความ error เป็น <idx> ไม่ใช่ตัวเลข)
    for line in text.splitlines():
        assert "NCCL_IB_GID_INDEX=3" not in line, line
    # และมีตัวหาแทนที่ พร้อมทางลัดของผู้ใช้แบบเดียวกับ HCA
    assert "detect_gid_index_for_device()" in text
    assert 'NCCL_IB_GID_INDEX_EXPLICIT="$NCCL_IB_GID_INDEX"' in text


def test_the_controller_still_parses_and_wires_the_gid_per_rank(tmp_path):
    """worker ต้องได้ค่าของตัวเอง (arg ที่สามของ _nccl_env_pairs) ไม่ใช่ของ head"""
    text = _stacked_controller(tmp_path)
    path = tmp_path / "ctl.sh"
    path.write_text(text, encoding="utf-8")
    done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert '_nccl_env_pairs "$wifname" "$whca" "$wgid"' in text
    assert '_resolve_gid_index_for_node "worker rank ${rank} (${wip})" "$wip" "$whca"' in text
    assert "  _resolve_gid_index\n" in text


# ═════════════════ 2. detect_gid_index_for_device — ตัวหาเลขจริง ═════════════════
def _detect(tmp_path, text, hca, ip, devices) -> subprocess.CompletedProcess:
    root = tmp_path / "infiniband"
    for dev_hca, (dev_iface, gids) in devices.items():
        _device(root, dev_hca, dev_iface, gids)
    script = (
        _harness(text, "_ipv4_to_gid_suffix", "detect_gid_index_for_device")
        + f'IB_SYSFS_ROOT={root}\n'
        + f'if out="$(detect_gid_index_for_device {hca} {ip})"; then echo "FOUND=$out"; else echo "NOTFOUND=$?"; fi\n'
    )
    return _run(script, tmp_path)


def test_it_finds_the_roce_v2_index_that_matches_this_nodes_own_ip(tmp_path):
    text = _stacked_controller(tmp_path)
    done = _detect(tmp_path, text, HCA1, IP1, {HCA1: (IFACE1, _normal_port(IP1, 3))})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FOUND=3" in done.stdout, done.stdout + done.stderr
    assert "unbound variable" not in done.stderr, done.stderr


def test_it_finds_the_index_when_it_is_not_3(tmp_path):
    """เลขที่เจอในสนามหลังอัปเฟิร์มแวร์มักไม่ใช่ 3 — ตัวที่ฝัง 3 ไว้จะพังตรงนี้พอดี"""
    text = _stacked_controller(tmp_path)
    done = _detect(tmp_path, text, HCA2, IP2, {HCA2: (IFACE2, _normal_port(IP2, 7))})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FOUND=7" in done.stdout, done.stdout + done.stderr


def test_it_rejects_an_index_whose_type_is_roce_v1(tmp_path):
    """IP ตรงแต่เป็น RoCE v1 — ใช้ไม่ได้ เพราะ v1 ไม่ routable และ NCCL คนละโหมด"""
    text = _stacked_controller(tmp_path)
    only_v1 = {0: (_link_local_gid(), "IB/RoCE v1"), 2: (_gid_for(IP1), "IB/RoCE v1")}
    done = _detect(tmp_path, text, HCA1, IP1, {HCA1: (IFACE1, only_v1)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOTFOUND=1" in done.stdout, done.stdout + done.stderr


def test_it_rejects_a_gid_that_encodes_another_nodes_ip(tmp_path):
    """แมตช์แค่ *ffff* (แบบที่ launcher ของ repo อ้างอิงทำ) จะคว้า index ของเครื่องอื่นมาเงียบ ๆ"""
    text = _stacked_controller(tmp_path)
    someone_else = {
        0: (_link_local_gid(), "IB/RoCE v1"),
        2: (_gid_for(OTHER_IP), "IB/RoCE v1"),
        3: (_gid_for(OTHER_IP), "RoCE v2"),
    }
    done = _detect(tmp_path, text, HCA1, IP1, {HCA1: (IFACE1, someone_else)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOTFOUND=1" in done.stdout, done.stdout + done.stderr


# ═════════════════ 3. ระดับเครื่อง — หาไม่เจอต้องตายดัง ไม่ใช่ตกไป 3 ═════════════════
def _resolve_node(tmp_path, text, hcas, devices, addrs, extra="") -> subprocess.CompletedProcess:
    root = tmp_path / "infiniband"
    for dev_hca, (dev_iface, gids) in devices.items():
        _device(root, dev_hca, dev_iface, gids)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _ip_shim(bin_dir, addrs)
    script = (
        _harness(text, "_ipv4_to_gid_suffix", "detect_gid_index_for_device",
                 "_gid_scan_script", "_resolve_gid_index_for_node")
        + f'IB_SYSFS_ROOT={root}\nSSH_USER=neronain\nSLUG=deepseek-v4-flash\n{extra}'
        + f'_resolve_gid_index_for_node "head (10.100.0.10)" "" "{hcas}"\n'
        + 'echo "RESOLVED=$GID_INDEX_RESOLVED"\n'
    )
    return _run(script, tmp_path, env={"PATH": f"{bin_dir}:{SAFE_PATH}"})


def test_a_node_with_a_healthy_port_resolves_its_own_index(tmp_path):
    text = _stacked_controller(tmp_path)
    done = _resolve_node(tmp_path, text, HCA1,
                         {HCA1: (IFACE1, _normal_port(IP1, 5))}, {IFACE1: IP1})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "RESOLVED=5" in done.stdout, done.stdout + done.stderr


def test_both_twins_are_resolved_not_just_the_first(tmp_path):
    """repo อ้างอิง probe แค่ HCA ตัวแรกแล้วเอาไปใช้กับทั้งคู่ — ของเราต้องดูครบทุกตัว"""
    text = _stacked_controller(tmp_path)
    done = _resolve_node(tmp_path, text, f"{HCA1},{HCA2}",
                         {HCA1: (IFACE1, _normal_port(IP1, 5)), HCA2: (IFACE2, _normal_port(IP2, 5))},
                         {IFACE1: IP1, IFACE2: IP2})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "RESOLVED=5" in done.stdout, done.stdout + done.stderr


def test_nothing_matches_dies_loudly_and_names_the_device(tmp_path):
    """หัวใจของงานนี้: หาไม่เจอ = ตาย ไม่ใช่ fallback 3 แล้วไปค้างที่ NCCL init"""
    text = _stacked_controller(tmp_path)
    nothing = {0: (_link_local_gid(), "IB/RoCE v1"), 1: (_link_local_gid(), "RoCE v2")}
    done = _resolve_node(tmp_path, text, HCA1, {HCA1: (IFACE1, nothing)}, {IFACE1: IP1})
    combined = done.stdout + done.stderr
    assert done.returncode != 0, combined
    assert "RESOLVED=" not in done.stdout, combined
    assert HCA1 in combined, combined              # ชื่อ device
    assert IFACE1 in combined, combined            # ชื่อสาย
    assert IP1 in combined, combined               # IP ที่มันไปหา
    assert "0a64:000a" in combined, combined       # suffix ที่ควรจะเจอ
    assert "/sys/class/infiniband/" in combined, combined   # คำสั่งที่ให้ไปตรวจต่อ
    assert "NCCL_IB_GID_INDEX=<idx>" in combined, combined  # ทางออกถ้าจะยืนยันเลขเอง
    # ต้องไม่มีท่า "ลองใช้ 3 ดูก่อน" หลงเหลืออยู่ในทางออกที่เสนอให้ผู้ใช้
    assert "NCCL_IB_GID_INDEX=3" not in combined, combined


def test_a_device_the_driver_never_brought_up_dies_instead_of_guessing(tmp_path):
    """sysfs ไม่มี device นั้นเลย (ไดรเวอร์ไม่ขึ้น / ทะเบียนชื่อเก่า) — คนละอาการกับ "หา GID ไม่เจอ" """
    text = _stacked_controller(tmp_path)
    done = _resolve_node(tmp_path, text, "rocep9s0f9", {HCA1: (IFACE1, _normal_port(IP1, 3))}, {IFACE1: IP1})
    combined = done.stdout + done.stderr
    assert done.returncode != 0, combined
    assert "unbound variable" not in combined, combined
    assert "rocep9s0f9" in combined, combined
    assert "ls /sys/class/infiniband/" in combined, combined


def test_a_cable_without_an_ipv4_says_so_rather_than_blaming_the_gid(tmp_path):
    """device มีแต่สายยังไม่ได้ตั้ง IP — GID ผูกกับ IP จึงยังไม่มีอะไรให้เลือก บอกให้ตรงอาการ"""
    text = _stacked_controller(tmp_path)
    done = _resolve_node(tmp_path, text, HCA1, {HCA1: (IFACE1, _normal_port(IP1, 3))}, {IFACE2: IP2})
    combined = done.stdout + done.stderr
    assert done.returncode != 0, combined
    assert "unbound variable" not in combined, combined
    assert IFACE1 in combined and "ip -br addr show" in combined, combined


def test_two_hcas_that_disagree_die_instead_of_picking_one(tmp_path):
    """NCCL_IB_GID_INDEX มีค่าเดียวต่อ process — เลือกข้างเอง = อีกสายใช้ GID ผิดแบบเงียบ ๆ"""
    text = _stacked_controller(tmp_path)
    done = _resolve_node(tmp_path, text, f"{HCA1},{HCA2}",
                         {HCA1: (IFACE1, _normal_port(IP1, 3)), HCA2: (IFACE2, _normal_port(IP2, 7))},
                         {IFACE1: IP1, IFACE2: IP2})
    combined = done.stdout + done.stderr
    assert done.returncode != 0, combined
    assert HCA1 in combined and HCA2 in combined, combined
    assert "RESOLVED=" not in done.stdout, combined


# ═════════════════ 4. override ของผู้ใช้ + DRY_RUN ═════════════════
def _resolve_head(tmp_path, text, env_lines: str) -> subprocess.CompletedProcess:
    script = (
        _harness(text, "_ipv4_to_gid_suffix", "detect_gid_index_for_device",
                 "_gid_scan_script", "_resolve_gid_index_for_node", "_resolve_gid_index")
        + "IB_SYSFS_ROOT=/nonexistent-sysfs\nSSH_USER=neronain\nTRANSPORT_IP_MASTER=10.100.0.10\n"
        + env_lines
        + "_resolve_gid_index\n"
        + 'echo "GID=[${NCCL_IB_GID_INDEX}]"\n'
    )
    return _run(script, tmp_path)


def test_an_explicit_gid_index_wins_and_never_touches_sysfs(tmp_path):
    """ลำดับความสำคัญเดียวกับ NCCL_IB_HCA_EXPLICIT — ผู้ใช้ตั้งเองแล้วต้องไม่มีใครไปแก้"""
    text = _stacked_controller(tmp_path)
    done = _resolve_head(tmp_path, text,
                         'NCCL_IB_GID_INDEX=11\nNCCL_IB_HCA=rocep1s0f0\nDRY_RUN=\n')
    assert done.returncode == 0, done.stdout + done.stderr
    assert "GID=[11]" in done.stdout, done.stdout + done.stderr


def test_dry_run_stays_parse_testable_on_a_machine_with_no_connectx(tmp_path):
    """--dry-run ต้องประกอบคำสั่งให้ดูได้บนโน้ตบุ๊ก — ห้ามไปตายเพราะไม่มี sysfs ให้อ่าน"""
    text = _stacked_controller(tmp_path)
    done = _resolve_head(tmp_path, text,
                         'NCCL_IB_GID_INDEX=\nNCCL_IB_HCA=rocep1s0f0\nDRY_RUN=1\n')
    assert done.returncode == 0, done.stdout + done.stderr
    assert "GID=[]" in done.stdout, done.stdout + done.stderr


def test_a_node_without_any_roce_hca_is_not_an_error(tmp_path):
    """ไม่มี HCA = _nccl_env_pairs สั่ง NCCL_IB_DISABLE=1 อยู่แล้ว ไม่ต้องมี GID และไม่ควรตาย"""
    text = _stacked_controller(tmp_path)
    done = _resolve_head(tmp_path, text, 'NCCL_IB_GID_INDEX=\nNCCL_IB_HCA=\nDRY_RUN=\n')
    assert done.returncode == 0, done.stdout + done.stderr
    assert "GID=[]" in done.stdout, done.stdout + done.stderr


# ═════════════════ 5. ค่าที่ resolve ได้ต้องไปถึง env ของ rank นั้นจริง ═════════════════
def _env_pairs(tmp_path, text, args: str, gid_global: str = "") -> subprocess.CompletedProcess:
    script = (
        _harness(text, "_nccl_env_pairs")
        + f'NCCL_IB_GID_INDEX={gid_global}\nNCCL_IB_HCA=rocep1s0f0\nNCCL_SOCKET_IFNAME=enp1s0f0np0\n'
        + 'NCCL_CROSS_NIC=\nTRANSPORT_IP_MASTER=10.100.0.10\n'
        + f'_nccl_env_pairs {args}\n'
    )
    return _run(script, tmp_path)


def test_each_rank_gets_the_gid_index_it_was_handed(tmp_path):
    text = _stacked_controller(tmp_path)
    head = _env_pairs(tmp_path, text, '', gid_global="5")
    assert "NCCL_IB_GID_INDEX=5" in head.stdout, head.stdout + head.stderr
    worker = _env_pairs(tmp_path, text, 'enP2p1s0f0np0 roceP2p1s0f0 9', gid_global="5")
    assert "NCCL_IB_GID_INDEX=9" in worker.stdout, worker.stdout + worker.stderr
    assert "NCCL_IB_GID_INDEX=5" not in worker.stdout, worker.stdout


def test_an_empty_gid_index_is_left_out_rather_than_exported_blank(tmp_path):
    """`NCCL_IB_GID_INDEX=` เปล่า ๆ NCCL อ่านเป็น 0 — เงียบพอ ๆ กับ 3 ที่เพิ่งเอาออก"""
    text = _stacked_controller(tmp_path)
    done = _env_pairs(tmp_path, text, 'enp1s0f0np0 rocep1s0f0 ""', gid_global="5")
    assert done.returncode == 0, done.stdout + done.stderr
    assert not [line for line in done.stdout.splitlines() if line.startswith("NCCL_IB_GID_INDEX=")], done.stdout


# ═════════ 6. end-to-end: รัน start จริงกับคลัสเตอร์ปลอม แล้วดูว่าแต่ละ rank ได้เลขของตัวเอง ═════════
def test_every_rank_resolves_its_own_gid_index_from_its_own_sysfs(tmp_path):
    """ยืมเลขของ head ไม่ได้ — เครื่องที่เพิ่งอัปเฟิร์มแวร์จะมีเลขคนละตัวกับเพื่อนในวงเดียวกัน

    ใช้วงแหวน 3 เครื่องของ test_multilink_cluster (ssh/docker/ip ปลอมต่อ node) แล้วขยับ GID
    ของ worker rank 1 ไปที่ 7 · head กับ rank 2 ยังอยู่ที่ 5 — ถ้าใครยัง push ค่าของ head
    ให้ทุกคนเหมือนเดิม เทสนี้จะจับได้ทันที
    """
    import shutil

    from tests.test_audit_stacked_controller import _seed_head_cache
    from tests.test_multilink_cluster import (
        GID_INDEX, H1, H2, _bundle, _calls, _ring_fixture, _run, _worker_sh,
    )

    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    moved = pathlib.Path(env["FAKE_REMOTE"]) / "10.0.1.2" / "infiniband"
    for hca, ip in ((H2, "10.0.1.2"), (H1, "10.0.2.1")):
        shutil.rmtree(moved / hca / "ports" / "1")
        _gids(moved, hca, {
            0: (_link_local_gid(), "IB/RoCE v1"),
            6: (_gid_for(ip), "IB/RoCE v1"),
            7: (_gid_for(ip), "RoCE v2"),
        })
    _seed_head_cache(tmp_path / "home")

    done = _run(bundle, ["start"], env)
    assert done.returncode == 0, done.stdout + done.stderr

    slug = bundle.directory.name
    head_run = next(l for l in _calls(env).splitlines() if l.startswith("docker[head] run -d"))
    assert f"-e NCCL_IB_GID_INDEX={GID_INDEX}" in head_run, head_run
    assert "export NCCL_IB_GID_INDEX=7" in _worker_sh(env, "10.0.1.2", slug)
    assert f"export NCCL_IB_GID_INDEX={GID_INDEX}" in _worker_sh(env, "10.0.3.1", slug)
    # ไม่มี rank ไหนได้เลข 3 ที่เคยฝังไว้
    assert "NCCL_IB_GID_INDEX=3" not in head_run
    for ip in ("10.0.1.2", "10.0.3.1"):
        assert "NCCL_IB_GID_INDEX=3" not in _worker_sh(env, ip, slug)


def test_a_worker_whose_fabric_has_no_roce_v2_stops_the_whole_start(tmp_path):
    """worker ที่ยังไม่ได้ตั้ง RoCEv2 ต้องหยุด start ตรงนั้น ไม่ใช่ปล่อยให้ไปค้างที่ NCCL init"""
    import shutil

    from tests.test_audit_stacked_controller import _seed_head_cache
    from tests.test_multilink_cluster import H1, H2, _bundle, _ring_fixture, _run

    bundle = _bundle(tmp_path, 3)
    env = _ring_fixture(tmp_path, bundle)
    broken = pathlib.Path(env["FAKE_REMOTE"]) / "10.0.3.1" / "infiniband"
    for hca, ip in ((H2, "10.0.2.2"), (H1, "10.0.3.1")):
        shutil.rmtree(broken / hca / "ports" / "1")
        _gids(broken, hca, {0: (_link_local_gid(), "IB/RoCE v1"),
                            2: (_gid_for(ip), "IB/RoCE v1")})               # มีแต่ v1
    _seed_head_cache(tmp_path / "home")

    done = _run(bundle, ["start"], env)
    combined = done.stdout + done.stderr
    assert done.returncode != 0, combined
    assert "worker rank 2 (10.0.3.1)" in combined, combined
    assert "RoCE v2" in combined, combined


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
