"""LMDS CLI — entry point (`lmds`)"""

from __future__ import annotations

from typing import Optional

import json
import os
import pathlib
import sys

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

import lmds
from lmds.config import DEFAULT_MODELS, ProviderName, Settings, config_dir
from lmds.secrets import (
    check_credentials_permissions,
    get_secret,
    mask_preview,
    secret_source,
    set_secret,
)

app = typer.Typer(
    name="lmds",
    help="Local Model Deploy Studio — สร้าง deployment bundle สำหรับ DGX Spark และ RTX server",
    no_args_is_help=True,
)
config_app = typer.Typer(help="จัดการ provider, credentials และ site profile", no_args_is_help=True)
app.add_typer(config_app, name="config")

node_app = typer.Typer(help="คุมเครื่องอื่นจากเครื่องนี้ (fleet หลายเครื่อง)", no_args_is_help=True)
app.add_typer(node_app, name="node")
# stacked (หลายเครื่อง = โมเดลเดียว) มีคำสั่งของตัวเอง — เดิมซ่อนอยู่ใต้ `node cluster` คำสั่งเดียว
cluster_app = typer.Typer(help="คลัสเตอร์ stacked: ดูคู่ · เขียน cluster.env · กุญแจ head→worker · หมอ · "
                               "ตั้งสาย ConnectX (inspect/plan/apply/remove-net)",
                          no_args_is_help=True)
app.add_typer(cluster_app, name="cluster")

agent_app = typer.Typer(help="ให้ hub เรียกผ่าน SSH — ปกติผู้ใช้ไม่ต้องเรียกเอง", no_args_is_help=True)
app.add_typer(agent_app, name="agent")

console = Console()
err_console = Console(stderr=True)


@app.callback()
def _entry() -> None:
    """Local Model Deploy Studio — โดย neronain (fb.com/neronain.minidev)"""
    from .banner import show_banner

    show_banner(err_console)


# ── Shell completion ───────────────────────────────────────────────────────────
# ต้องเร็วและห้าม crash: shell เรียกทุกครั้งที่กด TAB — ห้ามยิง docker/health check
def _complete_slug(incomplete: str) -> list[str]:
    """เติมชื่อ slug จากทะเบียน ~/.lmds/run/ + โฟลเดอร์ bundle ในไดเรกทอรีปัจจุบัน"""
    names: set[str] = set()
    try:
        from lmds.fleet import run_root

        root = run_root()
        if root.is_dir():
            names.update(d.name for d in root.iterdir() if d.is_dir())
    except Exception:
        pass
    try:
        from pathlib import Path as _Path

        bundles = _Path("./bundles")
        if bundles.is_dir():
            names.update(d.name for d in bundles.iterdir() if d.is_dir())
    except Exception:
        pass
    return sorted(n for n in names if n.startswith(incomplete))


def _complete_target(incomplete: str) -> list[str]:
    """เติมชื่อ target preset (dgx-spark-single, rtx-4090, ...)"""
    try:
        from lmds.fit.targets import PRESETS

        return sorted(name for name in PRESETS if name.startswith(incomplete))
    except Exception:
        return []


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _complete_node(incomplete: str) -> list[str]:
    try:
        from lmds.nodes import load

        return [n.name for n in load() if n.name.startswith(incomplete)]
    except Exception:  # noqa: BLE001 — completion ห้ามพังทั้งคำสั่ง
        return []


# ── agent: ให้ hub เรียกผ่าน SSH (node ไม่ต้องรัน daemon อะไรเลย) ─────────────
@agent_app.command("info")
def agent_info() -> None:
    """พิมพ์สถานะเครื่องนี้เป็น JSON — hub เรียกผ่าน `ssh <node> lmds agent info`"""
    import json as json_module

    from lmds.inventory import snapshot

    print(json_module.dumps(snapshot(), ensure_ascii=False))


@agent_app.command("bench")
def agent_bench(
    slug: str = typer.Option("", "--slug", help="เอาผลเต็มของโมเดลนี้ตัวเดียว (ไม่ใส่ = สรุปทุกตัว)"),
) -> None:
    """พิมพ์คะแนนที่วัดไว้บนเครื่องนี้เป็น JSON — hub เอาไปรวมเป็นตารางของทั้งฟลีต

    แยกจาก `agent info` โดยตั้งใจ: info ถูก poll ทุกไม่กี่วินาทีเพื่ออัปเดตสถานะ
    ส่วนคะแนนอ่านตอนผู้ใช้เปิดดูเท่านั้น — ยัดรวมกันคือทำให้ทุกการ poll แพงขึ้นเปล่า ๆ
    """
    import json as json_module

    from lmds.bench import all_runs, latest_merged, runs_for, summarize

    if slug:
        # หน้ารายละเอียดต้องการตัวเลขต่องานทุกงาน ไม่ใช่แค่บทสรุป — ส่งทั้งก้อนไปเลย
        merged = latest_merged(slug)
        print(json_module.dumps(
            {"run": merged,
             "history": [{"stamped_at": p.stem} for p in runs_for(slug)]},
            ensure_ascii=False))
        return
    print(json_module.dumps({"runs": [summarize(run) for run in all_runs()]},
                            ensure_ascii=False))


# ── node: ทะเบียนเครื่องที่ hub คุมอยู่ ───────────────────────────────────────
@node_app.command("add")
def node_add(
    host: str = typer.Argument(..., help="IP หรือ hostname ของเครื่องปลายทาง"),
    user: str = typer.Option(..., "--user", "-u", help="user ปกติที่อยู่ในกลุ่ม docker (ไม่ต้องเป็น root)"),
    name: Optional[str] = typer.Option(None, "--name", help="ชื่อเรียกในระบบ (ว่าง = ตั้งให้จาก host)"),
    port: int = typer.Option(22, "--port", help="พอร์ต SSH"),
    note: str = typer.Option("", "--note", help="โน้ตสั้น ๆ เช่นตำแหน่งเครื่อง"),
    cluster_ip: str = typer.Option(
        "", "--cluster-ip",
        help="IP บนสายเร็ว (ConnectX/200G) ที่ใช้คุยกันตอน stacked — ว่าง = เสนอให้จากที่ตรวจพบ",
    ),
    cluster_iface: str = typer.Option("", "--cluster-iface", help="ชื่อ interface ของสายเร็ว"),
    install: bool = typer.Option(
        False, "--install", help="ติดตั้ง LMDS บนเครื่องนั้นให้เลยถ้ายังไม่มี (ต้องมี Docker อยู่แล้ว)",
    ),
) -> None:
    """เพิ่มเครื่องเข้าทะเบียน — ถามรหัสผ่านครั้งเดียวเพื่อติดตั้ง SSH key แล้วทิ้งทันที

    ไม่ต้องใช้ root: user ที่อยู่ในกลุ่ม docker ทำได้ทุกอย่างที่ LMDS ต้องการ
    """
    from lmds.nodes import (
        Node,
        NodeError,
        add,
        check_login,
        ensure_key,
        install_key,
        load,
        probe,
        install_lmds,
        public_key_path,
        status_from_probe,
        suggest_cluster_ip,
        suggest_name,
    )

    try:
        ensure_key()
        chosen = name or suggest_name(host, {n.name for n in load()})
        node = Node(name=chosen, host=host, user=user, port=port, note=note,
                    cluster_ip=cluster_ip.strip(), cluster_iface=cluster_iface.strip())

        if check_login(host, user, port):
            console.print("[green]key ใช้ได้อยู่แล้ว[/green] — ข้ามการถามรหัสผ่าน")
        else:
            console.print(f"ติดตั้ง SSH key ของ LMDS ไปยัง {user}@{host}")
            console.print(f"[dim]public key: {public_key_path()}[/dim]")
            console.print("[dim]รหัสผ่านใช้ครั้งเดียวเพื่อติดตั้ง key — ไม่ถูกบันทึกลงดิสก์[/dim]")
            password = typer.prompt(f"รหัสผ่านของ {user}@{host}", hide_input=True)
            install_key(host, user, password, port)
            del password
            if not check_login(host, user, port):
                err_console.print("[red]ติดตั้ง key แล้วแต่ยัง login ไม่ได้[/red]")
                raise typer.Exit(code=1)
            console.print("[green]ติดตั้ง key สำเร็จ[/green]")

        # เครื่องที่ยังไม่ได้ติดตั้ง LMDS ต้องเพิ่มเข้าทะเบียนได้ — key ติดตั้งไปแล้ว
        # และ hub ยังใช้ node run สั่งติดตั้งต่อได้ การบังคับให้ติดตั้งก่อนเป็นการวางลำดับกลับหัว
        try:
            info = probe(node)
            reachable = True
        except NodeError as exc:
            info, reachable = {}, False
            node.last_error = str(exc)[:200]
            if install:
                console.print(f"ติดตั้ง LMDS บน {node.target} (ใช้เวลาสักพัก) …")
                result = install_lmds(node)
                if not result.ok:
                    err_console.print((result.stderr or result.stdout)[-800:])
                    err_console.print("[red]ติดตั้ง LMDS บนเครื่องนั้นไม่สำเร็จ[/red]")
                else:
                    info, reachable = probe(node), True
                    node.last_error = ""
        host_info = info.get("host") or {}
        for key, value in status_from_probe(info).items():
            setattr(node, key, value)
        node.last_seen = _now() if reachable else ""
        if reachable and not node.cluster_ip:
            # เสนอเฉย ๆ ไม่ตั้งให้เอง — เดา IP ผิดแล้ว stacked จะค้างตอน NCCL init โดยไม่บอกสาเหตุ
            suggestion = suggest_cluster_ip(host_info)
            if suggestion:
                console.print(f"[dim]พบสายเร็วที่ {suggestion} — ตั้งเป็น cluster IP ด้วย: "
                              f"lmds node set {chosen} --cluster-ip {suggestion}[/dim]")
        add(node)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if not reachable:
        console.print(f"\n[bold]เพิ่ม '{node.name}' แล้ว[/bold] — [yellow]แต่ยังอ่านสถานะไม่ได้[/yellow]")
        err_console.print(node.last_error)
        console.print(
            f"\n[dim]SSH key ใช้ได้แล้ว — ให้ระบบติดตั้ง LMDS ให้เลย:[/dim]\n"
            f"  lmds node install {node.name}\n"
            f"[dim](เครื่องนั้นต้องมี Docker + git อยู่แล้ว · ถ้ายังไม่มีต้องรัน ./install.sh "
            f"บนเครื่องนั้นเองเพราะขั้น sudo ต้องมีคนกรอกรหัสผ่าน)[/dim]"
        )
        return

    gpus = ", ".join(g["name"] for g in host_info.get("gpus", [])) or "ไม่พบ GPU"
    console.print(
        f"\n[bold]เพิ่ม '{node.name}' แล้ว[/bold] — {gpus} · "
        f"lmds {node.lmds_version} · โมเดล {len(info.get('models', []))} ตัว"
        + (f" · นอกระบบอีก {len(host_info['foreign'])}" if host_info.get("foreign") else "")
    )
    console.print("[dim]ดูทั้งหมด: lmds node list · สถานะรวมทุกเครื่อง: lmds ps --all[/dim]")


@node_app.command("list")
def node_list(
    check: bool = typer.Option(False, "--check", help="ต่อจริงเพื่อดูว่าเครื่องยังตอบไหม (ช้ากว่า)"),
) -> None:
    """เครื่องทั้งหมดที่อยู่ในทะเบียน — เรียงตามลำดับที่ลากจัดไว้ในหน้าเว็บ"""
    from lmds.config import Settings
    from lmds.nodes import (
        NodeError,
        in_saved_order,
        load,
        probe,
        status_from_probe,
        update,
    )

    nodes = in_saved_order(load(), Settings.load().ui.node_order)
    if not nodes:
        console.print("ยังไม่มีเครื่องในทะเบียน — เพิ่มด้วย: lmds node add <ip> --user <ชื่อ>")
        return

    # commit ของ hub ตัวนี้ — ไว้ป้าย "≠ hub" ให้เครื่องที่ยังรันโค้ดเก่า (เทียบแบบ prefix · ดู _same_commit)
    from lmds.inventory import source_commit

    hub_commit = source_commit()

    def row_for(node):
        # host ในทะเบียนคือ "ที่อยู่ที่ hub ใช้ SSH" ซึ่งเป็นชื่อได้ (`orb`, ชื่อบน Tailscale)
        # และเครื่องหลัง NAT มองจาก hub เป็นคนละที่อยู่กับที่เครื่องในวงเดียวกันใช้เรียกมัน
        local_ip = node.local_ip
        if check:
            try:
                info = probe(node)
                fields = status_from_probe(info)
                version = _version_label(fields.get("lmds_version", node.lmds_version),
                                         fields.get("lmds_commit", node.lmds_commit))
                node_commit = fields.get("lmds_commit", node.lmds_commit)
                local_ip = fields.get("local_ip", local_ip)
                update(node.name, last_seen=_now(), last_error="", **fields)
                status = f"[green]ต่อได้[/green] · โมเดล {len(info.get('models', []))} ตัว"
                # "0 ตัว" บนเครื่องที่มี inference server รันอยู่จริงคือคำตอบที่ผิด —
                # node รุ่นเก่าไม่ส่งคีย์นี้มา (ไม่มี = ไม่รู้ ไม่ใช่ไม่มี) จึงเงียบไว้
                outside = len((info.get("host") or {}).get("foreign") or [])
                if outside:
                    status += f" · [yellow]นอกระบบอีก {outside}[/yellow]"
            except NodeError as exc:
                update(node.name, last_error=str(exc)[:200])
                version = _version_label(node.lmds_version, node.lmds_commit)
                node_commit = node.lmds_commit
                status = f"[red]ต่อไม่ได้[/red] {str(exc)[:60]}"
        else:
            version = _version_label(node.lmds_version, node.lmds_commit)
            node_commit = node.lmds_commit
            status = node.last_seen or "—"
        if version and hub_commit and node_commit and not _same_commit(node_commit, hub_commit):
            version += " [yellow]≠ hub[/yellow]"
        return (node.name, f"{node.target}:{node.port}", local_ip or "—",
                version or "—", status, node.note)

    # จัดกลุ่มตาม site — เครื่องเยอะจากหลายไซต์จะได้ไม่กองรวมเป็นลิสต์ยาวจนหาไม่เจอ
    # เรียงให้ site ที่ยังไม่จัดกลุ่ม ("—") อยู่ท้ายสุด ที่เหลือเรียงตามชื่อไซต์
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for node in nodes:
        groups.setdefault(node.site or "", []).append(node)
    ordered = sorted(groups, key=lambda s: (s == "", s.lower()))

    multi = len([s for s in ordered if s]) > 0 and len(ordered) > 1
    for site in ordered:
        title = f"ไซต์: {site}" if site else ("เครื่องในทะเบียน" if not multi else "— ยังไม่จัดไซต์ —")
        table = Table(title=title)
        for col in ("ชื่อ", "ปลายทาง", "IP ของเครื่อง", "lmds",
                    "สถานะ" if check else "เห็นล่าสุด", "โน้ต"):
            table.add_column(col)
        for node in groups[site]:
            table.add_row(*row_for(node))
        console.print(table)

    console.print("[dim]จัดกลุ่มตามไซต์: lmds node set <ชื่อ> --site <ไซต์>"
                  + ("" if check else " · เช็กสถานะจริง: lmds node list --check")
                  + (f" · hub อยู่ที่ {hub_commit} — ≠ hub = อัปเดตด้วย lmds node install <ชื่อ>" if hub_commit else "")
                  + "[/dim]")


@node_app.command("remove")
def node_remove(
    name: str = typer.Argument(..., autocompletion=_complete_node),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """เอาเครื่องออกจากทะเบียน — ไม่แตะอะไรบนเครื่องนั้น"""
    from lmds.nodes import NodeError, remove

    if not yes and not typer.confirm(f"เอา '{name}' ออกจากทะเบียน?", default=True):
        raise typer.Exit(code=1)
    try:
        node = remove(name)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"เอา '{node.name}' ออกแล้ว")
    console.print(
        f"[dim]key ของ LMDS ยังอยู่บนเครื่องนั้น — ถอนเองได้ที่ {node.target}: "
        "ลบบรรทัดที่ลงท้ายด้วย lmds-hub ออกจาก ~/.ssh/authorized_keys[/dim]"
    )






@app.command("recipes")
def list_recipes(
    model: Optional[str] = typer.Argument(None, help="ดูสูตรของโมเดลนี้ (ว่าง = ทั้งหมด)"),
    sync: bool = typer.Option(False, "--sync", help="ดึงสูตรใหม่จากรีโป controller ของทีมก่อนแสดง"),
    publish: Optional[str] = typer.Option(
        None, "--publish", help="ส่ง controller ของ slug นี้ (ที่รันผ่าน + ทดสอบแล้ว) ขึ้นคลัง",
        autocompletion=_complete_slug),
    features: Optional[str] = typer.Option(
        None, "--features",
        help="publish: ความสามารถที่วัดได้จริง คั่นด้วย comma (เช่น tools,vision,reasoning) "
             "— ชนะค่า rule-based ของ profile"),
    repo: Optional[str] = typer.Option(None, "--repo", help="รีโป controller (ค่าเริ่มต้นคือของทีม)"),
    ref: Optional[str] = typer.Option(None, "--ref", help="branch/tag ที่จะดึง (ค่าเริ่มต้น main)"),
    no_push: bool = typer.Option(False, "--no-push", help="publish: commit ไว้ในเครื่องแต่ไม่ push ขึ้น remote"),
) -> None:
    """สูตรที่รันผ่านจริงแล้ว — ใช้อัตโนมัติเมื่อไม่มี LLM provider

    เครื่องที่ไม่มี API key จะได้ค่าเหล่านี้แทนการเดา: image ที่ถูกรุ่น, parser, และข้อบังคับ
    ของ quantization ที่ไม่ตั้งแล้ว start ไม่ขึ้น

    `--sync` ดึงจากรีโป controller ของทีมมาเพิ่ม (อ่านไฟล์อย่างเดียว ไม่รันสคริปต์) —
    แก้ที่รีโปแล้ว push จากนั้นสั่ง sync ที่ hub ทุกเครื่องก็ได้ชุดเดียวกัน
    """
    from lmds.recipes import find_recipe, load_catalog
    from lmds.recipes.sync import DEFAULT_REF, DEFAULT_REPO, SyncError
    from lmds.recipes.sync import sync as sync_recipes
    from lmds.recipes.sync import synced_source

    if publish:
        from pathlib import Path as _Path

        import yaml as _yaml

        from lmds.config.settings import Settings
        from lmds.fleet import find
        from lmds.recipes.publish import default_local_repo
        from lmds.recipes.publish import publish as publish_recipe

        server = find(publish)
        if server is None or not server.controller:
            err_console.print(f"[red]ไม่รู้จัก '{publish}'[/red] — ดู: lmds ps")
            raise typer.Exit(code=1)
        bundle_dir = _Path(server.controller).parent
        profile_path = bundle_dir / "MODEL_PROFILE.yaml"
        try:
            profile = _yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        except OSError:
            profile = {}
        settings = Settings.load()
        target = repo or settings.recipes.publish_repo or ""
        target_ref = ref or settings.recipes.publish_ref or "main"
        feat_list = ([f.strip() for f in features.split(",") if f.strip()]
                     if features is not None else None)
        try:
            result = publish_recipe(
                publish, server.controller, profile,
                features=feat_list, repo=target, ref=target_ref, now=_now(),
                push=not no_push,
            )
        except SyncError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        if not result["features"]:
            console.print("[yellow]⚠ ไม่มี measured features[/yellow] — profile เป็น rule-based "
                          "อาจไม่ครบ · ระบุเองด้วย --features tools,vision,reasoning ให้สูตรพก "
                          "ความสามารถที่วัดจริงไปด้วย")
        if result.get("overrides"):
            folded = " · ".join(f"{k}={v[:60]}" for k, v in result["overrides"].items())
            console.print(f"[dim]พับค่าที่ lmds set ไว้ลง header แล้ว: {folded}[/dim]")
        if result.get("unfolded"):
            console.print(f"[yellow]⚠ พับไม่ได้ (controller รุ่นเก่าไม่มีบรรทัดรองรับ): "
                          f"{', '.join(result['unfolded'])}[/yellow] — lmds rebuild แล้ว publish ใหม่")
        where = result["target"] if result["remote"] else f"local: {result['target']}"
        if not result["committed"]:
            console.print(f"[yellow]ไม่มีอะไรเปลี่ยน[/yellow] — {publish} ตรงกับที่อยู่ในคลังแล้ว ({where})")
        else:
            console.print(f"[green]publish {publish} แล้ว[/green] → {where}")
            if result["remote"] and no_push:
                console.print("[dim]commit ไว้ในเครื่องแล้ว (--no-push) — push เองภายหลัง[/dim]")
            elif not result["remote"]:
                console.print(f"[dim]local store: {default_local_repo()} · "
                              f"ตั้ง publish_repo เป็นรีโป candidates เพื่อ push ขึ้น[/dim]")
        return

    if sync:
        try:
            result = sync_recipes(repo or DEFAULT_REPO, ref or DEFAULT_REF, now=_now())
        except SyncError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[green]ดึงมาแล้ว {result['count']} สูตร[/green] จาก "
                      f"{result['repo']} @ {result['commit']}")
        for line in result["skipped"]:
            console.print(f"  [yellow]ข้าม[/yellow] {line}")
        console.print()

    if model:
        recipe = find_recipe(model)
        if recipe is None:
            console.print(f"[yellow]ยังไม่มีสูตรของ '{model}'[/yellow] — จะใช้ rule-based ตามปกติ")
            console.print("[dim]ดูทั้งหมด: lmds recipes[/dim]")
            raise typer.Exit(code=1)
        console.print(f"[bold]{recipe.label or recipe.match}[/bold]")
        console.print(f"  engine    : {recipe.engine}")
        if recipe.image:
            console.print(f"  image     : {recipe.image}")
        for key, value in (recipe.serving or {}).items():
            console.print(f"  {key:10}: {value}")
        if recipe.tool_calling.get("parser"):
            console.print(f"  tools     : {recipe.tool_calling['parser']}")
        if recipe.reasoning.get("parser"):
            console.print(f"  reasoning : {recipe.reasoning['parser']}")
        for note in recipe.notes or []:
            console.print(f"  [dim]· {note}[/dim]")
        console.print(f"\n  ทดสอบบน : {recipe.validated_on or '—'}")
        console.print(f"  ที่มา    : {recipe.source or '—'}")
        return

    catalog = load_catalog()
    table = Table(title=f"สูตรที่รันผ่านจริง ({len(catalog)} รุ่น)")
    table.add_column("โมเดล")
    table.add_column("engine")
    table.add_column("สิ่งที่สูตรกำหนด")
    table.add_column("ทดสอบบน")
    for recipe in catalog:
        sets = []
        if recipe.image:
            sets.append("image")
        sets += list((recipe.serving or {}).keys())
        if recipe.tool_calling.get("parser"):
            sets.append("tools")
        if recipe.reasoning.get("parser"):
            sets.append("reasoning")
        table.add_row(recipe.label or recipe.match, recipe.engine, ", ".join(sets),
                      recipe.validated_on or "—")
    console.print(table)

    # คำสั่งที่ก๊อปไปใช้ต่อได้เลย — เดิมตารางบอกว่ามีสูตรอะไร แต่ไม่บอกว่าจะใช้ยังไง
    console.print("\n[bold]ใช้สูตรไหนก็สั่งได้เลย[/bold] (ไม่ต้องมี LLM):")
    for recipe in catalog[:5]:
        console.print(f"  [dim]lmds deploy {recipe.match} --no-llm[/dim]")
    if len(catalog) > 5:
        console.print(f"  [dim]… อีก {len(catalog) - 5} รุ่น — ดูรายตัว: lmds recipes <model>[/dim]")

    source = synced_source()
    if source:
        console.print(f"\n[dim]{source.get('count', 0)} สูตรมาจาก {source.get('repo', '?')} "
                      f"@ {source.get('commit', '?')} · ดึงเมื่อ {source.get('synced_at') or '—'} · "
                      f"ดึงใหม่: lmds recipes --sync[/dim]")
    else:
        console.print(f"\n[dim]ดึงสูตรจากรีโป controller ของทีมเพิ่มได้: lmds recipes --sync "
                      f"(ค่าเริ่มต้น {DEFAULT_REPO})[/dim]")


@app.command("prune")
def prune_registrations(
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้วและไม่ได้รันอยู่

    เครื่องที่ใช้ "จัดการ" อย่างเดียว (เช่นโน้ตบุ๊กที่ใช้สร้าง bundle ให้เครื่องอื่น) จะสะสมทะเบียน
    ของ bundle ที่ย้าย/ลบไปแล้ว ทำให้หน้าจอเต็มไปด้วยรายการที่กดอะไรก็ไม่ได้ และเสี่ยงสั่งผิดเครื่อง

    **ลบเฉพาะไฟล์ทะเบียน** ไม่แตะ weight, bundle หรือ container ใด ๆ
    """
    from pathlib import Path as _Path

    from lmds.fleet import discover, run_root

    dead = []
    for server in discover():
        if server.running or not server.controller:
            continue
        if not _Path(server.controller).exists():
            dead.append(server)

    if not dead:
        console.print("ไม่มีทะเบียนค้าง — ทุกตัวชี้ไป bundle ที่มีอยู่จริง")
        return

    table = Table(title="ทะเบียนที่ชี้ไปของที่ไม่มีแล้ว")
    table.add_column("ชื่อ")
    table.add_column("controller ที่หายไป")
    for server in dead:
        table.add_row(server.slug, server.controller)
    console.print(table)
    console.print("[dim]ลบเฉพาะไฟล์ทะเบียน — weight, bundle และ container ไม่ถูกแตะ[/dim]")

    if not yes and not typer.confirm(f"ล้าง {len(dead)} รายการ?", default=True):
        raise typer.Exit(code=1)

    root = run_root()
    removed = 0
    for server in dead:
        meta = root / server.slug / "server.meta"
        try:
            meta.unlink()
            if not any(meta.parent.iterdir()):
                meta.parent.rmdir()
            removed += 1
        except OSError as exc:
            err_console.print(f"[yellow]ลบ {meta} ไม่ได้: {exc}[/yellow]")
    console.print(f"[green]ล้าง {removed} รายการแล้ว[/green]")

@app.command("scan")
def scan_models(
    root: list[str] = typer.Option([], "--root", help="ที่ค้นเพิ่มเติม (ระบุซ้ำได้)"),
    all_nodes: bool = typer.Option(False, "--all", "-a", help="ค้นทุกเครื่องในทะเบียนด้วย"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """หา weight ที่มีอยู่แล้วบนเครื่อง ไม่ว่าจะถูกเก็บไว้แบบไหน

    เครื่องลูกค้ามักมีโมเดลอยู่ก่อนติดตั้ง LMDS และไม่ได้จัดระเบียบแบบเดียวกับเรา —
    คำสั่งนี้บอกว่ามีอะไรอยู่ตรงไหนแล้ว จะได้ไม่โหลดซ้ำหลายสิบ GB · **อ่านอย่างเดียว ไม่ย้ายไม่ลบ**
    """
    import json as json_module

    from lmds.scanner import scan

    def rows_for(models) -> list[dict]:
        return [
            {"kind": m.kind, "name": m.name, "path": m.path, "size_gb": m.size_gb,
             "shards": m.shard_count, "layout": m.layout, "revisions": m.revisions,
             "hub_cache_root": m.hub_cache_root}
            for m in models
        ]

    local = rows_for(scan(list(root)))
    payload = {"host": local}

    if all_nodes:
        from lmds.nodes import NodeError, load, run as run_remote

        for node in load():
            try:
                result = run_remote(node, "lmds scan --json", timeout=300)
                payload[node.name] = json_module.loads(result.stdout).get("host", []) \
                    if result.ok else []
            except (NodeError, json_module.JSONDecodeError):
                payload[node.name] = []

    if json_out:
        print(json_module.dumps(payload, ensure_ascii=False))
        return

    for where, models in payload.items():
        table = Table(title=f"weight ที่เจอบน {where}")
        table.add_column("ชนิด")
        table.add_column("โมเดล")
        table.add_column("ขนาด", justify="right")
        table.add_column("shard", justify="right")
        table.add_column("ที่เก็บ")
        if not models:
            console.print(f"[dim]{where}: ไม่พบ weight (หรือติดต่อไม่ได้)[/dim]")
            continue
        for m in models:
            note = ""
            if m["kind"] == "hf" and m["layout"] == "root":
                # เลย์เอาต์เก่า: ไลบรารีของ HF จะมองไม่เห็นถ้าไม่ตั้ง HF_HUB_CACHE ให้ตรง
                note = "  [yellow](เลย์เอาต์เก่า — ต้องตั้ง HF_HUB_CACHE)[/yellow]"
            table.add_row(m["kind"], m["name"], f"{m['size_gb']} GB",
                          str(m["shards"] or "—"), m["path"] + note)
        console.print(table)

    legacy = [m for rows in payload.values() for m in rows
              if m["kind"] == "hf" and m["layout"] == "root"]
    if legacy:
        console.print(
            "\n[dim]โมเดลที่อยู่เลย์เอาต์เก่า controller จะตั้ง HF_HUB_CACHE ให้เองตอน start "
            "— ไม่ต้องย้ายไฟล์[/dim]"
        )

@node_app.command("install")
def node_install(
    name: str = typer.Argument("", autocompletion=_complete_node,
                               help="ชื่อเครื่อง (เว้นว่างคู่กับ --all = ทุกเครื่องในทะเบียน)"),
    all_nodes: bool = typer.Option(False, "--all", help="อัปเดตทุกเครื่องในทะเบียน"),
    with_prereq: bool = typer.Option(
        False, "--with-prereq",
        help="ให้ติดตั้ง Docker/NVIDIA toolkit ด้วย (ต้องรัน sudo ได้โดยไม่ถามรหัสผ่าน)",
    ),
) -> None:
    """ติดตั้งหรืออัปเดต LMDS บนเครื่องนั้นผ่าน SSH

    ทุกเครื่องที่ hub คุมต้องมี `lmds` อยู่บนเครื่อง — hub ไม่ได้ส่ง agent ไปรันเอง แต่เรียก
    `lmds agent info` ผ่าน SSH คำสั่งนี้จึงเป็นวิธีทำให้เครื่องปลายทางพร้อมโดยไม่ต้อง ssh เข้าไปเอง
    """
    from lmds.nodes import (
        NodeError,
        find,
        install_lmds,
        load,
        probe,
        status_from_probe,
        update,
    )

    # อัปเดตทีละเครื่องด้วยมือแปลว่ามีวันลืมเครื่องหนึ่ง แล้วมันค้างเวอร์ชันเก่าอยู่เงียบ ๆ
    # จนกว่าจะมีคนสังเกตเห็น (เจอจริง: msi-6 ค้างที่ 0.1.0 อยู่หลายรอบ)
    # หลังติดตั้ง สรุปให้ชัดว่าเครื่องนั้น *ตรงกับ hub แล้วหรือยัง* — เดิมพิมพ์แค่ "รัน lmds 0.6.0 (0ad1a59e)"
    # ซึ่งคนอ่านต้องไปเทียบ hash เอง และเทียบเป๊ะ ๆ จะผิดเมื่อ hash ย่อยาวไม่เท่ากัน (ดู _same_commit)
    from lmds.inventory import source_commit

    hub_commit = source_commit()

    if all_nodes:
        nodes = load()
        if not nodes:
            console.print("ยังไม่มีเครื่องในทะเบียน")
            return
        failed = []
        for index, target in enumerate(nodes, 1):
            console.print(f"\n[bold]{index}/{len(nodes)}[/bold] {target.name}")
            result = install_lmds(target, with_prereq=with_prereq)
            if not result.ok:
                failed.append(target.name)
                err_console.print(f"[red]ไม่สำเร็จ[/red] {(result.stderr or '').strip()[-200:]}")
                continue
            try:
                fields = status_from_probe(probe(target))
                version = _version_label(fields.get("lmds_version", ""), fields.get("lmds_commit", ""))
                update(target.name, last_seen=_now(), last_error="", **fields)
                console.print(f"[green]พร้อมแล้ว[/green] — lmds {version}"
                              + _hub_commit_note(fields.get("lmds_commit", ""), hub_commit))
            except NodeError as exc:
                failed.append(target.name)
                err_console.print(f"[red]ติดตั้งแล้วแต่อ่านสถานะไม่ได้: {exc}[/red]")
        if failed:
            err_console.print(f"\n[red]ไม่สำเร็จ {len(failed)} เครื่อง:[/red] {', '.join(failed)}")
            raise typer.Exit(code=1)
        console.print(f"\n[green]อัปเดตครบ {len(nodes)} เครื่อง[/green]")
        return

    if not name:
        err_console.print("[red]ต้องระบุชื่อเครื่อง[/red] หรือใช้ --all เพื่ออัปเดตทุกเครื่อง")
        raise typer.Exit(code=1)
    node = find(name)
    if node is None:
        err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
        raise typer.Exit(code=1)

    console.print(f"ติดตั้ง/อัปเดต LMDS บน {node.target} — hub ส่งโค้ดไปให้ (หรือดึงจาก GitHub ถ้า hub ไม่มี checkout) แล้วรัน install.sh บนเครื่องนั้น")
    if not with_prereq:
        console.print("[dim]ข้ามขั้น Docker/NVIDIA toolkit (ต้องใช้ sudo ซึ่งไม่มีคนกรอกรหัสผ่าน) "
                      "— ใส่ --with-prereq ถ้า sudo ผ่านโดยไม่ถาม[/dim]")

    result = install_lmds(node, with_prereq=with_prereq)
    tail = (result.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        console.print(f"[dim]{line}[/dim]")
    if not result.ok:
        from lmds.nodes import explain_install_failure

        combined = (result.stdout or "") + (result.stderr or "")
        hint = explain_install_failure(combined, node)
        err_console.print(combined.strip()[-600:])
        err_console.print(f"[red]ติดตั้งไม่สำเร็จบน {node.target}[/red]")
        if hint:
            err_console.print(f"\n[yellow]{hint}[/yellow]")
        raise typer.Exit(code=1)

    try:
        info = probe(node)
    except NodeError as exc:
        err_console.print(f"[red]ติดตั้งแล้วแต่ยังอ่านสถานะไม่ได้: {exc}[/red]")
        raise typer.Exit(code=1)
    fields = status_from_probe(info)
    version = _version_label(fields.get("lmds_version", ""), fields.get("lmds_commit", ""))
    update(name, last_seen=_now(), last_error="", **fields)
    console.print(f"[green]พร้อมแล้ว[/green] — {node.name} รัน lmds {version}"
                  + _hub_commit_note(fields.get("lmds_commit", ""), hub_commit))

@node_app.command("set")
def node_set(
    name: str = typer.Argument(..., autocompletion=_complete_node),
    cluster_ip: Optional[str] = typer.Option(None, "--cluster-ip", help="IP บนสายเร็วที่ใช้ตอน stacked"),
    cluster_iface: Optional[str] = typer.Option(None, "--cluster-iface", help="ชื่อ interface ของสายเร็ว"),
    note: Optional[str] = typer.Option(None, "--note"),
    site: Optional[str] = typer.Option(
        None, "--site",
        help="ป้ายไซต์/ลูกค้าที่เครื่องนี้ไปตั้งอยู่ — ใช้จัดกลุ่มบนหน้าจอเท่านั้น (ว่าง = เอาป้ายออก) "
             "เป็นตัวบังคับตอนจับกลุ่ม stacked ด้วย — คนละไซต์จับคู่กันไม่ได้",
    ),
    cluster_name: Optional[str] = typer.Option(
        None, "--cluster-name",
        help="ชื่อคลัสเตอร์ — ใช้แบ่งหลายคลัสเตอร์ในไซต์เดียวกัน (ว่าง = ให้ระบบแบ่งเองตาม subnet)",
    ),
    alt_host: Optional[str] = typer.Option(
        None, "--alt-host",
        help="ที่อยู่สำรองของเครื่องเดียวกัน เช่น Tailscale (คั่นด้วย , ได้ · ว่าง = ลบทิ้ง)",
    ),
    stack: Optional[bool] = typer.Option(
        None, "--stack/--no-stack",
        help="ยอมให้เครื่องนี้ถูกเสนอเข้ากลุ่ม stacked ไหม (--no-stack = ไม่เอาเข้ากลุ่มไม่ว่าฮาร์ดแวร์จะตรงแค่ไหน)",
    ),
) -> None:
    """แก้ค่าของเครื่องในทะเบียน — cluster IP/interface, ที่อยู่สำรอง, โน้ต และเอาเข้ากลุ่มหรือไม่

    เปลี่ยน host/user/port ไม่ได้ที่นี่โดยตั้งใจ: ที่อยู่เปลี่ยน = คนละเครื่อง ให้ remove แล้ว add ใหม่
    """
    from lmds.nodes import NodeError, find, probe, suggest_cluster_ip, update

    node = find(name)
    if node is None:
        err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
        raise typer.Exit(code=1)

    changes = {k: v for k, v in
               (("cluster_ip", cluster_ip), ("cluster_iface", cluster_iface), ("note", note),
                ("site", site.strip() if site is not None else None), ("stack", stack),
                ("cluster_name", cluster_name.strip() if cluster_name is not None else None))
               if v is not None}
    if alt_host is not None:
        changes["alt_hosts"] = [h.strip() for h in alt_host.split(",") if h.strip()]
    if not changes:
        console.print(f"[bold]{node.name}[/bold] — {node.target}:{node.port}")
        console.print(f"site: {node.site or '—'}  คลัสเตอร์: {node.cluster_name or '— (แบ่งเองตาม subnet)'}")
        console.print(f"cluster IP: {node.cluster_ip or '—'}  interface: {node.cluster_iface or '—'}")
        console.print(f"ที่อยู่: {' → '.join(node.all_hosts)}")
        console.print("เข้ากลุ่ม stacked: " + ("ได้" if node.stack else "[yellow]ไม่เอาเข้ากลุ่ม[/yellow]"))
        try:
            suggestion = suggest_cluster_ip(probe(node).get("host") or {})
        except NodeError:
            suggestion = ""
        if suggestion and suggestion != node.cluster_ip:
            console.print(f"[dim]สายเร็วที่ตรวจพบ: {suggestion}[/dim]")
        console.print("[dim]ตั้งค่า: lmds node set <ชื่อ> --cluster-ip <ip>[/dim]")
        return

    try:
        node = update(name, **changes)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"อัปเดต '{node.name}' แล้ว — cluster IP: {node.cluster_ip or '—'} · "
                  f"ที่อยู่: {' → '.join(node.all_hosts)}"
                  + ("" if node.stack else " · [yellow]ไม่เอาเข้ากลุ่ม stacked[/yellow]"))



@node_app.command(
    "ctl",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def node_ctl(
    name: str = typer.Argument(..., autocompletion=_complete_node),
    slug: str = typer.Argument(..., help="ชื่อ bundle บนเครื่องนั้น"),
    command: list[str] = typer.Argument(..., help="คำสั่งของ controller เช่น prepare-runtime, start"),
) -> None:
    """สั่งคำสั่งของ controller บนเครื่องปลายทาง — เช่น prepare-runtime / download / sync-worker

    ต่างจาก `lmds node run` ตรงที่อันนั้นสั่ง *คำสั่งของ lmds* ส่วนอันนี้สั่ง *สคริปต์ controller*
    ในตัว bundle ซึ่งมีขั้นตอนที่ lmds ไม่ได้ห่อไว้ (prepare-runtime, sync-worker, test-text ฯลฯ)
    """
    from lmds.nodes import NodeError, find, stream

    node = find(name)
    if node is None:
        err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
        raise typer.Exit(code=1)

    # หา bundle บนเครื่องนั้นเอง — path ต่างกันไปตามที่ผู้ใช้ deploy ไว้
    quoted = " ".join(shlex.quote(c) for c in command)
    # cd เข้า bundle ก่อนแล้วค่อยหา controller — เดิมคำนวณ path ก่อน cd แล้วใช้หลัง cd
    # ซึ่ง path แบบ relative จะชี้ผิดที่ทันที
    script = (
        f"dir=\"$(ls -d ~/bundles/{slug} ~/*/bundles/{slug} ./bundles/{slug} 2>/dev/null | head -1)\"; "
        f"[ -n \"$dir\" ] || {{ echo 'ไม่พบ bundle {slug} บน {name}' >&2; exit 1; }}; "
        f"cd \"$dir\" || exit 1; "
        f"ctl=\"$(ls ./*-single.sh ./*-stacked.sh 2>/dev/null | head -1)\"; "
        f"[ -n \"$ctl\" ] || {{ echo 'ไม่พบ controller ใน '\"$PWD\" >&2; exit 1; }}; "
        f"\"$ctl\" {quoted}"
    )
    # gated repo ต้องมี HF_TOKEN ที่ฝั่ง node — hub มีอยู่แล้ว แต่ bundle จงใจไม่พกไปด้วย
    #
    # เคสจริง 2026-08-31: `node ctl spark-head <slug> download` ตกทันทีด้วย "เป็น gated
    # repo — ต้องมี HF_TOKEN" ทั้งที่ hub ถือ token ที่ใช้ได้อยู่ · หน้าเว็บให้ยืมได้ตั้งแต่
    # รอบก่อน (lmds.web.jobs) แต่เส้นทาง CLI ไม่เคยถูกต่อให้ — คนที่ทำงานผ่าน CLI จึง
    # deploy โมเดล gated ข้ามเครื่องไม่ได้เลย
    #
    # ความลับเดินทางทาง stdin ของ ssh เท่านั้น (stream() จัดการให้) ไม่ผ่าน argv ที่ `ps`
    # ของทุก user มองเห็น และไม่เขียนลงไฟล์บนเครื่องปลายทาง
    token = get_secret("hf")
    secret_env = {"HF_TOKEN": token} if token else None

    # สตรีมทีละบรรทัดแทนการรอทั้งก้อน — download 90 GB ใช้เวลาเป็นสิบนาที ของเดิมเงียบ
    # สนิทจนจบแล้วค่อยพ่นออกมาทีเดียว ระหว่างนั้นแยกไม่ออกว่ากำลังทำงานหรือค้างไปแล้ว
    try:
        proc = stream(node, script, secret_env)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", "replace")
        # ปลายทางไม่ควรพิมพ์ token อยู่แล้ว — แต่ "ไม่ควร" กับ "ไม่เคย" คนละเรื่อง
        if token:
            line = line.replace(token, "***")
        print(line, end="", flush=True)
    raise typer.Exit(code=proc.wait())

@node_app.command("cluster")
def node_cluster(
    write: Optional[str] = typer.Option(
        None, "--write", "-w", metavar="SLUG",
        help="เขียน cluster.env ลง bundle ของโมเดลนั้น เพื่อให้ controller ไม่ต้องถาม IP ตอน start",
    ),
    worker: Optional[str] = typer.Option(
        None, "--worker", help="ชื่อเครื่องที่จะเป็น worker (ว่าง = ใช้เครื่องเดียวในกลุ่มที่พร้อม)",
    ),
    head: Optional[str] = typer.Option(
        None, "--head", help="ชื่อเครื่องที่จะเป็น head (ว่าง = เครื่องที่ใช้กับ --on หรือเครื่องนี้)",
    ),
    on: Optional[str] = typer.Option(
        None, "--on", metavar="NODE",
        help="เขียน cluster.env ลง bundle ที่อยู่บนเครื่องนั้นแทนเครื่องนี้ (bundle อยู่กับเครื่องที่รันจริง)",
    ),
    self_stack: Optional[bool] = typer.Option(
        None, "--self-stack/--no-self-stack",
        help="เอา 'เครื่องนี้' (hub) เข้ากลุ่ม stacked ด้วยไหม — จำค่าไว้ใน config (node อื่นใช้ node set --no-stack)",
    ),
) -> None:
    """เครื่องไหนจับคู่ stacked กันได้บ้าง — ต่อทุกเครื่องจริงจึงช้ากว่า node list"""
    from lmds.config import Settings
    from lmds.inventory import host_payload
    from lmds.nodes import (
        NodeError, check_cluster_ip, cluster_groups, cluster_note, in_saved_order, load, probe,
        stack_ready, suggest_cluster_ip,
    )

    settings = Settings.load()
    if self_stack is not None and self_stack != settings.cluster.stack_self:
        settings.cluster.stack_self = self_stack
        settings.save()
    stack_self = settings.cluster.stack_self

    local = host_payload()
    local_name = local.get("hostname") or "เครื่องนี้"
    machines = [{"name": local_name, "host": local, "cluster_ip": suggest_cluster_ip(local),
                 "site": "", "cluster_name": "", "stack": stack_self}]

    table = Table(title="สายเชื่อมของแต่ละเครื่อง")
    table.add_column("เครื่อง")
    table.add_column("สายเร็วสุด")
    table.add_column("cluster IP")
    table.add_column("stacked ได้")

    def add_row(name: str, host: dict, cluster_ip: str, stack: bool = True) -> None:
        fabric = host.get("fabric") or {}
        best = fabric.get("best_gbps")
        check = check_cluster_ip(host, cluster_ip)
        ready = stack_ready(host)
        # ผู้ใช้สั่งไม่เอาเข้ากลุ่ม = คนละเรื่องกับ "เครื่องไม่พร้อม" ต้องอ่านแล้วแยกออกทันที
        if not stack:
            verdict = "[dim]ไม่เอาเข้ากลุ่ม (ตั้งไว้เอง)[/dim]"
        elif ready:
            verdict = "[green]ได้[/green]"
        else:
            verdict = f"[yellow]ไม่ได้[/yellow] {cluster_note(host)[:40]}"
        table.add_row(name, f"{best}G" if best else "—", cluster_ip or "[yellow]—[/yellow]", verdict)
        if stack and check["state"] in {"mismatch", "slow", "link-local"}:
            table.add_row("", "", "", f"[yellow]{check['message']}[/yellow]")
        warning = check.get("warning")
        if stack and warning and warning["kind"] == "under-negotiated":
            table.add_row("", "", "", (
                f"[yellow]{check['iface']} ขึ้นแค่ {warning['speed_gbps']}G — ConnectX ของ Spark "
                f"ควรได้ >={warning['expected_gbps']}G · ตั้ง port speed ที่ switch เป็น 200G เอง "
                f"(auto-negotiate มักลงมาเหลือ 50G)[/yellow]"
            ))

    add_row(local_name + " (hub)", local, machines[0]["cluster_ip"], stack_self)
    # ลำดับเดียวกับที่ลากจัดไว้ในหน้าเว็บ — สมาชิกตัวแรกของกลุ่มคือเครื่องที่ถูกเสนอเป็น head
    for node in in_saved_order(load(), settings.ui.node_order):
        try:
            host = probe(node).get("host") or {}
        except NodeError as exc:
            table.add_row(node.name, "—", node.cluster_ip or "—", f"[red]ต่อไม่ได้[/red] {str(exc)[:40]}")
            continue
        machines.append({"name": node.name, "host": host, "cluster_ip": node.cluster_ip,
                         "site": node.site, "cluster_name": node.cluster_name,
                         "stack": node.stack})
        add_row(node.name, host, node.cluster_ip, node.stack)

    console.print(table)
    opted_out = [m["name"] for m in machines if m.get("stack") is False]
    if opted_out:
        console.print(f"[dim]ไม่เอาเข้ากลุ่มตามที่ตั้งไว้: {', '.join(opted_out)} — "
                      f"เปิดคืนด้วย `lmds node set <ชื่อ> --stack` "
                      f"(เครื่องนี้เอง: `lmds node cluster --self-stack`)[/dim]")

    groups = cluster_groups(machines)
    if not groups:
        console.print(
            "\n[dim]ยังไม่มีคู่ที่ stacked ได้ — ต้องมีอย่างน้อย 2 เครื่องที่ GPU รุ่นเดียวกัน "
            "จำนวนเท่ากัน มีสายเร็วอย่างน้อย 25G ที่ตั้ง IP จริงแล้ว (ไม่ใช่ 169.254.x.x) "
            "และ IP นั้นต้องอยู่ subnet เดียวกัน[/dim]"
        )
        return

    console.print("\n[bold]กลุ่มที่ stacked ด้วยกันได้[/bold]")
    for group in groups:
        names = " + ".join(m["name"] for m in group["members"])
        mark = "[green]พร้อม[/green]" if group["ready"] else "[yellow]ยังไม่พร้อม[/yellow]"
        note = group["parallelism"]
        parallel = (f"TP={note['world_size']}" if note["kind"] == "tensor-parallel"
                    else f"TP={note['largest_tp']} + pipeline (TP={note['world_size']} หาร head ไม่ลง)")
        usable = group.get("usable_world_size")
        usable_text = (f" · ตั้ง cluster IP ครบแล้ว {usable}"
                       if usable is not None and usable != group["world_size"] else "")
        console.print(
            f"  {mark} {names} — {group['gpu']} x{group['gpus_per_node']}/เครื่อง · "
            f"world size {group['world_size']} ({parallel}){usable_text} · {group['link_gbps']}G "
            f"{'RDMA' if group['rdma'] else 'ethernet'}"
        )
        # ฮาร์ดแวร์ตรงกันแต่เข้ากลุ่มไม่ได้ — ไม่ใช่ blocker แต่ต้องบอก ไม่งั้นดูเหมือนเครื่องหาย
        for gone in group.get("excluded", []):
            text = (f"{gone['name']} คือเครื่องเดียวกับ {gone.get('same_as')} ที่ถูกเพิ่มไว้สองชื่อ — นับครั้งเดียว"
                    if gone["reason"] == "same-machine"
                    else f"{gone['name']} ฮาร์ดแวร์ตรงกัน แต่ไม่มี subnet ร่วมกับกลุ่มนี้ — ไม่ถูกนับใน world size")
            console.print(f"    [dim]· {text}[/dim]")
        # เตือน ≠ บล็อก · กลุ่มยังพร้อม แต่ถ้าไม่พูดตรงนี้จะไม่มีใครรู้ว่าวิ่งไม่เต็มสาย
        for warning in group.get("warnings", []):
            if warning["kind"] == "under-negotiated":
                console.print(
                    f"    [yellow]· สายวิ่งแค่ {warning['speed_gbps']}G "
                    f"(ควรได้ >={warning['expected_gbps']}G): {', '.join(warning['names'])} — "
                    f"stacked ได้อยู่ แต่ช้ากว่าที่ควร ตรวจ port speed ที่ switch[/yellow]"
                )
        for blocker in group["blockers"]:
            names = ", ".join(blocker["names"])
            text = {
                "missing-ip": "ยังไม่ได้ตั้ง cluster IP: ",
                "duplicate-ip": "cluster IP ซ้ำกันระหว่างเครื่อง: ",
                "split-fabric": "cluster IP อยู่คนละวง ต่อกันไม่ติด: ",
            }[blocker["kind"]] + names
            console.print(f"    [yellow]· {text}[/yellow]")
        for member in group["members"]:
            if member["state"] == "unset" and member["suggested_ip"]:
                console.print(
                    f"    [dim]lmds node set {member['name']} --cluster-ip {member['suggested_ip']}[/dim]"
                )

    if write:
        _write_cluster_env(write, groups, head or on or local_name, worker, on)
    elif any(g["ready"] for g in groups):
        console.print(
            "\n[dim]ให้ controller ใช้ค่าเหล่านี้เลย: "
            "lmds node cluster --write <slug> --on <เครื่องที่มี bundle>[/dim]"
        )


def _write_cluster_env(slug, groups, head_name, worker_name, on_node=None) -> None:
    """เขียน cluster.env ลง bundle — controller จะ source ไฟล์นี้แทนการถาม IP ตอน start

    ตรรกะทั้งหมดอยู่ที่ `lmds.fleet.cluster_env` เพราะหน้าเว็บต้องเขียนไฟล์เดียวกันนี้ได้ด้วย
    ที่นี่เหลือแค่แปลง error เป็นข้อความบนเทอร์มินัลกับ exit code
    """
    from lmds.fleet.cluster_env import ClusterEnvError, write_cluster_env

    try:
        result = write_cluster_env(slug, groups, head_name, worker_name, on_node)
    except ClusterEnvError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[green]เขียน {result.target} แล้ว[/green]")
    console.print(f"[dim]head {result.head_ip} · worker {' '.join(result.worker_ips)}"
                  f" · {result.nnodes} เครื่อง (TP={result.nnodes})"
                  f"{' · NCCL ' + result.iface if result.iface else ''}[/dim]")
    console.print("[dim]controller จะใช้ค่านี้เองตอน start — ไม่ถาม IP ซ้ำ[/dim]")


# ── lmds cluster … ──────────────────────────────────────────────────────────
def _live_cluster_groups(names: set[str] | None = None) -> tuple[dict, dict, dict, list]:
    """(ทะเบียน, host payload, error, กลุ่ม) จากการต่อเครื่องจริง — ชุดเดียวกับที่หน้า Cluster ใช้

    `names` = ต่อเฉพาะเครื่องเหล่านี้ (หมอ/pair สนใจแค่คู่เดียว ไม่ต้องรอทั้งฟลีต)
    """
    from lmds.config import Settings
    from lmds.nodes import NodeError, cluster_groups, in_saved_order, load, probe

    nodes = {n.name: n for n in in_saved_order(load(), Settings.load().ui.node_order)}
    hosts: dict[str, dict | None] = {}
    errors: dict[str, str] = {}
    machines = []
    for node in nodes.values():
        if names is not None and node.name not in names:
            continue
        try:
            host = probe(node).get("host") or {}
        except NodeError as exc:
            hosts[node.name], errors[node.name] = None, str(exc)
            continue
        hosts[node.name] = host
        machines.append({"name": node.name, "host": host, "cluster_ip": node.cluster_ip,
                         "site": node.site, "cluster_name": node.cluster_name, "stack": node.stack})
    return nodes, hosts, errors, cluster_groups(machines)


def _require_nodes(*names: str):
    from lmds.nodes import find

    found = []
    for name in names:
        node = find(name)
        if node is None:
            err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
            raise typer.Exit(code=1)
        found.append(node)
    return found


def _print_pair_steps(steps: list[dict]) -> bool:
    for step in steps:
        mark = "[green]✓[/green]" if step["ok"] else "[red]✕[/red]"
        console.print(f"  {mark} {step['step']}" + (f" — {step['detail']}" if step.get("detail") and not step["ok"] else ""))
    return bool(steps) and all(s["ok"] for s in steps)


def _pair_cluster_ssh(head, workers) -> bool:
    """กุญแจ head → worker ทุกตัว — พิมพ์ผลทีละขั้น คืน True เมื่อครบ"""
    from lmds.nodes import NodeError, run
    from lmds.nodes.cluster_ssh import pair_workers

    missing = [w.name for w in workers if not w.cluster_ip]
    if missing:
        err_console.print(f"[red]{', '.join(missing)} ยังไม่มี cluster IP[/red] — head ต้องเข้า worker ทางที่อยู่นั้น: "
                          f"lmds node set <ชื่อ> --cluster-ip <ip>")
        return False
    console.print(f"กุญแจ {head.name} → {', '.join(w.name for w in workers)} (สร้างบน head ไม่ผ่านเครื่องนี้)")
    try:
        steps = pair_workers(head, [(w, w.cluster_ip) for w in workers], runner=run)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        return False
    return _print_pair_steps(steps)


@cluster_app.command("show")
def cluster_show(
    self_stack: Optional[bool] = typer.Option(
        None, "--self-stack/--no-self-stack",
        help="เอา 'เครื่องนี้' (hub) เข้ากลุ่ม stacked ด้วยไหม — จำค่าไว้ใน config"),
) -> None:
    """เครื่องไหนจับคู่ stacked กันได้บ้าง (เหมือน `lmds node cluster`)"""
    node_cluster(write=None, worker=None, head=None, on=None, self_stack=self_stack)


@cluster_app.command("write")
def cluster_write_cmd(
    slug: str = typer.Argument(..., help="ชื่อ bundle", autocompletion=_complete_slug),
    head: str = typer.Option(..., "--head", help="เครื่องที่จะเป็น head", autocompletion=_complete_node),
    worker: list[str] = typer.Option([], "--worker", help="worker เรียงตาม rank (ซ้ำได้หลายตัว · ว่าง = ตามกลุ่ม)"),
    on: Optional[str] = typer.Option(None, "--on", metavar="NODE",
                                     help="เขียนลง bundle บนเครื่องนั้น (ว่าง = head)"),
    nnodes: Optional[int] = typer.Option(None, "--nnodes", help="จำนวนเครื่อง (ว่าง = อ่านจาก bundle บนเครื่องนี้)"),
) -> None:
    """เขียน cluster.env ของ bundle ให้ตรงกับจำนวนเครื่องที่ bundle ถูก render มา"""
    from lmds.fleet import bundle_roots
    from lmds.fleet.cluster_env import ClusterEnvError, write_cluster_env
    from lmds.nodes.stacked import StackedError, bundle_node_count, select_members

    _require_nodes(head, *worker)
    if nnodes is None:
        local = next((r / slug for r in bundle_roots() if (r / slug).is_dir()), None)
        nnodes = bundle_node_count(local) if local is not None else None
        if nnodes == 1:
            nnodes = None      # ไม่ใช่ stacked บนเครื่องนี้ — เขียนตามทั้งกลุ่ม
    _, _, _, groups = _live_cluster_groups()
    try:
        trimmed = select_members(groups, head, workers=list(worker), nnodes=nnodes)
        result = write_cluster_env(slug, [trimmed], head, None, on or head)
    except (StackedError, ClusterEnvError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]เขียน {result.target} แล้ว[/green]")
    console.print(f"[dim]head {result.head_ip} · worker {' '.join(result.worker_ips)} "
                  f"({', '.join(m['name'] for m in trimmed['members'][1:])}) · {result.nnodes} เครื่อง"
                  f"{' · NCCL ' + result.iface if result.iface else ''}[/dim]")


@cluster_app.command("pair")
def cluster_pair_cmd(
    head: str = typer.Argument(..., autocompletion=_complete_node),
    workers: list[str] = typer.Argument(..., help="worker หนึ่งตัวหรือมากกว่า"),
) -> None:
    """ให้ head ssh เข้า worker ได้โดยไม่ถามรหัส — สิ่งที่ controller stacked ต้องใช้ตอน sync/start

    `lmds node setup` ติดตั้งแต่กุญแจของเครื่องนี้ (hub) · controller บน head ใช้กุญแจของ head เอง
    จึงตายด้วย Permission denied ที่ sync-worker/start มาตลอด · กุญแจถูกสร้างบน head ไม่ผ่านเครื่องนี้
    """
    nodes = _require_nodes(head, *workers)
    if not _pair_cluster_ssh(nodes[0], nodes[1:]):
        raise typer.Exit(code=1)
    console.print("[green]head เข้า worker ได้แล้ว[/green]")


@cluster_app.command("doctor")
def cluster_doctor_cmd(
    head: str = typer.Argument(..., autocompletion=_complete_node),
    worker: str = typer.Argument(..., autocompletion=_complete_node),
    slug: Optional[str] = typer.Option(None, "--slug", help="ตรวจ bundle บน head ด้วย (cluster.env ตรงทะเบียนไหม)"),
) -> None:
    """ทำไมคู่นี้ยังใช้ stacked ไม่ได้ — ทีละข้อพร้อมคำสั่งแก้ · อ่านอย่างเดียว ไม่แตะอะไรบนเครื่อง"""
    from lmds.nodes import run
    from lmds.nodes.doctor import describe, diagnose_pair

    nodes, hosts, errors, _ = _live_cluster_groups({head, worker})
    report = diagnose_pair(head, worker, nodes=nodes, hosts=hosts, errors=errors,
                           runner=run, slug=slug or "")
    console.print(f"[bold]{head} ⇄ {worker}[/bold]")
    for finding in report["findings"]:
        mark = {"pass": "[green]✓[/green]", "warn": "[yellow]![/yellow]"}.get(finding["level"], "[red]✕[/red]")
        console.print(f"  {mark} {describe(finding, 'th')}")
        if finding.get("fix"):
            console.print(f"      [dim]แก้: {finding['fix']}[/dim]")
    if report["ok"]:
        console.print("[green]คู่นี้พร้อมสำหรับ stacked[/green]")
    else:
        console.print("[red]ยังไม่พร้อม — แก้ตามข้อที่ ✕ ก่อน[/red]")
        raise typer.Exit(code=1)


# ── lmds cluster inspect / plan / apply / remove-net — ตั้งค่าสาย ConnectX จาก hub ─────────
def _print_net_findings(findings: list[dict]) -> None:
    from lmds.nodes.doctor import describe

    for finding in findings:
        mark = {"pass": "[green]✓[/green]", "warn": "[yellow]![/yellow]"}.get(finding["level"], "[red]✕[/red]")
        console.print(f"  {mark} {describe(finding, 'th')}")
        if finding.get("fix"):
            console.print(f"      [dim]แก้: {finding['fix']}[/dim]")


def _print_ports(inspection: dict) -> None:
    table = Table(title="พอร์ต QSFP (ConnectX-7)")
    for column in ("เครื่อง", "พอร์ต", "สาย", "ความเร็ว", "interface", "IP ปัจจุบัน", "RDMA"):
        table.add_column(column)
    for name, info in inspection["nodes"].items():
        if not info["reachable"]:
            table.add_row(name, "—", f"[red]ต่อไม่ได้: {info['error']}[/red]", "", "", "", "")
            continue
        if not info["ports"]:
            table.add_row(name, "—", "[red]ไม่พบพอร์ต ConnectX[/red]", "", "", "", "")
        for port in info["ports"]:
            table.add_row(
                name, str(port["port"]),
                "[green]มี[/green]" if port.get("carrier") else "[dim]ไม่มี (NO-CARRIER)[/dim]",
                f"{port['speed_gbps']}G" if port.get("speed_gbps") else "—",
                port.get("configured") or " / ".join(port.get("ifaces") or []),
                f"{port['ip']}/{port['prefix']}" if port.get("ip") else "[dim]ยังไม่ตั้ง[/dim]",
                ", ".join(port.get("rdma_devices") or []) or "—",
            )
        if info.get("nvidia_sync"):
            table.add_row("", "", "[yellow]มีไฟล์ของ NVIDIA Sync[/yellow]", "", "", "", "")
    console.print(table)


def _print_plan(plan: dict) -> None:
    if not plan.get("ok"):
        err_console.print(f"[red]วางแผนไม่ได้ — {plan.get('reason')}[/red]")
        return
    console.print(f"[bold]ผัง: {plan['topology']}[/bold] · วงเริ่มที่ {plan['base_subnet']} · "
                  f"ลำดับ {' → '.join(plan['order'])}")
    table = Table(title="ลิงก์")
    for column in ("ลิงก์", "วง", "ปลาย A", "ปลาย B"):
        table.add_column(column)
    for link in plan["links"]:
        ends = [f"{e['node']} p{e['port']} {e['iface']} {e['ip']}/{e['prefix']}"
                + ("" if e["changed"] else " (เดิม)") for e in link["ends"]]
        table.add_row(str(link["id"]), link["subnet"], ends[0], " · ".join(ends[1:]))
    console.print(table)
    for name, entry in plan["nodes"].items():
        console.print(f"[bold]{name}[/bold] → cluster_ip {entry['cluster_ip']} บน {entry['cluster_iface']}"
                      + ("" if entry["changed"] else " [dim](ไม่เปลี่ยน)[/dim]"))
    for warning in plan.get("warnings") or []:
        console.print(f"  [yellow]![/yellow] {warning}")


@cluster_app.command("inspect")
def cluster_inspect_cmd(
    nodes: list[str] = typer.Argument(..., help="เครื่อง 2–4 ตัว เรียงตามลำดับที่เสียบสาย (ตัวแรก = head)"),
    topology: Optional[str] = typer.Option(None, "--topology", help="บังคับผัง: direct | ring | switch"),
) -> None:
    """ดูว่าพอร์ต QSFP ไหนมีสาย ตั้ง IP อะไรอยู่ และเสียบเป็นผังไหน — อ่านอย่างเดียว"""
    from lmds.nodes.doctor import diagnose_network
    from lmds.nodes.netplan import inspect_nodes

    _require_nodes(*nodes)
    reg, hosts, errors, _ = _live_cluster_groups(set(nodes))
    inspection = inspect_nodes(list(nodes), hosts, errors, topology or "")
    _print_ports(inspection)
    report = diagnose_network(list(nodes), nodes=reg, hosts=hosts, errors=errors, topology=topology or "")
    _print_net_findings(report["findings"])
    inferred = inspection["topology"]
    if inferred["topology"] == "unknown":
        err_console.print(f"[red]ผัง: ไม่รู้จัก — {inferred['reason']}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]ผัง: {inferred['topology']}[/green] · ต่อไป: lmds cluster plan {' '.join(nodes)}")


@cluster_app.command("plan")
def cluster_plan_cmd(
    nodes: list[str] = typer.Argument(..., help="เครื่อง 2–4 ตัว เรียงตามลำดับที่เสียบสาย (ตัวแรก = head)"),
    subnet: str = typer.Option("10.100.152.0/24", "--subnet", help="วงแรก — ลิงก์ถัดไปได้วงถัดไป (.153, .154)"),
    topology: Optional[str] = typer.Option(None, "--topology", help="บังคับผัง: direct | ring | switch"),
    as_json: bool = typer.Option(False, "--json", help="พิมพ์แผนเป็น JSON (ส่งต่อให้ apply/หน้าเว็บได้)"),
) -> None:
    """คำนวณ IP + netplan ต่อเครื่องจากสายที่เสียบอยู่ — ยังไม่แตะเครื่อง"""
    from lmds.nodes.netplan import NetplanError, build_plan

    _require_nodes(*nodes)
    reg, hosts, _errors, _ = _live_cluster_groups(set(nodes))
    try:
        plan = build_plan(list(nodes), hosts, base_subnet=subnet, topology=topology or "", nodes=reg)
    except NetplanError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if as_json:
        console.print_json(json.dumps(plan, ensure_ascii=False))
    else:
        _print_plan(plan)
        for name, entry in plan.get("nodes", {}).items():
            console.print(f"[dim]── {name}: {'/etc/netplan/99-lmds-cluster.yaml'} ──[/dim]")
            console.print(entry["netplan"].rstrip(), highlight=False, markup=False)
    if not plan.get("ok"):
        raise typer.Exit(code=1)


@cluster_app.command("apply")
def cluster_apply_cmd(
    nodes: list[str] = typer.Argument(..., help="เครื่อง 2–4 ตัว เรียงตามลำดับที่เสียบสาย (ตัวแรก = head)"),
    subnet: str = typer.Option("10.100.152.0/24", "--subnet", help="วงแรก — ลิงก์ถัดไปได้วงถัดไป"),
    topology: Optional[str] = typer.Option(None, "--topology", help="บังคับผัง: direct | ring | switch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="ไม่ถามยืนยันแผน (ยังถามรหัส sudo)"),
    pair: bool = typer.Option(True, "--pair/--no-pair", help="ตั้งกุญแจ head→worker หลัง IP ขึ้น"),
    speed_test: bool = typer.Option(True, "--speed-test/--no-speed-test", help="วัดด้วย iperf3 ถ้ามีทั้งสองฝั่ง"),
) -> None:
    """เขียน /etc/netplan/99-lmds-cluster.yaml บนทุกเครื่องตามแผน แล้วยืนยันด้วย ping + กุญแจ head→worker

    ถามรหัส sudo ทีละเครื่อง (ไม่เก็บ ไม่โผล่ใน argv) · ล้มระหว่างทาง = ถอยกลับไปไฟล์เดิมของเครื่องนั้น
    """
    import getpass

    from lmds.nodes import run
    from lmds.nodes.netplan import NetplanError, apply_plan, build_plan

    _require_nodes(*nodes)
    reg, hosts, _errors, _ = _live_cluster_groups(set(nodes))
    try:
        plan = build_plan(list(nodes), hosts, base_subnet=subnet, topology=topology or "", nodes=reg)
    except NetplanError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    _print_plan(plan)
    if not plan.get("ok"):
        raise typer.Exit(code=1)
    if not yes and not typer.confirm("เขียน netplan ตามแผนนี้ลงทุกเครื่อง?", default=False):
        raise typer.Exit(code=1)
    passwords = {}
    for name in plan["order"]:
        passwords[name] = getpass.getpass(
            f"รหัส sudo ของ {name} ({reg[name].user}) — เว้นว่างถ้าเครื่องนั้น sudo ไม่ถามรหัส: ")

    def show(step: dict) -> None:
        mark = {"pass": "[green]✓[/green]", "warn": "[yellow]![/yellow]"}.get(step["level"], "[red]✕[/red]")
        console.print(f"  {mark} [{step['node'] or 'hub'}] {step['step']}"
                      + (f" — {step['detail']}" if step.get("detail") else ""))

    result = apply_plan(plan, passwords, nodes=reg, runner=run, progress=show, pair=pair, speed_test=speed_test)
    passwords.clear()
    if result["ok"]:
        console.print("[green]คลัสเตอร์พร้อม — IP ขึ้นครบ ping ถึงกัน กุญแจ head→worker ใช้ได้[/green]")
        console.print(f"[dim]ต่อไป: lmds cluster doctor {plan['order'][0]} {plan['order'][1]}[/dim]")
    elif result["applied"]:
        err_console.print("[yellow]IP ขึ้นครบแล้ว แต่ ping/กุญแจยังไม่ผ่านทุกข้อ — ดูบรรทัด ✕ ข้างบน[/yellow]")
        raise typer.Exit(code=1)
    else:
        err_console.print("[red]ตั้งค่าไม่สำเร็จ — เครื่องที่ล้มถูกถอยกลับไปไฟล์เดิมแล้ว (ถ้าถอยได้)[/red]")
        raise typer.Exit(code=1)


@cluster_app.command("remove-net")
def cluster_remove_net_cmd(
    node: str = typer.Argument(..., autocompletion=_complete_node),
) -> None:
    """ถอนไฟล์ netplan ของ LMDS บนเครื่องนั้น (ย้ายไป /root/netplan-disabled แบบ NVIDIA) และล้าง cluster_ip ในทะเบียน"""
    import getpass

    from lmds.nodes import run
    from lmds.nodes.netplan import remove_net

    (target,) = _require_nodes(node)
    password = getpass.getpass(f"รหัส sudo ของ {node} ({target.user}) — เว้นว่างถ้าเครื่องนั้น sudo ไม่ถามรหัส: ")
    result = remove_net(target, password, runner=run)
    password = ""
    for step in result["steps"]:
        mark = "[green]✓[/green]" if step["ok"] else "[red]✕[/red]"
        console.print(f"  {mark} {step['step']}" + (f" — {step['detail']}" if step.get("detail") else ""))
    if not result["ok"]:
        raise typer.Exit(code=1)
    console.print("[green]ถอนแล้ว[/green]" if result.get("removed") else "[dim]ไม่มีไฟล์ของ LMDS อยู่แล้ว[/dim]")


@node_app.command("clone")
def node_clone(
    slug: str = typer.Argument(..., help="ชื่อ bundle ที่จะทำสำเนา", autocompletion=_complete_slug),
    source: str = typer.Option(..., "--from", "-f", metavar="NODE",
                               help="เครื่องที่มีโมเดลอยู่แล้ว", autocompletion=_complete_node),
    target: str = typer.Option(..., "--to", "-t", metavar="NODE",
                               help="เครื่องที่อยากได้สำเนา", autocompletion=_complete_node),
    verify: bool = typer.Option(True, "--verify/--no-verify",
                                help="ตรวจ SHA-256 ที่ปลายทางหลังคัดลอกเสร็จ"),
    start: bool = typer.Option(False, "--start", help="สั่ง start ต่อทันทีถ้าตรวจผ่าน"),
    dry_run: bool = typer.Option(False, "--dry-run", help="ดูว่าจะคัดลอกอะไรบ้างโดยไม่แตะของจริง"),
    yes: bool = typer.Option(False, "--yes", "-y", help="ข้ามการถามยืนยัน"),
) -> None:
    """ทำสำเนาโมเดลจากเครื่องที่รันได้แล้ว ไปเครื่องอื่น — ไม่ต้องโหลดจาก Hugging Face ใหม่

    ใช้ตอนอยากได้ตัวสำรอง/กระจายโหลดหลายเครื่องในไซต์เดียวกัน · ไฟล์วิ่ง **ตรง**
    จากต้นทางไปปลายทางบนสายเร็วที่สุดที่ทั้งคู่มี ไม่ผ่าน hub และไม่ผ่านอินเทอร์เน็ต
    """
    from lmds.fleet.clone import (
        CloneError, build_rsync_command, inspect_source, make_marker, plan_clone,
        revoke_temp_key, _install_temp_key,
    )
    from lmds.nodes import NodeError, find, stream

    try:
        plan = inspect_source(plan_clone(slug, source, target))
    except CloneError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    gb = plan.total_bytes / (1024 ** 3)
    table = Table(title=f"clone {slug}: {source} → {target}")
    table.add_column("ไฟล์")
    table.add_column("ขนาด", justify="right")
    for name, size in sorted(plan.files, key=lambda f: -f[1])[:10]:
        table.add_row(name, f"{size / 1024 ** 3:.2f} GB")
    if len(plan.files) > 10:
        table.add_row(f"… อีก {len(plan.files) - 10} ไฟล์", "")
    console.print(table)
    console.print(f"[dim]รวม {gb:.1f} GB · เส้นทาง "
                  f"{plan.source_addr} → {plan.target_addr} "
                  f"({'สายคลัสเตอร์' if plan.link == 'cluster' else 'เส้นปกติ'})[/dim]")
    if not plan.same_site:
        console.print("[yellow]สองเครื่องนี้ไม่ได้อยู่ไซต์เดียวกัน — ข้อมูลอาจวิ่งข้ามเน็ตนอก[/yellow]")

    if dry_run:
        console.print("[dim]--dry-run: ยังไม่แตะเครื่องปลายทาง[/dim]")
        raise typer.Exit(code=0)
    if not yes and not typer.confirm(f"คัดลอก {gb:.1f} GB ไป {target} เลยไหม", default=True):
        raise typer.Exit(code=1)

    # กุญแจชั่วคราวสำหรับงานนี้ครั้งเดียว — hub ไม่เคยส่ง key ของตัวเองให้ node
    marker = make_marker()
    target_node = find(target)
    source_node = find(source)
    key_dir = tempfile.mkdtemp(prefix="lmds-clone-")
    key_file = os.path.join(key_dir, "id")
    try:
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", marker,
                        "-f", key_file], check=True)
        private = pathlib.Path(key_file).read_text(encoding="utf-8")
        public = pathlib.Path(key_file + ".pub").read_text(encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        err_console.print(f"[red]สร้างกุญแจชั่วคราวไม่ได้: {exc}[/red]")
        raise typer.Exit(code=1)
    finally:
        # อยู่บนดิสก์ของ hub แค่ช่วงที่อ่านเข้าหน่วยความจำ ไม่นานกว่านั้น
        shutil.rmtree(key_dir, ignore_errors=True)

    code = 1
    installed = False
    try:
        _install_temp_key(target_node, public, marker)
        installed = True
        console.print(f"[dim]ฝากกุญแจชั่วคราวไว้ที่ {target} ({marker})[/dim]")
        proc = stream(source_node, build_rsync_command(plan, target_node.user),
                      stdin_text=private)
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            print(raw.decode("utf-8", "replace"), end="", flush=True)
        code = proc.wait()
    except (CloneError, NodeError) as exc:
        err_console.print(f"[red]{exc}[/red]")
    finally:
        # ถอนกุญแจเสมอ แม้ copy จะล้มกลางคัน — กุญแจชั่วคราวที่ค้างคือประตูที่เปิดทิ้งไว้
        if installed and not revoke_temp_key(target_node, marker):
            err_console.print(
                f"[yellow]ถอนกุญแจชั่วคราวออกจาก {target} ไม่สำเร็จ — "
                f"ลบบรรทัดที่มี {marker} ใน ~/.ssh/authorized_keys ของเครื่องนั้นเองด้วย[/yellow]")
        elif installed:
            console.print(f"[dim]ถอนกุญแจชั่วคราวออกจาก {target} แล้ว[/dim]")

    if code != 0:
        err_console.print(f"[red]คัดลอกไม่สำเร็จ (exit {code})[/red]")
        raise typer.Exit(code=code)

    console.print(f"[green]คัดลอก {gb:.1f} GB ไป {target} แล้ว[/green]")
    if verify:
        console.print(f"[dim]ตรวจ SHA-256 ที่ {target} …[/dim]")
        node_ctl(target, slug, ["verify-files"])
    if start:
        node_ctl(target, slug, ["start"])


# flag ของคำสั่งปลายทางต้องผ่านไปทั้งดุ้น — ไม่งั้น `node run x logs y -n 100`
# จะโดน typer กินไปเป็น option ของ node run เอง

def _run_detached(node, command: str, log_name: str, timeout: int = 7200, poll: int = 15) -> int:
    """รันคำสั่งยาว ๆ บน node แบบ *ไม่ผูกกับ SSH session* แล้วคอยอ่าน log จนจบ

    เดิม push --download / node ctl download รันผ่าน `run()` ตรง ๆ — controller บน node เป็น
    ลูกของ session SSH · พอ session หลุด (เน็ตสะดุด, hub ปิดเทอร์มินัล, timeout) controller
    ตายแต่ container ที่มันสั่งไว้ยังอยู่ **โดยไม่มีใครเฝ้า** · watchdog กันค้าง (Xet) อยู่ใน
    controller จึงตายไปด้วย · เคสจริง spark-worker 2026-09-03: container โหลด scottgl ค้างที่
    "Fetching 33 files 0%" rx 0 MB/s อยู่ 90 นาทีหลัง session หลุด ส่วน hub ก็ค้างที่
    "โหลด weight บน spark-worker…" ตลอด

    ตอนนี้: setsid+nohup บน node → เขียน log + บรรทัด __RC=<code> ตอนจบ → hub อ่าน log
    เป็นช่วง ๆ · session หลุดกลางทางก็สั่ง `lmds node ctl <n> <slug> download` ซ้ำได้ มันจะ
    เห็นว่ายังรันอยู่หรือจบแล้วจากไฟล์เดียวกัน
    """
    import time

    from lmds.nodes import run

    log = f"~/.lmds/run/{log_name}.detached.log"
    launch = (
        f"mkdir -p ~/.lmds/run && rm -f {log} && "
        f"(setsid nohup bash -lc {shlex.quote(command + '; echo __RC=$? >> ' + log)} "
        f"> {log} 2>&1 < /dev/null &) ; sleep 1; echo started"
    )
    started = run(node, launch, timeout=60)
    if not started.ok:
        err_console.print(started.stderr.rstrip()[:400])
        return started.exit_code or 1
    shown = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        probe = run(node, f"cat {log} 2>/dev/null", timeout=60)
        # ssh สะดุดหนึ่งรอบ (probe ล้ม / stdout ว่าง) ไม่ใช่ "log ถูกล้าง" — เดิม shown ถูกรีเซ็ตเป็น 0
        # แล้วรอบถัดไปพิมพ์ log ทั้งก้อนซ้ำตั้งแต่บรรทัดแรก (รีวิว 2026-09-04)
        if not probe.ok:
            continue
        text = probe.stdout or ""
        if len(text) < shown:
            continue
        # พิมพ์เฉพาะส่วนใหม่ — ผู้ใช้เห็นความคืบหน้าเหมือนรันสด
        fresh = text[shown:]
        rc_at = fresh.find("__RC=")
        if rc_at != -1:
            print(fresh[:rc_at], end="")
            return int(fresh[rc_at + 5:].strip().splitlines()[0] or 1)
        print(fresh, end="", flush=True)
        shown = len(text)
    err_console.print(f"[yellow]เกิน {timeout}s — งานยังรันอยู่บน node · ดูต่อ: lmds node run {node.name} logs[/yellow]")
    return 124

@node_app.command("push")
def node_push(
    name: str = typer.Argument(..., help="ชื่อเครื่องปลายทาง", autocompletion=_complete_node),
    slug: str = typer.Argument(..., help="ชื่อ bundle ในเครื่องนี้", autocompletion=_complete_slug),
    start: bool = typer.Option(False, "--start", help="สั่ง start ต่อทันทีหลังส่งถึง"),
    download: bool = typer.Option(False, "--download", help="สั่งโหลด weight ต่อทันที (ใช้เวลานาน)"),
) -> None:
    """ส่ง bundle ที่สร้างไว้ในเครื่องนี้ไปติดตั้งบนเครื่องอื่น

    ใช้ตอนที่เครื่องที่คุณนั่งอยู่ไม่ใช่เครื่องที่จะรันโมเดล (เช่น controller ที่ไม่มี GPU)
    — สร้าง bundle ที่นี่ ตรวจแผนที่นี่ แล้วส่ง **ตัวเดียวกันนั้น** ไปรันที่เครื่องเป้าหมาย

    ต่างจาก `lmds node run <name> deploy ...` ตรงที่อันนั้นสั่งให้เครื่องปลายทาง**วางแผนใหม่เอง**
    ซึ่งอาจได้คนละค่ากับที่คุณเพิ่งอนุมัติไป
    """
    from lmds.fleet import bundle_roots
    from lmds.nodes import NodeError, find, push_file, run

    node = find(name)
    if node is None:
        err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
        raise typer.Exit(code=1)

    zips = [root / f"{slug}.zip" for root in bundle_roots()]
    archive = next((z for z in zips if z.is_file()), None)
    # zip ถูกสร้างตอน generate — ค่าที่ `lmds set` เขียนทีหลัง (bundle.env / bundle.args)
    # ไม่เคยไปถึงเครื่องปลายทาง · เคสจริง 2026-09-03: ตั้ง --engine-env ให้ Coder-Next แล้ว push
    # ไป spark-head container ขึ้นมาโดยไม่มี env สักตัว และ bundle.args ของ Sehyo ไม่ถึง
    # spark-worker ทั้งที่ hub บอกว่าบันทึกแล้ว · แพ็กใหม่จากโฟลเดอร์ทุกครั้งก่อนส่ง
    if archive is not None and archive.with_suffix("").is_dir():
        from lmds.packager.bundle import make_zip
        archive = make_zip(archive.with_suffix(""))
    if archive is None:
        err_console.print(f"[red]ไม่พบ {slug}.zip ในเครื่องนี้[/red] — ดูรายชื่อ: lmds list")
        err_console.print(f"[dim]ที่ค้นหา: {', '.join(str(z) for z in zips)}[/dim]")
        raise typer.Exit(code=1)

    size_mb = archive.stat().st_size / 1024**2
    console.print(f"ส่ง [bold]{archive.name}[/bold] ({size_mb:.1f} MB) → {name}…")
    try:
        result = push_file(node, str(archive), f"/tmp/{slug}.zip")
        if not result.ok:
            err_console.print(f"[red]ส่งไฟล์ไม่สำเร็จ[/red] {result.stderr.strip()[:300]}")
            raise typer.Exit(code=1)
        # แตกไฟล์ลง ~/bundles บนเครื่องนั้น แล้วลบ zip ชั่วคราวทิ้ง
        unpack = (
            f"mkdir -p ~/bundles && cd ~/bundles && "
            f"unzip -oq /tmp/{shlex.quote(slug)}.zip && rm -f /tmp/{shlex.quote(slug)}.zip && "
            f"chmod +x ~/bundles/{shlex.quote(slug)}/*.sh 2>/dev/null; "
            f"ls -d ~/bundles/{shlex.quote(slug)}"
        )
        unpacked = run(node, unpack, timeout=300)
        if not unpacked.ok:
            err_console.print(f"[red]แตกไฟล์บน {name} ไม่สำเร็จ[/red] {unpacked.stderr.strip()[:300]}")
            raise typer.Exit(code=1)
        console.print(f"[green]ติดตั้งแล้วที่[/green] {unpacked.stdout.strip()}")

        # stacked: bundle ที่ไปถึง head ยังไม่รู้ IP ของคู่ตัวเอง (start จะถาม IP ซึ่งไม่มีใครตอบ
        # ใน session ที่ไม่มี tty) · head ไม่มีกุญแจไป worker · และ download ที่ head อย่างเดียว
        # ไม่ทำให้ worker มี weight — เดิม `push --download --start` จบด้วย "สำเร็จ" ที่ head แล้ว
        # ตายที่ worker ทุกครั้ง
        download_cmd = f"lmds repair {shlex.quote(slug)}"
        local_bundle = archive.with_suffix("")
        if local_bundle.is_dir() and any(local_bundle.glob("*-stacked.sh")):
            if not _prepare_stacked_push(node, slug, local_bundle) and (download or start):
                raise typer.Exit(code=1)
            controller = (
                f"dir=\"$(ls -d ~/bundles/{shlex.quote(slug)} 2>/dev/null | head -1)\" && cd \"$dir\" && "
                f"ctl=\"$(ls ./*-stacked.sh 2>/dev/null | head -1)\""
            )
            download_cmd = f"{download_cmd} && {controller} && \"$ctl\" sync-worker && \"$ctl\" verify-worker"

        for flag, command, note in ((download, download_cmd, "โหลด weight"),
                                    (start, f"lmds start {shlex.quote(slug)}", "สั่งรัน")):
            if not flag:
                continue
            console.print(f"{note} บน {name}… (รันแยกจาก session นี้ — หลุดกลางทางงานยังเดินต่อ)")
            rc = _run_detached(node, command, f"{slug}.{command.split()[1]}")
            if rc != 0:
                raise typer.Exit(code=rc)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[dim]ต่อจากนี้สั่งจากที่นี่ได้เลย: "
                  f"[bold]lmds node run {name} doctor {slug}[/bold] · "
                  f"[bold]lmds node run {name} start {slug}[/bold][/dim]")


def _prepare_stacked_push(head, slug: str, local_bundle) -> bool:
    """หลัง push bundle stacked ถึง head: เขียน cluster.env ตามกลุ่มจริง + กุญแจ head→worker

    คืน False เมื่อทำไม่ครบ (พิมพ์เหตุผลแล้ว) — ผู้เรียกตัดสินว่าจะไปต่อไหม
    """
    from lmds.fleet.cluster_env import ClusterEnvError, write_cluster_env
    from lmds.nodes import find
    from lmds.nodes.stacked import StackedError, bundle_node_count, select_members

    nnodes = bundle_node_count(local_bundle)
    console.print(f"bundle นี้เป็น stacked {nnodes} เครื่อง — หา worker ของ {head.name} จากทะเบียน…")
    _, _, _, groups = _live_cluster_groups()
    try:
        trimmed = select_members(groups, head.name, nnodes=nnodes)
        result = write_cluster_env(slug, [trimmed], head.name, None, head.name)
    except (StackedError, ClusterEnvError) as exc:
        err_console.print(f"[red]เขียน cluster.env ไม่ได้: {exc}[/red]")
        err_console.print("[dim]ดูเหตุผลทีละข้อ: lmds cluster doctor <head> <worker>[/dim]")
        return False
    workers = [find(m["name"]) for m in trimmed["members"][1:]]
    console.print(f"[green]เขียน {result.target}[/green] [dim]head {result.head_ip} · worker "
                  f"{' '.join(result.worker_ips)} · {result.nnodes} เครื่อง[/dim]")
    return _pair_cluster_ssh(head, [w for w in workers if w is not None])


@node_app.command("setup")
def node_setup(
    name: str = typer.Argument("", autocompletion=_complete_node,
                               help="ชื่อเครื่อง (เว้นว่างคู่กับ --all = ทุกเครื่อง)"),
    all_nodes: bool = typer.Option(False, "--all", help="ทำกับทุกเครื่องในทะเบียน"),
    with_prereq: bool = typer.Option(False, "--with-prereq",
                                     help="ติดตั้ง Docker / NVIDIA toolkit ด้วย (ใช้เวลานาน)"),
) -> None:
    """ตั้งค่าที่ต้องใช้สิทธิ์ root บนเครื่องปลายทาง — ถามรหัสผ่านตอนนี้ ใช้ครั้งเดียว

    ขั้นพวกนี้ทำผ่าน SSH ตามปกติไม่ได้เพราะ sudo ไม่มี tty ให้กรอกรหัส เดิมจึงได้แค่
    พิมพ์คำสั่งให้ไป ssh ทำเอง ซึ่งขัดกับเหตุผลที่มี hub ตั้งแต่แรก

    **รหัสผ่านไม่ถูกเก็บ** — ส่งทาง stdin ของ ssh ใช้ครั้งเดียวแล้วหายไปกับ process
    ไม่เขียนลงดิสก์ ไม่อยู่ใน argv และทะเบียน node ไม่มีฟิลด์ให้เก็บ
    """
    from lmds.nodes import NodeError, find, load, run_privileged

    targets = load() if all_nodes else ([find(name)] if name else [])
    targets = [n for n in targets if n is not None]
    if not targets:
        err_console.print("[red]ต้องระบุชื่อเครื่อง[/red] หรือใช้ --all")
        raise typer.Exit(code=1)

    console.print(f"ตั้งค่าที่ต้องใช้ root บน {len(targets)} เครื่อง — "
                  "[dim]รหัสผ่านใช้ครั้งเดียว ไม่ถูกเก็บที่ไหน[/dim]")
    password = typer.prompt("รหัสผ่าน sudo ของ user บนเครื่องนั้น", hide_input=True)

    failed = []
    for node in targets:
        console.print(f"\n[bold]{node.name}[/bold]")
        try:
            outcomes = run_privileged(node, password, with_prereq=with_prereq)
        except NodeError as exc:
            failed.append(node.name)
            err_console.print(f"  [red]{exc}[/red]")
            continue
        for outcome in outcomes:
            if outcome.get("skipped"):
                console.print(f"  [dim]• {outcome['step']} — เรียบร้อยอยู่แล้ว[/dim]")
            elif outcome["ok"]:
                console.print(f"  [green]✓[/green] {outcome['step']}")
            else:
                failed.append(node.name)
                err_console.print(f"  [red]✕[/red] {outcome['step']} — {outcome['detail']}")

    if failed:
        err_console.print(f"\n[red]ไม่สำเร็จ:[/red] {', '.join(sorted(set(failed)))}")
        err_console.print("[dim]รหัสผ่านผิด หรือ user นั้นไม่อยู่ในกลุ่ม sudo[/dim]")
        raise typer.Exit(code=1)
    console.print("\n[green]ตั้งค่าครบทุกเครื่องแล้ว[/green]")


@node_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def node_run(
    name: str = typer.Argument(..., autocompletion=_complete_node),
    command: list[str] = typer.Argument(..., help="คำสั่ง lmds ที่จะรันบนเครื่องนั้น เช่น: ps"),
) -> None:
    """รันคำสั่ง lmds บนเครื่องปลายทาง เช่น `lmds node run spark1 doctor my-model`"""
    from lmds.nodes import NodeError, find, run

    node = find(name)
    if node is None:
        err_console.print(f"[red]ไม่รู้จักเครื่อง '{name}'[/red] — ดู: lmds node list")
        raise typer.Exit(code=1)
    # quote ทุก argument — เดิม join ด้วยช่องว่างเฉย ๆ แล้ว shell ปลายทางแตกคำใหม่:
    # `lmds node run X set s --extra-args "--kv-cache-dtype fp8 --max-num-batched-tokens 8192"`
    # กลายเป็น 4 argument แยกกัน แล้ว typer ตอบ "No such option: --max-num-batched-tokens"
    # (เจอจริง RTX4000 2026-09-03) · node ctl ทำถูกอยู่แล้ว — ใช้กติกาเดียวกัน
    try:
        result = run(node, "lmds " + " ".join(shlex.quote(c) for c in command), timeout=900)
    except NodeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        err_console.print(result.stderr.rstrip())
    raise typer.Exit(code=result.exit_code)



def _same_commit(a: str, b: str) -> bool:
    """hash ย่อสองตัวชี้ commit เดียวกันไหม — เทียบแบบ prefix ไม่ใช่เท่ากันเป๊ะ

    git เลือกความยาว hash ย่อเองตามจำนวน object ของแต่ละเครื่อง: spark-head รายงาน `0ad1a59e`
    ส่วน hub ได้ `0ad1a59` (2026-09-04) · เทียบเท่ากันเป๊ะ = เครื่องที่เพิ่งอัปเดตสำเร็จถูกป้ายว่า
    "ยังไม่ตรง hub" ตลอดกาล · หน้าเว็บมี sameCommit อยู่แล้ว — CLI ต้องใช้กติกาเดียวกัน
    สั้นกว่า 7 ตัวไม่ฟันธง (ชนกันได้ง่ายเกิน)
    """
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def _hub_commit_note(commit: str, hub_commit: str) -> str:
    """ท้ายป้าย version ของ node: ตรง hub หรือยัง — ว่างเมื่อฝั่งใดฝั่งหนึ่งไม่รู้ commit"""
    if not commit or not hub_commit:
        return ""
    if _same_commit(commit, hub_commit):
        return " · [green]ตรง hub[/green]"
    return f" · [yellow]ยังไม่ตรง hub (hub: {hub_commit})[/yellow]"


def _version_label(version: str, commit: str = "") -> str:
    """'0.5.0 (f9181ab)' — เลข version อย่างเดียวบอกไม่ได้ว่า node ตามหลัง hub หรือยัง

    เคสจริง 2026-09-03: ทุก node โชว์ 0.5.0 เท่ากันหมดทั้งที่ 13 เครื่องอยู่ af01a1e และ hub
    อยู่ f9181ab (ต่างกัน 6 คอมมิตที่แก้บั๊ก restart/bundle.env) — ต้องไล่ `lmds node run
    <n> version` ทีละเครื่องถึงรู้ · host_payload ส่ง lmds_commit มาตลอด แค่ไม่มีใครแสดง
    """
    version = version or ""
    return f"{version} ({commit})" if version and commit else version

@app.command()
def version() -> None:
    """แสดงเวอร์ชันโปรแกรมและมาตรฐาน template"""
    from .banner import CREDIT

    from lmds.inventory import source_commit

    # เลข version ไม่ขยับทุกคอมมิต — ถามว่า "ฟลีตเท่ากันหรือยัง" ต้องดู commit
    commit = source_commit()
    console.print(f"lmds {lmds.__version__}" + (f"  ({commit})" if commit else ""))
    console.print(f"template standard: {lmds.TEMPLATE_STANDARD}")
    console.print(f"[dim]{CREDIT}[/dim]")


@app.command("set")
def set_defaults(
    slug: str = typer.Argument(..., help="ชื่อ bundle", autocompletion=_complete_slug),
    port: Optional[int] = typer.Option(None, "--port", help="พอร์ตที่จะเสิร์ฟ"),
    context: Optional[int] = typer.Option(None, "--context", help="context (tokens)"),
    slots: Optional[int] = typer.Option(None, "--slots", help="จำนวน request พร้อมกัน"),
    bind: Optional[str] = typer.Option(None, "--bind", help="ที่อยู่ที่ผูก (0.0.0.0 / 127.0.0.1)"),
    gpu_util: Optional[float] = typer.Option(None, "--gpu-util", help="สัดส่วนหน่วยความจำของ vLLM"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="ชื่อที่ API เสิร์ฟออกไป"),
    image: Optional[str] = typer.Option(None, "--image", help="image ที่จะใช้แทนของ bundle"),
    engine_env: Optional[str] = typer.Option(
        None, "--engine-env",
        help='env ของ engine เอง เช่น "VLLM_NVFP4_GEMM_BACKEND=marlin" (หลายตัวคั่นด้วยช่องว่าง)'),
    tool_parser: Optional[str] = typer.Option(
        None, "--tool-parser", help="--tool-call-parser ของ vLLM/SGLang ที่จะใช้ทุกครั้งที่ start (เช่น qwen3_xml)"),
    reasoning_parser: Optional[str] = typer.Option(
        None, "--reasoning-parser", help="--reasoning-parser ที่จะใช้ทุกครั้งที่ start (เช่น qwen3)"),
    image_min_tokens: Optional[str] = typer.Option(
        None, "--image-min-tokens",
        help="--image-min-tokens ของ llama.cpp (vision): ตัวเลข เช่น 1024 สำหรับ Qwen-VL · "
             "auto = ใช้ค่าที่ฝังมากับ projector (Gemma-4 ต้องใช้ auto — เพดานแค่ 280)"),
    extra_args: Optional[str] = typer.Option(
        None, "--extra-args",
        help='แฟล็กเพิ่มของ engine เช่น \'--speculative-config {"method":"mtp","num_speculative_tokens":2}\' '
             '(คั่นด้วยช่องว่าง · JSON เขียนติดกันไม่มีช่องว่าง)'),
    auto: bool = typer.Option(
        False, "--auto",
        help="ให้ระบบเติม parser / image / env ตามโมเดล (สูตรที่รันผ่านจริง > กฎตระกูล) · flag ที่ระบุเองชนะ"),
    clear: bool = typer.Option(False, "--clear", help="ลบค่าที่บันทึกไว้ทั้งหมด"),
) -> None:
    """บันทึกค่า start ไว้กับ bundle — ทุกทางที่เรียก controller จะได้ค่าเดียวกัน

    ค่าที่ใส่ตอน `lmds start --port …` มีผลครั้งนั้นครั้งเดียว · systemd ตอน autostart
    และคำสั่งอย่าง `test-text` เรียก controller เปล่า ๆ จึงตกไปใช้ค่าเริ่มต้นของ bundle
    — พอ reboot โมเดลหลายตัวบนเครื่องเดียวกันก็ไปชนกันที่พอร์ตเดียว

    คำสั่งนี้เขียน `bundle.env` ไว้ข้าง controller ซึ่ง controller อ่านก่อนตั้ง default
    ทุกตัว env จากภายนอกและ flag บรรทัดคำสั่งยังชนะไฟล์นี้เสมอ

    ไม่เก็บ API key — โฟลเดอร์ bundle ถูก zip แจกต่อได้ ส่ง `API_KEY=` ตอน start แทน
    """
    from pathlib import Path as _Path

    from lmds.fleet import find
    from lmds.fleet.bundle_settings import (
        SettingsError,
        ensure_controller_reads,
        read,
        write,
    )

    server = find(slug)
    if server is None or not server.controller:
        err_console.print(f"[red]ไม่รู้จัก '{slug}'[/red] — ดู: lmds ps")
        raise typer.Exit(code=1)
    bundle_dir = _Path(server.controller).parent
    # bundle ที่สร้างก่อนฟีเจอร์นี้ยังไม่รู้จัก bundle.env — เขียนไฟล์ไปก็ไม่มีใครอ่าน
    if ensure_controller_reads(_Path(server.controller)):
        console.print("[dim]เติมบรรทัดอ่าน bundle.env ให้ controller ตัวนี้แล้ว (สำรองไฟล์เดิมไว้)[/dim]")

    if clear:
        write(bundle_dir, {})
        console.print(f"ลบค่าที่บันทึกไว้ของ [bold]{slug}[/bold] แล้ว — กลับไปใช้ค่าของ bundle")
        return

    incoming = {
        "port": port, "context": context, "slots": slots, "bind": bind,
        "gpu_util": gpu_util, "served_name": model_id, "image": image,
        "engine_env": engine_env, "extra_args": extra_args,
        "tool_parser": tool_parser, "reasoning_parser": reasoning_parser,
        "image_min_tokens": image_min_tokens,
    }
    given = {k: v for k, v in incoming.items() if v is not None}
    if auto:
        # ความรู้มีอยู่แล้ว (recipes / arch_notes) แต่ผู้ใช้ต้องอ่านคำเตือนแล้วพิมพ์ชื่อเอง
        # ซึ่งคนไม่รู้ก็ไม่กล้าพิมพ์ (2026-09-04) — เติมให้ พร้อมบอกที่มาทีละค่า
        from lmds.fleet import bundle_profile
        from lmds.fleet.suggest import suggest_settings

        prof = bundle_profile(server.controller) or {}
        model = prof.get("model") or {}
        sug = suggest_settings(
            model.get("id") or server.model_id or "",
            (prof.get("runtime") or {}).get("engine") or server.engine or "",
            architecture=model.get("architecture") or "",
            quantization=model.get("quantization") or "",
            memory_model=(prof.get("target") or {}).get("memory_model") or "",
            projector=bool((((prof.get("features") or {}).get("multimodal") or {}).get("projector_files"))),
        )
        for note in sug["notes"]:
            console.print(f"[dim]{note}[/dim]")
        table = Table("ค่า", "เติมให้", "ที่มา", box=None)
        for key, value in sug["values"].items():
            table.add_row(key, value, sug["sources"].get(key, ""))
            given.setdefault(key, value)   # flag ที่ผู้ใช้ระบุเองชนะ
        console.print(table)
    if not given:
        current = read(bundle_dir)
        if not current:
            console.print(f"[dim]{slug} ยังไม่มีค่าที่บันทึกไว้ — ใช้ค่าของ bundle ทั้งหมด[/dim]")
            return
        table = Table("ค่า", "ที่บันทึกไว้", box=None)
        for key, value in current.items():
            table.add_row(key, value)
        console.print(table)
        return

    # เขียนทับทั้งไฟล์ — รวมของเดิมเข้ากับที่เพิ่งสั่ง เพื่อให้แก้ทีละค่าได้
    merged = {**read(bundle_dir), **{k: str(v) for k, v in given.items()}}
    try:
        saved = write(bundle_dir, merged)
    except SettingsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"บันทึกไว้กับ [bold]{slug}[/bold] แล้ว: " +
                  " · ".join(f"{k}={v}" for k, v in saved.items()))
    console.print("[dim]มีผลกับ start ทุกทาง รวมถึง autostart ตอน reboot และคำสั่ง test-*[/dim]")


@app.command()
def inspect(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision", help="branch/tag/commit ที่ต้องการ"),
    targets: list[str] = typer.Option(
        [], "--target", help="ประเมิน fit กับ target ที่ระบุ (ซ้ำได้) เช่น rtx-pro-4000 — ค่าว่าง = เครื่องนี้ + dgx-spark-single",
        autocompletion=_complete_target,
    ),
    concurrency: int = typer.Option(1, "--concurrency", help="จำนวน request พร้อมกันที่ใช้คำนวณ KV cache"),
    context: Optional[int] = typer.Option(
        None, "--context", help="ถามว่าค่านี้ควรตั้งไหม — บอกว่าได้กี่คนพร้อมกันและควรเลี่ยงอะไร"),
    kv_dtype: str = typer.Option(
        "bf16", "--kv-dtype", help="ชนิดของ KV cache ที่จะใช้ตอนรัน: bf16 | fp8"),
    as_json: bool = typer.Option(False, "--json", help="พิมพ์ผลเป็น JSON (สำหรับ scripting)"),
) -> None:
    """วิเคราะห์โมเดลจากลิงก์ — ดึงเฉพาะ metadata ไม่ดาวน์โหลด weight

    ถ้าเป็น gated repo และยังไม่มี token จะถาม (กด Enter เพื่อข้ามได้)
    Exit codes: 0 สำเร็จ, 1 input ผิด, 4 ต้องการ token, 5 ปัญหาเครือข่าย/Hub
    """
    _, report = _resolve_and_inspect(model, revision, interactive_ok=not as_json)
    fit_reports = _compute_fits(report, targets, concurrency)

    if as_json:
        import json as json_module

        payload = {
            "model": report.model_dump(mode="json"),
            "fit": [f.model_dump(mode="json") for f in fit_reports],
            "context_advice": _context_payload(report, fit_reports, context, kv_dtype),
        }
        print(json_module.dumps(payload, indent=2, ensure_ascii=False))
        return

    _render_report(report)
    _render_fits(fit_reports)
    _render_context(report, fit_reports, context, kv_dtype)


def _engine_choice(name):
    """แปลชื่อที่พิมพ์มาเป็น Engine — ชื่อผิดต้องหยุดตรงนี้ ไม่ใช่ไปโผล่เป็น bundle ที่ start ไม่ขึ้น"""
    from lmds.brain.plan_schema import Engine

    if not name:
        return None
    try:
        return Engine(name.strip().lower())
    except ValueError:
        allowed = ", ".join(e.value for e in Engine)
        err_console.print(f"[red]ไม่รู้จัก engine '{name}'[/red] — มีให้เลือก: {allowed}")
        raise typer.Exit(code=1) from None


def _build_plan_safe(report, fit, provider, engine=None):
    """เรียก LLM วางแผน — ถ้า provider ล้ม (quota/เครือข่าย/schema) สลับเป็น rule-based พร้อมแจ้งชัด"""
    from lmds.brain import PlanError, ProviderError, build_plan

    if provider is not None:
        try:
            return build_plan(report, fit, provider, engine=engine)
        except (PlanError, ProviderError) as exc:
            err_console.print(f"[yellow]LLM ใช้ไม่ได้: {exc}[/yellow]")
            err_console.print("[yellow]→ สลับเป็น rule-based mode อัตโนมัติ (plan จะไม่มีการวิเคราะห์เชิงลึก)[/yellow]")
    return build_plan(report, fit, None, engine=engine)


def _resolve_and_inspect(model: str, revision: Optional[str], interactive_ok: bool):
    """flow ร่วมของ inspect/plan/deploy: parse source → inspect → จัดการ gated repo + token"""
    import sys

    from lmds.inspector import (
        AuthRequired,
        BudgetExceeded,
        HfClient,
        HfError,
        RepoNotFound,
        inspect_model,
    )
    from lmds.resolver import SourceError, parse_source

    try:
        source = parse_source(model)
    except SourceError as exc:
        err_console.print(f"[red]ผิดพลาด:[/red] {exc}")
        raise typer.Exit(code=1)
    if revision:
        from dataclasses import replace

        source = replace(source, revision=revision)

    token = get_secret("hf")
    try:
        try:
            return source, inspect_model(source, HfClient(token=token))
        except AuthRequired as exc:
            interactive = sys.stdin.isatty() and interactive_ok
            if not interactive or exc.had_token:
                err_console.print(f"[red]{exc}[/red]")
                if not exc.had_token:
                    err_console.print("ตั้ง token ด้วย: lmds config set-hf-token หรือ env HF_TOKEN")
                raise typer.Exit(code=4)
            err_console.print(f"[yellow]{source.repo_id} เป็น gated repo[/yellow]")
            entered = typer.prompt("Hugging Face token (Enter เพื่อข้าม)", hide_input=True, default="").strip()
            if not entered:
                err_console.print("ข้าม token — ไม่สามารถ inspect repo นี้ได้")
                raise typer.Exit(code=4)
            report = inspect_model(source, HfClient(token=entered))
            # token ที่พิมพ์ตรงนี้ใช้ได้แค่รอบนี้ — controller อ่านจาก env HF_TOKEN เสมอ
            # (ไม่ฝัง secret ลง bundle) ถ้าไม่บอกให้ชัด ผู้ใช้จะไปเจอ 401 ตอน download
            err_console.print(
                "[yellow]โมเดลนี้เป็น gated — ตอน download ต้องมี token ด้วย "
                "(ค่าที่เพิ่งพิมพ์ใช้แค่ขั้นวิเคราะห์)[/yellow]"
            )
            err_console.print(
                "  เก็บถาวร:  [bold]lmds config set-hf-token[/bold]\n"
                "  หรือชั่วคราว:  [bold]export HF_TOKEN=hf_xxx[/bold]  ก่อนรัน ./<controller>.sh download"
            )
            return source, report
    except RepoNotFound as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except BudgetExceeded as exc:
        # ไม่ใช่ปัญหาเครือข่าย — ไฟล์ metadata ใหญ่เกินเพดานที่ตั้งไว้
        err_console.print(f"[red]ไฟล์ metadata ใหญ่ผิดปกติ:[/red] {exc}")
        err_console.print(
            "[yellow]ถ้าเป็นโมเดล MoE/quant ละเอียดที่ index ยาวจริง แจ้งทีมพัฒนาให้ปรับเพดาน "
            "(INDEX_FILE_CAP ใน inspector/hf_api.py)[/yellow]"
        )
        raise typer.Exit(code=5)
    except HfError as exc:
        err_console.print(f"[red]ปัญหาเครือข่าย/Hub:[/red] {exc}")
        raise typer.Exit(code=5)


def _pick_gguf_variant(variants, wanted: str):
    """หา variant จาก --gguf: ชื่อไฟล์เต็ม → ชื่อ quant (Q8_K_XL) → ส่วนของชื่อ · คืน (ตัวที่เจอ, ตัวที่ชนกัน)

    หน้าเว็บส่ง selected_gguf เป็นชื่อไฟล์เต็มอยู่แล้ว ส่วนคนที่พิมพ์เองจำได้แค่ quant — รับทั้งสองแบบ
    """
    key = (wanted or "").strip().lower()
    if not key:
        return None, []
    by_name = [v for v in variants
               if v.filename.lower() == key or v.filename.rsplit("/", 1)[-1].lower() == key]
    if by_name:
        return by_name[0], []
    by_quant = [v for v in variants if _quant_from_filename(v.filename).lower() == key]
    if len(by_quant) == 1:
        return by_quant[0], []
    if by_quant:
        return None, by_quant
    by_part = [v for v in variants if key in v.filename.lower()]
    if len(by_part) == 1:
        return by_part[0], []
    return None, by_part


def _reinspect_gguf(source, report, chosen):
    """inspect ซ้ำด้วยไฟล์ที่เลือก → ได้ GGUF header (architecture/context/kv dims) มาคำนวณ fit จริง"""
    from dataclasses import replace as dc_replace

    from lmds.inspector import HfClient, HfError, inspect_model

    try:
        return inspect_model(dc_replace(source, filename=chosen.filename), HfClient(token=get_secret("hf")))
    except HfError as exc:
        err_console.print(f"[yellow]อ่าน header ของไฟล์ที่เลือกไม่ได้ ({exc}) — ใช้ขนาดไฟล์อย่างเดียว[/yellow]")
        report.selected_gguf = chosen.filename
        report.weight_bytes = chosen.size_bytes
        return report


def _ensure_gguf_selected(source, report, interactive: bool, wanted: str = ""):
    """repo GGUF หลาย variant ที่ยังไม่เลือกไฟล์ — ให้เลือกตั้งแต่ต้น flow ไม่ใช่ไปพังตอนท้าย

    wanted (--gguf): ชื่อไฟล์หรือชื่อ quant ที่ระบุมา — script/hub เลือกได้โดยไม่ต้องมี tty
    interactive: แสดงรายการให้เลือกหมายเลข แล้ว inspect ซ้ำด้วยไฟล์ที่เลือก (ได้ header/kv dims จริง)
    non-interactive: จบพร้อมวิธีระบุไฟล์ (--gguf หรือลิงก์ตรง)
    """
    from lmds.inspector import ArtifactType

    weight_variants = [v for v in report.gguf_variants if not v.is_mmproj and not v.is_mtp]
    if report.artifact_type is not ArtifactType.GGUF:
        if wanted:
            err_console.print(f"[red]{report.repo_id} ไม่ใช่ repo GGUF — --gguf ใช้กับ repo นี้ไม่ได้[/red]")
            raise typer.Exit(code=1)
        return report

    if wanted:
        # เคสจริง 2026-09-04: hub สั่ง `lmds deploy` แบบ non-interactive กับ repo 22 variant แล้วได้ exit 1
        # "ต้องระบุไฟล์" ทั้งที่หน้าเว็บให้ผู้ใช้เลือกไฟล์ไว้แล้ว — ต้องมีทางส่งชื่อนั้นมาโดยไม่ต้องมี tty
        chosen, clash = _pick_gguf_variant(weight_variants, wanted)
        if chosen is None:
            pool = clash or sorted(weight_variants, key=lambda v: v.size_bytes or 0)
            err_console.print(
                f"[red]--gguf '{wanted}' "
                + ("ตรงกับหลายไฟล์ — ระบุให้เจาะจงกว่านี้:" if clash else f"ไม่ตรงกับไฟล์ไหนใน {report.repo_id}:")
                + "[/red]"
            )
            for variant in pool:
                size = f"{variant.size_bytes / 1e9:.1f} GB" if variant.size_bytes else "?"
                err_console.print(f"  • {variant.filename} ({size})")
            raise typer.Exit(code=1)
        if report.selected_gguf == chosen.filename:
            return report
        return _reinspect_gguf(source, report, chosen)

    if report.selected_gguf or len(weight_variants) <= 1:
        return report

    variants = sorted(weight_variants, key=lambda v: v.size_bytes or 0)
    if not interactive:
        err_console.print(
            f"[red]repo นี้มี GGUF {len(variants)} variant — ต้องระบุไฟล์ (โหมด non-interactive)[/red]"
        )
        # เดิมตัดที่ 8 ตัวแรก ซึ่งเรียงจากเล็กไปใหญ่ = เห็นแต่ควอนต์ต่ำสุด และไม่บอกว่าตัดมา
        # เคสจริง spark-02 (2026-09-03): repo มี 22 variant คนอยากได้ Q8 แต่รายการหยุดแค่
        # Q3_K_XL จึงดูเหมือน repo นี้ไม่มี Q8 ขาย
        for variant in variants:
            size = f"{variant.size_bytes / 1e9:.1f} GB" if variant.size_bytes else "?"
            err_console.print(f"  • {variant.filename} ({size})")
        # ตัวอย่างเดิมเป็น variants[0] = ตัวเล็กสุด (IQ1_M) ซึ่งคุณภาพต่ำสุดใน repo
        # การยกตัวกลาง ๆ มาเป็นตัวอย่างสะท้อนความตั้งใจจริงมากกว่า
        example = variants[len(variants) // 2]
        err_console.print(
            f"\nระบุไฟล์ด้วย --gguf (ชื่อไฟล์ หรือชื่อ quant) เช่น:\n"
            f"  lmds deploy {report.repo_id} --gguf {_quant_from_filename(example.filename) or example.filename}\n"
            f'หรือลิงก์ตรง:\n  lmds deploy "https://huggingface.co/{report.repo_id}/blob/main/{example.filename}"'
        )
        raise typer.Exit(code=1)

    table = Table(title=f"เลือกไฟล์ GGUF ({report.repo_id})")
    table.add_column("#")
    table.add_column("ไฟล์")
    table.add_column("ขนาด")
    for i, variant in enumerate(variants, 1):
        size = f"{variant.size_bytes / 1e9:.1f} GB" if variant.size_bytes else "?"
        table.add_row(str(i), variant.filename, size)
    console.print(table)

    choice = typer.prompt("เลือกหมายเลขไฟล์", type=int)
    if not 1 <= choice <= len(variants):
        err_console.print("[red]หมายเลขไม่ถูกต้อง[/red]")
        raise typer.Exit(code=1)
    return _reinspect_gguf(source, report, variants[choice - 1])


# สำนวนไทยของรหัสคำแนะนำ — หน้าเว็บมีสำนวนอังกฤษของตัวเอง และผู้ช่วย LLM ได้รหัส
# กับตัวเลขดิบไปเรียบเรียงเป็นภาษาที่ผู้ใช้พิมพ์มา · ที่เดียวกันสามภาษาไม่ได้
_ADVICE_TH = {
    "over-native": lambda f: (
        f"context {f['context']:,} เกิน native ของตัวโมเดล ({f['native']:,}) — "
        f"ตัวโมเดลรับไม่ได้ ไม่ใช่เรื่องหน่วยความจำ"),
    "over-memory": lambda f: (
        f"KV ที่ context {f['context']:,} ต้องใช้ {f['kv_gb']} GB "
        f"แต่เหลือแค่ {f['kv_budget_gb']} GB"),
    "single-user": lambda f: (
        f"ใส่ได้ แต่ได้ {f['concurrency']} คนพร้อมกัน — "
        f"หนึ่งคำสนทนากิน KV pool เกือบหมด คนที่สองต้องรอคิว"),
    "thin-margin": lambda f: (
        f"เหลือแค่ {f['spare_gb']} GB จาก {f['kv_budget_gb']} GB — "
        f"ไม่พอให้ CUDA graph, activation ของ chunked prefill และ NCCL buffer "
        f"ซึ่งไม่ได้อยู่ในงบนี้"),
    "fp8-would-help": lambda f: (
        f"เปลี่ยน KV เป็น fp8 (--kv-cache-dtype fp8_e5m2) → KV {f['kv_gb_now']} GB "
        f"เหลือ {f['kv_gb_fp8']} GB · พร้อมกันจาก {f['concurrency_now']} เป็น "
        f"{f['concurrency_fp8']} คน โดยไม่ต้อง quantize checkpoint ใหม่"),
    "room-to-grow": lambda f: (
        f"ยังขยับขึ้นเป็น {f['suggest']:,} ได้ และยังรับ {f['concurrency']} คนพร้อมกัน"),
    "odd-step": lambda f: (
        f"{f['context']:,} ไม่ใช่ค่ายกกำลังสองที่ engine ใช้เป็นขั้น — ตั้งได้ "
        f"แต่บางตัวปัดลงเอง"),
    "stacked-comms-unbudgeted": lambda f: (
        f"target {f['nodes']} เครื่อง — งบนี้ยังไม่รวม NCCL buffer ของ tensor parallel "
        f"ข้ามเครื่อง ให้เผื่อไว้"),
    "kv-dims-unknown": lambda f: "อ่านมิติ KV จาก config ไม่ได้ — คำนวณให้ไม่ได้",
    "weights-unknown": lambda f: "ยังไม่รู้ขนาด weight — คำนวณให้ไม่ได้",
}


def _context_payload(report, fit_reports: list, context: int | None, kv_dtype: str) -> list:
    """ตารางกับคำแนะนำในรูปข้อมูล — ใช้ทั้งฝั่งพิมพ์และฝั่ง --json"""
    from lmds.fit import advise, ladder

    out = []
    for fit in fit_reports:
        if report.kv_dims is None or fit.weights_gb is None:
            continue
        steps = ladder(fit, report.kv_dims, kv_dtype, report.context_length)
        asked = context or fit.recommended_context
        out.append({
            "target": fit.target_name,
            "kv_dtype": kv_dtype,
            "kv_bytes_per_token": steps[0].per_token_bytes if steps else None,
            "ladder": [
                {"context": p.context, "kv_gb": p.kv_gb,
                 "concurrency": round(p.concurrency, 1), "fits": p.fits}
                for p in steps
            ],
            "asked": asked,
            "advice": [
                {"kind": a.kind, "level": a.level, "facts": a.facts}
                for a in advise(fit, report.kv_dims, asked, kv_dtype,
                                report.context_length, fit.node_count)
            ] if asked else [],
        })
    return out


def _render_context(report, fit_reports: list, context: int | None, kv_dtype: str) -> None:
    """context กับจำนวนคนพร้อมกันเป็นของแลกกัน — ตัวเลขเดี่ยวทำให้ดูเหมือนมีคำตอบเดียว"""
    payload = _context_payload(report, fit_reports, context, kv_dtype)
    for entry in payload:
        if not entry["ladder"]:
            continue
        per_token = entry["kv_bytes_per_token"]
        table = Table(title=f"Context กับจำนวนคนพร้อมกัน — {entry['target']} "
                            f"(KV {kv_dtype} {per_token / 1024:.0f} KiB/token)")
        table.add_column("context", justify="right")
        table.add_column("KV ต่อคน", justify="right")
        table.add_column("พร้อมกัน", justify="right")
        for row in entry["ladder"]:
            mark = "" if row["fits"] else " [red]ไม่พอ[/red]"
            people = f"{row['concurrency']:.1f}" if row["fits"] else "—"
            highlight = "[bold]" if row["context"] == entry["asked"] else ""
            table.add_row(f"{highlight}{row['context']:,}",
                          f"{highlight}{row['kv_gb']:.1f} GB{mark}",
                          f"{highlight}{people}")
        console.print(table)
        for item in entry["advice"]:
            phrase = _ADVICE_TH.get(item["kind"])
            if phrase is None:
                continue
            colour = {"bad": "red", "warn": "yellow"}.get(item["level"], "dim")
            err_console.print(f"[{colour}]• {phrase(item['facts'])}[/{colour}]")


def _compute_fits(report, target_names: list[str], concurrency: int) -> list:
    from lmds.fit import PRESETS, analyze, from_hardware_report

    specs = []
    detected_spec = None            # สเปกที่มาจากการตรวจเครื่องนี้จริง (ไม่ใช่ preset)
    if target_names:
        for name in target_names:
            spec = PRESETS.get(name)
            if spec is None:
                err_console.print(
                    f"[red]ไม่รู้จัก target: {name}[/red] — มีให้เลือก: {', '.join(sorted(PRESETS))}"
                )
                raise typer.Exit(code=1)
            specs.append(spec)
    else:
        from lmds.hardware import probe

        detected = from_hardware_report(probe())
        if detected is not None:
            detected_spec = detected
            specs.append(detected)
        if not any(s.name.startswith("dgx-spark") for s in specs):
            specs.append(PRESETS["dgx-spark-single"])

    return [
        # หักเฉพาะกับสเปกที่ *ตรวจจากเครื่องนี้จริง* — preset เป็นเครื่องสมมติ การเอา
        # ของที่รันบนเครื่องนี้ไปหักออกจากมันคือการปนคนละเครื่องเข้าด้วยกัน
        analyze(report, spec, concurrency=concurrency,
                reserved_gb=_memory_already_held_gb() if spec is detected_spec else 0.0)
        for spec in specs
    ]


def _memory_already_held_gb() -> float:
    """GPU memory ที่โมเดลอื่นถืออยู่บนเครื่องนี้ — ตัวเดียวกับที่หน้าเว็บใช้ (profiler.memory_held_gb)"""
    from lmds.hardware.profiler import memory_held_gb

    return memory_held_gb()


def _render_fits(fit_reports: list) -> None:
    from lmds.fit import Verdict

    if not fit_reports:
        return
    icons = {
        Verdict.FITS: "✅",
        Verdict.FITS_REDUCED_CONTEXT: "✅",
        Verdict.FITS_WITH_OFFLOAD: "🟡",
        Verdict.NEEDS_SMALLER_QUANT: "❌",
        Verdict.NO_FIT: "❌",
        Verdict.UNKNOWN: "❓",
    }
    table = Table(title="Fit Analysis")
    table.add_column("Target")
    table.add_column("ผล")
    table.add_column("รายละเอียด")
    for fit in fit_reports:
        details: list[str] = []
        if fit.weights_gb is not None:
            details.append(f"weights {fit.weights_gb} / budget {fit.budget_gb} GB")
        if fit.recommended_context:
            details.append(f"context แนะนำ {fit.recommended_context:,}")
        if fit.client_input_budget:
            details.append(f"client input {fit.client_input_budget:,} tokens")
        fitting = [v for v in fit.variant_fits if v.fits]
        if fit.variant_fits and not fit.weights_gb:
            details.append(f"variant ผ่าน {len(fitting)}/{len(fit.variant_fits)}")
        table.add_row(
            f"{fit.target_name} ({fit.engine_assumed})",
            f"{icons[fit.verdict]} {fit.verdict.value}",
            "\n".join(details) or "-",
        )
    console.print(table)
    for fit in fit_reports:
        for note in fit.notes:
            err_console.print(f"[dim]• {fit.target_name}: {note}[/dim]")
        for alt in fit.alternatives:
            err_console.print(f"[yellow]→ {fit.target_name}: {alt}[/yellow]")


def _render_report(report) -> None:
    from lmds.inspector import ArtifactType

    table = Table(title=f"Inspect: {report.repo_id}", show_header=False)
    table.add_row("Revision (pinned)", report.revision_sha)
    table.add_row("Artifact", report.artifact_type.value + ("  🔒 gated" if report.gated else ""))
    table.add_row("License", report.license or "ไม่ระบุ")
    if report.params_total:
        table.add_row("Parameters", f"{report.params_total / 1e9:.1f}B")
    if report.weight_bytes:
        table.add_row("Weight size", f"{report.weight_bytes / 1e9:.1f} GB")
    if report.shard_count:
        table.add_row("Shards", str(report.shard_count))
    if report.architecture or report.model_type:
        table.add_row("Architecture", report.architecture or report.model_type)
    if report.context_length:
        table.add_row("Native context", f"{report.context_length:,}")
    if report.quantization:
        table.add_row("Quantization", report.quantization)
    if report.has_chat_template is not None:
        table.add_row("Chat template", "✅ มี" if report.has_chat_template else "❌ ไม่พบ")
    if report.capabilities:
        # อยู่ในตารางเดียวกับที่เหลือ ไม่แยกออกไป — คนอ่านตรงนี้เพื่อตัดสินใจว่าจะเอา
        # โมเดลนี้ไหม และความสามารถคือเหตุผลหลักที่เขาจะเอาหรือไม่เอา
        marks = {"yes": "✅", "likely": "🟡", "no": "❌", "server": "⚙️", "unknown": "❔"}
        order = ("tool_calling", "vision", "reasoning", "system_prompt", "json_mode", "streaming")
        labels = {
            "tool_calling": "Tool calling", "vision": "Vision", "reasoning": "Reasoning",
            "system_prompt": "System prompt", "json_mode": "JSON mode", "streaming": "Streaming",
        }
        for name in order:
            cap = report.capabilities.get(name)
            if not cap:
                continue
            table.add_row(labels[name], f"{marks.get(cap['status'], '·')} {cap['evidence']}")
        # 🟡 คือ "ต้องมีอะไรบางอย่างเพิ่ม" ซึ่งเป็นข้อมูลที่ตัดสินใจได้จริงกว่าเครื่องหมาย
        caveats = [
            f"  {labels[n]}: {report.capabilities[n]['caveat']}"
            for n in order
            if report.capabilities.get(n, {}).get("caveat")
            and report.capabilities[n]["status"] in ("likely", "yes", "no")
        ]
        if caveats:
            console.print(table)
            console.print("[dim]อ่านจากไฟล์ของโมเดล ยังไม่ได้รัน — สิ่งที่ต้องยืนยันตอนรัน:[/dim]")
            for line in caveats:
                console.print(f"[dim]{line}[/dim]")
            return
    if report.artifact_type in (ArtifactType.GGUF, ArtifactType.MIXED) and report.gguf_variants:
        variants = [v for v in report.gguf_variants if not v.is_mmproj and not v.is_mtp]
        mmproj = [v for v in report.gguf_variants if v.is_mmproj]
        table.add_row("GGUF variants", str(len(variants)) + (f" (+mmproj {len(mmproj)})" if mmproj else ""))
        if report.selected_gguf:
            table.add_row("Selected GGUF", report.selected_gguf)
    console.print(table)
    for warning in report.warnings:
        err_console.print(f"[yellow]⚠ {warning}[/yellow]")


@app.command()
def plan(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    target: Optional[str] = typer.Option(
        None, "--target", help="target preset (เช่น dgx-spark-single) — ว่าง = เครื่องนี้ หรือ dgx-spark-single",
        autocompletion=_complete_target,
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
    as_json: bool = typer.Option(False, "--json"),
    engine: Optional[str] = typer.Option(
        None, "--engine",
        help="เลือกรันไทม์เอง: vllm | sglang — ว่าง = ตามชนิดไฟล์ (GGUF→llama.cpp, safetensors→vLLM)"),
) -> None:
    """สร้าง Deployment Plan (ขั้นวางแผนของ deploy) — ยังไม่ generate สคริปต์

    Exit codes: 0 สำเร็จ, 1 input ผิด, 4 ต้องการ token, 5 ปัญหา provider/เครือข่าย
    """
    from lmds.brain import MissingKey, PlanError, ProviderError, build_plan, make_provider
    from lmds.config import Settings

    _, report = _resolve_and_inspect(model, revision, interactive_ok=not as_json)
    fits = _compute_fits(report, [target] if target else [], concurrency)
    fit = fits[0]

    provider = None
    if not no_llm:
        settings = Settings.load()
        if settings.provider is None:
            err_console.print(
                "[yellow]ยังไม่ได้ตั้งค่า LLM provider (lmds config set-provider ...) — ใช้ rule-based mode[/yellow]"
            )
        else:
            try:
                provider = make_provider(settings.provider, get_secret(settings.provider.name.value))
            except MissingKey as exc:
                err_console.print(f"[yellow]{exc} — ใช้ rule-based mode[/yellow]")

    deployment_plan = _build_plan_safe(report, fit, provider, engine=_engine_choice(engine))

    if as_json:
        print(deployment_plan.model_dump_json(indent=2))
        return
    _render_plan(deployment_plan, fit)


def _render_plan(deployment_plan, fit) -> None:
    from lmds.brain import Confidence

    table = Table(title=f"Deployment Plan: {deployment_plan.model_id}", show_header=False)
    table.add_row("Generator", deployment_plan.generator)
    table.add_row("Revision (pinned)", deployment_plan.revision)
    table.add_row("Runtime", f"{deployment_plan.runtime.engine.value} — {deployment_plan.runtime.image_ref}")
    table.add_row("Topology", deployment_plan.topology.value + f"  (target: {fit.target_name})")
    table.add_row("Served name", deployment_plan.served_model_name)
    if deployment_plan.selected_gguf:
        table.add_row("GGUF file", deployment_plan.selected_gguf)
    table.add_row(
        "Serving",
        f"context {deployment_plan.serving.context:,} | max output {deployment_plan.serving.max_output_tokens:,} "
        f"| util {deployment_plan.serving.gpu_memory_utilization} | seqs {deployment_plan.serving.max_num_seqs}",
    )
    features = []
    if deployment_plan.tool_calling.enabled:
        features.append(f"tools ({deployment_plan.tool_calling.parser})")
    if deployment_plan.reasoning.enabled:
        features.append(f"reasoning ({deployment_plan.reasoning.parser})")
    if deployment_plan.multimodal.modalities:
        features.append("multimodal: " + ",".join(deployment_plan.multimodal.modalities))
    if deployment_plan.moe.experts:
        active = deployment_plan.moe.experts_active
        features.append(
            f"MoE {deployment_plan.moe.experts} experts"
            + (f" (เปิด {active}/token)" if active else "")
        )
    if deployment_plan.speculative.draft_files:
        features.append("MTP speculative decoding")
    table.add_row("Features", ", ".join(features) or "ไม่เปิด (ยังไม่มีหลักฐานยืนยัน parser)")
    if deployment_plan.serving.extra_flags:
        table.add_row("Extra flags", " ".join(deployment_plan.serving.extra_flags))
    if deployment_plan.runtime_assets:
        table.add_row(
            "Runtime files", ", ".join(a.filename for a in deployment_plan.runtime_assets) + " (อนุมัติแล้ว)"
        )
    if deployment_plan.assets_needing_approval:
        table.add_row(
            "รออนุมัติ", ", ".join(a.filename for a in deployment_plan.assets_needing_approval)
        )
    counts = {c: 0 for c in Confidence}
    for fact in deployment_plan.facts:
        counts[fact.confidence] += 1
    table.add_row(
        "Facts",
        f"verified {counts[Confidence.VERIFIED]} | inferred {counts[Confidence.INFERRED]} "
        f"| unverified {counts[Confidence.UNVERIFIED]}",
    )
    console.print(table)

    if deployment_plan.runtime.rationale:
        console.print(f"[dim]เหตุผล: {deployment_plan.runtime.rationale}[/dim]")
    for warning in deployment_plan.warnings:
        err_console.print(f"[yellow]⚠ {warning}[/yellow]")
    if deployment_plan.flags_needing_approval:
        err_console.print(
            "[red]ต้องอนุมัติก่อนใช้:[/red] " + ", ".join(deployment_plan.flags_needing_approval)
        )


@app.command()
def generate(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    target: Optional[str] = typer.Option(None, "--target", help="target preset — ว่าง = เครื่องนี้/dgx-spark-single · dgx-spark-stacked = multi-node (2 เครื่อง)", autocompletion=_complete_target),
    output: str = typer.Option("./bundles", "--output", help="โฟลเดอร์ output ของ bundle"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
    engine: Optional[str] = typer.Option(
        None, "--engine",
        help="เลือกรันไทม์เอง: vllm | sglang — ว่าง = ตามชนิดไฟล์ (GGUF→llama.cpp, safetensors→vLLM)"),
    gguf: Optional[str] = typer.Option(
        None, "--gguf",
        help="repo GGUF หลาย variant: เลือกไฟล์ด้วยชื่อเต็ม หรือชื่อ quant เช่น Q8_K_XL / Q4_K_M"),
) -> None:
    """สร้าง deployment bundle โดยไม่ถามยืนยัน: plan → render → quality gates → ZIP (เหมือน deploy --yes แต่ไม่ต่อรอง flag)

    Exit codes: 0 สำเร็จ, 1 input ผิด, 3 โมเดลไม่ fit, 4 ต้องการ token, 5 ปัญหา provider
    """
    from pathlib import Path

    from lmds.brain import MissingKey, PlanError, ProviderError, build_plan, make_provider
    from lmds.config import Settings
    from lmds.fit import Verdict
    from lmds.generator import render_bundle

    source, report = _resolve_and_inspect(model, revision, interactive_ok=True)
    report = _ensure_gguf_selected(source, report, interactive=False, wanted=gguf or "")
    fit = _compute_fits(report, [target] if target else [], concurrency)[0]

    if fit.verdict in (Verdict.NO_FIT, Verdict.NEEDS_SMALLER_QUANT):
        err_console.print(f"[red]โมเดลไม่ fit กับ target {fit.target_name} ({fit.verdict.value})[/red]")
        for alt in fit.alternatives:
            err_console.print(f"[yellow]→ {alt}[/yellow]")
        raise typer.Exit(code=3)

    provider = None
    if not no_llm:
        settings = Settings.load()
        if settings.provider is not None:
            try:
                provider = make_provider(settings.provider, get_secret(settings.provider.name.value))
            except MissingKey as exc:
                err_console.print(f"[yellow]{exc} — ใช้ rule-based mode[/yellow]")
        else:
            err_console.print("[yellow]ยังไม่ได้ตั้งค่า provider — ใช้ rule-based mode[/yellow]")

    deployment_plan = _build_plan_safe(report, fit, provider, engine=_engine_choice(engine))

    bundle, results, delivered = _render_and_package(deployment_plan, report, fit, output)
    _render_plan(deployment_plan, fit)
    _render_gates(results)
    _render_delivery(bundle, delivered, native_prepare=_is_native_prepare(deployment_plan, fit),
                     stacked=deployment_plan.topology.value == "stacked",
                     assets=bool(deployment_plan.runtime_assets))


@app.command()
def adopt(
    container: str = typer.Argument("", help="ชื่อ container ที่รันอยู่ (ดูจาก docker ps)"),
    port: int = typer.Option(0, "--port", help="รับ process ที่ฟังอยู่ที่พอร์ตนี้ (ไม่ใช่ container)"),
    pid: int = typer.Option(0, "--pid", help="รับ process ตาม PID"),
    slug: str = typer.Option("", "--slug", help="ชื่อที่จะใช้ใน lmds (ว่าง = ตั้งให้จากของที่เจอ)"),
    output: str = typer.Option("./bundles", "--output"),
    take_over: bool = typer.Option(
        False, "--take-over",
        help="หยุด+disable systemd unit เดิมแล้วให้ LMDS คุมแทน (ต้องใช้ sudo)"),
) -> None:
    """รับโมเดลที่รันอยู่ก่อน LMDS เข้ามาอยู่ในระบบ — สร้าง controller จากของที่รันจริง

    ลูกค้าที่มี vLLM/llama.cpp รันอยู่ก่อนแล้วเพิ่งมาติดตั้ง LMDS: `lmds ps` เห็น container
    พวกนั้นและ stop/restart/logs ได้ แต่ทำอย่างอื่นไม่ได้เพราะไม่มี controller

    อ่าน image/env/mount/port/args จาก container ที่รันอยู่ แล้วเขียนเป็นสคริปต์ที่
    **รันคำสั่งเดิมซ้ำได้เป๊ะ** · ของที่รันอยู่ตอนนี้ไม่ถูกแตะต้อง
    """
    from lmds.fleet import FleetError
    from lmds.fleet.adopt import adopt as adopt_container
    from lmds.fleet.adopt import adopt_process, inspect_container

    if not container and not port and not pid:
        err_console.print(
            "[red]ต้องบอกว่าจะรับตัวไหน[/red] — ชื่อ container, หรือ --port / --pid "
            "สำหรับ process ที่รันตรง ๆ (ดูรายชื่อ: lmds ps)"
        )
        raise typer.Exit(code=1)

    # process ที่รันตรง ๆ ใต้ systemd ที่ลูกค้าเขียนเอง เจอพอ ๆ กับ container
    if port or pid:
        try:
            path, proc = adopt_process(pid=pid, port=port, slug=slug, output=Path(output))
        except FleetError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

        name = slug or (proc.model or f"pid-{proc.pid}").replace("_", "-").lower()
        console.print(f"[green]รับ process {proc.pid} เข้าระบบแล้ว[/green] → [bold]{name}[/bold]")
        console.print(f"[dim]engine:  {proc.engine} (native)[/dim]")
        console.print(f"[dim]binary:  {proc.exe}[/dim]")
        console.print(f"[dim]weights: {proc.model_path or '(ไม่ระบุใน argv)'}[/dim]")
        console.print(f"[dim]port:    {proc.port} · context: {proc.context or 'ไม่ระบุ'}[/dim]")
        console.print(f"[dim]สคริปต์: {path}[/dim]")
        if proc.unit:
            console.print(f"\n[yellow]process นี้เป็นของ {proc.unit}[/yellow]")
            if take_over:
                _take_over_unit(proc.unit)
            else:
                console.print(
                    f"[dim]unit เดิมยังคุมอยู่ — LMDS start จะชน port {proc.port} · "
                    f"ให้ LMDS คุมแทนด้วย: lmds adopt --port {proc.port} --take-over[/dim]"
                )
        console.print(
            "\n[dim]ทำได้: start · stop · restart · status · logs · test-text · "
            "client-config · network-info[/dim]"
        )
        console.print(
            "[dim]ไม่มี download/verify-files — weight เป็น path ที่คุณจัดการเอง[/dim]"
        )
        return

    try:
        info = inspect_container(container)
        path = adopt_container(container, slug=slug, output=Path(output))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    name = slug or info.container.replace("_", "-").lower()
    console.print(f"[green]รับ {info.container} เข้าระบบแล้ว[/green] → [bold]{name}[/bold]")
    console.print(f"[dim]image:   {info.image}[/dim]")
    console.print(f"[dim]model:   {info.model or '(อ่านจาก env/argv ไม่ได้)'}[/dim]")
    # ไม่มี "ชื่อเดิม" ให้เทียบเหมือนสาขา process: container ไม่ได้จดชื่อที่เสิร์ฟไว้ที่ไหน
    # ชื่อที่เปลี่ยนแล้วแสดงอยู่ในบรรทัด "รับ … เข้าระบบแล้ว → <slug>" ข้างบนอยู่แล้ว
    console.print(f"[dim]port:    {info.port} · context: {info.context or 'ไม่ระบุ'}[/dim]")
    console.print(f"[dim]สคริปต์: {path}[/dim]")
    console.print("\n[dim]ทำได้: start · stop · restart · status · logs · test-text · client-config[/dim]")
    console.print("[dim]ไม่มี download/verify-files — weight ของ container นี้เป็น path ที่คุณจัดการเอง[/dim]")




def _take_over_unit(unit: str) -> None:
    """หยุด+disable unit เดิมเพื่อให้ LMDS คุมแทน — ทำเมื่อผู้ใช้สั่งเท่านั้น

    unit ที่ตั้ง Restart=always จะแย่ง port กลับทุกครั้งที่ LMDS stop · ปล่อยไว้แล้ว
    ผู้ใช้จะเห็นอาการ "start แล้วแต่ตัวที่ตอบไม่ใช่ของ LMDS" ซึ่งไล่หาสาเหตุยากมาก
    """
    import subprocess

    console.print(f"หยุดและ disable [bold]{unit}[/bold] …")
    result = subprocess.run(
        ["sudo", "-n", "systemctl", "disable", "--now", unit],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print(f"[green]{unit} ถูกปิดแล้ว[/green] — LMDS คุม port นี้ต่อได้")
        console.print(f"[dim]ย้อนกลับ: sudo systemctl enable --now {unit}[/dim]")
        return
    err_console.print(
        f"[yellow]ปิด {unit} ไม่สำเร็จ[/yellow] ({(result.stderr or '').strip()[:120]})\n"
        f"สั่งเองได้: sudo systemctl disable --now {unit}"
    )

@app.command()
def rebuild(
    slug: str = typer.Argument(..., help="ชื่อ bundle ที่จะสร้างใหม่", autocompletion=_complete_slug),
    output: Optional[str] = typer.Option(
        None, "--output",
        help="ปลายทาง (ว่าง = ที่เดิมของ bundle · in-place — controller ที่ start/publish อ่านจะได้ตัวใหม่)"),
) -> None:
    """สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน — เก็บค่าที่เคยตัดสินใจไว้ ไม่ต้องเดินผ่าน wizard อีก

    ใช้เมื่อ bundle เก่าใช้ไม่ได้เพราะสิ่งที่อยู่นอกเหนือค่าที่ตั้ง เช่น image ที่ tag หายไป
    หรือ template รุ่นใหม่มีคำสั่ง/ตัวกันพลาดที่ของเก่าไม่มี

    ค่าที่เก็บไว้ (context, flags, ฟีเจอร์, target) ถูกนำกลับมาใช้ · ส่วนที่ระบบเลือกเอง
    (image, ตัวกันพลาดในสคริปต์) คำนวณใหม่ตามตรรกะปัจจุบัน — ไม่เรียก LLM ซ้ำ

    ค่าเริ่มต้นเขียนทับ **ที่เดิม** ของ bundle · ก่อนหน้านี้ default `./bundles` (relative CWD)
    ทำให้ bundle ที่อยู่นอก ~/bundles ถูก rebuild เป็นสำเนาใหม่ที่อื่น ส่วนตัวจริงที่
    start/publish อ่าน (จาก server.meta) ยังเป็นของเก่า — regenerate แล้วเหมือนไม่มีอะไรเปลี่ยน
    """
    from pathlib import Path as _Path

    from lmds.fleet import bundle_profile, find

    server = find(slug)
    if server is None or not server.controller:
        err_console.print(f"[red]ไม่พบ bundle: {slug}[/red] — ดูรายชื่อ: lmds list")
        raise typer.Exit(code=1)
    # in-place: controller = <output>/<slug>/<slug>-single.sh → output คือ parent ของโฟลเดอร์ slug
    output = output or str(_Path(server.controller).parent.parent)
    profile = bundle_profile(server.controller)
    if not profile:
        err_console.print(f"[red]อ่าน MODEL_PROFILE.yaml ของ {slug} ไม่ได้[/red] — สร้างใหม่ด้วย lmds deploy")
        raise typer.Exit(code=1)

    model = (profile.get("model") or {})
    model_id, revision = model.get("id") or "", model.get("revision") or None
    target = (profile.get("target") or {}).get("name") or ""
    if not model_id:
        err_console.print("[red]profile ไม่มี model.id — สร้างใหม่ด้วย lmds deploy[/red]")
        raise typer.Exit(code=1)

    # target เดิมอาจเป็นค่าที่เลิกใช้แล้ว (เจอจริง: "this-machine" จาก bundle รุ่นเก่า) —
    # เมื่อก่อน _compute_fits จะโยน error แล้ว rebuild ตายทั้งคำสั่ง แปลว่า bundle เก่า
    # อัปเดตด้วย rebuild ไม่ได้เลย · ถอยไป auto-detect แทนดีกว่าปฏิเสธทั้งใบ
    from lmds.fit import PRESETS
    if target and target not in PRESETS:
        console.print(f"[yellow]target เดิม '{target}' ไม่ใช่ preset ที่รู้จักแล้ว[/yellow] — "
                      f"ใช้ auto-detect ตามฮาร์ดแวร์เครื่องนี้แทน")
        target = ""

    old_image = (profile.get("runtime") or {}).get("image") or ""
    console.print(f"สร้าง [bold]{slug}[/bold] ใหม่จากค่าเดิม — {model_id} · target {target or 'อัตโนมัติ'}")

    source, report = _resolve_and_inspect(model_id, revision, interactive_ok=True)
    if model.get("selected_gguf"):
        # inspect ซ้ำด้วยไฟล์ที่ profile เคยเลือกไว้ — repo ที่มีหลาย variant จะไม่เลือกให้เอง
        # แล้วไม่มีใครเปิด GGUF header เลย: architecture / context / kv dims / MoE หายหมด
        # (ตั้ง selected_gguf ทีหลังไม่ช่วย เพราะ header ถูกอ่านตอน inspect เท่านั้น)
        from dataclasses import replace as dc_replace

        from lmds.inspector import HfClient, HfError, inspect_model

        if report.selected_gguf != model["selected_gguf"]:
            try:
                report = inspect_model(
                    dc_replace(source, filename=model["selected_gguf"]),
                    HfClient(token=get_secret("hf")),
                )
            except HfError as exc:
                console.print(f"[yellow]อ่าน header ของ {model['selected_gguf']} ไม่ได้ ({exc}) — "
                              f"ใช้ค่าที่ profile เก็บไว้แทน[/yellow]")
        report.selected_gguf = model["selected_gguf"]
    report = _ensure_gguf_selected(source, report, interactive=False)
    fit = _compute_fits(report, [target] if target else [], 1)[0]

    # ไม่เรียก LLM ซ้ำ — แผนเดิมถูกตรวจและอนุมัติไปแล้ว เอาค่ากลับมาแล้วให้ harden จัดการ
    # ส่วนที่ระบบเป็นเจ้าของ (image, topology, ตัวกันพลาด) ตามตรรกะปัจจุบัน
    from lmds.brain import build_plan
    from lmds.brain.orchestrator import harden_plan

    plan = build_plan(report, fit, None)
    serving = profile.get("serving") or {}
    if serving.get("context"):
        plan.serving.context = int(serving["context"])
    if serving.get("max_num_seqs"):
        plan.serving.max_num_seqs = int(serving["max_num_seqs"])
    if serving.get("extra_flags"):
        plan.serving.extra_flags = list(serving["extra_flags"])
    # build_plan harden ไปรอบหนึ่งแล้วด้วยค่าที่มันคิดเอง — warning จากรอบนั้นพูดถึงตัวเลข
    # ที่เราเพิ่งเขียนทับไป การแสดงมันต่อคือเล่าการตัดสินใจที่ไม่ได้เกิดขึ้นจริง
    plan.warnings = []
    plan = harden_plan(plan, report, fit)

    if plan.runtime.image_ref != old_image:
        console.print(f"[yellow]image เปลี่ยน:[/yellow] {old_image or '(ไม่มี)'} → "
                      f"[bold]{plan.runtime.image_ref}[/bold]")
    for warning in plan.warnings:
        console.print(f"[dim]· {warning}[/dim]")

    bundle, results, delivered = _render_and_package(plan, report, fit, output)
    _render_gates(results)
    _render_delivery(bundle, delivered, native_prepare=_is_native_prepare(plan, fit),
                     stacked=plan.topology.value == "stacked",
                     assets=bool(plan.runtime_assets))
    console.print(f"\n[dim]ส่งไปเครื่องอื่น: [bold]lmds node push <เครื่อง> {slug}[/bold][/dim]")


# ขั้นของ smoke test — เรียงตามที่ต้องเป็นจริง ล้มขั้นไหนก็หยุดตรงนั้น
# (verify ไฟล์ที่โหลดไม่จบ หรือ test-text กับ server ที่ยังไม่ขึ้น ไม่มีความหมาย)
SMOKE_STEPS = [
    ("download", "โหลด weight (resume ได้)"),
    ("verify-files", "ตรวจไฟล์ครบและถูกต้อง"),
    ("start", "สตาร์ต server"),
    ("test-text", "ถามจริงแล้วดูว่าตอบไหม"),
]


@app.command()
def smoke(
    slug: str = typer.Argument(..., help="ชื่อ bundle", autocompletion=_complete_slug),
    node: str = typer.Option("", "--on", help="รันบนเครื่องอื่นในทะเบียน (ว่าง = เครื่องนี้)",
                             autocompletion=_complete_node),
    keep: bool = typer.Option(False, "--keep", help="ไม่ต้อง stop ตอนจบ (ปล่อยให้รันต่อ)"),
    skip_download: bool = typer.Option(False, "--skip-download", help="ข้ามขั้นโหลด (ไฟล์ครบแล้ว)"),
) -> None:
    """พิสูจน์ว่า bundle นี้รันได้จริง: download → verify → start → test-text → stop

    gate ทั้ง 12 ด่านตรวจได้แค่ว่า *สคริปต์ถูกต้อง* — ไม่ได้บอกว่ารันแล้วได้คำตอบจริง
    ทุกบั๊กใหญ่ที่เจอในรอบนี้ (image ที่ tag ไม่มีอยู่, head container ไม่เคยขึ้น,
    ชุดทดสอบไปโดนโมเดลอื่น) ผ่าน gate หมดแล้วไปตายตอนรัน

    exit 0 ผ่านทุกขั้น · 2 ล้มบางขั้น (บอกว่าขั้นไหนและ log ท้าย)
    """
    steps = [s for s in SMOKE_STEPS if not (skip_download and s[0] in ("download", "verify-files"))]
    where = node or "เครื่องนี้"
    console.print(f"[bold]smoke test {slug}[/bold] บน {where} — {len(steps)} ขั้น")

    def run_step(command: str) -> tuple[int, str]:
        if node:
            from lmds.nodes import NodeError, find, run as run_remote

            target = find(node)
            if target is None:
                err_console.print(f"[red]ไม่รู้จักเครื่อง '{node}'[/red]")
                raise typer.Exit(code=1)
            quoted = shlex.quote(slug)
            script = (
                f"dir=\"$(ls -d ~/bundles/{quoted} ~/*/bundles/{quoted} 2>/dev/null | head -1)\"; "
                f"[ -n \"$dir\" ] || {{ echo 'ไม่พบ bundle {slug} บน {node}' >&2; exit 1; }}; "
                f"cd \"$dir\" && ctl=\"$(ls ./*-single.sh ./*-stacked.sh 2>/dev/null | head -1)\" && "
                f"\"$ctl\" {shlex.quote(command)}"
            )
            try:
                result = run_remote(target, script, timeout=7200)
            except NodeError as exc:
                return 1, str(exc)
            return result.exit_code, result.stdout + result.stderr
        from lmds.fleet import find

        server = find(slug)
        if server is None or not server.controller:
            err_console.print(f"[red]ไม่พบ bundle: {slug}[/red]")
            raise typer.Exit(code=1)
        proc = subprocess.run([server.controller, command], cwd=str(Path(server.controller).parent),
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr

    failed_at = ""
    for index, (command, what) in enumerate(steps, 1):
        console.print(f"\n[bold]{index}/{len(steps)}[/bold] {command} — [dim]{what}[/dim]")
        code, output = run_step(command)
        tail = "\n".join(l for l in output.strip().splitlines() if l.strip())[-1200:]
        if code != 0:
            failed_at = command
            err_console.print(f"[red]ล้มที่ขั้น '{command}' (exit {code})[/red]")
            if tail:
                err_console.print(f"[dim]{tail}[/dim]")
            break
        last = tail.splitlines()[-1] if tail else ""
        console.print(f"[green]ผ่าน[/green] [dim]{last[:110]}[/dim]")

    # หยุด server เสมอแม้ขั้นก่อนหน้าจะล้ม — ไม่งั้น smoke test ทิ้งของค้างไว้บนเครื่อง
    if not keep and any(c == "start" for c, _ in steps):
        console.print("\n[dim]stop — คืนเครื่องให้อยู่สภาพเดิม (ใช้ --keep ถ้าอยากให้รันต่อ)[/dim]")
        run_step("stop")

    if failed_at:
        err_console.print(f"\n[red]smoke test ไม่ผ่าน — ติดที่ '{failed_at}'[/red]")
        raise typer.Exit(code=2)
    console.print(f"\n[green]smoke test ผ่านทุกขั้น[/green] — {slug} รันได้จริงบน {where}")


def _render_and_package(deployment_plan, report, fit, output: str):
    """render → gates → checksums → zip — ใช้ร่วมกันระหว่าง generate และ deploy"""
    from pathlib import Path

    from lmds.generator import render_bundle
    from lmds.packager import make_zip, write_checksums
    from lmds.validator import all_passed, run_gates

    try:
        bundle = render_bundle(deployment_plan, report, fit, Path(output))
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    results = run_gates(bundle.directory, include_checksums=False)
    if not all_passed(results):
        _render_gates(results)
        err_console.print("[red]bundle ไม่ผ่าน quality gates — ไม่สร้าง ZIP[/red]")
        raise typer.Exit(code=2)

    checksums_path = write_checksums(bundle.directory)
    zip_path = make_zip(bundle.directory)
    # ลงทะเบียนกับ fleet ทันที — เดิม lmds list ไม่เห็น bundle จนกว่าจะ start สำเร็จครั้งแรก
    from lmds.fleet import register_bundle

    register_bundle(bundle.controller)
    return bundle, results, [*bundle.files, checksums_path, zip_path]


def _render_delivery(bundle, delivered, native_prepare: bool = False, stacked: bool = False,
                     assets: bool = False) -> None:
    table = Table(title="Bundle (static-validated ✅)")
    table.add_column("ไฟล์")
    for file_path in delivered:
        table.add_row(str(file_path))
    console.print(table)
    if stacked:
        steps = ["prepare-runtime", "download", "verify-files", "sync-worker", "verify-worker", "start"]
    else:
        steps = ["download", "verify-files"]
        if native_prepare or assets:
            steps.append("prepare-runtime")  # build llama.cpp / ดึงไฟล์ runtime ภายนอก (ครั้งแรกครั้งเดียว)
        steps += ["start", "test-text"]
    chain = " && ".join(f"./{bundle.controller.name} {s}" for s in steps)
    console.print(f"\nเริ่มใช้งาน:\n  cd {bundle.directory}\n  {chain}")
    if stacked:
        console.print(
            "[yellow]stacked (multi-node): แก้ MASTER_IP/WORKER_IP/SSH_USER/NCCL_SOCKET_IFNAME/NCCL_IB_HCA "
            "ใน CONFIG ของสคริปต์ก่อน + ตั้ง passwordless SSH master→worker[/yellow]"
        )
    if native_prepare:
        console.print("[dim]prepare-runtime จะติดตั้ง build dependencies (git/cmake/ninja) ให้เองผ่าน apt — ใช้ sudo ครั้งเดียว[/dim]")


def _is_native_prepare(deployment_plan, fit) -> bool:
    return deployment_plan.runtime.engine.value == "llamacpp" and fit.memory_model.value == "unified"


def _render_gates(results) -> None:
    table = Table(title="Quality Gates")
    table.add_column("Gate")
    table.add_column("ผล")
    table.add_column("รายละเอียด")
    for result in results:
        table.add_row(result.name, "✅" if result.passed else "❌", result.detail or "-")
    console.print(table)


@app.command()
def deploy(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    target: Optional[str] = typer.Option(None, "--target", help="target preset — ว่าง = เครื่องนี้/dgx-spark-single · dgx-spark-stacked = multi-node (2 เครื่อง)", autocompletion=_complete_target),
    output: str = typer.Option("./bundles", "--output"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
    engine: Optional[str] = typer.Option(
        None, "--engine",
        help="เลือกรันไทม์เอง: vllm | sglang — ว่าง = ตามชนิดไฟล์ (GGUF→llama.cpp, safetensors→vLLM)"),
    task: Optional[str] = typer.Option(
        None, "--task", help="generate | embed — ปกติเดาจาก repo (pipeline_tag/tags/ชื่อ) · ใส่เมื่อเดาผิด"),
    gguf: Optional[str] = typer.Option(
        None, "--gguf",
        help="repo GGUF หลาย variant: เลือกไฟล์ด้วยชื่อเต็ม หรือชื่อ quant เช่น Q8_K_XL / Q4_K_M "
             "(จำเป็นเมื่อไม่มี tty ให้เลือกหมายเลข — script/hub ใช้ทางนี้)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="ข้ามขั้นยืนยัน (สำหรับ scripting; ไม่อนุมัติ flag ค้าง)"),
) -> None:
    """Flow หลัก: วิเคราะห์ → วางแผน → ยืนยัน → generate → validate → ZIP

    Exit codes: 0 สำเร็จ, 1 input ผิด/ยกเลิก, 2 ไม่ผ่าน gates, 3 ไม่ fit, 4 ต้องการ token, 5 provider
    """
    import sys

    from lmds.brain import (
        MissingKey,
        PlanError,
        ProviderError,
        apply_asset_approvals,
        apply_flag_approvals,
        build_plan,
        make_provider,
    )
    from lmds.config import Settings
    from lmds.fit import Verdict

    interactive = sys.stdin.isatty() and not yes
    source, report = _resolve_and_inspect(model, revision, interactive_ok=not yes)
    report = _ensure_gguf_selected(source, report, interactive=interactive, wanted=gguf or "")
    if task:
        if task.strip().lower() not in {"generate", "embed"}:
            err_console.print(f"[red]--task ต้องเป็น generate หรือ embed (ได้ '{task}')[/red]")
            raise typer.Exit(code=2)
        report.task = task.strip().lower()
    if report.task == "embed":
        console.print("[cyan]โมเดล embedding[/cyan] — จะเสิร์ฟ /v1/embeddings ไม่มี chat · เดาผิด? --task generate")
    fit = _compute_fits(report, [target] if target else [], concurrency)[0]

    if fit.verdict in (Verdict.NO_FIT, Verdict.NEEDS_SMALLER_QUANT):
        err_console.print(f"[red]โมเดลไม่ fit กับ target {fit.target_name} ({fit.verdict.value})[/red]")
        for alt in fit.alternatives:
            err_console.print(f"[yellow]→ {alt}[/yellow]")
        raise typer.Exit(code=3)

    provider = None
    if not no_llm:
        settings = Settings.load()
        if settings.provider is not None:
            try:
                provider = make_provider(settings.provider, get_secret(settings.provider.name.value))
            except MissingKey as exc:
                err_console.print(f"[yellow]{exc} — ใช้ rule-based mode[/yellow]")
        else:
            err_console.print("[yellow]ยังไม่ได้ตั้งค่า provider — ใช้ rule-based mode[/yellow]")

    deployment_plan = _build_plan_safe(report, fit, provider, engine=_engine_choice(engine))

    _render_plan(deployment_plan, fit)

    if interactive:
        # อนุมัติ flag นอก allowlist รายตัว — การอนุมัติเป็นสิทธิ์ของผู้ใช้เท่านั้น
        approved: list[str] = []
        for flag in list(deployment_plan.flags_needing_approval):
            if typer.confirm(f"อนุมัติ flag นอก allowlist: {flag} ?", default=False):
                approved.append(flag)
        if approved:
            apply_flag_approvals(deployment_plan, approved)

        # ไฟล์ runtime ภายนอก = โค้ดที่จะรันใน container — แสดง URL ให้เห็นเต็ม ๆ ก่อนถาม
        approved_assets: list[str] = []
        for asset in list(deployment_plan.assets_needing_approval):
            err_console.print(f"[yellow]ไฟล์ runtime ภายนอก:[/yellow] {asset.filename}")
            err_console.print(f"  จาก: {asset.url}")
            if asset.purpose:
                err_console.print(f"  ใช้ทำ: {asset.purpose}")
            err_console.print("  [dim]ไฟล์นี้จะถูก mount เข้า container และรันจริง — review ก่อนอนุมัติ[/dim]")
            if typer.confirm(f"อนุมัติดึง {asset.filename} ?", default=False):
                approved_assets.append(asset.filename)
        if approved_assets:
            apply_asset_approvals(deployment_plan, approved_assets)

        # เพดานจริงจาก fit มักสูงกว่าค่าที่แผนเสนอมาก (แผนถูก cap ไว้ที่ค่ามาตรฐาน v3.0.0)
        # ถ้าไม่บอก ผู้ใช้จะไม่มีทางรู้ว่าเครื่องรับได้อีกเยอะ — เคสจริง: เสนอ 65,536 แต่รันได้ 262,144
        ceiling = fit.max_safe_context or deployment_plan.serving.context
        if ceiling > deployment_plan.serving.context:
            console.print(
                f"[dim]หน่วยความจำรองรับได้ถึง {ceiling:,} tokens "
                f"(แผนเสนอ {deployment_plan.serving.context:,} ตามค่าเริ่มต้นมาตรฐาน) — พิมพ์ตัวเลขเองเพื่อใช้เพิ่ม[/dim]"
            )
        context_input = typer.prompt(
            "context (Enter = ใช้ค่าตามแผน)", default=str(deployment_plan.serving.context)
        ).strip()
        if context_input.isdigit() and int(context_input) > 0:
            requested = int(context_input)
            if requested > ceiling:
                err_console.print(f"[yellow]เกินเพดานที่ปลอดภัย — ใช้ {ceiling:,} แทน[/yellow]")
                requested = ceiling
            deployment_plan.serving.context = requested

        if not typer.confirm("สร้าง bundle ตามแผนนี้?", default=True):
            console.print("ยกเลิกโดยผู้ใช้")
            raise typer.Exit(code=1)

    bundle, results, delivered = _render_and_package(deployment_plan, report, fit, output)
    _render_gates(results)
    _render_delivery(bundle, delivered, native_prepare=_is_native_prepare(deployment_plan, fit),
                     stacked=deployment_plan.topology.value == "stacked",
                     assets=bool(deployment_plan.runtime_assets))
    console.print("\n[dim]สถานะ: static-validated — รัน acceptance ตามลำดับด้านบนเพื่อยืนยันบนเครื่องจริง[/dim]")


@app.command()
def validate(
    bundle_dir: str = typer.Argument(..., help="โฟลเดอร์ bundle ที่จะตรวจ"),
    fix: bool = typer.Option(False, "--fix", help="regenerate PACKAGE_SHA256SUMS ก่อนตรวจ"),
) -> None:
    """รัน quality gates กับ bundle ใด ๆ (รวม bundle ที่แก้มือ) — exit 0 ผ่าน, 2 ไม่ผ่าน"""
    from pathlib import Path

    from lmds.packager import write_checksums
    from lmds.validator import all_passed, run_gates

    directory = Path(bundle_dir)
    if not directory.is_dir():
        err_console.print(f"[red]ไม่พบโฟลเดอร์: {bundle_dir}[/red]")
        raise typer.Exit(code=1)

    if fix:
        write_checksums(directory)
        console.print("regenerate PACKAGE_SHA256SUMS แล้ว")

    results = run_gates(directory, include_checksums=True)
    _render_gates(results)
    if not all_passed(results):
        raise typer.Exit(code=2)
    console.print("[green]ผ่านทุก gate — static-validated[/green]")


def _ram_bar(used: float, total: float, width: int = 18) -> str:
    filled = min(width, round(width * used / total)) if total else 0
    percent = used / total * 100 if total else 0
    color = "green" if percent < 70 else ("yellow" if percent < 90 else "red")
    return f"[{color}]{'▰' * filled}[/{color}]{'▱' * (width - filled)} {percent:.0f}%"


def _address_label(address: dict) -> str:
    """`10.2.1.70/24 eth0` — prefix บอกวง ชื่อการ์ดบอกว่าเสียบอยู่เส้นไหน"""
    text = address["ip"]
    if address.get("prefix"):
        text += f"/{address['prefix']}"
    if address.get("iface"):
        text += f" [dim]{address['iface']}[/dim]"
    if address.get("link_local"):
        text += " [yellow](link-local)[/yellow]"
    return text


def _render_host_panel() -> None:
    from lmds.hardware import host_summary

    host = host_summary()
    parts = [f"[bold]{host.hostname}[/bold]", host.ip, host.arch, f"profile: {host.profile.value}"]
    console.print("🖥  " + " · ".join(parts))
    # เครื่องที่มีหลายเส้น (LAN + Tailscale + สายคลัสเตอร์) — บรรทัดบนบอกได้เส้นเดียวคือเส้นที่
    # ออกเน็ต ซึ่งบ่อยครั้งไม่ใช่เส้นที่คนอื่นใช้ยิงเข้ามา · เส้นเดียวก็ไม่ต้องพูดซ้ำ
    if len(host.addresses) > 1:
        console.print("🌐 IP: " + " · ".join(_address_label(a) for a in host.addresses))
    if host.gpus:
        gpu_bits = []
        for gpu in host.gpus:
            if gpu.vram_mib:
                mem = f"{gpu.vram_mib / 1024:.0f}GB"
            elif gpu.known is not None:
                mem = f"{gpu.known.vram_gb:.0f}GB {'unified' if gpu.known.memory_model.value == 'unified' else ''}".strip()
            else:
                mem = "?"
            gpu_bits.append(f"{gpu.name} ({mem})")
        console.print("🎮 GPU: " + " | ".join(gpu_bits))
    if host.ram_total_gb and host.ram_used_gb is not None:
        console.print(
            f"📊 RAM: {_ram_bar(host.ram_used_gb, host.ram_total_gb)}  "
            f"ใช้ไป {host.ram_used_gb} / {host.ram_total_gb} GB (เหลือ {host.ram_available_gb} GB)"
        )
    console.print()


def _status_symbol(server) -> str:
    """สัญลักษณ์สั้นสำหรับตารางแคบ (lmds list) — คำอธิบายอยู่ใต้ตาราง"""
    if server.healthy:
        return "[green]●[/green]"
    if server.running:
        return "[yellow]◐[/yellow]"
    return "[dim]○[/dim]"


def _status_label(server) -> str:
    """ป้ายสถานะที่ใช้ร่วมกันระหว่าง lmds ps และ lmds list — จะได้ไม่อ่านคนละภาษา"""
    if server.healthy:
        return "[green]● running[/green]"
    if server.running:
        return "[yellow]◐ loading[/yellow]"
    return "[dim]○ stopped[/dim]"


def _render_all_nodes() -> None:
    """สถานะรวมของเครื่องอื่นในทะเบียน — ถามทีละเครื่องผ่าน SSH

    เครื่องที่ต่อไม่ได้ต้องไม่ทำให้ทั้งตารางพัง แค่ขึ้นว่าติดต่อไม่ได้พร้อมเหตุผล
    """
    from lmds.nodes import NodeError, load, probe

    nodes = load()
    if not nodes:
        console.print("[dim]ยังไม่มีเครื่องอื่นในทะเบียน — เพิ่มด้วย: lmds node add <ip> --user <ชื่อ>[/dim]\n")
        return

    table = Table(title=f"เครื่องอื่นในทะเบียน ({len(nodes)})")
    table.add_column("เครื่อง")
    # ที่อยู่ที่ใช้ SSH ไม่ได้ตอบว่า "เครื่องนั้นอยู่ IP ไหนในวง" — เป็นชื่อได้ และเครื่องหลัง
    # NAT มองจาก hub เป็นคนละที่อยู่กับที่เครื่องข้าง ๆ มันใช้เรียกกันเอง
    table.add_column("IP")
    table.add_column("โมเดล (slug)")
    table.add_column("engine")
    table.add_column("port")
    table.add_column("สถานะ")

    for node in nodes:
        try:
            info = probe(node)
        except NodeError as exc:
            # ต่อไม่ได้ = ไม่มีค่าใหม่ · ค่าล่าสุดที่ทะเบียนจำไว้คือสิ่งเดียวที่ยังช่วยตามเครื่องได้
            table.add_row(f"[bold]{node.name}[/bold]", f"[dim]{node.local_ip or '—'}[/dim]",
                          "—", "—", "—",
                          f"[red]ติดต่อไม่ได้[/red] {str(exc).splitlines()[0][:60]}")
            continue
        host = info.get("host") or {}
        local_ip = host.get("ip") or node.local_ip or "—"
        models = info.get("models") or []
        gpu = ", ".join(g["name"].replace("NVIDIA ", "") for g in host.get("gpus", []))
        if not models:
            table.add_row(f"[bold]{node.name}[/bold]", local_ip,
                          "[dim](ยังไม่มีโมเดล)[/dim]", "—", "—",
                          f"[green]ต่อได้[/green] · {gpu or 'ไม่พบ GPU'}")
            continue
        for index, model in enumerate(models):
            if model.get("running"):
                status = "[green]● running[/green]" if model.get("healthy") else "[yellow]◐ loading[/yellow]"
            elif not model.get("downloaded"):
                status = "[yellow]○ ยังไม่โหลดไฟล์[/yellow]"
            else:
                status = "○ stopped"
            table.add_row(f"[bold]{node.name}[/bold]" if index == 0 else "",
                          local_ip if index == 0 else "",
                          model.get("slug", ""), model.get("engine", ""),
                          str(model.get("port") or "-"), status)

    console.print(table)
    console.print("[dim]สั่งงานเครื่องอื่น: lmds node run <เครื่อง> <คำสั่ง lmds>[/dim]\n")


@app.command()
def ps(
    all_nodes: bool = typer.Option(False, "--all", "-a", help="รวมเครื่องอื่นในทะเบียนด้วย (fleet)"),
) -> None:
    """แสดงเครื่อง + ทุกโมเดลที่ deploy ในเครื่องนี้ พร้อมสถานะจริง (running/health/endpoint)"""
    from lmds.fleet import discover

    _render_host_panel()
    if all_nodes:
        _render_all_nodes()
    servers = discover()
    if not servers:
        console.print("ยังไม่มีโมเดลที่เคย start ในเครื่องนี้ — deploy ก่อน: lmds deploy <model-url>")
        return
    table = Table(title="LMDS Fleet (เครื่องนี้)" if all_nodes else "LMDS Fleet")
    table.add_column("ชื่อ (slug)")
    table.add_column("โมเดล")
    table.add_column("engine")
    table.add_column("port")
    table.add_column("สถานะ")
    for server in servers:
        status = _status_label(server)
        if server.external:
            status += " [cyan]⚙ ไม่ได้มาจาก lmds[/cyan]"
        elif not server.registered:
            status += " [yellow]⚠ ไม่ลงทะเบียน[/yellow]"
        table.add_row(server.slug, server.model or server.model_id, f"{server.engine} ({server.mode})",
                      str(server.port or "-"), status)
    console.print(table)
    running = [s for s in servers if s.running]
    if running:
        example = running[0].slug
        console.print(
            "\n[dim]ใช้ชื่อจากคอลัมน์แรกกับทุกคำสั่ง — copy ไปใช้ได้เลย:[/dim]\n"
            f"  lmds logs {example} -f\n"
            f"  lmds restart {example}\n"
            f"  lmds stop {example}\n"
            f"  lmds restart {example} --port 8001   [dim]# flag ของ controller ส่งต่อได้เลย[/dim]\n"
            "\n"
            "[dim]logs -f[/dim] ดู log สด (Ctrl-C ออก ไม่หยุดโมเดล) · "
            "[dim]stop --all[/dim] หยุดทุกตัว · [dim]lmds list[/dim] ดู bundle ทั้งหมด + repair/remove"
        )
    if any(s.external for s in servers):
        console.print(
            "[cyan]⚙ ตัวที่ 'ไม่ได้มาจาก lmds' คือ container ที่คุณรันเอง — "
            "lmds stop/restart/logs/enable ใช้ได้ (stop = docker stop ไม่ลบ container ทิ้ง)[/cyan]"
        )
    if any(not s.registered and not s.external for s in servers):
        err_console.print(
            "[yellow]⚠ ตัวที่ 'ไม่ลงทะเบียน' มาจาก bundle รุ่นเก่า — lmds stop ใช้ได้ (fallback) "
            "แต่แนะนำ regenerate bundle (lmds deploy ลิงก์เดิม) เพื่อเข้าระบบเต็มรูป[/yellow]"
        )


@app.command()
def stop(
    slug: Optional[str] = typer.Argument(None, help="ชื่อ (slug) จาก lmds ps", autocompletion=_complete_slug),
    all_servers: bool = typer.Option(False, "--all", help="หยุดทุกตัวที่รันอยู่"),
) -> None:
    """หยุดโมเดล — ระบุชื่อ หรือ --all"""
    from lmds.fleet import FleetError, discover, find, stop_server

    if all_servers:
        running = [s for s in discover() if s.running]
        if not running:
            console.print("ไม่มีโมเดลรันอยู่")
            return
        for server in running:
            try:
                method = stop_server(server)
                console.print(f"หยุด {server.slug} แล้ว ({method})")
            except FleetError as exc:
                err_console.print(f"[red]{exc}[/red]")
        return

    if not slug:
        err_console.print("[red]ระบุชื่อ (lmds ps ดูรายชื่อ) หรือใช้ --all[/red]")
        raise typer.Exit(code=1)
    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps")
        raise typer.Exit(code=1)
    if not server.running:
        console.print(f"{slug} ไม่ได้รันอยู่")
        return
    try:
        method = stop_server(server)
        console.print(f"หยุด {slug} แล้ว ({method})")
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def logs(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps", autocompletion=_complete_slug),
    lines: int = typer.Option(200, "-n", "--lines"),
    follow: bool = typer.Option(False, "-f", "--follow", help="ตาม log แบบ realtime (Ctrl-C เพื่อออก)"),
) -> None:
    """ดู log ของโมเดลตามชื่อ — ไม่ต้องจำ path ของ bundle"""
    from lmds.fleet import FleetError, find, logs_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps")
        raise typer.Exit(code=1)
    if follow:
        err_console.print(f"[dim]ตาม log ของ {slug} แบบ realtime — Ctrl-C เพื่อออก (ไม่หยุดโมเดล)[/dim]")
    # โมเดลที่ยังไม่เคยรันไม่มีไฟล์ log — เดิมปล่อย "tail: cannot open ..." ดิบ ๆ ออกไป
    # ซึ่งอ่านเหมือนระบบพัง ทั้งที่แค่ยังไม่เคยสตาร์ต
    # bundle ที่ยังไม่เคย start ไม่มีทั้ง log และ container — `mode` ของมันเป็นค่าเดา
    # (ออกมาเป็น "docker" เสมอ) จึงใช้ตัดสินไม่ได้ · เกณฑ์ที่เชื่อได้คือ "ไม่มีบันทึกว่าเคยรัน":
    # ไม่ได้รันอยู่ + ไม่มี started_at ในทะเบียน + ไม่มีไฟล์ log
    log_file = (server.run_dir / "server.log") if server.run_dir else None
    never_ran = (not server.running and not server.started_at
                 and not (log_file and log_file.exists()))
    if never_ran:
        err_console.print(f"[yellow]ยังไม่มี log ของ {slug}[/yellow] — โมเดลนี้ยังไม่เคยรันบนเครื่องนี้")
        err_console.print(f"[dim]เริ่มด้วย: [bold]lmds start {slug}[/bold] แล้วค่อยดู log[/dim]")
        raise typer.Exit(code=1)
    try:
        raise typer.Exit(code=logs_server(server, lines, follow=follow))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=0)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def restart(
    ctx: typer.Context,
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list", autocompletion=_complete_slug),
) -> None:
    """restart โมเดลตามชื่อ — ใช้ได้กับ container ที่ไม่ได้มาจาก lmds ด้วย

    flag ที่ไม่ใช่ของ lmds ถูกส่งต่อให้ controller: lmds restart x --port 8001
    """
    from lmds.fleet import FleetError, find, restart_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps")
        raise typer.Exit(code=1)
    try:
        method = restart_server(server, list(ctx.args))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"restart {slug} แล้ว ({method})")


# flag ที่ lmds ไม่รู้จักถูกส่งต่อให้ controller — มันเป็นเจ้าของ flag พวกนั้นและตรวจค่าเอง
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    ctx: typer.Context,
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list", autocompletion=_complete_slug),
) -> None:
    """รันโมเดลที่เคย deploy ไว้แล้วตามชื่อ — ไม่ต้อง cd ไปหา bundle

    flag ที่ไม่ใช่ของ lmds จะถูกส่งต่อให้ controller ตรง ๆ:

        lmds start my-model --port 8001 --context 32768 --gpu-util 0.8

    controller เป็นเจ้าของ flag พวกนี้และตรวจค่าเอง (แต่ละ engine มีไม่เท่ากัน)
    """
    from lmds.fleet import FleetError, find, start_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps หรือ lmds list")
        raise typer.Exit(code=1)
    if server.running:
        console.print(f"{slug} รันอยู่แล้ว (port {server.port})")
        return
    # --force เป็นของ lmds ไม่ใช่ของ controller — คัดออกก่อนส่งต่อ
    options = [arg for arg in ctx.args if arg != "--force"]
    force = len(options) != len(ctx.args)
    try:
        raise typer.Exit(code=start_server(server, options, force=force))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def enable(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list", autocompletion=_complete_slug),
    now: bool = typer.Option(False, "--now", help="สั่ง start ทันทีด้วย (ไม่รอ reboot)"),
    timeout: int = typer.Option(1800, "--timeout", help="วินาทีที่รอตอน start ใน service (โมเดลใหญ่ควรเพิ่ม)"),
    system_scope: bool = typer.Option(False, "--system", help="ติดตั้งเป็น system service (ต้อง sudo — ให้สิทธิ์เท่ากับ root)"),
) -> None:
    """ตั้งให้โมเดลกลับมาทำงานเองหลังเปิด-ปิดเครื่อง (systemd autostart)

    ค่าเริ่มต้นเป็น **user service ซึ่งไม่ต้องใช้ sudo เลย** — hub สั่งข้ามเครื่องผ่าน SSH
    ที่ไม่มี tty ให้กรอกรหัส การพึ่ง sudo จึงทำให้ปุ่มบนหน้าเว็บล้มเสมอ

    `--system` เขียนลง /etc/systemd/system ซึ่งต้อง sudo · ทางนั้นให้สิทธิ์เท่ากับ root
    เพราะ systemd unit รันคำสั่งอะไรก็ได้ในนามของ root
    """
    from lmds.fleet import FleetError, enable_autostart, find, unit_name

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps หรือ lmds list")
        raise typer.Exit(code=1)
    scope = "system" if system_scope else "user"
    password = ""
    if system_scope:
        console.print("[yellow]--system เขียน unit ของระบบ ต้องใช้ sudo[/yellow] "
                      "[dim](unit ของระบบรันคำสั่งอะไรก็ได้ในนามของ root)[/dim]")
        if sys.stdin.isatty():
            # ถามตรงนี้ ใช้ครั้งเดียว ไม่เก็บที่ไหน — ทะเบียน node ไม่มีฟิลด์รหัสผ่านโดยตั้งใจ
            password = typer.prompt("รหัสผ่าน sudo (Enter = ให้ sudo ถามเอง)",
                                    default="", hide_input=True, show_default=False)
    console.print(f"ติดตั้ง autostart สำหรับ [bold]{slug}[/bold] ({scope} service)…")
    try:
        name = enable_autostart(server, timeout=timeout, start_now=now, scope=scope, password=password)
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✅ เปิด autostart แล้ว[/green] ({name}) — โมเดลจะกลับมาเองหลัง reboot")
    if scope == "user":
        from lmds.fleet.manager import _linger_on

        if not _linger_on():
            console.print("[yellow]แต่ยังไม่ขึ้นตอนบูตจนกว่าจะเปิด linger[/yellow] — "
                          "รันครั้งเดียวบนเครื่องนั้น: "
                          f"[bold]sudo loginctl enable-linger {os.environ.get('USER', '$USER')}[/bold]")
    console.print(f"[dim]เช็ก: systemctl status {name} | ปิด: lmds disable {slug}[/dim]")
    if not now:
        console.print(f"[dim]start เดี๋ยวนี้เลย: lmds start {slug}  (หรือ lmds enable {slug} --now)[/dim]")


@app.command()
def disable(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list", autocompletion=_complete_slug),
) -> None:
    """ยกเลิก autostart (systemd) ของโมเดล — ใช้ sudo · ไม่ได้หยุดตัวที่รันอยู่ตอนนี้"""
    from lmds.fleet import FleetError, disable_autostart

    try:
        name = disable_autostart(slug)
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]ปิด autostart แล้ว[/green] ({name}) — ตัวที่รันอยู่ตอนนี้ยังไม่หยุด (ใช้ lmds stop {slug} ถ้าต้องการ)")


@app.command("list")
def list_bundles() -> None:
    """แสดง bundle ทั้งหมดที่รู้จักในเครื่อง (เคย start อย่างน้อยหนึ่งครั้ง)"""
    from lmds.fleet import (
        autostart_status,
        bundle_profile,
        discover,
        feature_summary,
        profile_context,
    )

    servers = discover()
    if not servers:
        console.print("ยังไม่มี bundle ที่รู้จัก — deploy แล้ว start อย่างน้อยหนึ่งครั้งก่อน")
        return
    astat = {
        "enabled": "[green]● เปิด[/green]",
        "disabled": "[dim]○ ปิด[/dim]",
        "absent": "[dim]—[/dim]",
        "n/a": "[dim]n/a[/dim]",
    }
    table = Table(title="Bundles ในเครื่องนี้")
    table.add_column("ชื่อ (slug)")
    # สถานะเป็นสัญลักษณ์ตัวเดียว — ตารางนี้มี 7 คอลัมน์อยู่แล้ว ใส่คำเต็มจะเบียดจนหัวตารางหาย
    # บนจอแคบ (มีคำอธิบายสัญลักษณ์ใต้ตาราง) · รายละเอียดเต็ม + endpoint ดูที่ lmds ps
    table.add_column("", no_wrap=True)
    table.add_column("โมเดล", max_width=32, overflow="fold")
    table.add_column("engine")
    table.add_column("port", justify="right")
    table.add_column("context", justify="right")
    table.add_column("รองรับ (support)")
    table.add_column("autostart")
    for server in servers:
        profile = bundle_profile(server.controller) if server.controller_exists else None
        engine = ((profile or {}).get("runtime") or {}).get("engine") or server.engine or "-"
        context = profile_context(profile)
        context_str = f"{context:,}" if context else "-"
        support = feature_summary(profile) if profile else "-"
        status = autostart_status(server.slug) if server.controller_exists else "absent"
        # controller หาย = สั่ง start/restart ไม่ได้ — ใช้สัญลักษณ์เตือนแทนคอลัมน์แยก
        state = _status_symbol(server)
        if not server.controller_exists and not server.running:
            state = "[red]⚠[/red]"
        table.add_row(
            server.slug,
            state,
            server.model_id or server.model,
            engine,
            str(server.port) if server.port else "-",
            context_str,
            support,
            astat.get(status, status),
        )
    console.print(table)
    first = servers[0].slug if servers else "<ชื่อ>"
    console.print(
        "\n[dim]สถานะ:[/dim] [green]●[/green] [dim]running ·[/dim] [yellow]◐[/yellow] [dim]loading ·[/dim] "
        "○ [dim]stopped ·[/dim] [red]⚠[/red] [dim]ไฟล์ controller หาย (start/restart ไม่ได้)[/dim]\n"
        "[dim]คอลัมน์แรก (slug) คือชื่อที่ใช้กับทุกคำสั่ง — copy ไปใช้ได้เลย:[/dim]\n"
        f"  lmds start {first}\n"
        f"  lmds start {first} --port 8001   [dim]# flag ของ controller ส่งต่อได้เลย[/dim]\n"
        f"  lmds stop {first}\n"
        f"  lmds restart {first}\n"
        f"  lmds logs {first} -f\n"
        f"  lmds enable {first}\n"
        f"  lmds repair {first}\n"
        f"  lmds remove {first}\n"
        "\n"
        "[dim]start[/dim] เปิดโมเดล · [dim]stop[/dim] หยุด · [dim]restart[/dim] เปิดใหม่ (ใช้ตอนเปลี่ยน option)\n"
        "[dim]logs -f[/dim] ดู log สด (Ctrl-C ออก ไม่หยุดโมเดล) · [dim]enable[/dim] ให้กลับมาเองหลัง reboot\n"
        "[dim]repair[/dim] โหลดไฟล์ที่ขาดกลับมา · [dim]remove[/dim] ลบทิ้งทั้งหมด "
        "([dim]--keep-weights[/dim] = เก็บ weight ไว้)\n"
        "\n"
        "[dim]endpoint + สถานะ health เต็ม ๆ: lmds ps[/dim]"
    )


@app.command()
def remove(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds list", autocompletion=_complete_slug),
    keep_weights: bool = typer.Option(False, "--keep-weights", help="ไม่ลบ weight (เก็บไว้ deploy ใหม่โดยไม่ต้องโหลดซ้ำ)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="ไม่ต้องถามยืนยัน"),
    dry_run: bool = typer.Option(False, "--dry-run", help="แสดงว่าจะลบอะไรบ้าง แล้วจบ — ไม่ลบจริง"),
) -> None:
    """ลบโมเดลออกจากเครื่อง: หยุด → ยกเลิก autostart → ลบ bundle/ทะเบียน/log/weight

    `--dry-run` ใช้ดูรายการก่อนตัดสินใจ (และเป็นตัวที่หน้าเว็บเรียกก่อนถามยืนยัน)
    """
    from lmds.fleet import find, removal_failed, removal_plan, remove_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds list")
        raise typer.Exit(code=1)

    items = removal_plan(server, include_weights=not keep_weights)
    if not items:
        console.print(f"ไม่พบไฟล์ของ {slug} ที่ต้องลบ")
    else:
        table = Table(title=f"จะลบทั้งหมดนี้ ({slug})")
        table.add_column("รายการ")
        table.add_column("path")
        table.add_column("ขนาด", justify="right")
        total = 0
        for item in items:
            total += item.size_bytes
            table.add_row(item.label, str(item.path), _human_size(item.size_bytes))
        console.print(table)
        console.print(f"รวม [bold]{_human_size(total)}[/bold]")
    if server.running:
        err_console.print("[yellow]โมเดลนี้กำลังรันอยู่ — จะถูกหยุดก่อนลบ[/yellow]")
    if keep_weights:
        console.print("[dim]--keep-weights: weight จะไม่ถูกลบ (deploy ใหม่แล้วใช้ต่อได้เลย)[/dim]")

    if dry_run:
        console.print("[dim]--dry-run: ยังไม่ได้ลบอะไร · ลบจริง: "
                      f"[bold]lmds remove {slug} -y[/bold][/dim]")
        return

    # ไม่มี terminal ให้ตอบ (เช่นถูกเรียกผ่าน SSH จาก hub) — "Aborted." เฉย ๆ ไม่บอกอะไรเลย
    # ว่าทำไมและต้องทำยังไงต่อ ผู้ใช้จะคิดว่าคำสั่งทำงานแล้วไม่มีอะไรเกิดขึ้น
    if not yes and not sys.stdin.isatty():
        err_console.print("[red]ตอบยืนยันไม่ได้ — คำสั่งนี้ไม่ได้รันจาก terminal[/red]")
        err_console.print(f"[dim]ดูรายการก่อน: [bold]lmds remove {slug} --dry-run[/bold] · "
                          f"ลบจริง: [bold]lmds remove {slug} -y[/bold][/dim]")
        raise typer.Exit(code=2)

    if not yes and not typer.confirm("ยืนยันลบ? (กู้คืนไม่ได้)", default=False):
        console.print("ยกเลิก")
        raise typer.Exit(code=1)

    lines = remove_server(server, include_weights=not keep_weights)
    for line in lines:
        console.print(f"  {line}")

    # "เรียบร้อย" ทั้งที่ยังเหลือของอยู่ = คำโกหกที่ผู้ใช้จะรู้ตัวตอนดิสก์ไม่ลด
    # (เคสจริง: weight ที่ container โหลดมาเป็น root เหลือ 23 GB)
    failed = removal_failed(lines)
    if failed:
        err_console.print(f"\n[red]ลบ {slug} ไม่ครบ — ยังเหลือของที่ลบไม่ได้[/red]")
        err_console.print("[dim]มักเป็นไฟล์ที่ container เขียนไว้ในนามของ root — ต้องใช้ sudo:[/dim]")
        for line in failed:
            path = line.rsplit(": ", 1)[-1]
            err_console.print(f"  [bold]sudo rm -rf {path}[/bold]")
        raise typer.Exit(code=2)
    console.print(f"[green]ลบ {slug} เรียบร้อย[/green]")


@app.command()
def web(
    port: int = typer.Option(8600, "--port", help="พอร์ตของหน้าเว็บ"),
    bind: str = typer.Option("127.0.0.1", "--bind", help="127.0.0.1 = เครื่องนี้เท่านั้น · 0.0.0.0 = ทั้งวง network"),
    token: str = typer.Option("", "--token", help="บังคับ token (ว่าง = สุ่มให้เมื่อ bind ออก network)"),
    background: bool = typer.Option(False, "--background", "-b", help="รันเบื้องหลัง — terminal ว่างใช้ CLI ต่อได้"),
    stop_web: bool = typer.Option(False, "--stop", help="หยุดตัวที่รันเบื้องหลังอยู่"),
    restart_web: bool = typer.Option(False, "--restart", help="หยุดตัวที่รันอยู่แล้วเปิดใหม่ (ลิงก์เดิมใช้ได้ต่อ)"),
    status_only: bool = typer.Option(False, "--status", help="บอกว่ามีตัวไหนรันอยู่ + ลิงก์ของมัน"),
    new_token: bool = typer.Option(False, "--new-token", help="สุ่ม token ใหม่ (ลิงก์เดิมใช้ไม่ได้ทันที)"),
    enable: bool = typer.Option(False, "--enable", help="ให้ขึ้นเองหลัง reboot และฟื้นเองถ้าตาย (systemd user service)"),
    disable: bool = typer.Option(False, "--disable", help="เลิกให้ขึ้นเอง"),
) -> None:
    """เปิดหน้าเว็บคุมโมเดล — ดูสถานะ, start/stop, doctor, logs ในหน้าเดียว"""
    from lmds.web import daemon

    # ค่าที่ผู้ใช้พิมพ์มาเองต้องถูกตรวจก่อนอย่างอื่น — flag ที่ผิดควรถูกบอกทันที
    # ไม่ใช่ไปโผล่ทีหลังหรือถูกบังด้วยข้อความเรื่องพอร์ตซึ่งไม่เกี่ยวกัน
    for value, where in ((token, "--token"), (os.environ.get(daemon.TOKEN_ENV, ""), f"${daemon.TOKEN_ENV}")):
        if value:
            try:
                daemon.validate_token(value)
            except daemon.TokenError as exc:
                err_console.print(f"[red]{where}: {exc}[/red]")
                raise typer.Exit(code=1)

    def show_running(state: dict, prefix: str) -> None:
        """พิมพ์ลิงก์ของ *ตัวที่เสิร์ฟจริง* — ลิงก์ที่ใช้ไม่ได้แย่กว่าไม่พิมพ์เลย"""
        from lmds.hardware.profiler import primary_ip

        console.print(f"{prefix} (PID {state['pid']} · พอร์ต {state['port']})")
        exposed_now = state.get("bind") not in {"127.0.0.1", "localhost", "::1"}
        hosts = [h for h in (primary_ip() if exposed_now else "", "127.0.0.1") if h]
        for h in dict.fromkeys(hosts):
            console.print(f"  [bold]{daemon.url(state, h)}[/bold]")

    if status_only:
        state = daemon.running()
        if state is None and daemon.service_active():
            token_now = daemon.remembered_token() or os.environ.get(daemon.TOKEN_ENV, "")
            console.print(f"หน้าเว็บรันอยู่ [dim](systemd — {daemon.UNIT_NAME})[/dim]")
            from lmds.hardware.profiler import primary_ip

            for host in dict.fromkeys(h for h in (primary_ip(), "127.0.0.1") if h):
                console.print(f"  [bold]http://{host}:{port}/[/bold]")
            if token_now:
                console.print(f"\n  token: [bold]{token_now}[/bold]")
            console.print(f"[dim]เปิดใหม่: lmds web --restart · หยุด: lmds web --stop · "
                          f"เลิกให้ขึ้นเอง: lmds web --disable[/dim]")
            return
        if state is None:
            console.print("ไม่มีหน้าเว็บที่รันเบื้องหลังอยู่ — เปิดด้วย: [bold]lmds web -b --bind 0.0.0.0[/bold]")
            return
        show_running(state, "หน้าเว็บรันอยู่")
        if state.get("token"):
            console.print(f"\n  token: [bold]{state['token']}[/bold]")
        return

    # หน้าเว็บที่ systemd ดูแลอยู่ต้องสั่งผ่าน systemd — ไม่งั้น --restart ไปไม่ถึงตัวที่เสิร์ฟจริง
    # แล้วบ่นว่า "พอร์ตไม่ว่าง มีโปรแกรมอื่นยึดอยู่" ทั้งที่มันคือของเราเอง (เจอจริง: แก้โค้ดแล้ว
    # restart แต่ server ยังรันของเก่า → endpoint ใหม่ตอบ 404)
    if (stop_web or restart_web) and daemon.service_active():
        action = "stop" if stop_web else "restart"
        if not daemon.service_control(action):
            err_console.print(f"[red]สั่ง systemd {action} ไม่สำเร็จ[/red] — "
                              f"ดู: systemctl --user status {daemon.UNIT_NAME}")
            raise typer.Exit(code=1)
        console.print(f"{'หยุด' if stop_web else 'เปิดใหม่'}หน้าเว็บแล้ว "
                      f"[dim](ผ่าน systemd — {daemon.UNIT_NAME})[/dim]")
        if restart_web:
            state = daemon.running() or {}
            token_now = state.get("token") or daemon.remembered_token()
            if token_now:
                console.print(f"\n  token: [bold]{token_now}[/bold]")
        return

    if disable:
        import subprocess as _sp

        _sp.run(["systemctl", "--user", "disable", "--now", daemon.UNIT_NAME], capture_output=True)
        daemon.unit_path().unlink(missing_ok=True)
        _sp.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("เลิกให้หน้าเว็บขึ้นเองแล้ว")
        return

    if enable:
        import shutil as _shutil
        import subprocess as _sp

        if _shutil.which("systemctl") is None:
            err_console.print("[red]เครื่องนี้ไม่มี systemd[/red] — ใช้ lmds web -b แทน "
                              "(รันจนกว่าเครื่องจะรีบูต)")
            raise typer.Exit(code=1)
        token = token or os.environ.get(daemon.TOKEN_ENV) or daemon.remembered_token()
        if not token and bind not in {"127.0.0.1", "localhost", "::1"}:
            token = daemon.new_token()
            daemon.remember_token(token)
        # ตัวที่รันอยู่แบบ -b จะชนพอร์ตกับ service — หยุดให้ก่อน
        if daemon.running():
            daemon.stop()
            daemon.wait_until_free(bind, port)
        path = daemon.unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(daemon.render_unit(port, bind, token), encoding="utf-8")
        path.chmod(0o600)     # ไฟล์นี้มี token อยู่ข้างใน
        _sp.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        result = _sp.run(["systemctl", "--user", "enable", "--now", daemon.UNIT_NAME],
                         capture_output=True, text=True)
        if result.returncode != 0:
            err_console.print(f"[red]เปิดบริการไม่สำเร็จ[/red] {result.stderr.strip()[:300]}")
            raise typer.Exit(code=1)
        # ไม่มี linger = service ตายตอน logout และไม่ขึ้นตอนบูต ซึ่งขัดกับเหตุผลที่สั่ง --enable
        linger = _sp.run(["loginctl", "enable-linger"], capture_output=True, text=True)
        console.print(f"[green]หน้าเว็บขึ้นเองแล้ว[/green] — ฟื้นเองถ้าตาย ({daemon.UNIT_NAME})")
        if linger.returncode != 0:
            console.print("[yellow]แต่ยังไม่ขึ้นตอนบูต[/yellow] — รันเองครั้งเดียว: "
                          f"[bold]sudo loginctl enable-linger {os.environ.get('USER', '$USER')}[/bold]")
        if token:
            console.print(f"\n  token: [bold]{token}[/bold]")
        console.print("[dim]ดูสถานะ: systemctl --user status lmds-web · เลิก: lmds web --disable[/dim]")
        return

    if stop_web or restart_web:
        stopped = daemon.stop()
        if stopped is not None:
            # "หยุดแล้ว" ต้องแปลว่าพอร์ตว่างจริง — SIGTERM ไม่ได้คืน socket ทันที ถ้าไม่รอ
            # คำสั่งถัดไปของผู้ใช้ (ทั้ง --restart และ `lmds web -b` ที่พิมพ์เอง) จะฟ้อง
            # "พอร์ตไม่ว่าง" จากตัวที่เขาเพิ่งสั่งหยุดไปเอง
            daemon.wait_until_free(stopped.get("bind") or bind, int(stopped.get("port") or port))
            console.print(f"หยุดหน้าเว็บแล้ว (PID {stopped['pid']})")
        elif stop_web:
            err_console.print("ไม่พบหน้าเว็บที่รันเบื้องหลังอยู่")
            # พอร์ตไม่ว่างทั้งที่ไม่มีของเรา = มีอย่างอื่นยึดอยู่ ต้องบอก ไม่งั้นสตาร์ตรอบหน้าจะงงซ้ำ
            if daemon.port_busy(bind, port):
                err_console.print(f"[yellow]แต่พอร์ต {port} ไม่ว่าง[/yellow] — มีโปรแกรมอื่นยึดอยู่: "
                                  f"[bold]ss -ltnp | grep {port}[/bold]")
            raise typer.Exit(code=1)
        if stop_web:
            return

    try:
        from lmds.web import serve
    except ImportError:
        err_console.print("[red]ยังไม่ได้ติดตั้งส่วนเว็บ[/red] — ติดตั้ง: "
                          "[bold]~/.local/share/lmds/venv/bin/pip install 'fastapi>=0.110' 'uvicorn>=0.27'[/bold]")
        raise typer.Exit(code=1)

    # มีตัวรันอยู่แล้วต้องไม่สตาร์ตซ้อน — รอบสองจะ bind ไม่ได้แล้วตาย แต่เราเผลอพิมพ์
    # token ใหม่ให้ไปแล้ว ผู้ใช้จึงเปิดลิงก์แล้วเจอ "ต้องมี token" ทั้งที่ copy มาถูก
    existing = daemon.running()
    if existing is not None:
        show_running(existing, "[yellow]มีหน้าเว็บรันอยู่แล้ว[/yellow] — ใช้ลิงก์นี้")
        console.print("[dim]เปิดใหม่/เปลี่ยนพอร์ต: [bold]lmds web --restart -b[/bold] · "
                      "หยุด: [bold]lmds web --stop[/bold] · "
                      "เปลี่ยน token: [bold]lmds web --restart -b --new-token[/bold][/dim]")
        # ออกด้วย EXIT_ALREADY_RUNNING ไม่ใช่ 0 — systemd unit ตั้ง Restart=always ไว้
        # ถ้าจบด้วย 0 มันอ่านว่า "ทำงานเสร็จสวย" แล้วปลุกใหม่ทุก RestartSec ไม่รู้จบ
        # เจอจริง: มีคนรัน `lmds web` ด้วยมือค้างถือพอร์ตไว้ ตัวของ service จึงเจอตัวนี้
        # ทุกครั้งแล้วจบ — วนไป 144 รอบเงียบ ๆ โดยไม่มีใครรู้ ส่วนที่เสิร์ฟจริงคือตัวมือ
        # ซึ่งรันโค้ดคนละรุ่นกับที่ติดตั้งไว้ · unit จับคู่ด้วย RestartPreventExitStatus
        raise typer.Exit(code=daemon.EXIT_ALREADY_RUNNING)
    if daemon.port_busy(bind, port):
        err_console.print(f"[red]พอร์ต {port} ไม่ว่าง[/red] — มีโปรแกรมอื่นยึดอยู่ (ไม่ใช่ของ lmds)")
        err_console.print(f"[dim]ดูว่าใคร: [bold]ss -ltnp | grep {port}[/bold] · "
                          f"หรือใช้พอร์ตอื่น: [bold]lmds web --port {port + 1}[/bold][/dim]")
        raise typer.Exit(code=1)

    # หน้านี้สั่ง start/stop โมเดลได้ — เปิดออก network โดยไม่มี token = ใครในวงก็สั่งได้
    # จึงสุ่ม token ให้เองแทนที่จะปล่อยโล่ง (ผู้ใช้ตั้งเองได้ด้วย --token)
    exposed = bind not in {"127.0.0.1", "localhost", "::1"}
    if new_token:
        daemon.forget_token()

    # ลำดับที่มาของ token — ตัวบนสุดที่มีค่าชนะ · ผู้ใช้ต้องเดาได้ว่ามันมาจากไหน
    #   1. --token   2. $LMDS_WEB_TOKEN   3. ที่จำไว้ในเครื่อง   4. ถามตอนสตาร์ต   5. สุ่มให้
    source = "--token"
    if token:
        token = daemon.validate_token(token)
    elif os.environ.get(daemon.TOKEN_ENV):
        token = daemon.validate_token(os.environ[daemon.TOKEN_ENV])
        source = f"${daemon.TOKEN_ENV}"
    elif exposed and daemon.remembered_token():
        token, source = daemon.remembered_token(), "ที่จำไว้ในเครื่อง"
    elif exposed:
        # ครั้งแรกของเครื่องนี้ — ถามก่อน ปล่อยว่างแล้วสุ่มให้ · ไม่มี terminal ก็สุ่มเลย
        if sys.stdin.isatty():
            console.print(f"[bold]ตั้ง token สำหรับเข้าหน้าเว็บ[/bold] "
                          f"(อย่างน้อย {daemon.MIN_TOKEN_LEN} ตัว · Enter เฉย ๆ = สุ่มให้)")
            while True:
                typed = typer.prompt("token", default="", show_default=False).strip()
                if not typed:
                    token, source = daemon.new_token(), "สุ่มให้"
                    break
                try:
                    token, source = daemon.validate_token(typed), "ที่กรอกเอง"
                    break
                except daemon.TokenError as exc:
                    err_console.print(f"[red]{exc}[/red]")
        else:
            token, source = daemon.new_token(), "สุ่มให้"
        daemon.remember_token(token)

    # ไม่ใส่ token ใน URL อีกต่อไป — URL ไปอยู่ใน history/log/referrer ของเบราว์เซอร์
    # และคนที่ยืนดูจอก็อ่านได้ · หน้าเว็บมีช่องให้กรอก token เอง (จำไว้ให้ในเบราว์เซอร์)
    if not exposed:
        console.print(f"เปิดที่: [bold]http://127.0.0.1:{port}/[/bold]")
    else:
        # 0.0.0.0 เป็นที่อยู่สำหรับ bind ไม่ใช่ที่อยู่ที่เปิดในเบราว์เซอร์ได้ —
        # พิมพ์ IP จริงของเครื่องให้ ไม่งั้นผู้ใช้ต้องไปหา IP เอง
        from lmds.hardware.profiler import primary_ip

        hosts = [h for h in (primary_ip(), "127.0.0.1") if h]
        console.print("เปิดที่:")
        for h in dict.fromkeys(hosts):
            console.print(f"  [bold]http://{h}:{port}/[/bold]")
    if token:
        console.print(f"\n  token: [bold]{token}[/bold]  [dim]({source})[/dim]")
        console.print("[dim]กรอกในหน้าเว็บครั้งแรกครั้งเดียว เบราว์เซอร์จำให้ · "
                      "ดูอีกครั้ง: lmds web --status[/dim]")
    if exposed:
        console.print("[dim]ทุกคนที่เข้าถึงเครื่องนี้ในวง network เปิดหน้านี้ได้ — อย่าแชร์ token ออกนอกทีม[/dim]")
        console.print("[dim]เครื่องที่มีหลายวง (เช่น Tailscale/VPN) ใช้ IP ของวงนั้นแทนได้ พอร์ตและ token เดียวกัน[/dim]")
    if background:
        # หน้าเว็บกับ CLI ต้องใช้พร้อมกันได้ — รันค้าง terminal ไว้ทำให้เลือกได้อย่างเดียว
        import subprocess

        log_path = daemon.log_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "lmds.cli.main", "web",
                 "--port", str(port), "--bind", bind, *(["--token", token] if token else [])],
                stdout=log, stderr=log, start_new_session=True,
            )
        # ต้องรอให้มันรับ connection ได้จริงก่อน — เคสที่พังคือ bind ไม่ได้แล้วตายใน 0.2 วิ
        # ซึ่ง Popen มองว่าสำเร็จ แล้วเราก็พิมพ์ลิงก์พร้อม token ที่ไม่มีใครถืออยู่ให้ผู้ใช้
        if not daemon.wait_until_serving(bind, port, proc.pid):
            proc.poll()
            err_console.print(f"[red]เปิดหน้าเว็บไม่สำเร็จ[/red] — log: {log_path}")
            tail = daemon.log_tail()
            if tail:
                err_console.print(f"[dim]{tail}[/dim]")
            raise typer.Exit(code=1)
        daemon.write_state(proc.pid, port, bind, token)
        console.print(f"[dim]รันเบื้องหลัง (PID {proc.pid}) · log: {log_path}[/dim]")
        console.print("[dim]ดูลิงก์อีกครั้ง: lmds web --status · หยุด: lmds web --stop[/dim]")
        return

    console.print("[dim]หยุดด้วย Ctrl-C · หรือรันเบื้องหลังด้วย: lmds web --background[/dim]")
    try:
        serve(host=bind, port=port, token=token)
    except KeyboardInterrupt:
        console.print("ปิดหน้าเว็บแล้ว")


@app.command()
def doctor(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds list", autocompletion=_complete_slug),
) -> None:
    """ตรวจว่าทำไมโมเดลนี้ยัง download/start ไม่ผ่าน — บอกสาเหตุพร้อมคำสั่งแก้

    ตรวจด้วยข้อเท็จจริงบนเครื่องล้วน ไม่ใช้ LLM · exit 0 ผ่าน, 2 มีข้อที่ต้องแก้
    """
    from lmds.doctor import Status, diagnose

    result = diagnose(slug)
    icon = {Status.OK: "[green]✅[/green]", Status.WARN: "[yellow]⚠️ [/yellow]", Status.FAIL: "[red]❌[/red]"}

    table = Table(title=f"Doctor: {slug}", show_header=True)
    table.add_column("")
    table.add_column("ตรวจ")
    table.add_column("ผล")
    for finding in result.findings:
        table.add_row(icon[finding.status], finding.name, finding.detail)
    console.print(table)

    todo = result.failed + result.warnings
    if todo:
        console.print("\n[bold]วิธีแก้[/bold]")
        for finding in todo:
            if finding.fix:
                console.print(f"  {finding.name}: [cyan]{finding.fix}[/cyan]")

    if result.failed:
        err_console.print(f"\n[red]พบ {len(result.failed)} ข้อที่ต้องแก้ก่อนถึงจะรันได้[/red]")
        raise typer.Exit(code=2)
    console.print("\n[green]ไม่พบปัญหาที่บล็อกการรัน[/green]")


@app.command()
def repair(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds list", autocompletion=_complete_slug),
    force: bool = typer.Option(False, "--force", help="ยืนยันว่าเครื่องนี้รันโมเดลได้จริง ทั้งที่ตรวจไม่เจอ engine"),
) -> None:
    """ซ่อมไฟล์โมเดลที่ขาด/เสีย — โหลดเฉพาะส่วนที่หายแล้วตรวจซ้ำ (resume ได้)"""
    from lmds.fleet import FleetError, find, repair_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds list")
        raise typer.Exit(code=1)
    try:
        from lmds.hardware import serving

        blocked = serving.guard(slug, "repair", force=force)
        if blocked:
            err_console.print(f"[red]{blocked}[/red]")
            raise typer.Exit(code=1)
        console.print(f"ซ่อม {slug}: download (resume) → verify-files")
        code = repair_server(server, force=force)
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if code == 0:
        console.print(f"[green]{slug} ไฟล์ครบและถูกต้องแล้ว[/green]")
    else:
        err_console.print(f"[red]ยังไม่ผ่าน — ดูข้อความด้านบน (exit {code})[/red]")
    raise typer.Exit(code=code)


# ── คะแนนโมเดล ────────────────────────────────────────────────────────────────
bench_app = typer.Typer(help="วัดความเร็วและความสามารถของโมเดลที่รันอยู่", no_args_is_help=True)
app.add_typer(bench_app, name="bench")


def _quant_from_filename(filename: str) -> str:
    """ดึงชื่อ quant ออกจากชื่อไฟล์ GGUF — Qwen3.8-27B-Uncensored-Q4_K_M.gguf → Q4_K_M"""
    import re as _re

    match = _re.search(r"[.-]((?:IQ|Q)\d[^.]*?|BF16|F16|F32)(?:\.gguf|-\d{5}-of-\d{5}\.gguf)$",
                       filename or "", _re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _bench_environment(server, profile: dict) -> dict:
    """สภาพแวดล้อมที่ทำให้ตัวเลขความเร็วมีความหมาย — ขาดอันไหนไปก็เทียบข้ามรอบไม่ได้"""
    import httpx

    model = (profile or {}).get("model") or {}
    serving = (profile or {}).get("serving") or {}
    environment = {
        # ทาง GGUF ไม่ได้จด quantization ไว้ใน profile — ชื่อไฟล์บอกอยู่แล้ว (…-Q4_K_M.gguf)
        # การ์ดนำเสนอที่ช่อง quantization ว่างอ่านแล้วเหมือนข้อมูลไม่ครบ ทั้งที่รู้ได้
        "quant": (model.get("quantization") or model.get("quant")
                  or _quant_from_filename(model.get("selected_gguf") or profile.get("selected_gguf") or "")),
        "context": serving.get("context") or serving.get("context_tokens") or 0,
        "engine_build": "",
        "features": (profile or {}).get("features") or {},
    }
    # build ของ engine อ่านจากเซิร์ฟเวอร์ที่รันอยู่จริง ไม่ใช่จากที่จดไว้ตอน deploy
    try:
        response = httpx.get(f"http://127.0.0.1:{server.port}/props", timeout=5.0)
        if response.status_code == 200:
            props = response.json()
            environment["engine_build"] = str(props.get("build_info") or "")
            environment["context"] = props.get("default_generation_settings", {}).get(
                "n_ctx") or environment["context"]
    except Exception:
        pass
    return environment


@bench_app.command("run")
def bench_run(
    slug: str = typer.Argument(..., help="ชื่อ (slug) ของโมเดลที่รันอยู่", autocompletion=_complete_slug),
    quick: bool = typer.Option(False, "--quick", help="ชุดสั้น 2 งาน — ดูคร่าว ๆ ไม่ใช่ตัวเลขที่เอาไปเทียบ"),
    runs: int = typer.Option(3, "--runs", help="ยิงกี่รอบต่อหนึ่งงาน แล้วเอาค่ากลาง"),
    speed_only: bool = typer.Option(False, "--speed-only", help="วัดความเร็วอย่างเดียว"),
    caps_only: bool = typer.Option(False, "--caps-only", help="ตรวจความสามารถอย่างเดียว"),
) -> None:
    """วัดโมเดลที่รันอยู่ แล้วเก็บผลไว้เทียบทีหลัง

    ทุกอย่างวัดจากเซิร์ฟเวอร์จริงผ่าน OpenAI API — ได้ตัวเลขที่เทียบข้าม engine ได้
    ต่างจากคำสั่ง bench ของ controller ที่แต่ละ engine มีไม่เท่ากันและวัดคนละวิธี
    """
    from lmds import bench
    from lmds.fleet import bundle_profile, find

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps")
        raise typer.Exit(code=1)
    if not server.running:
        err_console.print(f"[red]{slug} ยังไม่ได้รัน[/red] — วัดได้เฉพาะโมเดลที่รันอยู่: lmds start {slug}")
        raise typer.Exit(code=1)

    profile = bundle_profile(server.controller) or {}
    environment = _bench_environment(server, profile)
    served = server.model or ((profile.get("model") or {}).get("served_name")) or slug
    endpoint = server.endpoint
    features = profile.get("features") or {}
    has_projector = bool((features.get("multimodal") or {}).get("projector_files"))
    context_limit = int(environment.get("context") or 0)

    console.print(f"วัด [bold]{slug}[/bold] · {served} · {endpoint}")
    if environment.get("engine_build"):
        console.print(f"[dim]build {environment['engine_build']} · context {context_limit:,}[/dim]")

    workload_rows: list[dict] = []
    if not caps_only:
        chosen = bench.select("quick" if quick else "full", context_limit)
        skipped = len(bench.select("quick" if quick else "full")) - len(chosen)
        if skipped:
            console.print(f"[dim]ข้าม {skipped} งานที่ยาวเกิน context ของโมเดลนี้[/dim]")

        def progress(workload, index, total):
            console.print(f"  {workload.label} ({workload.input_tokens} tok) — รอบ {index}/{total}")

        results = bench.measure(endpoint, served, chosen, runs=runs, on_progress=progress)
        workload_rows = [r.as_dict() for r in results]

    probe_rows: list[dict] = []
    if not speed_only:
        console.print("ตรวจความสามารถ…")
        probes = bench.run_probes(endpoint, served, has_projector, context_limit,
                                  on_progress=lambda name: console.print(f"  {name}…"))
        probe_rows = [{"key": p.key, "label": p.label, "passed": p.passed,
                       "detail": p.detail, "skipped": p.skipped} for p in probes]

    path = bench.record(slug, server.model_id or server.model, server.engine, served,
                        workload_rows, probe_rows, environment, bench.now_stamp())
    _print_bench(workload_rows, probe_rows)
    console.print(f"\n[dim]เก็บผลไว้ที่ {path}[/dim]")
    console.print(f"[dim]เทียบกับรอบก่อน: lmds bench show {slug}[/dim]")


@bench_app.command("remove")
def bench_remove(
    slug: str = typer.Argument(..., help="ชื่อ (slug)", autocompletion=_complete_slug),
    keep_last: int = typer.Option(0, "--keep-last", help="เก็บรอบล่าสุดไว้กี่รอบ (0 = ลบทั้งหมด)"),
) -> None:
    """ลบผลวัดที่เก็บไว้ของโมเดลหนึ่ง

    ผลสะสมเร็วกว่าที่คิด เพราะการวัดซ้ำเป็นเรื่องปกติ (ก่อน/หลังเปลี่ยน flag,
    ก่อน/หลังอัปเกรด engine) แล้วไม่มีใครกลับมาลบเอง
    """
    from lmds.bench import remove, runs_for

    before = len(runs_for(slug))
    if not before:
        console.print(f"[dim]{slug} ไม่มีผลวัดเก็บไว้[/dim]")
        return
    removed = remove(slug, keep_last=max(0, keep_last))
    kept = before - removed
    console.print(f"ลบผลวัดของ [bold]{slug}[/bold] ไป {removed} รอบ"
                  + (f" · เหลือไว้ {kept} รอบล่าสุด" if kept else ""))


def _print_bench(workloads: list[dict], probes: list[dict]) -> None:
    from lmds.bench import capability_score, speed_summary

    if workloads:
        table = Table(title="ความเร็ว (ค่ากลางจากทุกรอบ)")
        table.add_column("งาน")
        table.add_column("input", justify="right")
        table.add_column("decode tok/s", justify="right")
        table.add_column("TTFT", justify="right")
        table.add_column("prefill tok/s", justify="right")
        for row in workloads:
            if row.get("error"):
                table.add_row(row["label"], str(row["target_input"]), "—", "—",
                              f"[red]{row['error'][:40]}[/red]")
                continue
            table.add_row(row["label"], f"{row['prompt_tokens']:,}",
                          f"{row['decode_tps']:.1f}", f"{row['ttft_s']:.2f}s",
                          f"{row['prefill_tps']:.0f}")
        console.print(table)
        summary = speed_summary(workloads)
        if summary["decode_tps_avg"]:
            console.print(f"[dim]เฉลี่ย {summary['decode_tps_avg']} tok/s · "
                          f"ที่ context ยาวสุด ({summary['longest_context']:,}) "
                          f"{summary['decode_tps_long']} tok/s[/dim]")
        # prefix cache ที่ยังกินอยู่ทำให้ TTFT/prefill ต่ำกว่าความจริงมาก — ต้องบอก ไม่ใช่ปล่อยผ่าน
        # ส่วนหัวของ chat template ถูก cache เสมอไม่ว่าจะทำอะไร (~50 token ต่อคำขอ)
        # เตือนเฉพาะตอนที่มันมากพอจะบิดตัวเลขจริง ไม่งั้นคำเตือนจะขึ้นทุกครั้งจนไม่มีใครอ่าน
        cached = sum(w.get("cached_tokens") or 0 for w in workloads)
        total_prompt = sum((w.get("prompt_tokens") or 0) * (w.get("runs") or 0) for w in workloads)
        if cached and total_prompt and cached / total_prompt > 0.10:
            console.print(f"[yellow]⚠ เซิร์ฟเวอร์ใช้ prompt cache ไป {cached:,} token "
                          f"({cached / total_prompt:.0%} ของ prompt ทั้งหมด) — "
                          f"TTFT/prefill ต่ำกว่าความจริง[/yellow]")

    if probes:
        table = Table(title="ความสามารถ")
        table.add_column("")
        table.add_column("ข้อ")
        table.add_column("ที่ได้")
        for probe in probes:
            mark = "[dim]—[/dim]" if probe["skipped"] else ("✅" if probe["passed"] else "❌")
            table.add_row(mark, probe["label"], probe["detail"][:60])
        console.print(table)
        score = capability_score(probes)
        if score["score"] is not None:
            console.print(f"[dim]คะแนนความสามารถ {score['score']}/100 "
                          f"(นับ {score['counted']} ข้อที่วัดได้)[/dim]")


@bench_app.command("list")
def bench_list() -> None:
    """ตารางคะแนนของทุกโมเดลที่เคยวัด — เอารอบล่าสุดของแต่ละตัว"""
    from lmds.bench import all_runs, summarize

    runs = all_runs()
    if not runs:
        console.print("ยังไม่เคยวัดอะไรเลย — เริ่มที่: [bold]lmds bench run <slug>[/bold]")
        return
    rows = sorted((summarize(r) for r in runs),
                  key=lambda r: r["speed"]["decode_tps_avg"] or 0, reverse=True)
    table = Table(title="คะแนนโมเดล")
    table.add_column("โมเดล")
    table.add_column("engine")
    table.add_column("เครื่อง")
    table.add_column("tok/s", justify="right")
    table.add_column("ยาวสุด", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("ความสามารถ", justify="right")
    table.add_column("วัดเมื่อ")
    for row in rows:
        speed = row["speed"]
        capability = row["capability"]
        table.add_row(
            row["slug"], row["engine"] or "", row["hostname"] or "",
            f"{speed['decode_tps_avg']:.1f}" if speed["decode_tps_avg"] else "—",
            f"{speed['decode_tps_long']:.1f}" if speed["decode_tps_long"] else "—",
            f"{speed['ttft_s_short']:.2f}s" if speed["ttft_s_short"] else "—",
            f"{capability['score']}/100" if capability["score"] is not None else "—",
            (row["stamped_at"] or "").replace("T", " ")[:16],
        )
    console.print(table)


@bench_app.command("show")
def bench_show(
    slug: str = typer.Argument(..., help="ชื่อ (slug)", autocompletion=_complete_slug),
    history: bool = typer.Option(False, "--history", help="ทุกรอบที่เคยวัด ไม่ใช่แค่ล่าสุด"),
) -> None:
    """ผลละเอียดของรอบล่าสุด (หรือทั้งประวัติด้วย --history)"""
    from lmds.bench import load, runs_for, speed_summary

    paths = runs_for(slug)
    if not paths:
        err_console.print(f"[red]ยังไม่เคยวัด {slug}[/red] — วัดเลย: lmds bench run {slug}")
        raise typer.Exit(code=1)

    if history:
        table = Table(title=f"ประวัติของ {slug}")
        table.add_column("วัดเมื่อ")
        table.add_column("build")
        table.add_column("tok/s", justify="right")
        table.add_column("ยาวสุด", justify="right")
        for path in paths:
            run = load(path)
            summary = speed_summary(run.get("workloads") or [])
            table.add_row(
                (run.get("stamped_at") or "").replace("T", " ")[:16],
                (run.get("environment") or {}).get("engine_build", "")[:20],
                f"{summary['decode_tps_avg']:.1f}" if summary["decode_tps_avg"] else "—",
                f"{summary['decode_tps_long']:.1f}" if summary["decode_tps_long"] else "—",
            )
        console.print(table)
        return

    run = load(paths[0])
    machine = run.get("machine") or {}
    console.print(f"[bold]{slug}[/bold] · {run.get('model_id')} · {run.get('engine')}")
    console.print(f"[dim]{machine.get('hostname')} · {', '.join(g['name'] for g in machine.get('gpus') or []) or 'ไม่มี GPU'}"
                  f" · วัดเมื่อ {(run.get('stamped_at') or '').replace('T', ' ')}[/dim]")
    _print_bench(run.get("workloads") or [], run.get("probes") or [])


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


@app.command()
def hardware() -> None:
    """ตรวจฮาร์ดแวร์ของเครื่องนี้และแสดง target profile"""
    from lmds.hardware import probe

    report = probe()
    table = Table(title="Hardware Report", show_header=False)
    table.add_row("Arch", report.arch)
    if report.gpus:
        for i, gpu in enumerate(report.gpus):
            if gpu.vram_mib:
                vram = f"{gpu.vram_mib / 1024:.0f} GB"
            elif gpu.known is not None:
                vram = f"{gpu.known.vram_gb:.0f} GB ({'unified' if gpu.known.memory_model.value == 'unified' else 'spec'})"
            else:
                vram = "?"
            cc = f"SM{gpu.compute_capability.replace('.', '')}" if gpu.compute_capability else "?"
            tested = "✅ tested" if gpu.tested else "⚠️ conservative"
            table.add_row(f"GPU {i}", f"{gpu.name} ({vram}, {cc}) {tested}")
    else:
        table.add_row("GPU", "ไม่พบ")
    from lmds.hardware.profiler import detect_mem, local_addresses, primary_ip

    total_gb, available_gb = detect_mem()
    if total_gb and available_gb is not None:
        used_gb = round(total_gb - available_gb, 1)
        table.add_row("RAM", f"ใช้ไป {used_gb} / {total_gb} GB (เหลือ {available_gb} GB)")
    else:
        table.add_row("RAM", f"{report.ram_gb} GB" if report.ram_gb else "ตรวจไม่ได้")
    if report.disk_free_gb is not None and report.disk_total_gb:
        used_disk = round(report.disk_total_gb - report.disk_free_gb, 1)
        warn = " ⚠️" if report.disk_free_gb < 50 else ""
        table.add_row(
            "Disk ($HOME)",
            f"ใช้ไป {used_disk} / {report.disk_total_gb} GB (เหลือ {report.disk_free_gb} GB){warn}",
        )
    else:
        table.add_row("Disk ($HOME)", "ตรวจไม่ได้")
    # ทุกเส้น ไม่ใช่แค่เส้นที่ default route ออก — บนเครื่องที่ NAT ออกไป (คอนเทนเนอร์,
    # VM ของ OrbStack) เส้นนั้นไม่ใช่เส้นที่คนอื่นใช้ยิงเข้ามา และเป็นค่าที่คนก๊อปไปใช้ผิดบ่อย
    addresses = local_addresses()
    table.add_row("IP", "\n".join(_address_label(a) for a in addresses) or primary_ip())
    table.add_row("Docker", "✅" if report.docker else "❌")
    table.add_row("NVIDIA Container Toolkit", "✅" if report.nvidia_container_toolkit else "❌")
    table.add_row("Profile", report.profile.value)
    from lmds.hardware.profiler import suggest_target
    target = suggest_target(report.gpus)
    table.add_row("Target สำหรับ deploy", f"--target {target}" if target else "ไม่มี preset ตรงรุ่น — ดู lmds deploy --help")
    console.print(table)
    for note in report.notes:
        err_console.print(f"[yellow]• {note}[/yellow]")


@config_app.command("set-provider")
def set_provider(
    name: ProviderName = typer.Argument(..., help="openai | gemini | minimax | anthropic | openai-compat"),
    model: str = typer.Option("", "--model", help="ชื่อโมเดล (ว่าง = ใช้ default ของ provider)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="จำเป็นสำหรับ openai-compat"),
) -> None:
    """เลือก LLM provider ที่ใช้เป็นสมองของระบบ"""
    settings = Settings.load()
    try:
        provider = settings.set_provider(name, model=model, base_url=base_url)
    except ValueError as exc:
        err_console.print(f"[red]ผิดพลาด:[/red] {exc}")
        raise typer.Exit(code=1)
    settings.save()
    console.print(f"ตั้งค่า provider: [bold]{provider.name.value}[/bold] (model: {provider.model})")
    if provider.base_url:
        console.print(f"base URL: {provider.base_url}")
    if get_secret(provider.name.value) is None:
        console.print(f"[yellow]ยังไม่มี API key — รัน: lmds config set-key {provider.name.value}[/yellow]")


@config_app.command("set-key")
def set_key(
    provider: ProviderName = typer.Argument(..., help="provider ที่จะตั้ง key"),
    stdin: bool = typer.Option(False, "--stdin", help="อ่าน key จาก stdin (สำหรับ scripting)"),
) -> None:
    """เก็บ API key ของ provider (keyring ถ้ามี, ไม่งั้นไฟล์ 0600) — ไม่แสดงบนจอ"""
    if stdin:
        import sys

        value = sys.stdin.readline().strip()
    else:
        value = typer.prompt(f"API key ของ {provider.value}", hide_input=True).strip()
    if not value:
        err_console.print("[red]ไม่ได้ใส่ค่า — ยกเลิก[/red]")
        raise typer.Exit(code=1)
    backend = set_secret(provider.value, value)
    console.print(f"บันทึก key ของ {provider.value} แล้ว (เก็บใน: {backend})")


@config_app.command("set-hf-token")
def set_hf_token(
    stdin: bool = typer.Option(False, "--stdin", help="อ่าน token จาก stdin (สำหรับ scripting)"),
) -> None:
    """เก็บ Hugging Face token (optional — ใช้กับ gated/private repo เท่านั้น)"""
    if stdin:
        import sys

        value = sys.stdin.readline().strip()
    else:
        value = typer.prompt("Hugging Face token (Enter เพื่อยกเลิก)", hide_input=True, default="").strip()
    if not value:
        console.print("ยกเลิก — ไม่ได้บันทึก token (repo สาธารณะใช้งานได้ตามปกติ)")
        raise typer.Exit(code=0)
    backend = set_secret("hf", value)
    console.print(f"บันทึก HF token แล้ว (เก็บใน: {backend})")


@config_app.command("show")
def show() -> None:
    """แสดง config ปัจจุบัน — secret ถูก mask เสมอ"""
    settings = Settings.load()
    console.print(f"config dir: {config_dir()}")

    table = Table(title="LMDS Configuration")
    table.add_column("รายการ")
    table.add_column("ค่า")

    if settings.provider:
        table.add_row("provider", settings.provider.name.value)
        table.add_row("model", settings.provider.model)
        if settings.provider.base_url:
            table.add_row("base URL", settings.provider.base_url)
    else:
        table.add_row("provider", "(ยังไม่ได้ตั้งค่า — lmds config set-provider ...)")
    table.add_row("default target", settings.defaults.target)
    table.add_row("ภาษา", settings.defaults.language)

    for secret_name in [*[p.value for p in ProviderName], "hf"]:
        source = secret_source(secret_name)
        label = f"key: {secret_name}" if secret_name != "hf" else "HF token"
        if source:
            table.add_row(label, f"{mask_preview(get_secret(secret_name))}  (จาก {source})")
        else:
            table.add_row(label, "(ไม่ได้ตั้งค่า)")
    console.print(table)

    warning = check_credentials_permissions()
    if warning:
        err_console.print(f"[yellow]{warning}[/yellow]")


@config_app.command("defaults")
def defaults_list() -> None:
    """แสดง default models ต่อ provider"""
    table = Table(title="Default models")
    table.add_column("provider")
    table.add_column("model")
    for provider, model in DEFAULT_MODELS.items():
        table.add_row(provider.value, model or "(ต้องระบุเอง)")
    console.print(table)


if __name__ == "__main__":
    app()
