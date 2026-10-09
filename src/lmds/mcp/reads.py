"""ฟังก์ชันอ่านที่เครื่องมือ MCP ผูกได้ — ทั้งรายการอยู่ใน `READ_ALLOWLIST` ท้ายไฟล์ ไม่มีที่อื่น

ไม่มี logic ของตัวเอง: แต่ละตัวเรียกฟังก์ชันเดียวกับที่ `--json` ของ CLI หรือ payload ของหน้าเว็บเรียก แล้วคืนก้อนนั้นทั้งก้อน
(ที่มาเดียว — เทสเทียบผลของเครื่องมือกับ `lmds … --json` บนฟลีตจำลองเดียวกัน) · ถามเครื่องอื่น = คำสั่ง `lmds …` แบบอ่าน
ผ่าน `nodes.ssh.run` ทางเดิม: key เดิม timeout เดิม · เครื่องที่ต่อไม่ได้ = `ToolFailure` ของเครื่องนั้น ไม่ใช่ server ล้ม

argument ทุกตัวถูกตรวจมาแล้วที่ `tools.py` ก่อนถึงที่นี่ (ชื่อเครื่องต้องอยู่ในทะเบียน · slug ต้องผ่าน shellsafe.BUNDLE_SLUG ·
ตัวเลขอยู่ในช่วง) — ฟังก์ชันในไฟล์นี้ยังหาเครื่องจากทะเบียนเองอีกรอบ ไม่ประกอบคำสั่งจากชื่อที่ไม่ได้มาจากทะเบียน
"""

from __future__ import annotations

import io
import json
import shlex
from contextlib import contextmanager

# เพดานของ log ต่อคำขอ — ผู้ช่วยอ่านคำตอบทั้งก้อนเข้า context · 500 บรรทัดของ vLLM ตอนโหลดโมเดลก็เกินพออยู่แล้ว
LOG_LINES_DEFAULT = 100
LOG_LINES_MAX = 500
LOG_CHARS_MAX = 200_000
# timeout ของคำสั่งสั้นบนเครื่องอื่น — ค่าเดียวกับที่หน้าเว็บใช้กับ doctor/logs ของ node (web/api.node_command)
REMOTE_SHORT_TIMEOUT = 120
LOCAL_LOG_TIMEOUT = 60


class ToolFailure(Exception):
    """เครื่องมือตอบไม่ได้ด้วยเหตุที่ผู้ถามควรรู้ (เครื่องต่อไม่ได้ · ไม่รู้จัก slug · repo ไม่มี) — กลายเป็น isError

    `details` ไปกับคำตอบด้วย (เช่น `node` · `exit_code` · ตาราง fit ที่คำนวณได้ก่อนจะบอกว่าไม่พอ)
    """

    def __init__(self, message: str, **details) -> None:
        super().__init__(message)
        self.details = {k: v for k, v in details.items() if v is not None}


# ── ตัวช่วย ──────────────────────────────────────────────────────────────────────────────────────
def _node(name: str):
    """Node จากทะเบียน — ชื่อที่ไม่อยู่ในทะเบียนไม่มีวันกลายเป็นปลายทางของ ssh"""
    from lmds.nodes import NodeError, find

    try:
        node = find(name)
    except NodeError as exc:
        raise ToolFailure(str(exc)) from exc
    if node is None:
        raise ToolFailure(f"ไม่รู้จักเครื่อง '{name}' — ดูรายชื่อด้วยเครื่องมือ lmds_nodes")
    return node


def _remote(name: str, argv: list[str], timeout: int):
    """รัน `lmds …` แบบอ่านบนเครื่องนั้น — ต่อไม่ถึง/หมดเวลา = ToolFailure ของเครื่องนั้น พร้อมเหตุที่ ssh บอก"""
    from lmds.nodes import NodeError, run

    from .seal import REMOTE_PREFIX

    node = _node(name)
    # `LMDS_READ_ONLY=1 lmds …` — เครื่องที่ lmds รู้จักตัวแปรนี้ผนึก process ของตัวเองก่อนทำคำสั่ง
    command = " ".join([REMOTE_PREFIX, *(shlex.quote(part) for part in argv)])
    try:
        result = run(node, command, timeout=timeout)
    except (NodeError, OSError) as exc:
        raise ToolFailure(str(exc), node=name) from exc
    if result.exit_code == 124 or (result.exit_code == 255 and not (result.stdout or "").strip()):
        why = (result.stderr or "").strip()[-300:] or f"ssh จบด้วยรหัส {result.exit_code}"
        raise ToolFailure(f"ต่อ {node.target} ไม่ได้: {why}", node=name, exit_code=result.exit_code)
    return result


def _json_from(text: str, kind: type):
    """JSON ก้อนแรกชนิด `kind` ใน stdout ของเครื่องอื่น — ข้ามของที่ rc ของเครื่องนั้นพ่นมาข้างหน้า

    เคสจริง 2026-08-19 (dgx-70): login shell พ่น `declare -x …` 858 ไบต์ก่อน JSON ของ `lmds agent info` —
    เหตุเดียวกับ nodes.ssh._json_object ซึ่งรับได้แค่ object · ที่นี่ต้องรับ list ด้วย (`watchdog status --json`)
    """
    opener = "[" if kind is list else "{"
    decoder = json.JSONDecoder()
    start = text.find(opener)
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find(opener, start + 1)
            continue
        if isinstance(value, kind):
            return value
        start = text.find(opener, start + 1)
    return None


def _too_old(name: str, result, what: str) -> ToolFailure:
    text = ((result.stderr or "") + (result.stdout or "")).strip()
    if "No such option" in text or "No such command" in text or "no such option" in text.lower():
        return ToolFailure(f"{name} ยังไม่มี `{what}` — LMDS บนเครื่องนั้นเก่ากว่า hub · อัปเดตก่อน: lmds node install {name}",
                           node=name, exit_code=result.exit_code)
    return ToolFailure(text[-600:] or f"{name} ไม่ตอบผลของ `{what}`", node=name, exit_code=result.exit_code)


@contextmanager
def _cli_voice():
    """ฟังก์ชันของ CLI พูดกับคนผ่าน rich console แล้วจบด้วย `typer.Exit` — เก็บสิ่งที่มันพูดไว้เป็นข้อความ

    inspect/plan ใช้ทั้งเส้นของ CLI (parse ลิงก์ → ถาม Hub → fit → ตารางตัดสิน) เพื่อให้ได้คำตอบเดียวกับ `--json` ·
    เหตุที่ปฏิเสธ (repo ไม่มี · ไม่ fit · รูปแบบที่ไม่รองรับ) อยู่ในสิ่งที่มันพิมพ์บน stderr — ตรงนี้รับมาแทนจอ
    ไม่มีอะไรไปถึง stdout ของ process (ซึ่งเป็นท่อ JSON-RPC)
    """
    from rich.console import Console

    from lmds.cli import main as cli

    said = io.StringIO()
    voice = Console(file=said, force_terminal=False, no_color=True, width=100_000, highlight=False, emoji=False)
    saved = (cli.console, cli.err_console)
    cli.console = cli.err_console = voice
    try:
        yield said
    finally:
        cli.console, cli.err_console = saved


def _cli_exits() -> tuple:
    """ชนิดของ "CLI ขอจบด้วยรหัสนี้" — typer รุ่นเก่าใช้ของ click ตรง ๆ รุ่นใหม่ (0.2x) มีคลาสของตัวเอง ต้องจับทั้งคู่"""
    import typer

    kinds = [typer.Exit]
    try:
        import click

        kinds.append(click.exceptions.Exit)
    except ImportError:
        pass
    return tuple(dict.fromkeys(kinds))


def _through_cli(call):
    """เรียก `call()` ใต้ _cli_voice — คืน (ผล, บรรทัดที่ CLI พูด) · `typer.Exit` ≠ 0 = ToolFailure พร้อมเหตุและ exit code"""
    with _cli_voice() as said:
        try:
            value = call()
        except _cli_exits() as exc:
            lines = [line.strip() for line in said.getvalue().splitlines() if line.strip()]
            raise ToolFailure("\n".join(lines) or f"คำสั่งจบด้วยรหัส {exc.exit_code}", exit_code=exc.exit_code) from None
    return value, [line.strip() for line in said.getvalue().splitlines() if line.strip()]


class Noted(dict):
    """คำตอบที่มีคำเตือนแนบ (สิ่งที่ CLI พิมพ์บน stderr ทั้งที่สำเร็จ) — ตัวก้อนยังเท่ากับของ `--json` ทุกคีย์"""

    notes: list[str] = []


def _noted(payload: dict, notes: list[str]) -> dict:
    if not notes:
        return payload
    out = Noted(payload)
    out.notes = notes
    return out


# ── ฟังก์ชันอ่าน (ทั้งหมดที่เครื่องมือผูกได้) ─────────────────────────────────────────────────────────
def hub_version() -> dict:
    """ตัวตนของ hub: เวอร์ชัน + commit ที่ process นี้รัน · commit บนดิสก์ · มาตรฐาน template — ไม่ SSH ไม่ต่อเน็ต"""
    import socket

    import lmds
    from lmds.fleet.consistency import hub_facts
    from lmds.inventory import installed_commit

    facts = hub_facts()
    return {
        "version": facts["version"],
        "commit": facts["commit"],
        # ต่างจาก commit = ติดตั้งโค้ดใหม่แล้วแต่ process ที่รันอยู่ยังเป็นของเก่า (ความหมายเดียวกับ /api/version)
        "installed": installed_commit(),
        "template_standard": lmds.TEMPLATE_STANDARD,
        "template_hash": facts["template_hash"],
        "dirty": facts["dirty"],
        "hostname": socket.gethostname(),
    }


def nodes() -> dict:
    """ทะเบียนเครื่องทั้งใบตามที่ hub จำไว้ — ทุกฟิลด์ของ Node ชื่อเต็มไม่ตัด เรียงตามที่ผู้ใช้จัดไว้ · ไม่ SSH ไม่เขียน"""
    from dataclasses import asdict

    from lmds.config import Settings
    from lmds.nodes import NodeError, in_saved_order, load

    try:
        listed = in_saved_order(load(), Settings.load().ui.node_order)
    except NodeError as exc:
        raise ToolFailure(str(exc)) from exc
    return {"nodes": [asdict(node) for node in listed]}


def models(node: str = "") -> dict:
    """ก้อนของ `lmds agent info` — ของ hub เอง (ไม่ระบุเครื่อง) หรือของเครื่องในทะเบียน (SSH หนึ่งครั้ง · timeout ของ probe)"""
    if not node:
        from lmds.inventory import snapshot

        return snapshot()
    from lmds.nodes import NodeError, probe

    try:
        return probe(_node(node), read_only=True)
    except NodeError as exc:
        raise ToolFailure(str(exc), node=node) from exc


def inspect_repo(model: str, revision: str | None = None, targets: tuple[str, ...] = (), concurrency: int = 1,
                 context: int | None = None, kv_dtype: str = "bf16") -> dict:
    """ก้อนของ `lmds inspect <model> --json` — ถาม metadata จาก Hugging Face (ไม่โหลด weight)"""
    from lmds.cli.main import inspect_json

    payload, notes = _through_cli(lambda: inspect_json(
        model, revision=revision, targets=list(targets), concurrency=concurrency, context=context, kv_dtype=kv_dtype))
    return _noted(payload, notes)


def plan(model: str, revision: str | None = None, target: str | None = None, concurrency: int = 1,
         engine: str | None = None) -> dict:
    """ก้อนของ `lmds plan <model> --no-llm --json` — ตารางตัดสินล้วน · `no_llm=True` เขียนตายตัวที่นี่ ไม่มี argument เปิดได้"""
    from lmds.cli.main import _plan_and_fit

    (deployment_plan, _fit), notes = _through_cli(lambda: _plan_and_fit(
        model, revision=revision, target=target, no_llm=True, concurrency=concurrency, engine=engine,
        interactive_ok=False))
    return _noted(json.loads(deployment_plan.model_dump_json()), notes)


def fit(slug: str, node: str = "", slots: int | None = None, context: int | None = None) -> dict:
    """ก้อนของ `lmds fit <slug> --json` บนเครื่องนั้น — ตาราง RAM ของ slots/context ที่ถาม · dry run เสมอ ไม่มีทางเขียน"""
    if node:
        from lmds.web.fit import FitUnavailable, node_preview

        _node(node)
        try:
            return node_preview(node, slug, slots, context, read_only=True)
        except FitUnavailable as exc:
            raise ToolFailure(str(exc), node=node, plan=exc.plan) from exc
    from lmds.fleet import find
    from lmds.fleet.sizing import FitError, preview

    server = find(slug)
    if server is None or not server.controller:
        raise ToolFailure(f"ไม่รู้จัก '{slug}' บน hub — ดูรายชื่อด้วยเครื่องมือ lmds_models")
    try:
        return preview(server, slots=slots, context=context)
    except FitError as exc:
        raise ToolFailure(str(exc)) from exc


def fleet_check(check: bool = False) -> dict:
    """ก้อนของ `lmds fleet check [--check] --json` — code · controllers · runtime ต่อเครื่อง

    check=True ต่อเข้าทุกเครื่อง (SSH พร้อมกัน) แต่ **ไม่เขียนทะเบียน** ต่างจาก `--check` ของ CLI — เครื่องที่ต่อไม่ได้
    อยู่ในรายงานเป็น `reachable: false` พร้อม `error` ของเครื่องนั้น เครื่องอื่นยังได้ผลครบ
    """
    from lmds.cli.main import fleet_check_report

    report, _nodes = fleet_check_report(check, remember=False)
    return report


def watchdog_status(slug: str = "", node: str = "") -> list:
    """ก้อนของ `lmds watchdog status [slug] --json` ของ hub หรือของเครื่องในทะเบียน — `[]` = ไม่มีตัวไหนเปิดไว้"""
    if not node:
        from lmds.cli.main import watchdog_status_json

        return watchdog_status_json(slug)
    result = _remote(node, ["lmds", "watchdog", "status", *([slug] if slug else []), "--json"], REMOTE_SHORT_TIMEOUT)
    rows = _json_from(result.stdout or "", list)
    if rows is not None:
        return rows
    # lmds รุ่นก่อน 2026-10-09 พิมพ์ประโยคนี้ออก stdout แม้ขอ --json เมื่อไม่มีตัวไหนเปิด — ความหมายคือรายการว่าง
    if result.exit_code == 0 and "ยังไม่มี watchdog" in (result.stdout or ""):
        return []
    raise _too_old(node, result, "lmds watchdog status --json")


def logs(slug: str, node: str = "", lines: int = LOG_LINES_DEFAULT) -> dict:
    """ท้าย log ของโมเดล — hub: ก้อนของ GET /api/models/{slug}/logs · เครื่องอื่น: `lmds logs <slug> -n N` ผ่าน SSH"""
    lines = max(1, min(int(lines), LOG_LINES_MAX))
    if node:
        result = _remote(node, ["lmds", "logs", slug, "-n", str(lines)], REMOTE_SHORT_TIMEOUT)
        text = (result.stdout or "") + (result.stderr or "")
        if result.exit_code != 0:
            raise ToolFailure(text.strip()[-2000:] or f"lmds logs จบด้วยรหัส {result.exit_code}",
                              node=node, exit_code=result.exit_code)
        out = {"node": node, "slug": slug, "text": text}
    else:
        from lmds.fleet import FleetError, find, logs_text

        server = find(slug)
        if server is None:
            raise ToolFailure(f"ไม่รู้จัก '{slug}' บน hub — ดูรายชื่อด้วยเครื่องมือ lmds_models")
        try:
            # direct: ไม่รัน controller ของ bundle (สคริปต์ bash ที่อยู่นอกผนึกและเขียนได้) — แหล่งเดียวกับที่มันอ่าน
            out = {"slug": slug, "text": logs_text(server, lines, timeout=LOCAL_LOG_TIMEOUT, direct=True)}
        except FleetError as exc:
            raise ToolFailure(str(exc)) from exc
    if len(out["text"]) > LOG_CHARS_MAX:
        # เก็บท้ายไว้ — สาเหตุที่โมเดลล้มอยู่บรรทัดท้าย ๆ เสมอ (เหตุผลเดียวกับ assistant/runner._trim)
        out["text"] = out["text"][-LOG_CHARS_MAX:]
        out["truncated"] = True
    return out


def doctor(slug: str, node: str = "") -> dict:
    """ก้อนของ `lmds doctor <slug> --json --no-probe` — ทุกข้อของ doctor ยกเว้นการรัน container ไปถาม image (`skipped`)"""
    if node:
        result = _remote(node, ["lmds", "doctor", slug, "--json", "--no-probe"], REMOTE_SHORT_TIMEOUT)
        payload = _json_from(result.stdout or "", dict)
        if payload is None or "findings" not in payload:
            raise _too_old(node, result, "lmds doctor --json --no-probe")
        return payload
    from lmds.doctor import diagnose

    return diagnose(slug, probe=False).payload()


# รายการเดียวของสิ่งที่เครื่องมือผูกได้ — `tools.py` ปฏิเสธฟังก์ชันที่ไม่อยู่ในนี้ตอน import · เพิ่มชื่อที่นี่ = ยืนยันว่า
# อ่านมาแล้วว่ามันไม่เปลี่ยนสถานะ (และถึงพลาด process ก็ถูกผนึกไว้ — ดู seal.py)
READ_ALLOWLIST = frozenset({
    "hub_version", "nodes", "models", "inspect_repo", "plan", "fit", "fleet_check", "watchdog_status", "logs", "doctor",
})
