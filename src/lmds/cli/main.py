"""LMDS CLI — entry point (`lmds`)"""

from __future__ import annotations

from typing import Optional

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

console = Console()
err_console = Console(stderr=True)


@app.command()
def version() -> None:
    """แสดงเวอร์ชันโปรแกรมและมาตรฐาน template"""
    console.print(f"lmds {lmds.__version__}")
    console.print(f"template standard: {lmds.TEMPLATE_STANDARD}")


@app.command()
def inspect(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision", help="branch/tag/commit ที่ต้องการ"),
    targets: list[str] = typer.Option(
        [], "--target", help="ประเมิน fit กับ target ที่ระบุ (ซ้ำได้) เช่น rtx-pro-4000 — ค่าว่าง = เครื่องนี้ + dgx-spark-single"
    ),
    concurrency: int = typer.Option(1, "--concurrency", help="จำนวน request พร้อมกันที่ใช้คำนวณ KV cache"),
    as_json: bool = typer.Option(False, "--json", help="พิมพ์ผลเป็น JSON (สำหรับ scripting)"),
) -> None:
    """วิเคราะห์โมเดลจากลิงก์ — ดึงเฉพาะ metadata ไม่ดาวน์โหลด weight

    ถ้าเป็น gated repo และยังไม่มี token จะถาม (กด Enter เพื่อข้ามได้)
    Exit codes: 0 สำเร็จ, 1 input ผิด, 4 ต้องการ token, 5 ปัญหาเครือข่าย/Hub
    """
    import sys

    from lmds.inspector import AuthRequired, HfClient, HfError, RepoNotFound, inspect_model
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
            report = inspect_model(source, HfClient(token=token))
        except AuthRequired as exc:
            interactive = sys.stdin.isatty() and not as_json
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
            console.print("[dim]hint: เก็บ token ถาวรด้วย lmds config set-hf-token[/dim]")
    except RepoNotFound as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except HfError as exc:
        err_console.print(f"[red]ปัญหาเครือข่าย/Hub:[/red] {exc}")
        raise typer.Exit(code=5)

    fit_reports = _compute_fits(report, targets, concurrency)

    if as_json:
        import json as json_module

        payload = {
            "model": report.model_dump(mode="json"),
            "fit": [f.model_dump(mode="json") for f in fit_reports],
        }
        print(json_module.dumps(payload, indent=2, ensure_ascii=False))
        return

    _render_report(report)
    _render_fits(fit_reports)


def _compute_fits(report, target_names: list[str], concurrency: int) -> list:
    from lmds.fit import PRESETS, analyze, from_hardware_report

    specs = []
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
            specs.append(detected)
        if not any(s.name.startswith("dgx-spark") for s in specs):
            specs.append(PRESETS["dgx-spark-single"])

    return [analyze(report, spec, concurrency=concurrency) for spec in specs]


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
    if report.artifact_type in (ArtifactType.GGUF, ArtifactType.MIXED) and report.gguf_variants:
        variants = [v for v in report.gguf_variants if not v.is_mmproj]
        mmproj = [v for v in report.gguf_variants if v.is_mmproj]
        table.add_row("GGUF variants", str(len(variants)) + (f" (+mmproj {len(mmproj)})" if mmproj else ""))
        if report.selected_gguf:
            table.add_row("Selected GGUF", report.selected_gguf)
    console.print(table)
    for warning in report.warnings:
        err_console.print(f"[yellow]⚠ {warning}[/yellow]")


@app.command()
def hardware() -> None:
    """ตรวจฮาร์ดแวร์ของเครื่องนี้และแสดง target profile"""
    from lmds.hardware import probe

    report = probe()
    table = Table(title="Hardware Report", show_header=False)
    table.add_row("Arch", report.arch)
    if report.gpus:
        for i, gpu in enumerate(report.gpus):
            vram = f"{gpu.vram_mib / 1024:.0f} GB" if gpu.vram_mib else "?"
            cc = f"SM{gpu.compute_capability.replace('.', '')}" if gpu.compute_capability else "?"
            tested = "✅ tested" if gpu.tested else "⚠️ conservative"
            table.add_row(f"GPU {i}", f"{gpu.name} ({vram}, {cc}) {tested}")
    else:
        table.add_row("GPU", "ไม่พบ")
    table.add_row("RAM", f"{report.ram_gb} GB" if report.ram_gb else "ตรวจไม่ได้")
    table.add_row("Docker", "✅" if report.docker else "❌")
    table.add_row("NVIDIA Container Toolkit", "✅" if report.nvidia_container_toolkit else "❌")
    table.add_row("Profile", report.profile.value)
    console.print(table)
    for note in report.notes:
        err_console.print(f"[yellow]• {note}[/yellow]")


@config_app.command("set-provider")
def set_provider(
    name: ProviderName = typer.Argument(..., help="openai | gemini | anthropic | openai-compat"),
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
