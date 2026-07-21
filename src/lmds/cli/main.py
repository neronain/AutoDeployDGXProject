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
