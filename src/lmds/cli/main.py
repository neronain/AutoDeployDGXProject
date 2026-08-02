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


@app.callback()
def _entry() -> None:
    """Local Model Deploy Studio — โดย neronain (fb.com/neronain.minidev)"""
    from .banner import show_banner

    show_banner(err_console)


@app.command()
def version() -> None:
    """แสดงเวอร์ชันโปรแกรมและมาตรฐาน template"""
    from .banner import CREDIT

    console.print(f"lmds {lmds.__version__}")
    console.print(f"template standard: {lmds.TEMPLATE_STANDARD}")
    console.print(f"[dim]{CREDIT}[/dim]")


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
    _, report = _resolve_and_inspect(model, revision, interactive_ok=not as_json)
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


def _build_plan_safe(report, fit, provider):
    """เรียก LLM วางแผน — ถ้า provider ล้ม (quota/เครือข่าย/schema) สลับเป็น rule-based พร้อมแจ้งชัด"""
    from lmds.brain import PlanError, ProviderError, build_plan

    if provider is not None:
        try:
            return build_plan(report, fit, provider)
        except (PlanError, ProviderError) as exc:
            err_console.print(f"[yellow]LLM ใช้ไม่ได้: {exc}[/yellow]")
            err_console.print("[yellow]→ สลับเป็น rule-based mode อัตโนมัติ (plan จะไม่มีการวิเคราะห์เชิงลึก)[/yellow]")
    return build_plan(report, fit, None)


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
            console.print("[dim]hint: เก็บ token ถาวรด้วย lmds config set-hf-token[/dim]")
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


def _ensure_gguf_selected(source, report, interactive: bool):
    """repo GGUF หลาย variant ที่ยังไม่เลือกไฟล์ — ให้เลือกตั้งแต่ต้น flow ไม่ใช่ไปพังตอนท้าย

    interactive: แสดงรายการให้เลือกหมายเลข แล้ว inspect ซ้ำด้วยไฟล์ที่เลือก (ได้ header/kv dims จริง)
    non-interactive: จบพร้อมวิธีระบุไฟล์ตรง
    """
    from lmds.inspector import ArtifactType

    weight_variants = [v for v in report.gguf_variants if not v.is_mmproj]
    if report.artifact_type is not ArtifactType.GGUF or report.selected_gguf or len(weight_variants) <= 1:
        return report

    variants = sorted(weight_variants, key=lambda v: v.size_bytes or 0)
    if not interactive:
        err_console.print(
            f"[red]repo นี้มี GGUF {len(variants)} variant — ต้องระบุไฟล์ (โหมด non-interactive)[/red]"
        )
        for variant in variants[:8]:
            size = f"{variant.size_bytes / 1e9:.1f} GB" if variant.size_bytes else "?"
            err_console.print(f"  • {variant.filename} ({size})")
        err_console.print(
            f'\nระบุไฟล์ด้วยลิงก์ตรง เช่น:\n  lmds deploy "https://huggingface.co/{report.repo_id}/blob/main/{variants[0].filename}"'
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
    chosen = variants[choice - 1]

    # inspect ซ้ำด้วยไฟล์ที่เลือก → ได้ GGUF header (architecture/context/kv dims) มาคำนวณ fit จริง
    from dataclasses import replace as dc_replace

    from lmds.inspector import HfClient, HfError, inspect_model

    try:
        return inspect_model(dc_replace(source, filename=chosen.filename), HfClient(token=get_secret("hf")))
    except HfError as exc:
        err_console.print(f"[yellow]อ่าน header ของไฟล์ที่เลือกไม่ได้ ({exc}) — ใช้ขนาดไฟล์อย่างเดียว[/yellow]")
        report.selected_gguf = chosen.filename
        report.weight_bytes = chosen.size_bytes
        return report


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
def plan(
    model: str = typer.Argument(..., help="ลิงก์ Hugging Face หรือ org/model"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    target: Optional[str] = typer.Option(
        None, "--target", help="target preset (เช่น dgx-spark-single) — ว่าง = เครื่องนี้ หรือ dgx-spark-single"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
    as_json: bool = typer.Option(False, "--json"),
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

    deployment_plan = _build_plan_safe(report, fit, provider)

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
    target: Optional[str] = typer.Option(None, "--target", help="target preset — ว่าง = เครื่องนี้/dgx-spark-single · dgx-spark-stacked = multi-node (2 เครื่อง)"),
    output: str = typer.Option("./bundles", "--output", help="โฟลเดอร์ output ของ bundle"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
) -> None:
    """สร้าง deployment bundle: plan → render controller/README/MODEL_PROFILE (ยังไม่ validate/zip — M6)

    Exit codes: 0 สำเร็จ, 1 input ผิด, 3 โมเดลไม่ fit, 4 ต้องการ token, 5 ปัญหา provider
    """
    from pathlib import Path

    from lmds.brain import MissingKey, PlanError, ProviderError, build_plan, make_provider
    from lmds.config import Settings
    from lmds.fit import Verdict
    from lmds.generator import render_bundle

    source, report = _resolve_and_inspect(model, revision, interactive_ok=True)
    report = _ensure_gguf_selected(source, report, interactive=False)
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

    deployment_plan = _build_plan_safe(report, fit, provider)

    bundle, results, delivered = _render_and_package(deployment_plan, report, fit, output)
    _render_plan(deployment_plan, fit)
    _render_gates(results)
    _render_delivery(bundle, delivered, native_prepare=_is_native_prepare(deployment_plan, fit),
                     stacked=deployment_plan.topology.value == "stacked",
                     assets=bool(deployment_plan.runtime_assets))


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
    target: Optional[str] = typer.Option(None, "--target", help="target preset — ว่าง = เครื่องนี้/dgx-spark-single · dgx-spark-stacked = multi-node (2 เครื่อง)"),
    output: str = typer.Option("./bundles", "--output"),
    no_llm: bool = typer.Option(False, "--no-llm", help="rule-based mode: ไม่เรียก LLM"),
    concurrency: int = typer.Option(1, "--concurrency"),
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
    report = _ensure_gguf_selected(source, report, interactive=interactive)
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

    deployment_plan = _build_plan_safe(report, fit, provider)

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

        context_input = typer.prompt(
            "context (Enter = ใช้ค่าตามแผน)", default=str(deployment_plan.serving.context)
        ).strip()
        if context_input.isdigit() and int(context_input) > 0:
            requested = int(context_input)
            ceiling = fit.max_safe_context or deployment_plan.serving.context
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


def _render_host_panel() -> None:
    from lmds.hardware import host_summary

    host = host_summary()
    parts = [f"[bold]{host.hostname}[/bold]", host.ip, host.arch, f"profile: {host.profile.value}"]
    console.print("🖥  " + " · ".join(parts))
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


@app.command()
def ps() -> None:
    """แสดงเครื่อง + ทุกโมเดลที่ deploy ในเครื่องนี้ พร้อมสถานะจริง (running/health/endpoint)"""
    from lmds.fleet import discover

    _render_host_panel()
    servers = discover()
    if not servers:
        console.print("ยังไม่มีโมเดลที่เคย start ในเครื่องนี้ — deploy ก่อน: lmds deploy <model-url>")
        return
    table = Table(title="LMDS Fleet")
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
            "\n[dim]ใช้ชื่อจากคอลัมน์แรกกับทุกคำสั่ง เช่น:[/dim]\n"
            f"  lmds logs {example} -f      [dim]# ดู log realtime (Ctrl-C ออก ไม่หยุดโมเดล)[/dim]\n"
            f"  lmds restart {example}\n"
            f"  lmds stop {example}         [dim]# หรือ lmds stop --all[/dim]"
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
    slug: Optional[str] = typer.Argument(None, help="ชื่อ (slug) จาก lmds ps"),
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
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps"),
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
    try:
        raise typer.Exit(code=logs_server(server, lines, follow=follow))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=0)


@app.command()
def restart(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list"),
) -> None:
    """restart โมเดลตามชื่อ — ใช้ได้กับ container ที่ไม่ได้มาจาก lmds ด้วย"""
    from lmds.fleet import FleetError, find, restart_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps")
        raise typer.Exit(code=1)
    try:
        method = restart_server(server)
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"restart {slug} แล้ว ({method})")


@app.command()
def start(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list"),
) -> None:
    """รันโมเดลที่เคย deploy ไว้แล้วตามชื่อ — ไม่ต้อง cd ไปหา bundle"""
    from lmds.fleet import FleetError, find, start_server

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps หรือ lmds list")
        raise typer.Exit(code=1)
    if server.running:
        console.print(f"{slug} รันอยู่แล้ว (port {server.port})")
        return
    try:
        raise typer.Exit(code=start_server(server))
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def enable(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list"),
    now: bool = typer.Option(False, "--now", help="สั่ง start ทันทีด้วย (ไม่รอ reboot)"),
    timeout: int = typer.Option(1800, "--timeout", help="วินาทีที่รอตอน start ใน service (โมเดลใหญ่ควรเพิ่ม)"),
) -> None:
    """ตั้งให้โมเดลกลับมาทำงานเองหลังเปิด-ปิดเครื่อง (systemd autostart) — ใช้ sudo"""
    from lmds.fleet import FleetError, enable_autostart, find, unit_name

    server = find(slug)
    if server is None:
        err_console.print(f"[red]ไม่พบ: {slug}[/red] — ดูรายชื่อ: lmds ps หรือ lmds list")
        raise typer.Exit(code=1)
    console.print(f"ติดตั้ง autostart สำหรับ [bold]{slug}[/bold] (จะขอ sudo เพื่อเขียน systemd unit)…")
    try:
        name = enable_autostart(server, timeout=timeout, start_now=now)
    except FleetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✅ เปิด autostart แล้ว[/green] ({name}) — โมเดลจะกลับมาเองหลัง reboot")
    console.print(f"[dim]เช็ก: systemctl status {name} | ปิด: lmds disable {slug}[/dim]")
    if not now:
        console.print(f"[dim]start เดี๋ยวนี้เลย: lmds start {slug}  (หรือ lmds enable {slug} --now)[/dim]")


@app.command()
def disable(
    slug: str = typer.Argument(..., help="ชื่อ (slug) จาก lmds ps / lmds list"),
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
        state = "[red]⚠[/red]" if not server.controller_exists else _status_symbol(server)
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
        "[dim]คอลัมน์แรก (slug) คือชื่อที่ใช้กับทุกคำสั่ง เช่น:[/dim]\n"
        f"  lmds start {first}   ·   lmds stop {first}   ·   lmds restart {first}\n"
        f"  lmds logs {first} -f   [dim]# realtime[/dim]   ·   lmds enable {first}   [dim]# autostart[/dim]\n"
        "[dim]endpoint + สถานะ health เต็ม ๆ: lmds ps[/dim]"
    )


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
    from lmds.hardware.profiler import detect_mem, primary_ip

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
    table.add_row("IP", primary_ip())
    table.add_row("Docker", "✅" if report.docker else "❌")
    table.add_row("NVIDIA Container Toolkit", "✅" if report.nvidia_container_toolkit else "❌")
    table.add_row("Profile", report.profile.value)
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
