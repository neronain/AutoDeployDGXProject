import zipfile

from typer.testing import CliRunner

from lmds.brain import apply_flag_approvals, build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from tests.test_generator import gguf_report, safetensors_report

runner = CliRunner()


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)


def test_deploy_yes_produces_validated_zip(isolated_config, tmp_path, monkeypatch):
    _patch_inspect(monkeypatch, gguf_report())
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 0, result.output
    zip_path = tmp_path / "qwen3-8b-gguf.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert any(n.endswith("PACKAGE_SHA256SUMS") for n in archive.namelist())
    assert "static-validated" in result.output


def test_deploy_no_fit_exit_3(isolated_config, tmp_path, monkeypatch):
    # BF16 65GB บน RTX 24GB เดี่ยว → needs-smaller-quant → exit 3 พร้อมทางเลือก
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(
        app,
        ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "rtx-pro-4000",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 3
    assert "ไม่ fit" in result.output


def test_deploy_non_tty_skips_confirmation(isolated_config, tmp_path, monkeypatch):
    # CliRunner ไม่ใช่ tty → ต้องไม่ค้างรอ input แม้ไม่มี --yes
    _patch_inspect(monkeypatch, gguf_report())
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
         "--output", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output


def test_deploy_falls_back_to_rule_based_on_provider_error(isolated_config, tmp_path, monkeypatch):
    """เคสจริงจากเครื่อง gigabyte02: Gemini 429 → ต้องสลับ rule-based ไม่ใช่หยุดทำงาน"""
    from lmds.brain import ProviderError
    from lmds.config import ProviderName, Settings

    class QuotaExceededProvider:
        name = "gemini"
        model = "gemini-2.5-pro"

        def complete_json(self, system, user):
            raise ProviderError("gemini ตอบ HTTP 429: quota exceeded")

    settings = Settings.load()
    settings.set_provider(ProviderName.GEMINI)
    settings.save()
    monkeypatch.setattr("lmds.brain.make_provider", lambda c, k, client=None: QuotaExceededProvider())
    monkeypatch.setattr("lmds.secrets.store.get_secret", lambda n: "AIzaFakeKey123" if n == "gemini" else None)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda n: "AIzaFakeKey123" if n == "gemini" else None)
    _patch_inspect(monkeypatch, gguf_report())

    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--target", "dgx-spark-single",
         "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "rule-based" in result.output
    assert "429" in result.output  # แจ้งสาเหตุที่สลับ


def test_spark_unified_memory_from_allowlist_when_smi_reports_none(isolated_config):
    """GB10 บนเครื่องจริง: nvidia-smi ไม่รายงาน memory.total → ใช้สเปก 128GB จาก allowlist"""
    from lmds.fit import from_hardware_report
    from lmds.hardware import MemoryModel
    from lmds.hardware.profiler import DetectedGpu, HardwareReport
    from lmds.hardware.profiles import TargetProfile, lookup_gpu

    report = HardwareReport(
        arch="aarch64",
        gpus=[DetectedGpu("NVIDIA GB10", None, "12.1", lookup_gpu("NVIDIA GB10"))],
        ram_gb=121.7,
        profile=TargetProfile.DGX_SPARK_SINGLE,
    )
    spec = from_hardware_report(report)
    assert spec is not None
    assert spec.memory_gb == 128.0
    assert spec.memory_model is MemoryModel.UNIFIED
    assert spec.tested is True


def test_deploy_multi_variant_gguf_fails_early_with_guidance(isolated_config, tmp_path, monkeypatch):
    """เคสจริงจาก gigabyte02: repo หลาย variant ต้องแจ้งตั้งแต่ต้น flow ไม่ใช่หลังยืนยันแผน"""
    from lmds.inspector.report import GgufVariant

    report = gguf_report(
        selected_gguf=None,
        weight_bytes=None,
        gguf_variants=[
            GgufVariant(filename="m-Q4_K_M.gguf", size_bytes=18 * GIB),
            GgufVariant(filename="m-Q8_0.gguf", size_bytes=32 * GIB),
        ],
    )
    _patch_inspect(monkeypatch, report)
    result = runner.invoke(
        app,
        ["deploy", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "--no-llm",
         "--target", "dgx-spark-single", "--output", str(tmp_path), "--yes"],
    )
    assert result.exit_code == 1
    assert "m-Q4_K_M.gguf" in result.output       # แสดงรายการ variant
    assert "blob/main" in result.output           # บอกวิธีระบุไฟล์ตรง
    assert "ยืนยัน" not in result.output           # ต้องจบก่อนถึงขั้นยืนยัน


def test_ensure_gguf_selected_interactive_reinspects(isolated_config, monkeypatch):
    """โหมด interactive: เลือกหมายเลขแล้วต้อง inspect ซ้ำด้วยไฟล์ที่เลือก"""
    import typer

    from lmds.cli.main import _ensure_gguf_selected
    from lmds.inspector.report import GgufVariant
    from lmds.resolver import parse_source

    report = gguf_report(
        selected_gguf=None,
        weight_bytes=None,
        gguf_variants=[
            GgufVariant(filename="m-Q4_K_M.gguf", size_bytes=18 * GIB),
            GgufVariant(filename="m-Q8_0.gguf", size_bytes=32 * GIB),
        ],
    )
    seen = {}

    def fake_inspect(source, client):
        seen["filename"] = source.filename
        return gguf_report(selected_gguf=source.filename, weight_bytes=18 * GIB)

    monkeypatch.setattr("lmds.inspector.inspect_model", fake_inspect)
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: 1)  # เลือกข้อ 1 (เรียงจากเล็กไปใหญ่)

    source = parse_source("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
    out = _ensure_gguf_selected(source, report, interactive=True)
    assert seen["filename"] == "m-Q4_K_M.gguf"
    assert out.selected_gguf == "m-Q4_K_M.gguf"


def test_ensure_gguf_selected_passthrough_when_selected(isolated_config):
    from lmds.cli.main import _ensure_gguf_selected
    from lmds.resolver import parse_source

    report = gguf_report()  # selected_gguf ตั้งแล้ว
    out = _ensure_gguf_selected(parse_source("unsloth/Qwen3-8B-GGUF"), report, interactive=False)
    assert out is report


def test_apply_flag_approvals_moves_flag(isolated_config):
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.flags_needing_approval = ["--trust-remote-code", "--weird"]

    apply_flag_approvals(plan, ["--trust-remote-code"])
    assert "--trust-remote-code" in plan.serving.extra_flags
    assert plan.flags_needing_approval == ["--weird"]
    assert any("อนุมัติ" in w for w in plan.warnings)


def test_approved_flag_survives_into_script(isolated_config, tmp_path):
    from lmds.generator import render_bundle

    report = gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.flags_needing_approval = ["--no-mmap"]
    apply_flag_approvals(plan, ["--no-mmap"])

    bundle = render_bundle(plan, report, fit, tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "--no-mmap" in text


# ── lmds up (deploy แล้วเดินต่อจนเซิร์ฟเวอร์ตอบได้) ──────────────────────


class _FakeController:
    """แทนการเรียก controller — ปล่อย subprocess อื่น (เช่น bash -n ของ gate) ผ่านของจริง"""

    def __init__(self, fail_on: str = ""):
        import subprocess

        self.calls: list[str] = []
        self.fail_on = fail_on
        self._real = subprocess.run

    def __call__(self, args, **kwargs):
        import subprocess

        is_controller = (
            isinstance(args, (list, tuple)) and len(args) == 2 and str(args[0]).endswith(".sh")
        )
        if not is_controller:
            return self._real(args, **kwargs)
        self.calls.append(args[1])
        return subprocess.CompletedProcess(args, 3 if args[1] == self.fail_on else 0)


def _run_up(monkeypatch, tmp_path, report, target: str, model: str, fail_on: str = ""):
    _patch_inspect(monkeypatch, report)
    fake = _FakeController(fail_on)
    monkeypatch.setattr("subprocess.run", fake)
    result = runner.invoke(
        app,
        ["up", model, "--no-llm", "--target", target, "--output", str(tmp_path), "--yes"],
    )
    return result, fake


def test_up_runs_controller_steps_in_the_only_order_that_works(
    isolated_config, tmp_path, monkeypatch
):
    """start ก่อน download = ไม่มีไฟล์ · ข้าม verify-files = ไฟล์ครึ่งเดียวแล้วไปตายตอนโหลด"""
    result, fake = _run_up(
        monkeypatch, tmp_path, gguf_report(), "dgx-spark-single", "unsloth/Qwen3-8B-GGUF"
    )
    assert result.exit_code == 0, result.output
    # prepare-runtime แทรกเฉพาะ bundle ที่ build เอง (llama.cpp บน ARM64 ไม่มี image ทางการ)
    assert fake.calls == ["download", "verify-files", "prepare-runtime", "start", "test-text"]
    assert "พร้อมใช้งานแล้ว" in result.output
    assert "lmds connect qwen3-8b-gguf" in result.output


def test_up_stops_at_the_failing_step_and_says_where_to_look(
    isolated_config, tmp_path, monkeypatch
):
    """ล้มแล้วต้องรู้ทันทีว่าล้มขั้นไหนและดูต่อที่ไหน ไม่ใช่ไล่อ่าน log เอง"""
    result, fake = _run_up(
        monkeypatch, tmp_path, gguf_report(), "dgx-spark-single", "unsloth/Qwen3-8B-GGUF",
        fail_on="verify-files",
    )
    assert result.exit_code == 3  # คืน exit code ของ controller ตรง ๆ
    assert fake.calls == ["download", "verify-files"]  # ไม่เดินต่อหลังล้ม
    assert "ไม่ผ่านขั้น verify-files" in result.output
    assert "lmds doctor qwen3-8b-gguf" in result.output


def test_up_skips_prepare_runtime_when_bundle_uses_a_prebuilt_image(
    isolated_config, tmp_path, monkeypatch
):
    """bundle ที่รันด้วย docker image ไม่มีขั้น build — สั่งไปก็ไม่มีคำสั่งนั้นใน controller"""
    result, fake = _run_up(
        monkeypatch, tmp_path, gguf_report(weight_bytes=5 * GIB), "rtx-pro-4000",
        "unsloth/Qwen3-8B-GGUF",
    )
    assert result.exit_code == 0, result.output
    assert "prepare-runtime" not in fake.calls
    assert fake.calls == ["download", "verify-files", "start", "test-text"]


def test_up_does_not_pretend_to_handle_stacked(isolated_config, tmp_path, monkeypatch):
    """stacked มีขั้น sync-worker/verify-worker ที่ต้องตัดสินใจเรื่องเครื่องปลายทาง — ทำแทนไม่ได้

    ทำครึ่ง ๆ แล้วปล่อยไว้แย่กว่าบอกตรง ๆ ว่าต้องทำต่อเองตาม README
    """
    from tests.test_stacked import big_safetensors

    result, fake = _run_up(
        monkeypatch, tmp_path, big_safetensors(), "dgx-spark-stacked",
        "nvidia/DeepSeek-V4-Flash-NVFP4",
    )
    assert result.exit_code == 1, result.output
    assert fake.calls == []
    assert "stacked" in result.output
    assert "README" in result.output
