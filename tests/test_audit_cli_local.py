"""audit ของคำสั่ง local ใน `lmds` (2026-10-06) — ทุกข้อรันผ่าน entry point จริง (typer CliRunner) แล้วดูผลบนดิสก์/exit code

หัวข้อเรียงตามข้อใน audit:
  1. `rebuild` ต้องสร้าง *ใบที่สั่ง* ใหม่ ไม่ใช่ใบข้าง ๆ          (ท้ายไฟล์)
  2. `deploy --name` ทับ bundle ของ repo อื่น
  3. `smoke` หยุดโมเดลที่เสิร์ฟอยู่ก่อนแล้ว
  4. `PlanError` หลุดเป็น traceback
  5. input ผิดธรรมดา ๆ จบด้วย traceback / exit code ผิดช่อง

rich ตัดบรรทัดตามความกว้างจอ — เทียบข้อความด้วย `_flat()` (ตัด whitespace ทิ้ง) เสมอ
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from tests.test_generator import safetensors_report
from tests.test_kv_sizing import spark_head  # noqa: F401 — fixture: bundle Nemotron + ข้อเท็จจริงของ spark-head

GIB = 1024**3


def _runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)      # click < 8.2: ต้องขอแยก stderr เอง
    except TypeError:
        return CliRunner()                      # click ≥ 8.2 แยกให้เสมอ


def _run(args, input=None):
    return _runner().invoke(app, args, input=input)


def _flat(text: str) -> str:
    return "".join(text.split())


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model",
                        report if callable(report) else (lambda s, c: report))


def _embed_report() -> ModelReport:
    return ModelReport(
        repo_id="Qwen/Qwen3-Embedding-4B", revision_sha="sha", task="embed",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(8 * GIB),
        architecture="Qwen3ForCausalLM", context_length=32768,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
    )


def _clean_refusal(result, *needles: str) -> None:
    """ปฏิเสธแบบที่สัญญาไว้: exit 1 · ไม่มี exception หลุด · เหตุผลอยู่บน stderr"""
    assert not (result.exception and not isinstance(result.exception, SystemExit)), (
        f"หลุดเป็น {type(result.exception).__name__}: {result.exception}")
    assert result.exit_code == 1, result.output
    said = _flat(result.stderr)
    for needle in needles:
        assert _flat(needle) in said, result.stderr


# ═════════════════════ 4. PlanError ต้องเป็นข้อความแดง + exit 1 ไม่ใช่ traceback ═════════════════════
@pytest.mark.parametrize("command", ["plan", "generate", "deploy"])
def test_an_embedding_model_on_a_stacked_target_is_refused_cleanly(tmp_path, monkeypatch, command):
    """เคส audit: `lmds plan|deploy <embedding> --no-llm --target dgx-spark-stacked` จบด้วย traceback ของ PlanError"""
    _patch_inspect(monkeypatch, _embed_report())
    args = [command, "Qwen/Qwen3-Embedding-4B", "--no-llm", "--target", "dgx-spark-stacked"]
    if command != "plan":
        args += ["--output", str(tmp_path / "bundles")]
    if command == "deploy":
        args += ["--yes"]
    result = _run(args)
    _clean_refusal(result, "รันเครื่องเดียวเสมอ")
    assert not (tmp_path / "bundles").exists(), "ปฏิเสธแล้วต้องไม่มีอะไรลงดิสก์"


@pytest.mark.parametrize("command", ["plan", "generate", "deploy"])
def test_sglang_on_a_stacked_target_is_refused_cleanly(tmp_path, monkeypatch, command):
    _patch_inspect(monkeypatch, safetensors_report())
    args = [command, "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-stacked", "--engine", "sglang"]
    if command != "plan":
        args += ["--output", str(tmp_path / "bundles")]
    if command == "deploy":
        args += ["--yes"]
    result = _run(args)
    _clean_refusal(result, "SGLang ยังไม่มี controller แบบ stacked")
    assert not (tmp_path / "bundles").exists()


def test_plan_json_keeps_stdout_empty_when_the_plan_is_refused(monkeypatch):
    """สคริปต์ที่อ่าน `plan --json` ต้องได้ stdout ว่าง + exit ≠ 0 — ไม่ใช่ traceback และไม่ใช่ JSON ครึ่งก้อน"""
    _patch_inspect(monkeypatch, safetensors_report())
    result = _run(["plan", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-stacked",
                   "--engine", "sglang", "--json"])
    _clean_refusal(result, "SGLang")
    assert result.stdout.strip() == ""


class _CountingProvider:
    """provider ปลอมที่นับว่าถูกเรียกกี่ครั้ง — คืน plan ที่ผ่าน schema เพื่อให้ไปถึง harden"""

    name, model = "gemini", "gemini-test"

    def __init__(self, payload: str = ""):
        self.calls = 0
        self.payload = payload

    def complete_json(self, system, user):
        self.calls += 1
        return self.payload


def _with_provider(monkeypatch, provider) -> None:
    from lmds.config import ProviderName, Settings

    settings = Settings.load()
    settings.set_provider(ProviderName.GEMINI)
    settings.save()
    monkeypatch.setattr("lmds.brain.make_provider", lambda c, k, client=None: provider)
    monkeypatch.setattr("lmds.cli.main.get_secret", lambda n: "AIzaFakeKey123" if n == "gemini" else None)


def _llm_plan_json(report, target: str) -> str:
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze

    single = report.model_copy(update={"task": "generate"})
    return build_plan(single, analyze(single, PRESETS[target]), None).model_dump_json()


@pytest.mark.parametrize("command", ["plan", "deploy"])
def test_an_input_refusal_is_not_blamed_on_the_llm(tmp_path, monkeypatch, command):
    """มี provider ตั้งไว้ + embedding บน stacked: สาเหตุคือ input ไม่ใช่ LLM

    เดิม: เรียก LLM (เสียเงิน) → harden โยน PlanError → พิมพ์ "LLM ใช้ไม่ได้ … สลับเป็น rule-based" →
    rule-based โยน PlanError ซ้ำ → traceback · ต้องปฏิเสธก่อนเรียก LLM และไม่โทษ LLM
    """
    report = _embed_report()
    provider = _CountingProvider(_llm_plan_json(report, "dgx-spark-stacked"))
    _with_provider(monkeypatch, provider)
    _patch_inspect(monkeypatch, report)
    args = [command, report.repo_id, "--target", "dgx-spark-stacked"]
    if command == "deploy":
        args += ["--yes", "--output", str(tmp_path / "bundles")]
    result = _run(args)
    _clean_refusal(result, "รันเครื่องเดียวเสมอ")
    assert "LLMใช้ไม่ได้" not in _flat(result.stderr), result.stderr
    assert "rule-basedmode" not in _flat(result.stderr), result.stderr
    assert provider.calls == 0, "input ที่ไม่มีทางวางแผนได้ ต้องไม่เสียคำขอ LLM"


def test_an_llm_that_cannot_produce_a_plan_still_falls_back_to_rule_based(tmp_path, monkeypatch):
    """ทางเดิมต้องไม่หาย: LLM ตอบขยะครบโควตา → บอกว่า LLM ใช้ไม่ได้ แล้วได้ plan แบบ rule-based (exit 0)"""
    provider = _CountingProvider("not json at all")
    _with_provider(monkeypatch, provider)
    _patch_inspect(monkeypatch, safetensors_report(weight_bytes=20 * GIB))
    result = _run(["plan", "Qwen/Qwen3-32B", "--target", "dgx-spark-single", "--json"])
    assert result.exit_code == 0, result.output
    assert provider.calls >= 1
    assert "LLMใช้ไม่ได้" in _flat(result.stderr)
    assert json.loads(result.stdout)["generator"] == "rule-based"


# ═════════════════════ 5. input ผิดธรรมดา ๆ = ข้อความ + exit 1 · ไม่ใช่ traceback ไม่ใช่ exit ของช่องอื่น ═════════════════════
def _no_crash(result) -> None:
    assert not (result.exception and not isinstance(result.exception, SystemExit)), (
        f"หลุดเป็น {type(result.exception).__name__}: {result.exception}")


def _argv(command: str, tmp_path, *extra: str) -> list[str]:
    args = [command, "Qwen/Qwen3-32B", "--target", "dgx-spark-single"]
    if command != "inspect":
        args += ["--no-llm"]
    if command in ("generate", "deploy"):
        args += ["--output", str(tmp_path / "bundles")]
    if command == "deploy":
        args += ["--yes"]
    return args + list(extra)


@pytest.mark.parametrize("command", ["inspect", "plan", "generate", "deploy"])
@pytest.mark.parametrize("value", ["0", "-3"])
def test_concurrency_below_one_is_bad_input_not_a_crash_or_a_no_fit(tmp_path, monkeypatch, command, value):
    """`--concurrency 0` → ZeroDivisionError ที่ fit/analyzer.py · `--concurrency -3` → "ไม่ fit" exit 3 ทั้งที่โมเดล 20 GB
    ใส่ DGX Spark ได้สบาย (context ที่คำนวณได้ติดลบ) — ทั้งคู่คือ input ผิด = exit 1"""
    _patch_inspect(monkeypatch, safetensors_report(weight_bytes=20 * GIB))
    result = _run(_argv(command, tmp_path, "--concurrency", value))
    _clean_refusal(result, "--concurrency", "ตั้งแต่ 1 ขึ้นไป")
    assert result.stdout.strip() == ""
    assert not (tmp_path / "bundles").exists()


def test_fit_analysis_itself_rejects_a_concurrency_it_cannot_divide_by():
    """ชั้นล่างต้องปฏิเสธเองด้วย — ผู้เรียกที่ไม่ได้มาทาง option (เว็บ/สคริปต์) ต้องไม่ได้ ZeroDivisionError หรือตัวเลขติดลบ"""
    from lmds.fit import PRESETS, analyze

    report = safetensors_report(weight_bytes=20 * GIB)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="concurrency"):
            analyze(report, PRESETS["dgx-spark-single"], concurrency=bad)
    assert analyze(report, PRESETS["dgx-spark-single"], concurrency=2).concurrency == 2


def test_a_concurrency_the_fit_layer_rejects_reaches_the_user_as_a_message(tmp_path, monkeypatch):
    """ผู้เรียกใน CLI ที่ไม่ได้มาทาง option (`_compute_fits` ถูกเรียกจาก rebuild และใบ stacked ด้วย) — ValueError ของ
    analyze() ต้องกลายเป็นข้อความ + exit 1 ที่จุดเดียว ไม่ใช่ traceback"""
    import typer

    from lmds.cli import main as cli_main

    with pytest.raises(typer.Exit) as stopped:
        cli_main._compute_fits(safetensors_report(weight_bytes=20 * GIB), ["dgx-spark-single"], 0)
    assert stopped.value.exit_code == 1


def test_inspect_rejects_an_unknown_kv_dtype_and_accepts_any_case(monkeypatch):
    _patch_inspect(monkeypatch, safetensors_report(weight_bytes=20 * GIB))
    base = ["inspect", "Qwen/Qwen3-32B", "--target", "dgx-spark-single"]
    for extra in (["--kv-dtype", "bogus"], ["--kv-dtype", "bogus", "--json"]):
        result = _run(base + extra)
        _clean_refusal(result, "--kv-dtype", "bf16", "fp8")
        assert result.stdout.strip() == ""
    upper = _run(base + ["--kv-dtype", "FP8", "--json"])
    assert upper.exit_code == 0, upper.output
    assert {entry["kv_dtype"] for entry in json.loads(upper.stdout)["context_advice"]} == {"fp8"}


@pytest.mark.parametrize("value", ["0", "-5"])
def test_inspect_rejects_a_context_that_is_not_positive(monkeypatch, value):
    """เดิม `--context -5` ตอบ exit 0 พร้อมคำแนะนำของค่าติดลบ · `--context 0` ถูกอ่านเป็น "ไม่ได้ถาม" เงียบ ๆ"""
    _patch_inspect(monkeypatch, safetensors_report(weight_bytes=20 * GIB))
    result = _run(["inspect", "Qwen/Qwen3-32B", "--target", "dgx-spark-single", "--context", value])
    _clean_refusal(result, "--context", "ตั้งแต่ 1 ขึ้นไป")


def test_deploy_with_an_unknown_task_exits_1_not_the_gates_code(tmp_path, monkeypatch):
    """exit 2 ของ deploy แปลว่า "ไม่ผ่าน quality gates" (docstring + docs/CLI_SPEC.md) — `--task bogus` เคยออก 2"""
    import yaml

    called = []
    _patch_inspect(monkeypatch, lambda s, c: called.append(s) or safetensors_report(weight_bytes=20 * GIB))
    result = _run(_argv("deploy", tmp_path, "--task", "bogus"))
    _clean_refusal(result, "--task", "generate", "embed", "rerank")
    assert called == [], "ค่าที่ตรวจได้จาก argv ต้องไม่รอ inspect ก่อน"
    ok = _run(_argv("deploy", tmp_path, "--task", "Embed"))
    assert ok.exit_code == 0, ok.output
    profile = yaml.safe_load((tmp_path / "bundles" / "qwen3-32b" / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["model"]["task"] == "embed"


@pytest.mark.parametrize("argv", [["fit", "nemotron", "--slots", "0"], ["fit", "nemotron", "--context", "-5"],
                                  ["fit", "nemotron", "--slots", "-1", "--json"]])
def test_fit_rejects_slots_and_context_that_are_not_positive(spark_head, argv):  # noqa: F811
    """เดิม `fit --slots 0` ถูกอ่านเป็น "ไม่ได้ระบุ" แล้วตอบตารางของ 4 slots · `--context -5` ตอบ "จะเขียน: context=-5" exit 0"""
    result = _run(argv)
    _clean_refusal(result, argv[2], "ตั้งแต่ 1 ขึ้นไป")
    assert result.stdout.strip() == ""


def test_set_fit_refused_with_a_bad_port_is_an_error_message_not_a_traceback(spark_head):  # noqa: F811
    """`set <slug> --fit --slots 40 --port 99999999`: fit ปฏิเสธ → ทาง "เขียนเฉพาะค่าที่ไม่เกี่ยวกับหน่วยความจำ" เรียก
    write() นอก try → SettingsError หลุดเป็น traceback · ต้องบอกทั้งสองเรื่อง (port ผิด + fit ไม่พอ) แล้ว exit 1"""
    from lmds.fleet.bundle_settings import read

    before = read(spark_head)
    result = _run(["set", "nemotron", "--fit", "--slots", "40", "--port", "99999999"])
    _clean_refusal(result, "port ต้องเป็นเลข 1-65535", "ไม่ได้ตั้ง slots/context/KV ให้")
    assert read(spark_head) == before, "ค่าผิดต้องไม่ถูกเขียน และค่าที่มีอยู่ต้องไม่หาย"

    as_json = _run(["set", "nemotron", "--fit", "--slots", "40", "--port", "99999999", "--json"])
    _no_crash(as_json)
    assert as_json.exit_code == 1
    payload = json.loads(as_json.stdout)
    assert payload["written"] == {} and "1-65535" in payload["not_written"] and payload["error"]


@pytest.mark.parametrize("value", ["0", "99999999", "-1"])
def test_web_rejects_a_port_outside_the_tcp_range_before_starting_anything(monkeypatch, value):
    """พอร์ตนอกช่วงไปถึง uvicorn แล้วตายด้วย OverflowError — ตรวจที่ option ก่อนแตะ daemon/socket ใด ๆ

    ยามสองชั้นกันเทสนี้เปิดเซิร์ฟเวอร์จริงบนเครื่องที่รัน: ถ้า option ไม่ปฏิเสธ `port_busy` ปลอมตอบว่าไม่ว่าง → exit 1 อยู่ดี
    (แต่ข้อความไม่ตรง เทสจึงล้ม) ไม่ไปถึง serve()
    """
    touched = []
    monkeypatch.setattr("lmds.web.daemon.running", lambda *a, **k: touched.append("state") or None)
    monkeypatch.setattr("lmds.web.daemon.port_busy", lambda *a, **k: touched.append("socket") or True)
    result = _run(["web", "--port", value])
    _clean_refusal(result, "--port", "1 ถึง 65,535")
    assert touched == []
