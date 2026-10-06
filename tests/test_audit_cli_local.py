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
