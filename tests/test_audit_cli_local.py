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


# ═════════════════════ 2. `deploy --name` ต้องไม่ทับ bundle ของ repo อื่น ═════════════════════
def _profile(directory) -> dict:
    import yaml

    return yaml.safe_load((directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))


def _snapshot(directory) -> dict:
    """ทุกไฟล์ในโฟลเดอร์ bundle → เนื้อหา — "ไม่ถูกแตะ" ต้องพิสูจน์ด้วยของบนดิสก์ ไม่ใช่ด้วยการไม่มี error"""
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def _deploy(tmp_path, monkeypatch, report, *extra: str, output=None):
    _patch_inspect(monkeypatch, report)
    target = [] if "--target" in extra else ["--target", "dgx-spark-single"]
    return _run(["deploy", report.repo_id, "--no-llm", *target, "--yes",
                 "--output", str(output or tmp_path / "bundles"), *extra])


def test_deploy_name_refuses_a_folder_that_belongs_to_another_repo(tmp_path, monkeypatch):
    """เคส audit: `deploy A --name chat` แล้ว `deploy B --name chat` → exit 0 ไม่มีคำเตือน · `chat` พลิกจาก Qwen/Qwen3-32B เป็น
    OtherOrg/Totally-Different-70B โดย bundle.env · API key · server.meta ของตัวเดิมตกไปเป็นของโมเดลใหม่"""
    from lmds.fleet import apikey
    from lmds.fleet.bundle_settings import write as write_settings

    first = safetensors_report(weight_bytes=20 * GIB)
    other = safetensors_report(repo_id="OtherOrg/Totally-Different-70B", weight_bytes=30 * GIB)
    made = _deploy(tmp_path, monkeypatch, first, "--name", "chat")
    assert made.exit_code == 0, made.output
    chat = tmp_path / "bundles" / "chat"
    write_settings(chat, {"port": 8123})
    before, key_before = _snapshot(chat), apikey.read("chat")
    assert key_before

    result = _deploy(tmp_path, monkeypatch, other, "--name", "chat")
    _clean_refusal(result, "chat", "Qwen/Qwen3-32B", "lmds remove chat")
    assert _snapshot(chat) == before, "โฟลเดอร์ของโมเดลเดิมต้องไม่ถูกแตะแม้แต่ไฟล์เดียว"
    assert _profile(chat)["model"]["id"] == "Qwen/Qwen3-32B"
    assert apikey.read("chat") == key_before
    assert "DeploymentPlan" not in _flat(result.stdout), "ต้องปฏิเสธก่อนวางแผน/ถามยืนยัน ไม่ใช่หลังจากนั้น"
    assert sorted(p.name for p in (tmp_path / "bundles").iterdir()) == ["chat", "chat.zip"]


def test_deploying_the_same_repo_under_the_same_name_still_works(tmp_path, monkeypatch):
    """ยามต้องไม่กันทางปกติ: deploy ซ้ำ (ปรับ target/engine) ของ repo เดิมในชื่อเดิม = เขียนทับที่เดิมได้ · ไม่สนตัวพิมพ์ของ repo id"""
    first = safetensors_report(weight_bytes=20 * GIB)
    assert _deploy(tmp_path, monkeypatch, first, "--name", "chat").exit_code == 0
    again = _deploy(tmp_path, monkeypatch, safetensors_report(repo_id="qwen/qwen3-32b", weight_bytes=20 * GIB),
                    "--name", "chat")
    assert again.exit_code == 0, again.output
    assert sorted(p.name for p in (tmp_path / "bundles").iterdir() if p.is_dir()) == ["chat"]


def test_deploy_name_refuses_a_slug_this_machine_already_runs_for_another_repo(tmp_path, monkeypatch):
    """ชื่อเดียวกันแต่คนละ --output: โฟลเดอร์ปลายทางว่าง แต่ทะเบียนของเครื่อง (~/.lmds/run/<slug>) กับ API key ผูกกับชื่อ
    ไม่ใช่กับโฟลเดอร์ — bundle ใหม่จะได้ server.meta + key ของโมเดลเดิม และ `lmds start chat` ยังไปเปิดตัวเดิม"""
    first = safetensors_report(weight_bytes=20 * GIB)
    other = safetensors_report(repo_id="OtherOrg/Totally-Different-70B", weight_bytes=30 * GIB)
    assert _deploy(tmp_path, monkeypatch, first, "--name", "chat").exit_code == 0
    result = _deploy(tmp_path, monkeypatch, other, "--name", "chat", output=tmp_path / "elsewhere")
    _clean_refusal(result, "chat", "Qwen/Qwen3-32B", "lmds remove chat")
    assert not (tmp_path / "elsewhere").exists()


def test_deploy_name_with_a_bad_slug_is_refused_before_planning(tmp_path, monkeypatch):
    """`--name ../evil` เคยถูกจับตอน render — หลังวางแผน (และหลังถามยืนยันในโหมด interactive) ไปแล้ว"""
    result = _deploy(tmp_path, monkeypatch, safetensors_report(weight_bytes=20 * GIB), "--name", "../evil")
    _clean_refusal(result, "ชื่อ bundle ไม่ถูกต้อง")
    assert "DeploymentPlan" not in _flat(result.stdout)
    assert not (tmp_path / "bundles").exists() and not (tmp_path / "evil").exists()


def test_the_renderer_itself_refuses_a_named_folder_owned_by_another_repo(tmp_path):
    """ยามตัวจริงอยู่ที่ render_bundle — ทุกทางที่ส่ง slug เอง (deploy --name · ใบ stacked · rebuild · bundles refresh) ผ่านตรงนี้
    เดิม slug ที่ส่งมาเองข้าม `resolve_slug()` ซึ่งเป็นตัวกันชนข้าม repo ตัวเดียวที่มี"""
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.generator import render_bundle

    def render(report, slug):
        fit = analyze(report, PRESETS["dgx-spark-single"])
        return render_bundle(build_plan(report, fit, None), report, fit, tmp_path, slug=slug)

    owner = render(safetensors_report(weight_bytes=20 * GIB), "chat")
    before = _snapshot(owner.directory)
    with pytest.raises(ValueError, match="Qwen/Qwen3-32B"):
        render(safetensors_report(repo_id="OtherOrg/Totally-Different-70B", weight_bytes=30 * GIB), "chat")
    assert _snapshot(owner.directory) == before
    assert render(safetensors_report(weight_bytes=20 * GIB), "chat").directory == owner.directory   # เจ้าของเดิมเขียนทับได้


# ═════════════════════ 1. `rebuild` สร้าง *ใบที่สั่ง* ใหม่ ที่เดิม ด้วยค่าที่ profile จดไว้ ═════════════════════
STALE = "#!/bin/bash\n# stale — controller ของรุ่นเก่า\n"


def _bundle_state(directory) -> dict:
    """สิ่งที่ผู้ใช้ถือว่าเป็น "bundle ใบนี้": ทุกไฟล์ในโฟลเดอร์ ยกเว้นของที่เปลี่ยนทุกครั้งที่ generate โดยนิยาม

    (ตราเวลา origin.stamped_at ใน profile · checksum/zip ที่ครอบ profile นั้น) — controller · README · bundle.env ·
    score template · SPECIAL_FILES ต้องเหมือนเดิมทุกไบต์ และ profile ต้องเหมือนเดิมทุกคีย์
    """
    import yaml

    state = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "PACKAGE_SHA256SUMS" or path.suffix == ".zip":
            continue
        if path.name == "MODEL_PROFILE.yaml":
            profile = yaml.safe_load(path.read_text(encoding="utf-8"))
            (profile.get("origin") or {}).pop("stamped_at", None)
            state[path.name] = profile
        else:
            state[path.name] = path.read_text(encoding="utf-8")
    return state


def _all_bundles(root) -> dict:
    return {d.name: _bundle_state(d) for d in sorted(root.iterdir()) if d.is_dir()}


def _controller(directory):
    found = [p for p in directory.iterdir() if p.name.endswith(("-single.sh", "-stacked.sh"))]
    assert len(found) == 1, sorted(p.name for p in directory.iterdir())
    return found[0]


def _st(**overrides) -> ModelReport:
    return safetensors_report(weight_bytes=20 * GIB, **overrides)


def _rerank_report() -> ModelReport:
    return ModelReport(
        repo_id="Qwen/Qwen3-Reranker-4B", revision_sha="sha", task="rerank",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(8 * GIB),
        architecture="Qwen3ForCausalLM", context_length=40960,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
    )


# (ชื่อเคส, report ตอน deploy, option ของ deploy, slug ที่จะ rebuild, report ที่ inspector คืนตอน rebuild)
REBUILD_CASES = [
    ("vllm", _st, [], "qwen3-32b", _st),
    # เคส audit: `rebuild` ของ bundle --engine sglang → engine กลายเป็น vllm
    ("sglang", _st, ["--engine", "sglang"], "qwen3-32b", _st),
    # เคส audit: `rebuild prod-chat` (deploy --name) → งอก bundles/qwen3-32b ใหม่ · ของ prod-chat ไม่ถูกเขียน
    ("name", _st, ["--name", "prod-chat"], "prod-chat", _st),
    # เคส audit: `rebuild qwen3-32b-stacked` → เขียนทับใบ single `qwen3-32b` เป็น stacked · ใบที่สั่งไม่ถูกแตะ
    ("stacked-companion", _st, ["--also-stacked"], "qwen3-32b-stacked", _st),
    ("single-next-to-its-companion", _st, ["--also-stacked"], "qwen3-32b", _st),
    # เคส audit: bundle --task embed → task กลับเป็น generate (`--runner pooling --convert embed` หาย เสิร์ฟ chat แทน)
    # inspector รอบ rebuild ยังเดาว่า generate เหมือนตอน deploy — สิ่งที่รู้ว่าเป็น embed มีแต่ profile
    ("task-embed", _st, ["--task", "embed"], "qwen3-32b", _st),
    ("embed", _embed_report, [], "qwen3-embedding-4b", _embed_report),
    ("rerank", _rerank_report, [], "qwen3-reranker-4b", _rerank_report),
    ("stacked-target", _st, ["--target", "dgx-spark-stacked"], "qwen3-32b", _st),
    ("concurrency-8", _st, ["--concurrency", "8"], "qwen3-32b", _st),
    ("gguf", "gguf", [], "qwen3-8b-gguf", "gguf"),
    # llama.cpp: context = ก้อนรวมของทุก slot — fit ของ rebuild ต้องคิดที่จำนวน slot เดิม ไม่งั้น context ถูกบีบเหลือ 1/4
    ("gguf-4-slots", "gguf", ["--concurrency", "4"], "qwen3-8b-gguf", "gguf"),
    ("rtx", lambda: safetensors_report(weight_bytes=10 * GIB), ["--target", "rtx-5090"], "qwen3-32b",
     lambda: safetensors_report(weight_bytes=10 * GIB)),
]


def _make(report):
    from tests.test_generator import gguf_report

    return gguf_report() if report == "gguf" else report()


@pytest.mark.parametrize("case", REBUILD_CASES, ids=[c[0] for c in REBUILD_CASES])
def test_rebuild_regenerates_exactly_the_named_bundle_in_place(tmp_path, monkeypatch, case):
    """rebuild ของ bundle ที่เพิ่ง deploy ด้วย lmds รุ่นเดียวกัน = ได้ bundle เดิมกลับมาทุกไฟล์ และไม่มีใบไหนอื่นถูกแตะ

    ทำ controller ของใบที่สั่งให้เก่าก่อน (เขียนขยะทับ) แล้ว rebuild: ถ้ามัน regenerate ใบอื่น/ที่อื่น ขยะจะยังอยู่ ·
    ถ้ามันไม่พกค่าของ profile (engine · task · topology · slots · ชื่อ) controller ที่ได้จะไม่ตรงของเดิม
    """
    from lmds.fleet.bundle_settings import write as write_settings

    _, at_deploy, options, slug, at_rebuild = case
    root = tmp_path / "bundles"
    made = _deploy(tmp_path, monkeypatch, _make(at_deploy), *options)
    assert made.exit_code == 0, made.output
    write_settings(root / slug, {"port": 8123, "served_name": "my-alias"})   # `lmds set --port/--model-id` ของผู้ใช้
    before = _all_bundles(root)
    assert slug in before
    controller = _controller(root / slug)
    controller.write_text(STALE, encoding="utf-8")

    _patch_inspect(monkeypatch, lambda s, c: _make(at_rebuild))
    result = _run(["rebuild", slug])
    assert result.exit_code == 0, result.output
    after = _all_bundles(root)
    assert sorted(after) == sorted(before), "rebuild ต้องไม่สร้างหรือลบโฟลเดอร์ bundle ใด"
    for name in before:
        assert after[name].keys() == before[name].keys(), (name, sorted(after[name]), sorted(before[name]))
        for filename in before[name]:
            assert after[name][filename] == before[name][filename], f"{name}/{filename} ไม่ตรงของเดิมหลัง rebuild {slug}"
    assert f"lmdsnodepush<เครื่อง>{slug}" in _flat(result.stdout)


def test_rebuild_keeps_the_flag_and_runtime_file_the_user_approved(tmp_path, monkeypatch):
    """flag นอก allowlist กับไฟล์ runtime ภายนอกเข้า bundle ได้ทางเดียวคือผู้ใช้อนุมัติตอน deploy — harden ของ rebuild ย้ายทั้งคู่
    กลับไป "รออนุมัติ" แล้ว render โดยไม่มีมัน: โมเดลที่ต้อง --trust-remote-code หรือ parser plugin start ไม่ขึ้นหลัง rebuild"""
    from lmds.brain import RuntimeAsset, apply_asset_approvals, apply_flag_approvals, build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.fleet import register_bundle
    from lmds.generator import render_bundle

    report = _st()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, None)
    plan.flags_needing_approval.append("--trust-remote-code")          # ทางเดียวกับขั้นยืนยันของ deploy
    apply_flag_approvals(plan, ["--trust-remote-code"])
    plan.assets_needing_approval.append(RuntimeAsset(
        filename="super_parser.py", url="https://raw.githubusercontent.com/acme/parsers/main/super_parser.py",
        purpose="reasoning parser"))
    apply_asset_approvals(plan, ["super_parser.py"])
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles")
    register_bundle(bundle.controller)
    original = bundle.controller.read_text(encoding="utf-8")
    assert "--trust-remote-code" in original and "super_parser.py" in original   # ของตั้งต้นมีจริง ไม่ใช่เทียบว่างกับว่าง
    # ไฟล์ runtime ที่อนุมัติไว้ถูกจดอยู่ในตาราง ASSET_* ของ controller ที่เดียว (profile ไม่ได้จด) — ทำให้เก่าด้วยการต่อท้าย
    # ไม่ใช่เขียนทับทั้งไฟล์ ไม่งั้นคือลบหลักฐานการอนุมัติทิ้งเองก่อน rebuild
    bundle.controller.write_text(original + "\n# stale — แก้มือค้างไว้\n", encoding="utf-8")

    _patch_inspect(monkeypatch, lambda s, c: _st())
    result = _run(["rebuild", "qwen3-32b"])
    assert result.exit_code == 0, result.output
    assert bundle.controller.read_text(encoding="utf-8") == original
    profile = _profile(bundle.directory)
    assert "--trust-remote-code" in profile["serving"]["extra_flags"] and profile["flags_needing_approval"] == []


def test_rebuild_does_not_drop_a_parser_the_current_rules_know_nothing_about(tmp_path, monkeypatch):
    """parser ที่ LLM หาให้ตอน deploy (โมเดลที่กฎตระกูล/สูตรไม่รู้จัก): rebuild วางแผนใหม่แบบ rule-based — ไม่พกของเดิมมา
    tool calling ที่เคยใช้ได้ดับเงียบ ๆ"""
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.fleet import register_bundle
    from lmds.generator import render_bundle

    def report():
        return _st(repo_id="Acme/Wombat-7B")

    fit = analyze(report(), PRESETS["dgx-spark-single"])
    plan = build_plan(report(), fit, None)
    assert plan.tool_calling.parser is None, "เคสนี้ต้องเป็นโมเดลที่ตรรกะปัจจุบันไม่มี parser ให้"
    plan.tool_calling.enabled, plan.tool_calling.parser = True, "hermes"
    plan.reasoning.enabled, plan.reasoning.parser = True, "deepseek_r1"
    bundle = render_bundle(plan, report(), fit, tmp_path / "bundles")
    register_bundle(bundle.controller)
    original = bundle.controller.read_text(encoding="utf-8")
    bundle.controller.write_text(STALE, encoding="utf-8")

    _patch_inspect(monkeypatch, lambda s, c: report())
    result = _run(["rebuild", "wombat-7b"])
    assert result.exit_code == 0, result.output
    assert bundle.controller.read_text(encoding="utf-8") == original
    features = _profile(bundle.directory)["features"]
    assert (features["tool_calling"]["parser"], features["reasoning"]["parser"]) == ("hermes", "deepseek_r1")


def _edit_profile(directory, edit) -> None:
    import yaml

    path = directory / "MODEL_PROFILE.yaml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(profile)
    path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _refused_without_writing(tmp_path, result, slug: str, before: dict, *needles: str) -> None:
    _clean_refusal(result, *needles)
    assert _snapshot(tmp_path / "bundles" / slug) == before, "ปฏิเสธแล้วต้องไม่มีไฟล์ไหนใน bundle ถูกเขียน"
    assert sorted(p.name for p in (tmp_path / "bundles").iterdir() if p.is_dir()) == [slug]


def test_rebuild_refuses_when_the_profile_does_not_say_which_engine(tmp_path, monkeypatch):
    """safetensors เสิร์ฟได้ทั้ง vLLM และ SGLang — profile ที่ไม่ได้จด engine ไม่มีทาง rebuild ให้ตรงของเดิม: ปฏิเสธ ไม่เดา"""
    assert _deploy(tmp_path, monkeypatch, _st(), "--engine", "sglang").exit_code == 0
    bundle = tmp_path / "bundles" / "qwen3-32b"
    _edit_profile(bundle, lambda p: p["runtime"].pop("engine"))
    before = _snapshot(bundle)
    _refused_without_writing(tmp_path, _run(["rebuild", "qwen3-32b"]), "qwen3-32b", before, "runtime.engine", "ไม่มีไฟล์ไหนถูกเขียน")


def test_rebuild_refuses_when_the_fallback_target_would_change_the_topology(tmp_path, monkeypatch):
    """target เดิมไม่ใช่ preset แล้ว → ถอยไป auto-detect ได้ (ของเดิม) แต่ถ้าผลคือ single ทั้งที่ bundle เป็น stacked
    นั่นคือ bundle อีกใบ — controller คนละตัว คนละจำนวนเครื่อง"""
    made = _deploy(tmp_path, monkeypatch, _st(), "--target", "dgx-spark-stacked")
    assert made.exit_code == 0, made.output
    bundle = tmp_path / "bundles" / "qwen3-32b"
    _edit_profile(bundle, lambda p: p["target"].update(name="dgx-spark-cluster-of-old"))
    before = _snapshot(bundle)
    _refused_without_writing(tmp_path, _run(["rebuild", "qwen3-32b"]), "qwen3-32b", before, "stacked", "single")


def test_rebuild_refuses_when_the_result_would_be_a_different_engine(tmp_path, monkeypatch):
    """profile บอก vLLM แต่ repo ที่ inspect ได้ตอนนี้เป็น GGUF (→ llama.cpp เสมอ): harden แก้ engine ให้ตามข้อเท็จจริง
    ซึ่งถูกสำหรับ deploy ใหม่ — สำหรับ rebuild แปลว่ากำลังจะเขียน bundle llama.cpp ทับ bundle vLLM"""
    from tests.test_generator import gguf_report

    assert _deploy(tmp_path, monkeypatch, _st()).exit_code == 0
    bundle = tmp_path / "bundles" / "qwen3-32b"
    before = _snapshot(bundle)
    _patch_inspect(monkeypatch, lambda s, c: gguf_report(repo_id="Qwen/Qwen3-32B"))
    _refused_without_writing(tmp_path, _run(["rebuild", "qwen3-32b"]), "qwen3-32b", before, "engine: vllm → llamacpp")


def test_rebuild_of_an_impossible_recorded_plan_is_a_message_not_a_traceback(tmp_path, monkeypatch):
    """ขา rebuild ของข้อ 4: profile (แก้มือ/รุ่นเก่า) ที่จด SGLang บน stacked — planner ปฏิเสธด้วย PlanError ซึ่ง rebuild
    ไม่มียามเลย จึงหลุดเป็น traceback"""
    made = _deploy(tmp_path, monkeypatch, _st(), "--target", "dgx-spark-stacked")
    assert made.exit_code == 0, made.output
    bundle = tmp_path / "bundles" / "qwen3-32b"
    _edit_profile(bundle, lambda p: p["runtime"].update(engine="sglang"))
    before = _snapshot(bundle)
    _refused_without_writing(tmp_path, _run(["rebuild", "qwen3-32b"]), "qwen3-32b", before,
                             "SGLang ยังไม่มี controller แบบ stacked")


def test_rebuild_leaves_an_adopted_bundle_alone(tmp_path, monkeypatch):
    """bundle จาก `lmds adopt` (container ที่ลูกค้าตั้งเองก่อน LMDS เข้าไป) ไม่มี template — rebuild เคย inspect model.id จาก HF
    แล้ววาง <slug>-single.sh ของ LMDS ลงข้าง -adopted.sh พร้อมเขียนทับ profile (ที่อยู่ weight / container ต้นทางหาย)"""
    import yaml

    from lmds.fleet import run_root

    bundle = tmp_path / "bundles" / "coder-next"
    bundle.mkdir(parents=True)
    controller = bundle / "coder-next-adopted.sh"
    controller.write_text("#!/bin/bash\necho adopted \"$1\"\n", encoding="utf-8")
    controller.chmod(0o755)
    (bundle / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "profile_version": 1, "generated_by": "lmds adopt", "adopted": True,
        "model": {"id": "Qwen/Qwen3-32B", "artifact_type": "unknown"},
        "runtime": {"engine": "vllm", "image": "vllm/vllm-openai:latest"},
        "serving": {"context": 32768, "port": 8000}, "source_container": "coder-next",
        "weights": {"path": "/data/models/coder-next"},
    }), encoding="utf-8")
    run_dir = run_root() / "coder-next"
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        f"slug=coder-next\nmodel=coder-next\nmodel_id=Qwen/Qwen3-32B\nengine=vllm\nmode=docker\nport=8000\n"
        f"container=coder-next\ncontroller={controller}\nstarted_at=\n", encoding="utf-8")
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    inspected = []
    _patch_inspect(monkeypatch, lambda s, c: inspected.append(s) or _st())
    before = _snapshot(bundle)
    result = _run(["rebuild", "coder-next"])
    _clean_refusal(result, "lmds adopt")
    assert _snapshot(bundle) == before and inspected == []
