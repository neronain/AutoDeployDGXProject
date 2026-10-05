"""context ต่อคำขอ · จำนวน slot · ก้อนรวม (pool) — สามตัวเลขที่ต้องไม่ถูกเอามาปนกัน

`serving.context` ของ LMDS คือค่าที่ส่งเข้า engine และแต่ละ engine ไม่ได้หมายถึงตัวเดียวกัน:

    llama.cpp   --ctx-size       ก้อนรวม หารเท่ากันด้วย --parallel → คำขอเดียวได้ context ÷ slots
    vLLM        --max-model-len  เพดานต่อคำขอ (ก้อนรวมเป็นงบหน่วยความจำแยก: --kv-cache-memory)
    SGLang      --context-length เพดานต่อคำขอ

เพดานของโมเดล (native / max_position_embeddings) เป็นเพดาน **ต่อคำขอ** เสมอ

ตรวจ 2026-10-05: planner รู้เรื่องนี้ (ตั้งก้อนรวม = ต่อคำขอ × slot ให้ llama.cpp) แต่ fit ·
set · controller · prompt ของ deploy และป้ายบนจอ เอาก้อนรวมไปเทียบ/แสดงเหมือนเป็นค่าต่อคำขอ
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.cli.main import app, context_prompt_ceiling
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.fit.sizing import plan_kv_pin, settings_for
from lmds.fleet.bundle_settings import SettingsError, read, write
from lmds.generator import render_bundle
from lmds.generator.renderer import _client_input
from tests.test_review_templates import (
    _bundle,
    _gguf_report,
    _safetensors_report,
    _validate_numbers_script,
    run_bash,
)

SPARK = {"memory_model": "unified", "total_gb": 121.0, "free_gb": 110.0, "held_gb": 0.0, "others": []}


def _llamacpp_info(**over) -> dict:
    base = {"slug": "gguf", "engine": "llamacpp", "weight_bytes": 20 * GIB,
            "kv_bytes_per_token": 40960, "native_context": 131072, "running": False}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# lmds fit / set --fit
# ---------------------------------------------------------------------------
def test_fit_keeps_a_llamacpp_pool_of_native_times_slots():
    """GGUF native 131,072 deploy ด้วย --concurrency 4 → ก้อนรวม 524,288 คือค่าที่ถูก

    เดิม `lmds fit` ปัดก้อนรวมเหลือ 131,072 · รายงาน KV ต่ำไป 4 เท่า · แล้ว `set --fit`
    เขียนค่านั้นลง bundle — คำขอเดียวหล่นจาก 131,072 เหลือ 32,768
    """
    p = plan_kv_pin(_llamacpp_info(slots=4, context=524288), SPARK)
    assert p["context"] == 524288, p["notes"]
    assert p["per_slot_context"] == 131072
    assert p["kv_gb"] == 20.0                      # 524,288 × 40,960 B = 20 GiB ไม่ใช่ 5
    assert not any("เกิน native" in note for note in p["notes"])
    assert settings_for(p, "") == {"slots": "4", "context": "524288"}


def test_fit_clamps_a_llamacpp_pool_only_when_one_request_would_exceed_native():
    p = plan_kv_pin(_llamacpp_info(slots=4, context=1048576), SPARK)
    assert p["context"] == 524288 and p["per_slot_context"] == 131072
    assert any("262,144 ต่อคำขอ" in note and "524,288" in note for note in p["notes"]), p["notes"]


def test_fit_still_clamps_a_single_slot_llamacpp_and_vllm_to_native():
    one = plan_kv_pin(_llamacpp_info(slots=1, context=524288), SPARK)
    assert one["context"] == 131072
    vllm = plan_kv_pin({"slug": "v", "engine": "vllm", "weight_bytes": 20 * GIB,
                        "kv_bytes_per_token": 4096, "native_context": 131072, "running": False},
                       SPARK, slots=4, context=524288)
    assert vllm["context"] == 131072, "vLLM: context เป็นค่าต่อคำขอ จำนวน slot ไม่เกี่ยว"


# ---------------------------------------------------------------------------
# lmds set --context
# ---------------------------------------------------------------------------
def _llamacpp_bundle_dir(tmp_path: Path, *, native: int, slots: int) -> Path:
    (tmp_path / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({
        "model": {"native_context": native},
        "runtime": {"engine": "llamacpp"},
        "serving": {"max_num_seqs": slots},
    }), encoding="utf-8")
    return tmp_path


def test_set_accepts_a_llamacpp_pool_that_gives_each_slot_the_native_context(tmp_path):
    """เดิมปฏิเสธว่า "เกินเพดานที่โมเดลเทรนมา" ทั้งที่ planner ตั้งค่านี้ให้เอง"""
    bundle = _llamacpp_bundle_dir(tmp_path, native=131072, slots=4)
    write(bundle, {"context": 524288})
    assert read(bundle)["context"] == "524288"


def test_set_uses_the_slot_count_sent_in_the_same_call(tmp_path):
    bundle = _llamacpp_bundle_dir(tmp_path, native=131072, slots=1)
    write(bundle, {"context": 262144, "slots": 2})
    assert read(bundle)["context"] == "262144" and read(bundle)["slots"] == "2"


def test_set_refuses_a_llamacpp_pool_whose_per_request_share_exceeds_native(tmp_path):
    bundle = _llamacpp_bundle_dir(tmp_path, native=131072, slots=2)
    with pytest.raises(SettingsError) as refused:
        write(bundle, {"context": 524288})
    message = str(refused.value)
    assert "262,144 ต่อคำขอ" in message, "ต้องบอกว่าคำขอเดียวจะได้เท่าไร"
    assert "262,144 (131,072 × 2 slot)" in message, "ต้องบอกค่าสูงสุดที่ตั้งได้ในหน่วยเดียวกับที่กรอก"


# ---------------------------------------------------------------------------
# controller: เตือน RoPE เฉพาะเมื่อ "ต่อ slot" เกิน
# ---------------------------------------------------------------------------
def _controller_text(tmp_path: Path) -> str:
    bundle = _bundle(tmp_path, _gguf_report(context_length=131072))
    return next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")


def test_controller_does_not_warn_when_each_slot_gets_the_native_context(tmp_path):
    """4 slot × 131,072 เคยเตือน "เกินที่โมเดลเทรนมา" ทุกครั้งที่ start"""
    env = dict(API_PORT=8000, NATIVE_CONTEXT=131072, CLIENT_OUTPUT=4096, PARALLEL_SEQS=4)
    done = run_bash(_validate_numbers_script(_controller_text(tmp_path), CTX_SIZE=524288, **env))
    assert "VALID" in done.stdout and "WARN" not in done.stderr, done.stderr


def test_controller_warns_with_the_per_request_number_when_a_slot_exceeds_native(tmp_path):
    env = dict(API_PORT=8000, NATIVE_CONTEXT=131072, CLIENT_OUTPUT=4096, PARALLEL_SEQS=4)
    done = run_bash(_validate_numbers_script(_controller_text(tmp_path), CTX_SIZE=1048576, **env))
    assert "VALID" in done.stdout
    assert "WARN" in done.stderr and "262144" in done.stderr and "4 slot" in done.stderr


def test_controller_survives_a_non_numeric_slot_count(tmp_path):
    env = dict(API_PORT=8000, NATIVE_CONTEXT=131072, CLIENT_OUTPUT=4096, PARALLEL_SEQS="")
    done = run_bash(_validate_numbers_script(_controller_text(tmp_path), CTX_SIZE=131072, **env))
    assert "VALID" in done.stdout, done.stderr


# ---------------------------------------------------------------------------
# lmds deploy (เทอร์มินัล): เพดานของช่อง context
# ---------------------------------------------------------------------------
def test_pressing_enter_at_the_deploy_prompt_keeps_a_llamacpp_pool():
    """`lmds deploy <gguf> --concurrency 4` → แผนเสนอก้อนรวม = ต่อคำขอ × 4

    เดิมเพดานของ prompt เป็นค่าต่อ sequence จึงปัดค่าตามแผนลงเองว่า "เกินเพดานที่ปลอดภัย"
    """
    report = _gguf_report(context_length=131072)
    fit = analyze(report, PRESETS["dgx-spark-single"], concurrency=4)
    plan = build_plan(report, fit, provider=None)
    assert plan.runtime.engine is Engine.LLAMACPP and plan.serving.max_num_seqs == 4
    assert plan.serving.context == 4 * 131072
    assert context_prompt_ceiling(plan, fit) >= plan.serving.context


def test_the_deploy_prompt_ceiling_is_per_request_on_vllm():
    report = _safetensors_report(context_length=131072)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.VLLM)
    assert context_prompt_ceiling(plan, fit) == max(fit.max_safe_context or 0, plan.serving.context)


# ---------------------------------------------------------------------------
# planner: --concurrency N ต้องเสิร์ฟ N ได้จริงบน vLLM ด้วย
# ---------------------------------------------------------------------------
def test_vllm_plan_serves_as_many_sequences_as_the_concurrency_it_sized_for():
    """fit หาร context ด้วย concurrency ที่ขอ — max_num_seqs ที่ค้าง 4 คือได้ context หารแปด
    แต่เสิร์ฟพร้อมกันแค่สี่"""
    report = _safetensors_report(context_length=131072)
    fit = analyze(report, PRESETS["dgx-spark-single"], concurrency=8)
    plan = build_plan(report, fit, provider=None, engine=Engine.VLLM)
    assert plan.serving.max_num_seqs == 8
    default = build_plan(report, analyze(report, PRESETS["dgx-spark-single"]),
                         provider=None, engine=Engine.VLLM)
    assert default.serving.max_num_seqs == 4, "ค่าตั้งต้นเดิมต้องไม่เปลี่ยน"


# ---------------------------------------------------------------------------
# README: งบของ client เป็นของคำขอเดียว
# ---------------------------------------------------------------------------
def test_readme_client_budget_is_per_request_for_a_multi_slot_llamacpp_bundle():
    report = _gguf_report(context_length=131072)
    fit = analyze(report, PRESETS["dgx-spark-single"], concurrency=4)
    plan = build_plan(report, fit, provider=None)
    per_request = plan.serving.context // plan.serving.max_num_seqs
    assert _client_input(plan) == per_request - plan.serving.max_output_tokens - 2048
    assert _client_input(plan) < per_request, "ต้องไม่ใช่งบของก้อนรวม"


# ---------------------------------------------------------------------------
# ป้ายบนจอ: lmds list และ payload ของหน้าเว็บ ใช้ตัวเลขชุดเดียวกัน
# ---------------------------------------------------------------------------
def _registered_llamacpp_bundle(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    report = _gguf_report(context_length=262144)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles")
    controller = next(bundle.directory.glob("*-single.sh"))
    slug = bundle.directory.name
    run_root = tmp_path / "run"
    (run_root / slug).mkdir(parents=True)
    (run_root / slug / "server.meta").write_text(
        f"slug={slug}\nmodel={report.repo_id}\nmodel_id={report.repo_id}\n"
        f"engine=llamacpp\nmode=native\nport=8020\ncontainer=\n"
        f"pid_file=\ncontroller={controller}\nstarted_at=2026-10-05T00:00:00\n",
        encoding="utf-8")
    monkeypatch.setenv("LMDS_RUN_ROOT", str(run_root))
    return slug, controller


def test_list_shows_what_one_request_gets_from_the_saved_settings(tmp_path, monkeypatch):
    """เคสจริง 2026-10-05 spark-head · gemma: บันทึก context 65,536 · slots 2

    `lmds list` ขึ้นค่าที่แผนจดไว้ตอน deploy ขณะที่ /props ตอบ 32,768 ต่อคำขอ
    """
    slug, controller = _registered_llamacpp_bundle(tmp_path, monkeypatch)
    planned = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text())["serving"]["context"]
    write(controller.parent, {"context": 65536, "slots": 2})

    out = CliRunner().invoke(app, ["list"], env={"COLUMNS": "200"}).stdout
    row = next(line for line in out.splitlines() if slug[:12] in line)
    assert "32,768" in row and "×2" in row, row
    assert f"{planned:,}" not in row, "ค่าที่แผนจดไว้ไม่ใช่ค่าที่จะใช้"
    assert "llama.cpp แบ่ง --ctx-size" in out, "ต้องอธิบาย ×N ใต้ตาราง"


def test_the_web_payload_reports_the_same_three_numbers(tmp_path, monkeypatch):
    from lmds import inventory
    from lmds.fleet import discover

    slug, controller = _registered_llamacpp_bundle(tmp_path, monkeypatch)
    write(controller.parent, {"context": 65536, "slots": 2})
    server = next(s for s in discover() if s.slug == slug)
    payload = inventory.model_payload(server)
    assert (payload["context"], payload["slots"], payload["context_per_request"]) == (65536, 2, 32768)


def test_a_single_slot_bundle_is_listed_without_a_split(tmp_path, monkeypatch):
    slug, controller = _registered_llamacpp_bundle(tmp_path, monkeypatch)
    out = CliRunner().invoke(app, ["list"], env={"COLUMNS": "200"}).stdout
    row = next(line for line in out.splitlines() if slug[:12] in line)
    assert "×" not in row and "llama.cpp แบ่ง --ctx-size" not in out


# ---------------------------------------------------------------------------
# หน้าเว็บ: บูตหน้าจริงใน node แล้วเรียกฟังก์ชันที่วาดจอ — ผู้ใช้ทำงานผ่านหน้าเว็บ
# ---------------------------------------------------------------------------
def test_the_console_tags_a_split_llamacpp_context_with_what_one_request_gets(tmp_path):
    """การ์ดโมเดลของเครื่องนี้และตาราง Fleet models เคยโชว์ก้อนรวมเฉย ๆ (มีป้ายเฉพาะการ์ด node)"""
    from tests.test_console_shell import FLEET, run_scenario

    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        const split = { slug: "gemma", model_id: "org/gemma", engine: "llamacpp", context: 65536,
                        slots: 2, context_per_request: 32768, controller_exists: true, downloaded: true };
        const whole = { ...split, engine: "vllm", context: 262144, slots: 4, context_per_request: 262144 };
        console.log(JSON.stringify({
          split: ctxSplit(split), whole: ctxSplit(whole),
          tag: perRequestTag(split), none: perRequestTag(whole),
          card: rowMarkup(split), cardWhole: rowMarkup(whole) }));""")
    assert out["split"] is True and out["whole"] is False
    assert "32,768/request" in out["tag"] and "2 slots" in out["tag"]
    assert out["none"] == ""
    assert "65,536 tok" in out["card"] and "32,768/request" in out["card"]
    assert "/request" not in out["cardWhole"]
