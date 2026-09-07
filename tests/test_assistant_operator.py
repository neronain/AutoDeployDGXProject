"""ผู้ช่วยที่ *ทำงานแทน* ได้ — แคตตาล็อกชุดสอง, probe ที่คำนวณบน hub, ตั๋วหลายขั้น, สมองจากฟลีต

เจ้าของ 2026-09-07: "ช่วยเพิ่มความสามารถของ AI assistant ในระบบหน่อย … ไม่แน่ใจว่าจะมีแค่ไว้ถามตอบเอง"
สิ่งที่เทสชุดนี้ปกป้อง เรียงตามความเสียหายถ้าพลาด:

  1. **งานประกอบขยายเป็นขั้นจริงก่อนออกตั๋ว** — deploy_model ต้องกลายเป็นคำสั่งที่อ่านได้ทีละขั้น ไม่ใช่ชื่อรวม
  2. **งานลบถาวรไม่วิ่งจากปุ่มเดียว** — remove_model ตั้งต้นที่ "ยังไม่ทำ" และ "แก้เลย" ต้องยืนยันซ้ำ
  3. **งานของ hub รันบน hub** — node_install/push ถูกบังคับ target=this ต่อให้ LLM เลือกเครื่องปลายทาง
  4. **ขั้นที่ล้มมีสาเหตุจาก log แปะมา** — ผู้ใช้ไม่ต้องถามซ้ำว่า "แล้วทำไมล้ม"
  5. **แนะนำโมเดลจากที่มี weight จริง** — ไม่เดาชื่อโมเดลจากความจำ และ unknown ไม่นับเป็นผ่าน
"""

from __future__ import annotations

import pytest

from lmds.assistant import brain, catalog, insight, policy, router, runner


# ---------------------------------------------------------------------------
# พารามิเตอร์ชนิดใหม่: ยังคงแคบที่สุดที่ใช้งานได้จริง
# ---------------------------------------------------------------------------
def test_a_repo_param_accepts_org_name_and_strips_the_hf_url():
    p = catalog.Param("repo", "repo")
    assert p.clean("Qwen/Qwen3-8B-GGUF") == "Qwen/Qwen3-8B-GGUF"
    assert p.clean("https://huggingface.co/Qwen/Qwen3-8B-GGUF/") == "Qwen/Qwen3-8B-GGUF"
    for bad in ("Qwen3-8B", "a/b/c", "org/name; rm -rf", "org/$(id)"):
        with pytest.raises(catalog.ParamError):
            p.clean(bad)


def test_served_name_task_text_choice_and_bool_params():
    assert catalog.Param("name", "name").clean("gpt-4o") == "gpt-4o"
    assert catalog.Param("name", "name").clean("org/model:latest") == "org/model:latest"
    with pytest.raises(catalog.ParamError):
        catalog.Param("name", "name").clean("a b")
    assert catalog.Param("task", "text").clean("  coding   ภาษาไทย ") == "coding ภาษาไทย"
    with pytest.raises(catalog.ParamError):
        catalog.Param("task", "text").clean("x`id`")
    pick = catalog.Param("test", "choice", choices=("test-text", "score"))
    assert pick.clean("score") == "score"
    with pytest.raises(catalog.ParamError):
        pick.clean("rm")
    assert catalog.Param("k", "bool").clean("true") == "1"
    assert catalog.Param("k", "bool").clean("0") == "0"
    with pytest.raises(catalog.ParamError):
        catalog.Param("k", "bool").clean("maybe")


# ---------------------------------------------------------------------------
# probe ชุดสอง: คำสั่งประกอบจากโค้ด อ้างเครื่องมือเดิม
# ---------------------------------------------------------------------------
def test_fit_preview_is_a_dry_run_of_lmds_fit():
    command, clean = catalog.PROBES["fit_preview"].command({"slug": "nemo", "slots": "4", "context": "65536"})
    assert command == "lmds fit nemo --slots 4 --context 65536"
    assert clean == {"slug": "nemo", "slots": "4", "context": "65536"}
    assert catalog.PROBES["fit_preview"].command({"slug": "nemo"})[0] == "lmds fit nemo"


def test_last_failure_reads_both_native_log_and_docker_logs_and_never_fails():
    command, _ = catalog.PROBES["last_failure"].command({"slug": "qwen"})
    assert "~/.lmds/run/qwen/server.log" in command
    assert "docker logs --tail 600 lmds-qwen" in command
    assert "unknown model architecture" in command and "OutOfMemoryError" in command
    assert command.endswith("|| true"), "probe สำรวจต้องไม่รายงานว่าล้มเพราะ grep ไม่เจอ"


def test_the_survey_probes_of_the_second_batch_never_fail_either():
    for name in ("runtime_info", "bench_results", "usage", "weights_on_disk"):
        command, _ = catalog.PROBES[name].command({"slug": "x"} if name == "bench_results" else {})
        assert command.endswith("|| true"), name


def test_every_new_action_has_an_impact_and_reuses_existing_commands():
    expect = {
        "set_fit": "lmds set nemo --fit --slots 2",
        "set_slots": "lmds set nemo --slots 2",
        "set_served_name": "lmds set nemo --model-id gpt-4o",
        "regenerate_controller": "lmds bundles refresh nemo",
        "bundles_refresh": "lmds bundles refresh --all --if-older",
        "enable_autostart": "lmds enable nemo",
        "disable_autostart": "lmds disable nemo",
        "push_bundle": "lmds node push spark-head nemo",
        "deploy_plan": "lmds deploy org/Model --yes --target dgx-spark-single",
    }
    params = {"slug": "nemo", "slots": "2", "name": "gpt-4o", "node": "spark-head",
              "repo": "org/Model", "target": "dgx-spark-single"}
    for name, command in expect.items():
        action = catalog.ACTIONS[name]
        assert action.impact.strip(), f"{name} ไม่ได้บอกผลกระทบ"
        assert action.command(params)[0] == command, name
    assert catalog.ACTIONS["node_install"].command({})[0] == "lmds node install --all"
    assert catalog.ACTIONS["node_install"].command({"node": "spark-head"})[0] == "lmds node install spark-head"
    assert 'LLAMA_CPP_UPDATE=1 "$ctl" prepare-runtime' in catalog.ACTIONS["update_runtime"].command({"slug": "nemo"})[0]
    assert catalog.ACTIONS["stop_to_fit"].command({"slug": "nemo"})[0].endswith('"$ctl" stop')


def test_run_test_maps_choices_to_the_controller_or_lmds_bench():
    action = catalog.ACTIONS["run_test"]
    assert action.command({"slug": "nemo", "test": "test-tools"})[0].endswith('"$ctl" test-tools')
    assert action.command({"slug": "nemo", "test": "score"})[0] == "lmds bench run nemo --quick"
    with pytest.raises(catalog.ParamError):
        action.command({"slug": "nemo", "test": "rm -rf"})


def test_remove_model_keeps_weights_unless_told_otherwise_and_is_high_risk():
    action = catalog.ACTIONS["remove_model"]
    assert action.risk == "high"
    assert action.command({"slug": "nemo"})[0] == "lmds remove nemo --yes --keep-weights"
    assert action.command({"slug": "nemo", "keep_weights": "1"})[0] == "lmds remove nemo --yes --keep-weights"
    assert action.command({"slug": "nemo", "keep_weights": "0"})[0] == "lmds remove nemo --yes"


def test_the_menus_expose_choices_and_hub_only_for_the_router_and_the_page():
    actions = {row["name"]: row for row in catalog.action_menu()}
    assert actions["node_install"]["hub_only"] is True
    assert actions["deploy_model"]["composite"] is True
    test_param = next(p for p in actions["run_test"]["params"] if p["name"] == "test")
    assert "score" in test_param["choices"]
    assert "fleet_consistency" in {row["name"] for row in catalog.probe_menu()}


# ---------------------------------------------------------------------------
# probe ที่คำนวณบน hub — ไม่มี shell ไม่มี SSH แต่ผ่าน redact/trim เหมือนกัน
# ---------------------------------------------------------------------------
def test_a_compute_probe_runs_in_process_and_reports_errors_instead_of_raising(monkeypatch):
    called: list = []
    monkeypatch.setattr(runner, "_run_local", lambda *a, **k: called.append(a) or (0, "", ""))
    monkeypatch.setattr(insight, "model_recommend", lambda params, target: "sk-" + "a" * 48 + " อันดับ 1: coder")

    outcome = runner.run_probe("model_recommend", "this", {"task": "coding"})
    assert outcome.ok and "อันดับ 1: coder" in outcome.output
    assert "sk-" + "a" * 48 not in outcome.output, "ผลคำนวณต้องผ่าน redact เหมือน probe อื่น"
    assert not called, "probe ที่คำนวณต้องไม่แตะ shell"

    def boom(params, target):
        raise RuntimeError("แคชพัง")

    monkeypatch.setattr(insight, "model_recommend", boom)
    outcome = runner.run_probe("model_recommend", "this", {"task": "coding"})
    assert not outcome.ok and "แคชพัง" in outcome.error


def _snapshot(models_by_node: dict) -> dict:
    def host(unified=True):
        return {"memory_model": "unified", "ram_total_gb": 121.0, "ram_used_gb": 60.0,
                "gpus": [{"name": "GB10", "vram_gb": 121.0, "vram_used_gb": 60.0}]}
    snap = {"host": {"data": {"host": host(), "models": models_by_node.pop("this", [])}}, "nodes": {}}
    for name, models in models_by_node.items():
        snap["nodes"][name] = {"data": {"host": host(), "models": models}}
    return snap


def test_model_recommend_ranks_only_models_with_weights_and_explains_the_score(monkeypatch):
    from lmds.web import state

    snap = _snapshot({
        "this": [{"slug": "coder", "model_id": "Qwen/Qwen3-Coder-Next", "engine": "vllm", "features": "tools",
                  "context": 262144, "running": True, "downloaded": True, "memory_gb": 70.1}],
        "spark-worker": [
            {"slug": "gemma", "model_id": "google/gemma-4-26b", "engine": "llamacpp", "features": "image",
             "context": 32768, "running": False, "downloaded": True},
            {"slug": "ghost", "model_id": "org/not-here", "engine": "vllm", "features": "text",
             "context": 8192, "running": False, "downloaded": False},
        ],
    })
    monkeypatch.setattr(state.STORE, "snapshot", lambda: snap)
    monkeypatch.setattr("lmds.recipes.find_recipe", lambda repo: None)

    text = insight.model_recommend({"task": "coding tools"}, "this")
    first = text.split("\n1. ")[1].split("\n")[0]
    assert first.startswith("coder"), text
    assert "เครื่องนี้ (hub)" in first and "รันอยู่" in first and "70.1 GB" in first
    assert "ชื่อรุ่นเป็นสาย coder" in text and "เปิด tool calling ไว้แล้ว" in text
    assert "ไม่นับ (ยังไม่มี weight): ghost" in text
    assert "ยืนยันด้วย bench_results/run_test" in text, "คะแนนคือการจัดอันดับ ไม่ใช่ผลวัด — ต้องบอก"

    vision = insight.model_recommend({"task": "อ่านภาพ ภาษาไทย"}, "this")
    top = vision.split("\n1. ")[1].split("\n")[0]
    assert top.startswith("gemma @ spark-worker") or top.startswith("gemma (google/gemma-4-26b) @ spark-worker"), vision
    assert "ไม่รับภาพ" in vision


def test_fleet_consistency_never_calls_unknown_a_pass(monkeypatch):
    report = {
        "hub": {"version": "0.6.1", "commit": "aca7773", "template_hash": "abc", "dirty": ["src/x.py"], "verdict": None},
        "nodes": [
            {"name": "spark-head", "consistent": True, "level": "ok", "source": "cache",
             "code": {"state": "ok", "detail": "aca7773"}, "controllers": {"state": "ok", "detail": "3/3"},
             "runtimes": {"state": "ok", "detail": "รู้จักทุก arch"}},
            {"name": "spark-worker", "consistent": False, "level": "warn", "source": "registry",
             "code": {"state": "ok", "detail": "aca7773"}, "controllers": {"state": "unknown", "detail": "ยังไม่มี probe"},
             "runtimes": {"state": "stale", "detail": "build 18 ส.ค. ไม่รู้จัก qwen4exp"}},
        ],
        "summary": {"line": "ตรง hub 1 · ตรวจไม่ได้ 1"},
    }
    monkeypatch.setattr("lmds.fleet.consistency.fleet_report", lambda *a, **k: report)
    monkeypatch.setattr("lmds.nodes.load", lambda: [])

    text = insight.fleet_consistency({}, "this")
    assert "spark-head: ตรง hub — code ✓ · controllers ✓ · runtimes ✓" in text
    assert "spark-worker: ตรวจไม่ได้ครบ (จากทะเบียน ยังไม่มี probe ล่าสุด) — code ✓ · controllers ? · runtimes ✗" in text
    assert "controllers: ตรวจไม่ได้ — ยังไม่มี probe" in text
    assert "runtimes: stale — build 18 ส.ค. ไม่รู้จัก qwen4exp" in text
    assert "code: ok" not in text, "มิติที่ผ่านไม่ต้องขยาย — เปลืองงบ probe"
    assert "hub มีไฟล์แก้ค้าง 1 ไฟล์" in text
    assert "update_runtime" in text and "node_install" in text


# ---------------------------------------------------------------------------
# ตั๋ว: งานประกอบ · งานของ hub · งานลบถาวร · สาเหตุเมื่อล้ม
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_tickets():
    policy.reset()
    yield
    policy.reset()


@pytest.fixture
def ran(monkeypatch):
    log: list[tuple[str, str]] = []

    def fake(name, target="this", params=None):
        log.append((name, target))
        return runner.Outcome(name=name, title=name, target=target, exit_code=0, output="ok")

    monkeypatch.setattr(policy, "run_action", fake)
    return log


def test_deploy_model_expands_into_visible_steps_with_the_slug_the_cli_would_pick(ran):
    ticket = policy.propose([{"action": "deploy_model", "target": "spark-head",
                              "params": {"repo": "https://huggingface.co/Qwen/Qwen3-8B-GGUF", "gguf": "Q4_K_M"}}],
                            why="ลูกค้าขอโมเดลใหม่")
    steps = [(s.action, s.target) for s in ticket.steps]
    assert steps == [("deploy_plan", "this"), ("push_bundle", "this"), ("model_download", "spark-head"),
                     ("model_start", "spark-head"), ("run_test", "spark-head"), ("run_test", "spark-head")]
    assert ticket.steps[0].command == "lmds deploy Qwen/Qwen3-8B-GGUF --yes --gguf Q4_K_M"
    assert ticket.steps[1].command == "lmds node push spark-head qwen3-8b-gguf"
    assert ticket.steps[-1].command.endswith('"$ctl" test-tools')
    assert ticket.default_mode == policy.STEP, "หลายขั้นควรแนะนำ 'ทีละขั้น'"
    assert ticket.payload()["default_mode"] == "step" and ticket.payload()["destructive"] is False
    assert ran == [], "ขยายแผน ≠ ลงมือ"


def test_a_local_deploy_skips_the_push_step():
    ticket = policy.propose([{"action": "deploy_model", "target": "this",
                              "params": {"repo": "org/Model", "slug": "mine"}}])
    assert [s.action for s in ticket.steps] == ["deploy_plan", "model_download", "model_start", "run_test", "run_test"]
    assert all(s.target == "this" for s in ticket.steps)
    assert ticket.steps[1].params["slug"] == "mine"


def test_hub_only_work_runs_on_the_hub_even_if_the_llm_picked_a_node(ran):
    ticket = policy.propose([{"action": "node_install", "target": "spark-head", "params": {"node": "spark-head"}}])
    assert ticket.steps[0].target == "this"
    assert ticket.steps[0].command == "lmds node install spark-head"
    policy.choose(ticket.id, policy.APPLY)
    policy.advance(ticket.id)
    assert ran == [("node_install", "this")]


def test_destructive_work_defaults_to_hold_and_apply_needs_a_second_confirmation(ran):
    ticket = policy.propose([{"action": "remove_model", "target": "this", "params": {"slug": "old"}}])
    assert ticket.destructive and ticket.default_mode == policy.HOLD
    with pytest.raises(policy.PolicyError):
        policy.choose(ticket.id, policy.APPLY)
    assert ran == [] and ticket.mode == ""
    policy.choose(ticket.id, policy.APPLY, confirm=True)
    policy.advance(ticket.id)
    assert ran == [("remove_model", "this")]

    stepwise = policy.propose([{"action": "remove_model", "target": "this", "params": {"slug": "old"}}])
    policy.choose(stepwise.id, policy.STEP)   # ทีละขั้นไม่ต้องยืนยันซ้ำ — ทุกขั้นต้องกดอยู่แล้ว
    assert stepwise.mode == policy.STEP


def test_a_single_low_risk_step_defaults_to_apply():
    ticket = policy.propose([{"action": "set_fit", "target": "this", "params": {"slug": "nemo", "slots": "2"}}])
    assert ticket.default_mode == policy.APPLY


def test_a_failed_step_carries_the_real_cause_from_the_log(monkeypatch):
    def failing(name, target="this", params=None):
        return runner.Outcome(name=name, title=name, target=target, exit_code=1, output="start … failed (exit 1)")

    probes: list = []

    def fake_probe(name, target="this", params=None):
        probes.append((name, target, params))
        return runner.Outcome(name=name, title=name, target=target, exit_code=0,
                              output="== server.log\nunknown model architecture: 'qwen4exp'")

    monkeypatch.setattr(policy, "run_action", failing)
    monkeypatch.setattr(policy, "run_probe", fake_probe)
    ticket = policy.propose([
        {"action": "model_start", "target": "spark-worker", "params": {"slug": "qwen"}},
        {"action": "run_test", "target": "spark-worker", "params": {"slug": "qwen", "test": "test-text"}},
    ])
    policy.choose(ticket.id, policy.APPLY)
    policy.advance(ticket.id)
    assert probes == [("last_failure", "spark-worker", {"slug": "qwen"})]
    assert "qwen4exp" in ticket.steps[0].result["explain"]
    assert ticket.steps[1].done is False, "ขั้นถัดไปต้องไม่วิ่งหลังขั้นก่อนล้ม"


def test_an_explain_that_breaks_does_not_break_the_ticket(monkeypatch):
    monkeypatch.setattr(policy, "run_action", lambda name, target="this", params=None:
                        runner.Outcome(name=name, title=name, target=target, exit_code=1, output="พัง"))

    def boom(*a, **k):
        raise RuntimeError("ssh ล่ม")

    monkeypatch.setattr(policy, "run_probe", boom)
    ticket = policy.propose([{"action": "model_start", "target": "this", "params": {"slug": "x"}}])
    policy.choose(ticket.id, policy.APPLY)
    policy.advance(ticket.id)
    assert ticket.steps[0].result["explain"] == ""


# ---------------------------------------------------------------------------
# router: บริบทของการ์ดที่เปิดอยู่ และรายการใหม่ในเมนู
# ---------------------------------------------------------------------------
def test_the_router_prompt_lists_the_operator_catalog_with_choices():
    prompt = router.build_prompt(["spark-head"], ["nemo"])
    for name in ("fleet_consistency", "fit_preview", "last_failure", "model_recommend", "deploy_model",
                 "set_fit", "node_install", "remove_model", "run_test"):
        assert name in prompt, name
    assert "test=test-text|test-tools|test-vision|test-reasoning|score" in prompt


def test_the_card_the_user_is_looking_at_reaches_the_router():
    seen: dict = {}

    class Fake:
        def complete_json(self, system, user):
            seen["user"] = user
            return '{"probes": [{"name": "last_failure", "target": "spark-head", "params": {"slug": "nemo"}}]}'

    plan = router.choose("ทำไมไม่ขึ้น", ["spark-head"], ["nemo"], provider=Fake(),
                         context={"node": "spark-head", "slug": "nemo"})
    assert seen["user"].startswith("ผู้ใช้กำลังดูการ์ด: เครื่อง spark-head · โมเดล nemo")
    assert plan.probes == [{"name": "last_failure", "target": "spark-head", "params": {"slug": "nemo"}}]
    assert router.context_line({}) == "" and router.context_line(None) == ""


def test_a_multi_step_plan_survives_validation_in_order():
    plan = router.validate({"action": {"why": "ย้าย", "steps": [
        {"name": "push_bundle", "target": "this", "params": {"node": "spark-head", "slug": "nemo"}},
        {"name": "set_fit", "target": "spark-head", "params": {"slug": "nemo", "slots": "2"}},
        {"name": "model_start", "target": "spark-head", "params": {"slug": "nemo"}},
        {"name": "run_test", "target": "spark-head", "params": {"slug": "nemo", "test": "test-text"}},
    ]}}, targets={"spark-head"})
    assert [s["action"] for s in plan.action_steps] == ["push_bundle", "set_fit", "model_start", "run_test"]


# ---------------------------------------------------------------------------
# สมองจากฟลีต
# ---------------------------------------------------------------------------
def _model(**over) -> dict:
    base = {"slug": "nemo", "model_id": "nvidia/Nemotron", "engine": "vllm", "port": 8000, "running": True,
            "features": "tools", "served_name": "nemotron-3"}
    base.update(over)
    return base


def test_only_a_running_chat_model_can_become_the_brain():
    chosen = brain._pick(_model(), "spark-head", "10.0.0.5")
    assert chosen.base_url == "http://10.0.0.5:8000/v1" and chosen.model == "nemotron-3"
    with pytest.raises(brain.BrainError):
        brain._pick(_model(running=False), "this", "127.0.0.1")
    with pytest.raises(brain.BrainError):
        brain._pick(_model(features="embedding (mean)"), "this", "127.0.0.1")
    with pytest.raises(brain.BrainError):
        brain._pick(_model(engine="ollama"), "this", "127.0.0.1")
    # ไม่มี served name = ใช้ model_id — ชื่อที่ /v1/models ประกาศจริง ไม่ใช่ slug
    assert brain._pick(_model(served_name="", default_served_name=""), "this", "127.0.0.1").model == "nvidia/Nemotron"


def test_from_cache_uses_the_node_ip_and_the_hub_loopback(monkeypatch):
    from types import SimpleNamespace

    from lmds.web import state

    snap = {"host": {"data": {"models": [_model(slug="local", port=8020)]}},
            "nodes": {"spark-head": {"data": {"models": [_model()]}}}}
    monkeypatch.setattr(state.STORE, "snapshot", lambda: snap)
    monkeypatch.setattr("lmds.nodes.find", lambda name: SimpleNamespace(local_ip="192.168.10.30", host="spark-head")
                        if name == "spark-head" else None)

    assert brain.from_cache("", "local").base_url == "http://127.0.0.1:8020/v1"
    assert brain.from_cache("spark-head", "nemo").base_url == "http://192.168.10.30:8000/v1"
    with pytest.raises(brain.BrainError):
        brain.from_cache("spark-head", "missing")
    with pytest.raises(brain.BrainError):
        brain.from_cache("ghost", "nemo")


def test_apply_goes_through_the_same_provider_settings_as_the_provider_page():
    from lmds.config import Settings

    provider = brain.apply(brain.Brain(node="spark-head", slug="nemo", model="nemotron-3",
                                       base_url="http://192.168.10.30:8000/v1"))
    assert provider.name.value == "openai-compat"
    saved = Settings.load().provider
    assert saved.model == "nemotron-3" and saved.base_url == "http://192.168.10.30:8000/v1"


def test_owner_matches_the_configured_brain_back_to_a_fleet_model(monkeypatch):
    from types import SimpleNamespace

    from lmds.config import ProviderName, Settings

    snap = {"host": {"data": {"host": {"ip": "10.1.1.1"}, "models": [_model(slug="local", port=8020)]}},
            "nodes": {"spark-head": {"data": {"models": [_model()]}}}}
    monkeypatch.setattr("lmds.nodes.load", lambda: [SimpleNamespace(name="spark-head", local_ip="192.168.10.30",
                                                                    host="spark-head", alt_hosts=[])])
    settings = Settings()
    settings.set_provider(ProviderName.OPENAI_COMPAT, model="x", base_url="http://192.168.10.30:8000/v1")
    assert brain.owner(settings.provider, snap) == {"node": "spark-head", "slug": "nemo"}
    settings.set_provider(ProviderName.OPENAI_COMPAT, model="x", base_url="http://127.0.0.1:8020/v1")
    assert brain.owner(settings.provider, snap) == {"node": "this", "slug": "local"}
    settings.set_provider(ProviderName.OPENAI_COMPAT, model="x", base_url="http://100.92.10.49:8080/v1")
    assert brain.owner(settings.provider, snap) is None, "สมองข้างนอกไม่ใช่โมเดลในฟลีต"
    assert brain.owner(None, snap) is None


# ---------------------------------------------------------------------------
# ชั้นเว็บ: บริบท · ชิป · คำแนะนำถัดไป
# ---------------------------------------------------------------------------
def test_context_keeps_only_real_nodes_and_well_formed_slugs():
    from lmds.web import assistant

    state = {"nodes": [{"name": "spark-head", "models": [{"slug": "nemo"}]}], "models_here": []}
    assert assistant.clean_context({"node": "spark-head", "slug": "nemo"}, state) == {"node": "spark-head", "slug": "nemo"}
    assert assistant.clean_context({"node": "ghost", "slug": "a;b"}, state) == {}
    assert assistant.clean_context("junk", state) == {}


def test_the_focus_line_leads_the_state_block(monkeypatch):
    from lmds.web import assistant

    monkeypatch.setattr(assistant, "gather_state", lambda: {"nodes": [{"name": "spark-head"}], "brain": None})
    system, _ = assistant.build_messages([{"role": "user", "content": "ตัวนี้ทำไมช้า"}],
                                         {"probes": [], "docs": []}, {"node": "spark-head", "slug": "nemo"})
    assert "ผู้ใช้กำลังดูการ์ด: เครื่อง spark-head · โมเดล nemo\nSYSTEM STATE" in system


def test_capabilities_are_grouped_and_counted_from_the_catalog():
    from lmds.web import assistant

    caps = assistant.capabilities()
    assert [g["group"] for g in caps["groups"]] == ["ตรวจ", "แก้", "ตั้งค่า", "deploy", "แนะนำโมเดล"]
    assert caps["probes"] == len(catalog.PROBES) and caps["actions"] == len(catalog.ACTIONS)
    assert all(g["examples"] for g in caps["groups"])


def test_followups_depend_on_what_was_just_checked():
    from lmds.web import assistant

    after_logs = assistant.followups({"probes": [{"name": "model_logs", "target": "spark-head", "params": {"slug": "nemo"}}]})
    assert after_logs[0] == "สาเหตุที่ nemo start ล้มครั้งล่าสุดคืออะไร"
    assert any("Fit nemo บน spark-head" in s for s in after_logs)

    after_fleet = assistant.followups({"probes": [{"name": "fleet_consistency", "target": "this", "params": {}}]})
    assert "อัปเดตเครื่องที่ยังไม่ตรง hub ให้ครบ" in after_fleet

    with_ticket = assistant.followups({"probes": [], "ticket": {"ticket": "x"}})
    assert with_ticket[0] == "ถ้าขั้นไหนล้ม อธิบายสาเหตุจาก log ให้หน่อย"

    nothing = assistant.followups({"probes": []}, {"slug": "nemo"})
    assert nothing == ["nemo ตอนนี้เป็นยังไง"]
    assert len(assistant.followups({"probes": [{"name": n, "params": {"slug": "a"}, "target": "n1"}
                                               for n in ("model_logs", "fleet_consistency", "usage", "bench_results")]})) <= 4
