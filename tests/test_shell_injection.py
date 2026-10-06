"""ค่าจากข้างนอกต้องไม่กลายเป็นคำสั่งบน node — audit templates 2026-10-06 ข้อ 1

controller เป็น bash ที่ render จาก template แล้วรันบนเครื่องลูกค้า · ค่าที่ template แทรกลงไปมาจากสองแหล่งที่เราไม่ได้คุม:
ชื่อไฟล์ใน repo (Hugging Face Hub — ใครก็ตั้ง repo ได้) กับค่าในแผน (LLM เป็นคนเสนอ) · เดิมทั้งหมดถูกวางดิบใน `"…"`

    repo มีไฟล์ชื่อ   Qwen3-8B-Q4_K_M$(touch PWNED_gguf).gguf
    แผนมี            served_model_name = qwen$(touch PWNED_served_name)
    → gate ผ่านครบทุกด่าน → `controller help` สร้างไฟล์ PWNED_* บนเครื่อง

เทสในไฟล์นี้ render bundle จริงแล้ว **รันใต้ bash จริง** ในโฟลเดอร์เปล่า — ไฟล์ที่โผล่ในโฟลเดอร์นั้นคือหลักฐานว่าโค้ดถูกรัน
ไม่ได้เทียบสตริงในซอร์ส · แยกพิสูจน์ทีละชั้น เพราะสามชั้นต้องไม่พึ่งกัน:

  (a) renderer escape ทุกค่าตามบริบท       — เทสปิดชั้น (b) แล้วดูว่าค่าร้ายยังเฉย
  (b) ค่าที่ไม่ควรมี metacharacter ถูกปฏิเสธ — inspector · harden_plan · renderer
  (c) gate เทียบกับ render แบบ canary      — เทสปิดชั้น (a)+(b) แล้วดูว่า gate จับได้
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from lmds.brain import build_plan
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import Engine, RuntimeAsset
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle, renderer
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport, ShardFile
from lmds.shellsafe import UnsafeValueError
from lmds.validator.gates import gate_value_expansion, run_gates

# controller เล็งที่ bash 5 ของ Linux — บน macOS `/bin/bash` คือ 3.2 จึงหยิบของ homebrew ก่อนถ้ามี
BASH = next((p for p in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash") if os.access(p, os.X_OK)), "bash")
SAFE_PATH = "/usr/bin:/bin:/sbin:/usr/sbin"
KINDS = ("vllm", "sglang", "llamacpp", "stacked")

SHARDS = [("model-00001-of-00002.safetensors", 12), ("model-00002-of-00002.safetensors", 7)]


def _report(kind: str, **overrides) -> ModelReport:
    if kind == "llamacpp":
        base = dict(
            repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
            weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
            selected_gguf="Qwen3-8B-Q4_K_M.gguf",
            gguf_variants=[GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=12, sha256=None)],
            has_chat_template=True, license="apache-2.0",
        )
    elif kind == "stacked":
        base = dict(
            repo_id="nvidia/DeepSeek-V4-Flash-NVFP4", revision_sha="rev-ds4", artifact_type=ArtifactType.SAFETENSORS,
            weight_bytes=157 * GIB, shard_count=2, context_length=131072,
            kv_dims=KvDims(layers=61, kv_heads=128, head_dim=128), license="mit", has_chat_template=True,
            safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in SHARDS],
        )
    else:
        base = dict(
            repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned-123", artifact_type=ArtifactType.SAFETENSORS,
            weight_bytes=65 * GIB, shard_count=2, context_length=40960,
            kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
            safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in SHARDS],
        )
    base.update(overrides)
    return ModelReport(**base)


def _bundle(tmp_path: Path, kind: str, report: ModelReport | None = None, tweak=None):
    report = report or _report(kind)
    fit = analyze(report, PRESETS["dgx-spark-stacked" if kind == "stacked" else "dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    if tweak:
        tweak(plan)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def _run(controller: Path, *args: str, cwd: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """รัน controller จริงในโฟลเดอร์เปล่า — อะไรที่โผล่ใน cwd หลังจากนี้คือของที่ controller สร้าง"""
    cwd.mkdir(parents=True, exist_ok=True)
    home = cwd.parent / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": SAFE_PATH, "HOME": str(home), **(extra_env or {})}
    return subprocess.run([BASH, str(controller), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def _values(controller: Path, cwd: Path, *expressions: str) -> list[str]:
    """ค่าที่ bash เห็นจริงหลังอ่านหัว controller — source ไฟล์แล้วพิมพ์ตอน EXIT (สคริปต์จบด้วย exit เอง)"""
    cwd.mkdir(parents=True, exist_ok=True)
    home = cwd.parent / "home"
    home.mkdir(exist_ok=True)
    out = cwd.parent / "values.out"
    printer = " ".join(f'"{e}"' for e in expressions)
    script = f"trap 'printf \"%s\\0\" {printer} > \"$OUT\"' EXIT\nset -- help\n. \"$CONTROLLER\" >/dev/null 2>&1\n"
    subprocess.run([BASH, "-c", script], cwd=cwd, capture_output=True, text=True, timeout=60,
                   env={"PATH": SAFE_PATH, "HOME": str(home), "OUT": str(out), "CONTROLLER": str(controller)})
    return out.read_text(encoding="utf-8").split("\0")[:-1]


def _created(cwd: Path) -> list[str]:
    return sorted(p.name for p in cwd.iterdir())


# ═════════════════════ (b) ปฏิเสธตั้งแต่ต้นทาง ═════════════════════
@pytest.mark.parametrize("kind,evil", [
    ("llamacpp", "Qwen3-8B-Q4_K_M$(touch PWNED_gguf).gguf"),
    ("vllm", "model-00002-of-00002$(touch PWNED_shard).safetensors"),
    ("sglang", "model-00002-of-00002`touch PWNED_shard`.safetensors"),
    ("stacked", 'model-00002-of-00002"; touch PWNED_shard; ".safetensors'),
])
def test_a_repo_file_named_like_a_command_is_refused_before_anything_is_written(tmp_path, kind, evil):
    """ตัว repro ของ auditor (inject.py): ชื่อไฟล์จาก Hub ที่มี $(…) — เดิม render ได้ gate ผ่าน แล้ว help รันมัน"""
    if kind == "llamacpp":
        report = _report(kind, selected_gguf=evil, gguf_variants=[GgufVariant(filename=evil, size_bytes=12, sha256=None)])
    else:
        report = _report(kind, safetensor_shards=[ShardFile(filename=SHARDS[0][0], size_bytes=12),
                                                  ShardFile(filename=evil, size_bytes=7)])
    with pytest.raises(UnsafeValueError) as refused:
        _bundle(tmp_path, kind, report)
    assert evil in str(refused.value), "ข้อความต้องบอกว่าไฟล์ไหน ไม่งั้นผู้ใช้ไม่รู้ว่าต้องไปดูอะไรใน repo"
    assert not list((tmp_path / "bundles").rglob("*.sh")), "ปฏิเสธแล้วต้องไม่ทิ้ง controller ไว้ให้ใครรัน"


@pytest.mark.parametrize("kind", KINDS)
def test_file_names_real_repos_use_still_render_and_reach_bash_unchanged(tmp_path, kind):
    """กติกาชื่อไฟล์ต้องไม่ตัดของจริงทิ้ง: โฟลเดอร์ย่อย · + = @ , · ช่องว่าง · ชื่อภาษาไทย (สระ/วรรณยุกต์เป็น combining mark)"""
    names = ["weights/model+lora=v2@main,final.safetensors", "โมเดล ที่ปรับแล้ว.safetensors"]
    if kind == "llamacpp":
        names = [n.replace(".safetensors", ".gguf") for n in names]
        report = _report(kind, selected_gguf=names[0],
                         gguf_variants=[GgufVariant(filename=n, size_bytes=12, sha256=None) for n in names])
        bundle = _bundle(tmp_path, kind, report)
        seen = _values(bundle.controller, tmp_path / "cwd", "${MODEL_FILES[0]}", "${MODEL_URLS[0]}")
        assert seen[0] == "model+lora=v2@main,final.gguf"
        assert seen[1].endswith("/resolve/sha-gguf-456/weights/model+lora=v2@main,final.gguf")
    else:
        report = _report(kind, safetensor_shards=[ShardFile(filename=n, size_bytes=5) for n in names])
        bundle = _bundle(tmp_path, kind, report)
        assert _values(bundle.controller, tmp_path / "cwd", "${SHARD_FILES[0]}", "${SHARD_FILES[1]}") == names
    assert all(g.passed for g in run_gates(bundle.directory, include_checksums=False))


def test_harden_plan_puts_back_a_served_name_the_shell_would_interpret():
    """ตัว repro ของ auditor (inject_plan.py): ชื่อ parser ถูกตรวจกับรายชื่อจริงอยู่แล้ว แต่ served_model_name ผ่านมาทั้งดุ้น"""
    report = _report("vllm")
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.served_model_name = "qwen$(touch PWNED_served_name)"
    plan.tool_calling.chat_template_override = "tool.jinja`touch PWNED_tpl`"
    plan.serving.kv_cache_dtype = "fp8;touch PWNED_kv"
    plan.warnings = []

    hardened = harden_plan(plan, report, fit)

    assert hardened.served_model_name == "qwen3-32b"
    assert hardened.tool_calling.chat_template_override is None
    assert hardened.serving.kv_cache_dtype == "auto"
    assert any("served_model_name" in w and "PWNED_served_name" in w for w in hardened.warnings), hardened.warnings


@pytest.mark.parametrize("name", ["qwen3-32b", "Qwen/Qwen3-32B", "gpt-4o:latest", "my_model.v2"])
def test_harden_plan_keeps_ordinary_served_names(name):
    report = _report("vllm")
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.served_model_name = name
    assert harden_plan(plan, report, fit).served_model_name == name


def test_inspector_skips_repo_files_whose_names_are_not_plain_and_says_so():
    """ชื่อไฟล์เข้าระบบที่ inspector — ไฟล์ชื่อแปลกต้องไม่เคยเป็นตัวเลือกให้ใครเลือก และต้องบอกว่าข้ามอะไร"""
    from lmds.inspector.inspect import _sibling_files

    evil = "Qwen3-8B-Q4_K_M$(touch PWNED_gguf).gguf"
    info = {"siblings": [
        {"rfilename": "Qwen3-8B-Q4_K_M.gguf", "size": 12},
        {"rfilename": evil, "size": 12},
        {"rfilename": "docs/โมเดล ภาษาไทย+v2.gguf", "size": 3},
        {"rfilename": "a\nb.gguf", "size": 1},
        {"rfilename": "../escape.gguf", "size": 1},
    ]}
    skipped: list[str] = []
    kept = [name for name, _, _ in _sibling_files(info, skipped)]
    assert kept == ["Qwen3-8B-Q4_K_M.gguf", "docs/โมเดล ภาษาไทย+v2.gguf"]
    assert skipped == [evil, "a\nb.gguf", "../escape.gguf"]


def test_inspect_model_reports_the_files_it_skipped():
    from lmds.inspector.inspect import inspect_model
    from lmds.resolver import parse_source

    evil = "model-00002-of-00002$(touch PWNED_shard).safetensors"

    class Client:
        def model_info(self, repo_id, revision=None):
            return {"sha": "abc123", "siblings": [{"rfilename": "notes.txt", "size": 1}, {"rfilename": evil, "size": 7}]}

    report = inspect_model(parse_source("x/y"), Client())
    assert not report.safetensor_shards, "ไฟล์ชื่อแปลกต้องไม่ถูกนับเป็น shard"
    assert any("ข้ามไฟล์" in w and "PWNED_shard" in w for w in report.warnings), report.warnings


# ═════════════════════ (a) escape ตอน render — ปิดชั้น (b) แล้วค่าร้ายต้องยังเฉย ═════════════════════
PAYLOADS = {
    "served": "qwen$(touch PWNED_served)",
    "tool": "hermes`touch PWNED_tool`",
    "reasoning": 'qwen3"; touch PWNED_reasoning; echo "',
    "chat": "tpl$PWNED_var$(touch PWNED_chat).jinja",
    "image": "registry.local/img:1$(touch PWNED_image)",
    "kv": "fp8$(touch PWNED_kv)",
    "revision": "sha`touch PWNED_revision`",
    "generator": "llm:x/$(touch PWNED_generator)",
    "file": "part$(touch PWNED_file)\\\"`touch PWNED_file2`",
    "flag": '--chat-template-kwargs {"a":"$(touch PWNED_flag)"}',
    "env": "$(touch PWNED_env)",
}


def _hostile(tmp_path: Path, kind: str, monkeypatch):
    """bundle ที่ทุกค่าข้อความเป็น payload — ปิดด่านปฏิเสธ (b) ไว้ เพื่อพิสูจน์ว่าชั้น escape ยืนได้ด้วยตัวเอง"""
    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)
    file_name = PAYLOADS["file"] + (".gguf" if kind == "llamacpp" else ".safetensors")
    if kind == "llamacpp":
        report = _report(kind, revision_sha=PAYLOADS["revision"], selected_gguf=file_name,
                         gguf_variants=[GgufVariant(filename=file_name, size_bytes=12, sha256=None)])
    else:
        report = _report(kind, revision_sha=PAYLOADS["revision"],
                         safetensor_shards=[ShardFile(filename=file_name, size_bytes=12)])

    def tweak(plan):
        plan.served_model_name = PAYLOADS["served"]
        plan.tool_calling.enabled, plan.tool_calling.parser = True, PAYLOADS["tool"]
        plan.reasoning.enabled, plan.reasoning.parser = True, PAYLOADS["reasoning"]
        plan.tool_calling.chat_template_override = PAYLOADS["chat"]
        plan.runtime.image_ref, plan.runtime.image_pin = PAYLOADS["image"], None
        plan.serving.kv_cache_dtype = PAYLOADS["kv"]
        plan.generator = PAYLOADS["generator"]
        plan.serving.extra_flags = [PAYLOADS["flag"]]
        plan.serving.extra_env = {"LMDS_TEST_ENV": PAYLOADS["env"]}
        if kind != "llamacpp":
            plan.runtime_assets = [RuntimeAsset(filename="p$(touch PWNED_asset).py",
                                                url="https://example.com/$(touch PWNED_url)", sha256=None)]

    return _bundle(tmp_path, kind, report, tweak), file_name


@pytest.mark.parametrize("kind", KINDS)
def test_hostile_values_reach_bash_as_plain_text_and_run_nothing(tmp_path, kind, monkeypatch):
    bundle, file_name = _hostile(tmp_path, kind, monkeypatch)
    cwd = tmp_path / "cwd"

    done = _run(bundle.controller, "help", cwd=cwd)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _created(cwd) == [], f"controller help รันค่าที่แทรกมาเป็นคำสั่ง: {_created(cwd)}"

    image_var = {"vllm": "VLLM_IMAGE", "stacked": "VLLM_IMAGE", "sglang": "SGLANG_IMAGE", "llamacpp": "LLAMACPP_IMAGE"}[kind]
    files_var = "MODEL_FILES" if kind == "llamacpp" else "SHARD_FILES"
    seen = _values(bundle.controller, cwd, "$SERVED_MODEL_NAME", "$MODEL_REVISION", f"${image_var}", f"${{{files_var}[0]}}")
    # ไม่ใช่แค่ "ไม่รัน" — ค่าต้องไปถึงตัวแปรครบทุกตัวอักษร ไม่งั้น escape ผิดบริบทแล้วค่าเพี้ยนเงียบ ๆ
    assert seen == [PAYLOADS["served"], PAYLOADS["revision"], PAYLOADS["image"], file_name]
    if kind != "llamacpp":
        assert _values(bundle.controller, cwd, "$TOOL_CALL_PARSER", "$REASONING_PARSER", "$CHAT_TEMPLATE") == [
            PAYLOADS["tool"], PAYLOADS["reasoning"], PAYLOADS["chat"]]
    assert _created(cwd) == []


@pytest.mark.parametrize("kind", KINDS)
def test_hostile_values_stay_inert_on_the_way_to_the_engine_argv(tmp_path, kind, monkeypatch):
    """ค่าที่อยู่ในตัวฟังก์ชัน start (extra flag · kv dtype) ถูกอ่านตอนประกอบ argv ไม่ใช่ตอนโหลดไฟล์ — ต้องรันถึงตรงนั้นจริง"""
    bundle, _ = _hostile(tmp_path, kind, monkeypatch)
    cwd = tmp_path / "cwd"
    if kind in ("llamacpp", "stacked"):
        done = _run(bundle.controller, "serve-args", cwd=cwd,
                    extra_env={"MASTER_IP": "10.0.0.1", "WORKER_IP": "10.0.0.2", "SSH_USER": "u"})
    else:
        done = _run(bundle.controller, "start", cwd=cwd, extra_env={"DRY_RUN": "1"})
    printed = done.stdout + done.stderr
    assert _created(cwd) == [], f"ประกอบ argv แล้วมีคำสั่งถูกรัน: {_created(cwd)}\n{printed}"
    assert done.returncode == 0, printed
    assert '{"a":"$(touch PWNED_flag)"}' in printed, "flag ต้องไปถึง argv เป็นตัวหนังสือเดิมทั้งก้อน"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bad", ["name}tail", "it's"])
def test_a_value_bash_cannot_hold_in_a_default_is_refused_not_mangled(tmp_path, kind, bad, monkeypatch):
    """`}` ปิด ${VAR:-…} กลางคัน และ `'` เปิด quote ซ้อน — bash 3.2 กับ 5.3 ตีความ `\\}` ไม่ตรงกัน จึงไม่มี escape ที่ไว้ใจได้
    ทางเดียวที่ถูกคือไม่ render · เดิม: `"${SERVED_MODEL_NAME:-name}tail}"` = ชื่อผิดเงียบ ๆ / `it's` = syntax error ตอนรัน"""
    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)

    def tweak(plan):
        plan.runtime.image_ref, plan.runtime.image_pin = f"registry.local/{bad}:1", None

    with pytest.raises(UnsafeValueError):
        _bundle(tmp_path, kind, tweak=tweak)


@pytest.mark.parametrize("kind", KINDS)
def test_a_newline_cannot_turn_a_comment_into_a_command(tmp_path, kind, monkeypatch):
    """หัวไฟล์มี `# Target: … (generator)` — ค่าที่มีขึ้นบรรทัดใหม่ทำให้บรรทัดถัดไปเป็นคำสั่ง · ไม่มีค่าไหนควรมี จึงไม่ render"""
    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)

    def tweak(plan):
        plan.generator = "llm:x/y\ntouch PWNED_newline"

    with pytest.raises(UnsafeValueError):
        _bundle(tmp_path, kind, tweak=tweak)


@pytest.mark.parametrize("kind", KINDS)
def test_every_interpolation_in_a_controller_template_goes_through_the_escaper(tmp_path, kind):
    """ตรวจ `{{ … }}` ทุกจุดของ template ด้วยการรัน ไม่ใช่ด้วยการอ่าน: render แบบ canary แทน *ทุกค่าข้อความ* ด้วยคำเดียว —
    ค่าจริงตัวไหนยังโผล่ในผลลัพธ์ แปลว่ามีทางแทรกที่ไม่ผ่านกลไก escape ของ renderer"""
    marker = "Zq7Xw"
    file_name = f"{marker}-file" + (".gguf" if kind == "llamacpp" else ".safetensors")
    if kind == "llamacpp":
        report = _report(kind, repo_id=f"{marker}/{marker}-GGUF", revision_sha=f"{marker}rev", selected_gguf=file_name,
                         gguf_variants=[GgufVariant(filename=file_name, size_bytes=12, sha256=f"{marker}sha")])
    else:
        report = _report(kind, repo_id=f"{marker}/{marker}-model", revision_sha=f"{marker}rev",
                         tokenizer_files=[f"{marker}-tokenizer.json"],
                         safetensor_shards=[ShardFile(filename=file_name, size_bytes=12)])
    fit = analyze(report, PRESETS["dgx-spark-stacked" if kind == "stacked" else "dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    plan.served_model_name = f"{marker}-served"
    plan.tool_calling.enabled, plan.tool_calling.parser = True, f"{marker}tool"
    plan.reasoning.enabled, plan.reasoning.parser = True, f"{marker}reason"
    plan.tool_calling.chat_template_override = f"{marker}.jinja"
    plan.runtime.image_ref, plan.runtime.image_pin = f"{marker}/img:1", f"sha256:{marker}"
    plan.serving.kv_cache_dtype = f"{marker}dtype"
    plan.generator = f"llm:{marker}"
    plan.serving.extra_flags = [f"--{marker}-flag value"]
    plan.serving.extra_env = {"K": f"{marker}env"}
    plan.runtime_assets = [RuntimeAsset(filename=f"{marker}.py", url=f"https://example.com/{marker}", sha256="a" * 64)]

    canary = renderer.render_canary_controller(plan, report, fit, slug=f"{marker.lower()}-slug")

    leaked = [line for line in canary.splitlines() if marker.lower() in line.lower()]
    assert not leaked, "ค่าพวกนี้ถูกแทรกโดยไม่ผ่าน escaper:\n" + "\n".join(leaked[:10])
    assert "LMDSCANARY" in canary


def test_the_bundle_readme_is_a_document_and_is_not_shell_escaped(tmp_path):
    """README.md ไม่ถูกรัน — backtick ในคำเตือนคือ markdown ถ้า escape แบบ bash จะกลายเป็น \\` ให้คนอ่าน"""
    def tweak(plan):
        plan.warnings.append("ตั้งค่าได้ด้วย `lmds set` ราคา $5")

    bundle = _bundle(tmp_path, "vllm", tweak=tweak)
    assert "- ตั้งค่าได้ด้วย `lmds set` ราคา $5" in (bundle.directory / "README.md").read_text(encoding="utf-8")


# ═════════════════════ (c) gate: เทียบกับ render แบบ canary ═════════════════════
@pytest.mark.parametrize("kind", KINDS)
def test_the_gate_compares_every_template_and_passes_ordinary_bundles(tmp_path, kind):
    bundle = _bundle(tmp_path, kind)
    result = gate_value_expansion(bundle.directory)
    assert result.passed, result.detail
    assert "canary" in result.detail and "n/a" not in result.detail, (
        f"ต้องเทียบได้จริง ไม่ใช่ข้าม — ด่านที่ข้ามทุกครั้งคือด่านที่ไม่มี: {result.detail}")


def test_the_gate_still_compares_a_bundle_whose_repo_reported_no_shards(tmp_path):
    """Hub ไม่รายงาน shard → controller ไม่มีตาราง SHARD_FILES → refresh สร้างแผนกลับไม่ได้ · ด่านนี้ต้องยังเทียบได้
    (รอบแรก 184 จาก 410 bundle ในชุดเทสตกไปเป็น n/a ด้วยเหตุนี้ — ด่านที่ข้ามเกือบครึ่งคือด่านที่ไม่มี)"""
    bundle = _bundle(tmp_path, "vllm", _report("vllm", safetensor_shards=[], shard_count=17))
    assert "SHARD_FILES=(" not in bundle.controller.read_text(encoding="utf-8")
    result = gate_value_expansion(bundle.directory)
    assert result.passed and "n/a" not in result.detail, result.detail


@pytest.mark.parametrize("kind", KINDS)
def test_the_gate_fails_the_bundle_the_auditor_got_past_every_gate(tmp_path, kind, monkeypatch):
    """สร้างสิ่งที่ renderer เดิมทำ: ไม่ปฏิเสธ ไม่ escape — bundle นี้เคยผ่านครบทุกด่าน แล้ว help รันคำสั่ง
    gate ต้องไม่ผ่าน และเพื่อไม่ให้เป็นการจับผิดลม ๆ เทสรัน help ให้ดูว่าไฟล์ถูกสร้างจริง"""
    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)
    monkeypatch.setattr(renderer, "_sh", lambda value, context="dq": value)
    monkeypatch.setattr(renderer, "_ENVIRONMENTS", {})   # environment ที่เก็บไว้ผูกกับ _sh ตัวจริง

    def tweak(plan):
        plan.served_model_name = "qwen$(touch PWNED_served_name)"

    bundle = _bundle(tmp_path, kind, tweak=tweak)
    cwd = tmp_path / "cwd"
    _run(bundle.controller, "help", cwd=cwd)
    assert _created(cwd) == ["PWNED_served_name"], "ตัวตั้งของเทสผิด: bundle นี้ต้องเป็นตัวที่รันคำสั่งได้จริง"

    monkeypatch.undo()
    results = {g.name: g for g in run_gates(bundle.directory, include_checksums=False)}
    assert not results["value-expansion"].passed
    assert "$(" in results["value-expansion"].detail and bundle.controller.name in results["value-expansion"].detail


@pytest.mark.parametrize("before,after,why", [
    ('  "model-00001-of-00002.safetensors"', '  "model-00001`touch PWNED`.safetensors"', "`"),
    ('MODEL_ID="Qwen/Qwen3-32B"', 'MODEL_ID="Qwen/Qwen3-32B${IFS}x"', "${"),
    ('MODEL_REVISION="sha-pinned-123"', 'MODEL_REVISION="sha"; touch PWNED; X="y"', "quote"),
])
def test_the_gate_names_the_line_where_a_value_became_code(tmp_path, before, after, why):
    bundle = _bundle(tmp_path, "vllm")
    text = bundle.controller.read_text(encoding="utf-8")
    assert text.count(before) >= 1
    bundle.controller.write_text(text.replace(before, after, 1), encoding="utf-8")
    line = text[: text.index(before)].count("\n") + 1

    result = gate_value_expansion(bundle.directory)

    assert not result.passed
    assert f"{bundle.controller.name}:{line}:" in result.detail and why in result.detail, result.detail


def test_the_gate_does_not_mistake_quoted_text_for_code(tmp_path):
    """flag/env ที่มี $ อยู่ใน single quote (shlex.quote) เป็นตัวหนังสือ ไม่ใช่ expansion — ต้องไม่ถูกตีตก
    และค่าที่มี ' ในคอมเมนต์หัวไฟล์ (ชื่อผู้ถือไลเซนส์) ต้องไม่ทำให้นับ quote เพี้ยน"""
    def tweak(plan):
        plan.serving.extra_flags = ['--chat-template-kwargs {"price":"$(5)","it":"don\'t"}']
        plan.serving.extra_env = {"PROMPT": "${not_expanded} `x`"}
        plan.generator = "llm:o'brien/model"

    for kind in ("vllm", "stacked"):
        bundle = _bundle(tmp_path / kind, kind, tweak=tweak)
        result = gate_value_expansion(bundle.directory)
        assert result.passed and "n/a" not in result.detail, result.detail


def test_the_gate_steps_aside_for_bundles_it_cannot_rebuild(tmp_path):
    """bundle จาก `lmds adopt` ไม่มี template — เทียบไม่ได้ ต้องบอกว่า n/a ไม่ใช่ตีตก (และไม่ใช่เงียบ)"""
    directory = tmp_path / "adopted"
    directory.mkdir()
    (directory / "vllm-gemma4-adopted.sh").write_text("#!/usr/bin/env bash\necho \"$(date)\"\n", encoding="utf-8")
    (directory / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump({"generated_by": "lmds adopt"}), encoding="utf-8")
    result = gate_value_expansion(directory)
    assert result.passed and result.detail.startswith("n/a")


# ═════════════════════ ทุกบรรทัดที่มีค่า: ด่านต้องเฝ้า · และ bash ต้องเห็นบริบทตรงกับที่ renderer คิด ═════════════════════
# เคสจริง 2026-10-06 (หลัง merge งาน template อีกสองสาย): ด่าน value-expansion รุ่นแรกไล่นับ quote ของทั้งไฟล์ พอมีบรรทัด
#   API_IDS="$(printf '%s' "$body" | … | grep -o '"id":"[^"]*"' | sed 's/^"id":"//; s/"$//' || true)"
# ตัวนับก็ค้างอยู่ใน "single quote" ไปจนจบไฟล์ — ค่าที่แทรกหลังบรรทัดนั้นไม่ถูกตรวจเลยและด่านรายงานว่าผ่าน
# เทสเดิมแทรกค่าแค่ 3 บรรทัดของ template เดียว จึงจับไม่ได้จนกว่าบรรทัดนั้นจะมาอยู่เหนือมันพอดี → เทสชุดนี้ไล่ทุกบรรทัด
VARIANTS = (*KINDS, "vllm-rerank")
_TOKEN = re.compile("LMDS(?:CANARY|DFLTCNRY|EITHCNRY|WORDSCNRY|NUMBCNRY)")
_KIND_OF = {"LMDSCANARY": "dq", "LMDSDFLTCNRY": "default", "LMDSEITHCNRY": "either",
            "LMDSWORDSCNRY": "words", "LMDSNUMBCNRY": "number"}


def _rich(tmp_path: Path, variant: str):
    """bundle ที่เปิดทางแยกของ template ให้มากที่สุด (parser · asset · mmproj/MTP · pin · flag/env) + canary render ของแผนเดียวกัน"""
    kind = variant.split("-")[0]
    if variant == "vllm-rerank":
        report = ModelReport(
            repo_id="Qwen/Qwen3-Reranker-4B", revision_sha="sha-rerank", task="rerank",
            artifact_type=ArtifactType.SAFETENSORS, weight_bytes=8 * GIB, architecture="Qwen3ForCausalLM",
            context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
            tokenizer_files=["tokenizer.json"], safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in SHARDS])
    elif kind == "llamacpp":
        report = _report(kind, gated=True, gguf_variants=[
            GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=12, sha256=None),
            GgufVariant(filename="mmproj-F16.gguf", size_bytes=5, sha256="b" * 64, is_mmproj=True),
            GgufVariant(filename="mtp-draft.gguf", size_bytes=3, sha256=None, is_mtp=True)])
    else:
        report = _report(kind, gated=True, tokenizer_files=["tokenizer.json", "tokenizer_config.json"])
    fit = analyze(report, PRESETS["dgx-spark-stacked" if kind == "stacked" else "dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    if plan.task == "generate":
        plan.tool_calling.enabled, plan.tool_calling.parser = True, "hermes"
        plan.reasoning.enabled, plan.reasoning.parser = True, "qwen3"
        plan.tool_calling.chat_template_override = "tool_chat_template.jinja"
    plan.serving.kv_cache_dtype = "fp8"
    plan.serving.extra_flags = [*plan.serving.extra_flags, "--seed 7", '--chat-template-kwargs {"a":"b c"}']
    plan.serving.extra_env = {"LMDS_TEST": "a b"}
    plan.runtime.image_pin = "sha256:" + "a" * 64
    if kind == "llamacpp":
        plan.multimodal.modalities = ["image", "text"]
        plan.multimodal.projector_files = ["mmproj-F16.gguf"]
        plan.speculative.draft_files = ["mtp-draft.gguf"]
        plan.runtime.native_dir = "/opt/llama builds/b1"
    else:
        plan.runtime_assets = [RuntimeAsset(filename="parser.py", url="https://example.com/p.py?a=1&b=2", sha256="c" * 64)]
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles")
    canary = renderer.render_canary_controller(plan, report, fit, slug=bundle.directory.name)
    return bundle, canary


# ค่าที่ "กลายเป็นโค้ด" สามแบบต่อชนิดของตำแหน่ง: $(…) · backtick · หลุดออกนอก quote ของบริบทนั้น
_INJECTIONS = {
    "dq": ["$(touch PWNED)", "`touch PWNED`", '"; touch PWNED; "'],
    "default": ["$(touch PWNED)", "`touch PWNED`", '"; touch PWNED; "'],
    "either": ["$(touch PWNED)", "`touch PWNED`", "'; touch PWNED; '"],
    "words": ["$(touch PWNED) ", "`touch PWNED` ", "; touch PWNED; "],
    # ตัวเลขอยู่ใน "…" เกือบทุกที่ (ค่าตั้งต้น · ตารางขนาดไฟล์) — หลุดได้ด้วย " เหมือนค่าข้อความ
    "number": ["$(touch PWNED)", "`touch PWNED`", '1"; touch PWNED; "'],
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_gate_catches_a_value_that_became_code_on_every_value_bearing_line(tmp_path, variant):
    """ไล่ทุกบรรทัดที่ canary render บอกว่ามีค่า แทรกค่าร้ายทีละบรรทัด ทีละแบบ — ด่านต้องไม่ผ่าน และต้องชี้บรรทัดนั้น"""
    bundle, canary = _rich(tmp_path, variant)
    original = bundle.controller.read_text(encoding="utf-8")
    lines, canary_lines = original.split("\n"), canary.split("\n")
    assert len(lines) == len(canary_lines)
    sites = [(n, found) for n, line in enumerate(canary_lines) if (found := _TOKEN.search(line))]
    assert len(sites) >= 30, f"canary render ของ {variant} มีบรรทัดที่มีค่าแค่ {len(sites)} — ตัวตั้งของเทสผิด"

    clean = gate_value_expansion(bundle.directory)
    assert clean.passed and f"แล้ว {len(sites)} บรรทัด" in clean.detail, (
        f"bundle ที่ไม่ได้แตะต้องผ่าน และด่านต้องตรวจครบทุกบรรทัดที่มีค่า ({len(sites)}): {clean.detail}")

    missed: list[str] = []
    try:
        for n, first in sites:
            head = canary_lines[n][: first.start()]            # ข้อความของ template ก่อนค่าแรกของบรรทัด — เหมือนกันทั้งสองไฟล์
            assert lines[n].startswith(head)
            for payload in _INJECTIONS[_KIND_OF[first.group(0)]]:
                mutated = list(lines)
                mutated[n] = head + payload + lines[n][len(head):]
                bundle.controller.write_text("\n".join(mutated), encoding="utf-8")
                result = gate_value_expansion(bundle.directory)
                if result.passed or f"{bundle.controller.name}:{n + 1}:" not in result.detail:
                    missed.append(f"บรรทัด {n + 1} + {payload!r} → {result.detail[:100]!r}\n      {lines[n][:100]}")
    finally:
        bundle.controller.write_text(original, encoding="utf-8")
    assert not missed, f"ด่านไม่เห็น {len(missed)} จาก {3 * len(sites)} การแทรก:\n" + "\n".join(missed[:12])


# อักขระที่ encoder ชนิดนั้น "ปล่อยไว้ดิบ ๆ" — ในบริบทจริงของตำแหน่ง ต้องไม่มีตัวไหนมีความหมายต่อ parser ของ bash
_LEFT_RAW = {
    "dq": "a ' ; & | ) ( < > # } b",
    "default": "a ; & | ) ( < > # b",
    "either": "a ; & | ) ( < > # } b",
    "words": "'a ) ; & | ( b' c",
}


def _parses(script: Path, text: str) -> tuple[bool, str]:
    script.write_text(text, encoding="utf-8")
    done = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True, timeout=60)
    return done.returncode == 0, done.stderr.strip()[:160]


@pytest.mark.parametrize("variant", VARIANTS)
def test_bash_parses_every_value_site_the_way_the_renderer_assumed(tmp_path, variant):
    """renderer เดาบริบทของแต่ละตำแหน่งจากบรรทัดเดียวของ template — คนตัดสินว่าเดาถูกไหมคือ bash ไม่ใช่ตัวนับ quote ของเรา

    ทีละตำแหน่ง: ใส่อักขระทุกตัวที่ encoder ของตำแหน่งนั้นปล่อยไว้ไม่ escape แล้วให้ `bash -n` อ่านทั้งไฟล์ · ถ้าตำแหน่งนั้นอยู่ใน
    บริบทอื่นจริง ๆ (ค่าที่ escape แบบ "…" แต่อยู่ใน '…' หรือนอก quote หรือใน $( )) อักขระพวกนี้จะเปลี่ยนโครงของสคริปต์ → parse ไม่ผ่าน
    (heredoc ถูกอ่านเป็นตัวหนังสือตอน parse จึงผ่านเสมอ — ถูกต้อง: ใน heredoc/คอมเมนต์ อักขระชุดนี้ไม่มีความหมายจริง ๆ)
    """
    _, canary = _rich(tmp_path, variant)
    script = tmp_path / "probe.sh"
    ok, error = _parses(script, canary)
    assert ok, f"canary render ต้องเป็น bash ที่ parse ได้ก่อน: {error}"

    wrong: list[str] = []
    probed = 0
    for found in _TOKEN.finditer(canary):
        kind = _KIND_OF[found.group(0)]
        if kind == "number":
            continue
        ok, error = _parses(script, canary[: found.start()] + _LEFT_RAW[kind] + canary[found.end():])
        probed += 1
        if not ok:
            line_no = canary.count("\n", 0, found.start()) + 1
            shown = canary.split("\n")[line_no - 1][:110]
            wrong.append(f"บรรทัด {line_no} (renderer ใช้ encoding แบบ {kind}): {shown}\n      bash: {error}")
    assert probed >= 30
    assert not wrong, ("ตำแหน่งแทรกค่าที่บริบทจริงไม่ตรงกับ encoding ที่ renderer เลือก — ค่าที่มีอักขระพวกนี้จะเปลี่ยนโครงของสคริปต์:\n"
                       + "\n".join(wrong[:10]))


def test_the_bash_probe_itself_notices_a_double_quote_escape_inside_single_quotes(tmp_path):
    """ยืนยันว่าเทสข้างบนจับของจริงได้: บรรทัด `printf ' … lmds set <slug> %s …'` ของ _knob_fix อยู่ใน single quote —
    ถ้า renderer escape ค่าตรงนั้นแบบ "…" (สิ่งที่เกิดก่อนแก้) อักขระ ' ที่ dq() ปล่อยผ่านจะปิด quote แล้ว bash ต้อง parse ไม่ผ่าน"""
    _, canary = _rich(tmp_path, "vllm")
    knob = next(m for m in _TOKEN.finditer(canary) if m.group(0) == "LMDSEITHCNRY")
    line = canary[canary.rfind("\n", 0, knob.start()) + 1: canary.find("\n", knob.start())]
    assert line.lstrip().startswith("printf '") and "lmds set" in line, line
    ok, _ = _parses(tmp_path / "probe.sh", canary[: knob.start()] + _LEFT_RAW["dq"] + canary[knob.end():])
    assert not ok


@pytest.mark.parametrize("line,expected", [
    ('MODEL_ID="', "dq"),
    ('SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-', "default"),
    ('API_KEY_STORE="${LMDS_KEY_ROOT:-${HOME:-}/.lmds/keys}/', "dq"),
    ('SCORE_TEMPLATE="${SCORE_TEMPLATE-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/', "default"),
    ("  printf ' — แก้ค่าที่บันทึกไว้: lmds set ", "either"),
    ("# it's a comment about ", "dq"),
    ("  echo 'closed' ", "dq"),
    ("     (ถ้าตั้งค่าไว้ให้ถาวร: lmds set ", "dq"),
    # รูปของบรรทัดที่ทำให้ตัวนับ quote ของด่านรุ่นแรกเพี้ยน — quote ใน $( ) เริ่มนับใหม่ และ '…' ข้างในมี " จำนวนคี่
    ('  API_IDS="$(printf \'%s\' "$body" | grep -o \'"id":"[^"]*"\' | sed \'s/^"id":"//; s/"$//\' || true)/', "dq"),
    ('  LMDS_REQUIRED_FILES="$(printf \'%s\\n\' ', "dq"),
])
def test_the_renderer_reads_the_quoting_context_from_the_template_line(line, expected):
    assert renderer._site_context(line) == expected


@pytest.mark.parametrize("kind", ["vllm", "sglang", "llamacpp"])
def test_a_hostile_slug_in_the_single_quoted_knob_message_runs_nothing(tmp_path, kind, monkeypatch):
    """ตำแหน่งใหม่จากสาย single: `printf ' … lmds set {{ slug }} %s …'` — รันทางที่พิมพ์ข้อความนั้นจริง (knob ผิดตอน start)"""
    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)
    monkeypatch.setattr(renderer, "check_slug_name", lambda slug, allow_long=False: slug)
    slug = "x$(touch PWNED_slug)`touch PWNED_slug2`"
    report = _report(kind)
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None, engine=Engine.SGLANG if kind == "sglang" else None)
    bundle = render_bundle(plan, report, fit, tmp_path / "bundles", slug=slug)
    cwd = tmp_path / "cwd"

    done = _run(bundle.controller, "start", cwd=cwd, extra_env={"DRY_RUN": "1", "API_PORT": "not-a-port"})

    assert done.returncode != 0 and "lmds set" in done.stderr and "PWNED_slug" in done.stderr, done.stdout + done.stderr
    assert _created(cwd) == [], f"ข้อความแนะนำวิธีแก้ knob รันค่าที่แทรกมา: {_created(cwd)}"


def test_a_newline_in_a_flag_or_env_value_is_refused_like_everywhere_else(tmp_path):
    """ขึ้นบรรทัดใหม่ใน '…' ถูกต้องตาม bash แต่ทำให้ controller มีบรรทัดเกินจาก template — ด่านเทียบไม่ได้ และไม่มีค่าไหนควรมี"""
    def tweak(plan):
        plan.serving.extra_env = {"LMDS_TEST": "a\nb"}

    with pytest.raises(UnsafeValueError):
        _bundle(tmp_path, "stacked", tweak=tweak)


# ───── ตรวจไม่ได้ต้องพูด ไม่ใช่ผ่านเงียบ ๆ ─────
@pytest.mark.parametrize("before,value", [
    # คอมเมนต์หัวไฟล์: ค่าอยู่ท้ายบรรทัด — บรรทัดที่งอกต่อจากมันคือคำสั่งระดับบนสุดของสคริปต์
    ("# Origin:   ", None),
    # ค่ากลางบรรทัด: ข้อความของ template ที่ต้องปิดท้าย (`}"`) ไปโผล่บรรทัดอื่น
    ('SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"', 'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen\ntouch PWNED\n3-32b}"'),
    # heredoc ของ server.meta: บรรทัดที่งอกคือคีย์ปลอมที่ lmds อ่านกลับ
    ("slug=qwen3-32b", "slug=qwen3-32b\ncontroller=/tmp/evil.sh"),
])
def test_a_value_that_brought_its_own_newline_fails_the_gate_instead_of_being_skipped(tmp_path, before, value):
    """ค่าที่มีขึ้นบรรทัดใหม่ทำให้จำนวนบรรทัดไม่เท่า canary — รุ่นแรกตอบ "n/a (ไม่ได้ render จาก template ชุดนี้)" = ผ่าน"""
    bundle = _bundle(tmp_path, "vllm")
    lines = bundle.controller.read_text(encoding="utf-8").split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith(before))
    lines[at] = value if value is not None else lines[at] + "\ntouch PWNED"
    bundle.controller.write_text("\n".join(lines), encoding="utf-8")

    result = gate_value_expansion(bundle.directory)

    assert not result.passed, result.detail
    assert f"{bundle.controller.name}:{at + 1}:" in result.detail, result.detail


def test_hand_edits_outside_value_lines_do_not_stop_the_gate_from_checking_the_values(tmp_path):
    """`lmds validate` ใช้กับ bundle ที่แก้มือได้ — เพิ่ม/แก้บรรทัดที่ไม่มีค่า ต้องไม่ทำให้ด่านตีตก และต้องไม่ทำให้ด่านเลิกตรวจค่า"""
    bundle = _bundle(tmp_path, "vllm")
    lines = bundle.controller.read_text(encoding="utf-8").split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith('API_PORT="${API_PORT:-'))
    lines[at] = 'API_PORT="${API_PORT:-8011}"   # แก้มือ: เครื่องนี้ 8000 ไม่ว่าง'
    lines.insert(at, "# บรรทัดที่ผู้ดูแลเพิ่มเอง")
    lines.insert(at + 40, 'echo "debug: $(date)" >/dev/null')
    bundle.controller.write_text("\n".join(lines), encoding="utf-8")

    edited = gate_value_expansion(bundle.directory)
    assert edited.passed and "n/a" not in edited.detail, edited.detail

    text = "\n".join(lines).replace('  "model-00002-of-00002.safetensors"', '  "model-00002$(touch PWNED).safetensors"', 1)
    bundle.controller.write_text(text, encoding="utf-8")
    caught = gate_value_expansion(bundle.directory)
    assert not caught.passed and "$(" in caught.detail, caught.detail


def _as_older_template(bundle) -> None:
    """ทำให้ bundle อ้าง template_hash อื่น (= render จาก template รุ่นก่อน) และโครงต่างจากชุดปัจจุบันจริง ๆ"""
    profile_path = bundle.directory / "MODEL_PROFILE.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    current = profile["template_hash"]
    profile["template_hash"] = "0123456789ab"
    profile_path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    text = bundle.controller.read_text(encoding="utf-8").replace(current, "0123456789ab")
    # template รุ่นเก่า: บรรทัด RUN_DIR คนละรูปกับปัจจุบัน และมีฟังก์ชันที่รุ่นใหม่ไม่มี
    text = re.sub(r'(?m)^RUN_DIR="\$\{RUN_DIR:-[^\n]*\n', 'RUN_DIR="$HOME/.lmds/run/legacy-layout"\n', text, count=1)
    bundle.controller.write_text(text + "\nlegacy_helper() {\n  :\n}\n", encoding="utf-8")


def test_a_bundle_from_another_template_set_is_checked_where_lines_still_correspond(tmp_path):
    """bundle เก่าบน node (render ก่อนมี escape) คือที่ที่ค่าร้ายอยู่จริง — เดิมด่านข้ามทั้งใบ ("n/a") เพราะจำนวนบรรทัดไม่เท่า"""
    bundle = _bundle(tmp_path, "vllm")
    _as_older_template(bundle)
    aside = gate_value_expansion(bundle.directory)
    assert aside.passed and aside.detail.startswith("n/a") and "ตรวจไม่ได้" in aside.detail, aside.detail

    text = bundle.controller.read_text(encoding="utf-8")
    before = 'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"'
    assert text.count(before) == 1
    bundle.controller.write_text(text.replace(before, before.replace("qwen3-32b", "qwen$(touch PWNED_served_name)")), encoding="utf-8")
    result = gate_value_expansion(bundle.directory)
    assert not result.passed and "$(" in result.detail, result.detail


# ═════════════════════ ค่าที่เดินทางต่อจาก bash ไปถึง Python ที่ฝังอยู่ใน controller ═════════════════════
@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_hostile_values_stay_data_inside_the_embedded_download_python(tmp_path, kind, monkeypatch):
    """`download` ของ single vLLM/SGLang รัน Python ใน container (mount HF cache แบบเขียนได้ · ถือ HF_TOKEN) · ค่าที่ไปถึงมัน:
    MODEL_ID · MODEL_REVISION · รายชื่อ shard · รายชื่อไฟล์ที่ต้องมี — ต้องไปเป็น *ข้อมูล* ทาง env ไม่ใช่ถูกต่อเข้าไปในซอร์ส Python

    เดิม (มีมาก่อนรอบ audit): `repo, rev = '${MODEL_ID}', '${MODEL_REVISION}'` ใน `python3 -c "…"` — bash ขยายตัวแปรก่อน
    Python เห็น revision ที่มี `'` จึงเป็นโค้ด Python · ชั้นเดียวที่กันอยู่คือการปฏิเสธค่าตอน render ซึ่งเทสนี้ปิดไว้เพื่อดูชั้นถัดไป
    เทสรันทั้งเส้น: docker ปลอมรัน Python ที่ controller ฝังมาจริง กับ huggingface_hub ปลอมที่จดว่าถูกเรียกด้วยอะไร
    """
    import glob

    from tests import test_audit3_single_download_filter as real_download
    from tests.single_controller_harness import Box, render, st_report

    monkeypatch.setattr(renderer, "_check_values", lambda plan, context: None)
    monkeypatch.chdir(tmp_path)        # payload เขียนไฟล์ลง cwd ของ process ที่รันมัน — ต้องเป็นที่ที่เทสมองเห็นและเก็บกวาดเอง
    model_id = "Qwen/Qwen3-32B"     # id ที่มี ' ไป render ไม่ผ่านอยู่แล้ว (MODEL_LABEL เป็นค่าตั้งต้นของ ${VAR:-…})
    revision = "sha'+__import__('os').system('touch PWNED_python_rev')+'$(touch PWNED_shell_rev)"
    shard = "model'+__import__('os').system('touch PWNED_python_file')+'`touch PWNED_shell_file`.safetensors"
    required = "tok'$(touch PWNED_shell_required).json"
    report = st_report(repo_id=model_id, revision_sha=revision, tokenizer_files=[required],
                       safetensor_shards=[ShardFile(filename=shard, size_bytes=12)])
    made = Box(tmp_path, kind, render(tmp_path, kind, report=report))
    try:
        env = real_download._prepare(made, [("config.json", 2), (shard, 12), (required, 3), ("other-Q8_0.gguf", 99 * real_download.GB)])
        done = made.run("download", env=env)
        calls = real_download._hub_calls(made)
    finally:
        made.close()

    printed = done.stdout + done.stderr
    assert sorted(p.name for p in tmp_path.rglob("PWNED*")) == [], f"ค่าถูกรันเป็นโค้ด:\n{printed}"
    fetched = next((c for c in calls if c["call"] == "snapshot_download"), None)
    assert fetched is not None, f"Python ที่ฝังมาไปไม่ถึง snapshot_download:\n{printed}"
    assert (fetched["repo"], fetched["revision"]) == (model_id, revision)
    assert {glob.escape(shard), glob.escape(required)} <= set(fetched["allow_patterns"])
    assert glob.escape("other-Q8_0.gguf") not in fetched["allow_patterns"]
