"""`lmds set` / บันทึกจากหน้าเว็บ ต้องแก้เฉพาะค่าที่สั่ง — audit 2026-10-06

หัวไฟล์ `bundle.env` บอกผู้ใช้ว่า "แก้ด้วยมือได้" และ `manager.py` เองก็อ่าน `STARTUP_TIMEOUT` กับ `HF_HOME` /
`WORKER_HF_HOME` จากไฟล์นี้ · แต่ `write()` อ่านกลับได้เฉพาะบรรทัดรูป `NAME="${NAME:-v}"` ของ knob ที่รู้จัก แล้ว
**เขียนไฟล์ใหม่ทั้งไฟล์** — หลัง `lmds set --port 8001`:

    STARTUP_TIMEOUT 6906 → 1800 (ค่าเริ่มต้น) · HF_HOME /data/hf → ว่าง · `API_HOST=…` / `WORKER_HF_HOME=…` หายทั้งบรรทัด

และ `write(dir, {"port": …})` ตรง ๆ — คือสิ่งที่ `PUT /api/models/<slug>/settings` กับ body ไม่ครบ และ `web/deploy.py`
ทำ — ลบ `bundle.args` ทิ้งด้วย ซึ่งเป็นไฟล์ที่ผู้ดูแลเก็บ flag ของ tokenizer/engine ไว้ (คู่มือผู้ดูแลบอกให้ใช้ไฟล์นี้
เพราะรอด `lmds node install`)

ทุกเทสสั่งผ่านทางที่ผู้ใช้ใช้จริง (`lmds set` · PUT /settings · deploy · set --fit · set --auto · ฟอร์มบนหน้าเว็บ) กับ bundle
ที่ render จริง แล้วยืนยันจาก **ไบต์ในไฟล์** และจากสิ่งที่ **bash เห็นเมื่อ source ไฟล์นั้น** (ทางเดียวกับที่ controller อ่าน)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fleet import manager
from lmds.fleet.bundle_settings import ARGS_FILENAME, FILENAME, SOURCE_BLOCK, read, write
from lmds.generator import render_bundle
from tests.test_generator import safetensors_report

SLUG = "qwen3-32b"
# bundle.env แบบที่ผู้ดูแลทิ้งไว้: knob ที่ LMDS เขียนเอง 1 ตัว + ค่าที่ manager.py อ่านจากไฟล์นี้เอง
# (controller_startup_timeout → STARTUP_TIMEOUT · stacked_workers → HF_HOME / WORKER_HF_HOME) + รูป KEY=value ธรรมดา
# + คอมเมนต์ + export
HAND_EDITED = (
    "# สร้างโดย LMDS — ค่าที่ตั้งไว้สำหรับ bundle นี้\n"
    "# แก้ด้วยมือได้ · ลบไฟล์ = กลับไปใช้ค่าของ bundle\n"
    "\n"
    'MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"\n'
    'CTX_SIZE="${CTX_SIZE:-32768}"\n'
    'STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n'
    'HF_HOME="${HF_HOME:-/data/hf}"\n'
    "WORKER_HF_HOME=/data/hf\n"
    "API_HOST=10.2.1.11\n"
    "\n"
    "# โมเดลนี้โหลดนาน (122B) — อย่าลด timeout · 2026-09-28 msi-4\n"
    "export VLLM_LOGGING_LEVEL=DEBUG\n"
)
HAND_ARGS = "--tokenizer /cache/tokfix --kv-cache-memory 20000000000\n"
NAMES = ("API_PORT", "API_HOST", "MAX_MODEL_LEN", "CTX_SIZE", "STARTUP_TIMEOUT", "HF_HOME", "WORKER_HF_HOME",
         "VLLM_LOGGING_LEVEL", "GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS", "TOOL_CALL_PARSER")


@pytest.fixture
def bundle(tmp_path, monkeypatch) -> Path:
    """bundle vLLM ที่ render จริง + ลงทะเบียนจริง + bundle.env/bundle.args ที่ผู้ดูแลแก้ด้วยมือ"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(manager, "_container_running", lambda name: False)
    monkeypatch.setattr(manager, "_orphan_docker", lambda known: [])
    monkeypatch.setattr(manager, "_pgrep_llama", lambda: [])
    report = safetensors_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    rendered = render_bundle(build_plan(report, fit, provider=None), report, fit, tmp_path / "bundles")
    assert rendered.directory.name == SLUG
    manager.register_bundle(rendered.controller)
    (rendered.directory / FILENAME).write_text(HAND_EDITED, encoding="utf-8")
    (rendered.directory / ARGS_FILENAME).write_text(HAND_ARGS, encoding="utf-8")
    return rendered.directory


def _env(bundle: Path) -> str:
    return (bundle / FILENAME).read_text(encoding="utf-8")


def _sourced(bundle: Path) -> dict[str, str]:
    """ค่าที่ controller จะได้จริง — บล็อกเดียวกับที่ controller ใช้ source bundle.env (SOURCE_BLOCK) รันใต้ bash"""
    script = bundle / "probe-env.sh"
    script.write_text(f'BUNDLE_ENV="{bundle / FILENAME}"\n' + SOURCE_BLOCK
                      + "".join(f'echo "{name}=${{{name}-<unset>}}"\n' for name in NAMES), encoding="utf-8")
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    script.unlink()
    assert done.returncode == 0, done.stderr
    return dict(line.split("=", 1) for line in done.stdout.splitlines())


def _controller(bundle: Path) -> Path:
    return next(bundle.glob("*-single.sh"))


def _set(*args: str):
    from lmds.cli.main import app

    return CliRunner().invoke(app, ["set", SLUG, *args], env={"COLUMNS": "200"})


# ═════════════════════ CLI: lmds set ═════════════════════
def test_cli_set_port_adds_one_line_and_keeps_every_other_byte(bundle):
    """รีโปรของ auditor: `lmds set --port 8001` บนไฟล์ที่ผู้ดูแลแก้ไว้"""
    assert manager.controller_startup_timeout(_controller(bundle)) == 6906
    result = _set("--port", "8001")
    assert result.exit_code == 0, result.output

    after = _env(bundle)
    port_line = 'API_PORT="${API_PORT:-8001}"\n'
    assert after.count(port_line) == 1
    assert after.replace(port_line, "") == HAND_EDITED, "นอกจากบรรทัด port ทุกไบต์ต้องเหมือนเดิม"
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS
    # ผู้อ่านจริงของไฟล์นี้ยังได้ค่าเดิม: manager.py (systemd unit / watchdog / แผนลบ stacked) และ bash ของ controller
    assert manager.controller_startup_timeout(_controller(bundle)) == 6906
    assert manager._bundle_env_value(bundle, "HF_HOME") == "/data/hf"
    assert manager._bundle_env_value(bundle, "WORKER_HF_HOME") == "/data/hf"
    seen = _sourced(bundle)
    assert seen["API_PORT"] == "8001" and seen["STARTUP_TIMEOUT"] == "6906" and seen["HF_HOME"] == "/data/hf"
    assert seen["WORKER_HF_HOME"] == "/data/hf" and seen["API_HOST"] == "10.2.1.11"
    assert seen["VLLM_LOGGING_LEVEL"] == "DEBUG" and seen["MAX_MODEL_LEN"] == "32768"


def test_cli_set_replaces_a_value_in_place_whatever_form_it_was_written_in(bundle):
    """ค่าที่ผู้ดูแลเขียนเองแบบ `API_HOST=…` ธรรมดา: ตั้งทับต้องแทนบรรทัดนั้น — ต่อท้ายเป็น `${API_HOST:-…}` จะไม่มีผลเลย
    เพราะบรรทัดเดิมตั้งค่าไปก่อนแล้ว (ค่าใหม่ถูกบันทึก 'สำเร็จ' แต่ controller ยังได้ค่าเก่า)"""
    result = _set("--bind", "127.0.0.1")
    assert result.exit_code == 0, result.output
    after = _env(bundle)
    assert after == HAND_EDITED.replace("API_HOST=10.2.1.11\n", 'API_HOST="${API_HOST:-127.0.0.1}"\n')
    assert _sourced(bundle)["API_HOST"] == "127.0.0.1"

    # knob ที่อยู่ในไฟล์แล้ว (รูปที่ LMDS เขียนเอง): แก้ที่เดิม ไม่ย้ายไปท้ายไฟล์ ไม่ซ้ำ
    result = _set("--context", "16384")
    assert result.exit_code == 0, result.output
    assert _env(bundle) == after.replace("32768", "16384")
    assert _sourced(bundle)["MAX_MODEL_LEN"] == "16384" and _sourced(bundle)["CTX_SIZE"] == "16384"


def test_cli_set_without_arguments_shows_hand_written_values_too(bundle):
    """`lmds set <slug>` เปล่า ๆ คือ "ตอนนี้ตั้งอะไรไว้" — ค่าที่เขียนด้วยมือ (API_HOST=…) มีผลจริงตอน start จึงต้องเห็นด้วย"""
    saved = read(bundle)
    assert saved["bind"] == "10.2.1.11" and saved["context"] == "32768"
    assert saved["extra_args"] == HAND_ARGS.strip()
    result = _set()
    assert result.exit_code == 0 and "10.2.1.11" in result.output and "32768" in result.output
    assert _env(bundle) == HAND_EDITED, "การดูค่าต้องไม่เขียนอะไร"


def test_cli_extra_args_is_only_touched_when_named_and_cleared_only_by_an_explicit_empty_value(bundle):
    _set("--port", "8001", "--slots", "2")
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS
    done = _set("--extra-args", "--tokenizer /cache/tokfix")
    assert done.exit_code == 0, done.output
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == "--tokenizer /cache/tokfix\n"
    cleared = _set("--extra-args", "")
    assert cleared.exit_code == 0, cleared.output
    assert not (bundle / ARGS_FILENAME).exists()
    assert "extra_args" not in read(bundle) and read(bundle)["port"] == "8001"


def test_cli_clear_removes_what_lmds_set_wrote_and_nothing_else(bundle):
    """`--clear` = กลับไปใช้ค่าของ bundle สำหรับ knob ที่ `lmds set` ดูแล — บรรทัดที่ผู้ดูแลเพิ่มเอง (STARTUP_TIMEOUT · HF_HOME ·
    export …) กับ bundle.args ไม่ใช่ของที่ --clear เป็นเจ้าของ"""
    _set("--port", "8001", "--tool-parser", "qwen3_xml")
    result = _set("--clear")
    assert result.exit_code == 0, result.output
    after = _env(bundle)
    for gone in ("API_PORT", "API_HOST", "MAX_MODEL_LEN", "CTX_SIZE", "TOOL_CALL_PARSER"):
        assert gone not in after
    for kept in ('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n', 'HF_HOME="${HF_HOME:-/data/hf}"\n',
                 "WORKER_HF_HOME=/data/hf\n", "# โมเดลนี้โหลดนาน (122B) — อย่าลด timeout · 2026-09-28 msi-4\n",
                 "export VLLM_LOGGING_LEVEL=DEBUG\n"):
        assert kept in after
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS
    assert read(bundle) == {"extra_args": HAND_ARGS.strip()}
    seen = _sourced(bundle)
    assert seen["API_PORT"] == "<unset>" and seen["STARTUP_TIMEOUT"] == "6906"
    # บอกผู้ใช้ว่ามีอะไรเหลือและทำไม ไม่ใช่ "ลบทั้งหมดแล้ว"
    assert "bundle.args" in result.output and "STARTUP_TIMEOUT" in result.output


def test_clearing_a_file_that_only_held_lmds_values_still_removes_the_file(tmp_path):
    """ไฟล์ที่มีแต่ของ LMDS เอง: ล้างแล้วไม่ทิ้งไฟล์เปล่าที่มีแต่หัวคอมเมนต์ (หัวไฟล์บอกว่า "ลบไฟล์ = กลับไปใช้ค่าของ bundle")"""
    from lmds.fleet.bundle_settings import clear

    write(tmp_path, {"port": 8001, "context": 131072})
    assert (tmp_path / FILENAME).exists()
    assert clear(tmp_path) == {}
    assert not (tmp_path / FILENAME).exists()


def test_a_write_that_names_no_known_setting_changes_nothing(bundle):
    """dict ว่าง / มีแต่คีย์ที่ไม่รู้จัก = ไม่มีอะไรให้ตั้ง — ต้องไม่กลายเป็น "ล้างทุกอย่าง" โดยบังเอิญ"""
    assert write(bundle, {}) == read(bundle)
    write(bundle, {"api_key": "sk-do-not-store", "nonsense": 1})
    assert _env(bundle) == HAND_EDITED and (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS


def test_a_refused_value_leaves_both_files_exactly_as_they_were(bundle):
    result = _set("--port", "8001", "--context", "99999999")
    assert result.exit_code == 1 and "เกินเพดาน" in result.output
    assert _env(bundle) == HAND_EDITED and (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS


def test_cli_set_auto_fills_suggestions_without_dropping_hand_written_lines(bundle):
    """`--auto` (ค่าที่ระบบเสนอ) เดินทางเดียวกัน — ไม่ว่ามันเสนออะไร บรรทัดของผู้ดูแลต้องอยู่ครบ"""
    result = _set("--auto", "--port", "8001")
    assert result.exit_code == 0, result.output
    after = _env(bundle)
    for kept in ('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n', 'HF_HOME="${HF_HOME:-/data/hf}"\n',
                 "WORKER_HF_HOME=/data/hf\n", "API_HOST=10.2.1.11\n", "export VLLM_LOGGING_LEVEL=DEBUG\n"):
        assert kept in after
    assert 'API_PORT="${API_PORT:-8001}"\n' in after
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS


# ═════════════════════ set --fit / ปุ่ม Fit ═════════════════════
def _fit_ready(bundle: Path, monkeypatch) -> None:
    host = {"memory_model": "unified", "ram_total_gb": 121.0, "ram_used_gb": 6.0, "foreign": [],
            "gpus": [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": 0.0}]}
    models = [{"slug": SLUG, "running": False, "engine": "vllm"}]
    monkeypatch.setattr("lmds.fleet.sizing.local_facts", lambda: (host, models))
    # ปุ่ม Fit บนหน้าเว็บอ่าน host/models จากแคชของ hub ไม่ใช่จาก local_facts
    from lmds.web import state

    state.STORE.set_local({"host": host, "models": models})
    monkeypatch.setattr("lmds.fleet.sizing.measured_for", lambda server: {})


def test_set_fit_writes_its_own_keys_and_keeps_the_rest(bundle, monkeypatch):
    """`lmds set --fit` เขียน slots/context/gpu-util + แทน --kv-cache-memory ใน bundle.args — flag อื่นใน bundle.args
    (tokenizer) และบรรทัดของผู้ดูแลใน bundle.env ต้องอยู่"""
    _fit_ready(bundle, monkeypatch)
    result = _set("--fit", "--slots", "2", "--json")
    assert result.exit_code == 0, result.output
    saved = json.loads(result.output)["saved"]
    assert saved["slots"] == "2"
    args = (bundle / ARGS_FILENAME).read_text(encoding="utf-8")
    assert "--tokenizer /cache/tokfix" in args and args.count("--kv-cache-memory") == 1
    after = _env(bundle)
    for kept in ('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n', 'HF_HOME="${HF_HOME:-/data/hf}"\n',
                 "WORKER_HF_HOME=/data/hf\n", "API_HOST=10.2.1.11\n", "export VLLM_LOGGING_LEVEL=DEBUG\n"):
        assert kept in after
    assert _sourced(bundle)["MAX_NUM_SEQS"] == "2" and _sourced(bundle)["STARTUP_TIMEOUT"] == "6906"


def test_the_fit_button_route_keeps_the_rest_too(bundle, monkeypatch):
    from fastapi.testclient import TestClient

    from lmds.web.api import create_app

    _fit_ready(bundle, monkeypatch)
    done = TestClient(create_app()).post(f"/api/models/{SLUG}/fit", json={"slots": 2, "apply": True})
    assert done.status_code == 200, done.text
    assert done.json()["applied"] is True
    assert "--tokenizer /cache/tokfix" in (bundle / ARGS_FILENAME).read_text(encoding="utf-8")
    assert "WORKER_HF_HOME=/data/hf\n" in _env(bundle) and _sourced(bundle)["STARTUP_TIMEOUT"] == "6906"


# ═════════════════════ หน้าเว็บ: PUT /api/models/<slug>/settings ═════════════════════
@pytest.fixture
def web(bundle):
    from fastapi.testclient import TestClient

    from lmds.web.api import create_app

    return TestClient(create_app())


def test_web_put_with_a_partial_body_keeps_bundle_args_and_every_other_line(bundle, web):
    """รีโปรของ auditor ข้อสอง: ฟอร์มส่งเฉพาะช่องที่เปลี่ยน — เดิมลบ bundle.args และทุกบรรทัดที่ไม่ได้ส่งมา"""
    done = web.put(f"/api/models/{SLUG}/settings", json={"port": 8002})
    assert done.status_code == 200, done.text
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS
    assert _env(bundle).replace('API_PORT="${API_PORT:-8002}"\n', "") == HAND_EDITED
    # คำตอบคือสภาพทั้งหมดหลังบันทึก (หน้าเว็บ/ป้าย restart-to-apply ใช้) ไม่ใช่แค่คีย์ที่ส่งมา
    saved = done.json()["saved"]
    assert saved["port"] == "8002" and saved["bind"] == "10.2.1.11" and saved["extra_args"] == HAND_ARGS.strip()
    assert web.get(f"/api/models/{SLUG}/settings").json()["saved"] == saved


def test_web_put_blank_value_removes_only_that_setting(bundle, web):
    web.put(f"/api/models/{SLUG}/settings", json={"port": 8002})
    done = web.put(f"/api/models/{SLUG}/settings", json={"port": "", "bind": None})
    assert done.status_code == 200, done.text
    assert _env(bundle) == HAND_EDITED.replace("API_HOST=10.2.1.11\n", "")
    assert "port" not in done.json()["saved"] and "bind" not in done.json()["saved"]
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS


def test_web_put_sets_and_clears_extra_args_only_when_the_body_names_it(bundle, web):
    web.put(f"/api/models/{SLUG}/settings", json={"extra_args": "--tokenizer /cache/tokfix --foo 1"})
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == "--tokenizer /cache/tokfix --foo 1\n"
    assert _env(bundle) == HAND_EDITED
    web.put(f"/api/models/{SLUG}/settings", json={"extra_args": ""})
    assert not (bundle / ARGS_FILENAME).exists() and _env(bundle) == HAND_EDITED


def test_web_put_with_an_empty_body_is_the_clear_button_and_spares_what_it_does_not_own(bundle, web):
    """ปุ่ม Clear ของฟอร์มส่ง body ว่าง — ล้าง knob ของ LMDS เหมือน `lmds set --clear` · ไม่ลบ bundle.args / บรรทัดของผู้ดูแล"""
    web.put(f"/api/models/{SLUG}/settings", json={"port": 8002})
    done = web.put(f"/api/models/{SLUG}/settings", json={})
    assert done.status_code == 200, done.text
    assert done.json()["saved"] == {"extra_args": HAND_ARGS.strip()}
    after = _env(bundle)
    assert "API_PORT" not in after and "API_HOST" not in after and "MAX_MODEL_LEN" not in after
    assert 'STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n' in after and "WORKER_HF_HOME=/data/hf\n" in after
    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS


def test_the_web_form_sends_its_own_blank_fields_and_nothing_it_does_not_show(tmp_path):
    """ฟอร์ม Manage บนการ์ด: ช่องที่ผู้ใช้ลบจนว่าง = "เอาค่านั้นออก" ต้องถูกส่งเป็นค่าว่าง (เดิมไม่ส่งเลย แล้วพึ่งการเขียนทับทั้งไฟล์
    ให้มันหายไปเอง) · ช่องที่ฟอร์มนี้ไม่มี (parser · engine_env · extra_args) ต้องไม่อยู่ใน body — ไม่ส่ง = ไม่แตะ
    รัน handler จริงของ index.html ใน node แล้วดู body ของ PUT ที่มันยิงออกไป"""
    from tests.test_console_shell import FLEET, run_scenario

    prelude = FLEET + """
        fx.localModels = [{ slug: "qwen3-32b", model: "qwen3-32b", model_id: "Qwen/Qwen3-32B", running: false, healthy: false,
                            engine: "vllm", port: 8001, context: 32768, features: "text", controller_exists: true, registered: true }];
        H.routes = [["/api/models/qwen3-32b/settings", (url, opts) => ({ slug: "qwen3-32b", saved: {} })],
                    ...H.defaultRoutes(fx)];"""
    (out,) = run_scenario(tmp_path, prelude, """
        await H.tick(10);
        document.querySelector('button[data-act="opts"][data-slug="qwen3-32b"]').click(); await H.tick(10);
        const panel = document.getElementById("panel-qwen3-32b");
        panel.querySelector(".o-port").value = "8005";
        panel.querySelector(".o-ctx").value = "";
        panel.querySelector(".o-bind").value = "";
        document.querySelector('button[data-act="saveopts"][data-slug="qwen3-32b"]').click(); await H.tick(10);
        const put = H.calls.filter(c => c.method === "PUT" && c.url.endsWith("/settings")).pop();
        console.log(JSON.stringify({ body: JSON.parse(put.body), errors: H.errors }));""")
    assert out["errors"] == []
    body = out["body"]
    assert body["port"] == 8005
    assert body["context"] == "" and body["bind"] == "", "ช่องที่ว่างต้องถูกส่งเป็นค่าว่าง = เอาออก"
    assert "api_key" not in body
    assert not {"extra_args", "engine_env", "tool_parser", "reasoning_parser", "gpu_util"} & set(body)


# ═════════════════════ deploy ทับโฟลเดอร์เดิม ═════════════════════
def test_web_deploy_into_an_existing_bundle_keeps_bundle_args_and_hand_written_lines(bundle, monkeypatch):
    """`web/deploy.py` เขียนพอร์ตที่ wizard เลือกด้วย `write(dir, {"port": …})` — deploy โมเดลเดิมซ้ำ (render ทับโฟลเดอร์เดิม)
    เคยลบ bundle.args ของผู้ดูแลทิ้ง"""
    from lmds.web import deploy as dep
    from lmds.web import state

    state.STORE.set_node("spark2", {"host": {"gpus": [{"name": "NVIDIA GB10", "vram_gb": 128}], "memory_model": "unified"},
                                    "models": [{"slug": "m8000", "port": 8000, "running": True}],
                                    "summary": {"total": 1, "running": 1}})
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: safetensors_report())
    analyzed = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True, machine="spark2")
    assert analyzed["plan"]["port"] == 8001
    result = dep.generate(analyzed["id"], context=16384, output=str(bundle.parent))
    assert Path(result["directory"]) == bundle and result["port"] == 8001

    assert (bundle / ARGS_FILENAME).read_text(encoding="utf-8") == HAND_ARGS
    assert _env(bundle).replace('API_PORT="${API_PORT:-8001}"\n', "") == HAND_EDITED
    assert _sourced(bundle)["API_PORT"] == "8001" and _sourced(bundle)["STARTUP_TIMEOUT"] == "6906"
