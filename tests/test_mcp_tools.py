"""เครื่องมือของ `lmds mcp` คืนก้อนเดียวกับ `--json` ของ CLI บนฟลีตเดียวกัน — ที่มาเดียว ไม่มีการคำนวณซ้ำ

ทุกเทส: เปิด `lmds mcp` ตัวจริงเป็น process ลูก เรียกเครื่องมือด้วย JSON-RPC ทาง stdio แล้วเทียบกับ stdout ของ
`lmds … --json` ที่รันเป็น process ลูกอีกตัวบนฟลีตจำลองเดียวกัน (tests/mcp_fleet.py) · เครื่องอื่น = `ssh` ปลอมที่จดคำสั่ง
ที่ได้รับ — เทียบทั้งคำตอบและคำสั่งที่ hub ส่งไปจริง
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

import lmds
from tests import mcp_fleet
from tests.mcp_fleet import (
    DOCKER_SLUG,
    GHOST_SLUG,
    HOSTS,
    NATIVE_SLUG,
    NODE_DOCTOR,
    NODE_DOWN,
    NODE_FIT,
    NODE_OK,
    NODE_WATCHDOG,
    PLAIN_REPO,
    PLANTED,
    VLLM_LOG,
    node_agent_info,
)

MASK = "[REDACTED]"
RO = "LMDS_READ_ONLY=1 "          # หน้าคำสั่งที่ hub ส่งไปเครื่องอื่น — lmds ปลายทางผนึกตัวเอง
MLX_REPO = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    built = mcp_fleet.build(tmp_path, monkeypatch)
    yield built
    built.close()


@pytest.fixture
def client(fleet):
    mcp = fleet.mcp(fake_hub=True)
    mcp.initialize()
    return mcp


def cli_json(fleet, *args, fake_hub=False, ok=(0,)):
    done = fleet.lmds(*args, fake_hub=fake_hub)
    assert done.returncode in ok, f"lmds {' '.join(args)} → exit {done.returncode}\n{done.stderr[-2000:]}"
    return json.loads(done.stdout)


def without_secrets(text: str) -> str:
    """ข้อความเดียวกันโดยค่าที่ปลูกไว้ทุกตัวถูกแทนด้วยตัวปิด — รูปที่คำตอบของเครื่องมือควรเป็น"""
    for value in PLANTED.values():
        text = text.replace(value, MASK)
    return text


# ── ตัวตนของ hub · ทะเบียนเครื่อง ──────────────────────────────────────────────────────────────────
def test_version_is_what_the_cli_and_the_fleet_report_say(fleet, client):
    answer = client.call("lmds_version")
    assert answer["error"] is False
    payload = answer["payload"]
    hub = cli_json(fleet, "fleet", "check", "--json", ok=(0, 1))["hub"]
    assert {key: payload[key] for key in ("version", "commit", "template_hash", "dirty")} == \
        {key: hub[key] for key in ("version", "commit", "template_hash", "dirty")}
    assert payload["version"] == lmds.__version__ and payload["template_standard"] == lmds.TEMPLATE_STANDARD
    said = fleet.lmds("version").stdout
    assert payload["version"] in said and payload["template_standard"] in said
    assert payload["commit"] and payload["commit"] in said
    assert fleet.calls("ssh") == []


def test_nodes_are_the_registry_in_full_with_untruncated_names(fleet, client):
    payload = client.call("lmds_nodes")["payload"]
    on_disk = yaml.safe_load((fleet.config / "nodes.yaml").read_text(encoding="utf-8"))["nodes"]
    assert payload == {"nodes": on_disk}
    first = payload["nodes"][0]
    assert first["name"] == NODE_OK and first["site"] == "bangkok-dc1" and first["host"] == HOSTS[NODE_OK]
    assert first["alt_hosts"] == ["100.64.0.1"] and first["local_ip"] == "10.9.0.1"
    assert (first["lmds_version"], first["lmds_commit"], first["last_seen"]) == ("0.9.4", "0ad1a59e", "2026-10-01 08:15")
    assert (first["controllers_stale"], first["runtime_stale"], first["restart_pending"]) == (1, 0, 1)
    assert "last_error" in first and "password" not in first
    # ตารางของ `lmds node list` คือสิ่งที่ผู้ช่วยเคยต้องแกะ — ที่ความกว้างจอปกติชื่อนี้ไม่รอด
    table = fleet.lmds("node", "list", env={"COLUMNS": "80"}).stdout
    assert NODE_OK not in table, "ถ้าตารางไม่ตัดชื่อแล้ว เหตุผลของเครื่องมือนี้ต้องเขียนใหม่"
    assert fleet.calls("ssh") == []


def test_node_order_follows_what_the_owner_arranged(fleet, client):
    (fleet.config / "config.yaml").write_text(f"ui:\n  node_order: [{NODE_DOWN}, '{NODE_OK}']\n", encoding="utf-8")
    assert [node["name"] for node in client.call("lmds_nodes")["payload"]["nodes"]] == [NODE_DOWN, NODE_OK]


# ── โมเดลบน hub / บนเครื่องอื่น ───────────────────────────────────────────────────────────────────
def test_models_on_the_hub_equal_agent_info(fleet, client):
    answer = client.call("lmds_models")
    assert answer["error"] is False and answer["notes"] == []
    payload = answer["payload"]
    cli = cli_json(fleet, "agent", "info")          # หลังเครื่องมือ: `agent info` เขียน usage.samples และลบทะเบียนผี
    assert payload["models"] == cli["models"] and payload["summary"] == cli["summary"]
    assert set(payload["host"]) == set(cli["host"])
    for key in ("hostname", "lmds_version", "lmds_commit", "template_hash", "role", "runtimes"):
        assert payload["host"][key] == cli["host"][key], key

    by_slug = {model["slug"]: model for model in payload["models"]}
    assert set(by_slug) == {DOCKER_SLUG, NATIVE_SLUG}, "ทะเบียนผีไม่ใช่โมเดล — ทั้งสองทางต้องข้ามมัน"
    running, stopped = by_slug[NATIVE_SLUG], by_slug[DOCKER_SLUG]
    assert (running["engine"], running["mode"], running["running"], running["healthy"]) == ("llamacpp", "native", True, True)
    assert running["port"] == fleet.engine.server_address[1] and running["usage"]["requests_24h"] == 12
    assert (stopped["engine"], stopped["running"], stopped["port"]) == ("vllm", False, 8000)
    assert stopped["context"] and stopped["autostart"] and stopped["controller"]["state"] == "ok"
    assert payload["summary"]["total"] == 2 and payload["summary"]["running"] == 1


def test_models_on_a_node_are_what_that_node_reports(fleet, client):
    answer = client.call("lmds_models", {"node": NODE_OK})
    expected = json.loads(without_secrets(json.dumps(node_agent_info(), ensure_ascii=False)))
    assert answer["error"] is False and answer["payload"] == expected
    assert fleet.remote_commands() == [RO + "lmds agent info"]
    ssh = fleet.calls("ssh")[0]
    assert ssh[-2] == f"tkc@{HOSTS[NODE_OK]}" and "BatchMode=yes" in ssh


def test_an_unreachable_node_is_an_error_for_that_node_not_for_the_server(fleet, client):
    for name, arguments in (("lmds_models", {"node": NODE_DOWN}), ("lmds_fit", {"slug": "gemma-3-27b", "node": NODE_DOWN}),
                            ("lmds_logs", {"slug": "gemma-3-27b", "node": NODE_DOWN}),
                            ("lmds_doctor", {"slug": "gemma-3-27b", "node": NODE_DOWN}),
                            ("lmds_watchdog_status", {"node": NODE_DOWN})):
        answer = client.call(name, arguments)
        assert answer["error"] is True, name
        assert "Connection timed out" in answer["payload"]["error"], (name, answer["payload"])
    # server ยังตอบเครื่องอื่นตามปกติ
    assert client.call("lmds_models", {"node": NODE_OK})["error"] is False


# ── inspect · plan (Hugging Face ปลอมแบบเดียวกับ test_mlx_checkpoint) ─────────────────────────────────
def test_inspect_equals_the_cli_json(fleet, client):
    answer = client.call("lmds_inspect", {"model": PLAIN_REPO, "targets": ["dgx-spark-single", "rtx-4090"],
                                          "concurrency": 2, "context": 32768, "kv_dtype": "fp8"})
    assert answer["error"] is False
    cli = cli_json(fleet, "inspect", PLAIN_REPO, "--target", "dgx-spark-single", "--target", "rtx-4090",
                   "--concurrency", "2", "--context", "32768", "--kv-dtype", "fp8", "--json", fake_hub=True)
    assert answer["payload"] == cli
    assert cli["model"]["repo_id"] == PLAIN_REPO and cli["model"]["unsupported_format"] is None
    assert [fit["target_name"] for fit in cli["fit"]] == ["dgx-spark-single", "rtx-4090"]
    assert cli["context_advice"] and cli["context_advice"][0]["kv_dtype"] == "fp8"


def test_inspect_carries_the_unsupported_format_verdict(fleet, client):
    answer = client.call("lmds_inspect", {"model": MLX_REPO, "targets": ["dgx-spark-single"]})
    cli = cli_json(fleet, "inspect", MLX_REPO, "--target", "dgx-spark-single", "--json", fake_hub=True)
    assert answer["error"] is False and answer["payload"] == cli
    assert answer["payload"]["model"]["unsupported_format"] == "mlx"
    assert [fit["verdict"] for fit in answer["payload"]["fit"]] == ["unsupported"]


def test_inspect_of_a_repo_that_does_not_exist_is_an_error_with_the_cli_reason(fleet, client):
    answer = client.call("lmds_inspect", {"model": mcp_fleet.MISSING_REPO})
    done = fleet.lmds("inspect", mcp_fleet.MISSING_REPO, "--json", fake_hub=True)
    assert answer["error"] is True and answer["payload"]["exit_code"] == done.returncode != 0
    assert mcp_fleet.MISSING_REPO in answer["payload"]["error"]
    assert PLANTED["hf_token"] not in answer["text"]


class _Llm(BaseHTTPRequestHandler):
    """provider ปลอมที่จดว่าถูกเรียกไหม — ตอบ 500 เสมอ (CLI ถอยไป rule-based เอง)"""

    asked: list[str] = []

    def do_POST(self):  # noqa: N802 — ชื่อตาม http.server
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).asked.append(self.path)
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def test_plan_equals_the_rule_based_cli_plan_and_never_calls_an_llm(fleet):
    """provider ถูกตั้งไว้และมี key — `lmds plan` (ไม่ใส่ --no-llm) เรียก LLM จริง · เครื่องมือต้องไม่เรียกเลย"""
    _Llm.asked = []
    llm = ThreadingHTTPServer(("127.0.0.1", 0), _Llm)
    threading.Thread(target=llm.serve_forever, daemon=True).start()
    try:
        (fleet.config / "config.yaml").write_text(
            "provider:\n  name: openai-compat\n  model: planner-test\n"
            f"  base_url: http://127.0.0.1:{llm.server_address[1]}/v1\n", encoding="utf-8")
        llm_env = {"LMDS_OPENAI_COMPAT_API_KEY": "planted-llm-key-0001"}
        with_llm = fleet.mcp(fake_hub=True, env=llm_env)
        with_llm.initialize()
        answer = with_llm.call("lmds_plan", {"model": PLAIN_REPO, "target": "dgx-spark-single", "concurrency": 2})
        assert answer["error"] is False, answer["payload"]
        forced = with_llm.call("lmds_plan", {"model": PLAIN_REPO, "target": "dgx-spark-single", "engine": "sglang"})
        assert _Llm.asked == [], "เครื่องมือเรียก LLM"

        cli = cli_json(fleet, "plan", PLAIN_REPO, "--target", "dgx-spark-single", "--concurrency", "2", "--no-llm",
                       "--json", fake_hub=True)
        assert answer["payload"] == cli
        assert cli["generator"] == "rule-based" and cli["runtime"]["engine"] == "vllm"
        assert forced["payload"]["runtime"]["engine"] == "sglang"
        assert forced["payload"] == cli_json(fleet, "plan", PLAIN_REPO, "--target", "dgx-spark-single", "--engine",
                                             "sglang", "--no-llm", "--json", fake_hub=True)
        assert _Llm.asked == []

        # ตัวพิสูจน์ว่า provider ที่ตั้งไว้ใช้งานได้จริง: CLI แบบไม่ใส่ --no-llm ไปถามมัน
        fleet.lmds("plan", PLAIN_REPO, "--target", "dgx-spark-single", "--json", fake_hub=True, env=llm_env)
        assert _Llm.asked, "ถ้า CLI ไม่เรียก provider นี้ เทสข้างบนไม่ได้พิสูจน์ว่าเครื่องมือเลี่ยง LLM"
    finally:
        llm.shutdown()
        llm.server_close()


def test_plan_refusals_come_back_as_errors_with_the_reason(fleet, client):
    mlx = client.call("lmds_plan", {"model": MLX_REPO, "target": "dgx-spark-single"})
    assert mlx["error"] is True and mlx["payload"]["exit_code"] == 1 and "MLX" in mlx["payload"]["error"]
    assert fleet.lmds("plan", MLX_REPO, "--target", "dgx-spark-single", "--no-llm", "--json", fake_hub=True).returncode == 1

    too_big = client.call("lmds_plan", {"model": PLAIN_REPO, "target": "rtx-4090"})
    done = fleet.lmds("plan", PLAIN_REPO, "--target", "rtx-4090", "--no-llm", "--json", fake_hub=True)
    assert done.returncode == 3 and done.stdout == ""
    assert too_big["error"] is True and too_big["payload"]["exit_code"] == 3
    assert "rtx-4090" in too_big["payload"]["error"]


# ── fit ───────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("arguments,flags", [
    ({}, []),
    ({"slots": 4, "context": 16384}, ["--slots", "4", "--context", "16384"]),
])
def test_fit_on_the_hub_equals_the_cli_json(fleet, client, arguments, flags):
    answer = client.call("lmds_fit", {"slug": DOCKER_SLUG, **arguments})
    cli = cli_json(fleet, "fit", DOCKER_SLUG, *flags, "--json")
    assert answer["error"] is False, answer["payload"]
    assert answer["payload"] == json.loads(without_secrets(json.dumps(cli, ensure_ascii=False)))
    assert answer["payload"]["slug"] == DOCKER_SLUG and "verdict" in answer["payload"]
    # ค่าที่ตั้งอยู่มี --api-key ใน bundle.args — ต้องถูกปิด และ fit ต้องไม่เขียนอะไรลง bundle
    assert answer["payload"]["current"]["extra_args"] == f"--api-key {MASK}"
    assert not (fleet.bundles / DOCKER_SLUG / "bundle.env").exists()


def test_fit_on_a_node_runs_the_dry_run_command_there(fleet, client):
    answer = client.call("lmds_fit", {"slug": "gemma-3-27b", "node": NODE_OK, "slots": 2, "context": 32768})
    assert answer["error"] is False
    assert answer["payload"] == json.loads(without_secrets(json.dumps(NODE_FIT, ensure_ascii=False)))
    assert fleet.remote_commands() == [RO + "lmds fit gemma-3-27b --json --slots 2 --context 32768"]


def test_fit_of_an_unknown_bundle_is_an_error(fleet, client):
    answer = client.call("lmds_fit", {"slug": "no-such-bundle"})
    assert answer["error"] is True and "no-such-bundle" in answer["payload"]["error"]


# ── fleet check ───────────────────────────────────────────────────────────────────────────────────
def test_fleet_check_equals_the_cli_json(fleet, client):
    answer = client.call("lmds_fleet_check")
    cli = cli_json(fleet, "fleet", "check", "--json", ok=(0, 1))
    assert answer["error"] is False and answer["payload"] == cli
    assert [node["name"] for node in cli["nodes"]] == [NODE_OK, NODE_DOWN]
    assert all(node["source"] == "registry" for node in cli["nodes"])
    assert fleet.calls("ssh") == [], "ไม่ใส่ check = ไม่ SSH"


def test_fleet_check_live_matches_the_cli_but_leaves_the_registry_alone(fleet, client):
    registry = fleet.config / "nodes.yaml"
    before = registry.read_bytes()
    answer = client.call("lmds_fleet_check", {"check": True})
    assert answer["error"] is False
    assert registry.read_bytes() == before, "เครื่องมือ probe แล้วต้องไม่เขียนทะเบียน"
    assert sorted(fleet.remote_commands()) == [RO + "lmds agent info", RO + "lmds agent info"]

    cli = cli_json(fleet, "fleet", "check", "--check", "--json", ok=(0, 1))
    assert registry.read_bytes() != before, "`--check` ของ CLI เขียนทะเบียน — ถ้าไม่เขียนแล้ว เทสนี้ไม่ได้พิสูจน์อะไร"

    def comparable(report: dict) -> dict:
        # last_seen คือนาทีที่ probe — สอง process รันคนละเสี้ยววินาที อาจคร่อมนาที
        return json.loads(json.dumps(report), object_hook=lambda d: {k: v for k, v in d.items() if k != "last_seen"})

    assert comparable(answer["payload"]) == comparable(cli)
    nodes = {node["name"]: node for node in answer["payload"]["nodes"]}
    ok, down = nodes[NODE_OK], nodes[NODE_DOWN]
    assert ok["source"] == "cache" and ok["reachable"] is True and ok["controllers"]["state"] == "stale"
    assert down["reachable"] is False and "Connection timed out" in down["error"]
    assert answer["payload"]["summary"]["total"] == 2


# ── watchdog ──────────────────────────────────────────────────────────────────────────────────────
def _stable(rows: list) -> list:
    return [{k: v for k, v in row.items() if not k.startswith("seconds_since_")} for row in rows]


def test_watchdog_status_equals_the_cli_json(fleet, client):
    answer = client.call("lmds_watchdog_status")
    cli = cli_json(fleet, "watchdog", "status", "--json")
    assert answer["error"] is False and _stable(answer["payload"]) == _stable(cli)
    row = answer["payload"][0]
    assert row["slug"] == NATIVE_SLUG and row["armed"] is True and row["armed_by"] == "owner"
    assert row["service"]["state"] == "inactive" and row["service"]["unit"] == f"lmds-watchdog-{NATIVE_SLUG}.service"
    assert abs(row["seconds_since_last_probe"] - cli[0]["seconds_since_last_probe"]) < 120

    one = client.call("lmds_watchdog_status", {"slug": DOCKER_SLUG})
    assert _stable(one["payload"]) == _stable(cli_json(fleet, "watchdog", "status", DOCKER_SLUG, "--json"))
    assert one["payload"][0]["armed"] is False


def test_no_armed_watchdog_is_an_empty_list_on_both_sides(fleet, client):
    """เดิม `lmds watchdog status --json` พิมพ์ประโยคภาษาคนเมื่อไม่มีตัวไหนเปิด — ผู้เรียกที่ parse JSON พัง"""
    (fleet.watchdogs / f"{NATIVE_SLUG}.json").unlink()
    assert client.call("lmds_watchdog_status")["payload"] == []
    done = fleet.lmds("watchdog", "status", "--json")
    assert done.returncode == 0 and json.loads(done.stdout) == []
    assert "watchdog" in fleet.lmds("watchdog", "status").stdout, "แบบไม่ใส่ --json ยังบอกเป็นภาษาคนเหมือนเดิม"


def test_watchdog_status_on_a_node(fleet, client):
    assert client.call("lmds_watchdog_status", {"node": NODE_OK})["payload"] == NODE_WATCHDOG
    assert client.call("lmds_watchdog_status", {"node": NODE_OK, "slug": "gemma-3-27b"})["payload"] == NODE_WATCHDOG
    assert fleet.remote_commands() == [RO + "lmds watchdog status --json",
                                       RO + "lmds watchdog status gemma-3-27b --json"]


def test_an_older_node_that_answers_in_prose_means_no_watchdog(fleet, client):
    fleet.set_node(NODE_OK, "ok", [[RO + r"lmds watchdog status --json", {
        "stdout": "ยังไม่มี watchdog ที่เปิดไว้บนเครื่องนี้ — เปิด: lmds watchdog arm <slug>\n"}]])
    assert client.call("lmds_watchdog_status", {"node": NODE_OK})["payload"] == []


# ── logs ──────────────────────────────────────────────────────────────────────────────────────────
def test_logs_on_the_hub_are_the_controller_output_minus_secrets(fleet, client):
    answer = client.call("lmds_logs", {"slug": DOCKER_SLUG, "lines": 50})
    cli = fleet.lmds("logs", DOCKER_SLUG, "-n", "50")
    assert cli.returncode == 0 and cli.stdout == VLLM_LOG, "controller ตัวจริง → docker logs --tail 50"
    assert answer["error"] is False and answer["payload"] == {"slug": DOCKER_SLUG, "text": without_secrets(VLLM_LOG)}
    # เครื่องมืออ่านจากแหล่งเดียวกับที่ controller อ่าน โดยไม่รัน controller — ทั้งสองทางจบที่คำสั่งเดียวกัน
    assert fleet.calls("docker").count(["logs", "--tail", "50", f"lmds-{DOCKER_SLUG}"]) == 2


def test_log_lines_are_bounded(fleet, client):
    tail = client.call("lmds_logs", {"slug": DOCKER_SLUG, "lines": 2})["payload"]["text"]
    assert tail == without_secrets("".join(VLLM_LOG.splitlines(keepends=True)[-2:]))
    default = client.call("lmds_logs", {"slug": NATIVE_SLUG})["payload"]["text"]
    assert default.splitlines()[0].startswith("main: server is listening")
    assert ["logs", "--tail", "2", f"lmds-{DOCKER_SLUG}"] in fleet.calls("docker")


def test_logs_on_a_node(fleet, client):
    answer = client.call("lmds_logs", {"slug": "gemma-3-27b", "node": NODE_OK, "lines": 40})
    assert answer["error"] is False
    assert answer["payload"] == {"node": NODE_OK, "slug": "gemma-3-27b", "text": without_secrets(mcp_fleet.NODE_LOG)}
    assert fleet.remote_commands() == [RO + "lmds logs gemma-3-27b -n 40"]
    assert "CUDA out of memory" in answer["payload"]["text"]


def test_logs_of_an_unknown_bundle_are_an_error(fleet, client):
    assert client.call("lmds_logs", {"slug": "no-such-bundle"})["error"] is True
    remote = client.call("lmds_logs", {"slug": "no-such-bundle", "node": NODE_OK})
    assert remote["error"] is True and remote["payload"]["exit_code"] == 127 and remote["payload"]["node"] == NODE_OK


# ── doctor ────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("slug", [DOCKER_SLUG, NATIVE_SLUG, GHOST_SLUG, "no-such-bundle"])
def test_doctor_on_the_hub_equals_the_cli_json(fleet, client, slug):
    answer = client.call("lmds_doctor", {"slug": slug})
    cli = cli_json(fleet, "doctor", slug, "--json", "--no-probe", ok=(0, 2))
    assert answer["error"] is False and answer["payload"] == cli
    assert {"slug", "healthy", "findings"} <= set(cli)
    assert all(set(finding) == {"name", "status", "detail", "fix"} for finding in cli["findings"])
    assert not any(call[:1] == ["run"] for call in fleet.calls("docker")), "--no-probe: ไม่รัน container ไปถาม image"


def test_doctor_json_exit_code_matches_the_table_form(fleet):
    """`--json` เป็นของใหม่ (hub เรียกผ่าน SSH) — exit code ต้องชุดเดียวกับแบบตาราง: 2 เมื่อมีข้อที่ต้องแก้"""
    table = fleet.lmds("doctor", "no-such-bundle")
    as_json = fleet.lmds("doctor", "no-such-bundle", "--json")
    assert table.returncode == as_json.returncode == 2
    payload = json.loads(as_json.stdout)
    assert payload["healthy"] is False and payload["findings"][0]["status"] == "fail"
    assert "skipped" not in payload, "ไม่ใส่ --no-probe = ตรวจครบ ไม่มีข้อที่ข้าม"


def test_doctor_on_a_node(fleet, client):
    answer = client.call("lmds_doctor", {"slug": "gemma-3-27b", "node": NODE_OK})
    assert answer["error"] is False and answer["payload"] == NODE_DOCTOR
    assert fleet.remote_commands() == [RO + "lmds doctor gemma-3-27b --json --no-probe"]


def test_a_node_whose_lmds_is_too_old_for_doctor_json_says_so(fleet, client):
    fleet.set_node(NODE_OK, "ok", [[RO + r"lmds doctor gemma-3-27b --json --no-probe", {
        "stderr": "Usage: lmds doctor [OPTIONS] SLUG\nError: No such option: --json\n", "exit": 2}]])
    answer = client.call("lmds_doctor", {"slug": "gemma-3-27b", "node": NODE_OK})
    assert answer["error"] is True and "lmds node install" in answer["payload"]["error"]
    assert answer["payload"]["node"] == NODE_OK
