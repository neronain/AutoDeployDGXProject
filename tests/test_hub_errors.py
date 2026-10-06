"""สาเหตุที่รายงานต้องตรงกับที่ Hub ตอบ — repo ที่ไม่มีอยู่ไม่ใช่ "gated"

เคสจริง 2026-10-06 (audit · hub 6e2b474): `lmds inspect Qwen/Qwen3-8B-typo-zz --json` →
"repo นี้เป็น gated/private — ต้องใช้ Hugging Face token (HTTP 401)" exit 4 (โหมดโต้ตอบถาม token) ·
Hub ตอบ 401 ให้ repo ที่ไม่มีอยู่จริงเมื่อไม่ได้ล็อกอิน และ hf_api แปลทุก 401 เป็น AuthRequired

header ที่ใช้ในเทสคือของจริงที่วัดวันนั้น:
- `/api/models/Qwen/Qwen3-8B-typo-zz` → 401 · `x-error-message: Invalid username or password.` · ไม่มี x-error-code
- `/meta-llama/Prompt-Guard-86M/resolve/main/config.json` → 401 · `x-error-code: GatedRepo` (ส่วน /api/models ตอบ 200)
- `/api/models/openai/gpt-oss-20b/revision/nope-zz` → 404 · `x-error-code: RevisionNotFound`
- `/api/models/Qwen/Qwen3-8B/revision/refs%2Fpr%2F1` → 200 · แบบไม่ encode (`refs/pr/1`) → 404
"""

import httpx
import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.inspector import AuthRequired, HfClient, RepoNotFound, inspect_model
from lmds.resolver import parse_source
from tests.real_hub import flat

runner = CliRunner()
SHA = "a" * 40
GATED_INFO = {"id": "meta-llama/Prompt-Guard-86M", "sha": SHA, "gated": "manual", "private": False,
              "pipeline_tag": "text-classification", "tags": ["transformers", "safetensors"],
              "siblings": [{"rfilename": "config.json", "size": 800}, {"rfilename": "model.safetensors", "size": 10**9}]}


def hub(request: httpx.Request) -> httpx.Response:
    path = request.url.raw_path.decode().split("?")[0]
    if path.startswith("/api/models/meta-llama/Prompt-Guard-86M"):
        return httpx.Response(200, json=GATED_INFO)
    if path.startswith("/meta-llama/Prompt-Guard-86M/resolve/"):
        return httpx.Response(401, headers={"x-error-code": "GatedRepo",
                                            "x-error-message": "Access to model meta-llama/Prompt-Guard-86M is restricted."})
    if path == "/api/models/openai/gpt-oss-20b/revision/nope-zz":
        return httpx.Response(404, headers={"x-error-code": "RevisionNotFound", "x-error-message": "Invalid rev id: nope-zz"})
    if path == "/api/models/Qwen/Qwen3-8B/revision/refs%2Fpr%2F1":
        return httpx.Response(200, json={"id": "Qwen/Qwen3-8B", "sha": SHA, "siblings": [], "tags": []})
    if path.startswith("/api/models/Qwen/Qwen3-8B/revision/"):
        return httpx.Response(404, headers={"x-error-message": "Sorry, we can't find the page you are looking for."})
    if path.startswith("/api/models/org/deleted-repo"):
        return httpx.Response(404, headers={"x-error-code": "RepoNotFound", "x-error-message": "Repository not found"})
    # ทุกอย่างที่เหลือ = repo ที่ไม่มีอยู่ ถูกถามโดยไม่ล็อกอิน
    return httpx.Response(401, headers={"x-error-message": "Invalid username or password."},
                          json={"error": "Invalid username or password."})


def client(token=None) -> HfClient:
    return HfClient(token=token, client=httpx.Client(transport=httpx.MockTransport(hub)))


@pytest.fixture
def fake_hub(isolated_config, monkeypatch):
    monkeypatch.setattr("lmds.inspector.HfClient", client)


def test_a_missing_repo_is_not_reported_as_gated():
    with pytest.raises(AuthRequired) as err:
        inspect_model(parse_source("Qwen/Qwen3-8B-typo-zz"), client())
    assert err.value.reason == "unknown"
    said = str(err.value)
    assert "ไม่พบ" in said and "private" in said
    assert "เป็น gated" not in said, "Hub ไม่ได้บอกว่า gated — อย่าอ้างแทนมัน"


def test_a_really_gated_repo_is_reported_as_gated():
    with pytest.raises(AuthRequired) as err:
        inspect_model(parse_source("meta-llama/Prompt-Guard-86M"), client())
    assert err.value.reason == "gated" and "gated" in str(err.value)

    with pytest.raises(AuthRequired) as err:
        inspect_model(parse_source("meta-llama/Prompt-Guard-86M"), client(token="hf_not_granted"))
    assert err.value.had_token and "ยอมรับเงื่อนไข" in str(err.value)


def test_repo_not_found_code_wins_whatever_the_status():
    with pytest.raises(RepoNotFound, match="ไม่พบ model repo"):
        client(token="hf_valid").model_info("org/deleted-repo")


def test_a_wrong_revision_is_reported_as_a_wrong_revision():
    """เดิม "ไม่พบ model repo: openai/gpt-oss-20b" ทั้งที่ repo มีอยู่"""
    with pytest.raises(RepoNotFound) as err:
        client().model_info("openai/gpt-oss-20b", "nope-zz")
    said = str(err.value)
    assert "revision" in said and "nope-zz" in said and "ไม่พบ model repo" not in said


def test_a_revision_with_a_slash_reaches_the_hub_as_one_path_segment():
    """refs/pr/1 ต้องไปถึง Hub เป็น refs%2Fpr%2F1 — ไม่ encode ได้ 404"""
    assert client().model_info("Qwen/Qwen3-8B", "refs/pr/1")["sha"] == SHA


# ── CLI: ข้อความ + exit code ─────────────────────────────────────────────────────────────────────────


def test_cli_inspect_of_a_typo_says_not_found_or_private_and_keeps_exit_4(fake_hub):
    result = runner.invoke(app, ["inspect", "Qwen/Qwen3-8B-typo-zz", "--json"])
    assert result.exit_code == 4, "exit code ตามเอกสาร: ยังอาจเป็น repo private ที่ token แก้ได้"
    said = flat(result.output)
    assert "ไม่พบrepoนี้หรือเป็นrepoprivate" in said
    assert "เป็นgated" not in said


def test_cli_inspect_of_a_gated_repo_says_gated(fake_hub):
    result = runner.invoke(app, ["inspect", "meta-llama/Prompt-Guard-86M", "--json"])
    assert result.exit_code == 4
    assert "เป็นgated" in flat(result.output)


def test_cli_inspect_with_a_wrong_revision_is_an_input_error(fake_hub):
    result = runner.invoke(app, ["inspect", "openai/gpt-oss-20b", "--revision", "nope-zz", "--json"])
    assert result.exit_code == 1
    assert "revision" in flat(result.output)


def test_web_analyze_of_a_typo_does_not_claim_the_repo_is_gated(fake_hub):
    from lmds.web import deploy as dep

    with pytest.raises(dep.DeployError) as err:
        dep.analyze("Qwen/Qwen3-8B-typo-zz", target="dgx-spark-single", no_llm=True)
    assert "ไม่พบ repo นี้ หรือเป็น repo private" in err.value.message
    assert "เป็น gated" not in err.value.message
