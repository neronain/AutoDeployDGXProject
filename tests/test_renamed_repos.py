"""repo ที่ Hub ย้ายไปชื่อใหม่: แผนยังใช้ชื่อที่ผู้ใช้ให้มา แต่ต้องบอกว่าชื่อนั้นทำงานผ่าน redirect

audit 2026-10-06 (hub 6e2b474): `THUDM/glm-4-9b-chat` → Hub ตอบ `id: zai-org/glm-4-9b-chat` (307 redirect) และ
`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` → `TensorFold/…` · แผนและ bundle เก็บชื่อเดิมโดยไม่มีอะไรบอกผู้ใช้เลย

ทางที่เลือก = คงชื่อ + เตือน (ไม่สลับเป็นชื่อใหม่ให้เอง) — สลับแล้ว deploy ซ้ำจะลงโฟลเดอร์ใหม่ `<slug>-<org ใหม่>`
แทนที่จะทับ bundle เดิม (resolve_slug เห็นเป็นคนละ repo) · สูตรที่ผูกชื่อเดิมไม่ถูกเจอ · rebuild ได้ model id คนละค่ากับ
ที่เครื่องปลายทางโหลด weight ไว้ · ต่างกันแค่ตัวพิมพ์ใช้ตัวสะกดของ Hub ได้เลยเพราะทุกอย่างข้างบนเทียบแบบไม่สนตัวพิมพ์
"""

import json

from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, analyze
from lmds.generator.renderer import resolve_slug
from lmds.inspector.report import ArtifactType, ModelReport
from tests.real_hub import RealHub

runner = CliRunner()
OLD, NEW = "THUDM/glm-4-9b-chat", "zai-org/glm-4-9b-chat"     # fixture: ขอด้วยชื่อเก่า · Hub ตอบ id ชื่อใหม่
SPARK = PRESETS["dgx-spark-single"]


def test_a_renamed_repo_keeps_the_requested_name_and_reports_the_current_one():
    report = RealHub(OLD).inspect(OLD)
    assert report.repo_id == OLD
    assert report.canonical_repo_id == NEW
    said = next(w for w in report.warnings if NEW in w)
    assert OLD in said and "redirect" in said


def test_the_plan_carries_the_warning_and_the_requested_name():
    report = RealHub(OLD).inspect(OLD)
    plan = build_plan(report, analyze(report, SPARK), None)
    assert plan.model_id == OLD
    assert any(NEW in w and "redirect" in w for w in plan.warnings)


def test_redeploying_a_renamed_repo_lands_in_the_same_bundle_folder(tmp_path):
    """เหตุผลที่ไม่สลับชื่อให้เอง: bundle เดิมผูกกับชื่อเดิม — ชื่อใหม่จะถูกมองเป็น "คนละ repo ชื่อซ้ำ" แล้วได้โฟลเดอร์ใหม่"""
    existing = tmp_path / "glm-4-9b-chat"
    existing.mkdir()
    (existing / "MODEL_PROFILE.yaml").write_text(f"model:\n  id: {OLD}\n")

    report = RealHub(OLD).inspect(OLD)
    assert resolve_slug(tmp_path, report.repo_id) == ("glm-4-9b-chat", "")
    assert resolve_slug(tmp_path, NEW)[0] == "glm-4-9b-chat-zai-org", "ถ้าสลับเป็นชื่อใหม่ จะได้ bundle ใบที่สอง"


def test_a_name_that_differs_only_in_case_uses_the_hubs_spelling_without_a_warning():
    report = RealHub("openai/gpt-oss-20b").inspect("OpenAI/GPT-OSS-20B")
    assert report.repo_id == "openai/gpt-oss-20b"
    assert report.canonical_repo_id is None
    assert not any("redirect" in w for w in report.warnings)


def test_an_unchanged_name_reports_nothing():
    report = RealHub("openai/gpt-oss-20b").inspect("openai/gpt-oss-20b")
    assert report.repo_id == "openai/gpt-oss-20b" and report.canonical_repo_id is None


def test_a_recipe_written_for_the_current_name_is_found_from_the_old_one_and_the_typed_name_wins():
    from lmds.brain.rulebased import recipe_for
    from lmds.recipes import find_recipe

    wanted = find_recipe("zai-org/GLM-4.7-Flash")
    assert wanted is not None, "สูตรนี้อยู่ใน catalog"

    def report(repo_id, canonical=None):
        return ModelReport(repo_id=repo_id, revision_sha="a" * 40, artifact_type=ArtifactType.SAFETENSORS,
                           canonical_repo_id=canonical)

    assert find_recipe("THUDM/GLM-4.7-Flash") is None
    assert recipe_for(report("THUDM/GLM-4.7-Flash", "zai-org/GLM-4.7-Flash")) is wanted
    assert recipe_for(report("zai-org/GLM-4.7-Flash", "someone-else/Renamed")) is wanted
    assert recipe_for(report("org/unknown-model")) is None


def test_cli_plan_shows_the_move(isolated_config, monkeypatch):
    fake = RealHub(OLD)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    result = runner.invoke(app, ["plan", OLD, "--target", "dgx-spark-single", "--no-llm", "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["model_id"] == OLD
    assert any(NEW in w for w in plan["warnings"])
