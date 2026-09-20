"""`lmds deploy --also-stacked` — bundle single + stacked จากการวิเคราะห์รอบเดียว

ROADMAP เฟส 2 ข้อ 1 เรียกของนี้ว่า `--topology both` แต่ `lmds cluster inspect|plan|apply`
มี `--topology` อยู่แล้วซึ่งหมายถึง *ผังสาย* (direct|ring|switch) — และใน `deploy` เอง
topology เป็นของที่ *อนุมาน* จาก target เสมอ (`topology_for_target()` + harden บังคับกลับ)
การเปิด `--topology` ที่นี่จึงชวนให้พิมพ์ `--topology stacked` แล้วโดนปฏิเสธ · ชื่อ
`--also-stacked` บอกตรง ๆ ว่า "ทำใบ stacked เพิ่มอีกใบ" ไม่ใช่ "ตั้ง topology"
"""

from typer.testing import CliRunner

from lmds.cli.main import app
from tests.test_generator import gguf_report, safetensors_report

runner = CliRunner()

DEPLOY = ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-single"]


def _patch_inspect(monkeypatch, report):
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda s, c: report)


def test_also_stacked_makes_two_bundles_that_do_not_overwrite_each_other(
        isolated_config, tmp_path, monkeypatch):
    """สองใบต้องอยู่คนละโฟลเดอร์ — slug เดียวกันสองรอบ = ใบหลังทับใบหน้า

    `render_bundle()` ย้าย controller ที่ topology ไม่ตรงออกเป็น `.replaced-<เวลา>` เสมอ
    (กันเคส 2026-09-05 ที่ download วิ่ง controller หนึ่ง start วิ่งอีกตัว) — ถ้า companion
    ใช้ slug เดิม ผลลัพธ์คือ bundle *ใบเดียว* ที่เป็น stacked และไฟล์ single ที่ถูกเปลี่ยนชื่อทิ้ง
    """
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 0, result.output

    single, stacked = tmp_path / "qwen3-32b", tmp_path / "qwen3-32b-stacked"
    assert (single / "qwen3-32b-single.sh").is_file()
    assert (stacked / "qwen3-32b-stacked-stacked.sh").is_file()
    # ไม่มีใบไหนถูกเขียนทับกันเอง
    assert not list(single.glob("*.replaced-*")) and not list(stacked.glob("*.replaced-*"))
    assert (tmp_path / "qwen3-32b.zip").is_file() and (tmp_path / "qwen3-32b-stacked.zip").is_file()


def test_also_stacked_bundles_carry_their_own_topology(isolated_config, tmp_path, monkeypatch):
    """ใบ stacked ต้องได้ controller multi-node จริง ไม่ใช่ single ที่เปลี่ยนชื่อ

    gate `stacked-contract` มีไว้ปิดช่องนี้อยู่แล้ว — เทสนี้ยืนยันว่า companion เดินผ่าน
    ทางเดียวกัน (วางแผนใหม่จาก fit ของ target stacked) ไม่ได้ก๊อป plan เดิมมาแก้ field
    """
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 0, result.output

    single = (tmp_path / "qwen3-32b" / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    stacked = (tmp_path / "qwen3-32b-stacked" / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    assert "topology: single" in single and "dgx-spark-single" in single
    assert "topology: stacked" in stacked and "dgx-spark-stacked" in stacked
    script = (tmp_path / "qwen3-32b-stacked" / "qwen3-32b-stacked-stacked.sh").read_text(encoding="utf-8")
    assert "sync-worker" in script and "verify-worker" in script


def test_each_bundle_gets_its_own_api_key(isolated_config, tmp_path, monkeypatch):
    """`_mint_key_for_new_bundle` ทำงานต่อ slug — สองใบต้องได้คนละกุญแจ ไม่ใช่ใบหลังเงียบ ๆ ไม่มี"""
    from lmds.fleet import apikey

    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 0, result.output

    first, second = apikey.read("qwen3-32b"), apikey.read("qwen3-32b-stacked")
    assert first and second and first != second


def test_also_stacked_with_a_stacked_target_is_refused(isolated_config, tmp_path, monkeypatch):
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-stacked",
                                 "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 1
    assert "เป็น stacked อยู่แล้ว" in result.output
    assert not list(tmp_path.glob("qwen3-32b*"))


def test_impossible_companion_is_refused_before_touching_the_hub(isolated_config, tmp_path, monkeypatch):
    """`--also-stacked --target rtx-5090` ไม่ควรกิน inspect ไปหนึ่งรอบก่อนถึงจะบอกว่าใช้ไม่ได้

    เงื่อนไขที่อ่านออกจาก argv ล้วน ๆ ต้องตัดจบก่อนยิง Hub — ไม่งั้นคนที่ repo gated
    จะโดนถาม token เสียก่อน แล้วค่อยได้ error ที่ไม่เกี่ยวกับ token เลย
    """
    calls = []
    monkeypatch.setattr("lmds.inspector.inspect_model",
                        lambda s, c: calls.append(s) or safetensors_report())
    result = runner.invoke(app, ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "rtx-5090",
                                 "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 1
    assert "preset ของ DGX Spark" in result.output
    assert calls == [], "ต้องปฏิเสธก่อน inspect"


def test_also_stacked_with_gguf_is_refused_and_writes_nothing(isolated_config, tmp_path, monkeypatch):
    """stacked มีแต่ controller ของ vLLM ส่วน GGUF บังคับ llama.cpp — ต้องบอกตั้งแต่ยังไม่ render

    ถ้าปล่อยให้ไปตายที่ `render_bundle()` ผู้ใช้จะได้ bundle single ที่ใช้ได้หนึ่งใบ
    พร้อม traceback หนึ่งก้อน แล้วต้องเดาเองว่าของที่ได้มาใช้ได้หรือเปล่า
    """
    _patch_inspect(monkeypatch, gguf_report())
    result = runner.invoke(app, ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm",
                                 "--target", "dgx-spark-single", "--output", str(tmp_path),
                                 "--yes", "--also-stacked"])
    assert result.exit_code == 1
    assert "llama.cpp" in result.output
    assert not (tmp_path / "qwen3-8b-gguf").exists()


def test_also_stacked_with_sglang_is_refused(isolated_config, tmp_path, monkeypatch):
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--engine", "sglang", "--output", str(tmp_path),
                                 "--yes", "--also-stacked"])
    assert result.exit_code == 1
    assert "SGLang" in result.output


def test_no_fit_on_a_single_machine_points_at_the_stacked_target(isolated_config, tmp_path, monkeypatch):
    """โมเดลที่ใหญ่เกินเครื่องเดียว: --also-stacked ช่วยไม่ได้ (ต้องมีใบ single ที่ fit ก่อน)
    — ต้องชี้ทางไป --target dgx-spark-stacked ตรง ๆ ไม่ใช่ปล่อยให้งงว่าสั่งถูกแล้วทำไมยังไม่ fit
    """
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "rtx-pro-4000",
                                 "--output", str(tmp_path), "--yes"])
    assert result.exit_code == 3  # ทางปกติยังเป็น exit 3 เหมือนเดิม

    result = runner.invoke(app, ["deploy", "Qwen/Qwen3-32B", "--no-llm", "--target", "dgx-spark-single",
                                 "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 0, result.output  # ตัวนี้ fit — ทางชี้แนะอยู่ที่เทสข้างล่าง


def test_companion_refuses_to_overwrite_another_models_bundle(isolated_config, tmp_path, monkeypatch):
    """ส่ง slug ตรง ๆ ให้ render_bundle = ข้าม `resolve_slug()` ซึ่งเป็นตัวกันชื่อชนตามปกติ

    เคสจริง 2026-09-05 (ucbye/nvidia ชื่อโมเดลเดียวกันคนละเจ้าของ) โผล่มาได้อีกทางนี้:
    โฟลเดอร์ <slug>-stacked ที่เป็นของโมเดลอื่นอยู่แล้วจะถูกทับเงียบ ๆ
    """
    (tmp_path / "qwen3-32b-stacked").mkdir(parents=True)
    (tmp_path / "qwen3-32b-stacked" / "MODEL_PROFILE.yaml").write_text(
        "model:\n  id: someone-else/Qwen3-32B\n", encoding="utf-8")

    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes", "--also-stacked"])
    assert result.exit_code == 1
    assert "เป็นของโมเดลอื่น" in result.output
    # ใบ single ที่ทำเสร็จไปแล้วยังอยู่ และของคนอื่นไม่ถูกแตะ
    assert (tmp_path / "qwen3-32b" / "qwen3-32b-single.sh").is_file()
    assert "someone-else" in (tmp_path / "qwen3-32b-stacked" / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")


def test_companion_slug_stays_inside_the_length_limit(isolated_config):
    """ชื่อจาก repo ยาว 64 ตัวพอดีแล้วต่อ `-stacked` = 72 → `check_slug_name` โยน ValueError

    ValueError ตัวนั้นถูกจับใน `_render_and_package` แล้ว exit 1 — แปลว่าทิ้งงานทั้งรอบ
    *หลัง* ใบ single ลงดิสก์ไปแล้ว ซึ่งเป็นอาการที่แย่ที่สุดของทั้งฟีเจอร์
    """
    from lmds.brain.rulebased import MAX_SLUG_LEN
    from lmds.cli.main import _companion_slug
    from lmds.generator.renderer import check_slug_name

    for base in ("m", "a" * MAX_SLUG_LEN, "b" * (MAX_SLUG_LEN - 4), "c" * 40 + "-" * 20):
        slug = _companion_slug(base)
        assert slug.endswith("-stacked")
        assert check_slug_name(slug) == slug  # ไม่โยน


def test_companion_reuses_the_context_the_user_typed(isolated_config, tmp_path, monkeypatch):
    """ค่า context ที่ผู้ใช้พิมพ์เองตอนยืนยัน ต้องตามไปใบ stacked ด้วย (ไม่ถามซ้ำ)"""
    from lmds.cli.main import _deploy_companion_stacked

    slug = _deploy_companion_stacked(
        safetensors_report(), None, None, 1, str(tmp_path),
        base_slug="qwen3-32b", approved=[], approved_assets=[], chosen_context=16384,
    )
    assert slug == "qwen3-32b-stacked"
    profile = (tmp_path / slug / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    assert "context: 16384" in profile


def test_companion_replays_the_approvals_instead_of_asking_again(isolated_config, tmp_path, monkeypatch):
    """flag/ไฟล์ที่อนุมัติแล้วผูกกับ *โมเดล* ไม่ใช่กับจำนวนเครื่อง — ถามซ้ำคือฝึกให้กด y รัว ๆ"""
    from lmds.cli import main as cli_main

    seen = {}
    monkeypatch.setattr(cli_main, "_render_and_package",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr("lmds.brain.apply_flag_approvals",
                        lambda plan, approved: seen.setdefault("flags", list(approved)) or plan)
    monkeypatch.setattr("lmds.brain.apply_asset_approvals",
                        lambda plan, approved: seen.setdefault("assets", list(approved)) or plan)

    try:
        _ = cli_main._deploy_companion_stacked(
            safetensors_report(), None, None, 1, str(tmp_path),
            base_slug="qwen3-32b", approved=["--trust-remote-code"],
            approved_assets=["patch.py"], chosen_context=None,
        )
    except RuntimeError:
        pass
    assert seen == {"flags": ["--trust-remote-code"], "assets": ["patch.py"]}


def test_plain_deploy_is_unchanged_without_the_flag(isolated_config, tmp_path, monkeypatch):
    """ไม่ใส่ --also-stacked = ได้ใบเดียวเหมือนเดิม และไม่มีหัวข้อ 'ใบที่ 1/2' มากวน"""
    _patch_inspect(monkeypatch, safetensors_report())
    result = runner.invoke(app, [*DEPLOY, "--output", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "qwen3-32b-stacked").exists()
    assert "ใบที่ 1/2" not in result.output
