"""split GGUF ที่ repo มีไม่ครบชุดต้องถูกปฏิเสธพร้อมเลข part ที่ขาด — ไม่ใช่รวมขนาดของ part ที่เหลือแล้วตอบ "fits"

audit 2026-10-06 (hub 6e2b474): `_group_gguf_variants` ไม่เคยเทียบจำนวน part กับ `-of-NNNNN` ในชื่อไฟล์ — part 1 กับ 3
จาก 3 กลายเป็น variant เดียว ไม่มีคำเตือน ขนาดรวมขาดไปหนึ่ง part และ fit ตอบ "fits"

รายชื่อไฟล์คือของจริงจาก `bartowski/Llama-3.1-70B-Instruct-lorablated-GGUF` (Q5_K_M / Q6_K / Q8_0 แยกเป็น 2 part ใน
โฟลเดอร์ย่อย) — เคส "ขาด part" สร้างด้วยการตัดไฟล์ออกจากรายชื่อจริงนั้น
"""

import pytest
from typer.testing import CliRunner

from lmds.brain import Engine, PlanError, build_plan
from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from tests.real_hub import RealHub, flat, load

runner = CliRunner()
REPO = "bartowski/Llama-3.1-70B-Instruct-lorablated-GGUF"
Q8 = "Llama-3.1-70B-Instruct-lorablated-Q8_0/Llama-3.1-70B-Instruct-lorablated-Q8_0"
PART1, PART2 = f"{Q8}-00001-of-00002.gguf", f"{Q8}-00002-of-00002.gguf"
LINK = f"https://huggingface.co/{REPO}/blob/main/{PART1}"
SPARK = PRESETS["dgx-spark-single"]


def without(*names):
    def edit(fixture):
        fixture["info"]["siblings"] = [s for s in fixture["info"]["siblings"] if s["rfilename"] not in names]

    return {REPO: edit}


def size_of(name: str) -> int:
    return next(s["size"] for s in load(REPO)["info"]["siblings"] if s["rfilename"] == name)


def test_a_complete_split_set_is_one_variant_with_the_summed_size():
    report = RealHub(REPO).inspect(LINK)
    variant = next(v for v in report.gguf_variants if v.filename == PART1)
    assert [p.filename for p in variant.parts] == [PART1, PART2] and variant.missing_parts == []
    assert report.unsupported_format is None
    assert report.weight_bytes == size_of(PART1) + size_of(PART2)

    fit = analyze(report, SPARK)
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert build_plan(report, fit, None).runtime.engine is Engine.LLAMACPP


def test_a_split_set_with_a_missing_part_is_refused_with_the_part_number():
    report = RealHub(REPO, edits=without(PART2)).inspect(LINK)
    assert report.unsupported_format == "incomplete-gguf"
    assert "ขาด part 2 จาก 2" in report.unsupported_evidence[0]

    fit = analyze(report, SPARK)
    assert fit.verdict is Verdict.UNSUPPORTED, "เดิม fits ด้วยขนาดของ part ที่เหลือ (39.8 GB จาก 75 GB)"
    with pytest.raises(PlanError) as err:
        build_plan(report, fit, None)
    said = str(err.value)
    assert "ขาด part 2 จาก 2" in said
    assert "quant อื่นของ repo นี้ที่ครบชุด" in said and "Llama-3.1-70B-Instruct-lorablated-IQ" in said
    assert "Q8_0-00001-of-00002.gguf," not in said, "ตัวที่ขาดต้องไม่อยู่ในรายการทางเลือก"


def test_a_hole_in_the_middle_is_found_from_the_file_names():
    """part 1 กับ 3 จาก 3 — เคสของผู้ตรวจ: จำนวนที่ต้องมีอยู่ในชื่อไฟล์เอง"""
    from lmds.inspector.inspect import _group_gguf_variants

    variants = _group_gguf_variants([("big-Q8_0-00001-of-00003.gguf", 40 * 10**9, None),
                                     ("big-Q8_0-00003-of-00003.gguf", 40 * 10**9, None),
                                     ("big-Q4_K_M.gguf", 20 * 10**9, None)])
    split = next(v for v in variants if v.parts)
    assert split.missing_parts == [2]
    assert next(v for v in variants if not v.parts).missing_parts == []


def test_an_incomplete_set_is_flagged_and_left_out_of_the_choices_before_anything_is_selected():
    report = RealHub(REPO, edits=without(PART2)).inspect(REPO)
    assert report.unsupported_format is None, "repo ยังมี quant อื่นที่ครบชุด — ปฏิเสธเฉพาะเมื่อเลือกตัวที่ขาด"
    assert any("ไม่ครบชุด" in w and "Q8_0" in w for w in report.warnings)
    offered = [v.filename for v in analyze(report, SPARK).variant_fits]
    assert offered and PART1 not in offered


def test_cli_generate_refuses_the_incomplete_quant_and_builds_a_complete_one(isolated_config, monkeypatch, tmp_path):
    fake = RealHub(REPO, edits=without(PART2))
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    out = tmp_path / "bundles"
    base = ["generate", REPO, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)]

    result = runner.invoke(app, [*base, "--gguf", "Q8_0"])
    assert result.exit_code == 1, result.output
    assert "ขาดpart2จาก2" in flat(result.output) and not out.exists()

    result = runner.invoke(app, [*base, "--gguf", "Q4_K_M"])
    assert result.exit_code == 0, result.output


def test_fit_rejects_a_concurrency_below_one_as_bad_input():
    report = RealHub(REPO).inspect(LINK)
    for bad in (0, -2):
        with pytest.raises(ValueError, match="concurrency"):
            analyze(report, SPARK, concurrency=bad)
