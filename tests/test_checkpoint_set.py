"""ขนาด weight = checkpoint ที่ engine โหลดจริง ไม่ใช่ผลรวมของทุกไฟล์ .safetensors ใน repo

เคสจริง 2026-10-06 (audit ~90 repo บน Hub · hub 6e2b474):
- `openai/gpt-oss-120b` มี `original/` เป็นสำเนาที่สอง → รายงาน 130.5 GB (22 ไฟล์ · index ที่รากบอก 15 shard)
  ของจริง 65.25 GB → `needs-smaller-quant` บน dgx-spark-single (budget 113.5 GB) และ `lmds generate` ออก exit 3
  บอกให้ไปหา checkpoint ที่ quantize แล้ว ทั้งที่ตัวนี้ลงได้
- `mistralai/Devstral-2-123B-Instruct-2512` (`consolidated-*` 27 ไฟล์คู่กับ `model-*` 27 ไฟล์) บน dgx-spark-stacked:
  256.5 GB แทน 128.25 → ปฏิเสธ ทั้งที่ลงได้
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503`: 96 GB แทน 48 → context ถูกตัด 131,072 → 106,496
- `tencent/HunyuanOCR` (`v1.0/` · `dflash/`) · `Inferact/Qwen3.8-27B-NVFP4` (index ชี้ไฟล์ชื่อไม่มาตรฐาน)

metadata ใน tests/hub_fixtures/ คือของจริงจาก Hub วันนั้น — ดู tests/real_hub.py
"""

import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.fit import PRESETS, Verdict, analyze
from lmds.inspector import ArtifactType
from tests.real_hub import RealHub, load, shard_files_of

runner = CliRunner()

GPT_OSS = "openai/gpt-oss-120b"
DEVSTRAL = "mistralai/Devstral-2-123B-Instruct-2512"
MISTRAL_SMALL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
HUNYUAN_OCR = "tencent/HunyuanOCR"
INFERACT = "Inferact/Qwen3.8-27B-NVFP4"


def sizes_of(repo: str) -> dict[str, int]:
    return {s["rfilename"]: s["size"] for s in load(repo)["info"]["siblings"]}


@pytest.fixture
def hub(isolated_config, monkeypatch):
    fake = RealHub(GPT_OSS, DEVSTRAL, MISTRAL_SMALL, HUNYUAN_OCR, INFERACT)
    monkeypatch.setattr("lmds.inspector.HfClient", fake.client)
    return fake


# ── ขนาด / จำนวน shard มาจากชุดที่ index ที่ราก repo ชี้ ────────────────────────────────────────────


def test_gpt_oss_120b_counts_the_root_checkpoint_not_the_second_copy_in_original():
    sizes = sizes_of(GPT_OSS)
    root = sorted(n for n in sizes if n.startswith("model-") and n.endswith(".safetensors"))
    assert len(root) == 15, "fixture ต้องตรงกับ repo จริง: model-00000 … model-00014"

    report = RealHub(GPT_OSS).inspect(GPT_OSS)

    assert report.artifact_type is ArtifactType.SAFETENSORS
    assert report.shard_count == 15
    assert [s.filename for s in report.safetensor_shards] == root
    assert report.weight_bytes == sum(sizes[n] for n in root)
    assert round(report.weight_bytes / 1e9, 2) == 65.25, "เดิมรายงาน 130.5 GB"


def test_files_outside_the_checkpoint_are_reported_so_the_number_can_be_explained():
    sizes = sizes_of(GPT_OSS)
    spare = [n for n in sizes if n.startswith("original/") and n.endswith(".safetensors")] + ["metal/model.bin"]

    report = RealHub(GPT_OSS).inspect(GPT_OSS)

    assert report.other_weight_files == len(spare) == 8
    assert report.other_weight_bytes == sum(sizes[n] for n in spare)
    said = " ".join(report.warnings)
    assert "original/" in said and "metal/" in said, "ต้องบอกว่าไฟล์ที่ไม่นับอยู่ตรงไหน"


def test_gpt_oss_120b_fits_a_single_dgx_spark():
    """เดิม needs-smaller-quant (130.5 GB > budget 113.5 GB) ทั้งที่ของจริง 60.8 GiB"""
    fit = analyze(RealHub(GPT_OSS).inspect(GPT_OSS), PRESETS["dgx-spark-single"])
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)
    assert fit.weights_gb == pytest.approx(60.8, abs=0.1)
    assert fit.recommended_context == 131072


def test_devstral_2_fits_two_stacked_sparks():
    """`consolidated-*` คือสำเนาในรูปแบบ mistral ของ shard ชุดเดียวกัน — เดิมนับรวมเป็น 256.5 GB แล้วปฏิเสธ"""
    report = RealHub(DEVSTRAL).inspect(DEVSTRAL)
    assert report.shard_count == 27
    assert all(s.filename.startswith("model-") for s in report.safetensor_shards)
    assert round(report.weight_bytes / 1e9, 2) == 128.25
    assert report.other_weight_files == 27

    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    assert fit.verdict in (Verdict.FITS, Verdict.FITS_REDUCED_CONTEXT)


def test_mistral_small_keeps_its_native_context():
    """นับ `consolidated.safetensors` ซ้ำ = 96 GB → context ถูกตัดเหลือ 106,496 ทั้งที่ native 131,072 ลงได้"""
    report = RealHub(MISTRAL_SMALL).inspect(MISTRAL_SMALL)
    assert round(report.weight_bytes / 1e9, 2) == 48.02
    assert "consolidated.safetensors" not in [s.filename for s in report.safetensor_shards]

    fit = analyze(report, PRESETS["dgx-spark-single"])
    assert fit.verdict is Verdict.FITS
    assert fit.recommended_context == 131072


def test_without_a_root_index_the_root_model_file_is_the_checkpoint():
    """tencent/HunyuanOCR: `model.safetensors` ที่ราก · `v1.0/` กับ `dflash/` เป็นโมเดลคนละตัวในโฟลเดอร์ย่อย"""
    sizes = sizes_of(HUNYUAN_OCR)
    report = RealHub(HUNYUAN_OCR).inspect(HUNYUAN_OCR)
    assert [s.filename for s in report.safetensor_shards] == ["model.safetensors"]
    assert report.weight_bytes == sizes["model.safetensors"]
    assert report.shard_count == 1
    assert report.other_weight_files == 5


def test_a_shard_the_index_names_is_part_of_the_checkpoint_whatever_it_is_called():
    """Inferact/Qwen3.8-27B-NVFP4: index ที่รากชี้ `nvfp4_experts_mtp.safetensors` ด้วย — เชื่อ index ไม่ใช่ชื่อไฟล์"""
    sizes = sizes_of(INFERACT)
    report = RealHub(INFERACT).inspect(INFERACT)
    names = [s.filename for s in report.safetensor_shards]
    assert "nvfp4_experts_mtp.safetensors" in names and len(names) == 7
    assert report.weight_bytes == sum(sizes[n] for n in names)
    assert report.other_weight_files == 0
    assert not any("ไฟล์ weight อื่น" in w for w in report.warnings), "ไม่มีของเกิน ก็ไม่ต้องเตือน"


# ── ลำดับสำรองเมื่อไม่มี index ใช้ได้ — เทียบกับรายชื่อไฟล์จริง ──────────────────────────────────────


def chosen(names, index=None, mistral_native=False):
    from lmds.inspector.inspect import checkpoint_files

    return checkpoint_files(names, index, mistral_native=mistral_native)


def test_fallback_order_on_real_listings():
    # index ค้างจากต้นฉบับ: icefog72/IceWhiskeyRP-7b-8bpw-exl2 — index ชี้ 3 shard ที่ไม่มีอยู่ ไฟล์จริงคือ output.safetensors
    stale = {"weight_map": {"a": "model-00001-of-00003.safetensors", "b": "model-00002-of-00003.safetensors"}}
    assert chosen(["output.safetensors"], stale) == (["output.safetensors"], [], "other")
    # LiquidAI/LFM2.5-2.6B-GGUF: safetensors อยู่แต่ในโฟลเดอร์ย่อย → ไม่มี checkpoint ที่ราก
    assert chosen(["qad/model.safetensors"]) == ([], [], "none")
    # IFM/K2-Horizon-7B-Uno: มีแต่ adapter
    assert chosen(["adapter_model.safetensors"]) == ([], [], "none")
    # mistralai/Pixtral-12B-2409: รูปแบบ mistral ล้วน (params.json + consolidated.safetensors)
    assert chosen(["consolidated.safetensors"], mistral_native=True) == (["consolidated.safetensors"], [], "mistral")
    # model.safetensors ที่รากชนะไฟล์อื่นที่รากเสมอ
    assert chosen(["model.safetensors", "adapter_model.safetensors", "sub/model.safetensors"])[0] == ["model.safetensors"]


def test_a_partly_missing_shard_set_is_reported_not_silently_summed():
    index = {"weight_map": {str(i): f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)}}
    files, missing, origin = chosen(["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"], index)
    assert origin == "index" and len(files) == 2
    assert missing == ["model-00002-of-00003.safetensors"]


# ── bundle: รายการที่ controller ตรวจ/โหลด ต้องเป็นชุดเดียวกัน ───────────────────────────────────────


def test_cli_generate_builds_the_bundle_it_used_to_refuse(hub, tmp_path):
    """เดิม exit 3 "โมเดลไม่ fit … หา checkpoint ที่ quantize แล้ว" · และ SHARD_FILES ต้องไม่มีสำเนาใน original/"""
    out = tmp_path / "bundles"
    result = runner.invoke(app, ["generate", GPT_OSS, "--target", "dgx-spark-single", "--no-llm", "--output", str(out)])
    assert result.exit_code == 0, result.output

    controller = next(out.glob("*/*-single.sh")).read_text()
    shards = shard_files_of(controller)
    assert len(shards) == 15
    assert all(name.startswith("model-") and "/" not in name for name in shards)
    assert not hub.weight_requests


def test_cli_inspect_shows_the_real_size(hub):
    import json

    result = runner.invoke(app, ["inspect", GPT_OSS, "--target", "dgx-spark-single", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert round(payload["model"]["weight_bytes"] / 1e9, 2) == 65.25
    assert payload["model"]["shard_count"] == 15
    assert payload["fit"][0]["verdict"] in ("fits", "fits-reduced-context")
