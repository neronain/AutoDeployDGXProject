"""worker ต้องหาน้ำหนักเจอ แม้เลย์เอาต์ hub cache คนละแบบกับ head

เคสจริง 2026-09-01 บน spark-head/spark-worker ระหว่างสั่ง start แบบ stacked:

    LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder
    ERROR: worker container บน 10.100.152.2 หยุดก่อน head จะเริ่ม

ทั้งที่ worker มีน้ำหนักครบ 171 GB ทั้ง 17 shard และชื่อโฟลเดอร์ snapshot ตรง commit เป๊ะ

ต้นเหตุ: head โหลดเองด้วย snapshot_download(cache_dir='/cache') จึงได้เลย์เอาต์แบน
(/cache/models--…) ส่วน worker ได้ของมาจาก rsync ที่ลงเป็นเลย์เอาต์มาตรฐานของ HF
(/cache/hub/models--…) · แต่ _container_hub_cache รันบน head แล้วเอาคำตอบไปแปะใส่
สคริปต์ของ worker — เท่ากับเอาเลย์เอาต์ของเครื่องหนึ่งไปตอบแทนอีกเครื่องหนึ่ง
"""

import pathlib
import subprocess
import tempfile

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport

SLUG = "org--stacked-model"


@pytest.fixture(scope="module")
def controller() -> str:
    report = ModelReport(
        repo_id="org/stacked-model", revision_sha="deadbeef",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(160 * GIB),
        context_length=131072, kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, pathlib.Path(tempfile.mkdtemp()))
    return next(bundle.directory.glob("*-stacked.sh")).read_text(encoding="utf-8")


def _decision_block(text: str) -> str:
    """ดึงเฉพาะท่อนที่ตัดสิน HF_HUB_CACHE ในสคริปต์ของ worker"""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('if [ -d "/cache/models--'))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "fi")
    return "\n".join(lines[start : end + 1])


def _resolve(controller: str, tmp_path, layout: str) -> str:
    """สร้างเลย์เอาต์ปลอมแล้วรันท่อนตัดสินจริง — คืนค่า HF_HUB_CACHE ที่ได้"""
    root = tmp_path / layout
    if layout == "flat":
        (root / f"models--{SLUG}" / "snapshots").mkdir(parents=True)
    else:
        (root / "hub" / f"models--{SLUG}" / "snapshots").mkdir(parents=True)
    block = _decision_block(controller).replace("/cache", str(root))
    script = tmp_path / f"decide-{layout}.sh"
    script.write_text(
        f'_model_slug() {{ printf %s "{SLUG}"; }}\n{block}\necho "$HF_HUB_CACHE"\n',
        encoding="utf-8",
    )
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().replace(str(root), "/cache")


def test_worker_finds_weights_in_the_standard_hub_layout(controller, tmp_path):
    """rsync ลงของให้ worker เป็นเลย์เอาต์มาตรฐาน — เคสที่พังจริง"""
    assert _resolve(controller, tmp_path, "hub") == "/cache/hub"


def test_worker_finds_weights_in_the_flat_layout(controller, tmp_path):
    """เครื่องที่โหลดเองด้วย cache_dir=/cache ได้เลย์เอาต์แบน — ต้องยังหาเจอ"""
    assert _resolve(controller, tmp_path, "flat") == "/cache"


def test_the_choice_is_made_on_the_worker_not_baked_in_by_the_head(controller):
    """ถ้ายังเป็นค่าคงที่ที่ head คำนวณไว้ ก็จะผิดทุกครั้งที่สองเครื่องเลย์เอาต์ไม่ตรงกัน"""
    worker_part = controller[controller.index("export VLLM_HOST_IP=") :][:3000]
    baked = [
        l for l in worker_part.splitlines()
        if l.startswith("export HF_HUB_CACHE=") and "$(" in l
    ]
    assert not baked, f"HF_HUB_CACHE ของ worker ถูกคำนวณบน head: {baked}"
