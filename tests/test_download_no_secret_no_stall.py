"""ตัวโหลดฝั่ง vLLM/SGLang: ความลับต้องไม่อยู่ใน argv และต้องไม่ค้างตาย

สองเคสจริงจาก spark-head 2026-08-31 ระหว่างโหลด NVFP4 170.9 GB:

1. token โผล่เต็ม ๆ ใน `ps` ของทั้งเครื่อง —
   `docker run ... -e HF_TOKEN=hf_xxxx ...` เอาค่าจริงไปแปะไว้ใน argv
   LMDS มีหลักเรื่องนี้อยู่แล้วในเส้นทาง SSH (ส่งทาง stdin เท่านั้น) แต่เส้นทาง docker
   หลุดหลักนั้นไป

2. โหลดไปได้ 36 GB แล้วหยุดนิ่ง **1 ชั่วโมง 32 นาที** โดยไม่มี error ไม่มี log ·
   connection ไปหา CDN ยังเปิดค้าง process นอนหลับใน recv() รอข้อมูลที่ไม่มีวันมา ·
   ฝั่ง llama.cpp กันไว้แล้วด้วย --speed-limit/--speed-time ของ curl แต่
   huggingface_hub ในคอนเทนเนอร์ไม่มีตัวกันแบบนั้น
   คนที่เห็นว่าค้างมักสั่งใหม่ทับ แล้วตัวใหม่ไปติด lock ของตัวเดิม — กองซ้อนกันสามตัว
   ที่ไม่มีตัวไหนทำงานเลย
"""

import pathlib
import tempfile

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(preset: str, weight_gib: float) -> str:
    report = ModelReport(
        repo_id="org/gated-model", revision_sha="sha",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(weight_gib * GIB),
        context_length=131072, kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS[preset])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, pathlib.Path(tempfile.mkdtemp()))
    return next(bundle.directory.glob("*.sh")).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def controllers():
    return {"single": _controller("dgx-spark-single", 40),
            "stacked": _controller("dgx-spark-stacked", 160)}


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_the_token_value_never_reaches_docker_argv(controllers, kind):
    """`-e HF_TOKEN` เฉย ๆ = docker หยิบค่าจาก env · `-e HF_TOKEN=ค่า` = ทุกคนอ่านได้จาก ps"""
    text = controllers[kind]
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert 'HF_TOKEN=${HF_TOKEN}' not in code, "เอาค่า token ไปแปะใน argv ของ docker"
    assert 'HF_TOKEN="$HF_TOKEN"' not in code
    assert "-e HF_TOKEN" in code, "ยังต้องส่ง token ให้คอนเทนเนอร์ผ่านชื่อตัวแปร"


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_a_download_that_stops_moving_is_restarted(controllers, kind):
    """เงื่อนไขคือ 'ยังคืบหน้าอยู่ไหม' ไม่ใช่แค่ 'process ยังไม่ตาย'"""
    text = controllers[kind]
    assert "DOWNLOAD_STALL_SECONDS" in text, "ไม่มีตัวกันค้างตาย"
    assert "_download_bytes" in text, "ไม่ได้วัดว่าไบต์เพิ่มขึ้นจริงไหม"
    # ต้องหยุดตัวที่ค้างก่อนเริ่มใหม่ ไม่งั้นตัวใหม่ไปติด lock ของตัวเดิม
    assert "docker stop" in text
    assert "DOWNLOAD_MAX_ATTEMPTS" in text, "ต้องมีเพดานรอบ ไม่วนไม่รู้จบ"


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_the_socket_read_has_a_timeout(controllers, kind):
    """ปล่อยให้ค่า default ของ huggingface_hub ตัดสินใจเองไม่พอ — ระบุให้ชัด"""
    assert "HF_HUB_DOWNLOAD_TIMEOUT" in controllers[kind]


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_docker_format_strings_survive_templating(controllers, kind):
    """`{{.State.Running}}` ของ docker ชนกับไวยากรณ์ของ jinja — ต้องออกมาเป็นของ docker

    เจอจริงตอนเพิ่มตัวเฝ้า: escape ผิดที่ (โค้ดอยู่ใน {% raw %} อยู่แล้ว) ทำให้ไฟล์ที่เจน
    ออกมามี `{{ '{{.State.Running}}' }}` ตรง ๆ ซึ่ง docker อ่านไม่ออก
    """
    text = controllers[kind]
    assert "{{.State.Running}}" in text
    assert "'{{.State.Running}}' }}" not in text, "escape เกิน — jinja ไม่ได้คายออกมาให้"
