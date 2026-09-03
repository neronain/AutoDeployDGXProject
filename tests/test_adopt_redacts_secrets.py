"""adopt ต้องไม่คัดลอกความลับจาก env ของ container ลงสคริปต์บนดิสก์

เคสจริง dgx-spark03 2026-09-03: `lmds adopt trtllm-nemotron` เขียน `--env HF_TOKEN=hf_…`
ลง bundles/trtllm-nemotron/trtllm-nemotron-adopted.sh (0755) · ทุก user บนเครื่องอ่านได้
และไฟล์นี้ถูก zip ส่งต่อ/push ข้ามเครื่องได้ · หลักของ LMDS คือความลับเดินทางทาง env/stdin
"""

import importlib
import inspect

adopt = importlib.import_module("lmds.fleet.adopt")   # lmds.fleet ส่งออกฟังก์ชันชื่อ adopt ด้วย


def test_secret_values_are_dropped_but_names_stay():
    kept, redacted = adopt.redact_secrets([
        "HF_TOKEN=hf_abcdefghijklmnop", "MODEL_HANDLE=nvidia/x", "VLLM_API_KEY=sk-123",
        "AWS_SECRET_ACCESS_KEY=zzz", "MAX_MODEL_LEN=4096",
    ])
    assert kept == ["HF_TOKEN", "MODEL_HANDLE=nvidia/x", "VLLM_API_KEY", "AWS_SECRET_ACCESS_KEY", "MAX_MODEL_LEN=4096"]
    assert redacted == ["HF_TOKEN", "VLLM_API_KEY", "AWS_SECRET_ACCESS_KEY"]
    assert not any("hf_abc" in k or "sk-123" in k or "zzz" in k for k in kept)


def test_render_controller_goes_through_the_redaction():
    """คุมที่ซอร์ส: บรรทัด --env ในสคริปต์ต้องมาจาก redact_secrets เท่านั้น"""
    src = inspect.getsource(adopt.render_controller)
    assert "redact_secrets(" in src
    assert "for e in meaningful_env(adopted))" not in src, "ยังเขียน env ดิบลงสคริปต์อยู่"


def test_download_then_serve_is_detected():
    """เคสจริง dgx-spark03: `hf download nvidia/X && trtllm-serve nvidia/X …` วนล้ม 15 รอบหลัง
    token หมดอายุ — adopt ต้องชี้ให้เห็นตั้งแต่ตอนสร้างสคริปต์"""
    cmd = "hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 && PYTORCH_ALLOC_CONF=x trtllm-serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --port 8355"
    assert adopt.download_before_serve(cmd) == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    assert adopt.download_before_serve("trtllm-serve /root/.cache/huggingface/hub/x --port 8355") == ""
    assert adopt.download_before_serve("") == ""


def test_render_controller_warns_about_download_before_serve():
    src = inspect.getsource(adopt.render_controller)
    assert "download_before_serve(" in src and "HF_HUB_OFFLINE" in src
