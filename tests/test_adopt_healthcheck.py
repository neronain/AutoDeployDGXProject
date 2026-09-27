"""adopted container ต้องปิด HEALTHCHECK ที่ฝังมาใน image

เคสจริง 2026-09-27 msi-1: adopt คอนเทนเนอร์ vLLM (avarok/dgx-vllm-nvfp4-kernel:v23)
ซึ่ง image ฝัง `HEALTHCHECK curl -f http://localhost:8888/health` มาในตัว แต่ LMDS
สั่งรันบนพอร์ต 8000 → `docker ps` ขึ้น "(unhealthy)" ตลอดอายุคอนเทนเนอร์ ทั้งที่
`curl http://localhost:8000/health` ตอบ 200 และโมเดลตอบคำถามได้ปกติ

เทมเพลต controller ทุกตัวปิดทิ้งมาตั้งแต่ 2026-09-20 แล้ว แต่ทางของ adopt ตกสำรวจ
เพราะประกอบคำสั่ง docker run ขึ้นเองใน adopt.py ไม่ได้ผ่านเทมเพลต

ทำไมถึง "ปิด" ไม่ใช่ "เขียนทับด้วย --health-cmd": LMDS มี wait_health ของตัวเองที่รู้จัก
engine อยู่แล้ว (SGLang ต้องถาม /v1/models เพราะ /health ของมัน prefill จริงทุกครั้ง)
สัญญาณสุขภาพตัวที่สองที่โง่กว่าและขัดกับตัวแรกได้ แย่กว่าไม่มีสัญญาณเลย
"""

import importlib
import json
import re
import shutil
import subprocess
from unittest.mock import patch

import pytest

adopt_mod = importlib.import_module("lmds.fleet.adopt")

from lmds.fleet.adopt import Adopted, inspect_container, render_controller  # noqa: E402


def _adopted(**over) -> Adopted:
    payload = json.dumps([{
        "Name": "/coder-next",
        "Args": ["serve", "/models/X", "--port", "8000"],
        "Config": {"Image": "avarok/dgx-vllm-nvfp4-kernel:v23",
                   "Env": ["PORT=8000"], "Entrypoint": None},
        "HostConfig": {"Binds": ["/opt/models:/models"],
                       "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
                       "NetworkMode": over.get("network", "default"),
                       "Runtime": "nvidia", "IpcMode": "private", "ShmSize": 67108864},
    }])

    class R:
        returncode = 0
        stdout = payload

    with patch.object(adopt_mod.subprocess, "run", return_value=R()):
        return inspect_container("coder-next")


def _start_block(script: str) -> str:
    """เฉพาะคำสั่ง docker run ของ start — ไม่ใช่ทั้งไฟล์ที่มีคอมเมนต์ปนอยู่"""
    m = re.search(r"docker run -d[^\n]*(?:\n[^\n]*)*?\"\$\{IMAGE\}\"[^\n]*", script)
    assert m, "หา docker run ของ start ไม่เจอ — โครงสคริปต์เปลี่ยนไปแล้ว"
    return m.group(0)


def test_adopted_start_disables_the_images_healthcheck():
    block = _start_block(render_controller(_adopted(), slug="m"))
    assert "--no-healthcheck" in block


def test_it_does_not_invent_a_second_health_signal():
    """เจตนาคือ *ปิด* ไม่ใช่ย้ายพอร์ต — --health-cmd จะสร้างสัญญาณที่สองที่ขัดกับ wait_health"""
    script = render_controller(_adopted(), slug="m")
    assert "--health-cmd" not in script
    assert "--health-interval" not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")
def test_the_generated_script_still_parses():
    """ธงที่เติมเข้าไปอยู่ผิดที่ = บรรทัดก่อนหน้าจบด้วย \\ แล้วคำสั่งขาดกลางคัน

    เคยเกิดมาแล้วกับคอมเมนต์ที่แทรกกลาง docker run (2026-09-21) — คราวนั้น
    `start` ไม่ขึ้นเลยโดยไม่มีอะไรฟ้อง จึงต้องให้ bash ตรวจไวยากรณ์จริง
    """
    for network in ("default", "giant_default"):
        script = render_controller(_adopted(network=network), slug="m")
        done = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        assert done.returncode == 0, f"network={network}: {done.stderr}"


def test_the_flag_sits_on_its_own_continued_line():
    """ต่อท้าย --restart unless-stopped ในบรรทัดเดียวกันก็ได้ แต่ต้องไม่ไปคั่นระหว่าง
    บรรทัดที่ลงท้ายด้วย \\ กับตัว image"""
    block = _start_block(render_controller(_adopted(), slug="m"))
    line = next(ln for ln in block.splitlines() if "--no-healthcheck" in ln)
    assert line.rstrip().endswith("\\"), f"บรรทัดนี้ต้องต่อบรรทัด: {line!r}"
