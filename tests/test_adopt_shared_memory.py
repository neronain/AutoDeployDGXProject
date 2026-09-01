"""adopt ต้องเก็บ --ipc / --shm-size มาด้วย ไม่ใช่ทิ้งไปเงียบ ๆ

เคสจริง 2026-09-01: adopt คู่ stacked (MiniMax M3 บน SGLang สองเครื่อง) แล้วสั่ง start
ใหม่ · head ตายทันทีที่ NCCL ขอ shared memory:

    Error while creating shared memory segment /dev/shm/nccl-MUxDwa
    (size 34210184), error: No space left on device (28)

เพราะ container ต้นฉบับรันด้วย --ipc host --shm-size 16g แต่สคริปต์ที่ adopt เขียนออกมา
ไม่มีทั้งคู่ → docker ให้ /dev/shm มาแค่ 64 MB ตามค่าปริยาย

worker ที่ต่อกลับมาเจอ Connection refused ก็ตายตาม และ --restart unless-stopped
ปลุก head ซ้ำทุก 10 นาทีจนครบ 31 รอบ โดยไม่มีอะไรฟ้องสักคำ
"""

import json
from unittest.mock import patch

from lmds.fleet.adopt import Adopted, render_controller, inspect_container


def _inspect_payload(ipc: str, shm: int) -> str:
    return json.dumps([{
        "Name": "/sg-mm-head",
        "Args": ["-m", "sglang.launch_server", "--tp", "2"],
        "Config": {"Image": "org/sglang:v0", "Env": ["HF_HOME=/cache"],
                   "Entrypoint": ["python3"]},
        "HostConfig": {"Binds": ["/home/u/.cache:/cache"], "PortBindings": {},
                       "NetworkMode": "host", "Runtime": "nvidia",
                       "IpcMode": ipc, "ShmSize": shm},
    }])


def _adopted(ipc: str, shm: int) -> Adopted:
    class R:
        returncode = 0
        stdout = _inspect_payload(ipc, shm)
    with patch("lmds.fleet.adopt.subprocess.run", return_value=R()):
        return inspect_container("sg-mm-head")


def test_ipc_and_shm_survive_the_round_trip():
    a = _adopted("host", 17179869184)
    assert a.ipc_mode == "host"
    assert a.shm_size == 17179869184
    script = render_controller(a, slug="m")
    assert "--ipc host" in script, "ทิ้ง --ipc host ไป — NCCL จะไม่มี shared memory พอ"
    assert "--shm-size 17179869184" in script, "ทิ้ง --shm-size ไป"


def test_default_shared_memory_does_not_clutter_the_script():
    """container ธรรมดาไม่ต้องมีสองบรรทัดนี้ — ใส่มั่วทำให้อ่านยากโดยไม่ได้อะไร"""
    script = render_controller(_adopted("private", 67108864), slug="m")
    assert "--ipc" not in script
    assert "--shm-size" not in script
