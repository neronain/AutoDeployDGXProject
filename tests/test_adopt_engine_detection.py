"""adopt ต้องรู้จักเครื่องยนต์จากคำสั่งที่รันจริง ไม่ใช่เดาจากชื่อ image คำเดียว

เคสจริง 2026-09-01: MiniMax M3 รันบน image `scitrera/dgx-spark-sglang-mm:v0` ด้วยคำสั่ง
`python3 -m sglang.launch_server --model-path … --context-length 524288` · หน้าเว็บขึ้น

    minimax-m3-sglang   unknown      :8000   running

ไม่มีทั้งชื่อเครื่องยนต์ ไม่มี context ไม่มีป้ายความสามารถ ต่างจาก bundle อื่นที่ขึ้นครบ
เพราะ adopt เขียน engine ด้วย `"vllm" if "vllm" in image else "unknown"` และอ่าน
context จาก env อย่างเดียว ทั้งที่ SGLang ส่งมาทาง argv

ยังมีอีกจุด: ชื่อรุ่นขึ้นว่า "sglang.launch_server" เพราะตัวอ่านเจอ `-m` ก่อน
`--model-path` — `-m` ของ python คือชื่อโมดูล ไม่ใช่โมเดล
"""

import json
from unittest.mock import patch

from lmds.fleet.adopt import inspect_container


def _adopted(image: str, args: list[str], env=None, entrypoint=None):
    class R:
        returncode = 0
        stdout = json.dumps([{
            "Name": "/srv",
            "Args": args,
            "Config": {"Image": image, "Env": env or [],
                       "Entrypoint": entrypoint or ["python3"]},
            "HostConfig": {"Binds": [], "PortBindings": {}, "NetworkMode": "host",
                           "Runtime": "nvidia", "IpcMode": "host", "ShmSize": 0},
        }])
    with patch("lmds.fleet.adopt.subprocess.run", return_value=R()):
        return inspect_container("srv")


SGLANG_ARGS = ["-m", "sglang.launch_server", "--model-path",
               "/cache/models--org--M3/snapshots/abc", "--tp", "2",
               "--context-length", "524288", "--served-model-name", "minimax-m3"]


def test_sglang_is_recognised_from_the_command_not_the_image_name():
    a = _adopted("scitrera/dgx-spark-sglang-mm:v0", SGLANG_ARGS)
    assert a.engine == "sglang", "ขึ้น unknown ทั้งที่คำสั่งเขียนว่า sglang.launch_server"


def test_sglang_context_comes_from_argv():
    """SGLang ส่ง --context-length ทาง argv — อ่านแต่ env จะได้ 0 แล้วหน้าเว็บว่างเปล่า"""
    assert _adopted("img:v0", SGLANG_ARGS).context == 524288


def test_python_dash_m_module_is_not_mistaken_for_the_model():
    a = _adopted("img:v0", SGLANG_ARGS)
    assert a.model != "sglang.launch_server"
    assert a.model.endswith("/snapshots/abc")


def test_vllm_and_llamacpp_still_detected():
    v = _adopted("nvcr.io/nvidia/vllm:26.08-py3",
                 ["serve", "org/m", "--max-model-len", "8192"], entrypoint=["vllm"])
    assert v.engine == "vllm" and v.context == 8192
    l = _adopted("local/llamacpp:1", ["-m", "/models/x.gguf", "-c", "4096"],
                 entrypoint=["/usr/bin/llama-server"])
    assert l.engine == "llamacpp" and l.context == 4096
    assert l.model == "/models/x.gguf", "llama.cpp ใช้ -m เป็นโมเดลจริง ต้องยังอ่านได้"


def test_an_engine_we_do_not_know_still_says_unknown():
    """เดาไม่ออกต้องบอกว่าไม่รู้ ไม่ใช่ทายมั่ว"""
    assert _adopted("acme/mystery:1", ["--serve"], entrypoint=["/app/run"]).engine == "unknown"


def test_capabilities_are_read_from_the_real_config_not_guessed(tmp_path):
    """หน้าเว็บติดป้าย vision/MoE จาก profile["features"] — adopt ไม่เคยเขียนคีย์นี้

    bundle ที่ adopt มาจึงโล่งทั้งแถว ทั้งที่ config.json อยู่บนดิสก์ให้อ่านอยู่แล้ว
    path ที่ adopt เห็นเป็นของฝั่งคอนเทนเนอร์ ต้องแปลงกลับด้วย -v ที่มันถูกรันมา
    """
    import json as _json

    from lmds.fleet.adopt import Adopted, _features_from_model

    snap = tmp_path / "hf" / "models--org--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(_json.dumps({
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "text_config": {"num_local_experts": 64, "num_experts_per_tok": 8},
        "vision_config": {"model_type": "minimax_vit"},
    }), encoding="utf-8")

    a = Adopted(container="c", image="i",
                args=["-m", "sglang.launch_server", "--model-path",
                      "/cache/models--org--m/snapshots/abc"],
                binds=[f"{tmp_path / 'hf'}:/cache"])
    f = _features_from_model(a)
    assert f["moe"] == {"experts": 64, "experts_active": 8}
    assert f["multimodal"] == {"projector": True}


def test_a_model_path_we_cannot_reach_is_left_empty(tmp_path):
    """แปลง path ไม่ได้ก็ต้องเงียบ ไม่ใช่เดาป้ายมั่วหรือระเบิด"""
    from lmds.fleet.adopt import Adopted, _features_from_model
    a = Adopted(container="c", image="i", args=["--model-path", "/nowhere/x"], binds=[])
    assert _features_from_model(a) == {}
