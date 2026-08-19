"""รับ process ที่รันตรง ๆ (ไม่ใช่ container) เข้าระบบ — lmds adopt --port/--pid

เคสนี้เจอบ่อยกว่าที่คิด: ลูกค้ารัน llama-server ใต้ systemd unit ที่เขียนเอง แล้วเพิ่งมา
ติดตั้ง LMDS ทีหลัง · `lmds ps` เห็นมันอยู่แล้วแต่ตันตรงไม่มี controller
"""

from __future__ import annotations

import subprocess

import pytest

from lmds.fleet.adopt import AdoptedProcess, adopt_process, render_native_controller

REAL_ARGV = [
    "./llama-server",
    "-m", "/home/praisit/models/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf",
    "-ngl", "99", "-c", "65536", "-np", "1",
    "-sm", "layer", "-ts", "1,1,1",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "--host", "0.0.0.0", "--port", "8080",
]


def _proc(**over) -> AdoptedProcess:
    base = dict(
        pid=122081, argv=REAL_ARGV,
        exe="/home/praisit/llama.cpp/llama-server",
        cwd="/home/praisit/llama.cpp",
        unit="llama-qwen.service",
    )
    return AdoptedProcess(**{**base, **over})


def test_the_facts_are_read_from_argv_not_guessed():
    proc = _proc()
    assert proc.engine == "llamacpp"
    assert proc.port == 8080
    assert proc.context == 65536
    assert proc.model_path.endswith("Reasoning-Distilled.Q4_K_M.gguf")
    assert proc.model == "Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M"


def test_equals_form_flags_are_understood():
    proc = _proc(argv=["llama-server", "--port=9001", "--ctx-size=4096", "-m", "/x.gguf"])
    assert proc.port == 9001
    assert proc.context == 4096


def test_the_generated_script_reruns_the_same_command(tmp_path):
    script = render_native_controller(_proc(), "qwen-adopted")

    # bash ต้องอ่านรู้เรื่องก่อน — สคริปต์ที่ syntax พังคือรับเข้ามาแล้วใช้ไม่ได้เลย
    path = tmp_path / "c.sh"
    path.write_text(script, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0

    # ทุก flag ที่ของเดิมใช้ต้องอยู่ครบ — ตกตัวเดียวคือได้คนละพฤติกรรม
    for flag in ("-ngl", "99", "-ts", "1,1,1", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "-sm"):
        assert flag in script, f"argv หาย: {flag}"
    assert "/home/praisit/llama.cpp/llama-server" in script


def test_secrets_are_never_copied_into_the_bundle():
    """/proc/<pid>/environ มี API key ของ backend — เขียนลง bundle คือทำ secret หลุด"""
    import inspect as _inspect

    #  ได้ *ฟังก์ชัน* เพราะ __init__ re-export ทับชื่อโมดูล
    from importlib import import_module

    adopt_mod = import_module("lmds.fleet.adopt")

    source = _inspect.getsource(adopt_mod.inspect_process)
    assert "environ" not in source.replace("จงใจไม่อ่าน /proc/<pid>/environ", "")


def test_the_owning_unit_is_recorded_and_blocks_a_conflicting_start():
    """unit ที่ Restart=always จะแย่ง port กลับ — ต้องบอก ไม่ใช่ปล่อยให้ start แล้วงง"""
    script = render_native_controller(_proc(), "qwen-adopted")
    assert 'OWNING_UNIT="llama-qwen.service"' in script
    assert "systemctl disable --now" in script
    assert "is-active --quiet" in script


def test_a_process_with_no_unit_says_nothing_about_units():
    script = render_native_controller(_proc(unit=""), "qwen-adopted")
    assert 'OWNING_UNIT=""' in script


def test_something_that_is_not_a_model_server_is_refused(monkeypatch):
    #  ได้ *ฟังก์ชัน* เพราะ __init__ re-export ทับชื่อโมดูล
    from importlib import import_module

    adopt_mod = import_module("lmds.fleet.adopt")
    from lmds.fleet.manager import FleetError

    monkeypatch.setattr(
        adopt_mod, "inspect_process",
        lambda **kw: AdoptedProcess(pid=99, argv=["/usr/bin/nginx", "-g", "daemon off;"],
                                    exe="/usr/bin/nginx"),
    )
    with pytest.raises(FleetError, match="ไม่ใช่ตัวเสิร์ฟโมเดล"):
        adopt_process(pid=99)


def test_the_bundle_registers_itself_so_lmds_ps_can_see_it(tmp_path, monkeypatch):
    #  ได้ *ฟังก์ชัน* เพราะ __init__ re-export ทับชื่อโมดูล
    from importlib import import_module

    adopt_mod = import_module("lmds.fleet.adopt")

    monkeypatch.setattr(adopt_mod, "inspect_process", lambda **kw: _proc())
    monkeypatch.setattr(adopt_mod, "run_root", lambda: tmp_path / "run")

    controller, proc = adopt_process(pid=122081, slug="qwen-adopted", output=tmp_path / "bundles")

    assert controller.exists() and controller.stat().st_mode & 0o111
    meta = (tmp_path / "run" / "qwen-adopted" / "server.meta").read_text()
    assert "mode=native" in meta
    assert "engine=llamacpp" in meta
    assert "port=8080" in meta
    assert f"controller={controller}" in meta

    import yaml

    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text())
    assert profile["adopted"] is True
    assert profile["serving"]["context"] == 65536
    assert profile["source_process"]["unit"] == "llama-qwen.service"
    # argv เก็บไว้ทั้งชุด — ตรวจย้อนได้ว่า bundle นี้มาจากคำสั่งอะไร
    assert profile["source_process"]["argv"] == REAL_ARGV


def test_no_download_or_verify_is_offered_because_there_is_nothing_to_download():
    """คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มีคำสั่งนั้น"""
    script = render_native_controller(_proc(), "qwen-adopted")
    assert "download)" not in script
    assert "verify-files)" not in script


def test_doctor_checks_adopted_weights_where_they_actually_are(tmp_path):
    """เจอจากหน้าเว็บจริง: doctor หา weight ที่ ~/models/<slug> ตามธรรมเนียม LMDS

    bundle ที่รับเข้าระบบชี้ไป path เดิมของเจ้าของ ผลคือขึ้น "ยังไม่มีไฟล์โมเดล" ตลอดกาล
    ทั้งที่เซิร์ฟเวอร์กำลังเสิร์ฟไฟล์นั้นอยู่ แล้วยังแนะ `lmds repair` ที่ controller ของ
    adopt ไม่มีคำสั่งนั้น — ทำตามแล้วล้มแน่นอน
    """
    from lmds.doctor.checks import Status, _check_weights

    weights = tmp_path / "Qwen3.6-35B.Q4_K_M.gguf"
    weights.write_bytes(b"x" * 2048)
    profile = {
        "adopted": True,
        "model": {"id": str(weights)},
        "runtime": {"engine": "llamacpp"},
    }

    findings = _check_weights(profile, "qwen35-a3b-opus")

    assert [f.status for f in findings] == [Status.OK]
    assert str(weights) in findings[0].detail
    assert all("repair" not in (f.fix or "") for f in findings)


def test_doctor_says_so_when_the_adopted_weights_are_gone(tmp_path):
    from lmds.doctor.checks import Status, _check_weights

    profile = {
        "adopted": True,
        "model": {"id": str(tmp_path / "หายไปแล้ว.gguf")},
        "runtime": {"engine": "llamacpp"},
    }
    findings = _check_weights(profile, "x")
    assert findings[0].status is Status.FAIL
    # ห้ามแนะ repair — bundle นี้ไม่มีคำสั่งนั้นให้รัน
    assert "repair" not in (findings[0].fix or "")


def test_console_overrides_actually_reach_the_server(tmp_path):
    """เจอจากหน้าเว็บจริง: ตั้ง context 131968 แล้วเด้งกลับ 65536 ทุกครั้ง

    controller replay argv ดิบ ๆ ซึ่งมี `-c 65536` ติดมาด้วย ค่าที่ผู้ใช้ตั้งจึงถูกทับเสมอ
    · flag พวกนี้ต้องถูกดึงออกจาก argv แล้วใส่กลับจากตัวแปร
    """
    import subprocess

    from lmds.fleet.adopt import render_native_controller, split_managed

    rest, managed = split_managed(REAL_ARGV[1:])
    assert managed == {"ctx": "65536", "host": "0.0.0.0", "port": "8080"}
    # ที่เหลือต้องครบเป๊ะ ห้ามหล่นระหว่างทาง
    assert "-ngl" in rest and "99" in rest and "-ts" in rest and "1,1,1" in rest
    assert "-ctk" in rest and "q8_0" in rest
    for flag in ("--port", "-c", "--host"):
        assert flag not in rest, f"{flag} ต้องถูกดึงออก ไม่งั้นจะทับค่าที่ตั้ง"

    script = render_native_controller(_proc(), "x")
    path = tmp_path / "c.sh"
    path.write_text(script, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0

    out = subprocess.run(["bash", str(path), "--context", "131968", "info"],
                         capture_output=True, text=True).stdout
    assert "131968" in out, "flag --context ต้องมีผลจริง"


def test_the_container_controller_is_left_alone(tmp_path):
    """flag พวกนี้เป็นของ native — container ใช้ docker run คนละเรื่อง"""
    from lmds.fleet.adopt import Adopted, render_controller

    script = render_controller(Adopted(container="c", image="i"), "c")
    assert "--context)" not in script


def test_capabilities_come_from_the_running_server(tmp_path, monkeypatch):
    """ไม่ต้อง deploy ใหม่เพื่อให้ป้ายความสามารถขึ้น — เซิร์ฟเวอร์บอกเองได้"""
    from importlib import import_module

    adopt_mod = import_module("lmds.fleet.adopt")
    probe = {
        "props": {
            "build_info": "b10505-ee4c505a4",
            "modalities": {"vision": False, "audio": False},
            "chat_template_caps": {
                "supports_tools": True,
                "supports_parallel_tool_calls": True,
                "supports_preserve_reasoning": True,
            },
        },
        "models": {"data": [{"meta": {"n_ctx_train": 262144, "n_params": 34660610688,
                                      "size": 21155768832, "ftype": "Q4_K - Medium"}}]},
    }
    monkeypatch.setattr(adopt_mod, "inspect_process", lambda **kw: _proc())
    monkeypatch.setattr(adopt_mod, "probe_server", lambda *a, **k: probe)
    monkeypatch.setattr(adopt_mod, "run_root", lambda: tmp_path / "run")

    controller, _ = adopt_mod.adopt_process(pid=1, slug="x", output=tmp_path / "bundles")

    import yaml

    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text())
    feats = profile["features"]
    assert feats["tool_calling"]["enabled"] is True
    assert feats["tool_calling"]["parallel"] is True
    assert feats["reasoning"]["enabled"] is True
    assert feats["multimodal"]["modalities"] == ["text"]
    # เพดานของตัวโมเดล ไม่ใช่ค่าที่สั่งรันครั้งนี้ — คอนโซลใช้บอกว่าเพิ่ม context ได้ถึงไหน
    assert profile["model"]["native_context"] == 262144
    assert profile["limits"]["context_tokens"] == 262144
    assert profile["serving"]["context"] == 65536
    assert profile["runtime"]["build"] == "b10505-ee4c505a4"


def test_adopt_still_works_when_the_server_cannot_be_asked(tmp_path, monkeypatch):
    """probe ล้มต้องไม่ทำให้ adopt ล้ม — แค่ได้ข้อมูลน้อยลง"""
    from importlib import import_module

    adopt_mod = import_module("lmds.fleet.adopt")
    monkeypatch.setattr(adopt_mod, "inspect_process", lambda **kw: _proc())
    monkeypatch.setattr(adopt_mod, "probe_server", lambda *a, **k: {})
    monkeypatch.setattr(adopt_mod, "run_root", lambda: tmp_path / "run")

    controller, _ = adopt_mod.adopt_process(pid=1, slug="x", output=tmp_path / "bundles")
    assert controller.exists()

    import yaml

    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text())
    assert profile["features"]["tool_calling"]["enabled"] is False
    assert profile["model"]["native_context"] is None
