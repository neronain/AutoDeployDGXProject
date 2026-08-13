"""ค่า start ที่ตั้งบนหน้าเว็บต้องอยู่กับ bundle ไม่ใช่อยู่แค่ในเบราว์เซอร์

เคสจริง 2026-08-13 — ผู้ใช้ตั้ง port ให้โมเดลที่สองบน spark-head เป็น 8001 แล้ว:

  * กด enable autostart · reboot · โมเดลทุกตัวขึ้นที่ 8000 แล้วชนกันหมด
  * กด test-text · คำสั่งวิ่งไปหา 8000 คือโมเดลอีกตัว ไม่ใช่ตัวที่กดทดสอบ

ทั้งสองอาการมาจากรากเดียว: ค่าถูกส่งเป็น env เฉพาะตอนกดปุ่มนั้น ส่วน systemd และ
คำสั่ง test-* เรียก controller เปล่า ๆ จึงตกไปใช้ default ของ bundle
"""

import subprocess

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.fleet.bundle_settings import FILENAME, SettingsError, read, write
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _bundle(tmp_path):
    report = ModelReport(
        repo_id="unsloth/Qwen3-Coder-Next-GGUF", revision_sha="sha",
        artifact_type=ArtifactType.GGUF, weight_bytes=int(20 * GIB),
        selected_gguf="model.gguf", context_length=262144,
        kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    return render_bundle(plan, report, fit, tmp_path).directory


def _resolve(bundle_dir, env=None, home=None):
    """ให้ bash ประเมินส่วนหัวของ controller แล้วบอกว่า API_PORT ลงเอยเป็นอะไร"""
    script = next(bundle_dir.glob("*-single.sh"))
    text = script.read_text(encoding="utf-8")
    head = text[: text.index("\nMODEL_FILES=(")]
    probe = bundle_dir / "probe.sh"
    probe.write_text(head + '\necho "PORT=${API_PORT:-unset} CTX=${CTX_SIZE:-unset}"\n',
                     encoding="utf-8")
    done = subprocess.run(
        ["bash", str(probe)], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home or bundle_dir), **(env or {})},
    )
    out = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    values = dict(pair.split("=", 1) for pair in out.split() if "=" in pair)
    return values


# ── ไฟล์ที่เขียนออกมา ────────────────────────────────────────────────────────

def test_values_round_trip(tmp_path):
    saved = write(tmp_path, {"port": 8001, "context": 131072, "slots": 1})
    assert saved == {"port": "8001", "context": "131072", "slots": "1"}
    assert read(tmp_path) == saved


def test_an_api_key_is_never_written(tmp_path):
    """หน้าเว็บบอกผู้ใช้ว่า key อยู่ในเบราว์เซอร์เท่านั้น · โฟลเดอร์นี้ถูก zip แจกต่อได้"""
    write(tmp_path, {"port": 8001, "api_key": "sk-do-not-store"})
    assert "sk-do-not-store" not in (tmp_path / FILENAME).read_text(encoding="utf-8")
    assert "api_key" not in read(tmp_path)


def test_clearing_removes_the_file(tmp_path):
    write(tmp_path, {"port": 8001})
    assert (tmp_path / FILENAME).exists()
    write(tmp_path, {})
    assert not (tmp_path / FILENAME).exists()
    assert read(tmp_path) == {}


@pytest.mark.parametrize("bad", [
    {"port": 0}, {"port": 70000}, {"port": "eight"},
    {"context": -1}, {"slots": 0}, {"gpu_util": 2.0},
])
def test_values_that_would_break_start_are_refused(tmp_path, bad):
    """ปฏิเสธตั้งแต่ตอนบันทึกดีกว่าปล่อยให้ start พังตอน reboot ตอนตีสาม"""
    with pytest.raises(SettingsError):
        write(tmp_path, bad)
    assert not (tmp_path / FILENAME).exists()


# ── ค่าที่บันทึกไปถึง controller จริงไหม ────────────────────────────────────

def test_the_controller_reads_the_saved_port(tmp_path):
    """นี่คือเคสของ systemd ตอน reboot — เรียก controller โดยไม่มี env ไม่มี flag"""
    bundle = _bundle(tmp_path)
    write(bundle, {"port": 8001, "context": 131072})
    values = _resolve(bundle)
    assert values["PORT"] == "8001"
    assert values["CTX"] == "131072"


def test_the_environment_still_wins_over_the_file(tmp_path):
    bundle = _bundle(tmp_path)
    write(bundle, {"port": 8001})
    assert _resolve(bundle, env={"API_PORT": "9999"})["PORT"] == "9999"


def test_two_bundles_keep_their_own_ports(tmp_path):
    """อาการที่ผู้ใช้เจอ: หลาย bundle บนเครื่องเดียวกันไปชนกันที่ port เดียว"""
    first = _bundle(tmp_path / "a")
    second = _bundle(tmp_path / "b")
    write(first, {"port": 8000})
    write(second, {"port": 8001})
    assert _resolve(first)["PORT"] == "8000"
    assert _resolve(second)["PORT"] == "8001"
