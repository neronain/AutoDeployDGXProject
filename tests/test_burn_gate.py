"""burn gate — จับ GPU clock latch ของ GB10 ที่ `nvidia-smi` มองไม่เห็น (field notes §10)

อาการจริง (artifact-backed, Tech2wild 2026-09-10): EC ล็อก GPU ไว้ 631–949 MHz ขณะที่เครื่องอื่น
ในชุดเดียวกันวิ่ง 2177–2561 MHz · `nvidia-smi` ไม่แสดงอะไรผิดเลย · **รีบูตไม่หาย** ต้องถอดปลั๊ก
30–60 วินาที · ใน TP มันลากทั้งคลัสเตอร์เพราะทุก collective รอ rank ที่ช้าที่สุด

ชุดนี้คุมสามเรื่องที่ทำให้ฟีเจอร์นี้มีประโยชน์จริงแทนที่จะเป็นไฟแดงอีกดวง:

1. **สามสภาพที่ห้ามปนกัน** — latched (ถอดปลั๊ก) · throttled (ไม่ใช่ latch) · GPU ไม่ว่าง
   (ไม่ใช่ความผิดของเครื่อง) · บอกผิดข้อไหนก็คือส่งช่างไปทำผิดเรื่อง
2. **"ไม่เกี่ยว" ต้องไม่ใช่ "ตก"** — เครื่องที่ไม่ใช่ GB10 / ไม่มี NVIDIA
3. **สคริปต์ทำงานได้จริง** — เดินทั้งเส้นทาง (เชลล์ → heredoc → interpreter → JSON) ด้วย
   nvidia-smi ปลอมกับ torch ปลอม · ไม่แตะ GPU จริง ไม่ SSH ออกไปไหน
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.hardware import burn  # noqa: E402


def answer(payload: dict, *, gpu: str = "NVIDIA GB10") -> str:
    """stdout แบบที่เครื่องจริงคืนมา — บรรทัด GPU ก่อน แล้วค่อยบรรทัดผล"""
    return f"LMDS_BURN_GPU {gpu}\nLMDS_BURN {json.dumps({'gpu': gpu, **payload})}\n"


def runner_for(stdout: str, *, exit_code: int = 0, stderr: str = ""):
    """runner ปลอมรูปเดียวกับ `lmds.nodes.run` — จำคำสั่งที่ถูกส่งไปด้วย"""
    sent: list[str] = []

    def run(node, command, timeout=60, stdin_text=""):
        sent.append(command)
        return SimpleNamespace(ok=exit_code == 0, exit_code=exit_code,
                               stdout=stdout, stderr=stderr)

    run.sent = sent
    return run


NODE = SimpleNamespace(name="spark-head", host="10.2.1.9", user="lmds", port=22)

HEALTHY = {"tflops": 84.0, "clock_sm_mhz": 2321.0, "clock_gr_mhz": 2321.0,
           "power_w": 91.0, "temperature_c": 58.0, "throttle": [], "seconds": 15.0, "matrix": 4096}
LATCHED = {"tflops": 4.6, "clock_sm_mhz": 781.0, "clock_gr_mhz": 781.0,
           "power_w": 14.0, "temperature_c": 41.0, "throttle": [], "seconds": 15.0, "matrix": 4096}


# ── สามสภาพที่ห้ามปนกัน ────────────────────────────────────────────────────────
def test_a_latched_gb10_is_named_and_the_only_fix_offered_is_the_physical_one():
    result = burn.check_node(NODE, runner=runner_for(answer(LATCHED)))

    assert result["kind"] == "latched"
    assert result["ok"] is False
    assert result["node"] == "spark-head"
    steps = "\n".join(result["remedy"])
    # ขั้นตอนต้องเป็นสิ่งที่คนยืนหน้าเครื่องทำได้ ไม่ใช่คำสั่งให้พิมพ์
    assert "ถอดปลั๊ก" in steps
    assert "30-60" in steps
    # และต้องบอกด้วยว่าทางที่คนจะลองก่อนเป็นอันดับแรก (รีบูต) ใช้ไม่ได้
    assert "รีบูต" in steps and "ไม่หาย" in steps


def test_a_thermal_throttle_is_not_reported_as_the_ec_latch():
    """latch ของ EC ไม่รายงาน clock event reason เลย (§10) — มี reason = คนละเรื่อง

    ส่งช่างไปถอดปลั๊กทั้งที่จริง ๆ ครีบระบายความร้อนตัน คือความเสียหายที่ฟีเจอร์นี้สร้างเอง
    """
    hot = {**LATCHED, "throttle": ["hw_thermal_slowdown"], "temperature_c": 94.0}
    result = burn.check_node(NODE, runner=runner_for(answer(hot)))

    assert result["kind"] == "throttled"
    assert result["ok"] is False
    steps = "\n".join(result["remedy"])
    # ห้ามมีขั้นตอนถอดปลั๊ก — และต้องบอกตรง ๆ ว่าทางนั้นไม่ช่วย เพราะคนที่เคยเจอ latch มาก่อน
    # จะเดาเองว่าให้ถอด
    assert "ถอดปลั๊ก power adapter" not in steps
    assert "ถอดปลั๊กไม่ช่วย" in steps
    assert "hw_thermal_slowdown" in steps


def test_a_gpu_that_is_merely_busy_is_not_called_faulty():
    """คล็อกขึ้นปกติ ไฟสูง แต่ TFLOPS ต่ำ = มีโมเดลใช้ GPU อยู่ ไม่ใช่เครื่องเสีย"""
    busy = {**HEALTHY, "tflops": 21.0}
    result = burn.check_node(NODE, runner=runner_for(answer(busy)))

    assert result["kind"] == "contended"
    assert "ถอดปลั๊ก" not in "\n".join(result["remedy"])
    assert "lmds stop" in "\n".join(result["remedy"])


def test_a_failure_with_no_clock_or_power_reading_admits_it_cannot_say_why():
    """ไดรเวอร์บางรุ่นตอบ [N/A] ทั้งสองช่อง — ตกจริงแต่เดาสาเหตุให้ไม่ได้ ต้องพูดตรง ๆ

    เรียกมันว่า contended = อ้างว่าคล็อกขึ้นปกติทั้งที่ไม่รู้ · เรียกว่า latched = ส่งคนไปถอดปลั๊ก
    โดยไม่มีหลักฐาน · ทั้งสองทางคือการเดาแทนผู้ใช้ในเรื่องที่มีค่าใช้จ่ายทางกายภาพ
    """
    blind = {**LATCHED, "tflops": 12.0, "clock_sm_mhz": None, "clock_gr_mhz": None, "power_w": None}
    result = burn.check_node(NODE, runner=runner_for(answer(blind)))

    assert result["kind"] == "slow"
    assert result["ok"] is False
    assert "บอกสาเหตุไม่ได้" in result["summary"]
    steps = result["remedy"]
    stop_first = next(i for i, s in enumerate(steps) if "lmds stop" in s)
    unplug_last = next(i for i, s in enumerate(steps) if "ถอดปลั๊ก adapter" in s)
    assert stop_first < unplug_last, "ต้องให้ตัดเรื่อง GPU ไม่ว่างออกก่อนถึงจะสั่งถอดปลั๊ก"
    assert "อย่าเพิ่งสั่งถอดปลั๊ก" in steps[0]


def test_every_verdict_speaks_both_languages_because_the_console_is_english():
    """`nodes/doctor` ตั้งแบบแผนไว้แล้ว: CLI พูดไทย หน้าเว็บ/รายงานพูดอังกฤษ จากตารางเดียว"""
    result = burn.check_node(NODE, runner=runner_for(answer(LATCHED)))

    assert "latched" in result["summary_en"]
    assert result["summary_en"] != result["summary"]
    assert "unplug the power adapter" in "\n".join(burn.remedy(result, "en")).lower()


def test_the_allowlist_in_profiles_is_what_decides_which_machines_this_applies_to():
    """ไม่ตั้งกติกาใหม่ซ้อนของเดิม — unified memory ใน `hardware/profiles` คือ Spark อยู่แล้ว"""
    assert burn.applies_to("NVIDIA GB10") is True
    assert burn.applies_to("DGX Spark") is True
    assert burn.applies_to("NVIDIA RTX 6000 Ada Generation") is False
    assert burn.applies_to("") is False


def test_a_healthy_gb10_passes():
    result = burn.check_node(NODE, runner=runner_for(answer(HEALTHY)))

    assert result["kind"] == "ok"
    assert result["ok"] is True
    assert result.get("suspect") is False
    assert "84.0 TFLOPS" in result["summary"]


def test_a_pass_that_is_still_below_the_normal_band_says_so():
    """§10 บอกว่าปกติ 75–90 · 60 ผ่านเกณฑ์ตก (<50) แต่ไม่ใช่ตัวเลขของเครื่องที่สบายดี"""
    result = burn.check_node(NODE, runner=runner_for(answer({**HEALTHY, "tflops": 60.0})))

    assert result["kind"] == "ok"
    assert result["suspect"] is True


# ── "ไม่เกี่ยว" ต้องไม่ใช่ "ตก" ────────────────────────────────────────────────
@pytest.mark.parametrize("stdout, gpu", [
    ("LMDS_BURN_NOSMI\n", ""),
    ("LMDS_BURN_NOGPU\n", ""),
    ("LMDS_BURN_GPU NVIDIA RTX 6000 Ada Generation\n"
     "LMDS_BURN_SKIP NVIDIA RTX 6000 Ada Generation\n", "NVIDIA RTX 6000 Ada Generation"),
])
def test_a_machine_this_does_not_apply_to_is_not_a_failure(stdout, gpu):
    result = burn.check_node(NODE, runner=runner_for(stdout))

    assert result["kind"] == "not-applicable"
    assert result["ok"] is True
    assert result["applies"] is False
    assert "ไม่เกี่ยว" in result["summary"]
    if gpu:
        assert gpu in result["summary"]


def test_forcing_a_measurement_on_a_non_gb10_reports_the_number_without_judging_it():
    """--force บน RTX ให้ตัวเลขได้ แต่เกณฑ์ผ่าน/ตกใน §10 เป็นของ GB10 ล้วน"""
    stdout = answer({**HEALTHY, "tflops": 31.0}, gpu="NVIDIA RTX 6000 Ada Generation")
    result = burn.check_node(NODE, runner=runner_for(stdout), force=True)

    assert result["kind"] == "not-applicable"
    assert result["ok"] is True
    assert "31.0" in result["summary"]


# ── ตรวจไม่ได้ ≠ เครื่องเสีย ──────────────────────────────────────────────────
def test_a_machine_with_no_torch_anywhere_says_how_to_point_at_one():
    stdout = "LMDS_BURN_GPU NVIDIA GB10\nLMDS_BURN_NOPYTHON tried: python3 · last: python3 rc=3 no torch\n"
    result = burn.check_node(NODE, runner=runner_for(stdout))

    assert result["kind"] == "unknown"
    assert result["ok"] is True          # ตรวจไม่ได้ ไม่ใช่ตก
    assert "LMDS_BURN_PYTHON" in "\n".join(result["remedy"])
    assert "LMDS_BURN_IMAGE" in "\n".join(result["remedy"])


def test_a_gpu_too_full_to_allocate_asks_for_the_models_to_be_stopped():
    stdout = ("LMDS_BURN_GPU NVIDIA GB10\n"
              'LMDS_BURN {"error": "alloc", "detail": "CUDA out of memory", "matrix": 4096}\n')
    result = burn.check_node(NODE, runner=runner_for(stdout))

    assert result["kind"] == "unknown"
    assert "หยุดโมเดลก่อน" in result["summary"]


def test_an_unreachable_node_is_unknown_not_latched():
    """SSH ตาย ≠ เครื่องนั้น latch — บอกผิดข้อนี้คือส่งคนไปถอดปลั๊กเครื่องที่ไม่มีอะไรผิด"""
    run = runner_for("", exit_code=255, stderr="ssh: connect to host 10.2.1.9: No route to host")
    result = burn.check_node(NODE, runner=run)

    assert result["kind"] == "unknown"
    assert result["reason"] == "unreachable"
    assert result["remedy"] == []


def test_noise_around_the_answer_does_not_break_the_reading():
    """motd ของ sshd และคำเตือนของ docker แทรกมาก่อนบรรทัดของเราได้เสมอ"""
    noisy = ("Welcome to Ubuntu 24.04.3 LTS\n"
             "WARNING: The requested image's platform does not match\n"
             + answer(HEALTHY)
             + "\n")
    result = burn.check_node(NODE, runner=runner_for(noisy))

    assert result["kind"] == "ok"


# ── ทั้งฟลีต ─────────────────────────────────────────────────────────────────
def test_the_fleet_check_answers_for_every_machine_in_the_order_given():
    nodes = [SimpleNamespace(name=f"spark-{i}") for i in range(4)]
    stdout = {"spark-0": answer(HEALTHY), "spark-1": answer(LATCHED),
              "spark-2": answer(LATCHED), "spark-3": answer(HEALTHY)}

    def run(node, command, timeout=60, stdin_text=""):
        return SimpleNamespace(ok=True, exit_code=0, stdout=stdout[node.name], stderr="")

    results = burn.check_fleet(nodes, runner=run)

    assert [r["node"] for r in results] == ["spark-0", "spark-1", "spark-2", "spark-3"]
    assert [r["kind"] for r in results] == ["ok", "latched", "latched", "ok"]


# ── สัญญาที่โมดูลนี้ต้องไม่ผิด ─────────────────────────────────────────────────
def test_the_script_never_pulls_anything_from_the_network():
    """SECURITY.md สัญญาว่าไม่มีอะไรวิ่งออกเน็ตเอง และลูกค้า air-gapped มีจริง"""
    script = burn.burn_script()

    assert "docker image inspect" in script     # ใช้เฉพาะ image ที่มีอยู่แล้ว
    assert "docker pull" not in script
    assert "curl" not in script and "wget" not in script


def test_shortening_the_burn_keeps_the_sample_point_at_the_same_ratio():
    """§10 อ่านค่าที่วินาทีที่ 12 จาก 15 — ย่อ burn แล้วจุดอ่านต้องย่อตาม ไม่ใช่ค้างที่ 12"""
    assert "SAMPLE_AT = 12.0" in burn.burn_program(seconds=15.0)
    assert "SAMPLE_AT = 4.0" in burn.burn_program(seconds=5.0)


def test_the_stamp_kept_with_a_benchmark_carries_the_numbers_that_produced_the_verdict():
    """ติดป้ายไว้กับตัวเลข ไม่ใช่แค่เตือนบนจอ — คนที่เปิดดูทีหลังไม่ได้อยู่ตอนมันเตือน"""
    stamped = burn.stamp(burn.check_node(NODE, runner=runner_for(answer(LATCHED))))

    assert stamped["kind"] == "latched"
    assert stamped["ok"] is False
    assert stamped["tflops"] == 4.6
    assert stamped["clock_sm_mhz"] == 781.0
    assert stamped["gpu"] == "NVIDIA GB10"


# ── เดินทั้งเส้นทางจริง (เชลล์ + heredoc + interpreter) ───────────────────────
def _fake_gpu_box(tmp_path: Path, name: str, *, sm_mhz: int, power_w: float,
                  throttle: str = "Not Active") -> Path:
    """nvidia-smi ปลอมที่ตอบสองคำถามที่สคริปต์จริงถาม — ไม่มี GPU จริงเข้ามาเกี่ยว"""
    row = (f"{name}, {sm_mhz}, {sm_mhz}, {power_w}, 55, "
           f"{throttle}, Not Active, Not Active, Not Active")
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir(parents=True, exist_ok=True)
    smi.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f"  *clocks.sm*) printf '%s\\n' '{row}' ;;\n"
        f"  *) printf '%s\\n' '{name}' ;;\n"
        "esac\n", encoding="utf-8")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return smi.parent


def _fake_torch(tmp_path: Path, seconds_per_matmul: float) -> Path:
    """torch ปลอมที่ **เดินนาฬิกาเอง** ทีละ matmul แทนที่จะ `sleep` จริง

    ถ้าใช้ `time.sleep` ตัวเลข TFLOPS ที่ได้จะขึ้นกับว่าเครื่องที่รันเทสว่างแค่ไหน —
    เทสก็จะแดงสลับเขียวโดยที่โค้ดไม่ได้เปลี่ยนอะไร · แทนที่ `time.monotonic` ตอน import
    แล้วให้มันเดินเมื่อมี matmul เท่านั้น: เร็ว แม่นยำ และยังเดินผ่านเส้นทางจริงทั้งเส้น

    0.137439 / tick = TFLOPS ที่โปรแกรมจะคำนวณได้ (2·4096³ FLOP ต่อ matmul)
    """
    lib = tmp_path / "pylib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "torch.py").write_text(
        "import time as _time\n"
        "__version__ = '0.0-fake'\n"
        f"_TICK = {seconds_per_matmul!r}\n"
        "_now = [0.0]\n"
        "_time.monotonic = lambda: _now[0]\n"
        "float16 = 'float16'\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "    @staticmethod\n"
        "    def synchronize():\n"
        "        return None\n"
        "cuda = _Cuda()\n"
        "class _Tensor:\n"
        "    def __matmul__(self, other):\n"
        "        _now[0] += _TICK\n"
        "        return self\n"
        "def randn(*shape, device=None, dtype=None):\n"
        # ทิ้งร่องรอยไว้ว่า 'มีการแตะ GPU แล้ว' — เทสที่ต้องพิสูจน์ว่าเครื่องที่ไม่เกี่ยว
        # ไม่ถูกเผาเลย ใช้การ *ไม่มี* ไฟล์นี้เป็นหลักฐาน แทนการวัดว่าเร็วแค่ไหน
        "    import os\n"
        "    open(os.environ['LMDS_FAKE_TORCH_MARK'], 'w').close()\n"
        "    return _Tensor()\n", encoding="utf-8")
    return lib


def _run_script(tmp_path, monkeypatch, *, gpu: str, sm_mhz: int, power_w: float,
                per_matmul: float, throttle: str = "Not Active") -> dict:
    bindir = _fake_gpu_box(tmp_path, gpu, sm_mhz=sm_mhz, power_w=power_w, throttle=throttle)
    lib = _fake_torch(tmp_path, per_matmul)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(lib))
    # ชี้ interpreter ให้ชัดเพื่อไม่ให้ผลขึ้นกับว่า python3 ของเครื่องที่รันเทสเป็นตัวไหน
    monkeypatch.setenv("LMDS_BURN_PYTHON", sys.executable)
    monkeypatch.setenv("LMDS_FAKE_TORCH_MARK", str(tmp_path / "gpu-was-touched"))
    # docker ของเครื่องที่รันเทส (ถ้ามี) ต้องไม่ถูกเรียก — interpreter แรกตอบได้อยู่แล้ว
    return burn.check_local(seconds=0.25, timeout=90)


def test_the_script_really_produces_a_verdict_end_to_end_on_a_healthy_box(tmp_path, monkeypatch):
    """เชลล์ → heredoc → interpreter → JSON → verdict · tick 1.636 ms/matmul = 84 TFLOPS

    84.0 คือค่าที่ §10 วัดได้จริงบนเครื่องที่หายแล้วหลัง power-cycle
    """
    result = _run_script(tmp_path, monkeypatch, gpu="NVIDIA GB10",
                         sm_mhz=2321, power_w=91.0, per_matmul=0.00163618)

    assert result["kind"] == "ok", result
    assert result["clock_sm_mhz"] == 2321.0
    assert result["power_w"] == 91.0
    assert round(result["tflops"], 1) == 84.0


def test_the_script_really_catches_a_latched_box_end_to_end(tmp_path, monkeypatch):
    """tick 29.9 ms/matmul → 4.6 TFLOPS · คล็อก 781 MHz · 14 W = สภาพที่ §10 บันทึกไว้"""
    result = _run_script(tmp_path, monkeypatch, gpu="NVIDIA GB10",
                         sm_mhz=781, power_w=14.0, per_matmul=0.0298780)

    assert result["kind"] == "latched", result
    assert round(result["tflops"], 1) == 4.6
    assert result["tflops"] < burn.PASS_TFLOPS
    assert "ถอดปลั๊ก" in "\n".join(result["remedy"])


def test_the_script_does_not_even_burn_a_machine_the_problem_cannot_affect(tmp_path, monkeypatch):
    """เครื่องที่ไม่ใช่ GB10 ต้องออกตั้งแต่ก่อนแตะ GPU — ไม่ใช่เผา 15 วินาทีแล้วค่อยบอกว่าไม่เกี่ยว"""
    result = _run_script(tmp_path, monkeypatch, gpu="NVIDIA RTX 6000 Ada Generation",
                         sm_mhz=2505, power_w=290.0, per_matmul=0.001)

    assert result["kind"] == "not-applicable"
    assert result["gpu"] == "NVIDIA RTX 6000 Ada Generation"
    assert not (tmp_path / "gpu-was-touched").exists(), "ห้ามแตะ GPU ของเครื่องที่เรื่องนี้ไม่เกี่ยว"
