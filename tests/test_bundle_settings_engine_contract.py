"""ค่าที่ `lmds set` ยอมบันทึก ต้องเป็นค่าที่ controller ของ engine นั้นรับได้และอ่านจริง — audit 2026-10-06

สองช่องว่างระหว่าง `bundle_settings` กับ controller ที่ generate ออกมา:

  * **gpu-util** — `lmds set --gpu-util` รับ (0, 1] และช่องบนหน้าเว็บรับ 0.1–0.98 แต่ controller ของ vLLM / SGLang / stacked
    รับ 0.3–0.98 และตรวจค่านี้ใน **ทุกคำสั่ง** · หลัง `lmds set --gpu-util 0.99` (หรือ 0.1–0.29 จากหน้าเว็บ) `stop` `status`
    `logs` ออกด้วย exit 1 "invalid --gpu-util" — โมเดลที่รันอยู่หยุดผ่าน controller ไม่ได้
  * **image บน SGLang** — `FIELDS["image"]` เขียนแค่ `VLLM_IMAGE` กับ `LLAMACPP_IMAGE` · `lmds set --image` บน bundle SGLang
    จึงบันทึกสำเร็จ โชว์ในหน้าจอ แต่ controller ซึ่งอ่าน `SGLANG_IMAGE` ไม่เคยเห็น

เทสรันบรรทัดจริงจาก controller ที่ render จริงใต้ bash — ไม่ได้เทียบตัวเลข/ชื่อที่คัดลอกมาไว้ในเทส: ถ้า template เปลี่ยนช่วง
หรือเปลี่ยนชื่อ env วันไหน เทสนี้จะแดงจนกว่า `bundle_settings` จะตามทัน
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fleet import manager
from lmds.fleet.bundle_settings import FILENAME, SOURCE_BLOCK, SettingsError, read, write
from lmds.generator import render_bundle
from tests.test_generator import safetensors_report

# ขอบของช่วง + ค่าที่ audit ใช้รีโปร (0.99 จาก CLI · 0.1/0.29 จากช่องบนหน้าเว็บ) + ค่าที่ fit/sizing เขียนเอง (0.30 · 0.98)
GPU_UTIL_CASES = ["0.05", "0.1", "0.29", "0.3", "0.30", "0.65", "0.85", "0.98", "0.99", "1", "1.0"]


def _render(tmp_path: Path, engine: Engine, target: str = "dgx-spark-single"):
    report = safetensors_report()
    fit = analyze(report, PRESETS[target])
    return render_bundle(build_plan(report, fit, provider=None, engine=engine), report, fit, tmp_path / "bundles")


@pytest.fixture(params=[(Engine.VLLM, "dgx-spark-single"), (Engine.SGLANG, "dgx-spark-single"),
                        (Engine.VLLM, "dgx-spark-stacked")], ids=["vllm", "sglang", "stacked"])
def rendered(request, tmp_path):
    engine, target = request.param
    return _render(tmp_path, engine, target)


def _controller_accepts_gpu_util(controller: Path, value: str) -> bool:
    """รันบรรทัดตรวจ gpu-util ของ controller ตัวนี้เอง (awk ที่มัน `|| die "invalid --gpu-util"`) กับค่าที่ให้"""
    text = controller.read_text(encoding="utf-8")
    checks = re.findall(r"""^\s*(awk -v v="\$GPU_MEMORY_UTILIZATION" '[^']+')""", text, re.M)
    assert checks, f"{controller.name} ไม่มีบรรทัดตรวจ gpu-util แบบที่เทสนี้รู้จัก — อัปเดตเทสให้ตรง template"
    done = subprocess.run(["bash", "-c", checks[0]], env={"PATH": "/usr/bin:/bin", "GPU_MEMORY_UTILIZATION": value},
                          capture_output=True, text=True)
    return done.returncode == 0


@pytest.mark.parametrize("value", GPU_UTIL_CASES)
def test_set_accepts_a_gpu_util_exactly_when_the_generated_controller_does(rendered, value):
    """ค่าที่ controller ปฏิเสธ ต้องถูกปฏิเสธตั้งแต่ตอนบันทึก (และไม่มีอะไรถูกเขียน) · ค่าที่ controller รับ ต้องบันทึกได้"""
    accepted_by_controller = _controller_accepts_gpu_util(rendered.controller, value)
    try:
        write(rendered.directory, {"gpu_util": value})
        accepted_by_set = True
    except SettingsError as exc:
        accepted_by_set = False
        assert "0.3" in str(exc) and "0.98" in str(exc), "ข้อความต้องบอกช่วงที่ใช้ได้"
        assert not (rendered.directory / FILENAME).exists()
    assert accepted_by_set == accepted_by_controller, (
        f"gpu_util={value}: lmds set {'รับ' if accepted_by_set else 'ปฏิเสธ'} "
        f"แต่ controller {'รับ' if accepted_by_controller else 'ปฏิเสธ'}")


def test_cli_set_gpu_util_out_of_range_is_refused_with_the_range_and_nothing_is_written(tmp_path, monkeypatch):
    """รีโปรของ auditor ผ่าน CLI: `lmds set <slug> --gpu-util 0.99`"""
    from lmds.cli.main import app

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(manager, "_container_running", lambda name: False)
    monkeypatch.setattr(manager, "_orphan_docker", lambda known: [])
    monkeypatch.setattr(manager, "_pgrep_llama", lambda: [])
    built = _render(tmp_path, Engine.VLLM)
    manager.register_bundle(built.controller)
    slug = built.directory.name

    for bad in ("0.99", "0.1"):
        result = CliRunner().invoke(app, ["set", slug, "--gpu-util", bad, "--port", "8001"], env={"COLUMNS": "200"})
        assert result.exit_code == 1, result.output
        assert "0.3" in result.output and "0.98" in result.output
        assert not (built.directory / FILENAME).exists(), "ค่าที่ถูกปฏิเสธต้องไม่ทิ้ง port ที่สั่งมาพร้อมกันไว้ครึ่งทาง"
    good = CliRunner().invoke(app, ["set", slug, "--gpu-util", "0.98"], env={"COLUMNS": "200"})
    assert good.exit_code == 0, good.output
    assert read(built.directory)["gpu_util"] == "0.98"


def test_a_value_saved_before_the_range_was_enforced_is_still_shown_so_it_can_be_fixed(tmp_path):
    """bundle.env ที่มี 0.99 ค้างจากรุ่นก่อน: `lmds set <slug>` ต้องยังโชว์ค่านั้น (ซ่อน = ผู้ใช้หาไม่เจอว่าทำไม stop ล้ม)
    และตั้งค่าที่ถูกทับได้"""
    built = _render(tmp_path, Engine.VLLM)
    (built.directory / FILENAME).write_text('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.99}"\n', encoding="utf-8")
    assert read(built.directory)["gpu_util"] == "0.99"
    assert write(built.directory, {"gpu_util": "0.9"})["gpu_util"] == "0.9"
    assert _controller_accepts_gpu_util(built.controller, read(built.directory)["gpu_util"])


def test_the_web_gpu_util_field_offers_only_what_the_controller_accepts(tmp_path):
    """ช่อง gpu-util บนการ์ด node: ขอบ min/max ที่เบราว์เซอร์บังคับต้องเป็นค่าที่ controller รับ — เดิม min=0.1"""
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    fields = re.findall(r'<input type="number"[^>]*class="n-gpu"[^>]*>', page, re.S)
    assert fields, "ไม่พบช่อง gpu-util (.n-gpu) บนหน้าเว็บ"
    built = _render(tmp_path, Engine.VLLM)
    for field in fields:
        low, high = re.search(r'min="([0-9.]+)"', field).group(1), re.search(r'max="([0-9.]+)"', field).group(1)
        assert _controller_accepts_gpu_util(built.controller, low), f"min={low} ถูก controller ปฏิเสธ"
        assert _controller_accepts_gpu_util(built.controller, high), f"max={high} ถูก controller ปฏิเสธ"
        write(built.directory, {"gpu_util": low})
        write(built.directory, {"gpu_util": high})


# ═════════════════════ --image ต้องไปถึง env ที่ controller ของ engine นั้นอ่าน ═════════════════════
@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG], ids=["vllm", "sglang"])
def test_a_saved_image_reaches_the_variable_that_engines_controller_runs(tmp_path, engine):
    """`lmds set --image X`: source bundle.env (บล็อกเดียวกับ controller) แล้วประเมินบรรทัด `<ENGINE>_IMAGE="${…:-default}"`
    **ของ controller ตัวนั้นเอง** — ต้องได้ X · SGLang เดิมได้ image ของ bundle เพราะไม่มีใครเขียน SGLANG_IMAGE"""
    built = _render(tmp_path, engine)
    text = built.controller.read_text(encoding="utf-8")
    [line] = re.findall(r'^[A-Z]+_IMAGE="\$\{[A-Z]+_IMAGE:-[^\n]*$', text, re.M)
    name = line.split("=", 1)[0]

    saved = write(built.directory, {"image": "registry.local/custom:1"})
    assert saved["image"] == "registry.local/custom:1"
    script = tmp_path / "probe.sh"
    script.write_text(f'BUNDLE_ENV="{built.directory / FILENAME}"\n' + SOURCE_BLOCK + line + f'\necho "IMAGE=${{{name}}}"\n',
                      encoding="utf-8")
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "IMAGE=registry.local/custom:1", f"{name} ไม่ได้ค่าที่บันทึก: {done.stdout!r}"
    # เอาออก = กลับไปใช้ image ของ bundle — ทุกชื่อที่เขียนไว้ต้องหายพร้อมกัน
    write(built.directory, {"image": ""})
    assert "image" not in read(built.directory) and not (built.directory / FILENAME).exists()
