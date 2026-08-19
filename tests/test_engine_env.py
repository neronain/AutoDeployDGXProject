"""knob ของ engine ที่อ่านจาก environment ล้วน ๆ ต้องตั้งได้โดยไม่ต้องแก้สคริปต์มือ

เคสจริง 2026-08-20 บน msi-6: NVFP4 บน GB10 ต้องได้ VLLM_NVFP4_GEMM_BACKEND=marlin
ไม่งั้น vLLM ไป JIT cutlass FP4 kernel แล้ว ptxas ปฏิเสธ (`cvt .e2m1x2` ไม่มีบน sm_121)
engine core ตายก่อน health · ก่อนมีช่องนี้ทางเดียวคือแก้ controller ด้วยมือ
ซึ่งหายทุกครั้งที่ rebuild
"""

import pytest

from lmds.fleet.bundle_settings import FIELDS, SettingsError, _clean


def test_engine_env_is_a_saveable_knob():
    assert FIELDS["engine_env"] == ("ENGINE_ENV",)


def test_accepts_the_pair_that_fixes_nvfp4_on_gb10():
    assert _clean("engine_env", "VLLM_NVFP4_GEMM_BACKEND=marlin") == \
        "VLLM_NVFP4_GEMM_BACKEND=marlin"


def test_accepts_several_pairs():
    value = _clean("engine_env", "A=1   B=2\tC=3")
    assert value == "A=1 B=2 C=3"


@pytest.mark.parametrize("bad", [
    "NOEQUALS",
    "1BAD=x",
    "KEY=va lue",          # ช่องว่างในค่าจะถูกแตกเป็นคนละคู่
])
def test_rejects_shapes_that_would_break_the_controller(bad):
    with pytest.raises(SettingsError):
        _clean("engine_env", bad)


@pytest.mark.parametrize("bad", ["K=$(id)", "K=`id`", "K=a'b", 'K=a"b', "K=a\\b"])
def test_rejects_values_the_shell_would_interpret(bad):
    """controller แตกค่านี้ในเชลล์ — อักขระที่ถูกตีความคือช่องทางรันคำสั่ง"""
    with pytest.raises(SettingsError):
        _clean("engine_env", bad)


def test_every_controller_template_passes_it_to_the_engine():
    """ตั้งค่าได้แต่ engine ไม่เห็น = knob หลอก"""
    from pathlib import Path

    import lmds.generator as generator

    templates = Path(generator.__file__).parent / "templates"
    for path in sorted(templates.glob("*controller.sh.j2")):
        body = path.read_text()
        assert "ENGINE_ENV" in body, f"{path.name} ไม่ได้ส่ง ENGINE_ENV ต่อ"
