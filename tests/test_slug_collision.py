"""slug ชนกันข้ามเจ้าของ repo + controller สองแบบในโฟลเดอร์เดียว

เคสจริง 2026-09-05 ลูกค้า (cynbangkok): deploy `ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (เครื่องเดียว) แล้ว
ตามด้วย `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` แบบ stacked → slug เดียวกัน (`nvidia-nemotron-3-super-120b-a12b-nvfp4`)
โฟลเดอร์เดียวกัน มี *-single.sh (ucbye) + *-stacked.sh (nvidia) · fleet หยิบ single ก่อน → download/verify ผ่านที่ ucbye
แต่ start (stacked) หา snapshot ของ nvidia ไม่เจอ: "verify-files: OK … ERROR: ยังไม่ได้ download"
"""

from __future__ import annotations

from tests.test_review_templates import _plan, _safetensors_report
from lmds.generator import render_bundle
from lmds.generator.renderer import resolve_slug


def _render(tmp_path, repo_id, target="dgx-spark-single"):
    report = _safetensors_report(repo_id=repo_id, revision_sha="rev-" + repo_id.split("/")[0])
    plan, fit = _plan(report, target)
    return plan, render_bundle(plan, report, fit, tmp_path)


def test_same_model_name_from_another_owner_gets_its_own_bundle_folder(tmp_path):
    _, first = _render(tmp_path, "ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert first.directory.name == "nvidia-nemotron-3-super-120b-a12b-nvfp4"
    plan, second = _render(tmp_path, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", target="dgx-spark-stacked")
    assert second.directory.name == "nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia", "ต้องไม่ทับโฟลเดอร์ของ ucbye"
    assert second.controller.name == "nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia-stacked.sh"
    assert 'MODEL_ID="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"' in second.controller.read_text(encoding="utf-8")
    assert any("slug nvidia-nemotron-3-super-120b-a12b-nvfp4 เป็นของ ucbye/" in w for w in plan.warnings)
    # โฟลเดอร์แรกยังเป็นของ ucbye ทั้งโฟลเดอร์ ไม่มี controller ของ nvidia ปนเข้ามา
    assert sorted(p.name for p in first.directory.glob("*.sh")) == ["nvidia-nemotron-3-super-120b-a12b-nvfp4-single.sh"]
    assert "ucbye/" in (first.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")
    # render repo เดิมซ้ำ = slug เดิม ไม่งอกต่อท้ายไปเรื่อย
    assert resolve_slug(tmp_path, "ucbye/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4") == ("nvidia-nemotron-3-super-120b-a12b-nvfp4", "")
    assert resolve_slug(tmp_path, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")[0] == "nvidia-nemotron-3-super-120b-a12b-nvfp4-nvidia"


def test_switching_a_model_from_single_to_stacked_leaves_one_controller_and_fleet_picks_it(tmp_path):
    """โมเดลเดียวกันเปลี่ยน topology: controller เก่าถูกย้ายไป .replaced-<stamp> · discover เลือกตาม profile"""
    from lmds.fleet.manager import _pick_controller

    _, single = _render(tmp_path, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    _, stacked = _render(tmp_path, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", target="dgx-spark-stacked")
    assert stacked.directory == single.directory
    live = sorted(p.name for p in stacked.directory.glob("*-single.sh")) + sorted(p.name for p in stacked.directory.glob("*-stacked.sh"))
    assert live == ["nvidia-nemotron-3-super-120b-a12b-nvfp4-stacked.sh"], live
    assert any(p.name.startswith("nvidia-nemotron-3-super-120b-a12b-nvfp4-single.sh.replaced-") for p in stacked.directory.iterdir())
    # bundle เก่าก่อน 0.6.0 ที่ยังมีสองตัว: หยิบตามที่ MODEL_PROFILE บอก
    (stacked.directory / "nvidia-nemotron-3-super-120b-a12b-nvfp4-single.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    assert _pick_controller(stacked.directory, stacked.directory / "MODEL_PROFILE.yaml").name.endswith("-stacked.sh")
    (stacked.directory / "MODEL_PROFILE.yaml").write_text("topology: single\nmodel:\n  id: x\n", encoding="utf-8")
    assert _pick_controller(stacked.directory, stacked.directory / "MODEL_PROFILE.yaml").name.endswith("-single.sh")
