from lmds.hardware import MemoryModel, TargetProfile, classify, lookup_gpu
from lmds.hardware.profiler import DetectedGpu


def test_lookup_team_test_hardware():
    """เครื่องทดสอบจริงของทีมต้องอยู่ใน allowlist และ tested=True"""
    pro4000 = lookup_gpu("NVIDIA RTX PRO 4000 Blackwell")
    assert pro4000 is not None and pro4000.tested and pro4000.vram_gb == 24.0

    s4070 = lookup_gpu("NVIDIA GeForce RTX 4070 SUPER")
    assert s4070 is not None and s4070.tested

    ti4070 = lookup_gpu("NVIDIA GeForce RTX 4070 Ti SUPER")
    assert ti4070 is not None and ti4070.tested and ti4070.vram_gb == 16.0

    spark = lookup_gpu("NVIDIA GB10")
    assert spark is not None and spark.memory_model is MemoryModel.UNIFIED


def test_ti_super_not_shadowed_by_super():
    """'4070 Ti SUPER' ต้อง match แถว Ti (16GB) ไม่ใช่แถว 4070 SUPER (12GB)"""
    gpu = lookup_gpu("NVIDIA GeForce RTX 4070 Ti SUPER")
    assert gpu is not None and gpu.vram_gb == 16.0


def test_classify_profiles():
    assert classify(["NVIDIA GB10"]) is TargetProfile.DGX_SPARK_SINGLE
    assert classify(["NVIDIA GeForce RTX 4070 SUPER"]) is TargetProfile.RTX_SINGLE
    assert (
        classify(["NVIDIA RTX PRO 4000 Blackwell", "NVIDIA RTX PRO 4000 Blackwell"])
        is TargetProfile.RTX_MULTI_GPU
    )
    assert classify([]) is TargetProfile.UNKNOWN


def test_unknown_gpu_flagged_conservative():
    gpu = DetectedGpu(name="NVIDIA TITAN FUTURE 999", vram_mib=99999, compute_capability="15.0", known=lookup_gpu("TITAN FUTURE"))
    assert gpu.tested is False


def test_full_rtx_lineup_recognized():
    """RTX 30/40/50 series ที่เพิ่มเข้ามาต้องอยู่ใน allowlist (ยังไม่ tested → conservative)"""
    for smi_name in [
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5070 Ti",
        "NVIDIA GeForce RTX 4080 SUPER",
        "NVIDIA GeForce RTX 4060 Ti",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3060 Ti",
        "NVIDIA GeForce RTX 3050",
    ]:
        gpu = lookup_gpu(smi_name)
        assert gpu is not None, f"{smi_name} ควรอยู่ใน allowlist"
        assert gpu.memory_model is MemoryModel.DISCRETE
        assert gpu.tested is False


def test_hardware_validated_gpus_are_not_conservative():
    """GPU ที่รันจริงแล้วต้องไม่โดนหัก budget — ห้ามพลิกกลับโดยไม่ตั้งใจ

    RTX 5090: hardware-validated 2026-08-03 (gemma-4-12b-it UD-Q8_K_XL + vision)
    """
    for smi_name in [
        "NVIDIA GB10",
        "NVIDIA RTX PRO 4000 Blackwell",
        "NVIDIA GeForce RTX 4070 Ti SUPER",
        "NVIDIA GeForce RTX 4070 SUPER",
        "NVIDIA GeForce RTX 5090",
    ]:
        gpu = lookup_gpu(smi_name)
        assert gpu is not None, f"{smi_name} ควรอยู่ใน allowlist"
        assert gpu.tested is True, f"{smi_name} เคยรันจริงแล้ว ไม่ควรกลับไปโหมด conservative"


def test_ti_variants_not_shadowed_by_base():
    """'<รุ่น> Ti' ต้อง match แถว Ti ไม่ถูกแถวฐาน (ตัวเลข VRAM ต่างกัน) จับก่อน"""
    # 5070 Ti (16GB) ต้องไม่ตกไปที่ 5070 (12GB)
    assert lookup_gpu("NVIDIA GeForce RTX 5070 Ti").vram_gb == 16.0
    assert lookup_gpu("NVIDIA GeForce RTX 5070").vram_gb == 12.0
    # 3080 Ti (12GB) ต้องไม่ตกไปที่ 3080 (10GB)
    assert lookup_gpu("NVIDIA GeForce RTX 3080 Ti").vram_gb == 12.0
    assert lookup_gpu("NVIDIA GeForce RTX 3080").vram_gb == 10.0
    # 3090 Ti / 3090 แยกแถวกัน (VRAM เท่ากันแต่ต้องไม่บังกัน)
    assert lookup_gpu("NVIDIA GeForce RTX 3090 Ti") is not None
    assert lookup_gpu("NVIDIA GeForce RTX 3090").name_pattern == "rtx 3090"


def test_compute_capability_by_architecture():
    """compute capability คงที่ตามสถาปัตยกรรมทั้งไลน์"""
    assert lookup_gpu("NVIDIA GeForce RTX 3070").compute_capability == "8.6"   # Ampere
    assert lookup_gpu("NVIDIA GeForce RTX 4070").compute_capability == "8.9"   # Ada
    assert lookup_gpu("NVIDIA GeForce RTX 5070").compute_capability == "12.0"  # Blackwell


def test_classify_new_rtx_single_and_multi():
    assert classify(["NVIDIA GeForce RTX 5090"]) is TargetProfile.RTX_SINGLE
    assert (
        classify(["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"])
        is TargetProfile.RTX_MULTI_GPU
    )


def test_detect_disk_reports_free_and_total(tmp_path):
    """FR-2.1: ต้องรายงานพื้นที่ดิสก์ — ดิสก์เต็มคือสาเหตุ deploy ล้มที่พบบ่อยสุด"""
    from lmds.hardware.profiler import detect_disk_gb

    free, total = detect_disk_gb(str(tmp_path))
    assert free is not None and total is not None
    assert 0 < free <= total


def test_detect_disk_unreadable_path_is_none():
    from lmds.hardware.profiler import detect_disk_gb

    assert detect_disk_gb("/ไม่มีจริง/path/นี้") == (None, None)


def test_probe_warns_when_disk_low(monkeypatch):
    from lmds.hardware import profiler

    monkeypatch.setattr(profiler, "detect_gpus", lambda: ([], []))
    monkeypatch.setattr(profiler, "detect_docker", lambda: (True, True))
    monkeypatch.setattr(profiler, "detect_ram_gb", lambda: 128.0)
    monkeypatch.setattr(profiler, "detect_disk_gb", lambda: (12.0, 500.0))

    report = profiler.probe()
    assert report.disk_free_gb == 12.0
    assert any("ดิสก์เหลือ" in note for note in report.notes)


def test_toolkit_counts_when_binary_exists_but_runtime_is_not_registered(monkeypatch):
    """เคสจริง RTX4000 (Docker 29 + toolkit 1.20): --gpus all ใช้ได้ แต่ docker info ไม่มี runtime nvidia
    → เดิมรายงาน ❌ แล้วสั่งให้ไปลง toolkit ที่ลงอยู่แล้ว"""
    import shutil as _shutil

    from lmds.hardware import profiler
    monkeypatch.setattr(profiler, "_run", lambda cmd, timeout=15: '{"runc":{"path":"runc"}}')
    monkeypatch.setattr(_shutil, "which", lambda n: "/usr/bin/" + n if n in ("docker", "nvidia-ctk") else None)
    assert profiler.detect_docker() == (True, True)
    # ไม่มีทั้ง runtime ทั้ง binary → ❌ จริง
    monkeypatch.setattr(_shutil, "which", lambda n: "/usr/bin/docker" if n == "docker" else None)
    monkeypatch.setattr(profiler.Path, "is_file", lambda self: False)
    assert profiler.detect_docker() == (True, False)


def test_suggest_target_maps_cards_to_a_real_preset():
    from lmds.fit import PRESETS
    from lmds.hardware.profiler import DetectedGpu, suggest_target

    g = lambda name: DetectedGpu(name=name, vram_mib=24467, compute_capability="12.0")
    assert suggest_target([g("NVIDIA RTX PRO 4000 Blackwell"), g("NVIDIA RTX PRO 4000 Blackwell")]) == "rtx-pro-4000-dual"
    assert suggest_target([g("NVIDIA RTX PRO 4000 Blackwell")]) == "rtx-pro-4000"
    assert suggest_target([g("NVIDIA GeForce RTX 5090")]) == "rtx-5090"
    assert suggest_target([g("NVIDIA GeForce RTX 4070 Ti SUPER")]) == "rtx-4070-ti-super"
    assert suggest_target([g("NVIDIA GB10")]) == "dgx-spark-single"
    assert suggest_target([g("Tesla T4")]) == ""
    for name in ("rtx-pro-4000-dual", "rtx-5090", "rtx-4070-ti-super"):
        assert name in PRESETS
