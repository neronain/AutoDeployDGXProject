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
        "NVIDIA GeForce RTX 5090",
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
