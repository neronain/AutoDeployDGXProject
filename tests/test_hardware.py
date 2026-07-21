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
