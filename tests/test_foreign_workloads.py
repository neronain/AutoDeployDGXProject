"""เครื่องที่เพิ่งแอดเข้ามามักมีของรันอยู่ก่อนแล้ว

เคสจริง 2026-08-13 — msi-4 ถูกแอดเข้าฟลีตแล้วรายงานว่า "0 โมเดล" กับ
`vram_used_gb: None` ทั้งที่ container SGLang (Jackrong/Qwopus3.6-35B-A3B-Coder,
port 30000) รันมา 32 ชั่วโมงและถือ GPU ไว้ 96,073 MiB

สองอย่างประกอบกันทำให้เครื่องที่เหลือจริงไม่ถึง 20 GB ดูเหมือนว่างทั้ง 121 GB
แล้ว fit ก็จะวางแผน deploy ทับลงไปบนที่ที่ไม่มีอยู่จริง
"""

import lmds.hardware.profiler as profiler
import lmds.inventory as inventory
from lmds.hardware.profiler import DetectedGpu, compute_apps, detect_gpus


# ── ค่าจริงที่ nvidia-smi บน msi-4 (GB10, unified memory) ตอบกลับมา ──
GPU_ROW = "NVIDIA GB10, [N/A], 12.1, [N/A], 0, 52, 10.8, [N/A], [N/A], 2405, 3003, [N/A], 2405, 1, 1"
APPS_ROWS = "3150, python3, 218\n6330, sglang::scheduler, 96073"


def test_compute_apps_reads_what_the_gpu_query_cannot(monkeypatch):
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: APPS_ROWS)
    assert compute_apps() == [(3150, "python3", 218), (6330, "sglang::scheduler", 96073)]


def test_a_row_with_na_is_dropped_not_counted_as_zero(monkeypatch):
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: "1, proc, [N/A]\n2, other, 500")
    assert compute_apps() == [(2, "other", 500)]


def test_unified_memory_gpu_gets_its_used_vram_from_the_processes(monkeypatch):
    """memory.used เป็น [N/A] บน GB10 — ปล่อยเป็น None แปลว่า 'ว่างทั้งเครื่อง' ซึ่งไม่จริง"""
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    calls = []

    def fake_run(cmd, timeout=15):
        calls.append(cmd)
        return APPS_ROWS if "--query-compute-apps" in " ".join(cmd) else GPU_ROW

    monkeypatch.setattr(profiler, "_run", fake_run)
    gpus, _ = detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].vram_used_mib == 218 + 96073


def test_a_gpu_that_reports_its_own_usage_is_left_alone(monkeypatch):
    """การ์ดปกติรายงาน memory.used ได้เอง — ห้ามไปทับด้วยผลรวมของ process"""
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    row = "NVIDIA RTX 6000 Ada, 49140, 8.9, 12000, 30, 45, 100.0, 300.0, 40, 2000, 2500, 1000, 2000, 4, 16"
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: row)
    gpus, _ = detect_gpus()
    assert gpus[0].vram_used_mib == 12000


# ── งานของคนอื่นบนเครื่อง ──

def test_an_inference_server_we_did_not_start_is_reported(monkeypatch):
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [])
    monkeypatch.setattr(
        "lmds.hardware.profiler.compute_apps",
        lambda: [(6330, "sglang::scheduler", 96073)],
    )
    monkeypatch.setattr(inventory, "_docker_containers",
                        lambda: [("qwopus-sglang-safetensors", "neronain_sglang-server", "Up 32 hours")])
    monkeypatch.setattr(inventory, "_cmdline", lambda pid: "python3 -m sglang.launch_server --port 30000")

    found = inventory.foreign_workloads()
    kinds = {f["kind"] for f in found}
    assert kinds == {"process", "container"}
    assert any(f.get("vram_mib") == 96073 for f in found)


def test_work_that_is_not_an_inference_server_is_left_out(monkeypatch):
    """training job หรือ notebook ก็ถือ GPU ได้ — ไม่ควรชวนให้ adopt"""
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [])
    monkeypatch.setattr("lmds.hardware.profiler.compute_apps",
                        lambda: [(999, "python3", 4000)])
    monkeypatch.setattr(inventory, "_docker_containers",
                        lambda: [("jupyter", "jupyter/base-notebook", "Up 2 days")])
    monkeypatch.setattr(inventory, "_cmdline", lambda pid: "python3 train.py")
    assert inventory.foreign_workloads() == []


def test_our_own_bundle_is_not_reported_as_somebody_elses(monkeypatch):
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [("lmds-muse-glimmer-30b-gguf", "/x")])
    monkeypatch.setattr("lmds.hardware.profiler.compute_apps", lambda: [])
    monkeypatch.setattr(inventory, "_docker_containers",
                        lambda: [("lmds-muse-glimmer-30b-gguf", "vllm/vllm-openai:latest", "Up 1 hour")])
    monkeypatch.setattr(inventory, "_cmdline", lambda pid: "")
    assert inventory.foreign_workloads() == []



def _quiet(monkeypatch):
    monkeypatch.setattr(inventory, "_docker_containers", lambda: [])
    monkeypatch.setattr(inventory, "_cmdline", lambda pid: "")
    monkeypatch.setattr(inventory, "_managed_containers", set)
    monkeypatch.setattr(inventory, "_has_managed_ancestor", lambda pid, managed, depth=8: False)


def test_our_native_llama_server_is_not_foreign(monkeypatch):
    """เคสจริง dgx-veerasiam 2026-09-03: hub ขึ้น "นอกระบบอีก 3" ทั้งที่ทั้ง 3 คือ llama-server
    ที่ bundle ของ LMDS เอง start ไว้ — nvidia-smi เห็นเป็น process เหมือนกันหมด"""
    _quiet(monkeypatch)
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [("qwen3-coder-30b-a3b-instruct-gguf", "/x")])
    monkeypatch.setattr(inventory, "_managed_pids", lambda: {2780313})
    monkeypatch.setattr(inventory, "_container_of_pid", lambda pid: "")
    monkeypatch.setattr("lmds.hardware.profiler.compute_apps",
                        lambda: [(2780313, "llama-server", 41000), (4242, "llama-server", 30000)])
    found = inventory.foreign_workloads()
    assert [f["pid"] for f in found] == [4242]      # ตัวที่ไม่มี server.pid ยังต้องรายงาน


def test_engine_core_inside_our_container_is_not_foreign(monkeypatch):
    """vLLM แตก VLLM::EngineCore เป็น process แยก — เครื่องที่รัน vLLM ของเราจึงขึ้นซ้ำสองรายการ"""
    _quiet(monkeypatch)
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [("qwen3-coder-next-nvfp4-gb10", "/x")])
    monkeypatch.setattr(inventory, "_managed_pids", set)
    monkeypatch.setattr(inventory, "_container_of_pid",
                        lambda pid: "lmds-qwen3-coder-next-nvfp4-gb10" if pid == 103 else "rogue-vllm")
    monkeypatch.setattr("lmds.hardware.profiler.compute_apps",
                        lambda: [(103, "VLLM::EngineCore", 90000), (777, "VLLM::EngineCore", 50000)])
    found = inventory.foreign_workloads()
    assert [f["pid"] for f in found] == [777]


def test_adopted_container_counts_as_ours_even_without_lmds_prefix(monkeypatch):
    """adopt ไม่ได้เปลี่ยนชื่อ container — ชื่อจึงไม่ขึ้นต้นด้วย lmds- แต่ต้องถือว่าเป็นของเรา"""
    monkeypatch.setattr(inventory, "_running_slugs", lambda: [("vllm-gemma4", "/x")])
    monkeypatch.setattr(inventory, "_managed_containers", lambda: {"vllm-gemma4"})
    monkeypatch.setattr(inventory, "_managed_pids", set)
    monkeypatch.setattr(inventory, "_has_managed_ancestor", lambda pid, managed, depth=8: False)
    monkeypatch.setattr(inventory, "_container_of_pid", lambda pid: "vllm-gemma4")
    monkeypatch.setattr(inventory, "_cmdline", lambda pid: "")
    monkeypatch.setattr("lmds.hardware.profiler.compute_apps", lambda: [(55, "VLLM::EngineCore", 80000)])
    monkeypatch.setattr(inventory, "_docker_containers",
                        lambda: [("vllm-gemma4", "ghcr.io/aeon-7/aeon-vllm-ultimate", "Up 4 days")])
    assert inventory.foreign_workloads() == []


def test_container_of_pid_reads_the_docker_id_from_cgroup(monkeypatch, tmp_path):
    cid = "a" * 64
    monkeypatch.setattr(inventory, "_container_names_by_id", lambda: {cid: "lmds-x"})
    monkeypatch.setattr(inventory.Path, "read_text",
                        lambda self, **kw: f"0::/system.slice/docker-{cid}.scope\n")
    assert inventory._container_of_pid(123) == "lmds-x"
