"""วัดและให้คะแนนโมเดลที่รันอยู่จริง — ความเร็ว + ความสามารถ

ต่างจากตัวเลขบนหน้าโมเดลใน Hugging Face ตรงที่ทุกอย่างในนี้วัดจาก *quant ตัวที่คุณรัน
บนเครื่องที่คุณมี ด้วย engine build ที่คุณติดตั้ง* ซึ่งเป็นสามอย่างที่เปลี่ยนผลมากที่สุด
"""

from .capability import Probe, run_probes
from .runner import BenchError, Sample, WorkloadResult, measure
from .score import capability_score, speed_summary, summarize
from .store import all_runs, bench_root, load, now_stamp, record, runs_for
from .workloads import FULL, QUICK, Workload, select

__all__ = [
    "BenchError", "FULL", "Probe", "QUICK", "Sample", "Workload", "WorkloadResult",
    "all_runs", "bench_root", "capability_score", "load", "measure", "now_stamp",
    "record", "run_probes", "runs_for", "select", "speed_summary", "summarize",
]
