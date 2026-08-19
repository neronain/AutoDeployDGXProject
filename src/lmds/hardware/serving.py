"""เครื่องนี้รันโมเดลเองได้ หรือเป็นแค่ control plane ที่สร้าง bundle แล้วส่งต่อ

เหตุที่ต้องมี — เคสจริง 2026-08-19:
hub VM บน OrbStack (192.168.139.92) ไม่มี GPU ไม่มี docker ไม่มี llama.cpp และ
RAM 12 GB มันมีหน้าที่เดียวคือ *สร้าง bundle แล้ว `lmds node push` ไป DGX Spark*
แต่ `lmds repair` บนนั้นเริ่มโหลด weight 15.6 GB ลงเครื่องอย่างว่าง่าย — ไฟล์ที่
ต่อให้โหลดจบก็ไม่มีอะไรรันมันได้ ต่างจาก 10.2.3.100 ที่ดูแล RTX บนตัวเอง ซึ่ง
repair/start คือสิ่งที่ถูกต้อง

โค้ดเดิมถือว่าทุกเครื่อง "ทั้งสร้างและรัน" เหมือนกันหมด ทั้งที่ docstring ของ
`lmds node push` อธิบายบทบาท controller-ไม่มี-GPU ไว้ตั้งแต่ต้น — แนวคิดมีอยู่ใน
เอกสาร แต่ไม่เคยมีอยู่ในพฤติกรรม

เราไม่เดาจากชื่อเครื่องหรือ config แต่ดูจากของที่ *มีจริง*: engine ที่รันได้
ตัวเดียวก็พอให้เป็นเครื่องรันโมเดล — ไม่มีเลยแปลว่าเป็น control plane
"""

from __future__ import annotations

import functools
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# ผู้ใช้สั่งทับได้เมื่อการตรวจอัตโนมัติเดาผิด (เช่นกำลังจะ build llama.cpp ทีหลัง)
ROLE_ENV = "LMDS_ROLE"

_SERVER_RELPATHS = ("build/bin/llama-server", "llama-server", "bin/llama-server")


def llamacpp_server(pinned: str = "") -> Path | None:
    """llama-server ที่รันได้จริงในเครื่องนี้ (ไม่ใช่แค่ซอร์สที่ clone ไว้)"""
    roots: list[Path] = []
    if pinned:
        roots.append(Path(pinned).expanduser())
    roots += [Path.home() / "src" / "llama.cpp", Path.home() / "llama.cpp"]
    for root in roots:
        for rel in _SERVER_RELPATHS:
            candidate = root / rel
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which("llama-server")
    return Path(found) if found else None


def _gpu_count() -> int:
    """นับ GPU แบบถูก ๆ — ไม่เรียก detect_gpus() เพราะมันอ่านค่าละเอียดที่เราไม่ใช้"""
    if shutil.which("nvidia-smi") is None:
        return 0
    from lmds.hardware.profiler import _run

    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not out:
        return 0
    return len([line for line in out.strip().splitlines() if line.strip()])


@dataclass
class ServingCapability:
    """engine ที่เครื่องนี้รันโมเดลได้จริง + หลักฐานว่าทำไมถึงสรุปแบบนั้น"""

    gpus: int
    docker: bool
    llamacpp: Path | None
    engines: list[str]
    forced: str = ""

    @property
    def can_serve(self) -> bool:
        return bool(self.engines)

    @property
    def is_control_plane(self) -> bool:
        return not self.engines

    def supports(self, engine: str) -> bool:
        return (engine or "").lower() in self.engines

    def evidence(self) -> str:
        """บรรทัดเดียวบอกว่าเห็นอะไรบ้าง — ให้ผู้ใช้เถียงกับข้อสรุปได้"""
        if self.forced:
            return f"{ROLE_ENV}={self.forced} (ตั้งไว้เอง)"
        parts = [f"GPU {self.gpus} ตัว" if self.gpus else "ไม่พบ GPU",
                 "docker ใช้ได้" if self.docker else "ไม่มี docker",
                 f"llama-server: {self.llamacpp}" if self.llamacpp else "ไม่พบ llama-server"]
        return " · ".join(parts)

    def refusal(self, slug: str, action: str, node_hint: str = "") -> str:
        """ข้อความตอนปฏิเสธคำสั่งที่ต้องรันบนเครื่องที่เสิร์ฟได้เท่านั้น

        ต้องบอกสามอย่างเสมอ: ทำไมถึงไม่ทำ, เห็นอะไรถึงสรุปแบบนั้น, แล้วให้ทำอะไรแทน
        """
        target = node_hint or "<เครื่องปลายทาง>"
        return (
            f"เครื่องนี้เป็น control plane — {action} ของ {slug} ไม่มีประโยชน์ที่นี่\n"
            f"ที่ตรวจพบ: {self.evidence()}\n"
            f"weight ที่โหลดมาจะไม่มี engine ไหนรันมันได้ (โมเดลระดับนี้กิน 15-25 GB)\n"
            f"\n"
            f"สิ่งที่ควรทำแทน — ส่ง bundle ไปให้เครื่องที่รันได้ แล้วให้มันโหลด weight ของมันเอง:\n"
            f"  lmds node push {target} {slug} --download\n"
            f"  lmds node run {target} start {slug}\n"
            f"\n"
            f"ถ้าเครื่องนี้รันโมเดลได้จริงและการตรวจเดาผิด สั่งทับด้วย {ROLE_ENV}=serving "
            f"หรือใส่ --force"
        )


@functools.lru_cache(maxsize=1)
def _detect(pinned: str = "") -> ServingCapability:
    forced = (os.environ.get(ROLE_ENV) or "").strip().lower()
    gpus = _gpu_count()
    docker = shutil.which("docker") is not None
    server = llamacpp_server(pinned)

    engines: list[str] = []
    if server is not None:
        engines.append("llamacpp")
    # vLLM/TensorRT รันในคอนเทนเนอร์และต้องมี GPU — docker เปล่า ๆ ไม่พอ
    if docker and gpus:
        engines += ["vllm", "sglang", "trtllm"]

    if forced == "serving" and not engines:
        engines = ["llamacpp"]  # เชื่อผู้ใช้ แต่ไม่ไปเดาว่าเป็น engine อะไร
    elif forced == "hub":
        engines = []

    return ServingCapability(gpus=gpus, docker=docker, llamacpp=server,
                             engines=engines, forced=forced if forced in ("hub", "serving") else "")


def detect(pinned: str = "") -> ServingCapability:
    """ผลตรวจของเครื่องนี้ (แคชไว้ทั้ง process — ฮาร์ดแวร์ไม่เปลี่ยนกลางคำสั่ง)"""
    return _detect(pinned)


def reset_cache() -> None:
    """ให้เทสต์และคอนโซลที่รันยาวสั่งตรวจใหม่ได้หลังคนไป build llama.cpp เพิ่ม"""
    _detect.cache_clear()


# คำสั่งที่ต้องรันบนเครื่องที่เสิร์ฟได้เท่านั้น
#
# คัดเฉพาะตัวที่ "ทำแล้วเสียของจริง" — download/repair ดูด weight 15-25 GB ลงเครื่อง
# ที่ไม่มีอะไรรันมันได้ · start/restart/prepare-runtime ล้มแน่นอนแต่ล้มแบบงง ๆ
# ส่วน verify-files, status, doctor, logs ปล่อยผ่าน: มันตรวจของที่มีอยู่แล้ว
# ไม่ได้ดูดอะไรเพิ่ม และเป็นสิ่งที่คนบน hub อยากทำได้อยู่ดี
NEEDS_SERVING = frozenset({"download", "repair", "start", "restart", "prepare-runtime"})


def guard(slug: str, action: str, node_hint: str = "", force: bool = False) -> str:
    """คืนข้อความปฏิเสธถ้าคำสั่งนี้ไม่ควรรันบนเครื่องนี้ ไม่งั้นคืนสตริงว่าง

    แยกจากการ raise เพื่อให้ CLI, web job และคอนโซลใช้เกณฑ์เดียวกันได้
    โดยแต่ละที่เลือกวิธีแจ้งเอง
    """
    if force or action not in NEEDS_SERVING:
        return ""
    capability = detect()
    if capability.can_serve:
        return ""
    return capability.refusal(slug, action, node_hint)
