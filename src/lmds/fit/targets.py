"""Target spec ของเครื่องเป้าหมาย — preset จากเครื่องจริงของทีม + สร้างจากผลตรวจเครื่องปัจจุบัน"""

from __future__ import annotations

from dataclasses import dataclass

from lmds.hardware import HardwareReport, MemoryModel, TargetProfile

# ── "ต่อกันอย่างไร" ไม่ใช่แค่ "กี่เครื่อง" ────────────────────────────────────────
# DGX Spark มี QSFP **2 ช่องต่อเครื่อง** (ConnectX-7, 200G/ช่อง) · วงแหวน 3 เครื่อง
# (A.p1→B.p2, B.p1→C.p2, C.p1→A.p2) ใช้ช่องครบทั้งสองข้างของทุกเครื่องพอดี เครื่องที่ 4
# จึงไม่มีช่องว่างเหลือให้เสียบ — เป็นข้อจำกัดของ *จำนวนช่องบนตัวเครื่อง* ไม่ใช่ของซอฟต์แวร์
# ที่มา: คู่มือ NVIDIA "ConnectX-7 Networking" + ผังของ NVIDIA Sync (ถอดความ 2026-08-14)
# กฎเดียวกันนี้ถูกใช้เดาผังจากสายที่เสียบจริงใน lmds/nodes/netplan.py (`infer_topology`)
MAX_DIRECT_NODES = 3

# เกิน 3 เครื่องต้องผ่าน switch · **ไม่มีเพดานทางเทคนิคที่ฝั่ง switch เท่าที่เรายืนยันได้**
# เลข 8 นี้คือ "ใหญ่สุดที่มีคนอ้างว่า serve ได้จริง" ไม่ใช่ขีดจำกัดที่เราวัดเอง —
# ใช้เป็นเพดานของ *ตัววางแผน* เพื่อไม่ให้รับเลขที่ไม่มีใครเคยเห็นของจริงเลย
# ระดับหลักฐาน: คำกล่าวอ้างของผู้เขียนรีโปภายนอก (ดู docs/NVIDIA-CLUSTER-SOURCES.md §1)
MAX_SWITCH_NODES = 8

# ของที่ต้องซื้อเพิ่มจริง ๆ เมื่อเกิน 3 เครื่อง — รุ่นจาก docs/NVIDIA-CLUSTER-SOURCES.md
# หัวข้อ "ของที่ควรซื้อ" · บอกชื่อรุ่นเพราะ "ต้องมี switch" เฉย ๆ ไม่ช่วยคนที่ยังไม่มี
SWITCH_SHOPPING_LIST = (
    "switch 400G QSFP56-DD เช่น MikroTik CRS804-4DDQ-HRM (4 พอร์ต 1U ครึ่งแร็ค)",
    "สาย QSFP-DD → QSFP56 เครื่องละเส้น (ใช้ breakout 1→2 ถ้าจะเกิน 4 เครื่อง)",
    "ตั้ง port speed ที่ switch เป็น 200G ตายตัว — auto-negotiate มักตกมาที่ 50G เงียบ ๆ",
)
# สายต่อตรงที่ NVIDIA รับรองมีแค่สามรุ่น — ใส่ไว้เพราะคนที่ยังอยู่ใน ≤3 เครื่องถามว่า "สายอะไร"
DIRECT_SHOPPING_LIST = (
    "สายต่อตรงที่ NVIDIA รับรอง: Amphenol NJAAKK-N911 (400mm) · NJAAKK0006 (0.5m) · "
    "Luxshare LMTQF022-SD-R (400mm)",
)


@dataclass(frozen=True)
class Interconnect:
    """ต่อ N เครื่องนี้เข้าด้วยกันอย่างไร — ตอบก่อนลงมือ ไม่ใช่ตอน `cluster apply` ล้ม

    `cabling` เป็นรหัส (single / direct-2 / ring-3 / switch / unsupported) ให้หน้าเว็บ
    เรียบเรียงเป็นอังกฤษเอง · `note` เป็นไทยสำหรับ CLI และ `fit.notes` ตามแนวของไฟล์นี้
    """

    node_count: int
    cabling: str
    needs_switch: bool
    shopping: tuple[str, ...]
    note: str

    @property
    def supported(self) -> bool:
        return self.cabling != "unsupported"


def interconnect_for(node_count: int) -> Interconnect:
    """ผังสายที่คลัสเตอร์ขนาดนี้ต้องใช้ + ของที่ต้องมี

    เคสที่ต้องกันจริง: คนมี Spark 4 เครื่องแต่ไม่มี switch · เสียบวงแหวนไม่ได้เพราะช่องหมด
    ตั้งแต่เครื่องที่ 3 แล้วไปเจอ "unknown topology" ตอน `lmds cluster apply` ซึ่งไม่ได้บอก
    ว่าต้องซื้ออะไร — ต้องบอกตั้งแต่ตอนเลือก target
    """
    nodes = max(1, int(node_count))
    if nodes == 1:
        return Interconnect(1, "single", False, (), "เครื่องเดียว — ไม่ต้องต่อสายคลัสเตอร์")
    if nodes == 2:
        return Interconnect(
            2, "direct-2", False, DIRECT_SHOPPING_LIST,
            "2 เครื่อง: ต่อสายตรงถึงกัน 1 เส้น (ใช้ 2 เส้นก็ได้ถ้าอยากได้ทั้งสองช่อง) — ไม่ต้องมี switch",
        )
    if nodes <= MAX_DIRECT_NODES:
        return Interconnect(
            nodes, "ring-3", False, DIRECT_SHOPPING_LIST,
            f"{nodes} เครื่อง: ต่อตรงเป็นวงแหวน {nodes} เส้น ใช้ QSFP ครบทั้งสองช่องทุกเครื่อง — "
            "ไม่ต้องมี switch แต่ขยายต่อไม่ได้อีกแล้ว (ช่องหมด)",
        )
    if nodes <= MAX_SWITCH_NODES:
        # switch อ้างอิงมี 4 พอร์ต — เกิน 4 เครื่องต้องบอกเรื่อง breakout ด้วย ไม่งั้นเขาซื้อมาแล้วพอร์ตไม่พอ
        breakout = " + สาย breakout 1→2 (switch อ้างอิงมี 4 พอร์ต)" if nodes > 4 else ""
        return Interconnect(
            nodes, "switch", True, SWITCH_SHOPPING_LIST,
            f"{nodes} เครื่อง: **ต้องมี switch** — ต่อสายตรงถึงกันได้สูงสุด {MAX_DIRECT_NODES} เครื่อง "
            f"(QSFP 2 ช่อง/เครื่อง วงแหวน {MAX_DIRECT_NODES} เครื่องใช้ช่องครบพอดี) · "
            f"ที่ต้องมีเพิ่ม: {SWITCH_SHOPPING_LIST[0]}{breakout}",
        )
    return Interconnect(
        nodes, "unsupported", True, SWITCH_SHOPPING_LIST,
        f"{nodes} เครื่อง: ใหญ่กว่าที่เราเคยเห็นของจริง (ใหญ่สุดที่มีคนอ้างคือ {MAX_SWITCH_NODES} "
        "เครื่องผ่าน switch และยังไม่มี log ดิบยืนยัน) — LMDS วางแผนให้ไม่ได้",
    )


@dataclass(frozen=True)
class TargetSpec:
    name: str
    memory_model: MemoryModel
    memory_gb: float  # VRAM ต่อ GPU (discrete) หรือ unified ทั้งก้อน
    gpu_count: int = 1
    system_ram_gb: float | None = None  # ใช้ประเมิน offload ของ llama.cpp
    tested: bool = False  # False → คำนวณแบบ conservative (ลด budget เพิ่ม)
    # GPU ต่อ "เครื่อง" — ใช้แยกว่า gpu_count มาจากหลาย GPU ในเครื่องเดียว (RTX dual)
    # หรือหลายเครื่องเครื่องละใบ (DGX Spark stacked) ซึ่งคนละเรื่องกันตอน generate
    gpus_per_node: int = 1

    @property
    def node_count(self) -> int:
        return max(1, self.gpu_count // max(1, self.gpus_per_node))

    @property
    def total_gpu_memory_gb(self) -> float:
        return self.memory_gb * self.gpu_count

    @property
    def interconnect(self) -> Interconnect:
        """ต่อกันอย่างไร — คำนวณจาก node_count ไม่ใช่จากชื่อ preset

        target ที่สร้างเองจากเครื่องจริง (`from_hardware_report`) จึงได้คำตอบเดียวกัน
        · เครื่อง discrete หลายใบอยู่ในเครื่องเดียว node_count=1 → "single" ถูกแล้ว
        """
        return interconnect_for(self.node_count)


# preset ตามเครื่องทดสอบจริงของทีม (PRD §13 Decision Log 2026-07-21) + เครื่องอ้างอิงทั่วไป
PRESETS: dict[str, TargetSpec] = {
    "dgx-spark-single": TargetSpec(
        "dgx-spark-single", MemoryModel.UNIFIED, 128.0, 1, system_ram_gb=None, tested=True
    ),
    "dgx-spark-stacked": TargetSpec(
        "dgx-spark-stacked", MemoryModel.UNIFIED, 128.0, 2, system_ram_gb=None, tested=True
    ),
    # ── 3 เครื่อง: ขนาดสุดท้ายที่ยัง "ไม่ต้องซื้อ switch" ──
    # ต่อตรงเป็นวงแหวน 3 เส้น ใช้ QSFP ครบทั้งสองช่องทุกเครื่องพอดี (interconnect_for(3))
    # ซื้อแค่สายเพิ่มก็ขยายจาก 2 → 3 ได้ · จาก 3 → 4 ต้องมี switch เสมอ ไม่ใช่แค่สาย
    #
    # ข้อแลก: TP=3 หาร attention head ของโมเดลส่วนใหญ่ไม่ลง (Llama 3.3 70B = 64 head)
    # vLLM ปฏิเสธตั้งแต่ start ต้องไปทาง pipeline-parallel ซึ่งช้ากว่า —
    # `parallelism_note()` ใน nodes/cluster.py คำนวณเรื่องนี้ให้อยู่แล้ว
    #
    # หลักฐาน: ผังวงแหวน 3 เครื่องมาจากคู่มือ NVIDIA + NVIDIA Sync (ถอดความ 2026-08-14)
    # แต่ **ทีมเราไม่เคยรันโมเดลบน 3 เครื่องจริง** และไม่มี artifact ของใครที่เราตรวจได้
    # → tested=False (budget คิดแบบ conservative) · ตั้งเป็น True = ลูกค้าพังตอนรัน
    "dgx-spark-stacked-3": TargetSpec(
        "dgx-spark-stacked-3", MemoryModel.UNIFIED, 128.0, 3, system_ram_gb=None, tested=False
    ),
    # 4 เครื่อง: TP=4 หาร attention heads ของโมเดลส่วนใหญ่ลงตัว (64/4=16) ต่างจาก 3 เครื่อง
    # ที่ TP=3 มักหารไม่ลง — ยังไม่ได้ทดสอบจริง จึงคิดแบบ conservative
    # ต้องมี switch ด้วย: ต่อสายตรงถึงกันได้สูงสุด 3 เครื่อง เพราะวงแหวนใช้ cage ครบทั้งสอง
    # ฝั่งที่ 3 เครื่องพอดี เกินกว่านั้นต้องผ่าน switch (MAX_DIRECT_NODES ด้านบน)
    # ไม่มีเพดานที่ฝั่ง switch: เดิมเคยเขียนไว้ว่า "รองรับได้ถึง 4" แต่ถอนออกแล้ว —
    # เป็นการถอดความจาก URL เดียวที่อ่านเมื่อ 2026-08-14 โดยไม่ได้เก็บข้อความต้นฉบับไว้
    # และ docs/NVIDIA-CLUSTER-SOURCES.md ของเราเองก็ระบุว่าสาย breakout ขยายถึง 8 เครื่องได้
    # ปัจจุบันมีคลัสเตอร์ 8 เครื่องที่ serve อยู่จริง 2 ชุด (author-asserted ไม่มี log ยืนยัน)
    # ดู docs/NVIDIA-CLUSTER-SOURCES.md §1
    "dgx-spark-stacked-4": TargetSpec(
        "dgx-spark-stacked-4", MemoryModel.UNIFIED, 128.0, 4, system_ram_gb=None, tested=False
    ),
    # ── 8 เครื่อง = 1,024 GB รวม (128 × 8 ตามสเกลเดิมของ preset ทุกตัวข้างบน) ──
    # ต้องมี switch + สาย breakout 1→2 (switch 4 พอร์ตจ่ายได้ 8 เส้น) · TP=8 หาร 64 head ลงตัว
    #
    # **ระดับหลักฐาน: คำกล่าวอ้างของผู้เขียน — ไม่ใช่หลักฐานที่มี artifact**
    # มาจากรีโปภายนอกสองตัว (im0xMagnus/deepseek-v4.1-flash-uncensored-8x-dgx-spark
    # TP8 context 1,048,576 · im0xMagnus/glm-5.3-uncensored-8x-dgx-spark TP8 context 524,288)
    # ตัวแรก results/ มีไฟล์เดียวไม่มี log ดิบ · ตัวที่สอง **ไม่มีโฟลเดอร์ results/ เลย**
    # ทีมเราไม่เคยรัน ไม่มี log ของเราเองสักบรรทัด → tested=False เด็ดขาด
    # (tested=True ในไฟล์นี้แปลว่า "ทีมเรารันบนเครื่องจริงแล้ว" ซึ่งปิดโหมด conservative
    #  ของ budget — ตั้งเป็น True เพราะ README ของคนอื่น = ลูกค้า deploy แล้วพังตอนรัน)
    # ดู docs/NVIDIA-CLUSTER-SOURCES.md §1 ตารางระดับหลักฐาน
    "dgx-spark-stacked-8": TargetSpec(
        "dgx-spark-stacked-8", MemoryModel.UNIFIED, 128.0, 8, system_ram_gb=None, tested=False
    ),
    "rtx-pro-4000": TargetSpec("rtx-pro-4000", MemoryModel.DISCRETE, 24.0, 1, tested=True),
    # dual = สอง GPU ใน "เครื่องเดียว" ไม่ใช่สองเครื่อง — node_count จึงต้องเป็น 1
    "rtx-pro-4000-dual": TargetSpec(
        "rtx-pro-4000-dual", MemoryModel.DISCRETE, 24.0, 2, tested=True, gpus_per_node=2
    ),
    "rtx-4070-super": TargetSpec("rtx-4070-super", MemoryModel.DISCRETE, 12.0, 1, tested=True),
    "rtx-4070-ti-super": TargetSpec("rtx-4070-ti-super", MemoryModel.DISCRETE, 16.0, 1, tested=True),
    # hardware-validated 2026-08-03: gemma-4-12b-it UD-Q8_K_XL (GGUF + vision) ที่ context 16,384
    "rtx-5090": TargetSpec("rtx-5090", MemoryModel.DISCRETE, 32.0, 1, tested=True),
    # ── รุ่นอ้างอิงยอดนิยม (ยังไม่ทดสอบจริง → conservative) ──
    # RTX 50 series (Blackwell) — 5090 ย้ายขึ้นไปกลุ่มที่ทดสอบจริงแล้ว
    "rtx-5080": TargetSpec("rtx-5080", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-5070-ti": TargetSpec("rtx-5070-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-5070": TargetSpec("rtx-5070", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-5060-ti": TargetSpec("rtx-5060-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    # RTX 40 series (Ada Lovelace)
    "rtx-4090": TargetSpec("rtx-4090", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-4080-super": TargetSpec("rtx-4080-super", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-4080": TargetSpec("rtx-4080", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    "rtx-4070-ti": TargetSpec("rtx-4070-ti", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-4060-ti": TargetSpec("rtx-4060-ti", MemoryModel.DISCRETE, 16.0, 1, tested=False),
    # RTX 30 series (Ampere)
    "rtx-3090-ti": TargetSpec("rtx-3090-ti", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-3090": TargetSpec("rtx-3090", MemoryModel.DISCRETE, 24.0, 1, tested=False),
    "rtx-3080-ti": TargetSpec("rtx-3080-ti", MemoryModel.DISCRETE, 12.0, 1, tested=False),
    "rtx-3080": TargetSpec("rtx-3080", MemoryModel.DISCRETE, 10.0, 1, tested=False),
    "rtx-3060": TargetSpec("rtx-3060", MemoryModel.DISCRETE, 12.0, 1, tested=False),
}


def from_hardware_report(report: HardwareReport) -> TargetSpec | None:
    """สร้าง TargetSpec จากเครื่องที่ตรวจพบจริง — คืน None ถ้าไม่พบ GPU"""
    if not report.gpus:
        return None
    # nvidia-smi บนเครื่อง unified memory (DGX Spark GB10) มักรายงาน memory.total ไม่ได้ —
    # fallback ไปใช้สเปกจาก GPU allowlist
    memory_values: list[float] = []
    for gpu in report.gpus:
        if gpu.vram_mib:
            memory_values.append(gpu.vram_mib / 1024)
        elif gpu.known is not None:
            memory_values.append(gpu.known.vram_gb)
    if not memory_values:
        return None
    memory_model = (
        MemoryModel.UNIFIED
        if report.profile in (TargetProfile.DGX_SPARK_SINGLE, TargetProfile.DGX_SPARK_STACKED)
        else MemoryModel.DISCRETE
    )
    return TargetSpec(
        name="this-machine",
        memory_model=memory_model,
        memory_gb=round(min(memory_values), 1),
        gpu_count=len(report.gpus),
        # GPU ทุกตัวใน report อยู่ในเครื่องเดียว — ไม่ใส่ = dual-RTX ถูกนับเป็น 2 node แล้วแผน
        # ออกมาเป็น stacked ทั้งที่มีเครื่องเดียว (รีวิว 2026-09-04)
        gpus_per_node=len(report.gpus),
        system_ram_gb=report.ram_gb,
        tested=all(g.tested for g in report.gpus),
    )
