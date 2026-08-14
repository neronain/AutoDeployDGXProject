# เอกสารคลัสเตอร์ของ NVIDIA — อะไรที่ยืนยันของเรา อะไรที่เติม

อ่านเมื่อ 2026-08-14 จากสี่แหล่ง:

| แหล่ง | ให้อะไร |
|---|---|
| [DGX Spark clustering](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) | ขีดจำกัดจำนวนเครื่อง · รุ่นสายที่รับรอง · โครงสร้าง interface |
| [Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) | มันทำอะไรและ **ไม่** ทำอะไร |
| [NVIDIA blog · multi-node](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/) | เลข memory ต่อจำนวน node · vLLM |
| [Exxact · 4x Spark cluster](https://www.exxactcorp.com/blog/deep-learning/what-you-need-to-build-a-4x-nvidia-dgx-spark-cluster-switch-cabling-power) | switch รุ่นไหน · กับดักตอน negotiate ความเร็ว |

---

## ที่ยืนยันสิ่งที่ LMDS ทำอยู่แล้ว

- **stacked ใช้ vLLM** — blog พูดถึง vLLM อย่างเดียวเช่นกัน ตรงกับ decision tree ใน
  `RUNBOOK-MULTI-NODE.md` ที่บอกว่า GGUF stack ไม่ได้
- **memory รวมตามจำนวนเครื่อง** — 2 node = 256 GB (พอสำหรับ ~400B params) ·
  4 node = 512 GB ตรงกับที่ `fit/targets.py` คิด (128 GB × node_count)
- **ต้องมี passwordless SSH ระหว่างเครื่อง** — Cluster Assistant ก็ตั้ง SSH key ให้
  เป็นงานหลักงานหนึ่ง

## ที่เติมของใหม่ — สามข้อ

### 1. ขีดจำกัดจำนวนเครื่องขึ้นกับ*วิธีต่อ* ไม่ใช่แค่จำนวน

> ต่อสายตรงถึงกันได้ **สูงสุด 3 เครื่อง** · เกินกว่านั้นต้องผ่าน switch ซึ่งรองรับ
> **สูงสุด 4 เครื่อง**

`fit/targets.py` มี `dgx-spark-stacked` (2) และ `dgx-spark-stacked-4` (4, `tested=False`)
แต่ไม่มีแนวคิดว่า*ต่อกันอย่างไร* — คนที่มี 4 เครื่องแต่ไม่มี switch ต่อวงแหวนไม่ได้
และตอนนี้ไม่มีอะไรบอกเขา

### 2. ลิงก์ที่ขึ้น "200G" อาจ negotiate ได้จริงแค่ 50G

NVIDIA ตรวจว่าแต่ละลิงก์ต้องได้ **≥184 Gbit/s** · ส่วน Exxact เตือนว่าเมื่อผ่าน
switch **auto-negotiation มักตั้งพอร์ตไว้ที่ 50 Gbps** ต้องไปตั้งเป็น 200 Gbps เอง
และ throughput ที่วัดได้จริงราว 100 Gbps ต่อลิงก์ ซึ่งเป็นเพดานของ PCIe Gen5 x4
ไม่ใช่การตั้งค่าผิด

**นี่คือรูปแบบเดิมที่โปรเจกต์นี้เจอซ้ำ ๆ**: ค่าที่ถูกอยู่ในระบบแต่ไม่มีใครตรวจผลลัพธ์
`FLEET-MULTI-NODE.md` แยกได้แล้วว่าลิงก์มี IP จริงหรือเป็น `169.254.x.x` แต่**ไม่ได้ดู
ความเร็วที่ negotiate ได้** — เครื่องที่ช้าลง 4 เท่าจะดูปกติทุกอย่าง

### 3. Cluster Assistant ไม่ใช่คู่แข่งของ LMDS และไม่ทับกัน

เอกสารบอกตรง ๆ ว่ามัน **ไม่ตั้งค่า workload** — ทำแค่ readiness check, ตรวจ topology
ของสาย, วัด bandwidth, วางแผน IP + netplan, และสร้าง SSH key พร้อม alias แล้วส่งต่อ
ให้ playbook ของ NCCL/vLLM/PyTorch ทำต่อ

แปลว่า **Sync ทำ fabric · LMDS ทำ workload** และ LMDS *ใช้ผลของมันต่อได้*: SSH alias
ที่มันเขียนไว้คือสิ่งที่ controller ต้องการพอดี · ข้อจำกัด: GUI อย่างเดียว ไม่มี CLI/API
และต้องการ OS เดือนเมษายน 2026 ขึ้นไป

## ของที่ควรซื้อ ถ้าจะขยายเป็น 4 เครื่อง

- switch: **MikroTik CRS804-4DDQ-HRM** (4×400G QSFP56-DD, 1U ครึ่งแร็ค)
- สาย QSFP-DD → QSFP56 (หรือ breakout 2 เส้นเพื่อขยายถึง 8 node)
- สายต่อตรงที่ NVIDIA รับรองมีแค่สามรุ่น: Amphenol `NJAAKK-N911` (400mm),
  `NJAAKK0006` (0.5m), Luxshare `LMTQF022-SD-R` (400mm)
- ไฟรวมทั้งคลัสเตอร์ 4 เครื่อง + switch < 1200W (Spark 240W/เครื่อง · switch ~92–123W)

## งานที่ควรทำต่อใน LMDS

| # | งาน | สถานะ |
|---|---|---|
| 1 | เตือนเมื่อลิงก์ ConnectX ของ Spark negotiate ได้ต่ำกว่า 184 Gbit/s | **ทำแล้ว** — `link_warning()` ใน `nodes/cluster.py` ขึ้นทั้ง CLI และหน้าเว็บ |
| 2 | บันทึกเพดานการต่อสาย (ตรง ≤3 · ผ่าน switch ≤4) | **ทำแล้ว** — อยู่ที่ `dgx-spark-stacked-4` ใน `fit/targets.py` |
| 3 | preflight อ่าน SSH alias ที่ Sync เขียนไว้แล้วใช้ต่อ | ยังไม่ทำ — ต้องมีเครื่องที่ผ่าน Sync มาก่อนถึงจะรู้รูปแบบไฟล์จริง |

`NCCL_SOCKET_IFNAME` ไม่ได้อยู่ในรายการเพราะ `lmds node cluster --write` เติมให้จาก
`head["iface"]` อยู่แล้ว (`cli/main.py`) — ที่ว่างไว้ใน template คือ bundle ที่ยังไม่เคยรัน
คำสั่งนั้น

เครดิต: https://www.facebook.com/neronain.minidev
