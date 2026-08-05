# Runbook: รันโมเดลข้าม 2 เครื่อง (stacked) ผ่าน LMDS

> ลำดับคำสั่งที่ **ผ่านการรันจริง** บน DGX Spark 2 เครื่อง (5 ส.ค. 2569 · Llama 3.3 70B ·
> vLLM 26.05 NGC · TP=2 · `mp` backend ไม่ใช้ Ray) · ภาพรวมสถาปัตยกรรม: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md)

---

## 0. เตรียมครั้งเดียวต่อคลัสเตอร์

จาก **hub** (เครื่องที่คุณนั่งอยู่ — เป็นโน้ตบุ๊กที่ไม่มี GPU ก็ได้):

```bash
lmds node add <ip-head>   --user <user> --name spark-head   --install
lmds node add <ip-worker> --user <user> --name spark-worker --install
lmds node cluster                       # ดูว่าจับคู่กันได้ไหม + เห็น IP ที่ระบบเสนอ
lmds node set spark-head   --cluster-ip 10.100.152.1
lmds node set spark-worker --cluster-ip 10.100.152.2
lmds node cluster                       # ต้องขึ้น "พร้อม"
```

- ถามรหัสผ่านครั้งเดียวตอน `node add` แล้วทิ้ง — ทะเบียนไม่เก็บรหัสผ่าน
- `--install` ติดตั้ง LMDS บนเครื่องนั้นให้ (ต้องมี Docker + git อยู่แล้ว)
- **cluster IP ต้องอยู่วงเดียวกันทั้งคู่** — DGX Spark มี fabric หลายวง ระบบเสนอวงที่ทั้งคู่มีขาร่วมกัน
- ต้องมี **passwordless SSH จาก head → worker** ด้วย (controller ใช้ยิงคำสั่งไป worker)

## 1. สร้าง bundle — รันบนเครื่อง head

bundle ต้องอยู่บนเครื่องที่จะรันมันจริง:

```bash
lmds node run spark-head deploy meta-llama/Llama-3.3-70B-Instruct \
  --target dgx-spark-stacked --no-llm --yes
```

หรือ ssh เข้าไปรัน `lmds deploy …` เองก็ได้ผลเหมือนกัน · ตัด `--no-llm` ออกถ้าตั้ง provider ไว้แล้ว
(จะได้ parser/feature ที่ค้นมาให้ แทนค่าตั้งต้นแบบ rule-based)

โมเดล gated: LMDS อ่าน token จาก `~/.cache/huggingface/token` ที่ `huggingface-cli login` เขียนไว้ให้เอง

## 2. ใส่ค่าคลัสเตอร์ลง bundle

```bash
lmds node cluster --write llama-3-3-70b-instruct --on spark-head
```

เขียน `cluster.env` ลงใน bundle บนเครื่อง head:

```bash
MASTER_IP=10.100.152.1        WORKER_IP=10.100.152.2
WORKER_IPS="10.100.152.2"     NNODES=2      TENSOR_PARALLEL_SIZE=2
SSH_USER=neronain             NCCL_SOCKET_IFNAME=enp1s0f1np1
```

controller **source ไฟล์นี้ก่อน default ทั้งหมด** แล้วข้ามการถาม IP ตอน start
(env ที่ตั้งจากภายนอกยังชนะไฟล์นี้เสมอ · แก้ไฟล์เองได้)

> ต้องการ override เฉพาะไซต์ เช่น image คนละตัว ใส่เพิ่มในไฟล์นี้ได้เลย:
> `VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1`

## 3. ลำดับรันจริง

```bash
N=spark-head; S=llama-3-3-70b-instruct

lmds node run $N run-controller $S prepare-runtime   # ล็อก image ให้ตรงกันทุกเครื่อง
lmds node run $N run-controller $S download          # ข้ามได้ถ้า weight อยู่แล้ว
lmds node run $N run-controller $S verify-files      # ตรวจ shard ครบ + ขนาดตรงกับ Hub
lmds node run $N run-controller $S sync-worker       # rsync ไป worker (ไฟล์ตรงแล้วจะข้ามเอง)
lmds node run $N run-controller $S verify-worker
lmds node run $N run-controller $S start
lmds node run $N run-controller $S test-text
```

ยังไม่มี `run-controller` ใน LMDS — ระหว่างนี้เรียกสคริปต์ตรง ๆ ผ่าน SSH:

```bash
ssh <user>@<ip-head> 'cd ~/bundles/'$S' && ./'$S'-stacked.sh prepare-runtime'
```

หรือใช้คำสั่ง fleet ของ LMDS ที่ห่อให้แล้ว (ทำงานเทียบเท่า `start`/`stop`/`logs` ของ controller):

```bash
lmds node run spark-head start   llama-3-3-70b-instruct
lmds node run spark-head logs    llama-3-3-70b-instruct -n 100
lmds node run spark-head doctor  llama-3-3-70b-instruct
lmds node run spark-head stop    llama-3-3-70b-instruct
lmds ps --all                                   # เห็นทุกเครื่องในตารางเดียว
```

## 4. ปรับหน่วยความจำ (สำคัญบน unified memory)

DGX Spark ใช้ **unified memory** — CPU กับ GPU แย่ง pool เดียวกัน 121 GB ต่อเครื่อง การตั้ง
`--gpu-util` สูงเกินไปจะชน OOM ตอน warm-up ซึ่งไวกว่าการ์ดแยกมาก

```bash
./<slug>-stacked.sh restart --gpu-util 0.80        # ค่าตั้งต้นคือ 0.85
```

ตัวเลขจริงจากเครื่อง (Llama 3.3 70B · 2 เครื่อง · context 65,536):

| `--gpu-util` | RAM ที่ระบบใช้ (steady) | KV cache |
|---|---|---|
| 0.85 (ค่าตั้งต้น) | ~109 GB จาก 121 GB | 221,056 tokens |
| 0.80 | ~102 GB | ~185,000 tokens |

> **`--gpu-util` เป็นสัดส่วน ไม่ใช่จำนวน GB** — อยากให้เครื่องใช้ ~110 GB ต้อง **ลด** ค่านี้
> ไม่ใช่ตั้งเป็น 110 · ค่าที่รับได้คือ 0.3–0.98 นอกช่วงนี้ controller ปฏิเสธตั้งแต่ต้น
>
> KV cache ที่เหลือควรมากกว่า `context × max_num_seqs` ถ้าอยากให้รับงานพร้อมกันได้เต็มจำนวน
> (65,536 × 4 = 262,144 tokens) — ต่ำกว่านั้นไม่พัง แต่คำขอที่เกินจะเข้าคิว

## 5. เช็กหลังรัน

```bash
lmds node run spark-head run-controller <slug> status          # container ทั้งสองเครื่อง + API
lmds node run spark-head run-controller <slug> network-info    # IP/interface/HCA ที่ใช้จริง
lmds node run spark-head run-controller <slug> client-config   # ค่าให้ฝั่ง client
```

`network-info` ต้องขึ้น interface และ HCA จริง ไม่ใช่ค่าว่าง:

```text
NCCL if    : enp1s0f1np1
NCCL HCA   : rocep1s0f1
```

ถ้าว่าง แปลว่าหา fabric ไม่เจอ → NCCL จะตกไปใช้สายบริหารจัดการหรือ TCP ซึ่ง**ยังรันได้แต่ช้าลงมาก**
· ตั้งเองได้: `NCCL_SOCKET_IFNAME=... NCCL_IB_HCA=... ./<slug>-stacked.sh restart`

## 6. เวลาที่ใช้จริง (Llama 3.3 70B · 131 GB · 2 เครื่อง)

```text
deploy (สร้าง bundle)      ~30 วินาที
prepare-runtime            ~15 วินาที (image อยู่แล้ว) / 20+ นาที (ต้อง pull)
verify-files               ~5 วินาที (30 shards)
sync-worker + verify        ~1 วินาที (ไฟล์ตรงกันแล้ว)
start → /health ผ่าน       ~8 นาที
```

## 7. ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ |
|---|---|
| `This host does not own MASTER_IP` | รันสคริปต์ head ผิดเครื่อง — สลับ `MASTER_IP`/`WORKER_IP` หรือแก้ `cluster.env` |
| `worker container หยุดก่อน head จะเริ่ม` | ดู `logs worker` — มักเป็น image คนละตัวหรือ cache สิทธิ์ไม่ถูก (`prepare-runtime` ซ้ำ) |
| ค้างที่ NCCL init | cluster IP อยู่คนละวง — `lmds node cluster` จะขึ้น blocker `split-fabric` |
| OOM ตอน warm-up | ลด `--gpu-util` (ดูข้อ 4) |
| `lmds` ไม่เจอบน node | ติดตั้งด้วย `lmds node install <ชื่อ>` |

## 8. ขยายเป็น 3–4 เครื่อง

โครงสร้างพร้อมแล้ว (`WORKER_IPS` วนทุกเครื่องทุกขั้นตอน · preset `dgx-spark-stacked-4`) แต่
**ยังไม่ได้รันจริงเกิน 2 เครื่อง** · ข้อจำกัดที่แท้จริงคือ tensor parallel ต้องหาร attention head
ลงตัว: 2 และ 4 เครื่องใช้ได้ · **3 เครื่องใช้ TP=3 ไม่ได้** (Llama 3.3 70B มี 64 head) ต้อง TP=2 +
pipeline · `lmds node cluster` บอกให้เองว่ากลุ่มนั้นใช้แบบไหน
