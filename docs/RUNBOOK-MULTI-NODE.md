# Runbook — รันโมเดลข้ามหลายเครื่องจาก hub เครื่องเดียว

> ทุกคำสั่งในหน้านี้ **รันจริงบน DGX Spark 2 เครื่องแล้ว** (5 ส.ค. 2569) ไม่ใช่ตัวอย่างสมมติ
> ภาพรวมสถาปัตยกรรม: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md) · คู่มือทั่วไป: [USAGE.md](USAGE.md)

---

## เริ่มจากคำถามเดียว: โมเดลนี้ต้อง stacked ไหม

**stacked ไม่ได้แปลว่าเร็วขึ้น — แปลว่าใหญ่เกินหนึ่งเครื่อง** โมเดลที่ลงเครื่องเดียวได้
รันเครื่องเดียว**เร็วกว่าเสมอ** เพราะไม่ต้องส่ง activation ข้ามสายทุก token

```text
                    โมเดลลงเครื่องเดียวได้ไหม?
                              │
              ┌───────────────┴───────────────┐
             ได้                             ไม่ได้
              │                               │
      dgx-spark-single                 เป็น safetensors ไหม?
      (vLLM หรือ llama.cpp)                    │
                                ┌──────────────┴──────────────┐
                               ใช่                          ไม่ใช่ (GGUF)
                                │                             │
                       dgx-spark-stacked            ❌ stacked ไม่ได้ —
                       (vLLM · TP ข้ามเครื่อง)        เลือก quant ที่เล็กลง
```

| | เครื่องเดียว | Stacked |
|---|---|---|
| Engine | vLLM **หรือ** llama.cpp | **vLLM เท่านั้น** |
| Artifact | safetensors หรือ GGUF | **safetensors เท่านั้น** |
| ต้องมีสายเร็ว | ไม่ต้อง | **ต้องมี** ≥25G (จริงคือ 200G RoCE) |
| ตัวอย่างที่ทดสอบแล้ว | Qwen3-Coder-Next GGUF, Gemma-4-31B | Llama 3.3 70B, DeepSeek-V4-Flash |

> **GGUF stacked ไม่ได้** — llama.cpp ไม่มี tensor parallel ข้ามเครื่องในชุด template ของเรา
> ถ้าโมเดล GGUF ใหญ่เกิน ให้เลือก quant ที่เล็กลงแทน (`lmds inspect` บอกว่า variant ไหนลงได้)

---

## สองคำสั่งที่ต้องแยกให้ออก

นี่คือจุดที่คนสับสนบ่อยที่สุด:

```text
lmds node run <เครื่อง> <คำสั่งของ lmds>     →  ps · start · stop · logs · doctor · deploy · scan
lmds node ctl <เครื่อง> <slug> <คำสั่ง>       →  prepare-runtime · download · verify-files ·
                                                sync-worker · verify-worker · test-text ·
                                                network-info · client-config · bench
```

- **`node run`** = สั่ง *โปรแกรม lmds* บนเครื่องนั้น
- **`node ctl`** = สั่ง *สคริปต์ controller* ที่อยู่ในตัว bundle

ขั้นตอนของ stacked (`sync-worker`, `verify-worker`) มีเฉพาะใน controller — ต้องใช้ `node ctl`

---

## 0 · เตรียมคลัสเตอร์ (ทำครั้งเดียว)

จาก **hub** — เครื่องที่คุณนั่งอยู่ เป็นโน้ตบุ๊กที่ไม่มี GPU ก็ได้

```bash
lmds node add <ip-head>   --user <user> --name spark-head   --install
lmds node add <ip-worker> --user <user> --name spark-worker --install
```

ถามรหัสผ่าน **ครั้งเดียว** เพื่อติดตั้ง SSH key แล้วทิ้ง · `--install` ลง LMDS บนเครื่องนั้นให้เลย

```bash
lmds node cluster
```

```text
เครื่อง        สายเร็วสุด   cluster IP      stacked ได้
spark-head    200G        —               ได้
spark-worker  200G        —               ได้

กลุ่มที่ stacked ด้วยกันได้
  ยังไม่พร้อม spark-head + spark-worker — NVIDIA GB10 x1/เครื่อง · world size 2 (TP=2)
    · ยังไม่ได้ตั้ง cluster IP: spark-head, spark-worker
    lmds node set spark-head --cluster-ip 10.100.152.1
```

ระบบ**เสนอ IP ให้** แต่ให้คนยืนยัน เพราะเดาผิดแล้ว stacked จะค้างตอน NCCL init โดยไม่บอกสาเหตุ:

```bash
lmds node set spark-head   --cluster-ip 10.100.152.1
lmds node set spark-worker --cluster-ip 10.100.152.2
lmds node cluster                    # ต้องขึ้น "พร้อม"
```

**ออกนอกออฟฟิศบ่อย?** ใส่ที่อยู่สำรองไว้ hub จะสลับเองเมื่อ LAN ต่อไม่ถึง:

```bash
lmds node set spark-head --alt-host 100.124.77.93     # Tailscale/VPN
```

**สิ่งที่ต้องมีบนเครื่องปลายทาง:** sshd · user อยู่ในกลุ่ม `docker` (ไม่ต้องเป็น root) ·
Docker + NVIDIA Container Toolkit · **passwordless SSH จาก head → worker**

---

## 1 · สร้าง bundle — บนเครื่อง head

> **bundle ต้องอยู่บนเครื่อง head** ไม่ใช่บน hub เพราะ controller ต้องยิง SSH ไป worker เอง

```bash
lmds node run spark-head "deploy meta-llama/Llama-3.3-70B-Instruct \
  --target dgx-spark-stacked --no-llm --yes"
```

- ตัด `--no-llm` ออกถ้าตั้ง LLM provider ไว้ (จะได้ parser/feature ที่ค้นมาให้)
- **ไม่มี API key ก็ได้ค่าที่ถูก** — LMDS มี[สูตรที่รันผ่านจริง](USAGE.md) ในตัว ใช้อัตโนมัติ
- โมเดล gated: อ่าน token จาก `~/.cache/huggingface/token` ที่ `huggingface-cli login` เขียนไว้เอง
- **weight มีอยู่แล้ว?** `lmds node run spark-head scan` บอกว่ามีอะไรอยู่ตรงไหน จะได้ไม่โหลดซ้ำ

---

## 2 · ใส่ค่าคลัสเตอร์ลง bundle

```bash
lmds node cluster --write llama-3-3-70b-instruct --on spark-head
```

เขียน `cluster.env` ลงใน bundle:

```bash
MASTER_IP=10.100.152.1        WORKER_IP=10.100.152.2
WORKER_IPS="10.100.152.2"     NNODES=2      TENSOR_PARALLEL_SIZE=2
SSH_USER=neronain             NCCL_SOCKET_IFNAME=enp1s0f1np1
```

controller **source ไฟล์นี้ก่อน default ทั้งหมด** แล้วข้ามการถาม IP ตอน start ·
env จากภายนอกยังชนะไฟล์นี้เสมอ · แก้มือได้

> ต้องการ override เฉพาะไซต์ เติมบรรทัดเองได้เลย เช่น
> `VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1`

---

## 3 · ลำดับรัน

```bash
N=spark-head; S=llama-3-3-70b-instruct

lmds node ctl $N $S prepare-runtime      # ล็อก image ให้ตรงกันทุกเครื่อง
lmds node ctl $N $S download             # ข้ามได้ถ้า weight อยู่แล้ว
lmds node ctl $N $S verify-files         # ตรวจ shard ครบ + ขนาดตรงกับ Hub
lmds node ctl $N $S sync-worker          # rsync ไป worker (ไฟล์ตรงแล้วข้ามเอง)
lmds node ctl $N $S verify-worker
lmds node ctl $N $S start --gpu-util 0.80
lmds node ctl $N $S test-text
```

หลังจากนี้ใช้คำสั่ง fleet สั้น ๆ ได้เลย:

```bash
lmds ps --all                                  # ทุกเครื่องในตารางเดียว
lmds node run spark-head logs $S -n 100
lmds node run spark-head stop $S
lmds node run spark-head doctor $S             # บอกว่าติดตรงไหน + คำสั่งแก้
```

### เวลาที่ใช้จริง (Llama 3.3 70B · 131 GB · 2 เครื่อง)

```text
deploy (สร้าง bundle)      ~30 วินาที
prepare-runtime            ~15 วินาที (image อยู่แล้ว) · 20+ นาที (ต้อง pull)
verify-files               ~5 วินาที (30 shards)
sync-worker + verify        ~1 วินาที (ไฟล์ตรงกันแล้ว)
start → /health ผ่าน       ~8 นาที
```

---

## 4 · ปรับหน่วยความจำ — สำคัญที่สุดบน unified memory

DGX Spark ใช้ **unified memory**: CPU กับ GPU แย่ง pool เดียวกัน 121 GB ต่อเครื่อง
ตั้ง `--gpu-util` สูงเกินไปจะชน OOM ตอน warm-up ซึ่งไวกว่าการ์ดแยกมาก

```bash
lmds node ctl spark-head <slug> restart --gpu-util 0.80
```

**ตัวเลขที่วัดจากเครื่องจริง** (Llama 3.3 70B · context 65,536):

| `--gpu-util` | ใช้ (steady) | เหลือว่าง | KV cache | `test-text` |
|---|---|---|---|---|
| 0.85 (ค่าตั้งต้น) | 109 / 121 GB | 11 GB | 221,056 tokens | ผ่าน |
| **0.80 (แนะนำ)** | **103 GB** | **17–19 GB** | **184,080 tokens** | ผ่าน |

ตอนพีค (warm-up / CUDA graph capture) สูงกว่า steady ราว 8 GB — ที่ 0.85 แตะ **117–118 GB
จาก 121 GB** ซึ่งบางเกินไปเมื่อ docker/page cache/rsync เบียดเข้ามา

> ⚠️ **`--gpu-util` เป็นสัดส่วน ไม่ใช่จำนวน GB**
> อยากให้เครื่องใช้ ~110 GB ต้อง **ลด** ค่านี้ ไม่ใช่ตั้งเป็น 110
> รับค่า 0.3–0.98 · นอกช่วงนี้ controller ปฏิเสธตั้งแต่ต้น
>
> KV cache ควรมากกว่า `context × max_num_seqs` ถ้าอยากรับงานพร้อมกันเต็มจำนวน
> (65,536 × 4 = 262,144) — ต่ำกว่านั้นไม่พัง แต่คำขอที่เกินจะเข้าคิว

---

## 5 · ตรวจว่าใช้สายที่ถูกจริง

```bash
lmds node ctl spark-head <slug> network-info
```

ต้องขึ้น interface และ HCA **จริง** ไม่ใช่ค่าว่าง:

```text
NCCL if    : enp1s0f1np1
NCCL HCA   : rocep1s0f1
```

ถ้าว่าง = หา fabric ไม่เจอ → NCCL ตกไปใช้สายบริหารจัดการหรือ TCP · **ยังรันได้แต่ช้าลงหลายเท่า**
และไม่มีอะไรฟ้อง จึงเป็นอาการที่ไล่สาเหตุยากที่สุด · บังคับเองได้:

```bash
NCCL_SOCKET_IFNAME=... NCCL_IB_HCA=... lmds node ctl spark-head <slug> restart
```

---

## 6 · เมื่อมันพัง — อาการที่เจอจริงทั้งหมด

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| `This host does not own MASTER_IP` | รันสคริปต์ head ผิดเครื่อง | สลับ `MASTER_IP`/`WORKER_IP` หรือแก้ `cluster.env` |
| ค้างที่ NCCL init | cluster IP อยู่คนละวง | `lmds node cluster` จะขึ้น blocker `split-fabric` |
| `worker container หยุดก่อน head จะเริ่ม` | image คนละตัว / cache สิทธิ์ไม่ถูก | `node ctl <n> <slug> prepare-runtime` ซ้ำ |
| OOM ตอน warm-up | `--gpu-util` สูงไป | ลดทีละ 0.05 (ดูข้อ 4) |
| `LocalEntryNotFoundError` ทั้งที่ไฟล์ครบ | HF cache คนละ layout | `lmds node run <n> scan` บอกว่าอยู่แบบไหน · controller ตั้ง `HF_HUB_CACHE` ให้เอง |
| `only supports fp8 kv-cache, got auto` | สถาปัตยกรรมบังคับ kv-cache | อยู่ในสูตรแล้ว — `lmds recipes <model>` |
| `Expected 7 but got 8 arguments` | cudagraph mode ผิด / JIT cache ค้าง | `node ctl <n> <slug> clear-fi-cache` |
| `image ต่างจากที่ lock ไว้` | เปลี่ยน image หลัง `prepare-runtime` | รัน `prepare-runtime` ใหม่ |
| `lmds` ไม่เจอบน node | ยังไม่ได้ติดตั้ง | `lmds node install <ชื่อ>` |
| เครื่อง hub โชว์โมเดลที่ไม่ใช่ของตัวเอง | ทะเบียนค้างจาก bundle ที่ลบไปแล้ว | `lmds prune` |

---

## 7 · ขยายเป็น 3–4 เครื่อง

โครงสร้างพร้อมแล้ว — controller วน worker ทุกตัวจาก `WORKER_IPS` ทุกขั้นตอน และมี preset
`dgx-spark-stacked-4` · **แต่ยังไม่ได้รันจริงเกิน 2 เครื่อง**

ข้อจำกัดที่แท้จริงไม่ใช่โค้ด แต่คือ **tensor parallel ต้องหาร attention head ลงตัว**:

| เครื่อง | TP | ใช้ได้ไหม |
|---|---|---|
| 2 | 2 | ✅ ทดสอบแล้ว |
| **3** | 3 | ❌ Llama 3.3 70B มี 64 head — 64÷3 ไม่ลงตัว vLLM ปฏิเสธตั้งแต่ start · ต้อง **TP=2 + pipeline** |
| 4 | 4 | ✅ 64÷4 = 16 · หน่วยความจำรวม ~512 GB |

`lmds node cluster` บอกให้เองว่ากลุ่มนั้นใช้แบบไหน:

```text
พร้อม spark1 + spark2 + spark3 — world size 3 (TP=2 + pipeline (TP=3 หาร head ไม่ลง))
```


## Cluster network setup — ต่อสาย ConnectX แล้วให้ hub ตั้ง IP ให้ (0.6.0)

เทียบเท่า "Cluster Assistant" ของ NVIDIA Sync แต่สั่งจาก hub ผ่าน SSH · ใช้เมื่อ Spark มาใหม่ยังไม่มี
cluster IP (พอร์ต ConnectX ขึ้น `169.254.x.x`) หรือย้ายเครื่องมาต่อกลุ่มใหม่ · ทำจากหน้าเว็บ
(ปุ่ม **Set up cluster network** ที่หัว Other machines / หัวกลุ่ม) หรือ CLI ตามนี้

**สิ่งที่ต้องรู้เกี่ยวกับพอร์ตของ Spark** — QSFP 2 ช่อง/เครื่อง (200G) · **หนึ่งช่องคือสอง interface**
(PCIe x4 สองเส้น): พอร์ต 1 (ข้าง RJ45) = `enp1s0f0np0` + `enp1s0f1np1` · พอร์ต 2 = `enP2p1s0f0np0` +
`enP2p1s0f1np1` · ชื่อเหมือนกันทุกเครื่อง · LMDS ตั้ง IP บน **f1** ของช่องที่มีสาย (ตามฟลีตเดิม) อีกตัวปล่อยว่าง

**ผังที่รองรับ** (สายละเส้นต่อลิงก์ ห้ามปนตรงกับ switch):

| เครื่อง | วิธีต่อ | ลิงก์ | วง |
|---|---|---|---|
| 2 | ตรง 1 สาย (ช่องไหนก็ได้) · หรือ 2 สาย (ช่อง 1↔1, 2↔2) | 1–2 | `10.100.152.0/24` (+ `.153`) ปลาย .1/.2 |
| 3 | วงแหวน ใช้ทั้งสองช่องทุกเครื่อง: A.p1→B.p2, B.p1→C.p2, C.p1→A.p2 | 3 | `.152` `.153` `.154` ลิงก์ละวง |
| 2–4 | ผ่าน switch สายละเครื่อง (4 เครื่อง = ทางเดียว) · ตั้ง port ที่ switch เป็น **200G ตายตัว** | 1 | วงเดียว เครื่องที่ i = .i |

```bash
lmds cluster inspect spark-head spark-worker          # พอร์ตไหนมีสาย (carrier) · IP ปัจจุบัน · ผังที่เดาได้ · หมอสาย
lmds cluster plan    spark-head spark-worker [--subnet 10.100.152.0/24] [--topology direct|ring|switch] [--json]
lmds cluster apply   spark-head spark-worker          # ถามรหัส sudo ทีละเครื่อง (เว้นว่างได้ถ้าเครื่องนั้น NOPASSWD) (ไม่เก็บ ไม่โผล่ใน argv)
lmds cluster doctor  spark-head spark-worker          # ตรวจคู่แบบเดิมหลังตั้งเสร็จ
lmds cluster remove-net spark-worker                  # ถอน: ย้ายไฟล์ไป /root/netplan-disabled แล้วล้าง cluster_ip
```

ลำดับเครื่องที่พิมพ์ = ลำดับสาย (ตัวแรกเป็น head · วงแหวนเรียงตามสาย A→B→C)

**apply ทำอะไรบนแต่ละเครื่อง** — ตรวจรหัส sudo ของ *ทุก* เครื่องก่อนแตะเครื่องแรก → เขียน
`/etc/netplan/99-lmds-cluster.yaml` (`renderer: networkd` · `dhcp4: no` · `addresses` · `optional: true` ·
ไม่มี route/gateway · เฉพาะ interface ของคลัสเตอร์ สายบริหาร/Wi-Fi ไม่ถูกแตะ) → ไฟล์อื่นใน `/etc/netplan`
ที่อ้าง interface เดียวกัน (เช่น `99-nvidia-sync-cluster.yaml` ของ NVIDIA Sync ซึ่งชื่อเรียงหลังของเราและจะชนะ)
ถูกย้ายไป `/root/netplan-disabled/` ประทับเวลา → `netplan generate` + `netplan apply` → ยืนยันว่า `ip -br addr`
เห็น IP และ `LOWER_UP` (ลองซ้ำ ~18 วิ เพราะลิงก์กระพริบหลัง apply) → **ล้ม = ถอยกลับ**ไฟล์เดิมของเครื่องนั้นทันที
และไม่แตะเครื่องถัดไป → ping ทุกลิงก์จากทั้งสองปลาย → `lmds cluster pair` (กุญแจ head→worker บน IP ใหม่) →
iperf3 5 วิ ถ้ามีทั้งสองฝั่ง (เตือนเมื่อ <90 Gbit/s — เพดาน PCIe x4 คือ ~100 ไม่ใช่ 200) → ทะเบียน:
`cluster_ip`/`cluster_iface` = เส้นที่ head↔worker ใช้ + `cluster_links` ทุกลิงก์

**รันซ้ำได้** — IP เดิมที่เข้ากันอยู่แล้ว (เช่นคู่ที่ NVIDIA Sync ตั้งไว้) ถูกเก็บไว้ แผนขึ้นว่า "(เดิม)" · ไฟล์ของเรา
ถูกสำรองก่อนเขียนทับทุกครั้ง

**ข้อจำกัด** — carrier บอกได้แค่ว่า "ช่องนี้มีสาย" ไม่บอกว่าปลายอีกข้างคือใคร: วงแหวนและคู่ที่เสียบสองสาย
เป็น *สมมติฐาน* ตามผัง NVIDIA ซึ่ง ping ตอน apply เป็นคนยืนยัน (ping ไม่ถึง = เสียบไขว้ สลับสายหรือเรียง
ลำดับเครื่องใหม่) · วงแหวนทำให้ head/worker บางคู่อยู่คนละวง — `lmds cluster doctor` ข้อ same-subnet จะเตือน
ส่วน NCCL หลายลิงก์อ่านจาก `cluster_links` · ไฟล์ netplan บนเครื่องเป็น 0600 ของ root: `inspect` เห็นแค่ชื่อไฟล์
(รู้ว่ามีของ NVIDIA Sync ไหม) ไม่เห็นเนื้อหา · ถอนใช้ `netplan apply` ไม่ใช่ `netplan try` (ต้องมี tty)

## ลิงก์ขึ้นแล้ว แต่ขึ้นที่เท่าไร

`/sys/class/net/<iface>/speed` รายงานความเร็วที่ **negotiate ได้** ไม่ใช่ความสามารถของการ์ด
พอร์ต 200G ที่ต่อผ่าน switch แล้วปล่อยให้ auto-negotiate มักลงมาเหลือ 50G — ลิงก์ขึ้น ping ผ่าน
NCCL วิ่งได้ ทุกอย่างดูปกติ แต่ช้ากว่าที่ควรสี่เท่า

NVIDIA ตรวจรับลิงก์ระหว่าง Spark ที่ **≥184 Gbit/s** ต่อเส้น `lmds node cluster` จะเตือนเองเมื่อ
เจอ ConnectX ของ Spark ที่ต่ำกว่านั้น — เตือนอย่างเดียว ไม่ตัดเครื่องออกจากกลุ่ม เพราะยังใช้ได้จริง

```bash
cat /sys/class/net/enp1s0f1np1/speed    # 200000 = 200G · 50000 = ต้องไปแก้ที่ switch
```

แก้ที่ port ของ switch ให้ตั้ง 200 Gbps ตายตัว อย่าปล่อย auto ส่วน throughput จริงที่วัดได้
ราว 100 Gbps ต่อลิงก์เป็นเพดานของ PCIe Gen5 x4 ไม่ใช่การตั้งค่าผิด

เพดานจำนวนเครื่อง: ต่อสายตรงถึงกันได้สูงสุด **3 เครื่อง** เกินกว่านั้นต้องผ่าน switch ซึ่งรองรับถึง
**4 เครื่อง** — ที่มา [DGX Spark clustering](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
สรุปเทียบกับของเราอยู่ที่ [NVIDIA-CLUSTER-SOURCES.md](NVIDIA-CLUSTER-SOURCES.md)

## 8 · 3 เครื่องวงแหวน / 4 เครื่องผ่าน switch — หลายสายต่อเครื่อง (cluster.env v2)

DGX Spark มี QSFP **สองช่อง** (ช่อง 1 = `enp1s0f0np0`/`enp1s0f1np1` · ช่อง 2 = `enP2p1s0f0np0`/`enP2p1s0f1np1` ·
RoCE `rocep1s0f0`/`roceP2p1s0f0`) — ช่องละหนึ่ง function ถือสาย · 2 เครื่องต่อตรง = สายเดียว วงเดียว (ที่รันจริงมาตลอด)
· **3 เครื่องต่อตรงเป็นวงแหวน** A.ช่อง1→B.ช่อง2 · B.ช่อง1→C.ช่อง2 · C.ช่อง1→A.ช่อง2 = ทุกเครื่องมี **2 สาย 2 วง** และ head
ถึง worker แต่ละตัวด้วย**คนละ interface/IP** · **4 เครื่องผ่าน switch** = ทุกเครื่องสายเดียว วงเดียว · ~100 Gb/s ต่อสายคือปกติ
(เพดาน PCIe x4) · การต่อสาย/ตั้ง IP ทำผ่าน wizard ข้างบน (§Cluster network setup) ซึ่งบันทึกทุกสายลงทะเบียนเป็น `cluster_links`

**สิ่งที่ schema เดิมบอกไม่ได้**: `MASTER_IP`/`WORKER_IPS`/`NCCL_SOCKET_IFNAME` ตัวเดียว → NCCL ของ head ได้สายเดียว
หาทางไป worker ที่อยู่อีกสายไม่เจอ (ค้างที่ init เงียบ ๆ) · `cluster.env` **v2** จึงเพิ่มคีย์ต่อ rank โดยยังเขียนคีย์เดิมครบ —
bundle 2 เครื่องแบบเดิมได้ไฟล์เดิม**ทุกตัวอักษร** · controller ที่ไม่เห็นคีย์ v2 ทำงานแบบเดิมทุกประการ

```bash
CLUSTER_ENV_SCHEMA=2
CLUSTER_TOPOLOGY=ring-3                     # direct-2 | ring-3 | switch-N
CLUSTER_NODES="spark-a spark-b spark-c"     # rank 0 = head
LINKS_0="enp1s0f0np0:10.100.152.1/24:1:10.100.152.2 enP2p1s0f0np0:10.100.154.2/24:2:10.100.154.1"
NCCL_SOCKET_IFNAMES_0=enp1s0f0np0,enP2p1s0f0np0     # comma list — head ประกาศทุกสาย
NCCL_IB_HCAS_0=rocep1s0f0,roceP2p1s0f0              # ว่างได้ → controller เดิน sysfs หาเอง ทีละ interface
LINKS_1="enP2p1s0f0np0:10.100.152.2/24:0:10.100.152.1 enp1s0f0np0:10.100.153.1/24:2:10.100.153.2"
HEAD_TO_WORKER_IP_1=10.100.152.2   # IP ของ worker rank 1 ที่ head ใช้ถึง (ssh · rsync · VLLM_HOST_IP ของ worker)
WORKER_HEAD_IP_1=10.100.152.1      # IP ของ head ที่ worker rank 1 ต่อกลับ (--master-addr ของ worker ตัวนั้น)
HEAD_TO_WORKER_IP_2=10.100.154.1   # rank 2 อยู่อีกสาย อีกวง — ไม่ใช่ 152.x
WORKER_HEAD_IP_2=10.100.154.2
NCCL_CROSS_NIC=1                   # ring เท่านั้น: คู่ A-B วิ่งสายหนึ่ง คู่ A-C อีกสาย NCCL ต้องเลือก NIC ต่อคู่ได้
```

รูปแบบ `LINKS_<rank>` = `iface:ip/prefix:peer_rank:peer_ip` คั่นช่องว่าง · สายเข้า switch เขียน `peer_rank *` และ `peer_ip -`

**controller ทำอะไรต่างจากเดิมเมื่อเห็นคีย์ v2**

- head: `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`NCCL_IB_HCA` เป็น comma list ของ**ทุกสาย** · `NCCL_CROSS_NIC=1` ตามไฟล์ ·
  `check_running_on_master` ตรวจว่า head ถือ IP ของทุกสายใน `LINKS_0` (สายหลุดหนึ่งเส้น = worker rank นั้นถึง head ไม่ได้)
- worker rank N (`worker.sh` แต่ละตัวต่างกัน): `VLLM_HOST_IP=HEAD_TO_WORKER_IP_N` · `--master-addr WORKER_HEAD_IP_N` ·
  `NCCL_SOCKET_IFNAME=NCCL_SOCKET_IFNAMES_N` · HCA ถามที่ worker เองทีละ interface (`/sys/class/infiniband`) แล้ว comma-join
  (TCPStore ของ rank 0 ฟังทุก interface — ที่ต้องต่างกันคือฝั่งที่ต่อเข้าไป)
- ssh/rsync/`_check_worker_ssh` ไป worker ทาง `HEAD_TO_WORKER_IP_<rank>` (`WORKER_IPS` ถูก derive จากคีย์นี้ถ้าไม่ได้ตั้งเอง) ·
  `sync-worker` เลือกสายที่เร็วที่สุดที่ถึง worker นั้นได้ (มีหลายทางเฉพาะกรณีต่อสองเส้นเข้า switch เดียวกัน)
- `network-info` / `status` / `doctor` พิมพ์ตารางสายของทุก rank: `link=up speed=200000 ip=yes hca=rocep1s0f0 ping=ok` โดย
  **ping คู่ปลายสายออกทางสายนั้น** (`ping -I <iface>`) ทั้งจาก head และจาก worker ผ่าน ssh · `doctor` ล้มเมื่อมี `ping=FAIL` /
  `link=down` / `ip=no`
- `serve-args` และ `start --dry-run` แสดง argv + env ของ**ทุก rank** แยกกัน (`--tensor-parallel-size 3` · worker 2 ตัวคนละ
  `VLLM_HOST_IP`/`--master-addr`) โดยไม่แตะ docker/ssh — ใช้ตรวจก่อนสั่งจริง หรือประกอบจากค่าตัวอย่างก่อนมีคลัสเตอร์ก็ได้

```bash
lmds node run spark-a "deploy <model> --target dgx-spark-stacked-3 --no-llm --yes"   # TP=3 · NNODES=3
lmds node cluster --write <slug> --on spark-a          # เขียน cluster.env v2 จาก cluster_links ในทะเบียน
lmds node ctl spark-a <slug> start --dry-run           # ดู worker.sh ของ rank 1/2 และ docker run ของ head
lmds node ctl spark-a <slug> doctor                    # ทุกสายขึ้น + ping ถึงคู่ปลายสาย ก่อนปล่อย worker
```

**route ที่วงแหวนต้องมี** — torch/NCCL bootstrap ให้ทุก rank ต่อไปที่ *ที่อยู่เดียว* ของ rank 0 (สายที่ rank 0 ประกาศเป็นตัวแรก)
· worker ที่อยู่อีกวงถึงที่อยู่นั้นไม่ได้ถ้าไม่มี route — เพิ่ม `/32` ไปยัง IP อีกฝั่งของแต่ละคู่ผ่านสายตรง (Linux ตอบ ARP/รับ
packet ของ IP ตัวเองบนทุก interface อยู่แล้ว ไม่ต้องเปิด forwarding) เช่นบน C: `ip route add 10.100.152.1/32 via 10.100.154.2`
(IP ของ A บนสาย A-B ผ่านสาย C-A) และ `ip route add 10.100.152.2/32 via 10.100.153.2` — ทำครบทั้ง 3 เครื่อง (เครื่องละ 2 route)
แล้ว `doctor` ต้อง `ping=ok` ทุกสาย · wizard ยังไม่เขียน route ให้ (netplan `routes:` ทำได้ — ยังไม่ได้รันจริง)

**ยังไม่ได้พิสูจน์บนเครื่องจริง** — ฟลีตมี Spark คู่เดียว (2 เครื่อง สายเดียว) · ทุกอย่างในหัวข้อนี้ผ่านเทสกับ ssh/ip/ping/sysfs
ปลอมต่อเครื่อง (`tests/test_multilink_cluster.py`) แต่ยังไม่มี NCCL จริงวิ่งข้าม 3 เครื่อง · ถ้า ring ค้างที่ NCCL init ทั้งที่ doctor ผ่าน:
ลอง `NCCL_IB_DISABLE=1` (socket transport ใช้ route ของ kernel ตรง ๆ) แล้วค่อยไล่ GID/HCA · TP=3 ต้องหาร attention head ลงตัว
(ดู §7) — 3 เครื่องกับโมเดล 64/128 head ใช้ pipeline แทน
