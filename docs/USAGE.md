# คู่มือใช้งาน LMDS (ละเอียด)

> ก่อนอ่านเอกสารนี้ ต้องติดตั้งเสร็จตาม [INSTALL.md](INSTALL.md) แล้ว (`lmds version` ใช้ได้)

## แนวคิดหลัก 30 วินาที

LMDS รับ**ลิงก์โมเดล** (Hugging Face) → วิเคราะห์ + คำนวณว่า fit กับเครื่องไหม → สร้าง **bundle**
(โฟลเดอร์ + ZIP) ที่ข้างในมีสคริปต์ controller สำหรับ download / start / ทดสอบโมเดลนั้นบนเครื่องจริง

ทุก bundle ผ่าน quality gates ทุกด่านโดยอัตโนมัติ — ถ้าไม่ผ่านจะไม่มีไฟล์ออกมาให้ใช้เลย

```text
ลิงก์โมเดล ──lmds deploy──▶ bundle/ ──./xxx-single.sh──▶ โมเดลรันเป็น API ที่ :8000/v1
```

---

## รูปแบบคำสั่ง (อ่านก่อน)

ทุกครั้ง**ต้องมีคำสั่ง (subcommand) เสมอ** — ใส่ลิงก์โมเดลเฉย ๆ ไม่ได้:

```text
lmds <คำสั่ง> <ลิงก์โมเดล> [ตัวเลือก]
```

```bash
# ❌ ผิด — ไม่มีคำสั่ง → No such command 'https://...'
lmds https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF

# ✅ ถูก — มีคำสั่ง deploy
lmds deploy https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF --target dgx-spark-single
```

คำสั่งหลัก (ดูทั้งหมด: `lmds --help`):

| คำสั่ง | ทำอะไร |
|---|---|
| `lmds inspect <โมเดล>` | วิเคราะห์ + เช็ก fit อย่างเดียว — ไม่สร้างไฟล์ ไม่เสีย token |
| `lmds plan <โมเดล>` | ดู Deployment Plan (แผน) — ไม่สร้างไฟล์ |
| `lmds deploy <โมเดล>` | flow เต็ม: วิเคราะห์ → วางแผน → **ยืนยัน** → สร้าง bundle + ZIP |
| `lmds generate <โมเดล>` | เหมือน deploy แต่**ข้ามขั้นยืนยัน** |
| `lmds ps` / `list` | ดูว่ามีอะไรอยู่บ้าง + สถานะจริง (ดู §4) |
| `lmds start` / `stop` / `restart` / `logs` | สั่งงานโมเดลตามชื่อ (ดู §4) |
| `lmds enable` / `disable` | autostart หลัง reboot (ดู §4) |
| `lmds repair <ชื่อ>` | โหลดไฟล์ที่ขาด/เสียกลับมา แล้วตรวจซ้ำ (ดู §4.3) |
| `lmds remove <ชื่อ>` | ลบโมเดลออกจากเครื่องทั้งหมด (ดู §4.3) |
| `lmds doctor <ชื่อ>` | ตรวจว่าทำไม download/start ไม่ผ่าน + คำสั่งแก้ (ดู §4.4) |
| `lmds web` | หน้าเว็บคุมทุกอย่าง — UI ภาษาอังกฤษ (ดู §5) |
| `lmds validate <โฟลเดอร์>` | ตรวจ bundle ย้อนหลัง |
| `lmds hardware` | ตรวจเครื่อง + จำแนก target profile |
| `lmds config ...` | ตั้ง provider / key / HF token |

**ช่อง `<โมเดล>` ใส่ได้ 3 แบบ:**

| แบบ | ตัวอย่าง | พฤติกรรม |
|---|---|---|
| `org/model` | `Qwen/Qwen3-32B` | repo บน Hugging Face |
| ลิงก์ repo เต็ม | `https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF` | ถ้าเป็น **GGUF หลาย quant → ระบบให้เลือกไฟล์ (พิมพ์หมายเลข)** |
| ลิงก์ไฟล์ตรง | `.../blob/main/gemma-4-...-Q4_K_M.gguf` | ใช้ไฟล์นั้นเลย ไม่ต้องเลือก |

> ยังไม่ได้ตั้ง LLM provider? เติม `--no-llm` ท้ายคำสั่ง (rule-based mode — ฟรี ไม่ต้องมี key)

---

## 1. ครั้งแรกสุด: ทดสอบระบบด้วยโมเดลเล็ก (แนะนำอย่างยิ่ง)

ใช้โมเดล 0.6B (~0.4GB) เพื่อพิสูจน์ว่าทั้ง loop ทำงานก่อนไปโมเดลใหญ่:

```bash
lmds deploy "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf" --no-llm
```

ระบบจะแสดง **Deployment Plan** ให้ยืนยัน:

```text
┌───────────────────┬──────────────────────────────────────────┐
│ Generator         │ rule-based                               │
│ Revision (pinned) │ 50968a4468ef4233ed78cd7c3de230dd1d61a56b │
│ Runtime           │ llamacpp — ghcr.io/ggml-org/llama.cpp:…  │
│ Serving           │ context 32,768 | max output 8,192 | …    │
└───────────────────┴──────────────────────────────────────────┘
context (Enter = ใช้ค่าตามแผน) [32768]:        ← กด Enter
สร้าง bundle ตามแผนนี้? [Y/n]:                  ← กด Enter
```

เสร็จแล้วได้:

```text
bundles/qwen3-0-6b-gguf/
├── qwen3-0-6b-gguf-single.sh    ← controller (สคริปต์หลัก)
├── README.md                    ← คู่มือเฉพาะของ bundle นี้
├── MODEL_PROFILE.yaml           ← สเปกทั้งหมด (source of truth)
├── SPECIAL_FILES.md             ← ไฟล์พิเศษ/ข้อควรระวัง
└── PACKAGE_SHA256SUMS           ← checksum ทุกไฟล์
bundles/qwen3-0-6b-gguf.zip      ← สำหรับส่งมอบ/ก๊อปไปเครื่องอื่น
```

## 2. รันโมเดลจาก bundle

```bash
cd bundles/qwen3-0-6b-gguf

./qwen3-0-6b-gguf-single.sh download        # โหลดโมเดล (resume ได้ถ้าเน็ตหลุด)
./qwen3-0-6b-gguf-single.sh verify-files    # ตรวจขนาด exact + GGUF magic + SHA-256
./qwen3-0-6b-gguf-single.sh start           # เปิดเซิร์ฟเวอร์ — รอจนขึ้น "started: ..."
./qwen3-0-6b-gguf-single.sh test-text       # ให้โมเดลตอบ 1 คำถามทดสอบ
```

`test-text` ตอบ JSON ที่มีข้อความจากโมเดล = **สำเร็จ** 🎉

เอา endpoint ไปต่อกับแอป/n8n/OpenWebUI:

```bash
./qwen3-0-6b-gguf-single.sh client-config
```

```json
{
  "base_url": "http://192.168.1.50:8000/v1",
  "model": "qwen3-0-6b-gguf",
  "max_input_tokens": 22528,
  "max_output_tokens": 8192
}
```

ปิดเซิร์ฟเวอร์: `./qwen3-0-6b-gguf-single.sh stop`

### คำสั่งทั้งหมดของ controller

> **ไม่อยากพิมพ์ทีละคำสั่ง?** `lmds up <ลิงก์>` เดินให้ทั้งชุดตามลำดับที่ถูกต้อง
> (deploy → download → verify-files → prepare-runtime ถ้าจำเป็น → start → test-text)
> แล้วบอกวิธีต่อ client ให้ · ล้มขั้นไหนหยุดตรงนั้นพร้อมบอกว่าดูต่อที่ไหน
> · stacked ยังต้องใช้ `lmds deploy` แล้วทำตาม README ของ bundle

| คำสั่ง | หน้าที่ |
|---|---|
| `download` | ดาวน์โหลดโมเดล (pin revision, resume ได้) |
| `verify-files` | ตรวจความครบถ้วน/ความถูกต้องของไฟล์ |
| `prepare-runtime` | เตรียม engine — **จำเป็นเฉพาะบางกรณี ดูด้านล่าง** |
| `start` / `stop` / `restart` | เปิด-ปิดเซิร์ฟเวอร์ (ตรวจ GPU + ไฟล์ก่อน start เสมอ) |
| `status` | สถานะ container + API health |
| `logs [N]` | log ล่าสุด N บรรทัด (default 300) |
| `client-config` | ค่าตั้ง client เป็น JSON พร้อม token budget |
| `network-info` | bind address + endpoint ที่ประกาศให้ client |
| `test-text` | ทดสอบ chat completion หนึ่งครั้ง |
| `test-vision` | *(เฉพาะโมเดล multimodal)* สร้างภาพสีแดงแล้วถามว่าเห็นสีอะไร — พิสูจน์ว่า mmproj โหลดจริง |
| `wait-health` | รอ `/health` ต่อ (ใช้เมื่อ start timeout แต่โมเดลยังโหลดอยู่) |

> **คำอธิบายเต็มของทุก option + วิธีตั้ง API token อยู่ใน help ของ controller เอง** (ภาษาอังกฤษ):
> `./xxx-single.sh` เปล่า ๆ หรือ `./xxx-single.sh help` — มีค่า default จริงของ bundle นั้นกำกับทุกบรรทัด

> **`prepare-runtime` ต้องรันเมื่อไหร่?** ตอน deploy เสร็จ ระบบจะพิมพ์ลำดับคำสั่งที่ถูกต้องของ bundle นั้นให้เสมอ — ทำตามนั้นได้เลย
> - **GGUF บน DGX Spark (ARM64)** — จำเป็น ✅ เพราะไม่มี Docker image ทางการ ต้อง build llama.cpp จาก source (ครั้งแรกครั้งเดียว ~10–30 นาที, ขอ sudo ติดตั้ง build deps)
> - **stacked (2 เครื่อง)** — จำเป็น ✅ เพื่อ pull + ล็อค image-ID ให้ตรงกันทั้งสอง node
> - **GGUF/safetensors บน x86_64 (RTX)** — ไม่ต้อง ใช้ image ทางการได้เลย
>
> รายละเอียดว่าเกิดอะไรขึ้นเบื้องหลัง: [INSTALL.md §4](INSTALL.md)

### options ที่ทุก controller รองรับ (ใส่ท้ายคำสั่งใดก็ได้)

```bash
./xxx-single.sh start --port 8001                  # เปลี่ยน port
./xxx-single.sh start --context 16384              # ลด context (ประหยัด memory)
./xxx-single.sh start --bind 127.0.0.1             # ให้เข้าถึงได้เฉพาะในเครื่อง (default คือ 0.0.0.0)
./xxx-single.sh start --advertise-ip 10.0.0.5      # IP ที่ประกาศให้ client (ไม่ใช่ bind)
./xxx-single.sh start --interface eth1             # เลือก interface ที่ใช้ประกาศ IP
./xxx-single.sh client-config --client-output 4096 # ปรับ token budget
```

### env ที่ควรรู้ (ใส่นำหน้าคำสั่ง หรือ export ไว้ก่อน)

ทุก flag ด้านบนมี env คู่กัน และมีอีกหลายตัวที่ตั้งได้เฉพาะทาง env:

| env | ค่า default | ใช้ทำอะไร |
|---|---|---|
| `API_PORT` / `API_HOST` | `8000` / `0.0.0.0` | port และ bind address |
| `API_KEY` | *(ว่าง)* | บังคับ Bearer token — **ควรตั้งเสมอถ้าเปิดออก network** |
| `MAX_MODEL_LEN` | ตามแผน | context (เท่ากับ `--context`) |
| `HF_HOME` | `~/.cache/huggingface` | ที่เก็บ weight ของ **vLLM** — ย้ายลงดิสก์ใหญ่ได้ |
| `MODEL_DIR` | `~/models/<slug>` | ที่เก็บไฟล์ **GGUF** ของ llama.cpp |
| `RUNTIME_MODE` | ตามเครื่อง | `docker` หรือ `native` (llama.cpp เท่านั้น) |
| `HF_TOKEN` | *(ว่าง)* | ใช้ตอน `download` repo gated |
| `HEALTH_TIMEOUT` | ตามขนาดโมเดล | วินาทีที่รอ `/health` ตอน start |
| `GPU_MEMORY_UTILIZATION` | ตามแผน | สัดส่วน VRAM ที่ vLLM จองได้ (ลดถ้าแชร์ GPU กับงานอื่น) |
| `MAX_NUM_SEQS` | ตามแผน | จำนวน request พร้อมกันสูงสุด |
| `CONTAINER_NAME` | `lmds-<slug>` | ชื่อ container |
| `RUN_DIR` | `~/.lmds/run/<slug>` | ทะเบียน + log ที่ `lmds ps`/`lmds logs` อ่าน |

```bash
API_PORT=8001 API_KEY=secret123 ./xxx-single.sh start
HF_HOME=/data/hf-cache ./xxx-single.sh download     # ต้องใส่ตอน start ด้วยเสมอ
```

---

## 3. Deploy โมเดลจริง

### 3.1 เช็คก่อนว่า fit ไหม (ไม่สร้างไฟล์ ไม่เสีย token LLM)

```bash
lmds inspect Qwen/Qwen3-32B                              # เทียบกับเครื่องปัจจุบัน
lmds inspect Qwen/Qwen3-32B --target dgx-spark-single --target rtx-pro-4000-dual
```

ผลจริง (Qwen3-32B BF16, 65GB):

```text
┃ Target                   ┃ ผล                     ┃ รายละเอียด            ┃
│ rtx-pro-4000 (vllm)      │ ❌ needs-smaller-quant │ weights 61 / 17.9 GB │
│ dgx-spark-single (vllm)  │ ✅ fits                │ context แนะนำ 32,768  │
```

อ่านผล:
- `✅ fits` — deploy ได้เลย
- `❌ needs-smaller-quant` — ตัวเต็มไม่พอ ระบบจะแนะ GGUF quant ที่พอแทน
- `🟡 fits-with-offload` — รันได้แบบแบ่งลง RAM (ช้าลงมาก — เหมาะทดสอบ ไม่เหมาะ production)

### 3.2 Deploy

```bash
lmds deploy Qwen/Qwen3-32B --target dgx-spark-single
lmds deploy "https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-Q4_K_M.gguf" --target rtx-pro-4000
```

> **repo GGUF ที่มีหลาย quant** (ให้ลิงก์ repo ไม่ใช่ลิงก์ไฟล์) — ระบบจะแสดงรายการ variant พร้อมขนาด ให้**พิมพ์หมายเลขเลือก** ก่อน เช่น `lmds deploy https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF --target dgx-spark-single` แล้วเลือก Q4_K_M/Q5_K_M/… ตามที่พอกับเครื่อง · ถ้ารู้ไฟล์อยู่แล้ว ใส่ลิงก์ไฟล์ตรง (`.../blob/main/xxx.gguf`) เพื่อข้ามการเลือก

### 3.3 โมเดล gated (เช่น Llama)

```bash
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single
# → ระบบตรวจพบว่า gated แล้วถาม:
#   Hugging Face token (Enter เพื่อข้าม):
```

ต้องกดยอมรับเงื่อนไขของโมเดลบนเว็บ huggingface.co ด้วย account เดียวกับ token ก่อน ไม่งั้น token ก็เข้าไม่ได้
ตอน `download` บนเครื่อง ให้ตั้ง `export HF_TOKEN=hf_xxx` ไว้ใน shell (สคริปต์อ่านจาก env — ไม่ฝังในไฟล์)

### 3.3b Deploy แบบ stacked (โมเดลใหญ่เกิน 1 เครื่อง → 2× DGX Spark)

โมเดลที่ใหญ่เกิน unified memory ของ Spark เครื่องเดียว (เช่น DeepSeek-V4-Flash ~168GB) ให้ใช้ target `dgx-spark-stacked` — lmds จะสร้าง controller แบบ **multi-node** (worker-first startup, TP ข้าม node, mp backend) แทนแบบเดี่ยวอัตโนมัติ

```bash
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 --target dgx-spark-stacked
# → ได้ bundle: <slug>-stacked.sh (ไม่ใช่ -single.sh)
```

controller ที่ได้มีคำสั่งครบวงจร multi-node — รันจาก **master node ในฐานะ user ปกติ (ห้าม sudo)**:

```bash
cd bundles/<slug>
# แก้ CONFIG ต้นไฟล์ก่อน: MASTER_IP, WORKER_IP, SSH_USER, NCCL_SOCKET_IFNAME, NCCL_IB_HCA
./<slug>-stacked.sh prepare-runtime   # pull + lock image ให้ image-ID ตรงกันทั้งสอง node
./<slug>-stacked.sh download          # ดาวน์โหลดโมเดลลง master
./<slug>-stacked.sh verify-files      # ตรวจ shard + config
./<slug>-stacked.sh sync-worker       # rsync โมเดล → worker
./<slug>-stacked.sh verify-worker     # ตรวจ shard บน worker
./<slug>-stacked.sh start             # เปิด worker (rank 1) ก่อน แล้ว head (rank 0) + รอ /health
./<slug>-stacked.sh status
```

> **สถานะ**: bundle stacked ที่ LMDS สร้างยังเป็น `static-validated` — ยังไม่เคยรันจริงบนคลัสเตอร์
> (template port มาจาก reference ที่ hardware-validated แล้ว แต่ตัวที่ generate เองยังไม่ได้พิสูจน์)
> · `runtime_assets` (parser plugin) ยังไม่รองรับในโหมด stacked

ข้อกำหนด: 2× DGX Spark + fabric ระหว่าง node (แนะนำ 200 Gb/s RoCE) + passwordless SSH (master→worker) · `lmds ps`/`lmds stop`/`lmds logs` เห็น/สั่งงานตัวนี้ได้เหมือน deploy เดี่ยว (stop จะหยุดทั้งสอง node ให้) · stacked รองรับเฉพาะ vLLM (GGUF ยังไม่มี reference ที่ทดสอบแล้ว)

### 3.4 Target presets ที่มีให้เลือก

**ทดสอบบนเครื่องจริงแล้ว** (✅ tested — ใช้สูตรคำนวณปกติ):

| preset | เครื่อง | หน่วยความจำ |
|---|---|---|
| `dgx-spark-single` | DGX Spark 1 เครื่อง | unified 128GB |
| `dgx-spark-stacked` | DGX Spark 2 เครื่อง | unified 128GB × 2 — สร้าง controller multi-node (ดู 3.3b) |
| `rtx-pro-4000` / `rtx-pro-4000-dual` | RTX PRO 4000 Blackwell ×1/×2 | 24GB / 24GB×2 |
| `rtx-4070-super` / `rtx-4070-ti-super` | RTX 4070 Super / Ti Super | 12GB / 16GB |
| `rtx-5090` | RTX 5090 (Blackwell SM120) | 32GB — validated 2026-08-03 (gemma-4-12b-it GGUF + vision) |

**ยังไม่ได้ทดสอบจริง** (ระบบลด budget ให้อัตโนมัติแบบ conservative):

| preset | VRAM | | preset | VRAM |
|---|---|---|---|---|
| `rtx-5080` | 16GB | | `rtx-4090` | 24GB |
| `rtx-5070-ti` | 16GB | | `rtx-4080-super` / `rtx-4080` | 16GB |
| `rtx-5070` | 12GB | | `rtx-4070-ti` | 12GB |
| `rtx-5060-ti` | 16GB | | `rtx-4060-ti` | 16GB |
| | | | `rtx-3090-ti` / `rtx-3090` | 24GB |
| | | | `rtx-3080-ti` / `rtx-3080` | 12GB / 10GB |
| | | | `rtx-3060` | 12GB |

| preset | เครื่อง |
|---|---|
| *(ไม่ระบุ `--target`)* | ใช้เครื่องที่รันคำสั่งอยู่ — ตรวจอัตโนมัติด้วย `lmds hardware` |

> ระบุได้หลาย target พร้อมกันเฉพาะกับ `lmds inspect` (เทียบให้เห็นภาพ) ส่วน `deploy`/`generate` รับได้ทีละอัน

### 3.5 โหมดและ options ของ `lmds deploy`

```bash
--no-llm            # ไม่เรียก LLM (rule-based) — ฟรี, เร็ว, แต่ไม่วิเคราะห์ parser/feature เชิงลึก
--yes / -y          # ข้ามขั้นยืนยันทั้งหมด (สำหรับ script/CI) — flag ค้างอนุมัติจะไม่ถูกใส่
--output DIR        # เปลี่ยนที่เก็บ bundle (default: ./bundles)
--revision SHA      # ล็อค revision เอง (default: ล่าสุด ณ ตอนนั้น แล้ว pin ให้)
--target PRESET     # เครื่องเป้าหมาย (ดู 3.4) — ว่าง = เครื่องที่รันคำสั่งอยู่
--concurrency N     # จำนวน request พร้อมกันที่ใช้คำนวณ KV cache (default 1)
```

> **`--concurrency` มีผลกับ memory โดยตรง** — KV cache โตตามจำนวน request ที่รันพร้อมกัน
> ใส่ `--concurrency 4` แปลว่า "กันหน่วยความจำเผื่อ 4 คนใช้พร้อมกัน" ผลคือ context ที่แนะนำจะลดลง
> ตั้งให้ตรงกับการใช้งานจริง: เดโม่/คนเดียว = 1 · ทีมเล็ก = 2–4 · ตั้งสูงเกินจริงจะได้ context สั้นโดยไม่จำเป็น

### 3.6 ขั้นยืนยัน — จุดที่ต้องอ่านก่อนกด

1. **อนุมัติ flag นอก allowlist** — ถ้าแผนเสนอ flag พิเศษ (เช่น `--trust-remote-code`) ระบบถามทีละตัว
   ค่า default คือ**ไม่อนุมัติ** — อนุมัติเฉพาะเมื่อเข้าใจผลของ flag นั้น (อ่าน SPECIAL_FILES.md ประกอบ)
2. **context** — Enter ใช้ค่าที่คำนวณให้ หรือพิมพ์เลขใหม่ (เกินเพดานปลอดภัยระบบจะลดให้อัตโนมัติ)
3. **ยืนยันสร้าง bundle** — Y/n

---

## 4. จัดการหลายโมเดลในเครื่องเดียว (Fleet)

รันหลายโมเดลพร้อมกันได้ (คนละ port) — ไม่ต้องจำว่า bundle ไหนอยู่ที่ไหน ใช้ `lmds` เป็นศูนย์กลาง:

```bash
lmds ps                  # เครื่อง + ใครรันอยู่บ้าง: สถานะ ● running / ◐ loading / ○ stopped + endpoint
lmds list                # bundle ทั้งหมด + สถานะ + engine/port/context/ฟีเจอร์ + autostart
lmds start <ชื่อ>         # รันโมเดลที่เคย deploy ไว้ (เช่น หลัง reboot)
lmds stop <ชื่อ>          # หยุดตามชื่อ — ไม่ต้อง cd ไปหา .sh
lmds stop --all          # หยุดทุกตัวที่รันอยู่
lmds restart <ชื่อ>       # restart (ใช้ตอนอยากเปลี่ยน option เช่นเพิ่ม API_KEY)
lmds logs <ชื่อ> -n 500   # ดู log ย้อนหลัง
lmds logs <ชื่อ> -f       # ตาม log แบบ realtime (Ctrl-C ออก — ไม่หยุดโมเดล)
lmds repair <ชื่อ>        # โหลดไฟล์ที่ขาดกลับมา + ตรวจซ้ำ
lmds remove <ชื่อ>        # ลบทิ้งทั้งหมด (ถามยืนยันก่อน)
```

**เปลี่ยน option ตอน start ได้เลย** — flag ที่ไม่ใช่ของ `lmds` จะถูกส่งต่อให้ controller ของ bundle นั้น:

```bash
lmds start <ชื่อ> --port 8001            # ย้าย port โดยไม่ต้องแก้ .sh
lmds start <ชื่อ> --context 32768        # ลดบริบทให้พอดีหน่วยความจำที่เหลือ
lmds restart <ชื่อ> --gpu-util 0.8       # ลดสัดส่วน VRAM แล้วเปิดใหม่
```

controller เป็นเจ้าของ flag พวกนี้และตรวจค่าเอง — `lmds` ไม่พยายามรู้จักทุกตัว เพราะแต่ละ engine
มีไม่เท่ากันและเปลี่ยนตามเวอร์ชัน · ดูว่า bundle นั้นรับ flag อะไรบ้าง: `lmds logs <ชื่อ>` หรือเปิด
`.sh` ของมันดูหัวไฟล์

**ชื่อ (slug) เอามาจากคอลัมน์แรกของ `lmds ps` / `lmds list`** — ทั้งสองคำสั่งจะพิมพ์ตัวอย่าง
คำสั่งพร้อมชื่อจริงให้ copy ไปใช้ได้เลย · พิมพ์ไม่ครบก็กด TAB ได้ (ดู §4.4)

ตัวอย่างรัน 2 โมเดลพร้อมกัน:

```bash
cd bundles/model-a && ./model-a-single.sh start                # port 8000
cd ../model-b && ./model-b-single.sh start --port 8001         # port 8001
lmds ps                                                        # เห็นทั้งคู่
lmds stop --all                                                # ปิดทั้งคู่จบในคำสั่งเดียว
```

> ระบบรู้จักเซิร์ฟเวอร์จากไฟล์ทะเบียนที่ controller เขียนเองตอน `start` (ใต้ `~/.lmds/run/`)
> — ถ้า controller ถูกลบ/ย้าย `lmds stop` ยัง fallback หยุดตรง ๆ ให้ได้ (kill pid / docker rm)

### 4.1 อ่านสถานะใน `lmds list`

| สัญลักษณ์ | หมายถึง |
|---|---|
| ● | รันอยู่และ API ตอบ health |
| ◐ | รันอยู่แต่ยังไม่ตอบ health (กำลังโหลดโมเดล) |
| ○ | หยุดอยู่ |
| ⚠ | หยุดอยู่ **และหาไฟล์ controller ไม่เจอ** — `start`/`restart` ใช้ไม่ได้ ต้อง deploy ใหม่ |

### 4.2 container ที่ไม่ได้ deploy ผ่าน LMDS

`lmds ps` สแกน `docker ps` ด้วย และรับ container ที่ image ตรงกับ engine ที่รู้จัก
(vLLM / llama.cpp / Ollama / TGI) เข้ามาในตาราง ทำเครื่องหมาย **⚙ ไม่ได้มาจาก lmds**
— container อื่นในเครื่อง (ฐานข้อมูล ฯลฯ) ไม่ถูกดึงเข้ามา

สั่งงานได้เหมือนกัน: `lmds stop` / `restart` / `logs` / `enable`

> **`stop` ของตัวภายนอกใช้ `docker stop` ไม่ใช่ `docker rm -f`** — ไม่ลบ container ของคุณทิ้ง
> `enable` ก็ทำได้ แต่ unit ที่ได้จะเป็นแค่ `docker start <container>` (ไม่ได้สร้าง container ใหม่)
> ถ้าลบ container นั้นทิ้ง unit จะล้ม ต้อง enable ใหม่

### 4.3 ซ่อม / ลบโมเดล

```bash
lmds repair <ชื่อ>                    # download (resume) → verify-files
```

ใช้เมื่อไฟล์หายหรือขนาดไม่ตรง (เช่น download ค้างกลางคัน, เผลอลบไฟล์ใน cache) —
โหลดเฉพาะส่วนที่ขาด ไม่โหลดใหม่ทั้งก้อน · ถ้า **controller หายไปแล้ว** ซ่อมไม่ได้
ต้อง `lmds deploy` ลิงก์เดิม (weight ที่โหลดไว้ยังใช้ต่อได้ ไม่ต้องโหลดซ้ำ)

```bash
lmds remove <ชื่อ>                    # ลบทั้งหมด
lmds remove <ชื่อ> --keep-weights     # ลบ bundle แต่เก็บ weight ไว้
lmds remove <ชื่อ> -y                 # ไม่ถามยืนยัน (สำหรับ script)
```

`remove` จะ **แสดงรายการไฟล์ + ขนาดให้ดูก่อนเสมอ** แล้วค่อยถามยืนยัน (default = ไม่ลบ)
สิ่งที่ทำตามลำดับ: หยุดเซิร์ฟเวอร์ → ยกเลิก autostart → ลบ bundle + ZIP + ทะเบียน/log +
runtime files + weight ของโมเดล

- **`--keep-weights` คุ้มมากกับโมเดลใหญ่** — ลบ bundle ทิ้งแล้ว deploy ใหม่ได้โดยไม่ต้องโหลดซ้ำหลายสิบ GB
- weight หาจาก `MODEL_PROFILE.yaml` (vLLM → HF cache, llama.cpp → `MODEL_DIR`) —
  ถ้าหาไม่เจอระบบจะ**ไม่เดา** (ไม่ลบอะไรที่ไม่แน่ใจ) ต้องลบเองถ้าต้องการ

### 4.4 Tab completion (กด TAB เติมให้)

ติดตั้งครั้งเดียวต่อเครื่อง — `install.sh` ถามให้อยู่แล้ว หรือรันเอง:

```bash
lmds --install-completion
```

แล้ว**เปิด terminal ใหม่** (หรือ `source ~/.bashrc`) · รองรับ bash / zsh / fish

```text
lmds depl<TAB>                       → lmds deploy
lmds stop qwen<TAB>                  → เติมชื่อ bundle ให้
lmds deploy <url> --target dgx<TAB>  → dgx-spark-single / dgx-spark-stacked
```

### ให้โมเดลกลับมาเองหลังเปิด-ปิดเครื่อง (autostart)

ปกติหลัง reboot โมเดลจะไม่ขึ้นเอง ต้อง `lmds start <ชื่อ>` เอง — ถ้าอยากให้**กลับมาทำงานอัตโนมัติ**
(เหมาะกับเครื่องลูกค้า/ทีมที่เปิดทิ้งไว้เป็น server) ใช้ systemd autostart:

```bash
lmds enable gemma-4-26b-a4b-it-gguf          # ตั้ง autostart (ขอ sudo เขียน systemd unit)
lmds enable gemma-4-26b-a4b-it-gguf --now    # ตั้ง + start เดี๋ยวนี้เลย
lmds list                                    # ดูคอลัมน์ autostart: ● เปิด / ○ ปิด
lmds disable gemma-4-26b-a4b-it-gguf         # ยกเลิก autostart
```

- ทำงานผ่าน **systemd system service** (`lmds-<ชื่อ>.service`) — รันเป็น user เจ้าของ bundle, เปิดหลัง `docker.service` พร้อม, เคลียร์ container ค้างก่อน start เสมอ
- โมเดลใหญ่ที่โหลดนาน เพิ่มเวลา: `lmds enable <ชื่อ> --timeout 3600`
- เช็ก/ดู log ของ service: `systemctl status lmds-<ชื่อ>` · `journalctl -u lmds-<ชื่อ> -f`
- ต้องมี `systemd` (DGX OS/Ubuntu มีอยู่แล้ว) · **stacked (2 เครื่อง):** master ตั้ง autostart ได้ แต่ตอน boot worker ต้องเปิดอยู่ + SSH ถึงได้ ไม่งั้น start จะรอ/ล้ม

## 4.5 คุมหลายเครื่องจากเครื่องเดียว (fleet หลายเครื่อง)

หน้างานที่มีมากกว่า 1 เครื่อง — แทนที่จะ ssh ไล่ทีละตัว ให้เครื่องที่คุณนั่งอยู่ (**hub**) คุมเครื่องอื่นทั้งหมด

### ⚠️ ทุกเครื่องที่จะคุมต้องมี LMDS อยู่บนเครื่องนั้น

hub ไม่ได้ส่ง agent ไปรันบนเครื่องปลายทาง — มันเรียก `lmds agent info` **ที่ติดตั้งอยู่บนเครื่องนั้น**
ผ่าน SSH ("agent" ของระบบนี้คือตัวคำสั่ง `lmds` เอง ไม่ใช่โปรเซสที่รันค้าง) ฉะนั้น:

```bash
lmds node add 192.168.10.21 --user ops --install   # เพิ่ม + ติดตั้ง LMDS ให้เลยในคำสั่งเดียว
lmds node install spark2                           # ติดตั้ง/อัปเดตทีหลังก็ได้
```

`node install` จะ clone (หรือ `git pull`) จาก GitHub แล้วรัน `install.sh` บนเครื่องนั้นให้
· **ข้ามขั้น Docker/NVIDIA toolkit** เพราะต้องใช้ `sudo` ซึ่งไม่มีคนกรอกรหัสผ่านผ่าน SSH

### เตรียมเครื่องปลายทาง

1. sshd เปิดอยู่
2. user ที่จะใช้ **อยู่ในกลุ่ม `docker`** — **ไม่ต้องเป็น root**
3. Docker + NVIDIA Container Toolkit + git — ถ้ายังไม่มี ต้องรัน `./install.sh` บนเครื่องนั้นเอง (ขั้น sudo)
4. LMDS — ไม่ต้องทำเอง ใช้ `lmds node install` จาก hub ได้

### เพิ่มเครื่อง (กรอกรหัสผ่านครั้งเดียว)

```bash
lmds node add 192.168.10.21 --user ops
# ถามรหัสผ่าน → ติดตั้ง SSH key ของ LMDS → ทิ้งรหัสผ่านทันที
```

รหัสผ่าน**ไม่ถูกบันทึกลงดิสก์** และทะเบียน (`~/.config/lmds/nodes.yaml`, สิทธิ์ 0600) ไม่มีฟิลด์รหัสผ่าน
ตั้งแต่แรก · ตั้งแต่นั้นไปใช้ key ของ LMDS เอง (`~/.config/lmds/id_lmds`) แยกจาก key ส่วนตัวคุณ

### ใช้งานประจำวัน

```bash
lmds node list                      # ทะเบียนทั้งหมด (เร็ว — อ่านไฟล์)
lmds node list --check              # ต่อจริง ดูว่าเครื่องไหนยังตอบ
lmds ps --all                       # โมเดลของทุกเครื่องในตารางเดียว
lmds node remove spark2             # ออกจากทะเบียน (ไม่แตะเครื่องนั้น)
```

### สองคำสั่งที่ต้องแยกให้ออก

```bash
lmds node run spark2 doctor my-model          # สั่ง "คำสั่งของ lmds" บนเครื่องนั้น
lmds node ctl spark2 my-model prepare-runtime # สั่ง "สคริปต์ controller" ในตัว bundle
```

| | ใช้กับ |
|---|---|
| `node run` | `ps` `start` `stop` `restart` `logs` `doctor` `repair` `deploy` `scan` |
| `node ctl` | `prepare-runtime` `download` `verify-files` `sync-worker` `verify-worker` `test-text` `network-info` `client-config` `bench` `clear-fi-cache` |

ขั้นตอนของ stacked (`sync-worker`, `verify-worker`) มีเฉพาะใน controller — ต้องใช้ `node ctl`
· ลำดับเต็มดูที่ [RUNBOOK-MULTI-NODE.md](RUNBOOK-MULTI-NODE.md)

- เครื่องปลายทาง**ไม่ต้องรัน daemon** — hub เรียก `lmds agent info` ผ่าน SSH เอาสถานะเป็น JSON
- เครื่องหนึ่งล่ม เครื่องอื่นและหน้าเว็บไม่กระทบ (แถวนั้นขึ้นว่าติดต่อไม่ได้)

### ดูทรัพยากรของแต่ละเครื่อง

หน้าเว็บแสดงทรัพยากรเป็น**เกจชุดเดียวกันทั้งเครื่องนี้และเครื่องอื่น** — CPU · Unified/RAM ·
VRAM · Disk free และการ์ด GPU (compute/power/temp/fan + clocks + PCIe) อยู่ใต้กัน

- **ตัวเลขจริงอยู่ใต้ทุกเกจ** (`112 / 122 GB`) เพราะ 80% ของ 122 GB กับของ 8 GB ไม่ใช่เรื่องเดียวกัน
- **สีบอกก่อนที่จะไปเจอปัญหา**: หน่วยความจำ ≥75% เหลือง ≥90% แดง · ดิสก์กลับด้าน (เหลือ ≤15%
  เหลือง ≤5% แดง) เพราะคำถามจริงตอนจะโหลดโมเดล 150 GB คือ "เหลือพอไหม"
- **DGX Spark (unified) ไม่มีเกจ VRAM แยก** — GPU กินจาก pool เดียวกับ RAM โชว์แยกคือนับซ้ำ
  · การ์ดแยก (RTX) ถึงจะมีเกจ VRAM ของตัวเอง
- **จำนวนโมเดลที่รันอยู่** เป็นตัวเลข ไม่ใช่ใช่/ไม่ใช่ (llama.cpp รันหลายตัวพร้อมกันได้)

แต่ละโมเดลมีปุ่ม **start/stop** และปุ่ม **⋯** ที่กางช่องตั้งค่า + คำสั่งที่เหลือ

**ตั้ง port / context / gpu-util ก่อนสั่งรันได้เลย** (ไม่ต้อง ssh ไปแก้ `.sh`):

| ช่อง | ช่วงที่รับ | หมายเหตุ |
|---|---|---|
| `port` | 1–65535 | ว่าง = ใช้ค่าใน bundle |
| `context` | 256–10,000,000 tokens | เกินที่หน่วยความจำรับไหว vLLM จะไม่ขึ้น |
| `gpu-util` | 0.3–0.98 | ขึ้นเฉพาะ engine vLLM · ช่วงเดียวกับที่ controller ตรวจเอง |

ค่าถูกจำไว้ต่อ (เครื่อง/โมเดล) **ในเบราว์เซอร์** ไม่ได้เขียนทับ bundle บนเครื่องปลายทาง —
ค่าที่ใช้ทดลอง (เลี่ยง port ชนกัน, ลด context ชั่วคราว) ไม่ควรกลายเป็นค่าถาวรของเครื่องนั้น
· **server ตรวจค่าเองทุกครั้ง** เพราะค่าพวกนี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นผ่าน SSH
จะฝากการตรวจไว้กับ JS ในเบราว์เซอร์ไม่ได้ · ส่ง option ไปกับคำสั่งที่ไม่รับมัน (เช่น `doctor`)
จะได้ 400 ไม่ใช่เงียบ ๆ ทิ้งจนผู้ใช้เข้าใจว่าตั้งค่าแล้ว

**เมนูของโมเดลบนเครื่องอื่นทำได้เท่ากับโมเดลในเครื่องนี้** — เป็น controller ตัวเดียวกันและ
รับ env ชุดเดียวกัน จึงไม่มีเหตุผลให้ต่างกัน:

| กลุ่ม | มีอะไร |
|---|---|
| **ตั้งค่าตอน start** | `port` · `context` · `slots` · `bind` · `API key` · `gpu-util` (เฉพาะ vLLM) |
| **ทดสอบ** | `test-text` · `test-vision` · `test-reasoning` · `test-tools` · `bench` · `stress` · `client-config` · `network-info` · `status` |
| **stacked** | `prepare-runtime` · `sync-worker` · `verify-worker` · `clear-fi-cache` |
| **จัดการ** | `restart` · `doctor` · `logs` · `repair` · `verify-files` · `enable`/`disable` · `remove` |

ค่าที่ตั้งถูกส่งเป็น **env ของ controller** (`API_PORT`, `CTX_SIZE`, `PARALLEL_SEQS`, `API_KEY`…)
ตัวเดียวกับที่โมเดลในเครื่องใช้ · ตรวจค่าที่ฝั่ง server ด้วยโค้ดชุดเดียวกันทั้งสองทาง
จะได้ไม่มีสองมาตรฐาน · ปุ่มชุดทดสอบขึ้นตามที่ bundle นั้นรองรับจริง

> ⚠️ **ทุก bundle ตั้งต้นที่พอร์ตเดียวกัน (8000)** — ถ้าโมเดลอื่นรันอยู่บนพอร์ตนั้น ชุดทดสอบ
> จะยิงไปโดนตัวนั้นแล้วรายงานว่า "ผ่าน" ทั้งที่ทดสอบคนละโมเดล · bundle ที่สร้างตั้งแต่
> 2026-08-06 มี `assert_our_server()` กันไว้ในสคริปต์แล้ว (อ่าน `/v1/models` ก่อนยิง
> ถ้าชื่อไม่ตรงจะหยุดพร้อมบอกว่าใครยึดพอร์ตอยู่) · **bundle เก่าไม่มี** หน้าเว็บจึงติดป้ายแดง
> `พอร์ตชนกับ <ชื่อ>` ให้แทน — ตั้ง `port` ให้ต่างกันก่อน start

คำสั่งในเมนู (สรุป):
`restart` · `doctor` · `logs` (300 บรรทัดล่าสุด) · `repair` · `enable`/`disable` (autostart) · `remove`
— **ปุ่มขึ้นตามสถานะจริง**: ยังไม่ได้รัน = ไม่มี `restart` · เครื่องที่ไม่มี systemd (`n/a`) =
ไม่มีปุ่ม autostart เลย เพราะกดแล้วล้มแน่ ๆ · ป้าย `autostart` ขึ้นเฉพาะตอนสถานะเป็น `enabled` จริง
(ไม่ใช่ `absent`/`disabled` ซึ่งเคยถูกนับเป็น "เปิดอยู่" ทั้งหมด)

**`enable` ต้อง sudo บนเครื่องปลายทาง** — hub เรียกผ่าน SSH ซึ่งไม่มี tty ให้กรอกรหัสผ่าน
ถ้าเครื่องนั้นไม่ได้ตั้ง sudo แบบไม่ถามรหัส คำสั่งจะล้มและ**พิมพ์คำสั่งที่ต้องรันเองบนเครื่องนั้น**
ให้ในหน้าเว็บ — ไม่ใช่รายงานว่าสำเร็จ

**`remove` ทำได้จากหน้าเว็บ แต่ต้องผ่านสองขั้น** — กดครั้งแรกคือ `--dry-run`: เห็นรายการไฟล์
ที่จะถูกลบพร้อมขนาดจริง (bundle, ทะเบียน, weight) แล้วถึงมีปุ่ม **ยืนยันลบถาวร** · ฝั่ง server
ต้องได้ค่ายืนยันที่**ตรงกับชื่อโมเดลเป๊ะ** ถึงจะลบ (ส่ง `yes` หรือ `true` มาไม่ผ่าน) ปุ่มที่กดพลาด
จึงลบ weight หลายสิบ GB ไม่ได้ · allowlist บังคับที่ฝั่ง server ไม่ใช่แค่ซ่อนปุ่ม

> **`lmds remove` ผ่าน SSH ต้องมี `-y`** — ไม่มี terminal ให้ตอบยืนยัน · สั่งจาก hub ให้ดู
> รายการก่อนด้วย `lmds node run <เครื่อง> remove <โมเดล> --dry-run` แล้วค่อยเติม `-y`
> · เดิมมันตอบแค่ `Aborted.` ซึ่งอ่านเหมือนคำสั่งทำงานแล้วไม่มีอะไรเกิดขึ้น

### เครื่องไหน stacked ด้วยกันได้ (ConnectX / 200G)

```bash
lmds node cluster
```

ตรวจจาก `/sys` ให้เอง: ความเร็วลิงก์ · การ์ด ConnectX (vendor `0x15b3`) · อุปกรณ์ RDMA · IP ของแต่ละ interface
แล้วจับกลุ่มเฉพาะเครื่องที่ **arch/profile/รุ่น GPU/จำนวน GPU ตรงกัน และมีสาย ≥ 25G ทั้งคู่**

สิ่งที่ระบบเดาให้ไม่ได้คือ **NCCL จะคุยกันทาง IP ไหน** (เครื่องหนึ่งมักมีทั้งเส้น SSH และเส้น fabric) —
ระบบเสนอ IP ที่เจอบนสายเร็วสุด แต่ต้องยืนยันเอง:

```bash
lmds node set spark2 --cluster-ip 10.10.0.2     # ตั้งค่า
lmds node set spark2                            # ดูค่าปัจจุบัน + ค่าที่ตรวจพบ
lmds node cluster --write my-70b-model          # เขียน cluster.env ลง bundle
```

`cluster.env` มี `MASTER_IP` / `WORKER_IP` / `SSH_USER` / `TRANSPORT_IP_*` / `NCCL_SOCKET_IFNAME` —
stacked controller จะ source ไฟล์นี้**ก่อน default ทั้งหมด** แล้วไม่ถาม IP ตอน `start` อีก
(ตั้ง env จากภายนอกยังชนะไฟล์นี้เสมอ · ไม่มีไฟล์ = ถามแบบเดิม)

> ถ้าไม่ตั้ง `NCCL_SOCKET_IFNAME` ไว้ NCCL จะเลือก interface เอง และมักได้เส้นบริหารจัดการที่ช้ากว่า —
> ยังรันได้แต่ช้าลงแบบหาสาเหตุยาก

รายละเอียดทั้งหมด: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md)

## 4.5.1 ไม่มี API key ของ LLM ก็ deploy ได้ — `lmds recipes`

`lmds deploy --no-llm` ใช้ได้อยู่แล้ว แต่ rule-based รู้แค่ "GGUF → llama.cpp, safetensors → vLLM"
ไม่รู้เรื่องเฉพาะรุ่น เช่น DeepSeek V4 บังคับ `kv-cache fp8` หรือ Qwen3-Coder NVFP4 ต้องใช้ image
ที่มี kernel ตรงรุ่น — ผลคือ **deploy ผ่าน แต่ start ไม่ขึ้น**

LMDS จึงเก็บ **สูตรที่รันผ่านจริงบนฮาร์ดแวร์แล้ว** ไว้ในตัวโปรแกรม และใช้อัตโนมัติ:

```bash
lmds recipes                                    # มีสูตรอะไรบ้าง
lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4     # สูตรตัวนั้น + ที่มา + เคยรันบนอะไร
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 --target dgx-spark-stacked --no-llm --yes
```

ตอน deploy จะขึ้นบรรทัดบอกว่าใช้สูตรไหน:

```text
ใช้สูตรที่รันผ่านจริง: DeepSeek-V4-Flash (NVFP4) — https://github.com/neronain/deepseek-…
```

- สูตร **ไม่แตะ context** — ค่านั้นยังมาจากการวิเคราะห์หน่วยความจำของเครื่องคุณเอง
- image ที่ทดสอบบน DGX Spark จะไม่ถูกนำไปใช้กับ RTX เงียบ ๆ (จะเตือนแล้วใช้ค่าตั้งต้น)
- โมเดลที่ยังไม่มีสูตร ทำงานเหมือนเดิมทุกอย่าง

## 4.5.2 เครื่องจัดการโชว์โมเดลที่ไม่ใช่ของตัวเอง — `lmds prune`

เครื่องที่ใช้ **สร้าง bundle ให้เครื่องอื่น** จะสะสมทะเบียนของ bundle ที่ย้าย/ลบไปแล้ว
ทำให้เห็นรายการที่กดอะไรก็ไม่ได้ และ**เสี่ยงสั่งการผิดเครื่อง**

```bash
lmds prune          # แสดงรายการที่ตายแล้วให้ดูก่อน แล้วถามยืนยัน
```

**ลบเฉพาะไฟล์ทะเบียน** ไม่แตะ weight, bundle หรือ container · ตัวที่ไม่เคยถูก start เลย
ระบบเก็บกวาดให้อัตโนมัติอยู่แล้ว

## 4.6 เครื่องที่มีโมเดลอยู่ก่อนแล้ว — `lmds scan`

เครื่องลูกค้าที่ใช้งานมาก่อนติดตั้ง LMDS มักมี weight กระจายอยู่หลายที่และไม่ได้จัดระเบียบแบบเรา
— HF cache มีสองเลย์เอาต์, บางคนตั้ง `HF_HUB_CACHE` ไปดิสก์อื่น, ไฟล์ GGUF วางเป็นโฟลเดอร์ธรรมดา

```bash
lmds scan                    # เครื่องนี้
lmds scan --all              # ทุกเครื่องในทะเบียน
lmds scan --root /mnt/nvme   # เพิ่มที่ค้นเอง
```

ตัวอย่างจากเครื่องจริง:

```text
hf    nvidia/DeepSeek-V4-Flash-NVFP4    157.0 GB   46 shards   ~/.cache/huggingface/models--…
                                                               (เลย์เอาต์เก่า — ต้องตั้ง HF_HUB_CACHE)
hf    meta-llama/Llama-3.3-70B-Instruct 263.0 GB   30 shards   ~/.cache/huggingface/hub/models--…
gguf  gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf 25.7 GB               ~/models/gemma4-26b-a4b-q8xl/…
```

- **อ่านอย่างเดียว** — ไม่ย้าย ไม่ลบ ไม่แก้อะไรเลย
- ใช้ตอบว่า "ต้องโหลดใหม่ไหม" ก่อนจะเสียเวลาโหลดซ้ำหลายสิบ GB
- โมเดลที่อยู่**เลย์เอาต์เก่า** (`$HF_HOME/models--X` ไม่มี `hub/`) ยังใช้ได้ปกติ —
  stacked controller ตั้ง `HF_HUB_CACHE` ให้ตรงเองตอน start **ไม่ต้องย้ายไฟล์**

> ถ้าไม่ตั้งให้ตรง vLLM ในคอนเทนเนอร์จะฟ้อง `LocalEntryNotFoundError` ทั้งที่ `verify-files`
> เพิ่งบอกว่าไฟล์ครบ — เป็นอาการที่ไล่สาเหตุยากมากถ้าไม่รู้ว่ามีสองเลย์เอาต์

## 5. หน้าเว็บ (ทางเลือก) — `lmds web`

สำหรับคนที่ไม่ถนัด CLI หรืออยากให้ทีมดูสถานะได้โดยไม่ต้อง ssh · **หน้าเว็บเป็นภาษาอังกฤษ**
(ตัว CLI ยังเป็นไทย)

```bash
lmds web                          # http://127.0.0.1:8600 — เครื่องนี้เท่านั้น
lmds web --bind 0.0.0.0           # ให้ทั้งวง network เข้าได้ — ถาม token ก่อน (Enter = สุ่มให้)
lmds web --background             # รันเบื้องหลัง terminal ว่างใช้ CLI ต่อได้
lmds web --status                 # ลืมลิงก์/token? ถามตัวที่รันอยู่ได้เลย
lmds web --restart -b             # เปิดใหม่ (ลิงก์เดิมยังใช้ได้)
lmds web --stop                   # หยุดตัวที่รันเบื้องหลัง
lmds web -b --new-token           # เปลี่ยน token (ลิงก์เดิมใช้ไม่ได้ทันที)
```

### ส่ง bundle ไปรันบนเครื่องอื่น

เครื่องที่คุณเปิดหน้าเว็บอาจไม่ใช่เครื่องที่จะรันโมเดล (เช่น controller ที่ไม่มี GPU) —
สร้าง bundle ที่นี่ ตรวจแผนที่นี่ แล้ว**ส่งตัวเดียวกันนั้นไปติดตั้ง**บนเครื่องเป้าหมาย

- **หน้าเว็บ**: กด `manage` ของโมเดล → **Run on another machine** → เลือกเครื่อง → **Send bundle there**
- **CLI**: `lmds node push <เครื่อง> <slug> [--download] [--start]`

```bash
lmds node push dgx-veerasiam gemma-4-12b-it-gguf --download --start
```

> **ทำไมส่ง ZIP แทนสั่งให้ปลายทาง `lmds deploy` เอง** — คุณตรวจแผนและอนุมัติ flag ไปแล้ว
> กับ bundle ตัวนี้ · ให้ปลายทางวางแผนใหม่เองอาจได้คนละค่า (ฮาร์ดแวร์คนละตัว, recipe อัปเดต)
> กลายเป็นอนุมัติแผนหนึ่งแล้วได้อีกแผนหนึ่งไปรัน

ถึงแล้วสั่ง download / start ต่อได้จากการ์ดของเครื่องนั้นเลย

### เข้าใช้งานด้วย token

เปิดลิงก์แล้วจะเจอ**หน้า login ให้กรอก token** — กรอกครั้งเดียว เบราว์เซอร์จำให้
(มีปุ่ม **Sign out** ที่แถบบนถ้าอยากลืม) · **ลิงก์ไม่มี token ติดไปด้วย** เพราะ URL
ไปโผล่ใน history ของเบราว์เซอร์, log ของ proxy และ referrer — และคนที่ยืนดูจอก็อ่านได้

**token มาจากไหน** — ตัวบนสุดที่มีค่าชนะ:

| ลำดับ | ที่มา | ใช้เมื่อ |
|---|---|---|
| 1 | `--token <ค่า>` | บังคับเฉพาะครั้งนั้น |
| 2 | `$LMDS_WEB_TOKEN` | เครื่องที่รันด้วย systemd/compose ซึ่งไม่มีใครนั่งตอบคำถาม |
| 3 | ที่จำไว้ที่ `~/.config/lmds/web-token` (0600) | ปกติ — ลิงก์เดิมใช้ได้ตลอด |
| 4 | **ถามตอนสตาร์ตครั้งแรก** | Enter เฉย ๆ = สุ่มให้ · กรอกเองก็ได้ |
| 5 | สุ่มให้ | ไม่มี terminal ให้ถาม |

**ตั้งเอง**: อย่างน้อย **8 ตัว** ไม่จำกัดชนิดตัวอักษร (passphrase ภาษาไทยก็ได้)
ห้ามมีช่องว่างหรือตัวควบคุมเท่านั้น เพราะ copy ไป paste แล้วเพี้ยนโดยไม่มีใครรู้ตัว

```bash
lmds web -b --bind 0.0.0.0 --token 'รหัสของทีมเรา2569'   # ตั้งเอง
export LMDS_WEB_TOKEN='...'                              # หรือใส่ใน ~/.bashrc / systemd unit
lmds web --status                                        # ลืม token? ถามตัวที่รันอยู่
lmds web -b --new-token                                  # เปลี่ยนใหม่ (เครื่องที่ login ค้างไว้หลุดหมด)
```

**กันเดา token** — ผิดติดกันเกิน 5 ครั้งจาก IP เดียวกันจะเริ่มหน่วงแบบทวีคูณ (สูงสุด 60 วินาที)
คนพิมพ์ผิดจริง ๆ ไม่โดนลงโทษ แต่บอตที่ยิงรัวไม่คุ้ม · ล็อกอินผ่านครั้งเดียวล้างตัวนับ

> `?token=...` ใน URL ยังใช้ได้อยู่ (สคริปต์/curl ต้องใช้) · ถ้าเปิดหน้าเว็บด้วยลิงก์แบบนั้น
> token จะถูกย้ายเข้าที่เก็บของเบราว์เซอร์แล้ว**ลบออกจากแถบที่อยู่ทันที**

**ลิงก์อยู่ยาว — bookmark ได้** token ใช้ซ้ำทุกครั้งที่เปิด · restart/stop แล้วเปิดใหม่ก็ยังเป็น
ตัวเดิม เปลี่ยนเมื่อสั่ง `--new-token` หรือตั้งเองเท่านั้น

**เปิดลิงก์แล้วขึ้น "Token ไม่ตรงกับหน้าเว็บที่รันอยู่"** = ลิงก์นั้นมาจากรอบก่อน ไม่ใช่ copy ผิด
· `lmds web --status` จะบอกลิงก์ของตัวที่เสิร์ฟอยู่จริง · สั่ง `lmds web -b` ตอนที่มีตัวรันอยู่แล้ว
จะไม่สตาร์ตซ้อน แต่พิมพ์ลิงก์ของตัวเดิมให้แทน

หน้าเว็บพิมพ์ IP จริงของเครื่องให้ (ไม่ใช่ `0.0.0.0` ซึ่งเปิดในเบราว์เซอร์ไม่ได้) · เครื่องที่มีหลายวง
เช่น Tailscale/VPN ใช้ IP ของวงนั้นแทนได้ พอร์ตกับ token เดียวกัน

### ทำอะไรได้บ้าง

| แถวโมเดล | รายละเอียด |
|---|---|
| **download** | โหลด weight แล้ว **รัน `verify-files` ต่อให้อัตโนมัติ** พร้อม log สด — ปุ่มเปลี่ยนเป็น `start` เองเมื่อครบ |
| **start / stop / restart** | ใช้ตัวเลือกที่ตั้งไว้ในแท็บ manage |
| **tests** | `test-text` · `test-vision` · `test-reasoning` · `test-tools` · `bench` · `stress` · `client-config` · `network-info` · `status` |
| **manage** | port / context / slots / bind / API key · autostart · คำสั่ง stacked · repair · remove |
| **doctor** | ผลเดียวกับ `lmds doctor` พร้อมคำสั่งแก้ |
| **logs** | log ล่าสุด 300 บรรทัด |

**เครื่องอื่นในทะเบียน** (หัวข้อ *Other machines*) — แต่ละเครื่องมีการ์ดของตัวเอง กด **−/+** ย่อ-ขยายได้
· ในการ์ดมีเกจทรัพยากร, การ์ด GPU, แถบ cluster (ดู §4.5) และรายชื่อโมเดลพร้อมปุ่ม **⋯** ต่อโมเดล

> **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่านจาก dispatch table ของสคริปต์เอง
> bundle ที่สร้างก่อนมีคำสั่งใหม่ (เช่น `test-vision`) จะไม่มีปุ่มนั้น พร้อมบอกว่าต้อง deploy ใหม่

ปุ่ม **+ Deploy model** ทำ wizard ครบ flow: วางลิงก์ → เลือก target → วิเคราะห์ → เลือกไฟล์ GGUF /
ใส่ HF token ถ้าจำเป็น → ดูแผน + ปรับ context / อนุมัติ flag → สร้าง bundle ผ่าน quality gates → ZIP

### หน้าตาและการอ่านค่า

หน้านี้ออกแบบให้ **อ่านสถานะเครื่องได้ก่อนอ่านตัวหนังสือ**:

| องค์ประกอบ | อ่านยังไง |
|---|---|
| **การ์ดไล่เฉดสองใบบนสุด** | GPU และ *กี่โมเดลรันอยู่* — สองอย่างที่คนเปิดหน้านี้มาดูก่อนเสมอ |
| **เกจ CPU / Unified·RAM / VRAM / Disk free** | ชุดเดียวกันทั้งเครื่องนี้และเครื่องอื่น · ตัวเลขจริงอยู่ใต้ทุกเกจ |
| **สีของเกจ** | หน่วยความจำ ≥75% เหลือง ≥90% แดง · ดิสก์กลับด้าน (เหลือ ≤15% เหลือง ≤5% แดง) |
| **การ์ด GPU** | compute / power / temp / fan + สัญญาณนาฬิกา + PCIe — ค่าที่การ์ดไม่รายงานจะถูกซ่อน ไม่ใช่โชว์ 0 |
| **รั้วสี + ป้าย CLUSTER A/B** | เครื่องที่ stacked ด้วยกันได้ |

**ทำไมมีสีเยอะ แต่ยังอ่านง่าย** — ทั้งหน้าใช้ตระกูลสีเดียว (ฟ้า → คราม → ม่วง → ชมพู) ไล่เฉด
ไม่ใช่สีทึบคนละที่คนละสี · สีที่**มีความหมาย** (เขียว/เหลือง/แดง = ปกติ/ใกล้เต็ม/เต็ม) ยังแยกจาก
สีตกแต่งชัดเจน · พื้นหลังมีแสงจาง ๆ แต่จงใจให้เบามาก เพราะ **ช่องกรอกต้องเด้งออกจากพื้นเสมอ**
ไม่งั้นคนกรอกผิดช่องได้ — ช่องกรอกจึงใช้ผิวจมลงไป + ขอบเข้มกว่าการ์ด และเปลี่ยนสีขอบตอนโฟกัส

หน้าเว็บ **ปรับตามธีมของเครื่อง** (สว่าง/มืด) และเคารพ `prefers-reduced-motion`

### ความปลอดภัย

หน้านี้**สั่ง start/stop/ลบโมเดลได้** จึง:

- bind `127.0.0.1` เป็นค่าเริ่มต้น — ต้องตั้งใจเปิดออก network เอง
- `--bind 0.0.0.0` โดยไม่ตั้ง `--token` → **ถามก่อน** แล้วสุ่มให้ถ้าไม่กรอก (ดู §5 *เข้าใช้งานด้วย token*)
- **ต้องผ่านหน้า login ก่อนถึงจะวาดอะไร** — เดิมโหลดโครงหน้าขึ้นมาก่อนแล้วค่อยพังตอนเรียก API
  คนที่ไม่มีสิทธิ์จึงเห็นชื่อเครื่อง
- **ลิงก์ที่พิมพ์ออกมาไม่มี token ติดไปด้วย** — URL ไปโผล่ใน history, log ของ proxy และ referrer
- **กันเดา token** — ผิดติดกันเกิน 5 ครั้งจาก IP เดียวกันเริ่มหน่วงแบบทวีคูณ (สูงสุด 60 วินาที)
- API key ของโมเดลเก็บใน localStorage ของเบราว์เซอร์ ไม่ขึ้นไปอยู่บนเครื่อง
- หน้าเว็บไม่ดึงอะไรจากอินเทอร์เน็ตเลย ใช้บนเครื่องหลัง proxy / air-gapped ได้

### ยังต้องใช้ CLI สำหรับ

`lmds config` (ตั้ง provider / API key) · `lmds hardware` · และ `enable`/`disable` autostart ในเครื่อง
ที่ `sudo` ต้องกรอกรหัส (หน้าเว็บไม่มี tty — จะบอกคำสั่งให้ไปรันเอง)

## 6. คำสั่งอื่นที่ควรรู้

```bash
lmds plan Qwen/Qwen3-32B --target dgx-spark-single   # ดูแผนอย่างเดียว ไม่สร้างไฟล์
lmds plan Qwen/Qwen3-32B --json                      # แผนเป็น JSON (สำหรับ script)
lmds inspect Qwen/Qwen3-32B --json                   # ผลวิเคราะห์เป็น JSON
lmds generate ...                                    # เหมือน deploy แต่ไม่มีขั้นยืนยัน
lmds validate bundles/qwen3-32b                      # ตรวจ bundle ย้อนหลัง (เช็คว่าไม่มีใครแก้ไฟล์)
lmds validate bundles/qwen3-32b --fix                # regenerate checksum หลังตั้งใจแก้ไฟล์เอง
lmds hardware                                        # ตรวจเครื่อง (GPU/RAM/ดิสก์/Docker/profile)
lmds scan                                            # โมเดลที่มีอยู่แล้วบนเครื่อง (ทุกที่เก็บ)
lmds scan --all                                      # ค้นทุกเครื่องในทะเบียนด้วย
lmds recipes                                         # สูตรที่รันผ่านจริง (ใช้เองเมื่อไม่มี LLM)
lmds prune                                           # ล้างทะเบียนค้างของ bundle ที่ลบไปแล้ว
lmds config show                                     # ดู config (key ถูก mask)
lmds config defaults                                 # ดู default model ของแต่ละ provider
lmds repair <ชื่อ>                                    # ซ่อมไฟล์ที่ขาด (ดู §4.3)
lmds remove <ชื่อ> --keep-weights                     # ลบ bundle แต่เก็บ weight (ดู §4.3)
lmds --install-completion                            # เปิด tab completion (ดู §4.4)
lmds node list / lmds ps --all                       # เครื่องอื่นในทะเบียน (ดู §4.5)
lmds agent info                                      # JSON สถานะเครื่องนี้ (hub เรียกผ่าน SSH)
```

ตั้ง key แบบไม่ต้องพิมพ์มือ (สำหรับ script/automation):

```bash
echo "$MY_KEY" | lmds config set-key openai --stdin
echo "$HF_TOKEN" | lmds config set-hf-token --stdin
```

**exit code ที่ใช้เช็คใน script**: `0` สำเร็จ · `1` input ผิด/ยกเลิก · `2` ไม่ผ่าน quality gates · `3` โมเดลไม่ fit · `4` ต้องการ token/สิทธิ์ · `5` ปัญหา provider

`lmds` พิมพ์ banner ออก stderr และเงียบเองเมื่อถูก pipe — ปิดถาวรด้วย `export LMDS_NO_BANNER=1`

## 7. เอา bundle ไปใช้เครื่องอื่น

Bundle เป็นไฟล์ธรรมดา ไม่ผูกกับเครื่องที่สร้าง:

```bash
scp bundles/qwen3-32b.zip user@server:/home/user/
# บนเครื่องปลายทาง (ต้องมี Docker + NVIDIA toolkit):
unzip qwen3-32b.zip && cd qwen3-32b
./qwen3-32b-single.sh download && ./qwen3-32b-single.sh start
```

> สร้าง bundle จากเครื่องไหนก็ได้ (ไม่ต้องมี GPU) โดยระบุ `--target` ของเครื่องปลายทาง

---

## 8. แก้ปัญหาที่พบบ่อย (Troubleshooting)

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `lmds: command not found` | PATH ไม่มี `~/.local/bin` | `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `ต้องการ Python >= 3.10` | Python เก่า | `sudo apt install python3.11 python3.11-venv` แล้วรัน `python3.11 -m venv ...` หรืออัปเกรด OS |
| deploy แจ้ง `ต้องการ token` (exit 4) | โมเดล gated | `lmds config set-hf-token` + กดยอมรับเงื่อนไขบนเว็บ HF |
| deploy แจ้ง `ไม่ fit` (exit 3) | โมเดลใหญ่เกินเครื่อง | ทำตามคำแนะนำที่ระบบพิมพ์ (เลือก quant เล็กกว่า / เปลี่ยน target) |
| `start` แล้ว `GPU ใน container ใช้ไม่ได้` | ไม่มี NVIDIA Container Toolkit | ทำ INSTALL.md ส่วน 1.4–1.5 |
| `start` แล้ว `container ... มีอยู่แล้ว` | รันค้างจากรอบก่อน | `./xxx-single.sh stop` แล้ว start ใหม่ |
| `start` ค้างที่ "รอ /health" นาน | โมเดลใหญ่กำลังโหลดเข้า GPU | ปกติสำหรับโมเดล >30GB (รอได้ถึง 15 นาที) — ดูความคืบหน้า: `./xxx-single.sh logs 100` |
| health timeout / container ดับ | memory ไม่พอจริง | ลด context: `./xxx-single.sh start --context 16384` — ถ้ายังไม่รอด เก็บ `logs 500` ส่งให้ทีมพัฒนา |
| `ขนาดไฟล์ไม่ตรง` ตอน verify-files | download ไม่ครบ | รัน `download` ซ้ำ (resume ต่อจากเดิมอัตโนมัติ) |
| port ชน | มี service อื่นใช้ :8000 | `--port 8001` หรือหยุด service เดิม (`docker ps` ดูว่าตัวไหน) |
| `permission denied ... docker.sock` | user ไม่อยู่ใน group docker | INSTALL.md ส่วน 1.3 + logout/login |
| `HTTP 429 ... quota` จาก provider | โควตา LLM หมด | ระบบสลับ rule-based ให้อัตโนมัติ — งานเดินต่อได้; ระยะยาว: เติมโควตา หรือสลับไปใช้ Local AI (`set-provider openai-compat`) |
| อยากใช้ Ollama/vLLM local เป็นสมอง | — | `lmds config set-provider openai-compat --base-url http://<ip>:11434/v1 --model gpt-oss:20b` (Ollama) หรือ `--base-url http://<ip>:8000/v1` (vLLM) — ไม่มี key ก็ใช้ได้ · ตั้งไม่ติดดู [INSTALL §3.2.1](INSTALL.md) |
| `download` พังกลางคัน / `No space left on device` | ดิสก์เต็ม | `df -h ~` · ย้ายที่เก็บ: `HF_HOME=/data/hf-cache` (vLLM) หรือ `MODEL_DIR=/data/models` (GGUF) แล้ว download ใหม่ (resume ต่อได้) |
| `start` ครั้งแรกค้างนานผิดปกติ ยังไม่ขึ้น log อะไร | Docker กำลัง pull image (~10–20 GB) | ปกติ — ดูความคืบหน้าด้วย `docker pull vllm/vllm-openai:latest` แยกอีก terminal · ดึงล่วงหน้าได้ตาม [INSTALL §1.7](INSTALL.md) |
| `docker pull` ล้ม / `TLS handshake timeout` | เครื่องอยู่หลัง proxy หรือโดน rate limit | ตั้ง proxy ให้ **docker daemon** ด้วย ไม่ใช่แค่ shell ([INSTALL §1.7](INSTALL.md)) |
| `prepare-runtime` build ล้มบน DGX Spark | ขาด CUDA Toolkit หรือ CUDA arch ไม่ตรง | ดูบรรทัดเตือน `ไม่พบ nvcc` · override ได้: `CUDA_ARCHITECTURES=121 ./xxx-single.sh prepare-runtime` |
| `ยังไม่มี llama-server — รัน: ... prepare-runtime` | ข้ามขั้น prepare-runtime บนเครื่อง ARM64 | รัน `./xxx-single.sh prepare-runtime` ก่อน start (ดู §2) |
| ลิงก์ `ollama.com/...` ใช้ไม่ได้ | ยังรองรับเฉพาะ Hugging Face | ใช้ลิงก์ HF ของ GGUF ตัวเดียวกันแทน (roadmap เฟส 2) |
| `verify-files` แจ้ง shard หาย / ขนาดไม่ตรง | download ไม่ครบ หรือไฟล์ใน cache ถูกลบ | `lmds repair <ชื่อ>` (โหลดเฉพาะส่วนที่ขาด) |
| `lmds list` ขึ้น ⚠ (ไฟล์ controller หาย) | โฟลเดอร์ bundle ถูกลบ/ย้าย | `lmds deploy` ลิงก์เดิมเพื่อสร้าง bundle ใหม่ — weight เดิมใช้ต่อได้ · หรือ `lmds remove <ชื่อ>` ถ้าไม่ใช้แล้ว |
| มีแถวขยะค้างใน `lmds ps` / `lmds list` | process/ทะเบียนเก่าค้างจากรอบก่อน | `lmds remove <ชื่อ>` เก็บกวาดให้ครบทุกที่ |

### ถ้าแก้เองไม่ได้ — ข้อมูลที่ต้องเก็บส่งทีมพัฒนา

```bash
lmds version
lmds hardware
./xxx-single.sh logs 500 > failure.log
# + คำสั่งเต็มที่รันแล้วพัง + ข้อความ error ทั้งหมด
```

## 9. ความปลอดภัย — ข้อควรปฏิบัติ

> ภาพรวมเต็ม (ข้อมูลอะไรออกนอกเครื่อง, secret เก็บที่ไหน, prompt injection): **[SECURITY.md](../SECURITY.md)**

> ⚠️ **ค่า default คือเปิดออกทั้งวง LAN** — controller bind ที่ `0.0.0.0` และ**ไม่มี** API key
> ใครก็ตามที่เข้าถึงเครือข่ายเดียวกันยิง `http://<ip>:8000/v1` ได้ทันทีโดยไม่ต้องยืนยันตัวตน
> เครื่องที่ไม่ได้อยู่ในวงปิดจริง ๆ ให้ทำอย่างน้อยหนึ่งอย่างเสมอ:
>
> ```bash
> ./xxx-single.sh start --bind 127.0.0.1        # ใช้เฉพาะในเครื่อง (ปลอดภัยสุด)
> API_KEY=$(openssl rand -hex 24) ./xxx-single.sh start   # หรือบังคับ Bearer token
> ```

- วิธีตั้ง API token ของ endpoint อย่างละเอียด (พร้อมตัวอย่าง curl) อยู่ในหัวข้อ **API TOKEN** ของ `./xxx-single.sh help`
- API key / HF token ใส่ผ่าน `lmds config set-key` หรือ env เท่านั้น — **ห้าม**เขียนลงไฟล์/สคริปต์เอง
- เซิร์ฟเวอร์ที่เปิดใน network ที่มีคนอื่นใช้ร่วม ให้ตั้ง `API_KEY=xxx ./xxx-single.sh start` เสมอ
- `API_KEY` ถูกส่งเข้า container ผ่าน env — ผู้ที่ใช้ `docker` บนเครื่องเดียวกันอ่านได้ด้วย `docker inspect` (ไม่ใช่ช่องโหว่ต่อคนนอก แต่ไม่ควรใช้ key เดียวกับระบบอื่น)
- **ข้อมูลที่ออกจากเครื่อง**: ตอนวางแผน ระบบส่ง metadata ของโมเดล (model card, `config.json`, รายชื่อไฟล์) ไปยัง LLM provider ที่ตั้งไว้ — ไม่ส่ง weight, ไม่ส่ง key, ไม่ส่งข้อมูลผู้ใช้ · องค์กรที่ห้ามข้อมูลออก ให้ใช้ `--no-llm` หรือตั้ง provider เป็น Local AI ([INSTALL §3.2.1](INSTALL.md))
- สำเนา prompt/คำตอบของทุกครั้งที่เรียก LLM ถูกเก็บไว้ที่ `~/.config/lmds/sessions/` (redact secret แล้ว) — ลบได้ถ้าไม่ต้องการเก็บประวัติ
- flag `--trust-remote-code` อนุมัติเฉพาะหลัง review ไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน SPECIAL_FILES.md)
- bundle ที่รับมาจากคนอื่น ตรวจก่อนใช้: `lmds validate <โฟลเดอร์>`
