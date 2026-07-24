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
| `lmds ps` / `stop` / `logs` / `start` / `list` | จัดการโมเดลที่ deploy/รันอยู่ (ดู §4) |
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

| คำสั่ง | หน้าที่ |
|---|---|
| `download` | ดาวน์โหลดโมเดล (pin revision, resume ได้) |
| `verify-files` | ตรวจความครบถ้วน/ความถูกต้องของไฟล์ |
| `start` / `stop` / `restart` | เปิด-ปิดเซิร์ฟเวอร์ (ตรวจ GPU + ไฟล์ก่อน start เสมอ) |
| `status` | สถานะ container + API health |
| `logs [N]` | log ล่าสุด N บรรทัด (default 300) |
| `client-config` | ค่าตั้ง client เป็น JSON พร้อม token budget |
| `network-info` | bind address + endpoint ที่ประกาศให้ client |
| `test-text` | ทดสอบ chat completion หนึ่งครั้ง |

### options ที่ทุก controller รองรับ (ใส่ท้ายคำสั่งใดก็ได้)

```bash
./xxx-single.sh start --port 8001                  # เปลี่ยน port
./xxx-single.sh start --context 16384              # ลด context (ประหยัด memory)
./xxx-single.sh start --advertise-ip 10.0.0.5      # IP ที่ประกาศให้ client (ไม่ใช่ bind)
./xxx-single.sh client-config --client-output 4096 # ปรับ token budget
```

หรือใช้ env: `API_PORT=8001 ./xxx-single.sh start`

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

ข้อกำหนด: 2× DGX Spark + fabric ระหว่าง node (แนะนำ 200 Gb/s RoCE) + passwordless SSH (master→worker) · `lmds ps`/`lmds stop`/`lmds logs` เห็น/สั่งงานตัวนี้ได้เหมือน deploy เดี่ยว (stop จะหยุดทั้งสอง node ให้) · stacked รองรับเฉพาะ vLLM (GGUF ยังไม่มี reference ที่ทดสอบแล้ว)

### 3.4 Target presets ที่มีให้เลือก

| preset | เครื่อง | หน่วยความจำ |
|---|---|---|
| `dgx-spark-single` | DGX Spark 1 เครื่อง | unified 128GB |
| `dgx-spark-stacked` | DGX Spark 2 เครื่อง | unified 256GB (ประมาณ) — สร้าง controller multi-node (ดู 3.3b) |
| `rtx-pro-4000` / `rtx-pro-4000-dual` | RTX PRO 4000 Blackwell ×1/×2 | 24GB / 48GB |
| `rtx-4070-super` / `rtx-4070-ti-super` | RTX 4070 Super / Ti Super | 12GB / 16GB |
| `rtx-4090` / `rtx-5090` | (ยังไม่ tested — โหมด conservative) | 24GB / 32GB |
| *(ไม่ระบุ)* | ใช้เครื่องที่รันคำสั่งอยู่ | ตรวจอัตโนมัติ |

### 3.5 โหมดและ options ของ `lmds deploy`

```bash
--no-llm            # ไม่เรียก LLM (rule-based) — ฟรี, เร็ว, แต่ไม่วิเคราะห์ parser/feature เชิงลึก
--yes / -y          # ข้ามขั้นยืนยันทั้งหมด (สำหรับ script/CI) — flag ค้างอนุมัติจะไม่ถูกใส่
--output DIR        # เปลี่ยนที่เก็บ bundle (default: ./bundles)
--revision SHA      # ล็อค revision เอง (default: ล่าสุด ณ ตอนนั้น แล้ว pin ให้)
--concurrency N     # จำนวน request พร้อมกันที่ใช้คำนวณ KV cache (default 1)
```

### 3.6 ขั้นยืนยัน — จุดที่ต้องอ่านก่อนกด

1. **อนุมัติ flag นอก allowlist** — ถ้าแผนเสนอ flag พิเศษ (เช่น `--trust-remote-code`) ระบบถามทีละตัว
   ค่า default คือ**ไม่อนุมัติ** — อนุมัติเฉพาะเมื่อเข้าใจผลของ flag นั้น (อ่าน SPECIAL_FILES.md ประกอบ)
2. **context** — Enter ใช้ค่าที่คำนวณให้ หรือพิมพ์เลขใหม่ (เกินเพดานปลอดภัยระบบจะลดให้อัตโนมัติ)
3. **ยืนยันสร้าง bundle** — Y/n

---

## 4. จัดการหลายโมเดลในเครื่องเดียว (Fleet)

รันหลายโมเดลพร้อมกันได้ (คนละ port) — ไม่ต้องจำว่า bundle ไหนอยู่ที่ไหน ใช้ `lmds` เป็นศูนย์กลาง:

```bash
lmds ps               # ใครรันอยู่บ้าง: ชื่อ, โมเดล, port, สถานะ (● running / ◐ loading / ○ stopped)
lmds stop qwen3-coder-30b-a3b-instruct-gguf    # หยุดตามชื่อ — ไม่ต้อง cd ไปหา .sh
lmds stop --all       # หยุดทุกตัวที่รันอยู่
lmds logs <ชื่อ> -n 500   # ดู log ตามชื่อ
lmds start <ชื่อ>          # รันโมเดลที่เคย deploy ไว้ขึ้นมาใหม่ (เช่น หลัง reboot)
lmds enable <ชื่อ>         # ตั้งให้กลับมาเองหลัง reboot (systemd) · lmds disable <ชื่อ> = ยกเลิก
lmds list             # bundle ทั้งหมด + engine/port/context/ฟีเจอร์ที่รองรับ (tools/reasoning/vision) + autostart
```

ตัวอย่างรัน 2 โมเดลพร้อมกัน:

```bash
cd bundles/model-a && ./model-a-single.sh start                # port 8000
cd ../model-b && ./model-b-single.sh start --port 8001         # port 8001
lmds ps                                                        # เห็นทั้งคู่
lmds stop --all                                                # ปิดทั้งคู่จบในคำสั่งเดียว
```

> ระบบรู้จักเซิร์ฟเวอร์จากไฟล์ทะเบียนที่ controller เขียนเองตอน `start` (ใต้ `~/.lmds/run/`)
> — ถ้า controller ถูกลบ/ย้าย `lmds stop` ยัง fallback หยุดตรง ๆ ให้ได้ (kill pid / docker rm)

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

## 5. คำสั่งอื่นที่ควรรู้

```bash
lmds plan Qwen/Qwen3-32B --target dgx-spark-single   # ดูแผนอย่างเดียว ไม่สร้างไฟล์ (มี --json)
lmds generate ...                                    # เหมือน deploy แต่ไม่มีขั้นยืนยัน
lmds validate bundles/qwen3-32b                      # ตรวจ bundle ย้อนหลัง (เช็คว่าไม่มีใครแก้ไฟล์)
lmds validate bundles/qwen3-32b --fix                # regenerate checksum หลังตั้งใจแก้ไฟล์เอง
lmds hardware                                        # ตรวจเครื่อง
lmds config show                                     # ดู config (key ถูก mask)
```

## 6. เอา bundle ไปใช้เครื่องอื่น

Bundle เป็นไฟล์ธรรมดา ไม่ผูกกับเครื่องที่สร้าง:

```bash
scp bundles/qwen3-32b.zip user@server:/home/user/
# บนเครื่องปลายทาง (ต้องมี Docker + NVIDIA toolkit):
unzip qwen3-32b.zip && cd qwen3-32b
./qwen3-32b-single.sh download && ./qwen3-32b-single.sh start
```

> สร้าง bundle จากเครื่องไหนก็ได้ (ไม่ต้องมี GPU) โดยระบุ `--target` ของเครื่องปลายทาง

---

## 7. แก้ปัญหาที่พบบ่อย (Troubleshooting)

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
| อยากใช้ Ollama/vLLM local เป็นสมอง | — | `lmds config set-provider openai-compat --base-url http://<ip>:11434/v1 --model gpt-oss:20b` (Ollama) หรือ `--base-url http://<ip>:8000/v1` (vLLM) — ไม่มี key ก็ใช้ได้ |

### ถ้าแก้เองไม่ได้ — ข้อมูลที่ต้องเก็บส่งทีมพัฒนา

```bash
lmds version
lmds hardware
./xxx-single.sh logs 500 > failure.log
# + คำสั่งเต็มที่รันแล้วพัง + ข้อความ error ทั้งหมด
```

## 8. ความปลอดภัย — ข้อควรปฏิบัติ

- API key / HF token ใส่ผ่าน `lmds config set-key` หรือ env เท่านั้น — **ห้าม**เขียนลงไฟล์/สคริปต์เอง
- เซิร์ฟเวอร์ที่เปิดใน network ที่มีคนอื่นใช้ร่วม ให้ตั้ง `API_KEY=xxx ./xxx-single.sh start` เสมอ
- flag `--trust-remote-code` อนุมัติเฉพาะหลัง review ไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน SPECIAL_FILES.md)
- bundle ที่รับมาจากคนอื่น ตรวจก่อนใช้: `lmds validate <โฟลเดอร์>`
