# คู่มือใช้งาน LMDS (ละเอียด)

> ก่อนอ่านเอกสารนี้ ต้องติดตั้งเสร็จตาม [INSTALL.md](INSTALL.md) แล้ว (`lmds version` ใช้ได้)

## แนวคิดหลัก 30 วินาที

LMDS รับ**ลิงก์โมเดล** (Hugging Face) → วิเคราะห์ + คำนวณว่า fit กับเครื่องไหม → สร้าง **bundle**
(โฟลเดอร์ + ZIP) ที่ข้างในมีสคริปต์ controller สำหรับ download / start / ทดสอบโมเดลนั้นบนเครื่องจริง

ทุก bundle ผ่าน quality gates 7 ด่านโดยอัตโนมัติ — ถ้าไม่ผ่านจะไม่มีไฟล์ออกมาให้ใช้เลย

```text
ลิงก์โมเดล ──lmds deploy──▶ bundle/ ──./xxx-single.sh──▶ โมเดลรันเป็น API ที่ :8000/v1
```

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

### 3.3 โมเดล gated (เช่น Llama)

```bash
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single
# → ระบบตรวจพบว่า gated แล้วถาม:
#   Hugging Face token (Enter เพื่อข้าม):
```

ต้องกดยอมรับเงื่อนไขของโมเดลบนเว็บ huggingface.co ด้วย account เดียวกับ token ก่อน ไม่งั้น token ก็เข้าไม่ได้
ตอน `download` บนเครื่อง ให้ตั้ง `export HF_TOKEN=hf_xxx` ไว้ใน shell (สคริปต์อ่านจาก env — ไม่ฝังในไฟล์)

### 3.4 Target presets ที่มีให้เลือก

| preset | เครื่อง | หน่วยความจำ |
|---|---|---|
| `dgx-spark-single` | DGX Spark 1 เครื่อง | unified 128GB |
| `dgx-spark-stacked` | DGX Spark 2 เครื่อง | unified 256GB (ประมาณ) |
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

## 4. คำสั่งอื่นที่ควรรู้

```bash
lmds plan Qwen/Qwen3-32B --target dgx-spark-single   # ดูแผนอย่างเดียว ไม่สร้างไฟล์ (มี --json)
lmds generate ...                                    # เหมือน deploy แต่ไม่มีขั้นยืนยัน
lmds validate bundles/qwen3-32b                      # ตรวจ bundle ย้อนหลัง (เช็คว่าไม่มีใครแก้ไฟล์)
lmds validate bundles/qwen3-32b --fix                # regenerate checksum หลังตั้งใจแก้ไฟล์เอง
lmds hardware                                        # ตรวจเครื่อง
lmds config show                                     # ดู config (key ถูก mask)
```

## 5. เอา bundle ไปใช้เครื่องอื่น

Bundle เป็นไฟล์ธรรมดา ไม่ผูกกับเครื่องที่สร้าง:

```bash
scp bundles/qwen3-32b.zip user@server:/home/user/
# บนเครื่องปลายทาง (ต้องมี Docker + NVIDIA toolkit):
unzip qwen3-32b.zip && cd qwen3-32b
./qwen3-32b-single.sh download && ./qwen3-32b-single.sh start
```

> สร้าง bundle จากเครื่องไหนก็ได้ (ไม่ต้องมี GPU) โดยระบุ `--target` ของเครื่องปลายทาง

---

## 6. แก้ปัญหาที่พบบ่อย (Troubleshooting)

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

### ถ้าแก้เองไม่ได้ — ข้อมูลที่ต้องเก็บส่งทีมพัฒนา

```bash
lmds version
lmds hardware
./xxx-single.sh logs 500 > failure.log
# + คำสั่งเต็มที่รันแล้วพัง + ข้อความ error ทั้งหมด
```

## 7. ความปลอดภัย — ข้อควรปฏิบัติ

- API key / HF token ใส่ผ่าน `lmds config set-key` หรือ env เท่านั้น — **ห้าม**เขียนลงไฟล์/สคริปต์เอง
- เซิร์ฟเวอร์ที่เปิดใน network ที่มีคนอื่นใช้ร่วม ให้ตั้ง `API_KEY=xxx ./xxx-single.sh start` เสมอ
- flag `--trust-remote-code` อนุมัติเฉพาะหลัง review ไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน SPECIAL_FILES.md)
- bundle ที่รับมาจากคนอื่น ตรวจก่อนใช้: `lmds validate <โฟลเดอร์>`
