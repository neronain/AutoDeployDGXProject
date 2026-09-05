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
| `lmds inspect <โมเดล>` | วิเคราะห์ + เช็ก fit อย่างเดียว — ไม่สร้างไฟล์ ไม่เสีย token (`--context N` ถามว่าค่านี้ได้กี่คนพร้อมกัน) |
| `lmds plan <โมเดล>` | ดู Deployment Plan (แผน) — ไม่สร้างไฟล์ |
| `lmds deploy <โมเดล>` | flow เต็ม: วิเคราะห์ → วางแผน → **ยืนยัน** → สร้าง bundle + ZIP (`--gguf` เลือก quant ไม่ต้องมี tty · `--task embed` · `--engine sglang`) |
| `lmds generate <โมเดล>` | เหมือน deploy แต่**ข้ามขั้นยืนยัน** (ไม่ต่อรอง flag) |
| `lmds validate <โฟลเดอร์>` | ตรวจ bundle ย้อนหลัง — exit 0 ผ่าน / 2 ไม่ผ่าน |
| `lmds smoke <ชื่อ>` | พิสูจน์ว่า bundle รันได้จริง: download → verify → start → test-text → stop (ดู §5.5) |
| `lmds rebuild <ชื่อ>` | สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน ไม่เรียก LLM ซ้ำ (ดู §5.6) |
| `lmds adopt [container]` | รับโมเดลที่รันอยู่ก่อน LMDS เข้าระบบ (`--port` / `--pid` สำหรับ process ตรง ๆ) (ดู §4.7) |
| `lmds ps` / `list` | ดูว่ามีอะไรอยู่บ้าง + สถานะจริง (ดู §4) |
| `lmds start` / `stop` / `restart` / `logs` | สั่งงานโมเดลตามชื่อ (ดู §4) |
| `lmds enable` / `disable` | autostart หลัง reboot — user service ไม่ต้อง sudo (ดู §4) |
| `lmds set <ชื่อ> …` | บันทึกค่า start ไว้กับ bundle (port/context/parser/image/env/extra-args) — ทุกทางที่เรียก controller ได้ค่าเดียวกัน (ดู §4.2d) |
| `lmds repair <ชื่อ>` | โหลดไฟล์ที่ขาด/เสียกลับมา แล้วตรวจซ้ำ (ดู §4.3) |
| `lmds remove <ชื่อ>` | ลบโมเดลออกจากเครื่องทั้งหมด (`--dry-run` ดูก่อน) (ดู §4.3) |
| `lmds doctor <ชื่อ>` | ตรวจว่าทำไม download/start ไม่ผ่าน + คำสั่งแก้ (ดู §4.4) |
| `lmds scan` / `prune` | weight ที่มีอยู่แล้วบนเครื่อง · ล้างทะเบียนค้าง (ดู §4.6 / §4.5.2) |
| `lmds recipes` | สูตรที่รันผ่านจริง — `--sync` ดึงจากคลัง · `--publish` ส่งขึ้นคลัง (ดู §4.5.1) |
| `lmds bench run/list/show/remove` | ให้คะแนนโมเดลที่รันอยู่ — ความเร็ว + ความสามารถ 7 ข้อ (ดู [BENCH.md](BENCH.md)) |
| `lmds node …` | คุมเครื่องอื่นจากเครื่องนี้ — add/list/remove/install/setup/set/run/ctl/push/clone/cluster (ดู §4.5) |
| `lmds cluster show/write/pair/doctor` | คลัสเตอร์ stacked: ดูคู่ · เขียน cluster.env · กุญแจ head→worker · หมอ (ดู §4.5) |
| `lmds web` | หน้าเว็บคุมทุกอย่าง — UI ภาษาอังกฤษ · `--enable` = ขึ้นเองหลังรีบูต (ดู §5) |
| `lmds hardware` | ตรวจเครื่อง + จำแนก target profile |
| `lmds config ...` | ตั้ง provider / key / HF token |
| `lmds agent info/bench` | JSON ที่ hub เรียกผ่าน SSH — ปกติไม่ต้องพิมพ์เอง |
| `lmds version` | เวอร์ชัน + commit ที่รันอยู่จริง + มาตรฐาน template |

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
| `prepare-runtime` | เตรียม engine — **จำเป็นเฉพาะบางกรณี ดูด้านล่าง** (llama.cpp native: `start` เรียกให้เองถ้ายังไม่มี `llama-server`) |
| `start` / `stop` / `restart` | เปิด-ปิดเซิร์ฟเวอร์ (ตรวจ GPU + ไฟล์ก่อน start เสมอ · GGUF บน ARM64 ที่ยังไม่มี binary จะ build llama.cpp ให้ก่อน) |
| `status` | สถานะ container + API health |
| `logs [N]` | log ล่าสุด N บรรทัด (default 300) |
| `client-config` | ค่าตั้ง client เป็น JSON พร้อม token budget (bundle embedding: `max_input_tokens` = context ต่อ slot ทั้งก้อน · มี `pooling` · ไม่มี `max_output_tokens`) |
| `network-info` | bind address + endpoint ที่ประกาศให้ client |
| `test-text` | ทดสอบ chat completion หนึ่งครั้ง (bundle embedding: บอกให้ไปใช้ `test-embed` แทน) |
| `test-embed` | *(เฉพาะ bundle embedding)* ยิง `/v1/embeddings` 3 ประโยค — คู่ไทย↔อังกฤษความหมายเดียวกันต้องได้ cosine สูงกว่าประโยคที่ไม่เกี่ยว (ดู §4.9) |
| `test-vision` | *(เฉพาะโมเดล multimodal)* สร้างภาพสีแดงแล้วถามว่าเห็นสีอะไร — พิสูจน์ว่า mmproj โหลดจริง (vLLM/stacked: projector ฝังใน weight) |
| `parsers` | *(vLLM · SGLang · stacked)* ถามชื่อ `--tool-parser` / `--reasoning-parser` ที่ engine รองรับจริง — อ่าน registry `vllm.tool_parsers` (0.28 ย้ายที่) แล้วถอยไป grep `vllm serve --help` |
| `test-tools` | ตรวจว่าคำตอบถูกแปลงเป็น `tool_calls` ได้จริง (ค่าตั้งต้นวัดโหมด `auto` ที่ agent ใช้) — ใช้ได้ทุก bundle chat ไม่ใช่เฉพาะที่เปิด tool ไว้ตอนสร้าง · vLLM: ตัวแปลคือ `--tool-parser` · llama.cpp: **ไม่มี parser ให้เลือก** chat template ที่โหลดผ่าน `--jinja` เป็นคนแปล ถ้าไม่ผ่านคำสั่งจะอ่าน `chat_template_caps` จาก `/props` มาบอกว่า template รองรับ tools ไหม |
| `test-reasoning` | *(vLLM · SGLang · stacked)* ตรวจว่า `--reasoning-parser` แยก chain-of-thought ออกจากคำตอบได้จริง (37×43=1591) |
| `bench [RUNS] [TOKENS]` | *(vLLM เดี่ยว · stacked)* วัด ttft / tok/s ผ่าน API จริง — ค่าตั้งต้น 3 รอบ × 256 tokens พิมพ์ต่อรอบและ median (ดู [BENCH.md](BENCH.md)) |
| `stress [REQUESTS] [CONC]` | *(vLLM เดี่ยว · stacked)* ยิงพร้อมกันหลายสาย — ค่าตั้งต้น 16 คำขอ × 4 สาย พิมพ์ ok/total · latency p50/p95/max |
| `serve-args` | พิมพ์ argv จริงที่จะส่งให้ engine โดยไม่ start (llama.cpp / vLLM `DRY_RUN=1 start` / stacked: head+worker + engine env) — key ไม่ถูกพิมพ์ |
| `info` / `props` | สรุปโมเดล/พอร์ต/สถานะ · รายการโมเดลจาก `/v1/models` (vLLM/stacked) |
| `wait-health` | รอ `/health` ต่อ (ใช้เมื่อ start timeout แต่โมเดลยังโหลดอยู่) |
| `clear-fi-cache` | *(vLLM/stacked)* หยุดแล้วล้าง FlashInfer JIT cache — ใช้เมื่อ start พังหลังเปลี่ยน image |
| `runtime-info` · `sync-worker` · `verify-worker` · `doctor` · `logs [head\|worker] [N]` | *(stacked เท่านั้น)* ดู §3.3b |
| `remove-plan` | *(bundle ที่มาจาก `lmds adopt`)* บอกว่า `lmds remove` จะลบ weight ที่ไหน |

**คำสั่งไหนมีบน engine ไหน** (usage กับ dispatch table ของทุก template ถูกเทสว่าตรงกัน):

| | llama.cpp | vLLM เดี่ยว | SGLang | stacked (vLLM) |
|---|---|---|---|---|
| `download` `verify-files` `start` `stop` `restart` `status` `logs` `client-config` `network-info` `wait-health` | ✅ | ✅ | ✅ | ✅ (ไม่มี `wait-health`) |
| `prepare-runtime` | ✅ build จาก source (native) | เฉพาะ bundle ที่มี `runtime_assets` | เฉพาะ bundle ที่มี `runtime_assets` | ✅ pull + lock image ทุก node |
| `test-text` `test-tools` | ✅ (chat) | ✅ | ✅ | ✅ |
| `test-embed` | ✅ (embed) | ✅ (embed) | ❌ (embed บน SGLang ถูกปฏิเสธ) | ❌ |
| `test-vision` | ✅ ถ้ามี mmproj | ✅ ถ้าแผน multimodal | ❌ | ✅ ถ้าแผน multimodal |
| `test-reasoning` `parsers` | ❌ | ✅ | ✅ | ✅ |
| `bench` `stress` | ❌ (ใช้ `lmds bench`) | ✅ | ❌ | ✅ |
| `serve-args` | ✅ | `DRY_RUN=1 start` | `DRY_RUN=1 start` | ✅ |
| `info` | ✅ | ✅ | ❌ | ✅ (+ `props` `clear-fi-cache` `runtime-info` `doctor` `sync-worker` `verify-worker`) |

> **คำอธิบายเต็มของทุก option + วิธีตั้ง API token อยู่ใน help ของ controller เอง** (ภาษาอังกฤษ):
> `./xxx-single.sh` เปล่า ๆ หรือ `./xxx-single.sh help` — มีค่า default จริงของ bundle นั้นกำกับทุกบรรทัด

> **`prepare-runtime` ต้องรันเมื่อไหร่?** ตอน deploy เสร็จ ระบบจะพิมพ์ลำดับคำสั่งที่ถูกต้องของ bundle นั้นให้เสมอ — ทำตามนั้นได้เลย
> - **GGUF บน DGX Spark (ARM64)** — ต้อง build llama.cpp จาก source เพราะไม่มี Docker image ทางการ (ครั้งแรกครั้งเดียว ~10–30 นาที) · **`start` ทำให้เองถ้ายังไม่มี `llama-server`** — รัน `prepare-runtime` แยกเมื่ออยาก build ล่วงหน้า หรืออัปเดต (`LLAMA_CPP_UPDATE=1`) · ขอ sudo เฉพาะเมื่อขาด build deps จริง (ไม่มี sudo = บอกคำสั่ง apt ให้รันเอง)
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
./xxx-single.sh restart --tool-parser qwen3_coder  # เปิด tool calling (ดูหัวข้อถัดไป) — vLLM/SGLang/stacked
./xxx-single.sh restart --extra-args "--flag=value" # แฟล็กเพิ่มของ engine ต่อท้ายตรง ๆ (JSON เขียนติดกันไม่มีช่องว่าง)
./xxx-single.sh restart --name my-model            # ชื่อที่ client ใส่ในฟิลด์ model (llama.cpp — ชื่อเดิมยังโชว์คู่กัน)
./xxx-single.sh restart --image-min-tokens 1024    # llama.cpp vision: token ขั้นต่ำต่อภาพ (auto = ใช้ค่าจาก projector)
./xxx-single.sh restart --no-mmproj / --no-mtp     # ปิด vision / speculative decoding (โผล่เฉพาะ bundle ที่มีไฟล์)
```

ค่าที่อยากให้ติดถาวร (รวม autostart และปุ่ม test-* บนหน้าเว็บ) ใช้ `lmds set <slug> …` แทน (ดู §4.2d) —
flag บรรทัดคำสั่งมีผลครั้งนั้นครั้งเดียว

### ถามผู้ช่วยให้ไปดูเครื่องให้ (กล่องแชทมุมขวาล่าง)

ผู้ช่วยไม่ได้ตอบจากสถานะที่แคชไว้อย่างเดียวแล้ว — ก่อนตอบมันจะเลือกเครื่องมือจาก
แคตตาล็อกแล้วไปรันบนเครื่องที่เกี่ยวข้องผ่านทางเดียวกับ `lmds node` (SSH key ของ LMDS)

```text
คุณ:      โมเดล qwen3-coder บน spark-head ไม่ขึ้น ทำไม
ผู้ช่วย:  ดูมาแล้ว: ✓ log ของโมเดล  ✓ RAM และ swap  📖 gpu-util
          log ท้ายสุดเป็น CUDA out of memory ตอน warm-up · เครื่องมี Nemotron
          รันอยู่อีกตัวถือ VRAM ไว้ 42 GB — gpu-util 0.90 จึงไม่เหลือให้ตัวนี้
```

บรรทัด **"ดูมาแล้ว: …"** เหนือคำตอบคือรายการสิ่งที่มันไปดูจริง ✓ = สำเร็จ ✗ = คำสั่งนั้นล้ม
(เช่นเครื่องต่อไม่ติด) · 📖 = ไปเปิดเอกสารของ LMDS อ่าน

สิ่งที่มันดูได้ — ไม่ต้องจำชื่อ ถามเป็นภาษาคนได้เลย:

| กลุ่ม | ดูอะไร |
|---|---|
| โมเดลตัวหนึ่ง | `lmds doctor`, log ของ controller, สถานะ, ค่าที่ตั้งไว้ (context/พอร์ต/bind) |
| ตัวเครื่อง | GPU + process ที่ถือ VRAM, RAM/swap, ดิสก์และแคช, พอร์ตที่เปิดฟัง, docker |
| ทั้งคลัสเตอร์ | อินเทอร์เฟซ/IP/RoCE ของแต่ละเครื่อง, bundle ที่มีบนเครื่องนั้น |

ทั้งหมดเป็นการ**อ่านอย่างเดียว** จึงรันได้เลยโดยไม่ต้องขออนุมัติ

### เมื่อผู้ช่วยเสนอให้แก้ — เลือกจากเมนู

พอสาเหตุชัดพอ มันจะเสนอเป็นการ์ดพร้อม**คำสั่งเต็มที่จะรันจริง** ผลกระทบ แล้วถามกลับ:

| ปุ่ม | เกิดอะไรขึ้น | ใช้เมื่อ |
|---|---|---|
| **แก้เลย** | รันทุกขั้นให้จบในครั้งเดียว | งานเดียวจบ ความเสี่ยงต่ำ และคุณอ่านคำสั่งแล้ว |
| **ทีละขั้น** | รันขั้นเดียวแล้วหยุด ให้ดูผลก่อนกด "ทำขั้นถัดไป" | หลายขั้น หรืออยากรู้ว่าขั้นไหนได้ผล |
| **ยังไม่ทำ** | ไม่แตะเครื่อง แสดงคำสั่งไว้ให้เอาไปรันเอง | อยากทำเองบนเครื่อง หรือยังไม่ตัดสินใจ |

> **ไม่มีอะไรทำงานจนกว่าจะกดปุ่ม** — ตั๋วอนุมัติออกโดยเซิร์ฟเวอร์ ผู้ช่วยออกให้ตัวเองไม่ได้
> · ขั้นที่ล้มจะหยุดขั้นที่เหลือทันที เพราะขั้นถัดไปตั้งอยู่บนสมมติฐานว่าขั้นก่อนหน้าสำเร็จ
> · ตั๋วหมดอายุใน 30 นาที กดค้างไว้ข้ามวันแล้วมากดทีหลังไม่ได้

งานที่เสนอได้มีเท่าที่อยู่ในแคตตาล็อก (`src/lmds/assistant/catalog.py`): start/stop/restart,
เปลี่ยน context, พอร์ต, bind address, gpu-memory-utilization, ล้างแคช FlashInfer,
เตรียมรันไทม์ · **LLM เขียนคำสั่งเองไม่ได้** มันเลือกได้แค่ชื่อรายการกับค่าที่ผ่านการตรวจ

อยากแก้สิ่งที่ไม่มีในรายการ (เช่น flag เฉพาะของ vLLM รุ่นนั้น) ใช้ทางถัดไปแทน:

### ให้ผู้ช่วยเสนอวิธีแก้ controller (Manage → Edit the launch script)

บาง knob ไม่มีปุ่ม และบางอย่างที่ต้องแก้ก็เฉพาะเครื่องจริง ๆ (flag ของ vLLM รุ่นนั้น,
env ที่เครื่องนั้นต้องการ) การ generate bundle ใหม่ทั้งชุดเพื่อแก้บรรทัดเดียวแพงเกินไป
แล้วคนก็จะไปแก้ด้วย `vi` บนเครื่องแทน ซึ่ง LMDS ไม่รู้เลยว่าไฟล์เปลี่ยนไปแล้ว

พิมพ์เป็นภาษาคนว่าอยากให้รันต่างจากเดิมยังไง แล้วกด **Propose** ผู้ช่วยจะอ่าน
controller **ตัวจริงบนเครื่องนั้น** แล้วตอบมาอย่างใดอย่างหนึ่ง:

| ตอบมาแบบ | แปลว่า |
|---|---|
| **option** | ทำได้ด้วยคำสั่งที่มีอยู่แล้ว เช่น `restart --context 65536` — **ไม่แก้ไฟล์** |
| **edit** | ต้องแก้ไฟล์จริง แสดง diff ให้อ่านก่อน แล้วค่อยกด Apply |
| **unsupported** | ทำให้ไม่ได้ พร้อมบอกว่าควรไปทำอะไรแทน |

**option มาก่อนเสมอ** — ถ้าสิ่งที่ขอทำได้ด้วย knob ที่มีอยู่ ผู้ช่วยต้องตอบเป็นคำสั่ง
ไม่ใช่ patch เพราะการแก้ไฟล์ทำให้ bundle ต่างจากที่ LMDS สร้าง และหายไปเมื่อ
generate ใหม่

**ไม่มีอะไรถูกเขียนจนกว่าจะกด Apply** และก่อนเขียนทุกครั้ง:

1. ข้อความเดิมที่ LLM อ้างว่ามีในไฟล์ ต้องมีจริงและ**มีครั้งเดียวเป๊ะ** ไม่งั้น
   ปฏิเสธทั้งข้อเสนอ — ระบบไม่เดาว่าหมายถึงจุดไหน
2. คิดเนื้อไฟล์ใหม่จากไฟล์ **ณ ตอนกด Apply** ไม่ใช่ตอนกด Propose — ถ้ามีคนแก้
   ไฟล์ระหว่างนั้น ระบบจะบอกให้ขอข้อเสนอใหม่แทนที่จะเขียนทับงานเขา
3. `bash -n` ต้องผ่าน — สคริปต์ที่ syntax เสียคือโมเดลที่ start ไม่ขึ้นอีกเลย
4. สำรองไฟล์เดิมเป็น `<ชื่อ>.bak-YYYYmmdd-HHMMSS` เสมอ ไม่มีข้อยกเว้น

> **LLM เขียนทับทั้งไฟล์ไม่ได้** — มันส่งได้แค่คู่ (ข้อความเดิม, ข้อความใหม่) ที่ระบบ
> ตรวจกับไฟล์จริงได้เอง · ให้เขียนทั้งไฟล์เมื่อไหร่ ก็จะได้สคริปต์ที่ "ดูดี" แต่ไม่ตรงกับ
> ของจริง ซึ่งเป็นปัญหาที่แพงกว่าปัญหาที่กำลังจะแก้
>
> เสนอได้สูงสุด 8 จุดต่อครั้ง — ใหญ่กว่านั้นคนรีวิวไม่ไหวในรอบเดียว

หลังเขียนแล้วต้อง **Restart** โมเดลถึงจะมีผล

### แยกความคิดออกจากคำตอบทีหลัง (`--reasoning-parser`)

เรื่องเดียวกับ `--tool-parser` ต่างแค่ flag · โมเดลสายคิด (Qwen3, DeepSeek-R1, GLM
และอื่น ๆ) จะคิดเป็นขั้นตอนก่อนตอบ ถ้า vLLM ไม่ได้รับ `--reasoning-parser` ตอน start
ความคิดทั้งก้อนจะไปอยู่ใน `content` แทนที่จะอยู่ใน `reasoning_content`

**อาการไม่ใช่ error** — เซิร์ฟเวอร์รันปกติ แต่ทุกอย่างที่เอาคำตอบไปแสดงจะได้กระบวนการคิด
ติดมาด้วย:

```
Thinking Process:
1. ผู้ใช้ถามว่า...
2. ดูจากข้อมูลที่มี...
</think>

คำตอบจริงอยู่ตรงนี้
```

ฝั่งที่รับไปแสดงจึงต้องมานั่งเดาว่าตรงไหนคือคำตอบ ซึ่งเดาผิดได้เสมอ · ทางแก้จริงคือ flag
ไม่ใช่การตัดข้อความ:

```bash
./xxx-single.sh restart --reasoning-parser deepseek_r1   # ครั้งนี้
REASONING_PARSER=deepseek_r1 ./xxx-single.sh start       # ถาวร (ใส่ใน env/unit)
./xxx-single.sh test-reasoning                           # พิสูจน์ว่าได้ผลจริง (37×43=1591)
```

ค่าว่าง = ปิด ซึ่งยังเป็นค่าตั้งต้นของโมเดลที่ไม่ใช่สายคิด

**เลือก parser ตัวไหน** — เหมือน tool parser คือแยกตามตระกูลโมเดล ใส่ผิดไม่ error
แต่จะไม่แยกอะไรออกมาเลย ที่ใช้บ่อย: `deepseek_r1` (DeepSeek-R1 และ Qwen3 ที่ใช้
`<think>`), `qwen3`, `nemotron_v3` (Nemotron 3), `granite` · **ถามชื่อจาก engine
ด้วย `./xxx-single.sh parsers`** อย่าอ่านจากชื่อไฟล์ (เหตุผลเดียวกับ tool parser)

**อ่านผล `test-reasoning` ให้ถูก** — `WARN` มีสองความหมายที่คนละเรื่องกัน และคำสั่ง
แยกให้แล้ว:

| ที่เห็น | แปลว่า | ต้องทำอะไร |
|---|---|---|
| `PASS` + จำนวนตัวอักษร | parser แยกได้จริง | ไม่ต้องทำอะไร |
| `WARN` + พบ `<think>` ใน content | ความคิดหลุดออกมาดิบ ๆ | parser ผิดหรือยังไม่ได้ตั้ง — แก้ |
| `WARN` + ไม่มีร่องรอย `<think>` เลย | โมเดลไม่ได้คิดในรอบนั้น | **มักไม่ใช่ปัญหา** ลองคำถามที่ต้องคิดหลายขั้นก่อนสรุป |

> เดิมคำสั่งนี้อ่านแค่ `reasoning_content` · vLLM รุ่นใหม่บางตัวใช้ชื่อ `reasoning`
> ทำให้ขึ้น WARN ทั้งที่ parser ทำงานอยู่ ตอนนี้อ่านทั้งสองชื่อแล้ว

⚠️ **สองหน้าต่างของ vLLM แยก reasoning ไม่เท่ากัน** — วัดจริงกับ
Nemotron-3-Super บน image ล่าสุด: `/v1/messages` (endpoint Anthropic ในตัว)
คืน `thinking` block มีเนื้อครบ ขณะที่ `/v1/chat/completions` คืน
`reasoning_content` ว่าง · ถ้าลูกค้าปลายทางคือ Claude Code ให้ดูฝั่ง
`/v1/messages` เป็นหลัก เพราะนั่นคือเส้นทางที่มันเดินจริง

> **ถ้าใช้ LiteGate อยู่ด้วย** ชุดทดสอบของมันจะรายงานเองว่า `reasoning_not_separated`
> พร้อมคำสั่งข้างบน — ไม่ต้องรอให้ผู้ใช้มาบ่นว่าคำตอบแปลก

### เปิด tool calling ทีหลัง (`--tool-parser`)

โมเดลจำนวนมากเรียก tool ได้ แต่ vLLM จะไม่เปิดให้ถ้าไม่ได้รับ
`--enable-auto-tool-choice` กับ `--tool-call-parser` ตอน start — และถ้า bundle
ถูกสร้างตอนที่ยังไม่มีข้อมูลว่าโมเดลตัวนั้นเรียก tool ได้ ก็จะไม่ได้ flag พวกนี้
อาการคือทุก request ที่ส่ง `tools` มาโดน 400:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

เดิมต้อง generate bundle ใหม่ทั้งชุด ตอนนี้ parser เป็น knob เหมือน port/context:

```bash
./xxx-single.sh restart --tool-parser qwen3_coder   # ครั้งนี้
TOOL_CALL_PARSER=qwen3_coder ./xxx-single.sh start  # ถาวร (ใส่ใน env/unit)
./xxx-single.sh test-tools                          # พิสูจน์ว่าได้ผลจริง
```

ค่าว่าง = ปิด ซึ่งยังเป็นค่าตั้งต้นของโมเดลที่ไม่รู้ parser

**เลือก parser ตัวไหน** — ถามจาก engine เอง อย่าเดาจากชื่อไฟล์:

```bash
./xxx-single.sh parsers
```

```
tool parsers  (--tool-parser):
  deepseek_v3 glm47 granite hermes kimi_k2 llama3_json minimax_m3 mistral
  pythonic qwen3_coder qwen3_xml seed_oss xlam ...
reasoning parsers  (--reasoning-parser):
  deepseek_r1 glm47 kimi_k2 nemotron_v3 qwen3 seed_oss ...
```

เดิมหัวข้อนี้เคยแนะให้ `ls` โฟลเดอร์ `vllm/tool_parsers/` แล้วอ่านชื่อไฟล์ —
ซึ่งผิด เพราะ **ชื่อไฟล์กับชื่อที่ลงทะเบียนไม่ตรงกัน** (ไฟล์ `qwen3xml.py`
ลงทะเบียนไว้ว่า `qwen3_xml`) และรายชื่อจริงอยู่ใน lazy registry ที่ยังไม่ถูก
import จนกว่าจะมีคนเรียกใช้ · `parsers` อ่านทั้งสองที่แล้วรวมให้

ที่ใช้บ่อย: `qwen3_coder` / `qwen3_xml` (ตระกูล Qwen รวม Nemotron-3 —
สอง**ชื่อ**นี้ชี้ไป parser **ตัวเดียวกัน** ใน vLLM รุ่นใหม่), `gemma4`
(Gemma 4 — **ไม่ใช่** `hermes`), `hermes`, `llama3_json`, `mistral`,
`deepseek_v3`

**ใส่ผิดจะไม่ error** แต่จะไม่คืน `tool_calls` เลย — ยกเว้นตอนที่ client ส่ง
`tool_choice: "required"` มา ซึ่งเป็นกับดักของหัวข้อถัดไป

อาการเวลาใส่ผิดหน้าตาเหมือน "โมเดลตัวนี้เรียก tool ไม่เป็น" เป๊ะ ๆ ทั้งที่
เรียกได้ · สิ่งที่ต่างคือ **call ไม่ได้หายไปไหน มันโผล่ใน `content` ในรูปแบบดิบ
ของโมเดล** ดูตรงนั้นแล้วจะรู้ทันทีว่าควรใช้ parser ตัวไหน:

| ถ้าเห็นใน `content` | parser ที่ถูก |
|---|---|
| `<\|tool_call>call:name{…}` | `gemma4` |
| `<start_function_call>` | `functiongemma` |
| `<tool_call>{json}</tool_call>` | `hermes` |
| `[TOOL_CALLS]` | `mistral` |
| `<\|python_tag\|>` | `llama3_json` |

```bash
# พิสูจน์ในคำสั่งเดียว: finish_reason ต้องเป็น tool_calls ไม่ใช่ stop
./xxx-single.sh test-tools
```

> **เคสจริง (msi-2, 2026-09-02)** — `google/gemma-4-31B-it` deploy ด้วย
> `--tool-call-parser hermes` เพราะตอนนั้น LMDS ยังไม่มีคำแนะนำสำหรับ Gemma
> คนจึงหยิบค่าที่คุ้นมือที่สุด vLLM ขึ้นปกติ `/health` เขียว ตอบ 200 ทุก request
> แต่คืน `finish_reason: stop` + `tool_calls: null` มาเป็นสัปดาห์ กว่าจะรู้ว่า
> พังก็ตอนเอาไปต่อ agent จริง · ตอนนี้ `arch_notes` เตือนตั้งแต่ตอนวางแผนแล้ว

#### `test-tools` วัดโหมดที่ agent ใช้จริง

```bash
./xxx-single.sh test-tools           # = both · auto ก่อน แล้วค่อย required
./xxx-single.sh test-tools auto      # เฉพาะเคสจริง
./xxx-single.sh test-tools required  # เฉพาะโหมดบังคับ
```

ความต่างสำคัญกว่าที่เห็น:

| `tool_choice` | ใครส่งมา | เกิดอะไรขึ้น |
|---|---|---|
| `auto` | Claude Code, Hermes, OpenClaw, agent ทุกตัว | โมเดลเขียนตามรูปแบบของมันเอง แล้ว **parser ต้องแปลให้ได้** |
| `required` | สคริปต์ทดสอบเป็นหลัก | engine บังคับรูปแบบด้วย guided decoding — ผ่านได้**แม้ parser ผิด** |

เคสจริง 2026-08-14: `test-tools` ขึ้น PASS แต่ Claude Code เห็นเป็นข้อความเปล่า
เพราะเทสยิงด้วย `required` · โมเดลเขียน `<function=…>` แบบ Qwen แต่ parser ที่ตั้ง
ไว้คือ `hermes` ซึ่งรอ JSON — พอเป็น `auto` จึงแปลไม่ออกและหลุดมาเป็น content
เทสที่ผ่านทั้งที่ของจริงพัง แย่กว่าไม่มีเทส เพราะมันทำให้เลิกสงสัย

ตอนนี้ค่าตั้งต้นจึงเป็น `both` และ **`auto` ไม่ผ่าน = ทั้งคำสั่งไม่ผ่าน** พร้อม
พิมพ์สิ่งที่โมเดลเขียนออกมาจริงและเดา parser ที่ตรงกับรูปแบบนั้นให้:

```
  auto      → ไม่มี tool_calls
  required  → get_weather({"location": "Bangkok"})

FAIL(auto): ไม่มี tool_calls — Claude Code และ agent อื่นจะเห็นเป็นข้อความเปล่า
  โหมด required ผ่าน เพราะ engine บังคับรูปแบบให้เอง
  แปลว่าโมเดลเรียก tool เป็น แต่ --tool-parser แปลรูปแบบของมันไม่ออก
  สิ่งที่โมเดลเขียนออกมาจริง:
    <tool_call> <function=get_weather> <parameter=location> Bangkok </parameter> …
  รูปแบบนี้ตรงกับ parser: qwen3_xml (ถ้าไม่ผ่านลอง qwen3_coder)
```

> **หมายเหตุ** `test-tools` ติดมากับทุก bundle แล้ว ไม่ใช่เฉพาะที่เปิด tool ไว้
> ตอนสร้าง — การให้สวิตช์เปิดได้แต่ไม่มีทางพิสูจน์ว่าได้ผล คือย้ายจุดบอด
> ไปที่ใหม่เฉย ๆ

### env ที่ควรรู้ (ใส่นำหน้าคำสั่ง หรือ export ไว้ก่อน)

ทุก flag ด้านบนมี env คู่กัน และมีอีกหลายตัวที่ตั้งได้เฉพาะทาง env:

| env | ค่า default | ใช้ทำอะไร |
|---|---|---|
| `API_PORT` / `API_HOST` | `8000` / `0.0.0.0` | port และ bind address |
| `API_KEY` | *(ว่าง)* | บังคับ Bearer token — **ควรตั้งเสมอถ้าเปิดออก network** · ไม่เก็บใน bundle ต้องส่งทุกครั้งที่ start (ดูกล่องด้านล่างว่าแต่ละ engine รับยังไง) |
| `MAX_MODEL_LEN` / `CTX_SIZE` | ตามแผน | context (เท่ากับ `--context`) — vLLM/SGLang ใช้ชื่อแรก llama.cpp ใช้ชื่อหลัง |
| `HF_HOME` | `~/.cache/huggingface` | ที่เก็บ weight ของ **vLLM** — ย้ายลงดิสก์ใหญ่ได้ (stacked: `WORKER_HF_HOME` สำหรับฝั่ง worker) |
| `MODEL_DIR` | `~/models/<slug>` | ที่เก็บไฟล์ **GGUF** ของ llama.cpp |
| `RUNTIME_MODE` | ตามเครื่อง | `docker` หรือ `native` (llama.cpp เท่านั้น) |
| `LLAMA_CPP_UPDATE` | *(ว่าง)* | `=1` กับ `prepare-runtime` = ข้าม `runtime.lock` ไป build llama.cpp รุ่นล่าสุด |
| `HF_TOKEN` | *(ว่าง)* | ใช้ตอน `download` repo gated — ส่งเข้า curl ทาง stdin (`-K -`) / aria2c ทางไฟล์ conf 600 ไม่ขึ้น argv |
| `FETCH_PARTS` | `8` | llama.cpp: จำนวนส่วนที่โหลดขนานสำหรับไฟล์ ≥256 MB (`1` = ปิด) · ต้องมีดิสก์ว่าง ~2 เท่าของไฟล์ ไม่พอถอยไปสตรีมเดี่ยวเอง |
| `FETCH_MAX_ATTEMPTS` | `20` | llama.cpp: จำนวนรอบ resume เมื่อ CDN ตัดสตรีมกลางคัน |
| `HTTPS_PROXY` / `HF_ENDPOINT` | *(ว่าง)* | proxy / mirror — ส่งเข้าคอนเทนเนอร์ download ด้วยชื่อ ไม่ขึ้น argv |
| `HEALTH_TIMEOUT` / `STARTUP_TIMEOUT` | ตามขนาดโมเดล | วินาทีที่รอ `/health` ตอน start (single / stacked) — systemd unit ตั้ง `TimeoutStartSec` ตามค่านี้ +300 |
| `TOOL_CALL_PARSER` / `REASONING_PARSER` | *(ว่าง = ปิด)* | parser ของ vLLM/SGLang (เท่ากับ `--tool-parser` / `--reasoning-parser`) |
| `EXTRA_SERVE_ARGS` / `ENGINE_ENV` | *(ว่าง)* | แฟล็กเพิ่ม (เท่ากับ `--extra-args`) · env ของ engine เอง (`lmds set --engine-env`) — stacked ส่งถึง worker ด้วย |
| `GPU_MEMORY_UTILIZATION` | ตามแผน | สัดส่วน VRAM ที่ vLLM จองได้ (ลดถ้าแชร์ GPU กับงานอื่น) |
| `MAX_NUM_SEQS` / `PARALLEL_SEQS` | ตามแผน | จำนวน request พร้อมกันสูงสุด (vLLM / llama.cpp slot) |
| `POOLING` / `EMBED_UBATCH` | ตามตระกูล | bundle embedding บน llama.cpp: วิธี pool (`last`/`cls`/`mean`) และ ubatch (ดู §4.9) |
| `IMAGE_MIN_TOKENS` | ตาม projector | llama.cpp vision (เท่ากับ `--image-min-tokens`) |
| `DRY_RUN` | *(ว่าง)* | `=1 … start` (vLLM/SGLang): พิมพ์ image + argv ที่จะรันจริง ไม่แตะ docker/GPU |
| `CONTAINER_NAME` | `lmds-<slug>` | ชื่อ container |
| `RUN_DIR` | `~/.lmds/run/<slug>` | ทะเบียน + log ที่ `lmds ps`/`lmds logs` อ่าน (และไฟล์ API key ของ llama.cpp) |

```bash
API_PORT=8001 API_KEY=secret123 ./xxx-single.sh start
HF_HOME=/data/hf-cache ./xxx-single.sh download     # ต้องใส่ตอน start ด้วยเสมอ
FETCH_PARTS=16 ./xxx-single.sh download             # เน็ตที่ต่อสายเดียวช้าแต่หลายสายพร้อมกันเร็ว (Xet ของ HF)
```

> **`API_KEY` ไปถึง engine ยังไง — ไม่มีทางไหนอยู่บน argv** (`ps` อ่าน argv ได้ทั้งเครื่อง):
>
> | engine | กลไก |
> |---|---|
> | llama.cpp | controller เขียน key ลงไฟล์ 0600 ใน `RUN_DIR` แล้วส่ง `--api-key-file` (docker: mount แบบ ro) — **ไม่ใช่** env `LLAMA_ARG_API_KEY` ซึ่ง build จริงไม่มี ตั้งแล้วเซิร์ฟเวอร์รันแบบไม่มี auth เงียบ ๆ |
> | vLLM เดี่ยว / stacked | export แล้ว `docker run -e VLLM_API_KEY` (ไม่มีค่าบน argv) — head เท่านั้นที่เสิร์ฟ API |
> | SGLang | ยังต้องส่ง `--api-key <ค่า>` (engine ไม่มี env คู่) — ข้อจำกัดที่รู้ตัว |
>
> `serve-args` / `DRY_RUN=1` ไม่พิมพ์ key · ตั้งแล้วตรวจให้ครบสามแบบ: ไม่ใส่ key → 401 · key ผิด → 401 · key ถูก → 200

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

#### เลือก quant ยังไงเมื่อเครื่องมีโมเดลอื่นรันอยู่แล้ว

`fit` หักหน่วยความจำที่โมเดลอื่นบนเครื่องนั้นถืออยู่ออกจากงบให้แล้ว — `budget_gb` ใน
`MODEL_PROFILE.yaml` จึงเป็นที่ว่าง **จริง** ไม่ใช่ความจุเต็ม และ verdict จะกลายเป็น
`fits-reduced-context` พร้อมลด context ให้เองเมื่อที่ไม่พอ

แต่ **การเลือก quant ยังเป็นของคุณ** และตัวเลือกที่ใหญ่ที่สุดไม่ใช่ตัวที่ดีที่สุดเสมอไป
บนเครื่อง unified memory การ generate ถูกจำกัดด้วย memory bandwidth — เร็วแค่ไหน
คำนวณได้ตรง ๆ:

```text
tokens/s ≈ bandwidth ที่ใช้ได้จริง ÷ ขนาดไฟล์โมเดล
```

วัดจริงบน DGX Spark (GB10) ได้ **≈ 215 GB/s** (79% ของ 273 GB/s ตามสเปก ซึ่งเป็นสัดส่วน
ปกติของ llama.cpp) · ตัวเลขนี้พยากรณ์ได้แม่น — เทียบ Gemma-4-31B บนเครื่องเดียวกัน:

| quant | ขนาด | คาดการณ์ | วัดจริง |
|---|---|---|---|
| Q8_0 | 32.6 GB | 6.6 tok/s | **6.5 tok/s** |
| Q5_K_M | 21.8 GB | 9.8 tok/s | — |
| Q4_K_M | 18.7 GB | 11.5 tok/s | — |

เลือก Q8_0 บนเครื่องที่มีโมเดลอื่นอยู่แล้ว = ได้คุณภาพเพิ่มนิดเดียวแลกกับความเร็วครึ่งหนึ่ง
และเบียดที่ของโมเดลตัวเดิมด้วย · ถ้าเครื่องมีที่เหลือเยอะและต้องการคุณภาพสูงสุดค่อยใช้ Q8_0

### 3.3 โมเดล gated (เช่น Llama)

```bash
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single
# → ระบบตรวจพบว่า gated แล้วถาม:
#   Hugging Face token (Enter เพื่อข้าม):
```

ต้องกดยอมรับเงื่อนไขของโมเดลบนเว็บ huggingface.co ด้วย account เดียวกับ token ก่อน ไม่งั้น token ก็เข้าไม่ได้
ตอน `download` บนเครื่อง ให้ตั้ง `export HF_TOKEN=hf_xxx` ไว้ใน shell (สคริปต์อ่านจาก env — ไม่ฝังในไฟล์)

### 3.3-0 หลายคลัสเตอร์ในไซต์เดียวกัน

การจับกลุ่ม stacked ใช้ **ไซต์ · ชื่อคลัสเตอร์ · ลายเซ็นฮาร์ดแวร์** เป็นกุญแจ แล้วจึงแบ่งย่อย
ตาม subnet ที่ใช้ร่วมกัน

```bash
lmds node set n1 --cluster-name ทีมค้นหา     # n1+n2 คลัสเตอร์หนึ่ง
lmds node set n3 --cluster-name ทีมสำรอง     # n3+n4 อีกคลัสเตอร์ แม้อยู่วงเดียวกัน
lmds node cluster                            # เห็นสองกลุ่มแยกกัน
```

ว่าง = ระบบแบ่งเองตาม subnet · **คนละไซต์จับคู่กันไม่ได้เลย** แม้เลขวงจะบังเอิญตรงกัน
(stacked ต้องยิง NCCL บนสายในแร็ค ไม่ใช่ผ่าน WAN)

**บนหน้าเว็บ:** Check cluster → ช่อง **คลัสเตอร์** ในแถวของแต่ละเครื่อง → Save ·
หัวกลุ่มขึ้นป้ายไซต์และชื่อคลัสเตอร์ (หรือ "แบ่งอัตโนมัติ")

### 3.3a ทำสำเนาโมเดลไปเครื่องอื่น (failover / กระจายโหลด)

โมเดลตัวเดียวกันบนหลายเครื่อง = มีตัวสำรองเวลาเครื่องหนึ่งล่ม และแบ่งโหลดผ่าน gateway ได้ ·
แต่ไม่มีเหตุผลที่จะโหลดจาก Hugging Face ใหม่ทุกครั้ง ในเมื่อเครื่องข้าง ๆ ถือไฟล์ชุดเดียวกันอยู่แล้ว

```bash
lmds node clone <slug> --from msi-1 --to msi-2            # คัดลอก + ตรวจ SHA-256 ที่ปลายทาง
lmds node clone <slug> --from msi-1 --to msi-2 --start    # เปิดใช้งานต่อเลย
lmds node clone <slug> --from msi-1 --to msi-2 --dry-run  # ดูก่อนว่าจะคัดลอกอะไรบ้าง
```

- ไฟล์วิ่ง **ตรงจากต้นทางไปปลายทาง ไม่ผ่าน hub** — hub มักเป็นเครื่องเล็กที่จะเป็นคอขวด
- เลือก **สายเร็วที่สุดที่ทั้งคู่มี** เอง: ถ้าตั้ง `cluster_ip` (ConnectX 200G) ไว้ทั้งคู่ก็ใช้เส้นนั้น
  ไม่งั้นถอยไปเส้นปกติ · ดูว่าใครมีบ้างด้วย `lmds node cluster`
- คัดลอกทั้ง weight และ bundle แล้ว `verify-files` ที่ปลายทางให้อัตโนมัติ
- ต่อกันคนละไซต์ก็ทำได้ แต่จะเตือน เพราะข้อมูลจะวิ่งข้ามเน็ตนอก
- ต้นทางต้องมี `rsync` (`sudo apt install -y rsync`)

> **กุญแจไม่เคยออกจาก hub** — node แต่ละเครื่องไม่มี key ของกันและกันโดยตั้งใจ ·
> คำสั่งนี้สร้างกุญแจชั่วคราวสำหรับงานครั้งเดียว ฝาก public key ไว้ที่ปลายทางแบบ `restrict`
> ส่ง private key ให้ต้นทางทาง stdin เข้า `ssh-agent` **ในหน่วยความจำ ไม่แตะดิสก์**
> แล้วถอนกุญแจออกเสมอเมื่อจบ ไม่ว่าจะสำเร็จหรือล้มกลางคัน

### 3.3b Deploy แบบ stacked (โมเดลใหญ่เกิน 1 เครื่อง → 2× DGX Spark)

> **ทำจากหน้าเว็บได้แล้ว (ไม่ต้องแตะเทอร์มินัล)**
>
> กด **Check cluster** → กลุ่มที่ขึ้นว่า "stacked ได้" จะมีปุ่ม **Deploy ลงกลุ่มนี้** ·
> กดแล้ว wizard เปิดขึ้นโดยตั้ง target เป็น `dgx-spark-stacked` และเลือก head/worker
> ให้ตามสมาชิกของกลุ่มเรียบร้อย · พอ generate เสร็จ ระบบ push ไปเครื่อง head แล้ว
> **เขียน `cluster.env` ให้เองด้วย** (MASTER_IP/WORKER_IP/SSH_USER/NCCL iface) —
> ไม่ต้องไปแก้ CONFIG ต้นไฟล์เอง
>
> ก่อนหน้านี้หน้า Cluster ได้แค่พิมพ์คำสั่งให้ไปก็อป จึงเหมือน "deploy stacked ไม่มีในหน้าเว็บ"

โมเดลที่ใหญ่เกิน unified memory ของ Spark เครื่องเดียว (เช่น DeepSeek-V4-Flash ~168GB) ให้ใช้ target `dgx-spark-stacked` — lmds จะสร้าง controller แบบ **multi-node** (worker-first startup, TP ข้าม node, mp backend) แทนแบบเดี่ยวอัตโนมัติ

```bash
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 --target dgx-spark-stacked
# → ได้ bundle: <slug>-stacked.sh (ไม่ใช่ -single.sh)
```

controller ที่ได้มีคำสั่งครบวงจร multi-node — รันจาก **head (master) ในฐานะ user ปกติ (ห้าม sudo)**:

```bash
# บน hub — ก่อนแตะ controller:
lmds cluster doctor spark-head spark-worker --slug <slug>   # ทำไมคู่นี้ยังไม่พร้อม — ทีละข้อพร้อมคำสั่งแก้ (อ่านอย่างเดียว)
lmds cluster pair spark-head spark-worker                   # ให้ head ssh เข้า worker ได้ — กุญแจเกิดบน head ไม่ผ่าน hub
lmds cluster write <slug> --head spark-head                 # เขียน cluster.env ลง bundle บน head (ตัดกลุ่มตาม NNODES ของ bundle)
lmds node push spark-head <slug> --download --start         # หรือทั้งหมดในคำสั่งเดียว: push → cluster.env → pair → sync/verify → start

# บน head:
cd bundles/<slug>
./<slug>-stacked.sh prepare-runtime   # pull + lock image ให้ image-ID ตรงกันทุก node (บอกชื่อเครื่องที่ pull ล้มพร้อมสาเหตุ)
./<slug>-stacked.sh download          # ดาวน์โหลดโมเดลลง head (ตรวจดิสก์ก่อน · เขียนลง $HF_HOME/hub/)
./<slug>-stacked.sh verify-files      # ตรวจ shard + config
./<slug>-stacked.sh sync-worker       # rsync โมเดล → worker (verify ฝั่ง head ก่อน · แปล exit code ของ rsync ให้)
./<slug>-stacked.sh verify-worker     # ตรวจ**ขนาดทุก shard** บน worker เทียบ Hub (rsync --partial ทิ้งไฟล์ครึ่งเดียวชื่อเดิมไว้ได้)
./<slug>-stacked.sh start             # ตรวจสถาปัตยกรรม + image ทุก node → เปิด worker (rank 1) ก่อน แล้ว head (rank 0) + รอ /health
./<slug>-stacked.sh status            # เทียบ id จาก /v1/models กับ served name — ไม่ตอบ "healthy" ให้โมเดลอื่นที่ยึดพอร์ต
./<slug>-stacked.sh logs worker 200   # log ฝั่ง worker (หน้าเว็บ: ปุ่ม logs-worker)
./<slug>-stacked.sh test-tools        # ชุด test-tools / test-reasoning / test-vision / parsers / bench / stress คุยกับ head ที่ 127.0.0.1 — ไม่ต้องมี cluster.env
```

- **ทุกคำสั่งที่แตะ worker ต้องรู้คลัสเตอร์ก่อน** — มี tty จะถาม IP · ไม่มี (hub สั่ง) = หยุดพร้อมบอก `lmds cluster write …`
  หรือ env `MASTER_IP`/`WORKER_IP`/`SSH_USER` · `status`/`network-info` ติดป้าย "ยังไม่ตั้งค่า" · IP ที่ตอบ prompt ถูกใช้
  คำนวณ `TRANSPORT_IP_*`/`WORKER_IPS` ใหม่จริง (คลัสเตอร์ >2 เครื่องถามรายการ worker ทั้งหมด)
- **ssh head→worker ถูกทดสอบก่อนงานยาว** (`ssh -o BatchMode=yes`) แล้วบอกทีละขั้นว่าขาดอะไร — key ที่ hub ใช้เข้า head
  **ไม่ใช่** key ของ head จึงต้อง `lmds cluster pair` (หรือ `ssh-keygen` + `ssh-copy-id` บน head เอง)
- **ก่อนปล่อย worker** controller ถาม image ตัวเดียวกับที่จะ start ว่ารู้จัก `model_type` ใน `config.json` ไหม — ไม่รู้จัก
  = หยุดพร้อมคำสั่ง `lmds set <slug> --image <ใหม่กว่า>` แทนที่จะทำครบทุกขั้น 2.5 ชม. แล้วตายตอนอ่าน config ·
  ตรวจ image บน**ทุก** worker ด้วย (เครื่องท้ายไม่มี image เคยผ่านด่านไปตายตอน `docker run`)
- ระหว่างรอ head health แวะดู worker ทุก 60 วิ — worker ตายแล้วพิมพ์ log 100 บรรทัด หยุด head แล้วบอก IP
- env ของสูตร/`lmds set --engine-env`, image digest, `--tool-parser`, `--extra-args`/bundle.args ถึง **ทั้ง head และ worker** ·
  worker คุย NCCL ด้วย transport IP (`TRANSPORT_IP_WORKER` · 3–4 เครื่อง: `TRANSPORT_IPS_WORKER`) ไม่ใช่ management IP
- autostart: `render_unit` ตั้ง `TimeoutStartSec` ≥ `STARTUP_TIMEOUT` ของ bundle (+300) — โมเดล 150–220 GB โหลดเกิน 1800 วิ
  ไม่โดน systemd ฆ่ากลางทางอีก
- `lmds clone` กับ head ของ stacked ไม่ลาก `cluster.env` ของคู่เก่าไปคู่ใหม่ — เขียนใหม่ด้วย `lmds cluster write`

> **สถานะ**: bundle stacked ที่ generate จาก LMDS **รันจริงแล้ว** บน 2× DGX Spark — Llama 3.3 70B (2026-08-05),
> `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4` 173 GB (vLLM 0.28 nightly, TP=2, tool calling ผ่าน) และ
> `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`--trust-remote-code --mamba-ssm-cache-dtype float16` ·
> parser `qwen3_coder`/`nemotron_v3`) (2026-09-04) · เกิน 2 เครื่องยังไม่เคยรันจริง · `runtime_assets` (parser plugin)
> ยังไม่รองรับในโหมด stacked

**stacked ให้อะไร** — หน่วยความจำรวม (weights/N ต่อเครื่อง) · KV pool ใหญ่ขึ้น · รับคนพร้อมกันได้มากขึ้น — **ไม่ใช่ tok/s
ต่อคน** โมเดลที่ลงเครื่องเดียวได้รันเครื่องเดียวเร็วกว่าเสมอ · fit ของ stacked หัก NCCL buffer 3 GB/เครื่องและรายงาน
`per_node` (capacity · OS · engine · comm buffer · budget · weights/N · KV/N) · vLLM ที่เหลือ KV < 2 GB = start ไม่ขึ้น
ระบบตอบ "ไม่ fit" ไม่ใช่ "fits ที่ context 4096"

ข้อกำหนด: 2× DGX Spark (preset `dgx-spark-stacked` · 4 เครื่อง `dgx-spark-stacked-4`) + fabric ระหว่าง node (แนะนำ 200 Gb/s
RoCE) + ssh head→worker แบบไม่ถามรหัส (`lmds cluster pair`) · `lmds ps`/`lmds stop`/`lmds logs` เห็น/สั่งงานตัวนี้ได้เหมือน
deploy เดี่ยว (stop จะหยุดทุก node ให้ · การ์ด worker บนหน้าเว็บขึ้นแถว "stacked worker of <head>") · stacked รองรับเฉพาะ vLLM
+ safetensors — GGUF / SGLang / embedding กับ target stacked ถูกปฏิเสธตั้งแต่ analyze (422) พร้อมทางออก

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
| `dgx-spark-stacked-4` | unified 128GB × 4 (TP=4) | | `rtx-3090-ti` / `rtx-3090` | 24GB |
| | | | `rtx-3080-ti` / `rtx-3080` | 12GB / 10GB |
| | | | `rtx-3060` | 12GB |

รวม 22 preset (7 ตัว tested) — รายชื่อจริงอยู่ที่ `src/lmds/fit/targets.py` และกด TAB หลัง `--target` ได้

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
--gguf FILE|QUANT   # repo GGUF หลาย variant: เลือกไฟล์โดยไม่ต้องมี tty — ชื่อไฟล์เต็ม หรือชื่อ quant (Q8_K_XL, Q4_K_M)
--engine vllm|sglang  # เลือกรันไทม์เอง (ดู §5) · --task generate|embed บังคับชนิดงานเมื่อเดาผิด (ดู §4.9)
```

> **`--gguf`** — repo แบบ `unsloth/…-GGUF` มักมี 10–20 quant · โหมดโต้ตอบจะแสดงรายการให้เลือกหมายเลข แต่ script/hub
> ไม่มี tty จึงเคย exit 1 "ต้องระบุไฟล์" · ใส่ `--gguf Q8_0` (ไม่สนตัวพิมพ์) หรือชื่อไฟล์เต็ม · ชื่อที่ตรงหลายไฟล์
> (`--gguf q8` เจอทั้ง Q8_0 และ Q8_K_XL) จะถูกปฏิเสธพร้อมรายการให้เลือกใหม่ ไม่เดาให้ · ใช้กับ `generate` ได้เหมือนกัน

> **`--concurrency` มีผลกับ memory โดยตรง** — KV cache โตตามจำนวน request ที่รันพร้อมกัน
> ใส่ `--concurrency 4` แปลว่า "กันหน่วยความจำเผื่อ 4 คนใช้พร้อมกัน" ผลคือ context ที่แนะนำจะลดลง
> · กับ llama.cpp แผนจะตั้ง slot = N และ `--ctx-size` = N × context ต่อ slot (llama-server แบ่ง pool ให้ทุก slot เท่า ๆ กัน)
> ค่าที่แต่ละ request ได้จริงคือค่าที่ fit รายงาน ไม่ใช่ตัวเลข `--ctx-size`
> ตั้งให้ตรงกับการใช้งานจริง: เดโม่/คนเดียว = 1 · ทีมเล็ก = 2–4 · ตั้งสูงเกินจริงจะได้ context สั้นโดยไม่จำเป็น

### 3.6 ขั้นยืนยัน — จุดที่ต้องอ่านก่อนกด

1. **อนุมัติ flag นอก allowlist** — ถ้าแผนเสนอ flag พิเศษ (เช่น `--trust-remote-code`) ระบบถามทีละตัว
   ค่า default คือ**ไม่อนุมัติ** — อนุมัติเฉพาะเมื่อเข้าใจผลของ flag นั้น (อ่าน SPECIAL_FILES.md ประกอบ)
2. **context** — Enter ใช้ค่าที่คำนวณให้ หรือพิมพ์เลขใหม่ (เกินเพดานปลอดภัยระบบจะลดให้อัตโนมัติ)
3. **ยืนยันสร้าง bundle** — Y/n

**บนหน้าเว็บ (0.5.2+)** หน้า plan มีแถบหน่วยความจำของเครื่องปลายทาง:
`capacity · OS+engine overhead · already in use · weights · KV ที่ context ที่เลือก · spare`
— แถบวาดใหม่ทุกครั้งที่พิมพ์ context หรือสลับ fp8 · **already in use** คือของที่โมเดลตัวอื่น
บนเครื่องนั้นถืออยู่จริง (อ่านจากการ์ดเครื่องในหน้า Fleet) และถูกหักออกจาก budget แล้ว ·
ต้องเลือกเครื่องในช่อง **Run on** ถึงจะได้ค่านี้ — เลือกแค่ preset = คิดจากเครื่องว่าง
เพราะ preset เป็นเครื่องสมมติ · ขึ้นว่า "ยังไม่มีข้อมูล…คิดจากความจุเต็ม" = กด refresh
ที่การ์ดเครื่องนั้นก่อนแล้ววิเคราะห์ใหม่

**ไม่รู้ชื่อ tool/reasoning parser ต้องใส่อะไร (0.5.2+)** — ไม่ต้องรู้:
- bundle ที่สร้างใหม่ของ Qwen3/3.5/3.6, Qwen3-Coder, Gemma 4 ได้ parser ที่ถูกตั้งแต่ plan (มีคำเตือน
  "เปิด tool calling ให้แล้ว" บอกไว้) · ตระกูลที่ระบบไม่รู้จะไม่เดา
- bundle ที่มีอยู่แล้ว: ฟอร์ม settings ของโมเดล → หมวด Advanced → กด **Fill from model (parsers)** → ระบบเติม parser / image /
  engine env จากสูตรที่รันผ่านจริงหรือกฎตระกูล พร้อมบอกที่มาทีละค่า → ตรวจแล้วกด **Save**
  · ทาง CLI: `lmds set <slug> --auto` · ใส่ผิดไม่พังเงียบ: ชื่อที่ vLLM ไม่รู้จัก start ไม่ขึ้นและบอกชื่อที่ถูก ·
  พิสูจน์หลัง start ด้วยปุ่ม **test-tools** (ต้องได้ finish_reason = tool_calls)

**context / slots / gpu-util ตั้งเท่าไรถึงพอ (0.5.2+)** — ใต้แถวนั้นในฟอร์ม settings มีบรรทัดคำนวณสด:
vLLM จอง `gpu-util × แรมทั้งเครื่อง` → หัก weights + overhead → ที่เหลือคือ KV → หารด้วย KV ต่อคำขอที่ context
ที่ตั้ง = รับได้กี่คำขอเต็ม context พร้อมกัน · ❌ = ไม่ start แน่ (KV ไม่พอ 1 คำขอ หรือจองเกินที่เครื่องว่างอยู่)
พร้อมตัวเลขที่ต้องแก้ · ⚠️ = start ได้แต่ slots ที่ตั้งเกินที่ KV รองรับเมื่อทุกคนใช้เต็ม context ·
บนเครื่องที่รันโมเดลอื่นอยู่ ใช้สูตร `gpu-util ≈ (แรมว่าง − 2) ÷ แรมทั้งเครื่อง`

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

### 4.2b เครื่องนี้รันโมเดลได้ไหม (control plane)

ถ้าเครื่องที่คุณนั่งอยู่ไม่มี GPU ไม่มี docker และไม่มี `llama-server` LMDS ถือว่ามันเป็น
**control plane** — มีไว้สร้าง bundle แล้วส่งไปรันที่อื่น ปุ่ม Download/Start บนคอนโซลจะกลายเป็น
**ส่งไปเครื่องที่รันได้** และคำสั่ง `repair`/`start` จะถูกปฏิเสธพร้อมบอกคำสั่ง push ที่ควรใช้แทน

ทับด้วย `--force` หรือ `LMDS_ROLE=serving` ได้เมื่อการตรวจเดาผิด · รายละเอียดเต็มอยู่ที่
[FLEET-MULTI-NODE.md §1.5](FLEET-MULTI-NODE.md)

### 4.2c ตั้ง env ของ engine เอง (`--engine-env`)

vLLM/SGLang มี knob จำนวนมากที่อ่านจาก **environment ล้วน ๆ** ส่งผ่าน flag ไม่ได้เลย ·
ตั้งได้ด้วย:

```bash
lmds set <slug> --engine-env "VLLM_NVFP4_GEMM_BACKEND=marlin"
lmds set <slug> --engine-env "A=1 B=2"        # หลายตัวคั่นด้วยช่องว่าง
lmds restart <slug>
```

ค่าถูกเก็บใน `bundle.env` ข้าง controller เหมือน knob อื่น — **ทุกทางที่เรียก controller
ได้ค่าเดียวกัน** รวมถึง systemd autostart ตอน reboot และปุ่ม `test-*` บนหน้าเว็บ ·
docker-based engine แตกเป็น `-e` ให้เอง · llama.cpp รัน native จึง export ตรง ๆ ·
stacked ส่งให้ทั้ง head และ worker

ค่าถูกตรวจก่อนเขียน: ต้องเป็น `KEY=VALUE` และห้ามมีอักขระที่เชลล์ตีความ (`$`, backtick,
quote, backslash) เพราะ controller แตกค่านี้ในเชลล์

**เคสจริงที่ทำให้ต้องมีช่องนี้** — NVFP4 บน DGX Spark (GB10, sm_121):

vLLM ตรวจว่ามี flashinfer CUTLASS fused-MoE ไหม *ด้วยการ import* ซึ่ง JIT ทันที แล้ว
`ptxas` ปฏิเสธ (`cvt with .e2m1x2 not supported on .target 'sm_121'`) engine core ตาย
ก่อน health โดยหน้าเว็บบอกแค่ "container หยุดก่อน health ผ่าน"

⚠️ **`VLLM_NVFP4_GEMM_BACKEND=marlin` ไม่ช่วยกับโมเดล MoE** — ทดสอบบน msi-6 แล้ว
(ยืนยันว่า env ถึง container จริงด้วย `docker inspect`) ยังล้มที่เดิม เพราะตัวแปรนี้คุม
**GEMM** ไม่ใช่ **fused MoE** · image ที่มี FP4 kernel มาให้ (`avarok/dgx-vllm-nvfp4-kernel`)
ก็ยังล้ม · สำหรับ MoE + NVFP4 บน sm_121 ตอนนี้ยังไม่มีทางที่ใช้ได้ — ไปทาง GGUF/llama.cpp
หรือ checkpoint ที่ไม่ใช่ NVFP4 แทน

โมเดล **dense** NVFP4 เป็นคนละ kernel path ยังไม่ได้ทดสอบว่าล้มด้วยไหม — ถ้าจะลอง
ตั้ง `--engine-env "VLLM_NVFP4_GEMM_BACKEND=marlin"` ไว้ก่อนแล้วดู log ว่ามี ptxas error
หรือไม่ ก่อนสรุปว่ารันได้

> bundle ที่สร้างไว้ก่อนหน้านี้ยังใช้ controller เดิม — สั่ง `lmds rebuild <slug>` ก่อน
> ถึงจะรับค่านี้ได้

### 4.2d `lmds set` — ค่าที่บันทึกไว้กับ bundle

flag ตอน `lmds start --port …` มีผลครั้งเดียว · systemd ตอน autostart และปุ่ม `test-*` บนหน้าเว็บเรียก controller
เปล่า ๆ จึงตกไปใช้ค่าเริ่มต้นของ bundle — `lmds set` เขียน `bundle.env` (และ `bundle.args` สำหรับ `--extra-args`)
ไว้ข้าง controller ซึ่งอ่านก่อนตั้ง default ทุกตัว · env จากภายนอกและ flag บรรทัดคำสั่งยังชนะไฟล์นี้เสมอ

| flag | เขียนอะไร | ใช้กับ |
|---|---|---|
| `--port` `--context` `--slots` `--bind` | `API_PORT` `MAX_MODEL_LEN`/`CTX_SIZE` `MAX_NUM_SEQS`/`PARALLEL_SEQS` `API_HOST` | ทุก engine |
| `--gpu-util` | `GPU_MEMORY_UTILIZATION` (0–1) | vLLM / SGLang |
| `--model-id` | ชื่อที่ API เสิร์ฟออกไป | ทุก engine |
| `--image` | image ที่ใช้แทนของ bundle (ตรึง digest ได้) — stacked ส่งถึง worker | vLLM / SGLang / stacked |
| `--engine-env "A=1 B=2"` | env ของ engine เอง — docker แตกเป็น `-e` · llama.cpp export ตรง · stacked ถึง worker | ทุก engine |
| `--tool-parser` `--reasoning-parser` | `TOOL_CALL_PARSER` `REASONING_PARSER` | vLLM / SGLang / stacked |
| `--image-min-tokens N\|auto` | `IMAGE_MIN_TOKENS` — Qwen-VL ~1024 · Gemma-4 ต้อง `auto` (เพดานแค่ 280) | llama.cpp vision |
| `--extra-args '…'` | `bundle.args` — แฟล็กเพิ่มต่อท้าย argv (JSON เขียนติดกัน · ใช้รูป `--flag=value` ได้) | ทุก engine |
| `--auto` | เติม parser / image / env จากสูตรที่รันผ่านจริง > กฎตระกูล · flag ที่ระบุเองชนะ | — |
| `--clear` | ลบค่าที่บันทึกไว้ทั้งหมด (หน้าเว็บ: ปุ่ม **Reset to bundle**) | — |

ค่าถูกตรวจก่อนเขียน: `port` 1–65535 · `context`/`slots` จำนวนเต็มบวก · `gpu_util` 0–1 · `bind` เฉพาะ `0.0.0.0`/`127.0.0.1` ·
`served_name`/`image` ห้ามมี `" ' \` $ \ { }` และ engine env ห้ามมี `{}` — เพราะไฟล์นี้ถูก `source` ทุกครั้งที่ start
(เคยรันคำสั่งจากช่องกรอกบนหน้าเว็บได้) · **ไม่เก็บ API key** — โฟลเดอร์ bundle ถูก zip แจกต่อได้ ส่ง `API_KEY=` ตอน start แทน
· ค่าพวกนี้ (image · parser · env · extra-args) ถูกพับลง header ตอน `lmds recipes --publish` ส่วน port/context/slots/bind
ไม่พับโดยเจตนา

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

`remove` จะ **แสดงรายการไฟล์ + ขนาดให้ดูก่อนเสมอ** แล้วค่อยถามยืนยัน (default = ไม่ลบ · `--dry-run` = ดูแล้วจบ)
สิ่งที่ทำตามลำดับ: หยุดเซิร์ฟเวอร์ → ยกเลิก autostart → ลบ bundle + ZIP + ทะเบียน/log +
runtime files + weight ของโมเดล

- **`--keep-weights` คุ้มมากกับโมเดลใหญ่** — ลบ bundle ทิ้งแล้ว deploy ใหม่ได้โดยไม่ต้องโหลดซ้ำหลายสิบ GB
- weight หาจาก `MODEL_PROFILE.yaml` (vLLM → HF cache, llama.cpp → `~/models/<slug>` เสมอ ไม่อ่าน `MODEL_DIR` จาก environ
  ของ hub ซึ่งเป็นของ bundle ที่กำลัง start อยู่) — ถ้าหาไม่เจอระบบจะ**ไม่เดา** (ไม่ลบอะไรที่ไม่แน่ใจ) ต้องลบเองถ้าต้องการ ·
  bundle ที่มาจาก `lmds adopt` จด path ของ weight ไว้ใน `MODEL_PROFILE["weights"]` และมีคำสั่ง `remove-plan`
- **ไฟล์ที่ container เขียนเป็น root** (weight ใน HF cache ของ vLLM) — `rm` ล้มด้วย EACCES แล้วผู้สั่งผ่านหน้าเว็บ/ssh ไม่มี tty
  กรอกรหัส sudo → ให้ root *ในคอนเทนเนอร์* ลบแทน (`docker run --rm -v <parent>:/x <image ที่มีในเครื่อง> rm -rf /x/<ชื่อ>`) ·
  รั้ว: เฉพาะใต้ home / `HF_HOME` เท่านั้น · ใช้ image ที่มีอยู่แล้ว (เล็กสุดก่อน) ไม่ pull ใหม่ · ทำเฉพาะเมื่ออยู่ในกลุ่ม docker

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
lmds enable gemma-4-26b-a4b-it-gguf          # ตั้ง autostart (user service — ไม่ต้อง sudo)
lmds enable gemma-4-26b-a4b-it-gguf --now    # ตั้ง + start เดี๋ยวนี้เลย
lmds list                                    # ดูคอลัมน์ autostart: ● เปิด / ○ ปิด
lmds disable gemma-4-26b-a4b-it-gguf         # ยกเลิก autostart
```

- ค่าเริ่มต้นเป็น **systemd user service** (`~/.config/systemd/user/lmds-<ชื่อ>.service`) — **ไม่ต้อง sudo เลย** เพราะ hub สั่งข้ามเครื่องผ่าน SSH ที่ไม่มี tty ให้กรอกรหัส · `--system` เขียนลง `/etc/systemd/system` ซึ่งต้อง sudo และให้สิทธิ์เท่ากับ root
- โมเดลใหญ่ที่โหลดนาน เพิ่มเวลา: `lmds enable <ชื่อ> --timeout 3600`
- เช็ก/ดู log: `systemctl --user status lmds-<ชื่อ>` · `journalctl --user -u lmds-<ชื่อ> -f` (เติม `--user` เสมอสำหรับ user service)
- **user service ต้องมี linger** ไม่งั้นมันขึ้นตอน login ไม่ใช่ตอนบูต: `loginctl enable-linger $USER` แล้วเช็กด้วย `loginctl show-user $USER -p Linger`
- **stacked (2 เครื่อง):** master ตั้ง autostart ได้ แต่ตอน boot worker ต้องเปิดอยู่ + SSH ถึงได้ ไม่งั้น start จะรอ/ล้ม

#### ⚠️ ตั้งแล้วต้องพิสูจน์ว่ามันขึ้นได้จริง

`enable` สำเร็จ **ไม่ได้แปลว่า** unit จะ start ได้ · `is-enabled` ตอบ `enabled` แค่บอกว่ามีลิงก์
ให้บูตเรียก ไม่ได้ลองเรียกดู เคยมีบั๊กที่ unit ทุกตัวตายตอนบูตโดยที่ทุกสัญญาณบอกว่าปกติดี
สั่งให้มันเริ่มจริงหนึ่งครั้งเสมอ:

```bash
systemctl --user start lmds-<ชื่อ> && systemctl --user is-active lmds-<ชื่อ>
```

ได้ `active` ถึงจะมั่นใจได้ · ถ้า `failed` ดูเหตุที่ `journalctl --user -xeu lmds-<ชื่อ>`

#### สโคปซ้อนกัน = ชนกันตอนบูต

ถ้าเคย `enable --system` ไว้แล้วมา `enable` (user) ทีหลัง **จะมี unit สองตัวของ slug เดียวกัน**
ทั้งคู่ enabled แล้วตอนบูตต่างคนต่าง start โมเดลเดียวกันบนพอร์ตเดียวกัน — ตัวหลังล้มเสมอ
`lmds disable` ปิดฝั่ง user ให้ได้โดยไม่ต้อง sudo แต่ฝั่ง system ต้องสั่งเอง:

```bash
systemctl --user disable --now lmds-<ชื่อ>          # ฝั่ง user
sudo systemctl disable --now lmds-<ชื่อ>            # ฝั่ง system (ต้อง sudo)
sudo rm -f /etc/systemd/system/lmds-<ชื่อ>.service && sudo systemctl daemon-reload
```

ตรวจว่าเหลือฝั่งเดียวจริง: `ls ~/.config/systemd/user/lmds-*.service /etc/systemd/system/lmds-*.service`

## 4.5 คุมหลายเครื่องจากเครื่องเดียว (fleet หลายเครื่อง)

หน้างานที่มีมากกว่า 1 เครื่อง — แทนที่จะ ssh ไล่ทีละตัว ให้เครื่องที่คุณนั่งอยู่ (**hub**) คุมเครื่องอื่นทั้งหมด

### ⚠️ ทุกเครื่องที่จะคุมต้องมี LMDS อยู่บนเครื่องนั้น

hub ไม่ได้ส่ง agent ไปรันบนเครื่องปลายทาง — มันเรียก `lmds agent info` **ที่ติดตั้งอยู่บนเครื่องนั้น**
ผ่าน SSH ("agent" ของระบบนี้คือตัวคำสั่ง `lmds` เอง ไม่ใช่โปรเซสที่รันค้าง) ฉะนั้น:

```bash
lmds node add 192.168.10.21 --user ops --install   # เพิ่ม + ติดตั้ง LMDS ให้เลยในคำสั่งเดียว
lmds node install spark2                           # ติดตั้ง/อัปเดตทีหลังก็ได้
```

`node install` **ส่งโค้ดของ hub ไปให้เครื่องนั้น** (git bundle ~2 MB ผ่าน scp แล้ว `git clone -b main`/`pull` จากไฟล์นั้น ·
origin ชี้กลับ GitHub เผื่อวันหน้า) แล้วรัน `install.sh` บนเครื่องนั้น — **เครื่องปลายทางไม่ต้องเข้าถึง GitHub และไม่ต้องมี deploy key**
· แคช bundle ต่อ commit (15 เครื่องกดพร้อมกัน pack ครั้งเดียว) · hub ที่ไม่ได้ติดตั้งจาก checkout จะถอยไป clone จาก GitHub ตามเดิม
· โฟลเดอร์เดิมบน node ที่ไม่ใช่ git (ติดตั้งแบบ copy) ถูกย้ายไป `.bak-<เวลา>` · checkout ที่แก้ไว้/แยกสายถูกเก็บที่ branch
`local-<เวลา>` + stash แล้วตามโค้ดของ hub — **node เป็นของ hub ไม่ใช่ที่พัฒนาโค้ด** · สรุปท้ายบอก "ตรง hub" / "ยังไม่ตรง hub"
โดยเทียบ commit แบบ prefix (git ย่อ 7 หรือ 8 ตัวต่างกันตามเครื่อง)
· **ข้ามขั้น Docker/NVIDIA toolkit** เพราะต้องใช้ `sudo` ซึ่งไม่มีคนกรอกรหัสผ่านผ่าน SSH — ใช้ `lmds node setup <ชื่อ> --with-prereq`
(ถามรหัสตอนนี้ ใช้ครั้งเดียว ส่งทาง stdin) หรือ *Add machine* บนหน้าเว็บที่ทำให้ครบในครั้งเดียว

### เตรียมเครื่องปลายทาง

1. sshd เปิดอยู่
2. user ที่จะใช้ **อยู่ในกลุ่ม `docker`** — **ไม่ต้องเป็น root**
3. `git` + `python3` (hub ส่ง git bundle ไปให้ clone · `python3-venv` ขาดก็ลงให้/ถอยไป `--without-pip` เอง)
4. Docker + NVIDIA Container Toolkit — `lmds node setup <ชื่อ> --with-prereq` จาก hub (ถามรหัส sudo ครั้งเดียว) หรือ `./install.sh` บนเครื่องนั้น
5. LMDS — ไม่ต้องทำเอง ใช้ `lmds node install` จาก hub ได้

### อัปเดตทั้งฟลีต

```bash
lmds node install --all        # hub ส่งโค้ดที่ตัวเองรันอยู่ไปทุกเครื่อง (~1.5 นาที/เครื่อง ไม่แตะ GitHub)
lmds node list                 # ป้าย ≠ hub เฉพาะเครื่องที่ commit ต่างจริง
```

บนหน้าเว็บ: ปุ่ม **Update** ที่แถบบน = `git pull --ff-only` บน hub → `install.sh` → restart service (รอให้ลายเซ็น process
เปลี่ยน ไม่ใช่ ping แล้ว reload) → อัปเดตทุก node ด้วยโค้ดจาก hub · ป้าย "มีอัปเดต" อ่านจาก checkout ที่ `install.sh` ประทับไว้
(ไม่ใช่ตำแหน่งโมดูลใน site-packages) · hub ที่ไม่ได้รันใต้ systemd ตอบ 409 ให้ `lmds web --restart` เอง

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

ทั้งสองตารางมีคอลัมน์ **IP ของเครื่อง** — IP ที่เครื่องนั้นรายงานว่าตัวเองถืออยู่ ซึ่งไม่ใช่
ที่อยู่ที่ hub ใช้ SSH (อันนั้นเป็นชื่อได้ เช่น `orb` หรือชื่อบน Tailscale) · ดูรายละเอียดและ
ที่มาที่ [NETWORK.md](NETWORK.md#เครื่องนั้นอยู่-ip-ไหน)

### สองคำสั่งที่ต้องแยกให้ออก

```bash
lmds node run spark2 doctor my-model          # สั่ง "คำสั่งของ lmds" บนเครื่องนั้น
lmds node ctl spark2 my-model prepare-runtime # สั่ง "สคริปต์ controller" ในตัว bundle
```

| | ใช้กับ |
|---|---|
| `node run` | `ps` `start` `stop` `restart` `logs` `doctor` `repair` `deploy` `scan` `remove --dry-run` `set` `version` |
| `node ctl` | `prepare-runtime` `download` `verify-files` `sync-worker` `verify-worker` `test-text` `test-tools` `test-reasoning` `test-vision` `test-embed` `parsers` `bench` `stress` `status` `props` `network-info` `client-config` `clear-fi-cache` `logs worker N` |
| `node clone` | ทำสำเนาโมเดลจากเครื่องหนึ่งไปอีกเครื่อง — ไม่โหลดจาก HF ใหม่ (`--from` `--to` `--start` `--dry-run`) |
| `node push` | ส่ง bundle จากเครื่องนี้ไปติดตั้ง (`--download` `--start` · stacked: เขียน cluster.env + pair + sync/verify ให้ก่อน start) |
| `cluster …` | `show` (= `node cluster`) · `write <slug> --head` · `pair <head> <worker…>` · `doctor <head> <worker> [--slug]` |

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

ค่าช่องหลักถูกจำไว้ต่อ (เครื่อง/โมเดล) **ในเบราว์เซอร์** ไม่ได้เขียนทับ bundle บนเครื่องปลายทาง —
ค่าที่ใช้ทดลอง (เลี่ยง port ชนกัน, ลด context ชั่วคราว) ไม่ควรกลายเป็นค่าถาวรของเครื่องนั้น
· **server ตรวจค่าเองทุกครั้ง** เพราะค่าพวกนี้ถูกต่อเป็นคำสั่งที่รันบนเครื่องอื่นผ่าน SSH
จะฝากการตรวจไว้กับ JS ในเบราว์เซอร์ไม่ได้ · ส่ง option ไปกับคำสั่งที่ไม่รับมัน (เช่น `doctor`)
จะได้ 400 ไม่ใช่เงียบ ๆ ทิ้งจนผู้ใช้เข้าใจว่าตั้งค่าแล้ว

**หมวด Advanced (0.6)** — parsers · engine env · extra args · image · model ID พับอยู่ใต้ช่องหลัก และ**ส่งไปกับคำสั่ง
เฉพาะตอนที่หมวดนั้นเปิดอยู่** (= ผู้ใช้ตั้งใจแก้) ไม่จำข้ามครั้ง — เคสจริง 2026-09-04: ค่าขั้นสูงที่จำไว้จากรอบก่อนติดไปกับ
start รอบถัดไปโดยไม่มีใครเห็น · ค่าที่ตั้งใจให้ติดถาวรกด **Save** (เขียน `bundle.env` = `lmds set`) · **Reset to bundle**
ลบค่าที่บันทึกไว้กลับไปใช้ของ bundle (= `lmds set --clear`) · ปุ่ม **Fill from model (parsers)** = `lmds set --auto`

**เมนูของโมเดลบนเครื่องอื่นทำได้เท่ากับโมเดลในเครื่องนี้** — เป็น controller ตัวเดียวกันและ
รับ env ชุดเดียวกัน จึงไม่มีเหตุผลให้ต่างกัน:

| กลุ่ม | มีอะไร |
|---|---|
| **ตั้งค่าตอน start** | `port` · `context` · `slots` · `bind` · `API key` · `gpu-util` (เฉพาะ vLLM) · Advanced: `tool parser` · `reasoning parser` · `engine env` · `extra args` · `image` |
| **ทดสอบ** | `test-text` · `test-vision` · `test-reasoning` · `test-tools` · `test-embed` · `parsers` · `bench` · `stress` · `client-config` · `network-info` · `status` · `props` |
| **stacked** | `prepare-runtime` · `sync-worker` · `verify-worker` · `clear-fi-cache` · `logs-worker` · ปุ่ม **Pair SSH** / **Doctor** ที่หัวกลุ่ม |
| **จัดการ** | `restart` · `doctor` · `logs` · `repair` · `verify-files` · `enable`/`disable` · `remove` · **Cancel** งานที่ค้าง |

ปุ่ม download บนการ์ด head ของโมเดล stacked ต่อ `sync-worker && verify-worker` ให้เป็นงานเดียว · งานที่ ssh ค้างยกเลิกได้
(`POST /api/jobs/{id}/cancel`) แล้วล็อก (เครื่อง, โมเดล) หลุด · คำสั่งสั้น (`stop` ฯลฯ) หมดเวลาที่ 120 วิ ไม่ยึด thread ของเว็บ
ครึ่งชั่วโมง

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

**`enable` จากหน้าเว็บไม่ต้อง sudo** — ค่าเริ่มต้นเป็น systemd *user* service (`~/.config/systemd/user/`) เพราะ hub
เรียกผ่าน SSH ซึ่งไม่มี tty ให้กรอกรหัสผ่าน · ต้องมี linger (`loginctl enable-linger` — *Add machine* / `node setup` ตั้งให้)
ไม่งั้นขึ้นตอน login ไม่ใช่ตอนบูต · `--system` (sudo) ใช้ได้เฉพาะจาก CLI บนเครื่องนั้น

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
lmds cluster show                               # = lmds node cluster
lmds cluster doctor spark-head spark2 --slug my-70b-model   # ทีละข้อ: ทะเบียน · ต่อถึง · GPU · ไซต์ · ฮาร์ดแวร์ · cluster IP · วง · สาย · ssh head→worker · ดิสก์ · cluster.env
lmds cluster pair spark-head spark2             # กุญแจ head→worker (เกิดบน head · เขียน ~/.ssh/config ของ head · ยืนยันด้วย ssh เปล่า ๆ)
lmds cluster write my-70b-model --head spark-head [--worker spark2 …] [--on spark-head]   # เขียน cluster.env ลง bundle
```

`cluster.env` มี `MASTER_IP` / `WORKER_IP` / `WORKER_IPS` / `SSH_USER` / `TRANSPORT_IP_*` / `NNODES` / `TENSOR_PARALLEL_SIZE` /
`NCCL_SOCKET_IFNAME` — stacked controller จะ source ไฟล์นี้**ก่อน default ทั้งหมด** แล้วไม่ถาม IP ตอน `start` อีก
(ตั้ง env จากภายนอกยังชนะไฟล์นี้เสมอ · ไม่มีไฟล์ + ไม่มี tty = หยุดพร้อมบอกคำสั่งข้างบน) · `cluster write` อ่าน `NNODES`
จาก controller ใน bundle แล้วตัดกลุ่มให้เหลือ head + worker ตามจำนวนนั้น (`--worker` เรียง rank ได้) — กลุ่ม 4 เครื่องกับ
bundle 2 เครื่องไม่ได้ `NNODES=4` ทับแผนอีก · ไม่พอ/เกิน = ปฏิเสธพร้อมบอกว่าต้อง target ไหน

> ถ้าไม่ตั้ง `NCCL_SOCKET_IFNAME` ไว้ NCCL จะเลือก interface เอง และมักได้เส้นบริหารจัดการที่ช้ากว่า —
> ยังรันได้แต่ช้าลงแบบหาสาเหตุยาก · **key ที่ hub ใช้เข้า head ไม่ใช่ key ของ head** — `node setup` ติดตั้งกุญแจของ hub
> ลงทุกเครื่อง แต่ controller stacked รันบน head แล้ว ssh ไป worker ด้วยกุญแจของ head เอง จึงต้อง `lmds cluster pair`
> (หน้าเว็บทำให้เองหลังเขียน cluster.env ตอน push จาก wizard)

รายละเอียดทั้งหมด: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md)

### จัดกลุ่มเครื่องตาม site (หลายสถานที่)

พอมีเครื่องหลายสิบตัวจากหลายสถานที่ ลิสต์เดียวยาวจนหาไม่เจอ — ติดป้าย **site** ให้แต่ละเครื่อง
แล้วคอนโซล/CLI จะจัดกลุ่มให้:

```bash
lmds node set msi-3 --site TKC        # ติดป้าย
lmds node set msi-3                    # ดูค่าปัจจุบัน (รวม site)
lmds node list                        # ตารางแยกตาม site
```

**ในหน้าเว็บ**: ปุ่ม **🏷 Site** บนการ์ดเปิดช่องกรอกที่มี**รายการ site ที่มีอยู่แล้วให้เลือก**
(datalist + ชิป) — node ตัวใหม่กดเลือกชื่อเดิมได้เลย ไม่ต้องพิมพ์เอง กันสะกดเพี้ยนจนกลายเป็นคนละกลุ่ม

การจัด **2 ชั้น ยุบ/กางได้ทั้งคู่** (5 site × 5 เครื่องจะได้ไม่ยาวเหยียด):

- **หัวไซต์** มีปุ่ม `+`/`−` — ยุบ/กางทั้งกลุ่ม · ไซต์ที่ยังไม่ติดป้ายไปรวมท้ายสุดชื่อ "ยังไม่จัดไซต์"
- **การ์ดเครื่อง** เริ่มแบบ**ย่อบรรทัดเดียว** (ชื่อ + ป้าย site + จุดสถานะ) กด `+` กางดูรายละเอียดครบ
- สถานะยุบ/ย่อจำไว้ในเบราว์เซอร์ ข้าม refresh 5 วิ · เปิดฟอร์ม/ผลคำสั่ง/กระโดดหาเครื่อง = กางให้เอง

> **site ไม่เกี่ยวกับ cluster โดยตั้งใจ** — เป็นคนละมิติ: site แค่จัดระเบียบสายตา ส่วนการ stacked
> ดูจาก GPU/สายเชื่อมจริง และทำได้เฉพาะเครื่องใน site เดียวกันอยู่แล้ว (คนละสถานที่ = คนละสาย →
> ไม่ผ่านเกณฑ์ลิงก์) · เปลี่ยน site จึงไม่กระทบการจับกลุ่ม cluster เลย

## 4.4.5 โมเดลนี้ทำอะไรได้บ้าง — ตรวจก่อนดาวน์โหลด

`lmds inspect <repo>` และหน้า Deploy บอกความสามารถ 6 อย่างตั้งแต่ก่อนโหลดไฟล์:

```
│ Tool calling   │ 🟡 chat template มีที่ทางสำหรับ tools              │
│ Vision         │ ✅ config.json มี vision_config (gemma4_vision)   │
│ Reasoning      │ 🟡 chat template มีร่องรอยของ thinking             │
│ System prompt  │ ✅ chat template รองรับ role 'system'             │
│ JSON mode      │ ⚙️ vLLM guided decoding · llama.cpp GBNF grammar  │
│ Streaming      │ ⚙️ ทำได้กับทุกโมเดล                                │
```

**🟡 ไม่ใช่การเลี่ยงตอบ** — มันคือ "โมเดลรับได้ แต่ยังต้องตั้งค่าและพิสูจน์ตอนรัน"
ซึ่งเป็นความจริงคนละอย่างกับ ✅ · chat template ที่รับ tool ได้ ไม่ได้แปลว่า
เซิร์ฟเวอร์จะแปลงคำตอบเป็น `tool_calls` ให้ ต้องมี `--tool-call-parser` ที่ตรง
ตระกูลด้วย · ยุบเป็นติ๊กเดียวเมื่อไหร่ คนก็ไปวางแผนงานบนของที่ยังไม่ได้เปิด

| เครื่องหมาย | แปลว่า |
|---|---|
| ✅ | ไฟล์ยืนยันชัด |
| 🟡 | มีทางเป็นไปได้ แต่ต้องตั้งค่าเพิ่มและพิสูจน์ตอนรัน |
| ❌ | ไฟล์บอกว่าไม่มี — เปิด parser ก็ไม่ช่วย |
| ⚙️ | เป็นความสามารถของ**เซิร์ฟเวอร์** ไม่ใช่ของโมเดล |

**อ่านจากอะไร** — `chat_template` เป็นหลักฐานหลัก เพราะมันกำหนดว่าโมเดลจะ*ถูกป้อน*
tool/system/thinking ยังไง · ส่วน vision อ่านจาก `vision_config` ใน config.json
หรือไฟล์ mmproj (GGUF)

### MTP draft head — เปิด speculative decoding ให้อัตโนมัติ

repo GGUF บางตัวแถมไฟล์ `mtp-*.gguf` มาด้วย (multi-token prediction draft head)
ผู้ทำ repo วัดได้ราว **+35% ถึง +53% ตอน generate โดย output เหมือนเดิมเป๊ะ** เพราะ
target model verify ทุก token ที่ draft เสนอ — ได้มาแต่ความเร็ว ไม่แลกคุณภาพ

ระบบจัดการให้เองทั้งหมด เหมือนที่ทำกับ mmproj: เจอไฟล์ `mtp-*.gguf` ใน repo →
ผนวกเข้า `MODEL_FILES` (download + verify SHA-256 ครบ) แล้ว emit
`--spec-draft-model ... --spec-type draft-mtp` ให้ใน controller

```bash
./<controller>.sh restart --no-mtp     # ปิด speculative decoding
./<controller>.sh restart --mtp FILE   # ใช้ draft head ตัวอื่น
```

vision ก็มีคู่เดียวกัน: `--no-mmproj` / `--mmproj FILE`

### เปลี่ยนชื่อโมเดล — ชื่อเดิมไม่หาย

ชื่อที่ client ส่งมาในฟิลด์ `model` เปลี่ยนได้เหมือนย้าย port:

```bash
./<controller>.sh restart --name my-gemma
```

ชื่อที่ตอน generate ตั้งไว้ถูกตรึงเป็น `DEFAULT_SERVED_MODEL_NAME` ในไฟล์ **override
ไม่ได้** เพราะมันคือหลักฐานว่าเดิมคืออะไร · เมื่อชื่อไม่ตรงกัน จะโชว์คู่กันทุกที่:

```
Model:     my-gemma (Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf)
           ↳ เปลี่ยนจากชื่อเดิม: gemma4-26b-uncensored
```

`server.meta` จดทั้งสองค่า (`model=` กับ `default_model=`) — `lmds ps` จึงบอกได้ด้วย

> **ทำไมถึงต้องมี** — LLM ตั้งชื่อไม่เหมือนกันทุกครั้งที่ generate · repo เดียวกัน
> deploy สามรอบได้ `gemma4-26b-a4b-qat-uncensored`, `gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp`
> และ `gemma4-26b-uncensored` — regenerate ทีเดียว client ที่ตั้งชื่อไว้เดิมพังหมด
> ตอนนี้ `--name` ตรึงชื่อให้เท่ากับของเดิมได้โดยไม่ต้องแก้ฝั่ง client

### คำสั่งหลัง start ตามเซิร์ฟเวอร์ที่รันอยู่จริง

`status` / `logs` / `network-info` / `client-config` / `test-*` / `stop` เป็นคนละ
process กับตัวที่ start จึงไม่รู้ว่าเซิร์ฟเวอร์ถูกสั่งด้วย flag อะไร — เดิมไปใช้ค่า
default ในไฟล์แล้วรายงานผิด (`start --port 8020` แล้ว `status` บอก
`API: not responding` เพราะไปถาม port 8000)

ตอนนี้อ่าน `server.meta` ก่อน **เฉพาะตอนที่เซิร์ฟเวอร์ยังรันอยู่จริง** · flag ที่ระบุ
เองชนะเสมอ · `start`/`restart` ไม่สืบทอด — ยึด default + flag ตามเดิม ไม่งั้นค่าเก่า
จะติดมาเงียบ ๆ ทุกครั้งที่ start

### MoE — แจ้งเหมือน vision

โมเดล MoE รายงาน **จำนวน expert ทั้งหมด กับที่เปิดต่อ token** ตั้งแต่ตอน `deploy`
(แถว Features) ไปจนถึงคอลัมน์ *รองรับ (support)* ของ `lmds list`:

```
รองรับ (support)
image, MoE 128e/8a, MTP
```

**ทำไมต้องเห็นทั้งสองค่า** — total บอกว่าต้องมีหน่วยความจำเท่าไร active บอกว่าจะได้
ความเร็วเท่าไร · gemma4 26B-A4B โหลด 15.6 GB เท่าเดิมทุก token แต่อ่านแค่ ~4B
ส่วน 31B dense อ่านครบ 31B — บน DGX Spark ที่คอขวดคือ bandwidth สองตัวนี้ต่างกัน
หลายเท่า เอา total params ไปเทียบกับ dense ตรง ๆ จึงให้ภาพที่ผิด

อ่านจาก `config.json` (`num_local_experts`/`n_routed_experts`/`num_experts` คู่กับ
`num_experts_per_tok`) หรือ GGUF metadata (`{arch}.expert_count` /
`{arch}.expert_used_count`) — โมเดล multimodal ซุกไว้ใต้ `text_config` มองแค่ชั้นบน
จะได้ None เงียบ ๆ แล้ว MoE กลายเป็น dense ในสายตาระบบ

วัดจริงบน DGX Spark (gemma4-26B-A4B Q4_K_M, ctx 65536, 3 รอบต่อโหมด):

| | TG เฉลี่ย | draft acceptance |
|---|---:|---:|
| MTP เปิด | **123.58 tok/s** | ~83% |
| MTP ปิด | 69.52 tok/s | — |
| | **1.78x** | |

สูงกว่าที่ repo เคลม (1.35x) เพราะ Spark คอขวดที่ memory bandwidth ไม่ใช่ compute —
speculative decoding จึงคุ้มกว่าบนเครื่องแบบนี้

> **ทำไมต้องบังคับจาก repo ไม่ให้ LLM ตัดสิน** — เคสจริง `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-*-MTP`
> LLM เสนอ `--mtp mtp-gemma-4-26B-it.gguf` ซึ่งผิดสองชั้น: llama.cpp ไม่มี flag ชื่อ `--mtp`
> (ของจริงคือ `--spec-draft-model` คู่กับ `--spec-type draft-mtp`) และชื่อไฟล์ตก `-A4B` ไป
> allowlist กักไว้ได้ถูกแล้ว แต่ผลลัพธ์คือไม่มีใครโหลดไฟล์ MTP เลย เสียความเร็วที่ repo
> ตั้งใจให้ไปเปล่า ๆ · ตอนนี้ชื่อไฟล์มาจากรายการไฟล์จริงใน repo เท่านั้น
>
> **llama.cpp ต้องใหม่พอ** — `--spec-type` เพิ่งมี ถ้า build เก่าจะขึ้น unknown option
> ตรวจด้วย `llama-server --help | grep spec-type`

**สิ่งที่ตอบจากไฟล์ไม่ได้** JSON mode กับ streaming เป็นของเซิร์ฟเวอร์ — vLLM และ
llama.cpp ทำได้กับทุกโมเดล การไปบอกว่า "โมเดลนี้ทำ JSON mode ไม่ได้" ผิดตั้งแต่
ตั้งคำถาม

### แต่ละ bundle รันไทม์ของตัวเอง — ตรึงที่ digest

controller ระบุ image ของตัวเองอยู่แล้ว แต่ระบุเป็น *tag* ซึ่งเคลื่อนที่ได้:
`vllm/vllm-openai:latest` วันนี้กับเดือนหน้าเป็นคนละ image · bundle ที่ทดสอบผ่าน
แล้วจึงกลายเป็นคนละรันไทม์ได้โดยไม่มีอะไรในไฟล์เปลี่ยนเลย ซึ่งเป็นอาการที่ไล่หา
สาเหตุยากที่สุด เพราะผู้ใช้ยืนยันว่าไม่ได้แก้อะไร — และเขาพูดถูก

ตอน generate ระบบถาม registry ว่า tag นั้นชี้ไป digest ไหน แล้วเขียน digest ลง
controller แทน:

```bash
# tag ที่ digest นี้มาจาก ณ วันที่ generate: vllm/vllm-openai:latest
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b…}"
```

**ผลที่ได้ตรงกับที่ต้องการพอดี** — bundle สองตัวที่ tag ชี้ digest เดียวกันคือ image
เดียวกัน ไม่กินดิสก์เพิ่มและไม่ต้องโหลดซ้ำ · จะแยกกันก็ต่อเมื่อรุ่นต่างกันจริง ซึ่ง
เป็นตอนที่ควรแยกพอดี

**ยังเปลี่ยนเองได้** — ตรึงไม่ใช่ล็อกตาย:

```bash
VLLM_IMAGE=vllm/vllm-openai:nightly ./xxx-single.sh restart
```

**ถ้าถาม registry ไม่ได้** (nvcr.io ต้องล็อกอิน · เครื่องไม่มีเน็ต · proxy บล็อก)
ก็ใช้ tag ตามเดิม — การห้าม deploy เพราะถาม registry ไม่ได้ แพงกว่าประโยชน์ที่ได้

### รันไทม์โหลดสถาปัตยกรรมนี้ได้ไหม

โมเดลที่ออกใหม่กว่า image เป็นเรื่องปกติ แต่จบด้วย container ที่ตายเงียบ ๆ หลังโหลด
weight มาแล้วหลายสิบกิกะ:

```
The checkpoint you are trying to load has model type `muse_glimmer`
but Transformers does not recognize this architecture.
```

`lmds doctor <slug>` และตัว `start` เองถามคำถามนี้ก่อนแล้ว ใช้เวลาไม่กี่วินาที
เทียบกับ health timeout ที่ตั้งไว้เป็นหลักสิบนาทีเพราะโมเดลใหญ่โหลดช้า:

```
│ ❌ │ architecture │ image นี้ไม่รู้จักสถาปัตยกรรม 'muse_glimmer'
│    │              │ (transformers 5.6.0) — start แล้ว container จะตาย
│    │              │ ทันทีที่โหลด config
```

ทั้งสองจุด**เงียบเมื่อไม่แน่ใจ** — ยังไม่มี config.json, image ยังไม่ได้ pull,
native mode หรือถาม image ไม่ได้ ก็ปล่อยผ่านให้ vLLM ตัดสินเอง · การบล็อกคนที่ยัง
พอทำงานได้ แย่กว่าปล่อยให้เจอ error จริงของมันเอง

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

### ดึงสูตรจากรีโป controller ของทีม

ทีมเก็บ controller ที่รันผ่านจริงไว้ในรีโป Git อยู่แล้ว การพิมพ์ค่าเดิมซ้ำลง catalog ของ LMDS
แปลว่าต้องแก้สองที่ทุกครั้ง แล้วสองที่จะหลุดกันในที่สุด — ดึงจากรีโปที่เป็นต้นทางจึงเป็นทาง
เดียวที่ทำให้ตรงกันเสมอ **แก้ที่รีโปแล้ว push จากนั้นสั่ง sync ที่ hub**

```bash
lmds recipes --sync                              # ดึงจากรีโปของทีม (ค่าเริ่มต้น) แล้วแสดงผล
lmds recipes --sync --repo <git url> --ref main  # รีโปอื่น/branch อื่น
```

**ในหน้าเว็บ**: *Library → Recipes* → ปุ่ม **Sync from GitHub** · แถวบนบอกว่าชุดนี้มาจากรีโปไหน
commit ไหน ดึงเมื่อไหร่

| เรื่อง | พฤติกรรม |
|---|---|
| อ่านอย่างเดียว | ดึงมาแล้ว**อ่านส่วนหัวของสคริปต์** (`MODEL_ID`/`HF_REPO`, `VLLM_IMAGE`, `MODEL_FEATURES` ฯลฯ) — **ไม่รัน controller** บน hub |
| ทับกัน | สูตรจากรีโปชนะสูตรที่ฝังมากับ LMDS เมื่อเป็นรุ่นเดียวกัน (รีโปคือต้นทางที่อัปเดตบ่อยกว่า) |
| ไฟล์ที่อ่านไม่ได้ | รายงานว่าข้ามอะไรไปเพราะอะไร ไม่หายเงียบ — เช่น controller ที่คุมหลายโมเดลในไฟล์เดียว |
| single vs stacked | รุ่นเดียวกันที่มีทั้งสองแบบ ใช้ตัว single เพราะ LMDS เลือก topology เองจากเครื่องที่มี |
| context | ไม่ดึงมาจาก controller เหมือนเดิม — ต้องมาจากเครื่องเป้าหมาย ไม่ใช่เครื่องที่เคยรัน |

สำเนารีโปอยู่ที่ `~/.config/lmds/controllers/<ชื่อรีโป>` (เป็นแคช ลบทิ้งได้) · สูตรที่ดึงมาแล้ว
อยู่ใน `~/.config/lmds/recipes-synced.yaml`

### ส่งสูตรที่รันผ่านแล้วขึ้นคลัง — `lmds recipes --publish`

```bash
lmds recipes --publish <slug> --features tools,vision,reasoning   # ระบุที่วัดได้จริง ไม่ใช่ที่ profile เดา
```

**ตั้งแต่ 0.5.1 publish พับค่าที่ `lmds set` ไว้ลง header ให้เอง** — image ที่มี kernel ตรงรุ่น,
`--tool-parser` / `--reasoning-parser`, `--engine-env`, และ `--extra-args` (ลงที่
`EXTRA_SERVE_ARGS_DEFAULT='…'` แบบ single quote เพราะ JSON มี `}`) · เครื่องที่ sync ไปจึงได้สูตรที่
start ขึ้นจริง ไม่ใช่ค่าเดาของ plan ที่เคยล้ม

ค่าของเครื่อง (port, context, gpu-util, slots, bind, ชื่อที่เสิร์ฟ) **ไม่พับโดยเจตนา** — เครื่องปลายทาง
fit ใหม่ตามหน่วยความจำของตัวเอง · `PROFILE.yaml` ที่ไปด้วยมี `overrides:` ให้คน review เห็นว่า
ค่าไหนต่างจาก plan · controller ที่ generate ก่อน 0.5.1 ไม่มีบรรทัดรองรับ → CLI เตือน "พับไม่ได้"
ให้ `lmds rebuild <slug>` ก่อนแล้ว publish ใหม่

ปลายทางตั้งที่ `recipes.publish_repo` ใน config (ว่าง = local store ในเครื่อง ไม่แตะรีโปของทีม)

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

## 4.7 โมเดลที่รันอยู่ก่อนแล้ว — `lmds adopt`

`lmds scan` บอกว่ามี **ไฟล์** อะไรบนดิสก์ · หัวข้อนี้คือกรณีที่มี **เซิร์ฟเวอร์รันอยู่แล้ว**
ตั้งแต่ก่อนติดตั้ง LMDS

`lmds ps` มองเห็นมันอยู่แล้วและติดป้าย **`ไม่ลงทะเบียน`** — stop/logs ได้ แต่ทำอย่างอื่น
ไม่ได้เลยเพราะไม่มี controller · `lmds adopt` อ่านคำสั่งที่มันรันอยู่**จริง** แล้วเขียนเป็น
controller ที่รันซ้ำได้เป๊ะ **ของที่รันอยู่ตอนนั้นไม่ถูกแตะต้อง**

```bash
lmds adopt <container>              # รันด้วย Docker
lmds adopt --port 8080              # รันตรง ๆ (llama-server, vLLM ที่ไม่ได้อยู่ใน container)
lmds adopt --pid 122081             # ระบุ PID เอง
lmds adopt --port 8080 --take-over  # + ปิด systemd unit เดิมให้ LMDS คุมแทน
```

**อ่านจาก `/proc/<pid>/cmdline` ไม่ได้เดา** — flag ทุกตัวที่ของเดิมใช้ถูกยกมาครบ
(`-ngl 99 -c 65536 -ts 1,1,1 -sm layer -fa on -ctk q8_0 -ctv q8_0` ฯลฯ) ตกตัวเดียว
คือได้คนละพฤติกรรม

> **จงใจไม่อ่าน `/proc/<pid>/environ`** — API key ของ backend อยู่ในนั้น เขียนลง bundle
> คือทำให้ทุกคนที่อ่านไฟล์ได้เห็น secret · `cmdline` พอสำหรับรันซ้ำอยู่แล้ว ส่วน env ที่
> จำเป็นจริงตั้งเองใน `bundle.env` ซึ่งเป็นที่ของมัน

### unit เดิมจะแย่ง port กลับ

process ที่รันใต้ systemd unit ที่ลูกค้าเขียนเองมักตั้ง `Restart=always` — LMDS stop
เมื่อไหร่มันเด้งกลับมายึด port ทันที · `adopt` จึงบันทึกชื่อ unit ไว้ แล้ว:

- `start` **ปฏิเสธพร้อมบอกคำสั่งที่ต้องใช้** แทนที่จะปล่อยให้ชนกันเองแล้วงง
- `status` เตือนว่า *"ตัวที่ตอบอาจเป็นของ unit นั้น ไม่ใช่ของ LMDS"*
- `--take-over` สั่ง `systemctl disable --now` ให้ — **เฉพาะเมื่อสั่งเท่านั้น ไม่ทำเอง**

```text
ERROR: llama-qwen.service ยังรันอยู่และถือ port 8080 —
  หยุดก่อน: sudo systemctl disable --now llama-qwen.service
```

> โมเดลใหญ่ถือ VRAM หลายสิบ GB **หยุดช้า** — `disable --now` อาจใช้เวลาเป็นนาที
> ถ้า ssh หลุดกลางทางจะเหลือสถานะ *disabled แต่ยัง active* · แยกเป็น `disable` แล้ว
> `stop` ทีหลังจะคุมได้ง่ายกว่า

### ทำอะไรได้บ้างหลังรับเข้าระบบ

`start` · `stop` · `restart` · `status` · `logs` · `test-text` · `client-config` ·
`network-info` · `doctor` · `enable` (autostart)

**ไม่มี `download` / `verify-files`** — weight เป็น path ที่คุณจัดการเอง LMDS ไม่ได้เป็น
คนโหลดมา จึงไม่มีอะไรให้โหลดหรือตรวจ · *คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่
แพงกว่าการไม่มีคำสั่งนั้น* · `doctor` ก็ตรวจไฟล์ตรง path จริงและไม่แนะ `repair`

**ในหน้าเว็บ**: การ์ดที่ยังไม่มี controller มีปุ่ม **รับเข้าระบบ** อยู่ตรงที่เคยเป็นปุ่ม Start
ที่กดไม่ได้ · ถ้ามี unit เจ้าของ หน้าเว็บเตือนพร้อมคำสั่ง `disable` ให้ copy ไปใช้

### เคสจริง

เครื่องลูกค้า 3× RTX 3060 มี `llama-server` รัน Qwen3.6-35B-A3B ใต้ unit ที่เขียนเอง
มา 25 วัน · `lmds ps` ขึ้น `● running ⚠ ไม่ลงทะเบียน` แล้วจบแค่นั้น

```bash
lmds adopt --port 8080 --slug qwen35-a3b-opus
sudo systemctl disable --now llama-qwen.service
lmds start qwen35-a3b-opus
lmds enable qwen35-a3b-opus          # ให้กลับมาเองหลัง reboot
```

ผลลัพธ์: bundle เต็มรูปแบบ **โดยไม่ต้อง redeploy หรือโหลด weight ใหม่สักไบต์**

> **ตรวจ binary ที่ unit เดิมใช้ด้วย** — เครื่องนั้นมี `llama-server` สองตัวคนละรุ่นใน
> โฟลเดอร์เดียวกัน (`./llama-server` กับ `build/bin/llama-server`) unit สั่ง `exec ./llama-server`
> จึงรันตัวเก่ากว่าที่ทุกคนคิดอยู่หลายเดือน · `lmds adopt` ยกมาตามที่มันรันจริง ซึ่งถูกต้อง
> แต่ถ้าจะอัป llama.cpp ต้อง copy ทับตัวที่ unit ใช้จริง ไม่ใช่แค่ `cmake --build`

## 4.9 โมเดล embedding (`/v1/embeddings`)

ระบบดูจาก repo เองว่าเป็นโมเดล embedding (pipeline_tag `feature-extraction`/`sentence-similarity`, tag
`sentence-transformers`, หรือชื่อมีคำว่า embed) แล้ววางแผนเป็น embedding ให้ — ไม่มี chat, tool calling, reasoning

```bash
lmds deploy Qwen/Qwen3-Embedding-0.6B --no-llm            # safetensors → vLLM --runner pooling --convert embed
lmds deploy VesNFF/Qwen3-VL-Embedding-8B-GGUF --no-llm    # GGUF → llama.cpp --embedding --pooling last
lmds deploy <repo> --task embed                           # เดาผิด (ชื่อไม่บอก) → บังคับเอง · --task generate กลับด้าน
./<slug>-single.sh test-embed                             # ยิง /v1/embeddings 3 ประโยค เช็ค cosine ข้ามภาษา
POOLING=mean EMBED_UBATCH=4096 ./<slug>-single.sh restart # llama.cpp: เปลี่ยนวิธี pool / ขนาด batch
```

| | llama.cpp (GGUF) | vLLM (safetensors) |
|---|---|---|
| flag ที่ controller ใส่ | `--embedding --pooling <p> --batch-size N --ubatch-size N` | `--runner pooling --convert embed` |
| pooling ตั้งต้น | Qwen → `last` · BERT/XLM-R/bge/e5/gte → `cls` · อื่น `mean` | vLLM อ่านจาก config ของโมเดลเอง |
| ทดสอบ | `test-embed` | `test-embed` |
| client | `client-config` → `"task": "embed"`, `"endpoint": "/v1/embeddings"`, `max_input_tokens` = context ต่อ slot ทั้งก้อน (ไม่มี output token จึงไม่มี `max_output_tokens`), `pooling` · เรียก `POST /v1/embeddings` ด้วย `model` = served name | `"task": "embed"` · budget แบบเดียวกับ chat (vLLM ไม่แบ่ง slot) |

> **`test-embed` ขึ้น WARN** = คู่ประโยคความหมายเดียวกัน (ไทย↔อังกฤษ) ได้คะแนนไม่สูงกว่าประโยคที่ไม่เกี่ยวกัน
> มักเป็น pooling ผิดตระกูล — ลอง `POOLING=mean` หรือ `cls` แล้ว restart (vector ที่ pooling ผิดดูปกติทุกอย่างยกเว้นข้อนี้)
> · stacked และ SGLang ไม่รองรับ — ทั้ง analyze บนหน้าเว็บ (422) และ renderer ปฏิเสธ (SGLang ที่ขอมาถอยเป็น vLLM) เพราะโมเดล
> embed ≤ 8B ลงเครื่องเดียวเสมอ · แผนจาก LLM ตั้ง task เองไม่ได้ (harden บังคับจาก repo) · หน้าเว็บ/CLI ติดป้าย
> "embedding (last)" จาก `MODEL_PROFILE` · `lmds bench` วัดเฉพาะโมเดล chat — โมเดล embed ใช้ `test-embed`
> · แนะนำโมเดลไทย: Qwen3-Embedding-0.6B/4B (ทั่วไป) · bge-m3 (hybrid + reranker คู่กัน) · รันจริงแล้ว:
> `VesNFF/Qwen3-VL-Embedding-8B-GGUF` f16 บน dgx-spark03 (2026-09-04)

## 5. หน้าเว็บ (ทางเลือก) — `lmds web`

สำหรับคนที่ไม่ถนัด CLI หรืออยากให้ทีมดูสถานะได้โดยไม่ต้อง ssh · **หน้าเว็บเป็นภาษาอังกฤษ**
(ตัว CLI ยังเป็นไทย)

```bash
lmds web --enable --bind 0.0.0.0  # แนะนำ: systemd user service — ขึ้นเองหลังรีบูต ฟื้นเองถ้าตาย
lmds web --disable                # เลิกให้ขึ้นเอง
lmds web                          # http://127.0.0.1:8600 — เครื่องนี้เท่านั้น (รันค้าง terminal)
lmds web --bind 0.0.0.0           # ให้ทั้งวง network เข้าได้ — ถาม token ก่อน (Enter = สุ่มให้)
lmds web --background             # รันเบื้องหลัง terminal ว่างใช้ CLI ต่อได้
lmds web --status                 # ลืมลิงก์/token? ถามตัวที่รันอยู่ได้เลย
lmds web --restart -b             # เปิดใหม่ (ลิงก์เดิมยังใช้ได้)
lmds web --stop                   # หยุดตัวที่รันเบื้องหลัง
lmds web -b --new-token           # เปลี่ยน token (ลิงก์เดิมใช้ไม่ได้ทันที)
```

### หน้าตา 0.6 — เมนูซ้าย รายละเอียดตรงกลาง

| ที่ | มีอะไร |
|---|---|
| **แถบซ้าย** | Overview · This machine (hub) · All machines · ต้นไม้ **ไซต์ → เครื่อง** (จุดสถานะ + แถบ % หน่วยความจำ) · Library: Models (ทุก bundle ทุกเครื่อง คลิกแถว → หน้าเครื่อง + เปิดแผงของโมเดลนั้น) · Scores · Recipes · Weights · Settings |
| **แถบบน** | breadcrumb · ค้นเครื่อง/โมเดล (Enter = กระโดดไป) · **Deploy to one machine** · **Update** · ขนาดตัวอักษร/ธีม · Sign out |
| **Overview** | KPI ฟลีต · แถบหน่วยความจำ GPU ต่อเครื่องเรียงจากแน่นสุด (คลิกชื่อ → หน้าเครื่อง) · โดนัท llama.cpp/vLLM ที่รันอยู่ · **Needs attention** จากข้อมูลจริง: เครื่องต่อไม่ได้ · เกิน 90% · พอร์ตซ้อนบนเครื่องเดียว · commit ไม่ตรง hub · ตารางไซต์ |
| **หน้าเครื่อง** | การ์ดเดิมของเครื่องนั้นกางเต็ม (เกจ · GPU · โมเดล · แผง settings + บรรทัดคำนวณแรม) — ปุ่มทุกปุ่มเหมือนเดิม · เครื่อง/ไซต์ที่ไม่มีแล้ว (bookmark เก่า) บอกตรง ๆ พร้อมลิงก์กลับ |
| **route** | `#/overview` `#/node/<ชื่อ>` `#/site/<ไซต์>` `#/models` … — ลิงก์ตรง · back/forward · reload อยู่หน้าเดิม · route ชนะการยุบไซต์ (กดเครื่องใน rail แล้วหน้าไม่ว่าง) |

ทุกอย่างคำนวณจากแคชเดียวกับการ์ด ไม่ยิง SSH เพิ่ม · SSE หลุดแล้วถอยไป poll แคช `/api/nodes/<n>/inventory` ทุก 5 วิ ·
เมนู/ข้อความบนหน้าเว็บเป็นอังกฤษทั้งหมด (comment ในโค้ดยังไทย) · ฟอนต์ Geist / Geist Mono อยู่ในแพ็กเกจ hub เสิร์ฟเองที่
`GET /fonts/<ชื่อ>` (รับเฉพาะชื่อในรายการ · ไม่ต้องใช้ token) ตกไปฟอนต์ระบบสำหรับภาษาไทย — ยังไม่โหลดอะไรจากเน็ต

### บนการ์ดโมเดลในคอนโซล

**ป้ายบอกความสามารถ** — อ่านจาก `MODEL_PROFILE.yaml` ของ bundle ไม่ใช่การเดา:

| ป้าย | หมายถึง |
|---|---|
| 👁 `vision` | มีไฟล์ mmproj — รับภาพได้ |
| ⬚ `MoE 128e/8a` | 128 expert เปิด 8 ต่อ token · hover เพื่อดูว่าทำไมสองค่านี้ต่างกัน |
| ◔ `MTP` | มี draft head — เร็วขึ้นโดย output เท่าเดิม |

**`model ID`** คือชื่อที่ client ใส่ในฟิลด์ `model` — **ไม่เท่ากับ slug** ที่เป็นหัวการ์ด
เปลี่ยนได้ในเมนู ⋯ แล้วชื่อเดิมจะติดอยู่ข้าง ๆ (`↳ เดิม: …`) กันลืมว่าเดิมคืออะไร ·
ช่องกรอกใช้ชื่อเดิมเป็น placeholder จะได้รู้ว่ากำลังทับค่าอะไรอยู่

**หมวด Model features** ในเมนู ⋯ — ติ๊กเปิด/ปิด `vision` กับ `MTP` โดยไม่ต้อง deploy ใหม่
มีผลตอน start/restart ครั้งถัดไป · หมวดนี้โผล่เฉพาะโมเดลที่มีไฟล์จริง

> **ป้ายไม่ขึ้นบนเครื่องอื่น?** payload มาจาก `lmds agent info` ที่รันบน**เครื่องปลายทาง**
> ไม่ใช่ hub — เครื่องที่ยังเป็น lmds เวอร์ชันเก่าจะส่งข้อมูลไม่ครบ อัปเดตด้วย
> `lmds node install --all` (hub ส่งโค้ดของตัวเองไปให้ — เครื่องนั้นไม่ต้องเข้า GitHub) แล้ว refresh หน้าเว็บ

### เลือกรันไทม์เอง — vLLM หรือ SGLang

safetensors เสิร์ฟได้ทั้งสองตัว การเดาจากชนิดไฟล์จึงเป็นแค่ค่าตั้งต้น:

```bash
lmds deploy <repo> --engine sglang       # หรือ vllm · ใช้ได้กับ plan/generate ด้วย
```

| | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| ไฟล์ที่อ่านได้ | safetensors | safetensors | **GGUF** |
| stacked หลายเครื่อง | ✅ | ยังไม่รองรับใน LMDS | ❌ |
| เลือกได้ด้วย `--engine` | ✅ | ✅ | บังคับอัตโนมัติเมื่อเป็น GGUF |

**GGUF ไม่มีทางกลายเป็น SGLang** ต่อให้สั่ง — SGLang อ่านไฟล์นั้นไม่ได้ ยอมตามคำขอ
คือส่ง bundle ที่ start ไม่ขึ้นให้

ทำไมต้องมี SGLang: checkpoint NVFP4 บางตระกูล (เช่น `sparkarena/Minimax-M3-*`)
calibrate ด้วย w1/w3 scale ซึ่งรันถูกต้องเฉพาะบน SGLang · ไม่มีตัวนี้ก็ต้องยกทั้งตระกูล
ออกจากระบบ

**ในหน้าเว็บ**: deploy wizard มีช่อง **Engine** อยู่ข้าง Target — Auto / vLLM / SGLang
(llama.cpp ไม่อยู่ในรายการเพราะ GGUF บังคับใช้มันอยู่แล้ว ใส่ไปก็เลือกแล้วไม่มีผล)

> ช่อง **Plan without an LLM** ที่อยู่ใต้ลงมาเป็นคนละเรื่อง — มันพูดถึง*ผู้วางแผน*
> ไม่ใช่ตัวที่เสิร์ฟโมเดล (เดิมเขียนว่า "Skip the LLM" แล้วมีคนอ่านว่าเป็นการเลือก llama.cpp)

**ชื่อ knob เหมือนกันทุก engine** แม้ SGLang จะเรียกธงคนละชื่อ — `--context`,
`--gpu-util`, `MAX_MODEL_LEN`, `bundle.env` ใช้ชื่อเดิมหมด การแปลงเกิดที่จุดเดียว
ตอนประกอบคำสั่งส่งให้ engine:

| knob ของ controller | vLLM | SGLang |
|---|---|---|
| `MAX_MODEL_LEN` | `--max-model-len` | `--context-length` |
| `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | `--mem-fraction-static` |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `--max-running-requests` |
| `API_KEY` | env `VLLM_API_KEY` | ธง `--api-key` |
| tool calling | `--enable-auto-tool-choice` + parser | parser อย่างเดียว |

image ตั้งต้นบน DGX Spark คือ `nvcr.io/nvidia/sglang:26.02-py3` (build ที่มี kernel
ของ SM121) · เปลี่ยนได้ด้วย `SGLANG_IMAGE=<image> ./controller start`

### context ควรตั้งเท่าไร — ถามได้

`lmds inspect` บอกได้อยู่แล้วว่า context สูงสุดเท่าไร แต่ค่านั้นคือค่าที่ **คนเดียว**
ใช้แล้วเต็มพอดี — ตั้งตามนั้นแล้วคนที่สองต้องรอคิว โดยไม่มีอะไรบอก

```bash
lmds inspect <repo> --target dgx-spark-stacked --context 262144
```

ได้ตารางว่าแต่ละขั้นรับได้กี่คนพร้อมกัน แล้วตามด้วยข้อควรระวังของค่าที่ถามมา

| context | KV ต่อคน | พร้อมกัน |
|---:|---:|---:|
| 65,536 | 7.5 GB | 7.0 |
| 131,072 | 15.0 GB | 3.5 |
| 262,144 | 30.0 GB | 1.8 |

> • ใส่ได้ แต่ได้ 1.8 คนพร้อมกัน — หนึ่งคำสนทนากิน KV pool เกือบหมด
> • เปลี่ยน KV เป็น fp8 (`--kv-cache-dtype fp8_e5m2`) → KV 30.0 GB เหลือ 15.0 GB · พร้อมกันจาก 1.8 เป็น 3.5 คน

**ลด context ครึ่งหนึ่ง กับ ลด KV ครึ่งหนึ่ง ให้ผลเท่ากัน** — เลือกได้ว่าจะยอมเสียอะไร
fp8 เป็นสวิตช์ตอนรัน ไม่ต้อง quantize checkpoint ใหม่

เพิ่ม `--kv-dtype fp8` เพื่อให้ทั้งตารางคิดที่ fp8 ตั้งแต่แรก · `--json` ได้ทั้งก้อนไปใช้ต่อ

ผู้ช่วยในหน้าเว็บรู้กติกาชุดนี้ด้วย — ถามเป็นภาษาคนได้ แต่มัน**ไม่คิดเลขเอง** จะชี้ให้
มารันคำสั่งข้างบน เพราะเลขที่ LLM คูณเองในหัวผิดแบบดูน่าเชื่อ ซึ่งแย่กว่าตอบว่าไม่รู้

### ขนาดตัวอักษรและธีม

แถบบนมีปุ่มสองใบติดกัน กดแล้วเห็นผลทันทีและเบราว์เซอร์จำไว้ให้ (แยกกันต่อเครื่อง
ที่เปิด — ตั้งที่โน้ตบุ๊กของคุณไม่ไปเปลี่ยนจอในห้องเครื่อง)

| ปุ่ม | ทำอะไร |
|---|---|
| **Aa** | ขนาดตัวอักษรทั้งหน้า วนสี่ระดับ **S → M → L → XL** · ค่าตั้งต้นคือ M |
| ☀︎ / ☾ | ธีม สว่าง → มืด → ตามเครื่อง |

ทั้งหน้าคูณจากตัวแปร CSS ตัวเดียว (`--fs`) ตัวหนังสือ ป้าย ปุ่ม และช่องกรอกจึงโตขึ้น
พร้อมกันทั้งชุด ไม่ใช่ตัวหนังสือโตแต่ปุ่มเท่าเดิมจนล้นกรอบ

> ทำไมถึงให้ปรับเอง — เอาไปติดตั้งหลายที่แล้วได้ผลไม่เหมือนกัน จอ 4K ในห้องเครื่องกับ
> โน้ตบุ๊กบนโต๊ะต้องการคนละขนาด เลือกเลขกลางเลขเดียวยังไงก็มีที่ที่อ่านไม่สบาย

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
| **download** | โหลด weight แล้ว **รัน `verify-files` ต่อให้อัตโนมัติ** พร้อม log สด — ปุ่มเปลี่ยนเป็น `start` เองเมื่อครบ (stacked: ต่อ `sync-worker && verify-worker` ด้วย) |
| **start / stop / restart** | ใช้ตัวเลือกที่ตั้งไว้ในแท็บ manage (Advanced ส่งเฉพาะตอนเปิด) |
| **tests** | `test-text` · `test-vision` · `test-reasoning` · `test-tools` · `test-embed` · `parsers` · `bench` · `stress` · `client-config` · `network-info` · `status` · `props` |
| **manage** | port / context / slots / bind / API key / gpu-util · Advanced (parsers · engine env · extra args · image) · Save / Reset to bundle · autostart · คำสั่ง stacked (+ `logs-worker`) · repair · remove · Copy to another machine · Send to a serving machine |
| **doctor** | ผลเดียวกับ `lmds doctor` พร้อมคำสั่งแก้ |
| **logs** | log ล่าสุด 300 บรรทัด |

**เครื่องอื่นในทะเบียน** — หน้า *All machines* จัดกลุ่มตามไซต์ (หัวไซต์ยุบ/กางได้ · ลากการ์ดจัดลำดับ) หรือกดชื่อเครื่องใน
rail เพื่อเปิดหน้าเครื่องนั้นเต็มจอ · ในการ์ดมีเกจทรัพยากร, การ์ด GPU, แถบ cluster (ดู §4.5) และรายชื่อโมเดลพร้อมปุ่ม **⋯**
ต่อโมเดล · เครื่องที่ต่อไม่ได้ขึ้นเป็นแถบเหนือการ์ด ไม่ล้างการ์ดทิ้ง · โมเดล stacked ขึ้นทั้งการ์ด head ("stacked head · worker
<ชื่อ>") และการ์ด worker ("stacked worker of <head>" — ไม่มีปุ่ม)

> **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่านจาก dispatch table ของสคริปต์เอง
> bundle ที่สร้างก่อนมีคำสั่งใหม่ (เช่น `test-vision`) จะไม่มีปุ่มนั้น พร้อมบอกว่าต้อง deploy ใหม่

ปุ่มบนขวา **Deploy to one machine** = deploy ลง*เครื่องเดียว* · ส่วน **Deploy stacked to this group** ที่หัวกลุ่ม cluster
(หน้า All machines / site) = deploy แบบ *stacked* คือโมเดลเดียวแบ่งลง head + worker (TP=2) ได้ KV cache/context/จำนวนคนพร้อมกันมากกว่า
เครื่องเดียว — ปุ่มหลังเปิด wizard ตัวเดียวกันโดยตั้ง target เป็น `dgx-spark-stacked` และเลือก head/worker ให้แล้ว

ปุ่ม **Deploy to one machine** ทำ wizard ครบ flow: วางลิงก์ → เลือกเครื่องปลายทาง (**Run on**) / กลุ่ม stacked / target preset →
วิเคราะห์ → เลือกไฟล์ GGUF / ใส่ HF token ถ้าจำเป็น → ดูแผน + ปรับ context / อนุมัติ flag → สร้าง bundle ผ่าน quality gates → ZIP
→ push ไปเครื่องนั้น · เลือกเครื่องในฟลีตแต่ไม่เลือก preset = เดา preset จาก GPU ที่ refresher เห็นของเครื่องนั้น (ไม่ใช่ฮาร์ดแวร์ของ hub)
· **พอร์ต**: analyze เลือกพอร์ตว่างตัวแรกจาก inventory ของเครื่องปลายทาง (ทุก bundle + container นอกระบบ · stacked ดูทั้ง head
และ worker) เขียนลง `bundle.env` — bundle ใหม่ไม่ได้ 8000 ซ้ำกับตัวเดิมอีก แก้เองได้ในช่อง port · คู่ stacked ที่เป็นไปไม่ได้
(ไม่มี worker · worker = head · ไม่มี cluster IP · คนละไซต์ · GGUF/SGLang/embedding) ถูกปฏิเสธก่อนแตะ Hugging Face
(422 `{kind:"cluster"}`) · repo gated บอกวิธีใส่ token ตรง ๆ · analyze repo ที่ไฟล์อยู่บน Xet รอ byte แรกได้ถึง 120 วิ

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

`lmds config set-key` (API key ของ provider — หน้า Settings ตั้ง provider/base URL ได้ · HF token ใส่ได้ในกล่อง deploy) ·
`lmds enable --system` (system service ต้อง sudo — ค่าเริ่มต้น user service กดจากหน้าเว็บได้) · `lmds adopt --port/--pid`
(process ที่ไม่ได้อยู่ใน container — หน้าเว็บรับได้เฉพาะ container) · `lmds recipes --publish` · `lmds cluster write --nnodes`
· `lmds web --enable/--new-token`

## 5.5 พิสูจน์ว่ามันรันได้จริง — `lmds smoke`

quality gates ทั้ง 12 ด่านตรวจได้แค่ว่า **สคริปต์ถูกต้อง** ไม่ได้บอกว่ารันแล้วได้คำตอบจริง

```bash
lmds smoke <ชื่อ>                    # บนเครื่องนี้
lmds smoke <ชื่อ> --on spark-head    # บนเครื่องอื่นในทะเบียน
lmds smoke <ชื่อ> --skip-download    # ไฟล์ครบแล้ว ข้ามขั้นโหลด
lmds smoke <ชื่อ> --keep             # ไม่ต้อง stop ตอนจบ
```

ทำตามลำดับ `download → verify-files → start → test-text → stop` · **ล้มขั้นไหนหยุดตรงนั้น**
(verify ไฟล์ที่โหลดไม่จบ หรือ test-text กับ server ที่ยังไม่ขึ้น ไม่มีความหมาย) พร้อมบอกว่า
ขั้นไหนและ log ท้าย · **หยุด server เสมอแม้ล้มกลางทาง** เพราะทิ้งของค้างไว้คือทำให้เครื่อง
สกปรกกว่าเดิม · exit 0 ผ่านหมด · 2 ล้มบางขั้น

> **ทำไมต้องมี** — บั๊กที่เจ็บที่สุดทุกตัวของรุ่น 0.2.0 ผ่าน gate แบบ static ทั้งหมดแล้วไปตาย
> ตอนรันจริง: image ที่ tag ไม่มีอยู่จริง · head container ที่ไม่เคย start · ชุดทดสอบที่ไป
> ให้คะแนนเซิร์ฟเวอร์ของโมเดลอื่น

## 5.6 bundle เก่าใช้ไม่ได้ — `lmds rebuild`

ใช้เมื่อ bundle เสียเพราะสิ่งที่อยู่**นอกเหนือค่าที่ตั้ง** เช่น image ที่ tag ถูกถอนไปแล้ว
หรือ template รุ่นใหม่มีตัวกันพลาดที่ของเก่าไม่มี

```bash
lmds rebuild <ชื่อ>
```

เอาค่าที่เคยตัดสินใจไว้กลับมา (context, flags, target, GGUF ที่เลือก) จาก `MODEL_PROFILE.yaml`
ส่วนที่ระบบเป็นเจ้าของคำนวณใหม่ตามตรรกะปัจจุบัน (image, ตัวกันพลาดในสคริปต์)
· **ไม่เรียก LLM ซ้ำ** เพราะแผนเดิมถูกตรวจและอนุมัติไปแล้ว · บอกตรง ๆ ว่า image เปลี่ยนเป็นอะไร

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
| `ไฟล์ใหญ่เกินจริง` ตอน verify-files | resume ทับของที่ค้างจากรอบก่อน | ลบไฟล์นั้นแล้ว `download` ใหม่ — resume ซ้ำไม่ช่วย มีแต่ทำให้ใหญ่ขึ้น |
| `หลุดกลางคัน x/y ไบต์ — resume ต่อ` ตอน download | CDN ตัดสตรีมกลางไฟล์ใหญ่ (`curl: (92) ... CANCEL`) | **ไม่ต้องทำอะไร** — ตั้งแต่รุ่นนี้ `download` วน resume ต่อเองจนขนาดครบ |
| `resume แล้วไม่คืบหน้าเลย` ตอน download | เน็ตหลุด หรือต้นทางไม่รองรับ resume | ตรวจเน็ต/พร็อกซีแล้วรัน `download` ใหม่ — ของเดิมไม่ถูกลบ ต่อจากเดิมได้ |
| `ลองต่อ N รอบแล้วยังไม่ครบ` ตอน download | ต้นทางส่ง body สั้นกว่าจริงซ้ำ ๆ (มัก proxy คั่นกลาง) | สั่ง `download` ใหม่ (ตั้งรอบเองได้ `FETCH_MAX_ATTEMPTS=40`) หรือเลี่ยง proxy |
| port ชน | มี service อื่นใช้ :8000 | `--port 8001` หรือหยุด service เดิม (`docker ps` ดูว่าตัวไหน) |
| `permission denied ... docker.sock` | user ไม่อยู่ใน group docker | INSTALL.md ส่วน 1.3 + logout/login |
| `HTTP 429 ... quota` จาก provider | โควตา LLM หมด | ระบบสลับ rule-based ให้อัตโนมัติ — งานเดินต่อได้; ระยะยาว: เติมโควตา หรือสลับไปใช้ Local AI (`set-provider openai-compat`) |
| อยากใช้ Ollama/vLLM local เป็นสมอง | — | `lmds config set-provider openai-compat --base-url http://<ip>:11434/v1 --model gpt-oss:20b` (Ollama) หรือ `--base-url http://<ip>:8000/v1` (vLLM) — ไม่มี key ก็ใช้ได้ · ตั้งไม่ติดดู [INSTALL §3.2.1](INSTALL.md) |
| `download` พังกลางคัน / `No space left on device` | ดิสก์เต็ม | `df -h ~` · ย้ายที่เก็บ: `HF_HOME=/data/hf-cache` (vLLM) หรือ `MODEL_DIR=/data/models` (GGUF) แล้ว download ใหม่ (resume ต่อได้) |
| `start` ครั้งแรกค้างนานผิดปกติ ยังไม่ขึ้น log อะไร | Docker กำลัง pull image (~10–20 GB) | ปกติ — ดูความคืบหน้าด้วย `docker pull vllm/vllm-openai:latest` แยกอีก terminal · ดึงล่วงหน้าได้ตาม [INSTALL §1.7](INSTALL.md) |
| `docker pull` ล้ม / `TLS handshake timeout` | เครื่องอยู่หลัง proxy หรือโดน rate limit | ตั้ง proxy ให้ **docker daemon** ด้วย ไม่ใช่แค่ shell ([INSTALL §1.7](INSTALL.md)) |
| `prepare-runtime` build ล้มบน DGX Spark | ขาด CUDA Toolkit หรือ CUDA arch ไม่ตรง | ดูบรรทัดเตือน `ไม่พบ nvcc` · override ได้: `CUDA_ARCHITECTURES=121 ./xxx-single.sh prepare-runtime` |
| `start` บนเครื่อง ARM64 ใหม่ขึ้น `ยังไม่มี llama-server … build ให้ก่อน` แล้วเงียบนาน | กำลัง build llama.cpp ให้เอง (~10–30 นาที ครั้งแรกครั้งเดียว) | ปกติ — ไม่ต้องรัน `prepare-runtime` เองแล้ว · ถ้าจบด้วย `sudo apt-get … install -y git cmake` = ขาด build deps และ sudo ต้องใส่รหัส → รันคำสั่งนั้นเองแล้ว start ใหม่ |
| `download` ขึ้น `กำลังรันอยู่แล้ว (อีก process ถือ …/.download.lock)` | สั่ง download ซ้อนกัน (hub + CLI, หรือตัวเก่าที่ session หลุดแต่ curl ยังโหลดอยู่) | รอให้ตัวเดิมจบ (`ps -ef \| grep curl`) แล้วสั่งซ้ำ — มันจะต่อไฟล์ให้ครบเอง ไม่โหลดใหม่ |
| `download` ขึ้น `ดิสก์ … เหลือ X MB แต่ไฟล์ต้องการ Y MB` / `ถอยไปสตรีมเดี่ยว` | ดิสก์ไม่พอ · โหลดขนาน (`FETCH_PARTS`) ต้องมีที่ว่าง ~2 เท่าของไฟล์ชั่วคราว | ล้างที่ว่าง หรือ `MODEL_DIR=/data/models ./xxx-single.sh download` (ตั้ง `MODEL_DIR` เดียวกันตอน start) · ที่ว่างพอไฟล์เดียวแต่ไม่ถึง 2 เท่า = โหลดสตรีมเดี่ยวช้ากว่าแต่ได้ไฟล์ |
| `download` GGUF ได้ 0.3–1.4 MB/s ทั้งที่เน็ตเร็ว (ETA เป็นสิบชั่วโมง) | HF ย้ายไฟล์ใหญ่ไป **Xet bridge** — สตรีมเดี่ยวช้ามาก แต่ยิง range หลายส่วนพร้อมกันได้ ~50 MB/s | bundle ที่สร้างตั้งแต่ 0.6.0 โหลดขนาน 8 ส่วนเอง (`FETCH_PARTS=8` · ไฟล์ ≥256 MB) — bundle เก่า `lmds rebuild <ชื่อ>` · ปรับ `FETCH_PARTS=16` ได้ · ต้องมีดิสก์ว่าง ~2 เท่าของไฟล์ชั่วคราว (ส่วนย่อยใน `.parts/` + ไฟล์รวม) ไม่พอถอยไปสตรีมเดี่ยวเอง · ไม่มี aria2c ก็ได้ (ใช้ curl `-r`) |
| SHA-256 ไม่ตรงหลังโหลดขนานแล้ว resume | ส่วนย่อยถูก append ซ้ำ (curl retry เอง หรือสอง download ซ้อนกัน) | ลบ `<ไฟล์>.parts/` แล้ว `download` ใหม่ · ตั้งแต่ 0.6.0 ล็อก `.download.lock` กันสั่งซ้อน และให้ลูปนอก resume แทน curl |
| analyze บนหน้าเว็บขึ้น "ต่อ Hugging Face ไม่ได้" ทั้งที่เปิดเว็บ HF ได้ | byte แรกจาก Xet มาช้า 20–60 วิ เกิน read timeout เดิม (30 วิ) | แก้แล้ว 0.6.0 (read 120 วิ / connect 30 วิ) — ยังเจอ = อัปเดต hub · ข้อความจริงถึงเบราว์เซอร์เป็น 422 `{kind:"hub"}` |
| stacked `sync-worker`/`start` ตาย `Permission denied (publickey)` | head ไม่มีกุญแจไป worker (`node setup` ลงแต่กุญแจของ hub) | `lmds cluster pair <head> <worker>` แล้วสั่งใหม่ · ดูทีละข้อ: `lmds cluster doctor <head> <worker>` |
| stacked `start` ขึ้น `image นี้ไม่รู้จักสถาปัตยกรรม 'xxx'` ก่อนปล่อย worker | โมเดลใหม่กว่า transformers ใน image | `lmds set <ชื่อ> --image <image ใหม่กว่า>` → `prepare-runtime` → `start` (ตรวจก่อนได้ด้วยคำสั่ง `docker run … CONFIG_MAPPING_NAMES` ที่ error พิมพ์ให้) |
| `prepare-runtime` (stacked) บอกว่า pull ล้มที่เครื่อง X | node นั้นไม่ถึง registry / ghcr rate-limit / nvcr ต้อง NGC key / ไม่มีเน็ต | ทำตามที่ข้อความบอกต่อ registry นั้น (`docker login` · NGC key · proxy ของ docker daemon · `docker save \| ssh docker load`) แล้วสั่งซ้ำ (idempotent) |
| ปุ่ม Update: hub ผ่านแต่ node "ไม่ผ่าน" | node ไม่มี checkout (clone จาก bundle ได้โฟลเดอร์เปล่า) / โฟลเดอร์ไม่ใช่ git / checkout แยกสาย | แก้แล้ว 0.6.0 (`git clone -b main` · โฟลเดอร์เดิม → `.bak-<เวลา>` · แยกสาย → branch `local-<เวลา>`) — อัปเดต hub ก่อนแล้วกดใหม่ · ป้าย "ยังไม่ตรง hub" ทั้งที่อัปเดตแล้ว = hash ย่อ 7 กับ 8 ตัว (แก้แล้ว เทียบ prefix) |
| `install.sh` ล้มที่ pip (PyPI ช้า) แล้วเครื่องไม่มี `lmds` | รุ่นเก่า: `venv --clear` ทับก่อนแล้ว pip ค่อยล้ม | ตั้งแต่ 0.6.0 venv เดิมถูกย้ายไป `venv.old` แล้วคืนให้เมื่อล้ม — รุ่นเดิมยังใช้ได้ · ลองใหม่ `PIP_TIMEOUT=120 ./install.sh` (ค่าเริ่มต้น `PIP_RETRIES=8 PIP_TIMEOUT=60`) |
| `lmds remove` / ปุ่ม Remove ขึ้น "ต้องใช้ sudo rm -rf" | weight ที่ container เขียนเป็น root | ตั้งแต่ 0.6.0 ลบผ่าน docker ให้เอง (root ในคอนเทนเนอร์ · เฉพาะใต้ home/HF cache · ไม่ pull image) — ยังขึ้น = ผู้ใช้ไม่อยู่กลุ่ม docker หรือ path อยู่นอกรั้ว ลบเองตามคำสั่งที่พิมพ์ให้ |
| หน้าภาพรวมขึ้น "port shared" ทันทีหลัง deploy | bundle ใหม่ได้ 8000 ซ้ำกับตัวเดิม (รุ่นก่อน 0.6.0) | ตั้ง port ใหม่ในฟอร์ม settings แล้ว Save · bundle ที่สร้างตั้งแต่ 0.6.0 ได้พอร์ตว่างของเครื่องนั้นตั้งแต่ analyze |
| ตั้ง `API_KEY` กับ llama.cpp แล้วยิงไม่ใส่ key ก็ได้ 200 | bundle รุ่น 0.6.0 ช่วงสั้น ๆ ใช้ env `LLAMA_ARG_API_KEY` ซึ่ง llama-server ไม่มี | `lmds rebuild <ชื่อ>` — controller ปัจจุบันใช้ `--api-key-file` (พิสูจน์ 401/401/200 กับ b10799) |
| `User-specified max_model_len (262144) is greater than the derived max_model_len (max_position_embeddings=131072)` ตอน start (stacked: worker ตายก่อน head) | ตั้ง context เกินเพดานของโมเดล | ตั้งแต่ 0.6.0 ช่อง context มีป้าย max และระบบไม่ยอมบันทึกค่าที่เกิน (`lmds set --context` ก็เช่นกัน) · ใช้ค่า ≤ max_position_embeddings · ต้องการเกินจริง ๆ ใส่ engine env `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` ใน Advanced (ผลอาจเป็น nan) |
| `ดึง image … ไม่สำเร็จ` ตอน prepare-runtime/start (stacked หรือเดี่ยว vLLM) | docker pull ล้ม — ดูบรรทัด `docker บอกว่า:` และ `→ สาเหตุ:` ที่ controller พิมพ์ | ออกเน็ตไม่ถึง registry (DNS/ไฟร์วอลล์/proxy ของ docker daemon) · สายหลุด (controller ลองซ้ำ 3 รอบให้แล้ว สั่งใหม่ต่อได้เลย layer เดิมไม่โหลดซ้ำ) · rate limit → `docker login` · tag/digest หาย → `lmds set <slug> --image` · ดิสก์เต็ม → `docker system prune -a` · ไม่มีเน็ตเลย → `docker save <image> \| ssh <head> docker load` แล้ว `lmds set --image <repo:tag>` |
| `verify-files: OK (…/models--A--X/…)` แล้วตามด้วย `ERROR: ยังไม่ได้ download (ไม่พบ …/models--B--X/…)` | โมเดลชื่อเดียวกันคนละเจ้าของ (`A/X` กับ `B/X`) เคยลงโฟลเดอร์ bundle เดียวกัน มี controller สองตัว | ตั้งแต่ 0.6.0 bundle ของเจ้าของที่สองได้ slug `x-<owner>` แยกโฟลเดอร์ และเปลี่ยน single↔stacked แล้ว controller เก่าถูกย้ายเป็น `.replaced-<stamp>` · bundle เก่า: `lmds remove <slug>` ตัวที่ไม่ใช้ แล้ว deploy ใหม่ |
| การ์ด **Docker + GPU** ขึ้น `not ready · <user> is not in the docker group` / ติดตั้งแล้วขึ้น "มี Docker แต่ user ปัจจุบันเรียกไม่ได้" | user ของ hub ไม่อยู่ในกลุ่ม docker (หรือเพิ่งเพิ่มแต่ session ยังเก่า) | กด **Fix docker access** บนการ์ด ใส่รหัส sudo — ระบบ `usermod -aG docker` แล้วรีสตาร์ต session ให้ (คอนโซลหาย ~10 วิ) · ทำเอง: `sudo usermod -aG docker $USER && sudo systemctl restart user@$(id -u)` · ไม่มีสิทธิ์ sudo เลย = ต้องให้ผู้ดูแลเครื่องรันให้ |
| ssh ขาดกลาง job บนหน้าเว็บ (exit 255) | สายหลุด แต่คำสั่งปลายทางมักรันต่อ | ดู `lmds node run <n> logs <ชื่อ>` ก่อนสั่งซ้ำ — ไม่งั้นชน "กำลังรัน" หรือ download ซ้อน |
| ลิงก์ `ollama.com/...` ใช้ไม่ได้ | ยังรองรับเฉพาะ Hugging Face | ใช้ลิงก์ HF ของ GGUF ตัวเดียวกันแทน (roadmap เฟส 2) |
| `verify-files` แจ้ง shard หาย / ขนาดไม่ตรง | download ไม่ครบ หรือไฟล์ใน cache ถูกลบ | `lmds repair <ชื่อ>` (โหลดเฉพาะส่วนที่ขาด) |
| `lmds list` ขึ้น ⚠ (ไฟล์ controller หาย) | โฟลเดอร์ bundle ถูกลบ/ย้าย | `lmds deploy` ลิงก์เดิมเพื่อสร้าง bundle ใหม่ — weight เดิมใช้ต่อได้ · หรือ `lmds remove <ชื่อ>` ถ้าไม่ใช้แล้ว |
| มีแถวขยะค้างใน `lmds ps` / `lmds list` | process/ทะเบียนเก่าค้างจากรอบก่อน | `lmds remove <ชื่อ>` เก็บกวาดให้ครบทุกที่ |
| กราฟ/สถานะในหน้าเว็บตามหลังของจริงหลายวินาที | **แก้แล้วตั้งแต่รุ่นนี้** — เดิม refresher ไล่ SSH ทีละเครื่องแบบเรียงคิว ทำให้การอ่านค่าของเครื่องนี้ต้องรอครบทุก node ก่อน | อัปเดต LMDS ให้ครบทุกเครื่อง: `lmds node install <ชื่อ>` · เครื่องปลายทางอ่านตรงจาก `/api/version` ว่าตรงกันหรือยัง |
| ค่าของ node หนึ่งเครื่องเก่ากว่าเพื่อน | เครื่องนั้นต่อไม่ติด จึงถูกถอยจังหวะออกไปเรื่อย ๆ (สูงสุด 120 วิ) | ดู `age_seconds` ในการ์ดนั้น · กดปุ่ม Refresh รายเครื่องเพื่อบังคับอ่านใหม่ทันที |

### ถ้าแก้เองไม่ได้ — ข้อมูลที่ต้องเก็บส่งทีมพัฒนา

```bash
lmds version
lmds hardware
./xxx-single.sh logs 500 > failure.log
# + คำสั่งเต็มที่รันแล้วพัง + ข้อความ error ทั้งหมด
```

## 8.5 บันทึกจากของจริง — Nemotron-3-Super-120B-A12B-NVFP4 บน DGX Spark

วัดเองบน spark-head (GB10 ตัวเดียว) 14 ส.ค. 2569 · เก็บไว้เพราะเป็นโมเดลสายคิด +
เรียก tool ที่ deploy ยากที่สุดเท่าที่ผ่านมา และหลายอย่างไม่ตรงกับที่การ์ดเขียน

**flag ที่ใช้จริงแล้วผ่านครบ:**

```
--max-model-len 262144 --gpu-memory-utilization 0.85 --max-num-seqs 4
--kv-cache-dtype fp8
--enable-auto-tool-choice --tool-parser qwen3_xml --reasoning-parser nemotron_v3
```

| วัดได้ | ค่า |
|---|---|
| GPU KV cache | 1,297,920 tokens |
| concurrency ที่ 262k/คำขอ | **25.41x** (ไม่ใช่ 4.95x เพราะเป็น hybrid Mamba — มีแค่บาง layer ที่เก็บ KV โต) |
| โหลด weight 17 shard | ~9 นาที |
| prompt 100k tokens | 41 วินาที |
| tool calling `auto` · หลาย tool พร้อมกัน | ผ่านทั้งคู่ |

**สามเรื่องที่การ์ดของ NVIDIA เขียนไว้ต่างจากที่เราต้องใช้ — และเหตุผล:**

1. **`--tool-call-parser qwen3_coder` vs `qwen3_xml`** — ไม่ต่างกันเลย ใน vLLM
   รุ่นใหม่สองชื่อนี้ map ไป `Qwen3EngineToolParser` **คลาสเดียวกัน** จะใส่ชื่อไหน
   ก็ได้ · ที่ต้องระวังคือ `hermes` ซึ่งอ่านรูปแบบของโมเดลตระกูลนี้ไม่ออก
2. **`--reasoning-parser-plugin super_v3_reasoning_parser.py` + `super_v3`** —
   จำเป็นเฉพาะ `vllm==0.20.0` ที่การ์ดปักไว้ · image ใหม่กว่ามี `nemotron_v3`
   เป็น parser engine ในตัวแล้ว (`vllm.parser.nemotron_v3`) **ไม่ต้องโหลดไฟล์
   ปลั๊กอินมา** · ตัว `super_v3` เองเป็นแค่ subclass ของ `deepseek_r1` ที่เพิ่ม
   การกู้เคส thinking ว่าง
3. **`--trust-remote-code`** — จำเป็นบน 0.20.0 · image ใหม่รู้จักสถาปัตยกรรม
   NemotronH แล้ว จึงรันได้โดยไม่ต้องเปิด

**context 1M ทำได้จริง** — การ์ดบอกให้ตั้ง `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` กับ
`--max-model-len 1048576` · ด้วย KV cache 1.29M tokens ที่วัดได้ ยังเหลือราว 6 สาย
พร้อมกันที่ 1M ต่อคำขอ (เดิมเคยสรุปผิดว่าไปไม่ถึง — สรุปจากขนาด KV แบบ dense
ทั้งที่โมเดลนี้เป็น hybrid)

**ตั้งค่า engine ที่ไม่มี knob เฉพาะ** — ใช้ `lmds set` ได้แล้วทั้งหมด (เดิมหัวข้อนี้เคยเขียนว่า
`--speculative_config`, `--moe-backend`, `--enable-chunked-prefill` ฯลฯ "ยังตั้งผ่าน LMDS ไม่ได้"):

```bash
lmds set <slug> --tool-parser qwen3_xml --reasoning-parser qwen3        # ติดถาวร รวม autostart
lmds set <slug> --engine-env "VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_USE_FLASHINFER_MOE_FP4=0"
lmds set <slug> --extra-args '--speculative-config {"method":"mtp","num_speculative_tokens":2} --enable-chunked-prefill'
DRY_RUN=1 ./xxx-single.sh start        # vLLM: พิมพ์ image + argv ที่จะรันจริง ไม่แตะ docker/GPU
./xxx-single.sh serve-args             # llama.cpp: พิมพ์ argv ของ llama-server
```

`--extra-args` เก็บใน `bundle.args` (ไฟล์แยก ไม่ใช่ `bundle.env`) เพราะรูป `${VAR:-value}` ของ
bash หยุดที่ `}` ตัวแรก JSON จึงถูกตัดกลางคัน · ถูกแตกเป็น argv ด้วยช่องว่าง — **JSON ต้องเขียน
ติดกันไม่มีช่องว่าง** หรือใช้รูป `--flag=value` · ตรวจผลด้วย `DRY_RUN=1 … start` ก่อนรอโหลดโมเดล
หลายนาทีเพื่อรู้ว่าแฟล็กไม่ถึง

---

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
- `API_KEY` ไม่เคยอยู่บน argv (ดูกล่องใน §2) — vLLM/stacked รับผ่าน env ผู้ที่ใช้ `docker` บนเครื่องเดียวกันจึงอ่านได้ด้วย `docker inspect` · llama.cpp อ่านจากไฟล์ 0600 ใน `RUN_DIR` (docker: mount ro) · SGLang ยังต้องส่ง `--api-key` บน argv (ไม่ใช่ช่องโหว่ต่อคนนอก แต่ไม่ควรใช้ key เดียวกับระบบอื่น)
- **ข้อมูลที่ออกจากเครื่อง**: ตอนวางแผน ระบบส่ง metadata ของโมเดล (model card, `config.json`, รายชื่อไฟล์) ไปยัง LLM provider ที่ตั้งไว้ — ไม่ส่ง weight, ไม่ส่ง key, ไม่ส่งข้อมูลผู้ใช้ · องค์กรที่ห้ามข้อมูลออก ให้ใช้ `--no-llm` หรือตั้ง provider เป็น Local AI ([INSTALL §3.2.1](INSTALL.md))
- สำเนา prompt/คำตอบของทุกครั้งที่เรียก LLM ถูกเก็บไว้ที่ `~/.config/lmds/sessions/` (redact secret แล้ว) — ลบได้ถ้าไม่ต้องการเก็บประวัติ
- flag `--trust-remote-code` อนุมัติเฉพาะหลัง review ไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน SPECIAL_FILES.md)
- bundle ที่รับมาจากคนอื่น ตรวจก่อนใช้: `lmds validate <โฟลเดอร์>`
