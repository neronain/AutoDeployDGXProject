<div align="center">

# LMDS · Local Model Deploy Studio

**จากลิงก์ Hugging Face → เซิร์ฟเวอร์ที่ยิงได้จริงบนเครื่องของคุณเอง**

ระบบวางโมเดลภาษาลงเครื่องตัวเอง สำหรับ **NVIDIA DGX Spark** และ **Ubuntu + RTX**
เครื่องเดียวหรือหลายเครื่องรวมเป็นโมเดลเดียวก็ได้ · ไม่มีอะไรออกนอกเครื่องนอกจากที่คุณสั่ง

[![version](https://img.shields.io/badge/version-0.3.0-1f5fbf)](CHANGELOG.md)
[![tests](https://img.shields.io/badge/tests-903-17703f)](tests/)
[![platform](https://img.shields.io/badge/platform-Ubuntu%2022.04%20%7C%2024.04-555)](docs/INSTALL.md)
[![arch](https://img.shields.io/badge/arch-ARM64%20%C2%B7%20x86__64-555)](docs/INSTALL.md)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab)](pyproject.toml)
[![license](https://img.shields.io/badge/license-proprietary-8a5300)](LICENSE)

**[ติดตั้ง](docs/INSTALL.md)** · **[คู่มือใช้งาน](docs/USAGE.md)** · **[หลายเครื่อง](docs/RUNBOOK-MULTI-NODE.md)** · **[สิ่งที่ตรวจให้ก่อน deploy](docs/PREFLIGHT.md)** · **[English](README.en.md)**

สร้างและดูแลโดย **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>

---

## ปัญหาที่มันแก้

การเอาโมเดลลงเครื่องตัวเองไม่ได้ยากตรง "รันคำสั่งไหน" — มันยากตรงที่**คำสั่งที่ดูถูกทุกอย่าง
กลับให้ผลผิดโดยไม่มี error** ให้เห็น: context ถูกตัดเงียบ ๆ เหลือหนึ่งในสิบ, tool calling ที่
เปิดไว้แต่ไม่เคยแปลงคำตอบจริง, สายเชื่อม 200G ที่ negotiate ลงมาเหลือ 50G, KV cache ที่คำนวณ
เกินจริงยี่สิบเท่าจนตั้ง context ได้แค่เศษเดียวของที่เครื่องรับไหว

LMDS เกิดจากการไล่รันของจริงแล้วเก็บทุกอาการพวกนี้กลับมาเป็นการตรวจอัตโนมัติ

| | |
|---|---|
| 🧮 **คำนวณด้วยโค้ด ไม่ใช่ LLM** | memory fit, KV cache, token budget, ความเร็วลิงก์ — LLM มีหน้าที่แค่วิจัยโมเดลและเลือกค่าใน Deployment Plan ที่เป็น JSON schema ตายตัว **ไม่เคยเขียน Bash เอง** |
| 🛡️ **ทุก bundle ผ่านด่านก่อนถึงมือคุณ** | `bash -n`, audit rules, SHA-256 checksums — ไม่ผ่านคือไม่มี ZIP |
| 🔍 **บอกตอนที่ยังแก้ทัน** | ไม่ใช่ตอนที่ผู้ใช้มาบ่นว่าช้า · ทุกข้อที่ตรวจมาจากของที่พังจริงบนเครื่องจริง |
| 🔌 **ทำงานได้โดยไม่มี LLM** | โหมด rule-based ใช้สูตรที่รันผ่านจริงมาแล้ว · air-gapped ก็ใช้ได้ |

## เริ่มใน 3 คำสั่ง

```bash
./install.sh                 # ลง Docker / NVIDIA toolkit ที่ขาดให้ด้วย — ถามก่อนทุกขั้นที่ใช้ sudo
lmds hardware                # เครื่องนี้คือ target อะไร มีอะไรอยู่บ้าง
lmds deploy Qwen/Qwen3-32B   # วิเคราะห์ → วางแผน → ให้ยืนยัน → bundle + ZIP ที่ผ่านทุกด่าน
```

<details>
<summary>ตัวอย่างเพิ่มเติม</summary>

```bash
# ดูก่อนว่าลงได้ไหม โดยยังไม่สร้างอะไร
lmds inspect Qwen/Qwen3-32B --target rtx-pro-4000-dual

# context ที่จะตั้งนี้ ควรไหม — ตอบเป็นตาราง context x จำนวนคนพร้อมกัน
lmds inspect <repo> --target dgx-spark-stacked --context 262144

# โมเดล gated → ถาม HF token ให้เอง (Enter ข้ามได้)
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single

# ใหญ่เกินหนึ่งเครื่อง → stacked (worker-first + sync-worker ให้อัตโนมัติ)
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 --target dgx-spark-stacked
```

</details>

---

## สามอย่างที่ไม่ค่อยมีที่ไหนตอบให้

### 1 · "ตั้ง context เท่านี้แล้วจะมีกี่คนใช้พร้อมกันได้"

เครื่องมือทั่วไปตอบได้แค่ว่า context สูงสุดเท่าไร ซึ่งตามนิยามคือค่าที่**คนเดียว**กิน KV pool
หมดพอดี — ตั้งตามนั้นแล้วคนที่สองต่อคิว โดยไม่มีอะไรบอก

```
KV bf16 · 120 KiB ต่อ token
  context      KV ต่อคน    พร้อมกัน
   32,768       3.8 GB       14.1
  131,072        15 GB        3.5
  262,144        30 GB        1.8   ← ค่าที่กรอก
```
> • ใส่ได้ แต่ได้ 1.8 คนพร้อมกัน — หนึ่งคำสนทนากิน KV pool เกือบหมด
> • เปลี่ยน KV เป็น fp8 → 30 GB เหลือ 15 GB · พร้อมกันจาก 1.8 เป็น 3.5 คน
> • 2 เครื่อง — งบนี้ยังไม่รวม NCCL buffer ข้ามเครื่อง

ขึ้นทั้งใน CLI และ**ในหน้าเว็บระหว่างที่ยังพิมพ์เลขอยู่** · รองรับทั้ง GQA และ **MLA**
(DeepSeek-V2/V3, Kimi K2/K3) ซึ่งเก็บ KV เป็น latent ก้อนเดียว — สูตรเดียวใช้กับทุกตระกูลไม่ได้

### 2 · หลายเครื่อง = โมเดลเดียว

> **stacked ไม่ได้แปลว่าเร็วขึ้น — แปลว่าใหญ่เกินหนึ่งเครื่อง**
> โมเดลที่ลงเครื่องเดียวได้ รันเครื่องเดียวเร็วกว่าเสมอ

| | เครื่องเดียว | Stacked |
|---|---|---|
| Engine | vLLM **หรือ** llama.cpp | **vLLM เท่านั้น** |
| Artifact | safetensors หรือ GGUF | **safetensors เท่านั้น** |
| สายเชื่อม | ไม่ต้อง | **ต้องมี** ≥25G (ของจริง 200G RoCE) |
| จำนวนเครื่อง | 1 | ต่อตรง ≤3 · ผ่าน switch ≤4 |

ระบบตรวจ ConnectX/RDMA ให้เอง บอกว่าเครื่องคู่ไหน stacked กันได้ เขียน `cluster.env` ให้
และ**เตือนเมื่อลิงก์ negotiate ได้ต่ำกว่าที่การ์ดทำได้** (NVIDIA ตรวจรับที่ ≥184 Gbit/s —
พอร์ตที่ปล่อย auto มักลงมาเหลือ 50G แล้วทุกอย่างยังดูปกติ)

→ [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) · [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) · [เทียบกับเอกสารของ NVIDIA](docs/NVIDIA-CLUSTER-SOURCES.md)

### 3 · หน้าเว็บที่ทำได้เท่า CLI

```bash
lmds web --bind 0.0.0.0 -b      # ถาม token ก่อน แล้วจำไว้ — ลิงก์ bookmark ได้
```

deploy wizard, download + verify, start/stop/restart, ตั้ง port/context/slots/API key/bind,
doctor, logs, ชุดทดสอบ (`test-text` `test-vision` `test-tools` `bench` `stress`), autostart,
คำสั่ง stacked, repair, remove — **และคุมโมเดลบนเครื่องอื่นได้เท่ากับเครื่องตัวเอง**

- **อ่านสถานะได้ก่อนอ่านตัวหนังสือ** — เกจ CPU / Unified·RAM / VRAM / Disk ชุดเดียวกันทุกเครื่อง
  พร้อมสีเตือนก่อนของหมด · ค่าที่การ์ดไม่รายงานถูกซ่อน ไม่ใช่โชว์ 0
- **เครื่องที่ stacked ด้วยกันได้มีรั้วสีคร่อม** พร้อมป้าย `CLUSTER A/B`
- **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่านจาก dispatch table ของสคริปต์เอง
- **ปรับขนาดตัวอักษรได้ 4 ระดับ** (S/M/L/XL) และธีมสว่าง/มืด/ตามเครื่อง — จำไว้ต่อเบราว์เซอร์
- **ไม่ดึงอะไรจากอินเทอร์เน็ตเลย** ใช้ได้บนเครื่องหลัง proxy หรือ air-gapped

> 🔒 หน้านี้สั่ง start/stop/ลบโมเดลได้ จึง bind `127.0.0.1` เป็นค่าเริ่มต้น · **ลิงก์ที่พิมพ์ออกมา
> ไม่มี token ติดไปด้วย** เพราะ URL ไปโผล่ใน history, log ของ proxy และ referrer · เดา token
> ผิดติดกันจาก IP เดิมโดนหน่วงแบบทวีคูณ

**ผู้ช่วยมุมขวาล่าง** ตอบจาก*สถานะจริงของ fleet นี้* ไม่ใช่ความรู้ทั่วไป — "เครื่องไหนต่อไม่ติด",
"ทำไม msi-6 ยัง start ไม่ได้" · ใช้ LLM ตัวเดียวกับที่วางแผน deploy (ตั้งครั้งเดียวได้ทั้งสองอย่าง)
และ**ซ่อนตัวเองเมื่อยังไม่ได้ตั้ง provider** เพราะกล่องแชทที่ตอบว่า "ยังไม่ได้ตั้ง" ทุกครั้ง
แย่กว่าไม่มีกล่องแชท · มันรู้กติกาเรื่อง context/KV แต่**ถูกสั่งห้ามคิดเลขเอง** — ให้ชี้มาที่
`lmds inspect --context` เพราะเลขที่ LLM คูณเองผิดแบบดูน่าเชื่อ ซึ่งแย่กว่าตอบว่าไม่รู้

---

## คุมทั้ง fleet จากเครื่องเดียว

```bash
lmds node add 192.168.10.21 --user ops --install   # ถามรหัสผ่านครั้งเดียว → ติดตั้ง key + LMDS ให้
lmds ps --all                     # โมเดลของทุกเครื่องในตารางเดียว
lmds node cluster                 # เครื่องไหนมี 200G และจับคู่ stacked กันได้
lmds scan --all                   # weight ที่มีอยู่แล้วบนทุกเครื่อง — ไม่ต้องโหลดซ้ำ
lmds node push spark2 <slug>      # ส่ง bundle ตัวที่อนุมัติแล้วไปติดตั้งเครื่องอื่น
```

เครื่องปลายทาง**ไม่ต้องรัน daemon** ไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22 และ**ไม่ต้องใช้ root**
(อยู่ในกลุ่ม `docker` พอ) · รหัสผ่านถูกทิ้งทันทีหลังติดตั้ง key — ทะเบียนไม่มีฟิลด์รหัสผ่านโดยตั้งใจ

<details>
<summary>คำสั่งจัดการโมเดลทั้งหมด</summary>

```bash
lmds ps                  # ใครรันอยู่: ชื่อ, โมเดล, engine, port, ● running / ◐ loading / ○ stopped
lmds list                # bundle ทั้งหมด + engine/port/context/ฟีเจอร์ + autostart
lmds smoke <ชื่อ>         # พิสูจน์ว่ารันได้จริง: download → verify → start → test-text → stop
lmds start/stop/restart <ชื่อ>
lmds logs <ชื่อ> -f       # -n 500 = ย้อนหลัง
lmds enable <ชื่อ>        # กลับมาเองหลัง reboot (systemd) · disable = ยกเลิก
lmds doctor <ชื่อ>        # ทำไมยัง download/start ไม่ผ่าน + คำสั่งแก้
lmds repair <ชื่อ>        # โหลดไฟล์ที่ขาด/เสียกลับมา แล้วตรวจซ้ำ
lmds rebuild <ชื่อ>       # สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน
lmds remove <ชื่อ>        # ลบทั้งหมด (--keep-weights = เก็บ weight)
lmds recipes             # สูตรที่รันผ่านจริง — ใช้เองเมื่อไม่มี API key
```

`lmds ps` เห็น **container ที่ไม่ได้ deploy ผ่าน LMDS** ด้วย (vLLM/llama.cpp/Ollama/TGI ที่รันอยู่แล้ว)
— stop/restart/logs/enable ได้เหมือนกัน โดยกลุ่มนี้ใช้ `docker stop` ไม่ลบ container ทิ้ง

</details>

## รองรับอะไรบ้าง

| | ARM64 / unified (Spark) | x86_64 / discrete (RTX) |
|---|---|---|
| **llama.cpp** | ✅ native build | ✅ docker (+ multimodal) |
| **vLLM** | ✅ docker | ✅ docker |

ผ่าน hardware validation ครบทั้ง 5 ตระกูลโมเดล — GGUF, NVFP4, MoE, dense safetensors, gated repo
· **22 target preset** (7 ตัวทดสอบบนเครื่องจริงแล้ว) · **903 เทสต์**

> **แหล่งโมเดล: Hugging Face เท่านั้น** — Ollama registry และ NVIDIA NGC อยู่ในเฟส 2
> (ใส่ลิงก์เข้าไปแล้วระบบบอกเองว่ายังไม่รองรับ พร้อมแนะทางอื่น)

## อัปเดต

```bash
cd ~/AutoDeployDGXProject && git pull && ./install.sh
```

> ⚠️ **`git pull` อย่างเดียวไม่พอ** — ติดตั้งแบบ copy เข้า venv (ไม่ใช่ editable) คำสั่ง `lmds`
> จะยังเป็นโค้ดเก่าจนกว่าจะรัน `./install.sh` ซ้ำ · config และ key เดิมอยู่ครบ ไม่ต้องตั้งใหม่

## ใช้คู่กับ LiteGate (ทางเลือก)

**[LiteGate · AiGatewayLocal](https://github.com/neronain/AiGatewayLocal)** คืออีกครึ่งของชุดนี้ —
LMDS *deploy* โมเดลลงเครื่องคุณ ส่วน LiteGate เป็น *ประตูเดียว* หน้าโมเดลทั้งหมด: API key, โควตา,
สิทธิ์ต่อคน และตรวจว่าเซิร์ฟเวอร์ที่รันอยู่**ทำอะไรได้จริง**

| ติดตั้ง | ได้อะไร |
|---|---|
| LMDS อย่างเดียว | deploy และรันโมเดลบนเครื่องตัวเอง มีหน้าเว็บและผู้ช่วยครบ |
| LiteGate อย่างเดียว | ประตูเดียว + key + โควตา หน้าเซิร์ฟเวอร์ที่รันมาด้วยวิธีไหนก็ได้ |
| **ทั้งคู่** | LMDS สร้าง · LiteGate วัดของจริงแล้วบอกคำสั่งที่ต้องแก้ |

**ไม่มีตัวไหนต้องพึ่งอีกตัว** · จุดที่ต่อกันได้เป็นทางเลือกทั้งหมด — ให้ LMDS ใช้โมเดลของคุณเอง
เป็นสมอง (`lmds config set-provider openai-compat --base-url http://litegate:8080/v1`),
`managed_by` ที่ทำให้คำแนะนำของ LiteGate กลายเป็นคำสั่งที่ก๊อปไปวางได้, และ parser ที่ LiteGate
บอกว่าขาดคือ knob ที่ LMDS เปิดได้ทันทีด้วย `restart --tool-parser` แล้วพิสูจน์ด้วย `test-tools`
ซึ่งวัดโหมด `auto` — โหมดเดียวกับที่ agent ใช้จริง ไม่ใช่โหมดบังคับที่ผ่านได้แม้ parser ผิด

## เอกสาร

| | |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | ติดตั้งทีละขั้น — prerequisites, ดิสก์, proxy/air-gapped, ตั้ง provider, ถอนการติดตั้ง |
| [USAGE.md](docs/USAGE.md) | คู่มือใช้งานเต็ม — deploy, คำสั่ง controller ทุกตัว + env, fleet, หน้าเว็บ, troubleshooting |
| [PREFLIGHT.md](docs/PREFLIGHT.md) | สิ่งที่ระบบตรวจให้ก่อน deploy และทำไม — ทุกข้อมาจากของที่พังจริง |
| [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) | ลำดับคำสั่งข้ามเครื่องที่รันจริงแล้ว พร้อมตัวเลขและเวลาที่ใช้แต่ละขั้น |
| [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) | คุมหลายเครื่องจากเครื่องเดียว |
| [NVIDIA-CLUSTER-SOURCES.md](docs/NVIDIA-CLUSTER-SOURCES.md) | เอกสารคลัสเตอร์ของ NVIDIA — อะไรยืนยันของเรา อะไรเติมของใหม่ |
| [PRD.md](docs/PRD.md) · [CLI_SPEC.md](docs/CLI_SPEC.md) · [ROADMAP.md](docs/ROADMAP.md) | ข้อกำหนด, สเปกคำสั่ง, แผนพัฒนา |
| [SECURITY.md](SECURITY.md) | ข้อมูลอะไรออกนอกเครื่อง, secret เก็บที่ไหน, แจ้งช่องโหว่ |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | ตั้ง dev env + กฎที่ห้ามละเมิด · ประวัติการเปลี่ยนแปลง |

## Requirements

- **Ubuntu 22.04 / 24.04** (ARM64 หรือ x86_64) — พัฒนาบน macOS ได้
- **Python 3.10+**
- **Docker + NVIDIA Container Toolkit** บนเครื่องเป้าหมาย (`./install.sh` ลงให้ได้)
- **ดิสก์ว่าง** ≈ *(ขนาดโมเดล × 1.2) + 25 GB* — runtime image ของ vLLM อย่างเดียว ~10–20 GB
- **LLM provider** (ทางเลือก): OpenAI / Gemini / MiniMax / OpenAI-compatible — หรือไม่มีเลยก็ได้ ใช้ `--no-llm`

ข้อเดียวที่ `install.sh` ไม่ทำให้คือ **NVIDIA driver** เพราะต้อง reboot และบางเครื่องมี driver ที่ใช้ได้
อยู่แล้วแต่ `ubuntu-drivers install` ชน dependency จนพัง

## สำหรับผู้พัฒนา

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest
```

กฎที่ห้ามละเมิดและวิธีเพิ่ม target preset / provider / quality gate: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

**Proprietary — สงวนลิขสิทธิ์** · ดู [LICENSE](LICENSE)

การเปิดซอร์สให้อ่านได้ในรีโปนี้ไม่ได้ให้สิทธิ์ใช้งานหรือแจกจ่ายต่อ · **bundle ที่ผู้ใช้ generate
ออกมาเป็นของผู้ใช้เอง** ใช้/แก้/ส่งต่อได้อิสระ · โมเดล image และ runtime ของบุคคลที่สามอยู่ใต้
license ของเจ้าของนั้น ๆ

<div align="center">
<br>

สืบทอดมาตรฐาน controller จาก [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers)

**neronain** · [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>
