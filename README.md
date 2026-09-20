<div align="center">

# LMDS · Local Model Deploy Studio

**จากลิงก์ Hugging Face → เซิร์ฟเวอร์ที่ยิงได้จริงบนเครื่องของคุณเอง**

ระบบวางโมเดลภาษาลงเครื่องตัวเอง สำหรับ **NVIDIA DGX Spark** และ **Ubuntu + RTX**
เครื่องเดียวหรือหลายเครื่องรวมเป็นโมเดลเดียวก็ได้ · ไม่มีอะไรออกนอกเครื่องนอกจากที่คุณสั่ง

[![version](https://img.shields.io/badge/version-0.9.1-1f5fbf)](CHANGELOG.md)
[![tests](https://img.shields.io/badge/tests-2224-17703f)](tests/)
[![platform](https://img.shields.io/badge/platform-Ubuntu%2022.04%20%7C%2024.04%20%7C%2025.04-555)](docs/INSTALL.md)
[![arch](https://img.shields.io/badge/arch-ARM64%20%C2%B7%20x86__64-555)](docs/INSTALL.md)
[![python](https://img.shields.io/badge/python-3.10--3.13-3776ab)](pyproject.toml)
[![license](https://img.shields.io/badge/license-source--available-8a5300)](LICENSE)
[![free](https://img.shields.io/badge/free-1%20machine-17703f)](docs/COMMERCIAL.md)

**[ติดตั้ง](docs/INSTALL.md)** · **[คู่มือใช้งาน](docs/USAGE.md)** · **[หลายเครื่อง](docs/RUNBOOK-MULTI-NODE.md)** · **[ความปลอดภัย](SECURITY.md)** · **[พอร์ต &amp; เครือข่าย](docs/NETWORK.md)** · **[English](README.en.md)**

สร้างและดูแลโดย **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>

---

## หน้าตาระบบ

<div align="center">

<img src="docs/img/fleet.png" alt="หน้าเว็บ LMDS — ทั้งฟลีตในหน้าเดียว" width="900">

*ทุกเครื่องในฟลีตหน้าเดียว — GPU, RAM, อุณหภูมิ, โมเดลที่รันอยู่ · เครื่องที่ไม่มี GPU
รู้ตัวว่าเป็น control plane และไม่ยอมโหลด weight ลงมา*

<img src="docs/img/model-scores.png" alt="คะแนนโมเดลที่วัดจากเซิร์ฟเวอร์จริง" width="900">

*คะแนนที่ยิงผ่าน OpenAI API ของเซิร์ฟเวอร์จริง — decode tok/s, TTFT, context ที่ตั้งได้จริง
และความสามารถ 7 ข้อ · เทียบข้าม engine และข้ามเครื่องได้*

</div>

## ปัญหาที่มันแก้

การเอาโมเดลลงเครื่องตัวเองไม่ได้ยากตรง "รันคำสั่งไหน" — มันยากตรงที่**คำสั่งที่ดูถูกทุกอย่าง
กลับให้ผลผิดโดยไม่มี error**: context ถูกตัดเงียบ ๆ เหลือหนึ่งในสิบ, tool calling ที่เปิดไว้แต่
ไม่เคยแปลงคำตอบจริง, สาย 200G ที่ negotiate ลงมาเหลือ 50G, KV cache ที่คำนวณเกินจริงยี่สิบเท่า

LMDS เกิดจากการไล่รันของจริงแล้วเก็บทุกอาการพวกนี้กลับมาเป็นการตรวจอัตโนมัติ

| | |
|---|---|
| 🧮 **คำนวณด้วยโค้ด ไม่ใช่ LLM** | memory fit, KV cache, token budget, ความเร็วลิงก์ — LLM มีหน้าที่แค่วิจัยโมเดลและเลือกค่าใน Deployment Plan ที่เป็น JSON schema ตายตัว **ไม่เคยเขียน Bash เอง** |
| 🛡️ **ทุก bundle ผ่านด่านก่อนถึงมือคุณ** | `bash -n`, audit rules, SHA-256 checksums — ไม่ผ่านคือไม่มี ZIP |
| 🔌 **ทำงานได้โดยไม่มี LLM** | โหมด rule-based ใช้สูตรที่รันผ่านจริงมาแล้ว · air-gapped ก็ใช้ได้ |
| 🤝 **เครื่องที่มีโมเดลรันอยู่ก่อนแล้ว ไม่ต้องรื้อ** | `lmds adopt` อ่านคำสั่งที่มันรันอยู่จริง แล้วเขียนเป็น controller ที่รันซ้ำได้เป๊ะ — ไม่ต้อง redeploy ไม่ต้องโหลด weight ใหม่ |

## เริ่มใน 3 คำสั่ง

```bash
git clone https://github.com/neronain/AutoDeployDGXProject && cd AutoDeployDGXProject
./install.sh -y                      # ลง Docker / NVIDIA toolkit ที่ขาดให้ด้วย
lmds web --enable --bind 0.0.0.0     # คอนโซลที่ http://<ip>:8600 — ขึ้นเองหลังรีบูต · พิมพ์ token ให้
```

**เครื่องอื่นในฟลีตไม่ต้องติดตั้งเอง** — บนหน้าเว็บกด *Add machine* ใส่ host / user / รหัสผ่าน sudo
ครั้งเดียว: hub ใส่ SSH key ให้ แล้ว**ส่งโค้ดของตัวเองไปติดตั้ง** (git bundle ~2 MB ผ่าน scp — เครื่องนั้น
ไม่ต้องเข้าถึง GitHub เลย) · `install.sh` ล้มกลางทาง = รุ่นเดิมยังอยู่ ไม่ทิ้งเครื่องไว้แบบไม่มี `lmds`

ถนัด CLI มากกว่า: `lmds hardware` (เครื่องนี้คือ target อะไร) → `lmds deploy Qwen/Qwen3-32B`

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

# repo GGUF หลาย quant โดยไม่มี tty ให้เลือกหมายเลข (script / hub)
lmds deploy unsloth/gemma-4-26B-A4B-it-GGUF --gguf Q8_K_XL --yes

# embedding / reranker — ระบบเดาจาก repo เอง · เดาผิดบังคับด้วย --task
lmds deploy VesNFF/Qwen3-VL-Embedding-8B-GGUF --task embed
lmds deploy Qwen/Qwen3-Reranker-4B --target dgx-spark-single --no-llm
```

</details>

---

## สามอย่างที่ไม่ค่อยมีที่ไหนตอบให้

### 1 · "ตั้ง context เท่านี้แล้วจะมีกี่คนใช้พร้อมกันได้"

เครื่องมือทั่วไปตอบได้แค่ context สูงสุด ซึ่งตามนิยามคือค่าที่**คนเดียว**กิน KV pool หมดพอดี —
ตั้งตามนั้นแล้วคนที่สองต่อคิว โดยไม่มีอะไรบอก

```
KV bf16 · 120 KiB ต่อ token
  context      KV ต่อคน    พร้อมกัน
   32,768       3.8 GB       14.1
  131,072        15 GB        3.5
  262,144        30 GB        1.8   ← ค่าที่กรอก
```
> • ใส่ได้ แต่ได้ 1.8 คนพร้อมกัน — หนึ่งคำสนทนากิน KV pool เกือบหมด
> • เปลี่ยน KV เป็น fp8 → 30 GB เหลือ 15 GB · พร้อมกันจาก 1.8 เป็น 3.5 คน

ขึ้นทั้งใน CLI และ**ในหน้าเว็บระหว่างที่ยังพิมพ์เลขอยู่** · รองรับทั้ง GQA และ **MLA**
(DeepSeek-V2/V3, Kimi K2/K3) ซึ่งเก็บ KV เป็น latent ก้อนเดียว — สูตรเดียวใช้กับทุกตระกูลไม่ได้

### 2 · หลายเครื่อง = โมเดลเดียว

> **stacked ไม่ได้แปลว่าเร็วขึ้น — แปลว่าใหญ่เกินหนึ่งเครื่อง**
> โมเดลที่ลงเครื่องเดียวได้ รันเครื่องเดียวเร็วกว่าเสมอ

| | เครื่องเดียว | Stacked |
|---|---|---|
| Engine | vLLM · llama.cpp · SGLang | **vLLM เท่านั้น** |
| Artifact | safetensors หรือ GGUF | **safetensors เท่านั้น** |
| งาน | chat · vision · embedding · rerank | chat · vision |
| สายเชื่อม | ไม่ต้อง | **ต้องมี** ≥25G (ของจริง 200G RoCE) |
| จำนวนเครื่อง | 1 | ต่อตรง ≤3 · ผ่าน switch ยังไม่มีเพดานที่ยืนยันได้ |
| ที่ได้จริง | เร็วสุด | **หน่วยความจำ/KV/จำนวนคนพร้อมกันเพิ่ม** — ไม่ใช่ tok/s ต่อคน |

ระบบตรวจ ConnectX/RDMA ให้เอง บอกว่าเครื่องคู่ไหน stacked กันได้ เขียน `cluster.env` ให้
และ**เตือนเมื่อลิงก์ negotiate ได้ต่ำกว่าที่การ์ดทำได้** (NVIDIA ตรวจรับที่ ≥184 Gbit/s —
พอร์ตที่ปล่อย auto มักลงมาเหลือ 50G แล้วทุกอย่างยังดูปกติ)

fit รายงาน**ตัวเลขต่อเครื่อง** (capacity · OS · engine · NCCL buffer 3 GB/เครื่อง · weights/N · KV/N)
· `lmds cluster pair` สร้างกุญแจบน head ให้ head ssh เข้า worker ได้ · `lmds cluster doctor <head> <worker>`
ไล่ทีละข้อว่าทำไมคู่นี้ยังไม่ได้ · คู่ที่เป็นไปไม่ได้ถูกปฏิเสธตั้งแต่ analyze (422) ไม่ใช่ไปตายตอน push

**รันจริงแล้วบน 2× DGX Spark**: Llama 3.3 70B · `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4` 173 GB
(vLLM TP=2, tool calling ผ่าน) · `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`

→ [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) · [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) · [เทียบกับเอกสารของ NVIDIA](docs/NVIDIA-CLUSTER-SOURCES.md)

### 3 · หน้าเว็บที่ทำได้เท่า CLI

เมนูซ้าย: Overview · This machine · All machines · ต้นไม้ **ไซต์ → เครื่อง** · Library
(โมเดลทั้งฟลีต / คะแนน / สูตร / weights) · หน้าภาพรวมมีแถบหน่วยความจำต่อเครื่อง โดนัท engine
และ **"Needs attention"** ที่คำนวณจากข้อมูลจริง (ต่อไม่ได้ · เกิน 90% · พอร์ตซ้อน · commit ไม่ตรง hub)

ทำได้จากหน้าเว็บทั้งหมด: deploy wizard · download + verify · start/stop/restart · doctor · logs ·
ชุดทดสอบ (`test-text` `test-vision` `test-reasoning` `test-tools` `test-embed` `test-rerank` `bench` `stress`) ·
autostart · คำสั่ง stacked · repair · remove · ปุ่ม **Update** ทั้งฟลีต — **และคุมโมเดลบนเครื่องอื่น
ได้เท่ากับเครื่องตัวเอง**

- **อ่านสถานะได้ก่อนอ่านตัวหนังสือ** — เกจ CPU / Unified·RAM / VRAM / Disk ชุดเดียวกันทุกเครื่อง ·
  ค่าที่การ์ดไม่รายงานถูกซ่อน ไม่ใช่โชว์ 0
- **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่านจาก dispatch table ของสคริปต์เอง
- **จัดกลุ่มเครื่องตาม site** · เครื่องที่ stacked ด้วยกันได้มีรั้วสีคร่อม
- **ไม่ดึงอะไรจากอินเทอร์เน็ตเลย** — แม้แต่ฟอนต์ก็อยู่ในแพ็กเกจ · ใช้ได้หลัง proxy หรือ air-gapped

**ผู้ช่วยมุมขวาล่าง** ตอบจาก*สถานะจริงของฟลีตนี้* ไม่ใช่ความรู้ทั่วไป — และ**ลงไปดูเครื่องจริงก่อนตอบ**
(เปิด log ของ controller ตัวนั้น ดู GPU ดิสก์ พอร์ต หรือรัน `lmds doctor` ผ่าน SSH) แล้วค่อยตอบจากผลที่ได้
· มันรู้กติกาเรื่อง context/KV แต่**ถูกสั่งห้ามคิดเลขเอง** เพราะเลขที่ LLM คูณเองผิดแบบดูน่าเชื่อ
ซึ่งแย่กว่าตอบว่าไม่รู้

เมื่อสาเหตุชัดพอจะเสนอวิธีแก้ มันจะ**ถามกลับเป็นเมนู** — *แก้เลย* / *ทีละขั้น* / *ยังไม่ทำ* —
แทนที่จะลงมือเอง · **LLM สั่งงานเองไม่ได้**: มันเลือกได้แค่ชื่อรายการจากแคตตาล็อกที่กำหนดไว้
คำสั่งจริงประกอบด้วยโค้ด และตั๋วอนุมัติออกโดยเซิร์ฟเวอร์ ทางเดียวที่คำสั่งจะทำงานคือมีคนกดปุ่ม

---

## ความปลอดภัยที่เป็นค่าเริ่มต้น

| | |
|---|---|
| **คอนโซลต้องมี token เสมอ** | ไม่ว่า bind ที่ไหน · เดิม `127.0.0.1` ถูกปล่อยโล่ง ซึ่งเปิดให้ผู้ใช้อื่นบนเครื่องเดียวกัน และเพจใดก็ได้ที่เปิดในเบราว์เซอร์ (CSRF/DNS rebinding) สั่งได้ · `--no-auth` เปิดโล่งได้แต่ต้องสั่งเอง |
| **โมเดลที่ deploy ใหม่มี API key ตั้งแต่เกิด** | เก็บที่ `~/.lmds/keys/<slug>` (0600) **ไม่ได้อยู่ในโฟลเดอร์ bundle** ซึ่งถูก zip แจกต่อได้ · controller อ่านเองตอน start จึงไม่หายหลัง reboot |
| **ร่องรอยว่าใครสั่งอะไร** | `lmds audit` — เวลา · IP · คำสั่ง · ผล · เก็บเฉพาะคำสั่งที่เปลี่ยนสถานะกับคำขอที่ถูกปฏิเสธ · ไม่เก็บ body และ query string |
| **ปักหมุดเวอร์ชันได้** | `export LMDS_REPO_REF=v0.9.1` บน hub → ทุกเครื่องได้ tag นั้นตรง ๆ ไม่ใช่ปลาย branch |
| **key ไม่เคยอยู่บน argv** | llama.cpp ใช้ไฟล์ 0600 ผ่าน `--api-key-file` · vLLM ผ่าน env · `lmds key set` รับทาง stdin |

```bash
lmds key new <slug>      # สุ่ม key ใหม่ให้โมเดลนี้      lmds key show <slug> --reveal
lmds audit --failed      # ใครถูกปฏิเสธบ้าง — ไล่เดา token เห็นเป็นชุดจาก IP เดียวกัน
lmds doctor <slug>       # ข้อ endpoint บอกถ้ายังเสิร์ฟแบบเปิดอยู่
```

> **bundle ที่ติดตั้งไปก่อนหน้านี้ไม่ถูกแตะ** — ยังเสิร์ฟแบบเดิมจนกว่าจะสั่ง `lmds key new` เอง
> (ถ้าแจก key ให้อัตโนมัติ client ทุกตัวของลูกค้าจะพังพร้อมกันหลัง update)

รายละเอียดทั้งหมด: [SECURITY.md](SECURITY.md)

## คุมทั้ง fleet จากเครื่องเดียว

```bash
lmds node add 192.168.10.21 --user ops --install   # ถามรหัสผ่านครั้งเดียว → ติดตั้ง key + LMDS ให้
lmds ps --all                     # โมเดลของทุกเครื่องในตารางเดียว
lmds fleet check --check          # ทุกเครื่องตรง hub ครบ 3 มิติไหม (code · controller · runtime)
lmds cluster show                 # เครื่องไหนมี 200G และจับคู่ stacked กันได้
lmds cluster pair spark-head spark-worker          # ให้ head ssh เข้า worker ได้
lmds scan --all                   # weight ที่มีอยู่แล้วบนทุกเครื่อง — ไม่ต้องโหลดซ้ำ
lmds node push spark2 <slug>      # ส่ง bundle ตัวที่อนุมัติแล้วไปติดตั้งเครื่องอื่น
lmds node clone <slug> --from msi-1 --to msi-2     # สำเนาโมเดลข้ามเครื่อง ไม่โหลดจาก HF ใหม่
```

> **`node clone` — ทำตัวสำรอง/กระจายโหลดโดยไม่โหลดใหม่ทุกครั้ง**
>
> โมเดล 90 GB ที่โหลดจาก Hugging Face ใช้ 38 นาที · เครื่องข้าง ๆ ในแร็คถือไฟล์ชุดเดียวกันอยู่แล้ว
> — **วัดจริงบนฟลีต: 412 MB/s จบใน 3 นาที 47 วิ เร็วกว่า 10 เท่า** · ไฟล์วิ่งตรงระหว่างสองเครื่อง
> ไม่ผ่าน hub และกุญแจไม่เคยออกจาก hub (สร้างชั่วคราวต่อครั้ง ส่งทาง stdin เข้า `ssh-agent` แล้วถอนออกเสมอ)

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
lmds set <ชื่อ> --image <digest> --tool-parser qwen3_xml --extra-args "…"
lmds adopt <container> / --port N   # รับโมเดลที่รันอยู่ก่อน LMDS เข้ามาในระบบ
lmds remove <ชื่อ>        # ลบทั้งหมด (--keep-weights = เก็บ weight)
lmds recipes             # สูตรที่รันผ่านจริง — ใช้เองเมื่อไม่มี API key
```

`lmds ps` เห็น **container ที่ไม่ได้ deploy ผ่าน LMDS** ด้วย (vLLM/llama.cpp/Ollama/TGI ที่รันอยู่แล้ว)
— stop/restart/logs/enable ได้เหมือนกัน โดยกลุ่มนี้ใช้ `docker stop` ไม่ลบ container ทิ้ง

</details>

## คลังสูตร — เรียนรู้ครั้งเดียว ใช้ได้ทั้งกอง

เครื่องที่ไม่มี API key ของ LLM จะ deploy แบบ rule-based ซึ่งรู้แค่ "GGUF → llama.cpp" ไม่รู้เรื่อง
เฉพาะรุ่น (parser, image ที่มี kernel ตรง, mmproj) — deploy ผ่านแต่ start ไม่ขึ้น · **คลังสูตร**
เก็บ controller ที่ **รันผ่านจริงบนฮาร์ดแวร์แล้ว** ไว้ในรีโป Git กลาง

```bash
lmds recipes --sync                                      # ดึงสูตรล่าสุดมาใช้แทนการเดา
lmds recipes --publish <ชื่อ> --features tools,vision     # ส่งตัวที่เทสต์ผ่านขึ้นคลัง รอ review
```

สองชั้น: **canonical** ([`dgx-spark-all-controllers`](https://github.com/neronain/dgx-spark-all-controllers))
ที่ curate แล้ว และ **candidates** ([`script-update`](https://github.com/neronain/script-update)) ที่รอ review ·
ปลายทาง publish ตั้งใน config — **ว่าง = local store ในเครื่อง** ปลอดภัยสำหรับลูกค้า (ฟลีตแชร์กันเองโดยไม่แตะรีโปเรา)

> ส่งเฉพาะ**ค่าของโมเดล** (engine, image, parser, mmproj) — **ค่าของเครื่อง** (port, context, slots)
> อยู่ใน `bundle.env` ไม่ตามขึ้นไป เครื่องปลายทาง fit ใหม่ตามตัวเอง

## รองรับอะไรบ้าง

| | ARM64 / unified (Spark) | x86_64 / discrete (RTX) |
|---|---|---|
| **llama.cpp** | ✅ native build (`start` build ให้เอง) | ✅ docker (+ multimodal) |
| **vLLM** | ✅ docker · ✅ stacked 2 เครื่อง | ✅ docker |
| **SGLang** | ✅ docker (`--engine sglang`) | ✅ docker |

| งาน | llama.cpp (GGUF) | vLLM (safetensors) | stacked |
|---|---|---|---|
| chat / tool calling / reasoning | ✅ (`--jinja`) | ✅ (`--tool-parser` `--reasoning-parser`) | ✅ |
| vision | ✅ mmproj (+ `--image-min-tokens`) | ✅ | ✅ |
| embedding | ✅ `--embedding --pooling` | ✅ `--runner pooling` | ❌ ปฏิเสธ |
| rerank | ✅ `--reranking` | ✅ `--runner pooling --convert classify` | ❌ ปฏิเสธ |
| MTP / speculative | ✅ draft head จาก repo | ผ่าน `--extra-args` | ผ่าน `--extra-args` |

ผ่าน hardware validation ครบทั้ง 5 ตระกูลโมเดล — GGUF, NVFP4, MoE, dense safetensors, gated repo ·
**22 target preset** (7 ตัวทดสอบบนเครื่องจริงแล้ว) · **2,224 เทสต์** รันครบทุก push บน Python 3.10–3.13

**MoE กับ MTP ถูกรายงานเป็นข้อเท็จจริงจากไฟล์** ไม่ใช่สิ่งที่ LLM เดา — จำนวน expert ทั้งหมด/ที่เปิด
ต่อ token อ่านจาก `config.json` หรือ GGUF metadata เพราะ *total บอกว่าต้องมีหน่วยความจำเท่าไร ส่วน
active บอกว่าจะได้ความเร็วเท่าไร* · repo ที่แถม MTP draft head มาจะถูกต่อสายให้อัตโนมัติ
(วัดจริงบน DGX Spark: gemma4-26B-A4B ได้ **1.78x** โดย output เท่าเดิม)

> **แหล่งโมเดล: Hugging Face เท่านั้น** — Ollama registry และ NVIDIA NGC อยู่ในเฟส 2 · HF ย้ายไฟล์ใหญ่
> ไป **Xet** แล้ว — สตรีมเดี่ยวจากไทยได้ ~0.3 MB/s แต่ controller llama.cpp โหลดขนาน 8 ส่วนได้ ~50 MB/s

## อัปเดต

```bash
cd ~/AutoDeployDGXProject && git pull && ./install.sh     # hub ก่อนเสมอ — หรือกดปุ่ม Update บนหน้าเว็บ
lmds node install --all                                  # แล้วค่อยทั้งฟลีต — hub ส่งโค้ดไปให้เอง ไม่แตะ GitHub
lmds fleet check --check                                 # ทุกเครื่องตรง hub จริงไหม (ต่อเข้าไปดูสด ๆ)
```

> ⚠️ **อัปเดต hub ก่อน** — `lmds node install --all` ส่ง **โค้ดของ hub** ไปให้ node
> ถ้า hub ยังเก่า ทั้งฟลีตจะ "ตรง hub" ครบแต่ตรงกับของเก่า
>
> ⚠️ **`git pull` อย่างเดียวไม่พอ** — ติดตั้งแบบ copy เข้า venv คำสั่ง `lmds` จะยังเป็นโค้ดเก่าจนกว่าจะรัน
> `./install.sh` ซ้ำ · config และ key เดิมอยู่ครบ · `install.sh` ย้าย venv เดิมไว้ก่อนแล้วคืนให้ถ้า pip ล้ม

ไซต์ที่ต้องล็อกเวอร์ชัน: `export LMDS_REPO_REF=v0.9.1` บน hub แล้วสั่ง `lmds node install` ตามปกติ

## ใช้คู่กับ LiteGate (ทางเลือก)

**[LiteGate · AiGatewayLocal](https://github.com/neronain/AiGatewayLocal)** คืออีกครึ่งของชุดนี้ —
LMDS *deploy* โมเดลลงเครื่องคุณ ส่วน LiteGate เป็น *ประตูเดียว* หน้าโมเดลทั้งหมด: API key, โควตา,
สิทธิ์ต่อคน และตรวจว่าเซิร์ฟเวอร์ที่รันอยู่**ทำอะไรได้จริง**

| ติดตั้ง | ได้อะไร |
|---|---|
| LMDS อย่างเดียว | deploy และรันโมเดลบนเครื่องตัวเอง มีหน้าเว็บและผู้ช่วยครบ |
| LiteGate อย่างเดียว | ประตูเดียว + key + โควตา หน้าเซิร์ฟเวอร์ที่รันมาด้วยวิธีไหนก็ได้ |
| **ทั้งคู่** | LMDS สร้าง · LiteGate วัดของจริงแล้วบอกคำสั่งที่ต้องแก้ |

**ไม่มีตัวไหนต้องพึ่งอีกตัว** · parser ที่ LiteGate บอกว่าขาดคือ knob ที่ LMDS เปิดได้ทันทีด้วย
`restart --tool-parser` แล้วพิสูจน์ด้วย `test-tools` ซึ่งวัดโหมด `auto` — โหมดเดียวกับที่ agent ใช้จริง

## ระบบทั้งหมด — 4 repo ทำงานร่วมกัน

| Repository | บทบาท |
|---|---|
| **[AutoDeployDGXProject](https://github.com/neronain/AutoDeployDGXProject)** (LMDS) | โหลด weight, วิเคราะห์, สร้าง controller, deploy + รันโมเดลทั้งฟลีตผ่าน SSH |
| **[AiGatewayLocal](https://github.com/neronain/AiGatewayLocal)** (LiteGate) | Endpoint OpenAI/Anthropic เดียวหน้าโมเดลทั้งหมด พร้อม key/quota/สิทธิ์ |
| **[dgx-spark-all-controllers](https://github.com/neronain/dgx-spark-all-controllers)** (canonical) | Controller ที่ curate + ตรวจแล้ว ทุกเครื่องดึงไปใช้ |
| **[script-update](https://github.com/neronain/script-update)** (candidates) | Controller ใหม่ที่เพิ่ง publish รอ review ก่อน promote |

LMDS deploy ด้วย controller ที่สร้างจากการทดลองจริง → ตัวที่พิสูจน์แล้วส่งไป candidates รอ review →
promote ขึ้น canonical → ทุกเครื่องในฟลีต sync ไปใช้ · LiteGate วัดความสามารถจริงแล้วส่งคำแนะนำแก้กลับมา

## เอกสาร

| | |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | ติดตั้งทีละขั้น — prerequisites, ดิสก์, proxy/air-gapped, ตั้ง provider, ถอนการติดตั้ง |
| [USAGE.md](docs/USAGE.md) | คู่มือใช้งานเต็ม — deploy, คำสั่ง controller ทุกตัว + env, fleet, หน้าเว็บ, troubleshooting |
| [SECURITY.md](SECURITY.md) | ข้อมูลอะไรออกนอกเครื่อง, secret เก็บที่ไหน, auth/audit, แจ้งช่องโหว่ |
| [BENCH.md](docs/BENCH.md) | ให้คะแนนโมเดลที่รันอยู่ — ความเร็ว + ความสามารถ 7 ข้อ วัดจากเซิร์ฟเวอร์จริง |
| [PREFLIGHT.md](docs/PREFLIGHT.md) | สิ่งที่ระบบตรวจให้ก่อน deploy และทำไม — ทุกข้อมาจากของที่พังจริง |
| [NETWORK.md](docs/NETWORK.md) | พอร์ตและโปรโตคอลทุกตัวที่ระบบใช้ ใครคุยกับใคร |
| [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) · [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) | ลำดับคำสั่งข้ามเครื่องที่รันจริงแล้ว · คุมหลายเครื่องจากเครื่องเดียว |
| [NVIDIA-CLUSTER-SOURCES.md](docs/NVIDIA-CLUSTER-SOURCES.md) | เอกสารคลัสเตอร์ของ NVIDIA — อะไรยืนยันของเรา อะไรเติมของใหม่ |
| [LICENSING.md](docs/LICENSING.md) | ระบบไลเซนส์ทำงานยังไง — นับเครื่องแบบไหน และ **อะไรที่ไม่มีวันล็อก** |
| [RELEASE.md](docs/RELEASE.md) · [CHANGELOG.md](CHANGELOG.md) · [CONTRIBUTING.md](CONTRIBUTING.md) | ขั้นตอนปล่อยรุ่น · ประวัติการเปลี่ยนแปลง · ตั้ง dev env + กฎที่ห้ามละเมิด |
| [PRD.md](docs/PRD.md) · [CLI_SPEC.md](docs/CLI_SPEC.md) · [ROADMAP.md](docs/ROADMAP.md) | ข้อกำหนด, สเปกคำสั่ง, แผนพัฒนา |

## Requirements

- **Ubuntu 22.04 / 24.04 / 25.04** (ARM64 หรือ x86_64) — พัฒนาบน macOS ได้
- **Python 3.10–3.13** — CI ทดสอบครบทั้งสี่รุ่นทุก push (24.04 มาพร้อม 3.12 · 25.04 มาพร้อม 3.13)
- **Docker + NVIDIA Container Toolkit** บนเครื่องเป้าหมาย (`install.sh` ลงให้ได้)
- **git + python3** บนเครื่อง node — hub ส่งโค้ดเป็น git bundle ไปให้ ไม่ต้องมีสิทธิ์เข้า GitHub
- **ดิสก์ว่าง** ≈ *(ขนาดโมเดล × 1.2) + 25 GB* — runtime image ของ vLLM อย่างเดียว ~10–20 GB
- **stacked**: สายเร็ว ≥25G ระหว่าง DGX Spark + head ssh ถึง worker (`lmds cluster pair` ทำให้)
- **LLM provider** (ทางเลือก): OpenAI / Gemini / MiniMax / OpenAI-compatible — ไม่มีเลยก็ใช้ `--no-llm`

ข้อเดียวที่ `install.sh` ไม่ทำให้คือ **NVIDIA driver** เพราะต้อง reboot และบางเครื่องมี driver ที่ใช้ได้
อยู่แล้วแต่ `ubuntu-drivers install` ชน dependency จนพัง

## สำหรับผู้พัฒนา

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest
ruff check src tests scripts     # ด่านเดียวกับที่ CI ใช้
```

กฎที่ห้ามละเมิดและวิธีเพิ่ม target preset / provider / quality gate: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

**1 เครื่อง = ฟรี** ใช้ทำอะไรก็ได้รวมถึงเชิงพาณิชย์ · **ตั้งแต่ 2 เครื่องที่บริหารร่วมกัน = ต้องมี license**
ดู [LICENSE](LICENSE) · [เงื่อนไขเชิงพาณิชย์](docs/COMMERCIAL.md) · [ของบุคคลที่สาม](THIRD-PARTY-LICENSES)

ซอร์สอ่านได้ แต่ไม่ใช่ open source — ฟอร์ก แจกจ่ายต่อ หรือขายต่อไม่ได้ · **bundle ที่ผู้ใช้ generate
ออกมาเป็นของผู้ใช้เอง** ใช้/แก้/ส่งต่อได้อิสระ · โมเดล image และ runtime ของบุคคลที่สามอยู่ใต้
license ของเจ้าของนั้น ๆ

<div align="center">
<br>

สืบทอดมาตรฐาน controller จาก [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers)

**neronain** · [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>
