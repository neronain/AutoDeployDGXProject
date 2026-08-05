# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

> ⚡ สร้างและดูแลโดย **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)
>
> 🇬🇧 English summary: [README.en.md](README.en.md)

โปรแกรม CLI บน Ubuntu ที่รับ **ลิงก์โมเดล Hugging Face** (repo, ลิงก์ไฟล์ `.gguf` ตรง) แล้วใช้ **LLM API** (OpenAI, Gemini, MiniMax หรือ OpenAI-compatible endpoint — รวมถึงโมเดล local ของคุณเอง) เป็นสมองในการวิเคราะห์และ**สร้างชุดสคริปต์ deploy ที่ผ่านการ validate แล้ว** สำหรับ:

- **NVIDIA DGX Spark** — เครื่องเดี่ยว หรือ stacked หลายเครื่อง
- **Ubuntu + RTX GPU** — local AI server ทั่วไป (x86_64)

สืบทอดมาตรฐานจาก [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers) และ skill pack `dgx-spark-model-deployer-team-pack-v3.0.0` ซึ่งมี controller ที่รันจริงแล้วกว่า 12 โมเดล

> **แหล่งโมเดลที่รองรับตอนนี้: Hugging Face เท่านั้น** — ลิงก์ Ollama (`ollama.com/...`) และ NVIDIA NGC อยู่ใน roadmap เฟส 2 (ใส่แล้วระบบจะแจ้งว่ายังไม่รองรับ พร้อมแนะให้ใช้ลิงก์ HF ของ GGUF ตัวเดียวกันแทน) · provider `anthropic` ตั้งค่าได้แล้วแต่ adapter ยังอยู่ในเฟส 2

## สถานะโปรเจกต์

✅ **เฟส 1 — CLI MVP ผ่านเกณฑ์ครบทุกข้อแล้ว (3 ส.ค. 2569)**

hardware-validated **6 ครั้ง บน 2 เครื่อง** (DGX Spark + RTX 5090) ครอบคลุมครบทั้ง 5 ตระกูลโมเดล —
GGUF, NVFP4, MoE, dense safetensors, gated repo — และครบทั้ง 4 ช่องของเมทริกซ์ engine × สถาปัตยกรรม:

| | ARM64 / unified (Spark) | x86_64 / discrete (RTX) |
|---|---|---|
| llama.cpp | ✅ native build | ✅ docker (+ multimodal) |
| vLLM | ✅ docker | ✅ docker |

รายละเอียดต่อโมเดล: [ROADMAP.md](docs/ROADMAP.md) · สิ่งที่แก้จากการรันจริง: [CHANGELOG.md](CHANGELOG.md)

## 📖 เริ่มที่นี่

| ผู้อ่าน | เอกสาร |
|---|---|
| **ผู้ติดตั้ง/ทีมงาน/ลูกค้า** — ติดตั้งครั้งแรก | **[docs/INSTALL.md](docs/INSTALL.md)** — เตรียมเครื่อง (Docker, NVIDIA toolkit), ติดตั้ง, ตั้งค่า ทีละขั้น |
| **ผู้ใช้งาน** — deploy โมเดล | **[docs/USAGE.md](docs/USAGE.md)** — คู่มือใช้งานละเอียด + ตาราง troubleshooting |
| ผู้พัฒนา/ผู้ตัดสินใจ | [docs/PRD.md](docs/PRD.md), [docs/CLI_SPEC.md](docs/CLI_SPEC.md), [docs/ROADMAP.md](docs/ROADMAP.md) |

## หลักการออกแบบสำคัญ

> **Deterministic core + LLM assist** — LLM ไม่เขียน Bash เอง ทุกสคริปต์ render จาก template ที่ผ่านการตรวจแล้ว LLM ทำหน้าที่แค่วิจัยโมเดลและเลือกค่าใน Deployment Plan (JSON schema ตายตัว) ส่วนการคำนวณ memory fit / token budget ทำด้วยโค้ด 100% และทุก bundle ต้องผ่าน quality gates (`bash -n`, audit rules, SHA-256) ก่อนถึงมือผู้ใช้

## ตัวอย่างการใช้งาน

```bash
./install.sh                          # ติดตั้ง — ลง Docker/NVIDIA toolkit ที่ขาดให้ด้วย (ถามก่อนทุกขั้น)
source ~/.bashrc                      # สคริปต์บอกเองตอนจบว่าต้องรันอะไร

lmds hardware                         # ตรวจเครื่อง + จำแนก target profile
lmds inspect Qwen/Qwen3-32B --target rtx-pro-4000-dual    # วิเคราะห์ + fit โดยไม่ generate
lmds deploy https://huggingface.co/Qwen/Qwen3-32B --target dgx-spark-single
# → วิเคราะห์ → วางแผน → ยืนยัน (อนุมัติ flag/แก้ context ได้) → bundle + ZIP ที่ผ่าน quality gates ทุกด่าน

# โมเดล gated → ระบบถาม HF token อัตโนมัติ (กด Enter ข้ามได้)
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single

# โมเดลใหญ่เกิน 1 เครื่อง → stacked (2× DGX Spark, multi-node controller: worker-first + sync-worker)
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 --target dgx-spark-stacked
```

## อัปเดตเวอร์ชัน (เครื่องที่ `git clone` ไว้แล้ว)

```bash
cd ~/AutoDeployDGXProject && git pull && ./install.sh
```

> ⚠️ **`git pull` อย่างเดียวไม่พอ** — ต้องรัน `./install.sh` ซ้ำด้วย เพราะติดตั้งแบบ copy เข้า venv (ไม่ใช่ editable) คำสั่ง `lmds` เลยยังเป็นโค้ดเก่าจนกว่าจะติดตั้งใหม่ทับ · config/key เดิมอยู่ครบ ไม่ต้องตั้งใหม่

ตรวจว่าอัปเดตแล้ว (ควรเห็นคำสั่ง `repair` / `remove` / `restart`):

```bash
lmds --help
```

ถ้า `git pull` ฟ้อง local changes: `git stash && git pull && ./install.sh` (การแก้ PATH ครั้งก่อนอยู่ใน `~/.bashrc` ไม่ใช่ในโปรเจกต์ จึงไม่กระทบ) · รายละเอียด/ถอนการติดตั้ง: [docs/INSTALL.md](docs/INSTALL.md)

## จัดการหลายโมเดลในเครื่องเดียว (Fleet)

รันกี่โมเดลก็ได้ (คนละ port) แล้วคุมทั้งหมดจาก `lmds` — ไม่ต้องจำว่า bundle ไหนอยู่ที่ไหน:

```bash
lmds ps                  # ใครรันอยู่บ้าง: ชื่อ, โมเดล, engine, port, สถานะ ● running / ◐ loading / ○ stopped
lmds list                # bundle ทั้งหมด + สถานะ + engine/port/context/ฟีเจอร์ + autostart
lmds start <ชื่อ>         # รันโมเดลที่เคย deploy ไว้ขึ้นมาใหม่ (เช่น หลัง reboot)
lmds start <ชื่อ> --port 8001   # flag ที่ไม่ใช่ของ lmds ส่งต่อให้ controller เลย
lmds stop <ชื่อ>          # หยุดตามชื่อ — ไม่ต้อง cd ไปหา ./xxx.sh stop · stop --all = หยุดทุกตัว
lmds restart <ชื่อ>       # restart (ใช้ตอนเปลี่ยน option เช่นเพิ่ม API_KEY)
lmds logs <ชื่อ> -f       # ตาม log แบบ realtime (-n 500 = ย้อนหลัง)
lmds enable <ชื่อ>        # ให้โมเดลกลับมาเองหลังเปิด-ปิดเครื่อง (systemd autostart) · disable = ยกเลิก
lmds doctor <ชื่อ>       # ตรวจว่าทำไมยัง download/start ไม่ผ่าน + คำสั่งแก้
lmds repair <ชื่อ>        # โหลดไฟล์ที่ขาด/เสียกลับมา แล้วตรวจซ้ำ
lmds remove <ชื่อ>        # ลบออกจากเครื่องทั้งหมด (--keep-weights = เก็บ weight ไว้)
```

## เครื่องเดียว หรือ หลายเครื่อง?

**stacked ไม่ได้แปลว่าเร็วขึ้น — แปลว่าใหญ่เกินหนึ่งเครื่อง** โมเดลที่ลงเครื่องเดียวได้
รันเครื่องเดียวเร็วกว่าเสมอ เพราะไม่ต้องส่ง activation ข้ามสายทุก token

| | เครื่องเดียว | Stacked (หลายเครื่อง = โมเดลเดียว) |
|---|---|---|
| Engine | vLLM **หรือ** llama.cpp | **vLLM เท่านั้น** |
| Artifact | safetensors หรือ **GGUF** | **safetensors เท่านั้น** — GGUF stacked ไม่ได้ |
| ต้องมีสายเร็ว | ไม่ต้อง | **ต้องมี** ≥25G (จริงคือ 200G RoCE) |
| target | `dgx-spark-single` · `rtx-5090` … | `dgx-spark-stacked` · `dgx-spark-stacked-4` |

ลำดับคำสั่งเต็มของ stacked: **[docs/RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md)**

## คุมหลายเครื่องจากเครื่องเดียว (Multi-node Fleet)

หน้างานที่มีมากกว่า 1 เครื่อง ไม่ต้อง ssh ไล่ทีละตัว — เพิ่มเครื่องด้วย ip/user/รหัสผ่าน **ครั้งเดียว**
ระบบติดตั้ง SSH key ให้แล้วทิ้งรหัสผ่านทันที (ทะเบียนไม่มีฟิลด์รหัสผ่านโดยตั้งใจ)

```bash
lmds node add 192.168.10.21 --user ops --install   # ถามรหัสผ่านครั้งเดียว → ติดตั้ง key + ลง LMDS ให้
lmds node list --check                   # เครื่องไหนยังตอบบ้าง
lmds ps --all                            # โมเดลของทุกเครื่องในตารางเดียว
lmds node run spark2 doctor my-model     # สั่งคำสั่ง lmds อะไรก็ได้ข้ามเครื่อง
lmds node cluster                        # เครื่องไหนมี ConnectX/200G และจับคู่ stacked กันได้
lmds scan --all                          # โมเดลที่มีอยู่แล้วบนทุกเครื่อง (ไม่ต้องโหลดซ้ำ)
lmds node ctl spark1 <slug> start        # สั่งสคริปต์ controller บนเครื่องนั้นโดยตรง
lmds prune                               # ล้างทะเบียนค้างของ bundle ที่ลบไปแล้ว
lmds recipes                             # สูตรที่รันผ่านจริง — ใช้เองเมื่อไม่มี API key ของ LLM
```

- เครื่องปลายทาง **ไม่ต้องรัน daemon** และไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22 — hub เรียก `lmds agent info` ผ่าน SSH
- ⚠️ แต่**ทุกเครื่องต้องมี LMDS ติดตั้งอยู่** ("agent" = ตัวคำสั่ง `lmds` เอง) — สั่งจาก hub ได้ด้วย
  `lmds node install <ชื่อ>` ไม่ต้อง ssh เข้าไปเอง
- **ไม่ต้องใช้ root** — user ที่อยู่ในกลุ่ม `docker` พอ
- เห็นทรัพยากรสดของทุกเครื่อง: CPU · RAM/Unified · VRAM · ดิสก์ · ความเร็วสาย · **จำนวนโมเดลที่รัน**
  (llama.cpp รันหลายตัวพร้อมกันได้)
- ตรวจ **ConnectX/RDMA/ความเร็วลิงก์** ให้เอง แล้วบอกว่าเครื่องคู่ไหน stacked ด้วยกันได้ ·
  กรอก cluster IP ที่ NCCL จะใช้ แล้วสั่ง `lmds node cluster --write <slug>` เขียนลง bundle ได้เลย

รายละเอียด: [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md)

## หน้าเว็บ (ทางเลือก)

ไม่ถนัด CLI หรืออยากให้ทีมดูสถานะได้ — เปิดหน้าเดียวที่คุมได้ครบ: สถานะเครื่อง, โมเดลทั้งหมด,
start/stop/restart, doctor, logs

```bash
lmds web                                  # เปิดที่ http://127.0.0.1:8600 (เครื่องนี้เท่านั้น)
lmds web --bind 0.0.0.0 --port 8600       # ให้ทั้งวง network เข้าได้ — สุ่ม token ให้อัตโนมัติ
```

```bash
lmds web --background                     # รันเบื้องหลัง — terminal ว่างใช้ CLI ต่อได้ทันที
lmds web --status                         # ลืมลิงก์/token? ถามตัวที่รันอยู่
lmds web --restart -b                     # เปิดใหม่ + token ใหม่
lmds web --stop                           # หยุดตัวที่รันเบื้องหลัง
```

> 🔒 หน้านี้**สั่ง start/stop โมเดลได้** จึง bind `127.0.0.1` เป็นค่าเริ่มต้น · ถ้าเปิดออก network
> ระบบจะสุ่ม token ให้เองแล้วพิมพ์ลิงก์พร้อม token มาให้ (ตั้งเองด้วย `--token` ได้)
> · หน้าเว็บไม่ดึงอะไรจากอินเทอร์เน็ตเลย ใช้ได้บนเครื่องหลัง proxy/air-gapped

**หน้าเว็บเป็นภาษาอังกฤษ** (ตัว CLI ยังเป็นไทย) และทำได้เทียบเท่า CLI แล้ว — deploy wizard,
download (+verify อัตโนมัติ), start/stop/restart, ตั้ง port/context/slots/API key/bind, doctor, logs,
ชุดทดสอบ (`test-text`, `test-vision`, `bench`, `stress`, …), autostart, คำสั่ง stacked, repair,
remove (แสดงรายการ + ขนาดก่อนยืนยัน)

> **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่านจาก dispatch table ของสคริปต์เอง bundle เก่า
> ที่ยังไม่มีคำสั่งใหม่จะไม่มีปุ่มนั้นให้กดแล้วล้ม · `enable`/`disable` autostart ต้องใช้ `sudo` ซึ่ง
> หน้าเว็บไม่มี tty — ถ้าทำไม่ได้จะบอกคำสั่งให้ไปรันเอง · ยังต้องใช้ CLI: `lmds config`, `lmds hardware`

`lmds ps` เห็น **container ที่ไม่ได้ deploy ผ่าน LMDS** ด้วย (vLLM/llama.cpp/Ollama/TGI ที่รันอยู่แล้ว)
— stop/restart/logs/enable ได้เหมือนกัน โดย stop ของกลุ่มนี้ใช้ `docker stop` ไม่ลบ container ทิ้ง

กด TAB เติมชื่อคำสั่ง/bundle ได้ (`install.sh` ถามให้ หรือรัน `lmds --install-completion`)

ทุก controller ลงทะเบียนตัวเองอัตโนมัติตอน `start` — ต่อให้ bundle ถูกลบไปแล้ว `lmds stop` ก็ยัง
fallback หยุดโมเดลค้างให้ได้ (kill pid / docker rm) · รายละเอียด: [docs/USAGE.md §4](docs/USAGE.md)

## การติดตั้งเตรียมเครื่องให้เอง

`./install.sh` ไม่ได้แค่ตรวจ — **ติดตั้งของที่ขาดให้เลย** โดยถามยืนยันก่อนทุกขั้นที่ใช้ `sudo`
และพิมพ์คำสั่งจริงให้เห็นก่อนรัน: Docker, กลุ่ม `docker` ของ user, NVIDIA Container Toolkit (ครบ 5 ขั้น),
โมดูล `python3-venv` แล้วทดสอบว่า Docker เห็น GPU จริง · ต่อด้วยถามตั้ง LLM provider + API key +
tab completion · ตอบ `n` ข้ามได้ทุกข้อ แล้วสรุปตอนจบว่าเหลืออะไรต้องทำเอง

```bash
sudo -v && LMDS_ASSUME_YES=1 ./install.sh    # ติดตั้งรวดเดียวไม่ต้องนั่งตอบ
LMDS_SKIP_PREREQ=1 ./install.sh              # ลง LMDS อย่างเดียว ไม่แตะ Docker/toolkit
```

ข้อเดียวที่ไม่ทำให้คือ **NVIDIA driver** — ต้อง reboot และบางเครื่องมี driver ใช้ได้อยู่แล้วแต่
`ubuntu-drivers install` ชน dependency จนพัง · เมื่อไม่ได้รันบน terminal จริง (CI, pipe) จะไม่แตะเครื่องเลย
· รายละเอียด: [docs/INSTALL.md §2](docs/INSTALL.md)

ผลลัพธ์: โฟลเดอร์ bundle + ZIP ประกอบด้วย controller script (มาตรฐาน v3.0.0), `README.md`, `MODEL_PROFILE.yaml`, `SPECIAL_FILES.md`, `PACKAGE_SHA256SUMS`

## เอกสารทั้งหมด

| เอกสาร | เนื้อหา |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | **คู่มือติดตั้งละเอียด** — prerequisites, ดิสก์/ที่เก็บไฟล์, proxy/air-gapped, ตั้ง provider (รวม Local AI), โมเดลถูกดึงมาและรันยังไง, smoke test, ถอนการติดตั้ง |
| [docs/USAGE.md](docs/USAGE.md) | **คู่มือใช้งานละเอียด** — deploy ตั้งแต่โมเดลเล็กถึง gated repo, คำสั่ง controller ทุกตัว + env, fleet (ps/list/restart/logs -f/repair/remove/completion), target presets ครบ 20 ตัว, troubleshooting |
| [docs/RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) | **ลำดับคำสั่งรันข้ามหลายเครื่องที่ผ่านการรันจริง** — ตั้งแต่ `node add` ถึง `test-text` พร้อมตัวเลขหน่วยความจำ/KV cache จริง เวลาที่ใช้แต่ละขั้น และอาการเสียที่พบบ่อย |
| [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) | **คุมหลายเครื่องจากเครื่องเดียว** — เพิ่มเครื่องด้วย ip/user/รหัสผ่านครั้งเดียว, ดู CPU/RAM/VRAM/ดิสก์/โมเดลที่รันของทุกเครื่อง, ตรวจ ConnectX/200G และจับคู่เครื่องที่ stacked ด้วยกันได้ |
| [docs/PRD.md](docs/PRD.md) | Product Requirements Document ฉบับเต็ม — เป้าหมาย, user stories, functional requirements, สถาปัตยกรรม, security, risks |
| [docs/CLI_SPEC.md](docs/CLI_SPEC.md) | สเปกคำสั่ง CLI ทั้งหมดของเฟส 1 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | แผนพัฒนา 3 เฟส + work breakdown ของเฟส 1 |
| [SECURITY.md](SECURITY.md) | ข้อมูลอะไรออกนอกเครื่องบ้าง, secret เก็บที่ไหน, จุดที่ต้องอนุมัติเอง, แจ้งช่องโหว่ |
| [CONTRIBUTING.md](CONTRIBUTING.md) | ตั้ง dev env, กฎที่ห้ามละเมิด, วิธีเพิ่ม target preset / provider / gate |
| [CHANGELOG.md](CHANGELOG.md) | ประวัติการเปลี่ยนแปลง |

## Requirements (ตัวโปรแกรม)

- Ubuntu 22.04 / 24.04 (ARM64 หรือ x86_64) — พัฒนาบน macOS ได้
- Python 3.10+
- LLM provider อย่างน้อย 1 ทาง: OpenAI / Gemini / MiniMax / OpenAI-compatible (Ollama, vLLM local — ไม่ต้องมี key) — หรือไม่มีเลยก็ใช้ `--no-llm` (rule-based mode)
- Docker + NVIDIA Container Toolkit บนเครื่องเป้าหมาย (สำหรับรัน bundle ที่ generate)
- ดิสก์ว่างบนเครื่องเป้าหมาย ≈ *(ขนาดโมเดล × 1.2) + 25 GB* — runtime image ของ vLLM อย่างเดียว ~10–20 GB ([INSTALL §1.6](docs/INSTALL.md))

## สำหรับผู้พัฒนา

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest
```

รายละเอียด (กฎที่ห้ามละเมิด, วิธีเพิ่ม preset/provider/gate): [CONTRIBUTING.md](CONTRIBUTING.md)

## License

**Proprietary — สงวนลิขสิทธิ์** ดู [LICENSE](LICENSE)

การเปิดซอร์สให้อ่านได้ในรีโปนี้ไม่ได้ให้สิทธิ์ใช้งาน/แจกจ่ายต่อ · **bundle ที่ผู้ใช้ generate ออกมาเป็นของผู้ใช้เอง** ใช้/แก้/ส่งต่อได้อิสระ · โมเดล/image/runtime ของบุคคลที่สามอยู่ใต้ license ของเจ้าของนั้น ๆ

> เป็นการยืนยันสถานะเดิมที่ `pyproject.toml` ประกาศไว้ ไม่ใช่การเปลี่ยนนโยบาย — ถ้าจะเปลี่ยนเป็น open source (MIT/Apache-2.0) ต้องตัดสินใจก่อนส่งมอบลูกค้ารายแรก ดู Decision Log ใน [PRD §13](docs/PRD.md)
