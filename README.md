# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

> ⚡ สร้างและดูแลโดย **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

โปรแกรม CLI บน Ubuntu ที่รับ **ลิงก์โมเดล** (Hugging Face / Ollama / NGC / URL ตรง) แล้วใช้ **LLM API** (OpenAI, Gemini, Claude หรือ OpenAI-compatible endpoint) เป็นสมองในการวิเคราะห์และ**สร้างชุดสคริปต์ deploy ที่ผ่านการ validate แล้ว** สำหรับ:

- **NVIDIA DGX Spark** — เครื่องเดี่ยว หรือ stacked หลายเครื่อง
- **Ubuntu + RTX GPU** — local AI server ทั่วไป (x86_64)

สืบทอดมาตรฐานจาก [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers) และ skill pack `dgx-spark-model-deployer-team-pack-v3.0.0` ซึ่งมี controller ที่รันจริงแล้วกว่า 12 โมเดล

## สถานะโปรเจกต์

🟢 **เฟส 1 — CLI MVP: โค้ดครบทุกคำสั่งแล้ว (M1–M7a)** — เหลือ hardware validation บนเครื่องจริง (M7b)

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
./install.sh                          # ติดตั้ง (Ubuntu, python3 >= 3.10)
lmds config set-provider openai       # ตั้ง LLM provider ครั้งเดียว (หรือใช้ --no-llm)
lmds config set-key openai

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

ตรวจว่าอัปเดตแล้ว (ควรเห็นบรรทัด `dgx-spark-stacked`):

```bash
lmds deploy --help | grep -i stacked
```

ถ้า `git pull` ฟ้อง local changes: `git stash && git pull && ./install.sh` (การแก้ PATH ครั้งก่อนอยู่ใน `~/.bashrc` ไม่ใช่ในโปรเจกต์ จึงไม่กระทบ) · รายละเอียด/ถอนการติดตั้ง: [docs/INSTALL.md](docs/INSTALL.md)

## จัดการหลายโมเดลในเครื่องเดียว (Fleet)

รันกี่โมเดลก็ได้ (คนละ port) แล้วคุมทั้งหมดจาก `lmds` — ไม่ต้องจำว่า bundle ไหนอยู่ที่ไหน:

```bash
lmds ps                  # ใครรันอยู่บ้าง: ชื่อ, โมเดล, engine, port, สถานะ ● running / ◐ loading / ○ stopped
lmds stop <ชื่อ>          # หยุดตามชื่อจาก lmds ps — ไม่ต้อง cd ไปหา ./xxx.sh stop
lmds stop --all          # หยุดทุกโมเดลที่รันอยู่ในคำสั่งเดียว
lmds logs <ชื่อ> -n 500   # ดู log ตามชื่อ
lmds start <ชื่อ>         # รันโมเดลที่เคย deploy ไว้ขึ้นมาใหม่ (เช่น หลัง reboot)
lmds enable <ชื่อ>        # ให้โมเดลกลับมาเองหลังเปิด-ปิดเครื่อง (systemd autostart) · disable = ยกเลิก
lmds list                # bundle ทั้งหมด + engine/port/context/ฟีเจอร์ที่รองรับ + สถานะ autostart
```

ทุก controller ลงทะเบียนตัวเองอัตโนมัติตอน `start` — ต่อให้ bundle ถูกลบไปแล้ว `lmds stop` ก็ยัง
fallback หยุดโมเดลค้างให้ได้ (kill pid / docker rm) · รายละเอียด: [docs/USAGE.md §4](docs/USAGE.md)

ผลลัพธ์: โฟลเดอร์ bundle + ZIP ประกอบด้วย controller script (มาตรฐาน v3.0.0), `README.md`, `MODEL_PROFILE.yaml`, `SPECIAL_FILES.md`, `PACKAGE_SHA256SUMS`

## เอกสารทั้งหมด

| เอกสาร | เนื้อหา |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | **คู่มือติดตั้งละเอียด** — prerequisites (Docker/NVIDIA toolkit พร้อมคำสั่งครบ), ติดตั้ง, ตั้งค่า provider/token, อัปเดต/ถอนการติดตั้ง |
| [docs/USAGE.md](docs/USAGE.md) | **คู่มือใช้งานละเอียด** — deploy ตั้งแต่โมเดลเล็กถึง gated repo, คำสั่ง controller ทุกตัว, target presets, ย้าย bundle ข้ามเครื่อง, troubleshooting 12 อาการ |
| [docs/PRD.md](docs/PRD.md) | Product Requirements Document ฉบับเต็ม — เป้าหมาย, user stories, functional requirements, สถาปัตยกรรม, security, risks |
| [docs/CLI_SPEC.md](docs/CLI_SPEC.md) | สเปกคำสั่ง CLI ทั้งหมดของเฟส 1 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | แผนพัฒนา 3 เฟส + work breakdown ของเฟส 1 |

## Requirements (ตัวโปรแกรม)

- Ubuntu 22.04 / 24.04 (ARM64 หรือ x86_64) — พัฒนาบน macOS ได้
- Python 3.10+
- LLM provider อย่างน้อย 1 ทาง: OpenAI / Gemini / MiniMax / OpenAI-compatible (Ollama, vLLM local — ไม่ต้องมี key) — หรือไม่มีเลยก็ใช้ `--no-llm` (rule-based mode)
- Docker + NVIDIA Container Toolkit บนเครื่องเป้าหมาย (สำหรับรัน bundle ที่ generate)

## License

TBD — ดู Decision Log ใน [PRD §13](docs/PRD.md)
