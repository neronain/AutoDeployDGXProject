# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

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
# → วิเคราะห์ → วางแผน → ยืนยัน (อนุมัติ flag/แก้ context ได้) → bundle + ZIP ที่ผ่าน 7 quality gates

# โมเดล gated → ระบบถาม HF token อัตโนมัติ (กด Enter ข้ามได้)
lmds deploy meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single
```

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
- LLM API key อย่างน้อย 1 provider (OpenAI / Gemini / OpenAI-compatible) — มี degraded mode สำหรับโมเดลตระกูลที่รู้จักเมื่อไม่มี key
- Docker + NVIDIA Container Toolkit บนเครื่องเป้าหมาย (สำหรับรัน bundle ที่ generate)

## License

TBD — ดู Decision Log ใน [PRD §13](docs/PRD.md)
