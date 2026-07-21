# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

โปรแกรม CLI บน Ubuntu ที่รับ **ลิงก์โมเดล** (Hugging Face / Ollama / NGC / URL ตรง) แล้วใช้ **LLM API** (OpenAI, Gemini, Claude หรือ OpenAI-compatible endpoint) เป็นสมองในการวิเคราะห์และ**สร้างชุดสคริปต์ deploy ที่ผ่านการ validate แล้ว** สำหรับ:

- **NVIDIA DGX Spark** — เครื่องเดี่ยว หรือ stacked หลายเครื่อง
- **Ubuntu + RTX GPU** — local AI server ทั่วไป (x86_64)

สืบทอดมาตรฐานจาก [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers) และ skill pack `dgx-spark-model-deployer-team-pack-v3.0.0` ซึ่งมี controller ที่รันจริงแล้วกว่า 12 โมเดล

## สถานะโปรเจกต์

🚧 **เฟส 1 — CLI MVP (กำลังพัฒนา)** — ดู [ROADMAP.md](docs/ROADMAP.md)

## หลักการออกแบบสำคัญ

> **Deterministic core + LLM assist** — LLM ไม่เขียน Bash เอง ทุกสคริปต์ render จาก template ที่ผ่านการตรวจแล้ว LLM ทำหน้าที่แค่วิจัยโมเดลและเลือกค่าใน Deployment Plan (JSON schema ตายตัว) ส่วนการคำนวณ memory fit / token budget ทำด้วยโค้ด 100% และทุก bundle ต้องผ่าน quality gates (`bash -n`, audit rules, SHA-256) ก่อนถึงมือผู้ใช้

## ตัวอย่างการใช้งาน (เป้าหมายเฟส 1)

```bash
# ตั้งค่า provider ครั้งเดียว
lmds config set-provider openai

# สร้าง deployment bundle จากลิงก์โมเดล
lmds deploy https://huggingface.co/Qwen/Qwen3-32B --target rtx-single

# โมเดล gated → ระบบถาม HF token อัตโนมัติ (ข้ามได้)
lmds deploy https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct --target dgx-spark-single
```

ผลลัพธ์: โฟลเดอร์ bundle + ZIP ประกอบด้วย controller script (มาตรฐาน v3.0.0), `README.md`, `MODEL_PROFILE.yaml`, `SPECIAL_FILES.md`, `PACKAGE_SHA256SUMS`

## เอกสาร

| เอกสาร | เนื้อหา |
|---|---|
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
