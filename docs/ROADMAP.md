# Roadmap

> ตัดสินใจแล้ว (2026-07-21): เฟส 1 เป็น **CLI ล้วน** — เฟส 2/3 เป็นข้อเสนอ จะทบทวนและเลือกอีกครั้งเมื่อ CLI เสร็จ ≥98% หรือรันใช้งานจริงได้

## เฟส 1 — CLI MVP (~6–8 สัปดาห์)

### Milestones

| # | Milestone | ส่งมอบ | เกณฑ์ผ่าน |
|---|---|---|---|
| M1 ✅ | โครงโปรเจกต์ + config/secrets (เสร็จ 2026-07-21) | `lmds config`, keyring/0600 store, redaction filter, `lmds hardware` (โครง), GPU allowlist ตามเครื่องทดสอบจริง | set/show provider + HF token ได้, secret ไม่โผล่ใน log — 27 unit tests ผ่าน |
| M2 | Resolver + Inspector (HF) | `lmds inspect` สำหรับ HF repo (safetensors + GGUF) | ดึง metadata โดยไม่โหลด weight, ตรวจ gated → prompt token, pin revision |
| M3 | Hardware Profiler + Fit Analyzer | `lmds hardware`, fit report ต่อ profile (spark-single, rtx-single) | ตัวเลข fit ตรงกับเครื่องจริงของทีม (Spark 128GB) ± headroom ที่กำหนด |
| M4 | Brain (LLM Orchestrator) | adapters OpenAI/Gemini/OpenAI-compat, Deployment Plan schema + validation, `--no-llm` degraded mode | plan ผ่าน schema 100%, facts มี tag verified/inferred/unverified |
| M5 | Generator + Templates | Jinja2 templates: single-vllm (spark/rtx), single-llamacpp (spark/rtx) ตาม contract v3.0.0 | generate controller 12 ตัวเดิมซ้ำได้เทียบเท่า (regression fixtures) |
| M6 | Validator + Packager | quality gates ครบ + วนแก้อัตโนมัติ, bundle + ZIP + SHA256 | `lmds validate` ผ่านกับ controllers v3.0.0 เดิมทุกตัว |
| M7 | End-to-end + เอกสาร | `lmds deploy` ครบ flow, README ผู้ใช้, install script | เกณฑ์สำเร็จ MVP ด้านล่าง |

### เกณฑ์สำเร็จ MVP

โมเดลอ้างอิง 5 ตัว ครอบคลุม: dense safetensors, GGUF, NVFP4, MoE, gated repo —
ทุกตัวได้ bundle ที่ `static-validated` และอย่างน้อย 2 ตัวรันจริง (`hardware-validated`) บน DGX Spark 1 เครื่อง + เครื่อง RTX 1 เครื่อง

### นอกขอบเขตเฟส 1 (ตัดออกชัดเจน)

- Web UI, stacked controller generation (template มีแล้วแต่ยังไม่เปิดใน CLI), repair workflow, NGC/GitHub source, Anthropic adapter (โครง interface เตรียมไว้), SSH remote probe

## เฟส 2 — ข้อเสนอ (เลือก/จัดลำดับหลัง CLI เสร็จ)

เรียงตามที่แนะนำ:

1. **Stacked controller ใน CLI** (`--topology stacked|both`) — template พร้อมแล้ว เหลือ orchestration ของ sync/verify-worker
2. **Repair workflow** (`lmds repair`) — มูลค่าสูงกับลูกค้าจริง เพราะปัญหาหลัง deploy คืองานหลัก
3. **Ollama + NGC source** + ทางเลือก output แบบ Ollama Modelfile (รอคำตอบคำถามเปิดข้อ 1 ใน PRD)
4. **Web UI หน้าเดียว** (FastAPI) — reuse core ทั้งหมด, สำหรับลูกค้าที่ไม่ถนัด CLI
5. **Runtime smoke test อัตโนมัติ** บนเครื่องเป้าหมาย (download → start → /health → test-text → stop)
6. Anthropic provider, i18n ไทยเต็มรูป, SSH remote probe

## เฟส 3 — ข้อเสนอระยะยาว

- Multi-GPU RTX (tensor parallel), docker-compose / systemd hardened output
- Kubernetes / Helm chart output
- Bundle registry ภายในทีม (เก็บ/ค้น/แชร์ bundle + ประวัติ validation)
- Telemetry ต้นทุน LLM ต่อการ generate

## หลักการคุมคุณภาพตลอดทุกเฟส

- ทุก PR ต้องผ่าน: unit tests + regression เทียบ controllers v3.0.0 + secret-leak scan
- Template registry แยกเป็น data (อัปเดต image digest/runtime pin ได้โดยไม่ release โปรแกรมใหม่)
- ห้ามอ้าง `hardware-validated` โดยไม่ได้รันจริง — สถานะ validation ติดไปกับ bundle เสมอ
