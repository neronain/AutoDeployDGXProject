# Roadmap

> ตัดสินใจแล้ว (2026-07-21): เฟส 1 เป็น **CLI ล้วน** — เฟส 2/3 เป็นข้อเสนอ จะทบทวนและเลือกอีกครั้งเมื่อ CLI เสร็จ ≥98% หรือรันใช้งานจริงได้

## เฟส 1 — CLI MVP (~6–8 สัปดาห์)

### Milestones

| # | Milestone | ส่งมอบ | เกณฑ์ผ่าน |
|---|---|---|---|
| M1 ✅ | โครงโปรเจกต์ + config/secrets (เสร็จ 2026-07-21) | `lmds config`, keyring/0600 store, redaction filter, `lmds hardware` (โครง), GPU allowlist ตามเครื่องทดสอบจริง | set/show provider + HF token ได้, secret ไม่โผล่ใน log — 27 unit tests ผ่าน |
| M2 ✅ | Resolver + Inspector (HF) (เสร็จ 2026-07-21) | `lmds inspect` สำหรับ HF repo (safetensors + GGUF + ลิงก์ไฟล์ตรง), GGUF header ผ่าน HTTP Range, ตรวจ trust_remote_code | ดึง metadata โดยไม่โหลด weight ✅, gated → prompt token (exit 4 เมื่อ non-interactive) ✅, pin เป็น commit SHA ✅ — ทดสอบกับ Hub จริงผ่าน, รวม 57 tests |
| M3 ✅ | Hardware Profiler + Fit Analyzer (เสร็จ 2026-07-21) | Fit Analyzer คำนวณล้วน: KV cache จากมิติจริง (config.json/GGUF header), สูตรแยก unified/discrete, offload path ของ llama.cpp, conservative mode สำหรับ GPU นอก allowlist, `lmds inspect --target` + client token budget | เคสอ้างอิงคำนวณมือตรงทุกตัว (Qwen3-32B: Spark fits ctx 32,768 / RTX 24GB ต้อง quant), รวม 69 tests — เหลือเทียบกับเครื่องจริงตอน M7 |
| M4 ✅ | Brain (LLM Orchestrator) (เสร็จ 2026-07-21) | adapters OpenAI/Gemini/OpenAI-compat (REST ตรง ไม่พึ่ง SDK), DeploymentPlan schema + harden guard (บังคับ revision/context/engine/flag allowlist), retry พร้อม feedback, rule-based degraded mode, session audit log, `lmds plan` | plan ผ่าน schema 100% (validate + retry ≤3), facts tag verified/inferred/unverified, `--trust-remote-code` ต้องอนุมัติเสมอ — รวม 94 tests |
| M5 ✅ | Generator + Templates (เสร็จ 2026-07-21) | Jinja2 templates: single-vllm + single-llamacpp ตาม contract v3.0.0 (flags/env ครบ, bind/advertise แยก, pipefail-safe, exact GGUF size+SHA-256 จาก Hub lfs.oid, tensor parallel เมื่อ multi-GPU), README ตาม delivery contract, MODEL_PROFILE.yaml, SPECIAL_FILES.md, `lmds generate` | bash -n + audit rules ผ่านกับ bundle ที่ generate ทุกแบบ, flag ไม่อนุมัติไม่โผล่ในสคริปต์, ไม่มี secret รั่ว — 108 tests; หมายเหตุ: regression เทียบ 12 controllers เดิมยกไปทำตอน M7 คู่กับ hardware validation |
| M6 ✅ | Validator + Packager (เสร็จ 2026-07-21) | 7 quality gates (bash-syntax, numeric-underscore, pipefail-safe, controller-contract, profile-schema + pinned revision, secret-scan, checksums), `lmds validate [--fix]`, PACKAGE_SHA256SUMS + ZIP, generate → gates → package อัตโนมัติ (ไม่ผ่าน = ไม่มี ZIP, exit 2) | ทุก gate จับเคสพังได้จริง (มีเทส negative ครบ), tamper detection ผ่าน E2E — 125 tests; เทียบ controllers v3.0.0 เดิมยกไป M7 |
| M7a ✅ | End-to-end + เอกสาร (เสร็จ 2026-07-21) | `lmds deploy` ครบ flow: inspect → fit → plan → ขั้นยืนยัน interactive (อนุมัติ flag รายตัว + แก้ context ภายในเพดานปลอดภัย) → render → 7 gates → ZIP, `--yes` สำหรับ scripting, `install.sh` + docs/INSTALL.md | 130 tests + E2E จริงจาก Hub |
| M7b 🔄 | Hardware validation (คืบหน้า) | **2026-07-21: hardware-validated ตัวแรกสำเร็จ** — Qwen3-Coder-30B-A3B UD-Q8_K_XL บน DGX Spark (gigabyte02): สมอง Local AI (gemma-4-26b openai-compat) → native build llama.cpp (121a-real, pinned 76f46ad29) → start → test-text ตอบถูก ~60 tok/s; แก้ตามผลจริง: image allowlist, variant picker, auto-install build deps, client budget ต่อ slot | เหลือ: RTX PRO 4000 ×2 + 4070, โมเดลอ้างอิงอีก 4 ตระกูล (safetensors/NVFP4/MoE-vLLM/gated), regression เทียบ controllers v3.0.0 |
| M8 ✅ | Stacked (multi-node) generation (เสร็จ 2026-07-24) | template `stacked-vllm-controller.sh.j2` port จาก reference v8.2 (DeepSeek-V4-Flash 2×DGX Spark ที่ hardware-validated 2026-07-22) แบบ generic: worker-first startup, image-ID lock 2 node, FlashInfer cache versioning ต่อ image, NCCL/RoCE, sync-worker/verify-worker; planner emit STACKED + harden บังคับ topology จาก target; gate `stacked-contract` ปิดช่องโหว่ single-node ปลอม; เลือกผ่าน `--target dgx-spark-stacked` | bash -n + 8 gates ผ่านกับ bundle stacked, worker.sh/serve-args ตรง reference, GGUF+stacked ถูกปฏิเสธ — 191 tests (รวม test_stacked 12 ตัว); เหลือ hardware regression บนคลัสเตอร์จริง |

### เกณฑ์สำเร็จ MVP

โมเดลอ้างอิง 5 ตัว ครอบคลุม: dense safetensors, GGUF, NVFP4, MoE, gated repo —
ทุกตัวได้ bundle ที่ `static-validated` และอย่างน้อย 2 ตัวรันจริง (`hardware-validated`) บน DGX Spark 1 เครื่อง + เครื่อง RTX 1 เครื่อง

### นอกขอบเขตเฟส 1 (ตัดออกชัดเจน)

- Web UI, repair workflow, NGC/GitHub source, Anthropic adapter (โครง interface เตรียมไว้), SSH remote probe
- ~~stacked controller generation~~ → **ทำเสร็จแล้ว (M8, 2026-07-24)** เลือกผ่าน `--target dgx-spark-stacked`

## เฟส 2 — ข้อเสนอ (เลือก/จัดลำดับหลัง CLI เสร็จ)

เรียงตามที่แนะนำ:

1. ~~**Stacked controller ใน CLI**~~ — ✅ **เสร็จแล้ว (M8, 2026-07-24)** ผ่าน `lmds deploy --target dgx-spark-stacked` (worker-first + sync/verify-worker ครบ) · งานต่อยอด: hardware regression บนคลัสเตอร์จริง + ตัวเลือก `--topology both` (สร้าง single+stacked พร้อมกัน)
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
