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
| M7b ✅ | Hardware validation (เกณฑ์ MVP ครบ 2026-08-03) | **2026-07-21: hardware-validated ตัวแรกสำเร็จ** — Qwen3-Coder-30B-A3B UD-Q8_K_XL บน DGX Spark (gigabyte02): สมอง Local AI (gemma-4-26b openai-compat) → native build llama.cpp (121a-real, pinned 76f46ad29) → start → test-text ตอบถูก ~60 tok/s; แก้ตามผลจริง: image allowlist, variant picker, auto-install build deps, client budget ต่อ slot | **2026-08-02: hardware-validated ตัวที่สอง** — Qwen3.5-122B-A10B-abliterated-NVFP4 (safetensors + NVFP4 + MoE) บน DGX Spark เครื่องเดียว: สมอง Local AI (openai-compat) → vLLM 0.26.0 docker (FLASHINFER_CUTLASS NvFp4 MoE) → start → /health ผ่าน ที่ context 65,536 · แก้ตามผลจริง: เพดาน index.json (MoE/NVFP4 เกิน 4MB), verify shard+ขนาด, คำเตือน endpoint ไม่มี API key · **2026-08-02: ตัวที่สาม** — Qwen3-Coder-30B-A3B-Instruct UD-Q8_K_XL (GGUF, MoE) บน DGX Spark ที่ context **262,144** (4 เท่าของที่แผนแนะนำ) ผ่าน native build llama.cpp b10227 → test-text ตอบถูก ~58 tok/s · ยืนยันว่าสูตร unified memory คำนวณถูกแม้ที่ context สูงมาก · **2026-08-03: ตัวที่สี่ และเป็นเครื่อง RTX เครื่องแรก** — `unsloth/gemma-4-12b-it-GGUF` UD-Q8_K_XL (GGUF + **multimodal**) บน **RTX 5090** (x86_64, Blackwell SM120, VRAM 32GB แบบ discrete): docker `ghcr.io/ggml-org/llama.cpp:server-cuda` → `/health` ผ่านที่ context 16,384 → `test-text` ~96 tok/s → **ยืนยัน vision ด้วยภาพจริง** (ตอบสีถูก) · แก้ตามผลจริง: mmproj ไม่เคยถูกโหลดเลย (โมเดล multimodal GGUF ทุกตัวเป็น text-only เงียบ ๆ), llama.cpp โหมด docker ถูกนับซ้ำใน `lmds ps`, `test-text` budget น้อยเกินสำหรับโมเดล reasoning, `install.sh` ไม่ติดตั้ง prerequisites ให้ · `rtx-5090` เปลี่ยนเป็น `tested=True` แล้ว · **2026-08-03: ตัวที่ห้า** — `Qwen/Qwen3-8B` (dense safetensors) บน RTX 5090 ผ่าน vLLM 0.26.0 ที่ context 32,768 — ปิดเส้นทาง vLLM บน x86_64/discrete VRAM ที่ไม่เคยพิสูจน์ · **2026-08-03: ตัวที่หก — ปิดเกณฑ์ MVP ครบ** — `meta-llama/Llama-3.1-8B-Instruct` (**gated repo**) บน RTX 5090 ที่ context 65,536 → `test-text` ตอบถูก · แก้ตามผลจริง: gated ไม่มี token แล้วได้ traceback Python 60 บรรทัด, token ที่พิมพ์ตอน deploy ใช้ได้แค่ขั้นวิเคราะห์แต่ไม่มีใครบอก, Xet backend ของ Hub พังกับ repo ตระกูล Llama, help หลุดย่อหน้าจาก `{% endif -%}` · เหลือ: hardware regression ของ stacked บนคลัสเตอร์จริง |
| M8 ✅ | Stacked (multi-node) generation (เสร็จ 2026-07-24) | template `stacked-vllm-controller.sh.j2` port จาก reference v8.2 (DeepSeek-V4-Flash 2×DGX Spark ที่ hardware-validated 2026-07-22) แบบ generic: worker-first startup, image-ID lock 2 node, FlashInfer cache versioning ต่อ image, NCCL/RoCE, sync-worker/verify-worker; planner emit STACKED + harden บังคับ topology จาก target; gate `stacked-contract` ปิดช่องโหว่ single-node ปลอม; เลือกผ่าน `--target dgx-spark-stacked` | bash -n + 8 gates ผ่านกับ bundle stacked, worker.sh/serve-args ตรง reference, GGUF+stacked ถูกปฏิเสธ — 191 tests (รวม test_stacked 12 ตัว); เหลือ hardware regression บนคลัสเตอร์จริง |

### เกณฑ์สำเร็จ MVP

โมเดลอ้างอิง 5 ตัว ครอบคลุม: dense safetensors, GGUF, NVFP4, MoE, gated repo —
ทุกตัวได้ bundle ที่ `static-validated` และอย่างน้อย 2 ตัวรันจริง (`hardware-validated`) บน DGX Spark 1 เครื่อง + เครื่อง RTX 1 เครื่อง

## ✅ เกณฑ์ MVP ครบทุกข้อแล้ว (2026-08-03)

| เกณฑ์ | สถานะ |
|---|---|
| GGUF | ✅ Qwen3-Coder-30B (Spark) · gemma-4-12b-it (RTX, multimodal) |
| NVFP4 | ✅ Qwen3.5-122B (Spark) |
| MoE | ✅ สองตัวแรกเป็น MoE |
| dense safetensors | ✅ Qwen3-8B (RTX, vLLM 0.26.0, ctx 32,768) |
| gated repo | ✅ **Llama-3.1-8B-Instruct (RTX, ctx 65,536)** |
| **รันจริงบน DGX Spark ≥1 เครื่อง** | ✅ 3 ครั้ง |
| **รันจริงบนเครื่อง RTX ≥1 เครื่อง** | ✅ RTX 5090 — 3 ครั้ง (llama.cpp + vLLM ×2) |

รวม **hardware-validated 6 ครั้ง** บน 2 เครื่อง — เกินเกณฑ์ "อย่างน้อย 2 ตัว" ที่ตั้งไว้ 3 เท่า

เส้นทาง engine × สถาปัตยกรรมที่พิสูจน์แล้ว:

| | ARM64 / unified (Spark) | x86_64 / discrete (RTX) |
|---|---|---|
| llama.cpp | ✅ native build | ✅ docker (+ multimodal) |
| vLLM | ✅ docker | ✅ docker |

### นอกขอบเขตเฟส 1 (ตัดออกชัดเจน)

- Web UI, repair แบบวิเคราะห์ log, NGC/GitHub source, Anthropic adapter (โครง interface เตรียมไว้), SSH remote probe
  → ~~Web UI~~ **ทำแล้ว (2026-08-04)** · ~~SSH remote probe~~ **ทำแล้ว (2026-08-05)** เป็น fleet หลายเครื่อง
- ~~ไม่มี CI~~ → **มี CI แล้ว (2026-08-02)** · ~~จัดการ container ที่ไม่ได้มาจาก lmds / remove / repair ไฟล์~~ → **ทำแล้ว (2026-08-02)** ตามที่เจอจากการใช้งานจริง
- ~~stacked controller generation~~ → **ทำเสร็จแล้ว (M8, 2026-07-24)** เลือกผ่าน `--target dgx-spark-stacked`

## เฟส 2 — ข้อเสนอ (เลือก/จัดลำดับหลัง CLI เสร็จ)

เรียงตามที่แนะนำ:

1. ~~**Stacked controller ใน CLI**~~ — ✅ **เสร็จแล้ว (M8, 2026-07-24)** ผ่าน `lmds deploy --target dgx-spark-stacked` (worker-first + sync/verify-worker ครบ) · ~~งานต่อยอด: hardware regression บนคลัสเตอร์จริง~~ → ✅ **ผ่านแล้ว (5 ส.ค. 2569)** Llama 3.3 70B บน DGX Spark 2 เครื่อง (mp backend ไม่ใช้ Ray) · งานต่อยอด: 4 เครื่อง + ตัวเลือก `--topology both` (สร้าง single+stacked พร้อมกัน)
2. **Repair workflow ขั้นวิเคราะห์ log** — ส่วน *ไฟล์* ทำแล้ว (`lmds repair` = download resume →
   verify-files, 2026-08-02) · ที่เหลือคือรับ log ที่รันพังมาวิเคราะห์แล้วแก้ค่าใน controller ให้
   · **มีหลักฐานแล้วว่าคุ้มที่สุดในเฟส 2**: การรัน DeepSeek V4 ครั้งแรกพัง 4 รอบ แต่ละรอบสาเหตุ
   อ่านได้จาก traceback ตรง ๆ (`only supports fp8 kv-cache` → kv-cache-dtype ·
   `Expected 7 but got 8 arguments` → cudagraph PIECEWISE · `LocalEntryNotFoundError` →
   HF_HUB_CACHE) · `lmds doctor` จับได้เฉพาะอาการที่รู้จักล่วงหน้า ส่วน LLM อ่าน log จริงได้
3. **Ollama + NGC source** + ทางเลือก output แบบ Ollama Modelfile (รอคำตอบคำถามเปิดข้อ 1 ใน PRD)
4. ~~**Web UI หน้าเดียว**~~ — ✅ **ทำแล้ว (2026-08-04/06)** `lmds web` · host stats, deploy wizard,
   fleet หลายเครื่อง, cluster fabric, scan, recipes
   · **รอบ 2026-08-06 (จากการใช้จริงบน controller)**: SSE แทน polling · เกจ CPU/RAM/VRAM/Disk +
   telemetry ของ GPU · เมนู ⋯ ต่อโมเดลบนเครื่องอื่น (restart/doctor/logs/repair/autostart/remove)
   · ตั้ง port/context/gpu-util ตอนสั่งรันข้ามเครื่อง · **หน้า login + token ที่อยู่ยาว**
   · cluster IP ย้ายไปอยู่ในการ์ดของเครื่องนั้น + รั้วสีจับคู่ที่ stacked ได้
   · ~~แท็บ tests+manage สำหรับโมเดลบนเครื่องอื่น~~ **ทำแล้ว (2026-08-06)** — คุมได้เท่ากับ
   โมเดลในเครื่อง (env ชุดเดียวกัน) · ~~ส่ง bundle ไปเครื่องอื่น~~ **ทำแล้ว** `lmds node push`
   · งานต่อยอด: **job progress / log สด**, command palette (⌘K), deploy wizard เลือกเครื่อง
   ปลายทางได้ตั้งแต่ต้น (ตอนนี้ต้อง deploy แล้วค่อย push)
5. ~~**Runtime smoke test อัตโนมัติ**~~ — ✅ **ทำแล้ว (2026-08-06)** `lmds smoke <slug> [--on เครื่อง]`
   download → verify-files → start → test-text → stop · หยุด server เสมอแม้ล้มกลางทาง
   · **เหตุผลที่ต้องมี**: บั๊กที่เจ็บที่สุดทุกตัวของรอบ 0.2.0 ผ่าน gate แบบ static ทั้งหมด
   แล้วไปตายตอนรันจริง · งานต่อยอด: ให้รันอัตโนมัติหลัง deploy (ตอนนี้ต้องสั่งเอง)
6. **สูตรที่รันผ่านจริง (recipes)** — ✅ **ทำแล้ว (2026-08-05)** `lmds recipes` แก้ปัญหาลูกค้า/SI
   ที่ไม่มี API key แล้ว deploy ผ่านแต่ start ไม่ขึ้น · งานต่อยอด: ให้ LLM ร่างสูตรใหม่จาก config
   ที่รันสำเร็จ แล้วคนตรวจก่อนเข้าแคตตาล็อก (LLM สำรวจ · สูตรจดจำ)
7. Anthropic provider, i18n ไทยเต็มรูป, ~~SSH remote probe~~ → ✅ **ทำแล้ว (2026-08-05)** เป็น
   `lmds node` (ทะเบียนเครื่อง + `lmds agent info` ผ่าน SSH) พร้อมตรวจ ConnectX/200G และจับคู่ stacked
   · งานต่อยอด: push bundle ไปติดตั้งบน node ให้อัตโนมัติ, ยืนยัน fabric detection กับ ConnectX จริง

## เฟส 3 — ข้อเสนอระยะยาว

- Multi-GPU RTX (tensor parallel), docker-compose / systemd hardened output
- Kubernetes / Helm chart output
- Bundle registry ภายในทีม (เก็บ/ค้น/แชร์ bundle + ประวัติ validation)
- Telemetry ต้นทุน LLM ต่อการ generate

## หลักการคุมคุณภาพตลอดทุกเฟส

- ทุก PR ต้องผ่าน: unit tests + regression เทียบ controllers v3.0.0 + secret-leak scan
  - ✅ มี CI แล้ว (`.github/workflows/ci.yml`, 2026-08-02): pytest บน Python 3.10/3.11/3.12 +
    `bash -n`/shellcheck สคริปต์ในรีโป + secret scan
  - ✅ **regression เทียบ v3.0.0 แล้ว (2026-08-03)** — `tests/test_v3_regression.py` port กฎทั้ง 13 ข้อ
    จาก `audit-controllers.py` มาตรง ๆ แทนการ vendor controller อ้างอิงเข้ารีโป · bundle ทุกแบบ
    ได้ผล 0 error 0 warning เท่ากับ controller อ้างอิงทั้ง 21 ตัว
- Template registry แยกเป็น data (อัปเดต image digest/runtime pin ได้โดยไม่ release โปรแกรมใหม่)
- ห้ามอ้าง `hardware-validated` โดยไม่ได้รันจริง — สถานะ validation ติดไปกับ bundle เสมอ
