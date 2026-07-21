# CLI Specification — เฟส 1 (MVP)

สเปกคำสั่งของ `lmds` (ชื่อ command ชั่วคราว เปลี่ยนภายหลังได้โดยไม่กระทบโครงสร้าง)
เขียนด้วย Python 3.10+, `typer` + `rich`

## ภาพรวมคำสั่ง

```text
lmds deploy <MODEL_URL_OR_ID> [options]    # flow หลัก: วิเคราะห์ → ยืนยัน → generate → validate → package
lmds inspect <MODEL_URL_OR_ID>             # วิเคราะห์อย่างเดียว ไม่ generate (fit report + ข้อเสนอ runtime)
lmds plan <MODEL_URL_OR_ID>                # สร้าง Deployment Plan (ขั้นวางแผนของ deploy) โดยไม่ generate สคริปต์
                                           #   --no-llm = rule-based, --target <preset>, --json
lmds generate <MODEL_URL_OR_ID>            # plan → render bundle (controller/README/MODEL_PROFILE/SPECIAL_FILES)
                                           #   --output DIR, --target, --no-llm — validate+zip อยู่ใน M6
lmds hardware [--probe-ssh user@host]      # ตรวจ/แสดง hardware profile ของเครื่อง
lmds validate <BUNDLE_DIR>                 # รัน static quality gates กับ bundle ที่มีอยู่
lmds repair <BUNDLE_DIR> --log <FILE|->    # repair workflow จาก log ความล้มเหลว (เฟส 2 — สำรอง interface ไว้)
lmds config <subcommand>                   # จัดการ provider, credentials, site profile
lmds version                               # เวอร์ชันโปรแกรม + เวอร์ชัน template registry
```

## `lmds deploy`

```text
lmds deploy <MODEL_URL_OR_ID> [OPTIONS]

Arguments:
  MODEL_URL_OR_ID   ลิงก์ HF เต็ม | org/model | ลิงก์ไฟล์ .gguf ตรง | ลิงก์ ollama.com | (เฟส 2: NGC, GitHub)

Options:
  --target PROFILE        dgx-spark-single | dgx-spark-stacked | rtx-single | auto (default: auto = ตรวจเครื่องปัจจุบัน)
  --runtime ENGINE        vllm | llamacpp | auto (default: auto — ตาม decision matrix + LLM plan)
  --revision REV          pin revision/commit เอง (default: ล่าสุด ณ เวลา inspect แล้ว pin)
  --context TOKENS        override context เริ่มต้นที่คำนวณให้
  --output DIR            โฟลเดอร์ output (default: ./bundles/<model-slug>/)
  --topology TOPO         single | stacked | both (default: ตาม target)
  --yes / -y              ข้ามขั้นยืนยันแผน (ใช้ค่า plan ทั้งหมด) — สำหรับ scripting
  --no-llm                degraded mode: rule-based เท่านั้น (โมเดลตระกูลที่รู้จัก)
  --dry-run               แสดงแผน + รายการไฟล์ที่จะสร้าง โดยไม่เขียนไฟล์
```

### พฤติกรรมสำคัญ

1. **HF token (optional)** — เมื่อเจอ HTTP 401/403 ระหว่าง inspect:
   - ถ้ามี token ใน credential store หรือ env `HF_TOKEN` → ใช้เลย
   - ถ้าไม่มี → prompt แบบ interactive: `ใส่ Hugging Face token (Enter เพื่อข้าม):`
   - ข้าม → แจ้งข้อจำกัดและเสนอทางเลือก (mirror/quant สาธารณะ หรือยกเลิก)
   - โหมด `--yes`/non-interactive → ไม่ prompt, fail พร้อมข้อความบอกวิธีตั้ง token
2. **ขั้นยืนยันแผน** — แสดงตารางสรุป (model/revision, runtime+image digest, topology, context, VRAM/memory budget, feature ที่เปิด, คำเตือน, facts ที่เป็น `unverified`) ให้ผู้ใช้ ยืนยัน / แก้ค่า / ยกเลิก
3. **Extra flags จาก LLM ที่อยู่นอก allowlist** — แสดงเป็นรายการแยกสีเตือน ต้องกดยืนยันรายตัว
4. **Exit codes**: `0` สำเร็จ, `2` validation ไม่ผ่านหลังวนแก้ครบ N รอบ, `3` โมเดลไม่ fit กับ target, `4` ต้องการ token/สิทธิ์, `5` provider/network error

## `lmds inspect`

Output (human-readable + `--json`):

```text
Model:      Qwen/Qwen3-32B @ <sha>
Artifact:   safetensors (17 shards, 65.3 GB) | license: Apache-2.0 (commercial OK)
Fit (rtx-single, RTX 4090 24GB):  ❌ FP16 ไม่พอ → ✅ ทางเลือก: GGUF Q4_K_M (~19.8 GB) + offload
Fit (dgx-spark-single, 128GB):    ✅ context ปลอดภัยเริ่มต้น 65536
Runtime แนะนำ: vLLM (spark) / llama.cpp (rtx-24GB)
Special files: chat_template.jinja, tool parser: hermes
```

## `lmds config`

```text
lmds config set-provider <openai|gemini|anthropic|openai-compat> [--base-url URL] [--model NAME]
lmds config set-key <provider>          # prompt แบบซ่อน input → เก็บ keyring หรือ ~/.config/lmds/credentials (0600)
lmds config set-hf-token                # เก็บ HF token (optional)
lmds config show                        # แสดง config ปัจจุบัน (redact ทุก secret)
lmds config profile edit                # เปิด site profile (~/.config/lmds/profile.yaml — สืบทอด team-profile.yaml เดิม)
```

ลำดับความสำคัญ credentials: CLI flag > env var (`LMDS_<PROVIDER>_API_KEY`, `HF_TOKEN`) > keyring > credentials file

## `lmds hardware`

ตรวจและ cache hardware profile:

```text
Arch: x86_64 | GPU: NVIDIA RTX 4090 (24 GB, SM89) ×1 | RAM: 128 GB | Disk ว่าง: 1.2 TB
Docker: 27.x ✅ | NVIDIA Container Toolkit ✅ | โปรไฟล์: rtx-single
```

`--probe-ssh user@host` สำหรับตรวจเครื่องเป้าหมายระยะไกล (เฟส 1.5)

## `lmds validate`

รัน quality gates กับ bundle ใด ๆ (รวม bundle ที่แก้มือ):

- `bash -n` ทุกสคริปต์
- Audit rules v3.0.0 (numeric underscore, pipefail, flags/env ครบ, bind/advertise แยก)
- Schema `MODEL_PROFILE.yaml`
- Secret scan
- ตรวจ `PACKAGE_SHA256SUMS` (`--fix` เพื่อ regenerate)

Output: ตาราง pass/fail ต่อ gate + exit code `0/2`

## โครงสร้าง config บนเครื่องผู้ใช้

```text
~/.config/lmds/
├── config.yaml          # provider, default target, ภาษา UI (th/en)
├── credentials          # 0600 — ใช้เมื่อไม่มี keyring
├── profile.yaml         # site profile (master/workers, paths, api defaults)
└── sessions/            # audit log ต่อการ generate (prompt/response/decisions) — redacted
```

## โครงสร้าง source (เป้าหมายเฟส 1)

```text
src/lmds/
├── cli/                 # typer commands
├── resolver/            # hf.py, ollama.py, direct_url.py
├── inspector/           # metadata fetch, gguf_header.py, safetensors_index.py
├── hardware/            # profiler.py, profiles.py
├── fit/                 # memory model ต่อ hardware class
├── brain/               # provider adapters, prompts/, plan_schema.py (pydantic)
├── generator/           # jinja templates/, renderer.py, allowlists.py
├── validator/           # bash_check.py, audit_rules.py, secret_scan.py
├── packager/            # bundle.py, zip + sha256
└── secrets/             # keyring/file/env resolution + redaction filter
tests/
├── fixtures/            # metadata snapshots ของโมเดลอ้างอิง + controllers v3.0.0 (regression)
└── ...
```
