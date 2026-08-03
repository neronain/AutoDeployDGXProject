# Changelog

รูปแบบตาม [Keep a Changelog](https://keepachangelog.com/) · เวอร์ชันตาม [SemVer](https://semver.org/)

## [Unreleased]

รอบนี้เกือบทั้งหมดมาจาก **การรันจริงบน DGX Spark (dgx-msi)** — ทุกข้อคือปัญหาที่เจอหน้างานจริง

### Added

- **`lmds remove <slug>`** — ลบโมเดลออกจากเครื่องทั้งหมด: หยุดเซิร์ฟเวอร์ → ยกเลิก autostart →
  ลบ bundle + ZIP + ทะเบียน/log + runtime files + weight · แสดงรายการและขนาดให้ดูก่อนถามยืนยันเสมอ
  · `--keep-weights` เก็บ weight ไว้ (deploy ใหม่ได้โดยไม่ต้องโหลดหลายสิบ GB ซ้ำ)
- **`lmds repair <slug>`** — โหลดไฟล์ที่ขาด/เสียกลับมา (download resume → verify-files)
- **`lmds restart <slug>`**
- **`lmds logs <slug> -f`** — ตาม log แบบ realtime (`docker logs -f` / `tail -f`)
- **จัดการ container ที่ไม่ได้ deploy ผ่าน LMDS** — `lmds ps` สแกน `docker ps` และรับตัวที่ image
  ตรงกับ engine ที่รู้จัก (vLLM/llama.cpp/Ollama/TGI) เข้ามา · stop/restart/logs/enable ใช้ได้
  · **`stop` ของกลุ่มนี้ใช้ `docker stop` ไม่ใช่ `docker rm -f`** — ไม่ลบ container ของคุณทิ้ง
- **Tab completion** — เติมชื่อคำสั่ง (typer) + ชื่อ bundle + ชื่อ target preset
  (`lmds --install-completion` หรือให้ `install.sh` ถามตอนติดตั้ง)
- **`lmds hardware` ตรวจพื้นที่ดิสก์** ($HOME) พร้อมเตือนเมื่อเหลือ < 50 GB — ปิด PRD FR-2.1 ที่ค้าง
- **`lmds list` แสดงสถานะ** (● running / ◐ loading / ○ stopped / ⚠ controller หาย)
- **`verify-files` ฝั่ง safetensors ตรวจ shard ครบ + ขนาดตรงกับ Hub** และตรวจไฟล์ tokenizer
  (เดิมตรวจแค่ว่า config.json/index.json มีอยู่ — download ที่ขาดจึงหลุดไปพังตอน `start`)
- **รองรับไฟล์ runtime ภายนอก (`runtime_assets`)** — เคสอย่าง Nemotron-3-Super ที่ต้องมี parser
  plugin ซึ่งไม่ได้อยู่ใน repo ของโมเดล · คุมด้วย host allowlist (HTTPS จาก huggingface.co /
  raw.githubusercontent.com / github.com / gitlab.com) + **ผู้ใช้ต้องอนุมัติรายตัวเสมอ**
  · controller ได้คำสั่ง `prepare-runtime` ไปดึงไฟล์ + ตรวจ SHA-256 แล้ว mount แบบ read-only
- **CI** (`.github/workflows/ci.yml`) — pytest บน Python 3.10/3.11/3.12, `bash -n` + shellcheck,
  secret scan · ก่อนหน้านี้ ROADMAP ประกาศกฎ "ทุก PR ต้องผ่านเทส" ไว้แต่ไม่มีอะไรบังคับ
- **`install.sh` ติดตั้ง prerequisites ให้เอง ไม่ใช่แค่ตรวจแล้วบอกให้ไปพิมพ์เอง**
  (เคสจริงจากการติดตั้งบนเครื่อง RTX 5090: ผู้ใช้ต้องไล่พิมพ์เอง 5 คำสั่ง — get.docker.com →
  `usermod` → keyring ของ nvidia → `apt install` → `nvidia-ctk runtime configure`)
  · ตอนนี้ถามติดตั้งให้ทั้ง **Docker**, **กลุ่ม `docker` ของ user**, **NVIDIA Container Toolkit
  (ครบ 5 ขั้น)** และ **`python3-venv`** แล้วทดสอบว่า Docker เห็น GPU จริง
  · ทุกขั้นที่ใช้ `sudo` ถามยืนยันก่อนและ **พิมพ์คำสั่งจริงให้เห็นก่อนรัน** · ตอบ `n` ได้ทุกข้อ
  · `LMDS_ASSUME_YES=1` ติดตั้งรวดเดียวไม่ต้องตอบ · `LMDS_SKIP_PREREQ=1` ข้ามทั้งหมด
  · ไม่ใช่ tty (CI/pipe) = ไม่แตะเครื่องเลย
  · **NVIDIA driver ยังไม่ทำให้โดยตั้งใจ** — ต้อง reboot และเคยเจอ `ubuntu-drivers install`
  ชน dependency บนเครื่องที่ driver ใช้งานได้อยู่แล้ว
- **`install.sh` แยกสาเหตุที่ `docker info` ล้มออกจากกัน** — daemon ไม่ขึ้น / user ไม่อยู่ในกลุ่ม /
  อยู่ในกลุ่มแล้วแต่ shell ยังไม่เห็น (กรณีหลังยืม `sg docker` ทำงานต่อได้ทันทีไม่ต้อง logout)
  · ตรวจด้วย `systemctl`/`id` ไม่ใช่ `sudo docker info` จะได้ไม่เด้งขอรหัสผ่านก่อนผู้ใช้อนุญาต
- **banner ลาย RTX อีก 4 ชุด** (นิ่ง 2 + เคลื่อนไหว 2: พัดลมการ์ดหมุน, VRAM ค่อย ๆ เต็มพร้อมข้อความ
  fit) — เดิมมี 9 ลายที่เป็นธีม DGX Spark/ทั่วไปล้วน ทั้งที่เครื่องเป้าหมายครึ่งหนึ่งเป็น Ubuntu + RTX
  · รวมเป็น 13 ลาย · เทสใหม่บังคับว่าทุก frame ของ animation เดียวกันต้องสูง/กว้างเท่ากัน
  (ไม่งั้นภาพกระตุกตอน `Live` วาดทับ)
- **`LICENSE`**, **`SECURITY.md`**, **`CONTRIBUTING.md`**, **`CHANGELOG.md`**, **`README.en.md`**

### Changed

- **help ของ controller เขียนใหม่เป็นภาษาอังกฤษ** แบ่งเป็น COMMANDS / OPTIONS / **API TOKEN** /
  ENVIRONMENT VARIABLES / EXAMPLES พร้อมค่า default จริงของ bundle นั้น
  — เดิมมีแค่รายการคำสั่งและไม่พูดถึงการตั้ง API token เลย
- **controller เตือนหลัง `start`** เมื่อ bind `0.0.0.0` และไม่มี API key (ค่า default = เปิดทั้งวง LAN)
- **retry/backoff ตอนเรียก LLM provider** — 429/5xx/เน็ตกระตุก retry 3 ครั้ง (เคารพ `Retry-After`)
  ก่อนยอมตกไป rule-based · เดิมพลาดครั้งเดียวก็ได้แผนคุณภาพต่ำลงโดยแทบไม่ทันสังเกต
- **fallback เมื่อ endpoint local ไม่รู้จัก `response_format`** — ยิงซ้ำโดยตัด field ออก
  (vLLM/llama.cpp server รุ่นเก่าตอบ 400 ทั้งคำขอ ทำให้เส้นทาง "ใช้โมเดล local เป็นสมอง" ใช้ไม่ได้เลย)
- **`config set-provider anthropic` ปฏิเสธตั้งแต่ตอนตั้งค่า** (เดิมตั้งผ่านแล้วไปพังตอน deploy)
- **`install.sh` ย้ายข้อความ "ต้องทำอะไรต่อ" ไปไว้บรรทัดสุดท้ายที่เดียว** — เดิมบอกเรื่อง PATH ตั้งแต่
  กลางสคริปต์แล้วโดน banner + เมนู provider ดันหายขึ้นไป ปิดท้ายด้วยการแนะให้รัน `lmds hardware`
  ซึ่ง **รันไม่ได้แน่นอนใน shell เดิม** · ตอนนี้สรุปเป็นข้อ ๆ และใส่ `source ~/.bashrc` /
  `newgrp docker` เป็นข้อแรกเมื่อจำเป็น
- **`install.sh` ไม่แนะให้ `set-key` ตอนเลือก provider แบบ OpenAI-compatible** — endpoint ในองค์กร
  ส่วนใหญ่ไม่ต้องใช้ key แต่ข้อความเดิมทำให้เข้าใจว่าตั้งค่าไม่สำเร็จ
- **`install.sh` ปิด banner ระหว่างเรียก `lmds`** (`LMDS_NO_BANNER=1`) — เดิม banner ขึ้นกลางการติดตั้ง
  หลายรอบจนกลบผลการตรวจเครื่อง
- **เวอร์ชันมี source of truth เดียว** — hatch dynamic version อ่านจาก `src/lmds/__init__.py`
- **`install.sh` ลองติดตั้ง extra `keyring` ให้** (เดิม key ตกไฟล์ 0600 เสมอโดยผู้ใช้ไม่รู้ตัว)
- **เอกสารทั้งหมดไล่ปรับให้ตรงกับโค้ด** — `docs/INSTALL.md` ขยายจาก 211 เป็น 520+ บรรทัด
  (ที่เก็บไฟล์/ดิสก์, pre-pull image, proxy/air-gapped, ใช้โมเดล local เป็นสมองแบบละเอียด,
  โมเดลถูกดึงมาและรันยังไง, smoke test) · README/USAGE/CLI_SPEC/PRD/ROADMAP แก้จุดที่ไม่ตรงโค้ด

- **port การปรับปรุงจาก single → stacked** — `verify-files` ตรวจ shard ทีละไฟล์ + ขนาดตรง Hub
  (เดิมนับจำนวนอย่างเดียว ซึ่งหยาบกว่าฝั่ง single ทั้งที่ stacked มีขั้น rsync ข้ามเครื่องเพิ่มอีกจุด
  ที่ไฟล์ขาดได้), คำเตือน endpoint ไม่มี API key, และ help ภาษาอังกฤษพร้อมหัวข้อ **API TOKEN**
  · ยังไม่ port: `runtime_assets` และ `wait-health` (ดู Known gaps)

### Known gaps

- **stacked ยังไม่เคยรันจริงจาก bundle ที่ LMDS สร้าง** — reference v8.2 เคย hardware-validated
  (2026-07-22) แต่นั่นคือสคริปต์เขียนมือ · ตัวที่ generate ยังเป็น `static-validated` เท่านั้น
- **`runtime_assets` ยังไม่รองรับใน stacked** — ต้อง sync ไฟล์ plugin ไป worker และ mount ทั้งสอง node
  ทำครึ่งทางจะแย่กว่าไม่ทำ (head มี plugin แต่ worker ไม่มี = พังตอน start แบบไล่ยาก)
- **`wait-health` ยังไม่มีใน stacked** — ฝั่ง stacked มี `STARTUP_TIMEOUT` ยาวกว่าอยู่แล้ว ความจำเป็นน้อยกว่า

### Fixed

- **`install.sh` จบเงียบกลางคันบนระบบที่ `df` ไม่รองรับ `--output`** — บรรทัดตรวจดิสก์เป็นคำสั่ง
  สุดท้ายของ branch และเจอ `pipefail` เข้าไปด้วย ทำให้ `set -e` ฆ่าสคริปต์ทิ้งก่อนถึงขั้นตั้ง
  provider/completion โดยไม่มี error ให้เห็น (เจอตอนทดสอบ installer นอก GNU coreutils)
- **`model.safetensors.index.json` ของ MoE/NVFP4 ชนเพดาน 4 MB จน deploy ไม่ได้**
  (เคสจริง: `w341e/Qwen3.5-122B-A10B-abliterated-NVFP4`) — index โตตาม *จำนวน tensor* ไม่ใช่ขนาดโมเดล
  แยก `INDEX_FILE_CAP` (64 MB) ออกจาก `SMALL_FILE_CAP` (4 MB) และแยก error ของขนาดไฟล์ metadata
  ออกจาก "ปัญหาเครือข่าย/Hub" ที่ทำให้ไล่ผิดทาง
- **`pytest` ตรง ๆ collect ไม่ผ่าน** — เพิ่ม `tests/__init__.py` (เทสหลายไฟล์ import helper กันเอง
  ผ่าน `tests.*` · เดิมผ่านเฉพาะเมื่อรัน `python -m pytest`)
- **สัญลักษณ์ ⚠ ใน `lmds list` ทับสถานะ running** — โมเดลที่รันอยู่แต่ bundle ถูกลบเคยแสดงเป็น ⚠
  แทน ● · ตอนนี้ ⚠ ขึ้นเฉพาะตอนหยุดอยู่ (ซึ่งเป็นตอนที่ start/restart ใช้ไม่ได้จริง)
- **`lmds ps` / `lmds list` ไม่บอกว่าเอาอะไรไปสั่ง** — ตอนนี้พิมพ์คำสั่งจริงพร้อม slug ในตารางให้ copy ได้เลย

### Validated

- **hardware-validated ตัวที่สอง**: `Qwen3.5-122B-A10B-abliterated-NVFP4` (safetensors + NVFP4 + MoE)
  บน DGX Spark เครื่องเดียว — vLLM 0.26.0, FLASHINFER_CUTLASS NvFp4 MoE backend, context 65,536,
  `/health` ผ่าน (ตัวแรกคือ Qwen3-Coder-30B-A3B GGUF native build, 2026-07-21)
- **hardware-validated ตัวที่สาม**: `Qwen3-Coder-30B-A3B-Instruct` UD-Q8_K_XL (GGUF, MoE) บน DGX Spark
  ที่ **context 262,144** — 4 เท่าของที่แผนแนะนำ, native build llama.cpp b10227, `test-text` ตอบถูก
  ~58 tok/s · ยืนยันว่าสูตร unified memory ของ Fit Analyzer คำนวณถูกแม้ที่ context สูงมาก

---

## [0.1.0] — 2026-07-24

เฟส 1 CLI MVP: M1–M7a + M8 (stacked multi-node)

- `lmds` ครบ flow: `inspect` → `plan` → `deploy` (ขั้นยืนยัน + อนุมัติ flag รายตัว) → 8 quality gates → ZIP
- Resolver + Inspector สำหรับ Hugging Face (safetensors / GGUF / ลิงก์ไฟล์ตรง, GGUF header ผ่าน HTTP Range)
- Hardware profiler + Fit analyzer (คำนวณล้วน: KV cache จากมิติจริง, สูตรแยก unified/discrete)
- Brain: adapter OpenAI / Gemini / MiniMax / OpenAI-compatible + rule-based degraded mode
- Generator: template single-vllm, single-llamacpp, stacked-vllm ตาม controller contract v3.0.0
- Fleet: `ps` / `list` / `start` / `stop` / `logs` / `enable` / `disable` (systemd autostart)

รายละเอียดต่อ milestone: [docs/ROADMAP.md](docs/ROADMAP.md)
