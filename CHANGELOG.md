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
- **`lmds web` — หน้าเว็บคุมโมเดล (เฟส 2 เริ่มแล้ว)** · หน้าเดียว: สถานะเครื่อง (unified memory
  ของ Spark แสดงคนละแบบกับ VRAM แยกของ RTX), รายการโมเดลพร้อมสถานะ, start/stop/restart,
  doctor และ logs แบบกางในแถวเดียวกัน
  - **ไม่ดึงอะไรจากอินเทอร์เน็ตเลย** — ไม่มี CDN/ฟอนต์/ไอคอนภายนอก ใช้ได้บนเครื่องหลัง proxy
    หรือ air-gapped · เทสบังคับข้อนี้ไว้แล้ว
  - **ความปลอดภัยตั้งแต่แรก** (PRD §9): หน้านี้สั่ง start/stop ได้ จึง bind `127.0.0.1` เป็นค่าเริ่มต้น
    · `--bind 0.0.0.0` เมื่อไม่ได้ตั้ง `--token` จะ**สุ่ม token ให้เอง**แล้วพิมพ์ลิงก์พร้อม token มาให้
    · เทียบ token ด้วย `compare_digest` และคุมทุกเส้นทาง API ไม่ใช่แค่บางอัน
  - ชั้นเว็บไม่มี logic ของตัวเอง เรียก core เดิมทั้งหมด — เทสเทียบผล doctor ของเว็บกับ CLI ว่าตรงกัน
  - อัปเดตหน้าแบบ**เฉพาะส่วนที่เปลี่ยน** ไม่ re-render ทั้ง DOM ทุกรอบ (ไม่งั้น panel ที่เปิดอยู่กระพริบ
    และปุ่มหายไปใต้เมาส์ระหว่างที่ผู้ใช้กำลังจะกด)
  - **deploy wizard บนเว็บ**: วางลิงก์ → เลือก target → วิเคราะห์ → เห็นแผน+fit → ปรับ context /
    อนุมัติ flag → สร้าง bundle ผ่าน 9 gates → ZIP · ครบทุกทางแยกเหมือน CLI: repo GGUF หลาย variant
    ให้เลือกไฟล์ (เรียงตามขนาด ไม่มี mmproj ปน), gated repo ขอ token, ไม่ fit บอกทางเลือกอื่น
    · **inspect ซ้ำหลังเลือกไฟล์ GGUF** เพื่ออ่าน header จริง — ไม่ทำขั้นนี้ fit จะได้แค่ `unknown`
    (เจอตอนทดสอบ: verdict `unknown` + context 8,192 กลายเป็น `fits` + 16,384 หลังแก้)
    · เพดาน context ตัดที่ฝั่ง server ไม่เชื่อค่าจากหน้าเว็บ · session ใช้ได้ครั้งเดียว กันกดซ้ำ
  - ติดตั้ง: `install.sh` ลง `fastapi`/`uvicorn` ให้เอง (ล้มเหลวก็ไม่กระทบ CLI) · extra ชื่อ `web`
- **`lmds web` พิมพ์ IP จริงของเครื่องแทน `0.0.0.0`** — `0.0.0.0` เป็นที่อยู่สำหรับ bind ไม่ใช่ที่อยู่
  ที่เปิดในเบราว์เซอร์ได้ · เจอจากการใช้งานจริง: ผู้ใช้ต้องไปหา IP เอง
- **`lmds fleet.logs_text()`** — คืน log เป็นข้อความแทนพิมพ์ออกจอ (`logs_server` capture ไม่ได้)
- **`lmds doctor <slug>`** — ตรวจว่าทำไมโมเดลยัง download/start ไม่ผ่าน แล้วบอก**คำสั่งแก้ตรง ๆ**
  · ทุกข้อที่ตรวจมาจาก failure ที่เจอจริงตอน hardware validation 2026-08-03 ไม่ได้เดาว่าน่าจะพังตรงไหน:
  - **hf-token** — profile บอก gated แต่ไม่มี `HF_TOKEN` ใน env (เคสจริง: ผู้ใช้พิมพ์ token ตอน deploy
    แล้วเข้าใจว่าใช้ได้ตลอด ที่จริง controller อ่านจาก env เท่านั้น → 401 พร้อม traceback 60 บรรทัด)
  - **weights** — ไฟล์ที่ profile ประกาศไว้หายไป รวม **mmproj** (เคสที่ทำให้ multimodal กลายเป็น
    text-only เงียบ ๆ) และไฟล์ขนาด 0 ไบต์จาก download ที่ค้างกลางคัน
  - **permissions** — cache dir เขียนไม่ได้เพราะ container เคยสร้างเป็น root (จาก reference v8.2)
  - **port** — ถูกโปรเซสอื่นยึด (บอกด้วยว่าตัวไหน) · **disk** · **docker** · **runtime-image** · **server/health**
  · คำนวณล้วน ไม่ส่งอะไรให้ LLM ตีความ (PRD §8) · exit 2 เมื่อมีข้อที่บล็อกการรัน ใช้ใน script ได้
  · เป็นฐานของ `lmds repair` ขั้นวิเคราะห์ log (PRD FR-8) ต่อไป
- **ไล่ช่องว่าง stacked เทียบ reference v8.2 ตัวที่รันจริง** (`deepseek-v4-flash-nvfp4-stacked-v8.2`)
  — template ของเรา port มาจากรุ่นก่อนหน้า จึงขาดสิ่งที่ v8.2 เพิ่มจากการเจอหน้างาน:
  - **ซ่อมสิทธิ์ cache อัตโนมัติ** (`_ensure_local_owned_dir` / `_ensure_worker_owned_dir`) — คือที่มา
    ของคำว่า "permission-safe" ในชื่อไฟล์ v8.2 · docker เคยสร้าง cache เป็น root รอบถัดไป user
    เขียนไม่ได้แล้ว `start` ล้มแบบไล่สาเหตุยาก · ซ่อมด้วย container ตัวเดิม ไม่ต้องพึ่ง sudo บน host
  - **คืน shared memory ตอน `stop`** (`/dev/shm/psm_*`, `/dev/shm/sem.mp-*`) — mp backend ทิ้งไว้
    ไม่เก็บกวาดแล้ว `start` รอบหน้าชนของเก่า
  - **`clear-fi-cache`** — FlashInfer JIT cache ค้างจาก image เก่าทำให้โหลด TVM-FFI module
    คนละ signature แล้วพังตอน start · เดิมไม่มีคำสั่งกู้เลย ผู้ใช้ติดตาย
  - **`props`** — เรียก `/v1/models`
  - **แก้ port check ที่ใช้ไม่ได้จริง** — container รันด้วย `--network host` จึงไม่ publish port
    ตัวกรอง `docker ps --filter publish=` จับไม่เจอเลย · เปลี่ยนไปดู listening socket (`ss`/`netstat`)
- **`test-reasoning` / `test-tools` ตามฟีเจอร์ที่แผนเปิดจริง** (vLLM single + stacked) — เรา emit
  `--reasoning-parser` / `--tool-call-parser` ให้ vLLM มาตลอดแต่ไม่เคยมีทางพิสูจน์ว่า parser ตรงกับ
  โมเดลไหม · parser ผิดตัวจะเงียบจนลูกค้าเจอเอง · โมเดลที่ไม่เปิดฟีเจอร์จะไม่มีคำสั่งพวกนี้ติดมา
- **regression เทียบ controllers v3.0.0** (`tests/test_v3_regression.py`) — ROADMAP ประกาศกฎนี้ไว้
  ตั้งแต่ต้นว่า "ทุก PR ต้องผ่าน" แต่ไม่เคยมีอะไรบังคับ · port กฎทั้งชุดจาก `audit-controllers.py`
  ของ repo เดิม (13 ข้อ) มาเป็นเทส แทนการ vendor controller อ้างอิง 21 ไฟล์ (~400 KB) เข้ารีโป
  เพราะสิ่งที่ต้องคงไว้คือ *กฎ* ไม่ใช่ไฟล์ · รันกับ bundle ทุกแบบ (vLLM/llama.cpp × Spark/RTX + stacked)
  - **ทำให้ output ผ่านมาตรฐานจริง ๆ**: เดิมขาด 3 ข้อ — `SCRIPT_VERSION="${SCRIPT_VERSION:-X.Y.Z}"`,
    `banner()`/`info()` + dispatch `info|banner)`, และ `prompt_cluster_config()` ของ stacked
    · ตอนนี้ audit ให้ผล **0 error 0 warning เท่ากับ controller อ้างอิงทั้ง 21 ตัว**
  - `controller-contract` gate ตรวจ SCRIPT_VERSION/banner/info เพิ่ม · `stacked-contract` ตรวจ
    `prompt_cluster_config()` — bundle ที่ขาดจะไม่มี ZIP ออกมา
  - `info`/`banner` เป็นคำสั่งใหม่ของ controller: `./<slug>-single.sh info` บอกโมเดล/runtime/
    feature/context/endpoint/สถานะในหน้าจอเดียว
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

- **โมเดล multimodal GGUF ถูกเสิร์ฟเป็น text-only แบบเงียบ ๆ ทุกตัว** (เคสจริง:
  `unsloth/gemma-4-12b-it-GGUF` บน RTX 5090, 2026-08-03) — `plan.multimodal.projector_files`
  ไปถึงแค่ `SPECIAL_FILES.md` เท่านั้น controller ไม่เคยเห็นค่านี้เลย ผลคือ `download` โหลดไฟล์เดียว,
  `verify-files` ผ่าน, `start` ผ่าน, `/health` เขียว — แต่ `llama-server` ไม่ได้รับ `--mmproj`
  จึงรับแต่ข้อความ **ไม่มี error ให้เห็นสักจุด** · กระทบทั้ง DGX Spark (native) และ RTX (docker)
  เพราะใช้ template llama.cpp ตัวเดียวกัน — ที่ไม่เคยเจอเพราะโมเดลที่ hardware-validated ทั้งสามตัว
  เป็น text-only หรือเป็น vLLM (vision tower อยู่ใน safetensors อยู่แล้ว จึงไม่กระทบ)
  - ไฟล์ mmproj ถูกผนวกท้าย `MODEL_FILES` → `download`/`verify-files` (ขนาด + SHA-256) ครอบคลุมเอง
    · `MODEL_FILE` ยังเป็น weight เพราะ mmproj ต่อท้ายเสมอ
  - `server_args` ส่ง `--mmproj` จริง · env `MMPROJ_FILE` ตั้งว่างได้ถ้าอยากเสิร์ฟแบบข้อความล้วน
  - `harden_plan` บังคับ mmproj ให้ตรงไฟล์ที่มีจริงใน repo — ครอบคลุมทั้งกรณี LLM เดาชื่อผิด (เคยทำให้
    URL 404) และกรณี `--no-llm` ที่ไม่มีใครประกาศ (เปิด multimodal ให้เองเมื่อ repo มี mmproj,
    เลือกไฟล์เล็กสุด BF16 < F16 < F32) · `--mmproj` ถูกตัดจาก `extra_flags` เสมอ เพราะ path เป็นของ
    controller (native/docker คนละที่)
  - **gate ใหม่ `multimodal-assets`** (รวมเป็น 9 ด่าน) — profile ประกาศ mmproj แต่ controller ไม่โหลด
    หรือไม่ส่ง `--mmproj` = ไม่ผ่าน ไม่มี ZIP · bundle ที่มีบั๊กนี้เคยผ่าน gates ครบทุกด่าน
- **gated repo: `download` โยน traceback ของ Python 60 บรรทัดแทนที่จะบอกว่า "ยังไม่ได้ตั้ง token"**
  (เคสจริง `meta-llama/Llama-3.1-8B-Instruct` บน RTX 5090, 2026-08-03) — controller ตรวจ `HF_TOKEN`
  ก่อน download แล้ว die พร้อมขั้นตอนแก้ 3 ข้อ · เปิดเฉพาะ repo ที่ `gated` จริงตาม Hub API
- **token ที่พิมพ์ตอน `lmds deploy` ใช้ได้แค่ขั้นวิเคราะห์ แต่ไม่มีใครบอก** — ผู้ใช้พิมพ์ token ถูก
  ต้อง bundle ออกมาสวยงาม แล้วไปเจอ 401 ตอน `download` เพราะ controller อ่านจาก env `HF_TOKEN`
  เท่านั้น (เจตนา: ไม่ฝัง secret ลง bundle) · ตอนนี้บอกทันทีหลัง inspect พร้อมสองทางเลือก
  (`lmds config set-hf-token` หรือ `export HF_TOKEN=`)
- **`download` ล้มเมื่อ Hub ใช้ Xet backend** — `RuntimeError: Unable to parse string as hex hash value`
  กับ repo ตระกูล Llama · download รันในคอนเทนเนอร์ env บน host จึงไม่ถึงข้างใน — ส่ง
  `HF_HUB_DISABLE_XET` เข้าไปให้ และ**ลองซ้ำอัตโนมัติ**เมื่อรอบแรกล้ม (ไฟล์ที่โหลดแล้วยังอยู่ resume ต่อ)
- **help ของ controller vLLM: บรรทัด `start` หลุดการย่อหน้าไปชิดขอบ** — `{% raw %}{% endif -%}{% endraw %}` ของ Jinja
  กิน whitespace ของบรรทัดถัดไป · กระทบ 3 จุด (help, case dispatch, `docker_args+=(--entrypoint …)`)
  · เทสใหม่ไล่ตรวจการย่อหน้าในบล็อก COMMANDS ทุก template
- **โมเดล llama.cpp โหมด docker ถูกนับสองครั้งใน `lmds ps` / `lmds list`** (เจอจริงบน RTX 5090,
  2026-08-03) — process ใน container มองเห็นได้จาก process table ของ host ด้วย `_orphan_native`
  จึงเก็บมันมาเป็น "orphan" อีกแถว โดยใช้ค่า `--alias` เป็น slug ทำให้ดูเหมือนคนละโมเดล และ
  สั่ง `stop`/`logs` ตามชื่อนั้นไม่ได้ · กรองด้วย `/proc/<pid>/cgroup` + กันซ้ำด้วยพอร์ตอีกชั้น
- **`test-text` ดูเหมือนพังกับโมเดลสาย reasoning — ทั้ง 3 template** — `max_tokens: 64` ถูก
  chain-of-thought กินหมดก่อนจะได้ตอบ ผู้ใช้เห็นคำตอบว่างกับ `finish_reason: "length"`
  แล้วนึกว่าโมเดลเสีย · เจอสองรอบวันเดียวกัน: gemma-4-12b-it ฝั่ง llama.cpp (`reasoning_content`
  แยก field) แล้ว Qwen3-8B ฝั่ง vLLM (`<think>` อยู่ใน `content`) — รอบแรกแก้เฉพาะ llama.cpp
  ทั้งที่บั๊กเดียวกันอยู่ใน vLLM ทั้ง single และ stacked ด้วย · เพิ่มเป็น 512 ทุกตัวและสรุปผลเป็น
  ภาษาคนต่อท้าย โดยแยก "ยังคิดไม่จบ" ออกจาก "ตอบว่างจริง ๆ" · เทสคุมทั้ง 3 template แล้ว
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
- 🎉 **hardware-validated ตัวที่หก — เกณฑ์ MVP ของเฟส 1 ครบทุกข้อ**:
  `meta-llama/Llama-3.1-8B-Instruct` (**gated repo** — ข้อสุดท้ายที่เหลือ) บน RTX 5090 —
  vLLM 0.26.0, `/health` ผ่านที่ context 65,536, `test-text` ตอบถูก (`2+2 เท่ากับ 4`)
  · ระหว่างทาง `verify-files` จับ download ที่ค้างจาก Xet ได้เอง (`ไฟล์จำเป็นหายไป: tokenizer.json`)
  ก่อนจะหลุดไปพังตอน `start` — ด่านนี้ทำงานตามที่ออกแบบไว้
  · รวม hardware-validated **6 ครั้ง บน 2 เครื่อง** ครบทั้ง 4 ช่องของเมทริกซ์ engine × สถาปัตยกรรม
  (llama.cpp/vLLM × ARM64-unified/x86_64-discrete)
- **hardware-validated ตัวที่ห้า — ปิดตระกูล *dense safetensors* และเส้นทาง *vLLM บน x86_64***:
  `Qwen/Qwen3-8B` (dense safetensors, BF16 ~16 GB) บน **RTX 5090** — vLLM 0.26.0 docker,
  `/health` ผ่านที่ context 32,768, `test-text` ตอบได้ · ก่อนหน้านี้ vLLM เคยรันแต่บน ARM64/unified
  ของ DGX Spark เท่านั้น สูตร discrete VRAM ของ vLLM จึงไม่เคยถูกพิสูจน์เลย
  · **ผลที่ขัดกับที่คาดไว้**: `gpu-memory-utilization 0.85` ผ่านได้ทั้งที่จอต่ออยู่การ์ดใบเดียวกัน
  (Xorg + gnome ใช้ ~640 MiB) — ไม่ต้องลดค่าลงอย่างที่กลัว
- **hardware-validated ตัวที่สี่ — และเป็น *เครื่อง RTX เครื่องแรก* ของโปรเจกต์**:
  `unsloth/gemma-4-12b-it-GGUF` UD-Q8_K_XL (GGUF + multimodal) บน **RTX 5090** (x86_64,
  Blackwell SM120, VRAM 32 GB แบบ discrete) — docker `ghcr.io/ggml-org/llama.cpp:server-cuda`,
  `/health` ผ่านที่ context 16,384, `test-text` ~96 tok/s และ **ยืนยัน vision ด้วยภาพจริง**
  (ส่ง PNG สีแดง → ตอบ "สีแดงเข้ม" ถูกต้อง) · ปิดช่องว่างที่ค้างมาตลอด: เดิม hardware validation
  ทั้งหมดอยู่บน DGX Spark เครื่องเดียว สูตร discrete VRAM + เส้นทาง x86_64 ไม่เคยเจอเครื่องจริงเลย
  · `rtx-5090` เปลี่ยนจาก `tested=False` เป็น `True` แล้ว (ไม่ถูกหัก budget แบบ conservative อีก)
  · การรันครั้งนี้ทำให้เจอบั๊ก 4 ตัวที่แก้ไปแล้วในรอบนี้ (mmproj, นับซ้ำใน `lmds ps`,
  `test-text` กับโมเดล reasoning, `install.sh` ไม่ติดตั้ง prerequisites)
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
