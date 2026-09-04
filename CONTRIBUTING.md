# คู่มือผู้พัฒนา

## ตั้ง dev environment

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
```

> Ubuntu 24.04 / DGX OS บล็อก pip ระดับระบบ (PEP 668) — **ต้องใช้ venv เสมอ**
> พัฒนาบน macOS ได้ · บน Windows เทสที่เป็น POSIX-only (chmod 0600, exec bit, `os.kill`) จะ fail เป็นปกติ

`.venv/` ใช้สำหรับพัฒนาเท่านั้น — คนละตัวกับ venv ของคำสั่ง `lmds` ที่ `install.sh` สร้างไว้ที่
`~/.local/share/lmds/venv` · แก้โค้ดแล้วอยากให้คำสั่ง `lmds` เปลี่ยนตาม ต้องรัน `./install.sh` ซ้ำ
(ติดตั้งแบบ copy ไม่ใช่ editable)

## กฎที่ห้ามละเมิด

หลักการเดียวที่ทั้งโปรเจกต์ยืนอยู่บนมัน:

> **Deterministic core + LLM assist** — LLM ไม่เขียน Bash เอง ทุกสคริปต์ render จาก template
> ที่ผ่านการตรวจแล้ว LLM แค่เลือกค่าใน `DeploymentPlan` (JSON schema ตายตัว)
> ส่วนการคำนวณ memory fit / token budget ทำด้วยโค้ด 100%

ตามมาด้วย:

1. **ทุกอย่างที่ LLM เสนอต้องผ่าน `harden_plan()`** — revision/context/engine/topology ถูกบังคับกลับ
   ตามข้อเท็จจริงเสมอ · flag/image/runtime asset ต้องผ่าน allowlist
2. **อะไรที่เสี่ยงต้องให้ผู้ใช้อนุมัติเอง** ห้าม default เป็นอนุมัติ (flag นอก allowlist, ไฟล์ runtime ภายนอก, การลบไฟล์)
3. **ห้ามอ้าง `hardware-validated` โดยไม่ได้รันจริง** — bundle ที่ยังไม่ได้รันคือ `static-validated`
4. **secret ห้ามลง** config.yaml / bundle / log / commit — ใช้ `redact()` คุมทุกทางออก

## รันเทส

```bash
pytest                      # ทั้งหมด (~1,720 เทส · 110+ ไฟล์)
pytest tests/test_brain.py  # เฉพาะไฟล์
pytest -k stacked           # เฉพาะที่ชื่อตรง
pytest -v -rA -k embed      # อยากเห็นชื่อเทส/ผลทีละข้อ — pyproject ตั้ง addopts = "-q" ไว้ ผลปกติจึงเงียบ
```

CI (`.github/workflows/ci.yml`) รันให้ทุก push/PR: pytest บน Python 3.10/3.11/3.12,
`bash -n` + shellcheck สคริปต์ในรีโป, และ secret scan

### เทสที่ต้องมี `node` — หน้าเว็บรัน JS จริง

`tests/test_console_shell.py` (และ `test_page_javascript_parses`) บูตสคริปต์จริงของ `index.html` ใน **Node.js** ด้วย DOM
ย่อส่วน (`tests/console_shell_dom.js`) แล้วถามว่าการ์ดไหน "มองเห็น" จริง — grep สตริงในไฟล์จับบั๊กแบบ "กด site แล้วหน้าว่าง"
ไม่ได้ · ไม่มี `node` บนเครื่อง = **ข้ามพร้อมเหตุผล** ไม่ใช่ผ่านเงียบ ๆ (นับเป็น skipped ใน CI) · เทสหา node จาก PATH →
`~/.nvm/versions/node/*/bin` → `~/.volta/bin` → `/opt/homebrew/bin` → `/usr/local/bin` · ติดตั้งใน VM/เครื่อง dev ด้วย
`sudo apt install nodejs` หรือ nvm ก็พอ (ไม่ต้องมี npm package ใด — ไม่มี dependency)

### ไฟล์ทดสอบขนาดใหญ่ถูกลบทันทีที่เทสจบ

เทสโหลดขนาน (`tests/test_parallel_fetch.py` · `test_audit_cli_controllers.py`) สร้างไฟล์ ≥256 MB ใต้ `tmp_path` —
เคยค้างใน `/tmp` ของ VM ข้าม 3 รอบ pytest จนดิสก์เหลือ 66 MB แล้วเทสอื่นล้มแบบอ่านไม่รู้เรื่อง · fixture autouse ใน
`tests/conftest.py` ลบไฟล์ >64 MB ใต้ `tmp_path` ทันทีหลังแต่ละเทส · เทสใหม่ที่ต้องการไฟล์ใหญ่ให้สร้างใต้ `tmp_path` เท่านั้น

### เทส review / audit — "เทสที่ล้มก่อนแก้"

รอบตรวจทั้งระบบเก็บเป็นไฟล์ตามชุดที่ตรวจ ไม่ใช่ตามโมดูล:

| ไฟล์ | ตรวจอะไร |
|---|---|
| `tests/test_review_backend.py` · `test_review_web.py` · `test_review_templates.py` | review 0.6.0 ชุด fleet/nodes/brain/fit · ชั้นเว็บ · controller templates (render แล้วรันใต้ bash จริง) |
| `tests/test_audit_backend.py` · `test_audit_cli_controllers.py` | audit รอบสอง: web/fleet/nodes · CLI + controller ทั้งสี่ template แบบลูกค้ารัน |
| `tests/test_audit_stacked_plan.py` · `test_audit_stacked_orchestration.py` · `test_audit_stacked_controller.py` | stacked ฝั่ง planner · hub↔head↔worker (SSH ปลอม) · controller ทั้ง lifecycle (docker/ssh/rsync ปลอมบันทึก argv ต่อ node) |

กติกา: **ทุกข้อในรายงาน review/audit ต้องมีเทสที่ล้มก่อนแก้** แล้วผ่านหลังแก้ — ข้อที่ "ยืนยันว่าถูกอยู่แล้ว" ก็มีเทสคุมไว้
(ระบุในหัวข้อ CHANGELOG ว่าเทสไหน) · เทสพวกนี้รัน**ของจริง**เท่าที่ทำได้: controller ที่ render จริงใต้ `bash`, สคริปต์ติดตั้ง
กับ `git` จริง (`test_install_ship.py`), เซิร์ฟเวอร์ HTTP ปลอมสำหรับ `test-*`/`bench` (`test_stacked_test_commands.py`) ·
บั๊กที่ผ่าน gate แบบ static ทั้งหมดแล้วไปตายตอนรันคือแบบที่เจ็บที่สุด — อะไรที่ผู้ใช้จะรัน ต้องมีเทสที่รันมันจริงอย่างน้อยหนึ่งเส้นทาง
· เรื่อง auth/secret ต้องพิสูจน์กับ binary จริงก่อน (เคส `LLAMA_ARG_API_KEY` ที่ llama-server ไม่มี — เทสที่เชื่อ env ผ่านทั้งที่
เซิร์ฟเวอร์รันแบบไม่มี auth)

### เทสรันใน sandbox — `~` ไม่ใช่ home จริง

`tests/conftest.py` ย้าย `HOME`, `LMDS_CONFIG_DIR` และ `LMDS_RUN_ROOT` ไปโฟลเดอร์ชั่วคราว
ตั้งแต่ตอน import (ก่อน pytest collect) · เทสจึงเขียนทับ `~/.config/lmds` ของเครื่องที่รันไม่ได้
แม้แต่โค้ดที่ resolve home เอง — เคยลบ `nodes.yaml` ของผู้ใช้มาแล้วสองครั้ง

สิ่งที่ต้องรู้เวลาเขียนเทส:

- **อย่าใช้ `Path.home()` เพื่อหมายถึง "ของจริง"** — มันคืน sandbox · ถ้าต้องเทียบกับของจริง
  ให้ import `REAL_HOME` / `REAL_CONFIG_DIR` / `REAL_RUN_ROOT` จาก `tests.conftest`
  ซึ่งจำค่าไว้ตั้งแต่ก่อนย้าย
- **ห้ามชี้ env กลับไปที่ของจริง** — มีด่านตรวจก่อนและหลังทุกเทส ถ้าชี้กลับจะ fail ตรงเทสนั้น
  พร้อมบอกว่า path ไหนหลุด
- เทสไม่คุยกับ systemd และ registry จริง (`no_systemd_lookups`, `no_registry_lookups`) —
  เทสที่ตั้งใจทดสอบสายนั้น patch ทับเองได้ตามปกติ

`lmds web` ที่รันค้างอยู่บนเครื่อง dev เขียน `nodes.yaml` ทุก 1-3 วิ — เทสไม่สนใจมันแล้ว
และมันก็ไม่ทำให้เทสล้ม (เดิมล้ม เพราะด่านเก่าเทียบไฟล์จริงก่อน/หลัง session)

## งานที่ทำบ่อย

### เพิ่ม target preset (GPU รุ่นใหม่)

1. `src/lmds/hardware/profiles.py` — เพิ่มลง `KNOWN_GPUS` (ตั้ง `tested=False` ถ้ายังไม่ได้รันจริง
   → ระบบจะคำนวณแบบ conservative ให้เอง)
2. `src/lmds/fit/targets.py` — เพิ่มลง `PRESETS`
3. เพิ่มเทสใน `tests/test_hardware.py` และอัปเดตตารางใน [docs/USAGE.md §3.4](docs/USAGE.md)

### เพิ่ม LLM provider

1. `src/lmds/config/settings.py` — เพิ่มใน `ProviderName` + `DEFAULT_MODELS`
2. `src/lmds/secrets/store.py` — เพิ่ม env var ที่ยอมรับใน `SECRET_ENV_VARS`
3. `src/lmds/brain/providers.py` — เขียน adapter (เรียก REST ตรงผ่าน `httpx` ไม่พึ่ง SDK หนัก)
   แล้ว dispatch ใน `make_provider()` · **ใช้ `_post_with_retry()`** เพื่อให้ได้ backoff เหมือนตัวอื่น
4. เทสด้วย `httpx.MockTransport` ตามแบบใน `tests/test_providers.py`

### แก้ template ของ controller

`src/lmds/generator/templates/*.j2` (single-vllm · single-llamacpp · single-sglang · stacked-vllm) — หลังแก้ต้องผ่าน
quality gates ทั้ง 12 ด่านโดยอัตโนมัติ (เทสใน `tests/test_generator.py` เรียก `run_gates` ให้อยู่แล้ว) · ข้อควรระวังที่ gate จับ:

- ห้าม numeric underscore literal ใน arithmetic (`(( 65_536 ))`)
- ห้าม pipefail-unsafe (`... | grep -q`)
- ต้องมี flag/env ครบตาม controller contract v3.0.0
- แยก bind / advertise / cluster address ออกจากกัน
- ห้ามมี template tag เหลือในไฟล์ผลลัพธ์ (heredoc ใน `{% raw %}` เคยพิมพ์ `{{ slug }}` ดิบ)

และกติกาที่เทสคุมนอก gate: **ทุกคำสั่งในบล็อก COMMANDS ของ `usage()` ต้องถูก dispatch จริง** (และกลับกัน) ในทั้ง 6 รูปแบบ
controller (`tests/test_stacked_test_commands.py` ฯลฯ) · secret ห้ามขึ้น argv (API key → `--api-key-file`/env · HF token → stdin)
· `explain_crash`/pipeline ใต้ `set -e` ต้องปิดท้าย `|| true` · แก้ template แล้วอัปเดตตาราง "คำสั่งไหนมีบน engine ไหน" ใน
[docs/USAGE.md §2](docs/USAGE.md)

### เพิ่ม quality gate

`src/lmds/validator/gates.py` — **ต้องมีเทส negative** (พิสูจน์ว่า gate จับเคสพังได้จริง) ไม่ใช่แค่เทสว่า pass

### เพิ่มสิ่งที่ผู้ช่วยดูได้ (probe) หรือแก้ได้ (action)

`src/lmds/assistant/catalog.py` — ทั้งสองอย่างอยู่ไฟล์เดียวกันโดยตั้งใจ: อ่านไฟล์นี้จบแล้ว
ต้องรู้ครบว่าผู้ช่วยแตะอะไรได้บ้าง ไม่ต้องไปไล่อ่าน prompt แล้วเดาว่าโมเดลจะคิดอะไรออก

1. **probe** (อ่านอย่างเดียว รันได้ทันที) — เขียน `answers` ให้เป็น *คำถามที่มันตอบได้* ไม่ใช่
   ชื่อคำสั่ง เพราะข้อความนี้คือสิ่งที่ router ใช้เลือก · คำสั่งสำรวจที่เครื่องมือย่อยอาจไม่มี
   (nvidia-smi, docker, แคชที่ยังไม่เคยสร้าง) ให้ห่อด้วย `_survey()` ไม่งั้นมันรายงานว่า
   "ล้ม" แล้วผู้ช่วยทิ้งข้อมูลที่ใช้ได้
2. **action** (เปลี่ยนสภาพเครื่อง) — **ต้องมี `impact`** เป็นภาษาคน เพราะผู้ใช้อ่านข้อความนี้
   ก่อนกดอนุมัติ · มีเทสบังคับว่า action ทุกตัวต้องมี
3. พารามิเตอร์ทุกตัวต้องเป็นชนิดที่ `Param.clean` ตรวจได้ — **ห้ามรับสตริงอิสระไปต่อคำสั่ง**
   ค่าที่ LLM ส่งมาถือว่าไม่น่าเชื่อถือเท่าค่าที่ผู้ใช้พิมพ์
4. เทสใน `tests/test_assistant_tools.py` — เพิ่มเคส negative ด้วย (ค่าที่ควรถูกปฏิเสธ)

อยากให้มัน *คิด* ต่างไป (ไม่ใช่ทำได้มากขึ้น) แก้ที่ `src/lmds/assistant/playbook.md` ·
อย่าคัดลอกตารางอาการ→วิธีแก้จาก `docs/` มาไว้ใน playbook — มันค้นจากไฟล์จริงได้อยู่แล้ว
และสำเนาจะล้าสมัยเงียบ ๆ วันที่มีคนแก้เอกสาร

## Commit / PR

- commit message เป็นภาษาไทยได้ · บรรทัดแรกสั้น ๆ แล้วอธิบาย **ทำไม** ในย่อหน้าถัดไป
  (โค้ดบอก *อะไร* อยู่แล้ว — ที่หายากคือเหตุผล)
- ถ้าแก้ตามผลการรันจริงบนเครื่อง ให้บันทึกเคสจริงไว้ในข้อความ commit (เช่น "เคสจริง:
  Qwen3.5-122B NVFP4 index.json 6MB ชนเพดาน 4MB") — ประวัติแบบนี้ช่วยคนที่มาเจอทีหลังมาก
- แก้โค้ดที่กระทบพฤติกรรม → อัปเดตเอกสารที่เกี่ยวข้องใน PR เดียวกัน
- เอกสารหลักที่ต้องตามให้ทัน: [README.md](README.md) **และ [README.en.md](README.en.md)**,
  [docs/USAGE.md](docs/USAGE.md), [docs/CLI_SPEC.md](docs/CLI_SPEC.md), [CHANGELOG.md](CHANGELOG.md)
  · แก้ข้อกำหนด/ขอบเขต → [docs/PRD.md](docs/PRD.md) · แตะเรื่องสิทธิ์หรือการอนุมัติ →
  [SECURITY.md](SECURITY.md) · ปิดงานที่อยู่ในแผน → [docs/ROADMAP.md](docs/ROADMAP.md)
  · แตะ install.sh / node install → [docs/INSTALL.md](docs/INSTALL.md) + [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md)
  · เพิ่มการตรวจก่อน deploy/start → [docs/PREFLIGHT.md](docs/PREFLIGHT.md) · `bench`/`stress`/คะแนน → [docs/BENCH.md](docs/BENCH.md)
- badge จำนวนเทสใน README ทั้งสองไฟล์ = `pytest --collect-only -q tests/ | tail -1` · เวอร์ชันจาก `src/lmds/__init__.py`
- **ฟีเจอร์ใหม่ต้องมีทางกดบนหน้าเว็บด้วย** (PRD FR-1b.14) — ทำแค่ CLI/API เท่ากับไม่มีสำหรับทีมที่ทำงานผ่านคอนโซล ·
  ข้อความบนหน้าเว็บเป็นอังกฤษ comment ในโค้ดและ CLI เป็นไทย

## โครงสร้างโค้ด

ดู [docs/CLI_SPEC.md](docs/CLI_SPEC.md) หัวข้อ "โครงสร้าง source" — อัปเดตให้ตรงของจริงเสมอ
