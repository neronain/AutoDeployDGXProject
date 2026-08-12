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
pytest                      # ทั้งหมด
pytest tests/test_brain.py  # เฉพาะไฟล์
pytest -k stacked           # เฉพาะที่ชื่อตรง
```

CI (`.github/workflows/ci.yml`) รันให้ทุก push/PR: pytest บน Python 3.10/3.11/3.12,
`bash -n` + shellcheck สคริปต์ในรีโป, และ secret scan

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

`src/lmds/generator/templates/*.j2` — หลังแก้ต้องผ่าน quality gates ทั้ง 8 ด่านโดยอัตโนมัติ
(เทสใน `tests/test_generator.py` เรียก `run_gates` ให้อยู่แล้ว) · ข้อควรระวังที่ gate จับ:

- ห้าม numeric underscore literal ใน arithmetic (`(( 65_536 ))`)
- ห้าม pipefail-unsafe (`... | grep -q`)
- ต้องมี flag/env ครบตาม controller contract v3.0.0
- แยก bind / advertise / cluster address ออกจากกัน

### เพิ่ม quality gate

`src/lmds/validator/gates.py` — **ต้องมีเทส negative** (พิสูจน์ว่า gate จับเคสพังได้จริง) ไม่ใช่แค่เทสว่า pass

## Commit / PR

- commit message เป็นภาษาไทยได้ · บรรทัดแรกสั้น ๆ แล้วอธิบาย **ทำไม** ในย่อหน้าถัดไป
  (โค้ดบอก *อะไร* อยู่แล้ว — ที่หายากคือเหตุผล)
- ถ้าแก้ตามผลการรันจริงบนเครื่อง ให้บันทึกเคสจริงไว้ในข้อความ commit (เช่น "เคสจริง:
  Qwen3.5-122B NVFP4 index.json 6MB ชนเพดาน 4MB") — ประวัติแบบนี้ช่วยคนที่มาเจอทีหลังมาก
- แก้โค้ดที่กระทบพฤติกรรม → อัปเดตเอกสารที่เกี่ยวข้องใน PR เดียวกัน
- เอกสารหลักที่ต้องตามให้ทัน: [README.md](README.md), [docs/USAGE.md](docs/USAGE.md),
  [docs/CLI_SPEC.md](docs/CLI_SPEC.md), [CHANGELOG.md](CHANGELOG.md)

## โครงสร้างโค้ด

ดู [docs/CLI_SPEC.md](docs/CLI_SPEC.md) หัวข้อ "โครงสร้าง source" — อัปเดตให้ตรงของจริงเสมอ
