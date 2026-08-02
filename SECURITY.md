# Security

เอกสารนี้บอกว่า LMDS เก็บอะไรไว้ที่ไหน ส่งอะไรออกนอกเครื่องบ้าง และมีจุดไหนที่ผู้ใช้ต้องตัดสินใจเอง

## ข้อมูลอะไรออกจากเครื่องบ้าง

| ออกไปไหน | ส่งอะไร | เมื่อไหร่ | ปิดได้ไหม |
|---|---|---|---|
| **LLM provider ที่คุณตั้งไว้** | model card, `config.json`, รายชื่อไฟล์ + ขนาด, ผลคำนวณ fit | ตอน `plan` / `deploy` / `generate` | ได้ — `--no-llm` หรือใช้ Local AI เป็นสมอง ([INSTALL §3.2.1](docs/INSTALL.md)) |
| **huggingface.co** | request metadata + ดาวน์โหลด weight | ตอน `inspect` และตอน controller `download` | ไม่ได้ (เป็นแหล่งโมเดล) · ใช้ mirror ภายในได้ด้วย `HF_ENDPOINT` |
| **Docker registry** | pull runtime image | ตอน controller `start`/`download` ครั้งแรก | pre-pull ล่วงหน้าได้ / air-gapped ใช้ `docker save` |

**ไม่เคยส่งออก**: API key ทุกชนิด, HF token, weight ของโมเดล, ชื่อผู้ใช้/hostname, เนื้อหา prompt ของผู้ใช้ปลายทาง

LMDS **ไม่มี telemetry** ไม่มีการเก็บสถิติกลับมาที่ผู้พัฒนา

## Secret เก็บที่ไหน

ลำดับการอ่าน: **environment variable → OS keyring → ไฟล์ `~/.config/lmds/credentials` (0600)**

- `config.yaml` **ไม่เคย**มี secret — มีแต่ชื่อ provider / base URL / ค่า default
- `lmds config show` mask ทุก key เสมอ (ปลอดภัยต่อการแคปหน้าจอส่งกัน)
- ถ้าสิทธิ์ไฟล์ credentials หลวมกว่า 0600 ระบบจะเตือนตอน `config show`
- keyring เป็น optional extra — เครื่อง server ที่ไม่มี desktop session จะ fallback ไปไฟล์ 0600 เอง

**HF token ไม่ถูกฝังใน bundle** — controller อ่านจาก env `HF_TOKEN` ตอน `download` เท่านั้น

## Audit log

ทุกครั้งที่เรียก LLM ระบบเขียน prompt / คำตอบดิบ / แผนที่ได้ ลง `~/.config/lmds/sessions/`
โดยผ่าน `redact()` ก่อนเสมอ — ไว้ตรวจย้อนหลังว่าแผนแต่ละอันมาจากอะไร · ลบได้ถ้าไม่ต้องการเก็บ

## จุดที่ผู้ใช้ต้องอนุมัติเอง (ระบบไม่ตัดสินใจแทน)

1. **Flag นอก allowlist** เช่น `--trust-remote-code` — ถามทีละตัว default = ไม่อนุมัติ
   อนุมัติหลังอ่านไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน `SPECIAL_FILES.md` ของ bundle)
2. **ไฟล์ runtime ภายนอก** (`runtime_assets`) — เป็นโค้ดที่จะถูก mount เข้า container และรันจริง
   รับเฉพาะ HTTPS จาก huggingface.co / hf.co / raw.githubusercontent.com / github.com / gitlab.com,
   ชื่อไฟล์ต้องเป็น basename ล้วน, และ**ต้องอนุมัติรายตัวเสมอ**แม้ LLM จะเสนอมาเอง
3. **ลบไฟล์** — `lmds remove` แสดงรายการ + ขนาดทั้งหมดก่อน แล้วถามยืนยัน (default = ไม่ลบ)

## ความปลอดภัยของ endpoint ที่ deploy ออกไป

⚠️ **ค่า default คือ bind `0.0.0.0` และไม่มี API key** — ใครที่เข้าถึงเครือข่ายเดียวกันยิงโมเดลได้ทันที
controller จะพิมพ์คำเตือนหลัง `start` ทุกครั้งที่เป็นแบบนี้

```bash
./xxx-single.sh restart --bind 127.0.0.1              # ใช้เฉพาะในเครื่อง
API_KEY=$(openssl rand -hex 24) ./xxx-single.sh restart   # หรือบังคับ Bearer token
```

ข้อจำกัดที่ควรรู้: `API_KEY` ถูกส่งเข้า container ผ่าน env → ผู้ที่ใช้ `docker` บนเครื่องเดียวกัน
อ่านได้ด้วย `docker inspect` (ไม่ใช่ช่องโหว่ต่อคนนอก แต่ไม่ควรใช้ key เดียวกับระบบอื่น)

## Prompt injection

เนื้อหาใน model card / README ของโมเดลถือเป็น **ข้อมูล ไม่ใช่คำสั่ง** — system prompt สั่งให้ LLM
เพิกเฉยต่อข้อความที่พยายามสั่งงาน และให้ใส่คำเตือนแทน · ต่อให้ LLM หลงจริง ทุกค่ายังถูก harden
ซ้ำด้วยข้อเท็จจริงจาก ModelReport/FitReport และ flag/image/asset ยังต้องผ่าน allowlist + การอนุมัติของผู้ใช้อยู่ดี

## ตรวจ bundle ที่รับมาจากคนอื่น

```bash
lmds validate <โฟลเดอร์>
```

รัน quality gates ทั้ง 8 ด่านรวม secret scan และตรวจ `PACKAGE_SHA256SUMS` ว่าไม่มีใครแก้ไฟล์

## แจ้งช่องโหว่

อย่าเปิดเป็น public issue — ติดต่อผู้ดูแลโดยตรงที่
[facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)
