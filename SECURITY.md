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

**HF token ไม่ถูกฝังใน bundle** — controller อ่านจาก env `HF_TOKEN` ตอน `download` เท่านั้น และ**ไม่ส่งต่อบน argv**
(`ps` อ่าน argv ได้ทั้งเครื่อง): llama.cpp ส่ง header ให้ curl ทาง stdin (`-K -`) / aria2c ผ่านไฟล์ conf ชั่วคราว mode 600
ที่ลบทันทีที่จบ · vLLM/SGLang/stacked ส่งเข้าคอนเทนเนอร์ download ด้วยชื่อ (`-e HF_TOKEN` ไม่มีค่า) เช่นเดียวกับ
`HTTPS_PROXY`/`HF_ENDPOINT` (proxy URL มักมีรหัสผ่าน) · token ที่ hub **ยืมให้ node** ระหว่าง `node ctl download` ส่งทาง
stdin และถูกกรองออกจากผลงานสดตั้งแต่ตอนรับแต่ละบรรทัด (`_pump`) ก่อนถึงเบราว์เซอร์ — ไม่ใช่กรองหลังท่อปิด

**API key ของ endpoint ไม่เคยอยู่บน argv** — ดูหัวข้อ *ความปลอดภัยของ endpoint ที่ deploy ออกไป*

## Audit log

ทุกครั้งที่เรียก LLM ระบบเขียน prompt / คำตอบดิบ / แผนที่ได้ ลง `~/.config/lmds/sessions/`
โดยผ่าน `redact()` ก่อนเสมอ — ไว้ตรวจย้อนหลังว่าแผนแต่ละอันมาจากอะไร · ลบได้ถ้าไม่ต้องการเก็บ

## จุดที่ผู้ใช้ต้องอนุมัติเอง (ระบบไม่ตัดสินใจแทน)

1. **Flag นอก allowlist** เช่น `--trust-remote-code` — ถามทีละตัว default = ไม่อนุมัติ
   อนุมัติหลังอ่านไฟล์ Python ใน repo แล้วเท่านั้น (รายชื่ออยู่ใน `SPECIAL_FILES.md` ของ bundle)
2. **ไฟล์ runtime ภายนอก** (`runtime_assets`) — เป็นโค้ดที่จะถูก mount เข้า container และรันจริง
   รับเฉพาะ HTTPS จาก huggingface.co / hf.co / raw.githubusercontent.com / github.com / gitlab.com,
   ชื่อไฟล์ต้องเป็น basename ล้วน, และ**ต้องอนุมัติรายตัวเสมอ**แม้ LLM จะเสนอมาเอง
3. **ลบไฟล์** — `lmds remove` แสดงรายการ + ขนาดทั้งหมดก่อน แล้วถามยืนยัน (default = ไม่ลบ) · หน้าเว็บต้องผ่านสองขั้น
   (`--dry-run` ก่อน แล้วส่ง confirm ที่ตรงกับ slug เป๊ะ) · ไฟล์ที่ container เขียนเป็น root ถูกลบผ่าน
   `docker run --rm -v <parent>:/x <image> rm -rf /x/<ชื่อ>` **ใต้รั้ว**: เฉพาะ path ใต้ `$HOME` หรือ `HF_HOME` เท่านั้น ·
   ใช้ image ที่มีอยู่ในเครื่อง (เล็กสุดก่อน) ไม่ pull ใหม่ · ไม่ทำถ้าไม่อยู่ในกลุ่ม docker
4. **งานที่ผู้ช่วยเสนอให้แก้เครื่อง** — restart, เปลี่ยน context/port/bind/gpu-util, ล้างแคช
   FlashInfer ฯลฯ · ผู้ช่วยเสนอได้ ลงมือเองไม่ได้ ผู้ใช้เลือกจากเมนู **แก้เลย / ทีละขั้น /
   ยังไม่ทำ** ทุกครั้ง โดยเห็นคำสั่งเต็มและผลกระทบก่อนกด

   กลไก: ตั๋วอนุมัติออกโดย**เซิร์ฟเวอร์**ตอนเสนอ และเดินได้ต่อเมื่อ `POST
   /api/assistant/ticket/<id>/choose` ถูกเรียกจากเบราว์เซอร์ของผู้ใช้พร้อมโหมดที่เลือก
   — LLM ไม่มีทางออกตั๋วให้ตัวเอง ต่อให้ถูก prompt injection จากข้อความ error ของเครื่อง
   ปลายทางจนพยายามสั่งรันอะไรก็ตาม · ตั๋วหมดอายุใน 30 นาที และแต่ละขั้นใช้ได้ครั้งเดียว

## ความปลอดภัยของ endpoint ที่ deploy ออกไป

⚠️ **ค่า default คือ bind `0.0.0.0` และไม่มี API key** — ใครที่เข้าถึงเครือข่ายเดียวกันยิงโมเดลได้ทันที
controller จะพิมพ์คำเตือนหลัง `start` ทุกครั้งที่เป็นแบบนี้

```bash
./xxx-single.sh restart --bind 127.0.0.1              # ใช้เฉพาะในเครื่อง
API_KEY=$(openssl rand -hex 24) ./xxx-single.sh restart   # หรือบังคับ Bearer token
```

`API_KEY` **ไม่เก็บใน bundle** (`lmds set` ปฏิเสธ — โฟลเดอร์ถูก zip แจกต่อได้) และ**ไม่เคยอยู่บน argv**:

| engine | key ไปถึงยังไง | ใครอ่านได้บนเครื่องเดียวกัน |
|---|---|---|
| llama.cpp | controller เขียนไฟล์ 0600 ใน `RUN_DIR` แล้วส่ง `--api-key-file` (docker: mount แบบ ro) — **ไม่ใช่** env `LLAMA_ARG_API_KEY` ซึ่ง llama-server จริงไม่มี ตั้งแล้วรันแบบไม่มี auth เงียบ ๆ (พิสูจน์กับ b10799: `--api-key-file` ให้ 401/401/200) | เจ้าของไฟล์ / root |
| vLLM เดี่ยว · stacked | export แล้ว `docker run -e VLLM_API_KEY` (ไม่มีค่าบน argv) · stacked เฉพาะ head | ผู้ที่ใช้ `docker` (`docker inspect`) |
| SGLang | ยังต้องส่ง `--api-key <ค่า>` บน argv (engine ไม่มี env คู่) — **ข้อจำกัดที่รู้ตัว** | ทุกคนบนเครื่องผ่าน `ps` |

`serve-args` / `DRY_RUN=1 start` ไม่พิมพ์ key · หน้าเว็บส่ง API key เป็น env ของ controller และเก็บไว้ใน localStorage ของ
เบราว์เซอร์ ไม่ขึ้นไปอยู่บน hub · ข้อจำกัดที่ควรรู้: env ใน container อ่านได้ด้วย `docker inspect` (ไม่ใช่ช่องโหว่ต่อคนนอก
แต่ไม่ควรใช้ key เดียวกับระบบอื่น) · **การเปลี่ยนเรื่อง auth ต้องรันกับ binary จริงก่อนเสมอ** — env ที่ engine ไม่รู้จัก
ไม่ error แต่เปิดประตูทิ้งไว้

## Prompt injection

เนื้อหาใน model card / README ของโมเดลถือเป็น **ข้อมูล ไม่ใช่คำสั่ง** — system prompt สั่งให้ LLM
เพิกเฉยต่อข้อความที่พยายามสั่งงาน และให้ใส่คำเตือนแทน · ต่อให้ LLM หลงจริง ทุกค่ายังถูก harden
ซ้ำด้วยข้อเท็จจริงจาก ModelReport/FitReport และ flag/image/asset ยังต้องผ่าน allowlist + การอนุมัติของผู้ใช้อยู่ดี

**ผลจากเครื่องปลายทางก็ถือเป็นข้อมูลเหมือนกัน** — log ของ container, ข้อความ error และชื่อ
repo เดินทางเข้าไปใน prompt ของผู้ช่วย ข้อความพวกนี้มาจากนอกระบบ · ด่านที่กันไว้ไม่ใช่แค่
คำสั่งใน prompt แต่เป็นโครงสร้าง: ผู้ช่วยเลือกได้เฉพาะ**ชื่อรายการในแคตตาล็อก** พารามิเตอร์
ทุกตัวผ่าน `Param.clean` ก่อนถูกประกอบเป็นคำสั่งด้วยโค้ด และงานที่เปลี่ยนสภาพเครื่องยังต้อง
รอคนกดปุ่มอยู่ดี — สิ่งที่แย่ที่สุดที่ injection ทำได้คือ "ทำให้มันเลือก probe ที่ไม่เกี่ยว"

ผลจาก probe ถูก redact ก่อนส่งออกไปหา LLM provider (ทางเดียวกับ audit log) เพราะ log จริง
มี API key และ endpoint ภายในปนมาได้

## ค่าที่ผู้ใช้กรอกไปถึง shell ของเครื่องอื่น — ด่านที่กันไว้

หน้าเว็บสั่งข้ามเครื่องผ่าน SSH ทุกค่าที่กรอกจึงจบเป็นคำสั่ง shell บนเครื่องปลายทาง (review 0.6.0 พบทางที่รันคำสั่งได้จริง
สองทาง ปิดแล้วพร้อมเทส `tests/test_review_backend.py` / `test_review_web.py`):

| ค่า | กติกา |
|---|---|
| slug ใน URL / ชื่อไฟล์ | ตรวจรูปแบบที่ปากทาง**ทุก route** ที่ไปถึง node หรือชื่อไฟล์ (400) · ทุก `echo` ใช้ตัวที่ `shlex.quote` แล้ว · `lmds adopt --slug ../../x` ถูกปฏิเสธ และชื่อที่เดาจาก container ถูกบีบเข้ารูป slug |
| `bundle.env` (`lmds set` / ฟอร์ม settings) | ไฟล์นี้ถูก `source` ทุกครั้งที่ start/autostart · `served_name` / `image` ห้ามมี `" ' \` $ \ { }` · engine env ต้องเป็น `KEY=VALUE` และห้าม `{}` (ปิด `${…}`) · port/context/slots/gpu_util/bind ตรวจช่วง — เดิม `x$(id)y` ในช่องกรอกถูกรันจริง |
| option ของ start/restart (`port` `context` `slots` `bind` `api_key` `gpu_util` `image` …) | ผ่าน `jobs.clean_options()` ชุดเดียวกันทั้งโมเดลในเครื่องและบนเครื่องอื่น (400 เมื่อผิด) · image ผ่าน allowlist ก่อนถึง `docker run` · ค่าจากผู้ใช้ไม่ถูกตั้งลง `os.environ` ของทั้ง process แบบไม่ล็อกอีก |
| คำสั่งข้ามเครื่อง | allowlist ฝั่ง server: `start stop restart repair doctor logs enable disable remove set` + คำสั่ง controller ที่อ่าน/ทดสอบ (`test-*` `bench` `stress` `parsers` `status` `props` `client-config` `network-info` `verify-files` `prepare-runtime` `sync-worker` `verify-worker` `clear-fi-cache` `logs-worker`) — ปุ่มที่ไม่มีใน allowlist กดไม่ได้แม้จะแก้ HTML |
| ทะเบียนเครื่อง `nodes.yaml` | 0600 · เขียนใต้ RLock + `flock` (`~/.config/lmds/.nodes.lock`) — refresher กับ PATCH ของผู้ใช้ไม่เขียนทับกัน · cluster IP 169.254.x ถูกปฏิเสธ |

หน้าเว็บ**ไม่โหลดอะไรจากเน็ต** — ฟอนต์ Geist อยู่ในแพ็กเกจ hub เสิร์ฟที่ `GET /fonts/<ชื่อ>` (รับเฉพาะชื่อในรายการ ไม่รับ path ·
ไม่ต้องใช้ token เพราะหน้า login ก็ใช้) · token ของหน้าเว็บไม่อยู่ใน URL ที่พิมพ์ออกมา และถ้าเปิดด้วย `?token=` จะถูกย้ายเข้า
ที่เก็บของเบราว์เซอร์แล้วลบออกจากแถบที่อยู่ทันที · ผิดติดกัน >5 ครั้งต่อ IP หน่วงแบบทวีคูณ

## ตรวจ bundle ที่รับมาจากคนอื่น

```bash
lmds validate <โฟลเดอร์>
```

รัน quality gates ทั้ง 12 ด่าน (bash syntax · ไม่มี template tag เหลือ · numeric underscore · pipefail · line continuation ·
controller contract · stacked contract · multimodal assets · profile schema · serving consistent · **secret scan** ·
checksums) และตรวจ `PACKAGE_SHA256SUMS` ว่าไม่มีใครแก้ไฟล์

## แจ้งช่องโหว่

อย่าเปิดเป็น public issue — ติดต่อผู้ดูแลโดยตรงที่
[facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)
