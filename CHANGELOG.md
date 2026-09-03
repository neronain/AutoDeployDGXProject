# Changelog

## 0.5.2 — 2026-09-04

**หน้าเว็บหักหน่วยความจำที่เครื่องปลายทางใช้อยู่แล้ว ก่อนบอกว่าโมเดล fit — และวาดให้เห็นว่า budget มาจากอะไร**

รายงานจากฟลีต 2026-09-04: เริ่มมีคนใช้จริงหลายคนบนเครื่องเดียว (vLLM หนึ่ง · llama.cpp สอง) แล้ว
"ทำงานได้ไม่เต็มที่" · ต้นเหตุคือเคสเดียวกับ msi-5 เมื่อ 2026-08-28 — analyzer รับ `reserved_gb` ไปหักจาก
budget ได้ตั้งแต่วันนั้น แต่**หน้าเว็บไม่เคยส่ง** (โค้ดใน `_budget_gb` เขียนสารภาพไว้เองว่า "ตัวเลขนี้มีให้อ่าน
อยู่แล้ว แค่ไม่เคยถูกส่งเข้ามาถึงตรงนี้") · CLI ส่งเฉพาะตอน target คือเครื่องตัวเอง · ทุก deploy จากหน้าเว็บ
จึงถูกวางแผนจาก "เครื่องว่าง" เสมอ ไม่ว่าจะมีอะไรรันอยู่บนเครื่องนั้น

- `POST /api/deploy/analyze` รับ `machine`/`worker` (หน้าเว็บส่งให้เองจากช่อง Run on) → อ่านหน่วยความจำที่
  ใช้อยู่จาก**แคช inventory** ของเครื่องนั้น (ไม่ยิง SSH เพิ่ม — refresher สำรวจอยู่แล้วทุก 15 วิ) แล้วส่งเข้า
  `analyze(reserved_gb=…)`
  · stacked หักตามเครื่องที่แน่นสุด × จำนวนเครื่อง เพราะ tensor parallel แบ่งเท่ากัน เครื่องที่เหลือน้อยสุดเป็นตัวจำกัด
  · ไม่เลือกเครื่อง + ไม่มี target = เครื่องนี้เอง (เท่า CLI) · preset ล้วน = เครื่องสมมติ ไม่หัก
  · ยังไม่มีข้อมูลของเครื่องนั้น (เพิ่งแอด/ต่อไม่ได้) → คิดจากความจุเต็มพร้อมบอกตรง ๆ ว่าตัวเลขอาจสูงเกินจริง
    — แยกจาก "สำรวจแล้วว่าง" ซึ่งไม่เตือน
- `FitReport` มี `capacity_gb` / `reserved_gb` / `reserved_source` / `kv_budget_gb` และ payload มี
  `kv_at_context_gb` → หน้า plan วาดแถบ **capacity · OS+engine overhead · already in use · weights · KV ที่
  context ที่เลือก · spare** และวาดใหม่ทันทีที่พิมพ์ context หรือสลับ fp8 (ใช้ `kv_bytes_per_token` ที่
  `/context` ตอบอยู่แล้ว ไม่ยิงเพิ่ม)
- **ความแน่นชั่วคราวไม่บล็อกการสร้าง bundle** — ผู้ใช้ทักทันทีที่ลองบนฟลีต: "จริงต้องทำได้ เพราะลูกค้า
  อาจจะยังไม่ได้รัน เพียงแต่ต้องการรู้ค่าและ deploy ลงไปก่อน" · deploy = วาง bundle ไว้ที่เครื่อง ของอื่น
  หยุดทีหลังได้ · จึงคิดสองชั้น: **เครื่องเปล่า** ตัดสินว่าสร้างได้ไหม (ใส่ไม่ได้จริงถึงบล็อก) ·
  **ตอนนี้** (หักของที่รันอยู่) แค่ติดป้าย "deploy ได้ แต่ start ตอนนี้ไม่ได้ — ขาด X GB · ลด context เหลือ N
  แล้ว start ได้เลย หรือหยุด <ชื่อโมเดล@เครื่อง> ก่อน" · payload มี `now_verdict / now_budget_gb /
  now_max_safe_context / now_short_gb / running_now`
- หน้า plan วาด `fit.notes` ใต้แถบแล้ว (เดิมทิ้งเงียบ ๆ) — ผู้ใช้จึงเห็น "ยังไม่มีข้อมูลของเครื่องนี้ คิดจาก
  ความจุเต็ม" / "stacked หักตามเครื่องที่แน่นสุด" / "target ยังไม่เคยทดสอบ" ที่พ่วงอยู่กับตัวเลขบนแถบ
- `profiler.memory_held_gb()` เป็นตัวเดียวที่ทั้ง CLI และเว็บใช้ (เดิม CLI มีสำเนาของตัวเองใน main.py)
- เทส 9 ข้อใน `tests/test_web_reserved_memory.py` + 1 ข้อใน `test_fit_and_checksums.py` — ทุกข้อล้มกับโค้ดเดิม

ยังไม่ทำในรอบนี้ (คนละเรื่อง ต้องตกลงสมมติฐานก่อน): แปลง concurrency เป็น "จำนวนคน" ด้วยความยาวคำขอปกติ ·
โมเดลคำนวณของ llama.cpp `--parallel` ที่จอง KV ทั้งก้อนล่วงหน้า (ตารางตอนนี้คิดแบบ paged ของ vLLM ให้ทั้งคู่)

**ระบบเติม parser / image / env ให้ตามโมเดล — ไม่ต้องรู้ชื่อเอง**

ผู้ใช้ 2026-09-04 ดูฟอร์ม settings แล้วถาม "ช่องเริ่มเยอะ … tool/reasoning parser ถ้าไม่ทราบ จะทำอย่างไร
กลัวใส่ผิด แล้วไม่มีให้ใช้งาน" · ความจริงคือระบบ*รู้*อยู่แล้ว — `arch_notes()` เขียนคำเตือนว่า Qwen3 ต้องใช้
`qwen3_xml` + `qwen3`, Gemma 4 ต้องใช้ `gemma4` และ NVFP4 บน GB10 ต้องใช้ image+env ชุด marlin —
แต่เก็บเป็น**ข้อความ** แล้วปล่อย `parser = null` ให้ผู้ใช้ไปพิมพ์เอง ซึ่งคนไม่รู้ก็ไม่กด และคนที่เดาไป
hermes ก็ได้ tool call เป็นข้อความ (msi-2 2026-09-02)

สองชั้น:
- **bundle ใหม่เกิดมาถูก** — `brain/families.py` เก็บความรู้นั้นเป็น*ค่า* · planner ใส่ `tool_calling.parser`
  / `reasoning.parser` ให้ตระกูลที่รู้แน่ (Qwen3/3.5/3.6 · Qwen3-Coder · Gemma 4) ตาม engine
  (vLLM/SGLang ใช้คนละชุดชื่อ · llama.cpp ไม่มีแฟล็กนี้) = เปิด tool calling ให้เลย พร้อมคำเตือนใน plan
  ว่าเปิดให้แล้วและปิดยังไง · สูตรที่รันผ่านจริงยังชนะเสมอ · ตระกูลที่ไม่รู้ **ไม่เดา**
- **bundle ที่มีอยู่แล้ว** — ปุ่ม **เติมให้ตามโมเดล** ในฟอร์ม settings ของทุกเครื่อง (+ `lmds set <slug> --auto`)
  → `fleet/suggest.py` เสนอค่าจาก recipe > กฎตระกูล > กฎฮาร์ดแวร์ (NVFP4/SM121) พร้อม**ที่มาทีละค่า** ·
  แค่เติมให้ดู ยังไม่บันทึกจนกด บันทึกค่า · `GET /api/models/{slug}/settings/suggest` และ
  `/api/nodes/{name}/models/{slug}/settings/suggest` (อ่านจากแคช inventory ไม่ยิง SSH)
- ฟอร์ม node มีช่อง **engine env** แล้ว (เดิมตั้งได้แค่ CLI) และ `image min tokens` ถูกส่งต่อถึง `lmds set`
  บน node แล้ว (เดิมกรอกแล้วหายกลางทาง)
- ค่า NVFP4/SM121 ในกฎอ่านจาก bundle.env ของ spark-head ที่รันอยู่จริง (`avarok/dgx-vllm-nvfp4-kernel` +
  `VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_USE_FLASHINFER_MOE_FP4=0
  VLLM_MARLIN_USE_ATOMIC_ADD=1`) ไม่ใช่จากข้อความเตือน
- เทส: `test_suggest_settings.py` (กฎตระกูล · ทุกชื่อผ่าน `_harden_parsers` · recipe ชนะกฎ · ไม่รู้ = ไม่เดา)
  · `test_suggest_api_ui.py` · `test_brain` ปรับให้ Qwen3 บน vLLM ได้ parser ตั้งแต่ plan

**`--image-min-tokens 1024` บังคับเฉพาะ Qwen-VL — Gemma-4 และตระกูลอื่นกลับไปใช้ค่าของ projector**

0.5.1 (17ed363) ใส่ `IMAGE_MIN_TOKENS=1024` ให้ทุก controller ที่มี projector เพื่อแก้ความแม่นของ Qwen-VL
แต่คำเตือนนั้นเป็นของ Qwen-VL เท่านั้น · projector ตระกูลอื่นมีเพดานของตัวเอง — Gemma-4 รับได้ 280 tokens
(645,120 px) · บังคับ 1024 (2,359,296 px) → llama.cpp `clip_init: image_max_pixels is less than
image_min_pixels` → server ตายก่อน health · เคสจริง 2026-09-04 dgx-veerasiam/gemma-4-12b start ไม่ขึ้น
"ทั้งที่เมื่อวานยังรัน" · สแกนฟลีตพบ 5 bundle ที่จะพังเหมือนกันทันทีที่ถูก restart (รวม muse-glimmer บน msi-4
ที่กำลังรันอยู่)

- renderer ส่ง `image_min_tokens_default` = 1024 เมื่อ architecture/model_id เป็นตระกูล Qwen · อื่น ๆ ว่าง
  (ค่าจากไฟล์ = พฤติกรรมก่อน 0.5.1 ที่ผ่านการใช้งานจริง) · เทสเดิมใช้ fixture Gemma แต่ assert 1024 —
  คือบั๊กในเทสเอง แก้เป็น fixture Qwen3-VL และเพิ่มเคส Gemma ต้องว่าง
- `lmds set --image-min-tokens N|auto` และหน้าเว็บ settings — `auto` เขียนลง bundle.env เป็นค่าว่างแบบ
  "set แต่ว่าง" (`IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-}"`) ไม่ใช่ลบทิ้ง เพราะ controller ที่สร้างก่อนหน้านี้
  มี 1024 ฝังอยู่ · read() คืน `auto` กลับมาให้ round-trip ผ่าน `lmds set` ครั้งถัดไปได้
- bundle ที่ deploy ไปแล้ว 5 ตัวถูกตั้ง `auto` ให้ตรง ๆ บน node แล้ว (ไม่ต้องอัปเดต node ก่อน)

**ไฟล์ mmproj ที่ชื่อขึ้นต้นด้วยชื่อโมเดล ถูกจำได้แล้ว — vision ไม่หายเงียบ ๆ**

เจอตอนลองฟีเจอร์ข้างบนกับ `llmfan46/gemma-4-31B-it-uncensored-heretic-NVFP4-GGUF`: หน้าเลือกไฟล์เสนอ
`gemma-4-31B-it-uncensored-heretic-mmproj-BF16.gguf` (1.1 GB) เป็นตัวเลือก weights · เพราะตรวจแค่
`startswith("mmproj")` · ผลที่แย่กว่าคือ `has_mmproj=False` → capabilities บอก "โหลดภาพไม่ได้" และ controller
ไม่ได้ `--mmproj` ทั้งที่ Gemma-4 เป็นโมเดลภาพ

- จับ `mmproj` เป็น token ที่คั่นด้วย `-` `_` `.` หรือหัว/ท้ายชื่อ (ไม่จับกลางคำ) · **mtp ยังตรวจเฉพาะขึ้นต้น**
  เพราะชื่ออย่าง `…-Native-MTP-Preserved-APEX-…` คือ weights ที่เก็บหัว MTP ไว้ ไม่ใช่ไฟล์ mtp แยก
- เทส 4 ข้อ `tests/test_gguf_variant_roles.py` ด้วยชื่อไฟล์จาก repo จริง

## 0.5.1 — 2026-09-03

วันเดียวบนฟลีตจริง 14 เครื่อง: deploy 4 โมเดลใหม่ (spark-02 ×2, spark-head, spark-worker, spark04)
แล้วเก็บทุกอย่างที่พังกลับมาเป็นโค้ด · ทุกข้อมีเทสที่ล้มกับโค้ดเดิม

**install.sh สร้าง venv ได้บนเครื่องที่ไม่มี python3-venv โดยไม่ต้อง sudo**

เพิ่ม node RTX4000 (Ubuntu 24.04, Python 3.12) จากหน้าเว็บ 2026-09-03: `python3 -m venv --help` ผ่าน
แต่สร้าง venv จริงล้มด้วย "ensurepip is not available" (Ubuntu แยก ensurepip ไปไว้ใน python3-venv)
· install.sh ตายพร้อมคำแนะนำ "ลบ venv ทิ้งแล้วลองใหม่" ซึ่งไม่เกี่ยวกับสาเหตุ · หน้าเว็บรันแบบไม่มี sudo
จึงลง apt ไม่ได้อยู่แล้ว

- `make_venv()` ตรวจ `import ensurepip` ตรง ๆ · ไม่มี → ถ้าสั่ง sudo ได้และไม่ได้ skip prereq ก็ลง
  python3-venv ให้ · ไม่งั้น `venv --without-pip` แล้วดึง pip จาก bootstrap.pypa.io (ขั้น pip install
  ถัดไปต้องถึงเน็ตอยู่แล้ว) · ล้มจริงค่อยบอกว่า "sudo apt install python3-venv"
- เทสต์ด้วย python3 ปลอมที่ไม่มี ensurepip: ต้องได้ pip กลับมาโดยไม่แตะ sudo

**`lmds hardware` บน Docker รุ่นใหม่: toolkit ที่ลงแล้วไม่ถูกรายงานว่าหาย และบอกชื่อ target ที่ใช้ได้จริง**

node เดียวกัน (Docker 29 + nvidia-container-toolkit 1.20): `docker run --gpus all … nvidia-smi -L` เห็น GPU
ครบสองใบ แต่ตารางบอก "NVIDIA Container Toolkit ❌ — ติดตั้งก่อน" เพราะตรวจแค่ว่า docker info มี runtime
ชื่อ nvidia ซึ่ง Docker ≥25 ไม่ต้องมี (ส่ง GPU ผ่าน CDI) · และ "Profile: rtx-multi-gpu" ถูกเอาไปใส่
`--target` แล้วโดนปฏิเสธ ทั้งที่ preset `rtx-pro-4000-dual` มีอยู่

- toolkit นับว่ามีเมื่อมี runtime nvidia **หรือ** มี `nvidia-ctk`/`nvidia-container-cli` **หรือ** มี CDI spec ·
  โน้ตแยกกรณี "มี toolkit แต่ runtime ยังไม่ลงทะเบียน" พร้อมคำสั่งที่ใช้เมื่อ container ไม่เห็น GPU จริง ๆ
- แถว "Target สำหรับ deploy" ในตาราง hardware — `suggest_target()` แปลงชื่อการ์ด (+จำนวน) เป็นชื่อ preset
  ที่ `lmds deploy --target` รับ (RTX PRO 4000 ×2 → `rtx-pro-4000-dual`, GB10 → `dgx-spark-single`)

**`lmds node clone` ใช้กับโมเดล vLLM/SGLang ได้ · `lmds node run` ส่ง argument ที่มีช่องว่างถึงปลายทางครบ**

RTX4000 2026-09-03: clone Qwen3.6-35B NVFP4 จาก spark04 ตอบ "ยังไม่มีไฟล์โมเดลบน spark-04 ()" ทั้งที่ 22 GB
อยู่ครบ — controller ของ vLLM ไม่มี `MODEL_DIR` (weight อยู่ใน `HF_HOME/hub/models--org--name`) · และ
`lmds node run RTX4000 set … --extra-args "--a 1 --b 2"` ถึงปลายทางเป็น 4 argument แยกกัน typer จึงตอบ
"No such option: --b"

- `inspect_source` หาโฟลเดอร์ HF cache จาก `MODEL_ID`/`HF_HOME` เมื่อไม่มี `MODEL_DIR` นับไฟล์จริงใน blobs
  (snapshots เป็น symlink) และบอกตำแหน่งที่หาเมื่อไม่เจอ · เทสต์รันสคริปต์ฝั่งต้นทางด้วย bash จริง
- `node run` quote ทุก argument เหมือนที่ `node ctl` ทำอยู่แล้ว

**publish พับค่าที่ `lmds set` ไว้ลง header — คลังได้สูตรที่รันได้จริง ไม่ใช่ค่าเดาของ plan**

spark04 / spark-worker 2026-09-03: ทั้งคู่รันได้เพราะ `lmds set --image <digest v0.28.0>
--tool-parser qwen3_xml --reasoning-parser qwen3 --extra-args "…MTP…"` แต่ค่าพวกนี้อยู่ใน
`bundle.env`/`bundle.args` · header ของ controller ยังเป็น image จาก plan ที่ start ไม่ขึ้น
และไม่มี parser → `lmds recipes --publish` ส่งค่าที่ล้มขึ้นคลัง เครื่องที่ sync มาก็เจอปัญหาเดิมซ้ำ

- `publish` พับ **ค่าของโมเดล** จาก bundle.env (`VLLM_IMAGE`/`LLAMACPP_IMAGE`, `TOOL_CALL_PARSER`,
  `REASONING_PARSER`, `ENGINE_ENV`, `CHAT_TEMPLATE`, `MMPROJ_FILE`, `IMAGE_MIN_TOKENS`) ลงเป็นค่าตั้งต้น
  ใน header · ค่าของเครื่อง (port, context, gpu-util, slots, bind, ชื่อที่เสิร์ฟ) ไม่พับโดยเจตนา
- แฟล็กเพิ่มจาก `bundle.args` ลงที่ `EXTRA_SERVE_ARGS_DEFAULT='…'` (single quote — JSON มี `}` ที่จะตัด
  `${VAR:-…}` ขาด) · template ทั้งสามอ่านค่านี้เมื่อไม่มี env และไม่มี bundle.args
- `PROFILE.yaml` มี `overrides:` ให้คน review เห็นว่าค่าไหนต่างจาก plan · CLI บอกว่าพับอะไรไป
  และเตือนเมื่อ controller รุ่นเก่าไม่มีบรรทัดรองรับ (ต้อง `lmds rebuild` ก่อน)
- ฝั่ง sync อ่าน single quote ได้แล้ว · สูตรพก `tool_parser` / `reasoning_parser` / `engine_env` /
  `extra_args` ไปด้วย
- publish ตั้ง git identity ให้ repo ที่ clone มา (local config ของ repo นั้น) เมื่อเครื่องไม่เคยตั้ง —
  hub จริงล้มทั้ง 23 ตัวด้วย "unable to auto-detect email address" · และไม่รายงาน image ของอีก
  engine (`lmds set --image` เขียนทั้ง VLLM_IMAGE/LLAMACPP_IMAGE) ว่าพับไม่ได้

**controller บอกสาเหตุจริงเมื่อ container ตายก่อน health**

dgx-spark04 2026-09-03: bundle ตั้ง gpu-util 0.4 กับ context 262144 → vLLM โยน
`ValueError: No available memory for the cache blocks` แต่บรรทัดสุดท้ายของ log คือ
`RuntimeError: Engine core initialization failed. See root cause above` · hub เห็นแค่นั้น
กับข้อความ "container หยุดก่อน health ผ่าน — ดู logs" คนจึงต้อง ssh ไป grep เอง

- `explain_crash()` ใน single-vllm / single-sglang / stacked (head) หยิบ exception บรรทัดแรก
  ที่ไม่ใช่ wrapper ออกมาแสดง พร้อมคำแนะนำที่ผูกกับค่าที่ตั้งอยู่จริง (KV cache ไม่พอ →
  `lmds set --gpu-util` / ลด `--context` · ptxas e2m1 → ENGINE_ENV marlin)
- log ที่ไม่มี exception (เช่นโดน OOM-kill) ยังคงเงียบ ไม่พ่นบรรทัดว่าง

**adopt เตือนเมื่อคำสั่ง start ไป `hf download` ก่อนเสิร์ฟ**

dgx-spark03 2026-09-03: สร้าง container Nemotron ใหม่จากคำสั่งเดิมเป๊ะ
(`hf download nvidia/… && trtllm-serve nvidia/…`) แล้ววนล้ม 15 รอบ — repo gated + HF_TOKEN
หมดอายุ → ดึง revision ใหม่ได้ครึ่งเดียว (401, 6 ไฟล์ ไม่มี safetensors) แล้ว serve ชี้ไปที่นั่น
ทั้งที่ snapshot ที่ครบ (44 ไฟล์) อยู่บนดิสก์มาตั้งแต่ มิ.ย. · ตัวเดิมรอดมาได้เพราะไม่เคย restart

- สคริปต์ที่ adopt สร้างมีคำเตือนติดไว้ตรงคำสั่ง พร้อมทางแก้: ชี้ path ของ snapshot ที่ครบตรง ๆ
  และตั้ง `HF_HUB_OFFLINE=1`

**`lmds adopt` คัดลอก HF_TOKEN ลงสคริปต์บนดิสก์**

`lmds adopt trtllm-nemotron` บน dgx-spark03 เขียน `--env HF_TOKEN=hf_…` ลง
`bundles/…-adopted.sh` (0755) — ทุก user บนเครื่องอ่านได้ และไฟล์นี้ถูก zip/push ข้ามเครื่องได้ ·
ขัดกับหลักของ LMDS ที่ความลับเดินทางทาง env/stdin เท่านั้น

- env ที่ชื่อเข้าข่าย TOKEN/SECRET/PASSWORD/API_KEY/CREDENTIAL เหลือแค่ชื่อ (`--env HF_TOKEN`)
  docker หยิบค่าจากเชลล์ที่สั่ง start · สคริปต์บอกไว้ว่าต้อง export ก่อน

**bundle ของเราเองถูกนับเป็น "นอกระบบ"**

dgx-veerasiam ขึ้น "นอกระบบอีก 3" ทั้งที่ทั้ง 3 คือ llama-server ที่ bundle ของ LMDS start เอง ·
ทุกเครื่องที่รัน vLLM ขึ้นซ้ำสองรายการ (container + `VLLM::EngineCore`) · `foreign_workloads`
คัดเฉพาะ container ตามชื่อ แต่ process จาก nvidia-smi ไม่เคยถูกเทียบกับอะไรเลย

- process ที่ pid (หรือบรรพบุรุษ) อยู่ใน `server.pid` ของ bundle = ของเรา
- process ใน container ที่ `server.meta` ลงทะเบียนไว้ (รวม adopt ที่ชื่อไม่ขึ้นต้น lmds-) = ของเรา
  อ่านจาก `/proc/<pid>/cgroup` → `docker ps --no-trunc`

**`node push --download` หลุด session แล้วทิ้ง container โหลดไว้โดยไม่มีใครเฝ้า**

download บน node รันผ่าน SSH session ของ hub ตรง ๆ · session หลุด (เน็ตสะดุด / ปิดเทอร์มินัล /
timeout) → controller บน node ตาย แต่ container `lmds-dl-*` ที่มันสั่งไว้ยังอยู่ **โดยไม่มี
watchdog** (ตัวกัน Xet ค้างอยู่ใน controller จึงตายไปด้วย) · เคสจริง spark-worker 2026-09-03:
container โหลด scottgl ค้างที่ "Fetching 33 files 0%" rx 0 MB/s อยู่ 90 นาที ส่วน hub ก็ค้างที่
"โหลด weight บน spark-worker…"

- push --download/--start รันคำสั่งบน node ด้วย setsid+nohup เขียน log + `__RC=` ตอนจบ แล้ว hub
  อ่าน log เป็นช่วง ๆ · session หลุดกลางทางงานยังเดินต่อ สั่งซ้ำได้จากไฟล์เดียวกัน
- `node ctl … download` ยังสตรีมผ่าน session (ต้องส่ง HF_TOKEN ทาง stdin) — ข้อจำกัดที่รู้อยู่

**`lmds node push` ส่ง zip เก่า — ค่าจาก `lmds set` ไม่เคยไปถึงเครื่องปลายทาง**

zip ถูกสร้างตอน `deploy` แล้ว push หยิบไฟล์นั้นส่งตรง ๆ · ทุกอย่างที่ `lmds set` เขียนทีหลัง
(`bundle.env`, `bundle.args`) จึงตกหล่น · เคสจริง 2026-09-03: ตั้ง `--engine-env` marlin ให้
Coder-Next แล้ว push ไป spark-head — container ขึ้นมาโดย**ไม่มี env สักตัว** (`docker inspect`
ว่างเปล่า) · bundle.args ของ Sehyo/Qwen3.5-122B ไม่ถึง spark-worker · ทั้งสองเคส hub รายงาน
"บันทึกแล้ว" และ push รายงาน "ติดตั้งแล้ว" — เงียบทั้งสาย

- push แพ็ก zip ใหม่จากโฟลเดอร์ทุกครั้งก่อนส่ง (`make_zip`) · เทสยืนยันว่าไฟล์ที่เขียนหลัง
  generate อยู่ใน zip และไม่ยัด zip ซ้อน zip

**แฟล็กเพิ่มของ engine ตั้งผ่าน `lmds set` ได้แล้ว — MTP ของ vLLM ไม่ต้องแก้สคริปต์มือ**

จะเปิด MTP ให้ Qwen3.5-122B บน spark-worker ต้องส่ง
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` ซึ่ง `lmds set` ตั้งไม่ได้
(`docs/USAGE.md` ยอมรับไว้เองว่า "ยังตั้งผ่าน LMDS ไม่ได้") ทั้งที่คำเตือนในแผนของ LMDS
แนะให้เปิด · tool/reasoning parser ก็ตั้งได้แค่ตอน `start` จึงหายตอน autostart

ใส่ลง `bundle.env` ไม่ได้: รูป `${VAR:-value}` ของ bash หยุดที่ `}` ตัวแรก —
`X="${X:---speculative-config {"method":"mtp"} --foo}"` ได้ `{"method":"mtp"` กับ `--foo}`
(ทดสอบแล้ว) จึงเก็บใน `bundle.args` แยกต่างหาก controller อ่านทั้งบรรทัดแล้วแตกด้วยช่องว่าง

- `lmds set --extra-args / --tool-parser / --reasoning-parser` + ช่องบนหน้าเว็บ · ลำดับเดิมยังจริง:
  flag > env > ไฟล์ > bundle
- `DRY_RUN=1 ./xxx-single.sh start` (vLLM) พิมพ์ image + argv ที่จะรันจริงโดยไม่แตะ docker/GPU ·
  `./xxx-single.sh serve-args` (llama.cpp) — ใช้พิสูจน์ว่าค่าถึง argv ก่อนรอโหลดโมเดลหลายนาที
- template stacked ครอบด้วย (เดิม stacked ไม่ source `bundle.env` เลย `lmds set` จึงไม่มีผล)

**`lmds set --image` ให้ bundle vLLM ถูกเมินเงียบ ๆ — บั๊กเดียวกับ llama.cpp เมื่อเช้า**

ตั้ง image เป็น digest v0.28.0 ที่พิสูจน์แล้วให้ bundle ของ Sehyo/Qwen3.5-122B แต่ controller
ยังจะใช้ `nvcr.io/nvidia/vllm:26.05` ของ plan · `VLLM_IMAGE` กับ `SERVED_MODEL_NAME` ประกาศ
อยู่เหนือจุด source `bundle.env` ใน template vLLM (บรรทัด 17/28 เทียบกับ 38) · ย้ายบล็อกขึ้นบนสุด
ทั้ง single-vllm และ stacked-vllm พร้อมเทส `DRY_RUN` ที่อ่าน image จริงจาก argv

**คำเตือน "MoE + NVFP4 บน sm_121 = ทางตัน" ผิด — มีสูตรที่รันได้แล้ว**

spark-head 2026-09-03: `ucbye/Qwen3-Coder-Next-NVFP4-GB10` (MoE 512 expert, NVFP4) บน
`vllm/vllm-openai:cu130-nightly@3dbe092e` + env marlin ครบชุด
(`VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_USE_FLASHINFER_MOE_FP4=0
VLLM_MARLIN_USE_ATOMIC_ADD=1`) ได้ **61 tok/s** เดี่ยว / 103 tok/s รวม 3 สาย test-tools ผ่าน ·
ที่ msi-6 ล้มเพราะขาด `VLLM_USE_FLASHINFER_MOE_FP4=0` — vLLM import cutlass fused-MoE (JIT
ทันที) ก่อนดู env marlin · คำเตือนบอกทั้งสูตรที่ผ่านและอาการที่ล้มแทนที่จะบอกให้เลิก

**รายชื่อเครื่องบอก commit แล้ว ไม่ใช่แค่เลข version**

`lmds node list` โชว์ `0.5.0` ทั้ง 13 เครื่องเท่ากันหมด ทั้งที่ทุกเครื่องอยู่ `af01a1e` ส่วน hub อยู่
`f9181ab` (6 คอมมิตที่แก้ restart/bundle.env) — ต้องไล่ `lmds node run <n> version` ทีละเครื่อง ·
`host_payload` ส่ง `lmds_commit` มานานแล้ว แค่ `status_from_probe` ทิ้งไป · ตอนนี้แสดง
`0.5.0 (f9181ab)` ทั้ง CLI, `node install` และหน้าเว็บ

**checkpoint NVFP4 ถูกรายงานพารามิเตอร์แค่ครึ่งเดียว (122B โชว์เป็น 26.8B)**

`lmds inspect scottgl/Qwen3.5-122B-A10B-NVFP4-GB10` บอก **Parameters 26.8B** และ
`Sehyo/Qwen3.5-122B-A10B-NVFP4` บอก 71.2B ทั้งที่ชื่อบอกอยู่ว่า 122B · เอา `safetensors.total`
ของ Hub มาใช้ตรง ๆ ซึ่งนับ *element* ไม่ใช่พารามิเตอร์: U8 ที่อัด NVFP4 สองตัวต่อไบต์ถูกนับ
เป็นหนึ่ง และ scale F8_E4M3 (หนึ่งตัวต่อ 16 พารามิเตอร์ — 117B/16 = 7.3B ตรงเป๊ะกับที่ Hub
รายงาน) ถูกนับรวมเข้าไปด้วย

ไม่กระทบ fit (ใช้ขนาดไฟล์) แต่โผล่ในหน้า inspect และ `MODEL_PROFILE.yaml` ให้คนเข้าใจผิดว่า
โมเดล "เล็ก" แล้วตัดสินใจเรื่องเครื่องผิด

- เมื่อ repo ติดแท็ก 4-bit (nvfp4/mxfp4/awq/gptq/…) และมี U8: นับ U8 ×2 + ชั้นที่ไม่ได้
  quantize (BF16/F16/F32) และไม่นับ F8_E4M3 ที่เป็น scale
- checkpoint BF16 ล้วน หรือ U8 ที่ไม่มีแท็กบอกว่าอัด 4-bit → ใช้ค่าของ Hub ตามเดิม

**bundle llama.cpp ไม่มี `test-tools` ทั้งที่เอกสารและหน้าเว็บบอกว่ามี**

เจอบน spark-02: `./xxx-single.sh test-tools` บน bundle GGUF พิมพ์ help ออกมาแทน ·
`docs/USAGE.md` เขียนว่า "ใช้ได้ทุก bundle" และหน้าเว็บมีปุ่มให้กด แต่ controller ของ
llama.cpp ไม่เคยมีคำสั่งนี้ — มีแต่ฝั่ง vLLM ตั้งแต่เคส Nemotron (2026-08-14)

ต้องยิง `/v1/chat/completions` พร้อม `tools` เองถึงจะรู้ว่า tool calling ใช้ได้ ซึ่งคือ
สิ่งที่คำสั่งนี้มีไว้กันไม่ให้ต้องทำ

llama.cpp ต่างจาก vLLM ตรงที่ **ไม่มี `--tool-parser`** — chat template ที่โหลดผ่าน
`--jinja` เป็นทั้งคนสอนโมเดลให้เขียนรูปแบบและคนแปลกลับ · คำใบ้แบบ vLLM ("ลอง parser
ตัวอื่น") จึงใช้ไม่ได้ ต้องวินิจฉัยคนละทาง:

- ถาม `/props` ก่อน — `chat_template_caps.supports_tools` บอกตรง ๆ ว่า template ที่โหลด
  อยู่รองรับ tools ไหม ถ้าไม่ ทางแก้คือ `--chat-template-file` ไม่ใช่เปลี่ยน parser
- call ดิบหลุดมาใน content = โมเดลเรียกแล้วแต่ template แปลไม่ออก → มักเป็น llama.cpp
  เก่ากว่าโมเดล แนะ `prepare-runtime`
- คิดจนหมด 2048 tokens ก่อนเรียก tool (reasoning model) → บอกว่ายังสรุปไม่ได้ ไม่ตัดสินว่าพัง
- ค่าเริ่มต้นวัดโหมด `auto` ที่ agent ใช้จริง และ auto ไม่ผ่าน = FAIL exit 1 เหมือน vLLM

**`lmds set --model-id` เขียนไฟล์สำเร็จ แต่ชื่อที่ API เสิร์ฟไม่เปลี่ยน**

เจอบน spark-02 ตอนตั้งชื่อโมเดลตัวที่สอง · `lmds set … --model-id qwen3-6-35b-uncensored`
บอกว่าบันทึกแล้ว `bundle.env` ก็มีบรรทัดนั้นจริง แต่ `/v1/models` ยังคืนชื่อ slug เดิม

ทุกบรรทัดใน `bundle.env` เป็นรูป `${VAR:-value}` ซึ่ง **ไม่ทำอะไรเลยถ้าตัวแปรถูกตั้งไปแล้ว**
· บล็อกที่ source ไฟล์นี้อยู่กลาง controller ค่าที่ประกาศ *เหนือ* มันจึงถูกเมินทั้งหมด:

| ตัวแปร | อยู่เหนือ/ใต้บล็อก | `lmds set` มีผลไหม |
|---|---|---|
| `CTX_SIZE`, `API_PORT`, `API_HOST` | ใต้ | ✅ |
| `SERVED_MODEL_NAME` (`--model-id`) | เหนือ | ❌ |
| `LLAMACPP_IMAGE` (`--image`) | เหนือ | ❌ |
| `RUNTIME_MODE`, `CUDA_ARCHITECTURES` | เหนือ | ❌ |

ที่หลอกคือ `CTX_SIZE` ทำงานปกติ — ตั้ง context แล้วเห็นผลทันที จึงเชื่อว่าไฟล์ถูกอ่านแล้ว
และไปหาสาเหตุที่อื่นแทน

- ย้ายบล็อก `bundle.env` ขึ้นเหนือ default ทุกตัว ตามที่คอมเมนต์ของมันเองบอกไว้อยู่แล้ว
- เทสรัน controller จริงกับ `bundle.env` ที่ตั้งชื่อไว้ แล้วเช็คว่า `status` รายงานชื่อนั้น
  (ยืนยันแล้วว่าเทสล้มกับโค้ดเดิม) · อีกตัวคุมว่าลำดับ flag > env > ไฟล์ ยังจริง

**`restart` บอกว่าหยุดแล้ว ทั้งที่ตัวเก่ายังถือโมเดลทั้งก้อนอยู่**

เจอบน spark-02 ตอน restart โมเดล Q8 39 GB · `lmds ps` เขียว `/health` 200 ทุกอย่าง
ดูปกติ แต่ `ps` มี **llama-server สองตัว** และ RAM เหลือ 25 GB จาก 121 GB

```
3239579 llama-server ... --port 8080   (ตัวเก่า Sl RSS 2.6 GB)
3244209 llama-server ... --port 8080   (ตัวใหม่ ถือพอร์ตอยู่)
```

`stop()` ใช้:

```bash
if [[ -f "$PID_FILE" ]] && kill "$(cat "$PID_FILE")" 2>/dev/null; then
```

`kill` คืนค่าทันทีที่ **ส่งสัญญาณ** สำเร็จ ไม่ได้แปลว่า process จบ · controller ลบ
PID_FILE แล้วพิมพ์ `stopped` ต่อทันที จากนั้น `restart` ก็ start ตัวใหม่ทับ ·
llama-server ตั้ง `SO_REUSEADDR` ตัวใหม่จึง bind ได้ตามปกติ **ไม่มี error ให้เห็น**
และ `lmds ps` ก็เขียวเพราะอ่าน PID file ของตัวใหม่ · restart อีกครั้งเดียวคือ OOM

เป็นอาการเดียวกับที่คอมเมนต์เหนือฟังก์ชันนั้นบอกว่าเคยแก้ไปแล้วรอบหนึ่ง
("บอกว่าหยุดแล้วทั้งที่ยังรันอยู่") แค่ลึกลงไปอีกชั้น

- `stop()` รอจน process จบจริง (`STOP_TIMEOUT` เริ่มต้น 30s) แล้วค่อยรายงาน
- ไม่จบใน timeout → SIGKILL แล้วรอต่อ · ยังไม่จบอีก → `die` ไม่ยอมให้ start ทับ
- zombie ไม่นับว่ายังอยู่ — `kill -0` สำเร็จกับ zombie ด้วย แต่ตัวนั้นคืนหน่วยความจำ
  ไปหมดแล้ว (เทสจับจุดนี้ได้ ตอนแรกเขียนพลาดเป็น die ใส่ zombie)
- เทสรัน controller จริงกับ process ที่ `trap '' TERM` เพื่อยืนยันว่ารอจริงและฆ่าจริง

**vision ของ Qwen-VL แม่นไม่เต็มที่ เพราะไม่เคยส่ง `--image-min-tokens`**

เจอตอน deploy `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` บน spark-02 · llama-server
เตือนตั้งแต่ตอนโหลดแล้วเดินต่อเงียบ ๆ:

```
load_hparams: Qwen-VL models require at minimum 1024 image tokens
              to function correctly on grounding tasks
              if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

controller ไม่เคยส่งค่านี้ จึงใช้ค่าที่ฝังมากับไฟล์ซึ่งต่ำกว่า · อาการคือถาม "ในภาพมีอะไร"
ตอบถูก แต่ถาม "อยู่ตรงไหน / กล่องไหน" เริ่มเพี้ยน — **ความแม่นยำที่หายไปโดยไม่มี error**
เป็นอาการเดียวกับตอนลืม `--mmproj` แต่จับยากกว่ามาก เพราะโมเดลยัง "เห็น" ภาพอยู่

- controller ที่มี projector ตั้ง `IMAGE_MIN_TOKENS=1024` เป็นค่าเริ่มต้น
- ปรับได้ด้วย `--image-min-tokens N` หรือ env · ตั้งว่าง = ใช้ค่าที่ฝังมากับโมเดล
- โมเดลข้อความล้วนไม่มีแฟล็กนี้โผล่มาให้งง (มีเทสคุมทั้งสองทาง)

**ทุกหน้าจอบอกได้แล้วว่าเครื่องนั้นอยู่ IP ไหน**

ทะเบียนเก็บแค่ **ที่อยู่ที่ใช้ SSH** ซึ่งตอบคำถาม "เครื่องนี้อยู่ IP ไหนในวง" ไม่ได้:

- ที่อยู่ SSH เป็น **ชื่อ** ได้ — `orb`, `spark1.local`, ชื่อบนเครือข่าย Tailscale ·
  เจอตรง ๆ ตอนเทสด้วย OrbStack บน Mac: การ์ดเขียนว่า `ops@orb:22` ซึ่งไม่มีตัวเลข
  ให้เอาไปยิง API หรือกรอกเป็น cluster IP ได้เลยสักตัว
- เครื่องหลัง NAT (คอนเทนเนอร์ · VM) มองจาก hub เป็นคนละที่อยู่กับที่เครื่องข้าง ๆ มัน
  ในวงเดียวกันใช้เรียกกันเอง
- เครื่องหนึ่งมีหลายเส้นพร้อมกัน — สายบริหารจัดการ, ConnectX ของคลัสเตอร์, Tailscale ·
  `primary_ip()` ตอบได้ทีละเส้น (เส้นที่ default route ออก) ซึ่งบ่อยครั้งไม่ใช่เส้นที่คนอื่น
  ใช้เข้ามา

สิ่งที่เพิ่ม:

- `lmds agent info` ส่ง **ทุกเส้น** มาด้วย (`ips`: ip · prefix · ชื่อการ์ด · link-local ·
  เส้นที่ออกเน็ต) ไม่ใช่แค่ `ip` ตัวเดียวเหมือนเดิม · เส้นเสมือนของ docker/k8s ถูกตัดทิ้ง
  เพราะยิงจากเครื่องอื่นไม่ถึง — โชว์ปนไว้คือชวนให้ก๊อปผิดเส้น
- คอนโซล: ป้าย IP บน **บรรทัดชื่อ** ของทุกการ์ด (บรรทัดเดียวที่เหลือตอนย่อ — "เครื่องนี้
  คือ IP ไหน" เป็นคำถามตอนกวาดตาดูทั้งฟลีต ไม่ใช่ตอนกางดูทีละเครื่อง) และการ์ด
  **System** แจกแจงทุกเส้นพร้อมชื่อการ์ด
- CLI: `lmds node list` และ `lmds ps --all` มีคอลัมน์ **IP ของเครื่อง** ·
  `lmds info` แจกแจงทุกเส้นแทนที่จะตอบเส้นเดียว · `lmds ps` เพิ่มบรรทัด IP เฉพาะเมื่อ
  เครื่องมีมากกว่าหนึ่งเส้น
- ทะเบียนจำ `local_ip` ล่าสุดไว้ — **เครื่องที่ต่อไม่ได้ยังบอกได้ว่าครั้งสุดท้ายเห็นมันอยู่ IP
  ไหน** ซึ่งเป็นตอนที่ต้องใช้พอดี · ค่าที่ node ไม่ได้ส่งมาแปลว่า "ไม่รู้" ไม่ใช่ "ไม่มี" จึงไม่
  เขียนทับของที่เคยรู้จริงด้วยค่าว่าง
- อ่าน IP ผ่าน `ifconfig` ได้เมื่อไม่มี iproute2 — เครื่องที่ไม่มีคำสั่ง `ip` (macOS ที่ใช้
  ทดสอบ, image ที่ตัดมาแล้ว) เดิมรายงานว่า "ตรวจไม่ได้" ทั้งที่มี IP อยู่ครบ ·
  รับทั้งสำเนียงใหม่ (`netmask 0xffffff00`) และของ net-tools รุ่นเก่า (`inet addr:` / `Mask:`)
- รวมโค้ดที่แปลง `agent info` → ฟิลด์ในทะเบียนไว้ที่ `status_from_probe()` จุดเดียว ·
  ของเดิมกระจายอยู่หกจุด (CLI สามที่ · หน้าเว็บสองที่ · refresher อีกหนึ่ง) เพิ่มฟิลด์ใหม่
  ทีหนึ่งต้องไล่แก้ให้ครบ และจุดที่ลืมจะ **เงียบ ไม่ใช่พัง** — เครื่องเดียวกันจึงแสดงค่าบ้าง
  ไม่แสดงบ้างแล้วแต่ว่ารอบล่าสุดใครเป็นคนอัปเดต
**รายการ GGUF ในโหมด non-interactive ตัดเหลือ 8 ตัวเล็กสุด — คนหา Q8 ไม่เจอ**

เคสจริงบน spark-02: `lmds deploy unsloth/Qwen3.6-35B-A3B-MTP-GGUF --no-llm -y`
บอกว่า repo มี **22 variant** แล้วพิมพ์รายการออกมา **8 บรรทัด** จบที่ `UD-Q3_K_XL`
โดยไม่บอกว่ายังมีต่อ · รายการเรียงจากเล็กไปใหญ่ `variants[:8]` จึงตัด Q4 ขึ้นไปทิ้ง
ทั้งหมด อ่านแล้วเข้าใจว่า repo นี้ไม่มี Q8 ขาย ทั้งที่มี

ตัวอย่างลิงก์ที่แนบมาให้ก็ใช้ `variants[0]` = ตัวเล็กสุดของ repo (IQ1_M) ซึ่งเป็น
ควอนต์คุณภาพต่ำสุดที่มี — คนที่ copy ไปใช้ตรง ๆ จะได้ของที่แย่ที่สุดโดยไม่รู้ตัว

- แสดง variant ครบทุกตัว ไม่ตัด
- ตัวอย่างลิงก์ยกตัวขนาดกลางแทนตัวเล็กสุด

**คำเตือน MTP ยกแฟล็กของ vLLM มาให้ดู แม้ plan จะเป็น GGUF/llama.cpp**

`lmds deploy` ของไฟล์ GGUF สร้าง plan ที่รันด้วย llama.cpp แต่โน้ตบอกให้ใช้
`--speculative-config` ซึ่งเป็นแฟล็กของ vLLM · llama.cpp ใช้ `--spec-type draft-mtp`
คนละชุดกันสิ้นเชิง อ่านแล้วเอาไปใส่ไม่ได้

- โน้ต MTP บอกแฟล็กของทั้งสองรันไทม์ และย้ำว่าห้ามลอกข้ามกัน
- Qwen3.6 เข้าเงื่อนไข DeltaNet ได้จาก**ชื่อ** ไม่ต้องรอ `hybrid_attention` จากการ
  อ่านไฟล์อย่างเดียว — repo ที่ inspect ไม่ครบจะได้คำเตือนเหมือนกัน
- โน้ต tool calling บอกเพิ่มว่า `qwen3_xml` กับ `qwen3_coder` **map ไป
  `Qwen3EngineToolParser` ตัวเดียวกัน** ใน vLLM รุ่นใหม่ (ตรวจจาก
  `vllm/tool_parsers/__init__.py`) จะได้ไม่มีใครไล่แก้ของที่ไม่ได้พัง — และเติม
  `--reasoning-parser qwen3` ซึ่งขาดไปจริง ๆ ถ้าไม่ใส่ ส่วน thinking จะไม่ถูกแยก
  ออกจากคำตอบ

> ระหว่างตรวจเรื่องนี้เคยสรุปผิดสองครั้ง แล้วยืนยันกับ source จริงจึงถอยกลับ:
> model card ของ Qwen เขียน `qwen3_next_mtp` แต่ `vllm/config/speculative.py`
> deprecate ชื่อนั้นและแปลงกลับเป็น `mtp` เอง — ของเดิมที่ LMDS แนะไว้ถูกอยู่แล้ว ·
> และ `qwen3_coder` ไม่ได้ "ถูกกว่า" `qwen3_xml` เพราะเป็น alias กัน


**Gemma 4 ได้คำแนะนำ parser ของตัวเอง — เดิมเงียบจนกลายเป็นโมเดลที่ "ใช้ tool ไม่เป็น"**

เคสจริงบน msi-2: `google/gemma-4-31B-it` ถูก deploy ด้วย `--tool-call-parser hermes`
มาเป็นสัปดาห์ vLLM ขึ้นปกติ `/health` เขียว ตอบ 200 ทุก request แต่คืน
`finish_reason: stop` กับ `tool_calls: null` แล้วยัด call ดิบไว้ใน `content`:

```
<|tool_call>call:hr_query{sql_query:<|"|>SELECT count(*) FROM employees;<|"|>}<tool_call|>
```

`hermes` อ่าน JSON ส่วน Gemma 4 ไม่ได้พ่น JSON — parser จึงแกะไม่ออกและปล่อยผ่าน
เป็นข้อความ **โดยไม่ error สักบรรทัด** กว่าจะรู้ก็ตอนเอาไปต่อ agent จริงแล้วเห็น
ข้อความประหลาดโผล่ไปถึงผู้ใช้ปลายทาง

ต้นเหตุฝั่ง LMDS คือ `arch_notes` มีคำแนะนำ parser ให้ **แค่ตระกูล Qwen** ตัวอื่น
ไม่มีอะไรเลย คนที่ deploy Gemma จึงหยิบ `hermes` ซึ่งเป็นค่าที่คุ้นมือที่สุด — และ
`gemma4` มีอยู่ในรายชื่อ parser ที่ LMDS ยอมรับมาตลอด แค่ไม่เคยถูกแนะนำ

- `arch_notes` เตือน Gemma 4 ตั้งแต่ตอนวางแผน โดยบอกทั้งตัวที่ถูก (`gemma4`)
  และตัวที่คนมักใส่ผิด (`hermes`) — คำเตือนที่บอกแต่ตัวที่ถูก คนอ่านแล้วไม่รู้ว่า
  ที่ตัวเองตั้งไว้อยู่คือตัวผิด
- คำเตือนบอกด้วยว่าอาการคือ **พังเงียบ** ไม่ใช่ error เพราะคำเตือนที่ไม่บอกตรงนี้
  จะถูกข้าม — deploy แล้วดูเหมือนสำเร็จทุกอย่าง
- `docs/USAGE.md` เพิ่มตารางเทียบ "syntax ที่โผล่ใน `content` → parser ที่ถูก"
  ครบ 5 ตระกูล วินิจฉัยได้จากคำตอบที่ได้จริงโดยไม่ต้องเดาจากชื่อโมเดล


**`nvcr.io/nvidia/vllm:latest` ไม่หลุดไปถึงเครื่องลูกค้าอีก**

ลูกค้าเทส deploy stacked แล้วเจอ:

```
docker: Error response from daemon: manifest for nvcr.io/nvidia/vllm:latest
        not found: manifest unknown
ERROR: download ล้มเหลวแม้ปิด Xet แล้ว — ดูข้อความด้านบน
```

สองปัญหาซ้อนกัน:

- **tag ผีผ่านทุกด่าน** — NGC ไม่เคยมี `:latest` สำหรับ repo นี้ (ใช้ tag ตามเดือน เช่น
  `26.05-py3`) · allowlist ตรวจแค่ **ชื่อ repo ไม่ตรวจ tag** ส่วนตัวตรวจ tag มีอยู่แล้ว
  แต่ `_ANON_TOKEN` มีแค่ Docker Hub กับ ghcr.io — nvcr.io จึงคืน `None` = "ตรวจไม่ได้"
  แล้วปล่อยผ่าน · เพิ่ม NGC เข้าไปแล้วผ่าน `/proxy_auth` (ไม่ใช่ `/token` ซึ่งตอบ 401) ·
  ยืนยันกับ registry จริง: `26.05-py3` → 200 · `latest` → **404**
- **error โกหก** — `download()` ไม่ได้ยืนยันว่ามี image ก่อน พอ `docker run` ล้ม มันสรุป
  เองว่าเป็นปัญหา Xet → ลองใหม่โดยปิด Xet → ล้มอีก → พิมพ์ "download ล้มเหลวแม้ปิด
  Xet แล้ว" ซึ่งพาคนไปดูเรื่องดาวน์โหลดทั้งที่ปัญหาคือ image ไม่มีอยู่จริง ·
  ตอนนี้ `ensure_image` ทำงานก่อนแตะน้ำหนักโมเดล ทั้ง single และ stacked

**HF token ไม่โผล่ใน `ps` อีก และดาวน์โหลดที่ค้างตายถูกปลุกเอง**

สองเคสจริงจาก spark-head ระหว่างโหลด NVFP4 170.9 GB:

- **token รั่วผ่าน argv** — `docker run … -e HF_TOKEN=hf_xxxx …` เอาค่าจริงไปแปะไว้ใน
  บรรทัดคำสั่ง user ทุกคนบนเครื่องอ่านได้ด้วย `ps` เฉย ๆ · LMDS มีหลักเรื่องนี้อยู่แล้วใน
  เส้นทาง SSH (ความลับเดินทางทาง stdin เท่านั้น) แต่เส้นทาง docker หลุดหลักนั้นไป ·
  แก้เป็น `-e HF_TOKEN` เฉย ๆ ให้ docker หยิบค่าจาก env ของเชลล์เอง — แก้ทั้ง
  vLLM (single + stacked) และ SGLang ทั้งตอน download และตอน start
- **ค้างตายเงียบ ๆ** — โหลดไปได้ 36 GB จาก 170.9 GB แล้วหยุดนิ่ง **1 ชั่วโมง 32 นาที**
  ไม่มี error ไม่มี log · connection ไปหา CDN ยังเปิดค้าง process นอนหลับใน `recv()`
  รอข้อมูลที่ไม่มีวันมา · ฝั่ง llama.cpp กันไว้แล้วด้วย `--speed-limit/--speed-time`
  แต่ `huggingface_hub` ในคอนเทนเนอร์ไม่มีตัวกันแบบนั้นเลย
- ตอนนี้เฝ้าว่า **ไบต์เพิ่มขึ้นจริงไหม** ไม่ใช่แค่ "process ยังไม่ตาย" · เงียบเกิน
  `DOWNLOAD_STALL_SECONDS` (600) = หยุดแล้วเริ่มใหม่ต่อจากของเดิม สูงสุด
  `DOWNLOAD_MAX_ATTEMPTS` (10) รอบ · ตั้ง `HF_HUB_DOWNLOAD_TIMEOUT=30` ให้ด้วย
- **หยุดตัวที่ค้างก่อนเริ่มใหม่เสมอ** — คนที่เห็นว่ามันค้างมักสั่งใหม่ทับ แล้วตัวใหม่ไปติด
  lock ของตัวเดิม กลายเป็นกองซ้อนกันสามตัวที่ไม่มีตัวไหนทำงานเลย (เจอจริงบน spark-head)
- ระหว่างแก้เจอกับดักของตัวเอง: `{{.State.Running}}` ของ docker ชนกับไวยากรณ์ jinja
  แต่โค้ดตรงนั้นอยู่ใน `{% raw %}` อยู่แล้ว — escape เพิ่มทำให้ไฟล์ที่เจนออกมาพัง
  มีเทสคุมไว้แล้ว

**สถานะคลัสเตอร์อยู่บนหน้าจอตลอด ไม่ต้องกดปุ่มทุกครั้ง**

- ผู้ใช้ขอ: "ให้เห็นสถานะ cluster ตลอดการใช้งาน ไม่ต้องค่อยกด check ตลอดถึงจะเห็น"
- ของเดิม `/api/cluster` ไปต่อทุกเครื่องสด ๆ ทุกครั้ง จึงช้าเกินกว่าจะเรียกถี่ ๆ ได้ —
  หน้าเว็บเลยเรียกเฉพาะตอนกดปุ่ม · แต่ refresher probe ทุกเครื่องรอบละ 15 วิเพื่อทำ
  การ์ดสถานะอยู่แล้ว **ข้อมูลที่ `cluster_groups` ต้องใช้อยู่ในแคชครบ** ไม่ต้องถามซ้ำ
- อ่านจากแคชเป็นค่าเริ่มต้น (ตอบทันที) · `?refresh=true` = ต่อเครื่องจริง สงวนไว้ให้ปุ่ม
  **Check cluster** อย่างเดียว
- `refreshNodes()` ดึงสถานะคลัสเตอร์ใหม่ทุกรอบ — รั้วกลุ่มและป้าย "stacked ได้ /
  ยังไม่พร้อม" เดินตามความจริงตลอด และขึ้นตั้งแต่เปิดหน้าโดยไม่ต้องกดอะไร
- อ่านจากแคชล้มหนึ่งรอบ = คงของเดิมบนจอไว้ ไม่ล้างทิ้ง · เรียกทุกรอบแล้วถ้าล้างทุกครั้ง
  ที่เน็ตสะดุด รั้วกลุ่มจะกะพริบหาย
- เครื่องที่ refresher ยังไม่เคยสำรวจสำเร็จรายงานว่า "ยังไม่มีข้อมูล — รอ refresher
  รอบถัดไป" แทนที่จะไปต่อสดแล้วทำให้ทั้งหน้าค้าง

## 0.5.0 — 2026-08-31

รอบนี้เกิดจากการใช้งานจริงในวันโชว์เคส: deploy แบบ stacked กดจากหน้าเว็บไม่ได้เลย ·
คู่ที่ตั้ง cluster IP ครบแล้วถูกระบบเขี่ยทิ้ง · ทำสำเนาโมเดลข้ามเครื่องไม่ได้ต้องโหลดใหม่
ทุกครั้ง · และดาวน์โหลดที่ถูกตัดกลางคันถูกนับว่าสำเร็จ

**หัวข้อใหญ่**

- `lmds node clone` — สำเนาโมเดลข้ามเครื่อง **412 MB/s** แทนโหลดใหม่ 40.7 MB/s
- deploy แบบ stacked กดได้จากหน้าเว็บ พร้อมเขียน `cluster.env` ให้เอง
- จับกลุ่ม stacked เฉพาะในไซต์เดียวกัน และ **แยกหลายคลัสเตอร์ในไซต์เดียวได้**
- ดาวน์โหลดวน resume เองจนขนาดตรง — ไม่เชื่อ exit code อีกต่อไป
- `verify-files` ตรวจ SHA-256 ได้จริงเป็นครั้งแรก (เดิม `EXPECTED_SHAS` ว่างเสมอ)
- fit รู้แล้วว่าเครื่องเป้าหมายไม่ว่าง — ไม่วางแผนทับหน่วยความจำที่โมเดลอื่นถืออยู่
- กราฟของเครื่องนี้ไม่ถูกถ่วงด้วยการ SSH ไปหาเครื่องอื่น


**deploy แบบ stacked กดได้จากหน้าเว็บแล้ว — ไม่ใช่คำสั่งให้ไปก็อป**

- ผู้ใช้รายงานวันโชว์เคส: "Deploy Stack ไม่เจอ" · ของเดิมหน้า Cluster ตรวจเจอคู่ที่
  stacked ได้ รู้ด้วยว่าใครควรเป็น head แล้ว **พิมพ์ออกมาเป็นข้อความ** ให้ไปรัน
  `lmds deploy --target dgx-spark-stacked` กับ `lmds node cluster --write` เองที่
  เทอร์มินัล — เส้นทาง deploy แบบ stacked จึงไม่มีอยู่จริงในหน้าเว็บ ทั้งที่ข้อมูล
  ที่ต้องใช้อยู่ในมือ server ครบแล้ว
- กลุ่มที่ "stacked ได้" มีปุ่ม **Deploy ลงกลุ่มนี้** แล้ว · กดแล้วเปิด wizard พร้อมตั้ง
  target เป็น `dgx-spark-stacked` และเลือก head/worker ให้ตามสมาชิกของกลุ่ม
- **เลือกเครื่องแล้วไม่ทับ target ที่ปักไว้เองอีก** — เดิมช่อง Run on เซ็ต target ตาม
  ฮาร์ดแวร์ของเครื่องนั้นเสมอ พอกด Deploy จากหน้า Cluster (ได้ stacked มา) แล้วชื่อ
  เครื่องถูกเซ็ตตาม → onchange ยิง → กลายเป็น `dgx-spark-single` เงียบ ๆ
- เพิ่มช่อง **Worker** ในฟอร์ม โผล่เฉพาะตอน target เป็น multi-node
- **เขียน `cluster.env` ให้เองหลัง push** (MASTER_IP/WORKER_IP/SSH_USER/NCCL iface) ·
  ขั้นนี้เคยหายไปทั้งขั้นสำหรับคนที่ใช้หน้าเว็บล้วน ๆ — bundle ที่ไม่มีไฟล์นี้จะไปค้าง
  ที่ NCCL init โดยไม่มีอะไรบอกสาเหตุ
- endpoint ใหม่ `POST /api/cluster/write` · ตรรกะย้ายจาก `cli/main.py` ไปที่
  `lmds/fleet/cluster_env.py` เพราะของเดิมเรียก `typer.Exit`/`err_console` ตรง ๆ
  ซึ่งเรียกจาก FastAPI ไม่ได้ · CLI เหลือแค่แปลง error เป็นข้อความบนเทอร์มินัล

**`lmds node clone` — ทำสำเนาโมเดลจากเครื่องที่รันผ่านแล้ว ไม่ต้องโหลดจาก HF ใหม่**

```bash
lmds node clone <slug> --from msi-1 --to msi-2 [--start]
```

- ของเดิมทางเดียวที่จะมีโมเดลบนเครื่องที่สองคือ `download` ซึ่งดึงจาก Hugging Face
  ใหม่ทั้งก้อน · IQ4_XS ของ Qwen3.8-Flash-Next = 90.8 GB ที่ 40 MB/s คือ 38 นาที
  ทั้งที่เครื่องข้าง ๆ ในแร็คเดียวกันถือไฟล์ชุดเดียวกันอยู่แล้ว · ยิ่งอยากทำ failover
  หรือกระจายโหลดหลายเครื่อง ยิ่งเสียเวลาเป็นทวีคูณ
- ไฟล์วิ่ง **ตรงจากต้นทางไปปลายทาง ไม่ผ่าน hub** — hub มักเป็นเครื่องเล็ก (VM บน
  โน้ตบุ๊ก) ที่จะกลายเป็นคอขวดทันทีถ้าให้ 90 GB ไหลผ่าน
- เลือก **สายเร็วที่สุดที่ทั้งคู่มี** เอง: ใช้ `cluster_ip` (ConnectX 200G ที่ตั้งไว้ให้
  stacked อยู่แล้ว) ถ้ามีทั้งสองฝั่ง ไม่งั้นถอยไปเส้นปกติ
- คัดลอกทั้ง weight และ bundle แล้ว **ตรวจ SHA-256 ที่ปลายทาง** ให้อัตโนมัติ
  (`--no-verify` ถ้าไม่ต้องการ) · `--start` สั่งเปิดต่อได้เลย · `--dry-run` ดูก่อนได้
- ข้ามไซต์ทำได้แต่เตือน — คนสั่งควรรู้ตัวว่ากำลังลาก 90 GB ข้ามเน็ตนอก

**หนึ่งไซต์มีได้หลายคลัสเตอร์ — ตั้งชื่อเอง**

- ผู้ใช้ถาม: "ถ้าใน site นั้น ผมจะทำ cluster มากกว่า 1 cluster ต้องทำอย่างไร"
- ระบบแบ่งเองได้เฉพาะตอนที่แต่ละคู่อยู่**คนละวง** · เครื่องรุ่นเดียวกันสี่เครื่องบนวงเดียวกัน
  จะถูกมองเป็นก้อนเดียว TP=4 ซึ่งบางทีไม่ใช่สิ่งที่ต้องการ (อยากได้สองคู่แยกกันเพื่อรัน
  คนละโมเดล หรือให้คู่หนึ่งเป็นตัวสำรอง)
- ฟิลด์ใหม่ `cluster_name` · `lmds node set <ชื่อ> --cluster-name <ชื่อคลัสเตอร์>` ·
  ว่าง = พฤติกรรมเดิมทุกประการ (ระบบแบ่งเองตาม subnet)
- **หน้าเว็บ:** ช่อง "คลัสเตอร์" ต่อจากช่อง cluster IP ในแถวของแต่ละเครื่อง พร้อมปุ่ม Save ·
  หัวกลุ่มขึ้นป้ายไซต์และชื่อคลัสเตอร์ หรือ "แบ่งอัตโนมัติ" ถ้าไม่ได้ตั้ง
- ตั้งชื่อรวมกันแต่ไม่มีวงร่วมกันเลย = ขึ้น blocker `no-shared-fabric` · ชื่อบังคับการจัดกลุ่ม
  แต่ไม่ได้เสกสายให้ ต้องบอกตั้งแต่ตอนนี้ ไม่ใช่ปล่อยไปค้างที่ NCCL init ตอน start
- `site` ไม่เคยถูกส่งเข้าฟังก์ชันจัดกลุ่มเลย (`row()` กับ payload ของ `/api/cluster`
  ไม่มีฟิลด์นี้) — การแยกไซต์รอบก่อนจึงยังไม่มีผลจริงจนกระทั่งรอบนี้

**ตั้ง cluster IP แล้วกด Save เห็นกลุ่มที่พร้อม deploy จริง**

- ผู้ใช้รายงาน 2026-08-31: "พอกดตั้ง ip ที่เครื่อง A และ B พอกด save แล้วไม่มีคำสั่ง
  deploy cluster"
- ที่เกิดขึ้นจริงบนฟลีต: `spark-head` + `spark-worker` ตั้ง cluster IP ครบแล้ว
  (10.100.152.1/.2 · 200G ทั้งคู่ · ขึ้นว่า "stacked ได้" ทั้งคู่) แต่กลุ่มที่ระบบเสนอ
  กลับเป็น `msi-1` + `msi-2` ที่ **ยังไม่ได้ตั้ง IP เลย** ส่วนคู่ที่ตั้งแล้วถูกเขี่ยออกไปเป็น
  "ฮาร์ดแวร์ตรงกัน แต่ไม่มี subnet ร่วมกับกลุ่มนี้"
- ต้นเหตุ: ทั้งสี่เครื่องเป็น GB10 เหมือนกันจึงอยู่ถังเดียวกัน แล้ว `connected_subset()`
  คืน **กลุ่มเดียวที่ใหญ่ที่สุด** โยนที่เหลือทิ้งทั้งหมด · สองคู่มีสมาชิกเท่ากัน ตัวตัดสิน
  จึงไปตกที่ *เลขวง* ซึ่งไม่เกี่ยวอะไรเลยกับว่าใครตั้ง IP ไว้แล้ว
- `connected_subsets()` คืน **ทุก** กลุ่มที่มีขาอยู่ในวงเดียวกัน — ฟลีตที่มีเครื่องรุ่นเดียวกัน
  หลายคู่บนคนละวงจึงเห็นครบทุกคู่ ไม่ใช่ทีละคู่

**stacked จับคู่ข้ามไซต์ไม่ได้อีกแล้ว**

- `site` เคยเป็นแค่ป้ายแสดงผล ไม่ถูกใช้ตอนจับกลุ่มเลย · ตอนนี้เป็นส่วนหนึ่งของถัง
- stacked ข้ามไซต์ทำไม่ได้จริงอยู่แล้ว (NCCL ต้องวิ่งบนสายในแร็ค ไม่ใช่ผ่าน WAN/VPN)
  และการเอามารวมถังเดียวกันทำให้คู่ที่อยู่คนละที่มาแย่งกันเป็น "กลุ่มที่ถูกเลือก"
  ทั้งที่ไม่มีวันได้ทำงานร่วมกัน — ซึ่งเป็นกลไกที่ทำให้บั๊กข้างบนเกิด
- เลขวงซ้ำกันระหว่างไซต์ (10.55.0.x ซ้ำกันได้สบายในเน็ตส่วนตัว) ไม่ทำให้ถูกจับคู่อีก
- แต่ละกลุ่มบอก `site` ของตัวเองมาด้วย

**clone กดได้จากหน้าเว็บด้วย ไม่ใช่มีแต่ CLI**

- ผู้ใช้รายงานทันทีหลังรอบแรก: "หน้า gui ยังหาเมนูไม่เจอ" — เพราะรอบแรกทำแต่ CLI ·
  คนส่วนใหญ่ในทีมทำงานผ่านหน้าเว็บ ฟีเจอร์ที่มีแต่ใน CLI จึงเท่ากับไม่มี
- ใน `⋯` ของแต่ละโมเดลบนการ์ดเครื่อง มีหมวด **ทำสำเนาไปเครื่องอื่น** พร้อมช่องเลือก
  ปลายทางและปุ่ม Clone
- รายชื่อปลายทางเรียงให้แล้ว: **ไซต์เดียวกันและสายเร็วกว่าขึ้นก่อน** และบอกในตัวเลือก
  เลยว่าจะวิ่ง "สายคลัสเตอร์" หรือ "เส้นปกติ" — ต่างกันสิบเท่าและคนสั่งควรเห็นก่อนกด
- ทำเป็น job เบื้องหลังเหมือน download/start ความคืบหน้าไหลมากับการ์ดของเครื่องต้นทาง
- `jobs.start_remote()` รับ `stdin_text` และ `on_done` แล้ว — `on_done` ทำให้การถอน
  กุญแจชั่วคราวเกิดขึ้น **เสมอ** แม้งานจะล้มกลางคัน

**กุญแจ: hub ไม่เคยส่ง key ของตัวเองให้ node**

- node แต่ละเครื่องไม่มี key ของกันและกันโดยตั้งใจ (node หนึ่งถูกยึดไม่ควรแปลว่า
  ทั้งฟลีตถูกยึด) แต่การ copy ต้องวิ่งตรงระหว่างสองเครื่อง
- สร้าง **กุญแจชั่วคราวสำหรับงานนี้ครั้งเดียว**: hub ฝาก public key ที่ปลายทางพร้อม
  marker ที่สุ่มต่อครั้ง → ต้นทางรับ private key ทาง stdin เข้า `ssh-agent`
  **ในหน่วยความจำ ไม่แตะดิสก์** → rsync → hub ถอนกุญแจออก **เสมอ** แม้ copy จะล้ม
- กุญแจชั่วคราวติด `restrict` (ปิด port-forward/agent-forward/pty) · ถอนด้วย
  `grep -v -F <marker>` ไม่ใช่ pattern กว้าง ๆ เพราะไฟล์นั้นมีกุญแจของ hub และของ
  ผู้ใช้อยู่ด้วย — ลบพลาดคือล็อกตัวเองออกจากเครื่อง
- `stream()` รับ `stdin_text` ได้แล้ว (ของเดิมส่งได้แต่ค่าบรรทัดเดียวผ่าน `read -r`
  ซึ่งใช้กับ private key หลายบรรทัดไม่ได้)

**`lmds node ctl` ให้ node ยืม HF token ได้แล้ว และเห็นความคืบหน้าระหว่างทำงาน**

- เคสจริง 2026-08-31: `lmds node ctl spark-head <slug> download` ตกทันทีด้วย
  "เป็น gated repo — ต้องมี HF_TOKEN ก่อน download" ทั้งที่ hub ถือ token ที่ใช้ได้อยู่ ·
  กลไกให้ยืมมีมาตั้งแต่รอบก่อนแล้ว (`lmds.web.jobs`) แต่ต่อไว้เฉพาะเส้นทางหน้าเว็บ —
  คนที่ทำงานผ่าน CLI จึง deploy โมเดล gated ข้ามเครื่องไม่ได้เลย
- ความลับเดินทางทาง **stdin ของ ssh** เท่านั้น ไม่ผ่าน argv ที่ `ps` ของทุก user บน hub
  มองเห็น และไม่เขียนลงไฟล์บนเครื่องปลายทาง (กลไกเดิมของ `stream()`)
- token ถูกกรองออกจากผลที่พิมพ์ออกหน้าจอด้วย — ปลายทาง "ไม่ควร" พิมพ์ token อยู่แล้ว
  แต่ไม่ควรกับไม่เคยเป็นคนละเรื่อง
- เปลี่ยนจาก `run()` เป็น `stream()` — download 90 GB ใช้เวลาเป็นสิบนาที ของเดิมเงียบ
  สนิทจนจบแล้วค่อยพ่นผลออกมาทีเดียว ระหว่างนั้นแยกไม่ออกว่าทำงานอยู่หรือค้างไปแล้ว

**เทสที่รั่วสถานะข้ามไฟล์**

- `create_app()` เรียก `serving._detect` ซึ่งเป็น `lru_cache(maxsize=1)` ที่อ่าน
  `LMDS_ROLE` ตอนถูกเรียก *ครั้งแรกของ process* — คำตอบ "เครื่องนี้เป็น hub" จึงถูก
  ตรึงไว้ทั้งรัน แล้วเทสที่ `setenv` ทีหลังก็ไม่มีผล · เทสไฟล์ใหม่ที่ชื่อขึ้นต้นด้วย d
  ทำให้ 11 เทสใน `test_fleet`/`test_web` ล้มทันที (ของเดิมรอดเพราะ `test_web`
  อยู่ท้ายสุดพอดี) · ตอนนี้เทสที่สร้าง app คืนสถานะให้ครบก่อนจบ

**ดาวน์โหลดที่หลุดกลางคันไม่ถูกนับว่า "เสร็จ" อีก**

- เคสจริง 2026-08-28 บน msi-5 · ไฟล์ Q5_K_M 20.3GB หลุดที่ 3,967MB ด้วย
  `curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)` — CDN ของ HF
  ตัดสตรีมกลางคันเมื่อโหลดยาว ๆ · `--retry` ของ curl **ไม่ยิงซ้ำให้** เพราะ curl นับ
  transient error แค่ timeout / 408 / 429 / 5xx เท่านั้น error 92 ไม่อยู่ในชุดนั้น
  curl จึงจบทันทีโดยเหลือไฟล์ที่ถูกตัดครึ่งไว้เฉย ๆ
- `fetch_one` วนต่อเองจนขนาดตรงแล้ว (resume ต่อจากของเดิมด้วย `-C -` ไม่เริ่มใหม่)
  · เพิ่ม `--retry-all-errors` โดยถาม `curl --help` ก่อนใช้ เพราะ curl < 7.71
  (Ubuntu 20.04) ไม่รู้จัก flag นี้แล้วจะตายทันทีแทนที่จะโหลดได้
- **ขนาดไฟล์คือเงื่อนไขจบ ไม่ใช่ exit code** — proxy ที่ส่ง body สั้นแต่ปิดสตรีม
  เรียบร้อยได้ exit 0 พร้อมไฟล์ไม่ครบ · ตอนนี้เทียบขนาดหลังดึงทุกครั้ง
- resume แล้วไม่ได้ไบต์เพิ่มเลย = ตายพร้อมบอกเหตุ ไม่วนไม่รู้จบ · ของเดิมไม่ถูกลบทิ้ง
  รอบหน้ายัง resume ต่อได้
- aria2c ที่ล้มยังถอยไป curl ต่อในรอบเดียวกันเหมือนเดิม (ก่อนหน้านี้รอบแก้แรกทำหาย)
- `stat` ถามทั้งแบบ GNU และ BSD แล้ว (`file_size`) — เดิม `stat -c` อย่างเดียว

**กราฟของเครื่องนี้ไม่ถูกถ่วงด้วยการ SSH ไปหาเครื่องอื่นอีก**

- ผู้ใช้รายงาน: "การแสดงผลจะช้ากว่าเครื่องหลายวิเลย กว่าจะเห็นกราฟขึ้น"
- ต้นเหตุไม่ใช่ `NODE_INTERVAL=15` แต่เป็นการที่ refresher ไล่ probe **ทีละเครื่อง
  แบบเรียงคิว** และ `_refresh_local()` อยู่หัวลูปเดียวกัน · บน Tailscale relay การ
  probe หนึ่งเครื่องกินเวลาระดับวินาที 14 เครื่องจึงเป็นรอบละหลายสิบวินาที — กว่าลูป
  จะวนกลับมาถึงบรรทัดแรก กราฟของเครื่องนี้ก็ตามหลังของจริงไปแล้ว และ
  `LOCAL_INTERVAL=3` ไม่เคยเป็นจริงเลย
- probe ลง thread pool 8 ตัวแล้ว · วัดด้วยเทสที่จำลอง node ช้า 8 เครื่อง: ของเดิม
  เครื่องนี้รีเฟรชได้ **1 ครั้งใน 3.5 วินาที** ตอนนี้ได้ตามจังหวะลูป (ทุก 1 วินาที)
- เครื่องที่ยัง probe ค้างอยู่จะไม่ถูกสั่งซ้ำ — `due()` เป็นจริงจนกว่าผลจะเขียนกลับ
  ถ้าไม่กัน เครื่องช้าเครื่องเดียวจะยึดคิวทั้งหมดไว้เอง

**verify-files ไม่เคยตรวจ SHA-256 เลยสักครั้ง**

- `_sibling_files()` อ่านคีย์ `oid` จาก `lfs` แต่ `/api/models/<id>?blobs=true` ซึ่งเป็น
  endpoint ที่ LMDS ใช้จริง ส่งคีย์ชื่อ **`sha256`** · ค่าจึงเป็น `None` เสมอ แล้ว
  `EXPECTED_SHAS` ในทุก controller ก็ว่างเปล่า
- ตัวเทมเพลตเขียนโค้ดตรวจ SHA-256 ไว้ครบตั้งแต่แรก มันแค่ไม่เคยมีค่าให้เทียบ —
  `verify-files` จึงลดเหลือ "ขนาดตรงไหม" อย่างเดียวมาตลอด
- ขนาดตรงแต่เนื้อในเสียเกิดได้จริง (สายหลุดกลางทางแล้ว resume ทับ, ดิสก์คืนบล็อกเสีย)
  และ GGUF ที่เสียบางไบต์จะ **โหลดขึ้นแต่ตอบเพี้ยน** ซึ่งไล่หาสาเหตุยากกว่าไฟล์ที่โหลดไม่ขึ้น
- รับทั้ง `sha256` และ `oid` เพราะสอง endpoint ของ Hub ใช้ชื่อคีย์ต่างกัน

**fit วางแผนเหมือนเครื่องเป้าหมายว่างเสมอ**

- `_budget_gb()` คิดจากความจุเต็มของ target · เคสจริงบน msi-5: deploy Gemma-4-31B ลง
  เครื่องที่มี Qwen3.8-27B (Q8_0, ctx 256K) รันอยู่ก่อน · fit ได้ budget **114.5 GB**
  แล้วตอบ `fits` กับ Q8_0 (32.6 GB) พร้อม context สูงสุด **262,144**
- ผลจริง: เครื่องขึ้นไป 107/121 GB และ **ทั้งสองโมเดลเหลือ 5-7 tok/s**
- ตัวเลขที่ต้องใช้มีอยู่แล้ว — `compute_apps()` ที่ `inventory` ใช้รายงาน foreign
  workload · คอมเมนต์ใน `foreign_workloads()` ก็เขียนเตือนไว้เองว่า *"เครื่องจึงดูว่าง
  ทั้งที่หน่วยความจำเกือบหมด แล้ว fit ก็วางแผน deploy ทับลงไปบนที่ที่ไม่มีจริง"* —
  แค่ไม่เคยถูกส่งเข้า `analyze()`
- `analyze()` รับ `reserved_gb` แล้ว · CLI ส่งค่าที่วัดได้ให้เฉพาะสเปกที่ **ตรวจจาก
  เครื่องนี้จริง** ไม่ใช่กับ preset ที่เป็นเครื่องสมมติ · ไม่ส่ง = พฤติกรรมเดิมเป๊ะ
- ผลหลังแก้บนเครื่องเดียวกัน: budget **37.4 GB** · verdict `fits-reduced-context` ·
  เลือก context 131,072 ให้เองโดยไม่ต้องสั่ง

**สูตรของโมเดล GGUF ไม่รู้ว่า quant ไหนที่รันผ่าน**

- `recipes/controllers.py` อ่าน header ได้เฉพาะบรรทัด `KEY="ค่า"` และทิ้งค่าที่มี `$`
  ทั้งหมด · controller ของ llama.cpp เก็บรายชื่อ shard ไว้ในอาร์เรย์
  `MODEL_FILES=( "…gguf" … )` แล้วตั้ง `MODEL_FILE="${MODEL_FILES[0]}"` ต่อ — ค่าที่อ่าน
  ได้จึงเป็นสตริง `${MODEL_FILES[0]}` ซึ่งถูกทิ้ง
- ผลคือสูตรของ GGUF **ทุกตัว** ไม่มี `gguf_file` เลย · ตรวจกับ script-update จริง:
  15/15 สูตรไม่มีฟิลด์นี้ ทั้งที่ชื่อไฟล์เขียนไว้ชัดเจนอยู่บรรทัดบนของ controller
- เรื่องนี้สำคัญเพราะรีโป GGUF หนึ่งตัวมี Q3/Q4/Q6/Q8 ปนกันสิบกว่าไฟล์ และตัวที่ทดสอบ
  มาแล้วมีตัวเดียว — สูตรที่ไม่บอกว่าไฟล์ไหน ก็ไม่ได้ช่วยให้เครื่องถัดไปรันผ่าน
- ตอนนี้อ่านชิ้นแรกของอาร์เรย์ระดับบนสุดได้ · อาร์เรย์ที่อยู่ในฟังก์ชันยังถูกข้ามตามกติกาเดิม
  และค่าที่ไม่ได้ลงท้าย `.gguf` ไม่ถูกรายงานเป็นไฟล์โมเดล

**กด Update ตอนที่ไม่มีอะไรใหม่ ไม่ขึ้นว่าล้มเหลวอีก**

- กด Update ตอน hub อยู่ที่ commit ล่าสุดอยู่แล้ว (`git pull` บอก `Already up to date`)
  แล้วหน้าเว็บขึ้น "เซิร์ฟเวอร์ยังไม่กลับมา — ลองรีเฟรชหน้านี้" ทั้งที่บริการกลับมาตั้งแต่
  วินาทีแรก · `waitForHub` รอจนกว่า **commit จะเปลี่ยน** ซึ่งไม่มีวันเกิดเมื่อไม่มีอะไรให้ดึง
  มันจึงรอครบ 120 วินาทีแล้วสรุปว่าล้ม
- ที่แย่กว่าข้อความผิด: มัน `return` ตรงนั้น — **หยุดก่อนไล่อัปเดต node ที่เหลือ** ซึ่งคือ
  เหตุผลทั้งหมดที่กดปุ่มนี้ · node ตามหลัง hub ได้แม้ hub ไม่มีอะไรให้อัปเดตแล้ว
- `/api/version` ส่ง `boot` = ลายเซ็นของ process (สุ่มใหม่ทุกครั้งที่บริการเริ่ม) หน้าเว็บ
  รอให้ค่านี้เปลี่ยนแทน — ตอบ "restart เสร็จหรือยัง" ได้ตรง ๆ โดยไม่ผูกกับว่ามีโค้ดใหม่ไหม
- เซิร์ฟเวอร์รุ่นเก่าที่ยังไม่ส่ง `boot` ถอยไปใช้เกณฑ์เดิม ไม่ค้างรอค่าที่ไม่มีวันมา
- "ดึงมาแล้วเป็นของเดิม" แสดงเป็น "อยู่ที่ &lt;commit&gt; อยู่แล้ว" แล้วไปต่อที่ node ตามปกติ
- ตรวจกับของจริง: restart แล้ว commit เท่าเดิม (`48e9606`) แต่ `boot` เปลี่ยน —
  เดิมรอ 120 วิแล้วล้ม ตอนนี้คืนผลใน 37 วิ พร้อมข้อความ "อยู่ที่ 48e9606 อยู่แล้ว"

**ผู้ช่วยลงไปดูเครื่องจริงได้ และเสนอวิธีแก้ให้กดอนุมัติ**

- เดิม system prompt บอกมันตรง ๆ ว่า "คุณไม่มีเครื่องมือให้เรียกใช้เลย" แล้วให้ตอบจาก
  สรุปสถานะที่แคชไว้ (ต่อติดไหม, ดิสก์เหลือเท่าไร, โมเดลไหน running) · คำถามที่คนถาม
  บ่อยที่สุดคือ "ทำไมตัวนี้ไม่ขึ้น" ซึ่งตอบจากข้อมูลชุดนั้นไม่ได้เลย เพราะคำตอบอยู่ใน log
- ตอนนี้ก่อนตอบทุกครั้ง ระบบจะเลือก probe จากแคตตาล็อก (`lmds/assistant/catalog.py`)
  แล้วรันบนเครื่องที่เกี่ยวข้องผ่านทางเดิมของ `lmds node` — log ของ controller, GPU,
  ดิสก์, RAM, พอร์ตที่เปิด, docker, เน็ตเวิร์ก/fabric, `lmds doctor` · หน้าเว็บขึ้นบรรทัด
  "ดูมาแล้ว: …" ให้เห็นว่าคำตอบนี้มาจากการไปดูจริง ไม่ใช่จากแคช
- **LLM ไม่เขียน Bash** เหมือนขั้นวางแผน deploy — มันเลือกได้แค่ชื่อรายการกับพารามิเตอร์
  ที่ผ่าน `Param.clean` (slug ที่มี `;` หรือ `$( )` ตายตั้งแต่ประตู) คำสั่งจริงประกอบด้วยโค้ด
- งานที่เปลี่ยนสภาพเครื่อง (restart, เปลี่ยน context/port/bind/gpu-util, ล้างแคช
  FlashInfer) ถูกเสนอเป็น**ตั๋ว** พร้อมคำสั่งเต็มและผลกระทบ แล้วถามกลับเป็นเมนู
  **แก้เลย / ทีละขั้น / ยังไม่ทำ** · ตั๋วออกโดยเซิร์ฟเวอร์และเดินได้ต่อเมื่อเบราว์เซอร์
  ส่งกลับมาพร้อมโหมด — โมเดลออกตั๋วให้ตัวเองไม่ได้ แม้โดน prompt injection จากข้อความ
  error ของเครื่องปลายทาง
- โหมด "ทีละขั้น" มีไว้เพราะการเปลี่ยนสามค่าพร้อมกันแล้วดีขึ้น แปลว่าไม่มีใครรู้ว่าอะไรได้ผล
  · ขั้นที่ล้มหยุดขั้นที่เหลือทันที ขั้นถัดไปตั้งอยู่บนสมมติฐานว่าขั้นก่อนหน้าสำเร็จ
- ผลจากเครื่องถูก redact ก่อนส่งออกไปหา provider — log ของจริงมี token/endpoint ปนมาได้

**ผู้ช่วยได้ "วิธีคิดแบบ LMDS" ติดตัว ไม่ใช่แค่สถานะ**

- `lmds/assistant/playbook.md` ติดไปกับแพ็กเกจ: หลักที่ยึด (หลักฐานชนะความจำ, knob
  มาก่อนแก้ไฟล์, ไม่รู้ให้บอกว่าไม่รู้), สายความคิด artifact → engine → งบหน่วยความจำ →
  bundle → node, ตารางว่าอาการแบบไหนควรเริ่ม probe ตัวไหน และวิธีเสนอให้แก้
- ตารางอาการ→สาเหตุ→วิธีแก้ **ไม่ถูกคัดลอกเข้า prompt** โดยตั้งใจ — ผู้ช่วยค้นจาก
  `docs/` ตัวจริงตอนใช้งาน (USAGE, RUNBOOK, PREFLIGHT, FIELD-NOTES, NETWORK …)
  สำเนาใน prompt จะล้าสมัยเงียบ ๆ วันที่ทีมแก้เอกสาร แล้วมันจะตอบสูตรเก่าอย่างมั่นใจ
- งบ system prompt ขยับ 13,500 → 26,000 ตัวอักษรเพื่อรับวิธีคิดกับผลตรวจ · สถานะยัง
  ได้ที่ *ที่เหลือ* เหมือนเดิม จึงยังเบียดคำถามของผู้ใช้ไม่ได้
- ใช้ `complete_json` ซึ่งทุก provider มีเหมือนกัน ไม่ใช่ function calling ของ OpenAI —
  ไม่งั้นฟีเจอร์นี้ใช้ได้เฉพาะบางเจ้า และใช้กับ vLLM/Ollama ในบ้านไม่ได้

**probe สำรวจไม่รายงานว่า "ล้ม" เพราะเครื่องมือย่อยตัวเดียวหาย**

- `disk` เคยขึ้นว่าล้ม (exit 1) เพราะ `du` เจอ `~/.cache/huggingface` ที่ยังไม่มี ทั้งที่ผล
  `df` ที่ต้องการอยู่ครบแล้ว — ผู้ช่วยเห็นธง "ล้ม" แล้วทิ้งข้อมูลที่ใช้ได้ · เครื่องที่ไม่มี
  nvidia-smi/docker ก็เจอแบบเดียวกัน


**เครื่องที่มี AI รันอยู่แต่ไม่ได้มาจาก LMDS ไม่ถูกนับเป็นศูนย์อีก**

- `lmds node list --check` แสดง "นอกระบบอีก N" ต่อท้ายจำนวนโมเดล · ข้อมูลเดินทางมาถึง
  hub อยู่แล้วใต้ `host.foreign` แค่ไม่มีใครอ่าน · เจอจริง: spark-03 เสิร์ฟ Nemotron-3
  ผ่าน trtllm-serve อยู่ แต่หน้าจอบอก "โมเดล 0 ตัว" ซึ่งชวนให้วางแผน deploy ทับ
- node รุ่นเก่าที่ไม่ส่งคีย์นี้มายังเงียบเหมือนเดิม — ไม่มีข้อมูล ไม่ใช่ไม่มีงาน

**`lmds adopt <container>` ไม่ล้มตอนพิมพ์สรุปอีก**

- สาขา container อ้าง `info.default_model` ที่คลาส `Adopted` ไม่เคยมี — adopt เขียน
  bundle เสร็จแล้วค่อย crash ผู้ใช้จึงไม่รู้ว่าสำเร็จหรือไม่ · ตัดออกเพราะ container
  ไม่ได้จดชื่อที่เสิร์ฟไว้ที่ไหนให้เทียบ (สาขา process มี จึงยังอยู่)
- ชื่อโมเดลอ่านจาก argv ได้แล้วเมื่อไม่มีใน env — `trtllm-serve <org/name> --port …`
  วางชื่อเป็น positional · เดิมสรุปว่า "(ไม่ระบุใน env)" ทั้งที่ชื่ออยู่ใน docker inspect
- แก้คำสะกด `sคริปต์` → `สคริปต์`
- port ของ container ที่ adopt มาอ่านจาก `--port` บน argv ก่อน · container ที่เปิด
  หลายรู (metrics/API/notebook) เคยถูกคว้ารูแรกมาใช้ แล้ว health check เคาะผิดที่
  สถานะจึงค้าง "loading" ตลอด ทั้งที่โมเดลเสิร์ฟอยู่ปกติ (spark-03: 6006 แทน 8355)

**หัว MTP ที่ไม่ใช่โมเดล ไม่ถูกส่งเข้า --spec-draft-model อีก**

- repo ที่แถม "หัว MTP ล้วน ๆ" (เช่น `mtp-RVN.gguf` — 65 บล็อกแต่มี 16 tensor
  ไม่มี `token_embd.weight`) เคยถูกต่อเป็น draft model แยก ผลคือ llama-server ล้มที่
  `check_tensor_dims` แล้วปิดตัวก่อน health check — **โมเดลไม่ขึ้นเลย** ไม่ใช่แค่ไม่มี
  speculative decoding · เจอจริงกับ 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
- แยกไม่ได้ด้วยชื่อไฟล์: `mtp-gemma-4-26B-A4B-it.gguf` ก็ชื่อขึ้นต้น `mtp-` เหมือนกัน
  แต่เป็นโมเดล draft ย่อจริง (4 บล็อก 49 tensor) ที่ต่อเป็น draft ได้ถูกต้อง
- `GgufInfo.is_standalone_model` ตัดสินจาก header: จำนวน tensor ต้องสมเหตุผลกับจำนวน
  บล็อกที่ metadata ประกาศ · inspect อ่าน header ของไฟล์ฝั่ง speculative ทุกตัวแล้ว
  บันทึกไว้ที่ `GgufVariant.is_standalone_draft`
- ไฟล์ที่อ่าน header ไม่สำเร็จ (None) ยังผ่านเหมือนเดิม — เน็ตสะดุดตอน inspect ไม่ควร
  ปิดฟีเจอร์เงียบ ๆ · ปฏิเสธเฉพาะตัวที่รู้แน่ว่าเป็นหัวล้วน
- ไฟล์ที่เลือกฝัง head มาแล้วชนะเสมอ: ไม่ส่ง `--spec-draft-model` ควบกับ `--spec-type`
- คำเตือนบอกชื่อไฟล์ที่ควรเลือกแทนตรง ๆ (`เลือก RVN-Q6_K-multilingual-mtp.gguf แทน`)
  ไม่ใช่แค่ "เลือกตัวที่ลงท้าย -mtp" ซึ่งเดาไม่ถูกเมื่อ repo มีเป็นร้อย variant


## 0.4.1

**คะแนนโมเดล: หน้ารายละเอียดสำหรับนำเสนอ + ลบผลได้ + พับหมวดได้**

- ปุ่ม **Details** เปิดหน้าที่แสดงทุกงานเป็นแท่งเทียบกัน (decode / prefill / TTFT)
  พร้อมแถบสรุปด้านบนและการ์ดต่อหนึ่งงาน — ลูกค้าขอมาเพื่อใช้นำเสนอ · ตารางสรุป
  บรรทัดเดียวบอกไม่ได้ว่าความเร็วตกตรงไหน
- `lmds agent bench --slug` ส่งผลเต็มของโมเดลเดียว · `GET /api/nodes/{name}/bench/{slug}`
  ให้ hub อ่านรายละเอียดของเครื่องอื่นได้ (ไฟล์อยู่ที่เครื่องที่วัด)
- ลบผลวัดได้: ปุ่ม **Delete** ต่อแถว · `lmds bench remove <slug> [--keep-last N]`
  · `DELETE /api/bench/{slug}` · `POST /api/nodes/{name}/bench/{slug}/remove`
  วัดซ้ำเป็นเรื่องปกติแล้วไม่มีใครกลับไปลบเอง — เก็บได้อย่างเดียวคือออกแบบให้พังตามเวลา
- พับหมวด **Model scores** / **Models on this machine** ได้ และจำสถานะไว้
- quantization บนการ์ดนำเสนออ่านจากชื่อไฟล์ GGUF เมื่อ profile ไม่ได้จดไว้ ·
  เครื่อง unified memory แสดงขนาดหน่วยความจำแทนช่อง VRAM ที่ว่าง

**ภาษา**

- ป้าย UI ในหมวดคะแนนกลับไปเป็นอังกฤษให้ตรงกับทั้งหน้า (Model scores, longest context,
  median of all workloads, instructions/Thai/vision/recall) · ไทยเหลือไว้ที่คำอธิบาย
  ตามกติกาเดิม · มีเทสต์กันหลุดกลับ

**คอนโซล**

- ผลคำสั่ง/ผลเทสของเครื่องย้ายไปอยู่กล่องของตัวเองใต้การ์ด แทนที่จะเขียนทับเนื้อการ์ด
  ทั้งอัน · ลบทิ้งได้ · สูงสุด 40vh แล้วเลื่อนในกล่อง · งานที่สำเร็จผลไม่ถูกล้าง
- หมวด Model scores ย้ายลงมาอยู่หลังรายชื่อเครื่อง ก่อน Weights already on disk


## 0.4.0

**ให้คะแนนโมเดลที่รันอยู่จริง — `lmds bench` + หมวด "คะแนนโมเดล" บนคอนโซล**

แรงบันดาลใจจาก [Local-Bench](https://github.com/companionintelligence/Local-Bench) แต่ทำ
คนละทาง: Local-Bench ผูกกับ Ollama และแปะคะแนน IQ จาก Artificial Analysis ซึ่งเป็นดัชนีของ
**โมเดลต้นทาง** ไม่ใช่ของ quant ที่รันอยู่จริงบนเครื่องจริง · ที่นี่ทุกตัวเลขมาจากการยิงเซิร์ฟเวอร์
ที่รันอยู่ผ่าน OpenAI API จึงเทียบข้าม engine ได้ และไม่มี "คะแนนความฉลาด" ที่เราไม่ได้วัดเอง

- ชุดงานตายตัว 6 แบบ ไล่ context 512 → 8K · วัด TTFT / decode tok/s / prefill tok/s แยกกัน
  · median ไม่ใช่ mean · ทำลาย prompt cache ทุกรอบและเตือนถ้ายังมี cache เหลือเกิน 10%
- ความสามารถ 7 ข้อวัดจากพฤติกรรมจริงที่ปลายทาง ไม่ใช่จากที่ `MODEL_PROFILE.yaml` จดไว้
  · tool calling ต้องออกในฟิลด์ `tool_calls` จริง ไม่รับการพิมพ์ JSON ลง content
  · ข้อที่ข้ามไม่ถูกนับในตัวหาร · คะแนนรวมมาพร้อมรายชื่อข้อที่ตกเสมอ
- โมเดลที่คิดนานจนงบ token หมดก่อนตอบ ได้ลองซ้ำด้วยงบสามเท่าก่อนถูกตัดสิน
- ผลเก็บเป็น JSON หนึ่งไฟล์ต่อรอบ พร้อมสเปกเครื่อง + engine build + quant + context
- `lmds bench run|list|show` · `GET /api/bench` · `POST /api/bench/{slug}/run`
- เอกสารเต็ม: [docs/BENCH.md](docs/BENCH.md)

**node ที่ rc พ่นขยะออก stdout**

`dgx-70` มี rc ที่พ่น `declare -x …` ทุกตัวแปรออก stdout ทุกครั้งที่ login shell เริ่ม · JSON ของ
`lmds agent info` อยู่ครบและถูกต้อง แต่เริ่มที่ไบต์ที่ 858 — hub บอกว่า "เวอร์ชัน LMDS อาจ
ไม่ตรงกัน" แล้วผู้ใช้ก็ไปไล่หาเวอร์ชันที่ไม่ได้ผิดอะไรเลย · banner, motd และ rc แปลก ๆ เป็น
เรื่องปกติบนเครื่องลูกค้า การยืนกรานว่า stdout ต้องเป็น JSON ล้วนคือข้อสมมติที่ภาคสนาม
ไม่เคยจริง


## 0.3.7

**เครื่องรู้ว่าตัวเองมีไว้ทำอะไร — control plane กับเครื่องรันโมเดล**

เคสจริง 2026-08-19: `lmds repair` บน hub VM (OrbStack, 192.168.139.92) เริ่มดูด weight
**15.6 GB** ลงเครื่องที่ไม่มี GPU ไม่มี docker ไม่มี llama.cpp และ RAM 12 GB — ไฟล์ที่ต่อให้
โหลดจบก็ไม่มีอะไรรันมันได้ ส่วนเครื่องเดียวกันคำสั่งนี้บน 10.2.3.100 (คุม RTX บนตัวเอง) ถูกต้อง
ทุกประการ · แนวคิด "เครื่องนี้ไม่ใช่เครื่องที่จะรันโมเดล" มีอยู่ใน docstring ของ `lmds node push`
มาตั้งแต่ต้น แต่ไม่เคยมีอยู่ในพฤติกรรมของโค้ดเลย

- โมดูลใหม่ `lmds.hardware.serving` ตรวจจาก**ของที่มีจริง** ไม่ใช่ชื่อเครื่องหรือ config:
  `llama-server` ที่รันได้ → engine `llamacpp` · docker **คู่กับ GPU** → `vllm`/`sglang`/`trtllm`
  · ไม่มีสักอย่าง = control plane
- `download` · `repair` · `start` · `restart` · `prepare-runtime` ถูกปฏิเสธบน control plane
  พร้อมบอกทางออก (`lmds node push <เครื่อง> <slug> --download`) — ครอบทั้ง CLI และปุ่มบนคอนโซล
  เพราะเช็คที่ `jobs.start()` ซึ่งเป็นทางเดียวที่หน้าเว็บสั่ง controller
- คำสั่งที่แค่ *อ่าน* ของที่มีอยู่ (`verify-files` · `status` · `doctor` · `logs` · `stop`)
  ปล่อยผ่านตามเดิม — คนที่นั่งอยู่บน hub ต้องใช้มันได้
- ทางออกเมื่อการตรวจเดาผิด: `--force` หรือ `LMDS_ROLE=serving` (และ `LMDS_ROLE=hub`
  สำหรับเครื่องที่มี GPU แต่ตั้งใจให้เป็น hub)

**คอนโซล**

- การ์ดเครื่องมีการ์ด **บทบาท** บอกตรง ๆ ว่า control plane หรือเครื่องรันโมเดล พร้อมหลักฐาน
  ที่ใช้สรุป — เดิมเห็นแค่ "GPU not found" แล้วก็ยังกด Download อยู่ดี
- บน control plane ปุ่ม **Download/Start** เปลี่ยนเป็น **ส่งไปเครื่องที่รันได้** ซึ่งเปิดแผงที่มี
  ตัวเลือกเครื่องปลายทางอยู่แล้ว · แผง Repair บอกเหตุผลแทนที่จะยื่นปุ่มที่กดแล้วโดนปฏิเสธ

**`lmds doctor`**

- ขึ้นข้อ **บทบาท** เป็นข้อแรก เพราะมันเปลี่ยนความหมายของทุกข้อที่ตามมา
- บน control plane ข้อที่แปลว่า "รันไม่ได้" (`docker` · `image` · `architecture` · `grammar`
  · `weights` · `server`) ไม่นับเป็นตัวบล็อกอีกต่อไป — เดิมสรุปว่า "พบ 1 ข้อที่ต้องแก้ก่อนถึงจะรันได้"
  แล้วชวนให้ไปติดตั้ง docker บนเครื่องที่ไม่มี GPU ให้มันใช้อยู่ดี
- คำแนะนำของ `weights`/`server` เปลี่ยนจาก `lmds repair` / `lmds start` (คำสั่งที่เครื่องนี้
  ปฏิเสธอยู่แล้ว) เป็นคำสั่ง push


## 0.3.6

**รับโมเดลที่รันอยู่ก่อน LMDS เข้าระบบได้ทุกแบบ — `lmds adopt`**

- เดิมรับได้เฉพาะ **Docker container** ส่วนเคสที่เจอบ่อยพอ ๆ กันคือ `llama-server` ที่รัน
  ตรง ๆ ใต้ systemd unit ที่ลูกค้าเขียนเอง · `lmds ps` เห็นมันอยู่แล้ว (`_orphan_native`
  อ่าน cmdline) แต่ตันตรงเดียวกันคือไม่มี controller
- `lmds adopt --port N` / `--pid N` อ่าน argv/exe/cwd จาก `/proc` แล้วเขียน controller
  ที่รันคำสั่งเดิมซ้ำได้เป๊ะ — ของที่รันอยู่ไม่ถูกแตะต้อง
- **จงใจไม่อ่าน `/proc/<pid>/environ`** — API key ของ backend อยู่ในนั้น การเขียนลง bundle
  คือทำให้ทุกคนที่อ่านไฟล์ได้เห็น secret
- บันทึก systemd unit เจ้าของไว้ เพราะ unit ที่ตั้ง `Restart=always` จะแย่ง port กลับทุกครั้ง
  ที่ LMDS stop · `start` ปฏิเสธพร้อมบอกคำสั่งที่ต้องใช้ · `status` เตือนว่าตัวที่ตอบอาจไม่ใช่
  ของ LMDS · `--take-over` สั่ง disable ให้เมื่อผู้ใช้ยืนยัน ไม่ทำเอง

**คอนโซล**

- การ์ดที่ยังไม่มี controller เคยยื่น **ปุ่ม Start ที่กดไม่ได้** ให้ดูเฉย ๆ ทั้งที่สิ่งที่ต้องทำจริง
  คือรับเข้าระบบก่อน — ซึ่งไม่มีทางรู้ถ้าไม่เคยอ่าน CLI · เปลี่ยนเป็นปุ่ม **รับเข้าระบบ**
  (`POST /api/models/{slug}/adopt`)
- ป้ายเปลี่ยนจาก `not deployed by lmds` (บอกว่าไม่ใช่ของเรา) เป็น **`ยังไม่ได้รับเข้าระบบ`**
  (บอกว่าต้องทำอะไร) · ถ้ามี unit เจ้าของ เตือนพร้อมคำสั่ง `disable` ให้ copy ไปใช้

**`lmds doctor` ตรวจ weight ของ bundle ที่รับเข้าระบบตรง path ที่ใช้จริง**

- doctor หาไฟล์ที่ `~/models/<slug>` ตามธรรมเนียมของ bundle ที่ LMDS deploy เอง ส่วน
  bundle ที่ adopt มาชี้ไป path เดิมของเจ้าของ → ขึ้น `✗ ยังไม่มีไฟล์โมเดล` ตลอดกาลทั้งที่
  เซิร์ฟเวอร์กำลังเสิร์ฟไฟล์นั้นอยู่ และ `server ● running` อยู่บรรทัดถัดไป
- แย่กว่านั้นคือคำแนะนำ `lmds repair` ซึ่ง controller ของ adopt **ไม่มีคำสั่งนั้นโดยตั้งใจ**
  — ทำตามแล้วล้มแน่นอน · `self_managed_weights()` รู้เรื่องนี้อยู่แล้วแต่ doctor ไม่ได้เรียก

## 0.3.5

**MoE กับ MTP เป็นข้อเท็จจริงจากไฟล์ ไม่ใช่สิ่งที่ LLM เดา**

- อ่านจำนวน expert ทั้งหมด/ที่เปิดต่อ token จาก `config.json` หรือ GGUF metadata แล้วแสดง
  ตั้งแต่ตอน `deploy` ยันคอนโซล (`image, MoE 128e/8a, MTP`) — *total บอกว่าต้องมีหน่วยความจำ
  เท่าไร active บอกว่าจะได้ความเร็วเท่าไร* บนเครื่องที่คอขวดคือ bandwidth สองค่านี้ต่างกันหลายเท่า
- repo ที่แถมไฟล์ MTP draft head ถูกโหลด + ต่อสาย `--spec-draft-model` กับ
  `--spec-type draft-mtp` ให้อัตโนมัติ · **วัดจริงบน DGX Spark: gemma4-26B-A4B ได้ 1.78x
  โดย output เท่าเดิม** (repo เคลม 1.35x — Spark คอขวดที่ bandwidth จึงคุ้มกว่า)
- repo ตระกูล *Native-MTP-Preserved* ที่คง MTP head ไว้ในไฟล์เป้าหมายเอง
  (`nextn_predict_layers`) ก็เปิดให้ด้วย `--spec-type draft-mtp` เฉย ๆ ไม่มี draft แยก
- กัน `mtp-*.gguf` หลุดไปเป็นตัวเลือก weight ให้ผู้ใช้เลือกรันเป็นตัวโมเดล

**เปลี่ยนชื่อโมเดลได้เหมือนย้าย port**

- `--name NAME` ตั้งชื่อที่ client ใส่ในฟิลด์ `model` · ชื่อที่ generate ตั้งไว้ถูกตรึงเป็น
  `DEFAULT_SERVED_MODEL_NAME` ที่ override ไม่ได้ แล้วโชว์คู่กันเมื่อไม่ตรง — เปลี่ยนแล้วยังรู้
  ว่าเดิมคืออะไร
- `--mmproj/--no-mmproj` และ `--mtp/--no-mtp` เปิด/ปิดฟีเจอร์โดยไม่ต้อง deploy ใหม่
- หน้า help แบ่งเป็น Identity / Network / Memory & limits / Model features

**คำสั่งรายงานสิ่งที่เกิดขึ้นจริง**

- `status`/`logs`/`network-info`/`stop` อ่าน `server.meta` — เดิม `start --port 8020`
  แล้ว `status` ตอบ `API: not responding` เพราะไปถาม port default
- `doctor` กับตัวสแกน bundle อ่าน `bundle.env` — เดิมฟ้อง port ชนที่ไม่ได้ชนจริง
  แล้วแนะ port ที่ไม่ว่างอีกตัว
- `lmds version` บอก commit ด้วย — เลขเวอร์ชันอย่างเดียวตอบไม่ได้ว่าฟลีตเท่ากันหรือยัง
- ตั้งค่าว่างเพื่อปิด `MMPROJ_FILE` / `MTP_FILE` ได้จริง (เดิมใช้ `:-` ซึ่งกลืนค่าว่าง)

**ดาวน์โหลดไม่ค้างตาย**

- `curl` ได้ `--speed-limit`/`--speed-time` — สลับเน็ตแล้ว TCP ค้างใน `recv()` ไม่มี error
  `--retry` จึงไม่เคยทำงาน process ค้างถาวรจน `stop` ไม่ได้

**`rebuild` อ่าน GGUF header ของไฟล์ที่ profile เลือกไว้**

- repo หลาย quant ไม่เลือกไฟล์ให้เองตอน inspect · เดิมตั้ง `selected_gguf` หลัง inspect
  จบไปแล้ว ทำให้ architecture / context / kv_dims / MoE หายหมด

**คอนโซล**

- ป้าย MoE/MTP พร้อม icon และโทนสีแยกจาก ok/warn/bad
- บรรทัด `model ID` พร้อม `↳ เดิม: …` เมื่อถูกตั้งชื่อใหม่
- หมวด Model features — ติ๊กเปิด/ปิด vision กับ MTP

รูปแบบตาม [Keep a Changelog](https://keepachangelog.com/) · เวอร์ชันตาม [SemVer](https://semver.org/)

## [Unreleased]

### Added

- **แยก fleet ตามไซต์** — `Node.site` + `lmds node set --site <ไซต์>` · `lmds node list` แยก
  ตารางต่อไซต์ (เครื่องที่ยังไม่จัดกลุ่มอยู่ท้ายสุด) · หน้าเว็บมี badge บอกไซต์บนการ์ด ·
  เครื่องออกไปหลายไซต์แต่ยังอยู่ในทะเบียนเดียว หน้าจอเลยยาวจนหาไม่เจอว่าเครื่องไหนของที่ไหน
  · เป็นแค่การจัดกลุ่มเพื่อแสดงผล **ไม่แตะ cluster เลย** (stack/cluster_ip/NCCL คงเดิม) ·
  site ว่าง = ทุกเครื่องอยู่กลุ่มเดียวเหมือนก่อนมีฟิลด์นี้

- **`lmds recipes --publish <slug>`** — ทางกลับของ `--sync` · เมื่อ deploy โมเดลจนได้ค่าที่
  รันผ่าน + ทดสอบแล้ว ส่ง controller ตัวนั้นขึ้นคลังเพื่อให้เครื่องอื่น `--sync` ไปใช้ได้เลย
  ไม่ต้องเดา parser/image/mmproj ใหม่ทุกครั้ง · ส่ง **เฉพาะค่าของโมเดล** — bundle.env
  (port/context/slots ของเครื่อง) เป็นคนละไฟล์ ไม่ตามขึ้นไป และฝั่ง parse ตัด context ทิ้ง
  อยู่แล้ว โมเดลจึงไป fit ใหม่ตามเครื่องปลายทางได้ · ปลายทาง (`recipes.publish_repo` ใน
  config) ว่าง = local store ในเครื่อง (ลูกค้าใช้แบบนี้ ไม่แตะรีโปเรา) · ทีมตั้งเป็นรีโป
  candidates แล้ว push ขึ้นไป review ก่อน promote เข้า canonical · stamp measured
  capabilities ลง MODEL_FEATURES และเขียน PROFILE.yaml เก็บ provenance (validated_on,
  host, revision) ไว้ให้ตรวจย้อนได้

### Fixed

- **`rebuild` regenerate แล้วเหมือนไม่มีอะไรเปลี่ยน** สองสาเหตุ · (1) default `--output`
  เป็น `./bundles` (relative CWD) → bundle ที่อยู่คนละที่ถูกสร้างเป็นสำเนาใหม่ ส่วนตัวจริง
  ที่ start/publish อ่าน (จาก server.meta) ยังเก่า — ตอนนี้ default เขียนทับ **ที่เดิม** ·
  (2) bundle รุ่นเก่าเก็บ target ที่เลิกใช้แล้ว (`this-machine`) ทำให้ `_compute_fits` โยน error
  แล้ว rebuild ตายทั้งใบ — ตอนนี้ถอยไป auto-detect ตามฮาร์ดแวร์แทน · สองอย่างนี้คือเหตุที่
  controller บางตัว rebuild แล้วไม่ได้ตัวโหลดแบบ aria2c

- **`enable` — port ชนตอนบูต และ start ที่คืน 0 แต่ unit ล้ม** · หลายโมเดล default port
  8000 เท่ากัน · enable หลายตัวแล้ว reboot = ทุกตัวแย่ง 8000 พร้อมกัน ตัวหลังล้ม · ตอนนี้
  enable อ่าน effective port ของแต่ละ bundle แล้วปฏิเสธถ้าชนกับ unit ที่ enable ไว้แล้ว ·
  และ `--now` เช็ก `is-active` ต่อ (is-enabled = "จะถูกเรียกตอนบูต" ไม่ใช่ "เรียกแล้วขึ้น")
  ถ้า start คืน 0 แต่ unit failed → ล้มพร้อม log แทนที่จะรายงานสำเร็จ

- **download โดน CDN throttle ต่อ connection** · single curl stream โดนบีบเหลือ ~150KB/s
  ทั้งที่เครื่องมีแบนด์วิดท์เหลือ · controller ใช้ aria2c -x16 (16 connection ขนาน) ถ้ามี
  ไม่งั้นถอยไป curl · ทั้งคู่ resume ได้ และ verify-files เช็ก size ต่อทุกครั้งอยู่แล้ว

### Fixed

- **`enable` สร้าง unit ที่บูตไม่ขึ้น — และไม่มีอะไรบอกจนกว่าจะรีบูต** · user unit ถูก
  render ด้วยเทมเพลตของ **system** unit แล้วแก้แค่บรรทัด `WantedBy` ทิ้ง `User=` ค้างไว้ ·
  user manager รันเป็น user นั้นอยู่แล้วจึงไม่มีสิทธิ์สลับ user ให้ตัวเอง systemd ตายตั้งแต่
  ก่อนเรียก controller ด้วย `Failed to determine supplementary groups` → `status=216/GROUP`
  · ที่ร้ายคือมันเงียบสนิท: `is-enabled` ตอบ `enabled`, หน้าเว็บขึ้นแบดจ์ autostart,
  `loginctl` บอก `Linger=yes` — ครบทุกสัญญาณว่าใช้ได้ แล้วเครื่องบูตขึ้นมาไม่มีโมเดล ·
  เจอพร้อมกันสามเครื่อง (msi-4, spark-worker, dgx-veerasiam) 2026-08-15
  → `render_unit(..., scope=)` ประกอบ unit ตามสโคปของตัวเองตั้งแต่ต้น ไม่ใช่ render
  แบบหนึ่งแล้วไปแก้ทีหลัง · ครอบคลุม adopted container unit ด้วย

- **ประเมิน KV cache เกินจริงหลายเท่าบน arch แบบ hybrid** · `_kv_dims_from_gguf` คูณ
  KV ด้วยจำนวน layer ทั้งหมด · qwen3.5 / qwen3-next วาง full attention สลับกับ layer SSM
  ที่ state คงที่ไม่โตตาม context แล้วประกาศจังหวะไว้ที่ `full_attention_interval` แทนที่จะ
  ไล่เป็นลิสต์ต่อ layer อย่าง gemma-4 · ทางที่รองรับลิสต์อยู่แล้วจึงจับพวกนี้ไม่ได้เลย
  · เคสจริง `Qwen3.8-27B` (65 layer, interval 4) ถูกคิดเป็น 260 KiB/token → 95 GiB ที่
  context 262,144 ทั้งที่ของจริง 64 KiB/token → 49 GiB (เครื่องวัดได้ 54 GB) ผลคือ `fit`
  ปฏิเสธการรันคู่กับโมเดลอื่นที่จริง ๆ แล้วรันได้สบาย เสียเครื่องไปทั้งเครื่องโดยไม่มีใครรู้

- **`prepare-runtime` บังคับ sudo ทั้งที่ไม่จำเป็น** · `ninja` อยู่ในลิสต์ dependency ที่ต้องมี
  ทั้งที่ cmake ถอยไปใช้ Unix Makefiles ได้เอง · เครื่องที่มีของครบทุกอย่างแล้วจึงยังถูกส่งไป
  `sudo apt-get` เพื่อลง generator ที่ไม่ได้ใช้ แล้วตายทันทีบนเครื่องที่ sudo ขอรหัส
  (ค่าปกติของ Ubuntu และเป็นกรณีเดียวกับที่ hub เรียกผ่าน SSH ที่ไม่มี tty)
  → ตัด ninja ออกจากลิสต์ · เช็ค `sudo -n` ก่อน แล้วบอกคำสั่งที่ต้องรันเองแทนที่จะปล่อยให้
  sudo ตายพร้อม `a terminal is required` ที่ไม่ได้บอกว่าต้องทำอะไรต่อ

- **หา CUDA toolkit ไม่เจอทั้งที่ลงไว้ครบ** · DGX OS วาง CUDA ไว้ที่ `/usr/local/cuda`
  แต่ไม่ได้ใส่ใน PATH ของ shell ที่ไม่ใช่ login (เช่นที่ hub เรียกผ่าน SSH) · `command -v nvcc`
  จึงไม่เจอ เราเตือนว่า "ไม่พบ CUDA" แล้วปล่อยไป build ล้มอีกทีตอน cmake
  → `find_cuda_toolkit` มองหาใน `/usr/local/cuda*/bin` แล้วเติมเข้า PATH ให้เอง

- **`test-tools` ผ่านทั้งที่ tool calling ใช้งานจริงไม่ได้** · เทสยิงด้วย
  `tool_choice: "required"` ซึ่ง engine บังคับรูปแบบผลลัพธ์ด้วย guided decoding ผลจึงเป็น JSON
  ที่ parser อ่านออกเสมอ **ไม่ว่า `--tool-parser` จะตรงกับโมเดลหรือไม่** · แต่ Claude Code,
  Hermes, OpenClaw และ agent ทุกตัวส่ง `auto` ซึ่งโมเดลจะเขียนตามรูปแบบของมันเอง
  · เคสจริง `Nemotron-3-Super-120B-A12B-NVFP4` เขียน `<function=…>` แบบ Qwen ใส่ parser
  `hermes` ที่รอ JSON → ไม่มี `tool_calls` เลย หลุดมาเป็นข้อความธรรมดา ขณะที่ `test-tools`
  ขึ้น PASS · เทสที่ผ่านทั้งที่ของจริงพัง แย่กว่าไม่มีเทส เพราะมันคือเหตุผลให้เลิกสงสัย
  → ค่าตั้งต้นเป็น `both` (auto ก่อน), **auto ไม่ผ่าน = คำสั่งไม่ผ่าน**, และพิมพ์สิ่งที่โมเดล
  เขียนออกมาจริงพร้อมเดา parser ที่ตรงกับรูปแบบนั้นให้

- **`test-reasoning` เตือนผิดบน vLLM รุ่นใหม่** · อ่านแค่ `reasoning_content` ทั้งที่บางรุ่น
  ใช้ชื่อ `reasoning` → ขึ้น WARN ทั้งที่ parser ทำงานอยู่ แล้วส่งคนไปแก้ของที่ถูกแล้ว ·
  ตอนนี้อ่านทั้งสองชื่อ และแยก WARN สองแบบออกจากกัน: "โมเดลพ่น `<think>` ออกมาดิบ ๆ"
  (parser ผิดจริง) กับ "ไม่มีร่องรอยความคิดเลย" (โมเดลไม่ได้คิดในรอบนั้น ซึ่งมักไม่ใช่ปัญหา)

### Added

- **`parsers`** — ถามชื่อ `--tool-parser` / `--reasoning-parser` ที่ engine รองรับจริงจาก
  registry ของมันเอง · เดิมเอกสารแนะให้ `ls` โฟลเดอร์ `vllm/tool_parsers/` แล้วอ่านชื่อไฟล์
  ซึ่งผิดสองชั้น: **ชื่อไฟล์ไม่ตรงกับชื่อที่ลงทะเบียน** (`qwen3xml.py` → `qwen3_xml`) และชื่อจริง
  อยู่ใน lazy registry ที่ยังไม่ถูก import จนกว่าจะมีคนเรียก · เสียเวลาไปหนึ่งรอบโหลด weight
  เพราะเดาชื่อจากไฟล์

- **`docs/USAGE.md` §8.5 — บันทึกจากของจริงของ Nemotron-3-Super-120B-A12B-NVFP4**
  flag ที่ใช้แล้วผ่าน, KV cache 1,297,920 tokens, concurrency 25.41x ที่ 262k, และสามจุดที่
  การ์ดของ NVIDIA เขียนต่างจากที่ image รุ่นใหม่ต้องใช้ (`super_v3` เป็นของ vllm 0.20.0 เท่านั้น,
  `qwen3_coder` กับ `qwen3_xml` เป็นคลาสเดียวกัน, ไม่ต้อง `--trust-remote-code`)


- **บันไดของ context ตันที่ 262,144 จนกลายเป็นเพดานเสียเอง** · `CONTEXT_STEPS` มีขั้นสูงสุดที่
  262,144 ทั้งที่ตัวคำนวณบอกว่าหน่วยความจำรับได้มากกว่านั้น · เคสจริง Kimi-K3 (native 1,048,576)
  บน 2 เครื่อง มีที่พอถึง **735,631 tokens** แต่ถูกเสนอที่ 262,144 — เสียไป 2.8 เท่า ·
  และตารางในหน้าเว็บก็จบที่ขั้นนั้น ทั้งที่ผู้ใช้กรอก 524,288 แล้วระบบตอบว่าใส่ได้ อ่านแล้วขัดกันเอง
  · เป็นความผิดพลาดแบบเดียวกับ `DEFAULT_CONTEXT_CAP` ที่เคยถอดออกไปแล้ว แค่ไม่มีชื่อว่า cap

- **KV cache ของ hybrid Mamba ถูกประเมินเกินจริงสิบเอ็ดเท่า** · NemotronH (และ Jamba/Zamba
  ที่ใช้ท่าเดียวกัน) มี layer ส่วนใหญ่เป็น Mamba ซึ่ง state คงที่ ไม่โตตาม context ·
  เคสจริง `NVIDIA-Nemotron-3-Super-120B-A12B` มี 88 layers แต่ `hybrid_override_pattern`
  บอกว่าเป็น attention แค่ **8 ตัว** — คิด 88 KiB/token ทั้งที่ของจริง 8 KiB/token
  · ผลคือรายงานว่ารับได้ 1.8 คนที่ context 262,144 ทั้งที่ของจริง **19.4 คน**
  · pattern ที่อ่านไม่ออก (ไม่มี `*`) ตกไปทางเดิม — ประเมินเกินยังดีกว่าประเมินเป็นศูนย์แล้ว OOM

- **KV cache ของโมเดล MLA ถูกประเมินเกินจริงหลายสิบเท่า** · DeepSeek-V2/V3 และ Kimi K2/K3
  บีบ K กับ V ให้เหลือ latent ก้อนเดียวต่อ layer (`kv_lora_rank + qk_rope_head_dim`)
  แต่ตัวอ่าน config ใช้สูตร GQA ปกติ (`2 × kv_heads × head_dim`) กับทุกตระกูล ·
  เคสจริง `Kimi-K3-active-slice-32experts` ถูกคิดเป็น **2,581 KiB/token** ทั้งที่ของจริง
  **105 KiB/token** — เกินจริง 24.7 เท่า แล้วไปตัด context เหลือ 16,384 ทั้งที่รับได้ 262,144
  · รูปทรงของ KV ตอนนี้ตัดสินที่ `KvDims.elements_per_token` ที่เดียว

### Added

- **SGLang เป็นรันไทม์ทางเลือก (เครื่องเดียว)** · `lmds deploy <repo> --engine sglang`
  · safetensors เสิร์ฟได้ทั้ง vLLM และ SGLang การเดาจากชนิดไฟล์จึงเป็นค่าตั้งต้น ไม่ใช่คำตัดสิน
  · จำเป็นเพราะ checkpoint NVFP4 บางตระกูล calibrate ด้วย w1/w3 scale ซึ่งรันถูกต้องเฉพาะ
  บน SGLang — ไม่มีก็ต้องยกทั้งตระกูลออกจากระบบ
  · ธงทุกตัวยืนยันจาก `sglang serve --help` ใน `scitrera/dgx-spark-sglang-mm:v0` ที่รัน
  บนเครื่องจริง ไม่ได้อ่านจากเอกสาร
  · **ชื่อ knob ไม่เปลี่ยน** (`MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION` / `MAX_NUM_SEQS`)
  แม้ SGLang จะเรียกธงคนละชื่อ — การแปลงเกิดที่จุดเดียวตอนประกอบคำสั่ง เพราะชื่อพวกนี้คือ
  สัญญาที่ `lmds set`, `bundle.env` และหน้าเว็บใช้ร่วมกันทุก engine
  · GGUF ยังบังคับ llama.cpp เสมอ ต่อให้สั่ง `--engine sglang`

- **`lmds inspect --context <ค่า>` — ถามได้ว่าค่าที่จะตั้งนั้นควรไหม** · เดิมระบบตอบได้แค่
  "context สูงสุดเท่าไร" ซึ่งตามนิยามคือค่าที่คนเดียวกิน KV pool หมดพอดี อ่านแล้วเข้าใจว่า
  ใช้งานได้ตามปกติ · ตอนนี้มีตาราง context × จำนวนคนพร้อมกัน และคำแนะนำเป็นรหัส
  (`single-user`, `thin-margin`, `fp8-would-help`, `over-native`, …) ที่ CLI หน้าเว็บ
  และผู้ช่วย LLM เรียบเรียงเป็นภาษาของตัวเอง · `--kv-dtype fp8` คิดทั้งตารางที่ fp8
- **ผู้ช่วยในหน้าเว็บรู้กติกาเรื่อง context/KV/concurrency** พร้อมคำอธิบายรหัสที่ดึงมาจาก
  ต้นทางเดียวกับตัวคำนวณ (`ADVICE_LEGEND`) — และถูกสั่งห้ามคิดเลขเอง ให้ชี้มาที่
  `lmds inspect --context` แทน เพราะเลขที่ LLM คูณเองผิดแบบดูน่าเชื่อ

- **เตือนเมื่อลิงก์ระหว่างเครื่องขึ้นต่ำกว่าที่การ์ดทำได้** · `/sys/class/net/*/speed` รายงาน
  ความเร็วที่ negotiate ได้ ไม่ใช่ความสามารถของการ์ด พอร์ต 200G ที่ต่อผ่าน switch แล้ว
  auto-negotiate ลงมาเหลือ 50G จึงผ่านเกณฑ์ 25G ไปเงียบ ๆ · NVIDIA ตรวจรับลิงก์ระหว่าง
  DGX Spark ที่ ≥184 Gbit/s `lmds node cluster` และหน้าเว็บจะบอกเมื่อต่ำกว่านั้น —
  เตือนอย่างเดียว ไม่ตัดเครื่องออกจากกลุ่มเพราะยังใช้ได้จริง
- **ปุ่มปรับขนาดตัวอักษรบนหน้าเว็บ (S / M / L / XL)** · ค่าตั้งต้นใหญ่กว่าเดิมหนึ่งขั้น
  เบราว์เซอร์จำระดับที่เลือกไว้ · หน้างานหลายที่รายงานตรงกันว่าของเดิมเล็กไป และ
  "เล็กไป" ของแต่ละที่ไม่เท่ากัน
- **`docs/NVIDIA-CLUSTER-SOURCES.md`** — เอกสารคลัสเตอร์ของ NVIDIA อะไรยืนยันของเรา
  อะไรเติมของใหม่ รวมรุ่น switch/สายที่รับรอง และเพดานจำนวนเครื่อง (ต่อตรง ≤3 · ผ่าน switch ≤4)

### Changed

- **ไอคอนทั้งหน้าเว็บเป็น SVG ในไฟล์เดียวกัน** แทนตัวอักษรสัญลักษณ์ (`▦ ⌂ ◱ ✓ ⚙ ⇄`)
  ซึ่งได้หน้าตาตามฟอนต์ของเครื่องที่เปิด บางเครื่องเห็นกล่องว่าง · หัวข้อหลักได้ไอคอนในกรอบสี
  ปุ่มที่ซ้ำทุกแถว (Download / Tests / Manage / Doctor / Logs) ได้ไอคอนกำกับ

### Fixed

- **ปุ่มของหัวข้อ "Other machines" เคยขึ้นก่อนชื่อหัวข้อ** · `.sec button { order: 3 }` มีผล
  เฉพาะตอนปุ่มเป็นลูกโดยตรงของหัวข้อ แต่หัวข้อนั้นห่อปุ่มไว้ใน `<span>` ปุ่มจึงไม่ได้ order
  อะไรเลย · ตอนนี้ทุกหัวข้อใช้ `.sec-act` เหมือนกัน ไม่ต้องเดาจากรูปทรงของ DOM

## [0.3.0] — 2026-08-13

รุ่นนี้มาจากการไล่รันของจริงบนฟลีตทั้งชุด แล้วพบว่าบั๊กที่เจ็บที่สุดเป็นรูปแบบเดียวกันหมด:
**ระบบมีค่าที่ถูกต้องอยู่แล้ว แต่ค่านั้นไม่เดินทางไปถึงปลายทาง และไม่มีอะไรตรวจผลลัพธ์** จึงพัง
เงียบโดยไม่มี error ให้เห็น — tool parser, chat template, context, สถาปัตยกรรม, ตัวตนของ image
เป็นเรื่องเดียวกันคนละจุด สรุปทั้งหมดพร้อมวิธีตรวจอยู่ที่ [docs/PREFLIGHT.md](docs/PREFLIGHT.md)


### Added

- **`docs/PREFLIGHT.md` — สิ่งที่ระบบตรวจให้ก่อน deploy และทำไม** · ทุกหัวข้อมาจากของที่พังจริง
  บนเครื่องจริง เขียนเป็น อาการ → ต้นเหตุ → สิ่งที่จับมันได้ตอนนี้ · รวมคลาสเดียวที่ preflight
  จับไม่ได้ (container ไม่มี uid ที่ `--user` ส่งไป) ซึ่งบอกไว้ตรง ๆ ไม่ปล่อยให้ดูเหมือนมองข้าม

- **ตรวจสถาปัตยกรรมก่อน deploy สำหรับ llama.cpp** — เช็คเดิมขึ้นต้นด้วย
  `if server.mode == "native": return []` ซึ่ง llama.cpp บน DGX Spark รัน native เป็นปกติ
  เช็คที่มีไว้จับ "โมเดลใหม่กว่ารันไทม์" จึงไม่เคยทำงานกับ engine ที่ต้องการมันที่สุด
  · อ่าน `general.architecture` จากหัวไฟล์ GGUF ในเครื่อง (ผ่าน parser ตัวเดียวกับ inspector
  แบบ stream เพราะ metadata มี vocab อยู่ด้วย) แล้วหาชื่อนั้นใน `libllama.so` ของ build ที่
  โมเดลตัวนั้น pin ไว้
  · เคสจริง: Muse-Glimmer-30B ใช้ `muse-glimmer` ซึ่ง llama.cpp บน spark-head (ตามหลัง
  upstream 296 commit) ไม่รู้จัก — ถ้าไม่ตรวจ อาการแรกคือ start ไม่ขึ้นหลังโหลดไปแล้ว 30 GB

- **รันไทม์ผูกกับโมเดล ไม่ใช่ผูกกับเครื่อง (`RuntimeChoice.native_dir`)** — bundle ฝั่ง vLLM pin
  image ด้วย digest มานานแล้ว แต่ llama.cpp native ชี้ `~/src/llama.cpp` ร่วมกันหมด โมเดลที่
  ต้องการ llama.cpp ใหม่กว่าจึงต้องอัปเกรดตัวที่โมเดลอื่นทั้งเครื่องพึ่งพาอยู่
  · ไม่ pin = ใช้ของกลางตามเดิม (รุ่นเดียวกันใช้ร่วมกันได้ ไม่ต้อง build ซ้ำ)

- **เทสที่ *รัน* controller จริง ไม่ใช่แค่ `bash -n`** — สคริปต์ที่ generate ออกมามี `$4` ที่ไม่ได้
  quote (ตั้งใจให้เป็น field ของ awk แต่ escape หลุด) bash มองเป็น positional parameter แล้ว
  `set -u` ก็ตายตอนผู้ใช้กด start · **syntax ถูกต้องทุกประการ `bash -n` จึงผ่านฉลุย**
  · ตอนนี้ดึงฟังก์ชันออกมารันจริงใต้ `set -euo pipefail` ทั้ง port ว่างและ port ที่มีคนถืออยู่


- **ปุ่ม "อัปเดต" บนแถบหัว — อัปเดต hub แล้วไล่ทุก node ให้จบในกดเดียว** · เดิมหน้าเว็บ
  อัปเดตได้เฉพาะ node ส่วนตัว hub ขึ้นได้แค่ป้าย "มีอัปเดต" พร้อมคำสั่งให้ไปเปิด terminal เอง
  ผลคือลำดับกลับหัว: **hub ค้างที่ commit เก่า → node ทุกเครื่องจึง "ตรงกับ hub" และไม่มีปุ่ม
  update ขึ้น → พอกด update ให้ node สักเครื่อง มันดึงของล่าสุดจาก GitHub แล้วล้ำหน้า hub**
  ทั้งฟลีตจึงไม่เคยอยู่ที่ commit เดียวกัน และคำสั่งที่กดไปทำงานคนละรุ่นกับที่ตั้งใจ
  · `POST /api/update` — `git pull --ff-only` + `install.sh` + restart บริการ (สตรีม log
  เหมือนงานอื่น) แล้วหน้าเว็บรอเซิร์ฟเวอร์กลับมาเองและอัปเดต node ต่อทีละเครื่องพร้อมสถานะ
  · **ไม่ merge ไม่ reset**: เครื่องที่มีไฟล์แก้ค้างจะถูกปฏิเสธพร้อมรายชื่อไฟล์ ไม่กลืนงานที่ยัง
  ไม่ได้ commit ของใครเงียบ ๆ · ดึงจาก remote ที่ repo ตั้งไว้เท่านั้น ไม่รับ URL จาก request
  · restart ถูกยิงแบบหลุดจาก process ของเว็บ (`setsid`) ไม่งั้น systemd ฆ่าตัวที่สั่ง restart
  ไปพร้อมกับตัวเองก่อนคำสั่งได้ทำงาน
  · ปุ่ม `update` ของแต่ละ node ขึ้น**เสมอ**แล้ว ไม่ใช่เฉพาะตอนที่เทียบ commit ได้ว่าตามหลัง
  (hub ที่ไม่ใช่ git checkout หรือ node ที่ยังตอบ commit ไม่ได้ ก็ต้องกดอัปเดตได้อยู่ดี)


- **ตั้งชื่อ model ID เองได้ (`SERVED_MODEL_NAME`)** — ชื่อที่ API เสิร์ฟออกไปคือชื่อที่ client
  ใส่ในฟิลด์ `model` ของ request · ลูกค้าที่ย้ายมาจากระบบเดิมต้องใช้ชื่อเดิมเป๊ะ (เช่น
  `vllm-msi-03/aeon-ultimate`) ไม่งั้น client ทุกตัวต้องแก้ตาม — เดิมชื่อถูก hardcode ไว้ใน
  สคริปต์ เปลี่ยนไม่ได้เลยนอกจาก deploy ใหม่
  · ช่อง **model ID** ในเมนู ⋯ ของเครื่องอื่น และในแท็บจัดการของเครื่องนี้ (ว่าง = ใช้ชื่อที่
  bundle กำหนด) · ทั้งสาม template รับค่าผ่าน env เหมือน knob อื่น ๆ
  · ตรวจแค่ที่จำเป็น: ไม่ว่าง ไม่เกิน 200 ตัว ไม่มีช่องว่าง/ตัวควบคุม — ยอมให้มี `/` ข้างใน
  เพราะชื่อจริงบน gateway เป็นแบบนั้น บังคับรูปแบบแคบกว่านี้จะใช้กับของจริงไม่ได้

- **ชิปเตือน "สิทธิ์แคช ต้องแก้" บนการ์ดเครื่อง** — ปุ่ม "แก้สิทธิ์" มีมาก่อนแล้ว แต่ไม่มีอะไร
  บอกว่าเมื่อไหร่ต้องกด · `lmds agent info` ตรวจเจ้าของแคชโมเดล (ถึงชั้นลูกของ `models--X`
  เพราะเคสที่เจอบ่อยคือ `refs/` `.no_exist/` `.locks/` ข้างในเป็นของ root ทั้งที่โฟลเดอร์
  ข้างนอกไม่ใช่) แล้วส่งขึ้นมาเป็น `host.cache`

- **บอกได้แล้วว่าเครื่องไหนรันโค้ดเก่า + กด update ได้จากหน้าเว็บ** — เลข version ไม่ขยับ
  ทุกคอมมิต (0.2.0 มาหลายสิบคอมมิต) จึงไม่มีทางรู้เลยว่า node ไหนตามหลัง · **สำคัญมาก
  เพราะสถานะทุกอย่างของโมเดล (downloaded, commands, features) คำนวณด้วยโค้ดของ node
  เอง ไม่ใช่ของ hub** — แก้บั๊กที่ hub แล้วเข้าใจว่าทั้งฟลีตได้ของใหม่คือเข้าใจผิด (เจอจริง:
  แก้เคส adopted บน hub แล้ว msi-6 ยังขึ้น "not downloaded" เหมือนเดิมเพราะยังรัน
  โค้ดของเดือนก่อน)
  · `lmds agent info` รายงาน `lmds_commit` ของซอร์สที่ import อยู่จริง
  · การ์ดของ node ขึ้น commit และป้าย **"โค้ดเก่า"** พร้อมปุ่ม **update** เมื่อไม่ตรงกับ hub
  (ใช้ทางติดตั้งเดิมที่สตรีม log อยู่แล้ว ไม่แตะโมเดลที่รันอยู่)
  · `GET /api/version` คืน commit ด้วย · `?check_repo=true` ถาม GitHub (`git ls-remote`)
  ว่ามีของใหม่กว่าที่ hub ถืออยู่ไหม แล้วขึ้นป้าย **"มีอัปเดต"** ข้างเวอร์ชันบนแถบหัว

- **ช่องกรอก HF token ในกล่อง Deploy (ไม่บังคับ)** — ลูกค้าที่มี token ของตัวเองกรอกได้
  ตั้งแต่แรก ไม่ต้องรอให้รุ่น gated ตอบ 401 ก่อนถึงจะมีช่องให้กรอก · ติ๊ก "เก็บไว้ที่ hub"
  เพื่อบันทึกลง keyring (ไม่มีก็ไฟล์สิทธิ์ 0600) แล้วใช้กับรุ่น gated ครั้งต่อไปเอง ·
  ไม่ติ๊ก = ใช้เฉพาะรอบนั้น ไม่ถูกเก็บที่ไหน · `POST /api/secrets/hf` ไม่เคยตอบค่า token
  กลับออกไปและไม่เขียนลง bundle
  > แหล่งโมเดลที่รองรับตอนนี้มีแค่ Hugging Face (Ollama/NGC อยู่ roadmap เฟส 2) token
  > จึงมีช่องเดียว — เพิ่มแหล่งอื่นเมื่อไหร่ค่อยเพิ่มช่องของแหล่งนั้น

- **ปุ่ม "แก้สิทธิ์" ต่อเครื่อง** — คืนเจ้าของแคชโมเดล (`~/.cache/huggingface`,
  `~/.cache/flashinfer`) ให้เป็นของ user ด้วย `chown -R` ผ่านทางเดียวกับปุ่ม `setup`
  (ถามรหัส sudo ครั้งเดียว ส่งทาง stdin ไม่เก็บ ไม่อยู่ใน argv) · `POST
  /api/nodes/{name}/fix-permissions` · แตะเฉพาะแคชใน home ของ user เท่านั้น


- **ดึงสูตรจากรีโป controller ของทีมได้ (`lmds recipes --sync`)** — ต้นทางเดียวคือรีโป
  (ค่าเริ่มต้น `neronain/dgx-spark-all-controllers`) แก้ที่นั่นแล้ว push จากนั้น sync ที่ hub
  · อ่าน **ส่วนหัวของสคริปต์** (`MODEL_ID`/`HF_REPO`, `VLLM_IMAGE`, `MODEL_FEATURES`,
  `MAX_NUM_SEQS`, `KV_CACHE_DTYPE` …) — **ไม่รัน controller** บน hub
  · สูตรจากรีโปชนะสูตรที่ฝังมากับ LMDS เมื่อเป็นรุ่นเดียวกัน · ไฟล์ที่อ่านไม่ได้ถูกรายงานพร้อม
  เหตุผล ไม่หายเงียบ · รุ่นที่มีทั้ง single/stacked ใช้ตัว single (LMDS เลือก topology เอง)
  · หน้าเว็บ: ปุ่ม **ดึงจาก GitHub** ในแผง Proven recipes + บรรทัดบอก repo/commit/เวลาที่ดึง
  · endpoint ใหม่ `POST /api/recipes/sync` · ครั้งแรกดึงมาได้ 19 สูตรจาก 21 controller
- **เลือกสูตรได้ตั้งแต่ในกล่อง Deploy** — เดิมมีแค่ช่องพิมพ์ลิงก์ HF เปล่า ๆ ทั้งที่คนไม่มี LLM
  คือกลุ่มที่ต้องพึ่งสูตรมากที่สุด และสูตรถูกซ่อนอยู่ในแผงล่างสุดที่ต้องกด Show ก่อน
  · ชิปเลือกรุ่น กดแล้วเติมชื่อโมเดล + ติ๊ก *Skip the LLM* + บอก engine/image/ทดสอบบนอะไร
  · **ยังไม่ได้ตั้ง LLM = กางรายการให้เอง** ไม่ต้องกดหา
  · แผนที่ได้ขึ้นแถบเขียว "ใช้สูตรที่รันผ่านจริง: …" แทนที่จะเติมค่าเงียบ ๆ (`recipe` ใน
  payload ของ `/api/deploy/analyze`)
  · `lmds recipes` แสดงคำสั่งพร้อมใช้ต่อ (`lmds deploy <รุ่น> --no-llm`) และที่มาของชุดสูตร

- **เลือกธีมเองได้ (สว่าง / มืด / ตามเครื่อง)** — ปุ่มบนแถบหัว วนสามโหมด จำไว้ในเบราว์เซอร์
  (`localStorage`) · เดิมหน้าเว็บดูค่า `prefers-color-scheme` อย่างเดียว เครื่องที่ตั้ง OS เป็น
  โหมดมืดจึงถูกบังคับให้ใช้หน้าจอมืดตลอด เลือกสว่างไม่ได้เลย
  · **ค่าเริ่มต้นคือธีมสว่าง** — โทนเดียวกับที่ออกแบบไว้แต่แรก (พื้นเทาอ่อน การ์ดขาว เน้นสีน้ำเงิน)
  · ชุดสีมืดย้ายมาอยู่ที่ `:root[data-theme="dark"]` ที่เดียว (เดิมอยู่ใน media query) —
  สคริปต์เล็ก ๆ ใน `<head>` ตัดสินธีมก่อนวาดหน้า จึงไม่เห็นหน้าขาววาบก่อนเปลี่ยนเป็นมืด
  · โหมด "ตามเครื่อง" ยังเปลี่ยนตาม OS ทันทีแม้เปิดหน้าค้างไว้

- **ลากจัดลำดับการ์ดเครื่องเองได้** — จับที่ `⠿` บนหัวการ์ดแล้วลากขึ้นลง (ที่จับแยกต่างหาก
  เพราะในหัวการ์ดมีปุ่มและช่องกรอก cluster IP อยู่) · ใช้ pointer events จึงลากด้วยนิ้วบน
  แท็บเล็ตได้ ไม่ใช่ HTML5 drag-and-drop ที่จอสัมผัสใช้ไม่ได้
  · ลำดับเก็บที่ **hub** (`config.yaml` → `ui.node_order`) ไม่ใช่ในเบราว์เซอร์ — เปิดจากเครื่องไหน
  ก็เห็นลำดับเดียวกัน และ `lmds node list` / `lmds node cluster` เรียงตามลำดับเดียวกัน
  · endpoint ใหม่ `PUT /api/nodes/order` (เก็บเฉพาะชื่อที่มีในทะเบียนจริง เครื่องใหม่ต่อท้ายเอง)
  · **การจัดกลุ่ม stacked ยังทำงานเหมือนเดิม** — กลุ่มไปรวมกันที่ตำแหน่งของสมาชิกตัวแรก
  ตามลำดับที่ลาก และลากสลับลำดับภายในกลุ่มได้ ซึ่งมีผลถึงเครื่องที่ถูกเสนอเป็น **head**
  ตอน `lmds node cluster --write`
  · รีเฟรชอัตโนมัติทุก 5 วิ ไม่จัดลำดับทับระหว่างลาก

- **สั่ง "ไม่เอาเครื่องนี้เข้ากลุ่ม stacked" ได้** — กลุ่มเป็นสิ่งที่ระบบ *เสนอ* จากฮาร์ดแวร์ที่ตรงกัน
  ไม่ใช่สิ่งที่ประกาศไว้ เครื่องที่ตั้งใจให้รันงานของตัวเองจึงเคยถูกดูดเข้าคลัสเตอร์เองโดยไม่มีทางห้าม
  · node: `lmds node set <ชื่อ> --no-stack` (เก็บใน `Node.stack` ของทะเบียน)
  · hub เอง: `lmds node cluster --no-self-stack` (เก็บใน `config.yaml` → `cluster.stack_self`
  เพราะ hub ไม่มีแถวในทะเบียน)
  · หน้าเว็บ: ปุ่ม **ไม่เอาเข้ากลุ่ม / เอาเข้ากลุ่ม** ในแถบ cluster ของแต่ละเครื่อง + แถวของ hub
  เองเหนือลิสต์เครื่อง (ต้องอยู่นอกกลุ่ม ไม่งั้นพอปิดแล้วกลุ่มหาย = กดเปิดคืนไม่ได้)
  · `PATCH /api/nodes/{name}` รับฟิลด์ `stack` · เพิ่ม `PATCH /api/cluster/self`
  · ไม่มีค่า = เข้ากลุ่มได้ตามเดิม (ทะเบียนเก่าและ `stack: null` ไม่กลายเป็นปิดทั้งฟลีต)

### Changed

- **แนะนำ context เท่าที่เครื่องรับไหวจริง ไม่ตัดด้วยเลขที่ตั้งเอาเอง** — `DEFAULT_CONTEXT_CAP`
  = 65,536 คร่อมทุก recommendation ไม่ว่าหน่วยความจำจะเหลือแค่ไหน · โค้ดเดิมถึงกับเขียน note
  เตือนตัวเองว่า "รองรับได้ถึง 262,144" แล้วก็ยังส่ง 65,536 ให้อยู่ดี ซึ่งเป็นสัญญาณว่าค่า
  default ผิด ไม่ใช่ว่า note ยังไม่ดีพอ
  · ค่าที่แนะนำคือ `safe` ตรง ๆ = `min(หน่วยความจำที่เหลือ, native context)` ที่หารด้วย
  concurrency มาแล้ว จึงเป็นค่าที่ทั้งเครื่องและโมเดลรับไหวโดยนิยาม

- **llama.cpp เสิร์ฟ 1 slot เป็นค่าเริ่มต้น** — `--ctx-size` เป็น *pool* ที่ถูกหารเท่า ๆ กันให้ทุก
  slot ส่วน fit คำนวณที่ `concurrency=1` ค่าที่ได้จึงเป็นของ slot เดียวอยู่แล้ว พอ `max_num_seqs`
  default เป็น 4 ก็ถูกหารซ้ำอีกรอบ
  · เคสจริง: Muse-Glimmer แผน/README/banner บอก 131,072 แต่ `/props` รายงาน **32,768**
  · vLLM ไม่แตะ เพราะแชร์ KV แบบ dynamic จึงไม่เคยมีปัญหานี้
  · ตั้ง `PARALLEL_SEQS` เองได้ แล้วแผนจะเตือนว่าแต่ละ request จะเหลือเท่าไร

- **อ่าน KV dims ของโมเดล sliding-window ได้แล้ว** — gemma-4 เขียน `head_count_kv` เป็นลิสต์
  ต่อ layer · parser เช็ค `isinstance(int)` แล้วคืน `None` → analyser เข้าสาขา "ไม่รู้มิติ KV" ที่ตั้ง
  context ไว้แค่ 16,384 · ตอนนี้นับเฉพาะ layer full-attention (ที่ KV โตตาม context จริง) ได้
  81,920 B/token ตรงกับ 18.9 GiB ที่เครื่องจองเพิ่มจริงตอนขึ้นจาก 16,384 → 262,144

- **เลือก mmproj ตามตระกูลของ weight ไม่ใช่ตามขนาด** — กติกา "เล็กสุด" ถูกเมื่อ repo มี projector
  ตัวเดียวหลายระดับ quant แต่บาง repo ใส่ projector ของ *โมเดลคนละตัว* ไว้ด้วยกัน
  · เคสจริง: `mmproj-kquant.gguf` เล็กที่สุดแต่คู่กับ `dflash-kquant` ไม่ใช่ weight ที่เลือกไว้ —
  มันจะโหลดขึ้นได้และ vision จะผิดแบบเงียบ ๆ

- **ไม่ start ทับ port ที่คนอื่นถืออยู่** — `wait_health` ยิง `/health` ที่ port ของตัวเอง แล้วโมเดล
  ที่ยึด port อยู่ตอบ 200 ให้ → รายงานว่า start สำเร็จทั้งที่ตัวเองไม่ได้ bind เลย
  · ระบบ **ไม่เลือก port ให้** (เครื่องเดียวรัน llama.cpp หลายตัวได้ ผู้ใช้เป็นคนกำหนด) แต่ปฏิเสธ
  การ start ทับ พร้อมบอกว่าใครถืออยู่

### Fixed

- **นับ reasoning ที่ vLLM แยกออกมาได้ทั้งสองชื่อ** — build ใหม่คืน `reasoning` build เก่าคืน
  `reasoning_content` · ดูแค่ชื่อเดียวแปลว่า `--reasoning-parser` ที่ทำงานอยู่ถูกรายงานว่าไม่มี


- **container ที่รันเป็น root ทำให้แคชของ user พังทีละนิด** — `start` เดิม mount แคชเข้า
  container ที่เป็น root ทุกไฟล์ที่มันเขียน (tokenizer, `.locks`, ไฟล์ที่ขาดแล้วดึงเพิ่ม) จึง
  กลายเป็นของ root ในแคชของ user · **บน msi-5 ลามจนทั้ง `~/.cache/huggingface/hub`
  (73 GB) เป็นของ root และ user เขียนไม่ได้** — โมเดลตัวถัดไป download ไม่ลง, `remove`
  ลบไม่ออก, `sync-worker` ตายด้วย rsync exit 23 · ตอนนี้ `start` รันในฐานะ user เดียวกับ
  ที่สั่ง (ทางเดิน `download` ทำแบบนี้อยู่ก่อนแล้ว) · image ที่ต้องการ root จริง ๆ ข้ามได้ด้วย
  `LMDS_RUN_AS_ROOT=1`

- **weight ที่ผู้ใช้โหลดเองถูกมองว่า "ยังไม่ได้ download"** — HF cache มีสองเลย์เอาต์
  (`$HF_HOME/hub/models--X` ปัจจุบัน กับ `$HF_HOME/models--X` เก่า) ซึ่งเลย์เอาต์เก่าคือที่ที่
  weight ที่โหลดเองมักไปอยู่ · single-vllm รู้จักแค่แบบแรก จึงขึ้นว่ายังไม่ได้โหลดทั้งที่ไฟล์ครบ
  แล้ว `download`/`repair` ก็โหลดซ้ำลง `hub/` อีกหลายสิบ GB (แบบ stacked รองรับสอง
  เลย์เอาต์อยู่ก่อนแล้ว) · แก้ทั้ง `snapshot_dir()`, `HF_HUB_CACHE` ฝั่ง container และ
  `weights_path()` ที่ `remove` ใช้หา weight

- **กด repair บน bundle ที่ผู้ใช้ดูแล weight เองแล้วไม่มีอะไรเกิดขึ้น** — `repair` คือ
  `download` + `verify-files` ซึ่ง bundle จาก `lmds adopt` ไม่มีทั้งคู่ · เดิมสั่งไปแล้วได้ usage
  ของ bash กลับมา อ่านไม่รู้เรื่องและทำให้เข้าใจว่า repair พัง · ตอนนี้บอกตรง ๆ ว่าทำไมถึงไม่ใช่
  เรื่องของ bundle แบบนี้ พร้อมทางที่ใช้ได้จริง (`lmds start` เพื่อดูสาเหตุ หรือ `lmds deploy`
  ถ้าอยากให้ LMDS ดูแล weight ให้)
  · และตัดปุ่ม `download`/`repair` ออกจากหน้าเว็บสำหรับ bundle แบบนี้ — ตัวสคริปต์เองคือ
  ความจริงสุดท้าย ไม่ใช่การเดาจาก model id (bundle ที่ adopt มาแล้ว id บังเอิญเป็นรูป
  `org/name` เคยหลุดตัวกรองเดิม)

- **bundle ที่มาจาก `lmds adopt` ขึ้น "not downloaded" ตลอดกาล** — adopt บันทึก `model.id`
  เป็น path *ในคอนเทนเนอร์* (เช่น `/models/qwen3-coder-next`) ซึ่งบนโฮสต์ไม่มีอยู่จริง
  ตัวตรวจ weight เอาไปคิดเป็น repo id ของ HF จึงหาไม่เจอตลอด แล้วยื่นปุ่ม `download`
  ที่กดไปก็ล้มแน่นอน (เจอกับ qwen3-coder-next-nvfp4-gb10 บน msi-6) — ขัดกับหลักของ adopt
  เองที่เขียนไว้ว่า "ไม่แกล้งทำเป็นมี download"
  · bundle แบบนี้ขึ้นสถานะ **"stopped · weight จัดการเอง"** แทน · ซ่อนปุ่ม `download`/
  `repair`/`verify-files` ที่ทำอะไรไม่ได้จริง และบอกตรง ๆ ว่าทำไม
- **`sync-worker` ตายด้วย rsync exit 23 เพราะไฟล์เป็นของ root** — container ที่รันเป็น
  root โหลด weight ลงแคช พอคัดลอกไป worker ในฐานะ user จึงอ่านไฟล์โหมด 600 ของ root ไม่ได้
  (เจอจริงกับ DeepSeek-V4-Flash บน spark-head: 164 รายการเป็นของ root, 1 ไฟล์อ่านไม่ได้)
  - `doctor` เดิมเช็กแค่ "เขียนโฟลเดอร์ได้ไหม" — โฟลเดอร์ปกติแต่มีไฟล์ root ปนอยู่จะขึ้น ✅
    ทั้งที่ sync-worker พังแน่ · ตอนนี้หา**ไฟล์ที่อ่านไม่ได้**ด้วย (หยุดที่ไฟล์แรก มีเพดาน
    การไล่ไม่ให้ช้าบนแคชใหญ่)
  - งานที่จบด้วย Permission denied ของ rsync ขึ้นบรรทัดบอกวิธีแก้ต่อท้าย log แทนที่จะให้
    ไปนั่งอ่าน error ของ rsync เอง

- **กดปุ่มของ controller แล้ว "เงียบ"** (เจอกับ `sync-worker` บนหน้าเว็บ) — คำสั่งอย่าง
  `sync-worker` คัดลอก weight ทั้งก้อนข้ามเครื่อง กินเวลาเป็นสิบนาที แต่ endpoint
  `/ctl/{command}` รอให้จบใน HTTP request เดียว (timeout 3600 วิ) ปุ่มจึงค้างเป็น
  `sync-worker…` และมักถูกตัดสายก่อนงานจบ
  - คำสั่งยาว (`prepare-runtime`, `sync-worker`, `verify-worker`, `verify-files`,
    `clear-fi-cache`, `bench`, `stress`) รันเป็น **job แล้วสตรีมผล** เหมือน start/repair
    · คำสั่งสั้นที่เหลือ timeout ลดจาก 1 ชั่วโมงเหลือ 10 นาที (เดิมยึด thread ของเว็บได้นานมาก)
  - **ไม่มีปุ่มไหนค้างได้อีก**: เพิ่ม `withButton()` ที่คืนสถานะปุ่มใน `finally` เสมอและแสดง
    เหตุผลเมื่อพลาด · เดิม 14 จุดที่ปิดปุ่มแล้วยิง API ไม่มี try/catch เลย สายหลุดเมื่อไหร่
    ปุ่มตายเงียบทันที
  - ตาข่ายรับท้าย `unhandledrejection`: ปุ่มที่ค้างอยู่ในสถานะ "…" ถูกปลดคืนและมีข้อความบอก
    แม้จุดนั้นจะยังไม่ได้ย้ายมาใช้ `withButton`


- **เครื่องที่คุยกับกลุ่มไม่ได้ ถูกนับเข้าคลัสเตอร์จนแผน parallel เพี้ยน** — เจอหน้างาน: เครื่องที่สาม
  โผล่ใน `CLUSTER A` ทำให้ world size เป็น 3 แล้วระบบสั่งให้ใช้ pipeline ทั้งที่อีกสองเครื่องใช้ TP=2 ได้
  · การเข้ากลุ่มเดิมดูแค่ "ฮาร์ดแวร์ตรงกัน" ตอนนี้ต้องครบสามข้อ: **ฮาร์ดแวร์ตรงกัน · เป็นคนละเครื่องจริง ·
  มีขาอยู่ subnet เดียวกัน**
  - `stack_ready()` เลิกใช้ `best_gbps` (ซึ่งนับพอร์ตที่ยังไม่ได้ตั้ง IP · 169.254.x.x ด้วย) เปลี่ยนไปดู
    ลิงก์ที่ใช้ยิง NCCL ได้จริงตาม `fabric_links()` — เครื่องที่มีแต่พอร์ต link-local จึงไม่เข้ากลุ่มอีก
  - เครื่องที่ไม่มี subnet ร่วมกับกลุ่มไปอยู่ใน `excluded` (เหตุผล `no-shared-fabric`) — ไม่นับใน
    world size และ **ไม่ทำให้กลุ่มกลายเป็น "ยังไม่พร้อม"**
  - เครื่องเดียวกันที่ถูกเพิ่มไว้สองชื่อ (Tailscale/DNS/IP คนละทาง — ทะเบียนกันซ้ำไม่ได้) ถูกยุบเป็นตัวเดียว
    ด้วย hostname + ชุด IP บนสายเร็ว (`machine_identity()`) เหตุผล `same-machine`
  - เพิ่ม `usable_world_size` (เฉพาะเครื่องที่ตั้ง cluster IP ถูกต้องแล้ว) แสดงคู่กับ world size ทั้งใน
    หน้าเว็บและ CLI · การ์ดเครื่องในหน้าเว็บโชว์ hostname จริงเมื่อไม่ตรงกับชื่อในทะเบียน
  - `cluster_note()` แยก "สายช้าเกินไป" ออกจาก "สายเร็วพอแต่ยังไม่ได้ตั้ง IP" — คนละวิธีแก้

## [0.2.0] — 2026-08-06

**คุมหลายเครื่องจากหน้าเดียว + ทุกอย่างที่มาจากการรันจริง**

0.1.0 คือ CLI ที่สร้าง bundle ให้เครื่องตรงหน้าได้ · 0.2.0 คือระบบที่คุม **ฟลีตทั้งชุด**
จากเครื่องเดียว และเรียนรู้จากการเอาไปใช้จริงบน DGX Spark 4 เครื่อง + controller หนึ่งตัว

สามอย่างที่เปลี่ยนวิธีทำงานจริง ๆ:

1. **`lmds node`** — คุมทุกเครื่องผ่าน SSH โดยไม่ต้องมี daemon · ตรวจ ConnectX/200G แล้ว
   จับคู่ที่ stacked ได้เอง · ส่ง bundle ไปติดตั้งข้ามเครื่อง (`node push`)
2. **หน้าเว็บที่ทำได้เท่า CLI** — deploy wizard, ชุดทดสอบ, จัดการ, เกจทรัพยากร, หน้า login,
   สตรีมความคืบหน้าของงานยาว · โมเดลบนเครื่องอื่นคุมได้เท่ากับโมเดลในเครื่อง
3. **`lmds recipes` + `lmds smoke`** — สูตรที่ผ่านการรันจริงสำหรับคนที่ไม่มี API key
   และตัวพิสูจน์ว่า bundle รันได้จริง ไม่ใช่แค่สคริปต์ถูก

> **บทเรียนหลักของรอบนี้:** gate ที่ตรวจแบบ static ไม่พอ · บั๊กที่เจ็บที่สุดทุกตัว
> (image ที่ tag ไม่มีอยู่จริง, head container ที่ไม่เคย start, ชุดทดสอบที่ไปให้คะแนน
> เซิร์ฟเวอร์ของโมเดลอื่น, `disable` ที่รายงานสำเร็จทั้งที่ sudo ล้ม) **ผ่าน gate ทั้งหมด**
> แล้วไปตายตอนรันจริง — `lmds smoke` เกิดจากข้อนี้

รอบนี้เกือบทั้งหมดมาจาก **การรันจริงบน DGX Spark** — ทุกข้อคือปัญหาที่เจอหน้างานจริง

### Added

- **คุมหลายเครื่องจากเครื่องเดียว (fleet หลายเครื่อง)** — เครื่องที่คุณใช้เป็น *hub* คุมเครื่องอื่นผ่าน SSH
  · `lmds node add <ip> --user <u>` ถามรหัสผ่าน **ครั้งเดียว** เพื่อติดตั้ง SSH key ของ LMDS แล้วทิ้งทันที
  — **ทะเบียนไม่มีฟิลด์รหัสผ่านโดยตั้งใจ** (มีเทสกันไม่ให้เผลอเพิ่มกลับเข้ามา)
  · `node list [--check]` / `node set` / `node remove` / `node run <name> <คำสั่ง lmds...>` / `ps --all`
  · **node ไม่ต้องรัน daemon** และไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22 — hub เรียก `lmds agent info` ผ่าน SSH
  · **ไม่ต้องใช้ root** — user ที่อยู่ในกลุ่ม `docker` พอ
  · เครื่องหนึ่งล่มต้องไม่ทำให้หน้าเว็บพัง — แถวนั้นขึ้นว่าติดต่อไม่ได้แล้วจบ
- **ทรัพยากรสดต่อเครื่อง** — CPU (core + load 1 นาที) · RAM/Unified memory · **VRAM ที่ใช้จริง + %busy**
  · ดิสก์ (ใช้/ทั้งหมด) · ความเร็วสาย · **จำนวนโมเดลที่รันอยู่** (llama.cpp รันได้หลายตัวพร้อมกัน จึงเป็น
  ตัวเลข ไม่ใช่ใช่/ไม่ใช่) — แสดงทั้งหน้า "เครื่องนี้" และทุกเครื่องในทะเบียน จาก payload ชุดเดียวกัน
- **ตรวจ ConnectX / 200G / RDMA แล้วบอกว่าเครื่องไหน stacked ด้วยกันได้** — `lmds node cluster`
  · อ่านจาก `/sys` ล้วน (ไม่ต้อง root ไม่ต้องมี ethtool/ibstat): ความเร็วลิงก์, vendor `0x15b3`,
  `/sys/class/infiniband/*`, IP ของแต่ละ interface
  · จับกลุ่มเฉพาะเครื่องที่ **arch/profile/รุ่น GPU/จำนวน GPU ตรงกัน และมีสาย ≥ 25G ทั้งคู่** —
  ความเร็วที่รายงานเป็นของ **เครื่องที่ช้าที่สุดในกลุ่ม** เพราะ NCCL รอ rank ที่ช้าที่สุดเสมอ
- **กำหนด cluster IP ที่ NCCL จะใช้** — `lmds node set <name> --cluster-ip <ip>` (ตรวจว่าเป็น IPv4 ที่ใช้ได้จริง
  และตรงกับ interface ที่ตรวจพบ) · ระบบ **เสนอ** IP บนสายเร็วสุดให้ แต่ไม่ตั้งเอง เพราะเดาผิดแล้ว
  stacked จะค้างตอน NCCL init โดยไม่บอกสาเหตุ
- **`lmds node cluster --write <slug>`** — เขียน `cluster.env` ลง bundle
  (`MASTER_IP`/`WORKER_IP`/`SSH_USER`/`TRANSPORT_IP_*`/`NCCL_SOCKET_IFNAME`) → stacked controller
  source ไฟล์นี้**ก่อน default ทั้งหมด** แล้วข้ามการถาม IP ตอน `start` (env ภายนอกยังชนะไฟล์นี้เสมอ)
- **รองรับคลัสเตอร์เกิน 2 เครื่อง (เตรียมไว้ ยังไม่ได้รันจริง)** — controller วน worker ทุกตัวจาก
  `WORKER_IPS` ทุกขั้นตอน (prepare-runtime, sync/verify-worker, start ตาม node-rank, stop, status,
  logs, clear-fi-cache) · ค่าเริ่มต้นของ bundle 2 เครื่องยังเป็น worker เดียวเหมือนเดิมทุกประการ
  · target preset ใหม่ `dgx-spark-stacked-4` · `TargetSpec.node_count` แยก "หลาย GPU ในเครื่องเดียว"
  (RTX dual) ออกจาก "หลายเครื่องเครื่องละใบ" (Spark stacked) ซึ่งเดิมปนกันอยู่ที่ `gpu_count`
  · `lmds node cluster` เตือนเมื่อจำนวนเครื่องไม่เข้ากับ tensor parallel (3 เครื่อง = TP=3 หาร
  attention head ไม่ลง ต้องใช้ TP=2 + pipeline)
- **ที่อยู่สำรองต่อเครื่อง (`node set --alt-host`)** — เครื่องเดียวกันเข้าได้หลายทาง (LAN ที่ออฟฟิศ,
  Tailscale/VPN ตอนออกนอก) · hub ลองที่อยู่หลักก่อน ต่อไม่ถึงจึงค่อยลองสำรอง ไม่ต้องแก้ทะเบียน
  ตอนย้ายที่ทำงาน · **failover เฉพาะเมื่อต่อไม่ถึงจริง** (timeout/no route/refused) คำสั่งที่ต่อได้
  แต่ล้มด้วย exit code ของตัวเองจะไม่ถูกยิงซ้ำ — ไม่งั้น `start` จะทำงานสองรอบ
- **`lmds node ctl <เครื่อง> <slug> <คำสั่ง>`** — สั่ง *สคริปต์ controller* บนเครื่องปลายทางจาก hub
  · เดิมมีแต่ `node run` ที่สั่งได้เฉพาะ *คำสั่งของ lmds* ส่วนขั้นตอนของ stacked
  (`prepare-runtime`, `sync-worker`, `verify-worker`, `test-text`) ต้อง ssh เข้าไปเอง
  — runbook ถึงกับเขียนคำสั่ง `run-controller` ที่ไม่มีอยู่จริงไว้
- **`lmds prune`** — ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้ว · เครื่องที่ใช้ **จัดการ** อย่างเดียว
  จะสะสมรายการปลอมจนสั่งการผิดเครื่องได้ (เจอจริงบน Mac: โชว์ 8 โมเดลทั้งที่ไม่มี GPU)
- **`lmds recipes` — สูตรที่รันผ่านจริง ใช้แทน LLM เมื่อลูกค้าไม่มี API key** · ทีม SI หลายรายแจ้งว่า
  ไม่มี provider ให้ใส่ จึงสร้างอะไรไม่ได้ · `--no-llm` ใช้ได้ก็จริงแต่ rule-based ไม่รู้เรื่องเฉพาะรุ่น
  ทำให้ **deploy ผ่านแต่ start ไม่ขึ้น** · แคตตาล็อกใน `src/lmds/recipes/catalog.yaml` เก็บ image,
  serving flags, parser, env ที่ทดสอบบนฮาร์ดแวร์แล้ว — ทุกสูตร**ต้องมี `source` และ `validated_on`**
  (มีเทสบังคับ) · ไม่แตะ context เพราะต้องมาจากเครื่องเป้าหมาย · `image_for` ผูก image กับสถาปัตยกรรม
  ที่ทดสอบมา ไม่ให้ build ของ DGX Spark ถูกใช้กับ RTX เงียบ ๆ
  · สูตรชุดแรก 7 รุ่นจากรีโป deployment ของทีม: DeepSeek-V4-Flash, Llama-3.3-70B, Qwen3.5-122B GPTQ,
  Qwen3-Coder-Next NVFP4, GLM-4.7-Flash, Gemma-4-31B, Nemotron-3-Super
- **`lmds scan` — หา weight ที่มีอยู่แล้วบนเครื่อง ไม่ว่าจะเก็บไว้แบบไหน** · เครื่องลูกค้ามักมีโมเดล
  อยู่ก่อนติดตั้ง LMDS และไม่ได้จัดระเบียบแบบเดียวกับเรา · ค้นจาก env (`HF_HOME`, `HF_HUB_CACHE`,
  `TRANSFORMERS_CACHE`, `MODEL_DIR`, `LLAMA_CACHE`) + ที่ที่นิยมวางกัน (`~/.cache/huggingface[/hub]`,
  `~/models`, `/models`, `/opt/models`, …) · รายงานชนิด/ขนาด/shard/path/**เลย์เอาต์ของ HF cache**
  · `--all` ค้นทุกเครื่องในทะเบียน · **อ่านอย่างเดียว ไม่ย้ายไม่ลบ** — weight เป็นของผู้ใช้
- **`--gpu-util` ปรับได้จากบรรทัดคำสั่ง** (vLLM ทั้ง single และ stacked) — unified memory ชน OOM
  ง่ายกว่าการ์ดแยกเพราะ CPU/GPU ใช้ pool เดียวกัน · ตรวจค่านอกช่วง 0.3–0.98 ด้วย `awk`
  เพราะ bash เทียบทศนิยมไม่ได้ (`(( ))` จะตัด `0.80` เป็น `0` เงียบ ๆ)
  · วัดจริง: 0.85 → ใช้ 109/121 GB (KV 221,056 tokens) · 0.80 → 103 GB (KV 184,080 tokens)
- **`lmds agent info`** — พิมพ์สถานะเครื่องเป็น JSON ให้ hub อ่าน (ปกติไม่ได้พิมพ์เอง)
- **`lmds node install <ชื่อ>` / `node add --install`** — ติดตั้งหรืออัปเดต LMDS บนเครื่องปลายทาง
  จาก hub (clone/pull จาก GitHub → `install.sh` บนเครื่องนั้น) · ข้ามขั้น Docker/toolkit เป็นค่าเริ่มต้น
  เพราะขั้นนั้นต้องใช้ sudo ที่ไม่มีคนกรอกรหัสผ่านผ่าน SSH (`--with-prereq` ถ้า sudo ผ่านโดยไม่ถาม)
  · เอกสารทุกฉบับระบุชัดแล้วว่า **ทุกเครื่องต้องมี `lmds` ติดตั้งอยู่** — "agent" คือตัวคำสั่งเอง
- **เสนอ cluster IP จาก fabric ที่ทุกเครื่องมีขาร่วมกัน** — DGX Spark มี fabric มากกว่าหนึ่งวง
  (`10.100.152.0/24` และ `10.100.153.0/24` บนเครื่องจริง) ปล่อยให้แต่ละเครื่องเลือกเองอาจได้คนละวง
  · เพิ่ม blocker `split-fabric` เมื่อ IP ที่ตั้งไว้อยู่คนละวง — ต่อกันไม่ติดทั้งที่แต่ละเครื่องดูถูกหมด
- **stacked controller: หา NCCL interface เองจาก cluster IP** (ทั้ง head และ worker) — ชื่อพอร์ตบน
  DGX Spark ยาวและไม่เหมือนกันทุกเส้น (`enp1s0f1np1` vs `enP2p1s0f1np1` คนละ fabric บนเครื่องเดียวกัน)
  ให้คนพิมพ์เองแล้วผิดจะเงียบ ๆ ตกไปใช้เส้นช้า · ค่าที่ตั้งเองยังชนะเสมอ
- **stacked controller: หา RoCE HCA เองจาก interface** (`/sys/class/infiniband/*/device/net/`)
  — ไม่ตั้ง `NCCL_IB_HCA` แล้ว NCCL ตกไปใช้ TCP ทำให้สาย 200G ทำงานได้เท่าอีเทอร์เน็ตธรรมดา
  ยืนยันบนเครื่องจริง: `10.100.152.1` → `enp1s0f1np1` → `rocep1s0f1`
- **stacked controller: ตรวจว่ารันบนเครื่อง head จริง** ก่อนเริ่ม — รันผิดเครื่องตายทันทีพร้อมเหตุผล
  แทนที่จะไปตายตอน NCCL init · เพิ่ม `UCX_NET_DEVICES` / `OMPI_MCA_btl_tcp_if_include`
  (สองตัวนี้เลือกเส้นเองแยกจาก NCCL ไม่บอกด้วยจะหลุดไปใช้ management NIC)
  — ทั้งสามข้อมาจากสคริปต์ Llama 3.3 70B ที่ผู้ใช้รันจริงบน DGX Spark
- **หน้าเว็บ: ส่วน Other machines + Cluster fabric** — เพิ่มเครื่อง, ดูทรัพยากรสด, สั่ง start/stop/doctor
  ข้ามเครื่อง (allowlist ฝั่ง server: `start stop restart repair doctor`), แก้ cluster IP ได้ในตาราง
- **เอกสารใหม่ [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md)** — fleet vs stacked, สถาปัตยกรรม
  ที่ไม่มี daemon, ความปลอดภัย, cluster fabric และข้อจำกัดที่รู้ตัว

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
  - **download / start จากหน้าเว็บพร้อม progress สด** — สร้าง bundle เสร็จแล้วกดต่อได้ทันที
    ไม่ต้องกลับไป terminal · งานยาว (download หลายสิบ GB) รันเป็น subprocess แล้ว poll
    เอา output ล่าสุดมาแสดง · **หนึ่งโมเดลรันได้ทีละงาน** (download ซ้อน start = ไฟล์พัง)
    · ชื่อคำสั่งผ่าน allowlist ไม่ส่งต่อไปให้ shell ตรง ๆ
  - **ปุ่มเปลี่ยนตามสถานะจริง**: ยังไม่โหลด weight → ปุ่ม `download` + ป้าย "ยังไม่โหลดไฟล์"
    · โหลดครบแล้ว → `start` · ใช้ตัวตรวจชุดเดียวกับ `lmds doctor` ไม่คำนวณซ้ำคนละทาง
  - **`lmds web --background` / `--stop`** — เดิมหน้าเว็บกิน terminal ทั้งอัน ใช้ CLI ต่อไม่ได้เลย
    ต้องเลือกอย่างใดอย่างหนึ่ง (รายงานจากผู้ใช้จริง) · ตอนนี้รันคู่กันได้
  - **ตั้ง port / context / API key / bind ต่อโมเดลจากหน้าเว็บ** — เทียบเท่า
    `API_KEY=… ./<slug>-single.sh start --port … --context … --bind …` ของ CLI · ส่งผ่าน env
    ชุดเดียวกับที่ controller อ่านอยู่แล้ว · API key เก็บใน localStorage ของเบราว์เซอร์ ไม่ขึ้นเครื่อง
  - **`download` รัน `verify-files` ต่อให้อัตโนมัติ** — download อย่างเดียวไม่ตอบคำถาม
    "ไฟล์มาครบไหม" (CLI ให้รันต่อเสมอ) · ขั้นแรกล้ม = ไม่ทำขั้นถัดไป
  - **ปิดช่องว่างเทียบ CLI จนครบ** (รายงานจากผู้ใช้: "ถ้า feature ไม่ครบเหมือน CLI จะมี GUI ไปทำไม")
    — เพิ่มในแผง "จัดการ" ต่อโมเดล: `test-text`/`test-reasoning`/`test-tools`/`bench`/`stress`,
    `repair`, **`remove` พร้อมแสดงรายการ+ขนาดก่อนยืนยัน** (เลือกเก็บ weight ได้), autostart
    เปิด/ปิด, และคำสั่ง stacked ครบชุด (`prepare-runtime`/`sync-worker`/`verify-worker`/`clear-fi-cache`)
    · autostart ต้องใช้ sudo ซึ่งเว็บไม่มี tty — ล้มเหลวแล้วส่งคำสั่งกลับไปให้รันเอง ไม่ใช่ 500 เปล่า
  - **แยกปุ่ม "ทดสอบ" ออกจาก "จัดการ"** — สองเรื่องคนละจังหวะกัน (ตั้งค่าก่อน start / ทดสอบหลัง start)
  - **หน้าเว็บโชว์ข้อความจริงจาก controller ตอนงานล้ม** ไม่ใช่แค่ `exit 1` — เคสจริง:
    `client-config` ตายพร้อมเหตุผลที่ถูกต้อง (`context ต่อ slot เล็กเกิน (8000 = 32000/4)`)
    แต่เว็บกลืนข้อความไว้ ผู้ใช้เห็นแค่ "ล้มเหลว"
  - **ปรับ slots (`PARALLEL_SEQS`) ได้จากเว็บ** — เป็น knob ที่ข้อความ error ข้างบนบอกให้ปรับ
    แต่เดิมปรับจากเว็บไม่ได้
  - **UI เป็นภาษาอังกฤษทั้งหมด** (ตัว CLI ยังเป็นไทย) — หน้าเว็บเป็นสิ่งที่ส่งต่อให้ทีม/ลูกค้าดู
  - **ปุ่มขึ้นตามที่ controller ตัวนั้นรองรับจริง** — อ่าน dispatch table จากตัวสคริปต์เอง ไม่เดาจาก
    profile · เจอจริง: `test-vision` ไม่โผล่ให้ผู้ใช้เพราะ bundle สร้างก่อนมีคำสั่งนี้ · ตอนนี้บอกด้วยว่า
    "bundle นี้เก่ากว่า test-vision — deploy ใหม่เพื่อใช้งาน" แทนที่จะให้ปุ่มที่กดแล้วล้ม
  - ติดตั้ง: `install.sh` ลง `fastapi`/`uvicorn` ให้เอง (ล้มเหลวก็ไม่กระทบ CLI) · extra ชื่อ `web`
- **`test-vision` ใน controller ของโมเดล multimodal** — mmproj มาครบไม่ได้แปลว่า vision ทำงาน
  · สร้าง PNG สีแดงล้วนด้วย python stdlib (ไม่ต้องมีรูปในเครื่อง ไม่ต้องต่อเน็ต) แล้วถามโมเดลว่า
  เห็นสีอะไร · เดิมต้องเขียน curl + base64 เองซึ่งไม่มีใครทำ — ตอนนี้ `./<slug>-single.sh test-vision`
  หรือกดปุ่มในหน้าเว็บ · ขึ้นเฉพาะ bundle ที่มี mmproj จริง
- **`lmds list` / หน้าเว็บ เห็น bundle ทุกตัวบนดิสก์ ไม่ต้องรอ start ครั้งแรก** — เดิม controller
  เขียนทะเบียนเองตอน `start` เท่านั้น bundle ที่เพิ่งสร้างจึงหายไปจากรายการ ผู้ใช้ไม่รู้ว่าต้องไปต่อยังไง
  (รายงานจากผู้ใช้จริง: "สร้างเสร็จแล้วไม่มีหน้าอะไรให้กดต่อ")
  - `register_bundle()` ลงทะเบียนตั้งแต่ตอน generate ทั้ง CLI และเว็บ (ไม่เขียนทับทะเบียนของ
    controller ที่ start แล้ว)
  - **สแกนหา bundle ที่มีอยู่ก่อนแล้วด้วย** — ของที่สร้างไว้ก่อนหน้านี้ไม่มีทะเบียน ถ้าไม่สแกนก็ยัง
    ไม่โผล่อยู่ดี · หาใน `./bundles` ของโฟลเดอร์ปัจจุบัน, `~/bundles` และ `~/<โปรเจกต์>/bundles`
    รวมถึงเคส `bundles/bundles/` ที่เกิดจาก deploy ซ้อนโฟลเดอร์ · เพิ่มที่อื่นด้วย `LMDS_BUNDLE_DIRS`
- **ปุ่มบนหน้าเว็บไม่อัปเดตหลังงานเสร็จ ต้องกด F5 เอง** — คีย์เปรียบเทียบแถวไม่ได้นับสถานะ
  `downloaded`/งานที่รันอยู่ ปุ่มจึงค้างที่ `download` ทั้งที่โหลดเสร็จแล้ว (รายงานจากผู้ใช้จริง)
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

### Known gaps (fleet)

- hub ยังไม่เก็บ cluster IP **ของตัวเอง** ในทะเบียน (ตัวเองไม่ได้อยู่ในทะเบียน) — ใช้ค่าที่ตรวจพบตอนเขียน `cluster.env`
- `lmds deploy` ยังไม่ push bundle ไปติดตั้งบน node ให้ — node ต้องมี LMDS อยู่ก่อน
- กลุ่ม stacked > 2 เครื่อง แสดงผลได้ แต่ `--write` ต้องระบุ `--worker` เอง (template เป็น head+worker คู่เดียว)
- fabric detection มีเทสครอบด้วย sysfs จำลอง — **ยังไม่ได้ยืนยันกับ ConnectX บนเครื่องจริง**

### Known gaps

- **stacked ยังไม่เคยรันจริงจาก bundle ที่ LMDS สร้าง** — reference v8.2 เคย hardware-validated
  (2026-07-22) แต่นั่นคือสคริปต์เขียนมือ · ตัวที่ generate ยังเป็น `static-validated` เท่านั้น
- **`runtime_assets` ยังไม่รองรับใน stacked** — ต้อง sync ไฟล์ plugin ไป worker และ mount ทั้งสอง node
  ทำครึ่งทางจะแย่กว่าไม่ทำ (head มี plugin แต่ worker ไม่มี = พังตอน start แบบไล่ยาก)
- **`wait-health` ยังไม่มีใน stacked** — ฝั่ง stacked มี `STARTUP_TIMEOUT` ยาวกว่าอยู่แล้ว ความจำเป็นน้อยกว่า

### Added (รอบย่อย)

- **`lmds smoke`** — พิสูจน์ว่า bundle รันได้จริง ไม่ใช่แค่สคริปต์ถูก:
  `download → verify-files → start → test-text → stop` · ล้มขั้นไหนหยุดตรงนั้น ·
  **หยุด server เสมอแม้ล้มกลางทาง** · `--on <เครื่อง>` รันข้ามเครื่องได้
- **`lmds rebuild`** — สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน โดยเก็บค่าที่เคยตัดสินใจไว้
  (context/flags/target/GGUF) จาก `MODEL_PROFILE.yaml` และไม่เรียก LLM ซ้ำ ·
  ใช้ตอน image ที่ tag ถูกถอน หรือ template ใหม่มีตัวกันพลาดที่ของเก่าไม่มี
- **ย่อการ์ดเครื่องแล้วยังเห็น CPU / RAM / VRAM / GPU** — ชิปสรุปในหัวการ์ด ใช้เกณฑ์สี
  เดียวกับเกจข้างใน · อัปเดตแม้ตัวการ์ดถูกล็อกอยู่
- **ส่ง bundle ไปรันบนเครื่องอื่นได้** — `lmds node push <เครื่อง> <slug>` และปุ่ม
  *Run on another machine* ในแท็บ manage · wizard สร้าง bundle ลงเครื่องที่เปิดหน้าเว็บเสมอ
  ซึ่งบน controller ที่ไม่มี GPU แทบไม่มีประโยชน์ · ส่ง ZIP แทนสั่งให้ปลายทาง deploy เอง
  เพราะแผนที่ผู้ใช้อนุมัติต้องเป็นตัวเดียวกับที่ไปรัน
- **โมเดลบนเครื่องอื่นได้ชุดควบคุมเท่ากับโมเดลในเครื่อง** — เดิมตั้งได้แค่ port/context/gpu-util
  ส่วนในเครื่องตั้ง slots/bind/API key ได้ และมีชุดทดสอบครบ ทั้งที่เป็น controller ตัวเดียวกัน
  · node เปลี่ยนมาส่งค่าเป็น **env** เหมือน local (`jobs.controller_env`) และตรวจค่าด้วย
  `jobs.clean_options()` ตัวเดียวกันทั้งสองทาง · เพิ่ม endpoint สั่งคำสั่งของ controller
  ข้ามเครื่อง → ชุดทดสอบ + กลุ่ม stacked ครบ
- **cluster IP อยู่ในการ์ดของเครื่องนั้น + รั้วสีจับคู่ที่ stacked ได้** — เดิม cluster IP ของทุกเครื่อง
  กองรวมกันเป็นตารางท้ายหน้า อ่านแล้วไม่รู้ว่าแถวไหนของเครื่องไหนถ้าไม่ไล่อ่านชื่อทีละบรรทัด
  (ผู้ใช้จริงรายงาน) · ตอนนี้แถบ fabric / cluster IP / ชื่อคู่อยู่ใต้ชื่อเครื่องนั้นเลย · เครื่องที่จับคู่
  stacked ได้ถูกจัดให้อยู่ติดกันแล้วคร่อมด้วยรั้วสี + ป้าย `CLUSTER A/B` สีเดียวกัน · **ลบตารางเดิมทิ้ง**
  เพราะข้อมูลเดียวกันอยู่สองที่คือต้นเหตุความสับสน · ปุ่ม *Check cluster* ย้ายมาอยู่หัวข้อ *Other machines*
- **ระบบสีของทั้งหน้า** — ตระกูลสีเดียวไล่เฉด (ฟ้า → คราม → ม่วง → ชมพู) แทนพื้นดำแบน:
  พื้นหลังมีแสงจางสองมุม · การ์ด GPU และ *กี่ตัวรันอยู่* เป็นการ์ดไล่เฉดเต็มใบ · เกจทุกตัวใช้ stroke
  ไล่เฉด (นิยาม `<linearGradient>` ครั้งเดียวทั้งหน้า) · หัวข้อมีขีดสีนำ · ปุ่มหลักเป็นปุ่มไล่เฉด
  · **ความหมายของสียังเหมือนเดิมเป๊ะ** — เขียว/เหลือง/แดง = ปกติ/ใกล้เต็ม/เต็ม ไม่ได้ถูกกลืนไปกับ
  สีตกแต่ง · เพิ่มเทสว่า `url(#id)` ทุกตัวที่เกจอ้างถึงมี gradient รองรับจริง (ไม่งั้นเส้นเกจหายเงียบ ๆ)
- **ช่องกรอกแยกจากพื้นหลังเสมอ** — ผู้ใช้รายงานว่าพื้นหลังไปเสมอกับช่องกรอกจน "ดูพลาดได้"
  ซึ่งเป็นเรื่องกรอกผิดช่อง ไม่ใช่เรื่องความสวย · ลดแสงพื้นหลังลงกว่าครึ่ง **และ** ให้ช่องกรอกใช้
  ผิวจมลงไป + เงาใน + ขอบเข้มกว่าการ์ด + เปลี่ยนสีขอบตอนโฟกัส — แก้สองทางพร้อมกัน
- **หน้า login ของ GUI + ตั้ง token เองได้** — เดิม token อยู่ใน URL ซึ่งไปโผล่ใน history
  ของเบราว์เซอร์, log ของ proxy และ referrer · ตอนนี้ลิงก์ล้วน ไม่มี token และหน้าเว็บมีช่อง
  ให้กรอกก่อนวาดอะไร (เดิมโหลดโครงหน้าขึ้นมาก่อนแล้วค่อยพังตอนเรียก API คนที่ไม่มีสิทธิ์
  จึงเห็นชื่อเครื่อง) · ผ่านแล้วเบราว์เซอร์จำให้ มีปุ่ม Sign out
  · ที่มาของ token: `--token` → `$LMDS_WEB_TOKEN` → ที่จำไว้ → **ถามตอนสตาร์ตครั้งแรก**
  (Enter = สุ่มให้ · กรอกเองได้ ≥ 8 ตัว ไม่จำกัดชนิดตัวอักษร) → สุ่มให้เมื่อไม่มี tty
  · `?token=` ยังใช้ได้เพื่อความเข้ากันได้ แต่ถูกลบออกจากแถบที่อยู่ทันทีที่เปิด
- **กันเดา token** — ผิดติดกันเกิน 5 ครั้งจาก IP เดียวกันเริ่มหน่วงแบบทวีคูณ (สูงสุด 60 วินาที)
  คนพิมพ์ผิดไม่โดนลงโทษ บอตยิงรัวไม่คุ้ม · ล็อกอินผ่านล้างตัวนับ
- **`remove` สั่งจากหน้าเว็บได้แล้ว แบบมีขั้นยืนยันที่มีความหมาย** — กดครั้งแรก = `--dry-run`
  เห็นรายการ+ขนาดจริง แล้วถึงมีปุ่ม "ยืนยันลบถาวร" · server ต้องได้ค่ายืนยันที่ตรงกับ slug เป๊ะ
  (`yes`/`true` ไม่ผ่าน) · เพิ่ม `lmds remove --dry-run` ให้ใช้จาก CLI ได้ด้วย
- **กราฟ CPU / หน่วยความจำ / VRAM / ดิสก์** — เดิมเครื่องอื่นแสดงเป็นชิปข้อความ ส่วนเครื่องนี้
  เป็นแถบ คนละภาษาสองแบบในหน้าเดียว ทั้งที่การ์ด GPU ใต้มันเป็นเกจอยู่แล้ว · ตอนนี้เป็นเกจชุดเดียวกัน
  ทั้งสองฝั่ง พร้อมตัวเลขจริงใต้ทุกเกจ · สีเตือนก่อนของจะหมด (หน่วยความจำ ≥75%/≥90% ·
  ดิสก์กลับด้าน เหลือ ≤15%/≤5%) · unified ไม่มีเกจ VRAM แยกเพราะนับซ้ำกับ RAM
- **ลิงก์หน้าเว็บอยู่ยาว bookmark ได้** — token ถูกจำไว้ที่ `~/.config/lmds/web-token` (0600)
  และใช้ซ้ำทุกครั้งที่เปิด · restart/stop แล้วเปิดใหม่ก็ยังเป็นลิงก์เดิม เปลี่ยนเมื่อสั่ง `--new-token`
  หรือ `--token` เท่านั้น · เดิมสุ่มใหม่ทุกครั้ง ผู้ใช้ต้องกลับไปหา terminal ทุกรอบ
- **ตั้ง port / context / gpu-util ตอนสั่งรันโมเดลบนเครื่องอื่นได้จากหน้าเว็บ** — โมเดลในเครื่องมี
  แท็บ manage อยู่แล้ว แต่โมเดลบนเครื่องอื่นไม่มีเลย ต้อง ssh ไปแก้ `.sh` เอง ซึ่งขัดกับเหตุผล
  ที่มีหน้าเว็บตั้งแต่แรก · gpu-util ขึ้นเฉพาะ engine ที่รองรับจริง · ค่าจำไว้ในเบราว์เซอร์
  ไม่เขียนทับ bundle ปลายทาง · **server ตรวจค่าเอง** (ช่วงเดียวกับที่ controller ตรวจ) เพราะค่า
  พวกนี้ถูกต่อเป็นคำสั่งที่รันผ่าน SSH · ส่ง option ไปกับคำสั่งที่ไม่รับมัน = 400 ไม่ใช่เงียบ ๆ ทิ้ง
- **เมนูคำสั่งต่อโมเดลบนเครื่องอื่น** — เดิมโมเดลบนเครื่องอื่นมีแค่ `start`/`doctor` ทั้งที่ CLI ทำได้
  มากกว่านั้น · เพิ่มปุ่ม **⋯** ต่อโมเดล กาง `restart` · `doctor` · `logs` · `repair` ·
  `enable`/`disable` — ปุ่มขึ้นตามสถานะจริงของโมเดลนั้น · allowlist ฝั่ง server เพิ่ม
  `logs` (บังคับ `-n 300`) `enable` `disable` · **`remove` จงใจไม่อยู่ในเมนู** เพราะต้องใช้ `-y`
  ซึ่งข้ามหน้ายืนยันที่แสดงรายการ+ขนาดก่อนลบ weight หลายสิบ GB (มีเทสกันไม่ให้หลุดเข้ามา)
- **gate ตรวจ syntax ของ JS ในหน้าเว็บ** — JS พังตัวเดียว = หน้าขาวทั้งหน้าโดยที่เทส API
  ยังเขียวหมด · `node --check` ทุก `<script>` (ข้ามถ้าเครื่องไม่มี node)

### Fixed

รอบย่อยนี้มาจากการใช้งานจริงบน controller (Ubuntu VM) + dgx-msi — ทุกข้อเป็นเคสที่ผู้ใช้ทำตาม
คำแนะนำของตัวโปรแกรมเองแล้วเจอทางตัน

- **ชุดทดสอบให้คะแนนเซิร์ฟเวอร์ของโมเดลอื่น** — ทุก bundle ตั้งต้นที่พอร์ต 8000 เหมือนกัน ·
  `test-text` ของ gemma-4-31b ได้คำตอบกลับมาพร้อม `"model":"Qwen3-Coder-30B-A3B-Instruct"`
  แล้วรายงานว่า "ผ่าน" ทั้งที่ทดสอบคนละโมเดล (ตัวที่รันอยู่ก่อนบนพอร์ตนั้น)
  · controller ทุก template มี `assert_our_server()` อ่าน `/v1/models` ก่อนยิงทดสอบ ชื่อไม่ตรง
  = หยุดพร้อมบอกว่าใครยึดพอร์ตอยู่ · อ่านไม่ได้ = ไม่ฟันธง ปล่อยผ่าน (ไม่สร้าง false alarm)
  · bundle ที่สร้างไปแล้วไม่มีตัวตรวจนี้ หน้าเว็บจึงติดป้ายแดง `พอร์ตชนกับ <ชื่อ>` ให้แทน
- **`stop_refresher()` บอกว่าหยุดแล้วทั้งที่ยังไม่หยุด** — ตั้ง event แล้วคืนทันที thread จึงเขียน
  ผลลง cache หลังผู้เรียกคิดว่าหยุดไปแล้ว · ตอนนี้รอ thread จบจริง
- **ป้าย autostart แสดงสถานะไม่จริงกับทุกโมเดล** — `autostart` เป็นสตริง
  `enabled|disabled|absent|n/a` ซึ่ง**ทุกตัวเป็น truthy ใน JS** · หน้าเว็บเช็ก `m.autostart ? …`
  จึงติดป้าย "autostart" ให้ทุกโมเดลและเสนอปุ่ม `disable` ทั้งที่ไม่เคยเปิดเลย
  · เทียบ `=== "enabled"` ตรง ๆ และเครื่องที่ไม่มี systemd (`n/a`) ไม่มีปุ่มให้กด
- **`disable` รายงานว่าสำเร็จทั้งที่ไม่ได้ทำ** — `disable_autostart()` กลืน error ของ `sudo`
  ทุกตัวแล้วคืนค่าสำเร็จเสมอ · sudo ที่ขอรหัสผ่านไม่ได้ (ถูกเรียกผ่าน SSH จาก hub) จึงกลายเป็น
  "ปิด autostart แล้ว" ทั้งที่ยังเปิดอยู่ — ผู้ใช้จะรู้ตัวอีกทีตอน reboot แล้วโมเดลเด้งขึ้นมาเอง
  · ตอนนี้ตรวจสถานะจริงหลังรัน และบอกตั้งแต่แรกถ้าไม่มี unit ให้ปิด
- **`lmds restart <slug> --port 8001` รับ flag แล้วทิ้งเงียบ** — การแก้ passthrough รอบก่อน
  ไปโดนแต่ `start` (ตัวแปรคนละชื่อ) · ผู้ใช้เห็น "restarted" ทั้งที่ port ไม่เปลี่ยน ซึ่งแย่กว่า error
  เพราะไม่มีอะไรบอกว่าพลาด
- **`lmds logs` ของโมเดลที่ยังไม่เคยรัน โยน `tail: cannot open …` ดิบ ๆ** — อ่านเหมือนระบบพัง
  ทั้งที่แค่ยังไม่เคยสตาร์ต · บอกตรง ๆ พร้อมคำสั่งที่ต้องทำก่อน
- **`lmds remove` ผ่าน SSH ตอบแค่ `Aborted.`** — ไม่มี terminal ให้ยืนยัน ผู้ใช้จึงเห็นเหมือน
  คำสั่งทำงานแล้วไม่มีอะไรเกิดขึ้น · ตอนนี้ exit 2 พร้อมบอกว่าต้องใช้ `--dry-run` ดูก่อนแล้วเติม `-y`
- **โมเดลที่โหลดครบแล้วขึ้น "not downloaded" ตลอดกาล ปุ่ม start ไม่ขึ้น** — doctor นับ
  `projector_files` (mmproj) เป็นไฟล์บังคับ **และบังคับครบทุก precision** (BF16+F16+F32) ทั้งที่
  `llama-server` รับ `--mmproj` ได้ไฟล์เดียว และ controller ก็ไม่ได้โหลด mmproj มาด้วยซ้ำ
  → เรียกร้องไฟล์ที่ไม่มีใครจะโหลดให้ (gemma-4-31b-it-gguf บน dgx-veerasiam โหลดครบ 35 GB แล้ว)
  · แก้: แยกไฟล์ที่ขาดไม่ได้ (weight หลัก = FAIL) ออกจากไฟล์ทางเลือก (mmproj ขาด = **WARN**
  "โมเดลจะรับแต่ข้อความ" ไม่บล็อก) · มี precision ใดตัวหนึ่งก็พอ · `weights_present()` ใช้ตัวตรวจ
  ชุดเดียวกัน ป้าย "not downloaded" บนหน้าเว็บจึงหายตามไปด้วย
- **หน้าเว็บวาดทับตอนกำลังพิมพ์ จนกรอก port/context ไม่ทัน** — SSE ส่ง snapshot ทุก ~1 วิ แล้ว
  เขียนทับ body ของแถวเครื่องทั้งก้อน ตัวเลขที่พิมพ์ไปแล้วจึงหายทุกวินาที · หยุดวาดทับแถวที่
  **กางเมนูค้างอยู่ หรือมีเคอร์เซอร์อยู่ในช่องกรอก** จนกว่าผู้ใช้จะปิดเอง — และขึ้น `paused · Ns ago`
  บอกว่าตัวเลขที่เห็นเริ่มเก่า แทนที่จะเงียบไปเฉย ๆ
- **`lmds web --restart` ฟ้อง "พอร์ตไม่ว่าง" จากตัวที่ตัวเองเพิ่งฆ่า** — SIGTERM ไม่ได้คืน socket
  ทันที · รอให้ว่างจริงก่อนสตาร์ตใหม่
- **หน้าเว็บ "ใช้ได้บ้างไม่ได้บ้าง" แล้วสั่งรันใหม่ก็ error** — `lmds web -b` รอบสองขึ้นไป uvicorn
  bind ไม่ได้ (`address already in use`) แล้วตายใน 0.2 วินาที ซึ่ง `Popen` มองว่าสำเร็จ · CLI จึงเขียน
  PID ของศพทับ `web.pid` แล้วพิมพ์ token **ใหม่** ให้ผู้ใช้ ทั้งที่ตัวที่เสิร์ฟจริงเป็นตัวเก่าซึ่งถือ
  token คนละตัว → เปิดลิงก์ที่เพิ่ง copy มาแล้วเจอ `A token is required` · และ `lmds web --stop`
  ฆ่า PID ที่ตายไปแล้ว รายงานว่าสำเร็จ ทั้งที่ของจริงยังรันอยู่
  · แก้: จำ pid/port/bind/token ไว้ที่ `~/.lmds/run/web.json` (0600) — รันซ้ำจะพิมพ์ลิงก์ของ
  *ตัวที่เสิร์ฟจริง* แทนการสตาร์ตซ้อน · `-b` รอจนรับ connection ได้จริงก่อนบอกว่าสำเร็จ ตายก็พิมพ์
  ท้าย log ให้ · `--stop` ตรวจ cmdline ก่อนฆ่าเผื่อ PID ถูกใช้ซ้ำ · เพิ่ม `--status` / `--restart`
  · หน้า 401 เดิมบอก "open the link that lmds web printed" ซึ่งผู้ใช้ทำอยู่แล้ว — เปลี่ยนเป็นบอกว่า
  token ไม่ตรงกับตัวที่รันอยู่ แล้วชี้ไป `lmds web --status`

- **`lmds start <slug> --port 8001` ตอบ `No such option: --port`** — ทั้งที่ controller ของ bundle
  รองรับ flag นี้อยู่แล้ว · `start`/`restart` เปิด `ignore_unknown_options` แล้วส่ง flag ที่ไม่ใช่ของ
  `lmds` ต่อให้ controller ตรง ๆ — LMDS ไม่พยายามรู้จัก flag ทุกตัวเพราะแต่ละ engine มีไม่เท่ากัน
  และเปลี่ยนตามเวอร์ชัน · คำแนะนำใน `lmds ps` / `lmds list` แสดงตัวอย่างการส่ง flag ด้วยแล้ว
- **doctor บอกว่ายังไม่ได้ download ทั้งที่ weight อยู่ครบ** — `_weight_paths()` ดูแต่ `$HF_HOME/hub/`
  ส่วน weight ที่โหลดด้วย HF รุ่นเก่าอยู่ที่ `$HF_HOME/models--X` (เจอกับ DeepSeek V4 บน head node)
  · ตรวจทั้งสองเลย์เอาต์ และถ้ามี snapshot ใดก็ตามอยู่จริงก็ถือว่ามีแล้ว
- **กด doctor บนหน้าเว็บแล้วไม่มีอะไรเกิดขึ้น** — SSE ที่เพิ่งเพิ่มเข้ามา: snapshot รอบถัดไป (ทุก ~1 วิ)
  เขียนทับเนื้อในการ์ด node ทับผลลัพธ์ที่เพิ่งได้มา · ผลลัพธ์ที่ผู้ใช้สั่งเองถูก "ปัก" ไว้
  (`pinnedOutput`) จนกว่าจะกดปิด — snapshot อัปเดตแค่ส่วนที่ผู้ใช้ไม่ได้กำลังอ่านอยู่

- **เอกสารบอกคำสั่งที่ใช้ไม่ได้** — runbook เขียน `lmds node run <n> run-controller …` พร้อมหมายเหตุ
  ว่า "ยังไม่มีใน LMDS" · เขียนใหม่ทั้งฉบับให้ตรงกับคำสั่งจริง + เพิ่มเทสที่ตรวจว่าทุกคำสั่งใน
  หน้าเว็บมีอยู่จริงใน CLI (หน้าเว็บเคยแนะนำ `lmds deploy --topology stacked` ที่ไม่มีอยู่)
- **เทสเขียนทะเบียนลง `~/.lmds/run` ของเครื่องจริง** — `conftest` แยก `LMDS_CONFIG_DIR` แต่ลืม
  `LMDS_RUN_ROOT` · เทสทิ้งรายการค้างไว้ให้ผู้ใช้เห็นในหน้าเว็บ

- **stacked หา weight ไม่เจอเมื่อ HF cache เป็นเลย์เอาต์เก่า** — `$HF_HOME/models--X` (ไม่มี `hub/`)
  · controller ตรวจเจอทั้งสองแบบอยู่แล้ว แต่ vLLM ในคอนเทนเนอร์มองแค่ `hub/` จึงตาย
  `LocalEntryNotFoundError` ทั้งที่ `verify-files` เพิ่งบอกว่า 46 shards ครบ · ตั้ง `HF_HUB_CACHE`
  ให้ตรงกับของจริงทั้ง head และ worker — **ไม่ต้องย้ายไฟล์ของผู้ใช้**
- **DeepSeek V4 ตายตอน load_model** — สถาปัตยกรรมนี้ใช้ attention layout `fp8_ds_mla`
  ที่บังคับ kv-cache เป็น fp8 (`AssertionError: only supports fp8 kv-cache, got auto`)
  · เพิ่มตาราง `ARCH_REQUIREMENTS` ใน rule-based สำหรับข้อบังคับระดับสถาปัตยกรรมที่ไม่มี LLM ไปค้นให้
- **ล็อก runtime image ใช้ร่วมกันทั้งเครื่อง** (`$HF_HOME/.lmds-stacked-image-id`) — เครื่องเดียว
  รัน stacked หลายตัวที่ใช้คนละ image ไม่ได้ ตัวที่สองตายด้วย "image ต่างจากที่ lock ไว้"
  · แยกเป็นล็อกต่อ bundle
- **`lmds node run` กลืน flag ของคำสั่งปลายทาง** — `node run x logs y -n 100` ตอบ `No such option: -n`
- **Jinja หลุดเข้าไฟล์ผลลัพธ์ของ stacked controller** — `{% if report.gated %}` และ
  `{% if shard_files %}` ถูกวางไว้ใน `{% raw %}` จึงไม่เคยถูกแปลง สคริปต์ที่ส่งให้ผู้ใช้จึงมีบรรทัด
  `{%: command not found` และ **ไม่มีการตรวจ shard รายไฟล์เลย** · `bash -n` จับไม่ได้เพราะเป็น
  bash ที่ syntax ถูกต้อง — เจอตอนรัน `verify-files` บน DGX Spark จริง
- **gate ใหม่ `template-rendered`** — ปฏิเสธ bundle ที่มี Jinja tag เหลืออยู่ (กันคลาสบั๊กนี้ทั้งคลาส
  ไม่ใช่แค่เคสนี้) · quality gates เพิ่มเป็น 10 ด่าน
- **`install.sh` ล้มเมื่อรันซ้ำเพื่ออัปเดต** (`ensurepip ... returned non-zero exit status 1`)
  — venv เดิมอาจถูกสร้างด้วย python คนละตัว (เช่นเครื่องที่มี conda) การรัน `python3 -m venv`
  ทับของเดิมจึงพัง · ใช้ `--clear` เสมอและตายพร้อมบอกวิธีแก้ถ้าสร้างไม่ได้
  — พบตอนอัปเดต DGX Spark จริงสองเครื่อง ซึ่งเป็นทางอัปเดตที่เอกสารบอกไว้เอง

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

- 🎉 **DeepSeek-V4-Flash-NVFP4 ผ่านบน DGX Spark 2 เครื่องผ่าน LMDS** — โมเดล MoE 157 GB (46 shards)
  · **ต้องถูกพร้อมกัน 6 อย่างถึงจะ start ขึ้น** และไม่มีอันไหนบอกสาเหตุตรง ๆ: image ที่มี kernel
  ของ DeepSeek V4 · `HF_HUB_CACHE` ตรง cache layout · `kv-cache-dtype nvfp4_ds_mla` ·
  **`cudagraph_mode PIECEWISE`** (ตัวสุดท้ายที่ติด — `Expected 7 but got 8 arguments`) ·
  ล็อก image ต่อ bundle · `clear-fi-cache` ที่ลบไฟล์ของ root ได้
  · ทั้งหมดอยู่ในสูตรแล้ว — ครั้งต่อไปคำสั่งเดียวจบ ไม่ต้องมี API key

- 🎉 **stacked (multi-node) ผ่านบนฮาร์ดแวร์จริงเป็นครั้งแรก** — `meta-llama/Llama-3.3-70B-Instruct`
  บน **DGX Spark 2 เครื่อง** (gigabyte01 + gigabyte02) ผ่าน LMDS ตั้งแต่ `deploy` จนถึง `test-text`:
  `prepare-runtime → verify-files (30 shards + ขนาดตรงกับ Hub) → sync-worker → verify-worker →
  start → test-text` · vLLM 26.05 (NGC), TP=2 nnodes=2, **`mp` backend ไม่ใช้ Ray**,
  NCCL ผ่าน RoCE `enp1s0f1np1`/`rocep1s0f1` ที่ระบบหาให้เอง, context 65,536, โหลด 8 นาที,
  `/health` ผ่าน, `test-text` ตอบถูก
  · **คำตอบของคำถามค้างเรื่อง Ray**: vLLM native multi-node (`--nnodes/--node-rank/--headless`)
  ใช้แทน Ray cluster ได้จริงบน DGX Spark — ชิ้นส่วนน้อยกว่า ไม่ต้องมี Ray/tmux/run_cluster.sh
  · การรันจริงครั้งนี้เจอบั๊ก 3 ตัวที่ static gate จับไม่ได้ (head ไม่เคย start, Jinja หลุด,
  `node run` กลืน flag) — ดูหัวข้อ Fixed

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
