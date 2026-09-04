# สิ่งที่ระบบตรวจให้ก่อน deploy — และทำไมถึงต้องตรวจ

ทุกหัวข้อในนี้มาจากของที่พังจริงบนเครื่องจริง ไม่ใช่รายการที่คิดขึ้นเผื่อไว้
แต่ละอันบอกว่า **อาการเป็นยังไง**, **ต้นเหตุคืออะไร**, และ **ตอนนี้อะไรจับมันได้**

สิ่งที่เหมือนกันทุกเคส: ระบบ *มีข้อมูลที่ถูกต้องอยู่แล้ว* แต่ข้อมูลนั้นไม่เดินทาง
ไปถึงปลายทาง และไม่มีอะไรตรวจผลลัพธ์ — จึงพังเงียบ ไม่มี error ให้เห็น

---

## รันไทม์ไม่รู้จักสถาปัตยกรรมของโมเดล

**อาการ** — โหลด weight ครบ 30 GB แล้วค่อยพังตอน `start`

**เคสจริง (2026-08-13)** — Muse-Glimmer-30B ใช้ architecture `muse-glimmer`
llama.cpp บน spark-head อยู่ที่ commit ของ 23 ก.ค. ตามหลัง upstream 296 commit
ส่วนที่รองรับ `muse-glimmer` เพิ่ง merge ที่ `62bf73d25`

เช็คเรื่องนี้มีอยู่ก่อนแล้ว แต่ขึ้นต้นด้วย `if server.mode == "native": return []`
ซึ่ง llama.cpp บน DGX Spark รัน native เป็นปกติ — เช็คจึงไม่เคยทำงานกับ engine
ที่ต้องการมันมากที่สุด

**ตอนนี้** — `doctor` อ่าน `general.architecture` จากหัวไฟล์ GGUF ในเครื่อง แล้ว
หาชื่อนั้นใน `libllama.so` ของ build ที่โมเดลตัวนั้น pin ไว้ ไม่รู้จักก็ FAIL
พร้อมบอกให้ build ใหม่ ฝั่ง vLLM ยังตรวจผ่าน image เหมือนเดิม

---

## รันไทม์ผูกกับเครื่อง ไม่ได้ผูกกับโมเดล

**อาการ** — โมเดลใหม่ต้องการ llama.cpp รุ่นใหม่กว่า แต่การอัปเกรดไปกระทบทุกโมเดล
บนเครื่องนั้นที่พิสูจน์แล้วว่าใช้ได้

**ต้นเหตุ** — bundle ฝั่ง vLLM pin image ด้วย digest มานานแล้ว แต่ฝั่ง llama.cpp
native ชี้ `~/src/llama.cpp` ร่วมกันหมดทุกตัว

**ตอนนี้** — `RuntimeChoice.native_dir` เดินทางถึง controller โมเดลสองตัวบนเครื่อง
เดียวจึงใช้ llama.cpp คนละ build ได้ ไม่ pin = ใช้ของกลางตามเดิม (รุ่นเดียวกัน
ใช้ร่วมกันได้ ไม่ต้อง build ซ้ำ)

```bash
# ดูว่าโมเดลตัวนี้ผูกกับ build ไหน
grep LLAMA_CPP_DIR ~/bundles/<slug>/<slug>-single.sh
```

---

## เลือก projector ผิดตัว → vision ผิดแบบเงียบ ๆ

**อาการ** — โมเดลตอบเรื่องภาพได้ แต่ตอบผิด ไม่มี error

**เคสจริง** — `unsloth/Muse-Glimmer-30B-GGUF` มี projector สามไฟล์

| ไฟล์ | คู่กับ |
|---|---|
| `mmproj-Muse-Glimmer-30B-BF16.gguf` | weight ปกติ |
| `mmproj-Muse-Glimmer-30B-Q8_0.gguf` | weight ปกติ |
| `mmproj-kquant.gguf` | `dflash-kquant.gguf` — **คนละโมเดล** |

กติกาเดิมคือ "เลือกเล็กสุด" ซึ่งถูกเมื่อ repo มี projector ตัวเดียวหลายระดับ quant
แต่ repo นี้เล็กสุดคือ `mmproj-kquant` ที่ไม่ได้คู่กับ weight ที่เลือกไว้เลย

**ตอนนี้** — เลือกตัวที่ชื่อร่วมตระกูลกับ weight ก่อน แล้วค่อยเล็กสุดในกลุ่มนั้น
ไม่มีตัวไหนเข้าเกณฑ์จึงกลับไปใช้เล็กสุดตามเดิม

---

## context ถูกตัดโดยไม่มีใครรู้

**อาการ** — โมเดลรองรับ 262,144 แต่รันจริงได้ 16,384 หรือ 65,536

มาจากสามที่คนละจุด:

1. **cap ที่ตั้งเลขเอาเอง** — `DEFAULT_CONTEXT_CAP = 65536` คร่อมทุก recommendation
   ไม่ว่าเครื่องจะไหวแค่ไหน โค้ดเดิมถึงกับเขียน note เตือนตัวเองว่า
   "หน่วยความจำรองรับได้ถึง 262,144" แล้วก็ยังส่ง 65,536 ให้อยู่ดี
   → ตอนนี้แนะนำ `safe` ตรง ๆ = `min(หน่วยความจำที่เหลือ, native context)` หารด้วย
   concurrency แล้ว จึงเป็นค่าที่ทั้งเครื่องและโมเดลรับไหวโดยนิยาม

2. **KV dims อ่านไม่ออก** — โมเดล sliding-window (gemma-4) เขียน `head_count_kv`
   เป็นลิสต์ต่อ layer parser เช็ค `isinstance(int)` แล้วคืน None → analyser เข้า
   สาขา "ไม่รู้มิติ KV" ที่ตั้ง context ไว้ 16,384
   → ตอนนี้นับเฉพาะ layer full-attention (ที่ KV โตตาม context จริง) gemma-4 ได้
   81,920 B/token ตรงกับ 18.9 GiB ที่เครื่องจองเพิ่มจริงตอนขึ้นจาก 16,384 → 262,144

3. **multimodal ซ่อน context ไว้ชั้นใน** — `max_position_embeddings` อยู่ใต้
   `text_config` ไม่ใช่ระดับบนสุด
   → inspector มองทั้งสองชั้นแล้ว

**ข้อควรรู้เรื่อง llama.cpp** — `--ctx-size` คือ *pool รวม* ที่หารด้วย `--parallel`
ตั้ง `--ctx-size 262144 --parallel 4` แปลว่าแต่ละ request ได้ **65,536** ไม่ใช่ 262,144

เรื่องนี้เคยหารซ้ำสองรอบ: fit คำนวณที่ `concurrency=1` ค่าที่ได้จึงเป็นของ slot เดียว
อยู่แล้ว แต่ `max_num_seqs` default เป็น 4 พอเอาไปใส่เป็น pool ก็ถูกหารอีกครั้ง

เคสจริง 2026-08-13 — Muse-Glimmer แผน/README/banner บอก 131,072 แต่ `/props` รายงาน
**32,768** ตอนนี้ llama.cpp จึง default เป็น **1 slot** (ต่างจาก vLLM ที่แชร์ KV แบบ
dynamic จึงไม่มีปัญหานี้) ใครต้องการ concurrency ตั้ง `PARALLEL_SEQS` เองได้ แล้วแผน
จะเตือนว่าแต่ละ request จะเหลือเท่าไร

```bash
# ดูค่าที่ผู้ใช้ได้จริง ไม่ใช่ค่าที่ตั้งไว้
curl -s http://<host>:8000/props | jq '.default_generation_settings.n_ctx, .total_slots'
```

---

## start ทับ port ของโมเดลอื่นแล้วบอกว่าสำเร็จ

**อาการ** — controller บอก `started` แต่โมเดลที่ตอบคือตัวอื่น

**ต้นเหตุ** — `wait_health` ยิง `/health` ที่ port ของตัวเอง โมเดลที่ยึด port อยู่
ตอบ 200 ให้ → นับว่า start สำเร็จ ทั้งที่ตัวเราไม่ได้ bind เลย

**ตอนนี้** — ระบบ **ไม่เลือก port ให้** (เครื่องเดียวรัน llama.cpp หลายตัวได้ ผู้ใช้
เป็นคนกำหนด) แต่ปฏิเสธการ start ทับ พร้อมบอกว่าใครถืออยู่

```
ERROR: port 8000 ถูกใช้อยู่แล้วโดย users:(("llama-server",pid=18509,fd=32))
     เลือก port อื่น:  ./<controller> start --port <PORT>
```

---

## container ไม่มี uid ที่ `--user` ส่งไป

**อาการ** — `KeyError: 'getpwuid(): uid not found: 1000'` ตอน start

**ต้นเหตุ** — `--user $(id -u)` กับ image ที่ไม่มี uid นั้นใน `/etc/passwd`
`getpass.getuser()` อ่าน `LOGNAME/USER/LNAME/USERNAME` ก่อนค่อยไป `pwd.getpwuid()`

**ตอนนี้** — controller ส่ง `-e USER=lmds` ไปด้วย

หมายเหตุ: อันนี้เป็นคนละคลาสกับที่เหลือ — เป็น *ความไม่เข้ากันของสภาพแวดล้อมตอนรัน*
ที่ preflight ด้านบนไม่มีตัวไหนจับได้ ต้องเจอตอนรันเท่านั้น

---

## controller ผ่าน `bash -n` แต่รันไม่ได้

**อาการ** — bundle ผ่าน quality gate ทุกด่าน แต่ผู้ใช้กด `start` แล้วตาย

```
muse-glimmer-30b-gguf-single.sh: line 344: $4: unbound variable
```

**ต้นเหตุ** — โค้ดที่ generate ออกมามี `$4` ที่ไม่ได้ถูก quote (ตั้งใจจะให้เป็น field
ของ awk แต่ escape single quote หลุด) bash จึงมองเป็น positional parameter แล้ว
`set -u` ก็ตาย — **syntax ถูกต้องทุกประการ `bash -n` จึงผ่านฉลุย**

**ตอนนี้** — เทสดึงฟังก์ชันออกมา *รันจริง* ภายใต้ `set -euo pipefail` ทั้งกรณี port
ว่างและ port ที่มีคนถืออยู่ ไม่ใช่แค่ตรวจ syntax
(`tests/test_controller_runs_not_just_parses.py`)

บทเรียนกว้างกว่านั้น: gate ที่ตรวจแค่ syntax ไม่รู้จักบั๊กตอนรัน อะไรที่ผู้ใช้จะรัน
ต้องมีเทสที่รันมันจริงอย่างน้อยหนึ่งเส้นทาง

---

## reasoning model ตอบว่างเปล่าเมื่อ `max_tokens` น้อย

**อาการ** — โมเดลคืน `content` ว่าง ดูเหมือนพัง ทั้งที่ปกติดี

**เคสจริง** — Muse-Glimmer ปล่อย `[thinking]` ก่อนเสมอ ที่ `max_tokens=32` มันใช้
งบไปกับการคิดจนหมดแล้ว `stop_reason` เป็น `max_tokens` โดยไม่มี text ออกมาเลย
ที่ 2,000 token ตอบถูกต้องปกติ

**ข้อควรรู้** — โมเดลกลุ่มนี้ต้องตั้ง `max_tokens` เผื่อส่วนที่คิด ไม่ใช่แค่ความยาว
คำตอบ ถ้าจะเทสความสามารถ ให้เผื่ออย่างน้อยหลักพัน ไม่งั้นจะสรุปว่า "ใช้ไม่ได้" ทั้งที่
เป็นเรื่องงบ token ล้วน ๆ — ผมสรุปผิดมาแล้วสามข้อรวด

**ภาพทดสอบ vision ก็เช่นกัน** — PNG 8×8 เล็กกว่า patch ของ vision encoder จึงถูก
resize จนกลายเป็นว่างเปล่า แล้วโมเดลตอบว่า "เห็นภาพขาว" ทั้งที่ vision ทำงานปกติ
ตั้งแต่ 64×64 ขึ้นไปตอบถูกหมด (smoke test ในตัวใช้ 64×64 อยู่แล้ว)

---

## โมเดลขึ้นครบทุกอย่าง แต่ agent client เรียก tool ไม่ได้

**อาการ** — `lmds test-text` ผ่าน, chat ได้, streaming ได้, เรียก tool ด้วย schema ง่าย ๆ
ก็ได้ แต่พอ Claude Code ต่อเข้ามาก็ 400 ทันที

```
API Error: 400 Failed to initialize samplers: failed to parse grammar
```

**เคสจริง (2026-08-13)** — `kldzj-gpt-oss-120b-heretic-gguf` บน spark-worker · ไฟล์ครบ
ทุกอย่างจาก repo ไม่ได้ขาดอะไร โมเดลก็ไม่ผิด

ใน server log บอกต้นเหตุไว้ตรง ๆ:

```
parse: error parsing grammar: number of repetitions exceeds sane defaults
```

llama.cpp แปลง JSON schema ของ tool เป็น GBNF · `maxLength` / `maxItems` ค่าสูงถูกขยาย
เป็น repetition ตรง ๆ แล้วชน `MAX_REPETITION_THRESHOLD` (2000) จนโยน exception —
ซึ่ง agent client อย่าง Claude Code ส่ง schema แบบนั้นมาเป็นปกติ

ที่หลอกคือ **bound ตัวเลขไม่ใช่ปัญหา** llama.cpp จัดการ `maximum: 9007199254740991`
ได้สบาย ตัวที่พังคือความยาว string กับจำนวน item เท่านั้น:

| schema | ก่อนแก้ |
|---|---|
| `integer maximum: 9007199254740991` | ผ่าน |
| `string maxLength: 100000` | **ตาย** |
| `array maxItems: 100000` | **ตาย** |

**ตอนนี้** — `lmds doctor` ตรวจว่า llama.cpp ที่โมเดลนั้น pin ไว้มี `cd0fa6051`
(*grammar: degrade max repetition >= 2000 to unbounded*, upstream 2026-08-05) หรือยัง
ไม่มีก็ WARN พร้อมบอกวิธี build ใหม่

ตรวจจากข้อความ error ในไบนารีไม่ได้ เพราะรุ่นที่แก้แล้วก็ยังมีสตริงนั้นอยู่ (`min_times`
ยัง throw) จึงต้องดูที่ commit — และเช็คที่ **build ที่ pin ไว้ให้โมเดลนั้น** ไม่ใช่ของกลาง

**วิธีแก้** — build llama.cpp แยกให้โมเดลนั้นแล้วชี้ `LLAMA_CPP_DIR` ไปหา ไม่ต้องอัปเกรด
ของกลางที่โมเดลอื่นบนเครื่องเดียวกันพิสูจน์แล้วว่าใช้ได้

```bash
cd ~/src/llama.cpp && git fetch origin
git worktree add --detach ~/src/llama.cpp-<slug> origin/master
cd ~/src/llama.cpp-<slug>
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build -j "$(nproc)" --target llama-server
```

แล้วตั้ง `LLAMA_CPP_DIR` ใน controller ให้ชี้มาที่นั่น (bundle ที่สร้างใหม่ตั้งให้เองผ่าน
`plan.runtime.native_dir`)

**บทเรียนกว้างกว่านั้น** — smoke test ที่ผ่านไม่ได้แปลว่า client จริงจะใช้ได้ · schema ที่
เราทดสอบเองมักง่ายกว่าที่ client ส่งมาหลายเท่า เจอครั้งแรกตอนผู้ใช้ต่อเข้ามาจริงเสมอ

---

## stacked: ทำครบทุกขั้น 2.5 ชั่วโมง แล้ว head ตายตอนอ่าน config

**อาการ** — `prepare-runtime` → `download` → `sync-worker` → `verify-worker` ผ่านหมด `start` ปล่อย
worker ต่อ NCCL แล้ว head ตายด้วย `model type qwen4_exp … Transformers does not recognize`

**เคสจริง (2026-09-05)** — Qwen3.8-Flash-Next-NVFP4 บน spark-head + spark-worker · controller
เดี่ยวมีด่าน "image รู้จักสถาปัตยกรรมนี้ไหม" มาตั้งแต่ต้น แต่ stacked ไม่มี

**ตอนนี้** — `check_architecture` ใน controller stacked ถาม image ตัวเดียวกับที่จะ start
(`VLLM_IMAGE` จาก `bundle.env`) ว่ามี `model_type` ของ `config.json` ใน `CONFIG_MAPPING_NAMES`
ไหม **ก่อนปล่อย worker/HCA/NCCL ทั้งคลัสเตอร์** — ไม่รู้จัก = หยุดพร้อมคำสั่ง `lmds set <slug>
--image <ใหม่กว่า>` → `prepare-runtime` → `start` และคำสั่ง `docker run … CONFIG_MAPPING_NAMES`
ให้เช็คเองก่อน · ไม่มี config / ถาม image ไม่ได้ = ปล่อยผ่านให้ vLLM ตัดสิน

---

## stacked: image ครบที่ worker ตัวแรก แต่ตัวท้ายไม่มี

**อาการ** — 4 เครื่อง ผ่านด่านตรวจ image แล้วไปตายตอน `docker run` บนเครื่องท้ายสุด ระหว่างที่
เครื่องอื่นเปิด container ไปแล้ว

**ต้นเหตุ** — ด่านตรวจ image ก่อน start ถามแค่ worker ตัวแรก

**ตอนนี้** — วนทุก worker · ทุก `die` บอกชื่อเครื่อง + คำสั่ง `ssh <user>@<ip> docker pull '<image>'`
· `prepare-runtime` ไม่ตาย raw ใต้ `set -e` อีก: pull ล้มที่ node ไหนบอก node นั้นพร้อมสาเหตุที่พบบ่อยของ
registry นั้น (ghcr rate-limit → `docker login ghcr.io` · nvcr → NGC key · proxy ของ docker daemon ·
ไม่มีเน็ต → `docker save | ssh docker load`) · สั่งซ้ำผ่านทันที

---

## image ที่ตรึง digest ถูกตัดสินว่า "ไม่มีอยู่จริง" แล้วลดรุ่นเงียบ ๆ

**อาการ** — สูตรที่ sync มาระบุ `avarok/dgx-vllm-nvfp4-kernel@sha256:3654…` (ตัวที่รันอยู่จริงบน
spark-head) แต่แผนบน hub ออกมาเป็น nvcr 26.05 → stacked start ตาย `cvt .e2m1x2 not supported on
sm_121` ทุกครั้ง

**ต้นเหตุ** — `split_ref` ตัดที่ `:` ตัวสุดท้าย → repo `…kernel@sha256` + "tag" = เลข digest → registry
404 → แผนเปลี่ยน image ให้โดยไม่มีใครขอ

**ตอนนี้** — digest ถูกตรวจเป็น digest (`manifests/sha256:…`) และ `resolve_digest` ไม่ถาม registry ซ้ำ
· image ของ**สูตร**ที่ registry ตอบไม่พบถูกคงไว้พร้อมเตือน (สูตรคือหลักฐานว่ารันจริง registry ที่ถามไม่ได้
ไม่ใช่) · NVFP4 บน GB10 ที่ต้องถอย image ถอยไป `vllm/vllm-openai@sha256:61fc…` + env marlin ไม่ใช่ nvcr
· registry ที่มีพอร์ต (`registry.local:5000/vllm:tag`) ไม่ถูกตัดที่ `:` ตัวแรกอีก

---

## โหลดขนานแล้วดิสก์เต็มกลางทาง

**อาการ** — `download` ของ GGUF 60 GB ตายที่ 90% ด้วย `No space left` ทั้งที่ `df` ก่อนเริ่มเหลือ 70 GB

**ต้นเหตุ** — `fetch_parallel` เขียนส่วนย่อยลง `<ไฟล์>.parts/` แล้วต่อกันเป็นไฟล์รวม = ต้องมีที่ว่าง
**2 เท่า** ของไฟล์ชั่วขณะ

**ตอนนี้** — ตรวจดิสก์ก่อนเริ่ม: ไม่ถึง 2 เท่า = บอกแล้วถอยไปสตรีมเดี่ยว (ใช้ 1 เท่า ช้ากว่าแต่ได้ไฟล์)
· ไม่พอแม้ไฟล์เดียว = die ตั้งแต่ต้นพร้อมทาง `MODEL_DIR=/data/models` · stacked `download` ตรวจดิสก์
ก่อนเช่นกัน (โมเดล 150–220 GB · บอก `HF_HOME=/data/hf`) · `sync-worker` เช็คดิสก์ฝั่ง worker ก่อนลาก

---

## bundle ใหม่ได้พอร์ต 8000 ซ้ำกับตัวที่มีอยู่

**อาการ** — หน้าภาพรวมขึ้น "port shared" ทันทีหลัง deploy · autostart หลัง reboot ชนกัน ตัวหลังล้มเสมอ

**ต้นเหตุ** — ทุก bundle ตั้งต้นที่ 8000 · การ์ดยังโชว์พอร์ตผิดจนกว่าจะ start ครั้งแรก เพราะทะเบียนของ
bundle ที่ยังไม่เคย start เขียน 8000 เสมอ

**ตอนนี้** — `analyze` เลือกพอร์ตว่างตัวแรกจาก inventory ของ**เครื่องปลายทาง** (ทุก bundle + container
นอกระบบ ไม่ใช่แค่ที่รันอยู่ · stacked ดูทั้ง head และ worker) ส่งใน `plan.port` พร้อมโน้ตว่าทำไม ·
`generate` เขียนลง `bundle.env` (ทางเดียวกับ `lmds set --port` → start/autostart/ปุ่ม test-* ได้พอร์ต
เดียวกัน) · `register_bundle` อ่านพอร์ตจาก bundle.env ด้วย · ระบบยังไม่ยึดพอร์ตแทนคน — แก้เองได้ในช่อง port

---

## hub ที่ไม่มี GPU ดูด weight ลงมาเอง / ถูกเสนอเป็นสมาชิก stacked

**อาการ** — `lmds repair` บน hub VM (ไม่มี GPU/docker/llama.cpp, RAM 12 GB) เริ่มโหลด weight 15.6 GB
อย่างว่าง่าย · หน้า Cluster ขึ้น hub เป็น "not ready · 10G too slow" พร้อมปุ่ม Include/Exclude ที่ไม่มีความหมาย
· เลือกเครื่องในฟลีตแต่ไม่เลือก preset = วิเคราะห์ด้วยฮาร์ดแวร์ของ hub (VM ไม่มี GPU ตกไป dgx-spark-single
128 GB ทั้งที่ปลายทางคือ RTX 5090 32 GB → เสนอ context ที่การ์ดรับไม่ไหว)

**ตอนนี้** — LMDS ตรวจว่าเครื่องนี้รันโมเดลได้จริงไหมจากของที่มี (`llama-server` / docker คู่กับ GPU)
ไม่ใช่ชื่อเครื่อง: ไม่มีสักอย่าง = **control plane** — `download` `repair` `start` `restart`
`prepare-runtime` ถูกปฏิเสธพร้อมบอกคำสั่ง push (`--force` / `LMDS_ROLE=serving` ทับได้) · `lmds doctor`
ขึ้นข้อ **บทบาท** เป็นข้อแรก · หน้า Cluster ขึ้น "control plane — not a stacked candidate" ·
wizard เดา preset จาก GPU/memory ที่ refresher เห็นของเครื่องปลายทาง เดาไม่ได้ = บอกให้เลือกเอง

---

## stacked: คู่ที่เป็นไปไม่ได้ผ่าน analyze แล้วไปตายที่ push

**อาการ** — เลือก target stacked แต่ไม่เลือก worker / worker = head / คนละไซต์ / ไม่มี cluster IP /
GGUF หรือ SGLang / embedding → analyze ตอบ 200 ด้วยแผนที่ไม่รู้ว่าเครื่องที่สองคือใคร แล้วตายทีหลังที่
push/cluster.env/ValueError ตอน generate หรือ 500 เฉย ๆ

**ตอนนี้** — wizard ตรวจคู่ด้วยกติกาเดียวกับหน้า Cluster **ก่อนแตะ Hugging Face** → 422 `{kind:"cluster"}`
พร้อมทางออก · มี worker แต่ไม่ส่ง target = ตั้งใจ stacked จึงเลือก `dgx-spark-stacked` ให้ · worker ถูกส่ง
เฉพาะ target stacked (เลือก stacked แล้วเปลี่ยนใจเป็น single เคยหักหน่วยความจำของ worker ออกจาก budget) ·
`cluster write` อ่าน `NNODES` จาก bundle แล้วตัดกลุ่มให้ตรง — กลุ่ม 4 เครื่องกับ bundle 2 เครื่องไม่ได้
`NNODES=4/TP=4` ทับแผนอีก · `lmds cluster doctor <head> <worker>` ไล่ทีละข้อพร้อมคำสั่งแก้ก่อนลงมือ

---

## stacked: `verify-worker` พิมพ์ PASS ใน 1 วินาทีโดยไม่ตรวจอะไร

**อาการ** — `verify-worker` ผ่านทุกครั้ง แล้ว `start` ไปตายที่ safetensors header บน worker

**ต้นเหตุ** — สองชั้น: `docker run` ไม่มี `-i` → stdin ไม่ถึงคอนเทนเนอร์ python3 ได้สคริปต์ว่างแล้วจบ 0
· heredoc ของ Python อยู่ใน double quote ของ argument ที่ส่งให้ ssh → bash ถอด `"` ในโค้ด ถ้า stdin ถึงจริง
จะ SyntaxError · และ rsync `--partial` ทิ้งไฟล์ครึ่งเดียวชื่อเดิมไว้ นับจำนวนผ่านแล้ว

**ตอนนี้** — ส่งสคริปต์เป็น base64 ทาง stdin + `-i` และตรวจ**ขนาดทุก shard** เทียบ Hub · worker คุย
NCCL ด้วย transport IP (`TRANSPORT_IP_WORKER` / `TRANSPORT_IPS_WORKER`) ไม่ใช่ management IP · ระหว่างรอ
head health แวะดู worker ทุก 60 วิ — worker ตายแล้วไม่ต้องรอ NCCL timeout จนครบ

---

## fit ของ stacked บอกว่า "fits ที่ context 4096" ทั้งที่ start ไม่ขึ้น

**อาการ** — Qwen3-235B-A22B FP8 (220 GiB) บน 2×Spark ตอบ fits-reduced-context แล้ว vLLM ตาย
`No available memory for the cache blocks`

**ต้นเหตุ** — งบรวม 227 GB มีแต่โน้ต "ต้องเผื่อ NCCL" ไม่เคยหักจริง และไม่มีตัวเลขต่อเครื่อง

**ตอนนี้** — หัก NCCL buffer 3 GB/เครื่อง (221 GB) · FitReport/payload มี `per_node` (capacity · OS · engine ·
comm buffer · budget · weights/N · KV/N · reserved ของเครื่องที่แน่นสุด) · vLLM ที่เหลือ KV < 2 GB = ไม่ fit
· ทางเลือกชี้ preset ที่พอจริง (`dgx-spark-stacked-4`) แทน "ใช้ stacked" ทั้งที่กำลัง stacked อยู่ ·
flag ที่ controller เป็นเจ้าของ (`--tensor-parallel-size` `--nnodes` `--node-rank`) ที่หลุดมาจาก LLM ถูก
harden ตัดทิ้งพร้อมเตือน — vLLM ให้ตัวหลังชนะ TP=1 บน 2 เครื่องเคยทำให้ head รอ worker ที่ไม่มีวันมา

---

## ตั้ง `API_KEY` แล้วเซิร์ฟเวอร์รันแบบไม่มี auth เงียบ ๆ

**อาการ** — ตั้ง `API_KEY` กับ llama.cpp แล้วยิงโดยไม่ใส่ key ก็ได้ 200

**เคสจริง (2026-09-04, dgx-spark03, llama-server b10799)** — งาน audit ย้าย key ออกจาก argv ไปเป็น env
`LLAMA_ARG_API_KEY` ซึ่ง build จริง**ไม่มี** · env ที่ engine ไม่รู้จักไม่ error แต่เปิดประตูทิ้งไว้ · เทสที่เชื่อ env
ผ่านทั้งชุด

**ตอนนี้** — llama.cpp ใช้ `--api-key-file` (ไฟล์ 0600 ใน `RUN_DIR` · docker mount ro) พิสูจน์ 401/401/200
กับ binary จริง · เทสตรวจไฟล์+สิทธิ์แทนการเชื่อ env · vLLM/stacked ผ่าน env `VLLM_API_KEY` ที่ export
แล้ว `-e` ไม่มีค่า · **บทเรียน**: การเปลี่ยนเรื่อง auth ต้องรันกับ binary จริงก่อนเสมอ

---

## ตรวจเองก่อน deploy

```bash
lmds doctor <slug>                          # บทบาทเครื่อง, architecture, weight, image/build, grammar, port, สภาพ server
lmds inspect <repo>                         # context, KV dims, capability 6 อย่าง, variant ทั้งหมด
lmds inspect <repo> --target dgx-spark-stacked --context 262144   # ค่านี้ได้กี่คนพร้อมกัน (fp8 ด้วย --kv-dtype)
lmds cluster doctor <head> <worker> --slug <slug>   # stacked: ทำไมคู่นี้ยังไม่พร้อม — ทีละข้อพร้อมคำสั่งแก้
./<slug>-single.sh serve-args               # argv จริงที่จะส่งให้ engine (vLLM: DRY_RUN=1 start) ก่อนรอโหลดหลายนาที
```

`inspect` รายงาน Tool Calling · Vision · JSON Mode · Streaming · System Prompt ·
Reasoning พร้อมหลักฐานว่าดูจากอะไร และคำเตือนเมื่ออ่านไม่ชัด — ดูก่อนตัดสินใจ deploy
เครื่องจริงได้ · `doctor` ตรวจ `role` · `controller` · `hf-token` · `weights` · `permissions` · `disk` ·
`docker` · `image` · `architecture` · `grammar` (llama.cpp เก่ากว่า `cd0fa6051`) · `port` · `server`
· บน control plane ข้อที่แปลว่า "รันไม่ได้" ไม่นับเป็นตัวบล็อก
