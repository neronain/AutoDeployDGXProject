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

## ตรวจเองก่อน deploy

```bash
lmds doctor <slug>          # architecture, weight, image/build, port, สภาพ server
lmds inspect <repo>         # context, KV dims, capability 6 อย่าง, variant ทั้งหมด
```

`inspect` รายงาน Tool Calling · Vision · JSON Mode · Streaming · System Prompt ·
Reasoning พร้อมหลักฐานว่าดูจากอะไร และคำเตือนเมื่ออ่านไม่ชัด — ดูก่อนตัดสินใจ deploy
เครื่องจริงได้
