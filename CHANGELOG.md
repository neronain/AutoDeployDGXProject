# Changelog

รูปแบบตาม [Keep a Changelog](https://keepachangelog.com/) · เวอร์ชันตาม [SemVer](https://semver.org/)

## [Unreleased]

รอบนี้เกือบทั้งหมดมาจาก **การรันจริงบน DGX Spark (dgx-msi)** — ทุกข้อคือปัญหาที่เจอหน้างานจริง

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

- **โมเดลที่โหลดครบแล้วขึ้น "not downloaded" ตลอดกาล ปุ่ม start ไม่ขึ้น** — doctor นับ
  `projector_files` (mmproj) เป็นไฟล์บังคับ **และบังคับครบทุก precision** (BF16+F16+F32) ทั้งที่
  `llama-server` รับ `--mmproj` ได้ไฟล์เดียว และ controller ก็ไม่ได้โหลด mmproj มาด้วยซ้ำ
  → เรียกร้องไฟล์ที่ไม่มีใครจะโหลดให้ (gemma-4-31b-it-gguf บน dgx-veerasiam โหลดครบ 35 GB แล้ว)
  · แก้: แยกไฟล์ที่ขาดไม่ได้ (weight หลัก = FAIL) ออกจากไฟล์ทางเลือก (mmproj ขาด = **WARN**
  "โมเดลจะรับแต่ข้อความ" ไม่บล็อก) · มี precision ใดตัวหนึ่งก็พอ · `weights_present()` ใช้ตัวตรวจ
  ชุดเดียวกัน ป้าย "not downloaded" บนหน้าเว็บจึงหายตามไปด้วย
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
