# CLI Specification — เฟส 1 (MVP)

สเปกคำสั่งของ `lmds` (ชื่อ command ชั่วคราว เปลี่ยนภายหลังได้โดยไม่กระทบโครงสร้าง)
เขียนด้วย Python 3.10+, `typer` + `rich`

> **เอกสารนี้คือ *สเปก* ไม่ใช่คู่มือใช้งาน** — บางส่วนยังไม่ได้ implement และทำเครื่องหมาย ❌ ไว้
> สิ่งที่ใช้ได้จริงวันนี้ ดู [USAGE.md](USAGE.md) หรือ `lmds --help`

## ภาพรวมคำสั่ง

```text
lmds deploy <MODEL_URL_OR_ID> [options]    # flow หลัก: วิเคราะห์ → ยืนยัน → generate → validate → package
lmds inspect <MODEL_URL_OR_ID>             # วิเคราะห์อย่างเดียว ไม่ generate (fit report + ข้อเสนอ runtime)
lmds plan <MODEL_URL_OR_ID>                # สร้าง Deployment Plan (ขั้นวางแผนของ deploy) โดยไม่ generate สคริปต์
                                           #   --no-llm = rule-based, --target <preset>, --json
lmds generate <MODEL_URL_OR_ID>            # plan → render bundle (controller/README/MODEL_PROFILE/SPECIAL_FILES)
                                           #   --output DIR, --target, --no-llm — validate+zip อยู่ใน M6
lmds hardware                              # ตรวจ/แสดง hardware profile ของเครื่อง
lmds scan [--root DIR] [--all] [--json]    # หา weight ที่มีอยู่แล้วบนเครื่อง (อ่านอย่างเดียว)
lmds recipes [MODEL]                       # สูตรที่รันผ่านจริง — ใช้แทน LLM เมื่อไม่มี API key
lmds prune [-y]                            # ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้ว
lmds validate <BUNDLE_DIR> [--fix]         # รัน static quality gates กับ bundle ที่มีอยู่
lmds ps | list | start | stop | restart | logs | enable | disable   # fleet (ดูหัวข้อ fleet)
lmds adopt <CONTAINER> | --port N | --pid N   # รับโมเดลที่รันอยู่ก่อน LMDS เข้าระบบ (ดูหัวข้อ adopt)
lmds repair <SLUG> [--force]               # โหลดไฟล์ที่ขาดกลับมา: download (resume) → verify-files
                                          #   ถูกปฏิเสธบน control plane (ไม่มี engine ให้รัน) — --force เพื่อยืนยัน
lmds bench run <SLUG> [--quick|--speed-only|--caps-only] [--runs N]   # วัดความเร็ว+ความสามารถของโมเดลที่รันอยู่
lmds bench list                            # ตารางคะแนนของทุกโมเดลที่เคยวัด
lmds bench show <SLUG> [--history]         # ผลละเอียด / ประวัติทุกรอบ
lmds bench remove <SLUG> [--keep-last N]   # ลบผลที่เก็บไว้ (ผลสะสมจนตารางอ่านไม่ไหว)
lmds remove <SLUG> [--keep-weights] [-y]   # ลบ bundle/ทะเบียน/log/runtime files/weight ทั้งหมด
lmds node <subcommand>                     # ทะเบียนเครื่องอื่น (fleet หลายเครื่อง) — ดูหัวข้อ node
lmds agent info                            # JSON สถานะเครื่องนี้ (hub เรียกผ่าน SSH ไม่ได้พิมพ์เอง)
lmds config <subcommand>                   # จัดการ provider, credentials
lmds version                               # เวอร์ชันโปรแกรม + เวอร์ชัน template registry

lmds repair <BUNDLE_DIR> --log <FILE|->  ❌ # repair จาก log ความล้มเหลว (คนละอย่างกับ lmds repair ข้างบน
                                           #   ที่ทำแค่เรื่องไฟล์ — ตัววิเคราะห์ log ยังเป็นเฟส 2)
```

## `lmds deploy`

```text
lmds deploy <MODEL_URL_OR_ID> [OPTIONS]

Arguments:
  MODEL_URL_OR_ID   ลิงก์ HF เต็ม | org/model | ลิงก์ไฟล์ .gguf ตรง
                    (❌ ollama.com / NGC / GitHub release = เฟส 2 — ตอนนี้แจ้ง UnsupportedSource)

Options:
  --target PROFILE        dgx-spark-single | dgx-spark-stacked | rtx-* | auto (default: auto = ตรวจเครื่องปัจจุบัน)
                          ** dgx-spark-stacked → สร้าง controller แบบ multi-node (worker-first) อัตโนมัติ **
  --revision REV          pin revision/commit เอง (default: ล่าสุด ณ เวลา inspect แล้ว pin)
  --output DIR            โฟลเดอร์ output (default: ./bundles/<model-slug>/)
  --concurrency N         จำนวน request พร้อมกันที่ใช้คำนวณ KV cache (default 1)
  --yes / -y              ข้ามขั้นยืนยันแผน (ใช้ค่า plan ทั้งหมด) — สำหรับ scripting
  --no-llm                degraded mode: rule-based เท่านั้น (โมเดลตระกูลที่รู้จัก)

ยังไม่ implement:
  --runtime ENGINE   ❌  ตอนนี้ engine มาจาก decision matrix เสมอ (GGUF→llama.cpp, safetensors→vLLM)
  --context TOKENS   ❌  ใช้ขั้นยืนยัน interactive แทน (พิมพ์ค่า context ตอน deploy ถาม)
  --dry-run          ❌
```

### พฤติกรรมสำคัญ

1. **HF token (optional)** — เมื่อเจอ HTTP 401/403 ระหว่าง inspect:
   - ถ้ามี token ใน credential store หรือ env `HF_TOKEN` → ใช้เลย
   - ถ้าไม่มี → prompt แบบ interactive: `ใส่ Hugging Face token (Enter เพื่อข้าม):`
   - ข้าม → แจ้งข้อจำกัดและเสนอทางเลือก (mirror/quant สาธารณะ หรือยกเลิก)
   - โหมด `--yes`/non-interactive → ไม่ prompt, fail พร้อมข้อความบอกวิธีตั้ง token
2. **ขั้นยืนยันแผน** — แสดงตารางสรุป (model/revision, runtime+image digest, topology, context, VRAM/memory budget, feature ที่เปิด, คำเตือน, facts ที่เป็น `unverified`) ให้ผู้ใช้ ยืนยัน / แก้ค่า / ยกเลิก
3. **Extra flags จาก LLM ที่อยู่นอก allowlist** — แสดงเป็นรายการแยกสีเตือน ต้องกดยืนยันรายตัว
4. **Exit codes**: `0` สำเร็จ, `2` validation ไม่ผ่านหลังวนแก้ครบ N รอบ, `3` โมเดลไม่ fit กับ target, `4` ต้องการ token/สิทธิ์, `5` provider/network error
5. **Topology มาจาก target (ไม่มี flag แยก)** — เป็นสมบัติของเครื่องเป้าหมาย ไม่ใช่การตัดสินใจของ LLM · `dgx-spark-stacked` → `stacked` (multi-node controller: worker-first + sync/verify-worker), `rtx-*-dual`/`*-multi` → `multi-gpu` (tensor parallel ในเครื่อง), นอกนั้น → `single` · harden จะบังคับ topology กลับตาม target เสมอ · stacked ต้องใช้ vLLM (GGUF+stacked ถูกปฏิเสธ) · **`--topology both` (สร้าง single+stacked พร้อมกัน) = งานเฟสถัดไป**

## `lmds inspect`

Output (human-readable + `--json`):

```text
Model:      Qwen/Qwen3-32B @ <sha>
Artifact:   safetensors (17 shards, 65.3 GB) | license: Apache-2.0 (commercial OK)
Fit (rtx-single, RTX 4090 24GB):  ❌ FP16 ไม่พอ → ✅ ทางเลือก: GGUF Q4_K_M (~19.8 GB) + offload
Fit (dgx-spark-single, 128GB):    ✅ context ปลอดภัยเริ่มต้น 65536
Runtime แนะนำ: vLLM (spark) / llama.cpp (rtx-24GB)
Special files: chat_template.jinja, tool parser: hermes
```

## `lmds config`

```text
lmds config set-provider <openai|gemini|minimax|anthropic|openai-compat> [--base-url URL] [--model NAME]
                                        # anthropic: ตั้งค่าได้แต่ adapter ยังไม่ทำ (เฟส 2) — จะ error ตอนใช้จริง
lmds config set-key <provider> [--stdin]   # prompt แบบซ่อน input → เก็บ keyring หรือ ~/.config/lmds/credentials (0600)
lmds config set-hf-token [--stdin]      # เก็บ HF token (optional)
lmds config show                        # แสดง config ปัจจุบัน (redact ทุก secret) + ที่มาของ secret แต่ละตัว
lmds config defaults                    # แสดง default model ของแต่ละ provider
lmds config profile edit           ❌   # ยังไม่ implement (มี path ~/.config/lmds/profile.yaml เตรียมไว้แล้ว)
```

ลำดับความสำคัญ credentials: env var > keyring > credentials file
env ที่รองรับ: `LMDS_OPENAI_API_KEY`/`OPENAI_API_KEY`, `LMDS_GEMINI_API_KEY`/`GEMINI_API_KEY`/`GOOGLE_API_KEY`,
`LMDS_MINIMAX_API_KEY`/`MINIMAX_API_KEY`, `LMDS_ANTHROPIC_API_KEY`/`ANTHROPIC_API_KEY`,
`LMDS_OPENAI_COMPAT_API_KEY`, `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`

env อื่นของตัวโปรแกรม: `LMDS_CONFIG_DIR` (ย้าย config dir), `LMDS_NO_BANNER` (ปิด banner),
`LMDS_RUN_ROOT` (ย้ายทะเบียน fleet), `LMDS_SYSTEMD_DIR` (ที่เก็บ unit — ใช้ในเทส),
`LMDS_INSTALL_DIR`/`LMDS_BIN_DIR` (ใช้ตอนรัน `install.sh`)

## Shell completion

```text
lmds --install-completion    # ติดตั้งลง rc ของ shell (bash/zsh/fish)
lmds --show-completion       # แสดงสคริปต์เฉย ๆ ไม่เขียนไฟล์
```

`--install-completion` มากับ typer · เพิ่ม dynamic completion เอง 2 ตัว:
`_complete_slug` (stop/start/restart/logs/enable/disable/repair/remove — อ่านจาก `~/.lmds/run/` + `./bundles/`)
และ `_complete_target` (`--target` ทุกคำสั่ง — จาก `PRESETS`)
ทั้งคู่ห้ามยิง subprocess/network เพราะ shell เรียกทุกครั้งที่กด TAB

## `lmds hardware`

ตรวจและแสดง hardware profile:

```text
Arch: x86_64 | GPU: NVIDIA RTX 4090 (24 GB, SM89) ×1 | RAM: ใช้ไป 12 / 128 GB
Disk ($HOME): เหลือ 1589.5 / 1800.0 GB | IP: 192.168.1.50
Docker ✅ | NVIDIA Container Toolkit ✅ | โปรไฟล์: rtx-single
```

- ดิสก์ดูที่ `$HOME` (ที่เก็บ weight จริง) — เหลือ < 50 GB จะขึ้นคำเตือนใน notes
- ❌ `--probe-ssh user@host` สำหรับตรวจเครื่องเป้าหมายระยะไกล — ยังไม่ implement (เฟส 1.5)

## `lmds` fleet (จัดการโมเดลในเครื่อง)

```text
lmds ps                       # เครื่อง + โมเดลที่รัน/เคยรัน + สถานะจริง + endpoint
lmds list                     # bundle ทั้งหมด + สถานะ (●/◐/○/⚠) + engine/port/context/feature + autostart
lmds start <slug> [flag...]   # start ตามชื่อ (ไม่ต้อง cd ไป bundle) · flag ที่ไม่ใช่ของ lmds ส่งต่อให้ controller
lmds stop <slug> | --all      # stop ตามชื่อ หรือทุกตัว
lmds restart <slug> [flag...] # restart (controller ถ้ามี ไม่งั้น docker restart) · ส่ง flag ต่อได้เหมือน start
lmds logs <slug> [-n N] [-f]  # ดู log · -f = ตาม realtime (docker logs -f / tail -f)
lmds enable <slug> [--now] [--timeout SEC]   # autostart หลัง reboot (systemd, ใช้ sudo)
lmds disable <slug>           # ยกเลิก autostart
lmds repair <slug>            # download (resume) → verify-files
lmds rebuild <slug>           # สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน (เก็บค่าเดิม ไม่เรียก LLM ซ้ำ)
lmds smoke <slug> [--on N]    # พิสูจน์ว่ารันได้จริง: download → verify → start → test-text → stop
lmds remove <slug> [--keep-weights] [-y]     # ลบทุกอย่างที่เกี่ยวข้อง (แสดงรายการ+ขนาดก่อนถามยืนยัน)
```

- **flag ของ controller ส่งผ่านได้ตรง ๆ**: `lmds start <slug> --port 8001 --gpu-util 0.8` — LMDS ไม่พยายาม
  รู้จัก flag ทุกตัว (แต่ละ engine มีไม่เท่ากันและเปลี่ยนตามเวอร์ชัน) แค่ส่งต่อให้ controller ตรวจค่าเอง
  ซึ่งมันตรวจอยู่แล้ว · ก่อนหน้านี้ `--port` ตอบ `No such option` ทั้งที่ controller รองรับ
- **flag ของ controller (llama.cpp) ที่ตั้งได้ตอน start/restart**:

  | flag | ทำอะไร |
  |---|---|
  | `--name NAME` | ชื่อที่ client ใส่ในฟิลด์ `model` · ชื่อที่ generate ตั้งไว้ถูกตรึงเป็น `DEFAULT_SERVED_MODEL_NAME` ที่ override ไม่ได้ แล้วโชว์คู่กันเมื่อไม่ตรง |
  | `--port N` `--context N` `--bind ADDR` | ตามเดิม |
  | `--mmproj FILE` / `--no-mmproj` | เปิด/ปิด vision (โผล่เฉพาะ bundle ที่มีไฟล์ mmproj) |
  | `--mtp FILE` / `--no-mtp` | เปิด/ปิด speculative decoding (โผล่เฉพาะ bundle ที่มีไฟล์ MTP) |

  หน้า help แบ่งเป็น **Identity / Network / Memory & limits / Model features** — หมวดสุดท้าย
  โผล่เฉพาะโมเดลที่มีไฟล์จริง · env เดิม (`MTP_FILE=""`) ยังใช้ได้ และใช้ `${VAR-default}`
  (ไม่มี `:`) เพื่อให้ค่าว่างแปลว่า "ปิด" จริง ๆ ไม่ถูกแทนด้วย default

- **คำสั่งหลัง start อ่านสถานะจริงจาก `server.meta`** — `status`/`logs`/`network-info`/
  `client-config`/`test-*`/`stop` เป็นคนละ process กับตัวที่ start จึงไม่รู้ว่าใช้ flag อะไร
  เดิมไปใช้ default ในไฟล์แล้วรายงานผิด (`start --port 8020` แล้ว `status` บอก
  `API: not responding` เพราะไปถาม 8000) · อ่าน meta **เฉพาะตอนเซิร์ฟเวอร์ยังรันอยู่จริง** ·
  flag ที่ระบุเองชนะเสมอ · `start`/`restart` ไม่สืบทอด ไม่งั้นค่าเก่าติดมาเงียบ ๆ ทุกครั้ง

- **container ที่ไม่ได้มาจาก lmds**: `discover()` สแกน `docker ps` แล้วรับเฉพาะตัวที่ image ตรงกับ
  engine ที่รู้จัก (vLLM/llama.cpp/Ollama/TGI) ทำเครื่องหมาย `external=True` ·
  `stop` ของกลุ่มนี้ใช้ `docker stop` (ไม่ `docker rm -f`) · `enable` สร้าง unit แบบ `docker start <container>`
- **remove**: หยุด → disable autostart → ลบ bundle+ZIP, `~/.lmds/run/<slug>`,
  `~/.lmds/plugins/<slug>`, และ weight (vLLM → HF cache, llama.cpp → `MODEL_DIR`)
  · หา weight ไม่เจอ = ไม่ลบ (ไม่เดา)

- **autostart** = สร้าง systemd system unit `lmds-<slug>.service` (Type=oneshot + RemainAfterExit, `User=<เจ้าของ bundle>`, `ExecStartPre=stop` เคลียร์ container ค้าง, `WantedBy=multi-user.target`) → โมเดลกลับมาเองหลังเปิด-ปิดเครื่อง โดยไม่ต้อง login
- `--now` = start ทันทีด้วย · `--timeout` = เวลารอ health ตอน boot (โมเดลใหญ่ควรเพิ่ม) · ต้องมี `systemd`
- ทุก controller ลงทะเบียนตัวเองใต้ `~/.lmds/run/<slug>/server.meta` ตอน `start` — fleet อ่านจากตรงนี้ (ไม่มี daemon)

## `lmds adopt`

```text
lmds adopt [CONTAINER] [OPTIONS]

Arguments:
  CONTAINER          ชื่อ container ที่รันอยู่ (ดู docker ps) — เว้นว่างได้ถ้าใช้ --port/--pid

Options:
  --port N           รับ process ที่ฟังอยู่ที่พอร์ตนี้ (ไม่ใช่ container)
  --pid N            รับ process ตาม PID
  --slug NAME        ชื่อที่จะใช้ใน lmds (ว่าง = ตั้งให้จากของที่เจอ)
  --output DIR       ปลายทาง (default: ./bundles)
  --take-over        systemctl disable --now unit เดิม แล้วให้ LMDS คุมแทน (ต้อง sudo)
```

สร้าง controller จาก **สิ่งที่รันอยู่จริง** ไม่ใช่จากแผนที่เดาเอา:

| ชนิด | อ่านจาก | ได้อะไร |
|---|---|---|
| container | `docker inspect` | image · env · mount · port · args |
| process | `/proc/<pid>/cmdline`, `exe`, `cwd`, `cgroup` | argv ครบทุก flag · binary · cwd · unit เจ้าของ |

**สิ่งที่ตั้งใจไม่ทำ**

- **ไม่อ่าน `/proc/<pid>/environ`** — API key ของ backend อยู่ในนั้น เขียนลง bundle คือ
  ทำ secret หลุด · cmdline พอสำหรับรันซ้ำ ส่วน env ที่ต้องใช้จริงตั้งใน `bundle.env`
- **ไม่มี `download` / `verify-files` ใน controller ที่ได้** — weight เป็น path ที่ผู้ใช้
  จัดการเอง · คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มีคำสั่งนั้น ·
  `lmds doctor` ก็ตรวจไฟล์ตรง path จริงและไม่แนะ `repair` กับ bundle ประเภทนี้
- **ไม่ปิด unit เดิมให้เอง** — ต้องสั่ง `--take-over` เท่านั้น

**unit เจ้าของ** ถูกบันทึกไว้ใน `MODEL_PROFILE.yaml` (`source_process.unit`) เพราะ unit ที่
ตั้ง `Restart=always` จะแย่ง port กลับทุกครั้งที่ LMDS stop · controller ปฏิเสธ `start`
พร้อมบอกคำสั่งที่ต้องใช้ และ `status` เตือนว่าตัวที่ตอบอาจไม่ใช่ของ LMDS

**หน้าเว็บ**: `POST /api/models/{slug}/adopt` — การ์ดที่ยังไม่มี controller แสดงปุ่ม
*รับเข้าระบบ* แทนปุ่ม Start ที่กดไม่ได้

## `lmds node` (fleet หลายเครื่อง)

```text
lmds node add <host> --user <u>   # ติดตั้ง SSH key (ถามรหัสผ่านครั้งเดียว) + เพิ่มเข้าทะเบียน
                                  #   --name --port --note --cluster-ip --cluster-iface --install
lmds node install <name>          # ติดตั้ง/อัปเดต LMDS บนเครื่องนั้น (--with-prereq = ลง Docker ด้วย)
lmds node list [--check]          # ทะเบียน · --check = ต่อจริงเพื่อดูว่ายังตอบไหม
lmds node set <name> [...]        # แก้ --cluster-ip / --cluster-iface / --note / --site (ไม่มีอาร์กิวเมนต์ = ดูค่าปัจจุบัน)
                                  #   --site = ป้ายจัดกลุ่มเครื่องตามสถานที่ (คอนโซลจัดกลุ่มให้เอง) — ดู node list
lmds node remove <name> [-y]      # ออกจากทะเบียนอย่างเดียว ไม่แตะเครื่องนั้น
lmds node run <name> <cmd...>          # รันคำสั่ง *ของ lmds* บนเครื่องนั้น (ps/start/stop/logs/deploy)
lmds node ctl <name> <slug> <cmd...>   # รัน *สคริปต์ controller* ในตัว bundle บนเครื่องนั้น
                                       #   (prepare-runtime, download, sync-worker, test-text …)
lmds node cluster [--write SLUG] [--worker NAME]   # ตารางสายเชื่อม + กลุ่มที่ stacked ได้
lmds node push <name> <slug> [--download] [--start]  # ส่ง bundle จากเครื่องนี้ไปติดตั้งบนเครื่องนั้น
lmds ps --all                     # โมเดลของทุกเครื่องรวมกัน
```

- **ไม่มี daemon บน node** — hub เรียก `lmds agent info` ผ่าน SSH (`BatchMode=yes`) node เปิดแค่พอร์ต 22
- **แต่ทุก node ต้องมี `lmds` ติดตั้งอยู่** — "agent" คือตัวคำสั่งเอง ไม่ใช่โปรเซสที่รันค้าง
  · hub ติดตั้งให้ได้ด้วย `lmds node install` (ข้ามขั้น sudo/Docker เป็นค่าเริ่มต้น)
- คำสั่งที่ยิงผ่าน SSH ถูกห่อด้วย `bash -lc` — ไม่งั้นจะไม่เจอ `~/.local/bin/lmds`
  (shell แบบ non-interactive ไม่อ่าน `.profile`/`.bashrc`)
- **ไม่เก็บรหัสผ่าน** — ใช้ครั้งเดียวติดตั้ง key (`~/.config/lmds/id_lmds`, ed25519, comment `lmds-hub`)
  แล้วทิ้ง · `Node` dataclass ไม่มีฟิลด์รหัสผ่านและมีเทสกันไว้
- ทะเบียนอยู่ที่ `~/.config/lmds/nodes.yaml` (0600) · แก้ host/user/port ผ่าน `set` ไม่ได้โดยตั้งใจ
  (ที่อยู่เปลี่ยน = คนละเครื่อง → remove แล้ว add ใหม่)
- `node cluster` ตรวจ ConnectX/RDMA/ความเร็วลิงก์จาก `/sys` แล้วจับกลุ่มเครื่องที่ stacked ด้วยกันได้
  (ต้องตรง: arch, profile, รุ่น GPU, จำนวน GPU และมีสาย ≥ 25G ทั้งคู่)
- **`--site` เป็นแค่ป้ายจัดระเบียบ ไม่เกี่ยวกับ cluster** — เป็นคนละมิติกันโดยตั้งใจ: site บอกว่าเครื่องตั้งอยู่
  สถานที่ไหน (คอนโซลจัดกลุ่ม/ยุบ-กางตาม site), ส่วน cluster ดูจาก GPU/สายเชื่อมจริงเท่านั้น
  · เปลี่ยน site ไม่กระทบการจับกลุ่ม stacked และการ stacked ทำได้เฉพาะเครื่องใน site เดียวกันอยู่แล้ว
  (คนละสถานที่ = คนละ subnet/สาย → ไม่มีทางผ่านเกณฑ์ลิงก์) · `node list` จัดตารางแยกตาม site ให้
- **`node run` กับ `node ctl` ต่างกัน**: อันแรกสั่งโปรแกรม `lmds` อันหลังสั่งสคริปต์ controller
  ในตัว bundle · ขั้นตอนของ stacked (`sync-worker`, `verify-worker`) มีเฉพาะใน controller
- `--write <slug>` เขียน `cluster.env` ลง bundle (MASTER_IP/WORKER_IP/SSH_USER/TRANSPORT_IP_*/NCCL_SOCKET_IFNAME)
  → stacked controller source ไฟล์นี้ก่อน default แล้วไม่ถาม IP ตอน start

รายละเอียด: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md)

## `lmds prune`

ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้วและไม่ได้รันอยู่

เครื่องที่ใช้ **จัดการ** อย่างเดียว (โน้ตบุ๊กที่สร้าง bundle ให้เครื่องอื่น) จะสะสมทะเบียนของ bundle
ที่ย้าย/ลบไปแล้ว — หน้าจอเต็มไปด้วยรายการที่กดอะไรก็ไม่ได้ และ**เสี่ยงสั่งการผิดเครื่อง**

- ทะเบียนที่ **ไม่เคยถูก start** และ controller หายไปแล้ว → เก็บกวาดอัตโนมัติตอน `ps`/`list`
- ทะเบียนที่ **เคยรันจริง** แล้ว controller หาย → ยังแสดงพร้อมคำเตือน (ผู้ใช้ต้องรู้) ล้างด้วย `prune`
- **ลบเฉพาะไฟล์ทะเบียน** ไม่แตะ weight, bundle หรือ container

## `lmds recipes [MODEL]`

สูตรที่ **รันผ่านจริงบนฮาร์ดแวร์แล้ว** เก็บไว้ในตัวโปรแกรม (`src/lmds/recipes/catalog.yaml`)

ปัญหาที่แก้: ลูกค้า/ทีม SI จำนวนมากไม่มี API key ของ LLM → `deploy` ตกไปใช้ rule-based ซึ่งรู้แค่
"GGUF → llama.cpp, safetensors → vLLM" ไม่รู้เรื่องเฉพาะรุ่น → **deploy ผ่านแต่ start ไม่ขึ้น**

สูตรกำหนดได้: `image` · `serving` (kv_cache_dtype, quantization, moe_backend, …) · `tool_calling`
· `reasoning` · `env` · `notes` — และ **ต้องมี `source` + `validated_on` เสมอ** สูตรที่ไม่มีที่มา
คือการเดา มีเทสบังคับไว้

- **ไม่แตะ `context` / `max_output`** — สองค่านี้ต้องมาจากการวิเคราะห์หน่วยความจำของเครื่องเป้าหมาย
  ไม่ใช่ค่าคงที่จากเครื่องที่เคยรัน
- **`image_for`** ผูก image กับสถาปัตยกรรมที่ทดสอบมา — build ของ DGX Spark (ARM64/SM121)
  ไม่ถูกนำไปใช้กับ RTX โดยอัตโนมัติ จะเตือนแล้วใช้ค่าตั้งต้นแทน
- `match` เทียบแบบ prefix ไม่สนตัวพิมพ์ · สูตรที่เฉพาะเจาะจงกว่าชนะ

```bash
lmds recipes                              # ทั้งหมด
lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4   # รายตัว พร้อมที่มา
```

## `lmds scan`

หา weight ที่มีอยู่แล้วบนเครื่อง **ไม่ว่าจะถูกเก็บไว้แบบไหน** — เครื่องลูกค้ามักมีโมเดลอยู่ก่อน
ติดตั้ง LMDS และไม่ได้จัดระเบียบแบบเดียวกับเรา

ที่ค้น: `HF_HOME` · `HF_HUB_CACHE` · `TRANSFORMERS_CACHE` · `MODEL_DIR` · `LLAMA_CACHE` (จาก env)
· `~/.cache/huggingface[/hub]` · `~/models` · `~/data/models` · `/models` · `/opt/models`
· `/srv/models` · `/data/models` · `/mnt/models` · เพิ่มเองด้วย `--root`

รายงาน: ชนิด (hf/gguf) · ชื่อ · ขนาด · จำนวน shard · path จริง · **เลย์เอาต์ของ HF cache**

```text
hf    nvidia/DeepSeek-V4-Flash-NVFP4   157.0 GB   46   ~/.cache/huggingface/models--…
                                                       (เลย์เอาต์เก่า — ต้องตั้ง HF_HUB_CACHE)
gguf  gemma-4-26B-A4B-it-UD-Q8.gguf     25.7 GB    —   ~/models/…
```

- **อ่านอย่างเดียว ไม่ย้ายไม่ลบอะไรทั้งสิ้น** — weight เป็นของผู้ใช้ และการย้าย 150 GB เงียบ ๆ ยอมรับไม่ได้
- `--all` ค้นทุกเครื่องในทะเบียนด้วย (เรียก `lmds scan --json` บน node ผ่าน SSH)
- เลย์เอาต์ `root` (`$HF_HOME/models--X` ไม่มี `hub/`) ไลบรารีของ HF จะมองไม่เห็นถ้าไม่ตั้ง
  `HF_HUB_CACHE` ให้ตรง — **stacked controller ตั้งให้เองตอน start แล้ว ไม่ต้องย้ายไฟล์**

## `lmds doctor <slug>`

ตรวจว่าทำไมโมเดลยัง download/start ไม่ผ่าน แล้วบอกคำสั่งแก้ — คำนวณล้วน ไม่ใช้ LLM

ตรวจ: `bundle` · `hf-token` (gated แต่ไม่มี `HF_TOKEN`) · `weights` (รวม mmproj และไฟล์ 0 ไบต์) ·
`permissions` · `disk` · `docker` · `runtime-image` · `port` (บอกชื่อโมเดลที่ยึดอยู่) · `server`

Exit codes: 0 ไม่พบปัญหาที่บล็อก · 2 มีข้อที่ต้องแก้

- **`weights` ตรวจเฉพาะไฟล์ที่ขาดไม่ได้** — weight หลัก (GGUF ที่เลือก / snapshot ของ HF repo)
- **`multimodal` เป็น WARN ไม่ใช่ FAIL** — `mmproj` ขาดแปลว่าเสีย vision กลายเป็น text-only
  แต่โมเดล**ยังรันได้** จึงไม่บล็อก · repo มักมี mmproj หลาย precision (BF16/F16/F32) ให้เลือก
  แต่ `llama-server` รับ `--mmproj` ได้ไฟล์เดียว — **มีตัวใดตัวหนึ่งก็ผ่าน**
  · เดิมบังคับครบทุกตัว ทำให้โมเดลที่โหลดครบขึ้นว่า "ยังไม่ download" ตลอดกาล
  และปุ่ม start บนหน้าเว็บไม่ขึ้น (เจอจริงกับ gemma-4-31b-it-gguf บน dgx-veerasiam)

## `lmds web`

| ตัวเลือก | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--port` | 8600 | พอร์ตของหน้าเว็บ |
| `--bind` | `127.0.0.1` | `0.0.0.0` = ทั้งวง network (จะสุ่ม token ให้ถ้าไม่ตั้ง `--token`) |
| `--token` | ว่าง | บังคับ token เอง (≥ 8 ตัว ไม่มีช่องว่าง/ตัวควบคุม) |
| `--background` / `-b` | ปิด | รันเบื้องหลัง — terminal ว่างใช้ CLI ต่อได้ |
| `--stop` | — | หยุดตัวที่รันเบื้องหลัง |
| `--restart` | — | หยุดตัวเดิมแล้วเปิดใหม่ (**ลิงก์เดิมใช้ได้ต่อ**) |
| `--new-token` | — | สุ่ม token ใหม่ — ลิงก์เดิมใช้ไม่ได้ทันที |
| `--status` | — | บอกว่ามีตัวไหนรันอยู่ + **ลิงก์พร้อม token ของตัวนั้น** |

- **สตาร์ตซ้อนไม่ได้** — สั่ง `lmds web -b` ตอนที่มีตัวรันอยู่แล้ว จะพิมพ์ลิงก์ของ*ตัวที่เสิร์ฟจริง*ให้
  แทนการสตาร์ตใหม่ · เดิมรอบสอง bind ไม่ได้แล้วตาย แต่ CLI พิมพ์ token ใหม่ให้ ผู้ใช้จึงเจอ
  `token ไม่ตรง` ทั้งที่ copy ลิงก์มาถูก
- **`-b` รอจนหน้าเว็บรับ connection ได้จริง** ก่อนบอกว่าสำเร็จ — ขึ้นไม่ได้จะพิมพ์ท้าย log ให้เลย
- **ที่มาของ token** (ตัวบนสุดที่มีค่าชนะ): `--token` → `$LMDS_WEB_TOKEN` →
  `~/.config/lmds/web-token` (0600) → **ถามตอนสตาร์ตครั้งแรก** (Enter = สุ่มให้) → สุ่มให้
  · ไม่มี tty จะไม่ค้างรอ
- **ลิงก์ที่พิมพ์ออกมาไม่มี token** — URL ไปโผล่ใน history/log ของ proxy/referrer ·
  token พิมพ์แยกบรรทัดพร้อมบอกที่มา · หน้าเว็บมีหน้า login ให้กรอก แล้วจำไว้ในเบราว์เซอร์
- `GET /api/auth` → `{"required": bool}` · `POST /api/auth` (header `x-lmds-token`) → 200/401
  · ผิดติดกัน > 5 ครั้งต่อ IP → 429 หน่วงแบบทวีคูณ สูงสุด 60 วินาที
- สถานะเก็บที่ `~/.lmds/run/web.json` (สิทธิ์ 0600 เพราะมี token) · `--stop` ตรวจ cmdline ก่อนฆ่า
- `GET /api/version` คืน `commit` (โค้ดที่ process นี้รันอยู่), `installed` (ของบนดิสก์) และ
  **`boot`** = ลายเซ็นของ process ที่สุ่มใหม่ทุกครั้งที่บริการเริ่ม · หน้าเว็บใช้ตอบว่า
  "restart เสร็จหรือยัง" โดยไม่ผูกกับว่ามีโค้ดใหม่ไหม — เดิมรอให้ `commit` เปลี่ยน แล้วค้าง
  120 วินาทีทุกครั้งที่อัปเดตแล้วไม่มีอะไรใหม่ จนสรุปผิดว่าเซิร์ฟเวอร์ไม่กลับมา

### API ของผู้ช่วย (หน้าเว็บเรียก — ไม่ใช่ CLI)

| Endpoint | ทำอะไร |
|---|---|
| `POST /api/assistant/chat` | สตรีม SSE: `status` → `evidence` (ไปดูอะไรมา) → `ticket` (ถ้ามีงานเสนอ) → `delta` |
| `GET /api/assistant/ticket/{id}` | สถานะงานที่เสนอ + เมนูให้เลือก |
| `POST /api/assistant/ticket/{id}/choose` | ผู้ใช้เลือก `apply` / `step` / `hold` — **จุดเดียวที่งานเริ่มทำงานได้** |
| `POST /api/assistant/ticket/{id}/advance` | ทำขั้นถัดไป (โหมด `step`) |

ตั๋วออกโดยเซิร์ฟเวอร์เท่านั้น อายุ 30 นาที และแต่ละขั้นใช้ได้ครั้งเดียว — LLM ออกตั๋ว
ให้ตัวเองไม่ได้ (PRD FR-1c.5)
  เผื่อ PID ถูกใช้ซ้ำไปแล้ว · พอร์ตไม่ว่างโดยไม่ใช่ของ lmds จะบอกตรง ๆ พร้อมคำสั่งหาว่าใครยึด

REST ที่หน้าเว็บใช้ (token เดียวกับหน้าเว็บ): `/api/host` `/api/models` `/api/nodes`
`/api/nodes/{name}/inventory` `/api/cluster` — `PATCH /api/nodes/{name}` แก้ cluster IP ·
`POST /api/nodes/{name}/models/{slug}/start|restart` รับ body
`{"port","context","slots","bind","api_key","gpu_util"}` — ตรวจด้วย `jobs.clean_options()`
**ตัวเดียวกับโมเดลในเครื่อง** (port 1–65535 · context 256–10M · slots 1–1024 · gpu_util 0.3–0.98 ·
bind เฉพาะ 0.0.0.0/127.0.0.1 · api_key ห้ามมีช่องว่าง) แล้วแปลงเป็น **env ของ controller**
· คำสั่งอื่นส่ง option มาด้วย = 400

`POST /api/nodes/{name}/models/{slug}/ctl/{command}` — สั่ง *คำสั่งของ controller* บนเครื่องนั้น
(ชุดทดสอบ/ข้อมูล) · allowlist รับเฉพาะคำสั่งที่อ่านหรือทดสอบ: `test-*` `bench` `stress`
`client-config` `network-info` `status` `props` `verify-files` `prepare-runtime` `sync-worker`
`verify-worker` `clear-fi-cache` — `start`/`stop`/`download` มีทางของมันเองที่จัดการ option แล้ว

`POST /api/models/{slug}/push/{name}` — ส่ง bundle ZIP ของ slug นั้นไปแตกที่ `~/bundles` บนเครื่องนั้น

คำสั่งข้ามเครื่องจำกัดด้วย allowlist `start stop restart repair doctor logs enable disable remove`
(`logs` ถูกบังคับ `-n 300`) — **`remove` ต้องผ่านสองขั้น**: คำขอที่ไม่มี `confirm` จะรัน `--dry-run`
(ไม่ลบอะไร) · จะลบจริงต้องส่ง `{"confirm": "<slug>"}` ที่ตรงกับชื่อโมเดลเป๊ะ ไม่งั้น 400

UI เป็นภาษาอังกฤษ · ต้องมี extra `web` (`pip install 'lmds[web]'` — `install.sh` ลงให้เอง)
รายละเอียดการใช้งาน: [USAGE.md §5](USAGE.md)

## `lmds validate`

รัน quality gates กับ bundle ใด ๆ (รวม bundle ที่แก้มือ):

- `bash -n` ทุกสคริปต์
- Audit rules v3.0.0 (numeric underscore, pipefail, flags/env ครบ, bind/advertise แยก)
- Schema `MODEL_PROFILE.yaml`
- Secret scan
- ตรวจ `PACKAGE_SHA256SUMS` (`--fix` เพื่อ regenerate)

Output: ตาราง pass/fail ต่อ gate + exit code `0/2`

## โครงสร้าง config บนเครื่องผู้ใช้

```text
~/.config/lmds/
├── config.yaml          # provider, default target, ภาษา UI (th/en)
├── credentials          # 0600 — ใช้เมื่อไม่มี keyring
├── profile.yaml         # site profile (master/workers, paths, api defaults)
├── nodes.yaml           # 0600 — ทะเบียนเครื่องอื่น (ไม่มีรหัสผ่าน) + cluster IP
├── id_lmds[.pub]        # SSH key ของ hub สำหรับเข้า node
└── sessions/            # audit log ต่อการ generate (prompt/response/decisions) — redacted
```

## โครงสร้าง source (ของจริง ณ ปัจจุบัน)

```text
src/lmds/
├── cli/                 # main.py (typer commands ทั้งหมด), banner.py
├── config/              # settings.py (config.yaml + provider), paths.py
├── resolver/            # parse.py — HF เท่านั้น (Ollama/NGC โยน UnsupportedSource)
├── inspector/           # inspect.py, hf_api.py, gguf.py (header ผ่าน HTTP Range), report.py
├── hardware/            # profiler.py (nvidia-smi/docker), profiles.py (GPU allowlist)
├── fit/                 # analyzer.py (memory/KV cache), targets.py (target presets)
├── brain/               # providers.py, orchestrator.py, plan_schema.py, prompts.py,
│                        #   rulebased.py, allowlists.py (flag/image allowlist)
├── assistant/           # สิ่งที่ผู้ช่วยในหน้าเว็บ "ทำได้" — ไม่ผูกกับ FastAPI
│                        #   catalog.py (probe อ่านอย่างเดียว + action ที่เปลี่ยนเครื่อง),
│                        #   runner.py (รันผ่าน nodes/ssh + redact + ตัดตามงบ),
│                        #   policy.py (ตั๋วอนุมัติ: แก้เลย/ทีละขั้น/ยังไม่ทำ),
│                        #   router.py (LLM เลือกจากแคตตาล็อกเป็น JSON แล้วโค้ดตรวจซ้ำ),
│                        #   knowledge.py + playbook.md (วิธีคิด + ค้น docs/ ตัวจริง)
├── generator/           # renderer.py + templates/*.j2 (single-vllm, single-llamacpp, stacked-vllm)
├── validator/           # gates.py — quality gates ทั้ง 10 ด่านรวมอยู่ไฟล์เดียว
├── fleet/               # manager.py — discover/stop/start/restart/logs/remove/repair + systemd unit
├── nodes/               # registry.py (nodes.yaml), ssh.py (key/probe/run), cluster.py (จับคู่ stacked)
├── inventory.py         # payload ชุดเดียวที่หน้าเว็บและ `lmds agent info` ใช้ร่วมกัน
├── packager/            # bundle.py (PACKAGE_SHA256SUMS + zip)
└── secrets/             # store.py (env/keyring/file), redact.py
tests/                   # ~22 ไฟล์ test (unit + E2E) — ยังไม่มี tests/fixtures/
```

> ยังไม่มี: `.github/workflows` (CI), `tests/fixtures/` สำหรับ regression เทียบ controllers v3.0.0,
> template registry แบบไฟล์ data (image digest ยัง hardcode ในโค้ด)
