# CLI Specification — 0.6.0

สเปกคำสั่งของ `lmds` เขียนด้วย Python 3.10+, `typer` + `rich` · ปรับให้ตรง `lmds --help` ของ 0.6.0 (a4ec6bb, 2026-09-04)

> **เอกสารนี้คือ *สเปก* ไม่ใช่คู่มือใช้งาน** — ส่วนที่ยังไม่ได้ implement ทำเครื่องหมาย ❌ ไว้
> วิธีใช้และตัวอย่างจริง ดู [USAGE.md](USAGE.md) หรือ `lmds <คำสั่ง> --help` (help ทุกคำสั่งเป็นไทย · หน้าเว็บอังกฤษ)

## ภาพรวมคำสั่ง

```text
lmds deploy <MODEL> [options]              # flow หลัก: วิเคราะห์ → วางแผน → ยืนยัน → generate → validate → ZIP
lmds inspect <MODEL> [--target …] [--context N] [--kv-dtype bf16|fp8] [--json]
lmds plan <MODEL> [--no-llm] [--target] [--engine] [--json]     # Deployment Plan อย่างเดียว
lmds generate <MODEL> [--gguf] [--engine] …                      # เหมือน deploy --yes แต่ไม่ต่อรอง flag
lmds validate <BUNDLE_DIR> [--fix]         # quality gates 12 ด่าน — exit 0 ผ่าน, 2 ไม่ผ่าน
lmds smoke <SLUG> [--on NODE] [--keep] [--skip-download]        # download → verify → start → test-text → stop
lmds rebuild <SLUG> [--output DIR]         # สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน in-place ไม่เรียก LLM
lmds adopt [CONTAINER] | --port N | --pid N [--slug] [--take-over]   # รับโมเดลที่รันอยู่ก่อน LMDS
lmds ps [--all] | list | start | stop [--all] | restart | logs [-n] [-f] | enable [--now] [--timeout] [--system] | disable
lmds set <SLUG> [--port --context --slots --bind --gpu-util --model-id --image --engine-env
                 --tool-parser --reasoning-parser --image-min-tokens --extra-args --auto --clear]
lmds repair <SLUG> [--force]               # download (resume) → verify-files · ถูกปฏิเสธบน control plane
lmds remove <SLUG> [--keep-weights] [-y] [--dry-run]
lmds doctor <SLUG>                         # ทำไม download/start ไม่ผ่าน — exit 0/2
lmds hardware                              # ตรวจเครื่องนี้ + target profile
lmds scan [--root DIR]… [--all] [--json]   # weight ที่มีอยู่แล้ว (อ่านอย่างเดียว)
lmds recipes [MODEL] [--sync] [--repo] [--ref] [--publish SLUG --features … --no-push]
lmds prune [-y]                            # ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้ว
lmds bench run|list|show|remove            # คะแนนโมเดลที่รันอยู่ (ดู BENCH.md)
lmds node add|list|remove|install|setup|set|run|ctl|cluster|clone|push   # fleet หลายเครื่อง
lmds cluster show|write|pair|doctor        # คลัสเตอร์ stacked
lmds web [--port --bind --token -b --stop --restart --status --new-token --enable --disable]
lmds config set-provider|set-key|set-hf-token|show|defaults
lmds agent info|bench                      # JSON ให้ hub เรียกผ่าน SSH
lmds version                               # เวอร์ชัน + commit ที่รันอยู่ + template standard
lmds --install-completion | --show-completion

lmds repair <BUNDLE_DIR> --log <FILE|->  ❌ # repair จาก log ความล้มเหลว (เฟส 2 — ผู้ช่วยบนหน้าเว็บอ่าน log ให้แทน)
lmds deploy --topology both              ❌ # สร้าง single+stacked พร้อมกัน
lmds config profile edit                 ❌
```

## `lmds deploy` / `generate` / `plan`

```text
lmds deploy <MODEL> [OPTIONS]

Arguments:
  MODEL             ลิงก์ HF เต็ม | org/model | ลิงก์ไฟล์ .gguf ตรง
                    (❌ ollama.com / NGC / GitHub release = เฟส 2 — ตอนนี้แจ้ง UnsupportedSource)

Options:
  --target PRESET         ว่าง = เครื่องนี้ / dgx-spark-single · dgx-spark-stacked = multi-node 2 เครื่อง ·
                          dgx-spark-stacked-4 · rtx-* (22 preset · TAB เติมได้)
  --revision REV          pin revision/commit เอง (default: ล่าสุด ณ เวลา inspect แล้ว pin)
  --output DIR            โฟลเดอร์ output (default: ./bundles)
  --concurrency N         จำนวน request พร้อมกันที่ใช้คำนวณ KV cache (default 1) · llama.cpp: slot = N, ctx-size = N × ต่อ slot
  --engine vllm|sglang    เลือกรันไทม์เอง — ว่าง = ตามชนิดไฟล์ (GGUF→llama.cpp, safetensors→vLLM) · GGUF บังคับ llama.cpp เสมอ
  --task generate|embed   ชนิดงาน — ปกติเดาจาก repo (pipeline_tag/tags/ชื่อ) ใส่เมื่อเดาผิด · LLM ตั้งเองไม่ได้
  --gguf FILE|QUANT       repo GGUF หลาย variant: ชื่อไฟล์เต็ม / ชื่อ quant (Q8_K_XL ไม่สนตัวพิมพ์) / ส่วนของชื่อที่ตรงไฟล์เดียว
                          — จำเป็นเมื่อไม่มี tty ให้เลือกหมายเลข (script/hub) · ตรงหลายไฟล์ = ปฏิเสธพร้อมรายการ
  --no-llm                rule-based mode (ใช้สูตรจาก lmds recipes)
  --yes / -y              ข้ามขั้นยืนยัน — flag ค้างอนุมัติจะไม่ถูกใส่

ยังไม่ implement:
  --context TOKENS   ❌  ใช้ขั้นยืนยัน interactive แทน (deploy ถาม) · หลัง generate ใช้ lmds set --context
  --dry-run          ❌  (ใช้ lmds plan)
```

`generate` = deploy โดยไม่ถามยืนยัน (มี `--gguf` `--engine` แต่ไม่มี `--task`/`--yes`) · `plan` มี `--json` และไม่มี `--output`

### พฤติกรรมสำคัญ

1. **HF token (optional)** — เจอ 401/403 ตอน inspect: มี token ใน credential store / `HF_TOKEN` → ใช้เลย · ไม่มี → prompt
   `ใส่ Hugging Face token (Enter เพื่อข้าม)` · `--yes`/ไม่มี tty → fail exit 4 พร้อมบอกวิธีตั้ง · analyze บนหน้าเว็บบอกวิธีใส่ตรง ๆ
2. **ขั้นยืนยันแผน** — ตารางสรุป (model/revision, runtime+image digest, topology, context, budget, feature, คำเตือน, facts `unverified`)
   ให้ ยืนยัน / แก้ context / ยกเลิก · flag นอก allowlist ถามทีละตัว default = ไม่อนุมัติ
3. **Exit codes**: `0` สำเร็จ · `1` input ผิด/ยกเลิก · `2` ไม่ผ่าน gates · `3` ไม่ fit · `4` ต้องการ token · `5` provider/network
4. **Topology มาจาก target** — `dgx-spark-stacked[-4]` → stacked (controller multi-node) · `rtx-*-dual` → multi-gpu · นอกนั้น single ·
   harden บังคับกลับเสมอ และตัด flag ที่ controller เป็นเจ้าของ (`--tensor-parallel-size` `--nnodes` `--node-rank`
   `--distributed-executor-backend`) ที่หลุดมาจาก LLM · stacked ต้องใช้ vLLM + safetensors — GGUF / SGLang / embedding
   ถูกปฏิเสธ (CLI: PlanError · เว็บ: 422 `{kind}`)
5. **task** — `embed` มาจาก repo เท่านั้น (harden บังคับ) · llama.cpp `--embedding --pooling <ตามตระกูล>` · vLLM
   `--runner pooling --convert embed` · SGLang ที่ขอมาถอยเป็น vLLM · stacked ปฏิเสธ
6. **port** — `analyze`/`generate` เลือกพอร์ตว่างตัวแรกจาก inventory ของเครื่องปลายทาง (stacked: head และ worker) เขียนลง `bundle.env`
7. **image** — tag ถูก resolve เป็น digest ตอน generate · digest ที่ระบุมาถูกตรวจเป็น digest ไม่ถาม registry ซ้ำ · image ของสูตร
   ที่ registry ตอบไม่พบถูกคงไว้พร้อมเตือน · ถาม registry ไม่ได้ = ใช้ tag ตามเดิม

## `lmds inspect`

```text
--revision · --target PRESET (ซ้ำได้ · ว่าง = เครื่องนี้ + dgx-spark-single) · --concurrency N
--context N       ถามว่าค่านี้ควรตั้งไหม — ตาราง context × KV ต่อคน × พร้อมกัน + ข้อควรระวัง (GQA และ MLA)
--kv-dtype bf16|fp8   ให้ทั้งตารางคิดที่ fp8
--json
```

Output: model/revision · artifact · fit ต่อ target (verdict `fits` / `fits-reduced-context` / `needs-smaller-quant` /
`fits-with-offload`, stacked มี `per_node`) · runtime แนะนำ · ความสามารถ 6 อย่าง (Tool Calling · Vision · Reasoning ·
System prompt · JSON mode · Streaming) พร้อมหลักฐาน · MoE/MTP จากไฟล์ · variant GGUF ทั้งหมด · task (generate/embed)

Exit: `0` · `1` input ผิด · `4` ต้องการ token · `5` เครือข่าย/Hub (Xet: read timeout 120 วิ / connect 30 วิ)

## `lmds set <SLUG>`

บันทึกค่า start ไว้กับ bundle — เขียน `bundle.env` (และ `bundle.args` สำหรับ `--extra-args`) ข้าง controller ซึ่ง controller
อ่านก่อนตั้ง default ทุกตัว · env ภายนอกและ flag บรรทัดคำสั่งชนะไฟล์นี้เสมอ · **ไม่เก็บ API key**

```text
--port INT · --context INT · --slots INT · --bind 0.0.0.0|127.0.0.1 · --gpu-util FLOAT (0–1, vLLM/SGLang)
--model-id NAME           ชื่อที่ API เสิร์ฟออกไป
--image IMAGE             image ที่ใช้แทนของ bundle (digest ได้ · stacked ถึง worker)
--engine-env "K=V K2=V2"  env ของ engine เอง — docker แตกเป็น -e · llama.cpp export · stacked ถึง worker
--tool-parser NAME · --reasoning-parser NAME     (vLLM/SGLang/stacked)
--image-min-tokens N|auto  llama.cpp vision (Qwen-VL ~1024 · Gemma-4 ต้อง auto)
--extra-args '…'          แฟล็กเพิ่มต่อท้าย argv (JSON เขียนติดกัน · --flag=value ได้)
--auto                    เติม parser/image/env จากสูตรที่รันผ่านจริง > กฎตระกูล · flag ที่ระบุเองชนะ
--clear                   ลบค่าที่บันทึกไว้ทั้งหมด (หน้าเว็บ: Reset to bundle)
```

ตรวจก่อนเขียน (`SettingsError`): port 1–65535 · context/slots จำนวนเต็มบวก · gpu_util 0–1 · bind สองค่า ·
`served_name`/`image` ห้ามมี `" ' \` $ \ { }` · engine env ต้องเป็น `KEY=VALUE` ห้าม `{}` — ไฟล์ถูก `source` ทุก start

## `lmds config`

```text
lmds config set-provider <openai|gemini|minimax|anthropic|openai-compat> [--base-url URL] [--model NAME]
                                        # anthropic: ตั้งค่าได้แต่ adapter ยังไม่ทำ (เฟส 2) — error ตอนใช้จริง
lmds config set-key <provider> [--stdin]   # keyring หรือ ~/.config/lmds/credentials (0600)
lmds config set-hf-token [--stdin]
lmds config show                        # redact ทุก secret + ที่มาของ secret แต่ละตัว
lmds config defaults
lmds config profile edit           ❌
```

ลำดับความสำคัญ credentials: env var > keyring > credentials file
env ที่รองรับ: `LMDS_OPENAI_API_KEY`/`OPENAI_API_KEY`, `LMDS_GEMINI_API_KEY`/`GEMINI_API_KEY`/`GOOGLE_API_KEY`,
`LMDS_MINIMAX_API_KEY`/`MINIMAX_API_KEY`, `LMDS_ANTHROPIC_API_KEY`/`ANTHROPIC_API_KEY`,
`LMDS_OPENAI_COMPAT_API_KEY`, `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`

env อื่นของตัวโปรแกรม: `LMDS_CONFIG_DIR` · `LMDS_NO_BANNER` · `LMDS_RUN_ROOT` · `LMDS_SYSTEMD_DIR` (เทส) ·
`LMDS_ROLE=hub|serving` (ทับการตรวจ control plane) · `LMDS_WEB_TOKEN` · `LMDS_INSTALL_DIR`/`LMDS_BIN_DIR`/`LMDS_ASSUME_YES`/
`LMDS_SKIP_PREREQ`/`PIP_RETRIES`/`PIP_TIMEOUT` (ใช้ตอนรัน `install.sh`)

## Shell completion

```text
lmds --install-completion    # ติดตั้งลง rc ของ shell (bash/zsh/fish) — install.sh ถามให้
lmds --show-completion
```

dynamic completion 2 ตัว: `_complete_slug` (stop/start/restart/logs/enable/disable/repair/remove/set/smoke/rebuild — อ่านจาก
`~/.lmds/run/` + `./bundles/`) และ `_complete_target` (`--target` ทุกคำสั่ง — จาก `PRESETS`) · ห้ามยิง subprocess/network

## `lmds hardware`

```text
Arch: x86_64 | GPU: NVIDIA RTX 4090 (24 GB, SM89) ×1 | RAM: ใช้ไป 12 / 128 GB
Disk ($HOME): เหลือ 1589.5 / 1800.0 GB | IP: 192.168.1.50
Docker ✅ | NVIDIA Container Toolkit ✅ | โปรไฟล์: rtx-single
```

- ดิสก์ดูที่ `$HOME` — เหลือ < 50 GB ขึ้นคำเตือน · dual-GPU ในเครื่องเดียวนับเป็น 1 node (`gpus_per_node`)
- เครื่องอื่นตรวจผ่าน fleet (`lmds node add` → `lmds agent info`) — ไม่มี `--probe-ssh` แยก

## `lmds` fleet (จัดการโมเดลในเครื่อง)

```text
lmds ps [--all]               # เครื่อง + โมเดลที่รัน/เคยรัน + สถานะจริง + endpoint (--all = ทุกเครื่องในทะเบียน)
lmds list                     # bundle ทั้งหมด + สถานะ (●/◐/○/⚠) + engine/port/context/feature (รวม embedding (pooling)) + autostart
lmds start <slug> [flag...]   # flag ที่ไม่ใช่ของ lmds ส่งต่อให้ controller
lmds stop <slug> | --all
lmds restart <slug> [flag...] # controller ถ้ามี ไม่งั้น docker restart
lmds logs <slug> [-n N] [-f]
lmds enable <slug> [--now] [--timeout SEC] [--system]   # ค่าเริ่มต้น = systemd user service ไม่ต้อง sudo · --system = /etc/systemd/system (sudo)
lmds disable <slug>
lmds repair <slug> [--force]  # download (resume) → verify-files · ปฏิเสธบน control plane
lmds rebuild <slug> [--output DIR]   # in-place ตามค่าเดิม · ไม่เรียก LLM
lmds smoke <slug> [--on NODE] [--keep] [--skip-download]   # exit 0/2 · หยุด server เสมอแม้ล้ม
lmds remove <slug> [--keep-weights] [-y] [--dry-run]
```

- **flag ของ controller ส่งผ่านได้ตรง ๆ**: `lmds start <slug> --port 8001 --gpu-util 0.8` — controller ตรวจค่าเอง
- **flag ของ controller ที่ตั้งได้ตอน start/restart** (ทุก engine: `--port --context --bind --advertise-ip --interface
  --client-input --client-output --extra-args` · vLLM/SGLang/stacked: `--tool-parser --reasoning-parser` · llama.cpp: `--name`
  `--mmproj/--no-mmproj` `--image-min-tokens` `--mtp/--no-mtp`) — help ของ controller แบ่ง Identity / Network / Memory & limits /
  Model features (หมวดสุดท้ายโผล่เฉพาะโมเดลที่มีไฟล์จริง)
- **คำสั่งหลัง start อ่านสถานะจริงจาก `server.meta`** เฉพาะตอนเซิร์ฟเวอร์ยังรันอยู่ · flag ที่ระบุเองชนะ · `start`/`restart` ไม่สืบทอด
- **container ที่ไม่ได้มาจาก lmds**: `discover()` รับเฉพาะ image ที่ตรง engine ที่รู้จัก (vLLM/llama.cpp/Ollama/TGI) `external=True` ·
  `stop` ใช้ `docker stop` · `enable` สร้าง unit `docker start <container>`
- **remove**: หยุด → disable → ลบ bundle+ZIP, `~/.lmds/run/<slug>`, `~/.lmds/plugins/<slug>`, weight (vLLM → HF cache ·
  llama.cpp → `~/models/<slug>` เสมอ ไม่อ่าน `MODEL_DIR` จาก environ · adopt → `MODEL_PROFILE["weights"]`) · หาไม่เจอ = ไม่เดา ·
  ไฟล์ที่ root เป็นเจ้าของ (EACCES) ลบผ่าน `docker run --rm -v <parent>:/x <image> rm -rf` ใต้รั้ว home/HF_HOME ไม่ pull image
- **autostart** = systemd **user** unit `~/.config/systemd/user/lmds-<slug>.service` (ต้องมี linger) · `--system` = system unit
  (`User=<เจ้าของ bundle>` · `ExecStartPre=stop`) · `TimeoutStartSec` ≥ `STARTUP_TIMEOUT`/`HEALTH_TIMEOUT` ของ bundle +300 ·
  `--now` start ทันที · ทุก controller ลงทะเบียนตัวเองใต้ `~/.lmds/run/<slug>/server.meta` ตอน `start` (ไม่มี daemon)
- **control plane** — ไม่มี `llama-server` และไม่มี docker คู่กับ GPU = `download` `repair` `start` `restart` `prepare-runtime`
  ถูกปฏิเสธพร้อมบอกคำสั่ง push · `--force` / `LMDS_ROLE=serving` ทับได้ (ดู FLEET §1.5)

## `lmds adopt`

```text
lmds adopt [CONTAINER] [--port N | --pid N] [--slug NAME] [--output DIR] [--take-over]
```

สร้าง controller จาก**สิ่งที่รันอยู่จริง**: container → `docker inspect` (image · env · mount · port · args · **HostIp** ของ port —
ตัวที่ bind 127.0.0.1 ไม่กลายเป็นเปิดทุก interface) · process → `/proc/<pid>/cmdline`, `exe`, `cwd`, `cgroup`

- **ไม่อ่าน `/proc/<pid>/environ`** (API key อยู่ในนั้น) · **ไม่มี `download` / `verify-files`** ใน controller ที่ได้ — มี `remove-plan`
  และ `status`/`info` พิมพ์ weight ที่ `lmds remove` จะลบ (`MODEL_PROFILE["weights"]`: path/kind/binds)
- **ไม่ปิด unit เดิมให้เอง** — ต้อง `--take-over` · unit เจ้าของถูกจดใน `MODEL_PROFILE.yaml` (`source_process.unit`) controller
  ปฏิเสธ `start` พร้อมบอกคำสั่ง
- `--slug` ต้องเป็นรูป slug (ปฏิเสธ `../../x`) · ชื่อที่เดาจาก container/โมเดลถูกบีบเข้ารูป (org/Model_X → org-model-x)
- หน้าเว็บ: `POST /api/models/{slug}/adopt` — การ์ดที่ยังไม่มี controller มีปุ่ม *Adopt* (container เท่านั้น)

## `lmds node` (fleet หลายเครื่อง)

```text
lmds node add <host> --user <u> [--name] [--port 22] [--note] [--cluster-ip] [--cluster-iface] [--install]
lmds node install [<name>|--all] [--with-prereq]   # hub ส่งโค้ดของตัวเองไป (git bundle ผ่าน scp) → install.sh บนเครื่องนั้น
lmds node setup   [<name>|--all] [--with-prereq]   # ขั้นที่ใช้ sudo (Docker/toolkit/กลุ่ม docker/linger) — รหัสผ่านทาง stdin ใช้ครั้งเดียว
lmds node list [--check]
lmds node set <name> [--cluster-ip] [--cluster-iface] [--note] [--site] [--cluster-name] [--alt-host a,b] [--stack|--no-stack]
lmds node remove <name> [-y]
lmds node run <name> <คำสั่ง lmds...>          # ห่อด้วย bash -lc
lmds node ctl <name> <slug> <คำสั่ง controller...>   # prepare-runtime / download / sync-worker / test-* … · ยืม HF token ทาง stdin
lmds node clone <slug> --from NODE --to NODE [--verify/--no-verify] [--start] [--dry-run] [-y]
lmds node push <name> <slug> [--download] [--start]   # stacked: เขียน cluster.env + pair + sync/verify ก่อน start
lmds node cluster [--write SLUG --head NAME] [--worker NAME] [--on NODE] [--self-stack|--no-self-stack]
```

- **ไม่มี daemon บน node** — hub เรียก `lmds agent info` ผ่าน SSH (`BatchMode=yes` · `ServerAliveInterval`) node เปิดแค่พอร์ต 22 ·
  refresher เขียน `last_seen` ลงทะเบียนทุก 15 วิ
- **ทุก node ต้องมี `lmds`** — `node install` ส่ง git bundle ไป (`git clone -b main` / ff · โฟลเดอร์ไม่ใช่ git → `.bak-<เวลา>` ·
  แยกสาย → branch `local-<เวลา>`) · เครื่องปลายทางไม่ต้องเข้า GitHub · hub ที่ไม่ได้ติดตั้งจาก checkout ถอยไป clone
- **ไม่เก็บรหัสผ่าน** — key ของ hub `~/.config/lmds/id_lmds` (ed25519, comment `lmds-hub`) · `Node` dataclass ไม่มีฟิลด์รหัสผ่าน
- ทะเบียน `~/.config/lmds/nodes.yaml` (0600) เขียนใต้ RLock + `flock` · แก้ host/user/port ผ่าน `set` ไม่ได้โดยตั้งใจ ·
  cluster IP link-local ถูกปฏิเสธ · `alt_hosts:`/`labels:` ว่างในไฟล์ที่แก้มือ = ไม่พัง
- `node cluster` ตรวจ ConnectX/RDMA/ความเร็วลิงก์จาก `/sys` แล้วจับกลุ่มด้วยกุญแจ ไซต์ · ชื่อคลัสเตอร์ · ลายเซ็นฮาร์ดแวร์ แล้วแบ่งย่อยตาม
  subnet · `--site` เป็นป้ายจัดระเบียบและตัวบังคับตอนจับกลุ่ม (คนละไซต์จับคู่กันไม่ได้)
- `_same_commit` เทียบแบบ prefix — `node list` ป้าย `≠ hub` เฉพาะที่ต่างจริง

## `lmds cluster` (stacked)

```text
lmds cluster show [--self-stack|--no-self-stack]                  # = lmds node cluster
lmds cluster write <slug> --head NAME [--worker NAME…] [--on NODE] [--nnodes N]
lmds cluster pair <head> <worker…>
lmds cluster doctor <head> <worker> [--slug SLUG]
```

- **write** อ่าน `NNODES` จาก controller ใน bundle (`nodes/stacked.py`) แล้วตัดกลุ่มให้เหลือ head + worker ตามจำนวนนั้น
  (ระบุ `--worker` เรียง rank ได้ · ไม่พอ/เกิน = ปฏิเสธ) · เขียน `MASTER_IP` `WORKER_IP` `WORKER_IPS` `SSH_USER` `TRANSPORT_IP_*`
  `NNODES` `TENSOR_PARALLEL_SIZE` `NCCL_SOCKET_IFNAME` ลง `cluster.env` บนเครื่องที่ bundle อยู่
- **pair** — กุญแจเกิดบน head (ไม่ผ่าน hub) · public key ลง `authorized_keys` ของ worker · stanza ใน `~/.ssh/config` ของ head
  (`IdentityFile` + `StrictHostKeyChecking accept-new`) ทำซ้ำได้ · ยืนยันด้วย `ssh -o BatchMode=yes`
- **doctor** — อ่านอย่างเดียว · รหัส: `registered` `reachable` `gpu` `opted-out` `same-site` `hardware` `cluster-ip` `same-subnet`
  `iface-up` `link-speed` `ssh-head-to-worker` `fabric-ping` `disk` `bundle-on-head` `cluster-env` `cluster-env-match` ·
  ประโยคสองภาษา (CLI ไทย · เว็บอังกฤษ)
- REST: `POST /api/cluster/write` · `POST /api/cluster/pair` · `GET /api/cluster/doctor` · `PATCH /api/cluster/self` · ปุ่ม
  Pair SSH / Doctor บนหัวกลุ่ม

รายละเอียด: [FLEET-MULTI-NODE.md](FLEET-MULTI-NODE.md)

## `lmds bench`

```text
lmds bench run <SLUG> [--quick|--speed-only|--caps-only] [--runs N]
lmds bench list · show <SLUG> [--history] · remove <SLUG> [--keep-last N]
lmds agent bench          # JSON ให้ hub รวมเป็นตารางฟลีต
```

วัดโมเดล chat ที่รันอยู่ผ่าน OpenAI API — ความเร็ว (TTFT/decode/prefill) + ความสามารถ 7 ข้อ · โมเดล embedding ใช้
`test-embed` ของ controller · `bench`/`stress` ของ controller (vLLM เดี่ยว/stacked) เป็นคนละอย่าง (ดู [BENCH.md](BENCH.md))

## `lmds prune` · `lmds recipes` · `lmds scan`

- **prune** — ล้างทะเบียนที่ชี้ไป bundle ที่ไม่มีแล้วและไม่ได้รันอยู่ · ลบเฉพาะไฟล์ทะเบียน · ตัวที่ไม่เคย start ถูกเก็บกวาดเองตอน `ps`/`list`
- **recipes** — สูตรที่รันผ่านจริง (`src/lmds/recipes/catalog.yaml` + `~/.config/lmds/recipes-synced.yaml`) · ต้องมี `source` +
  `validated_on` · ไม่แตะ `context`/`max_output` · `image_for` ผูก image กับสถาปัตยกรรม · `--sync` อ่านหัวสคริปต์จากรีโป controller
  ของทีม (ไม่รัน) แล้ว**รวมทีละคีย์** (ที่ sync พูดถึงชนะ ที่เงียบไว้คงของ catalog · `engine_env`/`tool_parser`/`reasoning_parser`/
  `extra_args` ระดับบนสุดถูกแปลงให้) · `--publish <slug> --features …` พับค่าที่ `lmds set` ไว้ลง header ส่งขึ้น `recipes.publish_repo`
  (ว่าง = local store)
- **scan** — ค้น `HF_HOME` · `HF_HUB_CACHE` · `TRANSFORMERS_CACHE` · `MODEL_DIR` · `LLAMA_CACHE` · `~/.cache/huggingface[/hub]` ·
  `~/models` · `/models` `/opt/models` `/srv/models` `/data/models` `/mnt/models` · `--root` เพิ่มเอง · `--all` ถามทุกเครื่องพร้อมกัน ·
  อ่านอย่างเดียว · รายงานเลย์เอาต์ HF cache (แบบเก่าต้องตั้ง `HF_HUB_CACHE` — stacked controller ตั้งให้เองตอน start)

## `lmds doctor <slug>`

ตรวจด้วยข้อเท็จจริงบนเครื่อง ไม่ใช้ LLM · exit 0 / 2

ตรวจตามลำดับ: `role` (control plane?) · `controller` · `hf-token` · `weights` (weight หลัก + mmproj อย่างน้อยหนึ่ง · ไฟล์ 0 ไบต์ ·
adopt ตรวจตาม path จริง) · `permissions` · `disk` · `docker` · `image` (มีอยู่ไหม) · `architecture` (vLLM: transformers ใน
image รู้จัก `model_type` · llama.cpp native: `libllama.so` ของ build ที่ pin รู้จัก `general.architecture`) · `grammar`
(llama.cpp มี `cd0fa6051` ไหม — WARN) · `port` (ใครยึด) · `server` · บน control plane ข้อที่แปลว่า "รันไม่ได้" ไม่นับเป็นตัวบล็อก ·
`multimodal` เป็น WARN

## `lmds web`

| ตัวเลือก | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--port` | 8600 | พอร์ตของหน้าเว็บ |
| `--bind` | `127.0.0.1` | `0.0.0.0` = ทั้งวง network (ถาม/สุ่ม token ให้ถ้าไม่ตั้ง `--token`) |
| `--token` | ว่าง | บังคับ token เอง (≥ 8 ตัว ไม่มีช่องว่าง/ตัวควบคุม) |
| `--background` / `-b` | ปิด | รันเบื้องหลัง · รอจนรับ connection ได้จริงก่อนบอกว่าสำเร็จ |
| `--enable` / `--disable` | — | systemd **user** service — ขึ้นเองหลัง reboot ฟื้นเองถ้าตาย (ต้องมี linger) / เลิก |
| `--stop` · `--restart` · `--status` | — | หยุด / เปิดใหม่ (ลิงก์เดิมใช้ได้) / บอกลิงก์ + token ของตัวที่รันอยู่ |
| `--new-token` | — | สุ่ม token ใหม่ — ลิงก์เดิมใช้ไม่ได้ทันที |

- สตาร์ตซ้อนไม่ได้ — พิมพ์ลิงก์ของตัวที่เสิร์ฟจริงแทน · **ที่มาของ token**: `--token` → `$LMDS_WEB_TOKEN` → `~/.config/lmds/web-token`
  (0600) → ถามตอนสตาร์ตครั้งแรก → สุ่ม · ลิงก์ที่พิมพ์ไม่มี token · `?token=` ในลิงก์ถูกย้ายเข้าที่เก็บของเบราว์เซอร์แล้วลบออกจากแถบที่อยู่
- `GET /api/auth` → `{"required"}` · `POST /api/auth` (header `x-lmds-token`) → 200/401 · ผิด >5 ครั้งต่อ IP → 429 หน่วงทวีคูณสูงสุด 60 วิ
- สถานะที่ `~/.lmds/run/web.json` (0600) · `GET /api/version` คืน `commit` · `installed` · **`boot`** (ลายเซ็น process — หน้าเว็บ
  ใช้รอ restart) · restart/update นอก systemd → 409 · shutdown ตั้ง `timeout_graceful_shutdown=3` (SSE ค้างไม่ทำให้โดน SIGKILL)
- `GET /fonts/{name}` — ฟอนต์ Geist ในแพ็กเกจ (allowlist ชื่อ · ไม่ต้องใช้ token · cache 1 ปี)

### หน้าเว็บ 0.6 — app shell

router แบบ hash (`#/overview` `#/node/<ชื่อ>` `#/site/<ไซต์>` `#/models` `#/scores` …) · แถบซ้าย: Overview · This machine · All
machines · ต้นไม้ไซต์ → เครื่อง · Library (Models · Scores · Recipes · Weights · Settings) · การ์ดเครื่องยังอยู่ใน `#nodes` router
แค่เลือก view และซ่อนการ์ดอื่นด้วย `hidden` (id ทุกตัวเดิม — เทสหน้าเว็บ 500+ ข้อไม่แก้) · SSE `/api/events` · หลุดแล้ว poll แคช
inventory ทุก 5 วิ · UI อังกฤษทั้งหมด · ไม่โหลดอะไรจากเน็ต

### API ของผู้ช่วย (หน้าเว็บเรียก — ไม่ใช่ CLI)

| Endpoint | ทำอะไร |
|---|---|
| `POST /api/assistant/chat` | สตรีม SSE: `status` → `evidence` → `ticket` → `delta` |
| `GET /api/assistant/ticket/{id}` · `POST …/choose` · `POST …/advance` | ตั๋วอนุมัติ (apply / step / hold) — จุดเดียวที่งานเริ่มทำงานได้ · อายุ 30 นาที ขั้นละครั้ง |
| `GET /api/models/{slug}/script` · `POST …/script/propose` · `POST …/script/apply` | ผู้ช่วยเสนอแก้ controller (option ก่อน edit · diff · `bash -n` · สำรอง `.bak-…`) |

### REST ที่หน้าเว็บใช้ (token เดียวกับหน้าเว็บ)

```text
GET  / · /api/version · /api/host · /api/models · /api/events (SSE) · /api/targets · /api/fleet/summary
POST /api/restart · /api/update                       # update = git pull --ff-only → install.sh → restart unit ที่รันอยู่จริง

GET  /api/models/{slug}/doctor | logs | settings | settings/suggest | memory | removal-plan
PUT  /api/models/{slug}/settings                       # = lmds set (ผ่าน bundle_settings — SettingsError = 400)
POST /api/models/{slug}/start | stop | restart | adopt | remove | autostart | run/{command} | push/{name}
POST /api/deploy/analyze · GET /api/deploy/{sid}/context · POST /api/deploy/{sid}/generate   # 422 {kind: hub|input|cluster, message}
GET/PUT /api/provider · POST /api/provider/models · POST /api/secrets/hf
GET  /api/recipes · POST /api/recipes/sync · GET /api/scan[?all_nodes=true]
GET  /api/bench · /api/bench/fleet · /api/bench/{slug} · DELETE /api/bench/{slug} · POST /api/bench/{slug}/run
GET  /api/jobs/{id} · POST /api/jobs/{id}/cancel

GET  /api/nodes · POST /api/nodes · PATCH|DELETE /api/nodes/{name} · PUT /api/nodes/order
POST /api/nodes/{name}/install | setup | fix-permissions · GET /api/nodes/{name}/inventory[?refresh=true]
POST /api/nodes/{name}/models/{slug}/{command}        # allowlist: start stop restart repair doctor logs(-n 300) enable disable remove(--dry-run→confirm) set
POST /api/nodes/{name}/models/{slug}/ctl/{command}    # test-text test-vision test-reasoning test-tools test-embed bench stress client-config
                                                      # network-info status props verify-files prepare-runtime sync-worker verify-worker clear-fi-cache logs-worker
POST /api/nodes/{name}/models/{slug}/bench · GET /api/nodes/{name}/bench/{slug} · POST …/bench/{slug}/remove
GET  /api/nodes/{name}/models/{slug}/clone/targets · POST …/clone · GET …/memory · GET …/settings/suggest
GET  /api/cluster · POST /api/cluster/write · POST /api/cluster/pair · GET /api/cluster/doctor · PATCH /api/cluster/self {stack}
```

- option ของ start/restart (`port` `context` `slots` `bind` `api_key` `gpu_util` `tool_parser` `reasoning_parser` `image`
  `served_name` `engine_env` `extra_args`) ผ่าน `jobs.clean_options()` ชุดเดียวกันทั้งโมเดลในเครื่องและบนเครื่องอื่น (400) แล้วแปลงเป็น
  env ของ controller · slug ถูกตรวจรูปแบบทุก route (400) · งานยาว (prepare-runtime · sync-worker · verify-* · bench · stress ·
  download) เป็น job ที่ยกเลิกได้ · ผลงานสดถูกกรอง secret ตั้งแต่ตอนรับแต่ละบรรทัด · `remove` ต้องสองขั้น (`--dry-run` → `{"confirm": "<slug>"}`)
- ต้องมี extra `web` (`fastapi` + `uvicorn` — `install.sh` ลงให้) · รายละเอียดการใช้งาน: [USAGE.md §5](USAGE.md)

## `lmds validate`

รัน quality gates 12 ด่านกับ bundle ใด ๆ (รวม bundle ที่แก้มือ): `bash -n` · template rendered (ไม่มี tag เหลือ) · numeric
underscore · pipefail-safe · line continuation · controller contract v3.0.0 · stacked contract · multimodal assets · profile
schema (+ pinned revision) · serving consistent · secret scan · checksums (`--fix` regenerate `PACKAGE_SHA256SUMS`)

Output: ตาราง pass/fail ต่อ gate + exit `0/2`

## โครงสร้าง config บนเครื่องผู้ใช้

```text
~/.config/lmds/
├── config.yaml          # provider, default target, ui.node_order, cluster.stack_self, recipes.publish_repo
├── credentials          # 0600 — ใช้เมื่อไม่มี keyring
├── nodes.yaml           # 0600 — ทะเบียนเครื่องอื่น (ไม่มีรหัสผ่าน) + cluster IP/site/cluster_name/alt_hosts/stack/last_seen
├── .nodes.lock          # flock ของทะเบียน
├── id_lmds[.pub]        # SSH key ของ hub สำหรับเข้า node
├── web-token            # 0600 — token ของหน้าเว็บ
├── recipes-synced.yaml  # สูตรที่ --sync มา · controllers/<repo> = แคช clone
└── sessions/            # audit log ต่อการ generate (prompt/response/decisions) — redacted
~/.lmds/run/<slug>/      # server.meta · log · runtime.lock · ไฟล์ API key (llama.cpp) · web.json
~/.lmds/bench/<slug>/    # ผล bench ต่อรอบ
~/.local/share/lmds/venv # ตัวโปรแกรม (venv.old ระหว่างอัปเดต) · ~/.local/bin/lmds symlink
```

## โครงสร้าง source (ของจริง ณ 0.6.0)

```text
src/lmds/
├── cli/                 # main.py (typer commands ทั้งหมด: deploy/set/node/cluster/web/…), banner.py
├── config/              # settings.py (config.yaml + provider), paths.py
├── resolver/            # parse.py — HF เท่านั้น (Ollama/NGC โยน UnsupportedSource)
├── inspector/           # inspect.py, hf_api.py (read 120s / connect 30s), gguf.py (header ผ่าน HTTP Range), report.py (task embed)
├── hardware/            # profiler.py (nvidia-smi/docker · gpus_per_node), profiles.py (GPU allowlist · tested)
├── fit/                 # analyzer.py (memory/KV cache · GQA/MLA · per_node + NCCL buffer), targets.py (22 PRESETS)
├── brain/               # providers.py, orchestrator.py, plan_schema.py, prompts.py, rulebased.py, allowlists.py
├── recipes/             # catalog.yaml + sync/publish (รวมทีละคีย์)
├── assistant/           # catalog.py (probe/action), runner.py, policy.py (ตั๋ว), router.py, knowledge.py + playbook.md
├── generator/           # renderer.py + templates/ single-vllm · single-llamacpp · single-sglang · stacked-vllm · README · SPECIAL_FILES
├── validator/           # gates.py — quality gates 12 ด่าน
├── packager/            # bundle.py (PACKAGE_SHA256SUMS + zip)
├── doctor/              # checks.py — role/controller/hf-token/weights/…/architecture/grammar/port/server
├── bench/               # runner.py, workloads.py, capability.py, score.py, store.py
├── fleet/               # manager.py (discover/start/stop/remove/repair/systemd · ลบผ่าน docker), adopt.py, bundle_settings.py (lmds set),
│                        #   clone.py, cluster_env.py, suggest.py (--auto)
├── nodes/               # registry.py (nodes.yaml + lock), ssh.py (key/probe/run/ship git bundle/install script), cluster.py (จับคู่ stacked),
│                        #   cluster_ssh.py (pair), doctor.py (cluster doctor), stacked.py (NNODES ของ bundle)
├── inventory.py         # payload ชุดเดียวที่หน้าเว็บและ `lmds agent info` ใช้ร่วมกัน (+ read_cluster_env)
├── scanner.py           # lmds scan
├── secrets/             # store.py (env/keyring/file), redact.py
├── web/                 # api.py (FastAPI routes), daemon.py (refresher/SSE), deploy.py (analyze/generate · suggest_port), jobs.py
│                        #   (job/cancel/clean_options/_pump scrub), state.py (แคช · decorate_stacked), assistant.py, memory.py,
│                        #   scriptedit.py, selfupdate.py, static/index.html + static/fonts/ (Geist)
└── _build.py            # COMMIT/SOURCE ที่ install.sh ประทับ
tests/                   # ~115 ไฟล์ · 1,720 เทส (unit + E2E + review/audit + JS shell ใน node) · addopts = -q
.github/workflows/ci.yml # pytest 3.10/3.11/3.12 + bash -n/shellcheck + secret scan
```

> ยังไม่มี: `tests/fixtures/` สำหรับ regression เทียบ controllers v3.0.0 (ใช้ `tests/test_v3_regression.py` port กฎ 13 ข้อแทน) ·
> template registry แบบไฟล์ data สำหรับ image digest (ตอนนี้ image มาจาก allowlist ในโค้ด + สูตรใน recipes)
