# `lmds mcp` — ให้ผู้ช่วย AI ถาม hub ผ่านเครื่องมือ แทนการแกะตาราง

LMDS ถูกคุมผ่านผู้ช่วยเขียนโค้ด (Claude Code ฯลฯ) บ่อยกว่าที่คนพิมพ์เอง · ผู้ช่วยรัน `lmds …` แล้วแกะตารางของ rich
ซึ่งถูกตัดตามความกว้างจอ — ชื่อเครื่องยาว ๆ หายครึ่ง ตัวเลขเลื่อนคอลัมน์ · `lmds mcp` คือ
[MCP](https://modelcontextprotocol.io) server ที่ให้มันถามเรื่องเดียวกันผ่านเครื่องมือที่คืน **JSON ก้อนเดียวกับ
`--json` ของคำสั่งนั้น ๆ** (ฟังก์ชันเดียวกัน ไม่ได้คำนวณซ้ำ)

**อ่านอย่างเดียว** — ไม่มีเครื่องมือไหน start/stop/deploy/ลบ/ตั้งค่าอะไรได้ และ process ของ server ถูกผนึกไม่ให้เขียนไฟล์หรือ
สั่งคำสั่งที่เปลี่ยนสถานะ ([รายละเอียด](#สิ่งที่มันจะไม่ทำ-และทำไม)) · protocol เขียนด้วย standard library ล้วน ไม่มี
dependency เพิ่ม · คุยทาง stdin/stdout (JSON-RPC 2.0 หนึ่งข้อความต่อบรรทัด) ไม่เปิดพอร์ต

## เพิ่มเข้าผู้ช่วย

### Claude Code — รันบนเครื่อง hub

```bash
claude mcp add lmds -- lmds mcp          # ใช้ได้ในโปรเจกต์ที่ยืนอยู่
claude mcp add --scope user lmds -- lmds mcp    # ใช้ได้ทุกโปรเจกต์ของ user นี้
claude mcp list                          # ต้องขึ้น  lmds: lmds mcp - ✔ Connected
```

ตรวจแล้ว (2026-10-10 · Claude Code 2.1.251 บน macOS): `claude mcp add lmds -- lmds mcp` แล้ว `claude mcp list` ต่อ stdio ได้
และขึ้น `✔ Connected` — ทดสอบใน config dir ชั่วคราวกับ `lmds` ที่ชี้ไป sandbox ไม่ใช่ hub จริง

### Claude Code บน Mac → hub ที่เป็น OrbStack VM

```bash
claude mcp add lmds -- orb -m <ชื่อ-vm> bash -lc 'cd ~ && ~/.local/bin/lmds mcp'
```

- **ที่ตรวจแล้ว**: รูป `claude mcp add <ชื่อ> -- bash -lc 'cd ~ && <path>/lmds mcp'` (ตัวห่อเดียวกัน ไม่มี `orb -m <vm>`
  ข้างหน้า) — Claude Code เก็บเป็น `command: bash` · `args: ["-lc", "cd ~ && …/lmds mcp"]` และต่อ stdio ได้ `✔ Connected`
- **ที่ยังไม่ได้ตรวจ**: ท่อนที่ผ่าน `orb -m <vm>` จริง (งานนี้ไม่ได้แตะ VM จริง) — สิ่งที่มันต้องทำคือส่ง stdin/stdout ผ่านโดยไม่
  แตะ ซึ่ง `orb -m <vm> <คำสั่ง>` ทำกับคำสั่งทั่วไปอยู่แล้ว · ลองเองครั้งแรกด้วย:

  ```bash
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"ping"}' | orb -m <ชื่อ-vm> bash -lc 'cd ~ && ~/.local/bin/lmds mcp'
  # ต้องได้บรรทัดเดียว: {"jsonrpc":"2.0","id":1,"result":{}}
  ```
- **ถ้า `claude mcp list` ขึ้นว่าต่อไม่ได้**: login shell ของ VM (`bash -l`) พิมพ์อะไรออก stdout ก่อน server ขึ้นหรือเปล่า
  (motd · `echo` ใน `~/.profile` — เคสจริง dgx-70 พ่น `declare -x …` ทุกครั้งที่ login) · บรรทัดเดียวที่ไม่ใช่ JSON คือ
  client ตัดสาย · ลองคำสั่ง `printf` ข้างบน: ถ้ามีบรรทัดอื่นนำหน้า ให้เอา `echo` ออกจาก profile หรือใช้ `bash -c` แทน
  `bash -lc` (ต้องแน่ใจว่า `docker` อยู่ใน PATH ของ shell แบบนั้น)

### hub อยู่อีกเครื่อง (SSH) · client ตัวอื่น

ยังไม่ได้ตรวจกับของจริง — รูปตาม MCP stdio ทั่วไป:

```bash
claude mcp add lmds -- ssh -T ops@hub.example 'bash -lc "lmds mcp"'     # hub อยู่อีกเครื่อง
```

```jsonc
// Claude Desktop (claude_desktop_config.json) · Cursor (~/.cursor/mcp.json) — รูปเดียวกัน
{ "mcpServers": { "lmds": { "command": "lmds", "args": ["mcp"] } } }
```

```toml
# Codex CLI (~/.codex/config.toml)
[mcp_servers.lmds]
command = "lmds"
args = ["mcp"]
```

client ที่ไม่ได้รันบน hub ใช้ตัวห่อแบบเดียวกับข้างบน (`orb -m …` / `ssh -T …`) ในช่อง `command`/`args`

## เครื่องมือ

ทุกตัวอ่านอย่างเดียว · "เท่ากับ" = ก้อน JSON เดียวกับคำสั่งนั้นบนเครื่องนั้น (เทสเทียบให้ทุกตัว)

| เครื่องมือ | คืนอะไร | เท่ากับ | ราคาต่อการเรียก |
|---|---|---|---|
| `lmds_version` | เวอร์ชัน · commit ที่รัน · commit บนดิสก์ · มาตรฐาน/hash ของ template · ไฟล์แก้ค้าง | `lmds version` + `hub` ของ `fleet check --json` | local |
| `lmds_nodes` | ทะเบียนเครื่องทั้งใบ: ชื่อเต็ม · site · host/alt hosts/IP · version+commit · last_seen · last_error · ตัวนับ stale/unknown · restart_pending | `nodes.yaml` (ของที่ hub จำไว้ ไม่ใช่สภาพตอนนี้) | local |
| `lmds_models` `[node]` | ข้อมูลเครื่อง + ทุก bundle: engine · port · running/healthy · context/slots ที่ใช้จริง · autostart · adopted/external · pending_restart · สถานะ controller/runtime · คำขอ 24 ชม. | `lmds agent info` | local · มี `node` = SSH 1 เครื่อง (timeout 30 วิ) |
| `lmds_inspect` `model` | ชนิดไฟล์ · ขนาด · สถาปัตยกรรม · context · ความสามารถ · คำตัดสินรูปแบบที่ไม่รองรับ (`model.unsupported_format`) · fit ต่อ target · คำแนะนำ context | `lmds inspect … --json` | เครือข่าย: metadata จาก Hugging Face (ไม่โหลด weight) |
| `lmds_plan` `model` | แผน deploy แบบ rule-based (engine · image · context · slots · flag) — **ไม่เรียก LLM ไม่ว่ากรณีใด** | `lmds plan … --no-llm --json` | เครือข่าย: Hugging Face |
| `lmds_fit` `slug` `[node]` | ตาราง RAM ของ slots/context ที่ถาม + ค่าที่ *จะ* เขียน — **dry run เสมอ** | `lmds fit <slug> --json` | local · มี `node` = SSH 1 เครื่อง (180 วิ) · bundle เก่าที่ไม่มีมิติ KV ถาม Hugging Face หนึ่งครั้ง |
| `lmds_fleet_check` `[check]` | ทุกเครื่องตรง hub ไหม 3 มิติ (code · controllers · runtime) + สรุป | `lmds fleet check [--check] --json` | local · `check=true` = SSH **ทุกเครื่อง**พร้อมกัน (30 วิ/เครื่อง) |
| `lmds_watchdog_status` `[slug]` `[node]` | watchdog ที่เปิดไว้: policy · probe ล่าสุด · restart · paused/blocked/gave_up · systemd ว่าอย่างไร | `lmds watchdog status --json` | local · มี `node` = SSH 1 เครื่อง (120 วิ) |
| `lmds_logs` `slug` `[node]` `[lines]` | ท้าย log ของโมเดล (ปริยาย 100 บรรทัด · เพดาน 500 · ยาวเกิน 200,000 ตัวอักษรเก็บท้าย) | `lmds logs <slug> -n N` | local · มี `node` = SSH 1 เครื่อง (120 วิ) |
| `lmds_doctor` `slug` `[node]` | ทุกข้อของ doctor (name · ok/warn/fail · detail · คำสั่งแก้) ยกเว้นข้อที่ต้องรัน container ซึ่งอยู่ใน `skipped` | `lmds doctor <slug> --json --no-probe` | local · มี `node` = SSH 1 เครื่อง (120 วิ) |

รูปของคำตอบ: `content[0]` = JSON ของ payload แบบกระชับ · `content[1]` (มีเฉพาะเมื่อมีเรื่องต้องบอก) = `{"notes": […]}` —
คำเตือนที่ CLI จะพิมพ์บน stderr และคำสั่งที่ถูกผนึกกั้นไว้ · ล้มเหลว = `isError: true` พร้อม `{"error": …}` · เครื่องที่ต่อไม่ได้
เป็น error ของเครื่องนั้น (ใน `fleet check` คือ `reachable: false` + `error` ของแถวนั้น เครื่องอื่นยังได้ผลครบ)

ต่างจาก CLI สองจุดโดยตั้งใจ: `lmds_fleet_check` กับ `check=true` **ไม่เขียนผล probe กลับทะเบียน** (`--check` ของ CLI เขียน) และ
`lmds_doctor` ไม่รัน container ชั่วคราวไปถาม image ว่ารู้จักสถาปัตยกรรมของโมเดลไหม (เท่ากับ `--no-probe`)

## สิ่งที่มันจะไม่ทำ และทำไม

`lmds mcp` **จะไม่**: start · stop · restart · remove · deploy/generate · push · clone · install/update · `set` · adopt ·
`bundles refresh` · `watchdog arm/disarm` · enable/disable autostart · แก้ทะเบียนเครื่อง · เขียนไฟล์ใด ๆ บน hub หรือ node ·
เรียก LLM · รับคำสั่ง shell หรือ path จากผู้ช่วย · ไม่มี flag `apply`/`confirm`/`force` ในเครื่องมือไหนเลย

เหตุผล: ผู้ช่วย AI ทำตามสิ่งที่มันอ่าน — model card, README, issue, log ของโมเดลเอง · ข้อความในนั้นสั่งให้มัน
"เรียกเครื่องมือ X ด้วย argument Y" ได้ (prompt injection) · LMDS คุมเครื่องที่มีโมเดลของลูกค้ารันอยู่ · เครื่องมือที่ลบ weight
90 GB หรือ restart โมเดลของลูกค้าได้ ต้องไม่อยู่ห่างจากข้อความในหน้าเว็บแค่หนึ่ง tool call · จะเปลี่ยนอะไร: ผู้ช่วยเสนอคำสั่ง
`lmds …` ให้คนรันเอง (หรือรันผ่าน shell ที่เจ้าของอนุญาต ซึ่งมีการถามสิทธิ์ของ client เอง)

### อ่านอย่างเดียวเป็นโครงสร้าง ไม่ใช่ธรรมเนียม

สามชั้น ไม่มีชั้นไหนพึ่งชั้นอื่น:

1. **ทะเบียนเครื่องมือที่เดียว** (`src/lmds/mcp/tools.py`) ผูกได้เฉพาะฟังก์ชันใน `reads.READ_ALLOWLIST` — ผูกอย่างอื่น
   (`fleet.start_server`, `os.remove`, ฟังก์ชันชื่อเดียวกันจากโมดูลอื่น) = import ล้ม server ไม่ขึ้น
2. **process ถูกผนึก** (`src/lmds/mcp/seal.py` — audit hook ของ Python, PEP 578) ก่อนอ่านคำขอแรก และถอดไม่ได้:
   - เปิดไฟล์เพื่อเขียน · ลบ · ย้าย · mkdir · chmod → ถูกกั้น (`PermissionError` แบบดิสก์ read-only)
   - spawn ได้เฉพาะโปรแกรมในรายการอ่าน: `docker`/`systemctl`/`loginctl`/`git` เฉพาะท่าอ่าน (`ps` `inspect` `logs` `info` ·
     `is-active` `show` · `rev-parse` `status` …) · ตัวถามสภาพเครื่อง (`nvidia-smi` `ip` `ss` `pgrep` …) ·
     อย่างอื่นทั้งหมดถูกกั้น รวม `bash -c`, `scp`, `rsync`, `sudo` และ **controller ของ bundle ทุกคำสั่ง**
   - `ssh` ออกได้เฉพาะคำสั่งในรายการตายตัว `REMOTE_READS` (เทียบเต็มบรรทัด): `lmds agent info` · `lmds fit <slug> --json
     [--slots N] [--context N]` · `lmds logs <slug> -n N` · `lmds doctor <slug> --json --no-probe` · `lmds watchdog status
     [<slug>] --json` — ทุกตัวนำหน้าด้วย `LMDS_READ_ONLY=1`
   - ส่งสัญญาณไป process อื่นไม่ได้ (ยกเว้น `kill -0` และการฆ่าลูกของตัวเองเมื่อหมดเวลา)
3. **เทส** (`tests/test_mcp_*.py`) เปิด server จริงทาง stdio บนฟลีตจำลอง เรียกทุกเครื่องมือ แล้วเทียบทุกไฟล์ของ hub ก่อน/หลัง
   กับบันทึกคำสั่งที่ `ssh`/`docker` ปลอมได้รับ · อีกชุดผนึก process แล้วลองทางเขียนตรง ๆ 43 ทาง ต้องถูกกั้นทั้งหมด

ชั้นที่ 2 มีเพราะ **ฟังก์ชัน "อ่าน" ของ LMDS เขียนเงียบ ๆ อยู่จริง** — สำรวจตอนสร้าง (2026-10-09) แต่ละข้อมีเทสที่แสดงว่า CLI
เขียนจริงและเครื่องมือไม่เขียน:

| ทางอ่านที่เขียน | ผ่าน CLI/หน้าเว็บ | ผ่าน `lmds mcp` |
|---|---|---|
| `fleet.discover()` — ทุกคำสั่งที่ลิสต์โมเดล | ลบทะเบียนของ bundle ที่ไม่เคย start และ controller หายแล้ว | ถูกกั้น — ทะเบียนอยู่ที่เดิม (ไม่นับเป็นโมเดลเหมือนกัน) |
| `inventory.request_usage()` — `agent info` | เขียน `run/<slug>/usage.samples` ทุกครั้ง | ถูกกั้น — อ่านชุดเดิมมาคิดหน้าต่าง 24 ชม. ได้ตามปกติ |
| `brain.build_plan()` แบบ rule-based — `plan --no-llm` | เขียน session log ลง config dir | ถูกกั้น |
| `git status` ของ "hub มีไฟล์แก้ค้างไหม" | เขียน index ของ repo | `GIT_OPTIONAL_LOCKS=0` |
| `lmds doctor` ข้อ architecture | `docker run --rm` ถาม image แล้วจดผลลง `run/<slug>/runtime-arch.json` | ไม่ถาม (`probe=False`) — ข้อนั้นอยู่ใน `skipped` |
| `lmds fleet check --check` | เขียน last_seen/last_error/ตัวนับลงทะเบียนเครื่อง | ตัดสินจาก Node ในหน่วยความจำ — รายงานเท่ากัน ทะเบียนไม่เปลี่ยน |
| `<controller> logs N` (llama.cpp native) | ลบ `server.pid` ที่มันเห็นว่าค้าง — ทุกคำสั่งของ controller ผ่าน `_our_server_pid` | ไม่รัน controller — อ่าน `docker logs` / `server.log` ตรง ๆ |
| `lmds fit` ของ bundle เก่าที่ไม่มีมิติ KV | ถาม Hugging Face หนึ่งครั้ง (จำในหน่วยความจำ) | เหมือนกัน — ยอมรับไว้ (ไม่เขียนอะไร · บอกในคำอธิบายเครื่องมือ) |

### บนเครื่องอื่น (node)

hub ส่งคำสั่งอ่านไปเป็น `LMDS_READ_ONLY=1 lmds …` · **`lmds` รุ่นนี้ขึ้นไปผนึก process ของตัวเองเมื่อเห็นตัวแปรนี้** (ผนึกเดียวกับ
ข้อ 2) ก่อนเข้าคำสั่ง — บนเครื่องนั้นจึงไม่มีการเขียนเช่นกัน · `lmds` รุ่นเก่ากว่าไม่รู้จักตัวแปรนี้และทำงานตามเดิม:

- `lmds agent info` — เขียน `usage.samples` และเก็บกวาดทะเบียนที่ตายแล้ว (สิ่งเดียวกับที่หน้าเว็บของ hub ทำให้เกิดทุก 15 วิ)
- `lmds logs` — รัน controller ซึ่งลบ `server.pid` ที่ค้าง (สิ่งเดียวกับปุ่ม log ของหน้าเว็บ)
- `lmds doctor --json --no-probe` · `lmds watchdog status --json` (ตัวที่ตอบเป็นประโยคเมื่อไม่มีตัวไหนเปิด hub อ่านเป็น `[]`) —
  เครื่องที่ยังไม่มี flag ได้ error "อัปเดตก่อน: `lmds node install <ชื่อ>`"

อยากให้ไม่มีข้อยกเว้นเลย: อัปเดตทุกเครื่องให้ถึงรุ่นนี้ (`lmds node install --all`) — `lmds_fleet_check` บอกว่าเครื่องไหนยังตามหลัง

`LMDS_READ_ONLY=1` ใช้เองได้: `LMDS_READ_ONLY=1 lmds ps` รันได้ · `LMDS_READ_ONLY=1 lmds stop <slug>` ล้มพร้อมเหตุผล

## argument ไม่ถูกไว้ใจ

ทุก argument ถูกตรวจก่อนมีคำสั่งไหนถูกประกอบ ไม่ผ่าน = `isError` + `"refused": true`:

- `node` ต้องเป็นชื่อที่อยู่ในทะเบียนของ hub เป๊ะ ๆ (ไม่ใช่ host ไม่ใช่ IP ไม่ใช่ชื่อที่ "ดูถูกต้อง")
- `slug` ต้องผ่าน `shellsafe.BUNDLE_SLUG` — ตัวเดียวกับปากทางของหน้าเว็บ (`a-z A-Z 0-9 . _ -` ไม่เกิน 64)
- `model` ผ่าน `resolver.parse_source` ตัวเดียวกับ `lmds inspect` · `revision` เป็นชื่อ branch/tag/commit ธรรมดา
- `target` · `engine` · `kv_dtype` ต้องอยู่ในรายการของโปรเจกต์ · ตัวเลขมีช่วง · ข้อความมีความยาวสูงสุด · ไม่รับคีย์ที่ไม่รู้จัก
- คำตอบใหญ่เกิน 600,000 ตัวอักษรไม่ถูกส่ง (ได้ error ให้ถามแคบลง)

## ความลับในคำตอบ

คำตอบทุกก้อนผ่านตัวกรองก่อนออก (`src/lmds/mcp/redaction.py`) — คำตอบไปถึง LLM provider และค้างใน transcript:

- ตัวกรองเดียวกับ log ของงานบนหน้าเว็บ (`secrets.redact`): รูป `sk-…` · `hf_…` · `AIza…` · `Bearer …`
- ค่าที่ hub รู้: API key ของทุก bundle (`lmds key`) · token ของหน้าเว็บ · ไฟล์ credentials · HF token · keyring · ลายเซ็นไลเซนส์
- ค่าที่ตามหลังชื่อลับ (`--api-key X` · `API_KEY=X` · `'api_key': ['X']`) และรหัสผ่านใน URL — สำหรับ log ของเครื่องอื่นที่ hub
  ไม่รู้ key ของมัน

**ไม่ได้ปิด** (เป็นข้อมูลที่เครื่องมือมีไว้ตอบ): ชื่อเครื่อง · IP · ชื่อ user ของ SSH · path · ชื่อโมเดล · ข้อความใน log ที่ไม่มีรูปของ
key · ถ้าสิ่งเหล่านี้เป็นความลับของไซต์ อย่าต่อ MCP server นี้กับผู้ช่วยที่ส่งข้อมูลออกนอกองค์กร · ตัวกรองแบบรูปร่างพลาดได้กับ key
ที่ไม่มีทั้งชื่อนำและรูปที่รู้จัก — log ของ node ที่พิมพ์ key เปล่า ๆ กลางประโยคจะหลุด

## เมื่อมีปัญหา

| อาการ | สาเหตุ / ทางออก |
|---|---|
| client ต่อไม่ได้ / "Connection closed" | รัน `lmds mcp </dev/null` เอง ต้องจบเงียบ ๆ ด้วย exit 0 · ผ่านตัวห่อ (`orb`/`ssh`/`bash -lc`) ดูหัวข้อ OrbStack ข้างบน |
| คำตอบมี `notes: ["read-only guard refused: …"]` | ฟังก์ชันอ่านพยายามรันโปรแกรมนอกรายการ — คำตอบอาจขาดข้อมูลส่วนนั้น · แจ้งพร้อมบรรทัดนั้น (ต้องเพิ่มใน `seal.py` หรือเป็นบั๊ก) |
| `…ยังไม่มี lmds doctor --json…` | LMDS บนเครื่องนั้นเก่ากว่า hub — `lmds node install <ชื่อ>` |
| เครื่องมือช้า | `lmds_fleet_check` กับ `check=true` รอเครื่องที่ช้าที่สุด (สูงสุด 30 วิ) · เครื่องมือทำทีละคำขอ (คิวเดียว) — `ping` ยังตอบระหว่างรอ |
| อยากเห็นว่า server ทำอะไร | stderr ของ process (client เก็บไว้ใน log ของ MCP server) — บรรทัดขึ้นต้นด้วย `lmds-mcp:` รวมทุกอย่างที่ถูกกั้น |

## protocol

- รุ่นที่ตอบ: `2025-11-25` · `2025-06-18` · `2025-03-26` · `2024-11-05` (ขอรุ่นอื่นได้รุ่นใหม่สุด)
- รับ `initialize` · `notifications/*` (ไม่ตอบ) · `ping` · `tools/list` · `tools/call` · batch ตอบรายข้อ
- error: `-32700` JSON พัง · `-32600` ไม่ใช่ JSON-RPC 2.0 · `-32601` ไม่รู้จัก method · `-32602` params ผิดรูป/ไม่รู้จักเครื่องมือ ·
  argument ที่ไม่ผ่านการตรวจเป็น `isError` ของเครื่องมือ (ตามสเปก 2025-11-25 — ให้ผู้ช่วยแก้แล้วเรียกใหม่ได้)
- ไม่มี resources/prompts/sampling · ไม่มี transport แบบ HTTP/SSE · ไม่มี auth ของตัวเอง (ใครรัน `lmds mcp` ได้คือคนที่ login เครื่อง
  hub ได้อยู่แล้ว และอ่านไฟล์ชุดเดียวกันได้เอง)

## เพิ่มเครื่องมือ (สำหรับคนแก้โค้ด)

1. ฟังก์ชันอ่านใน `reads.py` — เรียกฟังก์ชันเดียวกับ `--json` ของ CLI (ไม่มีก็แยกออกมาจาก CLI ก่อน) แล้วเพิ่มชื่อใน `READ_ALLOWLIST`
2. แถวใน `tools.TOOLS` — คำอธิบายบอกว่าคืนอะไร · อ่านอย่างเดียว · ราคา · argument ทุกตัวมีตัวตรวจ
3. ถามเครื่องอื่น = เพิ่มบรรทัดใน `seal.REMOTE_READS` (คำสั่งอ่านของ `lmds` เท่านั้น เทียบเต็มบรรทัด)
4. เทส: ผลเท่ากับ `--json` · เพิ่มใน `EVERY_CALL` ของ `tests/test_mcp_readonly.py` (เทสไม่เปลี่ยนสถานะกับเทสความลับวิ่งผ่านมันเอง)

เครื่องมือที่เปลี่ยนสถานะ **ไม่เพิ่มที่นี่** — ถ้าวันหนึ่งต้องมี ต้องเป็น server คนละตัวที่ไม่ผนึก มีการยืนยันของคนต่อการเรียก และ
ลง `lmds audit`
