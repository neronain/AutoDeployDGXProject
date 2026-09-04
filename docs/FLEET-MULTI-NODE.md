# คุมหลายเครื่องจากเครื่องเดียว (Fleet + Cluster)

> เอกสารนี้ว่าด้วย **หลายเครื่อง** — ทั้งแบบต่างคนต่างรัน (fleet) และแบบหลายเครื่องรวมเป็นโมเดลเดียว (stacked)
> การจัดการโมเดลหลายตัวใน**เครื่องเดียว** อยู่ที่ [USAGE.md §4](USAGE.md)

---

## 1. สองคำที่ต้องแยกให้ออกก่อน

| | **Fleet** | **Stacked (cluster)** |
|---|---|---|
| หลายเครื่องทำอะไร | ต่างคนต่างรันโมเดลของตัวเอง | รวมกันเป็นโมเดล **ตัวเดียว** (tensor-parallel ข้ามเครื่อง) |
| เครื่องหนึ่งดับ | เครื่องอื่นไม่กระทบ | ทั้งคลัสเตอร์ล้ม |
| ต้องการสายเร็ว | ไม่ต้อง (SSH ก็พอ) | **ต้อง** — 200G RoCE/ConnectX (อย่างน้อย 25G) |
| ฮาร์ดแวร์ต้องเหมือนกัน | ไม่ต้อง | **ต้อง** — GPU รุ่นเดียวกัน จำนวนเท่ากัน |
| ใช้เมื่อ | โมเดลลงเครื่องเดียวได้ อยากกระจายงาน | โมเดลใหญ่เกินหน่วยความจำเครื่องเดียว |

LMDS ทำให้ทั้งสองแบบเห็นจากที่เดียวกัน: **hub** คือเครื่องที่คุณนั่งอยู่ ส่วน **node** คือเครื่องอื่นในทะเบียน

---

## 1.5 เครื่องนี้มีไว้ทำอะไร — control plane vs เครื่องรันโมเดล

LMDS ตรวจเองว่าเครื่องที่คุณนั่งอยู่ **รันโมเดลได้จริงไหม** โดยดูจากของที่มีอยู่ ไม่ใช่ชื่อเครื่อง:

| เห็นอะไร | สรุปว่า |
|---|---|
| `llama-server` ที่รันได้ (`~/src/llama.cpp/build/bin/` หรือใน PATH) | รัน engine `llamacpp` ได้ |
| docker **คู่กับ GPU** | รัน `vllm` / `sglang` / `trtllm` ได้ |
| ไม่มีสักอย่าง | **control plane** — สร้าง bundle แล้ว push ต่อ |

บน control plane คำสั่งพวกนี้ถูกปฏิเสธ: `download` `repair` `start` `restart` `prepare-runtime`

เหตุผล: เคสจริง 2026-08-19 `lmds repair` บน hub VM (ไม่มี GPU/docker/llama.cpp, RAM 12 GB)
เริ่มดูด weight 15.6 GB ลงมาอย่างว่าง่าย — ไฟล์ที่ต่อให้โหลดจบก็ไม่มีอะไรรันมันได้

สิ่งที่ทำแทน — bundle ตัวเดิมที่คุณอนุมัติแผนไปแล้ว ถูกส่งไปให้เครื่องที่รันได้โหลด weight ของมันเอง:

```bash
lmds node push spark-head my-model --download   # ส่ง bundle แล้วสั่งโหลดที่ปลายทาง
lmds node run spark-head start my-model
```

คำสั่งที่แค่ *อ่าน* ของที่มีอยู่ยังใช้ได้ตามปกติบน hub: `verify-files` `status` `doctor` `logs` `stop`

**เมื่อการตรวจเดาผิด** (เช่นกำลังจะ build llama.cpp ทีหลัง):

```bash
lmds repair my-model --force          # ครั้งเดียว
LMDS_ROLE=serving lmds repair my-model # ทั้ง session
LMDS_ROLE=hub lmds repair my-model     # ตรงข้าม: เครื่องมี GPU แต่ตั้งใจให้เป็น hub
```

`lmds doctor` ขึ้นข้อ **บทบาท** เป็นข้อแรกเสมอ และบน control plane ข้อที่แปลว่า "รันไม่ได้"
(`docker` `image` `architecture` `grammar` `weights` `server`) ไม่นับเป็นตัวบล็อก

## 2. สถาปัตยกรรม — ทำไมไม่มี daemon

hub คุย node ผ่าน **SSH เท่านั้น** แล้วเรียก `lmds agent info` บนเครื่องนั้นเพื่อขอสถานะกลับมาเป็น JSON

```text
   hub (เครื่องที่คุณใช้)                    node
   ┌──────────────────┐   ssh (key)    ┌──────────────────┐
   │ lmds web / CLI   │ ─────────────► │ lmds agent info  │ → JSON
   │ ~/.config/lmds/  │ ◄───────────── │ (ไม่มี daemon)    │
   │   nodes.yaml     │                └──────────────────┘
   └──────────────────┘
```

ผลที่ตามมา — เป็นการตัดสินใจ ไม่ใช่ข้อจำกัด:

- **node ไม่ต้องเปิดพอร์ตเพิ่มนอกจาก 22** ไม่มีเซอร์วิสใหม่ให้ patch ไม่มี agent ให้ค้างเป็นซอมบี้
- **hub ล่ม node ยังรันต่อ** โมเดลบน node ไม่ได้พึ่ง hub เลย
- **node ล่ม หน้าเว็บไม่พัง** — แถวนั้นขึ้นว่า unreachable แล้วจบ ไม่ค้างทั้งหน้า
- ทุกอย่างที่ hub แสดง มาจาก payload ชุดเดียวกับที่หน้า "เครื่องนี้" ใช้ (`lmds.inventory`) — ตัวเลขจึงหมายถึงสิ่งเดียวกันเสมอ

---

## 3. เพิ่มเครื่องเข้าทะเบียน

### ⚠️ ทุกเครื่องต้องมี LMDS ติดตั้งอยู่บนเครื่องนั้น

hub **ไม่ได้ส่ง agent ไปรัน** บนเครื่องปลายทาง — มันเรียก `lmds agent info` ที่ติดตั้งอยู่บนเครื่องนั้น
ผ่าน SSH ฉะนั้น "agent" ของระบบนี้ = ตัวคำสั่ง `lmds` เอง ไม่ใช่โปรเซสแยกที่ต้องรันค้างไว้

ข้อดีของการออกแบบแบบนี้: ไม่มี daemon ให้ค้างเป็นซอมบี้ ไม่ต้องเปิดพอร์ตเพิ่ม และ node เวอร์ชันต่างกัน
ก็ยังคุยกันได้ · ข้อแลกเปลี่ยน: **ต้องติดตั้งบนทุกเครื่องก่อน** ซึ่ง hub ทำให้ได้ด้วยคำสั่งเดียว:

```bash
lmds node install <ชื่อเครื่อง>      # hub ส่งโค้ดของตัวเองไป (git bundle ~2 MB ผ่าน scp) แล้วรัน install.sh บนเครื่องนั้น
                                     # — เครื่องนั้นไม่ต้องเข้า GitHub · hub ที่ไม่ได้ติดตั้งจาก checkout ถอยไป clone เอง
lmds node install --all              # อัปเดตทุกเครื่อง (แคช bundle ต่อ commit — pack ครั้งเดียว)
lmds node add <ip> --user <u> --install   # เพิ่ม + ติดตั้งในคำสั่งเดียว
lmds node setup <ชื่อ> --with-prereq  # ขั้นที่ใช้ sudo (Docker / toolkit / กลุ่ม docker / linger) — ถามรหัสตอนนี้ ใช้ครั้งเดียว
```

สิ่งที่ `node install` ทำบนเครื่องปลายทาง (สคริปต์รันกับ git จริงในเทส `tests/test_install_ship.py`):

- ยังไม่มี checkout → `git clone -b main <bundle>` (bundle มีแต่ ref `main` ไม่มี HEAD — clone เฉย ๆ ได้โฟลเดอร์เปล่า)
  แล้วชี้ origin กลับ GitHub เผื่อวันหน้าเครื่องนั้นมี key เอง
- มี checkout → `git fetch <bundle> main` แล้ว ff · **แก้ไว้/แยกสาย** → เก็บที่ branch `local-<เวลา>` + stash แล้วตามโค้ดของ hub
  (node เป็นของ hub ไม่ใช่ที่พัฒนาโค้ด) · **โฟลเดอร์ที่ไม่ใช่ git** (ติดตั้งแบบ copy) → ย้ายไป `.bak-<เวลา>` ก่อน clone
- รัน `install.sh` (`LMDS_SKIP_PREREQ=1`) — pip ล้ม = venv เดิมถูกคืน ไม่ทิ้ง node ไว้แบบไม่มี `lmds`
- สรุปว่า "ตรง hub" / "ยังไม่ตรง hub (hub: …)" โดยเทียบ commit แบบ prefix (git ย่อ 7 หรือ 8 ตัวต่างกันตามจำนวน object)

`node install` **ข้ามขั้น Docker/NVIDIA toolkit** เป็นค่าเริ่มต้น เพราะขั้นนั้นต้องใช้ `sudo`
ซึ่งไม่มีคนกรอกรหัสผ่านให้ผ่าน SSH — ใช้ `lmds node setup <ชื่อ> --with-prereq` (รหัสผ่านส่งทาง stdin ของ ssh ไม่เขียนดิสก์
ไม่อยู่ใน argv) หรือ *Add machine* บนหน้าเว็บที่ทำครบในครั้งเดียว (หรือ `--with-prereq` ของ `node install` ถ้าเครื่องนั้น sudo
ผ่านโดยไม่ถามรหัสผ่าน)

### สรุปสิ่งที่ต้องมีบนเครื่องปลายทาง

| ต้องมี | ใครจัดการได้ |
|---|---|
| sshd เปิดอยู่ + user ที่ login ได้ | ผู้ดูแลเครื่องนั้น |
| user อยู่ในกลุ่ม **`docker`** (ไม่ต้องเป็น root) | `lmds node setup` / Add machine (ต้องรหัส sudo ครั้งเดียว) |
| `git` + `python3` (clone bundle ที่ hub ส่งมา · `python3-venv` ขาดก็ลงให้/ถอยไป `--without-pip`) | ปกติมีอยู่แล้วบน Ubuntu/DGX OS |
| Docker + NVIDIA Container Toolkit | `lmds node setup --with-prereq` จาก hub · หรือ `./install.sh` บนเครื่องนั้น |
| **LMDS (`lmds` ใน PATH)** | **hub ทำให้ได้: `lmds node install`** |

> **ไม่ต้องใช้ root** — LMDS ไม่เคยต้องการสิทธิ์ root ในการรันโมเดล · `lmds enable` เป็น systemd *user* service
> (ไม่ต้อง sudo · ต้องมี linger ซึ่ง `node setup` ตั้งให้) · sudo ใช้เฉพาะขั้นติดตั้ง Docker/toolkit และ `enable --system`

### CLI

```bash
lmds node add 192.168.10.21 --user ops
```

จะถามรหัสผ่าน **ครั้งเดียว** เพื่อติดตั้ง SSH key ของ LMDS (`~/.config/lmds/id_lmds`, ed25519, comment `lmds-hub`)
แล้วรหัสผ่านถูกทิ้งทันที — **ทะเบียนไม่มีฟิลด์รหัสผ่านโดยตั้งใจ** (มีเทสกันไม่ให้เผลอเพิ่มกลับเข้ามา)

ตัวเลือกอื่น: `--name` ตั้งชื่อเอง · `--port` · `--note` · `--cluster-ip` / `--cluster-iface` (ดูข้อ 5)

### หน้าเว็บ

**Other machines → + Add machine** — กรอก IP, user, รหัสผ่าน แล้วกด Add · ผลเหมือน CLI ทุกอย่าง

### จัดลำดับการ์ดเอง

จับ `⠿` ที่หัวการ์ดแล้วลากขึ้นลงได้เลย (ที่จับแยกต่างหาก เพราะหัวการ์ดมีปุ่มและช่อง cluster IP
อยู่ กดตรงนั้นต้องได้คลิกตามปกติ) · ใช้ pointer events จึงลากด้วยนิ้วบนแท็บเล็ตได้ด้วย

ลำดับเก็บที่ **hub** ไม่ใช่ในเบราว์เซอร์ (`config.yaml` → `ui.node_order`) — เปิดจากเครื่องไหน
หรือบราว์เซอร์ไหนก็เห็นเหมือนกัน และ `lmds node list` / `lmds node cluster` เรียงตามลำดับเดียวกัน
· เครื่องที่เพิ่งเพิ่มต่อท้ายอัตโนมัติ · เครื่องที่ลบไปแล้วหลุดจากลำดับเอง

กลุ่ม stacked ยังคงรวมกันเหมือนเดิม แต่ไปรวมที่ตำแหน่งของ**สมาชิกตัวแรกตามลำดับที่คุณลาก**
และสลับลำดับภายในกลุ่มได้ — ซึ่งมีผลจริง เพราะสมาชิกตัวแรกคือเครื่องที่ถูกเสนอเป็น **head**
ตอน `lmds node cluster --write` (hub อยู่หน้าสุดเสมอ เพราะไม่มีการ์ดให้ลาก)

### ถอนเครื่องออก

```bash
lmds node remove spark2
```

ลบออกจากทะเบียนอย่างเดียว **ไม่แตะอะไรบนเครื่องนั้น** — key ยังอยู่ ถ้าจะถอนให้ลบบรรทัดที่ลงท้าย
ด้วย `lmds-hub` ออกจาก `~/.ssh/authorized_keys` ของเครื่องนั้นเอง (คำสั่งบอกไว้ตอน remove)

---

## 4. ดูและสั่งงานข้ามเครื่อง

```bash
lmds node list              # ทะเบียนทั้งหมด (เร็ว — อ่านไฟล์อย่างเดียว)
lmds node list --check      # ต่อจริงทุกเครื่องเพื่อดูว่ายังตอบไหม
lmds ps --all               # โมเดลของทุกเครื่องรวมกันในตารางเดียว
lmds node run spark2 doctor my-model   # รันคำสั่ง lmds อะไรก็ได้บนเครื่องนั้น
```

ในหน้าเว็บ แต่ละเครื่องแสดงทรัพยากรสดเป็นแถบ pill:

| ค่า | มาจาก | หมายเหตุ |
|---|---|---|
| **CPU** | `os.cpu_count()` + `/proc/loadavg` | เป็น load 1 นาทีคิดเป็น % ของทั้งเครื่อง · เกิน 100% = มีงานรอคิว (แสดงตรง ๆ ไม่ตัดเพดาน) |
| **RAM / Unified** | `/proc/meminfo` | DGX Spark เป็น unified memory — CPU กับ GPU ใช้ pool เดียวกัน จึงแสดงเป็นก้อนเดียว |
| **VRAM** | `nvidia-smi memory.used/total` | การ์ดแยก (RTX) เท่านั้น · GB10 ไม่รายงานค่านี้ → แสดง "shared" ไม่ใช่ 0 |
| **Disk** | `statvfs` ที่ `$HOME` | ที่เก็บ weight จริง |
| **Link** | `/sys/class/net/*/speed` | สายที่เร็วที่สุดที่ลิงก์ขึ้น + ป้าย RDMA |
| **IP ของเครื่อง** | `ip -o -4 addr` (สำรอง: `ifconfig`) | IP ที่ **เครื่องนั้น** ถืออยู่จริง — คนละอย่างกับที่อยู่ที่ hub ใช้ SSH ซึ่งเป็นชื่อได้ · ดู [NETWORK.md](NETWORK.md#เครื่องนั้นอยู่-ip-ไหน) |
| **Models running** | `lmds agent info` | เป็น **จำนวน** ไม่ใช่ใช่/ไม่ใช่ — llama.cpp รันหลายโมเดลพร้อมกันได้ (คนละพอร์ต) |

ปุ่มต่อโมเดลบน node จำกัดด้วย allowlist ฝั่ง server (ไม่ใช่แค่ซ่อนปุ่ม) — คำสั่งของ `lmds`: `start stop restart repair
doctor logs enable disable remove set` (`remove` ต้องผ่าน `--dry-run` ก่อนแล้วส่ง confirm = slug) และคำสั่งของ controller
ที่อ่าน/ทดสอบเท่านั้น: `test-*` `parsers` `bench` `stress` `client-config` `network-info` `status` `props` `verify-files`
`prepare-runtime` `sync-worker` `verify-worker` `clear-fi-cache` `logs-worker` — หน้าเว็บสั่งข้ามเครื่องได้ จึงต้องแคบไว้ก่อน ·
slug จาก URL ถูกตรวจรูปแบบที่ปากทางทุก route (400) ก่อนไปถึง shell ของเครื่องอื่น · งานที่ ssh ค้างยกเลิกได้ (ปุ่ม Cancel /
`POST /api/jobs/{id}/cancel`)

### ถามผู้ช่วยแทนการไล่กดเอง

พอเครื่องเยอะขึ้น การไล่เปิด log ทีละเครื่องเพื่อหาว่าตัวไหนมีปัญหาเริ่มไม่คุ้มเวลา ·
กล่องแชทในหน้าเว็บ **ลงไปดูเครื่องให้ก่อนตอบ** โดยใช้ทางเดียวกับ `lmds node run` (SSH key
ของ LMDS ทะเบียนเดียวกัน) — ถามได้ตรง ๆ ว่า "ทำไม qwen3-coder บน spark-head ไม่ขึ้น"
แล้วมันจะเปิด log ของ controller ตัวนั้น ดู RAM/ดิสก์/พอร์ต หรือรัน `lmds doctor` ให้

- สิ่งที่มันดูได้เป็น **แคตตาล็อกปิด** (`src/lmds/assistant/catalog.py`) ไม่ใช่คำสั่งอิสระ —
  ขอบเขตเดียวกับ allowlist ข้างบน: แคบไว้ก่อนเพราะสั่งข้ามเครื่องได้
- งานที่ **เปลี่ยนสภาพเครื่อง** (restart, เปลี่ยน context/port/bind/gpu-util) มันเสนอได้
  แต่ลงมือเองไม่ได้ · คุณเลือกจากเมนู **แก้เลย / ทีละขั้น / ยังไม่ทำ** โดยเห็นคำสั่งเต็ม
  และผลกระทบก่อนกด
- เครื่องที่ต่อไม่ติดจะรายงานว่าต่อไม่ติด แล้วจบ — ไม่ทำให้คำตอบทั้งอันล้ม

รายละเอียดการใช้งานอยู่ใน [USAGE — ถามผู้ช่วยให้ไปดูเครื่องให้](USAGE.md) · กติกาการอนุมัติ
อยู่ใน [SECURITY.md](../SECURITY.md)

---

## 5. Cluster fabric — เครื่องไหน stacked ด้วยกันได้

### ระบบตรวจอะไรให้บ้าง

อ่านจาก `/sys` ล้วน ไม่ต้อง root ไม่ต้องมี `ethtool`/`ibstat`:

- ความเร็วลิงก์ (`/sys/class/net/*/speed`) — อ่านได้เฉพาะตอนลิงก์ **ขึ้น**
- ยี่ห้อการ์ด (vendor `0x15b3` = Mellanox/ConnectX) และ driver (`mlx5_core` ฯลฯ)
- อุปกรณ์ RDMA (`/sys/class/infiniband/*`)
- IP ของแต่ละ interface (`ip -o -4 addr`)

แล้วสรุปเป็น tier:

| tier | เงื่อนไข | ความหมาย |
|---|---|---|
| `rdma` | ≥100G + มี RDMA device | stacked ได้เต็มที่ |
| `fast` | ≥25G | stacked ได้ แต่ถ้ายังไม่เปิด RoCE ควรเปิดก่อน |
| `basic` | <25G | **อย่า stacked** — ช้ากว่ารันแยกเครื่อง |
| `unknown` | ตรวจไม่ได้ | รายงานว่าตรวจไม่ได้ ไม่เดา |

> เกณฑ์ **25G** (`MIN_STACK_GBPS`) ไม่ใช่ตัวเลขสวย ๆ — ต่ำกว่านี้ activation/KV ที่วิ่งข้ามเครื่องทุก token
> จะกินเวลามากกว่าที่ประหยัดได้จากการแบ่งโมเดล

### เงื่อนไขการจับกลุ่ม

สองเครื่องจะถูกเสนอเป็นคู่ stacked ก็ต่อเมื่อ **ตรงกันทุกข้อ**: สถาปัตยกรรม (arch), profile,
**ชื่อรุ่น GPU**, **จำนวน GPU ต่อเครื่อง** และทั้งคู่มีสายเร็วพอ — เพราะ NCCL แบ่งงานเท่ากันทุก rank
เครื่องที่ช้ากว่าจะกลายเป็นเพดานของทั้งกลุ่ม (ตัวเลข `link_gbps` ที่แสดงจึงเป็นของ **เครื่องที่ช้าที่สุด**)

ฮาร์ดแวร์ตรงกันอย่างเดียว **ยังไม่พอ** — อีกสองข้อที่ต้องผ่านด้วย:

| ข้อ | ทำไม | ตกข้อนี้แล้วเป็นยังไง |
|---|---|---|
| **สายเร็วต้องตั้ง IP จริงแล้ว** | พอร์ต ConnectX ที่ยังไม่ได้ตั้งค่าจะได้ `169.254.x.x` มาเอง ลิงก์ขึ้น 200G เหมือนกันแต่ยิง NCCL ข้ามเครื่องไม่ถึง | ไม่ถูกนับว่า `stack_ready` จึงไม่เข้ากลุ่มเลย · การ์ดของเครื่องนั้นบอกว่ามีสายเร็วแต่ยังไม่ได้ตั้ง IP |
| **ต้องมีขาอยู่ subnet เดียวกับกลุ่ม** | อยู่คนละวง = ต่อกันไม่ติด ต่อให้ทุกอย่างดูถูกหมด | อยู่ใน `excluded` เหตุผล `no-shared-fabric` — **ไม่ถูกนับใน world size** และไม่ทำให้กลุ่มกลายเป็น "ยังไม่พร้อม" |

และเครื่องเดียวกันที่ถูกเพิ่มไว้สองชื่อ (คนละที่อยู่ เช่น Tailscale กับ IP ในวง — ทะเบียนกันซ้ำไม่ได้)
จะถูกยุบเป็นตัวเดียวด้วย hostname + ชุด IP บนสายเร็ว (`excluded` เหตุผล `same-machine`) ไม่งั้น
world size จะบวกเกินไปหนึ่งแล้วแผน parallel เพี้ยนทั้งกลุ่ม

> เหตุผลที่ต้องเข้มขนาดนี้: world size คือสิ่งที่ตัดสินว่าจะใช้ **TP** หรือต้องมี **pipeline stage**
> เครื่องปลอมหนึ่งตัวทำให้ 2 กลายเป็น 3 แล้ว TP=3 หาร attention head ของโมเดลส่วนใหญ่ไม่ลง
> — ระบบจะแนะนำแผนที่ช้ากว่าความจริงทั้งที่เครื่องนั้นเข้าร่วมไม่ได้เลย
>
> `world_size` = ทั้งกลุ่ม (แผนเมื่อตั้ง cluster IP ครบ) · `usable_world_size` = เฉพาะเครื่องที่ตั้ง
> cluster IP ถูกต้องแล้วจริง ๆ — หน้าเว็บและ CLI แสดงคู่กันเมื่อสองค่าไม่เท่ากัน

### สั่งไม่ให้เครื่องนี้เข้ากลุ่ม

กลุ่มคือ **ข้อเสนอที่คำนวณใหม่ทุกครั้ง** ไม่ใช่สมาชิกที่ประกาศไว้ — เครื่องที่ฮาร์ดแวร์ตรงกันจะถูก
จับใส่กลุ่มเองโดยไม่ต้องมีใครสั่ง (รวมถึง **hub เอง** ซึ่งเข้าลิสต์เป็นผู้สมัครเสมอโดยไม่ต้องลงทะเบียน)
เครื่องที่ตั้งใจให้รันงานของตัวเองจึงต้องปิดได้:

```bash
lmds node set msi-5 --no-stack        # node ในทะเบียน (เปิดคืน: --stack)
lmds node cluster --no-self-stack     # เครื่องนี้เอง/hub (เปิดคืน: --self-stack)
```

**ในหน้าเว็บ**: ปุ่ม **ไม่เอาเข้ากลุ่ม** อยู่ท้ายแถบ cluster ของแต่ละเครื่อง ส่วนของ hub อยู่บรรทัด
เหนือลิสต์เครื่อง (อยู่นอกกลุ่มโดยตั้งใจ — ถ้าอยู่ในหัวกลุ่ม พอปิดแล้วกลุ่มหายก็จะกดเปิดคืนไม่ได้)

| เก็บที่ไหน | ของใคร |
|---|---|
| `nodes.yaml` → `stack: false` | node ในทะเบียน |
| `config.yaml` → `cluster.stack_self: false` | hub (ไม่มีแถวในทะเบียน) |

> เครื่องที่ปิดไว้จะ **ไม่ถูกเสนอเลย** ไม่ว่าฮาร์ดแวร์จะตรงแค่ไหนหรือตั้ง IP ถูกวงแล้วก็ตาม
> และไม่ขึ้นใน `excluded` ด้วย เพราะนั่นคือรายการ "อยากเข้าแต่เข้าไม่ได้" ไม่ใช่ "เจ้าของสั่งไม่เอา"

### cluster IP — สิ่งเดียวที่ระบบเดาให้ไม่ได้

ตรวจเจอว่ามีการ์ด 200G เป็นคนละเรื่องกับรู้ว่า **NCCL ต้องคุยกันทาง IP ไหน** เครื่องหนึ่งมักมีหลายเส้น
(เส้นบริหารจัดการที่ใช้ SSH + เส้น fabric) LMDS จึง **เสนอ** IP ที่เจอบนสายเร็วที่สุด แต่ให้คนยืนยัน

```bash
lmds node cluster                                   # ตารางสายเชื่อม + กลุ่มที่ stacked ได้
lmds node set spark2 --cluster-ip 10.10.0.2         # ตั้งค่า
lmds node set spark2                                # ดูค่าปัจจุบัน + ค่าที่ตรวจพบ
```

**ในหน้าเว็บ**: หัวข้อ *Other machines* → ปุ่ม **Check cluster** · ช่อง cluster IP จะโผล่ขึ้นมา
**ในการ์ดของเครื่องนั้นเอง** (ใต้ชื่อเครื่อง) แก้แล้วกด `save` ได้เลย

> เดิม cluster IP ของทุกเครื่องกองรวมกันเป็นตารางอยู่การ์ดล่างสุดของหน้า ซึ่งอ่านแล้วไม่รู้ว่า
> แถวไหนเป็นของเครื่องไหนถ้าไม่ไล่อ่านชื่อทีละบรรทัด — เป็นจุดที่ผู้ใช้จริงรายงานว่าสับสน
> ตอนนี้ค่าอยู่ติดกับเครื่องที่มันเป็นเจ้าของ และตารางล่างสุดถูกลบทิ้ง (ของเดียวกันอยู่สองที่
> คือต้นเหตุความสับสน ไม่ใช่ความสะดวก)

**เครื่องที่ stacked ด้วยกันได้จะถูกจัดให้อยู่ติดกัน แล้วมีรั้วสีคร่อมทั้งกลุ่ม** พร้อมป้าย
`CLUSTER A` / `CLUSTER B` สีเดียวกันบนทุกเครื่องในกลุ่ม และแถบของแต่ละเครื่องบอกชื่อคู่ตรง ๆ
(`⇄ spark-worker`) — ดูรูปแล้วรู้ทันทีว่าใครจับคู่กับใคร ไม่ต้องอ่านชื่อเทียบทีละตัว
· หัวรั้วบอก GPU / world size / ความเร็วสาย และสิ่งที่ยังขาดของกลุ่มนั้น

สถานะที่ตรวจให้:

| state | หมายความว่า |
|---|---|
| `ok` | IP นี้อยู่บน interface ที่เร็วพอจริง |
| `unset` | ยังไม่ได้ตั้ง (มีค่าที่เสนอให้ถ้าตรวจเจอ) |
| `mismatch` | ไม่ตรงกับ interface ไหนบนเครื่องนั้นเลย — พิมพ์ผิดก็มาทางนี้ |
| `slow` | ตั้งไปบนเส้นที่ช้าเกินไป (มัก = เผลอใส่ IP ของสาย 1G ที่ใช้ SSH) |
| `link-local` | เป็น 169.254.x.x — พอร์ตนั้นลิงก์ขึ้นและเร็ว 200G จริง แต่**ยังไม่ได้ตั้ง IP** ยิง NCCL ข้ามเครื่องไม่ถึง |

> DGX Spark มีพอร์ต ConnectX หลายเส้น เส้นที่ยังไม่ได้ตั้งค่าจะได้ 169.254.x.x มาเอง —
> ระบบจึงไม่เสนอ link-local ให้เด็ดขาด (ตัดสินจากตัว IP เอง ไม่พึ่งเวอร์ชันของ node)

### ขยายเป็น 3–4 เครื่อง

ชั้นทะเบียนและการจับกลุ่มรองรับกี่เครื่องก็ได้อยู่แล้ว · controller วน worker ทุกตัวจาก `WORKER_IPS`
(ค่าเดียว = พฤติกรรมเดิมของ 2 เครื่องเป๊ะ) · ใช้ target preset `dgx-spark-stacked-4` สำหรับ 4 เครื่อง

**ข้อจำกัดที่แท้จริงไม่ใช่โค้ด แต่คือ tensor parallel ต้องหาร attention head ลงตัว:**

| เครื่อง | world size | ใช้ได้ไหม |
|---|---|---|
| 2 | 2 | ✅ TP=2 (ทดสอบแล้ว) |
| **3** | 3 | ⚠️ TP=3 หาร head ไม่ลง (Llama 3.3 70B มี 64 head) — vLLM ปฏิเสธตั้งแต่ start · ต้องใช้ **TP=2 + pipeline** |
| 4 | 4 | ✅ TP=4 (64÷4=16) · หน่วยความจำรวม ~512 GB |

`lmds node cluster` บอกให้เองว่ากลุ่มนั้นใช้ TP ตรง ๆ ได้ไหม:

```text
พร้อม spark1 + spark2 + spark3 — NVIDIA GB10 x1/เครื่อง ·
world size 3 (TP=2 + pipeline (TP=3 หาร head ไม่ลง)) · 200G RDMA
```

`--write` เขียน `WORKER_IPS` ครบทุกตัวพร้อม `NNODES` และ `TENSOR_PARALLEL_SIZE` ให้ตรงจำนวนเครื่อง

> สถานะ: โครงสร้างและคำสั่งพร้อมและมีเทสครอบแล้ว แต่ **ยังไม่ได้รันจริงเกิน 2 เครื่อง**
> (ยังไม่มีเครื่องที่สาม) — `dgx-spark-stacked-4` จึงตั้ง `tested=False` และคิด budget แบบ conservative

### เขียนค่าลง bundle

```bash
lmds cluster write my-70b-model --head spark-head [--worker spark2 --worker spark3] [--on spark-head]
lmds node cluster --write my-70b-model --head spark-head      # ทางเดิม ยังใช้ได้
```

สร้าง `cluster.env` ใน bundle (บนเครื่องที่ bundle อยู่จริง — ปกติคือ head):

```bash
MASTER_IP=10.10.0.1
WORKER_IP=10.10.0.2
WORKER_IPS="10.10.0.2"          # worker ทุกตัวเรียงตาม rank (3–4 เครื่องมีหลายค่า)
SSH_USER=ops
TRANSPORT_IP_MASTER=10.10.0.1
TRANSPORT_IP_WORKER=10.10.0.2
NNODES=2
TENSOR_PARALLEL_SIZE=2
NCCL_SOCKET_IFNAME=enp1s0f0np0
```

controller ของ stacked จะ **source ไฟล์นี้ก่อน default ทั้งหมด** แล้วข้ามการถาม IP ตอน `start`
(env ที่ตั้งมาจากภายนอกยังชนะไฟล์นี้เสมอ) — ไม่มีไฟล์: มี tty ถามแบบเดิม · ไม่มี tty (hub สั่ง) หยุดพร้อมบอกคำสั่งข้างบน
ไม่ ssh ไปเครื่องตัวอย่างเงียบ ๆ

**`write` อ่าน `NNODES` จาก controller ใน bundle ก่อน** (`nodes/stacked.py`) แล้วตัดกลุ่มให้เหลือ head + worker ตามจำนวนนั้น —
กลุ่ม 4 เครื่องกับ bundle ที่ render มาสำหรับ 2 เครื่องเคยได้ `NNODES=4/TP=4` ทับแผน (ไฟล์ถูก source ก่อน default จึงชนะ)
และ bundle 4 เครื่องที่เลือก worker ตัวเดียวเคยได้ TP=2 ที่โมเดลไม่ fit · ไม่พอ/เกิน = ปฏิเสธพร้อมบอกว่าต้อง target ไหน
(`--nnodes` ระบุเองได้เมื่อ bundle ไม่อยู่บนเครื่องนี้)

### กุญแจ head → worker — `lmds cluster pair`

`lmds node setup` / *Add machine* ติดตั้งแต่กุญแจของ **hub** ลงทุกเครื่อง แต่ controller stacked รันบน **head** แล้ว
`ssh ${SSH_USER}@${WORKER_IP}` ด้วยกุญแจของ head เอง → `sync-worker`/`start` ตายด้วย `Permission denied (publickey)`
ทุกครั้งที่ไม่ได้ตั้งมือ (เจอจริงทุกคลัสเตอร์ก่อน 0.6.0)

```bash
lmds cluster pair spark-head spark-worker [spark-3 …]
```

- กุญแจ **เกิดบน head** (ไม่ผ่าน hub) · public key ลง `authorized_keys` ของ worker · เขียน stanza ใน `~/.ssh/config` ของ head
  (`IdentityFile` + `StrictHostKeyChecking accept-new` — BatchMode ตอบคำถาม host key ครั้งแรกไม่ได้) ทำซ้ำได้ ไม่แตะบรรทัดของผู้ใช้
- ยืนยันด้วย `ssh -o BatchMode=yes` เปล่า ๆ แบบเดียวกับที่ controller ใช้
- หน้าเว็บ: ปุ่ม **Pair SSH** บนหัวกลุ่ม · wizard ทำให้เองหลังเขียน cluster.env ตอน push · `lmds node push <head> <slug> --download --start`
  กับ bundle stacked ก็ทำครบ: cluster.env → pair → `sync-worker && verify-worker` → start

### ทำไมคู่นี้ยังไม่พร้อม — `lmds cluster doctor`

```bash
lmds cluster doctor spark-head spark-worker [--slug my-70b-model]
```

อ่านอย่างเดียว ไม่แตะอะไรบนเครื่อง · ตรวจทีละข้อพร้อมคำสั่งแก้ (รหัสเดียวกันทั้ง CLI ไทย / เว็บอังกฤษ):

| ข้อ | ตรวจอะไร |
|---|---|
| `registered` `reachable` `gpu` `opted-out` | อยู่ในทะเบียน · hub ต่อถึง · มี GPU · ไม่ได้ `--no-stack` |
| `same-site` `hardware` | ไซต์เดียวกัน · GPU รุ่น/จำนวนตรงกันทุก rank |
| `cluster-ip` `same-subnet` `iface-up` `link-speed` | unset/mismatch/slow/link-local (+ IP ที่เสนอ) · วงเดียวกัน · สายขึ้น · negotiate ต่ำกว่าที่ควร (เตือน) |
| `ssh-head-to-worker` `fabric-ping` | **ssh จาก head ไป worker จริง** · ping บนสายคลัสเตอร์เมื่อ ssh ล้ม (ICMP อาจถูกบล็อก) |
| `disk` | < 150 GB ว่าง = เตือน (weight ลงทุกเครื่อง) |
| `bundle-on-head` `cluster-env` `cluster-env-match` | (มี `--slug`) bundle อยู่บน head ไหม · มี cluster.env ไหม · ค่าในไฟล์ตรงทะเบียนไหม |

หน้าเว็บ: ปุ่ม **Doctor** บนหัวกลุ่ม (`GET /api/cluster/doctor`) · hub ที่ไม่มี GPU (VM ควบคุม) ขึ้นเป็น *control plane — not a
stacked candidate* ไม่ใช่ "not ready · 10G too slow" พร้อมปุ่มที่ไม่มีความหมาย

### โมเดล stacked บนการ์ด

bundle อยู่ที่ head เท่านั้น การ์ด worker จึงเคยว่างทั้งที่เครื่องถูกใช้อยู่ · ตอนนี้ `lmds agent info` อ่าน `cluster.env`
ข้าง controller แล้ว hub เทียบ IP กับ cluster_ip ในทะเบียน: การ์ด head ได้ป้าย **stacked head · worker <ชื่อ>** · การ์ด worker
ได้แถว **stacked worker of <head>** พอร์ตเดียวกัน ไม่มีปุ่ม (ทั้งทาง SSE และ `/api/nodes/<n>/inventory`) · ปุ่ม `logs-worker`
บนการ์ด head = log ของ container บน worker · ยกเลิก start ของ stacked จะเตือนว่า container บน worker อาจยังรันอยู่

`NCCL_SOCKET_IFNAME` สำคัญกว่าที่คิด: ถ้าไม่ระบุ NCCL จะเลือก interface เอง และมักเลือกเส้นบริหารจัดการ
ที่ช้ากว่า — งานยังรันได้ แต่ช้าลงแบบหาสาเหตุยาก

ถ้าไม่ได้ตั้งไว้ controller จะ **หาชื่อ interface จาก IP ให้เอง** ทั้งฝั่ง head และ worker
(`ip -o -4 addr show` แล้วจับคู่กับ `TRANSPORT_IP_*`) — จำเป็นเพราะชื่อพอร์ตบน DGX Spark ยาวและ
ไม่เหมือนกันทุกเส้น เช่นเครื่องจริงเครื่องหนึ่งมี `enp1s0f1np1` (10.100.152.1) กับ
`enP2p1s0f1np1` (10.100.153.1) เป็นคนละ fabric · ค่าที่ตั้งมาเองยังชนะการตรวจอัตโนมัติเสมอ

controller ยัง **ตรวจก่อนว่าเครื่องที่รันอยู่คือ head จริง** (มี IP ของ `MASTER_IP` อยู่บนตัวเอง)
ถ้ารันผิดเครื่องจะตายทันทีพร้อมบอกเหตุผล ไม่ปล่อยไปตายตอน NCCL init ที่อ่านไม่รู้เรื่อง

---

> **สถานะคลัสเตอร์ขึ้นเองตลอด** — หน้าเว็บอ่านจากแคชของ refresher จึงเห็นรั้วกลุ่มและ
> ป้าย "stacked ได้ / ยังไม่พร้อม" ตั้งแต่เปิดหน้า ไม่ต้องกด Check cluster ก่อน ·
> ปุ่ม **Check cluster** เหลือไว้สำหรับ "ตรวจสดเดี๋ยวนี้" (เช่นเพิ่งเสียบสายใหม่)

## 5.05 หลายคลัสเตอร์ในไซต์เดียวกัน (v0.5)

การจับกลุ่มใช้สามอย่างเป็นกุญแจ: **ไซต์ · ชื่อคลัสเตอร์ · ลายเซ็นฮาร์ดแวร์**
แล้วภายในถังเดียวกันจึงแบ่งย่อยตาม subnet ที่ใช้ร่วมกันอีกที

| สถานการณ์ | ผลลัพธ์ |
|---|---|
| สองคู่ คนละ subnet ไม่ตั้งชื่อ | ระบบแยกให้เอง **สองกลุ่ม** |
| สี่เครื่อง วงเดียวกัน ไม่ตั้งชื่อ | **หนึ่งกลุ่ม** world size 4 (TP=4) |
| สี่เครื่อง วงเดียวกัน ตั้งชื่อสองชื่อ | **สองกลุ่ม** กลุ่มละ 2 |
| คนละไซต์ | **ไม่จับคู่กันเลย** แม้เลขวงจะบังเอิญตรงกัน |

```bash
lmds node set n1 --cluster-name ทีมค้นหา
lmds node set n2 --cluster-name ทีมค้นหา
lmds node set n3 --cluster-name ทีมสำรอง
lmds node set n4 --cluster-name ทีมสำรอง
lmds node cluster                      # เห็นสองกลุ่มแยกกัน
```

**บนหน้าเว็บ:** ปุ่ม **Check cluster** → ในแถวของแต่ละเครื่องมีช่อง **คลัสเตอร์** ต่อจากช่อง
cluster IP · พิมพ์ชื่อแล้วกด Save · หัวกลุ่มจะขึ้นป้ายชื่อคลัสเตอร์ (หรือ "แบ่งอัตโนมัติ"
ถ้าไม่ได้ตั้ง) พร้อมป้ายไซต์

> **ทำไมชื่อถึงจำเป็น** — ระบบแบ่งเองได้เฉพาะตอนที่แต่ละคู่อยู่คนละวง · เครื่องรุ่นเดียวกัน
> สี่เครื่องบนวงเดียวกันจะถูกมองเป็นก้อนเดียว TP=4 ซึ่งบางทีไม่ใช่สิ่งที่ต้องการ (อยากได้
> สองคู่แยกกันเพื่อรันคนละโมเดล หรือให้คู่หนึ่งเป็นตัวสำรอง)

> **ตั้งชื่อรวมกันแล้วยิงถึงกันไม่ได้ก็ยังฟ้อง** — ชื่อบังคับการจัดกลุ่ม แต่ไม่ได้เสกสายให้
> ถ้าเครื่องในกลุ่มไม่มีวงร่วมกันเลย กลุ่มจะขึ้น blocker `no-shared-fabric` ไม่ใช่ปล่อยให้
> ไปค้นพบตอน start แล้วค้างที่ NCCL init

## 5.2 ทำสำเนาโมเดลข้ามเครื่อง (v0.5)

```bash
lmds node clone <slug> --from msi-1 --to msi-2 --start
```

ใช้ตอนอยากได้ตัวสำรองหรือกระจายโหลดผ่าน gateway · **วัดจริงบนฟลีต: 412 MB/s จบใน
3 นาที 47 วิ สำหรับ 91.6 GB** เทียบกับโหลดจาก Hugging Face ที่ 40.7 MB/s (38 นาที)

- ไฟล์วิ่ง**ตรงระหว่างสองเครื่อง ไม่ผ่าน hub** — hub มักเป็นเครื่องเล็กที่จะเป็นคอขวด
- เลือก `cluster_ip` (200G) ถ้ามีทั้งสองฝั่ง ไม่งั้นถอยไปเส้นปกติ
- คัดลอกทั้ง weight และ bundle แล้ว `verify-files` ที่ปลายทาง
- **บนหน้าเว็บ:** การ์ดเครื่อง → `⋯` ของโมเดล → หมวด **ทำสำเนาไปเครื่องอื่น**

**กุญแจไม่เคยออกจาก hub** — node แต่ละเครื่องไม่มี key ของกันและกันโดยตั้งใจ (เครื่องหนึ่ง
ถูกยึดไม่ควรแปลว่าทั้งฟลีตถูกยึด) · คำสั่งนี้สร้างกุญแจชั่วคราวต่อครั้ง ฝาก public key ที่
ปลายทางแบบ `restrict` ส่ง private key ให้ต้นทางทาง stdin เข้า `ssh-agent` **ในหน่วยความจำ
ไม่แตะดิสก์** แล้วถอนออกเสมอเมื่อจบ ไม่ว่าสำเร็จหรือล้ม · ถอนด้วย `grep -v -F <marker>`
ที่สุ่มต่อครั้ง ไม่ใช่ pattern กว้าง ๆ เพราะไฟล์นั้นมีกุญแจของ hub และของผู้ใช้อยู่ด้วย

## 5.1 เข้าถึงจากนอกออฟฟิศ — ที่อยู่สำรอง

เครื่องเดียวกันมักเข้าได้หลายทาง: LAN ตอนอยู่ที่ออฟฟิศ และ Tailscale/VPN ตอนออกไปข้างนอก

```bash
lmds node set spark-head   --alt-host 100.124.77.93
lmds node set spark-worker --alt-host 100.115.254.108
lmds node set spark-head                                 # ดูที่อยู่ทั้งหมด
```

hub ลอง**ที่อยู่หลักก่อนเสมอ** ต่อไม่ถึงจึงค่อยลองสำรอง — ไม่ต้องแก้ทะเบียนตอนย้ายที่ทำงาน

> **ต่างจากการเปลี่ยน `host`**: เปลี่ยน host = คนละเครื่อง (ทำไม่ได้ ต้อง remove แล้ว add ใหม่)
> ส่วน `--alt-host` คือทางเข้าอีกทางของ**เครื่องเดิม**
>
> failover เกิดเฉพาะเมื่อ**ต่อไม่ถึง** (timeout / no route / connection refused) เท่านั้น
> คำสั่งที่ต่อได้แต่ล้มด้วย exit code ของตัวเอง จะไม่ถูกยิงซ้ำ — ไม่งั้นคำสั่งที่มีผลข้างเคียง
> อย่าง `start` จะทำงานสองรอบ

## 6. ความปลอดภัย

- **ไม่เก็บรหัสผ่าน** ที่ไหนเลย ใช้ครั้งเดียวตอนติดตั้ง key แล้วทิ้ง
- key เป็นของ LMDS เอง (`id_lmds`) แยกจาก key ส่วนตัวของคุณ — ถอนสิทธิ์ทีละเครื่องได้โดยไม่กระทบอย่างอื่น
- `~/.config/lmds/nodes.yaml` เป็น 0600 (มีชื่อ user/host ของเครื่องภายใน)
- SSH ใช้ `BatchMode=yes` — ไม่มีทางค้างรอ prompt · `StrictHostKeyChecking=accept-new`
- คำสั่งข้ามเครื่องผ่านหน้าเว็บถูกจำกัดด้วย allowlist ฝั่ง server
- เปิดหน้าเว็บด้วย `--bind 0.0.0.0` ครั้งแรก จะ**ถามก่อน**ว่าจะตั้ง token เองไหม (Enter = สุ่มให้)
  แล้วจำไว้ใช้ซ้ำทุกครั้ง · เข้าหน้าเว็บต้องผ่าน login ทุกครั้งที่เบราว์เซอร์ยังไม่เคยจำ

---

## 7. ข้อจำกัดที่รู้ตัว (ยังไม่ทำ)

- hub ยังไม่เก็บ cluster IP **ของตัวเอง** ในทะเบียน (ตัวเองไม่ได้อยู่ในทะเบียน — `PATCH /api/cluster/self` ตั้งได้แค่ว่าจะเอา hub
  เข้ากลุ่มไหม) — ใช้ค่าที่ตรวจพบตอนเขียน `cluster.env` ถ้าเครื่อง hub มีหลายเส้นเร็วเท่ากันแล้วเลือกผิด ให้แก้ `cluster.env` เอง
- `lmds deploy` จาก CLI ยังไม่ push bundle ไปติดตั้งบน node ให้อัตโนมัติ (ใช้ `lmds node push` ต่อ) — wizard บนหน้าเว็บ push ให้แล้ว
- กลุ่ม stacked **เกิน 2 เครื่อง** จับกลุ่ม/เขียน `cluster.env` (`WORKER_IPS` · `TRANSPORT_IPS_WORKER`) และ controller วน worker
  ทุกตัวได้ แต่**ยังไม่เคยรันจริงเกิน 2 เครื่อง** — `dgx-spark-stacked-4` เป็น `tested=False`
- ~~wizard ตั้งค่าเครือข่ายคลัสเตอร์~~ — **ทำแล้ว (2026-09-05)**: ปุ่ม **Set up cluster network** ที่หัว Other machines / หัวกลุ่ม cluster
  (Devices → Cabling check → Plan → Apply → Verify) เขียน netplan ให้ทุกเครื่อง ตรวจสาย/ping/pair SSH แล้วจดลงทะเบียน ·
  CLI `lmds cluster inspect|plan|apply|remove-net` · ดู RUNBOOK §Cluster network setup และ NETWORK.md
- fabric detection **ยืนยันกับ DGX Spark จริงแล้ว (5 ส.ค. 2569)** — เจอ ConnectX 200G RDMA 4 เส้น
  (`rocep1s0f0/f1`, `roceP2p1s0f0/f1`) ถูกต้อง · จับคู่ 2 เครื่องจริงผ่านแล้ว (Llama 3.3 70B 2026-08-05 ·
  Qwen3.8-Flash-Next-NVFP4 173 GB และ Nemotron-3-Super-120B 2026-09-04 บน spark-head ⇄ spark-worker)

---

## 8. อ้างอิงคำสั่งทั้งหมด

```text
lmds agent info | bench               # JSON สถานะ/คะแนนเครื่องนี้ (hub เรียกผ่าน SSH — ปกติไม่ต้องพิมพ์เอง)

lmds node add <host> --user <u>       # + --name --port --note --cluster-ip --cluster-iface --install
lmds node list [--check]
lmds node set <name> [--cluster-ip IP] [--cluster-iface NAME] [--note TEXT] [--site S] [--cluster-name C]
                     [--alt-host IP,IP] [--stack|--no-stack]
lmds node remove <name> [-y]
lmds node install [<name>|--all] [--with-prereq]    # hub ส่งโค้ดไปเอง
lmds node setup   [<name>|--all] [--with-prereq]    # ขั้น sudo — ถามรหัสตอนนี้ ใช้ครั้งเดียว
lmds node run <name> <คำสั่ง lmds...>
lmds node ctl <name> <slug> <คำสั่ง controller...>
lmds node push <name> <slug> [--download] [--start]
lmds node clone <slug> --from <node> --to <node> [--start] [--dry-run] [--no-verify] [-y]
lmds node cluster [--write SLUG --head NAME] [--worker NAME] [--on NODE] [--self-stack|--no-self-stack]

lmds cluster show [--self-stack|--no-self-stack]                    # = node cluster
lmds cluster write <slug> --head NAME [--worker NAME…] [--on NODE] [--nnodes N]
lmds cluster pair <head> <worker…>                                  # กุญแจ head→worker (เกิดบน head)
lmds cluster doctor <head> <worker> [--slug SLUG]                   # อ่านอย่างเดียว

lmds ps --all                         # โมเดลของทุกเครื่องในตารางเดียว
```

REST (หน้าเว็บใช้ชุดนี้ — token เดียวกับหน้าเว็บ):

```text
GET    /api/nodes                     · GET /api/fleet/summary · PUT /api/nodes/order
POST   /api/nodes                     {host, user, name?, port?, password?}   (port ไม่ใช่เลข/นอกช่วง = 400)
PATCH  /api/nodes/{name}              {cluster_ip?, cluster_iface?, note?, site?, cluster_name?, alt_hosts?, stack?}
DELETE /api/nodes/{name}
POST   /api/nodes/{name}/install      · POST /api/nodes/{name}/setup · POST /api/nodes/{name}/fix-permissions
GET    /api/nodes/{name}/inventory[?refresh=true]   (refresh เขียนแคชด้วย)
POST   /api/nodes/{name}/models/{slug}/{command}    # allowlist: start stop restart repair doctor logs enable disable remove set
POST   /api/nodes/{name}/models/{slug}/ctl/{command}  # test-* parsers bench stress client-config network-info status props
                                                      # verify-files prepare-runtime sync-worker verify-worker clear-fi-cache logs-worker
POST   /api/nodes/{name}/models/{slug}/bench · GET /api/nodes/{name}/bench/{slug} · POST …/bench/{slug}/remove
GET    /api/nodes/{name}/models/{slug}/clone/targets · POST …/clone
GET    /api/nodes/{name}/models/{slug}/memory · GET …/settings/suggest
POST   /api/models/{slug}/push/{name}   # แพ็กจากโฟลเดอร์ทุกครั้งก่อนส่ง (ไม่ใช่ zip ตอน generate)
GET    /api/cluster                   # ตารางสายเชื่อม + กลุ่มที่ stacked ได้
POST   /api/cluster/write             {slug, head, workers?, on?}
POST   /api/cluster/pair              {head, workers}
GET    /api/cluster/doctor?head=&worker=&slug=
PATCH  /api/cluster/self              {stack}   # hub เองเข้ากลุ่ม stacked ไหม (= --self-stack)
GET    /api/scan?all_nodes=true       # ถามทุกเครื่องพร้อมกัน (8 worker · 60 วิ/เครื่อง)
GET    /api/jobs/{id} · POST /api/jobs/{id}/cancel
```

## ตรวจว่าทุกเครื่องรันโค้ดชุดเดียวกันจริง

```bash
lmds node install --all             # อัปเดตทุกเครื่องในทะเบียน (hub ส่งโค้ดที่ตัวเองรันอยู่ไป — ไม่แตะ GitHub)
lmds node run <ชื่อ> version        # → lmds 0.6.0  (a4ec6bb)
lmds node list                      # ป้าย ≠ hub เฉพาะเครื่องที่ commit ต่างจริง (เทียบ prefix — 7 กับ 8 ตัวไม่นับว่าต่าง)
```

**ดูที่ commit ไม่ใช่เลข version** — `__version__` ไม่ขยับทุกคอมมิต (0.3.0 อยู่มาหลายสิบ
คอมมิตแล้ว) ทุกเครื่องจึงขึ้นเลขเดียวกันหมดทั้งก่อนและหลังอัปเดต · `lmds version` เลย
พ่วง commit ของโค้ดที่ *กำลังรันอยู่จริง* มาให้ และ `lmds agent info` ส่งค่าเดียวกันใน
`host.lmds_commit` ให้ hub เทียบทั้งฟลีตจากที่เดียว

> git checkout (hub) ถาม git · เครื่องที่ติดตั้งปกติใช้ commit ที่ `install.sh` ประทับไว้
> ใน `_build.py` (`--short=7` คงที่) — **editable install มี stamp ค้างตั้งแต่ติดตั้งครั้งแรก** จึงต้องถาม git ก่อนเสมอ
> · หน้าเว็บ: ป้ายบนการ์ด · Needs attention · ป้าย "hub ต้อง restart" เทียบ commit แบบ prefix (`sameCommit`) · CLI ใช้
> `_same_commit` กติกาเดียวกัน · **ปุ่ม Update** ที่แถบบน = pull บน hub → install → restart (รอลายเซ็น process เปลี่ยน) →
> อัปเดต node ทุกตัวด้วยโค้ดจาก hub ("ดึงมาแล้วเป็นของเดิม" บน hub ไม่ใช่ความล้มเหลว — node อาจยังตามหลังอยู่)

อยากมั่นใจถึงระดับไบต์ (เช่นสงสัยว่าไฟล์ถูกแก้มือ) เทียบ sha256 ของแพ็กเกจได้:

```bash
PY=~/.local/share/lmds/venv/bin/python
SITE=$($PY -c "import lmds.inventory,pathlib;print(pathlib.Path(lmds.inventory.__file__).parent)")
find "$SITE" \( -name '*.py' -o -name '*.j2' \) | LC_ALL=C sort | xargs sha256sum | sha256sum
```

ค่านี้ต้องตรงกันทุกเครื่อง **ยกเว้น `_build.py`** ซึ่งเป็น stamp ต่อการติดตั้ง ไม่ใช่โค้ด —
ถ้าจะเทียบให้ตรงเป๊ะต้องตัดไฟล์นั้นออก

**ข้อควรระวังจากของจริง:**

- อัปเดตทีละเครื่องแล้วมี commit ใหม่คั่นกลาง = ฟลีตแตกเป็นสองชุดโดยไม่มีอะไรฟ้อง
  (เจอจริง: spark-head ถูก install หลัง commit ถัดไป เลยต่างจากอีก 6 เครื่อง ทั้งที่
  `lmds version` ตอนนั้นยังไม่บอก commit) — ให้ `--all` **หลัง** push commit สุดท้ายเสมอ
- เครื่องที่เคยติดตั้งแบบ editable จาก git clone (`~/AutoDeployDGXProject/src`) จะถูก
  `install.sh` แปลงเป็น venv site-packages · **clone เก่ายังค้างบนดิสก์แต่ไม่ใช่โค้ดที่รันแล้ว**
  อย่าไปแก้ที่นั่น
- **ชื่อในทะเบียนไม่จำเป็นต้องตรงกับ hostname** — เป็นคนละอย่างโดยตั้งใจ (ทะเบียนชี้ที่
  `user@host:port`) แต่ถ้าไม่ตรงจะสับสนตอนไล่ปัญหา · เทียบได้ด้วย
  `lmds node run <ชื่อ> agent info` แล้วดู `host` ที่มันตอบกลับมา

## bundle เก่าไม่ได้อัปเดตตามโค้ด

`lmds node install` เปลี่ยนแค่ตัวโปรแกรม — **bundle ที่สร้างไว้แล้วยังเป็นไฟล์เดิม**
controller เก่าจึงไม่มีคำสั่ง/ตัวกันพลาดที่เพิ่มมาทีหลัง และ `MODEL_PROFILE.yaml` เก่า
ไม่มีคีย์ใหม่ (เช่น `features.moe`) คอนโซลจึงไม่มีข้อมูลจะแสดง

```bash
lmds rebuild <slug>     # สร้าง bundle เดิมใหม่ด้วยตรรกะปัจจุบัน ไม่เรียก LLM ซ้ำ
```

ปลอดภัยกับตัวที่รันอยู่ — เขียนแค่ไฟล์ bundle ตัวที่รันยังใช้ controller เดิมจนกว่าจะ restart

## อัปเดต llama.cpp บน node (native build)

DGX Spark ไม่มี docker image ทางการ (ARM64/SM121) — `prepare-runtime` จึง **clone แล้ว
build llama.cpp จาก source** ไว้ที่ `~/src/llama.cpp` · `lmds node install` อัปเดตแค่ตัว
lmds เอง **ไม่แตะ llama.cpp** ตัวนี้จึงค้างเวอร์ชันได้เป็นเดือนโดยไม่มีอะไรฟ้อง

```bash
cd ~/src/llama.cpp
git status -sb          # "## HEAD (no branch)" = detached ต้อง checkout ก่อน
git checkout master && git pull --ff-only
cmake --build build --config Release -j "$(nproc)"
```

> **ทำไมต้อง checkout ก่อน** — `prepare-runtime` ตรึงเวอร์ชันด้วย `git checkout <ref>`
> ซึ่งทิ้ง repo ไว้ใน **detached HEAD** · `git pull` ที่นั่นตอบ `git pull <remote> <branch>`
> แล้วจบเงียบ ๆ · เจอจริง 3 ใน 4 เครื่อง: สั่ง pull+build ไปแล้ว build สำเร็จ commit ไม่ขยับ
> เลยสักตัว ถ้าไม่ได้เทียบ commit ก่อน-หลังจะเข้าใจว่าอัปเดตแล้ว

**`lmds doctor <slug>` เตือนให้เมื่อ llama.cpp เก่าเกินไป** — เช่น ไม่มี commit
`cd0fa6051` ซึ่งแก้ tool schema ที่มี `maxLength`/`maxItems` เกิน 2000 (Claude Code
ส่งแบบนั้นมา) ถ้าไม่มีจะตอบ 400 `failed to parse grammar`

**ระวัง fork**: บาง arch มีเฉพาะใน checkout แยก เช่น `~/src/llama.cpp-muse` ที่รองรับ
`muse-glimmer` — **อย่า pull ทับหรือชี้ `LLAMA_CPP_DIR` ไปที่นั่นตอน `prepare-runtime`**
เพราะมันจะ `git checkout` ทับ fork ทิ้ง (controller ของ muse มี guard กันไว้แล้ว)

**หลัง build ต้อง restart โมเดล** — process ที่รันอยู่ยังถือ binary ตัวเก่าไว้ (inode เดิม)
จนกว่าจะ restart
