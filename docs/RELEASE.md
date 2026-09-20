# แผนการปล่อยเวอร์ชัน · LMDS + LiteGate

เขียน 2026-09-20 · ครอบสองรีโป: `AutoDeployDGXProject` (LMDS) และ `AiGatewayLocal` (LiteGate)

---

## ทำไมเอกสารนี้ถึงเกิด

จนถึงวันนี้เรา commit ลง `main` ตรง ๆ ได้เพราะคนใช้คือเราเอง · ตั้งแต่มีลูกค้าจริง เงื่อนไขเปลี่ยนไปข้อเดียวแต่เปลี่ยนทุกอย่าง:

> **`main` คือสิ่งที่เครื่องลูกค้าทุกเครื่อง clone**
> `lmds node install` รัน `git clone --depth 1` หรือ `git pull --ff-only` บน `main`
> — ดู `src/lmds/nodes/ssh.py` `_INSTALL_SCRIPT`

ถ้า `main` พังตอน 14:00 ลูกค้าที่กด update ตอน 14:05 พังตาม และเราไม่รู้ว่ามีใครกดบ้าง

**ทั้งสองรีโปเป็น PUBLIC** — `git push` คือการเผยแพร่ ย้อนกลับไม่ได้

---

## 1. `main` หมายถึงอะไร

| | LMDS | LiteGate |
|---|---|---|
| ใครดึงไปใช้ | เครื่องลูกค้าทุกเครื่องผ่าน `lmds node install` | คนที่ clone เอง / docker build |
| ความหมาย | **ติดตั้งได้เสมอ** | ติดตั้งได้เสมอ |
| License | Source-available (1 เครื่องฟรี) | MIT |

**กฎเดียวที่ห้ามละเมิด: ห้าม commit งานที่ยังไม่ผ่าน gate ลง `main` โดยตรง**

ทำงานบน branch สั้น ๆ แล้ว merge เมื่อเขียว · origin มี `feat/drag-reorder-nodes` และ `feat/tool-parser-retrofit` อยู่แล้ว ใช้ลายเดิมได้เลย

---

## 2. ด่านก่อน merge เข้า `main`

CI ของสองรีโป**ไม่เหมือนกัน** — รันของตัวเองให้ถูก

### LMDS — `.github/workflows/ci.yml`

| job | blocking? | คำสั่ง |
|---|---|---|
| `tests` | ✅ | `pytest` บน **3.10 / 3.11 / 3.12** |
| `shell` — `bash -n` | ✅ | `*.sh` + `scripts/*.sh` |
| `shell` — shellcheck | ❌ non-blocking | `-S warning` + `continue-on-error` |
| `secret-scan` | ✅ | grep `sk-` `AIza` `hf_` `ghp_` (ยกเว้น `tests/`) |

```bash
# รันให้ครบก่อน merge
python3.10 -m pytest -q && python3.11 -m pytest -q && python3.12 -m pytest -q
bash -c 'shopt -s nullglob; for s in *.sh scripts/*.sh; do bash -n "$s" || exit 1; done'
grep -rInE '(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,})' \
  --exclude-dir=.git --exclude-dir=.github --exclude-dir=tests . && echo FAIL || echo OK
```

> ⚠️ **บน macOS เทส 23 ตัวตกเสมอ** เพราะไม่มี GNU coreutils (`sha256sum` · `find -printf` · `stat -c`)
> → **เทียบรายชื่อ test ID ไม่ใช่จำนวน** · 23 ตัวเดิม = ผ่าน · มีชื่อใหม่โผล่ = ไม่ผ่าน
> เครื่อง dev มีแค่ python 3.11 กับ 3.14 → **รัน matrix ครบไม่ได้ในเครื่อง** ต้องพึ่ง CI สำหรับ 3.10/3.12

### LiteGate — `.github/workflows/ci.yml`

| job | blocking? | หมายเหตุ |
|---|---|---|
| `ruff check app tests scripts` | ✅ | **LMDS ไม่มี lint gate** |
| registry validation | ✅ | `load_snapshot(Path("config"))` ต้องไม่มี error |
| `pytest` | ✅ | **3.11 / 3.12 เท่านั้น ไม่มี 3.10** |
| `docker build` + `/healthz` | ✅ | ช้า · จำเป็นเมื่อแตะ Dockerfile/deps |
| `secret-scan` + tracked `.env` | ✅ | เพิ่ม 2026-09-20 · ครอบ `lg_sk_` / `edu_sk_` ด้วย · **ไม่ข้าม `tests/`** |

---

## 3. เวอร์ชันต้องมีความหมาย

**ปัญหาที่มีอยู่จริงตอนนี้:** `landing/2026-09` มี 7 commits ของจริง แต่ `__version__` ยังเป็น `0.6.1` — แยกได้ด้วย commit hash เท่านั้น (`lmds version` แสดง `0.6.1 (624cd5f)`)

ตอนอยู่กันเองไม่เป็นไร · ตอนมีลูกค้า *"ผมใช้ 0.6.1"* กลายเป็นประโยคที่ไม่มีข้อมูล

**บัมพ์เวอร์ชันใน commit เดียวกับที่ tag:**

```
src/lmds/__init__.py   __version__ = "<X.Y.Z>"
                       TEMPLATE_STANDARD  ← บัมพ์เมื่อ template contract เปลี่ยน เท่านั้น
CHANGELOG.md           หัวข้อ <X.Y.Z> พร้อมวันที่
README.md / README.en.md   badge version + badge tests + **ตัวเลขเทสในเนื้อความ**
                       (README.md และ README.en.md เขียนจำนวนเทสไว้ *สองที่* คือป้ายกับย่อหน้า
                        ท้ายไฟล์ ซึ่งประกาศเองว่า "ตัวเลขเดียวกับป้าย" — 2026-09-20 หลุดกันจริง
                        ป้ายขึ้น 2118 ส่วนเนื้อความยังเป็น 2037)
CONTRIBUTING.md        จำนวนเทส/ไฟล์ ในบรรทัดคำสั่ง pytest
docs/CLI_SPEC.md       หัวไฟล์ "CLI Specification — <X.Y.Z>" · จำนวนไฟล์เทส/เทส · โครงสร้าง source
                       · คำสั่งที่เพิ่มใหม่ (สเปกเคยตกหล่น license/fit/fleet/bundles ไปทั้งชุด)
git tag -a v<X.Y.Z>

ถ้าเพิ่ม/ลด quality gate ต้องไล่แก้ **แปดที่**: CLI_SPEC (×3) · CONTRIBUTING · SECURITY ·
PRD · ROADMAP · USAGE — เลขจำนวนด่านฝังไว้ทุกที่ ตัวจริงคือ `ALL_GATES` ใน validator/gates.py

วิธีนับเทสที่เชื่อได้ (RTK proxy กิน summary line ของ pytest):
  pytest --collect-only 2>&1 | tail -3
```

LiteGate: `pyproject.toml` `version` + `app/__init__.py` ถ้ามี + `CHANGELOG.md` + badge

**เกณฑ์เลือกเลข** (เราไม่ใช่ semver เคร่ง แต่ให้สม่ำเสมอ):
- **patch** — แก้บั๊ก ไม่เปลี่ยนสิ่งที่ผู้ใช้เห็น
- **minor** — ฟีเจอร์ใหม่ · flag ใหม่ · template contract เดิม
- **major** — `TEMPLATE_STANDARD` เปลี่ยน หรือ bundle เก่าใช้กับของใหม่ไม่ได้

---

## 4. ลูกค้าได้อัปเดตยังไง — และช่องว่างที่ยังไม่แก้

### วันนี้

```python
_INSTALL_SCRIPT = """
  cd AutoDeployDGXProject && git pull --ff-only     # ตามปลาย branch
  หรือ  git clone --depth 1 {repo}                   # ได้ default branch
"""
```

`LMDS_REPO_URL` เปลี่ยนได้แค่ *"repo ไหน"* **ไม่ใช่ *"commit ไหน"***

→ **เครื่องลูกค้าวิ่งตามปลาย `main` เสมอ ไม่มีทางบอกว่า "อยู่ที่ tag นั้นพอ"**

### ที่ต้องแก้ (เข้าคิว WS)

1. **`LMDS_REPO_REF`** — ให้ปักหมุด tag/commit ได้ · ลูกค้าองค์กรจะขอข้อนี้เป็นข้อแรก
2. **`source_bundle()` hardcode `main`**
   ```python
   git bundle create {target} main
   ```
   hub ที่ checkout branch อื่น **ส่งโค้ดของตัวเองไป node ไม่ได้** · และ CLI `lmds node install` ไม่เคยส่ง `bundle=` เลย — ใช้ `git pull` จาก GitHub ทางเดียว ทำให้เครื่อง air-gapped อัปเดตผ่าน CLI ไม่ได้
3. **`fleet check` อ่าน cache ไม่ probe สด** — หลัง `node install` เสร็จยังรายงานเวอร์ชันเก่า ต้อง `lmds node list --check` ก่อน · ป้าย `(registry)` บอกอยู่แล้วแต่อ่านง่ายเกินจะพลาด

---

## 5. ขั้นตอนปล่อยจริง

```
1. gate เขียวบน branch          (ข้อ 2)
2. เทียบ failure ID กับ baseline  ไม่ใช่เทียบจำนวน
3. บัมพ์เวอร์ชัน + CHANGELOG      (ข้อ 3)
4. merge → main                 --no-ff เพื่อให้ประวัติเห็นว่าเป็นชุดเดียวกัน
5. git tag -a vX.Y.Z
6. git push origin main --follow-tags
7. รอ CI เขียวบน main           ← อย่าเพิ่งบอกลูกค้าให้ update
8. lmds node install --all
9. lmds node list --check       บังคับ probe สด
10. lmds fleet check            ต้องขึ้น "ตรง hub" ครบทุกเครื่อง
```

**ข้อ 7 สำคัญ** — CI รันบน push · เครื่อง dev รัน matrix ครบไม่ได้ · ความมั่นใจจริงมาจาก CI ไม่ใช่จากเครื่องเรา

---

## 6. ถอยกลับเมื่อพัง

`main` พังแล้วมีลูกค้าดึงไปแล้ว:

```bash
git revert -m 1 <merge-commit>     # ไม่ใช่ reset — main เป็น public ห้ามเขียนประวัติทับ
git push origin main
# แล้วบอกลูกค้าให้ lmds node install <ชื่อ> อีกรอบ
```

**ห้าม `push --force` บน `main` ของทั้งสองรีโป** — มีคน clone ไปแล้วเสมอ

controller ที่ regenerate จะเก็บของเดิมไว้ที่ `*.replaced-<เวลา>` เสมอ → ถอยด้วยมือได้

---
## 7. ฟลีตไม่ตรงกันได้ — และเกิดจากอะไร

สถานะจริงของฟลีต ณ ตอนนี้ (ชื่อเครื่อง · IP · เวอร์ชันแต่ละเครื่อง · ใครเป็น hub ของใคร)
อยู่ใน **`docs/FLEET-STATE.md`** ซึ่ง **ไม่ขึ้น GitHub โดยตั้งใจ** — รีโปนี้เป็นสาธารณะ
และตารางนั้นคือรายชื่อเครื่องของลูกค้ากับ login ที่ใช้เข้า ไม่ใช่ของที่ควรอยู่บนอินเทอร์เน็ต
(หลักเดียวกับ `docs/UPGRADE-2026-09.md`) · ทั้งสองไฟล์อยู่ใน `.gitignore`

ส่วนที่อยู่ในเอกสารนี้คือ *กฎ* ที่ได้มาจากสถานะนั้น ซึ่งไม่ขึ้นกับว่าเครื่องไหนชื่ออะไร

### อย่าอัปเดตเครื่องก่อน merge

เคสจริง 2026-09-20: branch งานถูก push เข้าเครื่องจริง **ตรง ๆ ผ่าน SSH** ไม่ผ่าน GitHub
เพื่อทดสอบก่อน แล้วฟลีตก็แตกเป็นสองกลุ่มทันที — 2 เครื่องอยู่ข้างหน้า 16 เครื่องอยู่ข้างหลัง

```bash
# สิ่งที่ทำแล้วเจ็บ
git push ssh://<user>@<node>/<path> landing/2026-09
```

**ความเสี่ยงที่ต้องจำ:** `git pull --ff-only` บน branch ที่ **ไม่มีบน `origin`** จะหา remote ไม่เจอ
→ ถ้าใครรัน `lmds node install` ก่อน merge เสร็จ เครื่องอาจตกกลับไป `main` เก่า **เงียบ ๆ**
ไม่มี error ให้เห็น · เครื่องจะดูปกติทุกอย่างแต่รันโค้ดคนละชุดกับที่เราคิด

**ทางแก้ถาวรมีทางเดียว:** merge → tag → push → `lmds node install --all` จาก hub
แล้วให้ทุกเครื่องกลับมาเกาะ `main` ตามปกติ · อย่าทิ้งเครื่องไว้บน branch ที่ไม่มีบน origin

### hub ซ้อน hub — ข้อจำกัดเชิงโครงสร้างที่ยังไม่มีทางแก้

ฟลีตจริงมีมากกว่าหนึ่งชั้น: node บางเครื่องเป็น **hub ของตัวเอง** และมี node ของมันอยู่ข้างล่างอีกที
hub บนสุด **มองไม่เห็น node ชั้นล่าง และจะไม่เห็นด้วยกลไกปัจจุบัน** ด้วยเหตุผลสองข้อที่แยกกัน:

1. **เครือข่ายไม่ถึง** — node ชั้นล่างมักอยู่บน LAN ของไซต์ลูกค้าเท่านั้น ไม่ได้อยู่บน overlay
   เดียวกับ hub บนสุด · ping ไม่ผ่าน · port 22 ต่อไม่ได้ · LMDS ทำงานผ่าน SSH ล้วน
   จึงเพิ่มเข้าทะเบียนตรง ๆ **ไม่ได้**
2. **ไม่มีกลไกให้ hub เห็น node ของ node** — สองทะเบียนไม่รู้จักกัน · `fleet check` เดินเฉพาะทะเบียนตัวเอง

ข้อ 1 คือเหตุผลที่แข็งที่สุดว่าทำไมต้องมี **"hub คุม hub"**: hub ชั้นกลางเป็นสิ่งเดียวที่เข้าถึง
node ชั้นล่างได้ · hub บนสุดจะอยากจัดการยังไงก็ยิงไม่ถึง ทางเดียวคือ
*ถามมันว่ามีอะไรอยู่ข้างล่าง* แล้วเอามาแสดง

โครงที่เสนอ — ยกหลักการ **"เห็นได้ ≠ เป็นเจ้าของ"** ที่ LMDS ใช้กับ container อยู่แล้ว
(`fleet/manager.py` — stop external = `docker stop` เท่านั้น · repair external = ปฏิเสธ)
ขึ้นมาอีกหนึ่งชั้น จาก container เป็น node:

| | |
|---|---|
| `lmds agent info` | เพิ่ม `child_nodes` — node ที่มีทะเบียนของตัวเองรายงานกลับว่ามีอะไรอยู่ข้างล่าง |
| hub แม่ | แสดง node ลูกเป็นชั้นย่อยใต้ site นั้น ติดป้าย **observed** ไม่ใช่ owned |
| สิทธิ์ | `ps` `logs` `doctor` `bench` ผ่าน hub กลางได้ · `deploy` `push` `set` `remove` **ปฏิเสธ** |
| `fleet check` | ไล่ลงชั้น และบอกว่า node ลูกตรงกับ **hub ที่เป็นเจ้าของมัน** ไม่ใช่ hub บนสุด |
| ยึดเมื่อตั้งใจ | `lmds node adopt --take-over` — คำกริยาเดียวกับที่ `lmds adopt` ใช้กับ container อยู่แล้ว |

ข้อนี้ผูกกับไลเซนส์ด้วย: `LICENSE §1.3` นับ *"เครื่องที่บริหารร่วมกัน"* —
node ที่เป็น observed-not-owned **ไม่ควรถูกนับซ้ำสองฝั่ง**

---

## 8. LiteGate ต่างจาก LMDS ตรงไหน

| | |
|---|---|
| License | **MIT — ฟอร์กได้ ดึงกลับไม่ได้** เวอร์ชันที่ปล่อยแล้วเป็น MIT ตลอดกาล |
| บทบาท | ประตูหน้า · คนใช้เยอะ แล้วเจอปัญหา deploy → มาที่ LMDS |
| กฎเหล็ก | **ห้ามย้ายของที่เคยฟรีไปเสียเงิน** — ชุมชนจะจำและด่ายาว · เพิ่มของใหม่ฝั่งเสียเงินเท่านั้น |
| CI | เข้มกว่า (ruff + docker blocking) แต่ **ไม่มี 3.10** |
| ของที่ไม่ใช่ repo change | การตั้ง `GW_REDIS_URL` ทำที่ `/opt/litegate/.env` บนเครื่อง — **ไม่มี commit** ให้บันทึกใน CHANGELOG ว่าเป็น ops |

---

## 9. รายการที่ต้องแก้ก่อนรีลีสเชิงพาณิชย์จริง

เรียงตามความเร่ง · รายละเอียดอยู่ใน `docs/UPGRADE-2026-09.md`

| # | เรื่อง | ทำไมบล็อก |
|---|---|---|
| 1 | `LMDS_REPO_REF` ปักหมุด tag | ลูกค้าองค์กรไม่ยอมให้ production วิ่งตามปลาย branch |
| 2 | LMDS ยังไม่มี **auth/RBAC/audit** | คำถามแรกของลูกค้าเอกชนคือ "ใครทำอะไรเมื่อไหร่" |
| 3 | model server ออกมาเป็น `0.0.0.0` **ไม่มี API key** | ค่าเริ่มต้นที่อันตรายที่สุดที่ ship อยู่ |
| 4 | `source_bundle()` hardcode `main` | เครื่อง air-gapped อัปเดตผ่าน CLI ไม่ได้ |
| 5 | LMDS ไม่มี lint gate ใน CI | LiteGate มี · ของเราไม่มี |

---

## กฎสั้น ๆ ที่ต้องจำ

1. **`main` ต้องติดตั้งได้เสมอ** — มีคน clone อยู่ตลอดเวลา
2. **เทียบ failure ID ไม่ใช่จำนวน** — 23 ตัวบน macOS เป็นเรื่องปกติ
3. **บัมพ์เวอร์ชันพร้อม tag ในคอมมิตเดียว** — ไม่งั้นเลขเวอร์ชันไม่มีความหมาย
4. **ห้าม force-push บน `main`** ทั้งสองรีโป
5. **CI เขียวก่อน แล้วค่อยบอกลูกค้าให้ update**
6. **ของที่ยังไม่ merge อย่าปล่อยค้างบนเครื่องจริงนาน** — ดูข้อ 7
