# LMDS · การใช้งานเชิงพาณิชย์

**มีผลแล้ว** — อ้างอิง [LICENSE](../LICENSE) Version 1.0 ซึ่งมีผลตั้งแต่ 20 กันยายน 2026 · เอกสารนี้เป็นคำอธิบายของสัญญานั้น ถ้าขัดกัน **ให้ถือตาม LICENSE** · **[English version ↓](#english)**

| อยากรู้เรื่องอะไร | ไปที่ |
|---|---|
| ต้องจ่ายไหม | [กฎข้อเดียวที่ต้องจำ](#กฎข้อเดียวที่ต้องจำ) · [Tier ตามจำนวนเครื่อง](#tier-ของ-license-ตามจำนวนเครื่อง) |
| "เครื่อง" นับยังไง | [นับ "เครื่อง" ยังไง](#นับ-เครื่อง-ยังไง) |
| อยากลอง stacked | [stacked ต้องใช้ 2 เครื่อง](#-ก่อนอื่น-stacked-ต้องใช้-2-เครื่อง) · [Trial ฟรี 30 วัน](#trial-ฟรี-30-วัน--ระบบออกให้อัตโนมัติ) |
| หมดอายุแล้วอะไรดับบ้าง | [ไม่มีอะไรดับ](#หมดอายุ--อยู่ใน-cooldown-แล้วเกิดอะไรขึ้น--ไม่มีอะไรดับ) · [สัญญาเรา 3 ข้อ](#สัญญาเรา-3-ข้อ-ที่จะไม่ผิดคำพูด) |
| ตัดขาดจากอินเทอร์เน็ต | [องค์กรที่ตัดขาดจากอินเทอร์เน็ต](#องค์กรที่ตัดขาดจากอินเทอร์เน็ต-air-gapped) |

กลไกการนับและสิ่งที่ **ไม่มีวันล็อก** อยู่ที่ [LICENSING.md](LICENSING.md) · ตัวสัญญาอยู่ที่ [LICENSE](../LICENSE)

---

## กฎข้อเดียวที่ต้องจำ

> ## **1 เครื่อง = ฟรี**
> ใช้ทำอะไรก็ได้ **รวมถึงใช้ทำธุรกิจจริงในบริษัท**
> ไม่ต้องมีไฟล์ license · ไม่ต้องลงทะเบียน · ไม่ต้องขออะไรจากเรา
>
> ## **ตั้งแต่ 2 เครื่องที่ดูแลร่วมกัน = ต้องมี license เชิงพาณิชย์**

บริษัทที่รัน production inference บนเครื่องเดียว **จ่าย 0 บาท และมีสิทธิ์ถูกต้องสมบูรณ์**
ไม่ใช่ช่องโหว่ ไม่ใช่การอนุโลม — เราตั้งใจให้เป็นแบบนั้น

**เส้นทางกฎหมายกับเส้นที่ระบบล็อก เป็นเส้นเดียวกัน** จึงไม่มีสภาวะที่ "ระบบยอมให้ทำ
แต่คุณผิดสัญญาอยู่โดยไม่รู้ตัว" — **ถ้ามันรันได้ แปลว่าคุณมีสิทธิ์**

---

## 🔑 ก่อนอื่น: stacked ต้องใช้ 2 เครื่อง

ความสามารถที่เป็นจุดเด่นที่สุดของ LMDS คือ **stacked deployment** — รวมหลายเครื่องเป็นโมเดลเดียว
เพื่อรันโมเดลที่ใหญ่เกินกว่าเครื่องเดียวจะรับไหว

**โดยธรรมชาติของมัน stacked ต้องใช้อย่างน้อย 2 เครื่อง** ซึ่งเกินโควตาฟรี
แปลว่า **คุณประเมิน stacked บน tier ฟรีไม่ได้**

เราบอกตรงนี้ตั้งแต่ต้น ไม่ปล่อยให้ไปเจอกำแพงเอาเอง — และนี่คือเหตุผลหลักที่มี **trial ฟรี 30 วัน
ที่ระบบออกให้อัตโนมัติ** ด้านล่าง **ขอแล้วได้เลย ไม่มีคนมาคั่น ไม่ต้องให้ข้อมูลบัตร**

---

## ของที่ซื้อได้มี 2 อย่าง และแยกจากกันโดยสิ้นเชิง

นี่คือเรื่องที่คนเข้าใจผิดบ่อยที่สุด จึงขอพูดให้ชัดก่อน:

| | **1 · License ตามจำนวนเครื่อง** | **2 · Subscription (catalog + ซัพพอร์ต)** |
|---|---|---|
| ซื้อเพื่ออะไร | รันได้มากกว่า 1 เครื่อง | ได้สูตรที่พิสูจน์บนฮาร์ดแวร์จริง + ช่องทางถาม |
| จำเป็นไหมถ้าใช้ 1 เครื่อง | **ไม่ต้อง** | **ไม่ต้อง — แต่ซื้อได้ถ้าอยากได้** |
| ซื้ออันเดียวได้ไหม | ได้ | **ได้** |

> ### **"ฟรี" คือจำนวนเครื่อง ไม่ใช่เพดานฟีเจอร์**
>
> บริษัทที่รันเครื่องเดียว **ซื้อ subscription ได้เลย โดยไม่ต้องซื้อ license ตามจำนวนเครื่อง**
> อยากได้ catalog กับช่องทางซัพพอร์ต แต่ยังใช้เครื่องเดียวอยู่ — **ยินดีมาก ไม่มีข้อโต้แย้ง**

subscription **ไม่เพิ่มโควตาเครื่อง** และหมดอายุก็ **ไม่ลดโควตาเครื่อง** — สองเรื่องนี้แยกกันจริง ๆ

---

## นับ "เครื่อง" ยังไง

เรานับ **เครื่องที่รันโมเดลได้จริง** เท่านั้น ไม่ได้นับทุกเครื่องที่ติดตั้ง LMDS
LMDS ตรวจเองจากของที่มีอยู่จริงบนเครื่อง (มี `llama-server` ที่ใช้ได้ **หรือ** docker + GPU)

| กรณี | นับเป็น |
|---|---|
| **hub ที่ไม่มี GPU** — เครื่องที่คุณวางแผน generate bundle push และเปิดหน้าเว็บ | **0 · ฟรี ไม่นับ** |
| เครื่องจริง 1 เครื่องที่รันโมเดลได้ | **1** |
| **1 เครื่องที่มี GPU 8 ใบ** | **1** — เรานับเครื่อง ไม่นับการ์ด |
| **หลาย container / หลายโมเดลบนเครื่องเดียว** | **1** — container ไม่ใช่เครื่อง |
| VM หรือ cloud instance ที่รันโมเดลได้ | **1 ต่อตัว** |
| VM 8 ตัวบนเครื่องจริงเครื่องเดียว | **8** — virtualization ไม่ได้ลดจำนวน |
| instance ชั่วคราว / autoscale | นับตอนที่มันมีอยู่ (**สูงสุดพร้อมกัน** ไม่ใช่ยอดสะสม) |
| **เปลี่ยนเครื่องใหม่แทนเครื่องเก่าที่พัง** | **ไม่นับเพิ่ม** (เครื่องเดิมต้องหยุดเสิร์ฟใน 30 วัน) |

**ชุดที่คนใช้ฟรีใช้กันจริง คือ hub ไม่มี GPU + เครื่องรันโมเดล 1 เครื่อง = 1 เครื่อง ฟรี**
เครื่องที่คุณนั่งทำงานอยู่ ไม่ได้ถูกนับ ถ้ามันไม่ได้รันโมเดลเอง

### นับเฉพาะเครื่องที่ "ดูแลร่วมกัน"

โควตานับเฉพาะเครื่องที่ **ดูแลร่วมกันเป็นฟลีตเดียว** — อยู่ใน inventory เดียวกัน
deploy/push/clone ถึงกัน หรือมอนิเตอร์จากที่เดียวกัน

**การติดตั้งแยกกันสองชุด ที่ต่างคนต่างดูแลเครื่องของตัวเองอย่างละเครื่อง — ฟรีทั้งคู่**

เราพูดตรง ๆ ว่านี่คือขอบเขตที่ตั้งใจเขียนไว้ ไม่ใช่ช่องโหว่ที่เผลอ: คนที่อยากได้คลัสเตอร์จริง
หรือแค่อยากคุมหลายเครื่องจากที่เดียว — ซึ่งคือเหตุผลหลักที่ LMDS มีอยู่ — ต้องให้เครื่องดูแลร่วมกันอยู่ดี
และตรงนั้นคือจุดที่ license มีผล

---

## Tier ของ license ตามจำนวนเครื่อง

| Tier | เครื่องที่รันโมเดลได้ (ดูแลร่วมกัน) | เหมาะกับใคร | ราคา |
|---|---|---|---|
| **Free** | **1** | ทุกคน — นักศึกษา คนเล่นที่บ้าน นักพัฒนา **และบริษัทที่รัน production บนเครื่องเดียว** | **ฟรี ตลอดไป** |
| **Trial** | **ไม่จำกัด · 30 วัน** | คนที่กำลังประเมิน — โดยเฉพาะคนที่อยากลอง **stacked** | **ฟรี · ระบบออกให้อัตโนมัติ** |
| **Team** | สูงสุด 8 | SME · แล็บ · ทีมภายในบริษัท | ขอใบเสนอราคา |
| **Enterprise** | ไม่จำกัด + air-gap + SLA | ธนาคาร · หน่วยงานรัฐ · โรงพยาบาล | ขอใบเสนอราคา |
| **Partner / OEM** | ไม่จำกัด + **ออก sub-license ได้** | SI ที่ติดตั้งให้ลูกค้าหลายราย | ขอใบเสนอราคา |

**subscription ซื้อเพิ่มได้ทุกแถวในตารางนี้ รวมถึงแถว Free** — ดูหัวข้อถัดไป

---

## Subscription — catalog + ซัพพอร์ต (ซื้อได้ทุกระดับ รวมถึงคนใช้ฟรี)

ไม่ใช่ค่าโปรแกรม แต่เป็นค่า **catalog สูตรที่พิสูจน์บนฮาร์ดแวร์จริงแล้ว** และการอัปเดตของมัน

`recipes/` คือความรู้ที่ได้จากการเผาเครื่องจริง ไม่ใช่จากการอ่านเอกสาร ของจริงเช่น:

- โมเดลบางตระกูลต้องใช้ parser คนละตัวกับที่เอกสารบอก ใช้ผิด = tool calling เงียบ ๆ ไม่ทำงาน
- บางรุ่นเปิด prefix-caching แล้วพัง แบบไม่มี error ขึ้นให้เห็น
- KV cache บางรูปแบบบน SM121 พังเงียบ — ได้คำตอบผิดโดยไม่มีอะไรเตือน
- image tag บางตัวเป็น tag ผี ดึงมาได้แต่ไม่มี kernel ของสถาปัตยกรรมคุณ
- NVFP4 MoE COW break · GPU clock latch · GID index drift

**ความรู้แบบนี้ลอกไม่ได้ เพราะมันไม่ใช่โค้ด** — เดือนหน้า vLLM ออกรุ่นใหม่แล้วสูตรเดิมพัง
และเราคือคนที่รันจนเจอก่อน

สิ่งที่ subscription ให้:

- **อัปเดต catalog สม่ำเสมอ** — สูตรใหม่ · สูตรที่แก้แล้ว · image digest ที่ pin ไว้ให้
- **รู้ว่าอะไรพังก่อนที่คุณจะเจอเอง** พร้อมค่าที่ใช้แทนได้
- **สถานะ hardware-validated ที่ตรวจสอบได้** — ทุกตัวเลขมี `source:` และ `validated_on:`
  บอกว่ามาจากการรันจริงบนเครื่องอะไร เมื่อไหร่ · **เราไม่เขียน hardware-validated
  โดยไม่มีการรันจริง**
- **ช่องทางถามตรง** เมื่อ deploy ไม่ขึ้นแล้วหาสาเหตุไม่เจอ

**ซื้อได้แม้อยู่ tier Free** — รันเครื่องเดียว จ่ายค่า license 0 บาท แต่รับ catalog กับซัพพอร์ตได้
Tier ที่สูงกว่าได้เวลาตอบกลับเร็วกว่า · Enterprise ได้ SLA เป็นลายลักษณ์อักษร

**ราคา:** ขอใบเสนอราคา

---

## Trial ฟรี 30 วัน — ระบบออกให้อัตโนมัติ

- **ปลดล็อกไม่จำกัดจำนวนเครื่อง 30 วัน**
- **ขอแล้วได้เลย — ระบบออกให้เอง ไม่มีคนมาคั่น ไม่ต้องคุยกับฝ่ายขาย ไม่ต้องให้ข้อมูลบัตร**
- ไฟล์ license เป็น **ไฟล์ลงนามแบบออฟไลน์** ใช้บนเครื่องที่ไม่ต่อเน็ตได้ตามปกติ

### เว้นระยะ 7 วันระหว่าง trial

trial หมดอายุแล้ว **ขอใหม่ได้เมื่อครบ 7 วัน** — เราเขียนกติกานี้ไว้ล่วงหน้าใน `LICENSE` §4.1
เพื่อให้คุณรู้ก่อน ไม่ใช่ไปเจอตอนถูกปฏิเสธ

ระหว่าง 7 วันนั้น: โควตากลับเป็น **1 เครื่อง เฉพาะงานใหม่** และ **ไม่มีอะไรถูกปิด** (ดูหัวข้อถัดไป)

**ต้องประเมินนานกว่า 30 วัน? ถามมา เราต่อให้** ไม่ต้องรอ cooldown —
cooldown มีไว้ให้ trial เป็นระบบอัตโนมัติได้ ไม่ได้มีไว้กันคนที่ตั้งใจประเมินจริง

### หมดอายุ / อยู่ใน cooldown แล้วเกิดอะไรขึ้น — ไม่มีอะไรดับ

- ❌ deploy / push / clone **ใหม่** เกิน 1 เครื่อง — ทำไม่ได้
- ✅ **โมเดลที่รันอยู่ ยังรันต่อทุกเครื่อง**
- ✅ `start` `stop` `restart` `logs` `ps` `doctor` `status` — **ใช้ได้ครบทุกเครื่อง**
- ✅ **autostart ตอนบูตยังทำงาน**

เครื่องที่รีบูตตีสาม หลัง trial หมดอายุเมื่อวาน **กลับขึ้นมาเหมือนเดิม**
ข้อนี้เขียนไว้ใน `LICENSE` §10 เป็นข้อสัญญา ไม่ใช่คำโฆษณา

---

## ถ้า LMDS แจ้งว่าเจอ LMDS อีกตัวในเครือข่าย

LMDS อาจขึ้นข้อความว่า *"พบ LMDS อีกตัวที่ 10.0.0.5"* — เราอยากให้เข้าใจตรงกันว่ามันคืออะไร
และไม่ใช่อะไร (เขียนเป็นข้อสัญญาไว้ใน `LICENSE` §1.6 แล้ว):

- **ตรวจภายในเครือข่ายของคุณเท่านั้น — ไม่มีการส่งข้อมูลอะไรกลับมาหาเรา**
  ไม่ส่งข้อมูลเครื่อง เครือข่าย โมเดล หรือการใช้งานไปที่ไหนทั้งสิ้น
  **LMDS ไม่มี telemetry** (เขียนไว้ใน `SECURITY.md`) และข้อนี้ไม่ใช่ช่องทางส่งข้อมูล
- **เป็นข้อมูล ไม่ใช่การบังคับ** — ไม่บล็อกอะไรเลย ไม่เปลี่ยนสิทธิ์ของคุณ
- **เจออีกตัว ≠ คุณเกินโควตา** — การติดตั้งที่แยกกันและไม่ได้ดูแลร่วมกัน **ถูกต้องอยู่แล้ว**
  สองชุดในวงเดียวกันฟรีทั้งคู่

มันมีไว้บอกเฉย ๆ ว่ามี LMDS อีกตัวอยู่ เผื่อคุณอยากให้สองเครื่องทำงานเป็นฟลีตเดียวกัน
(ซึ่งเป็นฟีเจอร์ที่ต้องมี license และเป็นเหตุผลที่จะคุยกับเรา) — **ไม่ใช่การกล่าวหาว่าคุณผิดสัญญา**

---

## ต้องมีสัญญาเสมอ ไม่ว่าจะกี่เครื่อง

บางอย่างต้องมีสัญญาเชิงพาณิชย์ **แม้จะทำบนเครื่องเดียว** เพราะมันคือการเอา LMDS ไปอยู่ในมือลูกค้าคุณ:

| สถานการณ์ | ต้องมีสัญญา |
|---|---|
| ใช้ LMDS ส่งมอบสินค้า/บริการให้บุคคลที่สาม (เก็บเงินหรือไม่ก็ตาม) | ✅ แม้เครื่องเดียว |
| SI · ที่ปรึกษา · MSP ติดตั้งหรือดูแลให้ลูกค้า | ✅ แม้เครื่องเดียว |
| แจกจ่ายต่อ · ขายต่อ · ทำเป็นบริการให้คนอื่นใช้ (hosting) | ✅ |
| white-label · rebrand · ฝังใน product ของตัวเอง · OEM | ✅ |
| ออกสิทธิช่วง (sub-license) ให้ลูกค้าของคุณ | ✅ tier Partner/OEM |
| บริษัทรัน production ภายในของตัวเองบน **1** เครื่อง | ❌ **ฟรี** |
| บริษัทรัน production ภายในของตัวเองบน **3** เครื่องที่ดูแลร่วมกัน | ✅ tier Team |

**ไม่แน่ใจว่าอยู่ฝั่งไหน — ถามมา** เราตอบตรง ๆ และ **ไม่มีการเรียกเก็บย้อนหลังกับคนที่ถามก่อน**

---

## สัญญาเรา 3 ข้อ ที่จะไม่ผิดคำพูด

**1 · ของที่กันคุณเจ็บตัว ฟรีตลอดกาล**
quality gates · `doctor` · `repair` · `validate` · `preflight` · การปิดบัง secret · allowlist ·
ใบอนุมัติก่อนรันคำสั่งเสี่ยง — **ไม่มีวันอยู่หลังกำแพงเงิน** เขียนไว้ใน `LICENSE` §5 แล้ว
คนที่ใช้ฟรีแล้ว deploy พังจนโดนเจาะ จะจำชื่อเรา ไม่ใช่จำว่าเขาไม่ได้จ่าย

**2 · ของที่รันอยู่ จะไม่ถูกสั่งหยุด**
license หมดอายุ หรืออยู่ใน cooldown = จำกัดเฉพาะ **งานใหม่** · ของที่รันอยู่ยังรัน ควบคุมได้ครบ
รีบูตแล้วกลับมา · ถ้าลูกค้ารีบูตตีสามแล้ว autostart ไม่ขึ้นเพราะ license หมดเมื่อวาน
นั่นคือการเสียลูกค้าถาวร เราจะไม่ทำ และเขียนเป็นข้อสัญญาไว้แล้ว (`LICENSE` §10)

**3 · ล็อก "ขนาด" ไม่ล็อก "ความสามารถ"**
คนใช้ฟรีทำได้ครบทั้งลูปบนเครื่องเดียว ตั้งแต่ลิงก์ Hugging Face จนถึงเซิร์ฟเวอร์ที่ยิงได้จริง
ไม่มีฟีเจอร์ไหนถูกตัดออกเพื่อบีบให้อัปเกรด — ที่ต่างกันคือ **จำนวนเครื่อง** เท่านั้น
(ข้อยกเว้นเดียวคือ stacked ซึ่งต้องใช้ 2 เครื่องโดยธรรมชาติของมันเอง — ดูหัวข้อด้านบน)

---

## องค์กรที่ตัดขาดจากอินเทอร์เน็ต (air-gapped)

ตลาดที่ LMDS ไปคือ DGX ในองค์กร ซึ่ง **คือตลาด air-gapped** เราจึงออกแบบให้:

- **license เป็นไฟล์ลงนามแบบออฟไลน์ล้วน — ไม่มี phone-home บังคับ**
- เครื่องที่ไม่ต่อเน็ตเลย ใช้ license ได้ตามปกติ ไม่ต้องขอยกเว้นอะไร
- **ไม่ผูกกับฮาร์ดแวร์เป็นค่าเริ่มต้น** — เปลี่ยนเมนบอร์ดแล้วไม่ต้องโทรหาเรา
  (มีเป็นตัวเลือกเฉพาะเคส OEM ที่ขอมาเอง)
- จุดบังคับใช้ license **อยู่ที่เดียว อ่านได้ ตรวจได้** — repo เป็น public
  ทีม audit ขององค์กรเปิดดูเองได้ว่าล็อกอะไรบ้าง ไม่ต้องเชื่อคำพูดเรา
- **การตรวจเจอ LMDS ตัวอื่นในเครือข่าย ก็เป็นการตรวจภายในล้วน** ไม่ส่งอะไรออกนอกองค์กร

LMDS **ไม่มี telemetry** และจะไม่มี — เขียนไว้ใน `SECURITY.md` และเป็นเหตุผลที่เราไม่ใช้ระบบ activate ออนไลน์

---

## ตราประทับบน bundle — ไม่ใช่การล็อก

bundle ที่ generate ออกมามี `licensed_to` · `license_id` · `attestation` ใน
`MODEL_PROFILE.yaml` และหัว controller

**นี่ไม่ใช่การล็อก** — `LICENSE` §9 ระบุว่า **bundle เป็นของคนที่ generate มัน** ใช้ แก้ ส่งต่อได้อิสระ
ไม่ขึ้นกับ tier ไม่ต้องมีไฟล์ license หรือ subscription และไม่หายไปแม้สัญญาจะสิ้นสุด
ตราประทับบอก **ที่มา** เท่านั้น ประโยชน์จริง: ถ้าพาร์ตเนอร์เอางานไปขายต่อในนามอื่น ตรวจได้ทันที

---

## ติดต่อ

**neronain** — https://www.facebook.com/neronain.minidev

- **ขอ trial:** ขอผ่านระบบได้เลย **ออกให้อัตโนมัติ ไม่ต้องให้ข้อมูลบัตร**
- **ขอ subscription (รวมถึงตอนใช้ฟรี 1 เครื่อง):** บอกมาว่าใช้เครื่องอะไร รันอะไรอยู่
- **ขอใบเสนอราคา license:** มีกี่เครื่อง · เครื่องอะไร (DGX Spark / RTX) · จะเอาไปทำอะไร ·
  ต่อเน็ตได้หรือ air-gapped

**ใช้ฟรีบนเครื่องเดียวอยู่แล้ว ไม่ต้องติดต่อเราเลยก็ได้** — ติดต่อเมื่อคุณอยากได้เครื่องที่ 2
หรืออยากได้ catalog

---

# English

**In force** — see [LICENSE](../LICENSE) Version 1.0, in force from 20 September 2026. This page explains that licence; **where the two differ, the LICENSE governs**. · **[ฉบับภาษาไทย ↑](#lmds--การใช้งานเชิงพาณิชย์)**

| What you want to know | Go to |
|---|---|
| Do I have to pay | [The rule, in one line](#the-rule-in-one-line) · [Licence tiers](#machine-count-licence-tiers) |
| What counts as a machine | [What counts as a machine](#what-counts-as-a-machine) |
| I want to try stacked | [Stacked needs two machines](#-first-something-you-should-know-stacked-needs-two-machines) · [The free 30-day trial](#the-free-30-day-trial--issued-automatically) |
| What stops when it expires | [Nothing stops](#what-happens-at-expiry-or-during-a-cooldown--nothing-stops) · [Three promises](#three-promises-we-will-not-break) |
| We are air-gapped | [Air-gapped organisations](#air-gapped-organisations) |

How counting actually works, and what is **never locked**, is in [LICENSING.md](LICENSING.md);
the agreement itself is [LICENSE](../LICENSE).

## The rule, in one line

> ## **One machine = free**
> For **any** purpose, **including running your business on it**.
> No licence file, no registration, nothing to ask us for.
>
> ## **Two or more machines managed together = commercial licence**

A company running its production inference on one serving machine **pays
nothing and is fully licensed**. That is not a loophole or an indulgence — it
is the design.

The legal line and the technical gate are the same line, so there is no state
where the software runs normally but you are quietly in breach. **If it runs,
you are licensed.**

## 🔑 First, something you should know: stacked needs two machines

LMDS's flagship capability is **stacked deployment** — combining machines into
a single model, so you can run models too large for any one box.

**By its nature that needs at least two machines**, which is above the free
limit. **You cannot evaluate stacked on the free tier.**

We are telling you that up front rather than letting you hit a wall. It is the
main reason the **free 30-day trial** below is **issued automatically** — ask
and you have it, with no human in the loop and no payment details.

## There are two things you can buy, and they are independent

This is the most commonly misunderstood part, so here it is first:

| | **1 · Machine-count licence** | **2 · Subscription (catalog + support)** |
|---|---|---|
| What it buys | Running more than one machine | Hardware-proven recipes + a support channel |
| Needed on one machine? | **No** | **No — but you can buy it if you want it** |
| Can I buy just this one? | Yes | **Yes** |

> ### **Free is a machine count, not a feature ceiling.**
>
> A one-machine company can **buy the subscription without buying a
> machine-count licence**. You want the catalog and a support channel but you
> are staying on one machine — **that is welcome, and there is no objection.**

A subscription **does not raise your machine limit**, and letting it lapse
**does not lower it**. The two are genuinely separate.

## What counts as a machine

We count **serving machines** — machines that can actually run a model (a
working `llama-server`, or Docker plus at least one NVIDIA GPU). LMDS detects
this itself, from what is really on the machine.

| Situation | Counts as |
|---|---|
| **A GPU-less hub** — where you plan, generate bundles, push, and open the web console | **0 · free, not counted** |
| One physical machine that can serve | **1** |
| **One machine with 8 GPUs in it** | **1** — we count machines, not cards |
| **Many containers / many models on one machine** | **1** — containers are not machines |
| A VM or cloud instance that can serve | **1 each** |
| Eight serving VMs on one physical box | **8** — virtualising does not reduce it |
| Ephemeral / autoscaled instances | counted while they exist (**peak concurrent**, not cumulative) |
| **Replacing a dead machine with a new one** | **no increase** (the old one stops serving within 30 days) |

**The common free setup is a GPU-less hub plus one serving node — that is one
machine, and it is free.** The laptop you work from is not counted if it does
not serve models itself.

### Only machines managed together are counted

The limit counts machines **managed together as one fleet** — in one
inventory, deployed to, pushed to, cloned between, or monitored from one place.

**Two separate installations, each managing only its own single machine, are
each free.**

We will say plainly that this is a deliberate boundary rather than an
oversight: anyone who wants a real cluster, or simply wants to run more than
one machine from one place — which is most of what LMDS is for — needs them
managed together, and that is where the licence applies.

## Machine-count licence tiers

| Tier | Serving machines (managed together) | Who it is for | Price |
|---|---|---|---|
| **Free** | **1** | Everyone — students, home users, developers, **and companies running production on one machine** | **Free, permanently** |
| **Trial** | **Unlimited · 30 days** | Anyone evaluating — especially to try **stacked** | **Free · issued automatically** |
| **Team** | Up to 8 | SMEs, labs, teams inside a company | Contact us for a quote |
| **Enterprise** | Unlimited + air-gap + SLA | Banks, government, hospitals | Contact us for a quote |
| **Partner / OEM** | Unlimited + **may issue sub-licences** | Integrators deploying for many customers | Contact us for a quote |

**The subscription can be added to any row in this table, including Free.**

## Subscription — catalog + support, available at every tier including Free

Not the program — the **hardware-proven recipe catalog** and its updates.

`recipes/` is knowledge earned by burning real hardware, not by reading docs:

- some model families need a different parser than the documentation says —
  get it wrong and tool calling silently stops working;
- some releases break when prefix caching is enabled, with no error shown;
- certain KV cache layouts fail silently on SM121 — wrong answers, no warning;
- some image tags are ghost tags: they pull, but carry no kernel for your
  architecture;
- NVFP4 MoE COW break, GPU clock latch, GID index drift.

**This cannot be copied, because it is not code.** Next month vLLM ships a
release that breaks the old recipe, and we are the ones who ran it until it
broke.

A subscription gives you:

- **regular catalog updates** — new recipes, corrected recipes, pinned image
  digests;
- **advance warning of what is broken**, with working values to use instead;
- **verifiable hardware-validated status** — every figure carries `source:` and
  `validated_on:` naming the machine and the date. **We do not write
  hardware-validated without a real run.**
- **a direct channel** when a deployment will not come up and you cannot find
  out why.

**Available on the Free tier.** Run one machine, pay nothing for the licence,
and still get the catalog and a support channel. Higher tiers get faster
response times; Enterprise gets a written SLA.

**Price:** contact us for a quote.

## The free 30-day trial — issued automatically

- **Unlocks unlimited serving machines for 30 days.**
- **Ask and the system issues it — no human in the loop, no sales
  conversation, and no payment details.**
- The licence is an **offline signed file** and works on machines with no
  internet at all.

### A 7-day cooldown between trials

After a trial expires, the next one is issued **7 days later**. We wrote that
rule into `LICENSE` §4.1 in advance so you know it rather than meeting it as a
refusal.

During those 7 days your limit is **one machine, for new operations only**, and
**nothing you are running is switched off** (below).

**Need longer than 30 days? Ask — we will arrange an extension** rather than
make you wait out a cooldown. The cooldown exists to keep the trial
self-service, not to obstruct a serious evaluation.

### What happens at expiry or during a cooldown — nothing stops

- ❌ new `deploy` / `push` / `clone` beyond one machine
- ✅ **every model already deployed keeps running, on every machine**
- ✅ `start`, `stop`, `restart`, `logs`, `ps`, `doctor`, `status` on all of them
- ✅ **automatic start at boot still works**

A machine that reboots at 3am after a trial expired yesterday comes back up.
That is written into `LICENSE` §10 as a binding term, not a marketing promise.

## If LMDS tells you it found another LMDS on your network

LMDS may show *"found another LMDS at 10.0.0.5"*. Here is what that is, and
what it is not — all of it binding in `LICENSE` §1.6:

- **Detection inside your own network only — nothing is reported back to us.**
  No information about your machines, network, models or usage is sent
  anywhere. **LMDS has no telemetry** (see `SECURITY.md`), and this is not a
  channel for any.
- **It is information, not enforcement.** It blocks nothing and changes nothing
  about what you may do.
- **Finding another install does not mean you are over the limit.** Separate
  installations that are not managed together are **already compliant** — two
  independent installs on one network are each free.

It exists only to mention that another LMDS is there, in case you would like
the two to work as one fleet — which is a licensed feature and a reason to talk
to us. **It is never an accusation that you are in breach.**

## Always needs an agreement, at any machine count

Some things need a commercial agreement **even on one machine**, because they
put LMDS in your customers' hands:

| Situation | Agreement needed |
|---|---|
| Using LMDS to deliver a product or service to third parties (paid or not) | ✅ even on one machine |
| Integrators, consultancies and MSPs installing or operating it for clients | ✅ even on one machine |
| Redistribution, resale, or hosting it as a service | ✅ |
| White-labelling, rebranding, embedding in your own product, OEM | ✅ |
| Issuing sub-licences to your own customers | ✅ Partner/OEM tier |
| A company running its own internal production on **1** machine | ❌ **free** |
| A company running its own internal production on **3** machines managed together | ✅ Team tier |

**Not sure which side you are on? Ask.** We answer straight, and **nobody who
asks first gets a back-dated bill.**

## Three promises we will not break

**1 · Anything that stops you getting hurt is free forever.**
Quality gates, `doctor`, `repair`, `validate`, `preflight`, secret redaction,
allowlists and approval prompts will never sit behind a paywall — written into
`LICENSE` §5. A free user whose deployment breaks and gets compromised will
remember our name, not their invoice.

**2 · Nothing already running is ever switched off.**
An expired licence, or a cooldown, limits **new** work only. What is running
keeps running, stays fully controllable, and comes back after a reboot. If a
customer reboots at 3am and autostart fails because a licence lapsed yesterday,
that customer is gone forever. We will not do that, and we wrote it into the
licence (§10).

**3 · We limit size, not capability.**
Free users get the complete loop on one machine — Hugging Face link to a server
that actually serves. No feature is withheld to force an upgrade. The only
difference between tiers is **how many machines**. (The one exception is
stacked, which needs two machines by its own nature — see above.)

## Air-gapped organisations

LMDS targets DGX systems inside organisations, and **that is an air-gapped
market**. So:

- **licences are offline signed files — no mandatory phone-home**;
- a machine with no internet at all works normally, with no exception process;
- **no hardware binding by default** — replace a mainboard without calling us
  (an option only for OEM cases that ask for it);
- licence enforcement lives in **one auditable place**, in a **public
  repository** — your audit team can read exactly what is gated instead of
  taking our word for it;
- **same-network detection is purely internal too** — nothing leaves your
  organisation.

LMDS has **no telemetry** and will not gain any. That is written in
`SECURITY.md`, and it is why we do not use online activation.

## Bundle stamps are provenance, not locks

Generated bundles carry `licensed_to`, `license_id` and `attestation` in
`MODEL_PROFILE.yaml` and in the controller header.

**This is not a lock.** `LICENSE` §9 says the **bundle belongs to whoever
generated it** — free to use, modify and redistribute, regardless of tier, with
no licence file or subscription needed, and that does not end when an agreement
does. The stamp records origin. Its real purpose: if a partner resells work
under someone else's name, it is immediately visible.

## Contact

**neronain** — https://www.facebook.com/neronain.minidev

- **For a trial:** request it — **issued automatically, no payment details.**
- **For a subscription (including while free on one machine):** tell us what
  hardware you have and what you are running.
- **For a licence quote:** how many machines, what they are (DGX Spark / RTX),
  what you want to run, and whether you are connected or air-gapped.

**If you are running free on one machine, you never need to contact us at
all.** Get in touch when you want a second machine — or the catalog.
