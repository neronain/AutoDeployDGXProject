# การติดตั้งและทดสอบบนเครื่องจริง (Ubuntu)

## ติดตั้งตัวโปรแกรม LMDS

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject
./install.sh
```

ต้องการแค่ `python3` ≥ 3.10 + `python3-venv` — ไม่ต้องมี GPU/Docker บนเครื่องที่รัน LMDS
(แต่เครื่องที่จะรัน **bundle** ต้องมี NVIDIA driver + Docker + NVIDIA Container Toolkit)

## ตั้งค่าครั้งแรก

```bash
lmds hardware                        # ตรวจเครื่อง — ควรเห็น GPU + profile ถูกต้อง
lmds config set-provider openai      # หรือ gemini | openai-compat (--base-url http://...:8000/v1 --model ...)
lmds config set-key openai           # ใส่ API key (เก็บ keyring/0600 — ไม่โชว์บนจอ)
lmds config set-hf-token             # (optional) สำหรับ gated repo — ข้ามได้
```

ไม่มี LLM key ก็ใช้ได้: เติม `--no-llm` ทุกคำสั่ง (rule-based mode)

## สร้าง bundle

```bash
# ตัวอย่างเล็กสำหรับทดสอบระบบครั้งแรก (แนะนำ)
lmds deploy "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf"

# เต็มรูป — เลือก target เอง
lmds deploy Qwen/Qwen3-32B --target dgx-spark-single
lmds inspect Qwen/Qwen3-32B --target rtx-pro-4000-dual   # ดู fit ก่อนไม่ generate
```

target presets: `dgx-spark-single`, `dgx-spark-stacked`, `rtx-pro-4000`, `rtx-pro-4000-dual`,
`rtx-4070-super`, `rtx-4070-ti-super`, `rtx-4090`, `rtx-5090` — ไม่ระบุ = ใช้เครื่องปัจจุบัน

## Acceptance test บนเครื่องเป้าหมาย (ปิด M7)

รันตามลำดับใน bundle ที่ได้ แล้วเก็บผลทุกขั้น:

```bash
cd bundles/<slug>
./<slug>-single.sh download
./<slug>-single.sh verify-files
./<slug>-single.sh start          # รอ /health — โมเดลใหญ่ใช้เวลาหลายนาที
./<slug>-single.sh status
./<slug>-single.sh test-text
./<slug>-single.sh client-config
./<slug>-single.sh stop
```

**ถ้าพัง**: เก็บ output ของ `./<slug>-single.sh logs 500` + คำสั่งที่รัน แล้วส่งกลับมาใน session ถัดไป
เพื่อเข้าสู่ repair workflow (ปรับ template/ค่าคงที่ fit ให้ตรงเครื่องจริง)

## อัปเดต LMDS

```bash
cd AutoDeployDGXProject && git pull && ./install.sh
```
