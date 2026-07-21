# คู่มือติดตั้ง LMDS (ละเอียด)

> เอกสารนี้สำหรับผู้ติดตั้งครั้งแรกบนเครื่อง Ubuntu — อ่านคู่กับ [USAGE.md](USAGE.md) (วิธีใช้งานหลังติดตั้ง)

## ภาพรวม: ต้องติดตั้งอะไรที่เครื่องไหน

| เครื่อง | ต้องมี | หมายเหตุ |
|---|---|---|
| เครื่องที่รัน **LMDS** (ตัวสร้าง bundle) | Python ≥ 3.10, git | ไม่ต้องมี GPU ก็ได้ |
| เครื่องที่รัน **bundle** (ตัวเสิร์ฟโมเดล) | NVIDIA driver, Docker, NVIDIA Container Toolkit | ต้องมี GPU |

> กรณีทั่วไปคือ **เครื่องเดียวกัน** (เช่น DGX Spark หรือเครื่อง RTX) — ติดตั้งครบทั้งสองส่วนในเครื่องนั้น

รองรับ: Ubuntu 22.04 / 24.04 ทั้ง x86_64 (RTX) และ ARM64 (DGX Spark GB10)

---

## ส่วนที่ 1 — เตรียมเครื่อง (Prerequisites)

### 1.1 ตรวจ Python

```bash
python3 --version        # ต้องได้ 3.10 ขึ้นไป
```

ถ้าไม่มีหรือขาดโมดูล venv:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git
```

### 1.2 ตรวจ NVIDIA driver (เฉพาะเครื่องที่จะรันโมเดล)

```bash
nvidia-smi
```

ต้องเห็นตาราง GPU ของเครื่อง (เช่น `NVIDIA RTX PRO 4000 Blackwell`) — ถ้าไม่เห็น ให้ติดตั้ง driver ก่อน:

```bash
sudo ubuntu-drivers install    # แล้ว reboot
```

> DGX Spark: driver มากับ DGX OS อยู่แล้ว ปกติข้ามขั้นนี้ได้

### 1.3 ติดตั้ง Docker (ถ้ายังไม่มี)

```bash
docker --version || curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # ให้ user ปัจจุบันใช้ docker ได้โดยไม่ต้อง sudo
```

**สำคัญ**: หลัง `usermod` ต้อง **logout/login ใหม่** (หรือ `newgrp docker`) จึงจะมีผล

### 1.4 ติดตั้ง NVIDIA Container Toolkit (ให้ Docker เห็น GPU)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1.5 ทดสอบว่า Docker เห็น GPU (ขั้นชี้ขาด — ห้ามข้าม)

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

- ✅ เห็นตาราง GPU เหมือนรัน `nvidia-smi` บน host → เครื่องพร้อม
- ❌ error `could not select device driver` → ย้อนกลับไปทำข้อ 1.4 ใหม่
- ❌ error `permission denied ... docker.sock` → ย้อนกลับไปทำข้อ 1.3 (usermod + logout/login)

---

## ส่วนที่ 2 — ติดตั้ง LMDS

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject
./install.sh
```

สคริปต์จะ:
1. ตรวจ Python ≥ 3.10 และโมดูล venv (แจ้ง error พร้อมวิธีแก้ถ้าขาด)
2. สร้าง virtualenv ที่ `~/.local/share/lmds/venv` (ไม่ยุ่งกับ Python ระบบ)
3. ติดตั้ง lmds ลง venv นั้น
4. symlink คำสั่ง `lmds` ไปที่ `~/.local/bin/lmds`

### ตรวจว่าติดตั้งสำเร็จ

```bash
lmds version
```

ควรได้:

```text
lmds 0.1.0
template standard: dgx-spark-controllers-v3.0.0
```

### ถ้าขึ้น "lmds: command not found"

`~/.local/bin` ยังไม่อยู่ใน PATH — แก้โดย:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## ส่วนที่ 3 — ตั้งค่าครั้งแรก (ทำครั้งเดียวต่อเครื่อง)

### 3.1 ตรวจฮาร์ดแวร์

```bash
lmds hardware
```

ตัวอย่างผลบนเครื่อง RTX PRO 4000 สองใบ:

```text
Arch        x86_64
GPU 0       NVIDIA RTX PRO 4000 Blackwell (24 GB, SM120) ✅ tested
GPU 1       NVIDIA RTX PRO 4000 Blackwell (24 GB, SM120) ✅ tested
RAM         128.0 GB
Docker      ✅
NVIDIA Container Toolkit  ✅
Profile     rtx-multi-gpu
```

เช็ค 3 จุด: (1) เห็น GPU ครบทุกใบ (2) Docker ✅ ทั้งสองบรรทัด (3) Profile ตรงกับเครื่องจริง

### 3.2 ตั้ง LLM provider (สมองของระบบ)

เลือกอย่างใดอย่างหนึ่ง:

```bash
# แบบ A: OpenAI
lmds config set-provider openai
lmds config set-key openai            # วาง API key แล้ว Enter (จอไม่แสดงตัวอักษร — ปกติ)

# แบบ B: Google Gemini
lmds config set-provider gemini
lmds config set-key gemini

# แบบ C: ใช้โมเดล local ในองค์กรเป็นสมอง (endpoint แบบ OpenAI-compatible)
lmds config set-provider openai-compat --base-url http://10.100.152.1:8000/v1 --model qwen3-coder
lmds config set-key openai-compat

# แบบ D: ไม่มี key เลย — ไม่ต้องตั้งอะไร แล้วเติม --no-llm ทุกครั้งที่ deploy (rule-based mode)
```

### 3.3 (ทางเลือก) HF token — เฉพาะเมื่อใช้โมเดล gated

โมเดลอย่าง Llama ต้องกดยอมรับเงื่อนไขบนเว็บ Hugging Face ก่อน แล้วเอา token มาใส่:

```bash
lmds config set-hf-token      # ข้ามได้ — โมเดลสาธารณะไม่ต้องใช้
```

### 3.4 ตรวจการตั้งค่าทั้งหมด

```bash
lmds config show      # key ทุกตัวถูก mask — ปลอดภัยต่อการแคปหน้าจอส่งกัน
```

---

## การอัปเดตเวอร์ชัน

```bash
cd AutoDeployDGXProject
git pull
./install.sh          # รันซ้ำได้เลย — config/key เดิมอยู่ครบ
```

## การถอนการติดตั้ง

```bash
rm -rf ~/.local/share/lmds ~/.local/bin/lmds
rm -rf ~/.config/lmds        # ลบ config + key ด้วย (ถ้าต้องการ)
```

---

ติดตั้งเสร็จแล้ว → ไปต่อที่ **[USAGE.md](USAGE.md)** เพื่อ deploy โมเดลตัวแรก
