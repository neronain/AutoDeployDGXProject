# คู่มือติดตั้ง LMDS (ละเอียด)

> เอกสารนี้สำหรับผู้ติดตั้งครั้งแรกบนเครื่อง Ubuntu — อ่านคู่กับ [USAGE.md](USAGE.md) (วิธีใช้งานหลังติดตั้ง)

## ภาพรวม: ต้องติดตั้งอะไรที่เครื่องไหน

| เครื่อง | ต้องมี | หมายเหตุ |
|---|---|---|
| เครื่องที่รัน **LMDS** (ตัวสร้าง bundle) | Python ≥ 3.10, git | ไม่ต้องมี GPU ก็ได้ · ใช้ดิสก์ < 200 MB |
| เครื่องที่รัน **bundle** (ตัวเสิร์ฟโมเดล) | NVIDIA driver, Docker, NVIDIA Container Toolkit | ต้องมี GPU · ต้องมีดิสก์ว่างพอสำหรับ **runtime image + น้ำหนักโมเดล** (ดู §1.6) |

> กรณีทั่วไปคือ **เครื่องเดียวกัน** (เช่น DGX Spark หรือเครื่อง RTX) — ติดตั้งครบทั้งสองส่วนในเครื่องนั้น

รองรับ: Ubuntu 22.04 / 24.04 ทั้ง x86_64 (RTX) และ ARM64 (DGX Spark GB10)

**LMDS ไม่ได้ติดตั้ง vLLM หรือ llama.cpp ให้ตอนนี้** — ตัว engine จะถูกดึง/สร้างตอนรัน bundle
(vLLM = docker image, llama.cpp = docker image หรือ build จาก source) รายละเอียดทั้งหมดอยู่ที่ **[§4](#ส่วนที่-4--โมเดล-local-ถูกดึงมาและรันอย่างไร-vllm--llamacpp)**

---

## ส่วนที่ 1 — เตรียมเครื่อง (Prerequisites)

> **ทางลัด: ข้ามส่วนนี้ไปที่ [ส่วนที่ 2](#ส่วนที่-2--ติดตั้ง-lmds) ได้เลย**
> `install.sh` ตรวจและ**ติดตั้งของที่ขาดให้** — Docker, กลุ่ม `docker` ของ user, NVIDIA Container
> Toolkit, โมดูล `python3-venv` — โดยถามยืนยันก่อนทุกขั้นที่ใช้ `sudo` และแสดงคำสั่งจริงให้เห็นก่อนรัน
> ส่วนนี้เก็บไว้สำหรับคนที่อยากทำเอง เครื่องที่ติดตั้งแบบไม่มี prompt หรือเครื่อง air-gapped
>
> ข้อเดียวที่ `install.sh` **ไม่ทำให้** คือ NVIDIA driver (§1.2) เพราะต้อง reboot และบางเครื่องมี
> driver ใช้งานได้อยู่แล้วแต่ `ubuntu-drivers install` ชน dependency จนพัง

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

> `install.sh` ทำขั้นนี้ให้ได้ (ถามก่อน) — ทำเองก็ได้ตามนี้

```bash
docker --version || curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # ให้ user ปัจจุบันใช้ docker ได้โดยไม่ต้อง sudo
```

**สำคัญ**: หลัง `usermod` ต้อง **logout/login ใหม่** (หรือ `newgrp docker`) จึงจะมีผล

### 1.4 ติดตั้ง NVIDIA Container Toolkit (ให้ Docker เห็น GPU)

> `install.sh` ทำทั้ง 5 ขั้นนี้ให้ได้ (ถามก่อน) — ทำเองก็ได้ตามนี้

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

### 1.6 พื้นที่ดิสก์ + ไฟล์ถูกเก็บไว้ที่ไหน (อ่านก่อน deploy โมเดลใหญ่)

LMDS ตัวโปรแกรมเล็กมาก แต่**โมเดลกับ runtime image กินดิสก์เยอะ** — ตรวจก่อนเสมอ:

```bash
df -h ~            # ดูพื้นที่ว่างของ $HOME
docker system df   # ดูว่า image/container กินไปเท่าไหร่แล้ว
```

ตารางที่เก็บไฟล์ (ค่า default — ทุกอันเปลี่ยนได้ด้วย env ในตารางขวาสุด):

| อะไร | เก็บที่ | ขนาดคร่าว ๆ | เปลี่ยนที่ |
|---|---|---|---|
| ตัวโปรแกรม LMDS (venv) | `~/.local/share/lmds/` | ~150 MB | `LMDS_INSTALL_DIR` ตอนรัน `install.sh` |
| คำสั่ง `lmds` | `~/.local/bin/lmds` (symlink) | — | `LMDS_BIN_DIR` |
| config + API key | `~/.config/lmds/` | < 1 MB | — |
| bundle ที่ generate | `./bundles/<slug>/` (โฟลเดอร์ที่รันคำสั่ง) | 100–300 KB ต่อ bundle | `lmds deploy --output DIR` |
| **น้ำหนักโมเดล (vLLM / safetensors)** | `~/.cache/huggingface/hub/` | = ขนาดโมเดล (10–200 GB) | env `HF_HOME` |
| **น้ำหนักโมเดล (llama.cpp / GGUF)** | `~/models/<slug>/` | = ขนาดไฟล์ GGUF | env `MODEL_DIR` |
| **runtime image (vLLM)** | Docker (`/var/lib/docker`) | ~10–20 GB ต่อ image | ย้าย data-root ของ Docker |
| **runtime image (llama.cpp docker)** | Docker | ~3–5 GB | เหมือนกัน |
| llama.cpp source + build (โหมด native) | `~/src/llama.cpp/` | ~3 GB | env `LLAMA_CPP_DIR` |
| ทะเบียนเซิร์ฟเวอร์ที่รันอยู่ + log | `~/.lmds/run/<slug>/` | เล็ก | env `RUN_DIR` |

> **กฎง่าย ๆ**: เตรียมดิสก์ว่าง ≈ *(ขนาดโมเดล × 1.2) + 25 GB* ต่อเครื่อง
> โมเดลถูกดาวน์โหลด **ครั้งเดียว** แล้วใช้ซ้ำได้ทุก bundle ที่ชี้ไป revision เดียวกัน

ย้ายที่เก็บโมเดลไปดิสก์ลูกอื่น (เช่น NVMe ก้อนใหญ่) — ตั้ง env ก่อนรัน controller:

```bash
echo 'export HF_HOME=/data/hf-cache' >> ~/.bashrc     # vLLM / safetensors
echo 'export MODEL_DIR=/data/models' >> ~/.bashrc      # llama.cpp / GGUF (ต่อ bundle)
source ~/.bashrc
```

### 1.7 (ทางเลือก) ดึง runtime image ล่วงหน้า / เครื่องหลัง proxy

ครั้งแรกที่รัน `start` หรือ `download` Docker จะ pull image เอง (ใช้เวลานาน 5–20 นาทีตามเน็ต)
ถ้าอยากแยกขั้นนี้ออกมาทำตอนเน็ตว่าง:

```bash
docker pull vllm/vllm-openai:latest                  # สำหรับโมเดล safetensors (vLLM)
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda   # สำหรับโมเดล GGUF บน x86_64
```

เครื่องที่ออกเน็ตผ่าน proxy ต้องตั้งให้ **ทั้ง Docker daemon และ shell**:

```bash
# shell (ใช้ตอน lmds inspect/deploy และตอน curl โหลด GGUF)
export HTTPS_PROXY=http://proxy.example:3128 HTTP_PROXY=http://proxy.example:3128 NO_PROXY=localhost,127.0.0.1

# docker daemon (ใช้ตอน pull image)
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/proxy.conf >/dev/null <<'EOF'
[Service]
Environment="HTTPS_PROXY=http://proxy.example:3128"
Environment="HTTP_PROXY=http://proxy.example:3128"
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
```

องค์กรที่ใช้ mirror ของ Hugging Face ภายใน: `export HF_ENDPOINT=https://<mirror ภายใน>`

---

## ส่วนที่ 2 — ติดตั้ง LMDS

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject
./install.sh
```

สคริปต์จะ:
1. ตรวจ Python ≥ 3.10 · ถ้าขาดโมดูล `venv` **ถามติดตั้งให้** (`sudo apt install python3-venv`)
2. สร้าง virtualenv ที่ `~/.local/share/lmds/venv` (ไม่ยุ่งกับ Python ระบบ)
3. ติดตั้ง lmds ลง venv นั้น
4. symlink คำสั่ง `lmds` ไปที่ `~/.local/bin/lmds`
5. เติม `~/.local/bin` ลง PATH ใน `~/.bashrc` (หรือ `~/.zshrc`) ให้อัตโนมัติถ้ายังไม่มี
6. **ตรวจความพร้อมของเครื่อง แล้วติดตั้งของที่ขาดให้** — ดูตารางข้างล่าง
7. **ถามตั้งค่า LLM provider + API key** (ดู §3.2) — ข้ามได้
8. **ถามติดตั้ง tab completion** — กด TAB เติมชื่อคำสั่ง/bundle/target ให้ (ดู §5.1)
9. สรุปท้ายสุดว่า **ต้องพิมพ์อะไรต่อ** เป็นข้อ ๆ (รวมคำสั่งเปิดใช้งาน PATH / กลุ่ม docker ถ้าจำเป็น)

### ข้อ 6 ทำอะไรให้บ้าง

| ตรวจ | ถ้าขาด |
|---|---|
| NVIDIA driver (`nvidia-smi`) | ⚠️ แจ้งอย่างเดียว — **ไม่ติดตั้งให้** เพราะต้อง reboot (ทำเอง: `sudo ubuntu-drivers install`) |
| Docker | ถามติดตั้งด้วยสคริปต์ทางการ `https://get.docker.com` แล้ว `systemctl enable --now docker` |
| docker daemon ไม่ทำงาน | ถามสั่ง `sudo systemctl enable --now docker` |
| user ไม่อยู่ในกลุ่ม `docker` | ถามรัน `sudo usermod -aG docker $USER` แล้วบอกให้ `newgrp docker` ตอนท้าย |
| NVIDIA Container Toolkit | ถามติดตั้งครบทั้ง 5 ขั้น (keyring → apt source → install → `nvidia-ctk runtime configure` → restart docker) |
| Docker เห็น GPU จริงไหม | ทดสอบด้วย `docker run --rm --gpus all --entrypoint true nvidia/cuda:12.4.1-base-ubuntu22.04` (ครั้งแรกดึง image ~250 MB) |
| ดิสก์ `$HOME` | ⚠️ เตือนเมื่อเหลือ < 50 GB |

**ทุกขั้นที่ใช้ `sudo` จะถามยืนยันก่อนเสมอ และพิมพ์คำสั่งจริงให้เห็นก่อนรัน** — ตอบ `n` ได้ทุกข้อ
สคริปต์จะบอกคำสั่งให้ไปทำเองแทน · ของที่ขาดไม่ได้ทำให้ติดตั้ง LMDS ล้มเหลว (สร้าง bundle ได้ แต่ `start` จะยังไม่ผ่าน)

### env สำหรับติดตั้งแบบอัตโนมัติ / ในสคริปต์

| env | ผล |
|---|---|
| `LMDS_ASSUME_YES=1` | ตอบ Y ทุกคำถาม — ติดตั้งของที่ขาดให้เลยโดยไม่ถาม (ต้องรันได้ `sudo` โดยไม่ถามรหัส) |
| `LMDS_SKIP_PREREQ=1` | ข้ามการติดตั้ง Docker/toolkit ทั้งหมด — ตรวจแล้วรายงานอย่างเดียว |
| `LMDS_INSTALL_DIR` / `LMDS_BIN_DIR` | เปลี่ยนที่ติดตั้ง |

```bash
sudo -v && LMDS_ASSUME_YES=1 ./install.sh     # ติดตั้งรวดเดียวไม่ต้องนั่งตอบ
```

> เมื่อไม่ได้รันบน terminal จริง (CI, `curl | bash`, pipe) สคริปต์จะ**ไม่แตะเครื่องเลย** —
> ตรวจแล้วรายงานอย่างเดียว เว้นแต่ตั้ง `LMDS_ASSUME_YES=1`

หลังติดตั้งเสร็จ สคริปต์จะบอกเองที่บรรทัดสุดท้ายว่าต้อง `source ~/.bashrc` (และ `newgrp docker` ถ้าเพิ่งถูกเพิ่มเข้ากลุ่ม)
— PATH กับกลุ่มที่เพิ่งเติมยังไม่มีผลกับ shell เดิม จะเปิด terminal ใหม่แทนก็ได้

### ตรวจว่าติดตั้งสำเร็จ

```bash
lmds version
```

ควรได้:

```text
lmds 0.2.0
template standard: dgx-spark-controllers-v3.0.0
Local Model Deploy Studio — สร้างโดย neronain ⚡ fb.com/neronain.minidev
```

> ตอนรันบน terminal จริงจะมี banner ขึ้นก่อนด้วย — พิมพ์ออก **stderr** และเงียบเองเมื่อถูก pipe
> (`lmds inspect ... --json > out.json` ได้ JSON สะอาด) · ปิดถาวรได้ด้วย `export LMDS_NO_BANNER=1`

### ถ้าขึ้น "lmds: command not found"

`~/.local/bin` ยังไม่อยู่ใน PATH — แก้โดย:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### (ทางเลือก) เก็บ API key ใน OS keyring แทนไฟล์

`install.sh` พยายามติดตั้ง `keyring` ให้อยู่แล้ว (จะบอกในผลลัพธ์ว่าได้หรือไม่ได้) —
ถ้าไม่ได้ LMDS จะเก็บ key ลงไฟล์ `~/.config/lmds/credentials` (สิทธิ์ `0600`) แทน
ติดตั้งเองภายหลังได้ด้วย:

```bash
~/.local/share/lmds/venv/bin/pip install 'keyring>=24.0'
```

> เครื่อง server ที่ไม่มี desktop session มักไม่มี keyring backend ที่ใช้ได้ — LMDS จะ fallback ไปไฟล์ `0600` ให้เอง ไม่ต้องทำอะไร

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
RAM          ใช้ไป 12.4 / 128.0 GB (เหลือ 115.6 GB)
Disk ($HOME) ใช้ไป 210.5 / 1800.0 GB (เหลือ 1589.5 GB)
IP           192.168.1.50
Docker       ✅
NVIDIA Container Toolkit  ✅
Profile      rtx-multi-gpu
```

เช็ค 4 จุด: (1) เห็น GPU ครบทุกใบ (2) Docker ✅ ทั้งสองบรรทัด (3) Profile ตรงกับเครื่องจริง
(4) Disk เหลือพอตามสูตรใน §1.6 — ถ้าเหลือน้อยกว่า 50 GB ระบบจะขึ้น ⚠️ เตือนให้เอง

### 3.2 ตั้ง LLM provider (สมองของระบบ)

"สมอง" คือ LLM ที่ LMDS เรียกไปช่วย**วิเคราะห์โมเดลและเลือกค่าใน Deployment Plan** เท่านั้น —
มันไม่ได้เขียน Bash เอง และ**ไม่เกี่ยวกับโมเดลที่จะ deploy** เลย (คนละตัวกัน)

เลือกอย่างใดอย่างหนึ่ง:

```bash
# แบบ A: OpenAI (default model: gpt-4.1)
lmds config set-provider openai
lmds config set-key openai            # วาง API key แล้ว Enter (จอไม่แสดงตัวอักษร — ปกติ)

# แบบ B: Google Gemini (default model: gemini-2.5-pro)
lmds config set-provider gemini
lmds config set-key gemini

# แบบ C: Local AI ในองค์กรเป็นสมอง — vLLM / Ollama / endpoint OpenAI-compatible ใด ๆ (ดู 3.2.1)

# แบบ D: MiniMax (คลาวด์ — default model: MiniMax-M2)
lmds config set-provider minimax
lmds config set-key minimax

# แบบ E: ไม่มี LLM เลย — ไม่ต้องตั้งอะไร แล้วเติม --no-llm ทุกครั้งที่ deploy (rule-based mode)
```

เปลี่ยนโมเดลของ provider ได้ด้วย `--model` เช่น `lmds config set-provider openai --model gpt-4.1-mini`
ดูรายการ default ต่อ provider: `lmds config defaults`

> ⚠️ `anthropic` ตั้งค่าได้แต่ยัง**เรียกใช้จริงไม่ได้** (adapter อยู่ใน roadmap เฟส 2) — ตั้งแล้วจะ error ตอน deploy
> **หมายเหตุ**: ถ้า provider ที่ตั้งไว้ใช้ไม่ได้ตอน deploy (เช่น quota หมด / เครือข่ายล่ม)
> ระบบจะ**สลับเป็น rule-based mode ให้อัตโนมัติ**พร้อมแจ้งสาเหตุ — งานไม่สะดุด

#### 3.2.1 ใช้โมเดล local เป็นสมอง (vLLM / Ollama / OpenAI-compatible)

เหมาะกับองค์กรที่ไม่อยากส่งข้อมูลออกคลาวด์ หรือไม่มีงบ API — **ไม่ต้องมี API key**
(LMDS จะไม่ส่ง `Authorization` header ถ้าไม่มี key)

**ขั้นที่ 1 — ต้องมี endpoint ที่รันอยู่แล้ว**
LMDS ไม่ได้สตาร์ท LLM ตัวนี้ให้ ต้องมีอันใดอันหนึ่งอยู่ก่อน:

- โมเดลที่ **LMDS สร้าง bundle ให้เอง**แล้วรันอยู่ (ใช้ตัวเองเป็นสมองของตัวเองได้เลย — วิธีที่แนะนำ)
- vLLM ที่ทีมรันไว้อยู่แล้ว (port default `8000`)
- Ollama (`ollama serve` — port default `11434`, endpoint OpenAI-compatible อยู่ที่ `/v1`)
- LM Studio / llama.cpp server / TGI / gateway ใด ๆ ที่พูด `POST /v1/chat/completions`

**ขั้นที่ 2 — หา base URL และ "ชื่อโมเดล" ที่ถูกต้อง** (จุดที่พลาดกันบ่อยที่สุด)

```bash
curl -s http://10.10.10.1:8000/v1/models | python3 -m json.tool     # vLLM
curl -s http://10.10.10.1:11434/v1/models | python3 -m json.tool    # Ollama
```

ผลที่ได้จะมี `"id"` — **ค่านั้นคือค่าที่ต้องใส่ใน `--model` เป๊ะ ๆ**:

```json
{ "data": [ { "id": "qwen3-coder-30b-a3b-instruct-gguf", "object": "model" } ] }
```

**ขั้นที่ 3 — ตั้งค่า** (`--base-url` ต้องลงท้ายด้วย `/v1` และ `--model` ต้องระบุเสมอ)

```bash
# C1: vLLM (รวมถึง bundle ที่ LMDS สร้างเอง)
lmds config set-provider openai-compat \
  --base-url http://10.10.10.1:8000/v1 \
  --model qwen3-coder-30b-a3b-instruct-gguf

# C2: Ollama
lmds config set-provider openai-compat \
  --base-url http://10.10.10.1:11434/v1 \
  --model gpt-oss:20b

# C3: endpoint ที่บังคับ API key (gateway ภายใน / vLLM ที่ตั้ง API_KEY ไว้)
lmds config set-key openai-compat
```

**ขั้นที่ 4 — ทดสอบว่าสมองใช้ได้จริง** (deploy โมเดลจิ๋ว ไม่ต้องยืนยัน ไม่กิน GPU):

```bash
lmds plan Qwen/Qwen3-0.6B --target dgx-spark-single
```

ถ้าตารางที่ออกมาบรรทัด `Generator` เขียนว่า **ไม่ใช่** `rule-based` = สมอง local ทำงานแล้ว ✅
ถ้าขึ้นเตือนแล้วตกไป rule-based = ยังต่อไม่ติด ดูตารางล่าง

| อาการ | สาเหตุที่พบบ่อย | วิธีแก้ |
|---|---|---|
| `Connection refused` / timeout | ผิด IP/port, server bind แค่ `127.0.0.1`, firewall | เช็ก `curl` จาก**เครื่องที่รัน lmds** · ให้ server bind `0.0.0.0` (`./xxx-single.sh start --bind 0.0.0.0`) |
| HTTP 404 | ลืม `/v1` ท้าย base URL | `--base-url http://<ip>:8000/v1` |
| HTTP 400 `model not found` | ชื่อโมเดลไม่ตรง | เอา `id` จาก `/v1/models` มาใส่ตรง ๆ |
| HTTP 401 | endpoint บังคับ key | `lmds config set-key openai-compat` |
| HTTP 400 เรื่อง `response_format` | engine เก่าไม่รองรับ JSON mode | อัปเดต vLLM/Ollama หรือใช้ `--no-llm` |
| ตกเป็น rule-based เงียบ ๆ | โมเดลเล็กเกินไป ตอบ JSON ไม่ตรง schema | ใช้โมเดล instruct ขนาด ≥ 7B ที่ทำ tool/JSON ได้ดี |

> **ต้องใช้โมเดลสมองใหญ่แค่ไหน?** ที่ hardware-validated มาแล้วคือระดับ 26–30B instruct (เช่น gemma-4-26b, Qwen3-Coder-30B)
> เล็กกว่านั้นมักสร้าง Deployment Plan ที่ไม่ผ่าน schema แล้วระบบจะ retry จนตกไป rule-based

### 3.3 (ทางเลือก) HF token — เฉพาะเมื่อใช้โมเดล gated

โมเดลอย่าง Llama ต้องกดยอมรับเงื่อนไขบนเว็บ Hugging Face ก่อน แล้วเอา token มาใส่:

```bash
lmds config set-hf-token      # ข้ามได้ — โมเดลสาธารณะไม่ต้องใช้
```

token ตัวนี้ใช้ตอน `lmds inspect/deploy` (ฝั่ง LMDS) — ส่วนตอน **`download` บนเครื่องจริง**
controller อ่านจาก env เท่านั้น (ไม่ฝังไว้ในสคริปต์):

```bash
export HF_TOKEN=hf_xxxxxxxx
./<slug>-single.sh download
```

### 3.4 ตรวจการตั้งค่าทั้งหมด

```bash
lmds config show      # key ทุกตัวถูก mask — ปลอดภัยต่อการแคปหน้าจอส่งกัน
```

คอลัมน์ "จาก" บอกว่า secret มาจากไหน: `env` → `keyring` → `file` (ลำดับความสำคัญจากซ้ายไปขวา)
ตั้งผ่าน env ก็ได้ ไม่ต้องรัน `set-key`:

| secret | env ที่รองรับ |
|---|---|
| OpenAI | `LMDS_OPENAI_API_KEY`, `OPENAI_API_KEY` |
| Gemini | `LMDS_GEMINI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| MiniMax | `LMDS_MINIMAX_API_KEY`, `MINIMAX_API_KEY` |
| openai-compat | `LMDS_OPENAI_COMPAT_API_KEY` |
| Hugging Face | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` |

ทุกครั้งที่เรียก LLM ระบบเก็บ **audit log** (prompt/คำตอบ/การตัดสินใจ — redact secret แล้ว) ไว้ที่
`~/.config/lmds/sessions/` ไว้ตรวจย้อนหลังว่าแผนแต่ละอันมาจากอะไร · ย้ายที่เก็บ config ทั้งก้อนได้ด้วย env `LMDS_CONFIG_DIR`

---

## ส่วนที่ 4 — โมเดล local ถูกดึงมาและรันอย่างไร (vLLM / llama.cpp)

ส่วนนี้อธิบายสิ่งที่เกิดขึ้น**หลัง** `lmds deploy` สร้าง bundle เสร็จ — เพื่อให้รู้ว่าต้องเตรียมอะไร
และเวลาพังจะไปดูตรงไหน (วิธีใช้คำสั่งเต็ม ๆ อยู่ใน [USAGE.md](USAGE.md))

### 4.1 LMDS เลือก engine ให้อย่างไร

| ไฟล์โมเดลบน Hugging Face | engine ที่ได้ | รันแบบไหน |
|---|---|---|
| `.safetensors` (+ `model.safetensors.index.json`) | **vLLM** | Docker container |
| `.gguf` | **llama.cpp** | Docker (x86_64/RTX) หรือ **native build** (DGX Spark ARM64) |
| `.gguf` + target `dgx-spark-stacked` | — | ❌ ไม่รองรับ (stacked ใช้ vLLM เท่านั้น) |

เห็นค่าที่เลือกได้ก่อนสร้างไฟล์จริงด้วย `lmds plan <โมเดล>` (บรรทัด `Runtime`)

### 4.2 เส้นทาง vLLM (safetensors) — ทุกอย่างอยู่ใน Docker

ไม่ต้อง `pip install vllm` และไม่ต้องมี CUDA toolkit บน host — มีแค่ Docker + NVIDIA Container Toolkit พอ

```bash
cd bundles/<slug>
./<slug>-single.sh download      # ① docker pull image (ครั้งแรก ~10–20 GB) → ② โหลด weight ลง HF cache
./<slug>-single.sh verify-files  # ③ ตรวจว่าไฟล์ที่จำเป็นครบใน snapshot
./<slug>-single.sh start         # ④ เปิด container + รอ /health
./<slug>-single.sh test-text     # ⑤ ยิงคำถามทดสอบ 1 ครั้ง
```

สิ่งที่เกิดขึ้นจริงในแต่ละขั้น:

- **①–②** `download` รัน `snapshot_download()` **ข้างใน image ของ vLLM** โดย mount `$HF_HOME` เข้าไป
  → weight ลงที่ `~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<revision-sha>/`
  → revision ถูก **pin เป็น commit SHA** ตั้งแต่ตอน deploy — โหลดซ้ำได้ผลเหมือนเดิมเสมอ
  → repo gated ต้อง `export HF_TOKEN=...` ก่อน (สคริปต์ส่งต่อเข้า container ให้เอง)
- **④** `start` เปิด container ชื่อ `lmds-<slug>` แบบ `--network host --gpus all`
  แล้วรอ `GET /health` จนพร้อม (โมเดลใหญ่รอได้ถึง ~15 นาที — ปกติ)
- endpoint ที่ได้: `http://<ip เครื่อง>:8000/v1` (OpenAI-compatible)

ปรับได้ตอนรัน:

```bash
./<slug>-single.sh start --port 8001            # เปลี่ยน port
./<slug>-single.sh start --context 16384        # ลด context ถ้า memory ไม่พอ
./<slug>-single.sh start --bind 127.0.0.1       # ให้เข้าถึงได้เฉพาะในเครื่อง
API_KEY=secret123 ./<slug>-single.sh start      # บังคับ Bearer token (ตั้งเป็น VLLM_API_KEY ให้)
HF_HOME=/data/hf-cache ./<slug>-single.sh download   # เก็บ weight ไว้ดิสก์อื่น (ต้องใส่ตอน start ด้วย)
```

### 4.3 เส้นทาง llama.cpp (GGUF)

`download` ที่นี่ **ไม่ใช้ Docker** — ดึงไฟล์ `.gguf` ตรงจาก Hugging Face ด้วย `curl` (resume ได้)
ลงที่ `~/models/<slug>/` แล้วตรวจ **ขนาด exact + GGUF magic + SHA-256** ตอน `verify-files`

มี 2 โหมด (สคริปต์เลือกให้เองตอน generate, override ได้ด้วย `RUNTIME_MODE=docker|native`):

**โหมด docker** — เครื่อง x86_64/RTX ใช้ image ทางการ `ghcr.io/ggml-org/llama.cpp:server-cuda`

```bash
./<slug>-single.sh download && ./<slug>-single.sh verify-files && ./<slug>-single.sh start
```

**โหมด native** — DGX Spark (ARM64, SM121) ไม่มี image ทางการ จึง **build llama.cpp จาก source**
ต้องรัน `prepare-runtime` เพิ่ม **หนึ่งครั้ง** ก่อน start ครั้งแรก:

```bash
./<slug>-single.sh download
./<slug>-single.sh verify-files
./<slug>-single.sh prepare-runtime   # ← เฉพาะโหมด native
./<slug>-single.sh start
```

`prepare-runtime` จะ:
1. ติดตั้ง build deps ที่ขาด (`git`, `cmake`, `ninja-build`, `build-essential`, `curl`) ผ่าน `apt-get` — **ขอ sudo ครั้งเดียว**
   (ถ้าไม่พบ `nvcc` จะเตือน แต่ยัง build ต่อ — DGX OS มี CUDA Toolkit มาให้แล้ว)
2. `git clone` llama.cpp ไปที่ `~/src/llama.cpp` แล้ว build ด้วย CUDA (`-DGGML_CUDA=ON`) — ใช้เวลา ~10–30 นาที
3. **ล็อค commit ที่ build สำเร็จ**ไว้ใน `~/.lmds/run/<slug>/runtime.lock` → build ครั้งถัดไปได้ binary เดิมเป๊ะ

> เครื่องที่เคย `prepare-runtime` แล้ว bundle ตัวถัดไปยัง build ซ้ำ (คนละ lock) — แต่ `~/src/llama.cpp` ใช้ร่วมกัน จึงเร็วกว่าครั้งแรกมาก
> ต้องต่อเน็ตถึง github.com ตอน build — เครื่อง air-gapped ให้ก๊อป `~/src/llama.cpp` (ที่ build แล้ว) มาวางแทน

### 4.4 เอาโมเดลไปเครื่องที่ไม่มีเน็ต (air-gapped)

ทำบนเครื่องที่มีเน็ตก่อน แล้วค่อยขน 3 ก้อนนี้ไป:

```bash
# 1) bundle
scp bundles/<slug>.zip user@target:/home/user/

# 2) runtime image
docker save vllm/vllm-openai:latest | gzip > vllm.tar.gz    # ปลายทาง: gunzip -c vllm.tar.gz | docker load

# 3) น้ำหนักโมเดล
rsync -a ~/.cache/huggingface/hub/models--<org>--<model>/ user@target:~/.cache/huggingface/hub/models--<org>--<model>/
#    (GGUF: rsync -a ~/models/<slug>/ user@target:~/models/<slug>/)
```

ปลายทางข้าม `download` ไปได้เลย แล้วรัน `verify-files` → `start`

### 4.5 หลาย bundle ในเครื่องเดียว

รันพร้อมกันได้ ขอแค่ **คนละ port** (`--port 8001`, `8002`, …) และ memory รวมพอ
`lmds ps` / `lmds stop --all` คุมได้ทุกตัวจากที่เดียว — รายละเอียดใน [USAGE.md §4](USAGE.md)

---

## ส่วนที่ 5 — ตรวจความพร้อมทั้งระบบ (smoke test ~5 นาที)

รันตามลำดับนี้บนเครื่องจริงหลังติดตั้งเสร็จ ใช้โมเดล 0.6B (~0.4 GB) พิสูจน์ว่าทั้ง loop ทำงาน:

```bash
lmds version                                              # 1. โปรแกรมพร้อม
lmds hardware                                             # 2. เห็น GPU + Docker ✅ ทั้งสองบรรทัด
lmds config show                                          # 3. provider/key ตามที่ตั้งไว้
lmds plan Qwen/Qwen3-0.6B --target dgx-spark-single       # 4. สมองตอบได้ (Generator ≠ rule-based)
lmds deploy "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_K_M.gguf" --no-llm
cd bundles/qwen3-0-6b-gguf
./qwen3-0-6b-gguf-single.sh download && ./qwen3-0-6b-gguf-single.sh verify-files
./qwen3-0-6b-gguf-single.sh start                         # 5. โมเดลขึ้นจริง
./qwen3-0-6b-gguf-single.sh test-text                     # 6. ได้ JSON ที่มีคำตอบ = ผ่านทั้งระบบ 🎉
./qwen3-0-6b-gguf-single.sh stop
```

> DGX Spark: ขั้นที่ 5 ต้องแทรก `./qwen3-0-6b-gguf-single.sh prepare-runtime` ก่อน `start` (ดู §4.3)

ผ่านครบ = เครื่องพร้อม deploy โมเดลจริง · ไม่ผ่านข้อไหน กลับไปดูข้อนั้นในตารางของ [USAGE.md §7](USAGE.md)

---

## การอัปเดตเวอร์ชัน

```bash
cd AutoDeployDGXProject
git pull
./install.sh          # รันซ้ำได้เลย — config/key เดิมอยู่ครบ
```

> ⚠️ **`git pull` อย่างเดียวไม่พอ** — ติดตั้งแบบ copy เข้า venv (ไม่ใช่ editable) คำสั่ง `lmds` จึงยังเป็นโค้ดเก่าจนกว่าจะรัน `./install.sh` ซ้ำ
> bundle ที่ generate ไว้แล้ว**ไม่ถูกแก้ย้อนหลัง** — อยากได้ template ใหม่ต้อง `lmds deploy` โมเดลนั้นใหม่

## การถอนการติดตั้ง

```bash
rm -rf ~/.local/share/lmds ~/.local/bin/lmds   # ตัวโปรแกรม
rm -rf ~/.config/lmds                          # config + key (ถ้าต้องการ)
rm -rf ~/.lmds                                 # ทะเบียนเซิร์ฟเวอร์ + log
```

ของหนักที่ **ไม่ได้** ถูกลบไปด้วย — ลบเองถ้าต้องการคืนพื้นที่:

```bash
rm -rf ~/.cache/huggingface/hub    # weight ของ vLLM (หลายสิบ GB)
rm -rf ~/models                    # ไฟล์ GGUF
rm -rf ~/src/llama.cpp             # source + build ของโหมด native
docker rmi vllm/vllm-openai:latest ghcr.io/ggml-org/llama.cpp:server-cuda
```

> ถ้าเคยตั้ง autostart ไว้ ให้ `lmds disable <ชื่อ>` **ก่อน**ลบโปรแกรม ไม่งั้น systemd unit จะค้าง

---

### 5.1 Tab completion

`install.sh` ถามให้แล้ว — ถ้าข้ามไปหรืออยากติดตั้งทีหลัง:

```bash
lmds --install-completion
```

แล้ว**เปิด terminal ใหม่** (หรือ `source ~/.bashrc`) · รองรับ bash / zsh / fish

```text
lmds depl<TAB>                       → lmds deploy
lmds stop qwen<TAB>                  → เติมชื่อ bundle ให้
lmds deploy <url> --target dgx<TAB>  → dgx-spark-single / dgx-spark-stacked
```

---

ติดตั้งเสร็จแล้ว → ไปต่อที่ **[USAGE.md](USAGE.md)** เพื่อ deploy โมเดลตัวแรก
· ความปลอดภัย/ข้อมูลที่ออกนอกเครื่อง: **[SECURITY.md](../SECURITY.md)**
