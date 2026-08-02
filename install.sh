#!/usr/bin/env bash
# LMDS installer — Ubuntu 22.04/24.04 (ARM64/x86_64)
# ติดตั้งลง venv ที่ ~/.local/share/lmds แล้ว symlink คำสั่ง lmds ไป ~/.local/bin
set -Eeuo pipefail

INSTALL_DIR="${LMDS_INSTALL_DIR:-${HOME}/.local/share/lmds}"
BIN_DIR="${LMDS_BIN_DIR:-${HOME}/.local/bin}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "ต้องมี python3 (ติดตั้ง: sudo apt install python3 python3-venv)"

python3 - <<'EOF' || die "ต้องการ Python >= 3.10"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

python3 -m venv --help >/dev/null 2>&1 || die "ไม่มีโมดูล venv (ติดตั้ง: sudo apt install python3-venv)"

echo "ติดตั้ง LMDS ลง ${INSTALL_DIR} ..."
mkdir -p "$INSTALL_DIR" "$BIN_DIR"
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip
"${INSTALL_DIR}/venv/bin/pip" install --quiet "$REPO_DIR"

# keyring เป็น optional extra — ถ้าลงได้ key จะไปอยู่ใน keyring ของ OS แทนไฟล์ 0600
# เครื่อง server ที่ไม่มี desktop session มักไม่มี backend ที่ใช้ได้ → ข้ามไปเงียบ ๆ ไม่ให้ติดตั้งพัง
if "${INSTALL_DIR}/venv/bin/pip" install --quiet 'keyring>=24.0' 2>/dev/null; then
  echo "เก็บ key ผ่าน OS keyring ได้ (ถ้าเครื่องมี backend รองรับ)"
else
  echo "ไม่ได้ติดตั้ง keyring — key จะเก็บที่ ~/.config/lmds/credentials (สิทธิ์ 0600)"
fi

ln -sf "${INSTALL_DIR}/venv/bin/lmds" "${BIN_DIR}/lmds"

echo "ติดตั้งเสร็จ: $("${BIN_DIR}/lmds" version | head -1)"

# เติม BIN_DIR ลง PATH ให้อัตโนมัติ (เขียนลง shell rc ที่เหมาะกับ shell ปัจจุบัน)
ensure_path() {
  case ":${PATH}:" in
    *":${BIN_DIR}:"*) return 0 ;;  # อยู่ใน PATH แล้ว ไม่ต้องทำอะไร
  esac

  local export_line='export PATH="${HOME}/.local/bin:${PATH}"'
  # เลือกไฟล์ rc ตาม shell: zsh → ~/.zshrc, อื่น ๆ → ~/.bashrc (สร้างถ้ายังไม่มี)
  local rc_file="${HOME}/.bashrc"
  case "${SHELL:-}" in
    */zsh) rc_file="${HOME}/.zshrc" ;;
  esac

  if [ -f "$rc_file" ] && grep -qF "$export_line" "$rc_file"; then
    :  # มีบรรทัดนี้อยู่แล้ว
  else
    {
      echo ""
      echo "# เพิ่มโดย LMDS installer — ให้เรียกคำสั่ง lmds ได้"
      echo "$export_line"
    } >> "$rc_file"
    echo ""
    echo "✅ เพิ่ม ${BIN_DIR} ลง PATH ใน ${rc_file} แล้ว"
  fi

  # หมายเหตุ: installer รันเป็น subprocess — จะให้ shell ปัจจุบันเห็น PATH ใหม่
  # ต้องเปิด terminal ใหม่ หรือ source ไฟล์ rc เอง
  echo "   ใช้งานได้เลยด้วย:  source ${rc_file}   (หรือเปิด terminal ใหม่)"
}
ensure_path

LMDS="${BIN_DIR}/lmds"

# ── ตรวจ prerequisites ของ "เครื่องที่จะรันโมเดล" ──────────────────────────
# ตัว LMDS เองไม่ต้องใช้ GPU/Docker แต่ bundle ที่ generate ต้องใช้ — ตรวจให้ตั้งแต่ตอนนี้
# จะได้ไม่ไปเจอตอน start ซึ่งเสียเวลากว่ามาก
missing_prereq=0
echo ""
echo "── ตรวจความพร้อมของเครื่อง ──"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "✅ NVIDIA driver (nvidia-smi)"
else
  echo "⚠️  ไม่พบ nvidia-smi — ติดตั้ง driver ก่อนรันโมเดล: sudo ubuntu-drivers install (แล้ว reboot)"
  missing_prereq=1
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "✅ Docker (ใช้ได้โดยไม่ต้อง sudo)"
    if docker run --rm --gpus all --entrypoint true nvidia/cuda:12.4.1-base-ubuntu22.04 >/dev/null 2>&1; then
      echo "✅ NVIDIA Container Toolkit (Docker เห็น GPU)"
    else
      echo "⚠️  Docker ยังเห็น GPU ไม่ได้ — ติดตั้ง NVIDIA Container Toolkit (docs/INSTALL.md §1.4)"
      missing_prereq=1
    fi
  else
    echo "⚠️  มี docker แต่เรียกไม่ได้ — ต้อง: sudo usermod -aG docker \$USER แล้ว logout/login"
    missing_prereq=1
  fi
else
  echo "⚠️  ไม่พบ Docker — ติดตั้ง: curl -fsSL https://get.docker.com | sudo sh (docs/INSTALL.md §1.3)"
  missing_prereq=1
fi

free_gb="$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "$free_gb" ] && [ "$free_gb" -lt 50 ]; then
  echo "⚠️  ดิสก์ \$HOME เหลือ ${free_gb} GB — โมเดลขนาดกลางขึ้นไปอาจไม่พอ (ย้ายด้วย HF_HOME/MODEL_DIR ได้)"
else
  [ -n "$free_gb" ] && echo "✅ ดิสก์ \$HOME เหลือ ${free_gb} GB"
fi

# ── ตั้งค่า LLM provider (ครั้งเดียวต่อเครื่อง) ─────────────────────────────
# ผู้ใช้มักลืมขั้นนี้แล้วไปเจอ "ยังไม่ได้ตั้งค่า provider" ตอน deploy — ถามเลยตรงนี้
setup_provider() {
  if "$LMDS" config show 2>/dev/null | grep -q "^│ provider .*[a-z]" &&
     ! "$LMDS" config show 2>/dev/null | grep -q "ยังไม่ได้ตั้งค่า"; then
    echo "✅ ตั้ง LLM provider ไว้แล้ว (ดู: lmds config show)"
    return 0
  fi
  if [ ! -t 0 ]; then
    echo "ℹ️  ยังไม่ได้ตั้ง LLM provider — ตั้งภายหลังด้วย: lmds config set-provider <ชื่อ>"
    return 0
  fi

  echo ""
  echo "── ตั้งค่า LLM provider (สมองของระบบ) ──"
  echo "  1) OpenAI                     (ต้องมี API key)"
  echo "  2) Google Gemini              (ต้องมี API key)"
  echo "  3) MiniMax                    (ต้องมี API key)"
  echo "  4) Local AI / OpenAI-compatible — vLLM, Ollama ในองค์กร (ไม่ต้องมี key)"
  echo "  5) ข้ามไปก่อน                  (ใช้ --no-llm ทุกครั้งที่ deploy)"
  printf "เลือก [1-5] (Enter = 5): "
  read -r choice || choice=5

  case "${choice:-5}" in
    1) provider=openai ;;
    2) provider=gemini ;;
    3) provider=minimax ;;
    4)
      printf "base URL ของ endpoint (ต้องลงท้าย /v1 เช่น http://10.0.0.5:8000/v1): "
      read -r base_url
      printf "ชื่อโมเดล (ดูจาก: curl -s <base-url>/models): "
      read -r model_name
      if [ -z "$base_url" ] || [ -z "$model_name" ]; then
        echo "ℹ️  ข้อมูลไม่ครบ — ข้ามไปก่อน ตั้งภายหลังได้ด้วย lmds config set-provider openai-compat ..."
        return 0
      fi
      "$LMDS" config set-provider openai-compat --base-url "$base_url" --model "$model_name" || true
      echo "ทดสอบว่าต่อติดไหม: lmds plan Qwen/Qwen3-0.6B"
      return 0
      ;;
    *)
      echo "ℹ️  ข้ามการตั้ง provider — เติม --no-llm ทุกครั้งที่ deploy หรือตั้งภายหลังด้วย lmds config set-provider"
      return 0
      ;;
  esac

  "$LMDS" config set-provider "$provider" || return 0
  printf "ใส่ API key ของ %s ตอนนี้เลยไหม? [Y/n]: " "$provider"
  read -r want_key
  case "${want_key:-y}" in
    [Nn]*) echo "ℹ️  ตั้งภายหลังด้วย: lmds config set-key ${provider}" ;;
    *)     "$LMDS" config set-key "$provider" || echo "ℹ️  ตั้งภายหลังด้วย: lmds config set-key ${provider}" ;;
  esac
}
setup_provider

echo ""
if [ "$missing_prereq" -eq 1 ]; then
  echo "⚠️  ยังมีข้อที่ต้องแก้ก่อนรันโมเดลได้จริง (ดูรายการ ⚠️ ด้านบน + docs/INSTALL.md ส่วนที่ 1)"
  echo "    สร้าง bundle ได้เลยแม้ยังไม่ครบ — แต่ start จะยังไม่ผ่าน"
fi
echo "เริ่มต้นใช้งาน:"
echo "  lmds hardware                     # ตรวจเครื่องอีกครั้งแบบละเอียด"
echo "  lmds deploy https://huggingface.co/Qwen/Qwen3-0.6B"
