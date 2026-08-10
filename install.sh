#!/usr/bin/env bash
# LMDS installer — Ubuntu 22.04/24.04 (ARM64/x86_64)
# ติดตั้งลง venv ที่ ~/.local/share/lmds แล้ว symlink คำสั่ง lmds ไป ~/.local/bin
# แล้ว "ติดตั้งของที่ขาดให้" (Docker / NVIDIA Container Toolkit) — ถามก่อนทุกขั้นที่ใช้ sudo
#
# env ที่ปรับได้:
#   LMDS_INSTALL_DIR / LMDS_BIN_DIR  ที่ติดตั้ง (default ~/.local/share/lmds, ~/.local/bin)
#   LMDS_ASSUME_YES=1                ตอบ Y ทุกคำถาม — ติดตั้งของที่ขาดให้เลยโดยไม่ถาม
#   LMDS_SKIP_PREREQ=1               ข้ามส่วนติดตั้ง Docker/NVIDIA Container Toolkit (ตรวจอย่างเดียว)
set -Eeuo pipefail

INSTALL_DIR="${LMDS_INSTALL_DIR:-${HOME}/.local/share/lmds}"
BIN_DIR="${LMDS_BIN_DIR:-${HOME}/.local/bin}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_TEST_IMAGE="nvidia/cuda:12.4.1-base-ubuntu22.04"

# สถานะที่เก็บไว้สรุปตอนท้าย (ข้อความ "ต้องทำอะไรต่อ" ต้องอยู่บรรทัดสุดท้าย ไม่ใช่กลางจอ)
missing_prereq=0
need_newgrp=0        # เพิ่ง usermod -aG docker → กลุ่มยังไม่มีผลกับ shell นี้
need_source_rc=0     # เพิ่งเติม PATH ลง rc → shell นี้ยังไม่เห็นคำสั่ง lmds
rc_file="${HOME}/.bashrc"

die() { echo "ERROR: $*" >&2; exit 1; }

# ถามยืนยันก่อนทำอะไรที่ใช้ sudo — ไม่ใช่ terminal จริง (CI/pipe) = ไม่ทำ ไม่ค้าง
ask_yes() {
  local reply
  [ "${LMDS_ASSUME_YES:-0}" = "1" ] && return 0
  [ -t 0 ] || return 1
  printf "%s [Y/n]: " "$1"
  read -r reply || reply=n
  case "${reply:-y}" in [Nn]*) return 1 ;; *) return 0 ;; esac
}

# แสดงคำสั่ง sudo ทุกครั้งก่อนรัน — ผู้ใช้ต้องเห็นว่ากำลังจะรันอะไรด้วยสิทธิ์ root
sudo_run() {
  echo "   \$ sudo $*"
  sudo "$@"
}

# docker หลัง usermod: กลุ่มยังไม่มีผลใน shell นี้ ต้องยืมผ่าน sg เพื่อทดสอบต่อได้เลย
run_docker() {
  if [ "$need_newgrp" = "1" ] && command -v sg >/dev/null 2>&1; then
    sg docker -c "docker $*"
  else
    docker "$@"
  fi
}

# ── ส่วนที่ 1: Python ────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || die "ต้องมี python3 (ติดตั้ง: sudo apt install python3 python3-venv)"

python3 - <<'EOF' || die "ต้องการ Python >= 3.10"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "⚠️  ไม่มีโมดูล venv ของ Python"
  if ask_yes "   ติดตั้งให้เลยไหม? (sudo apt install python3-venv)"; then
    sudo_run apt-get update -qq
    sudo_run apt-get install -y python3-venv
  fi
  python3 -m venv --help >/dev/null 2>&1 || die "ยังไม่มีโมดูล venv (ติดตั้ง: sudo apt install python3-venv)"
fi

# ── ส่วนที่ 2: ติดตั้งตัว LMDS ────────────────────────────────────────────────
echo "ติดตั้ง LMDS ลง ${INSTALL_DIR} ..."
mkdir -p "$INSTALL_DIR" "$BIN_DIR"
# --clear เสมอ: ทางอัปเดตที่เอกสารบอกไว้คือรัน install.sh ซ้ำ ซึ่งเจอ venv เดิมอยู่แล้ว
# venv เดิมอาจถูกสร้างด้วย python คนละตัว (เช่นเปลี่ยนไปใช้ conda ทีหลัง) แล้ว ensurepip จะล้ม
# แบบอ่านไม่รู้เรื่อง — สร้างทับให้จบ ปลอดภัยเพราะโฟลเดอร์นี้เป็นของ LMDS ตัวเดียว
python3 -m venv --clear "${INSTALL_DIR}/venv" ||
  die "สร้าง venv ที่ ${INSTALL_DIR}/venv ไม่ได้ — ลบทิ้งแล้วลองใหม่: rm -rf '${INSTALL_DIR}/venv'"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip

# ประทับ commit ที่กำลังติดตั้งลงไปในแพ็กเกจ — ติดตั้งแบบปกติ (ไม่ใช่ editable) ทำให้โค้ดที่รัน
# อยู่ไม่ได้อยู่ใน git checkout อีกต่อไป จึงถามภายหลังไม่ได้ว่านี่คือโค้ดรุ่นไหน · เลข version
# ไม่ขยับทุกคอมมิต ฝั่ง hub เลยแยกไม่ออกว่า node ไหนตามหลัง (เจอจริงกับ msi-6)
#
# ประทับ *ที่อยู่ของ checkout* ไปด้วย — ปุ่มอัปเดตบนหน้าเว็บต้องรู้ว่าจะไป `git pull` ที่ไหน
# เดาจากตำแหน่งโค้ดที่รันอยู่ไม่ได้ เพราะมันอยู่ใน site-packages ของ venv ไปแล้ว
BUILD_COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || true)"
BUILD_SOURCE=""
[ -d "${REPO_DIR}/.git" ] && BUILD_SOURCE="$REPO_DIR"
printf '# สร้างโดย install.sh — commit และ checkout ที่ติดตั้งไว้ ณ ตอนนั้น\nCOMMIT = "%s"\nSOURCE = "%s"\n' \
  "$BUILD_COMMIT" "$BUILD_SOURCE" > "${REPO_DIR}/src/lmds/_build.py"

"${INSTALL_DIR}/venv/bin/pip" install --quiet "$REPO_DIR"

# keyring เป็น optional extra — ถ้าลงได้ key จะไปอยู่ใน keyring ของ OS แทนไฟล์ 0600
# เครื่อง server ที่ไม่มี desktop session มักไม่มี backend ที่ใช้ได้ → ข้ามไปเงียบ ๆ ไม่ให้ติดตั้งพัง
if "${INSTALL_DIR}/venv/bin/pip" install --quiet 'keyring>=24.0' 2>/dev/null; then
  echo "เก็บ key ผ่าน OS keyring ได้ (ถ้าเครื่องมี backend รองรับ)"
else
  echo "ไม่ได้ติดตั้ง keyring — key จะเก็บที่ ~/.config/lmds/credentials (สิทธิ์ 0600)"
fi

# ส่วนเว็บ (lmds web) เป็น optional extra — ลงให้ถ้าลงได้ ไม่ได้ก็ไม่กระทบ CLI
if "${INSTALL_DIR}/venv/bin/pip" install --quiet 'fastapi>=0.110' 'uvicorn>=0.27' 2>/dev/null; then
  echo "ติดตั้งหน้าเว็บด้วย — เปิดด้วย: lmds web"
else
  echo "ข้ามหน้าเว็บ (ติดตั้ง fastapi/uvicorn ไม่สำเร็จ) — CLI ใช้ได้ตามปกติ"
fi

ln -sf "${INSTALL_DIR}/venv/bin/lmds" "${BIN_DIR}/lmds"
LMDS="${BIN_DIR}/lmds"

echo "ติดตั้งเสร็จ: $(LMDS_NO_BANNER=1 "$LMDS" version | head -1)"

# เติม BIN_DIR ลง PATH ให้อัตโนมัติ (เขียนลง shell rc ที่เหมาะกับ shell ปัจจุบัน)
# หมายเหตุ: installer รันเป็น subprocess — shell ปัจจุบันจะเห็น PATH ใหม่ต่อเมื่อ source เอง
# จึงไม่พิมพ์เตือนตรงนี้ แต่ยกไปไว้บรรทัดสุดท้ายของสคริปต์ (ผู้ใช้เห็นแน่)
ensure_path() {
  case "${SHELL:-}" in
    */zsh) rc_file="${HOME}/.zshrc" ;;
  esac
  case ":${PATH}:" in
    *":${BIN_DIR}:"*) return 0 ;;  # อยู่ใน PATH ของ shell นี้แล้ว
  esac

  local export_line='export PATH="${HOME}/.local/bin:${PATH}"'
  if ! { [ -f "$rc_file" ] && grep -qF "$export_line" "$rc_file"; }; then
    {
      echo ""
      echo "# เพิ่มโดย LMDS installer — ให้เรียกคำสั่ง lmds ได้"
      echo "$export_line"
    } >> "$rc_file"
    echo "✅ เพิ่ม ${BIN_DIR} ลง PATH ใน ${rc_file} แล้ว"
  fi
  need_source_rc=1
}
ensure_path

# ── ส่วนที่ 3: ความพร้อมของ "เครื่องที่จะรันโมเดล" ────────────────────────────
# ตัว LMDS เองไม่ต้องใช้ GPU/Docker แต่ bundle ที่ generate ต้องใช้ — จัดการให้ตั้งแต่ตอนนี้
# จะได้ไม่ไปเจอตอน start ซึ่งเสียเวลากว่ามาก
echo ""
echo "── ตรวจความพร้อมของเครื่อง ──"

can_install=1
if [ "${LMDS_SKIP_PREREQ:-0}" = "1" ]; then
  can_install=0
elif [ ! -t 0 ] && [ "${LMDS_ASSUME_YES:-0}" != "1" ]; then
  can_install=0   # ไม่ใช่ terminal จริง (CI/pipe) — ตรวจอย่างเดียว ไม่แตะเครื่อง
fi

# 3.1 NVIDIA driver — ไม่ติดตั้งให้อัตโนมัติโดยตั้งใจ: ต้อง reboot และเคยพัง dependency
#     บนเครื่องที่มี driver ใช้งานได้อยู่แล้ว (เคสจริง: ubuntu-drivers install ชน 580-open)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "✅ NVIDIA driver ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
else
  echo "⚠️  ไม่พบ NVIDIA driver ที่ใช้งานได้"
  echo "    ขั้นนี้ต้องทำเองเพราะต้อง reboot: sudo ubuntu-drivers install && sudo reboot"
  missing_prereq=1
fi

# 3.2 Docker
install_docker() {
  echo "   จะติดตั้ง Docker ด้วยสคริปต์ทางการของ Docker (https://get.docker.com)"
  echo "   \$ curl -fsSL https://get.docker.com | sudo sh"
  curl -fsSL https://get.docker.com | sudo sh
  sudo_run systemctl enable --now docker
}

add_docker_group() {
  sudo_run usermod -aG docker "$USER"
  need_newgrp=1
  echo "   เพิ่ม ${USER} เข้ากลุ่ม docker แล้ว (กลุ่มมีผลกับ shell ใหม่)"
}

if ! command -v docker >/dev/null 2>&1; then
  echo "⚠️  ไม่พบ Docker"
  if [ "$can_install" = "1" ] && ask_yes "   ติดตั้ง Docker ให้เลยไหม?"; then
    install_docker
    add_docker_group
  else
    echo "    ติดตั้งเอง: curl -fsSL https://get.docker.com | sudo sh   (docs/INSTALL.md §1.3)"
    missing_prereq=1
  fi
fi

# แยกสองสาเหตุที่ `docker info` ล้มให้ขาดจากกัน: daemon ไม่ขึ้น กับ user ไม่มีสิทธิ์
# ตรวจด้วย systemctl/id ซึ่งไม่ต้องใช้ sudo — จะได้ไม่เด้งขอรหัสผ่านก่อนผู้ใช้อนุญาต
if command -v docker >/dev/null 2>&1 && ! run_docker info >/dev/null 2>&1; then
  if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active --quiet docker 2>/dev/null; then
    echo "⚠️  Docker daemon ไม่ทำงาน"
    if [ "$can_install" = "1" ] && ask_yes "   สั่งเปิด docker service ให้เลยไหม?"; then
      sudo_run systemctl enable --now docker
    else
      echo "    ทำเอง: sudo systemctl enable --now docker"
      missing_prereq=1
    fi
  fi

  if ! run_docker info >/dev/null 2>&1; then
    if id -nG "$USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
      # อยู่ในกลุ่มแล้วแต่ shell นี้เกิดก่อนการเพิ่มกลุ่ม — ยืม sg ทำงานต่อได้ทันที
      echo "⚠️  อยู่ในกลุ่ม docker แล้ว แต่ยังไม่มีผลกับ shell นี้"
      need_newgrp=1
    else
      echo "⚠️  มี Docker แต่ user ปัจจุบันเรียกไม่ได้ (ยังไม่อยู่ในกลุ่ม docker)"
      if [ "$can_install" = "1" ] && ask_yes "   เพิ่ม ${USER} เข้ากลุ่ม docker ให้เลยไหม?"; then
        add_docker_group
      else
        echo "    ทำเอง: sudo usermod -aG docker \$USER แล้ว logout/login (หรือ newgrp docker)"
        missing_prereq=1
      fi
    fi
  fi

  # ยังใช้ไม่ได้จริงหลังพยายามครบทุกทาง — อย่าให้ผ่านไปเงียบ ๆ
  if ! run_docker info >/dev/null 2>&1; then
    missing_prereq=1
  fi
fi

if command -v docker >/dev/null 2>&1 && run_docker info >/dev/null 2>&1; then
  echo "✅ Docker ($(run_docker version --format '{{.Server.Version}}' 2>/dev/null || echo ใช้ได้))"
fi

# 3.3 NVIDIA Container Toolkit — ขั้นที่ผู้ใช้ต้องพิมพ์เองมากที่สุด (5 คำสั่ง) จึงทำให้ครบในที่เดียว
install_nv_toolkit() {
  command -v curl >/dev/null 2>&1 || { sudo_run apt-get update -qq; sudo_run apt-get install -y curl; }
  command -v gpg  >/dev/null 2>&1 || sudo_run apt-get install -y gnupg

  echo "   \$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  echo "   \$ (เพิ่ม apt source ของ nvidia-container-toolkit)"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  sudo_run apt-get update -qq
  sudo_run apt-get install -y nvidia-container-toolkit
  sudo_run nvidia-ctk runtime configure --runtime=docker
  sudo_run systemctl restart docker
}

gpu_in_docker() {
  run_docker run --rm --gpus all --entrypoint true "$CUDA_TEST_IMAGE" >/dev/null 2>&1
}

if command -v docker >/dev/null 2>&1 && run_docker info >/dev/null 2>&1; then
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "⚠️  ไม่พบ NVIDIA Container Toolkit — Docker จะยังเห็น GPU ไม่ได้"
    if [ "$can_install" = "1" ] && ask_yes "   ติดตั้ง NVIDIA Container Toolkit ให้เลยไหม? (5 ขั้น, ใช้ sudo)"; then
      install_nv_toolkit
    else
      echo "    ติดตั้งเอง: docs/INSTALL.md §1.4"
      missing_prereq=1
    fi
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "   ทดสอบว่า Docker เห็น GPU จริง (ครั้งแรกจะดึง image ทดสอบ ~250 MB) ..."
    if gpu_in_docker; then
      echo "✅ NVIDIA Container Toolkit (Docker เห็น GPU)"
    else
      echo "⚠️  ติดตั้ง toolkit แล้วแต่ Docker ยังเห็น GPU ไม่ได้"
      echo "    ตรวจ: docker run --rm --gpus all ${CUDA_TEST_IMAGE} nvidia-smi   (docs/INSTALL.md §1.5)"
      missing_prereq=1
    fi
  fi
fi

# 3.4 ดิสก์
# ต้องกัน pipefail/set -e ให้ครบ: df ที่ไม่รองรับ --output (นอก GNU coreutils) เคยทำให้
# สคริปต์จบเงียบตรงนี้ทั้งที่ยังเหลืออีกครึ่ง
free_gb="$( { df -BG --output=avail "$HOME" 2>/dev/null || true; } | tail -1 | tr -dc '0-9')"
if [ -z "$free_gb" ]; then
  echo "ℹ️  ตรวจพื้นที่ดิสก์อัตโนมัติไม่ได้บนระบบนี้ — ดูเอง: df -h ~"
elif [ "$free_gb" -lt 50 ]; then
  echo "⚠️  ดิสก์ \$HOME เหลือ ${free_gb} GB — โมเดลขนาดกลางขึ้นไปอาจไม่พอ (ย้ายด้วย HF_HOME/MODEL_DIR ได้)"
else
  echo "✅ ดิสก์ \$HOME เหลือ ${free_gb} GB"
fi

# ── ส่วนที่ 4: ตั้งค่า LLM provider (ครั้งเดียวต่อเครื่อง) ────────────────────
# ผู้ใช้มักลืมขั้นนี้แล้วไปเจอ "ยังไม่ได้ตั้งค่า provider" ตอน deploy — ถามเลยตรงนี้
setup_provider() {
  if LMDS_NO_BANNER=1 "$LMDS" config show 2>/dev/null | grep -q "^│ provider .*[a-z]" &&
     ! LMDS_NO_BANNER=1 "$LMDS" config show 2>/dev/null | grep -q "ยังไม่ได้ตั้งค่า"; then
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
      LMDS_NO_BANNER=1 "$LMDS" config set-provider openai-compat \
        --base-url "$base_url" --model "$model_name" || true
      # endpoint ในองค์กรส่วนใหญ่ไม่ต้องใช้ key — ไม่ต้องไปแนะให้ set-key ให้สับสน
      echo "   endpoint แบบนี้ปกติไม่ต้องใช้ API key · ถ้า endpoint ของคุณบังคับ ให้ตั้งด้วย:"
      echo "     lmds config set-key openai-compat"
      echo "   ทดสอบว่าต่อติดไหม: lmds plan Qwen/Qwen3-0.6B"
      return 0
      ;;
    *)
      echo "ℹ️  ข้ามการตั้ง provider — เติม --no-llm ทุกครั้งที่ deploy หรือตั้งภายหลังด้วย lmds config set-provider"
      return 0
      ;;
  esac

  LMDS_NO_BANNER=1 "$LMDS" config set-provider "$provider" || return 0
  printf "ใส่ API key ของ %s ตอนนี้เลยไหม? [Y/n]: " "$provider"
  read -r want_key
  case "${want_key:-y}" in
    [Nn]*) echo "ℹ️  ตั้งภายหลังด้วย: lmds config set-key ${provider}" ;;
    *)     LMDS_NO_BANNER=1 "$LMDS" config set-key "$provider" \
             || echo "ℹ️  ตั้งภายหลังด้วย: lmds config set-key ${provider}" ;;
  esac
}
setup_provider

# ── ส่วนที่ 5: Tab completion ───────────────────────────────────────────────
# เติมชื่อคำสั่ง (depl<TAB> -> deploy) และชื่อ slug/target ให้ - ชื่อ bundle ยาวและพิมพ์ผิดง่าย
setup_completion() {
  case "${SHELL:-}" in
    */bash) comp_rc="${HOME}/.bashrc" ;;
    */zsh)  comp_rc="${HOME}/.zshrc" ;;
    */fish) comp_rc="fish" ;;
    *)      echo "ข้าม tab completion (ไม่รู้จัก shell: ${SHELL:-ไม่ระบุ})"; return 0 ;;
  esac

  # typer เขียน marker _LMDS_COMPLETE ไว้ใน rc ของ shell ตอนติดตั้งสำเร็จ
  if [ "$comp_rc" != "fish" ] && [ -f "$comp_rc" ] && grep -q "_LMDS_COMPLETE" "$comp_rc" 2>/dev/null; then
    echo "OK tab completion ติดตั้งไว้แล้ว"
    return 0
  fi

  if [ ! -t 0 ]; then
    echo "ติดตั้ง tab completion ภายหลังได้ด้วย: lmds --install-completion"
    return 0
  fi

  printf "ติดตั้ง tab completion ของคำสั่ง lmds ไหม? (เติมชื่อคำสั่ง/bundle ด้วย TAB) [Y/n]: "
  read -r want_comp
  case "${want_comp:-y}" in
    [Nn]*) echo "ติดตั้งภายหลังได้ด้วย: lmds --install-completion" ;;
    *)
      if LMDS_NO_BANNER=1 "$LMDS" --install-completion >/dev/null 2>&1; then
        echo "OK ติดตั้ง tab completion แล้ว - เปิด terminal ใหม่ (หรือ source ${comp_rc}) จึงจะใช้ได้"
        need_source_rc=1
      else
        echo "ติดตั้ง tab completion ไม่สำเร็จ - ลองรันเอง: lmds --install-completion"
      fi
      ;;
  esac
}
setup_completion

# ── ส่วนที่ 6: สรุป + สิ่งที่ต้องทำต่อ ────────────────────────────────────────
# ทุกอย่างที่ผู้ใช้ต้องลงมือทำเองอยู่ตรงนี้ที่เดียว (บรรทัดสุดท้ายของสคริปต์)
echo ""
if [ "$missing_prereq" -eq 1 ]; then
  echo "⚠️  ยังมีข้อที่ต้องแก้ก่อนรันโมเดลได้จริง (ดูรายการ ⚠️ ด้านบน + docs/INSTALL.md ส่วนที่ 1)"
  echo "    สร้าง bundle ได้เลยแม้ยังไม่ครบ — แต่ start จะยังไม่ผ่าน"
  echo ""
fi

echo "── เริ่มต้นใช้งาน ──"
step=1
if [ "$need_source_rc" = "1" ] || [ "$need_newgrp" = "1" ]; then
  activate="source ${rc_file}"
  [ "$need_newgrp" = "1" ] && activate="${activate} && newgrp docker"
  echo "  ${step}. เปิดใช้งานในหน้าต่างนี้ (หรือเปิด terminal ใหม่แทนก็ได้):"
  echo "       ${activate}"
  step=$((step + 1))
fi
echo "  ${step}. ตรวจเครื่องแบบละเอียด:  lmds hardware"
step=$((step + 1))
echo "  ${step}. ลอง deploy ตัวเล็กก่อน:  lmds deploy Qwen/Qwen3-0.6B"
