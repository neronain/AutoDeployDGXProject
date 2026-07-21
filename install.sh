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

ln -sf "${INSTALL_DIR}/venv/bin/lmds" "${BIN_DIR}/lmds"

echo "ติดตั้งเสร็จ: $("${BIN_DIR}/lmds" version | head -1)"

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    echo ""
    echo "หมายเหตุ: ${BIN_DIR} ยังไม่อยู่ใน PATH — เพิ่มบรรทัดนี้ใน ~/.bashrc:"
    echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    ;;
esac

echo ""
echo "เริ่มต้นใช้งาน:"
echo "  lmds hardware"
echo "  lmds config set-provider openai   # หรือ gemini / openai-compat"
echo "  lmds config set-key openai"
echo "  lmds deploy https://huggingface.co/Qwen/Qwen3-0.6B"
