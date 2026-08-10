"""อัปเดตตัว hub เองจากหน้าเว็บ — `git pull` + ติดตั้ง + restart บริการ

ทำไมต้องมี: หน้าเว็บบอกได้อยู่แล้วว่า node ไหนรันโค้ดเก่า และกด update ให้ node ได้
แต่ตัว hub เองอัปเดตจากหน้าเว็บไม่ได้เลย — ขึ้นแค่ป้าย "มีอัปเดต" พร้อมคำสั่งให้ไปเปิด
terminal เอง ผลคือลำดับที่เกิดขึ้นจริงกลับหัว:

  hub ค้างอยู่ที่ commit เก่า → node ทุกเครื่อง "ตรงกับ hub" จึงไม่มีปุ่ม update ขึ้น
  → กด update ให้ node สักเครื่อง มันไปดึงของล่าสุดจาก GitHub → node ล้ำหน้า hub

ทั้งฟลีตจึงไม่เคยอยู่ที่ commit เดียวกัน และคำสั่งที่กดไปทำงานคนละรุ่นกับที่ตั้งใจ

หลักที่ยึด:
  - restart ต้องหลุดจาก process ของเว็บ (setsid) ไม่งั้น systemd ฆ่าตัวที่สั่ง restart
    ไปพร้อมกับตัวเอง แล้วคำสั่งไม่ทันได้ทำงาน
  - ดึงจาก remote ที่ repo ตั้งไว้เท่านั้น ไม่รับ URL จาก request — ไม่งั้นใครยิง endpoint นี้
    ได้ก็สั่งให้เครื่องติดตั้งโค้ดจากที่ไหนก็ได้
  - `git pull --ff-only` ล้วน ๆ · ไม่ merge ไม่ reset — เครื่องที่มีของแก้ค้างอยู่ต้องล้มแล้ว
    บอกให้คนไปดู ดีกว่าเงียบ ๆ กลืนงานที่ยังไม่ได้ commit ของใครสักคน
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import lmds


def source_root() -> Path | None:
    """git checkout ที่โค้ดที่กำลังรันอยู่มาจาก — None เมื่อไม่ได้ติดตั้งจาก checkout

    เดาจากตำแหน่งของโมดูลไม่ได้: ติดตั้งแบบปกติแล้วโค้ดอยู่ใน site-packages ของ venv
    ซึ่งไม่มี `.git` อยู่ใกล้ ๆ เลย — เครื่องจริงทุกเครื่องเป็นแบบนี้ `install.sh` จึงประทับ
    ที่อยู่ของ checkout ไว้ให้ตอนติดตั้ง
    """
    candidates: list[Path] = []
    try:                                  # ติดตั้งปกติ — ค่าที่ install.sh ประทับไว้
        from lmds._build import SOURCE

        if SOURCE:
            candidates.append(Path(SOURCE))
    except Exception:
        pass
    # รันจาก checkout ตรง ๆ (นักพัฒนา) และค่าเผื่อสำหรับเครื่องที่ติดตั้งไว้ก่อนจะมี SOURCE
    candidates.append(Path(lmds.__file__).resolve().parents[2])
    candidates.append(Path.home() / "AutoDeployDGXProject")
    for root in candidates:
        if (root / ".git").is_dir() and (root / "install.sh").is_file():
            return root
    return None


def dirty_files(root: Path) -> list[str]:
    """ไฟล์ที่แก้ค้างไว้ใน checkout — มีของพวกนี้อยู่ `git pull --ff-only` จะล้ม"""
    try:
        done = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    return [line[3:] for line in done.stdout.splitlines() if line.strip()]


# restart ถูกยิงแบบหลุดจาก process นี้ (setsid + หน่วงสั้น ๆ) — ตัวสคริปต์ต้องได้ตอบ
# HTTP กลับไปก่อน ไม่งั้นเบราว์เซอร์เห็นแค่ connection ขาดโดยไม่รู้ว่าสำเร็จหรือล้ม
_SCRIPT = """
set -e
echo "── ดึงโค้ดใหม่จาก {remote} ──"
git pull --ff-only
echo ""
echo "── ติดตั้ง ──"
LMDS_ASSUME_YES=1 LMDS_SKIP_PREREQ=1 ./install.sh
echo ""
"$HOME/.local/bin/lmds" version | head -1
{restart}
"""

_RESTART = """
echo ""
echo "── restart บริการหน้าเว็บ (หน้านี้จะหลุดสักครู่แล้วต่อกลับเอง) ──"
setsid bash -c 'sleep 2; systemctl --user restart {unit}' >/dev/null 2>&1 < /dev/null &
"""


def update_script(restart: bool = True) -> str:
    """สคริปต์ที่งานอัปเดตรัน — แยกออกมาให้เทสอ่านได้โดยไม่ต้องรันจริง"""
    from .daemon import UNIT_NAME

    return _SCRIPT.format(
        remote="origin",
        restart=_RESTART.format(unit=UNIT_NAME) if restart else "",
    )
