"""ดึงสูตรจากรีโป controller ของทีมมาไว้ที่ hub — ต้นทางเดียว ไม่ต้องพิมพ์ซ้ำสองที่

ทีมเก็บ controller ที่รันผ่านจริงไว้ในรีโป Git อยู่แล้ว (ค่าตั้งต้น:
neronain/dgx-spark-all-controllers) แก้ที่นั่นแล้ว push · ฝั่ง hub สั่ง sync ทีเดียวก็ได้สูตรใหม่
ครบ ไม่มีสถานะ "แก้แล้วแต่ลืมอัปเดตอีกที่"

ดึงด้วย git เพราะ (1) ได้ commit มาอ้างเป็นที่มาแบบตรวจสอบได้ (2) รีโปส่วนตัวใช้ SSH key
เดิมของเครื่องได้เลย (3) ครั้งต่อไปเป็น fetch ไม่ใช่โหลดใหม่ทั้งก้อน
**อ่านไฟล์อย่างเดียว ไม่รัน controller** — ดู controllers.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from lmds.config.paths import config_dir, write_atomic

from . import load_catalog, synced_path
from .controllers import scan_directory

DEFAULT_REPO = "https://github.com/neronain/dgx-spark-all-controllers"
DEFAULT_REF = "main"


class SyncError(Exception):
    """ดึงไม่สำเร็จ — ข้อความต้องบอกว่าติดตรงไหน (เน็ต/สิทธิ์/ชื่อ ref)"""


def checkout_dir(repo: str) -> Path:
    """ที่เก็บสำเนารีโปบน hub — ชื่อโฟลเดอร์มาจากชื่อรีโป ไม่ใช่ URL เต็ม"""
    name = repo.rstrip("/").split("/")[-1].removesuffix(".git") or "controllers"
    return config_dir() / "controllers" / name


def _git(*args: str, cwd: Path | None = None, timeout: int = 300) -> str:
    if shutil.which("git") is None:
        raise SyncError("ไม่พบคำสั่ง git บนเครื่องนี้ — ติดตั้ง git ก่อน (apt install git)")
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SyncError(f"git {' '.join(args)} ไม่สำเร็จ: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise SyncError(f"git {args[0]} ไม่สำเร็จ: {detail[-1] if detail else 'ไม่มีข้อความ'}")
    return done.stdout.strip()


def fetch(repo: str = DEFAULT_REPO, ref: str = DEFAULT_REF) -> tuple[Path, str]:
    """โคลนหรืออัปเดตสำเนารีโป → (โฟลเดอร์, commit สั้น)"""
    target = checkout_dir(repo)
    if (target / ".git").is_dir():
        _git("remote", "set-url", "origin", repo, cwd=target)
        _git("fetch", "--depth", "1", "origin", ref, cwd=target)
        # reset ทิ้งของเดิมเสมอ — สำเนานี้เป็นแค่แคช ไม่ใช่ที่ทำงาน ใครไปแก้ในนี้ถือว่าหาย
        _git("reset", "--hard", "FETCH_HEAD", cwd=target)
    else:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        _git("clone", "--depth", "1", "--branch", ref, repo, str(target), timeout=600)
    return target, _git("rev-parse", "--short", "HEAD", cwd=target)


def sync(repo: str = DEFAULT_REPO, ref: str = DEFAULT_REF, now: str = "") -> dict:
    """ดึงรีโปแล้วเขียน recipes-synced.yaml — คืนสรุปว่าได้อะไรมาบ้าง/ข้ามอะไรไป"""
    target, commit = fetch(repo, ref)
    origin = f"{repo.rstrip('/').split('/')[-1].removesuffix('.git')}@{commit}"
    recipes, skipped = scan_directory(target, origin)
    if not recipes:
        raise SyncError(
            f"ดึง {repo} ({commit}) มาได้ แต่ไม่พบ controller ที่อ่านเป็นสูตรได้เลย — "
            f"ตรวจว่าเป็นรีโปที่ถูกต้องและสคริปต์มีตัวแปร MODEL_ID/HF_REPO ที่ระดับบนสุด"
        )
    write_atomic(synced_path(), yaml.safe_dump(
        {"version": 1, "source": {"repo": repo, "ref": ref, "commit": commit, "synced_at": now},
         "recipes": recipes},
        allow_unicode=True, sort_keys=False))
    load_catalog.cache_clear()          # ไม่งั้นกระบวนการที่รันค้างอยู่ยังเห็นสูตรชุดเก่า
    return {"repo": repo, "ref": ref, "commit": commit, "count": len(recipes),
            "skipped": skipped, "path": str(synced_path())}


def synced_source() -> dict:
    """ที่มาของสูตรชุดที่ดึงไว้ล่าสุด — {} ถ้ายังไม่เคย sync"""
    try:
        raw = yaml.safe_load(synced_path().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    source = raw.get("source")
    if not isinstance(source, dict):
        return {}
    return {**source, "count": len(raw.get("recipes") or [])}
