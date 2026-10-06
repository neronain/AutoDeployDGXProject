"""ดึงสูตรจากรีโป controller ของทีมมาไว้ที่ hub — ต้นทางเดียว ไม่ต้องพิมพ์ซ้ำสองที่

ทีมเก็บ controller ที่รันผ่านจริงไว้ในรีโป Git อยู่แล้ว (ค่าตั้งต้น:
neronain/dgx-spark-all-controllers) แก้ที่นั่นแล้ว push · ฝั่ง hub สั่ง sync ทีเดียวก็ได้สูตรใหม่
ครบ ไม่มีสถานะ "แก้แล้วแต่ลืมอัปเดตอีกที่"

ดึงด้วย git เพราะ (1) ได้ commit มาอ้างเป็นที่มาแบบตรวจสอบได้ (2) รีโปส่วนตัวใช้ SSH key
เดิมของเครื่องได้เลย (3) ครั้งต่อไปเป็น fetch ไม่ใช่โหลดใหม่ทั้งก้อน
**อ่านไฟล์อย่างเดียว ไม่รัน controller** — ดู controllers.py
"""

from __future__ import annotations

import hashlib
import re
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


# ── ค่าที่ไปถึง git ต้องผ่านด่านนี้ทุกตัว — ไม่ว่าใครเรียก ──
#
# audit 2026-10 (หน้าเว็บ): `POST /api/recipes/sync` ส่ง repo/ref จาก body มาที่นี่ตรง ๆ แล้วพังสองทาง
#   · ref ที่หน้าตาเป็น option (`--upload-pack=…`) ถูก git อ่านเป็น option จริงเพราะไม่มี `--` คั่น
#     → รันคำสั่งบน hub ในฐานะ user ของคอนโซล ทั้งที่ HTTP ตอบ 400 เฉย ๆ
#   · repo ที่ลงท้ายด้วย `/..` ทำให้ checkout_dir ชี้กลับไปที่ config dir เอง แล้ว rmtree ก่อน clone
#     ลบทะเบียนเครื่อง · กุญแจ SSH · credentials · web-token ทิ้งทั้งโฟลเดอร์ด้วย POST เดียว
# ปากทาง (endpoint) เลิกรับค่าพวกนี้แล้ว แต่ sink ต้องยืนเองได้ — CLI, publish และ caller ในอนาคต
# ผ่านทางเดียวกัน: รูปแบบ URL แคบ · ref ชุดอักษรแคบและห้ามขึ้นต้นด้วย `-` · `--` ก่อน positional
# ทุกคำสั่ง · ชื่อโฟลเดอร์มาจาก hash ไม่ใช่ข้อความของผู้เรียก · rmtree เฉพาะลูกตรงของแคช
_HOST = r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?"
_USER = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
_PATH = r"[A-Za-z0-9._~+-]+(?:/[A-Za-z0-9._~+-]+)*/?"
_REPO_FORMS = (
    re.compile(rf"https?://{_HOST}(?::\d{{1,5}})?/{_PATH}"),            # https://host/owner/repo(.git)
    re.compile(rf"ssh://(?:{_USER}@)?{_HOST}(?::\d{{1,5}})?/{_PATH}"),  # ssh://git@host/owner/repo
    re.compile(rf"{_USER}@{_HOST}:{_PATH}"),                            # git@host:owner/repo.git
    re.compile(rf"file:///{_PATH}"),                                    # mirror ในเครื่อง (air-gapped)
)
_REF_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_CHECKOUT_NAME = re.compile(r"repo-[0-9a-f]{16}")


def validate_repo(repo: str) -> str:
    """รีโปที่ยอมส่งให้ git — URL รูปแบบที่รู้จักเท่านั้น · ไม่ผ่าน = SyncError พร้อมเหตุผล

    ไม่รับ: ค่าที่ขึ้นต้นด้วย `-` (git อ่านเป็น option) · transport helper (`ext::…`) ·
    path ที่มี `..` · ช่องว่าง/ตัวควบคุม · user:password ใน URL (ความลับจะไปนอนใน
    recipes-synced.yaml — ใช้ SSH key หรือ credential helper ของ git แทน)
    """
    text = repo if isinstance(repo, str) else ""
    ok = bool(text) and any(form.fullmatch(text) for form in _REPO_FORMS)
    if ok:
        parts = [part for part in re.split(r"[/:@]", text) if part]
        ok = ".." not in parts and not any(part.startswith("-") for part in parts)
    if not ok:
        shown = text if len(text) <= 120 else text[:117] + "…"
        raise SyncError(
            f"รีโปสูตรไม่ถูกรูปแบบ: {shown!r} — รับเฉพาะ https://host/owner/repo · "
            "ssh://git@host/owner/repo · git@host:owner/repo.git · file:///path/ของ/mirror "
            "(ไม่มี `..` · ไม่มีรหัสผ่านใน URL · ห้ามขึ้นต้นด้วย `-`)"
        )
    return text


def validate_ref(ref: str) -> str:
    """branch/tag ที่ยอมส่งให้ git — ชุดอักษรแคบ และห้ามขึ้นต้นด้วย `-` (git อ่านเป็น option)"""
    text = ref if isinstance(ref, str) else ""
    bad = (not _REF_OK.fullmatch(text) or ".." in text or "//" in text
           or text.endswith(("/", ".", ".lock")))
    if bad:
        shown = text if len(text) <= 80 else text[:77] + "…"
        raise SyncError(
            f"ชื่อ branch/tag ไม่ถูกรูปแบบ: {shown!r} — ใช้ได้เฉพาะ a-z A-Z 0-9 . _ / - "
            "และต้องขึ้นต้นด้วยตัวอักษรหรือตัวเลข"
        )
    return text


def cache_root() -> Path:
    """โฟลเดอร์แคชของสำเนารีโป — ที่เดียวใต้ config dir ที่โมดูลนี้มีสิทธิ์ลบของข้างใน"""
    return config_dir() / "controllers"


def checkout_dir(repo: str) -> Path:
    """ที่เก็บสำเนารีโปบน hub — ชื่อโฟลเดอร์มาจาก **hash** ของ URL ไม่ใช่ข้อความของผู้เรียก

    เดิมใช้ส่วนท้ายของ URL เป็นชื่อโฟลเดอร์ · `…/nope/..` จึงได้ `controllers/..` = config dir เอง
    และ `…/published-local` ชนกับ local store ของ publish · hash ออกนอกแคชไม่ได้และไม่ชนกัน
    (สำเนาเก่าที่ตั้งชื่อตามรีโปถูกทิ้งไว้เฉย ๆ — เป็นแคช ลบเองได้ ครั้งถัดไป clone ใหม่ที่ชื่อ hash)
    """
    key = validate_repo(repo).rstrip("/").removesuffix(".git")
    return cache_root() / f"repo-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def repo_label(repo: str) -> str:
    """ชื่อสั้นของรีโปไว้แสดงเป็นที่มาของสูตร — ข้อความล้วน ไม่ถูกใช้เป็น path"""
    tail = repo.rstrip("/").split("/")[-1].split(":")[-1].removesuffix(".git")
    return tail if re.fullmatch(r"[A-Za-z0-9._-]+", tail or "") and tail.strip(".") else "controllers"


def _discard_checkout(target: Path) -> None:
    """ลบสำเนาเก่าก่อน clone ใหม่ — เฉพาะเมื่อพิสูจน์ได้ว่าเป็นลูกตรงของแคช ไม่ใช่อย่างอื่น

    ด่านสุดท้ายของ rmtree: ต่อให้ชั้นบนพลาดจนส่ง path แปลก ๆ มา ที่นี่ต้องไม่ลบ config dir,
    ไม่ตาม symlink ออกนอกแคช และไม่ลบโฟลเดอร์ที่ไม่ได้ตั้งชื่อเอง (published-local ของ publish)
    """
    root = cache_root().resolve()
    config = config_dir().resolve()
    resolved = target.resolve()
    safe = (not target.is_symlink() and resolved.parent == root and resolved not in (config, root)
            and bool(_CHECKOUT_NAME.fullmatch(target.name)))
    if not safe:
        raise SyncError(f"ปฏิเสธการลบ {target} — ไม่ใช่สำเนารีโปใต้ {root}")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


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
    repo, ref = validate_repo(repo), validate_ref(ref)
    target = checkout_dir(repo)
    # `--` ก่อน positional ทุกคำสั่งที่รับค่าจากผู้เรียก — ค่าที่ผ่านด่านข้างบนขึ้นต้นด้วย `-` ไม่ได้อยู่แล้ว
    # แต่ด่านตรวจกับตัวคั่นเป็นคนละชั้นกัน ชั้นใดชั้นหนึ่งหลุด อีกชั้นยังกันไว้
    if (target / ".git").is_dir():
        _git("remote", "set-url", "--", "origin", repo, cwd=target)
        _git("fetch", "--depth", "1", "--", "origin", ref, cwd=target)
        # reset ทิ้งของเดิมเสมอ — สำเนานี้เป็นแค่แคช ไม่ใช่ที่ทำงาน ใครไปแก้ในนี้ถือว่าหาย
        _git("reset", "--hard", "FETCH_HEAD", cwd=target)
    else:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _discard_checkout(target)
        _git("clone", "--depth", "1", "--branch", ref, "--", repo, str(target), timeout=600)
    return target, _git("rev-parse", "--short", "HEAD", cwd=target)


def configured_source() -> tuple[str, str]:
    """(repo, ref) ที่ hub เครื่องนี้ดึงสูตรมา — จาก config.yaml ไม่ใช่จากคำขอของใคร

    `recipes.sync_repo` / `recipes.sync_ref` ว่าง = รีโปของทีม · หน้าเว็บใช้ค่านี้อย่างเดียว
    (web/selfupdate.py ปฏิเสธ remote จาก request ด้วยเหตุผลเดียวกัน) · CLI ยังทับได้ด้วย
    `--repo/--ref` เพราะคนที่พิมพ์คือคนที่ถือ shell ของ hub อยู่แล้ว
    """
    from lmds.config import Settings

    recipes = Settings.load().recipes
    return (recipes.sync_repo or DEFAULT_REPO, recipes.sync_ref or DEFAULT_REF)


def sync(repo: str = DEFAULT_REPO, ref: str = DEFAULT_REF, now: str = "") -> dict:
    """ดึงรีโปแล้วเขียน recipes-synced.yaml — คืนสรุปว่าได้อะไรมาบ้าง/ข้ามอะไรไป"""
    target, commit = fetch(repo, ref)
    origin = f"{repo_label(repo)}@{commit}"
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
