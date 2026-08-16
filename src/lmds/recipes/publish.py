"""ส่ง controller ที่ "รันผ่านจริงแล้ว" ขึ้นคลัง — ปิดลูปของ recipes/sync.py

sync ดึงสูตรจากรีโปมาใช้ (pull) · แฟ้มนี้คือทางกลับ (push): เมื่อ deploy โมเดลหนึ่ง
บนเครื่องจนได้ค่าที่ถูกและทดสอบผ่านแล้ว ส่ง controller ตัวนั้นขึ้นคลังเพื่อให้เครื่อง
อื่นในกอง `--sync` ไปใช้ได้เลย ไม่ต้องเดา parser/image/feature ใหม่ทุกครั้ง

**publish เฉพาะ "ค่าของโมเดล"** — controller ที่ generate ไว้ถือค่าเหล่านี้อยู่แล้ว
(engine, image, mmproj, parser, gguf, native default) ส่วน "ค่าของเครื่อง" (port,
context cap, slots) อยู่ใน bundle.env คนละไฟล์ ไม่ได้ถูกส่งขึ้นไป และฝั่ง parse
(recipes/controllers.py `_serving`) ก็ตัด context ทิ้งอยู่แล้ว — โมเดลตัวเดียวจึงเอา
ไป fit ใหม่ตามเครื่องปลายทางได้ ไม่ใช่ยัดค่าของเครื่องที่เคยรันไปให้เครื่องอื่น

ปลายทาง (`publish_repo`) เป็น config: ว่าง = local store ในเครื่องนี้ (ลูกค้าใช้แบบนี้
ไม่แตะรีโปเรา) · ทีมเราตั้งเป็นรีโป candidates แล้ว push ขึ้นไป review
"""

from __future__ import annotations

import socket
from pathlib import Path

import yaml

from lmds.config.paths import config_dir

from .sync import SyncError, _git, checkout_dir


def default_local_repo() -> Path:
    """local store ของ hub เครื่องนี้ — ค่าเริ่มต้นเมื่อไม่ได้ตั้ง publish_repo"""
    return config_dir() / "controllers" / "published-local"


def _is_remote(repo: str) -> bool:
    return "://" in repo or repo.startswith("git@")


def measured_features(profile: dict) -> list[str]:
    """สรุปความสามารถของโมเดลจาก MODEL_PROFILE.yaml เป็นสตริงสั้น ๆ สำหรับ MODEL_FEATURES

    ฝั่ง parse อ่านบรรทัดนี้เข้า recipe.notes — เป็นวิธีที่ความสามารถที่ทดสอบมาเดินทาง
    ไปกับ controller ได้ · เอาเฉพาะที่ profile ยืนยัน ไม่เดาเพิ่ม
    """
    feats: list[str] = []
    features = profile.get("features") or {}
    modalities = ((features.get("multimodal") or {}).get("modalities")) or []
    if "image" in modalities:
        feats.append("vision")
    if "audio" in modalities:
        feats.append("audio")
    tools = features.get("tool_calling") or {}
    if tools.get("enabled"):
        feats.append(f"tools ({tools['parser']})" if tools.get("parser") else "tools")
    reasoning = features.get("reasoning") or {}
    if reasoning.get("enabled"):
        feats.append(f"reasoning ({reasoning['parser']})" if reasoning.get("parser")
                     else "reasoning")
    return feats


def stamp_features(controller_text: str, features: list[str]) -> str:
    """ใส่/แทน MODEL_FEATURES ใน header ให้ตรงกับที่วัดได้ · ไม่มีก็ไม่ยัด"""
    if not features:
        return controller_text
    line = f'MODEL_FEATURES="{" · ".join(features)}"'
    lines = controller_text.splitlines()
    for i, existing in enumerate(lines):
        if existing.startswith("MODEL_FEATURES="):
            lines[i] = line
            return "\n".join(lines) + ("\n" if controller_text.endswith("\n") else "")
    # ยังไม่มี — วางต่อจาก MODEL_ID/MODEL_LABEL ที่ header อ่าน
    for i, existing in enumerate(lines):
        if existing.startswith(("MODEL_LABEL=", "MODEL_ID=")):
            lines.insert(i + 1, line)
            return "\n".join(lines) + ("\n" if controller_text.endswith("\n") else "")
    return controller_text


def build_profile(slug: str, profile: dict, validated_on: str, host: str,
                  features: list[str] | None = None) -> str:
    """PROFILE.yaml ที่เก็บ provenance — ตรวจสอบย้อนได้ว่าใคร/เครื่องไหน/เวอร์ชันไหนยืนยัน

    lmds ไม่ได้ parse ไฟล์นี้ (มันอ่าน header ของ controller) — ไฟล์นี้ไว้ให้คน review
    ก่อน promote เข้า canonical เห็นว่าค่านี้มาจากไหน
    """
    model = profile.get("model") or {}
    feats = features if features is not None else measured_features(profile)
    return yaml.safe_dump(
        {
            "slug": slug,
            "validated_on": validated_on,
            "validated_host": host,
            "source": {
                "id": model.get("id", ""),
                "revision": model.get("revision", ""),
                "gguf": model.get("selected_gguf", ""),
            },
            "engine": (profile.get("runtime") or {}).get("engine", ""),
            "measured_features": feats,
            # ตัดค่าของเครื่องออกชัด ๆ — ไม่มี port/context/slots ในนี้โดยเจตนา
            "note": "model-intrinsic only; per-machine port/context live in bundle.env",
        },
        allow_unicode=True, sort_keys=False,
    )


def _write_into(repo_dir: Path, slug: str, controller_name: str,
                controller_text: str, profile_text: str) -> Path:
    dest = repo_dir / "controllers" / slug
    dest.mkdir(parents=True, exist_ok=True)
    (dest / controller_name).write_text(controller_text, encoding="utf-8")
    (dest / controller_name).chmod(0o755)
    (dest / "PROFILE.yaml").write_text(profile_text, encoding="utf-8")
    return dest


def _ensure_local_git(path: Path) -> None:
    if (path / ".git").is_dir():
        return
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    _git("config", "user.email", "lmds@localhost", cwd=path)
    _git("config", "user.name", "lmds", cwd=path)


def publish(slug: str, controller_path: Path, profile: dict, *,
            features: list[str] | None = None,
            repo: str = "", ref: str = "main", now: str = "", host: str = "",
            push: bool = True) -> dict:
    """ส่ง controller ของ slug ขึ้นปลายทาง — คืนสรุป {target, remote, path, committed, features}

    repo ว่าง → local store · repo เป็น URL → fetch/clone แล้ว push (ถ้า push=True)

    `features` ที่ operator ระบุมาชนะค่าจาก profile: MODEL_PROFILE เป็น rule-based ตอน
    generate (เดาแบบระวังไว้ก่อน — coder ที่มี tools จริงอาจถูกเขียน false) ส่วนคนที่กด
    publish คือคนที่เพิ่งทดสอบมาจึงรู้ว่าวัดได้อะไรจริง · เงียบ = ใช้ที่ profile มี
    """
    controller_path = Path(controller_path)
    if not controller_path.is_file():
        raise SyncError(f"ไม่พบ controller ของ {slug}: {controller_path}")
    host = host or socket.gethostname()
    validated_on = f"{now} · {host}".strip(" ·") or host
    feats = features if features is not None else measured_features(profile)

    text = stamp_features(controller_path.read_text(encoding="utf-8"), feats)
    profile_text = build_profile(slug, profile, validated_on, host, feats)

    remote = bool(repo) and _is_remote(repo)
    if remote:
        repo_dir = checkout_dir(repo)
        if (repo_dir / ".git").is_dir():
            _git("remote", "set-url", "origin", repo, cwd=repo_dir)
            _git("fetch", "--depth", "1", "origin", ref, cwd=repo_dir)
            _git("reset", "--hard", f"origin/{ref}", cwd=repo_dir)
        else:
            repo_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _git("clone", "--depth", "1", "--branch", ref, repo, str(repo_dir), timeout=600)
    else:
        repo_dir = Path(repo) if repo else default_local_repo()
        _ensure_local_git(repo_dir)

    _write_into(repo_dir, slug, controller_path.name, text, profile_text)
    _git("add", "-A", cwd=repo_dir)
    status = _git("status", "--porcelain", cwd=repo_dir)
    base = {"target": repo or str(repo_dir), "remote": remote, "features": feats,
            "path": str(repo_dir / "controllers" / slug)}
    if not status.strip():
        return {**base, "committed": False}
    _git("commit", "-q", "-m", f"publish {slug} — validated {validated_on}", cwd=repo_dir)
    if remote and push:
        _git("push", "origin", f"HEAD:{ref}", cwd=repo_dir, timeout=600)
    return {**base, "committed": True}
