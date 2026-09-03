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

import re
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


# ค่าที่ "พิสูจน์แล้ว" มักไม่ได้อยู่ใน controller — อยู่ใน bundle.env / bundle.args ที่ `lmds set` เขียน
#
# เคสจริง 2026-09-03: spark04 กับ spark-worker รันได้ก็เพราะ `lmds set --image <digest v0.28.0>
# --tool-parser qwen3_xml --reasoning-parser qwen3 --extra-args "…MTP…"` แต่ header ของ controller
# ยังเป็นค่าเดิมจาก plan (image nvcr ที่ล้ม, ไม่มี parser) · publish แบบเดิมส่ง header นั้นขึ้นไป
# เครื่องปลายทาง sync มาก็ได้สูตรที่ start ไม่ขึ้น — ตรงข้ามกับจุดประสงค์ของคลัง
#
# พับเฉพาะ "ค่าของโมเดล" (image ที่มี kernel ตรง, parser, env ของ engine, แฟล็กเพิ่ม, mmproj)
# ค่าของเครื่อง (port, context, gpu-util, slots, ชื่อที่เสิร์ฟ, bind) ไม่พับโดยเจตนา
MODEL_KEYS = ("VLLM_IMAGE", "LLAMACPP_IMAGE", "SGLANG_IMAGE", "TOOL_CALL_PARSER",
              "REASONING_PARSER", "ENGINE_ENV", "CHAT_TEMPLATE", "MMPROJ_FILE", "IMAGE_MIN_TOKENS")
_ENV_LINE = re.compile(r'^([A-Z][A-Z0-9_]*)="?\$\{\1(:?-)(.*)\}"?\s*$')


def bundle_overrides(bundle_dir: Path) -> dict[str, str]:
    """ค่าของโมเดลที่ bundle.env/bundle.args ของ bundle นี้ทับไว้ — {} ถ้าไม่มี"""
    out: dict[str, str] = {}
    env = bundle_dir / "bundle.env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = _ENV_LINE.match(line.strip())
            if m and m.group(1) in MODEL_KEYS and m.group(3):
                out[m.group(1)] = m.group(3)
    args = bundle_dir / "bundle.args"
    if args.is_file():
        text = " ".join(args.read_text(encoding="utf-8").split())
        if text:
            out["EXTRA_SERVE_ARGS"] = text
    return out


def fold_overrides(controller_text: str, overrides: dict[str, str]) -> tuple[str, dict[str, str], list[str]]:
    """เขียนค่าที่ทับไว้ลงเป็นค่าตั้งต้นใน header — คืน (text, ที่พับได้, ที่พับไม่ได้)

    พับได้เฉพาะบรรทัดที่ header มีอยู่แล้ว (`KEY="${KEY:-…}"`) — ไม่ประดิษฐ์ตัวแปรใหม่ให้
    controller ที่ไม่รู้จัก · EXTRA_SERVE_ARGS ลงที่ `EXTRA_SERVE_ARGS_DEFAULT='…'` (single quote)
    เพราะ JSON มี } ที่จะตัด ${VAR:-…} ขาด — ของเดิมที่ยังไม่มีบรรทัดนี้จะถูกรายงานว่าพับไม่ได้
    """
    lines = controller_text.splitlines()
    applied: dict[str, str] = {}
    skipped: list[str] = []
    for key, value in overrides.items():
        if key == "EXTRA_SERVE_ARGS":
            pat = re.compile(r"^EXTRA_SERVE_ARGS_DEFAULT='.*'\s*$")
            repl = "EXTRA_SERVE_ARGS_DEFAULT='" + value.replace("'", "'\\''") + "'"
        else:
            pat = re.compile(rf'^{key}="?\$\{{{key}(:?-)(.*)\}}"?\s*(#.*)?$')
            repl = None
        for i, line in enumerate(lines):
            m = pat.match(line)
            if not m:
                continue
            lines[i] = repl if repl is not None else f'{key}="${{{key}{m.group(1)}{value}}}"'
            applied[key] = value
            break
        else:
            skipped.append(key)
    # `lmds set --image` เขียน image ลงทั้ง VLLM_IMAGE และ LLAMACPP_IMAGE — controller มี engine เดียว
    # จึงมีแค่ key เดียว · อีก key ไม่ใช่ "พับไม่ได้" แต่ไม่เกี่ยวกับ engine นี้
    if any(k.endswith("_IMAGE") for k in applied):
        skipped = [k for k in skipped if not k.endswith("_IMAGE")]
    text = "\n".join(lines) + ("\n" if controller_text.endswith("\n") else "")
    return text, applied, skipped


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
                  features: list[str] | None = None,
                  overrides: dict[str, str] | None = None) -> str:
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
            # ค่าของโมเดลที่เครื่องนี้ต้องตั้งทับ plan ถึงจะรันได้ — คน review เห็นทันทีว่าต่างจากค่าเดา
            "overrides": dict(overrides or {}),
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


def _ensure_identity(repo_dir: Path, host: str) -> None:
    """commit ต้องมี user.email — hub ที่ไม่เคยตั้ง git identity (เคสจริง 2026-09-03) จะล้มที่นี่

    เดิมตั้งให้เฉพาะ repo ที่ init เอง · repo ที่ clone มา (candidates ของทีม) ไม่มี identity
    แล้ว git ปฏิเสธ commit ด้วย "unable to auto-detect email address" ทั้ง 23 ตัวรวด
    ตั้งเป็น local config ของ repo นั้น ไม่แตะ global ของเครื่อง
    """
    try:
        if _git("config", "user.email", cwd=repo_dir):
            return
    except SyncError:
        pass                                   # ไม่มีค่า — git คืน exit 1
    _git("config", "user.email", f"lmds@{host or 'localhost'}", cwd=repo_dir)
    _git("config", "user.name", f"lmds ({host})" if host else "lmds", cwd=repo_dir)


def publish(slug: str, controller_path: Path, profile: dict, *,
            features: list[str] | None = None,
            repo: str = "", ref: str = "main", now: str = "", host: str = "",
            push: bool = True, bundle_dir: Path | None = None) -> dict:
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

    overrides = bundle_overrides(Path(bundle_dir) if bundle_dir else controller_path.parent)
    text, applied, skipped = fold_overrides(controller_path.read_text(encoding="utf-8"), overrides)
    text = stamp_features(text, feats)
    profile_text = build_profile(slug, profile, validated_on, host, feats, overrides=applied)

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
    _ensure_identity(repo_dir, host)
    _git("add", "-A", cwd=repo_dir)
    status = _git("status", "--porcelain", cwd=repo_dir)
    base = {"target": repo or str(repo_dir), "remote": remote, "features": feats,
            "path": str(repo_dir / "controllers" / slug),
            "overrides": applied, "unfolded": skipped}
    if not status.strip():
        return {**base, "committed": False}
    _git("commit", "-q", "-m", f"publish {slug} — validated {validated_on}", cwd=repo_dir)
    if remote and push:
        _git("push", "origin", f"HEAD:{ref}", cwd=repo_dir, timeout=600)
    return {**base, "committed": True}
