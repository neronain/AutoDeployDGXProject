"""ตำแหน่งไฟล์ config ของ LMDS — override ได้ด้วย env LMDS_CONFIG_DIR (ใช้ในเทสด้วย)."""

from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    root = os.environ.get("LMDS_CONFIG_DIR")
    if root:
        return Path(root)
    return Path.home() / ".config" / "lmds"


def config_file() -> Path:
    return config_dir() / "config.yaml"


def credentials_file() -> Path:
    return config_dir() / "credentials"


def profile_file() -> Path:
    return config_dir() / "profile.yaml"


def sessions_dir() -> Path:
    return config_dir() / "sessions"


def ensure_config_dir() -> Path:
    d = config_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d
