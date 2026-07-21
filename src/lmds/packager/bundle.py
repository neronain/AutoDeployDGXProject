"""Packager — PACKAGE_SHA256SUMS + ZIP ตาม delivery contract"""

from __future__ import annotations

import zipfile
from pathlib import Path

from lmds.validator import CHECKSUM_FILE, compute_checksums


def write_checksums(bundle_dir: Path) -> Path:
    sums = compute_checksums(bundle_dir)
    path = bundle_dir / CHECKSUM_FILE
    body = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    path.write_text(body, encoding="utf-8")
    return path


def make_zip(bundle_dir: Path) -> Path:
    """สร้าง <slug>.zip ข้าง bundle dir — ไฟล์ใน zip อยู่ใต้โฟลเดอร์ <slug>/"""
    zip_path = bundle_dir.parent / f"{bundle_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(bundle_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix == ".zip":
                continue
            archive.write(file_path, f"{bundle_dir.name}/{file_path.relative_to(bundle_dir).as_posix()}")
    return zip_path
