"""ติดตั้ง LMDS บนเครื่องใหม่จากโค้ดที่ hub ส่งไปให้ — ไม่ต้องให้ทุกเครื่องเข้า GitHub เอง

repo เป็น private · เดิมทุกเครื่องที่เพิ่มเข้าฟลีตต้องมี deploy key ก่อน ไม่งั้น
"could not read Username" — ขั้นที่ยุ่งยากที่สุดของการติดตั้ง (ผู้ใช้ 2026-09-04: "ต้องติดตั้งง่าย ไม่ยุ่งยาก")
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from lmds.nodes import Node, ssh


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "install.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, env=env, capture_output=True)
    return root


def test_the_bundle_script_clones_from_the_shipped_file_and_points_origin_back_to_github():
    script = ssh.install_script(bundle="/tmp/lmds-src.bundle")
    assert "git clone /tmp/lmds-src.bundle AutoDeployDGXProject" in script
    assert "git pull --ff-only /tmp/lmds-src.bundle main" in script
    assert f"git remote set-url origin {ssh.REPO_URL}" in script
    assert "rm -f /tmp/lmds-src.bundle" in script
    assert "LMDS_SKIP_PREREQ=1 ./install.sh" in script
    # ทางเดิมยังอยู่ — hub ที่ไม่ได้ติดตั้งจาก git ใช้ต่อได้
    assert "git clone --depth 1" in ssh.install_script()


def test_the_hub_packs_its_own_checkout_once_per_commit(tmp_path, monkeypatch):
    root = _git_repo(tmp_path)
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    bundle = ssh.source_bundle()
    assert bundle is not None and bundle.is_file()
    verify = subprocess.run(["git", "bundle", "verify", str(bundle)], capture_output=True, text=True)
    assert verify.returncode == 0, verify.stderr
    first_mtime = bundle.stat().st_mtime
    assert ssh.source_bundle() == bundle and bundle.stat().st_mtime == first_mtime


def test_no_checkout_means_the_old_github_path(monkeypatch):
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: None)
    assert ssh.source_bundle() is None
    assert "git clone --depth 1" in ssh.prepare_install(Node(name="n", host="h", user="u"))


def test_prepare_install_ships_the_code_first_and_falls_back_when_scp_fails(tmp_path, monkeypatch):
    root = _git_repo(tmp_path)
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    node = Node(name="n", host="h", user="u")
    pushed = []

    def fake_push(n, local, remote, timeout=1800):
        pushed.append((local, remote))
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(ssh, "push_file", fake_push)
    script = ssh.prepare_install(node)
    assert pushed and pushed[0][1] == ssh.REMOTE_BUNDLE
    assert Path(pushed[0][0]).is_file()
    assert f"git clone {ssh.REMOTE_BUNDLE}" in script

    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: SimpleNamespace(ok=False))
    assert "git clone --depth 1" in ssh.prepare_install(node), "ส่งไม่ได้ → ถอยไป GitHub ไม่ใช่ล้ม"


def test_install_flag_y_equals_assume_yes():
    text = Path(__file__).resolve().parents[1].joinpath("install.sh").read_text(encoding="utf-8")
    assert "-y|--yes) export LMDS_ASSUME_YES=1" in text
    assert "lmds web --enable --bind 0.0.0.0" in text, "ท้าย install.sh ต้องบอกวิธีเปิดคอนโซล"
