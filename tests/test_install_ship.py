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
    assert "git clone -q -b main /tmp/lmds-src.bundle AutoDeployDGXProject" in script
    assert "git fetch -q /tmp/lmds-src.bundle main" in script
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
    assert f"git clone -q -b main {ssh.REMOTE_BUNDLE}" in script

    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: SimpleNamespace(ok=False))
    assert "git clone --depth 1" in ssh.prepare_install(node), "ส่งไม่ได้ → ถอยไป GitHub ไม่ใช่ล้ม"


def test_install_flag_y_equals_assume_yes():
    text = Path(__file__).resolve().parents[1].joinpath("install.sh").read_text(encoding="utf-8")
    assert "-y|--yes) export LMDS_ASSUME_YES=1" in text
    assert "lmds web --enable --bind 0.0.0.0" in text, "ท้าย install.sh ต้องบอกวิธีเปิดคอนโซล"


# ── สคริปต์บน node ทำงานจริงกับ git จริง ─────────────────────────────────────────────

def _git(root, *args):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(root)}
    return subprocess.run(["git", "-C", str(root), *args], check=True, env=env,
                          capture_output=True, text=True).stdout.strip()


def _node_home(tmp_path: Path) -> Path:
    home = tmp_path / "node-home"
    (home / ".local" / "bin").mkdir(parents=True)
    lmds = home / ".local" / "bin" / "lmds"
    lmds.write_text("#!/bin/bash\necho lmds-stub\n", encoding="utf-8")
    lmds.chmod(0o755)
    return home


def _bundle_of(root: Path, tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    _git(root, "bundle", "create", str(out), "main")
    return out


def _run_node_script(home: Path, bundle: Path) -> subprocess.CompletedProcess:
    script = ssh.install_script(bundle=str(bundle))
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin",
                               "GIT_AUTHOR_NAME": "n", "GIT_AUTHOR_EMAIL": "n@x",
                               "GIT_COMMITTER_NAME": "n", "GIT_COMMITTER_EMAIL": "n@x"})


def _source(tmp_path: Path) -> Path:
    root = _git_repo(tmp_path)
    (root / "install.sh").write_text("#!/bin/bash\necho installed $(git rev-parse --short=7 HEAD)\n",
                                     encoding="utf-8")
    (root / "install.sh").chmod(0o755)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "installer")
    return root


def test_node_script_installs_on_a_fresh_machine_and_points_origin_at_github(tmp_path):
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    done = _run_node_script(home, _bundle_of(src, tmp_path, "a.bundle"))
    assert done.returncode == 0, done.stderr
    checkout = home / "AutoDeployDGXProject"
    assert _git(checkout, "rev-parse", "HEAD") == _git(src, "rev-parse", "HEAD")
    assert _git(checkout, "remote", "get-url", "origin") == ssh.REPO_URL
    assert "installed" in done.stdout and "lmds-stub" in done.stdout


def test_node_script_moves_a_copied_non_git_folder_aside_instead_of_dying(tmp_path):
    """เครื่องที่เคยติดตั้งแบบ copy (ไม่มี .git) — เดิม `git clone` ชนโฟลเดอร์ → exit 128"""
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    copied = home / "AutoDeployDGXProject"
    copied.mkdir(); (copied / "install.sh").write_text("old copy", encoding="utf-8")
    done = _run_node_script(home, _bundle_of(src, tmp_path, "b.bundle"))
    assert done.returncode == 0, done.stderr
    assert (home / "AutoDeployDGXProject" / ".git").is_dir()
    backups = list(home.glob("AutoDeployDGXProject.bak-*"))
    assert backups and (backups[0] / "install.sh").read_text(encoding="utf-8") == "old copy"


def test_node_script_follows_the_hub_when_the_checkout_was_edited_or_diverged(tmp_path):
    """แพตช์มือ/commit ค้างบน node — เดิม ff-only ล้ม · ตอนนี้เก็บไว้ที่ branch local-* + stash แล้วตาม hub"""
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    assert _run_node_script(home, _bundle_of(src, tmp_path, "c1.bundle")).returncode == 0
    checkout = home / "AutoDeployDGXProject"
    # node แยกสาย: commit ของตัวเอง + ไฟล์แก้ค้าง
    (checkout / "local-patch.txt").write_text("mine", encoding="utf-8")
    _git(checkout, "add", "."); _git(checkout, "commit", "-q", "-m", "local hack")
    (checkout / "install.sh").write_text("#!/bin/bash\necho edited\n", encoding="utf-8")
    # hub เดินหน้าต่อ
    (src / "new.txt").write_text("hub", encoding="utf-8")
    _git(src, "add", "."); _git(src, "commit", "-q", "-m", "hub moves on")
    done = _run_node_script(home, _bundle_of(src, tmp_path, "c2.bundle"))
    assert done.returncode == 0, done.stderr
    assert _git(checkout, "rev-parse", "HEAD") == _git(src, "rev-parse", "HEAD")
    assert "installed" in done.stdout and "edited" not in done.stdout
    branches = _git(checkout, "branch", "--list", "local-*")
    assert branches, "ของเดิมของ node ต้องไม่หาย"
    assert _git(checkout, "stash", "list"), "ไฟล์ที่แก้ค้างต้องอยู่ใน stash"


def test_install_builds_a_new_venv_and_swaps_only_on_success():
    """spark-head 2026-09-04: pip ล้มเพราะ PyPI ช้า แต่ --clear ลบ venv เดิมไปแล้ว → node ไม่มี lmds เลย"""
    text = Path(__file__).resolve().parents[1].joinpath("install.sh").read_text(encoding="utf-8")
    # venv ต้องถูกสร้าง *ที่ path จริง* (shebang ฝัง path) — ของเดิมย้ายไป venv.old ก่อน ล้มค่อยย้ายกลับ
    assert 'NEW_VENV="${INSTALL_DIR}/venv"' in text and 'OLD_VENV="${INSTALL_DIR}/venv.old"' in text
    assert 'make_venv "$NEW_VENV"' in text and "restore_old_venv" in text
    assert 'if ! "${NEW_VENV}/bin/pip" install --quiet "$REPO_DIR"; then' in text
    assert "รุ่นเดิมยังอยู่และใช้ได้ตามปกติ" in text
    assert "venv.new" not in text.replace("ห้ามสร้างที่ venv.new", "").replace("venv.new/bin/python", "")
    assert 'PIP_RETRIES="${PIP_RETRIES:-8}" PIP_TIMEOUT="${PIP_TIMEOUT:-60}"' in text
    assert 'python3 -m venv --clear "${INSTALL_DIR}/venv"' not in text
    import subprocess
    assert subprocess.run(["bash", "-n", str(Path(__file__).resolve().parents[1] / "install.sh")]).returncode == 0
