"""อัปเดตต้องผ่านบนเครื่องที่ branch main ไม่มี upstream — รันสคริปต์จริงกับ git จริง

เคสจริง 2026-10-09: ลูกค้ากดปุ่ม Update LMDS บนหน้าเว็บ บางเครื่องล้ม exit 1 ทั้งที่ดึงโค้ดมาได้แล้ว

    From https://github.com/neronain/AutoDeployDGXProject
       6e2b4747f..45cb59c30  main       -> origin/main
    There is no tracking information for the current branch.

เครื่องพวกนั้นติดตั้งครั้งแรกจากโค้ดที่ hub ส่งมา (git bundle): `git checkout -B main HEAD` สร้าง branch main ที่ไม่มี
upstream แล้ว `git pull --ff-only` เปล่า ๆ ของปุ่ม Update (และของทาง GitHub ใน `lmds node install`) ไม่รู้ว่าจะ
merge กับอะไร · เครื่องที่ clone จาก GitHub ตรง ๆ ไม่เป็น — ฟลีตของเราทั้ง 15 เครื่องจึงไม่เคยเจอ
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from lmds.nodes import ssh
from lmds.web import selfupdate
from tests.test_install_ship import _bundle_of, _git, _node_home, _run_node_script, _source

ENV_PATH = "/usr/bin:/bin:/usr/local/bin"


def _installed_from_a_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """เครื่องที่ติดตั้งครั้งแรกจาก bundle ของ hub ด้วยสคริปต์ **รุ่นก่อนแก้** — main ไม่มี upstream · origin = "GitHub"

    คืน (src ที่ทำหน้าที่เป็น GitHub, home ของเครื่องนั้น, checkout)
    """
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    assert _run_node_script(home, _bundle_of(src, tmp_path, "first.bundle")).returncode == 0
    checkout = home / "AutoDeployDGXProject"
    for key in ("branch.main.remote", "branch.main.merge"):       # สภาพของเครื่องที่ติดตั้งไว้ก่อนสคริปต์จะตั้งให้
        subprocess.run(["git", "-C", str(checkout), "config", "--unset", key], capture_output=True)
    _git(checkout, "remote", "set-url", "origin", str(src))
    assert _upstream(checkout) == "", "ต้องเริ่มจากสภาพที่ลูกค้าเจอ: main ไม่มี upstream"
    return src, home, checkout


def _upstream(checkout: Path) -> str:
    done = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "main@{u}"],
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def _hub_moves_on(src: Path) -> str:
    (src / "new-release.txt").write_text("ของใหม่", encoding="utf-8")
    _git(src, "add", "."); _git(src, "commit", "-q", "-m", "new release")
    return _git(src, "rev-parse", "HEAD")


def _bash(script: str, cwd: Path, home: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=str(cwd), capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": ENV_PATH})


def test_the_update_button_works_on_a_machine_installed_from_the_hubs_bundle(tmp_path):
    src, home, checkout = _installed_from_a_bundle(tmp_path)
    latest = _hub_moves_on(src)

    done = _bash(selfupdate.update_script(restart=False), checkout, home)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "no tracking information" not in done.stderr
    assert _git(checkout, "rev-parse", "HEAD") == latest
    assert "installed" in done.stdout, "ต้องไปถึงขั้นติดตั้ง ไม่ใช่ตายที่ขั้นดึงโค้ด"
    assert _upstream(checkout) == "origin/main", "ซ่อม upstream ให้ด้วย — git pull ที่คนพิมพ์เองบนเครื่องนั้นต้องใช้ได้"


def test_node_install_from_github_works_on_a_machine_installed_from_the_hubs_bundle(tmp_path, monkeypatch):
    """hub ส่ง bundle ไม่ได้รอบนี้ (scp ล้ม · hub ไม่มี checkout) → สคริปต์ถอยไปดึงจาก GitHub เอง — ทางเดียวกับปุ่ม Update"""
    src, home, checkout = _installed_from_a_bundle(tmp_path)
    latest = _hub_moves_on(src)
    monkeypatch.setattr(ssh, "REPO_REF", "")

    done = _bash(ssh.install_script(), home, home)

    assert done.returncode == 0, done.stdout + done.stderr
    assert _git(checkout, "rev-parse", "HEAD") == latest
    assert "installed" in done.stdout and "lmds-stub" in done.stdout


def test_a_fresh_install_from_a_bundle_gets_an_upstream_so_a_plain_git_pull_works(tmp_path):
    """ต้นเหตุ: checkout ที่สคริปต์ bundle สร้างต้องมี upstream ตั้งแต่แรก — เครื่องนั้นอาจรันโค้ดรุ่นเก่าที่ยัง pull เปล่า ๆ"""
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    assert _run_node_script(home, _bundle_of(src, tmp_path, "a.bundle")).returncode == 0
    checkout = home / "AutoDeployDGXProject"
    assert _git(checkout, "config", "--get", "branch.main.remote") == "origin"
    assert _git(checkout, "config", "--get", "branch.main.merge") == "refs/heads/main"

    _git(checkout, "remote", "set-url", "origin", str(src))
    latest = _hub_moves_on(src)
    done = _bash("git pull --ff-only", checkout, home)            # คำสั่งของปุ่ม Update รุ่นก่อนแก้
    assert done.returncode == 0, done.stderr
    assert _git(checkout, "rev-parse", "HEAD") == latest


def test_the_next_bundle_install_repairs_a_machine_that_has_no_upstream(tmp_path):
    """เครื่องที่ติดตั้งไว้แล้วซ่อมได้โดยไม่ต้องให้ใครเข้าไปพิมพ์: hub อัปเดตเครื่องนั้นรอบถัดไป upstream ก็กลับมา"""
    src, home, checkout = _installed_from_a_bundle(tmp_path)
    _git(checkout, "remote", "set-url", "origin", ssh.REPO_URL)
    _hub_moves_on(src)

    done = _run_node_script(home, _bundle_of(src, tmp_path, "second.bundle"))

    assert done.returncode == 0, done.stderr
    assert _git(checkout, "config", "--get", "branch.main.remote") == "origin"
    assert _git(checkout, "config", "--get", "branch.main.merge") == "refs/heads/main"


def test_a_pinned_detached_checkout_is_explained_and_left_where_it_is(tmp_path):
    """detached HEAD (เครื่องที่ปักหมุดเวอร์ชันไว้) ไม่มี branch ให้ตาม — ต้องบอกเป็นภาษาคนและไม่ย้ายให้เอง"""
    src, home, checkout = _installed_from_a_bundle(tmp_path)
    pinned = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "checkout", "-q", "--detach", "HEAD")
    _hub_moves_on(src)

    done = _bash(selfupdate.update_script(restart=False), checkout, home)

    assert done.returncode != 0
    assert _git(checkout, "rev-parse", "HEAD") == pinned, "เครื่องที่ปักหมุดไว้ต้องอยู่ที่เดิม"
    assert "installed" not in done.stdout, "ดึงไม่ได้แล้วต้องไม่ติดตั้งต่อ"
    said = [line for line in done.stderr.splitlines() if "git checkout main" in line]
    assert said and "detached HEAD" in done.stderr, done.stderr
