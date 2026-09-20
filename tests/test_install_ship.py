"""ติดตั้ง LMDS บนเครื่องใหม่จากโค้ดที่ hub ส่งไปให้ — ไม่ต้องให้ทุกเครื่องเข้า GitHub เอง

repo เป็น private · เดิมทุกเครื่องที่เพิ่มเข้าฟลีตต้องมี deploy key ก่อน ไม่งั้น
"could not read Username" — ขั้นที่ยุ่งยากที่สุดของการติดตั้ง (ผู้ใช้ 2026-09-04: "ต้องติดตั้งง่าย ไม่ยุ่งยาก")
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    assert "git clone -q /tmp/lmds-src.bundle AutoDeployDGXProject" in script
    assert "git fetch -q /tmp/lmds-src.bundle HEAD" in script
    assert "git checkout -q -B main HEAD" in script, "clone จาก bundle ที่มีแต่ HEAD = detached"
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
    assert f"git clone -q {ssh.REMOTE_BUNDLE}" in script

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
    _git(root, "bundle", "create", str(out), "HEAD")
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


# ── Update path 2026-09-06: hub dirty guard · regenerate controller บน node ──

def test_prepare_install_refuses_or_warns_when_hub_dirty(tmp_path, monkeypatch):
    """hub dirty=8 ขณะที่ทุก node dirty=0 → stamp เท่ากัน (dcefd91) ทั้งที่โค้ดต่างกัน — ต้องปฏิเสธ เว้นแต่ force"""
    from lmds.nodes import HubDirtyError
    from lmds.web import selfupdate

    root = _git_repo(tmp_path)
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: SimpleNamespace(ok=True))
    node = Node(name="n", host="h", user="u")
    assert "git fetch -q" in ssh.prepare_install(node)          # สะอาด = ส่งได้

    (root / "install.sh").write_text("#!/bin/bash\necho edited\n", encoding="utf-8")
    monkeypatch.setattr(selfupdate, "dirty_files", _real_dirty_files)   # conftest ปิดไว้ — เปิดของจริงเฉพาะเทสนี้
    with pytest.raises(HubDirtyError) as caught:
        ssh.prepare_install(node)
    text = str(caught.value)
    assert "install.sh" in text and "commit" in text and "--force" in text
    assert "git fetch -q" in ssh.prepare_install(node, force=True), "force = ยืนยันว่าจงใจ"


def _real_dirty_files(root):
    done = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    return [line[3:] for line in done.stdout.splitlines() if line.strip()]


def test_install_script_refreshes_stale_bundles(tmp_path):
    """หลัง install.sh ผ่าน สคริปต์บน node ต่อด้วย `lmds bundles refresh --all --if-older` และไม่ล้มเมื่อไม่มี bundle/คำสั่ง"""
    script = ssh.install_script(bundle="/tmp/lmds-src.bundle")
    assert "bundles refresh --all --if-older" in script
    assert script.index("./install.sh") < script.index("bundles refresh"), "regenerate ต้องมาหลัง install.sh"
    # รันจริง: lmds stub บันทึก argv — และถ้า stub ล้ม (lmds รุ่นเก่าไม่มีคำสั่ง) install ยัง exit 0
    src = _source(tmp_path)
    home = _node_home(tmp_path)
    calls = home / "lmds-calls.log"
    (home / ".local" / "bin" / "lmds").write_text(
        f"#!/bin/bash\necho \"lmds $*\" >> {calls}\n[ \"$1\" = bundles ] && exit 2\necho lmds-stub\n", encoding="utf-8")
    done = _run_node_script(home, _bundle_of(src, tmp_path, "r.bundle"))
    assert done.returncode == 0, done.stderr
    assert "lmds bundles refresh --all --if-older" in calls.read_text(encoding="utf-8")
    assert "regenerate controller ไม่สำเร็จ" in done.stdout, "รุ่นเก่าไม่มีคำสั่ง = บอกแล้วไปต่อ ไม่ล้ม install"


# ── ปักหมุดเวอร์ชัน (LMDS_REPO_REF) + bundle ต้องเป็นโค้ดที่ hub รันอยู่จริง ──────────
#
# สองข้อนี้เป็นเรื่องเดียวกัน: "เครื่องลูกค้าปักหมุดเวอร์ชันไม่ได้" — องค์กรที่มีหน้าต่าง
# เปลี่ยนแปลง/ต้องผ่าน audit รับการที่กด update แล้วเดินตาม main ของเราทันทีไม่ได้

def test_the_bundle_carries_the_commit_the_hub_actually_runs_not_the_tip_of_main(tmp_path, monkeypatch):
    """hub ที่ checkout tag ไว้เคยส่ง *main* ไปให้ทั้งฟลีต โดย stamp ยังบอกว่า "ตรง hub"

    เทสนี้จำลองเป๊ะ: main เดินไปข้างหน้าแล้ว แต่ hub ยืนอยู่ที่ tag เก่า
    """
    src = _source(tmp_path)
    pinned = _git(src, "rev-parse", "HEAD")
    _git(src, "tag", "v-pinned")
    (src / "after.txt").write_text("main เดินต่อ", encoding="utf-8")
    _git(src, "add", "."); _git(src, "commit", "-q", "-m", "main moves on")
    moved = _git(src, "rev-parse", "HEAD")
    assert moved != pinned
    _git(src, "checkout", "-q", "--detach", "v-pinned")      # hub ยืนที่ tag

    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: src)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    bundle = ssh.source_bundle()
    assert bundle is not None, "hub ที่ checkout tag ไว้ต้อง pack ได้ ไม่ใช่ถอยไป GitHub เงียบ ๆ"

    home = _node_home(tmp_path)
    done = _run_node_script(home, bundle)
    assert done.returncode == 0, done.stderr
    checkout = home / "AutoDeployDGXProject"
    assert _git(checkout, "rev-parse", "HEAD") == pinned, "node ต้องได้ commit ที่ hub รันอยู่"
    assert not (checkout / "after.txt").exists(), "ของจาก main ที่ hub ไม่ได้รันต้องไม่หลุดไป"
    assert _git(checkout, "branch", "--show-current") == "main", "clone จาก bundle ต้องไม่ทิ้ง node ไว้ที่ detached"


def test_a_hub_cloned_at_a_tag_can_still_ship_its_code(tmp_path, monkeypatch):
    """`git clone --branch v0.8.0` ไม่มี ref `main` ในเครื่องเลย — ของเดิม pack ล้มทุกครั้ง"""
    src = _source(tmp_path)
    _git(src, "tag", "v0.8.0")
    shallow = tmp_path / "hub-at-tag"
    subprocess.run(["git", "clone", "-q", "--branch", "v0.8.0", str(src), str(shallow)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(shallow), "branch", "-D", "main"], capture_output=True)
    assert subprocess.run(["git", "-C", str(shallow), "rev-parse", "--verify", "-q", "main"],
                          capture_output=True).returncode != 0, "ต้องไม่มี main จริง ๆ"

    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: shallow)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    assert ssh.source_bundle() is not None


def test_a_pinned_ref_is_what_the_node_installs(tmp_path, monkeypatch):
    """$LMDS_REPO_REF=<tag> → node ต้องได้ tag นั้น ไม่ใช่ปลาย branch ปริยาย"""
    src = _source(tmp_path)
    pinned = _git(src, "rev-parse", "HEAD")
    _git(src, "tag", "v-locked")
    (src / "unreleased.txt").write_text("ยังไม่ปล่อย", encoding="utf-8")
    _git(src, "add", "."); _git(src, "commit", "-q", "-m", "unreleased work")

    monkeypatch.setattr(ssh, "REPO_REF", "v-locked")
    monkeypatch.setattr(ssh, "REPO_URL", str(src))
    script = ssh.install_script()
    assert "\nref=v-locked\n" in script and 'git clone -q --depth 1 --branch "$ref"' in script

    home = _node_home(tmp_path)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert done.returncode == 0, done.stderr
    checkout = home / "AutoDeployDGXProject"
    assert _git(checkout, "rev-parse", "HEAD") == pinned
    assert not (checkout / "unreleased.txt").exists(), "ของที่ยังไม่ปล่อยต้องไม่ไปถึงเครื่องลูกค้า"
    assert "ติดตั้งจาก ref ที่ปักหมุดไว้: v-locked" in done.stdout


def test_a_pinned_ref_stops_the_hub_from_shipping_its_own_code_over_it(tmp_path, monkeypatch):
    """ปักหมุดแล้วยังส่ง bundle ของ hub ไป = ลบล้างคำสั่งเงียบ ๆ (HEAD ของ hub ไม่ใช่ ref นั้น)"""
    root = _git_repo(tmp_path)
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    node = Node(name="n", host="h", user="u")
    pushed = []
    monkeypatch.setattr(ssh, "push_file", lambda n, l, r, timeout=1800:
                        (pushed.append(l), SimpleNamespace(ok=True))[1])

    assert ssh.REMOTE_BUNDLE in ssh.prepare_install(node) and pushed, "ไม่ปักหมุด = ส่ง bundle ตามเดิม"

    pushed.clear()
    monkeypatch.setattr(ssh, "REPO_REF", "v-locked")
    script = ssh.prepare_install(node)
    assert not pushed, "ปักหมุดแล้วต้องไม่ส่ง bundle ไปทับ"
    assert "\nref=v-locked\n" in script and ssh.REMOTE_BUNDLE not in script


def test_the_web_service_carries_the_pin_into_its_own_environment():
    """ปุ่ม install/update ทำงานในบริบทของ service ไม่ใช่ shell ที่ผู้ใช้ export ไว้ —
    ไม่ส่งต่อ = กดจากหน้าเว็บแล้วได้เวอร์ชันผิดเงียบ ๆ ทั้งที่ CLI ถูก"""
    from lmds.web import daemon

    assert "LMDS_REPO_REF" in daemon._FORWARD_ENV and "LMDS_REPO_URL" in daemon._FORWARD_ENV


def test_a_private_repo_error_does_not_promise_a_bundle_that_pinning_disabled(monkeypatch):
    node = Node(name="n", host="h", user="u")
    failure = "fatal: could not read Username for 'https://github.com'"
    assert "ปกติ hub จะส่งโค้ดของตัวเองไปให้" in ssh.explain_install_failure(failure, node)

    monkeypatch.setattr(ssh, "REPO_REF", "v0.8.0")
    pinned = ssh.explain_install_failure(failure, node)
    assert "ปกติ hub จะส่งโค้ดของตัวเองไปให้" not in pinned, "ปักหมุดแล้ว hub จงใจไม่ส่ง — บอกแบบนั้นคือโกหก"
    assert "v0.8.0" in pinned and "ถอนหมุด" in pinned
