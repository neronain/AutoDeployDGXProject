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
    assert f'git clone -q "$HOME"/{ssh.REMOTE_BUNDLE} AutoDeployDGXProject' in script, "วางใต้ home ไม่ใช่ /tmp"

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


# ── 15 เครื่องพร้อมกัน: ทุกตัวต้องได้ bundle ของ hub · ถอยไป GitHub ต้องบอกเสมอว่าทำไม ──────────
#
# audit 2026-10-06: วิธี rollout ที่ทีมใช้จริงคือยิง `lmds node install <n>` 15 process พร้อมกัน ·
# แคชเป็นแบบ "ดูว่ามีไหม → ไม่มีก็สร้าง" บนไฟล์เดียวกัน และ `git bundle create` ถือ `<ไฟล์>.lock`
# → แคชเย็น 14 ใน 15 ตัวตายด้วย exit 128 · source_bundle() คืน None · สคริปต์ถอยไป `git pull` จาก
# GitHub โดยไม่มีบรรทัดไหนบอก (และทางนั้นไม่ regenerate controller ด้วย) — hub ที่ HEAD ยังไม่ push /
# ยืนอยู่ที่ tag / node ที่ออกเน็ตไม่ได้ จบที่ "ฟลีตคนละ commit" โดยทุกตัวรายงานว่าสำเร็จ

_RACE_WORKER = """
import json, os, sys, time
from pathlib import Path
import lmds.web.selfupdate as selfupdate
selfupdate.source_root = lambda: Path(os.environ["RACE_REPO"])
from lmds.nodes import ssh
if os.environ.get("RACE_NO_FLOCK"):
    ssh.fcntl = None                      # ระบบไฟล์/แพลตฟอร์มที่ล็อกไม่ได้ — ต้องยังได้ครบทุกตัว
box = Path(os.environ["RACE_BOX"])
(box / f"ready-{sys.argv[1]}").write_text("")
deadline = time.time() + 60
while not (box / "go").exists() and time.time() < deadline:
    time.sleep(0.005)                     # ปล่อยพร้อมกันทุกตัว เหมือน `( lmds node install $n ) &` × 15
bundle = ssh.source_bundle()
print(json.dumps({"bundle": str(bundle) if bundle else None}))
"""


def _slow_git(tmp_path: Path) -> tuple[Path, Path]:
    """git จริงที่ `bundle create` ช้าลง 0.3 วิ — ให้ทุก process ซ้อนกันตรงขั้น pack แน่นอน ไม่ใช่แล้วแต่ดวง

    repo ของเทสเล็กจน pack เสร็จใน ~10 ms (ของจริง 6.7 MB ใช้ ~0.2 วิ) · คืน (โฟลเดอร์ที่ใส่หน้า PATH, log)
    """
    import shlex
    import shutil

    real = shutil.which("git")
    assert real, "เทสนี้ต้องมี git จริง"
    bin_dir = tmp_path / "slow-git"
    bin_dir.mkdir()
    log = tmp_path / "git-bundle-calls.log"
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/bin/bash\n"
        'case " $* " in *" bundle create "*) echo create >> ' + shlex.quote(str(log)) + "; sleep 0.3 ;; esac\n"
        f"exec {shlex.quote(real)} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return bin_dir, log


def _race(tmp_path: Path, count: int, *, flock: bool) -> tuple[list[dict], Path, Path, Path]:
    import json
    import os
    import sys
    import time

    src = _source(tmp_path)
    cache = tmp_path / "hub-tmp"
    box = tmp_path / "box"
    cache.mkdir(); box.mkdir()
    bin_dir, log = _slow_git(tmp_path)
    env = {**os.environ, "RACE_REPO": str(src), "RACE_BOX": str(box), "TMPDIR": str(cache),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    env.pop("RACE_NO_FLOCK", None)
    if not flock:
        env["RACE_NO_FLOCK"] = "1"
    procs = [subprocess.Popen([sys.executable, "-c", _RACE_WORKER, str(i)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(count)]
    try:
        deadline = time.time() + 120
        while len(list(box.glob("ready-*"))) < count:
            assert time.time() < deadline, "worker ไม่พร้อมใน 120 วิ"
            assert all(p.poll() is None for p in procs), [p.communicate() for p in procs if p.poll() is not None]
            time.sleep(0.01)
        (box / "go").write_text("")
        results = []
        for proc in procs:
            out, err = proc.communicate(timeout=180)
            assert proc.returncode == 0, err
            results.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    return results, src, cache, log


@pytest.mark.parametrize("flock", [True, False], ids=["with-flock", "no-flock"])
def test_fifteen_installs_at_once_all_get_the_hubs_own_bundle(tmp_path, flock):
    """rollout จริง: 15 process · แคชเย็น · ทุกตัวต้องได้ bundle ของ commit ที่ hub รัน ไม่มีตัวไหนได้ None"""
    results, src, cache, log = _race(tmp_path, 15, flock=flock)

    got = {r["bundle"] for r in results}
    assert None not in got, f"{sum(r['bundle'] is None for r in results)} ใน 15 ตัวไม่ได้ bundle → ถอยไป GitHub"
    assert len(got) == 1, got
    bundle = Path(got.pop())
    assert subprocess.run(["git", "bundle", "verify", str(bundle)], capture_output=True, cwd=src).returncode == 0
    heads = subprocess.run(["git", "bundle", "list-heads", str(bundle)], capture_output=True, text=True).stdout
    assert heads.split()[0] == _git(src, "rev-parse", "HEAD"), "ของข้างในต้องเป็น commit ที่ hub รันอยู่"
    # ไม่ทิ้งไฟล์ครึ่ง ๆ กลาง ๆ ไว้ให้รอบหน้าหยิบไปใช้ (ไฟล์ชั่วคราวของเรา / .lock ของ git)
    leftovers = [p.name for p in cache.iterdir() if p.name.endswith((".tmp", ".bundle.lock"))]
    assert not leftovers, leftovers
    packs = log.read_text(encoding="utf-8").count("create")
    if flock:
        assert packs == 1, f"ล็อกได้ = pack รอบเดียวพอ ที่เหลือรอแล้วใช้ของที่เสร็จ (pack ไป {packs} รอบ)"


def test_a_decoy_sitting_at_the_cache_path_is_not_shipped_to_the_fleet(tmp_path, monkeypatch):
    """แคชอยู่ใน temp dir ที่ทุก user เขียนได้และชื่อเดาได้ (lmds-src-<commit>.bundle) — ของที่วางรอไว้
    (symlink / ไฟล์ของ user อื่น) ต้องไม่ถูกหยิบไปส่งให้ทั้งฟลีตแทนโค้ดของ hub"""
    src = _source(tmp_path)
    head = _git(src, "rev-parse", "HEAD")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    decoy = _bundle_of(_git_repo(elsewhere), tmp_path, "decoy.bundle")
    decoy_bytes = decoy.read_bytes()

    cache = tmp_path / "hub-tmp"
    cache.mkdir()
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: src)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(cache))
    (cache / f"lmds-src-{_git(src, 'rev-parse', '--short', 'HEAD')}.bundle").symlink_to(decoy)

    bundle = ssh.source_bundle()
    assert bundle is not None
    heads = subprocess.run(["git", "bundle", "list-heads", str(bundle)], capture_output=True, text=True).stdout
    assert heads.split()[0] == head, "ส่งของที่คนอื่นวางรอไว้แทนโค้ดของ hub"
    assert decoy.read_bytes() == decoy_bytes, "ต้องไม่เขียนทะลุ symlink ไปทับไฟล์ปลายทาง"


def test_a_pack_that_fails_says_why_instead_of_quietly_using_github(tmp_path, monkeypatch):
    """pack ไม่ได้ (ที่นี่: temp dir เขียนไม่ได้ — git จริงล้มจริง) → สคริปต์ที่ได้ต้องพิมพ์เหตุผล และ notice ต้องมีข้อความของ git"""
    import os

    src = _source(tmp_path)
    cache = tmp_path / "read-only-tmp"
    cache.mkdir()
    cache.chmod(0o500)
    if os.access(cache, os.W_OK):
        pytest.skip("รันเป็น root — โฟลเดอร์ 0500 ยังเขียนได้ จำลองเคสนี้ไม่ได้")
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: src)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(cache))
    monkeypatch.setattr(ssh, "REPO_URL", str(src))
    pushed = []
    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: pushed.append(a) or SimpleNamespace(ok=True, stderr=""))
    try:
        script, notice = ssh.plan_install(Node(name="msi-4", host="h", user="u"))
    finally:
        cache.chmod(0o700)
    assert not pushed, "ไม่มี bundle ก็ต้องไม่มีอะไรให้ส่ง"
    assert "git bundle create" in notice and "Permission denied" in notice, notice

    home = _node_home(tmp_path)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert done.returncode == 0, done.stderr
    said = [ln for ln in done.stdout.splitlines() if "ไม่ได้ติดตั้งจากโค้ดของ hub" in ln]
    assert said and "Permission denied" in said[0] and str(src) in said[0], done.stdout
    assert done.stdout.index(said[0]) < done.stdout.index("installed"), "ต้องบอกก่อนลงมือ ไม่ใช่ท้ายงาน"


def test_a_failed_copy_says_why_and_no_checkout_says_so_too(tmp_path, monkeypatch):
    node = Node(name="msi-4", host="h", user="u")
    root = _git_repo(tmp_path)
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: ssh.Result(
        1, "", 'scp: dest open ".lmds-src.bundle": No space left on device\n'))
    script, notice = ssh.plan_install(node)
    assert "git clone --depth 1" in script
    assert "ส่งไฟล์ไป msi-4 ไม่ได้" in notice and "No space left on device" in notice

    def boom(*a, **k):
        raise ssh.NodeError("ไม่พบคำสั่ง scp — ติดตั้ง openssh-client ก่อน")
    monkeypatch.setattr(ssh, "push_file", boom)
    assert "ไม่พบคำสั่ง scp" in ssh.plan_install(node)[1]

    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: None)
    script, notice = ssh.plan_install(node)
    assert "ไม่ได้ติดตั้งจาก git checkout" in notice
    assert ssh.prepare_install(node) == script, "หน้าเว็บ (prepare_install) ต้องได้สคริปต์ตัวเดียวกับ CLI — ที่มีบรรทัดบอกเหตุผล"

    # ทางปกติ (ส่ง bundle ได้) และทางปักหมุด (จงใจไม่ส่ง) ไม่ใช่การถอย — ไม่มีอะไรต้องเตือน
    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: root)
    monkeypatch.setattr(ssh, "push_file", lambda *a, **k: ssh.Result(0, "", ""))
    assert ssh.plan_install(node)[1] == ""
    monkeypatch.setattr(ssh, "REPO_REF", "v0.8.0")
    assert ssh.plan_install(node)[1] == ""


def test_node_install_prints_the_fallback_reason_on_the_cli(tmp_path, monkeypatch):
    """`lmds node install` โชว์แค่ 6 บรรทัดท้ายของ stdout (และ --all ไม่โชว์เลย) — เหตุผลต้องมาถึงคนสั่งอยู่ดี"""
    from typer.testing import CliRunner

    from lmds.cli.main import app
    from lmds.nodes import add

    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: None)
    monkeypatch.setattr("lmds.fleet.consistency.hub_facts",
                        lambda: {"version": "0.9.0", "commit": "6e2b474", "template_hash": "aaaa1111", "dirty": []})
    add(Node(name="msi-4", host="10.0.0.4", user="ops"))
    sent = []

    def fake_ssh(target, port, wrapped, timeout, stdin_text=""):
        sent.append(wrapped)
        return ssh.Result(0, "\n".join(f"line {i}" for i in range(40)) + "\nlmds 0.9.0 (6e2b474)\n", "")
    monkeypatch.setattr(ssh, "_run_ssh", fake_ssh)
    monkeypatch.setattr("lmds.nodes.probe", lambda node: {
        "host": {"lmds_version": "0.9.0", "lmds_commit": "6e2b474"}, "models": []})

    runner = CliRunner(env={"COLUMNS": "300"})
    one = runner.invoke(app, ["node", "install", "msi-4"])
    assert one.exit_code == 0, one.output
    assert "ไม่ได้ติดตั้งจากโค้ดของ hub" in one.output and "ไม่ได้ติดตั้งจาก git checkout" in one.output, one.output
    assert any("git clone --depth 1" in wrapped for wrapped in sent), "ต้องเป็นสคริปต์ทาง GitHub จริง"

    every = runner.invoke(app, ["node", "install", "--all"])
    assert every.exit_code == 0, every.output
    assert "ไม่ได้ติดตั้งจากโค้ดของ hub" in every.output, every.output


def test_the_github_path_regenerates_stale_controllers_like_the_bundle_path(tmp_path, monkeypatch):
    """สองทางติดตั้งต้องจบที่สภาพเดียวกัน — เดิมทาง GitHub (hub ไม่มี checkout · ปักหมุด · ถอยมา) ข้าม
    `bundles refresh` → code ตรงแต่ controller ค้าง ทั้งที่ `lmds node install` สัญญาว่า regenerate ให้"""
    src = _source(tmp_path)
    monkeypatch.setattr(ssh, "REPO_URL", str(src))
    home = _node_home(tmp_path)
    calls = home / "lmds-calls.log"
    (home / ".local" / "bin" / "lmds").write_text(
        f"#!/bin/bash\necho \"lmds $*\" >> {calls}\n[ \"$1\" = bundles ] && exit 2\necho lmds-stub\n", encoding="utf-8")
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin"}

    done = subprocess.run(["bash", "-c", ssh.install_script()], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    said = calls.read_text(encoding="utf-8").splitlines()
    assert said == ["lmds version", "lmds bundles refresh --all --if-older"], said
    assert "regenerate controller ไม่สำเร็จ" in done.stdout, "lmds รุ่นเก่าไม่มีคำสั่ง = บอกแล้วไปต่อ ไม่ล้ม install"

    # รอบสอง = อัปเดต (git pull) — ทางเดียวกัน
    calls.write_text("", encoding="utf-8")
    again = subprocess.run(["bash", "-c", ssh.install_script()], capture_output=True, text=True, env=env)
    assert again.returncode == 0, again.stderr
    assert "lmds bundles refresh --all --if-older" in calls.read_text(encoding="utf-8")


def test_the_shipped_file_lands_in_the_users_home_not_in_shared_tmp(tmp_path, monkeypatch):
    """/tmp/lmds-src.bundle เป็นชื่อตายตัวในโฟลเดอร์ที่ทุก user บนเครื่องนั้นเขียนได้ แล้วถูก clone มาติดตั้ง:
    ไฟล์ค้างของ user อื่น (สคริปต์ตายก่อน rm) ทำให้ scp ล้มตลอดไป และไฟล์ที่วางรอไว้คือโค้ดที่จะถูกรัน
    → วางใต้ home ของ user ที่ ssh เข้าไป (scp ปลายทางแบบ path สัมพัทธ์) แล้วสคริปต์อ้างผ่าน $HOME"""
    assert not ssh.REMOTE_BUNDLE.startswith("/"), "path สัมพัทธ์ = scp วางใต้ home"
    src = _source(tmp_path)
    spaced = tmp_path / "home with space"
    spaced.mkdir()
    home = _node_home(spaced)
    landed = home / ssh.REMOTE_BUNDLE
    landed.write_bytes(_bundle_of(src, tmp_path, "h.bundle").read_bytes())    # สิ่งที่ scp ทำ

    monkeypatch.setattr("lmds.web.selfupdate.source_root", lambda: src)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    pushed = []
    monkeypatch.setattr(ssh, "push_file", lambda n, l, r, timeout=1800: pushed.append(r) or ssh.Result(0, "", ""))
    script = ssh.prepare_install(Node(name="n", host="h", user="u"))
    assert pushed == [ssh.REMOTE_BUNDLE]

    # รันจากโฟลเดอร์อื่น + HOME ที่มีช่องว่าง — path ต้องยังชี้ถูกหลัง `cd AutoDeployDGXProject`
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=str(tmp_path),
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert done.returncode == 0, done.stderr
    assert _git(home / "AutoDeployDGXProject", "rev-parse", "HEAD") == _git(src, "rev-parse", "HEAD")
    assert not landed.exists(), "ติดตั้งเสร็จต้องเก็บไฟล์ที่ส่งมา"
