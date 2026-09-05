"""ลบโมเดล stacked แล้วต้องเก็บกวาดบน worker ด้วย — ผู้ใช้ 2026-09-05: "ลบ stacked แล้วลบที่ worker ไหม"

เดิม `remove_server` หยุด (controller stop หยุด container ทั้ง head+worker) แล้วลบ bundle + weight ของ **head**
เท่านั้น · weight ที่ sync-worker คัดลอกไปทุก worker (75–173 GB) · container `lmds-<slug>-worker` · /tmp/lmds-<slug>
· FlashInfer cache ของ bundle ยังอยู่ทั้งหมด และไม่มีใครบอก

ssh ปลอมรันคำสั่งปลายทางในเครื่องนี้ในนาม FAKE_NODE=<ip> โดยเบี่ยง /wk/… และ /tmp/lmds-… ไปโฟลเดอร์ต่อ node ·
docker ปลอมจำลอง rm -f / inspect / images / run … rm -rf (root ในคอนเทนเนอร์) · du จริงของ Linux
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from lmds.fleet import manager
from tests.test_audit_stacked_controller import _bundle as _render_stacked, _shim

SAFE_PATH = "/usr/bin:/bin"
SLUG = "big-stacked"
MODEL = "org/Big"
CACHE = "models--org--Big"

_SSH = '''
host=""; cmd=""
while (( $# )); do
  case "$1" in
    -o) shift 2 ;;
    -*) shift ;;
    *) if [[ -z "$host" ]]; then host="$1"; shift; else cmd="$*"; break; fi ;;
  esac
done
node="${host#*@}"
echo "ssh ${host} :: ${cmd%%$'\\n'*}" >> "$FAKE_LOG"
for down in ${FAKE_SSH_DOWN:-}; do [[ "$node" == "$down" ]] && { echo "ssh: connect to host $node port 22: No route to host" >&2; exit 255; }; done
cmd="${cmd//\\/wk\\//${FAKE_REMOTE}/${node}/wk/}"
cmd="${cmd//\\/tmp\\/lmds-/${FAKE_REMOTE}/${node}/tmp/lmds-}"
export FAKE_NODE="$node"
exec bash -c "$cmd"
'''

_DOCKER = '''
node="${FAKE_NODE:-head}"
echo "docker[${node}] $*" >> "$FAKE_LOG"
case "$1" in
  rm) rm -f "${FAKE_REMOTE}/${node}/container" 2>/dev/null; exit 0 ;;
  inspect) [[ -e "${FAKE_REMOTE}/${node}/container" ]] && exit 0 || exit 1 ;;
  images) echo "vllm/vllm-openai:nightly"; echo "alpine:3.20"; exit 0 ;;
  run)
    # root ในคอนเทนเนอร์: -v <parent>:/x <img> rm -rf -- /x/<name> → คืนสิทธิ์แล้วลบจริง
    last="${@: -1}"; vol=""
    while (( $# )); do case "$1" in -v) vol="${2%%:*}"; shift 2 ;; *) shift ;; esac; done
    target="${vol}/$(basename "$last")"
    chmod -R u+rwx "$vol" 2>/dev/null; rm -rf -- "$target"; exit 0 ;;
  *) exit 0 ;;
esac
'''


def _cluster_bundle(tmp_path: Path, monkeypatch, workers: list[str], v2: bool = False,
                    image_id: str = "sha256:abcdef0123456789ffff") -> manager.ServerInfo:
    """bundle stacked บน head ปลอม + cluster.env + weight/container/cache บน worker ปลอมทุกตัว"""
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    bundle_dir = tmp_path / "bundles" / SLUG
    bundle_dir.mkdir(parents=True)
    controller = bundle_dir / f"{SLUG}-stacked.sh"
    controller.write_text('#!/bin/bash\necho "ctl $*" >> "$FAKE_LOG"\n', encoding="utf-8")
    controller.chmod(0o755)
    (bundle_dir / "MODEL_PROFILE.yaml").write_text(
        f"topology: stacked\nmodel:\n  id: {MODEL}\nruntime:\n  engine: vllm\n", encoding="utf-8")
    lines = [f"MASTER_IP=10.1.1.1", f"WORKER_IP={workers[0]}", f'WORKER_IPS="{" ".join(workers)}"',
             f"NNODES={len(workers) + 1}", "SSH_USER=neronain", "WORKER_HF_HOME=/wk/hf",
             "WORKER_FLASHINFER_CACHE=/wk/fi"]
    if v2:
        lines += ["CLUSTER_ENV_SCHEMA=2", f"CLUSTER_TOPOLOGY=switch-{len(workers) + 1}"]
        lines += [f"HEAD_TO_WORKER_IP_{i + 1}={ip}" for i, ip in enumerate(workers)]
        lines += [f"WORKER_HEAD_IP_{i + 1}=10.1.1.1" for i in range(len(workers))]
    (bundle_dir / "cluster.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if image_id:
        (home / ".cache" / "huggingface" / f".lmds-image-id-{SLUG}").write_text(image_id + "\n", encoding="utf-8")
    # weight บน head (เลย์เอาต์ hub/)
    head_weights = home / ".cache" / "huggingface" / "hub" / CACHE / "blobs"
    head_weights.mkdir(parents=True)
    (head_weights / "shard").write_bytes(b"h" * 500)

    remote = tmp_path / "remote"
    for ip in workers:
        _seed_worker(remote / ip)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, "ssh", _SSH)
    _shim(bin_dir, "docker", _DOCKER)
    monkeypatch.setenv("PATH", f"{bin_dir}:{SAFE_PATH}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setattr(manager, "have_systemctl", lambda: False)
    return manager.ServerInfo(slug=SLUG, controller=str(controller), engine="vllm", mode="docker",
                              model_id=MODEL, running=True)


def _seed_worker(node: Path) -> None:
    weights = node / "wk" / "hf" / "hub" / CACHE / "blobs"
    weights.mkdir(parents=True)
    (weights / "shard-1").write_bytes(b"w" * 4000)
    (node / "wk" / "hf" / "hub" / ".locks" / CACHE).mkdir(parents=True)
    (node / "wk" / "hf" / "hub" / ".locks" / CACHE / "x.lock").write_bytes(b"l" * 10)
    (node / "wk" / "fi" / "abcdef0123456789").mkdir(parents=True)
    (node / "wk" / "fi" / "abcdef0123456789" / "kernel.so").write_bytes(b"k" * 300)
    (node / "wk" / "fi" / "otherimage00000000").mkdir(parents=True)          # ของ bundle อื่น — ห้ามแตะ
    (node / "tmp" / f"lmds-{SLUG}").mkdir(parents=True)
    (node / "tmp" / f"lmds-{SLUG}" / "worker.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (node / "container").write_text("running", encoding="utf-8")


def _calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "calls.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _worker(tmp_path: Path, ip: str) -> Path:
    return tmp_path / "remote" / ip


# ═════════════════════ แผน (dry-run) ═════════════════════
@pytest.mark.parametrize("workers, v2", [(["10.1.1.2"], False), (["10.1.1.2", "10.1.1.3", "10.1.1.4"], True)])
def test_the_plan_lists_what_every_worker_holds_with_sizes_and_ips(tmp_path, monkeypatch, workers, v2):
    info = _cluster_bundle(tmp_path, monkeypatch, workers, v2=v2)
    items = manager.removal_plan(info)
    remote = [i for i in items if i.node]
    assert {i.node for i in remote} == set(workers)
    for ip in workers:
        mine = {i.label: i for i in remote if i.node == ip}
        assert f"container บน worker {ip}" in mine and mine[f"container บน worker {ip}"].kind == "container"
        weights = mine[f"weight ของโมเดล บน worker {ip}"]
        assert weights.size_bytes == 4000 and weights.is_weights and str(weights.path) == f"/wk/hf/hub/{CACHE}"
        assert mine[f"lock ของ HF cache บน worker {ip}"].size_bytes == 10
        assert str(mine[f"สคริปต์ worker บน worker {ip}"].path) == f"/tmp/lmds-{SLUG}"
        fi = mine[f"FlashInfer cache ของ bundle บน worker {ip}"]
        assert str(fi.path) == "/wk/fi/abcdef0123456789" and fi.size_bytes == 300
        assert all(i.ssh_user == "neronain" for i in mine.values())
    # ของ head ยังอยู่ครบ + image lock ของ bundle
    labels = [i.label for i in items if not i.node]
    assert "bundle" in labels and "weight ของโมเดล" in labels and "image lock ของ bundle" in labels
    # ssh หนึ่งครั้งต่อ worker (ถามขนาดทั้งชุดในคำสั่งเดียว) — ไม่ได้ลบอะไร
    calls = _calls(tmp_path)
    assert len([c for c in calls if c.startswith("ssh ")]) == len(workers)
    assert not any("rm -rf" in c and "docker" not in c for c in calls if not c.startswith("ssh"))
    assert _worker(tmp_path, workers[0]).joinpath("wk/hf/hub", CACHE).is_dir()


def test_keep_weights_leaves_the_weights_and_locks_on_every_worker(tmp_path, monkeypatch):
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2", "10.1.1.3"])
    plan = manager.removal_plan(info, include_weights=False)
    assert not [i for i in plan if i.is_weights]
    assert [i.label for i in plan if i.node == "10.1.1.3"] == [
        "container บน worker 10.1.1.3", "สคริปต์ worker บน worker 10.1.1.3", "FlashInfer cache ของ bundle บน worker 10.1.1.3"]

    lines = manager.remove_server(info, include_weights=False)
    assert not manager.removal_failed(lines), lines
    for ip in ("10.1.1.2", "10.1.1.3"):
        node = _worker(tmp_path, ip)
        assert (node / "wk/hf/hub" / CACHE / "blobs/shard-1").is_file() and (node / "wk/hf/hub/.locks" / CACHE).is_dir()
        assert not (node / "container").exists() and not (node / "tmp" / f"lmds-{SLUG}").exists()
        assert not (node / "wk/fi/abcdef0123456789").exists() and (node / "wk/fi/otherimage00000000").is_dir()
    assert (tmp_path / "home/.cache/huggingface/hub" / CACHE).is_dir()       # weight ของ head ก็ยังอยู่


def test_a_shared_image_lock_keeps_the_flashinfer_cache(tmp_path, monkeypatch):
    """bundle อื่นบน head ล็อก image เดียวกัน = cache JIT เป็นของร่วม ห้ามลบ"""
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2"])
    (tmp_path / "home/.cache/huggingface/.lmds-image-id-other").write_text("sha256:abcdef0123456789ffff\n")
    labels = [i.label for i in manager.removal_plan(info)]
    assert not any("FlashInfer" in l for l in labels)


# ═════════════════════ ลบจริง ═════════════════════
def test_remove_stops_first_then_head_then_every_worker_and_verifies(tmp_path, monkeypatch):
    workers = ["10.1.1.2", "10.1.1.3", "10.1.1.4"]
    info = _cluster_bundle(tmp_path, monkeypatch, workers, v2=True)
    lines = manager.remove_server(info)
    assert not manager.removal_failed(lines), lines

    # ลำดับ: หยุด (controller stop = head+worker) → head → worker ทีละเครื่องเรียง rank
    assert lines[0].startswith("หยุดเซิร์ฟเวอร์")
    head_end = max(i for i, l in enumerate(lines) if l.startswith("ลบ ") and ":" in l and "worker" not in l)
    worker_start = min(i for i, l in enumerate(lines) if "บน worker" in l or "10.1.1." in l)
    assert head_end < worker_start
    order = [ip for l in lines for ip in workers if f"{ip}:" in l]
    assert order and order == sorted(order, key=workers.index)
    calls = _calls(tmp_path)
    assert calls[0] == "ctl stop"
    assert calls.index("ctl stop") < min(i for i, c in enumerate(calls) if c.startswith("ssh "))

    assert not (tmp_path / "bundles" / SLUG).exists()
    assert not (tmp_path / "home/.cache/huggingface/hub" / CACHE).exists()
    assert not (tmp_path / "home/.cache/huggingface" / f".lmds-image-id-{SLUG}").exists()
    for ip in workers:
        node = _worker(tmp_path, ip)
        assert not (node / "wk/hf/hub" / CACHE).exists() and not (node / "wk/hf/hub/.locks" / CACHE).exists()
        assert not (node / "container").exists() and not (node / "tmp" / f"lmds-{SLUG}").exists()
        assert not (node / "wk/fi/abcdef0123456789").exists()
        assert (node / "wk/fi/otherimage00000000").is_dir()
        assert f"ลบ container บน worker {ip}: {ip}:lmds-{SLUG}-worker" in lines
        assert f"ลบ weight ของโมเดล บน worker {ip}: {ip}:/wk/hf/hub/{CACHE}" in lines
        assert f"docker[{ip}] rm -f lmds-{SLUG}-worker" in calls
    # ssh ต่อ worker: 1 ถามขนาด + 1 ลบ — ไม่ยิงทีละไฟล์
    assert len([c for c in calls if c.startswith("ssh ")]) == 2 * len(workers)


def test_an_unreachable_worker_is_reported_with_the_exact_commands_never_skipped(tmp_path, monkeypatch):
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2", "10.1.1.3"])
    monkeypatch.setenv("FAKE_SSH_DOWN", "10.1.1.3")
    plan = manager.removal_plan(info)
    down = [i for i in plan if i.node == "10.1.1.3"]
    assert down and all(not i.reachable for i in down)
    assert any("ติดต่อไม่ได้" in i.label for i in down)

    lines = manager.remove_server(info)
    failed = manager.removal_failed(lines)
    assert failed, lines
    left = next(l for l in failed if l.startswith("ยังเหลือบน 10.1.1.3:"))
    assert f"/wk/hf/hub/{CACHE}" in left and f"/tmp/lmds-{SLUG}" in left
    assert "ssh neronain@10.1.1.3" in left and f"docker rm -f lmds-{SLUG}-worker" in left and "sudo rm -rf" in left
    # เครื่องที่ถึงได้ยังถูกเก็บกวาดตามปกติ · head ก็ลบไปแล้ว
    assert not _worker(tmp_path, "10.1.1.2").joinpath("wk/hf/hub", CACHE).exists()
    assert _worker(tmp_path, "10.1.1.3").joinpath("wk/hf/hub", CACHE).is_dir()
    assert not (tmp_path / "bundles" / SLUG).exists()


def test_root_owned_files_on_a_worker_fall_back_to_docker_and_are_verified(tmp_path, monkeypatch):
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2"])
    blobs = _worker(tmp_path, "10.1.1.2") / "wk/hf/hub" / CACHE / "blobs"
    blobs.chmod(stat.S_IRUSR | stat.S_IXUSR)           # ลบข้างในไม่ได้ = อาการเดียวกับไฟล์ของ root
    try:
        lines = manager.remove_server(info)
    finally:
        if blobs.exists():
            blobs.chmod(0o700)
    assert not manager.removal_failed(lines), lines
    assert any(l.startswith("ลบผ่าน docker บน 10.1.1.2") and CACHE in l for l in lines), lines
    assert not (blobs.parent).exists()
    run = next(c for c in _calls(tmp_path) if c.startswith("docker[10.1.1.2] run"))
    # audit รอบ 2: image vLLM มี entrypoint เป็น vllm ไม่ใช่ shell — ต้อง --entrypoint rm ไม่งั้น rm -rf กลายเป็น argument ของ vllm
    assert "alpine:3.20" in run and "--entrypoint rm" in run and run.endswith(f"-rf -- /x/{CACHE}")


def test_a_stacked_bundle_without_cluster_env_says_so_instead_of_pretending(tmp_path, monkeypatch):
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2"])
    (tmp_path / "bundles" / SLUG / "cluster.env").unlink()
    assert not [i for i in manager.removal_plan(info) if i.node]
    lines = manager.remove_server(info)
    assert any("ไม่มี cluster.env" in l and "ต้องลบเอง" in l for l in lines), lines
    assert manager.removal_failed(lines)


def test_single_node_bundles_are_untouched_by_the_worker_logic(tmp_path, monkeypatch):
    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2"])
    single = tmp_path / "bundles" / "one" / "one-single.sh"
    single.parent.mkdir()
    single.write_text("#!/bin/bash\n", encoding="utf-8")
    plain = manager.ServerInfo(slug="one", controller=str(single), engine="vllm", mode="docker")
    assert manager.stacked_workers(plain) is None
    assert not [i for i in manager.removal_plan(plain) if i.node]
    assert not _calls(tmp_path)


# ═════════════════════ CLI --dry-run ═════════════════════
def test_cli_dry_run_shows_worker_items_with_ip_and_size(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from lmds.cli.main import app

    info = _cluster_bundle(tmp_path, monkeypatch, ["10.1.1.2", "10.1.1.3"], v2=True)
    monkeypatch.setattr("lmds.fleet.find", lambda slug: info if slug == SLUG else None)
    result = CliRunner().invoke(app, ["remove", SLUG, "--dry-run"], env={"COLUMNS": "220"})
    assert result.exit_code == 0, result.output
    assert "worker 10.1.1.2" in result.output and "worker 10.1.1.3" in result.output
    assert f"/wk/hf/hub/{CACHE}" in result.output and "3.9 KB" in result.output
    assert "ยังไม่ได้ลบอะไร" in result.output
    assert _worker(tmp_path, "10.1.1.2").joinpath("wk/hf/hub", CACHE).is_dir()


# ═════════════════════ controller `remove` (ทำเองจาก head โดยไม่มี lmds) ═════════════════════
def _controller_env(tmp_path: Path, bundle) -> dict:
    remote = tmp_path / "remote"
    for ip in ("10.1.1.2", "10.1.1.3"):
        _seed_worker(remote / ip)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, "ssh", _SSH)
    _shim(bin_dir, "docker", _DOCKER)
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    (home / ".cache/huggingface" / f".lmds-image-id-{bundle.directory.name}").write_text("sha256:abcdef0123456789ffff\n")
    (bundle.directory / "cluster.env").write_text(
        'MASTER_IP=10.1.1.1\nWORKER_IPS="10.1.1.2 10.1.1.3"\nSSH_USER=neronain\nWORKER_HF_HOME=/wk/hf\n'
        'WORKER_FLASHINFER_CACHE=/wk/fi\n', encoding="utf-8")
    return {"PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(home), "FAKE_LOG": str(tmp_path / "calls.log"),
            "FAKE_REMOTE": str(remote), "RUN_DIR": str(tmp_path / "run")}


def test_controller_remove_prints_the_worker_plan_and_only_deletes_with_y(tmp_path):
    bundle = _render_stacked(tmp_path)
    env = _controller_env(tmp_path, bundle)
    slug = bundle.directory.name
    cache = "models--nvidia--DeepSeek-V4-Flash-NVFP4"
    for ip in ("10.1.1.2", "10.1.1.3"):        # weight ของโมเดลใน bundle จริง
        node = tmp_path / "remote" / ip
        (node / "wk/hf/hub" / cache / "blobs").mkdir(parents=True)
        (node / "wk/hf/hub" / cache / "blobs" / "s").write_bytes(b"x" * 2048)
        (node / "tmp" / f"lmds-{slug}").mkdir(parents=True)
    plan = subprocess.run(["bash", str(bundle.controller), "remove"], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert "== worker 10.1.1.2 ==" in plan.stdout and "== worker 10.1.1.3 ==" in plan.stdout
    assert f"2048\t/wk/hf/hub/{cache}" in plan.stdout.replace(str(tmp_path / "remote" / "10.1.1.2"), "") \
        or f"/wk/hf/hub/{cache}" in plan.stdout
    assert f"container lmds-{slug}-worker" in plan.stdout and "ยังไม่ลบ" in plan.stdout and "remove -y" in plan.stdout
    assert (tmp_path / "remote/10.1.1.2/wk/hf/hub" / cache).is_dir()

    done = subprocess.run(["bash", str(bundle.controller), "remove", "-y"], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    for ip in ("10.1.1.2", "10.1.1.3"):
        node = tmp_path / "remote" / ip
        assert not (node / "wk/hf/hub" / cache).exists() and not (node / "tmp" / f"lmds-{slug}").exists()
        assert not (node / "container").exists() and not (node / "wk/fi/abcdef0123456789").exists()
        assert (node / "wk/fi/otherimage00000000").is_dir()
        assert f"ลบ container lmds-{slug}-worker" in done.stdout
    assert "ยังเหลือบน" not in done.stdout

    keep = subprocess.run(["bash", str(bundle.controller), "remove", "-y", "--keep-weights"], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert keep.returncode == 0 and cache not in keep.stdout

    down = subprocess.run(["bash", str(bundle.controller), "remove", "-y"], env={**env, "FAKE_SSH_DOWN": "10.1.1.3"},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert down.returncode == 0
    assert "ยังเหลือบน 10.1.1.3:" in down.stdout and "ssh neronain@10.1.1.3" in down.stdout and "sudo rm -rf" in down.stdout
