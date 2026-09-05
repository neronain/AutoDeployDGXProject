"""Audit 2026-09-04 — stacked (multi-node vLLM) controller ทั้ง lifecycle อย่างที่ลูกค้ารันจริง

ลูกค้ารายงาน "download fails" / "prepare-runtime fails" / "multi-node vLLM never comes up" /
"DeepSeek-V4-Flash-NVFP4 does not pass" · เทสทุกข้อรัน controller ที่ render แล้วจริง ๆ ใต้ bash
(`set -Eeuo pipefail`) กับ docker/ssh/rsync/ip/curl/df ปลอมบน PATH ที่บันทึก argv **ต่อ node**
(ssh ปลอมรันคำสั่งปลายทางในเครื่องนี้แล้วติดป้ายว่ามาจาก worker ไหน) — ไม่ใช่ grep template
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport, ShardFile

SAFE_PATH = "/usr/bin:/bin"
HEAD, WORKER = "10.1.1.1", "10.1.1.2"
SHARDS = [("model-00001-of-00002.safetensors", 12), ("model-00002-of-00002.safetensors", 7)]


# ───────────────────────── bundle ─────────────────────────
def _report(**overrides) -> ModelReport:
    base = dict(
        repo_id="nvidia/DeepSeek-V4-Flash-NVFP4", revision_sha="rev-ds4", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=157 * GIB, shard_count=2, context_length=131072,
        kv_dims=KvDims(layers=61, kv_heads=128, head_dim=128), license="mit", has_chat_template=True,
        safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in SHARDS],
    )
    base.update(overrides)
    return ModelReport(**base)


def _bundle(tmp_path, target="dgx-spark-stacked", report=None, tweak=None):
    report = report or _report()
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None)
    if tweak:
        tweak(plan)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


def _deepseek_recipe(plan):
    """สิ่งที่สูตร DeepSeek-V4 / NVFP4-on-GB10 ต้องส่งถึง engine — ทั้ง flag และ env"""
    plan.serving.extra_flags = ['--block-size 256', '--compilation-config {"cudagraph_mode":"PIECEWISE"}',
                               '--speculative-config {"method":"dspark","num_speculative_tokens":5}']
    plan.serving.extra_env = {"VLLM_NVFP4_GEMM_BACKEND": "marlin", "VLLM_USE_FLASHINFER_MOE_FP4": "0"}


# ───────────────────────── shims ─────────────────────────
def _shim(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)


# ssh ปลอม: ตัด option ออก จำ host แล้วรันคำสั่งปลายทางในเครื่องนี้ (FAKE_NODE บอก docker ปลอมว่า
# กำลังเล่นเป็น worker ไหน) · /tmp/lmds-<slug> ของ worker ถูกเบี่ยงเข้า FAKE_REMOTE
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
echo "ssh ${host} :: ${cmd}" >> "$FAKE_LOG"
if [[ -n "${FAKE_SSH_FAIL:-}" ]]; then echo "${FAKE_SSH_FAIL}" >&2; exit 255; fi
if [[ -n "${FAKE_SSH_NO_RSYNC:-}" && "$cmd" == *"command -v rsync"* ]]; then exit 1; fi
cmd="${cmd//\\/tmp\\/lmds-/${FAKE_REMOTE}/tmp/lmds-}"
export FAKE_NODE="$node"
exec bash -c "$cmd"
'''

# docker ปลอม: ตอบตามชนิดคำสั่ง · ค่าต่อ node ผ่าน FAKE_*_MAP="ip=ค่า ip=ค่า"
_DOCKER = '''
node="${FAKE_NODE:-head}"
echo "docker[${node}] $*" >> "$FAKE_LOG"
lookup() {  # $1 = map, $2 = default
  local pair
  for pair in $1; do [[ "${pair%%=*}" == "$node" ]] && { echo "${pair#*=}"; return; }; done
  echo "$2"
}
case "$1" in
  image) exit 0 ;;
  pull)
    fail="$(lookup "${FAKE_PULL_FAIL_MAP:-}" "")"
    if [[ -n "$fail" ]]; then echo "Error response from daemon: toomanyrequests: rate limit exceeded" >&2; exit 1; fi
    exit 0 ;;
  inspect)
    if [[ "$*" == *".Id"* ]]; then
      id="$(lookup "${FAKE_IMAGE_ID_MAP:-}" "${FAKE_IMAGE_ID:-sha256:aaaa1111}")"
      [[ "$id" == "none" ]] && exit 1
      echo "$id"
    elif [[ "$*" == *".State.Running"* ]]; then
      echo "$(lookup "${FAKE_RUNNING_MAP:-}" "${FAKE_RUNNING:-true}")"
    fi
    exit 0 ;;
  run)
    # check_architecture: ถาม registry ของ vLLM + transformers → ตอบตาม FAKE_ARCH_VERDICT (ว่าง = เงียบเหมือน image ที่ไม่มี python)
    if [[ "$*" == *"ModelRegistry"* ]]; then [[ -n "${FAKE_ARCH_VERDICT:-}" ]] && echo "$FAKE_ARCH_VERDICT"; exit 0; fi
    # verify-worker: python3 - อ่านสคริปต์จาก stdin · map /cache → โฟลเดอร์ที่ -v ชี้
    # docker จริงส่ง stdin ให้คอนเทนเนอร์เฉพาะเมื่อมี -i — ไม่มีก็ได้สคริปต์ว่าง (python3 จบ 0 เงียบ ๆ)
    if [[ "$*" == *"--entrypoint python3"* && "${@: -1}" == "-" ]]; then
      shift; vol=""; interactive=0
      while (( $# )); do
        case "$1" in
          -v) vol="${2%%:*}"; shift 2 ;;
          -e) export "$2"; shift 2 ;;
          -i) interactive=1; shift ;;
          *) shift ;;
        esac
      done
      if (( interactive )); then sed "s#/cache#${vol}#g" | python3 -; else python3 - </dev/null; fi
      exit $?
    fi
    echo "docker env[${node}]: VLLM_API_KEY=${VLLM_API_KEY:-unset}" >> "$FAKE_LOG"
    echo "cid-$RANDOM"; exit 0 ;;
  logs)
    # log ต่อ node อ่านจากไฟล์ FAKE_LOGS_DIR/<node>.log (map คั่นช่องว่างใส่ log จริงไม่ได้)
    if [[ -n "${FAKE_LOGS_DIR:-}" && -f "${FAKE_LOGS_DIR}/${node}.log" ]]; then cat "${FAKE_LOGS_DIR}/${node}.log"
    else printf '%s\\n' "${FAKE_LOGS:-}"; fi
    exit 0 ;;
  wait) echo "${FAKE_WAIT_CODE:-0}"; exit 0 ;;
  *) exit 0 ;;
esac
'''

# ip ปลอม: head มี 10.1.1.1 (management) + 10.200.0.1 (fabric) · worker 10.1.1.2 + 10.200.0.2
_IP = '''
if [[ "$1" == "-o" ]]; then
  echo "2: mgmt0    inet 10.1.1.1/24 brd 10.1.1.255 scope global mgmt0"
  echo "3: fabric0  inet 10.200.0.1/24 brd 10.200.0.255 scope global fabric0"
  echo "4: mgmt1    inet 10.1.1.2/24 brd 10.1.1.255 scope global mgmt1"
  echo "5: fabric1  inet 10.200.0.2/24 brd 10.200.0.255 scope global fabric1"
fi
exit 0
'''

_DF = '''
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "fake 1 1 ${FAKE_DF_KB:-999999999999} 1% /"
'''


def _bin(tmp_path: Path, **extra: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, "ssh", _SSH)
    _shim(bin_dir, "docker", _DOCKER)
    _shim(bin_dir, "ip", _IP)
    _shim(bin_dir, "df", _DF)
    _shim(bin_dir, "curl", "exit 0\n")
    _shim(bin_dir, "sudo", "exit 0\n")
    for name, body in extra.items():
        _shim(bin_dir, name, body)
    return bin_dir


def _run(bundle, cmd: list[str], tmp_path: Path, env: dict | None = None, cluster: bool = True,
         timeout: int = 90) -> subprocess.CompletedProcess:
    """รัน controller ทั้งสคริปต์แบบไม่มี tty (เหมือน hub สั่ง) พร้อม shim บน PATH"""
    bin_dir = tmp_path / "bin"
    remote = tmp_path / "remote"
    remote.mkdir(exist_ok=True)
    full = {
        "PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(tmp_path / "home"),
        "FAKE_LOG": str(tmp_path / "calls.log"), "FAKE_REMOTE": str(remote),
        "CLUSTER_ENV": "/nonexistent/cluster.env", "WORKER_INIT_WAIT": "0", "WORKER_CHECK_INTERVAL": "0",
        "RUN_DIR": str(tmp_path / "run"),
    }
    if cluster:
        full.update({"MASTER_IP": HEAD, "WORKER_IP": WORKER, "SSH_USER": "neronain"})
    full.update(env or {})
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run(["bash", str(bundle.controller), *cmd], env=full, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)


def _calls(tmp_path: Path) -> str:
    log = tmp_path / "calls.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _seed_head_cache(home: Path, model="nvidia/DeepSeek-V4-Flash-NVFP4", rev="rev-ds4", shards=SHARDS) -> Path:
    """snapshot ครบบน head (config.json + shard ขนาดตรง Hub) ในเลย์เอาต์ hub/ มาตรฐาน"""
    snap = home / ".cache/huggingface/hub" / f"models--{model.replace('/', '--')}" / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    for name, size in shards:
        (snap / name).write_bytes(b"x" * size)
    return snap


# ═════════════════════ 1. start: engine env + bundle.env/bundle.args ถึง head และ worker ทุกตัว ═════════════════════
def test_start_hands_engine_env_and_bundle_overrides_to_the_head_and_every_worker(tmp_path):
    """เคสลูกค้า: Qwen3-Coder-Next-NVFP4-GB10 / DeepSeek-V4 stacked ไม่ขึ้น — worker ที่ไม่ได้ env marlin
    ไป JIT cutlass FP4 แล้วตายบน SM121 ขณะที่ head ได้ env ครบ · ต้องได้เท่ากันทั้งสองทาง:
    extra_env ของสูตร (EXTRA_DOCKER_ENV) · ENGINE_ENV/TOOL_CALL_PARSER/VLLM_IMAGE จาก bundle.env ·
    extra_args จาก bundle.args"""
    bundle = _bundle(tmp_path, tweak=_deepseek_recipe)
    (bundle.directory / "bundle.env").write_text(
        'ENGINE_ENV="${ENGINE_ENV:-VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_MARLIN_USE_ATOMIC_ADD=1}"\n'
        'TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"\n'
        'VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:3dbe092e}"\n', encoding="utf-8")
    (bundle.directory / "bundle.args").write_text('--max-num-batched-tokens=4096\n', encoding="utf-8")
    _bin(tmp_path)
    _seed_head_cache(tmp_path / "home")

    done = _run(bundle, ["start"], tmp_path, env={"TRANSPORT_IP_MASTER": "10.200.0.1",
                                                  "TRANSPORT_IP_WORKER": "10.200.0.2", "API_KEY": "sekrit"})
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(tmp_path)
    head_run = next(l for l in calls.splitlines() if l.startswith("docker[head] run -d"))
    worker_run = next(l for l in calls.splitlines() if l.startswith(f"docker[{WORKER}] run -d"))
    worker_sh = (tmp_path / "remote" / "tmp" / f"lmds-{bundle.directory.name}" / "worker.sh").read_text(encoding="utf-8")

    # (1) env ของสูตร + ENGINE_ENV ถึงทั้งสองฝั่ง
    for pair in ("VLLM_NVFP4_GEMM_BACKEND=marlin", "VLLM_USE_FLASHINFER_MOE_FP4=0",
                 "VLLM_TEST_FORCE_FP8_MARLIN=1", "VLLM_MARLIN_USE_ATOMIC_ADD=1"):
        assert f"-e {pair}" in head_run, f"head ขาด {pair}: {head_run}"
        assert f"export {pair}" in worker_sh, f"worker.sh ขาด {pair}:\n{worker_sh}"
    # (2) image จาก bundle.env ตัวเดียวกันทุก node · parser ที่ override ถึง argv ทั้งคู่
    assert "vllm/vllm-openai@sha256:3dbe092e" in head_run and "vllm/vllm-openai@sha256:3dbe092e" in worker_run
    assert "--tool-call-parser hermes" in head_run and "--tool-call-parser hermes" in worker_sh
    assert "--tool-call-parser deepseek_v4" not in worker_sh
    # (3) flag ของสูตร + bundle.args ถึง argv ทั้งสองฝั่ง (JSON ไม่ถูกตัด)
    for flag in ("--block-size 256", '--compilation-config \\{\\"cudagraph_mode\\":\\"PIECEWISE\\"\\}',
                 "--max-num-batched-tokens=4096", "--kv-cache-dtype"):
        assert flag in worker_sh, f"worker.sh ขาด {flag}"
    assert "--block-size 256" in head_run and "--max-num-batched-tokens=4096" in head_run
    # worker คุย NCCL ด้วย transport IP ของมัน (สาย fabric) ไม่ใช่ IP ที่ head ssh ไปหา
    assert "export VLLM_HOST_IP=10.200.0.2" in worker_sh
    assert "export NCCL_SOCKET_IFNAME=fabric1" in worker_sh
    assert "-e VLLM_HOST_IP=10.200.0.1" in head_run and "-e NCCL_SOCKET_IFNAME=fabric0" in head_run
    # API key ไม่โผล่บน argv แต่ถึงคอนเทนเนอร์ผ่าน env
    assert "sekrit" not in head_run and "docker env[head]: VLLM_API_KEY=sekrit" in calls


def test_serve_args_shows_recipe_flags_bundle_args_and_engine_env_without_touching_docker(tmp_path):
    bundle = _bundle(tmp_path, tweak=_deepseek_recipe)
    (bundle.directory / "bundle.args").write_text('--speculative-config={"method":"mtp"}\n', encoding="utf-8")
    (bundle.directory / "bundle.env").write_text('ENGINE_ENV="${ENGINE_ENV:-VLLM_TEST_FORCE_FP8_MARLIN=1}"\n',
                                                 encoding="utf-8")
    _bin(tmp_path)
    done = _run(bundle, ["serve-args"], tmp_path)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    for token in ("--block-size", "256", '{"cudagraph_mode":"PIECEWISE"}', '--speculative-config={"method":"mtp"}',
                  "--kv-cache-dtype", "--nnodes", "--headless", "VLLM_NVFP4_GEMM_BACKEND=marlin",
                  "VLLM_TEST_FORCE_FP8_MARLIN=1"):
        assert token in lines, f"ขาด {token} ใน serve-args"
    assert "docker" not in _calls(tmp_path) and "ssh" not in _calls(tmp_path)


# ═════════════════════ 2. image ต้องตรงกันทุก node ไม่ใช่แค่ worker ตัวแรก ═════════════════════
def test_image_check_covers_every_worker_and_names_the_node_with_its_pull_command(tmp_path):
    """4 เครื่อง: worker ตัวท้ายไม่มี image — เดิม _assert_runtime_images ถาม ssh_worker (ตัวแรก) ตัวเดียว
    จึงผ่านแล้วไปตายตอน docker run บน 10.1.1.4 ด้วย 'Unable to find image' ระหว่างที่ตัวอื่นเปิดไปแล้ว"""
    bundle = _bundle(tmp_path, target="dgx-spark-stacked-4")
    _bin(tmp_path)
    workers = "10.1.1.2 10.1.1.3 10.1.1.4"
    missing = _run(bundle, ["doctor"], tmp_path, env={"WORKER_IPS": workers, "FAKE_IMAGE_ID_MAP": "10.1.1.4=none"})
    assert missing.returncode != 0
    assert "worker 10.1.1.4" in missing.stderr and "docker pull" in missing.stderr, missing.stderr
    assert "ssh neronain@10.1.1.4" in missing.stderr

    mismatch = _run(bundle, ["doctor"], tmp_path, env={"WORKER_IPS": workers,
                                                        "FAKE_IMAGE_ID_MAP": "10.1.1.3=sha256:bbbb2222"})
    assert mismatch.returncode != 0
    assert "10.1.1.3" in mismatch.stderr and "prepare-runtime" in mismatch.stderr, mismatch.stderr

    ok = _run(bundle, ["doctor"], tmp_path, env={"WORKER_IPS": workers})
    assert ok.returncode == 0, ok.stderr
    assert _calls(tmp_path).count(".Id") >= 3 * 3, "ต้องถาม image id ของ worker ทุกตัว"


def test_prepare_runtime_pulls_the_pinned_image_everywhere_and_reports_the_failing_node(tmp_path):
    bundle = _bundle(tmp_path, target="dgx-spark-stacked-4")
    _bin(tmp_path)
    workers = "10.1.1.2 10.1.1.3 10.1.1.4"
    image = "ghcr.io/anemll/dspark-vllm-gx10@sha256:a839484"

    failed = _run(bundle, ["prepare-runtime"], tmp_path,
                  env={"WORKER_IPS": workers, "VLLM_IMAGE": image, "FAKE_PULL_FAIL_MAP": "10.1.1.3=1"})
    assert failed.returncode != 0
    assert "worker 10.1.1.3" in failed.stderr, failed.stderr
    assert f"ssh neronain@10.1.1.3 docker pull '{image}'" in failed.stderr
    assert "ghcr.io" in failed.stderr and "docker login ghcr.io" in failed.stderr   # สาเหตุที่พบบ่อยของ registry นี้
    calls = _calls(tmp_path)
    assert f"docker[head] pull {image}" in calls and f"docker[10.1.1.2] pull {image}" in calls
    assert f"docker[10.1.1.4] pull" not in calls, "หยุดที่ตัวที่ล้ม ไม่วิ่งต่อเงียบ ๆ"

    head_fail = _run(bundle, ["prepare-runtime"], tmp_path,
                     env={"WORKER_IPS": workers, "VLLM_IMAGE": image, "FAKE_PULL_FAIL_MAP": "head=1"})
    assert head_fail.returncode != 0 and "head" in head_fail.stderr and "prepare-runtime" in head_fail.stderr

    (tmp_path / "calls.log").unlink()
    ok = _run(bundle, ["prepare-runtime"], tmp_path, env={"WORKER_IPS": workers, "VLLM_IMAGE": image})
    assert ok.returncode == 0, ok.stderr
    calls = _calls(tmp_path)
    for node in ("head", "10.1.1.2", "10.1.1.3", "10.1.1.4"):
        assert f"docker[{node}] pull {image}" in calls, f"{node} ไม่ได้ pull image ตัวเดียวกัน"
    assert ok.stdout.count("ตรงกับ head") == 3
    # idempotent: สั่งซ้ำผ่านเหมือนเดิม (lock ถูกเขียนทับด้วยค่าเดิม)
    assert _run(bundle, ["prepare-runtime"], tmp_path, env={"WORKER_IPS": workers, "VLLM_IMAGE": image}).returncode == 0


# ═════════════════════ 3. คลัสเตอร์ยังเป็นค่าตัวอย่าง / SSH head→worker ไม่ผ่าน ═════════════════════
@pytest.mark.parametrize("command", ["prepare-runtime", "sync-worker", "verify-worker", "start"])
def test_worker_commands_refuse_the_placeholder_cluster_without_a_tty(tmp_path, command):
    """hub สั่ง prepare-runtime แบบไม่มี tty โดยยังไม่มี cluster.env — เดิม ssh ไป 10.100.152.2 (เครื่องตัวอย่าง)
    แล้วตายหลัง 10 วิด้วย 'เช็ค SSH' ซึ่งพาไปแก้ผิดที่"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, [command], tmp_path, cluster=False)
    assert done.returncode != 0
    assert "lmds node cluster --write" in done.stderr and "MASTER_IP=" in done.stderr, done.stderr
    assert "10.100.152" in done.stderr
    assert "ssh" not in _calls(tmp_path), "ห้ามแตะเครื่องตัวอย่าง"


def test_cluster_env_or_explicit_ips_are_accepted_and_status_flags_the_placeholder(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    (bundle.directory / "cluster.env").write_text(f"MASTER_IP={HEAD}\nWORKER_IP={WORKER}\nSSH_USER=neronain\n",
                                                  encoding="utf-8")
    ok = _run(bundle, ["prepare-runtime"], tmp_path, cluster=False, env={"CLUSTER_ENV": str(bundle.directory / "cluster.env")})
    assert ok.returncode == 0, ok.stderr
    assert f"ssh neronain@{WORKER} :: true" in _calls(tmp_path)

    status = _run(bundle, ["status"], tmp_path, cluster=False)
    assert status.returncode == 0
    assert "ยังไม่ตั้งค่า" in status.stdout and "lmds node cluster --write" in status.stdout


def test_ssh_failure_explains_that_the_head_needs_its_own_key(tmp_path):
    """key ที่ hub ใช้เข้า head ไม่ใช่ key ของ head — 'Permission denied (publickey)' ดิบ ๆ ไม่บอกว่าต้องทำอะไรที่เครื่องไหน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, rsync="exit 0\n")
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["sync-worker"], tmp_path, env={"FAKE_SSH_FAIL": "neronain@10.1.1.2: Permission denied (publickey)."})
    assert done.returncode != 0
    assert "Permission denied (publickey)" in done.stderr
    assert "ssh-copy-id neronain@10.1.1.2" in done.stderr and "ssh-keygen" in done.stderr, done.stderr
    assert "rsync" not in _calls(tmp_path)


# ═════════════════════ 4. download: proxy/mirror · เลย์เอาต์ · ดิสก์ · สาเหตุที่ล้ม ═════════════════════
def _download_env(**extra):
    return {"FAKE_RUNNING": "false", **extra}


def test_download_forwards_proxy_and_mirror_env_and_writes_the_standard_hub_layout(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, ["download"], tmp_path, env=_download_env(
        HTTPS_PROXY="http://user:pw@proxy:3128", HF_ENDPOINT="https://hf-mirror.com", HF_TOKEN="hf_secret"))
    assert done.returncode == 0, done.stdout + done.stderr
    run_line = next(l for l in _calls(tmp_path).splitlines() if l.startswith("docker[head] run -d"))
    assert "-e HTTPS_PROXY " in run_line and "-e HF_ENDPOINT " in run_line and "-e HF_TOKEN " in run_line
    assert "pw@proxy" not in run_line and "hf_secret" not in run_line, "ค่าลับต้องไม่อยู่บน argv"
    assert "-e HTTP_PROXY " not in run_line, "ส่งเฉพาะที่ตั้งไว้"
    assert "CACHE_DIR=/cache/hub" in run_line, "โหลดใหม่ต้องลงเลย์เอาต์มาตรฐานเดียวกับ worker/single/clone"

    # ของเก่าที่ค้างไว้แบบแบนต้องโหลดต่อที่เดิม ไม่เริ่มใหม่ใน hub/
    (tmp_path / "home/.cache/huggingface/models--nvidia--DeepSeek-V4-Flash-NVFP4/snapshots").mkdir(parents=True)
    (tmp_path / "calls.log").unlink()
    again = _run(bundle, ["download"], tmp_path, env=_download_env())
    assert again.returncode == 0, again.stderr
    assert "CACHE_DIR=/cache " in next(l for l in _calls(tmp_path).splitlines() if "run -d" in l)


def test_download_refuses_when_the_disk_cannot_hold_what_is_left(tmp_path):
    bundle = _bundle(tmp_path)          # 157 GB
    _bin(tmp_path)
    done = _run(bundle, ["download"], tmp_path, env=_download_env(FAKE_DF_KB=str(50 * 1024 * 1024)))
    assert done.returncode != 0
    assert "50 GB" in done.stderr and "HF_HOME=/data/hf" in done.stderr, done.stderr
    assert "run -d" not in _calls(tmp_path), "ต้องหยุดก่อนเริ่มโหลด"
    ok = _run(bundle, ["download"], tmp_path, env=_download_env(FAKE_DF_KB=str(300 * 1024 * 1024)))
    assert ok.returncode == 0, ok.stderr


@pytest.mark.parametrize("log, expect, runs", [
    ("huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo", "HF_TOKEN", 1),
    ("requests.exceptions.ConnectionError: Max retries exceeded ... Name or service not known", "HTTPS_PROXY", 2),
    ("OSError: [Errno 28] No space left on device", "HF_HOME=/data/hf", 1),
    ("huggingface_hub.errors.RevisionNotFoundError: 404 Client Error", "revision", 1),
])
def test_download_failure_names_the_cause_and_skips_the_pointless_xet_retry(tmp_path, log, expect, runs):
    """เดิมทุกความล้มเหลว = 'ลองใหม่โดยปิด Xet' แล้ว 'ล้มเหลวแม้ปิด Xet แล้ว — ดูข้อความด้านบน'
    401 ของ gated repo จึงถูกลองสองรอบแล้วจบด้วยข้อความที่พาไปดู Xet"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, ["download"], tmp_path, env=_download_env(FAKE_WAIT_CODE="1", FAKE_LOGS=log))
    assert done.returncode != 0
    assert expect in done.stderr, done.stderr
    assert "download" in done.stderr.splitlines()[-1] or "download" in done.stderr[-300:], "ต้องบอกคำสั่งถัดไป"
    assert _calls(tmp_path).count("run -d") == runs


# ═════════════════════ 5. sync-worker / verify-worker ═════════════════════
_RSYNC = '''
echo "rsync $*" >> "$FAKE_LOG"
exit "${FAKE_RSYNC_RC:-0}"
'''


def test_sync_worker_uses_batch_ssh_checks_the_remote_rsync_and_explains_failures(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path, rsync=_RSYNC)
    _seed_head_cache(tmp_path / "home")

    ok = _run(bundle, ["sync-worker"], tmp_path)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    calls = _calls(tmp_path)
    rsync_line = next(l for l in calls.splitlines() if l.startswith("rsync "))
    assert "-e ssh -o BatchMode=yes -o ConnectTimeout=10" in rsync_line, rsync_line
    assert "--exclude=*.incomplete" in rsync_line and "--partial" in rsync_line
    assert f"neronain@{WORKER}:" in rsync_line and "/hub/models--nvidia--DeepSeek-V4-Flash-NVFP4/" in rsync_line
    assert f"ssh neronain@{WORKER} :: command -v rsync" in calls
    assert "verify-worker" in ok.stdout

    (tmp_path / "calls.log").unlink()
    no_rsync = _run(bundle, ["sync-worker"], tmp_path, env={"FAKE_SSH_NO_RSYNC": "1"})
    assert no_rsync.returncode != 0 and f"worker {WORKER} ไม่มี rsync" in no_rsync.stderr
    assert "apt-get install -y rsync" in no_rsync.stderr and "rsync " not in _calls(tmp_path).replace("command -v rsync", "")

    full = _run(bundle, ["sync-worker"], tmp_path, env={"FAKE_RSYNC_RC": "11"})
    assert full.returncode != 0
    assert f"worker {WORKER}" in full.stderr and "exit 11" in full.stderr and "sync-worker" in full.stderr, full.stderr


def test_sync_worker_verifies_the_head_first_and_checks_the_worker_disk(tmp_path):
    big = [("model-00001-of-00002.safetensors", 3_000_000), ("model-00002-of-00002.safetensors", 2_000_000)]
    bundle = _bundle(tmp_path, report=_report(safetensor_shards=[ShardFile(filename=n, size_bytes=s) for n, s in big]))
    _bin(tmp_path, rsync=_RSYNC)
    partial = _run(bundle, ["sync-worker"], tmp_path)          # ยังไม่ได้ download
    assert partial.returncode != 0 and "download" in partial.stderr
    assert "rsync " not in _calls(tmp_path)

    _seed_head_cache(tmp_path / "home", shards=big)
    # worker ถือ cache คนละ path (ว่างเปล่า) และ df ปลอม (ตอบผ่าน ssh ปลอมด้วย) เหลือ 1 KB
    tiny = _run(bundle, ["sync-worker"], tmp_path,
                env={"FAKE_DF_KB": "1024", "WORKER_HF_HOME": str(tmp_path / "worker-hf")})
    assert tiny.returncode != 0
    assert f"worker {WORKER}" in tiny.stderr and "WORKER_HF_HOME=/data/hf" in tiny.stderr, tiny.stderr
    assert "1 MB" in tiny.stderr and "5 MB" in tiny.stderr, tiny.stderr
    assert not [l for l in _calls(tmp_path).splitlines() if l.startswith("rsync ")], "ต้องหยุดก่อนเริ่มลากไฟล์"


def test_verify_worker_checks_every_shard_size_not_just_the_count(tmp_path):
    """rsync --partial ทิ้งไฟล์ครึ่งเดียวไว้ชื่อเดิม — นับจำนวนผ่าน แล้ว start ไปตายที่ safetensors header
    บน worker ด้วย 'unexpected end of file' ที่ไม่บอกว่าไฟล์ไหน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    worker_home = tmp_path / "worker-home"
    snap = _seed_head_cache(worker_home, shards=[(SHARDS[0][0], 12), (SHARDS[1][0], 3)])   # ตัวที่สองขาด 4 ไบต์
    env = {"WORKER_HF_HOME": str(worker_home / ".cache/huggingface")}

    short = _run(bundle, ["verify-worker"], tmp_path, env=env)
    assert short.returncode != 0, short.stdout
    assert "model-00002-of-00002.safetensors: size 3 != 7" in short.stdout, short.stdout + short.stderr
    assert "sync-worker" in short.stderr

    (snap / SHARDS[1][0]).write_bytes(b"x" * 7)
    ok = _run(bundle, ["verify-worker"], tmp_path, env=env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    # บรรทัดจาก python บน worker ต้องมา — บนเครื่องจริง (spark-head 2026-09-04) controller เก่าพิมพ์ PASS
    # ใน 1 วิ โดยไม่มี "worker shards:" เพราะ docker run ไม่มี -i → python3 ได้สคริปต์ว่างแล้วจบ 0
    assert "worker shards: 2" in ok.stdout and "verify-worker: PASS" in ok.stdout


def test_status_warns_when_the_port_answers_for_another_model(tmp_path):
    """เคสจริง spark-head 2026-09-04: status ของ bundle stacked บอก 'API: healthy' ทั้งที่ container ของมัน
    ไม่มีสักตัว — พอร์ต 8000 เป็นของ qwen3-coder-next (single) ที่รันอยู่ก่อน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl='''
        case "$*" in *"/v1/models"*) echo '{"object":"list","data":[{"id":"qwen3-coder-next-nvfp4-gb10","object":"model"}]}' ;; esac
        exit 0
    ''')
    done = _run(bundle, ["status"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert "API: healthy" in done.stdout
    assert "qwen3-coder-next-nvfp4-gb10" in done.stdout and "ไม่ใช่" in done.stdout, done.stdout
    assert "--port" in done.stdout


# ═════════════════════ 6. worker ตาย → คำอธิบายเดียวกับ head · NCCL / ds_mla ═════════════════════
def test_a_worker_that_dies_before_the_head_starts_is_explained_not_just_dumped(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    _seed_head_cache(tmp_path / "home")
    nccl = ("(EngineCore pid=1) torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1234, "
            "unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.27.3\n"
            "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / f"{WORKER}.log").write_text(nccl, encoding="utf-8")
    done = _run(bundle, ["start"], tmp_path, env={"FAKE_RUNNING_MAP": f"{WORKER}=false", "FAKE_LOGS_DIR": str(logs_dir)})
    assert done.returncode != 0
    assert f"worker container บน {WORKER} หยุดก่อน head" in done.stderr, done.stderr
    assert f"สาเหตุจาก log (worker {WORKER})" in done.stderr and "DistBackendError" in done.stderr
    assert "NCCL_SOCKET_IFNAME" in done.stderr and "logs worker" in done.stderr
    assert "docker[head] run -d" not in _calls(tmp_path), "head ต้องไม่ถูกเปิดเมื่อ worker ตายไปแล้ว"


def _explain_crash_fn(text: str) -> str:
    start = text.index("\nexplain_crash() {") + 1
    return text[start:text.index("\n}\n", start) + 3]


@pytest.mark.parametrize("log, hint", [
    ("AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto", "KV_CACHE_DTYPE=fp8"),
    ("RuntimeError: NCCL error in: ../csrc/…, unhandled system error", "network-info"),
    ("torch.distributed.DistStoreError: Socket Timeout", "MASTER_PORT"),
    ("torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 1/2 clients joined.", "nc -zv"),
    ("RuntimeError: concat_and_cache_mla, /workspace/csrc/libtorch_stable/cache_kernels.cu:928, pe_dim must be 64 for fp8_ds_mla", "vllm/vllm-openai:nightly"),
    ("RuntimeError: ptxas fatal: cvt with .e2m1x2 not supported on .target sm_121", "VLLM_NVFP4_GEMM_BACKEND=marlin"),
])
def test_explain_crash_knows_the_stacked_and_deepseek_failure_modes(tmp_path, log, hint):
    text = _bundle(tmp_path).controller.read_text(encoding="utf-8")
    script = ("set -Eeuo pipefail\nSLUG=ds MASTER_PORT=25000\n" + _explain_crash_fn(text)
              + "\nCRASH_LOG_TEXT=\"$FAKE\" CONTAINER_NAME=x explain_crash\necho reached\n")
    done = subprocess.run(["bash", "-c", script], env={"PATH": SAFE_PATH, "FAKE": "INFO boot\n" + log},
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0 and "reached" in done.stdout, done.stderr
    assert "สาเหตุจาก log" in done.stderr and hint in done.stderr, done.stderr


# ═════════════════════ 7. autostart unit · clone · เอกสารในตัว ═════════════════════
def test_autostart_unit_waits_at_least_as_long_as_the_controller_health_timeout(tmp_path):
    """systemd เคยฆ่า `start` ที่ 1800 วิ ระหว่าง stacked 157-220 GB ยังโหลด (controller เองรอ 5001-6906 วิ)"""
    from lmds.fleet.manager import ServerInfo, render_unit

    bundle = _bundle(tmp_path)
    info = ServerInfo(slug="ds", model="m", engine="vllm", mode="docker", port=8000, container="c",
                      controller=str(bundle.controller))
    unit = render_unit(info, timeout=1800)
    line = next(l for l in unit.splitlines() if l.startswith("TimeoutStartSec="))
    seconds = int(line.split("=")[1])
    text = bundle.controller.read_text(encoding="utf-8")
    default = int(text.split('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-')[1].split("}")[0])
    assert default > 1800 and seconds >= default + 300, (line, default)

    # bundle.env ที่ตั้ง STARTUP_TIMEOUT เองชนะ default ในสคริปต์
    (bundle.directory / "bundle.env").write_text('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-9000}"\n', encoding="utf-8")
    assert "TimeoutStartSec=9300" in render_unit(info, timeout=1800)
    # ค่าที่ขอมาใหญ่กว่าอยู่แล้วไม่ถูกลด
    assert "TimeoutStartSec=20000" in render_unit(info, timeout=20000)


class _Node:
    def __init__(self, name, host, cluster_ip=""):
        self.name, self.host, self.user, self.site = name, host, "neronain", "TKC"
        self.cluster_ip, self.port, self.all_hosts = cluster_ip, 22, [host]


class _Result:
    def __init__(self, done):
        self.ok, self.stdout, self.stderr = done.returncode == 0, done.stdout, done.stderr


def test_clone_finds_weights_in_the_flat_hf_layout_and_leaves_cluster_env_behind(tmp_path, monkeypatch):
    """head ของ stacked ที่โหลดก่อน 0.6.0 ถือ $HF_HOME/models--X (แบน) — clone มองแค่ hub/ แล้วตอบ
    'ยังไม่มีไฟล์โมเดล' ทั้งที่ 170 GB อยู่บนเครื่อง · และ cluster.env ของคู่เก่าต้องไม่ติดไปคู่ใหม่"""
    from lmds.fleet.clone import build_rsync_command, inspect_source, plan_clone

    nodes = {"a": _Node("a", "100.1.1.1", "10.100.152.1"), "b": _Node("b", "100.1.1.2", "10.100.152.3")}
    monkeypatch.setattr("lmds.nodes.find", lambda n: nodes.get(n), raising=False)
    monkeypatch.setattr("lmds.nodes.run", lambda node, script, timeout=120: _Result(
        subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=timeout)),
        raising=False)
    home = tmp_path / "home"
    bundle = tmp_path / "bundles" / "ds"; bundle.mkdir(parents=True)
    (bundle / "ds-stacked.sh").write_text(
        '#!/bin/bash\nMODEL_ID="nvidia/DeepSeek-V4-Flash-NVFP4"\nHF_HOME="${HF_HOME:-' + str(home) + '/.cache/huggingface}"\n')
    flat = home / ".cache/huggingface/models--nvidia--DeepSeek-V4-Flash-NVFP4"
    (flat / "blobs").mkdir(parents=True); (flat / "snapshots/r").mkdir(parents=True)
    (flat / "blobs/sha-1").write_bytes(b"x" * 500)

    plan = inspect_source(plan_clone("ds", "a", "b"))
    assert plan.model_dir == str(flat) and plan.files == [("blobs/sha-1", 500)]
    plan.bundle_dir = str(bundle)
    assert "--exclude=cluster.env" in build_rsync_command(plan, "neronain")


def test_usage_and_readme_point_at_cluster_env_and_the_head_key(tmp_path):
    """usage เคยพิมพ์ '{{ slug }}' ดิบ ๆ (heredoc อยู่ใน raw) และทั้ง usage/README บอกให้แก้ CONFIG ในสคริปต์
    ทั้งที่ hub เขียน cluster.env ให้ได้ตั้งแต่ 0.5"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    usage = _run(bundle, ["help"], tmp_path, cluster=False)
    assert usage.returncode == 0
    assert "{{ slug }}" not in usage.stdout and usage.stdout.startswith(bundle.directory.name)
    assert "lmds node cluster --write" in usage.stdout and "serve-args" in usage.stdout
    assert "ssh-copy-id" in usage.stdout
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "lmds node cluster --write" in readme and "ssh-copy-id" in readme
    assert "แก้ค่าใน CONFIG" not in readme


# ═════════════════════ docker pull บอกสาเหตุจริง + ลองซ้ำเมื่อสายหลุด ═════════════════════
_FLAKY_DOCKER = """
node="${FAKE_NODE:-head}"
echo "docker[${node}] $*" >> "$FAKE_LOG"
case "$1" in
  pull)
    n=$(( $(cat "${FAKE_PULL_COUNT_FILE}" 2>/dev/null || echo 0) + 1 )); echo "$n" > "${FAKE_PULL_COUNT_FILE}"
    if (( n <= ${FAKE_PULL_FLAKES:-2} )); then echo "unexpected EOF" >&2; exit 1; fi
    if [[ -n "${FAKE_PULL_FINAL_ERROR:-}" ]]; then echo "${FAKE_PULL_FINAL_ERROR}" >&2; exit 1; fi
    exit 0 ;;
  info) echo "/var/lib/docker"; exit 0 ;;
  inspect) echo "sha256:aaaa1111"; exit 0 ;;
  *) exit 0 ;;
esac
"""


def test_prepare_runtime_retries_a_dropped_pull_and_names_the_real_cause(tmp_path):
    """เคสจริง 2026-09-05 (ลูกค้า cynbangkok · DeepSeek-V4-Flash stacked): "ดึง image … ไม่สำเร็จ" ทั้งที่ digest
    ยังอยู่บน ghcr.io — เหตุผลจริงของ docker อยู่เหนือคำแนะนำแล้วเลื่อนหาย · ตอนนี้ pull ซ้ำได้ 3 รอบ (สายหลุด
    = unexpected EOF มักผ่านรอบสอง) และเมื่อล้มจริงพิมพ์บรรทัดของ docker + สาเหตุที่แปลแล้ว
    (ออกเน็ตไม่ได้ / rate limit / ไม่มีสิทธิ์ / tag หาย / ดิสก์เต็ม) · worker ใช้ helper เดียวกันผ่าน ssh"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, docker=_FLAKY_DOCKER)
    count = tmp_path / "pulls"
    image = "ghcr.io/anemll/dspark-vllm-gx10@sha256:a839484"
    base = {"WORKER_IPS": WORKER, "VLLM_IMAGE": image, "FAKE_PULL_COUNT_FILE": str(count),
            "DOCKER_PULL_RETRY_WAIT": "0"}

    # หลุด 2 ครั้งแล้วผ่าน (head) · worker ก็ผ่านรอบแรก (นับต่อจาก head เพราะไฟล์นับเดียวกัน)
    ok = _run(bundle, ["prepare-runtime"], tmp_path, env=base)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert ok.stderr.count("ลองใหม่ใน 0 วิ") == 2 and "unexpected EOF" in ok.stderr
    assert _calls(tmp_path).count(f"docker[head] pull {image}") == 3
    assert f"docker[{WORKER}] pull {image}" in _calls(tmp_path)

    # หลุดทุกรอบ = ล้มพร้อมสาเหตุ "สายหลุด" ไม่ใช่แค่ "ไม่สำเร็จ"
    count.unlink()
    dead = _run(bundle, ["prepare-runtime"], tmp_path, env={**base, "FAKE_PULL_FLAKES": "99"})
    assert dead.returncode != 0
    assert "docker บอกว่า: unexpected EOF" in dead.stderr and "สายหลุดระหว่างโหลด" in dead.stderr, dead.stderr
    assert _calls(tmp_path).count(f"docker[head] pull {image}") >= 3

    # ออกเน็ตไม่ได้: docker พิมพ์ no such host → บอกว่าเป็น DNS/เน็ต ไม่ใช่ image หาย · ไม่ลองซ้ำเกินจำเป็น
    count.unlink(); (tmp_path / "calls.log").unlink()
    offline = _run(bundle, ["prepare-runtime"], tmp_path,
                   env={**base, "FAKE_PULL_FLAKES": "0",
                        "FAKE_PULL_FINAL_ERROR": "Error response from daemon: Get \"https://ghcr.io/v2/\": dial tcp: lookup ghcr.io: no such host"})
    assert offline.returncode != 0
    assert "ออกเน็ตไปหา registry ไม่ได้" in offline.stderr and "ไม่ใช่ image หาย" in offline.stderr, offline.stderr

    # tag หาย = ไม่ลองซ้ำ (ซ้ำก็เท่าเดิม) + ชี้ไป lmds set --image
    count.unlink(); (tmp_path / "calls.log").unlink()
    gone = _run(bundle, ["prepare-runtime"], tmp_path,
                env={**base, "FAKE_PULL_FLAKES": "0", "FAKE_PULL_FINAL_ERROR": "manifest unknown"})
    assert gone.returncode != 0 and "ไม่มีบน registry แล้ว" in gone.stderr
    assert _calls(tmp_path).count(f"docker[head] pull {image}") == 1


def test_pull_warns_when_the_docker_disk_is_nearly_full_and_when_the_registry_is_unreachable(tmp_path):
    bundle = _bundle(tmp_path)
    _bin(tmp_path, docker=_FLAKY_DOCKER, curl="echo 000; echo 'curl: (6) Could not resolve host: ghcr.io' >&2; exit 6\n")
    image = "ghcr.io/anemll/dspark-vllm-gx10@sha256:a839484"
    done = _run(bundle, ["prepare-runtime"], tmp_path,
                env={"WORKER_IPS": WORKER, "VLLM_IMAGE": image, "FAKE_PULL_COUNT_FILE": str(tmp_path / "pulls"),
                     "FAKE_PULL_FLAKES": "0", "DOCKER_PULL_RETRY_WAIT": "0", "FAKE_DF_KB": str(5 * 1024 * 1024)})
    assert done.returncode == 0, done.stderr
    assert "เหลือ 5 GB" in done.stderr and "docker system prune" in done.stderr
    assert "ไปไม่ถึง https://ghcr.io/v2/" in done.stderr and "Could not resolve host" in done.stderr


# ═════════════════════ head ค้างก่อนโหลด weight = จับมือข้าม node ไม่ติด ═════════════════════
def test_start_names_a_stuck_rendezvous_before_the_health_timeout(tmp_path):
    """เคสจริง 2026-09-05 ลูกค้า (cynbangkok): head รอ /health 8 นาที หน่วยความจำ 7/122 GB (ยังไม่โหลด weight)
    แล้วอีก 20 นาทีล้มโดยไม่มีอะไรบอกว่าค้างที่ไหน · ตอนนี้หลัง STUCK_HINT_AFTER ถ้า log head ยังไม่มีบรรทัดโหลด weight
    → พิมพ์ท้าย log ทั้งสองฝั่ง ping สองทางบนสายเร็ว พอร์ต master และลำดับที่ต้องเช็ค · โหลดอยู่จริง = เงียบ"""
    bundle = _bundle(tmp_path)
    _seed_head_cache(tmp_path / "home")
    logs_dir = tmp_path / "logs"; logs_dir.mkdir()
    (logs_dir / "head.log").write_text("INFO vllm boot\nINFO Waiting for 1 more node(s) at tcp://10.200.0.1:25000\n", encoding="utf-8")
    (logs_dir / f"{WORKER}.log").write_text("INFO worker boot\nINFO connecting to master 10.200.0.1:25000 …\n", encoding="utf-8")
    _bin(tmp_path, curl="exit 7\n",                                             # /health ไม่เคยผ่าน
         ping='echo "ping $*" >> "$FAKE_LOG"; [[ "$*" == *10.200.0.2* ]] && exit 1; exit 0\n',   # head → worker ไม่ถึง
         ss='echo "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:*"\n')                      # master port ยังไม่เปิด
    env = {"TRANSPORT_IP_MASTER": "10.200.0.1", "TRANSPORT_IP_WORKER": "10.200.0.2", "FAKE_LOGS_DIR": str(logs_dir),
           "STARTUP_TIMEOUT": "2", "STUCK_HINT_AFTER": "0", "WORKER_CHECK_INTERVAL": "0"}
    done = _run(bundle, ["start"], tmp_path, env=env, timeout=120)
    assert done.returncode != 0
    err = done.stderr
    assert "head ยังไม่เริ่มโหลด weight" in err and "จับมือข้าม node" in err, err
    assert "Waiting for 1 more node(s)" in err and "connecting to master" in err, "ต้องเห็นท้าย log ของทั้ง head และ worker"
    assert "✕ head → worker 10.200.0.2 ping ไม่ถึง" in err and f"✓ worker {WORKER} → head 10.200.0.1 ping ถึง" in err, err
    assert "ยังไม่เปิดพอร์ต master 25000" in err and "ufw status" in err and "NCCL_DEBUG=INFO" in err
    assert err.count("head ยังไม่เริ่มโหลด weight") == 1, "เตือนครั้งเดียว ไม่ซ้ำทุกรอบ"
    assert "ไม่ health ภายใน 2s" in err

    # โหลด weight อยู่จริง (ช้าเพราะโมเดลใหญ่) = ไม่ใช่ค้าง ห้ามเตือน
    (logs_dir / "head.log").write_text("INFO Loading safetensors checkpoint shards:  40% 7/17\n", encoding="utf-8")
    (tmp_path / "calls.log").unlink()
    slow = _run(bundle, ["start"], tmp_path, env=env, timeout=120)
    assert slow.returncode != 0 and "head ยังไม่เริ่มโหลด weight" not in slow.stderr


def test_architecture_check_trusts_vllm_registry_when_transformers_is_too_old(tmp_path):
    """เคสจริง 2026-09-05 spark-head: image `vllm/vllm-openai:glm53-flash-arm64-cu130` ของ Red Hat (vLLM สาขาพิเศษ · transformers
    5.15.1) รู้จัก Glm5NextForConditionalGeneration ใน registry ของ vLLM เอง แต่ CONFIG_MAPPING_NAMES ของ transformers ไม่มี
    glm5_next → check_architecture เดิมหยุด "โมเดลใหม่กว่ารันไทม์" ทั้งที่ image นี้คือตัวเดียวที่รันโมเดลได้ · ตอนนี้ถาม
    ModelRegistry ของ vLLM ก่อน (ชื่อ architecture) แล้วค่อย transformers (model_type) · ไม่รู้จักทั้งคู่ = หยุดพร้อมบอกทั้งสองรุ่น"""
    import json

    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    snap = _seed_head_cache(tmp_path / "home")
    (snap / "config.json").write_text(json.dumps({"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"]}), encoding="utf-8")
    env = {"TRANSPORT_IP_MASTER": "10.200.0.1", "TRANSPORT_IP_WORKER": "10.200.0.2", "STARTUP_TIMEOUT": "2", "WORKER_CHECK_INTERVAL": "0"}
    known = _run(bundle, ["start"], tmp_path, env={**env, "FAKE_ARCH_VERDICT": "KNOWN vllm 0.1.dev20051+g487ecf187 (transformers 5.15.1)"}, timeout=120)
    assert "architecture: glm5_next — รองรับ (vllm 0.1.dev20051" in known.stdout, known.stdout + known.stderr
    assert "docker[head] run -d" in _calls(tmp_path), "ต้องเดินต่อไปถึงการเปิด container"
    assert "glm5_next Glm5NextForConditionalGeneration" in _calls(tmp_path), "ต้องส่งทั้ง model_type และชื่อ architecture ให้ probe"
    (tmp_path / "calls.log").unlink()
    unknown = _run(bundle, ["start"], tmp_path, env={**env, "FAKE_ARCH_VERDICT": "UNKNOWN vllm 0.28.0 transformers 5.14.0"}, timeout=120)
    assert unknown.returncode != 0 and "ไม่รู้จักสถาปัตยกรรม 'glm5_next' (vllm 0.28.0 transformers 5.14.0)" in unknown.stderr, unknown.stderr
    assert "lmds set " + bundle.directory.name + " --image" in unknown.stderr
    assert "docker[head] run -d" not in _calls(tmp_path)
