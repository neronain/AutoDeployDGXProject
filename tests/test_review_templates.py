"""Regression tests for the 2026-09-04 controller-template review (findings 1–12)

รัน controller ที่ render แล้วจริง ๆ ใต้ bash (ไม่ใช่แค่ grep template) ตามแนวของ harness ที่ผู้ review ใช้
พิสูจน์แต่ละข้อ — ฟังก์ชันที่แตะ docker/ssh/เน็ต ถูกดึงออกมารันเดี่ยว ๆ กับ stub ที่บันทึก argv
"""

from __future__ import annotations

import os
import pty
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle, renderer
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

TEMPLATES = Path(renderer.__file__).parent / "templates"
SAFE_PATH = "/usr/bin:/bin"


# ───────────────────────── helpers ─────────────────────────
def _gguf_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        gguf_variants=[GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=5 * GIB, sha256="a" * 64)],
        has_chat_template=True, license="apache-2.0",
    )
    base.update(overrides)
    return ModelReport(**base)


def _safetensors_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned-123", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, shard_count=17, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def _plan(report, target="dgx-spark-single", engine=None):
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None, engine=engine)
    return plan, fit


def _bundle(tmp_path, report, target="dgx-spark-single", engine=None, tweak=None):
    plan, fit = _plan(report, target, engine)
    if tweak:
        tweak(plan)
    return render_bundle(plan, report, fit, tmp_path)


def _stacked_bundle(tmp_path, tweak=None):
    report = _safetensors_report(repo_id="nvidia/Big-Model", weight_bytes=180 * GIB, shard_count=40,
                                 context_length=131072)
    return _bundle(tmp_path, report, target="dgx-spark-stacked", tweak=tweak)


def extract_fn(text: str, name: str) -> str:
    """ดึงฟังก์ชัน bash ออกมาทั้งก้อน — รองรับทั้งแบบบรรทัดเดียว `f() { …; }` และหลายบรรทัดที่ปิดด้วย `}`"""
    start = text.index(f"\n{name}() {{") + 1
    line_end = text.index("\n", start)
    if text[start:line_end].rstrip().endswith("}"):
        return text[start:line_end + 1]
    return text[start:text.index("\n}\n", start) + 3]


def run_bash(script: str, env: dict | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], env={"PATH": SAFE_PATH, **(env or {})},
                          capture_output=True, text=True, timeout=timeout)


def _fake_bin(tmp_path: Path, **scripts: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)
    return bin_dir


# ───────────────────────── 1: llama.cpp extra args + --jinja regardless of chat-template detection ─────────────────────────
@pytest.mark.parametrize("has_chat_template", [True, False, None])
def test_llamacpp_forwards_extra_args_and_jinja_whatever_the_inspector_said(tmp_path, has_chat_template):
    """inspector ตอบ None/False ให้ GGUF หลายตัว — เดิมทำให้ bundle.args, --extra-args และ --jinja หายทั้งชุด"""
    bundle = _bundle(tmp_path, _gguf_report(has_chat_template=has_chat_template))
    (bundle.directory / "bundle.args").write_text("--flash-attn on\n", encoding="utf-8")
    out = subprocess.run(["bash", str(bundle.controller), "serve-args", "--extra-args", "--threads 8"],
                         env={**os.environ, "RUN_DIR": str(tmp_path / "run")},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    argv = out.stdout.splitlines()
    assert "--jinja" in argv
    assert "--threads" in argv and "8" in argv          # --extra-args ชนะ bundle.args
    assert "--flash-attn" not in argv


# ───────────────────────── 2: explain_crash must not abort the script under set -e ─────────────────────────
def _explain_crash_script(controller_text: str) -> str:
    return "set -Eeuo pipefail\n" + extract_fn(controller_text, "explain_crash")


@pytest.mark.parametrize("kind", ["vllm", "sglang", "stacked"])
def test_explain_crash_survives_a_log_without_an_exception_line(tmp_path, kind):
    """log ที่ไม่มีบรรทัด XxxError (OOM-kill, image พัง) — grep คืน 1 → set -e เคยฆ่าสคริปต์ก่อน die() พิมพ์อะไร"""
    if kind == "stacked":
        bundle = _stacked_bundle(tmp_path)
    else:
        bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine(kind))
    fn = _explain_crash_script(bundle.controller.read_text(encoding="utf-8"))

    quiet = run_bash(fn + "\ndocker() { echo 'INFO loading weights'; }\nCONTAINER_NAME=x explain_crash\necho reached\n")
    assert quiet.returncode == 0, quiet.stderr
    assert "reached" in quiet.stdout and quiet.stderr.strip() == ""

    loud = run_bash(fn + "\ndocker() { echo 'ERROR ValueError: No available memory for the cache blocks.'; }\n"
                    "CONTAINER_NAME=x GPU_MEMORY_UTILIZATION=0.4 MAX_MODEL_LEN=8192 explain_crash\necho reached\n")
    assert loud.returncode == 0, loud.stderr
    assert "ValueError: No available memory" in loud.stderr and "reached" in loud.stdout


@pytest.mark.parametrize("engine", [Engine.VLLM, Engine.SGLANG])
def test_a_container_that_dies_before_health_is_reported_not_swallowed(tmp_path, engine):
    """ทางเดินจริง: wait-health กับ docker ปลอมที่บอกว่า container ไม่รันและ log ไม่มี exception"""
    bundle = _bundle(tmp_path, _safetensors_report(), engine=engine)
    fake = _fake_bin(tmp_path, docker='''
        case "$1" in
          ps)   exit 0 ;;
          logs) echo "INFO loading weights" ;;
          *)    exit 0 ;;
        esac
    ''')
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "API_PORT": "1", "HEALTH_TIMEOUT": "5",
           "RUN_DIR": str(tmp_path / "run")}
    out = subprocess.run(["bash", str(bundle.controller), "wait-health"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 1
    assert "หยุดก่อน health" in out.stderr


# ───────────────────────── 3: stacked prompt answers must reach the derived values ─────────────────────────
def _stacked_config_and_prompt(text: str) -> str:
    cfg = text[text.index('MASTER_IP="${MASTER_IP:-'):text.index('MASTER_PORT="${MASTER_PORT:-')]
    return cfg + extract_fn(text, "prompt_cluster_config")


def test_stacked_prompt_answers_drive_transport_ips_and_worker_list(tmp_path):
    """พิมพ์ IP ตอบ prompt แล้ว TRANSPORT_IP_* / WORKER_IPS ต้องตาม — เดิมค้างที่ 10.100.152.x ตัวอย่าง"""
    bundle = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    harness = ("set -Eeuo pipefail\nlog(){ :; }\nCLUSTER_ENV=/nonexistent\nNNODES=2\n"
               + _stacked_config_and_prompt(text)
               + '\nprompt_cluster_config\necho "RESULT MASTER_IP=$MASTER_IP WORKER_IP=$WORKER_IP '
                 'TRANSPORT_IP_MASTER=$TRANSPORT_IP_MASTER TRANSPORT_IP_WORKER=$TRANSPORT_IP_WORKER '
                 'WORKER_IPS=$WORKER_IPS"\n')
    script = tmp_path / "prompt.sh"
    script.write_text(harness, encoding="utf-8")

    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - child
        os.execvp("bash", ["bash", str(script)])
    os.write(fd, b"192.168.1.10\r192.168.1.11\rme\r")
    buf = b""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    os.waitpid(pid, 0)
    result = [l for l in buf.decode(errors="replace").splitlines() if "RESULT" in l][-1]
    assert "MASTER_IP=192.168.1.10" in result and "WORKER_IP=192.168.1.11" in result
    assert "TRANSPORT_IP_MASTER=192.168.1.10" in result
    assert "TRANSPORT_IP_WORKER=192.168.1.11" in result
    assert result.endswith("WORKER_IPS=192.168.1.11")


def test_explicit_transport_ip_still_wins_over_a_changed_master_ip(tmp_path):
    """ค่าที่ตั้งเอง (env/cluster.env) ต้องชนะเสมอ — derive ใหม่ต้องไม่ทับของที่ผู้ใช้ตั้ง"""
    bundle = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    cfg = text[text.index('MASTER_IP="${MASTER_IP:-'):text.index('MASTER_PORT="${MASTER_PORT:-')]
    out = run_bash("set -Eeuo pipefail\n" + cfg
                   + '\nMASTER_IP=2.2.2.2; WORKER_IP=3.3.3.3; _derive_cluster_defaults\n'
                     'echo "$TRANSPORT_IP_MASTER $TRANSPORT_IP_WORKER $WORKER_IPS"\n',
                   env={"TRANSPORT_IP_MASTER": "10.9.9.9"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "10.9.9.9 3.3.3.3 3.3.3.3"


# ───────────────────────── 4: SGLang gets bundle.args / --extra-args / DRY_RUN / download watchdog ─────────────────────────
def _sglang_dry_run(bundle, tmp_path, *flags, env=None):
    out = subprocess.run(["bash", str(bundle.controller), "start", *flags],
                         env={**os.environ, "DRY_RUN": "1", "RUN_DIR": str(tmp_path / "run"), **(env or {})},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out


def test_sglang_bundle_args_reach_serve_args_and_dry_run_works(tmp_path):
    bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.SGLANG)
    (bundle.directory / "bundle.args").write_text('--speculative-algorithm EAGLE --json-model-override-args {"a":1}\n',
                                                 encoding="utf-8")
    lines = _sglang_dry_run(bundle, tmp_path).stdout.splitlines()
    assert any(line.startswith("IMAGE=") for line in lines)
    assert "--speculative-algorithm" in lines and "EAGLE" in lines
    assert '{"a":1}' in lines, "JSON ต้องเป็น argv เดียว"
    assert "--context-length" in lines      # ของ bundle เองยังอยู่ครบ

    flagged = _sglang_dry_run(bundle, tmp_path, "--extra-args", "--from-flag").stdout.splitlines()
    assert "--from-flag" in flagged and "--speculative-algorithm" not in flagged


def test_sglang_download_has_the_stall_watchdog_like_vllm(tmp_path):
    bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.SGLANG)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "_snapshot_download_resilient" in text
    assert "DOWNLOAD_STALL_SECONDS" in text
    assert "--extra-args)" in text
    assert subprocess.run(["bash", "-n", str(bundle.controller)]).returncode == 0


# ───────────────────────── 5: _quote_flag keeps '=' inside values ─────────────────────────
@pytest.mark.parametrize("flag, expected", [
    ("-ot exps=CPU", "-ot exps=CPU"),
    ("--override-kv qwen3moe.expert_used_count=int:8", "--override-kv qwen3moe.expert_used_count=int:8"),
    ("--kv-cache-dtype=fp8", "--kv-cache-dtype fp8"),
    ("--kv-cache-dtype fp8", "--kv-cache-dtype fp8"),
    ("--flash-attn", "--flash-attn"),
])
def test_quote_flag_only_splits_the_flag_token(flag, expected):
    assert renderer._quote_flag(flag) == expected


def test_quote_flag_keeps_json_values_intact():
    quoted = renderer._quote_flag('--hf-overrides {"a":"b=c"}')
    assert quoted.startswith("--hf-overrides ")
    assert quoted.endswith("'{\"a\":\"b=c\"}'")


def test_quote_flag_reaches_the_controller_unmangled(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report(),
                     tweak=lambda plan: setattr(plan.serving, "extra_flags", ["-ot exps=CPU"]))
    out = subprocess.run(["bash", str(bundle.controller), "serve-args"],
                         env={**os.environ, "RUN_DIR": str(tmp_path / "run")},
                         capture_output=True, text=True, timeout=60)
    argv = out.stdout.splitlines()
    assert "exps=CPU" in argv and "exps CPU" not in argv


# ───────────────────────── 6: LLAMA_CPP_UPDATE=1 lets prepare-runtime move past runtime.lock ─────────────────────────
def _prepare_runtime_harness(tmp_path, bundle) -> tuple[str, Path, Path]:
    text = bundle.controller.read_text(encoding="utf-8")
    log = tmp_path / "calls.log"
    llama_dir = tmp_path / "llama.cpp"
    (llama_dir / ".git").mkdir(parents=True)
    server = llama_dir / "build" / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    server.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "runtime.lock").write_text("lockedcommit\n", encoding="utf-8")
    script = (
        "set -Eeuo pipefail\n"
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "install_build_dependencies() { :; }\nneed() { :; }\nnproc() { echo 1; }\n"
        f'git() {{ printf "git %s\\n" "$*" >> "{log}"; if [[ "$*" == *rev-parse* ]]; then echo newcommit; fi; }}\n'
        f'cmake() {{ printf "cmake %s\\n" "$*" >> "{log}"; }}\n'
        f'LLAMA_CPP_DIR="{llama_dir}"\nLLAMA_SERVER="{server}"\nRUN_DIR="{run_dir}"\n'
        f'RUNTIME_LOCK="{run_dir}/runtime.lock"\nLLAMA_CPP_REPO=x\nLLAMA_CPP_REF=master\nCUDA_ARCHITECTURES=121a-real\n'
        + extract_fn(text, "prepare_runtime") + "\nprepare_runtime\n"
    )
    return script, log, run_dir / "runtime.lock"


def test_prepare_runtime_honours_the_lock_by_default(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report())
    script, log, lock = _prepare_runtime_harness(tmp_path, bundle)
    out = run_bash(script)
    assert out.returncode == 0, out.stderr
    calls = log.read_text(encoding="utf-8")
    assert "fetch" not in calls
    assert "checkout --quiet lockedcommit" in calls
    assert lock.read_text().strip() == "lockedcommit"
    assert "LLAMA_CPP_UPDATE=1" in out.stdout      # บอกทางอัปเดตไว้ตรงบรรทัดที่ใช้ lock


def test_prepare_runtime_updates_when_asked(tmp_path):
    """เดิมไม่มีทางไปต่อ: test-tools บอกให้รัน prepare-runtime แต่มันวนกลับมา build commit เดิมเป๊ะ"""
    bundle = _bundle(tmp_path, _gguf_report())
    script, log, lock = _prepare_runtime_harness(tmp_path, bundle)
    out = run_bash(script, env={"LLAMA_CPP_UPDATE": "1"})
    assert out.returncode == 0, out.stderr
    calls = log.read_text(encoding="utf-8")
    assert "fetch --all" in calls
    assert "checkout --quiet master" in calls
    assert lock.read_text().strip() == "newcommit"


def test_test_tools_points_at_the_update_switch(tmp_path):
    text = _bundle(tmp_path, _gguf_report()).controller.read_text(encoding="utf-8")
    assert "LLAMA_CPP_UPDATE=1 {prog} prepare-runtime" in text


# ───────────────────────── 7: stacked watches workers while waiting, and asks each worker for its own HCA ─────────────────────────
def test_stacked_health_wait_checks_workers_and_workers_resolve_their_own_hca(tmp_path):
    bundle = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(bundle.controller)]).returncode == 0
    start = extract_fn(text, "start")
    health = start[start.index("Step 3: health poll"):]
    assert "WORKER_CHECK_INTERVAL" in health
    assert "docker inspect -f '{{.State.Running}}' '${WORKER_CONTAINER}'" in health
    assert "docker logs --tail 100 '${WORKER_CONTAINER}'" in health
    assert 'docker rm -f "$MASTER_CONTAINER"' in health
    assert "worker container บน ${wip} ตายระหว่างรอ head health" in health
    # HCA ของ worker ถามที่ worker เอง แล้วส่งเป็น arg ที่สองของ _nccl_env_pairs
    assert "/sys/class/infiniband/*" in start and 'ssh_at "$wip" "for d in /sys/class/infiniband/*' in start
    assert '_nccl_env_pairs "$wifname" "$whca"' in start


def test_nccl_env_pairs_uses_the_hca_it_is_given(tmp_path):
    text = _stacked_bundle(tmp_path).controller.read_text(encoding="utf-8")
    fn = ("set -Eeuo pipefail\nNCCL_IB_GID_INDEX=3\nNCCL_IB_HCA=mlx5_0\nNCCL_SOCKET_IFNAME=enp1\nTRANSPORT_IP_MASTER=x\n"
          + extract_fn(text, "_nccl_env_pairs"))
    head = run_bash(fn + "\n_nccl_env_pairs enp1\n").stdout.splitlines()
    assert "NCCL_IB_HCA=mlx5_0" in head
    worker = run_bash(fn + "\n_nccl_env_pairs enp7 mlx5_1\n").stdout.splitlines()
    assert "NCCL_IB_HCA=mlx5_1" in worker and "NCCL_SOCKET_IFNAME=enp7" in worker
    no_hca = run_bash(fn + '\n_nccl_env_pairs enp7 ""\n')
    assert "NCCL_IB_DISABLE=1" in no_hca.stdout.splitlines()
    assert not [l for l in no_hca.stdout.splitlines() if l.startswith("NCCL_IB_HCA=")]


# ───────────────────────── 8: HF_TOKEN never on a curl/aria2c argv ─────────────────────────
def test_llamacpp_download_keeps_the_token_off_argv(tmp_path):
    text = _bundle(tmp_path, _gguf_report(gated=True)).controller.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if re.search(r"\b(curl|aria2c)\b", line):   # คำสั่ง ไม่ใช่ตัวแปร curl_auth=
            assert "HF_TOKEN" not in line, line
    assert "-K -" in text and "--conf-path=" in text


def _fetch_one_harness(text: str) -> str:
    return ("set -Eeuo pipefail\n" 'die() { echo "ERROR: $*" >&2; exit 1; }\n'
            + "".join(extract_fn(text, n) for n in ("file_size", "curl_retry_all", "fetch_one")))


def test_curl_receives_the_token_on_stdin_not_argv(tmp_path):
    text = _bundle(tmp_path, _gguf_report()).controller.read_text(encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    fake = _fake_bin(tmp_path, curl=f'''
        printf '%s\\n' "$@" >> "{logs}/curl.argv"
        [[ "${{1:-}}" == "--help" ]] && exit 0
        cat > "{logs}/curl.stdin"
        for ((i=1; i<=$#; i++)); do
          if [[ "${{!i}}" == "-o" ]]; then j=$((i+1)); : > "${{!j}}"; fi
        done
        exit 0
    ''')
    out = run_bash(_fetch_one_harness(text) + f'\nfetch_one https://x/y "{tmp_path}/y.gguf" ""\n',
                   env={"PATH": f"{fake}:{SAFE_PATH}", "HF_TOKEN": "hf_SECRET_TOKEN_42"})
    assert out.returncode == 0, out.stderr
    assert "hf_SECRET_TOKEN_42" not in (logs / "curl.argv").read_text()
    assert 'header = "Authorization: Bearer hf_SECRET_TOKEN_42"' in (logs / "curl.stdin").read_text()


def test_aria2c_receives_the_token_via_a_private_conf_file(tmp_path):
    text = _bundle(tmp_path, _gguf_report()).controller.read_text(encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    fake = _fake_bin(tmp_path, aria2c=f'''
        printf '%s\\n' "$@" >> "{logs}/aria2c.argv"
        for a in "$@"; do
          case "$a" in
            --conf-path=*) cp "${{a#*=}}" "{logs}/aria2c.conf"; stat -c %a "${{a#*=}}" > "{logs}/aria2c.mode"
                           echo "${{a#*=}}" > "{logs}/aria2c.path" ;;
          esac
        done
        exit 0
    ''')
    out = run_bash(_fetch_one_harness(text) + f'\nfetch_one https://x/y "{tmp_path}/y.gguf" ""\n',
                   env={"PATH": f"{fake}:{SAFE_PATH}", "HF_TOKEN": "hf_SECRET_TOKEN_42", "TMPDIR": str(tmp_path)})
    assert out.returncode == 0, out.stderr
    assert "hf_SECRET_TOKEN_42" not in (logs / "aria2c.argv").read_text()
    assert "header=Authorization: Bearer hf_SECRET_TOKEN_42" in (logs / "aria2c.conf").read_text()
    assert (logs / "aria2c.mode").read_text().strip() == "600"
    assert not Path((logs / "aria2c.path").read_text().strip()).exists()   # ลบทิ้งหลังใช้


# ───────────────────────── 9: registries with a port survive the digest pin ─────────────────────────
DIGEST = "sha256:" + "b" * 64


def _pin(plan):
    plan.runtime.image_ref = "localhost:5000/x:tag"
    plan.runtime.image_pin = DIGEST


@pytest.mark.parametrize("kind", ["llamacpp", "vllm", "sglang", "stacked"])
def test_pinned_image_keeps_the_registry_port(tmp_path, kind):
    if kind == "llamacpp":
        bundle = _bundle(tmp_path, _gguf_report(), tweak=_pin)
    elif kind == "stacked":
        bundle = _stacked_bundle(tmp_path, tweak=_pin)
    else:
        bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine(kind), tweak=_pin)
    text = bundle.controller.read_text(encoding="utf-8")
    assert f"localhost:5000/x@{DIGEST}" in text
    assert f"localhost@{DIGEST}" not in text


# ───────────────────────── 10: start hashes once, then trusts the stamp ─────────────────────────
def test_start_path_verification_skips_the_hash_once_a_stamp_exists(tmp_path):
    text = _bundle(tmp_path, _gguf_report()).controller.read_text(encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    gguf = model_dir / "m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 60)
    sha = subprocess.run(["sha256sum", str(gguf)], capture_output=True, text=True).stdout.split()[0]
    log = tmp_path / "sha.log"
    fns = "".join(extract_fn(text, n) for n in (
        "file_size", "file_mtime", "_sha_stamp_path", "_sha_stamp_value", "_sha_stamp_matches",
        "_write_sha_stamp", "_verify_sha", "verify_files"))
    script = (
        "set -Eeuo pipefail\n" 'die() { echo "ERROR: $*" >&2; exit 1; }\nneed() { :; }\n'
        f'sha256sum() {{ echo call >> "{log}"; command sha256sum "$@"; }}\n'
        f'MODEL_DIR="{model_dir}"\nMODEL_FILES=(m.gguf)\nMODEL_FILE=m.gguf\nEXPECTED_SIZES=(64)\nEXPECTED_SHAS=({sha})\n'
        + fns
    )

    def calls() -> int:
        return len(log.read_text().splitlines()) if log.exists() else 0

    assert run_bash(script + "\nverify_files quick\n").returncode == 0     # ยังไม่มี stamp → hash ครั้งเดียว
    assert calls() == 1 and (model_dir / "m.gguf.sha256-ok").exists()
    assert run_bash(script + "\nverify_files quick\nverify_files quick\n").returncode == 0
    assert calls() == 1                                                   # stamp ตรง → ไม่อ่านไฟล์ซ้ำ
    full = run_bash(script + "\nverify_files\n")
    assert full.returncode == 0 and "verify-files: OK" in full.stdout
    assert calls() == 2                                                   # คำสั่ง verify-files ยัง hash เสมอ
    gguf.write_bytes(b"GGUF" + b"\x01" * 60)                              # ไฟล์เปลี่ยน (mtime/เนื้อหา) → hash ใหม่ → ไม่ตรง
    os.utime(gguf, (1, 1))
    bad = run_bash(script + "\nverify_files quick\n")
    assert bad.returncode == 1 and "SHA-256 ไม่ตรง" in bad.stderr
    assert calls() == 3


def test_start_calls_the_quick_verification(tmp_path):
    text = _bundle(tmp_path, _gguf_report()).controller.read_text(encoding="utf-8")
    assert "  verify_files quick\n" in extract_fn(text, "start")
    assert "verify-files) verify_files ;;" in text


# ───────────────────────── 11: port check must not confuse :18000 with :8000 ─────────────────────────
def test_stacked_port_check_matches_the_whole_port(tmp_path):
    text = _stacked_bundle(tmp_path).controller.read_text(encoding="utf-8")
    fn = ("set -Eeuo pipefail\n"
          "ss() { printf 'LISTEN 0 4096 *:18000 *:*\\nLISTEN 0 128 0.0.0.0:22 0.0.0.0:*\\nLISTEN 0 128 [::]:8001 [::]:*\\n'; }\n"
          + extract_fn(text, "_head_port_in_use")
          + '\nif _head_port_in_use; then echo used; else echo free; fi\n')
    assert run_bash(fn, env={"API_PORT": "8000"}).stdout.strip() == "free"
    assert run_bash(fn, env={"API_PORT": "18000"}).stdout.strip() == "used"
    assert run_bash(fn, env={"API_PORT": "22"}).stdout.strip() == "used"
    assert run_bash(fn, env={"API_PORT": "8001"}).stdout.strip() == "used"
    assert run_bash(fn, env={"API_PORT": "800"}).stdout.strip() == "free"


# ───────────────────────── 12: SGLang speaks its own kv-cache dtype names ─────────────────────────
def test_sglang_translates_vllm_fp8_to_fp8_e4m3(tmp_path):
    bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.SGLANG,
                     tweak=lambda plan: setattr(plan.serving, "kv_cache_dtype", "fp8"))
    lines = _sglang_dry_run(bundle, tmp_path).stdout.splitlines()
    assert lines[lines.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert "fp8" not in lines


def test_sglang_drops_a_vllm_only_dtype_loudly(tmp_path):
    bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.SGLANG,
                     tweak=lambda plan: setattr(plan.serving, "kv_cache_dtype", "nvfp4_ds_mla"))
    text = bundle.controller.read_text(encoding="utf-8")
    assert "KV_CACHE_DTYPE:-nvfp4_ds_mla" in text
    out = _sglang_dry_run(bundle, tmp_path)
    assert "--kv-cache-dtype" not in out.stdout.splitlines()          # auto = ไม่ส่ง flag
    assert "nvfp4_ds_mla" in out.stderr and "ใช้ auto แทน" in out.stderr
    # ตั้งเองได้ผ่าน env เหมือน knob อื่น
    forced = _sglang_dry_run(bundle, tmp_path, env={"KV_CACHE_DTYPE": "fp8_e5m2"}).stdout.splitlines()
    assert forced[forced.index("--kv-cache-dtype") + 1] == "fp8_e5m2"
