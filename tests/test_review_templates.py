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
        # prepare_runtime ถาม arch ของโมเดลก่อนเชื่อ lock (2026-09-06) — ไม่มีไฟล์โมเดล = ไม่ฟันธง ใช้ lock ตามเดิม
        f'RUNTIME_MODE=native\nLLAMACPP_IMAGE=x\nMODEL_DIR="{tmp_path}/models"\nMODEL_FILE=missing.gguf\n'
        + extract_fn(text, "gguf_architecture") + extract_fn(text, "runtime_knows_arch")
        + extract_fn(text, "runtime_build_info") + extract_fn(text, "runtime_commit")
        + extract_fn(text, "runtime_build_number") + extract_fn(text, "write_build_stamp")
        + extract_fn(text, "bundles_sharing_build") + extract_fn(text, "warn_running_servers")
        + extract_fn(text, "prepare_runtime") + "\nprepare_runtime\n"
    )
    return script, log, run_dir / "runtime.lock"


def test_prepare_runtime_honours_the_lock_by_default(tmp_path):
    """lock = ขั้นต่ำที่พิสูจน์แล้ว (0.6.1) — build ที่มี (HEAD=newcommit) ใหม่กว่า lock จึงใช้ที่มี ไม่ fetch ไม่ build ซ้ำ
    · lock เลื่อนมาที่ build จริง · ไม่มีวัน checkout ของเก่ามา build ทับ"""
    bundle = _bundle(tmp_path, _gguf_report())
    script, log, lock = _prepare_runtime_harness(tmp_path, bundle)
    out = run_bash(script)
    assert out.returncode == 0, out.stderr
    calls = log.read_text(encoding="utf-8")
    assert "fetch" not in calls and "checkout" not in calls and "cmake" not in calls
    assert "merge-base --is-ancestor lockedcommit newcommit" in calls
    assert lock.read_text().strip() == "newcommit"
    assert "action=reuse" in out.stdout and "LLAMA_CPP_UPDATE=1" in out.stdout      # บอกทางอัปเดตไว้ตรงบรรทัดที่ใช้ build เดิม


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


def test_stacked_start_checks_the_architecture_before_launching_workers(tmp_path):
    """Qwen3.8-Flash-Next stacked 2026-09-05: ทำครบทุกขั้นแล้ว head ตายที่ config (qwen4_exp) — ต้องรู้ก่อนปล่อย worker"""
    from lmds.brain import build_plan
    from lmds.fit import PRESETS, analyze
    from lmds.generator import render_bundle
    from lmds.fit.analyzer import GIB
    from lmds.inspector.report import ArtifactType, KvDims, ModelReport

    report = ModelReport(repo_id="acme/big-moe", revision_sha="sha", artifact_type=ArtifactType.SAFETENSORS,
                         weight_bytes=int(170 * GIB), context_length=262144, architecture="Qwen4ExpForCausalLM",
                         kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128), shard_count=95)
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    text = next(bundle.directory.glob("*-stacked.sh")).read_text(encoding="utf-8")
    assert "check_architecture() {" in text
    start = text.index("start() {")
    assert text.index("  check_architecture\n", start) < text.index("Starting worker rank", start)
    assert "lmds set acme-big-moe --image <image>" in text or "lmds set " in text


# ── เพดาน context ในตัว controller ────────────────────────────────────────────

def _validate_numbers_script(text: str, **env) -> str:
    fn = extract_fn(text, "validate_numbers")
    head = "\n".join(f'{k}="{v}"' for k, v in env.items())
    return f"die() {{ echo \"DIE: $*\" >&2; exit 9; }}\n{head}\n{fn}\nvalidate_numbers && echo VALID"


def test_vllm_controllers_refuse_a_context_above_the_model_maximum_before_touching_docker(tmp_path):
    """เคสจริง 2026-09-05 msi-4/msi-5: Llama-3.3-70B --context 262144 > 131072 → vLLM ตายตอน ModelConfig
    บน worker ก่อน head จะเริ่ม ข้อความยาวและอ่านยาก · ตอนนี้ validate_numbers ปฏิเสธก่อน พร้อมบอกเพดาน
    · ENGINE_ENV มี VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 = ผู้ใช้ตั้งใจ ปล่อยผ่าน"""
    single = _bundle(tmp_path / "s", _safetensors_report(context_length=131072), engine=Engine.VLLM)
    stacked = _stacked_bundle(tmp_path / "k")
    for bundle in (single, stacked):
        text = next(bundle.directory.glob("*-single.sh"), None) or next(bundle.directory.glob("*-stacked.sh"))
        text = text.read_text(encoding="utf-8")
        assert 'NATIVE_CONTEXT="131072"' in text
        base = dict(API_PORT=8000, GPU_MEMORY_UTILIZATION="0.9", NATIVE_CONTEXT=131072, ENGINE_ENV="")
        for extra in ("MAX_NUM_SEQS", "TENSOR_PARALLEL", "NNODES", "CLIENT_OUTPUT", "STARTUP_TIMEOUT"):
            base[extra] = 4 if extra != "STARTUP_TIMEOUT" else 600
        too_big = run_bash(_validate_numbers_script(text, MAX_MODEL_LEN=262144, **base))
        assert too_big.returncode == 9 and "131072" in too_big.stderr and "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in too_big.stderr, too_big.stderr
        ok = run_bash(_validate_numbers_script(text, MAX_MODEL_LEN=131072, **base))
        assert "VALID" in ok.stdout, ok.stderr
        forced = run_bash(_validate_numbers_script(text, MAX_MODEL_LEN=262144, **{**base, "ENGINE_ENV": "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"}))
        assert "VALID" in forced.stdout, forced.stderr
        unknown = run_bash(_validate_numbers_script(text, MAX_MODEL_LEN=262144, **{**base, "NATIVE_CONTEXT": 0}))
        assert "VALID" in unknown.stdout, "ไม่รู้เพดาน (bundle เก่า) = ไม่กีดขวาง"


def test_llamacpp_controller_only_warns_above_the_trained_context(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report(context_length=131072))
    text = next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")
    assert 'NATIVE_CONTEXT="131072"' in text
    env = dict(API_PORT=8000, NATIVE_CONTEXT=131072, CLIENT_OUTPUT=4096, PARALLEL_SEQS=1)
    warned = run_bash(_validate_numbers_script(text, CTX_SIZE=262144, **env))
    assert "VALID" in warned.stdout and "WARN" in warned.stderr and "131072" in warned.stderr


# ───────────────────────── TEMPLATE_HASH: bundle เก่านับว่า stale เมื่อ template เปลี่ยน แม้เลข version เท่ากัน ─────────────────────────
def test_template_hash_changes_when_template_changes(tmp_path):
    import shutil

    import yaml

    from lmds.fleet.consistency import controller_state
    from lmds.generator.renderer import template_hash

    current = template_hash()
    assert len(current) == 12 and current == template_hash()
    bundle = _bundle(tmp_path, _gguf_report())
    text = bundle.controller.read_text(encoding="utf-8")
    assert f'TEMPLATE_HASH="{current}"' in text, "หัว controller ต้องมีลายเซ็น template"
    profile = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["template_hash"] == current
    assert controller_state(profile, bundle.controller)["state"] == "ok"

    copy = tmp_path / "templates"
    shutil.copytree(TEMPLATES, copy)
    assert template_hash(copy) == current
    with open(copy / "single-llamacpp-controller.sh.j2", "ab") as handle:
        handle.write(b"#")                       # แก้ 1 ไบต์
    changed = template_hash(copy)
    assert changed != current
    version = profile["generated_by"].split()[-1]
    # bundle ที่ render ด้วยชุดเดิม เทียบกับแพ็กเกจที่ template เปลี่ยน = stale แม้ version เท่ากัน
    state = controller_state(profile, bundle.controller, package_version=version, package_hash=changed)
    assert state["state"] == "stale" and current in state["reason"]
    # ไม่มี template_hash (bundle ก่อน 0.6.1) = stale แม้ generated_by เท่ากับแพ็กเกจ · adopt = n/a
    old = {k: v for k, v in profile.items() if k != "template_hash"}
    assert controller_state(old, None, package_version=version, package_hash=current)["state"] == "stale"
    assert controller_state({**old, "generated_by": "lmds adopt"}, None)["state"] == "adopted"


# ───────────────────────── audit 2026-09-08: test-tools / test-reasoning / test-embed กับเซิร์ฟเวอร์ปลอม ─────────────────────────
# 10 เครื่องลูกค้า: test-tools บน Qwen3.6 ล้มบ้างผ่านบ้าง (คิด 150–500+ tokens แล้วหมดงบ 512 → "parser แปลไม่ออก" ทั้งที่ parser ถูก) ·
# test-reasoning PASS ทั้งที่ answer 0 ตัวอักษร (1591 อยู่ใน reasoning) · bundle embedding บน llama.cpp ได้ test-vision แต่ไม่มี test-embed
import http.server
import json
import threading


class _FakeChat(http.server.BaseHTTPRequestHandler):
    """/v1/chat/completions ปลอม — พฤติกรรมตาม `mode` · จดทุกคำขอไว้ใน `seen`"""
    mode = "tool_calls"
    seen: list[dict] = []

    def log_message(self, *_a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        type(self).seen.append({"path": self.path, "body": body})
        no_think = ((body.get("chat_template_kwargs") or {}).get("enable_thinking") is False)
        required = body.get("tool_choice") == "required"
        call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Bangkok"}'}}
        mode = type(self).mode
        if self.path == "/v1/embeddings":
            vecs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
            payload = {"object": "list", "data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}
        elif mode == "tool_calls" or required or (mode == "thinking_length" and no_think):
            payload = {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [call]}}]}
        elif mode in ("thinking_length", "length_forever"):
            payload = {"choices": [{"finish_reason": "length",
                                    "message": {"content": "", "reasoning_content": "Let me think about Bangkok weather " * 20}}]}
        elif mode == "leaked":
            payload = {"choices": [{"finish_reason": "stop",
                                    "message": {"content": '<tool_call>{"name": "get_weather", "arguments": {"location": "Bangkok"}}</tool_call>'}}]}
        elif mode == "reasoning_good":
            payload = {"choices": [{"finish_reason": "stop",
                                    "message": {"reasoning_content": "37*43: 37*40=1480, 37*3=111, total 1591",
                                                "content": "37 × 43 = **1591**"}}]}
        elif mode == "reasoning_empty":
            payload = {"choices": [{"finish_reason": "length",
                                    "message": {"reasoning_content": "37*40=1480 … 1480+111=1591 so the answer is 1591 but let me double check "
                                                                     * 3, "content": ""}}]}
        elif mode == "reasoning_wrong_answer":
            payload = {"choices": [{"finish_reason": "stop",
                                    "message": {"reasoning_content": "… 1591 …", "content": "The answer is 1519."}}]}
        elif mode == "reasoning_inline_think":
            # ไม่มี parser: <think> ปนมาใน content · งบเล็กถูกตัดกลาง think · งบใหญ่ตอบครบ
            if int(body.get("max_tokens") or 0) >= 4096:
                payload = {"choices": [{"finish_reason": "stop", "message": {"content": "<think>37*43 … 1591</think>\n\n1591"}}]}
            else:
                payload = {"choices": [{"finish_reason": "length", "message": {"content": "<think>37*43 … 1480+111=1591 hmm"}}]}
        else:
            raise AssertionError(mode)
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_chat():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeChat.seen = []
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()


def _controller_text(tmp_path, engine: str) -> str:
    if engine == "llamacpp":
        bundle = _bundle(tmp_path, _gguf_report())
    elif engine == "stacked":
        bundle = _stacked_bundle(tmp_path)
    else:
        bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.VLLM if engine == "vllm" else Engine.SGLANG)
    path = next(bundle.directory.glob("*-single.sh"), None) or next(bundle.directory.glob("*-stacked.sh"))
    return path.read_text(encoding="utf-8")


def _verb_script(text: str, name: str, *args: str) -> str:
    return ("set -Eeuo pipefail\nassert_our_server() { :; }\nneed() { :; }\ndie() { echo \"ERROR: $*\" >&2; exit 1; }\n"
            + extract_fn(text, name) + "\n" + name + " " + " ".join(args) + "\n")


def _run_verb(text, name, port, mode, *args, **env):
    _FakeChat.mode = mode
    _FakeChat.seen = []
    return run_bash(_verb_script(text, name, *args),
                    env={"PATH": os.environ["PATH"], "API_PORT": str(port), "SERVED_MODEL_NAME": "m", **env}, timeout=120)


ENGINES = ["llamacpp", "vllm", "sglang", "stacked"]


@pytest.mark.parametrize("engine", ENGINES)
def test_test_tools_sends_temperature_0_and_passes_on_tool_calls(tmp_path, fake_chat, engine):
    text = _controller_text(tmp_path, engine)
    done = _run_verb(text, "test_tools", fake_chat, "tool_calls")
    assert done.returncode == 0 and "PASS:" in done.stdout, done.stdout + done.stderr
    bodies = [s["body"] for s in _FakeChat.seen]
    assert bodies and all(b["temperature"] == 0 for b in bodies), "temperature 0 — ผลต้องซ้ำได้ ไม่แกว่งรอบต่อรอบ"
    assert all(b["max_tokens"] >= 512 for b in bodies)
    assert [b["tool_choice"] for b in bodies] == ["auto", "required"]


@pytest.mark.parametrize("engine", ENGINES)
def test_test_tools_grows_the_budget_then_retries_without_thinking_before_judging(tmp_path, fake_chat, engine):
    """Qwen3.6 คิดจนหมด 512 → เดิม FAIL "parser แปลไม่ออก" · ตอนนี้: เห็นว่าคิด → 2048 → ยัง length → ปิด thinking → tool_calls = PASS + คำเตือน"""
    text = _controller_text(tmp_path, engine)
    done = _run_verb(text, "test_tools", fake_chat, "thinking_length")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS:" in done.stdout and "ปิด thinking" in done.stdout
    auto = [s["body"] for s in _FakeChat.seen if s["body"]["tool_choice"] == "auto"]
    assert auto[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert max(b["max_tokens"] for b in auto) >= 2048
    assert all(b["max_tokens"] >= 2048 for b in auto if "chat_template_kwargs" in b)
    if engine != "llamacpp":
        # ไม่ตั้ง reasoning parser = เริ่มเล็ก แล้วขยายเมื่อเห็นว่าโมเดลคิด
        assert auto[0]["max_tokens"] == 512 and auto[1]["max_tokens"] == 2048
        # ตั้ง --reasoning-parser ไว้ = เริ่มที่งบใหญ่เลย
        _run_verb(text, "test_tools", fake_chat, "thinking_length", REASONING_PARSER="qwen3")
        first = [s["body"] for s in _FakeChat.seen if s["body"]["tool_choice"] == "auto"][0]
        assert first["max_tokens"] == 2048


@pytest.mark.parametrize("engine", ENGINES)
def test_test_tools_names_out_of_budget_not_the_parser(tmp_path, fake_chat, engine):
    text = _controller_text(tmp_path, engine)
    done = _run_verb(text, "test_tools", fake_chat, "length_forever")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "FAIL(auto)" in done.stdout and "หมดงบก่อนเรียก tool" in done.stdout
    assert "finish_reason=length" in done.stdout and "reasoning     : Let me think" in done.stdout
    assert "call หลุดออกมาเป็นข้อความ" not in done.stdout and "--tool-parser แปล" not in done.stdout, "หมดงบ ≠ parser ผิด"
    assert "ไม่ใช่" in done.stdout   # บอกตรง ๆ ว่าไม่ใช่ parser/template ผิด
    assert "--tool-parser" not in done.stdout or engine != "llamacpp"


@pytest.mark.parametrize("engine", ENGINES)
def test_test_tools_still_blames_the_parser_when_the_call_leaks_as_text(tmp_path, fake_chat, engine):
    text = _controller_text(tmp_path, engine)
    done = _run_verb(text, "test_tools", fake_chat, "leaked")
    assert done.returncode == 1 and "FAIL(auto)" in done.stdout, done.stdout + done.stderr
    assert "finish_reason : stop" in done.stdout and "content       : <tool_call>" in done.stdout
    if engine == "llamacpp":
        assert "template แปลรูปแบบของมันไม่ออก" in done.stdout and "LLAMA_CPP_UPDATE=1" in done.stdout
    else:
        assert "--tool-parser แปลรูปแบบของมันไม่ออก" in done.stdout and "qwen3_xml" in done.stdout
    assert "หมดงบ" not in done.stdout


@pytest.mark.parametrize("engine", ["vllm", "sglang", "stacked"])
def test_test_reasoning_requires_a_non_empty_final_answer(tmp_path, fake_chat, engine):
    text = _controller_text(tmp_path, engine)
    good = _run_verb(text, "test_reasoning", fake_chat, "reasoning_good")
    assert good.returncode == 0 and good.stdout.startswith("PASS:"), good.stdout + good.stderr
    body = _FakeChat.seen[-1]["body"]
    assert body["temperature"] == 0 and body["max_tokens"] >= 1024
    # เปิด parser = งบใหญ่ขึ้น
    _run_verb(text, "test_reasoning", fake_chat, "reasoning_good", REASONING_PARSER="qwen3")
    assert _FakeChat.seen[-1]["body"]["max_tokens"] >= 4096

    empty = _run_verb(text, "test_reasoning", fake_chat, "reasoning_empty")
    assert empty.returncode == 2, empty.stdout + empty.stderr
    assert empty.stdout.startswith("WARN: คำตอบว่าง — reasoning กินงบ") and "PASS" not in empty.stdout
    assert "finish_reason=length" in empty.stdout and "1591 อยู่ใน chain-of-thought" in empty.stdout

    wrong = _run_verb(text, "test_reasoning", fake_chat, "reasoning_wrong_answer")
    assert wrong.returncode == 1 and "FAIL: คำตอบสุดท้ายไม่มี 1591 (มีแต่ใน reasoning)" in wrong.stdout

    # ไม่มี parser: <think> ในบทสนทนาถูกตัดกลาง think ที่งบ 1024 → ขยายเป็น 4096 → ตอบครบ · เตือนว่า parser ยังไม่ตั้ง
    inline = _run_verb(text, "test_reasoning", fake_chat, "reasoning_inline_think")
    assert inline.returncode == 0 and "WARN: โมเดลพ่น <think>" in inline.stdout, inline.stdout + inline.stderr
    assert [s["body"]["max_tokens"] for s in _FakeChat.seen] == [1024, 4096]


def test_llamacpp_embedding_bundle_gets_test_embed_only_even_with_a_projector(tmp_path, fake_chat):
    """qwen3-vl-embedding-8b-gguf (llama.cpp --embedding --pooling last) มี mmproj → เคยได้ test-vision (chat ไปหา embedding
    = FAIL ขยะ) แต่ไม่มี test-embed · verb ต้องเลือกตาม task: embed → test-embed อย่างเดียว · test-text/test-vision ชี้ทาง"""
    from lmds.inventory import controller_commands

    report = _gguf_report(
        repo_id="Qwen/Qwen3-VL-Embedding-8B-GGUF", selected_gguf="Qwen3-VL-Embedding-8B-Q8_0.gguf",
        gguf_variants=[GgufVariant(filename="Qwen3-VL-Embedding-8B-Q8_0.gguf", size_bytes=8 * GIB, sha256="a" * 64),
                       GgufVariant(filename="mmproj-F16.gguf", size_bytes=1 * GIB, sha256="b" * 64, is_mmproj=True)])

    def make_embed(plan):
        plan.task = "embed"
        plan.tool_calling.enabled = False
        plan.reasoning.enabled = False

    bundle = _bundle(tmp_path, report, tweak=make_embed)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "--mmproj" in text and "MMPROJ_FILE=" in text, "ไฟล์ mmproj ยังโหลด/verify ตามปกติ"
    commands = controller_commands(str(bundle.controller))
    assert "test-embed" in commands and "test-vision" not in commands
    assert "test-tools" not in commands and "test-reasoning" not in commands and "test-text" not in commands
    assert "test_vision()" not in text
    for verb in ("test-text", "test-vision", "test-tools"):
        done = subprocess.run(["bash", str(bundle.controller), verb], capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
        assert done.returncode == 2 and "test-embed" in done.stderr, verb
    # test-embed ยิง /v1/embeddings ของ llama-server จริง (รูป OpenAI) แล้วเทียบ cosine
    done = _run_verb(text, "test_embed", fake_chat, "tool_calls")
    assert done.returncode == 0 and "test-embed: OK" in done.stdout, done.stdout + done.stderr
    assert _FakeChat.seen[-1]["path"] == "/v1/embeddings" and len(_FakeChat.seen[-1]["body"]["input"]) == 3


def test_llamacpp_serve_args_add_metrics_when_the_build_supports_it(tmp_path):
    """llama.cpp build ปัจจุบันไม่พิมพ์บรรทัดคำขอลง server.log — inventory นับการใช้งานจาก /metrics (ต้อง --metrics) ·
    เปิดเฉพาะเมื่อ --help ของ binary รู้จัก flag · LLAMA_METRICS=0 ปิดได้"""
    bundle = _bundle(tmp_path, _gguf_report())
    fake = _fake_bin(tmp_path, **{"llama-server": 'echo "--metrics   enable prometheus compatible metrics endpoint"\n'})
    (tmp_path / "old").mkdir()
    old = _fake_bin(tmp_path / "old", **{"llama-server": 'echo "--port PORT"\n'})
    env = {"PATH": os.environ["PATH"], "RUN_DIR": str(tmp_path / "run"), "RUNTIME_MODE": "native", "HOME": str(tmp_path)}
    def argv(**extra):
        done = subprocess.run(["bash", str(bundle.controller), "serve-args"], capture_output=True, text=True,
                              env={**env, **extra}, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout.splitlines()
    assert "--metrics" in argv(LLAMA_SERVER=str(fake / "llama-server"))
    assert "--metrics" not in argv(LLAMA_SERVER=str(old / "llama-server"))
    assert "--metrics" not in argv(LLAMA_SERVER=str(fake / "llama-server"), LLAMA_METRICS="0")
    assert "--metrics" in argv(LLAMA_SERVER=str(old / "llama-server"), RUNTIME_MODE="docker"), "image server-* รู้จักเสมอ"
