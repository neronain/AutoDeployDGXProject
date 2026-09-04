"""Audit 2026-09-04 — ชุด CLI / controller templates / adopt (regression ของเคสจริงวันนี้ + รอบไล่อ่าน template)

ทุกข้อรัน controller ที่ render แล้วจริง ๆ ใต้ bash (`set -Eeuo pipefail`) กับ curl/docker/git/cmake ปลอมบน PATH
หรือเรียก CLI จริงผ่าน CliRunner — ไม่ใช่ grep template · ตามแนว tests/test_download_resume.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import socket
import subprocess
import textwrap
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

SAFE_PATH = "/usr/bin:/bin"

# ใช้ harness ของ test_parallel_fetch (curl ปลอมที่รองรับ -r) ต่อ — ไม่เขียนซ้ำ
_spec = importlib.util.spec_from_file_location("_pf", Path(__file__).with_name("test_parallel_fetch.py"))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)


# ───────────────────────── helpers ─────────────────────────
def _gguf_report(**overrides) -> ModelReport:
    """GGUF เล็ก 12 ไบต์ตามที่ header บอก (fit ยังคิดจาก weight_bytes) — start ทั้งสคริปต์รันได้จริงใต้เทส"""
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        gguf_variants=[GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=12, sha256=None)],
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


def _bundle(tmp_path, report, target="dgx-spark-single", engine=None, tweak=None):
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None, engine=engine)
    if tweak:
        tweak(plan)
    return render_bundle(plan, report, fit, tmp_path)


def _extract(text: str, name: str) -> str:
    """ตัดฟังก์ชันออกมาด้วยการนับปีกกา (regex พลาดกับ { } ที่ซ้อนกัน)"""
    start = text.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอปีกกาปิดของ {name}")


def _shim(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_FAKE_DF = '''
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "fake 100 100 ${FAKE_DF_KB} 1% /"
'''


# ═════════════════════ 1. fetch_parallel: ดิสก์ต้องว่าง 2 เท่า + ไม่รอเปล่า 30 วิ ═════════════════════
def test_parallel_download_falls_back_to_one_stream_when_the_disk_cannot_hold_two_copies(tmp_path):
    """ส่วนย่อยใน .parts/ + ไฟล์รวมตอน cat = 2 เท่า · ดิสก์ 1.2 เท่าเคยโหลดขนานจนจบแล้วตายตอนต่อไฟล์"""
    src = _pf._source(tmp_path, 260 * 1024 * 1024)
    want = src.stat().st_size
    _pf._bin(tmp_path)
    _shim(tmp_path / "bin", "df", _FAKE_DF)
    done, out = _pf._run(tmp_path, src, want, env_extra={"FAKE_DF_KB": str(int(want * 1.2) // 1024)})
    assert done.returncode == 0, done.stderr + done.stdout
    assert out.read_bytes() == src.read_bytes(), "ถอยไปสตรีมเดี่ยวแล้วยังต้องได้ไฟล์ครบถูกต้อง"
    assert "ถอยไปสตรีมเดี่ยว" in done.stdout and "2 เท่า" in done.stdout
    log = (tmp_path / "curl.log").read_text()
    assert "range=none" in log and not [l for l in log.splitlines() if "range=" in l and "none" not in l]


def test_parallel_download_runs_when_the_disk_has_room_for_two_copies(tmp_path):
    src = _pf._source(tmp_path, 260 * 1024 * 1024)
    want = src.stat().st_size
    _pf._bin(tmp_path)
    _shim(tmp_path / "bin", "df", _FAKE_DF)
    done, out = _pf._run(tmp_path, src, want, env_extra={"FAKE_DF_KB": str(3 * want // 1024)})
    assert done.returncode == 0, done.stderr + done.stdout
    assert out.read_bytes() == src.read_bytes()
    assert "โหลดขนาน 8 ส่วน" in done.stdout


def test_download_dies_up_front_when_the_disk_cannot_hold_the_file_at_all(tmp_path):
    """ไม่พอแม้แต่ไฟล์เดียว — บอกก่อนเสียเวลา ไม่ใช่ curl ตาย 'No space left' ที่ 60 GB"""
    src = _pf._source(tmp_path, 4 * 1024 * 1024)
    want = src.stat().st_size
    _pf._bin(tmp_path)
    _shim(tmp_path / "bin", "df", _FAKE_DF)
    done, out = _pf._run(tmp_path, src, want, env_extra={"FAKE_DF_KB": str(want // 2 // 1024)})
    assert done.returncode != 0
    assert "ดิสก์" in done.stderr and "MODEL_DIR=/data/models" in done.stderr
    assert "range=" not in (tmp_path / "curl.log").read_text(), "ต้องไม่เริ่มโหลดเลย"


def test_parallel_download_returns_as_soon_as_every_part_is_done(tmp_path):
    """เดิม sleep 30 ก่อนเช็คครั้งแรก — ไฟล์ที่เสร็จใน 2 วิ ต้องรออีก 28 วิเปล่า ๆ ทุกไฟล์ ทุกรอบ resume"""
    src = _pf._source(tmp_path, 257 * 1024 * 1024)
    started = time.monotonic()
    done, out = _pf._run(tmp_path, src, src.stat().st_size)
    elapsed = time.monotonic() - started
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == src.read_bytes()
    assert elapsed < 25, f"โหลดขนานของไฟล์ 257 MB ในเครื่องใช้ {elapsed:.0f}s — ยังนั่งรอ sleep 30 อยู่"


# ═════════════════════ 2. start บนเครื่องที่ยังไม่มี llama-server → build ให้เอง ═════════════════════
def _native_env(tmp_path: Path, bundle, port: int, extra: dict | None = None) -> dict:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 8)      # 12 ไบต์ตรง EXPECTED_SIZES
    env = {
        "PATH": f"{tmp_path / 'bin'}:{SAFE_PATH}", "HOME": str(tmp_path),
        "RUNTIME_MODE": "native", "LLAMA_CPP_DIR": str(tmp_path / "llama.cpp"),
        "RUN_DIR": str(tmp_path / "run"), "MODEL_DIR": str(model_dir),
        "API_PORT": str(port), "ADVERTISE_IP": "10.0.0.9", "HEALTH_TIMEOUT": "5",
        "FAKE_LOG": str(tmp_path / "fake.log"),
    }
    env.update(extra or {})
    return env


def _build_shims(bin_dir: Path) -> None:
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, "git", '''
        echo "git $*" >> "$FAKE_LOG"
        case "$1" in
          clone) mkdir -p "$3/.git" ;;
          -C) [[ "$3" == "rev-parse" ]] && echo deadbeef ;;
        esac
        exit 0
    ''')
    _shim(bin_dir, "cmake", '''
        echo "cmake $*" >> "$FAKE_LOG"
        if [[ "$1" == "--build" ]]; then
          mkdir -p "$2/bin"
          cat > "$2/bin/llama-server" <<'SRV'
        #!/bin/bash
        echo "llama-server argv: $*" >> "$FAKE_LOG"
        echo "llama-server env: LLAMA_ARG_API_KEY=${LLAMA_ARG_API_KEY:-unset}" >> "$FAKE_LOG"
        [[ "${1:-}" == "--version" ]] && exit 0
        sleep 2
        SRV
          chmod +x "$2/bin/llama-server"
        fi
        exit 0
    ''')
    for tool in ("nvidia-smi", "nvcc", "gcc"):
        _shim(bin_dir, tool, "exit 0\n")
    _shim(bin_dir, "curl", 'case " $* " in *" --help "*) echo "--retry-all-errors";; esac; exit 0\n')


def test_start_builds_llama_cpp_itself_when_the_binary_is_missing(tmp_path):
    """เคสจริง 2026-09-04: node ใหม่กด start จากหน้าเว็บ → "ยังไม่มี llama-server — รัน prepare-runtime" ทั้งที่
    prepare-runtime ไม่ต้องถามอะไรเมื่อ build deps ครบ · ตอนนี้ start build ให้เอง แล้วเซิร์ฟเวอร์ต้องขึ้นจริง"""
    bundle = _bundle(tmp_path, _gguf_report())
    _build_shims(tmp_path / "bin")
    port = _free_port()
    env = _native_env(tmp_path, bundle, port, {"API_KEY": "sekrit-123"})

    done = subprocess.run(["bash", str(bundle.controller), "start"], capture_output=True, text=True,
                          env=env, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "build ให้ก่อน" in done.stdout and "prepare-runtime" in done.stdout
    log = (tmp_path / "fake.log").read_text(encoding="utf-8")
    assert "cmake --build" in log, "ต้อง build จริง ไม่ใช่แค่พิมพ์คำแนะนำ"
    assert (tmp_path / "run" / "runtime.lock").read_text().strip() == "deadbeef"
    assert (tmp_path / "run" / "server.pid").is_file() and "started: 10.0.0.9" in done.stdout

    # ── API key ไปทางไฟล์ 0600 + --api-key-file ไม่ใช่ argv และ *ไม่ใช่* env LLAMA_ARG_API_KEY
    # (build จริง b10799 ไม่มี env ตัวนั้น — ตั้งแล้วเซิร์ฟเวอร์รันแบบไม่มี auth · พิสูจน์บน dgx-spark03 2026-09-04)
    argv_lines = [l for l in log.splitlines() if l.startswith("llama-server argv:") and "--version" not in l]
    assert argv_lines and "sekrit-123" not in argv_lines[-1] and "--api-key " not in argv_lines[-1]
    assert "--api-key-file" in argv_lines[-1]
    key_file = argv_lines[-1].split("--api-key-file ", 1)[1].split()[0]
    assert Path(key_file).read_text(encoding="utf-8").strip() == "sekrit-123"
    assert oct(Path(key_file).stat().st_mode & 0o777) == "0o600"


def _path_without(tmp_path: Path, hidden: set[str]) -> str:
    """PATH ที่มีทุกอย่างของเครื่องยกเว้นชื่อที่ซ่อน — เครื่องเทสมี git จริงใน /usr/bin จึงซ่อนด้วยการ
    ไม่ใส่ /usr/bin ตรง ๆ แต่ symlink รายตัวเข้าโฟลเดอร์ใหม่แทน (ไม่งั้น "ขาด git" จำลองไม่ได้)"""
    real = tmp_path / "realbin"
    real.mkdir(exist_ok=True)
    for src_dir in (Path("/usr/bin"), Path("/bin")):
        for tool in src_dir.iterdir():
            if tool.name in hidden or (real / tool.name).exists():
                continue
            try:
                (real / tool.name).symlink_to(tool)
            except OSError:
                pass
    return f"{tmp_path / 'bin'}:{real}"


def test_start_without_build_deps_and_without_sudo_names_the_exact_apt_command(tmp_path):
    """ขาด git และ sudo ขอรหัสผ่าน → ต้องตายพร้อมคำสั่ง apt ที่ต้องรันเอง ไม่ใช่ 'a terminal is required'"""
    bundle = _bundle(tmp_path, _gguf_report())
    _build_shims(tmp_path / "bin")
    (tmp_path / "bin" / "git").unlink()
    _shim(tmp_path / "bin", "sudo", "exit 1\n")
    _shim(tmp_path / "bin", "apt-get", "exit 0\n")
    env = _native_env(tmp_path, bundle, _free_port(), {"PATH": _path_without(tmp_path, {"git"})})

    done = subprocess.run(["bash", str(bundle.controller), "start"], capture_output=True, text=True,
                          env=env, timeout=60)
    assert done.returncode != 0
    assert "sudo apt-get update -y && sudo apt-get install -y git" in done.stderr
    log = tmp_path / "fake.log"          # ไม่มีไฟล์ = ไม่มี shim ตัวไหนถูกเรียกเลย ซึ่งคือที่ต้องการ
    assert not log.exists() or "cmake" not in log.read_text(encoding="utf-8")


def test_serve_args_never_prints_the_api_key(tmp_path):
    bundle = _bundle(tmp_path, _gguf_report())
    done = subprocess.run(["bash", str(bundle.controller), "serve-args"], capture_output=True, text=True,
                          env={"PATH": SAFE_PATH, "HOME": str(tmp_path), "API_KEY": "sekrit-123"}, timeout=30)
    assert done.returncode == 0, done.stderr
    assert "sekrit-123" not in done.stdout and "--api-key" not in done.stdout
    assert "--jinja" in done.stdout


# ═════════════════════ 3. vLLM / stacked: VLLM_API_KEY ไม่อยู่บน argv ของ docker run ═════════════════════
def test_vllm_start_hands_the_api_key_to_docker_through_the_environment(tmp_path):
    text = _bundle(tmp_path, _safetensors_report()).controller.read_text(encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _shim(bin_dir, "docker", '''
        echo "docker $*" >> "$FAKE_LOG"
        echo "docker env: VLLM_API_KEY=${VLLM_API_KEY:-unset}" >> "$FAKE_LOG"
        exit 0
    ''')
    _shim(bin_dir, "id", "echo 1000\n")
    stubs = """
        die() { echo "ERROR: $*" >&2; exit 1; }
        need() { :; }; check_gpu() { :; }; verify_files() { :; }; check_architecture() { :; }
        verify_assets() { :; }; write_meta() { :; }; wait_health() { :; }
        warn_open_endpoint() { :; }; network_info() { :; }
        detect_advertise_ip() { echo 1.2.3.4; }
        _resolve_chat_template() { echo "$1"; }; _container_hub_cache() { echo /cache/hub; }
        CONTAINER_NAME=c MODEL_ID=Qwen/Qwen3-32B MODEL_REVISION=r SERVED_MODEL_NAME=x MAX_MODEL_LEN=4096
        GPU_MEMORY_UTILIZATION=0.9 MAX_NUM_SEQS=4 API_HOST=0.0.0.0 API_PORT=8000 EXTRA_SERVE_ARGS=""
        TOOL_CALL_PARSER="" REASONING_PARSER="" CHAT_TEMPLATE="" HF_HOME=/tmp/hf VLLM_IMAGE=img
        ENGINE_ENV="" API_KEY=sekrit-123 TENSOR_PARALLEL_SIZE=1
    """
    script = "set -Eeuo pipefail\n" + textwrap.dedent(stubs) + _extract(text, "start") + "\nstart\n"
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env={"PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(tmp_path),
                               "FAKE_LOG": str(tmp_path / "docker.log")})
    assert done.returncode == 0, done.stdout + done.stderr
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    run_line = next(l for l in log.splitlines() if l.startswith("docker run"))
    assert "sekrit-123" not in run_line, run_line
    assert re.search(r"-e VLLM_API_KEY(\s|$)", run_line), run_line
    assert "docker env: VLLM_API_KEY=sekrit-123" in log, "container ยังต้องได้ค่าเต็มผ่าน env"


def test_stacked_head_hands_the_api_key_to_docker_through_the_environment(tmp_path):
    report = _safetensors_report(repo_id="nvidia/Big-Model", weight_bytes=180 * GIB, shard_count=40,
                                 context_length=131072)
    text = _bundle(tmp_path, report, target="dgx-spark-stacked").controller.read_text(encoding="utf-8")
    assert 'VLLM_API_KEY=${API_KEY}' not in text
    assert 'export VLLM_API_KEY="$API_KEY"; hrun+=(-e VLLM_API_KEY)' in text
    # status/stop พูดถึง worker ทุกตัว ไม่ใช่แค่ตัวแรก (dgx-spark-stacked-4)
    assert 'worker(s)=${WORKER_IPS}' in text and 'Stopping worker(s) on ${WORKER_IPS}' in text


# ═════════════════════ 4. embedding: SGLang ปฏิเสธ · README · client-config ═════════════════════
def _st_embed_report():
    return ModelReport(
        repo_id="Qwen/Qwen3-Embedding-4B", revision_sha="sha", task="embed",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(8 * GIB),
        architecture="Qwen3ForCausalLM", context_length=32768,
        kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
    )


def _gguf_embed_report():
    return ModelReport(
        repo_id="VesNFF/Qwen3-VL-Embedding-8B-GGUF", revision_sha="sha", task="embed",
        artifact_type=ArtifactType.GGUF, weight_bytes=int(8.7 * GIB),
        selected_gguf="Qwen3-VL-Embedding-8B-Q8_0.gguf", architecture="qwen3vl",
        context_length=32768, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        gguf_variants=[], tags=["gguf"],
    )


def test_an_embedding_plan_on_sglang_is_refused_instead_of_rendering_a_chat_controller(tmp_path):
    """template ของ SGLang ไม่รู้จัก task embed — เดิม render controller แบบ chat ให้เงียบ ๆ ไม่มี test-embed"""
    report = _st_embed_report()
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    plan.runtime.engine = Engine.SGLANG
    with pytest.raises(ValueError, match="SGLang"):
        render_bundle(plan, report, fit, tmp_path)


def test_readme_of_an_embedding_bundle_says_test_embed_not_test_text(tmp_path):
    bundle = _bundle(tmp_path, _gguf_embed_report())
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "test-embed" in readme and "-single.sh test-text" not in readme
    assert "/v1/embeddings" in readme and "embedding ไม่มี output token" in readme
    assert "pooling `last`" in readme
    # bundle chat ไม่ถูกแตะ
    chat = (_bundle(tmp_path / "chat", _gguf_report()).directory / "README.md").read_text(encoding="utf-8")
    assert "test-text" in chat and "test-embed" not in chat


def test_embedding_client_config_gives_the_whole_slot_context_and_no_output_budget(tmp_path):
    """สูตร chat (context − output − 2048) ติดลบกับ embed context สั้น แล้ว die ทั้งที่เซิร์ฟเวอร์ปกติดี"""
    bundle = _bundle(tmp_path, _gguf_embed_report())
    done = subprocess.run(["bash", str(bundle.controller), "client-config"], capture_output=True, text=True,
                          env={"PATH": SAFE_PATH, "HOME": str(tmp_path), "ADVERTISE_IP": "10.0.0.9",
                               "CTX_SIZE": "512", "PARALLEL_SEQS": "1"}, timeout=30)
    assert done.returncode == 0, done.stderr
    import json
    cfg = json.loads(done.stdout)
    assert cfg["task"] == "embed" and cfg["endpoint"] == "/v1/embeddings"
    assert cfg["max_input_tokens"] == 512 and cfg["pooling"] == "last"
    assert "max_output_tokens" not in cfg


# ═════════════════════ 5. lmds deploy --gguf ═════════════════════
def _multi_variant_report(selected: str | None = None) -> ModelReport:
    return ModelReport(
        repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf-456", artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf=selected,
        gguf_variants=[
            GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=5 * GIB, sha256="a" * 64),
            GgufVariant(filename="Qwen3-8B-Q8_0.gguf", size_bytes=9 * GIB, sha256="b" * 64),
            GgufVariant(filename="Qwen3-8B-UD-Q8_K_XL.gguf", size_bytes=10 * GIB, sha256="c" * 64),
            GgufVariant(filename="mmproj-BF16.gguf", size_bytes=1 * GIB, sha256="d" * 64, is_mmproj=True),
        ],
        has_chat_template=True, license="apache-2.0",
    )


def _patch_multi_variant(monkeypatch):
    def fake_inspect(source, client):
        # inspect ซ้ำด้วยไฟล์ที่เลือก → report ที่รู้ไฟล์แล้ว (เหมือน Hub ตอบ header ของไฟล์นั้น)
        return _multi_variant_report(selected=getattr(source, "filename", None) or None)
    monkeypatch.setattr("lmds.inspector.inspect_model", fake_inspect)


def test_non_interactive_deploy_of_a_multi_variant_repo_points_at_gguf_flag(isolated_config, tmp_path, monkeypatch):
    _patch_multi_variant(monkeypatch)
    result = CliRunner().invoke(
        __import__("lmds.cli.main", fromlist=["app"]).app,
        ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
         "--output", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "--gguf" in result.output and "Q8_0" in result.output


@pytest.mark.parametrize("wanted, expected", [
    ("Q8_0", "Qwen3-8B-Q8_0.gguf"),                    # ชื่อ quant
    ("q8_k_xl", "Qwen3-8B-UD-Q8_K_XL.gguf"),           # ไม่สนตัวพิมพ์
    ("Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf"),  # ชื่อไฟล์เต็ม (หน้าเว็บส่งแบบนี้)
])
def test_deploy_gguf_flag_picks_the_variant_without_a_tty(isolated_config, tmp_path, monkeypatch, wanted, expected):
    """เคสจริง 2026-09-04: hub สั่ง deploy non-interactive กับ repo หลาย variant แล้ว exit 1 ให้ไปเลือกไฟล์เอง"""
    from lmds.cli.main import app

    _patch_multi_variant(monkeypatch)
    result = CliRunner().invoke(
        app, ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
              "--output", str(tmp_path), "--yes", "--gguf", wanted])
    assert result.exit_code == 0, result.output
    controller = next((tmp_path / "qwen3-8b-gguf").glob("*-single.sh")).read_text(encoding="utf-8")
    assert f'"{expected}"' in controller
    assert "mmproj-BF16.gguf" in controller, "mmproj ยังต้องตามมาเหมือน flow ปกติ"


def test_deploy_gguf_flag_that_matches_nothing_lists_the_real_files(isolated_config, tmp_path, monkeypatch):
    from lmds.cli.main import app

    _patch_multi_variant(monkeypatch)
    result = CliRunner().invoke(
        app, ["deploy", "unsloth/Qwen3-8B-GGUF", "--no-llm", "--target", "dgx-spark-single",
              "--output", str(tmp_path), "--yes", "--gguf", "Q9_9"])
    assert result.exit_code == 1
    assert "Q9_9" in result.output and "Qwen3-8B-Q8_0.gguf" in result.output
    assert "mmproj" not in result.output, "mmproj ไม่ใช่ตัวเลือกของ weight"


def test_pick_gguf_variant_reports_ambiguity_instead_of_guessing():
    from lmds.cli.main import _pick_gguf_variant

    variants = [v for v in _multi_variant_report().gguf_variants if not v.is_mmproj]
    chosen, clash = _pick_gguf_variant(variants, "q8")          # ส่วนของชื่อที่ตรงสองไฟล์
    assert chosen is None and {v.filename for v in clash} == {"Qwen3-8B-Q8_0.gguf", "Qwen3-8B-UD-Q8_K_XL.gguf"}
    chosen, _ = _pick_gguf_variant(variants, "UD-Q8")            # ส่วนของชื่อที่ตรงไฟล์เดียว
    assert chosen.filename == "Qwen3-8B-UD-Q8_K_XL.gguf"


# ═════════════════════ 6. commit ย่อยาวไม่เท่ากัน = commit เดียวกัน ═════════════════════
def test_short_hashes_of_different_length_compare_as_the_same_commit():
    from lmds.cli.main import _same_commit

    assert _same_commit("0ad1a59e", "0ad1a59")
    assert _same_commit("0AD1A59", "0ad1a59e2f")
    assert not _same_commit("0ad1a59", "0ad1a5")          # สั้นกว่า 7 ไม่ฟันธง
    assert not _same_commit("abc1234", "abc1235")
    assert not _same_commit("", "abc1234") and not _same_commit("abc1234", "")


def test_node_list_marks_only_nodes_whose_commit_really_differs_from_the_hub(isolated_config, tmp_path, monkeypatch):
    from lmds.cli.main import app
    from lmds.nodes import Node, add, update

    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("lmds.inventory.source_commit", lambda: "0ad1a59")
    for name, host, commit in (("fresh", "10.0.0.1", "0ad1a59e"), ("stale", "10.0.0.2", "abc1234")):
        add(Node(name=name, host=host, user="u"))
        update(name, lmds_version="0.6.0", lmds_commit=commit)

    result = CliRunner().invoke(app, ["node", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    rows = {line.split("│")[1].strip(): line for line in result.output.splitlines() if "│" in line and "0.6.0" in line}
    assert "≠ hub" not in rows["fresh"], rows["fresh"]
    assert "≠ hub" in rows["stale"], rows["stale"]


def test_node_install_summary_says_whether_the_node_now_matches_the_hub(isolated_config, tmp_path, monkeypatch):
    """สรุปหลังติดตั้ง: เดิมพิมพ์แค่ 'รัน lmds 0.6.0 (0ad1a59e)' ให้คนไปเทียบ hash เอง (แล้วเทียบผิดเพราะ 7≠8 ตัว)"""
    from types import SimpleNamespace

    from lmds.cli.main import app
    from lmds.nodes import Node, add

    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("lmds.inventory.source_commit", lambda: "0ad1a59")
    add(Node(name="n1", host="10.0.0.1", user="u"))
    monkeypatch.setattr("lmds.nodes.install_lmds",
                        lambda node, with_prereq=False: SimpleNamespace(ok=True, stdout="", stderr=""))
    monkeypatch.setattr("lmds.nodes.probe",
                        lambda node: {"host": {"lmds_version": "0.6.0", "lmds_commit": "0ad1a59e"}})

    result = CliRunner().invoke(app, ["node", "install", "n1"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "ตรง hub" in result.output and "ยังไม่ตรง" not in result.output

    monkeypatch.setattr("lmds.nodes.probe",
                        lambda node: {"host": {"lmds_version": "0.6.0", "lmds_commit": "abc1234"}})
    result = CliRunner().invoke(app, ["node", "install", "n1"], env={"COLUMNS": "200"})
    assert "ยังไม่ตรง hub (hub: 0ad1a59)" in result.output


# ═════════════════════ 7. adopt จดที่เก็บ weight — remove/status รู้ว่าจะลบอะไร ═════════════════════
def test_adopt_reads_the_weight_location_from_the_bind_mounts(tmp_path):
    from lmds.fleet.adopt import Adopted, weights_on_host

    (tmp_path / "srv" / "models" / "foo").mkdir(parents=True)
    by_path = Adopted(container="c", image="i", env=["MODEL=/models/foo"], binds=[f"{tmp_path}/srv/models:/models"])
    assert weights_on_host(by_path)["path"] == f"{tmp_path}/srv/models/foo"
    assert weights_on_host(by_path)["kind"] == "dir"

    (tmp_path / "hf" / "hub" / "models--org--name").mkdir(parents=True)
    by_repo = Adopted(container="c", image="i", args=["serve", "org/name"],
                      binds=[f"{tmp_path}/hf:/root/.cache/huggingface"])
    assert weights_on_host(by_repo) == {"path": f"{tmp_path}/hf/hub/models--org--name", "kind": "hf-cache",
                                        "source": "bind-mount", "binds": [f"{tmp_path}/hf:/root/.cache/huggingface"]}

    unknown = Adopted(container="c", image="i", args=["serve", "org/other"])
    assert weights_on_host(unknown) == {"hf_repo": "org/other", "kind": "hf-cache"}, "ไม่รู้ = ไม่เดา path"


def test_adopted_bundle_records_weights_and_its_controller_lists_what_remove_deletes(isolated_config, tmp_path, monkeypatch):
    import yaml

    # ได้ *ฟังก์ชัน* ถ้า from-import เพราะ __init__ re-export ทับชื่อโมดูล
    adopt_mod = importlib.import_module("lmds.fleet.adopt")

    (tmp_path / "srv" / "models" / "foo").mkdir(parents=True)
    fake = adopt_mod.Adopted(container="oldvllm", image="vllm/vllm-openai:v0.20.0",
                             args=["--model", "/models/foo", "--port", "8355"],
                             binds=[f"{tmp_path}/srv/models:/models"], runtime="nvidia")
    monkeypatch.setattr(adopt_mod, "inspect_container", lambda name: fake)

    controller = adopt_mod.adopt("oldvllm", slug="oldvllm", output=tmp_path / "bundles")
    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["weights"]["path"] == f"{tmp_path}/srv/models/foo"
    assert profile["weights"]["binds"] == [f"{tmp_path}/srv/models:/models"]

    env = {"PATH": SAFE_PATH, "HOME": str(tmp_path)}
    assert subprocess.run(["bash", "-n", str(controller)], capture_output=True).returncode == 0
    plan = subprocess.run(["bash", str(controller), "remove-plan"], capture_output=True, text=True, env=env)
    assert plan.returncode == 0, plan.stderr
    assert f"weights:   {tmp_path}/srv/models/foo" in plan.stdout
    assert str(controller.parent) in plan.stdout and "oldvllm" in plan.stdout
    info = subprocess.run(["bash", str(controller), "info"], capture_output=True, text=True, env=env)
    assert f"weights:   {tmp_path}/srv/models/foo" in info.stdout
    assert "remove-plan" in subprocess.run(["bash", str(controller)], capture_output=True, text=True, env=env).stdout


def test_native_adopted_bundle_records_the_gguf_path_it_serves(isolated_config, tmp_path, monkeypatch):
    import yaml

    # ได้ *ฟังก์ชัน* ถ้า from-import เพราะ __init__ re-export ทับชื่อโมดูล
    adopt_mod = importlib.import_module("lmds.fleet.adopt")

    proc = adopt_mod.AdoptedProcess(
        pid=4242, argv=["./llama-server", "-m", "/home/u/models/q.Q4_K_M.gguf", "--port", "8080", "-c", "8192"],
        exe="/home/u/llama.cpp/llama-server", cwd="/home/u/llama.cpp", unit="")
    monkeypatch.setattr(adopt_mod, "inspect_process", lambda pid=0, port=0: proc)
    monkeypatch.setattr(adopt_mod, "probe_server", lambda port, timeout=5.0: {})

    controller, _ = adopt_mod.adopt_process(pid=4242, slug="q-adopted", output=tmp_path / "bundles")
    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["weights"] == {"path": "/home/u/models/q.Q4_K_M.gguf", "kind": "file", "source": "argv"}
    plan = subprocess.run(["bash", str(controller), "remove-plan"], capture_output=True, text=True,
                          env={"PATH": SAFE_PATH, "HOME": str(tmp_path)})
    assert plan.returncode == 0, plan.stderr
    assert "weights:   /home/u/models/q.Q4_K_M.gguf" in plan.stdout


# ═════════════════════ 8. usage ต้องตรงกับ dispatch ของทุก template ═════════════════════
def _usage_commands(text: str) -> set[str]:
    block = text.split("\nCOMMANDS")[1].split("\nOPTIONS")[0]
    return {m.group(1) for m in re.finditer(r"^  ([a-z][a-z-]+)", block, re.M)}


def _dispatched(text: str) -> set[str]:
    tail = text.rsplit('case "${1:-help}" in', 1)[1]
    out: set[str] = set()
    for m in re.finditer(r"^\s*([a-z][a-z|-]*)\)", tail, re.M):
        out.update(m.group(1).split("|"))
    return out


@pytest.mark.parametrize("kind", ["llamacpp", "llamacpp-embed", "vllm", "vllm-embed", "sglang", "stacked"])
def test_every_command_in_usage_is_actually_dispatched(tmp_path, kind):
    if kind == "llamacpp":
        bundle = _bundle(tmp_path, _gguf_report())
    elif kind == "llamacpp-embed":
        bundle = _bundle(tmp_path, _gguf_embed_report())
    elif kind == "vllm":
        bundle = _bundle(tmp_path, _safetensors_report())
    elif kind == "vllm-embed":
        bundle = _bundle(tmp_path, _st_embed_report())
    elif kind == "sglang":
        bundle = _bundle(tmp_path, _safetensors_report(), engine=Engine.SGLANG)
    else:
        bundle = _bundle(tmp_path, _safetensors_report(repo_id="nvidia/Big-Model", weight_bytes=180 * GIB,
                                                       shard_count=40, context_length=131072),
                         target="dgx-spark-stacked")
    text = bundle.controller.read_text(encoding="utf-8")
    usage, dispatched = _usage_commands(text), _dispatched(text)
    assert usage, "อ่านบล็อก COMMANDS ไม่ได้"
    assert usage - dispatched == set(), f"{kind}: อยู่ใน usage แต่ไม่มีใน dispatch"


# ═════════════════════ 9. download ซ้อนกัน → ตัวที่สองต้องถูกปฏิเสธ ไม่ใช่ทำ .parts พังแล้ว exit 1 ═════════════════════
def test_a_second_download_of_the_same_bundle_is_refused_while_the_first_is_still_running(tmp_path):
    """เคสจริง 2026-09-04: `node push --download` จบ exit 1 แต่ curl ยังโหลดต่อ — สอง download เขียนส่วนเดียวกัน
    ตัวหนึ่งเห็น "ไม่คืบหน้า" แล้ว die ส่วนอีกตัวยังวิ่ง · ตอนนี้ตัวที่สองต้องบอกว่ามีคนโหลดอยู่แล้ว"""
    bundle = _bundle(tmp_path, _gguf_report())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _shim(bin_dir, "aria2c", "exit 1\n")
    _shim(bin_dir, "curl", '''
        case " $* " in *" --help "*) echo "--retry-all-errors"; exit 0;; esac
        out=""; prev=""
        for a in "$@"; do [[ "$prev" == "-o" ]] && out="$a"; prev="$a"; done
        cat >/dev/null            # -K - อ่าน config จาก stdin
        sleep 3
        [[ -s "$out" ]] || printf 'GGUF12345678' > "$out"
    ''')
    env = {"PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(tmp_path), "MODEL_DIR": str(tmp_path / "models")}
    first = subprocess.Popen(["bash", str(bundle.controller), "download"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    time.sleep(1)
    second = subprocess.run(["bash", str(bundle.controller), "download"], capture_output=True, text=True,
                            env=env, timeout=30)
    out, err = first.communicate(timeout=60)

    assert second.returncode != 0 and "กำลังรันอยู่แล้ว" in second.stderr, second.stdout + second.stderr
    assert first.returncode == 0, out + err
    assert (tmp_path / "models" / "Qwen3-8B-Q4_K_M.gguf").read_bytes() == b"GGUF12345678"
    # ตัวแรกจบแล้ว ล็อกหลุด — สั่งซ้ำได้ตามปกติ
    third = subprocess.run(["bash", str(bundle.controller), "download"], capture_output=True, text=True,
                           env=env, timeout=30)
    assert third.returncode == 0, third.stdout + third.stderr


# ═════════════════════ 10. _run_detached: ssh สะดุดหนึ่งรอบ ต้องไม่พิมพ์ log ซ้ำทั้งก้อน ═════════════════════
def test_run_detached_does_not_replay_the_whole_log_after_one_failed_poll(monkeypatch, capsys):
    from types import SimpleNamespace

    from lmds.cli.main import _run_detached
    from lmds.nodes import Node

    answers = iter([
        SimpleNamespace(ok=True, stdout="started\n", stderr="", exit_code=0),          # launch
        SimpleNamespace(ok=True, stdout="line1\n", stderr="", exit_code=0),
        SimpleNamespace(ok=False, stdout="", stderr="ssh: broken pipe", exit_code=255),  # สะดุด
        SimpleNamespace(ok=True, stdout="line1\nline2\n__RC=0\n", stderr="", exit_code=0),
    ])
    monkeypatch.setattr("lmds.nodes.run", lambda node, command, timeout=60: next(answers))

    rc = _run_detached(Node(name="n", host="h", user="u"), "lmds repair x", "x", timeout=30, poll=0)
    assert rc == 0
    assert capsys.readouterr().out == "line1\nline2\n"
