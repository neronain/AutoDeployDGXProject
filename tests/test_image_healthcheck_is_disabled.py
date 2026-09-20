"""HEALTHCHECK ที่ image แถมมาต้องถูกปิด ไม่ใช่ปล่อยให้ยิงผิดพอร์ตไปตลอด

เคสจริง 2026-09-20 บน BesthaiAi:

    $ docker inspect ghcr.io/ggml-org/llama.cpp:server-cuda --format '{{json .Config.Healthcheck}}'
    {"Test":["CMD","curl","-f","http://localhost:8080/health"]}
    $ docker ps
    lmds-qwen36-35b-abl  Up 4 minutes (unhealthy)
    $ curl -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health
    200

image ฝังพอร์ต 8080 (ค่า default ของตัวเอง) ไว้ในคำสั่ง healthcheck · LMDS เลือกพอร์ตว่างให้เอง
(รอบนี้ 8001 เพราะ portainer ถือ 8000 อยู่) healthcheck จึงไม่มีทางผ่าน แล้ว `docker ps` ขึ้น
"(unhealthy)" ตลอดชีวิตของ container ที่แข็งแรงดี — ปัญหาจริงจะแยกไม่ออกจากเสียงเตือนปลอมนี้

ตรวจ config blob บน registry จริงวันเดียวกัน: llama.cpp มี HEALTHCHECK · vllm/vllm-openai:latest
กับ lmsysorg/sglang:latest วันนี้ยังไม่มี — แต่ tag เคลื่อนที่ได้และผู้ใช้ pin image เองได้ ทุก
controller จึงต้องส่ง --no-healthcheck เองเสมอ ไม่ใช่พึ่งว่า "image นี้คงไม่มี"

เทสไม่ grep template เฉย ๆ แต่ **รันบรรทัด docker run ที่ render ออกมาจริง** ใต้
set -euo pipefail กับ docker ปลอมที่บันทึก argv — ตามกฎของโปรเจกต์ (ดู
tests/test_controller_runs_not_just_parses.py: `$4: unbound variable` เคยผ่าน bash -n
ไปโผล่บนเครื่องจริง)
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from lmds.brain import build_plan
from lmds.brain.plan_schema import Engine
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

SAFE_PATH = "/usr/bin:/bin"


# ───────────────────────── render bundle จริงของแต่ละ engine ─────────────────────────
def _gguf_report() -> ModelReport:
    return ModelReport(
        repo_id="unsloth/Qwen3-8B-GGUF", revision_sha="sha-gguf", artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB, context_length=40960, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        gguf_variants=[GgufVariant(filename="Qwen3-8B-Q4_K_M.gguf", size_bytes=5 * GIB, sha256="a" * 64)],
        has_chat_template=True, license="apache-2.0",
    )


def _safetensors_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="Qwen/Qwen3-32B", revision_sha="sha-pinned", artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=65 * GIB, shard_count=17, context_length=40960,
        kv_dims=KvDims(layers=64, kv_heads=8, head_dim=128), has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def _controller(tmp_path: Path, report: ModelReport, target="dgx-spark-single", engine=None) -> str:
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None, engine=engine)
    bundle = render_bundle(plan, report, fit, tmp_path)
    pattern = "*-stacked.sh" if target.endswith("stacked") else "*-single.sh"
    return next(bundle.directory.glob(pattern)).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def controllers(tmp_path_factory) -> dict[str, str]:
    """controller ที่ render จริงของทั้งสี่ template ที่สั่ง container"""
    root = tmp_path_factory.mktemp("bundles")
    return {
        "llamacpp": _controller(root / "l", _gguf_report()),
        "vllm": _controller(root / "v", _safetensors_report(), engine=Engine.VLLM),
        "sglang": _controller(root / "s", _safetensors_report(), engine=Engine.SGLANG),
        "stacked": _controller(
            root / "k",
            _safetensors_report(repo_id="nvidia/Big-Model", weight_bytes=180 * GIB,
                                shard_count=40, context_length=131072),
            target="dgx-spark-stacked"),
    }


# ───────────────────────── ตัดโค้ดจริงออกมารัน ─────────────────────────
def _slice(text: str, first: str, last: str) -> str:
    """ตัดช่วงบรรทัดจริงจาก controller (ตั้งแต่บรรทัดที่มี `first` ถึงบรรทัดที่มี `last`)"""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if first in line)
    end = next(i for i, line in enumerate(lines[start:], start) if last in line)
    return "\n".join(lines[start : end + 1])


def _bash(tmp_path: Path, body: str, fake_docker_log: Path | None = None) -> subprocess.CompletedProcess:
    """รันโค้ดชิ้นนั้นจริงใต้ set -euo pipefail · docker ปลอมอยู่บน PATH และบันทึก argv ทีละบรรทัด"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if fake_docker_log is not None:
        docker = bin_dir / "docker"
        docker.write_text(
            "#!/bin/bash\nprintf '%s\\n' \"$@\" >> " + str(fake_docker_log) + "\nexit 0\n",
            encoding="utf-8")
        docker.chmod(0o755)
    script = tmp_path / "harness.sh"
    script.write_text(body, encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=60,
        env={"PATH": f"{bin_dir}:{SAFE_PATH}", "HOME": str(tmp_path)},
    )


_PREAMBLE = textwrap.dedent("""
    set -euo pipefail
    CONTAINER_NAME=lmds-demo
    MASTER_CONTAINER=lmds-demo-head
    WORKER_CONTAINER=lmds-demo-worker
    LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
    VLLM_IMAGE=vllm/vllm-openai:latest
    SGLANG_IMAGE=lmsysorg/sglang:latest
    MODEL_DIR=/models
    HF_HOME=/cache
    VLLM_CACHE=/cache/vllm
    WORKER_VLLM_CACHE=/cache/vllm
    SLUG=demo
    API_KEY=
    API_KEY_FILE=/tmp/key
    HF_TOKEN=
    PLUGIN_DIR=/plugins
    PLUGIN_MOUNT=/plugins
    SCORE_TEMPLATE=
    ENGINE_ENV=
    TRANSPORT_IP_MASTER=10.0.0.1
    LMDS_RUN_AS_ROOT=
    head_fi=/cache/fi
    worker_fi=/cache/fi
    worker_hf=/cache
    SERVER_ARGS=(--port 8001)
    serve_args=(--port 8001)
    EXTRA_DOCKER_ENV=()
    nccl_docker=()
    _container_hub_cache() { echo /cache/hub; }
    _write_api_key_file() { :; }
    verify_assets() { :; }
""")


# ───────────────────────── รันจริง: llama.cpp ─────────────────────────
def test_the_llamacpp_docker_run_line_really_passes_no_healthcheck(tmp_path, controllers):
    """image ตัวนี้คือตัวที่ฝัง HEALTHCHECK localhost:8080 มาจริง — บรรทัดที่รันต้องปิดมัน"""
    log = tmp_path / "argv.log"
    body = _PREAMBLE + _slice(
        controllers["llamacpp"], 'docker run -d --name "$CONTAINER_NAME"', '"${SERVER_ARGS[@]}"')

    result = _bash(tmp_path, body, log)

    combined = result.stdout + result.stderr
    assert "unbound variable" not in combined, combined
    assert result.returncode == 0, combined
    argv = log.read_text(encoding="utf-8").split("\n")
    assert "--no-healthcheck" in argv, argv
    assert "--name" in argv and "lmds-demo" in argv
    for kept in ("--gpus", "all", "--network", "host"):
        assert kept in argv, f"ธง {kept} หายไป\n{argv}"
    assert not any(a.startswith("--health-") and a != "--health-" for a in argv if a != "--no-healthcheck"), argv


# ───────────────────────── รันจริง: vLLM / SGLang ─────────────────────────
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_single_node_docker_args_really_carry_no_healthcheck(tmp_path, controllers, engine):
    """array ถูกประกอบแล้วส่งให้ `docker` จริง — เทสจับที่ argv ที่ docker ได้รับ ไม่ใช่ที่ template"""
    log = tmp_path / "argv.log"
    body = _PREAMBLE + _slice(
        controllers[engine], "local docker_args=(", 'docker "${docker_args[@]}"'
    ).replace("local docker_args=(", "docker_args=(").replace("  local kv", "  kv")

    result = _bash(tmp_path, body, log)

    combined = result.stdout + result.stderr
    assert "unbound variable" not in combined, combined
    assert result.returncode == 0, combined
    argv = log.read_text(encoding="utf-8").split("\n")
    assert argv[0] == "run", argv[:4]
    assert "--no-healthcheck" in argv, argv
    assert "lmds-demo" in argv
    for kept in ("--gpus", "all", "--network", "host", "--ipc"):
        assert kept in argv, f"{engine}: ธง {kept} หายไป\n{argv}"


# ───────────────────────── รันจริง: stacked (head + worker) ─────────────────────────
def _array_literal(text: str, declaration: str) -> str:
    """ตัด `local -a name=( … )` ออกมาทั้งก้อนด้วยการนับวงเล็บ — คอมเมนต์ข้างในมี `(` ไม่ได้"""
    start = text.index(declaration)
    depth, i = 0, text.index("(", start)
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอวงเล็บปิดของ {declaration}")


@pytest.mark.parametrize("declaration,name", [
    ("local -a wrun=(", "wrun"),
    ("local -a hrun=(", "hrun"),
    ("local -a hdry=(", "hdry"),
])
def test_every_stacked_container_command_disables_the_image_healthcheck(
        tmp_path, controllers, declaration, name):
    """head, worker และคำสั่งที่ DRY RUN พิมพ์ให้ดู — ต้องปิดเหมือนกันทั้งสามที่

    DRY RUN ที่พิมพ์คำสั่งไม่ตรงกับของที่รันจริงก็เป็นหน้าจอที่โกหกอีกแบบหนึ่ง
    """
    literal = _array_literal(controllers["stacked"], declaration).replace("local -a ", "", 1)
    body = _PREAMBLE + f"\n{literal}\nprintf '%s\\n' \"${{{name}[@]}}\"\n"

    result = _bash(tmp_path, body)

    combined = result.stdout + result.stderr
    assert "unbound variable" not in combined, combined
    assert result.returncode == 0, combined
    argv = result.stdout.split("\n")
    assert argv[0] == "docker" and argv[1] == "run", argv[:4]
    assert "--no-healthcheck" in argv, argv
    # ธงที่มีอยู่ก่อนต้องไม่หายไปกับการแทรกธงใหม่ — เผลอทับบรรทัดเดิมแล้ว GPU/เครือข่ายหลุดทั้งชุด
    for kept in ("--gpus", "all", "--network", "host", "--ipc"):
        assert kept in argv, f"{name}: ธง {kept} หายไป\n{argv}"


# ───────────────────────── สัญญาณสุขภาพต้องมีแหล่งเดียว ─────────────────────────
def _code_lines(text: str) -> list[str]:
    """เฉพาะบรรทัดที่ bash รันจริง — คอมเมนต์อธิบาย "ทำไมไม่ใช้ --health-cmd" ต้องไม่ถูกนับเป็นการใช้มัน"""
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


@pytest.mark.parametrize("engine", ["llamacpp", "vllm", "sglang", "stacked"])
def test_no_controller_installs_a_second_health_command(controllers, engine):
    """ปิด ไม่ใช่เขียนทับ — wait_health/_health_ok ของ LMDS ต้องเป็นแหล่งความจริงแหล่งเดียว

    /health ของ SGLang รันโมเดลจริงหนึ่งรอบทุกครั้งที่ถูกเรียก (เคส spark-head 2026-09-01)
    healthcheck ฝั่ง docker ที่รันเองทุก ๆ กี่วินาทีจึงไม่ใช่แค่ "ซ้ำซ้อน" แต่กลับไปเปิดบั๊กเดิม
    """
    code = "\n".join(_code_lines(controllers[engine]))
    for flag in ("--health-cmd", "--health-interval", "--health-retries", "--health-start-period"):
        assert flag not in code, f"{engine}: ยังตั้ง {flag} — สัญญาณสุขภาพต้องมาจาก LMDS ที่เดียว"


@pytest.mark.parametrize("engine", ["llamacpp", "vllm", "sglang", "stacked"])
def test_the_controller_still_waits_for_health_itself(controllers, engine):
    """ปิดของ docker แล้ว ของเราต้องยังอยู่ — ไม่ใช่ปิดสัญญาณแล้วไม่เหลืออะไรเลย"""
    assert "wait_health" in controllers[engine]


@pytest.mark.parametrize("engine", ["llamacpp", "vllm", "sglang", "stacked"])
def test_every_long_lived_container_start_disables_the_healthcheck(controllers, engine):
    """สแกนทุกจุดที่ `docker run -d` — ไม่ให้มีจุดไหนหลุดตอนเพิ่ม container ใหม่ทีหลัง

    ตัวที่ `--rm` (probe nvidia-smi / นับไฟล์ใน image) ไม่นับ: มันจบใน 1-2 วิ ไม่มีใครดู `docker ps`
    ของมันทัน · ที่ต้องปิดคือ container ที่อยู่ยาวจนคนเห็นสถานะของมัน
    """
    lines = _code_lines(controllers[engine])
    starts = [i for i, line in enumerate(lines) if "docker run -d" in line]
    assert starts, f"{engine}: ไม่เจอ `docker run -d` เลย — เทสนี้เฝ้าอะไรอยู่?"
    for i in starts:
        window = "\n".join(lines[i : i + 14])
        assert "--no-healthcheck" in window, \
            f"{engine} บรรทัด {i + 1}: docker run -d ที่ไม่ได้ปิด healthcheck ของ image\n{window}"
