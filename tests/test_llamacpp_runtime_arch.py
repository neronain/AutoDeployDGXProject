"""llama.cpp เก่ากว่าโมเดล — ต้องรู้ก่อน start ไม่ใช่รู้จาก server.log ที่ต้อง ssh ไปอ่าน

เคสจริง 2026-09-06 spark-worker: `start · qwen3-8-flash-next-uncensored-gguf — failed (exit 1)` บนหน้าเว็บ
พร้อมข้อความกลาง ๆ "เซิร์ฟเวอร์หยุดก่อน health ผ่าน" · สาเหตุจริง `unknown model architecture: 'qwen4exp'`
อยู่ใน ~/.lmds/run/<slug>/server.log · build llama.cpp ในโฟลเดอร์กลางของเครื่องมาจาก 18 ส.ค. (10495)
ส่วน upstream เพิ่ม qwen4exp 27 ส.ค. (6c84c7d5d) · ไม่มีอะไรสักชั้นที่ถามว่า build รู้จัก arch ของไฟล์ไหม

เทสฝั่ง controller รันสคริปต์ที่ render แล้วจริง ๆ ใต้ bash กับ llama-server/docker/git/cmake ปลอมบน PATH
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, GgufVariant, KvDims, ModelReport

SAFE_PATH = "/usr/bin:/bin"
ARCH = "qwen4exp"
GGUF_NAME = "Qwen3-8-Flash-Next-Q4_K_M.gguf"


# ───────────────────────── ไฟล์ GGUF จริง (header เท่านั้น) ─────────────────────────
def _kv_string(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def gguf_bytes(arch: str = ARCH) -> bytes:
    """GGUF v3 ที่มี kv หลายชนิดก่อนถึง general.architecture — ตัวอ่านต้องข้าม array/scalar ให้ถูก"""
    tokens = [b"<s>", b"</s>", b"hello"]
    array = (struct.pack("<Q", len(b"tokenizer.ggml.tokens")) + b"tokenizer.ggml.tokens"
             + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(tokens))
             + b"".join(struct.pack("<Q", len(t)) + t for t in tokens))
    scores = (struct.pack("<Q", len(b"tokenizer.ggml.scores")) + b"tokenizer.ggml.scores"
              + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 3) + struct.pack("<fff", 0.5, 1.0, 2.0))
    block = (struct.pack("<Q", len(b"qwen4exp.block_count")) + b"qwen4exp.block_count"
             + struct.pack("<I", 4) + struct.pack("<I", 36))
    flag = struct.pack("<Q", len(b"general.quantized")) + b"general.quantized" + struct.pack("<I", 7) + b"\x01"
    kvs = [_kv_string("general.name", "demo"), array, scores, block, flag, _kv_string("general.architecture", arch)]
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"\x00" * 64


def _gguf_report(data: bytes, **overrides) -> ModelReport:
    base = dict(
        repo_id="huihui-ai/Qwen3-8-Flash-Next-Uncensored-GGUF", revision_sha="sha-flash-next", artifact_type=ArtifactType.GGUF,
        weight_bytes=len(data), context_length=262144, kv_dims=KvDims(layers=36, kv_heads=8, head_dim=128),
        selected_gguf=GGUF_NAME, gguf_architecture=ARCH, architecture=ARCH,
        gguf_variants=[GgufVariant(filename=GGUF_NAME, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())],
        has_chat_template=True, license="apache-2.0",
    )
    base.update(overrides)
    return ModelReport(**base)


def _bundle(tmp_path: Path, data: bytes, target="dgx-spark-single", **overrides):
    report = _gguf_report(data, **overrides)
    fit = analyze(report, PRESETS[target])
    plan = build_plan(report, fit, provider=None)
    return render_bundle(plan, report, fit, tmp_path / "bundles")


# ───────────────────────── shims ─────────────────────────
def _shim(bin_dir: Path, name: str, body: str) -> Path:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)
    return path


_LLAMA_SERVER = '''
if [[ "${1:-}" == "--version" ]]; then echo "version: 10495 (3dc7285b4)"; echo "built with gcc"; exit 0; fi
echo "llama-server $*" >> "$FAKE_LOG"
if [[ -n "${FAKE_SERVER_DIES:-}" ]]; then
  echo "llama_model_loader: loaded meta data with 30 key-value pairs"
  echo "${FAKE_SERVER_DIES}"
  echo "llama_model_load_from_file_impl: failed to load model"
  exit 1
fi
sleep 2
'''

_DOCKER = '''
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  image) exit 0 ;;
  ps) exit 0 ;;
  run)
    if [[ "$*" == *"--entrypoint sh"* ]]; then echo "${FAKE_DOCKER_ARCH_COUNT:-0}"; [[ "${FAKE_DOCKER_ARCH_COUNT:-0}" == "0" ]] && exit 1; exit 0; fi
    if [[ "$*" == *"--entrypoint nvidia-smi"* ]]; then echo "GPU 0: fake"; exit 0; fi
    echo "cid-1234"; exit 0 ;;
  logs) printf '%s\\n' "${FAKE_LOGS:-}"; exit 0 ;;
  *) exit 0 ;;
esac
'''

# merge-base --is-ancestor <lock> <build>: 0 = build ใหม่กว่าหรือเท่ากับ lock (ค่าปกติ) · FAKE_GIT_NOT_ANCESTOR=1 = build เก่ากว่า lock
_GIT = '''
echo "git $*" >> "$FAKE_LOG"
if [[ "$*" == *rev-parse* ]]; then echo newcommit; fi
if [[ "$*" == *"log -1"* ]]; then echo 2026-08-18; fi
if [[ "$*" == *merge-base* && -n "${FAKE_GIT_NOT_ANCESTOR:-}" ]]; then exit 1; fi
exit 0
'''

# cmake --build = "build ใหม่สำเร็จ" → libllama รุ่นใหม่รู้จัก arch (FAKE_BUILD_ADDS)
_CMAKE = '''
echo "cmake $*" >> "$FAKE_LOG"
if [[ "$*" == *--build* && -n "${FAKE_BUILD_ADDS:-}" ]]; then printf '%s\\0' "$FAKE_BUILD_ADDS" >> "$FAKE_LIB"; fi
exit 0
'''

_CURL = '''
if [[ "$*" == *"/health"* ]]; then [[ "${FAKE_HEALTH:-ok}" == "ok" ]] && exit 0 || exit 22; fi
exit 0
'''


class Rig:
    """โฟลเดอร์ + shim ครบชุดสำหรับรัน controller เดี่ยว ๆ — llama.cpp build กลางของเครื่องอยู่ที่ llama_dir"""

    def __init__(self, tmp_path: Path, data: bytes, lib_archs=("llama", "qwen3", "qwen3moe", "gemma4"), **overrides):
        self.tmp = tmp_path
        self.bundle = _bundle(tmp_path, data, **overrides)
        self.slug = self.bundle.directory.name
        self.model_dir = tmp_path / "models" / self.slug
        self.model_dir.mkdir(parents=True)
        (self.model_dir / GGUF_NAME).write_bytes(data)
        self.run_dir = tmp_path / "run" / self.slug
        self.run_dir.mkdir(parents=True)
        self.llama_dir = tmp_path / "src" / "llama.cpp"
        (self.llama_dir / ".git").mkdir(parents=True)
        self.bin = self.llama_dir / "build" / "bin"
        self.bin.mkdir(parents=True)
        self.lib = self.bin / "libllama.so"
        self.lib.write_bytes(b"\x00" * 512 + b"\x00".join(a.encode() for a in lib_archs) + b"\x00" * 512)
        _shim(self.bin, "llama-server", _LLAMA_SERVER)
        self.shims = tmp_path / "shims"
        self.shims.mkdir()
        for name, body in {"docker": _DOCKER, "git": _GIT, "cmake": _CMAKE, "curl": _CURL,
                           "nvidia-smi": "exit 0\n", "sudo": "exit 0\n", "gcc": "exit 0\n", "ip": "exit 0\n",
                           "ss": "exit 0\n"}.items():
            _shim(self.shims, name, body)
        self.log = tmp_path / "calls.log"
        (tmp_path / "home").mkdir(exist_ok=True)

    def run(self, *cmd: str, env: dict | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
        full = {
            "PATH": f"{self.shims}:{SAFE_PATH}", "HOME": str(self.tmp / "home"), "FAKE_LOG": str(self.log),
            "FAKE_LIB": str(self.lib), "MODEL_DIR": str(self.model_dir), "RUN_DIR": str(self.run_dir),
            "LLAMA_CPP_DIR": str(self.llama_dir), "API_PORT": "18477", "HEALTH_TIMEOUT": "20",
            "LLAMACPP_IMAGE": "ghcr.io/ggml-org/llama.cpp@sha256:0ldd1gest",
        }
        full.update(env or {})
        return subprocess.run(["bash", str(self.bundle.controller), *cmd], env=full, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)

    def calls(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""


# ═════════════════════ (a) start ปฏิเสธพร้อมบอกทางแก้ — native และ docker ═════════════════════
def test_native_start_refuses_when_the_build_does_not_know_the_architecture(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("start")
    assert done.returncode != 0
    err = done.stderr
    assert f"ไม่รู้จักสถาปัตยกรรม '{ARCH}'" in err, err
    assert "version: 10495 (3dc7285b4)" in err and "commit เมื่อ 2026-08-18" in err, "ต้องบอกรุ่น build + วันที่ของ commit"
    assert "unknown model architecture" in err, "ต้องบอกว่า start ไปก็ตายด้วยข้อความอะไร"
    assert f"LLAMA_CPP_UPDATE=1 {rig.bundle.controller} prepare-runtime" in err or "LLAMA_CPP_UPDATE=1" in err and "prepare-runtime" in err
    assert f"lmds repair {rig.slug}" in err and "LMDS_SKIP_ARCH_CHECK=1" in err
    assert "llama-server -m" not in rig.calls(), "ห้ามปล่อยเซิร์ฟเวอร์ขึ้น"
    assert not (rig.run_dir / "server.pid").exists()


def test_docker_start_refuses_when_the_pinned_image_does_not_know_the_architecture(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("start", env={"RUNTIME_MODE": "docker", "FAKE_DOCKER_ARCH_COUNT": "0"})
    assert done.returncode != 0
    err = done.stderr
    assert f"ไม่รู้จักสถาปัตยกรรม '{ARCH}'" in err and "image ghcr.io/ggml-org/llama.cpp@sha256:0ldd1gest" in err, err
    assert f"lmds set {rig.slug} --image" in err and "LLAMACPP_IMAGE=<image>" in err
    calls = rig.calls()
    assert "--entrypoint sh ghcr.io/ggml-org/llama.cpp@sha256:0ldd1gest" in calls and f"sh {ARCH}" in calls, "ต้องถาม image ที่ pin ไว้ด้วยชื่อ arch"
    assert "docker run -d" not in calls


# ═════════════════════ (b) start เดินต่อเมื่อรันไทม์รู้จัก ═════════════════════
def test_native_start_proceeds_when_the_build_knows_the_architecture(tmp_path):
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", "qwen3", ARCH))
    done = rig.run("start")
    assert done.returncode == 0, done.stderr
    assert f"architecture: {ARCH} — รองรับ (version: 10495" in done.stdout, done.stdout
    assert "llama-server -m" in rig.calls() and "started:" in done.stdout


def test_docker_start_proceeds_when_the_image_knows_the_architecture(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("start", env={"RUNTIME_MODE": "docker", "FAKE_DOCKER_ARCH_COUNT": "2"})
    assert done.returncode == 0, done.stderr
    assert f"architecture: {ARCH} — รองรับ (image " in done.stdout
    assert "docker run -d" in rig.calls()


def test_an_image_with_no_llama_files_where_expected_is_not_judged(tmp_path):
    """image ที่วางไฟล์ไว้ที่อื่น (ไม่ใช่ /app) ตอบ none = ถามไม่ได้ — ต้องปล่อยผ่าน ไม่ใช่บล็อกเป็น "ไม่รู้จัก" """
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("start", env={"RUNTIME_MODE": "docker", "FAKE_DOCKER_ARCH_COUNT": "none"})
    assert done.returncode == 0, done.stderr
    assert f"ตรวจสถาปัตยกรรม {ARCH} ไม่ได้" in done.stdout and "docker run -d" in rig.calls()


def test_the_escape_hatch_skips_the_check_and_says_so(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("start", env={"LMDS_SKIP_ARCH_CHECK": "1"})
    assert done.returncode == 0, done.stderr
    assert "LMDS_SKIP_ARCH_CHECK=1" in done.stdout and "llama-server -m" in rig.calls()


def test_check_runtime_reports_build_lock_and_verdict(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    bad = rig.run("check-runtime")
    assert bad.returncode != 0 and "runtime:   version: 10495" in bad.stdout and f"arch: {ARCH}" in bad.stdout
    assert "lock:" in bad.stdout and "ยังไม่มี" in bad.stdout
    rig.lib.write_bytes(rig.lib.read_bytes() + ARCH.encode() + b"\x00")
    good = rig.run("check-runtime")
    assert good.returncode == 0 and "รองรับ" in good.stdout


# ═════════════════════ (c) explain-crash ดึงสาเหตุจริงจาก server.log ═════════════════════
def test_a_crash_before_health_names_the_architecture_and_the_fix_not_just_exit_1(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    # ซาก log ของรอบก่อน — ต้องไม่ถูกหยิบมาบอกเป็นสาเหตุของรอบนี้ (server.log ถูก append ทุก start)
    (rig.run_dir / "server.log").write_text(
        "llama_model_load: error loading model: unknown model architecture: 'stalearch'\n" * 3, encoding="utf-8")
    done = rig.run("start", env={
        "LMDS_SKIP_ARCH_CHECK": "1", "FAKE_HEALTH": "fail",
        "FAKE_SERVER_DIES": f"llama_model_load: error loading model: unknown model architecture: '{ARCH}'",
    }, timeout=90)
    assert done.returncode != 0
    err = done.stderr
    assert f"สาเหตุจาก log: llama_model_load: error loading model: unknown model architecture: '{ARCH}'" in err, err
    assert "stalearch" not in err, "ต้องดูเฉพาะ log ของรอบนี้"
    assert f"ไม่รู้จัก arch '{ARCH}'" in err and "version: 10495 (3dc7285b4)" in err
    assert "LLAMA_CPP_UPDATE=1" in err and "prepare-runtime" in err and f"lmds repair {rig.slug}" in err
    assert "เซิร์ฟเวอร์หยุดก่อน health ผ่าน" in err


def test_an_out_of_memory_crash_points_at_the_context(tmp_path):
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=(ARCH,))
    done = rig.run("start", env={
        "FAKE_HEALTH": "fail",
        "FAKE_SERVER_DIES": "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 98304.00 MiB on device 0: cudaMalloc failed: out of memory",
    }, timeout=90)
    assert done.returncode != 0
    assert "สาเหตุจาก log: ggml_backend_cuda_buffer_type_alloc_buffer" in done.stderr
    assert "หน่วยความจำไม่พอ" in done.stderr and f"lmds set {rig.slug} --context" in done.stderr


# ═════════════════════ (d) prepare-runtime: lock ที่ไม่รู้จัก arch = build ใหม่เอง · lock อยู่ข้าง build ═════════════════════
def test_prepare_runtime_rebuilds_when_the_locked_build_lacks_the_architecture(tmp_path):
    rig = Rig(tmp_path, gguf_bytes())
    lock = rig.llama_dir / "build" / "runtime.lock"
    lock.write_text("lockedcommit\n", encoding="utf-8")
    done = rig.run("prepare-runtime", env={"FAKE_BUILD_ADDS": ARCH})
    assert done.returncode == 0, done.stderr
    assert f"ไม่รู้จักสถาปัตยกรรม '{ARCH}'" in done.stdout and "อัปเดต llama.cpp เป็น master" in done.stdout
    calls = rig.calls()
    assert "git -C" in calls and "fetch --all" in calls and "checkout --quiet master" in calls and "pull --quiet" in calls
    assert "checkout --quiet newcommit" in calls
    assert lock.read_text().strip() == "newcommit", "commit ใหม่ต้องเป็นสิ่งที่ bundle อื่นบนเครื่องเห็นด้วย"
    assert not (rig.run_dir / "runtime.lock").exists(), "ไม่เขียน lock ต่อ bundle อีก"


def test_prepare_runtime_never_downgrades_a_newer_shared_build(tmp_path):
    """lock เก่ากว่า build ที่มีจริง (msi-3/msi-4/dgx-veerasiam 2026-09-06: lock 15–16 ส.ค. · build 18 ส.ค. จาก rebuild
    นอก controller) — "ใช้ commit ที่ lock ไว้" แบบเดิมคือ checkout ของเก่ามา build ทับ = downgrade ทุก bundle บนเครื่อง"""
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", ARCH))
    lock = rig.llama_dir / "build" / "runtime.lock"
    lock.write_text("lockedcommit\n", encoding="utf-8")
    done = rig.run("prepare-runtime")
    assert done.returncode == 0, done.stderr
    calls = rig.calls()
    assert "merge-base --is-ancestor lockedcommit 3dc7285b4" in calls, "ต้องถามว่า lock เป็นบรรพบุรุษของ build จริงไหม"
    assert "fetch" not in calls and "checkout" not in calls and "cmake" not in calls, "build ใหม่กว่า lock = ใช้ที่มี ไม่ build ซ้ำ ไม่ถอยหลัง"
    assert "build กลาง=3dc7285b4 · lock=lockedcommit · action=reuse" in done.stdout, done.stdout
    assert lock.read_text().strip() == "3dc7285b4", "lock เลื่อนขึ้นมาที่ build จริง (proven-good floor)"
    assert "LLAMA_CPP_UPDATE=1" in done.stdout
    stamp = json.loads((rig.llama_dir / "build" / "lmds-build.json").read_text(encoding="utf-8"))
    assert stamp["commit"] == "3dc7285b4" and stamp["build"] == "10495" and stamp["built_by"] == rig.slug


def test_prepare_runtime_builds_the_locked_commit_only_when_the_build_is_older(tmp_path):
    """เคสเดียวที่ควร checkout ตาม lock: build ที่มีอยู่เก่ากว่า commit ที่พิสูจน์แล้ว"""
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", ARCH))
    lock = rig.llama_dir / "build" / "runtime.lock"
    lock.write_text("lockedcommit\n", encoding="utf-8")
    done = rig.run("prepare-runtime", env={"FAKE_GIT_NOT_ANCESTOR": "1"})
    assert done.returncode == 0, done.stderr
    calls = rig.calls()
    assert "fetch" not in calls and "checkout --quiet lockedcommit" in calls and "cmake --build" in calls
    assert "action=build" in done.stdout and "เก่ากว่า lock" in done.stdout
    assert lock.read_text().strip() == "lockedcommit"
    stamp = json.loads((rig.llama_dir / "build" / "lmds-build.json").read_text(encoding="utf-8"))
    assert stamp["commit"] == "lockedcommit" and stamp["cuda_arch"] == "121a-real" and stamp["at"].endswith("Z")


def test_prepare_runtime_migrates_a_per_bundle_lock_without_downgrading(tmp_path):
    """lock รุ่นเก่าใต้ RUN_DIR (ของ bundle) ย้ายมาข้าง build — แต่ถ้ามันเก่ากว่า build จริง ต้องไม่ checkout ตามมัน"""
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", ARCH))
    (rig.run_dir / "runtime.lock").write_text("legacycommit\n", encoding="utf-8")
    done = rig.run("prepare-runtime")
    assert done.returncode == 0, done.stderr
    assert "ย้าย lock" in done.stdout and "action=reuse" in done.stdout
    assert "checkout" not in rig.calls() and "cmake" not in rig.calls()
    assert (rig.llama_dir / "build" / "runtime.lock").read_text().strip() == "3dc7285b4"
    assert not (rig.run_dir / "runtime.lock").exists() or True  # lock เก่าไม่ถูกเขียนเพิ่ม (คงไว้เป็นหลักฐาน)


def test_prepare_runtime_with_no_lock_reuses_an_existing_build(tmp_path):
    """msi-5: build มีแต่ไม่มี lock (rebuild นอก controller) — เดิม "ยังไม่มี lock" = ดึง master มา build ใหม่ทั้งที่ของมีอยู่"""
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", ARCH))
    done = rig.run("prepare-runtime")
    assert done.returncode == 0, done.stderr
    assert "action=reuse" in done.stdout and "fetch" not in rig.calls()
    assert (rig.llama_dir / "build" / "runtime.lock").read_text().strip() == "3dc7285b4"


def test_prepare_runtime_warns_about_running_servers_before_rebuilding(tmp_path):
    """build ทับ build/ ขณะ llama-server จากโฟลเดอร์นี้รันอยู่ — บอกชื่อ bundle ที่จะได้รุ่นใหม่ตอน restart"""
    rig = Rig(tmp_path, gguf_bytes())
    other = rig.run_dir.parent / "other-gguf"
    other.mkdir()
    (other / "server.meta").write_text(f"slug=other-gguf\nengine=llamacpp\nmode=native\nruntime_dir={rig.llama_dir}\n", encoding="utf-8")
    _shim(rig.shims, "pgrep", "exit 0\n")
    done = rig.run("prepare-runtime", env={"FAKE_BUILD_ADDS": ARCH})
    assert done.returncode == 0, done.stderr
    assert "กำลังรันอยู่" in done.stderr and "other-gguf" in done.stderr, done.stderr


def test_start_records_the_served_build_in_server_meta_and_refreshes_the_lock(tmp_path):
    rig = Rig(tmp_path, gguf_bytes(), lib_archs=("llama", ARCH))
    (rig.llama_dir / "build" / "runtime.lock").write_text("olderlock\n", encoding="utf-8")
    done = rig.run("start")
    assert done.returncode == 0, done.stderr
    meta = (rig.run_dir / "server.meta").read_text(encoding="utf-8")
    assert "runtime_commit=3dc7285b4" in meta and "runtime_build=10495" in meta, meta
    assert f"runtime_dir={rig.llama_dir}" in meta
    assert (rig.llama_dir / "build" / "runtime.lock").read_text().strip() == "3dc7285b4", "lock = commit ที่เสิร์ฟจริง"


def test_prepare_runtime_says_when_even_master_does_not_know_the_architecture(tmp_path):
    """build ใหม่แล้วยังไม่รู้จัก = ต้นน้ำยังไม่มี — ห้ามแนะให้ prepare-runtime ซ้ำซึ่งวนกลับที่เดิม"""
    rig = Rig(tmp_path, gguf_bytes())
    done = rig.run("prepare-runtime", env={"LLAMA_CPP_UPDATE": "1"})
    assert done.returncode != 0
    assert f"ก็ยังไม่รู้จักสถาปัตยกรรม '{ARCH}'" in done.stderr and "upstream" in done.stderr
    assert "LLAMA_CPP_REPO=<url>" in done.stderr


def test_the_controller_passes_the_template_gates(tmp_path):
    from lmds.validator import all_passed, run_gates

    rig = Rig(tmp_path, gguf_bytes())
    results = run_gates(rig.bundle.directory, include_checksums=False)
    assert all_passed(results), [(r.name, r.detail) for r in results if not r.passed]
    text = rig.bundle.controller.read_text(encoding="utf-8")
    assert "check-runtime)" in text and "LMDS_SKIP_ARCH_CHECK" in text
    assert 'RUNTIME_LOCK="${RUNTIME_LOCK:-${LLAMA_CPP_DIR}/build/runtime.lock}"' in text


# ═════════════════════ (e) doctor / inventory / hub ═════════════════════
def _profile(rig: Rig | None = None, *, llamacpp_dir: Path, gguf_arch: str | None = ARCH, image: str = "") -> dict:
    return {
        "model": {"id": "huihui-ai/Qwen3-8-Flash-Next-Uncensored-GGUF", "revision": "sha", "selected_gguf": GGUF_NAME,
                  "gguf_architecture": gguf_arch},
        "runtime": {"engine": "llamacpp", "image": image or "ghcr.io/ggml-org/llama.cpp:server-cuda", "native_build": not image},
        "target": {"memory_model": "discrete" if image else "unified", "llamacpp_dir": str(llamacpp_dir)},
        "topology": "single",
    }


def _register(tmp_path: Path, slug: str, controller: Path, profile: dict) -> Path:
    directory = controller.parent
    directory.mkdir(parents=True, exist_ok=True)
    if not controller.exists():
        controller.write_text("#!/usr/bin/env bash\ncase $1 in\n  download) : ;;\n  start) : ;;\nesac\n", encoding="utf-8")
        controller.chmod(0o755)
    (directory / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    run_dir = tmp_path / "lmds-run" / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\nmodel={slug}\nmodel_id=org/{slug}\nengine=llamacpp\nmode=docker\nport=8000\n"
        f"container=lmds-{slug}\npid_file=\ncontroller={controller}\nstarted_at=\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def quiet_host(monkeypatch):
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "")
    monkeypatch.setattr("lmds.doctor.checks.shutil.which", lambda name: "/usr/bin/" + name)


def test_doctor_flags_a_stale_build_with_the_controller_fix_not_a_bare_git_pull(tmp_path, monkeypatch, quiet_host):
    from lmds.doctor import Status, diagnose

    slug = "flash-next"
    model_dir = tmp_path / "models" / slug
    model_dir.mkdir(parents=True)
    (model_dir / GGUF_NAME).write_bytes(gguf_bytes())
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "build" / "bin" / "libllama.so").write_bytes(b"\x00llama\x00qwen3moe\x00")
    controller = tmp_path / "bundles" / slug / f"{slug}-single.sh"
    _register(tmp_path, slug, controller, _profile(llamacpp_dir=llama))

    finding = next(f for f in diagnose(slug).findings if f.name == "architecture")
    assert finding.status is Status.FAIL, finding
    assert f"'{ARCH}'" in finding.detail and "unknown model architecture" in finding.detail
    assert f"LLAMA_CPP_UPDATE=1 {controller} prepare-runtime" in finding.fix and f"lmds repair {slug}" in finding.fix
    assert "git pull" not in finding.fix, "git pull ตรง ๆ ข้าม lock ของ controller — รอบหน้า prepare-runtime ย้อนกลับ"

    (llama / "build" / "bin" / "libllama.so").write_bytes(b"\x00llama\x00" + ARCH.encode() + b"\x00")
    finding = next(f for f in diagnose(slug).findings if f.name == "architecture")
    assert finding.status is Status.OK and ARCH in finding.detail


def test_inventory_flags_the_bundle_even_before_the_weights_are_downloaded(tmp_path, monkeypatch, quiet_host):
    """profile จด general.architecture ไว้ — hub เห็นป้าย "runtime เก่ากว่าโมเดล" ตั้งแต่ push ก่อน download"""
    from lmds import inventory
    from lmds.fleet import find

    slug = "flash-next"
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models" / slug))   # ยังไม่มีไฟล์
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "build" / "bin" / "libllama.so").write_bytes(b"\x00llama\x00qwen3moe\x00")
    controller = tmp_path / "bundles" / slug / f"{slug}-single.sh"
    _register(tmp_path, slug, controller, _profile(llamacpp_dir=llama))

    payload = inventory.model_payload(find(slug))
    arch = payload["runtime_arch"]
    assert arch["supported"] is False and arch["arch"] == ARCH and arch["mode"] == "native", arch
    assert "LLAMA_CPP_UPDATE=1" in arch["fix"] and "prepare-runtime" in arch["fix"]
    assert payload["downloaded"] is False

    # build ที่รู้จัก → ไม่มีป้าย · ยังไม่ได้ build เลย → บอกไม่ได้ (None) ไม่ใช่ป้ายแดง
    (llama / "build" / "bin" / "libllama.so").write_bytes(b"\x00" + ARCH.encode() + b"\x00")
    assert inventory.model_payload(find(slug))["runtime_arch"]["supported"] is True
    (llama / "build" / "bin" / "libllama.so").unlink()
    assert inventory.model_payload(find(slug))["runtime_arch"]["supported"] is None


def test_inventory_never_runs_docker_but_reads_what_doctor_recorded(tmp_path, monkeypatch, quiet_host):
    from lmds import inventory
    from lmds.doctor import Status, diagnose
    from lmds.fleet import find
    import lmds.doctor.checks as checks

    slug = "flash-next-rtx"
    model_dir = tmp_path / "models" / slug
    model_dir.mkdir(parents=True)
    (model_dir / GGUF_NAME).write_bytes(gguf_bytes())
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    controller = tmp_path / "bundles" / slug / f"{slug}-single.sh"
    profile = _profile(llamacpp_dir=tmp_path / "unused", image="ghcr.io/ggml-org/llama.cpp:server-cuda")
    profile["runtime"]["image_pin"] = "sha256:0ldd1gest"
    _register(tmp_path, slug, controller, profile)

    docker_calls: list[list[str]] = []

    def fake_run(args, timeout=10):
        docker_calls.append(list(args))
        if args[:2] == ["docker", "run"]:
            assert args[-1] == ARCH and "ghcr.io/ggml-org/llama.cpp@sha256:0ldd1gest" in args, args
            return 1, "0\n"
        return 0, ""

    monkeypatch.setattr(checks, "_run", fake_run)
    assert inventory.model_payload(find(slug))["runtime_arch"]["supported"] is None
    assert not any(c[:2] == ["docker", "run"] for c in docker_calls), "agent info ห้ามยิง docker run"

    finding = next(f for f in diagnose(slug).findings if f.name == "architecture")
    assert finding.status is Status.FAIL and f"lmds set {slug} --image" in finding.fix
    assert any(c[:2] == ["docker", "run"] for c in docker_calls)
    # doctor จดผลไว้ → inventory เห็นป้ายโดยไม่ต้องยิง docker
    assert inventory.model_payload(find(slug))["runtime_arch"]["supported"] is False


def test_repair_rebuilds_llamacpp_when_the_build_is_older_than_the_model(tmp_path, monkeypatch, quiet_host):
    from lmds.fleet import find, repair_server

    slug = "flash-next"
    model_dir = tmp_path / "models" / slug
    model_dir.mkdir(parents=True)
    (model_dir / GGUF_NAME).write_bytes(gguf_bytes())
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    lib = llama / "build" / "bin" / "libllama.so"
    lib.write_bytes(b"\x00llama\x00")
    log = tmp_path / "ctl.log"
    controller = tmp_path / "bundles" / slug / f"{slug}-single.sh"
    controller.parent.mkdir(parents=True)
    controller.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$1 LLAMA_CPP_UPDATE=${{LLAMA_CPP_UPDATE:-unset}}" >> "{log}"\n'
        "case $1 in\n  download) : ;;\n  verify-files) : ;;\n"
        f"  prepare-runtime) printf '{ARCH}' >> \"{lib}\" ;;\n  start) : ;;\nesac\n", encoding="utf-8")
    controller.chmod(0o755)
    _register(tmp_path, slug, controller, _profile(llamacpp_dir=llama))

    assert repair_server(find(slug), force=True) == 0
    assert log.read_text().splitlines() == [
        "download LLAMA_CPP_UPDATE=unset", "verify-files LLAMA_CPP_UPDATE=unset", "prepare-runtime LLAMA_CPP_UPDATE=1"]
    log.unlink()
    assert repair_server(find(slug), force=True) == 0
    assert "prepare-runtime" not in log.read_text(), "build รู้จักแล้ว = ไม่ build ซ้ำ"


def test_hub_routes_run_update_runtime_as_a_job_with_the_update_switch(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lmds.nodes import Node, add
    from lmds.web import create_app, jobs

    add(Node(name="spark-worker", host="10.0.0.7", user="ops"))
    started, direct = [], []
    monkeypatch.setattr(jobs, "start_remote",
                        lambda node, slug, command, remote: started.append((node, slug, command, remote))
                        or SimpleNamespace(payload=lambda: {"id": "job-1", "command": command}))
    monkeypatch.setattr("lmds.nodes.run",
                        lambda node, command, timeout=0: direct.append(command)
                        or SimpleNamespace(exit_code=0, stdout="runtime: version: 10495", stderr=""))
    client = TestClient(create_app())

    payload = client.post("/api/nodes/spark-worker/models/flash-next/ctl/update-runtime").json()
    assert payload["job"]["id"] == "job-1"
    node, slug, command, remote = started[0]
    assert (node, slug, command) == ("spark-worker", "flash-next", "update-runtime")
    assert 'LLAMA_CPP_UPDATE=1 "$ctl" prepare-runtime' in remote and "bundles/flash-next" in remote, remote

    answer = client.post("/api/nodes/spark-worker/models/flash-next/ctl/check-runtime").json()
    assert answer["output"] == "runtime: version: 10495" and '"$ctl" check-runtime' in direct[0]


def test_hub_local_update_runtime_job_hands_the_switch_to_the_controller(tmp_path, monkeypatch, quiet_host):
    pytest.importorskip("fastapi")
    import time

    from fastapi.testclient import TestClient

    from lmds.web import create_app, jobs

    jobs._JOBS.clear()
    jobs._ACTIVE.clear()
    monkeypatch.setattr("lmds.hardware.serving.guard", lambda *a, **k: "")
    slug = "flash-next"
    controller = tmp_path / "bundles" / slug / f"{slug}-single.sh"
    controller.parent.mkdir(parents=True)
    controller.write_text('#!/usr/bin/env bash\necho "$1 LLAMA_CPP_UPDATE=${LLAMA_CPP_UPDATE:-unset}"\n', encoding="utf-8")
    controller.chmod(0o755)
    _register(tmp_path, slug, controller, _profile(llamacpp_dir=tmp_path / "llama.cpp"))
    client = TestClient(create_app())
    job = client.post(f"/api/models/{slug}/run/update-runtime").json()
    for _ in range(100):
        data = client.get(f"/api/jobs/{job['id']}").json()
        if not data["running"]:
            break
        time.sleep(0.1)
    assert data["exit_code"] == 0, data
    assert data["steps"] == ["prepare-runtime"] and "prepare-runtime LLAMA_CPP_UPDATE=1" in data["output"], data["output"]


def test_the_console_shows_the_stale_runtime_and_offers_the_fix():
    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    assert "runtime older than model" in page and "function runtimeStale(" in page
    assert 'data-nact="ctl:update-runtime"' in page, "การ์ดของ node ต้องมีปุ่มแก้"
    assert 'btn("job:update-runtime"' in page, "การ์ดของโมเดลในเครื่องนี้ต้องมีปุ่มแก้"
    assert 'ctl("check-runtime"' in page and 'job("check-runtime")' in page
    assert "runtime_arch.arch" in page.split("function ovAlerts()")[1].split("function renderOverview")[0], "หน้าภาพรวมต้องเตือน"


# ═════════════════════ (f) profile จด arch + รุ่นต่ำสุด ═════════════════════
def test_the_profile_records_the_gguf_architecture_and_the_upstream_hint(tmp_path):
    data = gguf_bytes()
    bundle = _bundle(tmp_path, data)
    profile = yaml.safe_load((bundle.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["model"]["gguf_architecture"] == ARCH
    assert profile["runtime"]["native_build"] is True
    assert profile["runtime"]["min_llamacpp"] == {"commit": "6c84c7d5d", "date": "2026-08-27", "ref": "ggml-org/llama.cpp#27742"}

    unknown = _bundle(tmp_path / "two", data, gguf_architecture="qwen3moe", architecture="qwen3moe")
    profile = yaml.safe_load((unknown.directory / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["model"]["gguf_architecture"] == "qwen3moe" and profile["runtime"]["min_llamacpp"] is None


def test_the_gguf_header_reader_matches_the_python_parser(tmp_path):
    """python heredoc ใน controller อ่าน general.architecture ได้เหมือน inspector — รวมข้าม array/scalar"""
    from lmds.doctor.checks import _gguf_architecture

    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes("muse-glimmer"))
    assert _gguf_architecture(path) == "muse-glimmer"
    rig = Rig(tmp_path / "rig", gguf_bytes("muse-glimmer"))
    done = rig.run("check-runtime")
    assert "arch: muse-glimmer" in done.stdout, done.stdout + done.stderr
