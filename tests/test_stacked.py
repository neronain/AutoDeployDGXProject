"""เทส stacked (multi-node) deployment — port จาก reference v8.2 ที่ผ่านการทดสอบจริง

ครอบคลุม: planner emit STACKED, harden บังคับ topology จาก target, render controller
multi-node ที่ผ่าน bash -n + quality gates ครบ, กัน GGUF+stacked, และ gate ปิดช่องโหว่
"stacked profile แต่ controller เป็น single-node"
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from lmds.brain import build_plan
from lmds.brain.orchestrator import harden_plan
from lmds.brain.plan_schema import Topology
from lmds.brain.rulebased import rule_based_plan, topology_for_target
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport
from lmds.validator import all_passed, run_gates
from lmds.validator.gates import gate_stacked_contract

REQUIRED_FLAGS = ["--context", "--port", "--bind", "--advertise-ip", "--interface",
                  "--client-input", "--client-output"]
REQUIRED_COMMANDS = ["download", "verify-files", "start", "stop", "restart", "status",
                     "logs", "client-config", "network-info"]
MULTINODE_MARKERS = ["--nnodes", "--node-rank", "--headless", "--distributed-executor-backend",
                     "sync-worker)", "verify-worker)", "prepare-runtime)", "ssh_worker"]


def big_safetensors(**overrides) -> ModelReport:
    base = dict(
        repo_id="nvidia/DeepSeek-V4-Flash-NVFP4",
        revision_sha="sha-stacked-abc",
        artifact_type=ArtifactType.SAFETENSORS,
        weight_bytes=168 * GIB,
        shard_count=46,
        context_length=131072,
        kv_dims=KvDims(layers=61, kv_heads=128, head_dim=128),
        license="mit",
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def gguf_report(**overrides) -> ModelReport:
    base = dict(
        repo_id="unsloth/Qwen3-8B-GGUF",
        revision_sha="sha-gguf",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=5 * GIB,
        selected_gguf="Qwen3-8B-Q4_K_M.gguf",
        has_chat_template=True,
    )
    base.update(overrides)
    return ModelReport(**base)


def _stacked_bundle(tmp_path, report=None):
    report = report or big_safetensors()
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    return render_bundle(plan, report, fit, tmp_path), plan, fit


# ── topology mapping ────────────────────────────────────────────────
def test_topology_for_target_maps_correctly():
    assert topology_for_target("dgx-spark-stacked") is Topology.STACKED
    assert topology_for_target("rtx-pro-4000-dual") is Topology.MULTI_GPU
    assert topology_for_target("this-machine-multi") is Topology.MULTI_GPU
    assert topology_for_target("dgx-spark-single") is Topology.SINGLE
    assert topology_for_target("rtx-5090") is Topology.SINGLE


def test_rule_based_emits_stacked_for_stacked_target():
    report = big_safetensors()
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = rule_based_plan(report, fit)
    assert plan.topology is Topology.STACKED


def test_harden_forces_topology_from_target_over_llm():
    """LLM ตอบ topology ผิด → harden ต้องบังคับกลับตาม target เสมอ"""
    report = big_safetensors()
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = rule_based_plan(report, fit)
    plan.topology = Topology.SINGLE  # จำลอง LLM ตอบผิด
    hardened = harden_plan(plan, report, fit)
    assert hardened.topology is Topology.STACKED
    assert any("topology" in w for w in hardened.warnings)


# ── rendering ───────────────────────────────────────────────────────
def test_stacked_controller_name_and_bash_syntax(tmp_path):
    bundle, _, _ = _stacked_bundle(tmp_path)
    assert bundle.controller.name.endswith("-stacked.sh")
    result = subprocess.run(["bash", "-n", str(bundle.controller)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stacked_controller_has_multinode_machinery(tmp_path):
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    for marker in MULTINODE_MARKERS:
        assert marker in text, f"ขาด marker multi-node: {marker}"
    # worker-first: --headless (worker) + --node-rank 0/1
    assert "--node-rank 1" in text and "--node-rank 0" in text


def test_stacked_controller_satisfies_single_contract(tmp_path):
    """ต้องผ่าน controller contract เดิม (7 flags + 9 commands) เพื่อให้ lmds validate ทำงานเหมือนกัน"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    for flag in REQUIRED_FLAGS:
        assert flag + ")" in text, f"ขาด flag {flag}"
    for command in REQUIRED_COMMANDS:
        assert f"{command})" in text, f"ขาดคำสั่ง {command}"
    assert "set -Eeuo pipefail" in text


def test_stacked_bundle_passes_all_gates(tmp_path):
    from lmds.packager import write_checksums

    bundle, _, _ = _stacked_bundle(tmp_path)
    write_checksums(bundle.directory)
    results = run_gates(bundle.directory, include_checksums=True)
    failed = [f"{r.name}: {r.detail}" for r in results if not r.passed]
    assert all_passed(results), "gates ล้มเหลว: " + "; ".join(failed)


def test_stacked_shard_count_and_container_naming(tmp_path):
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert 'SHARD_COUNT:-46' in text  # จาก report.shard_count
    assert "lmds-" in text and "-head" in text and "-worker" in text  # container ตามแบบ fleet


def test_stacked_readme_first_run_order(tmp_path):
    bundle, _, _ = _stacked_bundle(tmp_path)
    readme = (bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "sync-worker" in readme and "verify-worker" in readme
    assert "prepare-runtime" in readme
    assert "passwordless SSH" in readme
    # ลำดับถูก: prepare-runtime มาก่อน sync-worker
    assert readme.index("prepare-runtime") < readme.index("sync-worker")


# ── guards ──────────────────────────────────────────────────────────
def test_gguf_stacked_rejected(tmp_path):
    report = gguf_report()
    fit = analyze(report, PRESETS["dgx-spark-stacked"])
    plan = build_plan(report, fit, provider=None)
    assert plan.topology is Topology.STACKED
    with pytest.raises(ValueError, match="stacked"):
        render_bundle(plan, report, fit, tmp_path)


def test_gate_catches_stacked_profile_with_single_node_controller(tmp_path):
    """ปิดช่องโหว่ 'validated ปลอม': profile บอก stacked แต่ controller เป็น single-node"""
    d = tmp_path / "fake"
    d.mkdir()
    (d / "MODEL_PROFILE.yaml").write_text("topology: stacked\nmodel:\n  id: x\n", encoding="utf-8")
    # single-node script — ไม่มี nnodes/node-rank/sync-worker
    (d / "ctl.sh").write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\ncase \"$1\" in start) echo hi;; esac\n",
        encoding="utf-8",
    )
    result = gate_stacked_contract(d)
    assert not result.passed
    assert "multi-node" in result.detail


def test_gate_passes_for_nonstacked_profile(tmp_path):
    d = tmp_path / "single"
    d.mkdir()
    (d / "MODEL_PROFILE.yaml").write_text("topology: single\n", encoding="utf-8")
    (d / "ctl.sh").write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n", encoding="utf-8")
    assert gate_stacked_contract(d).passed


def test_stacked_has_shard_size_check_and_security_warning(tmp_path):
    """stacked ต้องตรวจ shard เท่าฝั่ง single — มีขั้น rsync ข้ามเครื่องเพิ่มอีกจุดที่ไฟล์ขาดได้"""
    from lmds.inspector.report import ShardFile

    report = big_safetensors()
    report.safetensor_shards = [
        ShardFile(filename="model-00001-of-00002.safetensors", size_bytes=80_000_000_000),
        ShardFile(filename="model-00002-of-00002.safetensors", size_bytes=79_000_000_000),
    ]
    bundle, _, _ = _stacked_bundle(tmp_path, report=report)
    script = bundle.controller.read_text(encoding="utf-8")

    assert "SHARD_FILES=(" in script
    assert "80000000000" in script
    assert "ขนาดไม่ตรง" in script
    assert "warn_open_endpoint" in script
    assert "API TOKEN (authentication)" in script
    assert "Authorization: Bearer" in script

    result = subprocess.run(["bash", "-n", str(bundle.controller)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stacked_repairs_cache_permissions(tmp_path):
    """เทียบ reference v8.2 ("permission-safe"): docker เคยสร้าง cache เป็น root มาก่อน
    รอบถัดไป user เขียนไม่ได้ แล้ว start ล้มแบบไล่สาเหตุยาก — ต้องซ่อมให้เอง
    """
    bundle, _, _ = _stacked_bundle(tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert "_ensure_local_owned_dir()" in script
    assert "_ensure_worker_owned_dir()" in script
    assert "chown -R" in script
    # ต้องถูกเรียกจริงกับ cache ทุกก้อน ไม่ใช่ประกาศทิ้งไว้
    for target in ('"$FLASHINFER_CACHE"', '"$head_fi"', '"$VLLM_CACHE"'):
        assert f"_ensure_local_owned_dir {target}" in script
    assert '_ensure_worker_owned_dir "$worker_fi"' in script


def test_stacked_releases_shared_memory_on_stop(tmp_path):
    """mp backend ทิ้ง /dev/shm ไว้ — ไม่เก็บกวาด start รอบหน้าชนของเก่า (มาจาก v8.2)"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert "/dev/shm/psm_*" in script
    assert "/dev/shm/sem.mp-*" in script


def test_stacked_port_check_works_with_host_networking(tmp_path):
    """container ใช้ --network host จึงไม่ publish port — `docker ps --filter publish=`
    จับไม่เจอเลย ต้องดู listening socket จริง
    """
    bundle, _, _ = _stacked_bundle(tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert "ss -tln" in script
    assert 'docker ps --filter "publish=' not in script


def test_stacked_can_recover_from_stale_flashinfer_cache(tmp_path):
    """cache JIT ค้างจาก image เก่า = start พังโดยไม่มีทางกู้ถ้าไม่มีคำสั่งนี้"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    script = bundle.controller.read_text(encoding="utf-8")
    assert "clear_fi_cache()" in script
    assert "clear-fi-cache)" in script
    assert "props)" in script


def test_controller_reads_cluster_env_before_defaults(tmp_path):
    """hub เขียน cluster.env ให้แล้ว controller ต้องใช้เลย ไม่ใช่ถาม IP ซ้ำตอน start"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    env_line = text.index('CLUSTER_ENV="${CLUSTER_ENV:-')
    master_line = text.index('MASTER_IP="${MASTER_IP:-')
    # ต้องอ่านไฟล์ก่อนตั้ง default ไม่งั้นค่าใน cluster.env ไม่มีผล
    assert env_line < master_line
    assert 'set -a; . "$CLUSTER_ENV"; set +a' in text
    assert 'if [[ -f "$CLUSTER_ENV" ]]; then' in text


def test_controller_derives_the_interface_from_the_cluster_ip(tmp_path):
    """ชื่อ interface บน DGX Spark ยาวและต่างกันทุกพอร์ต — ให้สคริปต์หาเองจาก IP ดีกว่าให้คนพิมพ์"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert "detect_interface()" in text
    assert "detect_worker_interface()" in text
    # ค่าที่ผู้ใช้ตั้งมาเองต้องชนะการตรวจอัตโนมัติเสมอ
    assert '[[ -n "$NCCL_SOCKET_IFNAME" ]] && return 0' in text


def test_controller_refuses_to_start_on_the_wrong_machine(tmp_path):
    """รันสคริปต์ head ผิดเครื่องต้องตายทันที ไม่ใช่ไปตายตอน NCCL init ที่อ่านไม่รู้เรื่อง"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert "check_running_on_master" in text
    assert text.index("check_running_on_master\n") > text.index("start() {")


def test_fabric_env_covers_ucx_and_ompi(tmp_path):
    """UCX/OMPI เลือกเส้นเองแยกจาก NCCL — ไม่บอกด้วยจะหลุดไปใช้ management NIC"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    for key in ("NCCL_SOCKET_IFNAME=", "GLOO_SOCKET_IFNAME=", "TP_SOCKET_IFNAME=",
                "UCX_NET_DEVICES=", "OMPI_MCA_btl_tcp_if_include="):
        assert key in text, key
