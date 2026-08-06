"""เทส stacked (multi-node) deployment — port จาก reference v8.2 ที่ผ่านการทดสอบจริง

ครอบคลุม: planner emit STACKED, harden บังคับ topology จาก target, render controller
multi-node ที่ผ่าน bash -n + quality gates ครบ, กัน GGUF+stacked, และ gate ปิดช่องโหว่
"stacked profile แต่ controller เป็น single-node"
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

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
    # worker-first: worker ทุกตัว (rank 1..N-1, headless) ต้องขึ้นก่อน head (rank 0)
    assert "--node-rank ${rank} --host 127.0.0.1 --port 18000 --headless" in text
    assert "--node-rank 0" in text
    assert text.index("--node-rank ${rank}") < text.index("--node-rank 0")


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


def test_controller_derives_the_roce_hca(tmp_path):
    """ไม่ตั้ง NCCL_IB_HCA แล้ว NCCL ตกไปใช้ TCP — สาย 200G ทำงานเท่าอีเทอร์เน็ตธรรมดาแบบเงียบ ๆ"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert "detect_hca_for_interface()" in text
    # ค่าเริ่มต้นต้องเป็น sysfs ของจริง — ตัวแปรมีไว้ให้เทสชี้ไป tree ปลอมเท่านั้น
    assert 'INFINIBAND_ROOT="${INFINIBAND_ROOT:-/sys/class/infiniband}"' in text
    assert '[[ -n "$NCCL_IB_HCA" ]] && return 0' in text


def test_four_node_target_renders_four_workers(tmp_path):
    """เพิ่มเครื่องเป็น 4 ต้องได้ NNODES/TP/รายชื่อ worker ครบ ไม่ใช่แค่ตัวเลขเปลี่ยน"""
    from lmds.brain import build_plan as _build_plan
    from lmds.fit import PRESETS as _PRESETS, analyze as _analyze
    from lmds.generator import render_bundle as _render

    report = big_safetensors()
    fit = _analyze(report, _PRESETS["dgx-spark-stacked-4"])
    bundle = _render(_build_plan(report, fit, provider=None), report, fit, tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert 'NNODES="${NNODES:-4}"' in text
    assert 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"' in text
    # ค่าเริ่มต้นต้องมี worker ครบสามตัว ไม่ใช่ตัวเดียวแล้วเงียบ ๆ รันไม่ครบ
    assert "10.100.152.2 10.100.152.3 10.100.152.4" in text
    assert subprocess.run(["bash", "-n", str(bundle.controller)]).returncode == 0


def test_two_node_bundle_keeps_the_single_worker_default(tmp_path):
    """bundle เดิมต้องไม่เปลี่ยนพฤติกรรม — WORKER_IPS ตกมาที่ WORKER_IP ตัวเดียว"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")
    assert 'WORKER_IPS="${WORKER_IPS:-$WORKER_IP}"' in text


def test_every_worker_step_iterates_over_the_list(tmp_path):
    """ขั้นตอนที่แตะ worker ต้องวนทุกเครื่อง ไม่งั้น 4 เครื่องจะทำงานแค่เครื่องแรก"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    # sync/verify/start/stop/status/logs/prepare-runtime ต้องมีลูปของตัวเอง
    assert text.count("for wip in $WORKER_IPS; do") >= 8
    # ssh_worker เหลือไว้ได้เฉพาะงานที่อ้างถึง worker ตัวแรกโดยธรรมชาติ
    assert 'ssh_worker() { ssh_at "$WORKER_IP" "$@"; }' in text


def test_head_container_command_is_not_double_invoked(tmp_path):
    """เคสจริง: `docker "${hrun[@]}"` ทั้งที่ array ขึ้นต้นด้วย docker → `docker docker run -d`
    แล้ว head ไม่เคยขึ้นเลย · bash -n ไม่จับ เพราะเป็น syntax ที่ถูกต้อง"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert 'docker "${hrun[@]}"' not in text
    assert '"${hrun[@]}"' in text
    # worker ยิงผ่าน ssh เป็นสตริง จึงต้องคงรูปเดิมไว้
    assert 'ssh_at "$wip" "$(printf \'%q \' "${wrun[@]}")"' in text


def test_gpu_util_is_settable_and_validated(tmp_path):
    """unified memory ชน OOM ง่ายกว่าการ์ดแยก — ต้องปรับได้โดยไม่ต้องแก้ไฟล์ และต้องกันค่าพัง"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert "--gpu-util)" in text and "--gpu-util=*)" in text
    # bash เทียบทศนิยมไม่ได้ — ต้องใช้ awk ไม่ใช่ (( )) ที่จะตัดเป็น 0 เงียบ ๆ
    assert "awk -v v=\"$GPU_MEMORY_UTILIZATION\"" in text


def test_image_lock_is_per_bundle_not_per_machine(tmp_path):
    """เครื่องเดียวรัน stacked ได้หลายตัวและใช้คนละ image (DeepSeek V4 ต้องใช้ build เฉพาะ)
    — ล็อกร่วมกันทำให้ตัวที่สอง start ไม่ได้ด้วย 'image ต่างจากที่ lock ไว้'"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert ".lmds-stacked-image-id" not in text
    assert "IMAGE_LOCK_FILE=" in text
    assert "deepseek-v4-flash-nvfp4" in text.split("IMAGE_LOCK_FILE=")[1].split("\n")[0]


def test_container_hub_cache_handles_both_hf_layouts(tmp_path):
    """HF cache มีสองเลย์เอาต์ ($HF_HOME/hub/models--X และ $HF_HOME/models--X)
    เราตรวจเจอทั้งคู่ แต่ vLLM ในคอนเทนเนอร์มองแค่ hub/ — ต้องบอก HF_HUB_CACHE ให้ตรง
    ไม่งั้นได้ LocalEntryNotFoundError ทั้งที่ verify-files บอกว่าไฟล์ครบ"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = pathlib.Path(bundle.controller).read_text(encoding="utf-8")

    assert "_container_hub_cache()" in text
    # ต้องส่งให้ทั้ง head (docker -e) และ worker (export ในสคริปต์)
    assert '-e "HF_HUB_CACHE=$(_container_hub_cache "$HF_HOME")"' in text
    assert 'export HF_HUB_CACHE=$(_container_hub_cache "$WORKER_HF_HOME")' in text


def test_stacked_uses_every_active_roce_link(tmp_path):
    """ConnectX ใบเดียวมีสองพอร์ตที่ต่อสายพร้อมกันได้ — บอก NCCL ตัวเดียวคือใช้สายเส้นเดียว

    เคสจริงบน DGX Spark: rocep1s0f0 กับ roceP2p1s0f0 ขึ้นทั้งคู่ที่ 200 Gb/s
    ของเดิมผูก HCA กับ NCCL_SOCKET_IFNAME ตัวเดียวแล้ว return ทันที = ได้ครึ่งเดียว
    โดยไม่มีอะไรฟ้อง เพราะงาน "ก็รันได้"
    """
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "detect_active_hcas()" in text
    # ต้องกรองเฉพาะเส้นที่ขึ้นจริงและเร็วพอ ไม่งั้น NCCL ไปลองเส้นที่ตาย
    assert "operstate" in text
    assert "NCCL_HCA_MIN_SPEED_MBPS" in text
    # ยังต้องมีทางถอยเมื่อ driver ไม่เขียน speed
    assert "detect_hca_for_interface" in text


def test_stacked_hca_detection_joins_devices_with_commas(tmp_path):
    """NCCL_IB_HCA รับหลายตัวคั่นด้วยจุลภาค — ถ้า join ผิดจะกลายเป็นชื่อเดียวที่ไม่มีอยู่จริง"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    body = text.split("detect_active_hcas()", 1)[1].split("\n_resolve_hca", 1)[0]
    assert "local IFS=," in body
    assert '${found[*]}' in body


# ── การเดินสายแบบต่าง ๆ: รันฟังก์ชันในสคริปต์จริง ไม่ใช่ grep ข้อความ ────────────
def _fake_infiniband(root, devices):
    """สร้าง /sys/class/infiniband ปลอม: {ชื่อ RoCE: (ชื่อ netdev, operstate, speed)}"""
    for name, (netdev, state, speed) in devices.items():
        target = root / name / "device" / "net" / netdev
        target.mkdir(parents=True)
        (target / "operstate").write_text(state + "\n", encoding="utf-8")
        (target / "speed").write_text(f"{speed}\n", encoding="utf-8")
    return root


def _call_controller_fn(controller, fn: str, ib_root) -> subprocess.CompletedProcess:
    """source สคริปต์ controller แล้วเรียกฟังก์ชันเดียว — ไม่ให้ dispatch ท้ายไฟล์ทำงาน

    ต้องรันของจริง ไม่ใช่ grep เพราะเคสที่พังคือ *ตรรกะ* ของการเลือก HCA
    (ข้อความอยู่ครบแต่เลือกผิดตัวก็ยังผ่าน grep)
    """
    script = (
        f'INFINIBAND_ROOT={ib_root} '
        f'bash -c \'set -e; source "{controller}" >/dev/null 2>&1 || true; {fn}\''
    )
    return subprocess.run(script, shell=True, capture_output=True, text=True)


@pytest.mark.skipif(sys.platform == "win32", reason="ต้องมี bash + sysfs layout")
def test_single_cable_reports_both_twins(tmp_path):
    """สายเส้นเดียว = RoCE คู่แฝดสองตัว — บอก NCCL ตัวเดียวคือได้แบนด์วิดท์ครึ่งเดียว

    ผังของจริงจาก spark1: f0 ทั้งคู่ขึ้น (สายเสียบพอร์ตเดียว) ส่วน f1 ทั้งคู่ลง
    """
    bundle, _, _ = _stacked_bundle(tmp_path)
    ib_root = _fake_infiniband(tmp_path / "ib-one-cable", {
        "rocep1s0f0": ("enp1s0f0np0", "up", 200000),
        "roceP2p1s0f0": ("enP2p1s0f0np0", "up", 200000),
        "rocep1s0f1": ("enp1s0f1np1", "down", -1),
        "roceP2p1s0f1": ("enP2p1s0f1np1", "down", -1),
    })
    result = _call_controller_fn(bundle.controller, "detect_active_hcas", ib_root)
    assert sorted(result.stdout.strip().split(",")) == ["roceP2p1s0f0", "rocep1s0f0"]


def test_local_link_count_never_injects_topology_specific_nccl_env(tmp_path):
    """จำนวนลิงก์ local ไม่ยืนยัน topology จึงห้ามฉีด env ของ mesh อัตโนมัติ"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    text = bundle.controller.read_text(encoding="utf-8")
    assert "is_mesh_fabric" not in text
    assert "NCCL_IB_SUBNET_AWARE_ROUTING" not in text
    assert "NCCL_NET_PLUGIN=none" not in text


@pytest.mark.skipif(sys.platform == "win32", reason="ต้องมี bash + sysfs layout")
def test_links_that_are_down_are_never_offered_to_nccl(tmp_path):
    """สายที่ไม่ได้เสียบยังโผล่ใน sysfs ครบ — ใส่ให้ NCCL แล้วมันไปลองเส้นที่ตาย"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    ib_root = _fake_infiniband(tmp_path / "ib-down", {
        "rocep1s0f0": ("enp1s0f0np0", "down", -1),
        "roceP2p1s0f0": ("enP2p1s0f0np0", "down", -1),
    })
    assert _call_controller_fn(bundle.controller, "detect_active_hcas", ib_root).stdout.strip() == ""


@pytest.mark.skipif(sys.platform == "win32", reason="ต้องมี bash + sysfs layout")
def test_slow_links_are_skipped(tmp_path):
    """การ์ด 1G ที่บังเอิญมี RoCE ไม่ควรถูกเลือก — NCCL จะวิ่งช้าที่สุดตามเส้นที่ช้าที่สุด"""
    bundle, _, _ = _stacked_bundle(tmp_path)
    ib_root = _fake_infiniband(tmp_path / "ib-slow", {
        "rocep1s0f0": ("enp1s0f0np0", "up", 200000),
        "rocesomething": ("eth9", "up", 1000),
    })
    assert _call_controller_fn(bundle.controller, "detect_active_hcas", ib_root).stdout.strip() \
        == "rocep1s0f0"
