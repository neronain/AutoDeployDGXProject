"""สิ่งที่ container ถูกสั่งรันมา ต้องกลับไปอยู่ใน `docker run` ที่ adopt เขียน — หรือถูกบอกว่าไม่ได้ใส่

audit 2026-10-06 (ฟลีต TKC มี container ของลูกค้า 3 ตัวที่ adopt ไว้: `coder-next` · `gemma4-31b-vllm`
· `vllm-gemma4`): adopt เขียนแค่ controller แต่ `stop` ของ controller นั้นทำ `docker rm -f` และ
`start` รัน `docker run` ที่ประกอบขึ้นใหม่ — อะไรที่ adopt ไม่ได้ยกมา จึงหายในวันแรกที่มีคนกด restart
โดยไม่มีอะไรฟ้อง · สองข้อที่หลุดไปก่อนหน้า (`--gpus all` 965eb56 · HEALTHCHECK d7c1786) เป็นโรคเดียวกัน
และยังเหลืออีกทั้งกอง:

  - `HostConfig.Mounts` (`--mount type=bind` · volume แบบ long syntax ของ compose) → โฟลเดอร์โมเดลหาย
  - `DeviceRequests` ที่เจาะจง GPU (`--gpus device=1`) → กลายเป็น `--gpus all` ไปทับ GPU ของคนอื่น
  - `--ulimit` `--cap-add` `--device` `--user` `--workdir` `--add-host` `--memory` `--cpuset-cpus`
    `--security-opt` `--tmpfs` `--group-add` label และ restart policy (ถูกบังคับเป็น unless-stopped)
  - env ทุกตัวที่ไม่ขึ้นต้นตามรายการที่เดาไว้ (`NVIDIA_VISIBLE_DEVICES` `OMP_NUM_THREADS`
    `PYTORCH_CUDA_ALLOC_CONF` `HTTPS_PROXY` `TRANSFORMERS_OFFLINE`)
  - host binding ตัวที่สองของพอร์ตเดียวกัน และ `/udp`

หลักที่แก้: **คำสั่งที่สร้างใหม่ต้องเท่ากับที่ container ถูกสั่งรันมา หรือ adopt ต้องบอกว่าตัวไหนที่ทำซ้ำไม่ได้**
ห้ามทิ้งเงียบ ๆ · เทสชุดนี้รันสคริปต์จริงใต้ bash แล้วอ่าน argv ที่ docker (ปลอม) ได้รับ
"""

import shutil

import pytest

from tests.adopt_fakes import (
    adopt_mod, command_after_image, container_payload, has_option, image_payload, inspected,
    started, values_of,
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")

# สิ่งที่ลูกค้าพิมพ์ตอนสั่งรัน (หรือเขียนไว้ใน compose) — ทุกบรรทัดคือของที่หายไปก่อนแก้
COMPOSE_STYLE = dict(
    env=["HF_TOKEN=hf_abcdefghijklmnopqrstuvwx", "NVIDIA_VISIBLE_DEVICES=1", "OMP_NUM_THREADS=8",
         "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "HTTPS_PROXY=http://proxy.internal:3128",
         "TRANSFORMERS_OFFLINE=1", "VLLM_LOGGING_LEVEL=INFO"],
    config={"User": "1000:1000", "WorkingDir": "/workspace",
            "Labels": {"maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
                       "org.opencontainers.image.ref.name": "ubuntu",
                       "com.docker.compose.project": "llm", "com.docker.compose.service": "vllm"}},
    host={
        "Binds": ["/data/hf:/root/.cache/huggingface:ro"],
        "Mounts": [{"Type": "bind", "Source": "/data/models", "Target": "/models", "ReadOnly": True}],
        "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8001"}]},
        "DeviceRequests": [{"Driver": "", "Count": 0, "DeviceIDs": ["1"],
                            "Capabilities": [["gpu"]], "Options": {}}],
        "IpcMode": "host", "ShmSize": 17179869184,
        "Ulimits": [{"Name": "memlock", "Hard": -1, "Soft": -1},
                    {"Name": "stack", "Hard": 67108864, "Soft": 67108864}],
        "RestartPolicy": {"Name": "on-failure", "MaximumRetryCount": 3},
        "CapAdd": ["SYS_NICE"], "ExtraHosts": ["hub.internal:10.0.0.5"],
        "Devices": [{"PathOnHost": "/dev/infiniband/uverbs0",
                     "PathInContainer": "/dev/infiniband/uverbs0", "CgroupPermissions": "rwm"}],
        "Memory": 107374182400, "MemorySwap": 214748364800, "CpusetCpus": "0-15",
        "SecurityOpt": ["seccomp=unconfined"], "GroupAdd": ["video"], "Tmpfs": {"/tmp": "size=8g"},
    },
)


def _run(tmp_path, **over):
    adopted = inspected(container_payload(**{**COMPOSE_STYLE, **over}))
    return adopted, started(adopt_mod.render_controller(adopted, "vllm-gemma4"), tmp_path)


# ── mount / GPU ───────────────────────────────────────────────────────────────────────────
def test_a_bind_given_with_mount_syntax_is_still_mounted_after_restart(tmp_path):
    """`--mount type=bind,…` และ volume แบบ long syntax ของ compose ไปอยู่ใน HostConfig.Mounts ไม่ใช่ Binds
    — อ่านแต่ Binds = โฟลเดอร์โมเดลหายทั้งก้อน แล้ว vLLM ตายด้วย "model path does not exist" ตอน restart"""
    _, argv = _run(tmp_path)
    assert has_option(argv, "--volume", "/data/hf:/root/.cache/huggingface:ro"), "ของเดิมที่เคยยกมาได้ต้องยังอยู่"
    assert has_option(argv, "--mount", "type=bind,source=/data/models,target=/models,readonly")


def test_a_named_volume_and_a_tmpfs_mount_come_back_as_they_were(tmp_path):
    _, argv = _run(tmp_path, host={**COMPOSE_STYLE["host"], "Mounts": [
        {"Type": "volume", "Source": "llm_hf-cache", "Target": "/root/.cache/huggingface"},
        {"Type": "tmpfs", "Target": "/scratch", "TmpfsOptions": {"SizeBytes": 1073741824, "Mode": 0o1777}},
        {"Type": "bind", "Source": "/srv/my models,v2", "Target": "/models",
         "BindOptions": {"Propagation": "rslave"}},
    ]})
    mounts = values_of(argv, "--mount")
    assert "type=volume,source=llm_hf-cache,target=/root/.cache/huggingface" in mounts
    assert "type=tmpfs,target=/scratch,tmpfs-size=1073741824,tmpfs-mode=1777" in mounts
    # ค่าที่มี , ต้องห่อแบบ CSV — ไม่งั้น docker อ่าน "v2" เป็นคีย์ใหม่แล้ว start ไม่ขึ้น
    assert 'type=bind,"source=/srv/my models,v2",target=/models,bind-propagation=rslave' in mounts


def test_the_model_under_a_mount_is_found_on_the_host(tmp_path):
    """`lmds remove`/status อ่านที่เก็บ weight จาก bind mount — --mount ก็คือ bind mount เหมือนกัน"""
    models = tmp_path / "data" / "models"
    (models / "gemma-4-31b-it").mkdir(parents=True)
    adopted = inspected(container_payload(host={"Mounts": [
        {"Type": "bind", "Source": str(models), "Target": "/models", "ReadOnly": True}]}))
    weights = adopt_mod.weights_on_host(adopted)
    assert weights["path"] == str(models / "gemma-4-31b-it")
    assert weights["kind"] == "dir"


def test_a_pinned_gpu_stays_pinned(tmp_path):
    """เคสที่แพงที่สุด: เครื่อง 2 GPU รันโมเดลละใบ · `--gpus device=1` ถูกเขียนกลับเป็น `--gpus all`
    → restart แล้วโมเดลนี้ไปจอง VRAM บนการ์ดของอีกตัว ตัวที่อยู่ก่อน OOM"""
    _, argv = _run(tmp_path)
    assert values_of(argv, "--gpus") == ["device=1"]


def test_two_pinned_gpus_and_a_gpu_count(tmp_path):
    two = {**COMPOSE_STYLE["host"], "DeviceRequests": [
        {"Driver": "nvidia", "Count": 0, "DeviceIDs": ["0", "2"], "Capabilities": [["gpu"]], "Options": {}}]}
    _, argv = _run(tmp_path, host=two)
    # docker แยก --gpus ด้วย CSV: หลาย id ต้องอยู่ใน "…" ไม่งั้น "2" ถูกอ่านเป็นจำนวน GPU
    assert values_of(argv, "--gpus") == ['"device=0,2"']

    count = {**COMPOSE_STYLE["host"], "DeviceRequests": [
        {"Driver": "", "Count": 2, "DeviceIDs": None, "Capabilities": [["gpu"]], "Options": {}}]}
    _, argv = _run(tmp_path, host=count)
    assert values_of(argv, "--gpus") == ["2"]


def test_the_nvidia_runtime_with_a_chosen_device_is_not_widened_to_all_gpus(tmp_path):
    """ทางเก่า: `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=1` · `--gpus all` ทำให้ daemon เขียนทับ
    NVIDIA_VISIBLE_DEVICES เป็น all — ต้องคง runtime + env ไว้ ไม่ใช่แปลงเป็น --gpus all"""
    host = {**COMPOSE_STYLE["host"], "Runtime": "nvidia", "DeviceRequests": None}
    _, argv = _run(tmp_path, host=host)
    assert has_option(argv, "--runtime", "nvidia")
    assert has_option(argv, "--env", "NVIDIA_VISIBLE_DEVICES=1")
    assert "--gpus" not in argv


# ── ธงที่เหลือ ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("flag, value", [
    ("--ulimit", "memlock=-1:-1"),
    ("--ulimit", "stack=67108864:67108864"),
    ("--restart", "on-failure:3"),
    ("--cap-add", "SYS_NICE"),
    ("--add-host", "hub.internal:10.0.0.5"),
    ("--device", "/dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rwm"),
    ("--user", "1000:1000"),
    ("--workdir", "/workspace"),
    ("--memory", "107374182400"),
    ("--memory-swap", "214748364800"),
    ("--cpuset-cpus", "0-15"),
    ("--security-opt", "seccomp=unconfined"),
    ("--tmpfs", "/tmp:size=8g"),
    ("--group-add", "video"),
    ("--ipc", "host"),
    ("--label", "com.docker.compose.project=llm"),
    ("--label", "com.docker.compose.service=vllm"),
])
def test_every_option_the_customer_ran_with_reaches_docker_again(tmp_path, flag, value):
    _, argv = _run(tmp_path)
    assert has_option(argv, flag, value), f"{flag} {value} หายไปจาก docker run: {argv}"


def test_the_restart_policy_is_not_rewritten(tmp_path):
    """on-failure:3 ถูกบังคับเป็น unless-stopped = container ที่ล้มซ้ำถูกปลุกไม่มีวันจบ
    (เคส 2026-09-01: head ถูกปลุก 31 รอบโดยไม่มีใครรู้)"""
    _, argv = _run(tmp_path)
    assert values_of(argv, "--restart") == ["on-failure:3"]

    _, argv = _run(tmp_path, host={**COMPOSE_STYLE["host"], "RestartPolicy": {"Name": "always", "MaximumRetryCount": 0}})
    assert values_of(argv, "--restart") == ["always"]


@pytest.mark.parametrize("host, flag, value", [
    ({"Privileged": True}, "--privileged", None),
    ({"ReadonlyRootfs": True}, "--read-only", None),
    ({"Init": True}, "--init", None),
    ({"PidMode": "host"}, "--pid", "host"),
    ({"UTSMode": "host"}, "--uts", "host"),
    ({"CgroupnsMode": "host"}, "--cgroupns", "host"),
    ({"NanoCpus": 2500000000}, "--cpus", "2.5"),
    ({"CpuShares": 512}, "--cpu-shares", "512"),
    ({"MemoryReservation": 1073741824}, "--memory-reservation", "1073741824"),
    ({"PidsLimit": 4096}, "--pids-limit", "4096"),
    ({"OomScoreAdj": -500}, "--oom-score-adj", "-500"),
    ({"CapDrop": ["NET_RAW"]}, "--cap-drop", "NET_RAW"),
    ({"Dns": ["10.0.0.2"]}, "--dns", "10.0.0.2"),
    ({"Sysctls": {"net.core.somaxconn": "4096"}}, "--sysctl", "net.core.somaxconn=4096"),
    ({"ShmSize": 33554432}, "--shm-size", "33554432"),
    ({"VolumesFrom": ["weights-holder:ro"]}, "--volumes-from", "weights-holder:ro"),
    ({"LogConfig": {"Type": "json-file", "Config": {"max-size": "50m", "max-file": "3"}}}, "--log-opt", "max-size=50m"),
    ({"LogConfig": {"Type": "journald", "Config": {}}}, "--log-driver", "journald"),
    ({"Runtime": "nvidia"}, "--runtime", "nvidia"),
])
def test_less_common_options_are_carried_too(tmp_path, host, flag, value):
    adopted = inspected(container_payload(host=host))
    argv = started(adopt_mod.render_controller(adopted, "m"), tmp_path)
    assert has_option(argv, flag, value), f"{flag} {value}: {argv}"


@pytest.mark.parametrize("config, flag, value", [
    ({"Tty": True}, "--tty", None),
    ({"OpenStdin": True}, "--interactive", None),
    ({"Hostname": "llm-box"}, "--hostname", "llm-box"),
    ({"StopSignal": "SIGINT"}, "--stop-signal", "SIGINT"),
    ({"StopTimeout": 120}, "--stop-timeout", "120"),
])
def test_container_config_that_differs_from_the_image_is_carried(tmp_path, config, flag, value):
    adopted = inspected(container_payload(config=config))
    argv = started(adopt_mod.render_controller(adopted, "m"), tmp_path)
    assert has_option(argv, flag, value), f"{flag} {value}: {argv}"


# ── env ───────────────────────────────────────────────────────────────────────────────────
def test_env_the_customer_set_survives_whatever_it_is_called(tmp_path):
    """รายการ prefix (MODEL/PORT/VLLM_/HF_/…) เป็นการ *เดา* ว่าตัวไหนผู้ใช้ตั้งเอง · ของจริงคือ
    env ของ container ลบด้วย env ของ image — ถาม image ตรง ๆ แล้วไม่ต้องเดา"""
    _, argv = _run(tmp_path)
    env = values_of(argv, "--env")
    for wanted in ("NVIDIA_VISIBLE_DEVICES=1", "OMP_NUM_THREADS=8", "TRANSFORMERS_OFFLINE=1",
                   "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                   "HTTPS_PROXY=http://proxy.internal:3128", "VLLM_LOGGING_LEVEL=INFO"):
        assert wanted in env, f"{wanted} หายไป: {env}"
    assert "HF_TOKEN" in env and not any("hf_abc" in e for e in env), "ความลับยังต้องเหลือแค่ชื่อ"


def test_env_baked_into_the_image_is_still_left_to_the_image(tmp_path):
    """PATH/LD_LIBRARY_PATH/CUDA_VERSION ของ image ถูกตรึงลงสคริปต์ = วันที่เปลี่ยน IMAGE ได้ของเก่ามาทับ"""
    _, argv = _run(tmp_path)
    names = {e.split("=", 1)[0] for e in values_of(argv, "--env")}
    assert not names & {"PATH", "LD_LIBRARY_PATH", "CUDA_VERSION", "NVARCH", "DEBIAN_FRONTEND"}
    # ค่าเดียวกับที่ image ตั้ง = ไม่ใช่ของผู้ใช้ · ค่าที่ผู้ใช้ทับ (NVIDIA_VISIBLE_DEVICES=1) ต้องมา
    assert "NVIDIA_DRIVER_CAPABILITIES" not in names


def test_image_labels_and_workdir_are_not_repeated(tmp_path):
    adopted = inspected(container_payload())      # ไม่ได้ตั้งอะไรเกิน image เลย
    argv = started(adopt_mod.render_controller(adopted, "m"), tmp_path)
    assert "--label" not in argv and "--workdir" not in argv and "--user" not in argv


def test_when_the_image_cannot_be_asked_dropped_env_is_named_not_hidden(tmp_path):
    """image ถูกลบ/แท็กทับไปแล้ว = แยกไม่ออกว่า env ตัวไหนเป็นของผู้ใช้ · ถอยไปใช้รายการ prefix ได้
    แต่ต้อง **บอกชื่อ** ตัวที่ไม่ได้ใส่ — ชื่อเท่านั้น ไม่มีค่า (ค่าอาจเป็นความลับ)"""
    adopted = inspected(container_payload(env=["MY_CUSTOM_KNOB=abc-123-value", "OMP_NUM_THREADS=8"]),
                        with_image=False)
    script = adopt_mod.render_controller(adopted, "m")
    env_names = {e.split("=", 1)[0] for e in values_of(started(script, tmp_path), "--env")}
    said = " ".join(adopt_mod.not_reproduced(adopted))
    for name in ("MY_CUSTOM_KNOB", "PATH", "CUDA_VERSION"):
        assert name in env_names or name in said, f"{name} หายไปโดยไม่มีใครบอก"
    assert "abc-123-value" not in said


# ── พอร์ต ─────────────────────────────────────────────────────────────────────────────────
def test_every_host_binding_and_the_protocol_are_published_again(tmp_path):
    adopted = inspected(container_payload(host={"PortBindings": {
        "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}, {"HostIp": "10.0.0.5", "HostPort": "8000"}],
        "9000/udp": [{"HostIp": "", "HostPort": "9000"}],
        "9100/tcp": [{"HostIp": "::1", "HostPort": "9100"}],
    }}))
    published = values_of(started(adopt_mod.render_controller(adopted, "m"), tmp_path), "--publish")
    assert published == ["127.0.0.1:8000:8000", "10.0.0.5:8000:8000", "9000:9000/udp", "[::1]:9100:9100"]


# ── คำสั่งของ container (Path/Args) ───────────────────────────────────────────────────────
def test_an_image_without_entrypoint_keeps_its_executable(tmp_path):
    """`docker run img vllm serve org/m --port 8000` บน image ที่ไม่มี ENTRYPOINT:
    inspect ให้ Path="vllm" · Args=["serve", …] · Config.Entrypoint=null

    ของเดิมใช้แต่ Args และใส่ --entrypoint เฉพาะตอน Config.Entrypoint มีค่า → คำสั่งที่เขียนออกมาคือ
    `docker run … img serve org/m --port 8000` — ตัว `vllm` หายไป start ครั้งหน้าได้ "serve: not found"
    """
    image = "vllm/vllm-openai:v0.11.0"
    adopted = inspected(
        container_payload(path="vllm", args=["serve", "Qwen/Qwen3-8B", "--port", "8000"], entrypoint=None,
                          cmd=["vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000"]),
        image_payload({"Entrypoint": None, "Cmd": ["bash"]}))
    argv = started(adopt_mod.render_controller(adopted, "m"), tmp_path)
    resolved = values_of(argv, "--entrypoint") + command_after_image(argv, image)
    assert resolved == ["vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000"]
    assert adopted.model == "Qwen/Qwen3-8B" and adopted.engine == "vllm"


def test_an_image_with_entrypoint_runs_the_same_argv_as_before(tmp_path):
    image = "vllm/vllm-openai:v0.11.0"
    argv = started(adopt_mod.render_controller(inspected(container_payload()), "m"), tmp_path)
    resolved = values_of(argv, "--entrypoint") + command_after_image(argv, image)
    assert resolved == ["python3", "-m", "vllm.entrypoints.openai.api_server",
                        "--model", "/models/gemma-4-31b-it", "--port", "8000"]


def test_engine_is_read_from_the_executable_when_the_image_name_says_nothing(tmp_path):
    adopted = inspected(container_payload(image="registry.internal/llm/serving:2026-09", path="vllm",
                                          args=["serve", "Qwen/Qwen3-8B"], entrypoint=None,
                                          cmd=["vllm", "serve", "Qwen/Qwen3-8B"]))
    assert adopted.engine == "vllm"


# ── container ที่ไม่ได้ตั้งอะไรเป็นพิเศษ ─────────────────────────────────────────────────
def test_a_plain_container_gets_no_extra_flags(tmp_path):
    """`docker run -d --gpus all -p 8000:8000 <image> …` เปล่า ๆ — ค่า default ของ docker ทั้งชุด
    ต้องไม่กลายเป็นธงสักตัว · คำสั่งที่ได้ต้องเท่ากับที่ adopt เขียนมาตลอด"""
    image = "vllm/vllm-openai:v0.11.0"
    argv = started(adopt_mod.render_controller(inspected(container_payload()), "m"), tmp_path)
    assert argv == ["run", "-d", "--name", "vllm-gemma4", "--restart", "unless-stopped", "--no-healthcheck",
                    "--gpus", "all", "--network", "bridge", "--publish", "8000:8000",
                    "--entrypoint", "python3", image,
                    "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/gemma-4-31b-it",
                    "--port", "8000"]
