"""ของปลอมที่เทส adopt ใช้ร่วมกัน — `docker inspect` ที่ตอบ payload จริง และ `docker` ที่จด argv

เทส adopt ชุดเดิมแต่ละไฟล์ปั้น payload กับ fake docker ของตัวเอง ซึ่งพอสำหรับข้อเดียว · ชุด
audit 2026-10-06 ต้องถามคำถามเดียวกันซ้ำหลายสิบครั้ง ("สคริปต์ที่สร้างออกมา พอสั่ง start แล้ว
docker ได้รับ argv อะไร") จึงรวมไว้ที่เดียว

หลักที่ยึด (CONTRIBUTING: "เทสที่โกหกไม่ได้"): **ไม่ grep สตริงในสคริปต์** — รันสคริปต์จริงใต้ bash
แล้วอ่าน argv ที่ docker ปลอมได้รับ · ธงที่วางผิดบรรทัด/quote ผิด จะไม่ไปถึง docker และเทสจะเห็น
"""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

adopt_mod = importlib.import_module("lmds.fleet.adopt")   # lmds.fleet ส่งออกฟังก์ชันชื่อ adopt ทับชื่อโมดูล

# HostConfig ของ `docker run -d --gpus all -p 8000:8000 <image>` เปล่า ๆ (docker 27) — ค่า default ครบทุกคีย์
# ใช้เป็นฐาน: เทสแต่ละข้อแก้เฉพาะคีย์ที่ตัวเองสนใจ ที่เหลือต้อง "ไม่โผล่" ในคำสั่งที่สร้าง
DEFAULT_HOST_CONFIG = {
    "Binds": None, "ContainerIDFile": "", "LogConfig": {"Type": "json-file", "Config": {}},
    "NetworkMode": "bridge", "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}, "AutoRemove": False,
    "VolumeDriver": "", "VolumesFrom": None, "ConsoleSize": [24, 80],
    "CapAdd": None, "CapDrop": None, "CgroupnsMode": "private", "Dns": [], "DnsOptions": [],
    "DnsSearch": [], "ExtraHosts": None, "GroupAdd": None, "IpcMode": "private", "Cgroup": "",
    "Links": None, "OomScoreAdj": 0, "PidMode": "", "Privileged": False, "PublishAllPorts": False,
    "ReadonlyRootfs": False, "SecurityOpt": None, "UTSMode": "", "UsernsMode": "",
    "ShmSize": 67108864, "Runtime": "runc", "Isolation": "", "CpuShares": 0, "Memory": 0,
    "NanoCpus": 0, "CgroupParent": "", "BlkioWeight": 0, "BlkioWeightDevice": [],
    "BlkioDeviceReadBps": [], "BlkioDeviceWriteBps": [], "BlkioDeviceReadIOps": [],
    "BlkioDeviceWriteIOps": [], "CpuPeriod": 0, "CpuQuota": 0, "CpuRealtimePeriod": 0,
    "CpuRealtimeRuntime": 0, "CpusetCpus": "", "CpusetMems": "", "Devices": [],
    "DeviceCgroupRules": None,
    "DeviceRequests": [{"Driver": "", "Count": -1, "DeviceIDs": None,
                        "Capabilities": [["gpu"]], "Options": {}}],
    "MemoryReservation": 0, "MemorySwap": 0, "MemorySwappiness": None, "OomKillDisable": False,
    "PidsLimit": None, "Ulimits": [], "CpuCount": 0, "CpuPercent": 0, "IOMaximumIOps": 0,
    "IOMaximumBandwidth": 0,
    "MaskedPaths": ["/proc/asound", "/proc/acpi", "/proc/kcore", "/proc/keys", "/proc/latency_stats",
                    "/proc/timer_list", "/proc/timer_stats", "/proc/sched_debug", "/proc/scsi",
                    "/sys/firmware", "/sys/devices/virtual/powercap"],
    "ReadonlyPaths": ["/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger"],
}

IMAGE_ID = "sha256:5f2d1c0e9a7b4c3d8e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"
CONTAINER_ID = "9c1e7a5b3d2f4a6c8e0b1d3f5a7c9e2b4d6f8a0c1e3b5d7f9a2c4e6b8d0f1a3c"

# env/label/workdir ที่ image `vllm/vllm-openai` อบมาเอง — ของพวกนี้ **ไม่ใช่** สิ่งที่ผู้ใช้สั่งตอน docker run
VLLM_IMAGE_CONFIG = {
    "Env": ["PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "NVARCH=x86_64", "NVIDIA_REQUIRE_CUDA=cuda>=12.4", "CUDA_VERSION=12.4.1",
            "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "DEBIAN_FRONTEND=noninteractive", "VLLM_USAGE_SOURCE=production-docker-image"],
    "Entrypoint": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
    "Cmd": None, "User": "", "WorkingDir": "/vllm-workspace",
    "Labels": {"maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
               "org.opencontainers.image.ref.name": "ubuntu"},
    "ExposedPorts": None, "Volumes": None, "StopSignal": "",
}


def container_payload(*, name="vllm-gemma4", image="vllm/vllm-openai:v0.11.0", path="python3",
                      args=None, entrypoint=("python3", "-m", "vllm.entrypoints.openai.api_server"),
                      cmd=None, env=None, host=None, config=None, **top) -> dict:
    """payload ของ `docker inspect <container>` รูปเดียวกับของจริง — ครบทั้ง Path/Args และ Config.Entrypoint/Cmd"""
    host_config = copy.deepcopy(DEFAULT_HOST_CONFIG)
    host_config.update(host or {})
    # docker รวม env แบบ "ชื่อเดิมถูกแทนที่ตรงตำแหน่งเดิม · ชื่อใหม่ต่อท้าย" — Config.Env ไม่มีชื่อซ้ำ
    merged = list(VLLM_IMAGE_CONFIG["Env"])
    for item in env or []:
        name = item.split("=", 1)[0]
        at = next((i for i, have in enumerate(merged) if have.split("=", 1)[0] == name), None)
        if at is None:
            merged.append(item)
        else:
            merged[at] = item
    cfg = {
        "Hostname": CONTAINER_ID[:12], "Domainname": "", "User": "", "Tty": False, "OpenStdin": False,
        "Env": merged, "Cmd": list(cmd) if cmd is not None else None,
        "Image": image, "Volumes": None, "WorkingDir": "/vllm-workspace",
        "Entrypoint": list(entrypoint) if entrypoint is not None else None,
        "Labels": dict(VLLM_IMAGE_CONFIG["Labels"]), "ExposedPorts": None,
    }
    cfg.update(config or {})
    data = {
        "Id": CONTAINER_ID, "Name": f"/{name}", "Path": path,
        "Args": list(args) if args is not None else
        ["-m", "vllm.entrypoints.openai.api_server", "--model", "/models/gemma-4-31b-it", "--port", "8000"],
        "Image": IMAGE_ID, "State": {"Status": "running", "Running": True},
        "Config": cfg, "HostConfig": host_config, "Mounts": [],
        "NetworkSettings": {"Ports": {}, "Networks": {}},
    }
    data.update(top)
    return data


def image_payload(config: dict | None = None) -> dict:
    """payload ของ `docker image inspect` — มี RootFS/RepoTags ไม่มี State/HostConfig"""
    cfg = copy.deepcopy(VLLM_IMAGE_CONFIG)
    cfg.update(config or {})
    return {"Id": IMAGE_ID, "RepoTags": ["vllm/vllm-openai:v0.11.0"], "Architecture": "arm64",
            "Os": "linux", "Config": cfg, "RootFS": {"Type": "layers", "Layers": ["sha256:aa"]}}


class _Done:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


def fake_docker_inspect(container: dict, image: dict | None):
    """ตัวแทน subprocess.run ของ adopt — แยกคำตอบตามคำสั่งเหมือน docker จริง

    image=None = `docker image inspect` ล้ม (image ถูกลบ/แท็กใหม่ไปแล้ว) ซึ่งเป็นทางที่ adopt
    ต้องยังทำงานได้และ **บอก** ว่าแยก env ของ image ออกจากของผู้ใช้ไม่ได้
    """
    def run(argv, *a, **k):
        if list(argv[:3]) == ["docker", "image", "inspect"]:
            return _Done(json.dumps([image])) if image is not None else _Done("[]", 1)
        if list(argv[:2]) == ["docker", "inspect"]:
            return _Done(json.dumps([container]))
        raise AssertionError(f"adopt เรียกคำสั่งที่เทสไม่ได้เตรียมไว้: {argv}")
    return run


def inspected(container: dict, image: dict | None = None, *, with_image: bool = True):
    """Adopted จาก payload — ผ่าน inspect_container ตัวจริง ไม่ได้ปั้น dataclass เอง"""
    img = (image or image_payload()) if with_image else None
    with patch.object(adopt_mod.subprocess, "run", side_effect=fake_docker_inspect(container, img)):
        return adopt_mod.inspect_container(container["Name"].lstrip("/"))


def adopt_from(container: dict, output: Path, *, slug: str = "", image: dict | None = None,
               with_image: bool = True) -> Path:
    """`adopt()` ตัวจริงกับ docker ปลอม — คืน path ของ controller"""
    img = (image or image_payload()) if with_image else None
    with patch.object(adopt_mod.subprocess, "run", side_effect=fake_docker_inspect(container, img)):
        return adopt_mod.adopt(container["Name"].lstrip("/"), slug=slug, output=output)


FAKE_DOCKER = r'''#!/bin/bash
# docker ปลอม: จด argv ของ `docker run` ทีละตัวคั่นด้วย NUL — ช่องว่าง/quote/ขึ้นบรรทัดใหม่ในค่าไม่ทำให้อ่านเพี้ยน
if [ "$1" = run ]; then
  for a in "$@"; do printf '%s\0' "$a"; done > "$FAKE_ARGV"
  # env ที่ docker run "หยิบจากเชลล์" (--env NAME เฉย ๆ) — จดค่าที่เชลล์มีตอนนั้นไว้ให้เทสดู
  env > "$FAKE_ARGV.env"
fi
exit 0
'''


def run_controller(script: str, tmp_path: Path, command: str = "start", env: dict | None = None,
                   extra_bins: dict | None = None):
    """รันสคริปต์ที่สร้างออกมาใต้ bash จริง กับ docker ปลอม — คืน (CompletedProcess, argv ของ docker run หรือ None)"""
    home = tmp_path / "home"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in {"docker": FAKE_DOCKER, **(extra_bins or {})}.items():
        tool = bin_dir / name
        tool.write_text(body, encoding="utf-8")
        tool.chmod(0o755)
    controller = home / "c-adopted.sh"
    controller.write_text(script, encoding="utf-8")
    controller.chmod(0o755)
    argv_log = tmp_path / "docker-run.argv"
    if argv_log.exists():
        argv_log.unlink()
    full_env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_ARGV": str(argv_log)}
    full_env.update(env or {})
    done = subprocess.run(["bash", str(controller), command], capture_output=True, text=True, env=full_env)
    argv = None
    if argv_log.exists():
        argv = [a for a in argv_log.read_bytes().decode("utf-8").split("\0")][:-1]
    return done, argv


def started(script: str, tmp_path: Path, env: dict | None = None) -> list[str]:
    """argv ที่ docker ได้รับเมื่อสั่ง start — ล้มทันทีถ้า start ไม่ผ่านหรือไม่ได้เรียก docker run"""
    done, argv = run_controller(script, tmp_path, "start", env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "command not found" not in done.stderr, done.stderr
    assert argv, "start จบโดยไม่ได้เรียก docker run"
    return argv


def has_option(argv: list[str], flag: str, value: str | None = None) -> bool:
    """`flag value` อยู่ติดกันใน argv ไหม (value=None = ธงเดี่ยว)"""
    if value is None:
        return flag in argv
    return any(a == flag and b == value for a, b in zip(argv, argv[1:], strict=False))


def values_of(argv: list[str], flag: str) -> list[str]:
    return [b for a, b in zip(argv, argv[1:], strict=False) if a == flag]


def command_after_image(argv: list[str], image: str) -> list[str]:
    """ส่วนที่ตามหลังชื่อ image = คำสั่งที่ container จะรัน"""
    return argv[argv.index(image) + 1:]
