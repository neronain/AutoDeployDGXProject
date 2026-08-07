"""รับ container ที่รันอยู่ก่อน LMDS เข้ามาอยู่ในระบบ

ลูกค้าจำนวนมากมี vLLM/llama.cpp รันอยู่ก่อนแล้วเพิ่งมาติดตั้ง LMDS ทีหลัง · `lmds ps`
มองเห็น container พวกนั้นและ stop/restart/logs ได้ แต่ทำอย่างอื่นไม่ได้เลย เพราะไม่มี
controller — กด repair ก็ได้แต่คำว่า "ไม่พบ controller"

ตัวนี้อ่านสิ่งที่ container กำลังใช้อยู่จริง (image, env, mount, port, args) แล้วเขียนเป็น
controller ที่ **รันคำสั่งเดิมซ้ำได้เป๊ะ** — ของที่รันอยู่ไม่ถูกแตะต้อง

หลักที่ยึด:
  - **สร้างจากสิ่งที่รันอยู่จริง ไม่ใช่เดา** — อ่านจาก `docker inspect` ตรง ๆ
  - **ไม่แกล้งทำเป็นมี `download`/`verify-files`** — weight ของ container พวกนี้เป็น path
    ที่ผู้ใช้จัดการเอง ไม่ได้มาจาก Hugging Face ที่เรารู้จัก · คำสั่งที่ทำอะไรไม่ได้จริง
    แต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มีคำสั่งนั้น
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .manager import FleetError, run_root


@dataclass
class Adopted:
    """สิ่งที่อ่านได้จาก container ที่รันอยู่"""

    container: str
    image: str
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)
    ports: dict = field(default_factory=dict)
    network: str = ""
    runtime: str = ""
    entrypoint: list[str] = field(default_factory=list)

    @property
    def port(self) -> int:
        for item in self.env:
            if item.startswith("PORT="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    break
        for spec in self.ports or {}:
            try:
                return int(spec.split("/")[0])
            except ValueError:
                continue
        return 0

    @property
    def model(self) -> str:
        for key in ("MODEL=", "MODEL_ID=", "MODEL_PATH="):
            for item in self.env:
                if item.startswith(key):
                    return item.split("=", 1)[1]
        return ""

    @property
    def context(self) -> int:
        for key in ("MAX_MODEL_LEN=", "CTX_SIZE="):
            for item in self.env:
                if item.startswith(key):
                    try:
                        return int(item.split("=", 1)[1])
                    except ValueError:
                        break
        return 0


def inspect_container(container: str) -> Adopted:
    """อ่านทุกอย่างที่ต้องใช้เพื่อรันซ้ำ — ล้มเหลวชัด ๆ ถ้าไม่มี container นั้น"""
    try:
        proc = subprocess.run(["docker", "inspect", container],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FleetError(f"เรียก docker inspect ไม่ได้: {exc}") from exc
    if proc.returncode != 0:
        raise FleetError(f"ไม่พบ container '{container}' — ดูรายชื่อ: docker ps")
    data = json.loads(proc.stdout)[0]
    config, host = data.get("Config") or {}, data.get("HostConfig") or {}
    return Adopted(
        container=data["Name"].lstrip("/"),
        image=config.get("Image") or "",
        args=list(data.get("Args") or []),
        env=list(config.get("Env") or []),
        binds=list(host.get("Binds") or []),
        ports=dict(host.get("PortBindings") or {}),
        network=host.get("NetworkMode") or "",
        runtime=host.get("Runtime") or "",
        entrypoint=list(config.get("Entrypoint") or []),
    )


# env ของ image เองมีเป็นร้อยตัว (PATH, CUDA_*, LD_*) — เอาไปใส่ใน docker run ซ้ำ
# ไม่ได้ช่วยอะไรและทำให้สคริปต์อ่านไม่รู้เรื่อง · เก็บเฉพาะที่ผู้ใช้ตั้งเองจริง ๆ
_KEEP_ENV_PREFIXES = ("MODEL", "PORT", "MAX_", "VLLM_", "HF_", "CTX_", "API_", "SERVED_",
                      "NCCL_", "CUDA_VISIBLE_DEVICES", "TOKENIZERS_")


def meaningful_env(adopted: Adopted) -> list[str]:
    return [e for e in adopted.env if e.split("=", 1)[0].startswith(_KEEP_ENV_PREFIXES)]


def render_controller(adopted: Adopted, slug: str) -> str:
    """สคริปต์ที่รัน container เดิมซ้ำได้ — คำสั่งเดียวกับที่มันรันอยู่ตอนนี้"""
    env_lines = "".join(f'  --env {shlex.quote(e)} \\\n' for e in meaningful_env(adopted))
    bind_lines = "".join(f'  --volume {shlex.quote(b)} \\\n' for b in adopted.binds)
    port_lines = "".join(
        f'  --publish {shlex.quote(binding[0]["HostPort"])}:{shlex.quote(spec.split("/")[0])} \\\n'
        for spec, binding in (adopted.ports or {}).items() if binding
    )
    entry = f'  --entrypoint {shlex.quote(adopted.entrypoint[0])} \\\n' if adopted.entrypoint else ""
    network = f'  --network {shlex.quote(adopted.network)} \\\n' if adopted.network not in ("", "default") else ""
    runtime = '  --gpus all \\\n' if adopted.runtime == "nvidia" else ""
    args = " ".join(shlex.quote(a) for a in adopted.args)

    return f'''#!/usr/bin/env bash
# LMDS adopted controller — สร้างจาก container ที่รันอยู่ก่อนหน้า ไม่ได้ deploy ผ่าน LMDS
#
# สคริปต์นี้ทำได้แค่ "รันคำสั่งเดิมซ้ำ" — weight เป็น path ที่ผู้ใช้จัดการเอง จึงไม่มี
# download/verify-files ให้ · คำสั่งที่ทำอะไรไม่ได้จริงแต่คืน 0 คือคำโกหกที่แพงกว่าการไม่มี
set -Eeuo pipefail

SCRIPT_VERSION="${{SCRIPT_VERSION:-1.0.0}}"
ADOPTED=1
CONTAINER_NAME="{adopted.container}"
IMAGE="${{IMAGE:-{adopted.image}}}"
API_PORT="${{API_PORT:-{adopted.port or 8000}}}"
SLUG="{slug}"

die() {{ echo "ERROR: $*" >&2; exit 1; }}

banner() {{
  echo "LMDS adopted · {slug} · v${{SCRIPT_VERSION}}"
  echo "container: ${{CONTAINER_NAME}} · image: ${{IMAGE}}"
}}

info() {{
  banner
  echo "model:     {adopted.model or '(ไม่ระบุใน env)'}"
  echo "context:   {adopted.context or 0}"
  echo "port:      ${{API_PORT}}"
  echo "adopted:   ใช่ — สร้างจาก container ที่รันอยู่ก่อน LMDS"
}}

start() {{
  local running
  running="$(docker ps --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  [[ -z "$running" ]] || die "container ${{CONTAINER_NAME}} กำลังรันอยู่ — รัน: $0 stop ก่อน"
  local leftover
  leftover="$(docker ps -a --filter "name=^${{CONTAINER_NAME}}$" --format '{{{{.Names}}}}' 2>/dev/null || true)"
  if [[ -n "$leftover" ]]; then
    echo "เก็บซาก container จากรอบก่อน (${{CONTAINER_NAME}}) แล้วเริ่มใหม่"
    docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  fi
  docker run -d --name "${{CONTAINER_NAME}}" --restart unless-stopped \\
{runtime}{network}{port_lines}{bind_lines}{env_lines}{entry}  "${{IMAGE}}" {args}
  echo "started: ${{CONTAINER_NAME}} (port ${{API_PORT}})"
}}

stop() {{
  docker stop "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
  echo "stopped: ${{CONTAINER_NAME}}"
}}

restart() {{ stop; start; }}

status() {{
  docker ps -a --filter "name=^${{CONTAINER_NAME}}$" --format 'container: {{{{.Names}}}} · {{{{.Status}}}}'
  curl -fsS -m 5 "http://127.0.0.1:${{API_PORT}}/v1/models" >/dev/null 2>&1 \\
    && echo "api: ตอบปกติ" || echo "api: ยังไม่ตอบ"
}}

logs() {{ docker logs --tail "${{1:-300}}" "${{CONTAINER_NAME}}"; }}

test_text() {{
  local served
  served="$(curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models" | sed -E 's/.*"id":"([^"]+)".*/\\1/')" \\
    || die "เรียก /v1/models ไม่ได้ — server ขึ้นหรือยัง? ดู: $0 logs"
  curl -fsS "http://127.0.0.1:${{API_PORT}}/v1/chat/completions" \\
    -H "Content-Type: application/json" \\
    -d "{{\\"model\\": \\"$served\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": \\"ตอบสั้น ๆ: 2+2 เท่ากับเท่าไร\\"}}], \\"max_tokens\\": 256}}" \\
    || die "เรียก /v1/chat/completions ไม่สำเร็จ — ดู: $0 logs"
  echo ""
}}

client_config() {{
  local served
  served="$(curl -fsS -m 10 "http://127.0.0.1:${{API_PORT}}/v1/models" | sed -E 's/.*"id":"([^"]+)".*/\\1/')" || served="{slug}"
  echo "{{"
  echo "  \\"base_url\\": \\"http://$(hostname -I | awk '{{print $1}}'):${{API_PORT}}/v1\\","
  echo "  \\"model\\": \\"$served\\","
  echo "  \\"server_context\\": {adopted.context or 0}"
  echo "}}"
}}

usage() {{
  banner
  cat <<'USAGE'

คำสั่ง:
  start | stop | restart      รันคำสั่งเดิมของ container ซ้ำ
  status                      สถานะ container + API
  logs [N]                    log ล่าสุด N บรรทัด
  test-text                   ถามจริงแล้วดูว่าตอบไหม
  client-config               ค่าที่ client ต้องใช้
  info | banner               ข้อมูลของ bundle นี้

ไม่มี download / verify-files: weight ของ container นี้เป็น path ที่คุณจัดการเอง
LMDS จึงไม่มีอะไรให้โหลดหรือตรวจ — ดูแลไฟล์เองเหมือนเดิม
USAGE
}}

case "${{1:-}}" in
  start)          start ;;
  stop)           stop ;;
  restart)        restart ;;
  status)         status ;;
  logs)           shift; logs "${{1:-300}}" ;;
  test-text)      test_text ;;
  client-config)  client_config ;;
  info|banner)    info ;;
  *)              usage ;;
esac
'''


def adopt(container: str, slug: str = "", output: Path | None = None) -> Path:
    """สร้าง bundle จาก container ที่รันอยู่ แล้วลงทะเบียนกับ fleet — คืน path ของ controller"""
    adopted = inspect_container(container)
    slug = slug or adopted.container.replace("_", "-").lower()
    directory = (output or Path("./bundles")) / slug
    directory.mkdir(parents=True, exist_ok=True)

    controller = directory / f"{slug}-adopted.sh"
    controller.write_text(render_controller(adopted, slug), encoding="utf-8")
    controller.chmod(0o755)

    profile = {
        "profile_version": 1,
        "generated_by": "lmds adopt",
        "adopted": True,
        "model": {"id": adopted.model or adopted.container, "artifact_type": "unknown"},
        "runtime": {"engine": "vllm" if "vllm" in adopted.image.lower() else "unknown",
                    "image": adopted.image},
        "serving": {"context": adopted.context, "port": adopted.port},
        "source_container": adopted.container,
    }
    import yaml

    (directory / "MODEL_PROFILE.yaml").write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_dir = run_root() / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={adopted.model or adopted.container}\n"
        f"model_id={adopted.model or adopted.container}\n"
        f"engine={profile['runtime']['engine']}\n"
        f"mode=docker\n"
        f"port={adopted.port}\n"
        f"container={adopted.container}\n"
        f"pid_file=\n"
        f"controller={controller}\n"
        f"started_at=\n",
        encoding="utf-8")
    return controller
