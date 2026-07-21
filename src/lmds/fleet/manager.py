"""Fleet manager — มองเห็น/สั่งงานทุกโมเดลที่ deploy ด้วย LMDS ในเครื่องเดียว

หลักการ: controller ทุกตัวเขียน `server.meta` ใต้ ~/.lmds/run/<slug>/ ตอน start
lmds อ่าน meta ทั้งหมด + เช็คสถานะจริง (pid/container/health) — ไม่ต้องมี daemon
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx


def run_root() -> Path:
    return Path(os.environ.get("LMDS_RUN_ROOT", Path.home() / ".lmds" / "run"))


@dataclass
class ServerInfo:
    slug: str
    model: str = ""
    model_id: str = ""
    engine: str = ""
    mode: str = ""  # native | docker
    port: int = 0
    container: str = ""
    pid_file: str = ""
    pid: int = 0  # ใช้กับ process ที่ตรวจเจอโดยไม่มีทะเบียน
    controller: str = ""
    started_at: str = ""
    run_dir: Path = field(default_factory=Path)
    running: bool = False
    healthy: bool = False
    registered: bool = True  # False = ตรวจเจอแต่ไม่มี server.meta (bundle รุ่นเก่า)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1" if self.port else ""

    @property
    def controller_exists(self) -> bool:
        return bool(self.controller) and Path(self.controller).is_file()


def _parse_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _pid_alive(pid_file: str) -> bool:
    try:
        pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _container_running(container: str) -> bool:
    if not container or shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _health_ok(port: int) -> bool:
    if not port:
        return False
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _pgrep_llama() -> list[tuple[int, str]]:
    """หา process llama-server ทั้งหมด — คืน [(pid, cmdline)]"""
    if shutil.which("pgrep") is None:
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-a", "-f", "llama-server"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    out: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and "llama-server" in parts[1]:
            out.append((int(parts[0]), parts[1]))
    return out


def _cmdline_value(cmdline: str, flag: str) -> str:
    tokens = cmdline.split()
    for i, token in enumerate(tokens[:-1]):
        if token == flag:
            return tokens[i + 1]
    return ""


def _orphan_native(known_pids: set[int]) -> list[ServerInfo]:
    """llama-server ที่รันอยู่แต่ไม่มีทะเบียน (เช่น start จาก bundle รุ่นเก่า)"""
    orphans: list[ServerInfo] = []
    for pid, cmdline in _pgrep_llama():
        if pid in known_pids:
            continue
        alias = _cmdline_value(cmdline, "--alias")
        model_path = _cmdline_value(cmdline, "-m")
        port = _cmdline_value(cmdline, "--port")
        slug = alias or (Path(model_path).stem if model_path else f"pid-{pid}")
        orphans.append(ServerInfo(
            slug=slug, model=alias, engine="llamacpp", mode="native",
            port=int(port) if port.isdigit() else 0,
            pid=pid, running=True, registered=False,
        ))
    return orphans


def _orphan_docker(known_containers: set[str]) -> list[ServerInfo]:
    """container ชื่อ lmds-* ที่รันอยู่แต่ไม่มีทะเบียน"""
    if shutil.which("docker") is None:
        return []
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", "name=lmds-", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    orphans: list[ServerInfo] = []
    for name in proc.stdout.split():
        if name in known_containers:
            continue
        orphans.append(ServerInfo(
            slug=name.removeprefix("lmds-"), engine="?", mode="docker",
            container=name, running=True, registered=False,
        ))
    return orphans


def discover() -> list[ServerInfo]:
    """สแกนทุก server: ทั้งที่ลงทะเบียน (server.meta) และที่รันอยู่โดยไม่มีทะเบียน (bundle รุ่นเก่า)"""
    servers: list[ServerInfo] = []
    root = run_root()
    if not root.is_dir():
        root_metas: list[Path] = []
    else:
        root_metas = sorted(root.glob("*/server.meta"))
    for meta_path in root_metas:
        try:
            meta = _parse_meta(meta_path)
        except OSError:
            continue
        info = ServerInfo(
            slug=meta.get("slug", meta_path.parent.name),
            model=meta.get("model", ""),
            model_id=meta.get("model_id", ""),
            engine=meta.get("engine", ""),
            mode=meta.get("mode", ""),
            port=int(meta.get("port") or 0),
            container=meta.get("container", ""),
            pid_file=meta.get("pid_file", ""),
            controller=meta.get("controller", ""),
            started_at=meta.get("started_at", ""),
            run_dir=meta_path.parent,
        )
        if info.mode == "native":
            info.running = bool(info.pid_file) and _pid_alive(info.pid_file)
        else:
            info.running = _container_running(info.container)
        info.healthy = info.running and _health_ok(info.port)
        servers.append(info)

    # เก็บตกโมเดลที่รันอยู่โดยไม่มีทะเบียน (start จาก bundle รุ่นเก่า/เครื่องอื่น)
    known_pids: set[int] = set()
    for server in servers:
        if server.pid_file:
            try:
                known_pids.add(int(Path(server.pid_file).read_text(encoding="utf-8").strip()))
            except (OSError, ValueError):
                pass
    known_containers = {s.container for s in servers if s.container}
    for orphan in [*_orphan_native(known_pids), *_orphan_docker(known_containers)]:
        orphan.healthy = orphan.running and _health_ok(orphan.port)
        servers.append(orphan)
    return servers


def find(slug: str) -> ServerInfo | None:
    return next((s for s in discover() if s.slug == slug), None)


class FleetError(Exception):
    pass


def _run_controller(info: ServerInfo, command: str, extra: list[str] | None = None) -> int:
    if not info.controller_exists:
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} ({info.controller or 'ไม่ระบุ'}) — "
            "bundle อาจถูกย้าย/ลบ (ใช้ lmds stop จะ fallback หยุดตรง ๆ ให้)"
        )
    proc = subprocess.run([info.controller, command, *(extra or [])])
    return proc.returncode


def stop_server(info: ServerInfo) -> str:
    """หยุดผ่าน controller; ถ้า controller หาย/ไม่ลงทะเบียน ใช้ fallback (kill pid / docker rm)"""
    if info.controller_exists:
        _run_controller(info, "stop")
        return "controller"
    if info.mode == "native":
        if info.pid:
            os.kill(info.pid, 15)
        elif info.pid_file and _pid_alive(info.pid_file):
            pid = int(Path(info.pid_file).read_text(encoding="utf-8").strip())
            os.kill(pid, 15)
            Path(info.pid_file).unlink(missing_ok=True)
        return "kill"
    subprocess.run(["docker", "rm", "-f", info.container], capture_output=True)
    return "docker-rm"


def start_server(info: ServerInfo) -> int:
    return _run_controller(info, "start")


def logs_server(info: ServerInfo, lines: int = 200) -> int:
    return _run_controller(info, "logs", [str(lines)])
