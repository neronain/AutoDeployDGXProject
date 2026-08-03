"""Fleet manager — มองเห็น/สั่งงานทุกโมเดลที่ deploy ด้วย LMDS ในเครื่องเดียว

หลักการ: controller ทุกตัวเขียน `server.meta` ใต้ ~/.lmds/run/<slug>/ ตอน start
lmds อ่าน meta ทั้งหมด + เช็คสถานะจริง (pid/container/health) — ไม่ต้องมี daemon
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx


def run_root() -> Path:
    return Path(os.environ.get("LMDS_RUN_ROOT", Path.home() / ".lmds" / "run"))


# ── Autostart (systemd) ────────────────────────────────────────────────────────
# ให้โมเดลกลับมาทำงานเองหลัง reboot — สร้าง system service ที่เรียก controller start
def systemd_dir() -> Path:
    return Path(os.environ.get("LMDS_SYSTEMD_DIR", "/etc/systemd/system"))


def unit_name(slug: str) -> str:
    return f"lmds-{slug}.service"


def have_systemctl() -> bool:
    return shutil.which("systemctl") is not None


def _controller_owner(controller: str) -> str:
    """คืนชื่อ user เจ้าของไฟล์ controller (เจ้าของ bundle) — fallback เป็น user ปัจจุบัน"""
    try:
        import pwd

        return pwd.getpwuid(os.stat(controller).st_uid).pw_name
    except (KeyError, OSError, ImportError):
        return getpass.getuser()


def render_unit(info: "ServerInfo", timeout: int = 1800) -> str:
    """สร้างเนื้อ systemd unit (system service) สำหรับ autostart ของ bundle นี้

    - Type=oneshot + RemainAfterExit: controller start เปิด container/process แบบ detach
      แล้ว return เมื่อ health ผ่าน (unit คง active หลัง exec จบ)
    - ExecStartPre=stop: เคลียร์ container/process ค้างจากก่อน reboot ก่อน start ใหม่
    - User=<เจ้าของ bundle>: รันเป็น user ปกติ (docker/HF cache/สิทธิ์ตรงกับตอน deploy)
    """
    if not info.controller_exists and info.mode == "docker" and info.container:
        return _render_docker_unit(info)

    controller = info.controller
    workdir = str(Path(controller).parent)
    user = _controller_owner(controller)
    home = str(Path.home())
    model = info.model or info.model_id or info.slug
    return "\n".join([
        "[Unit]",
        f"Description=LMDS model: {info.slug} ({model})",
        "After=network-online.target docker.service",
        "Wants=network-online.target docker.service",
        "",
        "[Service]",
        "Type=oneshot",
        "RemainAfterExit=yes",
        f"User={user}",
        f"Environment=HOME={home}",
        f"WorkingDirectory={workdir}",
        f"ExecStartPre=-{controller} stop",
        f"ExecStart={controller} start",
        f"ExecStop={controller} stop",
        f"TimeoutStartSec={timeout}",
        "Restart=no",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ])


def autostart_status(slug: str) -> str:
    """คืน 'enabled' | 'disabled' | 'absent' (ไม่มี unit) | 'n/a' (ไม่มี systemd)"""
    if not have_systemctl():
        return "n/a"
    if not (systemd_dir() / unit_name(slug)).exists():
        return "absent"
    try:
        proc = subprocess.run(
            ["systemctl", "is-enabled", unit_name(slug)],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "absent"
    out = proc.stdout.strip()
    return out if out in {"enabled", "disabled"} else ("enabled" if proc.returncode == 0 else "disabled")


def _render_docker_unit(info: "ServerInfo") -> str:
    """unit สำหรับ container ที่ไม่ได้มาจาก lmds — แค่ start container เดิมกลับมาหลัง reboot

    ไม่มี controller ให้เรียก จึงทำได้แค่ `docker start` (ไม่ได้สร้าง container ใหม่)
    ถ้า container ถูกลบไป unit นี้จะล้ม — ต้อง enable ใหม่หลังสร้าง container ใหม่
    """
    import getpass

    return "\n".join([
        "[Unit]",
        f"Description=LMDS (adopted container): {info.container}",
        "After=network-online.target docker.service",
        "Wants=network-online.target docker.service",
        "Requires=docker.service",
        "",
        "[Service]",
        "Type=oneshot",
        "RemainAfterExit=yes",
        f"User={getpass.getuser()}",
        f"ExecStart=/usr/bin/docker start {info.container}",
        f"ExecStop=/usr/bin/docker stop {info.container}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ])


def enable_autostart(info: "ServerInfo", timeout: int = 1800, start_now: bool = False) -> str:
    """ติดตั้ง + enable systemd unit (ต้องใช้ sudo) — คืนชื่อ unit ที่ติดตั้ง

    เขียนไฟล์ unit ลง bundle dir ก่อน (ไม่ใช้ sudo) แล้ว sudo ติดตั้งเข้า /etc/systemd/system
    ถ้า sudo/systemctl ล้มเหลว → FleetError พร้อมคำสั่งให้รันมือ
    """
    if not have_systemctl():
        raise FleetError("เครื่องนี้ไม่มี systemd (systemctl) — autostart รองรับเฉพาะระบบ systemd")
    adopted = not info.controller_exists and info.mode == "docker" and bool(info.container)
    if not info.controller_exists and not adopted:
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} — ต้องมี bundle หรือเป็น container ที่รันอยู่ก่อนตั้ง autostart"
        )

    name = unit_name(info.slug)
    if adopted:
        # container ที่ไม่ได้มาจาก lmds ไม่มี bundle dir ให้ stage — ใช้ run dir ของ fleet แทน
        stage_dir = run_root() / info.slug
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_dir / name
    else:
        staged = Path(info.controller).parent / name
    staged.write_text(render_unit(info, timeout), encoding="utf-8")

    steps = [
        ["sudo", "install", "-m", "644", str(staged), str(systemd_dir() / name)],
        ["sudo", "systemctl", "daemon-reload"],
        ["sudo", "systemctl", "enable", name],
    ]
    if start_now:
        steps.append(["sudo", "systemctl", "start", name])
    for cmd in steps:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            manual = "\n  ".join(" ".join(c) for c in steps)
            raise FleetError(
                f"ติดตั้ง autostart ไม่สำเร็จ (คำสั่ง `{' '.join(cmd)}` ล้มเหลว)\n"
                f"ลองรันมือ:\n  {manual}"
            )
    return name


def disable_autostart(info_or_slug) -> str:
    """disable + ลบ systemd unit (ต้องใช้ sudo) — รับ ServerInfo หรือ slug"""
    slug = info_or_slug.slug if isinstance(info_or_slug, ServerInfo) else info_or_slug
    if not have_systemctl():
        raise FleetError("เครื่องนี้ไม่มี systemd (systemctl)")
    name = unit_name(slug)
    steps = [
        ["sudo", "systemctl", "disable", "--now", name],
        ["sudo", "rm", "-f", str(systemd_dir() / name)],
        ["sudo", "systemctl", "daemon-reload"],
    ]
    for cmd in steps:
        # disable อาจ error ถ้า unit ไม่ได้ enable — ไม่ถือว่า fatal
        subprocess.run(cmd)
    return name


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
    external: bool = False  # container ที่ไม่ได้มาจาก lmds — จัดการได้แต่ต้องระวังกว่า

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


def _in_container(pid: int) -> bool:
    """process ใน container มองเห็นได้จาก process table ของ host ด้วย

    ถ้าไม่กรองออก โมเดล llama.cpp ที่รันโหมด docker จะถูกนับซ้ำเป็น "native orphan" อีกตัว —
    เจอจริงบน RTX 5090 (2026-08-03): `lmds list` ขึ้นสองแถวสำหรับเซิร์ฟเวอร์ตัวเดียว
    แถวปลอมใช้ค่า --alias เป็น slug จึงดูเหมือนคนละโมเดล
    """
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "libpod"))


def _orphan_native(known_pids: set[int], busy_ports: set[int] | None = None) -> list[ServerInfo]:
    """llama-server ที่รันอยู่แต่ไม่มีทะเบียน (เช่น start จาก bundle รุ่นเก่า)"""
    orphans: list[ServerInfo] = []
    busy_ports = busy_ports or set()
    for pid, cmdline in _pgrep_llama():
        if pid in known_pids or _in_container(pid):
            continue
        port_value = _cmdline_value(cmdline, "--port")
        # กันซ้ำอีกชั้นเผื่ออ่าน cgroup ไม่ได้: พอร์ตเดียวกันย่อมเป็นเซิร์ฟเวอร์ตัวเดียวกัน
        if port_value.isdigit() and int(port_value) in busy_ports:
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


# image ของ engine ที่เรารู้จัก — ใช้เดาว่า container ที่ไม่ได้มาจาก lmds เป็น model server
_ENGINE_IMAGE_HINTS = {
    "vllm": ("vllm/vllm-openai", "nvcr.io/nvidia/vllm", "vllm"),
    "llamacpp": ("llama.cpp", "llamacpp"),
    "ollama": ("ollama/ollama",),
    "tgi": ("text-generation-inference",),
}


def _engine_from_image(image: str) -> str:
    low = image.lower()
    for engine, hints in _ENGINE_IMAGE_HINTS.items():
        if any(hint in low for hint in hints):
            return engine
    return ""


def _first_published_port(ports: str) -> int:
    """'0.0.0.0:8001->8000/tcp, ...' → 8001 (พอร์ตฝั่ง host ตัวแรก)"""
    for chunk in ports.split(","):
        host_side = chunk.strip().split("->")[0]
        tail = host_side.rsplit(":", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def _orphan_docker(known_containers: set[str]) -> list[ServerInfo]:
    """container ที่รันอยู่แต่ไม่มีทะเบียน — ทั้งชื่อ lmds-* และของที่คนอื่นรันไว้เอง

    ของที่ไม่ได้มาจาก lmds ถูกจับเฉพาะเมื่อ image ตรงกับ engine ที่รู้จัก (vLLM/llama.cpp/
    Ollama/TGI) เพื่อไม่ให้ container อื่นในเครื่อง (ฐานข้อมูล ฯลฯ) โผล่มาปนใน fleet
    """
    if shutil.which("docker") is None:
        return []
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return _parse_docker_ps(proc.stdout, known_containers)


def _parse_docker_ps(output: str, known_containers: set[str]) -> list[ServerInfo]:
    """แปลงผล `docker ps` → ServerInfo (แยกออกมาเป็นฟังก์ชันล้วนเพื่อเทสได้ตรง ๆ)"""
    orphans: list[ServerInfo] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        name = parts[0]
        image = parts[1] if len(parts) > 1 else ""
        ports = parts[2] if len(parts) > 2 else ""
        if name in known_containers:
            continue
        is_lmds = name.startswith("lmds-")
        engine = _engine_from_image(image)
        if not is_lmds and not engine:
            continue  # ไม่ใช่ model server — ไม่เอาเข้ามาใน fleet
        orphans.append(ServerInfo(
            slug=name.removeprefix("lmds-") if is_lmds else name,
            model_id=image,
            engine=engine or "?",
            mode="docker",
            container=name,
            port=_first_published_port(ports),
            running=True,
            registered=False,
            external=not is_lmds,
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
    busy_ports = {s.port for s in servers if s.running and s.port}
    for orphan in [*_orphan_native(known_pids, busy_ports), *_orphan_docker(known_containers)]:
        orphan.healthy = orphan.running and _health_ok(orphan.port)
        servers.append(orphan)
    return servers


def find(slug: str) -> ServerInfo | None:
    return next((s for s in discover() if s.slug == slug), None)


def bundle_profile(controller: str) -> dict | None:
    """อ่าน MODEL_PROFILE.yaml ที่อยู่ข้าง controller — คืน None ถ้าไม่มี/อ่านไม่ได้"""
    if not controller:
        return None
    path = Path(controller).parent / "MODEL_PROFILE.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def profile_context(profile: dict | None) -> int | None:
    """context (max_model_len) จาก profile"""
    ctx = ((profile or {}).get("serving") or {}).get("context")
    return ctx if isinstance(ctx, int) else None


def feature_summary(profile: dict | None) -> str:
    """สรุปฟีเจอร์ที่โมเดลนี้รองรับจาก profile → เช่น 'tools, reasoning, image' หรือ 'text'"""
    feats = (profile or {}).get("features") or {}
    labels: list[str] = []
    if (feats.get("tool_calling") or {}).get("enabled"):
        labels.append("tools")
    if (feats.get("reasoning") or {}).get("enabled"):
        labels.append("reasoning")
    modalities = (feats.get("multimodal") or {}).get("modalities") or []
    labels.extend(m for m in modalities if isinstance(m, str))
    return ", ".join(labels) if labels else "text"


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
    if info.external:
        # ของคนอื่น — หยุดอย่างเดียว ห้ามลบ container ทิ้ง
        subprocess.run(["docker", "stop", info.container], capture_output=True)
        return "docker-stop"
    subprocess.run(["docker", "rm", "-f", info.container], capture_output=True)
    return "docker-rm"


def restart_server(info: ServerInfo) -> str:
    """restart — controller ถ้ามี, ไม่งั้น docker restart (ใช้ได้กับ container ภายนอกด้วย)"""
    if info.controller_exists:
        _run_controller(info, "restart")
        return "controller"
    if info.mode == "docker" and info.container:
        proc = subprocess.run(["docker", "restart", info.container], capture_output=True)
        if proc.returncode != 0:
            raise FleetError(f"docker restart {info.container} ล้มเหลว")
        return "docker-restart"
    raise FleetError(
        f"restart {info.slug} ไม่ได้ — ไม่มี controller และไม่ใช่ container "
        "(หยุดด้วย lmds stop แล้ว start ใหม่เอง)"
    )


def start_server(info: ServerInfo) -> int:
    return _run_controller(info, "start")


def logs_server(info: ServerInfo, lines: int = 200, follow: bool = False) -> int:
    """ดู log — follow=True ตามแบบ realtime (Ctrl-C เพื่อออก)

    controller ไม่มีโหมด follow จึงต่อตรงที่แหล่ง log: docker logs -f / tail -f
    """
    if not follow:
        return _run_controller(info, "logs", [str(lines)])

    if info.mode == "docker" and info.container:
        return subprocess.run(
            ["docker", "logs", "-f", "--tail", str(lines), info.container]
        ).returncode
    log_file = info.run_dir / "server.log" if info.run_dir else None
    if log_file and log_file.is_file():
        return subprocess.run(["tail", "-n", str(lines), "-f", str(log_file)]).returncode
    raise FleetError(
        f"ตาม log ของ {info.slug} แบบ realtime ไม่ได้ — ไม่พบ container หรือไฟล์ log "
        f"({log_file or 'ไม่ระบุ'}) · ใช้แบบไม่ follow แทน: lmds logs {info.slug}"
    )



def logs_text(info: ServerInfo, lines: int = 200) -> str:
    """เหมือน logs_server แต่คืนข้อความแทนพิมพ์ออกจอ — ใช้กับ Web UI/สคริปต์

    controller พิมพ์ตรงไป stdout ของ terminal จึง capture ไม่ได้ผ่าน logs_server
    """
    if info.controller_exists:
        proc = subprocess.run(
            [info.controller, "logs", str(lines)], capture_output=True, text=True
        )
        return (proc.stdout or "") + (proc.stderr or "")
    if info.mode == "docker" and info.container:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(lines), info.container],
            capture_output=True, text=True,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    log_file = info.run_dir / "server.log" if info.run_dir else None
    if log_file and log_file.is_file():
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    raise FleetError(f"ไม่พบแหล่ง log ของ {info.slug}")


# ── Remove / repair ────────────────────────────────────────────────────────────
def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
    except OSError:
        pass
    return total


def weights_path(info: ServerInfo) -> Path | None:
    """ที่เก็บ weight ของโมเดลนี้ — vLLM ใช้ HF cache, llama.cpp ใช้ MODEL_DIR

    คืน None เมื่อเดาไม่ได้ (ไม่มี profile) — ดีกว่าเดามั่วแล้วลบผิดโฟลเดอร์
    """
    profile = bundle_profile(info.controller) if info.controller_exists else None
    engine = ((profile or {}).get("runtime") or {}).get("engine") or info.engine
    model_id = ((profile or {}).get("model") or {}).get("id") or info.model_id
    if engine == "llamacpp":
        candidate = Path(os.environ.get("MODEL_DIR", Path.home() / "models" / info.slug))
        return candidate if candidate.is_dir() else None
    if not model_id or "/" not in model_id:
        return None
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidate = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    return candidate if candidate.is_dir() else None


@dataclass
class RemovalItem:
    label: str
    path: Path
    size_bytes: int
    is_weights: bool = False


def removal_plan(info: ServerInfo, include_weights: bool = True) -> list[RemovalItem]:
    """รายการไฟล์ทั้งหมดที่เกี่ยวกับโมเดลนี้ — ให้ผู้ใช้ดูก่อนยืนยันลบ"""
    items: list[RemovalItem] = []

    if info.controller_exists:
        bundle_dir = Path(info.controller).parent
        items.append(RemovalItem("bundle", bundle_dir, _dir_size_bytes(bundle_dir)))
        zip_path = bundle_dir.with_suffix(".zip")
        if zip_path.is_file():
            items.append(RemovalItem("zip", zip_path, zip_path.stat().st_size))

    run_dir = run_root() / info.slug
    if run_dir.is_dir():
        items.append(RemovalItem("ทะเบียน/log", run_dir, _dir_size_bytes(run_dir)))

    plugin_dir = Path.home() / ".lmds" / "plugins" / info.slug
    if plugin_dir.is_dir():
        items.append(RemovalItem("runtime files", plugin_dir, _dir_size_bytes(plugin_dir)))

    if include_weights:
        weights = weights_path(info)
        if weights is not None:
            items.append(RemovalItem("weight ของโมเดล", weights, _dir_size_bytes(weights), is_weights=True))
    return items


def remove_server(info: ServerInfo, include_weights: bool = True) -> list[str]:
    """หยุด → ยกเลิก autostart → ลบไฟล์ทั้งหมด — คืนรายการสิ่งที่ทำจริง

    ลำดับสำคัญ: ต้องหยุด/ยกเลิก autostart ก่อนลบ ไม่งั้นเหลือ container ค้าง
    หรือ systemd unit ที่ชี้ไปไฟล์ที่ไม่มีแล้ว
    """
    import shutil as _shutil

    done: list[str] = []
    if info.running:
        try:
            done.append(f"หยุดเซิร์ฟเวอร์ ({stop_server(info)})")
        except (FleetError, OSError) as exc:
            done.append(f"หยุดไม่สำเร็จ: {exc}")
    if have_systemctl() and autostart_status(info.slug) in {"enabled", "disabled"}:
        try:
            disable_autostart(info)
            done.append("ยกเลิก autostart")
        except FleetError as exc:
            done.append(f"ยกเลิก autostart ไม่สำเร็จ: {exc}")

    for item in removal_plan(info, include_weights=include_weights):
        try:
            if item.path.is_dir():
                _shutil.rmtree(item.path)
            else:
                item.path.unlink(missing_ok=True)
            done.append(f"ลบ {item.label}: {item.path}")
        except OSError as exc:
            done.append(f"ลบ {item.path} ไม่ได้: {exc}")
    return done


def repair_server(info: ServerInfo) -> int:
    """ดาวน์โหลดไฟล์ที่ขาด/เสียใหม่ แล้วตรวจซ้ำ — download ของทุก controller resume ได้"""
    if not info.controller_exists:
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} — bundle ถูกลบไปแล้ว ซ่อมไม่ได้\n"
            f"สร้างใหม่ด้วย: lmds deploy <ลิงก์โมเดลเดิม>  (weight ที่โหลดไว้ยังใช้ต่อได้ ไม่ต้องโหลดซ้ำ)"
        )
    code = _run_controller(info, "download")
    if code != 0:
        return code
    return _run_controller(info, "verify-files")
