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


def enable_autostart(info: "ServerInfo", timeout: int = 1800, start_now: bool = False) -> str:
    """ติดตั้ง + enable systemd unit (ต้องใช้ sudo) — คืนชื่อ unit ที่ติดตั้ง

    เขียนไฟล์ unit ลง bundle dir ก่อน (ไม่ใช้ sudo) แล้ว sudo ติดตั้งเข้า /etc/systemd/system
    ถ้า sudo/systemctl ล้มเหลว → FleetError พร้อมคำสั่งให้รันมือ
    """
    if not have_systemctl():
        raise FleetError("เครื่องนี้ไม่มี systemd (systemctl) — autostart รองรับเฉพาะระบบ systemd")
    if not info.controller_exists:
        raise FleetError(f"ไม่พบ controller ของ {info.slug} — ต้องมี bundle ก่อนตั้ง autostart")

    name = unit_name(info.slug)
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
    subprocess.run(["docker", "rm", "-f", info.container], capture_output=True)
    return "docker-rm"


def start_server(info: ServerInfo) -> int:
    return _run_controller(info, "start")


def logs_server(info: ServerInfo, lines: int = 200) -> int:
    return _run_controller(info, "logs", [str(lines)])
