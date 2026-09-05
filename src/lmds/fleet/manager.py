"""Fleet manager — มองเห็น/สั่งงานทุกโมเดลที่ deploy ด้วย LMDS ในเครื่องเดียว

หลักการ: controller ทุกตัวเขียน `server.meta` ใต้ ~/.lmds/run/<slug>/ ตอน start
lmds อ่าน meta ทั้งหมด + เช็คสถานะจริง (pid/container/health) — ไม่ต้องมี daemon
"""

from __future__ import annotations

import getpass
import os
import re
import shlex
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


def user_systemd_dir() -> Path:
    """ที่เก็บ unit ของผู้ใช้ — เขียนได้โดยไม่ต้อง sudo"""
    return Path(os.environ.get("LMDS_USER_SYSTEMD_DIR", str(Path.home() / ".config/systemd/user")))


def _linger_on() -> bool:
    """service ของผู้ใช้จะขึ้นตอนบูตก็ต่อเมื่อเปิด linger ไว้ — ไม่งั้นขึ้นตอน login เท่านั้น"""
    import getpass

    try:
        proc = subprocess.run(["loginctl", "show-user", getpass.getuser(), "-p", "Linger"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "Linger=yes" in proc.stdout


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


def controller_startup_timeout(controller: str | Path) -> int | None:
    """STARTUP_TIMEOUT ที่ controller จะรอ /health จริง — ค่าใน bundle.env ชนะ default ในสคริปต์

    renderer สเกลค่านี้ตามขนาดโมเดล (~30 วิ/GB): stacked DeepSeek-V4 157 GB = 5001 วิ ·
    Qwen3-235B FP8 220 GB = 6906 วิ — ทั้งคู่เกิน 1800 ที่ unit เคยใช้เป็นเพดานตายตัว
    """
    controller = Path(controller)
    try:
        text = controller.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pattern = re.compile(r'^STARTUP_TIMEOUT="\$\{STARTUP_TIMEOUT:-(\d+)\}"', flags=re.M)
    match = pattern.search(text)
    value = int(match.group(1)) if match else None
    # bundle.env ใช้รูปเดียวกัน (NAME="${NAME:-value}") และถูก source ก่อน default ของสคริปต์ — ค่าที่ผู้ใช้
    # ตั้งไว้ตรงนั้นคือค่าที่ start จะรอจริง · ไม่ผ่าน bundle_settings.read เพราะมันคืนเฉพาะ knob ในรายการ
    bundle_env = controller.parent / "bundle.env"
    try:
        override = pattern.search(bundle_env.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        override = None
    if override:
        value = int(override.group(1))
    return value


def render_unit(info: "ServerInfo", timeout: int = 1800, scope: str = "system") -> str:
    """สร้างเนื้อ systemd unit สำหรับ autostart ของ bundle นี้

    - Type=oneshot + RemainAfterExit: controller start เปิด container/process แบบ detach
      แล้ว return เมื่อ health ผ่าน (unit คง active หลัง exec จบ)
    - ExecStartPre=stop: เคลียร์ container/process ค้างจากก่อน reboot ก่อน start ใหม่

    `scope` ตัดสินสองบรรทัดที่ **ห้ามเหมือนกัน** ระหว่างสองสโคป:

    - `User=` มีได้เฉพาะ system unit · user manager รันเป็น user นั้นอยู่แล้ว การสั่ง
      ให้มันสลับ user คือสิ่งที่มันไม่มีสิทธิ์ทำ systemd จึงตายตั้งแต่ยังไม่ทันเรียก
      controller ด้วย `Failed to determine supplementary groups` / `status=216/GROUP`
    - `WantedBy` ต้องเป็น default.target สำหรับ user scope (multi-user.target ไม่มีใน
      user manager)

    ทั้งคู่พังแบบเดียวกันคือ **เงียบจนกว่าจะ reboot**: `systemctl --user is-enabled`
    ตอบ enabled, หน้าเว็บขึ้นแบดจ์ autostart, `loginctl` บอก Linger=yes — แล้วเครื่อง
    บูตขึ้นมาโดยไม่มีโมเดล เจอจริงบน msi-4, spark-worker และ dgx-veerasiam (2026-08-15)
    """
    if not info.controller_exists and info.mode == "docker" and info.container:
        return _render_docker_unit(info, scope)

    controller = info.controller
    workdir = str(Path(controller).parent)
    model = info.model or info.model_id or info.slug
    # TimeoutStartSec ต้องไม่สั้นกว่าเวลาที่ controller เองยอมรอ /health — ไม่งั้น systemd ฆ่า
    # `start` ตอน 30 นาทีระหว่างที่โมเดล 150-220 GB (stacked) ยังโหลดอยู่ · container ที่ detach
    # ไปแล้วรันต่อจนขึ้นเอง แต่ unit เป็น failed และ server.meta/ข้อความ "started" ไม่เคยถูกเขียน
    # → หน้าเว็บ/`lmds ps` บอกว่า autostart ล้มทั้งที่โมเดลรันอยู่ (เจอเมื่อดู timeout ของ bundle
    # stacked จริง 2026-09-04: 5001 และ 6906 วิ) · +300 เผื่อ ExecStartPre=stop กับ worker-first
    floor = controller_startup_timeout(controller)
    if floor:
        timeout = max(timeout, floor + 300)
    lines = [
        "[Unit]",
        f"Description=LMDS model: {info.slug} ({model})",
        "After=network-online.target docker.service",
        "Wants=network-online.target docker.service",
        "",
        "[Service]",
        "Type=oneshot",
        "RemainAfterExit=yes",
    ]
    if scope == "system":
        lines.append(f"User={_controller_owner(controller)}")
        lines.append(f"Environment=HOME={Path.home()}")
    lines += [
        f"WorkingDirectory={workdir}",
        f"ExecStartPre=-{controller} stop",
        f"ExecStart={controller} start",
        f"ExecStop={controller} stop",
        f"TimeoutStartSec={timeout}",
        "Restart=no",
        "",
        "[Install]",
        f"WantedBy={_wanted_by(scope)}",
        "",
    ]
    return "\n".join(lines)


def _wanted_by(scope: str) -> str:
    return "default.target" if scope == "user" else "multi-user.target"


def autostart_status(slug: str) -> str:
    """คืน 'enabled' | 'disabled' | 'absent' (ไม่มี unit) | 'n/a' (ไม่มี systemd)

    ดูทั้ง user unit และ system unit — เปิดแบบ user แล้วรายงานว่า absent คือบอกผิด
    """
    if not have_systemctl():
        return "n/a"
    name = unit_name(slug)
    if (user_systemd_dir() / name).exists():
        proc = subprocess.run(["systemctl", "--user", "is-enabled", name],
                              capture_output=True, text=True)
        out = proc.stdout.strip()
        return out if out in {"enabled", "disabled"} else ("enabled" if proc.returncode == 0 else "disabled")
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


def _render_docker_unit(info: "ServerInfo", scope: str = "system") -> str:
    """unit สำหรับ container ที่ไม่ได้มาจาก lmds — แค่ start container เดิมกลับมาหลัง reboot

    ไม่มี controller ให้เรียก จึงทำได้แค่ `docker start` (ไม่ได้สร้าง container ใหม่)
    ถ้า container ถูกลบไป unit นี้จะล้ม — ต้อง enable ใหม่หลังสร้าง container ใหม่

    `User=` เฉพาะ system scope ด้วยเหตุผลเดียวกับ `render_unit`
    """
    import getpass

    lines = [
        "[Unit]",
        f"Description=LMDS (adopted container): {info.container}",
        "After=network-online.target docker.service",
        "Wants=network-online.target docker.service",
        "Requires=docker.service",
        "",
        "[Service]",
        "Type=oneshot",
        "RemainAfterExit=yes",
    ]
    if scope == "system":
        lines.append(f"User={getpass.getuser()}")
    lines += [
        f"ExecStart=/usr/bin/docker start {info.container}",
        f"ExecStop=/usr/bin/docker stop {info.container}",
        "",
        "[Install]",
        f"WantedBy={_wanted_by(scope)}",
        "",
    ]
    return "\n".join(lines)


def _enable_user_unit(info, name: str, timeout: int, start_now: bool, adopted: bool) -> str:
    """autostart แบบ **ไม่ต้องใช้ sudo เลย** — systemd user service

    ทำไมเป็นค่าเริ่มต้น: hub สั่งข้ามเครื่องผ่าน SSH ซึ่งไม่มี tty ให้กรอกรหัส sudo
    ปุ่ม enable บนหน้าเว็บจึงล้มเสมอบนเครื่องที่ sudo ต้องใช้รหัสผ่าน (ซึ่งคือค่าปกติ)
    · และการเปิดทาง sudo ให้เขียน unit ของ **ระบบ** ได้ = ให้สิทธิ์เท่ากับ root
      เพราะ unit รันคำสั่งอะไรก็ได้ในนามของ root — ไม่ใช่สิทธิ์แคบอย่างที่ดูเผิน ๆ
    """
    directory = user_systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # render ด้วย scope ของตัวเอง ไม่ใช่ render แบบ system แล้วค่อยแก้ทีหลัง — วิธีเดิม
    # แก้ WantedBy ได้บรรทัดเดียวและปล่อย User= ค้างไว้ ซึ่งทำให้ unit ล้มตอนบูตทุกตัว
    (directory / name).write_text(render_unit(info, timeout, scope="user"), encoding="utf-8")

    steps = [["systemctl", "--user", "daemon-reload"],
             ["systemctl", "--user", "enable", name]]
    if start_now:
        steps.append(["systemctl", "--user", "start", name])
    for cmd in steps:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FleetError(
                f"เปิด autostart (user service) ไม่สำเร็จ: {' '.join(cmd)}\n"
                f"{(proc.stderr or '').strip()[:300]}"
            )
    if start_now:
        _verify_started(["systemctl", "--user"], name)
    return name


def _verify_started(systemctl: list[str], name: str) -> None:
    """พิสูจน์ว่า unit ที่เพิ่งสั่ง start ขึ้นมาจริง ไม่ใช่แค่คำสั่ง start คืน 0

    `systemctl start` ของ oneshot คืน 0 เมื่อ ExecStart จบด้วยดี แต่ unit ที่ล้ม
    ตั้งแต่ก่อนถึง ExecStart (เช่น `User=` ใน user unit → status 216/GROUP) ก็เคย
    หลุดผ่านมาแล้ว เพราะไม่มีใครถาม is-active ต่อ · is-enabled บอกแค่ว่า "จะถูกเรียก
    ตอนบูต" ไม่ได้บอกว่า "เรียกแล้วขึ้น" — ช่องว่างนี้คือที่ที่ autostart พังเงียบ ๆ
    ไปทั้งกองโดยไม่มีใครรู้จนกว่าจะ reboot จริง

    ไม่โยน error ถ้า start ไม่ผ่านเพราะยังโหลด weight อยู่ (activating) — เฉพาะ
    failed เท่านั้นที่ถือว่าพัง
    """
    active = subprocess.run([*systemctl, "is-active", name],
                            capture_output=True, text=True).stdout.strip()
    if active in ("active", "activating"):
        return
    scope = "--user" if "--user" in systemctl else "--system"
    journal = subprocess.run(
        ["journalctl", scope, "-u", name, "-n", "15", "--no-pager"],
        capture_output=True, text=True).stdout.strip()
    raise FleetError(
        f"enable สำเร็จแต่ unit start ไม่ขึ้น (is-active = {active or 'unknown'}) — "
        f"autostart ที่ 'enabled' แบบนี้จะไม่กลับมาหลัง reboot\n"
        f"{journal[-800:] if journal else 'ไม่มี log'}"
    )


def _sudo(cmd: list[str], password: str = ""):
    """รัน sudo — ส่งรหัสผ่านทาง stdin ถ้ามี (ใช้ครั้งเดียว ไม่เก็บที่ไหน)

    `sudo -S` อ่านรหัสจาก stdin แทน tty · เป็นทางเดียวที่ทำงานได้ผ่าน SSH
    รหัสผ่านอยู่แค่ในหน่วยความจำของ process นี้จนกว่ามันจะจบ ไม่ถูกเขียนลงดิสก์
    และไม่ถูกใส่ใน argv (ซึ่งคนอื่นบนเครื่องเดียวกันอ่านได้จาก /proc)
    """
    if not password:
        return subprocess.run(cmd)
    with_stdin = [cmd[0], "-S", "-p", "", *cmd[1:]] if cmd[0] == "sudo" else cmd
    return subprocess.run(with_stdin, input=password + "\n", capture_output=True, text=True)


def effective_autostart_port(info: "ServerInfo") -> str | None:
    """port ที่ unit ของ slug นี้จะเสิร์ฟตอนบูต — ค่าที่ `lmds set` ไว้ หรือ default ในตัว controller

    ตอน start เองส่ง --port ได้ทุกครั้ง แต่ systemd เรียก controller เปล่า ๆ จึงตกไปใช้
    ค่านี้ · เทียบด้วยค่านี้จึงตรงกับสิ่งที่จะเกิดจริงตอน reboot
    """
    from lmds.fleet import bundle_settings
    if not info.controller:
        return None
    bundle_dir = Path(info.controller).parent
    saved = bundle_settings.read(bundle_dir).get("port")
    if saved:
        return saved
    try:
        text = Path(info.controller).read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(r'^API_PORT="\$\{API_PORT:-(\d+)\}"', text, re.M)
    return found.group(1) if found else None


def autostart_port_conflict(info: "ServerInfo") -> tuple[str, str] | None:
    """slug อื่นที่ enable ไว้แล้วและจะชน port กับ slug นี้ตอนบูต — คืน (slug, port) หรือ None"""
    my_port = effective_autostart_port(info)
    if my_port is None:
        return None
    for other in discover():
        if other.slug == info.slug or not other.controller:
            continue
        if autostart_status(other.slug) != "enabled":
            continue
        if effective_autostart_port(other) == my_port:
            return (other.slug, my_port)
    return None


def enable_autostart(info: "ServerInfo", timeout: int = 1800, start_now: bool = False,
                     scope: str = "user", password: str = "") -> str:
    """ติดตั้ง + enable systemd unit — คืนชื่อ unit ที่ติดตั้ง

    `scope="user"` (ค่าเริ่มต้น) ไม่ต้องใช้ sudo เลย · `scope="system"` เขียนลง
    /etc/systemd/system ซึ่งต้อง sudo และเท่ากับให้สิทธิ์ root
    """
    if not have_systemctl():
        raise FleetError("เครื่องนี้ไม่มี systemd (systemctl) — autostart รองรับเฉพาะระบบ systemd")
    adopted = not info.controller_exists and info.mode == "docker" and bool(info.container)
    if not info.controller_exists and not adopted:
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} — ต้องมี bundle หรือเป็น container ที่รันอยู่ก่อนตั้ง autostart"
        )

    # หลายโมเดลบนเครื่องเดียวกัน default port 8000 เท่ากันหมด · ตอน start เองทีละตัว
    # ไม่มีปัญหา แต่พอ enable autostart หลายตัว reboot ทีเดียวมันขึ้นพร้อมกันแล้วชน
    # port เดียว ตัวหลัง ๆ ล้ม · เจอจริงหลายรอบ — จับตั้งแต่ตอน enable ดีกว่าปล่อยให้
    # ไปพังตอนบูตที่ไซต์ลูกค้าโดยไม่มีใครดู
    clash = autostart_port_conflict(info)
    if clash:
        raise FleetError(
            f"port {clash[1]} ชนกับ '{clash[0]}' ที่ตั้ง autostart ไว้แล้ว — "
            f"ทั้งคู่จะขึ้น port เดียวกันตอน reboot แล้วตัวหลังล้ม\n"
            f"ตั้ง port อื่นก่อน: lmds set {info.slug} --port <พอร์ตที่ยังว่าง>"
        )

    name = unit_name(info.slug)
    if scope == "user":
        return _enable_user_unit(info, name, timeout, start_now, adopted)
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
        proc = _sudo(cmd, password)
        if proc.returncode != 0:
            manual = "\n  ".join(" ".join(c) for c in steps)
            raise FleetError(
                f"ติดตั้ง autostart ไม่สำเร็จ (คำสั่ง `{' '.join(cmd)}` ล้มเหลว)\n"
                f"ลองรันมือ:\n  {manual}"
            )
    if start_now:
        _verify_started(["sudo", "systemctl"], name)
    return name


def disable_autostart(info_or_slug, password: str = "") -> str:
    """disable + ลบ systemd unit (ต้องใช้ sudo) — รับ ServerInfo หรือ slug

    ตรวจผลลัพธ์จริงหลังรัน ไม่ใช่เชื่อว่า sudo สำเร็จ · เดิมกลืน error ทุกตัวแล้วรายงานว่า
    "ปิดแล้ว" เสมอ — sudo ที่ขอรหัสผ่านไม่ได้ (เช่นถูกเรียกผ่าน SSH) จึงกลายเป็นคำตอบว่า
    สำเร็จ ทั้งที่ autostart ยังเปิดอยู่ ผู้ใช้จะรู้ตัวอีกทีตอน reboot แล้วโมเดลเด้งขึ้นมาเอง
    """
    slug = info_or_slug.slug if isinstance(info_or_slug, ServerInfo) else info_or_slug
    if not have_systemctl():
        raise FleetError("เครื่องนี้ไม่มี systemd (systemctl)")
    name = unit_name(slug)
    if autostart_status(slug) == "absent":
        raise FleetError(f"ไม่ได้ตั้ง autostart ของ {slug} ไว้อยู่แล้ว (ไม่มี {name})")

    # เปิดแบบ user ก็ต้องปิดแบบ user — ไม่งั้นสั่ง disable แล้วมันยังขึ้นเองอยู่
    user_unit = user_systemd_dir() / name
    if user_unit.exists():
        _bounded(["systemctl", "--user", "disable", "--now", name], capture_output=True)
        user_unit.unlink(missing_ok=True)
        _bounded(["systemctl", "--user", "daemon-reload"], capture_output=True)
        if autostart_status(slug) != "absent":
            raise FleetError(f"ปิด autostart ไม่สำเร็จ — unit {name} ยังอยู่")
        return name
    steps = [
        ["sudo", "systemctl", "disable", "--now", name],
        ["sudo", "rm", "-f", str(systemd_dir() / name)],
        ["sudo", "systemctl", "daemon-reload"],
    ]
    for cmd in steps:
        # ตัว disable เองอาจ error ถ้า unit ไม่ได้ enable ไว้ — ไม่ fatal จึงไม่หยุดตรงนี้
        _sudo(cmd, password)
    # เกณฑ์เดียวที่นับ: สุดท้ายมันหายไปจริงไหม
    if autostart_status(slug) != "absent":
        manual = "\n  ".join(" ".join(c) for c in steps)
        raise FleetError(
            f"ปิด autostart ไม่สำเร็จ — unit {name} ยังอยู่ (มักเพราะ sudo ขอรหัสผ่านไม่ได้)\n"
            f"ลองรันมือบนเครื่องนั้น:\n  {manual}"
        )
    return name


@dataclass
class ServerInfo:
    slug: str
    model: str = ""
    # ชื่อที่ตอน generate ตั้งไว้ — ว่าง = bundle รุ่นก่อนมี --name หรือยังไม่เคย start
    default_model: str = ""
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


def _health_ok(port: int, engine: str = "") -> bool:
    """เช็คว่ายังเสิร์ฟอยู่ไหม — โดยไม่ไปปลุก GPU

    /health ของ SGLang **รันโมเดลจริงหนึ่งรอบ** ทุกครั้งที่ถูกเรียก (prefill 1 token)
    ไม่ใช่แค่ตอบว่ายังไม่ตาย · LMDS เช็คสถานะเป็นระยะทั้งจาก CLI และหน้าเว็บ โมเดลจึง
    ถูกสั่งคิดตลอดเวลาโดยไม่มีใครถามอะไรเลย

    เคสจริง 2026-09-01 บน spark-head: ผู้ใช้เห็น GPU 78% ทั้งที่ยังไม่เคยยิงคำสั่ง —
    log ฝั่งเซิร์ฟเวอร์มีแต่ `GET /health` 11 ครั้งใน 3 นาที และ `Prefill batch` 11 ครั้ง
    ตรงกันหนึ่งต่อหนึ่ง ไม่มี request อื่นเลย

    /v1/models ของ SGLang ให้คำตอบเดียวกัน (API ตอบได้ = โมเดลโหลดจบแล้ว) โดยไม่แตะ GPU
    ส่วน vLLM กับ llama.cpp ใช้ /health ตามเดิม เพราะของสองตัวนั้นถูกอยู่แล้ว และ
    แยกสถานะ "กำลังโหลด" (503) ออกจาก "พร้อม" ได้ ซึ่ง /v1/models ทำไม่ได้
    """
    if not port:
        return False
    path = "/v1/models" if engine == "sglang" else "/health"
    try:
        return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2.0).status_code == 200
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


# ธงที่บอก context จริงของแต่ละ engine — ค่าที่ *กำลังรัน* ไม่ใช่ค่าที่ bundle ตั้งไว้
_CONTEXT_FLAGS = ("--ctx-size", "--max-model-len")


def running_context(info: "ServerInfo") -> int | None:
    """context ที่ server ตัวนี้รันอยู่จริง — None ถ้าอ่านไม่ได้

    ทำไมต้องมี: ผู้ใช้ตั้ง context ตอน start (65,600) แต่หน้าเว็บโชว์ค่าใน bundle (16,384)
    ต่อไปเรื่อย ๆ — ดูแล้วเหมือนช่องที่กรอกไม่ทำงาน ทั้งที่ทำงานถูกต้อง
    ค่าที่แสดงต้องเป็นค่าที่ใช้จริง ไม่ใช่ค่าที่ตั้งใจไว้
    """
    if not info.running:
        return None
    words: list[str] = []
    if info.mode == "docker" and info.container:
        try:
            proc = subprocess.run(
                ["docker", "inspect", info.container, "--format", "{{join .Args \" \"}}"],
                capture_output=True, text=True, timeout=10)
            words = proc.stdout.split() if proc.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            words = []
    else:
        pid = info.pid
        if not pid and info.pid_file:
            try:
                pid = int(Path(info.pid_file).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = 0
        if pid:
            try:
                words = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    "utf-8", "replace").split("\0")
            except OSError:
                words = []
    for flag in _CONTEXT_FLAGS:
        if flag in words:
            index = words.index(flag)
            if index + 1 < len(words) and words[index + 1].isdigit():
                return int(words[index + 1])
    return None


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
    # sglang มาก่อน vllm: image ของ SGLang บางตัวมีคำว่า vllm ติดมาด้วยเพราะติดตั้งไว้เป็น
    # dependency · จับ vllm ก่อนจะได้เครื่องยนต์ผิด แล้ว /health ก็ไปปลุก GPU ทุกครั้งที่เช็ค
    "sglang": ("lmsysorg/sglang", "nvcr.io/nvidia/sglang", "sglang"),
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



def _drop_dead_registration(meta_path: Path) -> None:
    """เอาทะเบียนที่ชี้ไปของที่ไม่มีแล้วออก — self-healing ถ้า bundle กลับมาก็ลงทะเบียนใหม่เอง"""
    try:
        meta_path.unlink()
        parent = meta_path.parent
        if not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


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
            default_model=meta.get("default_model", ""),
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
        info.healthy = info.running and _health_ok(info.port, info.engine)

        # ทะเบียนของ bundle ที่ "ไม่เคยถูก start" และ controller ก็ไม่อยู่แล้ว = ตายสนิท
        # เกิดตอน generate ไว้ที่อื่นแล้วลบทิ้ง (เครื่อง hub ที่ใช้สร้างอย่างเดียวจะเต็มไปด้วยรายการปลอม)
        # ต่างจากตัวที่ "เคยรันจริง" แล้ว controller หายไป — อันนั้นต้องเตือน ไม่ใช่เก็บกวาดเงียบ ๆ
        # ลบเฉพาะไฟล์ทะเบียน ไม่แตะ weight หรือ bundle · bundle กลับมาก็ลงทะเบียนใหม่เอง
        never_started = not info.started_at
        if never_started and not info.running and info.controller \
                and not Path(info.controller).exists():
            _drop_dead_registration(meta_path)
            continue
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
        orphan.healthy = orphan.running and _health_ok(orphan.port, orphan.engine)
        servers.append(orphan)

    # bundle ที่อยู่บนดิสก์แต่ยังไม่เคย start — ต้องเห็นด้วย ไม่งั้น deploy เสร็จแล้วไปต่อไม่ถูก
    known = {s.slug for s in servers}
    for bundle in _scan_bundles(known):
        bundle.healthy = False
        servers.append(bundle)
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


def _pick_controller(directory: Path, profile_path: Path) -> Path | None:
    """controller ของโฟลเดอร์นี้ — ตาม topology ใน MODEL_PROFILE.yaml เมื่อมีทั้ง *-single.sh และ *-stacked.sh"""
    singles = sorted(directory.glob("*-single.sh"))
    stacked = sorted(directory.glob("*-stacked.sh"))
    if singles and stacked:
        try:
            import yaml

            topology = str((yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}).get("topology") or "")
        except Exception:  # noqa: BLE001
            topology = ""
        return stacked[0] if topology == "stacked" else singles[0]
    return next(iter(singles + stacked), None)


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
    moe = feats.get("moe") or {}
    if moe.get("experts"):
        active = moe.get("experts_active")
        labels.append(f"MoE {moe['experts']}e/{active}a" if active else f"MoE {moe['experts']}e")
    spec = feats.get("speculative") or {}
    if spec.get("draft_files") or spec.get("embedded"):
        labels.append("MTP")
    embedding = feats.get("embedding") or {}
    if embedding:
        pooling = embedding.get("pooling")
        labels.append(f"embedding ({pooling})" if pooling else "embedding")
    return ", ".join(labels) if labels else "text"


class FleetError(Exception):
    pass


def _suggest_node() -> str:
    """ชื่อเครื่องสักตัวในทะเบียนไว้เติมในคำแนะนำ — ไม่มีก็ปล่อยว่าง อย่าให้ล้มเพราะเรื่องนี้"""
    try:
        from lmds.nodes import load

        nodes = load()
        return nodes[0].name if nodes else ""
    except Exception:
        return ""


def _guard_serving(info: "ServerInfo", action: str, force: bool = False) -> None:
    """ห้ามคำสั่งที่กินทรัพยากรหนักบนเครื่องที่รันโมเดลไม่ได้

    เคสจริง 2026-08-19: `lmds repair` บน hub VM (ไม่มี GPU/docker/llama.cpp, RAM 12 GB)
    เริ่มดูด weight 15.6 GB ลงมาอย่างว่าง่าย — ไฟล์ที่ต่อให้โหลดจบก็ไม่มีอะไรรันมันได้
    """
    from lmds.hardware import serving

    message = serving.guard(info.slug, action, _suggest_node(), force=force)
    if message:
        raise FleetError(message)


def _run_controller(info: ServerInfo, command: str, extra: list[str] | None = None) -> int:
    if not info.controller_exists:
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} ({info.controller or 'ไม่ระบุ'}) — "
            "bundle อาจถูกย้าย/ลบ (ใช้ lmds stop จะ fallback หยุดตรง ๆ ให้)"
        )
    proc = subprocess.run([info.controller, command, *(extra or [])])
    return proc.returncode


# เพดานเวลาของคำสั่งที่ "ต้องจบเอง" — docker/systemctl ที่ถูกเรียกจากหน้าเว็บ
#
# เคสจริง 2026-08-14: กด Remove บนคอนโซล แล้ว `docker stop` ค้าง คำขอไม่เคยตอบ
# systemd ฆ่า lmds-web ทิ้งทั้งตัว (`Failed with result 'timeout'`) ผู้ใช้เห็นปุ่ม
# ค้างที่ "Removing…" แล้วต้องรีเฟรชเอง · คำสั่งเดียวที่แขวนไม่ควรล้มทั้งคอนโซล
#
# ไม่ใส่กับคำสั่งที่ *ตั้งใจ* ให้ยาว: `tail -f`, controller ที่สตรีมออกหน้าจอ,
# หรือ prompt ที่รอผู้ใช้พิมพ์ — พวกนั้นรันจาก CLI ไม่ใช่จากคำขอ HTTP
COMMAND_TIMEOUT = 30


def _bounded(args, **kwargs):
    """subprocess.run ที่ยอมแพ้เมื่อถึงเวลา แทนที่จะแขวนคำขอไว้ตลอดกาล"""
    kwargs.setdefault("timeout", COMMAND_TIMEOUT)
    try:
        return subprocess.run(args, **kwargs)
    except subprocess.TimeoutExpired:
        # คืนผลว่า "ไม่สำเร็จ" ให้ผู้เรียกตัดสินใจต่อ ดีกว่าโยนขึ้นไปทำให้ทั้งคำขอพัง
        return subprocess.CompletedProcess(args, returncode=124, stdout="", stderr="")


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
        _bounded(["docker", "stop", info.container], capture_output=True)
        return "docker-stop"
    _bounded(["docker", "rm", "-f", info.container], capture_output=True)
    return "docker-rm"


def restart_server(info: ServerInfo, options: list[str] | None = None) -> str:
    """restart — controller ถ้ามี, ไม่งั้น docker restart (ใช้ได้กับ container ภายนอกด้วย)"""
    if info.controller_exists:
        _run_controller(info, "restart", options)
        return "controller"
    if info.mode == "docker" and info.container:
        proc = _bounded(["docker", "restart", info.container], capture_output=True)
        if proc.returncode != 0:
            raise FleetError(f"docker restart {info.container} ล้มเหลว")
        return "docker-restart"
    raise FleetError(
        f"restart {info.slug} ไม่ได้ — ไม่มี controller และไม่ใช่ container "
        "(หยุดด้วย lmds stop แล้ว start ใหม่เอง)"
    )


def start_server(info: ServerInfo, options: list[str] | None = None,
                 force: bool = False) -> int:
    """options = flag ของ controller เช่น ["--port", "8001"] — ส่งผ่านไปตรง ๆ

    controller เป็นเจ้าของ flag พวกนี้ (แต่ละ engine มีไม่เท่ากัน) LMDS จึงไม่พยายาม
    รู้จักทุกตัว แค่ส่งต่อและปล่อยให้ controller ตรวจค่าเอง — มันตรวจอยู่แล้ว
    """
    _guard_serving(info, "start", force)
    return _run_controller(info, "start", options)


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





def bundle_roots() -> list[Path]:
    """ที่ที่ bundle มักอยู่ — `lmds deploy` เขียนลง ./bundles ของโฟลเดอร์ที่รันคำสั่ง
    ซึ่งต่างกันไปตามว่าผู้ใช้ยืนอยู่ตรงไหนตอน deploy

    เพิ่มที่อื่นเองได้ด้วย LMDS_BUNDLE_DIRS (คั่นด้วย :)
    """
    roots: list[Path] = []
    extra = os.environ.get("LMDS_BUNDLE_DIRS", "")
    roots += [Path(p).expanduser() for p in extra.split(":") if p.strip()]
    roots.append(Path.cwd() / "bundles")
    home = Path.home()
    roots.append(home / "bundles")
    try:
        # โฟลเดอร์โปรเจกต์ที่ clone ไว้ เช่น ~/AutoDeployDGXProject/bundles
        roots += [d / "bundles" for d in home.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        pass
    return [r for r in dict.fromkeys(roots) if r.is_dir()]


def _scan_bundles(known_slugs: set[str]) -> list[ServerInfo]:
    """bundle ที่อยู่บนดิสก์แต่ยังไม่มีทะเบียน (สร้างก่อนมี register_bundle หรือ copy มาจากเครื่องอื่น)

    ไม่งั้นผู้ใช้สร้าง bundle เสร็จแล้วมองไม่เห็นใน `lmds list`/หน้าเว็บ เลยไปต่อไม่ถูก
    """
    found: list[ServerInfo] = []
    for root in bundle_roots():
        # `<root>/<slug>/` เป็นรูปแบบปกติ · เผื่ออีกชั้นสำหรับคนที่ deploy ซ้อนโฟลเดอร์ bundles
        for pattern in ("*/MODEL_PROFILE.yaml", "*/*/MODEL_PROFILE.yaml"):
            for profile_path in sorted(root.glob(pattern)):
                directory = profile_path.parent
                slug = directory.name
                if slug in known_slugs:
                    continue
                # ถ้ามีทั้งสองแบบ (bundle เก่าก่อน 0.6.0 ที่ render ทับ) เลือกตามที่ MODEL_PROFILE บอก ไม่ใช่ single ก่อนเสมอ
                controller = _pick_controller(directory, profile_path)
                if controller is None:
                    continue
                known_slugs.add(slug)
                from lmds.fleet import bundle_settings

                profile = bundle_profile(str(controller)) or {}
                model = profile.get("model") or {}
                runtime = profile.get("runtime") or {}
                found.append(ServerInfo(
                    slug=slug,
                    model=model.get("served_name", slug),
                    default_model=model.get("served_name", slug),
                    model_id=model.get("id", ""),
                    engine=runtime.get("engine", ""),
                    mode="docker",
                    # bundle.env ชนะ profile — เป็นค่าที่ start จะใช้จริง
                    port=int(bundle_settings.read(controller.parent).get("port")
                             or (profile.get("serving") or {}).get("port") or 8000),
                    container=f"lmds-{slug}",
                    controller=str(controller),
                    registered=False,
                ))
    return found


def register_bundle(controller: Path | str) -> Path:
    """ลงทะเบียน bundle ที่เพิ่ง generate ให้ fleet เห็นทันที (ยังไม่เคย start)

    ปกติ controller เขียน server.meta เองตอน start — ก่อนหน้านั้น `lmds list`/หน้าเว็บจึงมองไม่เห็น
    bundle ที่เพิ่งสร้าง ผู้ใช้เลยไม่รู้ว่าต้องไปต่อยังไง (เจอจริงจากการใช้หน้าเว็บ)
    ทะเบียนนี้เก็บแค่ว่า bundle อยู่ที่ไหน — สถานะจริง (running/health) ยังตรวจสดทุกครั้งเหมือนเดิม
    """
    controller = Path(controller)
    slug = controller.parent.name
    profile = bundle_profile(str(controller)) or {}
    model = profile.get("model") or {}
    runtime = profile.get("runtime") or {}
    serving = profile.get("serving") or {}

    run_dir = run_root() / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = run_dir / "server.meta"
    if meta.exists():
        return meta  # เคย start แล้ว — ของจริงจาก controller ละเอียดกว่า อย่าเขียนทับ

    # พอร์ตที่บันทึกไว้กับ bundle (bundle.env — จาก wizard ที่เลือกพอร์ตว่างให้ หรือ `lmds set --port`)
    # ต้องชนะ 8000 ตั้งแต่ยังไม่เคย start · ไม่งั้นการ์ด/หน้าภาพรวมขึ้น "port shared" กับ bundle ที่จะ
    # ไม่มีวันรันที่ 8000 และ suggest_port ก็นับ 8000 เป็นของมันโดยไม่จำเป็น (audit 2026-09-04)
    from lmds.fleet.bundle_settings import read as read_settings

    saved_port = (read_settings(controller.parent) or {}).get("port", "")
    port = int(saved_port) if str(saved_port).isdigit() else serving.get("port", 8000)
    meta.write_text(
        f"slug={slug}\n"
        f"model={model.get('served_name', slug)}\n"
        f"default_model={model.get('served_name', slug)}\n"
        f"model_id={model.get('id', '')}\n"
        f"engine={runtime.get('engine', '')}\n"
        f"mode={'native' if runtime.get('native_build') else 'docker'}\n"
        f"port={port}\n"
        f"container=lmds-{slug}\n"
        "pid_file=\n"
        f"controller={controller}\n"
        "started_at=\n",
        encoding="utf-8",
    )
    return meta


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
    # bundle ที่มาจาก `lmds adopt` บันทึกตำแหน่ง weight ที่อ่านได้จาก bind mount ไว้ใน profile["weights"]
    # (audit 2026-09-04) — ใช้ค่านั้นก่อนเดาจากชื่อโมเดล · เชื่อเฉพาะ path ใต้ home ของผู้ใช้
    recorded = str(((profile or {}).get("weights") or {}).get("path") or "")
    if recorded:
        candidate = Path(recorded).expanduser()
        try:
            candidate.resolve().relative_to(Path.home().resolve())
        except ValueError:
            candidate = None
        if candidate is not None and candidate.exists():
            return candidate
    engine = ((profile or {}).get("runtime") or {}).get("engine") or info.engine
    model_id = ((profile or {}).get("model") or {}).get("id") or info.model_id
    if engine == "llamacpp":
        # ห้ามอ่าน MODEL_DIR จาก environ ของ process นี้ — ค่านั้นเป็นของ bundle ที่กำลัง start อยู่
        # (controller_env) ไม่ใช่ของโมเดลตัวนี้ → `lmds remove` เคยได้โฟลเดอร์ของโมเดลอื่นไปลบ
        # (รีวิว 2026-09-04) · ที่เก็บของ llama.cpp bundle คือ ~/models/<slug> เสมอ
        candidate = Path.home() / "models" / info.slug
        return candidate if candidate.is_dir() else None
    if not model_id or "/" not in model_id:
        return None
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = f"models--{model_id.replace('/', '--')}"
    # HF cache มีสองเลย์เอาต์ — เลย์เอาต์เก่า ($HF_HOME/models--X) คือที่ที่ weight ซึ่งผู้ใช้
    # โหลดเองมักไปอยู่ · รู้จักแค่ hub/ แปลว่า `remove` รายงานว่าไม่มี weight แล้วทิ้งไว้ทั้งก้อน
    for candidate in (hf_home / "hub" / slug, hf_home / slug):
        if candidate.is_dir():
            return candidate
    return None


@dataclass
class RemovalItem:
    label: str
    path: Path
    size_bytes: int
    is_weights: bool = False
    # ของบน worker ของ stacked — ลบผ่าน ssh จาก head · ว่าง = เครื่องนี้
    node: str = ""
    ssh_user: str = ""
    # path = โฟลเดอร์/ไฟล์ · container = ชื่อ container (path เก็บชื่อ)
    kind: str = "path"
    # ssh ไป worker ไม่ผ่านตอนวางแผน — ขนาดไม่รู้ และตอนลบต้องรายงานว่ายังเหลือ ไม่ใช่ข้ามเงียบ
    reachable: bool = True


def _bundle_env_value(bundle_dir: Path, key: str) -> str:
    """ค่าใน bundle.env — รับทั้ง KEY=value และ KEY="${KEY:-value}" ที่ lmds set เขียน"""
    path = bundle_dir / "bundle.env"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(f"{key}="):
            continue
        value = line[len(key) + 1:].strip().strip("\"'")
        match = re.fullmatch(r"\$\{" + re.escape(key) + r":-(.*)\}", value)
        return (match.group(1) if match else value).strip("\"'")
    return ""


def stacked_workers(info: ServerInfo) -> dict | None:
    """worker ของ bundle stacked นี้ + path ที่ sync-worker/start เขียนไว้บนแต่ละเครื่อง · None = ไม่ใช่ stacked

    ลบ stacked แล้วเหลือ weight 75–173 GB บน worker ทุกตัว (rsync ไปตอน sync-worker) container
    `lmds-<slug>-worker` และ cache ของ bundle — ผู้ใช้ 2026-09-05: "ลบโมเดล stacked แล้วมันลบที่ worker
    ด้วยไหม" · เดิมไม่ · ที่นี่อ่าน cluster.env ข้าง controller (คีย์ v2 ก่อน ไฟล์เก่าตาม) และ default
    ของ controller (WORKER_HF_HOME = HF_HOME ของ head · flashinfer/vllm cache ใต้ ~/.cache) — path ถูก
    expand บน head แล้วส่งไป worker ตรง ๆ เหมือนที่ controller ทำ
    """
    if not getattr(info, "controller_exists", False):
        return None
    controller = Path(info.controller)
    profile = bundle_profile(info.controller) or {}
    if not (controller.name.endswith("-stacked.sh") or profile.get("topology") == "stacked"):
        return None
    from lmds.fleet.cluster_env import parse_cluster_env, worker_targets

    bundle_dir = controller.parent
    values = parse_cluster_env(bundle_dir / "cluster.env")
    workers = worker_targets(values)
    if not workers:
        return {"workers": [], "bundle_dir": bundle_dir, "missing_cluster_env": True}
    home = Path.home()
    hf_home = (values.get("HF_HOME") or _bundle_env_value(bundle_dir, "HF_HOME")
               or os.environ.get("HF_HOME") or str(home / ".cache" / "huggingface"))
    worker_hf = (values.get("WORKER_HF_HOME") or _bundle_env_value(bundle_dir, "WORKER_HF_HOME") or hf_home)
    worker_fi = (values.get("WORKER_FLASHINFER_CACHE") or _bundle_env_value(bundle_dir, "WORKER_FLASHINFER_CACHE")
                 or values.get("FLASHINFER_CACHE") or _bundle_env_value(bundle_dir, "FLASHINFER_CACHE")
                 or str(home / ".cache" / "flashinfer"))
    model_id = ((profile.get("model") or {}).get("id") or info.model_id or "")
    cache_name = f"models--{model_id.replace('/', '--')}" if "/" in model_id else ""
    user = getpass.getuser()
    # FlashInfer JIT cache ของ bundle นี้อยู่ใต้ <cache>/<16 ตัวแรกของ image id ที่ prepare-runtime ล็อก>
    # — image เดียวกันใช้ร่วมกับ stacked ตัวอื่นได้ จึงลบเฉพาะเมื่อไม่มี bundle อื่นล็อก image เดียวกัน
    fi_key = ""
    lock = Path(hf_home) / f".lmds-image-id-{info.slug}"
    try:
        image_id = lock.read_text(encoding="utf-8").strip()
    except OSError:
        image_id = ""
    if image_id:
        shared = [other for other in Path(hf_home).glob(".lmds-image-id-*")
                  if other != lock and other.read_text(encoding="utf-8", errors="replace").strip() == image_id]
        if not shared:
            fi_key = image_id.removeprefix("sha256:")[:16]
    for worker in workers:
        worker["ssh_user"] = worker["ssh_user"] or user
        paths: list[tuple[str, str, bool]] = []      # (label, path, is_weights)
        if cache_name:
            paths.append(("weight ของโมเดล", f"{worker_hf}/hub/{cache_name}", True))
            paths.append(("weight ของโมเดล (เลย์เอาต์เก่า)", f"{worker_hf}/{cache_name}", True))
            paths.append(("lock ของ HF cache", f"{worker_hf}/hub/.locks/{cache_name}", True))
        paths.append(("สคริปต์ worker", f"/tmp/lmds-{info.slug}", False))
        if fi_key:
            paths.append(("FlashInfer cache ของ bundle", f"{worker_fi}/{fi_key}", False))
        worker["paths"] = paths
        worker["container"] = f"lmds-{info.slug}-worker"
    return {"workers": workers, "bundle_dir": bundle_dir, "image_lock": lock if lock.is_file() else None,
            "missing_cluster_env": False}


def _ssh_argv(user: str, ip: str, script: str) -> list[str]:
    # key ของ head ไป worker ถูกตั้งไว้แล้ว (nodes/cluster_ssh.py เขียน stanza ใน ~/.ssh/config ของ head)
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{user}@{ip}", script]


def _remote_sizes(worker: dict) -> dict[str, int] | None:
    """ขนาดของทุก path บน worker ด้วย ssh เดียว · None = ติดต่อไม่ได้ · path ที่ไม่มี = ไม่อยู่ใน dict"""
    # ตอบเป็น "<ลำดับ>\t<ไบต์>" ไม่ใช่ path — path ไม่ต้องเดินทางกลับมาให้เพี้ยน (quote/space/ชื่อไทย)
    lines = ["set -u"]
    for idx, (_label, path, _w) in enumerate(worker["paths"]):
        q = shlex.quote(path)
        lines.append(f"if [ -e {q} ]; then printf '{idx}\\t%s\\n' \"$(du -sb {q} 2>/dev/null | cut -f1)\"; fi")
    lines.append(f"docker inspect {shlex.quote(worker['container'])} >/dev/null 2>&1 && printf 'container\\t0\\n' || true")
    proc = _bounded(_ssh_argv(worker["ssh_user"], worker["ip"], "\n".join(lines)),
                    capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None
    sizes: dict[str, int] = {}
    for line in (proc.stdout or "").splitlines():
        key, _, size = line.partition("\t")
        if key == "container":
            sizes["@container"] = 0
        elif key.isdigit() and int(key) < len(worker["paths"]):
            try:
                sizes[worker["paths"][int(key)][1]] = int(size or 0)
            except ValueError:
                sizes[worker["paths"][int(key)][1]] = 0
    return sizes


def worker_removal_items(info: ServerInfo, include_weights: bool = True) -> list[RemovalItem]:
    """รายการบน worker ทุกตัว (ขนาดถามผ่าน ssh) — ต่อท้ายแผนของ head"""
    cluster = stacked_workers(info)
    if not cluster or not cluster["workers"]:
        return []
    items: list[RemovalItem] = []
    for worker in cluster["workers"]:
        ip, user = worker["ip"], worker["ssh_user"]
        sizes = _remote_sizes(worker)
        reachable = sizes is not None
        sizes = sizes or {}
        if reachable and "@container" in sizes:
            items.append(RemovalItem(f"container บน worker {ip}", Path(worker["container"]), 0,
                                     node=ip, ssh_user=user, kind="container"))
        elif not reachable:
            items.append(RemovalItem(f"container บน worker {ip} (ติดต่อไม่ได้ — ไม่รู้ว่ามีไหม)",
                                     Path(worker["container"]), 0, node=ip, ssh_user=user,
                                     kind="container", reachable=False))
        for label, path, is_weights in worker["paths"]:
            if is_weights and not include_weights:
                continue
            if reachable and path not in sizes:
                continue
            suffix = "" if reachable else " (ติดต่อไม่ได้ — ขนาดไม่รู้)"
            items.append(RemovalItem(f"{label} บน worker {ip}{suffix}", Path(path), sizes.get(path, 0),
                                     is_weights=is_weights, node=ip, ssh_user=user, reachable=reachable))
    return items


def removal_plan(info: ServerInfo, include_weights: bool = True) -> list[RemovalItem]:
    """รายการไฟล์ทั้งหมดที่เกี่ยวกับโมเดลนี้ — ให้ผู้ใช้ดูก่อนยืนยันลบ (stacked: รวมของบน worker ทุกตัว)"""
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
    cluster = stacked_workers(info)
    if cluster:
        lock = cluster.get("image_lock")
        if lock is not None:
            items.append(RemovalItem("image lock ของ bundle", lock, lock.stat().st_size))
        items += worker_removal_items(info, include_weights=include_weights)
    return items


def _remove_on_worker(worker_items: list[RemovalItem]) -> list[str]:
    """ลบของบน worker หนึ่งเครื่องด้วย ssh เดียว: container → rm -rf → ของ root ผ่าน docker → ตรวจ test -e

    ทุก path รายงานผลของตัวเอง — REMOVED / LEFT (พร้อมขนาดที่เหลือ) · ssh ไม่ผ่านทั้งก้อน = "ยังเหลือบน"
    พร้อมคำสั่งที่รันเองได้ ไม่ใช่ข้ามเงียบ ๆ (ของ 100 GB บนเครื่องที่ผู้ใช้มองไม่เห็นจาก hub)
    """
    first = worker_items[0]
    ip, user = first.node, first.ssh_user
    containers = [i for i in worker_items if i.kind == "container"]
    paths = [i for i in worker_items if i.kind != "container"]
    manual = [f"docker rm -f {shlex.quote(str(c.path))}" for c in containers]
    manual += [f"sudo rm -rf {shlex.quote(str(i.path))}" for i in paths]
    unreachable = [f"ยังเหลือบน {ip}: " + " ".join(str(i.path) for i in worker_items)
                   + f" — ลบเอง: ssh {user}@{ip} {shlex.quote('; '.join(manual))}"]
    if not first.reachable:
        return unreachable
    # ผลตอบเป็น "<สถานะ>\t<ลำดับ>[\t<ไบต์ที่เหลือ>]" — ลำดับอ้างถึงรายการที่ส่งไป path ไม่ต้องเดินทางกลับ
    ordered = containers + paths
    lines = ["set -u"]
    for idx, c in enumerate(containers):
        q = shlex.quote(str(c.path))
        lines.append(f"docker rm -f {q} >/dev/null 2>&1 && echo \"REMOVED\t{idx}\" || echo \"GONE\t{idx}\"")
    for offset, item in enumerate(paths):
        idx = len(containers) + offset
        q = shlex.quote(str(item.path))
        parent = shlex.quote(str(Path(str(item.path)).parent))
        name = shlex.quote(Path(str(item.path)).name)
        lines += [
            f"if [ -e {q} ]; then",
            f"  rm -rf -- {q} 2>/dev/null",
            # ไฟล์ของ root (container เขียนไว้) — root ในคอนเทนเนอร์ลบให้ เหมือน _docker_rm ฝั่ง head
            f"  if [ -e {q} ]; then",
            "    img=\"$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | grep -E 'alpine|busybox|ubuntu|debian' | head -1)\"",
            "    [ -n \"$img\" ] || img=\"$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -1)\"",
            f"    [ -n \"$img\" ] && docker run --rm --entrypoint rm -v {parent}:/x \"$img\" -rf -- /x/{name} >/dev/null 2>&1 && echo \"DOCKER\t{idx}\"",
            "  fi",
            "fi",
            f"if [ -e {q} ]; then echo \"LEFT\t{idx}\t$(du -sb {q} 2>/dev/null | cut -f1)\"; else echo \"REMOVED\t{idx}\"; fi",
        ]
    proc = _bounded(_ssh_argv(user, ip, "\n".join(lines)), capture_output=True, text=True, timeout=1800)
    if proc.returncode == 255 or proc.returncode == 124:
        return unreachable
    done: list[str] = []
    seen: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        status, _, rest = line.partition("\t")
        idx, _, size = rest.partition("\t")
        if not idx.isdigit() or int(idx) >= len(ordered):
            continue
        item = ordered[int(idx)]
        target = str(item.path)
        label = item.label
        if status == "REMOVED":
            done.append(f"ลบ {label}: {ip}:{target}")
        elif status == "GONE":
            done.append(f"ไม่มีอยู่แล้ว {label}: {ip}:{target}")
        elif status == "DOCKER":
            done.append(f"ลบผ่าน docker บน {ip} (ไฟล์เป็นของ root): {target}")
            continue        # บรรทัด REMOVED/LEFT ของรายการเดียวกันตามมา
        elif status == "LEFT":
            try:
                human = _human(int(size or 0))
            except ValueError:
                human = "?"
            done.append(f"เหลือ {label} ที่ลบไม่ได้ ({human}) บน {ip}: {target} — ลบเอง: ssh {user}@{ip} sudo rm -rf {shlex.quote(target)}")
        seen.add(target)
    missing = [str(i.path) for i in worker_items if str(i.path) not in seen]
    if missing:
        done.append(f"ยังเหลือบน {ip}: " + " ".join(missing)
                    + f" (ssh จบด้วย exit {proc.returncode}) — ลบเอง: ssh {user}@{ip} {shlex.quote('; '.join(manual))}")
    return done


def remove_server(info: ServerInfo, include_weights: bool = True) -> list[str]:
    """หยุด → ยกเลิก autostart → ลบไฟล์บน head → ลบของบน worker ทุกตัว (stacked) — คืนรายการสิ่งที่ทำจริง

    ลำดับสำคัญ: ต้องหยุด/ยกเลิก autostart ก่อนลบ ไม่งั้นเหลือ container ค้าง
    หรือ systemd unit ที่ชี้ไปไฟล์ที่ไม่มีแล้ว · worker ทีหลัง head — ต้องอ่าน cluster.env จาก bundle
    ก่อน bundle หาย และ controller `stop` (หยุด container ทุก node) ยังต้องมีสคริปต์ให้เรียก
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

    # แผนครั้งเดียวหลัง stop และก่อนลบ bundle — cluster.env ข้าง controller คือที่เดียวที่บอกว่า worker คือใคร
    # (worker ถูกถามขนาดผ่าน ssh ในนี้ · ถามซ้ำหลังลบ head ไม่ได้แล้ว)
    stacked = stacked_workers(info)
    if stacked and stacked.get("missing_cluster_env"):
        done.append(f"bundle stacked แต่ไม่มี cluster.env — ไม่รู้ว่า worker คือเครื่องไหน · "
                    f"ของบน worker (weight/container lmds-{info.slug}-worker) ต้องลบเอง")
    items = removal_plan(info, include_weights=include_weights)
    remote_items = [item for item in items if getattr(item, "node", "")]
    for item in items:
        if getattr(item, "node", ""):
            continue        # ของบน worker — ลบผ่าน ssh ด้านล่างหลัง head เสร็จ
        try:
            if item.path.is_dir():
                _shutil.rmtree(item.path)
            else:
                item.path.unlink(missing_ok=True)
        except PermissionError as exc:
            # ไฟล์ของ root (container ที่รันเป็น root โหลด weight ลงมา — bundle ที่ adopt มาเป็นแบบนี้
            # แทบทุกตัว) · เดิมจบที่ "ต้องใช้ sudo rm -rf" ซึ่งคนที่สั่งผ่านหน้าเว็บ/ssh ไม่มี tty ให้
            # กรอกรหัส แต่เขาอยู่ในกลุ่ม docker อยู่แล้ว — root ในคอนเทนเนอร์ลบให้ได้ (เคสจริง 2026-09-04)
            if _docker_rm(item.path):
                done.append(f"ลบผ่าน docker (ไฟล์เป็นของ root): {item.path}")
            else:
                done.append(f"ลบ {item.path} ไม่ได้: {exc}")
        except OSError as exc:
            done.append(f"ลบ {item.path} ไม่ได้: {exc}")
        # ไม่ยอมเชื่อว่า rmtree สำเร็จเพราะมันไม่ throw — ถามดิสก์อีกที
        # rmtree ลบสิ่งที่ลบได้แล้วค่อยโยน error ตัวเดียว ของที่เหลือจึงยังอยู่จริง
        # (เคสจริง: weight ที่ container โหลดมาเป็น root — เหลือ 23 GB ทั้งที่รายงานว่าลบเรียบร้อย)
        if item.path.exists():
            remaining = _dir_size(item.path) if item.path.is_dir() else item.path.stat().st_size
            done.append(f"เหลือ {item.label} ที่ลบไม่ได้ ({_human(remaining)}): {item.path}")
        else:
            done.append(f"ลบ {item.label}: {item.path}")
    # worker ทีละเครื่อง (ssh เดียวต่อเครื่อง) — เครื่องที่ติดต่อไม่ได้รายงานพร้อมคำสั่ง ไม่ข้ามเงียบ
    by_node: dict[str, list[RemovalItem]] = {}
    for item in remote_items:
        by_node.setdefault(item.node, []).append(item)
    for node_items in by_node.values():
        done += _remove_on_worker(node_items)
    return done


def _docker_rm(path: Path) -> bool:
    """ลบโฟลเดอร์ที่ user ลบไม่ได้ (ของ root) ด้วย root *ในคอนเทนเนอร์* — คืน True เมื่อหายไปจริง

    ทำไมทางนี้ถึงถูก: คนที่ deploy โมเดลอยู่ในกลุ่ม docker อยู่แล้ว (ไม่งั้นรันโมเดลไม่ได้ตั้งแต่แรก)
    และ `docker run -v` ให้ root ในคอนเทนเนอร์เขียนโฟลเดอร์ที่ mount ได้ — คือสิทธิ์เดียวกับที่
    container ตัวเดิมใช้ตอน *สร้าง* ไฟล์พวกนี้ · ไม่ต้องใช้ sudo ไม่ต้องมี tty

    รั้ว: ลบได้เฉพาะใต้ home ของผู้ใช้หรือ HF cache เท่านั้น และไม่ใช่ตัว home เอง — `-v` ในฐานะ root
    ลบอะไรก็ได้บนเครื่อง จึงจำกัดไว้ที่ที่ที่ remove มีสิทธิ์ลบอยู่แล้วตามปกติ · ใช้ image ที่มีในเครื่อง
    (ไม่ pull — เครื่อง air-gapped และไม่ควรดึงของใหม่แค่เพื่อลบไฟล์)
    """
    if shutil.which("docker") is None:
        return False
    try:
        target = Path(path).resolve()
    except OSError:
        return False
    hf_home = os.environ.get("HF_HOME")
    roots = [Path.home().resolve()] + ([Path(hf_home).resolve()] if hf_home else [])
    if not any(root in target.parents for root in roots):
        return False
    proc = _bounded(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                    capture_output=True, text=True)
    images = [line.strip() for line in (proc.stdout or "").splitlines()
              if line.strip() and "<none>" not in line]
    if proc.returncode != 0 or not images:
        return False
    # image เล็ก ๆ ก่อน — vLLM 20 GB ก็ใช้ได้แต่ start ช้ากว่าโดยไม่จำเป็น
    images.sort(key=lambda name: (not any(k in name for k in ("alpine", "busybox", "ubuntu", "debian")), name))
    done = _bounded(
        ["docker", "run", "--rm", "-v", f"{target.parent}:/x", images[0],
         "rm", "-rf", "--", f"/x/{target.name}"],
        capture_output=True, text=True, timeout=600,
    )
    return done.returncode == 0 and not target.exists()


def removal_failed(lines: list[str]) -> list[str]:
    """บรรทัดที่บอกว่าลบไม่สำเร็จ — ผู้เรียกใช้ตัดสินใจว่าจะรายงานว่าสำเร็จไหม (รวมของที่ยังเหลือบน worker)"""
    return [line for line in lines
            if "ลบไม่ได้" in line or "ไม่สำเร็จ" in line or "ยังเหลือบน" in line or "ต้องลบเอง" in line]


def _dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def repair_server(info: ServerInfo, force: bool = False) -> int:
    """ดาวน์โหลดไฟล์ที่ขาด/เสียใหม่ แล้วตรวจซ้ำ — download ของทุก controller resume ได้"""
    _guard_serving(info, "repair", force)
    if not info.controller_exists:
        # container ที่ไม่ได้มาจาก LMDS ไม่เคยมี bundle เลย — บอกว่า "ถูกลบไปแล้ว" คือเดาผิด
        # และพาไปทางที่ไม่ใช่ (deploy ใหม่ทั้งที่ของรันอยู่ดี ๆ)
        if info.external or info.container:
            raise FleetError(
                f"{info.slug} เป็น container ที่ไม่ได้ deploy ผ่าน LMDS — ไม่มี controller ให้ซ่อม\n"
                f"รับเข้าระบบก่อนเพื่อให้สั่งงานได้ครบ: lmds adopt {info.container or info.slug}\n"
                f"(อ่านคำสั่งที่มันรันอยู่จริงมาทำเป็นสคริปต์ — ตัวที่รันอยู่ไม่ถูกแตะต้อง)"
            )
        raise FleetError(
            f"ไม่พบ controller ของ {info.slug} — bundle ถูกลบไปแล้ว ซ่อมไม่ได้\n"
            f"สร้างใหม่ด้วย: lmds deploy <ลิงก์โมเดลเดิม>  (weight ที่โหลดไว้ยังใช้ต่อได้ ไม่ต้องโหลดซ้ำ)"
        )
    # bundle ที่ผู้ใช้ดูแล weight เอง (มาจาก `lmds adopt` หรือชี้ไปที่ path ตรง ๆ) ไม่มี
    # download/verify-files ในสคริปต์ — เดิมสั่งไปแล้วได้ usage ของ bash กลับมา ซึ่งอ่านไม่รู้เรื่อง
    # และทำให้เข้าใจว่า repair พัง ทั้งที่มันแค่ไม่ใช่เรื่องของ bundle แบบนี้
    from lmds.inventory import controller_commands

    commands = controller_commands(info.controller)
    # อ่าน dispatch table ไม่ออก (สคริปต์เขียนคนละสไตล์) = ไม่ฟันธง ปล่อยให้ลองรันไปตามเดิม
    if commands and "download" not in commands:
        raise FleetError(
            f"{info.slug} เป็น bundle ที่ weight อยู่ในความดูแลของคุณเอง — LMDS ไม่ได้เป็นคนโหลดมา\n"
            f"จึงไม่มีอะไรให้ซ่อม: ไม่รู้ว่าไฟล์ครบชุดคืออะไร และไม่รู้จะโหลดมาจากไหน\n"
            f"ตรวจว่ามันรันได้ไหมด้วย: lmds start {info.slug}   (ดูสาเหตุจริงจาก log ถ้าไม่ขึ้น)\n"
            f"อยากให้ LMDS ดูแล weight ให้: lmds deploy <ลิงก์โมเดลบน Hugging Face>"
        )
    code = _run_controller(info, "download")
    if code != 0:
        return code
    return _run_controller(info, "verify-files")
