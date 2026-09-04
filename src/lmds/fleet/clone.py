"""ทำสำเนาโมเดลที่ *รันผ่านแล้ว* จากเครื่องหนึ่งไปอีกเครื่อง — ไม่ต้องโหลดจาก HF ใหม่

ทำไมต้องมี: เครื่องข้าง ๆ ในไซต์เดียวกันถือไฟล์ชุดเดียวกันอยู่แล้ว แต่ทางเดียวที่ระบบ
มีให้คือ `download` ซึ่งดึงจาก Hugging Face ใหม่ทั้งก้อน · IQ4_XS ของ Qwen3.8-Flash-Next
คือ 90.8 GB ที่ 40 MB/s = 38 นาที ทั้งที่สาย 200G ในแร็คเดียวกันใช้เวลาไม่ถึงนาที
ถ้าวิ่งเต็มสาย · ยิ่งอยากทำ failover/กระจายโหลดหลายเครื่อง ยิ่งเสียเวลาเป็นทวีคูณ

**กุญแจไม่เคยออกจาก hub**

node แต่ละเครื่องไม่มี key ของกันและกัน (โดยตั้งใจ — node หนึ่งถูกยึดไม่ควรแปลว่า
ทั้งฟลีตถูกยึด) · แต่การ copy ต้องวิ่ง "ตรง" จากต้นทางไปปลายทาง ไม่ใช่ผ่าน hub
ซึ่งมักเป็นเครื่องเล็ก ๆ ที่จะกลายเป็นคอขวดทันที (90 GB ผ่าน VM บนโน้ตบุ๊ก)

ทางออก: สร้างกุญแจ **ชั่วคราวสำหรับงานนี้ครั้งเดียว**
  1. hub สร้างคู่กุญแจใหม่ในหน่วยความจำ
  2. hub เอา public key ไปฝากที่ปลายทาง (hub เข้าถึงได้อยู่แล้ว) พร้อมมาร์กเกอร์
  3. ต้นทางรับ private key ทาง **stdin** แล้วใส่ ssh-agent ในหน่วยความจำ ไม่เขียนลงดิสก์
  4. rsync ตรงจากต้นทาง → ปลายทาง
  5. hub ถอน public key ออกจากปลายทาง **เสมอ** แม้ copy จะล้ม

สิ่งที่ยังไม่ทำ (โดยตั้งใจ): ไม่แตะ gateway/registry ให้ · คนตัดสินใจว่าจะเอาสำเนานี้
เข้ารับโหลดเมื่อไหร่ ควรเป็นคน ไม่ใช่ผลข้างเคียงของการ copy
"""

from __future__ import annotations

import secrets
import shlex
from dataclasses import dataclass, field


class CloneError(Exception):
    """ปัญหาที่ผู้ใช้แก้ได้ — ข้อความพร้อมแสดงทั้งบน CLI และหน้าเว็บ"""


@dataclass
class ClonePlan:
    slug: str
    source: str
    target: str
    source_addr: str
    target_addr: str
    same_site: bool
    link: str                      # "cluster" = สายเร็ว · "host" = เส้นปกติ
    model_dir: str = ""
    bundle_dir: str = ""
    files: list[tuple[str, int]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files)


def _addr_pair(source, target) -> tuple[str, str, str]:
    """เลือกเส้นทางที่เร็วที่สุดที่ *ทั้งคู่* มี — สายคลัสเตอร์ก่อน แล้วค่อยเส้นปกติ

    cluster_ip คือขาบนการ์ดเร็ว (ConnectX 200G) ที่ตั้งไว้สำหรับ stacked อยู่แล้ว ·
    ถ้ามีทั้งสองฝั่งก็ใช้เส้นนั้น การ copy จะวิ่งบนสายเดียวกับที่ NCCL ใช้
    ไม่ใช่ผ่าน Tailscale relay ที่วัดได้ 82–154 ms
    """
    if source.cluster_ip and target.cluster_ip:
        return source.cluster_ip, target.cluster_ip, "cluster"
    return source.host, target.host, "host"


def plan_clone(slug: str, source_name: str, target_name: str) -> ClonePlan:
    """ตรวจว่าทำได้ไหมและจะวิ่งเส้นไหน — ยังไม่แตะเครื่องปลายทาง"""
    from lmds.nodes import find

    if source_name == target_name:
        raise CloneError("ต้นทางกับปลายทางเป็นเครื่องเดียวกัน")

    source = find(source_name)
    target = find(target_name)
    if source is None:
        raise CloneError(f"ไม่รู้จักเครื่องต้นทาง '{source_name}' — ดู: lmds node list")
    if target is None:
        raise CloneError(f"ไม่รู้จักเครื่องปลายทาง '{target_name}' — ดู: lmds node list")

    src_addr, dst_addr, link = _addr_pair(source, target)
    return ClonePlan(
        slug=slug, source=source_name, target=target_name,
        source_addr=src_addr, target_addr=dst_addr,
        # ไซต์ต่างกันยังทำได้ แต่คนสั่งควรรู้ตัวว่ากำลังลาก 90 GB ข้ามอินเทอร์เน็ต
        same_site=bool(source.site) and source.site == target.site,
        link=link,
    )


# {slug} ต้องเป็นค่าที่ shlex.quote แล้วเท่านั้น — เดิมใส่ดิบ ๆ ทั้งใน ls และ echo
# ซึ่ง slug ที่มาจาก URL ของหน้าเว็บกลายเป็นคำสั่งบนเครื่องต้นทางได้
_FIND_BUNDLE = (
    'dir="$(ls -d ~/bundles/{slug} ~/*/bundles/{slug} ./bundles/{slug} 2>/dev/null | head -1)"; '
    '[ -n "$dir" ] || {{ echo "ไม่พบ bundle "{slug} >&2; exit 1; }}; '
)


def inspect_source(plan: ClonePlan) -> ClonePlan:
    """อ่านว่าไฟล์อยู่ที่ไหนและใหญ่แค่ไหนบนต้นทาง — เอาไว้บอกคนสั่งก่อนเริ่มลาก"""
    from lmds.nodes import NodeError, find, run

    source = find(plan.source)
    script = _FIND_BUNDLE.format(slug=shlex.quote(plan.slug)) + (
        'ctl="$(ls "$dir"/*-single.sh "$dir"/*-stacked.sh 2>/dev/null | head -1)"; '
        '[ -n "$ctl" ] || { echo "ไม่พบ controller ใน $dir" >&2; exit 1; }; '
        # MODEL_DIR ประกาศไว้ในตัว controller เอง — ถามมันดีกว่าเดา path
        'md="$(grep -m1 "^MODEL_DIR=" "$ctl" | sed "s/^MODEL_DIR=//")"; '
        'md="$(eval echo "$md")"; '
        # controller ของ vLLM/SGLang ไม่มี MODEL_DIR — weight อยู่ใน cache ของ Hugging Face
        # (HF_HOME/hub/models--org--name ซึ่งมี blobs/ + snapshots/ ที่เป็น symlink) · เคสจริง
        # 2026-09-03: clone Qwen3.6-35B NVFP4 จาก spark04 → RTX4000 ได้ "ยังไม่มีไฟล์โมเดล ()"
        # ทั้งที่ 22 GB อยู่บนเครื่องครบ · ต้อง copy ทั้งโฟลเดอร์ models--… ไม่ใช่แค่ snapshot
        'if [ -z "$md" ]; then '
        '  hf="$(grep -m1 "^HF_HOME=" "$ctl" | sed "s/^HF_HOME=//")"; hf="$(eval echo "${hf:-\\$HOME/.cache/huggingface}")"; '
        '  mid="$(grep -m1 "^MODEL_ID=" "$ctl" | cut -d= -f2- | tr -d "\\"")"; '
        # HF cache มีสองเลย์เอาต์: มาตรฐาน $HF_HOME/hub/models--X และแบบแบน $HF_HOME/models--X
        # (head ของ stacked ที่โหลดด้วย cache_dir=/cache ก่อน 0.6.0 ได้แบบแบน) · มองแค่ hub/ =
        # "ยังไม่มีไฟล์โมเดล" ทั้งที่ 170 GB อยู่บนเครื่อง — controller เองหาเจอทั้งสองแบบมาตลอด
        '  if [ -n "$mid" ]; then '
        '    sl="$(echo "$mid" | sed "s#/#--#g")"; md="$hf/hub/models--$sl"; '
        '    [ -d "$md" ] || [ ! -d "$hf/models--$sl" ] || md="$hf/models--$sl"; '
        '  fi; '
        'fi; '
        'echo "BUNDLE=$dir"; echo "MODELDIR=$md"; '
        '[ -n "$md" ] && [ -d "$md" ] || { echo "ยังไม่มีไฟล์โมเดลบน '"$(hostname)"' (${md:-controller ไม่บอกที่เก็บ})" >&2; exit 2; }; '
        # -type f ไล่ลึก: layout ของ HF เก็บไฟล์จริงใน blobs/ ส่วน snapshots/ เป็น symlink
        'find "$md" -type f -printf "FILE=%s %P\\n"'
    )
    try:
        result = run(source, script, timeout=120)
    except NodeError as exc:
        raise CloneError(str(exc)) from exc
    if not result.ok:
        raise CloneError((result.stderr or result.stdout).strip()[:400])

    for line in result.stdout.splitlines():
        if line.startswith("BUNDLE="):
            plan.bundle_dir = line[7:].strip()
        elif line.startswith("MODELDIR="):
            plan.model_dir = line[9:].strip()
        elif line.startswith("FILE="):
            size, _, name = line[5:].partition(" ")
            if name:
                plan.files.append((name, int(size)))
    if not plan.files:
        raise CloneError(f"ไม่มีไฟล์โมเดลใน {plan.model_dir} บน {plan.source}")
    return plan


def _authorized_key_line(pubkey: str, marker: str) -> str:
    # restrict = ปิด port-forward / agent-forward / pty ทั้งหมด เหลือแค่รันคำสั่ง
    # กุญแจชั่วคราวไม่ควรเปิดอะไรมากกว่าที่งานนี้ต้องใช้
    return f'restrict {pubkey.strip()} {marker}\n'


def _install_temp_key(target, pubkey: str, marker: str) -> None:
    from lmds.nodes import NodeError, run

    line = _authorized_key_line(pubkey, marker)
    script = (
        'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && '
        'chmod 600 ~/.ssh/authorized_keys && cat >> ~/.ssh/authorized_keys'
    )
    try:
        result = run(target, script, timeout=60, stdin_text=line)
    except NodeError as exc:
        raise CloneError(f"ฝากกุญแจชั่วคราวที่ {target.name} ไม่ได้: {exc}") from exc
    if not result.ok:
        raise CloneError((result.stderr or "").strip()[:300] or "ฝากกุญแจชั่วคราวไม่สำเร็จ")


def revoke_temp_key(target, marker: str) -> bool:
    """ถอนกุญแจชั่วคราวออก — ต้องเรียกเสมอ แม้ copy จะล้มกลางคัน

    ใช้ marker ที่สุ่มต่อครั้งเป็นตัวชี้บรรทัด ไม่ลบด้วย pattern กว้าง ๆ เพราะไฟล์นี้
    มีกุญแจของ hub และของผู้ใช้เองอยู่ด้วย — ลบพลาดคือล็อกตัวเองออกจากเครื่อง
    """
    from lmds.nodes import NodeError, run

    quoted = shlex.quote(marker)
    script = (
        f'f=~/.ssh/authorized_keys; [ -f "$f" ] || exit 0; '
        f'grep -v -F {quoted} "$f" > "$f.tmp" && mv "$f.tmp" "$f" && chmod 600 "$f"'
    )
    try:
        return run(target, script, timeout=60).ok
    except NodeError:
        return False


def build_rsync_command(plan: ClonePlan, target_user: str, dry_run: bool = False) -> str:
    """คำสั่งที่จะรัน *บนเครื่องต้นทาง* — key อยู่ใน ssh-agent ในหน่วยความจำเท่านั้น"""
    dest = f"{target_user}@{plan.target_addr}"
    md = shlex.quote(plan.model_dir)
    bd = shlex.quote(plan.bundle_dir)
    ssh_opts = ("ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR")
    # --partial --append-verify = ขาดกลางคันแล้วต่อได้ ไม่เริ่มใหม่ (ไฟล์ระดับ 40 GB)
    # -H รักษา hardlink เผื่อ cache ของ HF ใช้ · --no-owner/--no-group เพราะ uid
    # สองเครื่องไม่จำเป็นต้องตรงกัน
    flags = "-aH --partial --append-verify --no-owner --no-group --info=progress2"
    if dry_run:
        flags += " --dry-run"
    return (
        f'set -e; '
        f'command -v rsync >/dev/null || {{ echo "ต้นทางไม่มี rsync — ติดตั้งก่อน: sudo apt install -y rsync" >&2; exit 1; }}; '
        # กุญแจมาทาง stdin แล้วเข้า ssh-agent ตรง ๆ — ไม่มีจังหวะไหนที่มันแตะดิสก์ของต้นทาง
        # ใช้ `cat` อ่านทั้งก้อน ไม่ใช่ `read` เพราะ private key เป็นข้อความหลายบรรทัด
        f'KEY="$(cat)"; '
        f'eval "$(ssh-agent -s)" >/dev/null; '
        f'trap \'ssh-agent -k >/dev/null 2>&1 || true\' EXIT; '
        f'printf "%s\\n" "$KEY" | ssh-add - >/dev/null 2>&1 || '
        f'{{ echo "ใส่กุญแจชั่วคราวไม่สำเร็จ" >&2; exit 1; }}; '
        f'unset KEY; '
        f'{ssh_opts} {shlex.quote(dest)} "mkdir -p {md} {bd}"; '
        f'rsync {flags} -e {shlex.quote(ssh_opts)} {md}/ {shlex.quote(dest + ":" + plan.model_dir + "/")}; '
        # cluster.env ผูกกับคู่เครื่องต้นทาง (IP/interface ของ head-worker คู่นั้น) — ลากไปด้วยแล้ว
        # controller บนปลายทางจะ ssh ไปหา worker ของคู่เก่าเงียบ ๆ · ไม่มีไฟล์ = controller ใหม่
        # ปฏิเสธพร้อมบอกให้ hub เขียนให้ (`lmds node cluster --write`) ซึ่งถูกกับคู่ใหม่เสมอ
        f'rsync -a --no-owner --no-group --exclude=cluster.env -e {shlex.quote(ssh_opts)} {bd}/ '
        f'{shlex.quote(dest + ":" + plan.bundle_dir + "/")}'
    )


def make_marker() -> str:
    """มาร์กเกอร์ที่สุ่มต่อครั้ง — ใช้ชี้บรรทัดที่ต้องถอนออกทีหลังแบบไม่กำกวม"""
    return f"lmds-clone-{secrets.token_hex(8)}"
