"""ผูก bundle แบบ stacked เข้ากับกลุ่มจริง — เลือกสมาชิกให้ตรงจำนวนที่ bundle ถูก render มา

`build_cluster_env` (fleet) เขียน worker *ทุกตัว* ในกลุ่มที่ส่งให้ หรือถ้าระบุชื่อ worker ก็เขียน
แค่ตัวเดียว · ทั้งสองทางไม่เคยดูเลยว่า bundle ถูก render มาสำหรับกี่เครื่อง (NNODES ในสคริปต์)
→ กลุ่ม 4 เครื่องกับ bundle 2 เครื่องได้ cluster.env ที่บอก NNODES=4/TP=4 ทับค่าของแผน
(cluster.env ถูก source ก่อน default ของสคริปต์จึงชนะ) และ bundle 4 เครื่องที่เลือก worker
ตัวเดียวจากหน้าเว็บได้ TP=2 ซึ่งโมเดลไม่มีทาง fit — ทั้งคู่ตายที่ NCCL/OOM โดยไม่มีใครบอกว่า
"จำนวนเครื่องไม่ตรงกับที่วางแผน"

ที่นี่ตัดกลุ่มให้เหลือ head + worker ตามจำนวนที่ bundle ต้องการ แล้วส่งสำเนานั้นให้
build_cluster_env เขียนทุก worker ที่เหลือ — ตรรกะของไฟล์ยังอยู่ที่เดิม (fleet/cluster_env)
"""

from __future__ import annotations

import re
from pathlib import Path

# บรรทัดที่ renderer เขียนไว้ในสคริปต์ stacked: NNODES="${NNODES:-2}"
_NNODES_LINE = re.compile(r'^NNODES="\$\{NNODES:-(\d+)\}"', re.M)


class StackedError(Exception):
    """ปัญหาที่ผู้ใช้แก้ได้ — ข้อความพร้อมแสดงทั้ง CLI และหน้าเว็บ (ภาษาอังกฤษ: หน้าเว็บโชว์ตรง ๆ)"""


def bundle_node_count(bundle_dir: str | Path) -> int:
    """bundle นี้ถูก render มาสำหรับกี่เครื่อง — 1 เมื่อไม่ใช่ stacked หรืออ่านไม่ได้

    MODEL_PROFILE.yaml มีแค่ topology ไม่มีจำนวนเครื่อง · ตัวเลขจริงอยู่ในสคริปต์ controller
    ซึ่งคือสิ่งที่จะรัน จึงอ่านจากตรงนั้น
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        return 1
    for controller in sorted(root.glob("*-stacked.sh")):
        try:
            match = _NNODES_LINE.search(controller.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if match:
            return max(1, int(match.group(1)))
    return 1


def _ready_group_with(groups, head_name: str) -> dict:
    ready = [g for g in groups if g.get("ready")
             and any(m["name"] == head_name for m in g.get("members") or [])]
    if ready:
        return ready[0]
    # บอกให้ตรงว่าติดตรงไหน — "ไม่พร้อม" เฉย ๆ คือสิ่งที่ลูกค้าบ่น
    holding = [g for g in groups if any(m["name"] == head_name for m in g.get("members") or [])]
    if holding:
        blockers = ", ".join(f"{b['kind']}: {', '.join(b.get('names') or [])}"
                             for b in holding[0].get("blockers") or []) or "not ready"
        raise StackedError(
            f"the cluster group of '{head_name}' is not ready — {blockers} · "
            f"run the pair doctor (lmds cluster doctor) for the full reason"
        )
    names = sorted({m["name"] for g in groups if g.get("ready") for m in g["members"]})
    raise StackedError(
        f"'{head_name}' is not a member of any ready cluster group"
        + (f" — heads available: {', '.join(names)}" if names
           else " — set the cluster IP on every machine of the pair first")
    )


def select_members(groups, head_name: str, workers: list[str] | tuple[str, ...] = (),
                   nnodes: int | None = None) -> dict:
    """สำเนากลุ่มที่มีสมาชิกแค่ head + worker ที่จะใช้จริง เรียงตาม node-rank

    `workers` ว่าง = เอา worker ตามลำดับในกลุ่มจนครบ `nnodes - 1` · ระบุมา = ต้องครบพอดี
    `nnodes` None = ทั้งกลุ่ม (พฤติกรรมเดิม)
    """
    group = _ready_group_with(groups, head_name)
    members = list(group.get("members") or [])
    head = next(m for m in members if m["name"] == head_name)
    others = [m for m in members if m["name"] != head_name]
    by_name = {m["name"]: m for m in others}

    chosen: list[dict] = []
    for name in [w for w in workers if w]:
        if name == head_name:
            raise StackedError(f"'{name}' cannot be both head and worker")
        if name not in by_name:
            raise StackedError(
                f"'{name}' is not in the ready cluster group of '{head_name}' "
                f"(members: {', '.join(m['name'] for m in members)})"
            )
        if by_name[name] not in chosen:
            chosen.append(by_name[name])

    want = None if nnodes is None else max(0, int(nnodes) - 1)
    if not chosen:
        chosen = others if want is None else others[:want]
    if want is not None and len(chosen) != want:
        have = len(chosen) if workers else len(others)
        raise StackedError(
            f"this bundle was built for {nnodes} machines (head + {want} worker"
            f"{'s' if want != 1 else ''}) but {have} worker{'s' if have != 1 else ''} "
            f"{'were' if have != 1 else 'was'} {'chosen' if workers else 'available'} in the group of "
            f"'{head_name}' — re-run analyse with the matching target "
            f"(dgx-spark-stacked = 2, dgx-spark-stacked-4 = 4) or pick the right workers"
        )
    if not chosen:
        raise StackedError(f"the group of '{head_name}' has no worker")
    return {**group, "members": [head, *chosen]}
