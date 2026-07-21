"""Banner สุ่มสไตล์ metasploit — มีทั้งภาพนิ่งและแบบเคลื่อนไหว (animation สั้น ~1s)

- โชว์เฉพาะ terminal จริง (stderr tty) ไม่รบกวน script/JSON
- ปิดทั้งหมด: LMDS_NO_BANNER=1 | ปิดเฉพาะ animation: LMDS_BANNER_STATIC=1
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.text import Text

CREDIT = "Local Model Deploy Studio — สร้างโดย neronain ⚡ fb.com/neronain.minidev"


@dataclass
class Banner:
    frames: list[str]  # 1 frame = ภาพนิ่ง, หลาย frame = animation (จบที่ frame สุดท้าย)
    interval: float = 0.12
    colors: list[str] = field(default_factory=lambda: [
        "bright_cyan", "bright_magenta", "bright_green", "bright_yellow", "bright_blue"
    ])


# ---------- ภาพนิ่ง ----------

_LMDS_BLOCK = Banner([r"""
██╗      ███╗   ███╗ ██████╗  ███████╗
██║      ████╗ ████║ ██╔══██╗ ██╔════╝
██║      ██╔████╔██║ ██║  ██║ ███████╗
██║      ██║╚██╔╝██║ ██║  ██║ ╚════██║
███████╗ ██║ ╚═╝ ██║ ██████╔╝ ███████║
╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚══════╝
      Deploy models. Anywhere. Verified.
"""])

_SPARK = Banner([r"""
        /\
       /  \      ____ ____ _  _    ____ ___  ____ ____ _  _
      / /\ \     |  \ | __  \/     [__  |__] |__| |__/ |_/
     / ____ \    |__/ |__] _/\_    ___] |    |  | |  \ | \_
    /_/    \_\
   ⚡ GB10 · SM121 · unified 128GB — one link, one bundle, run.
"""])

_GPU_RACK = Banner([r"""
   ┌─────────────────────────────────┐
   │ ▓▓▓▓▓▓▓▓▓▓▓▓  GPU 0  ██████ 24G │
   │ ▓▓▓▓▓▓▓▓▓▓▓▓  GPU 1  ██████ 24G │
   │ ─────────── L M D S ─────────── │
   │  HF link ─▶ plan ─▶ gates ─▶ ✓  │
   └─────────────────────────────────┘
"""])

_TERMINAL = Banner([r"""
   ╭─ lmds ──────────────────────────────╮
   │ $ lmds deploy <model-url>           │
   │   ├─ inspect ......... pinned ✓     │
   │   ├─ fit ............. 65,536 tok   │
   │   ├─ brain ........... plan ✓       │
   │   ├─ gates ........... 8/8 ✓        │
   │   ╰─ bundle.zip ...... delivered ⚡  │
   ╰─────────────────────────────────────╯
"""])

_AI_ROBOT = Banner([r"""
        ┌─────────────┐
        │   ◉     ◉   │      ██     ███    ███ ██████   ██████
        │      ▽      │      ██     ████  ████ ██   ██ ██
        │   ╰─────╯   │      ██     ██ ████ ██ ██   ██  █████
        └──┬───────┬──┘      ██     ██  ██  ██ ██   ██      ██
       ════╡ AI-OPS ╞════    ██████ ██      ██ ██████  ██████
           └───────┘         your models · your metal · verified
"""])

_NEURAL_NET = Banner([r"""
     ○───────○───────○
      ╲     ╱ ╲     ╱        L · M · D · S
       ○───○───○───○         ─────────────
      ╱     ╲ ╱     ╲        neural deploy engine
     ○───────○───────○       HF ─▶ fit ─▶ plan ─▶ serve
"""])

_CHIP_BRAIN = Banner([r"""
        ╔═══╡ L M D S ╞═══╗
     ───╢  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ╟───
     ───╢  │∿│ │∿│ │∿│ │∿│ ╟───     silicon in.
     ───╢  └─┘ └─┘ └─┘ └─┘ ╟───     intelligence out.
        ╚══════╡ ⚡ ╞══════╝
"""])

# ---------- แบบเคลื่อนไหว ----------

def _pulse_frame(active: int) -> str:
    """neural network ที่สัญญาณวิ่งจากซ้ายไปขวา (คอลัมน์ที่ active สว่างเป็น ●)"""
    cols = [("●" if i == active else "○") for i in range(4)]
    return f"""
     {cols[0]}───────{cols[1]}───────{cols[2]}───────{cols[3]}
      ╲     ╱ ╲     ╱ ╲     ╱      L M D S
       {cols[0]}───{cols[1]}───{cols[2]}───{cols[3]}───{cols[0] if active == 3 else '○'}       signal flowing…
      ╱     ╲ ╱     ╲ ╱     ╲
     {cols[0]}───────{cols[1]}───────{cols[2]}───────{cols[3]}
"""


_NEURAL_PULSE = Banner(
    frames=[_pulse_frame(i) for i in (0, 1, 2, 3, 2, 1, 0, 3)],
    interval=0.10,
)


def _loading_frame(step: int, total: int = 8) -> str:
    bar = "▰" * step + "▱" * (total - step)
    label = "loading models…" if step < total else "ready ⚡ deploy!"
    return f"""
   ██     ███    ███ ██████   ██████
   ██     ████  ████ ██   ██ ██
   ██     ██ ████ ██ ██   ██  █████
   ██     ██  ██  ██ ██   ██      ██
   ██████ ██      ██ ██████  ██████
        {bar}  {label}
"""


_LOADING_BAR = Banner(
    frames=[_loading_frame(i) for i in range(9)],
    interval=0.09,
)

BANNERS: list[Banner] = [
    _LMDS_BLOCK,
    _SPARK,
    _GPU_RACK,
    _TERMINAL,
    _AI_ROBOT,
    _NEURAL_NET,
    _CHIP_BRAIN,
    _NEURAL_PULSE,
    _LOADING_BAR,
]

_shown = False


def _render(console: Console, banner: Banner, color: str) -> None:
    if len(banner.frames) == 1 or os.environ.get("LMDS_BANNER_STATIC"):
        console.print(Text(banner.frames[-1], style=f"bold {color}"))
        return
    with Live(console=console, refresh_per_second=30, transient=False) as live:
        for frame in banner.frames:
            live.update(Text(frame, style=f"bold {color}"))
            time.sleep(banner.interval)


def show_banner(console: Console | None = None) -> None:
    """พิมพ์ banner หนึ่งครั้งต่อการรัน — เงียบเมื่อไม่ใช่ tty หรือถูกปิดด้วย env"""
    global _shown
    if _shown or os.environ.get("LMDS_NO_BANNER"):
        return
    if not sys.stderr.isatty():
        return
    _shown = True
    console = console or Console(stderr=True)
    banner = random.choice(BANNERS)
    _render(console, banner, random.choice(banner.colors))
    console.print(f"[dim]{CREDIT}[/dim]\n")
