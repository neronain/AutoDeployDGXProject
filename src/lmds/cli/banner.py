"""Banner สุ่มสไตล์ metasploit — โชว์เฉพาะ terminal จริง (stderr tty) ไม่รบกวน script/JSON

ปิดได้ด้วย env: LMDS_NO_BANNER=1
"""

from __future__ import annotations

import os
import random
import sys

from rich.console import Console

CREDIT = "Local Model Deploy Studio — สร้างโดย neronain ⚡ fb.com/neronain.minidev"

_LMDS_BLOCK = r"""
██╗      ███╗   ███╗ ██████╗  ███████╗
██║      ████╗ ████║ ██╔══██╗ ██╔════╝
██║      ██╔████╔██║ ██║  ██║ ███████╗
██║      ██║╚██╔╝██║ ██║  ██║ ╚════██║
███████╗ ██║ ╚═╝ ██║ ██████╔╝ ███████║
╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚══════╝
      Deploy models. Anywhere. Verified.
"""

_SPARK = r"""
        /\
       /  \      ____ ____ _  _    ____ ___  ____ ____ _  _
      / /\ \     |  \ | __  \/     [__  |__] |__| |__/ |_/
     / ____ \    |__/ |__] _/\_    ___] |    |  | |  \ | \_
    /_/    \_\
   ⚡ GB10 · SM121 · unified 128GB — one link, one bundle, run.
"""

_GPU_RACK = r"""
   ┌─────────────────────────────────┐
   │ ▓▓▓▓▓▓▓▓▓▓▓▓  GPU 0  ██████ 24G │
   │ ▓▓▓▓▓▓▓▓▓▓▓▓  GPU 1  ██████ 24G │
   │ ─────────── L M D S ─────────── │
   │  HF link ─▶ plan ─▶ gates ─▶ ✓  │
   └─────────────────────────────────┘
"""

_TERMINAL = r"""
   ╭─ lmds ──────────────────────────────╮
   │ $ lmds deploy <model-url>           │
   │   ├─ inspect ......... pinned ✓     │
   │   ├─ fit ............. 65,536 tok   │
   │   ├─ brain ........... plan ✓       │
   │   ├─ gates ........... 8/8 ✓        │
   │   ╰─ bundle.zip ...... delivered ⚡  │
   ╰─────────────────────────────────────╯
"""

BANNERS: list[str] = [_LMDS_BLOCK, _SPARK, _GPU_RACK, _TERMINAL]
_COLORS = ["bright_cyan", "bright_magenta", "bright_green", "bright_yellow", "bright_blue"]

_shown = False


def show_banner(console: Console | None = None) -> None:
    """พิมพ์ banner หนึ่งครั้งต่อการรัน — เงียบเมื่อไม่ใช่ tty หรือถูกปิดด้วย env"""
    global _shown
    if _shown or os.environ.get("LMDS_NO_BANNER"):
        return
    if not sys.stderr.isatty():
        return
    _shown = True
    console = console or Console(stderr=True)
    art = random.choice(BANNERS)
    color = random.choice(_COLORS)
    console.print(f"[bold {color}]{art}[/bold {color}]", highlight=False)
    console.print(f"[dim]{CREDIT}[/dim]\n")
