"""ค่าที่ผู้ใช้ตั้งไว้กับ bundle หนึ่ง ๆ — เก็บข้าง controller ไม่ใช่ในเบราว์เซอร์

หน้าเว็บมีช่อง port/context/slots มาตลอด แต่ค่าที่กรอกถูกส่งเป็น env เฉพาะตอน
กดปุ่มนั้นครั้งเดียวแล้วหายไป ผลคือ:

  * `enable autostart` สร้าง systemd unit ที่เรียก controller เปล่า ๆ พอเครื่อง
    reboot ทุกโมเดลบนเครื่องเดียวกันจึงขึ้นที่ port เดียวกันแล้วชนกันหมด
  * ปุ่ม test-text / test-vision / client-config ก็เรียก controller เปล่า ๆ
    เหมือนกัน คำสั่งจึงวิ่งไปหา port เริ่มต้น ไม่ใช่ port ที่โมเดลนั้นรันอยู่จริง

ไฟล์นี้แก้ที่ต้นเหตุ: controller source `bundle.env` ก่อนตั้ง default ทุกตัว
ทางที่เรียก controller — systemd, ปุ่มบนเว็บ, คนพิมพ์เอง — จึงได้ค่าเดียวกันหมด
โดยไม่ต้องมีใครจำว่าต้องส่ง env อะไรไปด้วย

**ไม่เก็บ API key ไว้ที่นี่** — ไฟล์นี้อยู่ในโฟลเดอร์ที่ถูก zip แจกต่อได้
key อยู่ที่ `fleet/apikey.py` (~/.lmds/keys/<slug> โหมด 0600) ซึ่งผูกกับ *เครื่อง*
ไม่ใช่กับ bundle · เดิมไม่ได้เก็บไว้ที่ไหนเลย ผลคือ systemd ตอน autostart ไม่เคยได้ key
แล้วโมเดลกลับมาเปิดโล่งบน 0.0.0.0 ทุกครั้งที่ reboot — เป็นความพังแบบเดียวกับที่ไฟล์นี้
แก้ให้ port/context ไปแล้ว ต่างกันแค่ว่าอันนั้นทำให้ใช้งานไม่ได้ ส่วนอันนี้เปิดช่องให้คนอื่น
"""

from __future__ import annotations

import re
from pathlib import Path

FILENAME = "bundle.env"

# knob ที่ยอมให้บันทึกได้ · ชื่อทางซ้ายคือสิ่งที่หน้าเว็บส่งมา ทางขวาคือ env ที่
# controller อ่าน (บาง knob มีสองชื่อเพราะ llama.cpp กับ vLLM เรียกไม่เหมือนกัน)
FIELDS: dict[str, tuple[str, ...]] = {
    "port": ("API_PORT",),
    "bind": ("API_HOST",),
    "context": ("CTX_SIZE", "MAX_MODEL_LEN"),
    "slots": ("PARALLEL_SEQS", "MAX_NUM_SEQS"),
    "gpu_util": ("GPU_MEMORY_UTILIZATION",),
    "served_name": ("SERVED_MODEL_NAME",),
    "image": ("VLLM_IMAGE", "LLAMACPP_IMAGE"),
    # env ของ engine เอง — knob ที่ vLLM/SGLang อ่านจาก environment ล้วน ๆ
    # ไม่มีทางส่งเข้าไปได้เลยถ้าไม่มีช่องนี้ (ดู _clean)
    "engine_env": ("ENGINE_ENV",),
    # parser ของ vLLM/SGLang — เดิมตั้งได้แค่ตอน start (--tool-parser) จึงหายตอน autostart
    # เคสจริง 2026-09-03: bundle ของ Sehyo/Qwen3.5-122B ที่ plan แบบ rule-based ไม่เปิด tool
    # ไว้ ต้องใส่ qwen3_xml + qwen3 ทุกครั้งที่ start ไม่งั้น agent เห็น tool call เป็นข้อความ
    "tool_parser": ("TOOL_CALL_PARSER",),
    "reasoning_parser": ("REASONING_PARSER",),
    # --image-min-tokens ของ llama.cpp (vision) — ตัวเลข หรือ "auto" = ใช้ค่าที่ฝังมากับ projector
    # "auto" ต้องเขียนลงไฟล์เป็นค่าว่าง (set แต่ว่าง) ไม่ใช่ลบทิ้ง: controller ที่สร้างก่อน
    # 0.5.2 มี default 1024 ฝังอยู่ ($\{VAR-1024\}) — ลบทิ้ง = กลับไปพังกับ Gemma-4
    # (เคสจริง 2026-09-04 dgx-veerasiam: clip_init ปฏิเสธเพราะ 1024 > เพดาน 280 ของ Gemma-4)
    "image_min_tokens": ("IMAGE_MIN_TOKENS",),
    # แฟล็กเพิ่มของ engine เช่น --speculative-config '{"method":"mtp",...}' · เก็บใน
    # ไฟล์แยก (bundle.args) ไม่ใช่ bundle.env เพราะรูป ${VAR:-value} ของ bash หยุดที่
    # `}` ตัวแรกที่เจอ — JSON จึงถูกตัดกลางคัน (ทดสอบแล้ว 2026-09-03)
    "extra_args": (),
}
ARGS_FILENAME = "bundle.args"


class SettingsError(ValueError):
    """ค่าที่ส่งมาใช้ไม่ได้ — บอกไปตรง ๆ ดีกว่าเขียนลงไฟล์แล้วให้ start พังทีหลัง"""


def _clean(name: str, value: object) -> str:
    text = str(value).strip()
    if name == "port":
        if not text.isdigit() or not (1 <= int(text) <= 65535):
            raise SettingsError(f"port ต้องเป็นเลข 1-65535 (ได้ {text!r})")
        return text
    if name in {"context", "slots"}:
        if not text.isdigit() or int(text) < 1:
            raise SettingsError(f"{name} ต้องเป็นจำนวนเต็มบวก (ได้ {text!r})")
        return text
    if name == "gpu_util":
        try:
            number = float(text)
        except ValueError as exc:
            raise SettingsError(f"gpu_util ต้องเป็นตัวเลข (ได้ {text!r})") from exc
        if not 0 < number <= 1:
            raise SettingsError(f"gpu_util ต้องอยู่ระหว่าง 0 ถึง 1 (ได้ {text!r})")
        return text
    if name == "bind":
        if not re.fullmatch(r"[0-9a-zA-Z_.:\[\]-]+", text):
            raise SettingsError(f"bind ไม่ใช่ที่อยู่ที่ใช้ได้ (ได้ {text!r})")
        return text
    if name == "image_min_tokens":
        if text.lower() in {"auto", "file", "none"}:
            return ""  # set แต่ว่าง — controller จะไม่ส่ง --image-min-tokens
        if not text.isdigit() or int(text) < 1:
            raise SettingsError(f"image_min_tokens ต้องเป็นจำนวนเต็มบวก หรือ auto (ได้ {text!r})")
        return text
    if name in {"tool_parser", "reasoning_parser"}:
        # ชื่อ parser เป็น identifier ล้วน — ค่าว่างคือปิด ซึ่ง write() ตัดออกก่อนถึงตรงนี้แล้ว
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
            raise SettingsError(f"{name} ต้องเป็นชื่อ parser เช่น qwen3_xml / qwen3 (ได้ {text!r})")
        return text
    if name == "extra_args":
        # ข้อความอิสระที่ controller จะแตกเป็น argv ด้วยช่องว่าง — JSON ต้องเขียนแบบไม่มีช่องว่าง
        # กันเฉพาะสิ่งที่ทำให้เชลล์รันของอื่นได้ ส่วน quote/วงเล็บปีกกาต้องผ่านเพราะ JSON ใช้
        # flag ที่ controller เป็นเจ้าของ (TP/nnodes/node-rank/master-*) — ใส่ซ้ำแล้ว vLLM ให้ตัวหลังชนะ = TP=1 บน 2 เครื่อง
        # (harden กันฝั่ง LLM ตั้งแต่ 0.6.0 แต่ `lmds set --extra-args` หลุด — audit รอบ 2)
        owned = ("--tensor-parallel-size", "-tp", "--nnodes", "--node-rank", "--master-addr", "--master-port",
                 "--headless", "--data-parallel-size", "--pipeline-parallel-size")
        for tok in text.split():
            key = tok.split("=", 1)[0]
            if key in owned:
                raise SettingsError(f"extra_args มี {key} ซึ่ง controller ตั้งให้เองตาม topology — ใส่ซ้ำไม่ได้ (ใช้ --target/plan แทน)")
        if any(ch in text for ch in "\n\r\x00`$"):
            raise SettingsError("extra_args มีอักขระที่เชลล์ตีความ (` $ หรือขึ้นบรรทัดใหม่) — ใส่ไม่ได้")
        return " ".join(text.split())
    if name == "engine_env":
        # รายการ KEY=VALUE คั่นด้วยช่องว่าง — controller แตกออกเป็น `-e KEY=VALUE` ต่อ docker
        #
        # เคสจริง 2026-08-20: NVFP4 บน GB10 ต้องได้ VLLM_NVFP4_GEMM_BACKEND=marlin ไม่งั้น
        # vLLM ไป JIT cutlass FP4 kernel แล้ว ptxas ปฏิเสธ (`cvt .e2m1x2` ไม่มีบน sm_121)
        # engine ตายก่อน health · knob นี้อ่านจาก environment ล้วน ๆ ส่งผ่าน flag ไม่ได้
        # ก่อนหน้านี้จึงไม่มีทางตั้งเลยนอกจากแก้สคริปต์ด้วยมือ ซึ่งหายไปทุกครั้งที่ rebuild
        cleaned = []
        for pair in text.split():
            key, sep, value = pair.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise SettingsError(
                    f"engine_env ต้องเป็น KEY=VALUE คั่นด้วยช่องว่าง (ได้ {pair!r})")
            if any(ch in value for ch in " \t\n\r\x00'\"$`\\{}"):
                raise SettingsError(f"ค่าของ {key} มีอักขระที่เชลล์ตีความ — ใส่ไม่ได้")
            cleaned.append(f"{key}={value}")
        return " ".join(cleaned)

    # ชื่อโมเดล/image เป็นข้อความอิสระ — แต่ไฟล์นี้ถูก `source` เป็น bash ทุกครั้งที่ start/autostart
    # เดิมกันแค่ขึ้นบรรทัดใหม่ ส่วน $(…) ` " ผ่านได้ → served_name="x$(id)y" จากช่องกรอกบนหน้าเว็บ
    # = รันคำสั่งบนเครื่องนั้นในฐานะผู้ใช้ (รีวิว 2026-09-04) · shlex.quote ตอนเขียนไม่ช่วย
    # เพราะค่าถูกวางใน "${VAR:-…}" ที่อยู่ใน double quote อยู่แล้ว
    if any(ch in text for ch in "\n\r\x00\"'`$\\{}"):
        raise SettingsError(
            f"{name} มีอักขระที่เชลล์ตีความ (\" ' ` $ \\ {{ }}) — ใส่ไม่ได้")
    return text


def path_for(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / FILENAME


# บรรทัดกำหนดค่าหนึ่งบรรทัดของ bundle.env — `NAME=…` หรือ `export NAME=…` (ย่อหน้าได้)
_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# env ทุกตัวที่ `lmds set` เป็นเจ้าของ → field ของมัน · บรรทัดของชื่ออื่นทั้งหมดเป็นของผู้ดูแล ห้ามแตะ
_FIELD_OF: dict[str, str] = {name: field for field, names in FIELDS.items() for name in names}


def _assigned_name(line: str) -> str:
    """ชื่อ env ที่บรรทัดนี้กำหนดค่า — "" ถ้าไม่ใช่บรรทัดกำหนดค่า (คอมเมนต์ · บรรทัดว่าง · คำสั่งอื่น)"""
    match = _ASSIGN_RE.match(line)
    return match.group(1) if match else ""


def _assigned_value(name: str, raw: str) -> tuple[str, bool] | None:
    """(ค่า, เป็นรูป default ไหม) ของด้านขวาเครื่องหมาย = · None = อ่านเป็นค่าคงที่ไม่ได้ (มี $… / คำสั่ง / quote ไม่ปิด)

    รูปที่ LMDS เขียนเองคือ `"${NAME:-value}"` (default — env จากภายนอกชนะ) · ผู้ดูแลที่แก้ไฟล์ด้วยมือเขียน `NAME=value` /
    `NAME="value"` ธรรมดา ซึ่ง bash ให้ผลเท่ากันตอน start · ค่าที่ต้องให้เชลล์คิด (`$(…)`, `$OTHER`) ไม่เดา — ถือว่าอ่านไม่ได้
    """
    raw = raw.strip()
    for pattern in (r'"\$\{%s:-(.*)\}"', r"\$\{%s:-(.*)\}"):
        match = re.fullmatch(pattern % re.escape(name), raw)
        if match:
            return match.group(1), True
    match = (re.fullmatch(r'"([^"]*)"(?:\s+#.*)?', raw) or re.fullmatch(r"'([^']*)'(?:\s+#.*)?", raw)
             or re.fullmatch(r"([^\s#\"']*)(?:\s+#.*)?", raw))
    if match is None or any(ch in match.group(1) for ch in "$`\\"):
        return None
    return match.group(1), False


def read(bundle_dir: Path) -> dict[str, str]:
    """ค่าที่บันทึกไว้ — คืน dict ว่างเมื่อยังไม่เคยบันทึก

    อ่านทั้งรูปที่ `lmds set` เขียน (`NAME="${NAME:-v}"`) และรูปที่คนเขียนเอง (`NAME=v`): หัวไฟล์บอกว่าแก้ด้วยมือได้ และค่าที่
    เขียนด้วยมือมีผลจริงตอน start — เดิมมองไม่เห็นรูปหลัง จึงโชว์ว่า "ไม่ได้ตั้ง" ทั้งที่ controller ได้ค่านั้น (audit 2026-10-06) ·
    หลายบรรทัดของชื่อเดียวกันตัดสินแบบ bash: รูป default ไม่ทับค่าที่ตั้งไปแล้ว · รูปธรรมดาทับเสมอ · ค่าที่ไม่ผ่าน `_clean`
    (เช่น port ที่ไม่ใช่ตัวเลข) ไม่ถูกรายงาน — ผู้เรียกหลายตัวเอาค่าไป `int()` ตรง ๆ
    """
    target = path_for(bundle_dir)
    if not target.is_file():
        args_file = Path(bundle_dir) / ARGS_FILENAME
        if args_file.is_file() and args_file.read_text(encoding="utf-8").strip():
            return {"extra_args": args_file.read_text(encoding="utf-8").strip()}
        return {}
    env: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        name = _assigned_name(line)
        if name not in _FIELD_OF:
            continue
        parsed = _assigned_value(name, _ASSIGN_RE.match(line).group(2))
        if parsed is None:
            continue
        value, is_default = parsed
        if is_default and env.get(name):
            continue        # `${NAME:-v}` หลังจากที่ NAME มีค่าแล้ว = ไม่มีผล
        env[name] = value
    out: dict[str, str] = {}
    for field, names in FIELDS.items():
        for name in names:
            if name in env:
                value = env[name]
                if field == "image_min_tokens" and value == "":
                    value = "auto"  # ค่าว่างในไฟล์ = auto — ให้ round-trip ผ่าน write() ได้โดยไม่หาย
                else:
                    try:
                        if value == "" or _clean(field, value) != value:
                            continue
                    except SettingsError:
                        continue
                out[field] = value
                break
    args_file = Path(bundle_dir) / ARGS_FILENAME
    if args_file.is_file():
        extra = args_file.read_text(encoding="utf-8").strip()
        if extra:
            out["extra_args"] = extra
    return out


def native_context(bundle_dir: Path) -> int:
    """เพดาน context ของโมเดล — MODEL_PROFILE.yaml ก่อน · ไม่มี (adopt/profile เก่า) ถอยไปอ่าน NATIVE_CONTEXT="N" ที่หัว controller
    (0 = ไม่รู้) · audit 2026-09-08: `lmds set --context 1048676` ผ่านเงียบ ๆ บน bundle ที่ profile ไม่มีค่านี้ แล้วทุก test เตือน RoPE"""
    import yaml

    bundle_dir = Path(bundle_dir)
    try:
        profile = yaml.safe_load((bundle_dir / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")) or {}
        value = ((profile.get("model") or {}).get("native_context"))
        if isinstance(value, int) and value > 0:
            return int(value)
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        pass
    for controller in sorted(bundle_dir.glob("*-single.sh")) + sorted(bundle_dir.glob("*-stacked.sh")):
        try:
            found = re.search(r'^NATIVE_CONTEXT="(\d+)"', controller.read_text(encoding="utf-8"), re.M)
        except OSError:
            continue
        if found and int(found.group(1)) > 0:
            return int(found.group(1))
    return 0


def _engine_of(bundle_dir: Path) -> str:
    import yaml

    try:
        profile = yaml.safe_load((Path(bundle_dir) / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")) or {}
        return str((profile.get("runtime") or {}).get("engine") or "")
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return ""


def _slots_for(bundle_dir: Path, cleaned: dict[str, str]) -> int:
    """จำนวน slot ที่ค่า context ชุดนี้จะถูกหารด้วย: ที่ส่งมาพร้อมกัน > ที่บันทึกไว้ > ที่แผนตั้ง > 1"""
    import yaml

    candidates: list[object] = [cleaned.get("slots"), read(bundle_dir).get("slots")]
    try:
        profile = yaml.safe_load((Path(bundle_dir) / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")) or {}
        candidates.append((profile.get("serving") or {}).get("max_num_seqs"))
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        pass
    for value in candidates:
        try:
            if value not in (None, "") and int(value) >= 1:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 1


def _check_context_cap(bundle_dir: Path, values: dict[str, object], cleaned: dict[str, str]) -> None:
    """context ที่ตั้งต้องไม่เกิน max_position_embeddings ของโมเดล — ไม่งั้น vLLM ปฏิเสธตอน start
    (เคสจริง 2026-09-05 msi-4/msi-5: Llama-3.3-70B ตั้ง 262144 > 131072 → worker ตายก่อน head จะเริ่ม)
    ยกเว้นเมื่อผู้ใช้ตั้งใจ: engine_env มี VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (ที่บันทึกไว้แล้วหรือส่งมาพร้อมกัน)"""
    if "context" not in cleaned:
        return
    cap = native_context(bundle_dir)
    if not cap or int(cleaned["context"]) <= cap:
        return
    if _engine_of(bundle_dir) == "llamacpp":
        # llama.cpp: context ที่บันทึกคือ --ctx-size = ก้อนรวมของทุก slot · เพดานของโมเดลเป็นเพดาน
        # ต่อคำขอ จึงต้องเทียบ context ÷ slots · 4 slot × 131,072 = 524,288 คือค่าที่ถูกต้องและ
        # planner ตั้งให้เองเมื่อ deploy ด้วย --concurrency 4 — เดิม `lmds set --context` ปฏิเสธ
        # ค่านี้ว่า "เกินเพดานที่โมเดลเทรนมา" (ตรวจ 2026-10-05)
        slots = _slots_for(bundle_dir, cleaned)
        if int(cleaned["context"]) // slots <= cap:
            return
    env = " ".join([str(values.get("engine_env") or ""), read(bundle_dir).get("engine_env", "")])
    if "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" in env.split():
        return
    if _engine_of(bundle_dir) == "llamacpp":
        # llama.cpp ไม่ปฏิเสธ แต่ทุกคำขอจะเตือน RoPE และคุณภาพหลังตำแหน่งที่เทรนมาไม่รับประกัน — ค่าที่เกินมักเป็น
        # เลขพิมพ์พลาด (1048676 vs 1048576 · audit 2026-09-08) จึงปฏิเสธพร้อมบอกเพดาน ไม่ปัดเงียบ ๆ
        per_request = int(cleaned["context"]) // slots
        split = f" ÷ {slots} slot = {per_request:,} ต่อคำขอ" if slots > 1 else ""
        raise SettingsError(
            f"context {int(cleaned['context']):,}{split} เกินเพดานที่โมเดลเทรนมา ({cap:,} tokens) — llama.cpp จะ start ได้แต่ทุกคำขอ"
            f"เตือน RoPE และคุณภาพเกินตำแหน่งนั้นไม่รับประกัน · ตั้งได้สูงสุด {cap * slots:,}"
            f"{f' ({cap:,} × {slots} slot)' if slots > 1 else ''} (ถ้าตั้งใจใช้ RoPE scaling ใส่ "
            f"--extra-args \"--rope-scaling yarn …\" แล้วตั้ง context ผ่าน env CTX_SIZE ตอน start แทน)")
    raise SettingsError(
        f"context {int(cleaned['context']):,} เกินเพดานของโมเดลนี้ ({cap:,} tokens = max_position_embeddings) — "
        f"vLLM จะไม่ยอม start · ตั้งได้สูงสุด {cap:,} · ถ้าต้องการเกินจริง ๆ ใส่ engine env "
        f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 ใน Advanced ก่อน (ตำแหน่งที่เกินอาจให้ผลเป็น nan)")


def _check_image_applies(bundle_dir: Path) -> None:
    """`--image` กับ bundle llama.cpp แบบ native build = no-op เงียบ (audit 2026-09-06 §6.1)

    controller native ไม่อ่าน LLAMACPP_IMAGE เลย — เขียนลงไฟล์สำเร็จ ผู้ใช้เข้าใจว่าเปลี่ยนรันไทม์แล้ว ทั้งที่ build
    กลางของเครื่องยังเป็นตัวเดิม · บอกทางที่ใช้ได้จริง (update-runtime) แทนที่จะรับค่าไว้เฉย ๆ
    """
    import yaml

    try:
        profile = yaml.safe_load((Path(bundle_dir) / "MODEL_PROFILE.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return
    if not isinstance(profile, dict) or (profile.get("runtime") or {}).get("engine") != "llamacpp":
        return
    from lmds.doctor.checks import llamacpp_mode

    if llamacpp_mode(profile) == "native":
        raise SettingsError(
            "bundle นี้เป็น llama.cpp แบบ native build (DGX Spark) — ไม่ได้ใช้ docker image จึงตั้ง --image ไม่ได้\n"
            "อัปเดตรันไทม์ด้วย: ปุ่ม update runtime บนการ์ด · lmds repair <slug> · "
            "LLAMA_CPP_UPDATE=1 <controller> prepare-runtime"
        )


_HEADER = [
    "# สร้างโดย LMDS — ค่าที่ตั้งไว้สำหรับ bundle นี้",
    "# controller อ่านไฟล์นี้ก่อนตั้ง default ทุกตัว ทุกบรรทัดเป็นรูป ${VAR:-value}",
    "# env จากภายนอกและ flag บรรทัดคำสั่งจึงยังชนะไฟล์นี้เสมอ",
    "#",
    "# แก้ด้วยมือได้ · ลบไฟล์ = กลับไปใช้ค่าของ bundle",
    "",
]


def write(bundle_dir: Path, values: dict[str, object]) -> dict[str, str]:
    """ตั้ง/เอาออกเฉพาะคีย์ที่อยู่ใน `values` — คืนค่าที่บันทึกไว้ทั้งหมดหลังเขียน (เท่ากับ `read()`)

    * คีย์ที่มีค่า = ตั้ง · คีย์ที่ค่าว่าง/None = เอา knob นั้นออก (กลับไปใช้ค่าของ bundle) · **คีย์ที่ไม่ได้ส่งมา = ไม่แตะ**
    * บรรทัดอื่นทุกบรรทัดของ bundle.env — env ที่ `lmds set` ไม่รู้จัก · รูป `NAME=value` ธรรมดา · คอมเมนต์ · บรรทัดว่าง —
      อยู่ที่เดิมไบต์ต่อไบต์ · ค่าที่ตั้งทับของเดิมถูกแทน **ที่บรรทัดเดิม** ไม่ว่าเดิมเขียนรูปไหน (ต่อท้ายเป็นรูป default
      หลังบรรทัด `NAME=เก่า` = ค่าใหม่ไม่มีผล) · ของใหม่ต่อท้ายไฟล์
    * `bundle.args` ถูกแตะเมื่อ `values` มีคีย์ `extra_args` เท่านั้น — ค่าว่างที่ส่งมาตรง ๆ คือทางเดียวที่ลบมัน

    เดิมฟังก์ชันนี้ "เขียนไฟล์ใหม่ทั้งไฟล์เสมอ" จากสิ่งที่ `read()` อ่านกลับได้ ซึ่งคือเฉพาะบรรทัดรูป `NAME="${NAME:-v}"` ของ
    knob ที่รู้จัก — ทั้งที่หัวไฟล์บอกว่าแก้ด้วยมือได้ และ manager.py เองอ่าน `STARTUP_TIMEOUT` / `HF_HOME` /
    `WORKER_HF_HOME` จากไฟล์นี้ · เคสจริงจาก audit 2026-10-06: หลัง `lmds set --port 8001` บน bundle stacked
    STARTUP_TIMEOUT 6906 → 1800 (โมเดล 122B โหลดไม่ทัน watchdog) · HF_HOME /data/hf → ว่าง · บรรทัด `API_HOST=…` /
    `WORKER_HF_HOME=…` หาย · และ `write(dir, {"port": …})` ตรง ๆ (PUT /settings ที่ body ไม่ครบ · web/deploy.py)
    ลบ `bundle.args` ที่ผู้ดูแลเก็บ flag ของ tokenizer/engine ไว้

    ล้างทุก knob ของ LMDS ใช้ `clear()` — dict ว่างไม่ใช่คำสั่งล้าง (เดิมเป็น · `keep = {…ที่กรองแล้วว่าง…}` จึงล้างทั้งไฟล์ได้)
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise SettingsError(f"ไม่พบโฟลเดอร์ bundle: {bundle_dir}")

    cleaned: dict[str, str] = {}
    removed: set[str] = set()
    for field, raw in values.items():
        if field not in FIELDS:
            continue  # ไม่รู้จักก็ไม่เขียน — รวมถึง api_key ที่ตั้งใจไม่เก็บ
        if raw is None or str(raw).strip() == "":
            removed.add(field)
            continue
        cleaned[field] = _clean(field, raw)  # image_min_tokens=auto → "" โดยตั้งใจ (ดู FIELDS)
    # ตรวจให้ครบก่อนแตะไฟล์ใดไฟล์หนึ่ง — ค่าที่ถูกปฏิเสธต้องไม่ทิ้ง bundle.args ที่เขียนไปแล้วครึ่งทาง
    _check_context_cap(bundle_dir, values, cleaned)
    if "image" in cleaned:
        _check_image_applies(bundle_dir)

    args_file = bundle_dir / ARGS_FILENAME
    if "extra_args" in cleaned:
        args_file.write_text(cleaned.pop("extra_args") + "\n", encoding="utf-8")
    elif "extra_args" in removed:
        args_file.unlink(missing_ok=True)

    wanted = {name: value for field, value in cleaned.items() for name in FIELDS[field]}
    dropped = {name for field in removed for name in FIELDS[field]}
    if not wanted and not dropped:
        return read(bundle_dir)

    target = path_for(bundle_dir)
    existed = target.is_file()
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if existed else []
    out: list[str] = []
    placed: set[str] = set()
    for line in lines:
        name = _assigned_name(line)
        if name in wanted:
            if name not in placed:
                # ค่าผ่าน _clean มาแล้ว (ไม่มี " ' ` $ \ { } หรือขึ้นบรรทัดใหม่) จึงวางตรง ๆ ได้ —
                # shlex.quote แล้ว strip quote ทิ้ง ไม่ได้ป้องกันอะไร แค่ทำให้ดูเหมือนปลอดภัย
                out.append(f'{name}="${{{name}:-{wanted[name]}}}"\n')
                placed.add(name)
            continue    # บรรทัดซ้ำของชื่อที่เพิ่งตั้ง — เหลือไว้จะกลับมาชนะ/สับสนว่าตัวไหนมีผล
        if name in dropped:
            continue
        out.append(line)
    missing = [name for name in wanted if name not in placed]
    if missing:
        if not existed:
            out = [line + "\n" for line in _HEADER]
        elif out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out += [f'{name}="${{{name}:-{wanted[name]}}}"\n' for name in missing]

    # เหลือแต่หัวไฟล์ของเราเอง/บรรทัดว่าง = ไม่มีอะไรให้ controller อ่าน — ลบไฟล์ ("ลบไฟล์ = กลับไปใช้ค่าของ bundle")
    # มีบรรทัดของผู้ดูแลเหลือแม้แต่คอมเมนต์เดียว = เก็บไฟล์ไว้
    if all(line.strip() == "" or line.rstrip("\n") in _HEADER for line in out):
        target.unlink(missing_ok=True)
    else:
        staged = target.with_name(target.name + ".tmp")
        staged.write_text("".join(out), encoding="utf-8")
        staged.replace(target)      # ไฟล์นี้ถูก source ตอน start/autostart — เขียนครึ่งไฟล์แล้วล้มคือ start พัง
    return read(bundle_dir)


def clear(bundle_dir: Path) -> dict[str, str]:
    """เอา knob ทุกตัวที่ `lmds set` ดูแลออกจาก bundle.env — คืนสิ่งที่ยังบันทึกอยู่ (เท่ากับ `read()`)

    ไม่แตะบรรทัดที่ผู้ดูแลเพิ่มเอง (ดู `foreign_names`) และไม่แตะ `bundle.args`: flag ของ tokenizer/engine ในไฟล์นั้นลบได้
    ทางเดียวคือสั่ง extra_args ว่างมาตรง ๆ · ไฟล์ที่เหลือแต่หัวของเราเองถูกลบ
    """
    return write(bundle_dir, {field: "" for field in FIELDS if field != "extra_args"})


def foreign_names(bundle_dir: Path) -> list[str]:
    """ชื่อ env ใน bundle.env ที่ไม่ใช่ knob ของ `lmds set` (ผู้ดูแลเพิ่มเอง) — ไว้บอกผู้ใช้ว่า --clear เหลืออะไรไว้"""
    target = path_for(bundle_dir)
    if not target.is_file():
        return []
    names = [_assigned_name(line) for line in target.read_text(encoding="utf-8", errors="replace").splitlines()]
    return list(dict.fromkeys(name for name in names if name and name not in _FIELD_OF))


# บล็อกเดียวกับที่ template ใส่ให้ bundle ใหม่ — เก็บไว้ที่นี่ด้วยเพื่อเติมให้ bundle
# ที่ deploy ไปก่อนหน้านี้ ซึ่งเป็นทุกตัวที่ผู้ใช้มีอยู่ตอนนี้
SOURCE_BLOCK = """# ── ค่าที่บันทึกไว้กับ bundle นี้ (เขียนโดย `lmds set` / หน้าเว็บ) ──
# อ่านก่อน default ทั้งหมดข้างล่าง และทุกบรรทัดในไฟล์เป็นรูป ${VAR:-value} ลำดับ
# ความสำคัญจึงเป็น: flag บรรทัดคำสั่ง > env จากภายนอก > ไฟล์นี้ > ค่าของ bundle
BUNDLE_ENV="${BUNDLE_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bundle.env}"
if [[ -f "$BUNDLE_ENV" ]]; then
  set -a; . "$BUNDLE_ENV"; set +a
fi

"""


def ensure_controller_reads(controller: Path) -> bool:
    """เติมบล็อกอ่าน bundle.env ให้ controller ที่สร้างก่อนฟีเจอร์นี้

    การแก้ template มีผลกับ bundle ที่ generate ใหม่เท่านั้น ส่วนที่ deploy ไปแล้ว
    จะเขียน bundle.env ไปก็ไม่มีใครอ่าน — ซึ่งคือทุก bundle ที่มีอยู่ตอนนี้

    คืน True เมื่อเพิ่งเติมให้ · False เมื่อมีอยู่แล้วหรือแก้ไม่ได้
    """
    import re
    import shutil
    import subprocess
    import time

    controller = Path(controller)
    if not controller.is_file():
        return False
    text = controller.read_text(encoding="utf-8")
    if "BUNDLE_ENV" in text:
        return False

    # วางก่อน default ตัวแรก (บรรทัดรูป NAME="${NAME:-…}") — ก่อนหน้านั้นเป็น
    # หัวไฟล์กับ set -euo pipefail ซึ่งต้องมาก่อนการ source
    m = re.search(r'^[A-Z_][A-Z0-9_]*="\$\{[A-Z_]', text, flags=re.M)
    if m is None:
        return False
    patched = text[: m.start()] + SOURCE_BLOCK + text[m.start():]

    candidate = controller.with_suffix(controller.suffix + ".cand")
    candidate.write_text(patched, encoding="utf-8")
    if subprocess.run(["bash", "-n", str(candidate)]).returncode != 0:
        candidate.unlink(missing_ok=True)
        return False
    shutil.copy2(controller, f"{controller}.bak-bundleenv-{time.strftime('%H%M%S')}")
    shutil.copymode(controller, candidate)
    candidate.replace(controller)
    return True
