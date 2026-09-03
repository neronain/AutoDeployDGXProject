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

**ไม่เก็บ API key ไว้ที่นี่** หน้าเว็บสัญญากับผู้ใช้ไว้ว่า key อยู่ในเบราว์เซอร์
เท่านั้น ไม่ได้เขียนลง bundle — ไฟล์นี้อยู่ในโฟลเดอร์ที่ถูก zip แจกต่อได้
"""

from __future__ import annotations

import re
import shlex
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


def read(bundle_dir: Path) -> dict[str, str]:
    """ค่าที่บันทึกไว้ — คืน dict ว่างเมื่อยังไม่เคยบันทึก"""
    target = path_for(bundle_dir)
    if not target.is_file():
        args_file = Path(bundle_dir) / ARGS_FILENAME
        if args_file.is_file() and args_file.read_text(encoding="utf-8").strip():
            return {"extra_args": args_file.read_text(encoding="utf-8").strip()}
        return {}
    env: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        # รูปที่เราเขียนเองคือ NAME="${NAME:-value}"
        m = re.match(r'^([A-Z_][A-Z0-9_]*)="\$\{\1:-(.*)\}"$', line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    out: dict[str, str] = {}
    for field, names in FIELDS.items():
        for name in names:
            if name in env:
                value = env[name]
                if field == "image_min_tokens" and value == "":
                    value = "auto"  # ค่าว่างในไฟล์ = auto — ให้ round-trip ผ่าน write() ได้โดยไม่หาย
                out[field] = value
                break
    args_file = Path(bundle_dir) / ARGS_FILENAME
    if args_file.is_file():
        extra = args_file.read_text(encoding="utf-8").strip()
        if extra:
            out["extra_args"] = extra
    return out


def write(bundle_dir: Path, values: dict[str, object]) -> dict[str, str]:
    """บันทึกค่าลง bundle.env — ค่าที่เป็นค่าว่างคือ "เอาออก ใช้ default ของ bundle"

    เขียนไฟล์ใหม่ทั้งไฟล์เสมอ ไม่ต่อท้าย เพราะการต่อท้ายจะทำให้ค่าเก่ากับใหม่อยู่
    ปนกันแล้วอ่านยากว่าตัวไหนมีผล
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise SettingsError(f"ไม่พบโฟลเดอร์ bundle: {bundle_dir}")

    cleaned: dict[str, str] = {}
    for field, raw in values.items():
        if field not in FIELDS:
            continue  # ไม่รู้จักก็ไม่เขียน — รวมถึง api_key ที่ตั้งใจไม่เก็บ
        if raw is None or str(raw).strip() == "":
            continue
        cleaned[field] = _clean(field, raw)  # image_min_tokens=auto → "" โดยตั้งใจ (ดู FIELDS)

    args_file = bundle_dir / ARGS_FILENAME
    extra = cleaned.pop("extra_args", None)
    if extra:
        args_file.write_text(extra + "\n", encoding="utf-8")
    else:
        args_file.unlink(missing_ok=True)

    target = path_for(bundle_dir)
    if not cleaned:
        target.unlink(missing_ok=True)
        return {"extra_args": extra} if extra else {}

    lines = [
        "# สร้างโดย LMDS — ค่าที่ตั้งไว้สำหรับ bundle นี้",
        "# controller อ่านไฟล์นี้ก่อนตั้ง default ทุกตัว ทุกบรรทัดเป็นรูป ${VAR:-value}",
        "# env จากภายนอกและ flag บรรทัดคำสั่งจึงยังชนะไฟล์นี้เสมอ",
        "#",
        "# แก้ด้วยมือได้ · ลบไฟล์ = กลับไปใช้ค่าของ bundle",
        "",
    ]
    for field, value in cleaned.items():
        for name in FIELDS[field]:
            # ค่าผ่าน _clean มาแล้ว (ไม่มี " ' ` $ \ { } หรือขึ้นบรรทัดใหม่) จึงวางตรง ๆ ได้ —
            # shlex.quote แล้ว strip quote ทิ้ง ไม่ได้ป้องกันอะไร แค่ทำให้ดูเหมือนปลอดภัย
            lines.append(f'{name}="${{{name}:-{value}}}"')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if extra:
        cleaned["extra_args"] = extra
    return cleaned


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
