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
}


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
    # ชื่อโมเดล/image เป็นข้อความอิสระ แต่ต้องไม่มีอักขระที่ทำให้ shell ตีความ
    if any(ch in text for ch in "\n\r\x00"):
        raise SettingsError(f"{name} มีอักขระขึ้นบรรทัดใหม่")
    return text


def path_for(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / FILENAME


def read(bundle_dir: Path) -> dict[str, str]:
    """ค่าที่บันทึกไว้ — คืน dict ว่างเมื่อยังไม่เคยบันทึก"""
    target = path_for(bundle_dir)
    if not target.is_file():
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
                out[field] = env[name]
                break
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
        cleaned[field] = _clean(field, raw)

    target = path_for(bundle_dir)
    if not cleaned:
        target.unlink(missing_ok=True)
        return {}

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
            lines.append(f'{name}="${{{name}:-{shlex.quote(value).strip(chr(39))}}}"')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cleaned
