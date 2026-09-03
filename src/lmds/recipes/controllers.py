"""อ่านสูตรจาก "controller ที่รันผ่านแล้ว" ในรีโปภายนอก (เช่น dgx-spark-all-controllers)

ทำไมต้องมี: ความรู้ว่าโมเดลไหนต้องใช้ image/ค่าอะไร อยู่ในสคริปต์ controller ที่ทีมรันจริง
บนเครื่องอยู่แล้ว การมานั่งพิมพ์ซ้ำลง catalog.yaml ของ LMDS แปลว่าต้องแก้สองที่ทุกครั้ง
แล้วสองที่ก็จะหลุดกันในที่สุด — ดึงจากรีโปที่เป็นต้นทางจริงจึงเป็นทางเดียวที่ทำให้ตรงกันเสมอ

**อ่านอย่างเดียว ไม่รัน** — controller เป็น Bash ที่ดาวน์โหลดโมเดลและสั่ง docker การรันเพื่อ
ถามข้อมูลจึงเป็นการรันโค้ดจากอินเทอร์เน็ตบนเครื่อง hub · ส่วนหัวของทุกตัวเป็นบล็อกตัวแปร
รูปแบบเดียวกัน (`KEY="ค่า"` หรือ `KEY="${KEY:-ค่า}"`) จึงอ่านด้วย regex ได้ตรงและปลอดภัย
"""

from __future__ import annotations

import re
from pathlib import Path

# รับเฉพาะบรรทัดที่เริ่มชิดซ้าย = ตัวแปรตั้งค่าระดับบนสุดของสคริปต์ · ตัวแปรที่อยู่ในฟังก์ชัน
# ย่อหน้าเสมอ จึงถูกคัดออกเอง (สำคัญกับ controller แบบ stacked ที่มีฟังก์ชันถามค่าคั่นกลาง
# ก่อนบล็อกตั้งค่าจริง — ถ้าหยุดอ่านที่ฟังก์ชันแรกจะพลาดทั้งไฟล์)
# รับทั้ง KEY="ค่า" และ KEY='ค่า' — แฟล็กเพิ่มที่มี JSON ต้องอยู่ใน single quote (ดู template)
_ASSIGN = re.compile(r'^([A-Z][A-Z0-9_]*)=(?:"([^"\n]*)"|\'([^\'\n]*)\')\s*(?:#.*)?$')
_DEFAULT = re.compile(r'^\$\{[A-Z][A-Z0-9_]*:-(.*)\}$')

# `MODEL_FILES=( "ก.gguf" "ข.gguf" )` — ชิ้นแรกคือไฟล์ที่ controller เสิร์ฟจริง
#
# controller ของ llama.cpp เก็บรายชื่อ shard ไว้ในอาร์เรย์ แล้วค่อยตั้ง
# `MODEL_FILE="${MODEL_FILES[0]}"` · ตัวอ่าน header รับได้เฉพาะบรรทัด KEY="ค่า" ธรรมดา
# ค่าที่ได้จึงเป็นสตริง '${MODEL_FILES[0]}' ซึ่งถูกทิ้งเพราะมี `$` — ผลคือสูตรของ GGUF
# ทุกตัวไม่มี `gguf_file` ทั้งที่ "quant ไหนที่รันผ่าน" คือข้อมูลชิ้นสำคัญที่สุดของสูตรนั้น
_ARRAY_HEAD = re.compile(r'^([A-Z][A-Z0-9_]*)=\(\s*(?:#.*)?$')
_ARRAY_ITEM = re.compile(r'^\s*"([^"\n$]+)"\s*(?:#.*)?$')

# สคริปต์ที่ไม่ใช่ controller ของโมเดล — เป็นเครื่องมือของรีโปเอง
TOOLING = {"install-canonical.sh", "verify-all.sh", "audit-controllers.py"}

# ฟิลด์ที่สูตรต้องมีถึงจะเข้าแคตตาล็อกได้ — ตรวจตอน sync คือตอนที่ยังบอกได้ว่า
# controller ตัวไหนเป็นต้นเหตุ · ปล่อยเข้าไปแล้วค่อยเจอตอน deploy คือสายเกินไป
REQUIRED_FIELDS = ("engine", "source", "validated_on")


def parse_header(text: str) -> dict[str, str]:
    """ตัวแปรตั้งค่าระดับบนสุดของ controller — คืน {} ถ้าไม่ใช่ไฟล์รูปแบบนี้

    ค่าแรกที่เจอชนะ: สคริปต์อาจเขียนทับตัวแปรเดิมทีหลังตามเงื่อนไข ซึ่งเดาไม่ได้ว่าเส้นทางไหน
    จะถูกใช้จริง — ค่าที่ประกาศไว้ตอนต้นคือค่าที่ตั้งใจให้เป็นค่าตั้งต้น
    """
    values: dict[str, str] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        array = _ARRAY_HEAD.match(line)
        if array:
            key = array.group(1)
            # เก็บเฉพาะชิ้นแรก — ชิ้นถัดไปคือ shard ที่ llama.cpp ต่อให้เอง
            for follow in lines[index + 1:]:
                if follow.startswith(")"):
                    break
                item = _ARRAY_ITEM.match(follow)
                if item:
                    values.setdefault(key, item.group(1).strip())
                    break
            continue

        found = _ASSIGN.match(line)          # ไม่ strip — บรรทัดที่ย่อหน้าคืออยู่ในฟังก์ชัน
        if not found:
            continue
        key, raw = found.group(1), (found.group(2) if found.group(2) is not None else found.group(3))
        if key in values:
            continue
        default = _DEFAULT.match(raw)
        value = default.group(1) if default else raw
        # ค่าที่อ้างตัวแปรอื่นต่อ (เช่น ${USER_HOME}/models/...) แปลไม่ได้โดยไม่รันสคริปต์
        if "$" in value:
            continue
        values[key] = value.strip()
    return values


def _engine(meta: dict[str, str]) -> str:
    runtime = (meta.get("RUNTIME_LABEL") or "").lower()
    if "llama.cpp" in runtime:
        return "llamacpp"
    if "sglang" in runtime:
        return "sglang"
    if "vllm" in runtime:
        return "vllm"
    # controller รุ่นเก่าไม่มี RUNTIME_LABEL — เดาจากตัวแปรเฉพาะ engine แทนที่จะตกไปเป็น
    # "ไม่รู้ engine" แล้วโดน skip · เจอจริง: qwen3-coder-30b-a3b-instruct ที่ generate
    # ด้วย lmds เก่า มี LLAMACPP_IMAGE/LLAMA_CPP_REPO ครบแต่ไม่มี RUNTIME_LABEL
    keys = " ".join(meta)
    if "LLAMACPP_IMAGE" in keys or "LLAMA_CPP_REPO" in keys or "LLAMA_SERVER" in keys:
        return "llamacpp"
    if "SGLANG_IMAGE" in keys or "SGLANG_" in keys:
        return "sglang"
    if "VLLM_IMAGE" in keys:
        return "vllm"
    return ""


def _serving(meta: dict[str, str]) -> dict:
    """เฉพาะค่าที่ Serving ของ LMDS มีจริง — คีย์ที่ไม่รู้จักจะไปโผล่เป็น flag แปลก ๆ ใน bundle

    ไม่ดึง context/max_model_len มาด้วยโดยตั้งใจ: ต้องมาจากการวิเคราะห์หน่วยความจำของ
    เครื่องเป้าหมาย ไม่ใช่ค่าคงที่ของเครื่องที่เคยรัน (กติกาเดียวกับ catalog.yaml)
    """
    out: dict = {}
    if meta.get("GPU_MEMORY_UTILIZATION"):
        try:
            out["gpu_memory_utilization"] = float(meta["GPU_MEMORY_UTILIZATION"])
        except ValueError:
            pass
    if meta.get("MAX_NUM_SEQS"):
        try:
            out["max_num_seqs"] = int(meta["MAX_NUM_SEQS"])
        except ValueError:
            pass
    if meta.get("KV_CACHE_DTYPE"):
        out["kv_cache_dtype"] = meta["KV_CACHE_DTYPE"]
    return out


def recipe_from_controller(filename: str, text: str, origin: str = "") -> dict | None:
    """แปลง controller หนึ่งตัวเป็นสูตร — คืน None ถ้าอ่านรุ่นโมเดลไม่ได้

    ไม่เดาแทนไฟล์ที่อ่านไม่ออก: สูตรที่ match ผิดจะไปทับค่าของโมเดลอื่นเงียบ ๆ
    ซึ่งแย่กว่าการไม่มีสูตรเลย
    """
    meta = parse_header(text)
    model = meta.get("MODEL_ID") or meta.get("HF_REPO") or meta.get("REPO_ID")
    if not model:
        # controller ที่คุมหลายโมเดลในไฟล์เดียว (เช่น GLM_REPO_ID + QWEN_REPO_ID) แปลงเป็น
        # สูตรเดียวไม่ได้ — ปล่อยให้ผู้เรียกรายงานว่าอ่านไม่ได้ ดีกว่าเดาเอาโมเดลใดโมเดลหนึ่ง
        return None

    stacked = "stacked" in (meta.get("RUNTIME_LABEL") or "").lower() or "stacked" in filename
    features = [f.strip() for f in (meta.get("MODEL_FEATURES") or "").split("·") if f.strip()]
    recipe: dict = {
        "match": model,
        "label": meta.get("MODEL_LABEL") or model,
        "engine": _engine(meta),
        "serving": _serving(meta),
        "notes": features,
        "source": f"{origin} · {filename}" if origin else filename,
        "validated_on": (f"{meta.get('RUNTIME_LABEL', '')} · controller {filename} "
                         f"v{meta.get('SCRIPT_VERSION', '?')}").strip(" ·"),
        # ข้อมูลที่ catalog.yaml ไม่มี — บอกว่าสูตรนี้มาจาก controller ตัวไหนและรันแบบไหน
        "controller": filename,
        "topology": "stacked" if stacked else "single",
    }
    if meta.get("VLLM_IMAGE") and recipe["engine"] == "vllm":
        recipe["image"] = meta["VLLM_IMAGE"]
    if meta.get("SERVED_MODEL_NAME"):
        recipe["served_model_name"] = meta["SERVED_MODEL_NAME"]
    # ค่าที่ publish พับลงมาจาก bundle.env/bundle.args ของเครื่องที่พิสูจน์แล้ว — ของโมเดล ไม่ใช่ของเครื่อง
    if meta.get("TOOL_CALL_PARSER"):
        recipe["tool_parser"] = meta["TOOL_CALL_PARSER"]
    if meta.get("REASONING_PARSER"):
        recipe["reasoning_parser"] = meta["REASONING_PARSER"]
    if meta.get("ENGINE_ENV"):
        recipe["engine_env"] = meta["ENGINE_ENV"]
    if meta.get("EXTRA_SERVE_ARGS_DEFAULT"):
        recipe["extra_args"] = meta["EXTRA_SERVE_ARGS_DEFAULT"]
    # llama.cpp — ไฟล์ GGUF ที่ทดสอบมา · รุ่นใหม่เก็บไว้ในอาร์เรย์ MODEL_FILES
    # (ชิ้นแรก) ส่วนรุ่นเก่าตั้ง MODEL_FILE ตรง ๆ
    gguf = meta.get("MODEL_FILE") or meta.get("MODEL_FILES")
    if gguf and gguf.endswith(".gguf"):
        recipe["gguf_file"] = gguf
    return recipe


def scan_directory(root: Path, origin: str = "") -> tuple[list[dict], list[str]]:
    """สูตรทั้งหมดที่อ่านได้จากรีโป controller หนึ่งชุด → (สูตร, ไฟล์ที่ข้าม + เหตุผล)

    เรียงตามชื่อไฟล์ให้ผลคงที่ · รายการที่ข้ามต้องถูกรายงานออกไป ไม่ใช่หายเงียบ —
    ไม่งั้นคนแก้รีโปจะไม่มีวันรู้ว่า controller ตัวใหม่ที่เพิ่งเพิ่มยังไม่ถูกดึงเข้ามา
    """
    recipes: list[dict] = []
    skipped: list[str] = []
    seen: dict[str, str] = {}
    # โมเดลเดียวกันมักมีทั้งตัว single และ stacked — ให้ single ชนะเสมอ เพราะ LMDS เลือก
    # topology เองจากขนาดโมเดลกับเครื่องที่มี ส่วนค่าที่เหลือ (image/parser) สองตัวใช้ร่วมกัน
    # rglob ไม่ใช่ glob: controller อยู่ได้ทั้งที่ root (แบบ flat เดิม) และใน controllers/<slug>/
    # (แบบที่ `--publish` เขียน) · ข้าม .git ไม่ให้ไปอ่านสคริปต์ hook ของ git เป็นสูตร
    candidates = (p for p in root.rglob("*.sh") if ".git" not in p.parts)
    for path in sorted(candidates, key=lambda p: ("stacked" in p.name, p.name)):
        if path.name in TOOLING:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"{path.name}: อ่านไฟล์ไม่ได้ ({exc})")
            continue
        recipe = recipe_from_controller(path.name, text, origin)
        if recipe is None:
            skipped.append(f"{path.name}: ไม่พบรุ่นโมเดล (MODEL_ID/HF_REPO/REPO_ID) ที่ระดับบนสุด")
            continue
        # สูตรที่ไม่มีที่มาคือการเดา — ห้ามเข้าแคตตาล็อก · เคสที่หลุดได้จริงคือ engine ว่าง
        # เพราะ RUNTIME_LABEL ไม่มีคำที่รู้จัก (llama.cpp/sglang/vllm) แล้ว _engine() คืน ""
        # ซึ่งจะกลายเป็น bundle ที่ไม่รู้ว่าจะรันด้วยอะไร · ข้ามแล้วบอก ดีกว่าเก็บไว้เงียบ ๆ
        missing = [f for f in REQUIRED_FIELDS if not recipe.get(f)]
        if missing:
            skipped.append(f"{path.name}: สูตรไม่ครบ (ขาด {', '.join(missing)}) — ไม่เอาเข้าแคตตาล็อก")
            continue
        first = seen.get(recipe["match"].lower())
        if first:
            # โมเดลเดียวกันมีสองสูตร (คนละ quant/คนละ image) — LMDS เลือกได้ตัวเดียวต่อหนึ่ง
            # repo id จึงต้องบอกให้รู้ว่าตัวไหนถูกใช้ ไม่ใช่เงียบแล้วให้งงทีหลัง
            skipped.append(f"{path.name}: {recipe['match']} ซ้ำกับ {first} — ใช้ตัวแรก")
            continue
        seen[recipe["match"].lower()] = path.name
        recipes.append(recipe)
    return recipes, skipped
