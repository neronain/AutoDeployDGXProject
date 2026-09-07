"""ตั้ง slots / context / KV ให้พอดี — สูตรเดียวที่ทำให้รัน 2 โมเดลบน vLLM ผ่านบน DGX Spark

เจ้าของ 2026-09-07: "ผมอยากทราบค่าที่ทำให้รัน 2 model บน vllm ผ่าน … ถ้าอนาคตรันแบบนี้อีก หรือกับ node
อื่นหรือลูกค้า จะทราบได้อย่างไรว่าควรใช้ค่าไหน เช่น ct = 262144 แต่ slot, gpu util จะตั้งค่าอย่างไรให้พอดี"
— ไม่ควรมีใครต้องคำนวณเองอีก ทั้ง CLI (`lmds fit` / `lmds set --fit`) หน้าเว็บ (ปุ่ม Fit) และตอน deploy
(pin ตั้งต้นบนเครื่อง unified) เรียกฟังก์ชันเดียวกันในไฟล์นี้

กติกาที่พิสูจน์แล้ว (unified memory: gpu-util ของ vLLM คิดจาก *ทั้ง* 128 GB รวม OS ด้วย 0.85 = 109 GB ต่อโมเดล
ซึ่งไม่เหลือให้ตัวที่สอง — จึงเลิกคุมด้วย gpu-util แล้วปักหมุด KV แทน):

    RAM ของโมเดล vLLM  = weights (ตามที่โหลดจริง) + overhead ~3 GB (activation/CUDA graph) + KV pin
    KV pin (bytes)      = slots × KV ของ 1 คำขอเต็ม context × 1.2      ← ส่งเป็น --kv-cache-memory ไม่ใช่ gpu-util
    เหลือให้ OS/desktop ≥ 12 GB → ใช้ได้ ≈ total − 12 (121 GiB ที่ free -g เห็น → ~109)
    โมเดลที่สองใส่ได้ก็ต่อเมื่อ  Σ RAM ของทุกโมเดล ≤ ที่ใช้ได้

    context คงไว้ที่ native ของโมเดล (pin แล้ว context ไม่ทำให้ RAM เพิ่ม) · slots = จำนวนคนที่ใช้พร้อมกันจริง (2–4 ต่อคนเดียว)
    llama.cpp: weights + KV(ctx ทั้ง pool) + overhead 1.5 — จอง KV ล่วงหน้าทั้งก้อน แบ่งให้ทุก slot เท่ากัน

วัดจริง 2026-09-07 (ดู docstring ใน tests/test_kv_sizing.py):
    spark-head   Nemotron-3-Super-120B NVFP4 (hybrid Mamba): loading took 69.62 GiB · pin 6 GiB → 1,179,648 tokens
                 (= 1.33 GiB ต่อคำขอ 262K ที่ KV fp8 · concurrency 4.5x) → slots 2 · ~79 GiB + Gemma-4 llama.cpp ctx 65536 ข้าง ๆ
    spark-worker Qwopus3.5-122B-A10B NVFP4: loading took 71.32 GiB · pin 12 GiB → 508,031 tokens (= 6.2 GiB ต่อคำขอ 262K)
                 → slots 3 · compute-apps ถือ 86,263 MiB · free -g used 95/121 (OS ~11 GB)

ไฟล์นี้คำนวณล้วน ไม่แตะเครื่อง — ข้อเท็จจริง (profile · log ของ vLLM · free -g · โมเดลอื่นที่รันอยู่) ให้ผู้เรียก
รวบรวมมา (fleet/sizing.py) · คืน dict ที่มีทั้งตัวเลขและรหัสเหตุผล ปลายทางแต่ละที่เรียบเรียงเป็นภาษาของตัวเอง
"""

from __future__ import annotations

import math
import re

from .analyzer import (
    GIB,
    LLAMACPP_OVERHEAD_GB,
    UNIFIED_OS_RESERVE_GB,
    VLLM_MIN_KV_GB,
)

# activation / CUDA graph / MoE workspace ของ vLLM ที่ไม่อยู่ใน weights และ KV — วัดบน Spark ~0.9–3 GiB
# (Qwopus: compute-apps 84.2 GiB − loading 71.32 − pin 12 = 0.9 · Nemotron ใช้ CUDA graph เต็ม ~3) เผื่อไว้ที่ 3
VLLM_RUNTIME_OVERHEAD_GB = 3.0
# เผื่อ KV เกินที่คำนวณ 20% — block ของ vLLM ปัดเป็นหน้า + mamba/linear state ของ hybrid ที่สูตร per-token ไม่นับ
KV_PIN_HEADROOM = 1.2
# ปัด pin ขึ้นเป็นขั้นละครึ่ง GiB — เลขกลม ๆ อ่านใน docker inspect แล้วรู้ทันทีว่ามาจากไหน
PIN_STEP_GIB = 0.5
# discrete (RTX): driver/display ถือไว้นิดหน่อย ไม่มี OS อยู่ใน pool เดียวกันแบบ Spark
DISCRETE_RESERVE_GB = 1.0
# DGX Spark ขาย "128 GB" (decimal) แต่ free -g เห็น 121 GiB — ตอนไม่มีเครื่องจริงให้ถาม (preset ตอน deploy)
# ต้องหักส่วนต่างนี้ ไม่งั้น pin ที่คิดจาก 128 จะกินเข้าไปในที่ที่ไม่มี (116/121 = swap)
UNIFIED_DECIMAL_TO_GIB = 121.0 / 128.0
# เหลือน้อยกว่านี้หลังวางทุกอย่างแล้ว = ใส่ได้แต่ตึง (page cache/desktop ขยับนิดเดียวก็ swap)
TIGHT_MARGIN_GB = 4.0

_PIN_RE = re.compile(r"(?:^|\s)--kv-cache-memory(?:=|\s+)(\d+)(?=\s|$)")
_DTYPE_RE = re.compile(r"(?:^|\s)--kv-cache-dtype(?:=|\s+)(\S+)")


# ── แฟล็กใน bundle.args ───────────────────────────────────────────────────────
def parse_kv_pin(extra_args: str | None) -> int | None:
    """ค่า --kv-cache-memory (bytes) ที่ตั้งไว้ใน extra args — None ถ้าไม่มี"""
    m = _PIN_RE.search(extra_args or "")
    return int(m.group(1)) if m else None


def strip_kv_pin(extra_args: str | None) -> str:
    return " ".join(_PIN_RE.sub(" ", extra_args or "").split())


def with_kv_pin(extra_args: str | None, pin_bytes: int) -> str:
    """เขียน --kv-cache-memory ทับของเดิม (ถ้ามี) — แฟล็กอื่นคงไว้ตามลำดับ"""
    rest = strip_kv_pin(extra_args)
    return f"{rest} --kv-cache-memory {int(pin_bytes)}".strip()


def kv_dtype_from_args(extra_args: str | None) -> str:
    """bf16 | fp8 — `--kv-cache-dtype fp8*` ลด KV ต่อ token ครึ่งหนึ่งของค่าใน profile (ซึ่งคิดที่ 2 ไบต์)"""
    m = _DTYPE_RE.search(extra_args or "")
    value = (m.group(1) if m else "auto").lower()
    return "fp8" if value.startswith("fp8") else "bf16"


# ── ตัวเลขจริงจาก log ของ vLLM ─────────────────────────────────────────────────
_LOADING_RE = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_AVAILABLE_RE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_RESERVED_RE = re.compile(r"reserved ([\d.]+) GiB memory for KV Cache")
_CONCURRENCY_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_INITIAL_FREE_RE = re.compile(r"Initial free memory ([\d.]+) GiB")


def measured_from_log(text: str) -> dict:
    """ดึงบรรทัดที่ vLLM พิมพ์ตอน start — ค่าล่าสุดชนะ (restart หลายรอบใน log เดียว)

    ``loading_took_gib``  weights ตามที่โหลดจริง (NVFP4 120B = 69.62 ทั้งที่ไฟล์บนดิสก์ 74.8)
    ``kv_cache_tokens``   ขนาด pool ที่ได้จริง — หารด้วย pin/available = KV ต่อ token ที่รวม block rounding และ
                          state ของ mamba/linear attention ซึ่งสูตรจาก config ไม่นับ
    ``kv_pool_gib``       pin (บรรทัด "reserved X GiB … kv_cache_memory_bytes") หรือที่ profiling ให้ ("Available KV cache memory")
    ``concurrency``       "Maximum concurrency for N tokens per request: X.XXx"
    """
    out: dict = {}
    if not text:
        return out
    for key, rx in (("loading_took_gib", _LOADING_RE), ("initial_free_gib", _INITIAL_FREE_RE),
                    ("available_kv_gib", _AVAILABLE_RE), ("reserved_kv_gib", _RESERVED_RE)):
        found = rx.findall(text)
        if found:
            out[key] = float(found[-1])
    tokens = _KV_TOKENS_RE.findall(text)
    if tokens:
        out["kv_cache_tokens"] = int(tokens[-1].replace(",", ""))
    conc = _CONCURRENCY_RE.findall(text)
    if conc:
        out["concurrency_context"] = int(conc[-1][0].replace(",", ""))
        out["concurrency"] = float(conc[-1][1])
    pool = out.get("reserved_kv_gib") or out.get("available_kv_gib")
    if pool:
        out["kv_pool_gib"] = pool
    return out


# ── สูตร ────────────────────────────────────────────────────────────────────────
def _round_up_half(gb: float) -> float:
    return math.ceil(gb / PIN_STEP_GIB) * PIN_STEP_GIB


def _r(x: float | None, digits: int = 1) -> float | None:
    return None if x is None else round(float(x), digits)


def plan_kv_pin(model_info: dict, host_info: dict, slots: int | None = None,
                context: int | None = None) -> dict:
    """ตั้ง slots/context เท่านี้บนเครื่องนี้ → ต้องใช้ RAM เท่าไร ปักหมุด KV เท่าไร ใส่ได้ไหม

    ``model_info``  engine · weight_bytes · kv_bytes_per_token (bf16, ต่อคลัสเตอร์) · native_context · context/slots
                    ที่ตั้งอยู่ · extra_args (หา pin/dtype) · running · own_gb (ที่ตัวมันถืออยู่ตอนนี้) ·
                    measured (จาก measured_from_log) · node_count · slug
    ``host_info``   memory_model · total_gb · free_gb (ตอนนี้) · held_gb (ที่ process GPU ทุกตัวถือรวมกัน) ·
                    others [{slug, gb}] (โมเดลอื่นที่รันอยู่ แยกรายตัวถ้ารู้)

    ค่าที่คืนถูกปัดที่นี่ (1 ตำแหน่ง) เพราะทุกปลายทางแสดงตรง ๆ · bytes เป็น int เสมอ
    """
    engine = (model_info.get("engine") or "vllm").lower()
    pin_supported = engine == "vllm"
    vllm_like = engine in ("vllm", "sglang")
    unified = (host_info.get("memory_model") or "unified") == "unified"
    node_count = max(1, int(model_info.get("node_count") or 1))
    notes: list[str] = []

    native = model_info.get("native_context") or None
    context = int(context or model_info.get("context") or native or 0)
    if native and context > native:
        notes.append(f"context {context:,} เกิน native {native:,} — ใช้ {native:,}")
        context = int(native)
    slots = int(slots or model_info.get("slots") or (4 if vllm_like else 1))
    slots = max(1, slots)

    extra_args = model_info.get("extra_args") or ""
    current_pin = parse_kv_pin(extra_args)
    kv_dtype = kv_dtype_from_args(extra_args)
    measured = dict(model_info.get("measured") or {})

    # KV ต่อ token — ค่าวัดจริงชนะค่าจาก config (self-correcting): pool ÷ tokens ที่ vLLM รายงาน
    per_token: float | None = None
    kv_source = None
    pool_gib = measured.get("kv_pool_gib") or (current_pin / GIB if current_pin else None)
    if measured.get("kv_cache_tokens") and pool_gib and node_count == 1:
        per_token = pool_gib * GIB / measured["kv_cache_tokens"]
        kv_source = "measured"
    elif model_info.get("kv_bytes_per_token"):
        per_token = float(model_info["kv_bytes_per_token"]) * (0.5 if kv_dtype == "fp8" else 1.0)
        kv_source = "profile"
        if kv_dtype == "fp8":
            notes.append("KV fp8 (--kv-cache-dtype) — คิด KV ต่อ token ครึ่งหนึ่งของ bf16")

    # weights — ที่โหลดจริงชนะขนาดไฟล์บนดิสก์ (NVFP4 120B: 69.62 GiB จริง vs 74.8 บนดิสก์)
    weights_gb: float | None = None
    weights_source = None
    if measured.get("loading_took_gib") and node_count == 1:
        weights_gb = float(measured["loading_took_gib"])
        weights_source = "measured"
    elif model_info.get("weight_bytes"):
        weights_gb = float(model_info["weight_bytes"]) / GIB
        weights_source = "profile"

    overhead_gb = VLLM_RUNTIME_OVERHEAD_GB if vllm_like else LLAMACPP_OVERHEAD_GB

    # KV ที่ต้องมี
    kv_per_request_gb = (per_token * context / GIB) if (per_token and context) else None
    kv_pin_bytes: int | None = None
    kv_gb: float | None = None
    if kv_per_request_gb is not None:
        if vllm_like:
            raw = max(VLLM_MIN_KV_GB, slots * kv_per_request_gb * KV_PIN_HEADROOM)
            kv_gb = _round_up_half(raw)
            kv_pin_bytes = int(kv_gb * GIB) if pin_supported else None
        else:
            # llama.cpp: --ctx-size คือ pool ทั้งก้อน แบ่งให้ทุก slot เท่ากัน — KV = ctx × per token ไม่คูณ slot
            kv_gb = kv_per_request_gb

    # เครื่อง
    total_gb = host_info.get("total_gb")
    os_reserve = UNIFIED_OS_RESERVE_GB if unified else DISCRETE_RESERVE_GB
    usable_gb = (float(total_gb) - os_reserve) if total_gb else None

    running = bool(model_info.get("running"))
    ram_needed_gb = (weights_gb + overhead_gb + kv_gb) if (weights_gb is not None and kv_gb is not None) else None

    # โมเดลอื่นบนเครื่อง — *ไม่นับตัวเอง* ตอนมันรันอยู่ (เดิมนับ = "Cannot start now" ทั้งที่รันอยู่แล้ว)
    held_gb = float(host_info.get("held_gb") or 0.0)
    own_gb = model_info.get("own_gb")
    if own_gb is None:
        own_gb = host_info.get("own_gb")
    if running and own_gb is None:
        own_gb = min(held_gb, ram_needed_gb or 0.0)
    own_gb = float(own_gb or 0.0) if running else 0.0
    others = [dict(o) for o in (host_info.get("others") or []) if o.get("slug") != model_info.get("slug")]
    itemised = float(sum(float(o.get("gb") or 0.0) for o in others))
    unattributed = max(0.0, held_gb - own_gb - itemised)
    others_gb = itemised + unattributed
    if unattributed > 1.0:
        # ยอดรวมจาก nvidia-smi ใหญ่กว่าที่จับคู่รายตัวได้ — ส่วนต่างต้องนับ ไม่ใช่หายไปเงียบ ๆ
        others.append({"slug": "other GPU processes", "gb": _r(unattributed)})
    others.sort(key=lambda o: -float(o.get("gb") or 0.0))

    free_now = host_info.get("free_gb")
    free_corrected = (float(free_now) + own_gb) if free_now is not None else None

    ram_after_gb = None
    fits = None
    if usable_gb is not None and ram_needed_gb is not None:
        ram_after_gb = usable_gb - others_gb - ram_needed_gb
        fits = ram_after_gb >= 0

    # ถ้าไม่ใส่: ตั้ง slots ได้มากสุดกี่ช่อง · หรือต้องหยุดใคร
    suggested_slots_max = None
    if vllm_like and usable_gb is not None and weights_gb is not None and kv_per_request_gb:
        room = usable_gb - others_gb - weights_gb - overhead_gb
        suggested_slots_max = max(0, int(room // (kv_per_request_gb * KV_PIN_HEADROOM)))
    suggested_context_max = None
    if not vllm_like and usable_gb is not None and weights_gb is not None and per_token:
        room = usable_gb - others_gb - weights_gb - overhead_gb
        suggested_context_max = max(0, int(room * GIB / per_token) // 4096 * 4096)
    stop_suggestion = others[0]["slug"] if (fits is False and others) else None

    # gpu-util ที่ "เทียบเท่า" — vLLM (V1) เช็คตอน start ว่า free ≥ gpu-util × total แม้ pin KV แล้ว
    # จึงตั้งให้พอดีกับที่ต้องใช้จริง (+2%) ไม่ใช่ 0.85 ซึ่งบนเครื่องที่มีโมเดลอื่นอยู่จะไม่ผ่านด่านนี้
    gpu_util_equivalent = None
    if total_gb and ram_needed_gb is not None:
        gpu_util_equivalent = min(0.98, max(0.3, math.ceil((ram_needed_gb / float(total_gb) + 0.02) * 100) / 100))

    # เหตุผลเป็นรหัส + ประโยคไทย (CLI) — หน้าเว็บเรียบเรียงอังกฤษจากรหัสเอง
    if ram_needed_gb is None or usable_gb is None:
        code = "unknown"
        why = "ยังคำนวณไม่ได้ — " + ("ไม่รู้ขนาด weights" if weights_gb is None else
                                  "ไม่รู้ KV ต่อ token (profile ไม่มี kv_bytes_per_token และยังไม่มี log ของ vLLM)"
                                  if kv_per_request_gb is None else "ไม่รู้ความจุของเครื่อง")
    elif fits and ram_after_gb is not None and ram_after_gb < TIGHT_MARGIN_GB:
        code = "tight"
        why = f"ใส่ได้แต่ตึง — เหลือ {ram_after_gb:.1f} GB หลังหัก OS {os_reserve:.0f} GB"
    elif fits:
        code = "fits"
        why = f"ใส่ได้ — เหลือ {ram_after_gb:.1f} GB หลังหัก OS {os_reserve:.0f} GB" + (
            f" และโมเดลอื่น {others_gb:.1f} GB" if others_gb > 0.05 else "")
    else:
        code = "no-fit"
        short = -ram_after_gb
        if suggested_slots_max and suggested_slots_max >= 1:
            why = (f"ไม่พอ — ขาด {short:.1f} GB · ลด slots เหลือ {suggested_slots_max} "
                   f"(KV ต่อคำขอ {kv_per_request_gb:.1f} GB × 1.2)")
        elif suggested_context_max and suggested_context_max >= 4096:
            why = f"ไม่พอ — ขาด {short:.1f} GB · ลด context เหลือ ≤ {suggested_context_max:,}"
        elif stop_suggestion:
            why = f"ไม่พอ — ขาด {short:.1f} GB แม้ slots 1 · หยุด {stop_suggestion} ({others[0]['gb']} GB) ก่อน"
        else:
            why = f"ไม่พอ — ขาด {short:.1f} GB แม้ slots 1 · โมเดลนี้ใหญ่เกินเครื่องนี้"
        if stop_suggestion and suggested_slots_max:
            why += f" · หรือหยุด {stop_suggestion} ({others[0]['gb']} GB)"

    if running and node_count == 1 and weights_source != "measured" and vllm_like:
        notes.append("รันอยู่แต่ยังไม่เจอบรรทัด 'Model loading took' ใน log — ใช้ขนาดไฟล์บนดิสก์แทน")
    if node_count > 1:
        notes.append(f"stacked {node_count} เครื่อง: ตารางนี้คิดยอดรวมของคลัสเตอร์ — pin ต่อ rank ยังต้องตั้งเอง")

    stack: list[dict] = []
    if total_gb:
        stack.append({"kind": "os", "label": "OS reserve", "gb": _r(os_reserve)})
        for o in others:
            stack.append({"kind": "other", "label": o.get("slug") or "other", "gb": _r(o.get("gb") or 0.0)})
        if weights_gb is not None:
            stack.append({"kind": "weights", "label": "weights", "gb": _r(weights_gb)})
            stack.append({"kind": "overhead", "label": "overhead", "gb": _r(overhead_gb)})
        if kv_gb is not None:
            stack.append({"kind": "kv", "label": "KV pin" if pin_supported else "KV", "gb": _r(kv_gb)})
        if ram_after_gb is not None:
            stack.append({"kind": "free" if ram_after_gb >= 0 else "over", "label": "free" if ram_after_gb >= 0 else "over",
                          "gb": _r(abs(ram_after_gb))})

    return {
        "slug": model_info.get("slug"),
        "engine": engine,
        "node_count": node_count,
        "pin_supported": pin_supported,
        "context": context,
        "slots": slots,
        "native_context": native,
        "kv_dtype": kv_dtype,
        "kv_per_token_bytes": int(per_token) if per_token else None,
        "kv_source": kv_source,
        "kv_per_request_gb": _r(kv_per_request_gb, 2),
        "kv_gb": _r(kv_gb),
        "kv_pin_gb": _r(kv_gb) if pin_supported and kv_gb is not None else None,
        "kv_pin_bytes": kv_pin_bytes,
        "current_pin_bytes": current_pin,
        "weights_gb": _r(weights_gb),
        "weights_source": weights_source,
        "overhead_gb": _r(overhead_gb),
        "ram_needed_gb": _r(ram_needed_gb),
        "total_gb": _r(total_gb),
        "os_reserve_gb": _r(os_reserve),
        "usable_gb": _r(usable_gb),
        "others_running_gb": _r(others_gb),
        "others": others,
        "own_gb_now": _r(own_gb) if running else None,
        "running": running,
        "free_now_gb": _r(free_now),
        "free_now_corrected_gb": _r(free_corrected),
        "ram_after_gb": _r(ram_after_gb),
        "fits": fits,
        "verdict": code,
        "reason": why,
        "suggested_slots_max": suggested_slots_max,
        "suggested_context_max": suggested_context_max,
        "stop_suggestion": stop_suggestion,
        "gpu_util_equivalent": gpu_util_equivalent,
        "per_slot_context": (context // slots) if (not vllm_like and context) else None,
        "measured": measured,
        "stack": stack,
        "notes": notes,
    }


def settings_for(plan: dict, extra_args: str | None) -> dict[str, str]:
    """ค่าที่ `lmds set` / หน้าเว็บต้องเขียน เพื่อให้ start ครั้งหน้าเป็นไปตามตาราง

    vLLM: slots · context · gpu_util (เทียบเท่า — ผ่านด่าน free ≥ util × total ของ vLLM) · extra_args ที่มี
    --kv-cache-memory ทับของเดิม · llama.cpp: slots · context เท่านั้น (--ctx-size คือ pool)
    """
    out = {"slots": str(plan["slots"]), "context": str(plan["context"])}
    if plan.get("pin_supported") and plan.get("kv_pin_bytes"):
        out["extra_args"] = with_kv_pin(extra_args, plan["kv_pin_bytes"])
        if plan.get("gpu_util_equivalent"):
            out["gpu_util"] = f"{plan['gpu_util_equivalent']:.2f}"
    return out


# ── ค่าตั้งต้นตอน deploy (unified / DGX Spark) ────────────────────────────────
def _flags_text(plan) -> str:
    return " ".join(str(f) for f in (plan.serving.extra_flags or []) if f)


def sizing_for_plan(fit, plan) -> dict | None:
    """ตาราง Fit ของ plan นี้บนเครื่องเป้าหมาย (preset ตอน deploy — เครื่องเปล่า ไม่มี free -g ให้ถาม)

    total = ความจุของ target แปลงเป็น GiB · pin ที่สูตร/ผู้ใช้ตั้งไว้แล้วใน extra_flags ถูกเคารพ (บันทึกเป็น
    source=explicit) · ถ้ายังไม่มี: pin = slots × KV เต็ม context × 1.2 · ใหญ่กว่าที่เหลือ → pin เท่าที่เหลือ (ปัดลง
    ครึ่ง GiB · capped=True) = RAM เท่า gpu-util เดิมแต่มีเลขให้ดู · None เมื่อประเมินไม่ได้ (ไม่รู้ KV/weights ·
    stacked · engine อื่น · unified เท่านั้น)
    """
    from lmds.hardware import MemoryModel

    if getattr(fit, "memory_model", None) is not MemoryModel.UNIFIED or getattr(fit, "node_count", 1) != 1:
        return None
    engine = getattr(getattr(plan, "runtime", None), "engine", None)
    if getattr(engine, "value", engine) != "vllm":
        return None
    serving = plan.serving
    if not fit.kv_bytes_per_token or fit.weights_gb is None:
        return None
    total_gib = (fit.capacity_gb or 0.0) * UNIFIED_DECIMAL_TO_GIB
    if total_gib <= 0:
        return None
    extra = _flags_text(plan)
    kv_dtype = getattr(serving, "kv_cache_dtype", "auto") or "auto"
    if "--kv-cache-dtype" not in extra and kv_dtype != "auto":
        extra = f"{extra} --kv-cache-dtype {kv_dtype}".strip()
    existing = parse_kv_pin(extra)
    # คิดตามสูตรก่อนเสมอ (ตัด pin ที่มีอยู่ออก) แล้วค่อยเทียบ — renderer เรียกซ้ำบน plan ที่ใส่ pin ไปแล้ว ต้องยังบอกได้ว่า
    # ค่านั้นคือของสูตร (source=sizing) หรือมีคนตั้งค่าอื่นไว้เอง เช่น recipe GLM (source=explicit)
    sized = plan_kv_pin(
        {"engine": "vllm", "weight_bytes": int(fit.weights_gb * GIB), "kv_bytes_per_token": fit.kv_bytes_per_token,
         "native_context": serving.context, "extra_args": strip_kv_pin(extra), "running": False, "node_count": 1},
        {"memory_model": "unified", "total_gb": total_gib, "held_gb": 0.0},
        slots=serving.max_num_seqs, context=serving.context,
    )
    if sized["kv_pin_bytes"] is None or sized["usable_gb"] is None:
        return None
    room_gb = sized["usable_gb"] - sized["weights_gb"] - sized["overhead_gb"]
    sized["source"] = "sizing"
    sized["capped"] = False
    if room_gb < VLLM_MIN_KV_GB and existing is None:
        return None  # ชิดจน pin ไม่ได้ — ปล่อยให้ analyzer/verdict เดิมพูดแทน
    if sized["kv_pin_gb"] > room_gb and room_gb >= VLLM_MIN_KV_GB:
        _set_pin(sized, math.floor(room_gb / PIN_STEP_GIB) * PIN_STEP_GIB)
        sized["capped"] = True
    if existing is not None and existing != sized["kv_pin_bytes"]:
        sized["source"] = "explicit"
        sized["capped"] = False
        _set_pin(sized, existing / GIB)
    sized["current_pin_bytes"] = existing
    # นิยามเดียวกับ "Maximum concurrency … Nx" ของ vLLM = pool ÷ KV ต่อคำขอ (ไม่มี 1.2 — นั่นคือเผื่อตอนตั้ง pin)
    sized["full_context_requests"] = (
        int(sized["kv_pin_gb"] // sized["kv_per_request_gb"]) if sized["kv_per_request_gb"] else None)
    return sized


def _set_pin(sized: dict, pin_gb: float) -> None:
    sized["kv_pin_gb"] = sized["kv_gb"] = round(pin_gb, 1)
    sized["kv_pin_bytes"] = int(pin_gb * GIB)
    sized["ram_needed_gb"] = round(sized["weights_gb"] + sized["overhead_gb"] + pin_gb, 1)
    sized["ram_after_gb"] = round(sized["usable_gb"] - sized["ram_needed_gb"], 1)
    sized["fits"] = sized["ram_after_gb"] >= 0


def apply_default_pin(plan, fit) -> dict | None:
    """pin KV ตั้งต้นสำหรับ bundle ใหม่บนเครื่อง unified — แทน gpu-util 0.85 ที่กิน 109 GB ไม่ว่าโมเดลจะเล็กแค่ไหน

    ใส่ --kv-cache-memory ลง plan.serving.extra_flags เมื่อยังไม่มี (สูตรอย่าง GLM ที่ตั้ง pin ไว้เองไม่ถูกแตะ) ·
    gpu-util ยังอยู่เป็น fallback เมื่อประเมินไม่ได้ · คืนตัวเลขที่ใช้หรือ None
    """
    if parse_kv_pin(_flags_text(plan)):
        return None
    sized = sizing_for_plan(fit, plan)
    if sized is None:
        return None
    if sized.get("full_context_requests") == 0 and sized.get("kv_per_token_bytes"):
        # pin ที่เหลือถือคำขอเต็ม context ไม่ได้แม้ตัวเดียว — vLLM ปฏิเสธตอน start ("max seq len is larger than the
        # maximum number of tokens that can be stored in KV cache") · gpu-util 0.85 เดิมก็ให้ KV น้อยกว่านี้อีก แค่ไม่มีใครเห็น
        # เลข · ลด context ให้เท่าที่ pin ถือได้ (ขั้น 4096) แล้วคิดใหม่
        from .analyzer import MIN_PRACTICAL_CONTEXT

        context_fit = int(sized["kv_pin_bytes"] / sized["kv_per_token_bytes"]) // 4096 * 4096
        if context_fit < MIN_PRACTICAL_CONTEXT:
            return None
        before = plan.serving.context
        plan.serving.context = context_fit
        sized = sizing_for_plan(fit, plan)
        if sized is None:
            plan.serving.context = before
            return None
        sized["context_reduced_from"] = before
    plan.serving.extra_flags = list(plan.serving.extra_flags) + [f"--kv-cache-memory {sized['kv_pin_bytes']}"]
    return sized


def sizing_record(sized: dict | None, plan=None) -> dict | None:
    """ก้อน `memory.sizing` ใน MODEL_PROFILE.yaml — ตัวเลขที่ตัดสิน pin ตั้งต้น ให้คนอ่านทีหลังรู้ที่มา"""
    if not sized:
        return None
    return {
        "rule": "RAM = weights + overhead + KV pin · pin = slots × KV(context) × 1.2 · usable = total − 12 GB OS",
        "source": sized.get("source") or "sizing",
        "kv_pin_bytes": sized.get("kv_pin_bytes"),
        "kv_pin_gb": sized.get("kv_pin_gb"),
        "kv_per_request_gb": sized.get("kv_per_request_gb"),
        "kv_per_token_bytes": sized.get("kv_per_token_bytes"),
        "context": sized.get("context"),
        "slots": sized.get("slots"),
        "full_context_requests": sized.get("full_context_requests"),
        "weights_gb": sized.get("weights_gb"),
        "overhead_gb": sized.get("overhead_gb"),
        "ram_needed_gb": sized.get("ram_needed_gb"),
        "usable_gb": sized.get("usable_gb"),
        "capped": bool(sized.get("capped")),
    }
