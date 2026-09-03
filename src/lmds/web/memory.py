"""ตัวเลขสำหรับบรรทัด "ต้องใช้แรมเท่าไร" ใต้ฟอร์ม settings — คำนวณสดในเบราว์เซอร์ตอนพิมพ์

ผู้ใช้ 2026-09-04: "หลังกรอกแล้ว จะมีให้ดูว่าต้องการแรมเท่าไร เพราะถ้าใส่ gpu-util น้อยกว่าที่
ต้องการ ระบบจะทำงานไม่ดีไหม" — ใช่: vLLM จอง gpu-util × ทั้งเครื่อง แล้วเอา weights ออก ที่เหลือ
คือ KV · เหลือไม่พอ 1 คำขอที่ context ที่ตั้ง = ไม่ start เลย · และถ้าจองมากกว่าที่ว่างอยู่ก็ไม่ start
เหมือนกัน · เดิมรู้ตอนกด start แล้วพังเท่านั้น

hub ส่งแค่ *ข้อเท็จจริง* (weights, KV ต่อ token, ความจุ, ที่ว่างตอนนี้, overhead) ส่วนเลขคณิต
ทำที่หน้าเว็บเพื่อให้ขยับตามทุกตัวอักษรที่พิมพ์โดยไม่ยิง request
"""

from __future__ import annotations

from lmds.fit.analyzer import GIB, LLAMACPP_OVERHEAD_GB, VLLM_OVERHEAD_GB_PER_GPU, UNIFIED_OS_RESERVE_GB

# KV ต่อ token ต่อโมเดล — bundle เก่าไม่มีค่านี้ใน MODEL_PROFILE ต้องถาม Hub ครั้งเดียวแล้วจำ
_KV_CACHE: dict[str, int | None] = {}


def kv_bytes_from_hub(model_id: str, revision: str | None = None) -> tuple[int | None, str]:
    """(bytes/token, note) — อ่าน config จาก Hugging Face · ล้มเหลว = (None, เหตุผล) ไม่โยน"""
    key = f"{model_id}@{revision or ''}"
    if key in _KV_CACHE:
        return _KV_CACHE[key], "hub (cached)"
    try:
        from dataclasses import replace

        from lmds.inspector import HfClient, inspect_model
        from lmds.resolver import parse_source
        from lmds.secrets import get_secret

        source = parse_source(model_id)
        if revision:
            source = replace(source, revision=revision)
        report = inspect_model(source, HfClient(token=get_secret("hf") or None))
        value = report.kv_dims.bytes_per_token_fp16 if report.kv_dims else None
        _KV_CACHE[key] = value
        return value, "hub" if value else "โมเดลไม่เปิดเผยมิติ KV"
    except Exception as exc:  # noqa: BLE001 — แค่ทำให้บรรทัดคำนวณหาย ไม่ใช่ทำแผงพัง
        return None, f"ถาม Hugging Face ไม่ได้: {str(exc)[:120]}"


def memory_facts(profile: dict | None, host: dict | None, model_entry: dict | None = None) -> dict:
    """ข้อเท็จจริงที่หน้าเว็บต้องใช้ — ทุกค่าเป็น None ได้ พร้อม note ว่าทำไม"""
    profile = profile or {}
    host = host or {}
    model = profile.get("model") or {}
    serving = profile.get("serving") or {}
    entry = model_entry or {}

    engine = (profile.get("runtime") or {}).get("engine") or entry.get("engine") or ""
    gpus = host.get("gpus") or []
    gpu_count = max(1, len(gpus))
    capacity = float(sum((g.get("vram_gb") or 0.0) for g in gpus)) or None
    used = float(sum((g.get("vram_used_gb") or 0.0) for g in gpus))
    unified = host.get("memory_model") == "unified"

    # "ว่างตอนนี้" ที่ vLLM จะเห็นตอน start (cudaMemGetInfo) — บนเครื่อง unified คือแรมว่างของทั้งระบบ
    # ไม่ใช่ความจุ - ของที่โมเดลถือ เพราะ OS/บริการอื่นกินจาก pool เดียวกัน
    free_now = None
    if unified and host.get("ram_total_gb") and host.get("ram_used_gb") is not None:
        free_now = round(float(host["ram_total_gb"]) - float(host["ram_used_gb"]), 1)
    elif capacity:
        free_now = round(capacity - used - (UNIFIED_OS_RESERVE_GB if unified else 0.0), 1)

    weight_bytes = model.get("weight_bytes")
    weights_gb = round(weight_bytes / GIB, 1) if weight_bytes else None

    kv = model.get("kv_bytes_per_token")
    kv_source = "profile" if kv else None
    notes: list[str] = []
    model_id = model.get("id") or entry.get("model_id") or ""
    if not kv and model_id:
        kv, why = kv_bytes_from_hub(model_id, model.get("revision"))
        kv_source = why if kv else None
        if not kv:
            notes.append(f"ไม่รู้ KV ต่อ token ({why}) — คำนวณได้แค่ weights")
    if weights_gb is None:
        notes.append("ไม่รู้ขนาด weights — MODEL_PROFILE ไม่มี weight_bytes")

    overhead = (VLLM_OVERHEAD_GB_PER_GPU if engine == "vllm" else LLAMACPP_OVERHEAD_GB) * gpu_count
    return {
        "engine": engine,
        "memory_model": host.get("memory_model"),
        "capacity_gb": capacity,
        "vram_used_gb": round(used, 1),
        "free_gb_now": free_now,
        "weights_gb": weights_gb,
        "kv_bytes_per_token": kv,
        "kv_source": kv_source,
        "native_context": model.get("native_context"),
        "overhead_gb": round(overhead, 1),
        "defaults": {
            "gpu_util": serving.get("gpu_memory_utilization"),
            "max_num_seqs": serving.get("max_num_seqs") or entry.get("max_num_seqs"),
            "context": serving.get("context") or entry.get("context_configured") or entry.get("context"),
        },
        "notes": notes,
    }


def read_node_profile(node, slug: str) -> dict:
    """MODEL_PROFILE.yaml ของ bundle บนเครื่องอื่น — ยิง SSH หนึ่งครั้งตอนเปิดแผง (ไฟล์ไม่กี่ KB)"""
    import shlex

    import yaml

    from lmds.nodes import NodeError, run

    quoted = shlex.quote(slug)
    script = (f"dir=\"$(ls -d ~/bundles/{quoted} ~/*/bundles/{quoted} 2>/dev/null | head -1)\"; "
              f"[ -n \"$dir\" ] || {{ echo 'ไม่พบ bundle {slug}' >&2; exit 1; }}; "
              f"cat \"$dir/MODEL_PROFILE.yaml\"")
    result = run(node, script, timeout=25)
    if not result.ok:
        raise NodeError((result.stderr or result.stdout or "อ่าน MODEL_PROFILE ไม่ได้").strip()[:200])
    data = yaml.safe_load(result.stdout) or {}
    return data if isinstance(data, dict) else {}
