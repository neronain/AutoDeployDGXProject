"""ปุ่ม Fit + ตาราง/แถบซ้อนบนการ์ดโมเดล — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน (tests/console_shell_dom.js)

เจ้าของ 2026-09-07: การ์ด vLLM เคยขึ้น "❌ Cannot start now — 12.9 GB free but this model needs at least 79.3 GB" ให้โมเดล
ที่ *รันอยู่แล้ว* (นับหน่วยความจำของตัวมันเองเป็นไม่ว่าง) และเครื่องคิดจาก gpu-util โดยไม่รู้จัก --kv-cache-memory · ตอนนี้
hub คืนตาราง Fit (weights · overhead · KV ต่อคำขอ · pin · โมเดลอื่น · เหลือ) + แถบทั้งเครื่อง · ปุ่ม Fit เขียนลง bundle
แล้วเสนอ restart · ช่อง gpu-util ถูกปิดเมื่อ pin แล้ว · เทสถามหน้าเว็บจริง ๆ ทั้งการ์ดในเครื่องและการ์ดของ node
"""

from __future__ import annotations

import json

from tests.test_console_shell import run_scenario

# ตารางที่ hub คืน — ตัวเลขจริงของ Nemotron บน spark-head 2026-09-07 (ดู tests/test_kv_sizing.py)
PLAN = {
    "slug": "nemo", "engine": "vllm", "pin_supported": True, "context": 262144, "slots": 2, "kv_dtype": "fp8",
    "kv_per_token_bytes": 5461, "kv_source": "measured", "kv_per_request_gb": 1.33, "kv_gb": 3.5, "kv_pin_gb": 3.5,
    "kv_pin_bytes": 3758096384, "current_pin_bytes": 6442450944, "weights_gb": 69.6, "weights_source": "measured",
    "overhead_gb": 3.0, "ram_needed_gb": 76.1, "total_gb": 121.0, "os_reserve_gb": 12.0, "usable_gb": 109.0,
    "others_running_gb": 19.5, "others": [{"slug": "gemma4", "gb": 19.5}], "own_gb_now": 78.5, "running": True,
    "free_now_gb": 4.0, "free_now_corrected_gb": 82.5, "ram_after_gb": 13.4, "fits": True, "verdict": "fits",
    "reason": "ใส่ได้", "suggested_slots_max": 10, "gpu_util_equivalent": 0.65, "notes": [],
    "stack": [{"kind": "os", "label": "OS reserve", "gb": 12.0}, {"kind": "other", "label": "gemma4", "gb": 19.5},
              {"kind": "weights", "label": "weights", "gb": 69.6}, {"kind": "overhead", "label": "overhead", "gb": 3.0},
              {"kind": "kv", "label": "KV pin", "gb": 3.5}, {"kind": "free", "label": "free", "gb": 13.4}],
    "settings": {"slots": "2", "context": "262144", "extra_args": "--kv-cache-dtype fp8 --kv-cache-memory 3758096384", "gpu_util": "0.65"},
}

LOCAL = f"""const fx = {{ nodes: [
  {{ name: "spark-01", site: "TKC", models: [{{ slug: "nemo", running: true, healthy: true, engine: "vllm", port: 8000, topology: "single",
      context: 262144, context_configured: 262144, max_num_seqs: 2, native_context: 262144, commands: ["start", "restart", "logs"] }}] }} ],
  localModels: [{{ slug: "nemo", running: true, healthy: true, controller_exists: true, downloaded: true,
                  engine: "vllm", port: 8000, topology: "single", context: 262144, max_num_seqs: 2, native_context: 262144 }}] }};
H.fx = fx;
H.routes = H.defaultRoutes(fx);
H.plan = {json.dumps(PLAN, ensure_ascii=False)};
const fitRoute = (url, opts) => {{
  const body = JSON.parse(opts.body || "{{}}");
  if (body.slots === 40) return {{ status: 409, body: {{ detail: "ไม่พอ — ขาด 50 GB · ลด slots เหลือ 10",
    plan: {{ ...H.plan, slots: 40, fits: false, verdict: "no-fit", ram_after_gb: -50.0, kv_pin_gb: 64.0, stop_suggestion: "gemma4",
             stack: [...H.plan.stack.slice(0, 5), {{ kind: "over", label: "over", gb: 50.0 }}] }} }} }};
  return {{ slug: "nemo", plan: {{ ...H.plan, slots: body.slots || 2 }}, applied: !!body.apply,
           saved: body.apply ? H.plan.settings : undefined, restart_needed: !!body.apply }};
}};
H.routes.push(["/api/models/nemo/fit", fitRoute]);
H.routes.push(["/api/nodes/spark-01/models/nemo/fit", fitRoute]);
H.routes.push(["/api/models/nemo/restart", () => ({{ ok: true }})]);
H.routes.push(["/api/nodes/spark-01/models/nemo/restart", () => ({{ ok: true, output: "" }})]);
"""


def _fit_calls(calls, url):
    return [json.loads(c["body"]) for c in calls if c["url"] == url and c["method"] == "POST"]


def test_local_card_shows_the_fit_table_and_no_false_cannot_start(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        document.querySelector('button[data-act="opts"][data-slug="nemo"]').click(); await H.tick(5);
        const panel = document.getElementById("panel-nemo");
        const table = panel.querySelector(".fit-table");
        const text = panel.textContent;
        const segs = [...panel.querySelectorAll(".fit-stack .fit-seg")].map(s => s.className.replace("fit-seg ", ""));
        const first = { table: !!table, segs, ok: (panel.querySelector(".ok-line") || {}).textContent || "",
                        cannot: /Cannot start now/.test(text), pin: /--kv-cache-memory 3758096384/.test(text),
                        other: /gemma4/.test(text), own: /holds now/.test(text), util: /0\\.65/.test(text),
                        preview: H.calls.filter(c => c.url === "/api/models/nemo/fit").length };
        // แก้ slots → ตารางถูกถามใหม่ (debounce) ด้วยค่าที่พิมพ์
        const slots = panel.querySelector(".o-slots"); slots.value = "3"; slots.dispatchEvent(new Event("input", { bubbles: true }));
        await H.sleep(900); await H.tick(4);
        const bodiesFn = () => H.calls.filter(c => c.url === "/api/models/nemo/fit").map(c => JSON.parse(c.body));
        const previews = bodiesFn();
        // กด Fit → apply:true → บันทึก → ถาม restart (ตอบไม่ก่อน เพื่ออ่านข้อความที่บันทึก — restart วาดการ์ดใหม่)
        H.confirmAnswer = false;
        panel.querySelector('button[data-act="fit"][data-slug="nemo"]').click(); await H.tick(8);
        const lines = [...panel.querySelectorAll(".fit-out .ok-line, .fit-out .warn-line")].map(el => el.textContent);
        const saved = lines[lines.length - 1] || "";
        const noRestart = H.calls.some(c => c.url === "/api/models/nemo/restart");
        // ตอบใช่ → ปุ่ม restart ของการ์ดถูกกดให้
        H.confirmAnswer = true;
        panel.querySelector('button[data-act="fit"][data-slug="nemo"]').click(); await H.tick(8);
        console.log(JSON.stringify({ first, previews, bodies: bodiesFn(), saved, lines, confirms: H.confirms, noRestart,
          restart: H.calls.some(c => c.url === "/api/models/nemo/restart" && c.method === "POST") }));
    """)
    f = out["first"]
    assert f["table"] and f["segs"] == ["os", "other", "weights", "overhead", "kv", "free"]
    assert f["ok"].startswith("✅ Fits") and "19.5 GB held by other models" in f["ok"]
    assert f["cannot"] is False, "โมเดลที่รันอยู่ต้องไม่โดน Cannot start now"
    assert f["pin"] and f["other"] and f["own"] and f["util"] and f["preview"] == 1
    assert out["previews"][0] == {"slots": 2, "context": 262144, "apply": False}
    assert out["previews"][-1] == {"slots": 3, "context": 262144, "apply": False}
    assert "Saved to the bundle" in out["saved"] and "--kv-cache-memory 3758096384" in out["saved"]
    assert [b for b in out["bodies"] if b["apply"]] == [{"slots": 3, "context": 262144, "apply": True}] * 2
    assert len(out["confirms"]) == 2 and "Restart nemo" in out["confirms"][0]
    assert out["noRestart"] is False and out["restart"] is True


def test_local_fit_that_does_not_fit_writes_nothing_and_shows_why(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        document.querySelector('button[data-act="opts"][data-slug="nemo"]').click(); await H.tick(5);
        const panel = document.getElementById("panel-nemo");
        panel.querySelector(".o-slots").value = "40";
        panel.querySelector('button[data-act="fit"][data-slug="nemo"]').click(); await H.tick(8);
        const warns = [...panel.querySelectorAll(".fit-out .warn-line")].map(w => w.textContent);
        console.log(JSON.stringify({ warns, over: !!panel.querySelector(".fit-seg.over"), confirms: H.confirms.length,
          restart: H.calls.some(c => c.url === "/api/models/nemo/restart") }));
    """)
    assert any("Does not fit" in w and "short by 50.0 GB" in w and "lower slots to 10" in w for w in out["warns"])
    assert any(w.startswith("Not written") and "ลด slots เหลือ 10" in w for w in out["warns"])
    assert out["over"] and out["confirms"] == 0 and out["restart"] is False


def test_node_card_uses_the_same_table_and_disables_gpu_util_when_pinned(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        await H.go("#/node/spark-01");
        document.querySelector('button[data-nact="menu"][data-node="spark-01"][data-slug="nemo"]').click(); await H.tick(8);
        const box = nodeRows.get("spark-01").body.querySelector('.submenu[data-for="nemo"]');
        const gpu = box.querySelector(".n-gpu");
        const before = { table: !!box.querySelector(".n-mem .fit-table"), segs: box.querySelectorAll(".n-mem .fit-seg").length,
                         cannot: /Cannot start now/.test(box.textContent), gpuDisabled: gpu.disabled, gpuHint: gpu.placeholder,
                         fitBtn: !!box.querySelector('button[data-nact="fit"][data-slug="nemo"]') };
        box.querySelector('button[data-nact="fit"][data-slug="nemo"]').click(); await H.tick(8);
        const bodies = H.calls.filter(c => c.url === "/api/nodes/spark-01/models/nemo/fit").map(c => JSON.parse(c.body));
        console.log(JSON.stringify({ before, bodies, confirms: H.confirms.length,
          restart: H.calls.some(c => c.url === "/api/nodes/spark-01/models/nemo/restart" && c.method === "POST") }));
    """)
    b = out["before"]
    assert b["table"] and b["segs"] == 6 and b["cannot"] is False and b["fitBtn"]
    assert b["gpuDisabled"] is True and b["gpuHint"] == "not used (KV pinned)"
    assert out["bodies"][0]["apply"] is False and out["bodies"][-1]["apply"] is True
    assert out["confirms"] == 1 and out["restart"] is True
