"""แผง Follow ของหน้าเว็บ — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน (tests/console_shell_dom.js)

เจ้าของ 2026-09-07: "ส่วนของ log ทำให้เป็นการแสดง detail แบบเกือบ realtime ได้ไหม user จะได้ดูว่า error อะไรด้วย
แทนการกด 1 ครั้งแสดง 1 รอบ" — เทสตรงนี้ถามหน้าเว็บจริง ๆ ว่า: ปุ่ม Logs เดิมยังเป็น one-shot · มีปุ่ม ▶ Follow
ข้าง ๆ · กดแล้วเปิด EventSource ถูก URL · บรรทัด error เป็นสีแดง · Stop ปิดสายจริง · start/restart เปิดแผง
เองในโหมด boot ที่ทนสายจบได้จนคำสั่งจบ · เก็บไม่เกิน 3000 บรรทัด · การ์ด node มีปุ่มเดียวกันและกล่อง worker
"""

from __future__ import annotations

import json

from tests.test_console_shell import FLEET, run_scenario

LOCAL = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [{ slug: "qwen3-8b", running: true, healthy: true, engine: "llamacpp", port: 8080, topology: "stacked" }] } ],
  localModels: [{ slug: "local-a", running: true, healthy: true, controller_exists: true, downloaded: true,
                  engine: "llamacpp", port: 8080, topology: "single", context: 8192 }] };
H.fx = fx;
H.routes = H.defaultRoutes(fx);
H.routes.push(["/api/models/local-a/logs", () => ({ slug: "local-a", text: "old line 1\\nold line 2" })]);
"""

LINE = "s => JSON.stringify({ line: s, ts: 1 })"


def test_local_logs_panel_keeps_the_one_shot_and_adds_a_follow_toggle(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, f"""
        const enc = {LINE};
        document.querySelector('button[data-act="logs"][data-slug="local-a"]').click(); await H.tick();
        const panel = document.getElementById("panel-local-a");
        const oneShot = {{ pre: panel.querySelector("pre").textContent, follow: !!panel.querySelector(".ll-follow"),
                          once: !!panel.querySelector(".ll-once"), streams: H.streams.length }};
        panel.querySelector(".ll-follow").click(); await H.tick();
        const s = H.streams[H.streams.length - 1];
        const status0 = panel.querySelector(".ll-status").textContent;
        s.emit("open");
        s.emit("message", {{ data: enc("INFO all good") }});
        s.emit("message", {{ data: enc("ERROR CUDA out of memory") }});
        s.emit("message", {{ data: enc("load failed: unknown model architecture 'qwen4exp'") }});
        s.emit("message", {{ data: enc("this line has no problem") }});
        const lines = [...panel.querySelectorAll(".ll-line")].map(el => ({{ text: el.textContent, err: el.classList.contains("err") }}));
        const live = {{ status: panel.querySelector(".ll-status").textContent, toggle: panel.querySelector(".ll-toggle").textContent,
                       dot: panel.querySelector(".ll-dot").className, worker: !!panel.querySelector(".ll-worker") }};
        panel.querySelector(".ll-toggle").click(); await H.tick();
        const stopped = {{ closed: s.closed, toggle: panel.querySelector(".ll-toggle").textContent,
                          status: panel.querySelector(".ll-status").textContent, kept: panel.querySelectorAll(".ll-line").length }};
        // ▶ Follow อีกครั้ง = สายใหม่ · Refresh = กลับไป one-shot และปิดสาย
        panel.querySelector(".ll-toggle").click(); await H.tick();
        const s2 = H.streams[H.streams.length - 1];
        panel.querySelector(".ll-once").click(); await H.tick();
        const back = {{ closed: s2.closed, pre: (panel.querySelector("pre") || {{}}).textContent, follow: !!panel.querySelector(".ll-follow") }};
        console.log(JSON.stringify({{ oneShot, url: s.url, status0, lines, live, stopped, back,
          isErr: [isLogError("Traceback (most recent call last):"), isLogError("INFO errorless run"),
                  isLogError("error loading model: x"), isLogError("0 failed"), isLogError("Error while parsing")] }}));
    """)
    assert out["oneShot"] == {"pre": "old line 1\nold line 2", "follow": True, "once": True, "streams": 1}
    assert out["url"] == "/api/models/local-a/logs/stream?tail=200&worker=0"
    assert out["status0"] == "connecting…"
    assert out["lines"] == [
        {"text": "INFO all good", "err": False},
        {"text": "ERROR CUDA out of memory", "err": True},
        {"text": "load failed: unknown model architecture 'qwen4exp'", "err": True},
        {"text": "this line has no problem", "err": False},
    ]
    assert out["live"] == {"status": "live", "toggle": "■ Stop", "dot": "dot ll-dot on", "worker": False}
    assert out["stopped"] == {"closed": True, "toggle": "▶ Follow", "status": "stopped", "kept": 4}, \
        "Stop ต้องปิดสายจริงแต่เก็บบรรทัดไว้ให้อ่านต่อ"
    assert out["back"] == {"closed": True, "pre": "old line 1\nold line 2", "follow": True}
    # ERROR แบบตัวใหญ่ + Traceback · วลีเฉพาะไม่สนตัวพิมพ์ · "errorless" ไม่ใช่ error · "Error while" ไม่เข้าข่าย (ไม่ใช่คำที่ระบุ)
    assert out["isErr"] == [True, False, True, True, False]


def test_restart_opens_follow_in_boot_mode_and_survives_until_the_command_returns(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL + """
        H.release = null;   // prelude กับ body เป็นคนละ scope — ฝากไว้กับ H
        H.routes.push(["/api/models/local-a/restart", () => new Promise(r => { H.release = () => r({ ok: true }); })]);
    """, f"""
        const enc = {LINE};
        document.querySelector('button[data-act="restart"][data-slug="local-a"]').click(); await H.tick();
        const panel = document.getElementById("panel-local-a");
        const s = H.streams[H.streams.length - 1];
        const boot = {{ url: s.url, status: panel.querySelector(".ll-status").textContent, streams: H.streams.length }};
        // ยังไม่มี container: hub ส่งบรรทัดเดียวแล้วจบ — โหมด boot ต้องไม่ปิดสาย (EventSource ต่อใหม่เอง)
        s.emit("open"); s.emit("message", {{ data: enc("ยังไม่มี container lmds-local-a — โมเดลยังไม่ได้ start") }});
        s.emit("end", {{ data: '{{"exit":0}}' }}); s.emit("error");
        const during = {{ closed: s.closed, status: panel.querySelector(".ll-status").textContent }};
        // ต่อใหม่ (เบราว์เซอร์ทำเอง) → open ล้างของเก่า → log จริงไหลมา
        s.emit("open");
        for (let i = 0; i < 3500; i++) s.emit("message", {{ data: enc("line " + i) }});
        const buffer = {{ count: panel.querySelectorAll(".ll-line").length, first: panel.querySelector(".ll-line").textContent }};
        H.release(); await H.tick();
        // คำสั่งจบแล้ว: สายจบขณะโมเดลยังรัน = ต่อใหม่ครั้งเดียว · จบอีก = หยุด
        s.emit("end", {{ data: '{{"exit":0}}' }}); s.emit("error");
        const once = {{ closed: s.closed, status: panel.querySelector(".ll-status").textContent }};
        s.emit("open"); s.emit("end", {{ data: '{{"exit":0}}' }});
        const done = {{ closed: s.closed, status: panel.querySelector(".ll-status").textContent, toggle: panel.querySelector(".ll-toggle").textContent }};
        console.log(JSON.stringify({{ boot, during, buffer, once, done }}));
    """)
    assert out["boot"]["url"] == "/api/models/local-a/logs/stream?tail=200&worker=0"
    assert out["boot"]["status"] == "waiting for the server to start…" and out["boot"]["streams"] == 2
    assert out["during"] == {"closed": False, "status": "waiting for the server to start…"}
    assert out["buffer"] == {"count": 3000, "first": "line 500"}, "เก็บได้ 3000 บรรทัด ตัดหัวทิ้ง"
    assert out["once"] == {"closed": False, "status": "stream ended — reconnecting once…"}
    assert out["done"] == {"closed": True, "status": "stream ended (exit 0)", "toggle": "▶ Follow"}


def test_closing_the_logs_panel_closes_the_stream(tmp_path):
    """ปิดแผงด้วยปุ่ม Logs (toggle) ต้องปิดสาย — ไม่งั้น hub ถือ docker logs -f ค้างโดยไม่มีใครดู"""
    (out,) = run_scenario(tmp_path, LOCAL, """
        document.querySelector('button[data-act="logs"][data-slug="local-a"]').click(); await H.tick();
        document.querySelector("#panel-local-a .ll-follow").click(); await H.tick();
        const s = H.streams[H.streams.length - 1];
        document.querySelector('button[data-act="logs"][data-slug="local-a"]').click(); await H.tick();
        console.log(JSON.stringify({ closed: s.closed, empty: document.getElementById("panel-local-a").innerHTML === "" }));
    """)
    assert out == {"closed": True, "empty": True}


def test_node_card_follow_button_worker_checkbox_and_close(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, f"""
        const enc = {LINE};
        // หน้าเครื่อง + กางเมนูของโมเดลก่อน (เหมือนผู้ใช้จริง) — ปุ่มจัดการอยู่ในเมนูนั้น
        await H.go("#/nodes");
        document.querySelector('button[data-nact="menu"][data-node="spark-01"][data-slug="qwen3-8b"]').click(); await H.tick();
        const btn = document.querySelector('button[data-nact="follow"][data-node="spark-01"][data-slug="qwen3-8b"]');
        H.assert(btn, "การ์ด node ต้องมีปุ่ม follow ข้างปุ่ม logs");
        btn.click(); await H.tick();
        const box = document.querySelector('#nodes .machine .nlive');
        const s = H.streams[H.streams.length - 1];
        s.emit("open"); s.emit("message", {{ data: enc("ERROR worker died") }});
        const opened = {{ url: s.url, visible: H.visible(box), hidden: box.hidden,
                         lines: [...box.querySelectorAll(".ll-line")].map(el => el.textContent + (el.classList.contains("err") ? "!" : "")),
                         worker: !!box.querySelector(".ll-worker") }};
        // stacked: ติ๊ก worker = สายใหม่ไปที่ worker=1 · สายเก่าปิด
        const cb = box.querySelector(".ll-worker");
        cb.checked = true; cb.dispatchEvent(new Event("change", {{ bubbles: true }})); await H.tick();
        const s2 = H.streams[H.streams.length - 1];
        const switched = {{ oldClosed: s.closed, url: s2.url }};
        box.querySelector(".ll-close").click(); await H.tick();
        console.log(JSON.stringify({{ opened, switched, closed: s2.closed, hidden: box.hidden, empty: box.innerHTML === "" }}));
    """)
    assert out["opened"]["url"] == "/api/nodes/spark-01/models/qwen3-8b/logs/stream?tail=200&worker=0"
    assert out["opened"]["visible"] is True and out["opened"]["hidden"] is False
    assert out["opened"]["lines"] == ["ERROR worker died!"] and out["opened"]["worker"] is True
    assert out["switched"] == {"oldClosed": True, "url": "/api/nodes/spark-01/models/qwen3-8b/logs/stream?tail=200&worker=1"}
    assert out["closed"] is True and out["hidden"] is True and out["empty"] is True


def test_node_start_opens_the_follow_box_in_boot_mode(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL + """
        H.routes.push(["/api/nodes/spark-01/models/qwen3-8b/restart", () => ({ job: { id: "j1", running: true, command: "restart", output: "" } })]);
        H.routes.push(["/api/jobs/j1", () => ({ id: "j1", running: false, exit_code: 0, command: "restart", output: "started" })]);
    """, """
        await H.go("#/nodes");
        document.querySelector('button[data-nact="menu"][data-node="spark-01"][data-slug="qwen3-8b"]').click(); await H.tick();
        document.querySelector('button[data-nact="model:restart"][data-node="spark-01"][data-slug="qwen3-8b"]').click(); await H.tick();
        const box = document.querySelector('#nodes .machine .nlive');
        const s = [...H.streams].reverse().find(x => x.url.includes("/logs/stream"));
        console.log(JSON.stringify({ url: s && s.url, visible: H.visible(box), status: box.querySelector(".ll-status").textContent }));
    """)
    assert out["url"] == "/api/nodes/spark-01/models/qwen3-8b/logs/stream?tail=200&worker=0"
    assert out["visible"] is True and out["status"] == "waiting for the server to start…"


def test_page_still_parses_and_boots_with_one_events_stream(tmp_path):
    """แผง Follow เปิดตามผู้ใช้เท่านั้น — ตอนบูตต้องมี EventSource เดียวคือ /api/events เหมือนเดิม"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        console.log(JSON.stringify({ streams: H.streams.map(s => s.url), errors: H.errors, live: liveLogs.size }));
    """)
    assert out["streams"] == ["/api/events"] and out["errors"] == [] and out["live"] == 0
