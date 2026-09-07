"""กล่องแชทแบบ operator — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน (tests/console_shell_dom.js)

เจ้าของ 2026-09-07: "ช่วยเพิ่มความสามารถของ AI assistant … ไม่แน่ใจว่าจะมีแค่ไว้ถามตอบเอง" — เทสตรงนี้ถามหน้าเว็บจริง ๆ ว่า:
การ์ด "ทำอะไรได้บ้าง" ขึ้นตอนกล่องว่าง · ชิปเติมช่องพิมพ์ (ไม่ส่งเอง) · แถบคำถามลัดตามการ์ดที่เพิ่งแตะ และ context {node, slug}
ติดไปกับคำถาม · หัวกล่องบอกว่าสมองคือโมเดลไหนในฟลีต · ปุ่ม "Use as assistant brain" บนการ์ด local/node ยิง POST ที่ถูก ·
ตั๋วลบถาวรเน้น "ยังไม่ทำ" และ "แก้เลย" ถาม confirm ก่อน · ขั้นที่ล้มโชว์สาเหตุจาก log · ชิป "ทำอะไรได้ต่อ" ใต้คำตอบล่าสุด
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

LOCAL = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [{ slug: "qwen3-8b", running: true, healthy: true, engine: "vllm", port: 8000,
      context: 32768, features: "tools", commands: ["start", "stop", "status", "logs", "test-text"], controller_exists: true, downloaded: true }] } ],
  localModels: [{ slug: "local-a", running: true, healthy: true, controller_exists: true, downloaded: true,
                  engine: "llamacpp", port: 8080, topology: "single", context: 8192, features: "text", model_id: "org/local-a" },
                { slug: "embed-x", running: true, healthy: true, controller_exists: true, downloaded: true,
                  engine: "llamacpp", port: 8090, topology: "single", context: 512, features: "embedding (mean)", model_id: "org/embed" }] };
H.fx = fx;
H.routes = H.defaultRoutes(fx).filter(([p]) => p !== "/api/assistant");
H.brain = { provider: "openai-compat", model: "nemotron-3", base_url: "http://192.168.10.30:8000/v1", from_fleet: { node: "spark-01", slug: "qwen3-8b" } };
H.routes.push(["/api/assistant", () => ({ available: true, brain: H.brain, capabilities: {
  probes: 21, actions: 25, modes: ["แก้เลย", "ทีละขั้น", "ยังไม่ทำ"],
  groups: [{ group: "ตรวจ", examples: ["ทำไม {slug} start ไม่ขึ้น", "{node} เป็นยังไงบ้าง"] },
           { group: "deploy", examples: ["deploy Qwen/Qwen3-8B-GGUF ไป {node}"] }] } })]);
H.routes.push(["/api/assistant/brain", (url, opts) => { const b = JSON.parse(opts.body);
  H.brain = { provider: "openai-compat", model: b.slug, base_url: "http://127.0.0.1:8080/v1", from_fleet: { node: b.node || "this", slug: b.slug } };
  return { ok: true, brain: H.brain }; }]);
H.routes.push(["/api/assistant/chat", () => ({})]);
H.routes.push([/^\\/api\\/assistant\\/ticket\\//, (url, opts) => ({ ticket: "t1", mode: JSON.parse(opts.body || "{}").mode, finished: true, steps: [], menu: [] })]);
"""


def test_capability_card_chips_quick_asks_and_the_context_travel_with_the_question(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        const fab = document.getElementById("chat-open");
        const shown = !fab.hidden;
        fab.click(); await H.tick();
        const brain = document.getElementById("chat-brain").textContent;
        const card = document.getElementById("chat-caps-card");
        const chips = [...card.querySelectorAll(".chat-chip")].map(c => c.textContent);
        card.querySelector(".chat-chip").click(); await H.tick();
        const filled = document.getElementById("chat-text").value;
        // แตะการ์ด local-a (Manage) → แถบคำถามลัดเปลี่ยนตาม และตัวอย่างเติมชื่อจริง
        document.querySelector('button[data-act="opts"][data-slug="local-a"]').click(); await H.tick(5);
        const quick = [...document.querySelectorAll("#chat-quick .chat-chip")].map(c => c.textContent);
        const quickHidden = document.getElementById("chat-quick").hidden;
        document.getElementById("chat-caps").click(); await H.tick();
        const refilled = [...document.querySelectorAll("#chat-caps-card .chat-chip")].map(c => c.textContent);
        document.querySelector("#chat-quick .chat-chip").click(); await H.tick();
        const asked = document.getElementById("chat-text").value;
        document.getElementById("chat-form").requestSubmit(); await H.tick(8);
        const sent = H.calls.filter(c => c.url === "/api/assistant/chat").map(c => JSON.parse(c.body));
        console.log(JSON.stringify({ shown, brain, chips, filled, quick, quickHidden, refilled, asked, sent }));
    """)
    assert out["shown"] is True
    assert out["brain"] == "🧠 spark-01 · qwen3-8b"
    assert out["chips"] == ["ทำไม <slug> start ไม่ขึ้น", "<เครื่อง> เป็นยังไงบ้าง", "deploy Qwen/Qwen3-8B-GGUF ไป <เครื่อง>"]
    assert out["filled"] == "ทำไม <slug> start ไม่ขึ้น", "ชิปเติมช่องพิมพ์ ไม่ส่งเอง"
    assert out["quick"][:2] == ["ทำไม local-a start ไม่ขึ้น", "Fit local-a ให้พอดี"] and out["quickHidden"] is False
    assert out["refilled"][0] == "ทำไม local-a start ไม่ขึ้น", "ตัวอย่างต้องเติมชื่อการ์ดที่เปิดอยู่"
    assert out["asked"] == "ทำไม local-a start ไม่ขึ้น"
    assert out["sent"] and out["sent"][0]["context"] == {"node": "", "slug": "local-a"}
    assert out["sent"][0]["messages"][-1]["content"] == "ทำไม local-a start ไม่ขึ้น"


def test_the_brain_button_sits_on_running_chat_models_only_and_updates_the_header(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        document.getElementById("chat-open").click(); await H.tick();
        document.querySelector('button[data-act="opts"][data-slug="local-a"]').click(); await H.tick(5);
        document.querySelector('button[data-act="opts"][data-slug="embed-x"]').click(); await H.tick(5);
        const localBtn = !!document.querySelector('#panel-local-a button[data-act="brain"][data-slug="local-a"]');
        const embedBtn = !!document.querySelector('#panel-embed-x button[data-act="brain"]');
        document.querySelector('#panel-local-a button[data-act="brain"]').click(); await H.tick(8);
        const posted = H.calls.filter(c => c.url === "/api/assistant/brain").map(c => JSON.parse(c.body));
        const header = document.getElementById("chat-brain").textContent;
        await H.go("#/node/spark-01");
        document.querySelector('button[data-nact="menu"][data-node="spark-01"][data-slug="qwen3-8b"]').click(); await H.tick(8);
        const nodeBtn = document.querySelector('button[data-nact="brain"][data-node="spark-01"][data-slug="qwen3-8b"]');
        nodeBtn.click(); await H.tick(8);
        const posted2 = H.calls.filter(c => c.url === "/api/assistant/brain").map(c => JSON.parse(c.body));
        console.log(JSON.stringify({ localBtn, embedBtn, posted, header, nodeBtn: !!nodeBtn, posted2,
          header2: document.getElementById("chat-brain").textContent }));
    """)
    assert out["localBtn"] is True and out["embedBtn"] is False, "embedding เสิร์ฟ chat ไม่ได้ ไม่ควรมีปุ่ม"
    assert out["posted"] == [{"node": "", "slug": "local-a"}]
    assert out["header"] == "🧠 this machine · local-a"
    assert out["nodeBtn"] is True
    assert out["posted2"][-1] == {"node": "spark-01", "slug": "qwen3-8b"}
    assert out["header2"] == "🧠 spark-01 · qwen3-8b"


def test_a_destructive_ticket_highlights_hold_and_asks_before_apply_and_a_failed_step_explains(tmp_path):
    (out,) = run_scenario(tmp_path, LOCAL, """
        document.getElementById("chat-open").click(); await H.tick();
        chatHistory = [
          { role: "user", content: "ลบ old ทิ้ง" },
          { role: "assistant", content: "เสนอให้ลบ", next: ["ถ้าขั้นไหนล้ม อธิบายสาเหตุจาก log ให้หน่อย", "old ตอนนี้เป็นยังไง"],
            plan: { ticket: "t1", why: "ผู้ใช้ขอ", mode: "", destructive: true, default_mode: "hold", finished: false, expired: false,
              menu: [{ mode: "apply", label: "แก้เลย", detail: "" }, { mode: "step", label: "ทีละขั้น", detail: "" }, { mode: "hold", label: "ยังไม่ทำ", detail: "" }],
              steps: [{ action: "model_start", title: "เปิดโมเดล", target: "spark-01", command: "./x start", impact: "ใช้ VRAM", done: true,
                        result: { ok: false, output: "start failed", explain: "สาเหตุจาก log: unknown model architecture: 'qwen4exp'" } },
                      { action: "remove_model", title: "ลบโมเดล", target: "spark-01", command: "lmds remove old --yes --keep-weights", impact: "ถาวร", done: false, result: null }] } },
        ];
        chatDraw(); await H.tick();
        const plan = document.querySelector(".chat-plan");
        const pick = plan.querySelector(".menu button.pick").textContent;
        const goCount = plan.querySelectorAll(".menu button.go").length;
        const explain = plan.querySelector(".explain").textContent;
        const marks = [...plan.querySelectorAll("li > b")].map(b => b.textContent.trim().slice(0, 1));
        const next = [...document.querySelectorAll(".chat-next .chat-chip")].map(c => c.textContent);
        H.confirmAnswer = false;
        plan.querySelector(".menu button").click(); await H.tick(5);   // แก้เลย
        const refused = { confirms: H.confirms.length, chose: H.calls.filter(c => c.url.includes("/ticket/t1/choose")).length };
        H.confirmAnswer = true;
        document.querySelector(".chat-plan .menu button").click(); await H.tick(8);
        const chose = H.calls.filter(c => c.url.includes("/ticket/t1/choose")).map(c => JSON.parse(c.body));
        document.querySelector(".chat-next .chat-chip").click(); await H.tick();
        console.log(JSON.stringify({ pick, goCount, explain, marks, next, refused, chose, filled: document.getElementById("chat-text").value }));
    """)
    assert out["pick"] == "ยังไม่ทำ" and out["goCount"] == 0, "งานลบถาวรต้องไม่มีปุ่มเขียวเชิญกด"
    assert "unknown model architecture: 'qwen4exp'" in out["explain"]
    assert out["marks"] == ["✗", "○"]
    assert out["next"] == ["ถ้าขั้นไหนล้ม อธิบายสาเหตุจาก log ให้หน่อย", "old ตอนนี้เป็นยังไง"]
    assert out["refused"] == {"confirms": 1, "chose": 0}, "ตอบ confirm ว่าไม่ = ไม่ยิงอะไร"
    assert out["chose"] == [{"mode": "apply", "confirm": True}]
    assert out["filled"] == "ถ้าขั้นไหนล้ม อธิบายสาเหตุจาก log ให้หน่อย"


def test_without_a_provider_the_brain_button_brings_the_assistant_to_life(tmp_path):
    """ยังไม่มีสมอง = กล่องแชทซ่อนอยู่ · กด 🧠 บนการ์ด → provider ถูกตั้ง → ปุ่มแชทโผล่และเปิดได้ทันที ไม่ต้อง reload"""
    (out,) = run_scenario(tmp_path, LOCAL + """
        H.avail = false;
        H.routes = H.routes.filter(([p]) => p !== "/api/assistant" && p !== "/api/assistant/brain");
        H.routes.push(["/api/assistant", () => H.avail
          ? ({ available: true, brain: H.brain, capabilities: { probes: 1, actions: 1, modes: [], groups: [] } })
          : ({ available: false, reason: "ยังไม่ได้ตั้ง LLM provider" })]);
        H.routes.push(["/api/assistant/brain", (url, opts) => { const b = JSON.parse(opts.body); H.avail = true;
          H.brain = { provider: "openai-compat", model: b.slug, base_url: "http://127.0.0.1:8080/v1", from_fleet: { node: "this", slug: b.slug } };
          return { ok: true, brain: H.brain }; }]);
    """, """
        const fab = document.getElementById("chat-open");
        const before = fab.hidden;
        document.querySelector('button[data-act="opts"][data-slug="local-a"]').click(); await H.tick(5);
        document.querySelector('#panel-local-a button[data-act="brain"]').click(); await H.tick(10);
        const after = fab.hidden;
        fab.click(); await H.tick();
        const panel = document.getElementById("chat-panel");
        console.log(JSON.stringify({ before, after, opened: !panel.hidden, header: document.getElementById("chat-brain").textContent }));
    """)
    assert out["before"] is True and out["after"] is False
    assert out["opened"] is True, "ปุ่มแชทต้องใช้งานได้ทันทีหลังตั้งสมอง ไม่ใช่โผล่มาแต่กดแล้วเงียบ"
    assert out["header"] == "🧠 this machine · local-a"
