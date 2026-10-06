"""audit หน้าคอนโซล 2026-10 — ทุกข้อเป็นเทสที่ล้มก่อนแก้ · รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน

แต่ละเทสคือ scenario ที่ผู้ตรวจใช้ยืนยันบั๊ก (payload รูปเดียวกับที่ hub ส่งจริง) แล้วถามที่ **ผลบนจอ**:
ข้อความที่วาด · ปุ่มที่กดได้ · คำขอที่ออกไป — ไม่ใช่สตริงในซอร์ส · เวลาของหน้าเว็บถูกเร่งด้วย
`H.fastTimers()` (วง poll 1.2 วิ / 5 วิ และ backoff ของมันจึงทดสอบได้ในเสี้ยววินาที) ส่วน `H.sleep` เป็นเวลาจริง
"""

from __future__ import annotations

from pathlib import Path

from tests.test_console_shell import run_scenario

MODEL = """const model = (o) => Object.assign({ slug: "qwen", model_id: "Q/q", engine: "llamacpp", port: 8080, context: 32768,
  running: false, healthy: false, downloaded: true, controller_exists: true, commands: ["status"] }, o);
"""

# hub ที่ต้องใช้ token · ฝั่ง server จำลองตาม require_token: token ไม่ตรง = 401 และนับไว้ให้เทสถาม
GUARDED = """H.fastTimers(50);
H.serverToken = "old"; localStorage.setItem("lmds:token", "old");
H.refused = []; H.authPosts = 0;
const guard = (url, opts) => {
  const sent = ((opts.headers || {})["x-lmds-token"]) || (new URL("http://x" + url).searchParams.get("token") || "");
  if (sent === H.serverToken) return null;
  H.refused.push(url);
  return { status: 401, body: { detail: "token ไม่ถูกต้อง" } };
};
const guarded = routes => routes.filter(r => r[0] !== "/api/auth").map(([pat, h]) =>
  [pat, (url, opts) => guard(url, opts) || (typeof h === "function" ? h(url, opts) : h)]);
const authRoute = ["/api/auth", (url, opts) => {
  if ((opts.method || "GET") !== "POST") return { required: true };
  H.authPosts++;
  return guard(url, opts) || { ok: true };
}];
"""


# ───────────────────── ข้อ 2 — token ถูกเปลี่ยนที่ hub ระหว่างเปิดหน้าค้างไว้ ─────────────────────

def test_a_rotated_token_stops_every_poller_before_the_login_screen_and_keeps_what_is_typed(tmp_path):
    """เดิม: หน้า login ถูกวาดทับทุก 5 วิ (ช่องที่กำลังพิมพ์หาย) และวง poll + ตัวตาม job ยื่น token เก่าให้ hub
    ไปเรื่อย ๆ — แต่ละครั้งถูกนับเป็นการเดาผิด จน IP โดนล็อก แล้ว token ที่ถูกต้องก็เข้าไม่ได้ (429)"""
    (boot, held, done) = run_scenario(tmp_path, MODEL + GUARDED + """
        const fx = { nodes: [{ name: "spark-01", site: "TKC" }, { name: "spark-02", site: "TKC" }],
                     localModels: [model({ slug: "local-m", job: { id: "j1", command: "download" } })] };
        H.fx = fx;
        H.routes = [authRoute, ...guarded([
          ["/api/jobs/j1", () => ({ id: "j1", command: "download", running: true, output: "1/9\\n", exit_code: null })],
          ...H.defaultRoutes(fx)])];
    """, """
        await H.tick(20);
        console.log(JSON.stringify({ booted: !!document.getElementById("signout"), authPostsAtBoot: H.authPosts,
                                     following: [...watching.keys()], refused: H.refused.length }));
        // ผู้ดูแลรีสตาร์ต `lmds web` ด้วย token ใหม่: SSE หลุดและต่อกลับไม่ได้ (401) → หน้าถอยไป poll
        H.serverToken = "new";
        const stream = H.streams[H.streams.length - 1];
        stream.readyState = 2; stream.onerror();
        await H.sleep(250); await H.tick(10);             // เกินหนึ่งรอบ poll (5 วิ ÷ 50) และหลายรอบของตัวตาม job
        const tok = document.getElementById("tok");
        const refusedAtLogin = H.refused.length, callsAtLogin = H.calls.length;
        tok.value = "new-token-half-typ";                 // ผู้ใช้กำลังพิมพ์
        await H.sleep(700); await H.tick(10);             // อีก 7 รอบ poll · ~29 รอบของตัวตาม job
        const now = document.getElementById("tok");
        console.log(JSON.stringify({ loginShown: !!tok, sameInput: now === tok, typed: now && now.value,
          message: document.getElementById("tok-err").textContent, refusedAtLogin,
          requestsWhileLoginIsUp: H.calls.slice(callsAtLogin).map(c => c.url), refusedNow: H.refused.length,
          streamClosed: stream.closed, authPosts: H.authPosts }));
        // กรอกตัวที่ถูกแล้วกด Sign in — คำขอเดียวที่ไป /api/auth ตลอดทั้งเรื่อง
        now.value = "new"; document.getElementById("tok-go").onclick(); await H.tick(10);
        console.log(JSON.stringify({ stored: localStorage.getItem("lmds:token"), reloads: H.reloads, authPosts: H.authPosts,
                                     message: (document.getElementById("tok-err") || {}).textContent }));
        H.errors.length = 0;
    """)
    assert boot == {"booted": True, "authPostsAtBoot": 0, "following": ["local-m"], "refused": 0}, \
        "token ที่จำไว้ต้องถูกตรวจด้วย GET ธรรมดา — POST /api/auth สงวนไว้ให้การกด Sign in"
    assert held["loginShown"] and held["sameInput"], "หน้า login ขึ้นครั้งเดียว ไม่ถูกวาดทับ"
    assert held["typed"] == "new-token-half-typ", "สิ่งที่กำลังพิมพ์ต้องรอด"
    assert "no longer works" in held["message"]
    assert held["requestsWhileLoginIsUp"] == [], "หลังหน้า login ขึ้นแล้วต้องไม่มีคำขอเบื้องหลังอีกแม้แต่ครั้งเดียว"
    assert held["refusedNow"] == held["refusedAtLogin"]
    # token เก่าถูกยื่นให้ hub ได้แค่คำขอที่ออกไปแล้วตอน 401 แรกกลับมา (host + models หรือ poll ของ job) — ไม่ใช่ทุก 5 วิ
    assert 1 <= held["refusedAtLogin"] <= 3, held
    assert held["streamClosed"] is True and held["authPosts"] == 0
    assert done == {"stored": "new", "reloads": 1, "authPosts": 1, "message": ""}


def test_a_wrong_token_typed_at_the_login_screen_says_so_and_one_press_is_one_attempt(tmp_path):
    (out,) = run_scenario(tmp_path, GUARDED + """
        localStorage.removeItem("lmds:token");
        const fx = { nodes: [] }; H.fx = fx;
        H.routes = [authRoute, ...guarded(H.defaultRoutes(fx))];
    """, """
        await H.tick(20);
        const tok = document.getElementById("tok"), go = document.getElementById("tok-go");
        const firstMessage = document.getElementById("tok-err").textContent;
        tok.value = "guess";
        go.onclick(); go.onclick(); tok.onkeydown({ key: "Enter" });      // ดับเบิลคลิก + Enter ค้าง
        await H.tick(10);
        console.log(JSON.stringify({ firstMessage, authPosts: H.authPosts, says: document.getElementById("tok-err").textContent,
          stored: localStorage.getItem("lmds:token"), reloads: H.reloads, buttonBack: !go.disabled,
          background: H.calls.filter(c => c.url !== "/api/auth").map(c => c.url) }));
        H.errors.length = 0;
    """)
    assert out["firstMessage"] == "", "เปิดหน้าครั้งแรกโดยไม่มี token — ไม่มีอะไร 'ใช้ไม่ได้แล้ว' ให้บอก"
    assert out["authPosts"] == 1, "กดรัว ๆ = ความพยายามเดียว (hub นับทุก POST ที่ผิด)"
    assert out["says"] == "Wrong token" and out["stored"] is None and out["reloads"] == 0 and out["buttonBack"]
    # คำขอเบื้องหลังของหน้า (ที่ไม่มี token) ออกไปได้แค่ชุดเดียวตอนโหลด — ไม่มีวง poll
    assert len(out["background"]) <= 2, out["background"]


def test_a_throttled_ip_does_not_throw_a_signed_in_page_back_to_the_login_screen(tmp_path):
    """429 = IP นี้ถูกหน่วงเพราะ *มีคน* กรอกผิดซ้ำ ๆ — token ของแท็บนี้ยังใช้ได้ · เดิมทิ้งทั้งหน้าไปหน้า login"""
    (out,) = run_scenario(tmp_path, """
        H.fastTimers(50);
        const fx = { nodes: [{ name: "spark-01", site: "TKC" }] }; H.fx = fx; H.locked = false;
        H.routes = [["/api/host", () => H.locked ? { status: 429, body: { detail: "ผิดหลายครั้งเกินไป — รออีก 8 วินาที" } }
                                                 : { hostname: "hub", gpus: [], ips: [], docker: true, toolkit: true, role: {} }],
                    ...H.defaultRoutes(fx)];
    """, """
        await H.tick(20);
        H.locked = true; await refresh(); await H.tick(5);
        console.log(JSON.stringify({ loginShown: !!document.getElementById("tok"), toast: document.getElementById("toast").textContent,
                                     cards: nodeRows.size }));
        H.errors.length = 0;
    """)
    assert out["loginShown"] is False and out["cards"] == 1
    assert "รออีก 8 วินาที" in out["toast"], "เหตุผลจาก server ต้องขึ้นให้เห็น"


# ───────────────────── ข้อ 4 — JSON จาก node ไปถึง innerHTML โดยไม่ถูก escape ─────────────────────

def test_no_payload_field_can_create_markup(tmp_path):
    """แทน *ทุกใบ* ของ payload ตัวอย่าง (node · model · bench · scan · job · error · cluster · fit · แผน deploy …)
    ด้วย markup ทีละใบ แล้ววาดด้วยทางที่หน้าเว็บใช้จริง — ต้องไม่มี element/attribute ไหนเกิดจากค่านั้นเลย

    scenario (tests/console_xss_walk.js) เดิน object เอง ไม่ได้ไล่ชื่อฟิลด์: ฟิลด์ที่เพิ่มเข้า payload ตัวอย่างทีหลัง
    ถูกตรวจโดยปริยาย · บนหน้าเดิม (6e2b474) เทสนี้เจอ 88 จาก 686 ใบที่สร้าง <img> ได้ และอีก 20 ใบที่ทำให้การ์ดพัง —
    port · slots · cores · pcie_gen ·
    score · size_gb … ฟิลด์ที่ template เชื่อว่าเป็นตัวเลขจึงแปะลงไปตรง ๆ (`x.toLocaleString()` ของสตริงคืนตัวมันเอง)
    """
    prelude, body = (Path(__file__).with_name("console_xss_walk.js").read_text(encoding="utf-8")
                     .split("\n// ---- boot ----\n"))
    (out,) = run_scenario(tmp_path, prelude, body)
    assert out["control"] >= 4, "ตัวตรวจต้องจับ marker ที่แปะลง markup ตรง ๆ ได้ทั้ง 4 บริบท — ไม่งั้นเทสนี้ผ่านเพราะตาบอด"
    report = out["report"]
    for surface in ("node card + fleet models + overview", "hub host + local models", "benchmarks list",
                    "benchmark details", "weights scan", "job on a node", "job on this machine", "error responses"):
        assert report[surface]["leaves"] > 0, f"{surface}: ไม่มีใบให้เดิน — payload ตัวอย่างหาย?"
    assert sum(r["leaves"] for r in report.values()) > 600
    found = {surface: r["found"] for surface, r in report.items() if r["found"]}
    assert found == {}, f"ค่าจาก payload สร้าง markup ได้: {found}"
    # ค่าที่ผิดชนิดต้องไม่ทำให้ทั้งการ์ดพัง (เดิม `.toFixed` ของสตริงโยน แล้วการ์ดของเครื่องนั้นไม่ถูกวาดเลย) ·
    # จอ error ที่ยังอ่าน body ที่ไม่ใช่ JSON ไม่ได้ เป็นเรื่องของข้อ 9 — คุมแยกที่เทสของข้อนั้น
    threw = {surface: r["threw"] for surface, r in report.items() if r["threw"] and surface != "error responses"}
    assert threw == {}, f"payload ผิดชนิดทำให้การวาดโยน: {threw}"


# ───────────────────── ข้อ 5 — ตัวตาม job ตายเพราะ poll หลุดรอบเดียว ─────────────────────

JOBS = MODEL + """H.fastTimers(50);
H.job = { running: true, elapsed: 0, output: "downloading 1/9\\n", exit_code: null };
H.drop = 0; H.gone = false; H.polls = 0; H.cancelled = [];
const fx = { nodes: [{ name: "spark-01", site: "TKC", models: [model({ job: { id: "j1", command: "repair", running: true } })] }],
             localModels: [model({ slug: "local-m", job: { id: "j2", command: "download" } })] };
H.fx = fx;
H.finish = (job) => { H.job = job; fx.nodes[0].models[0].job = null; fx.localModels[0].job = null; };
H.routes = [
  [/^\\/api\\/jobs\\/j[12]\\/cancel$/, url => { H.cancelled.push(url.split("/")[3]); return { cancelled: true }; }],
  [/^\\/api\\/jobs\\/j[12]$/, url => {
     H.polls++;
     if (H.drop > 0) { H.drop--; throw new TypeError("Failed to fetch"); }
     if (H.gone) return { status: 404, body: { detail: "ไม่พบงานนี้" } };
     const id = url.split("/").pop();
     return { ...H.job, id, command: id === "j1" ? "repair" : "download" }; }],
  ...H.defaultRoutes(fx),
];
"""
JOB_VIEW = """
        const flat = el => (el ? el.textContent : "").replace(/\\s+/g, " ").trim();
        const view = () => ({ node: flat(nodeRows.get("spark-01").out), local: flat(document.getElementById("panel-local-m")),
          lostMarker: !!document.querySelector("#panel-local-m [data-job-lost]"), toast: flat(document.getElementById("toast")),
          cancelButtons: document.querySelectorAll("button[data-cancel-job]").length,
          watchingLocal: [...watching.keys()], watchingNode: [...watchingNodes.keys()], alerts: [...H.alerts] });
"""


def test_a_dropped_poll_does_not_end_a_job_follower_and_the_failure_still_reaches_the_screen(tmp_path):
    """เดิม: คำขอเดียวที่หลุด → แผงค้างที่ "running…" พร้อมปุ่ม Cancel · งานล้มทีหลังก็ไม่มีใครบอก (ไม่มี alert)"""
    (following, lost, back, failed) = run_scenario(tmp_path, JOBS, """
        location.hash = "#/nodes"; await H.tick(20);""" + JOB_VIEW + """
        console.log(JSON.stringify(view()));
        H.drop = 6;                                    // wifi สะดุด: poll ถัดไปของตัวตามทั้งสองหลุดติดกัน
        await H.sleep(150); await H.tick(10);
        console.log(JSON.stringify({ ...view(), pollsDuringLoss: H.polls }));
        // server กลับมา งานเดินต่อ
        H.job = { running: true, elapsed: 600, output: "downloading 8/9\\n", exit_code: null };
        await H.sleep(500); await H.tick(10);
        console.log(JSON.stringify(view()));
        // …แล้วงานล้ม
        H.finish({ running: false, elapsed: 900, output: "sha256 mismatch on shard 9\\n", exit_code: 1 });
        await H.sleep(200); await H.tick(10);
        console.log(JSON.stringify(view()));
        H.errors.length = 0;
    """)
    assert "running…" in following["node"] and "downloading 1/9" in following["local"]
    assert following["cancelButtons"] == 2 and following["watchingLocal"] == ["local-m"]

    # ระหว่างสายหลุด: ไม่อ้างว่ากำลังรัน ไม่อ้างว่าจบ — บอกว่าติดต่อไม่ได้และยังตามอยู่
    assert "lost contact with the hub" in lost["node"] and "running…" not in lost["node"], lost["node"]
    assert "may still be running on spark-01" in lost["node"] and "downloading 1/9" in lost["node"]
    assert lost["lostMarker"] and "Lost contact with the hub" in lost["local"] and "downloading 1/9" in lost["local"]
    assert lost["watchingLocal"] == ["local-m"] and lost["watchingNode"] == ["spark-01/qwen"]
    assert "retries on its own" not in lost["toast"], "ข้อความ 'หน้านี้ลองใหม่เอง' เป็นจริงแล้ว จึงไม่ต้องมี toast จากตาข่ายรับท้าย"

    assert "downloading 8/9" in back["node"] and "downloading 8/9" in back["local"], "ต่อกลับมาตามเองโดยไม่ต้อง reload"
    assert not back["lostMarker"] and "lost contact" not in back["node"]

    assert "failed (exit 1)" in failed["node"] and "sha256 mismatch on shard 9" in failed["node"]
    assert any("download local-m failed (exit 1)" in a and "sha256 mismatch" in a for a in failed["alerts"]), failed["alerts"]
    assert failed["watchingLocal"] == [] and failed["watchingNode"] == [] and failed["cancelButtons"] == 0


def test_a_job_the_hub_no_longer_knows_stops_being_followed_and_says_so_in_the_panel(tmp_path):
    """หยุดตามถาวรได้ทางเดียว: hub ตอบ 404 (งานอยู่ในหน่วยความจำ — restart แล้วหาย) · ต้องปลดคีย์และบอกในแผง
    ไม่ใช่ toast ที่หายใน 6 วิ แล้วทิ้งแผง "running…" + ปุ่ม Cancel ไว้"""
    (out,) = run_scenario(tmp_path, JOBS, """
        location.hash = "#/nodes"; await H.tick(20);""" + JOB_VIEW + """
        H.gone = true; H.fx.nodes[0].models[0].job = null; H.fx.localModels[0].job = null;
        await H.sleep(150); await H.tick(10);
        const polls = H.polls; await H.sleep(300);
        console.log(JSON.stringify({ ...view(), pollsAfterGone: H.polls - polls }));
        H.errors.length = 0;
    """)
    for panel in (out["node"], out["local"]):
        assert "Lost track of job" in panel and "no longer knows" in panel, panel
        assert "downloading 1/9" in panel, "ผลล่าสุดที่เคยได้ยังต้องอ่านได้"
    assert out["cancelButtons"] == 0 and out["watchingLocal"] == [] and out["watchingNode"] == []
    assert out["pollsAfterGone"] == 0, "รู้แล้วว่างานไม่มี — ไม่ถามซ้ำ"


def test_the_cancel_button_of_a_followed_job_reaches_the_hub_without_inline_script(tmp_path):
    """ปุ่ม Cancel เดิมเป็น onclick="cancelJob('<id>')" ที่ประกอบจากสตริง — ตอนนี้เป็น data-* + listener ตัวเดียว"""
    (out,) = run_scenario(tmp_path, JOBS, """
        location.hash = "#/nodes"; await H.tick(20);
        const buttons = [...document.querySelectorAll("button[data-cancel-job]")];
        for (const b of buttons) b.click();
        await H.tick(10);
        console.log(JSON.stringify({ cancelled: H.cancelled.sort(), inline: buttons.map(b => b.getAttribute("onclick")) }));
        H.errors.length = 0;
    """)
    assert out == {"cancelled": ["j1", "j2"], "inline": [None, None]}


def test_the_score_table_is_read_again_when_the_benchmark_ends_not_when_it_starts(tmp_path):
    """ข้อ 9: `await followJob(…); loadBench()` — followJob เคย resolve ทันทีที่เริ่มตาม ตารางจึงถูกอ่านตอนงานเพิ่งเริ่ม
    (ยังว่าง) แล้วไม่ถูกอ่านอีกเลย: วัดเสร็จหลายนาทีต่อมา หน้าจอยังขึ้น "Nothing measured yet\""""
    out = run_scenario(tmp_path, MODEL + """H.fastTimers(50);
        H.jobs = { nb: { id: "nb", command: "bench", running: true, elapsed: 1, output: "workload 1/6\\n", exit_code: null },
                   lb: { id: "lb", command: "bench", running: true, elapsed: 1, output: "workload 1/6\\n", exit_code: null } };
        H.runs = [];
        const fx = { nodes: [{ name: "spark-01", site: "TKC", models: [model({ running: true, healthy: true, commands: ["status", "bench"] })] }],
                     localModels: [model({ slug: "local-m", running: true, healthy: true })] };
        H.fx = fx;
        H.routes = [
          [/^\\/api\\/jobs\\/(nb|lb)$/, url => H.jobs[url.split("/").pop()]],
          ["/api/bench/fleet", () => ({ runs: H.runs, unreachable: [] })],
          ["/api/nodes/spark-01/models/qwen/bench", () => ({ job: H.jobs.nb })],
          ["/api/bench/local-m/run", () => ({ job: H.jobs.lb })],
          [/\\/clone\\/targets$/, () => ({ targets: [] })], [/\\/fit$/, () => ({ status: 409, body: { detail: "n/a" } })],
          ...H.defaultRoutes(fx),
        ];
    """, """
        const reads = () => H.calls.filter(c => c.url === "/api/bench/fleet").length;
        const table = () => document.getElementById("bench").textContent.replace(/\\s+/g, " ").trim();
        for (const which of ["node", "local"]) {
          if (which === "node") {
            location.hash = "#/nodes"; await H.tick(20);
            jumpToNode("spark-01", "qwen"); await H.tick(10);
            nodeRows.get("spark-01").body.querySelector('button[data-nact="nbench:full"]').click();
          } else {
            location.hash = "#/hub"; await H.tick(20);
            document.querySelector('button[data-act="opts"][data-slug="local-m"]').click(); await H.tick(10);
            document.querySelector('button[data-act="bench:full"][data-slug="local-m"]').click();
          }
          await H.tick(20); await H.sleep(100);
          const id = which === "node" ? "nb" : "lb", whileRunning = reads();
          // หลายนาทีต่อมา: วัดเสร็จ ผลอยู่บน server แล้ว
          H.runs = [...H.runs, { slug: which === "node" ? "qwen" : "local-m", machine_name: which === "node" ? "spark-01" : "",
                                 speed: { decode_tps_avg: 42 }, capability: { score: 90 } }];
          H.jobs[id] = { ...H.jobs[id], running: false, exit_code: 0, output: "done\\n" };
          await H.sleep(150); await H.tick(20);
          console.log(JSON.stringify({ which, readsAfterFinish: reads() - whileRunning, table: table() }));
        }
        H.errors.length = 0;
    """)
    node, local = out
    assert node["readsAfterFinish"] >= 1 and "qwen" in node["table"] and "90/100" in node["table"], node
    assert local["readsAfterFinish"] >= 1 and "local-m" in local["table"], local


# ───────────────────── ข้อ 3 — คีย์ "เมนูเปิดอยู่" ที่ค้าง แช่แข็งการ์ดทั้งใบ ─────────────────────

MENUS = MODEL + """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [model({ slug: "a", running: true, healthy: true }), model({ slug: "b", port: 8081 })] },
  { name: "spark-02", site: "TKC", models: [model({ slug: "c", running: true, healthy: true })] }] };
H.fx = fx;
H.routes = [
  ["/api/nodes/spark-01/models/b/remove", (url, opts) => {
     if (!opts.body) return { exit_code: 0, output: "would remove bundles/b (1.2 GB)" };
     H.fx.nodes[0].models = H.fx.nodes[0].models.filter(m => m.slug !== "b");
     return { exit_code: 0, output: "removed" }; }],
  [/\\/clone\\/targets$/, () => ({ targets: [] })],
  [/\\/fit$/, () => ({ status: 409, body: { detail: "n/a" } })],
  ...H.defaultRoutes(fx),
];
"""
CARD_STATE = """
        const row = nodeRows.get("spark-01");
        const railDot = () => [...document.querySelectorAll("#rail-nav a")].find(a => a.getAttribute("href") === "#/node/spark-01").querySelector(".rdot").className;
        const overview = () => { const was = location.hash; route = { kind: "overview" }; renderOverview(true);
          const ov = document.getElementById("ov"); const out = { ring: ov.querySelector("svg[role=img]").getAttribute("aria-label"),
            attention: [...ov.querySelectorAll(".ov-alert b")].map(b => b.textContent) }; route = parseRoute(); return out; };
        const state = () => ({ body: row.body.textContent.replace(/\\s+/g, " "), headerDot: row.dot.className, railDot: railDot(),
          held: (row.version.querySelector("[data-held]") || {}).textContent || "", menus: [...openModelMenus],
          submenus: row.body.querySelectorAll(".submenu").length, inUse: nodeIsInUse("spark-01"), ...overview() });
        const frames = async n => { for (let i = 0; i < n; i++) { H.sse(H.snapshot(H.fx)); await H.tick(3); } };
"""


def test_removing_a_model_with_its_menu_open_does_not_freeze_the_card(tmp_path):
    """ลบโมเดลขณะเมนู ⋯ ของมันเปิดอยู่ → คีย์ค้าง → การ์ดไม่ถูกวาดอีกเลย: โมเดลอื่นดับ/ทั้งเครื่องดับ ก็ยังขึ้น running"""
    (removed, stopped, down) = run_scenario(tmp_path, MENUS, """
        location.hash = "#/nodes"; await H.tick(30);""" + CARD_STATE + """
        row.block.querySelector(".ntoggle").click();
        row.body.querySelector('button[data-nact="menu"][data-slug="b"]').click(); await H.tick(10);
        row.body.querySelector('button[data-nact="model:remove"][data-slug="b"]').click(); await H.tick(10);
        row.out.querySelector('button[data-confirm="1"]').click(); await H.tick(20);         // Delete permanently
        console.log(JSON.stringify(state()));
        H.fx.nodes[0].models[0].running = false; H.fx.nodes[0].models[0].healthy = false;      // a ดับ
        await frames(3);
        console.log(JSON.stringify(state()));
        H.fx.nodes[0].reachable = false; H.fx.nodes[0].error = "ssh: connect to host 10.0.0.1 port 22: No route to host";
        await frames(3);
        console.log(JSON.stringify(state()));
        H.errors.length = 0;
    """)
    assert removed["menus"] == [] and removed["inUse"] is False and removed["submenus"] == 0
    assert "stopped" in stopped["body"] and "running" not in stopped["body"].split("lmds")[1].split("System")[1], stopped["body"]
    assert stopped["ring"] == "1 running", "วงแหวนภาพรวมต้องเหลือแค่โมเดลของ spark-02"
    assert "Unreachable" in down["body"] and "No route to host" in down["body"]
    assert down["headerDot"] == "dot " and down["railDot"] == "rdot down"
    assert any("spark-01 unreachable" in t for t in down["attention"]), down["attention"]


def test_a_card_held_for_an_open_menu_still_tells_the_truth_everywhere_else(tmp_path):
    """เมนูที่เปิดจริงยังถือรายการโมเดลไว้ได้ (ค่าที่พิมพ์ต้องไม่หาย) — แต่ rail · ภาพรวม · หัวการ์ด ต้องตามความจริง
    และป้าย paused ต้องขึ้นทันทีโดยนับจากเวลาที่การ์ดถูกวาด ไม่ใช่อายุ probe ของ hub (ซึ่งสดอยู่ตลอด)"""
    (held, later, down) = run_scenario(tmp_path, MENUS, """
        const real = Date.now.bind(Date); H.skew = 0; Date.now = () => real() + H.skew;
        location.hash = "#/nodes"; await H.tick(30);""" + CARD_STATE + """
        row.block.querySelector(".ntoggle").click();
        row.body.querySelector('button[data-nact="menu"][data-slug="a"]').click(); await H.tick(10);
        row.body.querySelector(".n-port").value = "9001";                                      // ผู้ใช้กำลังกรอก
        H.fx.nodes[0].models[0].running = false; H.fx.nodes[0].models[0].healthy = false;      // a ดับ ขณะเมนูเปิดอยู่
        await frames(2);
        const typed = () => (row.body.querySelector(".n-port") || {}).value;
        console.log(JSON.stringify({ ...state(), typed: typed() }));
        H.skew = 95000; await frames(1);                                                       // ผ่านไปอีก 95 วิ · probe ยังอายุ 2 วิ
        console.log(JSON.stringify({ ...state(), typed: typed() }));
        H.fx.nodes[0].reachable = false; H.fx.nodes[0].error = "ssh: No route to host";        // ทั้งเครื่องดับ
        await frames(2);
        console.log(JSON.stringify({ ...state(), typed: typed() }));
        H.errors.length = 0;
    """)
    assert held["typed"] == "9001" and held["submenus"] == 1 and held["inUse"] is True, "เมนูที่เปิดอยู่จริงยังถูกถือไว้"
    assert held["held"].startswith("paused"), "ถือการ์ดไว้เมื่อไรต้องบอกเมื่อนั้น — ไม่รอ 20 วิ"
    assert held["ring"] == "1 running" and held["railDot"] == "rdot ", "rail กับภาพรวมอ่านความจริงล่าสุด ไม่ใช่ของที่ค้างบนการ์ด"
    assert later["held"] == "paused · 95s ago", later["held"]
    # เครื่องดับ: เมนูของโมเดลที่ไม่อยู่ใน payload แล้วไม่มีอะไรให้ถือ — การ์ดบอกตามจริง
    assert down["menus"] == [] and "Unreachable" in down["body"] and down["held"] == ""
    assert down["railDot"] == "rdot down" and any("spark-01 unreachable" in t for t in down["attention"])


def test_jumping_to_a_stacked_worker_shadow_row_does_not_pin_the_card(tmp_path):
    """แถวเงาของ worker ไม่มีเมนู ⋯ — คลิกจากตาราง Fleet models / ค้นหา / palette เคยเพิ่มคีย์ที่ไม่มีใครปิดได้"""
    (out,) = run_scenario(tmp_path, """
        const head = { slug: "qwopus", model_id: "Q/q", engine: "vllm", port: 8000, context: 262144, running: true, healthy: true,
          downloaded: true, controller_exists: true, topology: "stacked", stacked_role: "head", stacked_peers: ["spark-worker"], commands: ["status"] };
        const shadow = { slug: "qwopus", model_id: "Q/q", engine: "vllm", port: 8000, running: true, healthy: true, context: 262144,
          downloaded: true, topology: "stacked", stacked_role: "worker", stacked_head: "spark-head", commands: [], controller_exists: false };
        const fx = { nodes: [{ name: "spark-head", site: "N", models: [head] }, { name: "spark-worker", site: "N", models: [shadow] }] };
        H.fx = fx;
        H.routes = [[/\\/clone\\/targets$/, () => ({ targets: [] })], [/\\/fit$/, () => ({ status: 409, body: { detail: "n/a" } })], ...H.defaultRoutes(fx)];
    """, """
        location.hash = "#/nodes"; await H.tick(30);
        jumpToNode("spark-worker", "qwopus"); jumpToNode("spark-worker", "no-such-model"); await H.tick(10);
        const shadowKeys = [...openModelMenus];
        jumpToNode("spark-head", "qwopus"); await H.tick(10);
        H.fx.nodes[1].reachable = false; H.sse(H.snapshot(H.fx)); await H.tick(5);
        console.log(JSON.stringify({ shadowKeys, keysNow: [...openModelMenus], workerInUse: nodeIsInUse("spark-worker"),
          workerBody: nodeRows.get("spark-worker").body.textContent.replace(/\\s+/g, " ").trim().slice(0, 30),
          headMenu: nodeRows.get("spark-head").body.querySelectorAll(".submenu").length }));
        H.errors.length = 0;
    """)
    assert out["shadowKeys"] == [] and out["workerInUse"] is False
    assert out["keysNow"] == ["spark-head/qwopus"] and out["headMenu"] == 1, "แถวที่มีเมนูจริงยังกางได้ตามเดิม"
    assert out["workerBody"].startswith("Unreachable")


# ───────────────────── ข้อ 1 — wizard เครือข่ายคลัสเตอร์: poll หลุด ≠ "ล้มแล้วถอยกลับแล้ว" ─────────────────────

CNW = """H.fastTimers(50);
const spark = n => ({ name: n, site: "HQ", gpu: { name: "NVIDIA GB10", vram_gb: 128 } });
const fx = { nodes: [spark("spark-a"), spark("spark-b")] }; H.fx = fx;
H.polls = 0; H.drop = 0; H.answer = null;
const step = (node, s, ok = true, detail = "") => ({ node, step: s, ok, detail });
H.step = step;
H.running = { id: "n1", running: true, result: null,
  steps: [step("spark-a", "sudo password accepted"), step("spark-a", "write /etc/netplan/60-lmds-cluster.yaml")] };
H.routes = [
  ["/api/cluster/apply", () => ({ id: "n1", running: true, steps: [] })],
  ["/api/cluster/apply/n1", () => { H.polls++; if (H.drop > 0) { H.drop--; throw new TypeError("Failed to fetch"); }
     return H.answer || H.running; }],
  ...H.defaultRoutes(fx),
];
"""
CNW_OPEN = """
        location.hash = "#/nodes"; await H.tick(30);
        openClusterNetWizard(["spark-a", "spark-b"]);
        cnw.inspect = { nodes: { "spark-a": { sudo_needed: false }, "spark-b": { sudo_needed: false } } };
        cnw.plan = { ok: true, topology: "direct-2", order: ["spark-a", "spark-b"], links: [],
          nodes: { "spark-a": { netplan: "x", cluster_ip: "10.100.152.1" }, "spark-b": { netplan: "y", cluster_ip: "10.100.152.2" } } };
        cnw.step = "apply"; cnwRender();
        const says = () => document.getElementById("cnw-body").textContent.replace(/\\s+/g, " ").trim();
        const foot = () => [...document.querySelectorAll("#cnw button")].filter(b => ["back", "close", "next"].includes(b.dataset.cnw))
          .map(b => b.dataset.cnw + (b.disabled ? " (disabled)" : ""));
        document.querySelector('#cnw button[data-cnw="apply"]').click();
        await H.sleep(120); await H.tick(10);
"""


def test_a_lost_poll_of_the_apply_job_is_not_reported_as_failed_and_rolled_back(tmp_path):
    """poll เดียวที่หลุด เคยกลายเป็น "failed — rolled back" + ปุ่ม Back ทั้งที่งาน sudo/netplan ยังเดินอยู่บนเครื่องจริง"""
    (lost, back, ended) = run_scenario(tmp_path, CNW, CNW_OPEN + """
        H.drop = 3;                                         // wifi สะดุด: สาม poll ติดกันหลุด
        await H.sleep(250); await H.tick(10);
        const pollsAtLoss = H.polls;
        console.log(JSON.stringify({ says: says(), foot: foot(), result: cnw.result }));
        await H.sleep(400); await H.tick(10);               // hub กลับมา งานยังเดินอยู่
        console.log(JSON.stringify({ says: says(), foot: foot(), pollsAfterLoss: H.polls - pollsAtLoss }));
        H.answer = { id: "n1", running: false, steps: H.running.steps,
                     result: { ok: true, applied: true, steps: H.running.steps, pings: [], pairing: [], registry: {} } };
        await H.sleep(200); await H.tick(10);
        console.log(JSON.stringify({ step: cnw.step, result: !!(cnw.result && cnw.result.applied) }));
        H.errors.length = 0;
    """)
    assert "lost contact with the hub — the job may still be running" in lost["says"], lost["says"]
    assert "Failed to fetch" in lost["says"], "เหตุผลจริงของการติดต่อไม่ได้ต้องขึ้นบนจอ"
    assert "rolled back" not in lost["says"].replace("nothing was rolled back", ""), "ห้ามอ้าง rollback ที่ hub ไม่ได้รายงาน"
    assert lost["result"] is None, "poll ที่หลุดไม่ใช่ผลของงาน"
    assert "back" not in " ".join(lost["foot"]) and "close (disabled)" in lost["foot"] and "next (disabled)" in lost["foot"]
    assert back["pollsAfterLoss"] >= 2 and "running…" in back["says"] and "lost contact" not in back["says"]
    assert ended == {"step": "verify", "result": True}, "ตามต่อจนจบ แล้วไปขั้น Verify เองเหมือนไม่เคยหลุด"


def test_a_real_apply_failure_shows_the_hubs_reason_and_only_the_rollbacks_it_reported(tmp_path):
    out = run_scenario(tmp_path, CNW, CNW_OPEN + """
        const s = H.step;
        const cases = {
          "wrong password, nothing touched": { error: "", steps: [s("spark-a", "sudo password accepted"), s("spark-b", "sudo password rejected", false, "Sorry, try again.")] },
          "netplan failed, rolled back": { error: "", steps: [s("spark-a", "write /etc/netplan/60-lmds-cluster.yaml"),
              s("spark-a", "verify addresses", false, "10.100.152.1 not on enp1s0f0np0"), s("spark-a", "rollback to the previous netplan")] },
          "rollback itself failed": { error: "", steps: [s("spark-a", "verify addresses", false, "no carrier"),
              s("spark-a", "rollback to the previous netplan", false, "could not roll back")] },
          "the hub job crashed": { error: "ssh: connect to host spark-b: timed out", steps: [] },
        };
        for (const [name, c] of Object.entries(cases)) {
          cnw.job = "n1"; cnw.result = null; cnw.jobGone = ""; cnw.lost = "";
          H.answer = { id: "n1", running: false, steps: c.steps, result: { ok: false, applied: false, steps: c.steps, error: c.error } };
          await cnwFollowApply("n1"); cnwRender();
          console.log(JSON.stringify({ name, status: document.getElementById("cnw-status").textContent,
            warn: document.querySelector("#cnw-body .warn-line").textContent.replace(/\\s+/g, " ").trim(), foot: foot() }));
        }
        H.errors.length = 0;
    """)
    by = {o["name"]: o for o in out}
    wrong = by["wrong password, nothing touched"]
    assert wrong["status"] == "failed" and "sudo password rejected — Sorry, try again." in wrong["warn"]
    assert "reported no rollback" in wrong["warn"] and "was rolled back on" not in wrong["warn"]
    rolled = by["netplan failed, rolled back"]
    assert rolled["status"] == "failed — rolled back"
    assert "10.100.152.1 not on enp1s0f0np0" in rolled["warn"] and "The previous netplan was rolled back on: spark-a." in rolled["warn"]
    broken = by["rollback itself failed"]
    assert broken["status"] == "failed" and "Rollback FAILED on spark-a" in broken["warn"]
    crashed = by["the hub job crashed"]
    assert "ssh: connect to host spark-b: timed out" in crashed["warn"] and "was rolled back on" not in crashed["warn"]
    assert all("back" in o["foot"] for o in out), "งานจบแล้ว (ล้มจริง) กลับไปแก้แผนได้"


def test_an_apply_job_the_hub_forgot_is_unknown_not_failed(tmp_path):
    (out,) = run_scenario(tmp_path, CNW, CNW_OPEN + """
        H.answer = { status: 404, body: { detail: "ไม่รู้จักงานนี้" } };
        await H.sleep(150); await H.tick(10);
        const polls = H.polls; await H.sleep(200);
        console.log(JSON.stringify({ says: says(), foot: foot(), result: cnw.result, pollsAfter: H.polls - polls }));
        H.errors.length = 0;
    """)
    assert "unknown — the hub no longer knows this job" in out["says"] and "ไม่รู้จักงานนี้" in out["says"]
    assert "Nothing here says the machines were rolled back" in out["says"]
    assert out["result"] is None and out["pollsAfter"] == 0
    assert "back" in out["foot"] and "next (disabled)" in out["foot"], "กลับไปตรวจสายใหม่ได้ แต่ไปขั้น Verify ไม่ได้"


# ───────────────────── ข้อ 7 — wizard deploy ยิง /api/recipes ไม่หยุดเมื่อคำตอบไม่มีสูตร ─────────────────────

def test_the_deploy_wizard_asks_for_recipes_once_per_open_and_says_what_came_back(tmp_path):
    """เดิม: คำตอบที่ไม่มีสูตร (500/502 · คลังว่าง) → วาด → ลิสต์ว่าง → ถามใหม่ → … ~176 คำขอ/วินาทีตลอดที่ wizard เปิด"""
    out = run_scenario(tmp_path, """
        const fx = { nodes: [] }; H.fx = fx; H.mode = "500";
        const later = (v, ms) => new Promise(r => setTimeout(() => r(v), ms));      // "เครือข่าย" 5 ms
        H.routes = [
          ["/api/recipes", () => later({
              "500": { status: 500, body: { detail: "config.yaml อ่านไม่ได้ (บรรทัด 12)" } },
              "502": { status: 502, body: "<html><body><h1>502 Bad Gateway</h1></body></html>" },
              "empty": { recipes: [], source: null, default_repo: "x" },
              "ok": { recipes: [{ match: "Q/q", label: "Q recipe", engine: "vllm" }] } }[H.mode], 5)],
          ["/api/targets", () => ({ targets: [] })],
          ...H.defaultRoutes(fx),
        ];
    """, """
        await H.tick(20);
        const asked = () => H.calls.filter(c => c.url === "/api/recipes").length;
        const shown = () => document.getElementById("w-recipes").textContent.replace(/\\s+/g, " ").trim();
        for (const mode of ["500", "502", "empty"]) {
          H.mode = mode;
          const before = asked();
          document.getElementById("new").click();
          await H.sleep(300);
          const whileOpen = asked() - before, says = shown();
          // ฟอร์มถูกวาดใหม่ระหว่างที่ยังเปิดอยู่ (ถามต่อเรื่อง GGUF / token) — ไม่ใช่การเปิดครั้งใหม่
          wizForm("pick a file"); await H.sleep(100);
          const afterRedraw = asked() - before;
          document.getElementById("w-close").click(); await H.sleep(100);
          console.log(JSON.stringify({ mode, whileOpen, afterRedraw, afterClose: asked() - before - afterRedraw, says }));
        }
        // ผู้ใช้กดลองใหม่เอง — ได้หนึ่งคำขอ และคราวนี้ server ตอบดี
        H.mode = "500"; document.getElementById("new").click(); await H.sleep(100);
        H.mode = "ok"; const before = asked();
        document.getElementById("w-recipes-retry").click(); await H.sleep(100);
        console.log(JSON.stringify({ mode: "retry", whileOpen: asked() - before, says: shown(),
                                     chips: document.querySelectorAll("#w-recipes button[data-recipe]").length }));
        H.errors.length = 0;
    """)
    by = {o["mode"]: o for o in out}
    for mode in ("500", "502", "empty"):
        assert by[mode]["whileOpen"] == 1 and by[mode]["afterRedraw"] == 1 and by[mode]["afterClose"] == 0, by[mode]
    assert "config.yaml อ่านไม่ได้ (บรรทัด 12)" in by["500"]["says"] and "Try again" in by["500"]["says"]
    assert "HTTP 502" in by["502"]["says"] and "502 Bad Gateway" in by["502"]["says"] and "<" not in by["502"]["says"]
    assert by["empty"]["says"].startswith("No proven recipes on this hub yet")
    assert by["retry"]["whileOpen"] == 1 and by["retry"]["chips"] == 1 and "Proven recipes (1)" in by["retry"]["says"]


def test_one_stacked_model_is_one_model_everywhere_on_the_overview(tmp_path):
    """ข้อ 8: snapshot ของโมเดล stacked หนึ่งตัวมีสองแถว (head + เงาบนการ์ด worker) — วงแหวน · "bundles fleet-wide" ·
    ตาราง Sites · ตาราง Fleet models เคยนับเป็นสอง · rail กับไทล์อ่านจาก /api/fleet/summary (แก้ที่ api.py)"""
    (overview, models) = run_scenario(tmp_path, """
        const head = { slug: "qwopus-122b", model_id: "Q/q", engine: "vllm", port: 8000, context: 262144, running: true, healthy: true,
          downloaded: true, controller_exists: true, topology: "stacked", stacked_role: "head", stacked_peers: ["spark-worker"], commands: ["status"] };
        const shadow = { slug: "qwopus-122b", model_id: "Q/q", engine: "vllm", port: 8000, running: true, healthy: true, context: 262144,
          downloaded: true, topology: "stacked", stacked_role: "worker", stacked_head: "spark-head", commands: [], controller_exists: false };
        const fx = { nodes: [{ name: "spark-head", site: "Neronain", models: [head] }, { name: "spark-worker", site: "Neronain", models: [shadow] }] };
        H.fx = fx;
        H.routes = [
          ["/api/fleet/summary", () => ({ machines: 3, online: 3, pending: 0, gpus: 2, vram_gb: 256, models_running: 1, models_healthy: 1, models_total: 1 })],
          ...H.defaultRoutes(fx)];
    """, """
        await H.tick(30); H.sse(H.snapshot(H.fx)); await H.tick(10); renderOverview(true);
        const txt = el => el.textContent.replace(/\\s+/g, " ").trim();
        console.log(JSON.stringify({ ring: document.querySelector("#ov svg[role=img]").getAttribute("aria-label"),
          legend: txt(document.querySelector("#ov .ov-legend")),
          siteRunning: [...document.querySelectorAll("#ov tr.orow[data-href^='#/site'] td")].map(txt)[3] }));
        location.hash = "#/models"; await H.tick(10);
        console.log(JSON.stringify({ head: txt(document.querySelector("#allmodels .vhead")),
          rows: [...document.querySelectorAll("#allmodels tr.orow")].map(tr => tr.dataset.node),
          workerCard: txt(nodeRows.get("spark-worker").body).includes("stacked worker of spark-head") }));
        H.errors.length = 0;
    """)
    assert overview["ring"] == "1 running" and "1 bundles fleet-wide · 0 stopped" in overview["legend"], overview
    assert overview["siteRunning"] == "1"
    assert models["head"] == "Fleet models1 bundle1 running" and models["rows"] == ["spark-head"]
    assert models["workerCard"] is True, "การ์ดของ worker ยังต้องเห็นว่าเครื่องนี้ถูกใช้อยู่ — ซ่อนจากการนับ ไม่ได้ซ่อนจากการ์ด"


def test_the_recipes_page_shows_the_servers_reason_when_it_cannot_be_read(tmp_path):
    (out,) = run_scenario(tmp_path, """
        const fx = { nodes: [] }; H.fx = fx;
        H.routes = [["/api/recipes", () => ({ status: 500, body: { detail: "config.yaml อ่านไม่ได้ (บรรทัด 12)" } })], ...H.defaultRoutes(fx)];
    """, """
        await H.tick(20); await loadRecipes(); await H.tick(5);
        console.log(JSON.stringify({ says: document.getElementById("recipes").textContent.replace(/\\s+/g, " ").trim(), alerts: H.alerts }));
        H.errors.length = 0;
    """)
    assert "config.yaml อ่านไม่ได้ (บรรทัด 12)" in out["says"] and out["alerts"] == []
