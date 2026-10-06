"""audit หน้าคอนโซล 2026-10 — ทุกข้อเป็นเทสที่ล้มก่อนแก้ · รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน

แต่ละเทสคือ scenario ที่ผู้ตรวจใช้ยืนยันบั๊ก (payload รูปเดียวกับที่ hub ส่งจริง) แล้วถามที่ **ผลบนจอ**:
ข้อความที่วาด · ปุ่มที่กดได้ · คำขอที่ออกไป — ไม่ใช่สตริงในซอร์ส · เวลาของหน้าเว็บถูกเร่งด้วย
`H.fastTimers()` (วง poll 1.2 วิ / 5 วิ และ backoff ของมันจึงทดสอบได้ในเสี้ยววินาที) ส่วน `H.sleep` เป็นเวลาจริง
"""

from __future__ import annotations

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
