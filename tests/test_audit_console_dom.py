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
