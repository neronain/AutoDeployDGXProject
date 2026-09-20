"""i18n ไทยของหน้าเว็บ + command palette (⌘K) — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน
(tests/console_shell_dom.js)

ROADMAP เฟส 2 ข้อ 7: "i18n ไทยเต็มรูปของหน้าเว็บ (ตอนนี้หน้าเว็บอังกฤษ CLI ไทย)" และข้อ 4
"งานต่อยอด: command palette (⌘K)"

เทสตรงนี้ถามหน้าเว็บจริง ๆ ว่า:
 · ค่าตั้งต้นยังเป็นอังกฤษ — ไทยเป็น "ตัวเลือก" ไม่ใช่ของใหม่ที่บังคับใส่ให้ทุกคน
 · กดปุ่มภาษาแล้วเมนูซ้าย/เส้นทาง/หัวข้อหมวด/ภาพรวม เปลี่ยนเป็นไทย *โดยไม่ reload*
 · จำไว้ที่ localStorage คีย์ "lmds-lang" ชุดเดียวกับ lmds-theme / lmds-fontsize
   และเปิดหน้าครั้งถัดไปได้ไทยตั้งแต่ภาพแรก
 · สลับกลับเป็นอังกฤษแล้วได้ข้อความอังกฤษ *ตัวเดิมเป๊ะ* (ต้นฉบับยังอยู่ ไม่ได้แปลกลับแบบเดา)
 · i18n ไม่แตะ `hidden` เลย — บนหน้านี้ `hidden` แปลว่า "สิทธิ์ (แอปคุม)" ไม่ใช่ "ซ่อนไว้ก่อน"
 · palette ค้นเจอ: เครื่องในฟลีต · ไซต์ · หน้าในเมนูซ้าย · โมเดล/bundle
   แล้วไปด้วย router เดิม (#/node/… #/site/… #/hub) ไม่ใช่ระบบนำทางใหม่
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

# 2 ไซต์ + เครื่องที่ยังไม่จัดไซต์ + โมเดลทั้งบน hub และบน node — ครบทุกหมวดที่ palette ต้องค้นเจอ
FLEET = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [{ slug: "qwen3-8b", running: true, healthy: true, engine: "llamacpp",
      port: 8080, context: 32768, downloaded: true, controller_exists: true }] },
  { name: "spark-02", site: "TKC" },
  { name: "lonely", site: "" } ],
  localModels: [{ slug: "local-a", running: true, healthy: true, controller_exists: true, downloaded: true,
                  engine: "llamacpp", port: 8090, topology: "single", context: 8192, model_id: "org/local-a" }] };
H.fx = fx;
H.routes = H.defaultRoutes(fx);
"""

# ข้อความที่ผู้ใช้เห็นบ่อยที่สุด — เมนูซ้าย, เส้นทาง, หัวข้อหมวด, หัวตาราง
PROBE = """
const probe = () => ({
  lang: document.documentElement.lang,
  stored: localStorage.getItem("lmds-lang"),
  button: document.getElementById("lang").textContent.trim(),
  rail: document.getElementById("rail-nav").textContent,
  crumb: document.getElementById("crumb").textContent,
  glance: document.querySelector('[data-view="overview"] .sec-title').textContent,
  nodesTitle: document.getElementById("nodes-title").textContent,
  overview: document.getElementById("ov").textContent,
  fleetBtn: document.getElementById("node-new").textContent.trim(),
  search: document.getElementById("rsearch").getAttribute("placeholder"),
  restart: document.getElementById("restart-web").textContent.trim(),
});
"""


def test_english_is_the_default_and_nothing_thai_leaks_in(tmp_path):
    """ค่าตั้งต้นต้องเป็นอังกฤษเหมือนเดิมเป๊ะ — ลูกค้า/SI ที่ใช้อยู่ต้องไม่เจอหน้าเปลี่ยนภาษาเอง"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        console.log(JSON.stringify({ ...probe(), errors: H.errors, alerts: H.alerts }));
    """)
    assert out["stored"] is None, "ยังไม่เคยเลือก = ไม่ต้องเขียนอะไรลง storage"
    assert out["lang"] == "en" and out["button"] == "EN"
    for english in ["Overview", "All machines", "Fleet models", "Hub settings"]:
        assert english in out["rail"]
    assert "Fleet at a glance" == out["glance"]
    assert out["search"] == "Search machines / models…"
    assert out["restart"] == "Restart console"
    # ไม่มีอักษรไทยหลุดมาบนหน้าที่ยังเป็นอังกฤษ
    for field in ("rail", "crumb", "glance", "nodesTitle", "fleetBtn", "search", "restart"):
        assert not any("฀" <= c <= "๿" for c in out[field]), f"{field} มีอักษรไทยปนในโหมดอังกฤษ"
    assert out["errors"] == [] and out["alerts"] == []


def test_the_language_button_switches_the_page_to_thai_without_reloading(tmp_path):
    """กดปุ่มเดียวแล้วเมนูซ้าย เส้นทาง หัวข้อหมวด และภาพรวมต้องเป็นไทยทันที
    — ไม่ reload เพราะ reload ทิ้งเมนูที่กางอยู่ ช่องที่กรอกค้าง และผลคำสั่งที่กำลังอ่าน"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        const before = probe();
        document.getElementById("lang").click(); await H.tick(8);
        console.log(JSON.stringify({ before, after: probe(), reloads: H.reloads, errors: H.errors }));
    """)
    after = out["after"]
    assert out["reloads"] == 0, "สลับภาษาต้องไม่ reload ทั้งหน้า"
    assert after["lang"] == "th" and after["button"] == "ไทย"
    assert after["stored"] == "th", "ต้องจำไว้ที่ lmds-lang เหมือนที่ธีม/ขนาดตัวอักษรจำไว้"
    for thai in ["ภาพรวม", "เครื่องทั้งหมด", "โมเดลทั้งฟลีต", "ตั้งค่า hub", "ไซต์ / เครื่อง"]:
        assert thai in after["rail"], f"เมนูซ้ายยังไม่มี '{thai}'"
    assert "ฟลีต" in after["crumb"] and "ภาพรวม" in after["crumb"]
    assert after["glance"] == "ภาพรวมทั้งฟลีต"
    assert after["nodesTitle"] == "เครื่องอื่น"
    assert after["fleetBtn"] == "เพิ่มเครื่อง"
    assert after["search"] == "ค้นเครื่อง / โมเดล…"
    assert after["restart"] == "รีสตาร์ตคอนโซล"
    # การ์ดภาพรวมเป็น innerHTML ที่ JS วาดเอง — ต้องถูกวาดใหม่ด้วย ไม่ใช่ค้างเป็นอังกฤษ
    for thai in ["หน่วยความจำ GPU ต่อเครื่อง", "ต้องดู", "ไซต์"]:
        assert thai in after["overview"], f"ภาพรวมยังไม่มี '{thai}'"
    assert out["errors"] == []


def test_thai_is_already_on_at_the_first_paint_when_it_was_chosen_before(tmp_path):
    """เลือกไทยไว้แล้วเปิดหน้าใหม่ ต้องได้ไทยตั้งแต่ภาพแรก ไม่ใช่เห็นอังกฤษวาบก่อนแล้วค่อยเปลี่ยน
    (กติกาเดียวกับธีมมืดที่ตัดสินก่อนวาดหน้า)"""
    (out,) = run_scenario(tmp_path, 'localStorage.setItem("lmds-lang", "th");\n' + FLEET, PROBE + """
        console.log(JSON.stringify({ ...probe(), errors: H.errors }));
    """)
    assert out["lang"] == "th" and out["button"] == "ไทย"
    assert "ภาพรวม" in out["rail"] and "เครื่องทั้งหมด" in out["rail"]
    assert out["glance"] == "ภาพรวมทั้งฟลีต"
    assert out["errors"] == []


def test_switching_back_to_english_restores_the_original_words_exactly(tmp_path):
    """ไทย → อังกฤษ ต้องได้ข้อความอังกฤษ *ตัวเดิมเป๊ะ* — เพราะเก็บต้นฉบับไว้ ไม่ใช่แปลกลับแบบเดา
    (แปลกลับจากไทยคือทางที่ทำให้คำเพี้ยนสะสมทุกครั้งที่สลับ)"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        // ตัวเลขข้างชื่อเมนู (/api/fleet/summary) มาทีหลังการวาดรางครั้งแรก และรางจะวาดใหม่
        // ก็ต่อเมื่อมีอะไรมาสั่ง — วาดให้นิ่งก่อนถ่ายภาพ "อังกฤษต้นฉบับ" ไม่งั้นเทียบคนละรอบ
        await H.tick(10); renderRail(); await H.tick(2);
        const first = probe();
        document.getElementById("lang").click(); await H.tick(8);
        const thai = probe();
        document.getElementById("lang").click(); await H.tick(8);
        console.log(JSON.stringify({ first, thai, back: probe(), errors: H.errors }));
    """)
    first, back = out["first"], out["back"]
    assert out["thai"]["lang"] == "th" and back["lang"] == "en"
    assert back["stored"] == "en"
    for field in ("rail", "crumb", "glance", "nodesTitle", "fleetBtn", "search", "restart"):
        assert back[field] == first[field], f"{field} ไม่กลับเป็นอังกฤษตัวเดิม"


def test_i18n_never_touches_hidden(tmp_path):
    """`hidden` บนหน้านี้แปลว่า "สิทธิ์ (แอปคุม)" ไม่ใช่ "ซ่อนไว้ชั่วคราว" — เคยมีบั๊กสิทธิ์จากการ
    เอา hidden ไปใช้สลับการแสดงผล · การสลับภาษาต้องไม่ทำให้อะไรที่เคยเห็นหายไป และต้องไม่
    ไปตั้ง/ปลด hidden ให้ใครทั้งนั้น"""
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.go("#/nodes");
        const snap = () => ({
          machines: H.visibleMachines(),
          hidden: [...document.querySelectorAll("[hidden]")].map(el => el.id || el.className).sort(),
        });
        const before = snap();
        document.getElementById("lang").click(); await H.tick(8);
        const after = snap();
        document.getElementById("lang").click(); await H.tick(8);
        console.log(JSON.stringify({ before, after, back: snap(), errors: H.errors }));
    """)
    assert out["before"]["machines"] == ["spark-01", "spark-02", "lonely"]
    assert out["after"]["machines"] == out["before"]["machines"], "สลับภาษาแล้วการ์ดเครื่องต้องไม่หาย"
    assert out["after"]["hidden"] == out["before"]["hidden"], "สลับภาษาต้องไม่แตะ hidden ของใครเลย"
    assert out["back"] == out["before"]
    assert out["errors"] == []


# ── command palette (⌘K) ──────────────────────────────────────────────────────

PAL = """
const openPal = async () => {
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }));
  await H.tick(4);
};
const type = async (text) => {
  const input = document.getElementById("pal-q");
  input.value = text;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  await H.tick(2);
};
const press = async (key) => {
  const input = document.getElementById("pal-q");
  input.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
  await H.tick(4);
};
const rows = () => [...document.querySelectorAll(".pal-item")].map(el => ({
  label: el.querySelector("b").textContent, where: el.querySelector(".where").textContent,
  on: el.getAttribute("aria-selected") === "true" }));
"""


def test_palette_opens_with_cmd_k_and_finds_machines_sites_pages_and_models(tmp_path):
    """หน้านี้ยาวขึ้นตามจำนวนเครื่อง — palette ต้องพาไปได้ทุกอย่างที่เมนูซ้ายพาไปได้
    ไม่ใช่ค้นเจอแต่โมเดลอย่างเดียว"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        const closedBefore = !document.querySelector(".pal-back");
        await openPal();
        const all = rows();
        await type("spark"); const machines = rows();
        await type("TKC"); const site = rows();
        await type("benchmarks"); const page = rows();
        await type("qwen"); const model = rows();
        await type("zzzz"); const none = document.querySelector(".pal-empty").textContent;
        console.log(JSON.stringify({ closedBefore, total: all.length, machines, site, page, model, none,
                                     errors: H.errors }));
    """)
    assert out["closedBefore"] is True, "palette ต้องไม่ค้างเปิดอยู่ตั้งแต่โหลดหน้า"
    labels = lambda rs: [r["label"] for r in rs]  # noqa: E731

    # เครื่องในฟลีต — ชื่อเครื่องขึ้นก่อน โดยมีไซต์ของมันเป็นคำบอกที่อยู่ · bundle ที่อยู่บนเครื่องนั้น
    # ก็ควรติดมาด้วย เพราะ "spark" คือที่อยู่ของมัน
    assert labels(out["machines"]) == ["spark-01", "spark-02", "qwen3-8b"]
    assert [r["where"] for r in out["machines"]] == ["TKC", "TKC", "spark-01"]

    # ไซต์ — แถวของไซต์เองมาก่อนเครื่องที่อยู่ในไซต์นั้น
    assert labels(out["site"]) == ["TKC", "spark-01", "spark-02"]
    assert out["site"][0]["where"] == "site"

    # หน้าในเมนูซ้าย
    assert labels(out["page"]) == ["Benchmarks"]
    assert out["page"][0]["where"] == "page"

    # โมเดล/bundle บนเครื่องอื่น
    assert labels(out["model"]) == ["qwen3-8b"]
    assert out["model"][0]["where"] == "spark-01"

    assert "zzzz" in out["none"]
    assert out["errors"] == []


def test_palette_uses_the_existing_router_for_pages_sites_and_machines(tmp_path):
    """"อย่าสร้างระบบนำทางใหม่" — ทุกรายการต้องลงเอยที่ #/… ของ router เดิม"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        const go = async (query) => {
          await openPal(); await type(query); await press("Enter"); await H.tick(6);
          return { hash: location.hash, closed: !document.querySelector(".pal-back") };
        };
        const page = await go("Weights on disk");
        const site = await go("TKC");
        const machine = await go("lonely");
        const hub = await go("Overview");
        console.log(JSON.stringify({ page, site, machine, hub, route: route.kind, errors: H.errors }));
    """)
    assert out["page"] == {"hash": "#/weights", "closed": True}
    assert out["site"] == {"hash": "#/site/TKC", "closed": True}
    assert out["machine"] == {"hash": "#/node/lonely", "closed": True}
    assert out["hub"] == {"hash": "#/overview", "closed": True}
    assert out["route"] == "overview"
    assert out["errors"] == []


def test_palette_arrow_keys_move_the_selection_and_esc_closes_it(tmp_path):
    """↑↓ เลือก · Enter ไป · Esc ปิด — และ Esc ต้องปิดจริง ไม่ใช่แค่ซ่อน"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        await openPal(); await type("spark");
        const first = rows().findIndex(r => r.on);
        await press("ArrowDown");
        const second = rows().findIndex(r => r.on);
        const secondLabel = rows()[second].label;
        await press("ArrowUp");
        const back = rows().findIndex(r => r.on);
        await press("Escape");
        const node = document.querySelector(".pal-back");
        console.log(JSON.stringify({ first, second, secondLabel, back,
                                     gone: node === null, errors: H.errors }));
    """)
    assert out["first"] == 0 and out["second"] == 1 and out["back"] == 0
    assert out["secondLabel"] == "spark-02"
    assert out["gone"] is True, "Esc ต้องเอากล่องออกจาก DOM ไม่ใช่ตั้ง hidden ทิ้งไว้"
    assert out["errors"] == []


def test_palette_enter_on_a_model_jumps_to_that_machine_card(tmp_path):
    """โมเดลบนเครื่องอื่น: ต้องไปหน้าเครื่องนั้นแล้วกางเมนูของ bundle นั้นให้ — ใช้ jumpToNode เดิม"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        await openPal(); await type("qwen3-8b"); await press("Enter"); await H.tick(8);
        console.log(JSON.stringify({ hash: location.hash, route: route.kind,
                                     menus: [...openModelMenus], visible: H.visibleMachines(),
                                     errors: H.errors }));
    """)
    assert out["hash"] == "#/node/spark-01" and out["route"] == "node"
    assert out["menus"] == ["spark-01/qwen3-8b"]
    assert out["visible"] == ["spark-01"]
    assert out["errors"] == []


def test_palette_speaks_thai_once_the_page_is_thai(tmp_path):
    """สลับภาษาแล้ว palette ต้องตามไปด้วย — ไม่ใช่กล่องอังกฤษโดดอยู่กลางหน้าไทย"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        document.getElementById("lang").click(); await H.tick(8);
        await openPal();
        const box = document.querySelector(".pal");
        await type("ภาพรวม"); const hit = rows();
        console.log(JSON.stringify({
          placeholder: document.getElementById("pal-q").getAttribute("placeholder"),
          foot: box.querySelector(".pal-foot").textContent,
          hit, errors: H.errors }));
    """)
    assert out["placeholder"] == "ค้นโมเดล เครื่อง หน้า…"
    assert "↑↓ เลือก" in out["foot"] and "Esc ปิด" in out["foot"]
    assert [r["label"] for r in out["hit"]] == ["ภาพรวม"], "ค้นด้วยคำไทยต้องเจอหน้าที่ชื่อไทย"
    assert out["hit"][0]["where"] == "หน้า"
    assert out["errors"] == []


def test_palette_lists_exactly_the_pages_the_left_menu_lists(tmp_path):
    """"ควรค้นได้อย่างน้อย … หน้าในเมนูซ้าย" — ผูกสองที่เข้าด้วยกัน ไม่ใช่ลอกรายการไว้สองชุด
    แล้วรอให้มันหลุดกันตอนเพิ่มหน้าใหม่"""
    (out,) = run_scenario(tmp_path, FLEET, PAL + """
        // ลิงก์ระดับบนของราง (ไม่ใช่กิ่งไซต์/เครื่อง) — ตัวเลขนับข้างชื่อไม่ใช่ส่วนของชื่อ
        const rail = [...document.querySelectorAll("#rail-nav > a")]
          .filter(a => !/^#\\/(node|site)\\//.test(a.getAttribute("href")))
          .map(a => a.textContent.replace(/\\d+$/, "").trim());
        await openPal();
        const pages = rows().filter(r => r.where === "page").map(r => r.label);
        console.log(JSON.stringify({ rail, pages, errors: H.errors }));
    """)
    assert out["rail"], "รางซ้ายต้องมีลิงก์ระดับบน"
    assert out["pages"] == out["rail"], "palette กับเมนูซ้ายต้องมีหน้าชุดเดียวกัน เรียงเหมือนกัน"
    assert out["errors"] == []


def test_the_thai_dictionary_has_no_duplicate_keys():
    """คีย์ซ้ำในออบเจกต์เดียวกัน = ตัวหลังชนะเงียบ ๆ · คำแปลที่หายไปแบบนี้หาไม่เจอจนกว่าจะมีคนทัก"""
    import re
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    block = page.split("const TH = {")[1].split("\n  };")[0]
    keys = re.findall(r'^ {4}"((?:[^"\\]|\\.)*)":', block, re.M)
    assert len(keys) > 150, "พจนานุกรมหดลงผิดปกติ — อ่านบล็อกไม่เจอหรือถูกลบ?"
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert duplicates == [], f"คีย์ซ้ำในพจนานุกรมไทย: {duplicates}"


def test_language_is_remembered_the_same_way_as_theme_and_text_size():
    """สามอย่างนี้เป็นความชอบของคนที่นั่งดูจอ ไม่ใช่ค่าของ hub — ต้องเก็บที่เดียวกันแบบเดียวกัน"""
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html").read_text(encoding="utf-8")
    for key in ('"lmds-theme"', '"lmds-fontsize"', '"lmds-lang"'):
        assert f"localStorage.setItem({key}" in page, f"{key} ต้องถูกจำไว้ที่ localStorage"
        assert f"localStorage.getItem({key}" in page or "LANG_KEY" in page
    assert 'window.LANG = window.LANGS.includes(stored()) ? stored() : "en"' in page, \
        "ค่าที่อ่านไม่ออก/ของเก่าต้องตกกลับเป็นอังกฤษ ไม่ใช่ทำให้หน้าพัง"


def test_switching_language_reaches_the_buttons_that_are_only_built_once(tmp_path):
    """แถวของโมเดลถูกสร้างครั้งเดียวแล้วอัปเดตเฉพาะบางช่อง — ปุ่ม Tests/Manage/Doctor/Logs
    ไม่เคยถูกวาดใหม่ · ถ้าแปลตอนสร้างอย่างเดียว มันจะค้างภาษาเดิมตลอดเมื่อผู้ใช้สลับภาษา"""
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.go("#/hub");
        const labels = () => [...document.querySelectorAll('#fleet button[data-act]')]
          .map(b => b.textContent.trim()).filter(Boolean);
        const before = labels();
        document.getElementById("lang").click(); await H.tick(8);
        const thai = labels();
        document.getElementById("lang").click(); await H.tick(8);
        console.log(JSON.stringify({ before, thai, back: labels(), errors: H.errors }));
    """)
    assert "Manage" in out["before"] and "Logs" in out["before"]
    assert "ตั้งค่า" in out["thai"] and "หมอ" in out["thai"], "ปุ่มที่สร้างครั้งเดียวก็ต้องเปลี่ยนภาษาตาม"
    assert out["back"] == out["before"], "สลับกลับต้องได้ป้ายอังกฤษตัวเดิม"
    assert out["errors"] == []
