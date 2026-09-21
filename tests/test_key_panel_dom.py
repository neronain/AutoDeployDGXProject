"""แผงกุญแจและร่องรอยการใช้งานบนหน้าเว็บ — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน

หน้าเว็บคือ **ของหลัก** ของงานนี้ ไม่ใช่ของต่อยอดจาก CLI · ลูกค้าที่เริ่มเอาไปทดสอบใช้งาน
ผ่าน GUI เป็นหลัก คนกลุ่มนั้นตั้งกุญแจไม่ได้และดูร่องรอยไม่ได้เลย ทั้งที่เป็นงานความปลอดภัย
ที่ทำมาเพื่อพวกเขาโดยตรง · ที่ย้อนแย้งที่สุดคือ wizard deploy บนหน้าเดียวกันนี้ *พิมพ์ข้อความ
บอกให้ไปรัน* `lmds key show <slug> --reveal` ซึ่งเป็นคำสั่งที่คนกลุ่มนั้นพิมพ์ไม่ได้อยู่แล้ว

สิ่งที่ชุดนี้ถามหน้าเว็บจริง: ค่าเต็มของกุญแจ **ต้องไม่มาพร้อมสถานะ** (หน้านี้ถูกเปิดค้างไว้
ทั้งวันบนจอที่คนเดินผ่าน) · การทับของเดิมต้องถูกถามก่อน ไม่ใช่หลัง · การเอาออกต้องบอกว่า
จะเกิดอะไรขึ้น · audit ต้องไม่ยิงเองตอนโหลดหน้า · และ **ไม่แตะ `hidden` ของใครเลย**
(บนหน้านี้ `hidden` แปลว่า "สิทธิ์ (แอปคุม)" ไม่ใช่ "ซ่อนไว้ก่อน")
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

KEY = "kkkk1111222233334444555566667777888899990000aaaa"

FLEET = """const fx = { nodes: [{ name: "spark-01", site: "TKC", models: [] }],
  localModels: [{ slug: "demo", name: "demo", running: true, port: 8001, commands: [] }] };
H.fx = fx;
H.posted = [];
H.state = { slug: "demo", has_key: false, path: "/home/u/.lmds/keys/demo", hint: "" };
H.routes = [
  [/^\\/api\\/models\\/[^/]+\\/key\\/reveal/, url => ({ slug: "demo", key: "%KEY%" })],
  [/^\\/api\\/models\\/[^/]+\\/key/, (url, opts) => {
     const method = (opts.method || "GET").toUpperCase();
     if (method === "POST") {
       H.posted.push(JSON.parse(opts.body || "{}"));
       H.state = { ...H.state, has_key: true, hint: "kkkk…aaaa" };
       return { slug: "demo", key: "%KEY%", generated: true, restart_required: true };
     }
     if (method === "DELETE") {
       H.state = { ...H.state, has_key: false, hint: "" };
       return { slug: "demo", removed: true, restart_required: true,
                warning: "หลัง restart bundle นี้จะเสิร์ฟแบบเปิด" };
     }
     return H.state;
  }],
  [/^\\/api\\/audit/, url => ({
     enabled: true, path: "/home/u/.lmds/audit.log", refused: 1,
     entries: [{ at: "2026-09-21T09:00:00+07:00", ip: "10.0.0.9", method: "POST",
                 path: "/api/models/demo/key", status: 200, ms: 12 },
               { at: "2026-09-21T09:01:00+07:00", ip: "203.0.113.4", method: "GET",
                 path: "/api/models", status: 401, ms: 1 }] })],
  ...H.defaultRoutes(fx),
];
""".replace("%KEY%", KEY)

PROBE = """
const hiddenSet = () => [...document.querySelectorAll("[hidden]")].map(el => el.id || el.className).sort();
const panel = () => document.querySelector("#panel-demo");
const text = () => (panel() ? panel().textContent : "").replace(/\\s+/g, " ").trim();
const seen = () => H.calls.map(c => `${c.method || "GET"} ${c.url}`);
const click = async (sel) => { document.querySelector(sel).click(); await H.tick(8); };
const open = async () => { await H.go("#/models"); await click('button[data-act="key"][data-slug="demo"]'); };
"""


def test_the_status_never_carries_the_key_itself(tmp_path):
    """หน้านี้ถูกเปิดค้างไว้ทั้งวัน · ถ้าสถานะพ่วงค่าเต็มมาด้วย กุญแจจะนอนอยู่ใน DOM
    ตลอดเวลาโดยไม่มีใครขอ และใครที่เดินผ่านจอก็อ่านได้"""
    (out,) = run_scenario(tmp_path, FLEET.replace("has_key: false", "has_key: true")
                          .replace('hint: ""', 'hint: "kkkk…aaaa"'), PROBE + """
        await open();
        console.log(JSON.stringify({ text: text(), calls: seen(),
                                     html: panel().innerHTML, errors: H.errors, alerts: H.alerts }));
    """)
    assert KEY not in out["html"], "ค่าเต็มต้องไม่อยู่ในหน้าจนกว่าจะกดขอ"
    assert "kkkk…aaaa" in out["text"], "ต้องเห็นปลายกุญแจพอให้เทียบกับที่ client ถืออยู่"
    assert not any("reveal" in c for c in out["calls"]), "เปิดแผงแล้วต้องไม่ดึงค่าเต็มเอง"
    assert out["errors"] == [] and out["alerts"] == []


def test_a_bundle_without_a_key_says_plainly_that_it_is_open(tmp_path):
    """ช่องว่างไม่ได้แปลว่าปลอดภัย · ไม่มีกุญแจ = ใครยิงถึงพอร์ตก็ใช้โมเดลได้ ต้องพูดออกมา"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await open();
        console.log(JSON.stringify({ text: text(), errors: H.errors }));
    """)
    assert "reach this port" in out["text"]
    assert out["errors"] == []


def test_revealing_is_a_separate_press_and_can_be_copied(tmp_path):
    """แยกการกด 'ดูค่า' ออกมาเพื่อให้รู้ได้ว่าใครเปิดดู · และคนที่เพิ่งสร้างต้องเอาไปตั้งที่
    client ได้โดยไม่ต้องลากเมาส์เลือกสตริง 48 ตัว ซึ่งคือที่ที่ตัวอักษรหายไปหนึ่งตัว"""
    (out,) = run_scenario(tmp_path, FLEET.replace("has_key: false", "has_key: true"), PROBE + """
        await open();
        await click('button[data-act="key:reveal"][data-slug="demo"]');
        console.log(JSON.stringify({ shown: panel().querySelector(".key-out .mono").textContent.trim(),
                                     canCopy: !!panel().querySelector(".key-copy"),
                                     calls: seen(), errors: H.errors }));
    """)
    assert out["shown"] == KEY and out["canCopy"]
    assert sum(1 for c in out["calls"] if c.endswith("/key/reveal")) == 1


def test_replacing_an_existing_key_is_confirmed_before_anything_is_sent(tmp_path):
    """client ทุกตัวที่ถือใบเก่าใช้ไม่ได้ทันทีที่โมเดล restart · ถามหลังส่งไปแล้วคือสายเกินไป
    และการให้ผู้ใช้เจอ 409 แล้วเดาต่อเองคือการผลักงานของเราไปให้เขา"""
    refused = FLEET.replace("has_key: false", "has_key: true")
    (out,) = run_scenario(tmp_path, refused, PROBE + """
        H.confirmAnswer = false;
        await open();
        await click('button[data-act="key:new"][data-slug="demo"]');
        const afterNo = H.posted.length;
        H.confirmAnswer = true;
        await click('button[data-act="key:new"][data-slug="demo"]');
        console.log(JSON.stringify({ afterNo, posted: H.posted, asked: H.confirms, errors: H.errors }));
    """)
    assert out["afterNo"] == 0, "ตอบไม่ = ต้องไม่มีอะไรถูกส่งออกไปเลย"
    assert out["posted"] == [{"force": True}], "ตอบใช่ = ต้องส่ง force ไม่ใช่ปล่อยให้ชน 409"
    assert any("demo" in q for q in out["asked"]), "คำถามต้องบอกว่ากำลังจะทับของ bundle ไหน"


def test_a_first_key_needs_no_confirmation_and_says_a_restart_is_needed(tmp_path):
    """ยังไม่มีกุญแจ = ไม่มีใครพัง ไม่ต้องถาม · แต่ต้องบอกว่าตัวที่รันอยู่ยังใช้ของเดิม
    ไม่งั้นคนตั้งเสร็จแล้วงงว่าทำไม client ยังเข้าได้/เข้าไม่ได้"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await open();
        await click('button[data-act="key:new"][data-slug="demo"]');
        console.log(JSON.stringify({ posted: H.posted, asked: H.confirms, text: text(), errors: H.errors }));
    """)
    assert out["posted"] == [{"force": False}]
    assert out["asked"] == [], "ยังไม่มีกุญแจ = ไม่มีใครพัง ไม่ควรถาม"
    assert KEY in out["text"], "ค่าเต็มต้องแสดงครั้งนี้ครั้งเดียว — คนเพิ่งกดสร้างต้องเอาไปใช้"
    assert "restart" in out["text"].lower()


def test_a_key_the_operator_already_has_can_be_pasted_in(tmp_path):
    """หลายไซต์มีกุญแจของตัวเองอยู่แล้ว (ออกจากระบบอื่น) — บังคับให้ใช้ของที่เราสุ่มให้
    แปลว่าเขาต้องไปแก้ client ทุกตัวเพื่อความสะดวกของเรา"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await open();
        panel().querySelector(".key-own").value = "  brought-from-elsewhere  ";
        await click('button[data-act="key:set"][data-slug="demo"]');
        console.log(JSON.stringify({ posted: H.posted,
                                     cleared: panel().querySelector(".key-own").value,
                                     errors: H.errors }));
    """)
    assert out["posted"] == [{"force": False, "key": "brought-from-elsewhere"}]
    assert out["cleared"] == "", "ช่องต้องถูกล้าง ไม่ใช่ทิ้งกุญแจไว้ในหน้า"


def test_removing_the_key_says_what_happens_next(tmp_path):
    """ลบแล้วเงียบคือกับดัก · หลัง restart bundle นั้นเสิร์ฟแบบเปิด ต้องอ่านเจอบนจอ"""
    (out,) = run_scenario(tmp_path, FLEET.replace("has_key: false", "has_key: true"), PROBE + """
        H.confirmAnswer = true;
        await open();
        await click('button[data-act="key:clear"][data-slug="demo"]');
        console.log(JSON.stringify({ text: text(), calls: seen(), errors: H.errors }));
    """)
    assert "เสิร์ฟแบบเปิด" in out["text"]
    assert any(c == "DELETE /api/models/demo/key" for c in out["calls"])
    # สถานะต้องถูกอ่านใหม่หลังลบ ไม่ใช่ค้างแสดง "ตั้งกุญแจแล้ว" ต่อไป
    assert sum(1 for c in out["calls"] if c == "GET /api/models/demo/key") >= 2


def test_the_audit_is_read_only_when_asked_for(tmp_path):
    """ไฟล์นี้โตได้ถึง 5 MB และหน้าตั้งค่าถูกเปิดทิ้งไว้ · ดึงเองทุกรอบคือการอ่านดิสก์ซ้ำ ๆ
    เพื่อของที่ไม่มีใครดู · และต้องแยก "ถูกปฏิเสธ" ออกมาได้ เพราะนั่นคือเหตุผลที่คนเปิดดู"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/settings");
        const before = seen().filter(c => c.includes("audit")).length;
        await click("#audit-load");
        const plain = seen().filter(c => c.includes("audit"));
        document.getElementById("audit-failed").checked = true;
        await click("#audit-load");
        const withFlag = seen().filter(c => c.includes("audit"));
        console.log(JSON.stringify({ before, plain, withFlag,
                                     text: document.getElementById("audit-out").textContent.replace(/\\s+/g, " ").trim(),
                                     msg: document.getElementById("audit-msg").textContent,
                                     errors: H.errors, alerts: H.alerts }));
    """)
    assert out["before"] == 0, "เปิดหน้าตั้งค่าแล้วต้องยังไม่อ่านไฟล์"
    assert len(out["plain"]) == 1 and "failed_only" not in out["plain"][0]
    assert "failed_only=true" in out["withFlag"][-1]
    assert "203.0.113.4" in out["text"] and "401" in out["text"]
    assert "1" in out["msg"], "ต้องสรุปว่ามีกี่รายการและถูกปฏิเสธกี่ครั้ง"
    assert out["errors"] == [] and out["alerts"] == []


def test_the_key_panel_does_not_touch_hidden(tmp_path):
    """`hidden` บนหน้านี้แปลว่า "สิทธิ์ (แอปคุม)" — เคยมีบั๊กสิทธิ์จากการเอาไปใช้สลับการแสดงผล"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/models");
        const before = hiddenSet();
        await click('button[data-act="key"][data-slug="demo"]');
        await click('button[data-act="key:new"][data-slug="demo"]');
        console.log(JSON.stringify({ before, after: hiddenSet(),
                                     mine: [...panel().querySelectorAll("[hidden]")].length,
                                     errors: H.errors }));
    """)
    assert out["mine"] == 0
    assert sorted(out["before"]) == sorted(out["after"])
