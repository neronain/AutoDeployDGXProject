"""ฟอร์มเปลี่ยน hostname บนการ์ดเครื่อง — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน
(tests/console_shell_dom.js)

หน้าเว็บคือ **ของหลัก** ของฟีเจอร์นี้ ไม่ใช่ของต่อยอดจาก CLI: เหตุผลทั้งหมดคือ "เข้าถึงเครื่องได้
ทางพอร์ต 8600 ทางเดียว" — ฟีเจอร์ที่มีแต่ CLI จึงแก้ปัญหานั้นไม่ได้เลย ลูกค้าเข้าไปพิมพ์คำสั่งไม่ได้

ชุดนี้ถามหน้าเว็บจริงว่า: กดปุ่มแล้วได้ฟอร์มที่ใช้งานจบในตัวไหม (เห็นชื่อปัจจุบัน · เห็นชื่อที่ถูกใช้
ไปแล้ว · ช่องรหัส sudo ขึ้นเฉพาะเครื่องที่ต้องใช้) · error อ่านรู้เรื่องไหม · รู้ไหมว่าสำเร็จหรือไม่ ·
สลับภาษาแล้วเปลี่ยนตามทั้งฟอร์ม · และ **ไม่แตะ `hidden` ของใครเลย** (บนหน้านี้ `hidden` แปลว่า
"สิทธิ์ (แอปคุม)" ไม่ใช่ "ซ่อนไว้ก่อน" — เคยมีบั๊กสิทธิ์จากการเอาไปใช้สลับการแสดงผล)
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

# spark-01 กับ spark-02 มี hostname เท่ากับชื่อในทะเบียน (fixture ของ harness) · hub ชื่อ "hub"
FLEET = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [] },
  { name: "spark-02", site: "TKC", models: [] },
  { name: "lonely", site: "" } ],
  localModels: [] };
H.fx = fx;
H.posted = [];
H.pre = { node: "spark-01", current: "spark1", new: "", valid: false, error: "",
          blockers: [], warnings: [], sudo_needed: true, backup_dir: "/root/lmds-hostname" };
H.renamed = { ok: true, changed: true, node: "spark-01", old: "spark1", new: "spark-7",
              rolled_back: false, blockers: [], warnings: [],
              steps: [{ node: "spark-01", step: "read the current hostname", ok: true, detail: "spark1", level: "pass" },
                      { node: "spark-01", step: "sudo password accepted", ok: true, detail: "", level: "pass" },
                      { node: "spark-01", step: "set the hostname to spark-7 + update /etc/hosts", ok: true,
                        detail: "previous files kept in /root/lmds-hostname", level: "pass" },
                      { node: "spark-01", step: "verify the new name resolves", ok: true, detail: "spark-7", level: "pass" }] };
H.routes = [
  [/^\\/api\\/nodes\\/[^/]+\\/rename-host/, (url, opts) => {
     if ((opts.method || "GET").toUpperCase() === "POST") { H.posted.push(JSON.parse(opts.body)); return H.renamed; }
     return H.pre;
  }],
  ...H.defaultRoutes(fx),
];
"""

# ชุดตรวจที่ใช้ซ้ำทุกเทส — "อะไรถูกตั้ง hidden ไว้บ้าง" ต้องเท่าเดิมตลอดทั้งเส้นทาง
PROBE = """
const hiddenSet = () => [...document.querySelectorAll("[hidden]")].map(el => el.id || el.className).sort();
const form = () => document.querySelector('#rh-name');
const body = () => document.querySelector('.machine .nbody').textContent.replace(/\\s+/g, " ").trim();
const click = async (sel) => { document.querySelector(sel).click(); await H.tick(8); };
const openFor = (name) => `button[data-nact="rename-host"][data-node="${name}"]`;
"""


def test_every_machine_card_offers_the_rename_and_the_form_is_self_contained(tmp_path):
    """กดปุ่มเดียวแล้วต้องได้ทุกอย่างที่ต้องใช้ในหน้าเดียว — ชื่อปัจจุบัน ช่องชื่อใหม่ ช่องรหัส sudo
    และรายชื่อที่ถูกใช้ไปแล้วทั้งฟลีต (กันซ้ำตั้งแต่ก่อนพิมพ์ ไม่ใช่เด้งกลับหลังกด)"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/nodes");
        const before = hiddenSet();
        const buttons = [...document.querySelectorAll('button[data-nact="rename-host"]')]
          .map(b => ({ node: b.dataset.node, text: b.textContent.trim() }));
        await click(openFor("spark-01"));
        console.log(JSON.stringify({
          buttons, before, after: hiddenSet(),
          text: body(),
          hasName: !!form(), hasPw: !!document.querySelector("#rh-pw"),
          chips: [...document.querySelectorAll(".nbody .tag")].map(t => t.textContent.trim()).sort(),
          focused: document.activeElement === form(),
          mine: [...document.querySelectorAll(".nbody [hidden]")].length,
          errors: H.errors, alerts: H.alerts,
        }));
    """)
    assert [b["node"] for b in out["buttons"]] == ["spark-01", "spark-02", "lonely"]
    assert {b["text"] for b in out["buttons"]} == {"Rename host"}
    assert out["hasName"] and out["hasPw"] and out["focused"]
    assert "spark1" in out["text"], "ต้องเห็นชื่อปัจจุบันของเครื่อง"
    # ความต่างระหว่าง "ชื่อของเครื่อง" กับ "ชื่อในทะเบียน" คือหัวใจของฟอร์มนี้ ต้องอยู่บนจอ ไม่ใช่ในเอกสาร
    assert "not its name in this registry" in out["text"] and "spark-01" in out["text"]
    assert "/etc/hosts" in out["text"], "ผู้ใช้ควรรู้ว่าปุ่มนี้แก้ /etc/hosts ให้ด้วย"
    # hostname ของเครื่องอื่นในฟลีต + ของ hub — ที่มาจากแคชในเบราว์เซอร์ ไม่ต้องยิง API เพิ่ม
    assert out["chips"] == ["hub", "lonely", "spark-02"]
    # `hidden` บนหน้านี้แปลว่า "สิทธิ์ (แอปคุม)" — ฟอร์มนี้ต้องไม่เอาไปใช้ซ่อน/แสดงของตัวเองเลย
    # · ที่เปลี่ยนไปได้อย่างเดียวคือกล่อง .nout ของแอปเองที่ showNodeOutput เปิดให้ (ทางเดียวกับ
    #   ปุ่ม setup/site ทุกปุ่ม) — ไม่มีใครถูกตั้ง hidden เพิ่ม และไม่มีของเราสักชิ้นที่ใช้มัน
    assert out["mine"] == 0, "อย่าใช้ hidden สลับการแสดงผลในฟอร์มนี้"
    assert set(out["after"]) <= set(out["before"])
    assert sorted(out["before"]) == sorted(out["after"] + ["nout"])
    assert out["errors"] == [] and out["alerts"] == []


def test_a_machine_with_passwordless_sudo_gets_no_password_field(tmp_path):
    """เคสจริง msi-5: NOPASSWD อยู่แล้ว — ช่องที่กรอกไปก็ไม่มีความหมายคือทางตัน ไม่ใช่ความรอบคอบ"""
    (out,) = run_scenario(tmp_path, FLEET.replace("sudo_needed: true", "sudo_needed: false"), PROBE + """
        await H.go("#/nodes");
        await click(openFor("spark-01"));
        console.log(JSON.stringify({ hasPw: !!document.querySelector("#rh-pw"), text: body(), errors: H.errors }));
    """)
    assert out["hasPw"] is False
    assert "does not ask for a password" in out["text"]


def test_a_blocked_machine_shows_the_reason_instead_of_a_form_that_cannot_work(tmp_path):
    """stacked ที่รันอยู่ = ปฏิเสธ · ต้องบอกเหตุผลและบอกว่ายังไม่ได้แตะอะไร
    ไม่ใช่ปล่อยให้พิมพ์ชื่อ กรอกรหัส แล้วค่อยล้ม"""
    blocked = FLEET.replace('blockers: [], warnings: [], sudo_needed: true',
                            'blockers: [{ kind: "stacked-running", text: "glm-4.6 is running stacked across '
                            'machines — stop it on the head first" }], warnings: [], sudo_needed: true')
    (out,) = run_scenario(tmp_path, blocked, PROBE + """
        await H.go("#/nodes");
        await click(openFor("spark-01"));
        console.log(JSON.stringify({ hasName: !!form(), hasGo: !!document.querySelector('[data-nact="rename-host-go"]'),
                                     text: body(), errors: H.errors }));
    """)
    assert out["hasName"] is False and out["hasGo"] is False
    assert "glm-4.6 is running stacked" in out["text"]
    assert "Nothing was changed on the machine" in out["text"]


def test_typing_a_name_that_cannot_work_says_so_before_the_button_is_pressed(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/nodes");
        await click(openFor("spark-01"));
        const type = async (v) => { form().value = v;
          form().dispatchEvent(new Event("input", { bubbles: true })); await H.tick(2);
          return document.querySelector("#rh-out").textContent.trim(); };
        console.log(JSON.stringify({
          fqdn: await type("spark1.example.com"),
          dash: await type("-nope"),
          long: await type("x".repeat(64)),
          taken: await type("spark-02"),
          good: await type("spark-7"),
          hidden: hiddenSet(), errors: H.errors,
        }));
    """)
    assert "FQDN" in out["fqdn"]
    assert "-" in out["dash"] and out["dash"]
    assert "63" in out["long"]
    assert "spark-02" in out["taken"] and "already used" in out["taken"]
    assert out["good"] == "", "ชื่อที่ใช้ได้ต้องไม่ค้างคำเตือนไว้"
    assert out["errors"] == []


def test_renaming_sends_the_name_and_password_then_shows_every_step_and_forgets_the_password(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/nodes");
        await click(openFor("spark-01"));
        form().value = "  SPARK-7 ";
        document.querySelector("#rh-pw").value = "s3cret";
        await click('[data-nact="rename-host-go"]');
        console.log(JSON.stringify({
          posted: H.posted, out: document.querySelector("#rh-out").textContent.replace(/\\s+/g, " ").trim(),
          pw: document.querySelector("#rh-pw").value,
          dom: document.querySelector(".machine").innerHTML.includes("s3cret"),
          hidden: hiddenSet(), errors: H.errors, alerts: H.alerts,
        }));
    """)
    assert out["posted"] == [{"hostname": "SPARK-7", "password": "s3cret"}]
    for step in ["read the current hostname", "sudo password accepted",
                 "set the hostname to spark-7 + update /etc/hosts", "verify the new name resolves"]:
        assert step in out["out"], step
    assert "this machine is now “spark-7”" in out["out"]
    assert out["pw"] == "" and out["dom"] is False, "รหัสผ่านต้องไม่ค้างอยู่ใน DOM หลังส่งไปแล้ว"
    assert out["errors"] == [] and out["alerts"] == []


def test_a_failed_rename_says_the_machine_was_put_back_not_just_failed(tmp_path):
    """"ไม่สำเร็จ" เฉย ๆ ทิ้งผู้ใช้ไว้กับคำถามว่าตอนนี้เครื่องชื่ออะไร — ซึ่งเขาไปดูเองไม่ได้"""
    failed = FLEET.replace("H.renamed = {", """H.renamed = { ok: false, changed: false, node: "spark-01",
      old: "spark1", new: "spark-7", rolled_back: true, blockers: [], warnings: [],
      steps: [{ node: "spark-01", step: "set the hostname to spark-7 + update /etc/hosts", ok: false,
                detail: "Could not set property: Connection timed out", level: "fail" },
              { node: "spark-01", step: "roll back to spark1", ok: true, detail: "", level: "warn" }] };
    H.unused = {""")
    (out,) = run_scenario(tmp_path, failed, PROBE + """
        await H.go("#/nodes");
        await click(openFor("spark-01"));
        form().value = "spark-7";
        document.querySelector("#rh-pw").value = "s3cret";
        await click('[data-nact="rename-host-go"]');
        console.log(JSON.stringify({ out: document.querySelector("#rh-out").textContent.replace(/\\s+/g, " ").trim(),
                                     hidden: hiddenSet(), errors: H.errors }));
    """)
    assert "Connection timed out" in out["out"], "ต้องเห็นเหตุผลดิบจากเครื่อง ไม่ใช่ข้อความรวม ๆ"
    assert "roll back to spark1" in out["out"]
    assert "the machine is still “spark1”" in out["out"]
    assert out["errors"] == []


def test_the_whole_form_follows_the_language_button_like_the_rest_of_the_page(tmp_path):
    """ปุ่มของฟีเจอร์นี้ต้องไม่เป็นจุดเดียวบนหน้าที่ไม่เปลี่ยนภาษาตามที่เหลือ"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.go("#/nodes");
        const en = document.querySelector(openFor("spark-01")).textContent.trim();
        document.getElementById("lang").click(); await H.tick(8);
        const th = document.querySelector(openFor("spark-01")).textContent.trim();
        await click(openFor("spark-01"));
        const formTh = body();
        const placeholder = form().getAttribute("placeholder");
        document.getElementById("lang").click(); await H.tick(8);
        console.log(JSON.stringify({ en, th, formTh, placeholder, back: hiddenSet(), errors: H.errors }));
    """)
    assert out["en"] == "Rename host" and out["th"] == "เปลี่ยน hostname"
    for thai in ["เปลี่ยน hostname ของ spark-01", "ชื่อตอนนี้", "hostname ใหม่", "รหัส sudo",
                 "ชื่อที่ถูกใช้ไปแล้วในฟลีตนี้"]:
        assert thai in out["formTh"], f"ฟอร์มยังไม่มี '{thai}'"
    # ศัพท์ต้องตรงกับ CLI: hostname ทับศัพท์ และ "ชื่อในทะเบียน" ต้องยังแยกออกจากกันได้ในภาษาไทย
    assert "ไม่ใช่ชื่อในทะเบียน" in out["formTh"]
    assert "ยาวไม่เกิน 63" in out["placeholder"]
    assert out["errors"] == []
