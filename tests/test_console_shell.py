"""เทส shell ของคอนโซล (rail ซ้าย · router · การกรองการ์ดตาม route) โดยรัน JS จริงของหน้าเว็บ

ลูกค้า 2026-09-04: "กด site หรือ node ที่เมนูซ้ายแล้วตรงกลางว่าง ต้องกด All machines ก่อน"
บั๊กแบบนี้อยู่ใน *ลำดับเหตุการณ์* (route มาก่อนการ์ด · SSE วาดทับ · ไซต์ที่ยุบไว้) ซึ่งเทสแบบ
grep สตริงในไฟล์จับไม่ได้ · ที่นี่จึงบูตสคริปต์จริงของ index.html ใน node ด้วย DOM ย่อส่วน
(tests/console_shell_dom.js) แล้วถามว่าการ์ดไหน "มองเห็น" จริงตามกติกา hidden/style/CSS ที่หน้าใช้

ไม่มี node บนเครื่อง = ข้าม (เหมือน test_page_javascript_parses) — ไม่ใช่ผ่านเงียบ ๆ
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src/lmds/web/static/index.html"
HARNESS = Path(__file__).with_name("console_shell_dom.js")


def _node() -> str:
    found = shutil.which("node")
    if found:
        return found
    # เครื่องที่ติดตั้ง node ผ่าน nvm/volta/brew โดยไม่ได้อยู่ใน PATH ของ shell ที่รัน pytest
    home = Path(os.environ.get("REAL_HOME") or Path.home())
    for cand in [*sorted((home / ".nvm/versions/node").glob("*/bin/node"), reverse=True),
                 home / ".volta/bin/node", Path("/opt/homebrew/bin/node"), Path("/usr/local/bin/node")]:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    pytest.skip("ไม่มี node บนเครื่องนี้ — เทส shell ของหน้าเว็บต้องรัน JS จริง")


# ฟลีตตัวอย่าง: 2 ไซต์ + 1 เครื่องที่ยังไม่จัดไซต์ — ครบทุกกิ่งของ layoutSiteGroups/applyRouteFilter
FLEET = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [{ slug: "qwen3-8b", running: true, healthy: true, engine: "llamacpp", port: 8080, context: 32768, features: "tools" }] },
  { name: "spark-02", site: "TKC" },
  { name: "dgx-vs", site: "Veerasiam" },
  { name: "lonely", site: "" } ] };
H.fx = fx;
"""


def run_scenario(tmp_path: Path, prelude: str, body: str) -> list:
    """รัน scenario: `prelude` ก่อนบูตหน้า (ตั้ง fetch ปลอม/hash) · `body` หลังบูต · คืนบรรทัด JSON ที่พิมพ์"""
    script = tmp_path / "scenario.js"
    script.write_text(prelude + "\n// ---- boot ----\n" + body, encoding="utf-8")
    result = subprocess.run([_node(), str(HARNESS), str(PAGE), str(script)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"scenario failed:\n{result.stderr[-2000:]}\n--- stdout ---\n{result.stdout[-1500:]}"
    out = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            out.append(json.loads(line))
    return out


def test_harness_boots_the_real_page(tmp_path):
    """ถ้าบูตไม่ขึ้น (สคริปต์แตะ API ของเบราว์เซอร์ที่ DOM ย่อส่วนไม่มี) ทุกเทสข้างล่างจะล้มพร้อมกัน
    — แยกไว้ให้เห็นก่อนว่าพังที่ harness ไม่ใช่ที่หน้าเว็บ"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        console.log(JSON.stringify({ route: route.kind, rows: nodeRows.size, streams: H.streams.length,
                                     rail: document.getElementById("rail-nav").querySelectorAll("a").length,
                                     errors: H.errors, alerts: H.alerts }));""")
    assert out["route"] == "overview" and out["rows"] == 4 and out["streams"] == 1
    assert out["rail"] > 10 and out["errors"] == [] and out["alerts"] == []


def test_site_route_chosen_before_the_cards_exist(tmp_path):
    """เปิดหน้าด้วย #/site/TKC ค้างไว้ (bookmark/refresh) — route ถูกตัดสินตอน #nodes ยังว่าง
    การ์ดที่มาทีหลังต้องถูกกรองตาม route ไม่ใช่โผล่มาทั้งฟลีตหรือไม่โผล่เลย"""
    (out,) = run_scenario(tmp_path, FLEET + 'H.routes = H.defaultRoutes(fx); location.hash = "#/site/TKC";', """
        await H.tick();
        const first = H.visibleMachines();
        await H.go("#/node/dgx-vs");
        console.log(JSON.stringify({ site: first, node: H.visibleMachines() }));""")
    assert out == {"site": ["spark-01", "spark-02"], "node": ["dgx-vs"]}


def test_site_route_while_the_machine_list_is_still_loading(tmp_path):
    """กด site ที่ rail ระหว่าง /api/nodes ยังไม่ตอบ (SSH ช้า) — พอรายชื่อมาถึง การ์ดของไซต์นั้น
    ต้องขึ้น และ section ของไซต์อื่นต้องถูกซ่อน ไม่ใช่รอให้ผู้ใช้กด All machines ก่อน"""
    (out,) = run_scenario(tmp_path, FLEET + """
        let release; const gate = new Promise(r => release = r);
        H.routes = H.defaultRoutes(fx).map(([p, h]) => p === "/api/nodes" ? [p, async (...a) => { await gate; return h(...a); }] : [p, h]);
        H.release = release;""", """
        H.assert(nodeRows.size === 0, "cards must not exist yet");
        await H.go("#/site/TKC");
        H.release(); await H.tick(20);
        const secs = document.querySelectorAll("#nodes .sitesec").map(s => [s.querySelector(".sitehdr").dataset.site, s.hidden]);
        console.log(JSON.stringify({ visible: H.visibleMachines(), secs }));""")
    assert out["visible"] == ["spark-01", "spark-02"]
    assert sorted(out["secs"]) == [["", True], ["TKC", False], ["Veerasiam", True]]


def test_node_route_inside_a_collapsed_site_still_shows_the_card(tmp_path):
    """บั๊กของลูกค้า: ยุบไซต์ TKC ไว้ในหน้า All machines แล้วกด spark-01 ที่ rail → ตรงกลางว่าง
    เพราะการ์ดถูก style.display=none จากการยุบ ส่วน applyRouteFilter ปลดแค่ hidden
    route ต้องชนะการยุบ: มาดูเครื่องนี้ = ต้องเห็นเครื่องนี้"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        await H.go("#/nodes");
        document.querySelector('#nodes .sitehdr[data-site="TKC"]').click(); await H.tick();
        const collapsed = H.visibleMachines();
        await H.go("#/node/spark-01");
        const node = H.visibleMachines();
        const compact = nodeRows.get("spark-01").block.classList.contains("ncompact");
        await H.go("#/site/TKC");
        const site = H.visibleMachines();
        await H.go("#/nodes");
        console.log(JSON.stringify({ collapsed, node, compact, site, all: H.visibleMachines(), collapsedSites: [...collapsedSites] }));""")
    assert out["collapsed"] == ["dgx-vs", "lonely"], "การยุบไซต์ในหน้า All machines ยังต้องทำงาน"
    assert out["node"] == ["spark-01"] and out["compact"] is False
    assert out["site"] == ["spark-01", "spark-02"]
    assert out["all"] == ["spark-01", "spark-02", "dgx-vs", "lonely"], "กลับมา All machines ไซต์ที่ route กางให้ต้องยังกางอยู่"
    assert out["collapsedSites"] == []


def test_relayout_on_a_site_page_keeps_other_sites_hidden(tmp_path):
    """layoutClusterGroups สร้าง .sitesec/.gwrap ใหม่หมด (กดหัวไซต์ · ลากการ์ด · revealNode)
    เดิมมีแค่ loadCluster ที่กรองต่อ — ทางอื่นทำให้ไซต์อื่นทะลักกลับมาบนหน้า #/site/TKC"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        await H.go("#/site/TKC");
        layoutClusterGroups(); await H.tick();
        const afterLayout = document.querySelectorAll("#nodes .sitesec").filter(H.visible).map(s => s.querySelector(".sitehdr").dataset.site);
        revealNode("spark-02"); await H.tick();
        const afterReveal = document.querySelectorAll("#nodes .sitesec").filter(H.visible).map(s => s.querySelector(".sitehdr").dataset.site);
        const headerHidden = !H.visible(document.querySelector('#nodes .sitehdr[data-site="TKC"]'));
        console.log(JSON.stringify({ afterLayout, afterReveal, visible: H.visibleMachines(), headerHidden }));""")
    assert out["afterLayout"] == ["TKC"] and out["afterReveal"] == ["TKC"]
    assert out["visible"] == ["spark-01", "spark-02"]
    assert out["headerHidden"], "หน้าไซต์มี section เดียว หัวไซต์ (พร้อมปุ่มยุบ) ต้องไม่โผล่ให้กดจนหน้าว่าง"


def test_transient_node_list_failure_keeps_the_cards(tmp_path):
    """/api/nodes ล้มชั่วคราว (SQLite ล็อก) — เดิมล้าง #nodes ทั้งกล่องแต่ nodeRows ยังชี้การ์ดเดิม
    รอบถัดไป layout ดึงการ์ดกลับมาต่อท้ายข้อความ error ที่ค้างอยู่จนกว่าจะ reload"""
    (out,) = run_scenario(tmp_path, FLEET + """
        let fail = false;
        H.routes = H.defaultRoutes(fx).map(([p, h]) => p === "/api/nodes"
          ? [p, (...a) => fail ? { status: 500, body: { detail: "database is locked" } } : h(...a)] : [p, h]);
        H.setFail = v => { fail = v; };""", """
        await H.go("#/nodes");
        H.setFail(true); await refreshNodes(); await H.tick();
        const box = document.getElementById("nodes");
        const during = { visible: H.visibleMachines(), banner: !!box.querySelector(".nodes-err"),
                         text: box.querySelector(".nodes-err")?.textContent.replace(/\\s+/g, " ").trim() };
        H.setFail(false); await refreshNodes(); await H.tick();
        console.log(JSON.stringify({ during, after: { visible: H.visibleMachines(), banner: !!box.querySelector(".nodes-err") } }));""")
    assert out["during"]["visible"] == ["spark-01", "spark-02", "dgx-vs", "lonely"]
    assert out["during"]["banner"] and "database is locked" in out["during"]["text"]
    assert "Could not read the machine list" in out["during"]["text"], "ข้อความที่ผู้ใช้เห็นต้องเป็นอังกฤษ"
    assert out["after"] == {"visible": ["spark-01", "spark-02", "dgx-vs", "lonely"], "banner": False}


def test_forgotten_machine_disappears_from_overview_and_rail(tmp_path):
    """กด Forget แล้ว hub ลืมเครื่องนั้น แต่ lastNodeRegistry ฝั่งเบราว์เซอร์ไม่เคยถูกตัด —
    ภาพรวมยังโชว์ "spark-02 unreachable" และตารางไซต์ยังนับมันอยู่จนกว่าจะ reload"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        await H.go("#/overview");
        H.assert(document.getElementById("ov").innerHTML.includes("spark-02"), "overview lists spark-02 before forget");
        H.fx.nodes.splice(1, 1);
        await refreshNodes(); await H.tick();
        routeRender(); renderOverview(true);
        console.log(JSON.stringify({ registry: lastNodeRegistry.has("spark-02"), data: lastNodeData.has("spark-02"),
          overview: document.getElementById("ov").innerHTML.includes("spark-02"),
          rail: document.getElementById("rail-nav").innerHTML.includes("spark-02") }));""")
    assert out == {"registry": False, "data": False, "overview": False, "rail": False}


def test_live_frames_keep_the_route_filter_and_the_rail_dom(tmp_path):
    """SSE ทุก ~1 วิ วาด body ของการ์ดใหม่และวาด rail ใหม่ — การกรองตาม route ต้องคงอยู่
    และ rail ที่เนื้อหาเหมือนเดิมต้องไม่ถูกสร้าง element ใหม่ (hover/focus หลุด · คลิกหาย)"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        await H.go("#/site/TKC");
        const nav = document.getElementById("rail-nav");
        const before = nav.firstElementChild;
        for (let i = 0; i < 5; i++) { H.sse(H.snapshot(H.fx)); await H.tick(2); }
        const site = { visible: H.visibleMachines(), sameRail: nav.firstElementChild === before,
                       on: document.querySelector("#rail-nav a.on")?.getAttribute("href"),
                       body: nodeRows.get("spark-01").body.innerHTML.includes("qwen3-8b") };
        await H.go("#/node/spark-01");
        H.sse(H.snapshot(H.fx)); await H.tick(2);
        console.log(JSON.stringify({ site, node: H.visibleMachines(), compact: nodeRows.get("spark-01").block.classList.contains("ncompact") }));""")
    assert out["site"] == {"visible": ["spark-01", "spark-02"], "sameRail": True, "on": "#/site/TKC", "body": True}
    assert out["node"] == ["spark-01"] and out["compact"] is False


def test_unknown_node_or_site_route_explains_and_offers_a_way_back(tmp_path):
    """bookmark เก่า / เครื่องที่ถูกลืมหรือเปลี่ยนชื่อ — หน้าว่างเฉย ๆ แยกไม่ออกจากบั๊กข้างบน"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        const note = () => document.querySelector("#nodes .route-empty")?.textContent.replace(/\\s+/g, " ").trim() ?? null;
        await H.go("#/node/ghost");
        const node = { visible: H.visibleMachines(), note: note(), link: !!document.querySelector('#nodes .route-empty a[href="#/nodes"]') };
        await H.go("#/site/Nowhere");
        const site = note();
        await H.go("#/nodes");
        console.log(JSON.stringify({ node, site, gone: note() === null, all: H.visibleMachines() }));""")
    assert out["node"]["visible"] == [] and out["node"]["link"]
    assert "No machine named ghost" in out["node"]["note"]
    assert "No machines in site Nowhere" in out["site"]
    assert out["gone"] and out["all"] == ["spark-01", "spark-02", "dgx-vs", "lonely"]


def test_names_with_spaces_slashes_and_html_survive_the_router(tmp_path):
    """ชื่อเครื่อง/ไซต์ที่มีช่องว่าง จุด / และอักขระ HTML ต้องผ่าน encode→hash→decode→data-attr
    ได้ครบ และต้องไม่กลายเป็น element (ชื่อไซต์มาจาก CLI ของผู้ใช้ ไม่ใช่จากโค้ด)"""
    (out,) = run_scenario(tmp_path, """
        const fx = { nodes: [ { name: "spark 01.lab", site: "Bangkok HQ / floor 2" }, { name: "b&b<x>", site: 'R&D "west"' } ] };
        H.routes = H.defaultRoutes(fx);""", """
        const links = document.getElementById("rail-nav").querySelectorAll("a").map(a => a.getAttribute("href"));
        await H.go(links.find(h => h.startsWith("#/site/")));
        const site = { route, visible: H.visibleMachines() };
        await H.go(links.find(h => h.includes("b%26b")));
        console.log(JSON.stringify({ links: links.filter(h => /site|node/.test(h)), site, node: { route, visible: H.visibleMachines() },
          title: document.getElementById("nodes-title").textContent,
          injected: document.getElementById("nodes").querySelector("x") !== null || document.getElementById("rail-nav").querySelector("x") !== null }));""")
    assert "#/site/Bangkok%20HQ%20%2F%20floor%202" in out["links"] and "#/node/b%26b%3Cx%3E" in out["links"]
    assert out["site"] == {"route": {"kind": "site", "site": "Bangkok HQ / floor 2"}, "visible": ["spark 01.lab"]}
    assert out["node"] == {"route": {"kind": "node", "name": "b&b<x>"}, "visible": ["b&b<x>"]}
    assert out["title"] == "b&b<x>" and out["injected"] is False


def test_all_machines_route_is_unchanged(tmp_path):
    """#/nodes คือหน้าเดิมก่อนมี shell: ทุกไซต์ ทุกการ์ด หัวไซต์กดยุบได้ ไม่มี note เกิน"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        await H.go("#/site/TKC"); await H.go("#/nodes");
        const box = document.getElementById("nodes");
        console.log(JSON.stringify({ visible: H.visibleMachines(), title: document.getElementById("nodes-title").textContent,
          headers: box.querySelectorAll(".sitehdr").filter(H.visible).map(h => h.dataset.site),
          labels: box.querySelectorAll(".sitelbl").map(h => h.textContent), counts: box.querySelectorAll(".sitecount").map(h => h.textContent),
          note: !!box.querySelector(".route-empty"), cls: box.className }));""")
    assert out["visible"] == ["spark-01", "spark-02", "dgx-vs", "lonely"] and out["title"] == "Other machines"
    assert sorted(out["headers"]) == ["", "TKC", "Veerasiam"]
    assert out["labels"] == ["TKC", "Veerasiam", "Unassigned site"], "ป้ายไซต์ต้องเป็นอังกฤษและตรงกับ rail"
    assert out["counts"] == ["2 machines", "1 machine", "1 machine"]
    assert out["note"] is False and out["cls"] == ""


def test_poll_fallback_refreshes_the_other_machines_too(tmp_path):
    """SSE หลุด (proxy ตัด connection ยาว) → poll ทุก 5 วิ — เดิมดึงแค่ /api/host + /api/models
    การ์ดเครื่องอื่น/จุดสีใน rail/ภาพรวม แช่แข็งเงียบ ๆ ตลอดที่ SSE ไม่กลับมา โดยไม่มีอะไรบอก"""
    (out,) = run_scenario(tmp_path, FLEET + "H.routes = H.defaultRoutes(fx);", """
        H.streams[0].onerror();
        const polling = pollTimer != null;
        H.fx.nodes[1].models = [{ slug: "llama-70b", running: true, healthy: true, engine: "vllm", port: 8000, context: 8192, features: "" }];
        H.calls.length = 0;
        await pollNodesFallback(); await H.tick();
        const inventory = H.calls.filter(c => c.url.includes("/inventory")).map(c => c.url);
        const body = nodeRows.get("spark-02").body.innerHTML.includes("llama-70b");
        H.sse(H.snapshot(H.fx)); await H.tick();
        console.log(JSON.stringify({ polling, inventory, body, cleared: pollTimer == null }));
        clearInterval(pollTimer);""")
    assert out["polling"] and out["body"], "poll ต้องเห็นโมเดลใหม่บนเครื่องอื่น"
    assert sorted(out["inventory"]) == sorted(f"/api/nodes/{n}/inventory" for n in ["spark-01", "spark-02", "dgx-vs", "lonely"]), \
        "ต้องอ่านจากแคช (ไม่มี refresh=true) ไม่งั้น hub จะ SSH ทุกเครื่องทุก 5 วิ"
    assert out["cleared"], "SSE กลับมาแล้วต้องเลิก poll"

