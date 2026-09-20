"""เครื่องหลายการ์ด + ไซต์ของ hub เอง — สองบั๊กที่รายงานมาจากเครื่องจริง 2026-09-20

บั๊ก 1 — คอนโซลอ่าน `host.gpus[0]` ใบเดียวทุกที่ · เครื่องที่มี RTX 3060 12 GB สามใบจึงขึ้นว่า
"12 GB" และเครื่อง RTX 4000 หลายใบขึ้นความจุของใบเดียว ขณะที่ `lmds fit` บนเครื่องเดียวกันตอบ
ถูกว่า "เครื่องมี 36.0 GB" · เลขสองฝั่งขัดกันคือกรณีที่แย่ที่สุด — ผู้ใช้อ่านการ์ดแล้วสรุปว่า
โมเดลไม่ลง ทั้งที่ลงได้ · payload เป็น *ลิสต์* และไม่มีฟิลด์ผลรวม จึงรวมที่หน้าเว็บ (gpuTotals)
ให้รายละเอียดรายใบ (gpuCard) ยังอยู่ครบ

บั๊ก 2 — `site` เป็นฟิลด์ของ node ในทะเบียน แต่ hub ไม่มีแถวของตัวเองในทะเบียน · siteOfNode()
จึงคืน "" ให้ hub เสมอ แล้ว hub ไปกอง "Unassigned site" ตลอดกาล — ผู้ใช้ที่จัด node ไว้ไซต์ isit
ไม่มีทางดึง hub เข้าไปอยู่ด้วย · แก้ด้วยรูปแบบเดียวกับ `cluster.stack_self` ที่มีอยู่แล้ว
(ค่าของ hub เองที่เครื่องอื่นเก็บในทะเบียน → เก็บใน config.yaml)

เทส DOM รันสคริปต์จริงของ index.html ผ่าน tests/console_shell_dom.js เหมือนเทสหน้าเว็บตัวอื่น
"""

from __future__ import annotations

import json

import pytest

from tests.test_console_shell import run_scenario

# เครื่องจริงที่รายงานเข้ามา: RTX 3060 12 GB สามใบ = 36 GB — เลขเดียวกับที่ `lmds fit` ตอบ
THREE_3060 = [{"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": 1.5,
               "compute": "8.6", "tested": True, "utilization_pct": u, "temperature_c": t}
              for u, t in ((10, 45), (90, 71), (30, 52))]

# GB10 / unified — การ์ดรุ่นนี้ไม่รายงาน memory.used จริง ๆ (inventory.py เขียนกติกาไว้)
GB10 = [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": None, "compute": "12.1", "tested": True}]


def _totals(tmp_path, gpus) -> dict:
    """เรียก gpuTotals() ของหน้าเว็บตรง ๆ ด้วยลิสต์ที่กำหนด"""
    fleet = 'const fx = { nodes: [{ name: "n1", site: "TKC" }] };\nH.fx = fx;\nH.routes = H.defaultRoutes(fx);'
    (out,) = run_scenario(tmp_path, fleet,
                          f"console.log(JSON.stringify(gpuTotals({json.dumps(gpus)})));")
    return out


# ── bug 1: gpuTotals ────────────────────────────────────────────────────────

def test_three_identical_cards_add_up_instead_of_reporting_the_first(tmp_path):
    """เคสที่รายงานเข้ามา: 12+12+12 ต้องเป็น 36 ไม่ใช่ 12 · ป้ายต้องบอกด้วยว่ามีสามใบ"""
    t = _totals(tmp_path, THREE_3060)
    assert t["count"] == 3
    assert t["vram_gb"] == 36.0, "ผลรวมต้องตรงกับที่ lmds fit คิด (เครื่องมี 36.0 GB)"
    assert t["vram_used_gb"] == 4.5
    assert t["sameModel"] is True
    # ตัดแค่ "NVIDIA " นำหน้าเหมือนที่หน้าเว็บทำมาตลอด — ไม่ไปแต่งชื่อรุ่นเพิ่ม
    assert t["name"] == "3 × GeForce RTX 3060", "ป้ายต้องบอกจำนวนใบ ไม่ใช่ชื่อรุ่นลอย ๆ ที่อ่านเหมือนมีใบเดียว"


def test_a_card_that_does_not_report_used_vram_makes_the_total_null_not_a_partial_sum(tmp_path):
    """บวกบางใบคือโกหก — เกจจะดูว่างกว่าจริง · ต้องเป็น null เพื่อให้หน้าเว็บ *ซ่อน* ช่องนั้น

    กติกาเดียวกับที่ inventory.py เขียนไว้ตรง "temperature_c":
    None = การ์ดรุ่นนี้ไม่รายงาน หน้าเว็บต้องซ่อนช่องนั้น ไม่ใช่โชว์ 0
    """
    partial = [dict(THREE_3060[0]), dict(THREE_3060[1]), dict(THREE_3060[2], vram_used_gb=None)]
    t = _totals(tmp_path, partial)
    assert t["vram_used_gb"] is None, "มีใบเดียวที่ไม่รายงานก็ห้ามบวกที่เหลือมาโชว์"
    assert t["vram_gb"] == 36.0, "ความจุยังรวมได้ตามปกติ — คนละฟิลด์กับที่ใช้ไป"
    assert t["count"] == 3


def test_unified_memory_reports_no_used_vram_at_all(tmp_path):
    """GB10 คืน vram_used_gb: null ทุกใบ — เส้นทางนี้เกิดขึ้นจริงบน DGX Spark"""
    t = _totals(tmp_path, GB10)
    assert t["vram_gb"] == 128.0 and t["vram_used_gb"] is None
    assert t["count"] == 1 and t["name"] == "GB10", "ใบเดียวไม่ต้องมี '1 ×' นำหน้า"


def test_a_card_that_reports_no_capacity_at_all_still_counts_as_a_card(tmp_path):
    """ไม่รู้ความจุ ≠ ไม่มีการ์ด — เครื่องยังมี GPU อยู่ แค่บอกความจุไม่ได้"""
    t = _totals(tmp_path, [{"name": "NVIDIA Weird", "vram_gb": None, "vram_used_gb": None}])
    assert t["count"] == 1 and t["vram_gb"] is None and t["name"] == "Weird"


def test_mixed_models_in_one_box_do_not_claim_uniformity(tmp_path):
    """การ์ดคนละรุ่นในเครื่องเดียวมีจริง — ป้ายต้องไม่อ่านเหมือนเป็นรุ่นเดียวกันทั้งหมด"""
    mixed = [{"name": "NVIDIA RTX 4000 Ada Generation", "vram_gb": 20.0, "vram_used_gb": 2.0},
             {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "vram_used_gb": 1.0}]
    t = _totals(tmp_path, mixed)
    assert t["count"] == 2 and t["vram_gb"] == 32.0 and t["vram_used_gb"] == 3.0
    assert t["sameModel"] is False
    assert "2 ×" not in t["name"], "ห้ามนับ 2 ใบเป็นรุ่นเดียว"
    assert "RTX 4000 Ada Generation" in t["name"] and "RTX 3060" in t["name"]


def test_no_card_at_all_is_null_not_zero(tmp_path):
    """เครื่อง control-plane ไม่มี GPU — ต้องแยกออกจาก "มี GPU แต่ความจุ 0" """
    t = _totals(tmp_path, [])
    assert t == {"count": 0, "vram_gb": None, "vram_used_gb": None, "sameModel": True, "name": ""}


def test_rounding_does_not_leak_float_noise(tmp_path):
    """11.9 × 3 = 35.699999999999996 ถ้าไม่ปัด — ตัวเลขบนการ์ดต้องอ่านได้"""
    t = _totals(tmp_path, [{"name": "NVIDIA X", "vram_gb": 11.9, "vram_used_gb": 0.1} for _ in range(3)])
    assert t["vram_gb"] == 35.7 and t["vram_used_gb"] == 0.3


# ── bug 1: ไม่มีที่ไหนอ่าน gpus[0] เพื่อหาผลรวมอีกแล้ว ─────────────────────────

def test_the_console_no_longer_reads_only_the_first_gpu_for_totals():
    """กันไม่ให้ `[0]` กลับมาใหม่ — เจอบั๊กนี้ห้าจุดพร้อมกัน จึงต้องมีตัวจับที่ระดับไฟล์ด้วย

    อ่านรายละเอียด *รายใบ* ยังทำได้ (gpuCard, compute/tested ของใบแรก) แต่ต้องไม่มีที่ไหน
    หยิบ gpus[0] มาเป็นตัวแทนความจุของทั้งเครื่องอีก
    """
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html"
    text = page.read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in text.splitlines()
                 if "gpus" in ln and "[0]" in ln and not ln.lstrip().startswith("//")]
    # ใบแรกใช้ได้เฉพาะกับฟิลด์รายใบที่บวกกันไม่ได้ (compute/tested) และต้องเขียนกำกับไว้
    assert all("first" in ln for ln in offenders), (
        "ยังมีที่อ่าน gpus[0] เป็นตัวแทนทั้งเครื่อง:\n  " + "\n  ".join(offenders))
    assert "function gpuTotals(" in text, "ตัวรวมต้องอยู่ที่เดียว ไม่ใช่ inline ซ้ำ ๆ"


def test_every_call_site_asks_gpu_totals(tmp_path):
    """ห้าจุดที่เคยอ่านใบแรก: การ์ด hub · หัวหน้าเบนช์ · systemCard · ชิปตอนย่อ · แถบบนรางซ้าย"""
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "src/lmds/web/static/index.html"
    text = page.read_text(encoding="utf-8")
    assert text.count("gpuTotals(") >= 7, "ตัวรวมต้องถูกเรียกที่ทุกจุด ไม่ใช่แค่บางจุด"
    for fn in ("function renderHost(", "function benchDetailMarkup(", "function systemCard(",
               "function summaryMarkup(", "function nodeUsage("):
        body = text.split(fn, 1)[1][:2000]
        assert "gpuTotals(" in body, f"{fn} ยังไม่ได้รวมการ์ดทุกใบ"


# ── bug 1: หน้าจอจริง ────────────────────────────────────────────────────────

def _fleet(gpus, site="TKC") -> str:
    return (f'const fx = {{ nodes: [{{ name: "rtx-box", site: {json.dumps(site)}, '
            f'gpus: {json.dumps(gpus)} }}] }};\nH.fx = fx;\nH.routes = H.defaultRoutes(fx);')


def test_the_machine_card_head_and_rail_bar_speak_for_the_whole_box(tmp_path):
    """หัวการ์ดของเครื่อง และแถบ GPU บนรางซ้าย ต้องคิดจาก 36 GB ไม่ใช่ 12 GB

    ตัวเลขรายใบ (VRAM 1.5/12 GB) ยังต้องอยู่ในการ์ดรายใบข้างล่าง — ที่ห้ามคือเอาเลขของใบเดียว
    มาเป็นความจุของทั้งเครื่อง
    """
    (out,) = run_scenario(tmp_path, _fleet(THREE_3060), """
        H.sse(H.snapshot(H.fx)); await H.tick(6);
        const card = [...document.querySelectorAll("#nodes .machine")].find(m => /rtx-box/.test(m.textContent));
        const head = card.querySelector(".nbody .dim").textContent.replace(/\\s+/g, " ");
        console.log(JSON.stringify({ head, usage: nodeUsage("rtx-box"),
          perCard: [...card.querySelectorAll(".gpu .gpu-name")].map(g => g.textContent) }));""")
    assert out["usage"]["cap"] == 36.0, "แถบบนรางซ้ายยังคิดจากการ์ดใบเดียว"
    assert out["usage"]["used"] == 4.5
    assert out["usage"]["pct"] == 13   # 4.5 / 36
    assert "3 × GeForce RTX 3060" in out["head"], "หัวการ์ดต้องบอกว่ามีสามใบ"
    assert "36 GB" in out["head"], "ความจุบนหัวการ์ดต้องเป็นของทั้งเครื่อง"
    assert out["perCard"].count("NVIDIA GeForce RTX 3060") == 3, "รายละเอียดรายใบต้องยังอยู่ครบ"


def _gauges(tmp_path, host: dict) -> list:
    """ป้ายของทุกเกจที่ systemCard วาดให้ host ก้อนนี้"""
    fleet = 'const fx = { nodes: [{ name: "n1", site: "TKC" }] };\nH.fx = fx;\nH.routes = H.defaultRoutes(fx);'
    (out,) = run_scenario(tmp_path, fleet, f"""
        const box = document.createElement("div");
        box.innerHTML = systemCard({json.dumps(host)});
        console.log(JSON.stringify([...box.querySelectorAll(".gauge")]
          .map(g => g.textContent.replace(/\\\\s+/g, " ").trim())));""")
    return out


# host ที่มีของครบพอให้ systemCard วาดเกจทุกอัน (CPU/RAM/ดิสก์) ได้จริง
BOX = {"cpu": {"percent": 20, "cores": 16}, "ram_used_gb": 30, "ram_total_gb": 64,
       "disk_free_gb": 500, "disk_total_gb": 1000}


def test_the_vram_gauge_is_drawn_from_the_whole_box(tmp_path):
    labels = _gauges(tmp_path, {**BOX, "gpus": THREE_3060})
    vram = [l for l in labels if "VRAM" in l]
    assert vram, "เครื่องหลายใบต้องยังมีเกจ VRAM"
    assert "4.5 / 36 GB" in vram[0], f"เกจต้องคิดจากทั้งเครื่อง ไม่ใช่ใบแรก: {vram[0]}"
    assert "3 cards" in vram[0], "บอกด้วยว่าเลขนี้มาจากกี่ใบ"
    assert "13" in vram[0]   # 4.5 / 36


def test_a_partial_report_hides_the_vram_gauge_rather_than_drawing_a_wrong_one(tmp_path):
    """ใบหนึ่งไม่รายงาน = ไม่มีเกจ VRAM ดีกว่ามีเกจที่บอกว่าว่างกว่าจริง"""
    partial = [dict(THREE_3060[0]), dict(THREE_3060[1], vram_used_gb=None), dict(THREE_3060[2])]
    labels = _gauges(tmp_path, {**BOX, "gpus": partial})
    assert not [l for l in labels if "VRAM" in l], "ห้ามวาดเกจ VRAM จากผลบวกที่ไม่ครบ"
    assert [l for l in labels if "RAM" in l], "เกจอื่นต้องไม่หายไปด้วย"
    # แต่ความจุรวมยังรู้ — แถบบนรางซ้ายใช้ค่านี้ต่อได้
    assert _totals(tmp_path, partial)["vram_gb"] == 36.0


def test_unified_memory_still_has_no_separate_vram_gauge(tmp_path):
    """DGX Spark: GPU กินจาก pool เดียวกับ RAM — โชว์แยกอีกเกจคือนับซ้ำ · กติกานี้ต้องไม่หาย"""
    spark = [{"name": "NVIDIA GB10", "vram_gb": 128.0, "vram_used_gb": 40.0}] * 2
    labels = _gauges(tmp_path, {**BOX, "gpus": spark, "memory_model": "unified"})
    assert not [l for l in labels if "VRAM" in l], "unified ต้องไม่มีเกจ VRAM แยก — นับซ้ำกับ Unified"
    assert [l for l in labels if "Unified" in l], "ยังต้องมีเกจ Unified ตามเดิม"
    # เครื่อง discrete ที่มีการ์ดชุดเดียวกันต้องได้เกจ — พิสูจน์ว่าที่หายไปคือกติกา unified ไม่ใช่บั๊ก
    assert [l for l in _gauges(tmp_path, {**BOX, "gpus": spark}) if "VRAM" in l]


# ── bug 2: ไซต์ของ hub ──────────────────────────────────────────────────────

def test_the_hub_site_round_trips_through_config_and_shows_up_in_the_host_payload(monkeypatch):
    """เก็บที่ config.yaml (ไม่ใช่ทะเบียน) แล้วโผล่บน host payload ให้คอนโซลอ่าน"""
    from lmds.config import Settings

    assert Settings().cluster.site == "", "ค่าเริ่มต้นต้องเป็น 'ยังไม่จัดไซต์' ไม่ใช่เดาให้"

    s = Settings.load()
    s.cluster.site = "isit"
    s.save()
    assert Settings.load().cluster.site == "isit", "อ่านกลับจาก config.yaml ไม่ได้"
    assert Settings.load().cluster.stack_self is True, "ต้องไม่ไปทับค่า hub-only ตัวอื่น"

    import lmds.inventory as inventory

    payload = inventory.host_payload()
    assert payload["site"] == "isit", "host payload ต้องพา site ของ hub มาด้วย"

    # ลบป้ายออกได้
    s = Settings.load()
    s.cluster.site = ""
    s.save()
    assert inventory.host_payload()["site"] == ""


def test_a_broken_config_does_not_blank_the_whole_host_payload(monkeypatch, tmp_path):
    """config.yaml เสียแล้วต้องได้ "ไม่รู้ว่าอยู่ไซต์ไหน" ไม่ใช่เครื่องหายไปทั้งเครื่อง"""
    from lmds.config import SettingsError
    from lmds.config.paths import config_file

    import lmds.config.settings as settings_mod
    import lmds.inventory as inventory

    config_file().parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings_mod.Settings, "load",
                        classmethod(lambda cls: (_ for _ in ()).throw(SettingsError("พัง"))))
    payload = inventory.host_payload()
    assert payload["site"] == ""
    assert payload["hostname"], "ส่วนที่เหลือของ payload ต้องยังอยู่ครบ"


def test_the_self_settings_route_sets_the_site_without_touching_stack():
    """ต่อยอด PATCH /api/cluster/self ที่มีอยู่ — ไม่สร้าง endpoint ใหม่ให้ hub โดยเฉพาะ"""
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    from lmds.config import Settings
    from lmds.web.api import create_app

    client = TestClient(create_app())

    r = client.patch("/api/cluster/self", json={"site": "  isit  "})
    assert r.status_code == 200, r.text
    assert r.json() == {"stack": True, "site": "isit"}, "ต้องตัดช่องว่างหัวท้าย ไม่งั้นกลายเป็นคนละไซต์"
    assert Settings.load().cluster.site == "isit"

    # แก้ stack อย่างเดียวต้องไม่ล้างไซต์ทิ้ง
    assert client.patch("/api/cluster/self", json={"stack": False}).json() == {"stack": False, "site": "isit"}
    assert Settings.load().cluster.site == "isit"

    # ว่าง = เอาป้ายออก
    assert client.patch("/api/cluster/self", json={"site": ""}).json()["site"] == ""
    # ไม่ส่งอะไรมาเลยยังต้องเป็น 400 เหมือนเดิม
    assert client.patch("/api/cluster/self", json={}).status_code == 400


def test_the_hub_row_of_the_cluster_view_carries_its_site():
    """ไซต์เป็นคีย์หนึ่งของการจับกลุ่ม stacked — hub ที่ฝัง "" ตายตัวจับคู่กับใครไม่ได้เลย"""
    from pathlib import Path

    api = Path(__file__).resolve().parents[1] / "src/lmds/web/api.py"
    cli = Path(__file__).resolve().parents[1] / "src/lmds/cli/main.py"
    assert '"site": ""' not in api.read_text(encoding="utf-8"), "แถว hub ยังฝังไซต์ว่างไว้ตายตัว"
    assert '"site": ""' not in cli.read_text(encoding="utf-8")


def test_the_cli_has_a_home_for_the_hubs_own_site():
    """`lmds node set --site` ใช้กับ hub ไม่ได้ (ไม่มีในทะเบียน) — ต้องมีทางของมันเอง
    อยู่คู่กับ --self-stack ซึ่งเป็นค่า hub-only แบบเดียวกัน"""
    from typer.testing import CliRunner

    from lmds.cli.main import app

    runner = CliRunner()
    out = runner.invoke(app, ["node", "cluster", "--help"]).output.replace("\n", " ")
    assert "--self-site" in out, "ไม่มีคำสั่งไหนตั้งไซต์ให้ hub ได้"
    # `cluster show` เป็นชื่อพ้องของคำสั่งเดียวกันและเรียกฟังก์ชันนั้นตรง ๆ — ธงต้องมีครบทั้งคู่
    # ไม่งั้นอาร์กิวเมนต์ที่ไม่ได้ส่งจะกลายเป็น OptionInfo ของ typer แล้วพังตอนเอาไปใช้
    assert "--self-site" in runner.invoke(app, ["cluster", "show", "--help"]).output.replace("\n", " ")


def test_the_rail_puts_the_hub_in_its_site_next_to_the_node_at_the_same_place(tmp_path):
    """เรื่องที่ผู้ใช้รายงาน: จัด node ไว้ไซต์ isit แล้ว hub ต้องเข้าไปอยู่กลุ่มเดียวกันได้"""
    fleet = ('const fx = { nodes: [{ name: "spark-01", site: "isit" }, { name: "other", site: "HQ" }],\n'
             '             host: { hostname: "hub-box", site: "isit" } };\n'
             'H.fx = fx;\nH.routes = H.defaultRoutes(fx);')
    (out,) = run_scenario(tmp_path, fleet, """
        await H.tick(6);
        const nav = document.getElementById("rail-nav");
        const tree = nav.querySelector('.rtree[data-tree="isit"]');
        console.log(JSON.stringify({
          hubSite: siteOfHub(),
          group: [...tree.querySelectorAll("a")].map(a => a.getAttribute("href")),
          counts: [...nav.querySelectorAll(".rsite")].map(a =>
            [a.dataset.site, a.querySelector(".rcount").textContent]),
        }));""")
    assert out["hubSite"] == "isit"
    assert "#/hub" in out["group"], "hub ต้องอยู่ในกิ่งของไซต์ isit"
    assert "#/node/spark-01" in out["group"], "node เดิมต้องยังอยู่ที่เดิม"
    assert dict(out["counts"])["isit"] == "2", "นับ hub เป็นเครื่องหนึ่งของไซต์ด้วย"
    assert dict(out["counts"])["HQ"] == "1"


def test_an_unassigned_hub_keeps_falling_into_the_unassigned_group(tmp_path):
    """ยังไม่ได้ตั้งไซต์ = ยังอยู่ "Unassigned site" ตามเดิม — ไม่ใช่หายไปจากรางซ้าย"""
    fleet = ('const fx = { nodes: [{ name: "spark-01", site: "isit" }] };\n'
             'H.fx = fx;\nH.routes = H.defaultRoutes(fx);')
    (out,) = run_scenario(tmp_path, fleet, """
        await H.tick(6);
        const nav = document.getElementById("rail-nav");
        console.log(JSON.stringify({ hubSite: siteOfHub(),
          sites: [...nav.querySelectorAll(".rsite")].map(a => a.dataset.site),
          unassigned: [...(nav.querySelector('.rtree[data-tree=""]') || { querySelectorAll: () => [] })
            .querySelectorAll("a")].map(a => a.getAttribute("href")) }));""")
    assert out["hubSite"] == ""
    assert "" in out["sites"], "ต้องยังมีกลุ่ม Unassigned site"
    assert out["unassigned"] == ["#/hub"]


def test_a_site_holding_only_the_hub_does_not_say_it_has_no_machines(tmp_path):
    """หน้าไซต์โชว์เฉพาะการ์ดของ node (การ์ด hub อยู่คนละหน้า) — ไซต์ที่มีแต่ hub จึงไม่ใช่ไซต์ว่าง"""
    fleet = ('const fx = { nodes: [{ name: "spark-01", site: "HQ" }],\n'
             '             host: { hostname: "hub-box", site: "isit" } };\n'
             'H.fx = fx;\nH.routes = H.defaultRoutes(fx);\nlocation.hash = "#/site/isit";')
    (out,) = run_scenario(tmp_path, fleet, """
        await H.tick(8);
        const box = document.getElementById("nodes");
        console.log(JSON.stringify({ empty: !!box.querySelector(".route-empty"),
                                     hub: (box.querySelector(".route-hub") || {}).textContent || "" }));""")
    assert out["empty"] is False, "ไซต์ที่มี hub อยู่ต้องไม่ขึ้นว่า 'No machines in site'"
    assert "hub-box" in out["hub"] and "#/hub" not in out["hub"]


def test_hub_settings_can_set_the_site_from_the_console(tmp_path):
    """ทางตั้งค่าในหน้าเว็บ — node มีปุ่ม 🏷 Site บนการ์ด, hub มีช่องในหน้า Hub settings"""
    fleet = ('const fx = { nodes: [{ name: "spark-01", site: "isit" }],\n'
             '             host: { hostname: "hub-box", site: "" } };\n'
             'H.fx = fx;\nH.routes = H.defaultRoutes(fx);\n'
             'H.routes.push(["/api/cluster/self", (url, o) => ({ stack: true, site: JSON.parse(o.body).site })]);\n'
             'location.hash = "#/settings";')
    (out,) = run_scenario(tmp_path, fleet, """
        await H.tick(8);
        const inp = document.getElementById("hub-site");
        const picks = [...document.querySelectorAll("#hub-site-options option")].map(o => o.getAttribute("value"));
        inp.value = "isit";
        document.getElementById("hub-site-save").click();
        await H.tick(6);
        const tree = document.getElementById("rail-nav").querySelector('.rtree[data-tree="isit"]');
        console.log(JSON.stringify({
          picks, sent: H.calls.filter(c => c.url === "/api/cluster/self").map(c => JSON.parse(c.body)),
          hubSite: siteOfHub(),
          group: [...tree.querySelectorAll("a")].map(a => a.getAttribute("href")) }));""")
    assert out["picks"] == ["isit"], "ต้องเสนอไซต์ที่มีอยู่แล้วให้เลือก ไม่ใช่ให้พิมพ์ใหม่ทุกครั้ง"
    assert out["sent"] == [{"site": "isit"}], "ต้องยิงที่ PATCH /api/cluster/self เดิม"
    assert out["hubSite"] == "isit"
    assert "#/hub" in out["group"] and "#/node/spark-01" in out["group"], \
        "กดบันทึกแล้วรางซ้ายต้องย้าย hub เข้ากลุ่มทันที ไม่ต้องรีเฟรช"
