"""แผงบนหน้าภาพรวมต้องไม่ขัดกันเอง — เลขในวงแหวนกับบรรทัดใต้มันต้องบวกกันลง

เจอจริงในภาพหน้าจอที่ใช้ทำเอกสาร (`docs/img/fleet.png`): วงแหวน "Running models by
engine" บอก **21 running** (llama.cpp 11 · vLLM 10) แต่บรรทัดข้าง ๆ บอก
**"29 bundles fleet-wide · 22 stopped"** — 21 + 22 = 43 ไม่ใช่ 29 และไทล์
"Models running" ก็บอกอีกเลขหนึ่ง

สาเหตุ: วงแหวนนับจาก `lastNodeData` ซึ่งเบราว์เซอร์ดึง inventory มาเองทีละการ์ด
ส่วนบรรทัดใต้มันเอา `models_total` กับ `models_running` จาก `/api/fleet/summary`
ซึ่งนับเฉพาะเครื่องที่มีข้อมูลใน snapshot แล้ว · ตอนคอนโซลเพิ่งเปิด เครื่องที่ยังไม่ถูก
probe ทำให้ฝั่ง server ต่ำกว่ามาก (วันที่ถ่ายรูปคือ 7 เทียบกับ 21)

**เลขทั้งสองไม่มีอันไหนผิด — ที่ผิดคือเอามาวางคู่กันเหมือนมาจากที่เดียวกัน**
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

# spark-01 รันสองตัวจากสามตัว · spark-02 รันหนึ่งจากสอง · รวม = รัน 3 จาก 5
# ฝั่ง server จงใจส่งเลขที่ "ตามหลัง" มา (นับได้แค่เครื่องเดียว) เหมือนตอนเพิ่งเปิดคอนโซล
FLEET = """const fx = { nodes: [
  { name: "spark-01", site: "TKC", models: [
      { slug: "a", engine: "llamacpp", running: true,  healthy: true },
      { slug: "b", engine: "vllm",     running: true,  healthy: true },
      { slug: "c", engine: "llamacpp", running: false, healthy: false } ] },
  { name: "spark-02", site: "TKC", models: [
      { slug: "d", engine: "vllm",     running: true,  healthy: true },
      { slug: "e", engine: "vllm",     running: false, healthy: false } ] } ],
  localModels: [] };
H.fx = fx;
H.summary = { machines: 3, online: 2, pending: 1, gpus: 4, vram_gb: 512,
              models_running: 1, models_healthy: 1, models_total: 3 };
H.routes = [
  ["/api/fleet/summary", () => H.summary],
  ...H.defaultRoutes(fx),
];
"""

PROBE = """
const ov = () => document.getElementById("ov").textContent.replace(/\\s+/g, " ").trim();
const ring = () => document.querySelector("#ov svg[role=img]").getAttribute("aria-label");
const tiles = () => document.getElementById("fleet-summary").textContent.replace(/\\s+/g, " ").trim();
"""


def test_the_ring_and_the_line_under_it_add_up(tmp_path):
    """วงแหวนกับคำบรรยายใต้มันต้องมาจากข้อมูลชุดเดียวกัน · running + stopped = total"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.tick(20);
        console.log(JSON.stringify({ ring: ring(), text: ov(), errors: H.errors }));
    """)
    assert "3 running" in out["ring"], out["ring"]
    # 5 bundle ทั้งหมด · รัน 3 · หยุด 2 — ทุกตัวนับจากข้อมูลเดียวกับที่วาดวงแหวน
    assert "5 bundles fleet-wide" in out["text"]
    assert "2 stopped" in out["text"]
    assert "3 bundles fleet-wide" not in out["text"], (
        "ห้ามเอา models_total ของ server มาวางคู่กับเลขที่เบราว์เซอร์นับเอง"
    )
    assert out["errors"] == []


def test_a_fleet_wide_total_says_when_it_is_still_incomplete(tmp_path):
    """`pending` ถูกแสดงบนการ์ด Machines อยู่แล้ว แต่การ์ด Models แสดงยอดรวมทั้งฟลีต
    เหมือนเป็นเลขสมบูรณ์ · คนอ่านไม่มีทางรู้ว่ามันยังนับไม่ครบ"""
    (out,) = run_scenario(tmp_path, FLEET, PROBE + """
        await H.tick(20);
        console.log(JSON.stringify({ tiles: tiles(), errors: H.errors }));
    """)
    # เอกพจน์ — "1 machines" อ่านแล้วสะดุด และนี่คือค่าที่พบบ่อยที่สุด (เครื่องเดียวที่ probe ไม่ทัน)
    assert "1 machine not counted yet" in out["tiles"], out["tiles"]
    assert "1 machines" not in out["tiles"]


def test_a_fully_probed_fleet_does_not_get_the_caveat(tmp_path):
    """คำเตือนที่ขึ้นตลอดเวลาคือคำเตือนที่ไม่มีใครอ่าน"""
    complete = FLEET.replace("pending: 1", "pending: 0")
    (out,) = run_scenario(tmp_path, complete, PROBE + """
        await H.tick(20);
        console.log(JSON.stringify({ tiles: tiles(), errors: H.errors }));
    """)
    assert "not counted yet" not in out["tiles"]
    assert "serving" in out["tiles"] and "deployed" in out["tiles"]


def test_a_fleet_with_no_data_yet_shows_no_invented_total(tmp_path):
    """ยังไม่มีข้อมูลสักเครื่อง = ไม่รู้ ไม่ใช่ศูนย์ · '0 bundles fleet-wide' คือคำตอบที่ผิด"""
    empty = FLEET.replace('{ slug: "a", engine: "llamacpp", running: true,  healthy: true },', "") \
                 .replace('{ slug: "b", engine: "vllm",     running: true,  healthy: true },', "") \
                 .replace('{ slug: "c", engine: "llamacpp", running: false, healthy: false }', "") \
                 .replace('{ slug: "d", engine: "vllm",     running: true,  healthy: true },', "") \
                 .replace('{ slug: "e", engine: "vllm",     running: false, healthy: false }', "")
    (out,) = run_scenario(tmp_path, empty, PROBE + """
        await H.tick(20);
        console.log(JSON.stringify({ text: ov(), errors: H.errors }));
    """)
    assert "bundles fleet-wide" not in out["text"]
    assert out["errors"] == []


def test_more_than_one_uncounted_machine_reads_as_plural(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET.replace("pending: 1", "pending: 3"), PROBE + """
        await H.tick(20);
        console.log(JSON.stringify({ tiles: tiles(), errors: H.errors }));
    """)
    assert "3 machines not counted yet" in out["tiles"], out["tiles"]
