"""หน้าเว็บต้องหักหน่วยความจำที่เครื่องปลายทางใช้อยู่แล้ว ก่อนบอกว่าโมเดล fit

เคสจริง 2026-08-28 (msi-5) และรายงานจากฟลีต 2026-09-04: คนหนึ่งรัน vLLM อีกคนรัน
llama.cpp สองตัวบนเครื่องเดียว "แล้วทำงานได้ไม่เต็มที่" — ทุก deploy จากหน้าเว็บถูก
วางแผนจาก "เครื่องว่าง" เพราะหน้าเว็บไม่เคยส่ง reserved_gb ให้ analyzer
(analyzer รับได้ตั้งแต่ 2026-08-28 · CLI ส่ง แต่เฉพาะตอน target คือเครื่องตัวเอง)

เทสชุดนี้ยึดไว้ว่า:
  1. เลือกเครื่องในฟลีต → หักตามที่เครื่องนั้นถืออยู่จริง (อ่านจากแคช inventory ไม่ยิง SSH)
  2. preset ล้วนคือเครื่องสมมติ → ไม่หัก · ยังไม่มีข้อมูล → ไม่หักแต่ต้องบอก
  3. stacked หักตามเครื่องที่แน่นสุด × จำนวนเครื่อง (tensor parallel แบ่งเท่ากัน)
  4. payload มีครบสำหรับวาดแถบ capacity / ใช้อยู่แล้ว / weights / KV / เหลือ
  5. ความแน่นชั่วคราวไม่บล็อกการสร้าง bundle — แค่ติดป้าย "deploy ได้ แต่ start ตอนนี้ไม่ได้ ขาด X GB"
     (ที่ยังบล็อกคือโมเดลที่ใส่ไม่ได้แม้เครื่องเปล่า)
"""

from pathlib import Path

import pytest

from tests.test_generator import safetensors_report

GIB = 1024**3


def _seed(name: str, used_gb: float, total_gb: float = 128.0, running=()) -> None:
    """จำลองว่า refresher สำรวจเครื่องนี้มาแล้ว — รูปเดียวกับ `lmds agent info`"""
    from lmds.web import state

    state.STORE.set_node(name, {
        "host": {"memory_model": "unified",
                 "gpus": [{"name": "GB10", "vram_gb": total_gb, "vram_used_gb": used_gb}]},
        "models": [{"slug": r, "running": True} for r in running],
    })


@pytest.fixture
def thirty_gb_model(monkeypatch):
    rep = safetensors_report(weight_bytes=30 * GIB)
    monkeypatch.setattr("lmds.inspector.inspect_model", lambda source, client: rep)
    return rep


def _fit(**kw) -> dict:
    from lmds.web import deploy as dep

    return dep.analyze("Qwen/Qwen3-32B", no_llm=True, **kw)["plan"]["fit"]


def test_choosing_a_fleet_machine_subtracts_what_it_already_holds(thirty_gb_model):
    _seed("spark-02", 31.0)
    idle = _fit(target="dgx-spark-single")
    busy = _fit(target="dgx-spark-single", machine="spark-02")

    assert busy["reserved_gb"] == 31.0
    assert busy["reserved_source"] == "spark-02"
    # budget (คิดจากเครื่องเปล่า) ต้องเท่ากันทั้งสองกรณี — ความแน่นชั่วคราวไม่เปลี่ยนว่าสร้างได้ไหม
    assert busy["budget_gb"] == idle["budget_gb"]
    # แต่ชั้น "ตอนนี้" ต้องหักจริง
    assert busy["now_budget_gb"] == round(busy["budget_gb"] - 31.0, 1)
    assert busy["now_short_gb"] == 0.0  # 30 GB บนที่เหลือ 80+ GB ยัง start ได้
    assert any("start ตอนนี้ได้เลย" in n for n in busy["notes"])


def test_a_preset_without_a_machine_is_a_hypothetical_box_so_nothing_is_subtracted(thirty_gb_model):
    _seed("spark-02", 31.0)  # ฟลีตมีเครื่องแน่นอยู่ แต่ผู้ใช้ไม่ได้เลือกมัน
    fit = _fit(target="dgx-spark-single")
    assert fit["reserved_gb"] == 0.0
    assert fit["reserved_source"] == ""
    assert fit["now_verdict"] is None  # ไม่มีชั้น "ตอนนี้" สำหรับเครื่องสมมติ


def test_a_machine_with_no_inventory_yet_falls_back_to_full_capacity_and_says_so(thirty_gb_model):
    """เพิ่งแอดเครื่อง / ต่อไม่ได้ — ห้ามล้ม แต่ต้องเตือนว่าตัวเลขอาจสูงเกินจริง"""
    fit = _fit(target="dgx-spark-single", machine="spark-09")
    assert fit["reserved_gb"] == 0.0
    assert any("spark-09" in n and "คิดจากความจุเต็ม" in n for n in fit["notes"])


def test_an_idle_machine_is_reported_as_idle_not_as_unknown(thirty_gb_model):
    """สำรวจแล้วว่าง (0) กับยังไม่ได้สำรวจ (None) คนละความหมาย — โน้ตต้องไม่เตือนผิดเคส"""
    _seed("spark-02", 0.0)
    fit = _fit(target="dgx-spark-single", machine="spark-02")
    assert fit["reserved_gb"] == 0.0
    assert fit["reserved_source"] == "spark-02"
    assert not any("คิดจากความจุเต็ม" in n for n in fit["notes"])


def test_stacked_is_limited_by_the_busiest_member(thirty_gb_model):
    _seed("spark-head", 10.0)
    _seed("spark-worker", 40.0)
    fit = _fit(target="dgx-spark-stacked", machine="spark-head", worker="spark-worker")
    assert fit["reserved_gb"] == 80.0  # 40 × 2 เครื่อง — ไม่ใช่ 10 + 40
    assert fit["reserved_source"] == "spark-head + spark-worker"
    assert fit["now_budget_gb"] == round(fit["budget_gb"] - 80.0, 1)
    assert any("แน่นสุด" in n for n in fit["notes"])


def test_the_payload_carries_what_the_memory_bar_needs(thirty_gb_model):
    from lmds.web import deploy as dep

    _seed("spark-02", 31.0)
    plan = dep.analyze("Qwen/Qwen3-32B", target="dgx-spark-single", no_llm=True,
                       machine="spark-02")["plan"]
    f = plan["fit"]
    assert f["capacity_gb"] == 128.0
    assert f["kv_budget_gb"] == round(f["budget_gb"] - f["weights_gb"], 1)
    assert f["kv_at_context_gb"] == round(f["kv_bytes_per_token"] * plan["context"] / GIB, 1)
    for k in ("now_verdict", "now_budget_gb", "now_max_safe_context", "now_short_gb", "running_now"):
        assert k in f


def test_a_busy_machine_does_not_block_the_build_it_only_says_it_cannot_start_now(thirty_gb_model):
    """deploy = วาง bundle ไว้ก่อน · ของอื่นหยุดทีหลังได้ — ห้ามบล็อก แค่ต้องบอกให้ชัดว่าขาดเท่าไร

    ผู้ใช้ 2026-09-04: "จริงต้องทำได้ เพราะลูกค้าอาจจะยังไม่ได้รัน เพียงแต่ต้องการรู้ค่าและ
    deploy ลงไปก่อน" — เวอร์ชันแรกของ 0.5.2 โยน no-fit ทิ้งทั้งที่เครื่องเปล่าใส่ได้สบาย
    """
    _seed("spark-02", 100.0, running=("gemma-4-31b-gguf", "qwen3-8-27b-gguf"))
    fit = _fit(target="dgx-spark-single", machine="spark-02")   # ต้องไม่ raise

    assert fit["reserved_gb"] == 100.0
    assert fit["now_verdict"] in ("no-fit", "needs-smaller-quant")
    assert fit["now_short_gb"] > 0
    note = next(n for n in fit["notes"] if "deploy ได้ แต่ start ตอนนี้" in n)
    # บอกชื่อของที่ต้องหยุด ไม่ใช่แค่ "หยุดของอื่น"
    assert "gemma-4-31b-gguf@spark-02" in note and "qwen3-8-27b-gguf@spark-02" in note
    assert fit["running_now"] == ["gemma-4-31b-gguf@spark-02", "qwen3-8-27b-gguf@spark-02"]


def test_a_model_too_big_for_an_empty_machine_is_still_refused(monkeypatch):
    """ชั้นแรกยังต้องบล็อก — ฮาร์ดแวร์ใส่ไม่ได้จริง สร้าง bundle ไปก็ start ไม่ขึ้นตลอดกาล"""
    from lmds.web import deploy as dep

    monkeypatch.setattr("lmds.inspector.inspect_model",
                        lambda source, client: safetensors_report(weight_bytes=400 * GIB))
    _seed("spark-02", 0.0)
    with pytest.raises(dep.DeployError) as err:
        dep.analyze("Qwen/Huge", target="dgx-spark-single", no_llm=True, machine="spark-02")
    assert err.value.kind == "no-fit"
    assert "แม้เครื่องว่าง" in err.value.message


def test_a_tight_machine_offers_a_context_that_starts_now(thirty_gb_model):
    """พอ weights แต่ KV ไม่พอที่ context ที่แผนเสนอ → ต้องเสนอ context ที่ start ได้ทันที"""
    _seed("spark-02", 78.0, running=("other",))   # เหลือ ~35 GB: weights 30 + KV นิดหน่อย
    fit = _fit(target="dgx-spark-single", machine="spark-02")
    assert fit["now_verdict"] in ("fits", "fits-reduced-context")
    assert fit["now_max_safe_context"] is not None
    if fit["now_short_gb"] > 0:
        assert any(f"{fit['now_max_safe_context']:,}" in n for n in fit["notes"])


def test_the_api_forwards_the_chosen_machine(monkeypatch):
    from fastapi.testclient import TestClient

    from lmds.web.api import create_app

    seen: dict = {}

    def fake(model, **kw):
        seen.update(kw)
        return {"id": "x", "notes": [], "plan": {}}

    monkeypatch.setattr("lmds.web.deploy.analyze", fake)
    r = TestClient(create_app()).post(
        "/api/deploy/analyze",
        json={"model": "org/m", "machine": "spark-02", "worker": "spark-03"},
    )
    assert r.status_code == 200, r.text
    assert seen["machine"] == "spark-02" and seen["worker"] == "spark-03"


def test_the_page_sends_the_machine_and_draws_the_bar():
    page = (Path(__file__).resolve().parents[1] / "src" / "lmds" / "web" / "static" / "index.html"
            ).read_text(encoding="utf-8")
    body = page[page.index('api("/api/deploy/analyze"'):][:700]
    assert "machine:" in body and "worker:" in body, "หน้าเว็บยังไม่ส่งเครื่องปลายทางไปกับ analyze"
    assert 'id="w-mem"' in page and "already in use" in page, "ยังไม่มีแถบหน่วยความจำ"
    # โน้ตจาก fit (เช่น "ยังไม่มีข้อมูลหน่วยความจำของเครื่องนี้ คิดจากความจุเต็ม") ต้องถูกวาด —
    # เดิมหน้าเว็บทิ้ง f.notes ไปเงียบ ๆ ผู้ใช้จึงไม่รู้ว่าตัวเลขบนแถบมีเงื่อนไขอะไรพ่วง
    plan_view = page[page.index("function planView("):][:3000]
    assert "fitNotesView(f.notes)" in plan_view, "planView ยังไม่วาด f.notes"
    assert "Deployable, but cannot start right now" in page, "ยังไม่มีป้าย deploy ได้แต่ start ไม่ได้"
