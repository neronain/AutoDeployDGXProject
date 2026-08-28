"""กราฟของเครื่องนี้ต้องไม่ถูกถ่วงด้วยการ SSH ไปหาเครื่องอื่น

ผู้ใช้รายงาน 2026-08-28: "การแสดงผลจะช้ากว่าเครื่องหลายวิเลย กว่าจะเห็นกราฟขึ้น"

เดิม refresher เป็นลูปเดียวที่ไล่ probe ทีละเครื่องแบบเรียงคิว และ `_refresh_local()`
อยู่หัวลูปเดียวกัน · บน Tailscale relay การ probe หนึ่งเครื่องกินเวลาระดับวินาที
14 เครื่องจึงกลายเป็นรอบละหลายสิบวินาที แล้วค่า LOCAL_INTERVAL=3 ก็ไม่มีความหมาย
เพราะกว่าลูปจะวนกลับมาถึงบรรทัดแรกต้องรอ SSH ครบทุกเครื่องก่อน

เทสนี้จำลอง node ที่ช้าจริง ๆ แล้ววัดว่าเครื่องนี้ยังรีเฟรชได้ต่อเนื่อง
"""

import threading
import time

from lmds.web import state


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


def _run_loop_for(monkeypatch, seconds: float, node_delay: float, node_count: int):
    """รัน _loop จริงชั่วครู่ โดยแทนที่ทั้งการอ่านทะเบียนและการ probe ด้วยของปลอม"""
    names = [f"n{i}" for i in range(node_count)]
    monkeypatch.setattr("lmds.nodes.load", lambda: [_Node(n) for n in names], raising=False)

    local_calls: list[float] = []
    peak = 0
    live = 0
    lock = threading.Lock()

    def fake_local() -> None:
        local_calls.append(time.monotonic())
        state.STORE.set_local({"host": {}, "models": []})

    def fake_node(name: str) -> None:
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(node_delay)
        state.STORE.set_node(name, {"host": {}})
        with lock:
            live -= 1

    monkeypatch.setattr(state, "_refresh_local", fake_local)
    monkeypatch.setattr(state, "_refresh_node", fake_node)

    stop = threading.Event()
    thread = threading.Thread(target=state._loop, args=(stop,), daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join(timeout=10)
    return local_calls, peak


def test_local_refresh_keeps_its_cadence_while_nodes_are_slow(monkeypatch):
    """เครื่องนี้ครบกำหนดทุก 3 วิ — node ที่ช้าต้องไม่ทำให้เลยกำหนดไปด้วย"""
    monkeypatch.setattr(state, "LOCAL_INTERVAL", 0.2)
    state.STORE.__init__()  # แคชสะอาดต่อเทส
    state.STORE._local.interval = 0.2

    # 8 เครื่อง เครื่องละ 1 วิ · แบบเรียงคิวจะกินเวลา 8 วิต่อรอบ — ใน 3.5 วินาที
    # เครื่องนี้จะได้รีเฟรชแค่ครั้งเดียว (ครั้งแรกก่อนเข้าคิว) แล้วเงียบยาว
    local_calls, peak = _run_loop_for(monkeypatch, seconds=3.5, node_delay=1.0, node_count=8)

    # ลูปเดินทุก 1 วิ (stop.wait(1.0)) นั่นคือเพดานจริงของความถี่ — ที่ต้องพิสูจน์คือ
    # มันเดินตามเพดานนั้นได้ ไม่ใช่ถูกถ่วงจนเหลือรอบละหลายวิ
    assert len(local_calls) >= 3, f"เครื่องนี้รีเฟรชแค่ {len(local_calls)} ครั้งใน 3.5 วิ"
    gaps = [b - a for a, b in zip(local_calls, local_calls[1:])]
    assert max(gaps) < 1.5, f"มีช่วงที่เครื่องนี้เงียบไป {max(gaps):.1f} วิ"
    assert peak > 1, "probe ยังเรียงคิวอยู่ ไม่ได้ทำพร้อมกัน"


def test_a_slow_node_is_not_queued_again_while_it_is_still_being_probed(monkeypatch):
    """`due()` ยังเป็นจริงจนกว่าผลจะเขียนกลับ — ห้ามสั่งซ้ำจนคิวเต็มด้วยเครื่องเดียว"""
    monkeypatch.setattr(state, "LOCAL_INTERVAL", 60.0)
    state.STORE.__init__()

    calls: list[float] = []
    lock = threading.Lock()

    def fake_node(name: str) -> None:
        with lock:
            calls.append(time.monotonic())
        time.sleep(1.5)
        state.STORE.set_node(name, {"host": {}})

    monkeypatch.setattr(state, "_refresh_local", lambda: None)
    monkeypatch.setattr(state, "_refresh_node", fake_node)
    monkeypatch.setattr("lmds.nodes.load", lambda: [_Node("slow")], raising=False)

    stop = threading.Event()
    thread = threading.Thread(target=state._loop, args=(stop,), daemon=True)
    thread.start()
    time.sleep(1.2)          # ลูปวนทุก 1 วิ — ถ้าไม่กันซ้ำจะได้ 2 ครั้งขึ้นไป
    stop.set()
    thread.join(timeout=10)

    assert len(calls) == 1, f"สั่ง probe ซ้ำ {len(calls)} ครั้งทั้งที่รอบก่อนยังไม่จบ"
