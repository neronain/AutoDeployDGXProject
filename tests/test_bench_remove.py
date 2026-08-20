"""ผลวัดต้องลบได้ — ไม่งั้นตารางคะแนนบวมจนอ่านไม่ไหว

วัดซ้ำเป็นเรื่องปกติ (ก่อน/หลังเปลี่ยน flag, ก่อน/หลังอัปเกรด engine) แล้วไม่มีใคร
กลับไปลบไฟล์เอง — เก็บได้อย่างเดียวคือออกแบบให้พังตามเวลา
"""

import json

from lmds.bench import store


def _seed(tmp_path, slug, stamps):
    directory = tmp_path / slug
    directory.mkdir(parents=True)
    for stamp in stamps:
        (directory / f"{stamp}.json").write_text(
            json.dumps({"slug": slug, "stamped_at": stamp, "workloads": [], "probes": []}),
            encoding="utf-8")
    return directory


def test_removes_every_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "bench_root", lambda: tmp_path)
    _seed(tmp_path, "m1", ["20260820T090000", "20260820T100000"])
    assert store.remove("m1") == 2
    assert store.runs_for("m1") == []


def test_keep_last_leaves_the_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "bench_root", lambda: tmp_path)
    _seed(tmp_path, "m1", ["20260820T090000", "20260820T100000", "20260820T110000"])
    assert store.remove("m1", keep_last=1) == 2
    left = store.runs_for("m1")
    assert len(left) == 1 and "110000" in left[0].name


def test_empty_folder_is_cleaned_up(tmp_path, monkeypatch):
    """โฟลเดอร์ว่างที่ค้างไว้ทำให้ตารางยังนับโมเดลนั้นอยู่ทั้งที่ไม่มีข้อมูลแล้ว"""
    monkeypatch.setattr(store, "bench_root", lambda: tmp_path)
    _seed(tmp_path, "m1", ["20260820T090000"])
    store.remove("m1")
    assert not (tmp_path / "m1").exists()
    assert store.all_runs() == []


def test_removing_what_is_not_there_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "bench_root", lambda: tmp_path)
    assert store.remove("ไม่มีอยู่") == 0


def test_console_labels_stay_english_like_the_rest_of_the_page():
    """ป้าย UI เป็นอังกฤษทั้งหน้า ไทยไว้ที่คำอธิบาย — หมวดคะแนนเคยหลุดเป็นไทยทั้งแผง"""
    from pathlib import Path

    import lmds.web as web

    page = (Path(web.__file__).parent / "static/index.html").read_text(encoding="utf-8")
    assert '<span class="sec-title">Model scores</span>' in page
    for thai_label in ('"ที่ context ยาวสุด"', '"ค่ากลางทุกงาน"', '"prompt สั้นสุด"',
                       'instructions: "ทำตามคำสั่ง"', 'statCard("บทบาท"'):
        assert thai_label not in page, f"ป้ายนี้ควรเป็นอังกฤษแล้ว: {thai_label}"
