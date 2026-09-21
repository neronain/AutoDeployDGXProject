"""liveness watchdog — ยิง generate จริง แล้ว restart ให้เมื่อไม่ตอบ (UPGRADE-2026-09 §1.9 ข้อ 2)

`/health` ตอบ 200 ได้ทั้งที่ rank ค้าง · ผู้ดูแลเห็นไฟเขียว ผู้ใช้เห็นคำขอค้าง · ของที่พิสูจน์ได้ว่า
forward pass ยังเดินมีอย่างเดียวคือสั่ง generate แล้วได้ token กลับมา

ชุดนี้คุม **ด้านที่อันตราย** เป็นหลัก ไม่ใช่ด้านที่ทำงานถูก: ของชิ้นนี้ restart โมเดลของลูกค้าได้เอง
(`README.md` → `lmds adopt` รับเครื่องที่มีโมเดลรันอยู่ก่อนเข้ามาคุม) ความผิดพลาดของมันจึงแพงกว่า
อาการที่มันแก้ · ทุกเทสเดินด้วยนาฬิกาปลอมที่เราหมุนเอง ไม่มี sleep จริง ไม่มีเซิร์ฟเวอร์จริง
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmds.fleet import watchdog as wd  # noqa: E402
from lmds.fleet.manager import FleetError, ServerInfo  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_watchdog(tmp_path, monkeypatch):
    """สถานะ watchdog กับ audit ของแต่ละเทสแยกกัน — ไฟล์สถานะถือเพดาน restart ไว้"""
    monkeypatch.setenv("LMDS_WATCHDOG_ROOT", str(tmp_path / "watchdog"))
    monkeypatch.setenv("LMDS_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    return tmp_path


def server(slug: str = "qwen3-32b", **over) -> ServerInfo:
    fields = {"slug": slug, "model": slug, "port": 8000, "mode": "docker",
              "container": f"lmds-{slug}", "running": True, "registered": True,
              "external": False}
    fields.update(over)
    return ServerInfo(**fields)


class Clock:
    """นาฬิกาที่เราหมุนเอง — ทำให้ backoff/settle/หน้าต่างเวลาถูกทดสอบจริงโดยไม่ต้องรอ"""

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Engine:
    """เซิร์ฟเวอร์ปลอมที่ "ตาย" และ "ฟื้น" ได้จริง — restart ของเราต้องมีผลกับมัน

    ไม่ใช่ mock ที่นับว่าถูกเรียกกี่ครั้ง: มันถือสถานะ alive ของตัวเอง `restart()` ทำให้ตอบได้อีกครั้ง
    เทสจึงพิสูจน์ได้ว่าลูปหยุด restart เพราะ *มันหายแล้ว* ไม่ใช่เพราะเราบอกให้หยุด
    """

    def __init__(self, alive: bool = True, heals: bool = True):
        self.alive = alive
        self.heals = heals
        self.restarts = 0

    def probe(self) -> wd.ProbeResult:
        if self.alive:
            return wd.ProbeResult(True, True, 200, "", 12)
        return wd.ProbeResult(False, False, 0, "ReadTimeout: no answer in 30s", 30000)

    def restart(self) -> None:
        self.restarts += 1
        if self.heals:
            self.alive = True


def drive(info, engine, clock, *, rounds: int, policy: wd.Policy | None = None) -> wd.State:
    state = wd.load(info.slug)
    if policy is not None:
        state.policy = dataclasses.asdict(policy)
        wd.save(state)
    return wd.loop(info.slug, info=info, rounds=rounds, clock=clock, sleeper=clock.sleep,
                   prober=engine.probe, restarter=engine.restart)


def audit_lines() -> list[dict]:
    from lmds.web import audit

    return audit.read(500)


# ── หัวใจ: /health เขียว แต่ generate ไม่ออก ─────────────────────────────────
def test_a_server_that_answers_200_without_generating_anything_is_counted_as_down():
    """นี่คืออาการที่ `/health` มองไม่เห็นพอดี — 200 แต่ไม่มี token ออกมา"""
    def poster(url, json, headers, timeout):
        return SimpleNamespace(status_code=200, json=lambda: {"choices": []})

    result = wd.probe("http://127.0.0.1:8000/v1", "qwen3-32b", poster=poster)

    assert result.ok is False
    assert result.alive is False
    assert "no completion" in result.detail


def test_a_server_that_really_generates_two_tokens_is_alive():
    seen = {}

    def poster(url, json, headers, timeout):
        seen.update({"url": url, "body": json})
        return SimpleNamespace(status_code=200,
                               json=lambda: {"choices": [{"text": " ok"}]})

    result = wd.probe("http://127.0.0.1:8000/v1", "qwen3-32b", poster=poster)

    assert result.ok is True
    assert seen["url"] == "http://127.0.0.1:8000/v1/completions"
    # 2 token greedy ตามที่ §1.9 บอก — ไม่ใช่การยิงงานจริงใส่โมเดลของลูกค้าทุกสองนาที
    assert seen["body"]["max_tokens"] == 2
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["stream"] is False


def test_a_server_that_answers_404_is_alive_so_nothing_gets_restarted():
    """404/401 = ตอบอยู่ แค่ไม่ชอบคำถามของเรา — restart ตรงนั้นคือ restart เพราะเราถามผิด"""
    def poster(url, json, headers, timeout):
        return SimpleNamespace(status_code=404, json=lambda: {})

    result = wd.probe("http://127.0.0.1:8000/v1", "m", poster=poster)
    assert result.alive is True
    assert result.ok is False

    info = server()
    wd.arm(info)
    clock = Clock()
    state, report = wd.tick(info, wd.load(info.slug), wd.Policy(), now=clock(),
                            prober=lambda: result, restarter=lambda: pytest.fail("ห้าม restart"))

    assert report["action"] == "misconfigured"
    assert state.consecutive_failures == 0
    assert [a["method"] for a in audit_lines()] == ["WATCHDOG-ARM", "WATCHDOG-SKIPPED"]


def test_an_engine_that_returns_500_is_down_not_merely_unhappy():
    def poster(url, json, headers, timeout):
        return SimpleNamespace(status_code=500, json=lambda: {})

    result = wd.probe("http://127.0.0.1:8000/v1", "m", poster=poster)
    assert result.alive is False


# ── ความยับยั้งชั่งใจ ─────────────────────────────────────────────────────────
def test_one_missed_probe_never_restarts_anything():
    info = server()
    wd.arm(info)
    engine = Engine(alive=False, heals=False)
    clock = Clock()

    state = drive(info, engine, clock, rounds=1)

    assert engine.restarts == 0
    assert state.consecutive_failures == 1
    assert state.restarts == []


def test_it_takes_three_consecutive_misses_to_earn_one_restart():
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=3, settle_seconds=600))
    engine = Engine(alive=False)      # heals=True — restart แล้วกลับมาตอบได้
    clock = Clock()

    state = drive(info, engine, clock, rounds=3)

    assert engine.restarts == 1
    assert len(state.restarts) == 1
    assert "3 consecutive" in state.restarts[0]["reason"]
    assert state.restarts[0]["ok"] is True


def test_after_a_restart_it_stops_probing_until_the_model_has_had_time_to_load():
    """ยิงใส่โมเดลที่กำลังโหลดแล้วนับว่าตาย = สร้างลูป restart ด้วยมือตัวเอง"""
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=3, settle_seconds=900))
    engine = Engine(alive=False, heals=False)      # restart แล้วยังไม่ฟื้น (โมเดลใหญ่ยังโหลด)
    clock = Clock()

    state = drive(info, engine, clock, rounds=9)   # 3 รอบแรก restart · อีก 6 รอบ = 720 วิ < settle

    assert engine.restarts == 1, "ยังอยู่ในช่วง settle ห้าม restart ซ้ำ"
    assert state.settle_until > clock()


def test_the_restart_budget_is_spent_and_then_it_never_touches_the_model_again():
    """โมเดลที่ start ไม่ขึ้นแล้วถูก restart ทุก 2 นาทีทั้งคืนคือความเสียหายที่ watchdog สร้างเอง"""
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=2, max_restarts=3,
                                  window_seconds=6 * 3600, backoff_seconds=60,
                                  settle_seconds=120))
    engine = Engine(alive=False, heals=False)      # ไม่มีวันฟื้น
    clock = Clock()

    state = drive(info, engine, clock, rounds=400)  # ≈ 13 ชั่วโมงของเวลาปลอม

    assert engine.restarts == 3, "ต้องไม่เกินเพดาน ไม่ว่าจะปล่อยไว้นานแค่ไหน"
    assert state.gave_up is True
    methods = [a["method"] for a in audit_lines()]
    assert methods.count("WATCHDOG-RESTART") == 3
    # บรรทัด GIVE-UP ต้องมีครั้งเดียว ไม่ใช่ทุกรอบตลอดคืน
    assert methods.count("WATCHDOG-GIVE-UP") == 1


def test_the_budget_survives_the_watchdog_process_being_restarted():
    """เพดานอยู่ในไฟล์ ไม่ใช่ในหน่วยความจำ — ไม่งั้น `Restart=always` ของ systemd ล้างโควตาทุกครั้ง"""
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=1, max_restarts=2,
                                  backoff_seconds=60, settle_seconds=60))
    engine = Engine(alive=False, heals=False)
    clock = Clock()

    drive(info, engine, clock, rounds=200)
    assert engine.restarts == 2
    assert wd.load(info.slug).gave_up is True

    # ลูปตาย systemd ปลุกขึ้นมาใหม่ — อ่านสถานะจากดิสก์ล้วน ไม่มีอะไรตกทอดในหน่วยความจำ
    drive(info, engine, clock, rounds=50)

    assert engine.restarts == 2


def test_backoff_keeps_a_second_restart_from_following_the_first_immediately():
    info = server()
    wd.arm(info, policy=wd.Policy(interval=60, failures_before_restart=1, max_restarts=5,
                                  backoff_seconds=600, settle_seconds=0))
    engine = Engine(alive=False, heals=False)
    clock = Clock()

    drive(info, engine, clock, rounds=5)          # 5 รอบ × 60 วิ = 300 วิ < backoff 600
    assert engine.restarts == 1

    drive(info, engine, clock, rounds=10)         # ผ่าน 600 วิไปแล้ว
    assert engine.restarts == 2


def test_a_model_that_comes_back_clears_the_counter_instead_of_creeping_toward_a_restart():
    """พลาดสลับกับผ่านไปเรื่อย ๆ ต้องไม่สะสมจนถึงเพดาน — GC pause กับเน็ตสะดุดมีจริง"""
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=3))
    clock = Clock()
    flaky = iter([False, True, False, True, False, True] * 4)
    engine = Engine()

    def probe():
        engine.alive = next(flaky)
        return engine.probe()

    state = wd.loop(info.slug, info=info, rounds=24, clock=clock, sleeper=clock.sleep,
                    prober=probe, restarter=lambda: pytest.fail("ห้าม restart"))

    assert state.restarts == []


# ── สิ่งที่ห้ามแตะเลย ─────────────────────────────────────────────────────────
def test_a_container_lmds_did_not_create_cannot_be_armed():
    """`lmds adopt` รับโมเดลของลูกค้าที่รันอยู่ก่อนเข้ามาคุมได้ — ของพวกนั้นไม่ใช่ของเราที่จะ restart"""
    info = server(external=True)

    with pytest.raises(FleetError) as exc:
        wd.arm(info)

    assert "did not create" in str(exc.value)
    assert wd.load(info.slug).armed is False


def test_a_server_without_a_registration_cannot_be_armed():
    info = server(registered=False)

    with pytest.raises(FleetError) as exc:
        wd.arm(info)

    assert "server.meta" in str(exc.value)


def test_an_embedding_model_cannot_be_armed_because_a_generate_probe_is_meaningless_there(tmp_path):
    bundle = tmp_path / "bundles" / "bge-m3"
    bundle.mkdir(parents=True)
    controller = bundle / "bge-m3-single.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    (bundle / "MODEL_PROFILE.yaml").write_text(
        "topology: single\nfeatures:\n  embedding:\n    pooling: cls\n", encoding="utf-8")
    info = server("bge-m3", controller=str(controller), mode="native", container="")

    with pytest.raises(FleetError) as exc:
        wd.arm(info)

    assert "embedding" in str(exc.value)
    assert "restart would be wrong" in str(exc.value)


def test_arming_a_stacked_bundle_warns_that_a_restart_takes_the_whole_group(tmp_path):
    bundle = tmp_path / "bundles" / "ds-v4"
    bundle.mkdir(parents=True)
    controller = bundle / "ds-v4-stacked.sh"
    controller.write_text("#!/bin/bash\n", encoding="utf-8")
    (bundle / "MODEL_PROFILE.yaml").write_text("topology: stacked\n", encoding="utf-8")
    info = server("ds-v4", controller=str(controller), mode="native", container="")

    warnings = wd.warnings_for(info)

    assert warnings and "whole group" in warnings[0]
    assert wd.arm(info).armed is True        # เตือน ไม่ใช่ห้าม — head ของ stacked ก็ต้องเฝ้าได้


def test_nothing_is_armed_until_someone_says_so():
    assert wd.armed_slugs() == []
    assert wd.load("anything").armed is False


def test_disarming_stops_the_loop_without_touching_the_running_model():
    info = server()
    wd.arm(info)
    wd.disarm(info.slug)

    with pytest.raises(FleetError) as exc:
        wd.loop(info.slug, info=info, rounds=1)

    assert "ยังไม่ได้เปิด" in str(exc.value)


def test_a_bundle_that_disappeared_stops_the_loop_instead_of_complaining_forever():
    info = server()
    wd.arm(info)
    clock = Clock()

    state = wd.loop(info.slug, rounds=5, clock=clock, sleeper=clock.sleep,
                    finder=lambda slug: None,
                    restarter=lambda: pytest.fail("ห้าม restart"))

    assert state.armed is False
    assert [a["method"] for a in audit_lines()][-1] == "WATCHDOG-STOP"


# ── "ใครสั่ง" ต้องตอบได้ ──────────────────────────────────────────────────────
def test_every_automatic_restart_lands_in_the_audit_log_with_its_reason():
    """คำถามแรกเวลามีอะไรผิดปกติคือ "ใครสั่ง" · คำตอบ "ระบบสั่งเอง" ต้องหาเจอที่เดียวกับคำสั่งของคน"""
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=2, settle_seconds=0))
    engine = Engine(alive=False)
    clock = Clock()

    drive(info, engine, clock, rounds=2)

    restarts = [a for a in audit_lines() if a["method"] == "WATCHDOG-RESTART"]
    assert len(restarts) == 1
    entry = restarts[0]
    assert entry["actor"] == "watchdog"
    assert entry["path"] == "/models/qwen3-32b"
    assert entry["ip"] == "-"           # ไม่ใช่ "?" — ไม่มี IP ตั้งแต่ต้น คนละเรื่องกับ "ไม่รู้ IP"
    assert "2 consecutive 2-token probes got no answer" in entry["reason"]
    assert entry["status"] == 200


def test_a_restart_that_fails_is_logged_as_a_failure_not_silently_swallowed():
    info = server()
    wd.arm(info, policy=wd.Policy(failures_before_restart=1, settle_seconds=0))
    clock = Clock()

    def boom():
        raise FleetError("docker restart lmds-qwen3-32b ล้มเหลว")

    state = wd.loop(info.slug, info=info, rounds=1, clock=clock, sleeper=clock.sleep,
                    prober=lambda: wd.ProbeResult(False, False, 0, "ReadTimeout"),
                    restarter=boom)

    entry = [a for a in audit_lines() if a["method"] == "WATCHDOG-RESTART"][0]
    assert entry["status"] == 500
    assert "restart failed" in entry["reason"]
    assert state.restarts[0]["ok"] is False


def test_the_audit_file_is_the_same_one_lmds_audit_reads():
    """ไม่ได้เขียน log คนละที่ — `lmds audit` ต้องเห็นบรรทัดของ watchdog ปนกับของคน"""
    from lmds.web import audit

    info = server()
    wd.arm(info)
    raw = Path(audit.log_path()).read_text(encoding="utf-8").strip().splitlines()

    assert json.loads(raw[0])["method"] == "WATCHDOG-ARM"


# ── สถานะที่คนอ่านได้ ─────────────────────────────────────────────────────────
def test_the_status_says_it_gave_up_and_how_to_let_it_try_again():
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=1, max_restarts=1,
                                  backoff_seconds=0, settle_seconds=0))
    clock = Clock()
    engine = Engine(alive=False, heals=False)
    drive(info, engine, clock, rounds=10)
    assert engine.restarts == 1

    lines = "\n".join(wd.describe(wd.load(info.slug)))

    assert "เลิก restart แล้ว" in lines
    assert "lmds watchdog arm qwen3-32b" in lines


def test_arming_again_hands_back_a_fresh_budget_because_a_person_decided_to():
    info = server()
    wd.arm(info, policy=wd.Policy(interval=120, failures_before_restart=1, max_restarts=1,
                                  backoff_seconds=0, settle_seconds=0))
    clock = Clock()
    engine = Engine(alive=False, heals=False)
    drive(info, engine, clock, rounds=10)
    assert wd.load(info.slug).gave_up is True

    wd.arm(info)
    assert wd.load(info.slug).gave_up is False


def test_the_settle_window_follows_what_the_bundle_says_it_needs_to_load(tmp_path):
    """โมเดล 157-220 GB รอ /health 5001-6906 วิจริง — เดา 900 วิแล้วยิงใส่ตอนโหลดคือสร้างลูป"""
    bundle = tmp_path / "bundles" / "ds-v4"
    bundle.mkdir(parents=True)
    controller = bundle / "ds-v4-single.sh"
    controller.write_text('STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-6906}"\n', encoding="utf-8")
    info = server("ds-v4", controller=str(controller), mode="native", container="")

    state = wd.arm(info)

    assert wd.policy_of(state).settle_seconds == 6906 + 120


# ── unit / คำสั่งสำรองสำหรับเครื่องที่ไม่มี systemd ────────────────────────────
def test_the_unit_restarts_the_watchdog_itself_but_that_does_not_widen_the_budget():
    unit = wd.render_unit("qwen3-32b")

    assert "Restart=always" in unit
    # พาธเต็มเสมอ — systemd ไม่ค้น $PATH ให้ (ดู test_the_unit_uses_an_absolute_path…)
    assert "watchdog run qwen3-32b" in unit
    # เพดานอยู่ในไฟล์สถานะ ไม่ใช่ในหน่วยความจำ — เทสที่พิสูจน์ข้อนี้คือ
    # test_the_budget_survives_the_watchdog_process_being_restarted
    assert "WantedBy=default.target" in unit      # user scope ไม่ใช่ multi-user.target


def test_a_machine_without_systemd_still_gets_a_command_it_can_run():
    """เครื่องลูกค้าบางรายอยู่ใน LXC/คอนเทนเนอร์ที่ไม่มี init system เต็ม"""
    assert "lmds watchdog run qwen3-32b" in wd.manual_command("qwen3-32b")


# ── พาธของ ExecStart ────────────────────────────────────────────────────────

def test_the_unit_uses_an_absolute_path_because_systemd_does_not_search_path():
    """เจอจริงบน msi-4 (2026-09-22) · `ExecStart=lmds watchdog run <slug>` ล้มด้วย
    `status=203/EXEC` ทุกครั้งเพราะ **systemd ไม่ค้น `$PATH` ให้** แล้ว `Restart=always`
    พามันวนใหม่ทุก 30 วินาที — รีสตาร์ตไป 9 รอบโดยไม่เคยรันสำเร็จเลย

    ที่อันตรายที่สุดคือ `lmds watchdog status` ยังรายงานว่า **"เปิดอยู่"** เพราะสถานะ
    อ่านจากไฟล์บนดิสก์ ไม่ได้ถาม systemd · ผลคือ "เฝ้าอยู่ในนาม แต่ไม่มีใครเฝ้าจริง"
    ซึ่งเป็นความล้มเหลวที่แย่ที่สุดของฟีเจอร์นี้ — แย่กว่าไม่เปิดเลย
    """
    from lmds.fleet import watchdog as wd

    line = next(x for x in wd.render_unit("demo").splitlines() if x.startswith("ExecStart="))
    command = line.split("=", 1)[1]

    assert command.startswith("/"), f"ต้องเป็นพาธเต็ม: {command}"
    assert not command.startswith("lmds "), "ชื่อเปล่า ๆ คือบั๊กเดิม"
    assert " watchdog run demo" in command


def test_the_manual_command_is_absolute_too():
    """เครื่องที่ไม่มี systemd ใช้บรรทัด nohup นี้ · เชลล์ค้น PATH ให้ก็จริง แต่ผู้ใช้
    อาจก๊อปไปวางใน cron หรือ supervisor ที่ไม่มี PATH เหมือนกัน"""
    from lmds.fleet import watchdog as wd

    assert wd.manual_command("demo").split()[1].startswith("/")
