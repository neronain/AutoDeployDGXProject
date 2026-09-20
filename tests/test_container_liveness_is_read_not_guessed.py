"""สถานะ "รันอยู่" ของ container ต้องมาจาก docker ไม่ใช่จากการเดา

เคสจริง 2026-09-20 บน BesthaiAi (3× RTX 3060 · docker 29.1.3): หน้าเว็บโชว์โมเดลที่
container ของมัน `Exited (255) 2 weeks ago` เป็น running=True ค้างอยู่ที่ "loading"
ตลอดกาล พร้อมปุ่ม Stop ที่กดแล้วไม่มีอะไรเกิดขึ้น — ไม่มีอะไรฟังที่พอร์ตนั้น ไม่มี process
ไม่มี bundle ไม่มี controller

ต้นเหตุ: `_container_running` เดิมถาม `docker ps --filter "name=^<ชื่อ>$"` แล้วนับว่า
"มีบรรทัดออกมา = รันอยู่" · ตัวกรอง `name=` ของ docker เป็น **regex** ไม่ใช่การเทียบสตริง
จุดใน slug ของ LMDS (`qwen3.6-…`, `…-4.7-opus-…`) จึงเป็นไวลด์การ์ด แล้วชื่อของตัวที่
หยุดไปแล้วไปตรงกับ container **คนละตัว** ที่กำลังรันอยู่

พิสูจน์กับ docker 29.4.0 ตัวจริง (2026-09-20):

    $ docker create --name 'lmds-probe-qwen3.6-x' …      # สร้างไว้เฉย ๆ ไม่ได้รัน
    $ docker run -d  --name 'lmds-probe-qwen3-6-x' …      # คนละตัว · รันอยู่
    $ docker ps --filter 'name=^lmds-probe-qwen3.6-x$' --format '{{.Names}}'
    lmds-probe-qwen3-6-x

เทสในไฟล์นี้จำลอง docker ให้ตรงตามนั้น (ตัวกรอง name เป็น regex · `docker ps` เห็นเฉพาะ
ตัวที่ยังไม่หยุด · `container inspect` เทียบชื่อตรงตัว) — โค้ดเดิมจึงตกทุกข้อที่ควรตก
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from lmds.fleet import manager

# docker ลิสต์ตัวพวกนี้ใน `docker ps` เปล่า ๆ — "อยู่ในรายการ" จึงไม่เท่ากับ "กำลังเสิร์ฟ"
_PS_VISIBLE = {"running", "restarting", "paused"}


class FakeDocker:
    """docker จำลองที่ทำตัวเหมือนของจริงพอจะจับบั๊กชุดนี้ได้

    `containers` = ชื่อ → (state, image, ports) · state ตามคำของ docker เป๊ะ
    (`running` · `exited` · `created` · `restarting` · `paused`)
    """

    def __init__(self, containers: dict[str, tuple[str, str, str]]):
        self.containers = containers
        self.calls: list[list[str]] = []

    # --- พฤติกรรมของ docker จริง --------------------------------------------
    def _ps_rows(self, pattern: str | None) -> list[str]:
        rows = []
        for name, (state, image, ports) in self.containers.items():
            if state not in _PS_VISIBLE:
                continue
            # ตัวกรอง name= ของ docker คือ regex (Go regexp) ไม่ใช่การเทียบสตริง — จุดคือไวลด์การ์ด
            if pattern is not None and not re.search(pattern, name):
                continue
            rows.append("\t".join([name, image, ports, state]))
        return rows

    def run(self, cmd, **_kwargs):
        self.calls.append(list(cmd))
        if cmd[:3] == ["docker", "container", "inspect"] or cmd[:2] == ["docker", "inspect"]:
            name = cmd[-1]
            entry = self.containers.get(name)      # เทียบชื่อตรงตัว ไม่ใช่ regex
            if entry is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="No such container")
            return SimpleNamespace(returncode=0, stdout=f"{str(entry[0] == 'running').lower()}\n", stderr="")
        if cmd[:2] == ["docker", "ps"]:
            pattern = next((a.split("=", 1)[1] for a in cmd if a.startswith("name=")), None)
            return SimpleNamespace(returncode=0, stdout="\n".join(self._ps_rows(pattern)) + "\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def docker(monkeypatch):
    """ติดตั้ง docker จำลอง แล้วคืน factory ให้เทสกำหนดรายชื่อ container เอง"""
    def install(containers: dict[str, tuple[str, str, str]]) -> FakeDocker:
        fake = FakeDocker(containers)
        monkeypatch.setattr(manager.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(manager.subprocess, "run", fake.run)
        return fake
    return install


def _make_meta(root, slug: str, container: str, port: int = 8080, controller: str = "") -> None:
    """ทะเบียนของโมเดลที่ "เคย start จริง" แล้ว controller หายไป (started_at มี = ห้ามเก็บกวาดเงียบ ๆ)"""
    run_dir = root / slug
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\n"
        f"model={slug}\n"
        f"model_id=org/{slug}\n"
        "engine=llamacpp\n"
        "mode=docker\n"
        f"port={port}\n"
        f"container={container}\n"
        "pid_file=\n"
        f"controller={controller}\n"
        "started_at=2026-09-06T09:00:00\n",
        encoding="utf-8",
    )


# ───────────────────────── _container_running ─────────────────────────
def test_a_container_that_exited_is_not_reported_running(docker):
    """`Exited (255) 2 weeks ago` = ไม่ได้รัน — ไม่ว่าชื่อจะยังอยู่ในทะเบียนของ docker ก็ตาม"""
    docker({"lmds-ghost": ("exited", "ghcr.io/ggml-org/llama.cpp:server-cuda", "")})

    assert manager._container_running("lmds-ghost") is False


def test_a_container_that_was_only_created_is_not_reported_running(docker):
    """`created` (ยังไม่เคย start) ก็ยังไม่ได้เสิร์ฟอะไร"""
    docker({"lmds-ghost": ("created", "ghcr.io/ggml-org/llama.cpp:server-cuda", "")})

    assert manager._container_running("lmds-ghost") is False


def test_a_stopped_container_is_not_mistaken_for_a_different_running_one(docker):
    """บั๊กตัวจริง: จุดใน slug เป็นไวลด์การ์ดของ regex → ตัวที่ตายยืมสถานะของ "คนละตัว" มาใช้

    ชื่อสองตัวนี้ต่างกันแค่ตรงที่ตัวที่ตายมีจุด — `docker ps --filter name=^…$` จึงคืนชื่อของ
    ตัวที่รันอยู่ออกมา แล้วโค้ดเดิมอ่านว่า "ตัวที่ถามรันอยู่"
    """
    docker({
        "lmds-qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-q4_k_m":
            ("exited", "ghcr.io/ggml-org/llama.cpp:server-cuda", ""),
        "lmds-qwen3-6-35b-a3b-claude-4-7-opus-reasoning-distilled-q4_k_m":
            ("running", "ghcr.io/ggml-org/llama.cpp:server-cuda", ""),
    })

    assert manager._container_running(
        "lmds-qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-q4_k_m") is False
    assert manager._container_running(
        "lmds-qwen3-6-35b-a3b-claude-4-7-opus-reasoning-distilled-q4_k_m") is True


def test_liveness_is_asked_by_exact_name_never_by_a_name_pattern(docker):
    """กันไม่ให้ถอยกลับไปใช้ตัวกรองที่เป็น regex อีก — ชื่อต้องเดินทางเป็น "ชื่อ" ไม่ใช่ "แพตเทิร์น\""""
    fake = docker({"lmds-qwen3.6-x": ("running", "vllm/vllm-openai:latest", "")})

    manager._container_running("lmds-qwen3.6-x")

    assert fake.calls, "ต้องถาม docker จริง"
    issued = fake.calls[-1]
    assert not any(arg.startswith("name=") for arg in issued), \
        f"ยังส่งชื่อไปเป็นตัวกรองแบบ regex อยู่: {issued}"
    assert "lmds-qwen3.6-x" in issued


def test_a_missing_container_is_not_running_and_does_not_raise(docker):
    docker({})

    assert manager._container_running("lmds-never-existed") is False


def test_no_container_name_means_not_running_without_calling_docker(docker):
    fake = docker({"lmds-x": ("running", "vllm/vllm-openai:latest", "")})

    assert manager._container_running("") is False
    assert fake.calls == []


# ───────────────────────── discover() ─────────────────────────
def test_discover_reports_the_exited_container_as_stopped_but_still_lists_it(tmp_path, monkeypatch, docker):
    """ทั้งเส้นทางตั้งแต่ทะเบียนบนดิสก์จนถึงสิ่งที่หน้าเว็บได้รับ — ต้องเป็น "stopped" ไม่ใช่ "running"

    จำลองเครื่องจริง: โมเดลถูก deploy ใหม่หลัง slug ถูกทำให้สะอาดคนละแบบ (จุด → ขีด) ของเก่าจึง
    ค้างเป็น container ที่ exited คู่กับทะเบียนเดิม ส่วนของใหม่รันอยู่ · ชื่อสองตัวต่างกันแค่ตรงจุด
    ซึ่ง regex อ่านเป็นไวลด์การ์ด → ตัวที่ตาย "ยืม" สถานะของตัวที่รันอยู่มาแสดง แล้วการ์ดบนหน้าเว็บ
    ค้างที่ loading ตลอดกาลพร้อมปุ่ม Stop ที่กดแล้วไม่เกิดอะไรขึ้น

    และต้อง **ยังอยู่ในรายการ** — โมเดลที่ LMDS รู้จักแต่หยุดอยู่คือของที่กด start ได้
    หายไปจากจอ = ผู้ใช้ไปต่อไม่ถูก (คนละอาการกับการโกหกว่ารันอยู่ แต่แย่พอกัน)
    """
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(manager, "_pgrep_llama", lambda: [])
    monkeypatch.setattr(manager, "_scan_bundles", lambda known: [])
    monkeypatch.setattr(manager, "_health_ok", lambda *_a, **_k: False)
    _make_meta(tmp_path / "run", "qwen3.6-35b-a3b-q4_k_m", "lmds-qwen3.6-35b-a3b-q4_k_m", port=8080)
    docker({
        "lmds-qwen3.6-35b-a3b-q4_k_m": ("exited", "ghcr.io/ggml-org/llama.cpp:server-cuda", ""),
        # ของที่ deploy ใหม่ · ชื่อต่างจากข้างบนแค่ตรงที่ข้างบนเป็นจุด
        "lmds-qwen3-6-35b-a3b-q4_k_m": ("running", "ghcr.io/ggml-org/llama.cpp:server-cuda", ""),
    })

    found = {s.slug: s for s in manager.discover()}

    assert "qwen3.6-35b-a3b-q4_k_m" in found, "โมเดลที่หยุดอยู่ต้องยังเห็นได้ เพื่อให้กด start ได้"
    dead = found["qwen3.6-35b-a3b-q4_k_m"]
    assert dead.running is False, "container ที่ exited ห้ามรายงานว่า running"
    assert dead.healthy is False
    # ของที่รันอยู่จริงต้องยังถูกเก็บเข้ามาตามปกติ (ไม่มีทะเบียน = orphan)
    assert found["qwen3-6-35b-a3b-q4_k_m"].running is True


def test_the_live_container_next_to_it_is_still_reported_running(tmp_path, monkeypatch, docker):
    """แก้แล้วต้องไม่เหวี่ยงไปอีกทาง — ตัวที่รันอยู่จริงต้องยังขึ้นว่า running"""
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(manager, "_pgrep_llama", lambda: [])
    monkeypatch.setattr(manager, "_scan_bundles", lambda known: [])
    monkeypatch.setattr(manager, "_health_ok", lambda *_a, **_k: True)
    _make_meta(tmp_path / "run", "qwen36-35b-abl", "lmds-qwen36-35b-abl", port=8001)
    docker({"lmds-qwen36-35b-abl": ("running", "ghcr.io/ggml-org/llama.cpp:server-cuda", "")})

    found = {s.slug: s for s in manager.discover()}

    assert found["qwen36-35b-abl"].running is True
    assert found["qwen36-35b-abl"].healthy is True


# ───────────────────────── orphan scan ─────────────────────────
def test_an_orphan_container_that_is_not_running_says_so():
    """`docker ps` ลิสต์ restarting/paused ด้วย — "อยู่ในรายการ" ไม่ใช่ "กำลังเสิร์ฟ\""""
    tab, nl = chr(9), chr(10)
    rows = nl.join([
        tab.join(["lmds-up", "vllm/vllm-openai:latest", "0.0.0.0:8000->8000/tcp", "running"]),
        tab.join(["lmds-flapping", "vllm/vllm-openai:latest", "", "restarting"]),
        tab.join(["lmds-frozen", "vllm/vllm-openai:latest", "", "paused"]),
    ])

    by_slug = {s.slug: s for s in manager._parse_docker_ps(rows, set())}

    assert by_slug["up"].running is True
    assert by_slug["flapping"].running is False, "restarting = ยังไม่ได้เสิร์ฟ"
    assert by_slug["frozen"].running is False


def test_output_without_a_state_column_still_parses():
    """ผู้เรียกที่ยังส่งสามคอลัมน์ (`docker ps` เปล่า ๆ) ต้องไม่พัง"""
    tab, nl = chr(9), chr(10)
    rows = nl.join([tab.join(["lmds-up", "vllm/vllm-openai:latest", "0.0.0.0:8000->8000/tcp"])])

    found = manager._parse_docker_ps(rows, set())

    assert [(s.slug, s.running) for s in found] == [("up", True)]


def test_a_docker_that_errors_does_not_look_like_an_empty_machine(monkeypatch):
    """daemon ล่ม/สิทธิ์ไม่พอ = "ไม่รู้" — ห้ามอ่าน stdout ว่างแล้วสรุปว่าไม่มีอะไรรันอยู่"""
    monkeypatch.setattr(manager.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        manager.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="permission denied while trying to connect"),
    )

    assert manager._orphan_docker(set()) == []
    assert manager._container_running("lmds-anything") is False


# ───────────────────────── ของที่ LMDS ไม่ได้เป็นเจ้าของ ─────────────────────────
def test_a_container_lmds_does_not_own_is_still_visible_and_never_force_removed(monkeypatch):
    """ของคนอื่นต้องยังเห็นได้ และห้าม `docker rm -f` เด็ดขาด (manager.py:952)"""
    tab = chr(9)
    row = tab.join(["somebody-elses-vllm", "vllm/vllm-openai:latest", "0.0.0.0:9000->8000/tcp", "running"])
    found = {s.slug: s for s in manager._parse_docker_ps(row, set())}
    assert found["somebody-elses-vllm"].external is True
    assert found["somebody-elses-vllm"].running is True

    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager.subprocess, "run",
        lambda cmd, **kw: calls.append(list(cmd)) or SimpleNamespace(returncode=0, stdout=""),
    )
    assert manager.stop_server(found["somebody-elses-vllm"]) == "docker-stop"
    assert not any(c[:3] == ["docker", "rm", "-f"] for c in calls)
