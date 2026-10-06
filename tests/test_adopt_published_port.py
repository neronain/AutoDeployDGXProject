"""พอร์ตที่ adopt จด ต้องเป็นพอร์ตที่ **เครื่อง** เปิดให้ ไม่ใช่พอร์ตข้างใน container

audit 2026-10-06: container ที่รันด้วย `-p 8001:8000` — `lmds ps` โชว์ 8001 ก่อน adopt (อ่านจาก
`docker ps`) แล้วกลายเป็น 8000 หลัง adopt เพราะ adopt อ่านพอร์ตจาก `--port` บน argv ซึ่งเป็นพอร์ต
*ข้างใน* · เลขนั้นไหลลง API_PORT ของ controller, serving.port ของ profile และ server.meta →
health/status/test-text/watchdog ไปเคาะ 127.0.0.1:8000 ซึ่งอาจเป็นบริการอื่นไปเลย

เคสจริงที่ทำให้เรื่องนี้ไม่ใช่ทฤษฎี: บน AI-Local-ISIT พอร์ต 8000 คือ `portainer` (โมเดลอยู่ 8001) —
"/health ตอบ 200" จากพอร์ตผิดตัวเกือบถูกรายงานเป็นเขียวมาแล้วครั้งหนึ่ง

และ llama.cpp ที่ตั้งค่าผ่าน `LLAMA_ARG_*` (แบบที่เอกสารทางการของ image แนะนำ) ซึ่ง adopt ทิ้งทั้งชุด
เพราะชื่อไม่เข้ารายการ prefix: คำสั่งที่สร้างออกมาไม่มีโมเดล และ adopt รายงาน `model: '' context: 0`
"""

import shutil

import pytest
import yaml

from tests.adopt_fakes import (
    adopt_from, adopt_mod, container_payload, image_payload, inspected, run_controller, started,
    values_of,
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")

PUBLISHED_ON_8001 = {"PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8001"}]}}

# curl ปลอม: จด URL ที่ controller ไปเคาะ แล้วตอบเหมือน /v1/models ของ vLLM
FAKE_CURL = r'''#!/bin/bash
for a in "$@"; do case "$a" in http://*) echo "$a" >> "$FAKE_ARGV.curl" ;; esac; done
echo '{"object":"list","data":[{"id":"gemma4","object":"model"}]}'
'''


def test_the_recorded_port_is_the_host_side_of_the_publish():
    adopted = inspected(container_payload(host=PUBLISHED_ON_8001))
    assert adopted.port == 8001, "นี่คือพอร์ตที่คนนอก container ต้องใช้"
    assert adopted.container_port == 8000, "พอร์ตข้างในยังต้องรู้ — ใช้จับคู่กับ --publish"


def test_status_and_test_text_probe_the_published_port(tmp_path):
    """รันคำสั่งของ controller จริง แล้วดูว่า curl ถูกส่งไปที่พอร์ตไหน"""
    script = adopt_mod.render_controller(inspected(container_payload(host=PUBLISHED_ON_8001)), "m")
    for command in ("status", "test-text", "client-config"):
        done, _ = run_controller(script, tmp_path, command, extra_bins={"curl": FAKE_CURL})
        assert done.returncode == 0, done.stdout + done.stderr
    urls = (tmp_path / "docker-run.argv.curl").read_text(encoding="utf-8").split()
    assert urls and all(u.startswith("http://127.0.0.1:8001/") for u in urls), urls


def test_the_container_is_still_published_the_way_it_was(tmp_path):
    """แก้เลขที่ *จด* ไม่ใช่แก้การ publish — ข้างในยังฟัง 8000 และ argv ยังบอก --port 8000"""
    argv = started(adopt_mod.render_controller(inspected(container_payload(host=PUBLISHED_ON_8001)), "m"),
                   tmp_path)
    assert values_of(argv, "--publish") == ["8001:8000"]
    assert values_of(argv, "--port") == ["8000"]


def test_profile_and_registration_carry_the_published_port(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    controller = adopt_from(container_payload(name="coder-next", host=PUBLISHED_ON_8001), tmp_path / "bundles")
    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    assert profile["serving"]["port"] == 8001

    from lmds.fleet.manager import _parse_meta, run_root

    meta = _parse_meta(run_root() / "coder-next" / "server.meta")
    assert meta["port"] == "8001"


def test_host_networking_has_no_mapping_to_apply():
    """--network host: พอร์ตข้างในคือพอร์ตของเครื่อง · PortBindings ที่ค้างมาไม่มีผล"""
    adopted = inspected(container_payload(host={"NetworkMode": "host", "PortBindings": {}}))
    assert adopted.port == 8000


def test_the_api_port_is_matched_to_its_own_binding_not_the_first_one():
    """เคส spark-03 2026-08-27 (6006 metrics · 8355 API · 8888 notebook) ในรูปที่ publish ย้ายเลข"""
    adopted = inspected(container_payload(
        args=["-m", "vllm.entrypoints.openai.api_server", "--model", "/m", "--port", "8355"],
        host={"PortBindings": {"6006/tcp": [{"HostIp": "", "HostPort": "16006"}],
                               "8355/tcp": [{"HostIp": "", "HostPort": "18355"}],
                               "8888/tcp": [{"HostIp": "", "HostPort": "18888"}]}}))
    assert adopted.port == 18355


def test_an_unpublished_api_port_is_reported_instead_of_guessed():
    """โมเดลที่อยู่หลัง reverse proxy บน network ของ compose ไม่ publish พอร์ตเลย — LMDS เคาะ 127.0.0.1
    ไม่ถึงแน่ ๆ · ต้องบอกตรงนี้ ไม่ใช่ปล่อยให้คนเห็น "api: ยังไม่ตอบ" ตลอดกาลแล้วเดาเอง"""
    adopted = inspected(container_payload(host={"NetworkMode": "llm_default", "PortBindings": {}}))
    said = " ".join(adopt_mod.not_reproduced(adopted))
    assert "8000" in said and "publish" in said


def test_a_random_host_port_is_read_from_what_docker_assigned_and_flagged():
    """`-p 8000` (ไม่ระบุฝั่งเครื่อง): HostPort ว่างใน HostConfig · เลขจริงอยู่ใน NetworkSettings.Ports
    และจะ **เปลี่ยนทุกครั้ง** ที่สร้าง container ใหม่ — ต้องบอก ไม่งั้น restart แล้วเคาะพอร์ตเก่า"""
    adopted = inspected(container_payload(
        host={"PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": ""}]}},
        NetworkSettings={"Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "32768"},
                                                {"HostIp": "::", "HostPort": "32768"}]}, "Networks": {}}))
    assert adopted.port == 32768
    assert any("32768" in line or "สุ่ม" in line for line in adopt_mod.not_reproduced(adopted))


# ── llama.cpp ที่ตั้งค่าผ่าน LLAMA_ARG_* ──────────────────────────────────────────────────
LLAMA_IMAGE = image_payload({
    "Env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LLAMA_ARG_HOST=0.0.0.0"],
    "Entrypoint": ["/app/llama-server"], "WorkingDir": "/app", "Labels": {}})


def _llama():
    return container_payload(
        name="llama", image="ghcr.io/ggml-org/llama.cpp:server-cuda", path="/app/llama-server", args=[],
        entrypoint=["/app/llama-server"], config={"WorkingDir": "/app", "Labels": {},
                                                   "Env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                                           "LLAMA_ARG_HOST=0.0.0.0",
                                                           "LLAMA_ARG_MODEL=/models/qwen.gguf",
                                                           "LLAMA_ARG_CTX_SIZE=65536", "LLAMA_ARG_N_GPU_LAYERS=99",
                                                           "LLAMA_ARG_PORT=8080", "LLAMA_API_KEY=sk-llama-key-0123456789"]},
        host={"Binds": ["/srv/gguf:/models:ro"],
              "PortBindings": {"8080/tcp": [{"HostIp": "", "HostPort": "8090"}]}})


def test_llama_arg_env_is_what_the_model_is_configured_with():
    adopted = inspected(_llama(), LLAMA_IMAGE)
    assert adopted.model == "/models/qwen.gguf"
    assert adopted.context == 65536
    assert adopted.container_port == 8080 and adopted.port == 8090
    assert adopted.engine == "llamacpp"


def test_llama_arg_env_reaches_the_restarted_container(tmp_path):
    adopted = inspected(_llama(), LLAMA_IMAGE)
    script = adopt_mod.render_controller(adopted, "llama")
    assert "sk-llama-key-0123456789" not in script
    env = values_of(started(script, tmp_path), "--env")
    for wanted in ("LLAMA_ARG_MODEL=/models/qwen.gguf", "LLAMA_ARG_CTX_SIZE=65536",
                   "LLAMA_ARG_N_GPU_LAYERS=99", "LLAMA_ARG_PORT=8080"):
        assert wanted in env, f"{wanted} หายไป: {env}"
    assert "LLAMA_API_KEY" in env, "key ส่งเป็นชื่อเฉย ๆ ให้ docker หยิบค่าจากเชลล์"


def test_llama_arg_env_survives_even_when_the_image_cannot_be_asked(tmp_path):
    adopted = inspected(_llama(), with_image=False)
    env = values_of(started(adopt_mod.render_controller(adopted, "llama"), tmp_path), "--env")
    assert "LLAMA_ARG_MODEL=/models/qwen.gguf" in env and "LLAMA_ARG_CTX_SIZE=65536" in env


def test_argv_still_wins_over_llama_arg_env():
    """llama-server เองให้ argv ชนะ env — adopt ต้องรายงานตัวเดียวกับที่เซิร์ฟเวอร์ใช้จริง"""
    payload = _llama()
    payload["Args"] = ["-m", "/models/other.gguf", "-c", "8192"]
    adopted = inspected(payload, LLAMA_IMAGE)
    assert adopted.model == "/models/other.gguf"
    assert adopted.context == 8192
