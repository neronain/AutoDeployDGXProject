"""Audit รอบ 2 (2026-09-05) — stacked controller ทั้ง lifecycle ตามที่ hub และคนสั่งจริง

รอบแรก (`test_audit_stacked_controller.py`) ไล่ download/prepare-runtime/sync/start ที่ "ล้มแล้วไม่บอกสาเหตุ" ·
รอบนี้ไล่สิ่งที่ **ทำงานผิดเงียบ ๆ หรือฆ่าของดี**: doctor ที่ไม่เคยรันสคริปต์ในคอนเทนเนอร์ · flag ที่ test-tools
แนะให้ใช้แต่ controller ไม่รู้จัก · คำสั่งผิดคืน 0 · start ทับตัวเองแล้วโทษพอร์ต · MASTER_PORT/พอร์ต worker ชนกับ
bundle อื่น · ssh สะดุดครั้งเดียวระหว่างรอ health = ฆ่า head ที่โหลดมาชั่วโมง · เขียน worker.sh ไม่ได้/head
docker run ล้มแล้วออกเงียบ ทิ้ง worker ค้าง · props ไม่ส่ง key · download ดึง image 30 GB ก่อนบอกว่าไม่มี HF_TOKEN ·
remove -y ลบไฟล์ของ root ผ่าน image vLLM ที่ entrypoint ไม่ใช่เชลล์ · stop/status/logs ไป ssh เครื่องตัวอย่าง ·
--bind <ip เฉพาะ> แล้ว health ไม่เคยผ่าน · ทุกข้อรัน controller ที่ render แล้วจริงใต้ bash กับ shim ต่อ node
"""

from __future__ import annotations

import json
import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.test_audit_stacked_controller import (
    _DOCKER, _SSH, HEAD, WORKER, _bin, _bundle, _calls, _report, _run, _seed_head_cache, _shim,
)

_LOG_CURL = 'echo "curl $*" >> "$FAKE_LOG"; echo "{}"; exit 0\n'


def _listening(*ports: int) -> str:
    """ss ปลอม: พิมพ์ socket ที่ฟังอยู่ (รูปแบบ `ss -tln` จริง)"""
    lines = "".join(f'echo "LISTEN 0 4096 0.0.0.0:{p} 0.0.0.0:*"\n' for p in ports)
    return lines + "exit 0\n"


# ═════════════════════ doctor ═════════════════════
def test_doctor_actually_runs_its_probe_inside_the_image(tmp_path):
    """เคสจริง: `doctor` บน spark-head พิมพ์แค่ Image/ID แล้วจบ 0 ไม่มี torch/vllm/GPU สักบรรทัด — heredoc ถูกส่งให้
    `docker run … python3 -` โดยไม่มี `-i` docker จึงไม่ต่อ stdin ให้คอนเทนเนอร์ python3 ได้สคริปต์ว่างแล้วจบเงียบ
    (บั๊กเดียวกับ verify-worker ที่แก้ไป 0.6.0 แต่ doctor ตกหล่น)"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, ["doctor"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert "torch      :" in done.stdout and "python     :" in done.stdout, done.stdout


# ═════════════════════ --tool-parser / --reasoning-parser ═════════════════════
def test_tool_and_reasoning_parser_flags_work_like_on_the_single_controller(tmp_path):
    """test-tools ของ stacked บอกให้ `restart --tool-parser <ชื่อ>` (ข้อความเดียวกับ single) แต่ parse_options ของ
    stacked ไม่รู้จัก flag นี้ → ตกไป REMAINING_ARGS แล้วถูกเมิน restart ด้วย parser เดิม ผู้ใช้วนอยู่ที่เดิม"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    done = _run(bundle, ["serve-args", "--tool-parser", "qwen3_xml", "--reasoning-parser=deepseek_r1"], tmp_path)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "--tool-call-parser" in lines and "qwen3_xml" in lines, done.stdout
    assert "--reasoning-parser" in lines and "deepseek_r1" in lines, done.stdout
    usage = _run(bundle, ["help"], tmp_path, cluster=False)
    assert "--tool-parser" in usage.stdout and "--reasoning-parser" in usage.stdout


# ═════════════════════ คำสั่งที่ไม่รู้จัก ═════════════════════
def test_an_unknown_verb_fails_instead_of_printing_usage_with_exit_0(tmp_path):
    """hub/คนสั่ง `<ctl> repair` หรือพิมพ์ผิด → ได้ usage ยาว ๆ แล้ว exit 0 = job บนหน้าเว็บขึ้นว่าสำเร็จ"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    bogus = _run(bundle, ["repair"], tmp_path, cluster=False)
    assert bogus.returncode != 0
    assert "repair" in bogus.stderr, bogus.stderr
    for ok in ([], ["help"], ["--help"]):
        done = _run(bundle, ok, tmp_path, cluster=False)
        assert done.returncode == 0 and "COMMANDS" in done.stdout, ok


# ═════════════════════ start ทับตัวเอง / พอร์ตชน bundle อื่น ═════════════════════
def test_start_on_a_running_bundle_says_restart_not_port_conflict(tmp_path):
    """กด start ซ้ำขณะโมเดลรันอยู่ → "port 8000 ถูกใช้อยู่ — stop ตัวที่ชนก่อน" ทั้งที่ตัวที่ชนคือ head ของ bundle นี้เอง
    ผู้ใช้ไปไล่หา "ตัวที่ชน" ไม่เจอ"""
    bundle = _bundle(tmp_path)
    # `docker ps --filter name=` ตอบ id เมื่อ head ของ bundle นี้รันอยู่ (ไม่ใช่แค่ inspect ที่ตอบ true ให้ทุกชื่อ)
    _bin(tmp_path, ss=_listening(8000),
         docker=_DOCKER.replace("  image) exit 0 ;;", '  ps) echo "${FAKE_PS_RUNNING:-}"; exit 0 ;;\n  image) exit 0 ;;'))
    done = _run(bundle, ["start"], tmp_path, env={"FAKE_PS_RUNNING": "c0ffee01"})
    assert done.returncode != 0
    assert "restart" in done.stderr and f"lmds-{bundle.directory.name}-head" in done.stderr, done.stderr
    assert "stop ตัวที่ชน" not in done.stderr
    assert "docker[head] run -d" not in _calls(tmp_path)


def test_start_refuses_when_the_master_port_or_the_worker_port_is_taken(tmp_path):
    """สอง stacked bundle บน head คู่เดียวกัน (--network host): API_PORT คนละพอร์ตได้ แต่ MASTER_PORT 25000 และพอร์ต
    18000 ของ worker เป็นค่าคงที่ — bundle ที่สองผ่านด่านพอร์ต 8001 แล้วไปตายที่ TCPStore/bind ด้วยข้อความอ่านยาก
    หลังปล่อย worker ไปแล้ว"""
    bundle = _bundle(tmp_path)
    env = {"FAKE_RUNNING_MAP": "head=false", "API_PORT": "8001"}
    _bin(tmp_path, ss=_listening(8000, 25000))
    done = _run(bundle, ["start"], tmp_path, env=env)
    assert done.returncode != 0
    assert "MASTER_PORT" in done.stderr and "25000" in done.stderr, done.stderr
    assert "run -d" not in _calls(tmp_path)

    (tmp_path / "calls.log").unlink()
    _bin(tmp_path, ss=_listening(18000))
    done = _run(bundle, ["start"], tmp_path, env=env)
    assert done.returncode != 0
    assert "18000" in done.stderr and WORKER in done.stderr, done.stderr
    assert "run -d" not in _calls(tmp_path)


# ═════════════════════ ssh สะดุดระหว่างรอ health ≠ worker ตาย ═════════════════════
_FLAKY = '''
if [[ -n "${FAKE_SSH_FLAKY_CMD:-}" && "$cmd" == *"${FAKE_SSH_FLAKY_CMD}"* ]]; then
  n=$(( $(cat "$FAKE_SSH_FLAKY_FILE" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_SSH_FLAKY_FILE"
  if (( n <= ${FAKE_SSH_FLAKES:-1} )); then echo "ssh: connect to host $node port 22: Connection timed out" >&2; exit 255; fi
fi
'''
_SSH_FLAKY = _SSH.replace('export FAKE_NODE="$node"', _FLAKY + 'export FAKE_NODE="$node"')
_CURL_HEALTH_LATER = '''
n=$(( $(cat "$FAKE_CURL_FILE" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_CURL_FILE"
(( n <= 1 )) && exit 7
exit 0
'''


def test_a_flaky_management_ssh_during_health_wait_does_not_kill_the_head(tmp_path):
    """ระหว่างรอ head โหลดโมเดล (ชั่วโมงกว่า) controller แวะถาม worker ทุก 60 วิผ่าน ssh — ssh ที่ timeout ครั้งเดียว
    (สาย management สะดุด · sshd ของ worker ยุ่ง) ถูกอ่านว่า "worker ตาย" → docker rm -f head ทิ้งทั้งที่ทั้งสองฝั่ง
    กำลังโหลด weight อยู่ดี ๆ · ssh คืน 255 = ไม่รู้สถานะ ต้องเตือนแล้วลองรอบหน้า ไม่ใช่ตัดสิน"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, ssh=_SSH_FLAKY, curl=_CURL_HEALTH_LATER)
    _seed_head_cache(tmp_path / "home")
    env = {"FAKE_SSH_FLAKY_CMD": "docker inspect -f '{{.State.Running}}' 'lmds-",
           "FAKE_SSH_FLAKY_FILE": str(tmp_path / "flakes"), "FAKE_SSH_FLAKES": "2",
           "FAKE_CURL_FILE": str(tmp_path / "curls"), "STARTUP_TIMEOUT": "60"}
    done = _run(bundle, ["start"], tmp_path, env=env, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout + done.stderr
    assert "ไม่ผ่านชั่วคราว" in out and "worker ตาย" not in out.replace("ตีความว่า worker ตาย", ""), out
    calls = _calls(tmp_path)
    head_run = calls.index("docker[head] run -d")
    assert "docker[head] rm -f" not in calls[head_run:], "ห้ามฆ่า head เพราะ ssh สะดุด"
    assert "Server พร้อม" in done.stdout


# ═════════════════════ เขียน worker.sh ไม่ได้ / head docker run ล้ม ═════════════════════
def test_a_worker_script_that_cannot_be_written_is_explained_before_any_container_starts(tmp_path):
    """/tmp/lmds-<slug> บน worker ถูกยึด (เป็นไฟล์/เป็นของ user อื่น) → `mkdir -p && cat >` ล้ม แล้ว set -e ทิ้งสคริปต์
    ด้วยข้อความดิบของ bash ฝั่ง worker ไม่มีชื่อเครื่อง ไม่มีทางแก้"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    _seed_head_cache(tmp_path / "home")
    remote_tmp = tmp_path / "remote" / "tmp"
    remote_tmp.mkdir(parents=True)
    (remote_tmp / f"lmds-{bundle.directory.name}").write_text("not a dir", encoding="utf-8")
    done = _run(bundle, ["start"], tmp_path)
    assert done.returncode != 0
    assert "worker.sh" in done.stderr and WORKER in done.stderr and "sudo rm -rf" in done.stderr, done.stderr
    assert "run -d" not in _calls(tmp_path)


_DOCKER_HEAD_RUN_FAILS = _DOCKER.replace(
    '    echo "docker env[${node}]',
    '    if [[ -n "$(lookup "${FAKE_RUN_FAIL_MAP:-}" "")" ]]; then\n'
    '      echo "docker: Error response from daemon: could not select device driver \\"\\" with capabilities: [[gpu]]." >&2; exit 125; fi\n'
    '    echo "docker env[${node}]')


def test_a_head_that_fails_to_launch_stops_the_workers_it_already_started(tmp_path):
    """worker ทุกตัวเปิดไปแล้ว → `docker run` ของ head ล้ม (nvidia runtime หาย / daemon สะอึก) → set -e ออกทันทีด้วย
    ข้อความของ docker เปล่า ๆ · worker headless ค้างรอ head ที่ไม่มีวันมา ยึด GPU ไว้จน start รอบหน้า"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, docker=_DOCKER_HEAD_RUN_FAILS)
    _seed_head_cache(tmp_path / "home")
    done = _run(bundle, ["start"], tmp_path, env={"FAKE_RUN_FAIL_MAP": "head=1"})
    assert done.returncode != 0
    assert "head" in done.stderr and "docker run" in done.stderr and "device driver" in done.stderr, done.stderr
    calls = _calls(tmp_path)
    worker_run = calls.index(f"docker[{WORKER}] run -d")
    assert f"docker[{WORKER}] rm -f lmds-{bundle.directory.name}-worker" in calls[worker_run:], \
        "worker ที่เปิดไปแล้วต้องถูกหยุดเมื่อ head ไม่ขึ้น"


# ═════════════════════ props / download gated / remove -y ═════════════════════
def test_props_sends_the_api_key_like_status_does(tmp_path):
    """เปิด API_KEY แล้ว `props` ตอบ 401 → curl -f ล้ม → pipefail → exit 1 เงียบ ๆ ทั้งที่ status อ่านได้"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl=_LOG_CURL)
    done = _run(bundle, ["props"], tmp_path, env={"API_KEY": "sekrit"}, cluster=False)
    assert done.returncode == 0, done.stderr
    assert "Authorization: Bearer sekrit" in _calls(tmp_path), _calls(tmp_path)


def test_download_checks_the_gated_token_before_pulling_a_30gb_image(tmp_path):
    """repo gated + ยังไม่มี HF_TOKEN: เดิม ensure_image ดึง image 20-30 GB ให้เสร็จก่อน แล้วค่อยบอกว่าต้องมี token"""
    bundle = _bundle(tmp_path, report=_report(gated=True))
    _bin(tmp_path, docker=_DOCKER.replace("  image) exit 0 ;;", "  image) exit 1 ;;"))
    done = _run(bundle, ["download"], tmp_path, cluster=False)
    assert done.returncode != 0
    assert "HF_TOKEN" in done.stderr, done.stderr
    assert "pull" not in _calls(tmp_path), "ต้องหยุดก่อนดึง image"


def test_remove_deletes_root_owned_leftovers_with_a_real_rm_entrypoint(tmp_path):
    """`remove -y` เจอไฟล์ที่ root เขียนไว้ (rm ธรรมดาไม่ได้) → เดิม `docker run <image แรกที่เจอ> rm -rf …` — บน worker
    image เดียวที่มีคือ vLLM ซึ่ง entrypoint เป็น `vllm serve`/python: `rm -rf` กลายเป็น argument ของ vllm → ล้ม
    → "ยังเหลือบน worker … ลบเอง" ทุกครั้ง · ต้อง --entrypoint rm และใช้ image ของ bundle ที่รู้ว่ามีอยู่"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    locked = tmp_path / "remote" / "tmp" / f"lmds-{bundle.directory.name}" / "locked"
    locked.mkdir(parents=True)
    (locked / "worker.log").write_text("root wrote this", encoding="utf-8")
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    image = "vllm/vllm-openai@sha256:3dbe092e"
    try:
        done = _run(bundle, ["remove", "-y"], tmp_path, env={"VLLM_IMAGE": image})
    finally:
        locked.chmod(0o755)
    assert done.returncode == 0, done.stderr
    run = next((l for l in _calls(tmp_path).splitlines() if l.startswith(f"docker[{WORKER}] run")), "")
    assert "--entrypoint rm" in run and image in run, run
    assert run.endswith(f"-rf -- /x/lmds-{bundle.directory.name}"), run


# ═════════════════════ stop/status/logs กับคลัสเตอร์ตัวอย่าง ═════════════════════
def test_stop_status_and_logs_do_not_ssh_to_the_example_worker(tmp_path):
    """ยังไม่มี cluster.env: stop/status/logs ผ่านด่านได้ (hub เรียกบ่อย) แต่เดิมยัง ssh ไป 10.100.152.2 ซึ่งเป็นแค่ตัวอย่าง
    — ค้าง ConnectTimeout 10 วิ × 2 ต่อ worker และถ้ามีเครื่องนั้นอยู่จริงก็ไปสั่ง docker rm บนเครื่องคนอื่น
    · start ไม่มีทางผ่านโดยไม่ตั้งค่า จึงไม่มี container ของ bundle นี้บนเครื่องตัวอย่างให้หยุดอยู่แล้ว"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path)
    stop = _run(bundle, ["stop"], tmp_path, cluster=False)
    assert stop.returncode == 0, stop.stderr
    assert "ssh" not in _calls(tmp_path), _calls(tmp_path)
    assert "ยังไม่ตั้งค่า" in stop.stdout + stop.stderr
    status = _run(bundle, ["status"], tmp_path, cluster=False)
    assert status.returncode == 0 and "ssh" not in _calls(tmp_path)
    logs = _run(bundle, ["logs", "worker"], tmp_path, cluster=False)
    assert logs.returncode != 0 and "lmds node cluster --write" in logs.stderr, logs.stderr
    assert "ssh" not in _calls(tmp_path)


# ═════════════════════ --bind <ip เฉพาะ> ═════════════════════
class _Chat(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_specific_bind_address_is_used_for_health_status_and_the_test_suite(tmp_path):
    """`--bind 10.2.1.195` (ผูกเฉพาะสาย management) → vLLM ไม่ฟัง 127.0.0.1 · start รอ /health ที่ 127.0.0.1 จนครบ
    STARTUP_TIMEOUT (ชั่วโมงกว่า) แล้วบอกว่าไม่ health ทั้งที่เซิร์ฟเวอร์ขึ้นแล้ว · status/test-*/bench ก็บอด"""
    bundle = _bundle(tmp_path)
    _bin(tmp_path, curl=_LOG_CURL)
    done = _run(bundle, ["status", "--bind", "10.1.1.1"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert "http://10.1.1.1:8000/health" in _calls(tmp_path), _calls(tmp_path)
    (tmp_path / "calls.log").unlink()
    _run(bundle, ["status", "--bind", "0.0.0.0"], tmp_path)
    assert "http://127.0.0.1:8000/health" in _calls(tmp_path)

    # ชุดทดสอบ python (stress/bench/test-tools …) ต้องไปที่เดียวกัน — เซิร์ฟเวอร์ปลอมฟังที่ 127.0.0.2 เท่านั้น
    server = ThreadingHTTPServer(("127.0.0.2", 0), _Chat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = str(server.server_address[1])
        hit = _run(bundle, ["stress", "1", "1", "--bind", "127.0.0.2", "--port", port], tmp_path, cluster=False)
        assert hit.returncode == 0 and "stress: PASS" in hit.stdout, hit.stdout + hit.stderr
    finally:
        server.shutdown()
        server.server_close()
