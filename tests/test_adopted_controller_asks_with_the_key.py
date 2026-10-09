"""controller ที่ adopt มาต้องถาม server ของตัวเองด้วย key เดียวกับที่ start ส่งให้

เคสจริง AI-Local-ISIT 2026-10-10: adopt container ของ Strata ที่รันด้วย `-e API_KEY=…`
server ตอบ /health 200 และตอบคำขอที่มี key ปกติ แต่ `…-adopted.sh status` พิมพ์
"api: ยังไม่ตอบ" — status/test-text/client-config ยิง /v1/models โดยไม่แนบ key จึงได้ 401
ทุกครั้ง ทั้งที่ start ของสคริปต์เดียวกันรู้จัก key ตัวนั้นอยู่แล้ว (load_api_key)

ผลที่ตามมาไม่ได้มีแค่ข้อความผิด: test-text ล้มกับ server ที่ดีอยู่ และ client-config ตกไปใช้
slug เป็นชื่อโมเดล ซึ่ง client เอาไปเรียกแล้วได้ 404 · vLLM/SGLang ที่ตั้ง --api-key เป็นแบบเดียวกัน
(ทุก path ใต้ /v1 อยู่หลัง key) — ที่ไม่เคยเห็นเพราะ llama.cpp เปิด /v1/models สาธารณะเสมอ

เทสรันสคริปต์จริงกับ HTTP server จริงที่บังคับ key
"""

import importlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

adopt = importlib.import_module("lmds.fleet.adopt")

KEY = "sekrit-key-of-this-server"
SERVED = "the-name-the-server-really-serves"


class _Keyed(BaseHTTPRequestHandler):
    """server ที่ทำตัวแบบ vLLM/Strata: ทุก path ใต้ /v1 ต้องมี key · key = None คือไม่บังคับ"""

    def _answer(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _allowed(self) -> bool:
        key = self.server.key
        if key is None or self.headers.get("authorization") == f"Bearer {key}":
            return True
        self._answer(401, {"error": {"message": "missing or wrong API key"}})
        return False

    def do_GET(self):  # noqa: N802
        if self._allowed():
            self._answer(200, {"object": "list", "data": [{"id": SERVED, "permission": [{"id": "perm-1"}]}]})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        if self._allowed():
            self.server.asked.append(body.get("model"))
            self._answer(200, {"choices": [{"message": {"role": "assistant", "content": "4"}}]})

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    made = []

    def start(key):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Keyed)
        httpd.key, httpd.asked = key, []
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        made.append(httpd)
        return httpd

    yield start
    for httpd in made:
        httpd.shutdown()
        httpd.server_close()


def _container(env):
    return adopt.Adopted(container="c1", image="strata:trial", args=[], env=env, binds=[], ports={},
                         network="bridge", runtime="nvidia", entrypoint=["./docker-entrypoint.sh"],
                         ipc_mode="private", shm_size=0)


def _native(argv):
    return adopt.AdoptedProcess(pid=4242, argv=argv, exe="/opt/llama/llama-server", cwd="/tmp")


def _call(script, command, home, port, stored_key=KEY):
    bin_dir = home / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"                      # status ของทาง container ถาม docker ps ก่อนถาม API
    docker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    if stored_key is not None:
        keys = home / ".lmds" / "keys"
        keys.mkdir(parents=True, exist_ok=True)
        (keys / "demo").write_text(stored_key + "\n", encoding="utf-8")
    controller = home / "demo-adopted.sh"
    controller.write_text(script, encoding="utf-8")
    controller.chmod(0o755)
    # ไม่มี key ใน environment เลย — แบบเดียวกับที่ hub/systemd/คอนโซลเรียกสคริปต์นี้
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "API_PORT": str(port)}
    return subprocess.run(["bash", str(controller), command], capture_output=True, text=True, env=env, timeout=60)


SCRIPTS = {
    "container": lambda: adopt.render_controller(_container([f"API_KEY={KEY}", "MODEL=IQ3_S"]), "demo"),
    "native": lambda: adopt.render_native_controller(
        _native(["/opt/llama/llama-server", "-m", "/models/x.gguf", "--port", "8080", "--api-key", KEY]), "demo"),
}


@pytest.fixture(params=sorted(SCRIPTS))
def script(request):
    text = SCRIPTS[request.param]()
    assert KEY not in text, "ค่าของ key ต้องไม่อยู่ในไฟล์ 0755"
    return text


def test_status_of_a_healthy_keyed_server_says_it_answers(script, server, tmp_path):
    httpd = server(KEY)
    done = _call(script, "status", tmp_path, httpd.server_port)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "api: ตอบปกติ" in done.stdout, done.stdout
    assert "ยังไม่ตอบ" not in done.stdout


def test_test_text_reaches_the_model_with_the_name_it_really_serves(script, server, tmp_path):
    httpd = server(KEY)
    done = _call(script, "test-text", tmp_path, httpd.server_port)
    assert done.returncode == 0, done.stdout + done.stderr
    assert httpd.asked == [SERVED]


def test_client_config_names_the_served_model_not_the_slug(script, server, tmp_path):
    httpd = server(KEY)
    done = _call(script, "client-config", tmp_path, httpd.server_port)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["model"] == SERVED


def test_a_key_the_server_refuses_is_reported_as_refused_not_as_down(script, server, tmp_path):
    """server ขึ้นและตอบอยู่ แต่ key ที่ LMDS เก็บไว้ไม่ใช่ตัวที่มันใช้ — คนอ่านต้องไปแก้ key ไม่ใช่ไป restart"""
    httpd = server(KEY)
    done = _call(script, "status", tmp_path, httpd.server_port, stored_key="a-key-from-before-rotation")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "401" in done.stdout and "key" in done.stdout, done.stdout
    assert "api: ตอบปกติ" not in done.stdout and "ยังไม่ตอบ" not in done.stdout
    assert "a-key-from-before-rotation" not in done.stdout + done.stderr


def test_nothing_listening_is_still_reported_as_not_answering(script, server, tmp_path):
    httpd = server(KEY)
    port = httpd.server_port
    httpd.shutdown()
    httpd.server_close()
    done = _call(script, "status", tmp_path, port)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "api: ยังไม่ตอบ" in done.stdout


def test_a_server_without_a_key_is_asked_without_one(server, tmp_path):
    """container ที่ไม่มีตัวแปร *API_KEY*: ไม่มี key ให้แนบ และต้องไม่แนบของที่บังเอิญอยู่ในที่เก็บ"""
    seen = []

    class _Record(_Keyed):
        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("authorization"))
            super().do_GET()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Record)
    httpd.key, httpd.asked = None, []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        script = adopt.render_controller(_container(["MODEL=IQ3_S"]), "demo")
        done = _call(script, "status", tmp_path, httpd.server_port, stored_key="left-over-from-another-bundle")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert "api: ตอบปกติ" in done.stdout, done.stdout + done.stderr
    assert seen == [None]
