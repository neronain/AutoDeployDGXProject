"""adopt ต้องไม่คัดลอกความลับจาก env ของ container ลงสคริปต์บนดิสก์

เคสจริง dgx-spark03 2026-09-03: `lmds adopt trtllm-nemotron` เขียน `--env HF_TOKEN=hf_…`
ลง bundles/trtllm-nemotron/trtllm-nemotron-adopted.sh (0755) · ทุก user บนเครื่องอ่านได้
และไฟล์นี้ถูก zip ส่งต่อ/push ข้ามเครื่องได้ · หลักของ LMDS คือความลับเดินทางทาง env/stdin
"""

import importlib
import inspect

adopt = importlib.import_module("lmds.fleet.adopt")   # lmds.fleet ส่งออกฟังก์ชันชื่อ adopt ด้วย


def test_secret_values_are_dropped_but_names_stay():
    kept, redacted = adopt.redact_secrets([
        "HF_TOKEN=hf_abcdefghijklmnop", "MODEL_HANDLE=nvidia/x", "VLLM_API_KEY=sk-123",
        "AWS_SECRET_ACCESS_KEY=zzz", "MAX_MODEL_LEN=4096",
    ])
    assert kept == ["HF_TOKEN", "MODEL_HANDLE=nvidia/x", "VLLM_API_KEY", "AWS_SECRET_ACCESS_KEY", "MAX_MODEL_LEN=4096"]
    assert redacted == ["HF_TOKEN", "VLLM_API_KEY", "AWS_SECRET_ACCESS_KEY"]
    assert not any("hf_abc" in k or "sk-123" in k or "zzz" in k for k in kept)


def test_render_controller_goes_through_the_redaction():
    """คุมที่ซอร์ส: บรรทัด --env ในสคริปต์ต้องมาจาก redact_secrets เท่านั้น"""
    src = inspect.getsource(adopt.render_controller)
    assert "redact_secrets(" in src
    assert "for e in meaningful_env(adopted))" not in src, "ยังเขียน env ดิบลงสคริปต์อยู่"


def test_download_then_serve_is_detected():
    """เคสจริง dgx-spark03: `hf download nvidia/X && trtllm-serve nvidia/X …` วนล้ม 15 รอบหลัง
    token หมดอายุ — adopt ต้องชี้ให้เห็นตั้งแต่ตอนสร้างสคริปต์"""
    cmd = "hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 && PYTORCH_ALLOC_CONF=x trtllm-serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --port 8355"
    assert adopt.download_before_serve(cmd) == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    assert adopt.download_before_serve("trtllm-serve /root/.cache/huggingface/hub/x --port 8355") == ""
    assert adopt.download_before_serve("") == ""


def test_render_controller_warns_about_download_before_serve():
    src = inspect.getsource(adopt.render_controller)
    assert "download_before_serve(" in src and "HF_HUB_OFFLINE" in src


# ── key ที่เก็บไว้กับเครื่องต้องไปถึง container ตอน autostart ────────────────────────
#
# การถอดค่าออกถูกแล้ว (ไฟล์ 0755) แต่ `--env ชื่อ` เฉย ๆ ให้ docker หยิบจาก environment
# ของเชลล์ที่สั่ง start — systemd ตอน autostart ไม่มี environment นั้นให้ → docker ข้าม
# ตัวแปรไป → container ขึ้นมาแบบไม่มี auth เงียบ ๆ ทุก reboot (เจอ 2026-09-21 บน dgx-msi-01)

def _adopted_for(env: list[str]):
    return adopt.Adopted(container="c1", image="vllm/vllm-openai:v0.9", args=["--model", "/m"],
                         env=env, binds=[], ports={}, network="bridge", runtime="nvidia",
                         entrypoint=[], ipc_mode="private", shm_size=0)


def _run(script, home, fake_log, extra_env=None):
    import subprocess

    bin_dir = home / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/bash\n[ "$1" = run ] && echo "${VLLM_API_KEY:-<empty>}|${HF_TOKEN:-<empty>}" '
                      '>> "$FAKE"\nexit 0\n', encoding="utf-8")
    docker.chmod(0o755)
    controller = home / "c-adopted.sh"
    controller.write_text(script, encoding="utf-8")
    controller.chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE": str(fake_log)}
    env.update(extra_env or {})
    done = subprocess.run(["bash", str(controller), "start"], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    return fake_log.read_text(encoding="utf-8").strip()


def test_the_adopted_controller_fills_the_key_from_the_machine_store(tmp_path):
    """จำลอง autostart: ไม่มี env ของเชลล์เลย — key ต้องมาจาก ~/.lmds/keys/<slug>"""
    script = adopt.render_controller(_adopted_for(["VLLM_API_KEY=sekrit", "HF_TOKEN=hf_xxx"]), "demo")
    assert "sekrit" not in script and "hf_xxx" not in script, "ค่าความลับต้องไม่อยู่ในไฟล์ 0755"

    home = tmp_path / "h"
    (home / ".lmds" / "keys").mkdir(parents=True)
    (home / ".lmds" / "keys" / "demo").write_text("stored-key-abc\n", encoding="utf-8")
    assert _run(script, home, tmp_path / "a.log") == "stored-key-abc|<empty>"


def test_only_the_model_api_key_is_filled_never_a_token_or_password(tmp_path):
    """ที่เก็บถือ API key ของ model server ตัวเดียว — เติมลง HF_TOKEN = ส่งของผิดให้ engine"""
    script = adopt.render_controller(_adopted_for(["VLLM_API_KEY=a", "HF_TOKEN=b"]), "demo")
    assert "HF_TOKEN" not in script.split("load_api_key() {")[1].split("}")[0]


def test_without_a_stored_key_nothing_changes(tmp_path):
    """bundle ที่ติดตั้งไปแล้วต้องทำงานเหมือนเดิมทุกประการเมื่อยังไม่มีใครตั้ง key"""
    script = adopt.render_controller(_adopted_for(["VLLM_API_KEY=sekrit"]), "demo")
    home = tmp_path / "h"
    home.mkdir()
    assert _run(script, home, tmp_path / "b.log") == "<empty>|<empty>"


def test_an_api_key_from_the_shell_still_wins(tmp_path):
    """ลำดับเดียวกับ controller ปกติ: env/flag > ไฟล์ที่เก็บไว้"""
    script = adopt.render_controller(_adopted_for(["VLLM_API_KEY=sekrit"]), "demo")
    home = tmp_path / "h"
    (home / ".lmds" / "keys").mkdir(parents=True)
    (home / ".lmds" / "keys" / "demo").write_text("stored-key-abc\n", encoding="utf-8")
    assert _run(script, home, tmp_path / "c.log",
                {"VLLM_API_KEY": "from-shell"}) == "from-shell|<empty>"


def test_a_container_with_no_api_key_variable_says_so_instead_of_pretending(tmp_path):
    """ไม่รู้ว่า engine อ่าน key จากไหน = บอกตรง ๆ · เงียบไว้แล้วให้คนคิดว่า `lmds key` คุมอยู่
    อันตรายกว่า เพราะ doctor จะขึ้นเขียวทั้งที่ endpoint ยังเปิด"""
    script = adopt.render_controller(_adopted_for(["MAX_MODEL_LEN=4096"]), "demo")
    assert "LMDS ไม่เห็นตัวแปรชื่อ *API_KEY*" in script
    assert "lmds key demo` ไม่มีผลกับ bundle นี้" in script
    assert "LMDS_KEY_ROOT" not in script, "ต้องไม่อ้างว่าอ่านที่เก็บได้ — doctor ใช้เครื่องหมายนี้ตัดสิน"

    home = tmp_path / "h"
    home.mkdir()
    assert _run(script, home, tmp_path / "d.log") == "<empty>|<empty>"


def test_a_redaction_note_never_lands_inside_the_docker_run_command(tmp_path):
    """บั๊กที่ ship อยู่ก่อนหน้านี้ (เจอ 2026-09-21 ตอนเขียนเทสที่รันสคริปต์จริง)

    หมายเหตุถูกแทรกกลาง `docker run … \\` — บรรทัดก่อนหน้าจบด้วย `\\` ซึ่งต่อเข้าบรรทัด
    คอมเมนต์ แล้วคำสั่งจบตรงนั้น: docker ถูกยิงโดย **ไม่มี image และไม่มี --env สักตัว**
    ส่วนบรรทัด `--env …` ที่เหลือกลายเป็นคำสั่งใหม่ → "--env: command not found"

    ผลคือ adopted bundle ที่มี env ความลับสักตัวจะ `start` ไม่ขึ้นเลย — และเงียบ เพราะ
    ไฟล์ยังผ่าน `bash -n` สบาย ๆ (คอมเมนต์ถูกไวยากรณ์) เทสที่ดูแต่ข้อความจึงจับไม่ได้
    """
    import subprocess

    script = adopt.render_controller(_adopted_for(["VLLM_API_KEY=sekrit", "HF_TOKEN=hf_x"]), "demo")

    home = tmp_path / "h"
    (home / "bin").mkdir(parents=True)
    argv_log = tmp_path / "argv.log"
    docker = home / "bin" / "docker"
    docker.write_text(f'#!/bin/bash\necho "$*" >> {argv_log}\nexit 0\n', encoding="utf-8")
    docker.chmod(0o755)
    controller = home / "c-adopted.sh"
    controller.write_text(script, encoding="utf-8")
    controller.chmod(0o755)

    done = subprocess.run(["bash", str(controller), "start"], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": f"{home / 'bin'}:/usr/bin:/bin"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "command not found" not in done.stderr, done.stderr

    run_line = next(line for line in argv_log.read_text(encoding="utf-8").splitlines()
                    if line.startswith("run "))
    assert "vllm/vllm-openai:v0.9" in run_line, f"docker run ไม่ได้รับ image: {run_line}"
    assert "--env VLLM_API_KEY" in run_line and "--env HF_TOKEN" in run_line
    assert "--model /m" in run_line
