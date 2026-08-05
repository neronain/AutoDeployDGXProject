import pytest
from typer.testing import CliRunner

import lmds
from lmds.cli.main import app
from lmds.secrets import get_secret

runner = CliRunner()


def test_version(isolated_config):
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert lmds.__version__ in result.output
    assert "v3.0.0" in result.output


def test_config_set_provider_and_show(isolated_config):
    result = runner.invoke(app, ["config", "set-provider", "openai"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_config_set_key_via_stdin_and_masked_in_show(isolated_config):
    secret = "sk-test1234567890abcdefgh"
    result = runner.invoke(app, ["config", "set-key", "openai", "--stdin"], input=secret + "\n")
    assert result.exit_code == 0
    assert secret not in result.output  # ค่า key ห้ามโผล่ใน output
    assert get_secret("openai") == secret

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert secret not in result.output  # show ต้อง mask เสมอ


def test_set_hf_token_optional_skip(isolated_config):
    result = runner.invoke(app, ["config", "set-hf-token", "--stdin"], input="\n")
    assert result.exit_code == 0
    assert get_secret("hf") is None


def test_set_hf_token_via_stdin(isolated_config):
    token = "hf_ABCDEFGHIJKLMNOP1234"
    result = runner.invoke(app, ["config", "set-hf-token", "--stdin"], input=token + "\n")
    assert result.exit_code == 0
    assert get_secret("hf") == token
    assert token not in result.output


def test_openai_compat_without_base_url_fails(isolated_config):
    result = runner.invoke(app, ["config", "set-provider", "openai-compat", "--model", "qwen"])
    assert result.exit_code == 1


def test_hardware_command_runs_anywhere(isolated_config):
    # บนเครื่อง dev ที่ไม่มี nvidia-smi ต้องไม่ crash — แค่รายงานว่าตรวจไม่พบ
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Profile" in result.output or "profile" in result.output


def test_completion_options_available():
    """lmds ต้องมี --install-completion ให้ผู้ใช้เปิด tab completion ได้"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--install-completion" in result.output


def test_complete_slug_includes_local_bundles(tmp_path, monkeypatch):
    """เติมชื่อจากโฟลเดอร์ ./bundles ของ cwd ด้วย (bundle ที่เพิ่ง deploy แต่ยังไม่เคย start)"""
    from lmds.cli.main import _complete_slug

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    (tmp_path / "bundles" / "brand-new-model").mkdir(parents=True)

    assert _complete_slug("brand") == ["brand-new-model"]


def test_complete_slug_from_run_registry(tmp_path, monkeypatch):
    """TAB บน lmds stop/logs/... ต้องเติมชื่อ slug ที่มีอยู่จริง และต้องไม่ยิง docker"""
    from lmds.cli.main import _complete_slug

    # _complete_slug อ่าน ./bundles/ ของ cwd ด้วย — ต้อง chdir ออกจาก repo จริง
    # ไม่งั้นเทสจะเห็น bundle ที่ผู้ใช้ deploy ไว้จริงปนเข้ามา
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    (tmp_path / "qwen3-32b").mkdir()
    (tmp_path / "qwen3-0-6b-gguf").mkdir()
    (tmp_path / "gemma-4-26b").mkdir()

    assert _complete_slug("qwen") == ["qwen3-0-6b-gguf", "qwen3-32b"]
    assert _complete_slug("") == ["gemma-4-26b", "qwen3-0-6b-gguf", "qwen3-32b"]
    assert _complete_slug("zzz") == []


def test_complete_slug_never_raises(tmp_path, monkeypatch):
    """shell เรียกทุกครั้งที่กด TAB — พังไม่ได้เด็ดขาด"""
    from lmds.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMDS_RUN_ROOT", "/ไม่มีจริง/path")
    assert cli_main._complete_slug("a") == []


def test_complete_target_presets():
    from lmds.cli.main import _complete_target

    assert "dgx-spark-single" in _complete_target("dgx")
    assert "dgx-spark-stacked" in _complete_target("dgx")
    assert all(name.startswith("rtx-40") for name in _complete_target("rtx-40"))
    assert _complete_target("zzz") == []


# ── หน้าเว็บที่รันเบื้องหลัง ────────────────────────────────────────────────────
# ทั้งกลุ่มนี้มาจากเคสจริงบน controller: `lmds web -b` ซ้ำ ๆ แล้วหน้าเว็บใช้ได้บ้างไม่ได้บ้าง
# เพราะรอบหลัง bind ไม่ได้แล้วตาย แต่ CLI พิมพ์ token ใหม่ให้ ผู้ใช้จึงเจอ "ต้องมี token"

def test_web_refuses_to_start_when_one_is_already_running(tmp_path, monkeypatch):
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.write_state(pid=4242, port=8600, bind="0.0.0.0", token="เดิม")
    monkeypatch.setattr(daemon, "alive", lambda pid: pid == 4242)
    started = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: started.append(a) or None)

    result = runner.invoke(app, ["web", "-b", "--bind", "0.0.0.0"])

    assert result.exit_code == 0
    assert not started, "ต้องไม่สตาร์ตซ้อน"
    # ต้องพิมพ์ลิงก์ของตัวที่เสิร์ฟจริง ไม่ใช่ token ใหม่ที่ไม่มีใครถืออยู่
    assert "เดิม" in result.output
    assert "--restart" in result.output


def test_web_status_shows_the_link_of_the_live_server(tmp_path, monkeypatch):
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.write_state(pid=99, port=8611, bind="0.0.0.0", token="tok-live")
    monkeypatch.setattr(daemon, "alive", lambda pid: True)

    result = runner.invoke(app, ["web", "--status"])
    assert result.exit_code == 0
    assert "tok-live" in result.output and "8611" in result.output


def test_web_status_forgets_a_dead_server(tmp_path, monkeypatch):
    """ไฟล์สถานะค้างจากรอบที่ตายไปแล้วต้องไม่ถูกรายงานว่ายังรันอยู่"""
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.write_state(pid=123456, port=8600, bind="0.0.0.0", token="ศพ")
    monkeypatch.setattr(daemon, "alive", lambda pid: False)

    result = runner.invoke(app, ["web", "--status"])
    assert result.exit_code == 0
    assert "ศพ" not in result.output
    assert not daemon.state_file().exists()


def test_web_background_reports_failure_instead_of_a_dead_link(tmp_path, monkeypatch):
    """สตาร์ตไม่ขึ้นต้องบอกว่าไม่ขึ้น + เหตุผลจาก log — ไม่ใช่พิมพ์ลิงก์ที่เปิดไม่ได้"""
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.log_file().parent.mkdir(parents=True, exist_ok=True)
    daemon.log_file().write_text("ERROR:    [Errno 98] address already in use\n", encoding="utf-8")

    class DeadProc:
        pid = 777

        def poll(self):
            return 1

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: DeadProc())
    monkeypatch.setattr(daemon, "port_busy", lambda *a, **k: False)
    monkeypatch.setattr(daemon, "wait_until_serving", lambda *a, **k: False)

    result = runner.invoke(app, ["web", "-b", "--bind", "0.0.0.0"])
    assert result.exit_code == 1
    assert "ไม่สำเร็จ" in result.output
    assert "Errno 98" in result.output
    assert not daemon.state_file().exists(), "สตาร์ตไม่ขึ้นต้องไม่เขียนสถานะค้างไว้"


def test_web_stop_does_not_kill_a_recycled_pid(tmp_path, monkeypatch):
    """PID ถูกใช้ซ้ำได้ — ถ้า process นั้นไม่ใช่หน้าเว็บของเราแล้ว ห้ามส่งสัญญาณไปหา"""
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.write_state(pid=31337, port=8600, bind="0.0.0.0", token="t")
    monkeypatch.setattr(daemon, "alive", lambda pid: False)
    killed = []
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append(pid))

    result = runner.invoke(app, ["web", "--stop"])
    assert result.exit_code == 1
    assert killed == []


@pytest.mark.parametrize("flag", ["--restart", "--stop"])
def test_web_stop_waits_for_the_port_to_be_released(tmp_path, monkeypatch, flag):
    """"หยุดแล้ว" ต้องแปลว่าพอร์ตว่างจริง — SIGTERM ไม่ได้คืน socket ทันที ถ้าไม่รอ
    คำสั่งถัดไปของผู้ใช้จะฟ้อง "พอร์ตไม่ว่าง" จากตัวที่เขาเพิ่งสั่งหยุดไปเอง (เจอจริงทั้งสองทาง)
    """
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path))
    daemon.write_state(pid=555, port=8600, bind="0.0.0.0", token="เก่า")
    monkeypatch.setattr(daemon, "alive", lambda pid: pid == 555)
    monkeypatch.setattr("os.kill", lambda pid, sig: None)
    waited = []
    monkeypatch.setattr(daemon, "wait_until_free", lambda *a, **k: waited.append(a) or True)
    monkeypatch.setattr(daemon, "port_busy", lambda *a, **k: False)

    class Proc:
        pid = 556

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(daemon, "wait_until_serving", lambda *a, **k: True)

    result = runner.invoke(app, ["web", flag, "-b", "--bind", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert waited, "ต้องรอให้พอร์ตว่างก่อนบอกว่าหยุดแล้ว"
    if flag == "--restart":
        assert daemon.read_state()["pid"] == 556


def test_web_reuses_the_same_token_across_restarts(tmp_path, monkeypatch):
    """ลิงก์ที่ bookmark ไว้ต้องไม่ตายทุกครั้งที่เปิดใหม่ — ผู้ใช้จะต้องกลับไปหา terminal ทุกรอบ
    ซึ่งขัดกับเหตุผลที่มีหน้าเว็บตั้งแต่แรก
    """
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path / "cfg"))

    class Proc:
        pid = 900

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(daemon, "wait_until_serving", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "port_busy", lambda *a, **k: False)
    monkeypatch.setattr(daemon, "alive", lambda pid: False)

    first = runner.invoke(app, ["web", "-b", "--bind", "0.0.0.0"])
    token = daemon.read_state()["token"]
    assert token and token in first.output

    daemon.clear_state()
    second = runner.invoke(app, ["web", "-b", "--bind", "0.0.0.0"])
    assert daemon.read_state()["token"] == token, "token ต้องเป็นตัวเดิม"
    assert token in second.output


def test_web_new_token_rotates_it(tmp_path, monkeypatch):
    from lmds.web import daemon

    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path / "cfg"))
    daemon.remember_token("token-เก่า")

    class Proc:
        pid = 901

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(daemon, "wait_until_serving", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "port_busy", lambda *a, **k: False)
    monkeypatch.setattr(daemon, "alive", lambda pid: False)

    result = runner.invoke(app, ["web", "-b", "--bind", "0.0.0.0", "--new-token"])
    assert result.exit_code == 0, result.output
    assert daemon.remembered_token() not in ("", "token-เก่า")


def test_web_token_file_is_not_world_readable(tmp_path, monkeypatch):
    """token = สิทธิ์สั่ง start/stop โมเดลทุกเครื่องในทะเบียน — ผู้ใช้อื่นบนเครื่องเดียวกัน
    ไม่ควรอ่านได้ เพราะมันอยู่ยาวข้ามการ restart แล้ว
    """
    import stat

    from lmds.web import daemon

    monkeypatch.setenv("LMDS_CONFIG_DIR", str(tmp_path / "cfg"))
    daemon.remember_token("s3cret")
    mode = daemon.token_file().stat().st_mode
    assert not mode & (stat.S_IRGRP | stat.S_IROTH)


# ── ตัวบอกความคืบหน้า ────────────────────────────────────────────────


def test_working_is_silent_when_output_is_piped():
    """`lmds plan --json | jq` ต้องไม่มีอะไรปน — บาง shell/CI รวม stderr เข้า stdout ให้เอง"""
    from lmds.cli.main import _working, err_console

    assert err_console.is_terminal is False  # pytest ไม่ใช่ terminal
    with err_console.capture() as captured:
        with _working("ไม่ควรเห็นบรรทัดนี้"):
            pass
    assert captured.get() == ""


def test_working_shows_label_on_a_real_terminal(monkeypatch):
    """บนหน้าจอจริงต้องบอกว่ากำลังทำอะไร ไม่ใช่ค้างเงียบ ๆ"""
    from rich.console import Console

    from lmds.cli import main as cli_main

    fake = Console(force_terminal=True, width=100)
    monkeypatch.setattr(cli_main, "err_console", fake)
    with fake.capture() as captured:
        with cli_main._working("อ่านข้อมูลจาก Hugging Face"):
            pass
    assert "อ่านข้อมูลจาก Hugging Face" in captured.get()


def test_working_lets_errors_through():
    """สปินเนอร์ต้องไม่กลืน exception — ไม่งั้น error หายไปเฉย ๆ"""
    import pytest as _pytest

    from lmds.cli.main import _working

    with _pytest.raises(ValueError):
        with _working("งานที่จะพัง"):
            raise ValueError("พัง")
