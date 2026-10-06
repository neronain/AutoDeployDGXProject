"""ความลับที่อยู่บน argv ของของที่รับเข้ามา ต้องไม่ถูกคัดลอกลงสคริปต์ 0755 หรือ zip ที่ส่งข้ามเครื่อง

audit 2026-10-06 (สองทีมเจอเรื่องเดียวกันจากคนละทาง): `--api-key sk-…` / `--hf-token hf_…` /
`--token …` บน argv ถูกเขียนลง `<slug>-adopted.sh` ทั้งดุ้น — ทั้งทาง container (`Args` ของ docker
inspect) และทาง process (`/proc/<pid>/cmdline`) · ไฟล์นั้นเป็น 0755 อ่านได้ทุก user บนเครื่อง และอยู่ใน
zip ที่ `lmds node push` ส่งไปเครื่องอื่น · ทาง process ยังเขียน argv เต็มลง MODEL_PROFILE.yaml อีกชั้น

ของเดิมถอดค่าออกเฉพาะ **env** (dgx-spark03 2026-09-03 · ดู test_adopt_redacts_secrets.py) — argv
เป็นช่องเดียวกันที่ตกสำรวจ · ใช้กลไกเดียวกัน: ค่าไม่อยู่ในไฟล์ · API key ของ model server เติมจาก
`~/.lmds/keys/<slug>` (0600) ตอน start · ความลับอื่นหยิบจาก environment ของเชลล์ที่สั่ง start

ต่างจาก env อยู่ข้อเดียว และตั้งใจ: `--env NAME` ที่ไม่มีค่า docker แค่ข้าม แต่ `--api-key ""` คือ
**เซิร์ฟเวอร์ที่เปิดโล่ง** — ไม่มี key ให้เติม = ไม่ start ดีกว่า start แบบไม่มี auth เงียบ ๆ
"""

import shutil
import stat
import zipfile

import pytest
import yaml

from tests.adopt_fakes import (
    adopt_from, adopt_mod, command_after_image, container_payload, has_option, inspected,
    run_controller, started,
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="ต้องมี bash")

IMAGE = "vllm/vllm-openai:v0.11.0"
API_KEY = "sk-prod-9f8e7d6c5b4a39281716"
HF_TOKEN = "hf_abcdefghijklmnopqrstuv"
SERVE = ["-m", "vllm.entrypoints.openai.api_server", "--model", "/models/gemma", "--port", "8000"]


def _vllm(*extra, **over):
    return container_payload(args=[*SERVE, *extra], **over)


@pytest.fixture(autouse=True)
def key_store(tmp_path, monkeypatch):
    """ที่เก็บ key แยกต่อเทส — HOME ของชุดเทสเป็น sandbox ที่ใช้ร่วมกันทั้ง session"""
    monkeypatch.setenv("LMDS_KEY_ROOT", str(tmp_path / "keys"))
    return tmp_path / "keys"


# ── container ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("flags", [
    ["--api-key", API_KEY, "--hf-token", HF_TOKEN],
    [f"--api-key={API_KEY}", f"--hf-token={HF_TOKEN}"],
    ["--token", HF_TOKEN, "--api_key", API_KEY],
])
def test_secret_flag_values_never_reach_the_script(flags):
    script = adopt_mod.render_controller(inspected(_vllm(*flags)), "gemma")
    assert API_KEY not in script and HF_TOKEN not in script


def test_the_api_key_comes_back_from_the_machine_store_at_start(tmp_path, key_store):
    """จำลอง autostart: ไม่มี env ของเชลล์เลย — key ต้องมาจาก ~/.lmds/keys/<slug> แล้วไปถึง engine"""
    script = adopt_mod.render_controller(inspected(_vllm("--api-key", API_KEY)), "gemma")
    key_store.mkdir()
    (key_store / "gemma").write_text(API_KEY + "\n", encoding="utf-8")
    argv = started(script, tmp_path, env={"LMDS_KEY_ROOT": str(key_store)})
    assert command_after_image(argv, IMAGE) == [*SERVE, "--api-key", API_KEY]


def test_the_equals_form_is_rebuilt_as_one_argument(tmp_path, key_store):
    script = adopt_mod.render_controller(inspected(_vllm(f"--api-key={API_KEY}")), "gemma")
    key_store.mkdir()
    (key_store / "gemma").write_text(API_KEY + "\n", encoding="utf-8")
    argv = started(script, tmp_path, env={"LMDS_KEY_ROOT": str(key_store)})
    assert command_after_image(argv, IMAGE) == [*SERVE, f"--api-key={API_KEY}"]


def test_a_key_from_the_shell_wins_over_the_store(tmp_path, key_store):
    script = adopt_mod.render_controller(inspected(_vllm("--api-key", API_KEY)), "gemma")
    key_store.mkdir()
    (key_store / "gemma").write_text("stored-key\n", encoding="utf-8")
    argv = started(script, tmp_path, env={"LMDS_KEY_ROOT": str(key_store), "LMDS_ARG_API_KEY": "from-shell"})
    assert has_option(argv, "--api-key", "from-shell")


def test_without_the_key_the_server_is_not_started_open(tmp_path, key_store):
    """ของเดิมมี auth · start โดยไม่มี key = เปิดโล่งบน 0.0.0.0 ทั้งที่เจ้าของตั้ง key ไว้ — ต้องไม่ยอม"""
    script = adopt_mod.render_controller(inspected(_vllm("--api-key", API_KEY)), "gemma")
    done, argv = run_controller(script, tmp_path, "start", env={"LMDS_KEY_ROOT": str(key_store)})
    assert done.returncode != 0
    assert argv is None, f"docker run ต้องไม่ถูกเรียกเมื่อไม่มี key: {argv}"
    assert "lmds key set gemma" in done.stderr, done.stderr


def test_a_missing_non_auth_secret_is_left_out_like_an_unset_env(tmp_path, key_store):
    """`--hf-token` ไม่ใช่ auth ของเซิร์ฟเวอร์ — ไม่มีค่าก็แค่ไม่ใส่ธงนั้น (weight อยู่ในเครื่องแล้ว)
    เหมือนที่ `--env HF_TOKEN` ถูก docker ข้ามเมื่อเชลล์ไม่มีค่า · ใส่ `--hf-token ""` คือคำสั่งพัง"""
    script = adopt_mod.render_controller(inspected(_vllm("--hf-token", HF_TOKEN, "--trust-remote-code")), "gemma")
    done, argv = run_controller(script, tmp_path, "start", env={"LMDS_KEY_ROOT": str(key_store)})
    assert done.returncode == 0, done.stderr
    assert command_after_image(argv, IMAGE) == [*SERVE, "--trust-remote-code"]
    assert "LMDS_ARG_HF_TOKEN" in done.stderr, "ต้องบอกว่าข้ามธงไหนไป"

    argv = started(script, tmp_path, env={"LMDS_KEY_ROOT": str(key_store), "LMDS_ARG_HF_TOKEN": "hf_given"})
    assert command_after_image(argv, IMAGE) == [*SERVE, "--hf-token", "hf_given", "--trust-remote-code"]


def test_a_secret_inside_a_shell_wrapped_command_is_redacted_too(tmp_path, key_store):
    """เคส spark-03: ทั้งคำสั่งอยู่ในสตริงเดียวของ `bash -c` — ไล่ทีละ token ไม่เจอ --api-key"""
    command = (f'export HF_TOKEN={HF_TOKEN} && vllm serve "/models/my model" --port 8000 '
               f"--api-key {API_KEY} --served-model-name 'a b'")
    payload = container_payload(path="bash", args=["-c", command], entrypoint=None, cmd=["bash", "-c", command])
    script = adopt_mod.render_controller(inspected(payload), "gemma")
    assert API_KEY not in script and HF_TOKEN not in script

    argv = started(script, tmp_path, env={"LMDS_KEY_ROOT": str(key_store),
                                           "LMDS_ARG_API_KEY": API_KEY, "HF_TOKEN": HF_TOKEN})
    assert command_after_image(argv, IMAGE) == ["-c", command], "ต้องได้สตริงเดิมกลับมาทุกอักขระ"


def test_flags_that_only_look_secret_are_left_alone(tmp_path):
    """`--max-tokens` `--tokenizer` `--api-key-file` ไม่ใช่ความลับ — ถอดค่าออก = คำสั่งพังโดยไม่จำเป็น"""
    extra = ["--max-num-batched-tokens", "8192", "--tokenizer", "/models/tok", "--api-key-file", "/run/key",
             "--tokenizer-mode", "auto"]
    argv = started(adopt_mod.render_controller(inspected(_vllm(*extra)), "gemma"), tmp_path)
    assert command_after_image(argv, IMAGE) == [*SERVE, *extra]


def test_a_token_shaped_value_under_an_innocent_flag_is_caught(tmp_path):
    """ตาข่ายชั้นสอง: รูปแบบ token ที่รู้จัก (sk-… · hf_…) หลุดมาในธงที่ชื่อไม่บอกว่าเป็นความลับ"""
    script = adopt_mod.render_controller(
        inspected(_vllm("--header", f"Authorization: Bearer {API_KEY}")), "gemma")
    assert API_KEY not in script


def test_adopt_reports_which_values_it_withheld():
    adopted = inspected(_vllm("--api-key", API_KEY, "--hf-token", HF_TOKEN))
    said = "\n".join(adopt_mod.not_reproduced(adopted))
    assert "--api-key" in said and "--hf-token" in said
    assert API_KEY not in said and HF_TOKEN not in said


# ── adopt ทั้งเส้น: ไฟล์ที่วางบนดิสก์และ zip ─────────────────────────────────────────────
def test_nothing_adopt_writes_into_the_bundle_contains_the_secret(tmp_path):
    payload = _vllm("--api-key", API_KEY, "--hf-token", HF_TOKEN,
                    env=["VLLM_API_KEY=sk-env-should-be-redacted-123"])
    controller = adopt_from(payload, tmp_path / "bundles")
    for path in controller.parent.iterdir():
        text = path.read_text(encoding="utf-8", errors="replace")
        assert API_KEY not in text and HF_TOKEN not in text and "sk-env-should" not in text, path.name

    from lmds.packager import make_zip

    with zipfile.ZipFile(make_zip(controller.parent)) as archive:
        for name in archive.namelist():
            body = archive.read(name).decode("utf-8", "replace")
            assert API_KEY not in body and HF_TOKEN not in body, f"{name} ใน zip ที่ node push ส่ง"


def test_adopt_keeps_the_running_key_in_the_machine_store_so_restart_still_has_auth(tmp_path, key_store):
    """ถอดค่าออกจากสคริปต์แล้วไม่เก็บไว้ที่ไหนเลย = restart ครั้งแรกไม่มี key · ที่เก็บของเครื่อง
    (0600 · นอก bundle · ไม่ติดไปกับ zip) คือที่ของมัน — เครื่องเดียวกับที่ key อยู่บน argv อยู่แล้ว"""
    controller = adopt_from(_vllm("--api-key", API_KEY), tmp_path / "bundles")
    stored = key_store / "vllm-gemma4"
    assert stored.read_text(encoding="utf-8").strip() == API_KEY
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600

    argv = started(controller.read_text(encoding="utf-8"), tmp_path, env={"LMDS_KEY_ROOT": str(key_store)})
    assert has_option(argv, "--api-key", API_KEY)


def test_a_key_already_in_the_store_is_never_overwritten(tmp_path, key_store):
    key_store.mkdir()
    (key_store / "vllm-gemma4").write_text("operator-set-this\n", encoding="utf-8")
    adopt_from(_vllm("--api-key", API_KEY), tmp_path / "bundles")
    assert (key_store / "vllm-gemma4").read_text(encoding="utf-8").strip() == "operator-set-this"


def test_the_old_script_kept_as_a_backup_is_scrubbed(tmp_path):
    """สคริปต์ที่ adopt รุ่นก่อนเขียนไว้มี key ตัวเต็ม · เก็บสำรองไว้ทั้งอย่างนั้น = key ยังอยู่ใน bundle
    (และ zip กวาดทุกไฟล์ในโฟลเดอร์) — สำรองได้ แต่ต้องไม่มีค่าความลับ"""
    bundle = tmp_path / "bundles" / "vllm-gemma4"
    bundle.mkdir(parents=True)
    old = bundle / "vllm-gemma4-adopted.sh"
    old.write_text(f"#!/usr/bin/env bash\n# รุ่นเก่า\ndocker run img --api-key {API_KEY}\n", encoding="utf-8")
    old.chmod(0o755)

    adopt_from(_vllm("--api-key", API_KEY), tmp_path / "bundles")
    backups = [p for p in bundle.iterdir() if ".replaced-" in p.name]
    assert len(backups) == 1, "ของเดิมที่ต่างจากตัวใหม่ต้องถูกเก็บไว้ ไม่ใช่เขียนทับเงียบ ๆ"
    assert "# รุ่นเก่า" in backups[0].read_text(encoding="utf-8")
    assert API_KEY not in backups[0].read_text(encoding="utf-8")


# ── process ที่รันตรง ๆ (native) ──────────────────────────────────────────────────────────
NATIVE_ARGV = ["/opt/llama/llama-server", "-m", "/models/x.gguf", "-ngl", "99", "--port", "8080",
               "--api-key", API_KEY, "--hf-token", HF_TOKEN]

# ตัวแทน llama-server: จด argv ที่ได้รับแล้วจบ · setsid/nohup ปลอมเพราะ macOS ไม่มี setsid
FAKE_SERVER = '#!/bin/bash\nfor a in "$@"; do printf \'%s\\0\' "$a"; done > "$FAKE_ARGV.server"\n'
PASS_THROUGH = '#!/bin/bash\nexec "$@"\n'


def _native(argv=NATIVE_ARGV):
    return adopt_mod.AdoptedProcess(pid=4242, argv=list(argv), exe="/opt/llama/llama-server", cwd="/opt/llama")


def _start_native(script, tmp_path, env):
    import time

    home = tmp_path / "home"
    server = home / "llama-server"
    done, _ = run_controller(script, tmp_path, "start",
                             env={"SERVER_BIN": str(server), "WORK_DIR": str(home), **env},
                             extra_bins={"setsid": PASS_THROUGH, "nohup": PASS_THROUGH})
    log = tmp_path / "docker-run.argv.server"
    if done.returncode == 0:
        for _ in range(50):          # start ปล่อยเซิร์ฟเวอร์ไว้เบื้องหลัง — รอให้มันเขียน argv
            if log.exists() and log.read_bytes():
                break
            time.sleep(0.1)
    return done, (log.read_bytes().decode().split("\0")[:-1] if log.exists() else None)


def _install_fake_server(tmp_path):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    server = home / "llama-server"
    server.write_text(FAKE_SERVER, encoding="utf-8")
    server.chmod(0o755)


def test_native_secret_flag_values_never_reach_the_script():
    script = adopt_mod.render_native_controller(_native(), "x")
    assert API_KEY not in script and HF_TOKEN not in script
    equals = adopt_mod.render_native_controller(
        _native(["/opt/llama/llama-server", "-m", "/m.gguf", f"--api-key={API_KEY}"]), "x")
    assert API_KEY not in equals


def test_native_start_feeds_the_key_from_the_store(tmp_path, key_store):
    _install_fake_server(tmp_path)
    key_store.mkdir()
    (key_store / "x").write_text(API_KEY + "\n", encoding="utf-8")
    done, argv = _start_native(adopt_mod.render_native_controller(_native(), "x"), tmp_path,
                               {"LMDS_KEY_ROOT": str(key_store)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert has_option(argv, "--api-key", API_KEY)
    assert has_option(argv, "-m", "/models/x.gguf") and has_option(argv, "-ngl", "99")
    assert "--hf-token" not in argv, "ไม่มีค่าให้ใส่ = ไม่ใส่ธง"


def test_native_start_refuses_to_come_up_without_its_key(tmp_path, key_store):
    _install_fake_server(tmp_path)
    done, argv = _start_native(adopt_mod.render_native_controller(_native(), "x"), tmp_path,
                               {"LMDS_KEY_ROOT": str(key_store)})
    assert done.returncode != 0
    assert argv is None, "เซิร์ฟเวอร์ต้องไม่ถูกปล่อยขึ้นมาแบบไม่มี auth"


def test_native_adopt_leaves_no_secret_in_the_bundle_or_the_profile(tmp_path, monkeypatch, key_store):
    monkeypatch.setattr(adopt_mod, "inspect_process", lambda **kw: _native())
    monkeypatch.setattr(adopt_mod, "probe_server", lambda *a, **k: {})
    controller, _ = adopt_mod.adopt_process(pid=4242, slug="x", output=tmp_path / "bundles")
    for path in controller.parent.iterdir():
        text = path.read_text(encoding="utf-8", errors="replace")
        assert API_KEY not in text and HF_TOKEN not in text, path.name

    profile = yaml.safe_load((controller.parent / "MODEL_PROFILE.yaml").read_text(encoding="utf-8"))
    recorded = profile["source_process"]["argv"]
    assert recorded[:7] == NATIVE_ARGV[:7], "argv ที่ไม่ใช่ความลับยังต้องตรวจย้อนได้"
    assert "--api-key" in recorded and "--hf-token" in recorded, "ธงยังอยู่ — หายแค่ค่า"
    assert (key_store / "x").read_text(encoding="utf-8").strip() == API_KEY
