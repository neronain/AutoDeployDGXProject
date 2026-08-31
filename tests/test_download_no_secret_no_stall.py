"""ตัวโหลดฝั่ง vLLM/SGLang: ความลับต้องไม่อยู่ใน argv และต้องไม่ค้างตาย

สองเคสจริงจาก spark-head 2026-08-31 ระหว่างโหลด NVFP4 170.9 GB:

1. token โผล่เต็ม ๆ ใน `ps` ของทั้งเครื่อง —
   `docker run ... -e HF_TOKEN=hf_xxxx ...` เอาค่าจริงไปแปะไว้ใน argv
   LMDS มีหลักเรื่องนี้อยู่แล้วในเส้นทาง SSH (ส่งทาง stdin เท่านั้น) แต่เส้นทาง docker
   หลุดหลักนั้นไป

2. โหลดไปได้ 36 GB แล้วหยุดนิ่ง **1 ชั่วโมง 32 นาที** โดยไม่มี error ไม่มี log ·
   connection ไปหา CDN ยังเปิดค้าง process นอนหลับใน recv() รอข้อมูลที่ไม่มีวันมา ·
   ฝั่ง llama.cpp กันไว้แล้วด้วย --speed-limit/--speed-time ของ curl แต่
   huggingface_hub ในคอนเทนเนอร์ไม่มีตัวกันแบบนั้น
   คนที่เห็นว่าค้างมักสั่งใหม่ทับ แล้วตัวใหม่ไปติด lock ของตัวเดิม — กองซ้อนกันสามตัว
   ที่ไม่มีตัวไหนทำงานเลย
"""

import pathlib
import subprocess
import tempfile
import textwrap

import pytest

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(preset: str, weight_gib: float) -> str:
    report = ModelReport(
        repo_id="org/gated-model", revision_sha="sha",
        artifact_type=ArtifactType.SAFETENSORS, weight_bytes=int(weight_gib * GIB),
        context_length=131072, kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS[preset])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, pathlib.Path(tempfile.mkdtemp()))
    return next(bundle.directory.glob("*.sh")).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def controllers():
    return {"single": _controller("dgx-spark-single", 40),
            "stacked": _controller("dgx-spark-stacked", 160)}


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_the_token_value_never_reaches_docker_argv(controllers, kind):
    """`-e HF_TOKEN` เฉย ๆ = docker หยิบค่าจาก env · `-e HF_TOKEN=ค่า` = ทุกคนอ่านได้จาก ps"""
    text = controllers[kind]
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert 'HF_TOKEN=${HF_TOKEN}' not in code, "เอาค่า token ไปแปะใน argv ของ docker"
    assert 'HF_TOKEN="$HF_TOKEN"' not in code
    assert "-e HF_TOKEN" in code, "ยังต้องส่ง token ให้คอนเทนเนอร์ผ่านชื่อตัวแปร"


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_a_download_that_stops_moving_is_restarted(controllers, kind):
    """เงื่อนไขคือ 'ยังคืบหน้าอยู่ไหม' ไม่ใช่แค่ 'process ยังไม่ตาย'"""
    text = controllers[kind]
    assert "DOWNLOAD_STALL_SECONDS" in text, "ไม่มีตัวกันค้างตาย"
    assert "_download_bytes" in text, "ไม่ได้วัดว่าไบต์เพิ่มขึ้นจริงไหม"
    # ต้องหยุดตัวที่ค้างก่อนเริ่มใหม่ ไม่งั้นตัวใหม่ไปติด lock ของตัวเดิม
    assert "docker stop" in text
    assert "DOWNLOAD_MAX_ATTEMPTS" in text, "ต้องมีเพดานรอบ ไม่วนไม่รู้จบ"


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_the_socket_read_has_a_timeout(controllers, kind):
    """ปล่อยให้ค่า default ของ huggingface_hub ตัดสินใจเองไม่พอ — ระบุให้ชัด"""
    assert "HF_HUB_DOWNLOAD_TIMEOUT" in controllers[kind]


@pytest.mark.parametrize("kind", ["single", "stacked"])
def test_docker_format_strings_survive_templating(controllers, kind):
    """`{{.State.Running}}` ของ docker ชนกับไวยากรณ์ของ jinja — ต้องออกมาเป็นของ docker

    เจอจริงตอนเพิ่มตัวเฝ้า: escape ผิดที่ (โค้ดอยู่ใน {% raw %} อยู่แล้ว) ทำให้ไฟล์ที่เจน
    ออกมามี `{{ '{{.State.Running}}' }}` ตรง ๆ ซึ่ง docker อ่านไม่ออก
    """
    text = controllers[kind]
    assert "{{.State.Running}}" in text
    assert "'{{.State.Running}}' }}" not in text, "escape เกิน — jinja ไม่ได้คายออกมาให้"


def _watchdog_loop(text: str) -> str:
    """ตัดเฉพาะลูปเฝ้าความคืบหน้าออกมา — หา `while` แล้วไล่ถึง `done` บรรทัดแรกที่ระดับเดียวกัน"""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("while [[ \"$(docker inspect"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "done")
    return "\n".join(lines[start : end + 1])


def _run_watchdog(tmp_path, sizes: list[int]) -> subprocess.CompletedProcess:
    """รันลูปจริงโดยป้อนขนาดโฟลเดอร์ตามลำดับที่กำหนด — docker/sleep/log เป็นตัวปลอม

    คืน exit 75 หรือพิมพ์ KILLED = watchdog ตัดสินว่าค้าง
    """
    text = _controller("dgx-spark-stacked", 160)
    harness = textwrap.dedent(
        """
        set -euo pipefail
        DOWNLOAD_STALL_SECONDS=90
        DOWNLOAD_MAX_ATTEMPTS=1
        dl_name=stub
        SIZES=({sizes})
        i=0
        _download_bytes() { echo "${SIZES[$i]:-${SIZES[-1]}}"; }
        sleep() { i=$(( i + 1 )); }
        log() { echo "$*"; }
        die() { echo "KILLED: $*"; exit 75; }
        download() { echo "KILLED: restarted"; exit 75; }
        docker() {
          case "$1" in
            inspect) [[ $i -lt ${#SIZES[@]} ]] && echo true || echo false ;;
            *) : ;;
          esac
        }
        logs_pid=0
        kill() { :; }
        last="$(_download_bytes)"
        quiet=0
        now=
        {loop}
        echo SURVIVED
        """
    ).replace("{sizes}", " ".join(str(s) for s in sizes)).replace("{loop}", _watchdog_loop(text))
    script = tmp_path / "watchdog.sh"
    script.write_text(harness, encoding="utf-8")
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30)


def test_a_download_that_shrinks_then_grows_is_not_called_stalled(tmp_path):
    """restart ทำให้โฟลเดอร์ **เล็กลง** ก่อน แล้วค่อยโตใหม่ — นั่นคือกำลังโหลด ไม่ใช่ค้าง

    เคสจริง 2026-09-01 บน spark-head: ของเดิมเก็บ last เป็นค่าสูงสุดที่เคยเห็น (164.6 GB)
    พอ container ใหม่ตัด .incomplete ทิ้งเหลือ 153.9 GB ทุกตัวอย่างหลังจากนั้นก็ต่ำกว่า
    ค่าเดิมตลอด → watchdog ฆ่า download ที่ทำงานดี ๆ ทุก 10 นาที แล้ววนแบบนั้นไม่รู้จบ
    """
    shrink_then_grow = [164_600, 153_900, 154_100, 155_000, 156_400, 158_000]
    r = _run_watchdog(tmp_path, shrink_then_grow)
    assert "KILLED" not in r.stdout, f"หาว่าค้างทั้งที่กำลังโหลดอยู่:\n{r.stdout}\n{r.stderr}"
    assert "SURVIVED" in r.stdout, r.stdout + r.stderr


def test_a_download_that_truly_stops_moving_is_still_killed(tmp_path):
    """ค้างจริง = ขนาดไม่ขยับเลย — ตัวกันต้องยังทำงาน ไม่ใช่แก้จนกลายเป็นไม่กันอะไรเลย"""
    frozen = [153_900, 153_900, 153_900, 153_900, 153_900, 153_900]
    r = _run_watchdog(tmp_path, frozen)
    assert "KILLED" in r.stdout, f"ค้างนิ่งสนิทแล้วยังไม่ยอมเริ่มใหม่:\n{r.stdout}\n{r.stderr}"


def test_restarting_after_a_stall_turns_xet_off(controllers):
    """ค้างแล้วเริ่มใหม่ ต้องไม่เริ่มใหม่แบบเดิมเป๊ะ ๆ

    เคสจริง 2026-09-01 บน spark-head: Xet client ค้างสนิท ทุกเธรดจอดที่ futex_do_wait
    socket เปิดค้าง 11 เส้น log ว่าง 12 นาทีไม่ขยับสักไบต์ · "ค้าง" ไม่เคยคืน exit code
    จึงไม่เข้าทาง fallback ที่ปิด Xet · เริ่มใหม่ทั้งที่ยังเปิด Xet = ค้างซ้ำที่เดิมจนครบโควตา

    เจาะที่ *บรรทัดสั่งเริ่มใหม่* ตรง ๆ — ดูแค่ว่า HF_HUB_DISABLE_XET โผล่แถว ๆ นั้นไหม
    ไม่พอ เพราะ fallback เดิม (ที่ทำงานเฉพาะตอน exit code ไม่เป็นศูนย์) ก็อยู่ใกล้ ๆ กัน
    """
    stacked = [l for l in controllers["stacked"].splitlines() if "DL_ATTEMPT=$((" in l and "download" in l]
    assert stacked, "ไม่เจอบรรทัดสั่งเริ่มใหม่หลังค้างของ stacked"
    assert all("HF_HUB_DISABLE_XET=1" in l for l in stacked), (
        f"stacked เริ่มใหม่หลังค้างโดยยังเปิด Xet: {stacked}"
    )

    lines = controllers["single"].splitlines()
    i = next(n for n, l in enumerate(lines) if "code == 75" in l)
    window = "\n".join(lines[i : i + 8])
    assert "disable_xet=1" in window, (
        f"single เริ่มใหม่หลังค้างโดยยังเปิด Xet:\n{window}"
    )
