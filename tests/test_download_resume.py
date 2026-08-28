"""download ต้องได้ไฟล์ *ครบตามขนาดจริง* ไม่ใช่แค่ curl คืน 0

เคสจริง 2026-08-28 บน msi-5 · ไฟล์ Q5_K_M ขนาด 20.3GB หลุดที่ 3,967MB ด้วย

    curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)

CDN ของ HF ตัดสตรีมกลางคันเมื่อโหลดยาว ๆ · `--retry` ของ curl ไม่ยิงซ้ำให้ เพราะ
curl นับ transient error แค่ timeout / 408 / 429 / 5xx — error 92 ไม่อยู่ในชุดนั้น
curl จึงจบทันทีโดยเหลือไฟล์ถูกตัดครึ่งไว้ แล้วไม่มีใครรู้จนกว่าจะมีคนสั่ง verify-files เอง

เทสนี้รัน fetch_one ที่เจนออกมาจริง ๆ ใต้ set -euo pipefail กับ curl ปลอมที่จำลอง
พฤติกรรมนั้นเป๊ะ ๆ — ตรวจแค่ว่าสคริปต์ *มี* --retry-all-errors ไม่พอ เพราะเงื่อนไข
จบที่แท้จริงคือขนาดไฟล์ ไม่ใช่ exit code
"""

import subprocess
import textwrap

from lmds.brain import build_plan
from lmds.fit import PRESETS, analyze
from lmds.fit.analyzer import GIB
from lmds.generator import render_bundle
from lmds.inspector.report import ArtifactType, KvDims, ModelReport


def _controller(tmp_path) -> str:
    report = ModelReport(
        repo_id="wangzhang/gemma-4-31B-it-abliterated-GGUF",
        revision_sha="sha",
        artifact_type=ArtifactType.GGUF,
        weight_bytes=int(20.3 * GIB),
        selected_gguf="gemma-4-31B-it-abliterated-Q5_K_M.gguf",
        context_length=131072,
        kv_dims=KvDims(layers=48, kv_heads=4, head_dim=128),
    )
    fit = analyze(report, PRESETS["dgx-spark-single"])
    plan = build_plan(report, fit, provider=None)
    bundle = render_bundle(plan, report, fit, tmp_path)
    return next(bundle.directory.glob("*-single.sh")).read_text(encoding="utf-8")


def _extract(text: str, name: str) -> str:
    """ตัดฟังก์ชันออกมาด้วยการนับปีกกา — regex พลาดกับ { } ที่ซ้อนกัน"""
    start = text.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise AssertionError(f"ไม่เจอปีกกาปิดของ {name}")


def _fake_bin(tmp_path, curl_body: str):
    """curl/aria2c ปลอม · aria2c ล้มเสมอเพื่อบังคับเส้นทาง curl ให้เทสไม่ขึ้นกับเครื่อง"""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()

    aria = bin_dir / "aria2c"
    aria.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    aria.chmod(0o755)

    # curl_retry_all ถาม `curl --help all` ก่อนใช้ flag — ต้องตอบให้ตรงนั้น
    # ไม่งั้นการถามจะถูกนับเป็นการ "ดาวน์โหลด" หนึ่งครั้งแล้วนับรอบเพี้ยนทั้งเทส
    shim = '#!/bin/sh\ncase " $* " in *" --help "*) echo "--retry-all-errors"; exit 0;; esac\n'
    curl = bin_dir / "curl"
    curl.write_text(shim + curl_body.split("\n", 1)[1], encoding="utf-8")
    curl.chmod(0o755)
    return bin_dir


def _run(tmp_path, bin_dir, out, want, max_attempts=5):
    text = _controller(tmp_path)
    harness = textwrap.dedent(f"""
        set -euo pipefail
        die() {{ echo "ERROR: $*" >&2; exit 1; }}
        FETCH_MAX_ATTEMPTS={max_attempts}
        HF_TOKEN=""
    """) + "\n".join(
        _extract(text, name) for name in ("file_size", "curl_retry_all", "fetch_one")
    ) + f'\nfetch_one "https://example.invalid/model.gguf" "{out}" {want}\n'

    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, env=env, timeout=120
    )


def test_truncated_download_is_resumed_until_the_size_matches(tmp_path):
    """curl ตายกลางคันด้วย error 92 → ต้อง resume ต่อเองจนได้ครบ ไม่ใช่จบแล้วปล่อยไฟล์แหว่ง"""
    out = tmp_path / "model.gguf"
    state = tmp_path / "calls"
    bin_dir = _fake_bin(tmp_path, f"""#!/bin/sh
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
n=$(cat "{state}" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{state}"
if [ "$n" -eq 1 ]; then printf 'AAAA' > "$out"; exit 92; fi
printf 'BBBBBBBB' >> "$out"
""")

    proc = _run(tmp_path, bin_dir, out, 12)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.stat().st_size == 12
    assert state.read_text().strip() == "2"          # ต่อจากของเดิม ไม่เริ่มใหม่
    assert "หลุดกลางคัน 4/12" in proc.stdout
    assert "ถอยไป curl" in proc.stdout               # aria2c ล้ม → ยังไปต่อด้วย curl


def test_a_download_that_stops_making_progress_fails_loudly(tmp_path):
    """เน็ตหลุดจริง ๆ (resume แล้วไม่ได้ไบต์เพิ่ม) ต้องตายพร้อมบอกเหตุ ไม่ใช่วนไม่รู้จบ"""
    out = tmp_path / "model.gguf"
    bin_dir = _fake_bin(tmp_path, """#!/bin/sh
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -f "$out" ] || printf 'AAAA' > "$out"
exit 92
""")

    proc = _run(tmp_path, bin_dir, out, 12)

    assert proc.returncode != 0
    assert "resume แล้วไม่คืบหน้าเลย" in proc.stderr
    assert out.stat().st_size == 4        # ของเดิมไม่ถูกลบทิ้ง — resume รอบหน้าใช้ต่อได้


def test_a_clean_exit_with_a_short_body_is_still_not_complete(tmp_path):
    """proxy ที่ส่ง body สั้นแต่ปิดสตรีมเรียบร้อยได้ exit 0 — ขนาดคือเงื่อนไขจบ ไม่ใช่ exit code"""
    out = tmp_path / "model.gguf"
    state = tmp_path / "calls"
    bin_dir = _fake_bin(tmp_path, f"""#!/bin/sh
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
n=$(cat "{state}" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{state}"
printf 'AA' >> "$out"
exit 0
""")

    proc = _run(tmp_path, bin_dir, out, 12, max_attempts=3)

    assert proc.returncode != 0
    assert "ลองต่อ 3 รอบแล้วยังไม่ครบ (6/12)" in proc.stderr
