"""burn check: จับ GPU clock latch ของ GB10 — อาการเดียวของมันคือ "ช้า" เฉย ๆ

`docs/DGX-SPARK-VLLM-FIELD-NOTES.md` §10 (artifact-backed, Tech2wild 2026-09-10): EC ของ
DGX Spark ล็อก GPU ไว้ที่ **631–949 MHz** ทั้งตอน idle และตอนมีโหลด ขณะที่เครื่องอื่นในชุด
เดียวกันวิ่ง 2177–2561 MHz · และ `nvidia-smi` **ไม่แสดงอะไรผิดเลย** — P0 · persistence on ·
application clocks 2418 · ไม่มี clock event reason · ไม่ power cap ไม่ thermal · kernel log สะอาด

## ทำไมต้องมีโมดูลนี้ ทั้งที่ profiler อ่าน clocks.sm อยู่แล้ว

`profiler.detect_gpus()` เก็บ `clocks.sm`/`power.draw` มาตั้งแต่แรก — แต่ค่าพวกนั้นอ่าน **ตอน
GPU ว่าง** ซึ่งเครื่องดีกับเครื่องที่ latch ให้ตัวเลขต่ำเหมือนกันหมด (GPU ที่ไม่มีงานก็ลดคล็อกเป็น
เรื่องปกติ) · สิ่งที่แยกสองสภาพนี้ออกจากกันได้มีอย่างเดียวคือ **ใส่โหลดจริงแล้วดูว่ามันขึ้นไหม**
อินพุตเราครบมาตลอด ที่ขาดคือโหลดกับเกณฑ์ตัดสิน — นั่นคือทั้งหมดที่ไฟล์นี้เพิ่ม

## ของนี้ซ่อมจากระยะไกลไม่ได้ — หน้าที่ของสินค้าคือบอกให้ชัดว่าต้องทำอะไรทางกายภาพ

EC ตัดสินใจ DVFS อยู่ใต้ OS และยังกินไฟ standby ตราบที่ adapter เสียบอยู่ · **รีบูตไม่หาย**
ทางแก้ทางเดียวคือถอดปลั๊ก 30–60 วินาที · เพราะฉะนั้นผลลัพธ์ของที่นี่ไม่ใช่ "fail" เฉย ๆ แต่เป็น
ขั้นตอนที่คนซึ่งยืนอยู่หน้าเครื่องทำตามได้ทันที (`remedy()`) — ดูเหตุผลเต็มใน `verdict()`

## สามสภาพที่ห้ามปนกัน

ตัวเลข TFLOPS ต่ำอย่างเดียว **ไม่ได้แปลว่า latch**:

- คล็อกต่ำ + ไฟต่ำ → latch จริง (ต้องถอดปลั๊ก)
- คล็อกต่ำ + มี clock event reason (thermal/power cap) → **ไม่ใช่ latch** เพราะ latch ของ EC
  ไม่รายงาน reason ใด ๆ เลย (§10) — อันนี้ต้องไปดูลมกับไฟ ไม่ใช่ถอดปลั๊ก
- คล็อกปกติ + ไฟสูง + TFLOPS ต่ำ → มีคนอื่นใช้ GPU อยู่ (โมเดลกำลังเสิร์ฟ) ไม่ใช่ความผิดของเครื่อง

บอกลูกค้าให้ถอดปลั๊กทั้งที่จริง ๆ พัดลมตัน หรือทั้งที่จริง ๆ โมเดลกำลังทำงาน คือความเสียหาย
ที่ฟีเจอร์นี้สร้างขึ้นเอง — จึงแยกสามอย่างนี้ออกจากกันตั้งแต่ต้น

## โครง runner-injectable (เลียน `nodes/netplan.py` / `nodes/hostname.py`)

ทุกอย่างที่แตะเครื่องจริงผ่าน `runner` ตัวเดียว: `check_node()` รับ runner รูปเดียวกับ
`lmds.nodes.run(node, command, timeout=…)` · `check_local()` รับ runner รูป
`(command, timeout) -> Result` · เทสจึงเดินทั้งเส้นทางได้โดยไม่ต้องมี SSH และไม่ต้องมี GPU

**ส่งสคริปต์เชลล์ไป ไม่ได้เรียก `lmds` บน node** — ด้วยเหตุผลเดียวกับที่ netplan ทำ: node
ที่ lmds เก่ากว่า hub ยังตรวจได้ และเครื่องที่ยังไม่เคยติดตั้ง lmds ก็ตรวจได้

## ไม่ต่อเน็ต

หา torch จาก interpreter ที่มีอยู่ก่อน แล้วค่อยถอยไปใช้ image ที่ **มีอยู่ในเครื่องแล้ว**
(`docker image inspect` ไม่ใช่ `docker pull`) — SECURITY.md สัญญาว่าลูกค้า air-gapped ใช้ได้
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from .profiles import MemoryModel, lookup_gpu

# ── เกณฑ์ ────────────────────────────────────────────────────────────────────
# สองค่านี้มาจาก §10 ตรง ๆ (artifact-backed)
PASS_TFLOPS = 50.0        # "เกณฑ์ fail <50 TFLOPS"
HEALTHY_TFLOPS = 75.0     # "ปกติ 75–90 TFLOPS" — ต่ำกว่านี้แต่ยังผ่าน = ควรบอกว่าน่าสงสัย
# สองค่านี้ **ผมเลือกเอง** เป็นจุดกึ่งกลางของช่วงที่ §10 วัดไว้ ไม่ได้มีใครวัดว่าเส้นแบ่งอยู่ตรงนี้:
#   คล็อก  latched 631–949 · ปกติ 2177–2561  → เลือก 1500
#   ไฟ     latched <20 W   · ปกติ ≥80 W      → เลือก 50
# ใช้เป็น *คำอธิบาย* ว่าทำไมถึงตก ไม่ใช่เป็นตัวตัดสินว่าตกหรือไม่ — ตัวตัดสินคือ TFLOPS ตัวเดียว
LATCH_CLOCK_MHZ = 1500.0
LOW_POWER_W = 50.0

# นิยาม "เครื่องที่เรื่องนี้เกี่ยวข้อง" อยู่ที่เดียว แล้วส่งเข้าไปในเชลล์ด้วย — ถ้าปล่อยให้เชลล์มี
# กติกาของตัวเองอีกชุด สองฝั่งจะเพี้ยนกันวันที่มีคนแก้ฝั่งเดียว
GB10_MARKERS = ("gb10", "dgx spark")

DEFAULT_SECONDS = 15.0     # §10: "burn check 15 วินาที"
DEFAULT_SAMPLE_AT = 12.0   # §10: "อ่าน clocks.sm + power.draw ที่วินาทีที่ 12"
DEFAULT_MATRIX = 4096      # §10: "fp16 matmul 4096²"


@dataclass
class Result:
    """รูปเดียวกับ `lmds.nodes.ssh.Result` — พอสำหรับทั้งสองเส้นทาง"""

    exit_code: int
    stdout: str
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# ── โปรแกรมที่วิ่งบน GPU ──────────────────────────────────────────────────────
def _sample_point(seconds: float, sample_at: float | None) -> float:
    """จังหวะที่อ่าน nvidia-smi — ว่าง = 80% ของความยาว burn (12 จาก 15 ตาม §10)

    ใครที่ย่อ burn ให้สั้นลง (เทส, หรือคนที่รีบ) ต้องได้สัดส่วนเดิม ไม่ใช่ได้จุดอ่านที่เลย
    ปลายทางไปแล้ว — ซึ่งทำให้ลูปแรกวิ่งจนครบ 12 วินาทีเสมอไม่ว่าจะสั่งมาสั้นแค่ไหน
    """
    if sample_at is not None:
        return float(sample_at)
    return float(seconds) * (DEFAULT_SAMPLE_AT / DEFAULT_SECONDS)


def burn_program(seconds: float = DEFAULT_SECONDS, sample_at: float | None = None,
                 matrix: int = DEFAULT_MATRIX) -> str:
    """ซอร์ส Python ของตัว burn — คืนเป็นสตริงเพื่อให้เทสอ่านได้โดยไม่ต้องมี GPU

    **exit 3 = "interpreter ตัวนี้ทำไม่ได้ ลองตัวถัดไป"** (ไม่มี torch / ไม่มี CUDA) ·
    exit 0 พร้อมบรรทัด `LMDS_BURN {json}` = คำตอบสุดท้าย · รหัสอื่นคือพังจริง

    ลำดับการวัดที่วินาทีที่ ~12 มีเหตุผล:
      1. หมุน matmul ไปเรื่อย ๆ จนถึงวินาทีที่ 12 ก่อน — คล็อกต้องเข้าที่ก่อนถึงจะวัดได้
      2. วัด TFLOPS ในหน้าต่างที่ sync หัวท้ายชัดเจน (ไม่ปนกับเวลาที่ใช้เรียก nvidia-smi)
      3. **enqueue งานอีกชุดโดยไม่ sync แล้วค่อยอ่าน nvidia-smi** — ถ้าอ่านหลัง sync
         GPU จะว่างแล้วและคล็อกตกลงทันที ซึ่งทำให้เครื่องดี ๆ อ่านได้เหมือนเครื่องที่ latch
      4. sync ปิดท้าย แล้วหมุนต่อจนครบ 15 วินาทีตามโปรโตคอลใน §10
    """
    return f'''\
import json, subprocess, sys, time

SECONDS = {float(seconds)!r}
SAMPLE_AT = {_sample_point(seconds, sample_at)!r}
N = {int(matrix)}
CHUNK = 8

REASONS = ["clocks_event_reasons.sw_power_cap",
           "clocks_event_reasons.hw_thermal_slowdown",
           "clocks_event_reasons.sw_thermal_slowdown",
           "clocks_event_reasons.hw_slowdown"]
FIELDS = ["name", "clocks.sm", "clocks.gr", "power.draw", "temperature.gpu"] + REASONS


def smi():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return {{}}
    if proc.returncode != 0:
        return {{}}
    rows = [r for r in (proc.stdout or "").splitlines() if r.strip()]
    if not rows:
        return {{}}
    parts = [p.strip() for p in rows[0].split(",")]
    return dict(zip(FIELDS, parts))


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None      # "[N/A]" คือ "ไม่รายงาน" ไม่ใช่ศูนย์


try:
    import torch
except Exception as exc:
    sys.stderr.write("no torch: %s\\n" % str(exc)[:160])
    raise SystemExit(3)
if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
    # rc 4 ไม่ใช่ 3 — "ไม่มี torch" กับ "มี torch แต่มองไม่เห็น GPU" แก้คนละทางกันสิ้นเชิง
    sys.stderr.write("torch has no usable CUDA device\\n")
    raise SystemExit(4)

try:
    left = torch.randn(N, N, device="cuda", dtype=torch.float16)
    right = torch.randn(N, N, device="cuda", dtype=torch.float16)
except Exception as exc:
    # หน่วยความจำไม่พอเพราะโมเดลถืออยู่ = ตรวจไม่ได้ และ interpreter อื่นก็จะเจอเหมือนกัน
    print("LMDS_BURN " + json.dumps({{"error": "alloc", "detail": str(exc)[:200]}}))
    raise SystemExit(0)

torch.cuda.synchronize()
started = time.monotonic()
while time.monotonic() - started < SAMPLE_AT:
    for _ in range(CHUNK):
        left @ right
    torch.cuda.synchronize()

window = time.monotonic()
for _ in range(CHUNK):
    left @ right
torch.cuda.synchronize()
elapsed = time.monotonic() - window
tflops = (2.0 * N * N * N * CHUNK) / elapsed / 1e12 if elapsed > 0 else None

for _ in range(CHUNK):
    left @ right          # ค้างคิวไว้ให้ GPU ยังมีงานตอน nvidia-smi อ่าน
reading = smi()
torch.cuda.synchronize()

while time.monotonic() - started < SECONDS:
    for _ in range(CHUNK):
        left @ right
    torch.cuda.synchronize()

print("LMDS_BURN " + json.dumps({{
    "tflops": tflops,
    "gpu": reading.get("name", ""),
    "clock_sm_mhz": number(reading.get("clocks.sm")),
    "clock_gr_mhz": number(reading.get("clocks.gr")),
    "power_w": number(reading.get("power.draw")),
    "temperature_c": number(reading.get("temperature.gpu")),
    "throttle": [r.split(".")[-1] for r in REASONS
                 if (reading.get(r) or "").strip().lower() == "active"],
    "seconds": SECONDS,
    "matrix": N,
    "torch": getattr(torch, "__version__", ""),
}}))
'''


def burn_script(seconds: float = DEFAULT_SECONDS, sample_at: float | None = None,
                matrix: int = DEFAULT_MATRIX, force: bool = False) -> str:
    """สคริปต์เชลล์ที่ส่งไปรันบนเครื่อง — พิมพ์บรรทัดเดียวขึ้นต้นด้วย `LMDS_BURN…`

    หา interpreter ที่มี torch ตามลำดับ **จากถูกไปแพง**:
      1. `$LMDS_BURN_PYTHON` — ผู้ดูแลชี้เองได้เสมอ (venv, conda, อะไรก็ได้)
      2. `python3` / `python` ของเครื่อง
      3. `docker run --gpus all` ด้วย image ที่ **มีอยู่ในเครื่องแล้ว** — บน Spark ที่เคยรัน
         vLLM/SGLang torch อยู่ในนั้นแน่นอน และนั่นคือ torch ตัวเดียวกับที่เสิร์ฟจริงด้วย

    ข้อ 3 ใช้ `docker image inspect` คัดก่อน **ไม่ pull** — ลูกค้า air-gapped มีจริง และ
    SECURITY.md สัญญาว่าไม่มีอะไรวิ่งออกเน็ตนอกจากตอน download weight/image ที่ผู้ใช้สั่งเอง

    `force=True` ข้ามด่าน "ใช่ GB10 ไหม" — เกณฑ์ใน §10 เป็นของ GB10 ล้วน การเอาไปตัดสิน
    RTX จึงผิดตั้งแต่ต้น แต่ผู้ดูแลที่อยากได้ตัวเลขดิบก็ควรสั่งได้
    """
    program = burn_program(seconds, sample_at, matrix)
    markers = "|".join(f"*{m}*" for m in GB10_MARKERS).replace(" ", r"\ ")
    gate = "" if force else f'''
case "$lower" in
  {markers}) ;;
  *) printf 'LMDS_BURN_SKIP %s\\n' "$name"; exit 0 ;;
esac
'''
    return f'''
set -u
command -v nvidia-smi >/dev/null 2>&1 || {{ echo 'LMDS_BURN_NOSMI'; exit 0; }}
name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$name" ] || {{ echo 'LMDS_BURN_NOGPU'; exit 0; }}
lower="$(printf '%s' "$name" | tr 'A-Z' 'a-z')"
printf 'LMDS_BURN_GPU %s\\n' "$name"
{gate}
prog=$(cat <<'LMDS_BURN_PY'
{program}
LMDS_BURN_PY
)

tried=""; last=""; sawcuda=""
# mktemp ไม่ใช่ /tmp/<ชื่อที่เดาได้>.$$ — เครื่องที่ใช้ร่วมกันหลายคนมีจริง และ pid เดาได้
errf=$(mktemp "${{TMPDIR:-/tmp}}/lmds-burn.XXXXXX") || errf=/dev/null
trap 'rm -f "$errf"' EXIT INT TERM

attempt() {{
  tried="$tried $1"
  out=$(printf '%s\\n' "$prog" | "$@" - 2>"$errf"); rc=$?
  why="$1 rc=$rc $(tr '\\n' ' ' < "$errf" 2>/dev/null | cut -c1-160)"
  # 3 = ไม่มี torch · 4 = มี torch แต่มองไม่เห็น GPU · 126,127 = เรียกไม่ได้เลย
  # — ทั้งสี่คือ "ลองตัวถัดไป" แต่ 4 ต้องจำไว้ เพราะมันแก้คนละทางกับอีกสามตัว
  case "$rc" in 4) sawcuda=1; last="$why"; return 1 ;; esac
  case "$rc" in 3|126|127) last="$why"; return 1 ;; esac
  line=$(printf '%s\\n' "$out" | grep -m1 '^LMDS_BURN ' || true)
  if [ -n "$line" ]; then printf '%s\\n' "$line"; exit 0; fi
  # ตอบมาแล้วแต่ไม่ใช่คำตอบ = พังจริง ไม่ใช่ "ลองตัวอื่น" — หยุดแล้วบอกไปตรง ๆ
  printf 'LMDS_BURN_FAILED %s\\n' "$why"
  exit 0
}}

for py in "${{LMDS_BURN_PYTHON:-}}" python3 python; do
  [ -n "$py" ] || continue
  command -v "$py" >/dev/null 2>&1 || continue
  attempt "$py" || continue
done

if command -v docker >/dev/null 2>&1; then
  # image ที่ bundle บนเครื่องนี้ใช้จริง — จาก bundle.env (ผู้ใช้ตั้งเอง) และจากค่า default
  # ที่ฝังอยู่ในตัว controller · เอาเฉพาะที่ `docker image inspect` เห็น = มีอยู่แล้วจริง
  images=$( {{ sed -n 's/^[A-Z_]*IMAGE=//p' ~/bundles/*/bundle.env ~/*/bundles/*/bundle.env 2>/dev/null
              sed -n 's/.*[A-Z_]*IMAGE:-\\([^}}]*\\)}}.*/\\1/p' ~/bundles/*/*.sh ~/*/bundles/*/*.sh 2>/dev/null
              printf '%s\\n' "${{LMDS_BURN_IMAGE:-}}"; }} \\
            | sed 's/^ *//; s/ *$//; s/^"//; s/"$//' | grep -v '^$' | sort -u )
  for img in $images; do
    docker image inspect "$img" >/dev/null 2>&1 || continue
    attempt docker run --rm -i --gpus all --entrypoint python3 "$img" || continue
  done
fi

# มี torch อยู่แล้วแต่ CUDA ใช้ไม่ได้ = คนละปัญหากับ "ไม่มี torch" และคนละทางแก้
# บอกแยกกัน ไม่งั้นคนไปไล่ติดตั้ง torch ที่ติดตั้งอยู่แล้ว
if [ -n "$sawcuda" ]; then
  printf 'LMDS_BURN_NOCUDA tried:%s · last: %s\\n' "$tried" "$last"
else
  printf 'LMDS_BURN_NOPYTHON tried:%s · last: %s\\n' "$tried" "$last"
fi
'''


# ── อ่านผล ───────────────────────────────────────────────────────────────────
def parse(text: str) -> dict:
    """แปลผลดิบจากเครื่องเป็น dict — คีย์ `error` บอกว่าทำไมถึงไม่มีตัวเลข

    ไม่เคยโยน exception: สิ่งที่กลับมาจาก SSH เป็นอะไรก็ได้ รวมถึง banner ของ sshd,
    ข้อความ motd, หรือคำเตือนของ docker ที่แทรกมาก่อนบรรทัดของเรา
    """
    out: dict = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("LMDS_BURN "):
            try:
                payload = json.loads(line[len("LMDS_BURN "):])
            except ValueError:
                continue
            if isinstance(payload, dict):
                out.update(payload)
        elif line.startswith("LMDS_BURN_GPU "):
            out.setdefault("gpu", line[len("LMDS_BURN_GPU "):].strip())
        elif line == "LMDS_BURN_NOSMI":
            out["error"] = "no-nvidia-smi"
        elif line == "LMDS_BURN_NOGPU":
            out["error"] = "no-gpu"
        elif line.startswith("LMDS_BURN_SKIP "):
            out["error"] = "not-gb10"
            out["gpu"] = line[len("LMDS_BURN_SKIP "):].strip()
        elif line.startswith("LMDS_BURN_NOCUDA"):
            out["error"] = "no-cuda"
            out["detail"] = line[len("LMDS_BURN_NOCUDA"):].strip()
        elif line.startswith("LMDS_BURN_NOPYTHON"):
            out["error"] = "no-torch"
            out["detail"] = line[len("LMDS_BURN_NOPYTHON"):].strip()
        elif line.startswith("LMDS_BURN_FAILED "):
            out["error"] = "burn-failed"
            out["detail"] = line[len("LMDS_BURN_FAILED "):].strip()
    # `LMDS_BURN {"error": "alloc"}` เอาชนะบรรทัด GPU ที่พิมพ์ไปก่อนหน้า — ตัวหลังคือคำตอบ
    return out


def applies_to(gpu_name: str) -> bool:
    """เรื่อง clock latch เกี่ยวกับเครื่องนี้ไหม — GB10 เท่านั้น

    ใช้ `lookup_gpu` ด้วยเพื่อไม่ให้กติกาแตกเป็นสองชุด: allowlist ของ profiles บอกอยู่แล้วว่า
    ตัวไหน unified memory (= Spark) · ตัวเชลล์ใช้ substring เดียวกันเพื่อ *ประหยัด* เท่านั้น
    คำตัดสินจริงอยู่ที่นี่
    """
    lowered = (gpu_name or "").lower()
    if any(marker in lowered for marker in GB10_MARKERS):
        return True
    known = lookup_gpu(gpu_name or "")
    return bool(known and known.memory_model is MemoryModel.UNIFIED)


def verdict(payload: dict) -> dict:
    """ตัดสินจากผลดิบ — คืน `{"kind", "ok", "applies", …ค่าที่วัดได้}`

    `kind` มี 7 แบบและ **ห้ามปนกัน** (เหตุผลเต็มอยู่ใน docstring ของโมดูล):

    - `ok`           ผ่าน (≥50 TFLOPS) · `suspect=True` เมื่อ <75 ซึ่งยังต่ำกว่าช่วงปกติ
    - `latched`      ตก + คล็อกต่ำ + ไฟต่ำ + **ไม่มี** clock event reason → EC ล็อกจริง
    - `throttled`    ตก + คล็อกต่ำ + **มี** clock event reason → ไม่ใช่ latch (latch ไม่รายงานอะไรเลย)
    - `contended`    ตก แต่คล็อก/ไฟปกติ → มีใครใช้ GPU อยู่ ตัวเลขนี้ตัดสินเครื่องไม่ได้
    - `slow`         ตก แต่ `nvidia-smi` ไม่รายงานคล็อก/ไฟเลย → ตกจริงแต่บอกสาเหตุไม่ได้
    - `unknown`      ตรวจไม่ได้ (ไม่มี torch, alloc ไม่ผ่าน, สคริปต์ล้ม)
    - `not-applicable` ไม่ใช่ GB10 / ไม่มี NVIDIA → **ไม่ใช่ fail**

    `ok` เป็น False เฉพาะเมื่อ "ตรวจได้ และตก" — เครื่องที่ตรวจไม่ได้หรือไม่เกี่ยวต้องไม่ถูก
    นับเป็นเครื่องเสีย ไม่งั้นทุกเครื่องที่ไม่ใช่ Spark ในฟลีตจะขึ้นแดงตลอดกาล
    """
    gpu = str(payload.get("gpu") or "")
    error = payload.get("error") or ""
    base = {"gpu": gpu, "applies": applies_to(gpu), "detail": str(payload.get("detail") or ""),
            "tflops": payload.get("tflops"), "clock_sm_mhz": payload.get("clock_sm_mhz"),
            "clock_gr_mhz": payload.get("clock_gr_mhz"), "power_w": payload.get("power_w"),
            "temperature_c": payload.get("temperature_c"),
            "throttle": list(payload.get("throttle") or []),
            "seconds": payload.get("seconds"), "matrix": payload.get("matrix")}

    if error in {"no-nvidia-smi", "no-gpu", "not-gb10"}:
        return {**base, "kind": "not-applicable", "ok": True, "applies": False, "reason": error}
    if error or payload.get("tflops") is None:
        return {**base, "kind": "unknown", "ok": True, "reason": error or "no-reading"}
    if not base["applies"]:
        # วัดได้แต่ไม่ใช่ GB10 (เช่นสั่ง --force บน RTX) — รายงานตัวเลข ไม่ตัดสิน
        return {**base, "kind": "not-applicable", "ok": True, "reason": "thresholds-are-gb10"}

    tflops = float(payload["tflops"])
    clock = base["clock_sm_mhz"] if base["clock_sm_mhz"] is not None else base["clock_gr_mhz"]
    power = base["power_w"]
    if tflops >= PASS_TFLOPS:
        return {**base, "kind": "ok", "ok": True, "suspect": tflops < HEALTHY_TFLOPS}

    # ไฟที่ "ไม่รายงาน" ไม่ใช่ไฟต่ำ — แต่ก็ไม่ควรทำให้ latch ที่ชัดจากคล็อกกลายเป็นอย่างอื่น
    low_clock = clock is not None and clock < LATCH_CLOCK_MHZ
    low_power = power is None or power < LOW_POWER_W
    if base["throttle"]:
        return {**base, "kind": "throttled", "ok": False}
    if low_clock and low_power:
        return {**base, "kind": "latched", "ok": False}
    # จะบอกว่า "GPU ไม่ว่าง" ได้ ต้องมีหลักฐานว่าคล็อกหรือไฟขึ้นจริง · ไม่มีอะไรเลย (การ์ด/ไดรเวอร์
    # ที่ตอบ [N/A] ทั้งสองช่อง) แปลว่าตกเกณฑ์จริงแต่ **อธิบายไม่ได้** ซึ่งต้องพูดตรง ๆ ไม่ใช่เดาให้
    if (clock is not None and clock >= LATCH_CLOCK_MHZ) or (power is not None and power >= LOW_POWER_W):
        return {**base, "kind": "contended", "ok": False}
    return {**base, "kind": "slow", "ok": False}


# ── ข้อความ ──────────────────────────────────────────────────────────────────
# สองภาษาในตารางเดียว (รูปเดียวกับ `nodes/doctor._TEXT`) — CLI พูดไทย หน้าเว็บ/รายงานพูดอังกฤษ
_SUMMARY = {
    "ok": ("clocks are real: {tflops} TFLOPS · {clock} MHz · {power} W",
           "คล็อกจริง: {tflops} TFLOPS · {clock} MHz · {power} W"),
    "latched": ("GPU clock is latched low: {tflops} TFLOPS · {clock} MHz · {power} W "
                "(healthy GB10: 75-90 TFLOPS · 2.2-2.4 GHz · ≥80 W)",
                "GPU ถูกล็อกคล็อกไว้ต่ำ: {tflops} TFLOPS · {clock} MHz · {power} W "
                "(GB10 ปกติ: 75-90 TFLOPS · 2.2-2.4 GHz · ≥80 W)"),
    "throttled": ("slow but the driver says why: {reasons} — {tflops} TFLOPS · {clock} MHz · {temp}",
                  "ช้าแต่ไดรเวอร์บอกสาเหตุไว้: {reasons} — {tflops} TFLOPS · {clock} MHz · {temp}"),
    "contended": ("{tflops} TFLOPS but the clock is up ({clock} MHz, {power} W) — something else "
                  "is using this GPU, so this number judges the workload, not the machine",
                  "ได้ {tflops} TFLOPS แต่คล็อกขึ้นปกติ ({clock} MHz, {power} W) — มีอย่างอื่น"
                  "ใช้ GPU อยู่ ตัวเลขนี้จึงตัดสินงานที่รันอยู่ ไม่ได้ตัดสินเครื่อง"),
    "slow": ("{tflops} TFLOPS — below the 50 TFLOPS line, but nvidia-smi reported neither a clock "
             "nor a power draw, so this cannot say why",
             "ได้ {tflops} TFLOPS — ต่ำกว่าเส้น 50 TFLOPS แต่ nvidia-smi ไม่รายงานทั้งคล็อกและไฟ "
             "จึงบอกสาเหตุไม่ได้"),
    "unknown": ("cannot tell — {reason}", "ตรวจไม่ได้ — {reason}"),
    "not-applicable": ("not affected — {reason}", "ไม่เกี่ยว — {reason}"),
}

_REASON = {
    "no-nvidia-smi": ("no nvidia-smi on this machine", "เครื่องนี้ไม่มี nvidia-smi"),
    "no-gpu": ("nvidia-smi reports no GPU", "nvidia-smi ไม่เห็น GPU"),
    "not-gb10": ("{gpu} is not a GB10 — the EC clock latch is a DGX Spark problem",
                 "{gpu} ไม่ใช่ GB10 — clock latch ของ EC เป็นเรื่องของ DGX Spark"),
    "thresholds-are-gb10": ("measured {tflops} TFLOPS on {gpu}, but the pass/fail numbers in the "
                            "field notes are GB10-only — no verdict",
                            "วัดได้ {tflops} TFLOPS บน {gpu} แต่เกณฑ์ผ่าน/ตกในบันทึกภาคสนามเป็นของ "
                            "GB10 ล้วน — ไม่ตัดสิน"),
    "no-torch": ("no Python with torch and no local image that has one ({detail})",
                 "ไม่มี Python ที่มี torch และไม่มี image ในเครื่องที่มี ({detail})"),
    # เจอจริงบน dgx-spark02 (2026-09-21): torch ติดตั้งอยู่ แต่ `torch.cuda.is_available()`
    # เป็น False · ข้อความเดิมบอกว่า "ไม่มี torch" ซึ่งส่งคนไปติดตั้งของที่มีอยู่แล้ว
    "no-cuda": ("torch is installed but cannot see the GPU — the driver is fine "
                "(nvidia-smi answered), so it is this Python that cannot reach it ({detail})",
                "torch ติดตั้งอยู่แต่มองไม่เห็น GPU — ไดรเวอร์ปกติ (nvidia-smi ตอบ) "
                "ปัญหาอยู่ที่ Python ตัวนี้เข้าไม่ถึงการ์ด ({detail})"),
    "alloc": ("could not allocate {matrix}² fp16 on the GPU — stop the models first ({detail})",
              "จอง {matrix}² fp16 บน GPU ไม่ได้ — หยุดโมเดลก่อน ({detail})"),
    "burn-failed": ("the burn program failed: {detail}", "โปรแกรม burn ล้ม: {detail}"),
    "no-reading": ("the machine answered but without a number",
                   "เครื่องตอบกลับมาแต่ไม่มีตัวเลข"),
    "unreachable": ("the hub cannot reach this machine", "hub ต่อเครื่องนี้ไม่ได้"),
}


def _num(value, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "?"
    return f"{float(value):.{digits}f}{unit}" if digits else f"{int(value)}{unit}"


def summarize(result: dict, lang: str = "th") -> str:
    """หนึ่งบรรทัดที่บอกทั้งคำตัดสินและตัวเลขที่ทำให้ตัดสินแบบนั้น"""
    index = 1 if lang == "th" else 0
    kind = result.get("kind", "unknown")
    reason_key = result.get("reason") or ""
    reason = (_REASON.get(reason_key, ("{reason}", "{reason}"))[index]
              .format(gpu=result.get("gpu") or "?", detail=result.get("detail") or "?",
                      matrix=result.get("matrix") or DEFAULT_MATRIX,
                      tflops=_num(result.get("tflops")), reason=reason_key or "?"))
    text = _SUMMARY.get(kind, _SUMMARY["unknown"])[index]
    return text.format(
        tflops=_num(result.get("tflops")),
        clock=_num(result.get("clock_sm_mhz") if result.get("clock_sm_mhz") is not None
                   else result.get("clock_gr_mhz"), digits=0),
        power=_num(result.get("power_w")),
        temp=_num(result.get("temperature_c"), "°C", digits=0),
        reasons=", ".join(result.get("throttle") or []) or "?",
        reason=reason)


def remedy(result: dict, lang: str = "th") -> list[str]:
    """ขั้นตอนที่ **คนซึ่งยืนอยู่หน้าเครื่อง** ทำตามได้ — [] เมื่อไม่มีอะไรต้องทำ

    นี่คือเหตุผลทั้งหมดที่ฟีเจอร์นี้มีอยู่ · การตรวจเจอแล้วพิมพ์ว่า "FAIL" เฉย ๆ ไม่ช่วยใครเลย
    เพราะ **ซ่อมจากระยะไกลไม่ได้**: EC อยู่ใต้ OS และยังมีไฟเลี้ยงตราบที่ adapter เสียบอยู่
    `lmds restart` / `reboot` / `nvidia-smi -r` ไม่มีอันไหนแตะมันได้ — ข้อความจึงต้องเป็นลำดับ
    การกระทำทางกายภาพ ไม่ใช่คำสั่งให้พิมพ์
    """
    kind = result.get("kind")
    if kind == "latched":
        if lang == "th":
            return [
                "เครื่องนี้แก้จากระยะไกลไม่ได้ ต้องมีคนไปที่ตัวเครื่อง:",
                "  1. หยุดโมเดลบนเครื่องนี้ให้หมด (lmds stop <slug>) แล้วสั่ง sudo shutdown -h now",
                "  2. รอจนไฟหน้าเครื่องดับสนิท",
                "  3. ถอดปลั๊ก power adapter ออกจากตัวเครื่อง แล้วรอ 30-60 วินาที",
                "     (นานกว่านี้ได้ · สั้นกว่านี้ไม่ได้ — EC ต้องหมดไฟ standby ก่อน)",
                "  4. ระหว่างรอ ดูว่าเป็น adapter ตัวที่มากับเครื่อง และเสียบแน่นทั้งสองหัว",
                "     (EC ลดคล็อกเองเมื่อเห็นว่าไฟไม่พอ)",
                "  5. เสียบกลับ เปิดเครื่อง แล้วรัน lmds burn ซ้ำเพื่อยืนยันว่าคล็อกกลับมา",
                "รีบูตเฉย ๆ ไม่หาย — EC ยังกินไฟ standby ตราบที่ adapter ยังเสียบอยู่",
                "ระหว่างที่ยังไม่แก้: ตัวเลขจาก lmds bench ของเครื่องนี้เชื่อไม่ได้ และถ้าเครื่องนี้"
                "อยู่ในกลุ่ม stacked มันจะลากทั้งกลุ่มช้าตาม (ทุก collective รอ rank ที่ช้าที่สุด)",
            ]
        return [
            "This cannot be fixed from the hub — someone has to walk to the machine:",
            "  1. Stop every model on it (lmds stop <slug>), then sudo shutdown -h now",
            "  2. Wait until the front light is fully off",
            "  3. Unplug the power adapter from the machine and wait 30-60 seconds",
            "     (longer is fine, shorter is not — the EC has to lose standby power)",
            "  4. While waiting, check it is the adapter that shipped with the machine and that",
            "     both ends are seated (the EC lowers clocks by itself when power looks short)",
            "  5. Plug in, power on, and run lmds burn again to confirm the clocks came back",
            "A normal reboot does not clear it — the EC keeps standby power while the adapter is in.",
            "Until it is fixed: every lmds bench number from this machine is untrustworthy, and if "
            "it is in a stacked group it drags the whole group (every collective waits for the "
            "slowest rank).",
        ]
    if kind == "throttled":
        reasons = ", ".join(result.get("throttle") or [])
        if lang == "th":
            return [
                f"ไดรเวอร์รายงานสาเหตุไว้แล้ว ({reasons}) — **ไม่ใช่** clock latch ของ EC",
                "  (latch ของ EC ไม่รายงาน reason ใด ๆ เลย — §10) ถอดปลั๊กไม่ช่วยเคสนี้",
                "  thermal → ดูทางลมเข้า-ออก ฝุ่นที่ครีบระบายความร้อน และอุณหภูมิห้อง",
                "  power cap → ดูว่าเป็น adapter ตัวเดิมและสายไม่หลวม",
            ]
        return [
            f"The driver reported a reason ({reasons}) — this is NOT the EC clock latch",
            "  (the EC latch reports no reason at all). Unplugging will not help here.",
            "  thermal → check airflow, dust on the fins, and room temperature",
            "  power cap → check that it is the original adapter and the cable is seated",
        ]
    if kind == "slow":
        if lang == "th":
            return ["ไล่ตามลำดับนี้ — อย่าเพิ่งสั่งถอดปลั๊กจนกว่าจะตัดสองข้อแรกออก:",
                    "  1. หยุดโมเดล/งานที่ใช้ GPU อยู่ (lmds ps → lmds stop) แล้ววัดซ้ำ",
                    "  2. ดูว่า nvidia-smi บนเครื่องนั้นรายงาน clocks.sm/power.draw ได้ไหม "
                    "(ไดรเวอร์บางรุ่นตอบ [N/A])",
                    "  3. ยังตกอยู่และไม่มีอย่างอื่นใช้ GPU → ทำตามขั้นตอน latch: ปิดเครื่อง "
                    "ถอดปลั๊ก adapter 30-60 วินาที เสียบกลับ"]
        return ["Work through these in order — do not send anyone to unplug it until the first two "
                "are ruled out:",
                "  1. Stop whatever is using the GPU (lmds ps, lmds stop) and measure again",
                "  2. Check whether nvidia-smi on that machine reports clocks.sm/power.draw at all "
                "(some driver builds answer [N/A])",
                "  3. Still below the line with nothing else on the GPU → follow the latch steps: "
                "power off, unplug the adapter for 30-60 seconds, plug back in"]
    if kind == "contended":
        if lang == "th":
            return ["หยุดโมเดล/งานที่ใช้ GPU อยู่ก่อน (lmds ps แล้ว lmds stop) แล้ววัดซ้ำ — "
                    "ตัวเลขนี้บอกว่า GPU ไม่ว่าง ไม่ได้บอกว่าเครื่องเสีย"]
        return ["Stop what is using the GPU first (lmds ps, then lmds stop) and measure again — "
                "this number says the GPU is busy, not that the machine is faulty"]
    if kind == "unknown" and result.get("reason") == "no-cuda":
        # เรารู้ว่าไดรเวอร์ปกติเพราะ nvidia-smi ตอบชื่อการ์ดมาแล้ว — ตัดข้อสันนิษฐานนั้นออกได้เลย
        if lang == "th":
            return ["สาเหตุที่พบบ่อยที่สุดคือ torch เป็นรุ่น CPU-only (ล้อ ARM64 จาก PyPI มักไม่มี CUDA) "
                    "— เช็ก: python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'",
                    "ถ้าเป็น container: ต้องมี NVIDIA container toolkit และ --gpus all",
                    "ชี้ไปที่ interpreter หรือ image ที่มี torch แบบ CUDA ได้: "
                    "LMDS_BURN_PYTHON=/path/to/python หรือ LMDS_BURN_IMAGE=<image> lmds burn"]
        return ["Most often the torch build is CPU-only (ARM64 wheels from PyPI usually are) — "
                "check: python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'",
                "In a container this needs the NVIDIA container toolkit and --gpus all",
                "Or point at an interpreter or image whose torch has CUDA: "
                "LMDS_BURN_PYTHON=/path/to/python or LMDS_BURN_IMAGE=<image> lmds burn"]
    if kind == "unknown" and result.get("reason") == "no-torch":
        if lang == "th":
            return ["ชี้ interpreter ที่มี torch ให้เองได้: LMDS_BURN_PYTHON=/path/to/python lmds burn",
                    "หรือชี้ image ที่มีอยู่ในเครื่องแล้ว: LMDS_BURN_IMAGE=<image> lmds burn",
                    "(ไม่ pull ให้ — ลูกค้า air-gapped ต้องใช้ได้)"]
        return ["Point at an interpreter that has torch: LMDS_BURN_PYTHON=/path/to/python lmds burn",
                "or at an image already on the machine: LMDS_BURN_IMAGE=<image> lmds burn",
                "(nothing is pulled — air-gapped sites have to work)"]
    return []


# ── ทำจริง ───────────────────────────────────────────────────────────────────
def _local_runner(command: str, timeout: int = 120) -> Result:
    try:
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return Result(127, "", "ไม่พบ bash")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"หมดเวลา {timeout} วินาที")
    return Result(proc.returncode, proc.stdout or "", proc.stderr or "")


def _interpret(result, *, node: str = "") -> dict:
    """ผลดิบ (Result อะไรก็ได้ที่มี .ok/.stdout/.stderr) → คำตัดสินพร้อมข้อความ"""
    payload = parse(getattr(result, "stdout", "") or "")
    if not payload and not getattr(result, "ok", False):
        payload = {"error": "unreachable",
                   "detail": (getattr(result, "stderr", "") or "")[:200]}
    out = verdict(payload)
    out["node"] = node
    out["summary"] = summarize(out)
    out["summary_en"] = summarize(out, "en")
    out["remedy"] = remedy(out)
    return out


def check_local(*, runner=None, timeout: int = 120, force: bool = False,
                seconds: float = DEFAULT_SECONDS) -> dict:
    """burn เครื่องที่กำลังรัน lmds อยู่ตอนนี้ — `runner(command, timeout) -> Result`"""
    run = runner or _local_runner
    return _interpret(run(burn_script(seconds=seconds, force=force), timeout), node="")


def check_node(node, *, runner=None, timeout: int = 120, force: bool = False,
               seconds: float = DEFAULT_SECONDS) -> dict:
    """burn เครื่องปลายทางผ่าน SSH — `runner(node, command, timeout=…) -> Result`

    รูปของ `runner` เหมือน `lmds.nodes.run` เป๊ะ เพื่อให้เทสใช้ตัวปลอมตัวเดียวกับที่ netplan/
    hostname ใช้ได้ และเพื่อไม่ให้โมดูลนี้ import `nodes.ssh` ตอนที่ไม่จำเป็น
    """
    if runner is None:
        from lmds.nodes import run      # นำเข้าเมื่อใช้จริง — โมดูลนี้ไม่ผูกกับชั้น SSH

        runner = run
    script = burn_script(seconds=seconds, force=force)
    return _interpret(runner(node, script, timeout=timeout), node=getattr(node, "name", ""))


def check_fleet(nodes, *, runner=None, timeout: int = 120, force: bool = False,
                seconds: float = DEFAULT_SECONDS, workers: int = 8) -> list[dict]:
    """burn ทุกเครื่องพร้อมกัน — เรียงผลตามลำดับที่ส่งเข้ามา

    พร้อมกันไม่ใช่แค่เพื่อความเร็ว: §10 บอกว่าอาการนี้แสดงตัวตอนอยู่ใน TP ซึ่งทุกเครื่องมีโหลด
    พร้อมกัน · burn ทีละเครื่องจึงวัดในสภาพที่ไม่เหมือนตอนใช้งานจริง (ไฟ/ความร้อนในแร็คต่างกัน)
    """
    from concurrent.futures import ThreadPoolExecutor

    nodes = list(nodes)
    if not nodes:
        return []

    def one(node):
        return check_node(node, runner=runner, timeout=timeout, force=force, seconds=seconds)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(nodes)))) as pool:
        return list(pool.map(one, nodes))


def stamp(result: dict) -> dict:
    """ย่อผลให้เหลือเท่าที่ควรฝังไปกับผล bench — เก็บทั้ง verdict และตัวเลขที่ทำให้ตัดสินแบบนั้น

    ทำไมต้องฝัง: §10 บอกว่าข้อนี้ "ทำให้ตัวเลขทุกตัวข้างบนโกหกได้" · การเตือนตอนรันแล้วจบไป
    แก้ปัญหาได้แค่คนที่อยู่หน้าจอตอนนั้น — ส่วนคนที่เปิด `lmds bench show` อีกสามเดือนให้หลัง
    ยังเห็นตัวเลขเปล่า ๆ เหมือนเดิม · ติดป้ายไว้กับตัวผลเลยแล้วมันเดินทางไปพร้อมกัน
    """
    return {k: result.get(k) for k in
            ("kind", "ok", "applies", "tflops", "clock_sm_mhz", "power_w", "gpu", "reason")}
