"""Audit รอบ 3 (2026-10-06) — download ของ single vLLM/SGLang ต้องโหลดเฉพาะที่ bundle ใช้ ไม่ใช่ทั้ง repo

เดิม `snapshot_download(repo_id, revision)` ไม่มีตัวกรอง: checkpoint ซ้ำใน `original/` (openai/gpt-oss-* = 2 เท่า) ·
`consolidated*.safetensors` (mistralai) · GGUF ทุก quant ของ repo สองรูปแบบ (ตัวจริงตัวหนึ่ง 1.6 TB) · ONNX — ทั้งที่แผน/fit/
เช็คดิสก์นับแค่ shard ในแผน

เทสรัน `download` ของ controller ทั้งเส้น: docker ปลอม **รัน python ที่ controller ฝังมาจริง** กับ `huggingface_hub` ปลอมที่
(1) ตอบรายชื่อไฟล์ของ repo จาก fixture (2) กรองด้วยกติกาเดียวกับของจริง (allow ∧ ¬ignore · fnmatch) แล้ว "โหลด" ลงแคช —
สิ่งที่ยืนยันจึงเป็นไฟล์ที่ลงดิสก์จริงและ `verify-files` ที่รันต่อ ไม่ใช่ข้อความในสคริปต์
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap

import pytest

from tests.single_controller_harness import SHARDS, box, render, st_report, write_exe  # noqa: F401 — fixture

GB = 1_000_000_000
# repo ที่มีทุกอย่างปนกัน — (ชื่อ, ขนาด) · None = Hub ไม่รายงานขนาด
LISTING = [
    ("config.json", 2), ("model.safetensors.index.json", 2), ("generation_config.json", 20),
    ("tokenizer.json", 30), ("tokenizer_config.json", 30), ("chat_template.jinja", 40), ("README.md", 50),
    ("modeling_custom.py", 60),                                   # remote code
    ("processor/preprocessor_config.json", 25),                   # config ในโฟลเดอร์ย่อย
    ("tokenizer.bin", 4096),                                      # นามสกุลเหมือน weight แต่เป็นไฟล์เล็กของ tokenizer
    (SHARDS[0][0], SHARDS[0][1]), (SHARDS[1][0], SHARDS[1][1]),   # weight ในแผน
    ("original/model-00001-of-00002.safetensors", 30 * GB),       # checkpoint ซ้ำ
    ("original/model-00002-of-00002.safetensors", 30 * GB),
    ("original/config.json", 15),
    ("consolidated.safetensors", 60 * GB),
    ("model-Q4_K_M.gguf", 40 * GB), ("model-Q8_0.gguf", 70 * GB),
    ("onnx/model.onnx", 3 * GB), ("onnx/model.onnx_data", 50 * GB),
    ("pytorch_model.bin", 60 * GB),
    ("flax_model.msgpack", None),                                 # ไม่รู้ขนาด + รูปแบบ weight + ไม่อยู่ในแผน
]
UNWANTED = {"original/model-00001-of-00002.safetensors", "original/model-00002-of-00002.safetensors",
            "consolidated.safetensors", "model-Q4_K_M.gguf", "model-Q8_0.gguf", "onnx/model.onnx",
            "onnx/model.onnx_data", "pytorch_model.bin", "flax_model.msgpack"}
WANTED = {name for name, _ in LISTING} - UNWANTED

_FAKE_HUB = '''
"""huggingface_hub ปลอม — รายชื่อไฟล์จาก fixture · กรองแบบเดียวกับ huggingface_hub.utils.filter_repo_objects"""
import fnmatch, json, os

_LISTING = json.load(open(os.environ["FAKE_HUB_LISTING"], encoding="utf-8"))


def _note(record):
    with open(os.environ["FAKE_HUB_CALLS"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\\n")


class _File:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


class _Info:
    def __init__(self):
        self.siblings = [_File(name, size) for name, size in _LISTING]


class HfApi:
    def model_info(self, repo_id, revision=None, files_metadata=False, **_):
        _note({"call": "model_info", "repo": repo_id, "revision": revision, "files_metadata": files_metadata})
        return _Info()


def _host(path):
    for pair in filter(None, os.environ.get("FAKE_MOUNTS", "").split(";")):
        src, dst = pair.split("=", 1)
        if path == dst or path.startswith(dst.rstrip("/") + "/"):
            return src + path[len(dst):]
    return path


def _as_list(patterns):
    return [patterns] if isinstance(patterns, str) else list(patterns or [])


def snapshot_download(repo_id, revision=None, cache_dir=None, allow_patterns=None, ignore_patterns=None, **_):
    _note({"call": "snapshot_download", "repo": repo_id, "revision": revision,
           "allow_patterns": allow_patterns, "ignore_patterns": ignore_patterns})
    root = os.path.join(_host(os.environ["HF_HUB_CACHE"]), "models--" + repo_id.replace("/", "--"), "snapshots", revision)
    for name, size in _LISTING:
        if allow_patterns is not None and not any(fnmatch.fnmatch(name, p) for p in _as_list(allow_patterns)):
            continue
        if ignore_patterns is not None and any(fnmatch.fnmatch(name, p) for p in _as_list(ignore_patterns)):
            continue
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x" * (size if size is not None and size <= 8192 else 1))
    return root
'''


def _prepare(box, listing=LISTING):
    """ล้างแคชที่ harness วางไว้ให้ · ติดตั้ง huggingface_hub ปลอม + python3 ที่มองเห็นมัน (แทน python ในคอนเทนเนอร์)"""
    cache = box.home / ".cache" / "huggingface"
    shutil.rmtree(cache, ignore_errors=True)
    fake_site = box.tmp / "fakehub"
    (fake_site / "huggingface_hub").mkdir(parents=True)
    (fake_site / "huggingface_hub" / "__init__.py").write_text(textwrap.dedent(_FAKE_HUB), encoding="utf-8")
    (box.tmp / "listing.json").write_text(json.dumps(listing), encoding="utf-8")
    write_exe(box.bin / "python3", f'PYTHONPATH="{fake_site}" exec "{sys.executable}" "$@"\n')
    write_exe(box.bin / "id", 'case "${1:-}" in -u|-g) echo 1000 ;; *) echo lmds ;; esac\n')
    return {"FAKE_HUB_LISTING": str(box.tmp / "listing.json"), "FAKE_HUB_CALLS": str(box.tmp / "hub-calls.jsonl")}


def _hub_calls(box) -> list[dict]:
    path = box.tmp / "hub-calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _downloaded(box) -> set[str]:
    snap = box.home / ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/sha-pinned-123"
    return {str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_download_fetches_the_planned_weights_and_every_non_weight_file_and_nothing_else(box, kind):
    env = _prepare(box)
    done = box.run("download", env=env)
    assert done.returncode == 0, done.stdout + done.stderr + (box.state / "").as_posix()

    got = _downloaded(box)
    assert got & UNWANTED == set(), f"โหลดไฟล์ weight ที่ไม่อยู่ในแผน: {sorted(got & UNWANTED)}"
    assert got == WANTED, f"ขาด {sorted(WANTED - got)} · เกิน {sorted(got - WANTED)}"
    # ชิ้นที่พลาดง่าย: config ในโฟลเดอร์ย่อย · remote code · ไฟล์เล็กนามสกุล .bin · config ข้าง checkpoint ซ้ำ
    assert {"processor/preprocessor_config.json", "modeling_custom.py", "tokenizer.bin", "original/config.json"} <= got

    call = next(c for c in _hub_calls(box) if c["call"] == "snapshot_download")
    assert call["repo"] == "Qwen/Qwen3-32B" and call["revision"] == "sha-pinned-123"
    assert call["allow_patterns"] is not None and call["ignore_patterns"] is None
    assert "skipping 9 weight-format files" in done.stdout, done.stdout

    # verify-files ต้องเห็นตรงกับที่ download โหลด — รายการเดียวกัน
    verified = box.run("verify-files", env=env)
    assert verified.returncode == 0 and "shard ครบ 2 ไฟล์" in verified.stdout, verified.stdout + verified.stderr


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_the_file_lists_reach_the_container_through_the_environment_not_argv(box, kind):
    env = _prepare(box)
    assert box.run("download", env=env).returncode == 0
    run_line = next(line for line in box.calls().splitlines() if line.startswith("docker run -d") and "python3" in line)
    assert " -e LMDS_WEIGHT_FILES " in run_line and " -e LMDS_REQUIRED_FILES " in run_line, run_line[:400]
    assert SHARDS[0][0] not in run_line


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_a_file_name_with_glob_characters_is_matched_literally(box, kind):
    """allow_patterns เป็น fnmatch — ชื่ออย่าง `notes[v2].md` ต้องไม่กลายเป็น pattern ที่ไม่ match ตัวเอง"""
    env = _prepare(box, [*LISTING, ("notes[v2].md", 10), ("what?.txt", 10)])
    assert box.run("download", env=env).returncode == 0
    assert {"notes[v2].md", "what?.txt"} <= _downloaded(box)


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_a_plan_without_a_shard_list_still_downloads_the_repository_and_says_so(tmp_path, kind):
    """Hub ไม่รายงานรายชื่อ shard → controller ไม่มี SHARD_FILES ให้กรอง — โหลดทั้ง repo ตามเดิม (กรองโดยไม่รู้ว่า weight ตัวไหน
    คือของเรา = เสี่ยงไม่ได้ weight เลย) และต้องบอกไว้"""
    from tests.single_controller_harness import Box

    made = Box(tmp_path, kind, render(tmp_path, kind, report=st_report(safetensor_shards=[])))
    try:
        env = _prepare(made)
        done = made.run("download", env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        call = next(c for c in _hub_calls(made) if c["call"] == "snapshot_download")
        assert call["allow_patterns"] is None
        assert "downloading the whole repository" in done.stdout
        assert _downloaded(made) == {name for name, _ in LISTING}
    finally:
        made.close()
