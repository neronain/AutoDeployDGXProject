"""parse_source: ลิงก์ที่คนวางมาจริง ต้องได้ repo/revision/ไฟล์ที่ถูก — หรือ SourceError ที่อ่านรู้เรื่อง ไม่ใช่ traceback

เคสจริง 2026-10-06 (audit · hub 6e2b474):
- `lmds inspect https://huggingface.co/Qwen/Qwen3-8B/tree/refs/pr/1` → "ไม่พบ model repo" (เอา revision segment เดียว = "refs")
  · `…/resolve/feature/foo/x.gguf` → rev=`feature` file=`foo/x.gguf` โดยไม่บอกว่าเดา
- `lmds inspect 'https://[::1/unsloth/x'` → traceback `ValueError: Invalid IPv6 URL`
- `…/blob/main/README.md` → filename='README.md' (ไฟล์ที่ไม่ใช่ weight กลายเป็น "ไฟล์ของโมเดล")
- `hf.co/org/repo:Q4_K_M` → repo_id='org/repo:Q4_K_M' (path ของลิงก์ข้ามการตรวจรูปแบบ repo id) · `repo.git` ถูกเก็บทั้งชื่อ
- `huggingface.co/settings/tokens` · `/docs/…` · `/blog/…` ถูกรับเป็น repo · `%20` ในชื่อไฟล์ไม่ถูก decode
"""

import httpx
import pytest
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.inspector import HfClient
from lmds.resolver import SourceError, parse_source
from tests.real_hub import flat

runner = CliRunner()
REPO = "unsloth/Qwen3-8B-GGUF"
BASE = f"https://huggingface.co/{REPO}"


@pytest.mark.parametrize("link, revision, filename", [
    (f"{BASE}/tree/refs/pr/3", "refs/pr/3", None),
    (f"{BASE}/tree/refs%2Fpr%2F3", "refs/pr/3", None),
    (f"{BASE}/blob/refs/pr/3/Qwen3-8B-Q4_K_M.gguf", "refs/pr/3", "Qwen3-8B-Q4_K_M.gguf"),
    (f"{BASE}/tree/refs/convert/parquet", "refs/convert/parquet", None),
    (f"{BASE}/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true", None, "Qwen3-8B-Q4_K_M.gguf"),
    (f"{BASE}/resolve/main/sub%20dir/a%20b.gguf", None, "sub dir/a b.gguf"),
    (f"{BASE}/blob/v1.0/x.gguf", "v1.0", "x.gguf"),
    (f"{BASE}.git", None, None),
    (f"{REPO}.git", None, None),
])
def test_links_people_paste_resolve_to_the_right_repo_revision_and_file(link, revision, filename):
    source = parse_source(link)
    assert (source.repo_id, source.revision, source.filename) == (REPO, revision, filename)
    assert source.notes == (), "ลิงก์ที่ไม่กำกวมไม่ต้องมีหมายเหตุ"


@pytest.mark.parametrize("link, revision, filename, word", [
    # branch "feature" + path "foo" หรือ branch "feature/foo"? — ลิงก์บอกไม่ได้ ต้องพูดว่าเลือกอะไร
    (f"{BASE}/tree/feature/foo", "feature", None, "feature/foo"),
    (f"{BASE}/resolve/feature/foo/x.gguf", "feature", "foo/x.gguf", "feature/foo"),
])
def test_an_ambiguous_branch_with_a_slash_says_what_was_assumed(link, revision, filename, word):
    source = parse_source(link)
    assert (source.revision, source.filename) == (revision, filename)
    assert len(source.notes) == 1 and word in source.notes[0] and "--revision" in source.notes[0]


@pytest.mark.parametrize("name", ["README.md", "config.json", "model.safetensors", "tokenizer.json"])
def test_a_link_to_a_non_gguf_file_means_the_repo_not_that_file(name):
    source = parse_source(f"{BASE}/blob/main/{name}")
    assert source.repo_id == REPO and source.filename is None
    assert any(name in note for note in source.notes), "ต้องบอกว่าไม่ได้ใช้ไฟล์นั้น"


@pytest.mark.parametrize("text, word", [
    ("https://[::1/unsloth/x", "ลิงก์ผิดรูปแบบ"),
    ("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M", "--gguf Q4_K_M"),
    ("unsloth/Qwen3-8B-GGUF:Q4_K_M", "--gguf Q4_K_M"),
    ("unsloth/Qwen3-8B-GGUF@main", "--revision main"),
    ("https://huggingface.co/settings/tokens", "/settings"),
    ("https://huggingface.co/docs/transformers", "/docs"),
    ("https://huggingface.co/blog/some-post", "/blog"),
    ("https://huggingface.co/models/unsloth/Qwen3-8B-GGUF", "/models"),
    ("ftp://huggingface.co/unsloth/Qwen3-8B-GGUF", "ftp"),
    ("https://huggingface.co/unsloth/..%2F..%2Fetc", "ไม่เข้าใจ"),
    ("https://huggingface.co.evil.com/unsloth/x", "ไม่รู้จักโดเมน"),
])
def test_malformed_or_non_model_input_is_a_source_error_with_a_reason(text, word):
    with pytest.raises(SourceError) as err:
        parse_source(text)
    assert word in str(err.value)


def test_sources_compare_equal_whatever_was_noted_while_parsing():
    assert parse_source(f"{BASE}/blob/main/README.md") == parse_source(REPO)


# ── CLI: ไม่มี traceback · หมายเหตุถูกแสดง · PR ไปถึง Hub ด้วย revision ที่ถูก ───────────────────────


def test_cli_turns_a_malformed_url_into_a_message(isolated_config):
    result = runner.invoke(app, ["inspect", "https://[::1/unsloth/x"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), "เดิม ValueError หลุดเป็น traceback"
    assert "ลิงก์ผิดรูปแบบ" in flat(result.output)


def test_cli_asks_the_hub_for_the_pr_revision_and_shows_what_it_assumed(isolated_config, monkeypatch):
    asked: list[str] = []

    def hub(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.raw_path.decode().split("?")[0])
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json={"id": "Qwen/Qwen3-8B", "sha": "c" * 40, "tags": [], "siblings": [
                {"rfilename": "m-Q4_K_M.gguf", "size": 10}, {"rfilename": "m-Q8_0.gguf", "size": 20}]})
        return httpx.Response(404, headers={"x-error-code": "EntryNotFound"})

    monkeypatch.setattr("lmds.inspector.HfClient", lambda token=None: HfClient(
        token=token, client=httpx.Client(transport=httpx.MockTransport(hub))))

    result = runner.invoke(app, ["inspect", "https://huggingface.co/Qwen/Qwen3-8B/tree/refs/pr/1", "--json"])
    assert result.exit_code == 0, result.output
    assert asked[0] == "/api/models/Qwen/Qwen3-8B/revision/refs%2Fpr%2F1", "เดิมถาม revision 'refs' แล้วได้ ไม่พบ model repo"

    result = runner.invoke(app, ["inspect", "https://huggingface.co/Qwen/Qwen3-8B/tree/feature/foo", "--json"])
    assert "feature/foo" in flat(result.output) and "--revision" in flat(result.output)
