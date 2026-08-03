"""เทส lmds doctor — ทุกเคสมาจาก failure ที่เจอจริงตอน hardware validation 3 ส.ค. 2569

ที่ต้องมี doctor เพราะวันนั้นทุกอาการต้องส่ง log ให้คนอ่านถึงจะรู้สาเหตุ
ถ้าลูกค้าหรือทีมงานเจอเอง เขาไม่มีทางรู้ว่าต้องแก้อะไร
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lmds.cli.main import app
from lmds.doctor import Status, diagnose
from lmds.fleet import ServerInfo

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_host_scan(monkeypatch):
    """อย่าไปแตะ docker/ss ของเครื่อง dev จริง — เทสคุมผลเอง"""
    monkeypatch.setattr("lmds.fleet.manager._pgrep_llama", lambda: [])
    monkeypatch.setattr("lmds.fleet.manager._orphan_docker", lambda known: [])
    monkeypatch.setattr("lmds.doctor.checks._run", lambda args, timeout=10: (0, ""))
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "")
    monkeypatch.setattr("lmds.doctor.checks.shutil.which", lambda name: "/usr/bin/" + name)


def _bundle(tmp_path: Path, slug: str, profile: dict) -> Path:
    directory = tmp_path / "bundles" / slug
    directory.mkdir(parents=True)
    controller = directory / f"{slug}-single.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (directory / "MODEL_PROFILE.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    return controller


def _register(tmp_path: Path, slug: str, controller: Path, port: int = 8000, mode: str = "docker") -> None:
    run_dir = tmp_path / "run" / slug
    run_dir.mkdir(parents=True)
    (run_dir / "server.meta").write_text(
        f"slug={slug}\nmodel={slug}\nmodel_id=org/{slug}\nengine=llamacpp\n"
        f"mode={mode}\nport={port}\ncontainer=lmds-{slug}\npid_file=\n"
        f"controller={controller}\nstarted_at=2026-08-03T10:00:00\n",
        encoding="utf-8",
    )


def _gguf_profile(model_dir: Path, gated: bool = False, projectors: list[str] | None = None) -> dict:
    return {
        "model": {"id": "unsloth/demo-GGUF", "revision": "sha", "selected_gguf": "demo-Q8.gguf",
                  "gated": gated},
        "runtime": {"engine": "llamacpp", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda"},
        "features": {"multimodal": {"projector_files": projectors or []}},
    }


def _setup(tmp_path, monkeypatch, *, gated=False, projectors=None, files=("demo-Q8.gguf",)):
    slug = "demo-gguf"
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "run"))
    model_dir = tmp_path / "models" / slug
    model_dir.mkdir(parents=True)
    for name in files:
        (model_dir / name).write_bytes(b"x" * 1024)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    controller = _bundle(tmp_path, slug, _gguf_profile(model_dir, gated, projectors))
    _register(tmp_path, slug, controller)
    monkeypatch.setattr("lmds.fleet.manager._container_running", lambda c: False)
    return slug


def test_gated_model_without_token_is_a_hard_fail(tmp_path, monkeypatch):
    """เคสจริง Llama-3.1-8B: ผู้ใช้พิมพ์ token ตอน deploy แล้วเข้าใจว่าใช้ได้ตลอด
    แต่ controller อ่านจาก env เท่านั้น → 401 ตอน download พร้อม traceback 60 บรรทัด
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    slug = _setup(tmp_path, monkeypatch, gated=True)

    result = diagnose(slug)
    token = next(f for f in result.findings if f.name == "hf-token")
    assert token.status is Status.FAIL
    assert "HF_TOKEN" in token.fix
    assert not result.healthy


def test_gated_model_with_token_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    slug = _setup(tmp_path, monkeypatch, gated=True)
    token = next(f for f in diagnose(slug).findings if f.name == "hf-token")
    assert token.status is Status.OK


def test_public_model_never_asks_for_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    slug = _setup(tmp_path, monkeypatch, gated=False)
    assert not [f for f in diagnose(slug).findings if f.name == "hf-token"]


def test_missing_mmproj_is_caught(tmp_path, monkeypatch):
    """เคสจริง gemma-4-12b-it: mmproj ไม่ถูกโหลด → multimodal กลายเป็น text-only เงียบ ๆ
    ไม่มี error ให้เห็นเลยสักจุด
    """
    slug = _setup(tmp_path, monkeypatch, projectors=["mmproj-BF16.gguf"], files=("demo-Q8.gguf",))
    weights = next(f for f in diagnose(slug).findings if f.name == "weights")
    assert weights.status is Status.FAIL
    assert "mmproj-BF16.gguf" in weights.detail
    assert "repair" in weights.fix


def test_complete_multimodal_download_passes(tmp_path, monkeypatch):
    slug = _setup(tmp_path, monkeypatch, projectors=["mmproj-BF16.gguf"],
                  files=("demo-Q8.gguf", "mmproj-BF16.gguf"))
    weights = next(f for f in diagnose(slug).findings if f.name == "weights")
    assert weights.status is Status.OK


def test_zero_byte_file_is_caught(tmp_path, monkeypatch):
    """download ที่ค้างกลางคัน (เช่นตอน Xet พัง) ทิ้งไฟล์เปล่าไว้"""
    slug = _setup(tmp_path, monkeypatch)
    (tmp_path / "models" / slug / "demo-Q8.gguf").write_bytes(b"")
    weights = next(f for f in diagnose(slug).findings if f.name == "weights")
    assert weights.status is Status.FAIL


def test_port_taken_by_someone_else(tmp_path, monkeypatch):
    slug = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("lmds.doctor.checks._listening_on",
                        lambda port: "LISTEN 0 4096 *:8000 users:((\"nginx\",pid=1,fd=6))")
    port = next(f for f in diagnose(slug).findings if f.name == "port")
    assert port.status is Status.FAIL
    assert "nginx" in port.detail


def test_unwritable_cache_is_reported_with_chown(tmp_path, monkeypatch):
    """เคสจริงจาก reference v8.2: container เคยสร้าง cache เป็น root แล้ว user เขียนไม่ได้"""
    slug = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("lmds.doctor.checks.os.access", lambda path, mode: False)
    perms = next(f for f in diagnose(slug).findings if f.name == "permissions")
    assert perms.status is Status.FAIL
    assert "chown" in perms.fix


def test_unknown_slug_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDS_RUN_ROOT", str(tmp_path / "empty"))
    result = diagnose("ไม่มีอยู่จริง")
    assert not result.healthy
    assert "lmds list" in result.findings[0].fix


def test_cli_exit_code_signals_blocking_problems(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    slug = _setup(tmp_path, monkeypatch, gated=True)
    result = runner.invoke(app, ["doctor", slug])
    assert result.exit_code == 2
    assert "HF_TOKEN" in result.output


def test_cli_exit_zero_when_healthy(tmp_path, monkeypatch):
    slug = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["doctor", slug])
    assert result.exit_code == 0, result.output


def test_port_conflict_names_the_other_lmds_model(tmp_path, monkeypatch):
    """เคสจริงบน RTX 5090: ทุก bundle ตั้งต้น port 8000 เหมือนกัน ตัวที่ยึดอยู่มักเป็น
    โมเดล LMDS อีกตัว — บอกชื่อไปเลยดีกว่าให้ผู้ใช้ไปไล่อ่าน output ของ ss เอง
    """
    slug = _setup(tmp_path, monkeypatch)
    rival = ServerInfo(slug="llama-3-1-8b-instruct", port=8000, running=True)
    monkeypatch.setattr("lmds.doctor.checks.discover", lambda: [rival])
    monkeypatch.setattr("lmds.doctor.checks._listening_on", lambda port: "LISTEN 0 2048 0.0.0.0:8000")

    port = next(f for f in diagnose(slug).findings if f.name == "port")
    assert port.status is Status.FAIL
    assert "llama-3-1-8b-instruct" in port.detail
    assert "lmds stop llama-3-1-8b-instruct" in port.fix
