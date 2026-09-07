import stat

from lmds.config.paths import credentials_file
from lmds.secrets import (
    MASK,
    check_credentials_permissions,
    delete_secret,
    get_secret,
    mask_preview,
    redact,
    secret_source,
    set_secret,
)


def test_set_and_get_secret_file_backend(isolated_config):
    backend = set_secret("openai", "sk-test1234567890abcdef")
    assert backend == "file"
    assert get_secret("openai") == "sk-test1234567890abcdef"
    assert secret_source("openai") == "file"


def test_credentials_file_permissions_0600(isolated_config):
    set_secret("hf", "hf_abcdefghijklmnop1234")
    mode = stat.S_IMODE(credentials_file().stat().st_mode)
    assert mode == 0o600
    assert check_credentials_permissions() is None


def test_permission_warning_when_loose(isolated_config):
    set_secret("hf", "hf_abcdefghijklmnop1234")
    credentials_file().chmod(0o644)
    warning = check_credentials_permissions()
    assert warning is not None and "0600" in warning


def test_env_overrides_file(isolated_config, monkeypatch):
    set_secret("openai", "sk-fromfile1234567890")
    monkeypatch.setenv("LMDS_OPENAI_API_KEY", "sk-fromenv1234567890xx")
    assert get_secret("openai") == "sk-fromenv1234567890xx"
    assert secret_source("openai") == "env"


def test_delete_secret(isolated_config):
    set_secret("gemini", "AIzaSyTest1234567890abcdefghijklmnop")
    delete_secret("gemini")
    assert get_secret("gemini") is None


def test_get_missing_secret_returns_none(isolated_config):
    assert get_secret("anthropic") is None
    assert secret_source("anthropic") is None


def test_redact_known_secret():
    text = "error calling api with key sk-verysecretkey123456 failed"
    assert "sk-verysecretkey123456" not in redact(text, ["sk-verysecretkey123456"])


def test_redact_patterns_without_known_value():
    text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456 and hf_ABCDEFGHIJKLMNOPqrst"
    result = redact(text)
    assert "hf_ABCDEFGHIJKLMNOPqrst" not in result
    assert "abcdefghijklmnopqrstuvwxyz123456" not in result
    assert MASK in result


def test_redact_short_secret_not_replacing_everything():
    # secret สั้นเกิน (<8 ตัว) ไม่ replace เพื่อกัน false positive ทำลายข้อความ
    assert redact("port 8000", ["8000"]) == "port 8000"


def test_mask_preview():
    assert mask_preview(None) == "(ไม่ได้ตั้งค่า)"
    assert mask_preview("short") == "****"
    preview = mask_preview("sk-test1234567890abcdef")
    assert preview.startswith("sk-") and preview.endswith("cdef") and "…" in preview
    assert "1234567890" not in preview


def test_keyring_read_cannot_hang_get_secret(monkeypatch, tmp_path):
    """เคสจริง 2026-09-07 spark-head: keyring (D-Bus Secret Service) ค้าง → `lmds config show` ใน install.sh ค้าง 10 นาที
    → Update ทั้งฟลีตหยุด · อ่าน keyring ต้องมีเวลาจำกัด และ LMDS_NO_KEYRING=1 ต้องไม่แตะ keyring เลย"""
    import sys
    import threading
    import time
    import types

    from lmds.secrets import store

    calls: list[str] = []

    class HangingKeyring:
        @staticmethod
        def get_password(service, name):
            calls.append(name)
            time.sleep(5)
            return "late-secret"

    monkeypatch.setitem(sys.modules, "keyring", HangingKeyring)
    monkeypatch.setattr(store, "_read_credentials_file", lambda: {"hf": "file-token"})
    monkeypatch.setenv("LMDS_KEYRING_TIMEOUT", "0.2")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LMDS_NO_KEYRING", raising=False)
    t0 = time.monotonic()
    assert store.get_secret("hf") == "file-token"          # keyring ช้าเกิน → ข้ามไปแหล่งถัดไป
    assert store.secret_source("hf") == "file"
    assert time.monotonic() - t0 < 2, "ต้องไม่รอ keyring จนครบ 5 วิ"
    assert calls, "ต้องได้ลอง keyring ก่อน"

    calls.clear()
    monkeypatch.setenv("LMDS_NO_KEYRING", "1")
    assert store.get_secret("hf") == "file-token" and not calls, "LMDS_NO_KEYRING=1 ต้องไม่เรียก keyring"
