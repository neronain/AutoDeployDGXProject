import pytest
import yaml

from lmds.config import ProviderName, Settings
from lmds.config.paths import config_file


def test_load_empty_settings(isolated_config):
    settings = Settings.load()
    assert settings.provider is None
    assert settings.defaults.language == "th"


def test_set_provider_default_model(isolated_config):
    settings = Settings.load()
    provider = settings.set_provider(ProviderName.OPENAI)
    assert provider.model == "gpt-4.1"
    settings.save()

    reloaded = Settings.load()
    assert reloaded.provider is not None
    assert reloaded.provider.name is ProviderName.OPENAI


def test_anthropic_rejected_at_config_time(isolated_config):
    """เดิมตั้งค่าผ่าน แล้วไปพังตอน deploy — ต้องบอกตั้งแต่ตอนตั้งค่า"""
    settings = Settings.load()
    with pytest.raises(ValueError, match="เฟส 2"):
        settings.set_provider(ProviderName.ANTHROPIC)
    assert settings.provider is None  # ไม่เขียนทับ config เดิม


def test_openai_compat_requires_base_url(isolated_config):
    settings = Settings.load()
    with pytest.raises(ValueError):
        settings.set_provider(ProviderName.OPENAI_COMPAT, model="qwen3-coder")


def test_openai_compat_with_base_url(isolated_config):
    settings = Settings.load()
    provider = settings.set_provider(
        ProviderName.OPENAI_COMPAT, model="qwen3-coder", base_url="http://10.100.152.1:8000/v1"
    )
    assert provider.base_url == "http://10.100.152.1:8000/v1"


def test_config_file_never_contains_secrets(isolated_config):
    """config.yaml ต้องไม่มีช่องทางเก็บ secret เลย"""
    settings = Settings.load()
    settings.set_provider(ProviderName.GEMINI)
    settings.save()
    raw = config_file().read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    flat = yaml.safe_dump(data).lower()
    for word in ["key", "token", "secret", "password"]:
        assert word not in flat, f"config.yaml ไม่ควรมี field เกี่ยวกับ {word}"


def test_config_file_permissions(isolated_config):
    import stat

    settings = Settings.load()
    settings.set_provider(ProviderName.OPENAI)
    settings.save()
    assert stat.S_IMODE(config_file().stat().st_mode) == 0o600
