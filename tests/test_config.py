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


def test_concurrent_saves_never_leave_a_broken_file(isolated_config):
    """เจอจริงบน hub: ลากจัดลำดับเครื่องหลายครั้งติดกัน หน้าเว็บรัน endpoint ใน threadpool
    สองเธรดจึงเขียน config.yaml พร้อมกัน แล้วได้ไฟล์ที่เป็น "เนื้อของครั้งใหม่ + หางของครั้งเก่า"
    YAML พังทั้งไฟล์ หน้าเว็บ 500 ทั้งหน้า

    หลักประกันจริงมาจาก os.replace ที่เป็น atomic (ดู write_atomic) — เทสนี้เป็นตัวกันพลาด
    ระดับพฤติกรรม: เขียนสลับยาว/สั้นพร้อมกันแล้วไฟล์ต้องอ่านได้เสมอ ไม่ใช่ตัวจับ race โดยตรง
    เพราะจังหวะที่ทำให้พังขึ้นกับ OS/ไฟล์ระบบ
    """
    from concurrent.futures import ThreadPoolExecutor

    def write(size: int) -> None:
        settings = Settings.load()
        settings.ui.node_order = [f"เครื่อง-{i}" for i in range(size)]
        settings.save()

    with ThreadPoolExecutor(max_workers=8) as pool:
        # ยาวสลับสั้นคือเคสที่พัง — สั้นเขียนทับยาวแล้วเหลือหางเดิมค้าง
        list(pool.map(write, [40, 1, 60, 2, 50, 3, 70, 1] * 4))

    reloaded = Settings.load()      # พังตรงนี้ = ไฟล์เสีย
    assert all(name.startswith("เครื่อง-") for name in reloaded.ui.node_order)


def test_a_broken_config_says_which_file_and_what_to_do(isolated_config):
    from lmds.config import SettingsError

    config_file().parent.mkdir(parents=True, exist_ok=True)
    # หน้าตาเดียวกับไฟล์ที่พังจริง: รายการถูกเขียนทับกลางคันจนเหลือบรรทัดที่ไม่มี "- "
    config_file().write_text("provider: null\nui:\n  node_order:\n  - a\n b\n", encoding="utf-8")
    with pytest.raises(SettingsError) as caught:
        Settings.load()
    assert str(config_file()) in str(caught.value)
    assert "ลบทิ้ง" in str(caught.value)
