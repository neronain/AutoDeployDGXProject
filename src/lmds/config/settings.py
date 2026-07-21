"""config.yaml ของผู้ใช้: provider ที่เลือก, ค่า default ต่าง ๆ — ไม่มี secret ในไฟล์นี้เด็ดขาด"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .paths import config_file, ensure_config_dir


class ProviderName(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    MINIMAX = "minimax"
    ANTHROPIC = "anthropic"
    OPENAI_COMPAT = "openai-compat"


DEFAULT_MODELS: dict[ProviderName, str] = {
    ProviderName.OPENAI: "gpt-4.1",
    ProviderName.GEMINI: "gemini-2.5-pro",
    ProviderName.MINIMAX: "MiniMax-M2",
    ProviderName.ANTHROPIC: "claude-sonnet-5",
    ProviderName.OPENAI_COMPAT: "",  # ผู้ใช้ต้องระบุเองคู่กับ base_url
}


class ProviderConfig(BaseModel):
    name: ProviderName
    model: str = ""
    base_url: Optional[str] = None  # จำเป็นเฉพาะ openai-compat


class Defaults(BaseModel):
    target: str = "auto"
    language: str = "th"
    output_dir: str = "./bundles"


class Settings(BaseModel):
    provider: Optional[ProviderConfig] = None
    defaults: Defaults = Field(default_factory=Defaults)

    @classmethod
    def load(cls) -> "Settings":
        path = config_file()
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def save(self) -> None:
        ensure_config_dir()
        path = config_file()
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def set_provider(self, name: ProviderName, model: str = "", base_url: str | None = None) -> ProviderConfig:
        resolved_model = model or DEFAULT_MODELS[name]
        if name is ProviderName.OPENAI_COMPAT and not base_url:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --base-url")
        if name is ProviderName.OPENAI_COMPAT and not resolved_model:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --model")
        self.provider = ProviderConfig(name=name, model=resolved_model, base_url=base_url)
        return self.provider
