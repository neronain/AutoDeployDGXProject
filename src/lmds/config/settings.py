"""config.yaml ของผู้ใช้: provider ที่เลือก, ค่า default ต่าง ๆ — ไม่มี secret ในไฟล์นี้เด็ดขาด"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .paths import config_file, ensure_config_dir, write_atomic


class SettingsError(Exception):
    """config.yaml อ่านไม่ได้ — ข้อความต้องบอกไฟล์และวิธีแก้ ไม่ใช่แค่ว่า parse ไม่ผ่าน"""


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


class Cluster(BaseModel):
    """ค่าเกี่ยวกับ stacked ของ "เครื่องนี้" เอง

    node อื่นเก็บค่าแบบเดียวกันไว้ในทะเบียน (`Node.stack`) แต่ hub ไม่ได้อยู่ในทะเบียน
    จึงต้องมีที่เก็บของตัวเอง — hub มักเป็นเครื่องที่มีงานของมันอยู่แล้ว ไม่ได้ตั้งใจเอาไป stacked
    """

    stack_self: bool = True


class Ui(BaseModel):
    """ลำดับการ์ดเครื่องที่ผู้ใช้ลากจัดเอง

    เก็บที่ hub ไม่ใช่ในเบราว์เซอร์ — เปิดจากเครื่องไหน/บราว์เซอร์ไหนก็เห็นลำดับเดียวกัน
    และ CLI เรียงตามลำดับเดียวกันด้วย · ชื่อที่ไม่มีในทะเบียนแล้วถูกข้าม เครื่องใหม่ต่อท้าย
    """

    node_order: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    provider: Optional[ProviderConfig] = None
    defaults: Defaults = Field(default_factory=Defaults)
    cluster: Cluster = Field(default_factory=Cluster)
    ui: Ui = Field(default_factory=Ui)

    @classmethod
    def load(cls) -> "Settings":
        path = config_file()
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            # ไม่เดาว่าผู้ใช้ตั้งใจอะไรและไม่ลบไฟล์ให้เอง (มี provider/คีย์ตั้งค่าอยู่) —
            # แต่ต้องบอกให้ชัดว่าไฟล์ไหนและทำอะไรต่อ ไม่ใช่โยน stack trace ให้เดา
            raise SettingsError(
                f"อ่าน {path} ไม่ได้ — ไฟล์เสีย: {exc}\n"
                f"แก้ไฟล์นี้ให้ถูกต้อง หรือลบทิ้งเพื่อเริ่มจากค่าเริ่มต้น (จะเสียค่า provider ที่ตั้งไว้)"
            ) from exc
        return cls.model_validate(data)

    def save(self) -> None:
        ensure_config_dir()
        write_atomic(
            config_file(),
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        )

    def set_provider(self, name: ProviderName, model: str = "", base_url: str | None = None) -> ProviderConfig:
        # ปฏิเสธตั้งแต่ตอนตั้งค่า ดีกว่าปล่อยให้ผ่านแล้วไปพังตอน deploy จริง
        if name is ProviderName.ANTHROPIC:
            raise ValueError(
                "Anthropic adapter อยู่ใน roadmap เฟส 2 — ยังใช้เป็นสมองไม่ได้ · "
                "ใช้ openai / gemini / minimax / openai-compat แทน "
                "(Claude ผ่าน gateway ที่เป็น OpenAI-compatible ก็ใช้ openai-compat ได้)"
            )
        resolved_model = model or DEFAULT_MODELS[name]
        if name is ProviderName.OPENAI_COMPAT and not base_url:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --base-url")
        if name is ProviderName.OPENAI_COMPAT and not resolved_model:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --model")
        self.provider = ProviderConfig(name=name, model=resolved_model, base_url=base_url)
        return self.provider
