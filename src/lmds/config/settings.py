"""config.yaml ของผู้ใช้: provider ที่เลือก, ค่า default ต่าง ๆ — ไม่มี secret ในไฟล์นี้เด็ดขาด"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

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
    """ฟิลด์ของ "เครื่องนี้" เองที่เครื่องอื่นเก็บไว้ในทะเบียน

    node อื่นเก็บค่าแบบเดียวกันไว้ในทะเบียน (`Node.stack`, `Node.site`) แต่ hub ไม่ได้อยู่ใน
    ทะเบียน จึงต้องมีที่เก็บของตัวเอง — hub มักเป็นเครื่องที่มีงานของมันอยู่แล้ว ไม่ได้ตั้งใจเอาไป stacked
    (ข้อจำกัดเดียวกับที่ docs/FLEET-MULTI-NODE.md เขียนไว้เรื่อง cluster IP ของ hub เอง)
    """

    stack_self: bool = True
    # ไซต์ที่ hub ตัวเองตั้งอยู่ — ว่าง = ยังไม่จัดไซต์
    #
    # `site` เป็นฟิลด์ของ node ในทะเบียน (`lmds node set <ชื่อ> --site`) แต่ hub ไม่มีแถวใน
    # ทะเบียนให้เก็บ · ผลคือ siteOfNode() ของหน้าเว็บคืน "" ให้ hub เสมอ แล้ว hub ไปกอง
    # "Unassigned site" ตลอดกาล — ผู้ใช้ที่จัด node ไว้ไซต์ isit ไม่มีทางดึง hub เข้าไปอยู่ด้วยได้
    # และสองเครื่องนั้นไม่มีวันขึ้นกลุ่มเดียวกัน ทั้งบนรางซ้ายและตอนจับกลุ่ม stacked
    # (cluster.py จัดกลุ่มด้วย key ที่มี site อยู่ด้วย — hub ที่ site="" ตกกลุ่มเสมอ)
    site: str = ""


class Ui(BaseModel):
    """ลำดับการ์ดเครื่องที่ผู้ใช้ลากจัดเอง

    เก็บที่ hub ไม่ใช่ในเบราว์เซอร์ — เปิดจากเครื่องไหน/บราว์เซอร์ไหนก็เห็นลำดับเดียวกัน
    และ CLI เรียงตามลำดับเดียวกันด้วย · ชื่อที่ไม่มีในทะเบียนแล้วถูกข้าม เครื่องใหม่ต่อท้าย
    """

    node_order: list[str] = Field(default_factory=list)


class Recipes(BaseModel):
    """ปลายทางที่ `lmds recipes --publish` ส่ง controller ที่รันผ่านแล้วขึ้นไป

    ว่าง = local store ในเครื่อง hub เอง (`~/.config/lmds/controllers/published-local`)
    ซึ่งเป็นค่าที่ปลอดภัยสำหรับลูกค้า: fleet ของเขาแชร์กันได้โดยไม่แตะรีโปของเรา
    · ทีมเราตั้งเป็นรีโป candidates (เช่น git@github.com:neronain/script-update.git)
    เพื่อ push ขึ้นไปแล้ว review ก่อนเลื่อนเข้า canonical
    """

    publish_repo: str = ""
    publish_ref: str = "main"
    # ต้นทางที่ `lmds recipes --sync` และปุ่ม "Sync from GitHub" ดึงสูตรมา — ว่าง = รีโปของทีม
    #
    # อยู่ใน config ไม่ใช่ใน request: หน้าเว็บเคยรับ repo/ref จาก body ของ POST /api/recipes/sync
    # ซึ่งเท่ากับให้ใครก็ตามที่ยิง endpoint ได้เลือกว่า hub จะ clone อะไรลงที่ไหน (audit 2026-10 —
    # ลบ config dir ได้ทั้งโฟลเดอร์ และรันคำสั่งผ่าน option ของ git ได้) · ไซต์ที่มีรีโปสูตรของตัวเอง
    # ตั้งที่นี่ครั้งเดียว ทั้ง CLI และปุ่มบนหน้าเว็บใช้ค่าเดียวกัน
    sync_repo: str = ""
    sync_ref: str = ""


class Settings(BaseModel):
    provider: Optional[ProviderConfig] = None
    defaults: Defaults = Field(default_factory=Defaults)
    cluster: Cluster = Field(default_factory=Cluster)
    ui: Ui = Field(default_factory=Ui)
    recipes: Recipes = Field(default_factory=Recipes)

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
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            # YAML ถูกแต่รูปร่างผิด (ไฟล์ทั้งก้อนเป็นข้อความ · `recipes.sync_repo: 5`) — เดิมหลุดเป็น
            # ValidationError ของ pydantic ที่ไม่มีใครจับ หน้าเว็บจึงได้ 500 เปล่า ๆ ทุก route ที่อ่าน config
            # ทั้งที่ชั้นเว็บมีตัวจับ SettingsError รออยู่แล้ว (เจอตอนรัน repro ของ audit 2026-10)
            where = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'ทั้งไฟล์'}: {err['msg']}" for err in exc.errors()[:5])
            raise SettingsError(
                f"อ่าน {path} ไม่ได้ — ค่าในไฟล์ผิดรูป: {where}\n"
                f"แก้ไฟล์นี้ให้ถูกต้อง หรือลบทิ้งเพื่อเริ่มจากค่าเริ่มต้น (จะเสียค่า provider ที่ตั้งไว้)"
            ) from exc

    def save(self) -> None:
        ensure_config_dir()
        write_atomic(
            config_file(),
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        )

    def set_provider(self, name: ProviderName, model: str = "", base_url: str | None = None) -> ProviderConfig:
        resolved_model = model or DEFAULT_MODELS[name]
        if name is ProviderName.OPENAI_COMPAT and not base_url:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --base-url")
        if name is ProviderName.OPENAI_COMPAT and not resolved_model:
            raise ValueError("provider แบบ openai-compat ต้องระบุ --model")
        self.provider = ProviderConfig(name=name, model=resolved_model, base_url=base_url)
        return self.provider
