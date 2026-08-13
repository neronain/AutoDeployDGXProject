"""bundle รัน image ของตัวเอง ตรึงที่ digest ไม่ใช่ tag

tag เคลื่อนที่ได้ · `vllm/vllm-openai:latest` วันนี้กับเดือนหน้าเป็นคนละ image
bundle ที่ทดสอบผ่านแล้วจึงกลายเป็นคนละ runtime ได้โดยไม่มีอะไรในไฟล์เปลี่ยนเลย
ซึ่งเป็นอาการที่ไล่หายากที่สุด เพราะผู้ใช้ยืนยันว่าไม่ได้แก้อะไร — และเขาพูดถูก
"""

from __future__ import annotations

import httpx
import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lmds.brain.plan_schema import DeploymentPlan

TEMPLATES = "src/lmds/generator/templates"
BASE = {
    "model_id": "x/y", "revision": "main", "served_model_name": "y",
    "artifact_type": "safetensors", "topology": "single",
    "serving": {"context": 8192, "max_output_tokens": 1024},
}
DIGEST = "sha256:" + "a" * 64


def _image_line(template: str, plan: DeploymentPlan, var: str) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined)
    src = env.loader.get_source(env, template)[0]
    line = next(l for l in src.splitlines() if l.startswith(f"{var}="))
    return env.from_string(line).render(plan=plan)


def _plan(**runtime) -> DeploymentPlan:
    return DeploymentPlan.model_validate({**BASE, "runtime": {"engine": "vllm", **runtime}})


@pytest.mark.parametrize("template", [
    "single-vllm-controller.sh.j2", "stacked-vllm-controller.sh.j2",
])
def test_a_pinned_plan_runs_the_digest(template):
    plan = _plan(image_ref="vllm/vllm-openai:latest", image_pin=DIGEST)
    line = _image_line(template, plan, "VLLM_IMAGE")
    assert f"vllm/vllm-openai@{DIGEST}" in line
    # tag ต้องไม่หลุดมาเป็นสิ่งที่รัน — ไม่งั้นก็ไม่ได้ตรึงอะไรเลย
    assert ":latest}" not in line


@pytest.mark.parametrize("template", [
    "single-vllm-controller.sh.j2", "stacked-vllm-controller.sh.j2",
])
def test_without_a_digest_the_tag_still_works(template):
    """registry ที่ต้องล็อกอินหรือเครื่องไม่มีเน็ต ถาม digest ไม่ได้

    การห้าม deploy เพราะถาม registry ไม่ได้ แพงกว่าประโยชน์ที่ได้
    """
    plan = _plan(image_ref="vllm/vllm-openai:latest")
    assert "vllm/vllm-openai:latest" in _image_line(template, plan, "VLLM_IMAGE")


def test_the_override_still_works(request):
    """คนที่อยากลอง image อื่นต้องทำได้เหมือนเดิม — ตรึงไม่ใช่ล็อกตาย"""
    plan = _plan(image_ref="vllm/vllm-openai:latest", image_pin=DIGEST)
    line = _image_line("single-vllm-controller.sh.j2", plan, "VLLM_IMAGE")
    assert line.startswith('VLLM_IMAGE="${VLLM_IMAGE:-')


def test_llamacpp_is_pinned_too():
    plan = DeploymentPlan.model_validate({
        **BASE, "artifact_type": "gguf",
        "runtime": {"engine": "llamacpp", "image_ref": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                    "image_pin": DIGEST}})
    line = _image_line("single-llamacpp-controller.sh.j2", plan, "LLAMACPP_IMAGE")
    assert f"ghcr.io/ggml-org/llama.cpp@{DIGEST}" in line


# ---------------------------------------------------------------------------
# resolve_digest
# ---------------------------------------------------------------------------
def test_a_registry_that_cannot_be_asked_returns_none(monkeypatch):
    """nvcr.io ต้องล็อกอิน · None ต้องไม่ถูกตีความว่า 'ไม่มี image'"""
    from lmds.brain import registry

    monkeypatch.setattr(registry, "_ANON_TOKEN", {})
    assert registry.resolve_digest("nvcr.io/nvidia/vllm:26.05-py3") is None


def test_a_network_failure_returns_none(monkeypatch):
    from lmds.brain import registry

    class _Boom:
        def get(self, *a, **k):
            raise httpx.ConnectError("no route")

        def close(self):
            pass

    assert registry.resolve_digest("vllm/vllm-openai:latest", client=_Boom()) is None


def test_a_non_sha256_header_is_ignored(monkeypatch):
    """เฮดเดอร์แปลก ๆ ต้องไม่กลายเป็น image ref ที่รันไม่ได้"""
    from lmds.brain import registry

    class _Odd:
        def get(self, *a, **k):
            return httpx.Response(200, json={"token": "t"})

        def request(self, *a, **k):
            return httpx.Response(200, headers={"Docker-Content-Digest": "md5:whatever"})

        def close(self):
            pass

    assert registry.resolve_digest("vllm/vllm-openai:latest", client=_Odd()) is None
