"""ป้าย `restart to apply` และ `mmproj unused` บนการ์ด — รันสคริปต์จริงของ index.html ใน DOM ย่อส่วน (tests/console_shell_dom.js)

audit 2026-09-08 msi-6: `lmds set --model-id` แล้ว API ยังตอบชื่อเก่า ไม่มีอะไรบนหน้าเว็บบอกว่าต้อง restart · ป้าย `custom`
ที่มีอยู่คือกรณีกลับกัน (start ด้วย flag ที่ไม่ได้บันทึก) · llama.cpp VL-embedding ขึ้น vision ทั้งที่ /v1/embeddings ทิ้งภาพ
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario

FLEET = """const fx = { nodes: [
  { name: "msi-6", site: "TKC", models: [
      { slug: "qwen", running: true, healthy: true, engine: "vllm", port: 8000, context: 65536, context_configured: 65536,
        features: "tools, image", served_name: "old-name", commands: ["start", "restart", "logs"],
        pending_restart: { pending: true, changes: [{ field: "served_name", saved: "new-name", running: "old-name" }] } },
      { slug: "quiet", running: true, healthy: true, engine: "vllm", port: 8001, context: 65536, context_configured: 65536,
        features: "text", commands: ["start"], pending_restart: { pending: false, changes: [] } },
      { slug: "vlembed", running: true, healthy: true, engine: "llamacpp", port: 8080, context: 8192, context_configured: 8192,
        features: "text, embedding (last)", projector: false, commands: ["start", "test-embed"],
        feature_note: "mmproj มาพร้อมไฟล์ แต่ llama-server ไม่เอาภาพเข้า /v1/embeddings" } ] } ],
  localModels: [
    { slug: "local-qwen", running: true, healthy: true, controller_exists: true, downloaded: true, engine: "vllm", port: 8000,
      context: 65536, context_configured: 65536, features: "tools", served_name: "old", commands: ["start"],
      pending_restart: { pending: true, changes: [{ field: "context", saved: "131072", running: "65536" },
                                                  { field: "extra_args", saved: "--speculative-config {\\"method\\":\\"mtp\\"}", running: "(ไม่อยู่บน argv)" }] } } ] };
H.fx = fx;
H.routes = H.defaultRoutes(fx);
"""


def test_node_and_local_cards_show_restart_to_apply_with_the_changed_fields(tmp_path):
    (out,) = run_scenario(tmp_path, FLEET, """
        await H.tick(6);
        const tag = sel => [...document.querySelectorAll(sel)].map(el => ({ text: el.textContent.trim(), title: el.getAttribute("title") || "" }));
        const card = slug => document.querySelector(`[data-slug="${slug}"]`);
        const inRow = (slug, attr) => {
          const rows = [...document.querySelectorAll("*")].filter(el => el.children.length === 0 && el.textContent.trim() === slug);
          return rows.some(el => (el.closest(".field") || el.parentElement).querySelector(`[${attr}]`));
        };
        console.log(JSON.stringify({
          pending: tag("[data-restart-pending]"),
          notes: tag("[data-feature-note]"),
          vision: [...document.querySelectorAll(".tag.feat")].map(el => el.textContent.trim()),
          quietHasTag: inRow("quiet", "data-restart-pending"),
          qwenHasTag: inRow("qwen", "data-restart-pending"),
          localHasTag: !!document.querySelector('#local .tag[data-restart-pending], [data-restart-pending]'),
        }));
    """)
    texts = [p["text"] for p in out["pending"]]
    assert texts and all(t == "restart to apply" for t in texts)
    assert len(out["pending"]) == 2, out            # การ์ด node (qwen) + การ์ดในเครื่อง (local-qwen) — quiet ไม่มี
    titles = " | ".join(p["title"] for p in out["pending"])
    assert "served_name: saved new-name · running old-name" in titles
    assert "context: saved 131072 · running 65536" in titles and "extra_args" in titles
    assert out["qwenHasTag"] is True and out["quietHasTag"] is False
    # llama.cpp VL-embedding: ไม่มีป้าย vision · มีป้ายอธิบายว่า mmproj ไม่ถูกใช้
    assert [n["text"] for n in out["notes"]] == ["mmproj unused"] and "/v1/embeddings" in out["notes"][0]["title"]
    assert out["vision"].count("vision") == 1, "มีแค่ qwen (features มี image) ที่ได้ป้าย vision"
