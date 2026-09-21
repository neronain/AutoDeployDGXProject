"""API key ของ model server และร่องรอยการใช้งาน — สำหรับคอนโซลเว็บ

ทำไมอยู่คนละไฟล์กับ `api.py`: สองเรื่องนี้เป็นงานความปลอดภัยที่เพิ่งเพิ่มเข้ามา และ
`api.py` ยาว 2,800 บรรทัดแล้ว · แยกไว้ทำให้เห็นขอบเขตของสิ่งที่แตะความลับได้ชัดเจน
และรีวิวได้โดยไม่ต้องอ่านทั้งไฟล์

**ทำไมต้องมี**: `lmds key` กับ `lmds audit` มีแต่ทาง CLI · ลูกค้าที่ใช้งานผ่าน GUI
เป็นหลัก — ซึ่งตอนนี้คือส่วนใหญ่ — จึงตั้ง key ไม่ได้และดูร่องรอยไม่ได้เลย
ที่ย้อนแย้งกว่านั้นคือ wizard deploy บนหน้าเว็บ *พิมพ์ข้อความบอกให้ไปรัน*
`lmds key show <slug> --reveal` ซึ่งเป็นคำสั่งที่คนกลุ่มนั้นพิมพ์ไม่ได้อยู่แล้ว
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException


def register(app: FastAPI, guarded: list[Depends], check_slug) -> None:
    """ติด route ของ key/audit เข้ากับแอป

    รับ `guarded` กับ `check_slug` มาจาก `create_app()` แทนที่จะ import เอง — ทั้งสองตัว
    ผูกกับ token ของแอปนั้น ๆ (create_app รับ token เป็นพารามิเตอร์) การสร้างใหม่ที่นี่
    จะได้ตัวที่ตรวจคนละ token กับที่เหลือของระบบ
    """

    @app.get("/api/models/{slug}/key", dependencies=guarded)
    def key_status(slug: str) -> dict:
        """มี key เก็บไว้ให้ bundle นี้ไหม — **ไม่คืนค่า key**

        การคืนค่าเต็มทุกครั้งที่หน้าเว็บ poll แปลว่า key เดินทางผ่านเครือข่ายและไปนอน
        อยู่ใน memory ของเบราว์เซอร์ตลอดเวลาโดยไม่มีใครขอ · ดูค่าจริงต้องกดเอง (`?reveal=1`)
        """
        from lmds.fleet import apikey

        check_slug(slug)
        value = apikey.read(slug)
        return {
            "slug": slug,
            "has_key": bool(value),
            "path": str(apikey.path_for(slug)),
            # ปลายสี่ตัวพอให้เทียบกับที่ client ถืออยู่ว่าใช่ใบเดียวกันไหม โดยไม่ต้องเปิดค่าเต็ม
            "hint": f"{value[:4]}…{value[-4:]}" if len(value) >= 8 else "",
        }

    @app.get("/api/models/{slug}/key/reveal", dependencies=guarded)
    def key_reveal(slug: str) -> dict:
        """ค่าเต็มของ key — แยก endpoint ออกมาเพื่อให้ audit log แยกได้ว่าใครกด 'ดูค่า'

        middleware เก็บเฉพาะคำสั่งที่เปลี่ยนสถานะกับคำขอที่ถูกปฏิเสธ · GET ที่ผ่านไม่ถูกเก็บ
        การอ่าน key จึงไม่โผล่ใน audit — ยอมรับได้ เพราะคนที่ถือ token ของคอนโซลอ่าน
        อะไรก็ได้อยู่แล้ว และการเก็บทุกการอ่านจะกลบคำสั่งที่เปลี่ยนของจริงจนหาไม่เจอ
        """
        from lmds.fleet import apikey

        check_slug(slug)
        value = apikey.read(slug)
        if not value:
            raise HTTPException(status_code=404, detail=f"ยังไม่มี key เก็บไว้ให้ '{slug}'")
        return {"slug": slug, "key": value}

    @app.post("/api/models/{slug}/key", dependencies=guarded)
    def key_set(slug: str, body: dict | None = None) -> dict:
        """ตั้ง key — ไม่ส่ง `key` มา = สุ่มให้

        `force` จำเป็นตอนทับของเดิม เพราะ client ทุกตัวที่ถือใบเก่าจะใช้ไม่ได้ทันที
        ที่โมเดล restart · การทับโดยไม่ตั้งใจจากปุ่มที่กดพลาดแพงกว่าการต้องยืนยันหนึ่งครั้ง
        """
        from lmds.fleet import apikey

        check_slug(slug)
        body = body or {}
        if apikey.read(slug) and not body.get("force"):
            raise HTTPException(
                status_code=409,
                detail=f"'{slug}' มี key อยู่แล้ว — ยืนยันด้วย force เพื่อทับ "
                       "(client ทุกตัวที่ถือใบเก่าต้องเปลี่ยนตาม)",
            )
        supplied = str(body.get("key") or "").strip()
        try:
            value = supplied or apikey.mint()
            apikey.write(slug, value)
        except (apikey.ApiKeyError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # คืนค่าเต็มครั้งเดียวตรงนี้ — คนเพิ่งกดสร้าง ต้องได้เอาไปตั้งที่ client
        return {"slug": slug, "key": value, "generated": not supplied,
                "restart_required": True}

    @app.delete("/api/models/{slug}/key", dependencies=guarded)
    def key_clear(slug: str) -> dict:
        """เอา key ออก — bundle นี้จะกลับไปเสิร์ฟแบบไม่ต้องยืนยันตัวตนหลัง restart"""
        from lmds.fleet import apikey

        check_slug(slug)
        removed = apikey.clear(slug)
        return {"slug": slug, "removed": removed, "restart_required": removed,
                "warning": "หลัง restart bundle นี้จะเสิร์ฟแบบเปิด" if removed else ""}

    @app.get("/api/audit", dependencies=guarded)
    def audit_list(lines: int = 100, failed_only: bool = False) -> dict:
        """ใครสั่งอะไรกับคอนโซลนี้บ้าง

        ลูกค้าที่เปิดคอนโซลออกวง network ต้องเห็นเองว่ามีใครพยายามเข้า — ซึ่งเป็นสิ่งที่
        `lmds audit --failed` ทำได้อยู่แล้ว แต่คนที่ใช้ GUI อย่างเดียวเข้าไม่ถึง
        """
        from lmds.web import audit as audit_log

        limit = max(1, min(int(lines), 1000))
        # ขอเผื่อแล้วค่อยกรอง ไม่งั้น failed_only กับ limit น้อย ๆ ได้ไม่ครบเมื่อของส่วนใหญ่ผ่าน
        items = audit_log.read(limit * 20 if failed_only else limit)
        if failed_only:
            items = [i for i in items if int(i.get("status") or 0) in {401, 403, 429}]
        items = items[-limit:]
        refused = sum(1 for i in items if int(i.get("status") or 0) in {401, 403, 429})
        return {
            "enabled": audit_log.enabled(),
            "path": str(audit_log.log_path()),
            "entries": items,
            "refused": refused,
        }
