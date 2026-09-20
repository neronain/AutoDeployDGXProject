"""Ed25519 — ใช้เท่าที่ระบบไลเซนส์ต้องการ: ตรวจลายเซ็น (ทุกเครื่อง) และเซ็น (เฉพาะเครื่องเรา)

ทำไมมีของสองทาง
---------------
ตลาดที่ LMDS ไปคือ DGX ในองค์กร ซึ่งเป็น **ตลาด air-gapped** เครื่องพวกนั้นลง wheel ที่ต้อง
คอมไพล์ไม่ได้ง่าย ๆ และ `cryptography` ลาก `cffi` + OpenSSL binding มาด้วย · แต่ถ้ามันมีอยู่แล้ว
ก็ควรใช้ เพราะเป็นโค้ดที่ผ่านตามาเยอะกว่าของที่เราแปะเอง

จึง: มี `cryptography` → ใช้ · ไม่มี → ใช้ตัวที่อยู่ในไฟล์นี้ (RFC 8032 reference)
ทั้งสองทางถูกเทสด้วย **test vector ของ RFC 8032 ชุดเดียวกัน** (tests/test_licensing.py)

เรื่องที่ต้องรู้ก่อนแตะไฟล์นี้
---------------------------
- โค้ดในนี้ทำงานกับ **กุญแจสาธารณะและลายเซ็น** เท่านั้นตอนตรวจ — ไม่มีความลับให้รั่ว
  timing attack จึงไม่ใช่ประเด็นสำหรับฝั่ง verify
- ฝั่ง sign รันบนเครื่องของเราเองตอนออกไลเซนส์ ไม่ได้รันบนเครื่องลูกค้า
- อย่า "ปรับให้เร็วขึ้น" โดยไม่รัน test vector ซ้ำ — เลขพวกนี้ผิดนิดเดียวคือตรวจผ่านทุกอย่าง
  หรือไม่ผ่านอะไรเลย และสองอาการนั้นดูเหมือนกันจากข้างนอก
"""

from __future__ import annotations

import hashlib

# ── พารามิเตอร์ของเส้นโค้ง (RFC 8032 §5.1) ──────────────────────────────────────
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _I % _P
    if (x * x - xx) % _P != 0:
        return None
    if x % 2 != sign:
        x = _P - x
    return x


_BY = 4 * pow(5, _P - 2, _P) % _P
_BX = _recover_x(_BY, 0)
_B = (_BX, _BY, 1, _BX * _BY % _P)  # พิกัดแบบ extended (X, Y, Z, T)


def _point_add(p: tuple, q: tuple) -> tuple:
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    dd = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, dd - c, dd + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _point_mul(scalar: int, point: tuple) -> tuple:
    result = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            result = _point_add(result, point)
        point = _point_add(point, point)
        scalar >>= 1
    return result


def _point_equal(p: tuple, q: tuple) -> bool:
    if (p[0] * q[2] - q[0] * p[2]) % _P != 0:
        return False
    return (p[1] * q[2] - q[1] * p[2]) % _P == 0


def _point_compress(p: tuple) -> bytes:
    zinv = pow(p[2], _P - 2, _P)
    x, y = p[0] * zinv % _P, p[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _point_decompress(data: bytes) -> tuple | None:
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _sha512_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little")


_IDENTITY = (0, 1, 1, 0)


def _small_order(point: tuple) -> bool:
    """จุดที่อยู่ในกลุ่มย่อยเล็ก (order หาร 8 ลงตัว) — ต้องปฏิเสธทั้งกุญแจและลายเซ็น

    ถ้าไม่ตรวจ: public key ที่เป็นศูนย์ทั้ง 32 ไบต์ คู่กับลายเซ็นศูนย์ทั้ง 64 ไบต์ **ตรวจผ่าน**
    เพราะสมการกลายเป็น identity = identity ทั้งสองข้าง · เจอจริงตอนเขียนเทส 2026-09-20:
    `cryptography` ปฏิเสธเคสนี้ถูกต้อง แต่ตัวที่เราแปะเองรับ — แปลว่าเครื่อง air-gapped
    (ซึ่งคือเครื่องที่ใช้ทางนี้) จะยอมรับไฟล์ที่เครื่องอื่นปฏิเสธ · libsodium ตัดเคสนี้ทิ้ง
    ด้วยเหตุผลเดียวกัน
    """
    return _point_equal(_point_mul(8, point), _IDENTITY)


def _verify_pure(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    point = _point_decompress(public_key)
    if point is None or _small_order(point):
        return False
    candidate = _point_decompress(signature[:32])
    if candidate is None or _small_order(candidate):
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:  # RFC 8032 §5.1.7 — ปฏิเสธ s ที่ไม่ได้ reduce (ไม่งั้นลายเซ็นดัดแปลงได้)
        return False
    h = _sha512_int(signature[:32] + public_key + message) % _L
    return _point_equal(_point_mul(s, _B), _point_add(candidate, _point_mul(h, point)))


def _secret_expand(secret: bytes) -> tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("private key ของ Ed25519 ต้องยาว 32 ไบต์")
    digest = hashlib.sha512(secret).digest()
    a = int.from_bytes(digest[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, digest[32:]


def _public_from_secret_pure(secret: bytes) -> bytes:
    a, _ = _secret_expand(secret)
    return _point_compress(_point_mul(a, _B))


def _sign_pure(secret: bytes, message: bytes) -> bytes:
    a, prefix = _secret_expand(secret)
    public = _point_compress(_point_mul(a, _B))
    r = _sha512_int(prefix + message) % _L
    rr = _point_compress(_point_mul(r, _B))
    h = _sha512_int(rr + public + message) % _L
    return rr + int.to_bytes((r + h * a) % _L, 32, "little")


# ── ทางที่เร็วกว่า ถ้าเครื่องนี้มี cryptography อยู่แล้ว ─────────────────────────────
try:  # pragma: no cover - ขึ้นกับว่าเครื่องนั้นมีอะไรลงไว้
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _PrivateKey,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding as _Encoding,
    )
    from cryptography.hazmat.primitives.serialization import (
        PublicFormat as _PublicFormat,
    )

    HAVE_CRYPTOGRAPHY = True
except Exception:  # ไม่มีก็ไม่เป็นไร — ใช้ตัวในไฟล์นี้
    HAVE_CRYPTOGRAPHY = False


def _well_formed(public_key: bytes, signature: bytes) -> bool:
    """ด่านร่วมของทั้งสองทาง — ต้องรันก่อนเสมอ ไม่ว่าเบื้องหลังจะเป็นอะไร

    ทำไมต้องอยู่ตรงนี้ ไม่ใช่ในทางใดทางหนึ่ง
    ------------------------------------
    เจอจริง 2026-09-20: `verify(b"\x00"*32, b"x", b"\x00"*64)` ให้ **True** บน
    cryptography/OpenSSL (ซึ่งตรวจแบบ cofactorless ไม่ปฏิเสธจุด small-order) แต่ให้
    **False** บนตัวที่เราแปะเอง · แปลว่าเครื่องสองเครื่องในฟลีตเดียวกันตัดสินไฟล์เดียวกัน
    ไม่เหมือนกัน ขึ้นกับว่าเครื่องไหนบังเอิญมี cryptography ลงไว้ — อาการที่หาสาเหตุยากที่สุด
    แบบหนึ่ง และเป็นสิ่งที่ระบบไลเซนส์ยอมให้เกิดไม่ได้เลย

    ทางแก้คือตรวจให้ครบ *ก่อน* แยกทาง ทั้งสองทางจึงเข้มเท่ากันเสมอ · เราเลือกทางเข้ม
    (ปฏิเสธ) เพราะกุญแจที่ฝังมากับโปรแกรมเป็นกุญแจจริงอยู่แล้ว ไม่มีเหตุผลที่ต้องรับ
    จุด small-order และ libsodium ก็ปฏิเสธด้วยเหตุผลเดียวกัน
    """
    if len(public_key) != 32 or len(signature) != 64:
        return False
    point = _point_decompress(public_key)
    if point is None or _small_order(point):
        return False
    candidate = _point_decompress(signature[:32])
    if candidate is None or _small_order(candidate):
        return False
    # s ต้องถูก reduce แล้ว (RFC 8032 §5.1.7) — ไม่งั้นลายเซ็นเดียวเขียนได้หลายแบบ
    return int.from_bytes(signature[32:], "little") < _L


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """True เมื่อ signature เป็นลายเซ็นของ message ด้วย public_key — ไม่เคย raise

    ตั้งใจให้ไม่ raise เพราะผู้เรียกทุกจุดสนใจแค่ "ผ่านหรือไม่ผ่าน" · ไฟล์ไลเซนส์ที่พังรูปแบบ
    ต้องได้คำตอบว่า "ไม่ผ่าน" ไม่ใช่ traceback กลางคำสั่งที่ผู้ใช้กำลังทำอย่างอื่นอยู่
    """
    try:
        if not _well_formed(public_key, signature):
            return False
    except Exception:
        return False

    if not HAVE_CRYPTOGRAPHY:
        try:
            return _verify_pure(public_key, message, signature)
        except Exception:
            return False
    try:
        _PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except (_InvalidSignature, ValueError, TypeError):
        return False
    except Exception:
        return False


def sign(secret_key: bytes, message: bytes) -> bytes:
    """เซ็น message — ใช้เฉพาะตอนออกไลเซนส์บนเครื่องของเรา ไม่ได้รันบนเครื่องลูกค้า"""
    if not HAVE_CRYPTOGRAPHY:
        return _sign_pure(secret_key, message)
    return _PrivateKey.from_private_bytes(secret_key).sign(message)


def public_from_secret(secret_key: bytes) -> bytes:
    """คืน public key 32 ไบต์ของ private key นี้"""
    if not HAVE_CRYPTOGRAPHY:
        return _public_from_secret_pure(secret_key)
    return (
        _PrivateKey.from_private_bytes(secret_key)
        .public_key()
        .public_bytes(_Encoding.Raw, _PublicFormat.Raw)
    )
