"""Web UI (เฟส 2) — import แบบ lazy เพราะ fastapi/uvicorn เป็น optional extra

ติดตั้ง: pip install 'lmds[web]'  (หรือ ./install.sh ถามให้)
"""


def create_app(token: str = ""):
    from .api import create_app as _create

    return _create(token)


def serve(host: str = "127.0.0.1", port: int = 8600, token: str = "") -> None:
    from .api import serve as _serve

    _serve(host, port, token)


__all__ = ["create_app", "serve"]
