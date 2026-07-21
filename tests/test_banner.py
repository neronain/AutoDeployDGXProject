import io

from rich.console import Console

import lmds.cli.banner as banner_module
from lmds.cli.banner import BANNERS, CREDIT, show_banner


def _fresh(monkeypatch):
    monkeypatch.setattr(banner_module, "_shown", False)


def test_banner_art_is_reasonable_width():
    for art in BANNERS:
        for line in art.splitlines():
            assert len(line) <= 80, f"banner กว้างเกิน terminal มาตรฐาน: {line!r}"
    assert "neronain" in CREDIT and "fb.com/neronain.minidev" in CREDIT


def test_banner_suppressed_when_not_tty(monkeypatch):
    _fresh(monkeypatch)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False)
    monkeypatch.setattr("sys.stderr", buffer)  # StringIO ไม่มี isatty=True
    show_banner(console)
    assert buffer.getvalue() == ""


def test_banner_suppressed_by_env(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("LMDS_NO_BANNER", "1")

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    fake = FakeTty()
    monkeypatch.setattr("sys.stderr", fake)
    console = Console(file=fake, force_terminal=True)
    show_banner(console)
    assert fake.getvalue() == ""


def test_banner_shows_once_on_tty(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.delenv("LMDS_NO_BANNER", raising=False)

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    fake = FakeTty()
    monkeypatch.setattr("sys.stderr", fake)
    console = Console(file=fake, force_terminal=True, width=100)
    show_banner(console)
    first = fake.getvalue()
    assert "neronain" in first
    show_banner(console)  # ครั้งที่สองต้องเงียบ
    assert fake.getvalue() == first
