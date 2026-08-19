"""hub ให้ node ยืม HF token ได้ โดย secret ต้องไม่ไปนอนอยู่ที่ไหน

เคสจริง 2026-08-20: repo gated ถูก push ไป msi-5 แล้ว download ล้มเพราะ node ไม่มี token
· hub มีอยู่ แต่ไม่มีช่องทางส่งให้ · bundle จงใจไม่พก token ไปด้วย ซึ่งถูกแล้ว
"""

from collections import deque

from lmds.web import jobs


class _Job:
    def __init__(self, lines):
        self.lines = deque(lines, maxlen=400)


def test_secret_is_scrubbed_from_the_job_log():
    """คำสั่งปลายทาง "ไม่ควร" พิมพ์ token — แต่ไม่ควรกับไม่เคยคนละเรื่อง"""
    token = "hf_" + "a" * 34
    job = _Job([f"downloading with {token}\n", "ok\n"])
    jobs._scrub_secrets(job, {"HF_TOKEN": token})
    joined = "".join(job.lines)
    assert token not in joined
    assert "ok" in joined


def test_scrub_keeps_the_line_cap():
    token = "hf_" + "b" * 34
    job = _Job([f"{token}\n"] * 5)
    jobs._scrub_secrets(job, {"HF_TOKEN": token})
    assert job.lines.maxlen == 400
    assert token not in "".join(job.lines)


def test_nothing_to_scrub_is_not_an_error():
    job = _Job(["ok\n"])
    jobs._scrub_secrets(job, None)
    jobs._scrub_secrets(job, {})
    jobs._scrub_secrets(job, {"HF_TOKEN": ""})
    assert "".join(job.lines) == "ok\n"


def test_secret_travels_by_stdin_not_argv():
    """argv ของ ssh มองเห็นได้จาก `ps` ของทุก user บนเครื่อง hub"""
    import subprocess

    from lmds.nodes import ssh

    captured = {}

    class _FakeProc:
        def __init__(self):
            self.stdin = _FakeStdin()

    class _FakeStdin:
        def __init__(self):
            self.written = b""

        def write(self, data):
            self.written += data

        def flush(self):
            pass

        def close(self):
            captured["stdin"] = self.written

    def fake_popen(args, **kwargs):
        captured["argv"] = args
        captured["stdin_mode"] = kwargs.get("stdin")
        return _FakeProc()

    node = type("N", (), {"all_hosts": ["h"], "port": 22, "user": "u"})()
    original = subprocess.Popen
    subprocess.Popen = fake_popen
    try:
        ssh.stream(node, "lmds repair x", {"HF_TOKEN": "hf_secret_value_123456"})
    finally:
        subprocess.Popen = original

    assert "hf_secret_value_123456" not in " ".join(captured["argv"])
    assert captured["stdin"] == b"hf_secret_value_123456\n"
    assert "read -r HF_TOKEN" in " ".join(captured["argv"])
