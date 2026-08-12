"""แถบ progress ต้องไปถึงหน้าเว็บ — ทั้งที่มันไม่เคยขึ้นบรรทัดใหม่เลย

huggingface_hub, docker pull, rsync และ curl เลื่อนตัวเลขด้วย `\r` ล้วน ๆ
`for line in proc.stdout` ตัดที่ `\n` อย่างเดียว จึงไม่คายอะไรออกมาเลยจนกว่างานจะจบ
— download 50 GB เงียบสนิทครึ่งชั่วโมง แล้วผู้ใช้ก็ไปฆ่างานที่กำลังไปได้ดีทิ้ง
"""

from __future__ import annotations

from lmds.web import jobs


class _Pipe:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def read1(self, _size=-1):
        chunk = next(self._chunks, b"")
        return chunk


class _Proc:
    def __init__(self, chunks):
        self.stdout = _Pipe(chunks)


def _run(chunks) -> list[str]:
    job = jobs.Job(id="t", slug="s", command="c", steps=["c"])
    jobs._pump(job, _Proc(chunks))
    return list(job.lines)


def test_a_carriage_return_ends_a_line():
    """ถ้าไม่นับ \\r เป็นตัวจบบรรทัด แถบ progress จะไม่โผล่มาเลยสักตัว"""
    assert _run([b"Downloading  12%\r"]) == ["Downloading  12%\n"]


def test_each_progress_frame_replaces_the_last():
    """เฟรมถัดไปตั้งใจทับของเดิม — ถ้า append ทุกเฟรม 400 บรรทัดจะเต็มไปด้วยเลข %
    ของวินาทีที่แล้ว แล้วดันบรรทัดที่บอกสาเหตุจริงหายไปหมด"""
    lines = _run([b"10%\r20%\r30%\r"])
    assert lines == ["30%\n"]


def test_a_real_line_after_a_progress_bar_is_kept():
    lines = _run([b"10%\r90%\rdone\n"])
    assert lines == ["done\n"]


def test_normal_lines_still_accumulate():
    assert _run([b"first\nsecond\n"]) == ["first\n", "second\n"]


def test_a_line_split_across_reads_is_reassembled():
    """ท่อจริงไม่เคารพขอบเขตบรรทัด — 8 KB ตัดกลางคำได้เสมอ"""
    assert _run([b"half of a li", b"ne here\n"]) == ["half of a line here\n"]


def test_output_with_no_final_newline_is_not_dropped():
    """คำสั่งที่ตายกลางทางมักทิ้งบรรทัดสุดท้ายไว้ไม่จบ — และนั่นคือบรรทัดที่บอกสาเหตุ"""
    assert _run([b"Traceback: boom"]) == ["Traceback: boom\n"]


def test_crlf_counts_once():
    """ปลายทาง Windows หรือ log ที่ผ่าน tty จำลองมา ส่ง \\r\\n มาคู่กัน"""
    assert _run([b"line\r\n"]) == ["line\n"]


def test_invalid_utf8_does_not_kill_the_job():
    """ไบต์เสียหนึ่งตัวไม่ควรทำให้ทั้งงานที่รันอยู่หายไป"""
    assert _run([b"caf\xff\n"]) == ["caf�\n"]
