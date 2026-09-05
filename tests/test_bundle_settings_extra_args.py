"""แฟล็กเพิ่มของ engine ต้องรอดผ่านการบันทึกทั้งที่มี JSON อยู่ข้างใน

เคสจริง spark-worker (2026-09-03): จะเปิด MTP ให้ Qwen3.5-122B บน vLLM ต้องส่ง
--speculative-config '{"method":"mtp","num_speculative_tokens":2}' ซึ่ง `lmds set` ตั้งไม่ได้เลย
(docs/USAGE.md ก็ยอมรับไว้ว่า "ยังตั้งผ่าน LMDS ไม่ได้") · plan ของ LMDS เองแนะให้เปิด MTP
แต่ controller ไม่มีช่องให้ใส่

ใส่ลง bundle.env ไม่ได้เพราะรูป ${VAR:-value} ของ bash หยุดที่ `}` ตัวแรก — ทดสอบแล้ว:
    X="${X:---speculative-config {"method":"mtp"} --foo}"  →  ได้ '{"method":"mtp"' กับ '--foo}'
จึงเก็บในไฟล์แยก bundle.args ที่ controller อ่านทั้งบรรทัด
"""

import pytest

from lmds.fleet.bundle_settings import ARGS_FILENAME, SettingsError, read, write

MTP = '--speculative-config {"method":"mtp","num_speculative_tokens":2} --enable-chunked-prefill'


def test_extra_args_round_trip_keeps_json_intact(tmp_path):
    write(tmp_path, {"port": 8000, "extra_args": MTP})
    assert read(tmp_path)["extra_args"] == MTP
    # อยู่ในไฟล์ของมันเอง ไม่ใช่ใน bundle.env
    assert (tmp_path / ARGS_FILENAME).read_text(encoding="utf-8").strip() == MTP
    assert "speculative" not in (tmp_path / "bundle.env").read_text(encoding="utf-8")


def test_extra_args_survive_without_any_other_setting(tmp_path):
    """ตั้งแค่แฟล็กเพิ่มอย่างเดียว — bundle.env ไม่ถูกสร้าง แต่ต้องยังอ่านกลับได้"""
    write(tmp_path, {"extra_args": MTP})
    assert not (tmp_path / "bundle.env").exists()
    assert read(tmp_path) == {"extra_args": MTP}


def test_clearing_extra_args_removes_the_file(tmp_path):
    write(tmp_path, {"extra_args": MTP})
    write(tmp_path, {"port": 8000, "extra_args": ""})
    assert not (tmp_path / ARGS_FILENAME).exists()
    assert "extra_args" not in read(tmp_path)


def test_extra_args_reject_shell_expansion():
    """quote กับปีกกาต้องผ่าน (JSON ใช้) แต่ $ และ backtick ทำให้เชลล์รันของอื่นได้"""
    with pytest.raises(SettingsError):
        write.__wrapped__ if hasattr(write, "__wrapped__") else None
        from lmds.fleet.bundle_settings import _clean
        _clean("extra_args", "--foo $(rm -rf /)")
    from lmds.fleet.bundle_settings import _clean
    assert _clean("extra_args", '  --a   {"k":"v"}  ') == '--a {"k":"v"}'


def test_parsers_persist_in_bundle_env(tmp_path):
    """tool/reasoning parser เคยตั้งได้แค่ตอน start จึงหายตอน autostart"""
    write(tmp_path, {"tool_parser": "qwen3_xml", "reasoning_parser": "qwen3"})
    text = (tmp_path / "bundle.env").read_text(encoding="utf-8")
    assert 'TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"' in text
    assert 'REASONING_PARSER="${REASONING_PARSER:-qwen3}"' in text
    assert read(tmp_path) == {"tool_parser": "qwen3_xml", "reasoning_parser": "qwen3"}


def test_parser_names_are_identifiers_only():
    from lmds.fleet.bundle_settings import _clean
    with pytest.raises(SettingsError):
        _clean("tool_parser", "qwen3_xml; rm -rf /")


def test_extra_args_refuse_flags_the_controller_owns(tmp_path):
    """audit stacked รอบ 2: `lmds set --extra-args "--tensor-parallel-size 1"` ลง bundle.args ได้ทั้งดุ้น → vLLM ให้ตัวหลังชนะ
    TP=1 บน 2 เครื่อง (harden กันฝั่ง LLM ตั้งแต่ 0.6.0 แต่ทาง set หลุด)"""
    import pytest

    from lmds.fleet.bundle_settings import SettingsError, write

    for bad in ("--tensor-parallel-size 1", "--nnodes=1", "--master-addr 10.0.0.1 --port 1", "-tp 2"):
        with pytest.raises(SettingsError) as exc:
            write(tmp_path, {"extra_args": bad})
        assert "controller ตั้งให้เอง" in str(exc.value), bad
    assert write(tmp_path, {"extra_args": "--max-num-batched-tokens 4096"})["extra_args"] == "--max-num-batched-tokens 4096"
