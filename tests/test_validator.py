"""เทส Validator + Packager — ใช้ bundle จริงที่ generate จาก M5 เป็น fixture"""

from __future__ import annotations

import zipfile

import pytest

from lmds.packager import make_zip, write_checksums
from lmds.validator import CHECKSUM_FILE, all_passed, compute_checksums, run_gates
from tests.test_generator import gguf_report, make_bundle, safetensors_report


@pytest.fixture
def bundle_dir(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    return bundle.directory


def test_generated_bundle_passes_all_gates_before_checksums(bundle_dir):
    results = run_gates(bundle_dir, include_checksums=False)
    assert all_passed(results), [f"{r.name}: {r.detail}" for r in results if not r.passed]


def test_checksums_roundtrip(bundle_dir):
    write_checksums(bundle_dir)
    results = run_gates(bundle_dir, include_checksums=True)
    assert all_passed(results)


def test_checksum_detects_tampering(bundle_dir):
    write_checksums(bundle_dir)
    readme = bundle_dir / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nแก้ทีหลัง", encoding="utf-8")
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=True)}
    assert results["checksums"].passed is False


def test_missing_checksums_fails_with_hint(bundle_dir):
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=True)}
    assert results["checksums"].passed is False
    assert "--fix" in results["checksums"].detail


def test_numeric_underscore_gate_catches(bundle_dir):
    script = next(bundle_dir.glob("*.sh"))
    script.write_text(
        script.read_text(encoding="utf-8") + '\ncheck() { (( 1 > 25_000_000 )) || true; }\n',
        encoding="utf-8",
    )
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["numeric-underscore"].passed is False


def test_pipe_grep_q_gate_catches(bundle_dir):
    script = next(bundle_dir.glob("*.sh"))
    script.write_text(
        script.read_text(encoding="utf-8") + '\ncheck2() { docker info | grep -q nvidia; }\n',
        encoding="utf-8",
    )
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["pipefail-safe"].passed is False


def test_line_continuation_gate_catches_broken_command(bundle_dir):
    """เคสจริงจาก gigabyte02: '--host: command not found' — \\ ตามด้วยบรรทัดว่างกลางคำสั่ง docker run"""
    script = next(bundle_dir.glob("*.sh"))
    script.write_text(
        script.read_text(encoding="utf-8")
        + '\nbroken() {\n  docker run -d \\\n\n    --host "$API_HOST"\n}\n',
        encoding="utf-8",
    )
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["line-continuation"].passed is False
    assert results["bash-syntax"].passed is True  # พิสูจน์ว่า bash -n จับเคสนี้ไม่ได้ — ต้องมี gate แยก


def test_contract_gate_catches_missing_flag(bundle_dir):
    script = next(bundle_dir.glob("*.sh"))
    text = script.read_text(encoding="utf-8").replace("--client-input)", "--client-in)")
    script.write_text(text, encoding="utf-8")
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["controller-contract"].passed is False
    assert "--client-input" in results["controller-contract"].detail


def test_secret_scan_catches_leaked_token(bundle_dir):
    (bundle_dir / "README.md").write_text(
        "token: hf_ABCDEFGHIJKLMNOPQRSTUV", encoding="utf-8"
    )
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["secret-scan"].passed is False


def test_profile_schema_gate_catches_unpinned_revision(bundle_dir):
    profile = bundle_dir / "MODEL_PROFILE.yaml"
    text = profile.read_text(encoding="utf-8").replace("sha-gguf-456", "main")
    profile.write_text(text, encoding="utf-8")
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["profile-schema"].passed is False
    assert "pin" in results["profile-schema"].detail


def test_bash_syntax_gate_catches_broken_script(bundle_dir):
    script = next(bundle_dir.glob("*.sh"))
    script.write_text(script.read_text(encoding="utf-8") + "\nif [ x; then\n", encoding="utf-8")
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["bash-syntax"].passed is False


def test_zip_contains_all_files_under_slug(bundle_dir):
    write_checksums(bundle_dir)
    zip_path = make_zip(bundle_dir)
    assert zip_path.name == f"{bundle_dir.name}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert f"{bundle_dir.name}/README.md" in names
    assert f"{bundle_dir.name}/{CHECKSUM_FILE}" in names
    assert any(n.endswith("-single.sh") for n in names)


def test_zip_excluded_from_checksums(bundle_dir):
    write_checksums(bundle_dir)
    make_zip(bundle_dir)
    sums = compute_checksums(bundle_dir)
    assert not any(name.endswith(".zip") for name in sums)
    assert CHECKSUM_FILE not in sums


def test_vllm_bundle_also_passes(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    write_checksums(bundle.directory)
    assert all_passed(run_gates(bundle.directory, include_checksums=True))


def test_multimodal_gate_catches_controller_without_projector(isolated_config, tmp_path):
    """bundle ที่ประกาศ mmproj ใน profile แต่ controller ไม่โหลด/ไม่ส่ง --mmproj ต้องไม่ผ่าน

    นี่คือ bundle ที่หลุดถึงมือผู้ใช้จริง (gemma-4-12b-it-GGUF, 2026-08-03): gates เดิมเขียว
    ครบทุกด่านทั้งที่โมเดลเสิร์ฟภาพไม่ได้
    """
    from tests.test_generator import mmproj_gguf_report

    bundle, _, _ = make_bundle(mmproj_gguf_report(), tmp_path=tmp_path)
    results = {r.name: r for r in run_gates(bundle.directory, include_checksums=False)}
    assert results["multimodal-assets"].passed is True

    # ย้อน controller กลับไปเป็นเวอร์ชันที่มีบั๊ก: ลบทุกร่องรอยของ mmproj ออก
    controller = bundle.controller
    stripped = "\n".join(
        line for line in controller.read_text(encoding="utf-8").splitlines()
        if "mmproj" not in line.lower() and "MMPROJ" not in line
    )
    controller.write_text(stripped, encoding="utf-8")

    results = {r.name: r for r in run_gates(bundle.directory, include_checksums=False)}
    assert results["multimodal-assets"].passed is False
    assert "--mmproj" in results["multimodal-assets"].detail


def test_multimodal_gate_skips_text_only_bundles(bundle_dir):
    results = {r.name: r for r in run_gates(bundle_dir, include_checksums=False)}
    assert results["multimodal-assets"].passed is True
    assert "ไม่ใช่ multimodal" in results["multimodal-assets"].detail


def test_contract_gate_catches_missing_v3_banner(isolated_config, tmp_path):
    """ตัด banner()/info() ออก = ไม่ผ่าน controller-contract (กฎของ audit-controllers.py v3.0.0)"""
    bundle, _, _ = make_bundle(gguf_report(), tmp_path=tmp_path)
    assert run_gates(bundle.directory, include_checksums=False)

    controller = bundle.controller
    text = controller.read_text(encoding="utf-8")
    controller.write_text(text.replace("banner() {", "_disabled_banner() {"), encoding="utf-8")

    results = {r.name: r for r in run_gates(bundle.directory, include_checksums=False)}
    assert results["controller-contract"].passed is False
    assert "banner()" in results["controller-contract"].detail


def test_contract_gate_catches_missing_script_version(isolated_config, tmp_path):
    bundle, _, _ = make_bundle(safetensors_report(), tmp_path=tmp_path)
    controller = bundle.controller
    text = controller.read_text(encoding="utf-8")
    controller.write_text(
        text.replace('SCRIPT_VERSION="${SCRIPT_VERSION:-', 'SCRIPT_VERSION="', 1), encoding="utf-8"
    )

    results = {r.name: r for r in run_gates(bundle.directory, include_checksums=False)}
    assert results["controller-contract"].passed is False
    assert "SCRIPT_VERSION" in results["controller-contract"].detail


def test_stacked_gate_requires_cluster_prompt(isolated_config, tmp_path):
    """stacked ที่ไม่ถาม IP คลัสเตอร์ = ผู้ใช้ start ด้วย IP ตัวอย่างแล้วงงว่าทำไมต่อ worker ไม่ได้"""
    bundle, _, _ = make_bundle(safetensors_report(), target="dgx-spark-stacked", tmp_path=tmp_path)
    controller = bundle.controller
    text = controller.read_text(encoding="utf-8")
    controller.write_text(
        text.replace("prompt_cluster_config() {", "_disabled_prompt() {"), encoding="utf-8"
    )

    results = {r.name: r for r in run_gates(bundle.directory, include_checksums=False)}
    assert results["stacked-contract"].passed is False
    assert "prompt_cluster_config" in results["stacked-contract"].detail


def test_template_rendered_gate_catches_leftover_jinja(tmp_path):
    """Jinja ที่หลุดมาเป็น bash ที่ syntax ถูก — bash -n ผ่าน แล้วไปตายตอนรันจริง
    เคสจริง: {% if shard_files %} ถูกวางใน {% raw %} จึงไม่เคยถูกแปลง"""
    from lmds.validator.gates import gate_template_rendered

    bundle = tmp_path / "b"
    bundle.mkdir()
    script = bundle / "x-single.sh"
    script.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n{% if shard_files %}\necho hi\n",
                      encoding="utf-8")
    result = gate_template_rendered(bundle)
    assert not result.passed
    assert ":3:" in result.detail


def test_template_rendered_gate_catches_leftover_expression(tmp_path):
    """expression tag ก็หลุดได้: {{ slug }} ใน usage() ของ stacked controller อยู่ใน {% raw %}
    bash -n ผ่านเพราะมันเป็นแค่ข้อความใน heredoc — ผู้ใช้ถึงเห็น {{ slug }} ดิบ ๆ ตอนสั่ง help"""
    from lmds.validator.gates import gate_template_rendered

    bundle = tmp_path / "b3"
    bundle.mkdir()
    (bundle / "x-stacked.sh").write_text(
        "#!/usr/bin/env bash\nusage() {\n  cat <<EOF\n"
        "{{ slug }} \u2014 vLLM stacked controller\nEOF\n}\n",
        encoding="utf-8")
    result = gate_template_rendered(bundle)
    assert not result.passed
    assert ":4:" in result.detail


def test_template_rendered_gate_allows_docker_format_strings(tmp_path):
    """docker --format '{{.Names}}' ไม่ใช่ Jinja — ห้ามจับผิด"""
    from lmds.validator.gates import gate_template_rendered

    bundle = tmp_path / "b2"
    bundle.mkdir()
    (bundle / "x-single.sh").write_text(
        "#!/usr/bin/env bash\ndocker ps --format '{{.Names}}\\t{{.Status}}'\n"
        'echo "${VAR:-default}"\n', encoding="utf-8")
    assert gate_template_rendered(bundle).passed
