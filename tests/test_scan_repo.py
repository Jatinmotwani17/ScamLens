"""
tests/test_scan_repo.py - the scanner must catch PII and must not cry wolf.

Two properties matter:
  * a real-looking phone number / email / API key in a committable file FAILS
    the scan;
  * the placeholders this project uses everywhere (<PHONE>, <URL>, a masked
    example) do NOT fail it, or the gate becomes noise and gets bypassed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCANNER = Path(__file__).resolve().parents[1] / "scripts" / "scan_repo.py"


def load_scanner():
    spec = importlib.util.spec_from_file_location("scan_repo", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan_repo = load_scanner()


def write(tmp_path: Path, text: str, name: str = "sample.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_placeholder_tokens_are_not_findings(tmp_path):
    path = write(tmp_path, "Call <PHONE> or visit <URL>. Order <ORDER_ID>.\n")
    assert scan_repo.scan_file(path) == []


def test_documentation_example_is_clean(tmp_path):
    path = write(tmp_path, "Example only: +91 00000 00000 is not a real number.\n")
    # A run of zeroes is not a valid Indian mobile (must start 6-9).
    assert scan_repo.scan_file(path) == []


def test_a_real_looking_phone_number_fails(tmp_path):
    path = write(tmp_path, "contact me on 9876543210 please\n")
    names = [name for _, name, _ in scan_repo.scan_file(path)]
    assert "indian_phone" in names


def test_an_email_fails(tmp_path):
    path = write(tmp_path, "write to someone.here@example.com\n")
    names = [name for _, name, _ in scan_repo.scan_file(path)]
    assert "email" in names


@pytest.mark.parametrize("line,expected", [
    ("api_key = 'AIza" + "x" * 35 + "'", "google_api_key"),
    ("token = 'sk-" + "a" * 24 + "'", "openai_key"),
    ("AWS_KEY = 'AKIA" + "B" * 16 + "'", "aws_key"),
])
def test_api_keys_fail(tmp_path, line, expected):
    path = write(tmp_path, line + "\n")
    names = [name for _, name, _ in scan_repo.scan_file(path)]
    assert expected in names


def test_reading_a_key_from_the_environment_is_clean(tmp_path):
    path = write(tmp_path, "api_key = have_api_key()\n"
                           "token = os.environ.get('GEMINI_API_KEY')\n")
    assert scan_repo.scan_file(path) == []


def test_scan_allow_marker_skips_a_line(tmp_path):
    path = write(tmp_path, "call 9876543210  # scan:allow documentation example\n")
    assert scan_repo.scan_file(path) == []


def test_masking_never_prints_the_value():
    assert scan_repo.mask("9876543210") == "98******10"
    assert "7654321" not in scan_repo.mask("9876543210")
