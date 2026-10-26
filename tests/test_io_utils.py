"""
tests/test_io_utils.py - the shared loader must turn parquet's numpy arrays
back into real Python lists (Phase 2 amendment 4).

WHY: pandas/pyarrow return list columns as numpy.ndarray after a parquet
round-trip. Every baseline and evaluation script relies on `tactics` being a
plain Python list, so load_messages() is the single place that guarantees it.

Run:  python -m pytest tests/test_io_utils.py -v
"""
import pandas as pd
import pytest

from src.io_utils import load_messages


def make_frame() -> pd.DataFrame:
    """A tiny frame with the full Phase 2 schema."""
    return pd.DataFrame({
        "id": ["a1", "a2"],
        "text": ["scam message here", "normal message here"],
        "verdict": ["scam", "safe"],
        "tactics": [["otp_or_pin_request", "urgency_threat"], []],
        "channel": ["sms", "sms"],
        "script_style": ["english", "english"],
        "date_received": [None, None],
        "source": ["synthetic", "synthetic"],
        "template_family": ["kyc_f1", ""],
        "labeler": ["t", "t"],
        "label_confidence": ["high", "high"],
        "url_type": ["none", "none"],
        "had_phone_number": [False, False],
        "template_cluster": [0, 1],
        "split": ["train", "val"],
        "notes": ["", ""],
    })


def test_tactics_roundtrip_parquet(tmp_path):
    """Write -> read must give Python lists, not numpy arrays, twice in a row."""
    p = tmp_path / "t.parquet"
    make_frame().to_parquet(p, index=False)

    loaded = load_messages(p)
    for value in loaded["tactics"]:
        assert isinstance(value, list), f"tactics is {type(value)}, not a list"
        assert all(isinstance(tag, str) for tag in value)
    assert loaded["tactics"].tolist() == [["otp_or_pin_request", "urgency_threat"], []]

    # Idempotent: loading again must not change anything.
    again = load_messages(p)
    assert again["tactics"].tolist() == loaded["tactics"].tolist()


def test_null_template_family_becomes_empty_string(tmp_path):
    p = tmp_path / "f.parquet"
    df = make_frame()
    df["template_family"] = [None, "  "]
    df.to_parquet(p, index=False)
    loaded = load_messages(p)
    assert loaded["template_family"].tolist() == ["", ""]


def test_load_messages_csv_normalises_semicolon_tactics(tmp_path):
    """label_cli writes 'a;b' into CSV cells - the loader must accept that too."""
    p = tmp_path / "t.csv"
    df = make_frame()
    df["tactics"] = ["otp_or_pin_request;urgency_threat", ""]
    df.to_csv(p, index=False)
    loaded = load_messages(p)
    assert loaded["tactics"].iloc[0] == ["otp_or_pin_request", "urgency_threat"]
    assert loaded["tactics"].iloc[1] == []


def test_missing_file_raises_friendly_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_messages(tmp_path / "nope.parquet")
