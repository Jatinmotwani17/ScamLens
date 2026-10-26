"""
tests/test_phase25.py - the Phase 2.5 rules that protect every later score.

Covers:
  * exact-duplicate removal uses the normalised text (lowercase, collapsed
    whitespace, digits -> 0) and keeps the REAL copy over a synthetic one;
  * real_stratum is assigned only to test_real rows;
  * the main table is binary (scam/safe), never suspicious;
  * a metadata-only shortcut model trains and predicts.
"""
from __future__ import annotations

import pandas as pd

from src.build_dataset import assign_real_stratum, drop_exact_duplicates
from src.shortcut import SingleFeatureShortcut
from src.data_checks import binary_table_check


def _row(i: int, text: str, source: str = "synthetic",
         split: str = "", verdict: str = "scam") -> dict:
    return {
        "id": f"r{i}", "text": text, "verdict": verdict, "tactics": [],
        "channel": "sms", "script_style": "english", "date_received": "",
        "source": source, "template_family": "f1", "labeler": "t",
        "label_confidence": "high", "url_type": "none",
        "had_phone_number": False, "template_cluster": i, "split": split,
        "real_stratum": "", "notes": "",
    }


def test_exact_duplicate_normalisation():
    df = pd.DataFrame([
        _row(0, "Rs 500 OTP bhejo", "synthetic"),
        _row(1, "rs   500   otp BHEJO", "external_reported", split="test_real"),
        _row(2, "totally different", "synthetic"),
    ])
    out = drop_exact_duplicates(df)
    assert len(out) == 2
    # The real row wins the duplicate fight.
    assert "external_reported" in set(out["source"])


def test_assign_real_stratum_only_for_test_real():
    df = pd.DataFrame([
        _row(0, "a", "self_collected", "test_real", "scam"),
        _row(1, "b", "self_collected", "test_real", "safe"),
        _row(2, "c", "external_reported", "test_real", "scam"),
        _row(3, "d", "public_ham_eval", "test_real", "safe"),
        _row(4, "e", "synthetic", "train", "scam"),
    ])
    out = assign_real_stratum(df)
    assert out.loc[0, "real_stratum"] == "user_real_scam"
    assert out.loc[1, "real_stratum"] == "user_real_safe"
    assert out.loc[2, "real_stratum"] == "external_scam"
    assert out.loc[3, "real_stratum"] == "public_ham_english"
    assert out.loc[4, "real_stratum"] == ""


def test_binary_check_passes_without_suspicious():
    df = pd.DataFrame([_row(0, "a", verdict="scam"),
                       _row(1, "b", verdict="safe")])
    assert binary_table_check(df) == []


def test_style_shortcut_trains_and_predicts():
    df = pd.DataFrame({
        "script_style": ["english", "english", "roman_hinglish", "roman_hinglish"],
        "verdict": ["safe", "scam", "safe", "scam"],
    })
    model = SingleFeatureShortcut("script_style").fit(
        df["script_style"].tolist(), df["verdict"].tolist())
    preds = model.predict(["english", "roman_hinglish"])
    assert len(preds) == 2
    assert set(preds) <= {"safe", "scam"}
