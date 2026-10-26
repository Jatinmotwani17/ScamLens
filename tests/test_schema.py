"""
tests/test_schema.py - validate_schema must FAIL LOUDLY on bad data.

A silently-wrong dataset poisons every later phase, so we test that the
validator rejects: missing columns, bad enums, null/empty text, empty
verdicts, bad dates, unknown tactic tags, duplicate ids - and that the
leakage check catches a cluster spanning splits or synthetic rows in test.

Run:  python -m pytest tests/test_schema.py -v
"""
import pandas as pd
import pytest

from src.data_checks import binary_table_check, leakage_check, validate_schema
from src.splits import split_dataset


def valid_df(n: int = 4) -> pd.DataFrame:
    """A tiny but fully-valid frame."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"t{i:04d}",
            "text": f"scrubbed message number {i}",
            "verdict": ["scam", "suspicious", "safe", "scam"][i % 4],
            "tactics": [["otp_or_pin_request"], [], [], ["urgency_threat"]][i % 4],
            "channel": "sms",
            "script_style": "english",
            "date_received": "2024-01-05" if i % 2 == 0 else None,
            "source": "self_collected",
            "template_family": "",
            "labeler": "test",
            "label_confidence": "high",
            "url_type": "none",
            "had_phone_number": False,
            "template_cluster": i,
            "split": "train",
            "real_stratum": "",
            "notes": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. the happy path
# ---------------------------------------------------------------------------
def test_valid_frame_passes():
    validate_schema(valid_df())  # should not raise


def test_date_month_only_allowed():
    df = valid_df()
    df.loc[0, "date_received"] = "2023-11"
    validate_schema(df)


def test_empty_date_allowed():
    df = valid_df()
    df["date_received"] = None
    validate_schema(df)


# ---------------------------------------------------------------------------
# 2. each failure mode must raise ValueError with a useful message
# ---------------------------------------------------------------------------
def test_missing_column_raises():
    df = valid_df().drop(columns=["verdict"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df)


def test_bad_verdict_enum_raises():
    df = valid_df()
    df.loc[1, "verdict"] = "ham"  # not one of scam/suspicious/safe
    with pytest.raises(ValueError, match="bad verdict"):
        validate_schema(df)


def test_bad_source_enum_raises():
    df = valid_df()
    df.loc[0, "source"] = "whatsapp_forward"  # not in SOURCES
    with pytest.raises(ValueError, match="bad source"):
        validate_schema(df)


def test_bad_split_enum_raises():
    df = valid_df()
    df.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError, match="bad split"):
        validate_schema(df)


def test_old_two_letter_test_split_is_rejected():
    """Phase 1's plain 'test' split no longer exists in the vocabulary."""
    df = valid_df()
    df.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="bad split"):
        validate_schema(df)


def test_null_template_family_raises():
    df = valid_df()
    df.loc[0, "template_family"] = None
    with pytest.raises(ValueError, match="template_family"):
        validate_schema(df)


def test_empty_verdict_raises():
    df = valid_df()
    df.loc[2, "verdict"] = ""
    with pytest.raises(ValueError, match="empty verdict"):
        validate_schema(df)


def test_null_text_raises():
    df = valid_df()
    df.loc[0, "text"] = None
    with pytest.raises(ValueError, match="null text"):
        validate_schema(df)


def test_blank_text_raises():
    df = valid_df()
    df.loc[0, "text"] = "   "
    with pytest.raises(ValueError, match="empty text"):
        validate_schema(df)


def test_unknown_tactic_tag_raises():
    df = valid_df()
    # Rebuild the column (pandas 3 disallows list-into-cell .loc assignment).
    df["tactics"] = [["otp_or_pin", "made_up_tag"] if i == 0 else t
                     for i, t in enumerate(df["tactics"])]
    with pytest.raises(ValueError, match="unknown tactic tag"):
        validate_schema(df)


def test_bad_date_format_raises():
    df = valid_df()
    df.loc[0, "date_received"] = "next tuesday"
    with pytest.raises(ValueError, match="bad date_received"):
        validate_schema(df)


def test_duplicate_id_raises():
    df = valid_df()
    df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_schema(df)


def test_bad_url_type_raises():
    df = valid_df()
    df.loc[0, "url_type"] = "phishy"
    with pytest.raises(ValueError, match="bad url_type"):
        validate_schema(df)


# ---------------------------------------------------------------------------
# 3. leakage check
# ---------------------------------------------------------------------------
def test_leakage_check_passes_on_clean_data():
    df = valid_df()
    df.loc[df.index[1:], "split"] = "test_a"
    status, problems = leakage_check(df)
    assert status == "PASS", problems


def test_leakage_check_catches_cluster_spanning_splits():
    df = valid_df()
    df["split"] = ["train", "train", "test", "test"]
    df["template_cluster"] = [0, 0, 0, 1]  # cluster 0 spans train+test
    status, problems = leakage_check(df)
    assert status == "FAIL"
    assert any("span" in p for p in problems)


def test_leakage_check_allows_synthetic_in_ordinary_splits():
    """Phase 2: synthetic rows ARE allowed in train/val/test_a/test_b."""
    df = valid_df()
    df["source"] = "synthetic"
    df["split"] = ["train", "val", "test_a", "test_b"]
    status, problems = leakage_check(df)
    assert status == "PASS", problems


def test_leakage_check_catches_synthetic_in_test_real():
    """HARD RULE: a synthetic row must never land in test_real."""
    df = valid_df()
    df["source"] = "synthetic"
    df["split"] = ["train", "val", "test_a", "test_real"]
    status, problems = leakage_check(df)
    assert status == "FAIL"
    assert any("test_real" in p for p in problems)


def test_leakage_check_catches_template_family_spanning_splits():
    """A template family must never cross splits (test_b isolation included)."""
    df = valid_df()
    df["template_family"] = ["kyc_f1", "kyc_f1", "otp_f1", "otp_f1"]
    df["split"] = ["train", "test_a", "test_b", "test_b"]
    status, problems = leakage_check(df)
    assert status == "FAIL"
    assert any("template_family" in p for p in problems)


# ---------------------------------------------------------------------------
# 4. end-to-end: a split_dataset output must pass both checks
# ---------------------------------------------------------------------------
def test_exact_text_leakage_is_caught_even_with_different_clusters():
    """Phase 2.5: identical normalised text may never span two splits.

    The cluster check alone can miss this (e.g. if clustering ran before the
    duplicate was introduced), so leakage_check has a second, exact-text test.
    """
    df = valid_df()
    df["template_cluster"] = [0, 1, 2, 3]      # four separate clusters
    df["text"] = ["Rs 500 sent", "RS   500 SENT", "hello", "hi"]
    df["split"] = ["train", "test_a", "train", "test_b"]
    status, problems = leakage_check(df)
    assert status == "FAIL"
    assert any("EXACT-TEXT" in p for p in problems)


def test_binary_table_rejects_suspicious():
    """The main table is scam/safe only; suspicious is Phase 4 material."""
    df = valid_df()
    problems = binary_table_check(df)
    assert any("suspicious" in p for p in problems)


def test_split_output_passes_schema_and_leakage():
    rows = []
    for c in range(10):
        for j in range(4):
            rows.append({
                "id": f"e{c}_{j}",
                "text": f"template {c} v{j}",
                "verdict": "scam" if c % 2 else "safe",
                "tactics": [] if c % 2 else ["suspicious_link"],
                "channel": "sms",
                "script_style": "english",
                "date_received": f"2024-{(c % 9) + 1:02d}-01",
                "source": "synthetic" if c == 9 else "self_collected",
                "template_family": "",
                "labeler": "t",
                "label_confidence": "high",
                "url_type": "none",
                "had_phone_number": False,
                "template_cluster": c,
                "split": "train",
                "real_stratum": "",
                "notes": "",
            })
    df = split_dataset(pd.DataFrame(rows))
    validate_schema(df)
    status, problems = leakage_check(df)
    assert status == "PASS", problems
