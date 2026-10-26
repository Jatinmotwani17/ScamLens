"""
tests/test_splits.py - the leakage guards live here.

The promises of src/splits.py (Phase 2 rules):
  1. no template_cluster ever spans two splits
  2. public_weak rows stay in train; synthetic rows are allowed in
     train/val/test_a but NEVER in test_real
  3. rows pre-marked test_real (real messages) keep their split
  4. the split is deterministic (seed 42) and honours ~70/15/15

Run:  python -m pytest tests/test_splits.py -v
"""
import pandas as pd
import pytest

from src.splits import parse_date, split_dataset


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal frame that satisfies split_dataset's inputs."""
    base = pd.DataFrame(rows)
    for col, default in [
        ("tactics", []), ("channel", "sms"), ("script_style", "english"),
        ("date_received", None), ("labeler", "t"), ("label_confidence", "high"),
        ("url_type", "none"), ("had_phone_number", False), ("notes", ""),
    ]:
        if col not in base.columns:
            # NOTE: a list default must be copied per row, never shared.
            if isinstance(default, list):
                base[col] = [list(default) for _ in range(len(base))]
            else:
                base[col] = default
    if "template_cluster" not in base.columns:
        base["template_cluster"] = range(len(base))
    return base


def synthetic_frame(n_clusters: int = 20, per_cluster: int = 5) -> pd.DataFrame:
    """20 clusters x 5 rows = 100 rows, with dates spread over time."""
    rows = []
    for c in range(n_clusters):
        for j in range(per_cluster):
            rows.append({
                "id": f"r{c:03d}_{j}",
                "text": f"template {c} variant {j}",
                "verdict": "scam" if c % 2 == 0 else "safe",
                "source": "self_collected",
                # spread dates month by month so temporal ordering is meaningful
                "date_received": f"2024-{(c % 12) + 1:02d}-15",
                "template_cluster": c,
            })
    return make_df(rows)


# ---------------------------------------------------------------------------
def test_no_cluster_spans_splits():
    df = split_dataset(synthetic_frame())
    spans = df.groupby("template_cluster")["split"].nunique()
    assert (spans == 1).all(), f"cluster(s) span splits: {spans[spans > 1]}"


def test_split_ratio_roughly_70_15_15():
    df = split_dataset(synthetic_frame())
    frac = df["split"].value_counts(normalize=True)
    # Whole clusters make exact ratios impossible; allow generous slack.
    assert 0.55 <= frac.get("train", 0) <= 0.85
    assert 0.05 <= frac.get("val", 0) <= 0.30
    assert 0.05 <= frac.get("test_a", 0) <= 0.30


def test_temporal_split_puts_newest_in_test():
    """With full date coverage, the newest rows must land in test.

    Train holds the oldest 70% of clusters, so the newest train date must be
    older than or equal to the oldest test date (no time-traveling leakage).
    """
    df = split_dataset(synthetic_frame())
    df = df.assign(_d=df["date_received"].map(parse_date))
    newest_train = df[df["split"] == "train"]["_d"].max()
    oldest_test = df[df["split"] == "test_a"]["_d"].min()
    assert newest_train is not None and oldest_test is not None
    assert newest_train <= oldest_test, (
        f"temporal order broken: newest train {newest_train} > oldest test {oldest_test}"
    )


def test_group_split_when_dates_missing():
    """Below the date-coverage threshold we must fall back to a group split."""
    df = synthetic_frame()
    df.loc[df.index[::2], "date_received"] = None  # ~50% dated < 60% threshold
    out = split_dataset(df)
    spans = out.groupby("template_cluster")["split"].nunique()
    assert (spans == 1).all()


def test_synthetic_rows_never_in_test_real():
    """Phase 2: synthetic MAY sit in train/val/test_a, but never test_real."""
    df = synthetic_frame()
    # Mark 3 whole clusters as synthetic, with the NEWEST dates so a temporal
    # split would happily push them into the test side.
    df.loc[df["template_cluster"].isin([17, 18, 19]), "source"] = "synthetic"
    out = split_dataset(df)
    in_real = out[(out["source"] == "synthetic") & (out["split"] == "test_real")]
    assert len(in_real) == 0, f"synthetic rows in test_real: {in_real['id'].tolist()}"
    # ...but they are allowed in the ordinary splits now (Phase 2 change).
    assert set(out[out["source"] == "synthetic"]["split"]).issubset(
        {"train", "val", "test_a"}
    )


def test_synthetic_row_pre_marked_test_real_crashes():
    """Hard rule: handing split_dataset a synthetic test_real row must crash."""
    df = synthetic_frame()
    df["split"] = "train"
    df.loc[df.index[0], "source"] = "synthetic"
    df.loc[df.index[0], "split"] = "test_real"
    with pytest.raises(AssertionError, match="test_real"):
        split_dataset(df)


def test_public_weak_rows_always_in_train():
    df = synthetic_frame()
    df.loc[df["template_cluster"] == 19, "source"] = "public_weak"
    out = split_dataset(df)
    escaped = out[(out["source"] == "public_weak") & (out["split"] != "train")]
    assert len(escaped) == 0


def test_cluster_with_synthetic_row_stays_in_one_split():
    """A mixed cluster (one synthetic row inside) must still never span splits."""
    df = synthetic_frame()
    df.loc[df["id"] == "r019_0", "source"] = "synthetic"  # newest cluster
    out = split_dataset(df)
    cluster_rows = out[out["template_cluster"] == 19]
    assert cluster_rows["split"].nunique() == 1, "mixed cluster spanned splits!"


def test_pre_marked_test_real_rows_are_preserved():
    """Real rows pre-marked test_real keep it, and their whole cluster with them."""
    df = synthetic_frame()
    df["split"] = "train"
    # rows of cluster 0 (source stays self_collected = real) -> test_real
    df.loc[df["template_cluster"] == 0, "split"] = "test_real"
    out = split_dataset(df)
    c0 = out[out["template_cluster"] == 0]
    assert (c0["split"] == "test_real").all(), "real cluster lost its test_real split"
    others = out[out["template_cluster"] != 0]
    assert (others["split"] != "test_real").all(), "non-real row leaked into test_real"


def test_deterministic_across_runs():
    a = split_dataset(synthetic_frame())["split"].tolist()
    b = split_dataset(synthetic_frame())["split"].tolist()
    assert a == b, "split is not deterministic with seed 42"


def test_missing_template_cluster_raises():
    df = synthetic_frame()
    df["template_cluster"] = None
    with pytest.raises(ValueError):
        split_dataset(df)


def test_parse_date_formats():
    assert str(parse_date("2024-05-17")) == "2024-05-17"
    assert str(parse_date("2024-05")) == "2024-05-01"
    assert parse_date("") is None
    assert parse_date(None) is None
    assert parse_date("not-a-date") is None
