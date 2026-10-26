"""
tests/test_splits2.py - the leakage guarantees of the Phase 2 splitter.

Every test here encodes a rule from src/splits2.py's docstring. If any of
these ever fail, evaluation numbers from this project cannot be trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SEED, TACTIC_TAGS
from src.splits2 import build_groups, split_dataset2


def make_frame(n: int = 200, n_families: int = 40) -> pd.DataFrame:
    """A small but realistic table: scam rows spread over template families.

    Each row gets its own cluster; families repeat every n/n_families rows so
    several families exist for every tactic (the splitter needs >=4 groups
    per coverage key before it will hold one out).
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(n):
        rows.append({
            "id": f"t{i:04d}",
            "text": f"message number {i}",
            "verdict": "scam" if i % 4 else "safe",
            "tactics": [TACTIC_TAGS[i % len(TACTIC_TAGS)]] if i % 4 else [],
            "channel": "sms",
            "script_style": "english",
            "date_received": "",
            "source": "synthetic",
            "template_family": f"fam_{i % n_families}",
            "labeler": "test",
            "label_confidence": "high",
            "url_type": "none",
            "had_phone_number": False,
            "template_cluster": i,      # every row unique by default
            "split": "",
            "notes": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. the core guarantee: groups never span splits
# ---------------------------------------------------------------------------
def test_family_never_spans_splits():
    df = split_dataset2(make_frame())
    fam = df[df["template_family"] != ""]
    spans = fam.groupby("template_family")["split"].nunique()
    assert (spans == 1).all(), f"families leaked across splits: {spans[spans > 1]}"


def test_cluster_never_spans_splits():
    # Rows 0-3 deliberately share one cluster spanning two families: the
    # union-find must merge those families into ONE indivisible group.
    df = make_frame(120, n_families=24)
    df.loc[df.index[:2], "template_cluster"] = 999      # cluster links fam_0, fam_1
    out = split_dataset2(df)
    spans = out.groupby("template_cluster")["split"].nunique()
    assert (spans == 1).all(), "a template_cluster leaked across splits"


def test_shared_cluster_merges_families():
    df = make_frame(60, n_families=20)
    df.loc[df.index[:2], "template_cluster"] = 4242
    groups = build_groups(df)
    # the two rows sharing cluster 4242 must also share their group,
    # even though their template_family values differ.
    assert groups.iloc[0] == groups.iloc[1]


# ---------------------------------------------------------------------------
# 2. hard rules
# ---------------------------------------------------------------------------
def test_public_weak_forced_to_train():
    df = make_frame(200, n_families=40)
    df.loc[df.index[:20], "source"] = "public_weak"
    df.loc[df.index[:20], "template_family"] = ""   # weak rows have no family
    out = split_dataset2(df)
    weak = out[out["source"] == "public_weak"]
    assert (weak["split"] == "train").all(), "public_weak escaped the train split"


def test_synthetic_pre_marked_test_real_raises():
    df = make_frame(50, n_families=10)
    df.loc[df.index[:5], "split"] = "test_real"     # illegal: rows are synthetic
    with pytest.raises(AssertionError, match="test_real"):
        split_dataset2(df)


def test_pre_marked_real_rows_stay_in_test_real():
    df = make_frame(120, n_families=30)
    real = df.index[:6]
    df.loc[real, "source"] = "self_collected"
    df.loc[real, "split"] = "test_real"
    out = split_dataset2(df)
    assert (out.loc[list(real), "split"] == "test_real").all(), "real rows were moved"
    # Synthetic rows sharing a family with a test_real row must have been
    # dropped - otherwise they would leak the real template into training.
    real_fams = set(out.loc[list(real), "template_family"])
    syn_in_real_fams = out[(out["source"] == "synthetic")
                           & (out["template_family"].isin(real_fams))]
    assert len(syn_in_real_fams) == 0, "synthetic siblings of test_real rows survived"


# ---------------------------------------------------------------------------
# 3. test_b really holds out UNSEEN families
# ---------------------------------------------------------------------------
def test_test_b_families_are_unseen():
    df = split_dataset2(make_frame(240, n_families=48))
    tb_fams = set(df.loc[df["split"] == "test_b", "template_family"]) - {""}
    other = set(df.loc[df["split"] != "test_b", "template_family"]) - {""}
    assert tb_fams, "test_b ended up empty"
    assert not (tb_fams & other), "test_b families also appear in other splits"


def test_holdout_has_all_verdicts():
    # safe rows are held out by verdict key, so test_b must contain them.
    df = split_dataset2(make_frame(240, n_families=48))
    verdicts = set(df.loc[df["split"] == "test_b", "verdict"])
    assert "safe" in verdicts and "scam" in verdicts


# ---------------------------------------------------------------------------
# 4. ratios apply to the remaining groups (regression test for the bug where
#    the denominator wrongly included held-out rows and starved val/test_a)
# ---------------------------------------------------------------------------
def test_remaining_groups_fill_train_val_test_a():
    df = split_dataset2(make_frame(400, n_families=80))
    counts = df["split"].value_counts()
    assert counts.get("val", 0) > 0, "val split is empty"
    assert counts.get("test_a", 0) > 0, "test_a split is empty"
    # The 80/10/10 ratios are over REMAINING rows; allow generous slack.
    remaining = counts.get("train", 0) + counts.get("val", 0) + counts.get("test_a", 0)
    assert counts.get("train", 0) / remaining > 0.65
    assert counts.get("test_a", 0) / remaining < 0.30


def test_every_row_gets_a_split():
    df = split_dataset2(make_frame(150, n_families=30))
    assert df["split"].isin(
        ["train", "val", "test_a", "test_b", "test_real"]
    ).all()


def test_seeded_run_is_reproducible():
    a = split_dataset2(make_frame(120, n_families=24))["split"].tolist()
    b = split_dataset2(make_frame(120, n_families=24))["split"].tolist()
    assert a == b, "splitter is not reproducible despite seed 42"
