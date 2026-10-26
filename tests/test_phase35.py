"""
tests/test_phase35.py - the Phase 3.5 guarantees.

These tests read the BUILT table (data/processed/messages.parquet) and skip
when it does not exist yet, so a fresh checkout still runs the fast tests.
Run `python -m src.build_dataset` first to exercise them for real.

Covered rules:
  * every tactic tag has >=100 train positives, >=1 in val, >=1 in test_b;
  * a real near-duplicate cluster never crosses splits;
  * the same exact text never appears in two splits;
  * mix_synth_only excludes real rows, mix_synth_plus_real includes them;
  * public_weak (easy UCI spam) is never trained on;
  * a single-class stratum reports macro-F1 as None, not 0.5 or 0.0;
  * the external train cap is respected.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import CFG, TACTIC_TAGS, data_path
from src.data_checks import exact_text_key
from src.evaluate import verdict_metrics

REAL_SOURCES = {"external_reported", "public_ham_eval",
                "self_collected", "external"}
SYNTHETIC_SOURCES = {"synthetic", "synthetic_llm"}
MIN_TRAIN_POS = int(CFG["splits2"].get("min_train_positives_per_tag", 100))


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    path = data_path("messages_parquet")
    if not path.exists():
        pytest.skip("messages.parquet not built (run python -m src.build_dataset)")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def support(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag in TACTIC_TAGS:
        counts = {"tag": tag}
        for split in ("train", "val", "test_a", "test_b"):
            sub = df[df["split"] == split]
            counts[split] = int(sum(tag in t for t in sub["tactics"]))
        real = df[df["split"] == "test_real"]
        counts["real_test"] = int(sum(tag in t for t in real["tactics"]))
        rows.append(counts)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. tag support
# ---------------------------------------------------------------------------
def test_every_tag_has_100_train_positives(support: pd.DataFrame):
    bad = support[support["train"] < MIN_TRAIN_POS]
    assert bad.empty, (
        f"tags below {MIN_TRAIN_POS} train positives:\n"
        f"{bad[['tag', 'train']].to_string(index=False)}")


def test_every_tag_has_val_and_test_b_positive(support: pd.DataFrame):
    bad = support[(support["val"] < 1) | (support["test_b"] < 1)]
    assert bad.empty, (
        "tags with no val/test_b positive:\n"
        f"{bad[['tag', 'val', 'test_b']].to_string(index=False)}")


def test_every_tag_has_five_families_and_60pct_in_train(df: pd.DataFrame):
    """Tag-safe split rule: >=5 families per tag, >=60% of them in train."""
    syn = df[df["source"].isin(SYNTHETIC_SOURCES)
             & df["template_family"].astype(str).str.strip().ne("")]
    min_frac = float(CFG["splits2"].get("min_train_family_fraction", 0.60))
    problems = []
    for tag in TACTIC_TAGS:
        sub = syn[[tag in t for t in syn["tactics"]]]
        fams = set(sub["template_family"])
        train_fams = set(sub.loc[sub["split"] == "train", "template_family"])
        if len(fams) < 5:
            problems.append(f"{tag}: only {len(fams)} family/families (<5)")
        if fams and len(train_fams) / len(fams) < min_frac:
            problems.append(
                f"{tag}: {len(train_fams)}/{len(fams)} families in train "
                f"(<{min_frac:.0%})")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 2. leakage
# ---------------------------------------------------------------------------
def test_real_cluster_never_crosses_splits(df: pd.DataFrame):
    real = df[df["source"].isin(REAL_SOURCES)]
    spans = real.groupby("template_cluster")["split"].nunique()
    assert (spans == 1).all(), (
        f"{int((spans > 1).sum())} real cluster(s) in more than one split; "
        f"examples: {spans[spans > 1].index[:5].tolist()}")


def test_exact_text_never_crosses_splits(df: pd.DataFrame):
    keyed = df.assign(_key=df["text"].map(exact_text_key))
    spans = keyed.groupby("_key")["split"].nunique()
    assert (spans == 1).all(), (
        f"{int((spans > 1).sum())} exact text(s) in two splits")


def test_no_synthetic_row_in_real_test(df: pd.DataFrame):
    bad = df[df["source"].isin(SYNTHETIC_SOURCES) & (df["split"] == "test_real")]
    assert bad.empty, f"{len(bad)} synthetic row(s) in test_real"


# ---------------------------------------------------------------------------
# 3. training mixes
# ---------------------------------------------------------------------------
def test_mix_synth_only_uses_only_synthetic(df: pd.DataFrame):
    from src.train import load_frames

    frames = load_frames("mix_synth_only")
    sources = set(frames["train"]["source"])
    assert sources <= SYNTHETIC_SOURCES, f"unexpected train sources: {sources}"


def test_mix_plus_real_includes_real_rows(df: pd.DataFrame):
    from src.train import load_frames

    frames = load_frames("mix_synth_plus_real")
    sources = set(frames["train"]["source"])
    assert "external_reported" in sources
    assert "public_ham_eval" in sources
    assert frames["val_real"]["source"].isin(REAL_SOURCES).all()
    assert frames["val_synthetic"]["source"].isin(SYNTHETIC_SOURCES).all()


def test_unknown_mix_raises(df: pd.DataFrame):
    from src.train import load_frames

    with pytest.raises(ValueError, match="unknown train mix"):
        load_frames("mix_everything")


def test_public_weak_never_trained(df: pd.DataFrame):
    from src.train import load_frames

    for mix in ("mix_synth_only", "mix_synth_plus_real"):
        frames = load_frames(mix)
        assert "public_weak" not in set(frames["train"]["source"])
        assert "public_weak" not in set(frames["val"]["source"])


def test_external_train_cap_respected(df: pd.DataFrame):
    cap = int(CFG["phase35"]["external_train_max_rows"])
    ext = df[(df["split"] == "train")
             & df["source"].isin(["external_reported", "public_ham_eval"])]
    assert len(ext) <= cap, f"{len(ext)} external train rows > cap {cap}"


# ---------------------------------------------------------------------------
# 4. single-class metrics
# ---------------------------------------------------------------------------
def test_single_class_stratum_hides_macro_f1():
    one_class = pd.DataFrame({"verdict": ["scam", "scam", "scam"]})
    m = verdict_metrics(one_class, ["scam", "scam", "scam"])
    assert m["single_class"] is True
    assert m["macro_f1"] is None          # NOT 0.5 and NOT 0.0
    assert m["scam_recall"] == 1.0

    both = pd.DataFrame({"verdict": ["scam", "safe"]})
    m2 = verdict_metrics(both, ["scam", "safe"])
    assert m2["single_class"] is False
    assert m2["macro_f1"] == pytest.approx(1.0)


def test_real_strata_are_never_blended(df: pd.DataFrame):
    """test_real must be reported per stratum; the combined block is explicit."""
    real = df[df["split"] == "test_real"]
    assert set(real["real_stratum"]) <= {
        "user_real_scam", "user_real_safe", "external_scam",
        "public_ham_english", ""}
    # Every real row carries a stratum label.
    assert (real["real_stratum"].astype(str).str.len() > 0).all()


# ---------------------------------------------------------------------------
# 5. Phase 3.6: test_a is a BOTH-CLASS cluster hold-out from train
# ---------------------------------------------------------------------------
def test_test_a_has_both_classes(df: pd.DataFrame):
    """A one-class test_a has no macro-F1 and cannot test memorisation."""
    ta = df[df["split"] == "test_a"]
    assert len(ta) > 0, "test_a is empty"
    classes = set(ta["verdict"].astype(str))
    assert {"scam", "safe"} <= classes, (
        f"test_a must contain both classes; found {sorted(classes)} "
        f"({int((ta['verdict'] == 'scam').sum())} scam / "
        f"{int((ta['verdict'] == 'safe').sum())} safe)")


def test_test_a_is_held_out_from_train_by_cluster(df: pd.DataFrame):
    """test_a is ~10% of the TRAIN clusters, taken whole (never split)."""
    ta = df[df["split"] == "test_a"]
    train = df[df["split"] == "train"]
    assert set(ta["template_cluster"]).isdisjoint(set(train["template_cluster"])), (
        "test_a shares a template_cluster with train")
    assert ta["source"].isin(SYNTHETIC_SOURCES).all(), (
        "test_a must stay synthetic; real rows are evaluated in test_real")
    # Whole-cluster granularity makes the split land near 10% of train rows,
    # not exactly on it. The band is deliberately wide; it only has to prove
    # test_a is a hold-out slice, not a re-labelled copy of train.
    share = len(ta) / max(len(train), 1)
    assert 0.02 <= share <= 0.25, f"test_a is {share:.1%} of train rows"


# ---------------------------------------------------------------------------
# 6. Phase 3.6: metric guards
# ---------------------------------------------------------------------------
def test_recall_at_fpr_is_guarded_by_negative_count():
    """recall@FPR is only reported when n_negatives * FPR >= 5."""
    from src.evaluate import TOO_FEW_NEG, probability_metrics

    # 50 negatives: 1% -> 0.5 expected negatives, 5% -> 2.5. Both too few.
    y_small = ["safe"] * 50 + ["scam"] * 50
    s_small = [0.1] * 50 + [0.9] * 50
    small = probability_metrics(y_small, s_small)
    assert small["recall_at_1pct_fpr"] is None
    assert small["recall_at_1pct_fpr_note"] == TOO_FEW_NEG
    assert small["recall_at_5pct_fpr"] is None
    assert small["recall_at_5pct_fpr_note"] == TOO_FEW_NEG
    assert small["n_negatives"] == 50

    # 1,000 negatives: 1% -> 10 expected negatives, a measurable point.
    y_big = ["safe"] * 1000 + ["scam"] * 100
    s_big = [0.1] * 1000 + [0.9] * 100
    big = probability_metrics(y_big, s_big)
    assert big["recall_at_1pct_fpr"] == 1.0
    assert big["recall_at_1pct_fpr_note"] is None


def test_balanced_accuracy_is_none_for_one_class():
    from src.evaluate import verdict_metrics

    one = verdict_metrics(pd.DataFrame({"verdict": ["scam", "scam"]}),
                          ["scam", "safe"])
    assert one["balanced_accuracy"] is None       # not 0.5, not 0.0
    both = verdict_metrics(pd.DataFrame({"verdict": ["safe", "scam"]}),
                           ["scam", "scam"])
    # safe recall 0, scam recall 1 -> balanced accuracy 0.5
    assert both["balanced_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# 7. Phase 3.6: nearest-neighbour audit output
# ---------------------------------------------------------------------------
def test_nn_sim_parquet_contract(df: pd.DataFrame):
    """results/nn_sim.parquet: one row per non-train row, valid similarity."""
    path = data_path("nn_sim")
    if not path.exists():
        pytest.skip("nn_sim.parquet not built (run python -m src.nn_sim)")
    sim = pd.read_parquet(path)
    assert {"id", "split", "stratum", "max_train_sim"} <= set(sim.columns)
    assert sim["id"].is_unique
    assert "train" not in set(sim["split"].astype(str))
    assert sim["max_train_sim"].between(0.0, 1.0).all()
    expected = set(df.loc[df["split"] != "train", "id"].astype(str))
    assert set(sim["id"].astype(str)) == expected
