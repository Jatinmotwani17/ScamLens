"""
src/splits.py - leakage-safe train/val/test_a splitting (Phase 1 splitter).

THE ONE RULE THAT MATTERS:
  a template_cluster may NEVER appear in more than one split.
If two copies of the same scam template sit in train AND test, the model can
memorise the template and your test score becomes a lie. We crash loudly if
this ever happens.

STRATEGY (Phase 1 temporal/group split, kept for compatibility):
  * Rows already marked split=="test_real" (your REAL messages) are KEPT
    exactly where they are - their whole cluster is pinned to test_real and
    they are never re-assigned. Real data never trains a model in this phase.
  * If >= 60% of rows have a usable date_received -> TEMPORAL split of the
    rest: oldest 70% of rows -> train, next 15% -> val, newest 15% -> test_a.
    Whole clusters move together (a cluster belongs to the split of its
    earliest message).
  * Otherwise -> GROUP split: shuffled clusters (seed 42) assigned 70/15/15.
  * source=public_weak rows (English UCI spam) ALWAYS go to train only, and
    their whole cluster goes with them.

PHASE 2 RULE CHANGES vs Phase 1:
  * synthetic rows are ALLOWED in train / val / test_a (and in test_b, which
    is produced by src/splits2.py) - they are no longer forced into train.
  * synthetic rows may NEVER sit in test_real. We crash loudly if they do.
  * the split vocabulary is now train/val/test_a/test_b/test_real; this
    module only ever produces train/val/test_a (+ keeps test_real intact).
    test_b (unseen template families) comes from src/splits2.py, which is the
    MAIN splitter used by src/build_dataset.py in Phase 2.

Run:  python -m src.splits   (self-test on a synthetic frame; normally you
                              call split_dataset(df) from src/splits2.py)
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import CFG, SEED, SPLITS

SPLIT_CFG = CFG["splits"]

# Sources that may never leave the train split (English UCI spam - it would
# otherwise create a language shortcut in eval).
TRAIN_ONLY_SOURCES = ("public_weak",)

# Machine-generated rows: allowed in train/val/test_a/test_b, never test_real.
SYNTHETIC_SOURCES = ("synthetic", "synthetic_llm")

# REAL external rows (src/external_data.py): real human-reported messages
# (IMC25 smishing, UCI ham reserve) - always pinned to test_real for honest
# evaluation, never allowed into train/val/test_a/test_b.
REAL_EVAL_SOURCES = ("external_reported", "public_ham_eval")

# Your OWN real messages (pasted + labeled by a human in Phase 1) - in
# Phase 2 these never train a model either; they live in test_real only.
REAL_OWN_SOURCES = ("self_collected", "family_friends")


def parse_date(value) -> date | None:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM' into a date. Anything else -> None.

    A YYYY-MM value becomes the first of that month - good enough for ordering.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    parts = s.split("-")
    try:
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return date(int(parts[0]), int(parts[1]), 1)
    except ValueError:
        return None
    return None


def _assign_by_cumulative_ratio(cluster_order: list, cluster_sizes: dict, ratios: tuple[float, ...]) -> dict:
    """Walk clusters in the given order, filling train then val then test_a.

    Boundaries are hit as closely as whole clusters allow. Returns
    {cluster_id: split_name}. Every cluster gets a split (tail -> test_a).
    """
    cut_train = ratios[0]
    cut_val = ratios[0] + ratios[1]
    total = sum(cluster_sizes.values()) or 1
    assignment: dict = {}
    cum = 0
    for cid in cluster_order:
        frac = cum / total
        if frac < cut_train:
            assignment[cid] = "train"
        elif frac < cut_val:
            assignment[cid] = "val"
        else:
            assignment[cid] = "test_a"
        cum += cluster_sizes[cid]
    return assignment


def split_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column (train/val/test_a, keeping test_real) to a copy.

    Seeded (42) and loud: every guarantee is asserted, not assumed.
    """
    rng = np.random.default_rng(SEED)
    df = df.copy()

    ratios = (float(SPLIT_CFG["train"]), float(SPLIT_CFG["val"]), float(SPLIT_CFG["test_a"]))

    # ---- step 0: sanity - every row must already have a template_cluster ---
    if df["template_cluster"].isna().any():
        raise ValueError("split_dataset: some rows have no template_cluster. Run dedup first.")

    # ---- step 1: rows already marked test_real stay put --------------------
    # (test_real is filled by real, human-labeled messages only. Their whole
    #  cluster is pinned with them so no cluster can ever span splits.)
    if "split" in df.columns:
        pre_real = df["split"].astype(str) == "test_real"
    else:
        pre_real = pd.Series(False, index=df.index)
    real_clusters = set(df.loc[pre_real, "template_cluster"].unique().tolist())

    # HARD RULE: a synthetic row must never be in test_real. Crash loudly.
    bad_real = df[pre_real & df["source"].isin(SYNTHETIC_SOURCES)]
    if len(bad_real) > 0:
        raise AssertionError(
            f"HARD RULE VIOLATED: {len(bad_real)} synthetic row(s) are marked "
            f"test_real! ids: {bad_real['id'].head(10).tolist()}. test_real is "
            "for real human-labeled messages only."
        )

    # ---- step 2: decide temporal vs group split ---------------------------
    parsed = df["date_received"].map(parse_date)
    coverage = parsed.notna().mean()
    use_temporal = coverage >= float(SPLIT_CFG["temporal_min_date_coverage"])
    print(f"splits: date coverage = {coverage:.1%} -> "
          f"{'TEMPORAL' if use_temporal else 'GROUP'} split")

    # ---- step 3: clusters containing train-only sources are forced to train
    forced_clusters = set(
        df.loc[df["source"].isin(TRAIN_ONLY_SOURCES), "template_cluster"].unique().tolist()
    )
    # A cluster cannot be both "pinned to test_real" and "forced to train".
    conflict = forced_clusters & real_clusters
    if conflict:
        raise AssertionError(
            f"cluster(s) {sorted(conflict)[:10]} contain BOTH public_weak and "
            "test_real rows - that cannot be split safely."
        )
    if forced_clusters:
        print(f"splits: {len(forced_clusters)} cluster(s) contain public_weak "
              f"rows and are forced into TRAIN.")

    # ---- step 4: order the remaining clusters ------------------------------
    # Earliest date per cluster (undated clusters sort first so they are more
    # likely to land in train - conservative for evaluation purity).
    skip = forced_clusters | real_clusters
    if use_temporal:
        df["_parsed_date"] = parsed.map(lambda d: d or date(1900, 1, 1))
        earliest = df.groupby("template_cluster")["_parsed_date"].min()
        remaining = [c for c in earliest.index if c not in skip]
        cluster_order = sorted(remaining, key=lambda c: (earliest[c], c))
    else:
        remaining = [c for c in df["template_cluster"].unique() if c not in skip]
        shuffled = np.array(remaining, dtype=object)
        rng.shuffle(shuffled)
        cluster_order = [int(c) for c in shuffled]

    cluster_sizes = df.groupby("template_cluster").size().to_dict()
    assignment = _assign_by_cumulative_ratio(cluster_order, cluster_sizes, ratios)
    for c in forced_clusters:
        assignment[c] = "train"
    for c in real_clusters:
        assignment[c] = "test_real"   # real messages keep their own split

    df["split"] = df["template_cluster"].map(assignment).astype(str)

    # ---- step 5: CRASH LOUDLY if a cluster spans splits --------------------
    spans = df.groupby("template_cluster")["split"].nunique()
    bad = spans[spans > 1]
    if len(bad) > 0:
        raise AssertionError(
            f"LEAKAGE: {len(bad)} template_cluster(s) appear in more than one split! "
            f"Example cluster ids: {bad.index[:10].tolist()}"
        )

    # ---- step 6: CRASH LOUDLY on the Phase 2 hard rules --------------------
    escaped = df[(df["source"].isin(TRAIN_ONLY_SOURCES)) & (df["split"] != "train")]
    if len(escaped) > 0:
        raise AssertionError(
            f"LEAKAGE: {len(escaped)} public_weak row(s) outside train! "
            f"ids: {escaped['id'].head(10).tolist()}"
        )
    syn_real = df[(df["source"].isin(SYNTHETIC_SOURCES)) & (df["split"] == "test_real")]
    if len(syn_real) > 0:
        raise AssertionError(
            f"HARD RULE VIOLATED: {len(syn_real)} synthetic row(s) ended up in "
            f"test_real! ids: {syn_real['id'].head(10).tolist()}"
        )

    # ---- step 7: print a human summary -------------------------------------
    counts = df["split"].value_counts().reindex(SPLITS, fill_value=0)
    print("splits: rows per split:", counts.to_dict())
    bal = pd.crosstab(df["split"], df["verdict"])
    print("splits: verdict balance per split:")
    print(bal.reindex([s for s in SPLITS if s in bal.index], fill_value=0).to_string())

    # test_b and test_real are NOT produced by this module - warn only about
    # the three splits it is responsible for filling.
    test_scam = int(bal.loc["test_a", "scam"]) if "test_a" in bal.index and "scam" in bal.columns else 0
    min_rows = int(SPLIT_CFG["min_test_scam_rows"])
    if test_scam < min_rows:
        print(f"WARNING: test_a has only {test_scam} scam rows "
              f"(wanted >= {min_rows}). Collect more real scam messages before "
              f"trusting any evaluation number.")
    for s in ("train", "val", "test_a"):
        if counts.get(s, 0) == 0:
            print(f"WARNING: split '{s}' is EMPTY - your dataset is too small for 3 splits.")
    print("splits: cluster-disjointness check PASSED (no cluster spans splits).")
    print("splits: note - src/splits2.py builds the full five-way "
          "(train/val/test_a/test_b/test_real) family-aware split used in Phase 2.")
    return df


if __name__ == "__main__":
    # Tiny self-test so beginners can run this file alone.
    from src.make_synthetic import build_synthetic

    print(f"(seed={SEED}) src/splits.py self-test")
    print("=" * 60)
    df = pd.DataFrame(build_synthetic())
    df["template_cluster"] = range(len(df))  # pretend every message is unique
    df["template_family"] = ""
    out = split_dataset(df)
    print("\nself-test OK. split counts:", out["split"].value_counts().to_dict())
