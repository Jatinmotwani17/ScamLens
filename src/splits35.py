"""
src/splits35.py - the Phase 3.5 splitter.

WHY A NEW SPLITTER
------------------
Phase 2.5 put every real row in test_real. Phase 3.5 wants REAL external rows
in training too, so those rows arrive already split by cluster
(src/external_data.py -> train / val / test_real). This module must:

  * keep real train/val/test_real assignments exactly as given;
  * never let a real train cluster leak into test_real (drop the sibling);
  * still make train/val/test_a/test_b for synthetic rows;
  * guarantee EVERY tactic tag has:
        >=100 positives in train
        >=1 positive in val
        >=1 positive in test_b
    and keeps >=60% of a tag's synthetic families in train.

test_a is a real 10% hold-out of the TRAIN clusters (Phase 3.6), picked from
scam and safe pools separately so it always carries BOTH classes - a
one-class test_a has no macro-F1 and cannot test memorisation. With only five
families per tag, some rare-tag families must stay in train, so test_a is not
balanced across all 13 tags; we print a per-tag support table so a zero is
visible, never silently averaged away.

HARD CHECKS (all raise AssertionError):
  * no exact text in two splits;
  * no template_cluster or template_family in two splits;
  * no synthetic row in test_real;
  * no real external row in test_a / test_b;
  * no tag below the support floors above.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.config import CFG, REAL_STRATA, SEED, SPLITS, TACTIC_TAGS
from src.data_checks import exact_text_key
from src.splits import (
    REAL_EVAL_SOURCES,
    REAL_OWN_SOURCES,
    SYNTHETIC_SOURCES,
    TRAIN_ONLY_SOURCES,
)
from src.splits2 import build_groups

MIN_TRAIN_POS = int(CFG["splits2"].get("min_train_positives_per_tag", 100))
MIN_TRAIN_FRACTION = float(CFG["splits2"].get("min_train_family_fraction", 0.60))
MIN_TRAIN_FAMILIES = int(CFG["splits2"].get("min_train_families_per_tag", 2))

REAL_SOURCES = set(REAL_EVAL_SOURCES) | set(REAL_OWN_SOURCES)


def _family_key(row) -> str:
    """A real family name, or a cluster-based fallback for unfamiled rows."""
    fam = str(row.get("template_family", "") or "").strip()
    if fam.lower() in {"", "nan", "none", "null"}:
        return f"cluster:{int(row['template_cluster'])}"
    return fam


def _tags_of(sub: pd.DataFrame) -> set[str]:
    tags: set[str] = set()
    for value in sub["tactics"]:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            tags.update(str(t) for t in value)
        elif isinstance(value, str) and value.strip():
            tags.update(p.strip() for p in value.split(";") if p.strip())
    return tags & set(TACTIC_TAGS)


def _fams_with_tag(sub: pd.DataFrame, tag: str) -> set[str]:
    fams = set()
    for _, row in sub.iterrows():
        tags = row["tactics"]
        ok = (tag in tags) if isinstance(tags, (list, tuple, set, np.ndarray)) \
            else (tag in str(tags).split(";"))
        if ok:
            fams.add(_family_key(row))
    return fams


def _role_quotas(n_families: int) -> tuple[int, int, int]:
    """(train, val, test_b) family quotas for one tag.

    test_a is NOT given a quota: it is a 10% hold-out of leftover TRAIN
    clusters, and a group only moves there when every train floor still holds.
    """
    if n_families <= 0:
        return 0, 0, 0
    tb = 1 if n_families >= 4 else 0
    val = 1 if n_families >= 2 else 0
    train = max(MIN_TRAIN_FAMILIES, int(math.ceil(MIN_TRAIN_FRACTION * n_families)))
    # Never ask for more families than exist.
    while train + val + tb > n_families and train > 1:
        train -= 1
    return train, val, tb


def _support_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: tag. Columns: train/val/test_a/test_b/real_test positives."""
    rows = []
    for tag in TACTIC_TAGS:
        counts = {}
        for split in ("train", "val", "test_a", "test_b"):
            sub = df[df["split"] == split]
            counts[split] = int(sum(tag in t for t in sub["tactics"]))
        real = df[df["split"] == "test_real"]
        counts["real_test"] = int(sum(tag in t for t in real["tactics"]))
        rows.append({"tag": tag, **counts})
    return pd.DataFrame(rows)


def _print_support(table: pd.DataFrame) -> None:
    print("\nPER-TAG SUPPORT (rows containing the tag)")
    print(f"{'tag':32s} {'train':>7s} {'val':>6s} {'test_a':>7s} "
          f"{'test_b':>7s} {'real_test':>10s}")
    print("-" * 78)
    for _, r in table.iterrows():
        print(f"{r['tag']:32s} {r['train']:7d} {r['val']:6d} "
              f"{r['test_a']:7d} {r['test_b']:7d} {r['real_test']:10d}")


def _choose_for_role(groups, meta, tag_groups, tag_fams, quota, role,
                     already, role_fams, other_role_fams, other_quota):
    """Greedy helper: pick groups covering tags that still need `role`.

    `role_fams[tag]` counts families already assigned to this role.
    `quota[tag]` is the family quota for this role.
    We only accept a group if every tag it carries still has enough families
    left for the OTHER mandatory roles.
    """
    chosen = []
    for tag in sorted(TACTIC_TAGS, key=lambda t: (len(tag_groups.get(t, [])), t)):
        if role_fams.get(tag, 0) >= quota.get(tag, 0):
            continue
        best = None
        best_score = None
        for g in tag_groups.get(tag, []):
            if g in already:
                continue
            fams = meta[g]["families"]
            # Count families of this tag inside this group.
            fams_t = {f for f in fams if f in tag_fams[tag]}
            if not fams_t:
                continue
            feasible = True
            for t in meta[g]["tags"]:
                left = len(tag_fams[t]) - role_fams.get(t, 0) - len(
                    fams_t if t == tag else {f for f in fams if f in tag_fams[t]})
                need_other = other_quota.get(t, 0)
                if left < need_other:
                    feasible = False
                    break
            if not feasible:
                continue
            covers = sum(1 for t in meta[g]["tags"]
                         if role_fams.get(t, 0) < quota.get(t, 0))
            score = (covers, -meta[g]["size"])
            if best_score is None or score > best_score:
                best_score, best = score, g
        if best is not None:
            chosen.append(best)
            already.add(best)
            for t in meta[best]["tags"]:
                fams_t = {f for f in meta[best]["families"] if f in tag_fams[t]}
                role_fams[t] = role_fams.get(t, 0) + len(fams_t)
    return chosen


def split_dataset35(df: pd.DataFrame, enforce_support: bool = True) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if "template_cluster" not in df.columns or df["template_cluster"].isna().any():
        raise ValueError("splits35: missing template_cluster. Run dedup first.")
    if "template_family" not in df.columns:
        df["template_family"] = ""
    if "split" not in df.columns:
        df["split"] = ""
    df["split"] = df["split"].fillna("").astype(str)

    df["split_group"] = build_groups(df).values
    print(f"splits35: {df['split_group'].nunique()} indivisible group(s)")

    sources = set(df["source"].astype(str))
    synthetic_mask = df["source"].isin(SYNTHETIC_SOURCES)

    # Real rows whose split was not preassigned default to test_real. That is
    # the safe choice: real data evaluates, it does not silently train.
    real_source_mask = df["source"].isin(REAL_SOURCES)
    df.loc[real_source_mask & (df["split"] == ""), "split"] = "test_real"

    # HARD RULE: no synthetic row may be marked test_real.
    bad = df[synthetic_mask & (df["split"] == "test_real")]
    if len(bad):
        raise AssertionError(
            f"HARD RULE VIOLATED: {len(bad)} synthetic row(s) marked test_real. "
            f"ids: {bad['id'].head(10).tolist()}")

    # Groups that contain any test_real row are pinned. Any NON-test row in
    # such a group is a near-copy of a real test message -> drop it loudly.
    test_groups = set(df.loc[df["split"] == "test_real", "split_group"])
    sibling = df["split_group"].isin(test_groups) & (df["split"] != "test_real")
    if sibling.any():
        print(f"splits35: WARNING - dropping {int(sibling.sum())} non-test row(s) "
              "sharing a cluster/family with test_real (leak guard).")
        print("  ids:", df.loc[sibling, "id"].head(10).tolist())
        df = df[~sibling].reset_index(drop=True)
        test_groups = set(df.loc[df["split"] == "test_real", "split_group"])

    # Real train/val groups are preserved, but a group may not be half train
    # and half val.
    real_roles: dict[str, str] = {}
    for g, sub in df[df["source"].isin(REAL_SOURCES)].groupby("split_group"):
        vals = set(sub["split"]) - {"test_real"}
        if len(vals) > 1:
            raise AssertionError(
                f"real group {g} has conflicting splits {sorted(vals)}; "
                "re-cluster or rerun external_data.")
        if vals:
            real_roles[g] = next(iter(vals))

    # public_weak never leaves train. It is not loaded by build_dataset in
    # Phase 3.5, but keep the guard in case an old CSV slips in.
    weak_groups = set(df.loc[df["source"].isin(TRAIN_ONLY_SOURCES), "split_group"])
    for g in weak_groups:
        real_roles[g] = "train"

    synthetic_groups = [
        g for g in df["split_group"].unique()
        if g not in test_groups and g not in real_roles
    ]
    print(f"splits35: synthetic groups={len(synthetic_groups)}, "
          f"real train/val groups={len(real_roles)}, "
          f"test_real groups={len(test_groups)}")

    # ---- metadata for the synthetic groups --------------------------------
    meta: dict[str, dict] = {}
    for g in synthetic_groups:
        sub = df[df["split_group"] == g]
        meta[g] = {
            "tags": _tags_of(sub),
            "families": {_family_key(r) for _, r in sub.iterrows()},
            "size": len(sub),
            "style": str(sub["script_style"].mode().iat[0]),
            "verdict": str(sub["verdict"].mode().iat[0]),
        }

    tag_groups: dict[str, list[str]] = defaultdict(list)
    tag_fams: dict[str, set[str]] = defaultdict(set)
    for g, m in meta.items():
        for tag in m["tags"]:
            tag_groups[tag].append(g)
            tag_fams[tag] |= m["families"]

    quota_train = {t: _role_quotas(len(tag_fams.get(t, set())))[0] for t in TACTIC_TAGS}
    quota_val = {t: _role_quotas(len(tag_fams.get(t, set())))[1] for t in TACTIC_TAGS}
    quota_tb = {t: _role_quotas(len(tag_fams.get(t, set())))[2] for t in TACTIC_TAGS}

    role: dict[str, str] = {}
    val_fams: dict[str, int] = defaultdict(int)
    tb_fams: dict[str, int] = defaultdict(int)
    used: set[str] = set()

    # ---- 1. pick test_b groups (one family per tag where possible) ---------
    tb_groups = _choose_for_role(
        synthetic_groups, meta, tag_groups, tag_fams, quota_tb, "test_b",
        used, tb_fams, val_fams, {t: quota_train[t] + quota_val[t] for t in TACTIC_TAGS})
    for g in tb_groups:
        role[g] = "test_b"
        for t in meta[g]["tags"]:
            tb_fams[t] += len({f for f in meta[g]["families"] if f in tag_fams[t]})

    # ---- 2. pick val groups ------------------------------------------------
    val_groups = _choose_for_role(
        synthetic_groups, meta, tag_groups, tag_fams, quota_val, "val",
        used, val_fams, tb_fams, {t: quota_train[t] for t in TACTIC_TAGS})
    for g in val_groups:
        role[g] = "val"
        for t in meta[g]["tags"]:
            val_fams[t] += len({f for f in meta[g]["families"] if f in tag_fams[t]})

    # ---- 3. every remaining group starts in train -------------------------
    for g in synthetic_groups:
        if g not in role:
            role[g] = "train"

    # ---- 3b. put SAFE rows into test_b and val ----------------------------
    # Safe messages have no tags, so the tag-aware loop above never selects
    # them. Without this step test_b would be 100% scam and macro-F1 there
    # would be meaningless. We keep the move style-aware and leave plenty of
    # safe rows in train.
    def move_safe(target_rows: int, dest: str) -> None:
        # `target_rows` means SAFE rows in this destination, not all rows.
        current = sum(meta[g]["size"] for g in synthetic_groups
                      if role[g] == dest and not meta[g]["tags"])
        need = max(0, target_rows - current)
        if need <= 0:
            return
        candidates = [g for g in synthetic_groups
                      if role[g] == "train" and not meta[g]["tags"]]
        # Prefer one group per style first, then the smallest groups.
        chosen_styles: set[str] = set()
        candidates.sort(key=lambda g: (meta[g]["style"] in chosen_styles,
                                       meta[g]["size"]))
        moved = 0
        for g in candidates:
            if moved >= need:
                break
            role[g] = dest
            moved += meta[g]["size"]
            chosen_styles.add(meta[g]["style"])

    # test_b: ~30% safe rows (at least ~30 rows).
    tb_rows = sum(meta[g]["size"] for g in synthetic_groups if role[g] == "test_b")
    move_safe(max(30, int(0.30 * tb_rows)), "test_b")

    # val: ~20% safe among the synthetic val rows.
    val_rows = sum(meta[g]["size"] for g in synthetic_groups if role[g] == "val")
    move_safe(max(40, int(0.20 * val_rows)), "val")

    # ---- 4. borrow SAFE rows from TRAIN for test_b / val styles -----------
    # Safe messages have no tags, so the tag-aware loop above never picks them
    # and test_b/val would be almost all English scam. Moving whole safe
    # groups into test_b/val gives them a style mix instead of
    # "english = scam, devanagari = safe".
    #
    # Phase 3.6: this used to borrow from the old (safe-only) test_a. test_a is
    # now chosen AFTER this step, straight out of train, so the pool is train.
    def move_safe_from_train(dest: str, style: str, target_rows: int) -> None:
        current = sum(meta[g]["size"] for g in synthetic_groups
                      if role[g] == dest and not meta[g]["tags"]
                      and meta[g]["style"] == style)
        need = max(0, target_rows - current)
        if need <= 0:
            return
        candidates = [g for g in synthetic_groups
                      if role[g] == "train" and not meta[g]["tags"]
                      and meta[g]["style"] == style]
        candidates.sort(key=lambda g: meta[g]["size"])
        moved = 0
        for g in candidates:
            if moved >= need:
                break
            role[g] = dest
            moved += meta[g]["size"]

    for style in ("english", "roman_hinglish"):
        move_safe_from_train("test_b", style, 20)
        move_safe_from_train("val", style, 150)
    move_safe_from_train("val", "devanagari", 80)

    # ---- 5. hold out ~10% of the TRAIN clusters as test_a -----------------
    # WHY THIS CHANGED (Phase 3.6)
    # test_a used to collect only the groups the tag loop did not need, i.e.
    # UNTAGGED (safe) groups. Every scam family was already pinned to
    # train/val/test_b, so test_a ended up 100% safe: one class, so macro-F1
    # was n/a and the "seen-family" comparison in the gap check was empty.
    #
    # Now test_a is a genuine 10% hold-out of the TRAIN split's indivisible
    # groups (one group = a whole near-duplicate cluster / template family).
    # Both classes come along because we pick from scam and safe pools
    # separately. It stays leak-free: a group still lives in exactly ONE split,
    # which the hard checks below re-verify. Floors (>=100 tag positives in
    # train, >=60% of a tag's families in train) are checked before every move,
    # so test_a can never starve a tag out of training.
    tag_rows_by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    size_by_group: dict[str, int] = defaultdict(int)
    for g, tags in zip(df["split_group"], df["tactics"]):
        size_by_group[g] += 1
        if isinstance(tags, (list, tuple, set, np.ndarray)):
            for t in tags:
                if t in TACTIC_TAGS:
                    tag_rows_by_group[g][t] += 1

    train_fams: dict[str, int] = defaultdict(int)
    train_tag_rows: dict[str, int] = defaultdict(int)
    train_rows = train_scam_rows = train_safe_rows = 0
    for g in synthetic_groups:
        if role[g] != "train":
            continue
        train_rows += size_by_group[g]
        if meta[g]["verdict"] == "scam":
            train_scam_rows += size_by_group[g]
        else:
            train_safe_rows += size_by_group[g]
        for t in meta[g]["tags"]:
            train_fams[t] += len({f for f in meta[g]["families"] if f in tag_fams[t]})
            train_tag_rows[t] += tag_rows_by_group[g][t]

    # The >=100 row floor is a PRODUCTION rule. Tiny unit-test frames start
    # below it for every tag, and a floor that can never be met would freeze
    # the splitter; for those tags we fall back to the family quota only and
    # say so loudly. On the real dataset every tag starts well above 100.
    tag_row_floor = {t: (MIN_TRAIN_POS if train_tag_rows[t] >= MIN_TRAIN_POS else 0)
                     for t in TACTIC_TAGS}
    below = [t for t in TACTIC_TAGS
             if tag_fams.get(t) and train_tag_rows[t] < MIN_TRAIN_POS]
    if below:
        print(f"splits35: NOTE - {len(below)} tag(s) start below the {MIN_TRAIN_POS} "
              "train-positive floor (fixture-scale data); family quotas still "
              "apply, but the row floor is not enforced for them.")

    def _can_hold_out(g: str) -> bool:
        """True when moving this group to test_a keeps every TRAIN floor."""
        for t in meta[g]["tags"]:
            fams = {f for f in meta[g]["families"] if f in tag_fams[t]}
            if train_fams[t] - len(fams) < quota_train[t]:
                return False
            if train_tag_rows[t] - tag_rows_by_group[g][t] < tag_row_floor[t]:
                return False
        return True

    def _hold_out(g: str) -> None:
        role[g] = "test_a"
        for t in meta[g]["tags"]:
            train_fams[t] -= len({f for f in meta[g]["families"] if f in tag_fams[t]})
            train_tag_rows[t] -= tag_rows_by_group[g][t]

    # Seeded shuffles: same seed -> same test_a on every run.
    rng = np.random.default_rng(SEED)
    train_groups = [g for g in synthetic_groups if role[g] == "train"]
    scam_pool = [g for g in train_groups if meta[g]["verdict"] == "scam"]
    safe_pool = [g for g in train_groups if meta[g]["verdict"] != "scam"]
    for pool in (scam_pool, safe_pool):
        rng.shuffle(pool)

    target_rows = max(1, int(round(0.10 * train_rows)))
    # Per-class caps keep a style-skewed pool from turning test_a into one class.
    cap_scam = max(1, int(round(0.15 * train_scam_rows)))
    cap_safe = max(1, int(round(0.15 * train_safe_rows)))
    moved_rows = moved_scam = moved_safe = 0
    si = fi = 0
    while moved_rows < target_rows and (si < len(scam_pool) or fi < len(safe_pool)):
        # Alternate classes so a run of big scam groups cannot exhaust the budget.
        g = None
        if si < len(scam_pool) and moved_scam <= moved_safe:
            g, si = scam_pool[si], si + 1
        elif fi < len(safe_pool):
            g, fi = safe_pool[fi], fi + 1
        elif si < len(scam_pool):
            g, si = scam_pool[si], si + 1
        if g is None:
            break
        is_scam = meta[g]["verdict"] == "scam"
        if is_scam and moved_scam >= cap_scam:
            continue
        if not is_scam and moved_safe >= cap_safe:
            continue
        if not _can_hold_out(g):
            continue
        _hold_out(g)
        moved_rows += size_by_group[g]
        if is_scam:
            moved_scam += size_by_group[g]
        else:
            moved_safe += size_by_group[g]

    # ---- 6. write the split column ----------------------------------------
    def final_split(g: str) -> str:
        if g in test_groups:
            return "test_real"
        if g in real_roles:
            return real_roles[g]
        return role.get(g, "train")

    df["split"] = df["split_group"].map(final_split).astype(str)

    # test_a MUST carry both classes (Phase 3.6). A one-class test_a has no
    # macro-F1 and no scam recall, so it cannot answer "did the model memorise
    # seen wording?". Fail loudly instead of publishing a useless split.
    ta = df[df["split"] == "test_a"]
    ta_scam = int((ta["verdict"] == "scam").sum())
    ta_safe = int((ta["verdict"] == "safe").sum())
    print(f"test_a: {len(ta)} rows held out BY CLUSTER from train "
          f"({ta_scam} scam / {ta_safe} safe; target was ~{target_rows} rows)")
    if moved_rows < target_rows:
        print(f"  NOTE: stopped at {moved_rows} of {target_rows} target rows "
              "because the remaining TRAIN groups are needed to keep every "
              "tag's >=60% family share and >=100 train positives.")
    # A fixture-scale frame cannot meet the production tag floors, so its
    # test_a may legitimately be tiny; the real build is required to have both
    # classes. The distinction is explicit so a broken production split can
    # never hide behind a warning.
    production_scale = all(train_tag_rows[t] >= MIN_TRAIN_POS
                           for t in TACTIC_TAGS if tag_fams.get(t))
    if ta_scam == 0 or ta_safe == 0:
        message = ("test_a must contain BOTH classes so its macro-F1 means "
                   f"something; found scam={ta_scam}, safe={ta_safe}. Widen the "
                   "candidate pool or relax the train tag floors before "
                   "splitting.")
        if production_scale:
            raise AssertionError(message)
        print(f"WARNING (fixture-scale frame): {message}")

    # ---- 7. hard checks ----------------------------------------------------
    spans = df.groupby("split_group")["split"].nunique()
    if (spans > 1).any():
        raise AssertionError(f"LEAKAGE: split group spans splits: {spans[spans > 1].index[:5].tolist()}")

    cluster_spans = df.groupby("template_cluster")["split"].nunique()
    if (cluster_spans > 1).any():
        raise AssertionError(
            f"LEAKAGE: {int((cluster_spans > 1).sum())} template_cluster(s) span splits!")

    fam = df[df["template_family"].astype(str).str.strip() != ""]
    fam_spans = fam.groupby("template_family")["split"].nunique()
    if (fam_spans > 1).any():
        raise AssertionError(
            f"LEAKAGE: {int((fam_spans > 1).sum())} template_family(s) span splits!")

    # Exact text must never appear in two splits.
    df["_text_key"] = df["text"].map(exact_text_key)
    text_splits = df.groupby("_text_key")["split"].nunique()
    if (text_splits > 1).any():
        bad_keys = text_splits[text_splits > 1].index[:5].tolist()
        raise AssertionError(f"LEAKAGE: exact text in two splits: {bad_keys}")
    df.drop(columns=["_text_key"], inplace=True)

    # Synthetic vs real placement. Recompute the mask: rows may have been
    # dropped above, which changes the index (avoids a pandas reindex warning).
    synthetic_mask = df["source"].isin(SYNTHETIC_SOURCES)
    syn_real = df[synthetic_mask & (df["split"] == "test_real")]
    if len(syn_real):
        raise AssertionError("synthetic rows in test_real")
    real_bad = df[df["source"].isin(REAL_EVAL_SOURCES) &
                  df["split"].isin(["test_a", "test_b"])]
    if len(real_bad):
        raise AssertionError(
            f"{len(real_bad)} real external row(s) in test_a/test_b!")

    # ---- 8. tag support table + loud failures ------------------------------
    table = _support_table(df)
    _print_support(table)
    failures = []
    for _, r in table.iterrows():
        if r["train"] < MIN_TRAIN_POS:
            failures.append(f"{r['tag']}: train={r['train']} < {MIN_TRAIN_POS}")
        if r["val"] < 1:
            failures.append(f"{r['tag']}: val=0")
        if r["test_b"] < 1:
            failures.append(f"{r['tag']}: test_b=0")
    if failures and enforce_support:
        raise AssertionError("TAG SUPPORT FAILED:\n  - " + "\n  - ".join(failures))
    if failures:
        print("WARNING: tag-support floors not met (enforce_support=False): "
              + "; ".join(failures))

    # ---- 9. summary -------------------------------------------------------
    df.drop(columns=["split_group"], inplace=True)
    print("\nsplits35: rows per split:", df["split"].value_counts().to_dict())
    print("splits35: verdict balance per split:")
    print(pd.crosstab(df["split"], df["verdict"]).reindex(
        [s for s in SPLITS if s in set(df["split"])], fill_value=0).to_string())
    print("splits35: exact-text, cluster, family and real-placement checks PASSED.")
    return df
