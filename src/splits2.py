"""
src/splits2.py - THE Phase 2 splitter: five leakage-safe splits.

  train / val / test_a  -> normal development splits (synthetic allowed)
  test_b                -> UNSEEN template families (the "new variant" test)
  test_real             -> REAL human-labeled messages only (headline numbers)

WHY FAMILY-AWARE?
  Phase 1 split by template_cluster (near-duplicate text). That stops exact
  copies leaking, but two DIFFERENT families of the same scam tactic
  (e.g. "kyc_f1" and "kyc_f7") share wording, so test_a still flatters the
  model. test_b therefore holds out whole template FAMILIES the model has
  never seen - a much more honest measure of "does it catch a new scam?".

THE ONE RULE THAT MATTERS:
  a template_family and a template_cluster may NEVER appear in more than
  one split. We crash loudly (AssertionError) if that ever happens.

HOW A ROW GETS ITS GROUP
  Rows are merged with a tiny union-find: two rows join the same group if
  they share a template_family OR a template_cluster. One group = one
  indivisible chunk that always moves between splits together.

HOW test_b IS PICKED (greedy coverage)
  * Every group gets "coverage keys": the tactic tags it carries, plus
    "verdict:suspicious" / "verdict:safe" so those verdicts also appear in
    test_b (safe rows have no tactics, so tactics alone would miss them).
  * Target = ~25% of groups per key (config: splits2.holdout_families_per_tactic),
    but only when a key has >= 4 groups - we never strip a rare tactic out
    of training entirely.
  * Greedy fill: repeatedly pick the not-yet-held-out group that covers the
    most still-unfilled keys. Seeded (42), so the choice is reproducible.
  * Groups already pinned (test_real, public_weak -> train) are excluded.

HARD RULES (all asserted, not assumed)
  * source=public_weak (English UCI spam) -> TRAIN only, never evaluated.
  * synthetic rows may NEVER sit in test_real.
  * a synthetic row that shares a family/cluster with a test_real row is
    DROPPED from the dataset (a synthetic copy of a real test message would
    leak the answer into training).
  * no template_cluster spans splits; no non-empty template_family spans splits.

Run:  python -m src.splits2   (self-test on a synthetic frame)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CFG, SEED, SPLITS
from src.splits import (
    REAL_EVAL_SOURCES,
    REAL_OWN_SOURCES,
    SYNTHETIC_SOURCES,
    TRAIN_ONLY_SOURCES,
)

SPLIT2_CFG = CFG["splits2"]

# A group needs at least this many members before we dare hold one out,
# otherwise that tactic would vanish from training completely.
MIN_GROUPS_TO_HOLDOUT = 4


# ---------------------------------------------------------------------------
# union-find: rows sharing a family OR a cluster become one indivisible group
# ---------------------------------------------------------------------------
def _find(parent: dict, x):
    """Union-find 'find' with path compression (fancy way of saying:
    follow the chain of pointers until the root, and shorten it on the way)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict, a, b) -> None:
    """Merge the groups containing a and b."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def build_groups(df: pd.DataFrame) -> pd.Series:
    """Return a Series (row index -> group key string).

    Group key = the union-find root of ("F:<family>", "C:<cluster>").
    Every row belongs to exactly one group.
    """
    parent: dict = {}

    def node(tag: str):
        if tag not in parent:
            parent[tag] = tag
        return tag

    row_root = []
    for idx, row in df.iterrows():
        cluster_tag = node(f"C:{int(row['template_cluster'])}")
        family = str(row.get("template_family", "") or "").strip()
        # NaN / "nan" / "None" mean NO family - never treat them as a name,
        # or every unfamiled row would merge into one giant fake family.
        if family.lower() in {"nan", "none", "null"}:
            family = ""
        parts = [cluster_tag]
        if family:
            parts.append(node(f"F:{family}"))
        # merge all parts of this row into one group
        for p in parts[1:]:
            _union(parent, parts[0], p)
        row_root.append((idx, parts[0]))

    # Resolve to final roots (after all the merging above).
    return pd.Series(
        {idx: f"g{_find(parent, root):s}" for idx, root in row_root},
        name="split_group",
    )


# ---------------------------------------------------------------------------
# coverage keys - what each group "covers" for test_b bookkeeping
# ---------------------------------------------------------------------------
def _coverage_keys(group_rows: pd.DataFrame) -> set[str]:
    """The tactic tags + verdict placeholders a group contributes."""
    keys: set[str] = set()
    for tactics in group_rows["tactics"]:
        if isinstance(tactics, (list, tuple, set)):
            keys.update(str(t) for t in tactics)
    verdicts = set(group_rows["verdict"].astype(str))
    # Scam rows are covered through their tactics. safe/suspicious rows carry
    # no tactics, so we track them by verdict instead. A scam group with NO
    # tactics at all still needs a key, otherwise it could never be held out.
    if "suspicious" in verdicts:
        keys.add("verdict:suspicious")
    if "safe" in verdicts:
        keys.add("verdict:safe")
    if "scam" in verdicts and not keys:
        keys.add("verdict:scam")
    # STYLE x VERDICT keys (Phase 2.5): without these, test_b could hold out
    # only english safe families and every devanagari row would still be a
    # scam - so script_style alone would predict the label on the "unseen"
    # split too. Only keys with >=4 groups are actually held out.
    for style in set(group_rows["script_style"].astype(str)):
        for verdict in verdicts:
            if verdict in {"safe", "scam"}:
                keys.add(f"style:{style}|verdict:{verdict}")
    return keys


def _pick_holdout(groups: dict, keys_by_group: dict, pinned: set,
                  pct: float, rng) -> set:
    """Greedy: fill every coverage key's target with as few groups as possible.

    groups       : {group_key: row_count}
    keys_by_group: {group_key: set of coverage keys}
    pinned       : groups we may NOT hold out (test_real / public_weak)
    Returns the set of group keys to place in test_b.
    """
    # target[key] = how many groups holding that key should land in test_b
    groups_by_key: dict[str, list] = {}
    for g, keys in keys_by_group.items():
        for k in keys:
            groups_by_key.setdefault(k, []).append(g)

    targets: dict[str, int] = {}
    for key, members in groups_by_key.items():
        usable = [g for g in members if g not in pinned]
        if len(usable) >= MIN_GROUPS_TO_HOLDOUT:
            targets[key] = max(1, round(pct * len(usable)))
        else:
            targets[key] = 0
            print(f"  holdout: key '{key}' has only {len(usable)} group(s) "
                  f"(<{MIN_GROUPS_TO_HOLDOUT}) -> kept fully in training.")

    held: set = set()
    count = {k: 0 for k in targets}

    # Repeat passes until every target is met (or no progress is possible).
    key_order = list(targets)
    rng.shuffle(key_order)
    for _ in range(50):
        changed = False
        for key in key_order:
            while count.get(key, 0) < targets.get(key, 0):
                candidates = [g for g in groups
                              if g not in held and g not in pinned]
                if not candidates:
                    break
                # score = how many still-unfilled keys this group would cover
                def score(g):
                    return sum(1 for k in keys_by_group.get(g, ())
                               if count.get(k, 0) < targets.get(k, 0))
                best = max(score(g) for g in candidates)
                ties = [g for g in candidates if score(g) == best]
                pick = str(rng.choice(np.array(ties, dtype=object)))
                held.add(pick)
                for k in keys_by_group.get(pick, ()):
                    if k in count:
                        count[k] += 1
                changed = True
        if not changed:
            break
    return held


def _assign_groups_balanced(members: list, sizes: dict, ratios: tuple[float, ...],
                            rng) -> dict:
    """Assign one stratum's groups to train/val/test_a by largest remainder.

    Plain cumulative-ratio assignment (src/splits.py) can give a small
    stratum 7 train / 1 val / 0 test_a groups, which leaves test_a with no
    safe message in that script style - exactly the shortcut this project is
    trying to avoid. Apportioning by GROUP COUNT with largest remainders makes
    an 8-group stratum 6/1/1 and an 11-group stratum 9/1/1, so every writing
    style that has enough families shows up in every dev split.
    """
    members = list(members)
    rng.shuffle(members)
    frac = np.array(ratios, dtype=float)
    frac = frac / frac.sum()
    n = len(members)
    quotas = frac * n
    counts = np.floor(quotas).astype(int)
    # Largest remainders get the leftover seats (ties -> train, then val).
    order = np.argsort(-(quotas - counts), kind="stable")
    for i in order[: n - int(counts.sum())]:
        counts[i] += 1
    out: dict = {}
    pos = 0
    for name, count in zip(("train", "val", "test_a"), counts):
        for g in members[pos:pos + int(count)]:
            out[g] = name
        pos += int(count)
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def split_dataset2(df: pd.DataFrame, enforce_support: bool = False) -> pd.DataFrame:
    """Phase 3.5 entry point: delegate to src.splits35.split_dataset35.

    The old Phase 2.5 body is kept in git history. Phase 3.5 needs real
    train/val support and per-tag family quotas, which live in splits35.
    The default is enforce_support=False so the small unit-test fixtures in
    tests/test_splits2.py can still run (they have <100 rows per tag). The
    REAL build calls src.splits35.split_dataset35 directly from
    src/build_dataset.py, which always enforces the >=100 train / >=1 val /
    >=1 test_b floors.
    """
    from src.splits35 import split_dataset35

    return split_dataset35(df, enforce_support=enforce_support)


if __name__ == "__main__":
    # Tiny self-test so beginners can run this file alone.
    from src.make_synthetic import build_synthetic

    print(f"(seed={SEED}) src/splits2.py self-test")
    print("=" * 60)
    frame = pd.DataFrame(build_synthetic())
    frame["template_cluster"] = range(len(frame))
    frame["split"] = ""
    out = split_dataset2(frame)
    print("\nself-test OK. split counts:", out["split"].value_counts().to_dict())
