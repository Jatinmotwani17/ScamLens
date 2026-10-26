"""
src/build_dataset.py - assemble everything into data/processed/messages.parquet.

This is the glue step between the small CSVs and the single Parquet table
that every later phase reads. It:

  1. loads data/interim/labeled.csv     (your real labels, from label_cli)
     and, if present, data/interim/public_weak.csv
  2. loads the Phase 2 generator output: data/processed/synthetic.parquet
     (+ synthetic_llm.parquet paraphrases when they exist)
     NOTE: the ~63 Phase 1 seed rows now live in tests/fixtures/ and are
     TEST FIXTURES ONLY - deliberately NOT part of training/evaluation.
  3. normalises the `tactics` column into a real list of tag strings
  4. deduplicates EXACT texts (Phase 2.5, normalised: lower/collapse/digits)
  5. clusters near-duplicate templates        (src/dedup.py)
  6. assigns the five-way family-aware splits (src/splits2.py)
  7. labels real rows with `real_stratum` and moves every `suspicious`
     row out to data/processed/suspicious_holdout.parquet (Phase 4 material)
  8. validates the schema loudly              (src/data_checks.py)
  9. writes data/processed/messages.parquet

Run:  python -m src.build_dataset
"""
from __future__ import annotations

import pandas as pd

from src.config import CFG, MESSAGE_COLUMNS, SEED, data_path
from src.data_checks import (
    as_tactics_list,
    binary_table_check,
    normalize_text,
    validate_schema,
)
from src.dedup import cluster_templates
from src.io_utils import load_messages
from src.splits2 import split_dataset2


def parse_tactics(value) -> list[str]:
    """Turn whatever is in a CSV cell into a clean list of tactic tags.

    label_cli writes 'tag1;tag2'; synthetic.csv holds list reprs like
    "['a', 'b']". The shared helper accepts every shape (see its docstring).
    """
    return as_tactics_list(value)


def _load_csv_if_exists(key: str, what: str) -> pd.DataFrame:
    """Load an interim CSV or return an empty frame with a friendly message."""
    path = data_path(key)
    if not path.exists():
        print(f"  no {what} at {path} - skipping (optional)")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    print(f"  loaded {len(df)} {what} row(s) from {path}")
    return df


def _load_parquet_if_exists(key: str, what: str) -> pd.DataFrame:
    """Load a processed parquet or return an empty frame with a friendly note."""
    path = data_path(key)
    if not path.exists():
        print(f"  no {what} at {path} - skipping (optional)")
        return pd.DataFrame()
    df = load_messages(path)
    print(f"  loaded {len(df)} {what} row(s) from {path}")
    return df


# ---------------------------------------------------------------------------
# Phase 2.5 duplicate control
# ---------------------------------------------------------------------------
# When two rows are the same message, which copy do we KEEP? Lower number =
# higher priority. Real test data wins over everything (losing a real row
# shrinks our only honest test set); a machine paraphrase beats the template
# it came from; and the English public spam file loses last of all.
_DEDUPE_PRIORITY = {
    "external_reported": 0,   # REAL reported scams -> test_real
    "public_ham_eval": 0,     # REAL legit messages  -> test_real
    "self_collected": 1,      # YOUR messages        -> test_real
    "family_friends": 1,
    "synthetic_llm": 2,       # Gemini rewrites (more natural wording)
    "synthetic": 3,           # plain generated rows
    "public_weak": 4,         # weakest data, never evaluated
}


def report_unique_ratios(df: pd.DataFrame, title: str) -> dict[str, float]:
    """Print the unique-text ratio per source and per template_family.

    "Unique" uses normalize_text (lowercase, whitespace collapsed, digits
    -> 0), so a re-worded date or amount does not count as a new message.
    Returns {source: ratio} so the caller can enforce the >=95% target.
    """
    print(f"{title} unique-text ratios (normalised):")
    norm = df["text"].map(normalize_text)
    work = df.assign(_norm=norm)

    ratios: dict[str, float] = {}
    print("  per source:")
    for src, sub in work.groupby("source"):
        n, u = len(sub), int(sub["_norm"].nunique())
        r = u / n if n else 0.0
        ratios[str(src)] = r
        print(f"    {str(src):20s} {u:6d}/{n:<6d} = {r:.1%}")

    print("  per template_family (worst 5):")
    fam_stats = []
    for fam, sub in work.groupby(df["template_family"].astype(str).str.strip()):
        if len(sub) < 2:
            continue
        n, u = len(sub), int(sub["_norm"].nunique())
        fam_stats.append((u / n, u, n, fam or "(no family)"))
    fam_stats.sort()
    if not fam_stats:
        print("    (no family has 2+ rows)")
    for r, u, n, fam in fam_stats[:5]:
        print(f"    {fam:34s} {u:5d}/{n:<5d} = {r:.1%}")
    return ratios


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ONE row per normalised text (Phase 2.5 requirement #1).

    Duplicates are dropped BEFORE clustering/splitting, so the same message
    can never be trained on and then tested on (or counted twice). When a
    duplicate exists in several sources we keep the highest-priority copy
    (see _DEDUPE_PRIORITY): the real one, never the synthetic one.
    """
    work = df.copy()
    work["_norm"] = work["text"].map(normalize_text)
    work["_pri"] = work["source"].map(_DEDUPE_PRIORITY).fillna(9)
    # sort by priority, then keep the FIRST row of each normalised text.
    # stable sort -> tie-break follows input order, so the run is reproducible.
    order = work.sort_values(["_pri"], kind="stable").index
    seen: set[str] = set()
    keep: list = []
    for idx in order:
        key = work.at[idx, "_norm"]
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    n_before = len(work)
    out = df.loc[keep].copy()
    dropped = n_before - len(out)
    if dropped:
        # say HOW MANY we lost per source, so nothing disappears silently
        kept_mask = df.index.isin(keep)
        lost = df.loc[~kept_mask, "source"].value_counts().to_dict()
        print(f"  dedupe: dropped {dropped} exact duplicate row(s) "
              f"(kept the highest-priority copy). By source: {lost}")
    else:
        print("  dedupe: no exact duplicates found")
    return out


def assign_real_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every REAL row with the segment it belongs to (Phase 2.5 #3).

    test_real is never reported as one blended number, so each row says which
    bucket it came from. Non-real rows get "" (empty string = not applicable).
    """
    df = df.copy()
    df["real_stratum"] = ""                # default: not a real test row
    real = df["split"] == "test_real"
    own = df["source"].isin(["self_collected", "family_friends"])
    df.loc[real & own & (df["verdict"] == "scam"), "real_stratum"] = "user_real_scam"
    df.loc[real & own & (df["verdict"] == "safe"), "real_stratum"] = "user_real_safe"
    df.loc[real & (df["source"] == "external_reported"), "real_stratum"] = "external_scam"
    df.loc[real & (df["source"] == "public_ham_eval"), "real_stratum"] = "public_ham_english"
    counts = df.loc[df["real_stratum"] != "", "real_stratum"].value_counts()
    print("  real_stratum counts:", counts.to_dict())
    return df


def split_off_suspicious(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 2.5 #2: 'suspicious' is an UNSURE BAND, not a class.

    Returns (main_binary_table, suspicious_holdout). The holdout keeps its
    split/stratum so Phase 4 can test confidence-based abstention on it.
    The main table only ever contains scam + safe.
    """
    sus = df[df["verdict"] == "suspicious"].copy()
    main = df[df["verdict"] != "suspicious"].copy()
    if not sus.empty:
        sus["holdout_reason"] = "unsure_band_phase4"
        print(f"  moved {len(sus)} 'suspicious' row(s) -> "
              f"{data_path('suspicious_holdout').name} (Phase 4 abstention)")
    return main, sus


def build() -> pd.DataFrame:
    print(f"(seed={SEED}) src/build_dataset.py")
    print("=" * 60)
    print("step 1/9: loading CSVs + generator parquet...")
    labeled = _load_csv_if_exists("labeled", "labeled row")
    # Phase 3.5: public_weak commercial SPAM is no longer part of the table.
    # The legitimate UCI ham rows we DO want are reserved as real safe data by
    # src/external_data.py (source=public_ham_eval), so loading public_weak
    # here would duplicate them and drag spam back in.
    public = pd.DataFrame()
    print("  public_weak: NOT loaded (spam excluded; real ham comes from "
          "external_data as public_ham_eval)")
    external = _load_csv_if_exists("external_rows", "external real row")
    synthetic = _load_parquet_if_exists("synthetic_parquet", "synthetic row")
    llm = _load_parquet_if_exists("synthetic_llm_parquet", "synthetic_llm row")
    # Phase 5: a SECOND safe generator, written to fix the false-alarm problem.
    # It contributes legitimate bank / OTP / courier / chat messages that use
    # scam-adjacent vocabulary (hard negatives). Optional: an older checkout
    # simply skips it.
    safe_v2 = _load_parquet_if_exists("synthetic_safe_v2_parquet",
                                      "synthetic safe_v2 row")
    print("  (Phase 1 seed rows live in tests/fixtures/ - tests only, never training)")

    frames = [f for f in (labeled, public, external, synthetic, llm, safe_v2)
              if not f.empty]
    if not frames:
        raise SystemExit(
            "Nothing to build. Label some messages first (python -m src.label_cli), "
            "generate the corpus (python -m src.synth.generate), and/or fetch real "
            "test rows (python -m src.external_data)."
        )
    df = pd.concat(frames, ignore_index=True)
    print(f"  combined rows: {len(df)} (labeled={len(labeled)}, "
          f"public_weak={len(public)}, external={len(external)}, "
          f"synthetic={len(synthetic)}, llm={len(llm)}, "
          f"safe_v2={len(safe_v2)})")
    print(f"  real rows preassigned by external_data: "
          f"{df.loc[df['source'].isin(['external_reported', 'public_ham_eval']), 'split'].value_counts().to_dict()}")

    # Loud guard: ids must be unique or the parquet contract is broken.
    dup_ids = int(df["id"].duplicated().sum())
    if dup_ids:
        raise AssertionError(f"{dup_ids} duplicate id(s) across inputs - fix before splitting.")

    # Older CSVs (labeled/public_weak) have no template_family / split columns;
    # after concat pandas fills those with NaN. Normalise to honest empty
    # strings so the splitter never mistakes NaN for a family named "nan".
    if "template_family" not in df.columns:
        df["template_family"] = ""
    df["template_family"] = df["template_family"].fillna("").astype(str).str.strip()
    if "split" not in df.columns:
        df["split"] = ""
    df["split"] = df["split"].fillna("").astype(str)

    # ---- LEAKAGE GUARD ------------------------------------------------------
    # If a train-side row has the EXACT same text as a test_real row (this
    # happens when public_weak and the ham reserve come from the same public
    # collection), the model could simply memorise the test message. We drop
    # the train copy and keep the real one - loudly.
    real_texts = set(df.loc[df["split"] == "test_real", "text"])
    overlap = df["text"].isin(real_texts) & (df["split"] != "test_real")
    n_overlap = int(overlap.sum())
    if n_overlap:
        print(f"  LEAKAGE GUARD: dropping {n_overlap} train-side row(s) whose "
              "text also appears in test_real (kept the real copy)")
        df = df[~overlap].copy()

    print("step 2/9: normalising tactics into lists...")
    df["tactics"] = df["tactics"].map(parse_tactics)

    # Empty date cells -> None so the parquet column is honestly nullable.
    df["date_received"] = df["date_received"].map(
        lambda v: None if str(v).strip() in {"", "nan", "None"} else str(v).strip()
    )
    # had_phone_number arrives as 'True'/'False' strings from CSV.
    df["had_phone_number"] = df["had_phone_number"].astype(str).str.lower() == "true"

    # ---- Phase 2.5 #1: diversity report + EXACT duplicate removal --------
    # Both happen BEFORE clustering/splitting, so the same message can never
    # end up in training and in a test split at the same time.
    print("step 3/9: unique-text ratios + dropping exact duplicates...")
    ratios = report_unique_ratios(df, "BEFORE dedupe")
    min_ratio = float(CFG["diversity"]["min_unique_ratio"])
    syn_ratio = min((r for s, r in ratios.items()
                     if str(s).startswith("synthetic")), default=1.0)
    if syn_ratio < min_ratio:
        # Phase 2.5 makes this a HARD failure. A repetitive generator makes
        # every later score fragile, so silently continuing would be dishonest.
        raise SystemExit(
            f"DIVERSITY CHECK FAILED: synthetic unique ratio {syn_ratio:.1%} "
            f"< {min_ratio:.0%}. Expand src/synth/components.py pools/families "
            "and rerun: python -m src.synth.generate"
        )
    print(f"  diversity target: synthetic source(s) >= {min_ratio:.0%} unique "
          f"(lowest is {syn_ratio:.1%}) - PASS")
    df = drop_exact_duplicates(df)

    print("step 4/9: clustering near-duplicate templates (dedup)...")
    df = cluster_templates(df)

    print("step 5/9: assigning five-way family-aware splits (splits2)...")
    df = split_dataset2(df)

    print("step 6/9: tagging real rows with real_stratum...")
    df = assign_real_stratum(df)

    print("step 7/9: splitting off the 'suspicious' unsure band...")
    df, suspicious = split_off_suspicious(df)

    print("step 8/9: validating schema and saving parquet...")
    # Keep exactly the contract columns, in contract order.
    df = df[MESSAGE_COLUMNS].copy()
    df["template_cluster"] = df["template_cluster"].astype(int)
    validate_schema(df)

    # HARD RULE (Phase 2.5): the main table is BINARY - scam/safe only.
    problems = binary_table_check(df)
    if problems:
        raise AssertionError("BINARY TABLE CHECK FAILED:\n  - " + "\n  - ".join(problems))

    out = data_path("messages_parquet")
    df.to_parquet(out, index=False)
    print(f"\nWROTE {out}")
    print(f"  rows: {len(df)}")
    print(f"  split counts: {df['split'].value_counts().to_dict()}")
    print(f"  verdict counts: {df['verdict'].value_counts().to_dict()}")

    print("step 9/9: saving the suspicious unsure band (Phase 4 material)...")
    hold_out = data_path("suspicious_holdout")
    if suspicious.empty:
        print("  no suspicious rows to park")
    else:
        suspicious = suspicious[MESSAGE_COLUMNS + ["holdout_reason"]].copy()
        suspicious["template_cluster"] = suspicious["template_cluster"].astype(int)
        suspicious.to_parquet(hold_out, index=False)
        print(f"  WROTE {hold_out} ({len(suspicious)} rows)")
        print("  NOTE: these are NEVER trained on and NEVER evaluated in "
              "Phase 2 - Phase 4 uses them for confidence-based abstention.")
    report_unique_ratios(df, "AFTER dedupe")
    print("\nNext: python -m src.data_checks   (read the report!)")
    return df


if __name__ == "__main__":
    build()
