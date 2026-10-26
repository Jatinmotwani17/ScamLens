"""
src/nn_sim.py - nearest-neighbour leakage audit (Phase 3.6).

THE QUESTION
  For every row in val / test_a / test_b / test_real: how similar is it to the
  MOST similar TRAIN row? If a test message has a near-copy in training, its
  score partly measures memorisation, not generalisation.

HOW IT IS MEASURED
  * TF-IDF over character n-grams (char_wb 3-5), fitted on the TRAIN split
    ONLY. Fitting the vocabulary on train and never on the evaluated rows
    keeps the audit itself leak-free.
  * Cosine similarity = dot product of two L2-normalised TF-IDF rows.
  * Sparse maths in chunks (default 256 evaluated rows at a time), so the
    12k x 12k similarity matrix is never materialised in RAM.

OUTPUT
  results/nn_sim.parquet with exactly these columns:
      id              - the message id from data/processed/messages.parquet
      split           - train is NOT included; val/test_a/test_b/test_real
      stratum         - the bucket used in RESULTS.md:
                          val -> val_synthetic / val_real
                          test_a, test_b -> the split name
                          test_real -> its real_stratum
      max_train_sim   - max cosine similarity to any TRAIN row, in [0, 1]

  The per-stratum shares (sim > 0.9, 0.7-0.9, < 0.7) are printed here and
  re-read by src/evaluate.py, which writes them into RESULTS.md.

Run:  python -m src.nn_sim
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import SEED, data_path
from src.io_utils import load_messages

# Evaluated rows are scored this many at a time. Bigger = faster but more RAM.
CHUNK_ROWS = 256

# Character n-grams catch re-wording, typos and inserted emoji, which word
# n-grams would miss. 3-5 is the same family the TF-IDF baselines use.
NGRAM_RANGE = (3, 5)
MIN_DF = 2

SIM_HIGH = 0.9      # "near-copy"
SIM_MID = 0.7       # "same template, re-worded"
SYNTHETIC_SOURCES = ("synthetic", "synthetic_llm")


def add_stratum(df: pd.DataFrame) -> pd.Series:
    """Return the report-friendly stratum for each row.

    val is split into its synthetic and real parts (that is how RESULTS.md
    already reports it), test_a/test_b keep their split name, and a real test
    row uses its real_stratum. Nothing is blended.
    """
    split = df["split"].astype(str)
    stratum = split.copy()

    is_val = split == "val"
    is_syn = df["source"].isin(SYNTHETIC_SOURCES)
    stratum[is_val & is_syn] = "val_synthetic"
    stratum[is_val & ~is_syn] = "val_real"

    is_real = split == "test_real"
    real_labels = df["real_stratum"].astype(str) if "real_stratum" in df.columns \
        else pd.Series([""] * len(df), index=df.index)
    # A test_real row without a stratum keeps a readable fallback instead of "".
    stratum[is_real] = real_labels[is_real].where(
        real_labels[is_real].str.len() > 0, "test_real_unstratified")
    return stratum


def max_similarity_to_train(train_texts: list[str], eval_texts: list[str],
                            chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """Max cosine similarity of every eval row to any train row.

    Both matrices are L2-normalised by TfidfVectorizer itself, so the dot
    product IS the cosine. We multiply in chunks and keep only the row max.
    """
    if not train_texts:
        raise SystemExit("nn_sim: the TRAIN split is empty - run build_dataset")
    if not eval_texts:
        return np.zeros(0, dtype=np.float32)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    eval_matrix = vectorizer.transform(eval_texts)
    print(f"  tf-idf: {train_matrix.shape[1]:,} char n-gram features "
          f"(train rows={train_matrix.shape[0]:,}, eval rows={eval_matrix.shape[0]:,})")

    out = np.zeros(eval_matrix.shape[0], dtype=np.float32)
    for start in range(0, eval_matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, eval_matrix.shape[0])
        # Sparse x sparse -> sparse sims; each row is one block of scores.
        sims = (eval_matrix[start:stop] @ train_matrix.T).tocsr()
        out[start:stop] = np.asarray(sims.max(axis=1).toarray()).ravel()
    return out


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    """Per-stratum share of rows in each similarity band (for the console)."""
    rows = []
    for stratum, sub in sim.groupby("stratum", sort=False):
        n = len(sub)
        s = sub["max_train_sim"].astype(float)
        high = int((s > SIM_HIGH).sum())
        mid = int(((s >= SIM_MID) & (s <= SIM_HIGH)).sum())
        low = int((s < SIM_MID).sum())
        rows.append({
            "stratum": stratum,
            "n_rows": n,
            "share_gt_0.9": high / n if n else 0.0,
            "share_0.7_0.9": mid / n if n else 0.0,
            "share_lt_0.7": low / n if n else 0.0,
            "median_sim": float(s.median()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    print(f"(seed={SEED}) src/nn_sim.py - nearest-neighbour leakage audit")
    print("=" * 60)
    df = load_messages()
    if df["split"].eq("train").sum() == 0:
        raise SystemExit("nn_sim: no train rows - run python -m src.build_dataset")

    df = df.reset_index(drop=True).copy()
    df["stratum"] = add_stratum(df)

    train = df[df["split"] == "train"]
    evaluable = df[df["split"] != "train"].reset_index(drop=True)
    print(f"  train rows: {len(train):,} | val/test rows: {len(evaluable):,}")

    sims = max_similarity_to_train(
        train["text"].astype(str).tolist(),
        evaluable["text"].astype(str).tolist(),
    )
    out = pd.DataFrame({
        "id": evaluable["id"].astype(str).values,
        "split": evaluable["split"].astype(str).values,
        "stratum": evaluable["stratum"].astype(str).values,
        "max_train_sim": sims,
    })
    # Round only for readability of the file; the summary uses the same values.
    out["max_train_sim"] = out["max_train_sim"].round(6)

    path = data_path("nn_sim")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"\nWROTE {path} ({len(out):,} rows)")

    table = summarise(out)
    print(f"\n{'stratum':26s} {'rows':>7s} {'>0.9':>8s} {'0.7-0.9':>9s} "
          f"{'<0.7':>8s} {'median':>7s}")
    print("-" * 72)
    for _, r in table.iterrows():
        print(f"{r['stratum']:26s} {r['n_rows']:7d} {r['share_gt_0.9']:8.1%} "
              f"{r['share_0.7_0.9']:9.1%} {r['share_lt_0.7']:8.1%} "
              f"{r['median_sim']:7.3f}")

    real = out[out["split"] == "test_real"]
    if len(real):
        share = float((real["max_train_sim"] > SIM_HIGH).mean())
        print(f"\nREAL TEST: {share:.1%} of {len(real):,} rows have a train "
              f"near-copy (sim > {SIM_HIGH})")
        if share > 0.5:
            print("WARNING: over half of real test has sim > 0.9, so real-test "
                  "scores mostly measure near-copy matching, not scam detection.")
    print("\nNext: python -m src.evaluate (writes the table into RESULTS.md)")
    return out


if __name__ == "__main__":
    build()
