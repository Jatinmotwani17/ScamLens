"""
src/dedup.py - cluster near-duplicate messages into "templates".

WHY: scammers send the same template to thousands of people with tiny edits
("Dear <NAME>" vs "Dear user"). If two versions of one template end up in
different splits (train and test), the model can CHEAT by memorising the
template - and your test score becomes a lie. So we cluster first, and
src/splits.py keeps whole clusters inside one split.

HOW (4 steps, all explained):
  1. NORMALISE: lowercase, replace every digit run with "0", squeeze spaces.
     "Send Rs 123 to 98765" and "Send Rs 999 to 98765" become identical.
  2. SHINGLE: cut the text into character 5-grams (every window of 5 chars).
  3. MINHASH + LSH: datasketch finds candidate pairs that are probably
     similar (Jaccard >= 0.7) without comparing all N^2 pairs.
  4. CONNECTED COMPONENTS: any pair that really is >= 0.7 similar gets
     merged into one cluster; clusters get integer ids 0, 1, 2, ...

Run:  python -m src.dedup                 (report on data/processed/messages.parquet)
      python -m src.dedup path/to/file.csv (or any csv/parquet with a 'text' column)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

from src.config import CFG, SEED, data_path

DEDUP_CFG = CFG["dedup"]


class UnionFind:
    """Tiny union-find (disjoint set) - groups rows into clusters.

    datasketch 2.0 removed its UnionFind, and the algorithm is ~15 lines, so we
    keep our own beginner-readable version here. union(a, b) means "a and b are
    the same template"; find(x) returns the group's representative id.
    """

    def __init__(self) -> None:
        self.parent: dict = {}

    def add(self, x) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        self.add(x)
        # Path compression: make every node point straight at the root, which
        # keeps later finds fast.
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalize_text(text: str) -> str:
    """Lowercase, replace digit runs with '0', collapse whitespace.

    Digits becoming '0' is what makes "Rs 123" and "Rs 999" look identical -
    that is exactly what we want when hunting for message TEMPLATES.
    """
    import re

    text = str(text).lower()
    text = re.sub(r"\d+", "0", text)      # all digit runs -> single 0
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-gram shingles of a normalised text.

    Example (k=3, "abcd") -> {"abc", "bcd"}. Short texts get one shingle
    (the whole string) so they are not silently ignored.
    """
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _make_minhash(shingle_set: set[str], num_perm: int) -> MinHash:
    """Build one MinHash sketch for one message (seeded for reproducibility)."""
    m = MinHash(num_perm=num_perm, seed=SEED)
    for s in shingle_set:
        m.update(s.encode("utf-8"))
    return m


def cluster_templates(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float | None = None,
    num_perm: int | None = None,
    k: int | None = None,
) -> pd.DataFrame:
    """Add an integer `template_cluster` column to a copy of df.

    Rows whose normalised text is similar (Jaccard >= threshold on their
    5-gram shingles) end up in the same cluster. Every row gets a cluster id,
    even singletons (they form their own cluster of size 1).
    """
    threshold = threshold if threshold is not None else float(DEDUP_CFG["lsh_threshold"])
    num_perm = num_perm if num_perm is not None else int(DEDUP_CFG["num_perm"])
    k = k if k is not None else int(DEDUP_CFG["shingle_size"])

    df = df.copy().reset_index(drop=True)
    texts = df[text_col].astype(str).tolist()
    norms = [normalize_text(t) for t in texts]
    shingle_sets = [shingles(n, k=k) for n in norms]

    # Step 3a: build the LSH index (candidate pairs only).
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sketches: list[MinHash] = []
    for i, sh in enumerate(shingle_sets):
        m = _make_minhash(sh, num_perm)
        sketches.append(m)
        # Empty shingle set (empty text) cannot be indexed; give it a dummy.
        lsh.insert(str(i), m if sh else _make_minhash({"<empty>"}, num_perm))

    # Step 3b: for every row, ask LSH for candidates and verify with the real
    # Jaccard similarity. Verified pairs are unioned into components.
    uf = UnionFind()
    n_pairs = 0
    for i, m in enumerate(sketches):
        for key in lsh.query(m):
            j = int(key)
            if j <= i:
                continue  # each pair once
            sim = m.jaccard(sketches[j])
            if sim >= threshold:
                uf.union(i, j)
                n_pairs += 1

    # Step 4: turn components into dense integer ids 0..K-1, ordered by each
    # component's smallest row index (stable and reproducible).
    root_to_label: dict = {}
    labels: list[int] = []
    for i in range(len(df)):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])

    df["template_cluster"] = labels
    print(f"dedup: {len(df)} rows -> {df['template_cluster'].nunique()} clusters "
          f"(threshold={threshold}, k={k}, num_perm={num_perm}, "
          f"similar pairs merged={n_pairs})")
    return df


def cluster_report(df: pd.DataFrame, top: int = 5) -> None:
    """Print the biggest clusters with an example each, plus size histogram."""
    sizes = df["template_cluster"].value_counts()
    print(f"\nCLUSTER REPORT: {len(sizes)} clusters over {len(df)} rows")
    print("size distribution:",
          {int(s): int(c) for s, c in sizes.value_counts().sort_index().items()},
          "  (cluster size -> how many clusters have that size)")

    print(f"\nTop {top} biggest clusters:")
    for cid, size in sizes.head(top).items():
        members = df[df["template_cluster"] == cid]
        example = str(members.iloc[0]["text"])[:90].replace("\n", " ")
        # rapidfuzz sanity check: how similar are members to the first member?
        ratios = [fuzz.ratio(str(members.iloc[0]["text"]), str(t)) for t in members["text"]]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        print(f"  cluster {cid:>4}  size={size:>4}  mean fuzz-vs-example={mean_ratio:5.1f}%")
        print(f"      e.g. {example!r}")


def _load_any(path: Path) -> pd.DataFrame:
    """Load a .parquet or .csv file into a DataFrame."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str).fillna("")


if __name__ == "__main__":
    print(f"(seed={SEED}) src/dedup.py")
    print("=" * 60)
    # Default target is the processed parquet; any file with a text column works.
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else data_path("messages_parquet")
    if not target.exists():
        # Fall back to the interim pool so beginners can try it early.
        target = data_path("unlabeled_pool")
    if not target.exists():
        print("Nothing to cluster yet. Run ingest/label/build_dataset first.")
        sys.exit(0)
    print("Clustering:", target)
    df = _load_any(target)
    clustered = cluster_templates(df)
    cluster_report(clustered)
