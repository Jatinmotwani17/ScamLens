"""
src/external_data.py - REAL reported messages for test_real (Phase 2).

WHY THIS EXISTS
  Synthetic rows can never prove a model works. This module pulls REAL
  human-reported messages from two public, license-clear datasets and pins
  them into the test_real split:

  1. IMC25 Smishing Dataset (33,869 REAL user-reported smishing SMS)
     - paper: "Fishing for Smishing", IMC 2025 (ACM)
     - license: CC BY 4.0 (see LICENSE.txt in the repo)
     - source: https://github.com/reportsmishing/Smishing-Dataset-IMC25
     - texts arrive ALREADY placeholder-redacted (<URL>, <PHONE_NUMBER>...)
  2. UCI SMS Spam Collection - a small seeded reserve of REAL ham (legit)
     messages so test_real is not 100% scam. English and casual -> EASY
     negatives. We say so loudly: scores on them flatter the model.

HONESTY RULES
  * source="external_reported"  -> real reported scams (IMC25)
  * source="public_ham_eval"    -> real legit messages (UCI ham reserve)
  * neither is machine-generated, but NEITHER carries human-verified tactic
    tags from US. Tactic tags here are COARSE mappings from the IMC25
    authors' own labels (scam_type + lure_principles) - documented below.
  * rows whose scam_type is 'spam' or 'wrong number' are DROPPED: they are
    not clear fraud and we will not pretend otherwise.

Output: data/interim/external_real.csv   (picked up by src/build_dataset.py,
        which passes it through the splitter -> pinned to test_real)

Run:  python -m src.external_data          (downloads on first run, ~9 MB)
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG, MESSAGE_COLUMNS, SEED, TACTIC_TAGS, data_path
from src.ingest import detect_script_style
from src.public_data import load_sms_spam
from src.scrub import scrub

IMC25_URL = ("https://raw.githubusercontent.com/reportsmishing/"
             "Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv")

# IMC25 scam_type  ->  our tactic tags (coarse but honest; "" = no tag)
SCAM_TYPE_TO_TAGS = {
    "banking":    ["bank_kyc_impersonation"],
    "delivery":   ["parcel_courier_scam"],
    "government": ["authority_impersonation"],
    "telecom":    ["utility_disconnection_scam"],
    "hey mum/dad": ["relative_in_trouble"],
    "others":     [],
}
# IMC25 lure_principles  ->  our tactic tags. Only the two unambiguous ones
# are mapped; 'distraction', 'need and greed', 'kindness', 'herd' and
# 'dishonesty' are psychological lures that do NOT equal one of our tags,
# so we leave them in notes instead of inventing labels.
LURE_TO_TAGS = {
    "time/urgency": "urgency_threat",
    "authority": "authority_impersonation",
}

# scam_type values that are NOT clear fraud -> rows are dropped.
DROP_SCAM_TYPES = {"spam", "wrong number"}

# The only date shape our schema accepts (see src/data_checks.py).
DATE_OK_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def download_if_missing(path: Path, url: str) -> bool:
    """Download url -> path if path does not exist yet. Returns True if fresh."""
    if path.exists():
        print(f"  using cached {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, path)          # noqa: S310 - fixed https URL
    print(f"  saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return True


def load_imc25(limit: int | None = None) -> pd.DataFrame:
    """Load the IMC25 smishing CSV and convert it to OUR schema rows.

    Text choice (documented so nobody wonders later):
      * English messages   -> original text
      * other languages    -> the authors' English translation (when present)
      * no translation     -> row is dropped (we cannot evaluate what we cannot read)
    Every row: verdict='scam', source='external_reported', split='test_real'.
    """
    path = Path(data_path("external_dir")) / "imc25_smishing.csv"
    download_if_missing(path, IMC25_URL)
    raw = pd.read_csv(path)
    print(f"  IMC25 raw rows: {len(raw)}")

    # ---- decide which rows we can actually evaluate ------------------------
    keep = []
    dropped_lang = dropped_type = 0
    for _, r in raw.iterrows():
        scam_type = str(r.get("scam_type", "")).strip().lower()
        if scam_type in DROP_SCAM_TYPES:
            dropped_type += 1
            continue
        lang = str(r.get("language", "")).strip()
        # CAREFUL: NaN is truthy in Python, so `x or ""` does NOT turn NaN
        # into "" - it becomes the literal text "nan" later. Use pd.isna.
        trans = r.get("translation", "")
        has_translation = not pd.isna(trans) and bool(str(trans).strip())
        if lang == "English" or has_translation:
            keep.append(r)
        else:
            dropped_lang += 1
    print(f"  dropped {dropped_type} row(s) with scam_type in {sorted(DROP_SCAM_TYPES)}")
    print(f"  dropped {dropped_lang} row(s) with no readable (English) text")
    df = pd.DataFrame(keep)
    if limit:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(df), size=min(limit, len(df)), replace=False)
        df = df.iloc[sorted(idx)]
        print(f"  limited to {len(df)} rows (seeded sample, seed={SEED})")

    # ---- convert each row to our schema ------------------------------------
    rows = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        lang = str(getattr(r, "language", "")).strip()
        text_val = getattr(r, "text", "")
        text = "" if pd.isna(text_val) else str(text_val)
        if lang != "English":
            trans_val = getattr(r, "translation", "")
            trans = "" if pd.isna(trans_val) else str(trans_val).strip()
            if trans:
                text = trans

        # Coarse tactic tags from the IMC25 authors' own labels.
        tags: list[str] = list(SCAM_TYPE_TO_TAGS.get(
            str(getattr(r, "scam_type", "")).strip().lower(), []))
        for lure in str(getattr(r, "lure_principles", "") or "").split(","):
            tag = LURE_TO_TAGS.get(lure.strip().lower())
            if tag:
                tags.append(tag)
        if str(getattr(r, "url_shortener", "") or "").strip():
            tags.append("suspicious_link")
        tags = sorted({t for t in tags if t in TACTIC_TAGS})

        clean, meta = scrub(text)                  # extra safety pass
        # Some reports are pure placeholders/metadata: after scrubbing, the
        # text can end up EMPTY, or reduce to junk like the literal "nan".
        # Such rows teach us nothing -> drop them.
        if clean.strip().lower() in {"", "nan", "none", "null"}:
            continue
        time_raw = str(getattr(r, "time", "") or "").strip()
        date = time_raw if DATE_OK_RE.match(time_raw) else ""

        notes = (f"IMC25; scam_type={getattr(r, 'scam_type', '')}; "
                 f"lures={getattr(r, 'lure_principles', '')}; orig_lang={lang or '?'}")
        rows.append({
            "id": f"imc{i:05d}",
            "text": clean,
            "verdict": "scam",
            "tactics": tags,
            "channel": "sms",
            "script_style": detect_script_style(clean),
            "date_received": date,
            "source": "external_reported",
            "template_family": "",
            "labeler": "imc25_dataset_labels",
            "label_confidence": "medium",   # coarse tags; verdict is high
            "url_type": meta["url_type"],
            "had_phone_number": bool(meta["had_phone_number"]),
            "template_cluster": -1,         # filled by dedup later
            "split": "test_real",           # PINNED: real evaluation only
            "notes": notes,
        })
    out = pd.DataFrame(rows, columns=MESSAGE_COLUMNS)
    print(f"  -> {len(out)} real smishing row(s), all pinned to test_real")
    return out


def reserve_uci_ham(n: int = 400) -> pd.DataFrame:
    """A small seeded reserve of REAL legit (ham) messages for test_real.

    WHY: test_real would otherwise be 100% scam and could not measure how
    many legit messages the model falsely flags. UCI ham is real human text
    but English and casual - EASY negatives. Never compare scores on these
    with scores on real Indian scam SMS.
    """
    raw = load_sms_spam()          # full UCI collection
    if raw.empty:
        print("  no ham reserve (UCI download failed) - test_real stays scam-only")
        return pd.DataFrame(columns=MESSAGE_COLUMNS)

    def is_spam(label) -> bool:
        return str(label).strip().lower() in {"1", "spam"}

    ham = raw[~raw["label"].map(is_spam)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ham), size=min(n, len(ham)), replace=False)
    ham = ham.iloc[sorted(idx)]

    rows = []
    for i, r in enumerate(ham.itertuples(index=False), start=1):
        clean, meta = scrub(str(r.text))
        rows.append({
            "id": f"ham{i:05d}",
            "text": clean,
            "verdict": "safe",
            "tactics": [],
            "channel": "sms",
            "script_style": detect_script_style(clean),
            "date_received": "",
            "source": "public_ham_eval",
            "template_family": "",
            "labeler": "uci_ham_reserve",
            "label_confidence": "high",   # ham in this collection is unambiguous
            "url_type": meta["url_type"],
            "had_phone_number": bool(meta["had_phone_number"]),
            "template_cluster": -1,
            "split": "test_real",
            "notes": "UCI SMS Spam Collection ham; English casual chat; EASY negatives",
        })
    out = pd.DataFrame(rows, columns=MESSAGE_COLUMNS)
    print(f"  -> {len(out)} real ham row(s) reserved for test_real")
    return out


# ---------------------------------------------------------------------------
# Phase 3.5: split REAL rows by near-duplicate cluster
# ---------------------------------------------------------------------------
def split_real_rows_by_cluster(df: pd.DataFrame,
                               cap_per_cluster: int | None = None,
                               max_train_rows: int | None = None) -> pd.DataFrame:
    """Assign train / val / test_real to REAL rows BY WHOLE CLUSTER.

    Why by cluster? Two copies of the same smishing template must never sit in
    train and test. We first remove exact duplicates, then cluster near-dupes
    with the project's MinHash settings (Jaccard >= 0.7), and only then move
    whole clusters into splits.

    The train side is capped so one repeated template cannot dominate:
    `cap_per_cluster` rows per cluster, and `max_train_rows` overall.
    Val and test keep every row of their held-out clusters.
    """
    from src.data_checks import normalize_text
    from src.dedup import cluster_templates

    if df.empty:
        return df
    out = df.copy().reset_index(drop=True)

    # Exact duplicates first: same normalised text = same message.
    out["_norm"] = out["text"].map(normalize_text)
    before = len(out)
    out = out.drop_duplicates("_norm", keep="first").drop(columns="_norm")
    if len(out) != before:
        print(f"  exact dedupe: {before} -> {len(out)} real row(s)")

    # Near-duplicate clusters. cluster_templates sets template_cluster.
    print(f"  clustering {len(out)} real rows (MinHash Jaccard >= 0.70) ...")
    out = cluster_templates(out)
    sizes = out.groupby("template_cluster").size().sort_values(ascending=False)
    print(f"  real clusters: {len(sizes)} (largest has {int(sizes.iloc[0])} rows)")

    cfg = CFG["phase35"]
    ratios = {
        "train": float(cfg["real_train_ratio"]),
        "val": float(cfg["real_val_ratio"]),
        "test_real": float(cfg["real_test_ratio"]),
    }
    total = len(out)
    quotas = {k: v * total for k, v in ratios.items()}
    filled = {k: 0 for k in ratios}
    assignment: dict = {}

    # Assign the biggest clusters first to the split with the most room left.
    # Ties are shuffled with the project seed, so the result is reproducible.
    rng = np.random.default_rng(SEED)
    order = list(sizes.index)
    rng.shuffle(order)
    order.sort(key=lambda c: int(sizes[c]), reverse=True)

    # Make sure val and test_real are never empty: give each one small cluster
    # before the greedy pass (small first, so a giant template is not stolen).
    smallest = sorted(sizes.index, key=lambda c: (int(sizes[c]), str(c)))
    if len(smallest) >= 3:
        for split, cluster in (("test_real", smallest[0]), ("val", smallest[1])):
            assignment[cluster] = split
            filled[split] += int(sizes[cluster])
        order = [c for c in order if c not in assignment]

    for cluster in order:
        # Pick the split furthest below its quota. If all are full, train gets
        # the rest (train is allowed to be larger than its nominal ratio).
        deficits = {k: quotas[k] - filled[k] for k in ratios}
        split = max(deficits, key=deficits.get)
        if deficits[split] <= 0:
            split = "train"
        assignment[cluster] = split
        filled[split] += int(sizes[cluster])

    out["split"] = out["template_cluster"].map(assignment).astype(str)

    # Cap repeated copies in TRAIN only. Val/test keep the held-out clusters
    # whole, because those are evaluation rows.
    if cap_per_cluster is not None:
        train = out[out["split"] == "train"]
        keep = []
        for _, sub in train.groupby("template_cluster"):
            take = min(len(sub), int(cap_per_cluster))
            keep += sub.sample(n=take, random_state=SEED).index.tolist()
        out = out.loc[~out.index.isin(train.index) | out.index.isin(keep)].copy()

    if max_train_rows is not None:
        train = out[out["split"] == "train"]
        if len(train) > int(max_train_rows):
            keep = train.sample(n=int(max_train_rows), random_state=SEED).index
            out = out.loc[~out.index.isin(train.index) | out.index.isin(keep)].copy()

    print("  real split counts:", out["split"].value_counts().to_dict())
    return out.reset_index(drop=True)


def build_external(ham_n: int = 400, limit: int | None = None) -> pd.DataFrame:
    """Combine IMC25 smishing + UCI ham reserve into one schema table."""
    print(f"(seed={SEED}) src/external_data.py")
    print("=" * 60)
    print("step 1/2: IMC25 real reported smishing...")
    scams = load_imc25(limit=limit)
    print("step 2/2: UCI ham reserve (real legit messages)...")
    ham = reserve_uci_ham(n=ham_n)
    frames = [f for f in (scams, ham) if not f.empty]
    if not frames:
        raise SystemExit("No external data could be loaded (offline?).")

    # Phase 3.5: real data is split by cluster before it reaches build_dataset.
    print("\nstep 2b/2: cluster-grouped real train/val/test split...")
    cfg = CFG["phase35"]
    # Ham first, because the 8,000-row cap is a cap on ALL external rows in
    # train (scams + legit), not 8,000 scams plus however much ham exists.
    ham = split_real_rows_by_cluster(ham)
    ham_train = int((ham["split"] == "train").sum())
    total_cap = int(cfg["external_train_max_rows"])
    scams = split_real_rows_by_cluster(
        scams, cap_per_cluster=int(cfg["external_train_max_per_cluster"]),
        max_train_rows=max(0, total_cap - ham_train))
    out = pd.concat([scams, ham], ignore_index=True)
    n_ext_train = int((out["split"] == "train").sum())
    print(f"  train external rows: {n_ext_train} / cap {total_cap} "
          f"(scams + {ham_train} ham)")
    assert n_ext_train <= total_cap, (
        f"external train cap violated: {n_ext_train} > {total_cap}")

    unknown = {t for tags in out["tactics"] for t in tags} - set(TACTIC_TAGS)
    assert not unknown, f"unknown tactic tags produced: {unknown}"

    dest = data_path("external_rows")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # tactics -> 'tag1;tag2' strings so the CSV round-trips through
    # build_dataset's normaliser exactly like label_cli output does.
    csv = out.copy()
    csv["tactics"] = csv["tactics"].map(lambda ts: ";".join(ts))
    csv["had_phone_number"] = csv["had_phone_number"].map(str)
    csv.to_csv(dest, index=False)
    print(f"\nWROTE {dest}")
    print(f"  rows: {len(out)}  verdicts: {out['verdict'].value_counts().to_dict()}")
    print("  sources:", out["source"].value_counts().to_dict())
    print("\nNext: python -m src.build_dataset   (train/val/test_real are preassigned)")
    return out


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build real test_real rows.")
    # Default comes from configs/config.yaml (phase35.ham_reserve) so the YAML
    # is the single source of truth; --ham only overrides it for one run.
    parser.add_argument("--ham", type=int,
                        default=int(CFG["phase35"]["ham_reserve"]),
                        help="how many real ham messages to reserve")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap IMC25 smishing rows (default: all)")
    args = parser.parse_args(argv)
    build_external(ham_n=args.ham, limit=args.limit)


if __name__ == "__main__":
    main()
