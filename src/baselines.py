"""
src/baselines.py - the Phase 2.5 baseline models (NO transformers).

The verdict is BINARY (scam vs safe). "suspicious" is an unsure band, not a
class; those rows live in data/processed/suspicious_holdout.parquet.

  1. majority     - always predicts the most common training verdict.
  2. keyword      - hand-written rules from configs/keywords.yaml.
  3. verdict_lr   - TF-IDF (word 1-2 grams + char 2-5 grams) + Logistic
                    Regression for the BINARY verdict.
  4. tactics_lr   - the same features, one-vs-rest logistic regression for
                    the 13 multi-label tactic tags.
  5. style_only   - SHORTCUT baseline using ONLY script_style.
  6. source_only  - SHORTCUT baseline using ONLY source.
                    If either scores high, the generator leaks the answer
                    through metadata and must be fixed.

All four share one tiny interface so src/evaluate.py can treat them the
same way:
    predict_verdicts(model, texts) -> list[str]           ("scam"/...)
    predict_tactics(model, texts)   -> list[list[str]]    (may be [])

Models are trained on the TRAIN split only and saved with joblib to
models/. Seed 42 is used everywhere it can be.

PUBLIC_WEAK (Phase 1 amendment #2)
  public_weak rows (English SMS Spam Collection) are EXCLUDED from training
  by default - they would teach the model an "English = safe" shortcut.
  --include_public_weak turns them back on for ONE clearly labelled
  ablation run; main results never use it.

Run:  python -m src.baselines                       (train + save)
      python -m src.baselines --include_public_weak  (ablation only!)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import CFG, SEED, TACTIC_TAGS, data_path
from src.io_utils import load_messages
from src.keyword_rules import KeywordRuleBaseline
from src.shortcut import SingleFeatureShortcut

MODELS_DIR = Path(data_path("models_dir"))

# Phase 3.5 training mixes. Every run and every table states which one is used.
MIX_SYNTH_ONLY = "mix_synth_only"
MIX_PLUS_REAL = "mix_synth_plus_real"
ALLOWED_MIXES = (MIX_SYNTH_ONLY, MIX_PLUS_REAL)


def load_training_frame(mix: str = MIX_PLUS_REAL,
                        include_public_weak: bool = False) -> pd.DataFrame:
    """The TRAIN split for one mix.

    mix_synth_only      : only source=synthetic / synthetic_llm
    mix_synth_plus_real : synthetic + real external scam + real UCI ham
    """
    if mix not in ALLOWED_MIXES:
        raise ValueError(f"unknown train mix {mix!r}; allowed: {ALLOWED_MIXES}")
    if include_public_weak:
        print("  NOTE: public_weak spam is no longer part of the table; "
              "real ham is source=public_ham_eval.")
    df = load_messages()
    df = df[df["split"] == "train"].copy()
    n_all = len(df)
    if mix == MIX_SYNTH_ONLY:
        df = df[df["source"].isin(["synthetic", "synthetic_llm"])].copy()
    print(f"  training rows: {len(df)} (from {n_all} train rows, mix={mix})")
    if df.empty:
        raise SystemExit("No training rows - run python -m src.build_dataset first.")
    print(f"  source mix: {df['source'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 1. majority baseline
# ---------------------------------------------------------------------------
def train_majority(df: pd.DataFrame) -> dict:
    """Remember the most common training verdict; predict it always."""
    counts = df["verdict"].value_counts()
    top = str(counts.index[0])
    print(f"  majority baseline -> always predicts '{top}' "
          f"({counts.iloc[0]}/{len(df)} training rows)")
    return {"kind": "majority", "verdict": top}


# ---------------------------------------------------------------------------
# 2. keyword-rule baseline (class lives in src/keyword_rules.py so joblib
#    can pickle it by its real module name, no matter which script runs)
# ---------------------------------------------------------------------------
def train_keyword() -> dict:
    kw_path = data_path("keywords")
    with open(kw_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = KeywordRuleBaseline(cfg)
    n_strong = len(model.strong) + len(model.weak) + len(model.veto)
    print(f"  keyword baseline loaded {n_strong} verdict words + "
          f"{len(model.tactic_kw)} tactic lists from {kw_path}")
    return {"kind": "keywords", "model": model}


# ---------------------------------------------------------------------------
# 3 + 4. shared TF-IDF features
# ---------------------------------------------------------------------------
def make_tfidf_features() -> FeatureUnion:
    """Word 1-2 grams AND char 2-5 grams (char_wb) - the shared features.

    char_wb means character n-grams computed inside word boundaries - this
    catches Hinglish spelling noise ('karein' vs 'kareeen') that word n-grams
    miss.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=60_000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2,
            max_features=60_000, sublinear_tf=True,
        )),
    ])


def make_tfidf_pipeline() -> Pipeline:
    """The features above + a class-balanced Logistic Regression.

    class_weight='balanced' compensates for class imbalance.
    """
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=4.0, random_state=SEED,
    )
    return Pipeline([("tfidf", make_tfidf_features()), ("clf", clf)])


def train_verdict_lr(df: pd.DataFrame) -> dict:
    print("  training verdict_lr (TF-IDF + LogisticRegression) ...")
    pipe = make_tfidf_pipeline()
    pipe.fit(df["text"].astype(str).tolist(), df["verdict"].tolist())
    print(f"    done. classes: {pipe.named_steps['clf'].classes_.tolist()}")
    return {"kind": "verdict_lr", "pipeline": pipe}


# ---------------------------------------------------------------------------
# 5 + 6. metadata-only SHORTCUT baselines
# ---------------------------------------------------------------------------
def train_shortcut(df: pd.DataFrame, feature: str) -> dict:
    print(f"  training shortcut baseline on ONLY '{feature}' ...")
    model = SingleFeatureShortcut(feature)
    model.fit(df[feature].astype(str).tolist(), df["verdict"].tolist())
    return {"kind": "single_feature", "model": model, "feature": feature}


def train_tactics_lr(df: pd.DataFrame) -> dict:
    """Multi-label tactics: ONE shared TF-IDF transform + one LogReg per tag.

    We do the transform ONCE and fit a small classifier per tag on the same
    matrix (this is what OneVsRestClassifier does, except it would also
    re-transform the text once per tag - very slow on a 32k-row test set).
    """
    print("  training tactics_lr (shared TF-IDF + per-tag LogisticRegression) ...")
    mlb = MultiLabelBinarizer(classes=list(TACTIC_TAGS))
    Y = mlb.fit_transform([sorted(set(t)) for t in df["tactics"]])
    counts = Y.sum(axis=0)
    print("    tags present in train:", {
        tag: int(c) for tag, c in zip(mlb.classes_, counts) if c > 0})
    features = make_tfidf_features()   # FeatureUnion HAS fit_transform
    X = features.fit_transform(df["text"].astype(str).tolist())
    # A tag with zero positives cannot be learned - keep it out, loudly.
    present = [i for i, c in enumerate(counts) if c > 0]
    classifiers = []
    for i in present:
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=4.0, random_state=SEED)
        clf.fit(X, Y[:, i])
        classifiers.append(clf)
    print(f"    trained {len(classifiers)} per-tag classifier(s)")
    return {"kind": "tactics_lr", "tfidf": features,
            "classifiers": classifiers, "mlb": mlb, "present": present}


# ---------------------------------------------------------------------------
# prediction adapters (the interface src/evaluate.py uses)
# ---------------------------------------------------------------------------
def predict_tactic_probs(model: dict, texts: list[str]) -> dict[str, list[float]]:
    """Per-tag P(tag present) for a tactics_lr bundle.

    The tag heads expose predict_proba; the verdict evaluator only needed hard
    labels. Phase 5's "suspicious" rule needs the probabilities ("weak
    evidence" = no tag above 0.5), so this adapter exists.
    """
    if model.get("kind") != "tactics_lr" or not texts:
        return {}
    X = model["tfidf"].transform(texts)
    mlb = model["mlb"]
    out: dict[str, list[float]] = {}
    for col, clf in zip(model["present"], model["classifiers"]):
        if not hasattr(clf, "predict_proba"):
            continue
        proba = clf.predict_proba(X)
        # classes_ is [0, 1]; take the positive column.
        pos = list(clf.classes_).index(1) if 1 in list(clf.classes_) else -1
        out[str(mlb.classes_[col])] = [float(v) for v in proba[:, pos]]
    return out


def predict_verdicts(model: dict, texts: list[str],
                     meta: pd.DataFrame | None = None) -> list[str]:
    """Predict a verdict.

    `meta` is only needed by the metadata-only shortcut baselines
    (style_only / source_only); text models ignore it.
    """
    kind = model["kind"]
    if kind == "majority":
        return [model["verdict"]] * len(texts)
    if kind == "keywords":
        return [model["model"].predict_one(t)[0] for t in texts]
    if kind == "verdict_lr":
        return list(model["pipeline"].predict(texts))
    if kind == "single_feature":
        if meta is None:
            raise ValueError("single_feature model needs a metadata DataFrame")
        feature = model["feature"]
        return model["model"].predict(meta[feature].astype(str).tolist())
    # tactics-only models have no verdict head
    return ["safe"] * len(texts)


def predict_scores(model: dict, texts: list[str],
                   meta: pd.DataFrame | None = None) -> list[float] | None:
    """Return P(scam) / risk score, or None when the model cannot rank.

    Used only for AUROC, PR-AUC and recall-at-FPR. The verdict decision still
    comes from predict_verdicts(); these scores must never change it.
    """
    kind = model["kind"]
    if kind == "keyword":
        return [model["model"].score_one(t) for t in texts]
    if kind == "verdict_lr":
        pipe = model["pipeline"]
        classes = list(pipe.named_steps["clf"].classes_)
        if "scam" not in classes:
            return None
        idx = classes.index("scam")
        return [float(row[idx]) for row in pipe.predict_proba(texts)]
    if kind == "single_feature":
        if meta is None:
            return None
        values = meta[model["feature"]].astype(str).tolist()
        model_obj = model["model"]
        if not hasattr(model_obj, "predict_proba"):
            return None
        if getattr(model_obj, "pipeline_", None) is None:
            return None
        return model_obj.predict_proba(values)
    # majority / tactics-only have no meaningful ranking.
    return None


def predict_tactics(model: dict, texts: list[str]) -> list[list[str]]:
    kind = model["kind"]
    if kind == "keywords":
        return [model["model"].predict_one(t)[1] for t in texts]
    if kind == "tactics_lr":
        X = model["tfidf"].transform(texts)          # ONE transform, reused
        mlb, present = model["mlb"], model["present"]
        full = np.zeros((len(texts), len(mlb.classes_)), dtype=int)
        for col, clf in zip(present, model["classifiers"]):
            full[:, col] = clf.predict(X)
        return [sorted(mlb.classes_[row]) for row in full]
    return [[] for _ in texts]


# ---------------------------------------------------------------------------
# train everything + save/load
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "majority": "majority.joblib",
    "keyword": "keyword.joblib",
    "verdict_lr_synth": "verdict_lr_synth.joblib",
    "verdict_lr_plus": "verdict_lr_plus.joblib",
    "tactics_lr_synth": "tactics_lr_synth.joblib",
    "tactics_lr_plus": "tactics_lr_plus.joblib",
    "style_only_plus": "style_only_plus.joblib",
    "source_only_plus": "source_only_plus.joblib",
}


def train_all(include_public_weak: bool = False) -> dict[str, dict]:
    """Train TF-IDF+LR on BOTH mixes, plus keyword and shortcut baselines."""
    print(f"(seed={SEED}) src/baselines.py - training")
    print("=" * 60)
    df_synth = load_training_frame(MIX_SYNTH_ONLY)
    df_plus = load_training_frame(MIX_PLUS_REAL)
    models = {
        "majority": train_majority(df_plus),
        "keyword": train_keyword(),
        "verdict_lr_synth": train_verdict_lr(df_synth),
        "verdict_lr_plus": train_verdict_lr(df_plus),
        "tactics_lr_synth": train_tactics_lr(df_synth),
        "tactics_lr_plus": train_tactics_lr(df_plus),
        # Shortcut baselines on the real-data mix: if these are strong, the
        # DATA (not the model) is leaking the label.
        "style_only_plus": train_shortcut(df_plus, "script_style"),
        "source_only_plus": train_shortcut(df_plus, "source"),
    }
    return models


def save_models(models: dict[str, dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, bundle in models.items():
        path = MODELS_DIR / MODEL_FILES[name]
        joblib.dump(bundle, path)
        print(f"  saved {path}")


def load_models() -> dict[str, dict]:
    """Load every trained baseline bundle that exists.

    Missing variants are skipped with a warning so an older checkout can still
    evaluate the models it has; the comparison table simply omits absent rows.
    """
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"  WARNING: {path.name} missing - skipping {name}")
            continue
        models[name] = joblib.load(path)
    if not models:
        raise FileNotFoundError(
            f"No baseline bundles in {MODELS_DIR}. Train: python -m src.baselines")
    print(f"  loaded {len(models)} model bundle(s) from {MODELS_DIR}")
    return models


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 3.5 baselines.")
    parser.add_argument("--include_public_weak", action="store_true",
                        help="deprecated: public_weak spam is no longer loaded")
    args = parser.parse_args(argv)
    models = train_all(include_public_weak=args.include_public_weak)
    save_models(models)
    print("\nNext: python -m src.evaluate   (honest evaluation on all test sets)")


if __name__ == "__main__":
    main()
