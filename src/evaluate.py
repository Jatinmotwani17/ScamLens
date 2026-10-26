"""
src/evaluate.py - the HONEST evaluation harness (Phase 2.5 + Phase 3).

THE RULES (these override convenience):
  * The verdict head is BINARY: scam vs safe. "suspicious" is NOT a class;
    those rows live in data/processed/suspicious_holdout.parquet and come back
    in Phase 4 as a confidence-based abstention test.
  * test_real is NEVER reported as one blended number. It is split into
    real_stratum buckets (user_real_scam, user_real_safe, external_scam,
    public_ham_english) and every metric is computed per bucket.
  * A tactic tag is only a headline metric if test_real contains at least
    min_tag_support (20) real positives for it. Everything else is listed as
    "unvalidated (synthetic only)" and its F1 is NOT shown as a real number.
  * A model that cannot predict tags (majority, verdict_lr, the shortcut
    baselines, or a transformer trained with lambda_tags=0) gets "n/a" for tag
    metrics - never a fake 0.000.
  * test_a / test_b are logged for observation. They never choose a
    checkpoint or a hyperparameter; that is validation-only (src/train.py).

Outputs:
    results/metrics.json   machine-readable (all numbers, bootstrap CIs)
    results/RESULTS.md     human-readable tables, one block per test set,
                           one block PER REAL STRATUM, plus the E4 gate.

Run:  python -m src.evaluate
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.baselines import (
    load_models,
    predict_scores,
    predict_tactics,
    predict_verdicts,
)
from src.audit import source_origin_scores
from src.config import CFG, REAL_STRATA, SEED, TACTIC_TAGS, data_path
from src.io_utils import load_messages

EVAL_CFG = CFG["evaluate"]
VERDICT_LABELS = ["safe", "scam"]          # BINARY - no "suspicious"
SYNTHETIC_SPLITS = ["val", "test_a", "test_b"]
MIN_STRATUM_ROWS = int(EVAL_CFG.get("min_stratum_rows", 30))
MIN_TAG_SUPPORT = int(EVAL_CFG.get("min_tag_support", 20))
HEADLINE_NOTE = ("Synthetic results are not performance claims. Headline numbers "
                 "come only from test_real, and always PER STRATUM.")
# Phase 3.6: the checkpoint is chosen on validation only, and the validation
# score is the mean of two balanced numbers (see src/train.py).
SELECTION_NOTE = ("validation only: mean(val_synthetic macro-F1, "
                  "val_real balanced accuracy)")
NO_REAL_NOTE = ("NO VALID REAL TEST SET: results are not valid performance claims.")
E4_NOTE = ("E4 (real-message cross-validation) needs >=50 in-domain real "
           "messages AND >=20 user_real_safe hard negatives.")

# These two baselines really do output tactic tags. The others (majority,
# verdict_lr, style_only, source_only) have no tag head at all.
TAG_CAPABLE_BASELINES = {
    "keywords", "tactics_lr", "tactics_lr_synth", "tactics_lr_plus",
}

# Nice labels for the comparison tables. The JSON keys stay unchanged.
MODEL_LABELS = {
    "majority": "majority",
    "keyword": "keyword rules",
    "verdict_lr": "TF-IDF + LR",
    "verdict_lr_synth": "TF-IDF + LR (synth only)",
    "verdict_lr_plus": "TF-IDF + LR (synth+real)",
    "tactics_lr": "TF-IDF tags LR",
    "tactics_lr_synth": "TF-IDF tags LR (synth only)",
    "tactics_lr_plus": "TF-IDF tags LR (synth+real)",
    "style_only": "style_only (shortcut)",
    "style_only_plus": "style_only (shortcut)",
    "source_only": "source_only (shortcut)",
    "source_only_plus": "source_only (shortcut)",
}

# The combined English real block is explicitly optimistic and NOT Hinglish.
COMBINED_STRATUM = "real_english_combined"
COMBINED_TITLE = ("English real data, optimistic: sources differ in era and "
                  "style, NOT a Hinglish claim")
# Phase 3.6: every AUROC in that block must be read next to this warning.
ORIGIN_WARNING = ("AUROC close to source-origin AUROC means the score may "
                  "reflect dataset recognition, not scam detection.")

# Nearest-neighbour bands used in the leakage table (src/nn_sim.py).
SIM_HIGH = 0.9
SIM_MID = 0.7
NN_SIM_NOTE = (
    "For every val/test row we computed the maximum TF-IDF char-ngram cosine "
    "similarity to ANY train row (`results/nn_sim.parquet`). A high value "
    "means the row has a near-copy in training, so its score partly measures "
    "memorisation.")


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------
def _fmt(value, digits: int = 3) -> str:
    """Format a metric, or 'n/a' when it is None/NaN (e.g. tags not predicted)."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def _ci(r: dict) -> str:
    lo, hi = r.get("macro_f1_ci95", [None, None])
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def _prob(m: dict, key: str) -> str:
    p = m.get("probability") or {}
    return _fmt(p.get(key))


def _recall_at(m: dict, key: str) -> str:
    """Recall@FPR cell, honouring the "too few negatives" guard.

    The guard note is used verbatim so a reader cannot mistake an unmeasured
    point for a measured zero.
    """
    p = m.get("probability") or {}
    note = p.get(f"{key}_note")
    return str(note) if note else _fmt(p.get(key))


def _empty_tactics() -> dict:
    """Tag metrics for a model that has no tag head: explicit 'not available'."""
    return {
        "micro_f1": None,
        "macro_f1": None,
        "per_tag_f1": {},
        "headline_tags": [],
        "rows_with_gold_tags": 0,
        "available": False,
    }


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------
def bootstrap_macro_f1(y_true: list, y_pred: list, n: int = None) -> tuple[float, float, float]:
    """Seeded bootstrap 95% CI for binary verdict macro-F1.

    We count true/false positives directly instead of calling sklearn 1,000
    times: test_real's biggest stratum has 23k rows, and sklearn-per-resample
    made this take tens of minutes. The formula is exactly the same macro-F1
    (safe and scam each count once, zero_division=0).
    """
    n = n or int(EVAL_CFG["bootstrap_resamples"])
    # Encode the two classes as 0 (safe) / 1 (scam).
    yt = (np.asarray(y_true) == "scam").astype(np.int8)
    yp = (np.asarray(y_pred) == "scam").astype(np.int8)
    if len(yt) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        t = yt[idx]
        p = yp[idx]
        # codes: 0=true safe/pred safe, 1=true safe/pred scam,
        #        2=true scam/pred safe, 3=true scam/pred scam
        tn, fp, fn, tp = np.bincount(t * 2 + p, minlength=4)
        scam_den = 2 * tp + fp + fn
        safe_den = 2 * tn + fn + fp
        f_scam = (2 * tp / scam_den) if scam_den else 0.0
        f_safe = (2 * tn / safe_den) if safe_den else 0.0
        scores[i] = (f_scam + f_safe) / 2
    return (float(scores.mean()), float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


# Phase 3.6 metric guard: "recall at 1% FPR" is only a number if at least
# MIN_TNEG_FOR_FPR negatives exist. With 59 real ham rows, a 1% FPR point is
# 0.59 rows - reporting "0.000" there is a fake measurement, not a result.
MIN_TNEG_FOR_FPR = 5
TOO_FEW_NEG = "n/a (too few negatives)"


def _guarded_recall_at(limit: float, n_neg: int, tpr, fpr) -> tuple[float | None, str | None]:
    """Recall at an FPR budget, or (None, note) when there are too few negatives.

    The guard is `n_negatives * limit >= 5`: the budget must be worth at least
    five actual negatives before the threshold is measured rather than invented.
    """
    if n_neg * limit < MIN_TNEG_FOR_FPR:
        return None, TOO_FEW_NEG
    allowed = tpr[fpr <= limit]
    return (float(allowed.max()) if len(allowed) else 0.0), None


def probability_metrics(y_true: list, scores: list[float] | None) -> dict:
    """AUROC / PR-AUC / recall at 1% and 5% FPR.

    Returns None values when a model has no score or the subset has one class.
    These metrics never change the verdict; they only rank the same model.

    The recall@FPR numbers carry a `*_note` of "n/a (too few negatives)" when
    the subset does not contain enough negatives for that FPR budget (see
    _guarded_recall_at). The number is then None, never a flattering 0.000.
    """
    empty = {"auroc": None, "pr_auc": None,
             "recall_at_1pct_fpr": None, "recall_at_1pct_fpr_note": TOO_FEW_NEG,
             "recall_at_5pct_fpr": None, "recall_at_5pct_fpr_note": TOO_FEW_NEG,
             "n_negatives": 0, "available": False}
    if scores is None:
        # No scores at all (majority / rules): the reason is "nothing to rank",
        # not "too few negatives", so the cells must read plain "n/a".
        return {**empty,
                "recall_at_1pct_fpr_note": None,
                "recall_at_5pct_fpr_note": None}
    yt = (np.asarray(y_true) == "scam").astype(np.int8)
    sc = np.asarray(scores, dtype=float)
    if len(yt) == 0 or len(set(yt.tolist())) < 2:
        empty["n_negatives"] = int((yt == 0).sum()) if len(yt) else 0
        return empty
    n_neg = int((yt == 0).sum())
    try:
        auroc = float(roc_auc_score(yt, sc))
        pr_auc = float(average_precision_score(yt, sc))
        fpr, tpr, _ = roc_curve(yt, sc)
        r1, note1 = _guarded_recall_at(0.01, n_neg, tpr, fpr)
        r5, note5 = _guarded_recall_at(0.05, n_neg, tpr, fpr)
        return {"auroc": auroc, "pr_auc": pr_auc,
                "recall_at_1pct_fpr": r1, "recall_at_1pct_fpr_note": note1,
                "recall_at_5pct_fpr": r5, "recall_at_5pct_fpr_note": note5,
                "n_negatives": n_neg, "available": True}
    except ValueError:
        empty["n_negatives"] = n_neg
        return empty


def bootstrap_ranking(y_true: list, scores: list[float],
                      n: int | None = None) -> dict:
    """Seeded bootstrap 95% CIs for AUROC and PR-AUC (combined real block)."""
    n = n or int(EVAL_CFG["bootstrap_resamples"])
    yt = (np.asarray(y_true) == "scam").astype(np.int8)
    sc = np.asarray(scores, dtype=float)
    if len(set(yt.tolist())) < 2:
        return {"auroc_ci95": [None, None], "pr_auc_ci95": [None, None]}
    rng = np.random.default_rng(SEED)
    aucs, prs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        if len(set(yt[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(yt[idx], sc[idx]))
        prs.append(average_precision_score(yt[idx], sc[idx]))
    if not aucs:
        return {"auroc_ci95": [None, None], "pr_auc_ci95": [None, None]}
    return {
        "auroc_ci95": [float(np.percentile(aucs, 2.5)),
                       float(np.percentile(aucs, 97.5))],
        "pr_auc_ci95": [float(np.percentile(prs, 2.5)),
                        float(np.percentile(prs, 97.5))],
    }


def verdict_metrics(df: pd.DataFrame, preds: list[str]) -> dict:
    """Binary per-class P/R/F1 + macro-F1 + 2x2 confusion + scam-specific numbers."""
    y_true = df["verdict"].tolist()
    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=VERDICT_LABELS, zero_division=0)
    per_class = {
        cls: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f1[i]), "support": int(support[i])}
        for i, cls in enumerate(VERDICT_LABELS)
    }
    macro = float(f1_score(y_true, preds, labels=VERDICT_LABELS,
                           average="macro", zero_division=0))
    cm = confusion_matrix(y_true, preds, labels=VERDICT_LABELS).tolist()

    safe_mask = df["verdict"] == "safe"
    scam_on_safe = sum(1 for truth, pr in zip(y_true, preds)
                       if truth == "safe" and pr == "scam")

    # "Hard negative" is a note written by the synthetic generator. Real rows
    # are not annotated with it, so on test_real this is None (= not measured),
    # NOT 0.0. Showing 0.0 there would be a false claim.
    notes = df["notes"].astype(str) if "notes" in df.columns else pd.Series([""] * len(df))
    hard_mask = safe_mask & notes.str.contains("hard_negative")
    hard_n = int(hard_mask.sum())
    hard_preds = [pr for pr, m in zip(preds, hard_mask) if m]
    fpr_hard = (sum(1 for pr in hard_preds if pr == "scam") / hard_n) if hard_n else None

    single_class = len(set(str(v) for v in y_true)) < 2
    # Balanced accuracy = mean of the two per-class recalls. Unlike macro-F1 it
    # ignores precision, which is exactly what we want for the real validation
    # split: 98.3% of it is scam, so a model that flags everything gets a
    # beautiful macro-F1 but a balanced accuracy of 0.5. Phase 3.6 selects the
    # checkpoint on this (plus synthetic macro-F1), never on combined macro-F1.
    balanced_accuracy = None if single_class else (
        per_class["safe"]["recall"] + per_class["scam"]["recall"]) / 2
    return {
        "per_class": per_class,
        # A single-class stratum has no honest macro-F1: averaging the one real
        # class with a zero for the absent class is not a performance number.
        "macro_f1": None if single_class else macro,
        "balanced_accuracy": balanced_accuracy,
        "single_class": bool(single_class),
        "confusion_matrix": cm,
        "scam_recall": per_class["scam"]["recall"],
        "safe_recall": per_class["safe"]["recall"],
        "fpr_on_safe": scam_on_safe / max(int(safe_mask.sum()), 1),
        "n_safe": int(safe_mask.sum()),
        "hard_negatives": hard_n,
        "fpr_on_hard_negatives": fpr_hard,
        "n_rows": len(df),
    }


def tactic_metrics(df: pd.DataFrame, preds: list[list[str]],
                   validated_tags: list[str] | None = None) -> dict:
    """Multi-label tag metrics. Headline macro/micro use VALIDATED tags only.

    `validated_tags` = tags with >=20 real positives in this stratum. On
    synthetic splits we pass None, meaning "use every tag".
    """
    mlb = MultiLabelBinarizer(classes=list(TACTIC_TAGS))
    gold = mlb.fit_transform([sorted(set(t)) for t in df["tactics"]])
    pred = mlb.transform([sorted(set(t)) for t in preds])
    per_tag = f1_score(gold, pred, average=None, zero_division=0)
    all_per_tag = {tag: float(v) for tag, v in zip(TACTIC_TAGS, per_tag)}

    if validated_tags is None:
        used = list(TACTIC_TAGS)
    else:
        used = [t for t in TACTIC_TAGS if t in set(validated_tags)]
    idx = [TACTIC_TAGS.index(t) for t in used]
    if idx:
        micro = float(f1_score(gold[:, idx], pred[:, idx], average="micro", zero_division=0))
        macro = float(f1_score(gold[:, idx], pred[:, idx], average="macro", zero_division=0))
    else:
        micro = macro = None
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "per_tag_f1": all_per_tag,
        "headline_tags": used,
        "rows_with_gold_tags": int((gold.sum(axis=1) > 0).sum()),
        "available": True,
    }


def tag_support(df: pd.DataFrame) -> dict[str, int]:
    """How many REAL positive examples each tag has in this subset."""
    mlb = MultiLabelBinarizer(classes=list(TACTIC_TAGS))
    gold = mlb.fit_transform([sorted(set(t)) for t in df["tactics"]])
    return {tag: int(v) for tag, v in zip(TACTIC_TAGS, gold.sum(axis=0))}


def _tag_capable(model: dict) -> bool:
    """True only for baselines that actually produce tactic tags."""
    return model.get("kind") in TAG_CAPABLE_BASELINES


def evaluate_subset(test: pd.DataFrame, models: dict,
                    validated_tags: list[str] | None = None) -> dict:
    """Run every BASELINE on one subset and return {model: metrics}."""
    texts = test["text"].astype(str).tolist()
    out: dict = {}
    for name, model in models.items():
        # `test` is passed so metadata-only shortcuts can use script_style /
        # source. Text models ignore it.
        y_pred = predict_verdicts(model, texts, test)
        m = verdict_metrics(test, y_pred)
        scores = predict_scores(model, texts, test)
        m["probability"] = probability_metrics(test["verdict"].tolist(), scores)
        if _tag_capable(model):
            m["tactics"] = tactic_metrics(
                test, predict_tactics(model, texts), validated_tags=validated_tags)
        else:
            m["tactics"] = _empty_tactics()
        if m["macro_f1"] is None:
            m["macro_f1_bootstrap_mean"] = None
            m["macro_f1_ci95"] = [None, None]
        else:
            mean, lo, hi = bootstrap_macro_f1(test["verdict"].tolist(), y_pred)
            m["macro_f1_bootstrap_mean"] = mean
            m["macro_f1_ci95"] = [lo, hi]
        out[name] = m
    return out


# ---------------------------------------------------------------------------
# real-data gate (Phase 1 amendment #7, tightened in Phase 2.5/Phase 3)
# ---------------------------------------------------------------------------
def real_data_gate(df: pd.DataFrame) -> dict:
    """Return the Phase 3/E4 gate result and print it loudly when it fails.

    E4 (cluster-grouped CV on real messages) is only honest when there are at
    least 50 in-domain real messages and >=20 user_real_safe hard negatives.
    """
    real = df[df["split"] == "test_real"]
    n_scam = int((real["real_stratum"] == "user_real_scam").sum())
    n_safe = int((real["real_stratum"] == "user_real_safe").sum())
    n_in = n_scam + n_safe
    passed = bool(n_in >= 50 and n_safe >= 20)
    print(f"PHASE 3 GATE: in-domain real messages = {n_in} "
          f"(user_real_scam={n_scam}, user_real_safe={n_safe})")
    if not passed:
        print("!" * 70)
        print(f"E4 GATE FAILED: need >=50 in-domain real messages AND >=20 "
              f"user_real_safe hard negatives; found {n_in} and {n_safe}.")
        print("Phase 3 will SKIP E4 (the real-message cross-validation).")
        print("!" * 70)
    if n_in and n_scam < 20:
        print("NOTE: fewer than 20 user_real_scam rows - even a forced E4 "
              "could not measure scam recall honestly.")
    return {"in_domain": n_in, "user_real_scam": n_scam,
            "user_real_safe": n_safe, "passed": passed,
            "requirement": ">=50 in-domain and >=20 user_real_safe"}


def stratified_subsets(real: pd.DataFrame):
    """Yield (stratum, rows, is_too_small) for every non-empty real stratum."""
    for strat in REAL_STRATA:
        sub = real[real["real_stratum"] == strat].reset_index(drop=True)
        if len(sub) == 0:
            continue
        yield strat, sub, len(sub) < MIN_STRATUM_ROWS


def style_mix(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["script_style"].value_counts().items()}


def provenance_block(df: pd.DataFrame) -> dict:
    """Data provenance for RESULTS.md: splits, sources, tag support, Hinglish.

    This is written next to the scores so nobody can read a table without also
    seeing where the rows came from and which tags the training set could
    actually learn.
    """
    splits = {}
    for split in ("train", "val", "test_a", "test_b", "test_real"):
        sub = df[df["split"] == split]
        splits[split] = {
            "rows": int(len(sub)),
            "sources": {str(k): int(v) for k, v in sub["source"].value_counts().items()},
        }
    support = []
    for tag in TACTIC_TAGS:
        row = {"tag": tag}
        for split in ("train", "val", "test_a", "test_b", "test_real"):
            sub = df[df["split"] == split]
            row[split] = int(sum(tag in t for t in sub["tactics"]))
        support.append(row)
    real = df[df["source"].isin(
        ["external_reported", "public_ham_eval", "self_collected"])]
    ext = df[df["source"] == "external_reported"]
    ham = df[df["source"] == "public_ham_eval"]
    return {
        "mix": str(CFG.get("train_mix", {}).get("default", "?")),
        "splits": splits,
        "tag_support": support,
        "real_style": style_mix(real),
        "external_style": style_mix(ext),
        "ham_style": style_mix(ham),
        "real_rows": int(len(real)),
        "real_hinglish": int((real["script_style"] == "roman_hinglish").sum()),
    }


# ---------------------------------------------------------------------------
# nearest-neighbour leakage summary (Phase 3.6)
# ---------------------------------------------------------------------------
def nn_similarity_summary() -> dict | None:
    """Per-stratum shares of max-train similarity, read from nn_sim.parquet.

    Returns None when src/nn_sim.py has not been run, so a missing audit is
    visible as "not run" instead of silently claiming there is no leakage.
    """
    path = data_path("nn_sim")
    if not path.exists():
        return None
    sim = pd.read_parquet(path)
    if sim.empty:
        return None

    rows = []
    preferred = ["val_synthetic", "val_real", "test_a", "test_b",
                 *REAL_STRATA]
    seen = list(dict.fromkeys(sim["stratum"].astype(str).tolist()))
    ordered = [s for s in preferred if s in seen] + \
              [s for s in seen if s not in preferred]
    for stratum in ordered:
        sub = sim[sim["stratum"].astype(str) == stratum]
        n = len(sub)
        values = sub["max_train_sim"].astype(float)
        high = int((values > SIM_HIGH).sum())
        mid = int(((values >= SIM_MID) & (values <= SIM_HIGH)).sum())
        low = int((values < SIM_MID).sum())
        rows.append({
            "stratum": stratum,
            "split": str(sub["split"].iloc[0]) if n else "",
            "n_rows": n,
            "share_gt_0.9": high / n if n else 0.0,
            "share_0.7_0.9": mid / n if n else 0.0,
            "share_lt_0.7": low / n if n else 0.0,
            "median_sim": float(values.median()) if n else None,
        })

    real = sim[sim["split"] == "test_real"]
    real_share = (float((real["max_train_sim"].astype(float) > SIM_HIGH).mean())
                  if len(real) else None)
    return {
        "n_rows": int(len(sim)),
        "note": NN_SIM_NOTE,
        "strata": rows,
        "real_test_rows": int(len(real)),
        "real_test_share_gt_0.9": real_share,
        # The Phase 3.6 sentence that must appear when this is true.
        "real_test_mostly_near_copies": bool(real_share is not None
                                             and real_share > 0.5),
    }


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------
def evaluate_all() -> dict:
    print(f"(seed={SEED}) src/evaluate.py - honest BINARY evaluation")
    print("=" * 60)
    df = load_messages()

    if "real_stratum" not in df.columns:
        raise SystemExit("real_stratum column missing - run python -m src.build_dataset")
    if "suspicious" in set(df["verdict"]):
        raise SystemExit("suspicious rows found in the MAIN table. They belong in "
                         "data/processed/suspicious_holdout.parquet (run build_dataset).")

    real = df[df["split"] == "test_real"]
    if len(real) < int(EVAL_CFG["min_real_test_rows"]):
        print("!" * 70)
        print(NO_REAL_NOTE)
        print("!" * 70)
    else:
        print(f"test_real: {len(real)} rows across "
              f"{real['real_stratum'].nunique()} stratum/strata")

    gate = real_data_gate(df)
    models = load_models()
    results: dict = {"meta": {
        "provenance": provenance_block(df),
        "binary": True,
        "verdict_labels": VERDICT_LABELS,
        "min_stratum_rows": MIN_STRATUM_ROWS,
        "min_tag_support": MIN_TAG_SUPPORT,
        "note": HEADLINE_NOTE,
        "selection_metric": SELECTION_NOTE,
        "e4_gate": gate,
        "nn_similarity": nn_similarity_summary(),
        "metric_guards": {
            "min_negatives_for_recall_at_fpr": MIN_TNEG_FOR_FPR,
            "rule": "recall@FPR=x only if n_negatives * x >= 5, else 'n/a "
                    "(too few negatives)'",
        },
        "runs": {},
    }}

    # ---- synthetic development splits ------------------------------------
    for split in SYNTHETIC_SPLITS:
        test = df[df["split"] == split].reset_index(drop=True)
        if test.empty:
            print(f"\n--- {split}: EMPTY - skipped")
            continue
        print(f"\n--- evaluating on '{split}' ({len(test)} rows) ---")
        results[split] = {
            "n_rows": len(test),
            "scam_rate": float((test["verdict"] == "scam").mean()),
            "style_mix": style_mix(test),
            "single_class": len(set(test["verdict"])) < 2,
            "models": evaluate_subset(test, models),
        }
        # Phase 3.6: test_a is a 10% cluster hold-out from TRAIN (both classes).
        # Its scores test "seen wording"; test_b tests unseen families.
        if split == "test_a":
            results[split]["note"] = (
                "10% of TRAIN clusters held out whole (both classes); measures "
                "seen-template wording, not new scams.")
        for name, m in results[split]["models"].items():
            print(f"    {name:20s} macro-F1={_fmt(m['macro_f1'])} "
                  f"scam-recall={m['scam_recall']:.3f} "
                  f"hard-neg-FPR={_fmt(m['fpr_on_hard_negatives'])} "
                  f"AUROC={_prob(m, 'auroc')} "
                  f"tag-micro={_fmt(m['tactics']['micro_f1'])}")

    # ---- val is COMBINED (synthetic + real). Also report each part. --------
    val = df[df["split"] == "val"].reset_index(drop=True)
    if not val.empty:
        for part_name, mask in (
            ("val_synthetic", val["source"].isin(["synthetic", "synthetic_llm"])),
            ("val_real", val["source"].isin(["external_reported", "public_ham_eval"])),
        ):
            sub = val[mask].reset_index(drop=True)
            if sub.empty:
                continue
            results[part_name] = {
                "n_rows": len(sub),
                "scam_rate": float((sub["verdict"] == "scam").mean()),
                "style_mix": style_mix(sub),
                "models": evaluate_subset(sub, models),
            }
            print(f"  {part_name}: {len(sub)} rows, "
                  f"scam rate {(sub['verdict'] == 'scam').mean():.1%}")

    # ---- test_real: PER STRATUM ONLY -------------------------------------
    print("\n--- evaluating test_real PER STRATUM (never one blended number) ---")
    strata_results: dict = {}
    for strat, sub, too_small in stratified_subsets(real):
        support = tag_support(sub)
        validated = [t for t, n in support.items() if n >= MIN_TAG_SUPPORT]
        flag = "  <-- TOO SMALL FOR CLAIMS" if too_small else ""
        print(f"  {strat}: {len(sub)} rows, scam rate "
              f"{(sub['verdict'] == 'scam').mean():.1%}{flag}")
        strata_results[strat] = {
            "n_rows": len(sub),
            "too_small_for_claims": bool(too_small),
            "scam_rate": float((sub["verdict"] == "scam").mean()),
            "style_mix": style_mix(sub),
            "tag_support": support,
            "validated_tags": validated,
            "unvalidated_tags": [t for t in TACTIC_TAGS if t not in validated],
            "models": evaluate_subset(sub, models, validated_tags=validated),
        }
    # ---- combined English real block (explicitly optimistic) --------------
    combined = real[real["real_stratum"].isin(
        ["external_scam", "public_ham_english"])].reset_index(drop=True)
    if len(combined):
        support = tag_support(combined)
        validated = [t for t, n in support.items() if n >= MIN_TAG_SUPPORT]
        block = {
            "n_rows": len(combined),
            "too_small_for_claims": len(combined) < MIN_STRATUM_ROWS,
            "scam_rate": float((combined["verdict"] == "scam").mean()),
            "title": COMBINED_TITLE,
            "style_mix": style_mix(combined),
            "tag_support": support,
            "validated_tags": validated,
            "unvalidated_tags": [t for t in TACTIC_TAGS if t not in validated],
            "models": evaluate_subset(combined, models, validated_tags=validated),
        }
        # Bootstrap CIs for the combined block's ranking metrics.
        for name, m in block["models"].items():
            scores = predict_scores(models[name], combined["text"].tolist(), combined)
            if scores is not None and m["probability"]["available"]:
                m["probability"].update(bootstrap_ranking(
                    combined["verdict"].tolist(), scores))
            else:
                m["probability"].update({"auroc_ci95": [None, None],
                                         "pr_auc_ci95": [None, None]})

        # Phase 3.6: how separable ARE the two public corpora from text alone?
        # Printed next to every AUROC in this block, because "the model scores
        # 0.99" is not impressive if recognising the dataset alone scores 0.99.
        origin = source_origin_scores(df)
        block["source_origin"] = origin
        block["source_origin_warning"] = ORIGIN_WARNING
        origin_auc = origin.get("test_real", {}).get("auroc")
        print(f"  source-origin classifier (TEXT only): "
              f"test_real AUROC={_fmt(origin_auc)} "
              f"accuracy={_fmt(origin.get('test_real', {}).get('accuracy'))}")
        if origin_auc is not None:
            for name in ("verdict_lr", "verdict_lr_plus"):
                if name in block["models"]:
                    print(f"  {name} real_english_combined AUROC="
                          f"{_prob(block['models'][name], 'auroc')} | "
                          f"source-origin AUROC={_fmt(origin_auc)}")
            print(f"  WARNING: {ORIGIN_WARNING}")
        strata_results[COMBINED_STRATUM] = block

    # Loudly record rows that have no stratum: they would otherwise vanish.
    stray = real[~real["real_stratum"].isin(list(REAL_STRATA))]
    results["test_real"] = {
        "note": HEADLINE_NOTE,
        "overall_rows": int(len(real)),
        "unstratified_rows": int(len(stray)),
        "e4_gate": gate,
        "strata": strata_results,
    }
    return results


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def _model_table(models: dict, with_tags: bool = True,
                 origin_auroc: float | None = None) -> list[str]:
    """One comparison table with every model, including the shortcut rows.

    `origin_auroc` adds a "source-origin AUROC" column. It is only passed for
    the combined English real block, where the same number must sit next to
    every AUROC: a text classifier can score ~1.0 simply by recognising WHICH
    public dataset a message came from.
    """
    origin_head = " source-origin AUROC |" if origin_auroc is not None else ""
    origin_sep = "---|" if origin_auroc is not None else ""
    if with_tags:
        lines = ["| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC |" + origin_head + " PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |",
                 "|---|---|---|---|---|---|---|---|" + origin_sep + "---|---|---|---|---|"]
    else:
        lines = ["| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC |" + origin_head + " PR-AUC | R@1%FPR | R@5%FPR |",
                 "|---|---|---|---|---|---|---|---|" + origin_sep + "---|---|---|---|"]
    for name, r in models.items():
        scam_f1 = r["per_class"]["scam"]["f1"]
        base = (f"| {MODEL_LABELS.get(name, name)} | {_fmt(r['macro_f1'])} "
                f"| {scam_f1:.3f} | {_ci(r)} | {r['scam_recall']:.3f} "
                f"| {r['fpr_on_safe']:.3f} | {_fmt(r['fpr_on_hard_negatives'])} "
                f"| {_prob(r, 'auroc')} |")
        if origin_auroc is not None:
            base += f" {_fmt(origin_auroc)} |"
        base += (f" {_prob(r, 'pr_auc')} "
                 f"| {_recall_at(r, 'recall_at_1pct_fpr')} "
                 f"| {_recall_at(r, 'recall_at_5pct_fpr')} ")
        if with_tags:
            lines.append(base + f"| {_fmt(r['tactics'].get('micro_f1'))} "
                               f"| {_fmt(r['tactics'].get('macro_f1'))} |")
        else:
            lines.append(base + "|")
    return lines


def _tag_table(block: dict) -> list[str]:
    """Per-tag table: F1 ONLY for tags with >=20 real positives.

    Tags below the support gate are listed as 'unvalidated (synthetic only)'
    and their F1 is hidden, because a 3-example F1 is noise dressed as a metric.
    """
    models = {name: m for name, m in block["models"].items()
              if m.get("tactics", {}).get("available")}
    cols = list(models)
    lines = ["| tag | real positives | status | " + " | ".join(cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for tag, n in sorted(block["tag_support"].items(), key=lambda kv: (-kv[1], kv[0])):
        validated = tag in block["validated_tags"]
        status = "validated" if validated else "unvalidated (synthetic only)"
        if validated:
            f1s = [_fmt(models[c]["tactics"]["per_tag_f1"].get(tag)) for c in cols]
        else:
            f1s = ["—"] * len(cols)
        lines.append(f"| {tag} | {n} | {status} | " + " | ".join(f1s) + " |")
    lines += ["", f"Headline tag-F1 uses **{len(block['validated_tags'])}** validated "
                  f"tag(s): {block['validated_tags']}"]
    if not cols:
        lines.append("(No model in this comparison predicted tactic tags.)")
    return lines


def add_transformer_runs(results: dict, df: pd.DataFrame) -> None:
    """Add every outputs/<run>/preds_*.parquet to the same result tables.

    Predictions are written by src/predict.py. A run trained with
    lambda_tags=0 has an untrained tag head, so its tag metrics are reported
    as n/a - never as a real 0.000.
    """
    from src.config import PROJECT_ROOT
    import yaml

    outputs = Path(PROJECT_ROOT) / "outputs"
    if not outputs.exists():
        print("\n(no outputs/ folder - skipping transformer prediction tables)")
        return
    runs = sorted([p for p in outputs.iterdir()
                   if p.is_dir() and (p / "config_used.yaml").exists()])
    if not runs:
        print("\n(no trained transformer runs found in outputs/)")
        return
    print(f"\n--- adding {len(runs)} transformer run(s) from outputs/ ---")

    truth = df.set_index("id")
    for run in runs:
        run_name = run.name
        cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
        tags_trained = float(cfg.get("lambda_tags", 0) or 0) > 0
        best_row = _best_row(run) or {}
        results["meta"]["runs"][run_name] = {
            "model_name": cfg.get("model_name"),
            "lambda_tags": cfg.get("lambda_tags"),
            "tags_trained": tags_trained,
            "best_epoch": best_row.get("epoch"),
            # Validation-only selection score and the two numbers it averages.
            "selection_score": (_selection_value(best_row) if best_row else None),
            "val_synthetic_macro_f1": best_row.get("val_synth_macro_f1"),
            "val_real_balanced_accuracy": best_row.get("val_real_balanced_accuracy"),
            # Observation-only combined macro-F1 (not the selection metric).
            "val_macro_f1": best_row.get("val_macro_f1"),
        }
        added = 0

        # ---- synthetic splits: one table each ----------------------------
        for split in SYNTHETIC_SPLITS:
            pred_path = run / f"preds_{split}.parquet"
            if not pred_path.exists() or split not in results:
                continue
            pred = pd.read_parquet(pred_path)
            if pred.empty:
                continue
            # Old prediction files may not carry `notes`; merge it so the
            # hard-negative metric is computed from the same labels.
            if "notes" not in pred.columns:
                pred["notes"] = pred["id"].map(truth["notes"]).fillna("")
            y_pred = pred["pred_verdict"].tolist()
            m = verdict_metrics(pred, y_pred)
            scores = pred["prob_scam"].tolist() if "prob_scam" in pred else None
            m["probability"] = probability_metrics(pred["verdict"].tolist(), scores)
            m["tactics"] = (tactic_metrics(pred, pred["pred_tags"].tolist())
                            if tags_trained else _empty_tactics())
            if m["macro_f1"] is None:
                m["macro_f1_bootstrap_mean"] = None
                m["macro_f1_ci95"] = [None, None]
            else:
                mean, lo, hi = bootstrap_macro_f1(pred["verdict"].tolist(), y_pred)
                m["macro_f1_bootstrap_mean"] = mean
                m["macro_f1_ci95"] = [lo, hi]
            results[split]["models"][run_name] = m
            added += 1

        # ---- test_real: per stratum only ---------------------------------
        pred_path = run / "preds_test_real.parquet"
        if pred_path.exists():
            pred = pd.read_parquet(pred_path)
            if "notes" not in pred.columns:
                pred["notes"] = pred["id"].map(truth["notes"]).fillna("")
            strata = results.get("test_real", {}).get("strata", {})
            for strat, block in strata.items():
                sub = pred[pred["real_stratum"] == strat].reset_index(drop=True)
                if sub.empty:
                    continue
                # A tiny stratum silently missing from the prediction file is
                # a bug, not something to skip.
                if len(sub) != block["n_rows"]:
                    print(f"    WARNING {run_name}/{strat}: predictions={len(sub)} "
                          f"but evaluation rows={block['n_rows']}")
                m = verdict_metrics(sub, sub["pred_verdict"].tolist())
                scores = sub["prob_scam"].tolist() if "prob_scam" in sub else None
                m["probability"] = probability_metrics(
                    sub["verdict"].tolist(), scores)
                m["tactics"] = (tactic_metrics(
                    sub, sub["pred_tags"].tolist(),
                    validated_tags=block.get("validated_tags"))
                    if tags_trained else _empty_tactics())
                if m["macro_f1"] is None:
                    m["macro_f1_bootstrap_mean"] = None
                    m["macro_f1_ci95"] = [None, None]
                else:
                    mean, lo, hi = bootstrap_macro_f1(
                        sub["verdict"].tolist(), sub["pred_verdict"].tolist())
                    m["macro_f1_bootstrap_mean"] = mean
                    m["macro_f1_ci95"] = [lo, hi]
                block["models"][run_name] = m
                added += 1
        print(f"  {run_name}: added {added} prediction table(s)"
              f"{'' if tags_trained else ' (tags NOT trained -> tag metrics n/a)'}")


def _selection_value(row: dict) -> float:
    """The validation-only score that chose the checkpoint (Phase 3.6).

    New logs carry `selection_score` = mean(val_synthetic macro-F1, val_real
    balanced accuracy). Old logs only have `val_macro_f1`; we fall back to it
    so reading a pre-3.6 run still works, without ever touching test metrics.
    """
    value = row.get("selection_score")
    if value is not None:
        return float(value)
    return float(row.get("val_macro_f1") or 0.0)


def _log_rows(run: Path) -> list[dict]:
    log = run / "training_log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _best_row(run: Path) -> dict | None:
    """Best epoch as chosen by the validation score (never test metrics)."""
    rows = _log_rows(run)
    return max(rows, key=_selection_value) if rows else None


def _best_epoch(run: Path):
    row = _best_row(run)
    return row["epoch"] if row else None


def _best_val(run: Path):
    row = _best_row(run)
    return _selection_value(row) if row else None


def gap_check(results: dict) -> list[str]:
    """Synthetic test_b -> combined real English, macro-F1 AND AUROC."""
    lines = ["## Gap check (per model)",
             "",
             "`test_b` uses unseen synthetic families. The real reference is "
             f"`{COMBINED_STRATUM}` - {COMBINED_TITLE}. A large drop means the "
             "model learned the generator, not scams.",
             "",
             "| model | test_a macro | test_b macro | a-b | test_b AUROC "
             "| real macro | b-real | real AUROC | AUROC gap |",
             "|---|---|---|---|---|---|---|---|---|"]
    combined = results.get("test_real", {}).get("strata", {}).get(
        COMBINED_STRATUM, {}).get("models", {})
    for name in results.get("test_b", {}).get("models", {}):
        tb = results["test_b"]["models"][name]
        b = tb["macro_f1"]
        a = results.get("test_a", {}).get("models", {}).get(name, {}).get("macro_f1")
        c = combined.get(name, {}).get("macro_f1")
        b_auc = (tb.get("probability") or {}).get("auroc")
        auc = (combined.get(name, {}).get("probability") or {}).get("auroc")
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {_fmt(a)} | {_fmt(b)} "
            f"| {_fmt(None if a is None or b is None else a - b)} "
            f"| {_fmt(b_auc)} | {_fmt(c)} "
            f"| {_fmt(None if b is None or c is None else b - c)} "
            f"| {_fmt(auc)} "
            f"| {_fmt(None if b_auc is None or auc is None else b_auc - auc)} |")
    origin_auc = (results.get("test_real", {}).get("strata", {})
                  .get(COMBINED_STRATUM, {}).get("source_origin", {})
                  .get("test_real", {}).get("auroc"))
    if origin_auc is not None:
        # The `real AUROC` column is a real_english_combined number too, so the
        # source-origin reference and warning must sit next to it as well.
        lines += [f"> The `real AUROC` column is on `{COMBINED_STRATUM}`. The "
                  "source-origin classifier (TEXT only) reaches AUROC "
                  f"**{origin_auc:.3f}** on the same rows. "
                  f"**{ORIGIN_WARNING}**", ""]
    lines += ["`test_a` is a 10% hold-out of TRAIN clusters (Phase 3.6), so "
              "both classes exist and the seen-wording score is real. "
              "`test_b` uses UNSEEN families, so `a - b` is the memorisation gap "
              "and `b - real` is the domain gap; the AUROC gap is the same "
              "question for ranking quality.", "",
              f"> {HEADLINE_NOTE}"]
    return lines


def write_markdown(results: dict) -> None:
    out = Path(data_path("results_md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    gate = results.get("meta", {}).get("e4_gate", {})
    md = ["# ScamLens - evaluation results (BINARY verdict)", "",
          f"> {HEADLINE_NOTE}", "",
          "> `suspicious` is an UNSURE BAND, not a class. It is never trained or "
          "scored here; Phase 4 uses the holdout for confidence-based abstention.", "",
          f"> Checkpoint selection: **{SELECTION_NOTE}**. Test splits are "
          "observation only.", ""]

    # ---- data provenance (Phase 3.5), stated before any score -------------
    prov = results.get("meta", {}).get("provenance")
    if prov:
        md += ["## Data provenance", "",
               f"Training mix: **{prov['mix']}**. Every experiment reports its "
               "own mix; `mix_synth_only` excludes real rows.", "",
               "| split | rows | sources |", "|---|---:|---|"]
        for split, info in prov["splits"].items():
            src = ", ".join(f"{k} {v}" for k, v in info["sources"].items())
            md.append(f"| `{split}` | {info['rows']:,} | {src} |")
        md += ["", "**Real script_style distribution** "
               f"({prov['real_rows']:,} real rows): {prov['real_style']}. "
               f"Roman Hinglish = **{prov['real_hinglish']}** rows "
               f"({prov['real_hinglish'] / max(prov['real_rows'], 1):.2%}).", "",
               f"External scam styles: {prov['external_style']}  |  "
               f"public ham styles: {prov['ham_style']}", "",
               "**Per-tag training support** (rows containing the tag; the "
               "splitter and tests require ≥100 in train, ≥1 in val and "
               "test_b):", "",
               "| tag | train | val | test_a | test_b | real test |",
               "|---|---:|---:|---:|---:|---:|"]
        for row in prov["tag_support"]:
            md.append(f"| {row['tag']} | {row['train']} | {row['val']} | "
                      f"{row['test_a']} | {row['test_b']} | "
                      f"{row['test_real']} |")
        md += ["", "> **`test_a` (Phase 3.6):** 10% of the TRAIN clusters, held "
               "out whole and BY CLUSTER, drawn from both scam and safe pools. "
               "It therefore has both classes and a real macro-F1, and it "
               "measures seen-template wording. `test_b` is the unseen-family "
               "split.", ""]

    # ---- nearest-neighbour leakage (Phase 3.6) ---------------------------
    nn = results.get("meta", {}).get("nn_similarity")
    md += ["## Nearest-neighbour leakage audit (train similarity)", ""]
    if not nn:
        md += ["> **NOT RUN.** `results/nn_sim.parquet` is missing. Run "
               "`python -m src.nn_sim` before quoting any test score: without "
               "it we cannot say whether test rows have near-copies in train.",
               ""]
    else:
        md += [f"> {nn['note']}", "",
               "| stratum | rows | sim > 0.9 | 0.7-0.9 | < 0.7 | median sim |",
               "|---|---:|---:|---:|---:|---:|"]
        for r in nn["strata"]:
            median = "n/a" if r.get("median_sim") is None else f"{r['median_sim']:.3f}"
            md.append(f"| `{r['stratum']}` | {r['n_rows']:,} | "
                      f"{r['share_gt_0.9']:.1%} | {r['share_0.7_0.9']:.1%} | "
                      f"{r['share_lt_0.7']:.1%} | {median} |")
        md.append("")
        share = nn.get("real_test_share_gt_0.9")
        if nn.get("real_test_mostly_near_copies"):
            md += [f"> **{share:.1%} of the {nn['real_test_rows']:,} real-test "
                   "rows have a train row with cosine similarity > 0.9. That "
                   "means real-test scores mostly measure near-copy matching, "
                   "not scam detection.**", ""]
        elif share is not None:
            md += [f"> {share:.1%} of the {nn['real_test_rows']:,} real-test "
                   "rows have a train near-copy (sim > 0.9), so this is not a "
                   "copy-matching test set.", ""]

    # ---- E4 gate, stated before any table --------------------------------
    md += ["## Real-data gate (E4)", ""]
    md.append(f"- in-domain real messages (`user_real_scam` + `user_real_safe`): "
              f"**{gate.get('in_domain', '?')}**")
    md.append(f"- `user_real_safe` hard negatives: **{gate.get('user_real_safe', '?')}**")
    md.append(f"- requirement: {gate.get('requirement', E4_NOTE)}")
    if gate.get("passed"):
        md.append("- **E4 may run.**")
    else:
        md += ["- **E4 SKIPPED.** A cluster-grouped CV on 10 real rows would be "
               "noise presented as a result. Real performance below comes from "
               "the public strata only; the two user strata are too small."]
    md.append("")

    # ---- model key -------------------------------------------------------
    runs = results.get("meta", {}).get("runs", {})
    if runs:
        md += ["### Transformer runs included", "",
               "Selection score = mean(val_synthetic macro-F1, val_real "
               "balanced accuracy). The combined val macro-F1 is shown for "
               "observation only.", "",
               "| run | model | tags trained | best epoch (val) | selection "
               "score | val synth macro-F1 | val real balanced acc | combined "
               "val macro-F1 (observation) |",
               "|---|---|---|---|---|---|---|---|"]
        for name, info in runs.items():
            md.append(f"| {name} | {info.get('model_name')} | "
                      f"{'yes' if info.get('tags_trained') else 'no'} | "
                      f"{info.get('best_epoch')} | "
                      f"{_fmt(info.get('selection_score'))} | "
                      f"{_fmt(info.get('val_synthetic_macro_f1'))} | "
                      f"{_fmt(info.get('val_real_balanced_accuracy'))} | "
                      f"{_fmt(info.get('val_macro_f1'))} |")
        md.append("")

    for split in SYNTHETIC_SPLITS:
        if split not in results:
            continue
        block = results[split]
        label = "combined val (synthetic + real)" if split == "val" else "synthetic"
        md += [f"## test set: `{split}` ({label})", "",
               f"{block['n_rows']} rows, scam rate {block['scam_rate']:.1%}, "
               f"style mix: {block['style_mix']}", ""]
        if block.get("note"):
            md += [f"> {block['note']}", ""]
        if block.get("single_class"):
            md += [f"> **WARNING:** this split has ONE class only. Macro-F1 is "
                   "n/a (shown as n/a), not 0.5. Read recall / FPR instead.", ""]
        md += _model_table(block["models"])
        md.append("")
        if split == "val":
            for part in ("val_synthetic", "val_real"):
                if part not in results:
                    continue
                pb = results[part]
                md += [f"### `{part}` - {pb['n_rows']} rows, scam rate "
                       f"{pb['scam_rate']:.1%}", ""]
                md += _model_table(pb["models"])
                md.append("")
        for name in ("verdict_lr_plus", "keyword"):
            if name not in block["models"]:
                continue
            r = block["models"][name]
            md += [f"### {name} - confusion matrix on `{split}`", "",
                   "rows = gold, cols = predicted (order: safe, scam)", "", "```",
                   "           pred_safe pred_scam"]
            for cls, row in zip(VERDICT_LABELS, r["confusion_matrix"]):
                md.append(f"  gold_{cls:<10} {row[0]:>8} {row[1]:>9}")
            md += ["```", ""]

    # ---- test_real per stratum -------------------------------------------
    tr = results.get("test_real", {})
    md += ["## test set: `test_real` (REAL - reported PER STRATUM)", "",
           "> There is deliberately **no blended real number**. A stratum with "
           f"fewer than {MIN_STRATUM_ROWS} rows is marked 'too small for claims'.", "",
           "> A stratum with only ONE class (`external_scam` is all scam, "
           "`public_ham_english` is all safe) has **macro-F1 = n/a**. It is not "
           "0.5 and not 0.0: there is no second class to average. Read **scam "
           "recall** and **FPR(safe)** there. The combined English block below is "
           "the only place where a real macro-F1 is defined.", ""]
    if tr.get("unstratified_rows"):
        md.append(f"WARNING: {tr['unstratified_rows']} test_real row(s) have no "
                  "real_stratum and are not scored anywhere.")
        md.append("")
    for strat, block in tr.get("strata", {}).items():
        flag = " **TOO SMALL FOR CLAIMS**" if block["too_small_for_claims"] else ""
        title = block.get("title")
        heading = f"### `{strat}` - {block['n_rows']} rows{flag}"
        if title:
            heading += f"\n\n> **{title}.**"
        md += [heading, "",
               f"scam rate: {block['scam_rate']:.1%} | style mix: {block['style_mix']}", ""]
        origin = block.get("source_origin") or {}
        origin_info = origin.get("test_real", {}) or {}
        origin_auc = origin_info.get("auroc")
        md += _model_table(block["models"], origin_auroc=origin_auc)
        if origin_auc is not None:
            md += ["", f"> **{ORIGIN_WARNING}** The source-origin classifier "
                   "(TEXT only: IMC25 scam vs UCI ham) reaches AUROC "
                   f"**{origin_auc:.3f}** on the same real test rows "
                   f"(accuracy {_fmt(origin_info.get('accuracy'))}, "
                   f"n={origin_info.get('n', 0):,}). Compare it with each row's "
                   "AUROC above.", ""]
        md += ["", "Per-tag F1 (only tags with "
                   f">= {MIN_TAG_SUPPORT} real positives are shown):", ""]
        md += _tag_table(block)
        md.append("")
        if "verdict_lr" in block["models"]:
            r = block["models"]["verdict_lr"]
            md += ["Confusion matrix (TF-IDF + LR):", "", "```",
                   "           pred_safe pred_scam"]
            for cls, row in zip(VERDICT_LABELS, r["confusion_matrix"]):
                md.append(f"  gold_{cls:<10} {row[0]:>8} {row[1]:>9}")
            md += ["```", ""]

    md += gap_check(results)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWROTE {out}")


def main() -> None:
    results = evaluate_all()
    add_transformer_runs(results, load_messages())
    js = Path(data_path("results_json"))
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    print(f"WROTE {js}")
    write_markdown(results)


if __name__ == "__main__":
    main()
