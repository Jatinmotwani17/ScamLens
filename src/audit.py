"""
src/audit.py - the shortcut audit: is the model (or the DATA) cheating?

Phase 2.5 makes this audit stricter. It now answers four questions:

  1. STYLE / LABEL BALANCE: for `train` and every test split, what is the
     scam rate inside each `script_style` and each `source`? A style or source
     above 90% or below 10% scam is a label shortcut ("devanagari always
     means scam"). Those lines are FLAGGED loudly.
  2. METADATA-ONLY BASELINES: how well does a model do using ONLY
     `script_style`, or ONLY `source`? If either scores high, the generator
     leaks the answer through metadata.
  3. SINGLE-FEATURE MODELS: length, URL, digits, etc. - one cheap feature at
     a time. Above the config threshold is flagged.
  4. WHAT THE MODEL LEARNED: top TF-IDF words for the scam class, plus the
     most-confident wrong predictions on test_b (unseen families).

Everything is BINARY (scam vs safe). "suspicious" is an unsure band and is
never part of this audit.

Run:  python -m src.audit
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.baselines import load_models, predict_verdicts
from src.config import CFG, REAL_STRATA, SEED, data_path
from src.io_utils import load_messages

AUDIT_CFG = CFG["audit"]
FLAG_THRESHOLD = float(AUDIT_CFG["shortcut_flag_threshold"])
SCAM_HIGH = float(AUDIT_CFG["scam_rate_high"])
SCAM_LOW = float(AUDIT_CFG["scam_rate_low"])
MIN_STYLE_ROWS = int(AUDIT_CFG.get("min_rows_for_style_flag", 10))
BINARY_LABELS = ["safe", "scam"]

# Bank / payment brand words used by the "has bank word" feature.
BANK_WORDS = ("sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bank",
              "paytm", "phonepe", "gpay", "upi")


def macro_f1_binary(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=BINARY_LABELS,
                          average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# 1. style / source label balance
# ---------------------------------------------------------------------------
def _balance_lines(df: pd.DataFrame, title: str, column: str) -> tuple[list[str], list[str]]:
    """Return (markdown lines, flagged descriptions) for one metadata column."""
    lines = [f"**{title}**", "", f"| {column} | rows | scam | scam rate | flag |",
             "|---|---|---|---|---|"]
    flagged: list[str] = []
    for value, sub in df.groupby(df[column].astype(str)):
        n = len(sub)
        if n == 0:
            continue
        rate = float((sub["verdict"] == "scam").mean())
        flag = ""
        if n < MIN_STYLE_ROWS:
            # Too few rows to call this a shortcut - say so instead of
            # flooding the report with 2-row "100% scam" alarms.
            flag = f"(n<{MIN_STYLE_ROWS}: not flagged)"
        elif rate > SCAM_HIGH or rate < SCAM_LOW:
            flag = f"FLAGGED ({rate:.0%} scam)"
            flagged.append(f"{column}={value} ({n} rows, {rate:.0%} scam)")
        lines.append(f"| {value} | {n} | {int((sub['verdict'] == 'scam').sum())} "
                     f"| {rate:.1%} | {flag} |")
    lines.append("")
    return lines, flagged


def audit_style_label_balance(df: pd.DataFrame) -> list[str]:
    """Scam rate by script_style and source for train + every test split."""
    lines = ["## 1. Style / label balance audit (BINARY)",
             "",
             f"A style or source is FLAGGED when its scam rate is >{SCAM_HIGH:.0%} "
             f"or <{SCAM_LOW:.0%}. Flagged does not always mean broken: on "
             "`test_real`, source IS the stratum (external scams vs public ham), "
             "so a 100% rate there is structural, not a model shortcut.", ""]
    all_flags: list[str] = []

    # train (Phase 3.5: synthetic + real external + real ham)
    train = df[df["split"] == "train"]
    lines += [f"### `train` ({len(train)} rows)", ""]
    for col in ("script_style", "source"):
        block, flags = _balance_lines(train, f"train by {col}", col)
        lines += block
        all_flags += [f"train: {f}" for f in flags]

    # val / test_a / test_b
    for split in ("val", "test_a", "test_b"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        lines += [f"### `{split}` ({len(sub)} rows)", ""]
        for col in ("script_style", "source"):
            block, flags = _balance_lines(sub, f"{split} by {col}", col)
            lines += block
            all_flags += [f"{split}: {f}" for f in flags]

    # test_real, per stratum (never blended)
    real = df[df["split"] == "test_real"]
    if not real.empty:
        lines += ["### `test_real` per stratum", ""]
        for strat in REAL_STRATA:
            sub = real[real["real_stratum"] == strat]
            if sub.empty:
                continue
            lines += [f"#### `{strat}` ({len(sub)} rows)", ""]
            for col in ("script_style", "source"):
                block, flags = _balance_lines(sub, f"{strat} by {col}", col)
                lines += block
                all_flags += [f"{strat}: {f}" for f in flags]

    if all_flags:
        lines += ["**FLAGGED style/source groups:**", ""]
        lines += [f"* {f}" for f in all_flags]
    else:
        lines.append("No style or source crossed the 10% / 90% scam-rate band.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 2. metadata-only baselines (the two shortcut models)
# ---------------------------------------------------------------------------
def audit_shortcut_baselines(models: dict, df: pd.DataFrame) -> list[str]:
    """Macro-F1 of style_only and source_only on every test subset."""
    lines = ["## 2. Metadata-only shortcut baselines", "",
             "These models see ONE column and no message text. High scores mean "
             "the data leaks the label through metadata.", "",
             "| split | rows | style_only macro-F1 | source_only macro-F1 |",
             "|---|---|---|---|"]
    subsets: list[tuple[str, pd.DataFrame]] = []
    for split in ("val", "test_a", "test_b"):
        subsets.append((split, df[df["split"] == split]))
    real = df[df["split"] == "test_real"]
    for strat in REAL_STRATA:
        sub = real[real["real_stratum"] == strat]
        if not sub.empty:
            subsets.append((f"test_real/{strat}", sub))

    for name, sub in subsets:
        if sub.empty:
            continue
        row = [name, str(len(sub))]
        for model_name in ("style_only_plus", "source_only_plus"):
            if model_name not in models:
                row.append("n/a")
                continue
            pred = predict_verdicts(models[model_name], sub["text"].tolist(), sub)
            row.append(f"{macro_f1_binary(sub['verdict'].tolist(), pred):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "If style_only or source_only is close to the text baselines "
                  "on a SYNTHETIC split, the generator still has a metadata "
                  "shortcut. On test_real, source is the stratum definition, so "
                  "a high source-only score there is expected and not a model win.", ""]

    # Phase 3.5 gate: any shortcut above 0.65 on COMBINED val is a red flag.
    val = df[df["split"] == "val"]
    if not val.empty:
        for mname, label in (("style_only_plus", "style_only"),
                             ("source_only_plus", "source_only")):
            if mname not in models:
                continue
            pred = predict_verdicts(models[mname], val["text"].tolist(), val)
            score = macro_f1_binary(val["verdict"].tolist(), pred)
            if score > 0.65:
                lines.append(f"**FLAGGED: {label} = {score:.3f} on combined val "
                             "(> 0.65).**")
                if label == "source_only":
                    lines.append("  This is structural: `source` literally names "
                                 "the dataset (IMC25 scam vs UCI ham), so it "
                                 "cannot be rebalanced by adding synthetic rows "
                                 "without fabricating real data. Fix requires a "
                                 "mixed-origin real dataset.")
                else:
                    lines.append("  Fix direction: add synthetic Hinglish SAFE "
                                 "and synthetic English SCAM rows so every "
                                 "script_style has both classes.")
                lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 2b. dataset-origin classifier (Phase 3.5 SOURCE SHORTCUT CHECK)
def source_origin_scores(df: pd.DataFrame) -> dict:
    """Fit a TEXT-ONLY classifier that predicts which public corpus a real
    message came from (IMC25 scam vs UCI ham), then score it.

    Phase 3.6: this used to live entirely inside the markdown report. It is
    now a reusable function so src/evaluate.py can print the SAME number next
    to every real_english_combined AUROC. Why it matters: if a text model can
    tell the two corpora apart almost perfectly, then a scam/safe model on the
    same rows can score high by recognising the dataset (era, style, wording)
    instead of the scam intent.

    Returns a dict with `available`, `n_train`, and for both `val` and
    `test_real`: `n`, `accuracy`, `auroc`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    origins = ["external_reported", "public_ham_eval"]
    parts = {}
    for split in ("train", "val", "test_real"):
        sub = df[(df["split"] == split) & df["source"].isin(origins)]
        parts[split] = sub.reset_index(drop=True)

    result: dict = {"available": False, "n_train": int(len(parts["train"])),
                    "val": {"n": 0, "accuracy": None, "auroc": None},
                    "test_real": {"n": 0, "accuracy": None, "auroc": None}}
    train = parts["train"]
    if train["source"].nunique() < 2 or len(train) < 20:
        return result

    # Same tiny model as the audit report: char n-grams, balanced classes,
    # seeded so the number is reproducible.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  min_df=2, max_features=40_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=SEED)),
    ])
    pipe.fit(train["text"].astype(str).tolist(),
             (train["source"] == "external_reported").astype(int).tolist())
    result["available"] = True

    for split in ("val", "test_real"):
        sub = parts[split]
        result[split]["n"] = int(len(sub))
        if sub.empty or sub["source"].nunique() < 2:
            continue
        yy = (sub["source"] == "external_reported").astype(int).to_numpy()
        prob = pipe.predict_proba(sub["text"].astype(str).tolist())[:, 1]
        result[split]["accuracy"] = float(((prob >= 0.5).astype(int) == yy).mean())
        result[split]["auroc"] = float(roc_auc_score(yy, prob))
    return result


def audit_source_origin_classifier(df: pd.DataFrame) -> list[str]:
    """Can a char-ngram model tell IMC25 scam from UCI ham, ignoring labels?

    If yes, the real mix has a corpus/era/style confound. This is different
    from `source_only` (which is given the answer in a column): this model sees
    TEXT only and predicts which public dataset the message came from.
    """
    scores = source_origin_scores(df)
    if not scores["available"]:
        return ["## 2b. Source-origin classifier", "",
                "(not enough real train rows from both origins)", ""]

    lines = ["## 2b. Source-origin classifier (TEXT only)", "",
             "Predicts *which public dataset* a real message came from "
             "(IMC25 scam vs UCI ham). High accuracy means the real mix has a "
             "corpus/era/style confound that a model can exploit.", "",
             "| rows | accuracy | AUROC |", "|---|---|---|"]
    for split in ("val", "test_real"):
        info = scores[split]
        if info["auroc"] is None:
            lines.append(f"| {split} ({info['n']}) | n/a | n/a |")
        else:
            lines.append(f"| {split} ({info['n']}) | {info['accuracy']:.3f} "
                         f"| {info['auroc']:.3f} |")
    lines += ["", "If accuracy is above 0.65, the real-mix style/source shortcut "
                  "is real. It cannot be removed by relabeling synthetic rows "
                  "as real; it needs a mixed-origin real dataset.", ""]
    return lines


# ---------------------------------------------------------------------------
# 3. single-feature shortcut check
# ---------------------------------------------------------------------------
def extract_features(text: str) -> dict:
    """Cheap, shallow features - exactly the kind a generator can leak."""
    t = str(text)
    low = t.lower()
    letters = [c for c in t if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper())
    return {
        "msg_length": len(t),
        "has_url": float("<URL>" in t or "http" in low or "www." in low),
        "has_digits": float(any(c.isdigit() for c in t)),
        "has_devanagari": float(any("\u0900" <= c <= "\u097F" for c in t)),
        "has_bank_word": float(any(w in low for w in BANK_WORDS)),
        "n_exclamations": t.count("!"),
        "caps_ratio": caps / max(len(letters), 1),
    }


def audit_single_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Train one logistic model per single feature; flag suspicious ones."""
    from sklearn.linear_model import LogisticRegression

    lines = ["## 3. Single-feature shortcut check (macro-F1 on test_b)", ""]
    feats_train = pd.DataFrame([extract_features(t) for t in train["text"]])
    feats_test = pd.DataFrame([extract_features(t) for t in test["text"]])
    y_train, y_true = train["verdict"], test["verdict"]

    flagged = []
    for col in feats_train.columns:
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(feats_train[[col]].to_numpy(), y_train)
        score = macro_f1_binary(y_true, clf.predict(feats_test[[col]].to_numpy()))
        mark = ""
        if score >= FLAG_THRESHOLD:
            mark = f"  <-- FLAGGED (>= {FLAG_THRESHOLD:.2f}: possible shortcut!)"
            flagged.append(col)
        lines.append(f"* **{col}**: macro-F1 = {score:.3f}{mark}")
    lines.append("")
    if flagged:
        lines.append(f"FLAGGED features: {', '.join(flagged)} - the generator "
                     "leaks through these; make the generator harder.")
    else:
        lines.append("No single feature cleared the threshold - good sign.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 4. what did the real model learn?
# ---------------------------------------------------------------------------
def audit_top_features(models: dict, n: int = 20) -> list[str]:
    """Top +/- word features of verdict_lr for the scam class."""
    lines = [f"## 4. Top {n} word features of verdict_lr (scam class)", ""]
    bundle = models.get("verdict_lr_plus") or models.get("verdict_lr")
    if not bundle:
        return lines + ["(verdict_lr_plus not available)"]
    pipe = bundle["pipeline"]
    vec = pipe.named_steps["tfidf"].transformer_list[0][1]   # the word vectorizer
    clf = pipe.named_steps["clf"]
    classes = list(clf.classes_)
    if "scam" not in classes:
        return lines + ["(no scam class in this model)"]
    scam_idx = classes.index("scam")
    vocab = np.array(vec.get_feature_names_out())
    # sklearn stores ONE coefficient row for binary problems; its sign points
    # towards classes_[1]. For multiclass it stores one row per class. Handle
    # both so this audit never crashes on a legitimate binary model.
    coef = clf.coef_
    if coef.shape[0] == 1:
        signed = coef[0] if classes[1] == "scam" else -coef[0]
    else:
        signed = coef[scam_idx]
    # The TfidfVectorizer only owns the FIRST block of the FeatureUnion, so
    # the word coefficients are the first len(vocab) entries.
    coefs = signed[:len(vocab)]
    top_pos = vocab[np.argsort(coefs)[-n:]][::-1]
    top_neg = vocab[np.argsort(coefs)[:n]]
    lines.append("words that push TOWARDS scam:")
    lines.append("  " + ", ".join(top_pos.tolist()))
    lines.append("")
    lines.append("words that push AWAY from scam:")
    lines.append("  " + ", ".join(top_neg.tolist()))
    lines.append("")
    return lines


def audit_wrong_predictions(models: dict, df: pd.DataFrame,
                            n: int = None) -> list[str]:
    """The n most-confident wrong verdict predictions on test_b, with text."""
    n = n or int(AUDIT_CFG["top_wrong_predictions"])
    lines = [f"## 5. Top {n} most-confident WRONG predictions on test_b", ""]
    test = df[df["split"] == "test_b"].reset_index(drop=True)
    bundle = models.get("verdict_lr_plus") or models.get("verdict_lr")
    if test.empty or not bundle:
        return lines + ["(test_b or verdict_lr_plus not available)"]
    pipe = bundle["pipeline"]
    texts = test["text"].astype(str).tolist()
    proba = pipe.predict_proba(texts)
    classes = list(pipe.named_steps["clf"].classes_)
    preds = [classes[i] for i in proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    wrong = [(conf[i], preds[i], test["verdict"].iloc[i], texts[i])
             for i in range(len(test)) if preds[i] != test["verdict"].iloc[i]]
    wrong.sort(key=lambda x: -x[0])

    grouped: dict[str, dict] = {}
    for c, pred, gold, text in wrong:
        key = f"{pred}|{gold}|{text}"
        if key in grouped:
            grouped[key]["count"] += 1
        else:
            grouped[key] = {"conf": c, "pred": pred, "gold": gold,
                            "text": text, "count": 1}
    distinct = sorted(grouped.values(), key=lambda d: -d["conf"])

    for g in distinct[:n]:
        # PRIVACY: never store the message itself - only its shape.
        snippet = (f"[text withheld - {len(str(g['text']).split())} words, "
                   f"{len(str(g['text']))} chars]")
        times = f"  (identical message appeared {g['count']}x)" if g["count"] > 1 else ""
        lines.append(f"* conf={g['conf']:.2f} predicted **{g['pred']}**, "
                     f"gold **{g['gold']}**{times}")
        lines.append(f"    {snippet}")
    if not wrong:
        lines.append("(no wrong predictions on test_b!)")
    lines.append("")
    return lines


def main() -> None:
    # Windows consoles default to cp1252, which cannot print Devanagari.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"(seed={SEED}) src/audit.py - shortcut audit (BINARY)")
    print("=" * 60)
    df = load_messages()
    if "suspicious" in set(df["verdict"]):
        raise SystemExit("suspicious rows are in the MAIN table - run build_dataset first.")
    models = load_models()

    train = df[df["split"] == "train"]
    test_b = df[df["split"] == "test_b"]

    lines: list[str] = ["# ScamLens - shortcut audit (Phase 3.5, BINARY)", ""]
    lines += audit_style_label_balance(df)
    lines += audit_shortcut_baselines(models, df)
    lines += audit_source_origin_classifier(df)
    lines += audit_single_features(train.reset_index(drop=True),
                                   test_b.reset_index(drop=True))
    lines += audit_top_features(models, n=int(AUDIT_CFG["top_features"]))
    lines += audit_wrong_predictions(models, df)

    report = "\n".join(lines)
    print("\n" + report)
    out = data_path("results_md")
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + report + "\n")
    print(f"\n(appended to {out})")


if __name__ == "__main__":
    main()
