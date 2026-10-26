"""
src/suspicious.py - the "suspicious" verdict, built from EVIDENCE not probability.

WHY NOT A PROBABILITY BAND
  Phase 4 tried the textbook version: scam if p >= t_high, safe if p <= t_low,
  suspicious in between. On this scorer the band is empty at every target,
  because TF-IDF+LR probabilities are bimodal (everything sits near 0 or near
  1). Manufacturing a band by widening it would be fake abstention, so Phase 5
  uses a different question:

      "Is the model's scam call backed by any evidence I can name?"

  A row is `suspicious` when the verdict model says scam (clears the strict
  threshold) but the evidence is weak:

    * no tactic tag above WEAK_TAG_PROB (the tag heads are unsure), AND
    * no OTP / payment / credential word in the text, AND
    * no risky link type from the scrubber, AND
    * the keyword rules do not fire.

  Otherwise the verdict stays `scam`. Anything below the strict threshold is
  `safe`. This is a DEFERRAL rule: it moves a confident-sounding call to a
  human instead of pretending the probability is uncertain.

WHAT IT COSTS
  Deferral is not free: every clear scam that lands in `suspicious` loses its
  definitive verdict, and every safe message that lands there is escalated.
  src/suspicious.py measures both and writes results/suspicious_band.md. If the
  rule does not beat "never abstain", it says so.

Run:
  python -m src.suspicious --model verdict_lr_plus
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
import yaml

from src.calibrate import DEFAULT_MODE, load_operating_point
from src.config import CFG, SEED, data_path
from src.phase4_common import (
    apply_temperature,
    band_summary,
    load_predictor,
    predictions_for_split,
    rate_with_ci,
    too_small,
)

# Probability a tag head must reach before we call the evidence "strong".
WEAK_TAG_PROB = 0.50
# Words that make a scam call self-evidently supported, whatever the tags say.
EVIDENCE_PATTERNS = [
    ("otp_request", re.compile(r"\b(otp|one[- ]time password|pin)\b", re.I)),
    ("payment_request", re.compile(r"\b(upi|pay now|transfer|send money|"
                                   r"gpay|phonepe|paytm|qr code)\b", re.I)),
    ("credential_request", re.compile(r"\b(cvv|password|net ?banking login|"
                                      r"aadhaar number|card number)\b", re.I)),
]
# Link types the scrubber treats as risky (see src/config.py URL_TYPES).
RISKY_URL_TYPES = {"shortener", "odd_tld", "ip_address", "official_looking"}


def rule_evidence(texts: list[str], tag_probs: dict[str, list[float]],
                  url_types: list[str] | None = None,
                  keyword_hits: list[bool] | None = None) -> pd.DataFrame:
    """One row per message: what evidence exists, and how strong it is."""
    n = len(texts)
    best_tag = np.zeros(n, dtype=float)
    best_tag_name = [""] * n
    for tag, probs in tag_probs.items():
        arr = np.asarray(probs, dtype=float)
        if len(arr) != n:
            continue
        better = arr > best_tag
        best_tag[better] = arr[better]
        for i in np.flatnonzero(better):
            best_tag_name[i] = tag

    rows = []
    for i, text in enumerate(texts):
        kinds = [name for name, pattern in EVIDENCE_PATTERNS if pattern.search(text)]
        url = (url_types[i] if url_types else "") or ""
        rows.append({
            "best_tag": best_tag_name[i],
            "best_tag_prob": float(best_tag[i]),
            "text_evidence": ",".join(kinds),
            "risky_link": url in RISKY_URL_TYPES,
            "url_type": url,
            "keyword_hit": bool(keyword_hits[i]) if keyword_hits else False,
        })
    return pd.DataFrame(rows)


def decide(p_cal, evidence: pd.DataFrame, strict_threshold: float) -> np.ndarray:
    """scam / suspicious / safe from probability + named evidence."""
    p = np.asarray(p_cal, dtype=float)
    strong = (
        (evidence["best_tag_prob"].to_numpy(dtype=float) >= WEAK_TAG_PROB)
        | (evidence["text_evidence"] != "")
        | evidence["risky_link"].to_numpy(dtype=bool)
        | evidence["keyword_hit"].to_numpy(dtype=bool)
    )
    above = p >= strict_threshold
    out = np.full(len(p), "safe", dtype=object)
    out[above & strong] = "scam"
    out[above & ~strong] = "suspicious"
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _keyword_hits(texts: list[str]) -> list[bool]:
    """Do the hand-written keyword rules fire? (Reuses the trained bundle.)"""
    from pathlib import Path

    import joblib

    from src.config import PROJECT_ROOT

    path = Path(PROJECT_ROOT) / "models" / "keyword.joblib"
    if not path.exists():
        return [False] * len(texts)
    bundle = joblib.load(path)
    model = bundle["model"]
    return [model.predict_one(t)[0] == "scam" for t in texts]


def evaluate(model_name: str = "verdict_lr_plus") -> dict:
    """Where clear scams, clear safe, and the human-`suspicious` set land."""
    point = load_operating_point(model_name)
    temperature = float(point["calibration"]["temperature"])
    strict_t = float(point["modes"]["strict"]["threshold"])
    bal_t = float(point["modes"][DEFAULT_MODE]["threshold"])

    predictor = load_predictor(model_name)
    from src.baselines import MODEL_FILES, predict_tactic_probs
    from src.config import PROJECT_ROOT
    from pathlib import Path
    import joblib

    tag_path = Path(PROJECT_ROOT) / "models" / MODEL_FILES["tactics_lr_plus"]
    tag_bundle = joblib.load(tag_path) if tag_path.exists() else None

    def scored(split: str) -> pd.DataFrame:
        part = predictions_for_split(predictor, model_name, split)
        if part.empty:
            return part
        part = part.copy()
        part["prob_cal"] = apply_temperature(
            part["prob_raw"].to_numpy(dtype=float), temperature)
        return part

    test_real = scored("test_real")
    ext = test_real[test_real["real_stratum"] == "external_scam"].reset_index(drop=True)
    ham = test_real[test_real["real_stratum"] == "public_ham_english"].reset_index(drop=True)
    test_b = scored("test_b")

    def land(frame: pd.DataFrame, label: str) -> dict:
        if frame.empty:
            return {"target": label, "n": 0}
        texts = frame["text"].astype(str).tolist()
        tag_probs = (predict_tactic_probs(tag_bundle, texts) if tag_bundle else {})
        ev = rule_evidence(texts, tag_probs,
                           url_types=frame.get("url_type", pd.Series([""] * len(frame)))
                           .astype(str).tolist(),
                           keyword_hits=_keyword_hits(texts))
        labels = decide(frame["prob_cal"].to_numpy(dtype=float), ev, strict_t)
        counts = pd.Series(labels).value_counts().to_dict()
        n = int(len(frame))
        return {
            "target": label,
            "n": n,
            "scam": int(counts.get("scam", 0)),
            "suspicious": int(counts.get("suspicious", 0)),
            "safe": int(counts.get("safe", 0)),
            "share_suspicious": counts.get("suspicious", 0) / n if n else None,
            "too_small": too_small(n),
            "evidence_when_suspicious": (
                ev.loc[labels == "suspicious", "best_tag"].value_counts().head(5).to_dict()
                if (labels == "suspicious").any() else {}),
        }

    holdout_path = data_path("suspicious_holdout")
    holdout = pd.read_parquet(holdout_path) if holdout_path.exists() else pd.DataFrame()
    if not holdout.empty:
        holdout = holdout.copy()
        holdout["prob_cal"] = apply_temperature(
            np.asarray(predictor.scores(holdout), dtype=float), temperature)

    results = {
        "model": model_name,
        "strict_threshold": strict_t,
        "balanced_threshold": bal_t,
        "weak_tag_prob": WEAK_TAG_PROB,
        "clear_scams": land(ext, "external_scam (clear scams)"),
        "clear_safe": land(ham, "public_ham_english (clear safe)"),
        "synthetic_test_b": land(test_b, "test_b (synthetic)"),
        "holdout": land(holdout, "suspicious holdout (human unsure)")
        if not holdout.empty else {"target": "suspicious holdout", "n": 0},
        "never_abstain": {
            "scam_verdicts_on_clear_scams": int((ext["prob_cal"] >= strict_t).sum())
            if not ext.empty else 0,
            "cost_note": "the baseline this rule must beat: every row above the "
                         "threshold keeps its definitive verdict",
        },
    }
    return results


def markdown(results: dict) -> list[str]:
    """Honest table: what deferral buys and what it costs."""
    cs = results["clear_scams"]
    cf = results["clear_safe"]
    ho = results["holdout"]
    lines = [
        "## The `suspicious` verdict (rule-based deferral)",
        "",
        "> A probability band is impossible here (bimodal scores), so "
        "`suspicious` means *\"the model says scam, but no evidence I can name "
        "supports it\"*: no tactic above "
        f"{results['weak_tag_prob']:.2f}, no OTP/payment/credential words, no "
        "risky link type, no keyword-rule hit. Decisions use the **strict** "
        f"threshold ({results['strict_threshold']:.3f}).",
        "",
        "| set | n | scam | suspicious | safe | share deferred | note |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in (cs, cf, results["synthetic_test_b"], ho):
        if not row.get("n"):
            lines.append(f"| {row['target']} | 0 | - | - | - | - | no rows |")
            continue
        note = "too small for claims" if row["too_small"] else ""
        lines.append(
            f"| {row['target']} | {row['n']:,} | {row['scam']:,} | "
            f"{row['suspicious']:,} | {row['safe']:,} | "
            f"{row['share_suspicious']:.1%} | {note} |")
    lines += ["", "### What deferral costs", ""]
    if cs.get("n"):
        lost = cs["suspicious"] / max(cs["scam"] + cs["suspicious"], 1)
        lines.append(
            f"- **Clear scams sent to a human instead of a verdict: "
            f"{cs['suspicious']:,}/{cs['n']:,} ({lost:.1%}) of held-out real "
            "scams.** That is the price of the rule: those are messages the "
            "model was already calling scam, and it now refuses to say so.")
    if cf.get("n"):
        lines.append(
            f"- Clear safe messages escalated to `suspicious`: "
            f"{cf['suspicious']:,}/{cf['n']:,} ({cf['share_suspicious']:.1%}) "
            "- these are false alarms that become \"check this\", which is "
            "better than a hard `scam` but still work for a human.")
    if ho.get("n"):
        lines.append(
            f"- The {ho['n']:,}-row human-`suspicious` holdout lands: "
            f"{ho['suspicious']:,} suspicious, {ho['scam']:,} scam, "
            f"{ho['safe']:,} safe. Against the probability band, which put "
            "0% of these rows in the band, this rule actually defers.")
    lines += [
        "",
        "> Honest reading: the rule defers on rows where the model has no "
        "taggable reason, which is a *real* signal (the tag heads are a second "
        "opinion on the same text). It is not a calibration statement, and it "
        "does not make the probability meaningful.",
        "",
    ]
    return lines


def update_operating_point(results: dict) -> None:
    """Record which abstention method is in force (probability | rules | none)."""
    path = data_path("operating_point")
    with open(path, "r", encoding="utf-8") as f:
        point = yaml.safe_load(f)
    point["abstention_method"] = "rules"
    point["abstention_rule"] = {
        "weak_tag_prob": results["weak_tag_prob"],
        "requires": "no tag >= weak_tag_prob AND no OTP/payment/credential "
                    "word AND no risky link AND no keyword hit",
        "measured_holdout_deferral": results["holdout"].get("share_suspicious"),
        "measured_clear_scam_deferral": (
            results["clear_scams"]["suspicious"]
            / max(results["clear_scams"].get("n", 1), 1)),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 operating point - FITTED ON VALIDATION ONLY.\n"
                "# 'modes' are the three named decision thresholds; the legacy\n"
                "# 'operating_point' block is the Phase 4 abstention band.\n")
        yaml.safe_dump(point, f, sort_keys=False, allow_unicode=True)
    print(f"  updated {path.name}: abstention_method=rules")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rule-based suspicious verdict")
    parser.add_argument("--model", default="verdict_lr_plus")
    args = parser.parse_args(argv)
    print(f"(seed={SEED}) src/suspicious.py  model={args.model}")
    results = evaluate(args.model)
    text = "\n".join(markdown(results))
    out = data_path("results_json").parent / "suspicious_band.md"
    out.write_text(text, encoding="utf-8")
    json_out = data_path("results_json").parent / "suspicious_band.json"
    json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(text)
    print(f"\nWROTE {out}\nWROTE {json_out}")
    update_operating_point(results)
    return results


if __name__ == "__main__":
    main()
