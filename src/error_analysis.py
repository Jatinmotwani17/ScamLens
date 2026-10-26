"""
src/error_analysis.py - look at the actual mistakes (Phase 4).

WHAT IT DOES
  Takes the FIXED operating point from configs/operating_point.yaml (fitted on
  validation only, see src/calibrate.py), then lists, for every reported test
  set:
    * the 30 most-confident FALSE POSITIVES (gold safe, model says scam),
    * the 30 most-confident FALSE NEGATIVES (gold scam, model says safe),
  on the held-out real strata and on the synthetic `test_b`, and groups them
  by tactic tag and by script_style. For every failure pattern we print the
  count, 3 example texts and a one-line hypothesis.

WHY GROUPED BY TAG AND STYLE
  A model that misses every `parcel_courier_scam` is a different bug from a
  model that trips on short English ham. Tag grouping answers "which tactic is
  failing"; script_style answers "is Hinglish / Devanagari text failing".

WHY "HYPOTHESIS" AND NOT "CAUSE"
  A hypothesis is a testable reading of the examples, not a proven fact. It
  says what to check next; it never claims to have found the root cause.

MODEL-AGNOSTIC
  Every Phase 4 script takes --model <name>. It reads
  outputs/<name>/preds_<split>.parquet for a transformer run, or scores text
  with the TF-IDF baseline for --model verdict_lr_plus.

Run:
  python -m src.error_analysis --model verdict_lr_plus
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.calibrate import load_operating_point
from src.config import SEED, data_path
from src.phase4_common import (
    apply_temperature,
    load_predictor,
    predictions_for_split,
    too_small,
)

TOP_N = 30            # errors listed per direction per set
EXAMPLES_PER_PATTERN = 3
SNIPPET_CHARS = 220


# ---------------------------------------------------------------------------
# hypothesis language (a reading of the examples, testable, not a claim)
# ---------------------------------------------------------------------------
TAG_HYPOTHESES = {
    "suspicious_link": "the model leans on link/URL wording, so any safe row "
                       "with a link is at risk of being called scam",
    "parcel_courier_scam": "courier/delivery wording is shared with real "
                           "delivery notifications, so the boundary is thin",
    "authority_impersonation": "police/customs wording overlaps legitimate "
                               "authority notices the ham corpus rarely has",
    "bank_kyc_impersonation": "KYC/block wording is also used in real bank "
                              "service messages",
    "utility_disconnection_scam": "disconnection warnings look like real "
                                  "utility alerts; short safe rows are risky",
    "urgency_threat": "urgency words alone are a weak signal - real messages "
                      "also say 'today' and 'immediately'",
    "payment_or_upi_request": "UPI request wording overlaps genuine collect-"
                              "money messages between people",
    "otp_or_pin_request": "OTP text is rare in the ham corpus, so the model "
                          "over-triggers on digits + 'code'",
    "relative_in_trouble": "family-chat wording is exactly what the safe "
                           "corpus lacks, so it is easy to over-flag",
    "prize_or_lottery": "prize wording is rare in real ham; a single "
                        "'congratulations' can decide it",
    "investment_or_trading_scam": "investment wording overlaps real market "
                                  "alerts sent by brokers",
    "job_or_task_scam": "job-offer wording overlaps genuine hiring mails",
    "loan_app_harassment": "loan-app wording overlaps genuine repayment "
                           "reminders",
}
STYLE_HYPOTHESES = {
    "roman_hinglish": "roman-Hinglish text is nearly absent from real "
                      "training rows (8 of 15,151), so its errors are the "
                      "known Hinglish gap, not a new bug",
    "devanagari": "Devanagari only ever appears in synthetic rows, so this is "
                  "generator behaviour, not real-world evidence",
    "mixed": "mixed-script rows are the rarest and least represented style",
    "english": "English is the best-represented style, so an English error "
               "points at the threshold or the corpus, not the script",
}


def _hypothesis(target: str, direction: str, tag: str, style: str) -> str:
    """One line per pattern: the tag reading, else the style reading."""
    if tag != "(no tag)" and tag in TAG_HYPOTHESES:
        base = TAG_HYPOTHESES[tag]
    elif style in STYLE_HYPOTHESES:
        base = STYLE_HYPOTHESES[style]
    else:
        base = "no tag or style explains it - read the examples for a new pattern"
    where = "test_b (synthetic, unseen families)" if target == "test_b" \
        else f"the real stratum `{target.split('/', 1)[-1]}`"
    verb = ("over-flagged a safe message" if direction == "FP"
            else "missed a real scam")
    return f"{base}. That is a hypothesis to test, not a proven cause " \
           f"({verb} in {where})."


# ---------------------------------------------------------------------------
# error collection
# ---------------------------------------------------------------------------
def _primary_tag(tactics) -> str:
    tags = list(tactics) if isinstance(tactics, (list, tuple, set, np.ndarray)) else []
    tags = [str(t) for t in tags]
    return tags[0] if tags else "(no tag)"


def collect_errors(target: str, frame: pd.DataFrame, t_high, t_low,
                   temperature: float) -> dict:
    """Errors on one target, ranked by confidence, plus the two groupings."""
    y = (frame["verdict"].astype(str) == "scam").to_numpy(dtype=float)
    raw = frame["prob_raw"].to_numpy(dtype=float)
    cal = apply_temperature(raw, temperature)
    from src.phase4_common import apply_band
    labels = apply_band(cal, t_high, t_low)

    truth = np.where(y == 1, "scam", "safe")
    fp = (truth == "safe") & (labels == "scam")
    fn = (truth == "scam") & (labels == "safe")
    abstain = labels == "suspicious"

    def top(mask: np.ndarray, reverse: bool) -> pd.DataFrame:
        sub = frame[mask].copy()
        sub["prob_cal"] = cal[mask]
        sub["prob_raw"] = raw[mask]
        sub["pred"] = labels[mask]
        sub = sub.sort_values("prob_cal", ascending=not reverse, kind="stable")
        return sub.head(TOP_N).reset_index(drop=True)

    top_fp = top(fp, reverse=True)          # most confident wrong-scam
    top_fn = top(fn, reverse=False)         # most confident wrong-safe

    def group_table(errors: pd.DataFrame) -> list[dict]:
        if errors.empty:
            return []
        out = []
        for (tag, style), sub in errors.groupby(
                [errors["tactics"].map(_primary_tag), errors["script_style"]]):
            out.append({"target": target, "tag": str(tag), "style": str(style),
                        "n": int(len(sub))})
        return sorted(out, key=lambda r: -r["n"])

    return {
        "target": target,
        "n_rows": int(len(frame)),
        "n_scam": int(y.sum()),
        "n_safe": int((y == 0).sum()),
        "n_fp": int(fp.sum()),
        "n_fn": int(fn.sum()),
        "n_abstain": int(abstain.sum()),
        "fp_groups": group_table(top_fp),
        "fn_groups": group_table(top_fn),
        "top_fp": top_fp,
        "top_fn": top_fn,
    }


def targets_to_scan(name: str) -> list[tuple[str, pd.DataFrame]]:
    """test_b plus every real stratum (the places we are willing to claim on)."""
    predictor = load_predictor(name)
    out: list[tuple[str, pd.DataFrame]] = []
    test_b = predictions_for_split(predictor, name, "test_b")
    if not test_b.empty:
        out.append(("test_b", test_b))
    real = predictions_for_split(predictor, name, "test_real")
    if not real.empty and "real_stratum" in real.columns:
        for strat in ("external_scam", "public_ham_english",
                      "user_real_scam", "user_real_safe"):
            sub = real[real["real_stratum"] == strat].reset_index(drop=True)
            if not sub.empty:
                out.append((f"test_real/{strat}", sub))
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _snippet(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."


def _pattern_blocks(result: dict, direction: str) -> list[str]:
    """Counts + 3 examples + one hypothesis, for every tag/style pattern."""
    errors = result["top_fp"] if direction == "FP" else result["top_fn"]
    groups = result["fp_groups"] if direction == "FP" else result["fn_groups"]
    label = ("FALSE POSITIVES" if direction == "FP" else "FALSE NEGATIVES")
    lines = [f"**{label}** ({result['n_' + direction.lower()]} total, "
             f"top {len(errors)} shown)", ""]
    if errors.empty:
        return lines + ["No errors of this kind at the fixed operating point.", ""]

    lines += [f"| pattern (primary tag x style) | n in top {len(errors)} |",
              "|---|---:|"]
    for group in groups[:12]:
        lines.append(f"| {group['tag']} x {group['style']} | {group['n']} |")
    lines.append("")

    # Three examples per pattern: walk the group table, take the first example
    # of each pattern until each has EXAMPLES_PER_PATTERN texts.
    used: dict[tuple[str, str], list[str]] = {}
    for _, row in errors.iterrows():
        key = (_primary_tag(row["tactics"]), str(row["script_style"]))
        used.setdefault(key, [])
        if len(used[key]) < EXAMPLES_PER_PATTERN:
            used[key].append(row["text"])

    for group in groups[:8]:
        key = (group["tag"], group["style"])
        examples = used.get(key, [])
        lines += [f"### {key[0]} x {key[1]} ({group['n']})", ""]
        for text in examples:
            # PRIVACY: message text is never written to a report. Only a
            # shape fingerprint survives, which is enough to see whether the
            # failing messages are short replies or long notices.
            lines.append(f"> [text withheld - {len(str(text).split())} words, "
                         f"{len(str(text))} chars]")
            lines.append("")
        lines += [f"*Hypothesis:* {_hypothesis(result['target'], direction, key[0], key[1])}", ""]
    return lines


def markdown(name: str, results: list[dict]) -> list[str]:
    lines = [
        f"# Phase 4 error analysis - `{name}`",
        "",
        f"> Decisions use the FIXED operating point from "
        f"`configs/operating_point.yaml` (fitted on validation only, seed "
        f"{SEED}). FPs = gold `safe` -> verdict `scam`; FNs = gold `scam` -> "
        "verdict `safe`; rows in the abstain band are counted separately.",
        "",
        "## Counts", "",
        "| set | n | n safe | n scam | FP | FN | abstain | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        flag = too_small(result["n_rows"])
        note = "too small for claims" if flag else ""
        if result["n_fp"] == 0 and result["n_fn"] == 0:
            note = (note + "; no errors").strip("; ")
        lines.append(
            f"| `{result['target']}` | {result['n_rows']:,} | "
            f"{result['n_safe']:,} | {result['n_scam']:,} | {result['n_fp']} | "
            f"{result['n_fn']} | {result['n_abstain']} | {note} |")
    lines.append("")

    for result in results:
        lines += [f"## `{result['target']}`", ""]
        lines += _pattern_blocks(result, "FP")
        lines += _pattern_blocks(result, "FN")
    lines += ["> A hypothesis is a reading of the examples to test next - it is "
              "not a proven cause.", ""]
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: error analysis")
    parser.add_argument("--model", default="verdict_lr_plus")
    args = parser.parse_args(argv)

    point = load_operating_point(args.model)
    temperature = float(point["calibration"]["temperature"])
    op = point["operating_point"]
    t_high, t_low = op["t_high"], op["t_low"]

    print(f"(seed={SEED}) src/error_analysis.py  model={args.model}")
    print("=" * 60)
    print(f"  operating point: t_high={t_high} t_low={t_low} "
          f"(temperature={temperature:.4f})")

    results = [collect_errors(target, frame, t_high, t_low, temperature)
               for target, frame in targets_to_scan(args.model)]

    out = data_path("error_analysis")
    text = "\n".join(markdown(args.model, results))
    out.write_text(text, encoding="utf-8")
    print(f"\nWROTE {out}")
    for result in results:
        print(f"  {result['target']:34s} n={result['n_rows']:5d} "
              f"FP={result['n_fp']:3d} FN={result['n_fn']:3d} "
              f"abstain={result['n_abstain']:3d}")
    return results


if __name__ == "__main__":
    main()
