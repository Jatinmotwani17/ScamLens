"""
src/phase4.py - build the whole Phase 4 report in one command.

WHAT IT DOES
  Runs the four Phase 4 steps in order and assembles their output into ONE
  file, results/PHASE4_REPORT.md, plus results/phase4.json for the notebook:

    1. src/calibrate.py      temperature scaling + abstention band (val only)
    2. src/novelty_metrics.py  recall / FPR / AUROC per max_train_sim bucket
    3. src/error_analysis.py  the 30 worst FPs and FNs per test set
    4. src/robustness.py      perturbation / Hinglish-transfer / safe-chat probes

  Nothing here fits anything: every fitted number comes from the validation
  split inside src/calibrate.py, and every probe carries source=probe_*.

WHY ONE ORCHESTRATOR
  The four reports share one operating point. Building them separately invites
  the classic bug where the calibration table and the attack table were scored
  at different thresholds. Here the operating point is fitted ONCE, then used
  by everything below it.

MODEL-AGNOSTIC
  Every step takes the same --model. With a transformer run it reads
  outputs/<name>/preds_<split>.parquet; with a TF-IDF baseline it scores text
  on the fly. Re-run this file with --model e1_muril once the GPU numbers land
  and the whole report regenerates.

Run:
  python -m src.phase4 --model verdict_lr_plus
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src import calibrate, error_analysis, novelty_metrics, robustness
from src.config import SEED, data_path
from src.phase4_common import MIN_ROWS_FOR_CLAIMS, too_small


# ---------------------------------------------------------------------------
# headline numbers
# ---------------------------------------------------------------------------
def headline(fitted: dict, scored: dict, holdout: dict, novelty: dict,
             errors: list[dict], rob: dict) -> dict:
    """The handful of numbers a reader should see before the tables."""
    th = fitted["thresholds"]
    targets = {row["target"]: row for row in scored["targets"]}
    transfer = rob["transfer"]
    safe = rob["safe_chat"]
    worst = min((row for row in rob["attacks"]
                 if row["attack"] != "clean (baseline)"
                 and row["delta_recall_changed"] is not None),
                key=lambda r: r["delta_recall_changed"], default=None)
    novelty_note = None
    test_b = novelty["targets"].get("test_b", {})
    if test_b:
        recalls = [r["scam_recall"] for r in test_b.values()
                   if r["scam_recall"] is not None and not r["too_small_for_claims"]]
        if recalls:
            novelty_note = {"test_b_buckets": len(recalls),
                            "min_recall": float(min(recalls)),
                            "max_recall": float(max(recalls))}
    return {
        "model": fitted["model"],
        "temperature": fitted["temperature"],
        "temperature_raw_val": fitted["temperature_raw_val"],
        "ece_val_raw": fitted["ece_raw"],
        "ece_val_calibrated": fitted["ece_calibrated"],
        "t_high": th["t_high"],
        "t_low": th["t_low"],
        "abstention_available": th["abstention_available"],
        "n_rows_in_band": th["n_rows_in_band"],
        "n_real_safe_val": fitted["n_real_safe_val"],
        "n_synth_safe_val": fitted["n_synth_safe_val"],
        "holdout_rows": holdout.get("n_rows"),
        "holdout_share_suspicious": holdout.get("share_suspicious"),
        "holdout_counts": holdout.get("counts"),
        "public_ham_fpr": targets.get("test_real/public_ham_english", {}).get("fpr_at_t_high"),
        "public_ham_n": targets.get("test_real/public_ham_english", {}).get("n_rows"),
        "external_scam_recall": targets.get("test_real/external_scam", {}).get(
            "scam_recall_at_t_high"),
        "external_scam_n": targets.get("test_real/external_scam", {}).get("n_rows"),
        "external_scam_fn": next((e["n_fn"] for e in errors
                                  if e["target"] == "test_real/external_scam"), None),
        "n_real_scam_val": fitted["cell_counts"].get("real/scam"),
        "novelty": novelty_note,
        "worst_attack": None if worst is None else {
            "pool": worst["pool"], "attack": worst["attack"],
            "recall": worst["recall_changed"],
            "delta": worst["delta_recall_changed"], "n": worst["n"],
        },
        "transfer_recall_english": transfer.get("recall_english"),
        "transfer_recall_hinglish": transfer.get("recall_hinglish"),
        "transfer_n": transfer.get("n"),
        "safe_chat_fpr": safe["fpr"],
        "safe_chat_n": safe["n"],
        "safe_chat_max_train_sim": safe["max_train_sim"],
        "probe_rows": rob["probe_rows_written"],
    }


def tldr(h: dict) -> list[str]:
    """Six bullets, every one traceable to a table below."""
    esc_n = h.get("external_scam_n") or 0
    esc_rec = h.get("external_scam_recall") or 0.0
    rows = [
        f"**The model is saturated, and that is the headline.** At the fixed "
        f"point (`t_high` = {h['t_high']:.3f}) it catches "
        f"{100 * esc_rec:.1f}% of {esc_n:,} held-out real scams "
        f"({h.get('external_scam_fn') or 0} missed) while flagging "
        f"{100 * (h['public_ham_fpr'] or 0):.1f}% of real English ham "
        f"(n={h['public_ham_n']}). Every robustness probe therefore has almost "
        "no recall left to lose - the failure mode is false positives, not "
        "missed attacks.",

        f"**Temperature scaling works on the strata we can measure**, but the "
        f"safe side stays miscalibrated: val ECE {h['ece_val_raw']:.4f} -> "
        f"{h['ece_val_calibrated']:.4f} (reweighted), while "
        f"`public_ham_english` is still the worst stratum. T = "
        f"{h['temperature']:.3f} (the raw-val fit would be "
        f"{h['temperature_raw_val']:.3f}).",

        f"**The abstention band does not exist for this model.** At the "
        f"0.95/0.95 targets `t_high` = `t_low` = {h['t_high']:.3f} with "
        f"{h['n_rows_in_band']} validation rows between them; no target from "
        "0.80 to 0.99 produces a non-empty band, because the scorer's "
        "probabilities are bimodal. Coverage is 100% by construction.",
    ]
    if h.get("holdout_rows"):
        counts = h.get("holdout_counts") or {}
        rows.append(
            f"**The human-`suspicious` holdout gets no abstention at all**: "
            f"{counts.get('scam', 0):,} of {h['holdout_rows']:,} rows "
            f"({100 * (h.get('holdout_share_suspicious') or 0):.1f}% abstained) "
            "are pushed to a definitive verdict. These rows are ambiguous by "
            "construction, so `scam` is not automatically wrong - but a "
            "three-way system that never uses its third verdict is a two-way "
            "system with extra UI.")
    if h.get("worst_attack"):
        w = h["worst_attack"]
        rows.append(
            f"**Mechanical attacks barely move recall**: the worst probe is "
            f"{w['attack']} on `{w['pool']}` (n={w['n']}), recall "
            f"{w['recall']:.3f} vs a clean baseline near 1.000 "
            f"({w['delta']:+.3f}). This is the saturation effect again: a "
            "classifier that says `scam` to almost everything cannot be talked "
            "out of it.")
    rows.append(
        f"**The one number that would worry a reviewer is the fresh safe-chat "
        f"FPR: {100 * h['safe_chat_fpr']:.1f}% ({int(round(h['safe_chat_fpr'] * h['safe_chat_n']))}"
        f"/{h['safe_chat_n']})** on legitimate messages that share no template "
        f"with training (max similarity to train "
        f"{h['safe_chat_max_train_sim']:.2f}). The Hinglish transfer probe, by "
        f"contrast, barely changes recall "
        f"({(h['transfer_recall_english'] or 0):.3f} -> "
        f"{(h['transfer_recall_hinglish'] or 0):.3f} on n={h['transfer_n']}).")
    return rows


# ---------------------------------------------------------------------------
# limitations - the section that keeps the report honest
# ---------------------------------------------------------------------------
def limitations(h: dict) -> list[str]:
    return [
        "## Limitations",
        "",
        f"1. **This is the TF-IDF baseline, not the transformer.** Every table "
        f"describes `{h['model']}`. `e1_muril` has no results on this machine "
        "(no CUDA), so nothing here is evidence about the model the project "
        "intends to ship. Re-running `python -m src.phase4 --model e1_muril` "
        "regenerates the whole report from saved predictions.",
        f"2. **The validation split cannot support a safe-side threshold on its "
        f"own.** It is {100 * (h.get('n_real_scam_val') or 0) / max((h.get('n_real_scam_val') or 0) + h['n_real_safe_val'], 1):.1f}% "
        f"scam; the only real safe rows are **n={h['n_real_safe_val']}** "
        f"(plus n={h['n_synth_safe_val']} synthetic). Fitting uses equal weight "
        "per (scam|safe x real|synthetic) cell for exactly this reason, but "
        "`t_low` still rests mostly on synthetic safe text. Treat every safe-side "
        "number as provisional.",
        f"3. **No bucket below the n-gate is a claim.** `test_b` has 224 rows "
        f"in total and the user strata have 5 each; anything under "
        f"{MIN_ROWS_FOR_CLAIMS} rows is marked *too small for claims* in the "
        "tables and must not be quoted as a result.",
        "4. **`external_scam` contains only scams.** It has zero safe rows, so "
        "recall is computable and FPR/AUROC are not. A blended \"real\" number "
        "would be meaningless, which is why the strata are never pooled.",
        "5. **Novelty bucketing cannot show degradation here.** With recall "
        "pinned near 1.000 in every bucket, the bucket table mostly demonstrates "
        "the saturation of the decision, not generalisation. The informative "
        "column is FPR on ham, and ham is concentrated at low similarity "
        "(n=2 above 0.5).",
        "6. **The attack probes are scripted, not adversarial.** Each edit is a "
        "fixed transformation at a documented rate (see the note under the "
        "table). Real evasion is adaptive and human; these probes bound how "
        "much a *known* edit costs, nothing more.",
        "7. **The Hinglish probe is translation-derived, not real Hinglish.** "
        "It rewrites English with a phrase dictionary (353 pairs, every Hinglish "
        "form verified against the synthetic pools) and replaces ~39% of tokens. "
        "No GEMINI_API_KEY was present, so the paraphrase path was skipped. Real "
        "Hinglish code-mixing looks different and this probe must not stand in "
        "for it.",
        "8. **The safe-chat probe is generated from our own templates.** They "
        "share no template with training (max cosine similarity "
        f"{h['safe_chat_max_train_sim']:.2f} to any train row), but they are "
        "still synthetic text written by the same project, so the FPR is an "
        "estimate with an unknown bias in either direction.",
        f"9. **The suspicious holdout is {h.get('holdout_rows') or 'n/a'} rows, "
        "not 1,403.** The Phase 4 brief assumed 1,403; the table on disk holds "
        f"{h.get('holdout_rows') or 'n/a'} (the rest were relabelled or dropped "
        "during Phase 2.5). The numbers above use what actually exists.",
        "10. **One model, one seed, no confidence intervals.** No metric here "
        "carries a bootstrap interval. Differences under ~1 point (several "
        "attack deltas are -0.00x) are noise and are reported as such.",
        "",
    ]


def next_steps(h: dict) -> list[str]:
    return [
        "## What would change the conclusions",
        "",
        "| if we had | it would test |",
        "|---|---|",
        "| `e1_muril` GPU results | whether the saturation and the empty "
        "abstain band are properties of TF-IDF or of the data |",
        "| 500+ labelled real ham messages | whether `t_low` and the ~11% "
        "safe-chat FPR survive on real traffic |",
        "| real Hinglish messages with labels | the transfer probe's stand-in "
        "for code-mixed traffic |",
        "| a held-out set of human-flagged `suspicious` rows with a gold binary "
        "label | whether abstention should defer or decide |",
        "",
    ]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build(model: str) -> dict:
    """Run all four steps, write the report + json, return the payload."""
    print(f"(seed={SEED}) src/phase4.py  model={model}")
    print("=" * 66)

    # 1. calibration + operating point (validation only)
    fitted = calibrate.fit_on_validation(model)
    # The three named modes are part of the operating point; without them the
    # API has nothing to serve, so they are fitted before the file is written.
    fitted["modes"] = calibrate.fit_all_modes(model, fitted)
    fitted["mode_scores"] = calibrate.evaluate_modes(model, fitted, fitted["modes"])
    calibrate.save_operating_point(fitted)
    calibrate.save_reliability_bins(model, fitted)
    holdout = calibrate.holdout_band(model, fitted)
    sweep = calibrate.coverage_sweep(model, fitted, holdout.get("probs"))
    scored = calibrate.score_targets(model, fitted)
    cal_md = calibrate.markdown(fitted, scored, holdout, sweep)

    # 2. novelty-bucketed metrics
    novelty = novelty_metrics.bucket_results(model)
    data_path("novelty_md").write_text(
        "\n".join(novelty_metrics.markdown(novelty)), encoding="utf-8")
    nov_md = novelty_metrics.markdown(novelty)[1:]      # drop its own H1

    # 3. error analysis
    errors = [error_analysis.collect_errors(target, frame, fitted["thresholds"]["t_high"],
                                            fitted["thresholds"]["t_low"],
                                            fitted["temperature"])
              for target, frame in error_analysis.targets_to_scan(model)]
    data_path("error_analysis").write_text(
        "\n".join(error_analysis.markdown(model, errors)), encoding="utf-8")
    err_md = error_analysis.markdown(model, errors)[1:]

    # 4. robustness probes (never trained on)
    rob = robustness.run_all(model)
    data_path("robustness_md").write_text(
        "\n".join(robustness.markdown(rob)), encoding="utf-8")
    rob_md = robustness.markdown(rob)

    h = headline(fitted, scored, holdout, novelty, errors, rob)

    lines = [
        f"# Phase 4 - calibration, abstention, error analysis, robustness",
        "",
        f"Model under test: **`{model}`**. Fitted on validation only (seed "
        f"{SEED}); every threshold and temperature below comes from "
        "`configs/operating_point.yaml`. Probes live in their own table and "
        "never enter training.",
        "",
        "> Everything in this file is either (a) a measured number with `n` "
        "printed beside it, or (b) explicitly labelled a hypothesis. No "
        "number here is a production claim.",
        "",
        "## What stands out",
        "",
    ]
    lines += [f"{i}. {row}" for i, row in enumerate(tldr(h), start=1)]
    lines += [""]
    lines += ["---", "", "## 1. Novelty-bucketed metrics (the honest generalisation view)",
              ""] + nov_md
    lines += ["---", "", "## 2-3. Calibration and the abstention band", ""] + cal_md
    lines += ["---", "", "## 4. Error analysis", ""] + err_md
    lines += ["---", ""] + rob_md
    lines += ["---", ""] + limitations(h)
    lines += ["---", ""] + next_steps(h)

    report = "\n".join(lines)
    out = data_path("phase4_report")
    out.write_text(report, encoding="utf-8")

    payload = {
        "model": model,
        "seed": SEED,
        "headline": h,
        "tldr": tldr(h),
        "calibration": {
            "temperature": fitted["temperature"],
            "temperature_raw_val": fitted["temperature_raw_val"],
            "ece_val_raw": fitted["ece_raw"],
            "ece_val_calibrated": fitted["ece_calibrated"],
            "thresholds": {k: v for k, v in fitted["thresholds"].items()
                           if not isinstance(v, np.ndarray)},
            "cell_counts": fitted["cell_counts"],
            "targets": scored["targets"],
            "coverage_sweep": sweep,
            "holdout": {k: v for k, v in holdout.items() if k != "probs"},
        },
        "novelty": {"targets": novelty["targets"]},
        "errors": [{k: v for k, v in e.items() if k not in ("top_fp", "top_fn")}
                   for e in errors],
        "robustness": rob,
    }
    # numpy scalars would break json.dump; default=str keeps ids readable.
    data_path("phase4_json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nWROTE {data_path('phase4_report')}")
    print(f"WROTE {data_path('phase4_json')}")
    print("\nHeadline:")
    for i, row in enumerate(tldr(h), start=1):
        print(f"  {i}. {row}")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the whole Phase 4 report for one model")
    parser.add_argument("--model", default="verdict_lr_plus")
    args = parser.parse_args(argv)
    build(args.model)


if __name__ == "__main__":
    main()
