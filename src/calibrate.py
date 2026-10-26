"""
src/calibrate.py - temperature scaling + the abstention operating point.

WHAT IT DOES
  1. Fits ONE temperature parameter (temperature scaling) so the model's
     probabilities mean what they say. Fitted on **validation only**.
  2. Fits the two thresholds of the three-way decision:
         scam      if p >= t_high    (scam-verdict precision >= 0.95 on val)
         safe      if p <= t_low     (safe-verdict NPV       >= 0.95 on val)
         suspicious otherwise        (abstain; Phase 4 holdout material)
  3. Writes the fitted values to configs/operating_point.yaml so every other
     Phase 4 script uses the SAME operating point, and every table can say
     "at the fixed operating point".

WHY THE VALIDATION SPLIT IS REWEIGHTED (Phase 4 amendment)
  Real validation is 98.3% scam and ~87% of it comes from ONE source. Fitting
  on raw val optimises for "flag real-looking text as scam", and the only
  safe evidence for t_low would be 60 UCI ham rows. We therefore fit with
  EQUAL WEIGHT PER CELL (scam|safe x synthetic|real), so the 60 real ham rows
  count as much as the 3,511 real scams. The per-cell row counts are printed
  next to every threshold, and the raw-val fit plus a per-part fit are reported
  as a sensitivity check.

NEVER FIT ON TEST
  assert_val_only() raises before any fitting if the frame contains a split
  other than `val`. test_* rows are only ever scored with the saved values.

Run:
  python -m src.calibrate --model verdict_lr_plus
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
import yaml

from src.config import CFG, SEED, data_path
from src.phase4_common import (
    apply_temperature,
    assert_val_only,
    band_summary,
    apply_band,
    ece,
    equal_cell_weights,
    evaluation_targets,
    fit_temperature,
    fit_thresholds,
    load_predictor,
    predictions_for_split,
    rate_with_ci,
    reliability_bins,
    too_small,
    weighted_rate,
)

P4 = CFG.get("phase4", {})
MIN_PRECISION = float(P4.get("min_precision", 0.95))
MIN_NPV = float(P4.get("min_npv", 0.95))
N_BINS = int(P4.get("calibration_bins", 10))


# ---------------------------------------------------------------------------
# fitting (validation only)
# ---------------------------------------------------------------------------
def fit_on_validation(name: str) -> dict:
    """Temperature + thresholds + every number they rest on."""
    predictor = load_predictor(name)
    val = predictions_for_split(predictor, name, "val")
    assert_val_only(val, "temperature + thresholds")

    y = (val["verdict"].astype(str) == "scam").to_numpy(dtype=float)
    raw = val["prob_raw"].to_numpy(dtype=float)

    # ---- temperature scaling, two fits ---------------------------------
    weights, cell_counts = equal_cell_weights(val)
    temp_reweighted = fit_temperature(y, raw, weights)
    temp_raw = fit_temperature(y, raw, None)          # sensitivity only

    # The PRIMARY calibration is the reweighted one (see module docstring).
    cal_reweighted = apply_temperature(raw, temp_reweighted)
    cal_raw = apply_temperature(raw, temp_raw)

    # ---- thresholds, with the evidence behind them ---------------------
    th_reweighted = fit_thresholds(y, cal_reweighted, weights,
                                   MIN_PRECISION, MIN_NPV)
    th_raw = fit_thresholds(y, cal_raw, None, MIN_PRECISION, MIN_NPV)

    # Sensitivity: fit on each half of val separately, so a reader can see
    # how much the choice of validation half moves the operating point.
    per_part = {}
    is_synth = val["source"].isin(["synthetic", "synthetic_llm"]).to_numpy()
    for part, mask in (("val_synthetic", is_synth), ("val_real", ~is_synth)):
        if mask.sum() == 0 or len(set(y[mask])) < 1:
            continue
        p_part = apply_temperature(raw[mask], temp_reweighted)
        row = fit_thresholds(y[mask], p_part, None, MIN_PRECISION, MIN_NPV)
        row["cell_counts"] = {
            "scam": int((y[mask] == 1).sum()),
            "safe": int((y[mask] == 0).sum()),
        }
        per_part[part] = row

    n_real_safe = int(cell_counts.get("real/safe", 0))
    n_synth_safe = int(cell_counts.get("synthetic/safe", 0))

    print(f"\n(src/calibrate.py) model={name}  fitted on VAL ONLY "
          f"(seed={SEED})")
    print(f"  val rows: {len(val)}  cells: {cell_counts}")
    print(f"  temperature (reweighted val) = {temp_reweighted:.4f}   "
          f"(raw-val fit = {temp_raw:.4f}; T>1 = over-confident)")
    print(f"  ECE raw {ece(y, raw, None, N_BINS):.4f} -> "
          f"reweighted-calibrated {ece(y, cal_reweighted, weights, N_BINS):.4f}")
    print(f"  t_high = {_fmt(th_reweighted['t_high'])} "
          f"(precision {_fmt(th_reweighted['precision_at_t_high'])}, "
          f"{th_reweighted['n_scam_behind_t_high']} scam rows behind it)")
    print(f"  t_low  = {_fmt(th_reweighted['t_low'])} "
          f"(NPV {_fmt(th_reweighted['npv_at_t_low'])}, "
          f"{th_reweighted['n_safe_behind_t_low']} safe rows behind it)")
    print(f"  band: {th_reweighted['n_rows_in_band']} val rows between the two "
          f"thresholds -> abstention "
          f"{'AVAILABLE' if th_reweighted['abstention_available'] else 'NOT AVAILABLE'} "
          f"({th_reweighted['band_reason']})")
    print(f"  safe evidence: real/safe n={n_real_safe}, "
          f"synthetic/safe n={n_synth_safe}")
    if n_real_safe < 30:
        print(f"  WARNING: only {n_real_safe} REAL safe validation rows exist. "
              "t_low rests mostly on synthetic safe rows - say so everywhere.")

    return {
        "model": name,
        "fitted_on": "val (rewritten: equal weight per scam|safe x synthetic|real)",
        "val_rows": int(len(val)),
        "cell_counts": cell_counts,
        "temperature": float(temp_reweighted),
        "temperature_raw_val": float(temp_raw),
        "ece_raw": ece(y, raw, None, N_BINS),
        "ece_calibrated": ece(y, cal_reweighted, weights, N_BINS),
        "ece_raw_on_rawfit": ece(y, cal_raw, None, N_BINS),
        "thresholds": th_reweighted,
        "thresholds_raw_val_fit": th_raw,
        "thresholds_sensitivity": per_part,
        "n_real_safe_val": n_real_safe,
        "n_synth_safe_val": n_synth_safe,
        "reliability_bins_raw": reliability_bins(y, raw, None, N_BINS),
        "reliability_bins_calibrated": reliability_bins(y, cal_reweighted, weights, N_BINS),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# scoring the fitted point on every test target (never refitting)
# ---------------------------------------------------------------------------
def score_targets(name: str, fitted: dict) -> dict:
    """Apply the fitted temperature/thresholds to every test stratum.

    Nothing is refitted here: the temperature and thresholds come from the
    saved operating point, so a test stratum cannot influence them.
    """
    predictor = load_predictor(name)
    temp = fitted["temperature"]
    th = fitted["thresholds"]
    t_high, t_low = th["t_high"], th["t_low"]

    # One prediction pass per split, then slice - so every table for the same
    # split sees exactly the same probabilities.
    frames: dict[str, pd.DataFrame] = {
        split: predictions_for_split(predictor, name, split)
        for split in ("val", "test_b", "test_real")
    }

    def select(target: str) -> pd.DataFrame:
        if target.startswith("val_"):
            want_synth = target == "val_synthetic"
            sub = frames["val"]
            keep = sub["source"].isin(["synthetic", "synthetic_llm"]) == want_synth
            return sub[keep].reset_index(drop=True)
        if target.startswith("test_real/"):
            strat = target.split("/", 1)[1]
            sub = frames["test_real"]
            if "real_stratum" not in sub.columns:
                return sub.iloc[0:0]
            return sub[sub["real_stratum"] == strat].reset_index(drop=True)
        return frames[target].reset_index(drop=True)

    rows = []
    for target, _ in evaluation_targets():
        part = select(target)
        if part.empty:
            continue
        y = (part["verdict"].astype(str) == "scam").to_numpy(dtype=float)
        raw = part["prob_raw"].to_numpy(dtype=float)
        cal = apply_temperature(raw, temp)
        truth = np.where(y == 1, "scam", "safe")
        band = band_summary(part, cal, t_high, t_low, truth)
        band.update({
            "target": target,
            "n_scam": int(y.sum()),
            "n_safe": int((y == 0).sum()),
            "ece_raw": ece(y, raw, None, N_BINS),
            "ece_calibrated": ece(y, cal, None, N_BINS),
            "mean_p_raw": float(raw.mean()) if len(raw) else None,
            "mean_p_calibrated": float(cal.mean()) if len(cal) else None,
            "scam_recall_at_t_high": (
                float(((cal >= t_high) & (y == 1)).sum() / max((y == 1).sum(), 1))),
            "fpr_at_t_high": (
                float(((cal >= t_high) & (y == 0)).sum() / max((y == 0).sum(), 1))),
        })
        rows.append(band)
    return {"targets": rows}


def coverage_sweep(name: str, fitted: dict,
                   holdout_probs: np.ndarray | None = None) -> list[dict]:
    """What a different precision/NPV target would buy.

    The model's validation probabilities are strongly bimodal, so a target can
    end up with no band at all. This sweep shows whether ANY achievable target
    produces a useful band, and what it does to the human-`suspicious` holdout:
    an abstention rule that never fires is not abstention.
    """
    predictor = load_predictor(name)
    val = predictions_for_split(predictor, name, "val")
    assert_val_only(val, "coverage sweep")
    y = (val["verdict"].astype(str) == "scam").to_numpy(dtype=float)
    raw = val["prob_raw"].to_numpy(dtype=float)
    weights, _ = equal_cell_weights(val)
    cal = apply_temperature(raw, fitted["temperature"])

    rows = []
    for target in (0.80, 0.90, 0.95, 0.98, 0.99):
        th = fit_thresholds(y, cal, weights, target, target)
        labels = apply_band(cal, th["t_high"], th["t_low"])
        row = {
            "target": target,
            "t_high": th["t_high"],
            "t_low": th["t_low"],
            "band_rows": int((labels == "suspicious").sum()),
            "band_share": float((labels == "suspicious").mean()),
            "n_rows_in_band": th["n_rows_in_band"],
            "n_safe_behind_t_low": th["n_safe_behind_t_low"],
            "n_scam_behind_t_high": th["n_scam_behind_t_high"],
        }
        if holdout_probs is not None and len(holdout_probs):
            h_labels = apply_band(holdout_probs, th["t_high"], th["t_low"])
            row["holdout_abstain_share"] = float((h_labels == "suspicious").mean())
            row["holdout_scam_share"] = float((h_labels == "scam").mean())
            row["holdout_safe_share"] = float((h_labels == "safe").mean())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# THREE NAMED OPERATING POINTS (Phase 5)
#
# One threshold cannot serve both a cautious and an aggressive use case, and a
# single blended "accuracy" hides which trade-off you bought. So we fit three
# points on the SAME weighted validation rows and report all three every time:
#
#   strict    - the smallest threshold whose weighted-val FPR <= 5%
#   balanced  - the same rule at FPR <= 10%
#   sensitive - the LARGEST threshold whose weighted-val recall >= 99%
#
# "smallest threshold meeting an FPR cap" maximises recall inside that cap;
# "largest threshold meeting a recall floor" minimises false alarms inside it.
# Both are validation-only decisions (assert_val_only runs first).
# ---------------------------------------------------------------------------
MODES: dict[str, dict] = {
    "strict": {"constraint": "fpr", "target": 0.05,
               "blurb": "validation FPR target <= 5%: highest recall inside the cap"},
    "balanced": {"constraint": "fpr", "target": 0.10,
                 "blurb": "validation FPR target <= 10%: more recalls, more false alarms"},
    "sensitive": {"constraint": "recall", "target": 0.99,
                  "blurb": "validation recall target >= 99%: fewest flags if met"},
}
DEFAULT_MODE = "balanced"


def fit_one_mode(y, p, weights, constraint: str, target: float,
                 total_rows: int) -> dict:
    """One threshold for one constraint, plus the evidence behind it."""
    candidates = np.unique(np.round(p, 6))
    chosen = None
    if constraint == "fpr":
        # FPR falls as the threshold rises: take the LOWEST threshold that
        # already satisfies the cap, which keeps the most recall.
        for t in candidates:
            fpr = weighted_rate(y, p, weights, t, positive=False)
            if fpr is not None and fpr <= target:
                chosen = float(t)
                break
    else:
        # Recall falls as the threshold rises: take the HIGHEST threshold that
        # still clears the recall floor, which drops the most false alarms.
        for t in candidates[::-1]:
            rec = weighted_rate(y, p, weights, t, positive=True)
            if rec is not None and rec >= target:
                chosen = float(t)
                break
    if chosen is None:
        # The target is unreachable on this validation set. Say so; do not
        # quietly pick something that looks close.
        chosen = float(candidates.max() if constraint == "fpr" else candidates.min())
        met = False
    else:
        met = True

    recall = weighted_rate(y, p, weights, chosen, positive=True)
    fpr = weighted_rate(y, p, weights, chosen, positive=False)
    n_flagged = int((p >= chosen).sum())
    n_safe = int((y == 0).sum())
    n_scam = int((y == 1).sum())
    return {
        "constraint": constraint,
        "target": target,
        "threshold": chosen,
        "target_met": met,
        "val_recall_weighted": recall,
        "val_fpr_weighted": fpr,
        "val_rows_flagged": n_flagged,
        "val_rows": int(total_rows),
        "val_scam_rows": n_scam,
        "val_safe_rows": n_safe,
        # Unweighted counts at this threshold, so the raw evidence is visible.
        "val_safe_flagged": int(((y == 0) & (p >= chosen)).sum()),
        "val_scam_missed": int(((y == 1) & (p < chosen)).sum()),
    }


def fit_all_modes(name: str, fitted: dict) -> dict:
    """Fit the three named points on the SAME weighted validation rows."""
    predictor = load_predictor(name)
    val = predictions_for_split(predictor, name, "val")
    assert_val_only(val, "operating-point modes")
    y = (val["verdict"].astype(str) == "scam").to_numpy(dtype=float)
    raw = val["prob_raw"].to_numpy(dtype=float)
    weights, _ = equal_cell_weights(val)
    cal = apply_temperature(raw, fitted["temperature"])
    out = {}
    for mode, spec in MODES.items():
        out[mode] = fit_one_mode(y, cal, weights, spec["constraint"],
                                 spec["target"], len(val))
        out[mode]["blurb"] = spec["blurb"]
    return out


def evaluate_modes(name: str, fitted: dict, modes: dict) -> list[dict]:
    """Every mode on every reported stratum, with Wilson intervals.

    This is where "how much false-alarm reduction did we buy" is answered: the
    frozen 200-row safe-chat probe is the held-out test set for that question.
    """
    from src.phase4_common import rate_with_ci

    predictor = load_predictor(name)
    temperature = fitted["temperature"]

    def scored(split: str) -> pd.DataFrame:
        part = predictions_for_split(predictor, name, split)
        if part.empty:
            return part
        part = part.copy()
        part["prob_cal"] = apply_temperature(
            part["prob_raw"].to_numpy(dtype=float), temperature)
        return part

    test_b = scored("test_b")
    test_real = scored("test_real")
    ham = (test_real[test_real["real_stratum"] == "public_ham_english"]
           if not test_real.empty else test_real)
    ext = (test_real[test_real["real_stratum"] == "external_scam"]
           if not test_real.empty else test_real)

    probe_path = data_path("safe_chat_probe")
    probe = pd.read_parquet(probe_path) if probe_path.exists() else pd.DataFrame()
    if not probe.empty:
        probe = probe.copy()
        probe["prob_cal"] = apply_temperature(
            np.asarray(predictor.scores(probe), dtype=float), temperature)

    rows = []
    for mode, info in modes.items():
        t = info["threshold"]
        ham_flag = int((ham["prob_cal"] >= t).sum()) if not ham.empty else 0
        ext_hit = int((ext["prob_cal"] >= t).sum()) if not ext.empty else 0
        tb_mask = (test_b["verdict"].astype(str) == "scam") if not test_b.empty \
            else pd.Series([], dtype=bool)
        tb_scam = int(tb_mask.sum()) if not test_b.empty else 0
        # Recall counts SCAMS that clear the threshold, so the denominator and
        # the numerator must be restricted to scam rows (an earlier version
        # counted safe rows too and produced recall > 1).
        tb_hit = (int((test_b.loc[tb_mask, "prob_cal"] >= t).sum())
                  if tb_scam else 0)
        probe_flag = int((probe["prob_cal"] >= t).sum()) if not probe.empty else 0
        rows.append({
            "mode": mode,
            "threshold": t,
            "constraint": info["constraint"],
            "target": info["target"],
            "target_met": info["target_met"],
            "val_recall_weighted": info["val_recall_weighted"],
            "val_fpr_weighted": info["val_fpr_weighted"],
            # Held-out real ham: the number the project is judged on.
            "ham_n": int(len(ham)),
            "ham_fp": ham_flag,
            "ham_fpr": (ham_flag / len(ham)) if len(ham) else None,
            "ham_fpr_ci": rate_with_ci(ham_flag, len(ham)),
            # Held-out real scams.
            "ext_n": int(len(ext)),
            "ext_hit": ext_hit,
            "ext_recall": (ext_hit / len(ext)) if len(ext) else None,
            "ext_recall_ci": rate_with_ci(ext_hit, len(ext)),
            # Synthetic unseen families.
            "test_b_n": int(len(test_b)),
            "test_b_scam_n": tb_scam,
            "test_b_recall": (tb_hit / tb_scam) if tb_scam else None,
            "test_b_recall_ci": rate_with_ci(tb_hit, tb_scam),
            # The frozen safe-chat probe: the false-alarm test set.
            "probe_n": int(len(probe)),
            "probe_fp": probe_flag,
            "probe_fpr": (probe_flag / len(probe)) if len(probe) else None,
            "probe_fpr_ci": rate_with_ci(probe_flag, len(probe)),
        })
    return rows


def holdout_band(name: str, fitted: dict) -> dict:
    """Where the 'suspicious' rows from build_dataset land (Phase 4 material)."""
    path = data_path("suspicious_holdout")
    if not path.exists():
        return {"available": False, "path": str(path)}
    hold = pd.read_parquet(path)
    if hold.empty:
        return {"available": False, "path": str(path)}
    predictor = load_predictor(name)
    hold = hold.reset_index(drop=True)
    hold["split"] = "suspicious_holdout"
    if "real_stratum" not in hold.columns:
        hold["real_stratum"] = ""
    raw = predictor.scores(hold)
    if raw is None:
        return {"available": False, "reason": "model has no scores"}
    cal = apply_temperature(raw, fitted["temperature"])
    th = fitted["thresholds"]
    labels = apply_band(cal, th["t_high"], th["t_low"])
    out = {
        "available": True,
        "path": str(path),
        "n_rows": int(len(hold)),
        "counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        "share_suspicious": float((labels == "suspicious").mean()),
        "mean_p": float(cal.mean()),
        "median_p": float(np.median(cal)),
        # Reused by the coverage sweep: the holdout is scored ONCE, with the
        # fitted temperature, so every table sees the same probabilities.
        "probs": cal,
    }
    print(f"\n  suspicious_holdout: {out['n_rows']} rows -> "
          f"{out['counts']} (share abstained {out['share_suspicious']:.1%})")
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def operating_point_payload(fitted: dict) -> dict:
    th = fitted["thresholds"]
    modes = fitted.get("modes", {})
    return {
        "model": fitted["model"],
        # ---- Phase 5: the three named points every consumer should use ----
        "default_mode": DEFAULT_MODE,
        "modes": {
            name: {
                "threshold": round(float(info["threshold"]), 6),
                "rule": (f"{info['constraint']} <= {info['target']:.2f}"
                         if info["constraint"] == "fpr"
                         else f"recall >= {info['target']:.2f}"),
                "blurb": info["blurb"],
                "target_met": bool(info["target_met"]),
                "val_recall_weighted": info["val_recall_weighted"],
                "val_fpr_weighted": info["val_fpr_weighted"],
                "val_safe_flagged": info["val_safe_flagged"],
                "val_scam_missed": info["val_scam_missed"],
            }
            for name, info in modes.items()
        },
        # ---- measured behaviour of each mode on held-out data ----
        "mode_measured": {
            row["mode"]: {
                "ham_n": row["ham_n"], "ham_fp": row["ham_fp"],
                "ham_fpr": row["ham_fpr"], "ham_fpr_ci": row["ham_fpr_ci"],
                "probe_n": row["probe_n"], "probe_fp": row["probe_fp"],
                "probe_fpr": row["probe_fpr"], "probe_fpr_ci": row["probe_fpr_ci"],
                "ext_n": row["ext_n"], "ext_recall": row["ext_recall"],
                "ext_recall_ci": row["ext_recall_ci"],
                "test_b_scam_n": row["test_b_scam_n"],
                "test_b_recall": row["test_b_recall"],
            }
            for row in fitted.get("mode_scores", [])
        },
        # ---- legacy Phase 4 band (unchanged, still validation-only) ----
        "fitted_on": "validation only (never test)",
        "val_rows": fitted["val_rows"],
        "fit_weights": ("equal weight per cell: scam|safe x synthetic|real "
                        "(raw val is 98.3% scam / ~87% single-source)"),
        "cell_counts": fitted["cell_counts"],
        "calibration": {
            "method": "temperature scaling (one parameter, weighted NLL)",
            "temperature": round(fitted["temperature"], 6),
            "temperature_raw_val_sensitivity": round(fitted["temperature_raw_val"], 6),
            "ece_raw": fitted["ece_raw"],
            "ece_calibrated_reweighted_val": fitted["ece_calibrated"],
        },
        "operating_point": {
            "t_high": th["t_high"],
            "t_low": th["t_low"],
            "min_precision_target": th["min_precision_target"],
            "min_npv_target": th["min_npv_target"],
            "precision_at_t_high": th["precision_at_t_high"],
            "npv_at_t_low": th["npv_at_t_low"],
            "n_scam_behind_t_high": th["n_scam_behind_t_high"],
            "n_safe_behind_t_low": th["n_safe_behind_t_low"],
            "n_rows_in_band": th["n_rows_in_band"],
            "abstention_available": th["abstention_available"],
            "band_reason": th["band_reason"],
            "bands_overlap": th["bands_overlap"],
        },
        "sensitivity_thresholds": fitted["thresholds_sensitivity"],
        "fitted_date": str(date.today()),
        "seed": SEED,
    }


def save_operating_point(fitted: dict) -> None:
    path = data_path("operating_point")
    # Guard: the serving layer reads `modes`, so writing a payload without them
    # silently breaks the API. An orchestrator that forgets fit_all_modes()
    # fails here instead.
    if not fitted.get("modes"):
        raise SystemExit(
            "refusing to write operating_point.yaml without modes - call "
            "fit_all_modes() first (src/phase4.py and src/calibrate.main do).")
    payload = operating_point_payload(fitted)
    # Preserve the abstention decision written by src/suspicious.py: this
    # function owns calibration + modes, not the abstention method, and a
    # plain rewrite would silently drop it.
    if path.exists():
        try:
            previous = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            previous = {}
        for key in ("abstention_method", "abstention_rule"):
            if key in previous:
                payload[key] = previous[key]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 4 operating point - FITTED ON VALIDATION ONLY.\n"
                "# Written by src/calibrate.py. Never edit by hand and never\n"
                "# refit on a test split (src/phase4_common.assert_val_only).\n"
                "#   p >= t_high -> scam | p <= t_low -> safe | else suspicious\n")
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    print(f"  WROTE {path}")


def save_reliability_bins(name: str, fitted: dict) -> None:
    path = data_path("reliability_bins")
    rows = []
    for tag, bins in (("raw", fitted["reliability_bins_raw"]),
                      ("calibrated", fitted["reliability_bins_calibrated"])):
        for row in bins:
            rows.append({"model": name, "stage": tag, **row})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  WROTE {path} ({len(rows)} bin rows)")


def load_operating_point(name: str | None = None) -> dict:
    """The fitted point, for every other Phase 4 script."""
    path = data_path("operating_point")
    if not path.exists():
        raise SystemExit(
            "configs/operating_point.yaml not found - run: "
            "python -m src.calibrate --model <name>")
    with open(path, "r", encoding="utf-8") as f:
        point = yaml.safe_load(f)
    if name is not None and point.get("model") not in (None, name):
        print(f"  WARNING: operating point was fitted for {point.get('model')!r} "
              f"but you asked for {name!r}; re-run src.calibrate for {name!r}.")
    return point


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _modes_lines(fitted: dict) -> list[str]:
    """The three named operating points, with Wilson intervals and n."""
    modes = fitted.get("modes") or {}
    scores = fitted.get("mode_scores") or []
    if not modes or not scores:
        return []
    lines = [
        "### Three operating points (Phase 5)",
        "",
        "Same weighted validation rows, three different promises. Pick per use \n"
        "case; never quote one blended accuracy.",
        "",
        "| mode | rule | threshold | val recall (weighted) | val FPR (weighted) "
        "| real ham FPR | safe-chat probe FPR | external_scam recall "
        "| test_b recall |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in scores:
        rule = (f"FPR <= {row['target']:.2f}" if row["constraint"] == "fpr"
                else f"recall >= {row['target']:.2f}")
        flag = "" if row["target_met"] else " *(target not met)*"
        lines.append(
            f"| **{row['mode']}**{flag} | {rule} | {row['threshold']:.3f} | "
            f"{row['val_recall_weighted']:.3f} | {row['val_fpr_weighted']:.3f} | "
            f"{row['ham_fpr_ci']} | {row['probe_fpr_ci']} | "
            f"{row['ext_recall_ci']} | {row['test_b_recall_ci']} |")
    lines += ["",
              "| mode | val safe rows flagged | val scam rows missed | "
              "real ham false alarms | probe false alarms |",
              "|---|---:|---:|---:|---:|"]
    for row in scores:
        info = modes[row["mode"]]
        lines.append(
            f"| {row['mode']} | {info['val_safe_flagged']}/{info['val_safe_rows']} | "
            f"{info['val_scam_missed']}/{info['val_scam_rows']} | "
            f"{row['ham_fp']}/{row['ham_n']} | {row['probe_fp']}/{row['probe_n']} |")
    lines += ["",
              "> `real ham` is `public_ham_english` (held-out UCI ham); "
              "`safe-chat probe` is the frozen 200-row generated probe. Both are "
              "small: the intervals, not the point estimate, are the claim.", ""]
    strict = next((r for r in scores if r["mode"] == "strict"), None)
    if strict and strict["probe_fpr"] is not None and strict["probe_fpr"] > 0.10:
        lines += [f"> **NOT production-ready: even `strict` still false-alarms on "
                  f"{strict['probe_fpr']:.1%} of the safe-chat probe "
                  f"({strict['probe_fp']}/{strict['probe_n']}, 95% CI "
                  f"{strict['probe_fpr_ci'].split('(')[1].split(')')[0]}).** "
                  "The Phase 5 goal was under 10%; this model does not meet it.", ""]
    return lines


def markdown(fitted: dict, scored: dict, holdout: dict,
             sweep: list[dict] | None = None) -> list[str]:
    th = fitted["thresholds"]
    cells = fitted["cell_counts"]
    lines = [
        "## Calibration (validation only)",
        "",
        f"Model: **{fitted['model']}**. Temperature and thresholds are fitted on "
        f"the **validation split only** ({fitted['val_rows']:,} rows); a code "
        "guard raises if any test row is passed to the fitter.",
        "",
        "**Why val is reweighted:** real val is 98.3% scam and ~87% of it is one "
        "source. Fitting on raw val optimises for \"flag real-looking text as "
        "scam\". We weight the four cells (scam|safe x synthetic|real) equally "
        "so the safe evidence is not drowned out.",
        "",
        "| validation cell | rows |",
        "|---|---:|",
    ]
    for cell, n in sorted(cells.items()):
        lines.append(f"| `{cell}` | {n:,} |")
    lines += [
        "",
        f"- temperature (reweighted fit) = **{fitted['temperature']:.4f}**; "
        f"sensitivity, raw-val fit = {fitted['temperature_raw_val']:.4f} "
        "(T > 1 means the raw model is over-confident).",
        f"- ECE on validation: raw **{_fmt(fitted['ece_raw'])}** -> calibrated "
        f"**{_fmt(fitted['ece_calibrated'])}** (weighted).",
        f"- `t_high` = **{_fmt(th['t_high'], 3)}** (scam-verdict precision "
        f"{_fmt(th['precision_at_t_high'])}; {th['n_scam_behind_t_high']:,} "
        "scam rows behind it).",
        f"- `t_low` = **{_fmt(th['t_low'], 3)}** (safe-verdict NPV "
        f"{_fmt(th['npv_at_t_low'])}; {th['n_safe_behind_t_low']:,} safe rows "
        "behind it).",
        "",
    ]
    lines += _modes_lines(fitted)
    if not th.get("abstention_available", True):
        lines += [f"> **The abstain band is empty at this target.** "
                  f"`n_rows_in_band` = {th.get('n_rows_in_band', 0)}: validation "
                  f"has no rows between `t_low` and `t_high`, so the two bars "
                  f"meet at one cut ({_fmt(th['t_low'], 3)}). The three-way "
                  "decision then never returns `suspicious` and coverage is "
                  "100% by construction - a fact about this model's scores, not "
                  "a tuning choice. The sweep below shows every alternative "
                  "target.", ""]
    if fitted["n_real_safe_val"] < 30:
        lines += [f"> **Thin evidence:** only {fitted['n_real_safe_val']} REAL "
                  "safe validation rows exist. `t_low` therefore rests mostly on "
                  f"synthetic safe rows (n={fitted['n_synth_safe_val']}); treat "
                  "every safe-side threshold as provisional until real ham "
                  "grows.", ""]

    lines += ["### Sensitivity: the same fit per validation half", "",
              "| fit | t_high | precision | n above | t_low | NPV | n below |",
              "|---|---|---|---:|---|---|---:|"]
    for label, row in (("reweighted val (primary)",
                        fitted["thresholds"]),
                       ("raw val", fitted["thresholds_raw_val_fit"])):
        lines.append(
            f"| {label} | {_fmt(row['t_high'], 3)} | "
            f"{_fmt(row['precision_at_t_high'])} | {row['n_definitive_scam']} | "
            f"{_fmt(row['t_low'], 3)} | {_fmt(row['npv_at_t_low'])} | "
            f"{row['n_definitive_safe']} |")
    for part, row in sorted(fitted["thresholds_sensitivity"].items()):
        c = row["cell_counts"]
        lines.append(
            f"| {part} only (scam {c['scam']}, safe {c['safe']}) | "
            f"{_fmt(row['t_high'], 3)} | {_fmt(row['precision_at_t_high'])} | "
            f"{row['n_definitive_scam']} | {_fmt(row['t_low'], 3)} | "
            f"{_fmt(row['npv_at_t_low'])} | {row['n_definitive_safe']} |")
    lines += ["", "> `n` columns are validation counts behind each threshold. "
              "A part with few safe rows cannot place `t_low` at all (n/a).", ""]

    lines += ["### Calibration on every reported stratum", "",
              "| stratum | n | n safe | ECE raw | ECE calibrated | mean p raw "
              "| mean p calibrated |", "|---|---:|---:|---|---|---|---|"]
    for row in scored["targets"]:
        flag = "" if not too_small(row["n_rows"]) else " *(small)*"
        lines.append(
            f"| `{row['target']}`{flag} | {row['n_rows']:,} | {row['n_safe']:,} | "
            f"{_fmt(row['ece_raw'])} | {_fmt(row['ece_calibrated'])} | "
            f"{_fmt(row['mean_p_raw'], 3)} | {_fmt(row['mean_p_calibrated'], 3)} |")
    lines += ["", "> ECE here is computed on each stratum with its own rows "
              "(single-class strata included: a stratum of all-scam rows with "
              "mean p 0.99 is miscalibrated by 0.01).", ""]

    lines += ["### Abstention band on the reported strata (at the fixed point)", "",
              "| stratum | n | scam recall @t_high | FPR @t_high | coverage "
              "| accuracy on covered | band rows |", "|---|---:|---|---|---|---|---:|"]
    for row in scored["targets"]:
        flag = " *(small)*" if too_small(row["n_rows"]) else ""
        acc = ("n/a" if row["accuracy_on_covered"] is None
               else f"{row['accuracy_on_covered']:.3f}")
        cov = ("n/a" if row["coverage"] is None else f"{row['coverage']:.1%}")
        lines.append(
            f"| `{row['target']}`{flag} | {row['n_rows']:,} | "
            f"{row['scam_recall_at_t_high']:.3f} | {row['fpr_at_t_high']:.3f} | "
            f"{cov} | {acc} | {row['n_suspicious']:,} |")
    lines += ["", "> Strata flagged *(small)* sit under the n-gate; the two "
              "user strata (n=5 each) are printed for completeness, never as "
              "claims.", ""]

    if sweep:
        has_holdout = any("holdout_abstain_share" in row for row in sweep)
        header = ("| precision/NPV target | t_high | t_low | val rows in band "
                  "| val band share | holdout rows abstained | holdout called "
                  "scam | safe rows behind t_low |" if has_holdout else
                  "| precision/NPV target | t_high | t_low | val rows in band "
                  "| val band share | safe rows behind t_low |")
        lines += ["### What a different target would buy", "", header,
                  "|---|---|---|---:|---:|---:|---:|---:|" if has_holdout else
                  "|---|---|---|---:|---:|---:|"]
        for row in sweep:
            base = (f"| {row['target']:.2f} | {_fmt(row['t_high'], 3)} | "
                    f"{_fmt(row['t_low'], 3)} | {row['band_rows']:,} | "
                    f"{row['band_share']:.1%} |")
            if has_holdout:
                abst = row.get("holdout_abstain_share")
                scam = row.get("holdout_scam_share")
                base += (f" {'n/a' if abst is None else f'{abst:.1%}'} | "
                         f"{'n/a' if scam is None else f'{scam:.1%}'} | ")
            base += f" {row['n_safe_behind_t_low']:,} |"
            lines.append(base)
        lines += ["", "> Fitted on validation with the same equal-cell weights. "
                  "`val rows in band` counts validation rows the rule would "
                  "send to a human; `holdout rows abstained` counts the "
                  "human-labelled `suspicious` band that would reach that "
                  "human. A rule that abstains on neither is not abstention.", ""]

    lines += ["### Suspicious band on the `suspicious` holdout", ""]
    if not holdout.get("available"):
        lines += [f"(no holdout available: {holdout.get('reason', holdout.get('path'))})",
                  ""]
    else:
        counts = holdout["counts"]
        n_abstain = counts.get("suspicious", 0)
        if n_abstain == 0:
            reading = ("**The model claims certainty on every row a human could "
                       "not label.** Zero abstentions means the third verdict is "
                       "never used on exactly the traffic that needs it: the "
                       f"safest reading is that {counts.get('scam', 0):,} of these "
                       "rows would be pushed to a user as confident scams. "
                       "Caveat: these rows are risky-but-ambiguous by "
                       "construction (promos, loan offers, chain messages), so "
                       "`scam` is not automatically wrong - what is missing is "
                       "any willingness to defer. Either the band is "
                       "mis-specified for this model or the label `suspicious` "
                       "and the probability scale disagree - both are real "
                       "findings, neither is a tuning detail.")
        else:
            reading = ("These rows are unsure by construction, so most of them "
                       "landing in the band is the behaviour the band exists for; "
                       "the scam/safe shares are the confident minority a human "
                       "should still review.")
        lines += [
            f"{holdout['n_rows']:,} rows that a human labelled **`suspicious`** "
            "(the unsure band, never trained on) land as:",
            "",
            "| verdict at the fitted point | rows | share |",
            "|---|---:|---:|",
            f"| suspicious (abstain) | {n_abstain:,} | "
            f"{n_abstain / holdout['n_rows']:.1%} |",
            f"| scam | {counts.get('scam', 0):,} | "
            f"{counts.get('scam', 0) / holdout['n_rows']:.1%} |",
            f"| safe | {counts.get('safe', 0):,} | "
            f"{counts.get('safe', 0) / holdout['n_rows']:.1%} |",
            "",
            f"Mean calibrated p = {holdout['mean_p']:.3f} (median "
            f"{holdout['median_p']:.3f}). {reading}",
            "",
        ]
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: temperature scaling + abstention thresholds")
    parser.add_argument("--model", default="verdict_lr_plus",
                        help="run name in outputs/ or a baseline name")
    args = parser.parse_args(argv)

    fitted = fit_on_validation(args.model)
    fitted["modes"] = fit_all_modes(args.model, fitted)
    fitted["mode_scores"] = evaluate_modes(args.model, fitted, fitted["modes"])
    save_operating_point(fitted)
    save_reliability_bins(args.model, fitted)
    holdout = holdout_band(args.model, fitted)
    sweep = coverage_sweep(args.model, fitted, holdout.get("probs"))
    fitted["coverage_sweep"] = sweep
    scored = score_targets(args.model, fitted)
    text = "\n".join(markdown(fitted, scored, holdout, sweep))
    print("\n" + text)
    print("\nNext: python -m src.error_analysis --model", args.model)


if __name__ == "__main__":
    main()
