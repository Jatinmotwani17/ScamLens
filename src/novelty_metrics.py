"""
src/novelty_metrics.py - performance bucketed by nearest-train similarity.

THE QUESTION THIS ANSWERS
  "How well does the model do on messages it has NOT seen a near-copy of?"
  Every row already has `max_train_sim` in results/nn_sim.parquet (Phase 3.6:
  the maximum TF-IDF char-ngram cosine similarity to ANY train row). Bucketing
  the scores by that number turns the evaluation into the honest generalisation
  measure: a model that only works at high similarity is a copy-detector.

We report, per bucket (<0.5, 0.5-0.7, 0.7-0.9, >0.9):
  n, n_scam, n_safe, scam recall at the FIXED operating point, FPR, AUROC
  (only when a bucket holds both classes) and the mean calibrated probability.

`n` is printed next to every number and buckets under 30 rows are marked
"too small for claims" - that rule applies to real rows and to the 224-row
test_b alike.

Run:
  python -m src.novelty_metrics --model verdict_lr_plus
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.calibrate import load_operating_point
from src.config import SEED, data_path
from src.phase4_common import (
    apply_temperature,
    ece,
    load_predictor,
    predictions_for_split,
    too_small,
)

# Bucket edges on max_train_sim. The top bucket is open-ended (> 0.9).
BUCKETS = ("0.0-0.5", "0.5-0.7", "0.7-0.9", ">0.9")


def bucket_of(value: float) -> str:
    if value > 0.9:
        return ">0.9"
    if value >= 0.7:
        return "0.7-0.9"
    if value >= 0.5:
        return "0.5-0.7"
    return "0.0-0.5"


def load_nn_similarity() -> pd.DataFrame:
    path = data_path("nn_sim")
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run: python -m src.nn_sim (Phase 3.6 audit)")
    return pd.read_parquet(path)


def _metrics(frame: pd.DataFrame, t_high: float) -> dict:
    """Recall / FPR / AUROC for one bucket of rows."""
    y = (frame["verdict"].astype(str) == "scam").to_numpy(dtype=float)
    p = frame["prob_cal"].to_numpy(dtype=float)
    n_scam, n_safe = int(y.sum()), int((y == 0).sum())
    out = {
        "n": int(len(frame)),
        "n_scam": n_scam,
        "n_safe": n_safe,
        "scam_recall": (float((p[y == 1] >= t_high).sum() / n_scam)
                        if n_scam else None),
        "fpr": (float((p[y == 0] >= t_high).sum() / n_safe)
                if n_safe else None),
        "auroc": None,
        "mean_p": float(p.mean()) if len(p) else None,
        "ece": ece(y, p, None) if len(p) else None,
        "too_small_for_claims": too_small(len(frame)),
    }
    if n_scam and n_safe:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def bucket_results(name: str) -> dict:
    point = load_operating_point(name)
    temperature = float(point["calibration"]["temperature"])
    t_high = float(point["operating_point"]["t_high"])

    nn = load_nn_similarity()
    predictor = load_predictor(name)
    splits = {"test_b": "test_b", "test_real": "test_real",
              "val": "val"}
    frames: dict[str, pd.DataFrame] = {}
    for split in splits:
        preds = predictions_for_split(predictor, name, split)
        if preds.empty:
            continue
        merged = preds.merge(
            nn[["id", "max_train_sim"]], on="id", how="left", validate="one_to_one")
        merged["bucket"] = merged["max_train_sim"].map(
            lambda v: bucket_of(float(v)) if pd.notna(v) else "unknown")
        merged["prob_cal"] = apply_temperature(
            merged["prob_raw"].to_numpy(dtype=float), temperature)
        frames[split] = merged

    results: dict = {"model": name, "temperature": temperature, "t_high": t_high,
                     "targets": {}}

    def add(target: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        rows = {}
        for bucket in list(BUCKETS) + ["unknown"]:
            sub = frame[frame["bucket"] == bucket]
            if sub.empty:
                continue
            rows[bucket] = _metrics(sub, t_high)
        results["targets"][target] = rows

    for split, frame in frames.items():
        if split == "test_real" and "real_stratum" in frame.columns:
            for strat in ("external_scam", "public_ham_english",
                          "user_real_scam", "user_real_safe"):
                sub = frame[frame["real_stratum"] == strat].reset_index(drop=True)
                add(f"test_real/{strat}", sub)
        else:
            add(split, frame)
    return results


def markdown(results: dict) -> list[str]:
    lines = [
        f"# Novelty-bucketed metrics - `{results['model']}`",
        "",
        "> **This is the honest generalisation measure.** Each row is bucketed "
        "by `max_train_sim` - the maximum TF-IDF char-ngram cosine similarity "
        "to any TRAIN row (results/nn_sim.parquet). Performance that only "
        "holds at high similarity is near-copy matching, not detection.",
        "",
        f"Decisions use the fixed operating point (t_high = "
        f"{results['t_high']:.3f}, temperature = {results['temperature']:.3f}) "
        "fitted on validation only.",
        "",
    ]
    for target, buckets in results["targets"].items():
        lines += [f"## `{target}`", "",
                  "| max_train_sim | n | n scam | n safe | scam recall "
                  "@t_high | FPR | AUROC | mean p | ECE | note |",
                  "|---|---:|---:|---:|---|---|---|---|---|---|"]
        for bucket in list(BUCKETS) + ["unknown"]:
            row = buckets.get(bucket)
            if not row:
                continue
            def fmt(value, digits=3):
                return "n/a" if value is None else f"{value:.{digits}f}"
            note = "too small for claims" if row["too_small_for_claims"] else ""
            if row["auroc"] is None:
                note = (note + "; single class in bucket").strip("; ")
            lines.append(
                f"| {bucket} | {row['n']:,} | {row['n_scam']:,} | "
                f"{row['n_safe']:,} | {fmt(row['scam_recall'])} | "
                f"{fmt(row['fpr'])} | {fmt(row['auroc'])} | "
                f"{fmt(row['mean_p'])} | {fmt(row['ece'], 4)} | {note} |")
        lines.append("")
    lines += ["> AUROC needs both classes in a bucket; single-class buckets "
              "report recall/FPR only. Buckets under 30 rows are printed but "
              "are not claims.", ""]
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: metrics bucketed by nearest-train similarity")
    parser.add_argument("--model", default="verdict_lr_plus")
    args = parser.parse_args(argv)

    print(f"(seed={SEED}) src/novelty_metrics.py  model={args.model}")
    print("=" * 60)
    results = bucket_results(args.model)
    out = data_path("novelty_md")
    out.write_text("\n".join(markdown(results)), encoding="utf-8")
    print(f"WROTE {out}")
    for target, buckets in results["targets"].items():
        cells = ", ".join(f"{b}: n={r['n']} "
                          f"recall={'n/a' if r['scam_recall'] is None else f'{r['scam_recall']:.3f}'}"
                          for b, r in buckets.items())
        print(f"  {target:30s} {cells}")
    return results


if __name__ == "__main__":
    main()
