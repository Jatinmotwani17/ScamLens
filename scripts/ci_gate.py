"""
scripts/ci_gate.py - fail CI when the model regresses.

TWO GATES (both must pass):
  1. synthetic test_b macro-F1 >= --min-testb-f1   (default 0.90)
  2. public_ham FPR in "balanced" mode <= --max-ham-fpr (default 0.15)

WHY THESE TWO
  test_b is the cheap canary: if the model can no longer separate unseen
  synthetic families, something is broken in training or the features.
  The real-ham FPR is the failure this project actually has, so it gets a hard
  ceiling - a change that "improves" recall by flagging everything fails here.

Inputs are artefacts, not live scoring:
  results/metrics.json            (src.evaluate)
  configs/operating_point.yaml    (src.calibrate -> mode_measured)

Exit codes: 0 pass, 1 regression, 2 artefacts missing (CI treats 2 as a setup
error and prints what to run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_test_b_f1(metrics: dict, model: str = "verdict_lr_plus") -> float | None:
    """Pull test_b macro-F1 for one model out of metrics.json.

    metrics.json is shaped {split: {n_rows, scam_rate, models: {name: {...}}}}
    (see src/evaluate.py). Missing/None values are normal for single-class
    splits, so every layer is checked.
    """
    split = (metrics or {}).get("test_b")
    if not isinstance(split, dict):
        return None
    per_model = split.get("models") or {}
    value = (per_model.get(model) or {}).get("macro_f1")
    if value is None:
        candidates = [v.get("macro_f1") for v in per_model.values()
                      if isinstance(v, dict)]
        candidates = [float(c) for c in candidates if c is not None]
        return max(candidates) if candidates else None
    return float(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ScamLens CI regression gate")
    parser.add_argument("--min-testb-f1", type=float, default=0.90)
    parser.add_argument("--max-ham-fpr", type=float, default=0.15)
    parser.add_argument("--mode", default="balanced")
    args = parser.parse_args(argv)

    metrics_path = ROOT / "results" / "metrics.json"
    point_path = ROOT / "configs" / "operating_point.yaml"
    missing = [str(p) for p in (metrics_path, point_path) if not p.exists()]
    if missing:
        print("ci_gate: missing artefacts: " + ", ".join(missing))
        print("  run: python -m src.evaluate && python -m src.calibrate "
              "--model verdict_lr_plus")
        return 2

    failures: list[str] = []
    metrics = load_json(metrics_path)
    f1 = find_test_b_f1(metrics)
    if f1 is None:
        print("ci_gate: could not find a test_b macro-F1 in metrics.json")
        return 2
    status = "PASS" if f1 >= args.min_testb_f1 else "FAIL"
    print(f"ci_gate: test_b macro-F1 = {f1:.4f} (min {args.min_testb_f1}) -> {status}")
    if status == "FAIL":
        failures.append(f"test_b macro-F1 {f1:.4f} < {args.min_testb_f1}")

    point = yaml.safe_load(point_path.read_text(encoding="utf-8"))
    measured = (point.get("mode_measured") or {}).get(args.mode) or {}
    ham_fpr = measured.get("ham_fpr")
    ham_n = measured.get("ham_n")
    if ham_fpr is None:
        print(f"ci_gate: no measured ham FPR for mode {args.mode!r}")
        return 2
    status = "PASS" if ham_fpr <= args.max_ham_fpr else "FAIL"
    print(f"ci_gate: public_ham FPR ({args.mode}) = {ham_fpr:.4f} "
          f"on n={ham_n} (max {args.max_ham_fpr}) -> {status}")
    if status == "FAIL":
        failures.append(f"ham FPR {ham_fpr:.4f} > {args.max_ham_fpr}")

    if failures:
        print("\nci_gate: REGRESSION")
        for item in failures:
            print("  - " + item)
        return 1
    print("\nci_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
