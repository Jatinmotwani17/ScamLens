"""
src/run_experiments.py - run E1, E2, E3 (and E4 only if the gate allows).

EXPERIMENTS
  E1  MuRIL, verdict + tags       (lambda_tags=0.5) - the main model
  E2  MuRIL, verdict only         (lambda_tags=0)   - do tags help?
  E3  XLM-R base, verdict + tags  - model comparison
  E4  cluster-grouped 5-fold CV on real messages - ONLY if the gate passes
      (needs >=50 in-domain real messages AND >=20 user_real_safe)

Each experiment is a normal `src/train.py` run followed by `src/predict.py`.
Checkpoint selection is VALIDATION ONLY; test sets are logged for observation.

Run:
  python -m src.run_experiments --dry_run
  python -m src.run_experiments
  python -m src.run_experiments --skip_existing
  python -m src.run_experiments --only e1_muril e3_xlmr
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT, SEED
from src.e4_cv import gate_ok

EXPERIMENTS = [
    {"name": "e1_muril", "model": "google/muril-base-cased", "lambda_tags": 0.5,
     "mix": "mix_synth_plus_real", "required": True,
     "note": "MuRIL, verdict + tags, synth+real (main model)", "minutes_t4": 50},
    {"name": "e1s_muril_synthonly", "model": "google/muril-base-cased",
     "lambda_tags": 0.5, "mix": "mix_synth_only", "required": True,
     "note": "MuRIL, synth only (value of real data)", "minutes_t4": 35},
    {"name": "e2_muril_verdict_only", "model": "google/muril-base-cased",
     "lambda_tags": 0.0, "mix": "mix_synth_plus_real", "required": False,
     "note": "OPTIONAL: MuRIL verdict only (tag ablation)", "minutes_t4": 40},
    {"name": "e3_xlmr", "model": "xlm-roberta-base", "lambda_tags": 0.5,
     "mix": "mix_synth_plus_real", "required": False,
     "note": "OPTIONAL: XLM-R, verdict + tags", "minutes_t4": 55},
]


def verify_model_id(model_id: str) -> bool:
    """Check the id actually exists on the Hub before spending GPU time.

    Uses transformers' AutoConfig (strongest: downloads the config), else the
    public Hub API. We NEVER guess an id - an unreachable Hub means
    "unverified", and the run is skipped instead of silently training a typo.
    """
    try:
        from transformers import AutoConfig
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        AutoConfig.from_pretrained(model_id)
        print(f"  model id OK (config downloaded): {model_id}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  MODEL UNAVAILABLE: {model_id} ({type(e).__name__}: {str(e)[:120]})")
        return False

    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            ok = resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
        if ok:
            print(f"  model id OK (Hub API): {model_id} "
                  f"[{data.get('pipeline_tag', 'unknown task')}]")
        return ok
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"  UNVERIFIED: {model_id} (Hub API error: {type(e).__name__})")
        print("  -> not training on an unverified id; install transformers or "
              "check your network, then rerun.")
        return False


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def selection_value(row: dict) -> float:
    """Validation-only score that picked the checkpoint (Phase 3.6).

    Old logs (pre-3.6) only carry val_macro_f1, so we fall back to it rather
    than crash. Test metrics are never read here.
    """
    value = row.get("selection_score")
    if value is not None:
        return float(value)
    return float(row.get("val_macro_f1") or 0.0)


def read_best(run_dir: Path) -> dict | None:
    log = run_dir / "training_log.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return max(rows, key=selection_value) if rows else None


def _stratum_value(best: dict, stratum: str, key: str) -> str:
    for s in best.get("test_real_strata", []):
        if s["stratum"] == stratum:
            v = s.get(key)
            return "n/a" if v is None else f"{float(v):.3f}"
    return "—"


def summary_table() -> None:
    """Print the best epoch of every run, selected on VALIDATION only."""
    outputs = PROJECT_ROOT / "outputs"
    if not outputs.exists():
        print("(no outputs/ yet)")
        return
    lines = ["| run | mix | best epoch | selection score | val synth macro-F1 "
             "| val real bal-acc | test_a | test_b "
             "| user_real_scam recall | external_scam recall | public_ham FPR |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    found = False
    for run in sorted(p for p in outputs.iterdir() if p.is_dir()):
        best = read_best(run)
        if not best:
            continue
        found = True

        def _num(key: str) -> str:
            value = best.get(key)
            return "n/a" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {run.name} | {best.get('train_mix', '?')} | {best['epoch']} "
            f"| {selection_value(best):.4f} "
            f"| {_num('val_synth_macro_f1')} "
            f"| {_num('val_real_balanced_accuracy')} "
            f"| {_num('test_a_macro_f1')} "
            f"| {_num('test_b_macro_f1')} "
            f"| {_stratum_value(best, 'user_real_scam', 'scam_recall')} "
            f"| {_stratum_value(best, 'external_scam', 'scam_recall')} "
            f"| {_stratum_value(best, 'public_ham_english', 'fpr_on_safe')} |")
    print("\n".join(lines) if found else "(no completed run logs found)")
    print("\nBest epoch is chosen on VALIDATION only (mean of val_synthetic "
          "macro-F1 and val_real balanced accuracy). test_* columns are for "
          "observation and never selected the checkpoint.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 experiments E1-E4")
    parser.add_argument("--dry_run", action="store_true",
                        help="verify model ids / gate, print commands, run nothing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip a run whose best checkpoint already exists")
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only these experiment names")
    parser.add_argument("--include_optional", action="store_true",
                        help="also run optional E2 (verdict-only) and E3 (XLM-R)")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="do not run src.evaluate at the end")
    args = parser.parse_args(argv)

    print(f"(seed={SEED}) src/run_experiments.py")
    print("=" * 60)
    print("GPU estimates are for a free T4. CPU is roughly 8-20x slower; if this")
    print("machine has no CUDA, use --dry_run and run the training on Colab/Kaggle.")

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    selected = [e for e in EXPERIMENTS if e["required"] or args.include_optional]
    if args.only:
        wanted = set(args.only)
        selected = [e for e in EXPERIMENTS if e["name"] in wanted]
        unknown = wanted - {e["name"] for e in EXPERIMENTS}
        if unknown:
            raise SystemExit(f"unknown experiment name(s): {sorted(unknown)}")

    print("\nExperiment plan:")
    for exp in selected:
        print(f"  {exp['name']:26s} ~{exp['minutes_t4']} min on T4  "
              f"[{exp['mix']}] - {exp['note']}")
    if not args.include_optional and not args.only:
        print("  (E2 verdict-only and E3 XLM-R are OPTIONAL and OFF by default; "
              "pass --include_optional to run them.)")

    # ---- E4 gate is checked BEFORE any GPU time is spent -----------------
    from src.e4_cv import load_frames
    frames = load_frames()
    ok, n_in, n_safe = gate_ok(frames["test_real"])
    print(f"\nE4 GATE: in-domain={n_in}, user_real_safe={n_safe} -> "
          f"{'RUN' if ok else 'SKIP'}")
    if not ok:
        print("!" * 70)
        print("E4 SKIPPED: need >=50 in-domain real messages and >=20 "
              "user_real_safe hard negatives. A 10-row CV is not a result.")
        print("!" * 70)

    for exp in selected:
        print(f"\n=== {exp['name']}: {exp['note']} ===")
        if not verify_model_id(exp["model"]):
            print("  -> SKIPPED (do not guess a model id)")
            continue
        run_dir = PROJECT_ROOT / str(cfg.get("output_dir", "outputs")) / exp["name"]
        if args.skip_existing and (run_dir / cfg.get("checkpoint_name", "best_model.pt")).exists():
            print(f"  -> complete checkpoint exists, skipping ({run_dir})")
            continue
        cmd = [sys.executable, "-m", "src.train",
               "--config", args.config, "--model", exp["model"],
               "--run_name", exp["name"], "--lambda_tags", str(exp["lambda_tags"]),
               "--train_mix", exp["mix"]]
        predict_cmd = [sys.executable, "-m", "src.predict", "--run_name", exp["name"]]
        if args.dry_run:
            print("  " + " ".join(cmd))
            print("  " + " ".join(predict_cmd))
        else:
            run_cmd(cmd)
            run_cmd(predict_cmd)

    e4_cmd = [sys.executable, "-m", "src.e4_cv", "--config", args.config]
    print("\n=== e4_real_cv ===")
    if args.dry_run:
        print("  " + " ".join(e4_cmd) + ("    (gate PASSES)" if ok else "    (will SKIP: gate failed)"))
    else:
        run_cmd(e4_cmd)

    if args.dry_run:
        print("\nDRY RUN complete - no training was started.")
        return

    print("\nSummary (best epoch by VAL macro-F1 only):")
    summary_table()

    if not args.skip_evaluate:
        run_cmd([sys.executable, "-m", "src.evaluate"])


if __name__ == "__main__":
    main()
