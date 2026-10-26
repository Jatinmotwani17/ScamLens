# Claims we may and may not make

Every allowed claim is tied to a table and an `n`. If a sentence is not on the allowed
list, do not write it in a CV, README, tweet, or interview answer.

## ALLOWED (each tied to evidence in this repo)

1. **"I built a leakage-audited evaluation pipeline: cluster/family-grouped splits, a
   hard assertion that no synthetic row reaches a real test set, and a per-tag support
   table."** → `src/splits35.py`, `data_checks` leakage PASS,
   `results/RESULTS.md` per-tag support.
2. **"I found and documented a dataset-source shortcut: a classifier can tell real from
   synthetic text with AUROC 0.996, so I report real and synthetic strata separately
   and put a `source_only` baseline beside every model number."** →
   `results/RESULTS.md` (shortcut audit), `results/PHASE4_REPORT.md`.
3. **"I reduced false alarms on real English ham from 15.3% (n=59) to 6.4%
   (95% CI 4.2–9.5, n=345) in strict mode while keeping scam recall at 99.2%
   (95% CI 98.8–99.4, n=3,511)."** → `configs/operating_point.yaml`,
   `results/PHASE4_REPORT.md`. (Say also that the ham reserve grew from 400 to 2,400
   UCI rows, so the old n=59 was noisy, and that this ham is mostly
   conversational: only 5 of 345 rows are bank/OTP text, 4 of which are flagged.)
4. **"On a frozen 200-message safe-chat probe set the strict mode flags 5.5%
   (95% CI 3.1–9.6, n=200), down from 13.0% before the hard-negative work."** →
   `data/processed/safe_chat_probe_frozen.parquet`, `results/PHASE4_REPORT.md`.
5. **"I calibrated the probabilities: val ECE 0.030 → 0.005 after temperature scaling
   (T=0.4614)."** → `configs/operating_point.yaml`, `results/reliability_bins.csv`.
6. **"I shipped three named operating points fitted on weighted validation only, with
   Wilson intervals on every rate, instead of one blended accuracy number."** →
   `configs/operating_point.yaml`, `results/PHASE4_REPORT.md`.
7. **"I implemented a rule-based abstention band and measured its cost: 1.1% of clear
   real scams (40/3,511) are deferred, while 45.3% (355/783) of human-ambiguous
   messages are correctly deferred."** → `results/suspicious_band.md`.
8. **"I built a serving layer with a no-text-logging guarantee enforced by a test,
   mode switching, per-mode thresholds, and a fallback path."** → `tests/test_api.py`,
   `src/api.py`.
9. **"I wrote a robustness probe suite (homoglyph/leet/spacing/urgency edits, Hinglish
   transfer) and report worst-case recall change ≈0.1 pt — explicitly labelled as
   probes, not field performance."** → `results/robustness.md`.
10. **"Synthetic test_b macro-F1 is 0.995 (n=215)"** — but never without "synthetic".

## NOT ALLOWED

- Any claim of **real-world Hinglish accuracy**. All real evaluation data is English;
  Hinglish evidence is synthetic + translation-derived probes only.
- Any **blended real-data accuracy** ("97% accurate on real messages"). Report per
  stratum with `n`; scam recall and ham FPR are not interchangeable.
- Any statement about the precision of the **six unvalidated tags**
  (`payment_or_upi_request`, `prize_or_lottery`, `otp_or_pin_request`,
  `investment_or_trading_scam`, `job_or_task_scam`, `loan_app_harassment`).
- **"Production-ready", "deployed", "battle-tested", "state of the art".** Strict mode
  still false-alarms on 6.4% of real ham; that is a shipped-blocker for real use.
- Any claim that the model **detects new/novel scams**. It detects patterns resembling
  its training distribution; the probes are frozen, not adversarial.
- Any claim that the **transformer was shipped**. E1/E1s artifacts were absent, so the
  transformer comparison was **not run**; MuRIL is config-switchable only.
- Any claim that **`suspicious` is a calibrated uncertainty band**. It is a rule.
- Any implied **superiority over commercial detectors**; none were benchmarked.
