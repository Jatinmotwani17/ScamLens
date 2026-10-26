# ScamLens - evaluation results (BINARY verdict)

> Synthetic results are not performance claims. Headline numbers come only from test_real, and always PER STRATUM.

> `suspicious` is an UNSURE BAND, not a class. It is never trained or scored here; Phase 4 uses the holdout for confidence-based abstention.

> Checkpoint selection: **validation only: mean(val_synthetic macro-F1, val_real balanced accuracy)**. Test splits are observation only.

## Data provenance

Training mix: **mix_synth_plus_real**. Every experiment reports its own mix; `mix_synth_only` excludes real rows.

| split | rows | sources |
|---|---:|---|
| `train` | 15,873 | external_reported 8390, synthetic 5873, public_ham_eval 1610 |
| `val` | 4,540 | external_reported 3511, synthetic 684, public_ham_eval 345 |
| `test_a` | 667 | synthetic 667 |
| `test_b` | 215 | synthetic 215 |
| `test_real` | 3,866 | external_reported 3511, public_ham_eval 345, self_collected 10 |

**Real script_style distribution** (17,722 real rows): {'english': 17713, 'roman_hinglish': 9}. Roman Hinglish = **9** rows (0.05%).

External scam styles: {'english': 15410, 'roman_hinglish': 2}  |  public ham styles: {'english': 2299, 'roman_hinglish': 1}

**Per-tag training support** (rows containing the tag; the splitter and tests require ≥100 in train, ≥1 in val and test_b):

| tag | train | val | test_a | test_b | real test |
|---|---:|---:|---:|---:|---:|
| otp_or_pin_request | 195 | 45 | 0 | 17 | 0 |
| urgency_threat | 7132 | 2549 | 181 | 48 | 2481 |
| bank_kyc_impersonation | 3764 | 1342 | 16 | 34 | 1341 |
| authority_impersonation | 6871 | 2647 | 100 | 14 | 2635 |
| payment_or_upi_request | 758 | 164 | 170 | 11 | 3 |
| suspicious_link | 8756 | 3557 | 97 | 60 | 3513 |
| prize_or_lottery | 197 | 27 | 0 | 36 | 1 |
| job_or_task_scam | 141 | 26 | 0 | 31 | 0 |
| investment_or_trading_scam | 170 | 31 | 0 | 15 | 0 |
| loan_app_harassment | 128 | 32 | 0 | 31 | 0 |
| parcel_courier_scam | 1269 | 453 | 72 | 11 | 402 |
| utility_disconnection_scam | 925 | 359 | 38 | 34 | 336 |
| relative_in_trouble | 257 | 62 | 0 | 15 | 31 |

> **`test_a` (Phase 3.6):** 10% of the TRAIN clusters, held out whole and BY CLUSTER, drawn from both scam and safe pools. It therefore has both classes and a real macro-F1, and it measures seen-template wording. `test_b` is the unseen-family split.

## Nearest-neighbour leakage audit (train similarity)

> For every val/test row we computed the maximum TF-IDF char-ngram cosine similarity to ANY train row (`results/nn_sim.parquet`). A high value means the row has a near-copy in training, so its score partly measures memorisation.

| stratum | rows | sim > 0.9 | 0.7-0.9 | < 0.7 | median sim |
|---|---:|---:|---:|---:|---:|
| `val_synthetic` | 684 | 0.0% | 1.6% | 98.4% | 0.447 |
| `val_real` | 3,856 | 1.3% | 14.5% | 84.3% | 0.436 |
| `test_a` | 667 | 0.0% | 2.7% | 97.3% | 0.487 |
| `test_b` | 215 | 0.0% | 0.5% | 99.5% | 0.439 |
| `user_real_scam` | 5 | 0.0% | 0.0% | 100.0% | 0.371 |
| `user_real_safe` | 5 | 0.0% | 0.0% | 100.0% | 0.352 |
| `external_scam` | 3,511 | 1.7% | 16.4% | 81.9% | 0.449 |
| `public_ham_english` | 345 | 0.3% | 0.9% | 98.8% | 0.313 |

> 1.5% of the 3,866 real-test rows have a train near-copy (sim > 0.9), so this is not a copy-matching test set.

## Real-data gate (E4)

- in-domain real messages (`user_real_scam` + `user_real_safe`): **10**
- `user_real_safe` hard negatives: **5**
- requirement: >=50 in-domain and >=20 user_real_safe
- **E4 SKIPPED.** A cluster-grouped CV on 10 real rows would be noise presented as a result. Real performance below comes from the public strata only; the two user strata are too small.

## test set: `val` (combined val (synthetic + real))

4540 rows, scam rate 82.7%, style mix: {'english': 4122, 'roman_hinglish': 277, 'devanagari': 119, 'mixed': 22}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.453 | 0.905 | [0.449, 0.456] | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.246 | 0.061 | [0.170, 0.190] | 0.032 | 0.010 | 0.026 | n/a | n/a | n/a | n/a | 0.170 | 0.172 |
| TF-IDF + LR (synth only) | 0.668 | 0.793 | [0.653, 0.683] | 0.660 | 0.020 | 0.000 | 0.940 | 0.986 | 0.554 | 0.748 | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.964 | 0.987 | [0.956, 0.971] | 0.974 | 0.003 | 0.000 | 0.998 | 1.000 | 0.985 | 0.992 | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.147 | 0.000 | [0.140, 0.155] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.136 | 0.045 |
| TF-IDF tags LR (synth+real) | 0.147 | 0.000 | [0.140, 0.155] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.265 | 0.064 |
| style_only (shortcut) | 0.679 | 0.917 | [0.660, 0.697] | 0.964 | 0.668 | 0.511 | 0.638 | 0.871 | 0.000 | 0.002 | n/a | n/a |
| source_only (shortcut) | 0.916 | 0.966 | [0.906, 0.926] | 0.935 | 0.000 | 0.000 | 0.982 | 0.993 | 0.935 | 0.935 | n/a | n/a |

### `val_synthetic` - 684 rows, scam rate 35.7%

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.263 | 0.526 | [0.244, 0.282] | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.483 | 0.200 | [0.462, 0.536] | 0.115 | 0.018 | 0.026 | n/a | n/a | n/a | n/a | 0.313 | 0.275 |
| TF-IDF + LR (synth only) | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.998 | 0.998 | [0.995, 1.000] | 0.996 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.391 | 0.000 | [0.378, 0.404] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.192 | 0.056 |
| TF-IDF tags LR (synth+real) | 0.391 | 0.000 | [0.378, 0.404] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.198 | 0.057 |
| style_only (shortcut) | 0.517 | 0.410 | [0.477, 0.553] | 0.447 | 0.407 | 0.511 | 0.499 | 0.362 | n/a (too few negatives) | 0.025 | n/a | n/a |
| source_only (shortcut) | 0.391 | 0.000 | [0.378, 0.404] | 0.000 | 0.000 | 0.000 | 0.500 | 0.357 | n/a (too few negatives) | 0.000 | n/a | n/a |

### `val_real` - 3856 rows, scam rate 91.1%

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.477 | 0.953 | [0.474, 0.479] | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.176 | 0.051 | [0.100, 0.118] | 0.026 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.159 | 0.116 |
| TF-IDF + LR (synth only) | 0.556 | 0.776 | [0.540, 0.571] | 0.636 | 0.046 | n/a | 0.874 | 0.986 | n/a (too few negatives) | 0.661 | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.930 | 0.986 | [0.916, 0.944] | 0.973 | 0.006 | n/a | 0.997 | 1.000 | n/a (too few negatives) | 0.989 | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.082 | 0.000 | [0.074, 0.089] | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.130 | 0.043 |
| TF-IDF tags LR (synth+real) | 0.082 | 0.000 | [0.074, 0.089] | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.271 | 0.064 |
| style_only (shortcut) | 0.477 | 0.953 | [0.474, 0.479] | 1.000 | 1.000 | n/a | 0.500 | 0.911 | n/a (too few negatives) | 0.000 | n/a | n/a |
| source_only (shortcut) | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | n/a | 1.000 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |

### verdict_lr_plus - confusion matrix on `val`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe            783         2
  gold_scam             97      3658
```

### keyword - confusion matrix on `val`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe            664         8
  gold_scam           1641       119
```

## test set: `test_a` (synthetic)

667 rows, scam rate 53.7%, style mix: {'english': 412, 'roman_hinglish': 205, 'devanagari': 46, 'mixed': 4}

> 10% of TRAIN clusters held out whole (both classes); measures seen-template wording, not new scams.

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.349 | 0.699 | [0.334, 0.365] | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.316 | 0.183 | [0.387, 0.454] | 0.101 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.263 | 0.121 |
| TF-IDF + LR (synth only) | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.998 | 0.999 | [0.995, 1.000] | 0.997 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.317 | 0.000 | [0.299, 0.333] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.213 | 0.055 |
| TF-IDF tags LR (synth+real) | 0.317 | 0.000 | [0.299, 0.333] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.200 | 0.052 |
| style_only (shortcut) | 0.311 | 0.421 | [0.279, 0.346] | 0.455 | 0.819 | 0.940 | 0.334 | 0.467 | n/a (too few negatives) | 0.011 | n/a | n/a |
| source_only (shortcut) | 0.317 | 0.000 | [0.299, 0.333] | 0.000 | 0.000 | 0.000 | 0.500 | 0.537 | n/a (too few negatives) | 0.000 | n/a | n/a |

### verdict_lr_plus - confusion matrix on `test_a`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe            309         0
  gold_scam              1       357
```

### keyword - confusion matrix on `test_a`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe            137         0
  gold_scam            164        36
```

## test set: `test_b` (synthetic)

215 rows, scam rate 61.4%, style mix: {'english': 80, 'roman_hinglish': 75, 'mixed': 31, 'devanagari': 29}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.380 | 0.761 | [0.352, 0.404] | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.555 | 0.360 | [0.423, 0.562] | 0.220 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.427 | 0.323 |
| TF-IDF + LR (synth only) | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.995 | 0.996 | [0.981, 1.000] | 0.992 | 0.000 | 0.000 | 1.000 | 1.000 | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.279 | 0.000 | [0.243, 0.313] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.183 | 0.053 |
| TF-IDF tags LR (synth+real) | 0.279 | 0.000 | [0.243, 0.313] | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 0.184 | 0.054 |
| style_only (shortcut) | 0.494 | 0.560 | [0.426, 0.555] | 0.515 | 0.518 | 0.500 | 0.400 | 0.595 | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| source_only (shortcut) | 0.279 | 0.000 | [0.243, 0.313] | 0.000 | 0.000 | 0.000 | 0.500 | 0.614 | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |

### verdict_lr_plus - confusion matrix on `test_b`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe             83         0
  gold_scam              1       131
```

### keyword - confusion matrix on `test_b`

rows = gold, cols = predicted (order: safe, scam)

```
           pred_safe pred_scam
  gold_safe             76         0
  gold_scam             44        29
```

## test set: `test_real` (REAL - reported PER STRATUM)

> There is deliberately **no blended real number**. A stratum with fewer than 30 rows is marked 'too small for claims'.

> A stratum with only ONE class (`external_scam` is all scam, `public_ham_english` is all safe) has **macro-F1 = n/a**. It is not 0.5 and not 0.0: there is no second class to average. Read **scam recall** and **FPR(safe)** there. The combined English block below is the only place where a real macro-F1 is defined.

### `user_real_scam` - 5 rows **TOO SMALL FOR CLAIMS**

scam rate: 100.0% | style mix: {'roman_hinglish': 3, 'english': 2}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | n/a | 1.000 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | n/a | 0.333 | n/a | 0.200 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF + LR (synth only) | n/a | 1.000 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF + LR (synth+real) | n/a | 1.000 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF tags LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF tags LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| style_only (shortcut) | n/a | 0.571 | n/a | 0.400 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| source_only (shortcut) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |

Per-tag F1 (only tags with >= 20 real positives are shown):

| tag | real positives | status | keyword | tactics_lr_synth | tactics_lr_plus |
|---|---|---|---|---|---|
| payment_or_upi_request | 3 | unvalidated (synthetic only) | — | — | — |
| suspicious_link | 2 | unvalidated (synthetic only) | — | — | — |
| urgency_threat | 2 | unvalidated (synthetic only) | — | — | — |
| authority_impersonation | 1 | unvalidated (synthetic only) | — | — | — |
| bank_kyc_impersonation | 1 | unvalidated (synthetic only) | — | — | — |
| prize_or_lottery | 1 | unvalidated (synthetic only) | — | — | — |
| relative_in_trouble | 1 | unvalidated (synthetic only) | — | — | — |
| investment_or_trading_scam | 0 | unvalidated (synthetic only) | — | — | — |
| job_or_task_scam | 0 | unvalidated (synthetic only) | — | — | — |
| loan_app_harassment | 0 | unvalidated (synthetic only) | — | — | — |
| otp_or_pin_request | 0 | unvalidated (synthetic only) | — | — | — |
| parcel_courier_scam | 0 | unvalidated (synthetic only) | — | — | — |
| utility_disconnection_scam | 0 | unvalidated (synthetic only) | — | — | — |

Headline tag-F1 uses **0** validated tag(s): []

### `user_real_safe` - 5 rows **TOO SMALL FOR CLAIMS**

scam rate: 0.0% | style mix: {'roman_hinglish': 3, 'english': 2}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | n/a | 0.000 | n/a | 0.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF + LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.200 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF + LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.600 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF tags LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF tags LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| style_only (shortcut) | n/a | 0.000 | n/a | 0.000 | 0.400 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| source_only (shortcut) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |

Per-tag F1 (only tags with >= 20 real positives are shown):

| tag | real positives | status | keyword | tactics_lr_synth | tactics_lr_plus |
|---|---|---|---|---|---|
| authority_impersonation | 0 | unvalidated (synthetic only) | — | — | — |
| bank_kyc_impersonation | 0 | unvalidated (synthetic only) | — | — | — |
| investment_or_trading_scam | 0 | unvalidated (synthetic only) | — | — | — |
| job_or_task_scam | 0 | unvalidated (synthetic only) | — | — | — |
| loan_app_harassment | 0 | unvalidated (synthetic only) | — | — | — |
| otp_or_pin_request | 0 | unvalidated (synthetic only) | — | — | — |
| parcel_courier_scam | 0 | unvalidated (synthetic only) | — | — | — |
| payment_or_upi_request | 0 | unvalidated (synthetic only) | — | — | — |
| prize_or_lottery | 0 | unvalidated (synthetic only) | — | — | — |
| relative_in_trouble | 0 | unvalidated (synthetic only) | — | — | — |
| suspicious_link | 0 | unvalidated (synthetic only) | — | — | — |
| urgency_threat | 0 | unvalidated (synthetic only) | — | — | — |
| utility_disconnection_scam | 0 | unvalidated (synthetic only) | — | — | — |

Headline tag-F1 uses **0** validated tag(s): []

### `external_scam` - 3511 rows

scam rate: 100.0% | style mix: {'english': 3509, 'roman_hinglish': 2}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | n/a | 1.000 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | n/a | 0.045 | n/a | 0.023 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.162 | 0.183 |
| TF-IDF + LR (synth only) | n/a | 0.771 | n/a | 0.627 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF + LR (synth+real) | n/a | 0.989 | n/a | 0.979 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF tags LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.173 | 0.079 |
| TF-IDF tags LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | 0.349 | 0.119 |
| style_only (shortcut) | n/a | 1.000 | n/a | 0.999 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| source_only (shortcut) | n/a | 1.000 | n/a | 1.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |

Per-tag F1 (only tags with >= 20 real positives are shown):

| tag | real positives | status | keyword | tactics_lr_synth | tactics_lr_plus |
|---|---|---|---|---|---|
| suspicious_link | 3511 | validated | 0.271 | 0.000 | 0.000 |
| authority_impersonation | 2634 | validated | 0.033 | 0.000 | 0.000 |
| urgency_threat | 2479 | validated | 0.148 | 0.556 | 0.835 |
| bank_kyc_impersonation | 1340 | validated | 0.087 | 0.000 | 0.000 |
| parcel_courier_scam | 402 | validated | 0.356 | 0.000 | 0.000 |
| utility_disconnection_scam | 336 | validated | 0.011 | 0.000 | 0.000 |
| relative_in_trouble | 30 | validated | 0.378 | 0.000 | 0.000 |
| investment_or_trading_scam | 0 | unvalidated (synthetic only) | — | — | — |
| job_or_task_scam | 0 | unvalidated (synthetic only) | — | — | — |
| loan_app_harassment | 0 | unvalidated (synthetic only) | — | — | — |
| otp_or_pin_request | 0 | unvalidated (synthetic only) | — | — | — |
| payment_or_upi_request | 0 | unvalidated (synthetic only) | — | — | — |
| prize_or_lottery | 0 | unvalidated (synthetic only) | — | — | — |

Headline tag-F1 uses **7** validated tag(s): ['urgency_threat', 'bank_kyc_impersonation', 'authority_impersonation', 'suspicious_link', 'parcel_courier_scam', 'utility_disconnection_scam', 'relative_in_trouble']

### `public_ham_english` - 345 rows

scam rate: 0.0% | style mix: {'english': 345}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | n/a | 0.000 | n/a | 0.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| keyword rules | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF + LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.070 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF + LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.009 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| TF-IDF tags LR (synth only) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| TF-IDF tags LR (synth+real) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| style_only (shortcut) | n/a | 0.000 | n/a | 0.000 | 1.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |
| source_only (shortcut) | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a (too few negatives) | n/a (too few negatives) | n/a | n/a |

Per-tag F1 (only tags with >= 20 real positives are shown):

| tag | real positives | status | keyword | tactics_lr_synth | tactics_lr_plus |
|---|---|---|---|---|---|
| authority_impersonation | 0 | unvalidated (synthetic only) | — | — | — |
| bank_kyc_impersonation | 0 | unvalidated (synthetic only) | — | — | — |
| investment_or_trading_scam | 0 | unvalidated (synthetic only) | — | — | — |
| job_or_task_scam | 0 | unvalidated (synthetic only) | — | — | — |
| loan_app_harassment | 0 | unvalidated (synthetic only) | — | — | — |
| otp_or_pin_request | 0 | unvalidated (synthetic only) | — | — | — |
| parcel_courier_scam | 0 | unvalidated (synthetic only) | — | — | — |
| payment_or_upi_request | 0 | unvalidated (synthetic only) | — | — | — |
| prize_or_lottery | 0 | unvalidated (synthetic only) | — | — | — |
| relative_in_trouble | 0 | unvalidated (synthetic only) | — | — | — |
| suspicious_link | 0 | unvalidated (synthetic only) | — | — | — |
| urgency_threat | 0 | unvalidated (synthetic only) | — | — | — |
| utility_disconnection_scam | 0 | unvalidated (synthetic only) | — | — | — |

Headline tag-F1 uses **0** validated tag(s): []

### `real_english_combined` - 3856 rows

> **English real data, optimistic: sources differ in era and style, NOT a Hinglish claim.**

scam rate: 91.1% | style mix: {'english': 3854, 'roman_hinglish': 2}

| model | macro-F1 | scam F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | AUROC | source-origin AUROC | PR-AUC | R@1%FPR | R@5%FPR | tag micro | tag macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.477 | 0.953 | [0.474, 0.479] | 1.000 | 1.000 | n/a | n/a | 0.997 | n/a | n/a | n/a | n/a | n/a |
| keyword rules | 0.175 | 0.045 | [0.097, 0.115] | 0.023 | 0.000 | n/a | n/a | 0.997 | n/a | n/a | n/a | 0.162 | 0.183 |
| TF-IDF + LR (synth only) | 0.546 | 0.767 | [0.529, 0.562] | 0.627 | 0.070 | n/a | 0.863 | 0.997 | 0.985 | n/a (too few negatives) | 0.602 | n/a | n/a |
| TF-IDF + LR (synth+real) | 0.945 | 0.989 | [0.932, 0.957] | 0.979 | 0.009 | n/a | 0.998 | 0.997 | 1.000 | n/a (too few negatives) | 0.991 | n/a | n/a |
| TF-IDF tags LR (synth only) | 0.082 | 0.000 | [0.074, 0.089] | 0.000 | 0.000 | n/a | n/a | 0.997 | n/a | n/a | n/a | 0.173 | 0.079 |
| TF-IDF tags LR (synth+real) | 0.082 | 0.000 | [0.074, 0.089] | 0.000 | 0.000 | n/a | n/a | 0.997 | n/a | n/a | n/a | 0.349 | 0.119 |
| style_only (shortcut) | 0.476 | 0.953 | [0.474, 0.479] | 0.999 | 1.000 | n/a | 0.500 | 0.997 | 0.910 | n/a (too few negatives) | 0.000 | n/a | n/a |
| source_only (shortcut) | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | n/a | 1.000 | 0.997 | 1.000 | n/a (too few negatives) | 1.000 | n/a | n/a |

> **AUROC close to source-origin AUROC means the score may reflect dataset recognition, not scam detection.** The source-origin classifier (TEXT only: IMC25 scam vs UCI ham) reaches AUROC **0.997** on the same real test rows (accuracy 0.968, n=3,856). Compare it with each row's AUROC above.


Per-tag F1 (only tags with >= 20 real positives are shown):

| tag | real positives | status | keyword | tactics_lr_synth | tactics_lr_plus |
|---|---|---|---|---|---|
| suspicious_link | 3511 | validated | 0.271 | 0.000 | 0.000 |
| authority_impersonation | 2634 | validated | 0.033 | 0.000 | 0.000 |
| urgency_threat | 2479 | validated | 0.148 | 0.555 | 0.834 |
| bank_kyc_impersonation | 1340 | validated | 0.087 | 0.000 | 0.000 |
| parcel_courier_scam | 402 | validated | 0.356 | 0.000 | 0.000 |
| utility_disconnection_scam | 336 | validated | 0.011 | 0.000 | 0.000 |
| relative_in_trouble | 30 | validated | 0.378 | 0.000 | 0.000 |
| investment_or_trading_scam | 0 | unvalidated (synthetic only) | — | — | — |
| job_or_task_scam | 0 | unvalidated (synthetic only) | — | — | — |
| loan_app_harassment | 0 | unvalidated (synthetic only) | — | — | — |
| otp_or_pin_request | 0 | unvalidated (synthetic only) | — | — | — |
| payment_or_upi_request | 0 | unvalidated (synthetic only) | — | — | — |
| prize_or_lottery | 0 | unvalidated (synthetic only) | — | — | — |

Headline tag-F1 uses **7** validated tag(s): ['urgency_threat', 'bank_kyc_impersonation', 'authority_impersonation', 'suspicious_link', 'parcel_courier_scam', 'utility_disconnection_scam', 'relative_in_trouble']

## Gap check (per model)

`test_b` uses unseen synthetic families. The real reference is `real_english_combined` - English real data, optimistic: sources differ in era and style, NOT a Hinglish claim. A large drop means the model learned the generator, not scams.

| model | test_a macro | test_b macro | a-b | test_b AUROC | real macro | b-real | real AUROC | AUROC gap |
|---|---|---|---|---|---|---|---|---|
| majority | 0.349 | 0.380 | -0.031 | n/a | 0.477 | -0.096 | n/a | n/a |
| keyword rules | 0.316 | 0.555 | -0.239 | n/a | 0.175 | 0.379 | n/a | n/a |
| TF-IDF + LR (synth only) | 1.000 | 1.000 | 0.000 | 1.000 | 0.546 | 0.454 | 0.863 | 0.137 |
| TF-IDF + LR (synth+real) | 0.998 | 0.995 | 0.003 | 1.000 | 0.945 | 0.051 | 0.998 | 0.002 |
| TF-IDF tags LR (synth only) | 0.317 | 0.279 | 0.038 | n/a | 0.082 | 0.196 | n/a | n/a |
| TF-IDF tags LR (synth+real) | 0.317 | 0.279 | 0.038 | n/a | 0.082 | 0.196 | n/a | n/a |
| style_only (shortcut) | 0.311 | 0.494 | -0.183 | 0.400 | 0.476 | 0.017 | 0.500 | -0.100 |
| source_only (shortcut) | 0.317 | 0.279 | 0.038 | 0.500 | 1.000 | -0.721 | 1.000 | -0.500 |
> The `real AUROC` column is on `real_english_combined`. The source-origin classifier (TEXT only) reaches AUROC **0.997** on the same rows. **AUROC close to source-origin AUROC means the score may reflect dataset recognition, not scam detection.**

`test_a` is a 10% hold-out of TRAIN clusters (Phase 3.6), so both classes exist and the seen-wording score is real. `test_b` uses UNSEEN families, so `a - b` is the memorisation gap and `b - real` is the domain gap; the AUROC gap is the same question for ranking quality.

> Synthetic results are not performance claims. Headline numbers come only from test_real, and always PER STRATUM.

---

# ScamLens - shortcut audit (Phase 3.5, BINARY)

## 1. Style / label balance audit (BINARY)

A style or source is FLAGGED when its scam rate is >90% or <10%. Flagged does not always mean broken: on `test_real`, source IS the stratum (external scams vs public ham), so a 100% rate there is structural, not a model shortcut.

### `train` (15873 rows)

**train by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| devanagari | 364 | 206 | 56.6% |  |
| english | 13027 | 9065 | 69.6% |  |
| mixed | 33 | 33 | 100.0% | FLAGGED (100% scam) |
| roman_hinglish | 2449 | 824 | 33.6% |  |

**train by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| external_reported | 8390 | 8390 | 100.0% | FLAGGED (100% scam) |
| public_ham_eval | 1610 | 0 | 0.0% | FLAGGED (0% scam) |
| synthetic | 5873 | 1738 | 29.6% |  |

### `val` (4540 rows)

**val by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| devanagari | 119 | 31 | 26.1% |  |
| english | 4122 | 3614 | 87.7% |  |
| mixed | 22 | 6 | 27.3% |  |
| roman_hinglish | 277 | 104 | 37.5% |  |

**val by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| external_reported | 3511 | 3511 | 100.0% | FLAGGED (100% scam) |
| public_ham_eval | 345 | 0 | 0.0% | FLAGGED (0% scam) |
| synthetic | 684 | 244 | 35.7% |  |

### `test_a` (667 rows)

**test_a by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| devanagari | 46 | 46 | 100.0% | FLAGGED (100% scam) |
| english | 412 | 159 | 38.6% |  |
| mixed | 4 | 4 | 100.0% | (n<10: not flagged) |
| roman_hinglish | 205 | 149 | 72.7% |  |

**test_a by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| synthetic | 667 | 358 | 53.7% |  |

### `test_b` (215 rows)

**test_b by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| devanagari | 29 | 13 | 44.8% |  |
| english | 80 | 64 | 80.0% |  |
| mixed | 31 | 4 | 12.9% |  |
| roman_hinglish | 75 | 51 | 68.0% |  |

**test_b by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| synthetic | 215 | 132 | 61.4% |  |

### `test_real` per stratum

#### `user_real_scam` (5 rows)

**user_real_scam by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| english | 2 | 2 | 100.0% | (n<10: not flagged) |
| roman_hinglish | 3 | 3 | 100.0% | (n<10: not flagged) |

**user_real_scam by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| self_collected | 5 | 5 | 100.0% | (n<10: not flagged) |

#### `user_real_safe` (5 rows)

**user_real_safe by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| english | 2 | 0 | 0.0% | (n<10: not flagged) |
| roman_hinglish | 3 | 0 | 0.0% | (n<10: not flagged) |

**user_real_safe by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| self_collected | 5 | 0 | 0.0% | (n<10: not flagged) |

#### `external_scam` (3511 rows)

**external_scam by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| english | 3509 | 3509 | 100.0% | FLAGGED (100% scam) |
| roman_hinglish | 2 | 2 | 100.0% | (n<10: not flagged) |

**external_scam by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| external_reported | 3511 | 3511 | 100.0% | FLAGGED (100% scam) |

#### `public_ham_english` (345 rows)

**public_ham_english by script_style**

| script_style | rows | scam | scam rate | flag |
|---|---|---|---|---|
| english | 345 | 0 | 0.0% | FLAGGED (0% scam) |

**public_ham_english by source**

| source | rows | scam | scam rate | flag |
|---|---|---|---|---|
| public_ham_eval | 345 | 0 | 0.0% | FLAGGED (0% scam) |

**FLAGGED style/source groups:**

* train: script_style=mixed (33 rows, 100% scam)
* train: source=external_reported (8390 rows, 100% scam)
* train: source=public_ham_eval (1610 rows, 0% scam)
* val: source=external_reported (3511 rows, 100% scam)
* val: source=public_ham_eval (345 rows, 0% scam)
* test_a: script_style=devanagari (46 rows, 100% scam)
* external_scam: script_style=english (3509 rows, 100% scam)
* external_scam: source=external_reported (3511 rows, 100% scam)
* public_ham_english: script_style=english (345 rows, 0% scam)
* public_ham_english: source=public_ham_eval (345 rows, 0% scam)

## 2. Metadata-only shortcut baselines

These models see ONE column and no message text. High scores mean the data leaks the label through metadata.

| split | rows | style_only macro-F1 | source_only macro-F1 |
|---|---|---|---|
| val | 4540 | 0.679 | 0.916 |
| test_a | 667 | 0.311 | 0.317 |
| test_b | 215 | 0.494 | 0.279 |
| test_real/user_real_scam | 5 | 0.286 | 0.000 |
| test_real/user_real_safe | 5 | 0.375 | 0.500 |
| test_real/external_scam | 3511 | 0.500 | 0.500 |
| test_real/public_ham_english | 345 | 0.000 | 0.500 |

If style_only or source_only is close to the text baselines on a SYNTHETIC split, the generator still has a metadata shortcut. On test_real, source is the stratum definition, so a high source-only score there is expected and not a model win.

**FLAGGED: style_only = 0.679 on combined val (> 0.65).**
  Fix direction: add synthetic Hinglish SAFE and synthetic English SCAM rows so every script_style has both classes.

**FLAGGED: source_only = 0.916 on combined val (> 0.65).**
  This is structural: `source` literally names the dataset (IMC25 scam vs UCI ham), so it cannot be rebalanced by adding synthetic rows without fabricating real data. Fix requires a mixed-origin real dataset.

## 2b. Source-origin classifier (TEXT only)

Predicts *which public dataset* a real message came from (IMC25 scam vs UCI ham). High accuracy means the real mix has a corpus/era/style confound that a model can exploit.

| rows | accuracy | AUROC |
|---|---|---|
| val (3856) | 0.960 | 0.997 |
| test_real (3856) | 0.968 | 0.997 |

If accuracy is above 0.65, the real-mix style/source shortcut is real. It cannot be removed by relabeling synthetic rows as real; it needs a mixed-origin real dataset.

## 3. Single-feature shortcut check (macro-F1 on test_b)

* **msg_length**: macro-F1 = 0.380
* **has_url**: macro-F1 = 0.380
* **has_digits**: macro-F1 = 0.380
* **has_devanagari**: macro-F1 = 0.380
* **has_bank_word**: macro-F1 = 0.380
* **n_exclamations**: macro-F1 = 0.380
* **caps_ratio**: macro-F1 = 0.548

No single feature cleared the threshold - good sign.

## 4. Top 20 word features of verdict_lr (scam class)

words that push TOWARDS scam:
  url, date_time, your, to, named_entity, आपक, link, pay, you have, click, an, package, via, we, customer, karein, here, phone_number, whatsapp, fee

words that push AWAY from scam:
  no, nahi, needed, of rs, app, in the, official app, ok, hai, come, in, fyi, pm, da, your records, कल, gt, no action, debited, awb

## 5. Top 15 most-confident WRONG predictions on test_b

* conf=0.60 predicted **safe**, gold **scam**
    [text withheld - 34 words, 180 chars]

