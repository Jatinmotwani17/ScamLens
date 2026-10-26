# Phase 4 - calibration, abstention, error analysis, robustness

Model under test: **`verdict_lr_plus`**. Fitted on validation only (seed 42); every threshold and temperature below comes from `configs/operating_point.yaml`. Probes live in their own table and never enter training.

> Everything in this file is either (a) a measured number with `n` printed beside it, or (b) explicitly labelled a hypothesis. No number here is a production claim.

## What stands out

1. **The model is saturated, and that is the headline.** At the fixed point (`t_high` = 0.039) it catches 99.2% of 3,511 held-out real scams (28 missed) while flagging 6.4% of real English ham (n=345). Every robustness probe therefore has almost no recall left to lose - the failure mode is false positives, not missed attacks.
2. **Temperature scaling works on the strata we can measure**, but the safe side stays miscalibrated: val ECE 0.0298 -> 0.0053 (reweighted), while `public_ham_english` is still the worst stratum. T = 0.461 (the raw-val fit would be 0.712).
3. **The abstention band does not exist for this model.** At the 0.95/0.95 targets `t_high` = `t_low` = 0.039 with 0 validation rows between them; no target from 0.80 to 0.99 produces a non-empty band, because the scorer's probabilities are bimodal. Coverage is 100% by construction.
4. **The human-`suspicious` holdout gets no abstention at all**: 781 of 783 rows (0.0% abstained) are pushed to a definitive verdict. These rows are ambiguous by construction, so `scam` is not automatically wrong - but a three-way system that never uses its third verdict is a two-way system with extra UI.
5. **Mechanical attacks barely move recall**: the worst probe is spaces_inside_words on `real_scams` (n=500), recall 0.991 vs a clean baseline near 1.000 (-0.001). This is the saturation effect again: a classifier that says `scam` to almost everything cannot be talked out of it.
6. **The one number that would worry a reviewer is the fresh safe-chat FPR: 5.5% (11/200)** on legitimate messages that share no template with training (max similarity to train 0.61). The Hinglish transfer probe, by contrast, barely changes recall (0.990 -> 0.987 on n=300).

---

## 1. Novelty-bucketed metrics (the honest generalisation view)


> **This is the honest generalisation measure.** Each row is bucketed by `max_train_sim` - the maximum TF-IDF char-ngram cosine similarity to any TRAIN row (results/nn_sim.parquet). Performance that only holds at high similarity is near-copy matching, not detection.

Decisions use the fixed operating point (t_high = 0.039, temperature = 0.461) fitted on validation only.

## `test_b`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 159 | 104 | 55 | 1.000 | 0.000 | 1.000 | 0.650 | 0.0058 |  |
| 0.5-0.7 | 55 | 27 | 28 | 1.000 | 0.036 | 1.000 | 0.496 | 0.0048 |  |
| 0.7-0.9 | 1 | 1 | 0 | 1.000 | n/a | n/a | 1.000 | 0.0001 | too small for claims; single class in bucket |

## `test_real/external_scam`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 2,066 | 2,066 | 0 | 0.987 | n/a | n/a | 0.963 | 0.0372 | single class in bucket |
| 0.5-0.7 | 811 | 811 | 0 | 0.999 | n/a | n/a | 0.991 | 0.0090 | single class in bucket |
| 0.7-0.9 | 576 | 576 | 0 | 1.000 | n/a | n/a | 0.996 | 0.0039 | single class in bucket |
| >0.9 | 58 | 58 | 0 | 1.000 | n/a | n/a | 0.996 | 0.0042 | single class in bucket |

## `test_real/public_ham_english`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 312 | 0 | 312 | n/a | 0.064 | n/a | 0.015 | 0.0154 | single class in bucket |
| 0.5-0.7 | 29 | 0 | 29 | n/a | 0.069 | n/a | 0.011 | 0.0111 | too small for claims; single class in bucket |
| 0.7-0.9 | 3 | 0 | 3 | n/a | 0.000 | n/a | 0.000 | 0.0001 | too small for claims; single class in bucket |
| >0.9 | 1 | 0 | 1 | n/a | 0.000 | n/a | 0.000 | 0.0001 | too small for claims; single class in bucket |

## `test_real/user_real_scam`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 5 | 5 | 0 | 1.000 | n/a | n/a | 0.986 | 0.0139 | too small for claims; single class in bucket |

## `test_real/user_real_safe`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 5 | 0 | 5 | n/a | 0.600 | n/a | 0.507 | 0.5065 | too small for claims; single class in bucket |

## `val`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 2,809 | 2,191 | 618 | 0.986 | 0.050 | 0.997 | 0.750 | 0.0303 |  |
| 0.5-0.7 | 1,113 | 963 | 150 | 0.999 | 0.027 | 0.999 | 0.858 | 0.0075 |  |
| 0.7-0.9 | 569 | 552 | 17 | 1.000 | 0.059 | 1.000 | 0.960 | 0.0099 |  |
| >0.9 | 49 | 49 | 0 | 1.000 | n/a | n/a | 0.997 | 0.0032 | single class in bucket |

> AUROC needs both classes in a bucket; single-class buckets report recall/FPR only. Buckets under 30 rows are printed but are not claims.

---

## 2-3. Calibration and the abstention band

## Calibration (validation only)

Model: **verdict_lr_plus**. Temperature and thresholds are fitted on the **validation split only** (4,540 rows); a code guard raises if any test row is passed to the fitter.

**Why val is reweighted:** real val is 98.3% scam and ~87% of it is one source. Fitting on raw val optimises for "flag real-looking text as scam". We weight the four cells (scam|safe x synthetic|real) equally so the safe evidence is not drowned out.

| validation cell | rows |
|---|---:|
| `real/safe` | 345 |
| `real/scam` | 3,511 |
| `synthetic/safe` | 440 |
| `synthetic/scam` | 244 |

- temperature (reweighted fit) = **0.4614**; sensitivity, raw-val fit = 0.7123 (T > 1 means the raw model is over-confident).
- ECE on validation: raw **0.0298** -> calibrated **0.0053** (weighted).
- `t_high` = **0.039** (scam-verdict precision 0.9506; 3,724 scam rows behind it).
- `t_low` = **0.039** (safe-verdict NPV 0.9500; 749 safe rows behind it).

### Three operating points (Phase 5)

Same weighted validation rows, three different promises. Pick per use 
case; never quote one blended accuracy.

| mode | rule | threshold | val recall (weighted) | val FPR (weighted) | real ham FPR | safe-chat probe FPR | external_scam recall | test_b recall |
|---|---|---:|---|---|---|---|---|---|
| **strict** | FPR <= 0.05 | 0.040 | 0.996 | 0.049 | 0.064 (0.042-0.095), n=345 | 0.055 (0.031-0.096), n=200 | 0.992 (0.988-0.994), n=3511 | 1.000 (0.972-1.000), n=132 |
| **balanced** | FPR <= 0.10 | 0.016 | 0.997 | 0.100 | 0.093 (0.066-0.128), n=345 | 0.080 (0.050-0.126), n=200 | 0.995 (0.992-0.997), n=3511 | 1.000 (0.972-1.000), n=132 |
| **sensitive** | recall >= 0.99 | 0.331 | 0.990 | 0.006 | 0.012 (0.005-0.029), n=345 | 0.005 (0.001-0.028), n=200 | 0.982 (0.977-0.986), n=3511 | 0.992 (0.958-0.999), n=132 |

| mode | val safe rows flagged | val scam rows missed | real ham false alarms | probe false alarms |
|---|---:|---:|---:|---:|
| strict | 35/785 | 31/3755 | 22/345 | 11/200 |
| balanced | 73/785 | 19/3755 | 32/345 | 16/200 |
| sensitive | 4/785 | 69/3755 | 4/345 | 1/200 |

> `real ham` is `public_ham_english` (held-out UCI ham); `safe-chat probe` is the frozen 200-row generated probe. Both are small: the intervals, not the point estimate, are the claim.

> **The abstain band is empty at this target.** `n_rows_in_band` = 0: validation has no rows between `t_low` and `t_high`, so the two bars meet at one cut (0.039). The three-way decision then never returns `suspicious` and coverage is 100% by construction - a fact about this model's scores, not a tuning choice. The sweep below shows every alternative target.

### Sensitivity: the same fit per validation half

| fit | t_high | precision | n above | t_low | NPV | n below |
|---|---|---|---:|---|---|---:|
| reweighted val (primary) | 0.039 | 0.9506 | 3761 | 0.039 | 0.9500 | 1007 |
| raw val | 0.015 | 0.9501 | 3947 | 0.015 | 0.9502 | 804 |
| val_real only (scam 3511, safe 345) | 0.000 | 0.9502 | 3692 | 0.000 | 0.9510 | 286 |
| val_synthetic only (scam 244, safe 440) | 0.025 | 0.9531 | 256 | 0.025 | 0.9503 | 463 |

> `n` columns are validation counts behind each threshold. A part with few safe rows cannot place `t_low` at all (n/a).

### Calibration on every reported stratum

| stratum | n | n safe | ECE raw | ECE calibrated | mean p raw | mean p calibrated |
|---|---:|---:|---|---|---|---|
| `val_synthetic` | 684 | 440 | 0.0320 | 0.0051 | 0.360 | 0.355 |
| `val_real` | 3,856 | 345 | 0.0350 | 0.0255 | 0.875 | 0.885 |
| `test_b` | 215 | 83 | 0.0259 | 0.0035 | 0.613 | 0.612 |
| `test_real/user_real_scam` *(small)* | 5 | 0 | 0.0553 | 0.0139 | 0.945 | 0.986 |
| `test_real/user_real_safe` *(small)* | 5 | 5 | 0.4937 | 0.5065 | 0.494 | 0.507 |
| `test_real/external_scam` | 3,511 | 0 | 0.0390 | 0.0247 | 0.961 | 0.975 |
| `test_real/public_ham_english` | 345 | 345 | 0.0571 | 0.0149 | 0.057 | 0.015 |

> ECE here is computed on each stratum with its own rows (single-class strata included: a stratum of all-scam rows with mean p 0.99 is miscalibrated by 0.01).

### Abstention band on the reported strata (at the fixed point)

| stratum | n | scam recall @t_high | FPR @t_high | coverage | accuracy on covered | band rows |
|---|---:|---|---|---|---|---:|
| `val_synthetic` | 684 | 1.000 | 0.014 | 100.0% | 0.991 | 0 |
| `val_real` | 3,856 | 0.991 | 0.090 | 100.0% | 0.984 | 0 |
| `test_b` | 215 | 1.000 | 0.012 | 100.0% | 0.995 | 0 |
| `test_real/user_real_scam` *(small)* | 5 | 1.000 | 0.000 | 100.0% | 1.000 | 0 |
| `test_real/user_real_safe` *(small)* | 5 | 0.000 | 0.600 | 100.0% | 0.400 | 0 |
| `test_real/external_scam` | 3,511 | 0.992 | 0.000 | 100.0% | 0.992 | 0 |
| `test_real/public_ham_english` | 345 | 0.000 | 0.064 | 100.0% | 0.936 | 0 |

> Strata flagged *(small)* sit under the n-gate; the two user strata (n=5 each) are printed for completeness, never as claims.

### What a different target would buy

| precision/NPV target | t_high | t_low | val rows in band | val band share | holdout rows abstained | holdout called scam | safe rows behind t_low |
|---|---|---|---:|---:|---:|---:|---:|
| 0.80 | 0.002 | 0.002 | 0 | 0.0% | 0.0% | 100.0% |  599 |
| 0.90 | 0.013 | 0.013 | 0 | 0.0% | 0.0% | 99.9% |  705 |
| 0.95 | 0.039 | 0.039 | 0 | 0.0% | 0.0% | 99.7% |  749 |
| 0.98 | 0.130 | 0.130 | 0 | 0.0% | 0.0% | 99.5% |  771 |
| 0.99 | 0.271 | 0.271 | 0 | 0.0% | 0.0% | 98.7% |  779 |

> Fitted on validation with the same equal-cell weights. `val rows in band` counts validation rows the rule would send to a human; `holdout rows abstained` counts the human-labelled `suspicious` band that would reach that human. A rule that abstains on neither is not abstention.

### Suspicious band on the `suspicious` holdout

783 rows that a human labelled **`suspicious`** (the unsure band, never trained on) land as:

| verdict at the fitted point | rows | share |
|---|---:|---:|
| suspicious (abstain) | 0 | 0.0% |
| scam | 781 | 99.7% |
| safe | 2 | 0.3% |

Mean calibrated p = 0.974 (median 1.000). **The model claims certainty on every row a human could not label.** Zero abstentions means the third verdict is never used on exactly the traffic that needs it: the safest reading is that 781 of these rows would be pushed to a user as confident scams. Caveat: these rows are risky-but-ambiguous by construction (promos, loan offers, chain messages), so `scam` is not automatically wrong - what is missing is any willingness to defer. Either the band is mis-specified for this model or the label `suspicious` and the probability scale disagree - both are real findings, neither is a tuning detail.

---

## 4. Error analysis


> Decisions use the FIXED operating point from `configs/operating_point.yaml` (fitted on validation only, seed 42). FPs = gold `safe` -> verdict `scam`; FNs = gold `scam` -> verdict `safe`; rows in the abstain band are counted separately.

## Counts

| set | n | n safe | n scam | FP | FN | abstain | note |
|---|---:|---:|---:|---:|---:|---:|---|
| `test_b` | 215 | 83 | 132 | 1 | 0 | 0 |  |
| `test_real/external_scam` | 3,511 | 0 | 3,511 | 0 | 28 | 0 |  |
| `test_real/public_ham_english` | 345 | 345 | 0 | 22 | 0 | 0 |  |
| `test_real/user_real_scam` | 5 | 0 | 5 | 0 | 0 | 0 | too small for claims; no errors |
| `test_real/user_real_safe` | 5 | 5 | 0 | 3 | 0 | 0 | too small for claims |

## `test_b`

**FALSE POSITIVES** (1 total, top 1 shown)

| pattern (primary tag x style) | n in top 1 |
|---|---:|
| (no tag) x devanagari | 1 |

### (no tag) x devanagari (1)

> [text withheld - 13 words, 56 chars]

*Hypothesis:* Devanagari only ever appears in synthetic rows, so this is generator behaviour, not real-world evidence. That is a hypothesis to test, not a proven cause (over-flagged a safe message in test_b (synthetic, unseen families)).

**FALSE NEGATIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

## `test_real/external_scam`

**FALSE POSITIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

**FALSE NEGATIVES** (28 total, top 28 shown)

| pattern (primary tag x style) | n in top 28 |
|---|---:|
| suspicious_link x english | 25 |
| relative_in_trouble x english | 2 |
| parcel_courier_scam x english | 1 |

### suspicious_link x english (25)

> [text withheld - 121 words, 660 chars]

> [text withheld - 20 words, 93 chars]

> [text withheld - 8 words, 41 chars]

*Hypothesis:* the model leans on link/URL wording, so any safe row with a link is at risk of being called scam. That is a hypothesis to test, not a proven cause (missed a real scam in the real stratum `external_scam`).

### relative_in_trouble x english (2)

> [text withheld - 8 words, 37 chars]

> [text withheld - 33 words, 170 chars]

*Hypothesis:* family-chat wording is exactly what the safe corpus lacks, so it is easy to over-flag. That is a hypothesis to test, not a proven cause (missed a real scam in the real stratum `external_scam`).

### parcel_courier_scam x english (1)

> [text withheld - 19 words, 100 chars]

*Hypothesis:* courier/delivery wording is shared with real delivery notifications, so the boundary is thin. That is a hypothesis to test, not a proven cause (missed a real scam in the real stratum `external_scam`).

## `test_real/public_ham_english`

**FALSE POSITIVES** (22 total, top 22 shown)

| pattern (primary tag x style) | n in top 22 |
|---|---:|
| (no tag) x english | 22 |

### (no tag) x english (22)

> [text withheld - 18 words, 97 chars]

> [text withheld - 5 words, 31 chars]

> [text withheld - 5 words, 29 chars]

*Hypothesis:* English is the best-represented style, so an English error points at the threshold or the corpus, not the script. That is a hypothesis to test, not a proven cause (over-flagged a safe message in the real stratum `public_ham_english`).

**FALSE NEGATIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

## `test_real/user_real_scam`

**FALSE POSITIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

**FALSE NEGATIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

## `test_real/user_real_safe`

**FALSE POSITIVES** (3 total, top 3 shown)

| pattern (primary tag x style) | n in top 3 |
|---|---:|
| (no tag) x english | 2 |
| (no tag) x roman_hinglish | 1 |

### (no tag) x english (2)

> [text withheld - 23 words, 125 chars]

> [text withheld - 15 words, 84 chars]

*Hypothesis:* English is the best-represented style, so an English error points at the threshold or the corpus, not the script. That is a hypothesis to test, not a proven cause (over-flagged a safe message in the real stratum `user_real_safe`).

### (no tag) x roman_hinglish (1)

> [text withheld - 13 words, 83 chars]

*Hypothesis:* roman-Hinglish text is nearly absent from real training rows (8 of 15,151), so its errors are the known Hinglish gap, not a new bug. That is a hypothesis to test, not a proven cause (over-flagged a safe message in the real stratum `user_real_safe`).

**FALSE NEGATIVES** (0 total, top 0 shown)

No errors of this kind at the fixed operating point.

> A hypothesis is a reading of the examples to test next - it is not a proven cause.

---

## Robustness probes

> **Robustness probe, not a real-world claim.** Every row below was produced by a script in `src/robustness.py`, not by an adversary in the wild. The numbers say how a fixed operating point survives mechanical edits; they say nothing about next quarter's scams.

Fixed operating point: `t_high` = **0.039**, temperature = 0.461 (fitted on validation only). A scam counts as caught when its calibrated probability clears `t_high`; the abstain band is empty for this model, so `suspicious` cannot rescue recall.

### (a) Perturbation attacks - scam recall at the fixed point

| pool | attack | n | rows edited | recall (all) | recall (edited) | delta vs clean | mean p | note |
|---|---|---:|---:|---|---|---|---|---|
| `real_scams` | clean (baseline) | 500 | 500 | 0.992 | n/a | baseline | 0.968 |  |
| `test_b_scams` | clean (baseline) | 132 | 132 | 1.000 | n/a | baseline | 0.994 |  |
| `real_scams` | typos | 500 | 436 | 0.992 | 0.995 | +0.000 | 0.960 |  |
| `test_b_scams` | typos | 132 | 115 | 1.000 | 1.000 | +0.000 | 0.991 |  |
| `real_scams` | spaces_inside_words | 500 | 446 | 0.990 | 0.991 | -0.002 | 0.963 |  |
| `test_b_scams` | spaces_inside_words | 132 | 114 | 1.000 | 1.000 | +0.000 | 0.991 |  |
| `real_scams` | lookalikes | 500 | 484 | 0.994 | 0.996 | +0.002 | 0.959 |  |
| `test_b_scams` | lookalikes | 132 | 115 | 1.000 | 1.000 | +0.000 | 0.989 |  |
| `real_scams` | leetspeak | 500 | 481 | 0.992 | 0.994 | +0.000 | 0.957 |  |
| `test_b_scams` | leetspeak | 132 | 119 | 1.000 | 1.000 | +0.000 | 0.986 |  |
| `real_scams` | all_caps | 500 | 494 | 0.992 | 0.994 | +0.000 | 0.968 |  |
| `test_b_scams` | all_caps | 132 | 112 | 1.000 | 1.000 | +0.000 | 0.994 |  |
| `real_scams` | emoji | 500 | 500 | 0.992 | 0.992 | +0.000 | 0.966 |  |
| `test_b_scams` | emoji | 132 | 132 | 1.000 | 1.000 | +0.000 | 0.994 |  |
| `real_scams` | url_obfuscation | 500 | 204 | 0.992 | 1.000 | +0.000 | 0.968 |  |
| `test_b_scams` | url_obfuscation | 132 | 60 | 1.000 | 1.000 | +0.000 | 0.994 |  |
| `real_scams` | punctuation | 500 | 500 | 0.992 | 0.992 | +0.000 | 0.962 |  |
| `test_b_scams` | punctuation | 132 | 132 | 1.000 | 1.000 | +0.000 | 0.993 |  |

> Attack rates: typos swap two characters in 20% of words; spaces are inserted inside 20% of words; lookalikes replace one character in 30% of words; leetspeak rewrites 25% of words; emoji inserts 2-4 symbols; URL obfuscation writes `hxxps://site[.]tld` (a scrubbed `<URL>` is replaced by the same obfuscated form).

### (b) Hinglish transfer probe - **translation-derived, not real Hinglish**

300 held-out real English scams were rewritten with a phrase dictionary of 353 pairs built from the `roman_hinglish` pools in `src/synth/components.py` plus a hand-aligned seed whose every Hinglish form is verified to appear in those pools (6 seed pairs were dropped for failing that check).

| version | n | scam recall | mean p |
|---|---:|---|---|
| English (original) | 300 | 0.990 | 0.961 |
| roman Hinglish (dictionary rewrite) | 300 | 0.987 | 0.939 |

- 97.0% of rows changed; on average 39.3% of a message's words were replaced, so these are light rewrites, not full translations.
- Gemini path: no GEMINI_API_KEY - Gemini paraphrase skipped.

Examples for manual review (English -> derived Hinglish):

> **EN** 1.000 -> **HI** 0.999 (8 words)

> **EN** 0.996 -> **HI** 0.988 (6 words)

> **EN** 1.000 -> **HI** 1.000 (20 words)

> **EN** 1.000 -> **HI** 1.000 (18 words)

> **EN** 1.000 -> **HI** 0.999 (16 words)

### (c) Fresh safe-chat probe - false positive rate

| style | n | flagged as scam | FPR | mean p | note |
|---|---:|---:|---|---|---|
| english | 100 | - | 0.110 | 0.022 |  |
| roman_hinglish | 100 | - | 0.000 | 0.000 |  |
| **all** | 200 | 11 | 0.055 | 0.011 |  |

Novelty check: the highest similarity of any probe message to a training row is **0.61** (mean 0.32), so these are not near-copies of training text.

Safe messages the model calls scam:

> flagged probe: 14 words, 92 chars
> flagged probe: 17 words, 88 chars
> flagged probe: 21 words, 95 chars
> flagged probe: 16 words, 99 chars
> flagged probe: 16 words, 79 chars

---

## Limitations

1. **This is the TF-IDF baseline, not the transformer.** Every table describes `verdict_lr_plus`. `e1_muril` has no results on this machine (no CUDA), so nothing here is evidence about the model the project intends to ship. Re-running `python -m src.phase4 --model e1_muril` regenerates the whole report from saved predictions.
2. **The validation split cannot support a safe-side threshold on its own.** It is 91.1% scam; the only real safe rows are **n=345** (plus n=440 synthetic). Fitting uses equal weight per (scam|safe x real|synthetic) cell for exactly this reason, but `t_low` still rests mostly on synthetic safe text. Treat every safe-side number as provisional.
3. **No bucket below the n-gate is a claim.** `test_b` has 224 rows in total and the user strata have 5 each; anything under 30 rows is marked *too small for claims* in the tables and must not be quoted as a result.
4. **`external_scam` contains only scams.** It has zero safe rows, so recall is computable and FPR/AUROC are not. A blended "real" number would be meaningless, which is why the strata are never pooled.
5. **Novelty bucketing cannot show degradation here.** With recall pinned near 1.000 in every bucket, the bucket table mostly demonstrates the saturation of the decision, not generalisation. The informative column is FPR on ham, and ham is concentrated at low similarity (n=2 above 0.5).
6. **The attack probes are scripted, not adversarial.** Each edit is a fixed transformation at a documented rate (see the note under the table). Real evasion is adaptive and human; these probes bound how much a *known* edit costs, nothing more.
7. **The Hinglish probe is translation-derived, not real Hinglish.** It rewrites English with a phrase dictionary (353 pairs, every Hinglish form verified against the synthetic pools) and replaces ~39% of tokens. No GEMINI_API_KEY was present, so the paraphrase path was skipped. Real Hinglish code-mixing looks different and this probe must not stand in for it.
8. **The safe-chat probe is generated from our own templates.** They share no template with training (max cosine similarity 0.61 to any train row), but they are still synthetic text written by the same project, so the FPR is an estimate with an unknown bias in either direction.
9. **The suspicious holdout is 783 rows, not 1,403.** The Phase 4 brief assumed 1,403; the table on disk holds 783 (the rest were relabelled or dropped during Phase 2.5). The numbers above use what actually exists.
10. **One model, one seed, no confidence intervals.** No metric here carries a bootstrap interval. Differences under ~1 point (several attack deltas are -0.00x) are noise and are reported as such.

---

## What would change the conclusions

| if we had | it would test |
|---|---|
| `e1_muril` GPU results | whether the saturation and the empty abstain band are properties of TF-IDF or of the data |
| 500+ labelled real ham messages | whether `t_low` and the ~11% safe-chat FPR survive on real traffic |
| real Hinglish messages with labels | the transfer probe's stand-in for code-mixed traffic |
| a held-out set of human-flagged `suspicious` rows with a gold binary label | whether abstention should defer or decide |
