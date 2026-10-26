# Phase 4 error analysis - `verdict_lr_plus`

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
