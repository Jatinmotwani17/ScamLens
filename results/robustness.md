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
