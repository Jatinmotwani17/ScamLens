# ScamLens — Phase 2 report

> **Synthetic results are not performance claims. Headline numbers come only from `test_real`.**

## 0. What exists now

| piece | file | state |
|---|---|---|
| Lego components, 108 template families, 13 tactics | `src/synth/components.py` | done (every tactic has >= 4 families, asserted) |
| Synthetic generator (2,500 scam / 800 suspicious / 2,500 safe) | `src/synth/generate.py` | done, 40% noise, 1,250 hard negatives |
| Optional Gemini paraphrase pass | `src/synth/llm_paraphrase.py` | done, skips cleanly without `GEMINI_API_KEY` |
| Real reported smishing (IMC25, CC BY 4.0) | `src/external_data.py` | done — 31,685 real scams + 400 real ham |
| Family-aware 5-way split | `src/splits2.py` | done, leakage check PASS |
| Four baselines | `src/baselines.py` + `configs/keywords.yaml` | done, models in `models/` |
| Honest harness + bootstrap CIs | `src/evaluate.py` | done → `results/metrics.json`, `results/RESULTS.md` |
| Shortcut audit | `src/audit.py` | done (appended to RESULTS.md) |
| Notebook | `notebooks/02_baselines.ipynb` | done |
| Tests | `tests/` | 77 passing |

Dataset: **42,979 rows** → `train` 8,601 / `val` 385 / `test_a` 417 / `test_b` 1,491 / `test_real` 32,085.

## 1. Exact commands, in order

```bash
cd scamlens
python -m venv venv
source venv/Scripts/activate            # Windows Git Bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v              # 77 passed

python -m src.synth.generate            # synthetic corpus (seed 42)
python -m src.synth.llm_paraphrase      # optional; prints "Skipping LLM paraphrasing" without a key
python -m src.external_data             # IMC25 real scams + real ham -> test_real
python -m src.build_dataset             # dedup + family-aware split + schema check
python -m src.data_checks               # leakage check + data report
python -m src.baselines                 # trains the 4 baselines (public_weak excluded)
python -m src.evaluate                  # results/metrics.json + results/RESULTS.md
python -m src.audit                     # appends the shortcut audit to RESULTS.md
python -m src.baselines --include_public_weak   # ablation only, never the headline
jupyter notebook notebooks/02_baselines.ipynb
```

## 2. Results — every baseline x every test set

macro-F1 with 95% bootstrap CI (1,000 resamples). `FPR(hard neg)` counts only the
deliberately scam-sounding legitimate messages.

### `val` (synthetic, seen families) — 385 rows

| model | macro-F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | tactics micro | tactics macro |
|---|---|---|---|---|---|---|---|
| majority | 0.178 | [0.162, 0.195] | 0.000 | 0.000 | 0.000 | — | — |
| keyword | 0.333 | [0.285, 0.379] | 0.142 | 0.000 | 0.000 | 0.379 | 0.285 |
| **verdict_lr** | **0.579** | [0.561, 0.596] | 0.974 | 0.014 | 0.000 | — | — |
| tactics_lr | 0.178 | [0.162, 0.195] | 0.000 | 0.000 | 0.000 | 0.245 | 0.066 |

### `test_a` (synthetic, seen families) — 417 rows

| model | macro-F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | tactics micro | tactics macro |
|---|---|---|---|---|---|---|---|
| majority | 0.080 | [0.064, 0.098] | 0.000 | 0.000 | 0.000 | — | — |
| keyword | 0.309 | [0.265, 0.352] | 0.087 | 0.000 | 0.000 | 0.454 | 0.298 |
| **verdict_lr** | **0.572** | [0.546, 0.595] | 0.929 | 0.000 | 0.000 | — | — |
| tactics_lr | 0.080 | [0.064, 0.098] | 0.000 | 0.000 | 0.000 | 0.083 | 0.025 |

### `test_b` (synthetic, UNSEEN families) — 1,491 rows

| model | macro-F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | tactics micro | tactics macro |
|---|---|---|---|---|---|---|---|
| majority | 0.208 | [0.199, 0.215] | 0.000 | 0.000 | 0.000 | — | — |
| keyword | 0.517 | [0.489, 0.547] | 0.232 | 0.000 | 0.000 | 0.372 | 0.366 |
| **verdict_lr** | **0.523** | [0.506, 0.539] | 1.000 | 0.329 | 0.248 | — | — |
| tactics_lr | 0.208 | [0.199, 0.215] | 0.000 | 0.000 | 0.000 | 0.145 | 0.047 |

### `test_real` (REAL messages — the only claim that counts) — 32,085 rows

| model | macro-F1 | 95% CI | scam recall | FPR(safe) | FPR(hard neg) | tactics micro | tactics macro |
|---|---|---|---|---|---|---|---|
| majority | 0.008 | [0.008, 0.009] | 0.000 | 0.000 | 0.000 | — | — |
| keyword | 0.046 | [0.044, 0.048] | 0.037 | 0.005 | 0.000 | 0.242 | 0.143 |
| **verdict_lr** | **0.316** | [0.313, 0.318] | 0.834 | **0.728** | — | — | — |
| tactics_lr | 0.008 | [0.008, 0.009] | 0.000 | 0.000 | 0.000 | 0.124 | 0.039 |

`verdict_lr` on `test_real` (rows = gold, cols = predicted safe/scam/suspicious):

```
                safe   scam   susp
gold_safe        106    295      4      <- 73% of real legit messages called scam
gold_scam       4541  26437    712      <- 83% scam recall, 14% escape as "safe"
gold_suspicious    0      1      0
```

Tactic per-tag F1 on `test_real`: only `relative_in_trouble` 0.597,
`suspicious_link` 0.427, `parcel_courier_scam` 0.374 and `urgency_threat` 0.231
move at all; the other nine tags score 0.000–0.028.

## 3. Shortcut audit

Single-feature logistic models, macro-F1 on `test_a`:

| feature | macro-F1 | verdict |
|---|---|---|
| has_digits | 0.360 | ok |
| msg_length | 0.224 | ok (was 0.643 — see below) |
| has_url | 0.202 | ok |
| has_devanagari | 0.186 | ok |
| caps_ratio | 0.114 | ok |
| has_bank_word | 0.080 | ok |
| n_exclamations | 0.079 | ok |

**Flagged and fixed: `msg_length` at 0.643.** Scams averaged 120 characters and
safe messages 58, so "long = scam" was half the signal. I changed the generator
rather than the audit: added a set of short one-line scam families (21–60 chars)
and long legitimate bank/promo/appointment families (up to 210 chars). After
the fix the ranges overlap (scam 21–211, safe 24–210) and `msg_length` fell to
0.224, below the 0.60 threshold. This is the only generator change the audit
forced.

Top scam-pushing words the model learned: *number, कर, rs, url, today, abhi,
team, hai, now, ka, loan, upi, dear, kyc, hello, regards, pay, grahak, customer,
refund* — mostly sensible, with `dear`/`hello`/`regards` and `team` being
template artefacts (every family shares one greeting/closing pool).

## 4. The gap

| test set | verdict_lr macro-F1 |
|---|---|
| `test_a` (seen families) | 0.572 |
| `test_b` (unseen families) | 0.523 |
| `test_real` (real messages) | 0.316 |

* `test_a − test_b` = **+0.049** → modest memorisation of wording. Holding out
  families works: the drop is small, so the model generalises across templates
  reasonably well.
* `test_b − test_real` = **+0.207** → the big one. Two thirds of the remaining
  score comes from the generator's own regularity, not from real scam language.

Plain English: this baseline can recognise *my* scams, and it can partly
recognise unseen *variants of my* scams. It cannot yet recognise real reported
Indian smishing without false-alarming on three quarters of real legitimate
messages. Phase 3 needs far more real data (the gate is 50 labeled real rows;
there are 11).

## 5. What looks unrealistic about the synthetic data

1. **Repetition.** 1,548 suspicious rows contain only 839 distinct texts, and
   67% of the wrong test_b predictions are exact duplicates of one sentence
   ("Loan chahiye? Rs 50,000 tak turant. Call <PHONE>"). Composition with few
   slots produces many identical rows; real scammers vary far more.
2. **`suspicious` is not a learnable class as written.** Its messages are
   composed from scam vocabulary with the threat removed, so every model calls
   them `scam` (recall ~0 on all three synthetic test sets). The boundary
   between "risky" and "fraudulent" needs genuinely different ingredients
   (forwarded chains, unknown-sender promos with no demand, "is this you?"
   pings), not defanged scams.
3. **Style balance is off for India.** Only 15% of scams are devanagari and
   45% roman-Hinglish, while the real pool is 95% English (IMC25 is already
   translated), so the model is barely tested on the script mix real users see.
4. **Uniform length and structure.** Almost every family is greeting + hook +
   demand + closing in one or two sentences. Real smishing often arrives as a
   single fragment with no greeting, or as a multi-message thread.
5. **Tag co-occurrence is too generous.** On `test_real`, `suspicious_link`
   fires on 31,688 of 32,085 rows and `authority_impersonation` on 25,762 —
   the IMC25 mapping is coarse, so the real tactic labels are near-constant and
   per-tag F1 there is close to meaningless.
6. **The "real ham" reserve is tiny and easy.** 400 real legitimate messages
   carry almost all of the safe-class evidence in `test_real`; a 0.728 FPR on
   400 rows is directionally right but statistically thin.
