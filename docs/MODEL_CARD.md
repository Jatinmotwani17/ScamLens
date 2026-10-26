# ScamLens model card

**Version:** final phase (TF-IDF + Logistic Regression, calibrated, three operating
points). **Status: research prototype — NOT production-ready.** The strict mode still
flags 6.4% of real English ham (n=345) and 5.5% of held-out safe-chat probes (n=200);
those are real false alarms, and a false "scam" label on a bank alert is a harmful
outcome, not a rounding error.

---

## 1. Intended use

ScamLens reads **one short message** (SMS / WhatsApp / DM text, 1–1000 characters) and
returns a verdict (`scam`, `suspicious`, `safe`), a calibrated-ish scam probability,
named tactic signals with validation status, rule-based reasons, highlighted words, and
warnings. It is a **verification aid for a human reader** — a second opinion before you
click a link or share an OTP.

It is designed for:

- showing how a scam-detection pipeline is *evaluated* honestly (leakage audits,
  source-shortcut detection, per-stratum metrics with confidence intervals);
- a demo / teaching artefact for an interview or portfolio;
- research on synthetic-data augmentation and false-alarm tradeoffs.

## 2. Out-of-scope use (must not)

- **Do not block, ban, report, or auto-action any sender, account, or message.** The
  model must never be the sole basis for a decision about a person.
- Do not use as a filter that silently drops legitimate bank, OTP, courier, or
  government messages.
- Do not present it to end users as "AI that detects scams" without the prototype
  banner and the limitations below.
- Do not use outside English/roman-Hinglish SMS-like text (no email bodies, no long
  documents, no images/audio).
- Do not treat `scam_probability` as a calibrated confidence. The scorer is bimodal
  (see §7), which is why the `suspicious` band is rule-based.
- Do not redistribute the raw datasets (see `DATA_LICENSES.md`).

## 3. Training data provenance

| source | rows (train) | share | label origin | license / status |
|---|---:|---:|---|---|
| `external_reported` (IMC25 scam corpus) | 8,390 | 52.9% | dataset's own labels, `label_confidence=medium` | research use; **UNVERIFIED redistribution** — see `DATA_LICENSES.md` |
| `public_ham_eval` (UCI SMS Spam Collection, ham only) | 1,610 | 10.1% | dataset label (ham) | CC BY 4.0 (attribution required) |

(The UCI ham reserve is capped at 2,400 rows in `configs/config.yaml`; 2,300 survive
dedup, of which 1,610 are in train and 345 are the `test_real` ham stratum. UCI ships
~4,800 ham rows, so this is a deliberate cap, not everything licensed.)
| `synthetic` (template generator + `safe_v2` hard negatives) | 5,873 | 37.0% | generator, `labeler=synthetic_template` | our own generation, no external rights |
| **total train** | **15,873** | 100% | | |

Evaluation strata are separate and never trained on: `val` 4,540 rows, `test_a` 667,
`test_b` 215 (synthetic), `test_real` 3,866 (real: 3,511 scams + 345 English ham), plus
the 200 **frozen** safe-chat probes (`data/processed/safe_chat_probe_frozen.parquet`)
and the 783-row human-`suspicious` holdout.

**No synthetic row appears in any real test set.** Enforced by an assertion in
`splits35.py` (`HARD RULE VIOLATED` on any synthetic row marked `test_real`), not by
convention.

Real data is ~63% of training and 100% of the real evaluation. Synthetic rows exist to
cover tactic families and hard negatives that real data barely contains — they are
never used to claim real-world accuracy.

## 4. Measured performance (per stratum, never blended)

Model: TF-IDF (word 1–2 gram, 22,249 features) + Logistic Regression, temperature
T = 0.4614 fitted on weighted val, operating points fitted on weighted val only.

| stratum | metric | value | n |
|---|---|---:|---:|
| val (weighted cells) | macro-F1 | 0.964 | 4,540 |
| val | balanced accuracy | 0.986 | 4,540 |
| test_b (synthetic) | macro-F1 | 0.995 | 215 |
| test_real — external_scam | recall (strict) | 0.992 (0.988–0.994) | 3,511 |
| test_real — public_ham_english | FPR (strict) | 0.064 (0.042–0.095) | 345 |
| safe-chat probes (frozen) | FPR (strict) | 0.055 (0.031–0.096) | 200 |
| source-origin classifier | AUROC (real vs synthetic) | 0.996 | — |
| model | AUROC on val | 0.998 | 4,540 |

### The three operating points (measured, not targets)

| mode | threshold | public_ham FPR (95% CI) | probe FPR (95% CI) | external scam recall (95% CI) |
|---|---:|---|---|---|
| strict | 0.040 | 0.064 (0.042–0.095) n=345 | 0.055 (0.031–0.096) n=200 | 0.992 (0.988–0.994) n=3511 |
| balanced | 0.016 | 0.093 (0.066–0.128) n=345 | 0.080 (0.050–0.126) n=200 | 0.995 (0.992–0.997) n=3511 |
| sensitive | 0.331 | 0.012 (0.005–0.029) n=345 | 0.005 (0.001–0.028) n=200 | 0.982 (0.977–0.986) n=3511 |

Confusion counts and the cost tradeoff: `results/PHASE4_REPORT.md` (operating-point
section) and `configs/operating_point.yaml` (fitted values, cell counts, measurement).

**False-alarm improvement from Part 1:** the earlier model flagged **15.3% of real
English ham (n=59)** and **13.0% of safe-chat probes (English 20%, Hinglish 6%)**. The
current model, at strictly *higher* scam recall, flags 6.4% (n=345) and 5.5% (n=200).
The comparison is not like-for-like on n: the ham set is ~6× larger because the project
previously reserved only 400 UCI ham rows and now reserves 2,400 (2,300 after dedup,
345 of them in `test_real`). UCI ships ~4,800 ham rows, so the reserve is still a
fraction of what is licensed — a known cap, not a data limit.

## 5. The `suspicious` verdict

`abstention_method: rules`. TF-IDF scores are bimodal (almost nothing between 0.01 and
0.9), so a probability band is empty and is **not** faked. Instead `suspicious` =
"model clears the strict threshold but I can name no evidence": no tactic tag above
0.50, no OTP/PIN/payment/credential wording, no risky link type, no keyword-rule hit.

Measured cost (`results/suspicious_band.md`): on clear held-out scams, **40/3,511
(1.1%)** are deferred instead of called scam; on clear real ham, 16/345 (4.6%) are
escalated to "check this"; on the 783-row human-unsure holdout, **355 (45.3%)** land in
suspicious (vs 0% under a probability band). It helps, at the price of a small recall
loss on already-detected scams.

## 6. Robustness probes — probes, not real-world claims

Every robustness number comes from a scripted transformation, not from an adversary in
the wild (`results/robustness.md`). Highlights: worst-case l33t/space-injection edits
move recall by ≈0.1 pt; Hinglish translation-derived probes retain 0.987 of English
recall; safe-chat probes are the sensitive spot (5.5% FPR). Treat these as *sensitivity
checks on a fixed decision boundary*, not as field performance.

## 7. Calibration & known failure modes

- Temperature scaling: val ECE 0.030 → **0.005** on the reweighted val. This does not
  make the score smooth: it is still bimodal, and the API warns when the calibrated
  probability is low but the threshold is cleared.
- **False alarms on formal legitimate messages** are the dominant error: real bank
  debit alerts, OTP-delivery notices, and official-looking notices share vocabulary
  (account, KYC, link, urgent) with scams. This is the failure mode Part 1 attacked; it
  improved but did not disappear.
- **The real ham set barely contains formal notices.** Of 345 `public_ham_english`
  rows, only **5** are explicitly bank/OTP/balance text, and **4 of those 5 are
  flagged** at the strict threshold. So "6.4% FPR" describes *conversational* ham much
  better than *transactional* ham. Hand-written bank alerts ("Rs 2,000 debited from
  A/c XX4821 … Not you? Call …") are flagged too. Treat the FPR as a lower bound for
  formal, official-looking messages.
- **English-only real validation.** All real evaluation rows are English. Hinglish
  evidence is synthetic-only (generator + translation-derived probes). Any Hinglish
  accuracy claim would be fabricated.
- **Source-origin shortcut.** A classifier can tell real from synthetic text with AUROC
  0.996. Metrics computed on mixed real/synthetic sets can therefore be inflated; the
  evaluation reports real and synthetic strata separately and includes a `source_only`
  baseline beside every model number.
- **Six unvalidated tactic tags** (<20 real positives): `payment_or_upi_request`,
  `prize_or_lottery`, `otp_or_pin_request`, `investment_or_trading_scam`,
  `job_or_task_scam`, `loan_app_harassment`. They are shown greyed with
  `validated=false`; their precision is unknown.
- **Dataset shift.** IMC25 scams are 2023-era; language and link tactics move fast. The
  probes are frozen by design, so the reported FPR is a fixed reference point that will
  drift out of date.

## 8. Ethical notes

- No message text is logged or stored by the API — only length, verdict, latency and
  mode. A test enforces this.
- Training text is third-party reported scam messages and public ham; it may contain
  real phone numbers/URLs. The scrubber replaces them before modelling, and the raw
  bundle is not redistributed.
- The system must not be used to surveil, profile, or accuse individuals. A `scam`
  verdict means "this text looks like the scam patterns I learned", nothing about the
  sender's intent or identity.
- Reporting guidance in the UI points to official channels (cybercrime.gov.in / 1930)
  rather than encouraging vigilantism or retaliation.

## 9. Reproduce

```
python -m src.build_dataset      # rebuild splits (leakage guards run here)
python -m src.baselines          # TF-IDF + LR + tag heads
python -m src.calibrate          # temperature + three operating points
python -m src.suspicious         # suspicious-band evaluation
python -m src.evaluate           # per-stratum metrics + CIs
python -m src.phase4             # calibration/abstention/robustness report
python scripts/ci_gate.py        # regression gate (fails on F1/FPR drift)
```

All seeds come from `configs/config.yaml`; every random split, model fit and probe
generation is seeded.
