# Master prompt — write the ScamLens research paper

> Copy everything between the lines into the agent. The attached reference paper is used
> for **structure only** — no sentence, value, or citation may be taken from it.

---

## ROLE

You are a researcher writing an IEEE conference paper (two-column, 5–6 pages, ~3,000–3,500
words, 20–28 references) about a completed project called **ScamLens**. The code, data
pipeline, reports and documenration all exist in this repository. Your job is to turn that
evidence into a publishable, honest paper.

## NON-NEGOTIABLE RULES

1. **Every number must come from the ledger in §3.** Do not estimate, round up,
   extrapolate, or invent. If a sentence needs a number that is not in the ledger, write
   `[[MISSING: what is needed]]` and list it in your final message.
2. **Never edit a number to make the story better.** If a result is weak, write the weak
   result and explain it. The paper's contribution is evaluation rigour, not a high score.
3. **No claims about things that were not run.** In particular there are **no transformer
   (MuRIL) results** in this project. Do not fabricate a comparison table. Say explicitly
   that the transformer path is implemented but untrained, and that TF-IDF is shipped
   because the comparison could not be made.
4. **No blended accuracy.** Never write "97% accurate". Report per stratum with `n` and
   95% confidence intervals (Wilson). Real and synthetic results stay separate.
5. **No real-Hinglish accuracy claim.** All real evaluation data is English. Hinglish
   evidence is synthetic + translation-derived only.
6. **Do not claim production readiness**, deployment, novelty of the algorithm, or state of
   the art. This is a systems + evaluation paper.
7. **Do not copy phrasing from the reference PDF.** Use only its section order, table
   idioms, and the "research questions + gap in literature + observations on design"
   pattern.
8. **Citations must be real and verifiable.** Every reference needs authors, venue, year,
   and (where applicable) DOI/arXiv id that you are confident exists. If unsure, drop it.
   Never invent a citation to support a number.
9. **Privacy:** no message text in the paper beyond the 3 short demo strings given in §4.
   Never include dataset rows, phone numbers, UPI IDs or real names.
10. Write in plain, specific academic English. No hype adjectives ("revolutionary",
    "seamless", "state-of-the-art"). Prefer numbers and verbs.

## 1. WHAT THE PAPER ARGUES

**Thesis:** for short Indian scam SMS, the hard problem is not catching scams — a simple
lexical model catches ~99% — it is (a) the false-alarm cost on legitimate bank/OTP-style
messages, and (b) trusting your own evaluation when the dataset leaks its source. We
present a leakage-audited pipeline, quantify a dataset-source shortcut (AUROC 0.996),
reduce false alarms by adding real ham and a disjoint hard-negative generator, and replace
a single threshold with three validation-fitted operating points plus a rule-based
abstention band with a measured cost.

**Three research questions (mirror this pattern in the Introduction and results):**

- **RQ1 — Detection:** can a lightweight TF-IDF + logistic-regression model, trained on
  real reported scams plus controlled synthetic augmentation, detect real scam messages
  at high recall?
- **RQ2 — False alarms:** what is the measured false-alarm rate on real legitimate
  messages and on frozen safe-chat probes, and does a disjoint hard-negative generator
  plus more licensed real ham reduce it without sacrificing recall?
- **RQ3 — Trust:** is the evaluation itself trustworthy — i.e. can a model distinguish
  real from synthetic text, how should operating points be fitted and reported, and what
  does a "suspicious" band cost?

## 2. REQUIRED STRUCTURE (IEEE conference style)

Use exactly this skeleton. Word budgets assume 5–6 pages.

| § | Section | Words | Content requirements |
|---|---|---:|---|
| — | Title | 12–18 | "ScamLens: A Leakage-Audited Pipeline and Operating-Point Analysis for Short-Message Scam Detection" (or better, same information). No colon-chain longer than this. |
| — | Authors | 2 lines | Placeholder names + emails. No fake affiliations. |
| — | Abstract | 140–170 | Problem, method (one sentence), the three headline findings with numbers (recall, ham FPR, probe FPR, source AUROC), and the honest scope sentence ("English-only real validation; not production-ready"). Self-contained, no citations. |
| — | Keywords | 6–9 | scam SMS detection, SMS spam, TF-IDF, logistic regression, calibration, operating points, abstention, dataset leakage, evaluation methodology |
| I | Introduction | 550–650 | Indian scam-SMS context; why legitimate and fraudulent messages share vocabulary; the false-alarm cost argument; bullets of what ScamLens does; **RQ1–RQ3 verbatim as questions**; a "paper is organised as follows" paragraph. Cite 4–6 works. |
| II | Related Work | 550–650 | Subsections: (A) SMS spam/scam detection (classical + neural); (B) synthetic data and augmentation for low-resource text; (C) calibration and selective prediction / abstention; (D) dataset artefacts, shortcuts and leakage in NLP evaluation; (E) **Gap in the literature** — no work reports per-stratum false alarms with CIs on licensed Indian ham, nor quantifies a real-vs-synthetic source shortcut in scam detection. End with one sentence positioning this paper. |
| III | Data and Methodology | 900–1,100 | (A) Data sources + provenance table (Table I); (B) scrubbing and PII handling; (C) deduplication and cluster/family-grouped splits, with the hard assertion (no synthetic row in any real test set); (D) the model: TF-IDF (word 1–2 gram, 22,249 features) + LR verdict + 13 tactic heads + keyword rules; (E) calibration (temperature) and three operating points fitted on weighted validation; (F) the rule-based suspicious band and why a probability band is empty; (G) evaluation protocol: Wilson intervals, source-only baselines, frozen probes. Include **Fig. 1 = architecture diagram** described in §5. |
| IV | Results | 1,000–1,200 | (A) RQ1 detection: Table II per-stratum (val, test_b, real scam recall); (B) RQ2 false alarms: Table III before/after + three modes with CIs; (C) RQ3 trust: source-shortcut AUROC, novelty buckets, robustness probes, calibration ECE; (D) abstention cost: Table IV; (E) **Observations on design** — 4–6 bullets of engineering judgement (e.g. why weight validation cells equally, why the band is a rule, why ham had to be real). |
| V | Discussion & Limitations | 300–400 | Error analysis (formal bank/OTP notices fail; 5-of-345 caveat), English-only real validation, 6 unvalidated tags, single-dataset provenance, frozen probes age, no transformer comparison. Ethical/dual-use paragraph: must not auto-block or report anyone. |
| VI | Conclusion & Future Work | 250–300 | Restate 3 findings with numbers; 4 numbered future items (real bank/OTP traffic, MuRIL fp32 vs int8 benchmark with a pre-registered ship rule, real Hinglish labels, human-in-the-loop review queue). |
| — | References | 20–28 | IEEE numeric style. See §6. |
| — | Reproducibility statement | 60–80 | Repo layout, seeds, commands, test count (176 tests), CI gate thresholds, licence split (MIT code / datasets not redistributed). |

## 3. NUMBERS LEDGER (the only permitted source of numbers)

Do not change these. If a number is needed that is absent, emit `[[MISSING: ...]]`.

**Corpus (Table I material)**

- Total rows: **25,161**
- Real reported scams (IMC25-derived): **15,412**
- Real legitimate (UCI SMS Spam Collection, ham only): **2,300** (reserve capped at 2,400, deduped from ~4,800 licensed ham rows)
- Synthetic training rows: **~7,400** (template generator + 2,500 hard negatives)
- Train: **15,873** rows, of which **37.0%** synthetic, **63.0%** real
- Val: **4,540** · test_a: **667** · test_b (synthetic only): **215**
- test_real: **3,866** = **3,511 real scams** + **345 real English ham**
- Frozen safe-chat probes: **200** (100 English / 100 roman-Hinglish; never trained on; max similarity to any training row **0.61**)
- Human-labelled ambiguous holdout: **783**
- Real test-set sources: 0 synthetic rows (enforced by assertion)

**Model**

- TF-IDF word 1–2 gram, **22,249 features**; logistic regression verdict; **13** tactic tag heads
- Shipped artefact **1.0 MB**, **22,250** parameters
- Temperature scaling **T = 0.4614**, fitted on validation only
- AUROC on val **0.998**
- Calibration ECE (reweighted val): **0.0298 → 0.0053**

**Detection / false alarm (Wilson 95% CI, `n` always printed)**

| mode | threshold | real scam recall | real ham FPR | frozen probe FPR |
|---|---:|---|---|---|
| strict | 0.0397 | **0.992** (0.988–0.994), n=3,511 | **0.064** (0.042–0.095), n=345 | **0.055** (0.031–0.096), n=200 |
| balanced | 0.0157 | 0.995 (0.992–0.997), n=3,511 | 0.093 (0.066–0.128), n=345 | 0.080 (0.050–0.126), n=200 |
| sensitive | 0.3313 | 0.982 (0.977–0.986), n=3,511 | 0.012 (0.005–0.029), n=345 | 0.005 (0.001–0.028), n=200 |

- Val macro-F1 **0.964**, val balanced accuracy **0.986**
- test_b (synthetic) macro-F1 **0.995** — always labelled synthetic, never a real claim
- Weighted-val cells used for fitting: real/scam **3,511**; synthetic/safe **440**; real/safe **345**; synthetic/scam **244**

**Before/after (RQ2 story)**

- Previous model: real-ham false alarm **15.3% (n=59)**; safe-chat probe FPR **13.0% (26/200)** (English **20%**, Hinglish **6%**)
- After: strict **6.4% (n=345)** and **5.5% (n=200)** at **unchanged or higher** recall
- Cause 1: ham reserve raised from 400 → 2,400 licensed rows
- Cause 2: 2,500 hard negatives; **62.5%** contain scam-adjacent vocabulary; exact-duplicate rate 0%; max similarity to probes < 0.7

**RQ3 trust evidence**

- Source-only classifier (real vs synthetic, no scam label): **AUROC 0.996**
- Style-only baseline: report as the table row labelled `style_only` (do not invent a value; use the value in `results/RESULTS.md`)
- Novelty buckets: recall stays ≥0.987 in the 0.0–0.5 similarity bucket (real scams **n=2,066**); accuracy does NOT collapse with novelty
- Robustness probes: worst perturbation changes recall by **−0.001** (space injection, n=500); Hinglish translation-derived probe recall **0.987** vs 0.990 English (n=300)
- Abstention (rule `rules`): **355/783 = 45.3%** of human-ambiguous rows deferred; costs **40/3,511 = 1.1%** of clear scams; **16/345 = 4.6%** of clear ham escalated to "check this"

**Serving**

- CPU single-message latency: **p50 1.80 ms, p95 4.03 ms** (TF-IDF); end-to-end `/analyze` with 40-word occlusion **~123–160 ms** on this laptop
- Docker images: serving-only **814 MB**; full research image **2.97 GB**
- **176** tests; CI gate: test_b macro-F1 ≥ 0.90 and balanced-mode ham FPR ≤ 0.15 (measured 0.9951 and 0.0928)

**Known weaknesses that MUST appear in Discussion**

- Of 345 real ham rows, only **5** are explicit bank/OTP text and **4 of those are flagged** → 6.4% describes conversational ham, not transactional ham
- Real validation is **English only**
- **6 of 13** tactic tags have <20 real positives and are marked unvalidated: `payment_or_upi_request`, `prize_or_lottery`, `otp_or_pin_request`, `investment_or_trading_scam`, `job_or_task_scam`, `loan_app_harassment`
- Probabilities are **bimodal**; the calibrated score is not a confidence statement
- Scam data is 2023-era; probes are frozen and will age
- **No transformer results exist**

## 4. THE ONLY TEXT ALLOWED TO APPEAR

Use at most these three demo strings (and label them as illustrative, not dataset rows):

1. `Dear customer, your KYC is expiring today. Account will be blocked. Update immediately: <URL>`
2. `Rs 2,000 debited from A/c XX4821 on 26-10-26. Not you? Call 1800xxxxxxx.`
3. `bhai kal ka match dekhne ja rahe ho? main 7 baje aa jaunga`

## 5. FIGURES AND TABLES

Produce LaTeX/TikZ or vector-friendly descriptions for:

- **Fig. 1** Architecture: raw sources → scrub → dedup/cluster → family-grouped splits (with the no-synthetic-in-real-test assertion) → TF-IDF + LR + tag heads → calibration + 3 operating points → FastAPI (+ suspicious rule) → UI. Boxes, arrows, one legend.
- **Fig. 2** Reliability diagram: raw vs temperature-scaled on reweighted val; annotate ECE 0.0298 → 0.0053.
- **Fig. 3** Recall vs real-ham FPR for the three operating points, with Wilson CI bars (n=3,511 / n=345). This is the key figure — make it readable in one column.
- **Fig. 4** Performance by nearest-training-neighbour similarity bucket (0–0.5, 0.5–0.7, 0.7–0.9, >0.9) with `n` per bucket.
- **Table I** Data provenance: source, role, rows, label origin, licence status (IMC25 = research use, not redistributed; UCI = CC BY 4.0).
- **Table II** RQ1 per-stratum metrics with `n` and CIs (add `source_only` and `style_only` columns beside the model).
- **Table III** RQ2 before/after + three modes.
- **Table IV** Abstention outcomes on clear scams / clear ham / ambiguous holdout.
- **Table V** Serving: parameters, disk, p50/p95, plus the two image sizes.

Every table caption must state the test set and `n`. Never a table of a single blended number.

## 6. CITATIONS TO USE (verify each before including)

Use real works only; suggested anchors, in IEEE numeric order:

1. SMS Spam Collection (Almeida et al., 2011) — the ham source.
2. A recent Indian/regional scam-SMS or smishing dataset paper — **you must verify the exact
   IMC25 citation from the repository notes; if it cannot be verified, cite it as an
   anonymous released corpus and say so.**
3. TF-IDF (Salton & Buckley) and its use for SMS spam.
4. Logistic regression for text classification (or a standard IR/ML reference).
5. Vaswani et al., "Attention is all you need" (only as background for the unimplemented
   transformer path).
6. Conneau et al., XLM-R and/or Khanuja et al., MuRIL (background; explicitly not evaluated here).
7. Guo et al., "On Calibration of Modern Neural Networks" (temperature scaling).
8. Wilson (1927) score interval.
9. Broder, MinHash / similarity estimation (dedup).
10. Kaufman et al., "Leakage in Data Mining" (leakage framing).
11. Geirhos et al., "Shortcut learning in deep neural networks" (source shortcut framing).
12. A selective prediction / abstention reference (e.g. Geifman & El-Yaniv).
13. A dataset-shift or annotation-artefact reference.
14. One or two Hinglish/code-mixed NLP references.

That is 14 anchors; add 6–12 more only if you can verify them. Target 20–28 total.

## 7. STYLE AND FORMAT

- IEEE conference template, 10 pt, two columns, 5–6 pages including references.
- Past tense for what was done, present tense for what the results show.
- No first-person plural unless describing the system ("we ship", "we measure").
- Define every acronym at first use (SMS, FPR, CI, ECE, AUROC, TF-IDF, OTP, UPI, KYC).
- Numbers: two significant decimals for rates, three for thresholds, always with `n`.
- Every claim of improvement needs: old value, new value, CI, `n`, and the mechanism.
- Sentences ≤ 30 words. No bullet lists in the abstract.

## 8. DELIVERABLES

1. `paper.tex` (IEEEtran, compile-clean) or `paper.md` with IEEE section numbering if LaTeX
   is unavailable — state which.
2. `refs.bib` with only the verified references.
3. Figure sources (`fig1_architecture.tex`/`.py`, `fig2_reliability.py`,
   `fig3_operating_points.py`, `fig4_novelty.py`) or a clear spec if plotting is out of scope.
4. `numbers_ledger.md` — every number used, with the file it came from.
5. `claims_audit.md` — table of every quantitative claim in the paper → ledger row → PASS/FAIL.
6. A short `TODO.md` listing every `[[MISSING: ...]]` you emitted.

## 9. SELF-CHECK BEFORE YOU RETURN

Run these checks and report the result of each:

- [ ] Every number in the abstract appears in the ledger, unchanged.
- [ ] No sentence claims real Hinglish performance.
- [ ] No sentence claims production readiness, deployment, or novelty of the method.
- [ ] No transformer result or comparison table exists anywhere.
- [ ] Every rate has `n` and a 95% CI on first mention.
- [ ] Real and synthetic rows are never merged into one accuracy.
- [ ] The 5-of-345 bank-notice caveat appears in Discussion.
- [ ] The 6 unvalidated tags are named and marked unvalidated.
- [ ] No message text beyond the three demo strings.
- [ ] Every citation has authors + venue + year, and none is invented.
- [ ] Page count is 5–6 when compiled; abstract ≤ 170 words.
- [ ] The paper reads as an evaluation/measurement contribution, not a product pitch.

## 10. TONE FOR THE FINAL MESSAGE

Return: the paper, the ledger, the claims audit, the TODO list, and a 5-line summary of what
a reviewer will most likely attack and where the paper already answers it. Do not pad the
summary with praise for the work.
