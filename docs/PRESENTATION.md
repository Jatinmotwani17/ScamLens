# ScamLens — presentation script (simple language)

> How to use this: each slide has **WHAT'S ON SCREEN**, **SAY THIS** (word-for-word if you
> want), and **IF ASKED**. Total run time ≈ 8 minutes + demo + questions. Everything on
> the slides must come from this repo's results — never invent a number live.

---

## Before you start (5-minute checklist)

1. `make serve` → open http://127.0.0.1:8000
2. Have these three messages ready to paste (they are the demo):
   - **Scam:** `Dear customer, your KYC is expiring today. Account will be blocked. Update immediately: http://sbi-kyc-verify.xyz`
   - **Legit OTP:** `Your OTP for login is 482913. Never share it with anyone, including bank staff.`
   - **Casual Hinglish:** `bhai kal ka match dekhne ja rahe ho? main 7 baje aa jaunga`
3. Open `results/PHASE4_REPORT.md` in a second tab — for the confidence-interval tables.
4. Open `docs/CLAIMS.md` — this is your safety net for any question that asks you to
   over-claim.
5. Say out loud once: "It is a research prototype, not production." Then you will not
   accidentally say "97% accurate" and get destroyed.

---

## Slide 1 — Title

**ON SCREEN**
ScamLens — detecting scam SMS without crying wolf
Your name · course · date

**SAY THIS**
"Every week my family gets messages that look exactly like their bank. Some are real.
Some are scams. I built a detector for that — and then I spent most of the project
proving my own numbers were honest, because that turned out to be the hard part."

**IF ASKED "why this topic?"**
"Because the failure that matters isn't a missed scam, it's a false alarm. If the app
screams at a real bank message, people stop using it."

---

## Slide 2 — The problem in one picture

**ON SCREEN**
Two messages side by side:

- `Rs 2,000 debited from A/c XX4821 on 26-10-26. Not you? Call 1800xxxxxxx.` (real)
- `Dear customer, your KYC is expiring today. Account will be blocked.` (fake)

And one line: **they share almost every word.**

**SAY THIS**
"The words 'account', 'bank', numbers, urgency — all identical. A model that learns
'bank words mean scam' catches both. So the real task isn't catching scams, it's
separating scams from boring legitimate messages that use the same vocabulary."

**IF ASKED "how do you separate them?"**
"Order, phrasing and context: 'do not share this OTP' appears in real OTP messages;
'update your KYC at this link' appears in fake ones. The model learns word-pairs, not
just words. But it's not perfect — I'll show you exactly where it breaks."

---

## Slide 3 — What I built (architecture)

**ON SCREEN**
The ASCII diagram from the README, plus three boxes:

1. **Data pipeline** — scrub phone numbers/UPI/links, remove duplicates, cluster, split.
2. **Model** — text → numbers (TF-IDF) → weighted checklist (logistic regression) →
   probability + 13 tactic scores.
3. **Serving** — FastAPI with three modes, a UI, Docker, CI.

**SAY THIS**
"The model is deliberately simple: a word-and-word-pair checklist with learned weights.
22,000 features, one megabyte, two milliseconds. No GPU. I chose that so I could spend
my time on evaluation instead of babysitting training runs — and so every number is
explainable."

**IF ASKED "why not BERT/MuRIL?"**
"Fair question, and I have an honest answer: I never got the transformer comparison to
run — no GPU artifacts exist in the repo. The serving code has a MuRIL path and falls
back to TF-IDF, but I have no evidence MuRIL is better, so I don't claim it is."

---

## Slide 4 — Data, and the one rule that matters

**ON SCREEN**
Table: 25,161 messages total.
- 15,412 real reported scams (IMC25)
- 2,300 real legitimate (UCI SMS Spam Collection, ham only)
- ~7.4k synthetic training rows (mine)
- Test set: 3,511 real scams + 345 real ham + 200 frozen safe-chat probes

Then one line in red: **NO SYNTHETIC ROW MAY ENTER A REAL TEST SET.**

**SAY THIS**
"Two thirds of training is real. The synthetic part exists only to cover patterns real
data doesn't have — like legitimate bank alerts. And the rule 'no fake data in a real
test' isn't a promise in a slide: it's an assertion in the code that crashes the build."

**IF ASKED "why is synthetic data a problem?"**
"Because a model can learn your generator instead of learning scams. I measured exactly
that — next slide."

---

## Slide 5 — The finding I'm proudest of (source shortcut)

**ON SCREEN**
Big number: **AUROC 0.996**
Caption: "A classifier that never sees a scam label can tell real text from synthetic
text with AUROC 0.996."

**SAY THIS**
"I trained a model whose only job was 'real or synthetic?'. It scored 0.996. That means
my synthetic test scores were partly measuring 'did you memorise my template generator'.
So every table in this project now separates real and synthetic rows, and prints a
'source-only' baseline next to the model. Most student projects never check this — and
their 99% numbers are quietly meaningless."

**IF ASKED "does that invalidate your results?"**
"No — it invalidates *mixed* results. Real-only numbers are unaffected: I report real
scam recall and real ham false-alarm rate separately, with n and confidence intervals.
The synthetic F1 of 0.995 is labelled 'not a performance claim' everywhere."

---

## Slide 6 — The false-alarm story (your main result)

**ON SCREEN**
Table:

| | before | strict | balanced | sensitive |
|---|---|---|---|---|
| real ham false alarms | 15.3% (n=59) | **6.4%** (4.2–9.5) n=345 | 9.3% | 1.2% |
| frozen safe-chat probes | 13.0% (n=200) | **5.5%** (3.1–9.6) n=200 | 8.0% | 0.5% |
| real scam recall | ~99% | 99.2% n=3,511 | 99.5% | 98.2% |

**SAY THIS**
"First version: it caught 99% of scams and screamed at 15% of real messages. Useless.
Two fixes, both real data work, no tricks: I was only loading 400 of the licensed
legitimate messages — a bug, now 2,400. And I generated 2,500 *legitimate* messages that
deliberately contain scam words — 'KYC', 'OTP', 'link', 'urgent' — so the model has to
learn structure, not vocabulary. False alarms dropped to 6.4% and 5.5% with recall
unchanged."

**IF ASKED "6.4% is still high."**
"Agreed — and it's worse than it looks. Of those 345 real ham messages, only 5 are
actually bank or OTP notices, and 4 of those get flagged. My 6.4% mostly describes
casual chat, not formal bank messages. That's written in the README, the model card and
the claims file, because it's the biggest remaining weakness."

---

## Slide 7 — Three modes, because one threshold is a lie

**ON SCREEN**
Three cards: Strict / Balanced / Sensitive, each with the one-line cost.

**SAY THIS**
"There's no single correct threshold, so I ship three, all fitted on validation data
only. Strict: 6.4% false alarms, misses 0.8% of scams. Sensitive: 1.2% false alarms,
misses 1.8% of scams. The person who owns the user experience picks. What I refuse to do
is print one blended accuracy number and call it a day."

**IF ASKED "how did you choose the thresholds?"**
"Weighted validation — real/synthetic × scam/safe cells weighted equally, so a flood of
synthetic rows can't drown the real signal. Then measured on held-out real data with
Wilson intervals."

---

## Slide 8 — "Suspicious" and why I didn't fake it

**ON SCREEN**
Left: a probability histogram that is two spikes at 0 and 1.
Right: the rule, in plain words.
Bottom: 45.3% (355/783) of human-ambiguous messages deferred; 1.1% of clear scams deferred.

**SAY THIS**
"The obvious design is a 40–60% 'not sure' band. But my scores are bimodal — almost
nothing sits in the middle, so the band would be empty. I could have shipped it anyway
and it would look clever. Instead 'suspicious' means: 'the score says scam, but I can't
name a single reason'. It costs me 1.1% of clear scams that now go to a human, and it
correctly defers 45% of messages that even human labellers found ambiguous."

**IF ASKED "isn't that just a hack?"**
"It's a rule with a measured cost, which is better than a calibrated band that never
fires. If I had transformer probabilities I'd test whether they're smoother — that's
explicitly future work."

---

## Slide 9 — Live demo (the money slide)

**DO THIS, IN ORDER (90 seconds)**

1. Paste the **KYC scam**. Click Analyze.
   - Point: red `scam`, probability ~1.0, tactic chips all **validated**, reasons
     ("keyword rule hit: 'account will be blocked'"), link type, highlighted words.
   - Say: "The tags agree with the verdict — that's when I trust it most."
2. Paste the **legit OTP**. Click Analyze.
   - Point: green `safe`, but a warning appears, and one tag is greyed as
     **unvalidated**.
   - Say: "It's honest about what it doesn't know. And this one is borderline — switch
     to Balanced and it flips. That's a real weakness, not a UI bug."
3. Switch mode to **Balanced**, re-run the OTP.
   - Say: "Same message, different threshold. This is why the mode selector is visible
     instead of hidden."
4. Paste the **Hinglish** message.
   - Point: `safe`, and the warning "validated on English real messages only".
   - Say: "No real Hinglish validation exists in this project. That warning is how I
     avoid lying about it."

**IF THE SERVER DIES MID-DEMO**
Fall back to the screenshots in the README and the curl examples. Say: "the API is in
Docker too, one command" — and don't debug live.

---

## Slide 10 — Honest limits + what I'd do next

**ON SCREEN**
Four bullets, no spin:
- 6.4% false alarms on conversational ham; formal bank/OTP notices are worse.
- Real validation is English only. Hinglish = synthetic only.
- 6 of 13 tactic tags are unvalidated (<20 real examples).
- No transformer results; the MuRIL path is code, not evidence.

**SAY THIS**
"This is a prototype. It must never auto-block or report anyone. If I continued, I'd do
three things in this order: get real bank/OTP traffic into the evaluation, run the MuRIL
comparison on a GPU so the ship decision is evidence-based, and buy Hinglish labels from
real messages instead of generating them."

**IF ASKED "so what's the contribution?"**
"A rigorous case study, not a new algorithm: grouped splits, a documented dataset
shortcut, three measured operating points, and a no-text-logging serving layer. The
interesting engineering is in the evaluation, and it's reproducible — 176 tests, Docker,
CI gate."

---

## 3-minute version (if you get cut short)

1. Problem: scam and legit messages share vocabulary; false alarms are the real failure.
2. Method: TF-IDF + logistic regression + 13 tactic heads; grouped, dedup'd splits;
   thresholds from validation only.
3. Finding: the dataset leaks its source (AUROC 0.996), so I report real vs synthetic
   separately.
4. Result: false alarms 15.3% → 6.4% on real ham and 13% → 5.5% on frozen probes, recall
   99.2% on 3,511 real scams.
5. Honesty: 6.4% is mostly casual chat; bank notices still fail; no transformer results;
   English-only real validation.
6. Demo: one scam, one OTP, one mode switch.

---

## Viva / interview question bank (with honest answers)

1. **"What's your accuracy?"** — "There isn't one number. 99.2% recall on real scams
   (n=3,511), 6.4% false-alarm on real ham (n=345), 5.5% on 200 frozen probes, each with
   a 95% Wilson interval. Blending them would hide the tradeoff."
2. **"Why TF-IDF?"** — "1 MB, 2 ms, explainable, and no GPU needed. With ~16k training
   rows and 63% real data, a transformer wasn't obviously better — but I never got to
   prove that, so I don't claim it."
3. **"How do you know you didn't overfit?"** — "Cluster-grouped splits made after
   deduplication, frozen probes never trained on, thresholds fitted on validation, and a
   clean-room rebuild in a temp directory that reproduced the same numbers."
4. **"What's your biggest mistake?"** — "Measuring false alarms on 59 ham rows. The rate
   moved 20 points between runs and I nearly reported noise as progress. Then I found I
   was only using 400 of 2,400 licensed ham rows."
5. **"Is it production ready?"** — "No. 6.4% false alarms on formal messages, 6
   unvalidated tags, English-only real validation, and no transformer comparison. It's
   built so a human stays in the loop."
6. **"What would you do with 2 more weeks?"** — "Real bank/OTP test traffic, then the
   MuRIL fp32 vs int8 benchmark, then Hinglish labels from real messages."

---

## Where the brownie points actually are

Ranked by how much a professor rewards them, not by how fun they are to build:

1. **Error analysis with a named failure mode.** Don't just say "6.4%". Show *which*
   messages fail: "formal bank/OTP notices are 5 of 345 ham rows and 4 are flagged." This
   is the difference between a student and a researcher. One chart, one sentence.
2. **The source-shortcut audit (AUROC 0.996).** This is your unique slide. Most projects
   never test whether their evaluation is measuring the right thing.
3. **Frozen probes.** Explain that you froze 200 legitimate messages *before* tuning, so
   you can't cheat by accident. Cheap to build, very high signal.
4. **The refused fake band.** Shows judgement, not just skill. Frame it as "I chose the
   less impressive option because the impressive one would have been dishonest."
5. **Three operating points instead of one accuracy.** Shows you understand deployment
   cost, not just model fitting.
6. **Reproducibility:** 176 tests, CI regression gate, Docker image that builds and
   serves, no-text-in-logs enforced by a test. A live `make test` is worth more than a
   slide of numbers.
7. **A file that lists what you're not allowed to claim** (`docs/CLAIMS.md`). Almost
   nobody does this; it reads as maturity.

Low-value things to *skip* in the talk: model hyperparameters, the generator's template
details, code walkthroughs. Keep them in the repo for questions.

## Slide deck outline (if you need one in 10 minutes)

1. Title
2. Two messages, same words (the problem)
3. Architecture in 3 boxes
4. Data table + the no-synthetic-in-test rule
5. AUROC 0.996 (source shortcut)
6. False-alarm before/after table
7. Three operating points
8. Suspicious band + its measured cost
9. Live demo
10. Limits + next steps

Keep every slide to ≤ 25 words plus one table or picture. Put `n` and CIs on the tables —
that's the detail that makes people trust the rest.
