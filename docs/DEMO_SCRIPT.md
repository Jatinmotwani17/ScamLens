# 60-second demo script

> Say this calmly; the numbers are the point, not the enthusiasm.

**0:00 — frame it (10 s).**
"ScamLens reads one SMS and says scam, suspicious, or safe. It's a research prototype:
the interesting part isn't that it scores high, it's that it tries to be honest about
where the score comes from."

**0:10 — scam example (15 s).**
Paste: *"Dear customer, your KYC is expiring today. Account will be blocked. Update
immediately: http://sbi-kyc-verify.xyz"* → **Analyze**.
Point at: verdict badge `scam`, the probability bar, the tactic chips
(`bank_kyc_impersonation`, `urgency_threat`, both marked *validated*), and the
highlighted words. Say: "The tag heads agree with the verdict — that's when I trust it
most."

**0:25 — legitimate OTP delivery (15 s).**
Paste: *"Your OTP for login is 482913. Never share it with anyone, including bank
staff."* → keep mode **Strict** → verdict `safe`.
Read the warning line out loud: "This is the side that used to fail — 13% of safe-chat
probes were flagged before I added 2,500 hard negatives from a disjoint generator. It's
now 5.5% on the frozen 200-message probe set and 6.4% on 345 real ham messages, with
recall still 99.2% on 3,511 real scams."
Then be honest: "Flip to **Balanced** and this same OTP alert gets flagged — its
threshold is lower. And **formal bank alerts with a call-centre number still fail more
often than conversational text**: only 5 real ham rows are explicitly bank/OTP, and 4
of them are flagged. That's the next thing to fix."

**0:40 — mode switch (10 s).**
Re-paste the KYC scam and flip to **Sensitive** → still `scam`, same evidence. Explain the
trade: "Sensitive cuts real-ham false alarms to 1.2% but misses 1.8% of scams and would
wave that OTP message through in any mode. Strict is 6.4% false alarms and misses 0.8%.
There is no single right threshold, so I ship three and name the cost of each."

**0:50 — the honest ending (10 s).**
"A classifier can tell real from synthetic text with AUROC 0.996, so every number here
is per-stratum with n and a confidence interval. It's English-only for real validation,
six tactic tags are unvalidated, and it must never auto-block anyone. Not a substitute
for checking with your bank."

---

## Five likely interview questions

**1. "Your synthetic test F1 is 0.995 — doesn't that just mean you learned your own
generator?"**
"It mostly does, which is why I don't lead with it. I report synthetic test_b separately
and put a `source_only` baseline next to every number; real-vs-synthetic is separable at
AUROC 0.996. The real claims are the ones on 3,511 real scams and 345 real ham messages,
with Wilson intervals."

**2. "How did you decide the threshold?"**
"Not by eyeballing. I fit on a weighted validation set where the four cells
(real/synthetic × scam/safe) each count equally, then produced three named points under
FPR targets of 5%, 10%, and a recall floor of 99%. Final rates are measured on held-out
real data with CIs — strict gives 6.4% ham FPR, sensitive gives 1.2% at 98.2% recall."

**3. "Why is your 'suspicious' band a rule and not a probability band?"**
"Because the TF-IDF scores are bimodal — almost nothing lands between 0.01 and 0.9, so a
band would be empty. Faking one would be theatre. Instead `suspicious` means 'clears the
threshold but I can name no supporting evidence': no tactic above 0.5, no OTP/payment
wording, no risky link. On human-ambiguous holdout rows it defers 45%, and it costs 1.1%
of clear scams that get a human check instead of a verdict."

**4. "What was your biggest bug?"**
"Two. First, the pipeline only loaded 400 of the ~4,800 licensed UCI ham rows, so the
false-alarm rate was measured on n=59 and moved 20 points between runs — noise dressed
as progress. Second, early splits were made before deduplication, so near-duplicate
templates straddled train and test. Both are now guarded: splits are cluster/family
grouped, and any synthetic row flagged `test_real` raises."

**5. "Would you ship this?"**
"As a prototype with the banner, yes. As a filter that silently drops messages, no: 6.4%
of real ham is still flagged, unvalidated tags can't be trusted, and real validation is
English-only. Shipping needs a human-in-the-loop review queue, per-language evaluation,
and a false-alarm budget agreed with whoever owns the user's experience."
