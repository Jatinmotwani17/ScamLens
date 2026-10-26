## The `suspicious` verdict (rule-based deferral)

> A probability band is impossible here (bimodal scores), so `suspicious` means *"the model says scam, but no evidence I can name supports it"*: no tactic above 0.50, no OTP/payment/credential words, no risky link type, no keyword-rule hit. Decisions use the **strict** threshold (0.040).

| set | n | scam | suspicious | safe | share deferred | note |
|---|---:|---:|---:|---:|---|---|
| external_scam (clear scams) | 3,511 | 3,443 | 40 | 28 | 1.1% |  |
| public_ham_english (clear safe) | 345 | 6 | 16 | 323 | 4.6% |  |
| test_b (synthetic) | 215 | 123 | 10 | 82 | 4.7% |  |
| suspicious holdout (human unsure) | 783 | 426 | 355 | 2 | 45.3% |  |

### What deferral costs

- **Clear scams sent to a human instead of a verdict: 40/3,511 (1.1%) of held-out real scams.** That is the price of the rule: those are messages the model was already calling scam, and it now refuses to say so.
- Clear safe messages escalated to `suspicious`: 16/345 (4.6%) - these are false alarms that become "check this", which is better than a hard `scam` but still work for a human.
- The 783-row human-`suspicious` holdout lands: 355 suspicious, 426 scam, 2 safe. Against the probability band, which put 0% of these rows in the band, this rule actually defers.

> Honest reading: the rule defers on rows where the model has no taggable reason, which is a *real* signal (the tag heads are a second opinion on the same text). It is not a calibration statement, and it does not make the probability meaningful.
